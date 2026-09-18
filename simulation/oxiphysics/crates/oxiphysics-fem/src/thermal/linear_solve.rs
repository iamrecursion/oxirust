//! Direct solver for the dense but narrow-banded systems of the 1-D thermal
//! integrators.
//!
//! [`ThermalMesh1D`](super::types::ThermalMesh1D) assembles its conductance
//! matrix densely.  The matrix is tridiagonal only while every coupling joins
//! neighbouring nodes; a thermal contact resistance between two arbitrary nodes
//! `a`, `b` adds the entries `K[a][b]` and `K[b][a]` with `|a − b| > 1`, and the
//! matrix of a user supplied 2-D or 3-D conduction problem is banded or full.
//! Cutting such a matrix down to its three diagonals, as the Thomas algorithm
//! does, silently discards those couplings.
//!
//! This module measures the actual half-bandwidths `(kl, ku)` of the non-zero
//! pattern and factorises the matrix accordingly:
//!
//! * `kl, ku ≤ 1` and weakly diagonally dominant (by rows or by columns, as every
//!   conduction matrix with positive capacitance is): Thomas' algorithm, `O(n)`,
//!   which is backward stable without pivoting for such matrices;
//! * otherwise (and whenever Thomas meets a vanishing pivot): LU with partial
//!   pivoting restricted to the band, `O(n · kl · (kl + ku))`, which degrades
//!   gracefully to a dense LU for a full matrix.
//!
//! The factorisation is separate from the solve so the constant left-hand side of
//! a time-stepping loop is factorised once.  A matrix that is singular to working
//! precision (a pivot not larger than `n · ε · max|aᵢⱼ|`) is reported as
//! [`Error::SingularSystem`] instead of producing `inf`/`NaN` temperatures.

use crate::error::{Error, Result};

/// Lower and upper half-bandwidths `(kl, ku)` of the non-zero pattern of a dense
/// square matrix: the largest `i − j` over non-zeros below the diagonal and the
/// largest `j − i` over non-zeros above it.
pub(super) fn half_bandwidths(a: &[Vec<f64>]) -> (usize, usize) {
    let mut lower = 0;
    let mut upper = 0;
    for (i, row) in a.iter().enumerate() {
        for (j, &value) in row.iter().enumerate() {
            if value != 0.0 {
                if i > j {
                    lower = lower.max(i - j);
                } else {
                    upper = upper.max(j - i);
                }
            }
        }
    }
    (lower, upper)
}

/// Whether a tridiagonal matrix is weakly diagonally dominant by rows or by
/// columns, the condition under which elimination without pivoting is backward
/// stable (the caller has verified that `a` has no entry beyond the first
/// off-diagonals).
fn is_diagonally_dominant(a: &[Vec<f64>]) -> bool {
    let n = a.len();
    let by_rows = (0..n).all(|i| {
        let left = if i > 0 { a[i][i - 1].abs() } else { 0.0 };
        let right = if i + 1 < n { a[i][i + 1].abs() } else { 0.0 };
        a[i][i].abs() >= left + right
    });
    let by_columns = (0..n).all(|j| {
        let above = if j > 0 { a[j - 1][j].abs() } else { 0.0 };
        let below = if j + 1 < n { a[j + 1][j].abs() } else { 0.0 };
        a[j][j].abs() >= above + below
    });
    by_rows || by_columns
}

/// Replace row `node` of `a` by the unit row for every Dirichlet condition and
/// store the prescribed temperature in `rhs`.
///
/// The columns of the constrained unknowns are kept: the constrained row simply
/// states `T[node] = value`, so the remaining rows still see the prescribed
/// temperature through their own `K[i][node]` entries.
///
/// # Errors
///
/// [`Error::DofOutOfRange`] if a constrained node does not exist and
/// [`Error::InvalidInput`] if `a` is not `n × n` for `n = rhs.len()` or a
/// prescribed temperature is not finite.
pub(super) fn impose_dirichlet(
    a: &mut [Vec<f64>],
    rhs: &mut [f64],
    conditions: &[(usize, f64)],
) -> Result<()> {
    let n = rhs.len();
    if a.len() != n || a.iter().any(|row| row.len() != n) {
        return Err(Error::InvalidInput(format!(
            "system matrix must be {n} x {n} to match the right-hand side"
        )));
    }
    for &(node, value) in conditions {
        if node >= n {
            return Err(Error::DofOutOfRange { dof: node, n });
        }
        if !value.is_finite() {
            return Err(Error::InvalidInput(format!(
                "prescribed temperature at node {node} is not finite"
            )));
        }
        a[node].iter_mut().for_each(|entry| *entry = 0.0);
        a[node][node] = 1.0;
        rhs[node] = value;
    }
    Ok(())
}

/// Factorisation of a square matrix with a narrow band (see the module docs).
#[derive(Debug, Clone)]
pub(super) enum BandedSystem {
    /// Thomas elimination of a tridiagonal matrix.
    Tridiagonal {
        /// Elimination multipliers `m[i] = a[i][i-1] / pivot[i-1]` (`m[0] = 0`).
        multipliers: Vec<f64>,
        /// Eliminated diagonal.
        pivots: Vec<f64>,
        /// Super-diagonal `a[i][i+1]` (last entry unused).
        upper: Vec<f64>,
    },
    /// LU with partial pivoting restricted to the band.
    Banded {
        /// Rows of `U` (entries right of the diagonal up to `kl + ku` columns).
        u_rows: Vec<Vec<f64>>,
        /// Multipliers of every elimination step, for rows `k+1 ..= k+kl`.
        multipliers: Vec<Vec<f64>>,
        /// Row exchanged with row `k` at step `k`.
        swaps: Vec<usize>,
        /// Lower half-bandwidth.
        kl: usize,
        /// Upper half-bandwidth.
        ku: usize,
    },
}

impl BandedSystem {
    /// Factorise `a` (square, any pattern; the band is measured from the
    /// non-zeros).
    ///
    /// # Errors
    ///
    /// [`Error::InvalidInput`] if `a` is not square or has a non-finite entry,
    /// [`Error::SingularSystem`] if it is singular to working precision.
    pub(super) fn new(a: Vec<Vec<f64>>) -> Result<Self> {
        let n = a.len();
        if a.iter().any(|row| row.len() != n) {
            return Err(Error::InvalidInput(
                "system matrix must be square".to_string(),
            ));
        }
        let mut largest = 0.0_f64;
        for row in &a {
            for &value in row {
                if !value.is_finite() {
                    return Err(Error::InvalidInput(
                        "system matrix contains a non-finite entry".to_string(),
                    ));
                }
                largest = largest.max(value.abs());
            }
        }
        let tolerance = n as f64 * f64::EPSILON * largest;
        let (kl, ku) = half_bandwidths(&a);
        if kl <= 1
            && ku <= 1
            && is_diagonally_dominant(&a)
            && let Some(thomas) = Self::thomas(&a, tolerance)
        {
            return Ok(thomas);
        }
        Self::banded_lu(a, kl, ku, tolerance)
    }

    /// Thomas elimination; `None` when a pivot is not larger than `tolerance`
    /// (the caller then falls back to pivoted elimination).
    fn thomas(a: &[Vec<f64>], tolerance: f64) -> Option<Self> {
        let n = a.len();
        let mut pivots: Vec<f64> = (0..n).map(|i| a[i][i]).collect();
        let upper: Vec<f64> = (0..n)
            .map(|i| if i + 1 < n { a[i][i + 1] } else { 0.0 })
            .collect();
        let mut multipliers = vec![0.0_f64; n];
        for i in 0..n {
            if i > 0 {
                let multiplier = a[i][i - 1] / pivots[i - 1];
                multipliers[i] = multiplier;
                pivots[i] -= multiplier * upper[i - 1];
            }
            if pivots[i].is_nan() || pivots[i].abs() <= tolerance {
                return None;
            }
        }
        Some(Self::Tridiagonal {
            multipliers,
            pivots,
            upper,
        })
    }

    /// Band-restricted LU with partial pivoting.
    fn banded_lu(mut a: Vec<Vec<f64>>, kl: usize, ku: usize, tolerance: f64) -> Result<Self> {
        let n = a.len();
        let mut multipliers = Vec::with_capacity(n);
        let mut swaps = Vec::with_capacity(n);
        for k in 0..n {
            let last_row = (k + kl).min(n - 1);
            let last_col = (k + kl + ku).min(n - 1);
            let mut pivot_row = k;
            let mut best = a[k][k].abs();
            for (r, row) in a.iter().enumerate().take(last_row + 1).skip(k + 1) {
                let candidate = row[k].abs();
                if candidate > best {
                    best = candidate;
                    pivot_row = r;
                }
            }
            if best.is_nan() || best <= tolerance {
                return Err(Error::SingularSystem);
            }
            a.swap(k, pivot_row);
            swaps.push(pivot_row);
            // Rows 0..=k are final; the rows below receive the elimination.
            let (done, rest) = a.split_at_mut(k + 1);
            let pivot_values = &done[k];
            let pivot = pivot_values[k];
            let mut step = Vec::with_capacity(last_row - k);
            for target in rest.iter_mut().take(last_row - k) {
                let factor = target[k] / pivot;
                step.push(factor);
                if factor != 0.0 {
                    for j in (k + 1)..=last_col {
                        target[j] -= factor * pivot_values[j];
                    }
                }
                target[k] = 0.0;
            }
            multipliers.push(step);
        }
        Ok(Self::Banded {
            u_rows: a,
            multipliers,
            swaps,
            kl,
            ku,
        })
    }

    /// Solve `A x = rhs` with the stored factors.
    ///
    /// # Errors
    ///
    /// [`Error::InvalidInput`] if `rhs` has the wrong length and
    /// [`Error::SingularSystem`] if the solution is not finite (overflow).
    pub(super) fn solve(&self, rhs: &[f64]) -> Result<Vec<f64>> {
        let n = match self {
            Self::Tridiagonal { pivots, .. } => pivots.len(),
            Self::Banded { u_rows, .. } => u_rows.len(),
        };
        if rhs.len() != n {
            return Err(Error::InvalidInput(format!(
                "right-hand side has {} entries for a system of order {n}",
                rhs.len()
            )));
        }
        let mut x = rhs.to_vec();
        match self {
            Self::Tridiagonal {
                multipliers,
                pivots,
                upper,
            } => {
                for i in 1..n {
                    x[i] -= multipliers[i] * x[i - 1];
                }
                for i in (0..n).rev() {
                    let tail = if i + 1 < n { upper[i] * x[i + 1] } else { 0.0 };
                    x[i] = (x[i] - tail) / pivots[i];
                }
            }
            Self::Banded {
                u_rows,
                multipliers,
                swaps,
                kl,
                ku,
            } => {
                for k in 0..n {
                    x.swap(k, swaps[k]);
                    let pivot_value = x[k];
                    for (offset, &factor) in multipliers[k].iter().enumerate() {
                        x[k + 1 + offset] -= factor * pivot_value;
                    }
                }
                for i in (0..n).rev() {
                    let last_col = (i + kl + ku).min(n - 1);
                    let mut value = x[i];
                    for j in (i + 1)..=last_col {
                        value -= u_rows[i][j] * x[j];
                    }
                    x[i] = value / u_rows[i][i];
                }
            }
        }
        if x.iter().any(|v| !v.is_finite()) {
            return Err(Error::SingularSystem);
        }
        Ok(x)
    }
}
