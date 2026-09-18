// Copyright 2026 COOLJAPAN OU (Team KitaSan)
// SPDX-License-Identifier: Apache-2.0

//! Data-driven reduced-order model of a distribution-function field by exact
//! dynamic mode decomposition (DMD).
//!
//! Given snapshots `x_0, ..., x_{n-1}` of a flattened field, DMD finds the best-fit
//! linear operator `A` with `x_{k+1} ≈ A x_k`, i.e. `Y ≈ A X` for the snapshot
//! matrices `X = [x_0 .. x_{n-2}]` and `Y = [x_1 .. x_{n-1}]`, and returns its
//! eigenvalues `lambda_j` and eigenvectors (the DMD modes `phi_j`).  The field is
//! then advanced analytically,
//!
//! ```text
//! x_k ≈ Re( sum_j phi_j lambda_j^k b_j ) ,
//! ```
//!
//! where the amplitudes `b_j` fit the first snapshot.  The decomposition follows
//! the *exact DMD* algorithm of Tu et al. in the method-of-snapshots form, which
//! only ever touches the small `(n-1) x (n-1)` Gram matrices of the snapshots and
//! is therefore cheap for large fields:
//!
//! 1. `X^T X = V Σ² V^T` (symmetric eigendecomposition), rank truncation;
//! 2. `Ã = Σ⁻¹ V^T (X^T Y) V Σ⁻¹ = U^T A U` with `U = X V Σ⁻¹`;
//! 3. `Ã W = W Λ` (complex eigendecomposition of the non-symmetric `Ã`);
//! 4. exact modes `Φ = Y V Σ⁻¹ W`;
//! 5. amplitudes `b = argmin ‖Φ b − x_0‖` (complex least squares).

use super::complex_eigen::{ComplexMatrix, eigen_decomposition, least_squares};
use crate::error::{Error, Result};
use oxiphysics_core::complex::Complex;
use oxiphysics_core::numerical_methods::symmetric_eigen_n;

/// Columns of `Φ` whose component orthogonal to the preceding modes is below this
/// fraction of their length are treated as linearly dependent in the amplitude fit.
const MODE_DEPENDENCE_TOLERANCE: f64 = 1.0e-10;

/// Relative eigenvalue tolerance of the Gram matrix `X^T X`: POD directions with
/// `σ_i² <= tolerance σ_0²` are discarded.
///
/// The eigenvalues of `X^T X` carry an absolute rounding error of a few
/// `ε σ_0²` (growing with the number of snapshots and the field size), so
/// directions with `σ_i / σ_0` below roughly `1e-6` cannot be told from noise and
/// would be amplified by `Σ⁻¹` into spurious modes.
fn gram_rank_tolerance(snapshots: usize, field_size: usize) -> f64 {
    let size = snapshots as f64 + (field_size as f64).sqrt();
    (16.0 * size * f64::EPSILON).max(1.0e-12)
}

/// Result of an exact DMD.
#[derive(Debug, Clone)]
struct ExactDmd {
    singular_values: Vec<f64>,
    eigenvalues: Vec<Complex>,
    modes: Vec<Vec<Complex>>,
    amplitudes: Vec<Complex>,
}

/// Dot product of two real vectors.
fn dot(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b.iter()).map(|(&x, &y)| x * y).sum()
}

/// Validates the snapshot set and returns the field size.
fn validate_snapshots(snapshots: &[Vec<f64>]) -> Result<usize> {
    if snapshots.len() < 2 {
        return Err(Error::IncompatibleParameters {
            message: format!(
                "dynamic mode decomposition needs at least two snapshots, got {}",
                snapshots.len()
            ),
        });
    }
    let size = snapshots[0].len();
    if size == 0 {
        return Err(Error::IncompatibleParameters {
            message: "dynamic mode decomposition needs non-empty snapshots".to_string(),
        });
    }
    for (k, snapshot) in snapshots.iter().enumerate() {
        if snapshot.len() != size {
            return Err(Error::ComponentCountMismatch {
                expected: size,
                found: snapshot.len(),
                field: format!("snapshot {k}"),
            });
        }
        if let Some((index, &value)) = snapshot.iter().enumerate().find(|(_, v)| !v.is_finite()) {
            return Err(Error::NonFiniteField {
                field: format!("snapshot {k}"),
                index,
                value,
            });
        }
    }
    Ok(size)
}

/// Exact DMD of the snapshot sequence, optionally truncated to `max_rank` modes.
fn exact_dmd(snapshots: &[Vec<f64>], max_rank: Option<usize>) -> Result<ExactDmd> {
    let size = validate_snapshots(snapshots)?;
    let n = snapshots.len();
    let m = n - 1;

    // Gram matrix S_ij = x_i . x_j of all snapshots; X^T X, X^T Y are sub-blocks.
    let mut gram = vec![vec![0.0_f64; n]; n];
    for i in 0..n {
        for j in i..n {
            let g = dot(&snapshots[i], &snapshots[j]);
            if !g.is_finite() {
                // Finite snapshots whose inner products overflow.
                return Err(Error::NonFiniteField {
                    field: "snapshot Gram matrix".to_string(),
                    index: i * n + j,
                    value: g,
                });
            }
            gram[i][j] = g;
            gram[j][i] = g;
        }
    }
    let xtx: Vec<Vec<f64>> = gram.iter().take(m).map(|row| row[..m].to_vec()).collect();
    let (values, vectors) = symmetric_eigen_n(&xtx);

    // Retained POD directions, largest singular value first.
    let largest = values.last().copied().unwrap_or(0.0);
    if !largest.is_finite() {
        return Err(Error::NonFiniteField {
            field: "snapshot Gram eigenvalue".to_string(),
            index: m - 1,
            value: largest,
        });
    }
    if largest <= 0.0 {
        return Ok(ExactDmd {
            singular_values: Vec::new(),
            eigenvalues: Vec::new(),
            modes: Vec::new(),
            amplitudes: Vec::new(),
        });
    }
    let threshold = gram_rank_tolerance(n, size) * largest;
    let mut retained: Vec<usize> = (0..values.len())
        .rev()
        .filter(|&i| values[i] > threshold)
        .collect();
    if let Some(cap) = max_rank {
        retained.truncate(cap);
    }
    let rank = retained.len();

    // B = V Σ⁻¹ (one column per retained direction, each of length m).
    let singular_values: Vec<f64> = retained.iter().map(|&i| values[i].sqrt()).collect();
    let basis: Vec<Vec<f64>> = retained
        .iter()
        .zip(singular_values.iter())
        .map(|(&i, &sigma)| vectors[i].iter().map(|&v| v / sigma).collect())
        .collect();

    // Ã = B^T (X^T Y) B with (X^T Y)_ij = x_i . x_{j+1}.
    let mut reduced = vec![0.0_f64; rank * rank];
    for (a, left) in basis.iter().enumerate() {
        let mut projected = vec![0.0_f64; m];
        for (i, &li) in left.iter().enumerate() {
            for (j, slot) in projected.iter_mut().enumerate() {
                *slot += li * gram[i][j + 1];
            }
        }
        for (b, right) in basis.iter().enumerate() {
            reduced[a * rank + b] = dot(&projected, right);
        }
    }
    let eigen = eigen_decomposition(&ComplexMatrix::from_real_row_major(rank, &reduced)?)?;

    // P = Y B (field-sized columns), then the exact modes Φ = P W.
    let projected_fields: Vec<Vec<f64>> = basis
        .iter()
        .map(|column| {
            let mut field = vec![0.0_f64; size];
            for (i, &weight) in column.iter().enumerate() {
                for (slot, &y) in field.iter_mut().zip(snapshots[i + 1].iter()) {
                    *slot += weight * y;
                }
            }
            field
        })
        .collect();
    let mut modes: Vec<Vec<Complex>> = Vec::with_capacity(rank);
    for w in &eigen.vectors {
        let mut mode = vec![Complex::zero(); size];
        for (field, &coefficient) in projected_fields.iter().zip(w.iter()) {
            for (slot, &value) in mode.iter_mut().zip(field.iter()) {
                *slot = *slot + Complex::new(coefficient.re * value, coefficient.im * value);
            }
        }
        let norm = mode.iter().map(|z| z.norm_sq()).sum::<f64>().sqrt();
        if norm > 0.0 && norm.is_finite() {
            for z in mode.iter_mut() {
                *z = Complex::new(z.re / norm, z.im / norm);
            }
        }
        modes.push(mode);
    }

    let first: Vec<Complex> = snapshots[0].iter().map(|&x| Complex::new(x, 0.0)).collect();
    let amplitudes = least_squares(&modes, &first, MODE_DEPENDENCE_TOLERANCE);
    Ok(ExactDmd {
        singular_values,
        eigenvalues: eigen.values,
        modes,
        amplitudes,
    })
}

/// `lambda^k` for a complex `lambda` and an integer `k`.
fn complex_power(lambda: Complex, k: usize) -> Complex {
    let exponent = k as f64;
    Complex::from_polar(lambda.norm().powf(exponent), lambda.arg() * exponent)
}

/// Data-driven LBM using Dynamic Mode Decomposition (DMD) for reduced-order
/// modelling.
///
/// Snapshots `x_0, ..., x_{n-1}` of the distribution-function field are stored
/// and used to extract the dynamic modes of the best-fit linear one-step
/// propagator `A` (`x_{k+1} ≈ A x_k`) by an exact DMD in the method-of-snapshots
/// form: the POD of the snapshot matrix from the eigendecomposition of its Gram
/// matrix, the reduced operator `Ã = U^T A U` and its complex eigenpairs
/// `Ã W = W Λ`, the exact modes `Φ = Y V Σ⁻¹ W` and the amplitudes
/// `b = argmin ‖Φ b − x_0‖`.  The resulting ROM predicts the state at any time
/// step, past or future, `x̂_k = Re(Φ Λ^k b)`, without running the full LBM.
#[derive(Debug, Clone)]
pub struct DataDrivenLbm {
    /// Collection of solution snapshots; each snapshot is a flattened
    /// distribution-function field.
    pub snapshots: Vec<Vec<f64>>,
    /// Exact DMD modes `phi_j`, normalised to unit Euclidean length; one complex
    /// vector of the field size per eigenvalue.  Modes of real data come in
    /// complex-conjugate pairs.
    pub modes: Vec<Vec<Complex>>,
    /// DMD eigenvalues `lambda_j` of the one-step propagator, ordered by
    /// decreasing modulus.  `|lambda| < 1` decays, `|lambda| > 1` grows and
    /// `arg(lambda)` is the phase advance per step.
    pub eigenvalues: Vec<Complex>,
    /// Mode amplitudes `b_j` fitted to the first snapshot.
    pub amplitudes: Vec<Complex>,
    /// Retained POD singular values `σ_j` of the snapshot matrix `X`.
    pub singular_values: Vec<f64>,
    /// Number of snapshots used by the last successful decomposition (0 before
    /// the first one).
    fit_count: usize,
}

impl DataDrivenLbm {
    /// Create an empty `DataDrivenLbm`.
    pub fn new() -> Self {
        Self {
            snapshots: Vec::new(),
            modes: Vec::new(),
            eigenvalues: Vec::new(),
            amplitudes: Vec::new(),
            singular_values: Vec::new(),
            fit_count: 0,
        }
    }

    /// Append a new snapshot to the data set.
    pub fn add_snapshot(&mut self, snapshot: Vec<f64>) {
        self.snapshots.push(snapshot);
    }

    /// Compute the exact DMD of the stored snapshots.
    ///
    /// After a successful call `self.modes`, `self.eigenvalues`,
    /// `self.amplitudes` and `self.singular_values` hold the decomposition.  All
    /// POD directions of the snapshot matrix that rise above the numerical noise
    /// of its Gram matrix (singular values below about `1e-6` of the largest one
    /// are discarded) are retained; use
    /// [`dmd_modes_with_rank`](Self::dmd_modes_with_rank) to truncate further, as
    /// is usual for noisy data.
    ///
    /// Data without any signal (all-zero snapshots) yields an empty decomposition.
    ///
    /// # Errors
    /// Fewer than two snapshots, snapshots of different or zero length, non-finite
    /// entries, or a failed eigenvalue iteration.  The previous decomposition is
    /// left untouched on error.
    pub fn dmd_modes(&mut self) -> Result<()> {
        self.fit(None)
    }

    /// Like [`dmd_modes`](Self::dmd_modes), keeping at most the `max_rank`
    /// leading POD directions (and hence at most `max_rank` modes).
    ///
    /// # Errors
    /// As [`dmd_modes`](Self::dmd_modes), and
    /// [`Error::ParameterOutOfBounds`] for `max_rank == 0`.
    pub fn dmd_modes_with_rank(&mut self, max_rank: usize) -> Result<()> {
        if max_rank == 0 {
            return Err(Error::ParameterOutOfBounds {
                name: "max_rank".to_string(),
                value: 0.0,
                min: 1.0,
                max: f64::MAX,
            });
        }
        self.fit(Some(max_rank))
    }

    fn fit(&mut self, max_rank: Option<usize>) -> Result<()> {
        let dmd = exact_dmd(&self.snapshots, max_rank)?;
        self.modes = dmd.modes;
        self.eigenvalues = dmd.eigenvalues;
        self.amplitudes = dmd.amplitudes;
        self.singular_values = dmd.singular_values;
        self.fit_count = self.snapshots.len();
        Ok(())
    }

    /// Predict the state `steps_ahead` time steps after the first snapshot.
    ///
    /// The ROM state is `x̂_k = Re( sum_j phi_j lambda_j^k b_j )` with
    /// `k = steps_ahead`: `rom_predict(0)` reconstructs the first snapshot,
    /// `rom_predict(k)` the `k`-th one for `k` below the snapshot count, and larger
    /// `k` extrapolate the learned dynamics beyond the data window.
    ///
    /// Returns an empty vector until [`dmd_modes`](Self::dmd_modes) has succeeded,
    /// and the zero field for data without signal.
    pub fn rom_predict(&self, steps_ahead: usize) -> Vec<f64> {
        if self.fit_count == 0 {
            return Vec::new();
        }
        let size = self.snapshots.first().map_or(0, Vec::len);
        let mut prediction = vec![0.0_f64; size];
        for ((mode, &lambda), &amplitude) in self
            .modes
            .iter()
            .zip(self.eigenvalues.iter())
            .zip(self.amplitudes.iter())
        {
            let coefficient = amplitude * complex_power(lambda, steps_ahead);
            for (slot, &component) in prediction.iter_mut().zip(mode.iter()) {
                *slot += coefficient.re * component.re - coefficient.im * component.im;
            }
        }
        prediction
    }

    /// Return the number of snapshots currently stored.
    pub fn snapshot_count(&self) -> usize {
        self.snapshots.len()
    }

    /// Euclidean reconstruction error `‖x_k − x̂_k‖₂` of snapshot `k` by the ROM
    /// (`x̂_k` as in [`rom_predict`](Self::rom_predict)).
    ///
    /// Returns `NaN` if `k` is not a stored snapshot or no decomposition exists.
    pub fn reconstruction_error(&self, k: usize) -> f64 {
        if k >= self.snapshots.len() || self.fit_count == 0 {
            return f64::NAN;
        }
        let prediction = self.rom_predict(k);
        self.snapshots[k]
            .iter()
            .zip(prediction.iter())
            .map(|(&s, &p)| (s - p).powi(2))
            .sum::<f64>()
            .sqrt()
    }
}

impl Default for DataDrivenLbm {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests;
