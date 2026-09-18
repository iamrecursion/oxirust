//! Chain maps between cellular chain complexes and the degree of a cellular map
//! on top homology.
//!
//! A cellular map `f : X → Y` induces the chain map `f_k : C_k(X) → C_k(Y)`
//! (integer matrices, see [`CellularApproximation::chain_map`]) with
//! `f_{k-1} ∂_k = ∂_k f_k`.  The degree of `f` is read off the action of `f_n` on
//! the fundamental cycle, which is found by exact integer elimination.

use super::{CellularApproximation, CwComplex};
use crate::error::{Error, Result};

/// Operation name reported in errors raised while computing a degree.
const MAP_DEGREE: &str = "CellularApproximation::map_degree";

impl CellularApproximation {
    /// Create a cellular approximation from a source, a target and the chain
    /// map `f_k : C_k(source) → C_k(target)` of a cellular map between them.
    ///
    /// The shape of every matrix and the chain-map identity
    /// `f_{k-1} ∂_k = ∂_k f_k` are verified (see
    /// [`CellularApproximation::validate_chain_map`]).
    ///
    /// # Errors
    ///
    /// Returns an error when `chain_map` is not a chain map from `source` to
    /// `target`.
    pub fn with_chain_map(
        source: CwComplex,
        target: CwComplex,
        chain_map: Vec<Vec<Vec<i64>>>,
    ) -> Result<Self> {
        let approximation = Self {
            source,
            target,
            chain_map,
        };
        approximation.validate_chain_map()?;
        Ok(approximation)
    }

    /// The identity map of a complex to itself, as a chain map.
    pub fn identity(complex: CwComplex) -> Self {
        let chain_map = (0..=complex.max_dim)
            .map(|k| {
                let cells = complex.count(k);
                (0..cells)
                    .map(|i| (0..cells).map(|j| i64::from(i == j)).collect())
                    .collect()
            })
            .collect();
        Self {
            source: complex.clone(),
            target: complex,
            chain_map,
        }
    }

    /// Check that [`CellularApproximation::chain_map`] is a chain map from the
    /// source to the target complex.
    ///
    /// There must be one matrix `f_k` for each dimension `0..=source.max_dim`
    /// with one row per `k`-cell of the target and one column per `k`-cell of
    /// the source, and `f_{k-1} ∂_k = ∂_k f_k` must hold for every `k >= 1`
    /// (`∂` being the boundary matrix of the respective complex).  The maps need
    /// not come from a continuous map: any chain map has an algebraic degree.
    ///
    /// # Errors
    ///
    /// Returns an error when no chain map is attached, when a matrix has the
    /// wrong shape, when the chain-map identity fails, or when the exact
    /// integer arithmetic overflows.
    pub fn validate_chain_map(&self) -> Result<()> {
        if self.chain_map.is_empty() {
            return Err(Error::unsupported(
                "CellularApproximation::validate_chain_map",
                "no chain map is attached to this approximation",
            ));
        }
        let top = self.source.max_dim;
        if self.chain_map.len() != top + 1 {
            return Err(Error::dimension_mismatch(
                "chain map degrees",
                self.chain_map.len(),
                "source dimensions 0..=max_dim",
                top + 1,
            ));
        }
        for (k, matrix) in self.chain_map.iter().enumerate() {
            let rows = self.target.count(k);
            let cols = self.source.count(k);
            if matrix.len() != rows {
                return Err(Error::dimension_mismatch(
                    "chain map rows",
                    matrix.len(),
                    "target cells",
                    rows,
                ));
            }
            if let Some(row) = matrix.iter().find(|row| row.len() != cols) {
                return Err(Error::dimension_mismatch(
                    "chain map columns",
                    row.len(),
                    "source cells",
                    cols,
                ));
            }
        }
        for k in 1..=top {
            let columns = self.source.count(k);
            let through_source = multiply(
                &widen(&self.chain_map[k - 1]),
                &widen(&self.source.boundary_matrix(k)),
                columns,
            )?;
            let through_target = multiply(
                &widen(&self.target.boundary_matrix(k)),
                &widen(&self.chain_map[k]),
                columns,
            )?;
            if through_source != through_target {
                return Err(Error::general(format!(
                    "the attached maps are not a chain map: f_{} ∘ ∂_{k} differs from ∂_{k} ∘ f_{k}",
                    k - 1
                )));
            }
        }
        Ok(())
    }

    /// Degree of the cellular map on top-dimensional homology.
    ///
    /// Both complexes must have the same top dimension `n` and a top homology
    /// `H_n ≅ Z` of rank one (a sphere, or any closed orientable `n`-manifold).
    /// With no `(n+1)`-cells `H_n` is the kernel of `∂_n`; its generator, the
    /// fundamental cycle `z`, is the primitive integer `n`-chain in that kernel
    /// whose first non-zero coefficient (cells ordered by id) is positive.  The
    /// chain map takes the fundamental cycle `z_X` of the source to a cycle of
    /// the target, hence to `d · z_Y`; the integer `d` is the degree.
    ///
    /// The sign of the degree is relative to this orientation convention for the
    /// two fundamental classes.  For a complex mapped to itself both classes
    /// agree, so the identity has degree `1`, a reflection `-1` and the
    /// antipodal map of `S^n` has degree `(-1)^(n+1)`.
    ///
    /// # Errors
    ///
    /// Returns an error when no valid chain map is attached
    /// ([`CellularApproximation::validate_chain_map`]), when the top dimensions
    /// of the complexes differ, when the top homology of either complex does not
    /// have rank one (a ball, a wedge of spheres, a non-orientable surface, ...),
    /// or when the exact integer arithmetic overflows (very large complexes).
    pub fn map_degree(&self) -> Result<i64> {
        let dimension = self.source.max_dim;
        if self.target.max_dim != dimension {
            return Err(Error::dimension_mismatch(
                "source dimension",
                dimension,
                "target dimension",
                self.target.max_dim,
            ));
        }
        self.validate_chain_map()?;
        let source_cycle = fundamental_cycle(&self.source, dimension, "source")?;
        let target_cycle = fundamental_cycle(&self.target, dimension, "target")?;
        let image = apply(&self.chain_map[dimension], &source_cycle)?;
        degree_of_image(&image, &target_cycle)
    }
}

/// Widen the entries of an integer matrix to `i128` for exact arithmetic.
fn widen<T: Copy + Into<i128>>(matrix: &[Vec<T>]) -> Vec<Vec<i128>> {
    matrix
        .iter()
        .map(|row| row.iter().map(|&entry| entry.into()).collect())
        .collect()
}

/// Error for an integer that outgrew the 128-bit exact arithmetic.
fn overflow_error() -> Error {
    Error::unsupported(
        MAP_DEGREE,
        "integer overflow in exact arithmetic; the complex is too large for 128-bit integers",
    )
}

/// Product of an `r x m` and an `m x cols` integer matrix with overflow checks.
///
/// `cols` is passed explicitly because `b` may have no rows (`m == 0`).
fn multiply(a: &[Vec<i128>], b: &[Vec<i128>], cols: usize) -> Result<Vec<Vec<i128>>> {
    a.iter()
        .map(|a_row| {
            (0..cols)
                .map(|j| {
                    a_row
                        .iter()
                        .zip(b.iter())
                        .try_fold(0i128, |sum, (&a_entry, b_row)| {
                            let b_entry = b_row.get(j).copied().ok_or_else(|| {
                                Error::dimension_mismatch(
                                    "matrix columns",
                                    b_row.len(),
                                    "cells",
                                    cols,
                                )
                            })?;
                            let term = a_entry.checked_mul(b_entry).ok_or_else(overflow_error)?;
                            sum.checked_add(term).ok_or_else(overflow_error)
                        })
                })
                .collect()
        })
        .collect()
}

/// The matrix `matrix` applied to the vector `vector`, with overflow checks.
fn apply(matrix: &[Vec<i64>], vector: &[i64]) -> Result<Vec<i128>> {
    matrix
        .iter()
        .map(|row| {
            row.iter()
                .zip(vector.iter())
                .try_fold(0i128, |sum, (&m, &v)| {
                    let term = i128::from(m)
                        .checked_mul(i128::from(v))
                        .ok_or_else(overflow_error)?;
                    sum.checked_add(term).ok_or_else(overflow_error)
                })
        })
        .collect()
}

/// Greatest common divisor (`gcd(0, 0) = 0`).
fn gcd(mut a: u128, mut b: u128) -> u128 {
    while b != 0 {
        let r = a % b;
        a = b;
        b = r;
    }
    a
}

/// `gcd` of two signed values as a positive `i128`.
fn gcd_i128(a: i128, b: i128) -> Result<i128> {
    i128::try_from(gcd(a.unsigned_abs(), b.unsigned_abs())).map_err(|_| overflow_error())
}

/// Divide a row by the gcd of its entries (a zero row is left alone).
fn divide_by_content(row: &mut [i128]) {
    let content = row.iter().fold(0u128, |g, &x| gcd(g, x.unsigned_abs()));
    if content > 1
        && let Ok(content) = i128::try_from(content)
    {
        for entry in row.iter_mut() {
            *entry /= content;
        }
    }
}

/// Fully reduced row echelon form of an integer matrix: every pivot column has a
/// single non-zero entry, in its pivot row.
struct Echelon {
    /// Reduced rows; the first `pivot_cols.len()` rows are the pivot rows.
    rows: Vec<Vec<i128>>,
    /// Pivot column of each pivot row.
    pivot_cols: Vec<usize>,
}

/// Reduce an integer matrix with `ncols` columns by fraction-free elimination.
///
/// Every row stays an integer vector: a row is combined with the pivot row by
/// cross-multiplication (using the pivot/entry gcd to keep the multipliers
/// small) and then divided by the gcd of its entries.
fn echelon_reduce(matrix: &[Vec<i32>], ncols: usize) -> Result<Echelon> {
    let mut rows: Vec<Vec<i128>> = Vec::with_capacity(matrix.len());
    for row in matrix {
        if row.len() != ncols {
            return Err(Error::dimension_mismatch(
                "boundary matrix row",
                row.len(),
                "cells",
                ncols,
            ));
        }
        rows.push(row.iter().map(|&entry| i128::from(entry)).collect());
    }
    let mut pivot_cols: Vec<usize> = Vec::new();
    for col in 0..ncols {
        let rank = pivot_cols.len();
        if rank == rows.len() {
            break;
        }
        let Some(offset) = rows[rank..].iter().position(|row| row[col] != 0) else {
            continue;
        };
        rows.swap(rank, rank + offset);
        let pivot_row = rows[rank].clone();
        let pivot = pivot_row[col];
        for (i, row) in rows.iter_mut().enumerate() {
            let lead = row[col];
            if i == rank || lead == 0 {
                continue;
            }
            let g = gcd_i128(pivot, lead)?;
            let (row_scale, pivot_scale) = (pivot / g, lead / g);
            for (entry, &pivot_entry) in row.iter_mut().zip(pivot_row.iter()) {
                let kept = entry.checked_mul(row_scale).ok_or_else(overflow_error)?;
                let removed = pivot_entry
                    .checked_mul(pivot_scale)
                    .ok_or_else(overflow_error)?;
                *entry = kept.checked_sub(removed).ok_or_else(overflow_error)?;
            }
            divide_by_content(row);
        }
        pivot_cols.push(col);
    }
    Ok(Echelon { rows, pivot_cols })
}

/// The primitive integer generator of the kernel of the reduced matrix, whose
/// first non-zero entry is positive.  The kernel must have dimension one.
fn kernel_generator(echelon: &Echelon, ncols: usize) -> Result<Vec<i64>> {
    let mut is_pivot = vec![false; ncols];
    for &col in &echelon.pivot_cols {
        is_pivot[col] = true;
    }
    let mut free_cols = (0..ncols).filter(|&col| !is_pivot[col]);
    let (Some(free_col), None) = (free_cols.next(), free_cols.next()) else {
        return Err(Error::general(
            "the kernel of the boundary matrix does not have dimension one",
        ));
    };
    // The pivot row of pivot column `c` reads `pivot x_c + free x_free = 0`, so
    // `x_c = -free x_free / pivot`; choose `x_free` as the least common multiple
    // of the denominators `|pivot| / gcd(pivot, free)` to keep every `x_c` integral.
    let mut scale: i128 = 1;
    for (k, &col) in echelon.pivot_cols.iter().enumerate() {
        let pivot = echelon.rows[k][col];
        let free = echelon.rows[k][free_col];
        let denominator =
            i128::try_from(pivot.unsigned_abs() / gcd(pivot.unsigned_abs(), free.unsigned_abs()))
                .map_err(|_| overflow_error())?;
        let common = gcd_i128(scale, denominator)?;
        scale = (scale / common)
            .checked_mul(denominator)
            .ok_or_else(overflow_error)?;
    }
    let mut cycle = vec![0i128; ncols];
    cycle[free_col] = scale;
    for (k, &col) in echelon.pivot_cols.iter().enumerate() {
        let pivot = echelon.rows[k][col];
        let free = echelon.rows[k][free_col];
        let numerator = free
            .checked_mul(scale)
            .and_then(i128::checked_neg)
            .ok_or_else(overflow_error)?;
        cycle[col] = numerator.checked_div(pivot).ok_or_else(overflow_error)?;
    }
    // Primitive (the entries have no common factor) with a positive first entry.
    let content = cycle.iter().fold(0u128, |g, &x| gcd(g, x.unsigned_abs()));
    let content = i128::try_from(content).map_err(|_| overflow_error())?;
    for entry in cycle.iter_mut() {
        *entry /= content;
    }
    if cycle
        .iter()
        .find(|&&entry| entry != 0)
        .is_some_and(|&entry| entry < 0)
    {
        for entry in cycle.iter_mut() {
            *entry = entry.checked_neg().ok_or_else(overflow_error)?;
        }
    }
    cycle
        .iter()
        .map(|&entry| i64::try_from(entry).map_err(|_| overflow_error()))
        .collect()
}

/// `true` when `∂ cycle = 0` for the boundary matrix `boundary`.
fn is_cycle(boundary: &[Vec<i32>], cycle: &[i64]) -> Result<bool> {
    for row in boundary {
        let mut sum = 0i128;
        for (&entry, &coefficient) in row.iter().zip(cycle.iter()) {
            let term = i128::from(entry)
                .checked_mul(i128::from(coefficient))
                .ok_or_else(overflow_error)?;
            sum = sum.checked_add(term).ok_or_else(overflow_error)?;
        }
        if sum != 0 {
            return Ok(false);
        }
    }
    Ok(true)
}

/// The fundamental cycle of the top-dimensional (`dim`) homology of `complex`:
/// the primitive generator of `ker ∂_dim` with a positive first coefficient.
///
/// `which` names the complex in error messages.  The top homology must have
/// rank one.
pub(super) fn fundamental_cycle(complex: &CwComplex, dim: usize, which: &str) -> Result<Vec<i64>> {
    let ncols = complex.count(dim);
    let boundary = complex.boundary_matrix(dim);
    let echelon = echelon_reduce(&boundary, ncols)?;
    let kernel_dim = ncols - echelon.pivot_cols.len();
    if kernel_dim != 1 {
        return Err(Error::unsupported(
            MAP_DEGREE,
            format!(
                "the top homology H_{dim} of the {which} complex has rank {kernel_dim}; \
                 a degree needs rank 1 (a sphere or any closed orientable {dim}-manifold)"
            ),
        ));
    }
    let cycle = kernel_generator(&echelon, ncols)?;
    if !is_cycle(&boundary, &cycle)? {
        return Err(Error::general(format!(
            "internal error: the computed fundamental cycle of the {which} complex has a boundary"
        )));
    }
    Ok(cycle)
}

/// The integer `d` with `image = d * target_cycle`.
fn degree_of_image(image: &[i128], target_cycle: &[i64]) -> Result<i64> {
    let not_a_multiple = || {
        Error::general(
            "the image of the fundamental cycle is not a multiple of the target's fundamental cycle",
        )
    };
    let anchor = target_cycle
        .iter()
        .position(|&entry| entry != 0)
        .ok_or_else(not_a_multiple)?;
    let denominator = i128::from(target_cycle[anchor]);
    let numerator = image.get(anchor).copied().ok_or_else(not_a_multiple)?;
    if numerator % denominator != 0 {
        return Err(not_a_multiple());
    }
    let degree = numerator / denominator;
    for (&image_entry, &cycle_entry) in image.iter().zip(target_cycle.iter()) {
        let expected = degree
            .checked_mul(i128::from(cycle_entry))
            .ok_or_else(overflow_error)?;
        if image_entry != expected {
            return Err(not_a_multiple());
        }
    }
    i64::try_from(degree).map_err(|_| overflow_error())
}
