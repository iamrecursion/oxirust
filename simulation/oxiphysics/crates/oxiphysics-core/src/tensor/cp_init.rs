// Copyright 2026 COOLJAPAN OU (Team KitaSan)
// SPDX-License-Identifier: Apache-2.0

//! Start values for the alternating-least-squares CP decomposition.
//!
//! The ALS sweeps of [`cp_als`](super::cp_als) and of
//! [`TensorDecomposition::cp_als`](crate::numerical_linear_algebra::TensorDecomposition::cp_als)
//! solve one factor matrix after the other, so only the start values of the
//! factors decide which basin the iteration falls into and how many sweeps it
//! spends in the flat valleys ("swamps") of the CP objective.  [`CpInit`] selects
//! the start; the default is the HOSVD start, which is deterministic and lets the
//! sweeps start from the dominant subspaces of the tensor instead of from an
//! arbitrary point.

use crate::numerical_methods::symmetric_eigen_n;

/// Seed of the generator that completes an HOSVD start when the CP rank exceeds
/// a mode dimension (offset by the mode index, so every mode draws its own
/// columns).
const PAD_SEED: u64 = 0x2545_F491_4F6C_DD1D;

/// Columns shorter than this (in the Euclidean norm) are not rescaled to unit
/// length; they are replaced by a unit vector.
const MIN_PAD_NORM: f64 = 1e-12;

/// Relative slack within which two components count as equally large when the
/// sign of an eigenvector is fixed (see [`canonical_sign`]).
const SIGN_TIE_TOLERANCE: f64 = 1e-9;

/// Start values of the factor matrices of a CP-ALS run on a 3-way tensor.
///
/// Every variant is deterministic: the same tensor, rank and variant always give
/// the same start.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
#[non_exhaustive]
pub enum CpInit {
    /// Higher-order SVD (HOSVD) start, the default.
    ///
    /// Column `r` of the mode-`n` factor is the `r`-th leading left singular
    /// vector of the mode-`n` unfolding of the tensor, i.e. the `r`-th
    /// eigenvector of `X₍ₙ₎ X₍ₙ₎ᵀ` in descending order of the eigenvalue.  For a
    /// rank `R ≤ min(n₀, n₁, n₂)` tensor the factors therefore start with
    /// orthonormal columns that span the column spaces of the true factors, so
    /// the first normal matrix of the ALS sweep is the identity and the sweeps
    /// begin close to the solution.  The sign of every column is fixed (its
    /// component of largest magnitude is positive) so that the start does not
    /// depend on the sign the eigen-solver happens to return.
    ///
    /// A mode that has fewer than `rank` rows (`nₙ < rank`) only has `nₙ`
    /// singular vectors; the remaining columns are filled with seeded
    /// pseudo-random unit vectors, which makes the start deterministic but no
    /// better than a random one for those columns.  ALS is known to be slow
    /// when the rank exceeds a mode dimension, whatever the start.
    #[default]
    Hosvd,
    /// Pseudo-random start: every factor entry is drawn uniformly from
    /// `[-0.5, 0.5)` by a 64-bit linear congruential generator that is seeded
    /// with `seed`.  The factors are filled in the order `A`, `B`, `C`, row by
    /// row.  Different seeds give independent restarts of the iteration.
    Random {
        /// Seed of the generator.
        seed: u64,
    },
    /// The fixed ramp `A[i][r] = (i + r + 1)/R`, `B[j][r] = (j + r + 1)/R`,
    /// `C[k][r] = (k + r + 2)/R` that [`cp_als`](super::cp_als) used before the
    /// HOSVD start existed; it reproduces results computed with older versions.
    ///
    /// Every column is an affine function of the row index, so each factor has
    /// at most rank 2: for `R ≥ 3` the start is rank-deficient and the ALS
    /// sweeps need many iterations (or fail to converge at all) even on exactly
    /// low-rank tensors.
    Ramp,
}

/// 64-bit linear congruential generator behind [`CpInit::Random`] and the
/// padding of [`CpInit::Hosvd`].
struct Lcg(u64);

impl Lcg {
    /// Next value, uniform in `[-0.5, 0.5)`.
    fn next_centered(&mut self) -> f64 {
        self.0 = self
            .0
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        (self.0 >> 11) as f64 / (1u64 << 53) as f64 - 0.5
    }
}

/// Start factors `[A, B, C]` for a CP-ALS run of the given `rank` on the tensor
/// stored row-major in `data` (last index fastest) with the given `shape`.
///
/// Factor `n` is returned as `shape[n]` rows of `rank` entries each.
pub(crate) fn initial_cp_factors(
    data: &[f64],
    shape: [usize; 3],
    rank: usize,
    init: CpInit,
) -> [Vec<Vec<f64>>; 3] {
    match init {
        CpInit::Hosvd => [0, 1, 2].map(|mode| hosvd_factor(data, shape, rank, mode)),
        CpInit::Random { seed } => random_factors(shape, rank, seed),
        CpInit::Ramp => ramp_factors(shape, rank),
    }
}

/// Leading left singular vectors of the mode-`mode` unfolding as a
/// `shape[mode] × rank` factor (see [`CpInit::Hosvd`]).
fn hosvd_factor(data: &[f64], shape: [usize; 3], rank: usize, mode: usize) -> Vec<Vec<f64>> {
    let gram = mode_gram(data, shape, mode);
    let rows = gram.len();
    // `symmetric_eigen_n` lists the eigenvalues in ascending order and
    // `eigenvectors[i]` belongs to the i-th eigenvalue, so the leading singular
    // vectors are the last eigenvectors, the largest first.
    let (_, eigenvectors) = symmetric_eigen_n(&gram);
    let mut factor = vec![vec![0.0_f64; rank]; rows];
    for (column, vector) in eigenvectors.iter().rev().take(rank).enumerate() {
        let sign = canonical_sign(vector);
        for (row, &value) in factor.iter_mut().zip(vector) {
            row[column] = sign * value;
        }
    }
    if rank > rows {
        pad_columns(&mut factor, rows, mode);
    }
    factor
}

/// Gram matrix `X₍ₙ₎ X₍ₙ₎ᵀ` (`shape[mode] × shape[mode]`) of the mode-`mode`
/// unfolding of the row-major tensor `data`, without forming the unfolding.
fn mode_gram(data: &[f64], shape: [usize; 3], mode: usize) -> Vec<Vec<f64>> {
    let [n0, n1, n2] = shape;
    let mut gram = vec![vec![0.0_f64; shape[mode]]; shape[mode]];
    match mode {
        0 => {
            // Row `i` of the mode-0 unfolding is the contiguous slab `x[i, :, :]`.
            let width = n1 * n2;
            for i in 0..n0 {
                let row_i = &data[i * width..(i + 1) * width];
                for j in 0..=i {
                    let value = dot(row_i, &data[j * width..(j + 1) * width]);
                    gram[i][j] = value;
                    gram[j][i] = value;
                }
            }
        }
        1 => {
            // G[j][l] = Σ_i Σ_k x[i, j, k] x[i, l, k]; the fibres `x[i, j, :]` are contiguous.
            let slab_len = n1 * n2;
            for i in 0..n0 {
                let slab = &data[i * slab_len..(i + 1) * slab_len];
                for j in 0..n1 {
                    let fibre_j = &slab[j * n2..(j + 1) * n2];
                    for l in 0..=j {
                        let value = dot(fibre_j, &slab[l * n2..(l + 1) * n2]);
                        gram[j][l] += value;
                        if l != j {
                            gram[l][j] += value;
                        }
                    }
                }
            }
        }
        _ => {
            // G[k][l] = Σ_{i,j} x[i, j, k] x[i, j, l]; the fibres `x[i, j, :]` are contiguous.
            for fibre in 0..n0 * n1 {
                let values = &data[fibre * n2..(fibre + 1) * n2];
                for (k, &value_k) in values.iter().enumerate() {
                    for (l, &value_l) in values.iter().enumerate().take(k + 1) {
                        let product = value_k * value_l;
                        gram[k][l] += product;
                        if l != k {
                            gram[l][k] += product;
                        }
                    }
                }
            }
        }
    }
    gram
}

/// Euclidean inner product of two slices of the same length.
fn dot(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}

/// `+1` or `-1`, chosen so that the first component of `vector` whose magnitude
/// is (within [`SIGN_TIE_TOLERANCE`]) the largest one becomes positive.
///
/// The tolerance makes the choice stable when two components have the same
/// magnitude in exact arithmetic and differ only by rounding.
fn canonical_sign(vector: &[f64]) -> f64 {
    let largest = vector.iter().fold(0.0_f64, |m, v| m.max(v.abs()));
    let threshold = largest * (1.0 - SIGN_TIE_TOLERANCE);
    match vector.iter().find(|v| v.abs() >= threshold) {
        Some(&value) if value < 0.0 => -1.0,
        _ => 1.0,
    }
}

/// Fill the columns `first..` of `factor` with seeded pseudo-random unit vectors.
///
/// Used when the rank exceeds the number of rows, where the unfolding has no
/// further singular vectors.  The columns are drawn from the mode's own stream,
/// so they are distinct (with probability one) and reproducible.
fn pad_columns(factor: &mut [Vec<f64>], first: usize, mode: usize) {
    let rows = factor.len();
    let rank = factor.first().map_or(0, Vec::len);
    let mut rng = Lcg(PAD_SEED.wrapping_add(mode as u64));
    for column in first..rank {
        let mut values: Vec<f64> = (0..rows).map(|_| rng.next_centered()).collect();
        let norm = dot(&values, &values).sqrt();
        if norm > MIN_PAD_NORM {
            for value in &mut values {
                *value /= norm;
            }
        } else {
            values.fill(0.0);
            if let Some(slot) = values.get_mut(column % rows) {
                *slot = 1.0;
            }
        }
        for (row, &value) in factor.iter_mut().zip(&values) {
            row[column] = value;
        }
    }
}

/// Factors for [`CpInit::Random`].
fn random_factors(shape: [usize; 3], rank: usize, seed: u64) -> [Vec<Vec<f64>>; 3] {
    let mut rng = Lcg(seed);
    shape.map(|rows| {
        (0..rows)
            .map(|_| (0..rank).map(|_| rng.next_centered()).collect())
            .collect()
    })
}

/// Factors for [`CpInit::Ramp`].
fn ramp_factors(shape: [usize; 3], rank: usize) -> [Vec<Vec<f64>>; 3] {
    let step = 1.0 / rank as f64;
    let offsets = [1_usize, 1, 2];
    [0, 1, 2].map(|mode| {
        (0..shape[mode])
            .map(|row| {
                (0..rank)
                    .map(|r| (row + r + offsets[mode]) as f64 * step)
                    .collect()
            })
            .collect()
    })
}
