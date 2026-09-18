// Copyright 2026 COOLJAPAN OU (Team KitaSan)
// SPDX-License-Identifier: Apache-2.0

//! Lyapunov exponents.
//!
//! * [`LyapunovSpectrum`] accumulates the full spectrum from a sequence of
//!   tangent maps (Jacobians) with QR re-orthonormalisation (Benettin et al.).
//! * [`lyapunov_exponents`] estimates the spectrum from a measured trajectory
//!   with the Eckmann–Ruelle local-linear-fit method.
//! * [`max_lyapunov_from_trajectory`] estimates the largest exponent of a measured
//!   trajectory from the divergence of nearby states (Rosenstein et al.).
//! * [`logistic_lyapunov`] evaluates the exponent of the logistic map from its
//!   known derivative.

use crate::error::PhysicsError;
use nalgebra::{DMatrix, SVD};

/// Neighbourhoods whose displacement matrix has `σ_min <= RANK_TOLERANCE · σ_max`
/// do not determine a local Jacobian uniquely and are not used.
const RANK_TOLERANCE: f64 = 1e-10;

// ─── Lyapunov Spectrum ────────────────────────────────────────────────────────

/// Analyser for the full Lyapunov spectrum of a dynamical system via QR
/// decomposition of the accumulated tangent map.
///
/// Lyapunov exponents characterise the average rate of divergence (or
/// convergence) of nearby trajectories.  A positive leading exponent is a
/// necessary condition for chaos.
///
/// The analyser carries an orthonormal frame `Q` (initially the identity)
/// along the orbit.  Each [`update_qr`](Self::update_qr) call with the
/// Jacobian `J_k` of one step forms `M = J_k · Q_{k-1}`, factors `M = Q_k R_k`
/// by Householder QR, stores `Q_k` for the next step and accumulates
/// `ln |R_k[j, j]|` for every mode `j`.  Carrying the frame is essential: it is
/// what makes the accumulated diagonal the growth of the *product*
/// `J_k ⋯ J_1` rather than the singular values of the individual steps.  The
/// sum of all accumulated logarithms equals `Σ_k ln |det J_k|`.
#[derive(Debug, Clone)]
pub struct LyapunovSpectrum {
    /// Dimension of the phase space.
    pub dim: usize,
    /// Accumulated log-stretching rates for each mode.
    pub log_sums: Vec<f64>,
    /// Number of QR steps taken.
    pub steps: usize,
    /// Current orthonormal frame, row-major `dim × dim`, starting as the identity.
    frame: Vec<f64>,
}

impl LyapunovSpectrum {
    /// Construct a new spectrum analyser for a phase space of dimension `dim`.
    pub fn new(dim: usize) -> Self {
        let mut frame = vec![0.0_f64; dim * dim];
        for k in 0..dim {
            frame[k * dim + k] = 1.0;
        }
        Self {
            dim,
            log_sums: vec![0.0_f64; dim],
            steps: 0,
            frame,
        }
    }

    /// Perform one QR step given the current `dim × dim` Jacobian matrix
    /// stored in row-major order.
    ///
    /// The Jacobian is applied to the carried orthonormal frame, the product is
    /// QR-factorised, and the logarithms of the absolute values of the `R`
    /// diagonal are added to the per-mode sums.  The new `Q` becomes the frame
    /// for the next step.
    ///
    /// A mode annihilated by the Jacobian (`R[j, j] = 0`, a singular tangent map)
    /// has a true exponent of `-∞`; it is accumulated as `ln(f64::MIN_POSITIVE)`
    /// so that the resulting exponent is a very large finite negative number.  A
    /// non-finite Jacobian entry propagates as `NaN`.
    ///
    /// # Panics
    /// Panics if `jacobian.len() != dim * dim`.
    pub fn update_qr(&mut self, jacobian: &[f64]) {
        let d = self.dim;
        assert_eq!(jacobian.len(), d * d, "jacobian size mismatch");
        if d == 0 {
            self.steps += 1;
            return;
        }
        let j = DMatrix::from_row_slice(d, d, jacobian);
        let q = DMatrix::from_row_slice(d, d, &self.frame);
        let (q_next, r) = (j * q).qr().unpack();
        for (k, sum) in self.log_sums.iter_mut().enumerate() {
            let growth = r[(k, k)].abs();
            // `NaN < MIN_POSITIVE` is false, so a NaN diagonal stays NaN.
            let growth = if growth < f64::MIN_POSITIVE {
                f64::MIN_POSITIVE
            } else {
                growth
            };
            *sum += growth.ln();
        }
        for (idx, entry) in self.frame.iter_mut().enumerate() {
            *entry = q_next[(idx / d, idx % d)];
        }
        self.steps += 1;
    }

    /// Return the current Lyapunov exponents (per unit time `dt`).
    ///
    /// Divides each accumulated sum by total elapsed time `steps * dt`.
    /// Returns zeros if no steps have been taken.
    pub fn exponents(&self, dt: f64) -> Vec<f64> {
        let total = self.steps as f64 * dt;
        if total < 1e-300 {
            return vec![0.0_f64; self.dim];
        }
        self.log_sums.iter().map(|&s| s / total).collect()
    }
}

/// Legacy analyser wrapper kept for backwards compatibility.
pub struct LyapunovAnalyzer {
    /// Dimension of the phase space.
    pub dim: usize,
}

impl LyapunovAnalyzer {
    /// Construct a new analyser for a phase space of dimension `dim`.
    pub fn new(dim: usize) -> Self {
        Self { dim }
    }
}

/// Build an [`PhysicsError::InvalidInput`] for `field`.
fn invalid_input(field: &str, reason: String) -> PhysicsError {
    PhysicsError::InvalidInput {
        field: field.to_string(),
        reason,
    }
}

/// Validate a trajectory and flatten its states into one row-major array.
///
/// Returns the dimension of the states together with the `N · dim` values.
fn flatten_trajectory(trajectory: &[Vec<f64>]) -> Result<(usize, Vec<f64>), PhysicsError> {
    let Some(first) = trajectory.first() else {
        return Err(invalid_input(
            "trajectory",
            "trajectory must not be empty".to_string(),
        ));
    };
    let dim = first.len();
    if dim == 0 {
        return Err(invalid_input(
            "trajectory",
            "state vectors must have at least one component".to_string(),
        ));
    }
    if let Some((idx, state)) = trajectory
        .iter()
        .enumerate()
        .find(|(_, state)| state.len() != dim)
    {
        return Err(invalid_input(
            "trajectory",
            format!("state {idx} has {} components, expected {dim}", state.len()),
        ));
    }
    if trajectory.iter().flatten().any(|v| !v.is_finite()) {
        return Err(invalid_input(
            "trajectory",
            "trajectory contains non-finite values".to_string(),
        ));
    }
    Ok((dim, trajectory.iter().flatten().copied().collect()))
}

/// Estimate the Lyapunov spectrum of a measured trajectory with the
/// Eckmann–Ruelle method (local linear tangent-map fits followed by QR
/// re-orthonormalisation).
///
/// `trajectory` holds the successive states `x_0, …, x_{N-1}` of a
/// `d`-dimensional system sampled at the constant interval `dt`, so that
/// `x_{i+1} = F(x_i)` for the unknown time-`dt` map `F`.  For every transition
/// `i = 0 … N-2` the estimator
///
/// 1. selects the `2d + 1` states `x_j` (`j ≤ N-2`) nearest to `x_i` in
///    Euclidean distance, skipping temporally correlated candidates with
///    `|i − j| ≤ theiler`;
/// 2. fits the tangent map `J_i` by least squares,
///    `x_{j+1} − x_{i+1} ≈ J_i (x_j − x_i)` over those neighbours, using the
///    pseudo-inverse of the displacement matrix from a singular value
///    decomposition.  A neighbourhood whose displacement matrix is numerically
///    rank deficient (`σ_min ≤ 1e-10 σ_max`) does not determine `J_i` and is
///    skipped;
/// 3. feeds `J_i` to [`LyapunovSpectrum::update_qr`], which propagates an
///    orthonormal frame and accumulates the logarithmic growth rates.
///
/// The accumulated sums are divided by `used_steps · dt` and returned in
/// descending order (exponents per unit time).  The cost is `O(N² d)`.
///
/// # Reliability
/// * The leading exponents are robust.  Strongly contracting directions are
///   resolved only if the data scatter in that direction: for an attractor of
///   lower dimension than the embedding the corresponding exponents are
///   biased, and for data confined to a lower-dimensional subspace every
///   neighbourhood is rank deficient and no estimate is possible.
/// * `theiler` should be at least the decorrelation time in samples for flows
///   sampled faster than their dynamics; use `0` only for maps (it still
///   excludes `j = i`).
/// * Noise-free or weakly noisy data are assumed; the local fit does not model
///   measurement noise.
///
/// # Errors
/// [`PhysicsError::InvalidInput`] if `dt` is not positive and finite, the
/// trajectory is empty, has zero-dimensional or ragged states, contains
/// non-finite values, has fewer than `2d + 2·theiler + 3` states, or yields no
/// usable (full-rank) neighbourhood at all (for example a constant series);
/// [`PhysicsError::NumericalDivergence`] if the singular value decomposition of a
/// neighbourhood fails to converge or yields non-finite singular values
/// (displacements that overflow `f64`).
pub fn lyapunov_exponents(
    trajectory: &[Vec<f64>],
    dt: f64,
    theiler: usize,
) -> Result<Vec<f64>, PhysicsError> {
    if !(dt.is_finite() && dt > 0.0) {
        return Err(invalid_input(
            "dt",
            format!("time step must be positive and finite, got {dt}"),
        ));
    }
    let (dim, states) = flatten_trajectory(trajectory)?;

    let neighbours = 2 * dim + 1;
    let transitions = trajectory.len() - 1;
    let required = neighbours
        .saturating_add(theiler.saturating_mul(2))
        .saturating_add(1);
    if transitions < required {
        return Err(invalid_input(
            "trajectory",
            format!(
                "need at least {} states for dimension {dim} with Theiler window {theiler}, got {}",
                required.saturating_add(1),
                trajectory.len()
            ),
        ));
    }

    let state = |i: usize| &states[i * dim..(i + 1) * dim];

    let mut spectrum = LyapunovSpectrum::new(dim);
    let mut candidates: Vec<(f64, usize)> = Vec::with_capacity(transitions);
    let mut jacobian = vec![0.0_f64; dim * dim];
    for i in 0..transitions {
        let x_i = state(i);
        let image_i = state(i + 1);

        // Nearest admissible neighbours of x_i.  The (distance, index) order is
        // total, so ties between equidistant states resolve deterministically.
        candidates.clear();
        for j in (0..transitions).filter(|&j| i.abs_diff(j) > theiler) {
            let dist_sq: f64 = x_i
                .iter()
                .zip(state(j).iter())
                .map(|(a, b)| (a - b) * (a - b))
                .sum();
            candidates.push((dist_sq, j));
        }
        candidates.select_nth_unstable_by(neighbours - 1, |a, b| {
            a.0.total_cmp(&b.0).then(a.1.cmp(&b.1))
        });
        let chosen = &candidates[..neighbours];

        // Displacements u_j = x_j − x_i and their images w_j = x_{j+1} − x_{i+1}.
        let displacements =
            DMatrix::from_fn(neighbours, dim, |r, c| state(chosen[r].1)[c] - x_i[c]);
        let images = DMatrix::from_fn(neighbours, dim, |r, c| {
            state(chosen[r].1 + 1)[c] - image_i[c]
        });

        let Some(svd) = SVD::try_new(displacements, true, true, f64::EPSILON, 1000 * dim) else {
            return Err(PhysicsError::NumericalDivergence {
                message: format!(
                    "singular value decomposition of the neighbourhood of state {i} did not converge"
                ),
            });
        };
        // Displacements that overflow give non-finite singular values.  That is a
        // failed fit, not a rank-deficient neighbourhood.
        if svd.singular_values.iter().any(|s| !s.is_finite()) {
            return Err(PhysicsError::NumericalDivergence {
                message: format!(
                    "singular values of the neighbourhood of state {i} are not finite"
                ),
            });
        }
        let sigma_max = svd.singular_values.iter().fold(0.0_f64, |m, &s| m.max(s));
        let sigma_min = svd
            .singular_values
            .iter()
            .fold(f64::INFINITY, |m, &s| m.min(s));
        // An all-zero neighbourhood (sigma_max = 0) is rank deficient as well.
        if sigma_min <= RANK_TOLERANCE * sigma_max {
            continue;
        }

        // W ≈ U Jᵀ, so Jᵀ = U⁺ W (every singular value is above the tolerance).
        let Ok(j_transposed) = svd.solve(&images, 0.0) else {
            return Err(PhysicsError::NumericalDivergence {
                message: "singular vectors of the neighbourhood were not computed".to_string(),
            });
        };
        for (idx, entry) in jacobian.iter_mut().enumerate() {
            *entry = j_transposed[(idx % dim, idx / dim)];
        }
        spectrum.update_qr(&jacobian);
    }

    if spectrum.steps == 0 {
        return Err(invalid_input(
            "trajectory",
            format!(
                "no usable neighbourhood: all {transitions} neighbour sets are rank deficient, \
                 so the trajectory does not span the {dim}-dimensional state space"
            ),
        ));
    }
    let mut exponents = spectrum.exponents(dt);
    exponents.sort_by(|a, b| b.total_cmp(a));
    Ok(exponents)
}

/// Euclidean distance between two states, scaled by the largest component
/// difference so that squaring neither overflows nor underflows.
fn euclidean_distance(a: &[f64], b: &[f64]) -> f64 {
    let scale = a
        .iter()
        .zip(b.iter())
        .fold(0.0_f64, |largest, (x, y)| largest.max((x - y).abs()));
    if scale == 0.0 || !scale.is_finite() {
        return scale;
    }
    let sum_sq: f64 = a
        .iter()
        .zip(b.iter())
        .map(|(x, y)| {
            let ratio = (x - y) / scale;
            ratio * ratio
        })
        .sum();
    scale * sum_sq.sqrt()
}

/// Mean logarithmic divergence of initially close states, the divergence curve
/// of the Rosenstein–Collins–De Luca algorithm.
///
/// `trajectory` holds the successive states `x_0, …, x_{N-1}` of a system
/// sampled at a constant interval (use a delay embedding first for a scalar
/// time series).  For every reference state `x_i` with `i + k_max ≤ N-1` the
/// nearest neighbour `x_j` (Euclidean distance, `j + k_max ≤ N-1`) is selected
/// among the states at least `theiler + 1` samples away in time
/// (`|i − j| > theiler`, the Theiler window that removes temporally correlated
/// neighbours); coincident states (distance zero) are not neighbours.  The
/// returned vector holds, for the lags `k = 0 … k_max`, the average over all
/// reference states of the log separation `ln ‖x_{i+k} − x_{j+k}‖`, so entry
/// `k` is the mean log distance after `k` further steps.
///
/// A pair whose separation becomes exactly zero at some lag (the two orbits
/// merged) has no logarithm and is left out for all lags, which keeps the
/// averages over the same set of pairs.
///
/// # Errors
/// [`PhysicsError::InvalidInput`] if `k_max` is zero, the trajectory is empty,
/// has zero-dimensional or ragged states, contains non-finite values, has no
/// more than `k_max` states, or contains no admissible pair at all (for example
/// a constant series); [`PhysicsError::NumericalDivergence`] if a separation
/// overflows `f64`.
pub fn rosenstein_divergence_curve(
    trajectory: &[Vec<f64>],
    theiler: usize,
    k_max: usize,
) -> Result<Vec<f64>, PhysicsError> {
    if k_max == 0 {
        return Err(invalid_input(
            "k_max",
            "the divergence must be followed for at least one step, got k_max = 0".to_string(),
        ));
    }
    let (dim, states) = flatten_trajectory(trajectory)?;
    let count = trajectory.len();
    if count <= k_max {
        return Err(invalid_input(
            "trajectory",
            format!("need more than k_max = {k_max} states, got {count}"),
        ));
    }
    let state = |i: usize| &states[i * dim..(i + 1) * dim];

    // Reference states and neighbours must leave room for `k_max` further steps.
    let usable = count - k_max;
    let mut log_sums = vec![0.0_f64; k_max + 1];
    let mut log_separation = vec![0.0_f64; k_max + 1];
    let mut pairs = 0_usize;
    for i in 0..usable {
        let nearest = (0..usable)
            .filter(|&j| i.abs_diff(j) > theiler)
            .filter_map(|j| {
                let distance = euclidean_distance(state(i), state(j));
                (distance > 0.0).then_some((distance, j))
            })
            .fold(None::<(f64, usize)>, |best, candidate| match best {
                Some(current) if current.0 <= candidate.0 => Some(current),
                _ => Some(candidate),
            });
        let Some((_, j)) = nearest else {
            continue;
        };
        let mut merged = false;
        for (k, slot) in log_separation.iter_mut().enumerate() {
            let separation = euclidean_distance(state(i + k), state(j + k));
            if !separation.is_finite() {
                return Err(PhysicsError::NumericalDivergence {
                    message: format!(
                        "the separation of states {i} and {j} after {k} steps overflows f64"
                    ),
                });
            }
            if separation == 0.0 {
                merged = true;
                break;
            }
            *slot = separation.ln();
        }
        if merged {
            continue;
        }
        for (sum, term) in log_sums.iter_mut().zip(log_separation.iter()) {
            *sum += term;
        }
        pairs += 1;
    }
    if pairs == 0 {
        return Err(invalid_input(
            "trajectory",
            format!(
                "no admissible pair of nearby states: every candidate neighbour is within the \
                 Theiler window {theiler}, coincides with its reference state or merges with it \
                 within {k_max} steps"
            ),
        ));
    }
    Ok(log_sums.into_iter().map(|sum| sum / pairs as f64).collect())
}

/// Estimate the maximal Lyapunov exponent of a measured trajectory with the
/// Rosenstein–Collins–De Luca algorithm (M. T. Rosenstein, J. J. Collins and
/// C. J. De Luca, *A practical method for calculating largest Lyapunov exponents
/// from small data sets*, Physica D 65 (1993) 117).
///
/// Nearby states separate on average as `d(k) ≈ d₀ e^{λ k Δt}`, so the mean log
/// separation of the [`rosenstein_divergence_curve`] grows linearly with slope
/// `λ Δt` per step while the pairs are close.  The exponent is the least-squares
/// slope of that curve over the lags `0 … k_max`, divided by `dt` (exponent per
/// unit time).  Unlike [`lyapunov_exponents`] no tangent map is fitted, so the
/// estimate also works for short and moderately noisy data, but it yields only
/// the largest exponent.
///
/// # Reliability
/// * `k_max` must lie inside the linear region of the divergence curve.  A
///   chaotic orbit saturates once the separation reaches the size of the
///   attractor, which flattens the curve; choose `k_max` well below that lag
///   (inspect [`rosenstein_divergence_curve`]).  For a map whose nearest
///   neighbours start at a distance `d₀` the linear region is about
///   `ln(attractor size / d₀) / λ` steps long.
/// * `theiler` should be at least the decorrelation time of the signal in
///   samples (use `0` only for maps).
/// * Only the sign and magnitude of the largest exponent are meaningful.  The
///   estimate scales linearly with `1 / dt` and is unchanged by rescaling the
///   coordinates.
///
/// # Errors
/// [`PhysicsError::InvalidInput`] if `dt` is not positive and finite or the
/// trajectory is unusable for [`rosenstein_divergence_curve`];
/// [`PhysicsError::NumericalDivergence`] if a separation overflows `f64`.
pub fn max_lyapunov_from_trajectory(
    trajectory: &[Vec<f64>],
    dt: f64,
    theiler: usize,
    k_max: usize,
) -> Result<f64, PhysicsError> {
    if !(dt.is_finite() && dt > 0.0) {
        return Err(invalid_input(
            "dt",
            format!("time step must be positive and finite, got {dt}"),
        ));
    }
    let curve = rosenstein_divergence_curve(trajectory, theiler, k_max)?;
    let points = curve.len() as f64;
    let mean_lag = (points - 1.0) / 2.0;
    let mean_value = curve.iter().sum::<f64>() / points;
    let (mut covariance, mut variance) = (0.0_f64, 0.0_f64);
    for (lag, value) in curve.iter().enumerate() {
        let centred_lag = lag as f64 - mean_lag;
        covariance += centred_lag * (value - mean_value);
        variance += centred_lag * centred_lag;
    }
    Ok(covariance / variance / dt)
}

/// Estimate the maximal Lyapunov exponent of the logistic map `x → r·x·(1-x)`.
///
/// Uses the analytic formula: `λ = (1/N) Σ ln|r - 2·r·x_t|`.
/// For `r = 4` the value converges to `ln 2 ≈ 0.693`.
pub fn logistic_lyapunov(r: f64, x0: f64, n: usize) -> f64 {
    let mut x = x0;
    let mut sum = 0.0_f64;
    for _ in 0..n {
        let deriv = (r - 2.0_f64 * r * x).abs();
        if deriv > 1e-300 {
            sum += deriv.ln();
        }
        x = r * x * (1.0_f64 - x);
    }
    if n == 0 { 0.0_f64 } else { sum / n as f64 }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::f64::consts::LN_2;

    // --- LyapunovSpectrum ---

    #[test]
    fn test_lyapunov_spectrum_new() {
        let ls = LyapunovSpectrum::new(2);
        assert_eq!(ls.dim, 2);
        assert_eq!(ls.steps, 0);
        assert_eq!(ls.log_sums, vec![0.0, 0.0]);
    }

    #[test]
    fn test_lyapunov_spectrum_identity_jacobian() {
        let mut ls = LyapunovSpectrum::new(2);
        let identity = [1.0, 0.0, 0.0, 1.0];
        for _ in 0..10 {
            ls.update_qr(&identity);
        }
        for e in ls.exponents(1.0) {
            assert!(e.abs() < 1e-14, "exponent should vanish for identity: {e}");
        }
    }

    #[test]
    fn test_lyapunov_spectrum_diagonal_stretching() {
        let mut ls = LyapunovSpectrum::new(2);
        // Diagonal Jacobian with eigenvalues 2 and 0.5.
        let diag = [2.0, 0.0, 0.0, 0.5];
        for _ in 0..100 {
            ls.update_qr(&diag);
        }
        let exp = ls.exponents(1.0);
        assert_eq!(exp.len(), 2);
        assert!((exp[0] - LN_2).abs() < 1e-12, "leading exponent {}", exp[0]);
        assert!(
            (exp[1] + LN_2).abs() < 1e-12,
            "trailing exponent {}",
            exp[1]
        );
        // Exponents are per unit time.
        let per_second = ls.exponents(0.5);
        assert!((per_second[0] - 2.0 * LN_2).abs() < 1e-12);
    }

    #[test]
    fn test_lyapunov_spectrum_exponents_zero_dt() {
        let ls = LyapunovSpectrum::new(3);
        let exp = ls.exponents(0.0);
        for &e in &exp {
            assert_eq!(e, 0.0);
        }
    }

    #[test]
    fn test_lyapunov_spectrum_constant_swap_matrix_has_zero_exponents() {
        // J = [[0, 2], [0.5, 0]] satisfies J² = I, so the spectrum is (0, 0).
        // Factorising the raw Jacobian at every step (no carried frame) would
        // report ln(0.5) and ln(2).
        let jac = [0.0, 2.0, 0.5, 0.0];
        let mut ls = LyapunovSpectrum::new(2);
        for _ in 0..1000 {
            ls.update_qr(&jac);
        }
        for e in ls.exponents(1.0) {
            assert!(e.abs() < 1e-12, "exponent {e} should vanish");
        }
        // After an odd number of steps one half-period of the swap remains.
        let mut odd = LyapunovSpectrum::new(2);
        for _ in 0..3 {
            odd.update_qr(&jac);
        }
        let exp = odd.exponents(1.0);
        assert!((exp[0] + LN_2 / 3.0).abs() < 1e-12, "{exp:?}");
        assert!((exp[1] - LN_2 / 3.0).abs() < 1e-12, "{exp:?}");
    }

    #[test]
    fn test_lyapunov_spectrum_non_normal_constant_matrix() {
        // Lower-triangular J with eigenvalues 2 and 0.5.  The singular values of
        // one step differ from the eigenvalue moduli (the column norm of J is
        // √5), but the exponents of the product are ln 2 and ln 0.5.
        let jac = [2.0, 0.0, 1.0, 0.5];
        let mut ls = LyapunovSpectrum::new(2);
        for _ in 0..4000 {
            ls.update_qr(&jac);
        }
        let exp = ls.exponents(1.0);
        assert!((exp[0] - LN_2).abs() < 5e-4, "leading exponent {}", exp[0]);
        assert!((exp[1] + LN_2).abs() < 5e-4, "trailing exponent {}", exp[1]);
    }

    #[test]
    fn test_lyapunov_spectrum_elliptic_matrix_is_neutral() {
        // J = diag(2, 0.5) · R(0.7) has determinant 1 and trace 2.5 cos(0.7) < 2,
        // hence complex eigenvalues of modulus one: both exponents vanish.
        let (s, c) = 0.7_f64.sin_cos();
        let jac = [2.0 * c, -2.0 * s, 0.5 * s, 0.5 * c];
        let mut ls = LyapunovSpectrum::new(2);
        for _ in 0..4000 {
            ls.update_qr(&jac);
        }
        for e in ls.exponents(1.0) {
            assert!(e.abs() < 1e-3, "exponent {e} should vanish");
        }
    }

    #[test]
    fn test_lyapunov_spectrum_sum_of_exponents_is_log_determinant() {
        // The product of the R diagonals is |det J| at every step, whatever the
        // carried frame is.
        let mut state = 0xC0FF_EE12_3456_789A_u64;
        let mut next = move || {
            state = state
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            (state >> 11) as f64 / (1u64 << 53) as f64 * 2.0 - 1.0
        };
        let mut ls = LyapunovSpectrum::new(3);
        let mut log_det_sum = 0.0;
        for _ in 0..50 {
            let m: Vec<f64> = (0..9).map(|_| next()).collect();
            let det = m[0] * (m[4] * m[8] - m[5] * m[7]) - m[1] * (m[3] * m[8] - m[5] * m[6])
                + m[2] * (m[3] * m[7] - m[4] * m[6]);
            log_det_sum += det.abs().ln();
            ls.update_qr(&m);
        }
        let total: f64 = ls.log_sums.iter().sum();
        assert!(
            (total - log_det_sum).abs() < 1e-9,
            "{total} vs {log_det_sum}"
        );
    }

    #[test]
    fn test_lyapunov_spectrum_singular_jacobian_and_nan() {
        // A zero mode is accumulated at ln(MIN_POSITIVE): finite, hugely negative.
        let mut ls = LyapunovSpectrum::new(2);
        ls.update_qr(&[2.0, 0.0, 0.0, 0.0]);
        let exp = ls.exponents(1.0);
        assert!((exp[0] - LN_2).abs() < 1e-12);
        assert!(exp[1].is_finite() && exp[1] < -700.0, "{exp:?}");
        // Non-finite input is not silently converted into a finite exponent.
        let mut nan = LyapunovSpectrum::new(2);
        nan.update_qr(&[f64::NAN, 0.0, 0.0, 1.0]);
        assert!(nan.exponents(1.0).iter().any(|e| e.is_nan()));
    }

    #[test]
    #[should_panic(expected = "jacobian size mismatch")]
    fn test_lyapunov_spectrum_update_rejects_wrong_size() {
        LyapunovSpectrum::new(2).update_qr(&[1.0, 0.0, 0.0]);
    }

    #[test]
    fn test_lyapunov_spectrum_zero_dimension() {
        let mut ls = LyapunovSpectrum::new(0);
        ls.update_qr(&[]);
        assert_eq!(ls.steps, 1);
        assert!(ls.exponents(1.0).is_empty());
    }

    #[test]
    fn test_lyapunov_spectrum_fibonacci_matrix_gives_ln_golden_ratio() {
        // J = [[1, 1], [1, 0]] has the eigenvalues φ and -1/φ, so the exponents
        // are ±ln φ = ±0.4812.  Factorising the raw Jacobian at every step (no
        // carried frame) gives ±ln √2 = ±0.3466 instead.
        let ln_phi = ((1.0 + 5.0_f64.sqrt()) / 2.0).ln();
        let jac = [1.0, 1.0, 1.0, 0.0];
        let mut ls = LyapunovSpectrum::new(2);
        for _ in 0..2000 {
            ls.update_qr(&jac);
        }
        let exp = ls.exponents(1.0);
        assert!(
            (exp[0] - ln_phi).abs() < 1e-3,
            "leading exponent {}",
            exp[0]
        );
        assert!(
            (exp[1] + ln_phi).abs() < 1e-3,
            "trailing exponent {}",
            exp[1]
        );
        // |det J| = 1, so the exponents sum to zero for every frame.
        assert!((exp[0] + exp[1]).abs() < 1e-12, "{exp:?}");
    }

    #[test]
    fn test_lyapunov_spectrum_henon_tangent_map_matches_known_spectrum() {
        // Hénon map x' = 1 - a x² + y, y' = b x with a = 1.4, b = 0.3: the
        // tangent map is J = [[-2 a x, 1], [b, 0]] with |det J| = b at every
        // step, so λ1 + λ2 = ln b exactly; the literature value is λ1 ≈ 0.419.
        let (a, b) = (1.4_f64, 0.3_f64);
        let (mut x, mut y) = (0.1_f64, 0.1_f64);
        for _ in 0..1000 {
            (x, y) = (1.0 - a * x * x + y, b * x);
        }
        let steps = 30_000;
        let mut ls = LyapunovSpectrum::new(2);
        for _ in 0..steps {
            ls.update_qr(&[-2.0 * a * x, 1.0, b, 0.0]);
            (x, y) = (1.0 - a * x * x + y, b * x);
        }
        let exp = ls.exponents(1.0);
        assert!((exp[0] - 0.419).abs() < 0.01, "λ1 = {}", exp[0]);
        assert!((exp[1] + 1.623).abs() < 0.01, "λ2 = {}", exp[1]);
        assert!(
            (exp[0] + exp[1] - b.ln()).abs() < 1e-9,
            "λ1 + λ2 = {}",
            exp[0] + exp[1]
        );
    }

    // --- logistic_lyapunov ---

    #[test]
    fn test_logistic_lyapunov_r4_positive() {
        // At r = 4 the Lyapunov exponent is ln(2) ≈ 0.693.
        let le = logistic_lyapunov(4.0, 0.3, 10_000);
        assert!(le > 0.5, "logistic LE at r=4 should be > 0.5, got {}", le);
        assert!(le < 1.0, "logistic LE at r=4 should be < 1.0, got {}", le);
    }

    #[test]
    fn test_logistic_lyapunov_r2_negative() {
        // At r = 2 the map has a stable fixed point → negative LE.
        let le = logistic_lyapunov(2.0, 0.4, 5_000);
        assert!(
            le < 0.0,
            "logistic LE at r=2 should be negative, got {}",
            le
        );
    }

    #[test]
    fn test_logistic_lyapunov_zero_steps() {
        assert_eq!(logistic_lyapunov(4.0, 0.3, 0), 0.0);
    }

    // --- LyapunovAnalyzer ---

    #[test]
    fn test_lyapunov_analyzer_new() {
        let la = LyapunovAnalyzer::new(3);
        assert_eq!(la.dim, 3);
    }

    // --- lyapunov_exponents (Eckmann–Ruelle) ---

    /// Orbit of the logistic map as a one-dimensional trajectory with `n` states.
    fn logistic_orbit(r: f64, x0: f64, n: usize) -> Vec<Vec<f64>> {
        let mut x = x0;
        (0..n)
            .map(|_| {
                let state = vec![x];
                x = r * x * (1.0 - x);
                state
            })
            .collect()
    }

    /// Orbit of the Hénon map (a = 1.4, b = 0.3) after a transient of 1000 steps.
    fn henon_orbit(n: usize) -> Vec<Vec<f64>> {
        let (mut x, mut y) = (0.1_f64, 0.1_f64);
        let mut orbit = Vec::with_capacity(n);
        for step in 0..n + 1000 {
            let next = (1.0 - 1.4 * x * x + y, 0.3 * x);
            x = next.0;
            y = next.1;
            if step >= 1000 {
                orbit.push(vec![x, y]);
            }
        }
        orbit
    }

    #[test]
    fn test_lyapunov_exponents_henon_map() {
        // Known spectrum: λ1 ≈ 0.419, λ2 ≈ -1.623 (λ1 + λ2 = ln 0.3).  The local
        // linear fit biases the strongly contracting direction, so only its sign
        // and magnitude range are asserted.
        let orbit = henon_orbit(2000);
        let exps = lyapunov_exponents(&orbit, 1.0, 5).expect("valid trajectory");
        assert_eq!(exps.len(), 2);
        assert!(
            (exps[0] - 0.42).abs() < 0.05,
            "leading exponent {} should be near 0.42",
            exps[0]
        );
        assert!(exps[1] < -1.0, "trailing exponent {}", exps[1]);
        assert!(exps[0] >= exps[1], "exponents must be sorted descending");
    }

    #[test]
    fn test_lyapunov_exponents_logistic_map_r4_is_ln2() {
        let orbit = logistic_orbit(4.0, 0.3, 2000);
        let exps = lyapunov_exponents(&orbit, 1.0, 2).expect("valid trajectory");
        assert_eq!(exps.len(), 1);
        assert!((exps[0] - LN_2).abs() < 0.03, "λ = {}", exps[0]);
    }

    #[test]
    fn test_lyapunov_exponents_agree_with_analytic_logistic_derivative() {
        let n = 3000;
        for r in [3.7, 3.9, 4.0] {
            let orbit = logistic_orbit(r, 0.3, n);
            let estimated = lyapunov_exponents(&orbit, 1.0, 2).expect("valid trajectory")[0];
            let analytic = logistic_lyapunov(r, 0.3, n - 1);
            assert!(
                (estimated - analytic).abs() < 0.04,
                "r = {r}: estimated {estimated}, analytic {analytic}"
            );
        }
    }

    #[test]
    fn test_lyapunov_exponents_scale_with_time_step() {
        let orbit = logistic_orbit(4.0, 0.3, 1500);
        let per_step = lyapunov_exponents(&orbit, 1.0, 2).expect("valid")[0];
        let per_second = lyapunov_exponents(&orbit, 0.25, 2).expect("valid")[0];
        assert!((per_second - 4.0 * per_step).abs() < 1e-12);
    }

    #[test]
    fn test_lyapunov_exponents_exact_for_linear_one_dimensional_map() {
        // x_t = 2^t is the orbit of x → 2x: every local fit is exact and the
        // exponent is ln 2, independent of the (huge) state range.  The previous
        // implementation summed ln|Δx| per coordinate and depended on the units.
        let orbit: Vec<Vec<f64>> = (0..30).map(|t| vec![2.0_f64.powi(t)]).collect();
        let exps = lyapunov_exponents(&orbit, 1.0, 0).expect("valid trajectory");
        assert!((exps[0] - LN_2).abs() < 1e-9, "λ = {}", exps[0]);
        // Rescaling the coordinate (a change of units) must not change λ.
        let scaled: Vec<Vec<f64>> = orbit.iter().map(|s| vec![1.0e6 * s[0]]).collect();
        let exps_scaled = lyapunov_exponents(&scaled, 1.0, 0).expect("valid trajectory");
        assert!((exps_scaled[0] - exps[0]).abs() < 1e-9);
    }

    #[test]
    fn test_lyapunov_exponents_contracting_spiral() {
        // x → 0.99 R(0.3) x spirals inward; both exponents are ln 0.99 exactly,
        // and a unit conversion of the states does not change them.
        let (s, c) = 0.3_f64.sin_cos();
        let mut x = [1.0_f64, 0.0];
        let mut orbit = Vec::new();
        for _ in 0..200 {
            orbit.push(x.to_vec());
            x = [0.99 * (c * x[0] - s * x[1]), 0.99 * (s * x[0] + c * x[1])];
        }
        let exps = lyapunov_exponents(&orbit, 1.0, 0).expect("valid trajectory");
        assert_eq!(exps.len(), 2);
        for e in exps {
            assert!((e - 0.99_f64.ln()).abs() < 1e-8, "λ = {e}");
        }
    }

    #[test]
    fn test_lyapunov_exponents_linear_map_with_real_eigenvalues() {
        // x → A x with the upper-triangular A = [[1.1, 0.5], [0, 0.9]].  The first
        // frame vector e1 is the unstable eigenvector, so the QR iteration is
        // exact and numerically stable at every step: ln 1.1 and ln 0.9.
        let mut x = [1.0_f64, 1.0];
        let mut orbit = Vec::new();
        for _ in 0..60 {
            orbit.push(x.to_vec());
            x = [1.1 * x[0] + 0.5 * x[1], 0.9 * x[1]];
        }
        let exps = lyapunov_exponents(&orbit, 1.0, 0).expect("valid trajectory");
        assert!((exps[0] - 1.1_f64.ln()).abs() < 1e-6, "{exps:?}");
        assert!((exps[1] - 0.9_f64.ln()).abs() < 1e-6, "{exps:?}");
    }

    #[test]
    fn test_lyapunov_exponents_match_exact_qr_for_generic_linear_map() {
        // A = R(θ) diag(1.2, 0.8) R(-θ) has orthogonal eigenvectors that are not
        // the coordinate axes, so the identity start frame is generic.  Every
        // local fit reproduces A, hence the estimate must equal the finite-time
        // QR spectrum of the exact matrix, and it lies within the O(1/T)
        // finite-time bias of the asymptotic exponents ln 1.2 and ln 0.8.
        let (s, c) = 0.5_f64.sin_cos();
        let a = [
            1.2 * c * c + 0.8 * s * s,
            0.4 * s * c,
            0.4 * s * c,
            1.2 * s * s + 0.8 * c * c,
        ];
        let transitions = 29;
        let mut x = [1.0_f64, 0.3];
        let mut orbit = Vec::new();
        for _ in 0..=transitions {
            orbit.push(x.to_vec());
            x = [a[0] * x[0] + a[1] * x[1], a[2] * x[0] + a[3] * x[1]];
        }
        let exps = lyapunov_exponents(&orbit, 1.0, 0).expect("valid trajectory");

        let mut exact = LyapunovSpectrum::new(2);
        for _ in 0..transitions {
            exact.update_qr(&a);
        }
        let mut reference = exact.exponents(1.0);
        reference.sort_by(|p, q| q.total_cmp(p));
        for (est, expected) in exps.iter().zip(reference.iter()) {
            assert!(
                (est - expected).abs() < 1e-6,
                "estimate {exps:?} vs exact finite-time QR {reference:?}"
            );
        }
        // The two exponents add up to ln det A = ln 0.96 for any frame.
        assert!((exps.iter().sum::<f64>() - 0.96_f64.ln()).abs() < 1e-6);
        assert!((exps[0] - 1.2_f64.ln()).abs() < 0.02, "{exps:?}");
        assert!((exps[1] - 0.8_f64.ln()).abs() < 0.02, "{exps:?}");
    }

    #[test]
    fn test_lyapunov_exponents_rejects_invalid_input() {
        let orbit = logistic_orbit(4.0, 0.3, 50);
        let field_of = |result: Result<Vec<f64>, PhysicsError>| match result {
            Err(PhysicsError::InvalidInput { field, .. }) => field,
            other => panic!("expected InvalidInput, got {other:?}"),
        };
        assert_eq!(field_of(lyapunov_exponents(&[], 0.01, 0)), "trajectory");
        for dt in [0.0, -1.0, f64::NAN, f64::INFINITY] {
            assert_eq!(field_of(lyapunov_exponents(&orbit, dt, 0)), "dt");
        }
        assert_eq!(
            field_of(lyapunov_exponents(&[vec![], vec![]], 1.0, 0)),
            "trajectory"
        );
        let ragged = vec![vec![1.0, 2.0], vec![1.0], vec![0.0, 0.0]];
        assert_eq!(field_of(lyapunov_exponents(&ragged, 1.0, 0)), "trajectory");
        let mut with_nan = orbit.clone();
        with_nan[10][0] = f64::NAN;
        assert_eq!(
            field_of(lyapunov_exponents(&with_nan, 1.0, 0)),
            "trajectory"
        );
        // Too short for d = 1: needs 2d + 2θ + 3 = 5 + 2θ states.
        assert_eq!(
            field_of(lyapunov_exponents(&orbit[..4], 1.0, 0)),
            "trajectory"
        );
        assert!(lyapunov_exponents(&orbit[..5], 1.0, 0).is_ok());
        assert_eq!(
            field_of(lyapunov_exponents(&orbit[..20], 1.0, 8)),
            "trajectory"
        );
        assert_eq!(
            field_of(lyapunov_exponents(&orbit, 1.0, usize::MAX)),
            "trajectory"
        );
    }

    #[test]
    fn test_lyapunov_exponents_constant_series_has_no_usable_neighbourhood() {
        // A fixed point carries no information about the tangent map; the old
        // implementation reported finite "near zero" exponents for it.
        let constant: Vec<Vec<f64>> = vec![vec![1.0, 2.0]; 20];
        match lyapunov_exponents(&constant, 0.1, 0) {
            Err(PhysicsError::InvalidInput { field, reason }) => {
                assert_eq!(field, "trajectory");
                assert!(reason.contains("rank deficient"), "{reason}");
            }
            other => panic!("expected InvalidInput, got {other:?}"),
        }
        // A series confined to a line in a 2-D state space is rank deficient too.
        let line: Vec<Vec<f64>> = (0..40).map(|t| vec![t as f64, 2.0 * t as f64]).collect();
        assert!(lyapunov_exponents(&line, 1.0, 0).is_err());
    }

    #[test]
    fn test_lyapunov_exponents_never_return_non_finite_exponents() {
        // Displacements of order 1e200 overflow the squared distances and the
        // singular value decomposition.  The estimator must either report the
        // numerical failure or return finite exponents, never `Ok(NaN)`.
        let orbit: Vec<Vec<f64>> = (0..20)
            .map(|t| vec![1.0e200 * (1.0 + t as f64), 1.0e200 * (t * t) as f64])
            .collect();
        match lyapunov_exponents(&orbit, 1.0, 0) {
            Ok(exps) => assert!(exps.iter().all(|e| e.is_finite()), "{exps:?}"),
            Err(PhysicsError::NumericalDivergence { .. }) => {}
            Err(other) => panic!("unexpected error {other:?}"),
        }
    }

    #[test]
    fn test_lyapunov_exponents_theiler_window_excludes_adjacent_samples() {
        // For a slowly sampled quasi-continuous signal the nearest neighbours of
        // a state are its temporal neighbours; the Theiler window removes them.
        // The Hénon map has no such correlation, so the result is unchanged
        // within the estimator's accuracy.
        let orbit = henon_orbit(1500);
        let narrow = lyapunov_exponents(&orbit, 1.0, 0).expect("valid")[0];
        let wide = lyapunov_exponents(&orbit, 1.0, 25).expect("valid")[0];
        assert!((narrow - 0.42).abs() < 0.06, "θ = 0: {narrow}");
        assert!((wide - 0.42).abs() < 0.06, "θ = 25: {wide}");
    }

    // --- max_lyapunov_from_trajectory (Rosenstein) ---

    #[test]
    fn test_max_lyapunov_logistic_map_r4_is_ln2() {
        // The mean log distance of nearest-neighbour pairs grows by ln 2 per step
        // until it saturates at the size of the attractor after about 11 steps.
        let orbit = logistic_orbit(4.0, 0.3, 3000);
        let lambda = max_lyapunov_from_trajectory(&orbit, 1.0, 2, 5).expect("valid trajectory");
        assert!((lambda - LN_2).abs() < 0.02, "λ = {lambda}");
        let curve = rosenstein_divergence_curve(&orbit, 2, 14).expect("valid trajectory");
        let early_growth = (curve[7] - curve[0]) / 7.0;
        let late_growth = (curve[14] - curve[12]) / 2.0;
        assert!(
            (early_growth - LN_2).abs() < 0.02,
            "early growth {early_growth}"
        );
        assert!(late_growth < 0.3 * LN_2, "saturated growth {late_growth}");
    }

    #[test]
    fn test_max_lyapunov_ignores_units_and_scales_with_time_step() {
        // The previous estimator averaged ln|Δx| per coordinate, so a change of
        // units shifted every "exponent" by ln(factor) / dt.
        let orbit = logistic_orbit(4.0, 0.3, 2000);
        let scaled: Vec<Vec<f64>> = orbit.iter().map(|s| vec![10.0 * s[0]]).collect();
        let base = max_lyapunov_from_trajectory(&orbit, 1.0, 2, 5).expect("valid");
        let rescaled = max_lyapunov_from_trajectory(&scaled, 1.0, 2, 5).expect("valid");
        assert!((base - rescaled).abs() < 1e-10, "{base} vs {rescaled}");
        let per_second = max_lyapunov_from_trajectory(&orbit, 0.25, 2, 5).expect("valid");
        assert!((per_second - 4.0 * base).abs() < 1e-12);
    }

    #[test]
    fn test_max_lyapunov_henon_map() {
        // Literature value of the largest exponent: λ1 = 0.419 (a = 1.4, b = 0.3).
        let orbit = henon_orbit(2000);
        let lambda = max_lyapunov_from_trajectory(&orbit, 1.0, 5, 6).expect("valid trajectory");
        assert!((lambda - 0.419).abs() < 0.03, "λ = {lambda}");
    }

    #[test]
    fn test_max_lyapunov_exact_for_self_similar_orbits() {
        // Every pair of states on the orbit of x → 2x + 1 separates by exactly
        // the factor 2 per step, whichever neighbours are selected.
        let orbit: Vec<Vec<f64>> = (0..30).map(|t| vec![2.0_f64.powi(t) - 1.0]).collect();
        let curve = rosenstein_divergence_curve(&orbit, 0, 4).expect("valid trajectory");
        assert_eq!(curve.len(), 5);
        for (k, value) in curve.iter().enumerate() {
            assert!(
                (value - curve[0] - k as f64 * LN_2).abs() < 1e-9,
                "lag {k}: {value}"
            );
        }
        let lambda = max_lyapunov_from_trajectory(&orbit, 1.0, 0, 4).expect("valid trajectory");
        assert!((lambda - LN_2).abs() < 1e-9, "λ = {lambda}");

        // A contracting spiral x → 0.99 R(0.3) x is a similarity map: all pairs
        // contract by 0.99 per step.
        let (s, c) = 0.3_f64.sin_cos();
        let mut x = [1.0_f64, 0.0];
        let mut spiral = Vec::new();
        for _ in 0..200 {
            spiral.push(x.to_vec());
            x = [0.99 * (c * x[0] - s * x[1]), 0.99 * (s * x[0] + c * x[1])];
        }
        let lambda = max_lyapunov_from_trajectory(&spiral, 1.0, 0, 5).expect("valid trajectory");
        assert!((lambda - 0.99_f64.ln()).abs() < 1e-9, "λ = {lambda}");
    }

    #[test]
    fn test_max_lyapunov_of_periodic_orbit_is_zero() {
        // States 0, 1, 0, 1, ...: equal states are not neighbours, the nearest
        // distinct state stays at distance 1 for ever.
        let orbit: Vec<Vec<f64>> = (0..40).map(|t| vec![(t % 2) as f64]).collect();
        let lambda = max_lyapunov_from_trajectory(&orbit, 0.5, 0, 6).expect("valid trajectory");
        assert!(lambda.abs() < 1e-12, "λ = {lambda}");
    }

    #[test]
    fn test_rosenstein_curve_skips_pairs_that_merge() {
        // The orbits of states 0 and 1 merge in the fixed point 5, where the
        // separation is exactly zero and has no logarithm.  Every pair of this
        // series merges within two steps, so no pair is left.
        let merging: Vec<Vec<f64>> = [0.0, 1.0, 5.0, 5.0, 5.0, 5.0]
            .iter()
            .map(|&v| vec![v])
            .collect();
        match rosenstein_divergence_curve(&merging, 0, 2) {
            Err(PhysicsError::InvalidInput { field, reason }) => {
                assert_eq!(field, "trajectory");
                assert!(reason.contains("no admissible pair"), "{reason}");
            }
            other => panic!("expected InvalidInput, got {other:?}"),
        }
        // The same series is usable for a shorter horizon.
        let curve = rosenstein_divergence_curve(&merging, 0, 1).expect("one step is enough");
        assert_eq!(curve.len(), 2);
        assert!(curve.iter().all(|v| v.is_finite()), "{curve:?}");
    }

    #[test]
    fn test_max_lyapunov_rejects_invalid_input() {
        let orbit = logistic_orbit(4.0, 0.3, 200);
        let field_of = |result: Result<f64, PhysicsError>| match result {
            Err(PhysicsError::InvalidInput { field, .. }) => field,
            other => panic!("expected InvalidInput, got {other:?}"),
        };
        for dt in [0.0, -1.0, f64::NAN, f64::INFINITY] {
            assert_eq!(
                field_of(max_lyapunov_from_trajectory(&orbit, dt, 2, 5)),
                "dt"
            );
        }
        assert_eq!(
            field_of(max_lyapunov_from_trajectory(&orbit, 1.0, 2, 0)),
            "k_max"
        );
        assert_eq!(
            field_of(max_lyapunov_from_trajectory(&[], 1.0, 2, 5)),
            "trajectory"
        );
        assert_eq!(
            field_of(max_lyapunov_from_trajectory(&[vec![], vec![]], 1.0, 0, 1)),
            "trajectory"
        );
        let ragged = vec![vec![1.0, 2.0], vec![1.0], vec![0.0, 0.0]];
        assert_eq!(
            field_of(max_lyapunov_from_trajectory(&ragged, 1.0, 0, 1)),
            "trajectory"
        );
        let mut with_nan = orbit.clone();
        with_nan[10][0] = f64::NAN;
        assert_eq!(
            field_of(max_lyapunov_from_trajectory(&with_nan, 1.0, 0, 1)),
            "trajectory"
        );
        // k_max must be smaller than the number of states.
        assert_eq!(
            field_of(max_lyapunov_from_trajectory(&orbit[..5], 1.0, 0, 5)),
            "trajectory"
        );
        // A Theiler window that excludes every other state leaves no pair.
        assert_eq!(
            field_of(max_lyapunov_from_trajectory(&orbit, 1.0, usize::MAX, 5)),
            "trajectory"
        );
        // A constant series has no neighbour at positive distance.
        let constant: Vec<Vec<f64>> = vec![vec![3.0, 4.0]; 30];
        assert_eq!(
            field_of(max_lyapunov_from_trajectory(&constant, 1.0, 0, 5)),
            "trajectory"
        );
    }

    #[test]
    fn test_max_lyapunov_reports_overflowing_separations() {
        let orbit: Vec<Vec<f64>> = (0..12)
            .map(|t| vec![if t % 2 == 0 { f64::MAX } else { -f64::MAX }])
            .collect();
        match max_lyapunov_from_trajectory(&orbit, 1.0, 0, 3) {
            Err(PhysicsError::NumericalDivergence { .. }) => {}
            other => panic!("expected NumericalDivergence, got {other:?}"),
        }
    }
}
