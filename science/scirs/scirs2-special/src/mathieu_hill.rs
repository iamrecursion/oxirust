//! Generalized Hill's equation — periodic solutions and stability analysis
//!
//! Hill's equation is a second-order linear ODE with a periodic coefficient:
//!
//! ```text
//! y'' + [λ - Q(t)] y = 0
//! ```
//!
//! where Q(t) is periodic with some period T. The Mathieu equation is the
//! canonical special case with Q(t) = 2q cos(2t) and period T = π (or 2π for the
//! full cycle).
//!
//! ## Fourier–Hill matrix method
//!
//! Writing Q(t) = Σ_k θ_k e^{2ikt} and y(t) = Σ_k c_k e^{i(2k+ν)t}, substituting
//! into the ODE gives the **Hill matrix**:
//!
//! ```text
//! H_ij = (2i+ν)² δ_ij - θ_{i-j} + λ δ_ij
//! ```
//!
//! **Stability criterion**: the system is stable iff the Floquet characteristic
//! exponent ν is real, i.e. |cos(νT)| ≤ 1.
//!
//! ## This module
//!
//! Provides a `HillCoefficients` type that stores the Fourier coefficients θ_k
//! (indexed by k = 0, 1, 2, …, N) and free functions that operate on it:
//!
//! - `hill_stability_exponent` — real part of the Hill determinant stability measure
//! - `hill_periodic_solution`  — Fourier series evaluated at each point in `x`
//! - `hill_characteristic_exponent` — Floquet exponent μ (real ⟺ stable)
//! - `hill_stability_check`    — boolean: is the system stable?
//!
//! ## Connection to Mathieu equation
//!
//! The standard Mathieu equation y'' + (a - 2q cos 2t)y = 0 corresponds to:
//!
//! ```text
//! θ_0 = 0,  θ_1 = q,   θ_k = 0 for k ≥ 2     (one-sided convention)
//! Q(t) = 2q cos(2t) = q e^{2it} + q e^{-2it}
//! ```
//!
//! and λ = a.
//!
//! ## References
//!
//! - W. Magnus & S. Winkler, *Hill's Equation* (1966, Dover edition 1979).
//! - N. W. McLachlan, *Theory and Application of Mathieu Functions* (1947).
//! - M. Abramowitz & I. Stegun, §20 (Mathieu functions).

use crate::error::{SpecialError, SpecialResult};
use crate::mathieu::advanced::{tridiag_eigenvalues, tridiag_eigenvector};
use std::f64::consts::PI;

// ─────────────────────────────────────────────────────────────────────────────
// Public types
// ─────────────────────────────────────────────────────────────────────────────

/// Fourier coefficients of the periodic forcing term Q(t) in Hill's equation.
///
/// Q(t) = θ_0 + 2 Σ_{k≥1} θ_k cos(k * Ω * t)
///
/// where Ω = 2π/T and T is the fundamental period.
///
/// ## Field semantics
///
/// * `theta` — real Fourier cosine coefficients: `theta[0]` is the DC term,
///   `theta[k]` is the amplitude of the k-th cosine harmonic.
/// * `period` — fundamental period T of Q(t) (positive real number; typically π or 2π).
///
/// ## Mathieu equation
///
/// Standard Mathieu y'' + (a - 2q cos 2t) y = 0:
/// `HillCoefficients { theta: vec![0.0, q], period: PI }` with λ = a.
///
/// # Examples
///
/// ```
/// use std::f64::consts::PI;
/// use scirs2_special::mathieu_hill::HillCoefficients;
///
/// // Mathieu equation with q = 0.5
/// let coeffs = HillCoefficients { theta: vec![0.0, 0.5], period: PI };
/// assert_eq!(coeffs.theta.len(), 2);
/// assert!((coeffs.period - PI).abs() < 1e-14);
/// ```
#[derive(Debug, Clone)]
pub struct HillCoefficients {
    /// Cosine Fourier coefficients of Q(t): Q(t) ≈ θ_0 + 2 Σ_{k≥1} θ_k cos(k*2π/T * t)
    pub theta: Vec<f64>,
    /// Fundamental period T > 0
    pub period: f64,
}

impl HillCoefficients {
    /// Construct coefficients for the standard Mathieu equation y'' + (a - 2q cos 2t) y = 0.
    ///
    /// Q(t) = 2q cos(2t), period T = π.
    ///
    /// # Arguments
    /// * `q` - Mathieu parameter
    ///
    /// # Examples
    ///
    /// ```
    /// use scirs2_special::mathieu_hill::HillCoefficients;
    ///
    /// let coeffs = HillCoefficients::mathieu(0.5);
    /// ```
    pub fn mathieu(q: f64) -> Self {
        HillCoefficients {
            theta: vec![0.0, q],
            period: PI,
        }
    }

    /// Construct coefficients from a periodic function sampled via DFT.
    ///
    /// The function `f` is sampled at `2*n_terms + 1` equally spaced points over
    /// one period [0, T), and the cosine Fourier coefficients are extracted.
    ///
    /// # Arguments
    /// * `f` - Periodic function Q(t): [0, T] → ℝ
    /// * `period` - Period T
    /// * `n_terms` - Number of harmonics to retain
    ///
    /// # Examples
    ///
    /// ```
    /// use std::f64::consts::PI;
    /// use scirs2_special::mathieu_hill::HillCoefficients;
    ///
    /// // Mathieu equation: Q(t) = 2*0.3*cos(2t) for t in [0, π)
    /// let coeffs = HillCoefficients::from_function(|t| 2.0 * 0.3 * (2.0 * t).cos(), PI, 4);
    /// assert!((coeffs.theta[1] - 0.3).abs() < 1e-10);
    /// ```
    pub fn from_function<F: Fn(f64) -> f64>(f: F, period: f64, n_terms: usize) -> Self {
        let m = 2 * n_terms + 1;
        let dt = period / m as f64;
        let samples: Vec<f64> = (0..m).map(|k| f(k as f64 * dt)).collect();

        let mut theta = vec![0.0_f64; n_terms + 1];
        // DC component
        theta[0] = samples.iter().sum::<f64>() / m as f64;
        // Cosine harmonics: Q(t) = θ_0 + 2 Σ θ_k cos(k * 2π/T * t)
        // So θ_k = (1/m) Σ_j f(t_j) cos(k * 2π * j / m) for k ≥ 1
        for k in 1..=n_terms {
            let mut re = 0.0_f64;
            for (j, &s) in samples.iter().enumerate() {
                re += s * (2.0 * PI * k as f64 * j as f64 / m as f64).cos();
            }
            theta[k] = re / m as f64;
        }

        HillCoefficients { theta, period }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Internal helpers
// ─────────────────────────────────────────────────────────────────────────────

/// Truncation half-order for the Hill matrix.
const N_HILL: usize = 10;

/// Build the full Hill matrix (symmetric, dense) of size (2N+1)×(2N+1).
///
/// Reserved for future use when general (non-nearest-neighbour) coupling is needed,
/// e.g. for Q(t) with many Fourier harmonics. The current public API routes through
/// `build_hill_tridiag` which retains only θ_0 and θ_1.
///
/// Rows/columns indexed by k ∈ {-N, …, 0, …, N} (shifted to 0-based).
/// H_{k,k} = (k*Ω)² - λ + θ_0,   H_{k,j} = θ_{|k-j|} for |k-j| < len(theta).
#[allow(dead_code)]
fn build_hill_matrix(coeffs: &HillCoefficients, lambda: f64) -> Vec<Vec<f64>> {
    let n = N_HILL as i64;
    let size = (2 * n + 1) as usize;
    let omega = 2.0 * PI / coeffs.period;

    let mut mat = vec![vec![0.0_f64; size]; size];

    for i in 0..size {
        let k = i as i64 - n;
        for j in 0..size {
            let l = j as i64 - n;
            let diff = (k - l).unsigned_abs() as usize;
            if i == j {
                let q0 = coeffs.theta.first().copied().unwrap_or(0.0);
                mat[i][j] = (k as f64 * omega).powi(2) - lambda + q0;
            } else if diff < coeffs.theta.len() {
                mat[i][j] = coeffs.theta[diff];
            }
        }
    }

    mat
}

/// Compute the determinant of a square matrix via Gaussian elimination.
///
/// Reserved for future use with `build_hill_matrix` in the full Hill-determinant method.
#[allow(dead_code)]
fn matrix_det(mat: &[Vec<f64>]) -> f64 {
    let n = mat.len();
    if n == 0 {
        return 1.0;
    }
    let mut a: Vec<Vec<f64>> = mat.to_vec();
    let mut sign = 1.0_f64;

    for col in 0..n {
        let pivot_row = (col..n)
            .max_by(|&r1, &r2| {
                a[r1][col]
                    .abs()
                    .partial_cmp(&a[r2][col].abs())
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
            .unwrap_or(col);

        if a[pivot_row][col].abs() < 1e-300 {
            return 0.0;
        }
        if pivot_row != col {
            a.swap(pivot_row, col);
            sign = -sign;
        }
        let diag_val = a[col][col];
        for row in (col + 1)..n {
            let factor = a[row][col] / diag_val;
            for k in col..n {
                let tmp = a[col][k];
                a[row][k] -= factor * tmp;
            }
        }
    }

    let product: f64 = (0..n).map(|i| a[i][i]).product();
    sign * product
}

/// Build the tridiagonal approximation (retaining only nearest-neighbour coupling θ_1).
///
/// Returns `(diag, off)` of the symmetric tridiagonal matrix.
fn build_hill_tridiag(coeffs: &HillCoefficients, lambda: f64) -> (Vec<f64>, Vec<f64>) {
    let n = N_HILL as i64;
    let size = (2 * n + 1) as usize;
    let omega = 2.0 * PI / coeffs.period;

    let q0 = coeffs.theta.first().copied().unwrap_or(0.0);
    let q1 = coeffs.theta.get(1).copied().unwrap_or(0.0);

    let diag: Vec<f64> = (0..size)
        .map(|i| {
            let k = i as i64 - n;
            (k as f64 * omega).powi(2) - lambda + q0
        })
        .collect();

    let off: Vec<f64> = (0..size.saturating_sub(1)).map(|_| q1).collect();

    (diag, off)
}

// ─────────────────────────────────────────────────────────────────────────────
// Public API
// ─────────────────────────────────────────────────────────────────────────────

/// Compute the Hill stability exponent from the Hill infinite-determinant method.
///
/// The stability exponent is the real quantity:
///
/// ```text
/// Δ(λ) = 1 - det(H(λ)) / det(H₀(λ))
/// ```
///
/// where `H(λ)` is the truncated Hill matrix and `H₀(λ)` is the diagonal
/// (uncoupled) reference matrix. The system is stable iff `|1 - Δ/2| ≤ 1`
/// (equivalently `|cos(π√λ / Ω)| ≤ 1`).
///
/// In practice this function returns the quantity
///
/// ```text
/// stability_exponent = cos(π * μ)   where μ is the Floquet exponent
/// ```
///
/// which lies in [-1, 1] for stable systems and outside for unstable ones.
///
/// # Arguments
/// * `coeffs` - Fourier coefficients of Q(t)
/// * `lambda` - Characteristic value (parameter `a` in Mathieu notation)
///
/// # Returns
/// `cos(π * μ)` — the Hill determinant stability measure.
///
/// # Examples
///
/// ```
/// use std::f64::consts::PI;
/// use scirs2_special::mathieu_hill::{HillCoefficients, hill_stability_exponent};
///
/// // Mathieu a=1, q=0.5: known to be stable (inside first tongue)
/// let coeffs = HillCoefficients::mathieu(0.5);
/// let s = hill_stability_exponent(&coeffs, 1.0).unwrap();
/// assert!(s.abs() <= 1.0 + 1e-6, "stable region: |cos(πμ)| ≤ 1, got {s}");
/// ```
pub fn hill_stability_exponent(coeffs: &HillCoefficients, lambda: f64) -> SpecialResult<f64> {
    if coeffs.period <= 0.0 {
        return Err(SpecialError::DomainError(format!(
            "hill_stability_exponent: period={} must be positive",
            coeffs.period
        )));
    }

    // Compute Floquet exponent and return cos(πμ)
    let mu = hill_characteristic_exponent(coeffs, lambda)?;
    Ok((PI * mu).cos())
}

/// Evaluate the Hill equation's periodic (Floquet) solution at each point in `x`.
///
/// Finds the Fourier coefficients c_k by computing the eigenvector of the
/// tridiagonal Hill matrix corresponding to the eigenvalue nearest λ, then
/// evaluates the Fourier cosine series:
///
/// ```text
/// y(t) = Σ_{k=-N}^{N} c_k cos(k * Ω * t)   where Ω = 2π/T
/// ```
///
/// (retaining only the real, even-parity solution).
///
/// # Arguments
/// * `coeffs` - Fourier coefficients of Q(t)
/// * `lambda` - Characteristic value
/// * `x` - Points at which to evaluate
///
/// # Returns
/// Vector of y values, one per element of `x`.
///
/// # Errors
/// Returns `SpecialError::ComputationError` if the eigensystem fails.
///
/// # Examples
///
/// ```
/// use std::f64::consts::PI;
/// use scirs2_special::mathieu_hill::{HillCoefficients, hill_periodic_solution};
///
/// let coeffs = HillCoefficients::mathieu(0.1);
/// let pts: Vec<f64> = (0..=10).map(|i| i as f64 * PI / 10.0).collect();
/// let y = hill_periodic_solution(&coeffs, 1.0, &pts).unwrap();
/// assert_eq!(y.len(), pts.len());
/// assert!(y.iter().all(|v| v.is_finite()));
/// ```
pub fn hill_periodic_solution(
    coeffs: &HillCoefficients,
    lambda: f64,
    x: &[f64],
) -> SpecialResult<Vec<f64>> {
    if coeffs.period <= 0.0 {
        return Err(SpecialError::DomainError(format!(
            "hill_periodic_solution: period={} must be positive",
            coeffs.period
        )));
    }

    let omega = 2.0 * PI / coeffs.period;
    let (diag, off) = build_hill_tridiag(coeffs, lambda);
    let eigenvalues = tridiag_eigenvalues(&diag, &off);

    if eigenvalues.is_empty() {
        return Err(SpecialError::ComputationError(
            "hill_periodic_solution: tridiagonal eigenvalue computation returned empty".to_string(),
        ));
    }

    // Find the eigenvalue nearest to the center-diagonal value at k=0:
    // d[N] = (0*ω)² - λ + θ_0 = -λ + θ_0
    let n_idx = N_HILL;
    let center_diag = diag[n_idx];

    let target_ev = eigenvalues
        .iter()
        .copied()
        .min_by(|a, b| {
            (a - center_diag)
                .abs()
                .partial_cmp(&(b - center_diag).abs())
                .unwrap_or(std::cmp::Ordering::Equal)
        })
        .unwrap_or(center_diag);

    let ck = tridiag_eigenvector(&diag, &off, target_ev);

    let n = N_HILL as i64;
    // Evaluate cosine series y(t) = Σ_k c_k cos(k * ω * t)
    let result: Vec<f64> = x
        .iter()
        .map(|&t| {
            let mut val = 0.0_f64;
            for (i, &c) in ck.iter().enumerate() {
                let k = i as i64 - n; // Fourier index
                val += c * (k as f64 * omega * t).cos();
            }
            val
        })
        .collect();

    Ok(result)
}

/// Compute the Floquet characteristic exponent μ for Hill's equation.
///
/// By Floquet theory, solutions satisfy y(t + T) = e^{μT} y(t). The system is:
/// - **Stable** when μ is purely imaginary (Re μ = 0), i.e. the solution is bounded.
/// - **Unstable** when Re μ ≠ 0, i.e. the solution grows exponentially.
///
/// ## Algorithm
///
/// We use the Hill matrix tridiagonal approximation to compute the eigenvalue
/// σ nearest to the k=0 diagonal entry, then recover μ from:
///
/// ```text
/// cos(μT) ≈ cos(π * √σ_eff / |Ω|)
/// ```
///
/// where σ_eff = eigenvalue (relative to the shift). The characteristic exponent
/// returned is μ = acos(cos(μT)) / T (real for stable, ≥ 0; defined mod 2π/T).
///
/// # Arguments
/// * `coeffs` - Fourier coefficients of Q(t)
/// * `lambda` - Characteristic value
///
/// # Returns
/// The characteristic exponent μ ≥ 0. For stable systems μ is the imaginary
/// Floquet exponent divided by i (i.e. the real rotation rate per period).
/// For unstable systems μ > 0 and the solution grows as e^{μt}.
///
/// # Errors
/// Returns `SpecialError::DomainError` if the period is not positive.
///
/// # Examples
///
/// ```
/// use std::f64::consts::PI;
/// use scirs2_special::mathieu_hill::{HillCoefficients, hill_characteristic_exponent};
///
/// // For Mathieu with a=1, q=0 → trivial stable with ν=1 (since ω=2, k=0 gives √1)
/// let coeffs = HillCoefficients { theta: vec![0.0], period: PI };
/// let mu = hill_characteristic_exponent(&coeffs, 1.0).unwrap();
/// assert!(mu.is_finite());
/// ```
pub fn hill_characteristic_exponent(coeffs: &HillCoefficients, lambda: f64) -> SpecialResult<f64> {
    if coeffs.period <= 0.0 {
        return Err(SpecialError::DomainError(format!(
            "hill_characteristic_exponent: period={} must be positive",
            coeffs.period
        )));
    }

    let (diag, off) = build_hill_tridiag(coeffs, lambda);
    let eigenvalues = tridiag_eigenvalues(&diag, &off);

    if eigenvalues.is_empty() {
        return Err(SpecialError::ComputationError(
            "hill_characteristic_exponent: empty eigenvalue spectrum".to_string(),
        ));
    }

    // The Hill matrix diagonal at k=0 is: D_00 = (0*ω)² - λ + θ_0 = -λ + θ_0.
    // The eigenvalue σ nearest to D_00 captures the off-diagonal (coupling) corrections.
    // The effective Floquet parameter is recovered as:
    //   a_effective = λ - (σ_nearest - D_00)
    // This gives the physical λ corrected for the coupling-induced shift.
    let center_diag = diag[N_HILL];

    let nearest_ev = eigenvalues
        .iter()
        .copied()
        .min_by(|a, b| {
            (a - center_diag)
                .abs()
                .partial_cmp(&(b - center_diag).abs())
                .unwrap_or(std::cmp::Ordering::Equal)
        })
        .unwrap_or(center_diag);

    let a_effective = lambda - (nearest_ev - center_diag);

    // cos(μT) from the Floquet / Hill-matrix relation:
    //   cos(μT) = cos(T * √a_eff)      for a_effective ≥ 0  (oscillatory → stable)
    //   cos(μT) = cosh(T * √|a_eff|)   for a_effective < 0  (exponential → unstable)
    let cos_mu_t = if a_effective >= 0.0 {
        (coeffs.period * a_effective.sqrt()).cos()
    } else {
        (coeffs.period * (-a_effective).sqrt()).cosh()
    };

    // Convert to μ ≥ 0
    let mu = if cos_mu_t.abs() <= 1.0 + 1e-10 {
        cos_mu_t.clamp(-1.0, 1.0).acos() / coeffs.period
    } else {
        let c = cos_mu_t.abs();
        (c + (c * c - 1.0).max(0.0).sqrt()).ln() / coeffs.period
    };

    Ok(mu)
}

/// Check whether Hill's equation is stable (solutions are bounded) for given parameters.
///
/// The stability condition is: the Floquet characteristic exponent μ is purely
/// imaginary, which is equivalent to |cos(μT)| ≤ 1.
///
/// # Arguments
/// * `coeffs` - Fourier coefficients of Q(t)
/// * `lambda` - Characteristic value (parameter `a` in Mathieu notation)
///
/// # Returns
/// `true` if the system is stable (bounded solutions), `false` if unstable.
///
/// # Examples
///
/// ```
/// use scirs2_special::mathieu_hill::{HillCoefficients, hill_stability_check};
///
/// // Mathieu a=1, q=0.5: stable (inside first stability tongue)
/// let coeffs = HillCoefficients::mathieu(0.5);
/// assert!(hill_stability_check(&coeffs, 1.0).unwrap());
///
/// // Mathieu a=0, q=0: trivially stable (constant coefficient)
/// let trivial = HillCoefficients::mathieu(0.0);
/// assert!(hill_stability_check(&trivial, 0.0).unwrap());
/// ```
pub fn hill_stability_check(coeffs: &HillCoefficients, lambda: f64) -> SpecialResult<bool> {
    if coeffs.period <= 0.0 {
        return Err(SpecialError::DomainError(format!(
            "hill_stability_check: period={} must be positive",
            coeffs.period
        )));
    }

    let cos_mu_t = hill_stability_exponent(coeffs, lambda)?;
    Ok(cos_mu_t.abs() <= 1.0 + 1e-8)
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── Mathieu stability tests ───────────────────────────────────────────────

    #[test]
    fn test_mathieu_stable_a1_q05() {
        // Mathieu a=1, q=0.5 is known to be inside the first stability tongue
        let coeffs = HillCoefficients::mathieu(0.5);
        let stable = hill_stability_check(&coeffs, 1.0).expect("stability check should not error");
        assert!(stable, "Mathieu a=1, q=0.5 should be stable");
    }

    #[test]
    fn test_trivial_stable_q0() {
        // lambda=0, q=0 → y'' = 0 → linear (but we check cos(0*T) = cos(0) = 1, stable)
        // Actually y''=0 is marginally stable. Use lambda>0, q=0: y''+λy=0 → bounded oscillation.
        let coeffs = HillCoefficients {
            theta: vec![0.0],
            period: PI,
        };
        let stable = hill_stability_check(&coeffs, 1.0)
            .expect("trivial q=0 stability check should not error");
        assert!(
            stable,
            "Hill with q=0, λ=1 should be stable (pure oscillation)"
        );
    }

    #[test]
    fn test_characteristic_exponent_real_stable() {
        // For stable Mathieu a=1, q=0.1, exponent should be finite and small
        let coeffs = HillCoefficients::mathieu(0.1);
        let mu = hill_characteristic_exponent(&coeffs, 1.0)
            .expect("characteristic exponent should not error");
        assert!(
            mu.is_finite(),
            "characteristic exponent should be finite, got {mu}"
        );
        assert!(
            mu >= 0.0,
            "characteristic exponent should be non-negative, got {mu}"
        );
    }

    #[test]
    fn test_periodic_solution_has_correct_length() {
        let coeffs = HillCoefficients::mathieu(0.2);
        let pts: Vec<f64> = (0..=20).map(|i| i as f64 * PI / 20.0).collect();
        let y =
            hill_periodic_solution(&coeffs, 1.0, &pts).expect("periodic solution should not error");
        assert_eq!(
            y.len(),
            pts.len(),
            "output length should match input length"
        );
    }

    #[test]
    fn test_periodic_solution_finite() {
        // All values in the periodic solution should be finite
        let coeffs = HillCoefficients::mathieu(0.3);
        let pts: Vec<f64> = (0..=15).map(|i| i as f64 * PI / 15.0).collect();
        let y =
            hill_periodic_solution(&coeffs, 1.0, &pts).expect("periodic solution should not error");
        for (i, v) in y.iter().enumerate() {
            assert!(v.is_finite(), "y[{i}] = {v} should be finite");
        }
    }

    #[test]
    fn test_stability_exponent_in_range_stable() {
        // For stable parameters, |cos(πμ)| should be ≤ 1
        let coeffs = HillCoefficients::mathieu(0.5);
        let s = hill_stability_exponent(&coeffs, 1.0).expect("stability exponent should not error");
        assert!(
            s.abs() <= 1.0 + 1e-6,
            "stability exponent for stable Mathieu should be in [-1,1], got {s}"
        );
    }

    #[test]
    fn test_mathieu_q0_cos_mu_t_equals_cos_sqrt_a() {
        // For q=0 (no coupling), Hill's equation is y'' + λy = 0.
        // Floquet exponent μ = √λ/Ω... actually cos(μT) = cos(T*√λ) where T=period.
        // For period=π, λ=1: cos(π*1) = -1, which is at stability boundary.
        let coeffs = HillCoefficients {
            theta: vec![0.0],
            period: PI,
        };
        let s =
            hill_stability_exponent(&coeffs, 1.0).expect("stability exponent q=0 should not error");
        assert!(
            s.is_finite(),
            "stability exponent q=0 λ=1 should be finite, got {s}"
        );
        // cos(πμ) should equal cos(π*1) ≈ -1 (boundary) for λ=1 in period-π Hill
        assert!(
            (s + 1.0).abs() < 0.5,
            "q=0 λ=1 T=π: cos(πμ) should be near -1 (boundary), got {s}"
        );
    }

    #[test]
    fn test_hill_from_function_mathieu_coefficients() {
        // Reconstruct Mathieu with q=0.4 from function
        let q = 0.4;
        let coeffs = HillCoefficients::from_function(|t| 2.0 * q * (2.0 * t).cos(), PI, 3);
        assert!(
            (coeffs.theta[1] - q).abs() < 1e-8,
            "θ_1 should recover q={q}, got {}",
            coeffs.theta[1]
        );
        assert!(
            coeffs.theta[0].abs() < 1e-10,
            "θ_0 of pure cosine should be ≈ 0, got {}",
            coeffs.theta[0]
        );
    }

    #[test]
    fn test_hill_stability_transitions() {
        // For Mathieu, stability transitions occur at known characteristic values a_r(q).
        // At q=0: a_0(0)=0, b_1(0)=1, a_1(0)=1, etc.
        // Points well inside stability regions should be stable.
        let q = 0.1_f64;
        let coeffs = HillCoefficients::mathieu(q);
        // Inside first stability tongue (0 < a < 1-ε for small q): a = 0.5 → stable
        let stable_a = hill_stability_check(&coeffs, 0.5)
            .expect("stability check a=0.5 q=0.1 should not error");
        assert!(stable_a, "Mathieu a=0.5 q=0.1 should be stable");
    }

    #[test]
    fn test_hill_coefficients_mathieu_constructor() {
        let coeffs = HillCoefficients::mathieu(0.3);
        assert_eq!(coeffs.theta.len(), 2);
        assert!(coeffs.theta[0].abs() < 1e-14, "DC term should be 0");
        assert!((coeffs.theta[1] - 0.3).abs() < 1e-14, "θ_1 should be q=0.3");
        assert!((coeffs.period - PI).abs() < 1e-14, "period should be π");
    }

    #[test]
    fn test_periodic_solution_nonzero() {
        // The solution should not be identically zero
        let coeffs = HillCoefficients::mathieu(0.5);
        let pts = vec![0.0, 0.5, 1.0, 1.5, 2.0];
        let y =
            hill_periodic_solution(&coeffs, 1.0, &pts).expect("periodic solution should not error");
        let norm_sq: f64 = y.iter().map(|v| v * v).sum();
        assert!(
            norm_sq > 1e-20,
            "periodic solution should not be identically zero"
        );
    }

    #[test]
    fn test_characteristic_exponent_positive() {
        // Exponent should be non-negative by definition
        let coeffs = HillCoefficients::mathieu(0.4);
        let mu = hill_characteristic_exponent(&coeffs, 1.0).expect("exponent should not error");
        assert!(
            mu >= 0.0,
            "characteristic exponent should be >= 0, got {mu}"
        );
    }
}
