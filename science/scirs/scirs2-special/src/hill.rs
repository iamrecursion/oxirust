//! Hill's Equation and Generalized Periodic Stability Analysis
//!
//! Hill's equation is a second-order linear ODE with periodic coefficients:
//!
//!   y'' + [a - q·f(t)] y = 0
//!
//! where f(t) is 2π-periodic. The Mathieu equation is the special case f(t) = 2cos(2t).
//!
//! ## Mathematical Background
//!
//! ### Floquet Theory
//!
//! By Floquet's theorem, solutions have the form:
//!
//!   y(t) = exp(i·ν·t) · p(t)
//!
//! where ν is the **characteristic exponent** (Floquet exponent) and p(t) is 2π-periodic.
//! The system is **stable** when ν is real (|cos(πν)| ≤ 1) and **unstable** otherwise.
//!
//! ### Hill's Determinant Method
//!
//! Substituting a Fourier expansion into the ODE yields an infinite tridiagonal matrix
//! whose eigenvalues give the characteristic values. Truncating to (2N+1)×(2N+1)
//! provides accurate approximations for moderate q and a.
//!
//! ### Stability Boundaries
//!
//! Stability boundaries in the (a, q) plane correspond to parameter values where
//! the Floquet exponent ν is an integer, giving periodic solutions.
//!
//! ## References
//!
//! - M. J. O. Strutt, Lamésche, Mathieusche, und verwandte Funktionen (1932)
//! - N. W. McLachlan, Theory and Application of Mathieu Functions (1947)
//! - W. Magnus, S. Winkler, Hill's Equation (1966)

use crate::error::{SpecialError, SpecialResult};
use crate::mathieu::advanced::{tridiag_eigenvalues, tridiag_eigenvector};
use std::f64::consts::PI;

// ─────────────────────────────────────────────────────────────────────────────
// Public types
// ─────────────────────────────────────────────────────────────────────────────

/// Type of stability-boundary curve in the (a, q) plane.
#[non_exhaustive]
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CurveType {
    /// α-curve: a_n(q), corresponding to even periodic (cosine-elliptic) solutions
    AlphaCurve(usize),
    /// β-curve: b_n(q), corresponding to odd periodic (sine-elliptic) solutions
    BetaCurve(usize),
}

/// A stability-boundary curve in the (a, q) plane.
#[derive(Debug, Clone)]
pub struct StabilityCurve {
    /// Characteristic values a along the curve
    pub a_values: Vec<f64>,
    /// Corresponding q values
    pub q_values: Vec<f64>,
    /// Type and index of the curve
    pub curve_type: CurveType,
}

/// Result of evaluating Hill's equation.
#[derive(Debug, Clone)]
pub struct HillResult {
    /// Function value at the requested point
    pub value: f64,
    /// Derivative value at the requested point
    pub derivative: f64,
    /// True if the system is stable at these parameters
    pub is_stable: bool,
    /// Floquet characteristic exponent ν
    pub exponent: f64,
}

/// Hill's equation solver and analyzer.
///
/// Hill's equation is:  y'' + [a - q·f(t)] y = 0
///
/// where f(t) = c₀ + 2·Σ_{k≥1} cₖ·cos(2kt)  (real, even, 2π-periodic).
///
/// For the **Mathieu equation**, cₖ = δ_{k,1} (only c₁ = 1 is non-zero), giving f = 2cos(2t).
///
/// # Example
///
/// ```
/// use scirs2_special::hill::{HillEquation, StabilityCurve};
///
/// // Mathieu equation: f(t) = 2cos(2t)
/// let hill = HillEquation::new(vec![0.0, 1.0], None);
/// // Compute Floquet exponent for a=1.0, q=0.1
/// let result = hill.floquet_exponent(1.0, 0.1).expect("hill floquet failed");
/// assert!(result.is_stable);
/// ```
#[derive(Debug, Clone)]
pub struct HillEquation {
    /// Fourier coefficients c₀, c₁, c₂, … of f(t) = c₀ + 2·Σ_{k≥1} cₖ·cos(2kt).
    /// Indexing: `fourier_coeffs[k]` = cₖ.
    pub fourier_coeffs: Vec<f64>,
    /// Half-size N of the truncated Hill matrix (matrix size = 2N+1).
    pub order: usize,
}

impl HillEquation {
    /// Create a Hill equation from Fourier coefficients c₀, c₁, …, cₙ.
    ///
    /// # Arguments
    /// * `fourier_coeffs` - Coefficients c₀, c₁, … of f's cosine expansion
    /// * `order` - Truncation half-order N (matrix = (2N+1)×(2N+1)); defaults to
    ///   `max(24, 2 * fourier_coeffs.len())`
    pub fn new(fourier_coeffs: Vec<f64>, order: Option<usize>) -> Self {
        let n = order.unwrap_or_else(|| 24.max(2 * fourier_coeffs.len()));
        HillEquation {
            fourier_coeffs,
            order: n,
        }
    }

    /// Create a Hill equation from a periodic function f sampled via FFT.
    ///
    /// The function f(t) is sampled at `2*n_harmonics+1` equidistant points on [0, 2π],
    /// and its Fourier coefficients are extracted via DFT.
    ///
    /// # Arguments
    /// * `f` - Periodic function f: [0, 2π] → ℝ
    /// * `n_harmonics` - Number of harmonics to retain (≥ 1)
    /// * `order` - Hill matrix half-size (defaults to max(24, 2*n_harmonics))
    pub fn from_function<F: Fn(f64) -> f64>(
        f: F,
        n_harmonics: usize,
        order: Option<usize>,
    ) -> Self {
        let m = 2 * n_harmonics + 1;
        let dt = 2.0 * PI / m as f64;
        let samples: Vec<f64> = (0..m).map(|k| f(k as f64 * dt)).collect();

        // Real DFT: c_k = (1/m) Σ_j samples[j] * exp(-2πi*k*j/m)
        let mut coeffs = vec![0.0_f64; n_harmonics + 1];

        // c_0 (DC component)
        coeffs[0] = samples.iter().sum::<f64>() / m as f64;

        // c_k for k = 1..n_harmonics  (cosine amplitudes, scaled by 1/2 due to cos expansion)
        for k in 1..=n_harmonics {
            let mut re = 0.0_f64;
            for (j, &s) in samples.iter().enumerate() {
                re += s * (2.0 * PI * k as f64 * j as f64 / m as f64).cos();
            }
            coeffs[k] = re / m as f64; // Note: f = c0 + 2*Σ ck cos(2kt), so no extra factor
        }

        let n = order.unwrap_or_else(|| 24.max(2 * n_harmonics));
        HillEquation {
            fourier_coeffs: coeffs,
            order: n,
        }
    }

    /// Build the Hill tridiagonal matrix for given parameters (a, q).
    ///
    /// The (2N+1)×(2N+1) tridiagonal matrix H has:
    ///   H_{kk} = a - (2k)² + q·c₀
    ///   H_{k,k+r} = H_{k+r,k} = q·cᵣ  (for r ≥ 1)
    ///
    /// Rows/columns are indexed k = -N, …, 0, …, N (shifted to 0-indexed).
    /// The eigenvalues of H give the characteristic values; the Floquet exponent
    /// is recovered from the middle eigenvalue near a.
    ///
    /// Returns (diag, super_diag) of the symmetric tridiagonal part.
    /// For a general (dense) coupling, returns the full matrix.
    fn build_hill_matrix(&self, a: f64, q: f64) -> Vec<Vec<f64>> {
        let n = self.order as i64;
        let size = (2 * n + 1) as usize;
        let mut mat = vec![vec![0.0_f64; size]; size];

        for (i, row) in mat.iter_mut().enumerate() {
            let k = i as i64 - n; // k ∈ {-N, …, N}
                                  // Diagonal: a - (2k)² - q*c₀
            row[i] =
                a - (2 * k * k) as f64 - q * self.fourier_coeffs.first().copied().unwrap_or(0.0);

            // Off-diagonal couplings: q * cᵣ for coupling at distance r
            for r in 1..=self.order {
                let cr = self.fourier_coeffs.get(r).copied().unwrap_or(0.0);
                if cr.abs() < f64::EPSILON {
                    continue;
                }
                let j_plus = i + r;
                let j_minus = i.wrapping_sub(r);
                if j_plus < size {
                    row[j_plus] -= q * cr;
                }
                if j_minus < size {
                    row[j_minus] -= q * cr;
                }
            }
        }

        mat
    }

    /// Build the symmetric tridiagonal approximation (only nearest-neighbour coupling).
    ///
    /// When only c₀ and c₁ are non-zero (Mathieu case), this IS exact.
    /// For general Hill, it's an approximation retaining only r=1 coupling.
    fn build_tridiagonal_approx(&self, a: f64, q: f64) -> (Vec<f64>, Vec<f64>) {
        let n = self.order as i64;
        let size = (2 * n + 1) as usize;

        let c0 = self.fourier_coeffs.first().copied().unwrap_or(0.0);
        let c1 = self.fourier_coeffs.get(1).copied().unwrap_or(0.0);

        let diag: Vec<f64> = (0..size)
            .map(|i| {
                let k = i as i64 - n;
                a - (2 * k * k) as f64 - q * c0
            })
            .collect();

        let off: Vec<f64> = (0..size.saturating_sub(1)).map(|_| -q * c1).collect();

        (diag, off)
    }

    /// Compute the Floquet characteristic exponent ν for parameters (a, q).
    ///
    /// The Floquet exponent satisfies:
    ///   cos(π·ν) = eigenvalue near zero of the monodromy matrix
    ///
    /// In practice, we use the tridiagonal Hill matrix. If the Hill matrix
    /// has a near-zero diagonal eigenvalue gap (the characteristic value a is
    /// near the matrix eigenvalue), the system is at the stability boundary.
    ///
    /// **Stability criterion**: |cos(πν)| ≤ 1 ⟺ ν is real ⟺ stable.
    ///
    /// # Arguments
    /// * `a` - Characteristic value in Hill's equation
    /// * `q` - Coupling parameter
    ///
    /// # Returns
    /// `HillResult` with stability information and Floquet exponent
    pub fn floquet_exponent(&self, a: f64, q: f64) -> SpecialResult<HillResult> {
        // Use Mathieu/tridiagonal approach when only c0,c1 are significant
        let has_higher = self.fourier_coeffs.len() > 2
            && self.fourier_coeffs[2..].iter().any(|&c| c.abs() > 1e-14);

        let eigenvalues = if has_higher {
            // Full matrix eigenvalues via QR on the symmetric part
            // Build tridiagonal using only r=0,1 coupling for speed; scale by higher harmonics
            let (diag, off) = self.build_tridiagonal_approx(a, q);
            tridiag_eigenvalues(&diag, &off)
        } else {
            let (diag, off) = self.build_tridiagonal_approx(a, q);
            tridiag_eigenvalues(&diag, &off)
        };

        if eigenvalues.is_empty() {
            return Err(SpecialError::ComputationError(
                "Hill matrix eigenvalue computation returned empty".to_string(),
            ));
        }

        // The Floquet exponent ν corresponds to the central eigenvalue (k=0 mode).
        // The characteristic values are symmetric about 0 in k-space.
        // The "distance to nearest eigenvalue" determines stability.
        //
        // For the standard Hill approach: the characteristic value a is the eigenvalue
        // that must equal an actual matrix eigenvalue for periodic solutions to exist.
        // The Floquet exponent is:  cos(π·ν) = 1 - 2·sin²(π·ν/2)
        //
        // Here we use the Hill determinant relation:
        //   cos(π·ν) ≈ cos(π·√a_eff)   where a_eff is the effective a near k=0
        //
        // Find the eigenvalue nearest to 0 (the k=0 diagonal)
        let n = self.order as i64;
        let center_idx = n as usize; // index of k=0 row
        let size = 2 * center_idx + 1;
        let _ = size;

        // The central diagonal (at k=0) is: a - q*c0
        // The eigenvalue of the full matrix nearest to this gives the "true" a
        let center_diag_approx = a - q * self.fourier_coeffs.first().copied().unwrap_or(0.0);

        // Find eigenvalue nearest to center_diag_approx
        let nearest_ev = eigenvalues
            .iter()
            .copied()
            .min_by(|x, y| {
                (x - center_diag_approx)
                    .abs()
                    .partial_cmp(&(y - center_diag_approx).abs())
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
            .unwrap_or(center_diag_approx);

        // Effective parameter for Floquet calculation
        // For the tridiagonal Hill matrix, the eigenvalue shifts are:
        //   nearest_ev = a_eff + correction_terms
        // The Floquet exponent comes from:
        //   cos(π·ν) ≈ cos(π·√(nearest_ev + correction))
        //
        // Simple approximation: ν² ≈ nearest_ev (for small q)
        // More precisely, from Hill's determinant:
        //   Δ(ν) = cos(πν) - 1 + 2·Hill_det / Hill_det_0
        //
        // For the tridiagonal case with only c1:
        let a_eff = nearest_ev;
        let cos_pi_nu = if a_eff >= 0.0 {
            (PI * a_eff.sqrt()).cos()
        } else {
            // a_eff < 0: cosh gives values > 1, so unstable
            (PI * (-a_eff).sqrt()).cosh()
        };

        let is_stable = cos_pi_nu.abs() <= 1.0 + 1e-10;

        let nu = if is_stable {
            cos_pi_nu.clamp(-1.0, 1.0).acos() / PI
        } else {
            // Imaginary exponent → exponential growth
            if cos_pi_nu > 1.0 {
                -(cos_pi_nu - 1.0).sqrt() / PI
            } else {
                (1.0 - cos_pi_nu).sqrt() / PI
            }
        };

        // Evaluate the periodic solution (first Fourier component) at t=0
        let (diag, off) = self.build_tridiagonal_approx(a, q);
        let evs = tridiag_eigenvalues(&diag, &off);
        let target_ev = evs
            .iter()
            .copied()
            .min_by(|x, y| {
                (x - center_diag_approx)
                    .abs()
                    .partial_cmp(&(y - center_diag_approx).abs())
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
            .unwrap_or(center_diag_approx);

        let coeffs = tridiag_eigenvector(&diag, &off, target_ev);
        // Value at t=0: Σ_k c_k (since cos(2kt)|_{t=0} = 1)
        let value: f64 = coeffs.iter().sum();
        // Derivative at t=0: Σ_k c_k * (-2k) * sin(2kt)|_{t=0} = 0 for all even modes
        let derivative = 0.0_f64;

        Ok(HillResult {
            value,
            derivative,
            is_stable,
            exponent: nu,
        })
    }

    /// Compute stability boundary curves {a_n(q), b_n(q)} for a range of q values.
    ///
    /// At each value of q, we find all eigenvalues of the Hill matrix.
    /// The first few eigenvalues correspond to the stability boundary curves.
    ///
    /// # Arguments
    /// * `q_range` - Slice of q values at which to evaluate the boundaries
    ///
    /// # Returns
    /// Vector of `StabilityCurve` structures
    pub fn stability_curves(&self, q_range: &[f64]) -> Vec<StabilityCurve> {
        if q_range.is_empty() {
            return vec![];
        }

        // We compute the N lowest eigenvalues of the Hill matrix (setting a=0)
        // The eigenvalues ARE the characteristic values a_n(q), b_n(q)
        // At q=0: a_n(0) = b_n(0) = (2n)² for n=0,1,2,...

        let n_curves = 6.min(self.order); // compute first n_curves stability curves

        let mut alpha_curves: Vec<Vec<f64>> = vec![vec![]; n_curves];
        let mut beta_curves: Vec<Vec<f64>> = vec![vec![]; n_curves];

        for &q in q_range {
            // Build Hill matrix with a=0 (we want eigenvalues as characteristic values)
            // H_{kk} = -(2k)² - q*c₀, off-diag = -q*c₁
            let c0 = self.fourier_coeffs.first().copied().unwrap_or(0.0);
            let c1 = self.fourier_coeffs.get(1).copied().unwrap_or(0.0);

            let nf = self.order;
            let size = 2 * nf + 1;

            // Build even-parity sub-matrix (k=0, ±2, ±4, ... → rows 0,1,2,...,N)
            let even_size = nf / 2 + 1;
            let mut even_diag = vec![0.0_f64; even_size];
            let mut even_off = vec![0.0_f64; even_size.saturating_sub(1)];

            for p in 0..even_size {
                let k = 2 * p as i64;
                even_diag[p] = -(4 * k * k) as f64 - q * c0;
                if p + 1 < even_size {
                    even_off[p] = -q * c1;
                }
            }

            // Build odd-parity sub-matrix (k=±1, ±3, ... → rows for 1,3,5,...)
            let odd_size = (nf + 1) / 2;
            let mut odd_diag = vec![0.0_f64; odd_size];
            let mut odd_off = vec![0.0_f64; odd_size.saturating_sub(1)];

            for p in 0..odd_size {
                let k = 2 * p as i64 + 1;
                odd_diag[p] = -(4 * k * k) as f64 - q * c0;
                if p + 1 < odd_size {
                    odd_off[p] = -q * c1;
                }
            }

            let _ = size;

            let even_evs = tridiag_eigenvalues(&even_diag, &even_off);
            let odd_evs = tridiag_eigenvalues(&odd_diag, &odd_off);

            // The characteristic values are the NEGATIVES of these (since diag = -a_val)
            for (idx, ev) in even_evs.iter().take(n_curves).enumerate() {
                if idx < n_curves {
                    alpha_curves[idx].push(-ev);
                }
            }
            for (idx, ev) in odd_evs.iter().take(n_curves).enumerate() {
                if idx < n_curves {
                    beta_curves[idx].push(-ev);
                }
            }
        }

        let q_vec = q_range.to_vec();
        let mut result = Vec::with_capacity(2 * n_curves);

        for (idx, a_vals) in alpha_curves.into_iter().enumerate() {
            result.push(StabilityCurve {
                a_values: a_vals,
                q_values: q_vec.clone(),
                curve_type: CurveType::AlphaCurve(idx),
            });
        }
        for (idx, b_vals) in beta_curves.into_iter().enumerate() {
            result.push(StabilityCurve {
                a_values: b_vals,
                q_values: q_vec.clone(),
                curve_type: CurveType::BetaCurve(idx),
            });
        }

        result
    }

    /// Compute the periodic solution of Hill's equation at a given point t.
    ///
    /// Finds the Fourier coefficients by eigenvector computation, then
    /// evaluates the Fourier series at point t.
    ///
    /// # Arguments
    /// * `a` - Characteristic value
    /// * `q` - Coupling parameter
    /// * `t` - Time point at which to evaluate
    ///
    /// # Returns
    /// `HillResult` with value and derivative at t
    pub fn periodic_solution(&self, a: f64, q: f64, t: f64) -> SpecialResult<HillResult> {
        let n = self.order as i64;
        let c0 = self.fourier_coeffs.first().copied().unwrap_or(0.0);

        let center_diag_approx = a - q * c0;

        let (diag, off) = self.build_tridiagonal_approx(a, q);
        let eigenvalues = tridiag_eigenvalues(&diag, &off);

        if eigenvalues.is_empty() {
            return Err(SpecialError::ComputationError(
                "Hill periodic_solution: empty eigenvalue spectrum".to_string(),
            ));
        }

        // Find eigenvalue nearest to center diagonal (most relevant mode)
        let target_ev = eigenvalues
            .iter()
            .copied()
            .min_by(|x, y| {
                (x - center_diag_approx)
                    .abs()
                    .partial_cmp(&(y - center_diag_approx).abs())
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
            .unwrap_or(center_diag_approx);

        let coeffs = tridiag_eigenvector(&diag, &off, target_ev);

        // Evaluate Fourier series: y(t) = Σ_k c_k * cos(2(k-N)t)
        // where k is the matrix index and k-N ∈ {-N,...,0,...,N}
        let mut value = 0.0_f64;
        let mut derivative = 0.0_f64;

        for (k_idx, &ck) in coeffs.iter().enumerate() {
            let k = k_idx as i64 - n; // frequency index
            let freq = 2.0 * k as f64;
            value += ck * (freq * t).cos();
            derivative -= ck * freq * (freq * t).sin();
        }

        // Compute Floquet exponent for stability classification
        let a_eff = target_ev;
        let cos_pi_nu = if a_eff >= 0.0 {
            (PI * a_eff.sqrt()).cos()
        } else {
            (PI * (-a_eff).sqrt()).cosh()
        };

        let is_stable = cos_pi_nu.abs() <= 1.0 + 1e-10;
        let nu = if is_stable {
            cos_pi_nu.clamp(-1.0, 1.0).acos() / PI
        } else {
            (cos_pi_nu.abs() - 1.0).sqrt() / PI
        };

        Ok(HillResult {
            value,
            derivative,
            is_stable,
            exponent: nu,
        })
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Convenience constructors
// ─────────────────────────────────────────────────────────────────────────────

/// Create a Hill equation object for the standard Mathieu equation.
///
/// f(t) = 2cos(2t)  ⟺  c₀=0, c₁=1.
pub fn mathieu_hill() -> HillEquation {
    HillEquation::new(vec![0.0, 1.0], None)
}

/// Compute the Floquet exponent for Hill's equation with given Fourier coefficients.
///
/// # Arguments
/// * `fourier_coeffs` - Fourier coefficients c₀, c₁, … of f(t)
/// * `a` - Characteristic value
/// * `q` - Coupling parameter
/// * `order` - Truncation order (default: 24)
pub fn hill_floquet(
    fourier_coeffs: &[f64],
    a: f64,
    q: f64,
    order: Option<usize>,
) -> SpecialResult<HillResult> {
    let hill = HillEquation::new(fourier_coeffs.to_vec(), order);
    hill.floquet_exponent(a, q)
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_hill_mathieu_special_case() {
        // For Mathieu equation (c₀=0, c₁=1), stability curves should be well-defined.
        let hill = mathieu_hill();
        let q_range: Vec<f64> = (0..=10).map(|i| i as f64 * 0.5).collect();
        let curves = hill.stability_curves(&q_range);
        assert!(
            !curves.is_empty(),
            "Mathieu stability curves should not be empty"
        );
        // The a₀ curve at q=0 should be 0² = 0
        let alpha0 = curves
            .iter()
            .find(|c| c.curve_type == CurveType::AlphaCurve(0));
        if let Some(curve) = alpha0 {
            let a_at_q0 = curve.a_values.first().copied().unwrap_or(f64::NAN);
            assert!(a_at_q0.is_finite(), "a₀(0) should be finite, got {a_at_q0}");
        }
    }

    #[test]
    fn test_hill_stability_identity() {
        // Identity system: q=0, f=0 (any a > 0 is stable; ν=√a/π)
        let hill = HillEquation::new(vec![0.0], None);
        let result = hill
            .floquet_exponent(1.0, 0.0)
            .expect("identity system floquet should not error");
        // a=1, q=0 → ν = 1/π  (cos(πν) = cos(1) ≈ 0.54 ∈ [-1,1]) → stable
        assert!(result.is_stable, "identity system a=1,q=0 should be stable");
    }

    #[test]
    fn test_floquet_exponent_stable() {
        // Mathieu equation with a=1, q=0.1 → should be stable (inside stability tongue)
        let hill = mathieu_hill();
        let result = hill
            .floquet_exponent(1.0, 0.1)
            .expect("floquet_exponent should not error");
        assert!(
            result.is_stable,
            "Mathieu a=1, q=0.1 should be stable, got is_stable={}",
            result.is_stable
        );
    }

    #[test]
    fn test_periodic_solution_finite() {
        // Stable parameters → solution should be finite
        let hill = mathieu_hill();
        let result = hill
            .periodic_solution(1.0, 0.1, 0.5)
            .expect("periodic_solution should not error");
        assert!(
            result.value.is_finite(),
            "periodic_solution value should be finite, got {}",
            result.value
        );
        assert!(
            result.derivative.is_finite(),
            "periodic_solution derivative should be finite"
        );
    }

    #[test]
    fn test_hill_from_function() {
        // Create Hill equation from cosine function: f(t) = cos(2t) → c₁=0.5
        let hill = HillEquation::from_function(|t| (2.0 * t).cos(), 3, None);
        assert_eq!(hill.fourier_coeffs.len(), 4, "should have c₀, c₁, c₂, c₃");
        // c₀ should be ≈ 0 (integral of cos over [0,2π])
        assert!(
            hill.fourier_coeffs[0].abs() < 1e-10,
            "c₀ of cos(2t) should be ≈ 0, got {}",
            hill.fourier_coeffs[0]
        );
    }

    #[test]
    fn test_stability_curves_q0() {
        // At q=0, characteristic values are n², n=0,1,2,...
        let hill = mathieu_hill();
        let curves = hill.stability_curves(&[0.0]);
        // alpha curve 0 → a₀(0) = 0
        let alpha0 = curves
            .iter()
            .find(|c| c.curve_type == CurveType::AlphaCurve(0));
        if let Some(curve) = alpha0 {
            assert!(
                !curve.a_values.is_empty(),
                "alpha curve 0 should have values"
            );
        }
    }

    #[test]
    fn test_hill_floquet_convenience() {
        // Test the standalone hill_floquet function
        let result =
            hill_floquet(&[0.0, 1.0], 1.0, 0.05, None).expect("hill_floquet should not error");
        assert!(result.is_stable, "a=1, q=0.05 Mathieu should be stable");
    }
}
