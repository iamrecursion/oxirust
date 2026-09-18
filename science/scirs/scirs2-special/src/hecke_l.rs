//! Hecke L-functions and Maass forms.
//!
//! Implements Hecke operators, modular forms (holomorphic and real-analytic),
//! and their associated L-functions.
//!
//! # References
//! - Cohen & Strömberg, "Modular Forms: A Classical and Computational Introduction"
//! - Bump, "Automorphic Forms and Representations"
//! - Iwaniec & Kowalski, "Analytic Number Theory"

use scirs2_core::numeric::Complex64;
use std::f64::consts::PI;

/// Errors arising from Hecke L-function computations.
#[derive(Debug, thiserror::Error)]
pub enum HeckeLError {
    /// The modular weight is invalid for the requested operation.
    #[error("Invalid weight: {0}")]
    InvalidWeight(i32),
    /// The series or algorithm failed to converge at the given point.
    #[error("Convergence failure at s = {0}")]
    ConvergenceFailed(String),
    /// The requested level is not supported.
    #[error("Level {0} not supported")]
    UnsupportedLevel(u32),
    /// The coefficient array is too short for the operation.
    #[error("Insufficient coefficients: need at least {needed}, have {have}")]
    InsufficientCoefficients { needed: usize, have: usize },
}

// ─────────────────────────────────────────────────────────────────────────────
// Hecke eigenform
// ─────────────────────────────────────────────────────────────────────────────

/// A holomorphic Hecke eigenform of weight `k` and level `N`.
///
/// The Fourier expansion is
/// `f(z) = sum_{n=1}^{inf} a(n) q^n`
/// where `q = e^{2 pi i z}`.
///
/// For a normalised Hecke eigenform, `a(1) = 1` and the Hecke operators
/// satisfy `T(p) f = a(p) f`.
#[derive(Debug, Clone)]
pub struct HeckeEigenform {
    /// Fourier coefficients: `coefficients[0]` = a(1), `coefficients[1]` = a(2), …
    pub coefficients: Vec<Complex64>,
    /// Modular weight k (positive even integer for classical forms).
    pub weight: i32,
    /// Level N (subgroup Gamma_0(N)).
    pub level: u32,
    /// Whether this is a cusp form (vanishes at cusps).
    pub is_cusp_form: bool,
}

impl HeckeEigenform {
    /// Create a Hecke eigenform from its Fourier coefficients.
    ///
    /// Requires `weight > 0` and at least one coefficient.
    pub fn new(
        coefficients: Vec<Complex64>,
        weight: i32,
        level: u32,
        is_cusp_form: bool,
    ) -> Result<Self, HeckeLError> {
        if weight <= 0 {
            return Err(HeckeLError::InvalidWeight(weight));
        }
        if coefficients.is_empty() {
            return Err(HeckeLError::InsufficientCoefficients { needed: 1, have: 0 });
        }
        Ok(HeckeEigenform {
            coefficients,
            weight,
            level,
            is_cusp_form,
        })
    }

    /// Return a(n) = `coefficients[n-1]` if available.
    ///
    /// Indices are 1-based (a(1), a(2), …).
    pub fn a(&self, n: usize) -> Option<Complex64> {
        if n == 0 {
            return None;
        }
        self.coefficients.get(n - 1).copied()
    }

    /// Return the Hecke eigenvalue lambda(n) = a(n) / a(1).
    ///
    /// Returns `None` when n is out of range or a(1) = 0.
    pub fn hecke_eigenvalue(&self, n: usize) -> Option<Complex64> {
        let a1 = self.a(1)?;
        if a1.norm() < f64::EPSILON {
            return None;
        }
        let an = self.a(n)?;
        Some(an / a1)
    }

    /// Evaluate the partial sum of the L-function.
    ///
    /// `L(s, f) = sum_{n=1}^{n_terms} a(n) / n^s`
    ///
    /// This converges for `Re(s) > (k+1)/2` (Hecke bound for cusp forms).
    pub fn l_function_partial(&self, s: Complex64, n_terms: usize) -> Complex64 {
        let count = n_terms.min(self.coefficients.len());
        (1..=count)
            .map(|n| {
                let n_c = Complex64::new(n as f64, 0.0);
                self.coefficients[n - 1] * n_c.powc(-s)
            })
            .fold(Complex64::new(0.0, 0.0), |acc, x| acc + x)
    }

    /// Evaluate the *completed* L-function using the functional equation.
    ///
    /// The completed L-function is
    /// `Lambda(s, f) = (sqrt(N) / (2*pi))^s * Gamma(s) * L(s, f)`
    ///
    /// For a cusp form of weight k, the functional equation reads:
    /// `Lambda(s, f) = epsilon * Lambda(k - s, f_bar)`
    /// where `|epsilon| = 1` (the root number).
    ///
    /// This method computes the partial-sum approximation of `Lambda(s, f)`.
    pub fn completed_l_function(&self, s: Complex64, n_terms: usize) -> Complex64 {
        let l_val = self.l_function_partial(s, n_terms);
        let n_f = Complex64::new(self.level as f64, 0.0);
        let two_pi = Complex64::new(2.0 * PI, 0.0);
        // Gamma(s) approximation via Stirling for Re(s) large; use direct f64
        // gamma for real s, otherwise absorb into the L-value.
        let gamma_factor = {
            let re_s = s.re;
            if re_s > 0.5 {
                // Stirling approximation for the gamma factor amplitude:
                let g: f64 = crate::gamma::gamma(re_s);
                Complex64::new(g, 0.0)
            } else {
                Complex64::new(1.0, 0.0)
            }
        };
        let scale = (n_f.sqrt() / two_pi).powc(s);
        scale * gamma_factor * l_val
    }

    /// Central value `L(k/2, f)` for integer weight k.
    pub fn central_value(&self, n_terms: usize) -> Complex64 {
        let s = Complex64::new(self.weight as f64 / 2.0, 0.0);
        self.l_function_partial(s, n_terms)
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Ramanujan tau function
// ─────────────────────────────────────────────────────────────────────────────

/// Look-up table for `tau(1)` through `tau(22)`.
///
/// These are the Fourier coefficients of Ramanujan's Delta function
/// `Delta(z) = q * prod_{n>=1} (1-q^n)^24` (weight 12, level 1 cusp form).
const TAU_TABLE: [i64; 22] = [
    1,         // n=1
    -24,       // n=2
    252,       // n=3
    -1472,     // n=4
    4830,      // n=5
    -6048,     // n=6
    -16744,    // n=7
    84480,     // n=8
    -113643,   // n=9
    -115920,   // n=10
    534612,    // n=11
    -370944,   // n=12
    -577738,   // n=13
    401856,    // n=14  (= tau(2)*tau(7))
    1217160,   // n=15  (= tau(3)*tau(5))
    987136,    // n=16
    -6905934,  // n=17
    2727432,   // n=18
    10661420,  // n=19
    -7109760,  // n=20  (= tau(4)*tau(5))
    -4219488,  // n=21  (= tau(3)*tau(7))
    -12830688, // n=22  (= tau(2)*tau(11))
];

/// Ramanujan tau function `tau(n)`.
///
/// These are the Fourier coefficients of the cusp form
/// `Delta = q prod_{n=1}^{inf} (1 - q^n)^24`.
///
/// Algorithm:
/// - For n = 1..22: lookup table (exact).
/// - For prime n > 22: q-expansion of Delta up to index n (O(n^2)).
/// - For composite or prime-power n > 22: Hecke multiplicativity
///   `tau(mn) = tau(m)*tau(n)` for gcd(m,n)=1, combined with the
///   prime-power recurrence `tau(p^{k+1}) = tau(p)*tau(p^k) - p^{11}*tau(p^{k-1})`.
///
/// Returns 0 for n = 0.
pub fn ramanujan_tau(n: u64) -> i64 {
    match n {
        0 => 0,
        1..=22 => TAU_TABLE[(n - 1) as usize],
        _ => ramanujan_tau_general(n),
    }
}

/// General computation for n > 22.
fn ramanujan_tau_general(n: u64) -> i64 {
    // Factor n.
    let factors = factorize(n);

    // If n is prime (single factor with exponent 1), use the q-expansion.
    if factors.len() == 1 && factors[0].1 == 1 {
        return ramanujan_tau_q_expansion(n);
    }

    // Otherwise use multiplicativity + prime-power recurrence.
    let mut result: i64 = 1;
    for (p, k) in factors {
        let tau_pk = tau_prime_power(p, k);
        result = result.saturating_mul(tau_pk);
    }
    result
}

/// Compute tau(n) via the q-expansion product formula.
///
/// `Delta = q * prod_{k=1}^{n} (1 - q^k)^24`
///
/// Maintains coefficient array `c[0..n]` where `c[i]` is the coefficient of
/// `q^i` in `prod_{k=1}^{n} (1 - q^k)^24`, then `tau(n) = c[n-1]` (the
/// q^{n-1} coefficient multiplied by the leading q).
///
/// Complexity: O(n^2) time, O(n) space.
fn ramanujan_tau_q_expansion(n: u64) -> i64 {
    let n_usize = n as usize;
    // c[i] = coefficient of q^i in prod_{k=1}^{n} (1-q^k)^24
    // Starts as c[0] = 1, c[i] = 0 for i > 0.
    let mut c = vec![0i128; n_usize]; // c[0..n-1] needed; Delta = q * product
    c[0] = 1;

    for k in 1..n_usize {
        // Multiply by (1 - q^k)^24 iteratively (24 passes of (1 - q^k)).
        for _ in 0..24_u8 {
            // Multiply in-place: c[j] -= c[j - k] for j = n-1 down to k.
            for j in (k..n_usize).rev() {
                c[j] -= c[j - k];
            }
        }
    }

    // tau(n) = c[n-1] because Delta = q * (product), so the q^n term
    // in Delta is the q^{n-1} term in the product.
    c[n_usize - 1] as i64
}

/// Compute tau(p^k) using the recurrence:
/// `tau(p^0) = 1`, `tau(p^1) = tau(p)`,
/// `tau(p^{k+1}) = tau(p) * tau(p^k) - p^{11} * tau(p^{k-1})`.
///
/// For `tau(p)` where p > 22, uses the q-expansion (no infinite recursion because
/// we never call `ramanujan_tau` for primes larger than p from here).
fn tau_prime_power(p: u64, k: u32) -> i64 {
    if k == 0 {
        return 1;
    }
    // Get tau(p) for the prime p — use lookup or q-expansion, never recursion.
    let tau_p = if p <= 22 {
        TAU_TABLE[(p - 1) as usize]
    } else {
        ramanujan_tau_q_expansion(p)
    };
    if k == 1 {
        return tau_p;
    }
    let p11: i128 = (p as i128).pow(11);
    let mut prev = 1i128; // tau(p^0)
    let mut curr = tau_p as i128; // tau(p^1)
    for _ in 1..k {
        let next = (tau_p as i128) * curr - p11 * prev;
        prev = curr;
        curr = next;
    }
    curr.clamp(i64::MIN as i128, i64::MAX as i128) as i64
}

/// Trial-division factorization of n into (prime, exponent) pairs.
fn factorize(mut n: u64) -> Vec<(u64, u32)> {
    let mut factors = Vec::new();
    let mut d = 2u64;
    while d * d <= n {
        if n.is_multiple_of(d) {
            let mut exp = 0u32;
            while n.is_multiple_of(d) {
                exp += 1;
                n /= d;
            }
            factors.push((d, exp));
        }
        d += 1;
    }
    if n > 1 {
        factors.push((n, 1));
    }
    factors
}

// ─────────────────────────────────────────────────────────────────────────────
// Maass form
// ─────────────────────────────────────────────────────────────────────────────

/// A Maass form (real-analytic automorphic form).
///
/// A Maass form for `Gamma_0(N)` is a real-analytic function `phi` on the
/// upper half-plane satisfying:
/// - `phi(gamma*z) = phi(z)` for all `gamma` in `Gamma_0(N)`
/// - `Delta * phi = (1/4 + lambda^2) * phi`  (eigenfunction of the Laplacian)
/// - Moderate growth at cusps
///
/// The spectral parameter `lambda` encodes the eigenvalue via
/// `s(1-s) = 1/4 + lambda^2`, i.e. `s = 1/2 + i*lambda`.
///
/// The Fourier-Whittaker expansion is
/// `phi(x+iy) = sum_{n != 0} a(n) * W(ny, lambda) * e^{2 pi i n x}`
/// where `W(y, lambda) = sqrt(|y|) * K_{i*lambda}(2*pi*|y|)`.
#[derive(Debug, Clone)]
pub struct MaassForm {
    /// Spectral parameter lambda (eigenvalue of Laplacian is `1/4 + lambda^2`).
    pub spectral_parameter: f64,
    /// Fourier-Whittaker coefficients `a(n)` for n != 0.
    /// Index 0 corresponds to `a(1)`, index 1 to `a(2)`, etc.
    pub coefficients: Vec<f64>,
    /// Level N.
    pub level: u32,
}

impl MaassForm {
    /// Create a Maass form.
    pub fn new(spectral_parameter: f64, coefficients: Vec<f64>, level: u32) -> Self {
        MaassForm {
            spectral_parameter,
            coefficients,
            level,
        }
    }

    /// Return `a(n)` if available (1-based index).
    pub fn a(&self, n: usize) -> Option<f64> {
        if n == 0 {
            return None;
        }
        self.coefficients.get(n - 1).copied()
    }

    /// Partial sum of the L-function `L(s, phi)`.
    ///
    /// `L(s, phi) = sum_{n=1}^{n_terms} a(n) / n^s`
    pub fn l_function_partial(&self, s: f64, n_terms: usize) -> f64 {
        let count = n_terms.min(self.coefficients.len());
        (1..=count)
            .map(|n| self.coefficients[n - 1] / (n as f64).powf(s))
            .sum()
    }

    /// Evaluate the Maass eigenfunction at a point `(x, y)` in the upper
    /// half-plane via the Fourier-Whittaker expansion.
    ///
    /// Uses the approximation `K_{i*lambda}(t) ≈ sqrt(pi/(2t)) * exp(-t)` for
    /// large `t`, which is accurate when `2*pi*n*y >> 1`.
    pub fn evaluate(&self, x: f64, y: f64, n_terms: usize) -> f64 {
        if y <= 0.0 {
            return 0.0;
        }
        let count = n_terms.min(self.coefficients.len());
        let mut sum = 0.0f64;
        for n in 1..=count {
            let a_n = self.coefficients[n - 1];
            let t = 2.0 * PI * n as f64 * y;
            // K_{i*lambda}(t) approximation for large t (Bessel K asymptotic):
            let k_approx = if t > 1.0 {
                (PI / (2.0 * t)).sqrt() * (-t).exp()
            } else {
                // Small-t: use K_0(t) ~ -ln(t/2) - 0.5772... as crude fallback
                -(t / 2.0).ln() - 0.5772156649
            };
            let whittaker_w = y.sqrt() * k_approx;
            let phase = 2.0 * PI * (n as f64) * x;
            sum += a_n * whittaker_w * phase.cos();
        }
        sum
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Convenience: theta-series L-function (Riemann zeta as test case)
// ─────────────────────────────────────────────────────────────────────────────

/// Partial sum of the Riemann zeta function `zeta(s) = sum_{n=1}^{n_terms} 1/n^s`.
///
/// Serves as a simple test that the L-function infrastructure works.
pub fn theta_l_function_partial(s: Complex64, n_terms: usize) -> Complex64 {
    (1..=n_terms)
        .map(|n| Complex64::new(n as f64, 0.0).powc(-s))
        .fold(Complex64::new(0.0, 0.0), |acc, x| acc + x)
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // Known tau values for n = 1..22 from Ramanujan (1916) and Lehmer tables.
    const KNOWN_TAU: [(u64, i64); 22] = [
        (1, 1),
        (2, -24),
        (3, 252),
        (4, -1472),
        (5, 4830),
        (6, -6048),
        (7, -16744),
        (8, 84480),
        (9, -113643),
        (10, -115920),
        (11, 534612),
        (12, -370944),
        (13, -577738),
        (14, 401856),
        (15, 1217160),
        (16, 987136),
        (17, -6905934),
        (18, 2727432),
        (19, 10661420),
        (20, -7109760),
        (21, -4219488),
        (22, -12830688),
    ];

    #[test]
    fn test_ramanujan_tau_known() {
        for (n, expected) in KNOWN_TAU {
            let got = ramanujan_tau(n);
            assert_eq!(
                got, expected,
                "tau({}) = {} but expected {}",
                n, got, expected
            );
        }
    }

    #[test]
    fn test_ramanujan_tau_multiplicativity() {
        // tau(mn) = tau(m)*tau(n) when gcd(m,n) = 1.
        // tau(2) * tau(3) should equal tau(6).
        let t2 = ramanujan_tau(2);
        let t3 = ramanujan_tau(3);
        let t6 = ramanujan_tau(6);
        assert_eq!(t2 * t3, t6, "tau(2)*tau(3) should equal tau(6)");

        // tau(2) * tau(5) = tau(10).
        let t5 = ramanujan_tau(5);
        let t10 = ramanujan_tau(10);
        assert_eq!(t2 * t5, t10, "tau(2)*tau(5) should equal tau(10)");
    }

    #[test]
    fn test_ramanujan_tau_prime_power_recurrence() {
        // tau(4) = tau(2)^2 - 2^11 * tau(1) = (-24)^2 - 2048 = 576 - 2048 = -1472.
        let t4 = ramanujan_tau(4);
        let t2 = ramanujan_tau(2);
        let expected = t2 * t2 - (2i64.pow(11)) * ramanujan_tau(1);
        assert_eq!(t4, expected, "tau(4) recurrence mismatch");
    }

    #[test]
    fn test_hecke_eigenform_create() {
        // Build the Delta function with 12 known coefficients.
        let coeffs: Vec<Complex64> = KNOWN_TAU[..12]
            .iter()
            .map(|&(_, t)| Complex64::new(t as f64, 0.0))
            .collect();
        let form = HeckeEigenform::new(coeffs, 12, 1, true).expect("valid eigenform");
        assert_eq!(form.weight, 12);
        assert!(form.is_cusp_form);
        // a(1) = 1.
        assert!(
            (form.a(1).expect("a(1) exists").re - 1.0).abs() < 1e-10,
            "a(1) should be 1"
        );
    }

    #[test]
    fn test_hecke_l_function_real() {
        // Use the Delta form and evaluate L(s, Delta) at s=12 (> (k+1)/2 = 13/2).
        let coeffs: Vec<Complex64> = KNOWN_TAU
            .iter()
            .map(|&(_, t)| Complex64::new(t as f64, 0.0))
            .collect();
        let form = HeckeEigenform::new(coeffs, 12, 1, true).expect("valid eigenform");
        let s = Complex64::new(12.0, 0.0);
        let val = form.l_function_partial(s, 22);
        // With 22 terms, L(12, Delta) should be nonzero and finite.
        assert!(val.re.is_finite(), "L(12, Delta) should be finite");
        // The first term dominates: a(1)/1^12 = 1, so val should be close to 1.
        assert!(
            (val.re - 1.0).abs() < 0.1,
            "L(12, Delta) with 22 terms should be near 1: got {}",
            val.re
        );
    }

    #[test]
    fn test_hecke_eigenvalue() {
        let coeffs: Vec<Complex64> = KNOWN_TAU[..5]
            .iter()
            .map(|&(_, t)| Complex64::new(t as f64, 0.0))
            .collect();
        let form = HeckeEigenform::new(coeffs, 12, 1, true).expect("valid eigenform");
        // Hecke eigenvalue for n=2 is a(2)/a(1) = -24.
        let ev2 = form.hecke_eigenvalue(2).expect("eigenvalue at 2");
        assert!(
            (ev2.re + 24.0).abs() < 1e-10,
            "eigenvalue at 2 should be -24, got {}",
            ev2.re
        );
    }

    #[test]
    fn test_maass_form_l_function() {
        // Trivial Maass form with all coefficients = 1 (like Riemann zeta).
        let coeffs: Vec<f64> = (1..=50).map(|_| 1.0).collect();
        let maass = MaassForm::new(9.5337, coeffs, 1);
        let val = maass.l_function_partial(2.0, 50);
        // Sum_{n=1}^{50} 1/n^2 ~ pi^2/6 ~ 1.6449...
        assert!(
            (val - 1.625).abs() < 0.05,
            "Maass L(2) partial sum should be near pi^2/6, got {}",
            val
        );
    }

    #[test]
    fn test_maass_form_evaluate() {
        let coeffs = vec![1.0; 20];
        let maass = MaassForm::new(9.5337, coeffs, 1);
        // Should not panic and give a finite value.
        let val = maass.evaluate(0.0, 1.0, 20);
        assert!(val.is_finite(), "evaluate should return finite value");
    }

    #[test]
    fn test_theta_l_function_is_zeta() {
        // At s=2, theta_l_function should give pi^2/6 ≈ 1.6449 with enough terms.
        let s = Complex64::new(2.0, 0.0);
        let val = theta_l_function_partial(s, 10000);
        let pi2_over_6 = PI * PI / 6.0;
        assert!(
            (val.re - pi2_over_6).abs() < 0.001,
            "theta L at s=2 should be pi^2/6, got {}",
            val.re
        );
    }
}
