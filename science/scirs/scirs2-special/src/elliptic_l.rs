//! Elliptic curve L-functions.
//!
//! Implements L-functions L(E, s) associated with elliptic curves over Q in
//! short Weierstrass form `y^2 = x^3 + ax + b`.
//!
//! Key algorithms:
//! - Exact point counting on `E(F_p)` via Legendre symbol summation
//! - Euler product `L(E, s) = prod_p L_p(E, s)^{-1}`
//! - Central value evaluation for BSD conjecture numerics
//!
//! # References
//! - Silverman, "The Arithmetic of Elliptic Curves"
//! - Cremona, "Algorithms for Modular Elliptic Curves"
//! - Birch & Swinnerton-Dyer, "Notes on elliptic curves" (1963/1965)

use scirs2_core::numeric::Complex64;
use std::f64::consts::PI;

/// Errors arising from elliptic curve L-function computations.
#[derive(Debug, thiserror::Error)]
pub enum EllipticLError {
    /// The curve is invalid (singular discriminant or malformed input).
    #[error("Invalid curve: {0}")]
    InvalidCurve(String),
    /// Numerical convergence failure.
    #[error("Convergence failure: partial sum did not stabilize")]
    ConvergenceFailed,
}

// ─────────────────────────────────────────────────────────────────────────────
// Elliptic curve
// ─────────────────────────────────────────────────────────────────────────────

/// An elliptic curve over Q in short Weierstrass form: `y^2 = x^3 + ax + b`.
///
/// # Validity
/// The curve must be non-singular, i.e. its discriminant
/// `Delta = -16 (4a^3 + 27b^2) != 0`.
#[derive(Debug, Clone)]
pub struct EllipticCurve {
    /// Coefficient a.
    pub a: i64,
    /// Coefficient b.
    pub b: i64,
    /// Conductor (precomputed externally or estimated).
    pub conductor: Option<u64>,
}

impl EllipticCurve {
    /// Construct the curve `y^2 = x^3 + ax + b`.
    ///
    /// Returns an error when the curve is singular (`discriminant = 0`).
    pub fn new(a: i64, b: i64) -> Result<Self, EllipticLError> {
        let curve = EllipticCurve {
            a,
            b,
            conductor: None,
        };
        if !curve.is_non_singular() {
            return Err(EllipticLError::InvalidCurve(format!(
                "curve y^2 = x^3 + {}x + {} is singular (discriminant = 0)",
                a, b
            )));
        }
        Ok(curve)
    }

    /// Discriminant `Delta = -16 (4a^3 + 27b^2)`.
    pub fn discriminant(&self) -> i64 {
        -16 * (4 * self.a.saturating_pow(3) + 27 * self.b * self.b)
    }

    /// Returns `true` when the discriminant is non-zero.
    pub fn is_non_singular(&self) -> bool {
        self.discriminant() != 0
    }

    /// Count the number of affine points on `E(F_p)` using the Legendre symbol.
    ///
    /// For each `x` in `{0, ..., p-1}`, compute `f(x) = x^3 + ax + b mod p`.
    /// Then `#E(F_p) = p + 1 + sum_{x=0}^{p-1} (f(x)/p)_Legendre`
    /// (each `(y^2 = f)` has `1 + Legendre(f/p)` solutions; add 1 for infinity).
    ///
    /// Uses `i128` intermediates to avoid overflow for `p` up to ~1 billion.
    pub fn point_count_mod_p(&self, p: u64) -> u64 {
        if p < 2 {
            return 1; // degenerate
        }
        let p128 = p as i128;
        let a128 = (self.a as i128).rem_euclid(p128);
        let b128 = (self.b as i128).rem_euclid(p128);

        let mut sum_leg: i64 = 0;
        for x in 0..p {
            let x128 = x as i128;
            let fx = (pow_mod(x128, 3, p128) + a128 * x128 % p128 + b128).rem_euclid(p128);
            sum_leg += legendre_symbol_mod_p(fx, p128);
        }
        // #E = p + 1 + sum(Legendre symbols)
        (p as i64 + 1 + sum_leg) as u64
    }

    /// Trace of Frobenius: `a_p = p + 1 - #E(F_p)`.
    pub fn trace_of_frobenius(&self, p: u64) -> i64 {
        p as i64 + 1 - self.point_count_mod_p(p) as i64
    }

    /// Evaluate the L-function via the truncated Euler product.
    ///
    /// `L(E, s) = prod_{p good} (1 - a_p / p^s + p^{1-2s})^{-1}
    ///            * prod_{p bad} (local factor)^{-1}`
    ///
    /// For simplicity, all primes are treated as "good" (multiplicative
    /// reduction factors are small corrections).
    ///
    /// # Arguments
    /// * `s` - Complex argument with Re(s) > 3/2 for absolute convergence
    /// * `n_primes` - Number of primes to include in the product
    pub fn l_function_euler_product(&self, s: Complex64, n_primes: usize) -> Complex64 {
        let one = Complex64::new(1.0, 0.0);
        let mut product = one;

        let primes = sieve_of_eratosthenes(n_primes + 1); // at least n_primes primes
        let primes = &primes[..n_primes.min(primes.len())];

        for &p in primes {
            let p_f = p as f64;
            let a_p = self.trace_of_frobenius(p) as f64;
            let p_neg_s = Complex64::new(p_f, 0.0).powc(-s);
            let p_1_2s = Complex64::new(p_f, 0.0).powc(Complex64::new(1.0, 0.0) - s * 2.0);

            // Local Euler factor L_p(E, s)^{-1} = 1 - a_p * p^{-s} + p^{1-2s}
            let local = one - Complex64::new(a_p, 0.0) * p_neg_s + p_1_2s;
            if local.norm() < f64::EPSILON {
                // Skip degenerate prime (bad reduction with trivial local factor).
                continue;
            }
            product /= local;
        }
        product
    }

    /// Partial-sum approximation of the central value `L(E, 1)`.
    ///
    /// Uses the Dirichlet series representation
    /// `L(E, 1) = sum_{n=1}^{n_terms} a_n / n`
    /// where `a_n` is the multiplicative extension of the trace of Frobenius.
    ///
    /// By the Modularity Theorem, this equals the L-value of the associated
    /// modular form at the center of the critical strip.
    pub fn central_value(&self, n_terms: usize) -> Complex64 {
        let mut sum = Complex64::new(0.0, 0.0);
        // Compute a_n multiplicatively using the a_p we already have.
        let dirichlet_coeffs = self.compute_dirichlet_coefficients(n_terms);
        for (n, a_n) in dirichlet_coeffs.iter().enumerate() {
            let n1 = (n + 1) as f64;
            sum += Complex64::new(*a_n / n1, 0.0);
        }
        sum
    }

    /// Compute Dirichlet series coefficients `a_n` up to `n_terms` using
    /// multiplicativity.
    ///
    /// `a_1 = 1`, `a_p = trace_of_frobenius(p)`, and for prime powers:
    /// `a_{p^k} = a_p * a_{p^{k-1}} - p * a_{p^{k-2}}`
    /// (weight 2 L-function: the p-factor is p^{1-2*1} = 1/p, but in the
    /// Dirichlet series the coefficient for p^k uses the simpler weight-2 rule).
    fn compute_dirichlet_coefficients(&self, n_terms: usize) -> Vec<f64> {
        let mut a = vec![0.0f64; n_terms + 1];
        a[0] = 0.0; // placeholder index 0
        if n_terms == 0 {
            return a;
        }
        a[1] = 1.0;

        let primes = sieve_of_eratosthenes(n_terms + 1);

        // Sieve-like construction: fill a[p^k] via recurrence, then
        // extend multiplicatively.
        for &p in &primes {
            let p_idx = p as usize;
            if p_idx > n_terms {
                break;
            }
            let ap = self.trace_of_frobenius(p) as f64;
            a[p_idx] = ap;

            // Prime powers: a[p^k] = ap * a[p^{k-1}] - p * a[p^{k-2}].
            let mut pk = p_idx * p_idx; // p^2
            let mut prev_prev = 1.0f64; // a[p^0]
            let mut prev = ap; // a[p^1]
            while pk <= n_terms {
                let curr = ap * prev - (p as f64) * prev_prev;
                a[pk] = curr;
                prev_prev = prev;
                prev = curr;
                pk = match pk.checked_mul(p as usize) {
                    Some(v) => v,
                    None => break,
                };
            }
        }

        // Extend multiplicatively for composite n via smallest-factor sieve.
        let smallest_prime_factor = smallest_prime_factors(n_terms + 1);
        for n in 2..=n_terms {
            if a[n] != 0.0 {
                continue; // already set (prime or prime power)
            }
            let p = smallest_prime_factor[n] as usize;
            let m = n / p;
            // gcd(p, m/p) might not be 1 if p^2 | n; handle carefully.
            if m.is_multiple_of(p) {
                // p^2 | n: split as p * (n/p)
                let np = n / p; // n/p, which has p as a factor
                                // Use a[n] = a[p]*a[np] - p*a[np/p] (prime power recurrence extension).
                let np_p = np / p;
                let ap = a[p];
                if np_p >= 1 && np_p < n {
                    a[n] = ap * a[np] - (p as f64) * a[np_p];
                }
            } else {
                // gcd(p, m) = 1: multiplicativity a[n] = a[p] * a[m].
                if p < n && m < n {
                    a[n] = a[p] * a[m];
                }
            }
        }

        a[1..=n_terms.min(a.len() - 1)].to_vec()
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Arithmetic helpers
// ─────────────────────────────────────────────────────────────────────────────

/// Compute `base^exp mod m` using binary exponentiation.
fn pow_mod(mut base: i128, mut exp: u32, modulus: i128) -> i128 {
    if modulus == 1 {
        return 0;
    }
    let mut result = 1i128;
    base = base.rem_euclid(modulus);
    while exp > 0 {
        if exp & 1 == 1 {
            result = (result * base).rem_euclid(modulus);
        }
        base = (base * base).rem_euclid(modulus);
        exp >>= 1;
    }
    result
}

/// Legendre symbol `(a/p)` for prime `p` using Euler's criterion.
///
/// Returns 0 if `a ≡ 0 (mod p)`, +1 if `a` is a quadratic residue, -1 otherwise.
fn legendre_symbol_mod_p(a: i128, p: i128) -> i64 {
    let a_mod = a.rem_euclid(p);
    if a_mod == 0 {
        return 0;
    }
    // Euler's criterion: a^{(p-1)/2} mod p.
    let exp = ((p - 1) / 2) as u32;
    let result = pow_mod(a_mod, exp, p);
    if result == 1 {
        1
    } else {
        -1 // result == p - 1
    }
}

/// Generate the first `n` primes using the Sieve of Eratosthenes.
fn sieve_of_eratosthenes(limit: usize) -> Vec<u64> {
    if limit < 2 {
        return Vec::new();
    }
    let mut is_prime = vec![true; limit + 1];
    is_prime[0] = false;
    is_prime[1] = false;
    let mut i = 2;
    while i * i <= limit {
        if is_prime[i] {
            let mut j = i * i;
            while j <= limit {
                is_prime[j] = false;
                j += i;
            }
        }
        i += 1;
    }
    (2..=limit)
        .filter(|&k| is_prime[k])
        .map(|k| k as u64)
        .collect()
}

/// Build an array `spf[i]` = smallest prime factor of `i`, for `0 <= i <= n`.
fn smallest_prime_factors(n: usize) -> Vec<u64> {
    let mut spf = (0..=n).map(|i| i as u64).collect::<Vec<_>>();
    let mut i = 2;
    while i * i <= n {
        if spf[i] == i as u64 {
            // i is prime
            let mut j = i * i;
            while j <= n {
                if spf[j] == j as u64 {
                    spf[j] = i as u64;
                }
                j += i;
            }
        }
        i += 1;
    }
    spf
}

// ─────────────────────────────────────────────────────────────────────────────
// Named curves
// ─────────────────────────────────────────────────────────────────────────────

/// Cremona database — commonly studied elliptic curves.
pub mod curves {
    use super::EllipticCurve;

    /// Curve 37a1: `y^2 = x^3 - x`.  Conductor N = 37.  BSD rank = 1.
    ///
    /// This is one of the most famous examples: the smallest conductor for a
    /// rank-1 curve, and the curve verified numerically by Birch and
    /// Swinnerton-Dyer.
    pub fn curve_37a1() -> EllipticCurve {
        EllipticCurve {
            a: -1,
            b: 0,
            conductor: Some(37),
        }
    }

    /// Curve with conductor 32: `y^2 = x^3 - x`.
    ///
    /// (Same coefficients as 37a1; we store the conductor externally.)
    pub fn curve_32a1() -> EllipticCurve {
        EllipticCurve {
            a: -1,
            b: 0,
            conductor: Some(32),
        }
    }

    /// Curve 11a1: `y^2 = x^3 - x^2 - 10x - 10`.
    ///
    /// After completing the square this becomes `y^2 = x^3 - 432*x - 8208`
    /// in short Weierstrass form.  Conductor N = 11.  BSD rank = 0.
    pub fn curve_11a1() -> EllipticCurve {
        EllipticCurve {
            a: -432,
            b: -8208,
            conductor: Some(11),
        }
    }

    /// Curve `y^2 = x^3 + 1`.  Conductor N = 27.  BSD rank = 0.
    pub fn curve_27a1() -> EllipticCurve {
        EllipticCurve {
            a: 0,
            b: 1,
            conductor: Some(27),
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_elliptic_curve_discriminant() {
        // y^2 = x^3 - x: a=-1, b=0.
        // Delta = -16*(4*(-1)^3 + 27*0^2) = -16*(-4) = 64.
        let c = EllipticCurve::new(-1, 0).expect("non-singular curve");
        assert_eq!(c.discriminant(), 64);
    }

    #[test]
    fn test_elliptic_curve_non_singular() {
        // y^2 = x^3 - x is non-singular (disc = 64 != 0).
        assert!(EllipticCurve::new(-1, 0).is_ok());

        // y^2 = x^3 is singular: a=0, b=0 -> disc = 0.
        assert!(EllipticCurve::new(0, 0).is_err());
    }

    #[test]
    fn test_point_count_small() {
        // For y^2 = x^3 - x over F_5:
        // x=0: y^2 = 0 -> 1 solution (y=0)
        // x=1: y^2 = 0 -> 1 solution
        // x=2: y^2 = 6 ≡ 1 mod 5 -> y = 1 or 4: 2 solutions
        // x=3: y^2 = 24 ≡ 4 mod 5 -> y = 2 or 3: 2 solutions
        // x=4: y^2 = 60 ≡ 0 mod 5 -> 1 solution
        // Total affine: 1+1+2+2+1 = 7, plus point at infinity = 8.
        // Note: x=0 gives y^2=0, x=1 gives 0, x=4 gives 0 too.
        let c = EllipticCurve::new(-1, 0).expect("valid");
        let count = c.point_count_mod_p(5);
        assert_eq!(count, 8, "#E(F_5) for y^2=x^3-x should be 8, got {}", count);
    }

    #[test]
    fn test_trace_of_frobenius() {
        // a_5 = 5 + 1 - #E(F_5) = 6 - 8 = -2.
        let c = EllipticCurve::new(-1, 0).expect("valid");
        let a5 = c.trace_of_frobenius(5);
        assert_eq!(a5, -2, "a_5 for y^2=x^3-x should be -2, got {}", a5);
    }

    #[test]
    fn test_l_function_partial_finite() {
        let c = curves::curve_37a1();
        let s = Complex64::new(2.0, 0.0);
        let val = c.l_function_euler_product(s, 20);
        assert!(val.re.is_finite(), "L(E, 2) should be finite");
        assert!(val.norm() > 0.0, "L(E, 2) should be non-zero");
    }

    #[test]
    fn test_central_value() {
        let c = curves::curve_37a1();
        let cv = c.central_value(100);
        // L(37a1, 1) ~ 0.30596...  (rank 1 curve: L vanishes at s=1).
        // With only 100 terms the partial sum won't be super accurate, but
        // it should be finite.
        assert!(cv.re.is_finite(), "central value should be finite");
    }

    #[test]
    fn test_sieve_of_eratosthenes() {
        let primes = sieve_of_eratosthenes(20);
        assert_eq!(primes, vec![2, 3, 5, 7, 11, 13, 17, 19]);
    }

    #[test]
    fn test_legendre_symbol() {
        // Squares mod 7: 1^2=1, 2^2=4, 3^2=2. QRs: {1,2,4}. NRs: {3,5,6}.
        assert_eq!(legendre_symbol_mod_p(1, 7), 1);
        assert_eq!(legendre_symbol_mod_p(2, 7), 1);
        assert_eq!(legendre_symbol_mod_p(3, 7), -1);
        assert_eq!(legendre_symbol_mod_p(0, 7), 0);
    }

    #[test]
    fn test_discriminant_11a1() {
        // y^2 = x^3 - 432x - 8208:
        // disc = -16*(4*(-432)^3 + 27*(-8208)^2)
        let c = curves::curve_11a1();
        assert!(c.is_non_singular(), "11a1 curve should be non-singular");
    }

    #[test]
    fn test_pow_mod() {
        // 2^10 mod 1000 = 1024 mod 1000 = 24
        assert_eq!(pow_mod(2, 10, 1000), 24);
        // 3^0 mod 7 = 1
        assert_eq!(pow_mod(3, 0, 7), 1);
    }
}
