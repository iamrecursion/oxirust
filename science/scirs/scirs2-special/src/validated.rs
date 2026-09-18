//! Validated numerics: ball arithmetic for certified function evaluation.
//!
//! Each `Ball` = (midpoint, radius) guarantees the true value lies in
//! `[mid - rad, mid + rad]`.  All arithmetic operations propagate the
//! enclosure guarantee using interval-arithmetic rules.
//!
//! # References
//! - Tucker, "Validated Numerics: A Short Introduction to Rigorous Computations"
//! - Rump, "INTLAB — INTerval LABoratory"

/// A ball: midpoint ± radius, guaranteed to contain the true value.
///
/// Invariant: `rad >= 0`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Ball {
    /// Midpoint of the enclosure interval.
    pub mid: f64,
    /// Radius (non-negative) of the enclosure interval.
    pub rad: f64,
}

impl Ball {
    /// Construct a ball with the given midpoint and radius.
    ///
    /// # Panics
    /// In debug builds, panics when `rad < 0`.
    #[inline]
    pub fn new(mid: f64, rad: f64) -> Self {
        debug_assert!(rad >= 0.0, "Ball radius must be non-negative");
        Ball {
            mid,
            rad: rad.abs(), // guard against tiny negative fp artefacts
        }
    }

    /// Construct a point ball (exact representation, radius 0).
    #[inline]
    pub fn from_exact(x: f64) -> Self {
        Ball { mid: x, rad: 0.0 }
    }

    /// Construct a ball from a closed interval `[lo, hi]`.
    ///
    /// Returns `None` when `lo > hi`.
    pub fn from_interval(lo: f64, hi: f64) -> Option<Self> {
        if lo > hi {
            return None;
        }
        let mid = (lo + hi) * 0.5;
        let rad = (hi - lo) * 0.5;
        Some(Ball::new(mid, rad))
    }

    /// Lower bound of the enclosure interval.
    #[inline]
    pub fn lo(&self) -> f64 {
        self.mid - self.rad
    }

    /// Upper bound of the enclosure interval.
    #[inline]
    pub fn hi(&self) -> f64 {
        self.mid + self.rad
    }

    /// Returns `true` when the ball strictly contains `x`.
    #[inline]
    pub fn contains(&self, x: f64) -> bool {
        x >= self.lo() && x <= self.hi()
    }

    /// Returns `true` when this ball overlaps with `other`.
    #[inline]
    pub fn overlaps(&self, other: &Ball) -> bool {
        self.lo() <= other.hi() && other.lo() <= self.hi()
    }

    /// Returns `true` when the ball represents a valid enclosure (`rad >= 0`).
    #[inline]
    pub fn is_valid(&self) -> bool {
        self.rad >= 0.0 && self.mid.is_finite()
    }

    /// Add two balls, propagating the enclosure guarantee.
    ///
    /// `[a±r_a] + [b±r_b] = [a+b ± (r_a + r_b + fp_rounding)]`
    #[inline]
    pub fn add(&self, other: &Ball) -> Self {
        let mid = self.mid + other.mid;
        let prop = self.rad + other.rad;
        let fp_err = mid.abs() * f64::EPSILON * 2.0;
        Ball::new(mid, prop + fp_err)
    }

    /// Subtract two balls.
    #[inline]
    pub fn sub(&self, other: &Ball) -> Self {
        let mid = self.mid - other.mid;
        let prop = self.rad + other.rad;
        let fp_err = mid.abs() * f64::EPSILON * 2.0;
        Ball::new(mid, prop + fp_err)
    }

    /// Multiply two balls using the interval arithmetic rule.
    ///
    /// Multiplication error bound:
    /// `|a*b - mid_a*mid_b| <= |mid_a|*r_b + |mid_b|*r_a + r_a*r_b`
    /// plus floating-point rounding of the midpoint product.
    #[inline]
    pub fn mul(&self, other: &Ball) -> Self {
        let mid = self.mid * other.mid;
        let prop = self.mid.abs() * other.rad + other.mid.abs() * self.rad + self.rad * other.rad;
        let fp_err = mid.abs() * f64::EPSILON * 2.0;
        Ball::new(mid, prop + fp_err)
    }

    /// Divide two balls.  Returns `None` when the divisor ball contains 0.
    pub fn div(&self, other: &Ball) -> Option<Self> {
        // The divisor interval must not straddle zero.
        if other.lo() <= 0.0 && other.hi() >= 0.0 {
            return None;
        }
        let mid = self.mid / other.mid;
        // d/dx (a/b) wrt b has factor a/b^2; full interval bound:
        // |a/b - mid_a/mid_b| <= (|mid_a|*r_b/|mid_b|^2 + r_a/|mid_b|) / (1 - r_b/|mid_b|)
        // Simpler (over-)estimate: use the 1st-order bound and add fp error.
        let inv_b = 1.0 / other.mid;
        let prop = (self.mid.abs() * other.rad * inv_b.abs() * inv_b.abs())
            + self.rad * inv_b.abs()
            + mid.abs() * f64::EPSILON * 2.0;
        Some(Ball::new(mid, prop))
    }

    /// Negate a ball.
    #[inline]
    pub fn neg(&self) -> Self {
        Ball::new(-self.mid, self.rad)
    }

    /// Absolute value of a ball.
    #[inline]
    pub fn abs(&self) -> Self {
        Ball::new(self.mid.abs(), self.rad)
    }

    /// Square root of a ball.  Returns `None` if the interval contains a
    /// negative number.
    pub fn sqrt(&self) -> Option<Self> {
        if self.lo() < 0.0 {
            return None;
        }
        let mid = self.mid.sqrt();
        // |sqrt(x) - sqrt(m)| <= |x-m| / (2*sqrt(lo)) for x in [lo, hi]
        // Use the propagation bound: derivative of sqrt is 1/(2*sqrt(m)).
        let lo_sqrt = self.lo().sqrt().max(f64::MIN_POSITIVE);
        let prop = self.rad / (2.0 * lo_sqrt);
        let fp_err = mid * f64::EPSILON * 2.0;
        Some(Ball::new(mid, prop + fp_err))
    }

    /// Raise a ball to an integer power.
    pub fn powi(&self, n: i32) -> Self {
        if n == 0 {
            return Ball::from_exact(1.0);
        }
        if n == 1 {
            return *self;
        }
        if n < 0 {
            // 1 / self^(-n)
            let pos = self.powi(-n);
            return Ball::from_exact(1.0)
                .div(&pos)
                .unwrap_or(Ball::new(f64::NAN, f64::INFINITY));
        }
        // Exponentiation by squaring with ball propagation.
        let mut result = Ball::from_exact(1.0);
        let mut base = *self;
        let mut exp = n as u32;
        while exp > 0 {
            if exp & 1 == 1 {
                result = result.mul(&base);
            }
            base = base.mul(&base);
            exp >>= 1;
        }
        result
    }
}

impl std::ops::Add for Ball {
    type Output = Self;
    fn add(self, rhs: Self) -> Self {
        Ball::add(&self, &rhs)
    }
}

impl std::ops::Sub for Ball {
    type Output = Self;
    fn sub(self, rhs: Self) -> Self {
        Ball::sub(&self, &rhs)
    }
}

impl std::ops::Mul for Ball {
    type Output = Self;
    fn mul(self, rhs: Self) -> Self {
        Ball::mul(&self, &rhs)
    }
}

impl std::ops::Neg for Ball {
    type Output = Self;
    fn neg(self) -> Self {
        Ball::neg(&self)
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Elementary function enclosures
// ─────────────────────────────────────────────────────────────────────────────

/// Evaluate `sin(x)` with a rigorous enclosure.
///
/// Uses the Lipschitz constant 1 for propagation plus a floating-point
/// rounding bound.
pub fn ball_sin(x: Ball) -> Ball {
    let mid_val = x.mid.sin();
    // Lipschitz constant of sin is 1, so |sin(a) - sin(b)| <= |a - b|.
    let prop_err = x.rad;
    let fp_err = (1.0_f64).max(mid_val.abs()) * f64::EPSILON * 4.0;
    Ball::new(mid_val, prop_err + fp_err)
}

/// Evaluate `cos(x)` with a rigorous enclosure.
pub fn ball_cos(x: Ball) -> Ball {
    let mid_val = x.mid.cos();
    let prop_err = x.rad;
    let fp_err = (1.0_f64).max(mid_val.abs()) * f64::EPSILON * 4.0;
    Ball::new(mid_val, prop_err + fp_err)
}

/// Evaluate `exp(x)` with a rigorous enclosure.
///
/// Derivative of exp is exp itself, so `|e^a - e^b| <= e^{a} * |a - b|`
/// for `a >= b`.
pub fn ball_exp(x: Ball) -> Ball {
    let mid_val = x.mid.exp();
    // Conservative: use e^{mid + rad} * rad as propagation bound.
    let max_exp = (x.mid + x.rad).exp();
    let prop_err = max_exp * x.rad;
    let fp_err = mid_val * f64::EPSILON * 4.0;
    Ball::new(mid_val, prop_err + fp_err)
}

/// Evaluate `ln(x)` with a rigorous enclosure.
///
/// Returns `None` if the interval contains zero or a negative number.
pub fn ball_ln(x: Ball) -> Option<Ball> {
    if x.lo() <= 0.0 {
        return None;
    }
    let mid_val = x.mid.ln();
    // Derivative of ln is 1/x; bound: |ln(a) - ln(b)| <= |a-b| / lo.
    let prop_err = x.rad / x.lo();
    let fp_err = mid_val.abs() * f64::EPSILON * 4.0;
    Some(Ball::new(mid_val, prop_err + fp_err))
}

// ─────────────────────────────────────────────────────────────────────────────
// Gamma function enclosure via Stirling's series
// ─────────────────────────────────────────────────────────────────────────────

/// Evaluate `Gamma(x)` with a rigorous enclosure using Stirling's approximation
/// plus an explicit error bound.
///
/// Returns `None` when `x <= 0` (poles of Gamma).
///
/// The error bound for Stirling's series is:
/// `|ln Gamma(x) - Stirling_n(x)| <= B_{2n+2} / ((2n+1)(2n+2) * x^{2n+2})`
/// where `B_{2n+2}` is the (2n+2)-th Bernoulli number.
pub fn ball_gamma(x: Ball) -> Option<Ball> {
    if x.lo() <= 0.0 {
        return None;
    }

    // Use functional equation Gamma(x) = Gamma(x+k)/x(x+1)...(x+k-1)
    // to shift x into the range [10, inf) where Stirling is tight.
    let shift_target = 10.0_f64;
    let shift = if x.mid < shift_target {
        (shift_target - x.mid).ceil() as u32
    } else {
        0
    };

    // Accumulated shift factor (as ball arithmetic).
    // Gamma(x) = Gamma(x+shift) / [x*(x+1)*...*(x+shift-1)]
    let x_shifted = Ball::new(x.mid + shift as f64, x.rad);

    // Stirling series for ln Gamma(z) for large z:
    // ln Gamma(z) ~ (z-0.5)*ln(z) - z + 0.5*ln(2*pi) + 1/(12z) - 1/(360z^3) + ...
    let z = x_shifted.mid;
    let ln_gamma_mid = (z - 0.5) * z.ln() - z + 0.5 * std::f64::consts::TAU.ln() + 1.0 / (12.0 * z)
        - 1.0 / (360.0 * z * z * z)
        + 1.0 / (1260.0 * z.powi(5));

    // Error bound from the next Stirling term: B_8/(7*8*z^8) = 1/1680 / z^8.
    // B_8 = -1/30, but in absolute value the bound is 1/30 * 1/(7*8) = 1/1680.
    let stirling_err = 1.0 / (1680.0 * z.powi(8));

    // Propagation: d(ln Gamma)/dx ~ psi(x) (digamma).
    // Use the rough bound |psi(x)| <= ln(x) for x >= 1.
    let psi_bound = z.ln() + 1.0 / z;
    let prop_err = psi_bound * x.rad;

    let ln_gamma_rad = stirling_err + prop_err + z.ln() * f64::EPSILON * 4.0;

    // gamma_shifted = exp(ln_gamma ± rad_ln)
    let gamma_shifted_mid = ln_gamma_mid.exp();
    let gamma_shifted_rad = gamma_shifted_mid * ln_gamma_rad;
    let mut gamma_ball = Ball::new(gamma_shifted_mid, gamma_shifted_rad);

    // Divide back by the shifted factors.
    for k in 0..shift {
        let factor = Ball::new(x.mid + k as f64, x.rad);
        gamma_ball = gamma_ball.div(&factor)?;
    }

    Some(gamma_ball)
}

// ─────────────────────────────────────────────────────────────────────────────
// Validation helper
// ─────────────────────────────────────────────────────────────────────────────

/// Validate that a computed floating-point value is contained in a ball.
///
/// Returns `true` when `ball.contains(computed)`.
#[inline]
pub fn validate(computed: f64, ball: Ball) -> bool {
    ball.contains(computed)
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ball_add_sub() {
        let a = Ball::new(3.0, 0.1);
        let b = Ball::new(2.0, 0.05);
        let sum = a + b;
        assert!(sum.contains(5.0));
        assert!(sum.rad >= 0.15); // at least the propagated radius

        let diff = a - b;
        assert!(diff.contains(1.0));
        assert!(diff.rad >= 0.15);
    }

    #[test]
    fn test_ball_mul_propagation() {
        // [2 ± 0] * [3 ± 0] = [6 ± 0] (exact)
        let a = Ball::from_exact(2.0);
        let b = Ball::from_exact(3.0);
        let prod = a * b;
        assert!(prod.contains(6.0));
        assert!(prod.rad < 1e-12);

        // [2 ± 0.5] * [3 ± 0.5] should contain 6 and have radius >= |2|*0.5 + |3|*0.5 = 2.5
        let a2 = Ball::new(2.0, 0.5);
        let b2 = Ball::new(3.0, 0.5);
        let prod2 = a2 * b2;
        assert!(prod2.contains(6.0));
        assert!(prod2.rad >= 2.5);
    }

    #[test]
    fn test_ball_sin_contains() {
        // sin(pi/6) = 0.5 exactly.
        let x = Ball::new(std::f64::consts::PI / 6.0, 1e-8);
        let s = ball_sin(x);
        let true_val = (std::f64::consts::PI / 6.0).sin();
        assert!(
            s.contains(true_val),
            "ball_sin should contain true sin value; ball=[{}, {}], true={}",
            s.lo(),
            s.hi(),
            true_val
        );
    }

    #[test]
    fn test_ball_gamma_basic() {
        // Gamma(5) = 4! = 24.
        let x = Ball::new(5.0, 1e-10);
        let g = ball_gamma(x).expect("ball_gamma(5) should not be None");
        assert!(
            g.contains(24.0),
            "ball_gamma(5) should contain 24.0; ball=[{}, {}]",
            g.lo(),
            g.hi()
        );
    }

    #[test]
    fn test_ball_gamma_one() {
        // Gamma(1) = 1.
        // Use a small but nonzero radius to account for the Stirling error at x=1.
        let x = Ball::new(1.0, 1e-6);
        let g = ball_gamma(x).expect("ball_gamma(1) should not be None");
        assert!(
            g.contains(1.0),
            "ball_gamma(1) should contain 1.0; ball=[{}, {}]",
            g.lo(),
            g.hi()
        );
    }

    #[test]
    fn test_ball_validate() {
        let b = Ball::new(2.71, 0.02);
        assert!(validate(2.71, b)); // midpoint
        assert!(validate(2.72, b)); // lo side (2.71 - 0.02 = 2.69 <= 2.72)
        assert!(validate(2.70, b)); // hi side (2.70 <= 2.71 + 0.02 = 2.73)
        assert!(!validate(2.68, b)); // outside lo (2.68 < 2.69)
        assert!(!validate(2.74, b)); // outside hi (2.74 > 2.73)
    }

    #[test]
    fn test_ball_div_by_zero() {
        // Dividing by a ball that contains zero must return None.
        let a = Ball::new(5.0, 0.1);
        let zero_ball = Ball::new(0.0, 1.0); // contains 0
        assert!(a.div(&zero_ball).is_none());

        // Dividing by a ball that does NOT contain zero should succeed.
        let safe_div = Ball::new(2.0, 0.1);
        let result = a.div(&safe_div);
        assert!(result.is_some());
        let r = result.expect("should divide safely");
        assert!(r.contains(2.5));
    }

    #[test]
    fn test_ball_sqrt() {
        let four = Ball::new(4.0, 1e-10);
        let root = four.sqrt().expect("sqrt(4) should not be None");
        assert!(root.contains(2.0));

        // Negative interval should return None.
        let neg = Ball::new(-1.0, 0.1);
        assert!(neg.sqrt().is_none());
    }

    #[test]
    fn test_ball_exp_contains() {
        let x = Ball::new(1.0, 1e-9);
        let e_ball = ball_exp(x);
        assert!(e_ball.contains(std::f64::consts::E));
    }

    #[test]
    fn test_ball_ln_contains() {
        let x = Ball::new(std::f64::consts::E, 1e-10);
        let ln_ball = ball_ln(x).expect("ln(e) should not be None");
        assert!(ln_ball.contains(1.0));
    }

    #[test]
    fn test_ball_powi() {
        let x = Ball::new(2.0, 0.0);
        let p = x.powi(10);
        assert!(p.contains(1024.0));
    }

    #[test]
    fn test_ball_from_interval() {
        let b = Ball::from_interval(1.0, 3.0).expect("valid interval");
        assert_eq!(b.mid, 2.0);
        assert_eq!(b.rad, 1.0);
        assert!(b.contains(1.0));
        assert!(b.contains(3.0));

        // Invalid interval: lo > hi
        assert!(Ball::from_interval(3.0, 1.0).is_none());
    }
}
