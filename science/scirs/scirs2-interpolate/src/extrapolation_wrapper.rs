//! Generic extrapolation wrapper for 1-D interpolators.
//!
//! [`ExtrapolatingInterpolator`] wraps any `Fn(f64) -> f64` closure (or any
//! type that implements [`Interpolate1D`]) and handles out-of-domain queries by
//! applying a configurable [`ExtrapolationMode`].
//!
//! # Examples
//!
//! ```rust
//! use scirs2_interpolate::extrapolation_wrapper::{
//!     ExtrapolatingInterpolator, ExtrapolationMode,
//! };
//!
//! // A simple piecewise-linear interpolant on [0, 1].
//! let inner = |x: f64| x * x;          // any Fn(f64) -> f64
//! let interp = ExtrapolatingInterpolator::new(inner, 0.0, 1.0,
//!     ExtrapolationMode::Fill(f64::NAN));
//!
//! assert!((interp.eval(0.5).unwrap() - 0.25).abs() < 1e-12);
//! assert!(interp.eval(-0.1).unwrap().is_nan());
//! ```
//!
//! ## Periodic wrapping
//!
//! ```rust
//! use std::f64::consts::PI;
//! use scirs2_interpolate::extrapolation_wrapper::{
//!     ExtrapolatingInterpolator, ExtrapolationMode,
//! };
//!
//! let inner = |x: f64| x.sin();
//! let interp = ExtrapolatingInterpolator::new(inner, 0.0, 2.0 * PI,
//!     ExtrapolationMode::Periodic);
//!
//! // x = 3π is equivalent to x = π (period = 2π)
//! let y1 = interp.eval(PI).unwrap();
//! let y2 = interp.eval(3.0 * PI).unwrap();
//! assert!((y1 - y2).abs() < 1e-10);
//! ```

use crate::error::{InterpolateError, InterpolateResult};

// ─────────────────────────────────────────────────────────────────────────────
// ExtrapolationMode
// ─────────────────────────────────────────────────────────────────────────────

/// How to handle queries outside the interpolation domain `[x_min, x_max]`.
#[derive(Debug, Clone, PartialEq)]
pub enum ExtrapolationMode {
    /// Clamp `x` to the nearest boundary and evaluate the interpolant there.
    Nearest,
    /// Linear extrapolation using a finite-difference gradient at the boundary.
    ///
    /// The gradient is estimated with step `h` (defaults to
    /// `(x_max - x_min) * 1e-5`).
    Linear,
    /// Polynomial extrapolation of degree `deg` using `deg+1` equally-spaced
    /// points near the boundary.
    Polynomial(usize),
    /// Reflect `x` about the boundary and evaluate at the reflected point.
    ///
    /// For a lower boundary at `x_min`: reflected point is `2 x_min - x`.
    /// For an upper boundary at `x_max`: reflected point is `2 x_max - x`.
    Reflect,
    /// Wrap `x` periodically so the period equals `x_max - x_min`.
    Periodic,
    /// Return `v` for every out-of-domain query.
    Fill(f64),
    /// Return [`InterpolateError::OutOfBounds`] for out-of-domain queries.
    Error,
}

// ─────────────────────────────────────────────────────────────────────────────
// Trait for 1-D interpolants
// ─────────────────────────────────────────────────────────────────────────────

/// Minimal trait for 1-D interpolants that can be wrapped with extrapolation
/// handling.  Implemented automatically for `Fn(f64) -> f64`.
pub trait Interpolate1D {
    /// Evaluate the interpolant at `x`.  May assume `x` is within the
    /// training domain.
    fn interpolate(&self, x: f64) -> f64;
}

impl<F: Fn(f64) -> f64> Interpolate1D for F {
    fn interpolate(&self, x: f64) -> f64 {
        (self)(x)
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// ExtrapolatingInterpolator
// ─────────────────────────────────────────────────────────────────────────────

/// A 1-D interpolant wrapped with configurable out-of-domain behaviour.
///
/// The same mode applies to both the lower (`x < x_min`) and upper
/// (`x > x_max`) boundaries.  Use [`ExtrapolatingInterpolatorAsymmetric`] if
/// you need different modes per boundary.
pub struct ExtrapolatingInterpolator<I: Interpolate1D> {
    inner: I,
    x_min: f64,
    x_max: f64,
    mode: ExtrapolationMode,
}

impl<I: Interpolate1D> ExtrapolatingInterpolator<I> {
    /// Create a new wrapped interpolant.
    ///
    /// # Arguments
    /// * `inner`  – The underlying interpolant.
    /// * `x_min`  – Lower bound of the valid domain.
    /// * `x_max`  – Upper bound of the valid domain (must be > `x_min`).
    /// * `mode`   – Extrapolation strategy.
    pub fn new(inner: I, x_min: f64, x_max: f64, mode: ExtrapolationMode) -> Self {
        assert!(
            x_max > x_min,
            "x_max ({x_max}) must be strictly greater than x_min ({x_min})"
        );
        Self {
            inner,
            x_min,
            x_max,
            mode,
        }
    }

    /// Evaluate the (possibly extrapolated) interpolant at `x`.
    pub fn eval(&self, x: f64) -> InterpolateResult<f64> {
        if x >= self.x_min && x <= self.x_max {
            return Ok(self.inner.interpolate(x));
        }
        let period = self.x_max - self.x_min;
        match &self.mode {
            ExtrapolationMode::Nearest => {
                let clamped = x.clamp(self.x_min, self.x_max);
                Ok(self.inner.interpolate(clamped))
            }

            ExtrapolationMode::Linear => {
                let h = period * 1e-5;
                if x < self.x_min {
                    // Gradient at left boundary by forward difference.
                    let f0 = self.inner.interpolate(self.x_min);
                    let f1 = self.inner.interpolate(self.x_min + h);
                    let slope = (f1 - f0) / h;
                    Ok(f0 + slope * (x - self.x_min))
                } else {
                    // Gradient at right boundary by backward difference.
                    let f0 = self.inner.interpolate(self.x_max);
                    let f1 = self.inner.interpolate(self.x_max - h);
                    let slope = (f0 - f1) / h;
                    Ok(f0 + slope * (x - self.x_max))
                }
            }

            ExtrapolationMode::Polynomial(deg) => {
                let deg = *deg;
                self.poly_extrapolate(x, deg)
            }

            ExtrapolationMode::Reflect => {
                let mapped = if x < self.x_min {
                    2.0 * self.x_min - x
                } else {
                    2.0 * self.x_max - x
                };
                // The reflected point may still be outside domain; clamp to be safe.
                let clamped = mapped.clamp(self.x_min, self.x_max);
                Ok(self.inner.interpolate(clamped))
            }

            ExtrapolationMode::Periodic => {
                let wrapped = wrap_periodic(x, self.x_min, self.x_max);
                Ok(self.inner.interpolate(wrapped))
            }

            ExtrapolationMode::Fill(v) => Ok(*v),

            ExtrapolationMode::Error => Err(InterpolateError::OutOfBounds(format!(
                "x={x:.6} is outside domain [{:.6}, {:.6}]",
                self.x_min, self.x_max
            ))),
        }
    }

    /// Polynomial extrapolation of degree `deg` using `deg+1` boundary samples.
    ///
    /// Samples are taken in the interior near the violated boundary and the
    /// polynomial is evaluated at `x` via Lagrange interpolation.
    fn poly_extrapolate(&self, x: f64, deg: usize) -> InterpolateResult<f64> {
        let n = deg + 1;
        let h = period_step(self.x_min, self.x_max, n);
        // Choose `n` sample nodes near the boundary that was violated.
        let nodes: Vec<f64> = if x < self.x_min {
            (0..n).map(|k| self.x_min + k as f64 * h).collect()
        } else {
            (0..n)
                .map(|k| self.x_max - (n - 1 - k) as f64 * h)
                .collect()
        };
        let ys: Vec<f64> = nodes.iter().map(|&xi| self.inner.interpolate(xi)).collect();
        Ok(lagrange_eval(&nodes, &ys, x))
    }

    /// Domain lower bound.
    pub fn x_min(&self) -> f64 {
        self.x_min
    }

    /// Domain upper bound.
    pub fn x_max(&self) -> f64 {
        self.x_max
    }

    /// Reference to the extrapolation mode.
    pub fn mode(&self) -> &ExtrapolationMode {
        &self.mode
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Asymmetric wrapper (different modes per boundary)
// ─────────────────────────────────────────────────────────────────────────────

/// Like [`ExtrapolatingInterpolator`] but with independent extrapolation modes
/// for the lower and upper boundaries.
pub struct ExtrapolatingInterpolatorAsymmetric<I: Interpolate1D> {
    inner: I,
    x_min: f64,
    x_max: f64,
    lower_mode: ExtrapolationMode,
    upper_mode: ExtrapolationMode,
}

impl<I: Interpolate1D> ExtrapolatingInterpolatorAsymmetric<I> {
    /// Create a new asymmetric wrapper.
    pub fn new(
        inner: I,
        x_min: f64,
        x_max: f64,
        lower_mode: ExtrapolationMode,
        upper_mode: ExtrapolationMode,
    ) -> Self {
        assert!(x_max > x_min);
        Self {
            inner,
            x_min,
            x_max,
            lower_mode,
            upper_mode,
        }
    }

    /// Evaluate, applying the appropriate boundary mode.
    pub fn eval(&self, x: f64) -> InterpolateResult<f64> {
        if x >= self.x_min && x <= self.x_max {
            return Ok(self.inner.interpolate(x));
        }
        let mode = if x < self.x_min {
            &self.lower_mode
        } else {
            &self.upper_mode
        };
        // Delegate to the symmetric impl by constructing a temporary one.
        let tmp = ExtrapolatingInterpolator {
            inner: DummyInner(&self.inner),
            x_min: self.x_min,
            x_max: self.x_max,
            mode: mode.clone(),
        };
        tmp.eval(x)
    }
}

/// Helper wrapper so we can borrow the inner interpolant.
struct DummyInner<'a, I: Interpolate1D>(&'a I);

impl<'a, I: Interpolate1D> Interpolate1D for DummyInner<'a, I> {
    fn interpolate(&self, x: f64) -> f64 {
        self.0.interpolate(x)
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Internal helpers
// ─────────────────────────────────────────────────────────────────────────────

/// Wrap `x` into `[x_min, x_max)` with period = `x_max - x_min`.
fn wrap_periodic(x: f64, x_min: f64, x_max: f64) -> f64 {
    let period = x_max - x_min;
    let shifted = x - x_min;
    let wrapped = shifted - period * (shifted / period).floor();
    (x_min + wrapped).clamp(x_min, x_max)
}

/// Spacing for `n` equidistant nodes that fit inside `[x_min, x_max]`.
fn period_step(x_min: f64, x_max: f64, n: usize) -> f64 {
    if n <= 1 {
        0.0
    } else {
        (x_max - x_min) / (n - 1) as f64
    }
}

/// Evaluate the Lagrange interpolating polynomial defined by `(nodes, values)`
/// at `x`.
fn lagrange_eval(nodes: &[f64], values: &[f64], x: f64) -> f64 {
    let n = nodes.len();
    let mut result = 0.0_f64;
    for i in 0..n {
        let mut basis = 1.0_f64;
        for j in 0..n {
            if i != j {
                let denom = nodes[i] - nodes[j];
                if denom.abs() < 1e-300 {
                    continue;
                }
                basis *= (x - nodes[j]) / denom;
            }
        }
        result += values[i] * basis;
    }
    result
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use std::f64::consts::PI;

    // Helper: linear interpolant on [0, 1] with slope 1.
    fn linear_unit() -> impl Fn(f64) -> f64 {
        |x| x
    }

    // Helper: sin on [0, 2π].
    fn sin_interp() -> impl Fn(f64) -> f64 {
        |x: f64| x.sin()
    }

    #[test]
    fn test_extrapolation_nearest_below() {
        let interp =
            ExtrapolatingInterpolator::new(linear_unit(), 0.0, 1.0, ExtrapolationMode::Nearest);
        // x < 0 → clamp to 0.0 → f(0) = 0.
        let val = interp.eval(-0.5).expect("should succeed");
        assert!((val - 0.0).abs() < 1e-12, "nearest below: {val}");
    }

    #[test]
    fn test_extrapolation_nearest_above() {
        let interp =
            ExtrapolatingInterpolator::new(linear_unit(), 0.0, 1.0, ExtrapolationMode::Nearest);
        // x > 1 → clamp to 1.0 → f(1) = 1.
        let val = interp.eval(2.0).expect("should succeed");
        assert!((val - 1.0).abs() < 1e-12, "nearest above: {val}");
    }

    #[test]
    fn test_extrapolation_linear_below() {
        // f(x) = 2x + 3. Slope = 2.  Evaluate at x = -1.
        let inner = |x: f64| 2.0 * x + 3.0;
        let interp = ExtrapolatingInterpolator::new(inner, 0.0, 1.0, ExtrapolationMode::Linear);
        // linear extrapolation: f(0) + slope * (-1 - 0) = 3 + 2*(-1) = 1
        let val = interp.eval(-1.0).expect("linear below");
        assert!((val - 1.0).abs() < 1e-4, "linear extrap below: {val}");
    }

    #[test]
    fn test_extrapolation_linear_above() {
        let inner = |x: f64| 2.0 * x + 3.0;
        let interp = ExtrapolatingInterpolator::new(inner, 0.0, 1.0, ExtrapolationMode::Linear);
        // f(1) = 5, slope ≈ 2, at x=2: 5 + 2*(2-1) = 7
        let val = interp.eval(2.0).expect("linear above");
        assert!((val - 7.0).abs() < 1e-3, "linear extrap above: {val}");
    }

    #[test]
    fn test_extrapolation_fill() {
        let fill_val = -999.0;
        let interp = ExtrapolatingInterpolator::new(
            linear_unit(),
            0.0,
            1.0,
            ExtrapolationMode::Fill(fill_val),
        );
        assert_eq!(interp.eval(-5.0).unwrap(), fill_val);
        assert_eq!(interp.eval(5.0).unwrap(), fill_val);
    }

    #[test]
    fn test_extrapolation_fill_nan() {
        let interp = ExtrapolatingInterpolator::new(
            linear_unit(),
            0.0,
            1.0,
            ExtrapolationMode::Fill(f64::NAN),
        );
        assert!(interp.eval(-1.0).unwrap().is_nan());
    }

    #[test]
    fn test_extrapolation_error_mode() {
        let interp =
            ExtrapolatingInterpolator::new(linear_unit(), 0.0, 1.0, ExtrapolationMode::Error);
        assert!(interp.eval(-0.1).is_err(), "Should error below range");
        assert!(interp.eval(1.1).is_err(), "Should error above range");
        // Inside domain is fine.
        assert!(interp.eval(0.5).is_ok());
    }

    #[test]
    fn test_extrapolation_periodic() {
        // sin on [0, 2π] is periodic.
        let interp = ExtrapolatingInterpolator::new(
            sin_interp(),
            0.0,
            2.0 * PI,
            ExtrapolationMode::Periodic,
        );
        // sin(π) ≈ sin(3π) since period = 2π
        let y1 = interp.eval(PI).unwrap();
        let y2 = interp.eval(3.0 * PI).unwrap();
        assert!(
            (y1 - y2).abs() < 1e-10,
            "Periodic: sin(π)={y1} should equal sin(3π)={y2}"
        );
    }

    #[test]
    fn test_extrapolation_periodic_negative() {
        let interp = ExtrapolatingInterpolator::new(
            sin_interp(),
            0.0,
            2.0 * PI,
            ExtrapolationMode::Periodic,
        );
        // sin(x) = sin(x + 2π), so sin(-π/2) ≈ sin(3π/2)
        let y1 = interp.eval(-PI / 2.0).unwrap();
        let y2 = interp.eval(3.0 * PI / 2.0).unwrap();
        assert!((y1 - y2).abs() < 1e-10, "Periodic negative: {y1} vs {y2}");
    }

    #[test]
    fn test_extrapolation_reflect_below() {
        // f(x) = x² on [0, 1]. Reflect x=-0.3 → x=0.3.
        let interp =
            ExtrapolatingInterpolator::new(|x: f64| x * x, 0.0, 1.0, ExtrapolationMode::Reflect);
        let val = interp.eval(-0.3).unwrap();
        let expected = 0.3_f64 * 0.3;
        assert!(
            (val - expected).abs() < 1e-12,
            "reflect below: {val} vs {expected}"
        );
    }

    #[test]
    fn test_extrapolation_reflect_above() {
        let interp =
            ExtrapolatingInterpolator::new(|x: f64| x * x, 0.0, 1.0, ExtrapolationMode::Reflect);
        // Reflect x=1.4 → 2*1-1.4 = 0.6
        let val = interp.eval(1.4).unwrap();
        let expected = 0.6_f64 * 0.6;
        assert!(
            (val - expected).abs() < 1e-12,
            "reflect above: {val} vs {expected}"
        );
    }

    #[test]
    fn test_extrapolation_polynomial_linear_exact() {
        // f(x) = x + 1 (degree 1). Polynomial extrapolation with deg=1 should reproduce this.
        let inner = |x: f64| x + 1.0;
        let interp =
            ExtrapolatingInterpolator::new(inner, 0.0, 1.0, ExtrapolationMode::Polynomial(1));
        let val = interp.eval(-0.5).unwrap();
        let expected = -0.5 + 1.0; // = 0.5
        assert!(
            (val - expected).abs() < 1e-8,
            "poly extrap degree 1: {val} vs {expected}"
        );
    }

    #[test]
    fn test_extrapolation_polynomial_quadratic() {
        // f(x) = x². Polynomial(2) should extrapolate x²  exactly.
        let inner = |x: f64| x * x;
        let interp =
            ExtrapolatingInterpolator::new(inner, 0.0, 1.0, ExtrapolationMode::Polynomial(2));
        let val = interp.eval(2.0).unwrap();
        // Lagrange extrapolation of x² should give 4.
        assert!(
            (val - 4.0).abs() < 1e-6,
            "poly extrap degree 2 above: {val}"
        );
    }

    #[test]
    fn test_inside_domain_uses_inner() {
        let interp =
            ExtrapolatingInterpolator::new(|x: f64| x * x, 0.0, 1.0, ExtrapolationMode::Error);
        assert!((interp.eval(0.5).unwrap() - 0.25).abs() < 1e-15);
    }

    #[test]
    fn test_asymmetric_different_modes() {
        // Lower: Nearest, Upper: Error
        let interp = ExtrapolatingInterpolatorAsymmetric::new(
            linear_unit(),
            0.0,
            1.0,
            ExtrapolationMode::Nearest,
            ExtrapolationMode::Error,
        );
        // Below: nearest
        let below = interp.eval(-0.5).expect("lower Nearest");
        assert!((below - 0.0).abs() < 1e-12);
        // Above: error
        let above = interp.eval(1.5);
        assert!(above.is_err(), "upper Error mode should fail");
    }

    #[test]
    fn test_extrapolation_in_range_all_modes() {
        let modes = vec![
            ExtrapolationMode::Nearest,
            ExtrapolationMode::Linear,
            ExtrapolationMode::Polynomial(2),
            ExtrapolationMode::Reflect,
            ExtrapolationMode::Periodic,
            ExtrapolationMode::Fill(0.0),
            ExtrapolationMode::Error,
        ];
        for mode in modes {
            let interp = ExtrapolatingInterpolator::new(|x: f64| x, 0.0, 1.0, mode);
            // In-range queries should always succeed.
            let val = interp
                .eval(0.5)
                .expect("in-range should succeed for any mode");
            assert!((val - 0.5).abs() < 1e-12, "in-range eval failed: {val}");
        }
    }

    #[test]
    fn test_lagrange_eval_linear() {
        let nodes = vec![0.0, 1.0];
        let vals = vec![1.0, 3.0]; // f(x) = 2x+1
        let y = lagrange_eval(&nodes, &vals, 2.0);
        assert!((y - 5.0).abs() < 1e-10, "Lagrange extrapolation: {y}");
    }

    #[test]
    fn test_wrap_periodic() {
        // 5.0 is within [0, 2π] ≈ [0, 6.283], so it should wrap to itself.
        let wrapped_inside = wrap_periodic(5.0, 0.0, 2.0 * std::f64::consts::PI);
        assert!(
            wrapped_inside >= 0.0 && wrapped_inside <= 2.0 * std::f64::consts::PI,
            "inside wrap failed: {wrapped_inside}"
        );
        assert!(
            (wrapped_inside - 5.0).abs() < 1e-12,
            "inside wrap should be 5.0, got {wrapped_inside}"
        );

        // 7.0 > 2π: should wrap to 7.0 - 2π ≈ 0.717
        let wrapped_above = wrap_periodic(7.0, 0.0, 2.0 * std::f64::consts::PI);
        let expected_above = 7.0 - 2.0 * std::f64::consts::PI;
        assert!(
            (wrapped_above - expected_above).abs() < 1e-12,
            "above wrap: {wrapped_above} vs {expected_above}"
        );

        // -1.0 < 0: should wrap to -1.0 + 2π
        let wrapped_below = wrap_periodic(-1.0, 0.0, 2.0 * std::f64::consts::PI);
        let expected_below = -1.0 + 2.0 * std::f64::consts::PI;
        assert!(
            (wrapped_below - expected_below).abs() < 1e-12,
            "below wrap: {wrapped_below} vs {expected_below}"
        );
    }
}
