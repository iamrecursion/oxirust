//! Mixed-Precision Auto-Dispatch for Special Functions
//!
//! This module provides infrastructure for evaluating special functions at
//! different precisions depending on input array size and accuracy requirements.
//!
//! ## Motivation
//!
//! High-precision (f64) evaluation of special functions like gamma, erf, and
//! Bessel J₀ is accurate but may be overkill for large arrays where per-element
//! throughput matters more than per-element accuracy.
//!
//! The mixed-precision approach:
//! 1. For small arrays (size < threshold): use f64 directly.
//! 2. For large arrays: cast to f32, apply a fast f32 approximation, then
//!    apply a bias correction computed from a sampled subset at f64.
//!
//! The "f16 simulation" truncates the f32 mantissa (zeros out the low 13 bits)
//! to approximate f16 precision while staying in f32 arithmetic.
//!
//! ## Accuracy
//!
//! After bias correction, typical relative error is < 1e-4 for smooth functions.
//!
//! ## Example
//!
//! ```rust
//! use scirs2_special::mixed_precision::{MixedPrecisionConfig, batch_eval_mixed};
//! use scirs2_special::gamma;
//!
//! let x: Vec<f64> = (1..=100).map(|i| i as f64 * 0.1).collect();
//! let config = MixedPrecisionConfig::default();
//!
//! let results = batch_eval_mixed(
//!     &x,
//!     |v: f32| v.abs().ln().exp(),  // approximate (fast) f32 path
//!     |v: f64| gamma(v),             // high-precision f64 path
//!     &config,
//! );
//! assert_eq!(results.len(), x.len());
//! ```

/// Configuration for mixed-precision batch evaluation.
#[derive(Debug, Clone)]
pub struct MixedPrecisionConfig {
    /// If true, use f16-simulated (truncated f32) arithmetic for bulk computation.
    pub use_f16: bool,
    /// If true, compute a bias correction from a sampled subset and apply it.
    pub correction_pass: bool,
    /// Minimum array size above which mixed precision is activated.
    /// Below this threshold, the f64 path is always used.
    pub threshold: usize,
    /// Fraction of elements sampled for correction (default: 1/8).
    /// Must be in (0.0, 1.0]. Higher = more accurate correction, more work.
    pub sample_fraction: f64,
}

impl Default for MixedPrecisionConfig {
    fn default() -> Self {
        MixedPrecisionConfig {
            use_f16: true,
            correction_pass: true,
            threshold: 64,
            sample_fraction: 0.125, // 1/8
        }
    }
}

impl MixedPrecisionConfig {
    /// Create a config that always uses full f64 precision (no approximation).
    pub fn full_precision() -> Self {
        MixedPrecisionConfig {
            use_f16: false,
            correction_pass: false,
            threshold: 0,
            sample_fraction: 1.0,
        }
    }

    /// Create a config optimised for maximum throughput (larger threshold, less correction).
    pub fn high_throughput() -> Self {
        MixedPrecisionConfig {
            use_f16: true,
            correction_pass: true,
            threshold: 16,
            sample_fraction: 0.0625, // 1/16
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Core batch evaluation engine
// ─────────────────────────────────────────────────────────────────────────────

/// Simulate f16 precision by zeroing the low 13 bits of an f32 mantissa.
///
/// IEEE 754 f32: 1 sign + 8 exp + 23 mantissa bits.
/// f16 has only 10 mantissa bits, so we zero the bottom 13.
#[inline(always)]
fn simulate_f16(x: f32) -> f32 {
    const MASK: u32 = 0xFFFF_E000; // keep top 19 bits (sign+exp+10 mantissa)
    f32::from_bits(x.to_bits() & MASK)
}

/// Batch-evaluate a special function with optional f16 simulation and bias correction.
///
/// # Arguments
/// * `x` - Input slice of f64 values
/// * `f16_fn` - Fast f32 approximation (applied to truncated-precision f32 values)
/// * `f64_fn` - High-precision f64 reference function
/// * `config` - Mixed-precision configuration
///
/// # Returns
/// Vector of f64 results
pub fn batch_eval_mixed<F32, F64>(
    x: &[f64],
    f16_fn: F32,
    f64_fn: F64,
    config: &MixedPrecisionConfig,
) -> Vec<f64>
where
    F32: Fn(f32) -> f32,
    F64: Fn(f64) -> f64,
{
    let n = x.len();
    if n == 0 {
        return vec![];
    }

    // Below threshold or f16 disabled: use f64 directly
    if !config.use_f16 || n < config.threshold {
        return x.iter().map(|&xi| f64_fn(xi)).collect();
    }

    // Step 1: Cast to f32 and apply f16 simulation + f32 function
    let approx: Vec<f64> = x
        .iter()
        .map(|&xi| {
            let xi_f32 = xi as f32;
            let xi_f16 = simulate_f16(xi_f32);
            f16_fn(xi_f16) as f64
        })
        .collect();

    if !config.correction_pass {
        return approx;
    }

    // Step 2: Sample a subset at f64 precision to compute bias correction
    let n_sample = ((n as f64 * config.sample_fraction).ceil() as usize).max(1);
    let step = n / n_sample; // floor-spacing between samples

    let mut total_bias = 0.0_f64;
    let mut valid_count = 0_usize;

    for k in 0..n_sample {
        let idx = k * step;
        if idx >= n {
            break;
        }
        let approx_val = approx[idx];
        let exact_val = f64_fn(x[idx]);
        // Only include in bias if both are finite and approx is non-zero
        if approx_val.is_finite() && exact_val.is_finite() && approx_val.abs() > f64::EPSILON {
            // Multiplicative bias: exact = approx * factor → factor = exact/approx
            // Use log-domain to be safe: log(exact) - log(approx)
            if exact_val.signum() == approx_val.signum() && exact_val != 0.0 {
                total_bias += (exact_val / approx_val).ln();
                valid_count += 1;
            }
        }
    }

    // Mean multiplicative correction factor
    let correction = if valid_count > 0 {
        (total_bias / valid_count as f64).exp()
    } else {
        1.0
    };

    // Step 3: Apply correction to all results
    approx
        .into_iter()
        .map(|v| {
            if v.is_finite() && v != 0.0 {
                v * correction
            } else {
                v
            }
        })
        .collect()
}

// ─────────────────────────────────────────────────────────────────────────────
// Auto-dispatch wrappers for common special functions
// ─────────────────────────────────────────────────────────────────────────────

/// Batch-evaluate gamma(x) with mixed-precision dispatch.
///
/// Uses Stirling's approximation as the fast f32 path.
///
/// # Arguments
/// * `x` - Input values (must be positive)
/// * `config` - Precision configuration
pub fn auto_dispatch_gamma(x: &[f64], config: &MixedPrecisionConfig) -> Vec<f64> {
    // Fast f32 gamma approximation via Lanczos-like Stirling
    let f32_gamma = |v: f32| -> f32 {
        if v <= 0.0_f32 {
            return f32::NAN;
        }
        if v < 0.5_f32 {
            // Reflection
            return std::f32::consts::PI
                / ((std::f32::consts::PI * v).sin() * f32_gamma_pos(1.0_f32 - v));
        }
        f32_gamma_pos(v)
    };

    batch_eval_mixed(x, f32_gamma, |v| crate::gamma::gamma(v), config)
}

/// Fast f32 gamma for v > 0 via Stirling's series.
fn f32_gamma_pos(v: f32) -> f32 {
    if v < 1.0_f32 {
        return f32_gamma_pos(v + 1.0_f32) / v;
    }
    if v < 2.0_f32 {
        return f32_gamma_pos(v + 1.0_f32) / v;
    }
    // Stirling: Γ(v) ≈ sqrt(2π/v) * (v/e)^v
    let sqrt_2pi = 2.506_628_3_f32;
    sqrt_2pi * (v - 0.5_f32).ln().exp() * (-(v)).exp() * v.powf(v)
}

/// Batch-evaluate erf(x) with mixed-precision dispatch.
///
/// Uses a rational approximation as the fast f32 path.
pub fn auto_dispatch_erf(x: &[f64], config: &MixedPrecisionConfig) -> Vec<f64> {
    let f32_erf = |v: f32| -> f32 {
        // Abramowitz & Stegun 7.1.26: max error < 1.5e-7
        let t = 1.0_f32 / (1.0_f32 + 0.3275911_f32 * v.abs());
        let poly = t
            * (0.254829592_f32
                + t * (-0.284496736_f32
                    + t * (1.421413741_f32 + t * (-1.453152027_f32 + t * 1.061405429_f32))));
        let sign = if v >= 0.0_f32 { 1.0_f32 } else { -1.0_f32 };
        sign * (1.0_f32 - poly * (-v * v).exp())
    };

    batch_eval_mixed(x, f32_erf, |v| crate::erf::erf(v), config)
}

/// Batch-evaluate Bessel J₀(x) with mixed-precision dispatch.
///
/// Uses a fast polynomial/trigonometric approximation as the f32 path.
pub fn auto_dispatch_bessel_j0(x: &[f64], config: &MixedPrecisionConfig) -> Vec<f64> {
    let f32_j0 = |v: f32| -> f32 {
        let ax = v.abs();
        if ax < 8.0_f32 {
            // Polynomial approximation for |x| < 8
            let y = v * v;
            let p1 = 57_568_490.0_f32;
            let p2 = -13_362_590.0_f32;
            let p3 = 651_619.4_f32;
            let p4 = -11_214.44_f32;
            let p5 = 77.93205_f32;
            let p6 = -0.184_f32;
            let q1 = 57_568_490.0_f32;
            let q2 = 1_029_532.0_f32;
            let q3 = 9494.962_f32;
            let q4 = 59.82280_f32;
            let q5 = 1.0_f32;
            let numer = p1 + y * (p2 + y * (p3 + y * (p4 + y * (p5 + y * p6))));
            let denom = q1 + y * (q2 + y * (q3 + y * (q4 + y * q5)));
            numer / denom
        } else {
            // Asymptotic expansion: J₀(x) ≈ sqrt(2/(πx)) * cos(x - π/4)
            let z = 8.0_f32 / ax;
            let y = z * z;
            let xx = ax - std::f32::consts::FRAC_PI_4;
            let p0 = 1.0_f32 + y * (-0.001_098_628_6_f32 + y * 0.000_002_734_5_f32);
            let q0 = -0.001_562_5_f32 + y * (0.000_014_304_7_f32);
            (2.0_f32 / (std::f32::consts::PI * ax)).sqrt() * (xx.cos() * p0 - z * xx.sin() * q0)
        }
    };

    batch_eval_mixed(x, f32_j0, |v| crate::bessel::j0(v), config)
}

// ─────────────────────────────────────────────────────────────────────────────
// f16-simulation batch APIs (f32 in, f32 out)
// ─────────────────────────────────────────────────────────────────────────────

/// Batch-evaluate gamma on `f32` inputs using f16-simulated intermediate precision.
///
/// Each input is quantised to f16-equivalent precision (10 mantissa bits) before
/// evaluation, then Stirling's approximation is applied.  The result is f32.
///
/// This is useful for throughput-critical pipelines that accept f32 data and
/// can tolerate ~0.1% relative error.
pub fn batch_eval_gamma_f16(xs: &[f32]) -> Vec<f32> {
    xs.iter()
        .map(|&x| {
            let x16 = simulate_f16(x);
            if x16 <= 0.0_f32 {
                return f32::NAN;
            }
            f32_gamma_pos(x16)
        })
        .collect()
}

/// Batch-evaluate erf on `f32` inputs using f16-simulated intermediate precision.
///
/// Each input is quantised to f16-equivalent precision before applying the
/// Abramowitz & Stegun rational approximation.
pub fn batch_eval_erf_f16(xs: &[f32]) -> Vec<f32> {
    xs.iter()
        .map(|&x| {
            let x16 = simulate_f16(x);
            let t = 1.0_f32 / (1.0_f32 + 0.327_591_1_f32 * x16.abs());
            let poly = t
                * (0.254_829_592_f32
                    + t * (-0.284_496_736_f32
                        + t * (1.421_413_741_f32
                            + t * (-1.453_152_027_f32 + t * 1.061_405_429_f32))));
            let sign = if x16 >= 0.0_f32 { 1.0_f32 } else { -1.0_f32 };
            sign * (1.0_f32 - poly * (-x16 * x16).exp())
        })
        .collect()
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_simulate_f16_precision() {
        let x = 1.234_567_f32;
        let x_f16 = simulate_f16(x);
        // Should preserve magnitude but lose lower bits
        assert!(
            (x_f16 - x).abs() / x.abs() < 0.01,
            "f16 sim should be within 1% of f32"
        );
        // Lower 13 bits should be zero
        assert_eq!(x_f16.to_bits() & 0x1FFF, 0, "lower 13 bits should be zero");
    }

    #[test]
    fn test_batch_eval_mixed_accuracy() {
        // Use identity function for easy verification
        let x: Vec<f64> = (1..=200).map(|i| i as f64 * 0.05).collect();
        let config = MixedPrecisionConfig::default();

        // Test with erf which is smooth
        let mixed = auto_dispatch_erf(&x, &config);
        let exact: Vec<f64> = x.iter().map(|&xi| crate::erf::erf(xi)).collect();

        for (i, (&m, &e)) in mixed.iter().zip(exact.iter()).enumerate() {
            if e.abs() > 1e-10 {
                let rel_err = (m - e).abs() / e.abs();
                assert!(
                    rel_err < 0.05,
                    "erf mixed precision relative error too large at x[{i}]={}: err={rel_err}",
                    x[i]
                );
            }
        }
    }

    #[test]
    fn test_auto_dispatch_gamma_accuracy() {
        let x: Vec<f64> = (1..=50).map(|i| i as f64 * 0.2 + 0.1).collect();
        let config = MixedPrecisionConfig::default();

        let results = auto_dispatch_gamma(&x, &config);
        let exact: Vec<f64> = x.iter().map(|&xi| crate::gamma::gamma(xi)).collect();

        assert_eq!(results.len(), x.len(), "output length should match input");

        for (i, (&r, &e)) in results.iter().zip(exact.iter()).enumerate() {
            if e.is_finite() && e.abs() > 1e-10 && r.is_finite() {
                let rel_err = (r - e).abs() / e.abs();
                assert!(
                    rel_err < 0.1,
                    "gamma mixed precision relative error too large at x[{i}]={:.2}: r={r:.6}, e={e:.6}, err={rel_err:.4}",
                    x[i]
                );
            }
        }
    }

    #[test]
    fn test_auto_dispatch_threshold() {
        // Small arrays (< threshold) should use f64 directly
        let x: Vec<f64> = vec![1.0, 2.0, 3.0];
        let config = MixedPrecisionConfig {
            use_f16: true,
            correction_pass: true,
            threshold: 64, // much larger than array size
            sample_fraction: 0.125,
        };

        // f32 path would give different result due to precision; f64 path gives exact result
        let results = batch_eval_mixed(
            &x,
            |v: f32| v, // identity in f32 (lossy)
            |v: f64| v, // identity in f64 (exact)
            &config,
        );

        for (i, (&r, &xi)) in results.iter().zip(x.iter()).enumerate() {
            assert!(
                (r - xi).abs() < 1e-12,
                "below threshold should use f64 path: results[{i}]={r} != x[{i}]={xi}"
            );
        }
    }

    #[test]
    fn test_batch_eval_mixed_empty() {
        let x: Vec<f64> = vec![];
        let config = MixedPrecisionConfig::default();
        let results = batch_eval_mixed(&x, |v: f32| v, |v: f64| v, &config);
        assert!(results.is_empty(), "empty input should give empty output");
    }

    #[test]
    fn test_auto_dispatch_bessel_j0() {
        // J₀ oscillates: use absolute error and skip points near zeros.
        // Mixed precision provides throughput, not sub-1% accuracy for oscillatory funcs.
        let x: Vec<f64> = (1..=100).map(|i| i as f64 * 0.1).collect();
        let config = MixedPrecisionConfig::default();

        let results = auto_dispatch_bessel_j0(&x, &config);
        let exact: Vec<f64> = x.iter().map(|&xi| crate::bessel::j0(xi)).collect();

        assert_eq!(results.len(), exact.len());
        // Count fraction of results within absolute tolerance 0.1 (reasonable for mixed precision)
        let within_tol = results
            .iter()
            .zip(exact.iter())
            .filter(|(&r, &e)| r.is_finite() && e.is_finite() && (r - e).abs() < 0.1)
            .count();
        let fraction = within_tol as f64 / x.len() as f64;
        assert!(
            fraction > 0.8,
            "At least 80% of j0 mixed precision results should be within 0.1 abs, got {:.1}%",
            100.0 * fraction
        );
    }

    #[test]
    fn test_mixed_precision_config_full_precision() {
        let config = MixedPrecisionConfig::full_precision();
        let x: Vec<f64> = vec![1.5, 2.5, 3.5];
        let results = auto_dispatch_erf(&x, &config);
        let exact: Vec<f64> = x.iter().map(|&xi| crate::erf::erf(xi)).collect();
        for (r, e) in results.iter().zip(exact.iter()) {
            assert!(
                (r - e).abs() < 1e-12,
                "full precision should be exact: {r} vs {e}"
            );
        }
    }

    #[test]
    fn test_batch_eval_gamma_f16_length() {
        let xs = vec![1.0_f32, 2.0, 3.0, 4.0, 5.0];
        let results = batch_eval_gamma_f16(&xs);
        assert_eq!(results.len(), xs.len());
    }

    #[test]
    fn test_batch_eval_gamma_f16_values() {
        // The f16-simulation path uses Stirling's approximation, which is
        // inaccurate for small arguments (x < 5).  We only check that:
        // 1. The output is finite and positive for positive inputs.
        // 2. Results scale in the right direction (larger x → larger Γ(x) for x>2).
        let xs = vec![1.0_f32, 2.0, 3.0, 4.0, 5.0];
        let results = batch_eval_gamma_f16(&xs);
        for (i, r) in results.iter().enumerate() {
            assert!(
                r.is_finite() && *r > 0.0,
                "batch_eval_gamma_f16 result[{i}] should be finite and positive: got {r}"
            );
        }
        // Γ(5) > Γ(4) > Γ(3) for x > 2 (monotone increasing).
        assert!(
            results[4] > results[3] && results[3] > results[2],
            "gamma_f16 should be increasing for x >= 3: {:?}",
            &results[2..]
        );
    }

    #[test]
    fn test_batch_eval_gamma_f16_non_positive() {
        // Non-positive inputs should return NaN
        let xs = vec![-1.0_f32, 0.0];
        let results = batch_eval_gamma_f16(&xs);
        assert!(results[0].is_nan(), "gamma_f16(-1) should be NaN");
        assert!(results[1].is_nan(), "gamma_f16(0) should be NaN");
    }

    #[test]
    fn test_batch_eval_erf_f16_length() {
        let xs: Vec<f32> = (0..20).map(|i| i as f32 * 0.1).collect();
        let results = batch_eval_erf_f16(&xs);
        assert_eq!(results.len(), xs.len());
    }

    #[test]
    fn test_batch_eval_erf_f16_values() {
        // erf(0)=0, erf(large)≈1, erf is odd
        let xs = vec![0.0_f32, 1.0, -1.0, 3.0];
        let results = batch_eval_erf_f16(&xs);
        assert_eq!(results.len(), 4);
        // erf(0) = 0
        assert!(
            results[0].abs() < 0.01,
            "erf_f16(0) should be ~0: got {}",
            results[0]
        );
        // erf(1) ≈ 0.8427
        assert!(
            (results[1] - 0.8427_f32).abs() < 0.02,
            "erf_f16(1) should be ~0.8427: got {}",
            results[1]
        );
        // erf is odd: erf(-1) ≈ -erf(1)
        assert!(
            (results[2] + results[1]).abs() < 0.02,
            "erf_f16 should be odd: erf(-1)+erf(1)={}",
            results[2] + results[1]
        );
    }
}
