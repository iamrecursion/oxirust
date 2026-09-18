//! Activation function SIMD tests Part 2: elu, leaky_relu, prelu, selu, hardsigmoid, hardswish, sinc, log-softmax

use scirs2_core::ndarray::{array, Array1};
use scirs2_core::ndarray_ext::elementwise::{
    elu_simd, hardsigmoid_simd, hardswish_simd, leaky_relu_simd, prelu_simd, selu_simd,
    sigmoid_simd, sinc_simd, swish_simd,
};
#[cfg(feature = "random")]
use scirs2_core::random::{thread_rng, Distribution, Uniform};

// ELU SIMD Tests
// ============================================================================

/// Test ELU with f64 - basic behavior
#[test]
fn test_elu_simd_f64_basic() {
    let x = array![-2.0_f64, -1.0, 0.0, 1.0, 2.0];
    let alpha = 1.0;
    let result = elu_simd(&x.view(), alpha);

    // ELU(x) = x for x >= 0
    assert!((result[2] - 0.0).abs() < 1e-10, "ELU(0) should be 0");
    assert!((result[3] - 1.0).abs() < 1e-10, "ELU(1) should be 1");
    assert!((result[4] - 2.0).abs() < 1e-10, "ELU(2) should be 2");

    // ELU(x) = α * (exp(x) - 1) for x < 0
    let expected_neg1 = alpha * ((-1.0_f64).exp() - 1.0); // ≈ -0.6321
    let expected_neg2 = alpha * ((-2.0_f64).exp() - 1.0); // ≈ -0.8647
    assert!(
        (result[1] - expected_neg1).abs() < 1e-10,
        "ELU(-1) should be {}, got {}",
        expected_neg1,
        result[1]
    );
    assert!(
        (result[0] - expected_neg2).abs() < 1e-10,
        "ELU(-2) should be {}, got {}",
        expected_neg2,
        result[0]
    );
}

/// Test ELU with f32
#[test]
fn test_elu_simd_f32_basic() {
    let x = array![-2.0_f32, -1.0, 0.0, 1.0, 2.0];
    let alpha = 1.0_f32;
    let result = elu_simd(&x.view(), alpha);

    // ELU(x) = x for x >= 0
    assert!((result[2] - 0.0).abs() < 1e-6, "ELU(0) should be 0");
    assert!((result[3] - 1.0).abs() < 1e-6, "ELU(1) should be 1");

    // ELU(x) = α * (exp(x) - 1) for x < 0
    let expected_neg1 = alpha * ((-1.0_f32).exp() - 1.0);
    assert!(
        (result[1] - expected_neg1).abs() < 1e-5,
        "ELU(-1) should be approximately {}",
        expected_neg1
    );
}

/// Test ELU empty array
#[test]
fn test_elu_simd_empty() {
    let x = array![] as Array1<f64>;
    let result = elu_simd(&x.view(), 1.0);
    assert!(result.is_empty(), "ELU of empty array should be empty");
}

/// Test ELU with different alpha values
#[test]
fn test_elu_simd_different_alpha() {
    let x = array![-1.0_f64, -0.5, 0.5, 1.0];

    // Alpha = 0.5
    let alpha_half = 0.5;
    let result_half = elu_simd(&x.view(), alpha_half);
    let expected_neg1_half = alpha_half * ((-1.0_f64).exp() - 1.0);
    assert!(
        (result_half[0] - expected_neg1_half).abs() < 1e-10,
        "ELU(-1, α=0.5) should be {}, got {}",
        expected_neg1_half,
        result_half[0]
    );

    // Alpha = 2.0
    let alpha_double = 2.0;
    let result_double = elu_simd(&x.view(), alpha_double);
    let expected_neg1_double = alpha_double * ((-1.0_f64).exp() - 1.0);
    assert!(
        (result_double[0] - expected_neg1_double).abs() < 1e-10,
        "ELU(-1, α=2.0) should be {}, got {}",
        expected_neg1_double,
        result_double[0]
    );

    // Positive values should be unchanged regardless of alpha
    assert!((result_half[2] - 0.5).abs() < 1e-10);
    assert!((result_double[2] - 0.5).abs() < 1e-10);
    assert!((result_half[3] - 1.0).abs() < 1e-10);
    assert!((result_double[3] - 1.0).abs() < 1e-10);
}

/// Test ELU large array (SIMD path)
#[test]
fn test_elu_simd_large_array() {
    let n = 10000;
    let x: Array1<f64> = Array1::linspace(-5.0, 5.0, n);
    let alpha = 1.0;
    let result = elu_simd(&x.view(), alpha);

    // Check length
    assert_eq!(result.len(), n, "Result should have same length as input");

    // Verify all values are computed correctly for a sample
    for i in (0..n).step_by(500) {
        let xi = x[i];
        let expected = if xi >= 0.0 {
            xi
        } else {
            alpha * (xi.exp() - 1.0)
        };
        assert!(
            (result[i] - expected).abs() < 1e-10,
            "ELU({}) should be {}, got {}",
            xi,
            expected,
            result[i]
        );
    }
}

/// Test ELU NaN handling
#[test]
fn test_elu_simd_nan() {
    let x = array![f64::NAN, 1.0, f64::NAN, -1.0];
    let result = elu_simd(&x.view(), 1.0);

    assert!(result[0].is_nan(), "ELU(NaN) should be NaN");
    assert!(result[2].is_nan(), "ELU(NaN) should be NaN");
    assert!((result[1] - 1.0).abs() < 1e-10);
}

/// Test ELU infinity handling
#[test]
fn test_elu_simd_infinity() {
    let x = array![f64::INFINITY, f64::NEG_INFINITY, 0.0];
    let result = elu_simd(&x.view(), 1.0);

    // ELU(+inf) = +inf
    assert!(
        result[0].is_infinite() && result[0] > 0.0,
        "ELU(+inf) should be +inf"
    );
    // ELU(-inf) = α * (exp(-inf) - 1) = α * (0 - 1) = -α
    assert!(
        (result[1] - (-1.0)).abs() < 1e-10,
        "ELU(-inf) should be -α = -1.0, got {}",
        result[1]
    );
}

/// Test ELU vs ReLU: ELU is smoother and has non-zero gradient for x < 0
#[test]
fn test_elu_simd_vs_relu() {
    let x = array![-2.0_f64, -1.0, -0.5, 0.0, 0.5, 1.0];
    let alpha = 1.0;
    let elu_result = elu_simd(&x.view(), alpha);

    // For x >= 0, ELU(x) = x (same as ReLU)
    assert!((elu_result[3] - 0.0).abs() < 1e-10); // x = 0
    assert!((elu_result[4] - 0.5).abs() < 1e-10); // x = 0.5
    assert!((elu_result[5] - 1.0).abs() < 1e-10); // x = 1

    // For x < 0, ELU(x) = α * (exp(x) - 1) ∈ (-α, 0)
    // ReLU(x) = 0 for x < 0
    // ELU is strictly greater than -α for finite x
    assert!(elu_result[0] > -alpha, "ELU(-2) should be > -α");
    assert!(elu_result[1] > -alpha, "ELU(-1) should be > -α");
    assert!(elu_result[2] > -alpha, "ELU(-0.5) should be > -α");

    // ELU is negative for x < 0
    assert!(elu_result[0] < 0.0, "ELU(-2) should be < 0");
    assert!(elu_result[1] < 0.0, "ELU(-1) should be < 0");
    assert!(elu_result[2] < 0.0, "ELU(-0.5) should be < 0");
}

/// Test ELU continuity at x = 0
#[test]
fn test_elu_simd_continuity_at_zero() {
    // Test that ELU approaches 0 from both sides
    let alpha = 1.0_f64;
    let eps = 1e-8;
    let x = array![-eps, 0.0, eps];
    let result = elu_simd(&x.view(), alpha);

    // All values should be close to 0
    assert!(
        result[0].abs() < 1e-7,
        "ELU(-eps) should be close to 0, got {}",
        result[0]
    );
    assert!(
        result[1].abs() < 1e-10,
        "ELU(0) should be 0, got {}",
        result[1]
    );
    assert!(
        result[2].abs() < 1e-7,
        "ELU(eps) should be close to 0, got {}",
        result[2]
    );

    // Check derivative continuity: derivative at 0- = α, derivative at 0+ = 1
    // With α = 1, derivative is continuous
}

/// Test ELU asymptotic behavior
#[test]
fn test_elu_simd_asymptotic() {
    let alpha = 1.0_f64;

    // For large positive x, ELU(x) = x
    let large_positive = array![10.0_f64, 50.0, 100.0];
    let result_pos = elu_simd(&large_positive.view(), alpha);
    assert!((result_pos[0] - 10.0).abs() < 1e-10);
    assert!((result_pos[1] - 50.0).abs() < 1e-10);
    assert!((result_pos[2] - 100.0).abs() < 1e-10);

    // For large negative x, ELU(x) → -α
    let large_negative = array![-10.0_f64, -50.0, -100.0];
    let result_neg = elu_simd(&large_negative.view(), alpha);
    // exp(-10) ≈ 4.5e-5, so ELU(-10) ≈ -1 + 4.5e-5 ≈ -0.99995
    assert!(
        (result_neg[0] - (-alpha)).abs() < 0.001,
        "ELU(-10) should be approximately -α, got {}",
        result_neg[0]
    );
    assert!(
        (result_neg[1] - (-alpha)).abs() < 1e-10,
        "ELU(-50) should be approximately -α, got {}",
        result_neg[1]
    );
    assert!(
        (result_neg[2] - (-alpha)).abs() < 1e-10,
        "ELU(-100) should be approximately -α, got {}",
        result_neg[2]
    );
}

/// Test ELU in neural network use case
#[test]
fn test_elu_simd_neural_network_use_case() {
    // Simulate activation in a hidden layer
    let features = array![0.5_f64, 1.2, -0.8, 2.0, -1.5, 0.0, 3.0, -3.0];
    let alpha = 1.0;
    let activated = elu_simd(&features.view(), alpha);

    // All results should be finite
    for i in 0..features.len() {
        assert!(
            activated[i].is_finite(),
            "ELU output should be finite for input {}",
            features[i]
        );
    }

    // Zero should produce zero
    assert!(activated[5].abs() < 1e-10);

    // Positive values should be unchanged
    assert!((activated[0] - 0.5).abs() < 1e-10);
    assert!((activated[1] - 1.2).abs() < 1e-10);
    assert!((activated[3] - 2.0).abs() < 1e-10);
    assert!((activated[6] - 3.0).abs() < 1e-10);

    // Negative values should be in range (-α, 0)
    assert!(activated[2] > -alpha && activated[2] < 0.0);
    assert!(activated[4] > -alpha && activated[4] < 0.0);
    assert!(activated[7] > -alpha && activated[7] < 0.0);
}

// ============================================================================
// ============================================================================
// Leaky ReLU / PReLU SIMD Tests
// ============================================================================

/// Test Leaky ReLU with f64 - basic behavior
#[test]
fn test_leaky_relu_simd_f64_basic() {
    let x = array![-2.0_f64, -1.0, 0.0, 1.0, 2.0];
    let alpha = 0.01;
    let result = leaky_relu_simd(&x.view(), alpha);

    // Leaky ReLU(x) = x for x >= 0
    assert!((result[2] - 0.0).abs() < 1e-10, "Leaky ReLU(0) should be 0");
    assert!((result[3] - 1.0).abs() < 1e-10, "Leaky ReLU(1) should be 1");
    assert!((result[4] - 2.0).abs() < 1e-10, "Leaky ReLU(2) should be 2");

    // Leaky ReLU(x) = alpha * x for x < 0
    assert!(
        (result[1] - (-0.01)).abs() < 1e-10,
        "Leaky ReLU(-1) should be -0.01, got {}",
        result[1]
    );
    assert!(
        (result[0] - (-0.02)).abs() < 1e-10,
        "Leaky ReLU(-2) should be -0.02, got {}",
        result[0]
    );
}

/// Test Leaky ReLU with f32
#[test]
fn test_leaky_relu_simd_f32_basic() {
    let x = array![-2.0_f32, -1.0, 0.0, 1.0, 2.0];
    let alpha = 0.1_f32;
    let result = leaky_relu_simd(&x.view(), alpha);

    // Leaky ReLU(x) = x for x >= 0
    assert!((result[2] - 0.0).abs() < 1e-6, "Leaky ReLU(0) should be 0");
    assert!((result[3] - 1.0).abs() < 1e-6, "Leaky ReLU(1) should be 1");

    // Leaky ReLU(x) = alpha * x for x < 0
    assert!(
        (result[1] - (-0.1)).abs() < 1e-6,
        "Leaky ReLU(-1) should be -0.1"
    );
}

/// Test Leaky ReLU empty array
#[test]
fn test_leaky_relu_simd_empty() {
    let x = array![] as Array1<f64>;
    let result = leaky_relu_simd(&x.view(), 0.01);
    assert!(
        result.is_empty(),
        "Leaky ReLU of empty array should be empty"
    );
}

/// Test PReLU alias works correctly
#[test]
fn test_prelu_simd_alias() {
    let x = array![-2.0_f64, -1.0, 0.0, 1.0, 2.0];
    let alpha = 0.25; // Typical learned PReLU value

    let leaky_result = leaky_relu_simd(&x.view(), alpha);
    let prelu_result = prelu_simd(&x.view(), alpha);

    // PReLU should produce identical results to Leaky ReLU
    for i in 0..x.len() {
        assert!(
            (leaky_result[i] - prelu_result[i]).abs() < 1e-15,
            "PReLU and Leaky ReLU should be identical"
        );
    }
}

/// Test Leaky ReLU with different alpha values
#[test]
fn test_leaky_relu_simd_different_alpha() {
    let x = array![-1.0_f64, -0.5, 0.5, 1.0];

    // Standard Leaky ReLU: alpha = 0.01
    let alpha_001 = 0.01;
    let result_001 = leaky_relu_simd(&x.view(), alpha_001);
    assert!((result_001[0] - (-0.01)).abs() < 1e-10);

    // More aggressive: alpha = 0.3
    let alpha_03 = 0.3;
    let result_03 = leaky_relu_simd(&x.view(), alpha_03);
    assert!((result_03[0] - (-0.3)).abs() < 1e-10);

    // Positive values unchanged regardless of alpha
    assert!((result_001[2] - 0.5).abs() < 1e-10);
    assert!((result_03[2] - 0.5).abs() < 1e-10);
}

/// Test Leaky ReLU large array (SIMD path)
#[test]
fn test_leaky_relu_simd_large_array() {
    let n = 10000;
    let x: Array1<f64> = Array1::linspace(-5.0, 5.0, n);
    let alpha = 0.01;
    let result = leaky_relu_simd(&x.view(), alpha);

    // Check length
    assert_eq!(result.len(), n, "Result should have same length as input");

    // Verify all values are computed correctly for a sample
    for i in (0..n).step_by(500) {
        let xi = x[i];
        let expected = if xi >= 0.0 { xi } else { alpha * xi };
        assert!(
            (result[i] - expected).abs() < 1e-10,
            "Leaky ReLU({}) should be {}, got {}",
            xi,
            expected,
            result[i]
        );
    }
}

/// Test Leaky ReLU NaN handling
#[test]
fn test_leaky_relu_simd_nan() {
    let x = array![f64::NAN, 1.0, f64::NAN, -1.0];
    let result = leaky_relu_simd(&x.view(), 0.01);

    assert!(result[0].is_nan(), "Leaky ReLU(NaN) should be NaN");
    assert!(result[2].is_nan(), "Leaky ReLU(NaN) should be NaN");
    assert!((result[1] - 1.0).abs() < 1e-10);
}

/// Test Leaky ReLU vs ReLU: Leaky ReLU doesn't "die"
#[test]
fn test_leaky_relu_simd_vs_relu() {
    let x = array![-2.0_f64, -1.0, -0.5, 0.0, 0.5, 1.0];
    let alpha = 0.01;
    let result = leaky_relu_simd(&x.view(), alpha);

    // For x >= 0, Leaky ReLU(x) = x (same as ReLU)
    assert!((result[3] - 0.0).abs() < 1e-10); // x = 0
    assert!((result[4] - 0.5).abs() < 1e-10); // x = 0.5
    assert!((result[5] - 1.0).abs() < 1e-10); // x = 1

    // For x < 0, Leaky ReLU(x) = alpha * x (unlike ReLU which outputs 0)
    // This ensures gradient flow for negative inputs
    assert!(
        result[0] < 0.0 && result[0] > -0.1,
        "Leaky ReLU(-2) should be small negative"
    );
    assert!(
        result[1] < 0.0 && result[1] > -0.1,
        "Leaky ReLU(-1) should be small negative"
    );
    assert!(
        result[2] < 0.0 && result[2] > -0.1,
        "Leaky ReLU(-0.5) should be small negative"
    );
}

/// Test Leaky ReLU continuity at x = 0
#[test]
fn test_leaky_relu_simd_continuity_at_zero() {
    let alpha = 0.01_f64;
    let eps = 1e-10;
    let x = array![-eps, 0.0, eps];
    let result = leaky_relu_simd(&x.view(), alpha);

    // All values should be very close to 0
    assert!(
        result[0].abs() < 1e-9,
        "Leaky ReLU(-eps) should be close to 0"
    );
    assert!(result[1].abs() < 1e-15, "Leaky ReLU(0) should be 0");
    assert!(
        result[2].abs() < 1e-9,
        "Leaky ReLU(eps) should be close to 0"
    );
}

/// Test Leaky ReLU in neural network use case
#[test]
fn test_leaky_relu_simd_neural_network_use_case() {
    // Simulate activations in a hidden layer
    let features = array![0.5_f64, 1.2, -0.8, 2.0, -1.5, 0.0, 3.0, -3.0];
    let alpha = 0.01;
    let activated = leaky_relu_simd(&features.view(), alpha);

    // All results should be finite
    for i in 0..features.len() {
        assert!(
            activated[i].is_finite(),
            "Leaky ReLU output should be finite for input {}",
            features[i]
        );
    }

    // Zero should produce zero
    assert!(activated[5].abs() < 1e-10);

    // Positive values should be unchanged
    assert!((activated[0] - 0.5).abs() < 1e-10);
    assert!((activated[1] - 1.2).abs() < 1e-10);
    assert!((activated[3] - 2.0).abs() < 1e-10);
    assert!((activated[6] - 3.0).abs() < 1e-10);

    // Negative values should be scaled by alpha
    assert!((activated[2] - (-0.008)).abs() < 1e-10); // -0.8 * 0.01
    assert!((activated[4] - (-0.015)).abs() < 1e-10); // -1.5 * 0.01
    assert!((activated[7] - (-0.03)).abs() < 1e-10); // -3.0 * 0.01
}

// ============================================================================
// ============================================================================
// SELU SIMD Tests
// ============================================================================

// SELU constants for reference in tests
const SELU_ALPHA: f64 = 1.6732632423543772;
const SELU_LAMBDA: f64 = 1.0507009873554805;

/// Test SELU with f64 - basic behavior
#[test]
fn test_selu_simd_f64_basic() {
    let x = array![-2.0_f64, -1.0, 0.0, 1.0, 2.0];
    let result = selu_simd(&x.view());

    // SELU(x) = λ * x for x > 0
    assert!(
        (result[3] - SELU_LAMBDA * 1.0).abs() < 1e-10,
        "SELU(1) should be λ * 1 = {}, got {}",
        SELU_LAMBDA,
        result[3]
    );
    assert!(
        (result[4] - SELU_LAMBDA * 2.0).abs() < 1e-10,
        "SELU(2) should be λ * 2 = {}, got {}",
        SELU_LAMBDA * 2.0,
        result[4]
    );

    // SELU(0) = 0 (boundary case: 0 is treated as x <= 0, so λ * α * (exp(0) - 1) = 0)
    assert!(
        result[2].abs() < 1e-10,
        "SELU(0) should be 0, got {}",
        result[2]
    );

    // SELU(x) = λ * α * (exp(x) - 1) for x <= 0
    let expected_neg1 = SELU_LAMBDA * SELU_ALPHA * ((-1.0_f64).exp() - 1.0);
    let expected_neg2 = SELU_LAMBDA * SELU_ALPHA * ((-2.0_f64).exp() - 1.0);
    assert!(
        (result[1] - expected_neg1).abs() < 1e-10,
        "SELU(-1) should be {}, got {}",
        expected_neg1,
        result[1]
    );
    assert!(
        (result[0] - expected_neg2).abs() < 1e-10,
        "SELU(-2) should be {}, got {}",
        expected_neg2,
        result[0]
    );
}

/// Test SELU with f32
#[test]
fn test_selu_simd_f32_basic() {
    let x = array![-1.0_f32, 0.0, 1.0, 2.0];
    let result = selu_simd(&x.view());

    // SELU(1) = λ * 1 ≈ 1.0507
    assert!(
        (result[2] - 1.050_701_f32).abs() < 1e-5,
        "SELU(1) should be approximately 1.0507"
    );

    // SELU(0) = 0
    assert!(result[1].abs() < 1e-6, "SELU(0) should be 0");

    // SELU(-1) should be negative
    assert!(result[0] < 0.0, "SELU(-1) should be negative");
}

/// Test SELU empty array
#[test]
fn test_selu_simd_empty() {
    let x = array![] as Array1<f64>;
    let result = selu_simd(&x.view());
    assert!(result.is_empty(), "SELU of empty array should be empty");
}

/// Test SELU large array (SIMD path)
#[test]
fn test_selu_simd_large_array() {
    let n = 10000;
    let x: Array1<f64> = Array1::linspace(-5.0, 5.0, n);
    let result = selu_simd(&x.view());

    // Check length
    assert_eq!(result.len(), n, "Result should have same length as input");

    // Verify all values are computed correctly for a sample
    for i in (0..n).step_by(500) {
        let xi = x[i];
        let expected = if xi > 0.0 {
            SELU_LAMBDA * xi
        } else {
            SELU_LAMBDA * SELU_ALPHA * (xi.exp() - 1.0)
        };
        assert!(
            (result[i] - expected).abs() < 1e-10,
            "SELU({}) should be {}, got {}",
            xi,
            expected,
            result[i]
        );
    }
}

/// Test SELU NaN handling
#[test]
fn test_selu_simd_nan() {
    let x = array![f64::NAN, 1.0, f64::NAN, -1.0];
    let result = selu_simd(&x.view());

    assert!(result[0].is_nan(), "SELU(NaN) should be NaN");
    assert!(result[2].is_nan(), "SELU(NaN) should be NaN");
    assert!((result[1] - SELU_LAMBDA).abs() < 1e-10);
}

/// Test SELU vs ELU: SELU is scaled ELU with specific constants
#[test]
fn test_selu_simd_vs_elu() {
    let x = array![-2.0_f64, -1.0, 0.0, 1.0, 2.0];
    let selu_result = selu_simd(&x.view());
    let elu_result = elu_simd(&x.view(), SELU_ALPHA);

    // SELU(x) = λ * ELU(x, α) for the same α
    for i in 0..x.len() {
        let expected = SELU_LAMBDA * elu_result[i];
        assert!(
            (selu_result[i] - expected).abs() < 1e-10,
            "SELU should be λ * ELU at index {}",
            i
        );
    }
}

/// Test SELU scaling property
#[test]
fn test_selu_simd_scaling() {
    // For positive x, SELU(x) = λ * x
    // So SELU(x) / x = λ for all x > 0
    let x = array![0.5_f64, 1.0, 2.0, 5.0, 10.0];
    let result = selu_simd(&x.view());

    for i in 0..x.len() {
        let ratio = result[i] / x[i];
        assert!(
            (ratio - SELU_LAMBDA).abs() < 1e-10,
            "SELU(x)/x should be λ for positive x"
        );
    }
}

/// Test SELU asymptotic behavior
#[test]
fn test_selu_simd_asymptotic() {
    // For large positive x, SELU(x) = λ * x
    let large_positive = array![10.0_f64, 50.0, 100.0];
    let result_pos = selu_simd(&large_positive.view());
    assert!((result_pos[0] - SELU_LAMBDA * 10.0).abs() < 1e-10);
    assert!((result_pos[1] - SELU_LAMBDA * 50.0).abs() < 1e-10);
    assert!((result_pos[2] - SELU_LAMBDA * 100.0).abs() < 1e-10);

    // For large negative x, SELU(x) → -λ * α ≈ -1.7581
    let large_negative = array![-10.0_f64, -50.0, -100.0];
    let result_neg = selu_simd(&large_negative.view());
    let asymptote = -SELU_LAMBDA * SELU_ALPHA;
    assert!(
        (result_neg[0] - asymptote).abs() < 0.001,
        "SELU(-10) should approach -λα"
    );
    assert!(
        (result_neg[1] - asymptote).abs() < 1e-10,
        "SELU(-50) should be very close to -λα"
    );
    assert!(
        (result_neg[2] - asymptote).abs() < 1e-10,
        "SELU(-100) should be very close to -λα"
    );
}

/// Test SELU self-normalizing property (informal check)
#[test]
fn test_selu_simd_self_normalizing_property() {
    // With properly initialized weights, SELU maintains mean ≈ 0, variance ≈ 1
    // This is a simplified test showing the basic transformation properties

    // Create input with approximate mean=0, var=1
    let n = 1000;
    let x: Array1<f64> = Array1::linspace(-3.0, 3.0, n);

    let result = selu_simd(&x.view());

    // All results should be finite
    for i in 0..n {
        assert!(result[i].is_finite(), "SELU output should be finite");
    }

    // Outputs should be bounded below by -λα ≈ -1.7581
    let lower_bound = -SELU_LAMBDA * SELU_ALPHA;
    for i in 0..n {
        assert!(
            result[i] >= lower_bound - 0.001,
            "SELU output should be >= -λα"
        );
    }
}

/// Test SELU in self-normalizing neural network use case
#[test]
fn test_selu_simd_snn_use_case() {
    // Simulate activation in a Self-Normalizing Neural Network
    let features = array![0.5_f64, 1.2, -0.8, 2.0, -1.5, 0.0, 3.0, -3.0];
    let activated = selu_simd(&features.view());

    // All results should be finite
    for i in 0..features.len() {
        assert!(
            activated[i].is_finite(),
            "SELU output should be finite for input {}",
            features[i]
        );
    }

    // Zero should produce zero
    assert!(activated[5].abs() < 1e-10);

    // Positive values should be scaled by λ
    assert!((activated[0] - SELU_LAMBDA * 0.5).abs() < 1e-10);
    assert!((activated[1] - SELU_LAMBDA * 1.2).abs() < 1e-10);
    assert!((activated[3] - SELU_LAMBDA * 2.0).abs() < 1e-10);
    assert!((activated[6] - SELU_LAMBDA * 3.0).abs() < 1e-10);

    // Negative values should be in range (-λα, 0)
    let lower_bound = -SELU_LAMBDA * SELU_ALPHA;
    assert!(activated[2] > lower_bound && activated[2] < 0.0);
    assert!(activated[4] > lower_bound && activated[4] < 0.0);
    assert!(activated[7] > lower_bound && activated[7] < 0.0);
}

// ============================================================================
// ============================================================================
// Hardsigmoid SIMD Tests
// ============================================================================

/// Test Hardsigmoid with f64 - basic behavior
#[test]
fn test_hardsigmoid_simd_f64_basic() {
    let x = array![-4.0_f64, -3.0, -1.5, 0.0, 1.5, 3.0, 4.0];
    let result = hardsigmoid_simd(&x.view());

    // Hardsigmoid(x) = 0 for x <= -3
    assert!(result[0].abs() < 1e-10, "Hardsigmoid(-4) should be 0");
    assert!(result[1].abs() < 1e-10, "Hardsigmoid(-3) should be 0");

    // Hardsigmoid(x) = (x + 3) / 6 for -3 < x < 3
    assert!(
        (result[2] - 0.25).abs() < 1e-10,
        "Hardsigmoid(-1.5) should be 0.25"
    );
    assert!(
        (result[3] - 0.5).abs() < 1e-10,
        "Hardsigmoid(0) should be 0.5"
    );
    assert!(
        (result[4] - 0.75).abs() < 1e-10,
        "Hardsigmoid(1.5) should be 0.75"
    );

    // Hardsigmoid(x) = 1 for x >= 3
    assert!(
        (result[5] - 1.0).abs() < 1e-10,
        "Hardsigmoid(3) should be 1"
    );
    assert!(
        (result[6] - 1.0).abs() < 1e-10,
        "Hardsigmoid(4) should be 1"
    );
}

/// Test Hardsigmoid with f32
#[test]
fn test_hardsigmoid_simd_f32_basic() {
    let x = array![-4.0_f32, 0.0, 4.0];
    let result = hardsigmoid_simd(&x.view());

    assert!(result[0].abs() < 1e-6, "Hardsigmoid(-4) should be 0");
    assert!(
        (result[1] - 0.5).abs() < 1e-6,
        "Hardsigmoid(0) should be 0.5"
    );
    assert!((result[2] - 1.0).abs() < 1e-6, "Hardsigmoid(4) should be 1");
}

/// Test Hardsigmoid empty array
#[test]
fn test_hardsigmoid_simd_empty() {
    let x = array![] as Array1<f64>;
    let result = hardsigmoid_simd(&x.view());
    assert!(
        result.is_empty(),
        "Hardsigmoid of empty array should be empty"
    );
}

/// Test Hardsigmoid large array (SIMD path)
#[test]
fn test_hardsigmoid_simd_large_array() {
    let n = 10000;
    let x: Array1<f64> = Array1::linspace(-5.0, 5.0, n);
    let result = hardsigmoid_simd(&x.view());

    assert_eq!(result.len(), n);

    for i in (0..n).step_by(500) {
        let xi = x[i];
        let expected = if xi <= -3.0 {
            0.0
        } else if xi >= 3.0 {
            1.0
        } else {
            (xi + 3.0) / 6.0
        };
        assert!(
            (result[i] - expected).abs() < 1e-10,
            "Hardsigmoid({}) should be {}, got {}",
            xi,
            expected,
            result[i]
        );
    }
}

/// Test Hardsigmoid output range [0, 1]
#[test]
fn test_hardsigmoid_simd_output_range() {
    let x: Array1<f64> = Array1::linspace(-10.0, 10.0, 1000);
    let result = hardsigmoid_simd(&x.view());

    for i in 0..result.len() {
        assert!(
            result[i] >= 0.0 && result[i] <= 1.0,
            "Hardsigmoid output should be in [0, 1], got {}",
            result[i]
        );
    }
}

/// Test Hardsigmoid vs Sigmoid approximation
#[test]
fn test_hardsigmoid_simd_vs_sigmoid() {
    // Hardsigmoid approximates sigmoid in the linear region
    let x = array![0.0_f64];
    let hardsigmoid_result = hardsigmoid_simd(&x.view());
    let sigmoid_result = sigmoid_simd(&x.view());

    // Both should be 0.5 at x = 0
    assert!(
        (hardsigmoid_result[0] - 0.5).abs() < 1e-10,
        "Hardsigmoid(0) should be 0.5"
    );
    assert!(
        (sigmoid_result[0] - 0.5).abs() < 1e-10,
        "Sigmoid(0) should be 0.5"
    );
}

/// Test Hardsigmoid NaN handling
#[test]
fn test_hardsigmoid_simd_nan() {
    let x = array![f64::NAN, 0.0, f64::NAN];
    let result = hardsigmoid_simd(&x.view());

    assert!(result[0].is_nan(), "Hardsigmoid(NaN) should be NaN");
    assert!(result[2].is_nan(), "Hardsigmoid(NaN) should be NaN");
    assert!((result[1] - 0.5).abs() < 1e-10);
}

// ============================================================================
// ============================================================================
// Hardswish SIMD Tests
// ============================================================================

/// Test Hardswish with f64 - basic behavior
#[test]
fn test_hardswish_simd_f64_basic() {
    let x = array![-4.0_f64, -3.0, -1.5, 0.0, 1.5, 3.0, 4.0];
    let result = hardswish_simd(&x.view());

    // Hardswish(x) = 0 for x <= -3
    assert!(result[0].abs() < 1e-10, "Hardswish(-4) should be 0");
    assert!(result[1].abs() < 1e-10, "Hardswish(-3) should be 0");

    // Hardswish(x) = x * (x + 3) / 6 for -3 < x < 3
    let expected_neg15 = -1.5 * (-1.5 + 3.0) / 6.0; // = -1.5 * 1.5 / 6 = -0.375
    assert!(
        (result[2] - expected_neg15).abs() < 1e-10,
        "Hardswish(-1.5) should be {}, got {}",
        expected_neg15,
        result[2]
    );
    assert!(result[3].abs() < 1e-10, "Hardswish(0) should be 0");
    let expected_15 = 1.5 * (1.5 + 3.0) / 6.0; // = 1.5 * 4.5 / 6 = 1.125
    assert!(
        (result[4] - expected_15).abs() < 1e-10,
        "Hardswish(1.5) should be {}, got {}",
        expected_15,
        result[4]
    );

    // Hardswish(x) = x for x >= 3
    assert!((result[5] - 3.0).abs() < 1e-10, "Hardswish(3) should be 3");
    assert!((result[6] - 4.0).abs() < 1e-10, "Hardswish(4) should be 4");
}

/// Test Hardswish with f32
#[test]
fn test_hardswish_simd_f32_basic() {
    let x = array![-4.0_f32, 0.0, 4.0];
    let result = hardswish_simd(&x.view());

    assert!(result[0].abs() < 1e-6, "Hardswish(-4) should be 0");
    assert!(result[1].abs() < 1e-6, "Hardswish(0) should be 0");
    assert!((result[2] - 4.0).abs() < 1e-6, "Hardswish(4) should be 4");
}

/// Test Hardswish empty array
#[test]
fn test_hardswish_simd_empty() {
    let x = array![] as Array1<f64>;
    let result = hardswish_simd(&x.view());
    assert!(
        result.is_empty(),
        "Hardswish of empty array should be empty"
    );
}

/// Test Hardswish large array (SIMD path)
#[test]
fn test_hardswish_simd_large_array() {
    let n = 10000;
    let x: Array1<f64> = Array1::linspace(-5.0, 5.0, n);
    let result = hardswish_simd(&x.view());

    assert_eq!(result.len(), n);

    for i in (0..n).step_by(500) {
        let xi = x[i];
        let expected = if xi <= -3.0 {
            0.0
        } else if xi >= 3.0 {
            xi
        } else {
            xi * (xi + 3.0) / 6.0
        };
        assert!(
            (result[i] - expected).abs() < 1e-10,
            "Hardswish({}) should be {}, got {}",
            xi,
            expected,
            result[i]
        );
    }
}

/// Test Hardswish relation to Hardsigmoid
#[test]
fn test_hardswish_simd_vs_hardsigmoid() {
    // Hardswish(x) = x * hardsigmoid(x)
    let x = array![-2.0_f64, -1.0, 0.0, 1.0, 2.0];
    let hardswish_result = hardswish_simd(&x.view());
    let hardsigmoid_result = hardsigmoid_simd(&x.view());

    for i in 0..x.len() {
        let expected = x[i] * hardsigmoid_result[i];
        assert!(
            (hardswish_result[i] - expected).abs() < 1e-10,
            "Hardswish should be x * hardsigmoid(x)"
        );
    }
}

/// Test Hardswish vs Swish approximation
#[test]
fn test_hardswish_simd_vs_swish() {
    // Hardswish approximates Swish but is faster (no exp)
    let x = array![0.0_f64, 1.0, 2.0];
    let hardswish_result = hardswish_simd(&x.view());
    let swish_result = swish_simd(&x.view());

    // At x = 0, both should be 0
    assert!(hardswish_result[0].abs() < 1e-10);
    assert!(swish_result[0].abs() < 1e-10);

    // At x = 1 and 2, they should be close but not identical
    // Swish(1) ≈ 0.731, Hardswish(1) = 1 * 4/6 ≈ 0.667
    // Swish(2) ≈ 1.762, Hardswish(2) = 2 * 5/6 ≈ 1.667
    assert!(hardswish_result[1] > 0.5 && hardswish_result[1] < 1.0);
    assert!(hardswish_result[2] > 1.5 && hardswish_result[2] < 2.0);
}

/// Test Hardswish NaN handling
#[test]
fn test_hardswish_simd_nan() {
    let x = array![f64::NAN, 0.0, f64::NAN];
    let result = hardswish_simd(&x.view());

    assert!(result[0].is_nan(), "Hardswish(NaN) should be NaN");
    assert!(result[2].is_nan(), "Hardswish(NaN) should be NaN");
    assert!(result[1].abs() < 1e-10);
}

/// Test Hardswish in MobileNetV3 use case
#[test]
fn test_hardswish_simd_mobilenetv3_use_case() {
    // Simulate activations in a MobileNetV3 layer
    let features = array![0.5_f64, 1.2, -0.8, 2.0, -1.5, 0.0, 3.5, -3.5];
    let activated = hardswish_simd(&features.view());

    // All results should be finite
    for i in 0..features.len() {
        assert!(
            activated[i].is_finite(),
            "Hardswish output should be finite"
        );
    }

    // Zero should produce zero
    assert!(activated[5].abs() < 1e-10);

    // Large positive should be identity
    assert!((activated[6] - 3.5).abs() < 1e-10);

    // Large negative should be zero
    assert!(activated[7].abs() < 1e-10);
}

// ============================================================================
// ============================================================================
// Sinc SIMD Tests
// ============================================================================

/// Test Sinc with f64 - basic behavior
#[test]
fn test_sinc_simd_f64_basic() {
    let x = array![0.0_f64, 0.5, 1.0, 2.0, 3.0, -1.0, -2.0];
    let result = sinc_simd(&x.view());

    // sinc(0) = 1
    assert!(
        (result[0] - 1.0).abs() < 1e-10,
        "sinc(0) should be 1, got {}",
        result[0]
    );

    // sinc(0.5) = sin(π/2) / (π/2) = 1 / (π/2) ≈ 0.6366
    let expected_05 = (std::f64::consts::FRAC_PI_2).sin() / std::f64::consts::FRAC_PI_2;
    assert!(
        (result[1] - expected_05).abs() < 1e-10,
        "sinc(0.5) should be {}, got {}",
        expected_05,
        result[1]
    );

    // sinc(n) = 0 for all non-zero integers n
    assert!(
        result[2].abs() < 1e-10,
        "sinc(1) should be 0, got {}",
        result[2]
    );
    assert!(
        result[3].abs() < 1e-10,
        "sinc(2) should be 0, got {}",
        result[3]
    );
    assert!(
        result[4].abs() < 1e-10,
        "sinc(3) should be 0, got {}",
        result[4]
    );

    // sinc is an even function: sinc(-x) = sinc(x)
    assert!(
        result[5].abs() < 1e-10,
        "sinc(-1) should be 0, got {}",
        result[5]
    );
    assert!(
        result[6].abs() < 1e-10,
        "sinc(-2) should be 0, got {}",
        result[6]
    );
}

/// Test Sinc with f32
#[test]
fn test_sinc_simd_f32_basic() {
    let x = array![0.0_f32, 0.5, 1.0, -1.0];
    let result = sinc_simd(&x.view());

    // sinc(0) = 1
    assert!((result[0] - 1.0).abs() < 1e-6, "sinc(0) should be 1");

    // sinc(1) = 0
    assert!(result[2].abs() < 1e-6, "sinc(1) should be 0");

    // sinc(-1) = 0
    assert!(result[3].abs() < 1e-6, "sinc(-1) should be 0");
}

/// Test Sinc empty array
#[test]
fn test_sinc_simd_empty() {
    let x = array![] as Array1<f64>;
    let result = sinc_simd(&x.view());
    assert!(result.is_empty(), "sinc of empty array should be empty");
}

/// Test Sinc even function property
#[test]
fn test_sinc_simd_even_function() {
    // sinc(-x) = sinc(x) for all x
    let x = array![-3.0_f64, -2.5, -1.5, -0.5, 0.0, 0.5, 1.5, 2.5, 3.0];
    let result = sinc_simd(&x.view());

    // Verify symmetry
    assert!(
        (result[0] - result[8]).abs() < 1e-10,
        "sinc(-3) should equal sinc(3)"
    );
    assert!(
        (result[1] - result[7]).abs() < 1e-10,
        "sinc(-2.5) should equal sinc(2.5)"
    );
    assert!(
        (result[2] - result[6]).abs() < 1e-10,
        "sinc(-1.5) should equal sinc(1.5)"
    );
    assert!(
        (result[3] - result[5]).abs() < 1e-10,
        "sinc(-0.5) should equal sinc(0.5)"
    );
}

/// Test Sinc large array (SIMD path)
#[test]
fn test_sinc_simd_large_array() {
    let n = 10000;
    let x: Array1<f64> = Array1::linspace(-5.0, 5.0, n);
    let result = sinc_simd(&x.view());

    assert_eq!(result.len(), n);

    // All values should be in range [-0.22, 1.0] (sinc is bounded)
    for i in 0..result.len() {
        assert!(
            result[i] >= -0.25 && result[i] <= 1.0,
            "sinc output should be bounded, got {} at index {}",
            result[i],
            i
        );
    }

    // Maximum is at x = 0
    let mid = n / 2;
    assert!(
        (result[mid] - 1.0).abs() < 0.01,
        "sinc(0) should be approximately 1"
    );
}

/// Test Sinc zeros at integers
#[test]
fn test_sinc_simd_zeros_at_integers() {
    // sinc(n) = 0 for all non-zero integers n
    let x = array![-5.0_f64, -4.0, -3.0, -2.0, -1.0, 1.0, 2.0, 3.0, 4.0, 5.0];
    let result = sinc_simd(&x.view());

    for i in 0..result.len() {
        assert!(
            result[i].abs() < 1e-10,
            "sinc({}) should be 0, got {}",
            x[i],
            result[i]
        );
    }
}

/// Test Sinc NaN handling
#[test]
fn test_sinc_simd_nan() {
    let x = array![f64::NAN, 0.0, f64::NAN, 1.0];
    let result = sinc_simd(&x.view());

    assert!(result[0].is_nan(), "sinc(NaN) should be NaN");
    assert!(result[2].is_nan(), "sinc(NaN) should be NaN");
    assert!((result[1] - 1.0).abs() < 1e-10);
    assert!(result[3].abs() < 1e-10);
}

/// Test Sinc numerical stability near zero
#[test]
fn test_sinc_simd_near_zero() {
    // Test that we handle very small x values correctly (avoid 0/0)
    let x = array![1e-10_f64, 1e-15_f64, 1e-20_f64, -1e-10, -1e-15, -1e-20];
    let result = sinc_simd(&x.view());

    // All should be very close to 1
    for i in 0..result.len() {
        assert!(
            (result[i] - 1.0).abs() < 1e-8,
            "sinc({}) should be approximately 1, got {}",
            x[i],
            result[i]
        );
    }
}

/// Test Sinc for signal processing use case
#[test]
fn test_sinc_simd_signal_processing_use_case() {
    // In ideal low-pass filter design, sinc is the impulse response
    // Test the typical values used in reconstruction
    let n = 21;
    let x: Array1<f64> = Array1::linspace(-10.0, 10.0, n);
    let result = sinc_simd(&x.view());

    // All results should be finite
    for i in 0..n {
        assert!(
            result[i].is_finite(),
            "sinc output should be finite for input {}",
            x[i]
        );
    }

    // Center should be 1
    assert!(
        (result[10] - 1.0).abs() < 1e-10,
        "sinc(0) should be 1 in the center"
    );

    // Integer points should be 0 (except center)
    for i in [
        0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20,
    ] {
        if (x[i] - x[i].round()).abs() < 1e-10 {
            assert!(
                result[i].abs() < 1e-10,
                "sinc({}) should be 0, got {}",
                x[i],
                result[i]
            );
        }
    }
}

// =============================================================================
// =============================================================================
// Log-Softmax Tests (Phase 58)
// =============================================================================

use scirs2_core::ndarray_ext::elementwise::log_softmax_simd;

/// Test Log-Softmax basic f64
#[test]
fn test_log_softmax_simd_basic_f64() {
    let x = array![1.0_f64, 2.0, 3.0];
    let result = log_softmax_simd(&x.view());

    // exp(log_softmax) should sum to 1
    let sum_exp: f64 = result.mapv(|v| v.exp()).sum();
    assert!(
        (sum_exp - 1.0).abs() < 1e-10,
        "exp(log_softmax) should sum to 1, got {}",
        sum_exp
    );

    // All outputs should be <= 0 (log of probabilities <= 1)
    for (i, &v) in result.iter().enumerate() {
        assert!(v <= 0.0, "log_softmax[{}] = {} should be <= 0", i, v);
    }
}

/// Test Log-Softmax basic f32
#[test]
fn test_log_softmax_simd_basic_f32() {
    let x = array![1.0_f32, 2.0, 3.0, 4.0];
    let result = log_softmax_simd(&x.view());

    // exp(log_softmax) should sum to 1
    let sum_exp: f32 = result.mapv(|v| v.exp()).sum();
    assert!(
        (sum_exp - 1.0).abs() < 1e-5,
        "exp(log_softmax) should sum to 1, got {}",
        sum_exp
    );

    // Largest input should have log_softmax closest to 0
    let max_idx = 3; // x[3] = 4.0 is largest
    for i in 0..result.len() {
        if i != max_idx {
            assert!(
                result[i] < result[max_idx],
                "Largest input should have largest log_softmax output"
            );
        }
    }
}

/// Test Log-Softmax empty array
#[test]
fn test_log_softmax_simd_empty() {
    let x: Array1<f64> = array![];
    let result = log_softmax_simd(&x.view());
    assert_eq!(result.len(), 0, "Empty input should give empty output");
}

/// Test Log-Softmax large array (SIMD path)
#[test]
fn test_log_softmax_simd_large_array() {
    let n = 10000;
    let x: Array1<f64> = Array1::linspace(-10.0, 10.0, n);
    let result = log_softmax_simd(&x.view());

    assert_eq!(result.len(), n, "Output should have same length as input");

    // exp(log_softmax) should sum to 1
    let sum_exp: f64 = result.mapv(|v| v.exp()).sum();
    assert!(
        (sum_exp - 1.0).abs() < 1e-8,
        "exp(log_softmax) should sum to 1, got {}",
        sum_exp
    );
}

/// Test Log-Softmax shift invariance
#[test]
fn test_log_softmax_simd_shift_invariance() {
    // log_softmax(x + c) = log_softmax(x) for any constant c
    let x = array![1.0_f64, 2.0, 3.0, 4.0, 5.0];
    let x_shifted = &x + 100.0;

    let result1 = log_softmax_simd(&x.view());
    let result2 = log_softmax_simd(&x_shifted.view());

    for i in 0..x.len() {
        assert!(
            (result1[i] - result2[i]).abs() < 1e-10,
            "log_softmax should be shift-invariant, but got {} vs {} at index {}",
            result1[i],
            result2[i],
            i
        );
    }
}

/// Test Log-Softmax equals log(softmax)
#[test]
fn test_log_softmax_simd_equals_log_softmax() {
    use scirs2_core::simd_ops::SimdUnifiedOps;

    let x = array![0.5_f64, 1.5, 2.5, -0.5, -1.5];
    let log_sm = log_softmax_simd(&x.view());

    // Compute softmax and take log
    let softmax = f64::simd_softmax(&x.view());
    let log_of_softmax = softmax.mapv(|v| v.ln());

    for i in 0..x.len() {
        assert!(
            (log_sm[i] - log_of_softmax[i]).abs() < 1e-10,
            "log_softmax should equal log(softmax), got {} vs {} at index {}",
            log_sm[i],
            log_of_softmax[i],
            i
        );
    }
}

/// Test Log-Softmax NaN handling
#[test]
fn test_log_softmax_simd_nan() {
    let x = array![1.0_f64, f64::NAN, 3.0];
    let result = log_softmax_simd(&x.view());

    // When there's a NaN in input, the log_sum_exp will be NaN,
    // so all outputs will be NaN
    for &v in result.iter() {
        assert!(v.is_nan(), "log_softmax with NaN input should produce NaN");
    }
}

/// Test Log-Softmax numerical stability for large values
#[test]
fn test_log_softmax_simd_large_values() {
    // Large positive values that would overflow in naive softmax
    let x = array![1000.0_f64, 1001.0, 1002.0];
    let result = log_softmax_simd(&x.view());

    // Should still be valid (not NaN or -Inf)
    for (i, &v) in result.iter().enumerate() {
        assert!(
            v.is_finite(),
            "log_softmax should be stable for large values, got {} at index {}",
            v,
            i
        );
    }

    // exp(log_softmax) should sum to 1
    let sum_exp: f64 = result.mapv(|v| v.exp()).sum();
    assert!(
        (sum_exp - 1.0).abs() < 1e-10,
        "exp(log_softmax) should sum to 1 even for large inputs, got {}",
        sum_exp
    );
}

/// Test Log-Softmax numerical stability for large negative values
#[test]
fn test_log_softmax_simd_large_negative() {
    // Large negative values
    let x = array![-1000.0_f64, -999.0, -998.0];
    let result = log_softmax_simd(&x.view());

    // Should still be valid
    for (i, &v) in result.iter().enumerate() {
        assert!(
            v.is_finite(),
            "log_softmax should be stable for large negative values, got {} at index {}",
            v,
            i
        );
    }

    // exp(log_softmax) should sum to 1
    let sum_exp: f64 = result.mapv(|v| v.exp()).sum();
    assert!(
        (sum_exp - 1.0).abs() < 1e-10,
        "exp(log_softmax) should sum to 1 even for large negative inputs, got {}",
        sum_exp
    );
}

/// Test Log-Softmax for cross-entropy loss use case
#[test]
fn test_log_softmax_simd_cross_entropy_use_case() {
    // Simulate classification with 10 classes
    let logits = array![1.0_f64, 0.5, 2.5, -1.0, 0.0, 3.0, 1.5, -0.5, 0.25, 0.75];
    let log_probs = log_softmax_simd(&logits.view());

    // One-hot target (class 5 is correct)
    let target = array![0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0];

    // Cross-entropy loss = -sum(target * log_probs)
    let cross_entropy: f64 = -(0..10).map(|i| target[i] * log_probs[i]).sum::<f64>();

    // Loss should be positive (since log_probs are negative)
    assert!(
        cross_entropy > 0.0,
        "Cross-entropy loss should be positive, got {}",
        cross_entropy
    );

    // Loss should equal -log_probs[5] (the correct class)
    assert!(
        (cross_entropy - (-log_probs[5])).abs() < 1e-10,
        "Cross-entropy should equal -log_probs[correct_class], got {} vs {}",
        cross_entropy,
        -log_probs[5]
    );
}

/// Test Log-Softmax for Transformer use case
#[test]
fn test_log_softmax_simd_transformer_use_case() {
    // Simulate vocabulary logits in a language model (vocab size = 50)
    let n = 50;
    let mut logits = Array1::zeros(n);
    for i in 0..n {
        logits[i] = (i as f64 - 25.0) * 0.1; // Some variation
    }

    let log_probs = log_softmax_simd(&logits.view());

    // All log probabilities should be finite
    for (i, &lp) in log_probs.iter().enumerate() {
        assert!(
            lp.is_finite(),
            "Log probability at index {} should be finite, got {}",
            i,
            lp
        );
        assert!(lp <= 0.0, "Log probability should be <= 0, got {}", lp);
    }

    // Perplexity = exp(-mean(log_probs)) should be reasonable
    let mean_log_prob = log_probs.mean().expect("Test: operation failed");
    let perplexity = (-mean_log_prob).exp();
    assert!(
        perplexity > 0.0 && perplexity < f64::INFINITY,
        "Perplexity should be finite positive, got {}",
        perplexity
    );
}

// =============================================================================
