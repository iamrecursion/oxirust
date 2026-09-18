//! Activation function SIMD tests Part 1: sigmoid, gelu, swish, softplus, mish

use scirs2_core::ndarray::{array, Array1};
use scirs2_core::ndarray_ext::elementwise::{
    gelu_simd, mish_simd, sigmoid_simd, softplus_simd, swish_simd,
};
#[cfg(feature = "random")]
use scirs2_core::random::{thread_rng, Distribution, Uniform};

// Sigmoid (Logistic) Function Tests
// =============================================================================

/// Test sigmoid(0) = 0.5
#[test]
fn test_sigmoid_simd_f64_at_zero() {
    let x = array![0.0_f64];
    let result = sigmoid_simd(&x.view());
    assert!(
        (result[0] - 0.5).abs() < 1e-15,
        "sigmoid(0) should be 0.5, got {}",
        result[0]
    );
}

/// Test sigmoid basic values
#[test]
fn test_sigmoid_simd_f64_basic() {
    let x = array![0.0_f64, 1.0, -1.0, 2.0, -2.0];
    let result = sigmoid_simd(&x.view());

    // sigmoid(0) = 0.5
    assert!(
        (result[0] - 0.5).abs() < 1e-15,
        "sigmoid(0) should be 0.5, got {}",
        result[0]
    );

    // sigmoid(1) ≈ 0.7310585786
    let expected_1 = 1.0 / (1.0 + (-1.0_f64).exp());
    assert!(
        (result[1] - expected_1).abs() < 1e-10,
        "sigmoid(1) should be {}, got {}",
        expected_1,
        result[1]
    );

    // sigmoid(-1) = 1 - sigmoid(1)
    assert!(
        (result[2] - (1.0 - expected_1)).abs() < 1e-10,
        "sigmoid(-1) should be {}, got {}",
        1.0 - expected_1,
        result[2]
    );

    // Verify σ(-x) = 1 - σ(x) property
    assert!(
        (result[1] + result[2] - 1.0).abs() < 1e-10,
        "sigmoid(1) + sigmoid(-1) should be 1, got {}",
        result[1] + result[2]
    );
    assert!(
        (result[3] + result[4] - 1.0).abs() < 1e-10,
        "sigmoid(2) + sigmoid(-2) should be 1, got {}",
        result[3] + result[4]
    );
}

/// Test sigmoid range is (0, 1) for moderate values
#[test]
fn test_sigmoid_simd_f64_range() {
    // For moderate values, sigmoid is strictly in (0, 1)
    let x = array![-10.0_f64, -5.0, -1.0, 0.0, 1.0, 5.0, 10.0];
    let result = sigmoid_simd(&x.view());

    for i in 0..x.len() {
        assert!(
            result[i] > 0.0 && result[i] < 1.0,
            "sigmoid({}) should be in (0,1), got {}",
            x[i],
            result[i]
        );
    }

    // For extreme values, check bounds (may round to 0 or 1)
    let x_extreme = array![-100.0_f64, 100.0];
    let result_extreme = sigmoid_simd(&x_extreme.view());
    for i in 0..x_extreme.len() {
        assert!(
            result_extreme[i] >= 0.0 && result_extreme[i] <= 1.0,
            "sigmoid({}) should be in [0,1], got {}",
            x_extreme[i],
            result_extreme[i]
        );
    }
}

/// Test sigmoid symmetry: σ(-x) = 1 - σ(x)
#[test]
fn test_sigmoid_simd_f64_symmetry() {
    let x_pos = array![0.5_f64, 1.0, 2.0, 5.0, 10.0];
    let x_neg = array![-0.5_f64, -1.0, -2.0, -5.0, -10.0];
    let result_pos = sigmoid_simd(&x_pos.view());
    let result_neg = sigmoid_simd(&x_neg.view());

    for i in 0..x_pos.len() {
        assert!(
            (result_pos[i] + result_neg[i] - 1.0).abs() < 1e-10,
            "sigmoid({}) + sigmoid({}) should be 1, got {} + {} = {}",
            x_pos[i],
            x_neg[i],
            result_pos[i],
            result_neg[i],
            result_pos[i] + result_neg[i]
        );
    }
}

/// Test sigmoid numerical stability for large positive values
#[test]
fn test_sigmoid_simd_f64_large_positive() {
    let x = array![50.0_f64, 100.0, 500.0, 700.0];
    let result = sigmoid_simd(&x.view());

    for i in 0..x.len() {
        // Should be very close to 1 but not exactly 1
        assert!(
            result[i] > 0.999,
            "sigmoid({}) should be close to 1, got {}",
            x[i],
            result[i]
        );
        assert!(
            result[i] <= 1.0,
            "sigmoid({}) should be <= 1, got {}",
            x[i],
            result[i]
        );
    }
}

/// Test sigmoid numerical stability for large negative values
#[test]
fn test_sigmoid_simd_f64_large_negative() {
    let x = array![-50.0_f64, -100.0, -500.0, -700.0];
    let result = sigmoid_simd(&x.view());

    for i in 0..x.len() {
        // Should be very close to 0 but not exactly 0
        assert!(
            result[i] < 0.001,
            "sigmoid({}) should be close to 0, got {}",
            x[i],
            result[i]
        );
        assert!(
            result[i] >= 0.0,
            "sigmoid({}) should be >= 0, got {}",
            x[i],
            result[i]
        );
    }
}

/// Test sigmoid with infinity
#[test]
fn test_sigmoid_simd_f64_infinity() {
    let x = array![f64::INFINITY, f64::NEG_INFINITY];
    let result = sigmoid_simd(&x.view());

    // sigmoid(+∞) = 1
    assert!(
        (result[0] - 1.0).abs() < 1e-10,
        "sigmoid(+∞) should be 1, got {}",
        result[0]
    );

    // sigmoid(-∞) = 0
    assert!(
        result[1].abs() < 1e-10,
        "sigmoid(-∞) should be 0, got {}",
        result[1]
    );
}

/// Test sigmoid with NaN
#[test]
fn test_sigmoid_simd_f64_nan() {
    let x = array![f64::NAN];
    let result = sigmoid_simd(&x.view());

    assert!(result[0].is_nan(), "sigmoid(NaN) should be NaN");
}

/// Test sigmoid with f32
#[test]
fn test_sigmoid_simd_f32_basic() {
    let x = array![0.0_f32, 1.0, -1.0, 5.0, -5.0];
    let result = sigmoid_simd(&x.view());

    // sigmoid(0) = 0.5
    assert!(
        (result[0] - 0.5).abs() < 1e-6,
        "sigmoid(0) should be 0.5, got {}",
        result[0]
    );

    // Verify symmetry
    assert!(
        (result[1] + result[2] - 1.0).abs() < 1e-5,
        "sigmoid(1) + sigmoid(-1) should be 1, got {}",
        result[1] + result[2]
    );
    assert!(
        (result[3] + result[4] - 1.0).abs() < 1e-5,
        "sigmoid(5) + sigmoid(-5) should be 1, got {}",
        result[3] + result[4]
    );
}

/// Test sigmoid empty array
#[test]
fn test_sigmoid_simd_empty() {
    let x = array![] as Array1<f64>;
    let result = sigmoid_simd(&x.view());
    assert!(result.is_empty(), "sigmoid of empty array should be empty");
}

/// Test sigmoid large array (SIMD path)
#[test]
fn test_sigmoid_simd_large_array() {
    let n = 10000;
    let x: Array1<f64> = Array1::linspace(-10.0, 10.0, n);
    let result = sigmoid_simd(&x.view());

    assert_eq!(result.len(), n);

    // Verify monotonically increasing
    for i in 1..n {
        assert!(
            result[i] >= result[i - 1],
            "sigmoid should be monotonically increasing: sigmoid({}) = {} < sigmoid({}) = {}",
            x[i - 1],
            result[i - 1],
            x[i],
            result[i]
        );
    }

    // Verify range (allow boundary values for extreme inputs)
    for i in 0..n {
        assert!(
            result[i] >= 0.0 && result[i] <= 1.0,
            "sigmoid({}) out of range: {}",
            x[i],
            result[i]
        );
    }

    // Verify symmetry using explicitly symmetric values
    let symmetric_x = array![-5.0_f64, -2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0, 5.0];
    let symmetric_result = sigmoid_simd(&symmetric_x.view());
    // Check pairs: sigmoid(-x) + sigmoid(x) = 1
    assert!((symmetric_result[0] + symmetric_result[8] - 1.0).abs() < 1e-10); // -5, 5
    assert!((symmetric_result[1] + symmetric_result[7] - 1.0).abs() < 1e-10); // -2, 2
    assert!((symmetric_result[2] + symmetric_result[6] - 1.0).abs() < 1e-10); // -1, 1
    assert!((symmetric_result[3] + symmetric_result[5] - 1.0).abs() < 1e-10); // -0.5, 0.5
    assert!((symmetric_result[4] - 0.5).abs() < 1e-10); // 0
}

/// Test sigmoid derivative property: σ'(x) = σ(x)(1 - σ(x))
/// Using numerical differentiation to verify
#[test]
fn test_sigmoid_simd_derivative_property() {
    let h = 1e-8_f64;
    let x_vals = [0.0_f64, 0.5, 1.0, -1.0, 2.0];

    for &x in &x_vals {
        let x_arr = array![x];
        let x_plus_h = array![x + h];
        let x_minus_h = array![x - h];

        let s = sigmoid_simd(&x_arr.view())[0];
        let s_plus = sigmoid_simd(&x_plus_h.view())[0];
        let s_minus = sigmoid_simd(&x_minus_h.view())[0];

        // Numerical derivative
        let numerical_deriv = (s_plus - s_minus) / (2.0 * h);

        // Analytical derivative: σ(x)(1 - σ(x))
        let analytical_deriv = s * (1.0 - s);

        assert!(
            (numerical_deriv - analytical_deriv).abs() < 1e-5,
            "Derivative at x={}: numerical {} vs analytical {}",
            x,
            numerical_deriv,
            analytical_deriv
        );
    }
}

/// Test logistic regression example: probability = sigmoid(weights · features + bias)
#[test]
fn test_sigmoid_simd_logistic_regression() {
    // Typical logistic regression scenario
    let logits = array![0.0_f64, -2.0, 2.0, -5.0, 5.0];
    let probs = sigmoid_simd(&logits.view());

    // Verify probabilities sum-related properties
    // For balanced classes (logit=0), probability should be 0.5
    assert!((probs[0] - 0.5).abs() < 1e-10);

    // Negative logits -> probability < 0.5
    assert!(probs[1] < 0.5);
    assert!(probs[3] < 0.5);

    // Positive logits -> probability > 0.5
    assert!(probs[2] > 0.5);
    assert!(probs[4] > 0.5);
}

// =============================================================================
// =============================================================================
// GELU (Gaussian Error Linear Unit) Tests
// =============================================================================

/// Test GELU(0) = 0
#[test]
fn test_gelu_simd_f64_at_zero() {
    let x = array![0.0_f64];
    let result = gelu_simd(&x.view());
    assert!(
        result[0].abs() < 1e-15,
        "GELU(0) should be 0, got {}",
        result[0]
    );
}

/// Test GELU basic values
#[test]
fn test_gelu_simd_f64_basic() {
    let x = array![0.0_f64, 1.0, -1.0, 2.0, -2.0];
    let result = gelu_simd(&x.view());

    // GELU(0) = 0
    assert!(
        result[0].abs() < 1e-15,
        "GELU(0) should be 0, got {}",
        result[0]
    );

    // GELU(1) ≈ 0.8413 (1 * Φ(1) where Φ(1) ≈ 0.8413)
    // More precisely: 1 * 0.5 * (1 + erf(1/√2)) ≈ 0.8413447460685429
    let expected_1 = 0.8413447460685429_f64;
    assert!(
        (result[1] - expected_1).abs() < 1e-6,
        "GELU(1) should be approximately {}, got {}",
        expected_1,
        result[1]
    );

    // GELU(-1) ≈ -0.1587 (-1 * Φ(-1) where Φ(-1) ≈ 0.1587)
    let expected_neg1 = -0.15865525393145707_f64;
    assert!(
        (result[2] - expected_neg1).abs() < 1e-6,
        "GELU(-1) should be approximately {}, got {}",
        expected_neg1,
        result[2]
    );

    // GELU(2) > GELU(1) (monotonically increasing for positive x)
    assert!(
        result[3] > result[1],
        "GELU should be increasing: GELU(2)={} should be > GELU(1)={}",
        result[3],
        result[1]
    );
}

/// Test GELU asymptotic behavior: GELU(x) ≈ x for large positive x
#[test]
fn test_gelu_simd_f64_large_positive() {
    let x = array![5.0_f64, 10.0, 20.0];
    let result = gelu_simd(&x.view());

    for i in 0..x.len() {
        // For large positive x, GELU(x) ≈ x because Φ(x) ≈ 1
        assert!(
            (result[i] - x[i]).abs() < 0.01,
            "GELU({}) should be approximately {}, got {}",
            x[i],
            x[i],
            result[i]
        );
    }
}

/// Test GELU asymptotic behavior: GELU(x) ≈ 0 for large negative x
#[test]
fn test_gelu_simd_f64_large_negative() {
    let x = array![-5.0_f64, -10.0, -20.0];
    let result = gelu_simd(&x.view());

    for i in 0..x.len() {
        // For large negative x, GELU(x) ≈ 0 because Φ(x) ≈ 0
        assert!(
            result[i].abs() < 0.01,
            "GELU({}) should be approximately 0, got {}",
            x[i],
            result[i]
        );
    }
}

/// Test GELU is smooth (no discontinuities)
#[test]
fn test_gelu_simd_f64_smoothness() {
    // Check that GELU is continuous around 0
    let eps = 1e-6_f64;
    let x = array![-eps, 0.0, eps];
    let result = gelu_simd(&x.view());

    // Values should be close to each other (smooth transition)
    assert!(
        (result[1] - result[0]).abs() < 0.001,
        "GELU should be smooth at 0: GELU({})={}, GELU(0)={}",
        -eps,
        result[0],
        result[1]
    );
    assert!(
        (result[2] - result[1]).abs() < 0.001,
        "GELU should be smooth at 0: GELU(0)={}, GELU({})={}",
        result[1],
        eps,
        result[2]
    );
}

/// Test GELU monotonicity: GELU is monotonically increasing for x > 0
#[test]
fn test_gelu_simd_f64_monotonicity() {
    let n = 100;
    let x: Array1<f64> = Array1::linspace(0.0, 10.0, n);
    let result = gelu_simd(&x.view());

    for i in 1..n {
        assert!(
            result[i] >= result[i - 1],
            "GELU should be monotonically increasing for positive x: GELU({})={} < GELU({})={}",
            x[i - 1],
            result[i - 1],
            x[i],
            result[i]
        );
    }
}

/// Test GELU with NaN
#[test]
fn test_gelu_simd_f64_nan() {
    let x = array![f64::NAN];
    let result = gelu_simd(&x.view());

    assert!(result[0].is_nan(), "GELU(NaN) should be NaN");
}

/// Test GELU with f32
#[test]
fn test_gelu_simd_f32_basic() {
    let x = array![0.0_f32, 1.0, -1.0, 2.0, -2.0];
    let result = gelu_simd(&x.view());

    // GELU(0) = 0
    assert!(
        result[0].abs() < 1e-6,
        "GELU(0) should be 0, got {}",
        result[0]
    );

    // GELU(1) > 0
    assert!(result[1] > 0.0, "GELU(1) should be positive");

    // GELU(-1) < 0
    assert!(result[2] < 0.0, "GELU(-1) should be negative");
}

/// Test GELU empty array
#[test]
fn test_gelu_simd_empty() {
    let x = array![] as Array1<f64>;
    let result = gelu_simd(&x.view());
    assert!(result.is_empty(), "GELU of empty array should be empty");
}

/// Test GELU large array (SIMD path)
#[test]
fn test_gelu_simd_large_array() {
    let n = 10000;
    let x: Array1<f64> = Array1::linspace(-5.0, 5.0, n);
    let result = gelu_simd(&x.view());

    assert_eq!(result.len(), n);

    // Verify basic properties
    for i in 0..n {
        // GELU should be between x and 0 for negative x
        if x[i] < 0.0 {
            assert!(
                result[i] >= x[i] && result[i] <= 0.0,
                "For x={}, GELU={} should be between {} and 0",
                x[i],
                result[i],
                x[i]
            );
        }
        // GELU should be between 0 and x for positive x
        if x[i] > 0.0 {
            assert!(
                result[i] >= 0.0 && result[i] <= x[i],
                "For x={}, GELU={} should be between 0 and {}",
                x[i],
                result[i],
                x[i]
            );
        }
    }
}

/// Test GELU vs ReLU relationship: GELU is a smooth approximation of ReLU
#[test]
fn test_gelu_simd_vs_relu() {
    // ReLU(x) = max(0, x)
    // GELU is a smooth version of ReLU
    let x = array![-3.0_f64, -1.0, 0.0, 1.0, 3.0];
    let gelu_result = gelu_simd(&x.view());

    // For large positive x, GELU ≈ x (like ReLU)
    assert!(
        (gelu_result[4] - x[4]).abs() < 0.01,
        "GELU(3) should be close to 3"
    );

    // For large negative x, GELU ≈ 0 (like ReLU)
    assert!(gelu_result[0].abs() < 0.01, "GELU(-3) should be close to 0");

    // At 0, both are 0
    assert!(gelu_result[2].abs() < 1e-10, "GELU(0) should be 0");
}

/// Test GELU in Transformer-like computation
#[test]
fn test_gelu_simd_transformer_use_case() {
    // Simulate feed-forward layer in Transformer: GELU(W*x + b)
    let hidden = array![0.5_f64, 1.2, -0.8, 2.0, -1.5];
    let activated = gelu_simd(&hidden.view());

    // All results should be finite
    for i in 0..hidden.len() {
        assert!(
            activated[i].is_finite(),
            "GELU output should be finite for input {}",
            hidden[i]
        );
    }

    // Positive values should remain positive (scaled down)
    assert!(activated[0] > 0.0);
    assert!(activated[1] > 0.0);
    assert!(activated[3] > 0.0);
}

// ============================================================================
// ============================================================================
// Swish (SiLU) SIMD Tests
// ============================================================================

/// Test Swish(0) = 0
#[test]
fn test_swish_simd_f64_at_zero() {
    let x = array![0.0_f64];
    let result = swish_simd(&x.view());
    assert!(
        result[0].abs() < 1e-15,
        "Swish(0) should be 0, got {}",
        result[0]
    );
}

/// Test Swish basic values
#[test]
fn test_swish_simd_f64_basic() {
    let x = array![0.0_f64, 1.0, -1.0, 2.0, -2.0];
    let result = swish_simd(&x.view());

    // Swish(x) = x * sigmoid(x) = x / (1 + exp(-x))
    // Swish(0) = 0 * sigmoid(0) = 0 * 0.5 = 0
    assert!(result[0].abs() < 1e-10, "Swish(0) should be 0");

    // Swish(1) = 1 * sigmoid(1) = 1 / (1 + e^-1) ≈ 0.7310585786
    let expected_swish_1 = 1.0 / (1.0 + (-1.0_f64).exp());
    assert!(
        (result[1] - expected_swish_1).abs() < 1e-10,
        "Swish(1) should be approximately {}, got {}",
        expected_swish_1,
        result[1]
    );

    // Swish(-1) = -1 * sigmoid(-1) = -1 / (1 + e) ≈ -0.2689414214
    let expected_swish_neg1 = -1.0 / (1.0 + 1.0_f64.exp());
    assert!(
        (result[2] - expected_swish_neg1).abs() < 1e-10,
        "Swish(-1) should be approximately {}, got {}",
        expected_swish_neg1,
        result[2]
    );

    // Swish(2) = 2 * sigmoid(2) ≈ 1.7615942
    let expected_swish_2 = 2.0 / (1.0 + (-2.0_f64).exp());
    assert!(
        (result[3] - expected_swish_2).abs() < 1e-10,
        "Swish(2) should be approximately {}, got {}",
        expected_swish_2,
        result[3]
    );

    // Swish(-2) = -2 * sigmoid(-2) ≈ -0.2384058
    let expected_swish_neg2 = -2.0 / (1.0 + 2.0_f64.exp());
    assert!(
        (result[4] - expected_swish_neg2).abs() < 1e-10,
        "Swish(-2) should be approximately {}, got {}",
        expected_swish_neg2,
        result[4]
    );
}

/// Test Swish asymptotic behavior: Swish(x) ≈ x for large positive x
#[test]
fn test_swish_simd_f64_large_positive() {
    let x = array![5.0_f64, 10.0, 20.0];
    let result = swish_simd(&x.view());

    // For large x, sigmoid(x) ≈ 1, so Swish(x) ≈ x
    for i in 0..x.len() {
        // The relative error decreases as x increases
        let relative_error = (result[i] - x[i]).abs() / x[i];
        assert!(
            relative_error < 0.01,
            "Swish({}) ≈ {}, relative error should be small: {}",
            x[i],
            result[i],
            relative_error
        );
    }
}

/// Test Swish asymptotic behavior: Swish(x) ≈ 0 for large negative x
#[test]
fn test_swish_simd_f64_large_negative() {
    let x = array![-5.0_f64, -10.0, -20.0];
    let result = swish_simd(&x.view());

    // For large negative x, sigmoid(x) ≈ 0, so Swish(x) ≈ 0
    for i in 0..x.len() {
        assert!(
            result[i].abs() < 0.1,
            "Swish({}) should be approximately 0, got {}",
            x[i],
            result[i]
        );
    }
}

/// Test Swish is smooth (no discontinuities)
#[test]
fn test_swish_simd_f64_smoothness() {
    // Check that Swish is continuous around 0
    let eps = 1e-6_f64;
    let x = array![-eps, 0.0, eps];
    let result = swish_simd(&x.view());

    // Values should be close to each other (continuous)
    let diff_left = (result[0] - result[1]).abs();
    let diff_right = (result[2] - result[1]).abs();

    assert!(
        diff_left < 1e-5,
        "Swish should be continuous at 0 from the left"
    );
    assert!(
        diff_right < 1e-5,
        "Swish should be continuous at 0 from the right"
    );
}

/// Test Swish non-monotonicity: Swish has a global minimum around x ≈ -1.278
#[test]
fn test_swish_simd_f64_global_minimum() {
    // Swish has a global minimum at approximately x = -1.278
    // Minimum value ≈ -0.2784
    let x = array![-1.278_f64, -1.5, -1.0, -2.0];
    let result = swish_simd(&x.view());

    // The minimum should be around -0.278
    let min_value = result[0];
    assert!(
        min_value < -0.27 && min_value > -0.29,
        "Swish global minimum should be around -0.278, got {}",
        min_value
    );

    // Values at other points should be >= minimum (within numerical tolerance)
    for i in 1..result.len() {
        assert!(
            result[i] >= min_value - 0.01,
            "Swish({}) = {} should be >= minimum {}",
            x[i],
            result[i],
            min_value
        );
    }
}

/// Test Swish with NaN
#[test]
fn test_swish_simd_f64_nan() {
    let x = array![f64::NAN];
    let result = swish_simd(&x.view());

    assert!(result[0].is_nan(), "Swish(NaN) should be NaN");
}

/// Test Swish with f32
#[test]
fn test_swish_simd_f32_basic() {
    let x = array![0.0_f32, 1.0, -1.0, 2.0, -2.0];
    let result = swish_simd(&x.view());

    // Swish(0) = 0
    assert!(result[0].abs() < 1e-6, "Swish(0) should be 0");

    // Swish(1) ≈ 0.7311
    let expected_swish_1 = 1.0_f32 / (1.0 + (-1.0_f32).exp());
    assert!(
        (result[1] - expected_swish_1).abs() < 1e-5,
        "Swish(1) should be approximately {}",
        expected_swish_1
    );

    // Swish(-1) ≈ -0.2689
    let expected_swish_neg1 = -1.0_f32 / (1.0 + 1.0_f32.exp());
    assert!(
        (result[2] - expected_swish_neg1).abs() < 1e-5,
        "Swish(-1) should be approximately {}",
        expected_swish_neg1
    );
}

/// Test Swish empty array
#[test]
fn test_swish_simd_empty() {
    let x = array![] as Array1<f64>;
    let result = swish_simd(&x.view());
    assert!(result.is_empty(), "Swish of empty array should be empty");
}

/// Test Swish large array (SIMD path)
#[test]
fn test_swish_simd_large_array() {
    let n = 10000;
    let x: Array1<f64> = Array1::linspace(-5.0, 5.0, n);
    let result = swish_simd(&x.view());

    // Check length
    assert_eq!(result.len(), n, "Result should have same length as input");

    // Check symmetry property: Swish(-x) = -x * sigmoid(-x)
    // Note: Swish is NOT point-symmetric, but has a specific relationship
    let mid = n / 2;
    for i in 0..100 {
        let pos_idx = mid + i;
        let neg_idx = mid - i;
        if neg_idx > 0 && pos_idx < n {
            let x_pos = x[pos_idx];
            let x_neg = x[neg_idx];
            // Verify both values are computed correctly
            let expected_pos = x_pos / (1.0 + (-x_pos).exp());
            let expected_neg = x_neg / (1.0 + (-x_neg).exp());
            assert!(
                (result[pos_idx] - expected_pos).abs() < 1e-10,
                "Swish({}) should be {}",
                x_pos,
                expected_pos
            );
            assert!(
                (result[neg_idx] - expected_neg).abs() < 1e-10,
                "Swish({}) should be {}",
                x_neg,
                expected_neg
            );
        }
    }
}

/// Test Swish relation to sigmoid: Swish(x) = x * sigmoid(x)
#[test]
fn test_swish_simd_sigmoid_relation() {
    let x = array![-2.0_f64, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0];
    let swish_result = swish_simd(&x.view());
    let sigmoid_result = sigmoid_simd(&x.view());

    for i in 0..x.len() {
        let expected = x[i] * sigmoid_result[i];
        assert!(
            (swish_result[i] - expected).abs() < 1e-10,
            "Swish({}) should equal {} * sigmoid({}) = {}",
            x[i],
            x[i],
            x[i],
            expected
        );
    }
}

/// Test Swish in EfficientNet-like computation
#[test]
fn test_swish_simd_efficientnet_use_case() {
    // Simulate activation in EfficientNet: Swish applied after convolution
    let features = array![0.5_f64, 1.2, -0.8, 2.0, -1.5, 0.0, 3.0, -3.0];
    let activated = swish_simd(&features.view());

    // All results should be finite
    for i in 0..features.len() {
        assert!(
            activated[i].is_finite(),
            "Swish output should be finite for input {}",
            features[i]
        );
    }

    // Positive values should produce positive outputs
    assert!(activated[0] > 0.0);
    assert!(activated[1] > 0.0);
    assert!(activated[3] > 0.0);
    assert!(activated[6] > 0.0);

    // Zero should produce zero
    assert!(activated[5].abs() < 1e-10);

    // Negative values should produce small negative or positive outputs
    // (depending on how negative they are)
    for i in 0..features.len() {
        // All outputs should be >= global minimum ≈ -0.278
        assert!(
            activated[i] > -0.3,
            "Swish output should be >= -0.278, got {}",
            activated[i]
        );
    }
}

/// Test Swish vs ReLU: Swish is smoother than ReLU
#[test]
fn test_swish_simd_vs_relu() {
    let x = array![-3.0_f64, -1.0, 0.0, 1.0, 3.0];
    let swish_result = swish_simd(&x.view());

    // ReLU(x) = max(0, x)
    // Unlike ReLU which is 0 for all x < 0, Swish has a small negative region
    // and smoothly transitions to positive

    // For x < 0, Swish is small but not exactly 0
    assert!(swish_result[0] < 0.0 && swish_result[0] > -0.3);
    assert!(swish_result[1] < 0.0 && swish_result[1] > -0.3);

    // At x = 0, both Swish and ReLU are 0
    assert!(swish_result[2].abs() < 1e-10);

    // For x > 0, Swish is close to x (slightly less)
    assert!(swish_result[3] > 0.0 && swish_result[3] < 1.0);
    assert!(swish_result[4] > 0.0 && swish_result[4] < 3.0);
}

/// Test Swish derivative approximation
#[test]
fn test_swish_simd_derivative() {
    // Swish'(x) = sigmoid(x) + x * sigmoid(x) * (1 - sigmoid(x))
    // = sigmoid(x) * (1 + x * (1 - sigmoid(x)))
    // = swish(x) / x + sigmoid(x) * (1 - sigmoid(x)) * x  (for x != 0)

    // Test numerical derivative at a few points
    let eps = 1e-6_f64;
    let test_points = array![0.5_f64, 1.0, 2.0, -0.5, -1.0];

    for i in 0..test_points.len() {
        let x = test_points[i];
        let x_plus = array![x + eps];
        let x_minus = array![x - eps];

        let swish_plus = swish_simd(&x_plus.view())[0];
        let swish_minus = swish_simd(&x_minus.view())[0];

        let numerical_derivative = (swish_plus - swish_minus) / (2.0 * eps);

        // Analytical derivative: σ(x) + x * σ(x) * (1 - σ(x)) = σ(x) * (1 + x * (1 - σ(x)))
        let sigmoid_x = 1.0 / (1.0 + (-x).exp());
        let analytical_derivative = sigmoid_x * (1.0 + x * (1.0 - sigmoid_x));

        assert!(
            (numerical_derivative - analytical_derivative).abs() < 1e-4,
            "Swish derivative at {} should be approximately {}, numerical: {}",
            x,
            analytical_derivative,
            numerical_derivative
        );
    }
}

// ============================================================================
// ============================================================================
// Softplus SIMD Tests
// ============================================================================

/// Test Softplus(0) = ln(2) ≈ 0.693
#[test]
fn test_softplus_simd_f64_at_zero() {
    let x = array![0.0_f64];
    let result = softplus_simd(&x.view());
    let expected = (2.0_f64).ln(); // ln(1 + exp(0)) = ln(2)
    assert!(
        (result[0] - expected).abs() < 1e-15,
        "Softplus(0) should be ln(2) ≈ {}, got {}",
        expected,
        result[0]
    );
}

/// Test Softplus basic values
#[test]
fn test_softplus_simd_f64_basic() {
    let x = array![0.0_f64, 1.0, -1.0, 2.0, -2.0];
    let result = softplus_simd(&x.view());

    // Softplus(x) = ln(1 + exp(x))
    // Softplus(0) = ln(2)
    let expected_0 = (2.0_f64).ln();
    assert!(
        (result[0] - expected_0).abs() < 1e-10,
        "Softplus(0) should be ln(2)"
    );

    // Softplus(1) = ln(1 + e) ≈ 1.3133
    let expected_1 = (1.0_f64 + 1.0_f64.exp()).ln();
    assert!(
        (result[1] - expected_1).abs() < 1e-10,
        "Softplus(1) should be approximately {}, got {}",
        expected_1,
        result[1]
    );

    // Softplus(-1) = ln(1 + 1/e) ≈ 0.3133
    let expected_neg1 = (1.0_f64 + (-1.0_f64).exp()).ln();
    assert!(
        (result[2] - expected_neg1).abs() < 1e-10,
        "Softplus(-1) should be approximately {}, got {}",
        expected_neg1,
        result[2]
    );

    // Softplus(2) = ln(1 + e²) ≈ 2.1269
    let expected_2 = (1.0_f64 + 2.0_f64.exp()).ln();
    assert!(
        (result[3] - expected_2).abs() < 1e-10,
        "Softplus(2) should be approximately {}, got {}",
        expected_2,
        result[3]
    );

    // Softplus(-2) = ln(1 + e^-2) ≈ 0.1269
    let expected_neg2 = (1.0_f64 + (-2.0_f64).exp()).ln();
    assert!(
        (result[4] - expected_neg2).abs() < 1e-10,
        "Softplus(-2) should be approximately {}, got {}",
        expected_neg2,
        result[4]
    );
}

/// Test Softplus asymptotic behavior: Softplus(x) ≈ x for large positive x
#[test]
fn test_softplus_simd_f64_large_positive() {
    let x = array![10.0_f64, 20.0, 50.0];
    let result = softplus_simd(&x.view());

    // For large x, exp(x) >> 1, so ln(1 + exp(x)) ≈ ln(exp(x)) = x
    for i in 0..x.len() {
        let relative_error = (result[i] - x[i]).abs() / x[i];
        assert!(
            relative_error < 1e-4,
            "Softplus({}) ≈ {}, relative error should be small: {}",
            x[i],
            result[i],
            relative_error
        );
    }
}

/// Test Softplus asymptotic behavior: Softplus(x) ≈ exp(x) for large negative x
#[test]
fn test_softplus_simd_f64_large_negative() {
    let x = array![-10.0_f64, -20.0, -50.0];
    let result = softplus_simd(&x.view());

    // For large negative x, ln(1 + exp(x)) ≈ exp(x) ≈ 0
    for i in 0..x.len() {
        let expected = x[i].exp();
        // Use relative error for small positive values
        let relative_error = if expected > 1e-15 {
            (result[i] - expected).abs() / expected
        } else {
            result[i].abs()
        };
        assert!(
            relative_error < 0.01, // 1% relative error tolerance
            "Softplus({}) should be approximately exp({}) = {}, got {}, relative error: {}",
            x[i],
            x[i],
            expected,
            result[i],
            relative_error
        );
    }
}

/// Test Softplus is always positive
#[test]
fn test_softplus_simd_f64_always_positive() {
    let x = array![-100.0_f64, -10.0, -1.0, 0.0, 1.0, 10.0, 100.0];
    let result = softplus_simd(&x.view());

    for i in 0..x.len() {
        assert!(
            result[i] > 0.0,
            "Softplus({}) should be positive, got {}",
            x[i],
            result[i]
        );
    }
}

/// Test Softplus is monotonically increasing
#[test]
fn test_softplus_simd_f64_monotonicity() {
    let n = 100;
    let x: Array1<f64> = Array1::linspace(-10.0, 10.0, n);
    let result = softplus_simd(&x.view());

    for i in 1..n {
        assert!(
            result[i] > result[i - 1],
            "Softplus should be monotonically increasing: softplus({}) = {} <= softplus({}) = {}",
            x[i - 1],
            result[i - 1],
            x[i],
            result[i]
        );
    }
}

/// Test Softplus is smooth (no discontinuities)
#[test]
fn test_softplus_simd_f64_smoothness() {
    let eps = 1e-6_f64;
    let x = array![-eps, 0.0, eps];
    let result = softplus_simd(&x.view());

    // Values should be close to each other (continuous)
    let diff_left = (result[0] - result[1]).abs();
    let diff_right = (result[2] - result[1]).abs();

    assert!(
        diff_left < 1e-5,
        "Softplus should be continuous at 0 from the left"
    );
    assert!(
        diff_right < 1e-5,
        "Softplus should be continuous at 0 from the right"
    );
}

/// Test Softplus with NaN
#[test]
fn test_softplus_simd_f64_nan() {
    let x = array![f64::NAN];
    let result = softplus_simd(&x.view());

    assert!(result[0].is_nan(), "Softplus(NaN) should be NaN");
}

/// Test Softplus with f32
#[test]
fn test_softplus_simd_f32_basic() {
    let x = array![0.0_f32, 1.0, -1.0, 2.0, -2.0];
    let result = softplus_simd(&x.view());

    // Softplus(0) = ln(2)
    let expected_0 = (2.0_f32).ln();
    assert!(
        (result[0] - expected_0).abs() < 1e-6,
        "Softplus(0) should be ln(2)"
    );

    // Softplus(1) = ln(1 + e)
    let expected_1 = (1.0_f32 + 1.0_f32.exp()).ln();
    assert!(
        (result[1] - expected_1).abs() < 1e-5,
        "Softplus(1) should be approximately {}",
        expected_1
    );

    // Softplus(-1) = ln(1 + 1/e)
    let expected_neg1 = (1.0_f32 + (-1.0_f32).exp()).ln();
    assert!(
        (result[2] - expected_neg1).abs() < 1e-5,
        "Softplus(-1) should be approximately {}",
        expected_neg1
    );
}

/// Test Softplus empty array
#[test]
fn test_softplus_simd_empty() {
    let x = array![] as Array1<f64>;
    let result = softplus_simd(&x.view());
    assert!(result.is_empty(), "Softplus of empty array should be empty");
}

/// Test Softplus large array (SIMD path)
#[test]
fn test_softplus_simd_large_array() {
    let n = 10000;
    let x: Array1<f64> = Array1::linspace(-10.0, 10.0, n);
    let result = softplus_simd(&x.view());

    // Check length
    assert_eq!(result.len(), n, "Result should have same length as input");

    // Verify all values are positive
    for i in 0..n {
        assert!(
            result[i] > 0.0,
            "Softplus({}) should be positive, got {}",
            x[i],
            result[i]
        );
    }

    // Verify monotonicity in samples
    for i in (0..n).step_by(100) {
        if i > 0 {
            assert!(
                result[i] >= result[i - 100],
                "Softplus should be monotonically increasing"
            );
        }
    }
}

/// Test Softplus derivative relation: softplus'(x) = sigmoid(x)
#[test]
fn test_softplus_simd_derivative() {
    // softplus'(x) = exp(x) / (1 + exp(x)) = 1 / (1 + exp(-x)) = sigmoid(x)
    let eps = 1e-6_f64;
    let test_points = array![0.0_f64, 0.5, 1.0, 2.0, -0.5, -1.0, -2.0];

    for i in 0..test_points.len() {
        let x = test_points[i];
        let x_plus = array![x + eps];
        let x_minus = array![x - eps];

        let softplus_plus = softplus_simd(&x_plus.view())[0];
        let softplus_minus = softplus_simd(&x_minus.view())[0];

        let numerical_derivative = (softplus_plus - softplus_minus) / (2.0 * eps);

        // Analytical derivative: sigmoid(x)
        let analytical_derivative = 1.0 / (1.0 + (-x).exp());

        assert!(
            (numerical_derivative - analytical_derivative).abs() < 1e-4,
            "Softplus derivative at {} should be sigmoid({}) = {}, numerical: {}",
            x,
            x,
            analytical_derivative,
            numerical_derivative
        );
    }
}

/// Test Softplus in probabilistic model use case
#[test]
fn test_softplus_simd_probabilistic_model() {
    // Softplus is often used to ensure positive parameters (e.g., variance)
    // log_var = raw_output
    // var = softplus(log_var)  (ensures var > 0)
    let raw_outputs = array![-5.0_f64, -2.0, 0.0, 2.0, 5.0];
    let variances = softplus_simd(&raw_outputs.view());

    // All variances should be positive
    for i in 0..raw_outputs.len() {
        assert!(
            variances[i] > 0.0,
            "Variance should be positive, got {}",
            variances[i]
        );
        assert!(
            variances[i].is_finite(),
            "Variance should be finite, got {}",
            variances[i]
        );
    }

    // Larger raw outputs should give larger variances
    for i in 1..raw_outputs.len() {
        assert!(
            variances[i] > variances[i - 1],
            "Variance should increase with raw output"
        );
    }
}

/// Test Softplus vs ReLU approximation
#[test]
fn test_softplus_simd_vs_relu() {
    // Softplus is a smooth approximation of ReLU
    // For large positive x, Softplus(x) ≈ x ≈ ReLU(x)
    // For large negative x, Softplus(x) ≈ 0 ≈ ReLU(x)
    // But Softplus is always > 0, while ReLU(x) = 0 for x < 0
    // Note: Softplus(x) = x + ln(1 + exp(-x)) > x for all x

    let x = array![-3.0_f64, -1.0, 0.0, 1.0, 3.0];
    let softplus_result = softplus_simd(&x.view());

    // At x = 0, Softplus(0) = ln(2) ≠ 0 = ReLU(0)
    assert!(softplus_result[2] > 0.0, "Softplus(0) = ln(2) > 0");

    // For x > 0, Softplus(x) > x (by ln(1 + exp(-x)))
    // but approaches x as x increases
    assert!(softplus_result[3] > x[3], "Softplus(1) > 1");
    assert!(softplus_result[4] > x[4], "Softplus(3) > 3");

    // The difference Softplus(x) - x = ln(1 + exp(-x)) decreases as x increases
    let diff_1 = softplus_result[3] - x[3]; // ln(1 + exp(-1)) ≈ 0.313
    let diff_3 = softplus_result[4] - x[4]; // ln(1 + exp(-3)) ≈ 0.049
    assert!(
        diff_3 < diff_1,
        "Softplus - x should decrease as x increases"
    );

    // For large positive x, Softplus(x) approaches x
    let large_x = array![10.0_f64];
    let large_result = softplus_simd(&large_x.view())[0];
    assert!((large_result - 10.0).abs() < 0.001);
}

// ============================================================================
// ============================================================================
// Mish SIMD Tests
// ============================================================================

/// Test Mish(0) = 0
#[test]
fn test_mish_simd_f64_at_zero() {
    let x = array![0.0_f64];
    let result = mish_simd(&x.view());
    // Mish(0) = 0 * tanh(softplus(0)) = 0 * tanh(ln(2)) = 0
    assert!(
        result[0].abs() < 1e-15,
        "Mish(0) should be 0, got {}",
        result[0]
    );
}

/// Test Mish basic values
#[test]
fn test_mish_simd_f64_basic() {
    let x = array![0.0_f64, 1.0, -1.0, 2.0, -2.0];
    let result = mish_simd(&x.view());

    // Mish(x) = x * tanh(softplus(x)) = x * tanh(ln(1 + exp(x)))
    // Mish(0) = 0
    assert!(result[0].abs() < 1e-10, "Mish(0) should be 0");

    // Mish(1) = 1 * tanh(ln(1 + e)) ≈ 0.8651
    let softplus_1 = (1.0_f64 + 1.0_f64.exp()).ln();
    let expected_mish_1 = 1.0 * softplus_1.tanh();
    assert!(
        (result[1] - expected_mish_1).abs() < 1e-10,
        "Mish(1) should be approximately {}, got {}",
        expected_mish_1,
        result[1]
    );

    // Mish(-1) = -1 * tanh(ln(1 + 1/e)) ≈ -0.3034
    let softplus_neg1 = (1.0_f64 + (-1.0_f64).exp()).ln();
    let expected_mish_neg1 = -softplus_neg1.tanh();
    assert!(
        (result[2] - expected_mish_neg1).abs() < 1e-10,
        "Mish(-1) should be approximately {}, got {}",
        expected_mish_neg1,
        result[2]
    );

    // Mish(2) = 2 * tanh(ln(1 + e²)) ≈ 1.9439
    let softplus_2 = (1.0_f64 + 2.0_f64.exp()).ln();
    let expected_mish_2 = 2.0 * softplus_2.tanh();
    assert!(
        (result[3] - expected_mish_2).abs() < 1e-10,
        "Mish(2) should be approximately {}, got {}",
        expected_mish_2,
        result[3]
    );
}

/// Test Mish asymptotic behavior: Mish(x) ≈ x for large positive x
#[test]
fn test_mish_simd_f64_large_positive() {
    let x = array![5.0_f64, 10.0, 20.0];
    let result = mish_simd(&x.view());

    // For large x, softplus(x) ≈ x, tanh(x) ≈ 1, so Mish(x) ≈ x * 1 = x
    for i in 0..x.len() {
        let relative_error = (result[i] - x[i]).abs() / x[i];
        assert!(
            relative_error < 0.01,
            "Mish({}) ≈ {}, relative error should be small: {}",
            x[i],
            result[i],
            relative_error
        );
    }
}

/// Test Mish asymptotic behavior: Mish(x) ≈ 0 for large negative x
#[test]
fn test_mish_simd_f64_large_negative() {
    let x = array![-5.0_f64, -10.0, -20.0];
    let result = mish_simd(&x.view());

    // For large negative x, softplus(x) ≈ 0, tanh(0) = 0, so Mish(x) ≈ 0
    for i in 0..x.len() {
        assert!(
            result[i].abs() < 0.1,
            "Mish({}) should be approximately 0, got {}",
            x[i],
            result[i]
        );
    }
}

/// Test Mish has a global minimum around x ≈ -1.2 with value ≈ -0.31
#[test]
fn test_mish_simd_f64_global_minimum() {
    // Mish has a global minimum around x ≈ -1.2
    let x = array![-1.2_f64, -1.5, -1.0, -2.0];
    let result = mish_simd(&x.view());

    // The minimum should be around -0.31
    let min_value = result[0];
    assert!(
        min_value < -0.28 && min_value > -0.35,
        "Mish global minimum should be around -0.31, got {}",
        min_value
    );
}

/// Test Mish is smooth (no discontinuities)
#[test]
fn test_mish_simd_f64_smoothness() {
    let eps = 1e-6_f64;
    let x = array![-eps, 0.0, eps];
    let result = mish_simd(&x.view());

    // Values should be close to each other (continuous)
    let diff_left = (result[0] - result[1]).abs();
    let diff_right = (result[2] - result[1]).abs();

    assert!(
        diff_left < 1e-5,
        "Mish should be continuous at 0 from the left"
    );
    assert!(
        diff_right < 1e-5,
        "Mish should be continuous at 0 from the right"
    );
}

/// Test Mish with NaN
#[test]
fn test_mish_simd_f64_nan() {
    let x = array![f64::NAN];
    let result = mish_simd(&x.view());

    assert!(result[0].is_nan(), "Mish(NaN) should be NaN");
}

/// Test Mish with f32
#[test]
fn test_mish_simd_f32_basic() {
    let x = array![0.0_f32, 1.0, -1.0, 2.0, -2.0];
    let result = mish_simd(&x.view());

    // Mish(0) = 0
    assert!(result[0].abs() < 1e-6, "Mish(0) should be 0");

    // Mish(1) ≈ 0.8651
    let softplus_1 = (1.0_f32 + 1.0_f32.exp()).ln();
    let expected_mish_1 = 1.0_f32 * softplus_1.tanh();
    assert!(
        (result[1] - expected_mish_1).abs() < 1e-5,
        "Mish(1) should be approximately {}",
        expected_mish_1
    );
}

/// Test Mish empty array
#[test]
fn test_mish_simd_empty() {
    let x = array![] as Array1<f64>;
    let result = mish_simd(&x.view());
    assert!(result.is_empty(), "Mish of empty array should be empty");
}

/// Test Mish large array (SIMD path)
#[test]
fn test_mish_simd_large_array() {
    let n = 10000;
    let x: Array1<f64> = Array1::linspace(-5.0, 5.0, n);
    let result = mish_simd(&x.view());

    // Check length
    assert_eq!(result.len(), n, "Result should have same length as input");

    // Verify all values are computed correctly for a sample
    for i in (0..n).step_by(500) {
        let xi = x[i];
        let softplus_xi = (1.0_f64 + xi.exp()).ln();
        let expected = xi * softplus_xi.tanh();
        assert!(
            (result[i] - expected).abs() < 1e-8,
            "Mish({}) should be {}, got {}",
            xi,
            expected,
            result[i]
        );
    }
}

/// Test Mish relation to Swish: both are self-gating activations
#[test]
fn test_mish_simd_vs_swish() {
    // Both Mish and Swish are self-gating: x * f(x)
    // Mish: x * tanh(softplus(x))
    // Swish: x * sigmoid(x)
    let x = array![-2.0_f64, -1.0, 0.0, 1.0, 2.0];
    let mish_result = mish_simd(&x.view());
    let swish_result = swish_simd(&x.view());

    // Both should be 0 at x = 0
    assert!(mish_result[2].abs() < 1e-10);
    assert!(swish_result[2].abs() < 1e-10);

    // For positive x, Mish > Swish (tanh(softplus(x)) > sigmoid(x) for x > 0)
    assert!(mish_result[3] > swish_result[3]);
    assert!(mish_result[4] > swish_result[4]);

    // Both approach x for large positive x
    let large_x = array![10.0_f64];
    let mish_large = mish_simd(&large_x.view())[0];
    let swish_large = swish_simd(&large_x.view())[0];
    assert!((mish_large - 10.0).abs() < 0.01);
    assert!((swish_large - 10.0).abs() < 0.01);
}

/// Test Mish in YOLOv4-like computation
#[test]
fn test_mish_simd_yolov4_use_case() {
    // Simulate activation in YOLOv4: Mish applied after convolution
    let features = array![0.5_f64, 1.2, -0.8, 2.0, -1.5, 0.0, 3.0, -3.0];
    let activated = mish_simd(&features.view());

    // All results should be finite
    for i in 0..features.len() {
        assert!(
            activated[i].is_finite(),
            "Mish output should be finite for input {}",
            features[i]
        );
    }

    // Zero should produce zero
    assert!(activated[5].abs() < 1e-10);

    // Positive values should produce positive outputs
    assert!(activated[0] > 0.0);
    assert!(activated[1] > 0.0);
    assert!(activated[3] > 0.0);
    assert!(activated[6] > 0.0);

    // All outputs should be >= global minimum ≈ -0.31
    for i in 0..features.len() {
        assert!(
            activated[i] > -0.35,
            "Mish output should be >= -0.31, got {}",
            activated[i]
        );
    }
}

// ============================================================================
// ============================================================================
