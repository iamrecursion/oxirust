//! SIMD Softmax integration tests
//!
//! This module tests the SIMD-accelerated softmax implementation
//! in scirs2-neural, ensuring correctness and performance.

use scirs2_core::ndarray::Array;
use scirs2_neural::{Activation, Softmax};

#[test]
fn test_simd_softmax_f64_basic() {
    let softmax = Softmax::new(-1); // axis=-1 for last axis
    let input = Array::from_vec(vec![1.0f64, 2.0, 3.0]).into_dyn();
    let output = softmax.forward(&input).expect("Operation failed");

    // Expected: exp(1)/(exp(1)+exp(2)+exp(3)), etc.
    let exp1 = 1.0f64.exp();
    let exp2 = 2.0f64.exp();
    let exp3 = 3.0f64.exp();
    let sum = exp1 + exp2 + exp3;

    let expected = [exp1 / sum, exp2 / sum, exp3 / sum];

    for (i, (&out, &exp)) in output.iter().zip(expected.iter()).enumerate() {
        assert!(
            (out - exp).abs() < 1e-10,
            "Mismatch at index {}: got {}, expected {}",
            i,
            out,
            exp
        );
    }
}

#[test]
fn test_simd_softmax_f32_basic() {
    let softmax = Softmax::new(-1);
    let input = Array::from_vec(vec![1.0f32, 2.0, 3.0]).into_dyn();
    let output = softmax.forward(&input).expect("Operation failed");

    // Expected: exp(1)/(exp(1)+exp(2)+exp(3)), etc.
    let exp1 = 1.0f32.exp();
    let exp2 = 2.0f32.exp();
    let exp3 = 3.0f32.exp();
    let sum = exp1 + exp2 + exp3;

    let expected = [exp1 / sum, exp2 / sum, exp3 / sum];

    for (i, (&out, &exp)) in output.iter().zip(expected.iter()).enumerate() {
        assert!(
            (out - exp).abs() < 1e-6,
            "Mismatch at index {}: got {}, expected {}",
            i,
            out,
            exp
        );
    }
}

#[test]
fn test_simd_softmax_f64_large() {
    // Test with large array to ensure SIMD path is used
    let softmax = Softmax::new(-1);
    let input_vec: Vec<f64> = (0..1000).map(|i| i as f64 * 0.01).collect();
    let input = Array::from_vec(input_vec.clone()).into_dyn();
    let output = softmax.forward(&input).expect("Operation failed");

    // Sum should be 1.0
    let sum: f64 = output.iter().sum();
    assert!(
        (sum - 1.0).abs() < 1e-10,
        "Softmax output should sum to 1.0, got {}",
        sum
    );

    // All values should be positive
    for (i, &val) in output.iter().enumerate() {
        assert!(val > 0.0, "Softmax value at index {} should be positive", i);
        assert!(
            val < 1.0,
            "Softmax value at index {} should be less than 1.0",
            i
        );
    }
}

#[test]
fn test_simd_softmax_f32_large() {
    // Test with large array to ensure SIMD path is used
    let softmax = Softmax::new(-1);
    let input_vec: Vec<f32> = (0..1000).map(|i| i as f32 * 0.01).collect();
    let input = Array::from_vec(input_vec.clone()).into_dyn();
    let output = softmax.forward(&input).expect("Operation failed");

    // Sum should be 1.0
    let sum: f32 = output.iter().sum();
    assert!(
        (sum - 1.0).abs() < 1e-6,
        "Softmax output should sum to 1.0, got {}",
        sum
    );

    // All values should be positive
    for (i, &val) in output.iter().enumerate() {
        assert!(val > 0.0, "Softmax value at index {} should be positive", i);
        assert!(
            val < 1.0,
            "Softmax value at index {} should be less than 1.0",
            i
        );
    }
}

#[test]
fn test_simd_softmax_numerical_stability() {
    // Test with large values that could cause overflow without numerical stability
    let softmax = Softmax::new(-1);
    let input = Array::from_vec(vec![1000.0f64, 1001.0, 1002.0]).into_dyn();
    let output = softmax.forward(&input).expect("Operation failed");

    // Sum should still be 1.0
    let sum: f64 = output.iter().sum();
    assert!(
        (sum - 1.0).abs() < 1e-10,
        "Softmax should handle large values, sum got {}",
        sum
    );

    // All values should be finite
    for (i, &val) in output.iter().enumerate() {
        assert!(
            val.is_finite(),
            "Softmax value at index {} should be finite",
            i
        );
    }
}

#[test]
fn test_simd_softmax_single_element() {
    // Test with single element array
    let softmax = Softmax::new(-1);
    let input = Array::from_vec(vec![5.0f64]).into_dyn();
    let output = softmax.forward(&input).expect("Operation failed");

    // Single element should have probability 1.0
    assert!(
        (output[[0]] - 1.0).abs() < 1e-10,
        "Single element softmax should be 1.0, got {}",
        output[[0]]
    );
}

#[test]
fn test_simd_softmax_uniform() {
    // Test with uniform input
    let softmax = Softmax::new(-1);
    let input = Array::from_vec(vec![5.0f64; 10]).into_dyn();
    let output = softmax.forward(&input).expect("Operation failed");

    // All values should be equal (1/10)
    let expected = 0.1f64;
    for (i, &val) in output.iter().enumerate() {
        assert!(
            (val - expected).abs() < 1e-10,
            "Uniform input should produce uniform output, index {} got {}, expected {}",
            i,
            val,
            expected
        );
    }
}

#[test]
fn test_simd_softmax_negative_values() {
    // Test with negative values
    let softmax = Softmax::new(-1);
    let input = Array::from_vec(vec![-5.0f64, -3.0, -1.0]).into_dyn();
    let output = softmax.forward(&input).expect("Operation failed");

    // Sum should be 1.0
    let sum: f64 = output.iter().sum();
    assert!(
        (sum - 1.0).abs() < 1e-10,
        "Softmax with negative values should sum to 1.0, got {}",
        sum
    );

    // Larger input should have larger output
    assert!(output[[2]] > output[[1]], "Softmax should be monotonic");
    assert!(output[[1]] > output[[0]], "Softmax should be monotonic");
}

#[test]
fn test_simd_softmax_accuracy_vs_scalar() {
    // Compare SIMD implementation against known correct values
    let softmax = Softmax::new(-1);
    let input = Array::from_vec(vec![0.0f64, 1.0, 2.0, 3.0, 4.0]).into_dyn();
    let output = softmax.forward(&input).expect("Operation failed");

    // Manually computed expected values
    let exp_vals: Vec<f64> = [0.0f64, 1.0, 2.0, 3.0, 4.0]
        .iter()
        .map(|&x| x.exp())
        .collect();
    let sum: f64 = exp_vals.iter().sum();
    let expected: Vec<f64> = exp_vals.iter().map(|&e| e / sum).collect();

    for (i, (&out, &exp)) in output.iter().zip(expected.iter()).enumerate() {
        assert!(
            (out - exp).abs() < 1e-10,
            "Accuracy check failed at index {}: got {}, expected {}",
            i,
            out,
            exp
        );
    }
}

#[test]
fn test_simd_softmax_backward_pass() {
    // Test backward pass with SIMD-computed forward pass
    let softmax = Softmax::new(-1);
    let input = Array::from_vec(vec![1.0f64, 2.0, 3.0]).into_dyn();
    let output = softmax.forward(&input).expect("Operation failed");

    // Gradient for classification: one-hot encoded target
    let grad_output = Array::from_vec(vec![0.0f64, 0.0, 1.0]).into_dyn();
    let grad_input = softmax
        .backward(&grad_output, &output)
        .expect("Operation failed");

    // Gradient should have same shape as input
    assert_eq!(grad_input.shape(), input.shape());

    // Gradient sum should be zero (property of softmax gradient)
    let grad_sum: f64 = grad_input.iter().sum();
    assert!(
        grad_sum.abs() < 1e-10,
        "Softmax gradient should sum to zero, got {}",
        grad_sum
    );
}
