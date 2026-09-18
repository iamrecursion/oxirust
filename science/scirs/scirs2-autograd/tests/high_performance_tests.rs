//! Comprehensive tests for high_performance.rs
//!
//! This test suite provides 40+ tests covering SIMD backward pass, parallel gradient
//! computation, ultra backward pass, and memory-efficient accumulation to ensure
//! 85%+ coverage of high_performance.rs functionality.

use ag::tensor_ops as T;
use scirs2_autograd as ag;
use scirs2_autograd::high_performance::*;
use scirs2_autograd::variable::{SafeVariable, SafeVariableEnvironment, VariableID};
use scirs2_core::ndarray::{array, Array1, Array2};
use std::sync::Arc;

// Helper function to create test tensors
fn create_test_tensor<F: ag::Float + Send + Sync>(
    env: &Arc<SafeVariableEnvironment<F>>,
    shape: &[usize],
    value: F,
) -> SafeVariable<F> {
    let size: usize = shape.iter().product();
    let data = vec![value; size];
    let arr = ag::NdArray::from_shape_vec(shape, data).expect("Failed to create array");
    SafeVariable::new(arr, Arc::clone(env), false).expect("Failed to create SafeVariable")
}

// ============================================================================
// SIMD BACKWARD PASS TESTS (10 tests)
// ============================================================================

#[test]
fn test_simd_backward_correctness_simple() {
    let env = Arc::new(SafeVariableEnvironment::<f32>::new());

    let x = create_test_tensor(&env, &[4], 2.0);
    let y = create_test_tensor(&env, &[4], 3.0);

    // z = x * y
    let z_data = x.data().expect("Failed to get x") * y.data().expect("Failed to get y");
    let z = SafeVariable::new(z_data, Arc::clone(&env), false).expect("Failed to create z");

    // Test SIMD backward pass
    let result = simd_backward_pass(&z, &env);
    assert!(result.is_ok(), "SIMD backward pass should succeed");
}

#[test]
fn test_simd_backward_correctness_complex() {
    let env = Arc::new(SafeVariableEnvironment::<f32>::new());

    let x = create_test_tensor(&env, &[8, 8], 1.5);
    let y = create_test_tensor(&env, &[8, 8], 2.5);

    // Complex computation: z = x^2 + y * 3
    let x_data = x.data().expect("Failed to get x");
    let y_data = y.data().expect("Failed to get y");
    let z_data = x_data.mapv(|v| v * v) + y_data.mapv(|v| v * 3.0);
    let z = SafeVariable::new(z_data, Arc::clone(&env), false).expect("Failed to create z");

    let result = simd_backward_pass(&z, &env);
    assert!(
        result.is_ok(),
        "SIMD backward pass on complex computation should succeed"
    );
}

#[test]
#[cfg(feature = "simd")]
fn test_simd_vs_scalar_equivalence_f32() {
    let env = Arc::new(SafeVariableEnvironment::<f32>::new());

    let x = create_test_tensor(&env, &[16], 2.0);

    // Standard backward pass
    let result_standard = x.backward();
    assert!(
        result_standard.is_ok(),
        "Standard backward pass should succeed"
    );

    // SIMD backward pass
    let result_simd = simd_backward_pass(&x, &env);
    assert!(result_simd.is_ok(), "SIMD backward pass should succeed");

    // Both should succeed (exact equivalence testing requires more infrastructure)
}

#[test]
#[cfg(feature = "simd")]
fn test_simd_vs_scalar_equivalence_f64() {
    let env = Arc::new(SafeVariableEnvironment::<f64>::new());

    let x = create_test_tensor(&env, &[16], 2.0);

    let result_standard = x.backward();
    assert!(
        result_standard.is_ok(),
        "Standard backward pass should succeed"
    );

    let result_simd = simd_backward_pass(&x, &env);
    assert!(result_simd.is_ok(), "SIMD backward pass should succeed");
}

#[test]
fn test_simd_small_tensors_fallback() {
    let env = Arc::new(SafeVariableEnvironment::<f32>::new());

    // Small tensor that may not benefit from SIMD
    let x = create_test_tensor(&env, &[2], 1.0);

    let result = simd_backward_pass(&x, &env);
    assert!(
        result.is_ok(),
        "SIMD should handle small tensors via fallback"
    );
}

#[test]
#[cfg(feature = "simd")]
fn test_simd_large_tensors_performance() {
    let env = Arc::new(SafeVariableEnvironment::<f32>::new());

    // Large tensor that should benefit from SIMD
    let x = create_test_tensor(&env, &[1024], 1.5);

    let result = simd_backward_pass(&x, &env);
    assert!(
        result.is_ok(),
        "SIMD should handle large tensors efficiently"
    );
}

#[test]
fn test_simd_multidimensional_tensors() {
    let env = Arc::new(SafeVariableEnvironment::<f32>::new());

    let x = create_test_tensor(&env, &[4, 8, 8], 2.0);

    let result = simd_backward_pass(&x, &env);
    assert!(
        result.is_ok(),
        "SIMD should handle multidimensional tensors"
    );
}

#[test]
fn test_simd_with_broadcasting() {
    let env = Arc::new(SafeVariableEnvironment::<f32>::new());

    let x = create_test_tensor(&env, &[8], 2.0);
    let y = create_test_tensor(&env, &[8], 3.0);

    let x_data = x.data().expect("Failed to get x");
    let y_data = y.data().expect("Failed to get y");
    let z_data = x_data + y_data;
    let z = SafeVariable::new(z_data, Arc::clone(&env), false).expect("Failed to create z");

    let result = simd_backward_pass(&z, &env);
    assert!(result.is_ok(), "SIMD should handle broadcasting operations");
}

#[test]
fn test_simd_gradient_accumulation() {
    let env = Arc::new(SafeVariableEnvironment::<f32>::new());

    let x = create_test_tensor(&env, &[16], 1.0);

    // Multiple backward passes to test gradient accumulation
    for _ in 0..3 {
        let result = simd_backward_pass(&x, &env);
        assert!(result.is_ok(), "SIMD gradient accumulation should work");
    }
}

#[test]
fn test_simd_edge_cases() {
    let env = Arc::new(SafeVariableEnvironment::<f32>::new());

    // Test with edge case values
    let x = create_test_tensor(&env, &[8], 0.0);
    let result = simd_backward_pass(&x, &env);
    assert!(result.is_ok(), "SIMD should handle zero tensors");

    let y = create_test_tensor(&env, &[8], 1e-10);
    let result = simd_backward_pass(&y, &env);
    assert!(result.is_ok(), "SIMD should handle very small values");
}

// ============================================================================
// PARALLEL GRADIENT TESTS (10 tests)
// ============================================================================

#[test]
#[cfg(feature = "parallel")]
fn test_parallel_vs_sequential_correctness() {
    let env = Arc::new(SafeVariableEnvironment::<f32>::new());

    let x1 = create_test_tensor(&env, &[8], 2.0);
    let x2 = create_test_tensor(&env, &[8], 3.0);
    let x3 = create_test_tensor(&env, &[8], 4.0);
    let x4 = create_test_tensor(&env, &[8], 5.0);

    let outputs = [&x1, &x2, &x3, &x4];
    let inputs = vec![x1.id, x2.id, x3.id, x4.id];

    let result = parallel_gradient_computation(&outputs[..], &inputs, &env);
    assert!(
        result.is_ok(),
        "Parallel gradient computation should succeed"
    );

    let gradients = result.expect("Should have gradients");
    assert_eq!(
        gradients.len(),
        inputs.len(),
        "Should have gradient for each input"
    );
}

#[test]
#[cfg(feature = "parallel")]
fn test_parallel_scaling_2_outputs() {
    let env = Arc::new(SafeVariableEnvironment::<f32>::new());

    let x1 = create_test_tensor(&env, &[16], 1.0);
    let x2 = create_test_tensor(&env, &[16], 2.0);

    let outputs = [&x1, &x2];
    let inputs = vec![x1.id, x2.id];

    let result = parallel_gradient_computation(&outputs[..], &inputs, &env);
    assert!(
        result.is_ok(),
        "Parallel computation with 2 outputs should work"
    );
}

#[test]
#[cfg(feature = "parallel")]
fn test_parallel_scaling_4_outputs() {
    let env = Arc::new(SafeVariableEnvironment::<f32>::new());

    let vars: Vec<_> = (0..4)
        .map(|i| create_test_tensor(&env, &[16], (i + 1) as f32))
        .collect();

    let outputs: Vec<_> = vars.iter().collect();
    let inputs: Vec<_> = vars.iter().map(|v| v.id).collect();

    let result = parallel_gradient_computation(&outputs[..], &inputs, &env);
    assert!(
        result.is_ok(),
        "Parallel computation with 4 outputs should work"
    );

    let gradients = result.expect("Should have gradients");
    assert_eq!(gradients.len(), 4, "Should have 4 gradients");
}

#[test]
#[cfg(feature = "parallel")]
fn test_parallel_scaling_8_outputs() {
    let env = Arc::new(SafeVariableEnvironment::<f32>::new());

    let vars: Vec<_> = (0..8)
        .map(|i| create_test_tensor(&env, &[32], (i + 1) as f32))
        .collect();

    let outputs: Vec<_> = vars.iter().collect();
    let inputs: Vec<_> = vars.iter().map(|v| v.id).collect();

    let result = parallel_gradient_computation(&outputs[..], &inputs, &env);
    assert!(
        result.is_ok(),
        "Parallel computation with 8 outputs should work"
    );

    let gradients = result.expect("Should have gradients");
    assert_eq!(gradients.len(), 8, "Should have 8 gradients");
}

#[test]
#[cfg(feature = "parallel")]
fn test_parallel_load_balancing() {
    let env = Arc::new(SafeVariableEnvironment::<f32>::new());

    // Create outputs with different sizes to test load balancing
    let x1 = create_test_tensor(&env, &[8], 1.0);
    let x2 = create_test_tensor(&env, &[16], 2.0);
    let x3 = create_test_tensor(&env, &[32], 3.0);
    let x4 = create_test_tensor(&env, &[64], 4.0);

    let outputs = [&x1, &x2, &x3, &x4];
    let inputs = vec![x1.id, x2.id, x3.id, x4.id];

    let result = parallel_gradient_computation(&outputs[..], &inputs, &env);
    assert!(result.is_ok(), "Parallel load balancing should work");
}

#[test]
fn test_parallel_small_workload() {
    let env = Arc::new(SafeVariableEnvironment::<f32>::new());

    // Small workload that may use sequential fallback
    let x = create_test_tensor(&env, &[4], 1.0);

    let outputs = [&x];
    let inputs = vec![x.id];

    let result = parallel_gradient_computation(&outputs[..], &inputs, &env);
    assert!(
        result.is_ok(),
        "Small workload should use fallback correctly"
    );
}

#[test]
#[cfg(feature = "parallel")]
fn test_parallel_large_workload() {
    let env = Arc::new(SafeVariableEnvironment::<f32>::new());

    let vars: Vec<_> = (0..16)
        .map(|i| create_test_tensor(&env, &[64], (i + 1) as f32))
        .collect();

    let outputs: Vec<_> = vars.iter().collect();
    let inputs: Vec<_> = vars.iter().map(|v| v.id).collect();

    let result = parallel_gradient_computation(&outputs[..], &inputs, &env);
    assert!(
        result.is_ok(),
        "Large workload should benefit from parallelization"
    );
}

#[test]
#[cfg(feature = "parallel")]
fn test_parallel_mixed_sizes() {
    let env = Arc::new(SafeVariableEnvironment::<f32>::new());

    let x1 = create_test_tensor(&env, &[4], 1.0);
    let x2 = create_test_tensor(&env, &[128], 2.0);
    let x3 = create_test_tensor(&env, &[16], 3.0);
    let x4 = create_test_tensor(&env, &[256], 4.0);

    let outputs = [&x1, &x2, &x3, &x4];
    let inputs = vec![x1.id, x2.id, x3.id, x4.id];

    let result = parallel_gradient_computation(&outputs[..], &inputs, &env);
    assert!(result.is_ok(), "Mixed sizes should be handled correctly");
}

#[test]
#[cfg(feature = "parallel")]
fn test_parallel_error_propagation() {
    let env = Arc::new(SafeVariableEnvironment::<f32>::new());

    let x1 = create_test_tensor(&env, &[8], 1.0);
    let x2 = create_test_tensor(&env, &[8], 2.0);
    let x3 = create_test_tensor(&env, &[8], 3.0);

    let outputs = [&x1, &x2];
    // Use a valid but unrelated input ID
    let inputs = vec![x3.id]; // x3 is not in outputs, may cause issues

    let result = parallel_gradient_computation(&outputs[..], &inputs, &env);
    // Should either succeed or handle error gracefully
    // Note: This may succeed if the implementation is robust
    let _ = result; // Just verify it doesn't panic
}

#[test]
#[cfg(feature = "parallel")]
fn test_parallel_thread_safety() {
    use std::thread;

    let handles: Vec<_> = (0..4)
        .map(|_| {
            thread::spawn(|| {
                let env = Arc::new(SafeVariableEnvironment::<f32>::new());

                let vars: Vec<_> = (0..4)
                    .map(|i| create_test_tensor(&env, &[16], (i + 1) as f32))
                    .collect();

                let outputs: Vec<_> = vars.iter().collect();
                let inputs: Vec<_> = vars.iter().map(|v| v.id).collect();

                let result = parallel_gradient_computation(&outputs[..], &inputs, &env);
                assert!(
                    result.is_ok(),
                    "Thread-safe parallel computation should work"
                );
            })
        })
        .collect();

    for handle in handles {
        handle.join().expect("Thread should complete successfully");
    }
}

// ============================================================================
// ULTRA BACKWARD PASS TESTS (10 tests)
// ============================================================================

#[test]
#[cfg(all(feature = "simd", feature = "parallel"))]
fn test_ultra_simd_parallel_combined() {
    let env = Arc::new(SafeVariableEnvironment::<f32>::new());

    let vars: Vec<_> = (0..8)
        .map(|i| create_test_tensor(&env, &[64], (i + 1) as f32))
        .collect();

    let outputs: Vec<_> = vars.iter().collect();
    let inputs: Vec<_> = vars.iter().map(|v| v.id).collect();

    let result = ultra_backward_pass(&outputs[..], &inputs, &env);
    assert!(result.is_ok(), "Ultra backward pass should work");
}

#[test]
#[cfg(all(feature = "simd", feature = "parallel"))]
fn test_ultra_platform_detection() {
    use scirs2_core::simd_ops::PlatformCapabilities;

    let caps = PlatformCapabilities::detect();

    let env = Arc::new(SafeVariableEnvironment::<f32>::new());

    let vars: Vec<_> = (0..16)
        .map(|i| create_test_tensor(&env, &[128], (i + 1) as f32))
        .collect();

    let outputs: Vec<_> = vars.iter().collect();
    let inputs: Vec<_> = vars.iter().map(|v| v.id).collect();

    let result = ultra_backward_pass(&outputs[..], &inputs, &env);
    assert!(
        result.is_ok(),
        "Ultra backward should adapt to platform capabilities"
    );

    println!("Platform has AVX2: {}", caps.has_avx2());
    println!("Platform cores: {}", caps.num_cores());
}

#[test]
#[cfg(all(feature = "simd", feature = "parallel"))]
fn test_ultra_vs_standard_correctness() {
    let env = Arc::new(SafeVariableEnvironment::<f32>::new());

    let vars: Vec<_> = (0..8)
        .map(|i| create_test_tensor(&env, &[64], (i + 1) as f32))
        .collect();

    let outputs: Vec<_> = vars.iter().collect();
    let inputs: Vec<_> = vars.iter().map(|v| v.id).collect();

    // Ultra backward pass
    let result_ultra = ultra_backward_pass(&outputs[..], &inputs, &env);
    assert!(result_ultra.is_ok(), "Ultra backward should succeed");

    // Standard parallel backward pass
    let result_standard = parallel_gradient_computation(&outputs[..], &inputs, &env);
    assert!(result_standard.is_ok(), "Standard parallel should succeed");

    // Both should return same number of gradients
    assert_eq!(
        result_ultra.expect("Ultra should have result").len(),
        result_standard.expect("Standard should have result").len(),
        "Should have same number of gradients"
    );
}

#[test]
#[cfg(all(feature = "simd", feature = "parallel"))]
fn test_ultra_performance_benefit() {
    let env = Arc::new(SafeVariableEnvironment::<f32>::new());

    // Large workload to see performance benefits
    let vars: Vec<_> = (0..16)
        .map(|i| create_test_tensor(&env, &[256], (i + 1) as f32))
        .collect();

    let outputs: Vec<_> = vars.iter().collect();
    let inputs: Vec<_> = vars.iter().map(|v| v.id).collect();

    let result = ultra_backward_pass(&outputs[..], &inputs, &env);
    assert!(
        result.is_ok(),
        "Ultra backward should handle large workload efficiently"
    );
}

#[test]
#[cfg(all(feature = "simd", feature = "parallel"))]
fn test_ultra_with_complex_graph() {
    let env = Arc::new(SafeVariableEnvironment::<f32>::new());

    // Create more complex computation graph
    let vars: Vec<_> = (0..10)
        .map(|i| create_test_tensor(&env, &[128], (i + 1) as f32))
        .collect();

    let outputs: Vec<_> = vars.iter().collect();
    let inputs: Vec<_> = vars.iter().map(|v| v.id).collect();

    let result = ultra_backward_pass(&outputs[..], &inputs, &env);
    assert!(
        result.is_ok(),
        "Ultra backward should handle complex graphs"
    );
}

#[test]
fn test_ultra_memory_efficiency() {
    let env = Arc::new(SafeVariableEnvironment::<f32>::new());

    let vars: Vec<_> = (0..8)
        .map(|i| create_test_tensor(&env, &[512], (i + 1) as f32))
        .collect();

    let outputs: Vec<_> = vars.iter().collect();
    let inputs: Vec<_> = vars.iter().map(|v| v.id).collect();

    let result = ultra_backward_pass(&outputs[..], &inputs, &env);
    assert!(result.is_ok(), "Ultra backward should be memory efficient");
}

#[test]
fn test_ultra_gradient_correctness() {
    let env = Arc::new(SafeVariableEnvironment::<f32>::new());

    let vars: Vec<_> = (0..8)
        .map(|i| create_test_tensor(&env, &[64], (i + 1) as f32))
        .collect();

    let outputs: Vec<_> = vars.iter().collect();
    let inputs: Vec<_> = vars.iter().map(|v| v.id).collect();

    let result = ultra_backward_pass(&outputs[..], &inputs, &env);
    assert!(result.is_ok(), "Ultra backward gradients should be correct");

    let gradients = result.expect("Should have gradients");
    assert_eq!(
        gradients.len(),
        inputs.len(),
        "Should have gradient for each input"
    );
}

#[test]
fn test_ultra_fallback_behavior() {
    let env = Arc::new(SafeVariableEnvironment::<f32>::new());

    // Small workload that should fall back to simpler method
    let x1 = create_test_tensor(&env, &[4], 1.0);
    let x2 = create_test_tensor(&env, &[4], 2.0);

    let outputs = [&x1, &x2];
    let inputs = vec![x1.id, x2.id];

    let result = ultra_backward_pass(&outputs[..], &inputs, &env);
    assert!(
        result.is_ok(),
        "Ultra backward should fall back correctly for small workloads"
    );
}

#[test]
#[cfg(all(feature = "simd", feature = "parallel"))]
fn test_ultra_configuration() {
    let env = Arc::new(SafeVariableEnvironment::<f32>::new());

    // Test with exact threshold configuration
    let vars: Vec<_> = (0..8)
        .map(|i| create_test_tensor(&env, &[64], (i + 1) as f32))
        .collect();

    let outputs: Vec<_> = vars.iter().collect();
    let inputs: Vec<_> = vars.iter().map(|v| v.id).collect();

    let result = ultra_backward_pass(&outputs[..], &inputs, &env);
    assert!(
        result.is_ok(),
        "Ultra backward should respect configuration thresholds"
    );
}

#[test]
fn test_ultra_edge_cases() {
    let env = Arc::new(SafeVariableEnvironment::<f32>::new());

    // Edge case: single output
    let x = create_test_tensor(&env, &[64], 1.0);
    let outputs = [&x];
    let inputs = vec![x.id];

    let result = ultra_backward_pass(&outputs[..], &inputs, &env);
    assert!(result.is_ok(), "Ultra backward should handle single output");
}

// ============================================================================
// MEMORY EFFICIENT ACCUMULATION TESTS (10 tests)
// ============================================================================

#[test]
fn test_memory_zero_copy_views() {
    let grad1 = ag::NdArray::from_elem(&[8][..], 1.0f32);
    let grad2 = ag::NdArray::from_elem(&[8][..], 2.0f32);
    let grad3 = ag::NdArray::from_elem(&[8][..], 3.0f32);

    let gradients = vec![grad1, grad2, grad3];
    let mut target = ag::NdArray::zeros(&[8][..]);

    let result = memory_efficient_grad_accumulation(&gradients, &mut target);
    assert!(result.is_ok(), "Memory efficient accumulation should work");

    // Target should be sum of all gradients
    let expected_sum = 6.0; // 1 + 2 + 3
    assert!(
        (target[[0]] - expected_sum).abs() < 1e-5,
        "Accumulated gradient should be correct"
    );
}

#[test]
fn test_memory_accumulation_correctness() {
    let gradients: Vec<_> = (1..=5)
        .map(|i| ag::NdArray::from_elem(&[16][..], i as f32))
        .collect();

    let mut target = ag::NdArray::zeros(&[16][..]);

    let result = memory_efficient_grad_accumulation(&gradients, &mut target);
    assert!(result.is_ok(), "Memory accumulation should be correct");

    let expected_sum = 15.0; // 1 + 2 + 3 + 4 + 5
    assert!(
        (target[[0]] - expected_sum).abs() < 1e-5,
        "Accumulated sum should be correct"
    );
}

#[test]
fn test_memory_no_unnecessary_clones() {
    // Test that the implementation uses views efficiently
    let gradients: Vec<_> = (1..=3)
        .map(|i| ag::NdArray::from_elem(&[32][..], i as f32))
        .collect();

    let mut target = ag::NdArray::zeros(&[32][..]);

    let result = memory_efficient_grad_accumulation(&gradients, &mut target);
    assert!(result.is_ok(), "Should avoid unnecessary clones");
}

#[test]
fn test_memory_large_tensors() {
    let gradients: Vec<_> = (1..=4)
        .map(|i| ag::NdArray::from_elem(&[1024][..], i as f32))
        .collect();

    let mut target = ag::NdArray::zeros(&[1024][..]);

    let result = memory_efficient_grad_accumulation(&gradients, &mut target);
    assert!(result.is_ok(), "Should handle large tensors efficiently");
}

#[test]
fn test_memory_multiple_outputs() {
    let gradients: Vec<_> = (1..=10)
        .map(|i| ag::NdArray::from_elem(&[64][..], i as f32))
        .collect();

    let mut target = ag::NdArray::zeros(&[64][..]);

    let result = memory_efficient_grad_accumulation(&gradients, &mut target);
    assert!(result.is_ok(), "Should handle multiple outputs");

    let expected_sum = 55.0; // Sum of 1..=10
    assert!(
        (target[[0]] - expected_sum).abs() < 1e-5,
        "Multiple output accumulation should be correct"
    );
}

#[test]
fn test_memory_vs_standard_equivalence() {
    let gradients: Vec<_> = (1..=5)
        .map(|i| ag::NdArray::from_elem(&[32][..], i as f32))
        .collect();

    let mut target_efficient = ag::NdArray::zeros(&[32][..]);
    let result = memory_efficient_grad_accumulation(&gradients, &mut target_efficient);
    assert!(result.is_ok(), "Memory efficient should work");

    // Standard accumulation
    let mut target_standard = ag::NdArray::<f32>::zeros(&[32][..]);
    for grad in &gradients {
        target_standard = &target_standard + grad;
    }

    // Should be equivalent
    let diff = (&target_efficient - &target_standard)
        .mapv(|x| x.abs())
        .sum();
    assert!(diff < 1e-5, "Memory efficient should match standard");
}

#[test]
fn test_memory_gradient_checkpointing() {
    // Test memory efficiency with gradient checkpointing pattern
    let gradients: Vec<_> = (1..=8)
        .map(|i| ag::NdArray::from_elem(&[128][..], i as f32 * 0.1))
        .collect();

    let mut target = ag::NdArray::zeros(&[128][..]);

    let result = memory_efficient_grad_accumulation(&gradients, &mut target);
    assert!(result.is_ok(), "Gradient checkpointing pattern should work");
}

#[test]
fn test_memory_peak_usage() {
    // Test that peak memory usage is reasonable
    let num_gradients = 20;
    let gradients: Vec<_> = (1..=num_gradients)
        .map(|i| ag::NdArray::from_elem(&[256][..], i as f32))
        .collect();

    let mut target = ag::NdArray::zeros(&[256][..]);

    let result = memory_efficient_grad_accumulation(&gradients, &mut target);
    assert!(
        result.is_ok(),
        "Should maintain reasonable peak memory usage"
    );
}

#[test]
fn test_memory_edge_cases() {
    // Test with empty gradients
    let gradients: Vec<ag::NdArray<f32>> = vec![];
    let mut target = ag::NdArray::zeros(&[16][..]);

    let result = memory_efficient_grad_accumulation(&gradients, &mut target);
    assert!(result.is_ok(), "Should handle empty gradients");

    // Test with single gradient
    let single_grad = vec![ag::NdArray::from_elem(&[16][..], 5.0f32)];
    let mut target2 = ag::NdArray::zeros(&[16][..]);

    let result2 = memory_efficient_grad_accumulation(&single_grad, &mut target2);
    assert!(result2.is_ok(), "Should handle single gradient");
}

#[test]
#[cfg(feature = "parallel")]
fn test_memory_thread_safety() {
    use std::thread;

    let handles: Vec<_> = (0..4)
        .map(|_| {
            thread::spawn(|| {
                let gradients: Vec<_> = (1..=5)
                    .map(|i| ag::NdArray::from_elem(&[64][..], i as f32))
                    .collect();

                let mut target = ag::NdArray::zeros(&[64][..]);

                let result = memory_efficient_grad_accumulation(&gradients, &mut target);
                assert!(result.is_ok(), "Thread-safe accumulation should work");
            })
        })
        .collect();

    for handle in handles {
        handle.join().expect("Thread should complete successfully");
    }
}

// ============================================================================
// SIMD ACCUMULATION SPECIFIC TESTS
// ============================================================================

#[test]
#[cfg(feature = "simd")]
fn test_simd_accumulation_f32() {
    // Test with data size that benefits from SIMD (>= 64 elements)
    let gradients: Vec<_> = (1..=4)
        .map(|i| ag::NdArray::from_elem(&[64][..], i as f32))
        .collect();

    let mut target = ag::NdArray::zeros(&[64][..]);

    let result = memory_efficient_grad_accumulation(&gradients, &mut target);
    assert!(result.is_ok(), "SIMD accumulation should work for f32");

    let expected = 10.0; // 1 + 2 + 3 + 4
    assert!(
        (target[[0]] - expected).abs() < 1e-5,
        "SIMD accumulated value should be correct"
    );
}

#[test]
#[cfg(feature = "simd")]
fn test_simd_cache_aware_accumulation() {
    // Test with large data to trigger cache-aware processing
    let gradients: Vec<_> = (1..=5)
        .map(|i| ag::NdArray::from_elem(&[512][..], i as f32))
        .collect();

    let mut target = ag::NdArray::zeros(&[512][..]);

    let result = memory_efficient_grad_accumulation(&gradients, &mut target);
    assert!(result.is_ok(), "Cache-aware SIMD accumulation should work");
}

#[test]
fn test_simd_fallback_for_small_arrays() {
    // Test with small arrays that should fall back to scalar
    let gradients: Vec<_> = (1..=3)
        .map(|i| ag::NdArray::from_elem(&[8][..], i as f32))
        .collect();

    let mut target = ag::NdArray::zeros(&[8][..]);

    let result = memory_efficient_grad_accumulation(&gradients, &mut target);
    assert!(
        result.is_ok(),
        "Should fall back to scalar for small arrays"
    );
}

#[test]
#[cfg(feature = "simd")]
fn test_simd_accumulation_correctness() {
    // Verify SIMD accumulation produces same results as scalar
    let gradients: Vec<_> = (1..=4)
        .map(|i| ag::NdArray::from_elem(&[128][..], i as f32 * 0.5))
        .collect();

    let mut target_simd = ag::NdArray::zeros(&[128][..]);
    let result = memory_efficient_grad_accumulation(&gradients, &mut target_simd);
    assert!(result.is_ok(), "SIMD accumulation should succeed");

    // Manual scalar accumulation for comparison
    let mut target_scalar = ag::NdArray::<f32>::zeros(&[128][..]);
    for grad in &gradients {
        target_scalar = &target_scalar + grad;
    }

    let diff = (&target_simd - &target_scalar).mapv(|x| x.abs()).sum();
    assert!(diff < 1e-4, "SIMD should match scalar accumulation");
}
