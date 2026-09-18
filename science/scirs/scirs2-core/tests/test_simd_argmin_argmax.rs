//! SIMD argmin/argmax integration tests
//!
//! This module tests the SIMD-accelerated argmin/argmax implementations
//! in scirs2-core, ensuring correctness and performance.

use scirs2_core::ndarray::{array, Array1, Array2};
use scirs2_core::ndarray_ext::reduction::{
    argmax_k, argmax_simd, argmin_k, argmin_simd, cumprod_simd, cumsum_simd,
};
#[cfg(feature = "random")]
use scirs2_core::random::Distribution;

// ============================================================================
// 1D argmin Tests
// ============================================================================

#[test]
fn test_argmin_simd_f64_basic() {
    let x = array![3.0f64, 1.0, 4.0, 1.0, 5.0, 9.0, 2.0];
    let idx = argmin_simd(&x.view()).expect("Test: operation failed");
    assert_eq!(idx, 1, "Should find first occurrence of minimum (1.0)");
}

#[test]
fn test_argmin_simd_f32_basic() {
    let x = array![3.0f32, 1.0, 4.0, 1.0, 5.0, 9.0, 2.0];
    let idx = argmin_simd(&x.view()).expect("Test: operation failed");
    assert_eq!(idx, 1, "Should find first occurrence of minimum (1.0)");
}

#[test]
fn test_argmin_simd_f64_negative() {
    let x = array![1.0f64, -5.0, 3.0, -2.0];
    let idx = argmin_simd(&x.view()).expect("Test: operation failed");
    assert_eq!(idx, 1, "Should find minimum negative value");
}

#[test]
fn test_argmin_simd_f32_negative() {
    let x = array![-10.0f32, -5.0, -20.0, -2.0];
    let idx = argmin_simd(&x.view()).expect("Test: operation failed");
    assert_eq!(idx, 2, "Should find most negative value");
}

#[test]
fn test_argmin_simd_f64_single() {
    let x = array![42.0f64];
    let idx = argmin_simd(&x.view()).expect("Test: operation failed");
    assert_eq!(idx, 0, "Single element should return index 0");
}

#[test]
fn test_argmin_simd_f32_single() {
    let x = array![42.0f32];
    let idx = argmin_simd(&x.view()).expect("Test: operation failed");
    assert_eq!(idx, 0, "Single element should return index 0");
}

#[test]
fn test_argmin_simd_empty() {
    let x: Array1<f64> = array![];
    assert_eq!(
        argmin_simd(&x.view()),
        None,
        "Empty array should return None"
    );
}

#[test]
fn test_argmin_simd_f64_large() {
    // Test with large array (> 1000 elements) to ensure SIMD path is used
    let mut data = vec![100.0f64; 10000];
    data[5000] = -999.0; // Minimum value at index 5000

    let x = Array1::from_vec(data);
    let idx = argmin_simd(&x.view()).expect("Test: operation failed");
    assert_eq!(idx, 5000, "Should find minimum in large array");
}

#[test]
fn test_argmin_simd_f32_large() {
    let mut data = vec![100.0f32; 10000];
    data[7500] = -999.0; // Minimum value at index 7500

    let x = Array1::from_vec(data);
    let idx = argmin_simd(&x.view()).expect("Test: operation failed");
    assert_eq!(idx, 7500, "Should find minimum in large f32 array");
}

#[test]
fn test_argmin_simd_f64_first_minimum() {
    let x = array![1.0f64, 2.0, 1.0, 3.0, 1.0];
    let idx = argmin_simd(&x.view()).expect("Test: operation failed");
    assert_eq!(
        idx, 0,
        "Should find FIRST occurrence when multiple minimums"
    );
}

#[test]
fn test_argmin_simd_f64_last_minimum() {
    let x = array![5.0f64, 4.0, 3.0, 2.0, 1.0];
    let idx = argmin_simd(&x.view()).expect("Test: operation failed");
    assert_eq!(idx, 4, "Should find minimum at last position");
}

// ============================================================================
// 1D argmax Tests
// ============================================================================

#[test]
fn test_argmax_simd_f64_basic() {
    let x = array![3.0f64, 1.0, 4.0, 1.0, 5.0, 9.0, 2.0];
    let idx = argmax_simd(&x.view()).expect("Test: operation failed");
    assert_eq!(idx, 5, "Should find maximum value (9.0)");
}

#[test]
fn test_argmax_simd_f32_basic() {
    let x = array![3.0f32, 1.0, 4.0, 1.0, 5.0, 9.0, 2.0];
    let idx = argmax_simd(&x.view()).expect("Test: operation failed");
    assert_eq!(idx, 5, "Should find maximum value (9.0)");
}

#[test]
fn test_argmax_simd_f64_negative() {
    let x = array![-10.0f64, -5.0, -20.0, -2.0];
    let idx = argmax_simd(&x.view()).expect("Test: operation failed");
    assert_eq!(idx, 3, "Should find least negative value (-2.0)");
}

#[test]
fn test_argmax_simd_f32_negative() {
    let x = array![-10.0f32, -5.0, -20.0, -2.0];
    let idx = argmax_simd(&x.view()).expect("Test: operation failed");
    assert_eq!(idx, 3, "Should find least negative value");
}

#[test]
fn test_argmax_simd_f64_single() {
    let x = array![42.0f64];
    let idx = argmax_simd(&x.view()).expect("Test: operation failed");
    assert_eq!(idx, 0, "Single element should return index 0");
}

#[test]
fn test_argmax_simd_f32_single() {
    let x = array![42.0f32];
    let idx = argmax_simd(&x.view()).expect("Test: operation failed");
    assert_eq!(idx, 0, "Single element should return index 0");
}

#[test]
fn test_argmax_simd_empty() {
    let x: Array1<f64> = array![];
    assert_eq!(
        argmax_simd(&x.view()),
        None,
        "Empty array should return None"
    );
}

#[test]
fn test_argmax_simd_f64_large() {
    // Test with large array (> 1000 elements) to ensure SIMD path is used
    let mut data = vec![-100.0f64; 10000];
    data[5000] = 999.0; // Maximum value at index 5000

    let x = Array1::from_vec(data);
    let idx = argmax_simd(&x.view()).expect("Test: operation failed");
    assert_eq!(idx, 5000, "Should find maximum in large array");
}

#[test]
fn test_argmax_simd_f32_large() {
    let mut data = vec![-100.0f32; 10000];
    data[7500] = 999.0; // Maximum value at index 7500

    let x = Array1::from_vec(data);
    let idx = argmax_simd(&x.view()).expect("Test: operation failed");
    assert_eq!(idx, 7500, "Should find maximum in large f32 array");
}

#[test]
fn test_argmax_simd_f64_first_maximum() {
    let x = array![9.0f64, 2.0, 9.0, 3.0, 9.0];
    let idx = argmax_simd(&x.view()).expect("Test: operation failed");
    assert_eq!(
        idx, 0,
        "Should find FIRST occurrence when multiple maximums"
    );
}

#[test]
fn test_argmax_simd_f64_last_maximum() {
    let x = array![1.0f64, 2.0, 3.0, 4.0, 5.0];
    let idx = argmax_simd(&x.view()).expect("Test: operation failed");
    assert_eq!(idx, 4, "Should find maximum at last position");
}

// ============================================================================
// argmin_k Tests
// ============================================================================

#[test]
fn test_argmin_k_basic() {
    let x = array![3.0f32, 1.0, 4.0, 1.0, 5.0, 9.0, 2.0];
    let indices = argmin_k(&x.view(), 3).expect("Test: operation failed");

    assert_eq!(indices.len(), 3, "Should return 3 indices");

    // Verify all returned indices have values <= third smallest
    for &idx in indices.iter() {
        assert!(
            x[idx] <= 2.0,
            "Index {} has value {}, expected <= 2.0",
            idx,
            x[idx]
        );
    }
}

#[test]
fn test_argmin_k_all_elements() {
    let x = array![3.0f32, 1.0, 4.0];
    let indices = argmin_k(&x.view(), 10).expect("Test: operation failed");

    // Should cap at array length
    assert_eq!(indices.len(), 3, "Should cap at array length");
}

#[test]
fn test_argmin_k_zero() {
    let x = array![1.0f64, 2.0, 3.0];
    assert_eq!(argmin_k(&x.view(), 0), None, "k=0 should return None");
}

#[test]
fn test_argmin_k_empty() {
    let x: Array1<f64> = array![];
    assert_eq!(
        argmin_k(&x.view(), 3),
        None,
        "Empty array should return None"
    );
}

#[test]
fn test_argmin_k_large() {
    let x: Array1<f32> = Array1::from_vec((0..10000).map(|i| (i as f32) % 100.0).collect());
    let indices = argmin_k(&x.view(), 5).expect("Test: operation failed");

    assert_eq!(indices.len(), 5, "Should return 5 indices");

    // All values should be 0.0 (minimum in the % 100 pattern)
    for &idx in indices.iter() {
        assert_eq!(x[idx], 0.0, "Index {} should have value 0.0", idx);
    }
}

// ============================================================================
// argmax_k Tests
// ============================================================================

#[test]
fn test_argmax_k_basic() {
    let x = array![3.0f32, 1.0, 4.0, 1.0, 5.0, 9.0, 2.0];
    let indices = argmax_k(&x.view(), 3).expect("Test: operation failed");

    assert_eq!(indices.len(), 3, "Should return 3 indices");

    // Verify all returned indices have values >= third largest
    for &idx in indices.iter() {
        assert!(
            x[idx] >= 4.0,
            "Index {} has value {}, expected >= 4.0",
            idx,
            x[idx]
        );
    }
}

#[test]
fn test_argmax_k_all_elements() {
    let x = array![3.0f32, 1.0, 4.0];
    let indices = argmax_k(&x.view(), 10).expect("Test: operation failed");

    // Should cap at array length
    assert_eq!(indices.len(), 3, "Should cap at array length");
}

#[test]
fn test_argmax_k_zero() {
    let x = array![1.0f64, 2.0, 3.0];
    assert_eq!(argmax_k(&x.view(), 0), None, "k=0 should return None");
}

#[test]
fn test_argmax_k_empty() {
    let x: Array1<f64> = array![];
    assert_eq!(
        argmax_k(&x.view(), 3),
        None,
        "Empty array should return None"
    );
}

#[test]
fn test_argmax_k_large() {
    let x: Array1<f32> = Array1::from_vec((0..10000).map(|i| (i as f32) % 100.0).collect());
    let indices = argmax_k(&x.view(), 5).expect("Test: operation failed");

    assert_eq!(indices.len(), 5, "Should return 5 indices");

    // All values should be 99.0 (maximum in the % 100 pattern)
    for &idx in indices.iter() {
        assert_eq!(x[idx], 99.0, "Index {} should have value 99.0", idx);
    }
}

// ============================================================================
// SIMD vs Scalar Equivalence Tests
// ============================================================================

#[cfg(feature = "random")]
#[test]
fn test_argmin_simd_equivalence_random() {
    use scirs2_core::random::{thread_rng, Uniform};

    let mut rng = thread_rng();
    let uniform = Uniform::new(-1000.0, 1000.0).expect("Test: operation failed");

    // Generate random data
    let data: Vec<f64> = (0..5000).map(|_| uniform.sample(&mut rng)).collect();
    let x = Array1::from_vec(data.clone());

    // Find minimum using SIMD
    let simd_idx = argmin_simd(&x.view()).expect("Test: operation failed");

    // Verify using scalar approach
    let mut scalar_idx = 0;
    let mut min_val = data[0];
    for (i, &val) in data.iter().enumerate() {
        if val < min_val {
            min_val = val;
            scalar_idx = i;
        }
    }

    assert_eq!(
        simd_idx, scalar_idx,
        "SIMD result should match scalar result"
    );
    assert_eq!(
        x[simd_idx], min_val,
        "SIMD result should give correct minimum value"
    );
}

#[cfg(feature = "random")]
#[test]
fn test_argmax_simd_equivalence_random() {
    use scirs2_core::random::{thread_rng, Uniform};

    let mut rng = thread_rng();
    let uniform = Uniform::new(-1000.0, 1000.0).expect("Test: operation failed");

    // Generate random data
    let data: Vec<f64> = (0..5000).map(|_| uniform.sample(&mut rng)).collect();
    let x = Array1::from_vec(data.clone());

    // Find maximum using SIMD
    let simd_idx = argmax_simd(&x.view()).expect("Test: operation failed");

    // Verify using scalar approach
    let mut scalar_idx = 0;
    let mut max_val = data[0];
    for (i, &val) in data.iter().enumerate() {
        if val > max_val {
            max_val = val;
            scalar_idx = i;
        }
    }

    assert_eq!(
        simd_idx, scalar_idx,
        "SIMD result should match scalar result"
    );
    assert_eq!(
        x[simd_idx], max_val,
        "SIMD result should give correct maximum value"
    );
}

// ============================================================================
// 2D argmin_simd Tests (from manipulation module)
// ============================================================================

#[test]
fn test_argmin_simd_2d_axis_none() {
    use scirs2_core::ndarray_ext::manipulation::argmin_simd as argmin_simd_2d;

    let a = array![[5.0f32, 2.0, 3.0], [4.0, 1.0, 6.0]];

    let result = argmin_simd_2d(a.view(), None).expect("Test: operation failed");
    assert_eq!(
        result[0], 4,
        "Should find global minimum at index 4 (value 1.0)"
    );
}

#[test]
fn test_argmin_simd_2d_axis_0() {
    use scirs2_core::ndarray_ext::manipulation::argmin_simd as argmin_simd_2d;

    let a = array![[5.0f32, 2.0, 3.0], [4.0, 1.0, 6.0]];

    let result = argmin_simd_2d(a.view(), Some(0)).expect("Test: operation failed");
    assert_eq!(result.len(), 3, "Should return 3 column minimums");
    assert_eq!(result[0], 1, "Column 0 minimum at row 1");
    assert_eq!(result[1], 1, "Column 1 minimum at row 1");
    assert_eq!(result[2], 0, "Column 2 minimum at row 0");
}

#[test]
fn test_argmin_simd_2d_axis_1() {
    use scirs2_core::ndarray_ext::manipulation::argmin_simd as argmin_simd_2d;

    let a = array![[5.0f32, 2.0, 3.0], [4.0, 1.0, 6.0]];

    let result = argmin_simd_2d(a.view(), Some(1)).expect("Test: operation failed");
    assert_eq!(result.len(), 2, "Should return 2 row minimums");
    assert_eq!(result[0], 1, "Row 0 minimum at col 1");
    assert_eq!(result[1], 1, "Row 1 minimum at col 1");
}

// ============================================================================
// 2D argmax_simd Tests (from manipulation module)
// ============================================================================

#[test]
fn test_argmax_simd_2d_axis_none() {
    use scirs2_core::ndarray_ext::manipulation::argmax_simd as argmax_simd_2d;

    let a = array![[5.0f32, 2.0, 3.0], [4.0, 1.0, 6.0]];

    let result = argmax_simd_2d(a.view(), None).expect("Test: operation failed");
    assert_eq!(
        result[0], 5,
        "Should find global maximum at index 5 (value 6.0)"
    );
}

#[test]
fn test_argmax_simd_2d_axis_0() {
    use scirs2_core::ndarray_ext::manipulation::argmax_simd as argmax_simd_2d;

    let a = array![[5.0f32, 2.0, 3.0], [4.0, 1.0, 6.0]];

    let result = argmax_simd_2d(a.view(), Some(0)).expect("Test: operation failed");
    assert_eq!(result.len(), 3, "Should return 3 column maximums");
    assert_eq!(result[0], 0, "Column 0 maximum at row 0");
    assert_eq!(result[1], 0, "Column 1 maximum at row 0");
    assert_eq!(result[2], 1, "Column 2 maximum at row 1");
}

#[test]
fn test_argmax_simd_2d_axis_1() {
    use scirs2_core::ndarray_ext::manipulation::argmax_simd as argmax_simd_2d;

    let a = array![[5.0f32, 2.0, 3.0], [4.0, 1.0, 6.0]];

    let result = argmax_simd_2d(a.view(), Some(1)).expect("Test: operation failed");
    assert_eq!(result.len(), 2, "Should return 2 row maximums");
    assert_eq!(result[0], 0, "Row 0 maximum at col 0");
    assert_eq!(result[1], 2, "Row 1 maximum at col 2");
}

#[test]
fn test_argmax_simd_2d_large() {
    use scirs2_core::ndarray_ext::manipulation::argmax_simd as argmax_simd_2d;

    // Test with large 2D array
    let mut data = vec![1.0f32; 1000 * 100];
    data[50 * 100 + 37] = 999.0; // Maximum at row 50, col 37

    let a = Array2::from_shape_vec((1000, 100), data).expect("Test: operation failed");

    let result = argmax_simd_2d(a.view(), None).expect("Test: operation failed");
    assert_eq!(
        result[0],
        50 * 100 + 37,
        "Should find maximum in large 2D array"
    );
}

// ============================================================================
// Cumulative Sum Tests
// ============================================================================

#[test]
fn test_cumsum_simd_f64_basic() {
    let x = array![1.0f64, 2.0, 3.0, 4.0, 5.0];
    let result = cumsum_simd(&x.view());

    assert_eq!(result.len(), 5);
    assert_eq!(result[0], 1.0);
    assert_eq!(result[1], 3.0);
    assert_eq!(result[2], 6.0);
    assert_eq!(result[3], 10.0);
    assert_eq!(result[4], 15.0);
}

#[test]
fn test_cumsum_simd_f32_basic() {
    let x = array![1.0f32, 2.0, 3.0, 4.0, 5.0];
    let result = cumsum_simd(&x.view());

    assert_eq!(result.len(), 5);
    assert!((result[0] - 1.0).abs() < 1e-6);
    assert!((result[1] - 3.0).abs() < 1e-6);
    assert!((result[2] - 6.0).abs() < 1e-6);
    assert!((result[3] - 10.0).abs() < 1e-6);
    assert!((result[4] - 15.0).abs() < 1e-6);
}

#[test]
fn test_cumsum_simd_negative() {
    let x = array![1.0f64, -2.0, 3.0, -4.0, 5.0];
    let result = cumsum_simd(&x.view());

    assert_eq!(result.len(), 5);
    assert_eq!(result[0], 1.0);
    assert_eq!(result[1], -1.0);
    assert_eq!(result[2], 2.0);
    assert_eq!(result[3], -2.0);
    assert_eq!(result[4], 3.0);
}

#[test]
fn test_cumsum_simd_zeros() {
    let x = array![0.0f64, 0.0, 0.0, 0.0];
    let result = cumsum_simd(&x.view());

    assert_eq!(result.len(), 4);
    for &val in result.iter() {
        assert_eq!(val, 0.0);
    }
}

#[test]
fn test_cumsum_simd_single() {
    let x = array![42.0f64];
    let result = cumsum_simd(&x.view());

    assert_eq!(result.len(), 1);
    assert_eq!(result[0], 42.0);
}

#[test]
fn test_cumsum_simd_empty() {
    let x: Array1<f64> = array![];
    let result = cumsum_simd(&x.view());

    assert_eq!(result.len(), 0);
}

#[test]
fn test_cumsum_simd_large() {
    // Test with large array to ensure SIMD path is used
    let x: Array1<f64> = Array1::from_vec((1..=10000).map(|i| i as f64).collect());
    let result = cumsum_simd(&x.view());

    assert_eq!(result.len(), 10000);
    assert_eq!(result[0], 1.0);
    assert_eq!(result[9999], 50005000.0); // Sum of 1..10000 = n*(n+1)/2 = 10000*10001/2
}

#[test]
fn test_cumsum_simd_alternating() {
    let x = array![1.0f32, -1.0, 1.0, -1.0, 1.0];
    let result = cumsum_simd(&x.view());

    assert_eq!(result.len(), 5);
    assert!((result[0] - 1.0).abs() < 1e-6);
    assert!((result[1] - 0.0).abs() < 1e-6);
    assert!((result[2] - 1.0).abs() < 1e-6);
    assert!((result[3] - 0.0).abs() < 1e-6);
    assert!((result[4] - 1.0).abs() < 1e-6);
}

// ============================================================================
// Cumulative Product Tests
// ============================================================================

#[test]
fn test_cumprod_simd_f64_basic() {
    let x = array![1.0f64, 2.0, 3.0, 4.0, 5.0];
    let result = cumprod_simd(&x.view());

    assert_eq!(result.len(), 5);
    assert_eq!(result[0], 1.0);
    assert_eq!(result[1], 2.0);
    assert_eq!(result[2], 6.0);
    assert_eq!(result[3], 24.0);
    assert_eq!(result[4], 120.0);
}

#[test]
fn test_cumprod_simd_f32_basic() {
    let x = array![1.0f32, 2.0, 3.0, 4.0, 5.0];
    let result = cumprod_simd(&x.view());

    assert_eq!(result.len(), 5);
    assert!((result[0] - 1.0).abs() < 1e-6);
    assert!((result[1] - 2.0).abs() < 1e-6);
    assert!((result[2] - 6.0).abs() < 1e-6);
    assert!((result[3] - 24.0).abs() < 1e-6);
    assert!((result[4] - 120.0).abs() < 1e-6);
}

#[test]
fn test_cumprod_simd_with_zero() {
    let x = array![1.0f64, 2.0, 0.0, 4.0, 5.0];
    let result = cumprod_simd(&x.view());

    assert_eq!(result.len(), 5);
    assert_eq!(result[0], 1.0);
    assert_eq!(result[1], 2.0);
    assert_eq!(result[2], 0.0);
    assert_eq!(result[3], 0.0); // Once zero, stays zero
    assert_eq!(result[4], 0.0);
}

#[test]
fn test_cumprod_simd_negative() {
    let x = array![1.0f64, -2.0, 3.0, -4.0];
    let result = cumprod_simd(&x.view());

    assert_eq!(result.len(), 4);
    assert_eq!(result[0], 1.0);
    assert_eq!(result[1], -2.0);
    assert_eq!(result[2], -6.0);
    assert_eq!(result[3], 24.0); // negative * negative = positive
}

#[test]
fn test_cumprod_simd_fractional() {
    let x = array![2.0f64, 0.5, 0.5, 2.0];
    let result = cumprod_simd(&x.view());

    assert_eq!(result.len(), 4);
    assert!((result[0] - 2.0).abs() < 1e-10);
    assert!((result[1] - 1.0).abs() < 1e-10);
    assert!((result[2] - 0.5).abs() < 1e-10);
    assert!((result[3] - 1.0).abs() < 1e-10);
}

#[test]
fn test_cumprod_simd_ones() {
    let x = array![1.0f64, 1.0, 1.0, 1.0];
    let result = cumprod_simd(&x.view());

    assert_eq!(result.len(), 4);
    for &val in result.iter() {
        assert_eq!(val, 1.0);
    }
}

#[test]
fn test_cumprod_simd_single() {
    let x = array![42.0f64];
    let result = cumprod_simd(&x.view());

    assert_eq!(result.len(), 1);
    assert_eq!(result[0], 42.0);
}

#[test]
fn test_cumprod_simd_empty() {
    let x: Array1<f64> = array![];
    let result = cumprod_simd(&x.view());

    assert_eq!(result.len(), 0);
}

#[test]
fn test_cumprod_simd_large() {
    // Test with large array to ensure SIMD path is used
    // Use small values to avoid overflow
    let x: Array1<f64> = Array1::from_vec(vec![1.001; 1000]);
    let result = cumprod_simd(&x.view());

    assert_eq!(result.len(), 1000);
    assert_eq!(result[0], 1.001);
    // Product of 1.001^1000 ≈ 2.717 (close to e)
    assert!((result[999] - 2.717).abs() < 0.01);
}

#[test]
fn test_cumprod_simd_powers_of_two() {
    let x = array![2.0f32, 2.0, 2.0, 2.0, 2.0];
    let result = cumprod_simd(&x.view());

    assert_eq!(result.len(), 5);
    assert!((result[0] - 2.0).abs() < 1e-6);
    assert!((result[1] - 4.0).abs() < 1e-6);
    assert!((result[2] - 8.0).abs() < 1e-6);
    assert!((result[3] - 16.0).abs() < 1e-6);
    assert!((result[4] - 32.0).abs() < 1e-6);
}

// ============================================================================
// SIMD vs Scalar Equivalence Tests for Cumulative Operations
// ============================================================================

#[cfg(feature = "random")]
#[test]
fn test_cumsum_simd_equivalence_random() {
    use scirs2_core::random::{thread_rng, Uniform};

    let mut rng = thread_rng();
    let uniform = Uniform::new(-100.0, 100.0).expect("Test: operation failed");

    // Generate random data
    let data: Vec<f64> = (0..5000).map(|_| uniform.sample(&mut rng)).collect();
    let x = Array1::from_vec(data.clone());

    // Compute cumsum using SIMD
    let simd_result = cumsum_simd(&x.view());

    // Verify using scalar approach
    let mut cumsum = 0.0;
    for (i, &val) in data.iter().enumerate() {
        cumsum += val;
        assert!(
            (simd_result[i] - cumsum).abs() < 1e-8,
            "SIMD cumsum mismatch at index {}: SIMD={}, Scalar={}",
            i,
            simd_result[i],
            cumsum
        );
    }
}

#[cfg(feature = "random")]
#[test]
fn test_cumprod_simd_equivalence_random() {
    use scirs2_core::random::{thread_rng, Uniform};

    let mut rng = thread_rng();
    let uniform = Uniform::new(0.95, 1.05).expect("Test: operation failed"); // Small range to avoid overflow/underflow

    // Generate random data
    let data: Vec<f64> = (0..5000).map(|_| uniform.sample(&mut rng)).collect();
    let x = Array1::from_vec(data.clone());

    // Compute cumprod using SIMD
    let simd_result = cumprod_simd(&x.view());

    // Verify using scalar approach
    let mut cumprod: f64 = 1.0;
    for (i, &val) in data.iter().enumerate() {
        cumprod *= val;
        let relative_error = if cumprod.abs() > 1e-10 {
            ((simd_result[i] - cumprod) / cumprod).abs()
        } else {
            (simd_result[i] - cumprod).abs()
        };
        assert!(
            relative_error < 1e-10,
            "SIMD cumprod mismatch at index {}: SIMD={}, Scalar={}, Relative Error={}",
            i,
            simd_result[i],
            cumprod,
            relative_error
        );
    }
}
