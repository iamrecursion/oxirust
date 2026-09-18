//! Statistical function SIMD tests: statistical, reduction, norm, binary, normalization

use scirs2_core::ndarray::{array, Array1};
use scirs2_core::ndarray_ext::elementwise::{
    abs_simd, exp_simd, ln_simd, sigmoid_simd, softplus_simd, sqrt_simd, tanh_simd,
};
#[cfg(feature = "random")]
use scirs2_core::random::{thread_rng, Distribution, Uniform};

// =============================================================================
// Phase 70: Statistical Functions Tests
// =============================================================================

use scirs2_core::ndarray_ext::elementwise::{
    add_simd, argmax_simd, argmin_simd, cosine_similarity_simd, distance_chebyshev_simd,
    distance_cosine_simd, distance_euclidean_simd, distance_manhattan_simd, div_simd, dot_simd,
    fma_simd, log_sum_exp_simd, max_element_simd, max_simd, mean_simd, min_element_simd, min_simd,
    mul_simd, norm_l1_simd, norm_linf_simd, norm_simd, normalize_simd, relu_simd, scalar_mul_simd,
    softmax_simd, standardize_simd, std_simd, sub_simd, sum_cubes_simd, sum_simd, sum_squares_simd,
    trunc_simd, variance_simd, weighted_mean_simd, weighted_sum_simd,
};

/// Test variance_simd with known values
#[test]
fn test_variance_simd_basic() {
    // Sample variance (Bessel's correction) of [2,4,4,4,5,5,7,9]
    // mean = 5, sum of squared deviations = 32, n = 8
    // sample variance = 32/(8-1) = 32/7 ≈ 4.571
    let x = array![2.0_f64, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0];
    let var = variance_simd::<f64>(&x.view());
    let expected = 32.0 / 7.0;
    assert!(
        (var - expected).abs() < 1e-10,
        "Variance should be {}, got {}",
        expected,
        var
    );
}

/// Test variance_simd f32
#[test]
fn test_variance_simd_f32() {
    let x = array![1.0_f32, 2.0, 3.0, 4.0, 5.0];
    let var = variance_simd::<f32>(&x.view());
    // Sample variance: Mean = 3, sum of squared deviations = 10
    // sample variance = 10/(5-1) = 2.5
    let expected = 10.0 / 4.0;
    assert!(
        (var - expected).abs() < 1e-5,
        "Variance should be {}, got {}",
        expected,
        var
    );
}

/// Test variance_simd empty array
#[test]
fn test_variance_simd_empty() {
    let x: Array1<f64> = Array1::zeros(0);
    let var = variance_simd::<f64>(&x.view());
    assert!(var.abs() < 1e-14, "Empty array variance should be 0");
}

/// Test std_simd with known values
#[test]
fn test_std_simd_basic() {
    // Sample std = sqrt(sample variance) = sqrt(32/7) ≈ 2.138
    let x = array![2.0_f64, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0];
    let s = std_simd::<f64>(&x.view());
    let expected = (32.0_f64 / 7.0).sqrt();
    assert!(
        (s - expected).abs() < 1e-10,
        "Std should be {}, got {}",
        expected,
        s
    );
}

/// Test std_simd f32
#[test]
fn test_std_simd_f32() {
    let x = array![1.0_f32, 2.0, 3.0, 4.0, 5.0];
    let s = std_simd::<f32>(&x.view());
    // Sample std = sqrt(10/4) = sqrt(2.5) ≈ 1.581
    let expected = (10.0_f32 / 4.0).sqrt();
    assert!(
        (s - expected).abs() < 1e-5,
        "Std should be {}, got {}",
        expected,
        s
    );
}

/// Test sum_simd basic
#[test]
fn test_sum_simd_basic() {
    let x = array![1.0_f64, 2.0, 3.0, 4.0, 5.0];
    let s = sum_simd::<f64>(&x.view());
    assert!((s - 15.0).abs() < 1e-14, "Sum should be 15.0, got {}", s);
}

/// Test sum_simd f32
#[test]
fn test_sum_simd_f32() {
    let x = array![10.0_f32, 20.0, 30.0];
    let s = sum_simd::<f32>(&x.view());
    assert!((s - 60.0).abs() < 1e-5, "Sum should be 60.0, got {}", s);
}

/// Test sum_simd large array
#[test]
fn test_sum_simd_large() {
    let n = 10000;
    let x = Array1::from_vec((1..=n).map(|i| i as f64).collect());
    let s = sum_simd::<f64>(&x.view());
    // Sum of 1 to n = n*(n+1)/2
    let expected = (n * (n + 1) / 2) as f64;
    assert!(
        (s - expected).abs() < 1e-8,
        "Sum should be {}, got {}",
        expected,
        s
    );
}

/// Test mean_simd basic
#[test]
fn test_mean_simd_basic() {
    let x = array![1.0_f64, 2.0, 3.0, 4.0, 5.0];
    let m = mean_simd::<f64>(&x.view());
    assert!((m - 3.0).abs() < 1e-14, "Mean should be 3.0, got {}", m);
}

/// Test mean_simd f32
#[test]
fn test_mean_simd_f32() {
    let x = array![2.0_f32, 4.0, 6.0, 8.0];
    let m = mean_simd::<f32>(&x.view());
    assert!((m - 5.0).abs() < 1e-5, "Mean should be 5.0, got {}", m);
}

/// Test weighted_sum_simd basic
#[test]
fn test_weighted_sum_simd_basic() {
    let values = array![10.0_f64, 20.0, 30.0];
    let weights = array![0.2_f64, 0.3, 0.5];
    let ws = weighted_sum_simd::<f64>(&values.view(), &weights.view());
    // 10*0.2 + 20*0.3 + 30*0.5 = 2 + 6 + 15 = 23
    assert!(
        (ws - 23.0).abs() < 1e-10,
        "Weighted sum should be 23.0, got {}",
        ws
    );
}

/// Test weighted_sum_simd f32
#[test]
fn test_weighted_sum_simd_f32() {
    let values = array![1.0_f32, 2.0, 3.0];
    let weights = array![1.0_f32, 1.0, 1.0];
    let ws = weighted_sum_simd::<f32>(&values.view(), &weights.view());
    assert!(
        (ws - 6.0).abs() < 1e-5,
        "Weighted sum should be 6.0, got {}",
        ws
    );
}

/// Test weighted_mean_simd basic
#[test]
fn test_weighted_mean_simd_basic() {
    let values = array![10.0_f64, 20.0, 30.0];
    let weights = array![1.0_f64, 2.0, 2.0];
    let wm = weighted_mean_simd::<f64>(&values.view(), &weights.view());
    // (10*1 + 20*2 + 30*2) / (1+2+2) = 110/5 = 22
    assert!(
        (wm - 22.0).abs() < 1e-10,
        "Weighted mean should be 22.0, got {}",
        wm
    );
}

/// Test weighted_mean_simd f32
#[test]
fn test_weighted_mean_simd_f32() {
    let values = array![100.0_f32, 200.0];
    let weights = array![3.0_f32, 1.0];
    let wm = weighted_mean_simd::<f32>(&values.view(), &weights.view());
    // (100*3 + 200*1) / (3+1) = 500/4 = 125
    assert!(
        (wm - 125.0).abs() < 1e-4,
        "Weighted mean should be 125.0, got {}",
        wm
    );
}

// =============================================================================
// Phase 71: Reduction Functions Tests
// =============================================================================

/// Test max_element_simd basic
#[test]
fn test_max_element_simd_basic() {
    let x = array![3.0_f64, 1.0, 4.0, 1.0, 5.0, 9.0, 2.0, 6.0];
    let max = max_element_simd::<f64>(&x.view());
    assert!((max - 9.0).abs() < 1e-14, "Max should be 9.0, got {}", max);
}

/// Test max_element_simd f32
#[test]
fn test_max_element_simd_f32() {
    let x = array![-5.0_f32, -3.0, -1.0, -7.0];
    let max = max_element_simd::<f32>(&x.view());
    assert!(
        (max - (-1.0)).abs() < 1e-5,
        "Max should be -1.0, got {}",
        max
    );
}

/// Test min_element_simd basic
#[test]
fn test_min_element_simd_basic() {
    let x = array![3.0_f64, 1.0, 4.0, 1.0, 5.0, 9.0, 2.0, 6.0];
    let min = min_element_simd::<f64>(&x.view());
    assert!((min - 1.0).abs() < 1e-14, "Min should be 1.0, got {}", min);
}

/// Test min_element_simd f32
#[test]
fn test_min_element_simd_f32() {
    let x = array![5.0_f32, 3.0, 7.0, 2.0, 8.0];
    let min = min_element_simd::<f32>(&x.view());
    assert!((min - 2.0).abs() < 1e-5, "Min should be 2.0, got {}", min);
}

/// Test argmax_simd basic
#[test]
fn test_argmax_simd_basic() {
    let x = array![3.0_f64, 1.0, 4.0, 1.0, 5.0, 9.0, 2.0, 6.0];
    let idx = argmax_simd::<f64>(&x.view());
    assert_eq!(idx, Some(5), "Argmax should be 5 (x[5] = 9.0)");
}

/// Test argmax_simd f32
#[test]
fn test_argmax_simd_f32() {
    let x = array![1.0_f32, 2.0, 5.0, 3.0];
    let idx = argmax_simd::<f32>(&x.view());
    assert_eq!(idx, Some(2), "Argmax should be 2 (x[2] = 5.0)");
}

/// Test argmax_simd empty
#[test]
fn test_argmax_simd_empty() {
    let x: Array1<f64> = Array1::zeros(0);
    let idx = argmax_simd::<f64>(&x.view());
    assert_eq!(idx, None, "Empty array argmax should be None");
}

/// Test argmin_simd basic
#[test]
fn test_argmin_simd_basic() {
    let x = array![3.0_f64, 1.0, 4.0, 1.0, 5.0, 9.0, 2.0, 6.0];
    let idx = argmin_simd::<f64>(&x.view());
    assert_eq!(
        idx,
        Some(1),
        "Argmin should be 1 (x[1] = 1.0, first occurrence)"
    );
}

/// Test argmin_simd f32
#[test]
fn test_argmin_simd_f32() {
    let x = array![4.0_f32, 2.0, 5.0, 1.0, 3.0];
    let idx = argmin_simd::<f32>(&x.view());
    assert_eq!(idx, Some(3), "Argmin should be 3 (x[3] = 1.0)");
}

/// Test sum_squares_simd basic
#[test]
fn test_sum_squares_simd_basic() {
    let x = array![1.0_f64, 2.0, 3.0, 4.0];
    let ss = sum_squares_simd::<f64>(&x.view());
    // 1 + 4 + 9 + 16 = 30
    assert!(
        (ss - 30.0).abs() < 1e-14,
        "Sum of squares should be 30, got {}",
        ss
    );
}

/// Test sum_squares_simd f32
#[test]
fn test_sum_squares_simd_f32() {
    let x = array![3.0_f32, 4.0];
    let ss = sum_squares_simd::<f32>(&x.view());
    // 9 + 16 = 25
    assert!(
        (ss - 25.0).abs() < 1e-5,
        "Sum of squares should be 25, got {}",
        ss
    );
}

/// Test sum_cubes_simd basic
#[test]
fn test_sum_cubes_simd_basic() {
    let x = array![1.0_f64, 2.0, 3.0];
    let sc = sum_cubes_simd::<f64>(&x.view());
    // 1 + 8 + 27 = 36
    assert!(
        (sc - 36.0).abs() < 1e-14,
        "Sum of cubes should be 36, got {}",
        sc
    );
}

/// Test sum_cubes_simd f32
#[test]
fn test_sum_cubes_simd_f32() {
    let x = array![2.0_f32, 3.0];
    let sc = sum_cubes_simd::<f32>(&x.view());
    // 8 + 27 = 35
    assert!(
        (sc - 35.0).abs() < 1e-4,
        "Sum of cubes should be 35, got {}",
        sc
    );
}

/// Test log_sum_exp_simd basic
#[test]
fn test_log_sum_exp_simd_basic() {
    let x = array![1.0_f64, 2.0, 3.0];
    let lse = log_sum_exp_simd::<f64>(&x.view());
    // log(e^1 + e^2 + e^3)
    let expected = (1.0_f64.exp() + 2.0_f64.exp() + 3.0_f64.exp()).ln();
    assert!(
        (lse - expected).abs() < 1e-10,
        "Log-sum-exp should be {}, got {}",
        expected,
        lse
    );
}

/// Test log_sum_exp_simd numerical stability
#[test]
fn test_log_sum_exp_simd_stability() {
    // Large values that would overflow naive exp
    let x = array![1000.0_f64, 1001.0, 1002.0];
    let lse = log_sum_exp_simd::<f64>(&x.view());
    // Should be approximately 1002 + log(1 + e^-1 + e^-2)
    assert!(
        lse > 1000.0 && lse < 1003.0,
        "Log-sum-exp should handle large values stably, got {}",
        lse
    );
}

/// Test log_sum_exp_simd f32
#[test]
fn test_log_sum_exp_simd_f32() {
    let x = array![0.0_f32, 1.0, 2.0];
    let lse = log_sum_exp_simd::<f32>(&x.view());
    let expected = (1.0_f32 + 1.0_f32.exp() + 2.0_f32.exp()).ln();
    assert!(
        (lse - expected).abs() < 1e-5,
        "Log-sum-exp should be {}, got {}",
        expected,
        lse
    );
}

// =============================================================================
// Phase 72: Norm and Distance Functions Tests
// =============================================================================

/// Test norm_simd (L2) basic
#[test]
fn test_norm_simd_basic() {
    let x = array![3.0_f64, 4.0];
    let n = norm_simd::<f64>(&x.view());
    assert!((n - 5.0).abs() < 1e-14, "L2 norm should be 5.0, got {}", n);
}

/// Test norm_simd f32
#[test]
fn test_norm_simd_f32() {
    let x = array![1.0_f32, 2.0, 2.0];
    let n = norm_simd::<f32>(&x.view());
    // sqrt(1 + 4 + 4) = 3
    assert!((n - 3.0).abs() < 1e-5, "L2 norm should be 3.0, got {}", n);
}

/// Test norm_l1_simd basic
#[test]
fn test_norm_l1_simd_basic() {
    let x = array![1.0_f64, -2.0, 3.0, -4.0];
    let n = norm_l1_simd::<f64>(&x.view());
    // |1| + |-2| + |3| + |-4| = 10
    assert!(
        (n - 10.0).abs() < 1e-14,
        "L1 norm should be 10.0, got {}",
        n
    );
}

/// Test norm_l1_simd f32
#[test]
fn test_norm_l1_simd_f32() {
    let x = array![-3.0_f32, 4.0];
    let n = norm_l1_simd::<f32>(&x.view());
    assert!((n - 7.0).abs() < 1e-5, "L1 norm should be 7.0, got {}", n);
}

/// Test norm_linf_simd basic
#[test]
fn test_norm_linf_simd_basic() {
    let x = array![1.0_f64, -5.0, 3.0, -2.0];
    let n = norm_linf_simd::<f64>(&x.view());
    // max(|1|, |-5|, |3|, |-2|) = 5
    assert!(
        (n - 5.0).abs() < 1e-14,
        "L-inf norm should be 5.0, got {}",
        n
    );
}

/// Test norm_linf_simd f32
#[test]
fn test_norm_linf_simd_f32() {
    let x = array![2.0_f32, -7.0, 4.0];
    let n = norm_linf_simd::<f32>(&x.view());
    assert!(
        (n - 7.0).abs() < 1e-5,
        "L-inf norm should be 7.0, got {}",
        n
    );
}

/// Test distance_euclidean_simd basic
#[test]
fn test_distance_euclidean_simd_basic() {
    let a = array![0.0_f64, 0.0];
    let b = array![3.0_f64, 4.0];
    let d = distance_euclidean_simd::<f64>(&a.view(), &b.view());
    assert!(
        (d - 5.0).abs() < 1e-14,
        "Euclidean distance should be 5.0, got {}",
        d
    );
}

/// Test distance_euclidean_simd f32
#[test]
fn test_distance_euclidean_simd_f32() {
    let a = array![1.0_f32, 1.0, 1.0];
    let b = array![2.0_f32, 2.0, 2.0];
    let d = distance_euclidean_simd::<f32>(&a.view(), &b.view());
    // sqrt(3) ≈ 1.732
    assert!(
        (d - 3.0_f32.sqrt()).abs() < 1e-5,
        "Euclidean distance should be sqrt(3), got {}",
        d
    );
}

/// Test distance_manhattan_simd basic
#[test]
fn test_distance_manhattan_simd_basic() {
    let a = array![1.0_f64, 2.0, 3.0];
    let b = array![4.0_f64, 0.0, 3.0];
    let d = distance_manhattan_simd::<f64>(&a.view(), &b.view());
    // |1-4| + |2-0| + |3-3| = 3 + 2 + 0 = 5
    assert!(
        (d - 5.0).abs() < 1e-14,
        "Manhattan distance should be 5.0, got {}",
        d
    );
}

/// Test distance_manhattan_simd f32
#[test]
fn test_distance_manhattan_simd_f32() {
    let a = array![0.0_f32, 0.0];
    let b = array![3.0_f32, 4.0];
    let d = distance_manhattan_simd::<f32>(&a.view(), &b.view());
    assert!(
        (d - 7.0).abs() < 1e-5,
        "Manhattan distance should be 7.0, got {}",
        d
    );
}

/// Test distance_chebyshev_simd basic
#[test]
fn test_distance_chebyshev_simd_basic() {
    let a = array![1.0_f64, 2.0, 3.0];
    let b = array![4.0_f64, 0.0, 3.0];
    let d = distance_chebyshev_simd::<f64>(&a.view(), &b.view());
    // max(|1-4|, |2-0|, |3-3|) = max(3, 2, 0) = 3
    assert!(
        (d - 3.0).abs() < 1e-14,
        "Chebyshev distance should be 3.0, got {}",
        d
    );
}

/// Test distance_chebyshev_simd f32
#[test]
fn test_distance_chebyshev_simd_f32() {
    let a = array![0.0_f32, 0.0];
    let b = array![3.0_f32, 4.0];
    let d = distance_chebyshev_simd::<f32>(&a.view(), &b.view());
    assert!(
        (d - 4.0).abs() < 1e-5,
        "Chebyshev distance should be 4.0, got {}",
        d
    );
}

/// Test distance_cosine_simd orthogonal
#[test]
fn test_distance_cosine_simd_orthogonal() {
    let a = array![1.0_f64, 0.0, 0.0];
    let b = array![0.0_f64, 1.0, 0.0];
    let d = distance_cosine_simd::<f64>(&a.view(), &b.view());
    // Orthogonal vectors: cosine similarity = 0, distance = 1
    assert!(
        (d - 1.0).abs() < 1e-10,
        "Cosine distance for orthogonal vectors should be 1.0, got {}",
        d
    );
}

/// Test distance_cosine_simd parallel
#[test]
fn test_distance_cosine_simd_parallel() {
    let a = array![1.0_f64, 2.0, 3.0];
    let b = array![2.0_f64, 4.0, 6.0];
    let d = distance_cosine_simd::<f64>(&a.view(), &b.view());
    // Parallel vectors: cosine similarity = 1, distance = 0
    assert!(
        (d - 0.0).abs() < 1e-10,
        "Cosine distance for parallel vectors should be 0.0, got {}",
        d
    );
}

/// Test cosine_similarity_simd parallel
#[test]
fn test_cosine_similarity_simd_parallel() {
    let a = array![1.0_f64, 2.0, 3.0];
    let b = array![2.0_f64, 4.0, 6.0];
    let sim = cosine_similarity_simd::<f64>(&a.view(), &b.view());
    assert!(
        (sim - 1.0).abs() < 1e-10,
        "Cosine similarity for parallel vectors should be 1.0, got {}",
        sim
    );
}

/// Test cosine_similarity_simd orthogonal
#[test]
fn test_cosine_similarity_simd_orthogonal() {
    let a = array![1.0_f64, 0.0];
    let b = array![0.0_f64, 1.0];
    let sim = cosine_similarity_simd::<f64>(&a.view(), &b.view());
    assert!(
        (sim - 0.0).abs() < 1e-10,
        "Cosine similarity for orthogonal vectors should be 0.0, got {}",
        sim
    );
}

/// Test cosine_similarity_simd f32
#[test]
fn test_cosine_similarity_simd_f32() {
    let a = array![1.0_f32, 1.0];
    let b = array![1.0_f32, 0.0];
    let sim = cosine_similarity_simd::<f32>(&a.view(), &b.view());
    // cos(45°) = 1/sqrt(2) ≈ 0.707
    let expected = 1.0 / 2.0_f32.sqrt();
    assert!(
        (sim - expected).abs() < 1e-5,
        "Cosine similarity should be {}, got {}",
        expected,
        sim
    );
}

// =============================================================================
// Phase 73: Binary Operations Tests
// =============================================================================

/// Test add_simd basic
#[test]
fn test_add_simd_basic() {
    let a = array![1.0_f64, 2.0, 3.0];
    let b = array![4.0_f64, 5.0, 6.0];
    let c = add_simd::<f64>(&a.view(), &b.view());
    assert!((c[0] - 5.0).abs() < 1e-14);
    assert!((c[1] - 7.0).abs() < 1e-14);
    assert!((c[2] - 9.0).abs() < 1e-14);
}

/// Test add_simd f32
#[test]
fn test_add_simd_f32() {
    let a = array![1.5_f32, 2.5];
    let b = array![3.5_f32, 4.5];
    let c = add_simd::<f32>(&a.view(), &b.view());
    assert!((c[0] - 5.0).abs() < 1e-5);
    assert!((c[1] - 7.0).abs() < 1e-5);
}

/// Test sub_simd basic
#[test]
fn test_sub_simd_basic() {
    let a = array![5.0_f64, 7.0, 9.0];
    let b = array![1.0_f64, 2.0, 3.0];
    let c = sub_simd::<f64>(&a.view(), &b.view());
    assert!((c[0] - 4.0).abs() < 1e-14);
    assert!((c[1] - 5.0).abs() < 1e-14);
    assert!((c[2] - 6.0).abs() < 1e-14);
}

/// Test sub_simd f32
#[test]
fn test_sub_simd_f32() {
    let a = array![10.0_f32, 20.0];
    let b = array![3.0_f32, 5.0];
    let c = sub_simd::<f32>(&a.view(), &b.view());
    assert!((c[0] - 7.0).abs() < 1e-5);
    assert!((c[1] - 15.0).abs() < 1e-5);
}

/// Test mul_simd basic
#[test]
fn test_mul_simd_basic() {
    let a = array![2.0_f64, 3.0, 4.0];
    let b = array![5.0_f64, 6.0, 7.0];
    let c = mul_simd::<f64>(&a.view(), &b.view());
    assert!((c[0] - 10.0).abs() < 1e-14);
    assert!((c[1] - 18.0).abs() < 1e-14);
    assert!((c[2] - 28.0).abs() < 1e-14);
}

/// Test mul_simd f32
#[test]
fn test_mul_simd_f32() {
    let a = array![2.0_f32, 0.5];
    let b = array![4.0_f32, 8.0];
    let c = mul_simd::<f32>(&a.view(), &b.view());
    assert!((c[0] - 8.0).abs() < 1e-5);
    assert!((c[1] - 4.0).abs() < 1e-5);
}

/// Test div_simd basic
#[test]
fn test_div_simd_basic() {
    let a = array![10.0_f64, 20.0, 30.0];
    let b = array![2.0_f64, 4.0, 5.0];
    let c = div_simd::<f64>(&a.view(), &b.view());
    assert!((c[0] - 5.0).abs() < 1e-14);
    assert!((c[1] - 5.0).abs() < 1e-14);
    assert!((c[2] - 6.0).abs() < 1e-14);
}

/// Test div_simd f32
#[test]
fn test_div_simd_f32() {
    let a = array![9.0_f32, 16.0];
    let b = array![3.0_f32, 4.0];
    let c = div_simd::<f32>(&a.view(), &b.view());
    assert!((c[0] - 3.0).abs() < 1e-5);
    assert!((c[1] - 4.0).abs() < 1e-5);
}

/// Test max_simd basic
#[test]
fn test_max_simd_basic() {
    let a = array![1.0_f64, 5.0, 3.0];
    let b = array![4.0_f64, 2.0, 6.0];
    let c = max_simd::<f64>(&a.view(), &b.view());
    assert!((c[0] - 4.0).abs() < 1e-14);
    assert!((c[1] - 5.0).abs() < 1e-14);
    assert!((c[2] - 6.0).abs() < 1e-14);
}

/// Test max_simd f32
#[test]
fn test_max_simd_f32() {
    let a = array![-1.0_f32, 0.0, 1.0];
    let b = array![0.0_f32, 0.0, 0.0];
    let c = max_simd::<f32>(&a.view(), &b.view());
    assert!((c[0] - 0.0).abs() < 1e-5);
    assert!((c[1] - 0.0).abs() < 1e-5);
    assert!((c[2] - 1.0).abs() < 1e-5);
}

/// Test min_simd basic
#[test]
fn test_min_simd_basic() {
    let a = array![1.0_f64, 5.0, 3.0];
    let b = array![4.0_f64, 2.0, 6.0];
    let c = min_simd::<f64>(&a.view(), &b.view());
    assert!((c[0] - 1.0).abs() < 1e-14);
    assert!((c[1] - 2.0).abs() < 1e-14);
    assert!((c[2] - 3.0).abs() < 1e-14);
}

/// Test min_simd f32
#[test]
fn test_min_simd_f32() {
    let a = array![10.0_f32, 5.0];
    let b = array![7.0_f32, 8.0];
    let c = min_simd::<f32>(&a.view(), &b.view());
    assert!((c[0] - 7.0).abs() < 1e-5);
    assert!((c[1] - 5.0).abs() < 1e-5);
}

/// Test scalar_mul_simd basic
#[test]
fn test_scalar_mul_simd_basic() {
    let a = array![1.0_f64, 2.0, 3.0];
    let c = scalar_mul_simd::<f64>(&a.view(), 2.5);
    assert!((c[0] - 2.5).abs() < 1e-14);
    assert!((c[1] - 5.0).abs() < 1e-14);
    assert!((c[2] - 7.5).abs() < 1e-14);
}

/// Test scalar_mul_simd f32
#[test]
fn test_scalar_mul_simd_f32() {
    let a = array![2.0_f32, 4.0];
    let c = scalar_mul_simd::<f32>(&a.view(), 0.5);
    assert!((c[0] - 1.0).abs() < 1e-5);
    assert!((c[1] - 2.0).abs() < 1e-5);
}

/// Test fma_simd basic
#[test]
fn test_fma_simd_basic() {
    let a = array![1.0_f64, 2.0, 3.0];
    let b = array![4.0_f64, 5.0, 6.0];
    let c = array![7.0_f64, 8.0, 9.0];
    let result = fma_simd::<f64>(&a.view(), &b.view(), &c.view());
    // 1*4+7=11, 2*5+8=18, 3*6+9=27
    assert!((result[0] - 11.0).abs() < 1e-14);
    assert!((result[1] - 18.0).abs() < 1e-14);
    assert!((result[2] - 27.0).abs() < 1e-14);
}

/// Test fma_simd f32
#[test]
fn test_fma_simd_f32() {
    let a = array![2.0_f32, 3.0];
    let b = array![4.0_f32, 5.0];
    let c = array![1.0_f32, 2.0];
    let result = fma_simd::<f32>(&a.view(), &b.view(), &c.view());
    // 2*4+1=9, 3*5+2=17
    assert!((result[0] - 9.0).abs() < 1e-5);
    assert!((result[1] - 17.0).abs() < 1e-5);
}

/// Test dot_simd basic
#[test]
fn test_dot_simd_basic() {
    let a = array![1.0_f64, 2.0, 3.0];
    let b = array![4.0_f64, 5.0, 6.0];
    let d = dot_simd::<f64>(&a.view(), &b.view());
    // 1*4 + 2*5 + 3*6 = 4 + 10 + 18 = 32
    assert!(
        (d - 32.0).abs() < 1e-14,
        "Dot product should be 32.0, got {}",
        d
    );
}

/// Test dot_simd f32
#[test]
fn test_dot_simd_f32() {
    let a = array![1.0_f32, 0.0, 1.0];
    let b = array![2.0_f32, 3.0, 4.0];
    let d = dot_simd::<f32>(&a.view(), &b.view());
    // 1*2 + 0*3 + 1*4 = 6
    assert!(
        (d - 6.0).abs() < 1e-5,
        "Dot product should be 6.0, got {}",
        d
    );
}

/// Test dot_simd orthogonal
#[test]
fn test_dot_simd_orthogonal() {
    let a = array![1.0_f64, 0.0];
    let b = array![0.0_f64, 1.0];
    let d = dot_simd::<f64>(&a.view(), &b.view());
    assert!(
        (d - 0.0).abs() < 1e-14,
        "Dot product of orthogonal vectors should be 0.0"
    );
}

// =============================================================================
// Phase 74: Normalization and Activation Functions Tests
// =============================================================================

/// Test relu_simd basic
#[test]
fn test_relu_simd_basic() {
    let x = array![-2.0_f64, -1.0, 0.0, 1.0, 2.0];
    let result = relu_simd::<f64>(&x.view());
    assert!((result[0] - 0.0).abs() < 1e-14);
    assert!((result[1] - 0.0).abs() < 1e-14);
    assert!((result[2] - 0.0).abs() < 1e-14);
    assert!((result[3] - 1.0).abs() < 1e-14);
    assert!((result[4] - 2.0).abs() < 1e-14);
}

/// Test relu_simd f32
#[test]
fn test_relu_simd_f32() {
    let x = array![-5.0_f32, 0.0, 5.0];
    let result = relu_simd::<f32>(&x.view());
    assert!((result[0] - 0.0).abs() < 1e-5);
    assert!((result[1] - 0.0).abs() < 1e-5);
    assert!((result[2] - 5.0).abs() < 1e-5);
}

/// Test relu_simd all negative
#[test]
fn test_relu_simd_all_negative() {
    let x = array![-3.0_f64, -2.0, -1.0];
    let result = relu_simd::<f64>(&x.view());
    for val in result.iter() {
        assert!(*val == 0.0, "ReLU of negative should be 0");
    }
}

/// Test normalize_simd basic
#[test]
fn test_normalize_simd_basic() {
    let x = array![3.0_f64, 4.0];
    let result = normalize_simd::<f64>(&x.view());
    // ||result|| = 1
    let norm = (result[0] * result[0] + result[1] * result[1]).sqrt();
    assert!(
        (norm - 1.0).abs() < 1e-10,
        "Normalized vector should have unit length"
    );
    // x/5 = [0.6, 0.8]
    assert!((result[0] - 0.6).abs() < 1e-10);
    assert!((result[1] - 0.8).abs() < 1e-10);
}

/// Test normalize_simd f32
#[test]
fn test_normalize_simd_f32() {
    let x = array![1.0_f32, 1.0, 1.0, 1.0];
    let result = normalize_simd::<f32>(&x.view());
    // ||result|| = 1
    let norm: f32 = result.iter().map(|&v| v * v).sum::<f32>().sqrt();
    assert!(
        (norm - 1.0).abs() < 1e-5,
        "Normalized vector should have unit length"
    );
}

/// Test normalize_simd preserves direction
#[test]
fn test_normalize_simd_direction() {
    let x = array![2.0_f64, 4.0, 6.0];
    let result = normalize_simd::<f64>(&x.view());
    // All elements should have same ratio
    let ratio1 = result[1] / result[0];
    let ratio2 = result[2] / result[0];
    assert!(
        (ratio1 - 2.0).abs() < 1e-10,
        "Direction should be preserved"
    );
    assert!(
        (ratio2 - 3.0).abs() < 1e-10,
        "Direction should be preserved"
    );
}

/// Test standardize_simd basic
#[test]
fn test_standardize_simd_basic() {
    let x = array![2.0_f64, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0];
    let result = standardize_simd::<f64>(&x.view());
    // Mean should be ~0
    let mean: f64 = result.iter().sum::<f64>() / result.len() as f64;
    assert!(
        mean.abs() < 1e-10,
        "Standardized mean should be 0, got {}",
        mean
    );
}

/// Test standardize_simd f32
#[test]
fn test_standardize_simd_f32() {
    let x = array![1.0_f32, 2.0, 3.0, 4.0, 5.0];
    let result = standardize_simd::<f32>(&x.view());
    let mean: f32 = result.iter().sum::<f32>() / result.len() as f32;
    assert!(
        mean.abs() < 1e-4,
        "Standardized mean should be 0, got {}",
        mean
    );
}

/// Test softmax_simd sums to one
#[test]
fn test_softmax_simd_sums_to_one() {
    let x = array![1.0_f64, 2.0, 3.0];
    let result = softmax_simd::<f64>(&x.view());
    let sum: f64 = result.iter().sum();
    assert!(
        (sum - 1.0).abs() < 1e-10,
        "Softmax should sum to 1.0, got {}",
        sum
    );
}

/// Test softmax_simd ordering preserved
#[test]
fn test_softmax_simd_ordering() {
    let x = array![1.0_f64, 2.0, 3.0];
    let result = softmax_simd::<f64>(&x.view());
    assert!(
        result[2] > result[1],
        "Higher logit should have higher probability"
    );
    assert!(
        result[1] > result[0],
        "Higher logit should have higher probability"
    );
}

/// Test softmax_simd f32
#[test]
fn test_softmax_simd_f32() {
    let x = array![0.0_f32, 0.0, 0.0];
    let result = softmax_simd::<f32>(&x.view());
    // Equal inputs should give equal probabilities
    let expected = 1.0_f32 / 3.0;
    for val in result.iter() {
        assert!(
            (val - expected).abs() < 1e-5,
            "Equal inputs should give equal probabilities"
        );
    }
}

/// Test softmax_simd numerical stability
#[test]
fn test_softmax_simd_stability() {
    let x = array![1000.0_f64, 1001.0, 1002.0];
    let result = softmax_simd::<f64>(&x.view());
    let sum: f64 = result.iter().sum();
    assert!(
        (sum - 1.0).abs() < 1e-10,
        "Softmax should handle large values stably"
    );
    assert!(
        result.iter().all(|&v| v.is_finite()),
        "All softmax values should be finite"
    );
}

/// Test trunc_simd basic
#[test]
fn test_trunc_simd_basic() {
    let x = array![2.7_f64, -2.7, 0.9, -0.9];
    let result = trunc_simd::<f64>(&x.view());
    assert!((result[0] - 2.0).abs() < 1e-14);
    assert!((result[1] - (-2.0)).abs() < 1e-14);
    assert!((result[2] - 0.0).abs() < 1e-14);
    assert!((result[3] - 0.0).abs() < 1e-14);
}

/// Test trunc_simd f32
#[test]
fn test_trunc_simd_f32() {
    let x = array![3.9_f32, -3.9];
    let result = trunc_simd::<f32>(&x.view());
    assert!((result[0] - 3.0).abs() < 1e-5);
    assert!((result[1] - (-3.0)).abs() < 1e-5);
}

/// Test trunc_simd integers unchanged
#[test]
fn test_trunc_simd_integers() {
    let x = array![1.0_f64, 2.0, -3.0, 0.0];
    let result = trunc_simd::<f64>(&x.view());
    for i in 0..x.len() {
        assert!(
            (result[i] - x[i]).abs() < 1e-14,
            "Integer values should be unchanged"
        );
    }
}

// =============================================================================
// Integration and Edge Case Tests
// =============================================================================

/// Test large array operations
#[test]
fn test_large_array_operations() {
    let n = 10000;
    let x: Array1<f64> = Array1::from_vec((0..n).map(|i| i as f64).collect());
    let y: Array1<f64> = Array1::from_vec((0..n).map(|i| (i * 2) as f64).collect());

    // Test add
    let sum = add_simd::<f64>(&x.view(), &y.view());
    assert!((sum[0] - 0.0).abs() < 1e-10);
    assert!((sum[n - 1] - (3.0 * (n - 1) as f64)).abs() < 1e-10);

    // Test dot
    let d = dot_simd::<f64>(&x.view(), &y.view());
    // sum(i * 2i) = 2 * sum(i^2) = 2 * n*(n-1)*(2n-1)/6
    let expected = 2.0 * (n as f64) * ((n - 1) as f64) * ((2 * n - 1) as f64) / 6.0;
    assert!((d - expected).abs() / expected < 1e-10, "Large dot product");
}

/// Test empty array handling
#[test]
fn test_empty_array_handling() {
    let empty: Array1<f64> = Array1::zeros(0);
    let non_empty = array![1.0_f64, 2.0, 3.0];

    assert!(sum_simd::<f64>(&empty.view()).abs() < 1e-14);
    assert!(mean_simd::<f64>(&empty.view()).abs() < 1e-14);
    assert_eq!(add_simd::<f64>(&empty.view(), &non_empty.view()).len(), 0);
    assert_eq!(relu_simd::<f64>(&empty.view()).len(), 0);
    assert_eq!(softmax_simd::<f64>(&empty.view()).len(), 0);
}

/// Test numerical properties of distance metrics
#[test]
fn test_distance_metric_properties() {
    let a = array![1.0_f64, 2.0, 3.0];
    let b = array![4.0_f64, 5.0, 6.0];
    let c = array![7.0_f64, 8.0, 9.0];

    // Identity: d(a, a) = 0
    assert!(distance_euclidean_simd::<f64>(&a.view(), &a.view()).abs() < 1e-14);
    assert!(distance_manhattan_simd::<f64>(&a.view(), &a.view()).abs() < 1e-14);
    assert!(distance_chebyshev_simd::<f64>(&a.view(), &a.view()).abs() < 1e-14);

    // Symmetry: d(a, b) = d(b, a)
    let d_ab = distance_euclidean_simd::<f64>(&a.view(), &b.view());
    let d_ba = distance_euclidean_simd::<f64>(&b.view(), &a.view());
    assert!((d_ab - d_ba).abs() < 1e-14);

    // Triangle inequality: d(a, c) <= d(a, b) + d(b, c)
    let d_ac = distance_euclidean_simd::<f64>(&a.view(), &c.view());
    let d_bc = distance_euclidean_simd::<f64>(&b.view(), &c.view());
    assert!(d_ac <= d_ab + d_bc + 1e-10);
}

/// Test statistical consistency
#[test]
fn test_statistical_consistency() {
    let x = array![2.0_f64, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0];

    let mean = mean_simd::<f64>(&x.view());
    let var = variance_simd::<f64>(&x.view());
    let std = std_simd::<f64>(&x.view());

    // std = sqrt(var)
    assert!((std - var.sqrt()).abs() < 1e-10);

    // Sample variance = sum((x - mean)^2) / (n-1) (Bessel's correction)
    let n = x.len() as f64;
    let manual_var: f64 = x.iter().map(|&xi| (xi - mean).powi(2)).sum::<f64>() / (n - 1.0);
    assert!((var - manual_var).abs() < 1e-10);
}
