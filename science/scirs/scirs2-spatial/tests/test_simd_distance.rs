//! SIMD distance metrics integration tests
//!
//! This module tests the SIMD-accelerated distance implementations
//! in scirs2-spatial, ensuring correctness and performance.

use scirs2_spatial::distance::{chebyshev, euclidean, manhattan};

#[test]
fn test_simd_euclidean_f64_basic() {
    let point1 = &[1.0f64, 2.0, 3.0];
    let point2 = &[4.0f64, 5.0, 6.0];

    let dist = euclidean(point1, point2);

    // Expected: sqrt((4-1)^2 + (5-2)^2 + (6-3)^2) = sqrt(9+9+9) = sqrt(27) ≈ 5.196
    let expected = 27.0f64.sqrt();
    assert!(
        (dist - expected).abs() < 1e-10,
        "Got {}, expected {}",
        dist,
        expected
    );
}

#[test]
fn test_simd_euclidean_f32_basic() {
    let point1 = &[1.0f32, 2.0, 3.0];
    let point2 = &[4.0f32, 5.0, 6.0];

    let dist = euclidean(point1, point2);

    let expected = 27.0f32.sqrt();
    assert!(
        (dist - expected).abs() < 1e-6,
        "Got {}, expected {}",
        dist,
        expected
    );
}

#[test]
fn test_simd_euclidean_f64_large() {
    // Test with large array to ensure SIMD path is used
    let point1: Vec<f64> = (0..1000).map(|i| i as f64).collect();
    let point2: Vec<f64> = (0..1000).map(|i| (i + 1) as f64).collect();

    let dist = euclidean(&point1, &point2);

    // Each dimension differs by 1, so euclidean distance = sqrt(1000 * 1^2) = sqrt(1000)
    let expected = 1000.0f64.sqrt();
    assert!(
        (dist - expected).abs() < 1e-10,
        "Got {}, expected {}",
        dist,
        expected
    );
}

#[test]
fn test_simd_euclidean_f32_large() {
    let point1: Vec<f32> = (0..1000).map(|i| i as f32).collect();
    let point2: Vec<f32> = (0..1000).map(|i| (i + 1) as f32).collect();

    let dist = euclidean(&point1, &point2);

    let expected = 1000.0f32.sqrt();
    assert!(
        (dist - expected).abs() < 1e-4,
        "Got {}, expected {}",
        dist,
        expected
    );
}

#[test]
fn test_simd_manhattan_f64_basic() {
    let point1 = &[1.0f64, 2.0, 3.0];
    let point2 = &[4.0f64, 5.0, 6.0];

    let dist = manhattan(point1, point2);

    // Expected: |4-1| + |5-2| + |6-3| = 3 + 3 + 3 = 9
    let expected = 9.0f64;
    assert!(
        (dist - expected).abs() < 1e-10,
        "Got {}, expected {}",
        dist,
        expected
    );
}

#[test]
fn test_simd_manhattan_f32_basic() {
    let point1 = &[1.0f32, 2.0, 3.0];
    let point2 = &[4.0f32, 5.0, 6.0];

    let dist = manhattan(point1, point2);

    let expected = 9.0f32;
    assert!(
        (dist - expected).abs() < 1e-6,
        "Got {}, expected {}",
        dist,
        expected
    );
}

#[test]
fn test_simd_manhattan_f64_large() {
    // Test with large array
    let point1: Vec<f64> = (0..1000).map(|i| i as f64).collect();
    let point2: Vec<f64> = (0..1000).map(|i| (i + 1) as f64).collect();

    let dist = manhattan(&point1, &point2);

    // Each dimension differs by 1, so manhattan distance = 1000 * 1 = 1000
    let expected = 1000.0f64;
    assert!(
        (dist - expected).abs() < 1e-10,
        "Got {}, expected {}",
        dist,
        expected
    );
}

#[test]
fn test_simd_manhattan_f32_large() {
    let point1: Vec<f32> = (0..1000).map(|i| i as f32).collect();
    let point2: Vec<f32> = (0..1000).map(|i| (i + 1) as f32).collect();

    let dist = manhattan(&point1, &point2);

    let expected = 1000.0f32;
    assert!(
        (dist - expected).abs() < 1e-4,
        "Got {}, expected {}",
        dist,
        expected
    );
}

#[test]
fn test_simd_chebyshev_f64_basic() {
    let point1 = &[1.0f64, 2.0, 3.0];
    let point2 = &[4.0f64, 5.0, 6.0];

    let dist = chebyshev(point1, point2);

    // Expected: max(|4-1|, |5-2|, |6-3|) = max(3, 3, 3) = 3
    let expected = 3.0f64;
    assert!(
        (dist - expected).abs() < 1e-10,
        "Got {}, expected {}",
        dist,
        expected
    );
}

#[test]
fn test_simd_chebyshev_f32_basic() {
    let point1 = &[1.0f32, 2.0, 3.0];
    let point2 = &[4.0f32, 5.0, 6.0];

    let dist = chebyshev(point1, point2);

    let expected = 3.0f32;
    assert!(
        (dist - expected).abs() < 1e-6,
        "Got {}, expected {}",
        dist,
        expected
    );
}

#[test]
fn test_simd_chebyshev_f64_large() {
    // Test with large array where one dimension has max difference
    let point1: Vec<f64> = (0..1000)
        .map(|i| if i == 500 { 0.0 } else { i as f64 })
        .collect();
    let point2: Vec<f64> = (0..1000)
        .map(|i| if i == 500 { 100.0 } else { i as f64 })
        .collect();

    let dist = chebyshev(&point1, &point2);

    // Max difference is at index 500: |100 - 0| = 100
    let expected = 100.0f64;
    assert!(
        (dist - expected).abs() < 1e-10,
        "Got {}, expected {}",
        dist,
        expected
    );
}

#[test]
fn test_simd_chebyshev_f32_large() {
    let point1: Vec<f32> = (0..1000)
        .map(|i| if i == 500 { 0.0 } else { i as f32 })
        .collect();
    let point2: Vec<f32> = (0..1000)
        .map(|i| if i == 500 { 100.0 } else { i as f32 })
        .collect();

    let dist = chebyshev(&point1, &point2);

    let expected = 100.0f32;
    assert!(
        (dist - expected).abs() < 1e-4,
        "Got {}, expected {}",
        dist,
        expected
    );
}

#[test]
fn test_simd_euclidean_negative_values() {
    let point1 = &[-1.0f64, -2.0, -3.0];
    let point2 = &[1.0f64, 2.0, 3.0];

    let dist = euclidean(point1, point2);

    // Expected: sqrt((1-(-1))^2 + (2-(-2))^2 + (3-(-3))^2) = sqrt(4+16+36) = sqrt(56)
    let expected = 56.0f64.sqrt();
    assert!((dist - expected).abs() < 1e-10);
}

#[test]
fn test_simd_manhattan_negative_values() {
    let point1 = &[-1.0f64, -2.0, -3.0];
    let point2 = &[1.0f64, 2.0, 3.0];

    let dist = manhattan(point1, point2);

    // Expected: |1-(-1)| + |2-(-2)| + |3-(-3)| = 2 + 4 + 6 = 12
    let expected = 12.0f64;
    assert!((dist - expected).abs() < 1e-10);
}

#[test]
fn test_simd_euclidean_zero_distance() {
    let point1 = &[1.0f64, 2.0, 3.0];
    let point2 = &[1.0f64, 2.0, 3.0];

    let dist = euclidean(point1, point2);

    assert!(
        dist.abs() < 1e-10,
        "Distance between identical points should be 0"
    );
}

#[test]
fn test_simd_manhattan_zero_distance() {
    let point1 = &[1.0f64, 2.0, 3.0];
    let point2 = &[1.0f64, 2.0, 3.0];

    let dist = manhattan(point1, point2);

    assert!(
        dist.abs() < 1e-10,
        "Distance between identical points should be 0"
    );
}

#[test]
fn test_simd_chebyshev_zero_distance() {
    let point1 = &[1.0f64, 2.0, 3.0];
    let point2 = &[1.0f64, 2.0, 3.0];

    let dist = chebyshev(point1, point2);

    assert!(
        dist.abs() < 1e-10,
        "Distance between identical points should be 0"
    );
}

#[test]
fn test_simd_euclidean_single_dimension() {
    let point1 = &[5.0f64];
    let point2 = &[8.0f64];

    let dist = euclidean(point1, point2);

    let expected = 3.0f64;
    assert!((dist - expected).abs() < 1e-10);
}

#[test]
fn test_simd_euclidean_two_dimensions() {
    let point1 = &[3.0f64, 4.0];
    let point2 = &[0.0f64, 0.0];

    let dist = euclidean(point1, point2);

    // Expected: sqrt(3^2 + 4^2) = sqrt(9 + 16) = sqrt(25) = 5
    let expected = 5.0f64;
    assert!((dist - expected).abs() < 1e-10);
}
