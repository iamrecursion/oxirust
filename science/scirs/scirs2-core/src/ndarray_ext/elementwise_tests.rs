use super::*;
use ::ndarray::array;

#[test]
fn test_abs_simd_f64_basic() {
    let x = array![-3.0, -1.5, 0.0, 1.5, 3.0];
    let result = abs_simd(&x.view());

    assert_eq!(result[0], 3.0);
    assert_eq!(result[1], 1.5);
    assert_eq!(result[2], 0.0);
    assert_eq!(result[3], 1.5);
    assert_eq!(result[4], 3.0);
}

#[test]
fn test_abs_simd_f32_basic() {
    let x = array![-3.0f32, -1.5, 0.0, 1.5, 3.0];
    let result = abs_simd(&x.view());

    assert_eq!(result[0], 3.0);
    assert_eq!(result[1], 1.5);
    assert_eq!(result[2], 0.0);
    assert_eq!(result[3], 1.5);
    assert_eq!(result[4], 3.0);
}

#[test]
fn test_sqrt_simd_f64_basic() {
    let x = array![1.0, 4.0, 9.0, 16.0, 25.0];
    let result = sqrt_simd(&x.view());

    assert_eq!(result[0], 1.0);
    assert_eq!(result[1], 2.0);
    assert_eq!(result[2], 3.0);
    assert_eq!(result[3], 4.0);
    assert_eq!(result[4], 5.0);
}

#[test]
fn test_sqrt_simd_f32_basic() {
    let x = array![1.0f32, 4.0, 9.0, 16.0, 25.0];
    let result = sqrt_simd(&x.view());

    assert_eq!(result[0], 1.0);
    assert_eq!(result[1], 2.0);
    assert_eq!(result[2], 3.0);
    assert_eq!(result[3], 4.0);
    assert_eq!(result[4], 5.0);
}
