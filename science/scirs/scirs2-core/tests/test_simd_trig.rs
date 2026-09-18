//! Trigonometric SIMD tests: sin, cos, tan, atan, asin, acos, atan2, to_radians, to_degrees

use scirs2_core::ndarray::{array, Array1};
use scirs2_core::ndarray_ext::elementwise::{
    acos_simd, asin_simd, atan2_simd, atan_simd, cos_simd, sin_simd, tan_simd,
};
#[cfg(feature = "random")]
use scirs2_core::random::{thread_rng, Distribution, Uniform};

// ============================================================================
// Sin SIMD Tests
// ============================================================================

#[test]
fn test_sin_simd_f64_basic() {
    use std::f64::consts::PI;
    let x = array![0.0, PI / 6.0, PI / 4.0, PI / 2.0, PI];
    let result = sin_simd(&x.view());

    assert!((result[0] - 0.0).abs() < 1e-10);
    assert!((result[1] - 0.5).abs() < 1e-10);
    assert!((result[2] - (2.0_f64.sqrt() / 2.0)).abs() < 1e-10);
    assert!((result[3] - 1.0).abs() < 1e-10);
    assert!((result[4] - 0.0).abs() < 1e-10);
}

#[test]
fn test_sin_simd_f32_basic() {
    use std::f32::consts::PI;
    let x = array![0.0f32, PI / 6.0, PI / 4.0, PI / 2.0, PI];
    let result = sin_simd(&x.view());

    assert!((result[0] - 0.0).abs() < 1e-6);
    assert!((result[1] - 0.5).abs() < 1e-6);
    assert!((result[2] - (2.0_f32.sqrt() / 2.0)).abs() < 1e-6);
    assert!((result[3] - 1.0).abs() < 1e-6);
    assert!((result[4] - 0.0).abs() < 1e-6);
}

#[test]
fn test_sin_simd_f64_empty() {
    let x: Array1<f64> = array![];
    let result = sin_simd(&x.view());
    assert_eq!(result.len(), 0);
}

#[test]
fn test_sin_simd_f64_zero() {
    let x = array![0.0];
    let result = sin_simd(&x.view());
    assert_eq!(result[0], 0.0);
}

#[test]
fn test_sin_simd_f64_negative() {
    use std::f64::consts::PI;
    let x = array![-PI / 2.0, -PI / 6.0];
    let result = sin_simd(&x.view());

    assert!((result[0] - (-1.0)).abs() < 1e-10);
    assert!((result[1] - (-0.5)).abs() < 1e-10);
}

#[test]
fn test_sin_simd_f64_nan() {
    let x = array![f64::NAN];
    let result = sin_simd(&x.view());
    assert!(result[0].is_nan());
}

#[test]
fn test_sin_simd_f64_infinity() {
    let x = array![f64::INFINITY, f64::NEG_INFINITY];
    let result = sin_simd(&x.view());
    assert!(result[0].is_nan());
    assert!(result[1].is_nan());
}

#[test]
fn test_sin_simd_f64_large_array() {
    use std::f64::consts::PI;
    let x: Array1<f64> = Array1::linspace(0.0, 2.0 * PI, 10000);
    let result = sin_simd(&x.view());

    // Check periodicity
    assert!((result[0] - 0.0).abs() < 1e-10);
    assert!((result[result.len() - 1] - 0.0).abs() < 1e-9);
}

#[test]
fn test_sin_simd_f32_large_array() {
    use std::f32::consts::PI;
    let x: Array1<f32> = Array1::linspace(0.0, 2.0 * PI, 10000);
    let result = sin_simd(&x.view());

    // Check range: sin(x) ∈ [-1, 1]
    for &val in result.iter() {
        assert!((-1.0..=1.0).contains(&val));
    }
}

#[cfg(feature = "random")]
#[test]
fn test_sin_simd_f64_range() {
    // sin(x) should always be in [-1, 1]
    let mut rng = thread_rng();
    let uniform = Uniform::new(-100.0, 100.0).expect("Test: operation failed");
    let x: Array1<f64> = Array1::from_vec((0..1000).map(|_| uniform.sample(&mut rng)).collect());
    let result = sin_simd(&x.view());

    for &val in result.iter() {
        assert!(
            (-1.0..=1.0).contains(&val),
            "sin value {} out of range",
            val
        );
    }
}

// ============================================================================
// ============================================================================
// Cos SIMD Tests
// ============================================================================

#[test]
fn test_cos_simd_f64_basic() {
    use std::f64::consts::PI;
    let x = array![0.0, PI / 3.0, PI / 2.0, PI];
    let result = cos_simd(&x.view());

    assert!((result[0] - 1.0).abs() < 1e-10);
    assert!((result[1] - 0.5).abs() < 1e-10);
    assert!((result[2] - 0.0).abs() < 1e-10);
    assert!((result[3] - (-1.0)).abs() < 1e-10);
}

#[test]
fn test_cos_simd_f32_basic() {
    use std::f32::consts::PI;
    let x = array![0.0f32, PI / 3.0, PI / 2.0, PI];
    let result = cos_simd(&x.view());

    assert!((result[0] - 1.0).abs() < 1e-6);
    assert!((result[1] - 0.5).abs() < 1e-6);
    assert!((result[2] - 0.0).abs() < 1e-6);
    assert!((result[3] - (-1.0)).abs() < 1e-6);
}

#[test]
fn test_cos_simd_f64_empty() {
    let x: Array1<f64> = array![];
    let result = cos_simd(&x.view());
    assert_eq!(result.len(), 0);
}

#[test]
fn test_cos_simd_f64_zero() {
    let x = array![0.0];
    let result = cos_simd(&x.view());
    assert_eq!(result[0], 1.0);
}

#[test]
fn test_cos_simd_f64_negative() {
    use std::f64::consts::PI;
    let x = array![-PI, -PI / 3.0];
    let result = cos_simd(&x.view());

    assert!((result[0] - (-1.0)).abs() < 1e-10);
    assert!((result[1] - 0.5).abs() < 1e-10);
}

#[test]
fn test_cos_simd_f64_nan() {
    let x = array![f64::NAN];
    let result = cos_simd(&x.view());
    assert!(result[0].is_nan());
}

#[test]
fn test_cos_simd_f64_infinity() {
    let x = array![f64::INFINITY, f64::NEG_INFINITY];
    let result = cos_simd(&x.view());
    assert!(result[0].is_nan());
    assert!(result[1].is_nan());
}

#[test]
fn test_cos_simd_f64_large_array() {
    use std::f64::consts::PI;
    let x: Array1<f64> = Array1::linspace(0.0, 2.0 * PI, 10000);
    let result = cos_simd(&x.view());

    // Check periodicity
    assert!((result[0] - 1.0).abs() < 1e-10);
    assert!((result[result.len() - 1] - 1.0).abs() < 1e-9);
}

#[test]
fn test_cos_simd_f32_large_array() {
    use std::f32::consts::PI;
    let x: Array1<f32> = Array1::linspace(0.0, 2.0 * PI, 10000);
    let result = cos_simd(&x.view());

    // Check range: cos(x) ∈ [-1, 1]
    for &val in result.iter() {
        assert!((-1.0..=1.0).contains(&val));
    }
}

#[cfg(feature = "random")]
#[test]
fn test_cos_simd_f64_range() {
    // cos(x) should always be in [-1, 1]
    let mut rng = thread_rng();
    let uniform = Uniform::new(-100.0, 100.0).expect("Test: operation failed");
    let x: Array1<f64> = Array1::from_vec((0..1000).map(|_| uniform.sample(&mut rng)).collect());
    let result = cos_simd(&x.view());

    for &val in result.iter() {
        assert!(
            (-1.0..=1.0).contains(&val),
            "cos value {} out of range",
            val
        );
    }
}

// ============================================================================
// ============================================================================
// Tan SIMD Tests
// ============================================================================

#[test]
fn test_tan_simd_f64_basic() {
    use std::f64::consts::PI;
    let x = array![0.0, PI / 4.0, PI / 6.0];
    let result = tan_simd(&x.view());

    assert!((result[0] - 0.0).abs() < 1e-10);
    assert!((result[1] - 1.0).abs() < 1e-10);
    assert!((result[2] - (1.0 / 3.0_f64.sqrt())).abs() < 1e-10);
}

#[test]
fn test_tan_simd_f32_basic() {
    use std::f32::consts::PI;
    let x = array![0.0f32, PI / 4.0, PI / 6.0];
    let result = tan_simd(&x.view());

    assert!((result[0] - 0.0).abs() < 1e-6);
    assert!((result[1] - 1.0).abs() < 1e-6);
    assert!((result[2] - (1.0 / 3.0_f32.sqrt())).abs() < 1e-6);
}

#[test]
fn test_tan_simd_f64_empty() {
    let x: Array1<f64> = array![];
    let result = tan_simd(&x.view());
    assert_eq!(result.len(), 0);
}

#[test]
fn test_tan_simd_f64_zero() {
    let x = array![0.0];
    let result = tan_simd(&x.view());
    assert_eq!(result[0], 0.0);
}

#[test]
fn test_tan_simd_f64_negative() {
    use std::f64::consts::PI;
    let x = array![-PI / 4.0, -PI / 6.0];
    let result = tan_simd(&x.view());

    assert!((result[0] - (-1.0)).abs() < 1e-10);
    assert!((result[1] - (-1.0 / 3.0_f64.sqrt())).abs() < 1e-10);
}

#[test]
fn test_tan_simd_f64_nan() {
    let x = array![f64::NAN];
    let result = tan_simd(&x.view());
    assert!(result[0].is_nan());
}

#[test]
fn test_tan_simd_f64_infinity() {
    let x = array![f64::INFINITY, f64::NEG_INFINITY];
    let result = tan_simd(&x.view());
    assert!(result[0].is_nan());
    assert!(result[1].is_nan());
}

#[test]
fn test_tan_simd_f64_singularity() {
    use std::f64::consts::PI;
    // tan has singularities at π/2, 3π/2, etc.
    let x = array![PI / 2.0, 3.0 * PI / 2.0];
    let result = tan_simd(&x.view());

    // Should return very large values or infinity
    assert!(result[0].abs() > 1e10 || result[0].is_infinite());
    assert!(result[1].abs() > 1e10 || result[1].is_infinite());
}

#[test]
fn test_tan_simd_f64_large_array() {
    use std::f64::consts::PI;
    let x: Array1<f64> = Array1::linspace(-PI / 3.0, PI / 3.0, 10000);
    let result = tan_simd(&x.view());

    // Check monotonicity in (-π/2, π/2)
    for i in 1..result.len() {
        assert!(
            result[i] >= result[i - 1],
            "tan not monotonic at index {}",
            i
        );
    }
}

#[test]
fn test_tan_simd_f32_large_array() {
    use std::f32::consts::PI;
    let x: Array1<f32> = Array1::linspace(-PI / 3.0, PI / 3.0, 10000);
    let result = tan_simd(&x.view());

    // All values should be finite in this range
    for &val in result.iter() {
        assert!(val.is_finite(), "tan value {} not finite", val);
    }
}

// ============================================================================
// Trigonometric Identity Tests
// ============================================================================

#[cfg(feature = "random")]
#[test]
fn test_sin_cos_pythagorean_identity() {
    // sin²(x) + cos²(x) = 1
    let mut rng = thread_rng();
    let uniform = Uniform::new(-10.0, 10.0).expect("Test: operation failed");
    let x: Array1<f64> = Array1::from_vec((0..1000).map(|_| uniform.sample(&mut rng)).collect());

    let sin_x = sin_simd(&x.view());
    let cos_x = cos_simd(&x.view());

    for i in 0..x.len() {
        let sum = sin_x[i].powi(2) + cos_x[i].powi(2);
        assert!(
            (sum - 1.0).abs() < 1e-10,
            "Pythagorean identity failed at index {}: sin²={}, cos²={}, sum={}",
            i,
            sin_x[i].powi(2),
            cos_x[i].powi(2),
            sum
        );
    }
}

#[test]
fn test_tan_sin_cos_identity() {
    // tan(x) = sin(x) / cos(x)
    use std::f64::consts::PI;
    let x: Array1<f64> = Array1::linspace(-PI / 3.0, PI / 3.0, 100);

    let sin_x = sin_simd(&x.view());
    let cos_x = cos_simd(&x.view());
    let tan_x = tan_simd(&x.view());

    for i in 0..x.len() {
        if cos_x[i].abs() > 1e-10 {
            let expected = sin_x[i] / cos_x[i];
            assert!(
                (tan_x[i] - expected).abs() < 1e-10,
                "tan(x) != sin(x)/cos(x) at index {}: tan={}, sin/cos={}",
                i,
                tan_x[i],
                expected
            );
        }
    }
}

#[cfg(feature = "random")]
#[test]
fn test_sin_negative_angle() {
    // sin(-x) = -sin(x)
    let mut rng = thread_rng();
    let uniform = Uniform::new(-10.0, 10.0).expect("Test: operation failed");
    let x: Array1<f64> = Array1::from_vec((0..100).map(|_| uniform.sample(&mut rng)).collect());

    let sin_x = sin_simd(&x.view());
    let neg_x = x.mapv(|v| -v);
    let sin_neg_x = sin_simd(&neg_x.view());

    for i in 0..x.len() {
        assert!(
            (sin_neg_x[i] - (-sin_x[i])).abs() < 1e-10,
            "sin(-x) != -sin(x) at index {}",
            i
        );
    }
}

#[cfg(feature = "random")]
#[test]
fn test_cos_negative_angle() {
    // cos(-x) = cos(x)
    let mut rng = thread_rng();
    let uniform = Uniform::new(-10.0, 10.0).expect("Test: operation failed");
    let x: Array1<f64> = Array1::from_vec((0..100).map(|_| uniform.sample(&mut rng)).collect());

    let cos_x = cos_simd(&x.view());
    let neg_x = x.mapv(|v| -v);
    let cos_neg_x = cos_simd(&neg_x.view());

    for i in 0..x.len() {
        assert!(
            (cos_neg_x[i] - cos_x[i]).abs() < 1e-10,
            "cos(-x) != cos(x) at index {}",
            i
        );
    }
}

#[cfg(feature = "random")]
#[test]
fn test_tan_negative_angle() {
    // tan(-x) = -tan(x)
    let mut rng = thread_rng();
    let uniform = Uniform::new(-1.0, 1.0).expect("Test: operation failed"); // Avoid singularities
    let x: Array1<f64> = Array1::from_vec((0..100).map(|_| uniform.sample(&mut rng)).collect());

    let tan_x = tan_simd(&x.view());
    let neg_x = x.mapv(|v| -v);
    let tan_neg_x = tan_simd(&neg_x.view());

    for i in 0..x.len() {
        assert!(
            (tan_neg_x[i] - (-tan_x[i])).abs() < 1e-10,
            "tan(-x) != -tan(x) at index {}",
            i
        );
    }
}

// ============================================================================
// ============================================================================
// Atan SIMD Tests
// ============================================================================

#[test]
fn test_atan_simd_f64_basic() {
    let x = array![0.0_f64, 1.0, -1.0, f64::INFINITY, f64::NEG_INFINITY];
    let result = atan_simd(&x.view());

    assert!((result[0] - 0.0).abs() < 1e-10);
    assert!((result[1] - std::f64::consts::FRAC_PI_4).abs() < 1e-10);
    assert!((result[2] + std::f64::consts::FRAC_PI_4).abs() < 1e-10);
    assert!((result[3] - std::f64::consts::FRAC_PI_2).abs() < 1e-10);
    assert!((result[4] + std::f64::consts::FRAC_PI_2).abs() < 1e-10);
}

#[test]
fn test_atan_simd_f32_basic() {
    let x = array![0.0_f32, 1.0, -1.0];
    let result = atan_simd(&x.view());

    assert!((result[0] - 0.0).abs() < 1e-6);
    assert!((result[1] - std::f32::consts::FRAC_PI_4).abs() < 1e-6);
    assert!((result[2] + std::f32::consts::FRAC_PI_4).abs() < 1e-6);
}

#[test]
fn test_atan_simd_empty() {
    let x: Array1<f64> = array![];
    let result = atan_simd(&x.view());
    assert_eq!(result.len(), 0);
}

// Anti-symmetry: atan(-x) = -atan(x)
#[test]
fn test_atan_anti_symmetry() {
    let x = array![0.5_f64, 1.0, 2.0, 3.0];
    let neg_x = x.mapv(|v| -v);

    let atan_x = atan_simd(&x.view());
    let atan_neg_x = atan_simd(&neg_x.view());

    for i in 0..x.len() {
        assert!(
            (atan_neg_x[i] + atan_x[i]).abs() < 1e-10,
            "atan(-x) should equal -atan(x)"
        );
    }
}

#[cfg(feature = "random")]
#[test]
fn test_atan_simd_large_array() {
    let mut rng = thread_rng();
    let uniform = Uniform::new(-10.0, 10.0).expect("Test: operation failed");
    let data: Vec<f64> = (0..10000).map(|_| uniform.sample(&mut rng)).collect();
    let x = Array1::from(data.clone());

    let result = atan_simd(&x.view());

    for i in 0..100.min(data.len()) {
        assert_eq!(result[i], data[i].atan());
    }
}

// ============================================================================
// ============================================================================
// Asin SIMD Tests
// ============================================================================

#[test]
fn test_asin_simd_f64_basic() {
    let x = array![0.0_f64, 0.5, 1.0, -1.0];
    let result = asin_simd(&x.view());

    assert!((result[0] - 0.0).abs() < 1e-10);
    assert!((result[1] - std::f64::consts::FRAC_PI_6).abs() < 1e-10);
    assert!((result[2] - std::f64::consts::FRAC_PI_2).abs() < 1e-10);
    assert!((result[3] + std::f64::consts::FRAC_PI_2).abs() < 1e-10);
}

#[test]
fn test_asin_simd_f32_basic() {
    let x = array![0.0_f32, 0.5, 1.0, -1.0];
    let result = asin_simd(&x.view());

    assert!((result[0] - 0.0).abs() < 1e-6);
    assert!((result[1] - std::f32::consts::FRAC_PI_6).abs() < 1e-6);
    assert!((result[2] - std::f32::consts::FRAC_PI_2).abs() < 1e-6);
    assert!((result[3] + std::f32::consts::FRAC_PI_2).abs() < 1e-6);
}

#[test]
fn test_asin_simd_empty() {
    let x: Array1<f64> = array![];
    let result = asin_simd(&x.view());
    assert_eq!(result.len(), 0);
}

// Out of domain returns NaN
#[test]
fn test_asin_out_of_domain() {
    let x = array![0.5_f64, 1.5, -1.5, 2.0];
    let result = asin_simd(&x.view());

    assert!(!result[0].is_nan());
    assert!(result[1].is_nan());
    assert!(result[2].is_nan());
    assert!(result[3].is_nan());
}

// Anti-symmetry: asin(-x) = -asin(x)
#[test]
fn test_asin_anti_symmetry() {
    let x = array![0.0_f64, 0.5, 0.7, 1.0];
    let neg_x = x.mapv(|v| -v);

    let asin_x = asin_simd(&x.view());
    let asin_neg_x = asin_simd(&neg_x.view());

    for i in 0..x.len() {
        assert!(
            (asin_neg_x[i] + asin_x[i]).abs() < 1e-10,
            "asin(-x) should equal -asin(x)"
        );
    }
}

#[cfg(feature = "random")]
#[test]
fn test_asin_simd_large_array() {
    let mut rng = thread_rng();
    let uniform = Uniform::new(-1.0, 1.0).expect("Test: operation failed");
    let data: Vec<f64> = (0..10000).map(|_| uniform.sample(&mut rng)).collect();
    let x = Array1::from(data.clone());

    let result = asin_simd(&x.view());

    for i in 0..100.min(data.len()) {
        assert_eq!(result[i], data[i].asin());
    }
}

// ============================================================================
// ============================================================================
// Acos SIMD Tests
// ============================================================================

#[test]
fn test_acos_simd_f64_basic() {
    let x = array![1.0_f64, 0.5, 0.0, -1.0];
    let result = acos_simd(&x.view());

    assert!((result[0] - 0.0).abs() < 1e-10);
    assert!((result[1] - std::f64::consts::FRAC_PI_3).abs() < 1e-10);
    assert!((result[2] - std::f64::consts::FRAC_PI_2).abs() < 1e-10);
    assert!((result[3] - std::f64::consts::PI).abs() < 1e-10);
}

#[test]
fn test_acos_simd_f32_basic() {
    let x = array![1.0_f32, 0.5, 0.0, -1.0];
    let result = acos_simd(&x.view());

    assert!((result[0] - 0.0).abs() < 1e-6);
    assert!((result[1] - std::f32::consts::FRAC_PI_3).abs() < 1e-6);
    assert!((result[2] - std::f32::consts::FRAC_PI_2).abs() < 1e-6);
    assert!((result[3] - std::f32::consts::PI).abs() < 1e-6);
}

#[test]
fn test_acos_simd_empty() {
    let x: Array1<f64> = array![];
    let result = acos_simd(&x.view());
    assert_eq!(result.len(), 0);
}

// Out of domain returns NaN
#[test]
fn test_acos_out_of_domain() {
    let x = array![0.5_f64, 1.5, -1.5, 2.0];
    let result = acos_simd(&x.view());

    assert!(!result[0].is_nan());
    assert!(result[1].is_nan());
    assert!(result[2].is_nan());
    assert!(result[3].is_nan());
}

// Identity: acos(x) + asin(x) = π/2
#[test]
fn test_acos_asin_identity() {
    let x = array![0.0_f64, 0.25, 0.5, 0.75, 1.0];
    let acos_result = acos_simd(&x.view());
    let asin_result = asin_simd(&x.view());

    for i in 0..x.len() {
        let sum = acos_result[i] + asin_result[i];
        assert!(
            (sum - std::f64::consts::FRAC_PI_2).abs() < 1e-10,
            "acos(x) + asin(x) should equal π/2"
        );
    }
}

#[cfg(feature = "random")]
#[test]
fn test_acos_simd_large_array() {
    let mut rng = thread_rng();
    let uniform = Uniform::new(-1.0, 1.0).expect("Test: operation failed");
    let data: Vec<f64> = (0..10000).map(|_| uniform.sample(&mut rng)).collect();
    let x = Array1::from(data.clone());

    let result = acos_simd(&x.view());

    for i in 0..100.min(data.len()) {
        assert_eq!(result[i], data[i].acos());
    }
}

// ============================================================================
// ============================================================================
// Atan2 SIMD Tests
// ============================================================================

#[test]
fn test_atan2_simd_f64_basic() {
    let y = array![1.0_f64, 1.0, -1.0, -1.0];
    let x = array![1.0_f64, -1.0, -1.0, 1.0];
    let result = atan2_simd(&y.view(), &x.view());

    // Quadrant I: (1, 1) -> π/4
    assert!((result[0] - std::f64::consts::FRAC_PI_4).abs() < 1e-10);
    // Quadrant II: (1, -1) -> 3π/4
    assert!((result[1] - 3.0 * std::f64::consts::FRAC_PI_4).abs() < 1e-10);
    // Quadrant III: (-1, -1) -> -3π/4
    assert!((result[2] + 3.0 * std::f64::consts::FRAC_PI_4).abs() < 1e-10);
    // Quadrant IV: (-1, 1) -> -π/4
    assert!((result[3] + std::f64::consts::FRAC_PI_4).abs() < 1e-10);
}

#[test]
fn test_atan2_simd_f32_basic() {
    let y = array![1.0_f32, 0.0, -1.0];
    let x = array![0.0_f32, 1.0, 0.0];
    let result = atan2_simd(&y.view(), &x.view());

    assert!((result[0] - std::f32::consts::FRAC_PI_2).abs() < 1e-6);
    assert!((result[1] - 0.0).abs() < 1e-6);
    assert!((result[2] + std::f32::consts::FRAC_PI_2).abs() < 1e-6);
}

#[test]
fn test_atan2_simd_empty() {
    let y: Array1<f64> = array![];
    let x: Array1<f64> = array![];
    let result = atan2_simd(&y.view(), &x.view());
    assert_eq!(result.len(), 0);
}

// Test all four quadrants
#[test]
fn test_atan2_quadrants() {
    // Quadrant I: x > 0, y > 0
    let y1 = array![1.0_f64];
    let x1 = array![1.0_f64];
    let result1 = atan2_simd(&y1.view(), &x1.view());
    assert!(result1[0] > 0.0 && result1[0] < std::f64::consts::FRAC_PI_2);

    // Quadrant II: x < 0, y > 0
    let y2 = array![1.0_f64];
    let x2 = array![-1.0_f64];
    let result2 = atan2_simd(&y2.view(), &x2.view());
    assert!(result2[0] > std::f64::consts::FRAC_PI_2 && result2[0] < std::f64::consts::PI);

    // Quadrant III: x < 0, y < 0
    let y3 = array![-1.0_f64];
    let x3 = array![-1.0_f64];
    let result3 = atan2_simd(&y3.view(), &x3.view());
    assert!(result3[0] < -std::f64::consts::FRAC_PI_2 && result3[0] > -std::f64::consts::PI);

    // Quadrant IV: x > 0, y < 0
    let y4 = array![-1.0_f64];
    let x4 = array![1.0_f64];
    let result4 = atan2_simd(&y4.view(), &x4.view());
    assert!(result4[0] < 0.0 && result4[0] > -std::f64::consts::FRAC_PI_2);
}

// Anti-symmetry: atan2(-y, x) = -atan2(y, x)
#[test]
fn test_atan2_anti_symmetry() {
    let y = array![0.5_f64, 1.0, 2.0];
    let x = array![1.0_f64, 1.0, 1.0];
    let neg_y = y.mapv(|v| -v);

    let atan2_y_x = atan2_simd(&y.view(), &x.view());
    let atan2_neg_y_x = atan2_simd(&neg_y.view(), &x.view());

    for i in 0..y.len() {
        assert!(
            (atan2_neg_y_x[i] + atan2_y_x[i]).abs() < 1e-10,
            "atan2(-y, x) should equal -atan2(y, x)"
        );
    }
}

#[cfg(feature = "random")]
#[test]
fn test_atan2_simd_large_array() {
    let mut rng = thread_rng();
    let uniform = Uniform::new(-10.0, 10.0).expect("Test: operation failed");
    let y_data: Vec<f64> = (0..10000).map(|_| uniform.sample(&mut rng)).collect();
    let x_data: Vec<f64> = (0..10000).map(|_| uniform.sample(&mut rng)).collect();
    let y = Array1::from(y_data.clone());
    let x = Array1::from(x_data.clone());

    let result = atan2_simd(&y.view(), &x.view());

    for i in 0..100.min(y_data.len()) {
        assert_eq!(result[i], y_data[i].atan2(x_data[i]));
    }
}

// ============================================================================
// Inverse Function Property Tests
// ============================================================================

// Property: sin(asin(x)) = x for x ∈ [-1, 1]
#[test]
fn test_sin_asin_inverse() {
    let x = array![0.0_f64, 0.25, 0.5, 0.75, 1.0, -0.5, -1.0];
    let asin_result = asin_simd(&x.view());
    let sin_result = sin_simd(&asin_result.view());

    for i in 0..x.len() {
        assert!(
            (sin_result[i] - x[i]).abs() < 1e-10,
            "sin(asin(x)) should equal x"
        );
    }
}

// Property: cos(acos(x)) = x for x ∈ [-1, 1]
#[test]
fn test_cos_acos_inverse() {
    let x = array![1.0_f64, 0.75, 0.5, 0.25, 0.0, -0.5, -1.0];
    let acos_result = acos_simd(&x.view());
    let cos_result = cos_simd(&acos_result.view());

    for i in 0..x.len() {
        assert!(
            (cos_result[i] - x[i]).abs() < 1e-10,
            "cos(acos(x)) should equal x"
        );
    }
}

// Property: tan(atan(x)) = x for all x
#[test]
fn test_tan_atan_inverse() {
    let x = array![0.0_f64, 0.5, 1.0, 2.0, -1.0, -2.0];
    let atan_result = atan_simd(&x.view());
    let tan_result = tan_simd(&atan_result.view());

    for i in 0..x.len() {
        assert!(
            (tan_result[i] - x[i]).abs() < 1e-10,
            "tan(atan(x)) should equal x"
        );
    }
}

// ============================================================================
// ============================================================================
// To_radians SIMD Tests (Phase 42b)
// ============================================================================

use scirs2_core::ndarray_ext::elementwise::to_radians_simd;

#[test]
fn test_to_radians_simd_f64_basic() {
    let degrees = array![0.0_f64, 90.0, 180.0, 270.0, 360.0];
    let result = to_radians_simd(&degrees.view());

    let pi = std::f64::consts::PI;
    assert!((result[0] - 0.0).abs() < 1e-14);
    assert!((result[1] - pi / 2.0).abs() < 1e-14);
    assert!((result[2] - pi).abs() < 1e-14);
    assert!((result[3] - 3.0 * pi / 2.0).abs() < 1e-14);
    assert!((result[4] - 2.0 * pi).abs() < 1e-13);
}

#[test]
fn test_to_radians_simd_f32_basic() {
    let degrees = array![0.0_f32, 90.0, 180.0, 270.0, 360.0];
    let result = to_radians_simd(&degrees.view());

    let pi = std::f32::consts::PI;
    assert!((result[0] - 0.0).abs() < 1e-6);
    assert!((result[1] - pi / 2.0).abs() < 1e-6);
    assert!((result[2] - pi).abs() < 1e-6);
    assert!((result[3] - 3.0 * pi / 2.0).abs() < 1e-5);
    assert!((result[4] - 2.0 * pi).abs() < 1e-5);
}

#[test]
fn test_to_radians_simd_negative() {
    let degrees = array![-90.0_f64, -180.0, -45.0];
    let result = to_radians_simd(&degrees.view());

    let pi = std::f64::consts::PI;
    assert!((result[0] - (-pi / 2.0)).abs() < 1e-14);
    assert!((result[1] - (-pi)).abs() < 1e-14);
    assert!((result[2] - (-pi / 4.0)).abs() < 1e-14);
}

#[test]
fn test_to_radians_simd_empty() {
    let x: Array1<f64> = array![];
    let result = to_radians_simd(&x.view());
    assert_eq!(result.len(), 0);
}

// ============================================================================
// To_degrees SIMD Tests (Phase 42b)
// ============================================================================

use scirs2_core::ndarray_ext::elementwise::to_degrees_simd;

#[test]
fn test_to_degrees_simd_f64_basic() {
    let pi = std::f64::consts::PI;
    let radians = array![0.0_f64, pi / 2.0, pi, 3.0 * pi / 2.0, 2.0 * pi];
    let result = to_degrees_simd(&radians.view());

    assert!((result[0] - 0.0).abs() < 1e-14);
    assert!((result[1] - 90.0).abs() < 1e-13);
    assert!((result[2] - 180.0).abs() < 1e-13);
    assert!((result[3] - 270.0).abs() < 1e-12);
    assert!((result[4] - 360.0).abs() < 1e-12);
}

#[test]
fn test_to_degrees_simd_f32_basic() {
    let pi = std::f32::consts::PI;
    let radians = array![0.0_f32, pi / 2.0, pi, 3.0 * pi / 2.0, 2.0 * pi];
    let result = to_degrees_simd(&radians.view());

    assert!((result[0] - 0.0).abs() < 1e-5);
    assert!((result[1] - 90.0).abs() < 1e-4);
    assert!((result[2] - 180.0).abs() < 1e-4);
    assert!((result[3] - 270.0).abs() < 1e-4);
    assert!((result[4] - 360.0).abs() < 1e-3);
}

#[test]
fn test_to_degrees_simd_negative() {
    let pi = std::f64::consts::PI;
    let radians = array![-pi / 2.0, -pi, -pi / 4.0];
    let result = to_degrees_simd(&radians.view());

    assert!((result[0] - (-90.0)).abs() < 1e-13);
    assert!((result[1] - (-180.0)).abs() < 1e-13);
    assert!((result[2] - (-45.0)).abs() < 1e-13);
}

#[test]
fn test_to_radians_to_degrees_roundtrip() {
    // Converting degrees -> radians -> degrees should be identity
    let degrees = array![0.0_f64, 45.0, 90.0, 135.0, 180.0, 270.0, 360.0];
    let radians = to_radians_simd(&degrees.view());
    let back = to_degrees_simd(&radians.view());

    for i in 0..degrees.len() {
        assert!(
            (back[i] - degrees[i]).abs() < 1e-12,
            "to_degrees(to_radians(x)) should equal x at index {}: got {}, expected {}",
            i,
            back[i],
            degrees[i]
        );
    }
}

#[test]
fn test_to_degrees_to_radians_roundtrip() {
    // Converting radians -> degrees -> radians should be identity
    let pi = std::f64::consts::PI;
    let radians = array![0.0_f64, pi / 4.0, pi / 2.0, pi, 3.0 * pi / 2.0, 2.0 * pi];
    let degrees = to_degrees_simd(&radians.view());
    let back = to_radians_simd(&degrees.view());

    for i in 0..radians.len() {
        assert!(
            (back[i] - radians[i]).abs() < 1e-14,
            "to_radians(to_degrees(x)) should equal x at index {}: got {}, expected {}",
            i,
            back[i],
            radians[i]
        );
    }
}
