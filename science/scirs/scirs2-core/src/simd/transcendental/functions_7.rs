//! Auto-generated module
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use ndarray::{Array1, ArrayView1};

use super::functions::{simd_exp_f32, simd_exp_f64, simd_exp_fast_f32};
use super::functions_2::{simd_sigmoid_f32, simd_sigmoid_f64};
use super::functions_3::{simd_gelu_f32, simd_gelu_f64, simd_swish_f32, simd_swish_f64};
use super::functions_4::{simd_softplus_f32, simd_softplus_f64, simd_tanh_f32, simd_tanh_f64};
use super::functions_5::{simd_ln_f32, simd_ln_f64, simd_mish_f32, simd_mish_f64, simd_sin_f32};
use super::functions_6::{
    simd_cos_f32, simd_cos_f64, simd_log10_f32, simd_log10_f64, simd_log2_f32, simd_log2_f64,
    simd_sin_f64,
};

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;
    #[test]
    fn test_simd_exp_f32_basic() {
        let x = array![0.0f32, 1.0, -1.0, 2.0, -2.0];
        let result = simd_exp_f32(&x.view());
        assert!((result[0] - 1.0).abs() < 1e-6);
        assert!((result[1] - std::f32::consts::E).abs() < 1e-5);
        assert!((result[2] - 1.0 / std::f32::consts::E).abs() < 1e-6);
        assert!((result[3] - std::f32::consts::E.powi(2)).abs() < 1e-4);
        assert!((result[4] - 1.0 / std::f32::consts::E.powi(2)).abs() < 1e-6);
    }
    #[test]
    fn test_simd_exp_f64_basic() {
        let x = array![0.0f64, 1.0, -1.0, 2.0, -2.0];
        let result = simd_exp_f64(&x.view());
        assert!((result[0] - 1.0).abs() < 1e-10);
        assert!((result[1] - std::f64::consts::E).abs() < 1e-9);
        assert!((result[2] - 1.0 / std::f64::consts::E).abs() < 1e-10);
        assert!((result[3] - std::f64::consts::E.powi(2)).abs() < 1e-8);
        assert!((result[4] - 1.0 / std::f64::consts::E.powi(2)).abs() < 1e-10);
    }
    #[test]
    fn test_simd_exp_f32_large_array() {
        let x: Array1<f32> = Array1::linspace(-10.0, 10.0, 1000);
        let result = simd_exp_f32(&x.view());
        for (i, (&xi, &ri)) in x.iter().zip(result.iter()).enumerate() {
            let expected = xi.exp();
            let rel_error = if expected.abs() > 1e-30 {
                (ri - expected).abs() / expected.abs()
            } else {
                (ri - expected).abs()
            };
            assert!(
                rel_error < 1e-5,
                "Index {}: exp({}) = {}, got {}, rel_error = {}",
                i,
                xi,
                expected,
                ri,
                rel_error
            );
        }
    }
    #[test]
    fn test_simd_exp_f64_large_array() {
        let x: Array1<f64> = Array1::linspace(-10.0, 10.0, 1000);
        let result = simd_exp_f64(&x.view());
        for (i, (&xi, &ri)) in x.iter().zip(result.iter()).enumerate() {
            let expected = xi.exp();
            let rel_error = if expected.abs() > 1e-300 {
                (ri - expected).abs() / expected.abs()
            } else {
                (ri - expected).abs()
            };
            assert!(
                rel_error < 1e-9,
                "Index {}: exp({}) = {}, got {}, rel_error = {}",
                i,
                xi,
                expected,
                ri,
                rel_error
            );
        }
    }
    #[test]
    fn test_simd_exp_fast_f32() {
        let x = array![0.0f32, 1.0, -1.0, 2.0];
        let result = simd_exp_fast_f32(&x.view());
        assert!((result[0] - 1.0).abs() < 0.5);
        assert!((result[1] - std::f32::consts::E).abs() < 1.0);
    }
    #[test]
    fn test_simd_exp_f32_edge_cases() {
        let x = array![10.0f32, -10.0f32, 0.0f32];
        let result = simd_exp_f32(&x.view());
        assert!(result[0].is_finite() && result[0] > 0.0);
        assert!(result[1].is_finite() && result[1] > 0.0);
        assert!((result[2] - 1.0).abs() < 1e-6);
        let x_moderate = array![50.0f32, -50.0f32];
        let result_moderate = simd_exp_f32(&x_moderate.view());
        assert!(result_moderate[0].is_finite() && result_moderate[0] > 0.0);
        assert!(result_moderate[1].is_finite() && result_moderate[1] > 0.0);
    }
    #[test]
    fn test_simd_exp_f32_empty() {
        let x: Array1<f32> = Array1::zeros(0);
        let result = simd_exp_f32(&x.view());
        assert_eq!(result.len(), 0);
    }
    #[test]
    fn test_simd_sigmoid_f32_basic() {
        let x = array![0.0f32, 2.0, -2.0, 5.0, -5.0];
        let result = simd_sigmoid_f32(&x.view());
        assert!((result[0] - 0.5).abs() < 1e-5);
        assert!((result[1] - 0.8807970779778823).abs() < 1e-4);
        assert!((result[2] - 0.11920292202211755).abs() < 1e-4);
        assert!((result[3] - 0.9933071490757153).abs() < 1e-4);
        assert!((result[4] - 0.006692850924284856).abs() < 1e-4);
    }
    #[test]
    fn test_simd_sigmoid_f64_basic() {
        let x = array![0.0f64, 2.0, -2.0];
        let result = simd_sigmoid_f64(&x.view());
        assert!((result[0] - 0.5).abs() < 1e-10);
        assert!((result[1] - 0.8807970779778823).abs() < 1e-8);
        assert!((result[2] - 0.11920292202211755).abs() < 1e-8);
    }
    #[test]
    fn test_simd_sigmoid_f32_large_array() {
        let x: Array1<f32> = Array1::linspace(-10.0, 10.0, 1000);
        let result = simd_sigmoid_f32(&x.view());
        for (i, (&xi, &ri)) in x.iter().zip(result.iter()).enumerate() {
            let expected = if xi >= 0.0 {
                1.0 / (1.0 + (-xi).exp())
            } else {
                let exp_x = xi.exp();
                exp_x / (1.0 + exp_x)
            };
            let error = (ri - expected).abs();
            assert!(
                error < 1e-4,
                "Index {}: sigmoid({}) = {}, got {}, error = {}",
                i,
                xi,
                expected,
                ri,
                error
            );
        }
    }
    #[test]
    fn test_simd_sigmoid_f32_properties() {
        let x = array![1.0f32, -1.0f32];
        let result = simd_sigmoid_f32(&x.view());
        let sum = result[0] + result[1];
        assert!(
            (sum - 1.0).abs() < 1e-5,
            "sigmoid(x) + sigmoid(-x) should equal 1"
        );
        let x_moderate = array![-10.0f32, 10.0f32];
        let result_moderate = simd_sigmoid_f32(&x_moderate.view());
        assert!(result_moderate[0] > 0.0 && result_moderate[0] < 0.001);
        assert!(result_moderate[1] > 0.999 && result_moderate[1] <= 1.0);
    }
    #[test]
    fn test_simd_sigmoid_f32_empty() {
        let x: Array1<f32> = Array1::zeros(0);
        let result = simd_sigmoid_f32(&x.view());
        assert_eq!(result.len(), 0);
    }
    #[test]
    fn test_simd_gelu_f32_basic() {
        let x = array![0.0f32, 1.0, -1.0, 2.0, -2.0];
        let result = simd_gelu_f32(&x.view());
        assert!(result[0].abs() < 1e-6);
        assert!((result[1] - 0.8412).abs() < 0.02);
        assert!((result[2] - (-0.1588)).abs() < 0.02);
        assert!(
            (result[3] - 1.9546).abs() < 0.05,
            "GELU(2) = {}, expected ~1.9546",
            result[3]
        );
    }
    #[test]
    fn test_simd_gelu_f64_basic() {
        let x = array![0.0f64, 1.0, -1.0];
        let result = simd_gelu_f64(&x.view());
        assert!(result[0].abs() < 1e-10);
        assert!((result[1] - 0.8413).abs() < 0.01);
        assert!((result[2] - (-0.1587)).abs() < 0.01);
    }
    #[test]
    fn test_simd_gelu_f32_large_array() {
        let x: Array1<f32> = Array1::linspace(-5.0, 5.0, 1000);
        let result = simd_gelu_f32(&x.view());
        for (&xi, &ri) in x.iter().zip(result.iter()) {
            if xi > 3.0 {
                assert!((ri - xi).abs() < 0.1);
            } else if xi < -3.0 {
                assert!(ri.abs() < 0.1);
            }
        }
    }
    #[test]
    fn test_simd_gelu_f32_empty() {
        let x: Array1<f32> = Array1::zeros(0);
        let result = simd_gelu_f32(&x.view());
        assert_eq!(result.len(), 0);
    }
    #[test]
    fn test_simd_swish_f32_basic() {
        let x = array![0.0f32, 1.0, -1.0, 2.0, -2.0];
        let result = simd_swish_f32(&x.view());
        assert!(result[0].abs() < 1e-6);
        assert!((result[1] - 0.7311).abs() < 0.01);
        assert!((result[2] - (-0.2689)).abs() < 0.01);
        assert!((result[3] - 1.7616).abs() < 0.01);
    }
    #[test]
    fn test_simd_swish_f64_basic() {
        let x = array![0.0f64, 1.0, -1.0];
        let result = simd_swish_f64(&x.view());
        assert!(result[0].abs() < 1e-10);
        assert!((result[1] - 0.7311).abs() < 0.01);
        assert!((result[2] - (-0.2689)).abs() < 0.01);
    }
    #[test]
    fn test_simd_swish_f32_large_array() {
        let x: Array1<f32> = Array1::linspace(-10.0, 10.0, 1000);
        let result = simd_swish_f32(&x.view());
        for (i, (&xi, &ri)) in x.iter().zip(result.iter()).enumerate() {
            let sigmoid = if xi >= 0.0 {
                1.0 / (1.0 + (-xi).exp())
            } else {
                let exp_x = xi.exp();
                exp_x / (1.0 + exp_x)
            };
            let expected = xi * sigmoid;
            let error = (ri - expected).abs();
            assert!(
                error < 1e-4,
                "Index {}: swish({}) = {}, got {}, error = {}",
                i,
                xi,
                expected,
                ri,
                error
            );
        }
    }
    #[test]
    fn test_simd_swish_f32_empty() {
        let x: Array1<f32> = Array1::zeros(0);
        let result = simd_swish_f32(&x.view());
        assert_eq!(result.len(), 0);
    }
    #[test]
    fn test_simd_softplus_f32_basic() {
        let x = array![0.0f32, 1.0, -1.0, 5.0, -5.0];
        let result = simd_softplus_f32(&x.view());
        assert!((result[0] - std::f32::consts::LN_2).abs() < 1e-4);
        assert!((result[1] - 1.3133).abs() < 0.01);
        assert!((result[2] - 0.3133).abs() < 0.01);
        assert!((result[3] - 5.0067).abs() < 0.01);
    }
    #[test]
    fn test_simd_softplus_f64_basic() {
        let x = array![0.0f64, 1.0, -1.0];
        let result = simd_softplus_f64(&x.view());
        assert!((result[0] - std::f64::consts::LN_2).abs() < 1e-10);
        assert!((result[1] - 1.3133).abs() < 0.01);
        assert!((result[2] - 0.3133).abs() < 0.01);
    }
    #[test]
    fn test_simd_softplus_f32_properties() {
        let x = array![100.0f32, -100.0f32];
        let result = simd_softplus_f32(&x.view());
        assert!((result[0] - 100.0).abs() < 0.1);
        assert!(result[1] < 1e-10);
        let x2: Array1<f32> = Array1::linspace(-10.0, 10.0, 100);
        let result2 = simd_softplus_f32(&x2.view());
        for &val in result2.iter() {
            assert!(val > 0.0, "Softplus should always be positive");
        }
    }
    #[test]
    fn test_simd_softplus_f32_large_array() {
        let x: Array1<f32> = Array1::linspace(-10.0, 10.0, 1000);
        let result = simd_softplus_f32(&x.view());
        for (i, (&xi, &ri)) in x.iter().zip(result.iter()).enumerate() {
            let expected = (1.0 + xi.exp()).ln();
            let error = (ri - expected).abs();
            assert!(
                error < 1e-4,
                "Index {}: softplus({}) = {}, got {}, error = {}",
                i,
                xi,
                expected,
                ri,
                error
            );
        }
    }
    #[test]
    fn test_simd_softplus_f32_empty() {
        let x: Array1<f32> = Array1::zeros(0);
        let result = simd_softplus_f32(&x.view());
        assert_eq!(result.len(), 0);
    }
    #[test]
    fn test_simd_tanh_f32_basic() {
        let x = array![0.0f32, 1.0, -1.0, 2.0, -2.0];
        let result = simd_tanh_f32(&x.view());
        assert!(result[0].abs() < 1e-6);
        assert!((result[1] - 0.7616).abs() < 0.01);
        assert!((result[2] - (-0.7616)).abs() < 0.01);
        assert!((result[3] - 0.9640).abs() < 0.01);
    }
    #[test]
    fn test_simd_tanh_f64_basic() {
        let x = array![0.0f64, 1.0, -1.0];
        let result = simd_tanh_f64(&x.view());
        assert!(result[0].abs() < 1e-10);
        assert!((result[1] - 0.7615941559557649).abs() < 1e-6);
        assert!((result[2] - (-0.7615941559557649)).abs() < 1e-6);
    }
    #[test]
    fn test_simd_tanh_f32_large_array() {
        let x: Array1<f32> = Array1::linspace(-5.0, 5.0, 1000);
        let result = simd_tanh_f32(&x.view());
        for (i, (&xi, &ri)) in x.iter().zip(result.iter()).enumerate() {
            let expected = xi.tanh();
            let error = (ri - expected).abs();
            assert!(
                error < 1e-4,
                "Index {}: tanh({}) = {}, got {}, error = {}",
                i,
                xi,
                expected,
                ri,
                error
            );
        }
    }
    #[test]
    fn test_simd_tanh_f32_properties() {
        let x = array![1.5f32, -1.5f32];
        let result = simd_tanh_f32(&x.view());
        assert!((result[0] + result[1]).abs() < 1e-5);
        let x_extreme = array![100.0f32, -100.0f32];
        let result_extreme = simd_tanh_f32(&x_extreme.view());
        assert!((result_extreme[0] - 1.0).abs() < 1e-5);
        assert!((result_extreme[1] - (-1.0)).abs() < 1e-5);
    }
    #[test]
    fn test_simd_tanh_f32_empty() {
        let x: Array1<f32> = Array1::zeros(0);
        let result = simd_tanh_f32(&x.view());
        assert_eq!(result.len(), 0);
    }
    #[test]
    fn test_simd_mish_f32_basic() {
        let x = array![0.0f32, 1.0, -1.0, 2.0];
        let result = simd_mish_f32(&x.view());
        assert!(result[0].abs() < 1e-6);
        assert!((result[1] - 0.8651).abs() < 0.02);
        assert!((result[2] - (-0.3034)).abs() < 0.02);
    }
    #[test]
    fn test_simd_mish_f64_basic() {
        let x = array![0.0f64, 1.0, -1.0];
        let result = simd_mish_f64(&x.view());
        assert!(result[0].abs() < 1e-10);
        assert!((result[1] - 0.8651).abs() < 0.02);
    }
    #[test]
    fn test_simd_mish_f32_large_array() {
        let x: Array1<f32> = Array1::linspace(-5.0, 5.0, 100);
        let result = simd_mish_f32(&x.view());
        for (&xi, &ri) in x.iter().zip(result.iter()) {
            let sp = if xi > 20.0 { xi } else { (1.0 + xi.exp()).ln() };
            let expected = xi * sp.tanh();
            let error = (ri - expected).abs();
            assert!(
                error < 1e-3,
                "mish({}) = {}, got {}, error = {}",
                xi,
                expected,
                ri,
                error
            );
        }
    }
    #[test]
    fn test_simd_mish_f32_empty() {
        let x: Array1<f32> = Array1::zeros(0);
        let result = simd_mish_f32(&x.view());
        assert_eq!(result.len(), 0);
    }
    #[test]
    fn test_simd_ln_f32_basic() {
        let x = array![1.0f32, std::f32::consts::E, 10.0, 100.0];
        let result = simd_ln_f32(&x.view());
        assert!(result[0].abs() < 1e-5);
        assert!((result[1] - 1.0).abs() < 1e-4);
        assert!((result[2] - std::f32::consts::LN_10).abs() < 0.01);
        assert!((result[3] - 4.6052).abs() < 0.02);
    }
    #[test]
    fn test_simd_ln_f64_basic() {
        let x = array![1.0f64, std::f64::consts::E, 10.0];
        let result = simd_ln_f64(&x.view());
        assert!(result[0].abs() < 1e-10);
        assert!((result[1] - 1.0).abs() < 1e-8);
        assert!((result[2] - std::f64::consts::LN_10).abs() < 1e-6);
    }
    #[test]
    fn test_simd_ln_f32_large_array() {
        let x: Array1<f32> = Array1::linspace(0.1, 100.0, 1000);
        let result = simd_ln_f32(&x.view());
        for (i, (&xi, &ri)) in x.iter().zip(result.iter()).enumerate() {
            let expected = xi.ln();
            let error = (ri - expected).abs();
            assert!(
                error < 0.15,
                "Index {}: ln({}) = {}, got {}, error = {}",
                i,
                xi,
                expected,
                ri,
                error
            );
        }
    }
    #[test]
    fn test_simd_ln_f32_empty() {
        let x: Array1<f32> = Array1::zeros(0);
        let result = simd_ln_f32(&x.view());
        assert_eq!(result.len(), 0);
    }
    #[test]
    fn test_simd_sin_f32_basic() {
        let x = array![
            0.0f32,
            std::f32::consts::FRAC_PI_2,
            std::f32::consts::PI,
            std::f32::consts::FRAC_PI_4
        ];
        let result = simd_sin_f32(&x.view());
        assert!(result[0].abs() < 1e-5);
        assert!((result[1] - 1.0).abs() < 1e-4);
        assert!(result[2].abs() < 1e-4);
        assert!((result[3] - std::f32::consts::FRAC_1_SQRT_2).abs() < 1e-3);
    }
    #[test]
    fn test_simd_sin_f64_basic() {
        let x = array![0.0f64, std::f64::consts::FRAC_PI_2, std::f64::consts::PI];
        let result = simd_sin_f64(&x.view());
        assert!(result[0].abs() < 1e-5);
        assert!((result[1] - 1.0).abs() < 1e-4);
        assert!(result[2].abs() < 1e-4);
    }
    #[test]
    fn test_simd_sin_f32_large_array() {
        let x: Array1<f32> = Array1::linspace(
            -2.0 * std::f32::consts::PI,
            2.0 * std::f32::consts::PI,
            1000,
        );
        let result = simd_sin_f32(&x.view());
        for (i, (&xi, &ri)) in x.iter().zip(result.iter()).enumerate() {
            let expected = xi.sin();
            let error = (ri - expected).abs();
            assert!(
                error < 1e-3,
                "Index {}: sin({}) = {}, got {}, error = {}",
                i,
                xi,
                expected,
                ri,
                error
            );
        }
    }
    #[test]
    fn test_simd_sin_f32_properties() {
        let x = array![1.0f32, -1.0f32];
        let result = simd_sin_f32(&x.view());
        assert!((result[0] + result[1]).abs() < 1e-4);
        let x_large: Array1<f32> = Array1::linspace(-10.0, 10.0, 100);
        let result_large = simd_sin_f32(&x_large.view());
        for &val in result_large.iter() {
            assert!(val >= -1.0 && val <= 1.0);
        }
    }
    #[test]
    fn test_simd_cos_f32_basic() {
        let x = array![
            0.0f32,
            std::f32::consts::PI,
            std::f32::consts::FRAC_PI_2,
            std::f32::consts::FRAC_PI_4
        ];
        let result = simd_cos_f32(&x.view());
        assert!((result[0] - 1.0).abs() < 1e-5);
        assert!((result[1] - (-1.0)).abs() < 1e-4);
        assert!(result[2].abs() < 1e-4);
        assert!((result[3] - std::f32::consts::FRAC_1_SQRT_2).abs() < 1e-3);
    }
    #[test]
    fn test_simd_cos_f64_basic() {
        let x = array![0.0f64, std::f64::consts::PI, std::f64::consts::FRAC_PI_2];
        let result = simd_cos_f64(&x.view());
        assert!((result[0] - 1.0).abs() < 1e-4);
        assert!((result[1] - (-1.0)).abs() < 1e-4);
        assert!(result[2].abs() < 1e-4);
    }
    #[test]
    fn test_simd_cos_f32_properties() {
        let x = array![1.5f32, -1.5f32];
        let result = simd_cos_f32(&x.view());
        assert!((result[0] - result[1]).abs() < 1e-4);
        let x_test = array![1.0f32];
        let sin_result = simd_sin_f32(&x_test.view());
        let cos_result = simd_cos_f32(&x_test.view());
        let sum_sq = sin_result[0] * sin_result[0] + cos_result[0] * cos_result[0];
        assert!((sum_sq - 1.0).abs() < 1e-3);
    }
    #[test]
    fn test_simd_sin_f32_empty() {
        let x: Array1<f32> = Array1::zeros(0);
        let result = simd_sin_f32(&x.view());
        assert_eq!(result.len(), 0);
    }
    #[test]
    fn test_simd_cos_f32_empty() {
        let x: Array1<f32> = Array1::zeros(0);
        let result = simd_cos_f32(&x.view());
        assert_eq!(result.len(), 0);
    }
    #[test]
    fn test_simd_log2_f32_basic() {
        let x = array![1.0f32, 2.0, 4.0, 8.0];
        let result = simd_log2_f32(&x.view());
        assert!(result[0].abs() < 1e-4);
        assert!((result[1] - 1.0).abs() < 0.02);
        assert!((result[2] - 2.0).abs() < 0.05);
        assert!((result[3] - 3.0).abs() < 0.1);
    }
    #[test]
    fn test_simd_log10_f32_basic() {
        let x = array![1.0f32, 10.0, 100.0];
        let result = simd_log10_f32(&x.view());
        assert!(result[0].abs() < 1e-4);
        assert!((result[1] - 1.0).abs() < 0.02);
        assert!((result[2] - 2.0).abs() < 0.05);
    }
    #[test]
    fn test_simd_log2_f64_basic() {
        let x = array![1.0f64, 2.0, 4.0];
        let result = simd_log2_f64(&x.view());
        assert!(result[0].abs() < 1e-10);
        assert!((result[1] - 1.0).abs() < 1e-6);
        assert!((result[2] - 2.0).abs() < 1e-6);
    }
    #[test]
    fn test_simd_log10_f64_basic() {
        let x = array![1.0f64, 10.0, 100.0];
        let result = simd_log10_f64(&x.view());
        assert!(result[0].abs() < 1e-10);
        assert!((result[1] - 1.0).abs() < 1e-6);
        assert!((result[2] - 2.0).abs() < 1e-6);
    }
}
