//! Auto-generated module
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use ::ndarray::{Array1, Array2, ArrayView1, ArrayView2, ArrayViewMut1};

use super::functions::SimdUnifiedOps;

/// Standalone ultra-optimized dot product function for f32
pub fn simd_dot_f32_ultra(a: &ArrayView1<f32>, b: &ArrayView1<f32>) -> f32 {
    f32::simd_dot_f32_ultra(a, b)
}
/// Standalone ultra-optimized FMA function for f32
pub fn simd_fma_f32_ultra(
    a: &ArrayView1<f32>,
    b: &ArrayView1<f32>,
    c: &ArrayView1<f32>,
    result: &mut ArrayViewMut1<f32>,
) {
    f32::simd_fma_f32_ultra(a, b, c, result)
}
/// Additional standalone functions that might be needed
pub fn simd_add_f32_adaptive(a: &ArrayView1<f32>, b: &ArrayView1<f32>) -> Array1<f32> {
    f32::simd_add_adaptive(a, b)
}
pub fn simd_mul_f32_hyperoptimized(a: &ArrayView1<f32>, b: &ArrayView1<f32>) -> Array1<f32> {
    f32::simd_mul(a, b)
}
/// Helper functions for `Vec<T>` compatibility
/// These functions accept `Vec<T>` and internally convert to Array types
///
/// Helper function for Vec-based SIMD multiplication
pub fn simd_mul_f32_ultra_vec(a: &Vec<f32>, b: &Vec<f32>, result: &mut Vec<f32>) {
    let a_array = Array1::from_vec(a.clone());
    let b_array = Array1::from_vec(b.clone());
    let mut result_array = Array1::from_vec(result.clone());
    f32::simd_mul_f32_ultra(
        &a_array.view(),
        &b_array.view(),
        &mut result_array.view_mut(),
    );
    *result = result_array.to_vec();
}
/// Helper function for Vec-based SIMD addition
pub fn simd_add_f32_ultra_vec(a: &Vec<f32>, b: &Vec<f32>, result: &mut Vec<f32>) {
    let a_array = Array1::from_vec(a.clone());
    let b_array = Array1::from_vec(b.clone());
    let mut result_array = Array1::from_vec(result.clone());
    f32::simd_add_f32_ultra(
        &a_array.view(),
        &b_array.view(),
        &mut result_array.view_mut(),
    );
    *result = result_array.to_vec();
}
/// Helper function for Vec-based SIMD division
pub fn simd_div_f32_ultra_vec(a: &Vec<f32>, b: &Vec<f32>, result: &mut Vec<f32>) {
    let a_array = Array1::from_vec(a.clone());
    let b_array = Array1::from_vec(b.clone());
    let mut result_array = Array1::from_vec(result.clone());
    f32::simd_div_f32_ultra(
        &a_array.view(),
        &b_array.view(),
        &mut result_array.view_mut(),
    );
    *result = result_array.to_vec();
}
/// Helper function for Vec-based SIMD sine
pub fn simd_sin_f32_ultra_vec(a: &[f32], result: &mut Vec<f32>) {
    let a_array = Array1::from_vec(a.to_owned());
    let mut result_array = Array1::from_vec(result.clone());
    f32::simd_sin_f32_ultra(&a_array.view(), &mut result_array.view_mut());
    *result = result_array.to_vec();
}
/// Helper function for Vec-based SIMD subtraction
pub fn simd_sub_f32_ultra_vec(a: &[f32], b: &[f32], result: &mut Vec<f32>) {
    let a_array = Array1::from_vec(a.to_owned());
    let b_array = Array1::from_vec(b.to_owned());
    let mut result_array = Array1::from_vec(result.clone());
    f32::simd_sub_f32_ultra(
        &a_array.view(),
        &b_array.view(),
        &mut result_array.view_mut(),
    );
    *result = result_array.to_vec();
}
/// Helper function for Vec-based SIMD FMA
pub fn simd_fma_f32_ultra_vec(a: &[f32], b: &[f32], c: &[f32], result: &mut Vec<f32>) {
    let a_array = Array1::from_vec(a.to_owned());
    let b_array = Array1::from_vec(b.to_owned());
    let c_array = Array1::from_vec(c.to_owned());
    let mut result_array = Array1::from_vec(result.clone());
    f32::simd_fma_f32_ultra(
        &a_array.view(),
        &b_array.view(),
        &c_array.view(),
        &mut result_array.view_mut(),
    );
    *result = result_array.to_vec();
}
/// Helper function for Vec-based SIMD power
pub fn simd_pow_f32_ultra_vec(a: &[f32], b: &[f32], result: &mut Vec<f32>) {
    let a_array = Array1::from_vec(a.to_owned());
    let b_array = Array1::from_vec(b.to_owned());
    let mut result_array = Array1::from_vec(result.clone());
    f32::simd_pow_f32_ultra(
        &a_array.view(),
        &b_array.view(),
        &mut result_array.view_mut(),
    );
    *result = result_array.to_vec();
}
/// Helper function for Vec-based SIMD exp
pub fn simd_exp_f32_ultra_vec(a: &[f32], result: &mut Vec<f32>) {
    let a_array = Array1::from_vec(a.to_owned());
    let mut result_array = Array1::from_vec(result.clone());
    f32::simd_exp_f32_ultra(&a_array.view(), &mut result_array.view_mut());
    *result = result_array.to_vec();
}
/// Helper function for Vec-based SIMD cos
pub fn simd_cos_f32_ultra_vec(a: &[f32], result: &mut Vec<f32>) {
    let a_array = Array1::from_vec(a.to_owned());
    let mut result_array = Array1::from_vec(result.clone());
    f32::simd_cos_f32_ultra(&a_array.view(), &mut result_array.view_mut());
    *result = result_array.to_vec();
}
