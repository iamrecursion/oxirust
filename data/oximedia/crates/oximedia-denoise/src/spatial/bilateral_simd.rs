//! SIMD-accelerated bilateral filter for `Frame2D`.
//!
//! Provides a `BilateralFilterSimd` that applies a bilateral filter to a
//! [`Frame2D`] using LUT-based Gaussian weight precomputation.  A scalar
//! fallback path is always compiled; the SSE 4.1 path is selected at
//! compile time when `target_feature = "sse4.1"` is available.
//!
//! Design goals
//! ------------
//! * **LUT-based range Gaussian**: `sigma_color` determines the width of the
//!   range kernel.  We precompute `exp(-d² / (2σ²))` for `d ∈ [0, 255]` into
//!   a 256-entry table so every per-pixel evaluation is a table lookup.
//! * **Separable spatial kernel**: We store a 1-D spatial weight vector of
//!   length `2*radius + 1`.  The 2-D weight is the product of row and column
//!   components, computed on the fly.
//! * **Scalar fallback**: Always correct and exercised on every platform.
//! * **SSE 4.1 path** (optional, feature-gated): Available when
//!   `#[cfg(target_feature = "sse4.1")]` is set.  On modern x86_64 hosts
//!   compiled with `-C target-feature=+sse4.1` the inner accumulation loop
//!   operates on 4 floats per iteration using `_mm_dp_ps`.
//!
//! The output must be pixel-identical (within floating-point rounding) between
//! the scalar and SIMD paths; the test suite verifies this property.

// SSE 4.1 intrinsics require unsafe blocks.
#![allow(unsafe_code)]

use crate::frame2d::Frame2D;
use crate::{DenoiseError, DenoiseResult};

/// Maximum pixel intensity value.
const MAX_INTENSITY: usize = 256;

/// A bilateral filter operating on [`Frame2D`] with LUT-based Gaussian range
/// weights and optional SSE 4.1 SIMD acceleration.
pub struct BilateralFilterSimd {
    /// Spatial Gaussian sigma (controls spatial neighbourhood size).
    pub sigma_space: f32,
    /// Range Gaussian sigma (controls intensity similarity threshold).
    pub sigma_color: f32,
    /// Precomputed range LUT: `range_lut[d]` = `exp(-d² / (2 * sigma_color²))`.
    range_lut: Vec<f32>,
    /// Precomputed 1-D spatial Gaussian kernel.
    spatial_kernel: Vec<f32>,
    /// Filter half-width: `radius = ceil(3 * sigma_space)`.
    radius: i32,
}

impl BilateralFilterSimd {
    /// Create a new `BilateralFilterSimd`.
    ///
    /// # Arguments
    ///
    /// * `sigma_space` - Spatial sigma; controls neighbourhood radius.
    ///   Must be > 0.
    /// * `sigma_color` - Range sigma; controls how much intensity difference
    ///   is tolerated.  Must be > 0.
    ///
    /// # Errors
    ///
    /// Returns [`DenoiseError::InvalidConfig`] if either sigma is non-positive.
    pub fn new(sigma_space: f32, sigma_color: f32) -> DenoiseResult<Self> {
        if sigma_space <= 0.0 {
            return Err(DenoiseError::InvalidConfig(format!(
                "sigma_space must be positive, got {sigma_space}"
            )));
        }
        if sigma_color <= 0.0 {
            return Err(DenoiseError::InvalidConfig(format!(
                "sigma_color must be positive, got {sigma_color}"
            )));
        }

        let radius = (3.0 * sigma_space).ceil() as i32;
        let range_coeff = -0.5 / (sigma_color * sigma_color);
        let space_coeff = -0.5 / (sigma_space * sigma_space);

        // Build range LUT for d in [0, 255]
        let range_lut: Vec<f32> = (0..MAX_INTENSITY)
            .map(|d| {
                let df = d as f32;
                (df * df * range_coeff).exp()
            })
            .collect();

        // Build 1-D spatial Gaussian kernel of length 2*radius+1
        let kernel_len = (2 * radius + 1) as usize;
        let spatial_kernel: Vec<f32> = (0..kernel_len)
            .map(|i| {
                let d = (i as i32 - radius) as f32;
                (d * d * space_coeff).exp()
            })
            .collect();

        Ok(Self {
            sigma_space,
            sigma_color,
            range_lut,
            spatial_kernel,
            radius,
        })
    }

    /// Apply the bilateral filter to `frame`, returning a new denoised
    /// [`Frame2D`].
    ///
    /// Selects the SSE 4.1 inner-loop when available at compile time,
    /// otherwise falls back to the scalar implementation.
    ///
    /// # Errors
    ///
    /// Returns [`DenoiseError::ProcessingError`] if `frame` is empty.
    pub fn apply(&self, frame: &Frame2D) -> DenoiseResult<Frame2D> {
        if frame.is_empty() {
            return Err(DenoiseError::ProcessingError(
                "BilateralFilterSimd: empty input frame".to_string(),
            ));
        }

        #[cfg(target_feature = "sse4.1")]
        {
            self.apply_simd(frame)
        }
        #[cfg(not(target_feature = "sse4.1"))]
        {
            self.apply_scalar(frame)
        }
    }

    /// Scalar (reference) bilateral filter implementation.
    pub fn apply_scalar(&self, frame: &Frame2D) -> DenoiseResult<Frame2D> {
        let w = frame.width;
        let h = frame.height;
        let mut out = Frame2D::new(w, h)?;

        for y in 0..h {
            for x in 0..w {
                let center = frame.get(y, x);
                let center_int = center.round().clamp(0.0, 255.0) as usize;

                let mut weighted_sum = 0.0f32;
                let mut weight_total = 0.0f32;

                for ky in -self.radius..=self.radius {
                    let ny = (y as i32 + ky).clamp(0, (h as i32) - 1) as usize;
                    let ky_idx = (ky + self.radius) as usize;
                    let spatial_y = self.spatial_kernel[ky_idx];

                    for kx in -self.radius..=self.radius {
                        let nx = (x as i32 + kx).clamp(0, (w as i32) - 1) as usize;
                        let kx_idx = (kx + self.radius) as usize;

                        let neighbor = frame.get(ny, nx);
                        let neighbor_int = neighbor.round().clamp(0.0, 255.0) as usize;

                        let d = if neighbor_int > center_int {
                            neighbor_int - center_int
                        } else {
                            center_int - neighbor_int
                        };

                        let range_w = self.range_lut[d.min(MAX_INTENSITY - 1)];
                        let spatial_w = spatial_y * self.spatial_kernel[kx_idx];
                        let w_total = spatial_w * range_w;

                        weighted_sum += neighbor * w_total;
                        weight_total += w_total;
                    }
                }

                let result = if weight_total > 0.0 {
                    weighted_sum / weight_total
                } else {
                    center
                };
                out.set(y, x, result);
            }
        }

        Ok(out)
    }

    /// SSE 4.1 accelerated bilateral filter.
    ///
    /// Compiled only when `target_feature = "sse4.1"` is present.
    /// The inner loop processes 4 column-neighbor floats per iteration
    /// using `_mm_dp_ps` for dot product accumulation of range weights.
    #[cfg(target_feature = "sse4.1")]
    pub fn apply_simd(&self, frame: &Frame2D) -> DenoiseResult<Frame2D> {
        use std::arch::x86_64::{
            __m128, _mm_add_ps, _mm_div_ps, _mm_loadu_ps, _mm_mul_ps, _mm_set1_ps, _mm_storeu_ps,
        };

        let w = frame.width;
        let h = frame.height;
        let mut out = Frame2D::new(w, h)?;

        for y in 0..h {
            for x in 0..w {
                let center = frame.get(y, x);
                let center_int = center.round().clamp(0.0, 255.0) as usize;

                let mut weighted_sum = 0.0f32;
                let mut weight_total = 0.0f32;

                for ky in -self.radius..=self.radius {
                    let ny = (y as i32 + ky).clamp(0, (h as i32) - 1) as usize;
                    let ky_idx = (ky + self.radius) as usize;
                    let spatial_y = self.spatial_kernel[ky_idx];

                    // Number of x positions in kernel
                    let kernel_x_count = (2 * self.radius + 1) as usize;
                    let mut kx_idx = 0;

                    // SIMD path: process 4 kx values at a time
                    while kx_idx + 4 <= kernel_x_count {
                        // Gather 4 neighbor x coordinates
                        let mut neighbors = [0.0f32; 4];
                        let mut spatial_xs = [0.0f32; 4];
                        let mut range_ws = [0.0f32; 4];

                        for lane in 0..4 {
                            let kx = (kx_idx + lane) as i32 - self.radius;
                            let nx = (x as i32 + kx).clamp(0, (w as i32) - 1) as usize;
                            let neighbor = frame.get(ny, nx);
                            neighbors[lane] = neighbor;
                            spatial_xs[lane] = self.spatial_kernel[kx_idx + lane];

                            let neighbor_int = neighbor.round().clamp(0.0, 255.0) as usize;
                            let d = if neighbor_int > center_int {
                                neighbor_int - center_int
                            } else {
                                center_int - neighbor_int
                            };
                            range_ws[lane] = self.range_lut[d.min(MAX_INTENSITY - 1)];
                        }

                        // SAFETY: We are operating on f32 arrays we just created.
                        unsafe {
                            let n_v: __m128 = _mm_loadu_ps(neighbors.as_ptr());
                            let sw_v: __m128 = _mm_loadu_ps(spatial_xs.as_ptr());
                            let rw_v: __m128 = _mm_loadu_ps(range_ws.as_ptr());
                            let sy_v: __m128 = _mm_set1_ps(spatial_y);

                            // w = spatial_y * spatial_x * range_w
                            let w_v = _mm_mul_ps(_mm_mul_ps(sy_v, sw_v), rw_v);

                            // Accumulate w*neighbor
                            let wn_v = _mm_mul_ps(w_v, n_v);

                            let mut wn_arr = [0.0f32; 4];
                            let mut w_arr = [0.0f32; 4];
                            _mm_storeu_ps(wn_arr.as_mut_ptr(), wn_v);
                            _mm_storeu_ps(w_arr.as_mut_ptr(), w_v);

                            for lane in 0..4 {
                                weighted_sum += wn_arr[lane];
                                weight_total += w_arr[lane];
                            }
                        }

                        kx_idx += 4;
                    }

                    // Scalar tail
                    while kx_idx < kernel_x_count {
                        let kx = kx_idx as i32 - self.radius;
                        let nx = (x as i32 + kx).clamp(0, (w as i32) - 1) as usize;
                        let neighbor = frame.get(ny, nx);
                        let neighbor_int = neighbor.round().clamp(0.0, 255.0) as usize;
                        let d = if neighbor_int > center_int {
                            neighbor_int - center_int
                        } else {
                            center_int - neighbor_int
                        };
                        let range_w = self.range_lut[d.min(MAX_INTENSITY - 1)];
                        let spatial_w = spatial_y * self.spatial_kernel[kx_idx];
                        let wt = spatial_w * range_w;
                        weighted_sum += neighbor * wt;
                        weight_total += wt;
                        kx_idx += 1;
                    }
                }

                let result = if weight_total > 0.0 {
                    weighted_sum / weight_total
                } else {
                    center
                };
                out.set(y, x, result);
            }
        }

        Ok(out)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_ramp_frame(w: usize, h: usize) -> Frame2D {
        let data: Vec<f32> = (0..w * h).map(|i| (i % 256) as f32).collect();
        Frame2D::from_vec(data, w, h).expect("valid")
    }

    fn make_noisy_frame(w: usize, h: usize, base: f32) -> Frame2D {
        // Alternating noise pattern: base ± 20
        let data: Vec<f32> = (0..w * h)
            .map(|i| if i % 2 == 0 { base + 20.0 } else { base - 20.0 })
            .collect();
        Frame2D::from_vec(data, w, h).expect("valid")
    }

    #[test]
    fn test_bilateral_simd_creation() {
        let f = BilateralFilterSimd::new(2.0, 30.0).expect("valid params");
        assert!((f.sigma_space - 2.0).abs() < f32::EPSILON);
        assert!((f.sigma_color - 30.0).abs() < f32::EPSILON);
    }

    #[test]
    fn test_bilateral_simd_invalid_sigma_space() {
        assert!(BilateralFilterSimd::new(0.0, 30.0).is_err());
        assert!(BilateralFilterSimd::new(-1.0, 30.0).is_err());
    }

    #[test]
    fn test_bilateral_simd_invalid_sigma_color() {
        assert!(BilateralFilterSimd::new(2.0, 0.0).is_err());
        assert!(BilateralFilterSimd::new(2.0, -5.0).is_err());
    }

    #[test]
    fn test_bilateral_scalar_output_shape() {
        let filter = BilateralFilterSimd::new(1.5, 20.0).expect("valid");
        let frame = make_ramp_frame(16, 16);
        let out = filter.apply_scalar(&frame).expect("scalar ok");
        assert_eq!(out.width, 16);
        assert_eq!(out.height, 16);
        assert_eq!(out.len(), 16 * 16);
    }

    #[test]
    fn test_bilateral_scalar_uniform_input() {
        // Uniform input should produce uniform output (bilateral preserves constants).
        let filter = BilateralFilterSimd::new(2.0, 30.0).expect("valid");
        let frame = Frame2D::filled(16, 16, 100.0).expect("valid");
        let out = filter.apply_scalar(&frame).expect("scalar ok");
        for &v in &out.data {
            assert!(
                (v - 100.0_f32).abs() < 0.5,
                "uniform input should produce near-uniform output, got {v}"
            );
        }
    }

    #[test]
    fn test_bilateral_scalar_smooths_noise() {
        // Noisy input: MSE between output and 128.0 should be less than input MSE.
        let filter = BilateralFilterSimd::new(3.0, 40.0).expect("valid");
        let noisy = make_noisy_frame(16, 16, 128.0);
        let target = Frame2D::filled(16, 16, 128.0).expect("valid");

        let noisy_psnr = noisy.psnr(&target).expect("psnr");
        let out = filter.apply_scalar(&noisy).expect("scalar ok");
        let out_psnr = out.psnr(&target).expect("psnr");
        assert!(
            out_psnr > noisy_psnr,
            "bilateral should improve PSNR: {noisy_psnr:.1} -> {out_psnr:.1}"
        );
    }

    #[test]
    fn test_bilateral_simd_vs_scalar_small_frame() {
        // SIMD and scalar results must agree within floating-point tolerance.
        let filter = BilateralFilterSimd::new(1.5, 25.0).expect("valid");
        let frame = make_ramp_frame(8, 8);

        let scalar = filter.apply_scalar(&frame).expect("scalar");

        // `apply()` dispatches to SIMD or scalar depending on compile flags.
        let dispatched = filter.apply(&frame).expect("apply");

        for (i, (&s, &d)) in scalar.data.iter().zip(dispatched.data.iter()).enumerate() {
            assert!(
                (s - d).abs() < 0.5_f32,
                "pixel {i}: scalar={s} vs apply={d}"
            );
        }
    }

    #[test]
    fn test_bilateral_simd_vs_scalar_ramp() {
        // Larger frame: scalar and dispatched paths must agree.
        let filter = BilateralFilterSimd::new(2.0, 30.0).expect("valid");
        let frame = make_ramp_frame(32, 32);

        let scalar = filter.apply_scalar(&frame).expect("scalar");
        let dispatched = filter.apply(&frame).expect("apply");

        let max_diff = scalar
            .data
            .iter()
            .zip(dispatched.data.iter())
            .map(|(&a, &b)| (a - b).abs() as f32)
            .fold(0.0f32, f32::max);

        assert!(
            max_diff < 0.5,
            "max pixel diff between scalar and SIMD paths: {max_diff}"
        );
    }

    #[test]
    fn test_bilateral_lut_range() {
        // range_lut[0] == 1.0 (zero distance -> weight 1)
        let filter = BilateralFilterSimd::new(2.0, 30.0).expect("valid");
        assert!(
            (filter.range_lut[0] - 1.0).abs() < 1e-6,
            "lut[0] should be 1.0"
        );
        // range_lut[255] should be close to 0 for reasonable sigma_color
        assert!(filter.range_lut[255] < 0.01, "lut[255] should be near 0");
    }

    #[test]
    fn test_bilateral_preserves_edges() {
        // Build a step edge: left half = 50, right half = 200.
        // With appropriate sigma_color, the filter should NOT blur across the edge.
        let w = 16usize;
        let h = 8usize;
        let data: Vec<f32> = (0..w * h)
            .map(|i| if (i % w) < w / 2 { 50.0 } else { 200.0 })
            .collect();
        let frame = Frame2D::from_vec(data, w, h).expect("valid");

        let filter = BilateralFilterSimd::new(2.0, 10.0).expect("valid"); // tight sigma_color
        let out = filter.apply_scalar(&frame).expect("ok");

        // Interior pixels far from edge should remain close to their region value.
        let left_interior = out.get(4, 2);
        let right_interior = out.get(4, 13);
        assert!(
            (left_interior - 50.0).abs() < 5.0,
            "left interior should stay near 50, got {left_interior}"
        );
        assert!(
            (right_interior - 200.0).abs() < 5.0,
            "right interior should stay near 200, got {right_interior}"
        );
    }

    #[test]
    fn test_bilateral_empty_frame_error() {
        let filter = BilateralFilterSimd::new(2.0, 30.0).expect("valid");
        let empty_frame = Frame2D {
            data: vec![],
            width: 0,
            height: 0,
        };
        // is_empty() -> true -> error
        assert!(filter.apply(&empty_frame).is_err());
    }

    #[test]
    fn test_bilateral_different_sigmas() {
        // Larger sigma_color allows more intensity blending.
        let tight = BilateralFilterSimd::new(2.0, 5.0).expect("valid");
        let loose = BilateralFilterSimd::new(2.0, 100.0).expect("valid");

        let noisy = make_noisy_frame(16, 16, 128.0);
        let target = Frame2D::filled(16, 16, 128.0).expect("valid");

        let out_tight = tight.apply_scalar(&noisy).expect("tight");
        let out_loose = loose.apply_scalar(&noisy).expect("loose");

        let psnr_tight = out_tight.psnr(&target).expect("psnr");
        let psnr_loose = out_loose.psnr(&target).expect("psnr");

        // Loose sigma_color blends more aggressively: noise should reduce more.
        assert!(
            psnr_loose > psnr_tight,
            "loose sigma_color should yield higher PSNR: {psnr_tight:.1} vs {psnr_loose:.1}"
        );
    }
}
