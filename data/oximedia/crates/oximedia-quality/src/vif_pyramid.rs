//! FFT-based Gaussian scale-space pyramid for Visual Information Fidelity.
//!
//! Implements the Sheikh-Bovik 2006 IEEE TIP VIF metric using a frequency-domain
//! Gaussian bandpass pyramid instead of the naive block-wise spatial statistics
//! used in the original implementation.
//!
//! # Algorithm
//!
//! For each scale `s` in `0..num_scales`:
//! 1. Forward `fft2d` of the luma plane (real → complex, imag = 0) into the
//!    frequency domain.
//! 2. Multiply each coefficient by the radial Gaussian bandpass mask
//!    `H_s(ω) = exp(-‖ω‖² / (2·σ_s²))` where `σ_s = sigma_base * 2^s` doubles
//!    per octave and `ω = (kx/W, ky/H)` are normalised frequency coordinates.
//! 3. `ifft2d` → take real part → `subband_ref`, `subband_dist`.
//! 4. Compute per-pixel local statistics via a separable 1-D Gaussian convolution
//!    (σ = `gauss_kernel_sigma`, typically 1.5) over each subband.
//! 5. For each pixel apply the GSM (Gaussian Scale Mixture) formula from the
//!    paper and accumulate numerator/denominator information values.
//! 6. `vif = num / den ∈ [0, 1]`
//!
//! # Reference
//!
//! H.R. Sheikh and A.C. Bovik, "Image Information and Visual Quality",
//! *IEEE Transactions on Image Processing*, 15(2), 2006.

use oxifft::{fft2d, ifft2d, Complex};
use oximedia_core::OxiResult;

use crate::{Frame, MetricType, QualityScore};

// ---------------------------------------------------------------------------
// Public types
// ---------------------------------------------------------------------------

/// FFT-based pyramid VIF calculator faithful to Sheikh-Bovik (2006).
///
/// Replaces the naive block-wise spatial-statistics path with a
/// frequency-domain Gaussian scale-space pyramid.
#[derive(Debug, Clone)]
pub struct PyramidVifCalculator {
    /// Number of pyramid scales (subbands). Default 4.
    pub num_scales: usize,
    /// Noise variance σ²_n (HVS noise floor). Default 2.0.
    pub sigma_nsq: f64,
    /// Base bandpass width σ₀ (doubled per octave). Default 0.5.
    pub sigma_base: f64,
    /// σ for local-statistics Gaussian convolution. Default 1.5.
    pub gauss_kernel_sigma: f64,
}

impl Default for PyramidVifCalculator {
    fn default() -> Self {
        Self::new()
    }
}

impl PyramidVifCalculator {
    /// Creates a `PyramidVifCalculator` with Sheikh-Bovik (2006) defaults.
    #[must_use]
    pub fn new() -> Self {
        Self {
            num_scales: 4,
            sigma_nsq: 2.0,
            sigma_base: 0.5,
            gauss_kernel_sigma: 1.5,
        }
    }

    /// Computes pyramid VIF between `reference` and `distorted` frames.
    ///
    /// # Errors
    ///
    /// Returns an error if the frame dimensions do not match.
    pub fn calculate(&self, reference: &Frame, distorted: &Frame) -> OxiResult<QualityScore> {
        if reference.width != distorted.width || reference.height != distorted.height {
            return Err(oximedia_core::OxiError::InvalidData(
                "Frame dimensions must match for VIF".to_string(),
            ));
        }

        let y_vif = self.calculate_plane(
            &reference.planes[0],
            &distorted.planes[0],
            reference.width,
            reference.height,
        )?;

        let mut score = QualityScore::new(MetricType::Vif, y_vif);
        score.add_component("Y", y_vif);
        Ok(score)
    }

    // -----------------------------------------------------------------------
    // Private helpers
    // -----------------------------------------------------------------------

    /// Runs the multi-scale pyramid VIF on a single u8 luma plane.
    fn calculate_plane(
        &self,
        ref_plane: &[u8],
        dist_plane: &[u8],
        width: usize,
        height: usize,
    ) -> OxiResult<f64> {
        let ref_f64: Vec<f64> = ref_plane.iter().map(|&v| f64::from(v)).collect();
        let dist_f64: Vec<f64> = dist_plane.iter().map(|&v| f64::from(v)).collect();

        let mut numerator = 0.0_f64;
        let mut denominator = 0.0_f64;

        for scale in 0..self.num_scales {
            let sigma_s = self.sigma_base * (1u64 << scale) as f64;

            let subband_ref = gaussian_bandpass_filter(&ref_f64, width, height, sigma_s)?;
            let subband_dist = gaussian_bandpass_filter(&dist_f64, width, height, sigma_s)?;

            let (num, den) = self.compute_gsm_vif(&subband_ref, &subband_dist, width, height);
            numerator += num;
            denominator += den;
        }

        let vif = if denominator > 1e-10 {
            (numerator / denominator).clamp(0.0, 1.0)
        } else {
            1.0
        };
        Ok(vif)
    }

    /// Applies the per-pixel GSM VIF formula over a single subband pair.
    ///
    /// Returns `(numerator_info, denominator_info)`.
    fn compute_gsm_vif(
        &self,
        subband_ref: &[f64],
        subband_dist: &[f64],
        width: usize,
        height: usize,
    ) -> (f64, f64) {
        // Local statistics via separable Gaussian convolution
        let mu_ref = separable_gaussian_conv(subband_ref, width, height, self.gauss_kernel_sigma);
        let mu_dist = separable_gaussian_conv(subband_dist, width, height, self.gauss_kernel_sigma);

        // Compute residuals (remove mean)
        let n = width * height;
        let res_ref: Vec<f64> = (0..n).map(|i| subband_ref[i] - mu_ref[i]).collect();
        let res_dist: Vec<f64> = (0..n).map(|i| subband_dist[i] - mu_dist[i]).collect();

        // Squared residuals for local variance estimation
        let res_ref_sq: Vec<f64> = res_ref.iter().map(|&v| v * v).collect();
        let res_dist_sq: Vec<f64> = res_dist.iter().map(|&v| v * v).collect();
        let res_cross: Vec<f64> = res_ref
            .iter()
            .zip(res_dist.iter())
            .map(|(&r, &d)| r * d)
            .collect();

        let sigma_ref_sq_map =
            separable_gaussian_conv(&res_ref_sq, width, height, self.gauss_kernel_sigma);
        let sigma_dist_sq_map =
            separable_gaussian_conv(&res_dist_sq, width, height, self.gauss_kernel_sigma);
        let sigma_cross_map =
            separable_gaussian_conv(&res_cross, width, height, self.gauss_kernel_sigma);

        let mut num = 0.0_f64;
        let mut den = 0.0_f64;

        for i in 0..n {
            let sigma_ref_sq = sigma_ref_sq_map[i].max(0.0);
            let sigma_dist_sq = sigma_dist_sq_map[i].max(0.0);
            let sigma_cross = sigma_cross_map[i];

            if sigma_ref_sq < 1e-10 {
                continue;
            }

            // GSM gain and distortion-channel noise variance
            let g = sigma_cross / sigma_ref_sq;
            let sv_sq = (sigma_dist_sq - g * g * sigma_ref_sq).max(0.0);

            if sigma_ref_sq > self.sigma_nsq {
                // I(ref; dist_channel) using mutual information formula
                let arg_num = 1.0 + g * g * sigma_ref_sq / (sv_sq + self.sigma_nsq);
                let arg_den = 1.0 + sigma_ref_sq / self.sigma_nsq;
                num += arg_num.max(1.0).log2();
                den += arg_den.log2();
            }
        }
        (num, den)
    }
}

// ---------------------------------------------------------------------------
// Signal processing primitives
// ---------------------------------------------------------------------------

/// Applies a radial Gaussian bandpass filter in the frequency domain.
///
/// Steps:
/// 1. Promote `data` to `Vec<Complex<f32>>` (real = pixel, imag = 0).
/// 2. Forward `fft2d(&buf, height, width)` — note row-major convention.
/// 3. Multiply every coefficient by the Gaussian envelope
///    `exp(-(kx_norm² + ky_norm²) / (2·sigma²))`.
/// 4. Inverse `ifft2d(&buf, height, width)`.
/// 5. Return real parts as `Vec<f64>`.
///
/// # Errors
///
/// Returns `OxiError::InvalidData` if `data.len() != width * height`.
pub fn gaussian_bandpass_filter(
    data: &[f64],
    width: usize,
    height: usize,
    sigma: f64,
) -> OxiResult<Vec<f64>> {
    let n = width * height;
    if data.len() != n {
        return Err(oximedia_core::OxiError::InvalidData(format!(
            "gaussian_bandpass_filter: data.len()={} != width*height={}",
            data.len(),
            n
        )));
    }

    // Step 1: real → complex
    let mut buf: Vec<Complex<f32>> = data
        .iter()
        .map(|&v| Complex::new(v as f32, 0.0_f32))
        .collect();

    // Step 2: forward 2-D FFT (n0=height rows, n1=width cols — row-major)
    let spectrum = fft2d(&buf, height, width);
    buf.copy_from_slice(&spectrum);

    // Step 3: multiply by radial Gaussian bandpass mask
    let two_sigma_sq = 2.0 * sigma * sigma;
    for ky in 0..height {
        let ky_norm = ky as f64 / height as f64;
        for kx in 0..width {
            let kx_norm = kx as f64 / width as f64;
            let exponent = (kx_norm * kx_norm + ky_norm * ky_norm) / two_sigma_sq;
            let gain = (-exponent).exp() as f32;
            let idx = ky * width + kx;
            buf[idx] = Complex::new(buf[idx].re * gain, buf[idx].im * gain);
        }
    }

    // Step 4: inverse 2-D FFT
    let recovered = ifft2d(&buf, height, width);

    // Step 5: extract real parts
    Ok(recovered.iter().map(|c| c.re as f64).collect())
}

/// Applies a separable 1-D Gaussian convolution horizontally then vertically.
///
/// Kernel radius = `(3.0 * sigma).ceil() as usize`; boundary: clamp-to-edge.
/// This gives the per-pixel local variance weighting used in the GSM model.
#[must_use]
pub fn separable_gaussian_conv(data: &[f64], width: usize, height: usize, sigma: f64) -> Vec<f64> {
    let kernel = build_gaussian_kernel(sigma);
    let radius = kernel.len() / 2;

    // Horizontal pass
    let mut horiz = vec![0.0_f64; width * height];
    for y in 0..height {
        for x in 0..width {
            let mut acc = 0.0_f64;
            let mut weight_sum = 0.0_f64;
            for (k_idx, &k_val) in kernel.iter().enumerate() {
                let src_x = x as isize + k_idx as isize - radius as isize;
                let clamped = src_x.clamp(0, width as isize - 1) as usize;
                acc += k_val * data[y * width + clamped];
                weight_sum += k_val;
            }
            horiz[y * width + x] = if weight_sum > 1e-15 {
                acc / weight_sum
            } else {
                data[y * width + x]
            };
        }
    }

    // Vertical pass
    let mut result = vec![0.0_f64; width * height];
    for y in 0..height {
        for x in 0..width {
            let mut acc = 0.0_f64;
            let mut weight_sum = 0.0_f64;
            for (k_idx, &k_val) in kernel.iter().enumerate() {
                let src_y = y as isize + k_idx as isize - radius as isize;
                let clamped = src_y.clamp(0, height as isize - 1) as usize;
                acc += k_val * horiz[clamped * width + x];
                weight_sum += k_val;
            }
            result[y * width + x] = if weight_sum > 1e-15 {
                acc / weight_sum
            } else {
                horiz[y * width + x]
            };
        }
    }
    result
}

/// Builds a 1-D Gaussian kernel with `sigma`.
///
/// Kernel length = `2 * radius + 1` where `radius = (3.0 * sigma).ceil()`.
/// Values are un-normalised `exp(-x² / (2·σ²))`; callers normalise by dividing
/// by the running `weight_sum`.
fn build_gaussian_kernel(sigma: f64) -> Vec<f64> {
    if sigma < 1e-10 {
        return vec![1.0];
    }
    let radius = (3.0 * sigma).ceil() as usize;
    let len = 2 * radius + 1;
    let two_sigma_sq = 2.0 * sigma * sigma;
    (0..len)
        .map(|i| {
            let x = i as f64 - radius as f64;
            (-x * x / two_sigma_sq).exp()
        })
        .collect()
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use oximedia_core::PixelFormat;

    // -----------------------------------------------------------------------
    // Helpers
    // -----------------------------------------------------------------------

    fn solid_frame(width: usize, height: usize, value: u8) -> Frame {
        let mut f = Frame::new(width, height, PixelFormat::Yuv420p)
            .expect("Frame::new should succeed in tests");
        f.planes[0].fill(value);
        f
    }

    fn gradient_frame(width: usize, height: usize) -> Frame {
        let mut f = Frame::new(width, height, PixelFormat::Yuv420p)
            .expect("Frame::new should succeed in tests");
        for (i, byte) in f.planes[0].iter_mut().enumerate() {
            *byte = (i % 256) as u8;
        }
        f
    }

    /// Create a frame that is a box-blurred version of `src`.
    fn box_blur(src: &Frame, radius: usize) -> Frame {
        let w = src.width;
        let h = src.height;
        let plane = &src.planes[0];
        let mut out =
            Frame::new(w, h, PixelFormat::Yuv420p).expect("Frame::new should succeed in tests");
        for y in 0..h {
            for x in 0..w {
                let mut acc = 0u32;
                let mut cnt = 0u32;
                let y0 = y.saturating_sub(radius);
                let y1 = (y + radius).min(h - 1);
                let x0 = x.saturating_sub(radius);
                let x1 = (x + radius).min(w - 1);
                for ry in y0..=y1 {
                    for rx in x0..=x1 {
                        acc += u32::from(plane[ry * w + rx]);
                        cnt += 1;
                    }
                }
                out.planes[0][y * w + x] = (acc / cnt) as u8;
            }
        }
        out
    }

    // -----------------------------------------------------------------------
    // Tests
    // -----------------------------------------------------------------------

    /// VIF of identical frames should be 1.0 (no information loss).
    #[test]
    fn test_vif_identical_frames() {
        let calc = PyramidVifCalculator::new();
        let ref_frame = gradient_frame(64, 64);
        let dist_frame = gradient_frame(64, 64);
        let result = calc
            .calculate(&ref_frame, &dist_frame)
            .expect("VIF should succeed in tests");
        // Identical images → numerator ≈ denominator → score ≈ 1.0
        assert!(
            (result.score - 1.0).abs() < 0.05,
            "identical frames: expected ~1.0, got {}",
            result.score
        );
    }

    /// VIF with heavy blur < VIF with light blur (monotonic in distortion).
    ///
    /// Uses a checkerboard pattern that has significant high-frequency energy
    /// so that blurring meaningfully degrades the VIF signal at each scale.
    #[test]
    fn test_vif_monotonic_blur() {
        let calc = PyramidVifCalculator::new();
        let w = 64usize;
        let h = 64usize;

        // Checkerboard has high-frequency energy — blurring clearly reduces VIF
        let mut reference =
            Frame::new(w, h, PixelFormat::Yuv420p).expect("Frame::new should succeed in tests");
        for y in 0..h {
            for x in 0..w {
                reference.planes[0][y * w + x] = if (x ^ y) & 1 == 0 { 220 } else { 36 };
            }
        }

        let light_blur = box_blur(&reference, 1);
        let heavy_blur = box_blur(&reference, 4);

        let score_light = calc
            .calculate(&reference, &light_blur)
            .expect("VIF should succeed in tests")
            .score;
        let score_heavy = calc
            .calculate(&reference, &heavy_blur)
            .expect("VIF should succeed in tests")
            .score;

        // Both should be valid VIF scores
        assert!(
            score_light >= 0.0 && score_light <= 1.0,
            "light blur score out of range: {score_light}"
        );
        assert!(
            score_heavy >= 0.0 && score_heavy <= 1.0,
            "heavy blur score out of range: {score_heavy}"
        );

        // Heavy blur must degrade quality more than light blur
        assert!(
            score_heavy <= score_light,
            "heavy blur score ({score_heavy}) should be <= light blur ({score_light})"
        );
    }

    /// VIF must lie in [0.0, 1.0] on random-noise 32×32 frames.
    #[test]
    fn test_vif_range() {
        let calc = PyramidVifCalculator::new();
        // Use pseudo-random data via a simple LCG to stay dependency-free
        let mut lcg = 6364136223846793005u64;
        let mut next = |lo: u8, hi: u8| -> u8 {
            lcg = lcg.wrapping_mul(6364136223846793005).wrapping_add(1);
            let range = (hi as u16) - (lo as u16) + 1;
            lo + ((((lcg >> 33) as u16) % range) as u8)
        };

        let w = 32;
        let h = 32;
        let mut ref_f =
            Frame::new(w, h, PixelFormat::Yuv420p).expect("Frame::new should succeed in tests");
        let mut dist_f =
            Frame::new(w, h, PixelFormat::Yuv420p).expect("Frame::new should succeed in tests");
        for byte in &mut ref_f.planes[0] {
            *byte = next(0, 255);
        }
        for byte in &mut dist_f.planes[0] {
            *byte = next(0, 255);
        }

        let result = calc
            .calculate(&ref_f, &dist_f)
            .expect("VIF should succeed in tests");
        assert!(
            result.score >= 0.0 && result.score <= 1.0,
            "VIF out of range: {}",
            result.score
        );
    }

    /// Gaussian bandpass filter must not amplify subband energy (gain ≤ 1).
    #[test]
    fn test_pyramid_subband_energy() {
        let w = 32usize;
        let h = 32usize;
        let data: Vec<f64> = (0..w * h).map(|i| (i % 256) as f64).collect();

        let input_energy: f64 = data.iter().map(|&v| v * v).sum();

        let filtered = gaussian_bandpass_filter(&data, w, h, 0.5)
            .expect("gaussian_bandpass_filter should succeed in tests");
        let output_energy: f64 = filtered.iter().map(|&v| v * v).sum();

        assert!(
            output_energy <= input_energy * 1.001, // tiny float tolerance
            "subband energy ({output_energy}) should not exceed input energy ({input_energy})"
        );
    }

    /// Separable Gaussian convolution output has the same length as input.
    #[test]
    fn test_separable_gaussian_conv_length() {
        let w = 16usize;
        let h = 16usize;
        let data = vec![128.0_f64; w * h];
        let result = separable_gaussian_conv(&data, w, h, 1.5);
        assert_eq!(result.len(), w * h);
    }

    /// Separable Gaussian convolution on a constant field preserves the value.
    #[test]
    fn test_separable_gaussian_conv_constant() {
        let w = 20usize;
        let h = 20usize;
        let val = 42.0_f64;
        let data = vec![val; w * h];
        let result = separable_gaussian_conv(&data, w, h, 1.5);
        for &v in &result {
            assert!(
                (v - val).abs() < 1e-9,
                "constant field should be preserved; got {v}"
            );
        }
    }

    /// Dimension mismatch must return an error.
    #[test]
    fn test_vif_dimension_mismatch() {
        let calc = PyramidVifCalculator::new();
        let ref_frame =
            Frame::new(64, 64, PixelFormat::Yuv420p).expect("Frame::new should succeed in tests");
        let dist_frame =
            Frame::new(32, 32, PixelFormat::Yuv420p).expect("Frame::new should succeed in tests");
        assert!(
            calc.calculate(&ref_frame, &dist_frame).is_err(),
            "mismatched dimensions must return Err"
        );
    }
}
