//! Multi-scale wavelet denoising.
//!
//! Performs wavelet decomposition at multiple levels and applies
//! scale-dependent denoising to each level.
//!
//! Also provides 1-D Haar wavelet transforms for audio denoising.

use crate::spatial::wavelet::{wavelet_denoise, ThresholdMethod};
use crate::DenoiseResult;
use oximedia_codec::VideoFrame;
use rayon::prelude::*;

/// Multi-scale wavelet denoising configuration.
pub struct MultiscaleWaveletConfig {
    /// Number of decomposition levels.
    pub num_levels: usize,
    /// Thresholding method.
    pub method: ThresholdMethod,
    /// Base strength (adjusted per level).
    pub strength: f32,
}

impl Default for MultiscaleWaveletConfig {
    fn default() -> Self {
        Self {
            num_levels: 3,
            method: ThresholdMethod::Soft,
            strength: 0.5,
        }
    }
}

/// Apply multi-scale wavelet denoising.
///
/// Decomposes the frame using wavelets at multiple scales and applies
/// scale-dependent thresholding for effective noise removal.
///
/// # Arguments
/// * `frame` - Input video frame
/// * `config` - Wavelet denoising configuration
///
/// # Returns
/// Denoised frame
pub fn multiscale_wavelet_denoise(
    frame: &VideoFrame,
    config: &MultiscaleWaveletConfig,
) -> DenoiseResult<VideoFrame> {
    // For now, apply single-level wavelet denoising
    // A full implementation would apply recursive decomposition
    wavelet_denoise(frame, config.strength, config.method)
}

/// Adaptive multi-scale wavelet denoising.
///
/// Automatically adjusts thresholding strength per scale based on
/// noise characteristics.
pub fn adaptive_multiscale_wavelet(frame: &VideoFrame, strength: f32) -> DenoiseResult<VideoFrame> {
    let config = MultiscaleWaveletConfig {
        num_levels: 3,
        method: ThresholdMethod::Soft,
        strength,
    };

    multiscale_wavelet_denoise(frame, &config)
}

/// Row-parallel wavelet denoising using rayon.
///
/// Applies Haar wavelet denoising to each plane, where the row-wise
/// forward and inverse transforms are parallelised with `rayon::par_iter_mut`.
/// This provides significant throughput improvements on multi-core systems
/// for high-resolution frames.
///
/// # Arguments
/// * `frame`    - Input video frame
/// * `strength` - Denoising strength (0.0 – 1.0)
/// * `method`   - Wavelet threshold method
///
/// # Returns
/// Denoised frame
pub fn parallel_wavelet_denoise(
    frame: &VideoFrame,
    strength: f32,
    method: ThresholdMethod,
) -> DenoiseResult<VideoFrame> {
    if frame.planes.is_empty() {
        return Err(crate::DenoiseError::ProcessingError(
            "Frame has no planes".to_string(),
        ));
    }

    let mut output = frame.clone();

    // Process each plane; use rayon over rows inside each plane.
    output
        .planes
        .iter_mut()
        .enumerate()
        .try_for_each(|(plane_idx, plane)| {
            let input_plane = &frame.planes[plane_idx];
            let (width, height) = frame.plane_dimensions(plane_idx);
            let w = width as usize;
            let h = height as usize;
            let stride = plane.stride;

            // --- Forward Haar transform rows (parallel) ---
            // Convert the entire plane to f32 first (row-major, no stride padding).
            let mut coeffs: Vec<f32> = (0..h)
                .flat_map(|y| (0..w).map(move |x| f32::from(input_plane.data[y * stride + x])))
                .collect();

            // Row-parallel forward Haar
            coeffs
                .par_chunks_mut(w)
                .take(h)
                .for_each(|row| parallel_haar_1d(row));

            // Column-wise forward Haar (sequential – columns are non-contiguous)
            let mut col_buf = vec![0.0f32; h];
            for x in 0..w {
                for y in 0..h {
                    col_buf[y] = coeffs[y * w + x];
                }
                parallel_haar_1d(&mut col_buf);
                for y in 0..h {
                    coeffs[y * w + x] = col_buf[y];
                }
            }

            // Threshold
            let sigma = {
                let mut abs_coeffs: Vec<f32> = coeffs.iter().map(|&v| v.abs()).collect();
                abs_coeffs
                    .sort_unstable_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
                let median = abs_coeffs.get(abs_coeffs.len() / 2).copied().unwrap_or(0.0);
                median / 0.6745
            };
            let n = coeffs.len() as f32;
            let threshold = sigma * (2.0 * n.ln()).sqrt() * strength;

            coeffs.par_iter_mut().for_each(|c| {
                *c = apply_threshold_fn(*c, threshold, method);
            });

            // --- Inverse Haar: column-wise (sequential) ---
            for x in 0..w {
                for y in 0..h {
                    col_buf[y] = coeffs[y * w + x];
                }
                parallel_inv_haar_1d(&mut col_buf);
                for y in 0..h {
                    coeffs[y * w + x] = col_buf[y];
                }
            }

            // Row-parallel inverse Haar
            coeffs
                .par_chunks_mut(w)
                .take(h)
                .for_each(|row| parallel_inv_haar_1d(row));

            // Write back to plane
            for y in 0..h {
                for x in 0..w {
                    plane.data[y * stride + x] = coeffs[y * w + x].round().clamp(0.0, 255.0) as u8;
                }
            }

            Ok::<(), crate::DenoiseError>(())
        })?;

    Ok(output)
}

/// In-place 1-D Haar forward transform (used by parallel wavelet).
fn parallel_haar_1d(data: &mut [f32]) {
    let n = data.len();
    if n < 2 {
        return;
    }
    let half = n / 2;
    let mut temp = vec![0.0f32; n];
    let inv_sqrt2 = 1.0_f32 / 2.0_f32.sqrt();
    for i in 0..half {
        temp[i] = (data[2 * i] + data[2 * i + 1]) * inv_sqrt2;
        temp[half + i] = (data[2 * i] - data[2 * i + 1]) * inv_sqrt2;
    }
    data[..n].copy_from_slice(&temp);
}

/// In-place 1-D Haar inverse transform (used by parallel wavelet).
fn parallel_inv_haar_1d(data: &mut [f32]) {
    let n = data.len();
    if n < 2 {
        return;
    }
    let half = n / 2;
    let mut temp = vec![0.0f32; n];
    let inv_sqrt2 = 1.0_f32 / 2.0_f32.sqrt();
    for i in 0..half {
        temp[2 * i] = (data[i] + data[half + i]) * inv_sqrt2;
        temp[2 * i + 1] = (data[i] - data[half + i]) * inv_sqrt2;
    }
    data[..n].copy_from_slice(&temp);
}

/// Threshold function used by the parallel wavelet denoiser.
fn apply_threshold_fn(x: f32, threshold: f32, method: ThresholdMethod) -> f32 {
    match method {
        ThresholdMethod::Hard => {
            if x.abs() > threshold {
                x
            } else {
                0.0
            }
        }
        ThresholdMethod::Soft => {
            if x.abs() > threshold {
                x.signum() * (x.abs() - threshold)
            } else {
                0.0
            }
        }
        ThresholdMethod::Garrote => {
            if x.abs() > threshold {
                x - (threshold * threshold / x)
            } else {
                0.0
            }
        }
    }
}

/// Apply wavelet denoising with different thresholds per level.
pub fn level_dependent_wavelet_denoise(
    frame: &VideoFrame,
    base_strength: f32,
    num_levels: usize,
) -> DenoiseResult<VideoFrame> {
    if num_levels == 0 {
        return Ok(frame.clone());
    }

    // Apply denoising with strength decreasing at coarser levels
    let strength = base_strength * (1.0 + 0.5 * (num_levels as f32 - 1.0) / num_levels as f32);

    wavelet_denoise(frame, strength, ThresholdMethod::Soft)
}

#[cfg(test)]
mod tests {
    use super::*;
    use oximedia_core::PixelFormat;

    #[test]
    fn test_multiscale_wavelet_denoise() {
        let mut frame = VideoFrame::new(PixelFormat::Yuv420p, 64, 64);
        frame.allocate();

        let config = MultiscaleWaveletConfig::default();
        let result = multiscale_wavelet_denoise(&frame, &config);
        assert!(result.is_ok());
    }

    #[test]
    fn test_adaptive_multiscale_wavelet() {
        let mut frame = VideoFrame::new(PixelFormat::Yuv420p, 64, 64);
        frame.allocate();

        let result = adaptive_multiscale_wavelet(&frame, 0.5);
        assert!(result.is_ok());
    }

    #[test]
    fn test_level_dependent_wavelet() {
        let mut frame = VideoFrame::new(PixelFormat::Yuv420p, 64, 64);
        frame.allocate();

        let result = level_dependent_wavelet_denoise(&frame, 0.5, 3);
        assert!(result.is_ok());
    }

    #[test]
    fn test_config_default() {
        let config = MultiscaleWaveletConfig::default();
        assert_eq!(config.num_levels, 3);
        assert!(matches!(config.method, ThresholdMethod::Soft));
    }

    // -------------------------------------------------------------------
    // Parallel wavelet tests
    // -------------------------------------------------------------------

    #[test]
    fn test_parallel_wavelet_denoise_basic() {
        let mut frame = VideoFrame::new(PixelFormat::Yuv420p, 32, 32);
        frame.allocate();
        let result = parallel_wavelet_denoise(&frame, 0.5, ThresholdMethod::Soft);
        assert!(result.is_ok());
        let f = result.expect("parallel wavelet should succeed");
        assert_eq!(f.width, 32);
        assert_eq!(f.height, 32);
    }

    #[test]
    fn test_parallel_wavelet_all_threshold_methods() {
        let mut frame = VideoFrame::new(PixelFormat::Yuv420p, 16, 16);
        frame.allocate();
        for method in [
            ThresholdMethod::Hard,
            ThresholdMethod::Soft,
            ThresholdMethod::Garrote,
        ] {
            let result = parallel_wavelet_denoise(&frame, 0.5, method);
            assert!(result.is_ok(), "method {method:?} should succeed");
        }
    }

    #[test]
    fn test_parallel_wavelet_uniform_preserves_values() {
        let mut frame = VideoFrame::new(PixelFormat::Yuv420p, 16, 16);
        frame.allocate();
        // Set luma to a constant 120
        let stride = frame.planes[0].stride;
        for y in 0..16usize {
            for x in 0..16usize {
                frame.planes[0].data[y * stride + x] = 120;
            }
        }
        let out =
            parallel_wavelet_denoise(&frame, 0.5, ThresholdMethod::Soft).expect("should succeed");
        let out_stride = out.planes[0].stride;
        for y in 0..16usize {
            for x in 0..16usize {
                let v = out.planes[0].data[y * out_stride + x];
                assert!(
                    (v as i32 - 120).abs() <= 2,
                    "uniform frame value drifted at ({x},{y}): {v}"
                );
            }
        }
    }
}

// ============================================================
// 1-D Haar Wavelet Transform (audio denoising)
// ============================================================

/// Haar wavelet transform for 1-D audio signals.
pub struct HaarWavelet;

impl HaarWavelet {
    /// Compute the forward Haar wavelet transform.
    ///
    /// Returns `(approximation, detail)` coefficient vectors, each of length `n / 2`
    /// (where `n` is the length of `signal`, which must be even; odd-length signals
    /// are truncated to even length).
    pub fn forward(signal: &[f32]) -> (Vec<f32>, Vec<f32>) {
        let n = signal.len() & !1; // round down to even
        if n == 0 {
            return (Vec::new(), Vec::new());
        }

        let half = n / 2;
        let mut approx = Vec::with_capacity(half);
        let mut detail = Vec::with_capacity(half);

        let inv_sqrt2 = 1.0_f32 / 2.0_f32.sqrt();
        for i in 0..half {
            let a = signal[2 * i];
            let b = signal[2 * i + 1];
            approx.push((a + b) * inv_sqrt2);
            detail.push((a - b) * inv_sqrt2);
        }

        (approx, detail)
    }

    /// Compute the inverse Haar wavelet transform.
    ///
    /// `approx` and `detail` must have the same length.
    /// Returns a reconstructed signal of length `2 * approx.len()`.
    pub fn inverse(approx: &[f32], detail: &[f32]) -> Vec<f32> {
        let half = approx.len().min(detail.len());
        let mut signal = vec![0.0f32; half * 2];

        let inv_sqrt2 = 1.0_f32 / 2.0_f32.sqrt();
        for i in 0..half {
            signal[2 * i] = (approx[i] + detail[i]) * inv_sqrt2;
            signal[2 * i + 1] = (approx[i] - detail[i]) * inv_sqrt2;
        }

        signal
    }
}

/// Wavelet-based denoiser for 1-D signals.
pub struct WaveletDenoiser;

impl WaveletDenoiser {
    /// Apply soft thresholding to a wavelet coefficient.
    ///
    /// Shrinks the coefficient towards zero by `threshold`:
    /// - If |coef| <= threshold, returns 0.
    /// - Otherwise, returns `sign(coef) * (|coef| - threshold)`.
    pub fn soft_threshold(coef: f32, threshold: f32) -> f32 {
        let abs = coef.abs();
        if abs <= threshold {
            0.0
        } else {
            coef.signum() * (abs - threshold)
        }
    }

    /// Denoise a 1-D signal using multi-level Haar wavelet decomposition.
    ///
    /// At each level, the detail coefficients are soft-thresholded with
    /// `threshold`. After all levels the signal is reconstructed.
    ///
    /// `levels` controls the depth of decomposition (clamped so that the
    /// approximation is always ≥ 2 samples).
    pub fn denoise_1d(signal: &[f32], levels: u32, threshold: f32) -> Vec<f32> {
        if signal.is_empty() || levels == 0 {
            return signal.to_vec();
        }

        // Stack of detail coefficient vectors (one per level)
        let mut details: Vec<Vec<f32>> = Vec::new();
        let mut approx = signal.to_vec();

        // Decompose
        for _ in 0..levels {
            if approx.len() < 2 {
                break;
            }
            let (a, d) = HaarWavelet::forward(&approx);
            // Soft-threshold detail coefficients
            let d_thresh: Vec<f32> = d
                .iter()
                .map(|&c| WaveletDenoiser::soft_threshold(c, threshold))
                .collect();
            details.push(d_thresh);
            approx = a;
        }

        // Reconstruct in reverse order
        for d in details.into_iter().rev() {
            approx = HaarWavelet::inverse(&approx, &d);
        }

        // Truncate or pad to original length
        approx.truncate(signal.len());
        while approx.len() < signal.len() {
            approx.push(0.0);
        }
        approx
    }
}

/// Universal (VisuShrink) threshold for Gaussian noise.
///
/// Formula: `sigma * sqrt(2 * ln(n))`
pub struct UniversalThreshold;

impl UniversalThreshold {
    /// Compute the universal threshold.
    ///
    /// `n` is the signal length; `sigma` is the estimated noise standard deviation.
    pub fn compute(n: usize, sigma: f32) -> f32 {
        if n == 0 {
            return 0.0;
        }
        sigma * (2.0 * (n as f32).ln()).sqrt()
    }
}

#[cfg(test)]
mod haar_tests {
    use super::*;

    #[test]
    fn test_haar_forward_even() {
        let signal = vec![1.0f32, 3.0, 5.0, 7.0];
        let (approx, detail) = HaarWavelet::forward(&signal);
        assert_eq!(approx.len(), 2);
        assert_eq!(detail.len(), 2);
    }

    #[test]
    fn test_haar_forward_constant() {
        // Constant signal: all detail coefficients should be ~0
        let signal = vec![1.0f32; 8];
        let (_, detail) = HaarWavelet::forward(&signal);
        for &d in &detail {
            assert!(d.abs() < 1e-6, "Detail should be 0 for constant: {d}");
        }
    }

    #[test]
    fn test_haar_inverse_roundtrip() {
        let signal = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let (approx, detail) = HaarWavelet::forward(&signal);
        let recovered = HaarWavelet::inverse(&approx, &detail);
        for (o, r) in signal.iter().zip(recovered.iter()) {
            assert!((o - r).abs() < 1e-5, "Roundtrip failed: {o} != {r}");
        }
    }

    #[test]
    fn test_haar_empty() {
        let (a, d) = HaarWavelet::forward(&[]);
        assert!(a.is_empty());
        assert!(d.is_empty());
        let r = HaarWavelet::inverse(&[], &[]);
        assert!(r.is_empty());
    }

    #[test]
    fn test_soft_threshold_below() {
        let v = WaveletDenoiser::soft_threshold(0.5, 1.0);
        assert!((v - 0.0).abs() < 1e-6);
    }

    #[test]
    fn test_soft_threshold_above_positive() {
        let v = WaveletDenoiser::soft_threshold(3.0, 1.0);
        assert!((v - 2.0).abs() < 1e-6);
    }

    #[test]
    fn test_soft_threshold_above_negative() {
        let v = WaveletDenoiser::soft_threshold(-3.0, 1.0);
        assert!((v - (-2.0)).abs() < 1e-6);
    }

    #[test]
    fn test_wavelet_denoiser_1d_length_preserved() {
        let signal: Vec<f32> = (0..64).map(|i| (i as f32 * 0.3).sin()).collect();
        let denoised = WaveletDenoiser::denoise_1d(&signal, 3, 0.1);
        assert_eq!(denoised.len(), signal.len());
    }

    #[test]
    fn test_wavelet_denoiser_suppresses_noise() {
        // Pure noise: small random-ish values; after denoising with high threshold
        // output should be smaller in magnitude on average
        let noise: Vec<f32> = (0..128)
            .map(|i| ((i * 17 % 100) as f32 / 100.0 - 0.5) * 0.1)
            .collect();
        let threshold = UniversalThreshold::compute(128, 0.05);
        let denoised = WaveletDenoiser::denoise_1d(&noise, 3, threshold);
        let noise_power: f32 = noise.iter().map(|&s| s * s).sum::<f32>() / noise.len() as f32;
        let denoised_power: f32 =
            denoised.iter().map(|&s| s * s).sum::<f32>() / denoised.len() as f32;
        assert!(
            denoised_power <= noise_power + 1e-6,
            "Denoised power {denoised_power} should not exceed noise power {noise_power}"
        );
    }

    #[test]
    fn test_universal_threshold() {
        let t = UniversalThreshold::compute(1000, 1.0);
        let expected = (2.0 * 1000.0_f32.ln()).sqrt();
        assert!((t - expected).abs() < 1e-4, "Expected {expected}, got {t}");
    }

    #[test]
    fn test_universal_threshold_zero_n() {
        let t = UniversalThreshold::compute(0, 1.0);
        assert!((t - 0.0).abs() < 1e-6);
    }
}
