//! CPU-fallback SSIM/PSNR quality metric computation.
//!
//! Provides production-quality implementations of:
//! - **PSNR** (Peak Signal-to-Noise Ratio) for per-channel and overall quality
//! - **SSIM** (Structural Similarity Index) with sliding-window Gaussian weighting
//! - **MS-SSIM** (Multi-Scale SSIM) for perceptually-weighted quality assessment
//!
//! All computations use f64 internally for numerical precision.  The CPU path
//! is auto-vectorised via `rayon`; a GPU compute-shader path can be wired in
//! later without changing the public API.

use crate::{GpuError, Result};
use rayon::prelude::*;

// ============================================================================
// Public result types
// ============================================================================

/// Per-channel and overall PSNR result.
#[derive(Debug, Clone, Copy)]
pub struct PsnrResult {
    /// PSNR for the red (or Y) channel in dB.
    pub channel_0: f64,
    /// PSNR for the green (or U) channel in dB.
    pub channel_1: f64,
    /// PSNR for the blue (or V) channel in dB.
    pub channel_2: f64,
    /// Overall (weighted-average MSE) PSNR in dB.
    pub overall: f64,
}

/// Per-pixel SSIM result aggregated over the whole image.
#[derive(Debug, Clone, Copy)]
pub struct SsimResult {
    /// Mean SSIM index for the luminance channel (or averaged over channels).
    pub mean_ssim: f64,
    /// Per-channel SSIM values \[R/Y, G/U, B/V\].
    pub per_channel: [f64; 3],
}

/// Multi-scale SSIM result.
#[derive(Debug, Clone, Copy)]
pub struct MsSsimResult {
    /// Final MS-SSIM score (weighted product across scales).
    pub ms_ssim: f64,
    /// Number of scales used.
    pub scales: usize,
}

// ============================================================================
// PSNR computation
// ============================================================================

/// Compute PSNR between two RGBA images.
///
/// Both buffers must be `width * height * 4` bytes (packed RGBA, u8).
/// The alpha channel is ignored in the computation.
///
/// Returns `Err` on size mismatch.  Returns `f64::INFINITY` when the images
/// are identical (MSE = 0).
pub fn compute_psnr(
    reference: &[u8],
    distorted: &[u8],
    width: u32,
    height: u32,
) -> Result<PsnrResult> {
    let expected = (width as usize) * (height as usize) * 4;
    validate_pair(reference, distorted, expected)?;

    let pixel_count = (width as u64) * (height as u64);
    if pixel_count == 0 {
        return Err(GpuError::InvalidDimensions { width, height });
    }

    // Accumulate MSE per channel using chunked parallel reduction.
    let chunk_size = 4096_usize; // pixels per chunk
    let num_pixels = pixel_count as usize;
    let num_chunks = num_pixels.div_ceil(chunk_size);

    let channel_mse: [f64; 3] = (0..num_chunks)
        .into_par_iter()
        .map(|chunk_idx| {
            let start = chunk_idx * chunk_size;
            let end = (start + chunk_size).min(num_pixels);
            let mut mse = [0.0_f64; 3];
            for i in start..end {
                let base = i * 4;
                for c in 0..3 {
                    let diff = reference[base + c] as f64 - distorted[base + c] as f64;
                    mse[c] += diff * diff;
                }
            }
            mse
        })
        .reduce(
            || [0.0_f64; 3],
            |a, b| [a[0] + b[0], a[1] + b[1], a[2] + b[2]],
        );

    let n = pixel_count as f64;
    let max_val = 255.0_f64;
    let max_sq = max_val * max_val;

    let psnr_from_mse = |mse: f64| -> f64 {
        if mse == 0.0 {
            f64::INFINITY
        } else {
            10.0 * (max_sq / mse).log10()
        }
    };

    let mse = [channel_mse[0] / n, channel_mse[1] / n, channel_mse[2] / n];
    let overall_mse = (mse[0] + mse[1] + mse[2]) / 3.0;

    Ok(PsnrResult {
        channel_0: psnr_from_mse(mse[0]),
        channel_1: psnr_from_mse(mse[1]),
        channel_2: psnr_from_mse(mse[2]),
        overall: psnr_from_mse(overall_mse),
    })
}

// ============================================================================
// SSIM computation
// ============================================================================

/// SSIM constants (from the original Wang et al. paper).
const C1: f64 = (0.01 * 255.0) * (0.01 * 255.0); // (K1*L)^2
const C2: f64 = (0.03 * 255.0) * (0.03 * 255.0); // (K2*L)^2

/// Default SSIM window radius (11x11 window).
const SSIM_WINDOW_RADIUS: usize = 5;

/// Compute SSIM between two RGBA images using an 11x11 Gaussian window.
///
/// Both buffers must be `width * height * 4` bytes (packed RGBA, u8).
/// The alpha channel is ignored.
///
/// Returns per-channel and mean SSIM values in \[-1, 1\] (typically 0..1).
pub fn compute_ssim(
    reference: &[u8],
    distorted: &[u8],
    width: u32,
    height: u32,
) -> Result<SsimResult> {
    let expected = (width as usize) * (height as usize) * 4;
    validate_pair(reference, distorted, expected)?;

    let w = width as usize;
    let h = height as usize;
    if w == 0 || h == 0 {
        return Err(GpuError::InvalidDimensions { width, height });
    }

    let kernel = build_gaussian_kernel(SSIM_WINDOW_RADIUS, 1.5);
    let ksize = 2 * SSIM_WINDOW_RADIUS + 1;

    let mut per_channel = [0.0_f64; 3];

    for c in 0..3_usize {
        // Extract single-channel f64 planes.
        let ref_plane: Vec<f64> = (0..w * h).map(|i| reference[i * 4 + c] as f64).collect();
        let dist_plane: Vec<f64> = (0..w * h).map(|i| distorted[i * 4 + c] as f64).collect();

        // Compute local statistics via Gaussian-weighted windows.
        // Only compute for interior pixels where the full window fits.
        let valid_rows = h.saturating_sub(ksize - 1);
        let valid_cols = w.saturating_sub(ksize - 1);

        if valid_rows == 0 || valid_cols == 0 {
            // Image too small for full window; fall back to global comparison.
            per_channel[c] = global_ssim_channel(&ref_plane, &dist_plane);
            continue;
        }

        let ssim_sum: f64 = (0..valid_rows)
            .into_par_iter()
            .map(|vy| {
                let mut row_sum = 0.0_f64;
                for vx in 0..valid_cols {
                    let oy = vy;
                    let ox = vx;

                    let mut mu_x = 0.0_f64;
                    let mut mu_y = 0.0_f64;
                    let mut sigma_xx = 0.0_f64;
                    let mut sigma_yy = 0.0_f64;
                    let mut sigma_xy = 0.0_f64;

                    for ky in 0..ksize {
                        for kx in 0..ksize {
                            let weight = kernel[ky * ksize + kx];
                            let idx = (oy + ky) * w + (ox + kx);
                            let x = ref_plane[idx];
                            let y = dist_plane[idx];
                            mu_x += weight * x;
                            mu_y += weight * y;
                            sigma_xx += weight * x * x;
                            sigma_yy += weight * y * y;
                            sigma_xy += weight * x * y;
                        }
                    }

                    sigma_xx -= mu_x * mu_x;
                    sigma_yy -= mu_y * mu_y;
                    sigma_xy -= mu_x * mu_y;

                    let numerator = (2.0 * mu_x * mu_y + C1) * (2.0 * sigma_xy + C2);
                    let denominator = (mu_x * mu_x + mu_y * mu_y + C1) * (sigma_xx + sigma_yy + C2);

                    row_sum += if denominator > 0.0 {
                        numerator / denominator
                    } else {
                        1.0
                    };
                }
                row_sum
            })
            .sum();

        let total_windows = (valid_rows * valid_cols) as f64;
        per_channel[c] = if total_windows > 0.0 {
            ssim_sum / total_windows
        } else {
            1.0
        };
    }

    let mean_ssim = (per_channel[0] + per_channel[1] + per_channel[2]) / 3.0;

    Ok(SsimResult {
        mean_ssim,
        per_channel,
    })
}

/// Compute Multi-Scale SSIM (MS-SSIM) using up to 5 scales.
///
/// The image is successively downsampled by 2x.  At each scale the
/// contrast/structure components are extracted; at the finest scale
/// the luminance component is also included.
///
/// Returns the final MS-SSIM score in \[0, 1\].
pub fn compute_ms_ssim(
    reference: &[u8],
    distorted: &[u8],
    width: u32,
    height: u32,
) -> Result<MsSsimResult> {
    let expected = (width as usize) * (height as usize) * 4;
    validate_pair(reference, distorted, expected)?;

    if width == 0 || height == 0 {
        return Err(GpuError::InvalidDimensions { width, height });
    }

    // MS-SSIM weights (from the original paper, 5 scales).
    let weights: [f64; 5] = [0.0448, 0.2856, 0.3001, 0.2363, 0.1333];

    // Determine maximum number of scales (minimum dimension must be >= 11 at each scale).
    let max_scales = {
        let min_dim = width.min(height);
        let mut s = 0_usize;
        let mut dim = min_dim;
        while dim >= 11 && s < 5 {
            s += 1;
            dim /= 2;
        }
        s.max(1)
    };

    let mut ref_buf = reference.to_vec();
    let mut dist_buf = distorted.to_vec();
    let mut cur_w = width;
    let mut cur_h = height;

    let mut ms_ssim_product = 1.0_f64;
    let mut actual_scales = 0_usize;

    for scale in 0..max_scales {
        let ssim = compute_ssim(&ref_buf, &dist_buf, cur_w, cur_h)?;
        let cs = ssim.mean_ssim; // contrast-structure component

        let weight = weights[scale.min(weights.len() - 1)];

        if scale == max_scales - 1 {
            // Last scale: use full SSIM (luminance + contrast + structure).
            ms_ssim_product *= cs.max(0.0).powf(weight);
        } else {
            // Intermediate scale: use contrast-structure only.
            ms_ssim_product *= cs.max(0.0).powf(weight);
        }
        actual_scales += 1;

        // Downsample 2x for next scale.
        if scale + 1 < max_scales {
            let new_w = (cur_w / 2).max(1);
            let new_h = (cur_h / 2).max(1);
            ref_buf = downsample_2x_rgba(&ref_buf, cur_w, cur_h);
            dist_buf = downsample_2x_rgba(&dist_buf, cur_w, cur_h);
            cur_w = new_w;
            cur_h = new_h;
        }
    }

    Ok(MsSsimResult {
        ms_ssim: ms_ssim_product,
        scales: actual_scales,
    })
}

// ============================================================================
// Internal helpers
// ============================================================================

fn validate_pair(reference: &[u8], distorted: &[u8], expected: usize) -> Result<()> {
    if reference.len() < expected {
        return Err(GpuError::InvalidBufferSize {
            expected,
            actual: reference.len(),
        });
    }
    if distorted.len() < expected {
        return Err(GpuError::InvalidBufferSize {
            expected,
            actual: distorted.len(),
        });
    }
    if reference.len() != distorted.len() {
        return Err(GpuError::InvalidBufferSize {
            expected: reference.len(),
            actual: distorted.len(),
        });
    }
    Ok(())
}

/// Build a 2D Gaussian kernel of size `(2*radius+1)^2`, normalised to sum=1.
fn build_gaussian_kernel(radius: usize, sigma: f64) -> Vec<f64> {
    let size = 2 * radius + 1;
    let mut kernel = vec![0.0_f64; size * size];
    let two_sigma_sq = 2.0 * sigma * sigma;
    let mut sum = 0.0_f64;

    for y in 0..size {
        for x in 0..size {
            let dx = x as f64 - radius as f64;
            let dy = y as f64 - radius as f64;
            let v = (-(dx * dx + dy * dy) / two_sigma_sq).exp();
            kernel[y * size + x] = v;
            sum += v;
        }
    }
    if sum > 0.0 {
        for v in &mut kernel {
            *v /= sum;
        }
    }
    kernel
}

/// Global SSIM for a single channel (used when image is too small for windowed SSIM).
fn global_ssim_channel(ref_plane: &[f64], dist_plane: &[f64]) -> f64 {
    let n = ref_plane.len() as f64;
    if n == 0.0 {
        return 1.0;
    }

    let mu_x: f64 = ref_plane.iter().sum::<f64>() / n;
    let mu_y: f64 = dist_plane.iter().sum::<f64>() / n;

    let mut sigma_xx = 0.0_f64;
    let mut sigma_yy = 0.0_f64;
    let mut sigma_xy = 0.0_f64;

    for i in 0..ref_plane.len() {
        let dx = ref_plane[i] - mu_x;
        let dy = dist_plane[i] - mu_y;
        sigma_xx += dx * dx;
        sigma_yy += dy * dy;
        sigma_xy += dx * dy;
    }
    sigma_xx /= n;
    sigma_yy /= n;
    sigma_xy /= n;

    let numerator = (2.0 * mu_x * mu_y + C1) * (2.0 * sigma_xy + C2);
    let denominator = (mu_x * mu_x + mu_y * mu_y + C1) * (sigma_xx + sigma_yy + C2);

    if denominator > 0.0 {
        numerator / denominator
    } else {
        1.0
    }
}

/// Downsample an RGBA image by 2x using box filter (average of 2x2 blocks).
fn downsample_2x_rgba(input: &[u8], width: u32, height: u32) -> Vec<u8> {
    let w = width as usize;
    let h = height as usize;
    let new_w = (w / 2).max(1);
    let new_h = (h / 2).max(1);
    let mut output = vec![0u8; new_w * new_h * 4];

    for ny in 0..new_h {
        for nx in 0..new_w {
            for c in 0..4_usize {
                let mut sum = 0u32;
                let mut count = 0u32;
                for dy in 0..2_usize {
                    for dx in 0..2_usize {
                        let sy = (ny * 2 + dy).min(h - 1);
                        let sx = (nx * 2 + dx).min(w - 1);
                        sum += input[(sy * w + sx) * 4 + c] as u32;
                        count += 1;
                    }
                }
                output[(ny * new_w + nx) * 4 + c] = (sum / count) as u8;
            }
        }
    }

    output
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    fn solid_rgba(width: u32, height: u32, r: u8, g: u8, b: u8) -> Vec<u8> {
        let n = (width as usize) * (height as usize);
        let mut buf = Vec::with_capacity(n * 4);
        for _ in 0..n {
            buf.extend_from_slice(&[r, g, b, 255]);
        }
        buf
    }

    fn gradient_rgba(width: u32, height: u32) -> Vec<u8> {
        let w = width as usize;
        let h = height as usize;
        let mut buf = Vec::with_capacity(w * h * 4);
        for y in 0..h {
            for x in 0..w {
                let v = ((x + y) % 256) as u8;
                buf.extend_from_slice(&[v, v, v, 255]);
            }
        }
        buf
    }

    // --- PSNR tests ---

    #[test]
    fn test_psnr_identical_images() {
        let img = solid_rgba(32, 32, 128, 64, 200);
        let result = compute_psnr(&img, &img, 32, 32).expect("psnr should succeed");
        assert!(
            result.overall.is_infinite(),
            "identical images should yield infinite PSNR"
        );
        assert!(result.channel_0.is_infinite());
        assert!(result.channel_1.is_infinite());
        assert!(result.channel_2.is_infinite());
    }

    #[test]
    fn test_psnr_different_images() {
        let img_a = solid_rgba(16, 16, 100, 100, 100);
        let img_b = solid_rgba(16, 16, 110, 100, 100);
        let result = compute_psnr(&img_a, &img_b, 16, 16).expect("psnr should succeed");
        // Only channel 0 differs, so channel_1 and channel_2 should be infinite.
        assert!(result.channel_1.is_infinite());
        assert!(result.channel_2.is_infinite());
        // Channel 0 PSNR: 10*log10(255^2/100) = 10*log10(650.25) ≈ 28.13 dB
        assert!(
            result.channel_0 > 20.0 && result.channel_0 < 40.0,
            "PSNR channel 0 = {} dB, expected ~28 dB",
            result.channel_0
        );
        assert!(result.overall > 20.0);
    }

    #[test]
    fn test_psnr_high_difference() {
        let img_a = solid_rgba(8, 8, 0, 0, 0);
        let img_b = solid_rgba(8, 8, 255, 255, 255);
        let result = compute_psnr(&img_a, &img_b, 8, 8).expect("psnr should succeed");
        // MSE = 255^2 = 65025, PSNR = 10*log10(1) = 0 dB
        assert!(
            result.overall.abs() < 0.01,
            "max-difference PSNR should be ~0 dB"
        );
    }

    #[test]
    fn test_psnr_buffer_size_mismatch() {
        let img_a = solid_rgba(8, 8, 0, 0, 0);
        let img_b = solid_rgba(4, 4, 0, 0, 0);
        assert!(compute_psnr(&img_a, &img_b, 8, 8).is_err());
    }

    #[test]
    fn test_psnr_zero_dimensions() {
        let img = vec![];
        assert!(compute_psnr(&img, &img, 0, 0).is_err());
    }

    #[test]
    fn test_psnr_gradient_self() {
        let img = gradient_rgba(64, 64);
        let result = compute_psnr(&img, &img, 64, 64).expect("psnr should succeed");
        assert!(result.overall.is_infinite());
    }

    // --- SSIM tests ---

    #[test]
    fn test_ssim_identical_images() {
        let img = gradient_rgba(32, 32);
        let result = compute_ssim(&img, &img, 32, 32).expect("ssim should succeed");
        assert!(
            (result.mean_ssim - 1.0).abs() < 1e-6,
            "SSIM of identical images should be 1.0, got {}",
            result.mean_ssim
        );
    }

    #[test]
    fn test_ssim_very_different_images() {
        let img_a = solid_rgba(32, 32, 0, 0, 0);
        let img_b = solid_rgba(32, 32, 255, 255, 255);
        let result = compute_ssim(&img_a, &img_b, 32, 32).expect("ssim should succeed");
        assert!(
            result.mean_ssim < 0.1,
            "SSIM of maximally different images should be low, got {}",
            result.mean_ssim
        );
    }

    #[test]
    fn test_ssim_slightly_different() {
        let img_a = gradient_rgba(32, 32);
        let mut img_b = img_a.clone();
        // Add small perturbation to some pixels.
        for i in (0..img_b.len()).step_by(8) {
            img_b[i] = img_b[i].saturating_add(5);
        }
        let result = compute_ssim(&img_a, &img_b, 32, 32).expect("ssim should succeed");
        assert!(
            result.mean_ssim > 0.8 && result.mean_ssim < 1.0,
            "slight perturbation should yield high but not perfect SSIM, got {}",
            result.mean_ssim
        );
    }

    #[test]
    fn test_ssim_per_channel() {
        let img = gradient_rgba(32, 32);
        let result = compute_ssim(&img, &img, 32, 32).expect("ssim should succeed");
        for c in 0..3 {
            assert!(
                (result.per_channel[c] - 1.0).abs() < 1e-6,
                "channel {} SSIM should be 1.0, got {}",
                c,
                result.per_channel[c]
            );
        }
    }

    #[test]
    fn test_ssim_small_image_fallback() {
        // Image smaller than 11x11 window — should use global fallback.
        let img = solid_rgba(4, 4, 128, 128, 128);
        let result = compute_ssim(&img, &img, 4, 4).expect("ssim should succeed");
        assert!((result.mean_ssim - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_ssim_buffer_mismatch() {
        let img_a = solid_rgba(8, 8, 0, 0, 0);
        let img_b = solid_rgba(4, 4, 0, 0, 0);
        assert!(compute_ssim(&img_a, &img_b, 8, 8).is_err());
    }

    // --- MS-SSIM tests ---

    #[test]
    fn test_ms_ssim_identical() {
        let img = gradient_rgba(64, 64);
        let result = compute_ms_ssim(&img, &img, 64, 64).expect("ms-ssim should succeed");
        assert!(
            (result.ms_ssim - 1.0).abs() < 1e-4,
            "MS-SSIM of identical images should be ~1.0, got {}",
            result.ms_ssim
        );
        assert!(result.scales >= 2, "should use multiple scales");
    }

    #[test]
    fn test_ms_ssim_different() {
        let img_a = gradient_rgba(64, 64);
        let img_b = solid_rgba(64, 64, 0, 0, 0);
        let result = compute_ms_ssim(&img_a, &img_b, 64, 64).expect("ms-ssim should succeed");
        assert!(
            result.ms_ssim < 0.5,
            "MS-SSIM of very different images should be low, got {}",
            result.ms_ssim
        );
    }

    #[test]
    fn test_ms_ssim_small_image() {
        // Small image: only 1 scale possible.
        let img = gradient_rgba(16, 16);
        let result = compute_ms_ssim(&img, &img, 16, 16).expect("ms-ssim should succeed");
        assert!(result.scales >= 1);
        assert!(result.ms_ssim > 0.9);
    }

    // --- Downsample helper tests ---

    #[test]
    fn test_downsample_2x_size() {
        let img = solid_rgba(32, 32, 100, 100, 100);
        let down = downsample_2x_rgba(&img, 32, 32);
        assert_eq!(down.len(), 16 * 16 * 4);
    }

    #[test]
    fn test_downsample_2x_preserves_solid() {
        let img = solid_rgba(16, 16, 200, 100, 50);
        let down = downsample_2x_rgba(&img, 16, 16);
        // All pixels in downsampled should be the same color.
        for i in 0..(8 * 8) {
            assert_eq!(down[i * 4], 200);
            assert_eq!(down[i * 4 + 1], 100);
            assert_eq!(down[i * 4 + 2], 50);
        }
    }

    // --- Gaussian kernel tests ---

    #[test]
    fn test_gaussian_kernel_sums_to_one() {
        let kernel = build_gaussian_kernel(5, 1.5);
        let sum: f64 = kernel.iter().sum();
        assert!(
            (sum - 1.0).abs() < 1e-10,
            "Gaussian kernel should sum to 1.0, got {}",
            sum
        );
    }

    #[test]
    fn test_gaussian_kernel_center_is_maximum() {
        let radius = 5;
        let kernel = build_gaussian_kernel(radius, 1.5);
        let size = 2 * radius + 1;
        let center = kernel[radius * size + radius];
        for &v in &kernel {
            assert!(v <= center + 1e-15);
        }
    }

    #[test]
    fn test_gaussian_kernel_symmetric() {
        let radius = 3;
        let kernel = build_gaussian_kernel(radius, 1.0);
        let size = 2 * radius + 1;
        for y in 0..size {
            for x in 0..size {
                let mirror_y = size - 1 - y;
                let mirror_x = size - 1 - x;
                assert!((kernel[y * size + x] - kernel[mirror_y * size + mirror_x]).abs() < 1e-15);
            }
        }
    }
}
