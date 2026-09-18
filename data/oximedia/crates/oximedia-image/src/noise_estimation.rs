//! Image noise estimation and spatial denoising algorithms.
//!
//! This module provides:
//!
//! - **Noise estimation** — Donoho-Johnstone MAD estimator via Laplacian
//!   filtering, SNR computation, and heuristic noise-type classification.
//! - **Median filter** — fast separable 1D median passes (x then y).
//! - **Bilateral filter** — edge-preserving smoothing with spatial and range
//!   Gaussian weights.
//! - **Non-Local Means (NLM)** — patch-based denoising with self-similarity
//!   weighting.
//!
//! All functions operate on flat, row-major single-channel pixel buffers.
//! Floating-point functions expect values in an arbitrary finite range (the
//! algorithms are scale-invariant); `u8` functions work in `[0, 255]`.

#![allow(dead_code)]

// ---------------------------------------------------------------------------
// Public types
// ---------------------------------------------------------------------------

/// Characterisation of the noise present in an image.
#[derive(Debug, Clone)]
pub struct NoiseProfile {
    /// Estimated noise standard deviation (same units as input pixel values).
    pub sigma: f32,
    /// Signal-to-noise ratio in decibels: `20·log₁₀(signal_rms / sigma)`.
    pub snr_db: f32,
    /// Heuristic classification of the dominant noise type.
    pub noise_type: NoiseType,
    /// Dominant noise frequency.  `0.0` indicates low-frequency spatial noise;
    /// `1.0` indicates high-frequency (pixel-level) noise.
    pub spatial_frequency: f32,
}

/// Heuristic noise type classification.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NoiseType {
    /// Additive white Gaussian noise.
    Gaussian,
    /// Impulse noise (salt-and-pepper).
    SaltAndPepper,
    /// Poisson (shot) noise proportional to signal level.
    Poisson,
    /// Periodic / structured noise (visible in frequency domain).
    Periodic,
    /// Could not be classified with confidence.
    Unknown,
}

// ---------------------------------------------------------------------------
// Noise estimation
// ---------------------------------------------------------------------------

/// Estimate the noise standard deviation using the Donoho-Johnstone median
/// absolute deviation (MAD) estimator applied to the 2D Laplacian of the image.
///
/// Algorithm:
/// 1. Apply the discrete 2D Laplacian kernel `[0,1,0; 1,-4,1; 0,1,0]` to
///    every interior pixel (border pixels are skipped).
/// 2. Collect the absolute values of the Laplacian responses.
/// 3. Return `median(|laplacian|) / 0.6745`.
///
/// The divisor 0.6745 converts the median absolute deviation to an unbiased
/// estimate of the Gaussian standard deviation.
///
/// Returns `0.0` if the image has fewer than 9 pixels (too small for a valid
/// estimate).
#[must_use]
pub fn estimate_noise_sigma(pixels: &[f32], width: u32, height: u32) -> f32 {
    let w = width as usize;
    let h = height as usize;

    if w < 3 || h < 3 {
        return 0.0;
    }

    let mut responses: Vec<f32> = Vec::with_capacity((w - 2) * (h - 2));

    for y in 1..h - 1 {
        for x in 1..w - 1 {
            let center = pixels[y * w + x];
            let top = pixels[(y - 1) * w + x];
            let bottom = pixels[(y + 1) * w + x];
            let left = pixels[y * w + x - 1];
            let right = pixels[y * w + x + 1];
            // Laplacian: top + bottom + left + right - 4·center
            let lap = top + bottom + left + right - 4.0 * center;
            responses.push(lap.abs());
        }
    }

    if responses.is_empty() {
        return 0.0;
    }

    let med = median_f32(&mut responses);
    med / 0.6745
}

/// Estimate a full `NoiseProfile` for an image.
///
/// Steps:
/// 1. Estimate `sigma` via `estimate_noise_sigma`.
/// 2. Compute `signal_rms` as the RMS of the pixel values.
/// 3. Compute `snr_db = 20 · log₁₀(signal_rms / sigma)` (clamped to `[−60, 120]` dB).
/// 4. Classify noise type heuristically from the spatial autocorrelation of the
///    noise residual (image minus a 3-pixel box-blurred copy).
/// 5. Estimate `spatial_frequency` as the fraction of noise energy in the
///    high-frequency half of the autocorrelation.
#[must_use]
pub fn estimate_noise_profile(pixels: &[f32], width: u32, height: u32) -> NoiseProfile {
    let sigma = estimate_noise_sigma(pixels, width, height);

    let n = pixels.len();
    let signal_rms = if n == 0 {
        0.0_f32
    } else {
        let sum_sq: f32 = pixels.iter().map(|&v| v * v).sum();
        (sum_sq / n as f32).sqrt()
    };

    let snr_db = if sigma > 1e-10 && signal_rms > 1e-10 {
        (20.0 * (signal_rms / sigma).log10()).clamp(-60.0, 120.0)
    } else if sigma <= 1e-10 {
        120.0 // very low noise
    } else {
        -60.0
    };

    // --- Noise residual: image minus box-blurred image (3-wide) ------------
    let blurred = box_blur_3(pixels, width, height);
    let residual: Vec<f32> = pixels
        .iter()
        .zip(blurred.iter())
        .map(|(&p, &b)| p - b)
        .collect();

    // --- Autocorrelation at lag 0 and lag 1 (horizontal) -------------------
    let w = width as usize;
    let h = height as usize;
    let ac0: f64 = residual.iter().map(|&v| (v as f64) * (v as f64)).sum();
    let ac1: f64 = if w > 1 {
        (0..h)
            .flat_map(|y| (0..w - 1).map(move |x| (y, x)))
            .map(|(y, x)| residual[y * w + x] as f64 * residual[y * w + x + 1] as f64)
            .sum()
    } else {
        0.0
    };

    let spatial_frequency = if ac0.abs() > 1e-12 {
        // ac1/ac0 close to 1 → low-freq correlated noise; close to 0 → white noise
        let normalized_ac1 = (ac1 / ac0).clamp(-1.0, 1.0);
        (1.0 - normalized_ac1) as f32 * 0.5 + 0.5
    } else {
        0.5
    };

    // --- Heuristic noise type classification --------------------------------
    let noise_type = classify_noise_type(pixels, &residual, sigma, spatial_frequency);

    NoiseProfile {
        sigma,
        snr_db,
        noise_type,
        spatial_frequency,
    }
}

/// Heuristic classification based on residual statistics.
fn classify_noise_type(
    pixels: &[f32],
    residual: &[f32],
    sigma: f32,
    spatial_frequency: f32,
) -> NoiseType {
    if residual.is_empty() || pixels.is_empty() {
        return NoiseType::Unknown;
    }

    // Count impulse-like outliers in the residual (> 5·sigma)
    let threshold = 5.0 * sigma;
    let outlier_fraction = if sigma > 1e-10 {
        let count = residual.iter().filter(|&&v| v.abs() > threshold).count();
        count as f32 / residual.len() as f32
    } else {
        0.0
    };

    if outlier_fraction > 0.01 {
        return NoiseType::SaltAndPepper;
    }

    // Periodic noise: very high spatial frequency with periodic structure.
    // Detected when spatial_frequency is near 1.0 and the sigma is non-trivial.
    if spatial_frequency > 0.85 && sigma > 1e-6 {
        // Check if residual has significant periodicity (autocorrelation at lag 2)
        let w = pixels.len().max(1);
        let ac2: f64 = if w > 2 {
            residual[..w - 2]
                .iter()
                .zip(residual[2..].iter())
                .map(|(&a, &b)| a as f64 * b as f64)
                .sum()
        } else {
            0.0
        };
        let ac0: f64 = residual.iter().map(|&v| (v as f64) * (v as f64)).sum();
        if ac0 > 1e-12 && (ac2 / ac0).abs() > 0.1 {
            return NoiseType::Periodic;
        }
    }

    // Poisson noise: variance proportional to mean signal level
    if !pixels.is_empty() {
        let mean: f32 = pixels.iter().sum::<f32>() / pixels.len() as f32;
        // For Poisson noise sigma ≈ sqrt(mean) when normalised to counts
        // We check if sigma² / mean is near 1 (within 2×)
        if mean > 1e-4 {
            let ratio = (sigma * sigma) / mean;
            if ratio > 0.1 && ratio < 10.0 {
                return NoiseType::Poisson;
            }
        }
    }

    // Default: Gaussian additive noise
    if sigma > 1e-10 {
        NoiseType::Gaussian
    } else {
        NoiseType::Unknown
    }
}

// ---------------------------------------------------------------------------
// Denoising — Median filter
// ---------------------------------------------------------------------------

/// Median filter on a `u8` image using separable 1D passes (x then y).
///
/// The filter radius determines the window size: `(2*radius+1)` pixels per
/// pass.  Using separable passes approximates a 2D median at reduced cost;
/// it is not mathematically identical to a true 2D median but is effective in
/// practice for typical use cases.
///
/// Out-of-bounds positions use border replication (clamp addressing).
#[must_use]
pub fn denoise_median(pixels: &[u8], width: u32, height: u32, radius: u32) -> Vec<u8> {
    if radius == 0 || pixels.is_empty() {
        return pixels.to_vec();
    }
    // Horizontal pass
    let h_pass = median_pass_horizontal(pixels, width, height, radius);
    // Vertical pass
    median_pass_vertical(&h_pass, width, height, radius)
}

/// Apply a 1D horizontal median filter.
fn median_pass_horizontal(pixels: &[u8], width: u32, height: u32, radius: u32) -> Vec<u8> {
    let w = width as usize;
    let h = height as usize;
    let r = radius as usize;
    let mut output = vec![0u8; w * h];

    for y in 0..h {
        let row = &pixels[y * w..(y + 1) * w];
        for x in 0..w {
            let mut window: Vec<u8> = (0..2 * r + 1)
                .map(|k| {
                    let src = (x + k).saturating_sub(r).min(w - 1);
                    row[src]
                })
                .collect();
            window.sort_unstable();
            output[y * w + x] = window[r];
        }
    }

    output
}

/// Apply a 1D vertical median filter.
fn median_pass_vertical(pixels: &[u8], width: u32, height: u32, radius: u32) -> Vec<u8> {
    let w = width as usize;
    let h = height as usize;
    let r = radius as usize;
    let mut output = vec![0u8; w * h];

    for x in 0..w {
        for y in 0..h {
            let mut window: Vec<u8> = (0..2 * r + 1)
                .map(|k| {
                    let sy = (y + k).saturating_sub(r).min(h - 1);
                    pixels[sy * w + x]
                })
                .collect();
            window.sort_unstable();
            output[y * w + x] = window[r];
        }
    }

    output
}

// ---------------------------------------------------------------------------
// Denoising — Bilateral filter
// ---------------------------------------------------------------------------

/// Bilateral filter for `f32` images.
///
/// Each output pixel is a weighted average of neighbouring pixels where the
/// weight combines:
/// - **Spatial Gaussian**: `exp(-dist² / (2·σ_s²))`  — downweights distant pixels.
/// - **Range Gaussian**: `exp(-|I(p)-I(q)|² / (2·σ_r²))` — downweights pixels
///   with dissimilar intensity (preserving edges).
///
/// The search window radius is `ceil(3·sigma_spatial)`.
#[must_use]
pub fn denoise_bilateral(
    pixels: &[f32],
    width: u32,
    height: u32,
    sigma_spatial: f32,
    sigma_range: f32,
) -> Vec<f32> {
    let w = width as usize;
    let h = height as usize;

    if pixels.is_empty() || sigma_spatial <= 0.0 || sigma_range <= 0.0 {
        return pixels.to_vec();
    }

    let radius = (3.0 * sigma_spatial).ceil() as usize;
    let two_ss_sq = 2.0 * sigma_spatial * sigma_spatial;
    let two_sr_sq = 2.0 * sigma_range * sigma_range;
    let mut output = vec![0.0_f32; w * h];

    for cy in 0..h {
        for cx in 0..w {
            let center_val = pixels[cy * w + cx];
            let mut weighted_sum = 0.0_f32;
            let mut weight_total = 0.0_f32;

            let y_min = cy.saturating_sub(radius);
            let y_max = (cy + radius).min(h - 1);
            let x_min = cx.saturating_sub(radius);
            let x_max = (cx + radius).min(w - 1);

            for sy in y_min..=y_max {
                for sx in x_min..=x_max {
                    let neighbor_val = pixels[sy * w + sx];
                    let dx = sx as f32 - cx as f32;
                    let dy = sy as f32 - cy as f32;
                    let dist_sq = dx * dx + dy * dy;
                    let range_sq = (center_val - neighbor_val) * (center_val - neighbor_val);

                    let w_spatial = (-dist_sq / two_ss_sq).exp();
                    let w_range = (-range_sq / two_sr_sq).exp();
                    let weight = w_spatial * w_range;

                    weighted_sum += neighbor_val * weight;
                    weight_total += weight;
                }
            }

            output[cy * w + cx] = if weight_total > 1e-12 {
                weighted_sum / weight_total
            } else {
                center_val
            };
        }
    }

    output
}

// ---------------------------------------------------------------------------
// Denoising — Non-Local Means
// ---------------------------------------------------------------------------

/// Non-Local Means (NLM) denoising for `f32` images.
///
/// For each pixel `i`, a weighted average is computed over all pixels `j`
/// within a `search_window × search_window` neighbourhood:
///
/// ```text
/// output[i] = Σ_j w(i,j) · I(j)  /  Σ_j w(i,j)
/// where  w(i,j) = exp(-||P(i) - P(j)||² / h²)
/// ```
///
/// `P(i)` is the `patch_size × patch_size` patch centred at pixel `i`.
/// `h` is the filter strength parameter (larger → more smoothing).
///
/// Out-of-bounds pixels in patches and search windows use clamp addressing.
#[must_use]
pub fn denoise_nlm(
    pixels: &[f32],
    width: u32,
    height: u32,
    h_param: f32,
    patch_size: u32,
    search_window: u32,
) -> Vec<f32> {
    let w = width as usize;
    let ht = height as usize;

    if pixels.is_empty() || patch_size == 0 || h_param <= 0.0 {
        return pixels.to_vec();
    }

    let patch_r = (patch_size / 2) as isize;
    let search_r = (search_window / 2) as isize;
    let h_sq = h_param * h_param;
    let patch_area = ((2 * patch_r + 1) * (2 * patch_r + 1)) as f32;
    let mut output = vec![0.0_f32; w * ht];

    for cy in 0..ht as isize {
        for cx in 0..w as isize {
            let mut weighted_sum = 0.0_f32;
            let mut weight_total = 0.0_f32;

            // Search window bounds
            let sy_min = (cy - search_r).max(0);
            let sy_max = (cy + search_r).min(ht as isize - 1);
            let sx_min = (cx - search_r).max(0);
            let sx_max = (cx + search_r).min(w as isize - 1);

            for sy in sy_min..=sy_max {
                for sx in sx_min..=sx_max {
                    // Compute patch distance ||P(i) - P(j)||²
                    let mut patch_dist_sq = 0.0_f32;
                    for py in -patch_r..=patch_r {
                        for px in -patch_r..=patch_r {
                            let ai = ((cy + py).clamp(0, ht as isize - 1) as usize) * w
                                + (cx + px).clamp(0, w as isize - 1) as usize;
                            let bi = ((sy + py).clamp(0, ht as isize - 1) as usize) * w
                                + (sx + px).clamp(0, w as isize - 1) as usize;
                            let diff = pixels[ai] - pixels[bi];
                            patch_dist_sq += diff * diff;
                        }
                    }
                    // Normalise by patch area
                    let normed = patch_dist_sq / patch_area;
                    let weight = (-normed / h_sq).exp();
                    let neighbor_val = pixels[sy as usize * w + sx as usize];

                    weighted_sum += neighbor_val * weight;
                    weight_total += weight;
                }
            }

            output[cy as usize * w + cx as usize] = if weight_total > 1e-12 {
                weighted_sum / weight_total
            } else {
                pixels[cy as usize * w + cx as usize]
            };
        }
    }

    output
}

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------

/// Compute the median of a mutable `f32` slice (sorts in-place).
fn median_f32(values: &mut Vec<f32>) -> f32 {
    if values.is_empty() {
        return 0.0;
    }
    values.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let mid = values.len() / 2;
    if values.len() % 2 == 0 {
        (values[mid - 1] + values[mid]) * 0.5
    } else {
        values[mid]
    }
}

/// Simple 3×3 box blur (normalised) for generating a smooth reference image.
///
/// Border pixels are replicated (clamp addressing).
fn box_blur_3(pixels: &[f32], width: u32, height: u32) -> Vec<f32> {
    let w = width as usize;
    let h = height as usize;
    if w == 0 || h == 0 {
        return vec![];
    }
    let mut output = vec![0.0_f32; w * h];

    for cy in 0..h {
        for cx in 0..w {
            let mut sum = 0.0_f32;
            let mut count = 0u32;
            for dy in -1isize..=1 {
                for dx in -1isize..=1 {
                    let sx = (cx as isize + dx).clamp(0, w as isize - 1) as usize;
                    let sy = (cy as isize + dy).clamp(0, h as isize - 1) as usize;
                    sum += pixels[sy * w + sx];
                    count += 1;
                }
            }
            output[cy * w + cx] = sum / count as f32;
        }
    }

    output
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // -----------------------------------------------------------------------
    // estimate_noise_sigma
    // -----------------------------------------------------------------------

    #[test]
    fn test_estimate_noise_sigma_constant_image() {
        // A perfectly constant image has no Laplacian response → sigma ≈ 0
        let pixels = vec![0.5_f32; 25];
        let sigma = estimate_noise_sigma(&pixels, 5, 5);
        assert!(
            sigma < 1e-5,
            "sigma of constant image should be ~0, got {sigma}"
        );
    }

    #[test]
    fn test_estimate_noise_sigma_too_small() {
        // Image smaller than 3×3 → returns 0
        let pixels = vec![1.0_f32; 4];
        let sigma = estimate_noise_sigma(&pixels, 2, 2);
        assert_eq!(sigma, 0.0);
    }

    #[test]
    fn test_estimate_noise_sigma_noisy_image() {
        // Alternating checkerboard introduces strong Laplacian response
        let pixels: Vec<f32> = (0..25_usize)
            .map(|i| if (i / 5 + i % 5) % 2 == 0 { 1.0 } else { 0.0 })
            .collect();
        let sigma = estimate_noise_sigma(&pixels, 5, 5);
        assert!(
            sigma > 0.0,
            "noisy image should have sigma > 0, got {sigma}"
        );
    }

    // -----------------------------------------------------------------------
    // estimate_noise_profile
    // -----------------------------------------------------------------------

    #[test]
    fn test_estimate_noise_profile_zero_noise() {
        let pixels = vec![0.5_f32; 25];
        let profile = estimate_noise_profile(&pixels, 5, 5);
        // Very low sigma → very high SNR
        assert!(
            profile.snr_db > 60.0,
            "constant image SNR should be very high, got {} dB",
            profile.snr_db
        );
    }

    #[test]
    fn test_estimate_noise_profile_noise_type_present() {
        // Any image should produce a valid noise type
        let pixels: Vec<f32> = (0..64_usize).map(|i| i as f32 / 64.0).collect();
        let profile = estimate_noise_profile(&pixels, 8, 8);
        let _ = profile.noise_type; // just ensure it does not panic
        assert!(profile.sigma >= 0.0);
    }

    #[test]
    fn test_estimate_noise_profile_snr_bounded() {
        let pixels = vec![0.3_f32; 16];
        let profile = estimate_noise_profile(&pixels, 4, 4);
        assert!(profile.snr_db >= -60.0 && profile.snr_db <= 120.0);
    }

    // -----------------------------------------------------------------------
    // denoise_median
    // -----------------------------------------------------------------------

    #[test]
    fn test_denoise_median_radius_0_passthrough() {
        let pixels: Vec<u8> = (0..25).collect();
        let result = denoise_median(&pixels, 5, 5, 0);
        assert_eq!(result, pixels);
    }

    #[test]
    fn test_denoise_median_uniform_image() {
        let pixels = vec![128u8; 25];
        let result = denoise_median(&pixels, 5, 5, 1);
        assert!(
            result.iter().all(|&v| v == 128),
            "median of uniform image should be unchanged"
        );
    }

    #[test]
    fn test_denoise_median_removes_impulse() {
        // Place a salt pixel in an otherwise uniform image
        let mut pixels = vec![100u8; 25];
        pixels[12] = 255; // impulse at centre
        let result = denoise_median(&pixels, 5, 5, 1);
        // After median filtering the impulse should be suppressed
        assert!(
            result[12] < 255,
            "median should suppress impulse, got {}",
            result[12]
        );
    }

    #[test]
    fn test_denoise_median_output_length() {
        let pixels = vec![50u8; 9];
        let result = denoise_median(&pixels, 3, 3, 1);
        assert_eq!(result.len(), 9);
    }

    // -----------------------------------------------------------------------
    // denoise_bilateral
    // -----------------------------------------------------------------------

    #[test]
    fn test_denoise_bilateral_uniform_image() {
        let pixels = vec![0.5_f32; 25];
        let result = denoise_bilateral(&pixels, 5, 5, 1.0, 0.2);
        for (orig, filtered) in pixels.iter().zip(result.iter()) {
            assert!(
                (orig - filtered).abs() < 1e-5,
                "bilateral of uniform image should be unchanged"
            );
        }
    }

    #[test]
    fn test_denoise_bilateral_output_length() {
        let pixels: Vec<f32> = (0..16).map(|i| i as f32 / 16.0).collect();
        let result = denoise_bilateral(&pixels, 4, 4, 1.0, 0.1);
        assert_eq!(result.len(), 16);
    }

    #[test]
    fn test_denoise_bilateral_preserves_range() {
        let pixels: Vec<f32> = (0..25).map(|i| i as f32 / 25.0).collect();
        let result = denoise_bilateral(&pixels, 5, 5, 1.5, 0.3);
        // Output should stay within the original min/max range (±small tolerance)
        let orig_min = pixels.iter().cloned().fold(f32::INFINITY, f32::min);
        let orig_max = pixels.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        for &v in &result {
            assert!(
                v >= orig_min - 1e-4 && v <= orig_max + 1e-4,
                "bilateral output out of range: {v}"
            );
        }
    }

    // -----------------------------------------------------------------------
    // denoise_nlm
    // -----------------------------------------------------------------------

    #[test]
    fn test_denoise_nlm_uniform_image() {
        let pixels = vec![0.7_f32; 25];
        let result = denoise_nlm(&pixels, 5, 5, 0.1, 3, 5);
        for (orig, filtered) in pixels.iter().zip(result.iter()) {
            assert!(
                (orig - filtered).abs() < 1e-4,
                "NLM of uniform image should be unchanged"
            );
        }
    }

    #[test]
    fn test_denoise_nlm_output_length() {
        let pixels: Vec<f32> = (0..9).map(|i| i as f32 / 9.0).collect();
        let result = denoise_nlm(&pixels, 3, 3, 0.1, 1, 3);
        assert_eq!(result.len(), 9);
    }

    #[test]
    fn test_denoise_nlm_reduces_noise_energy() {
        // Create a noisy image (alternating high/low) and verify NLM reduces energy
        let pixels: Vec<f32> = (0..25_usize)
            .map(|i| if i % 2 == 0 { 1.0 } else { 0.0 })
            .collect();
        let result = denoise_nlm(&pixels, 5, 5, 0.5, 3, 5);
        // The variance of the output should be less than the variance of the input
        let mean_in: f32 = pixels.iter().sum::<f32>() / pixels.len() as f32;
        let var_in: f32 = pixels.iter().map(|&v| (v - mean_in).powi(2)).sum::<f32>();
        let mean_out: f32 = result.iter().sum::<f32>() / result.len() as f32;
        let var_out: f32 = result.iter().map(|&v| (v - mean_out).powi(2)).sum::<f32>();
        assert!(
            var_out < var_in,
            "NLM should reduce variance: in={var_in:.4}, out={var_out:.4}"
        );
    }

    // -----------------------------------------------------------------------
    // Internal helpers
    // -----------------------------------------------------------------------

    #[test]
    fn test_median_f32_odd() {
        let mut v = vec![3.0_f32, 1.0, 2.0];
        assert!((median_f32(&mut v) - 2.0).abs() < 1e-6);
    }

    #[test]
    fn test_median_f32_even() {
        let mut v = vec![4.0_f32, 1.0, 3.0, 2.0];
        assert!((median_f32(&mut v) - 2.5).abs() < 1e-6);
    }

    #[test]
    fn test_box_blur_3_constant() {
        let pixels = vec![0.6_f32; 9];
        let blurred = box_blur_3(&pixels, 3, 3);
        for v in blurred {
            assert!((v - 0.6).abs() < 1e-6);
        }
    }
}
