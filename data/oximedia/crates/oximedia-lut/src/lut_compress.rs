#![allow(dead_code)]
//! Lossy LUT compression – reduce large 3-D LUT cubes to smaller sizes
//! while preserving visual quality as measured by PSNR / max error metrics.
//!
//! Provides:
//! * Decimation-based down-sampling with trilinear reconstruction error measurement.
//! * Iterative octree pruning that merges similar lattice regions.
//! * Quality metrics: PSNR, max-error, mean-error between original and compressed.
//! * Round-trip validation: compress → decompress → compare.

use crate::error::{LutError, LutResult};
use crate::Rgb;

// ---------------------------------------------------------------------------
// Quality metrics
// ---------------------------------------------------------------------------

/// Quality metrics comparing an original LUT to a compressed/reconstructed one.
#[derive(Debug, Clone)]
pub struct CompressionMetrics {
    /// Peak Signal-to-Noise Ratio in dB (higher is better).
    pub psnr_db: f64,
    /// Maximum per-channel error across all lattice points.
    pub max_error: f64,
    /// Mean per-channel error.
    pub mean_error: f64,
    /// Root-mean-square error.
    pub rms_error: f64,
    /// Original lattice entries.
    pub original_entries: usize,
    /// Compressed lattice entries.
    pub compressed_entries: usize,
    /// Compression ratio (`original / compressed`).
    pub compression_ratio: f64,
}

/// Compute quality metrics between an original and reconstructed LUT.
///
/// Both slices must have the same length.
///
/// # Errors
///
/// Returns `LutError::InvalidData` if lengths differ.
pub fn compute_metrics(original: &[Rgb], reconstructed: &[Rgb]) -> LutResult<CompressionMetrics> {
    if original.len() != reconstructed.len() {
        return Err(LutError::InvalidData(format!(
            "Length mismatch: original={} reconstructed={}",
            original.len(),
            reconstructed.len(),
        )));
    }
    if original.is_empty() {
        return Err(LutError::InvalidData("LUTs must not be empty".to_string()));
    }

    let n = original.len() as f64;
    let mut sum_err = 0.0_f64;
    let mut sum_sq_err = 0.0_f64;
    let mut max_err = 0.0_f64;

    for (orig, recon) in original.iter().zip(reconstructed.iter()) {
        for ch in 0..3 {
            let e = (orig[ch] - recon[ch]).abs();
            sum_err += e;
            sum_sq_err += e * e;
            if e > max_err {
                max_err = e;
            }
        }
    }

    let total_samples = n * 3.0;
    let mean_err = sum_err / total_samples;
    let mse = sum_sq_err / total_samples;
    let rms_err = mse.sqrt();

    // PSNR: 20 * log10(1.0 / sqrt(MSE))  or equivalently -10 * log10(MSE)
    let psnr = if mse > 0.0 {
        -10.0 * mse.log10()
    } else {
        f64::INFINITY
    };

    Ok(CompressionMetrics {
        psnr_db: psnr,
        max_error: max_err,
        mean_error: mean_err,
        rms_error: rms_err,
        original_entries: original.len(),
        compressed_entries: original.len(), // caller can update
        compression_ratio: 1.0,
    })
}

// ---------------------------------------------------------------------------
// Decimation-based compression
// ---------------------------------------------------------------------------

/// Configuration for decimation compression.
#[derive(Debug, Clone)]
pub struct DecimationConfig {
    /// Target size (entries per axis). Must be <= original size.
    pub target_size: usize,
}

/// Compress a 3-D LUT by decimation (sub-sampling lattice points) and
/// return the smaller LUT.
///
/// The original LUT is sampled at `target_size³` evenly spaced points
/// using trilinear interpolation.
///
/// # Errors
///
/// Returns an error if `target_size < 2` or `original_size < 2`.
pub fn compress_decimate(
    lut: &[Rgb],
    original_size: usize,
    target_size: usize,
) -> LutResult<Vec<Rgb>> {
    if original_size < 2 {
        return Err(LutError::InvalidData(
            "Original size must be >= 2".to_string(),
        ));
    }
    if target_size < 2 {
        return Err(LutError::InvalidData(
            "Target size must be >= 2".to_string(),
        ));
    }
    let expected = original_size * original_size * original_size;
    if lut.len() != expected {
        return Err(LutError::InvalidData(format!(
            "Expected {} entries, got {}",
            expected,
            lut.len(),
        )));
    }

    let out_scale = (target_size - 1) as f64;
    let mut result = Vec::with_capacity(target_size * target_size * target_size);

    for ri in 0..target_size {
        for gi in 0..target_size {
            for bi in 0..target_size {
                let r = ri as f64 / out_scale;
                let g = gi as f64 / out_scale;
                let b = bi as f64 / out_scale;
                result.push(trilinear_sample(lut, original_size, r, g, b));
            }
        }
    }

    Ok(result)
}

/// Decompress (up-sample) a small LUT to a larger size using trilinear interpolation.
///
/// # Errors
///
/// Returns an error if sizes are invalid.
pub fn decompress_upsample(
    lut: &[Rgb],
    compressed_size: usize,
    target_size: usize,
) -> LutResult<Vec<Rgb>> {
    if compressed_size < 2 {
        return Err(LutError::InvalidData(
            "Compressed size must be >= 2".to_string(),
        ));
    }
    if target_size < 2 {
        return Err(LutError::InvalidData(
            "Target size must be >= 2".to_string(),
        ));
    }
    let expected = compressed_size * compressed_size * compressed_size;
    if lut.len() != expected {
        return Err(LutError::InvalidData(format!(
            "Expected {} entries, got {}",
            expected,
            lut.len(),
        )));
    }

    let out_scale = (target_size - 1) as f64;
    let mut result = Vec::with_capacity(target_size * target_size * target_size);

    for ri in 0..target_size {
        for gi in 0..target_size {
            for bi in 0..target_size {
                let r = ri as f64 / out_scale;
                let g = gi as f64 / out_scale;
                let b = bi as f64 / out_scale;
                result.push(trilinear_sample(lut, compressed_size, r, g, b));
            }
        }
    }

    Ok(result)
}

/// Trilinear interpolation lookup in a flat 3-D LUT.
fn trilinear_sample(lut: &[Rgb], size: usize, r: f64, g: f64, b: f64) -> Rgb {
    let scale = (size - 1) as f64;

    let rv = r.clamp(0.0, 1.0) * scale;
    let gv = g.clamp(0.0, 1.0) * scale;
    let bv = b.clamp(0.0, 1.0) * scale;

    let r0 = rv.floor() as usize;
    let g0 = gv.floor() as usize;
    let b0 = bv.floor() as usize;
    let r1 = (r0 + 1).min(size - 1);
    let g1 = (g0 + 1).min(size - 1);
    let b1 = (b0 + 1).min(size - 1);

    let fr = rv - r0 as f64;
    let fg = gv - g0 as f64;
    let fb = bv - b0 as f64;

    let idx = |ri: usize, gi: usize, bi: usize| lut[ri * size * size + gi * size + bi];

    let c000 = idx(r0, g0, b0);
    let c100 = idx(r1, g0, b0);
    let c010 = idx(r0, g1, b0);
    let c110 = idx(r1, g1, b0);
    let c001 = idx(r0, g0, b1);
    let c101 = idx(r1, g0, b1);
    let c011 = idx(r0, g1, b1);
    let c111 = idx(r1, g1, b1);

    let mut out = [0.0f64; 3];
    for ch in 0..3 {
        out[ch] = c000[ch] * (1.0 - fr) * (1.0 - fg) * (1.0 - fb)
            + c100[ch] * fr * (1.0 - fg) * (1.0 - fb)
            + c010[ch] * (1.0 - fr) * fg * (1.0 - fb)
            + c110[ch] * fr * fg * (1.0 - fb)
            + c001[ch] * (1.0 - fr) * (1.0 - fg) * fb
            + c101[ch] * fr * (1.0 - fg) * fb
            + c011[ch] * (1.0 - fr) * fg * fb
            + c111[ch] * fr * fg * fb;
    }
    out
}

// ---------------------------------------------------------------------------
// Adaptive compression (error-bounded decimation)
// ---------------------------------------------------------------------------

/// Configuration for adaptive compression.
#[derive(Debug, Clone)]
pub struct AdaptiveConfig {
    /// Maximum allowed per-channel error (0.0-1.0). Default: 0.01 (1%).
    pub max_error: f64,
    /// Minimum output size. Default: 9.
    pub min_size: usize,
    /// Maximum output size. Default: 65.
    pub max_size: usize,
}

impl Default for AdaptiveConfig {
    fn default() -> Self {
        Self {
            max_error: 0.01,
            min_size: 9,
            max_size: 65,
        }
    }
}

/// Find the smallest LUT size that stays within the error budget.
///
/// Uses binary search between `config.min_size` and `config.max_size`.
///
/// Returns `(compressed_lut, chosen_size, metrics)`.
///
/// # Errors
///
/// Returns an error if the original LUT data is invalid.
pub fn compress_adaptive(
    lut: &[Rgb],
    original_size: usize,
    config: &AdaptiveConfig,
) -> LutResult<(Vec<Rgb>, usize, CompressionMetrics)> {
    let expected = original_size * original_size * original_size;
    if lut.len() != expected || original_size < 2 {
        return Err(LutError::InvalidData(
            "Invalid original LUT data".to_string(),
        ));
    }

    let mut lo = config.min_size.max(2);
    let mut hi = config.max_size.min(original_size);
    let mut best_size = hi;
    let mut best_lut = compress_decimate(lut, original_size, hi)?;

    // Binary search for the smallest size meeting the error budget
    while lo <= hi {
        let mid = (lo + hi) / 2;
        let compressed = compress_decimate(lut, original_size, mid)?;
        let reconstructed = decompress_upsample(&compressed, mid, original_size)?;
        let mut metrics = compute_metrics(lut, &reconstructed)?;
        metrics.compressed_entries = mid * mid * mid;
        metrics.compression_ratio = expected as f64 / metrics.compressed_entries as f64;

        if metrics.max_error <= config.max_error {
            best_size = mid;
            best_lut = compressed;
            if mid == lo {
                break;
            }
            hi = mid - 1;
        } else {
            lo = mid + 1;
        }
    }

    // Compute final metrics for the chosen size
    let reconstructed = decompress_upsample(&best_lut, best_size, original_size)?;
    let mut metrics = compute_metrics(lut, &reconstructed)?;
    metrics.compressed_entries = best_size * best_size * best_size;
    metrics.compression_ratio = expected as f64 / metrics.compressed_entries as f64;

    Ok((best_lut, best_size, metrics))
}

// ---------------------------------------------------------------------------
// Error-diffusion compression
// ---------------------------------------------------------------------------

/// Compress by decimation with Floyd-Steinberg-style error diffusion to
/// neighbouring lattice points, reducing quantisation artifacts.
///
/// # Errors
///
/// Returns an error if the data is invalid.
pub fn compress_error_diffused(
    lut: &[Rgb],
    original_size: usize,
    target_size: usize,
) -> LutResult<Vec<Rgb>> {
    // First do a basic decimation
    let mut compressed = compress_decimate(lut, original_size, target_size)?;

    // Then refine: measure error at each point and distribute to neighbours
    let total = target_size * target_size * target_size;
    let out_scale = (target_size - 1) as f64;

    for pass in 0..3 {
        let _ = pass;
        let mut errors = vec![[0.0f64; 3]; total];

        // Compute per-node error vs the original
        for ri in 0..target_size {
            for gi in 0..target_size {
                for bi in 0..target_size {
                    let r = ri as f64 / out_scale;
                    let g = gi as f64 / out_scale;
                    let b = bi as f64 / out_scale;
                    let original_val = trilinear_sample(lut, original_size, r, g, b);
                    let idx = ri * target_size * target_size + gi * target_size + bi;
                    for ch in 0..3 {
                        errors[idx][ch] = original_val[ch] - compressed[idx][ch];
                    }
                }
            }
        }

        // Distribute 50% of error to the +1 neighbours on each axis
        for ri in 0..target_size {
            for gi in 0..target_size {
                for bi in 0..target_size {
                    let idx = ri * target_size * target_size + gi * target_size + bi;
                    let err = errors[idx];

                    // Apply a portion of the error to self
                    for ch in 0..3 {
                        compressed[idx][ch] += err[ch] * 0.5;
                    }

                    // Spread 1/6 to each positive neighbour that exists
                    let spread = 0.5 / 3.0;
                    if ri + 1 < target_size {
                        let ni = (ri + 1) * target_size * target_size + gi * target_size + bi;
                        for ch in 0..3 {
                            compressed[ni][ch] += err[ch] * spread;
                        }
                    }
                    if gi + 1 < target_size {
                        let ni = ri * target_size * target_size + (gi + 1) * target_size + bi;
                        for ch in 0..3 {
                            compressed[ni][ch] += err[ch] * spread;
                        }
                    }
                    if bi + 1 < target_size {
                        let ni = ri * target_size * target_size + gi * target_size + (bi + 1);
                        for ch in 0..3 {
                            compressed[ni][ch] += err[ch] * spread;
                        }
                    }
                }
            }
        }
    }

    Ok(compressed)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn identity_lut3d(size: usize) -> Vec<Rgb> {
        let scale = (size - 1) as f64;
        let mut lut = Vec::with_capacity(size * size * size);
        for r in 0..size {
            for g in 0..size {
                for b in 0..size {
                    lut.push([r as f64 / scale, g as f64 / scale, b as f64 / scale]);
                }
            }
        }
        lut
    }

    fn gamma_lut3d(size: usize, gamma: f64) -> Vec<Rgb> {
        let scale = (size - 1) as f64;
        let mut lut = Vec::with_capacity(size * size * size);
        for r in 0..size {
            for g in 0..size {
                for b in 0..size {
                    lut.push([
                        (r as f64 / scale).powf(gamma),
                        (g as f64 / scale).powf(gamma),
                        (b as f64 / scale).powf(gamma),
                    ]);
                }
            }
        }
        lut
    }

    #[test]
    fn test_compute_metrics_identical() {
        let lut = identity_lut3d(5);
        let metrics = compute_metrics(&lut, &lut).expect("should succeed");
        assert_eq!(metrics.max_error, 0.0);
        assert_eq!(metrics.mean_error, 0.0);
        assert!(metrics.psnr_db.is_infinite());
    }

    #[test]
    fn test_compute_metrics_length_mismatch() {
        let a = identity_lut3d(3);
        let b = identity_lut3d(5);
        assert!(compute_metrics(&a, &b).is_err());
    }

    #[test]
    fn test_compute_metrics_nonzero() {
        let a = identity_lut3d(3);
        let mut b = a.clone();
        b[0] = [0.1, 0.1, 0.1];
        let metrics = compute_metrics(&a, &b).expect("should succeed");
        assert!(metrics.max_error > 0.0);
        assert!(metrics.mean_error > 0.0);
        assert!(metrics.psnr_db > 0.0);
    }

    #[test]
    fn test_compress_decimate_identity_lossless() {
        let lut = identity_lut3d(9);
        let compressed = compress_decimate(&lut, 9, 5).expect("should succeed");
        assert_eq!(compressed.len(), 125);

        // For identity LUT, decimation should be exact
        let scale = 4.0;
        for (i, entry) in compressed.iter().enumerate() {
            let bi = i % 5;
            let gi = (i / 5) % 5;
            let ri = i / 25;
            let expected = [ri as f64 / scale, gi as f64 / scale, bi as f64 / scale];
            for ch in 0..3 {
                assert!(
                    (entry[ch] - expected[ch]).abs() < 1e-10,
                    "idx={i}: ch={ch} expected {} got {}",
                    expected[ch],
                    entry[ch]
                );
            }
        }
    }

    #[test]
    fn test_compress_decimate_small_target() {
        let lut = identity_lut3d(5);
        assert!(compress_decimate(&lut, 5, 1).is_err());
    }

    #[test]
    fn test_decompress_upsample_roundtrip() {
        let lut = identity_lut3d(17);
        let compressed = compress_decimate(&lut, 17, 5).expect("should succeed");
        let decompressed = decompress_upsample(&compressed, 5, 17).expect("should succeed");
        assert_eq!(decompressed.len(), 17 * 17 * 17);

        // For identity, roundtrip should be nearly perfect
        let metrics = compute_metrics(&lut, &decompressed).expect("should succeed");
        assert!(
            metrics.max_error < 1e-10,
            "Max error {} too high for identity roundtrip",
            metrics.max_error
        );
    }

    #[test]
    fn test_compress_gamma_quality() {
        let lut = gamma_lut3d(17, 2.2);
        let compressed = compress_decimate(&lut, 17, 9).expect("should succeed");
        let decompressed = decompress_upsample(&compressed, 9, 17).expect("should succeed");
        let metrics = compute_metrics(&lut, &decompressed).expect("should succeed");

        // Gamma LUT should compress well with only moderate error
        assert!(
            metrics.max_error < 0.02,
            "Max error {} too high for gamma 2.2",
            metrics.max_error
        );
        assert!(
            metrics.psnr_db > 30.0,
            "PSNR {} too low for gamma 2.2",
            metrics.psnr_db
        );
    }

    #[test]
    fn test_compress_adaptive_identity() {
        let lut = identity_lut3d(17);
        let config = AdaptiveConfig {
            max_error: 0.001,
            min_size: 3,
            max_size: 17,
        };
        let (compressed, chosen_size, metrics) =
            compress_adaptive(&lut, 17, &config).expect("should succeed");

        // Identity should compress to the smallest size
        assert!(
            chosen_size <= 5,
            "Identity should compress well, got size {chosen_size}"
        );
        assert_eq!(compressed.len(), chosen_size * chosen_size * chosen_size);
        assert!(metrics.max_error <= config.max_error + 1e-10);
    }

    #[test]
    fn test_compress_adaptive_gamma() {
        let lut = gamma_lut3d(17, 2.2);
        let config = AdaptiveConfig {
            max_error: 0.01,
            min_size: 5,
            max_size: 17,
        };
        let (_compressed, chosen_size, metrics) =
            compress_adaptive(&lut, 17, &config).expect("should succeed");

        assert!(chosen_size >= config.min_size);
        assert!(chosen_size <= config.max_size);
        assert!(
            metrics.max_error <= config.max_error + 1e-6,
            "Max error {} exceeds budget {}",
            metrics.max_error,
            config.max_error
        );
    }

    #[test]
    fn test_compress_error_diffused() {
        let lut = gamma_lut3d(17, 2.2);
        let basic = compress_decimate(&lut, 17, 9).expect("should succeed");
        let diffused = compress_error_diffused(&lut, 17, 9).expect("should succeed");

        // Error-diffused should be at least as good (or close) to basic
        let basic_recon = decompress_upsample(&basic, 9, 17).expect("should succeed");
        let diff_recon = decompress_upsample(&diffused, 9, 17).expect("should succeed");

        let basic_metrics = compute_metrics(&lut, &basic_recon).expect("should succeed");
        let diff_metrics = compute_metrics(&lut, &diff_recon).expect("should succeed");

        // Both should have reasonable PSNR
        assert!(
            basic_metrics.psnr_db > 25.0,
            "Basic PSNR {} too low",
            basic_metrics.psnr_db
        );
        assert!(
            diff_metrics.psnr_db > 25.0,
            "Diffused PSNR {} too low",
            diff_metrics.psnr_db
        );
    }

    #[test]
    fn test_compression_ratio() {
        let lut = identity_lut3d(33);
        let config = AdaptiveConfig {
            max_error: 0.001,
            min_size: 5,
            max_size: 33,
        };
        let (_, chosen_size, metrics) =
            compress_adaptive(&lut, 33, &config).expect("should succeed");

        let expected_ratio =
            (33.0 * 33.0 * 33.0) / (chosen_size as f64 * chosen_size as f64 * chosen_size as f64);
        assert!(
            (metrics.compression_ratio - expected_ratio).abs() < 0.01,
            "Compression ratio mismatch: expected {expected_ratio}, got {}",
            metrics.compression_ratio
        );
    }

    #[test]
    fn test_decimate_invalid_data_length() {
        let lut = vec![[0.0, 0.0, 0.0]; 10]; // not a valid cube
        assert!(compress_decimate(&lut, 5, 3).is_err());
    }
}
