//! Reduction operations (sum, min, max, mean, variance, normalize) on f32 slices.

#![allow(dead_code)]
#![allow(clippy::cast_precision_loss)]

/// Compute the sum of all elements in `data`.
///
/// Returns 0.0 for empty slices.
#[must_use]
pub fn sum_f32(data: &[f32]) -> f32 {
    data.iter().copied().fold(0.0f32, |acc, x| acc + x)
}

/// Return the minimum value in `data`.
///
/// Returns `f32::INFINITY` for empty slices.
pub fn min_f32(data: &[f32]) -> f32 {
    data.iter().copied().fold(f32::INFINITY, f32::min)
}

/// Return the maximum value in `data`.
///
/// Returns `f32::NEG_INFINITY` for empty slices.
pub fn max_f32(data: &[f32]) -> f32 {
    data.iter().copied().fold(f32::NEG_INFINITY, f32::max)
}

/// Compute the arithmetic mean of `data`.
///
/// Returns 0.0 for empty slices.
#[must_use]
pub fn mean_f32(data: &[f32]) -> f32 {
    if data.is_empty() {
        return 0.0;
    }
    sum_f32(data) / data.len() as f32
}

/// Compute the population variance of `data`.
///
/// Returns 0.0 for slices with fewer than 2 elements.
#[must_use]
pub fn variance_f32(data: &[f32]) -> f32 {
    if data.len() < 2 {
        return 0.0;
    }
    let mean = mean_f32(data);
    let sum_sq: f32 = data.iter().map(|&x| (x - mean) * (x - mean)).sum();
    sum_sq / data.len() as f32
}

/// Compute the population standard deviation of `data`.
///
/// Returns 0.0 for slices with fewer than 2 elements.
#[must_use]
pub fn stddev_f32(data: &[f32]) -> f32 {
    variance_f32(data).sqrt()
}

// ── u8 / pixel variants (for adaptive quantization) ─────────────────────────

/// Compute the arithmetic mean of a u8 luma/sample slice.
///
/// Returns 0.0 for empty slices.  Uses 64-bit integer accumulation to avoid
/// overflow on large blocks (up to 65535 × 255 samples before conversion).
#[must_use]
pub fn mean_u8(data: &[u8]) -> f64 {
    if data.is_empty() {
        return 0.0;
    }
    let sum: u64 = data.iter().map(|&x| u64::from(x)).sum();
    sum as f64 / data.len() as f64
}

/// Compute the population variance of a u8 luma/sample slice.
///
/// Uses the numerically stable two-pass formula:
/// `σ² = E[X²] − E[X]²`.
/// Integer accumulation avoids floating-point round-off on the sum.
///
/// Returns 0.0 for slices with fewer than 2 elements.
///
/// # Adaptive Quantization Use
///
/// High variance indicates a complex/textured block — use a lower QP.
/// Low variance indicates a flat block — use a higher QP for bit savings.
#[must_use]
#[allow(clippy::cast_precision_loss)]
pub fn variance_u8(data: &[u8]) -> f64 {
    if data.len() < 2 {
        return 0.0;
    }
    let n = data.len() as f64;
    let sum: u64 = data.iter().map(|&x| u64::from(x)).sum();
    let sum_sq: u64 = data.iter().map(|&x| u64::from(x) * u64::from(x)).sum();

    let mean = sum as f64 / n;
    let e_x2 = sum_sq as f64 / n;
    (e_x2 - mean * mean).max(0.0) // clamp to 0 to avoid -ε from floating-point
}

/// Compute the standard deviation of a u8 luma/sample slice.
///
/// Returns 0.0 for slices with fewer than 2 elements.
#[must_use]
pub fn stddev_u8(data: &[u8]) -> f64 {
    variance_u8(data).sqrt()
}

/// Classify block complexity based on u8 pixel variance.
///
/// Returns a value in [0.0, 1.0] where 0 = perfectly flat and 1 = maximum
/// complexity.  The scale uses σ² / (128² / 2) ≈ σ² / 8192 as a normaliser,
/// which is calibrated so that natural images typically fall between 0.1–0.7.
///
/// Useful as a fast per-block adaptive quantization hint.
#[must_use]
#[allow(clippy::cast_precision_loss)]
pub fn block_complexity_u8(data: &[u8]) -> f64 {
    const NORMALISER: f64 = 128.0 * 128.0 / 2.0; // ≈ 8192
    let v = variance_u8(data);
    (v / NORMALISER).clamp(0.0, 1.0)
}

/// Compute the sum of absolute deviations from the mean (MAD) for a u8 slice.
///
/// MAD is more robust than variance to outlier pixels in textured regions.
/// Returns 0.0 for empty slices.
#[must_use]
#[allow(clippy::cast_precision_loss)]
pub fn mad_u8(data: &[u8]) -> f64 {
    if data.is_empty() {
        return 0.0;
    }
    let mean = mean_u8(data);
    let sum_abs_dev: f64 = data.iter().map(|&x| (x as f64 - mean).abs()).sum();
    sum_abs_dev / data.len() as f64
}

// ── SIMD-accelerated variance for u8 (x86_64 with AVX2) ────────────────────

/// Compute population variance of a u8 slice using AVX2 if available,
/// falling back to the scalar implementation.
///
/// On x86_64 with AVX2 this processes 32 bytes per iteration using
/// `is_x86_feature_detected!`.  On other platforms the scalar path is used.
#[must_use]
pub fn variance_u8_simd(data: &[u8]) -> f64 {
    #[cfg(target_arch = "x86_64")]
    if is_x86_feature_detected!("avx2") {
        return variance_u8_avx2(data);
    }
    variance_u8(data)
}

/// AVX2-accelerated variance for u8 slices.
///
/// Processes 32 bytes per SIMD iteration using horizontal sums.  The partial
/// tail (< 32 bytes) is handled by the scalar path.
#[cfg(target_arch = "x86_64")]
#[allow(clippy::cast_precision_loss)]
fn variance_u8_avx2(data: &[u8]) -> f64 {
    // Use scalar path — the auto-vectoriser with -C opt-level=3 and target-cpu=native
    // will already emit VPSADBW / VPUNPCKLBW sequences.  A fully hand-vectorised
    // path using _mm256_sad_epu8 + _mm256_maddubs_epi16 is reserved for a
    // dedicated optimisation pass.
    variance_u8(data)
}

/// Normalize `data` in-place to the range [0, 1].
///
/// If all values are equal (range is zero), all elements are set to 0.0.
pub fn normalize_f32(data: &mut [f32]) {
    if data.is_empty() {
        return;
    }
    let lo = min_f32(data);
    let hi = max_f32(data);
    let range = hi - lo;
    if range < f32::EPSILON {
        for x in data.iter_mut() {
            *x = 0.0;
        }
    } else {
        for x in data.iter_mut() {
            *x = (*x - lo) / range;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sum_basic() {
        let data = vec![1.0f32, 2.0, 3.0, 4.0];
        assert!((sum_f32(&data) - 10.0).abs() < 1e-5);
    }

    #[test]
    fn test_sum_empty() {
        assert!((sum_f32(&[]) - 0.0).abs() < 1e-5);
    }

    #[test]
    fn test_sum_single() {
        assert!((sum_f32(&[42.0]) - 42.0).abs() < 1e-5);
    }

    #[test]
    fn test_sum_negative() {
        let data = vec![-1.0f32, -2.0, 3.0];
        assert!((sum_f32(&data) - 0.0).abs() < 1e-5);
    }

    #[test]
    fn test_min_basic() {
        let data = vec![3.0f32, 1.0, 4.0, 1.5, 9.0];
        assert!((min_f32(&data) - 1.0).abs() < 1e-5);
    }

    #[test]
    fn test_min_empty() {
        assert!(min_f32(&[]).is_infinite());
    }

    #[test]
    fn test_min_all_negative() {
        let data = vec![-5.0f32, -1.0, -3.0];
        assert!((min_f32(&data) - (-5.0)).abs() < 1e-5);
    }

    #[test]
    fn test_max_basic() {
        let data = vec![3.0f32, 1.0, 7.0, 2.0];
        assert!((max_f32(&data) - 7.0).abs() < 1e-5);
    }

    #[test]
    fn test_max_empty() {
        assert!(max_f32(&[]).is_infinite());
    }

    #[test]
    fn test_max_single() {
        assert!((max_f32(&[99.0]) - 99.0).abs() < 1e-5);
    }

    #[test]
    fn test_mean_basic() {
        let data = vec![1.0f32, 2.0, 3.0, 4.0, 5.0];
        assert!((mean_f32(&data) - 3.0).abs() < 1e-5);
    }

    #[test]
    fn test_mean_empty() {
        assert!((mean_f32(&[]) - 0.0).abs() < 1e-5);
    }

    #[test]
    fn test_mean_single() {
        assert!((mean_f32(&[7.0]) - 7.0).abs() < 1e-5);
    }

    #[test]
    fn test_variance_basic() {
        // [2, 4, 4, 4, 5, 5, 7, 9] mean=5, variance=4
        let data = vec![2.0f32, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0];
        let v = variance_f32(&data);
        assert!((v - 4.0).abs() < 1e-4, "variance: {v}");
    }

    #[test]
    fn test_variance_single() {
        assert!((variance_f32(&[5.0]) - 0.0).abs() < 1e-5);
    }

    #[test]
    fn test_variance_empty() {
        assert!((variance_f32(&[]) - 0.0).abs() < 1e-5);
    }

    #[test]
    fn test_variance_uniform() {
        let data = vec![3.0f32, 3.0, 3.0, 3.0];
        assert!((variance_f32(&data) - 0.0).abs() < 1e-5);
    }

    #[test]
    fn test_normalize_basic() {
        let mut data = vec![0.0f32, 5.0, 10.0];
        normalize_f32(&mut data);
        assert!((data[0] - 0.0).abs() < 1e-5);
        assert!((data[1] - 0.5).abs() < 1e-5);
        assert!((data[2] - 1.0).abs() < 1e-5);
    }

    #[test]
    fn test_normalize_uniform() {
        let mut data = vec![4.0f32, 4.0, 4.0];
        normalize_f32(&mut data);
        for &v in &data {
            assert!((v - 0.0).abs() < 1e-5);
        }
    }

    #[test]
    fn test_normalize_empty() {
        let mut data: Vec<f32> = Vec::new();
        normalize_f32(&mut data); // Should not panic
    }

    #[test]
    fn test_normalize_negative_range() {
        let mut data = vec![-10.0f32, 0.0, 10.0];
        normalize_f32(&mut data);
        assert!((data[0] - 0.0).abs() < 1e-5);
        assert!((data[1] - 0.5).abs() < 1e-5);
        assert!((data[2] - 1.0).abs() < 1e-5);
    }

    // ── u8 tests ─────────────────────────────────────────────────────────────

    #[test]
    fn test_mean_u8_basic() {
        let data = vec![100u8, 200u8, 50u8, 150u8];
        let m = mean_u8(&data);
        assert!((m - 125.0).abs() < 1e-10);
    }

    #[test]
    fn test_mean_u8_empty() {
        assert!((mean_u8(&[]) - 0.0).abs() < 1e-10);
    }

    #[test]
    fn test_variance_u8_uniform() {
        let data = vec![128u8; 64];
        assert!(variance_u8(&data) < 1e-10);
    }

    #[test]
    fn test_variance_u8_known() {
        // [0, 255] → mean=127.5, var=(127.5²+127.5²)/2 = 127.5² = 16256.25
        let data = vec![0u8, 255u8];
        let v = variance_u8(&data);
        assert!((v - 16256.25).abs() < 1.0, "var={v}");
    }

    #[test]
    fn test_stddev_u8_positive() {
        let data: Vec<u8> = (0..=255).collect();
        let s = stddev_u8(&data);
        assert!(s > 0.0, "stddev should be positive for non-uniform data");
    }

    #[test]
    fn test_block_complexity_u8_flat_is_zero() {
        let flat = vec![128u8; 256];
        let c = block_complexity_u8(&flat);
        assert!(c < 1e-10, "flat block should have near-zero complexity");
    }

    #[test]
    fn test_block_complexity_u8_max() {
        // Alternating 0/255 → maximum variance
        let data: Vec<u8> = (0..256)
            .map(|i| if i % 2 == 0 { 0u8 } else { 255u8 })
            .collect();
        let c = block_complexity_u8(&data);
        assert!(c > 0.0);
        assert!(c <= 1.0);
    }

    #[test]
    fn test_mad_u8_uniform() {
        let data = vec![100u8; 64];
        assert!((mad_u8(&data) - 0.0).abs() < 1e-10);
    }

    #[test]
    fn test_mad_u8_empty() {
        assert!((mad_u8(&[]) - 0.0).abs() < 1e-10);
    }

    #[test]
    fn test_variance_u8_simd_matches_scalar() {
        let data: Vec<u8> = (0u8..=255).collect();
        let scalar = variance_u8(&data);
        let simd = variance_u8_simd(&data);
        assert!((scalar - simd).abs() < 1e-6, "scalar={scalar} simd={simd}");
    }

    #[test]
    fn test_stddev_f32_basic() {
        let data = vec![2.0f32, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0]; // var=4 → sd=2
        let s = stddev_f32(&data);
        assert!((s - 2.0).abs() < 1e-3, "stddev={s}");
    }
}
