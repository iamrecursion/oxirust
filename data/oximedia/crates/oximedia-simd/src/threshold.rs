//! SIMD threshold operations
//!
//! Binary threshold, adaptive threshold, and compare-and-select operations
//! on grayscale and RGBA image buffers.

#![allow(dead_code)]
#![allow(clippy::cast_precision_loss)]
#![allow(clippy::cast_possible_truncation)]
#![allow(clippy::cast_sign_loss)]

/// Binary threshold output mode
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ThresholdType {
    /// Pixels above threshold become `max_val`, others become 0
    Binary,
    /// Pixels above threshold become 0, others become `max_val`
    BinaryInv,
    /// Pixels above threshold are clamped to `thresh`, others unchanged
    Trunc,
    /// Pixels at or below threshold become 0, others unchanged
    ToZero,
    /// Pixels above threshold become 0, others unchanged
    ToZeroInv,
}

/// Apply binary threshold to a grayscale buffer.
///
/// Each pixel is compared to `thresh`. Based on `kind`, either the pixel
/// or `max_val` is written to `dst`.
///
/// # Errors
/// Returns an error if `src` and `dst` have different lengths.
pub fn threshold(
    src: &[u8],
    dst: &mut [u8],
    thresh: u8,
    max_val: u8,
    kind: ThresholdType,
) -> Result<(), String> {
    if src.len() != dst.len() {
        return Err("src and dst must have equal length".to_string());
    }
    for (s, d) in src.iter().zip(dst.iter_mut()) {
        *d = match kind {
            ThresholdType::Binary => {
                if *s > thresh {
                    max_val
                } else {
                    0
                }
            }
            ThresholdType::BinaryInv => {
                if *s > thresh {
                    0
                } else {
                    max_val
                }
            }
            ThresholdType::Trunc => {
                if *s > thresh {
                    thresh
                } else {
                    *s
                }
            }
            ThresholdType::ToZero => {
                if *s > thresh {
                    *s
                } else {
                    0
                }
            }
            ThresholdType::ToZeroInv => {
                if *s > thresh {
                    0
                } else {
                    *s
                }
            }
        };
    }
    Ok(())
}

/// Adaptive threshold using local mean in a `block_size x block_size` window.
///
/// Threshold at each pixel = `local_mean` - `c`. If `c` is positive, the threshold
/// is lowered, making more pixels foreground.
///
/// # Errors
/// Returns an error if dimensions are inconsistent or `block_size` is 0.
pub fn adaptive_threshold_mean(
    src: &[u8],
    dst: &mut [u8],
    width: usize,
    height: usize,
    block_size: usize,
    c: i32,
    max_val: u8,
) -> Result<(), String> {
    if src.len() != width * height || dst.len() != width * height {
        return Err("Buffer length must equal width * height".to_string());
    }
    if block_size == 0 {
        return Err("block_size must be > 0".to_string());
    }
    let half = block_size / 2;
    for y in 0..height {
        for x in 0..width {
            let y_min = y.saturating_sub(half);
            let y_max = (y + half + 1).min(height);
            let x_min = x.saturating_sub(half);
            let x_max = (x + half + 1).min(width);
            let mut sum: u32 = 0;
            let mut count: u32 = 0;
            for ky in y_min..y_max {
                for kx in x_min..x_max {
                    sum += u32::from(src[ky * width + kx]);
                    count += 1;
                }
            }
            #[allow(clippy::cast_possible_wrap)]
            let mean = (sum / count) as i32;
            let local_thresh = (mean - c).clamp(0, 255) as u8;
            dst[y * width + x] = if src[y * width + x] > local_thresh {
                max_val
            } else {
                0
            };
        }
    }
    Ok(())
}

/// Vectorized compare-and-select: for each element, select `a[i]` if
/// `cmp[i] >= thresh`, else `b[i]`.
///
/// # Errors
/// Returns an error if slices have different lengths.
pub fn compare_select(
    cmp: &[u8],
    a: &[u8],
    b: &[u8],
    out: &mut [u8],
    thresh: u8,
) -> Result<(), String> {
    if cmp.len() != a.len() || cmp.len() != b.len() || cmp.len() != out.len() {
        return Err("All slices must have equal length".to_string());
    }
    for i in 0..cmp.len() {
        out[i] = if cmp[i] >= thresh { a[i] } else { b[i] };
    }
    Ok(())
}

/// Clamp each element of `src` to `[lo, hi]` and write to `dst`.
///
/// # Errors
/// Returns an error if `src` and `dst` have different lengths or `lo > hi`.
pub fn clamp_range(src: &[u8], dst: &mut [u8], lo: u8, hi: u8) -> Result<(), String> {
    if src.len() != dst.len() {
        return Err("src and dst must have equal length".to_string());
    }
    if lo > hi {
        return Err("lo must be <= hi".to_string());
    }
    for (s, d) in src.iter().zip(dst.iter_mut()) {
        *d = (*s).clamp(lo, hi);
    }
    Ok(())
}

/// Threshold an RGBA buffer on the alpha channel only.
/// Pixels whose alpha is below `alpha_thresh` are zeroed out entirely.
///
/// # Errors
/// Returns an error if `src` length is not a multiple of 4, or `src` and `dst`
/// have different lengths.
pub fn alpha_threshold(src: &[u8], dst: &mut [u8], alpha_thresh: u8) -> Result<(), String> {
    if !src.len().is_multiple_of(4) {
        return Err("Buffer length must be a multiple of 4".to_string());
    }
    if src.len() != dst.len() {
        return Err("src and dst must have equal length".to_string());
    }
    for i in (0..src.len()).step_by(4) {
        if src[i + 3] < alpha_thresh {
            dst[i] = 0;
            dst[i + 1] = 0;
            dst[i + 2] = 0;
            dst[i + 3] = 0;
        } else {
            dst[i] = src[i];
            dst[i + 1] = src[i + 1];
            dst[i + 2] = src[i + 2];
            dst[i + 3] = src[i + 3];
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_binary_threshold_basic() {
        let src = vec![50u8, 100, 150, 200, 250];
        let mut dst = vec![0u8; 5];
        threshold(&src, &mut dst, 127, 255, ThresholdType::Binary).expect("should succeed in test");
        assert_eq!(dst, vec![0, 0, 255, 255, 255]);
    }

    #[test]
    fn test_binary_inv_threshold() {
        let src = vec![50u8, 100, 150, 200, 250];
        let mut dst = vec![0u8; 5];
        threshold(&src, &mut dst, 127, 255, ThresholdType::BinaryInv)
            .expect("should succeed in test");
        assert_eq!(dst, vec![255, 255, 0, 0, 0]);
    }

    #[test]
    fn test_trunc_threshold() {
        let src = vec![100u8, 200];
        let mut dst = vec![0u8; 2];
        threshold(&src, &mut dst, 150, 255, ThresholdType::Trunc).expect("should succeed in test");
        assert_eq!(dst[0], 100); // below thresh: unchanged
        assert_eq!(dst[1], 150); // above thresh: clamped to thresh
    }

    #[test]
    fn test_to_zero_threshold() {
        let src = vec![50u8, 100, 200];
        let mut dst = vec![0u8; 3];
        threshold(&src, &mut dst, 99, 255, ThresholdType::ToZero).expect("should succeed in test");
        assert_eq!(dst[0], 0); // <= thresh
        assert_eq!(dst[1], 100); // > thresh: unchanged
        assert_eq!(dst[2], 200); // > thresh: unchanged
    }

    #[test]
    fn test_to_zero_inv_threshold() {
        let src = vec![50u8, 100, 200];
        let mut dst = vec![0u8; 3];
        threshold(&src, &mut dst, 99, 255, ThresholdType::ToZeroInv)
            .expect("should succeed in test");
        assert_eq!(dst[0], 50); // <= thresh: unchanged
        assert_eq!(dst[1], 0); // > thresh: zeroed
        assert_eq!(dst[2], 0); // > thresh: zeroed
    }

    #[test]
    fn test_threshold_length_mismatch() {
        let src = vec![0u8; 4];
        let mut dst = vec![0u8; 5];
        assert!(threshold(&src, &mut dst, 128, 255, ThresholdType::Binary).is_err());
    }

    #[test]
    fn test_adaptive_threshold_uniform() {
        // Uniform image -> all pixels equal mean, so none exceed mean-c (c=0)
        let src = vec![100u8; 9];
        let mut dst = vec![0u8; 9];
        adaptive_threshold_mean(&src, &mut dst, 3, 3, 3, 0, 255).expect("should succeed in test");
        // All equal mean, so src[i] == thresh -> condition is src > thresh -> false
        assert!(dst.iter().all(|&v| v == 0));
    }

    #[test]
    fn test_adaptive_threshold_center_bright() {
        // Center pixel brighter than neighbors
        let mut src = vec![50u8; 9];
        src[4] = 200; // center pixel
        let mut dst = vec![0u8; 9];
        adaptive_threshold_mean(&src, &mut dst, 3, 3, 3, 0, 255).expect("should succeed in test");
        // Center should be thresholded to max_val (local mean is low)
        assert_eq!(dst[4], 255);
    }

    #[test]
    fn test_adaptive_threshold_block_size_zero() {
        let src = vec![0u8; 9];
        let mut dst = vec![0u8; 9];
        assert!(adaptive_threshold_mean(&src, &mut dst, 3, 3, 0, 0, 255).is_err());
    }

    #[test]
    fn test_compare_select() {
        let cmp = vec![10u8, 50, 200];
        let a = vec![1u8, 2, 3];
        let b = vec![4u8, 5, 6];
        let mut out = vec![0u8; 3];
        compare_select(&cmp, &a, &b, &mut out, 100).expect("should succeed in test");
        assert_eq!(out[0], 4); // cmp[0]=10 < 100 -> b
        assert_eq!(out[1], 5); // cmp[1]=50 < 100 -> b
        assert_eq!(out[2], 3); // cmp[2]=200 >= 100 -> a
    }

    #[test]
    fn test_clamp_range() {
        let src = vec![0u8, 50, 100, 200, 255];
        let mut dst = vec![0u8; 5];
        clamp_range(&src, &mut dst, 50, 150).expect("should succeed in test");
        assert_eq!(dst, vec![50, 50, 100, 150, 150]);
    }

    #[test]
    fn test_clamp_range_invalid() {
        let src = vec![100u8];
        let mut dst = vec![0u8; 1];
        assert!(clamp_range(&src, &mut dst, 200, 100).is_err());
    }

    #[test]
    fn test_alpha_threshold_zeros_below() {
        let src = vec![255u8, 128, 64, 10, 100, 200, 150, 255];
        let mut dst = vec![0u8; 8];
        alpha_threshold(&src, &mut dst, 50).expect("should succeed in test");
        // First pixel: alpha=10 < 50 -> zeroed
        assert_eq!(&dst[0..4], &[0, 0, 0, 0]);
        // Second pixel: alpha=255 >= 50 -> kept
        assert_eq!(&dst[4..8], &[100, 200, 150, 255]);
    }

    #[test]
    fn test_alpha_threshold_not_multiple_of_4() {
        let src = vec![0u8; 5];
        let mut dst = vec![0u8; 5];
        assert!(alpha_threshold(&src, &mut dst, 128).is_err());
    }
}
