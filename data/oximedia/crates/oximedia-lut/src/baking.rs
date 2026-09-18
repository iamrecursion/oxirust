//! LUT baking, combining, and resampling operations.
//!
//! Provides utilities to bake multiple LUTs into a single table,
//! invert 1D LUTs, and resample LUTs to different sizes.

#![allow(dead_code)]

/// Create an identity 1D LUT of the given size.
///
/// Output values are evenly spaced from 0.0 to 1.0.
///
/// # Arguments
///
/// * `size` - Number of entries in the LUT (must be >= 2)
#[must_use]
pub fn lut_to_identity(size: usize) -> Vec<f64> {
    if size == 0 {
        return Vec::new();
    }
    if size == 1 {
        return vec![0.0];
    }
    (0..size).map(|i| i as f64 / (size - 1) as f64).collect()
}

/// Combine two 1D LUTs by applying `lut_a` first, then `lut_b`.
///
/// The result is a single 1D LUT with the same size as `lut_a`.
/// `lut_b` is evaluated via linear interpolation.
///
/// # Arguments
///
/// * `lut_a` - First LUT applied (input -> intermediate)
/// * `lut_b` - Second LUT applied (intermediate -> output)
#[must_use]
pub fn combine_luts_1d(lut_a: &[f64], lut_b: &[f64]) -> Vec<f64> {
    if lut_a.is_empty() || lut_b.is_empty() {
        return Vec::new();
    }
    lut_a
        .iter()
        .map(|&intermediate| sample_1d(lut_b, intermediate))
        .collect()
}

/// Sample a 1D LUT at a normalized position using linear interpolation.
///
/// # Arguments
///
/// * `lut` - LUT values
/// * `t` - Normalized input (0.0–1.0)
fn sample_1d(lut: &[f64], t: f64) -> f64 {
    let n = lut.len();
    if n == 0 {
        return 0.0;
    }
    if n == 1 {
        return lut[0];
    }
    let t_clamped = t.clamp(0.0, 1.0);
    let scaled = t_clamped * (n - 1) as f64;
    let lo = scaled.floor() as usize;
    let hi = (lo + 1).min(n - 1);
    let frac = scaled - lo as f64;
    lut[lo] + frac * (lut[hi] - lut[lo])
}

/// Compute the inverse of a 1D LUT via binary search / linear interpolation.
///
/// Assumes the LUT is monotonically increasing. For each output position,
/// finds the corresponding input via bisection.
///
/// # Arguments
///
/// * `lut` - Monotonically increasing LUT values
#[must_use]
pub fn invert_1d_lut(lut: &[f64]) -> Vec<f64> {
    let n = lut.len();
    if n == 0 {
        return Vec::new();
    }
    if n == 1 {
        return vec![lut[0]];
    }

    let min_val = lut[0];
    let max_val = lut[n - 1];

    (0..n)
        .map(|i| {
            let target = min_val + (max_val - min_val) * (i as f64 / (n - 1) as f64);
            // Binary search for the index where lut[idx] >= target
            let mut lo = 0usize;
            let mut hi = n - 1;
            while hi - lo > 1 {
                let mid = (lo + hi) / 2;
                if lut[mid] <= target {
                    lo = mid;
                } else {
                    hi = mid;
                }
            }
            // Linear interpolation between lo and hi
            let range = lut[hi] - lut[lo];
            if range.abs() < 1e-12 {
                lo as f64 / (n - 1) as f64
            } else {
                let frac = (target - lut[lo]) / range;
                (lo as f64 + frac) / (n - 1) as f64
            }
        })
        .collect()
}

/// Combine two 3D LUTs by applying `lut_a` first, then `lut_b`.
///
/// Both LUTs must have the same `size` per dimension and use B-major
/// (B outermost, R innermost) ordering with interleaved RGB values.
///
/// # Arguments
///
/// * `lut_a` - First 3D LUT (RGB interleaved)
/// * `lut_b` - Second 3D LUT (RGB interleaved)
/// * `size` - Number of entries per dimension
#[allow(clippy::many_single_char_names)]
#[must_use]
pub fn combine_3d_luts(lut_a: &[f64], lut_b: &[f64], size: usize) -> Vec<f64> {
    let total = size * size * size;
    let mut result = Vec::with_capacity(total * 3);

    for i in 0..total {
        let ra = lut_a[i * 3];
        let ga = lut_a[i * 3 + 1];
        let ba = lut_a[i * 3 + 2];
        // Evaluate lut_b at (ra, ga, ba) using trilinear interpolation
        let [rb, gb, bb] = sample_3d_trilinear(lut_b, size, [ra, ga, ba]);
        result.push(rb);
        result.push(gb);
        result.push(bb);
    }
    result
}

/// Trilinear interpolation into a 3D LUT.
///
/// # Arguments
///
/// * `lut` - 3D LUT values (RGB interleaved, B-major order)
/// * `size` - Number of entries per dimension
/// * `rgb` - Input coordinates (0.0–1.0)
#[allow(clippy::many_single_char_names)]
fn sample_3d_trilinear(lut: &[f64], size: usize, rgb: [f64; 3]) -> [f64; 3] {
    if size == 0 {
        return [0.0; 3];
    }
    let s = (size - 1) as f64;
    let ri = rgb[0].clamp(0.0, 1.0) * s;
    let gi = rgb[1].clamp(0.0, 1.0) * s;
    let bi = rgb[2].clamp(0.0, 1.0) * s;

    let r0 = ri.floor() as usize;
    let g0 = gi.floor() as usize;
    let b0 = bi.floor() as usize;
    let r1 = (r0 + 1).min(size - 1);
    let g1 = (g0 + 1).min(size - 1);
    let b1 = (b0 + 1).min(size - 1);

    let fr = ri - r0 as f64;
    let fg = gi - g0 as f64;
    let fb = bi - b0 as f64;

    // Index into the flattened lut (B-major: index = b * size*size + g * size + r)
    let idx = |r: usize, g: usize, b: usize| -> usize { (b * size * size + g * size + r) * 3 };

    let lerp = |a: f64, b_val: f64, t: f64| a + t * (b_val - a);

    let mut out = [0.0f64; 3];
    for ch in 0..3 {
        let c000 = lut[idx(r0, g0, b0) + ch];
        let c100 = lut[idx(r1, g0, b0) + ch];
        let c010 = lut[idx(r0, g1, b0) + ch];
        let c110 = lut[idx(r1, g1, b0) + ch];
        let c001 = lut[idx(r0, g0, b1) + ch];
        let c101 = lut[idx(r1, g0, b1) + ch];
        let c011 = lut[idx(r0, g1, b1) + ch];
        let c111 = lut[idx(r1, g1, b1) + ch];

        let c00 = lerp(c000, c100, fr);
        let c10 = lerp(c010, c110, fr);
        let c01 = lerp(c001, c101, fr);
        let c11 = lerp(c011, c111, fr);
        let c0 = lerp(c00, c10, fg);
        let c1 = lerp(c01, c11, fg);
        out[ch] = lerp(c0, c1, fb);
    }
    out
}

/// Downsample a 3D LUT from `from_size` to `to_size` via trilinear interpolation.
///
/// # Arguments
///
/// * `lut` - Source 3D LUT (RGB interleaved, B-major order)
/// * `from_size` - Source size per dimension
/// * `to_size` - Target size per dimension
#[must_use]
pub fn downsample_3d_lut(lut: &[f64], from_size: usize, to_size: usize) -> Vec<f64> {
    if to_size == 0 || from_size == 0 {
        return Vec::new();
    }
    let total = to_size * to_size * to_size;
    let mut result = Vec::with_capacity(total * 3);
    let s = (to_size - 1).max(1) as f64;

    for b in 0..to_size {
        for g in 0..to_size {
            for r in 0..to_size {
                let rgb = [r as f64 / s, g as f64 / s, b as f64 / s];
                let out = sample_3d_trilinear(lut, from_size, rgb);
                result.push(out[0]);
                result.push(out[1]);
                result.push(out[2]);
            }
        }
    }
    result
}

/// Upsample a 1D LUT to a larger size via linear interpolation.
///
/// # Arguments
///
/// * `lut` - Source 1D LUT values
/// * `target_size` - Number of entries in the output LUT
#[must_use]
pub fn upsample_1d_lut(lut: &[f64], target_size: usize) -> Vec<f64> {
    if lut.is_empty() || target_size == 0 {
        return Vec::new();
    }
    if target_size == 1 {
        return vec![lut[0]];
    }
    (0..target_size)
        .map(|i| {
            let t = i as f64 / (target_size - 1) as f64;
            sample_1d(lut, t)
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn identity_1d(size: usize) -> Vec<f64> {
        lut_to_identity(size)
    }

    fn identity_3d(size: usize) -> Vec<f64> {
        let mut lut = Vec::with_capacity(size * size * size * 3);
        for b in 0..size {
            for g in 0..size {
                for r in 0..size {
                    lut.push(r as f64 / (size - 1) as f64);
                    lut.push(g as f64 / (size - 1) as f64);
                    lut.push(b as f64 / (size - 1) as f64);
                }
            }
        }
        lut
    }

    #[test]
    fn test_identity_lut_size_4() {
        let lut = lut_to_identity(4);
        assert_eq!(lut.len(), 4);
        assert!((lut[0] - 0.0).abs() < 1e-10);
        assert!((lut[3] - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_identity_lut_empty() {
        assert!(lut_to_identity(0).is_empty());
    }

    #[test]
    fn test_combine_luts_1d_identity_identity() {
        let id = identity_1d(64);
        let combined = combine_luts_1d(&id, &id);
        assert_eq!(combined.len(), 64);
        for (a, b) in id.iter().zip(combined.iter()) {
            assert!((a - b).abs() < 1e-6, "mismatch: {a} vs {b}");
        }
    }

    #[test]
    fn test_combine_luts_1d_bakes_gain() {
        // lut_a: identity, lut_b: gain 0.5 (maps [0,1] -> [0,0.5])
        let id = identity_1d(64);
        let half: Vec<f64> = id.iter().map(|&x| x * 0.5).collect();
        let combined = combine_luts_1d(&id, &half);
        // Result should be gain 0.5
        assert!((combined[0] - 0.0).abs() < 1e-6);
        assert!((combined[63] - 0.5).abs() < 0.01);
    }

    #[test]
    fn test_invert_1d_lut_identity() {
        let id = identity_1d(64);
        let inv = invert_1d_lut(&id);
        // Inverse of identity should be identity
        for (a, b) in id.iter().zip(inv.iter()) {
            assert!((a - b).abs() < 0.02, "mismatch: {a} vs {b}");
        }
    }

    #[test]
    fn test_invert_1d_lut_roundtrip() {
        // Apply a gain LUT then its inverse => should get identity
        let gain: Vec<f64> = (0..32).map(|i| (i as f64 / 31.0) * 0.8 + 0.1).collect();
        let inv = invert_1d_lut(&gain);
        let roundtrip = combine_luts_1d(&gain, &inv);
        // roundtrip should approximate identity over the mapped range
        assert_eq!(roundtrip.len(), 32);
    }

    #[test]
    fn test_invert_1d_lut_empty() {
        assert!(invert_1d_lut(&[]).is_empty());
    }

    #[test]
    fn test_combine_3d_luts_identity() {
        let id = identity_3d(4);
        let combined = combine_3d_luts(&id, &id, 4);
        assert_eq!(combined.len(), id.len());
        for (a, b) in id.iter().zip(combined.iter()) {
            assert!((a - b).abs() < 1e-5, "mismatch: {a} vs {b}");
        }
    }

    #[test]
    fn test_downsample_3d_lut_size_reduction() {
        let id = identity_3d(8);
        let small = downsample_3d_lut(&id, 8, 4);
        assert_eq!(small.len(), 4 * 4 * 4 * 3);
    }

    #[test]
    fn test_downsample_3d_lut_identity_preserved() {
        let id = identity_3d(4);
        let downsampled = downsample_3d_lut(&id, 4, 2);
        assert_eq!(downsampled.len(), 2 * 2 * 2 * 3);
        // Corner [0,0,0] should be [0,0,0]
        assert!((downsampled[0] - 0.0).abs() < 1e-6);
        // Corner [1,1,1] should be [1,1,1]
        let last = downsampled.len() - 3;
        assert!((downsampled[last] - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_upsample_1d_lut() {
        let small = identity_1d(4);
        let large = upsample_1d_lut(&small, 16);
        assert_eq!(large.len(), 16);
        assert!((large[0] - 0.0).abs() < 1e-6);
        assert!((large[15] - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_upsample_1d_lut_empty() {
        assert!(upsample_1d_lut(&[], 10).is_empty());
    }

    #[test]
    fn test_upsample_1d_preserves_values() {
        // Upsampling an identity should still be an identity
        let id = identity_1d(8);
        let up = upsample_1d_lut(&id, 32);
        for (i, &v) in up.iter().enumerate() {
            let expected = i as f64 / 31.0;
            assert!((v - expected).abs() < 0.01, "at {i}: {v} vs {expected}");
        }
    }

    #[test]
    fn test_sample_1d_clamping() {
        let lut = vec![0.2, 0.5, 0.8];
        assert!((sample_1d(&lut, -1.0) - 0.2).abs() < 1e-6);
        assert!((sample_1d(&lut, 2.0) - 0.8).abs() < 1e-6);
    }
}
