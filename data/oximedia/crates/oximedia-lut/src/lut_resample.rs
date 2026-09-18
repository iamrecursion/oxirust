#![allow(dead_code)]
//! LUT resampling utilities for changing grid resolution.
//!
//! When working with LUTs of different sizes, it is often necessary to
//! resample from one grid resolution to another (e.g., 17x17x17 to 33x33x33).
//! This module provides trilinear and nearest-neighbor resampling strategies
//! along with 1D LUT resampling.

/// Resampling method used when changing LUT resolution.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ResampleMethod {
    /// Nearest-neighbor: pick the closest sample.
    Nearest,
    /// Trilinear interpolation between surrounding samples.
    Trilinear,
}

/// Configuration for a LUT resampling operation.
#[derive(Clone, Debug)]
pub struct ResampleConfig {
    /// Source grid size (entries per axis for 3D, total entries for 1D).
    pub source_size: usize,
    /// Target grid size.
    pub target_size: usize,
    /// Interpolation method.
    pub method: ResampleMethod,
}

impl ResampleConfig {
    /// Create a new resample configuration.
    #[must_use]
    pub fn new(source_size: usize, target_size: usize, method: ResampleMethod) -> Self {
        Self {
            source_size,
            target_size,
            method,
        }
    }
}

/// Result of a 1D LUT resampling operation.
#[derive(Clone, Debug)]
pub struct Resample1dResult {
    /// The resampled LUT data.
    pub data: Vec<f64>,
    /// The target size that was used.
    pub target_size: usize,
}

/// Result of a 3D LUT resampling operation.
#[derive(Clone, Debug)]
pub struct Resample3dResult {
    /// The resampled 3D LUT data (flattened RGB triplets).
    pub data: Vec<[f64; 3]>,
    /// The target grid size per axis.
    pub target_size: usize,
}

/// Resample a 1D LUT from one size to another using linear interpolation.
///
/// # Arguments
/// * `source` - Source 1D LUT values
/// * `target_size` - Desired number of entries in the output
///
/// # Returns
/// A vector of resampled values with `target_size` entries.
#[must_use]
pub fn resample_1d_linear(source: &[f64], target_size: usize) -> Vec<f64> {
    if target_size == 0 || source.is_empty() {
        return Vec::new();
    }
    if target_size == 1 {
        return vec![source[0]];
    }
    let src_len = source.len();
    let mut result = Vec::with_capacity(target_size);
    for i in 0..target_size {
        let t = i as f64 / (target_size - 1) as f64;
        let src_pos = t * (src_len - 1) as f64;
        let lo = (src_pos.floor() as usize).min(src_len - 1);
        let hi = (lo + 1).min(src_len - 1);
        let frac = src_pos - lo as f64;
        let value = source[lo] * (1.0 - frac) + source[hi] * frac;
        result.push(value);
    }
    result
}

/// Resample a 1D LUT using nearest-neighbor interpolation.
///
/// # Arguments
/// * `source` - Source 1D LUT values
/// * `target_size` - Desired number of entries in the output
#[must_use]
pub fn resample_1d_nearest(source: &[f64], target_size: usize) -> Vec<f64> {
    if target_size == 0 || source.is_empty() {
        return Vec::new();
    }
    let src_len = source.len();
    let mut result = Vec::with_capacity(target_size);
    for i in 0..target_size {
        let t = if target_size == 1 {
            0.0
        } else {
            i as f64 / (target_size - 1) as f64
        };
        let idx = (t * (src_len - 1) as f64).round() as usize;
        let idx = idx.min(src_len - 1);
        result.push(source[idx]);
    }
    result
}

/// Trilinear interpolation helper: look up an RGB value from a flat 3D LUT.
fn trilinear_lookup(data: &[[f64; 3]], size: usize, r: f64, g: f64, b: f64) -> [f64; 3] {
    let max_idx = (size - 1) as f64;
    let ri = (r * max_idx).clamp(0.0, max_idx);
    let gi = (g * max_idx).clamp(0.0, max_idx);
    let bi = (b * max_idx).clamp(0.0, max_idx);

    let r0 = ri.floor() as usize;
    let g0 = gi.floor() as usize;
    let b0 = bi.floor() as usize;
    let r1 = (r0 + 1).min(size - 1);
    let g1 = (g0 + 1).min(size - 1);
    let b1 = (b0 + 1).min(size - 1);

    let fr = ri - r0 as f64;
    let fg = gi - g0 as f64;
    let fb = bi - b0 as f64;

    let idx = |r: usize, g: usize, b: usize| r * size * size + g * size + b;

    let c000 = data[idx(r0, g0, b0)];
    let c100 = data[idx(r1, g0, b0)];
    let c010 = data[idx(r0, g1, b0)];
    let c110 = data[idx(r1, g1, b0)];
    let c001 = data[idx(r0, g0, b1)];
    let c101 = data[idx(r1, g0, b1)];
    let c011 = data[idx(r0, g1, b1)];
    let c111 = data[idx(r1, g1, b1)];

    let mut out = [0.0; 3];
    for ch in 0..3 {
        let c00 = c000[ch] * (1.0 - fr) + c100[ch] * fr;
        let c01 = c001[ch] * (1.0 - fr) + c101[ch] * fr;
        let c10 = c010[ch] * (1.0 - fr) + c110[ch] * fr;
        let c11 = c011[ch] * (1.0 - fr) + c111[ch] * fr;
        let c0 = c00 * (1.0 - fg) + c10 * fg;
        let c1 = c01 * (1.0 - fg) + c11 * fg;
        out[ch] = c0 * (1.0 - fb) + c1 * fb;
    }
    out
}

/// Nearest-neighbor lookup in a flat 3D LUT.
fn nearest_lookup(data: &[[f64; 3]], size: usize, r: f64, g: f64, b: f64) -> [f64; 3] {
    let max_idx = (size - 1) as f64;
    let ri = (r * max_idx).round() as usize;
    let gi = (g * max_idx).round() as usize;
    let bi = (b * max_idx).round() as usize;
    let ri = ri.min(size - 1);
    let gi = gi.min(size - 1);
    let bi = bi.min(size - 1);
    data[ri * size * size + gi * size + bi]
}

/// Resample a 3D LUT from one grid size to another.
///
/// The source data is a flat array of `[f64; 3]` RGB triplets, indexed as
/// `data[r * src_size * src_size + g * src_size + b]`.
///
/// # Arguments
/// * `source` - Source 3D LUT data
/// * `src_size` - Source grid size per axis
/// * `target_size` - Desired grid size per axis
/// * `method` - Interpolation method
#[must_use]
pub fn resample_3d(
    source: &[[f64; 3]],
    src_size: usize,
    target_size: usize,
    method: ResampleMethod,
) -> Resample3dResult {
    let total = target_size * target_size * target_size;
    let mut data = Vec::with_capacity(total);

    for ri in 0..target_size {
        let r = if target_size == 1 {
            0.0
        } else {
            ri as f64 / (target_size - 1) as f64
        };
        for gi in 0..target_size {
            let g = if target_size == 1 {
                0.0
            } else {
                gi as f64 / (target_size - 1) as f64
            };
            for bi in 0..target_size {
                let b = if target_size == 1 {
                    0.0
                } else {
                    bi as f64 / (target_size - 1) as f64
                };
                let rgb = match method {
                    ResampleMethod::Trilinear => trilinear_lookup(source, src_size, r, g, b),
                    ResampleMethod::Nearest => nearest_lookup(source, src_size, r, g, b),
                };
                data.push(rgb);
            }
        }
    }

    Resample3dResult { data, target_size }
}

/// Compute the maximum absolute error between two 3D LUTs of the same size.
///
/// Both LUTs must have the same number of entries.
#[must_use]
pub fn max_resample_error(a: &[[f64; 3]], b: &[[f64; 3]]) -> f64 {
    a.iter()
        .zip(b.iter())
        .map(|(va, vb)| {
            let dr = (va[0] - vb[0]).abs();
            let dg = (va[1] - vb[1]).abs();
            let db = (va[2] - vb[2]).abs();
            dr.max(dg).max(db)
        })
        .fold(0.0_f64, f64::max)
}

/// Generate an identity 3D LUT of the given size.
///
/// Each entry maps to its own normalized coordinate.
#[must_use]
pub fn generate_identity_3d(size: usize) -> Vec<[f64; 3]> {
    let total = size * size * size;
    let mut data = Vec::with_capacity(total);
    let max = if size <= 1 { 1.0 } else { (size - 1) as f64 };
    for ri in 0..size {
        for gi in 0..size {
            for bi in 0..size {
                data.push([ri as f64 / max, gi as f64 / max, bi as f64 / max]);
            }
        }
    }
    data
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_resample_1d_linear_identity() {
        let src = vec![0.0, 0.25, 0.5, 0.75, 1.0];
        let result = resample_1d_linear(&src, 5);
        for (i, &v) in result.iter().enumerate() {
            assert!((v - src[i]).abs() < 1e-12);
        }
    }

    #[test]
    fn test_resample_1d_linear_upsample() {
        let src = vec![0.0, 1.0];
        let result = resample_1d_linear(&src, 5);
        assert_eq!(result.len(), 5);
        assert!((result[0] - 0.0).abs() < 1e-12);
        assert!((result[2] - 0.5).abs() < 1e-12);
        assert!((result[4] - 1.0).abs() < 1e-12);
    }

    #[test]
    fn test_resample_1d_linear_downsample() {
        let src = vec![0.0, 0.25, 0.5, 0.75, 1.0];
        let result = resample_1d_linear(&src, 3);
        assert_eq!(result.len(), 3);
        assert!((result[0] - 0.0).abs() < 1e-12);
        assert!((result[1] - 0.5).abs() < 1e-12);
        assert!((result[2] - 1.0).abs() < 1e-12);
    }

    #[test]
    fn test_resample_1d_nearest_upsample() {
        let src = vec![0.0, 1.0];
        let result = resample_1d_nearest(&src, 3);
        assert_eq!(result.len(), 3);
        assert!((result[0] - 0.0).abs() < 1e-12);
        assert!((result[2] - 1.0).abs() < 1e-12);
    }

    #[test]
    fn test_resample_1d_empty() {
        let result = resample_1d_linear(&[], 5);
        assert!(result.is_empty());
        let result2 = resample_1d_linear(&[1.0], 0);
        assert!(result2.is_empty());
    }

    #[test]
    fn test_resample_1d_single() {
        let result = resample_1d_linear(&[0.42], 1);
        assert_eq!(result.len(), 1);
        assert!((result[0] - 0.42).abs() < 1e-12);
    }

    #[test]
    fn test_generate_identity_3d() {
        let id = generate_identity_3d(3);
        assert_eq!(id.len(), 27);
        // First entry should be [0, 0, 0]
        assert!((id[0][0]).abs() < 1e-12);
        // Last entry should be [1, 1, 1]
        assert!((id[26][0] - 1.0).abs() < 1e-12);
        assert!((id[26][1] - 1.0).abs() < 1e-12);
        assert!((id[26][2] - 1.0).abs() < 1e-12);
    }

    #[test]
    fn test_resample_3d_identity_to_same_size() {
        let src = generate_identity_3d(3);
        let result = resample_3d(&src, 3, 3, ResampleMethod::Trilinear);
        assert_eq!(result.data.len(), 27);
        for (a, b) in src.iter().zip(result.data.iter()) {
            assert!((a[0] - b[0]).abs() < 1e-10);
            assert!((a[1] - b[1]).abs() < 1e-10);
            assert!((a[2] - b[2]).abs() < 1e-10);
        }
    }

    #[test]
    fn test_resample_3d_upsample_identity() {
        let src = generate_identity_3d(3);
        let result = resample_3d(&src, 3, 5, ResampleMethod::Trilinear);
        assert_eq!(result.data.len(), 125);
        // For an identity LUT, resampled values should match coordinates
        let max = 4.0;
        for ri in 0..5 {
            for gi in 0..5 {
                for bi in 0..5 {
                    let idx = ri * 25 + gi * 5 + bi;
                    let expected = [ri as f64 / max, gi as f64 / max, bi as f64 / max];
                    assert!((result.data[idx][0] - expected[0]).abs() < 1e-10);
                    assert!((result.data[idx][1] - expected[1]).abs() < 1e-10);
                    assert!((result.data[idx][2] - expected[2]).abs() < 1e-10);
                }
            }
        }
    }

    #[test]
    fn test_resample_3d_nearest() {
        let src = generate_identity_3d(3);
        let result = resample_3d(&src, 3, 3, ResampleMethod::Nearest);
        assert_eq!(result.data.len(), 27);
        for (a, b) in src.iter().zip(result.data.iter()) {
            assert!((a[0] - b[0]).abs() < 1e-10);
        }
    }

    #[test]
    fn test_max_resample_error_identical() {
        let a = generate_identity_3d(3);
        let err = max_resample_error(&a, &a);
        assert!(err < 1e-15);
    }

    #[test]
    fn test_max_resample_error_known() {
        let a = vec![[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]];
        let b = vec![[0.1, 0.0, 0.0], [1.0, 0.8, 1.0]];
        let err = max_resample_error(&a, &b);
        assert!((err - 0.2).abs() < 1e-12);
    }

    #[test]
    fn test_resample_config() {
        let cfg = ResampleConfig::new(17, 33, ResampleMethod::Trilinear);
        assert_eq!(cfg.source_size, 17);
        assert_eq!(cfg.target_size, 33);
        assert_eq!(cfg.method, ResampleMethod::Trilinear);
    }

    #[test]
    fn test_resample_1d_nearest_empty() {
        let result = resample_1d_nearest(&[], 3);
        assert!(result.is_empty());
    }
}
