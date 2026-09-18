#![allow(dead_code)]
//! Identity LUT generation and delta analysis.
//!
//! Provides:
//! * [`IdentityLut`] – generates exact identity 1-D and 3-D LUTs and verifies them.
//! * [`LutDelta`]    – measures the per-channel deviation between two LUTs.

use crate::Rgb;

// ---------------------------------------------------------------------------
// IdentityLut
// ---------------------------------------------------------------------------

/// Factory and inspector for identity look-up tables.
///
/// An identity LUT maps every input value to itself – it is a no-op transform
/// useful as a baseline for delta analysis or as a starting point for grading.
pub struct IdentityLut;

impl IdentityLut {
    /// Generate an interleaved 1-D identity LUT with `size` entries per channel.
    ///
    /// The returned vector has `size * 3` elements in `[R0, G0, B0, R1, G1, B1, …]`
    /// order with values in `[0.0, 1.0]`.
    #[allow(clippy::cast_precision_loss)]
    #[must_use]
    pub fn generate_1d(size: usize) -> Vec<f64> {
        assert!(size >= 2, "size must be at least 2");
        let scale = (size - 1) as f64;
        (0..size)
            .flat_map(|i| {
                let v = i as f64 / scale;
                [v, v, v]
            })
            .collect()
    }

    /// Generate a 3-D identity LUT with `size` divisions per axis.
    ///
    /// Returns `size³` [`Rgb`] entries in row-major `[r][g][b]` order.
    #[allow(clippy::cast_precision_loss)]
    #[must_use]
    pub fn generate_3d(size: usize) -> Vec<Rgb> {
        assert!(size >= 2, "size must be at least 2");
        let scale = (size - 1) as f64;
        let mut data = Vec::with_capacity(size * size * size);
        for r in 0..size {
            for g in 0..size {
                for b in 0..size {
                    data.push([r as f64 / scale, g as f64 / scale, b as f64 / scale]);
                }
            }
        }
        data
    }

    /// Returns `true` when `data` represents an identity 1-D LUT.
    ///
    /// `size` is the number of entries per channel; `data` must have `size * 3`
    /// elements in interleaved `[R, G, B, …]` order.
    #[allow(clippy::cast_precision_loss)]
    #[must_use]
    pub fn is_identity_1d(data: &[f64], size: usize) -> bool {
        if size < 2 || data.len() != size * 3 {
            return false;
        }
        let scale = (size - 1) as f64;
        const EPS: f64 = 1e-6;
        for i in 0..size {
            let expected = i as f64 / scale;
            for ch in 0..3 {
                if (data[i * 3 + ch] - expected).abs() > EPS {
                    return false;
                }
            }
        }
        true
    }

    /// Returns `true` when `data` is an identity 3-D LUT with `size` per axis.
    #[allow(clippy::cast_precision_loss)]
    #[must_use]
    pub fn is_identity_3d(data: &[Rgb], size: usize) -> bool {
        if size < 2 || data.len() != size * size * size {
            return false;
        }
        let scale = (size - 1) as f64;
        const EPS: f64 = 1e-6;
        for r in 0..size {
            for g in 0..size {
                for b in 0..size {
                    let idx = r * size * size + g * size + b;
                    let exp = [r as f64 / scale, g as f64 / scale, b as f64 / scale];
                    if (data[idx][0] - exp[0]).abs() > EPS
                        || (data[idx][1] - exp[1]).abs() > EPS
                        || (data[idx][2] - exp[2]).abs() > EPS
                    {
                        return false;
                    }
                }
            }
        }
        true
    }
}

// ---------------------------------------------------------------------------
// LutDelta
// ---------------------------------------------------------------------------

/// Per-channel deviation statistics between two 3-D LUTs of the same size.
#[derive(Debug, Clone)]
pub struct LutDelta {
    /// Maximum absolute per-channel deviation across all lattice points.
    pub max_deviation: f64,
    /// Mean absolute deviation per channel.
    pub mean_deviation: f64,
    /// Root-mean-square deviation.
    pub rms_deviation: f64,
    /// Number of lattice points compared.
    pub count: usize,
}

impl LutDelta {
    /// Compute the delta between `a` and `b` (both must have `size` per axis).
    ///
    /// Returns `None` when sizes are inconsistent.
    #[allow(clippy::cast_precision_loss)]
    #[must_use]
    pub fn from_luts(a: &[Rgb], b: &[Rgb], size: usize) -> Option<Self> {
        let expected_len = size * size * size;
        if a.len() != expected_len || b.len() != expected_len || size < 2 {
            return None;
        }
        let mut max_dev = 0.0_f64;
        let mut sum_abs = 0.0_f64;
        let mut sum_sq = 0.0_f64;
        let count = expected_len;

        for i in 0..count {
            for ch in 0..3 {
                let diff = (a[i][ch] - b[i][ch]).abs();
                if diff > max_dev {
                    max_dev = diff;
                }
                sum_abs += diff;
                sum_sq += diff * diff;
            }
        }

        let n = (count * 3) as f64;
        Some(Self {
            max_deviation: max_dev,
            mean_deviation: sum_abs / n,
            rms_deviation: (sum_sq / n).sqrt(),
            count,
        })
    }

    /// Maximum absolute per-channel deviation.
    #[must_use]
    pub fn max_deviation(&self) -> f64 {
        self.max_deviation
    }

    /// Returns `true` when every lattice point differs by less than `tolerance`.
    #[must_use]
    pub fn within_tolerance(&self, tolerance: f64) -> bool {
        self.max_deviation < tolerance
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_generate_1d_length() {
        let data = IdentityLut::generate_1d(17);
        assert_eq!(data.len(), 17 * 3);
    }

    #[test]
    fn test_generate_1d_endpoints() {
        let data = IdentityLut::generate_1d(17);
        assert!((data[0] - 0.0).abs() < 1e-9);
        assert!((data[(17 - 1) * 3] - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_generate_3d_length() {
        let data = IdentityLut::generate_3d(5);
        assert_eq!(data.len(), 5 * 5 * 5);
    }

    #[test]
    fn test_generate_3d_corner_black() {
        let data = IdentityLut::generate_3d(5);
        assert_eq!(data[0], [0.0, 0.0, 0.0]);
    }

    #[test]
    fn test_generate_3d_corner_white() {
        let size = 5usize;
        let data = IdentityLut::generate_3d(size);
        let last = data[size * size * size - 1];
        assert!((last[0] - 1.0).abs() < 1e-9);
        assert!((last[1] - 1.0).abs() < 1e-9);
        assert!((last[2] - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_is_identity_1d_true() {
        let data = IdentityLut::generate_1d(33);
        assert!(IdentityLut::is_identity_1d(&data, 33));
    }

    #[test]
    fn test_is_identity_1d_false_after_mutation() {
        let mut data = IdentityLut::generate_1d(9);
        data[6] = 0.999;
        assert!(!IdentityLut::is_identity_1d(&data, 9));
    }

    #[test]
    fn test_is_identity_1d_wrong_size() {
        let data = IdentityLut::generate_1d(9);
        assert!(!IdentityLut::is_identity_1d(&data, 10));
    }

    #[test]
    fn test_is_identity_3d_true() {
        let data = IdentityLut::generate_3d(5);
        assert!(IdentityLut::is_identity_3d(&data, 5));
    }

    #[test]
    fn test_is_identity_3d_false_after_mutation() {
        let mut data = IdentityLut::generate_3d(5);
        data[10][2] = 0.5;
        assert!(!IdentityLut::is_identity_3d(&data, 5));
    }

    #[test]
    fn test_lut_delta_identity_vs_identity() {
        let a = IdentityLut::generate_3d(5);
        let b = IdentityLut::generate_3d(5);
        let delta = LutDelta::from_luts(&a, &b, 5).expect("should succeed in test");
        assert!(delta.max_deviation() < 1e-12);
        assert!(delta.within_tolerance(1e-9));
    }

    #[test]
    fn test_lut_delta_detects_shift() {
        let a = IdentityLut::generate_3d(5);
        let mut b = IdentityLut::generate_3d(5);
        b[0][0] = 0.1;
        let delta = LutDelta::from_luts(&a, &b, 5).expect("should succeed in test");
        assert!(delta.max_deviation() > 0.09);
    }

    #[test]
    fn test_lut_delta_size_mismatch_returns_none() {
        let a = IdentityLut::generate_3d(5);
        let b = IdentityLut::generate_3d(3);
        assert!(LutDelta::from_luts(&a, &b, 5).is_none());
    }

    #[test]
    fn test_lut_delta_count() {
        let a = IdentityLut::generate_3d(4);
        let b = IdentityLut::generate_3d(4);
        let delta = LutDelta::from_luts(&a, &b, 4).expect("should succeed in test");
        assert_eq!(delta.count, 64);
    }
}
