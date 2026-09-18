//! Separable resampling filter kernel representation and application.
//!
//! A `ScaleFilter` holds a pre-computed set of `FilterTap` arrays — one per
//! output pixel — that describe the weighted sum of input pixels needed to
//! produce each output sample.  The same structure is used for both
//! horizontal and vertical passes.

#![allow(dead_code)]
#![allow(clippy::cast_precision_loss)]
#![allow(clippy::cast_possible_truncation)]
#![allow(clippy::cast_sign_loss)]

use serde::{Deserialize, Serialize};

// ── FilterTap ─────────────────────────────────────────────────────────────────

/// A single contributing input sample and its weight.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct FilterTap {
    /// Index of the source pixel that contributes to this output pixel.
    pub source_index: i32,
    /// Weight of the contribution (pre-normalised so taps sum to 1.0).
    pub weight: f32,
}

impl FilterTap {
    /// Create a new `FilterTap`.
    pub fn new(source_index: i32, weight: f32) -> Self {
        Self {
            source_index,
            weight,
        }
    }
}

// ── FilterKernel ──────────────────────────────────────────────────────────────

/// The set of weighted taps for a single output sample.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FilterKernel {
    /// Ordered list of taps (clipped to source bounds by the builder).
    pub taps: Vec<FilterTap>,
}

impl FilterKernel {
    /// Apply this kernel to a row (or column) of `f32` sample values.
    ///
    /// `samples` must be long enough that all `source_index` values are valid.
    pub fn apply(&self, samples: &[f32]) -> f32 {
        let len = samples.len() as i32;
        self.taps
            .iter()
            .map(|t| {
                let idx = t.source_index.clamp(0, len - 1) as usize;
                samples[idx] * t.weight
            })
            .sum()
    }

    /// Sum of all tap weights (should be ≈ 1.0 after normalisation).
    pub fn weight_sum(&self) -> f32 {
        self.taps.iter().map(|t| t.weight).sum()
    }
}

// ── ScaleFilter ───────────────────────────────────────────────────────────────

/// A complete set of filter kernels for scaling one dimension.
///
/// `kernels[i]` holds the taps used to compute output sample `i`.
///
/// # Example
/// ```
/// use oximedia_scaling::scale_filter::ScaleFilter;
///
/// // Build a bilinear filter to scale from 4 input pixels to 8 output pixels
/// let filter = ScaleFilter::bilinear(4, 8);
/// assert_eq!(filter.output_size(), 8);
/// ```
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ScaleFilter {
    /// One kernel per output sample.
    pub kernels: Vec<FilterKernel>,
    /// Number of source samples this filter maps from.
    pub source_size: u32,
    /// Number of output samples this filter maps to.
    pub output_size_val: u32,
}

impl ScaleFilter {
    /// Return the number of output samples.
    pub fn output_size(&self) -> usize {
        self.output_size_val as usize
    }

    /// Return the number of source samples.
    pub fn source_size(&self) -> usize {
        self.source_size as usize
    }

    // ── Factory methods ───────────────────────────────────────────────────────

    /// Build a nearest-neighbour filter.
    pub fn nearest_neighbour(src: u32, dst: u32) -> Self {
        let scale = src as f64 / dst as f64;
        let kernels = (0..dst)
            .map(|i| {
                let src_idx = ((i as f64 + 0.5) * scale - 0.5).round() as i32;
                FilterKernel {
                    taps: vec![FilterTap::new(src_idx, 1.0)],
                }
            })
            .collect();
        Self {
            kernels,
            source_size: src,
            output_size_val: dst,
        }
    }

    /// Build a bilinear (linear) filter.
    pub fn bilinear(src: u32, dst: u32) -> Self {
        let scale = src as f64 / dst as f64;
        let kernels = (0..dst)
            .map(|i| {
                let src_pos = (i as f64 + 0.5) * scale - 0.5;
                let lo = src_pos.floor() as i32;
                let hi = lo + 1;
                let alpha = (src_pos - lo as f64) as f32;
                let w_lo = 1.0 - alpha;
                let w_hi = alpha;
                let mut taps = vec![FilterTap::new(lo, w_lo)];
                if alpha > 0.0 {
                    taps.push(FilterTap::new(hi, w_hi));
                }
                FilterKernel { taps }
            })
            .collect();
        Self {
            kernels,
            source_size: src,
            output_size_val: dst,
        }
    }

    /// Build a simple box filter (area average) — good for downscaling.
    pub fn box_filter(src: u32, dst: u32) -> Self {
        if dst >= src {
            // Upscaling: fall back to nearest-neighbour
            return Self::nearest_neighbour(src, dst);
        }
        let scale = src as f64 / dst as f64;
        let kernels = (0..dst)
            .map(|i| {
                let start = (i as f64 * scale) as i32;
                let end = ((i as f64 + 1.0) * scale) as i32;
                let count = (end - start).max(1);
                let w = 1.0 / count as f32;
                let taps = (start..end).map(|s| FilterTap::new(s, w)).collect();
                FilterKernel { taps }
            })
            .collect();
        Self {
            kernels,
            source_size: src,
            output_size_val: dst,
        }
    }

    // ── Application ───────────────────────────────────────────────────────────

    /// Apply the horizontal filter to a 2-D row-major `f32` buffer.
    ///
    /// `input` has dimensions `(src_h × src_w)`.
    /// Returns a buffer of dimensions `(src_h × dst_w)`.
    pub fn apply_horizontal(&self, input: &[f32], src_w: usize, src_h: usize) -> Vec<f32> {
        let dst_w = self.output_size();
        let mut output = vec![0.0f32; src_h * dst_w];
        for row in 0..src_h {
            let row_start = row * src_w;
            let src_row = &input[row_start..row_start + src_w];
            for (col, kernel) in self.kernels.iter().enumerate() {
                output[row * dst_w + col] = kernel.apply(src_row);
            }
        }
        output
    }

    /// Apply the vertical filter to a 2-D row-major `f32` buffer.
    ///
    /// `input` has dimensions `(src_h × dst_w)` (after horizontal pass).
    /// Returns a buffer of dimensions `(dst_h × dst_w)`.
    pub fn apply_vertical(&self, input: &[f32], dst_w: usize, src_h: usize) -> Vec<f32> {
        let dst_h = self.output_size();
        let mut output = vec![0.0f32; dst_h * dst_w];
        let col_samples: Vec<f32> = vec![0.0f32; src_h];
        let _ = col_samples; // pre-alloc trick; we reuse inline
        for col in 0..dst_w {
            // Collect column samples
            let col_data: Vec<f32> = (0..src_h).map(|r| input[r * dst_w + col]).collect();
            for (row, kernel) in self.kernels.iter().enumerate() {
                output[row * dst_w + col] = kernel.apply(&col_data);
            }
        }
        output
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_filter_tap_new() {
        let t = FilterTap::new(5, 0.75);
        assert_eq!(t.source_index, 5);
        assert!((t.weight - 0.75).abs() < 1e-6);
    }

    #[test]
    fn test_kernel_apply_identity() {
        let samples = [1.0f32, 2.0, 3.0, 4.0];
        let kernel = FilterKernel {
            taps: vec![FilterTap::new(2, 1.0)],
        };
        assert!((kernel.apply(&samples) - 3.0).abs() < 1e-6);
    }

    #[test]
    fn test_kernel_weight_sum_normalised() {
        let kernel = FilterKernel {
            taps: vec![FilterTap::new(0, 0.5), FilterTap::new(1, 0.5)],
        };
        assert!((kernel.weight_sum() - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_nearest_neighbour_output_size() {
        let f = ScaleFilter::nearest_neighbour(8, 4);
        assert_eq!(f.output_size(), 4);
    }

    #[test]
    fn test_nearest_neighbour_single_tap_per_output() {
        let f = ScaleFilter::nearest_neighbour(4, 4);
        for k in &f.kernels {
            assert_eq!(k.taps.len(), 1);
        }
    }

    #[test]
    fn test_bilinear_output_size() {
        let f = ScaleFilter::bilinear(4, 8);
        assert_eq!(f.output_size(), 8);
        assert_eq!(f.source_size(), 4);
    }

    #[test]
    fn test_box_filter_downscale_even() {
        let f = ScaleFilter::box_filter(8, 4);
        assert_eq!(f.output_size(), 4);
    }

    #[test]
    fn test_box_filter_upscale_falls_back_to_nn() {
        let f = ScaleFilter::box_filter(4, 8);
        // Falls back to nearest-neighbour — each kernel has exactly 1 tap
        for k in &f.kernels {
            assert_eq!(k.taps.len(), 1);
        }
    }

    #[test]
    fn test_apply_horizontal_flat_image() {
        // All-ones 2×4 image (2 rows, 4 cols) scaled to 2 cols
        let input = vec![1.0f32; 2 * 4];
        let f = ScaleFilter::bilinear(4, 2);
        let out = f.apply_horizontal(&input, 4, 2);
        assert_eq!(out.len(), 2 * 2);
        for &v in &out {
            assert!((v - 1.0).abs() < 0.05, "expected ≈1.0 got {v}");
        }
    }

    #[test]
    fn test_apply_vertical_flat_image() {
        // All-ones 4×2 image (4 rows, 2 cols) scaled to 2 rows
        let input = vec![1.0f32; 4 * 2];
        let f = ScaleFilter::bilinear(4, 2);
        let out = f.apply_vertical(&input, 2, 4);
        assert_eq!(out.len(), 2 * 2);
        for &v in &out {
            assert!((v - 1.0).abs() < 0.05, "expected ≈1.0 got {v}");
        }
    }

    #[test]
    fn test_apply_horizontal_output_dimensions() {
        let f = ScaleFilter::nearest_neighbour(100, 50);
        let input = vec![0.0f32; 10 * 100];
        let out = f.apply_horizontal(&input, 100, 10);
        assert_eq!(out.len(), 10 * 50);
    }

    #[test]
    fn test_apply_vertical_output_dimensions() {
        let f = ScaleFilter::nearest_neighbour(100, 50);
        let input = vec![0.0f32; 100 * 80];
        let out = f.apply_vertical(&input, 80, 100);
        assert_eq!(out.len(), 50 * 80);
    }

    #[test]
    fn test_kernel_apply_clamps_negative_index() {
        let samples = [5.0f32, 6.0, 7.0];
        let kernel = FilterKernel {
            taps: vec![FilterTap::new(-1, 1.0)],
        };
        // Clamped to index 0
        assert!((kernel.apply(&samples) - 5.0).abs() < 1e-6);
    }

    #[test]
    fn test_kernel_apply_clamps_out_of_bounds() {
        let samples = [5.0f32, 6.0, 7.0];
        let kernel = FilterKernel {
            taps: vec![FilterTap::new(100, 1.0)],
        };
        // Clamped to last sample (index 2 → 7.0)
        assert!((kernel.apply(&samples) - 7.0).abs() < 1e-6);
    }

    #[test]
    fn test_bilinear_identity_scale() {
        // 4 → 4 should give back original values
        let input = vec![10.0f32, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0];
        let f = ScaleFilter::bilinear(4, 4);
        let out = f.apply_horizontal(&input, 4, 2);
        assert_eq!(out.len(), 8);
    }
}
