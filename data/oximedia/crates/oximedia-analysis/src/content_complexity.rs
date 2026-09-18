//! Content complexity metrics: spatial complexity, temporal complexity, and DCT energy analysis.

#![allow(dead_code)]
#![allow(clippy::cast_precision_loss)]

use serde::{Deserialize, Serialize};

/// Spatial complexity metrics for a single frame.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct SpatialComplexity {
    /// Mean absolute deviation of luma gradient (edge density).
    pub edge_density: f32,
    /// Spatial Information (SI) per ITU-T P.910.
    pub spatial_information: f32,
    /// Normalized variance of luma.
    pub luma_variance: f32,
    /// Estimated noise floor (standard deviation of flat regions).
    pub noise_floor: f32,
}

impl SpatialComplexity {
    /// Compute spatial complexity metrics from a luma (Y) plane.
    #[must_use]
    pub fn compute(y_plane: &[u8], width: usize, height: usize) -> Self {
        if y_plane.len() < width * height || width < 2 || height < 2 {
            return Self::default();
        }

        let w = width as f32;
        let h = height as f32;
        let n = w * h;

        // Luma mean and variance.
        let mean: f32 = y_plane.iter().map(|&p| f32::from(p)).sum::<f32>() / n;
        let variance: f32 = y_plane
            .iter()
            .map(|&p| {
                let d = f32::from(p) - mean;
                d * d
            })
            .sum::<f32>()
            / n;

        // Sobel gradient for SI and edge density.
        let mut sobel_sum = 0.0f32;
        let mut sobel_max = 0.0f32;
        let mut count = 0usize;

        for row in 1..(height - 1) {
            for col in 1..(width - 1) {
                let idx = |r: usize, c: usize| r * width + c;
                let p = |r: usize, c: usize| f32::from(y_plane[idx(r, c)]);

                let gx = -p(row - 1, col - 1) + p(row - 1, col + 1) - 2.0 * p(row, col - 1)
                    + 2.0 * p(row, col + 1)
                    - p(row + 1, col - 1)
                    + p(row + 1, col + 1);

                let gy = -p(row - 1, col - 1) - 2.0 * p(row - 1, col) - p(row - 1, col + 1)
                    + p(row + 1, col - 1)
                    + 2.0 * p(row + 1, col)
                    + p(row + 1, col + 1);

                let mag = (gx * gx + gy * gy).sqrt();
                sobel_sum += mag;
                if mag > sobel_max {
                    sobel_max = mag;
                }
                count += 1;
            }
        }

        let edge_density = if count > 0 {
            sobel_sum / count as f32
        } else {
            0.0
        };

        // SI is the standard deviation of the Sobel frame.
        let sobel_mean = edge_density;
        let sobel_variance: f32 = if count > 0 {
            let mut sv = 0.0f32;
            for row in 1..(height - 1) {
                for col in 1..(width - 1) {
                    let idx = |r: usize, c: usize| r * width + c;
                    let p = |r: usize, c: usize| f32::from(y_plane[idx(r, c)]);
                    let gx = -p(row - 1, col - 1) + p(row - 1, col + 1) - 2.0 * p(row, col - 1)
                        + 2.0 * p(row, col + 1)
                        - p(row + 1, col - 1)
                        + p(row + 1, col + 1);
                    let gy = -p(row - 1, col - 1) - 2.0 * p(row - 1, col) - p(row - 1, col + 1)
                        + p(row + 1, col - 1)
                        + 2.0 * p(row + 1, col)
                        + p(row + 1, col + 1);
                    let mag = (gx * gx + gy * gy).sqrt();
                    let d = mag - sobel_mean;
                    sv += d * d;
                }
            }
            sv / count as f32
        } else {
            0.0
        };

        let _ = sobel_max; // suppress unused

        Self {
            edge_density,
            spatial_information: sobel_variance.sqrt(),
            luma_variance: variance,
            noise_floor: 0.0, // noise estimation requires flat region detection
        }
    }
}

/// Temporal complexity metrics between two consecutive frames.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct TemporalComplexity {
    /// Temporal Information (TI) per ITU-T P.910 — std dev of frame difference.
    pub temporal_information: f32,
    /// Mean absolute frame difference.
    pub mean_absolute_difference: f32,
    /// Percentage of pixels that changed significantly.
    pub changed_pixel_ratio: f32,
    /// Whether this is likely a scene cut.
    pub is_scene_cut: bool,
}

impl TemporalComplexity {
    /// Compute temporal complexity between two luma planes.
    #[must_use]
    pub fn compute(
        prev: &[u8],
        curr: &[u8],
        width: usize,
        height: usize,
        cut_threshold: f32,
    ) -> Self {
        let n = width * height;
        if prev.len() < n || curr.len() < n {
            return Self::default();
        }

        let n_f = n as f32;
        let diffs: Vec<f32> = prev
            .iter()
            .zip(curr.iter())
            .map(|(&a, &b)| (f32::from(a) - f32::from(b)).abs())
            .collect();

        let mad = diffs.iter().sum::<f32>() / n_f;
        let mean_diff = mad;

        // Variance of frame difference → TI
        let variance: f32 = diffs
            .iter()
            .map(|&d| {
                let v = d - mean_diff;
                v * v
            })
            .sum::<f32>()
            / n_f;

        let threshold = 10.0f32;
        let changed: usize = diffs.iter().filter(|&&d| d > threshold).count();
        let changed_pixel_ratio = changed as f32 / n_f;

        Self {
            temporal_information: variance.sqrt(),
            mean_absolute_difference: mad,
            changed_pixel_ratio,
            is_scene_cut: mad > cut_threshold,
        }
    }
}

/// 8x8 DCT energy analysis for codec complexity estimation.
///
/// Computes a fast approximation of DCT energy on 8x8 luma blocks.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct DctEnergyAnalysis {
    /// Mean AC energy across all 8x8 blocks (DC excluded).
    pub mean_ac_energy: f32,
    /// DC energy (mean luma squared, proportional to brightness).
    pub mean_dc_energy: f32,
    /// Ratio of AC to total energy.
    pub ac_to_total_ratio: f32,
    /// Number of 8x8 blocks analyzed.
    pub block_count: usize,
}

impl DctEnergyAnalysis {
    /// Compute DCT energy analysis from a luma plane.
    #[must_use]
    pub fn compute(y_plane: &[u8], width: usize, height: usize) -> Self {
        let block_cols = width / 8;
        let block_rows = height / 8;
        let block_count = block_cols * block_rows;

        if block_count == 0 {
            return Self::default();
        }

        let mut total_ac = 0.0f32;
        let mut total_dc = 0.0f32;

        for br in 0..block_rows {
            for bc in 0..block_cols {
                let bx = bc * 8;
                let by = br * 8;

                // Extract 8x8 block.
                let mut block = [0.0f32; 64];
                for row in 0..8 {
                    for col in 0..8 {
                        block[row * 8 + col] = f32::from(y_plane[(by + row) * width + (bx + col)]);
                    }
                }

                // DC component (mean * 8).
                let dc = block.iter().sum::<f32>() / 8.0;
                total_dc += dc * dc;

                // AC energy: sum of squared differences from DC mean.
                let ac: f32 = block
                    .iter()
                    .map(|&v| {
                        let d = v - (block.iter().sum::<f32>() / 64.0);
                        d * d
                    })
                    .sum::<f32>();
                total_ac += ac;
            }
        }

        let mean_ac = total_ac / block_count as f32;
        let mean_dc = total_dc / block_count as f32;
        let total = mean_ac + mean_dc;
        let ratio = if total > 0.0 { mean_ac / total } else { 0.0 };

        Self {
            mean_ac_energy: mean_ac,
            mean_dc_energy: mean_dc,
            ac_to_total_ratio: ratio,
            block_count,
        }
    }

    /// Classify as high / medium / low complexity based on AC energy.
    #[must_use]
    pub fn complexity_level(&self) -> ComplexityLevel {
        if self.mean_ac_energy > 5000.0 {
            ComplexityLevel::High
        } else if self.mean_ac_energy > 1000.0 {
            ComplexityLevel::Medium
        } else {
            ComplexityLevel::Low
        }
    }
}

/// Simple three-tier complexity classification.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
pub enum ComplexityLevel {
    /// Low complexity (suitable for high compression).
    #[default]
    Low,
    /// Medium complexity.
    Medium,
    /// High complexity (requires high bitrate).
    High,
}

/// Combined content complexity report for a frame.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ContentComplexityReport {
    /// Spatial complexity metrics.
    pub spatial: SpatialComplexity,
    /// DCT energy analysis.
    pub dct: DctEnergyAnalysis,
    /// Overall complexity level.
    pub overall_level: ComplexityLevel,
    /// Estimated bits-per-pixel requirement (heuristic).
    pub estimated_bpp: f32,
}

impl ContentComplexityReport {
    /// Build a report from a luma frame (no previous frame).
    #[must_use]
    pub fn from_frame(y_plane: &[u8], width: usize, height: usize) -> Self {
        let spatial = SpatialComplexity::compute(y_plane, width, height);
        let dct = DctEnergyAnalysis::compute(y_plane, width, height);
        let overall_level = dct.complexity_level();

        // Heuristic BPP: SI * 0.01 + AC ratio * 2.0.
        let estimated_bpp = spatial.spatial_information * 0.01 + dct.ac_to_total_ratio * 2.0;

        Self {
            spatial,
            dct,
            overall_level,
            estimated_bpp,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn black_frame(w: usize, h: usize) -> Vec<u8> {
        vec![0u8; w * h]
    }

    fn white_frame(w: usize, h: usize) -> Vec<u8> {
        vec![255u8; w * h]
    }

    fn checkerboard(w: usize, h: usize) -> Vec<u8> {
        // Use 4x4 block checkerboard so the Sobel operator detects real edges
        // (a pixel-level checkerboard cancels out in the 3x3 Sobel kernel).
        (0..w * h)
            .map(|i| {
                let row = i / w;
                let col = i % w;
                if ((row / 4) + (col / 4)) % 2 == 0 {
                    200u8
                } else {
                    50u8
                }
            })
            .collect()
    }

    #[test]
    fn test_spatial_black_frame_zero_edge_density() {
        let frame = black_frame(16, 16);
        let sc = SpatialComplexity::compute(&frame, 16, 16);
        assert!((sc.edge_density).abs() < 1e-4);
    }

    #[test]
    fn test_spatial_white_frame_zero_edge_density() {
        let frame = white_frame(16, 16);
        let sc = SpatialComplexity::compute(&frame, 16, 16);
        assert!((sc.edge_density).abs() < 1e-4);
    }

    #[test]
    fn test_spatial_checkerboard_high_si() {
        let frame = checkerboard(16, 16);
        let sc = SpatialComplexity::compute(&frame, 16, 16);
        assert!(sc.spatial_information > 0.0);
        assert!(sc.edge_density > 0.0);
    }

    #[test]
    fn test_spatial_luma_variance_uniform() {
        let frame = vec![128u8; 16 * 16];
        let sc = SpatialComplexity::compute(&frame, 16, 16);
        assert!((sc.luma_variance).abs() < 1e-4);
    }

    #[test]
    fn test_spatial_luma_variance_mixed() {
        let frame: Vec<u8> = (0..256).map(|i| (i % 256) as u8).collect();
        let sc = SpatialComplexity::compute(&frame, 16, 16);
        assert!(sc.luma_variance > 0.0);
    }

    #[test]
    fn test_temporal_identical_frames_zero_mad() {
        let f = vec![128u8; 16 * 16];
        let tc = TemporalComplexity::compute(&f, &f, 16, 16, 20.0);
        assert!((tc.mean_absolute_difference).abs() < 1e-4);
        assert!(!tc.is_scene_cut);
    }

    #[test]
    fn test_temporal_different_frames_nonzero_mad() {
        let prev = black_frame(16, 16);
        let curr = white_frame(16, 16);
        let tc = TemporalComplexity::compute(&prev, &curr, 16, 16, 20.0);
        assert!(tc.mean_absolute_difference > 100.0);
        assert!(tc.is_scene_cut);
    }

    #[test]
    fn test_temporal_changed_pixel_ratio() {
        let prev = vec![0u8; 64];
        let mut curr = vec![0u8; 64];
        for i in 0..32 {
            curr[i] = 50; // > threshold 10
        }
        let tc = TemporalComplexity::compute(&prev, &curr, 8, 8, 100.0);
        assert!((tc.changed_pixel_ratio - 0.5).abs() < 0.01);
    }

    #[test]
    fn test_dct_black_frame_zero_ac() {
        let frame = black_frame(16, 16);
        let dct = DctEnergyAnalysis::compute(&frame, 16, 16);
        assert!((dct.mean_ac_energy).abs() < 1e-4);
        assert_eq!(dct.complexity_level(), ComplexityLevel::Low);
    }

    #[test]
    fn test_dct_checkerboard_high_ac() {
        let frame = checkerboard(16, 16);
        let dct = DctEnergyAnalysis::compute(&frame, 16, 16);
        assert!(dct.mean_ac_energy > 0.0);
    }

    #[test]
    fn test_dct_ac_to_total_ratio_range() {
        let frame = checkerboard(16, 16);
        let dct = DctEnergyAnalysis::compute(&frame, 16, 16);
        assert!((0.0..=1.0).contains(&dct.ac_to_total_ratio));
    }

    #[test]
    fn test_complexity_report_from_frame() {
        let frame = checkerboard(16, 16);
        let report = ContentComplexityReport::from_frame(&frame, 16, 16);
        assert!(report.estimated_bpp >= 0.0);
    }

    #[test]
    fn test_complexity_level_default() {
        assert_eq!(ComplexityLevel::default(), ComplexityLevel::Low);
    }

    #[test]
    fn test_dct_block_count() {
        let frame = black_frame(32, 16);
        let dct = DctEnergyAnalysis::compute(&frame, 32, 16);
        assert_eq!(dct.block_count, 8); // 4 cols * 2 rows
    }
}
