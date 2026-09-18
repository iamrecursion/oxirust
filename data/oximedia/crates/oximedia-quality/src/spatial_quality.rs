//! Spatial quality metrics.
//!
//! Computes spatial quality descriptors such as edge sharpness, ringing
//! artifact detection, and spatial information (SI) for a video frame.

#![allow(dead_code)]
#![allow(clippy::cast_precision_loss)]

use serde::{Deserialize, Serialize};

/// Spatial information (SI) value as defined in ITU-T P.910.
///
/// SI is the standard deviation of the Sobel-filtered luma plane,
/// and ranges from 0 (flat content) to ~200+ (highly detailed content).
#[must_use]
pub fn spatial_information(luma: &[u8], width: usize, height: usize) -> f64 {
    if width < 3 || height < 3 || luma.len() < width * height {
        return 0.0;
    }

    let mut sobel_values: Vec<f64> = Vec::with_capacity((width - 2) * (height - 2));

    for y in 1..height - 1 {
        for x in 1..width - 1 {
            let px = |dy: isize, dx: isize| -> f64 {
                let row = (y as isize + dy) as usize;
                let col = (x as isize + dx) as usize;
                f64::from(luma[row * width + col])
            };

            // Sobel kernels
            let gx =
                -px(-1, -1) + px(-1, 1) - 2.0 * px(0, -1) + 2.0 * px(0, 1) - px(1, -1) + px(1, 1);
            let gy =
                -px(-1, -1) - 2.0 * px(-1, 0) - px(-1, 1) + px(1, -1) + 2.0 * px(1, 0) + px(1, 1);

            sobel_values.push((gx * gx + gy * gy).sqrt());
        }
    }

    std_dev(&sobel_values)
}

/// Computes standard deviation of a slice.
fn std_dev(values: &[f64]) -> f64 {
    if values.is_empty() {
        return 0.0;
    }
    let n = values.len() as f64;
    let mean = values.iter().sum::<f64>() / n;
    let var = values.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / n;
    var.sqrt()
}

/// Edge sharpness measurement result.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EdgeSharpness {
    /// Mean edge response (Sobel magnitude) over all detected edges
    pub mean_edge_response: f64,
    /// Estimated blur radius in pixels (0 = no blur)
    pub estimated_blur_radius: f64,
    /// Sharpness score in [0, 100]
    pub sharpness_score: f64,
    /// Number of edge pixels detected
    pub edge_pixel_count: usize,
}

impl EdgeSharpness {
    /// Computes edge sharpness from a luma plane.
    ///
    /// `edge_threshold` is the minimum Sobel magnitude to count as an edge.
    #[must_use]
    pub fn compute(luma: &[u8], width: usize, height: usize, edge_threshold: f64) -> Self {
        if width < 3 || height < 3 || luma.len() < width * height {
            return Self {
                mean_edge_response: 0.0,
                estimated_blur_radius: 0.0,
                sharpness_score: 0.0,
                edge_pixel_count: 0,
            };
        }

        let mut edge_responses: Vec<f64> = Vec::new();

        for y in 1..height - 1 {
            for x in 1..width - 1 {
                let px = |dy: isize, dx: isize| -> f64 {
                    let row = (y as isize + dy) as usize;
                    let col = (x as isize + dx) as usize;
                    f64::from(luma[row * width + col])
                };

                let gx = -px(-1, -1) + px(-1, 1) - 2.0 * px(0, -1) + 2.0 * px(0, 1) - px(1, -1)
                    + px(1, 1);
                let gy = -px(-1, -1) - 2.0 * px(-1, 0) - px(-1, 1)
                    + px(1, -1)
                    + 2.0 * px(1, 0)
                    + px(1, 1);

                let magnitude = (gx * gx + gy * gy).sqrt();
                if magnitude >= edge_threshold {
                    edge_responses.push(magnitude);
                }
            }
        }

        if edge_responses.is_empty() {
            return Self {
                mean_edge_response: 0.0,
                estimated_blur_radius: 0.0,
                sharpness_score: 0.0,
                edge_pixel_count: 0,
            };
        }

        let mean_response = edge_responses.iter().sum::<f64>() / edge_responses.len() as f64;

        // Estimate blur radius: stronger edges → sharper → lower blur radius
        // Max Sobel response for a perfect step edge is ~1442 (255*sqrt(2)*4)
        let max_possible = 1442.0;
        let blur_radius = ((max_possible - mean_response) / max_possible * 5.0).max(0.0);

        // Score: clamp mean_response / max_possible into [0, 100]
        let sharpness_score = (mean_response / max_possible * 100.0).clamp(0.0, 100.0);

        Self {
            mean_edge_response: mean_response,
            estimated_blur_radius: blur_radius,
            sharpness_score,
            edge_pixel_count: edge_responses.len(),
        }
    }
}

/// Ringing artifact detector.
///
/// Ringing manifests as oscillations near sharp edges, common in
/// over-sharpened or heavily compressed images.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RingingDetector {
    /// Minimum edge strength to examine for ringing
    pub edge_threshold: f64,
    /// Search radius around edges for ringing oscillations (pixels)
    pub search_radius: usize,
    /// Minimum oscillation amplitude to count as ringing
    pub min_oscillation_amplitude: f64,
}

impl Default for RingingDetector {
    fn default() -> Self {
        Self {
            edge_threshold: 20.0,
            search_radius: 5,
            min_oscillation_amplitude: 5.0,
        }
    }
}

impl RingingDetector {
    /// Estimates a ringing index in [0, 100] for a luma plane.
    ///
    /// A score of 0 means no ringing; 100 means severe ringing.
    #[must_use]
    pub fn ringing_index(&self, luma: &[u8], width: usize, height: usize) -> f64 {
        if width < 5 || height < 5 || luma.len() < width * height {
            return 0.0;
        }

        let mut ring_scores: Vec<f64> = Vec::new();

        let r = self.search_radius.min((width / 2).min(height / 2));

        for y in r..height - r {
            for x in r..width - r {
                // Check for an edge at this pixel using a simple horizontal gradient
                let left = f64::from(luma[y * width + x.saturating_sub(1)]);
                let right = f64::from(luma[y * width + (x + 1).min(width - 1)]);
                let edge_strength = (right - left).abs();

                if edge_strength < self.edge_threshold {
                    continue;
                }

                // Sample the row left of the edge to detect oscillations
                let mut prev = f64::from(luma[y * width + x]);
                let mut sign_changes = 0_usize;
                let mut last_diff = 0.0_f64;

                for dx in 1..=r {
                    if x + dx >= width {
                        break;
                    }
                    let cur = f64::from(luma[y * width + x + dx]);
                    let diff = cur - prev;
                    if last_diff.abs() > self.min_oscillation_amplitude
                        && diff.signum() != last_diff.signum()
                        && diff.abs() > self.min_oscillation_amplitude
                    {
                        sign_changes += 1;
                    }
                    last_diff = diff;
                    prev = cur;
                }

                if sign_changes > 0 {
                    ring_scores.push(sign_changes as f64 * edge_strength);
                }
            }
        }

        if ring_scores.is_empty() {
            return 0.0;
        }

        let mean_ring = ring_scores.iter().sum::<f64>() / ring_scores.len() as f64;
        // Normalise to [0, 100] with a rough scale
        (mean_ring / 500.0 * 100.0).clamp(0.0, 100.0)
    }
}

/// Summary of spatial quality for a single frame.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SpatialQualitySummary {
    /// Spatial information (SI) value
    pub spatial_information: f64,
    /// Edge sharpness measurement
    pub edge_sharpness: EdgeSharpness,
    /// Ringing index [0, 100]
    pub ringing_index: f64,
    /// Overall spatial quality score in [0, 100]
    pub overall_score: f64,
}

impl SpatialQualitySummary {
    /// Computes a full spatial quality summary for a luma plane.
    #[must_use]
    pub fn compute(luma: &[u8], width: usize, height: usize) -> Self {
        let si = spatial_information(luma, width, height);
        let sharpness = EdgeSharpness::compute(luma, width, height, 20.0);
        let detector = RingingDetector::default();
        let ringing = detector.ringing_index(luma, width, height);

        // Composite score: favour sharpness, penalise ringing
        let overall_score =
            (sharpness.sharpness_score * 0.7 + (100.0 - ringing) * 0.3).clamp(0.0, 100.0);

        Self {
            spatial_information: si,
            edge_sharpness: sharpness,
            ringing_index: ringing,
            overall_score,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_checkerboard(width: usize, height: usize) -> Vec<u8> {
        (0..height)
            .flat_map(|y| (0..width).map(move |x| if (x + y) % 2 == 0 { 255u8 } else { 0u8 }))
            .collect()
    }

    fn make_flat(width: usize, height: usize, value: u8) -> Vec<u8> {
        vec![value; width * height]
    }

    #[test]
    fn test_spatial_information_flat_is_low() {
        let luma = make_flat(32, 32, 128);
        let si = spatial_information(&luma, 32, 32);
        assert!(si < 5.0, "flat image should have low SI, got {si}");
    }

    #[test]
    fn test_spatial_information_checkerboard_uniform_gradient() {
        // A perfect checkerboard has uniform Sobel magnitudes, so std_dev is low.
        let luma = make_checkerboard(32, 32);
        let si = spatial_information(&luma, 32, 32);
        assert!(
            si < 5.0,
            "uniform checkerboard gradient std-dev should be low, got {si}"
        );
    }

    #[test]
    fn test_spatial_information_too_small() {
        let si = spatial_information(&[128u8; 4], 2, 2);
        assert_eq!(si, 0.0);
    }

    #[test]
    fn test_edge_sharpness_flat_image() {
        let luma = make_flat(16, 16, 200);
        let result = EdgeSharpness::compute(&luma, 16, 16, 20.0);
        assert_eq!(result.edge_pixel_count, 0);
        assert_eq!(result.sharpness_score, 0.0);
    }

    #[test]
    fn test_edge_sharpness_high_contrast_image() {
        // Left half black, right half white
        let width = 32;
        let height = 32;
        let luma: Vec<u8> = (0..height)
            .flat_map(|_| (0..width).map(|x| if x < width / 2 { 0u8 } else { 255u8 }))
            .collect();
        let result = EdgeSharpness::compute(&luma, width, height, 10.0);
        assert!(result.edge_pixel_count > 0);
        assert!(result.mean_edge_response > 0.0);
    }

    #[test]
    fn test_edge_sharpness_too_small() {
        let luma = make_flat(2, 2, 128);
        let result = EdgeSharpness::compute(&luma, 2, 2, 20.0);
        assert_eq!(result.edge_pixel_count, 0);
    }

    #[test]
    fn test_ringing_detector_flat_image() {
        let luma = make_flat(32, 32, 128);
        let detector = RingingDetector::default();
        let index = detector.ringing_index(&luma, 32, 32);
        assert_eq!(index, 0.0, "flat image should have no ringing");
    }

    #[test]
    fn test_ringing_detector_too_small() {
        let luma = make_flat(4, 4, 128);
        let detector = RingingDetector::default();
        let index = detector.ringing_index(&luma, 4, 4);
        assert_eq!(index, 0.0);
    }

    #[test]
    fn test_ringing_detector_index_in_range() {
        let luma = make_checkerboard(64, 64);
        let detector = RingingDetector::default();
        let index = detector.ringing_index(&luma, 64, 64);
        assert!((0.0..=100.0).contains(&index));
    }

    #[test]
    fn test_spatial_quality_summary_flat() {
        let luma = make_flat(32, 32, 100);
        let summary = SpatialQualitySummary::compute(&luma, 32, 32);
        assert!(summary.spatial_information < 5.0);
        assert!(summary.overall_score >= 0.0);
        assert!(summary.overall_score <= 100.0);
    }

    #[test]
    fn test_spatial_quality_summary_checkerboard() {
        // Uniform checkerboard produces uniform Sobel magnitudes -> low std_dev SI.
        let luma = make_checkerboard(64, 64);
        let summary = SpatialQualitySummary::compute(&luma, 64, 64);
        assert!(
            summary.spatial_information < 5.0,
            "uniform gradient -> low SI"
        );
        assert!((0.0..=100.0).contains(&summary.overall_score));
    }

    #[test]
    fn test_std_dev_uniform() {
        let v = vec![5.0f64; 10];
        assert!(std_dev(&v).abs() < 1e-9);
    }

    #[test]
    fn test_std_dev_empty() {
        assert_eq!(std_dev(&[]), 0.0);
    }
}
