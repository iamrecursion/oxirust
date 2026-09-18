//! Quality assessment and adaptive configuration for frame interpolation.
//!
//! This module provides quality metrics and adaptive tuning for interpolation
//! parameters based on motion characteristics and quality requirements.

use crate::error::{CvError, CvResult};
use crate::interpolate::optical_flow::FlowField;
use oximedia_codec::VideoFrame;

/// Interpolation quality metrics.
#[derive(Debug, Clone, Default)]
pub struct InterpolationQualityMetrics {
    /// Temporal consistency score (0.0 to 1.0, higher is better).
    pub temporal_consistency: f32,
    /// Motion smoothness score (0.0 to 1.0, higher is better).
    pub motion_smoothness: f32,
    /// Artifact score (0.0 to 1.0, lower is better).
    pub artifact_score: f32,
    /// Edge preservation score (0.0 to 1.0, higher is better).
    pub edge_preservation: f32,
    /// Overall quality score (0.0 to 1.0, higher is better).
    pub overall_quality: f32,
    /// Occlusion percentage (0.0 to 100.0).
    pub occlusion_percentage: f32,
    /// Average motion magnitude.
    pub avg_motion: f32,
    /// Maximum motion magnitude.
    pub max_motion: f32,
}

impl InterpolationQualityMetrics {
    /// Create new quality metrics with all scores set to zero.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Compute overall quality score from individual metrics.
    pub fn compute_overall_quality(&mut self) {
        // Weighted average of quality components
        self.overall_quality = (self.temporal_consistency * 0.3
            + self.motion_smoothness * 0.2
            + (1.0 - self.artifact_score) * 0.3
            + self.edge_preservation * 0.2)
            .clamp(0.0, 1.0);
    }

    /// Check if quality is acceptable.
    #[must_use]
    pub fn is_acceptable(&self, threshold: f32) -> bool {
        self.overall_quality >= threshold
    }

    /// Get a human-readable quality level.
    #[must_use]
    pub fn quality_level(&self) -> &str {
        if self.overall_quality >= 0.9 {
            "Excellent"
        } else if self.overall_quality >= 0.75 {
            "Good"
        } else if self.overall_quality >= 0.6 {
            "Fair"
        } else if self.overall_quality >= 0.4 {
            "Poor"
        } else {
            "Very Poor"
        }
    }
}

/// Quality assessor for interpolated frames.
///
/// Evaluates the quality of interpolated frames and provides metrics
/// for adaptive parameter tuning.
pub struct QualityAssessor {
    /// Minimum acceptable quality threshold.
    quality_threshold: f32,
    /// Enable detailed analysis.
    detailed_analysis: bool,
}

impl QualityAssessor {
    /// Create a new quality assessor.
    #[must_use]
    pub fn new() -> Self {
        Self {
            quality_threshold: 0.6,
            detailed_analysis: true,
        }
    }

    /// Set the quality threshold.
    pub fn set_threshold(&mut self, threshold: f32) {
        self.quality_threshold = threshold.clamp(0.0, 1.0);
    }

    /// Enable or disable detailed analysis.
    pub fn set_detailed_analysis(&mut self, enabled: bool) {
        self.detailed_analysis = enabled;
    }

    /// Assess interpolation quality.
    ///
    /// # Arguments
    ///
    /// * `interpolated` - The interpolated frame
    /// * `frame1` - First source frame
    /// * `frame2` - Second source frame
    /// * `flow_forward` - Forward optical flow
    /// * `flow_backward` - Backward optical flow
    ///
    /// # Returns
    ///
    /// Quality metrics for the interpolation.
    #[allow(clippy::too_many_arguments)]
    pub fn assess(
        &self,
        interpolated: &VideoFrame,
        frame1: &VideoFrame,
        frame2: &VideoFrame,
        flow_forward: &FlowField,
        flow_backward: &FlowField,
    ) -> CvResult<InterpolationQualityMetrics> {
        let mut metrics = InterpolationQualityMetrics::new();

        // Assess temporal consistency
        metrics.temporal_consistency =
            self.assess_temporal_consistency(interpolated, frame1, frame2)?;

        // Assess motion smoothness
        metrics.motion_smoothness = self.assess_motion_smoothness(flow_forward, flow_backward);

        // Assess artifacts if detailed analysis is enabled
        if self.detailed_analysis {
            metrics.artifact_score = self.assess_artifacts(interpolated, frame1, frame2)?;
            metrics.edge_preservation =
                self.assess_edge_preservation(interpolated, frame1, frame2)?;
        }

        // Compute motion statistics
        metrics.avg_motion =
            (flow_forward.average_magnitude() + flow_backward.average_magnitude()) / 2.0;
        metrics.max_motion = flow_forward
            .max_magnitude()
            .max(flow_backward.max_magnitude());

        // Compute overall quality
        metrics.compute_overall_quality();

        Ok(metrics)
    }

    /// Assess temporal consistency.
    ///
    /// Measures how well the interpolated frame fits temporally between
    /// the source frames.
    fn assess_temporal_consistency(
        &self,
        interpolated: &VideoFrame,
        frame1: &VideoFrame,
        frame2: &VideoFrame,
    ) -> CvResult<f32> {
        if interpolated.planes.is_empty() || frame1.planes.is_empty() || frame2.planes.is_empty() {
            return Ok(0.0);
        }

        let plane_interp = &interpolated.planes[0];
        let plane1 = &frame1.planes[0];
        let plane2 = &frame2.planes[0];

        let size = plane_interp
            .data
            .len()
            .min(plane1.data.len())
            .min(plane2.data.len());

        let mut diff_sum = 0.0f64;
        let mut count = 0u64;

        for i in 0..size {
            let v_interp = plane_interp.data[i] as f64;
            let v1 = plane1.data[i] as f64;
            let v2 = plane2.data[i] as f64;

            // Interpolated value should be between source values
            let expected_range = (v1.min(v2), v1.max(v2));
            let deviation = if v_interp < expected_range.0 {
                expected_range.0 - v_interp
            } else if v_interp > expected_range.1 {
                v_interp - expected_range.1
            } else {
                0.0
            };

            diff_sum += deviation;
            count += 1;
        }

        if count == 0 {
            return Ok(0.0);
        }

        let avg_deviation = diff_sum / count as f64;

        // Convert to score (0.0 to 1.0, higher is better)
        let score = (-avg_deviation / 50.0).exp() as f32;

        Ok(score.clamp(0.0, 1.0))
    }

    /// Assess motion smoothness.
    ///
    /// Measures the smoothness and consistency of the optical flow field.
    fn assess_motion_smoothness(&self, flow_forward: &FlowField, flow_backward: &FlowField) -> f32 {
        let smoothness_fwd = self.compute_flow_smoothness(flow_forward);
        let smoothness_bwd = self.compute_flow_smoothness(flow_backward);

        (smoothness_fwd + smoothness_bwd) / 2.0
    }

    /// Compute flow smoothness for a single flow field.
    fn compute_flow_smoothness(&self, flow: &FlowField) -> f32 {
        let mut diff_sum = 0.0f64;
        let mut count = 0u64;

        for y in 0..flow.height - 1 {
            for x in 0..flow.width - 1 {
                let (dx1, dy1) = flow.get(x, y);
                let (dx2, dy2) = flow.get(x + 1, y);
                let (dx3, dy3) = flow.get(x, y + 1);

                // Horizontal smoothness
                let h_diff = ((dx1 - dx2).powi(2) + (dy1 - dy2).powi(2)).sqrt();
                diff_sum += h_diff as f64;
                count += 1;

                // Vertical smoothness
                let v_diff = ((dx1 - dx3).powi(2) + (dy1 - dy3).powi(2)).sqrt();
                diff_sum += v_diff as f64;
                count += 1;
            }
        }

        if count == 0 {
            return 0.0;
        }

        let avg_diff = diff_sum / count as f64;

        // Convert to score (lower difference = higher smoothness)
        let score = (-avg_diff / 5.0).exp() as f32;

        score.clamp(0.0, 1.0)
    }

    /// Assess artifacts in the interpolated frame.
    ///
    /// Detects common artifacts like halos, ghosting, and blocking.
    fn assess_artifacts(
        &self,
        interpolated: &VideoFrame,
        frame1: &VideoFrame,
        frame2: &VideoFrame,
    ) -> CvResult<f32> {
        if interpolated.planes.is_empty() || frame1.planes.is_empty() || frame2.planes.is_empty() {
            return Ok(0.0);
        }

        let (width, height) = interpolated.plane_dimensions(0);
        let plane_interp = &interpolated.planes[0];
        let plane1 = &frame1.planes[0];
        let plane2 = &frame2.planes[0];

        let mut artifact_score = 0.0f32;

        // Check for halo artifacts (bright/dark rings)
        artifact_score += self.detect_halos(plane_interp, width, height);

        // Check for ghosting (temporal inconsistencies)
        artifact_score += self.detect_ghosting(plane_interp, plane1, plane2, width, height);

        // Check for blocking artifacts
        artifact_score += self.detect_blocking(plane_interp, width, height);

        // Normalize and invert (lower artifact = better)
        let normalized = (artifact_score / 3.0).clamp(0.0, 1.0);

        Ok(normalized)
    }

    /// Detect halo artifacts.
    fn detect_halos(&self, plane: &oximedia_codec::Plane, width: u32, height: u32) -> f32 {
        let mut halo_score = 0.0f32;
        let mut count = 0u32;

        for y in 2..height - 2 {
            for x in 2..width - 2 {
                let center = self.get_pixel_safe(plane, width, x, y);

                // Check for rings around center
                let ring1_avg = self.compute_ring_average(plane, width, height, x, y, 1);
                let ring2_avg = self.compute_ring_average(plane, width, height, x, y, 2);

                // Halo detected if there's a significant peak/valley pattern
                let diff1 = (center as f32 - ring1_avg).abs();
                let diff2 = (ring1_avg - ring2_avg).abs();

                if diff1 > 30.0 && diff2 > 20.0 {
                    halo_score += 1.0;
                }

                count += 1;
            }
        }

        if count > 0 {
            halo_score / count as f32
        } else {
            0.0
        }
    }

    /// Detect ghosting artifacts.
    fn detect_ghosting(
        &self,
        plane_interp: &oximedia_codec::Plane,
        plane1: &oximedia_codec::Plane,
        plane2: &oximedia_codec::Plane,
        width: u32,
        height: u32,
    ) -> f32 {
        let mut ghost_score = 0.0f32;
        let mut count = 0u32;

        for y in 0..height {
            for x in 0..width {
                let v_interp = self.get_pixel_safe(plane_interp, width, x, y);
                let v1 = self.get_pixel_safe(plane1, width, x, y);
                let v2 = self.get_pixel_safe(plane2, width, x, y);

                // Ghosting appears as semi-transparent copies
                // Check if interpolated value is unexpectedly different from both sources
                let diff1 = (v_interp as i32 - v1 as i32).abs();
                let diff2 = (v_interp as i32 - v2 as i32).abs();
                let diff_sources = (v1 as i32 - v2 as i32).abs();

                if diff1 > 40 && diff2 > 40 && diff_sources > 40 {
                    ghost_score += 1.0;
                }

                count += 1;
            }
        }

        if count > 0 {
            ghost_score / count as f32
        } else {
            0.0
        }
    }

    /// Detect blocking artifacts.
    fn detect_blocking(&self, plane: &oximedia_codec::Plane, width: u32, height: u32) -> f32 {
        let mut block_score = 0.0f32;
        let mut count = 0u32;
        let block_size = 8;

        // Check for discontinuities at block boundaries
        for y in (block_size..height).step_by(block_size as usize) {
            for x in 0..width - 1 {
                let v1 = self.get_pixel_safe(plane, width, x, y - 1);
                let v2 = self.get_pixel_safe(plane, width, x, y);

                let diff = (v1 as i32 - v2 as i32).abs();

                if diff > 25 {
                    block_score += 1.0;
                }

                count += 1;
            }
        }

        for y in 0..height - 1 {
            for x in (block_size..width).step_by(block_size as usize) {
                let v1 = self.get_pixel_safe(plane, width, x - 1, y);
                let v2 = self.get_pixel_safe(plane, width, x, y);

                let diff = (v1 as i32 - v2 as i32).abs();

                if diff > 25 {
                    block_score += 1.0;
                }

                count += 1;
            }
        }

        if count > 0 {
            block_score / count as f32
        } else {
            0.0
        }
    }

    /// Assess edge preservation.
    ///
    /// Measures how well edges are preserved in the interpolated frame.
    fn assess_edge_preservation(
        &self,
        interpolated: &VideoFrame,
        frame1: &VideoFrame,
        frame2: &VideoFrame,
    ) -> CvResult<f32> {
        if interpolated.planes.is_empty() || frame1.planes.is_empty() || frame2.planes.is_empty() {
            return Ok(0.0);
        }

        let (width, height) = interpolated.plane_dimensions(0);
        let plane_interp = &interpolated.planes[0];
        let plane1 = &frame1.planes[0];
        let plane2 = &frame2.planes[0];

        let mut preservation_score = 0.0f32;
        let mut count = 0u32;

        for y in 1..height - 1 {
            for x in 1..width - 1 {
                // Compute edge strength in source frames
                let edge1 = self.compute_edge_strength(plane1, width, x, y);
                let edge2 = self.compute_edge_strength(plane2, width, x, y);
                let avg_edge_source = (edge1 + edge2) / 2.0;

                // Compute edge strength in interpolated frame
                let edge_interp = self.compute_edge_strength(plane_interp, width, x, y);

                // Edges should be preserved
                if avg_edge_source > 20.0 {
                    let ratio = edge_interp / avg_edge_source;
                    preservation_score += ratio.min(1.0);
                    count += 1;
                }
            }
        }

        if count > 0 {
            Ok((preservation_score / count as f32).clamp(0.0, 1.0))
        } else {
            Ok(1.0)
        }
    }

    /// Compute edge strength at a pixel using Sobel operator.
    fn compute_edge_strength(
        &self,
        plane: &oximedia_codec::Plane,
        width: u32,
        x: u32,
        y: u32,
    ) -> f32 {
        let center = self.get_pixel_safe(plane, width, x, y) as i32;
        let left = self.get_pixel_safe(plane, width, x.saturating_sub(1), y) as i32;
        let right = self.get_pixel_safe(plane, width, x + 1, y) as i32;
        let top = self.get_pixel_safe(plane, width, x, y.saturating_sub(1)) as i32;
        let bottom = self.get_pixel_safe(plane, width, x, y + 1) as i32;

        let gx = right - left;
        let gy = bottom - top;

        ((gx * gx + gy * gy) as f32).sqrt()
    }

    /// Compute average pixel value in a ring around center.
    fn compute_ring_average(
        &self,
        plane: &oximedia_codec::Plane,
        width: u32,
        height: u32,
        cx: u32,
        cy: u32,
        radius: u32,
    ) -> f32 {
        let mut sum = 0u32;
        let mut count = 0u32;

        let r = radius as i32;

        for dy in -r..=r {
            for dx in -r..=r {
                let dist_sq = dx * dx + dy * dy;
                let r_sq = r * r;

                // Only include pixels in the ring (not inside or outside)
                if dist_sq >= r_sq && dist_sq < (r + 1) * (r + 1) {
                    let x = (cx as i32 + dx).clamp(0, width as i32 - 1) as u32;
                    let y = (cy as i32 + dy).clamp(0, height as i32 - 1) as u32;

                    sum += self.get_pixel_safe(plane, width, x, y) as u32;
                    count += 1;
                }
            }
        }

        if count > 0 {
            sum as f32 / count as f32
        } else {
            0.0
        }
    }

    /// Safely get pixel value.
    fn get_pixel_safe(&self, plane: &oximedia_codec::Plane, width: u32, x: u32, y: u32) -> u8 {
        let idx = (y * width + x) as usize;
        if idx < plane.data.len() {
            plane.data[idx]
        } else {
            0
        }
    }
}

impl Default for QualityAssessor {
    fn default() -> Self {
        Self::new()
    }
}

/// Adaptive parameter tuner for interpolation.
///
/// Automatically adjusts interpolation parameters based on motion
/// characteristics and quality feedback.
pub struct AdaptiveParameterTuner {
    /// Target quality threshold.
    target_quality: f32,
    /// Adaptation strength (0.0 to 1.0).
    adaptation_strength: f32,
}

impl AdaptiveParameterTuner {
    /// Create a new adaptive parameter tuner.
    #[must_use]
    pub fn new(target_quality: f32) -> Self {
        Self {
            target_quality: target_quality.clamp(0.0, 1.0),
            adaptation_strength: 0.5,
        }
    }

    /// Set adaptation strength.
    pub fn set_adaptation_strength(&mut self, strength: f32) {
        self.adaptation_strength = strength.clamp(0.0, 1.0);
    }

    /// Suggest parameter adjustments based on quality metrics.
    ///
    /// Returns suggested changes to window size, search range, etc.
    #[must_use]
    pub fn suggest_adjustments(
        &self,
        metrics: &InterpolationQualityMetrics,
    ) -> ParameterAdjustments {
        let mut adjustments = ParameterAdjustments::default();

        // Adjust based on overall quality
        if metrics.overall_quality < self.target_quality {
            let quality_gap = self.target_quality - metrics.overall_quality;

            // If quality is low, suggest increasing window size and pyramid levels
            adjustments.window_size_delta = (quality_gap * 10.0 * self.adaptation_strength) as i32;
            adjustments.pyramid_levels_delta = i32::from(quality_gap > 0.2);
        }

        // Adjust based on motion magnitude
        if metrics.avg_motion > 5.0 {
            // High motion: increase search range
            adjustments.search_range_delta =
                (metrics.avg_motion * 0.5 * self.adaptation_strength) as i32;
        }

        // Adjust based on artifacts
        if metrics.artifact_score > 0.3 {
            // High artifacts: suggest artifact reduction
            adjustments.enable_artifact_reduction = true;
            adjustments.artifact_reduction_strength =
                metrics.artifact_score * self.adaptation_strength;
        }

        // Adjust based on occlusion
        if metrics.occlusion_percentage > 20.0 {
            // High occlusion: ensure occlusion detection is enabled
            adjustments.enable_occlusion_detection = true;
        }

        adjustments
    }
}

/// Suggested parameter adjustments.
#[derive(Debug, Clone, Default)]
pub struct ParameterAdjustments {
    /// Change in window size (can be negative).
    pub window_size_delta: i32,
    /// Change in search range.
    pub search_range_delta: i32,
    /// Change in pyramid levels.
    pub pyramid_levels_delta: i32,
    /// Whether to enable artifact reduction.
    pub enable_artifact_reduction: bool,
    /// Artifact reduction strength.
    pub artifact_reduction_strength: f32,
    /// Whether to enable occlusion detection.
    pub enable_occlusion_detection: bool,
}

impl ParameterAdjustments {
    /// Check if any adjustments are suggested.
    #[must_use]
    pub fn has_adjustments(&self) -> bool {
        self.window_size_delta != 0
            || self.search_range_delta != 0
            || self.pyramid_levels_delta != 0
            || self.enable_artifact_reduction
            || self.enable_occlusion_detection
    }
}
