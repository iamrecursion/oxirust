//! Blending strategies for combining warped frames.
//!
//! This module provides various blending modes for combining forward and backward
//! warped frames in frame interpolation, including motion-aware and occlusion-aware
//! blending.

use crate::error::{CvError, CvResult};
use crate::interpolate::occlusion::OcclusionMap;
use crate::interpolate::optical_flow::FlowField;
use bytes::Bytes;
use oximedia_codec::{Plane, VideoFrame};

/// Blending mode for combining warped frames.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum BlendMode {
    /// Simple linear blending based on time position.
    Linear,
    /// Motion-weighted blending based on flow magnitude.
    MotionWeighted,
    /// Adaptive blending using both motion and occlusion information.
    #[default]
    Adaptive,
}

/// Frame blender for combining warped frames.
///
/// Provides various blending strategies to combine forward and backward
/// warped frames while handling occlusions and motion discontinuities.
pub struct Blender {
    /// Blending mode.
    mode: BlendMode,
    /// Motion threshold for motion-weighted blending.
    motion_threshold: f32,
    /// Occlusion threshold for adaptive blending.
    occlusion_threshold: f32,
}

impl Blender {
    /// Create a new blender with the specified mode.
    #[must_use]
    pub fn new(mode: BlendMode) -> Self {
        Self {
            mode,
            motion_threshold: 2.0,
            occlusion_threshold: 0.5,
        }
    }

    /// Set the blending mode.
    pub fn set_mode(&mut self, mode: BlendMode) {
        self.mode = mode;
    }

    /// Set the motion threshold for motion-weighted blending.
    pub fn set_motion_threshold(&mut self, threshold: f32) {
        self.motion_threshold = threshold;
    }

    /// Set the occlusion threshold for adaptive blending.
    pub fn set_occlusion_threshold(&mut self, threshold: f32) {
        self.occlusion_threshold = threshold;
    }

    /// Blend two warped frames.
    ///
    /// # Arguments
    ///
    /// * `warped_from_1` - Frame warped from the first source frame
    /// * `warped_from_2` - Frame warped from the second source frame
    /// * `t` - Time position (0.0 to 1.0)
    /// * `flow_forward` - Forward optical flow
    /// * `flow_backward` - Backward optical flow
    /// * `occlusion_map` - Optional occlusion map
    #[allow(clippy::too_many_arguments)]
    pub fn blend(
        &self,
        warped_from_1: &VideoFrame,
        warped_from_2: &VideoFrame,
        t: f32,
        flow_forward: &FlowField,
        flow_backward: &FlowField,
        occlusion_map: Option<&OcclusionMap>,
    ) -> CvResult<VideoFrame> {
        self.validate_frames(warped_from_1, warped_from_2)?;

        let mut result = VideoFrame::new(
            warped_from_1.format,
            warped_from_1.width,
            warped_from_1.height,
        );
        result.allocate();
        result.timestamp = warped_from_1.timestamp;
        result.frame_type = warped_from_1.frame_type;
        result.color_info = warped_from_1.color_info;

        // Blend each plane
        for plane_idx in 0..warped_from_1.planes.len() {
            let (plane_w, plane_h) = warped_from_1.plane_dimensions(plane_idx);

            self.blend_plane(
                &warped_from_1.planes[plane_idx],
                &warped_from_2.planes[plane_idx],
                &mut result.planes[plane_idx],
                plane_w,
                plane_h,
                t,
                flow_forward,
                flow_backward,
                occlusion_map,
            )?;
        }

        Ok(result)
    }

    /// Blend two planes.
    #[allow(clippy::too_many_arguments)]
    fn blend_plane(
        &self,
        plane1: &Plane,
        plane2: &Plane,
        result_plane: &mut Plane,
        width: u32,
        height: u32,
        t: f32,
        flow_forward: &FlowField,
        flow_backward: &FlowField,
        occlusion_map: Option<&OcclusionMap>,
    ) -> CvResult<()> {
        let mut result_data = vec![0u8; (width * height) as usize];

        // Scale factor for plane (for chroma subsampling)
        let scale_x = width as f32 / flow_forward.width as f32;
        let scale_y = height as f32 / flow_forward.height as f32;

        match self.mode {
            BlendMode::Linear => {
                self.blend_linear(plane1, plane2, &mut result_data, width, height, t)?;
            }
            BlendMode::MotionWeighted => {
                self.blend_motion_weighted(
                    plane1,
                    plane2,
                    &mut result_data,
                    width,
                    height,
                    t,
                    flow_forward,
                    flow_backward,
                    scale_x,
                    scale_y,
                )?;
            }
            BlendMode::Adaptive => {
                self.blend_adaptive(
                    plane1,
                    plane2,
                    &mut result_data,
                    width,
                    height,
                    t,
                    flow_forward,
                    flow_backward,
                    occlusion_map,
                    scale_x,
                    scale_y,
                )?;
            }
        }

        result_plane.data = result_data;
        result_plane.stride = width as usize;

        Ok(())
    }

    /// Simple linear blending.
    fn blend_linear(
        &self,
        plane1: &Plane,
        plane2: &Plane,
        result: &mut [u8],
        width: u32,
        height: u32,
        t: f32,
    ) -> CvResult<()> {
        let w1 = 1.0 - t;
        let w2 = t;

        for y in 0..height {
            for x in 0..width {
                let idx = (y * width + x) as usize;
                if idx < plane1.data.len() && idx < plane2.data.len() {
                    let v1 = plane1.data[idx] as f32;
                    let v2 = plane2.data[idx] as f32;
                    result[idx] = (v1 * w1 + v2 * w2).round().clamp(0.0, 255.0) as u8;
                }
            }
        }

        Ok(())
    }

    /// Motion-weighted blending.
    #[allow(clippy::too_many_arguments)]
    fn blend_motion_weighted(
        &self,
        plane1: &Plane,
        plane2: &Plane,
        result: &mut [u8],
        width: u32,
        height: u32,
        t: f32,
        flow_forward: &FlowField,
        flow_backward: &FlowField,
        scale_x: f32,
        scale_y: f32,
    ) -> CvResult<()> {
        for y in 0..height {
            for x in 0..width {
                let idx = (y * width + x) as usize;
                if idx >= plane1.data.len() || idx >= plane2.data.len() {
                    continue;
                }

                // Get flow magnitudes
                let flow_x = (x as f32 / scale_x) as u32;
                let flow_y = (y as f32 / scale_y) as u32;

                let mag_forward = flow_forward.magnitude(flow_x, flow_y);
                let mag_backward = flow_backward.magnitude(flow_x, flow_y);

                // Compute weights based on motion reliability
                // Lower motion magnitude often indicates more reliable regions
                let w1 = self.motion_weight(mag_forward);
                let w2 = self.motion_weight(mag_backward);

                // Normalize weights
                let total_w = w1 + w2;
                let (w1_norm, w2_norm) = if total_w > 0.0 {
                    (w1 / total_w, w2 / total_w)
                } else {
                    (0.5, 0.5)
                };

                // Apply temporal weighting
                let w1_final = w1_norm * (1.0 - t);
                let w2_final = w2_norm * t;

                let v1 = plane1.data[idx] as f32;
                let v2 = plane2.data[idx] as f32;

                result[idx] = (v1 * w1_final + v2 * w2_final).round().clamp(0.0, 255.0) as u8;
            }
        }

        Ok(())
    }

    /// Adaptive blending using motion and occlusion information.
    #[allow(clippy::too_many_arguments)]
    fn blend_adaptive(
        &self,
        plane1: &Plane,
        plane2: &Plane,
        result: &mut [u8],
        width: u32,
        height: u32,
        t: f32,
        flow_forward: &FlowField,
        flow_backward: &FlowField,
        occlusion_map: Option<&OcclusionMap>,
        scale_x: f32,
        scale_y: f32,
    ) -> CvResult<()> {
        for y in 0..height {
            for x in 0..width {
                let idx = (y * width + x) as usize;
                if idx >= plane1.data.len() || idx >= plane2.data.len() {
                    continue;
                }

                // Get flow magnitudes
                let flow_x = (x as f32 / scale_x) as u32;
                let flow_y = (y as f32 / scale_y) as u32;

                let mag_forward = flow_forward.magnitude(flow_x, flow_y);
                let mag_backward = flow_backward.magnitude(flow_x, flow_y);

                // Check occlusion
                let (is_occluded_forward, is_occluded_backward) =
                    if let Some(occ_map) = occlusion_map {
                        (
                            occ_map.is_occluded_forward(flow_x, flow_y),
                            occ_map.is_occluded_backward(flow_x, flow_y),
                        )
                    } else {
                        (false, false)
                    };

                // Compute adaptive weights
                let w1 = if is_occluded_forward {
                    0.0
                } else {
                    self.motion_weight(mag_forward) * (1.0 - t)
                };

                let w2 = if is_occluded_backward {
                    0.0
                } else {
                    self.motion_weight(mag_backward) * t
                };

                // Normalize weights
                let total_w = w1 + w2;
                let (w1_final, w2_final) = if total_w > 0.0 {
                    (w1 / total_w, w2 / total_w)
                } else {
                    // If both are occluded or zero weight, use temporal weighting
                    (1.0 - t, t)
                };

                let v1 = plane1.data[idx] as f32;
                let v2 = plane2.data[idx] as f32;

                result[idx] = (v1 * w1_final + v2 * w2_final).round().clamp(0.0, 255.0) as u8;
            }
        }

        Ok(())
    }

    /// Compute motion-based weight.
    ///
    /// Returns a weight that decreases with motion magnitude, as high motion
    /// areas are often less reliable for interpolation.
    fn motion_weight(&self, magnitude: f32) -> f32 {
        if magnitude < self.motion_threshold {
            1.0
        } else {
            // Exponential decay for high motion
            (-((magnitude - self.motion_threshold) / 10.0).powi(2)).exp()
        }
    }

    /// Validate that frames are compatible for blending.
    fn validate_frames(&self, frame1: &VideoFrame, frame2: &VideoFrame) -> CvResult<()> {
        if frame1.width != frame2.width || frame1.height != frame2.height {
            return Err(CvError::invalid_dimensions(frame1.width, frame1.height));
        }

        if frame1.format != frame2.format {
            return Err(CvError::unsupported_format("Frame format mismatch"));
        }

        if frame1.planes.len() != frame2.planes.len() {
            return Err(CvError::insufficient_data(
                frame1.planes.len(),
                frame2.planes.len(),
            ));
        }

        Ok(())
    }
}

impl Default for Blender {
    fn default() -> Self {
        Self::new(BlendMode::Adaptive)
    }
}
