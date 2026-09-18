//! Occlusion detection and handling for frame interpolation.
//!
//! This module provides occlusion detection using bidirectional flow consistency,
//! which helps identify regions that become visible or hidden between frames.

use crate::error::{CvError, CvResult};
use crate::interpolate::optical_flow::FlowField;

/// Occlusion map indicating occluded regions.
///
/// An occlusion map marks pixels that are occluded in forward or backward warping,
/// which helps the blending algorithm make better decisions about which frame
/// to use for each pixel.
#[derive(Debug, Clone)]
pub struct OcclusionMap {
    /// Forward occlusion mask (true if occluded in forward direction).
    pub forward_occluded: Vec<bool>,
    /// Backward occlusion mask (true if occluded in backward direction).
    pub backward_occluded: Vec<bool>,
    /// Width of the occlusion map.
    pub width: u32,
    /// Height of the occlusion map.
    pub height: u32,
}

impl OcclusionMap {
    /// Create a new occlusion map.
    #[must_use]
    pub fn new(width: u32, height: u32) -> Self {
        let size = width as usize * height as usize;
        Self {
            forward_occluded: vec![false; size],
            backward_occluded: vec![false; size],
            width,
            height,
        }
    }

    /// Check if a pixel is occluded in the forward direction.
    #[must_use]
    pub fn is_occluded_forward(&self, x: u32, y: u32) -> bool {
        if x >= self.width || y >= self.height {
            return false;
        }

        let idx = (y * self.width + x) as usize;
        idx < self.forward_occluded.len() && self.forward_occluded[idx]
    }

    /// Check if a pixel is occluded in the backward direction.
    #[must_use]
    pub fn is_occluded_backward(&self, x: u32, y: u32) -> bool {
        if x >= self.width || y >= self.height {
            return false;
        }

        let idx = (y * self.width + x) as usize;
        idx < self.backward_occluded.len() && self.backward_occluded[idx]
    }

    /// Set forward occlusion status for a pixel.
    pub fn set_forward_occluded(&mut self, x: u32, y: u32, occluded: bool) {
        if x >= self.width || y >= self.height {
            return;
        }

        let idx = (y * self.width + x) as usize;
        if idx < self.forward_occluded.len() {
            self.forward_occluded[idx] = occluded;
        }
    }

    /// Set backward occlusion status for a pixel.
    pub fn set_backward_occluded(&mut self, x: u32, y: u32, occluded: bool) {
        if x >= self.width || y >= self.height {
            return;
        }

        let idx = (y * self.width + x) as usize;
        if idx < self.backward_occluded.len() {
            self.backward_occluded[idx] = occluded;
        }
    }

    /// Get the percentage of forward occluded pixels.
    #[must_use]
    pub fn forward_occlusion_percentage(&self) -> f32 {
        let count = self.forward_occluded.iter().filter(|&&x| x).count();
        count as f32 / self.forward_occluded.len() as f32 * 100.0
    }

    /// Get the percentage of backward occluded pixels.
    #[must_use]
    pub fn backward_occlusion_percentage(&self) -> f32 {
        let count = self.backward_occluded.iter().filter(|&&x| x).count();
        count as f32 / self.backward_occluded.len() as f32 * 100.0
    }

    /// Dilate the occlusion map to expand occluded regions.
    ///
    /// This helps handle occlusion boundaries more conservatively.
    pub fn dilate(&mut self, radius: u32) {
        let dilated_forward = self.dilate_mask(&self.forward_occluded, radius);
        let dilated_backward = self.dilate_mask(&self.backward_occluded, radius);

        self.forward_occluded = dilated_forward;
        self.backward_occluded = dilated_backward;
    }

    /// Dilate a binary mask.
    fn dilate_mask(&self, mask: &[bool], radius: u32) -> Vec<bool> {
        let mut result = vec![false; mask.len()];
        let r = radius as i32;

        for y in 0..self.height {
            for x in 0..self.width {
                let idx = (y * self.width + x) as usize;

                // Check if any neighbor is occluded
                let mut is_dilated = false;
                for dy in -r..=r {
                    for dx in -r..=r {
                        let nx = x as i32 + dx;
                        let ny = y as i32 + dy;

                        if nx >= 0 && nx < self.width as i32 && ny >= 0 && ny < self.height as i32 {
                            let nidx = (ny as u32 * self.width + nx as u32) as usize;
                            if nidx < mask.len() && mask[nidx] {
                                is_dilated = true;
                                break;
                            }
                        }
                    }
                    if is_dilated {
                        break;
                    }
                }

                result[idx] = is_dilated;
            }
        }

        result
    }
}

/// Occlusion detector using bidirectional flow consistency.
///
/// Detects occluded regions by checking the consistency between forward
/// and backward optical flow fields.
pub struct OcclusionDetector {
    /// Consistency threshold for occlusion detection.
    consistency_threshold: f32,
    /// Enable post-processing (dilation) of occlusion maps.
    enable_dilation: bool,
    /// Dilation radius for occlusion boundaries.
    dilation_radius: u32,
}

impl OcclusionDetector {
    /// Create a new occlusion detector.
    #[must_use]
    pub fn new() -> Self {
        Self {
            consistency_threshold: 1.0,
            enable_dilation: true,
            dilation_radius: 1,
        }
    }

    /// Set the consistency threshold.
    ///
    /// Higher values are more permissive (detect fewer occlusions).
    #[must_use]
    pub const fn with_threshold(mut self, threshold: f32) -> Self {
        self.consistency_threshold = threshold;
        self
    }

    /// Enable or disable dilation of occlusion maps.
    #[must_use]
    pub const fn with_dilation(mut self, enabled: bool, radius: u32) -> Self {
        self.enable_dilation = enabled;
        self.dilation_radius = radius;
        self
    }

    /// Detect occlusions using bidirectional flow consistency.
    ///
    /// # Arguments
    ///
    /// * `flow_forward` - Forward optical flow (frame1 → frame2)
    /// * `flow_backward` - Backward optical flow (frame2 → frame1)
    /// * `width` - Frame width
    /// * `height` - Frame height
    ///
    /// # Returns
    ///
    /// An occlusion map indicating which pixels are occluded in each direction.
    ///
    /// # Errors
    ///
    /// Returns an error if flow fields have incompatible dimensions.
    pub fn detect(
        &self,
        flow_forward: &FlowField,
        flow_backward: &FlowField,
        width: u32,
        height: u32,
    ) -> CvResult<OcclusionMap> {
        if flow_forward.width != width || flow_forward.height != height {
            return Err(CvError::invalid_dimensions(
                flow_forward.width,
                flow_forward.height,
            ));
        }

        if flow_backward.width != width || flow_backward.height != height {
            return Err(CvError::invalid_dimensions(
                flow_backward.width,
                flow_backward.height,
            ));
        }

        let mut occlusion_map = OcclusionMap::new(width, height);

        // Check bidirectional flow consistency
        for y in 0..height {
            for x in 0..width {
                // Forward flow consistency check
                let forward_occluded = self.check_forward_consistency(
                    flow_forward,
                    flow_backward,
                    x,
                    y,
                    width,
                    height,
                );
                occlusion_map.set_forward_occluded(x, y, forward_occluded);

                // Backward flow consistency check
                let backward_occluded = self.check_backward_consistency(
                    flow_forward,
                    flow_backward,
                    x,
                    y,
                    width,
                    height,
                );
                occlusion_map.set_backward_occluded(x, y, backward_occluded);
            }
        }

        // Apply dilation if enabled
        if self.enable_dilation {
            occlusion_map.dilate(self.dilation_radius);
        }

        Ok(occlusion_map)
    }

    /// Check forward flow consistency.
    ///
    /// A pixel is considered occluded in the forward direction if:
    /// flow_forward(x,y) + flow_backward(x + flow_forward(x,y)) ≠ 0
    fn check_forward_consistency(
        &self,
        flow_forward: &FlowField,
        flow_backward: &FlowField,
        x: u32,
        y: u32,
        width: u32,
        height: u32,
    ) -> bool {
        let (dx_fwd, dy_fwd) = flow_forward.get(x, y);

        // Compute destination position in frame2
        let x2 = (x as f32 + dx_fwd).round() as i32;
        let y2 = (y as f32 + dy_fwd).round() as i32;

        // Check bounds
        if x2 < 0 || x2 >= width as i32 || y2 < 0 || y2 >= height as i32 {
            return true; // Out of bounds = occluded
        }

        // Get backward flow at destination
        let (dx_bwd, dy_bwd) = flow_backward.get(x2 as u32, y2 as u32);

        // Check consistency: forward + backward should be close to zero
        let consistency_error = ((dx_fwd + dx_bwd).powi(2) + (dy_fwd + dy_bwd).powi(2)).sqrt();

        consistency_error > self.consistency_threshold
    }

    /// Check backward flow consistency.
    ///
    /// Similar to forward check but in the opposite direction.
    fn check_backward_consistency(
        &self,
        flow_forward: &FlowField,
        flow_backward: &FlowField,
        x: u32,
        y: u32,
        width: u32,
        height: u32,
    ) -> bool {
        let (dx_bwd, dy_bwd) = flow_backward.get(x, y);

        // Compute source position in frame1
        let x1 = (x as f32 + dx_bwd).round() as i32;
        let y1 = (y as f32 + dy_bwd).round() as i32;

        // Check bounds
        if x1 < 0 || x1 >= width as i32 || y1 < 0 || y1 >= height as i32 {
            return true; // Out of bounds = occluded
        }

        // Get forward flow at source
        let (dx_fwd, dy_fwd) = flow_forward.get(x1 as u32, y1 as u32);

        // Check consistency: backward + forward should be close to zero
        let consistency_error = ((dx_bwd + dx_fwd).powi(2) + (dy_bwd + dy_fwd).powi(2)).sqrt();

        consistency_error > self.consistency_threshold
    }
}

impl Default for OcclusionDetector {
    fn default() -> Self {
        Self::new()
    }
}
