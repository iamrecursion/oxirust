//! Artifact reduction for interpolated frames.
//!
//! This module provides techniques to reduce common interpolation artifacts
//! including halos, ghosting, blocking, and blurring.

use crate::error::{CvError, CvResult};
use bytes::Bytes;
use oximedia_codec::{Plane, VideoFrame};

/// Type of artifact to reduce.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ArtifactType {
    /// Halo artifacts around moving objects.
    Halo,
    /// Ghosting from motion blur.
    Ghosting,
    /// Blocking artifacts from block-based methods.
    Blocking,
    /// General blur from interpolation.
    Blur,
}

/// Artifact reduction configuration.
#[derive(Debug, Clone)]
#[allow(clippy::struct_excessive_bools)]
pub struct ArtifactReductionConfig {
    /// Enable halo reduction.
    pub reduce_halo: bool,
    /// Enable ghosting reduction.
    pub reduce_ghosting: bool,
    /// Enable blocking reduction.
    pub reduce_blocking: bool,
    /// Enable sharpening to counter blur.
    pub sharpen: bool,
    /// Halo reduction strength (0.0 to 1.0).
    pub halo_strength: f32,
    /// Ghosting reduction strength (0.0 to 1.0).
    pub ghosting_strength: f32,
    /// Median filter size for blocking reduction.
    pub median_filter_size: usize,
    /// Sharpening strength (0.0 to 1.0).
    pub sharpen_strength: f32,
}

impl Default for ArtifactReductionConfig {
    fn default() -> Self {
        Self {
            reduce_halo: true,
            reduce_ghosting: true,
            reduce_blocking: false,
            sharpen: true,
            halo_strength: 0.5,
            ghosting_strength: 0.3,
            median_filter_size: 3,
            sharpen_strength: 0.2,
        }
    }
}

/// Artifact reducer for interpolated frames.
///
/// Provides various techniques to reduce common artifacts that appear
/// in frame interpolation.
pub struct ArtifactReducer {
    /// Configuration.
    config: ArtifactReductionConfig,
}

impl ArtifactReducer {
    /// Create a new artifact reducer with default configuration.
    #[must_use]
    pub fn new() -> Self {
        Self {
            config: ArtifactReductionConfig::default(),
        }
    }

    /// Create an artifact reducer with custom configuration.
    #[must_use]
    pub fn with_config(config: ArtifactReductionConfig) -> Self {
        Self { config }
    }

    /// Set the configuration.
    pub fn set_config(&mut self, config: ArtifactReductionConfig) {
        self.config = config;
    }

    /// Apply artifact reduction to a frame.
    ///
    /// # Arguments
    ///
    /// * `frame` - The interpolated frame to process
    /// * `frame1` - First source frame (for reference)
    /// * `frame2` - Second source frame (for reference)
    ///
    /// # Errors
    ///
    /// Returns an error if frame processing fails.
    pub fn reduce_artifacts(
        &self,
        frame: &VideoFrame,
        frame1: Option<&VideoFrame>,
        frame2: Option<&VideoFrame>,
    ) -> CvResult<VideoFrame> {
        let mut result = frame.clone();

        // Apply various artifact reduction techniques in sequence
        if self.config.reduce_halo {
            if let (Some(f1), Some(f2)) = (frame1, frame2) {
                result = self.reduce_halo(&result, f1, f2)?;
            }
        }

        if self.config.reduce_ghosting {
            result = self.reduce_ghosting(&result)?;
        }

        if self.config.reduce_blocking {
            result = self.reduce_blocking(&result)?;
        }

        if self.config.sharpen {
            result = self.sharpen(&result)?;
        }

        Ok(result)
    }

    /// Reduce halo artifacts.
    ///
    /// Halos appear as bright or dark rings around moving objects.
    /// This method detects and reduces them using bilateral filtering
    /// guided by the source frames.
    fn reduce_halo(
        &self,
        frame: &VideoFrame,
        frame1: &VideoFrame,
        frame2: &VideoFrame,
    ) -> CvResult<VideoFrame> {
        let mut result = frame.clone();

        for plane_idx in 0..frame.planes.len() {
            let (width, height) = frame.plane_dimensions(plane_idx);

            let filtered = self.bilateral_filter(
                &frame.planes[plane_idx],
                &frame1.planes[plane_idx],
                &frame2.planes[plane_idx],
                width,
                height,
            )?;

            // Blend filtered result with original based on strength
            let blended = self.blend_planes(
                &frame.planes[plane_idx],
                &filtered,
                self.config.halo_strength,
                width,
                height,
            )?;

            result.planes[plane_idx] = blended;
        }

        Ok(result)
    }

    /// Reduce ghosting artifacts.
    ///
    /// Ghosting appears as semi-transparent copies of moving objects.
    /// This method uses temporal median filtering to reduce it.
    fn reduce_ghosting(&self, frame: &VideoFrame) -> CvResult<VideoFrame> {
        let mut result = frame.clone();

        for plane_idx in 0..frame.planes.len() {
            let (width, height) = frame.plane_dimensions(plane_idx);

            let filtered = self.temporal_median_filter(&frame.planes[plane_idx], width, height)?;

            // Blend filtered result with original
            let blended = self.blend_planes(
                &frame.planes[plane_idx],
                &filtered,
                self.config.ghosting_strength,
                width,
                height,
            )?;

            result.planes[plane_idx] = blended;
        }

        Ok(result)
    }

    /// Reduce blocking artifacts.
    ///
    /// Blocking artifacts appear as visible block boundaries from
    /// block-based optical flow methods.
    fn reduce_blocking(&self, frame: &VideoFrame) -> CvResult<VideoFrame> {
        let mut result = frame.clone();

        for plane_idx in 0..frame.planes.len() {
            let (width, height) = frame.plane_dimensions(plane_idx);

            let filtered = self.deblock_filter(&frame.planes[plane_idx], width, height)?;

            result.planes[plane_idx] = filtered;
        }

        Ok(result)
    }

    /// Apply sharpening to counter interpolation blur.
    fn sharpen(&self, frame: &VideoFrame) -> CvResult<VideoFrame> {
        let mut result = frame.clone();

        for plane_idx in 0..frame.planes.len() {
            let (width, height) = frame.plane_dimensions(plane_idx);

            let sharpened = self.unsharp_mask(
                &frame.planes[plane_idx],
                width,
                height,
                self.config.sharpen_strength,
            )?;

            result.planes[plane_idx] = sharpened;
        }

        Ok(result)
    }

    /// Bilateral filter for edge-preserving smoothing.
    #[allow(clippy::too_many_arguments)]
    fn bilateral_filter(
        &self,
        plane: &Plane,
        reference1: &Plane,
        reference2: &Plane,
        width: u32,
        height: u32,
    ) -> CvResult<Plane> {
        let mut result_data = vec![0u8; (width * height) as usize];
        let radius = 2;
        let sigma_spatial = 3.0;
        let sigma_range = 30.0;

        for y in 0..height {
            for x in 0..width {
                let mut sum_weight = 0.0f32;
                let mut sum_value = 0.0f32;

                let center_idx = (y * width + x) as usize;
                let center_val = plane.data[center_idx] as f32;

                // Get reference values for guidance
                let ref1_val = if center_idx < reference1.data.len() {
                    reference1.data[center_idx] as f32
                } else {
                    center_val
                };
                let ref2_val = if center_idx < reference2.data.len() {
                    reference2.data[center_idx] as f32
                } else {
                    center_val
                };

                for dy in -radius..=radius {
                    for dx in -radius..=radius {
                        let nx = x as i32 + dx;
                        let ny = y as i32 + dy;

                        if nx >= 0 && nx < width as i32 && ny >= 0 && ny < height as i32 {
                            let nidx = (ny as u32 * width + nx as u32) as usize;
                            if nidx < plane.data.len() {
                                let neighbor_val = plane.data[nidx] as f32;

                                // Spatial weight
                                let spatial_dist =
                                    ((dx * dx + dy * dy) as f32).sqrt() / sigma_spatial;
                                let spatial_weight = (-spatial_dist * spatial_dist / 2.0).exp();

                                // Range weight (intensity difference)
                                let range_diff = (neighbor_val - center_val).abs() / sigma_range;
                                let range_weight = (-range_diff * range_diff / 2.0).exp();

                                // Reference guidance weight
                                let ref_diff = ((neighbor_val - ref1_val).abs()
                                    + (neighbor_val - ref2_val).abs())
                                    / 2.0
                                    / sigma_range;
                                let ref_weight = (-ref_diff * ref_diff / 2.0).exp();

                                let weight = spatial_weight * range_weight * ref_weight;

                                sum_weight += weight;
                                sum_value += neighbor_val * weight;
                            }
                        }
                    }
                }

                if sum_weight > 0.0 {
                    result_data[center_idx] =
                        (sum_value / sum_weight).round().clamp(0.0, 255.0) as u8;
                } else {
                    result_data[center_idx] = plane.data[center_idx];
                }
            }
        }

        Ok(Plane {
            data: result_data,
            stride: width as usize,
            width,
            height,
        })
    }

    /// Temporal median filter.
    fn temporal_median_filter(&self, plane: &Plane, width: u32, height: u32) -> CvResult<Plane> {
        let mut result_data = vec![0u8; (width * height) as usize];
        let radius = 1;

        for y in 0..height {
            for x in 0..width {
                let mut values = Vec::new();
                let center_idx = (y * width + x) as usize;

                // Collect neighborhood values
                for dy in -radius..=radius {
                    for dx in -radius..=radius {
                        let nx = x as i32 + dx;
                        let ny = y as i32 + dy;

                        if nx >= 0 && nx < width as i32 && ny >= 0 && ny < height as i32 {
                            let nidx = (ny as u32 * width + nx as u32) as usize;
                            if nidx < plane.data.len() {
                                values.push(plane.data[nidx]);
                            }
                        }
                    }
                }

                // Compute median
                if values.is_empty() {
                    result_data[center_idx] = plane.data[center_idx];
                } else {
                    values.sort_unstable();
                    result_data[center_idx] = values[values.len() / 2];
                }
            }
        }

        Ok(Plane {
            data: result_data,
            stride: width as usize,
            width,
            height,
        })
    }

    /// Deblocking filter.
    fn deblock_filter(&self, plane: &Plane, width: u32, height: u32) -> CvResult<Plane> {
        let mut result_data = plane.data.clone();
        let block_size = 8; // Common block size

        // Apply smoothing at block boundaries
        for y in (0..height).step_by(block_size as usize) {
            for x in 0..width {
                if y > 0 && y < height {
                    self.smooth_boundary_horizontal(&mut result_data, width, x, y);
                }
            }
        }

        for y in 0..height {
            for x in (0..width).step_by(block_size as usize) {
                if x > 0 && x < width {
                    self.smooth_boundary_vertical(&mut result_data, width, x, y);
                }
            }
        }

        Ok(Plane {
            data: result_data,
            stride: width as usize,
            width,
            height,
        })
    }

    /// Smooth horizontal boundary.
    fn smooth_boundary_horizontal(&self, data: &mut [u8], width: u32, x: u32, y: u32) {
        if y < 2 || y >= width - 2 {
            return;
        }

        let idx = (y * width + x) as usize;
        let idx_above = ((y - 1) * width + x) as usize;
        let idx_below = ((y + 1) * width + x) as usize;

        if idx < data.len() && idx_above < data.len() && idx_below < data.len() {
            let diff = (data[idx_above] as i32 - data[idx_below] as i32).abs();

            // Only smooth if there's a significant discontinuity
            if diff > 20 {
                let smoothed =
                    ((data[idx_above] as u32 + data[idx] as u32 * 2 + data[idx_below] as u32) / 4)
                        as u8;
                data[idx] = smoothed;
            }
        }
    }

    /// Smooth vertical boundary.
    fn smooth_boundary_vertical(&self, data: &mut [u8], width: u32, x: u32, y: u32) {
        if x < 2 || x >= width - 2 {
            return;
        }

        let idx = (y * width + x) as usize;
        let idx_left = (y * width + (x - 1)) as usize;
        let idx_right = (y * width + (x + 1)) as usize;

        if idx < data.len() && idx_left < data.len() && idx_right < data.len() {
            let diff = (data[idx_left] as i32 - data[idx_right] as i32).abs();

            // Only smooth if there's a significant discontinuity
            if diff > 20 {
                let smoothed =
                    ((data[idx_left] as u32 + data[idx] as u32 * 2 + data[idx_right] as u32) / 4)
                        as u8;
                data[idx] = smoothed;
            }
        }
    }

    /// Unsharp mask for sharpening.
    fn unsharp_mask(
        &self,
        plane: &Plane,
        width: u32,
        height: u32,
        strength: f32,
    ) -> CvResult<Plane> {
        // Apply Gaussian blur
        let blurred = self.gaussian_blur(plane, width, height, 1.0)?;

        // Compute unsharp mask
        let mut result_data = vec![0u8; (width * height) as usize];

        for i in 0..(width * height) as usize {
            if i < plane.data.len() && i < blurred.data.len() {
                let original = plane.data[i] as f32;
                let blurred_val = blurred.data[i] as f32;
                let detail = original - blurred_val;
                let sharpened = original + detail * strength;
                result_data[i] = sharpened.round().clamp(0.0, 255.0) as u8;
            }
        }

        Ok(Plane {
            data: result_data,
            stride: width as usize,
            width,
            height,
        })
    }

    /// Gaussian blur.
    fn gaussian_blur(&self, plane: &Plane, width: u32, height: u32, sigma: f32) -> CvResult<Plane> {
        let mut result_data = vec![0u8; (width * height) as usize];
        let radius = (sigma * 3.0).ceil() as i32;

        // Create Gaussian kernel
        let mut kernel = Vec::new();
        let mut kernel_sum = 0.0f32;

        for dy in -radius..=radius {
            for dx in -radius..=radius {
                let dist_sq = (dx * dx + dy * dy) as f32;
                let weight = (-dist_sq / (2.0 * sigma * sigma)).exp();
                kernel.push((dx, dy, weight));
                kernel_sum += weight;
            }
        }

        // Normalize kernel
        for (_, _, weight) in &mut kernel {
            *weight /= kernel_sum;
        }

        // Apply convolution
        for y in 0..height {
            for x in 0..width {
                let mut sum = 0.0f32;

                for (dx, dy, weight) in &kernel {
                    let nx = x as i32 + dx;
                    let ny = y as i32 + dy;

                    if nx >= 0 && nx < width as i32 && ny >= 0 && ny < height as i32 {
                        let nidx = (ny as u32 * width + nx as u32) as usize;
                        if nidx < plane.data.len() {
                            sum += plane.data[nidx] as f32 * weight;
                        }
                    }
                }

                let idx = (y * width + x) as usize;
                result_data[idx] = sum.round().clamp(0.0, 255.0) as u8;
            }
        }

        Ok(Plane {
            data: result_data,
            stride: width as usize,
            width,
            height,
        })
    }

    /// Blend two planes.
    fn blend_planes(
        &self,
        plane1: &Plane,
        plane2: &Plane,
        alpha: f32,
        width: u32,
        height: u32,
    ) -> CvResult<Plane> {
        let mut result_data = vec![0u8; (width * height) as usize];

        for i in 0..(width * height) as usize {
            if i < plane1.data.len() && i < plane2.data.len() {
                let v1 = plane1.data[i] as f32;
                let v2 = plane2.data[i] as f32;
                result_data[i] = (v1 * (1.0 - alpha) + v2 * alpha).round().clamp(0.0, 255.0) as u8;
            }
        }

        Ok(Plane {
            data: result_data,
            stride: width as usize,
            width,
            height,
        })
    }
}

impl Default for ArtifactReducer {
    fn default() -> Self {
        Self::new()
    }
}

/// Edge-aware interpolation weights.
///
/// Computes interpolation weights that preserve edges better by
/// considering local gradients and structure.
#[allow(dead_code)]
pub struct EdgeAwareWeights {
    /// Gradient threshold for edge detection.
    gradient_threshold: f32,
    /// Structure preservation strength.
    structure_strength: f32,
}

impl EdgeAwareWeights {
    /// Create new edge-aware weights computer.
    #[must_use]
    pub const fn new(gradient_threshold: f32, structure_strength: f32) -> Self {
        Self {
            gradient_threshold,
            structure_strength,
        }
    }

    /// Compute edge-aware weights for a pixel.
    #[must_use]
    pub fn compute_weight(&self, plane: &Plane, width: u32, height: u32, x: u32, y: u32) -> f32 {
        let gradient = self.compute_gradient(plane, width, height, x, y);

        if gradient > self.gradient_threshold {
            // At edges, reduce weight to preserve structure
            1.0 / (1.0 + gradient * self.structure_strength)
        } else {
            // In flat regions, use full weight
            1.0
        }
    }

    /// Compute gradient magnitude at a pixel.
    fn compute_gradient(&self, plane: &Plane, width: u32, height: u32, x: u32, y: u32) -> f32 {
        if x == 0 || x >= width - 1 || y == 0 || y >= height - 1 {
            return 0.0;
        }

        let idx = (y * width + x) as usize;
        let idx_left = (y * width + (x - 1)) as usize;
        let idx_right = (y * width + (x + 1)) as usize;
        let idx_top = ((y - 1) * width + x) as usize;
        let idx_bottom = ((y + 1) * width + x) as usize;

        if idx >= plane.data.len()
            || idx_left >= plane.data.len()
            || idx_right >= plane.data.len()
            || idx_top >= plane.data.len()
            || idx_bottom >= plane.data.len()
        {
            return 0.0;
        }

        let gx = (plane.data[idx_right] as f32 - plane.data[idx_left] as f32) / 2.0;
        let gy = (plane.data[idx_bottom] as f32 - plane.data[idx_top] as f32) / 2.0;

        (gx * gx + gy * gy).sqrt()
    }
}
