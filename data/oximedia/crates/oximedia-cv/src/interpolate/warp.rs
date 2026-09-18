//! Forward and backward warping for frame interpolation.
//!
//! This module provides warping operations to move pixels according to optical flow,
//! supporting both forward warping (from source to target) and backward warping
//! (from target to source).

use crate::error::{CvError, CvResult};
use crate::interpolate::optical_flow::FlowField;
use bytes::Bytes;
use oximedia_codec::{Plane, VideoFrame};

/// Warping mode for interpolation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum WarpMode {
    /// Nearest neighbor interpolation (fastest).
    Nearest,
    /// Bilinear interpolation (good balance).
    #[default]
    Bilinear,
    /// Bicubic interpolation (highest quality).
    Bicubic,
}

/// Frame warper for optical flow-based motion compensation.
///
/// Supports both forward warping (pushing pixels) and backward warping
/// (pulling pixels) with various interpolation modes.
pub struct Warper {
    /// Warping/interpolation mode.
    mode: WarpMode,
}

impl Warper {
    /// Create a new warper with the specified mode.
    #[must_use]
    pub const fn new(mode: WarpMode) -> Self {
        Self { mode }
    }

    /// Set the warping mode.
    pub fn set_mode(&mut self, mode: WarpMode) {
        self.mode = mode;
    }

    /// Warp a frame forward using optical flow.
    ///
    /// Forward warping moves pixels from the source frame to their destination
    /// positions according to the flow field, scaled by time factor `t`.
    ///
    /// # Arguments
    ///
    /// * `frame` - Source frame
    /// * `flow` - Optical flow field
    /// * `t` - Time factor (0.0 to 1.0)
    pub fn warp_forward(
        &self,
        frame: &VideoFrame,
        flow: &FlowField,
        t: f32,
    ) -> CvResult<VideoFrame> {
        self.validate_frame_and_flow(frame, flow)?;

        let mut result = VideoFrame::new(frame.format, frame.width, frame.height);
        result.allocate();
        result.timestamp = frame.timestamp;
        result.frame_type = frame.frame_type;
        result.color_info = frame.color_info;

        // Warp each plane
        for plane_idx in 0..frame.planes.len() {
            let (plane_w, plane_h) = frame.plane_dimensions(plane_idx);
            self.warp_plane_forward(
                &frame.planes[plane_idx],
                &mut result.planes[plane_idx],
                flow,
                plane_w,
                plane_h,
                t,
            )?;
        }

        Ok(result)
    }

    /// Warp a frame backward using optical flow.
    ///
    /// Backward warping pulls pixels from the source frame based on the
    /// reverse flow field, scaled by time factor `t`.
    ///
    /// # Arguments
    ///
    /// * `frame` - Source frame
    /// * `flow` - Optical flow field (should be reversed for backward warping)
    /// * `t` - Time factor (0.0 to 1.0)
    pub fn warp_backward(
        &self,
        frame: &VideoFrame,
        flow: &FlowField,
        t: f32,
    ) -> CvResult<VideoFrame> {
        self.validate_frame_and_flow(frame, flow)?;

        let mut result = VideoFrame::new(frame.format, frame.width, frame.height);
        result.allocate();
        result.timestamp = frame.timestamp;
        result.frame_type = frame.frame_type;
        result.color_info = frame.color_info;

        // Warp each plane
        for plane_idx in 0..frame.planes.len() {
            let (plane_w, plane_h) = frame.plane_dimensions(plane_idx);
            self.warp_plane_backward(
                &frame.planes[plane_idx],
                &mut result.planes[plane_idx],
                flow,
                plane_w,
                plane_h,
                t,
            )?;
        }

        Ok(result)
    }

    /// Warp a single plane using forward warping.
    #[allow(clippy::too_many_arguments)]
    fn warp_plane_forward(
        &self,
        src_plane: &Plane,
        dst_plane: &mut Plane,
        flow: &FlowField,
        width: u32,
        height: u32,
        t: f32,
    ) -> CvResult<()> {
        // Create a mutable buffer for the destination
        let mut dst_data = vec![0u8; (width * height) as usize];
        let mut weight_map = vec![0.0f32; (width * height) as usize];

        // Scale factor for plane (for chroma subsampling)
        let scale_x = width as f32 / flow.width as f32;
        let scale_y = height as f32 / flow.height as f32;

        // Forward warp: push pixels from source to destination
        for y in 0..height {
            for x in 0..width {
                let src_idx = (y * width + x) as usize;
                if src_idx >= src_plane.data.len() {
                    continue;
                }

                let src_value = src_plane.data[src_idx];

                // Get flow vector at this position (scaled for plane size)
                let flow_x = (x as f32 / scale_x) as u32;
                let flow_y = (y as f32 / scale_y) as u32;
                let (dx, dy) = flow.get(flow_x, flow_y);

                // Calculate destination position
                let dest_x = x as f32 + dx * scale_x * t;
                let dest_y = y as f32 + dy * scale_y * t;

                // Splat to nearby pixels with bilinear weights
                self.splat_pixel(
                    &mut dst_data,
                    &mut weight_map,
                    width,
                    height,
                    dest_x,
                    dest_y,
                    src_value,
                );
            }
        }

        // Normalize by weights and handle holes
        for i in 0..dst_data.len() {
            if weight_map[i] > 0.0 {
                dst_data[i] = (dst_data[i] as f32 / weight_map[i]) as u8;
            } else {
                // Fill holes with nearest neighbor from source
                let x = (i % width as usize) as u32;
                let y = (i / width as usize) as u32;
                dst_data[i] = self.get_pixel_nearest(src_plane, width, height, x as f32, y as f32);
            }
        }

        // Update the plane data
        dst_plane.data = dst_data;
        dst_plane.stride = width as usize;

        Ok(())
    }

    /// Warp a single plane using backward warping.
    #[allow(clippy::too_many_arguments)]
    fn warp_plane_backward(
        &self,
        src_plane: &Plane,
        dst_plane: &mut Plane,
        flow: &FlowField,
        width: u32,
        height: u32,
        t: f32,
    ) -> CvResult<()> {
        let mut dst_data = vec![0u8; (width * height) as usize];

        // Scale factor for plane (for chroma subsampling)
        let scale_x = width as f32 / flow.width as f32;
        let scale_y = height as f32 / flow.height as f32;

        // Backward warp: pull pixels from source
        for y in 0..height {
            for x in 0..width {
                // Get flow vector at this position (scaled for plane size)
                let flow_x = (x as f32 / scale_x) as u32;
                let flow_y = (y as f32 / scale_y) as u32;
                let (dx, dy) = flow.get(flow_x, flow_y);

                // Calculate source position (backward mapping)
                let src_x = x as f32 - dx * scale_x * t;
                let src_y = y as f32 - dy * scale_y * t;

                // Sample from source using interpolation
                let value = match self.mode {
                    WarpMode::Nearest => {
                        self.get_pixel_nearest(src_plane, width, height, src_x, src_y)
                    }
                    WarpMode::Bilinear => {
                        self.get_pixel_bilinear(src_plane, width, height, src_x, src_y)
                    }
                    WarpMode::Bicubic => {
                        self.get_pixel_bicubic(src_plane, width, height, src_x, src_y)
                    }
                };

                let dst_idx = (y * width + x) as usize;
                dst_data[dst_idx] = value;
            }
        }

        // Update the plane data
        dst_plane.data = dst_data;
        dst_plane.stride = width as usize;

        Ok(())
    }

    /// Splat a pixel to the destination with bilinear weights.
    #[allow(clippy::too_many_arguments)]
    fn splat_pixel(
        &self,
        dst_data: &mut [u8],
        weight_map: &mut [f32],
        width: u32,
        height: u32,
        x: f32,
        y: f32,
        value: u8,
    ) {
        let x0 = x.floor() as i32;
        let y0 = y.floor() as i32;

        let fx = x - x0 as f32;
        let fy = y - y0 as f32;

        // Bilinear weights
        let weights = [
            (1.0 - fx) * (1.0 - fy),
            fx * (1.0 - fy),
            (1.0 - fx) * fy,
            fx * fy,
        ];

        let positions = [(x0, y0), (x0 + 1, y0), (x0, y0 + 1), (x0 + 1, y0 + 1)];

        for i in 0..4 {
            let (px, py) = positions[i];
            if px >= 0 && px < width as i32 && py >= 0 && py < height as i32 {
                let idx = (py as u32 * width + px as u32) as usize;
                if idx < dst_data.len() {
                    dst_data[idx] += (value as f32 * weights[i]) as u8;
                    weight_map[idx] += weights[i];
                }
            }
        }
    }

    /// Get pixel value using nearest neighbor interpolation.
    fn get_pixel_nearest(&self, plane: &Plane, width: u32, height: u32, x: f32, y: f32) -> u8 {
        let xi = x.round().clamp(0.0, (width - 1) as f32) as u32;
        let yi = y.round().clamp(0.0, (height - 1) as f32) as u32;

        let idx = (yi * width + xi) as usize;
        if idx < plane.data.len() {
            plane.data[idx]
        } else {
            0
        }
    }

    /// Get pixel value using bilinear interpolation.
    fn get_pixel_bilinear(&self, plane: &Plane, width: u32, height: u32, x: f32, y: f32) -> u8 {
        let x = x.clamp(0.0, (width - 1) as f32);
        let y = y.clamp(0.0, (height - 1) as f32);

        let x0 = x.floor() as u32;
        let y0 = y.floor() as u32;
        let x1 = (x0 + 1).min(width - 1);
        let y1 = (y0 + 1).min(height - 1);

        let fx = x - x0 as f32;
        let fy = y - y0 as f32;

        let p00 = self.get_pixel_at(plane, width, x0, y0);
        let p10 = self.get_pixel_at(plane, width, x1, y0);
        let p01 = self.get_pixel_at(plane, width, x0, y1);
        let p11 = self.get_pixel_at(plane, width, x1, y1);

        let v0 = p00 as f32 * (1.0 - fx) + p10 as f32 * fx;
        let v1 = p01 as f32 * (1.0 - fx) + p11 as f32 * fx;
        let result = v0 * (1.0 - fy) + v1 * fy;

        result.round().clamp(0.0, 255.0) as u8
    }

    /// Get pixel value using bicubic interpolation.
    fn get_pixel_bicubic(&self, plane: &Plane, width: u32, height: u32, x: f32, y: f32) -> u8 {
        let x = x.clamp(0.0, (width - 1) as f32);
        let y = y.clamp(0.0, (height - 1) as f32);

        let x0 = x.floor() as i32;
        let y0 = y.floor() as i32;

        let fx = x - x0 as f32;
        let fy = y - y0 as f32;

        let mut result = 0.0f32;

        for j in -1..=2 {
            let mut row_result = 0.0f32;
            for i in -1..=2 {
                let px = (x0 + i).clamp(0, width as i32 - 1) as u32;
                let py = (y0 + j).clamp(0, height as i32 - 1) as u32;
                let pixel = self.get_pixel_at(plane, width, px, py) as f32;
                row_result += pixel * cubic_weight(fx - i as f32);
            }
            result += row_result * cubic_weight(fy - j as f32);
        }

        result.round().clamp(0.0, 255.0) as u8
    }

    /// Get pixel at exact coordinates.
    fn get_pixel_at(&self, plane: &Plane, width: u32, x: u32, y: u32) -> u8 {
        let idx = (y * width + x) as usize;
        if idx < plane.data.len() {
            plane.data[idx]
        } else {
            0
        }
    }

    /// Validate frame and flow field compatibility.
    fn validate_frame_and_flow(&self, frame: &VideoFrame, flow: &FlowField) -> CvResult<()> {
        if frame.width != flow.width || frame.height != flow.height {
            return Err(CvError::invalid_dimensions(frame.width, frame.height));
        }

        if frame.planes.is_empty() {
            return Err(CvError::insufficient_data(1, 0));
        }

        Ok(())
    }
}

impl Default for Warper {
    fn default() -> Self {
        Self::new(WarpMode::Bilinear)
    }
}

/// Cubic interpolation weight function.
///
/// Uses the Mitchell-Netravali filter with B=1/3, C=1/3.
fn cubic_weight(x: f32) -> f32 {
    let x = x.abs();

    if x < 1.0 {
        (1.5 * x - 2.5) * x * x + 1.0
    } else if x < 2.0 {
        ((-0.5 * x + 2.5) * x - 4.0) * x + 2.0
    } else {
        0.0
    }
}
