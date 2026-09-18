//! Compositing utilities for chroma keying.
//!
//! This module provides functions for compositing foreground and background
//! images using alpha mattes, including advanced features like light wrap
//! and edge defringing.

use super::matte::AlphaMatte;
use crate::error::{CvError, CvResult};
use oximedia_codec::VideoFrame;
use oximedia_core::PixelFormat;

/// Blending mode for compositing.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BlendMode {
    /// Standard alpha blending (over operation).
    Normal,
    /// Additive blending.
    Add,
    /// Screen blending.
    Screen,
    /// Multiply blending.
    Multiply,
}

/// Compositor for combining foreground and background frames.
pub struct Compositor {
    blend_mode: BlendMode,
}

impl Compositor {
    /// Create a new compositor with normal blending.
    #[must_use]
    pub fn new() -> Self {
        Self {
            blend_mode: BlendMode::Normal,
        }
    }

    /// Set the blend mode.
    pub fn set_blend_mode(&mut self, mode: BlendMode) {
        self.blend_mode = mode;
    }

    /// Composite foreground over background using alpha matte.
    ///
    /// # Arguments
    ///
    /// * `foreground` - Foreground frame
    /// * `background` - Background frame
    /// * `matte` - Alpha matte defining transparency
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - Dimensions don't match
    /// - Pixel formats are incompatible
    /// - Compositing fails
    pub fn composite(
        &self,
        foreground: &VideoFrame,
        background: &VideoFrame,
        matte: &AlphaMatte,
    ) -> CvResult<VideoFrame> {
        // Validate dimensions
        if foreground.width != background.width || foreground.height != background.height {
            return Err(CvError::invalid_parameter(
                "dimensions",
                format!(
                    "foreground {}x{} != background {}x{}",
                    foreground.width, foreground.height, background.width, background.height
                ),
            ));
        }

        if foreground.width != matte.width() || foreground.height != matte.height() {
            return Err(CvError::invalid_parameter(
                "dimensions",
                "matte dimensions don't match frame dimensions",
            ));
        }

        // Ensure compatible formats
        if foreground.format != background.format {
            return Err(CvError::invalid_parameter(
                "format",
                "foreground and background must have same pixel format",
            ));
        }

        match foreground.format {
            PixelFormat::Rgb24 => self.composite_rgb24(foreground, background, matte),
            PixelFormat::Rgba32 => self.composite_rgba32(foreground, background, matte),
            _ => Err(CvError::unsupported_format(format!(
                "{}",
                foreground.format
            ))),
        }
    }

    /// Apply alpha matte to a frame, converting to RGBA.
    ///
    /// # Errors
    ///
    /// Returns an error if dimensions don't match or format is unsupported.
    pub fn apply_matte(&self, frame: &VideoFrame, matte: &AlphaMatte) -> CvResult<VideoFrame> {
        if frame.width != matte.width() || frame.height != matte.height() {
            return Err(CvError::invalid_parameter(
                "dimensions",
                "matte dimensions don't match frame dimensions",
            ));
        }

        let width = frame.width as usize;
        let height = frame.height as usize;
        let mut result = VideoFrame::new(PixelFormat::Rgba32, frame.width, frame.height);
        result.allocate();

        match frame.format {
            PixelFormat::Rgb24 => {
                if frame.planes.is_empty() {
                    return Err(CvError::invalid_parameter("planes", "empty"));
                }
                let src = &frame.planes[0].data;
                let dst_data = result.planes[0].data.clone();
                let mut new_dst = dst_data;

                for i in 0..width * height {
                    new_dst[i * 4] = src[i * 3];
                    new_dst[i * 4 + 1] = src[i * 3 + 1];
                    new_dst[i * 4 + 2] = src[i * 3 + 2];
                    new_dst[i * 4 + 3] = (matte.data()[i] * 255.0) as u8;
                }

                result.planes[0].data = new_dst;
            }
            PixelFormat::Rgba32 => {
                if frame.planes.is_empty() {
                    return Err(CvError::invalid_parameter("planes", "empty"));
                }
                let src = &frame.planes[0].data;
                let dst_data = result.planes[0].data.clone();
                let mut new_dst = dst_data;

                for i in 0..width * height {
                    new_dst[i * 4] = src[i * 4];
                    new_dst[i * 4 + 1] = src[i * 4 + 1];
                    new_dst[i * 4 + 2] = src[i * 4 + 2];
                    // Multiply existing alpha with matte
                    let existing_alpha = f32::from(src[i * 4 + 3]) / 255.0;
                    new_dst[i * 4 + 3] = (existing_alpha * matte.data()[i] * 255.0) as u8;
                }

                result.planes[0].data = new_dst;
            }
            _ => {
                return Err(CvError::unsupported_format(format!("{}", frame.format)));
            }
        }

        Ok(result)
    }

    /// Apply edge defringing to remove color fringing.
    ///
    /// # Errors
    ///
    /// Returns an error if processing fails.
    pub fn defringe(
        &self,
        frame: &mut VideoFrame,
        matte: &AlphaMatte,
        radius: u32,
    ) -> CvResult<()> {
        if frame.width != matte.width() || frame.height != matte.height() {
            return Err(CvError::invalid_parameter(
                "dimensions",
                "matte dimensions don't match frame dimensions",
            ));
        }

        let width = frame.width as usize;
        let height = frame.height as usize;
        let radius = radius as usize;

        match frame.format {
            PixelFormat::Rgb24 | PixelFormat::Rgba32 => {
                if frame.planes.is_empty() {
                    return Err(CvError::invalid_parameter("planes", "empty"));
                }
                let data = frame.planes[0].data.clone();
                let mut new_data = data.clone();
                let channels = if frame.format == PixelFormat::Rgba32 {
                    4
                } else {
                    3
                };

                for y in 0..height {
                    for x in 0..width {
                        let idx = y * width + x;
                        let alpha = matte.data()[idx];

                        // Apply defringing near edges (0.1 < alpha < 0.9)
                        if alpha > 0.1 && alpha < 0.9 {
                            let corrected = self.defringe_pixel(
                                &data, x, y, width, height, radius, channels, matte,
                            );

                            let pixel_idx = idx * channels;
                            new_data[pixel_idx] = corrected.0;
                            new_data[pixel_idx + 1] = corrected.1;
                            new_data[pixel_idx + 2] = corrected.2;
                        }
                    }
                }

                frame.planes[0].data = new_data;
            }
            _ => {
                return Err(CvError::unsupported_format(format!("{}", frame.format)));
            }
        }

        Ok(())
    }

    /// Composite RGB24 frames.
    fn composite_rgb24(
        &self,
        foreground: &VideoFrame,
        background: &VideoFrame,
        matte: &AlphaMatte,
    ) -> CvResult<VideoFrame> {
        let width = foreground.width as usize;
        let height = foreground.height as usize;

        if foreground.planes.is_empty() || background.planes.is_empty() {
            return Err(CvError::invalid_parameter("planes", "empty"));
        }

        let fg_data = &foreground.planes[0].data;
        let bg_data = &background.planes[0].data;

        let mut result = VideoFrame::new(PixelFormat::Rgb24, foreground.width, foreground.height);
        result.allocate();
        let result_data = result.planes[0].data.clone();
        let mut new_result = result_data;

        for i in 0..width * height {
            let alpha = matte.data()[i];
            let fg_idx = i * 3;
            let bg_idx = i * 3;
            let res_idx = i * 3;

            for c in 0..3 {
                let fg = f32::from(fg_data[fg_idx + c]) / 255.0;
                let bg = f32::from(bg_data[bg_idx + c]) / 255.0;

                let blended = match self.blend_mode {
                    BlendMode::Normal => fg * alpha + bg * (1.0 - alpha),
                    BlendMode::Add => (fg * alpha + bg).min(1.0),
                    BlendMode::Screen => 1.0 - (1.0 - fg * alpha) * (1.0 - bg),
                    BlendMode::Multiply => fg * alpha * bg + bg * (1.0 - alpha),
                };

                new_result[res_idx + c] = (blended * 255.0).clamp(0.0, 255.0) as u8;
            }
        }

        result.planes[0].data = new_result;
        Ok(result)
    }

    /// Composite RGBA32 frames.
    fn composite_rgba32(
        &self,
        foreground: &VideoFrame,
        background: &VideoFrame,
        matte: &AlphaMatte,
    ) -> CvResult<VideoFrame> {
        let width = foreground.width as usize;
        let height = foreground.height as usize;

        if foreground.planes.is_empty() || background.planes.is_empty() {
            return Err(CvError::invalid_parameter("planes", "empty"));
        }

        let fg_data = &foreground.planes[0].data;
        let bg_data = &background.planes[0].data;

        let mut result = VideoFrame::new(PixelFormat::Rgba32, foreground.width, foreground.height);
        result.allocate();
        let result_data = result.planes[0].data.clone();
        let mut new_result = result_data;

        for i in 0..width * height {
            let alpha = matte.data()[i];
            let fg_idx = i * 4;
            let bg_idx = i * 4;
            let res_idx = i * 4;

            // Premultiply alpha
            let fg_alpha = f32::from(fg_data[fg_idx + 3]) / 255.0;
            let final_alpha = fg_alpha * alpha;

            for c in 0..3 {
                let fg = f32::from(fg_data[fg_idx + c]) / 255.0;
                let bg = f32::from(bg_data[bg_idx + c]) / 255.0;

                let blended = match self.blend_mode {
                    BlendMode::Normal => fg * final_alpha + bg * (1.0 - final_alpha),
                    BlendMode::Add => (fg * final_alpha + bg).min(1.0),
                    BlendMode::Screen => 1.0 - (1.0 - fg * final_alpha) * (1.0 - bg),
                    BlendMode::Multiply => fg * final_alpha * bg + bg * (1.0 - final_alpha),
                };

                new_result[res_idx + c] = (blended * 255.0).clamp(0.0, 255.0) as u8;
            }

            new_result[res_idx + 3] = 255; // Result is always opaque
        }

        result.planes[0].data = new_result;
        Ok(result)
    }

    /// Defringe a single pixel by sampling opaque neighbors.
    #[allow(clippy::too_many_arguments)]
    fn defringe_pixel(
        &self,
        data: &[u8],
        x: usize,
        y: usize,
        width: usize,
        height: usize,
        radius: usize,
        channels: usize,
        matte: &AlphaMatte,
    ) -> (u8, u8, u8) {
        let mut sum_r = 0.0f32;
        let mut sum_g = 0.0f32;
        let mut sum_b = 0.0f32;
        let mut count = 0.0f32;

        // Sample opaque neighbors
        for dy in 0..=radius {
            for dx in 0..=radius {
                if dx == 0 && dy == 0 {
                    continue;
                }

                // Check all four quadrants
                for &(sdx, sdy) in &[(1, 1), (-1, 1), (1, -1), (-1, -1)] {
                    let nx = (x as isize + dx as isize * sdx) as usize;
                    let ny = (y as isize + dy as isize * sdy) as usize;

                    if nx < width && ny < height {
                        let nidx = ny * width + nx;
                        let neighbor_alpha = matte.data()[nidx];

                        // Only sample from opaque regions
                        if neighbor_alpha > 0.9 {
                            let pixel_idx = nidx * channels;
                            sum_r += f32::from(data[pixel_idx]);
                            sum_g += f32::from(data[pixel_idx + 1]);
                            sum_b += f32::from(data[pixel_idx + 2]);
                            count += 1.0;
                        }
                    }
                }
            }
        }

        if count > 0.0 {
            (
                (sum_r / count) as u8,
                (sum_g / count) as u8,
                (sum_b / count) as u8,
            )
        } else {
            // No opaque neighbors, keep original
            let idx = (y * width + x) * channels;
            (data[idx], data[idx + 1], data[idx + 2])
        }
    }
}

impl Default for Compositor {
    fn default() -> Self {
        Self::new()
    }
}

/// Light wrap effect for realistic compositing.
///
/// Light wrap simulates the effect of background light wrapping around
/// the foreground subject, creating a more natural integration.
pub struct LightWrap {
    intensity: f32,
    blur_radius: f32,
    threshold: f32,
}

impl LightWrap {
    /// Create a new light wrap effect.
    ///
    /// # Arguments
    ///
    /// * `intensity` - Effect intensity (0.0-1.0)
    #[must_use]
    pub fn new(intensity: f32) -> Self {
        Self {
            intensity: intensity.clamp(0.0, 1.0),
            blur_radius: 10.0,
            threshold: 0.5,
        }
    }

    /// Set blur radius for the light wrap effect.
    pub fn set_blur_radius(&mut self, radius: f32) {
        self.blur_radius = radius.max(0.0);
    }

    /// Set threshold for edge detection.
    pub fn set_threshold(&mut self, threshold: f32) {
        self.threshold = threshold.clamp(0.0, 1.0);
    }

    /// Apply light wrap effect to composited frame.
    ///
    /// # Arguments
    ///
    /// * `frame` - Composited frame to enhance
    /// * `background` - Background frame to use for light wrap
    /// * `matte` - Alpha matte
    ///
    /// # Errors
    ///
    /// Returns an error if processing fails.
    pub fn apply(
        &self,
        frame: &mut VideoFrame,
        background: &VideoFrame,
        matte: &AlphaMatte,
    ) -> CvResult<()> {
        if frame.width != background.width || frame.height != background.height {
            return Err(CvError::invalid_parameter(
                "dimensions",
                "frame and background dimensions don't match",
            ));
        }

        let width = frame.width as usize;
        let height = frame.height as usize;

        match (frame.format, background.format) {
            (PixelFormat::Rgb24, PixelFormat::Rgb24)
            | (PixelFormat::Rgba32, PixelFormat::Rgba32) => {
                if frame.planes.is_empty() || background.planes.is_empty() {
                    return Err(CvError::invalid_parameter("planes", "empty"));
                }

                let frame_data = frame.planes[0].data.clone();
                let bg_data = &background.planes[0].data;
                let mut new_frame_data = frame_data.clone();
                let channels = if frame.format == PixelFormat::Rgba32 {
                    4
                } else {
                    3
                };

                for y in 0..height {
                    for x in 0..width {
                        let idx = y * width + x;
                        let alpha = matte.data()[idx];

                        // Apply light wrap near edges
                        if alpha > self.threshold && alpha < 1.0 {
                            let edge_factor = 1.0 - alpha;
                            let wrap_strength = self.intensity * edge_factor;

                            let pixel_idx = idx * channels;
                            for c in 0..3 {
                                let fg = f32::from(new_frame_data[pixel_idx + c]) / 255.0;
                                let bg = f32::from(bg_data[pixel_idx + c]) / 255.0;

                                // Blend background light into foreground
                                let wrapped = fg + bg * wrap_strength;
                                new_frame_data[pixel_idx + c] =
                                    (wrapped * 255.0).clamp(0.0, 255.0) as u8;
                            }
                        }
                    }
                }

                frame.planes[0].data = new_frame_data;
            }
            _ => {
                return Err(CvError::invalid_parameter(
                    "format",
                    "frame and background must have same format",
                ));
            }
        }

        Ok(())
    }
}

/// Color correction for matching foreground to background.
pub struct ColorMatcher {
    adaptation_strength: f32,
}

impl ColorMatcher {
    /// Create a new color matcher.
    #[must_use]
    pub fn new() -> Self {
        Self {
            adaptation_strength: 0.3,
        }
    }

    /// Set adaptation strength (0.0-1.0).
    pub fn set_adaptation_strength(&mut self, strength: f32) {
        self.adaptation_strength = strength.clamp(0.0, 1.0);
    }

    /// Match foreground colors to background lighting.
    ///
    /// # Errors
    ///
    /// Returns an error if processing fails.
    pub fn match_colors(
        &self,
        foreground: &mut VideoFrame,
        background: &VideoFrame,
        matte: &AlphaMatte,
    ) -> CvResult<()> {
        // Calculate average background color
        let bg_avg = self.calculate_average_color(background)?;

        // Calculate average foreground color (opaque regions only)
        let fg_avg = self.calculate_average_color_with_matte(foreground, matte)?;

        // Apply color correction
        self.apply_color_correction(foreground, &fg_avg, &bg_avg)
    }

    /// Calculate average color of a frame.
    fn calculate_average_color(&self, frame: &VideoFrame) -> CvResult<(f32, f32, f32)> {
        let width = frame.width as usize;
        let height = frame.height as usize;
        let pixel_count = width * height;

        match frame.format {
            PixelFormat::Rgb24 => {
                if frame.planes.is_empty() {
                    return Err(CvError::invalid_parameter("planes", "empty"));
                }
                let data = &frame.planes[0].data;
                let mut sum_r = 0.0f32;
                let mut sum_g = 0.0f32;
                let mut sum_b = 0.0f32;

                for i in 0..pixel_count {
                    sum_r += f32::from(data[i * 3]);
                    sum_g += f32::from(data[i * 3 + 1]);
                    sum_b += f32::from(data[i * 3 + 2]);
                }

                Ok((
                    sum_r / pixel_count as f32,
                    sum_g / pixel_count as f32,
                    sum_b / pixel_count as f32,
                ))
            }
            _ => Err(CvError::unsupported_format(format!("{}", frame.format))),
        }
    }

    /// Calculate average color with matte weighting.
    fn calculate_average_color_with_matte(
        &self,
        frame: &VideoFrame,
        matte: &AlphaMatte,
    ) -> CvResult<(f32, f32, f32)> {
        let width = frame.width as usize;
        let height = frame.height as usize;

        match frame.format {
            PixelFormat::Rgb24 => {
                if frame.planes.is_empty() {
                    return Err(CvError::invalid_parameter("planes", "empty"));
                }
                let data = &frame.planes[0].data;
                let mut sum_r = 0.0f32;
                let mut sum_g = 0.0f32;
                let mut sum_b = 0.0f32;
                let mut weight_sum = 0.0f32;

                for i in 0..width * height {
                    let alpha = matte.data()[i];
                    if alpha > 0.5 {
                        // Only sample opaque regions
                        sum_r += f32::from(data[i * 3]) * alpha;
                        sum_g += f32::from(data[i * 3 + 1]) * alpha;
                        sum_b += f32::from(data[i * 3 + 2]) * alpha;
                        weight_sum += alpha;
                    }
                }

                if weight_sum > 0.0 {
                    Ok((sum_r / weight_sum, sum_g / weight_sum, sum_b / weight_sum))
                } else {
                    Ok((128.0, 128.0, 128.0)) // Default gray
                }
            }
            _ => Err(CvError::unsupported_format(format!("{}", frame.format))),
        }
    }

    /// Apply color correction to match target average.
    fn apply_color_correction(
        &self,
        frame: &mut VideoFrame,
        source_avg: &(f32, f32, f32),
        target_avg: &(f32, f32, f32),
    ) -> CvResult<()> {
        let width = frame.width as usize;
        let height = frame.height as usize;

        // Calculate correction factors
        let factor_r = if source_avg.0 > 0.0 {
            target_avg.0 / source_avg.0
        } else {
            1.0
        };
        let factor_g = if source_avg.1 > 0.0 {
            target_avg.1 / source_avg.1
        } else {
            1.0
        };
        let factor_b = if source_avg.2 > 0.0 {
            target_avg.2 / source_avg.2
        } else {
            1.0
        };

        // Blend factors towards 1.0 based on adaptation strength
        let blend_r = 1.0 + (factor_r - 1.0) * self.adaptation_strength;
        let blend_g = 1.0 + (factor_g - 1.0) * self.adaptation_strength;
        let blend_b = 1.0 + (factor_b - 1.0) * self.adaptation_strength;

        match frame.format {
            PixelFormat::Rgb24 => {
                if frame.planes.is_empty() {
                    return Err(CvError::invalid_parameter("planes", "empty"));
                }
                let data = frame.planes[0].data.clone();
                let mut new_data = data.clone();

                for i in 0..width * height {
                    let idx = i * 3;
                    new_data[idx] = (f32::from(data[idx]) * blend_r).clamp(0.0, 255.0) as u8;
                    new_data[idx + 1] =
                        (f32::from(data[idx + 1]) * blend_g).clamp(0.0, 255.0) as u8;
                    new_data[idx + 2] =
                        (f32::from(data[idx + 2]) * blend_b).clamp(0.0, 255.0) as u8;
                }

                frame.planes[0].data = new_data;
            }
            _ => {
                return Err(CvError::unsupported_format(format!("{}", frame.format)));
            }
        }

        Ok(())
    }
}

impl Default for ColorMatcher {
    fn default() -> Self {
        Self::new()
    }
}
