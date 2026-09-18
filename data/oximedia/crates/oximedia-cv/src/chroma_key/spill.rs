//! Spill suppression algorithms.
//!
//! Spill occurs when the key color (green or blue screen) reflects onto
//! the foreground subject, causing unwanted color contamination. This module
//! provides algorithms to detect and remove this color bleeding.

use super::{Hsv, Rgb};
use crate::chroma_key::matte::AlphaMatte;
use crate::error::{CvError, CvResult};
use oximedia_codec::VideoFrame;
use oximedia_core::PixelFormat;

/// Despill algorithm type.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DespillAlgorithm {
    /// Simple spill removal based on color averaging.
    Simple,
    /// Advanced spill removal with edge preservation.
    Advanced,
    /// Limit-based despill (limits key color channel).
    Limit,
    /// Green screen specific algorithm.
    GreenScreen,
    /// Blue screen specific algorithm.
    BlueScreen,
}

/// Spill suppression processor.
///
/// Removes color bleeding from the key color onto the foreground subject.
pub struct SpillSuppressor {
    key_color: Rgb,
    key_color_hsv: Hsv,
    strength: f32,
    algorithm: DespillAlgorithm,
}

impl SpillSuppressor {
    /// Create a new spill suppressor.
    ///
    /// # Arguments
    ///
    /// * `key_color` - The key color that's causing spill
    /// * `strength` - Spill suppression strength (0.0-1.0)
    /// * `algorithm` - Despill algorithm to use
    #[must_use]
    pub fn new(key_color: Rgb, strength: f32, algorithm: DespillAlgorithm) -> Self {
        let key_color_hsv = key_color.to_hsv();
        Self {
            key_color,
            key_color_hsv,
            strength: strength.clamp(0.0, 1.0),
            algorithm,
        }
    }

    /// Set the key color.
    pub fn set_key_color(&mut self, color: Rgb) {
        self.key_color = color;
        self.key_color_hsv = color.to_hsv();
    }

    /// Set suppression strength.
    pub fn set_strength(&mut self, strength: f32) {
        self.strength = strength.clamp(0.0, 1.0);
    }

    /// Set the despill algorithm.
    pub fn set_algorithm(&mut self, algorithm: DespillAlgorithm) {
        self.algorithm = algorithm;
    }

    /// Apply spill suppression to a frame.
    ///
    /// # Arguments
    ///
    /// * `frame` - The frame to process (modified in place)
    /// * `matte` - Alpha matte to guide suppression intensity
    ///
    /// # Errors
    ///
    /// Returns an error if frame format is unsupported or dimensions mismatch.
    pub fn suppress(&self, frame: &mut VideoFrame, matte: &AlphaMatte) -> CvResult<()> {
        if frame.width != matte.width() || frame.height != matte.height() {
            return Err(CvError::invalid_parameter(
                "dimensions",
                format!(
                    "frame {}x{} != matte {}x{}",
                    frame.width,
                    frame.height,
                    matte.width(),
                    matte.height()
                ),
            ));
        }

        match self.algorithm {
            DespillAlgorithm::Simple => self.suppress_simple(frame, matte),
            DespillAlgorithm::Advanced => self.suppress_advanced(frame, matte),
            DespillAlgorithm::Limit => self.suppress_limit(frame, matte),
            DespillAlgorithm::GreenScreen => self.suppress_green_screen(frame, matte),
            DespillAlgorithm::BlueScreen => self.suppress_blue_screen(frame, matte),
        }
    }

    /// Simple spill suppression using color averaging.
    fn suppress_simple(&self, frame: &mut VideoFrame, matte: &AlphaMatte) -> CvResult<()> {
        let width = frame.width as usize;
        let height = frame.height as usize;

        match frame.format {
            PixelFormat::Rgb24 => {
                if frame.planes.is_empty() {
                    return Err(CvError::invalid_parameter("planes", "empty"));
                }
                let data = frame.planes[0].data.clone();
                let mut new_data = data.clone();

                for y in 0..height {
                    for x in 0..width {
                        let idx = y * width + x;
                        let pixel_idx = idx * 3;

                        let r = f32::from(data[pixel_idx]) / 255.0;
                        let g = f32::from(data[pixel_idx + 1]) / 255.0;
                        let b = f32::from(data[pixel_idx + 2]) / 255.0;

                        let alpha = matte.data()[idx];
                        let spill_amount = self.detect_spill_rgb(r, g, b);

                        if spill_amount > 0.0 {
                            let suppression = self.strength * (1.0 - alpha) * spill_amount;
                            let corrected = self.remove_spill_simple(r, g, b, suppression);

                            new_data[pixel_idx] = (corrected.0 * 255.0) as u8;
                            new_data[pixel_idx + 1] = (corrected.1 * 255.0) as u8;
                            new_data[pixel_idx + 2] = (corrected.2 * 255.0) as u8;
                        }
                    }
                }

                frame.planes[0].data = new_data;
            }
            PixelFormat::Rgba32 => {
                if frame.planes.is_empty() {
                    return Err(CvError::invalid_parameter("planes", "empty"));
                }
                let data = frame.planes[0].data.clone();
                let mut new_data = data.clone();

                for y in 0..height {
                    for x in 0..width {
                        let idx = y * width + x;
                        let pixel_idx = idx * 4;

                        let r = f32::from(data[pixel_idx]) / 255.0;
                        let g = f32::from(data[pixel_idx + 1]) / 255.0;
                        let b = f32::from(data[pixel_idx + 2]) / 255.0;

                        let alpha = matte.data()[idx];
                        let spill_amount = self.detect_spill_rgb(r, g, b);

                        if spill_amount > 0.0 {
                            let suppression = self.strength * (1.0 - alpha) * spill_amount;
                            let corrected = self.remove_spill_simple(r, g, b, suppression);

                            new_data[pixel_idx] = (corrected.0 * 255.0) as u8;
                            new_data[pixel_idx + 1] = (corrected.1 * 255.0) as u8;
                            new_data[pixel_idx + 2] = (corrected.2 * 255.0) as u8;
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

    /// Advanced spill suppression with edge preservation.
    fn suppress_advanced(&self, frame: &mut VideoFrame, matte: &AlphaMatte) -> CvResult<()> {
        let width = frame.width as usize;
        let height = frame.height as usize;

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
                        let pixel_idx = idx * channels;

                        let r = f32::from(data[pixel_idx]) / 255.0;
                        let g = f32::from(data[pixel_idx + 1]) / 255.0;
                        let b = f32::from(data[pixel_idx + 2]) / 255.0;

                        let pixel = Rgb::new(r, g, b);
                        let pixel_hsv = pixel.to_hsv();

                        let alpha = matte.data()[idx];
                        let spill_amount = self.detect_spill_hsv(&pixel_hsv);

                        if spill_amount > 0.0 {
                            let suppression = self.strength * (1.0 - alpha) * spill_amount;
                            let corrected = self.remove_spill_advanced(&pixel_hsv, suppression);

                            new_data[pixel_idx] = (corrected.0 * 255.0) as u8;
                            new_data[pixel_idx + 1] = (corrected.1 * 255.0) as u8;
                            new_data[pixel_idx + 2] = (corrected.2 * 255.0) as u8;
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

    /// Limit-based despill (limits the key color channel).
    fn suppress_limit(&self, frame: &mut VideoFrame, matte: &AlphaMatte) -> CvResult<()> {
        let width = frame.width as usize;
        let height = frame.height as usize;

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
                        let pixel_idx = idx * channels;

                        let r = f32::from(data[pixel_idx]) / 255.0;
                        let g = f32::from(data[pixel_idx + 1]) / 255.0;
                        let b = f32::from(data[pixel_idx + 2]) / 255.0;

                        let alpha = matte.data()[idx];
                        let corrected = self.remove_spill_limit(r, g, b, alpha);

                        new_data[pixel_idx] = (corrected.0 * 255.0) as u8;
                        new_data[pixel_idx + 1] = (corrected.1 * 255.0) as u8;
                        new_data[pixel_idx + 2] = (corrected.2 * 255.0) as u8;
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

    /// Green screen optimized despill.
    fn suppress_green_screen(&self, frame: &mut VideoFrame, matte: &AlphaMatte) -> CvResult<()> {
        let width = frame.width as usize;
        let height = frame.height as usize;

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
                        let pixel_idx = idx * channels;

                        let r = f32::from(data[pixel_idx]) / 255.0;
                        let g = f32::from(data[pixel_idx + 1]) / 255.0;
                        let b = f32::from(data[pixel_idx + 2]) / 255.0;

                        let alpha = matte.data()[idx];

                        // Green spill detection: green is higher than both red and blue
                        let spill = (g - r.max(b)).max(0.0);

                        if spill > 0.0 {
                            let suppression = self.strength * (1.0 - alpha);
                            // Reduce green channel
                            let new_g = g - spill * suppression;
                            // Balance by increasing magenta (red + blue)
                            let boost = spill * suppression * 0.5;
                            let new_r = (r + boost).min(1.0);
                            let new_b = (b + boost).min(1.0);

                            new_data[pixel_idx] = (new_r * 255.0) as u8;
                            new_data[pixel_idx + 1] = (new_g * 255.0) as u8;
                            new_data[pixel_idx + 2] = (new_b * 255.0) as u8;
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

    /// Blue screen optimized despill.
    fn suppress_blue_screen(&self, frame: &mut VideoFrame, matte: &AlphaMatte) -> CvResult<()> {
        let width = frame.width as usize;
        let height = frame.height as usize;

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
                        let pixel_idx = idx * channels;

                        let r = f32::from(data[pixel_idx]) / 255.0;
                        let g = f32::from(data[pixel_idx + 1]) / 255.0;
                        let b = f32::from(data[pixel_idx + 2]) / 255.0;

                        let alpha = matte.data()[idx];

                        // Blue spill detection: blue is higher than both red and green
                        let spill = (b - r.max(g)).max(0.0);

                        if spill > 0.0 {
                            let suppression = self.strength * (1.0 - alpha);
                            // Reduce blue channel
                            let new_b = b - spill * suppression;
                            // Balance by increasing yellow (red + green)
                            let boost = spill * suppression * 0.5;
                            let new_r = (r + boost).min(1.0);
                            let new_g = (g + boost).min(1.0);

                            new_data[pixel_idx] = (new_r * 255.0) as u8;
                            new_data[pixel_idx + 1] = (new_g * 255.0) as u8;
                            new_data[pixel_idx + 2] = (new_b * 255.0) as u8;
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

    /// Detect spill amount in RGB color.
    fn detect_spill_rgb(&self, r: f32, g: f32, b: f32) -> f32 {
        let pixel = Rgb::new(r, g, b);
        let distance = pixel.distance(&self.key_color);

        // Spill is inversely related to distance
        let spill = (0.5 - distance).max(0.0) * 2.0;
        spill.clamp(0.0, 1.0)
    }

    /// Detect spill amount in HSV color.
    fn detect_spill_hsv(&self, pixel_hsv: &Hsv) -> f32 {
        let hue_dist = pixel_hsv.hue_distance(&self.key_color_hsv);
        let sat_sim = 1.0 - (pixel_hsv.s - self.key_color_hsv.s).abs();

        // Spill when hue is similar and saturation is moderate to high
        let hue_factor = (60.0 - hue_dist).max(0.0) / 60.0;
        let sat_factor = pixel_hsv.s * sat_sim;

        (hue_factor * sat_factor).clamp(0.0, 1.0)
    }

    /// Remove spill using simple color averaging.
    fn remove_spill_simple(&self, r: f32, g: f32, b: f32, suppression: f32) -> (f32, f32, f32) {
        // Determine which channel is the key color's dominant channel
        let key_channel =
            if self.key_color.g > self.key_color.r && self.key_color.g > self.key_color.b {
                1 // Green
            } else if self.key_color.b > self.key_color.r && self.key_color.b > self.key_color.g {
                2 // Blue
            } else {
                0 // Red
            };

        match key_channel {
            1 => {
                // Green spill: average red and blue, reduce green
                let avg = (r + b) * 0.5;
                let new_g = g * (1.0 - suppression) + avg * suppression;
                (r, new_g, b)
            }
            2 => {
                // Blue spill: average red and green, reduce blue
                let avg = (r + g) * 0.5;
                let new_b = b * (1.0 - suppression) + avg * suppression;
                (r, g, new_b)
            }
            _ => (r, g, b), // Red or unknown
        }
    }

    /// Remove spill using advanced HSV-based method.
    fn remove_spill_advanced(&self, pixel_hsv: &Hsv, suppression: f32) -> (f32, f32, f32) {
        // Rotate hue away from key color
        let hue_diff = pixel_hsv.hue_distance(&self.key_color_hsv);
        let hue_rotation = if hue_diff < 90.0 {
            suppression * 30.0 // Rotate up to 30 degrees
        } else {
            0.0
        };

        let new_hue = (pixel_hsv.h + hue_rotation) % 360.0;

        // Reduce saturation slightly to remove color cast
        let new_sat = pixel_hsv.s * (1.0 - suppression * 0.3);

        let corrected_hsv = Hsv::new(new_hue, new_sat, pixel_hsv.v);
        let rgb = corrected_hsv.to_rgb();
        (rgb.r, rgb.g, rgb.b)
    }

    /// Remove spill by limiting key color channel.
    fn remove_spill_limit(&self, r: f32, g: f32, b: f32, alpha: f32) -> (f32, f32, f32) {
        let limit_strength = self.strength * (1.0 - alpha);

        if self.key_color.g > 0.8 {
            // Green screen: limit green to max of red and blue
            let limit = r.max(b);
            let new_g = g * (1.0 - limit_strength) + limit * limit_strength;
            (r, new_g.min(g), b)
        } else if self.key_color.b > 0.8 {
            // Blue screen: limit blue to max of red and green
            let limit = r.max(g);
            let new_b = b * (1.0 - limit_strength) + limit * limit_strength;
            (r, g, new_b.min(b))
        } else {
            (r, g, b)
        }
    }
}
