//! Core chroma keying algorithms.
//!
//! This module implements the fundamental color keying operations that
//! determine which pixels should be transparent based on their color
//! similarity to the key color.

use super::{Hsv, Rgb};
use crate::chroma_key::matte::AlphaMatte;
use crate::error::{CvError, CvResult};
use oximedia_codec::VideoFrame;
use oximedia_core::PixelFormat;

/// Color space to perform keying in.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum KeySpace {
    /// RGB color space - Euclidean distance in RGB space.
    Rgb,
    /// HSV color space - More intuitive for color selection.
    Hsv,
    /// YUV color space - Separates luminance from chrominance.
    Yuv,
}

/// Keying method algorithm.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum KeyMethod {
    /// Simple distance-based keying.
    Simple,
    /// Advanced keying with edge preservation.
    Advanced,
    /// Luma key (brightness-based).
    Luma,
}

/// Core color keying engine.
///
/// Generates alpha mattes by analyzing color similarity between
/// pixels and the key color.
pub struct ColorKeyer {
    key_color: Rgb,
    key_color_hsv: Hsv,
    threshold: f32,
    tolerance: f32,
    key_space: KeySpace,
    method: KeyMethod,
}

impl ColorKeyer {
    /// Create a new color keyer.
    ///
    /// # Arguments
    ///
    /// * `key_color` - The color to key out (remove)
    /// * `threshold` - Primary threshold for keying (0.0-1.0)
    /// * `tolerance` - Additional tolerance for edge softness (0.0-1.0)
    /// * `key_space` - Color space to perform keying in
    #[must_use]
    pub fn new(key_color: Rgb, threshold: f32, tolerance: f32, key_space: KeySpace) -> Self {
        let key_color_hsv = key_color.to_hsv();
        Self {
            key_color,
            key_color_hsv,
            threshold: threshold.clamp(0.0, 1.0),
            tolerance: tolerance.clamp(0.0, 1.0),
            key_space,
            method: KeyMethod::Advanced,
        }
    }

    /// Set the key color.
    pub fn set_key_color(&mut self, color: Rgb) {
        self.key_color = color;
        self.key_color_hsv = color.to_hsv();
    }

    /// Set keying thresholds.
    pub fn set_thresholds(&mut self, threshold: f32, tolerance: f32) {
        self.threshold = threshold.clamp(0.0, 1.0);
        self.tolerance = tolerance.clamp(0.0, 1.0);
    }

    /// Set the keying method.
    pub fn set_method(&mut self, method: KeyMethod) {
        self.method = method;
    }

    /// Generate an alpha matte for the given frame.
    ///
    /// # Errors
    ///
    /// Returns an error if the frame format is unsupported.
    pub fn key_frame(&self, frame: &VideoFrame) -> CvResult<AlphaMatte> {
        match self.method {
            KeyMethod::Simple => self.key_frame_simple(frame),
            KeyMethod::Advanced => self.key_frame_advanced(frame),
            KeyMethod::Luma => self.key_frame_luma(frame),
        }
    }

    /// Simple distance-based keying.
    fn key_frame_simple(&self, frame: &VideoFrame) -> CvResult<AlphaMatte> {
        let width = frame.width as usize;
        let height = frame.height as usize;
        let mut alpha_data = vec![0.0f32; width * height];

        // Convert frame to RGB for processing
        let rgb_data = self.frame_to_rgb(frame)?;

        for y in 0..height {
            for x in 0..width {
                let idx = y * width + x;
                let pixel = Rgb::new(
                    rgb_data[idx * 3],
                    rgb_data[idx * 3 + 1],
                    rgb_data[idx * 3 + 2],
                );

                let alpha = self.compute_alpha_simple(&pixel);
                alpha_data[idx] = alpha;
            }
        }

        Ok(AlphaMatte::new(width as u32, height as u32, alpha_data))
    }

    /// Advanced keying with edge preservation.
    fn key_frame_advanced(&self, frame: &VideoFrame) -> CvResult<AlphaMatte> {
        let width = frame.width as usize;
        let height = frame.height as usize;
        let mut alpha_data = vec![0.0f32; width * height];

        let rgb_data = self.frame_to_rgb(frame)?;

        for y in 0..height {
            for x in 0..width {
                let idx = y * width + x;
                let pixel = Rgb::new(
                    rgb_data[idx * 3],
                    rgb_data[idx * 3 + 1],
                    rgb_data[idx * 3 + 2],
                );

                let alpha = self.compute_alpha_advanced(&pixel);
                alpha_data[idx] = alpha;
            }
        }

        Ok(AlphaMatte::new(width as u32, height as u32, alpha_data))
    }

    /// Luma-based keying (brightness).
    fn key_frame_luma(&self, frame: &VideoFrame) -> CvResult<AlphaMatte> {
        let width = frame.width as usize;
        let height = frame.height as usize;
        let mut alpha_data = vec![0.0f32; width * height];

        let rgb_data = self.frame_to_rgb(frame)?;

        for y in 0..height {
            for x in 0..width {
                let idx = y * width + x;
                let pixel = Rgb::new(
                    rgb_data[idx * 3],
                    rgb_data[idx * 3 + 1],
                    rgb_data[idx * 3 + 2],
                );

                // Compute luminance (BT.709 coefficients)
                let luma = pixel.r * 0.2126 + pixel.g * 0.7152 + pixel.b * 0.0722;
                let key_luma = self.key_color.r * 0.2126
                    + self.key_color.g * 0.7152
                    + self.key_color.b * 0.0722;

                let diff = (luma - key_luma).abs();
                let alpha = if diff < self.threshold {
                    0.0
                } else if diff < self.threshold + self.tolerance {
                    (diff - self.threshold) / self.tolerance
                } else {
                    1.0
                };

                alpha_data[idx] = alpha;
            }
        }

        Ok(AlphaMatte::new(width as u32, height as u32, alpha_data))
    }

    /// Compute alpha value using simple distance metric.
    fn compute_alpha_simple(&self, pixel: &Rgb) -> f32 {
        match self.key_space {
            KeySpace::Rgb => {
                let distance = pixel.distance(&self.key_color);
                self.distance_to_alpha(distance)
            }
            KeySpace::Hsv => {
                let pixel_hsv = pixel.to_hsv();
                let hue_dist = pixel_hsv.hue_distance(&self.key_color_hsv) / 180.0;
                let sat_dist = (pixel_hsv.s - self.key_color_hsv.s).abs();
                let val_dist = (pixel_hsv.v - self.key_color_hsv.v).abs();

                // Weight hue more heavily for chroma keying
                let distance =
                    (hue_dist * hue_dist * 2.0 + sat_dist * sat_dist + val_dist * val_dist).sqrt();
                self.distance_to_alpha(distance)
            }
            KeySpace::Yuv => {
                // Convert to YUV for keying
                let (y1, u1, v1) = self.rgb_to_yuv(pixel);
                let (y2, u2, v2) = self.rgb_to_yuv(&self.key_color);

                // Focus on chrominance (U, V)
                let chroma_dist = ((u1 - u2) * (u1 - u2) + (v1 - v2) * (v1 - v2)).sqrt();
                let luma_dist = (y1 - y2).abs() * 0.3; // Reduce luma influence

                let distance = (chroma_dist + luma_dist) / 1.3;
                self.distance_to_alpha(distance)
            }
        }
    }

    /// Compute alpha value using advanced algorithm with better edge handling.
    fn compute_alpha_advanced(&self, pixel: &Rgb) -> f32 {
        match self.key_space {
            KeySpace::Rgb => {
                let distance = pixel.distance(&self.key_color);
                self.distance_to_alpha_smooth(distance)
            }
            KeySpace::Hsv => {
                let pixel_hsv = pixel.to_hsv();

                // Hue distance with wrapping
                let hue_dist = pixel_hsv.hue_distance(&self.key_color_hsv) / 180.0;

                // Saturation distance with non-linear weighting
                let sat_dist = (pixel_hsv.s - self.key_color_hsv.s).abs();

                // Value distance with reduced influence
                let val_dist = (pixel_hsv.v - self.key_color_hsv.v).abs() * 0.5;

                // Advanced weighting: hue is most important, then saturation
                let distance = if pixel_hsv.s < 0.1 {
                    // Desaturated colors (grays) - don't key based on hue
                    (val_dist * val_dist).sqrt()
                } else {
                    // Saturated colors - use full HSV distance
                    let hue_weight = 3.0;
                    let sat_weight = 1.5;
                    let val_weight = 0.5;

                    (hue_dist * hue_dist * hue_weight
                        + sat_dist * sat_dist * sat_weight
                        + val_dist * val_dist * val_weight)
                        .sqrt()
                        / 2.0
                };

                self.distance_to_alpha_smooth(distance)
            }
            KeySpace::Yuv => {
                let (y1, u1, v1) = self.rgb_to_yuv(pixel);
                let (y2, u2, v2) = self.rgb_to_yuv(&self.key_color);

                // Advanced YUV keying
                let chroma_dist = ((u1 - u2) * (u1 - u2) + (v1 - v2) * (v1 - v2)).sqrt();
                let luma_dist = (y1 - y2).abs();

                // Adaptive luma weighting based on chrominance similarity
                let luma_weight = if chroma_dist < self.threshold {
                    0.2 // Reduce luma influence for similar chroma
                } else {
                    0.5 // Increase for different chroma
                };

                let distance = (chroma_dist + luma_dist * luma_weight) / 1.5;
                self.distance_to_alpha_smooth(distance)
            }
        }
    }

    /// Convert distance to alpha with linear transition.
    fn distance_to_alpha(&self, distance: f32) -> f32 {
        if distance < self.threshold {
            0.0 // Fully transparent
        } else if distance < self.threshold + self.tolerance {
            // Linear ramp
            (distance - self.threshold) / self.tolerance
        } else {
            1.0 // Fully opaque
        }
    }

    /// Convert distance to alpha with smooth transition (using smoothstep).
    fn distance_to_alpha_smooth(&self, distance: f32) -> f32 {
        if distance < self.threshold {
            0.0
        } else if distance < self.threshold + self.tolerance {
            // Smoothstep interpolation for better edge quality
            let t = (distance - self.threshold) / self.tolerance;
            t * t * (3.0 - 2.0 * t)
        } else {
            1.0
        }
    }

    /// Convert RGB to YUV (BT.709 coefficients).
    fn rgb_to_yuv(&self, rgb: &Rgb) -> (f32, f32, f32) {
        let y = 0.2126 * rgb.r + 0.7152 * rgb.g + 0.0722 * rgb.b;
        let u = (rgb.b - y) / 1.8556;
        let v = (rgb.r - y) / 1.5748;
        (y, u, v)
    }

    /// Convert video frame to RGB data.
    fn frame_to_rgb(&self, frame: &VideoFrame) -> CvResult<Vec<f32>> {
        let width = frame.width as usize;
        let height = frame.height as usize;
        let mut rgb_data = vec![0.0f32; width * height * 3];

        match frame.format {
            PixelFormat::Rgb24 => {
                // Already RGB, just convert to f32
                if frame.planes.is_empty() {
                    return Err(CvError::invalid_parameter("planes", "empty"));
                }
                let data = &frame.planes[0].data;
                for i in 0..width * height * 3 {
                    rgb_data[i] = f32::from(data[i]) / 255.0;
                }
            }
            PixelFormat::Rgba32 => {
                // RGBA to RGB, skip alpha
                if frame.planes.is_empty() {
                    return Err(CvError::invalid_parameter("planes", "empty"));
                }
                let data = &frame.planes[0].data;
                for i in 0..width * height {
                    rgb_data[i * 3] = f32::from(data[i * 4]) / 255.0;
                    rgb_data[i * 3 + 1] = f32::from(data[i * 4 + 1]) / 255.0;
                    rgb_data[i * 3 + 2] = f32::from(data[i * 4 + 2]) / 255.0;
                }
            }
            PixelFormat::Yuv420p | PixelFormat::Yuv422p | PixelFormat::Yuv444p => {
                // YUV to RGB conversion
                if frame.planes.len() < 3 {
                    return Err(CvError::invalid_parameter(
                        "planes",
                        format!("expected 3, got {}", frame.planes.len()),
                    ));
                }
                self.yuv_to_rgb_frame(frame, &mut rgb_data)?;
            }
            _ => {
                return Err(CvError::unsupported_format(format!("{}", frame.format)));
            }
        }

        Ok(rgb_data)
    }

    /// Convert YUV frame to RGB data.
    fn yuv_to_rgb_frame(&self, frame: &VideoFrame, rgb_data: &mut [f32]) -> CvResult<()> {
        let width = frame.width as usize;
        let height = frame.height as usize;

        let y_plane = &frame.planes[0].data;
        let u_plane = &frame.planes[1].data;
        let v_plane = &frame.planes[2].data;

        let (h_ratio, v_ratio) = frame.format.chroma_subsampling();

        for y in 0..height {
            for x in 0..width {
                let y_idx = y * frame.planes[0].stride + x;
                let chroma_x = x / h_ratio as usize;
                let chroma_y = y / v_ratio as usize;
                let chroma_idx = chroma_y * frame.planes[1].stride + chroma_x;

                let y_val = f32::from(y_plane[y_idx]) / 255.0;
                let u_val = (f32::from(u_plane[chroma_idx]) - 128.0) / 255.0;
                let v_val = (f32::from(v_plane[chroma_idx]) - 128.0) / 255.0;

                // BT.709 YUV to RGB conversion
                let r = y_val + 1.5748 * v_val;
                let g = y_val - 0.1873 * u_val - 0.4681 * v_val;
                let b = y_val + 1.8556 * u_val;

                let idx = (y * width + x) * 3;
                rgb_data[idx] = r.clamp(0.0, 1.0);
                rgb_data[idx + 1] = g.clamp(0.0, 1.0);
                rgb_data[idx + 2] = b.clamp(0.0, 1.0);
            }
        }

        Ok(())
    }
}

/// Multi-pass keying for improved quality.
///
/// Performs multiple keying passes with different parameters
/// and combines the results for better edge quality.
pub struct MultiPassKeyer {
    passes: Vec<ColorKeyer>,
    blend_weights: Vec<f32>,
}

impl MultiPassKeyer {
    /// Create a new multi-pass keyer with default configuration.
    #[must_use]
    pub fn new(key_color: Rgb, base_threshold: f32) -> Self {
        let mut passes = Vec::new();
        let mut blend_weights = Vec::new();

        // Pass 1: Aggressive keying for core transparency
        passes.push(ColorKeyer::new(
            key_color,
            base_threshold * 0.8,
            0.05,
            KeySpace::Hsv,
        ));
        blend_weights.push(0.5);

        // Pass 2: Conservative keying for edges
        passes.push(ColorKeyer::new(
            key_color,
            base_threshold * 1.2,
            0.15,
            KeySpace::Hsv,
        ));
        blend_weights.push(0.3);

        // Pass 3: YUV-based keying for color accuracy
        let mut yuv_keyer = ColorKeyer::new(key_color, base_threshold, 0.1, KeySpace::Yuv);
        yuv_keyer.set_method(KeyMethod::Advanced);
        passes.push(yuv_keyer);
        blend_weights.push(0.2);

        Self {
            passes,
            blend_weights,
        }
    }

    /// Process frame with multiple passes.
    ///
    /// # Errors
    ///
    /// Returns an error if any pass fails.
    pub fn key_frame(&self, frame: &VideoFrame) -> CvResult<AlphaMatte> {
        let width = frame.width;
        let height = frame.height;
        let size = (width * height) as usize;

        let mut combined_alpha = vec![0.0f32; size];

        for (keyer, &weight) in self.passes.iter().zip(&self.blend_weights) {
            let matte = keyer.key_frame(frame)?;
            for i in 0..size {
                combined_alpha[i] += matte.data()[i] * weight;
            }
        }

        // Normalize to ensure values stay in [0, 1]
        let max_val = combined_alpha.iter().fold(0.0f32, |acc, &val| acc.max(val));
        if max_val > 1.0 {
            for val in &mut combined_alpha {
                *val /= max_val;
            }
        }

        Ok(AlphaMatte::new(width, height, combined_alpha))
    }
}
