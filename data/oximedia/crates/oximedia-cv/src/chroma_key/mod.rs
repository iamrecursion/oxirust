//! Chroma key (green screen) processing module.
//!
//! This module provides comprehensive chroma keying functionality for video compositing,
//! including color keying, spill suppression, matte generation, and edge refinement.
//!
//! # Features
//!
//! - **Color keying**: RGB and HSV color space keying with configurable thresholds
//! - **Screen support**: Green screen and blue screen optimized algorithms
//! - **Spill suppression**: Advanced color bleeding removal and despill
//! - **Matte generation**: Alpha matte creation with refinement capabilities
//! - **Edge processing**: Defringing and feathering for smooth compositing
//! - **Auto detection**: Automatic key color detection from sample regions
//!
//! # Example
//!
//! ```
//! use oximedia_cv::chroma_key::{ChromaKey, ChromaKeyConfig, KeyColor};
//!
//! // Create a green screen keyer
//! let config = ChromaKeyConfig::green_screen();
//! let keyer = ChromaKey::new(config);
//!
//! // Process frames (requires actual VideoFrame instances)
//! // let result = keyer.process(&foreground, Some(&background));
//! ```

pub mod auto_key;
pub mod composite;
pub mod keyer;
pub mod matte;
pub mod matting;
pub mod spill;

use crate::error::{CvError, CvResult};
use oximedia_codec::VideoFrame;

// Re-export commonly used items
pub use auto_key::AutoKeyDetector;
pub use composite::{BlendMode, Compositor, LightWrap};
pub use keyer::{ColorKeyer, KeyMethod, KeySpace};
pub use matte::{AlphaMatte, MatteRefiner, RefineOperation};
pub use spill::{DespillAlgorithm, SpillSuppressor};

/// RGB color representation.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Rgb {
    /// Red component (0.0-1.0).
    pub r: f32,
    /// Green component (0.0-1.0).
    pub g: f32,
    /// Blue component (0.0-1.0).
    pub b: f32,
}

impl Rgb {
    /// Create a new RGB color.
    ///
    /// # Examples
    ///
    /// ```
    /// use oximedia_cv::chroma_key::Rgb;
    ///
    /// let green = Rgb::new(0.0, 1.0, 0.0);
    /// assert_eq!(green.g, 1.0);
    /// ```
    #[must_use]
    pub const fn new(r: f32, g: f32, b: f32) -> Self {
        Self { r, g, b }
    }

    /// Create from 8-bit RGB values.
    ///
    /// # Examples
    ///
    /// ```
    /// use oximedia_cv::chroma_key::Rgb;
    ///
    /// let green = Rgb::from_u8(0, 255, 0);
    /// assert!((green.g - 1.0).abs() < 0.01);
    /// ```
    #[must_use]
    pub fn from_u8(r: u8, g: u8, b: u8) -> Self {
        Self {
            r: f32::from(r) / 255.0,
            g: f32::from(g) / 255.0,
            b: f32::from(b) / 255.0,
        }
    }

    /// Convert to 8-bit RGB values.
    #[must_use]
    pub fn to_u8(&self) -> (u8, u8, u8) {
        (
            (self.r * 255.0).clamp(0.0, 255.0) as u8,
            (self.g * 255.0).clamp(0.0, 255.0) as u8,
            (self.b * 255.0).clamp(0.0, 255.0) as u8,
        )
    }

    /// Convert RGB to HSV color space.
    ///
    /// Returns (hue, saturation, value) where hue is in degrees [0-360],
    /// saturation and value are in range [0-1].
    #[must_use]
    #[allow(clippy::float_cmp)]
    pub fn to_hsv(&self) -> Hsv {
        let max = self.r.max(self.g).max(self.b);
        let min = self.r.min(self.g).min(self.b);
        let delta = max - min;

        let v = max;
        let s = if max == 0.0 { 0.0 } else { delta / max };

        let h = if delta == 0.0 {
            0.0
        } else if max == self.r {
            60.0 * (((self.g - self.b) / delta) % 6.0)
        } else if max == self.g {
            60.0 * (((self.b - self.r) / delta) + 2.0)
        } else {
            60.0 * (((self.r - self.g) / delta) + 4.0)
        };

        let h = if h < 0.0 { h + 360.0 } else { h };

        Hsv::new(h, s, v)
    }

    /// Calculate distance to another RGB color.
    #[must_use]
    pub fn distance(&self, other: &Self) -> f32 {
        let dr = self.r - other.r;
        let dg = self.g - other.g;
        let db = self.b - other.b;
        (dr * dr + dg * dg + db * db).sqrt()
    }

    /// Green screen color (chroma green).
    #[must_use]
    pub const fn green_screen() -> Self {
        Self {
            r: 0.0,
            g: 1.0,
            b: 0.0,
        }
    }

    /// Blue screen color (chroma blue).
    #[must_use]
    pub const fn blue_screen() -> Self {
        Self {
            r: 0.0,
            g: 0.0,
            b: 1.0,
        }
    }
}

/// HSV color representation.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Hsv {
    /// Hue in degrees (0.0-360.0).
    pub h: f32,
    /// Saturation (0.0-1.0).
    pub s: f32,
    /// Value/Brightness (0.0-1.0).
    pub v: f32,
}

impl Hsv {
    /// Create a new HSV color.
    #[must_use]
    pub const fn new(h: f32, s: f32, v: f32) -> Self {
        Self { h, s, v }
    }

    /// Convert HSV to RGB color space.
    #[must_use]
    pub fn to_rgb(&self) -> Rgb {
        let c = self.v * self.s;
        let h_prime = self.h / 60.0;
        let x = c * (1.0 - ((h_prime % 2.0) - 1.0).abs());
        let m = self.v - c;

        let (r, g, b) = match h_prime as i32 {
            0 => (c, x, 0.0),
            1 => (x, c, 0.0),
            2 => (0.0, c, x),
            3 => (0.0, x, c),
            4 => (x, 0.0, c),
            5 => (c, 0.0, x),
            _ => (0.0, 0.0, 0.0),
        };

        Rgb::new(r + m, g + m, b + m)
    }

    /// Calculate hue distance to another HSV color (wraps around at 360 degrees).
    #[must_use]
    pub fn hue_distance(&self, other: &Self) -> f32 {
        let diff = (self.h - other.h).abs();
        diff.min(360.0 - diff)
    }
}

/// Predefined key colors for common chroma key scenarios.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum KeyColor {
    /// Standard green screen.
    GreenScreen,
    /// Standard blue screen.
    BlueScreen,
    /// Custom RGB color.
    Custom,
}

impl KeyColor {
    /// Get the RGB color for this key color.
    #[must_use]
    pub fn to_rgb(&self) -> Rgb {
        match self {
            Self::GreenScreen => Rgb::green_screen(),
            Self::BlueScreen => Rgb::blue_screen(),
            Self::Custom => Rgb::new(0.0, 0.0, 0.0),
        }
    }
}

/// Configuration for chroma key processing.
#[derive(Debug, Clone)]
pub struct ChromaKeyConfig {
    /// Key color to remove.
    pub key_color: Rgb,
    /// Primary threshold for keying (0.0-1.0).
    pub threshold: f32,
    /// Tolerance for color variation (0.0-1.0).
    pub tolerance: f32,
    /// Edge softness/feather amount (0.0-1.0).
    pub softness: f32,
    /// Spill suppression strength (0.0-1.0).
    pub spill_suppression: f32,
    /// Despill algorithm to use.
    pub despill_algorithm: DespillAlgorithm,
    /// Color space to perform keying in.
    pub key_space: KeySpace,
    /// Enable edge defringing.
    pub defringe: bool,
    /// Defringe radius in pixels.
    pub defringe_radius: u32,
    /// Enable light wrap effect.
    pub light_wrap: bool,
    /// Light wrap intensity (0.0-1.0).
    pub light_wrap_intensity: f32,
    /// Number of matte erosion iterations.
    pub erosion_iterations: u32,
    /// Number of matte dilation iterations.
    pub dilation_iterations: u32,
    /// Matte blur radius for edge smoothing.
    pub matte_blur_radius: f32,
}

impl ChromaKeyConfig {
    /// Create a new configuration with default values.
    #[must_use]
    pub fn new(key_color: Rgb) -> Self {
        Self {
            key_color,
            threshold: 0.3,
            tolerance: 0.1,
            softness: 0.05,
            spill_suppression: 0.5,
            despill_algorithm: DespillAlgorithm::Simple,
            key_space: KeySpace::Hsv,
            defringe: true,
            defringe_radius: 2,
            light_wrap: false,
            light_wrap_intensity: 0.3,
            erosion_iterations: 0,
            dilation_iterations: 0,
            matte_blur_radius: 1.0,
        }
    }

    /// Create a configuration optimized for green screen.
    #[must_use]
    pub fn green_screen() -> Self {
        Self::new(Rgb::green_screen())
    }

    /// Create a configuration optimized for blue screen.
    #[must_use]
    pub fn blue_screen() -> Self {
        Self::new(Rgb::blue_screen())
    }

    /// Enable advanced quality settings.
    #[must_use]
    pub fn with_quality_settings(mut self) -> Self {
        self.despill_algorithm = DespillAlgorithm::Advanced;
        self.defringe = true;
        self.defringe_radius = 3;
        self.matte_blur_radius = 1.5;
        self
    }

    /// Enable light wrap effect.
    #[must_use]
    pub fn with_light_wrap(mut self, intensity: f32) -> Self {
        self.light_wrap = true;
        self.light_wrap_intensity = intensity.clamp(0.0, 1.0);
        self
    }

    /// Set matte refinement operations.
    #[must_use]
    pub fn with_matte_refinement(mut self, erode: u32, dilate: u32, blur: f32) -> Self {
        self.erosion_iterations = erode;
        self.dilation_iterations = dilate;
        self.matte_blur_radius = blur;
        self
    }
}

impl Default for ChromaKeyConfig {
    fn default() -> Self {
        Self::green_screen()
    }
}

/// Main chroma key processor.
///
/// Provides a high-level API for chroma keying operations, combining
/// color keying, spill suppression, matte generation, and compositing.
pub struct ChromaKey {
    config: ChromaKeyConfig,
    keyer: ColorKeyer,
    spill_suppressor: SpillSuppressor,
    matte_refiner: MatteRefiner,
    compositor: Compositor,
}

impl ChromaKey {
    /// Create a new chroma key processor with the given configuration.
    ///
    /// # Examples
    ///
    /// ```
    /// use oximedia_cv::chroma_key::{ChromaKey, ChromaKeyConfig};
    ///
    /// let config = ChromaKeyConfig::green_screen();
    /// let keyer = ChromaKey::new(config);
    /// ```
    #[must_use]
    pub fn new(config: ChromaKeyConfig) -> Self {
        let keyer = ColorKeyer::new(
            config.key_color,
            config.threshold,
            config.tolerance,
            config.key_space,
        );

        let spill_suppressor = SpillSuppressor::new(
            config.key_color,
            config.spill_suppression,
            config.despill_algorithm,
        );

        let matte_refiner = MatteRefiner::new();

        let compositor = Compositor::new();

        Self {
            config,
            keyer,
            spill_suppressor,
            matte_refiner,
            compositor,
        }
    }

    /// Process a foreground frame with chroma keying.
    ///
    /// If a background frame is provided, the foreground will be composited
    /// over it. Otherwise, returns the keyed foreground with alpha channel.
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - Frame dimensions don't match (when background is provided)
    /// - Frame format is unsupported
    /// - Processing fails
    pub fn process(
        &self,
        foreground: &VideoFrame,
        background: Option<&VideoFrame>,
    ) -> CvResult<VideoFrame> {
        // Validate inputs
        if let Some(bg) = background {
            if foreground.width != bg.width || foreground.height != bg.height {
                return Err(CvError::invalid_parameter(
                    "dimensions",
                    format!(
                        "foreground {}x{} != background {}x{}",
                        foreground.width, foreground.height, bg.width, bg.height
                    ),
                ));
            }
        }

        // Step 1: Generate initial matte
        let mut matte = self.keyer.key_frame(foreground)?;

        // Step 2: Refine matte
        if self.config.erosion_iterations > 0 {
            matte = self
                .matte_refiner
                .erode(&matte, self.config.erosion_iterations)?;
        }
        if self.config.dilation_iterations > 0 {
            matte = self
                .matte_refiner
                .dilate(&matte, self.config.dilation_iterations)?;
        }
        if self.config.matte_blur_radius > 0.0 {
            matte = self
                .matte_refiner
                .blur(&matte, self.config.matte_blur_radius)?;
        }

        // Step 3: Apply edge feathering
        if self.config.softness > 0.0 {
            matte = self.matte_refiner.feather(&matte, self.config.softness)?;
        }

        // Step 4: Remove spill
        let mut despilled = foreground.clone();
        if self.config.spill_suppression > 0.0 {
            self.spill_suppressor.suppress(&mut despilled, &matte)?;
        }

        // Step 5: Apply defringing
        if self.config.defringe {
            self.compositor
                .defringe(&mut despilled, &matte, self.config.defringe_radius)?;
        }

        // Step 6: Composite with background if provided
        if let Some(bg) = background {
            let mut result = self.compositor.composite(&despilled, bg, &matte)?;

            // Step 7: Apply light wrap if enabled
            if self.config.light_wrap {
                let light_wrap = LightWrap::new(self.config.light_wrap_intensity);
                light_wrap.apply(&mut result, bg, &matte)?;
            }

            Ok(result)
        } else {
            // Return foreground with alpha matte
            self.compositor.apply_matte(&despilled, &matte)
        }
    }

    /// Generate only the alpha matte for a frame.
    ///
    /// Useful for previewing the matte or performing custom compositing.
    ///
    /// # Errors
    ///
    /// Returns an error if matte generation fails.
    pub fn generate_matte(&self, frame: &VideoFrame) -> CvResult<AlphaMatte> {
        let mut matte = self.keyer.key_frame(frame)?;

        // Apply same refinements as in process()
        if self.config.erosion_iterations > 0 {
            matte = self
                .matte_refiner
                .erode(&matte, self.config.erosion_iterations)?;
        }
        if self.config.dilation_iterations > 0 {
            matte = self
                .matte_refiner
                .dilate(&matte, self.config.dilation_iterations)?;
        }
        if self.config.matte_blur_radius > 0.0 {
            matte = self
                .matte_refiner
                .blur(&matte, self.config.matte_blur_radius)?;
        }
        if self.config.softness > 0.0 {
            matte = self.matte_refiner.feather(&matte, self.config.softness)?;
        }

        Ok(matte)
    }

    /// Update the key color.
    ///
    /// This allows changing the keying color without recreating the processor.
    pub fn set_key_color(&mut self, color: Rgb) {
        self.config.key_color = color;
        self.keyer.set_key_color(color);
        self.spill_suppressor.set_key_color(color);
    }

    /// Update keying thresholds.
    pub fn set_thresholds(&mut self, threshold: f32, tolerance: f32) {
        self.config.threshold = threshold;
        self.config.tolerance = tolerance;
        self.keyer.set_thresholds(threshold, tolerance);
    }

    /// Update spill suppression strength.
    pub fn set_spill_suppression(&mut self, strength: f32) {
        self.config.spill_suppression = strength.clamp(0.0, 1.0);
        self.spill_suppressor.set_strength(strength);
    }

    /// Get the current configuration.
    #[must_use]
    pub fn config(&self) -> &ChromaKeyConfig {
        &self.config
    }

    /// Auto-detect key color from a sample region in the frame.
    ///
    /// # Arguments
    ///
    /// * `frame` - The video frame to sample from
    /// * `x`, `y`, `width`, `height` - Sample region coordinates
    ///
    /// # Errors
    ///
    /// Returns an error if the sample region is invalid or detection fails.
    pub fn auto_detect_key_color(
        &mut self,
        frame: &VideoFrame,
        x: u32,
        y: u32,
        width: u32,
        height: u32,
    ) -> CvResult<Rgb> {
        let detector = AutoKeyDetector::new();
        let color = detector.detect_from_region(frame, x, y, width, height)?;
        self.set_key_color(color);
        Ok(color)
    }
}
