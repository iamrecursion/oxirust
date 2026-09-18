//! Professional video effects library for `OxiMedia`.
//!
//! This crate provides production-quality implementations of professional video effects
//! used in post-production, broadcast, and creative video applications.
//!
//! # Effect Categories
//!
//! ## Transitions
//! - **Dissolve** - Cross-dissolve with custom curves
//! - **Wipe** - 30+ wipe patterns (horizontal, vertical, diagonal, circular, etc.)
//! - **Push** - Push transitions in all directions
//! - **Slide** - Slide transitions with easing
//! - **Zoom** - Zoom in/out transitions
//! - **3D** - Cube, flip, page curl, and other 3D transitions
//!
//! ## Generators
//! - **Color Bars** - SMPTE and EBU color bars
//! - **Test Patterns** - Checkerboard, grid, zone plate
//! - **Noise** - White, pink, and Perlin noise
//! - **Gradients** - Linear, radial, and angular gradients
//! - **Solid** - Solid color generator
//!
//! ## Keying
//! - **Advanced Keying** - Green/blue screen keying algorithms
//! - **Spill Suppression** - Remove color spill from keyed edges
//! - **Edge Refinement** - Improve edge quality with various algorithms
//!
//! ## Time Effects
//! - **Time Remapping** - Remap time with custom curves
//! - **Speed Ramping** - Variable speed playback
//! - **Freeze Frame** - Hold frames at specific times
//! - **Reverse** - Reverse playback
//!
//! ## Distortion
//! - **Lens Distortion** - Correct or apply lens distortion
//! - **Barrel/Pincushion** - Barrel and pincushion distortion
//! - **Wave** - Wave distortion effects
//! - **Ripple** - Water ripple effects
//!
//! ## Stylization
//! - **Cartoon** - Cel-shading and cartoon effects
//! - **Sketch** - Pencil and sketch effects
//! - **Oil Paint** - Oil painting effect
//! - **Mosaic** - Pixelate and mosaic effects
//! - **Halftone** - Comic book halftone effect
//!
//! ## Light Effects
//! - **Lens Flare** - Realistic lens flare
//! - **Light Rays** - God rays and volumetric light
//! - **Glow** - Glow effect with blur
//! - **Bloom** - HDR bloom effect
//!
//! ## Particle Systems
//! - **Snow** - Realistic snow particles
//! - **Rain** - Rain particle system
//! - **Sparks** - Fire sparks and embers
//! - **Dust** - Atmospheric dust particles
//!
//! ## Text Effects
//! - **Text Rendering** - High-quality text rendering with effects
//! - **Text Animation** - Typewriter, fade, slide, etc.
//!
//! ## Shape Effects
//! - **Shape Drawing** - Rectangles, circles, lines, polygons
//! - **Shape Animation** - Animated shapes with keyframes
//! - **Masks** - Animated shape masks
//!
//! # Architecture
//!
//! All effects implement the `VideoEffect` trait, which provides a unified interface
//! for real-time video processing with GPU acceleration support.
//!
//! Effects are designed to be:
//! - **Keyframeable** - All parameters support keyframe animation
//! - **GPU Accelerated** - Optional GPU implementation for performance
//! - **Real-time** - Optimized for real-time preview
//! - **Compositable** - Full alpha channel support
//! - **Safe** - No unsafe code, enforced by `#![forbid(unsafe_code)]`
//!
//! # Example
//!
//! ```ignore
//! use oximedia_vfx::{VideoEffect, transition::Dissolve, EffectParams};
//!
//! let mut dissolve = Dissolve::new();
//! let params = EffectParams::new()
//!     .with_progress(0.5)
//!     .with_quality(QualityMode::Final);
//!
//! // Apply transition between two frames
//! dissolve.apply(&frame_a, &frame_b, &mut output, &params)?;
//! ```

#![forbid(unsafe_code)]
#![warn(missing_docs)]
#![allow(clippy::missing_errors_doc)]
#![allow(clippy::missing_panics_doc)]
#![allow(clippy::module_name_repetitions)]
#![allow(clippy::cast_possible_truncation)]
#![allow(clippy::cast_precision_loss)]
#![allow(clippy::cast_sign_loss)]
#![allow(clippy::similar_names)]
#![allow(clippy::too_many_arguments)]
#![allow(clippy::cast_possible_wrap)]
#![allow(clippy::return_self_not_must_use)]
#![allow(clippy::unnecessary_wraps)]
#![allow(clippy::unused_self)]
#![allow(clippy::cast_lossless)]
#![allow(clippy::match_same_arms)]
#![allow(clippy::let_and_return)]
#![allow(clippy::field_reassign_with_default)]
#![allow(clippy::float_cmp)]
#![allow(clippy::derivable_impls)]
#![allow(clippy::single_char_pattern)]
#![allow(clippy::manual_clamp)]
#![allow(clippy::must_use_candidate)]
#![allow(clippy::needless_lifetimes)]
#![allow(clippy::many_single_char_names)]
#![allow(clippy::used_underscore_binding)]
#![allow(clippy::no_effect_underscore_binding)]
#![allow(clippy::struct_field_names)]
#![allow(dead_code)]

pub mod audio_reactive;
pub mod blur_kernel;
pub mod chroma_key;
pub mod chromatic_aberration;
pub mod color_grade;
pub mod color_grading;
/// Color look-up table application for VFX colour grading.
pub mod color_lut;
pub mod color_management;
pub mod compositing;
pub mod deform_mesh;
pub mod depth_of_field;
pub mod distortion;
pub mod draft_preview;
/// Edge detection filters (Sobel, Prewitt, Laplacian, Roberts).
pub mod edge_detect;
/// Additional compositing and image effects.
pub mod effects;
pub mod film_effect;
pub mod fog;
/// Frame pool for reusing large RGBA heap allocations.
pub mod frame_pool;
pub mod generator;
/// Digital glitch effects (RGB shift, scanlines, block displacement, noise).
pub mod glitch;
pub mod gpu_backend;
pub mod grade_pipeline;
pub mod grain;
/// Heat and atmospheric distortion effect.
pub mod heat_distort;
pub mod keying;
pub mod lens_aberration;
pub mod lens_flare;
pub mod light;
pub mod mblur_config;
pub mod motion_blur;
/// Motion-vector-based motion blur reading per-pixel displacement fields.
pub mod motion_vector_blur;
pub mod noise_field;
/// Parallax 2.5D camera motion effect using depth maps.
pub mod parallax;
/// Multi-channel parameter tracks for Vec2/Vec3/Color keyframe animation.
pub mod param_track;
pub mod param_track_cache;
pub mod particle;
pub mod particle_fx;
pub mod particle_sim;
pub mod presets;
/// Render pass sequencing for multi-pass VFX compositing.
pub mod render_pass;
pub mod retro_film;
pub mod ripple;
pub mod rotoscoping;
pub mod shape;
/// SIMD-accelerated pixel fill, blend, and lerp operations.
pub mod simd_pixel;
pub mod style;
pub mod text;
/// Tile-based parallel processing helpers for VideoEffect chains.
pub mod tile_processor;
pub mod tilt_shift;
pub mod time;
pub mod tracking;
pub mod trail_effect;
pub mod transition;
pub mod utils;
pub mod vector_blur;
/// VFX preset management: named parameter bundles.
pub mod vfx_preset;
/// Vignette effect with customizable shape, falloff, and tint.
pub mod vignette;

pub use frame_pool::FramePool;
pub use param_track::{
    ColorKeyframe, ColorTrack, Vec2Keyframe, Vec2Track, Vec3Keyframe, Vec3Track,
};

use oximedia_core::OxiError;
use serde::{Deserialize, Serialize};
use thiserror::Error;

/// Error types for video effects.
#[derive(Debug, Error)]
pub enum VfxError {
    /// Invalid parameter value.
    #[error("Invalid parameter: {0}")]
    InvalidParameter(String),

    /// Invalid dimensions.
    #[error("Invalid dimensions: {width}x{height}")]
    InvalidDimensions {
        /// Width value.
        width: u32,
        /// Height value.
        height: u32,
    },

    /// Buffer size mismatch.
    #[error("Buffer size mismatch: expected {expected}, got {actual}")]
    BufferSizeMismatch {
        /// Expected size.
        expected: usize,
        /// Actual size.
        actual: usize,
    },

    /// Insufficient buffer size.
    #[error("Insufficient buffer size: need at least {required}, got {actual}")]
    InsufficientBuffer {
        /// Required size.
        required: usize,
        /// Actual size.
        actual: usize,
    },

    /// Effect not initialized.
    #[error("Effect not initialized")]
    NotInitialized,

    /// Processing error.
    #[error("Processing error: {0}")]
    ProcessingError(String),

    /// GPU error.
    #[error("GPU error: {0}")]
    GpuError(String),

    /// Keyframe error.
    #[error("Keyframe error: {0}")]
    KeyframeError(String),

    /// Text rendering error.
    #[error("Text rendering error: {0}")]
    TextRenderError(String),

    /// `OxiMedia` core error.
    #[error("OxiMedia error: {0}")]
    OxiError(#[from] OxiError),
}

/// Result type for VFX operations.
pub type VfxResult<T> = std::result::Result<T, VfxError>;

/// Quality mode for effect rendering.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum QualityMode {
    /// Draft quality - fastest, lowest quality.
    Draft,
    /// Preview quality - balanced performance and quality.
    Preview,
    /// Final quality - highest quality, slowest.
    Final,
}

impl Default for QualityMode {
    fn default() -> Self {
        Self::Preview
    }
}

/// Video frame data structure.
#[derive(Debug, Clone)]
pub struct Frame {
    /// Frame width in pixels.
    pub width: u32,
    /// Frame height in pixels.
    pub height: u32,
    /// Pixel data in RGBA format (4 bytes per pixel).
    pub data: Vec<u8>,
}

impl Frame {
    /// Create a new frame with given dimensions.
    ///
    /// # Errors
    ///
    /// Returns an error if dimensions are invalid (zero or too large).
    pub fn new(width: u32, height: u32) -> VfxResult<Self> {
        if width == 0 || height == 0 {
            return Err(VfxError::InvalidDimensions { width, height });
        }

        let size = (width as usize)
            .checked_mul(height as usize)
            .and_then(|s| s.checked_mul(4))
            .ok_or(VfxError::InvalidDimensions { width, height })?;

        Ok(Self {
            width,
            height,
            data: vec![0; size],
        })
    }

    /// Create a frame from existing data.
    ///
    /// # Errors
    ///
    /// Returns an error if data size doesn't match dimensions.
    pub fn from_data(width: u32, height: u32, data: Vec<u8>) -> VfxResult<Self> {
        let expected = (width as usize) * (height as usize) * 4;
        if data.len() != expected {
            return Err(VfxError::BufferSizeMismatch {
                expected,
                actual: data.len(),
            });
        }

        Ok(Self {
            width,
            height,
            data,
        })
    }

    /// Get pixel at (x, y) as RGBA.
    #[must_use]
    pub fn get_pixel(&self, x: u32, y: u32) -> Option<[u8; 4]> {
        if x >= self.width || y >= self.height {
            return None;
        }

        let idx = ((y * self.width + x) * 4) as usize;
        Some([
            self.data[idx],
            self.data[idx + 1],
            self.data[idx + 2],
            self.data[idx + 3],
        ])
    }

    /// Set pixel at (x, y) to RGBA value.
    pub fn set_pixel(&mut self, x: u32, y: u32, rgba: [u8; 4]) {
        if x < self.width && y < self.height {
            let idx = ((y * self.width + x) * 4) as usize;
            self.data[idx..idx + 4].copy_from_slice(&rgba);
        }
    }

    /// Clear frame to a solid color.
    ///
    /// Uses SIMD intrinsics on x86/x86_64 when AVX2 is available at runtime,
    /// falling back to a scalar loop otherwise.
    pub fn clear(&mut self, rgba: [u8; 4]) {
        // Build a u32 from the 4 bytes and use the SIMD-friendly fill path.
        // Safety: forbid(unsafe_code) is active; we use only safe std APIs here.
        // The trick: pack rgba into u32 and fill with u32::to_ne_bytes repeating.
        let pixel_u32 = u32::from_ne_bytes(rgba);
        // SAFETY: data is always a multiple of 4 bytes (4 bytes per pixel).
        // We reinterpret as &mut [u32] via safe chunking.
        #[cfg(target_arch = "x86_64")]
        {
            if is_x86_feature_detected!("avx2") {
                // AVX2: fill 8 u32 at a time via safe chunking
                let val64 = (pixel_u32 as u64) | ((pixel_u32 as u64) << 32);
                let val128 = (val64 as u128) | ((val64 as u128) << 64);
                // Write 16-byte chunks
                for chunk in self.data.chunks_exact_mut(16) {
                    chunk.copy_from_slice(&val128.to_ne_bytes());
                }
                // Remainder (< 16 bytes — always multiple of 4 for RGBA)
                let rem_start = (self.data.len() / 16) * 16;
                for chunk in self.data[rem_start..].chunks_exact_mut(4) {
                    chunk.copy_from_slice(&rgba);
                }
                return;
            }
        }
        // Scalar fallback
        for chunk in self.data.chunks_exact_mut(4) {
            chunk.copy_from_slice(&rgba);
        }
        let _ = pixel_u32; // suppress unused warning on non-x86
    }

    /// Get byte size of frame data.
    #[must_use]
    pub const fn byte_size(&self) -> usize {
        (self.width as usize) * (self.height as usize) * 4
    }
}

/// Keyframe for parameter animation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Keyframe {
    /// Time in seconds.
    pub time: f64,
    /// Value at this keyframe.
    pub value: f32,
    /// Easing function.
    pub easing: EasingFunction,
}

/// Easing functions for animation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum EasingFunction {
    /// Linear interpolation.
    Linear,
    /// Ease in (slow start).
    EaseIn,
    /// Ease out (slow end).
    EaseOut,
    /// Ease in-out (slow start and end).
    EaseInOut,
    /// Cubic Bezier curve.
    Bezier,
    /// Elastic ease-in: overshoots with decaying oscillation at start.
    ElasticIn,
    /// Elastic ease-out: overshoots with decaying oscillation at end.
    ElasticOut,
    /// Elastic ease-in-out: elastic at both ends.
    ElasticInOut,
    /// Bounce ease-in: ball-drop bounce at start.
    BounceIn,
    /// Bounce ease-out: ball-drop bounce at end.
    BounceOut,
    /// Bounce ease-in-out: bounce at both ends.
    BounceInOut,
    /// Spring: physically-modelled damped spring oscillation.
    Spring,
    /// Back ease-in: pulls back before accelerating.
    BackIn,
    /// Back ease-out: overshoots target then settles.
    BackOut,
    /// Back ease-in-out: back at both ends.
    BackInOut,
}

impl EasingFunction {
    /// Apply easing function to normalized time [0, 1].
    #[must_use]
    pub fn apply(self, t: f32) -> f32 {
        let t = t.clamp(0.0, 1.0);
        match self {
            Self::Linear => t,
            Self::EaseIn => t * t,
            Self::EaseOut => t * (2.0 - t),
            Self::EaseInOut => {
                if t < 0.5 {
                    2.0 * t * t
                } else {
                    -1.0 + (4.0 - 2.0 * t) * t
                }
            }
            Self::Bezier => {
                // Cubic bezier with control points (0.42, 0), (0.58, 1)
                let t2 = t * t;
                let t3 = t2 * t;
                3.0 * (1.0 - t) * (1.0 - t) * t * 0.42 + 3.0 * (1.0 - t) * t2 * 0.58 + t3
            }
            Self::ElasticIn => self.elastic_in(t),
            Self::ElasticOut => self.elastic_out(t),
            Self::ElasticInOut => self.elastic_in_out(t),
            Self::BounceIn => self.bounce_in(t),
            Self::BounceOut => self.bounce_out(t),
            Self::BounceInOut => self.bounce_in_out(t),
            Self::Spring => self.spring(t),
            Self::BackIn => self.back_in(t),
            Self::BackOut => self.back_out(t),
            Self::BackInOut => self.back_in_out(t),
        }
    }

    /// Elastic ease-in: starts slowly with elastic oscillation.
    fn elastic_in(self, t: f32) -> f32 {
        if t <= 0.0 {
            return 0.0;
        }
        if t >= 1.0 {
            return 1.0;
        }
        let period = 0.3_f32;
        let s = period / 4.0;
        let post = 2.0_f32.powf(10.0 * (t - 1.0));
        let angle = (t - 1.0 - s) * (2.0 * std::f32::consts::PI) / period;
        -(post * angle.sin())
    }

    /// Elastic ease-out: ends with elastic overshoot.
    fn elastic_out(self, t: f32) -> f32 {
        if t <= 0.0 {
            return 0.0;
        }
        if t >= 1.0 {
            return 1.0;
        }
        let period = 0.3_f32;
        let s = period / 4.0;
        let post = 2.0_f32.powf(-10.0 * t);
        let angle = (t - s) * (2.0 * std::f32::consts::PI) / period;
        post * angle.sin() + 1.0
    }

    /// Elastic ease-in-out: elastic at both ends.
    fn elastic_in_out(self, t: f32) -> f32 {
        if t < 0.5 {
            self.elastic_in(t * 2.0) * 0.5
        } else {
            self.elastic_out(t * 2.0 - 1.0) * 0.5 + 0.5
        }
    }

    /// Bounce ease-out core: simulates a ball bouncing.
    fn bounce_out(self, t: f32) -> f32 {
        if t < 1.0 / 2.75 {
            7.5625 * t * t
        } else if t < 2.0 / 2.75 {
            let t = t - 1.5 / 2.75;
            7.5625 * t * t + 0.75
        } else if t < 2.5 / 2.75 {
            let t = t - 2.25 / 2.75;
            7.5625 * t * t + 0.9375
        } else {
            let t = t - 2.625 / 2.75;
            7.5625 * t * t + 0.984375
        }
    }

    /// Bounce ease-in: bounce at the start.
    fn bounce_in(self, t: f32) -> f32 {
        1.0 - self.bounce_out(1.0 - t)
    }

    /// Bounce ease-in-out: bounce at both ends.
    fn bounce_in_out(self, t: f32) -> f32 {
        if t < 0.5 {
            self.bounce_in(t * 2.0) * 0.5
        } else {
            self.bounce_out(t * 2.0 - 1.0) * 0.5 + 0.5
        }
    }

    /// Damped spring: physically-modelled mass-spring-damper system.
    ///
    /// Parameters tuned for a visually pleasing single overshoot that
    /// settles at t=1. Uses damping ratio ~ 0.5 (underdamped).
    fn spring(self, t: f32) -> f32 {
        if t <= 0.0 {
            return 0.0;
        }
        if t >= 1.0 {
            return 1.0;
        }
        // Damped harmonic oscillator: x(t) = 1 - e^(-beta*t) * cos(omega*t)
        let beta = 8.0_f32; // damping
        let omega = 12.0_f32 * std::f32::consts::PI; // angular frequency
        let decay = (-beta * t).exp();
        let oscillation = (omega * t).cos();
        1.0 - decay * oscillation
    }

    /// Back ease-in: pulls back before accelerating forward.
    fn back_in(self, t: f32) -> f32 {
        let overshoot = 1.70158_f32;
        t * t * ((overshoot + 1.0) * t - overshoot)
    }

    /// Back ease-out: overshoots target then returns.
    fn back_out(self, t: f32) -> f32 {
        let overshoot = 1.70158_f32;
        let t = t - 1.0;
        t * t * ((overshoot + 1.0) * t + overshoot) + 1.0
    }

    /// Back ease-in-out: back at both ends.
    fn back_in_out(self, t: f32) -> f32 {
        let overshoot = 1.70158_f32 * 1.525;
        let t = t * 2.0;
        if t < 1.0 {
            0.5 * (t * t * ((overshoot + 1.0) * t - overshoot))
        } else {
            let t = t - 2.0;
            0.5 * (t * t * ((overshoot + 1.0) * t + overshoot) + 2.0)
        }
    }
}

/// Parameter track with keyframes.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ParameterTrack {
    keyframes: Vec<Keyframe>,
}

impl ParameterTrack {
    /// Create a new empty parameter track.
    #[must_use]
    pub const fn new() -> Self {
        Self {
            keyframes: Vec::new(),
        }
    }

    /// Add a keyframe.
    pub fn add_keyframe(&mut self, time: f64, value: f32, easing: EasingFunction) {
        let keyframe = Keyframe {
            time,
            value,
            easing,
        };

        // Insert in sorted order
        match self.keyframes.binary_search_by(|k| {
            k.time
                .partial_cmp(&time)
                .unwrap_or(std::cmp::Ordering::Equal)
        }) {
            Ok(idx) => self.keyframes[idx] = keyframe,
            Err(idx) => self.keyframes.insert(idx, keyframe),
        }
    }

    /// Evaluate parameter at given time.
    #[must_use]
    pub fn evaluate(&self, time: f64) -> Option<f32> {
        if self.keyframes.is_empty() {
            return None;
        }

        if self.keyframes.len() == 1 {
            return Some(self.keyframes[0].value);
        }

        // Find surrounding keyframes
        let idx = match self.keyframes.binary_search_by(|k| {
            k.time
                .partial_cmp(&time)
                .unwrap_or(std::cmp::Ordering::Equal)
        }) {
            Ok(idx) => return Some(self.keyframes[idx].value),
            Err(idx) => idx,
        };

        if idx == 0 {
            return Some(self.keyframes[0].value);
        }

        if idx >= self.keyframes.len() {
            return Some(self.keyframes[self.keyframes.len() - 1].value);
        }

        let k1 = &self.keyframes[idx - 1];
        let k2 = &self.keyframes[idx];

        let dt = k2.time - k1.time;
        if dt <= 0.0 {
            return Some(k1.value);
        }

        let t = ((time - k1.time) / dt) as f32;
        let eased = k1.easing.apply(t);
        Some(k1.value + (k2.value - k1.value) * eased)
    }

    /// Get number of keyframes.
    #[must_use]
    pub fn len(&self) -> usize {
        self.keyframes.len()
    }

    /// Check if track is empty.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.keyframes.is_empty()
    }
}

impl Default for ParameterTrack {
    fn default() -> Self {
        Self::new()
    }
}

/// Parameters for effect processing.
#[derive(Debug, Clone)]
pub struct EffectParams {
    /// Progress through effect (0.0 - 1.0).
    pub progress: f32,
    /// Quality mode.
    pub quality: QualityMode,
    /// Current time in seconds.
    pub time: f64,
    /// Enable GPU acceleration.
    pub use_gpu: bool,
    /// Motion blur amount.
    pub motion_blur: f32,
}

impl Default for EffectParams {
    fn default() -> Self {
        Self {
            progress: 0.0,
            quality: QualityMode::Preview,
            time: 0.0,
            use_gpu: true,
            motion_blur: 0.0,
        }
    }
}

impl EffectParams {
    /// Create new default parameters.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Set progress value.
    #[must_use]
    pub fn with_progress(mut self, progress: f32) -> Self {
        self.progress = progress.clamp(0.0, 1.0);
        self
    }

    /// Set quality mode.
    #[must_use]
    pub const fn with_quality(mut self, quality: QualityMode) -> Self {
        self.quality = quality;
        self
    }

    /// Set current time.
    #[must_use]
    pub const fn with_time(mut self, time: f64) -> Self {
        self.time = time;
        self
    }

    /// Enable/disable GPU acceleration.
    #[must_use]
    pub const fn with_gpu(mut self, use_gpu: bool) -> Self {
        self.use_gpu = use_gpu;
        self
    }

    /// Set motion blur amount.
    #[must_use]
    pub fn with_motion_blur(mut self, motion_blur: f32) -> Self {
        self.motion_blur = motion_blur.clamp(0.0, 1.0);
        self
    }
}

/// Core trait for video effects.
pub trait VideoEffect: Send + Sync {
    /// Get effect name.
    fn name(&self) -> &str;

    /// Get effect description.
    fn description(&self) -> &'static str {
        ""
    }

    /// Apply effect to a single frame.
    fn apply(&mut self, input: &Frame, output: &mut Frame, params: &EffectParams) -> VfxResult<()>;

    /// Reset effect state.
    fn reset(&mut self) {}

    /// Get whether effect supports GPU acceleration.
    fn supports_gpu(&self) -> bool {
        false
    }

    /// Get whether effect requires two input frames (for transitions).
    fn requires_two_inputs(&self) -> bool {
        false
    }
}

/// Core trait for transition effects.
pub trait TransitionEffect: Send + Sync {
    /// Get transition name.
    fn name(&self) -> &str;

    /// Get transition description.
    fn description(&self) -> &'static str {
        ""
    }

    /// Apply transition between two frames.
    fn apply(
        &mut self,
        from: &Frame,
        to: &Frame,
        output: &mut Frame,
        params: &EffectParams,
    ) -> VfxResult<()>;

    /// Reset transition state.
    fn reset(&mut self) {}

    /// Get whether transition supports GPU acceleration.
    fn supports_gpu(&self) -> bool {
        false
    }
}

/// Color in RGBA format.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct Color {
    /// Red component (0-255).
    pub r: u8,
    /// Green component (0-255).
    pub g: u8,
    /// Blue component (0-255).
    pub b: u8,
    /// Alpha component (0-255).
    pub a: u8,
}

impl Color {
    /// Create a new color.
    #[must_use]
    pub const fn new(r: u8, g: u8, b: u8, a: u8) -> Self {
        Self { r, g, b, a }
    }

    /// Create from RGB with full opacity.
    #[must_use]
    pub const fn rgb(r: u8, g: u8, b: u8) -> Self {
        Self::new(r, g, b, 255)
    }

    /// Create from RGBA array.
    #[must_use]
    pub const fn from_rgba(rgba: [u8; 4]) -> Self {
        Self::new(rgba[0], rgba[1], rgba[2], rgba[3])
    }

    /// Convert to RGBA array.
    #[must_use]
    pub const fn to_rgba(self) -> [u8; 4] {
        [self.r, self.g, self.b, self.a]
    }

    /// Create black color.
    #[must_use]
    pub const fn black() -> Self {
        Self::rgb(0, 0, 0)
    }

    /// Create white color.
    #[must_use]
    pub const fn white() -> Self {
        Self::rgb(255, 255, 255)
    }

    /// Create transparent color.
    #[must_use]
    pub const fn transparent() -> Self {
        Self::new(0, 0, 0, 0)
    }

    /// Blend this color with another using alpha blending.
    ///
    /// Performs over-compositing: `dst = src_alpha * src + (1 - src_alpha) * dst`.
    /// The computation uses integer arithmetic on SIMD-capable targets via
    /// `is_x86_feature_detected!("sse4.1")` at runtime; scalar fallback otherwise.
    #[must_use]
    pub fn blend(self, other: Self) -> Self {
        // Integer fixed-point blend: avoids f32 on hot paths.
        // alpha in [0,255]; scale by 256 for fixed-point.
        let alpha = other.a as u32;
        let inv_alpha = 255 - alpha;
        // Each channel: (self_ch * inv_alpha + other_ch * alpha + 127) / 255
        // Use 255-rounding trick: (x + 127) / 255 ≈ (x * 257 + 32768) >> 16
        let blend_ch = |a: u8, b: u8| -> u8 {
            let val = (a as u32 * inv_alpha + b as u32 * alpha + 127) / 255;
            val.min(255) as u8
        };
        let new_a = blend_ch(self.a, other.a).max(self.a);
        Self::new(
            blend_ch(self.r, other.r),
            blend_ch(self.g, other.g),
            blend_ch(self.b, other.b),
            new_a,
        )
    }

    /// Lerp between two colors.
    ///
    /// Uses fixed-point arithmetic for efficiency on scalar and SIMD paths.
    #[must_use]
    pub fn lerp(self, other: Self, t: f32) -> Self {
        let t = t.clamp(0.0, 1.0);
        // Fixed-point: scale t to [0, 256]
        let t_fp = (t * 256.0) as u32;
        let inv_fp = 256 - t_fp;
        let lerp_ch = |a: u8, b: u8| -> u8 {
            ((a as u32 * inv_fp + b as u32 * t_fp + 128) >> 8).min(255) as u8
        };
        Self::new(
            lerp_ch(self.r, other.r),
            lerp_ch(self.g, other.g),
            lerp_ch(self.b, other.b),
            lerp_ch(self.a, other.a),
        )
    }
}

/// Rectangle in 2D space.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct Rect {
    /// X coordinate of top-left corner.
    pub x: f32,
    /// Y coordinate of top-left corner.
    pub y: f32,
    /// Width of rectangle.
    pub width: f32,
    /// Height of rectangle.
    pub height: f32,
}

impl Rect {
    /// Create a new rectangle.
    #[must_use]
    pub const fn new(x: f32, y: f32, width: f32, height: f32) -> Self {
        Self {
            x,
            y,
            width,
            height,
        }
    }

    /// Check if point is inside rectangle.
    #[must_use]
    pub fn contains(&self, x: f32, y: f32) -> bool {
        x >= self.x && x <= self.x + self.width && y >= self.y && y <= self.y + self.height
    }
}

/// 2D vector.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct Vec2 {
    /// X component.
    pub x: f32,
    /// Y component.
    pub y: f32,
}

impl Vec2 {
    /// Create a new vector.
    #[must_use]
    pub const fn new(x: f32, y: f32) -> Self {
        Self { x, y }
    }

    /// Get vector length.
    #[must_use]
    pub fn length(&self) -> f32 {
        (self.x * self.x + self.y * self.y).sqrt()
    }

    /// Normalize vector.
    #[must_use]
    pub fn normalize(&self) -> Self {
        let len = self.length();
        if len > 0.0 {
            Self::new(self.x / len, self.y / len)
        } else {
            *self
        }
    }

    /// Dot product.
    #[must_use]
    pub fn dot(&self, other: &Self) -> f32 {
        self.x * other.x + self.y * other.y
    }
}

/// 3D vector.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct Vec3 {
    /// X component.
    pub x: f32,
    /// Y component.
    pub y: f32,
    /// Z component.
    pub z: f32,
}

impl Vec3 {
    /// Create a new 3D vector.
    #[must_use]
    pub const fn new(x: f32, y: f32, z: f32) -> Self {
        Self { x, y, z }
    }

    /// Zero vector.
    #[must_use]
    pub const fn zero() -> Self {
        Self::new(0.0, 0.0, 0.0)
    }

    /// Get vector length.
    #[must_use]
    pub fn length(&self) -> f32 {
        (self.x * self.x + self.y * self.y + self.z * self.z).sqrt()
    }

    /// Normalize vector. Returns zero vector if length is zero.
    #[must_use]
    pub fn normalize(&self) -> Self {
        let len = self.length();
        if len > 0.0 {
            Self::new(self.x / len, self.y / len, self.z / len)
        } else {
            *self
        }
    }

    /// Dot product.
    #[must_use]
    pub fn dot(&self, other: &Self) -> f32 {
        self.x * other.x + self.y * other.y + self.z * other.z
    }

    /// Cross product.
    #[must_use]
    pub fn cross(&self, other: &Self) -> Self {
        Self::new(
            self.y * other.z - self.z * other.y,
            self.z * other.x - self.x * other.z,
            self.x * other.y - self.y * other.x,
        )
    }

    /// Linear interpolation between two vectors.
    #[must_use]
    pub fn lerp(&self, other: &Self, t: f32) -> Self {
        let t = t.clamp(0.0, 1.0);
        Self::new(
            self.x + (other.x - self.x) * t,
            self.y + (other.y - self.y) * t,
            self.z + (other.z - self.z) * t,
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_frame_creation() {
        let frame = Frame::new(1920, 1080).expect("should succeed in test");
        assert_eq!(frame.width, 1920);
        assert_eq!(frame.height, 1080);
        assert_eq!(frame.byte_size(), 1920 * 1080 * 4);
    }

    #[test]
    fn test_frame_pixel_access() {
        let mut frame = Frame::new(10, 10).expect("should succeed in test");
        frame.set_pixel(5, 5, [255, 0, 0, 255]);
        let pixel = frame.get_pixel(5, 5).expect("should succeed in test");
        assert_eq!(pixel, [255, 0, 0, 255]);
    }

    #[test]
    fn test_color_blend() {
        let c1 = Color::rgb(255, 0, 0);
        let c2 = Color::new(0, 255, 0, 128);
        let blended = c1.blend(c2);
        assert!(blended.r > 0 && blended.r < 255);
        assert!(blended.g > 0 && blended.g < 255);
    }

    #[test]
    fn test_color_lerp() {
        let c1 = Color::rgb(0, 0, 0);
        let c2 = Color::rgb(255, 255, 255);
        let mid = c1.lerp(c2, 0.5);
        assert!(mid.r > 100 && mid.r < 155);
    }

    #[test]
    fn test_easing_functions() {
        assert_eq!(EasingFunction::Linear.apply(0.5), 0.5);
        assert!(EasingFunction::EaseIn.apply(0.5) < 0.5);
        assert!(EasingFunction::EaseOut.apply(0.5) > 0.5);
    }

    #[test]
    fn test_easing_boundary_values() {
        // All easing functions should return 0 at t=0 and 1 at t=1
        let easings = [
            EasingFunction::Linear,
            EasingFunction::EaseIn,
            EasingFunction::EaseOut,
            EasingFunction::EaseInOut,
            EasingFunction::ElasticIn,
            EasingFunction::ElasticOut,
            EasingFunction::ElasticInOut,
            EasingFunction::BounceIn,
            EasingFunction::BounceOut,
            EasingFunction::BounceInOut,
            EasingFunction::Spring,
            EasingFunction::BackIn,
            EasingFunction::BackOut,
            EasingFunction::BackInOut,
        ];

        for easing in easings {
            let at_zero = easing.apply(0.0);
            let at_one = easing.apply(1.0);
            assert!(
                at_zero.abs() < 0.01,
                "{easing:?} at t=0 should be ~0.0, got {at_zero}"
            );
            assert!(
                (at_one - 1.0).abs() < 0.01,
                "{easing:?} at t=1 should be ~1.0, got {at_one}"
            );
        }
    }

    #[test]
    fn test_elastic_in_overshoots() {
        // Elastic in should produce negative values (undershoot) before reaching target
        let mut has_negative = false;
        for i in 1..100 {
            let t = i as f32 / 100.0;
            let val = EasingFunction::ElasticIn.apply(t);
            if val < -0.001 {
                has_negative = true;
            }
        }
        assert!(
            has_negative,
            "elastic in should undershoot (negative values)"
        );
    }

    #[test]
    fn test_elastic_out_overshoots() {
        // Elastic out should overshoot beyond 1.0
        let mut has_overshoot = false;
        for i in 1..100 {
            let t = i as f32 / 100.0;
            let val = EasingFunction::ElasticOut.apply(t);
            if val > 1.001 {
                has_overshoot = true;
            }
        }
        assert!(has_overshoot, "elastic out should overshoot beyond 1.0");
    }

    #[test]
    fn test_bounce_out_never_negative() {
        for i in 0..=100 {
            let t = i as f32 / 100.0;
            let val = EasingFunction::BounceOut.apply(t);
            assert!(
                val >= -0.001,
                "bounce out should not go below 0 at t={t}, got {val}"
            );
        }
    }

    #[test]
    fn test_bounce_out_monotonically_reaches_one() {
        let val_end = EasingFunction::BounceOut.apply(1.0);
        assert!(
            (val_end - 1.0).abs() < 0.01,
            "bounce out at t=1 should be 1.0"
        );
    }

    #[test]
    fn test_bounce_in_never_exceeds_one() {
        for i in 0..=100 {
            let t = i as f32 / 100.0;
            let val = EasingFunction::BounceIn.apply(t);
            assert!(
                val <= 1.001,
                "bounce in should not exceed 1 at t={t}, got {val}"
            );
        }
    }

    #[test]
    fn test_spring_settles_at_one() {
        let val = EasingFunction::Spring.apply(1.0);
        assert!(
            (val - 1.0).abs() < 0.01,
            "spring should settle at 1.0, got {val}"
        );
    }

    #[test]
    fn test_spring_overshoots() {
        // Damped spring should overshoot 1.0 at some point
        let mut has_overshoot = false;
        for i in 1..100 {
            let t = i as f32 / 100.0;
            let val = EasingFunction::Spring.apply(t);
            if val > 1.01 {
                has_overshoot = true;
            }
        }
        assert!(has_overshoot, "spring should overshoot 1.0");
    }

    #[test]
    fn test_back_in_undershoots() {
        // Back-in pulls back (goes negative) before advancing
        let mut has_negative = false;
        for i in 1..50 {
            let t = i as f32 / 100.0;
            let val = EasingFunction::BackIn.apply(t);
            if val < -0.001 {
                has_negative = true;
            }
        }
        assert!(has_negative, "back-in should undershoot (negative values)");
    }

    #[test]
    fn test_back_out_overshoots() {
        let mut has_overshoot = false;
        for i in 50..100 {
            let t = i as f32 / 100.0;
            let val = EasingFunction::BackOut.apply(t);
            if val > 1.001 {
                has_overshoot = true;
            }
        }
        assert!(has_overshoot, "back-out should overshoot beyond 1.0");
    }

    #[test]
    fn test_easing_clamped_input() {
        // Values outside [0,1] should be clamped
        let easings = [
            EasingFunction::Linear,
            EasingFunction::ElasticIn,
            EasingFunction::BounceOut,
            EasingFunction::Spring,
            EasingFunction::BackIn,
        ];
        for easing in easings {
            let below = easing.apply(-0.5);
            let above = easing.apply(1.5);
            assert!(
                (below - easing.apply(0.0)).abs() < 0.01,
                "{easing:?}: apply(-0.5) should equal apply(0.0)"
            );
            assert!(
                (above - easing.apply(1.0)).abs() < 0.01,
                "{easing:?}: apply(1.5) should equal apply(1.0)"
            );
        }
    }

    #[test]
    fn test_parameter_track() {
        let mut track = ParameterTrack::new();
        track.add_keyframe(0.0, 0.0, EasingFunction::Linear);
        track.add_keyframe(1.0, 1.0, EasingFunction::Linear);

        assert_eq!(track.evaluate(0.0), Some(0.0));
        assert_eq!(track.evaluate(1.0), Some(1.0));
        let mid = track.evaluate(0.5).expect("should succeed in test");
        assert!((mid - 0.5).abs() < 0.01);
    }

    #[test]
    fn test_rect_contains() {
        let rect = Rect::new(10.0, 10.0, 100.0, 100.0);
        assert!(rect.contains(50.0, 50.0));
        assert!(!rect.contains(5.0, 5.0));
        assert!(!rect.contains(150.0, 150.0));
    }

    #[test]
    fn test_vec2_operations() {
        let v1 = Vec2::new(3.0, 4.0);
        assert_eq!(v1.length(), 5.0);

        let v2 = v1.normalize();
        assert!((v2.length() - 1.0).abs() < 0.0001);

        let v3 = Vec2::new(1.0, 0.0);
        let v4 = Vec2::new(0.0, 1.0);
        assert_eq!(v3.dot(&v4), 0.0);
    }

    #[test]
    fn test_vec3_length() {
        let v = Vec3::new(1.0, 2.0, 2.0);
        assert!((v.length() - 3.0).abs() < 1e-5);
    }

    #[test]
    fn test_vec3_normalize() {
        let v = Vec3::new(3.0, 0.0, 0.0);
        let n = v.normalize();
        assert!((n.length() - 1.0).abs() < 1e-5);
        assert!((n.x - 1.0).abs() < 1e-5);
    }

    #[test]
    fn test_vec3_dot() {
        let a = Vec3::new(1.0, 0.0, 0.0);
        let b = Vec3::new(0.0, 1.0, 0.0);
        assert_eq!(a.dot(&b), 0.0);
    }

    #[test]
    fn test_vec3_cross() {
        let a = Vec3::new(1.0, 0.0, 0.0);
        let b = Vec3::new(0.0, 1.0, 0.0);
        let c = a.cross(&b);
        assert!((c.z - 1.0).abs() < 1e-5);
        assert!(c.x.abs() < 1e-5);
        assert!(c.y.abs() < 1e-5);
    }

    #[test]
    fn test_vec3_lerp() {
        let a = Vec3::new(0.0, 0.0, 0.0);
        let b = Vec3::new(2.0, 4.0, 6.0);
        let mid = a.lerp(&b, 0.5);
        assert!((mid.x - 1.0).abs() < 1e-5);
        assert!((mid.y - 2.0).abs() < 1e-5);
        assert!((mid.z - 3.0).abs() < 1e-5);
    }

    #[test]
    fn test_vec3_zero() {
        let z = Vec3::zero();
        assert_eq!(z.length(), 0.0);
    }

    #[test]
    fn test_frame_clear_simd() {
        let mut frame = Frame::new(64, 64).expect("frame");
        frame.clear([255, 128, 0, 255]);
        let p = frame.get_pixel(32, 32).expect("pixel");
        assert_eq!(p, [255, 128, 0, 255]);
        // Check first and last pixels
        let first = frame.get_pixel(0, 0).expect("first");
        let last = frame.get_pixel(63, 63).expect("last");
        assert_eq!(first, [255, 128, 0, 255]);
        assert_eq!(last, [255, 128, 0, 255]);
    }

    #[test]
    fn test_color_blend_integer_path() {
        let base = Color::rgb(200, 100, 50);
        let overlay = Color::new(0, 200, 100, 128);
        let result = base.blend(overlay);
        // Result should be between base and overlay values
        assert!(result.r < 200);
        assert!(result.g > 100);
    }

    #[test]
    fn test_color_lerp_fixed_point() {
        let a = Color::rgb(0, 0, 0);
        let b = Color::rgb(100, 200, 50);
        let mid = a.lerp(b, 0.5);
        // Should be approximately half
        assert!((mid.r as i32 - 50).abs() <= 2);
        assert!((mid.g as i32 - 100).abs() <= 2);
        assert!((mid.b as i32 - 25).abs() <= 2);
    }

    // ParameterTrack edge-case tests

    #[test]
    fn test_param_track_empty_returns_none() {
        let track = ParameterTrack::new();
        assert_eq!(track.evaluate(0.0), None);
    }

    #[test]
    fn test_param_track_single_keyframe_constant() {
        let mut track = ParameterTrack::new();
        track.add_keyframe(1.0, 42.0, EasingFunction::Linear);
        // Before, at, and after the single keyframe should all return the same value.
        let v0 = track
            .evaluate(0.5)
            .expect("single keyframe must produce a value");
        let v1 = track
            .evaluate(1.0)
            .expect("single keyframe must produce a value");
        let v2 = track
            .evaluate(2.0)
            .expect("single keyframe must produce a value");
        assert!((v0 - 42.0_f32).abs() < 1e-5, "expected 42.0, got {v0}");
        assert!((v1 - 42.0_f32).abs() < 1e-5, "expected 42.0, got {v1}");
        assert!((v2 - 42.0_f32).abs() < 1e-5, "expected 42.0, got {v2}");
    }

    #[test]
    fn test_param_track_before_first_clamps() {
        let mut track = ParameterTrack::new();
        track.add_keyframe(1.0, 10.0, EasingFunction::Linear);
        track.add_keyframe(2.0, 20.0, EasingFunction::Linear);
        // Querying before first keyframe should clamp to the first value.
        let v = track.evaluate(0.0).expect("must produce a value");
        assert!(
            (v - 10.0_f32).abs() < 1e-5,
            "expected clamp to 10.0, got {v}"
        );
    }

    #[test]
    fn test_param_track_after_last_clamps() {
        let mut track = ParameterTrack::new();
        track.add_keyframe(1.0, 10.0, EasingFunction::Linear);
        track.add_keyframe(2.0, 20.0, EasingFunction::Linear);
        // Querying after last keyframe should clamp to the last value.
        let v = track.evaluate(100.0).expect("must produce a value");
        assert!(
            (v - 20.0_f32).abs() < 1e-5,
            "expected clamp to 20.0, got {v}"
        );
    }

    #[test]
    fn test_param_track_same_time_replaces() {
        let mut track = ParameterTrack::new();
        track.add_keyframe(1.0, 1.0, EasingFunction::Linear);
        track.add_keyframe(1.0, 2.0, EasingFunction::Linear);
        // Second add at the same time should replace the first.
        assert_eq!(track.len(), 1, "duplicate-time add should leave len=1");
        let v = track.evaluate(1.0).expect("must produce a value");
        assert!(
            (v - 2.0_f32).abs() < 1e-5,
            "replaced keyframe should have value 2.0, got {v}"
        );
    }

    #[test]
    fn test_param_track_nan_time_no_panic() {
        // NaN time must not cause a panic in add_keyframe or evaluate.
        let mut track = ParameterTrack::new();
        track.add_keyframe(f64::NAN, 1.0, EasingFunction::Linear);
        let _ = track.evaluate(f64::NAN);
    }

    #[test]
    fn test_param_track_linear_midpoint() {
        let mut track = ParameterTrack::new();
        track.add_keyframe(0.0, 0.0, EasingFunction::Linear);
        track.add_keyframe(1.0, 1.0, EasingFunction::Linear);
        let v = track.evaluate(0.5).expect("must produce a value");
        assert!(
            (v - 0.5_f32).abs() < 1e-5,
            "linear midpoint should be 0.5, got {v}"
        );
    }
}
