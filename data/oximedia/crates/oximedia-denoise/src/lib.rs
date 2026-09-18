//! Professional video denoising for `OxiMedia`.
//!
//! This crate provides comprehensive video denoising capabilities including:
//!
//! - **Spatial Denoising**: Remove noise within individual frames
//!   - Bilateral filtering (edge-preserving)
//!   - Non-Local Means (patch-based)
//!   - Wiener filtering (frequency-domain)
//!   - Wavelet denoising (multi-resolution)
//!
//! - **Temporal Denoising**: Remove noise across frame sequences
//!   - Temporal averaging (weighted)
//!   - Temporal median filtering
//!   - Motion-compensated filtering
//!   - Kalman filtering (prediction/correction)
//!
//! - **Hybrid Denoising**: Combined spatial and temporal
//!   - Spatio-temporal filtering
//!   - Adaptive content-aware denoising
//!
//! - **Advanced Features**:
//!   - Motion estimation and compensation
//!   - Film grain analysis and preservation
//!   - Multi-scale pyramid and wavelet processing
//!   - Automatic noise level estimation
//!
//! # Example
//!
//! ```
//! use oximedia_denoise::{DenoiseConfig, Denoiser};
//! use oximedia_codec::VideoFrame;
//! use oximedia_core::PixelFormat;
//!
//! let config = DenoiseConfig::medium();
//!
//! let mut denoiser = Denoiser::new(config).expect("valid config");
//! let mut frame = VideoFrame::new(PixelFormat::Yuv420p, 1920, 1080);
//! frame.allocate();
//!
//! // Process frame
//! let denoised = denoiser.process(&frame).expect("processing ok");
//! ```

// SIMD modules (spatial::bilateral_simd, spatial::bilateral apply_simd) use
// unsafe intrinsics and override this via #[allow(unsafe_code)] at module level.
#![deny(unsafe_code)]
// Algorithmic casts for bounds checking in image processing are necessary
#![allow(
    clippy::cast_possible_wrap,
    clippy::cast_possible_truncation,
    clippy::cast_precision_loss,
    clippy::cast_sign_loss,
    clippy::missing_errors_doc,
    clippy::missing_panics_doc,
    clippy::must_use_candidate,
    clippy::similar_names,
    clippy::too_many_arguments,
    clippy::module_name_repetitions
)]
#![warn(missing_docs)]

pub mod adaptive_denoise;
pub mod adaptive_temporal_window;
pub mod audio;
pub mod audio_denoise;
pub mod bilateral;
pub mod bm3d;
pub mod chroma_denoise;
pub mod deblock;
pub mod deep_prior;
pub mod denoise_config;
pub mod denoise_metrics;
pub mod estimator;
pub mod frame2d;
pub mod grain;
pub mod hdr_denoise;
pub mod hybrid;
pub mod motion;
pub mod multiscale;
pub mod noise_estimate;
pub mod noise_model;
pub mod noise_monitor;
pub mod perceptual_shaping;
pub mod pipeline_mode;
pub mod profile;
pub mod region_denoise;
pub mod spatial;
pub mod spectral_gate;
pub mod spectral_subtraction;
pub mod streaming;
pub mod temporal;
pub mod video;
pub mod video_denoise;

#[cfg(test)]
mod regression_tests;

use noise_model::NoiseType;
use oximedia_codec::VideoFrame;
use oximedia_core::PixelFormat;
use std::collections::VecDeque;
use thiserror::Error;

/// Denoising error types.
#[derive(Error, Debug)]
pub enum DenoiseError {
    /// Unsupported pixel format.
    #[error("Unsupported pixel format: {0:?}")]
    UnsupportedFormat(PixelFormat),

    /// Invalid configuration.
    #[error("Invalid configuration: {0}")]
    InvalidConfig(String),

    /// Processing error.
    #[error("Processing error: {0}")]
    ProcessingError(String),

    /// Motion estimation failed.
    #[error("Motion estimation failed: {0}")]
    MotionEstimationError(String),

    /// Insufficient frames for temporal processing.
    #[error("Insufficient frames: need at least {0}")]
    InsufficientFrames(usize),
}

/// Result type for denoising operations.
pub type DenoiseResult<T> = Result<T, DenoiseError>;

/// Denoising mode.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DenoiseMode {
    /// Fast bilateral filter (real-time capable).
    Fast,
    /// Balanced motion-compensated temporal + spatial.
    Balanced,
    /// High quality `NLMeans` or wavelet (slow but best quality).
    Quality,
    /// Grain-aware mode that preserves film grain.
    GrainAware,
    /// Custom mode with manual algorithm selection.
    Custom,
}

/// Spatial algorithm used in Custom mode.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SpatialAlgorithm {
    /// Bilateral filter (edge-preserving, fast).
    Bilateral,
    /// Non-Local Means (patch-based, high quality).
    NlMeans,
    /// Wiener filter (frequency-domain optimal).
    Wiener,
    /// Wavelet denoising (multi-resolution).
    Wavelet,
    /// BM3D (block-matching 3D, highest quality).
    Bm3d,
}

/// Configuration for Custom denoising mode.
#[derive(Clone, Debug)]
pub struct CustomConfig {
    /// Spatial algorithm to use.
    pub spatial_algorithm: SpatialAlgorithm,
    /// Enable temporal filtering in addition to spatial.
    pub enable_temporal: bool,
    /// Enable grain-aware processing.
    pub enable_grain_aware: bool,
    /// Wavelet threshold method (only used when spatial_algorithm is Wavelet).
    pub wavelet_threshold: spatial::wavelet::ThresholdMethod,
}

impl Default for CustomConfig {
    fn default() -> Self {
        Self {
            spatial_algorithm: SpatialAlgorithm::Bilateral,
            enable_temporal: false,
            enable_grain_aware: false,
            wavelet_threshold: spatial::wavelet::ThresholdMethod::Soft,
        }
    }
}

/// Per-channel denoising strength configuration.
///
/// Allows independent control of luma (Y) and chroma (Cb/Cr) denoising
/// strength, which is important because chroma channels are typically
/// noisier and can tolerate more aggressive filtering.
#[derive(Clone, Debug)]
pub struct ChannelStrength {
    /// Luma (Y) denoising strength (0.0 = none, 1.0 = maximum).
    pub luma: f32,
    /// Chroma Cb denoising strength (0.0 = none, 1.0 = maximum).
    pub chroma_cb: f32,
    /// Chroma Cr denoising strength (0.0 = none, 1.0 = maximum).
    pub chroma_cr: f32,
}

impl Default for ChannelStrength {
    fn default() -> Self {
        Self {
            luma: 0.5,
            chroma_cb: 0.5,
            chroma_cr: 0.5,
        }
    }
}

impl ChannelStrength {
    /// Create uniform strength across all channels.
    #[must_use]
    pub fn uniform(strength: f32) -> Self {
        let s = strength.clamp(0.0, 1.0);
        Self {
            luma: s,
            chroma_cb: s,
            chroma_cr: s,
        }
    }

    /// Create with stronger chroma denoising (common for high-ISO footage).
    #[must_use]
    pub fn chroma_heavy(luma: f32, chroma: f32) -> Self {
        Self {
            luma: luma.clamp(0.0, 1.0),
            chroma_cb: chroma.clamp(0.0, 1.0),
            chroma_cr: chroma.clamp(0.0, 1.0),
        }
    }

    /// Get strength for a given plane index (0=Y, 1=Cb, 2=Cr).
    #[must_use]
    pub fn for_plane(&self, plane_idx: usize) -> f32 {
        match plane_idx {
            0 => self.luma,
            1 => self.chroma_cb,
            _ => self.chroma_cr,
        }
    }

    /// Validate all channel strengths are in range.
    pub fn validate(&self) -> DenoiseResult<()> {
        for (name, val) in [
            ("luma", self.luma),
            ("chroma_cb", self.chroma_cb),
            ("chroma_cr", self.chroma_cr),
        ] {
            if !(0.0..=1.0).contains(&val) {
                return Err(DenoiseError::InvalidConfig(format!(
                    "Channel strength {name} must be between 0.0 and 1.0, got {val}"
                )));
            }
        }
        Ok(())
    }
}

/// Denoising configuration.
#[derive(Clone, Debug)]
pub struct DenoiseConfig {
    /// Processing mode.
    pub mode: DenoiseMode,
    /// Denoising strength (0.0 = none, 1.0 = maximum).
    pub strength: f32,
    /// Temporal window size (number of frames to consider).
    pub temporal_window: usize,
    /// Preserve edges while denoising.
    pub preserve_edges: bool,
    /// Preserve film grain.
    pub preserve_grain: bool,
    /// Noise model type to assume/target (affects algorithm tuning).
    pub noise_model: NoiseType,
    /// Per-channel denoising strength (overrides global `strength` when set).
    pub channel_strength: Option<ChannelStrength>,
    /// Custom mode configuration (only used when mode is Custom).
    pub custom_config: Option<CustomConfig>,
}

impl Default for DenoiseConfig {
    fn default() -> Self {
        Self {
            mode: DenoiseMode::Balanced,
            strength: 0.5,
            temporal_window: 5,
            preserve_edges: true,
            preserve_grain: false,
            noise_model: NoiseType::Gaussian,
            channel_strength: None,
            custom_config: None,
        }
    }
}

impl DenoiseConfig {
    /// Create configuration for light denoising.
    #[must_use]
    pub fn light() -> Self {
        Self {
            strength: 0.3,
            ..Default::default()
        }
    }

    /// Create configuration for medium denoising.
    #[must_use]
    pub fn medium() -> Self {
        Self {
            strength: 0.5,
            ..Default::default()
        }
    }

    /// Create configuration for strong denoising.
    #[must_use]
    pub fn strong() -> Self {
        Self {
            strength: 0.8,
            temporal_window: 7,
            ..Default::default()
        }
    }

    /// Create configuration with a specific noise model.
    #[must_use]
    pub fn with_noise_model(mut self, noise_model: NoiseType) -> Self {
        self.noise_model = noise_model;
        self
    }

    /// Create configuration with per-channel strength.
    #[must_use]
    pub fn with_channel_strength(mut self, channel_strength: ChannelStrength) -> Self {
        self.channel_strength = Some(channel_strength);
        self
    }

    /// Create configuration for Custom mode with specific algorithm.
    #[must_use]
    pub fn custom(spatial_algorithm: SpatialAlgorithm) -> Self {
        Self {
            mode: DenoiseMode::Custom,
            custom_config: Some(CustomConfig {
                spatial_algorithm,
                ..Default::default()
            }),
            ..Default::default()
        }
    }

    /// Get effective strength for a given plane index.
    #[must_use]
    pub fn effective_strength(&self, plane_idx: usize) -> f32 {
        if let Some(ref cs) = self.channel_strength {
            cs.for_plane(plane_idx)
        } else {
            self.strength
        }
    }

    /// Validate configuration.
    pub fn validate(&self) -> DenoiseResult<()> {
        if !(0.0..=1.0).contains(&self.strength) {
            return Err(DenoiseError::InvalidConfig(
                "Strength must be between 0.0 and 1.0".to_string(),
            ));
        }

        if self.temporal_window < 3 || self.temporal_window > 15 {
            return Err(DenoiseError::InvalidConfig(
                "Temporal window must be between 3 and 15".to_string(),
            ));
        }

        if self.temporal_window % 2 == 0 {
            return Err(DenoiseError::InvalidConfig(
                "Temporal window must be odd".to_string(),
            ));
        }

        if let Some(ref cs) = self.channel_strength {
            cs.validate()?;
        }

        Ok(())
    }
}

/// Main denoiser interface.
pub struct Denoiser {
    config: DenoiseConfig,
    frame_buffer: VecDeque<VideoFrame>,
    motion_estimator: Option<motion::estimation::MotionEstimator>,
    noise_estimator: estimator::noise::NoiseEstimator,
}

impl Denoiser {
    /// Create a new denoiser with the given configuration.
    ///
    /// # Errors
    ///
    /// Returns [`DenoiseError::InvalidConfig`] if the configuration is invalid.
    pub fn new(config: DenoiseConfig) -> DenoiseResult<Self> {
        config.validate()?;

        Ok(Self {
            config,
            frame_buffer: VecDeque::new(),
            motion_estimator: None,
            noise_estimator: estimator::noise::NoiseEstimator::new(),
        })
    }

    /// Process a single frame.
    pub fn process(&mut self, frame: &VideoFrame) -> DenoiseResult<VideoFrame> {
        // Estimate noise level if not already done
        if self.noise_estimator.noise_level().is_none() {
            self.noise_estimator.estimate(frame)?;
        }

        // Add frame to buffer
        self.frame_buffer.push_back(frame.clone());

        // Keep buffer size limited using VecDeque for O(1) removal at front
        let max_buffer = self.config.temporal_window;
        if self.frame_buffer.len() > max_buffer {
            self.frame_buffer.pop_front();
        }

        // Apply denoising based on mode
        match self.config.mode {
            DenoiseMode::Fast => self.process_fast(frame),
            DenoiseMode::Balanced => self.process_balanced(frame),
            DenoiseMode::Quality => self.process_quality(frame),
            DenoiseMode::GrainAware => self.process_grain_aware(frame),
            DenoiseMode::Custom => self.process_custom(frame),
        }
    }

    /// Reset the denoiser state.
    pub fn reset(&mut self) {
        self.frame_buffer.clear();
        self.motion_estimator = None;
        self.noise_estimator = estimator::noise::NoiseEstimator::new();
    }

    /// Get the current configuration.
    #[must_use]
    pub fn config(&self) -> &DenoiseConfig {
        &self.config
    }

    /// Get estimated noise level.
    #[must_use]
    pub fn noise_level(&self) -> Option<f32> {
        self.noise_estimator.noise_level()
    }

    fn process_fast(&self, frame: &VideoFrame) -> DenoiseResult<VideoFrame> {
        spatial::bilateral::bilateral_filter(frame, self.config.strength)
    }

    fn process_balanced(&mut self, frame: &VideoFrame) -> DenoiseResult<VideoFrame> {
        if self.frame_buffer.len() < 3 {
            // Not enough frames for temporal processing, use spatial only
            return self.process_fast(frame);
        }

        // Use motion-compensated temporal + spatial
        // Collect VecDeque into a contiguous slice for the hybrid filter
        let buf_slice: Vec<VideoFrame> = self.frame_buffer.iter().cloned().collect();
        hybrid::spatiotemporal::spatio_temporal_denoise(
            frame,
            &buf_slice,
            self.config.strength,
            self.config.preserve_edges,
        )
    }

    fn process_quality(&self, frame: &VideoFrame) -> DenoiseResult<VideoFrame> {
        // Use Non-Local Means for highest quality
        spatial::nlmeans::nlmeans_filter(frame, self.config.strength)
    }

    fn process_grain_aware(&mut self, frame: &VideoFrame) -> DenoiseResult<VideoFrame> {
        // Analyze grain pattern
        let grain_map = grain::analysis::analyze_grain(frame)?;

        // Apply denoising with grain preservation
        grain::preserve::preserve_grain_denoise(frame, &grain_map, self.config.strength)
    }

    fn process_custom(&mut self, frame: &VideoFrame) -> DenoiseResult<VideoFrame> {
        let custom = self.config.custom_config.clone().unwrap_or_default();

        // Step 1: Apply spatial algorithm based on selection
        let spatial_result = match custom.spatial_algorithm {
            SpatialAlgorithm::Bilateral => self.apply_per_channel(frame, |f, strength| {
                spatial::bilateral::bilateral_filter(f, strength)
            })?,
            SpatialAlgorithm::NlMeans => self.apply_per_channel(frame, |f, strength| {
                spatial::nlmeans::nlmeans_filter(f, strength)
            })?,
            SpatialAlgorithm::Wiener => self.apply_per_channel(frame, |f, strength| {
                spatial::wiener::wiener_filter(f, strength)
            })?,
            SpatialAlgorithm::Wavelet => {
                let method = custom.wavelet_threshold;
                self.apply_per_channel(frame, |f, strength| {
                    spatial::wavelet::wavelet_denoise(f, strength, method)
                })?
            }
            SpatialAlgorithm::Bm3d => {
                // BM3D: two-step denoise (basic bilateral + Wiener refinement)
                self.apply_per_channel(frame, |f, strength| {
                    spatial::wiener::two_step_denoise(f, strength)
                })?
            }
        };

        // Step 2: Optionally apply temporal filtering
        let temporal_result = if custom.enable_temporal && self.frame_buffer.len() >= 3 {
            let buf_slice: Vec<VideoFrame> = self.frame_buffer.iter().cloned().collect();
            hybrid::spatiotemporal::spatio_temporal_denoise(
                &spatial_result,
                &buf_slice,
                self.config.strength * 0.5,
                self.config.preserve_edges,
            )?
        } else {
            spatial_result
        };

        // Step 3: Optionally apply grain-aware post-processing
        if custom.enable_grain_aware {
            let grain_map = grain::analysis::analyze_grain(&temporal_result)?;
            grain::preserve::preserve_grain_denoise(
                &temporal_result,
                &grain_map,
                self.config.strength * 0.3,
            )
        } else {
            Ok(temporal_result)
        }
    }

    /// Apply a spatial filter function with per-channel strength control.
    ///
    /// When `channel_strength` is configured, processes each plane separately
    /// with its own strength value. Otherwise applies uniform strength.
    fn apply_per_channel<F>(&self, frame: &VideoFrame, filter_fn: F) -> DenoiseResult<VideoFrame>
    where
        F: Fn(&VideoFrame, f32) -> DenoiseResult<VideoFrame>,
    {
        let channel_strength = match &self.config.channel_strength {
            Some(cs) => cs.clone(),
            None => return filter_fn(frame, self.config.strength),
        };

        // We need to process each plane with its own strength.
        // Create single-plane frames, filter each, then reassemble.
        let mut output = frame.clone();

        for (plane_idx, plane) in output.planes.iter_mut().enumerate() {
            let plane_strength = channel_strength.for_plane(plane_idx);
            if plane_strength < 1e-6 {
                // Skip filtering for near-zero strength
                continue;
            }

            // Build a temporary single-frame with only this plane's data
            // and apply the filter at the plane-specific strength.
            // We use a simpler approach: apply the full filter then blend
            // with original based on per-channel strength ratio.
            let filtered = filter_fn(frame, plane_strength)?;
            if plane_idx < filtered.planes.len() {
                let src = &filtered.planes[plane_idx];
                let (w, h) = frame.plane_dimensions(plane_idx);
                for y in 0..h as usize {
                    for x in 0..w as usize {
                        let idx = y * plane.stride + x;
                        if idx < src.data.len() && idx < plane.data.len() {
                            plane.data[idx] = src.data[idx];
                        }
                    }
                }
            }
        }

        Ok(output)
    }

    /// Get noise-model-aware parameters for filter tuning.
    ///
    /// Different noise models benefit from different filter parameter scaling:
    /// - Gaussian: standard sigma-based filtering
    /// - Poisson: signal-dependent — scale strength with local brightness
    /// - Salt-and-pepper: impulse noise — prefer median-type filtering
    /// - Speckle: multiplicative — scale strength with local variance
    #[must_use]
    pub fn noise_model_strength_scale(&self, local_mean: f32) -> f32 {
        match self.config.noise_model {
            NoiseType::Gaussian | NoiseType::Mixed => 1.0,
            NoiseType::Poisson => {
                // Shot noise variance is proportional to signal level
                // Scale strength inversely with brightness (darker = more noise)
                let brightness_factor = (local_mean / 128.0).max(0.1);
                1.0 / brightness_factor.sqrt()
            }
            NoiseType::SaltAndPepper => {
                // Impulse noise: stronger filtering needed
                1.5
            }
            NoiseType::Speckle => {
                // Multiplicative noise: variance scales with signal squared
                let brightness_factor = (local_mean / 128.0).max(0.1);
                brightness_factor
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_config_default() {
        let config = DenoiseConfig::default();
        assert_eq!(config.mode, DenoiseMode::Balanced);
        assert!((config.strength - 0.5).abs() < f32::EPSILON);
        assert_eq!(config.temporal_window, 5);
    }

    #[test]
    fn test_config_presets() {
        let light = DenoiseConfig::light();
        assert!((light.strength - 0.3).abs() < f32::EPSILON);

        let medium = DenoiseConfig::medium();
        assert!((medium.strength - 0.5).abs() < f32::EPSILON);

        let strong = DenoiseConfig::strong();
        assert!((strong.strength - 0.8).abs() < f32::EPSILON);
        assert_eq!(strong.temporal_window, 7);
    }

    #[test]
    fn test_config_validation() {
        let mut config = DenoiseConfig::default();
        assert!(config.validate().is_ok());

        config.strength = 1.5;
        assert!(config.validate().is_err());

        config.strength = 0.5;
        config.temporal_window = 2;
        assert!(config.validate().is_err());

        config.temporal_window = 4; // even
        assert!(config.validate().is_err());
    }

    #[test]
    fn test_denoiser_creation() {
        let config = DenoiseConfig::default();
        let denoiser = Denoiser::new(config).expect("valid config");
        assert_eq!(denoiser.config().mode, DenoiseMode::Balanced);
    }

    #[test]
    fn test_denoiser_process() {
        let config = DenoiseConfig::default();
        let mut denoiser = Denoiser::new(config).expect("valid config");

        let mut frame = VideoFrame::new(PixelFormat::Yuv420p, 64, 64);
        frame.allocate();

        let result = denoiser.process(&frame);
        assert!(result.is_ok());
    }

    #[test]
    fn test_denoiser_reset() {
        let config = DenoiseConfig::default();
        let mut denoiser = Denoiser::new(config).expect("valid config");

        let mut frame = VideoFrame::new(PixelFormat::Yuv420p, 64, 64);
        frame.allocate();

        let _ = denoiser.process(&frame);
        assert!(!denoiser.frame_buffer.is_empty());

        denoiser.reset();
        assert!(denoiser.frame_buffer.is_empty());
    }

    // -------------------------------------------------------------------
    // Noise model selection tests
    // -------------------------------------------------------------------

    #[test]
    fn test_config_with_noise_model() {
        let config = DenoiseConfig::default().with_noise_model(NoiseType::Poisson);
        assert_eq!(config.noise_model, NoiseType::Poisson);
    }

    #[test]
    fn test_noise_model_all_variants() {
        for noise in [
            NoiseType::Gaussian,
            NoiseType::Poisson,
            NoiseType::SaltAndPepper,
            NoiseType::Speckle,
            NoiseType::Mixed,
        ] {
            let config = DenoiseConfig::default().with_noise_model(noise);
            assert_eq!(config.noise_model, noise);
            assert!(config.validate().is_ok());
        }
    }

    #[test]
    fn test_noise_model_strength_scale_gaussian() {
        let config = DenoiseConfig::default().with_noise_model(NoiseType::Gaussian);
        let denoiser = Denoiser::new(config).expect("valid config");
        let scale = denoiser.noise_model_strength_scale(128.0);
        assert!((scale - 1.0).abs() < f32::EPSILON, "Gaussian should be 1.0");
    }

    #[test]
    fn test_noise_model_strength_scale_poisson() {
        let config = DenoiseConfig::default().with_noise_model(NoiseType::Poisson);
        let denoiser = Denoiser::new(config).expect("valid config");
        // Dark regions should have higher scale (more noise)
        let dark_scale = denoiser.noise_model_strength_scale(16.0);
        let bright_scale = denoiser.noise_model_strength_scale(200.0);
        assert!(
            dark_scale > bright_scale,
            "Poisson: dark regions should need stronger filtering"
        );
    }

    #[test]
    fn test_noise_model_strength_scale_salt_pepper() {
        let config = DenoiseConfig::default().with_noise_model(NoiseType::SaltAndPepper);
        let denoiser = Denoiser::new(config).expect("valid config");
        let scale = denoiser.noise_model_strength_scale(128.0);
        assert!(scale > 1.0, "Salt-and-pepper should use stronger filtering");
    }

    #[test]
    fn test_noise_model_strength_scale_speckle() {
        let config = DenoiseConfig::default().with_noise_model(NoiseType::Speckle);
        let denoiser = Denoiser::new(config).expect("valid config");
        // Brighter regions have more speckle noise
        let dark_scale = denoiser.noise_model_strength_scale(32.0);
        let bright_scale = denoiser.noise_model_strength_scale(200.0);
        assert!(
            bright_scale > dark_scale,
            "Speckle: brighter regions should have higher scale"
        );
    }

    // -------------------------------------------------------------------
    // process_custom() algorithm selection tests
    // -------------------------------------------------------------------

    #[test]
    fn test_custom_bilateral() {
        let config = DenoiseConfig::custom(SpatialAlgorithm::Bilateral);
        let mut denoiser = Denoiser::new(config).expect("valid config");
        let mut frame = VideoFrame::new(PixelFormat::Yuv420p, 32, 32);
        frame.allocate();
        let result = denoiser.process(&frame);
        assert!(result.is_ok());
    }

    #[test]
    fn test_custom_nlmeans() {
        let config = DenoiseConfig::custom(SpatialAlgorithm::NlMeans);
        let mut denoiser = Denoiser::new(config).expect("valid config");
        let mut frame = VideoFrame::new(PixelFormat::Yuv420p, 24, 24);
        frame.allocate();
        let result = denoiser.process(&frame);
        assert!(result.is_ok());
    }

    #[test]
    fn test_custom_wiener() {
        let config = DenoiseConfig::custom(SpatialAlgorithm::Wiener);
        let mut denoiser = Denoiser::new(config).expect("valid config");
        let mut frame = VideoFrame::new(PixelFormat::Yuv420p, 32, 32);
        frame.allocate();
        let result = denoiser.process(&frame);
        assert!(result.is_ok());
    }

    #[test]
    fn test_custom_wavelet() {
        let config = DenoiseConfig::custom(SpatialAlgorithm::Wavelet);
        let mut denoiser = Denoiser::new(config).expect("valid config");
        let mut frame = VideoFrame::new(PixelFormat::Yuv420p, 32, 32);
        frame.allocate();
        let result = denoiser.process(&frame);
        assert!(result.is_ok());
    }

    #[test]
    fn test_custom_bm3d_two_step() {
        let config = DenoiseConfig::custom(SpatialAlgorithm::Bm3d);
        let mut denoiser = Denoiser::new(config).expect("valid config");
        let mut frame = VideoFrame::new(PixelFormat::Yuv420p, 32, 32);
        frame.allocate();
        let result = denoiser.process(&frame);
        assert!(result.is_ok());
    }

    #[test]
    fn test_custom_with_temporal() {
        let mut config = DenoiseConfig::custom(SpatialAlgorithm::Bilateral);
        if let Some(ref mut cc) = config.custom_config {
            cc.enable_temporal = true;
        }
        let mut denoiser = Denoiser::new(config).expect("valid config");
        let mut frame = VideoFrame::new(PixelFormat::Yuv420p, 32, 32);
        frame.allocate();

        // Feed enough frames for temporal
        for _ in 0..5 {
            let result = denoiser.process(&frame);
            assert!(result.is_ok());
        }
    }

    #[test]
    fn test_custom_with_grain_aware() {
        let mut config = DenoiseConfig::custom(SpatialAlgorithm::Bilateral);
        if let Some(ref mut cc) = config.custom_config {
            cc.enable_grain_aware = true;
        }
        let mut denoiser = Denoiser::new(config).expect("valid config");
        let mut frame = VideoFrame::new(PixelFormat::Yuv420p, 32, 32);
        frame.allocate();
        let result = denoiser.process(&frame);
        assert!(result.is_ok());
    }

    // -------------------------------------------------------------------
    // Per-channel denoising tests
    // -------------------------------------------------------------------

    #[test]
    fn test_channel_strength_uniform() {
        let cs = ChannelStrength::uniform(0.7);
        assert!((cs.luma - 0.7).abs() < f32::EPSILON);
        assert!((cs.chroma_cb - 0.7).abs() < f32::EPSILON);
        assert!((cs.chroma_cr - 0.7).abs() < f32::EPSILON);
    }

    #[test]
    fn test_channel_strength_chroma_heavy() {
        let cs = ChannelStrength::chroma_heavy(0.3, 0.9);
        assert!((cs.luma - 0.3).abs() < f32::EPSILON);
        assert!((cs.chroma_cb - 0.9).abs() < f32::EPSILON);
        assert!((cs.chroma_cr - 0.9).abs() < f32::EPSILON);
    }

    #[test]
    fn test_channel_strength_for_plane() {
        let cs = ChannelStrength {
            luma: 0.2,
            chroma_cb: 0.6,
            chroma_cr: 0.8,
        };
        assert!((cs.for_plane(0) - 0.2).abs() < f32::EPSILON);
        assert!((cs.for_plane(1) - 0.6).abs() < f32::EPSILON);
        assert!((cs.for_plane(2) - 0.8).abs() < f32::EPSILON);
    }

    #[test]
    fn test_channel_strength_validate() {
        let cs = ChannelStrength::uniform(0.5);
        assert!(cs.validate().is_ok());

        let bad = ChannelStrength {
            luma: 1.5,
            chroma_cb: 0.5,
            chroma_cr: 0.5,
        };
        assert!(bad.validate().is_err());

        let bad2 = ChannelStrength {
            luma: 0.5,
            chroma_cb: -0.1,
            chroma_cr: 0.5,
        };
        assert!(bad2.validate().is_err());
    }

    #[test]
    fn test_per_channel_denoising() {
        let cs = ChannelStrength::chroma_heavy(0.2, 0.8);
        let config = DenoiseConfig::custom(SpatialAlgorithm::Bilateral).with_channel_strength(cs);
        let mut denoiser = Denoiser::new(config).expect("valid config");
        let mut frame = VideoFrame::new(PixelFormat::Yuv420p, 32, 32);
        frame.allocate();
        let result = denoiser.process(&frame);
        assert!(result.is_ok());
    }

    #[test]
    fn test_effective_strength_with_channels() {
        let cs = ChannelStrength {
            luma: 0.1,
            chroma_cb: 0.5,
            chroma_cr: 0.9,
        };
        let config = DenoiseConfig::default().with_channel_strength(cs);
        assert!((config.effective_strength(0) - 0.1).abs() < f32::EPSILON);
        assert!((config.effective_strength(1) - 0.5).abs() < f32::EPSILON);
        assert!((config.effective_strength(2) - 0.9).abs() < f32::EPSILON);
    }

    #[test]
    fn test_effective_strength_without_channels() {
        let config = DenoiseConfig {
            strength: 0.7,
            ..Default::default()
        };
        assert!((config.effective_strength(0) - 0.7).abs() < f32::EPSILON);
        assert!((config.effective_strength(1) - 0.7).abs() < f32::EPSILON);
    }

    #[test]
    fn test_per_channel_zero_luma() {
        // Zero luma strength should skip luma filtering
        let cs = ChannelStrength {
            luma: 0.0,
            chroma_cb: 0.5,
            chroma_cr: 0.5,
        };
        let config = DenoiseConfig::custom(SpatialAlgorithm::Bilateral).with_channel_strength(cs);
        let mut denoiser = Denoiser::new(config).expect("valid config");
        let mut frame = VideoFrame::new(PixelFormat::Yuv420p, 32, 32);
        frame.allocate();
        let result = denoiser.process(&frame);
        assert!(result.is_ok());
    }

    // -------------------------------------------------------------------
    // Config builder / noise model integration tests
    // -------------------------------------------------------------------

    #[test]
    fn test_config_custom_builder() {
        let config = DenoiseConfig::custom(SpatialAlgorithm::Bm3d);
        assert_eq!(config.mode, DenoiseMode::Custom);
        let cc = config
            .custom_config
            .as_ref()
            .expect("should have custom config");
        assert_eq!(cc.spatial_algorithm, SpatialAlgorithm::Bm3d);
        assert!(!cc.enable_temporal);
        assert!(!cc.enable_grain_aware);
    }

    #[test]
    fn test_combined_noise_model_and_channels() {
        let config = DenoiseConfig::custom(SpatialAlgorithm::Wiener)
            .with_noise_model(NoiseType::Poisson)
            .with_channel_strength(ChannelStrength::chroma_heavy(0.3, 0.8));
        assert_eq!(config.noise_model, NoiseType::Poisson);
        assert!(config.channel_strength.is_some());
        assert!(config.validate().is_ok());

        let mut denoiser = Denoiser::new(config).expect("valid config");
        let mut frame = VideoFrame::new(PixelFormat::Yuv420p, 32, 32);
        frame.allocate();
        let result = denoiser.process(&frame);
        assert!(result.is_ok());
    }
}
