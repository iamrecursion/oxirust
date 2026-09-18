#![allow(
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::cast_possible_wrap,
    clippy::cast_sign_loss,
    clippy::too_many_arguments,
    clippy::similar_names,
    clippy::many_single_char_names
)]
//! Professional video scopes for broadcast-quality video analysis.
//!
//! This crate provides industry-standard video scopes for analyzing video signals,
//! including waveform monitors, vectorscopes, histograms, and parade displays.
//! All scopes are ITU-R BT.709/BT.2020 compliant and suitable for broadcast workflows.
//!
//! # Features
//!
//! - **Waveform Monitor**: Luma, RGB parade, RGB overlay, YCbCr waveform with graticule
//! - **Vectorscope**: YUV vectorscope with SMPTE color bars, skin tone line, gamut warnings
//! - **Histogram**: RGB and luma histograms with statistical overlays
//! - **Parade**: RGB and YCbCr parade displays with component selection
//! - **High Precision**: 8-bit and 10-bit support
//! - **Real-time**: Optimized for real-time video analysis
//! - **Broadcast Quality**: ITU-R BT.709/BT.2020 compliant
//!
//! # Example
//!
//! ```
//! use oximedia_scopes::{VideoScopes, ScopeType, WaveformMode, ScopeConfig};
//!
//! # fn example() -> Result<(), Box<dyn std::error::Error>> {
//! // Create video scopes analyzer
//! let mut scopes = VideoScopes::new(ScopeConfig::default());
//!
//! // Analyze frame with waveform
//! // let frame_data: &[u8] = /* your video frame */;
//! // let waveform = scopes.analyze(frame_data, 1920, 1080, ScopeType::WaveformLuma)?;
//!
//! // Render scope to image
//! // let image = scopes.render(&waveform)?;
//! # Ok(())
//! # }
//! ```

#![warn(missing_docs)]
#![allow(clippy::module_name_repetitions)]
#![allow(clippy::too_many_arguments)]

pub mod audio_phase_meter;
pub mod audio_scope;
pub mod audio_vectorscope;
pub mod bit_depth_scope;
pub mod chroma_level_scope;
pub mod cie;
pub mod cie_chromaticity;
pub mod cie_xy_diagram;
pub mod clipping_detector;
pub mod color_checker_scope;
pub mod color_temperature;
pub mod compliance;
pub mod exposure_histogram;
pub mod exposure_meter;
pub mod false_color;
pub mod false_color_mapping;
pub mod focus;
pub mod focus_assist;
pub mod gamut_scope;
pub mod gamut_scope_overlay;
pub mod hdr;
pub mod histogram;
pub mod histogram_3d;
pub mod histogram_parallel;
pub mod histogram_stats;
pub mod incremental_update;
pub mod lissajous;
pub mod loudness_scope;
pub mod luma_parade;
pub mod motion_vector_scope;
pub mod noise_floor_scope;
pub mod overlay;
pub mod parade;
pub mod parade_parallel;
pub mod peaking;
pub mod preview_downsample;
pub mod render;
pub mod rgb_balance;
pub mod rgb_parade;
pub mod safe_area_overlay;
pub mod scope_comparison;
pub mod scope_layout;
pub mod scope_recording;
pub mod scope_resolution;
pub mod scope_snapshot;
pub mod scope_snapshot_store;
pub mod signal_stats;
pub mod simd_convert;
pub mod stats;
pub mod timecode_overlay;
pub mod vectorscope;
pub mod vectorscope_targets;
pub mod waveform;
pub mod waveform_analyzer;
pub mod zebra;

use oximedia_core::OxiResult;

/// Type of video scope to generate.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ScopeType {
    /// Luma waveform (Y channel only).
    WaveformLuma,

    /// RGB parade waveform (R|G|B side-by-side).
    WaveformRgbParade,

    /// RGB overlay waveform (all channels overlaid).
    WaveformRgbOverlay,

    /// YCbCr waveform (Y|Cb|Cr parade).
    WaveformYcbcr,

    /// YUV vectorscope (Cb/Cr circular display).
    Vectorscope,

    /// RGB histogram.
    HistogramRgb,

    /// Luma histogram (Y channel only).
    HistogramLuma,

    /// RGB parade (R|G|B side-by-side vertical bars).
    ParadeRgb,

    /// YCbCr parade (Y|Cb|Cr side-by-side).
    ParadeYcbcr,

    /// False color exposure visualization.
    FalseColor,

    /// CIE 1931 chromaticity diagram.
    CieDiagram,

    /// Focus assist with edge peaking.
    FocusAssist,

    /// HDR waveform with PQ/HLG/nits scale.
    HdrWaveform,
}

/// Waveform display mode.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WaveformMode {
    /// Overlay all scanlines (brightest where most pixels).
    Overlay,

    /// Side-by-side parade (R|G|B or Y|Cb|Cr).
    Parade,

    /// Blended/averaged display.
    Blend,
}

/// Vectorscope display mode.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VectorscopeMode {
    /// Circular display (traditional).
    Circular,

    /// Rectangular display.
    Rectangular,
}

/// Histogram display mode.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HistogramMode {
    /// Overlay all channels.
    Overlay,

    /// Stacked channels.
    Stacked,

    /// Logarithmic scale.
    Logarithmic,
}

/// Configuration for video scopes.
#[derive(Debug, Clone)]
#[allow(clippy::struct_excessive_bools)]
pub struct ScopeConfig {
    /// Width of the scope display in pixels.
    pub width: u32,

    /// Height of the scope display in pixels.
    pub height: u32,

    /// Whether to show graticule overlay.
    pub show_graticule: bool,

    /// Whether to show text labels.
    pub show_labels: bool,

    /// Whether to enable anti-aliasing.
    pub anti_alias: bool,

    /// Waveform display mode.
    pub waveform_mode: WaveformMode,

    /// Vectorscope display mode.
    pub vectorscope_mode: VectorscopeMode,

    /// Histogram display mode.
    pub histogram_mode: HistogramMode,

    /// Vectorscope gain (1.0 = normal, 2.0 = 2x zoom).
    pub vectorscope_gain: f32,

    /// Whether to highlight out-of-gamut colors.
    pub highlight_gamut: bool,

    /// Color space for gamut warnings (709, 2020, P3).
    pub gamut_colorspace: GamutColorspace,

    /// Optional independent scope output resolution.
    ///
    /// When set, the scope display is rendered at this resolution instead of
    /// the source frame resolution.  The frame is analysed at the given
    /// scope size, enabling faster rendering independent of input resolution.
    /// `None` means "use `width` × `height` as declared above".
    pub resolution: Option<ScopeResolution>,
}

/// Independent scope output resolution (Item 6).
///
/// Decouples the scope rendering grid from the input video frame dimensions,
/// enabling faster processing by analysing at a lower resolution.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ScopeResolution {
    /// Width of the scope rendering grid in pixels.
    pub width: u32,
    /// Height of the scope rendering grid in pixels.
    pub height: u32,
}

/// Color space for gamut warnings.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GamutColorspace {
    /// Rec.709 (HD).
    Rec709,

    /// Rec.2020 (UHD/HDR).
    Rec2020,

    /// DCI-P3 (Digital Cinema).
    DciP3,
}

impl Default for ScopeConfig {
    fn default() -> Self {
        Self {
            width: 512,
            height: 512,
            show_graticule: true,
            show_labels: true,
            anti_alias: true,
            waveform_mode: WaveformMode::Overlay,
            vectorscope_mode: VectorscopeMode::Circular,
            histogram_mode: HistogramMode::Overlay,
            vectorscope_gain: 1.0,
            highlight_gamut: false,
            gamut_colorspace: GamutColorspace::Rec709,
            resolution: None,
        }
    }
}

/// Scope data ready for rendering.
#[derive(Debug, Clone)]
pub struct ScopeData {
    /// Width of the scope.
    pub width: u32,

    /// Height of the scope.
    pub height: u32,

    /// Scope pixel data (RGBA, row-major).
    pub data: Vec<u8>,

    /// Type of scope.
    pub scope_type: ScopeType,
}

/// Main video scopes analyzer.
pub struct VideoScopes {
    config: ScopeConfig,
    /// Cached last-seen frame (RGB24, row-major) for incremental region updates.
    cached_frame: Option<Vec<u8>>,
    /// Cached frame width (pixels).
    cached_width: u32,
    /// Cached frame height (pixels).
    cached_height: u32,
    /// Running per-channel histogram accumulators for incremental updates.
    /// `channel_bins[c][v]` = number of pixels with channel `c` value `v`
    /// in the currently-cached frame.
    channel_bins: [[u32; 256]; 3],
}

/// Configuration for downsampled real-time preview analysis (Item 2).
///
/// Downsampling allows scopes to be updated every frame at reduced accuracy
/// but significantly higher throughput for large frames.
#[derive(Debug, Clone, Copy)]
pub struct AnalysisConfig {
    /// Spatial downsampling factor (1 = full resolution, 2 = every 2nd pixel, …).
    ///
    /// Values ≥ 1 are valid; 1 means "process every pixel" (same as `analyze`).
    pub downsample_factor: usize,
}

impl Default for AnalysisConfig {
    fn default() -> Self {
        Self {
            downsample_factor: 2,
        }
    }
}

/// A rectangular region update for incremental scope accumulation (Item 3).
///
/// Supplying only the *changed* region of the frame allows the scope
/// accumulators to be updated in O(region pixels) time rather than O(frame)
/// time, which is significant for small motion regions in large frames.
#[derive(Debug, Clone)]
pub struct RegionUpdate {
    /// Left edge of the updated region in source-frame pixels.
    pub x: u32,
    /// Top edge of the updated region in source-frame pixels.
    pub y: u32,
    /// Width of the updated region in source-frame pixels.
    pub width: u32,
    /// Height of the updated region in source-frame pixels.
    pub height: u32,
    /// New pixel data for the region (RGB24, row-major, `width * height * 3` bytes).
    pub pixels: Vec<[u8; 3]>,
}

impl VideoScopes {
    /// Creates a new video scopes analyzer with the given configuration.
    ///
    /// # Examples
    ///
    /// ```
    /// use oximedia_scopes::{VideoScopes, ScopeConfig};
    ///
    /// let scopes = VideoScopes::new(ScopeConfig::default());
    /// ```
    #[must_use]
    pub fn new(config: ScopeConfig) -> Self {
        Self {
            config,
            cached_frame: None,
            cached_width: 0,
            cached_height: 0,
            channel_bins: [[0u32; 256]; 3],
        }
    }

    /// Analyzes a video frame and generates the specified scope.
    ///
    /// # Arguments
    ///
    /// * `frame` - Frame pixel data (RGB24 or `YUV420p`)
    /// * `width` - Frame width in pixels
    /// * `height` - Frame height in pixels
    /// * `scope_type` - Type of scope to generate
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - Frame dimensions are invalid
    /// - Frame data is insufficient
    /// - Scope generation fails
    ///
    /// # Examples
    ///
    /// ```no_run
    /// use oximedia_scopes::{VideoScopes, ScopeType, ScopeConfig};
    ///
    /// # fn example() -> Result<(), Box<dyn std::error::Error>> {
    /// let scopes = VideoScopes::new(ScopeConfig::default());
    /// let frame_data = vec![0u8; 1920 * 1080 * 3]; // RGB24
    /// let scope = scopes.analyze(&frame_data, 1920, 1080, ScopeType::WaveformLuma)?;
    /// # Ok(())
    /// # }
    /// ```
    pub fn analyze(
        &self,
        frame: &[u8],
        width: u32,
        height: u32,
        scope_type: ScopeType,
    ) -> OxiResult<ScopeData> {
        match scope_type {
            ScopeType::WaveformLuma => {
                waveform::generate_luma_waveform(frame, width, height, &self.config)
            }
            ScopeType::WaveformRgbParade => {
                waveform::generate_rgb_parade(frame, width, height, &self.config)
            }
            ScopeType::WaveformRgbOverlay => {
                waveform::generate_rgb_overlay(frame, width, height, &self.config)
            }
            ScopeType::WaveformYcbcr => {
                waveform::generate_ycbcr_waveform(frame, width, height, &self.config)
            }
            ScopeType::Vectorscope => {
                vectorscope::generate_vectorscope(frame, width, height, &self.config)
            }
            ScopeType::HistogramRgb => {
                histogram::generate_rgb_histogram(frame, width, height, &self.config)
            }
            ScopeType::HistogramLuma => {
                histogram::generate_luma_histogram(frame, width, height, &self.config)
            }
            ScopeType::ParadeRgb => parade::generate_rgb_parade(frame, width, height, &self.config),
            ScopeType::ParadeYcbcr => {
                parade::generate_ycbcr_parade(frame, width, height, &self.config)
            }
            ScopeType::FalseColor => {
                let scale = false_color::FalseColorScale::default();
                false_color::generate_false_color(
                    frame,
                    width,
                    height,
                    false_color::FalseColorMode::Ire,
                    &scale,
                )
            }
            ScopeType::CieDiagram => cie::generate_cie_diagram(frame, width, height, &self.config),
            ScopeType::FocusAssist => {
                let config = focus::FocusAssistConfig::default();
                focus::generate_focus_assist(frame, width, height, &config)
            }
            ScopeType::HdrWaveform => {
                let config = hdr::HdrWaveformConfig::default();
                hdr::generate_hdr_waveform(frame, width, height, &config)
            }
        }
    }

    /// Renders scope data to an RGBA image.
    ///
    /// # Errors
    ///
    /// Returns an error if rendering fails.
    pub fn render(&self, scope: &ScopeData) -> OxiResult<Vec<u8>> {
        // Scope data is already rendered, just return a copy
        Ok(scope.data.clone())
    }

    /// Analyzes a downsampled version of the frame for real-time preview (Item 2).
    ///
    /// Every `config.downsample_factor`-th pixel in both X and Y is sampled,
    /// reducing the number of pixels processed by `factor²` and enabling
    /// real-time scope updates at reduced spatial accuracy.
    ///
    /// `config.downsample_factor == 1` is identical to calling [`Self::analyze`].
    ///
    /// # Errors
    ///
    /// Returns an error if the frame is too small or if scope generation fails.
    pub fn analyze_downsampled(
        &self,
        frame: &[u8],
        width: u32,
        height: u32,
        config: &AnalysisConfig,
        scope_type: ScopeType,
    ) -> OxiResult<ScopeData> {
        let step = config.downsample_factor.max(1) as u32;
        if step == 1 {
            return self.analyze(frame, width, height, scope_type);
        }

        let expected = (width * height * 3) as usize;
        if frame.len() < expected {
            return Err(oximedia_core::OxiError::InvalidData(format!(
                "Frame data too small: expected {expected}, got {}",
                frame.len()
            )));
        }

        // Subsample the frame: retain every `step`-th pixel in both axes.
        let out_w = (width + step - 1) / step;
        let out_h = (height + step - 1) / step;
        let mut sub = Vec::with_capacity((out_w * out_h * 3) as usize);
        let mut sy = 0u32;
        while sy < height {
            let mut sx = 0u32;
            while sx < width {
                let off = ((sy * width + sx) * 3) as usize;
                sub.push(frame[off]);
                sub.push(frame[off + 1]);
                sub.push(frame[off + 2]);
                sx += step;
            }
            sy += step;
        }

        self.analyze(&sub, out_w, out_h, scope_type)
    }

    /// Generates a scope display at an independent resolution (Item 6).
    ///
    /// When `config.resolution` is `Some(res)`, the internal scope config is
    /// temporarily overridden to render at `res.width × res.height`.
    /// This decouples the scope grid from the source video resolution, enabling
    /// faster rendering for large input frames.
    ///
    /// # Errors
    ///
    /// Returns an error if the frame is too small or if scope generation fails.
    pub fn generate_scopes(
        &self,
        frame: &[u8],
        width: u32,
        height: u32,
        scope_type: ScopeType,
    ) -> OxiResult<ScopeData> {
        match self.config.resolution {
            None => self.analyze(frame, width, height, scope_type),
            Some(res) => {
                // Temporarily construct a config with the overridden resolution.
                let override_config = ScopeConfig {
                    width: res.width,
                    height: res.height,
                    resolution: None, // avoid infinite recursion
                    ..self.config.clone()
                };
                let temp_scopes = Self::new(override_config);
                temp_scopes.analyze(frame, width, height, scope_type)
            }
        }
    }

    /// Applies an incremental region update to the running accumulators (Item 3).
    ///
    /// The old pixel contributions from the changed region are subtracted from
    /// the per-channel histogram bins; the new pixel values are then added.
    /// After the call, [`VideoScopes::channel_bins`] reflects the histogram of
    /// the updated frame without re-scanning the entire image.
    ///
    /// The updated frame (with the new region blended in) is returned as a
    /// reference to the internal cache so callers can inspect it.
    ///
    /// # Errors
    ///
    /// Returns an error if the region is out of bounds or the pixel buffer
    /// size is inconsistent.
    pub fn update_region(
        &mut self,
        frame_width: u32,
        frame_height: u32,
        update: &RegionUpdate,
    ) -> OxiResult<()> {
        // Validate region bounds.
        if update.x + update.width > frame_width || update.y + update.height > frame_height {
            return Err(oximedia_core::OxiError::InvalidData(format!(
                "Region ({}, {}, {}×{}) exceeds frame {}×{}",
                update.x, update.y, update.width, update.height, frame_width, frame_height
            )));
        }
        let expected_pixels = (update.width * update.height) as usize;
        if update.pixels.len() != expected_pixels {
            return Err(oximedia_core::OxiError::InvalidData(format!(
                "pixels.len()={} but expected {}×{}={}",
                update.pixels.len(),
                update.width,
                update.height,
                expected_pixels
            )));
        }

        // Ensure the frame cache is initialised to zeros if not yet set.
        let frame_bytes = (frame_width * frame_height * 3) as usize;
        if self.cached_frame.is_none()
            || self.cached_width != frame_width
            || self.cached_height != frame_height
        {
            self.cached_frame = Some(vec![0u8; frame_bytes]);
            self.cached_width = frame_width;
            self.cached_height = frame_height;
            self.channel_bins = [[0u32; 256]; 3];
        }

        let cache = self
            .cached_frame
            .as_mut()
            .ok_or_else(|| oximedia_core::OxiError::InvalidData("cache missing".into()))?;

        // Subtract old contributions and apply new pixels.
        for row in 0..update.height {
            for col in 0..update.width {
                let frame_x = update.x + col;
                let frame_y = update.y + row;
                let cache_off = ((frame_y * frame_width + frame_x) * 3) as usize;
                let old_r = cache[cache_off];
                let old_g = cache[cache_off + 1];
                let old_b = cache[cache_off + 2];

                // Subtract old values (saturate at 0).
                self.channel_bins[0][old_r as usize] =
                    self.channel_bins[0][old_r as usize].saturating_sub(1);
                self.channel_bins[1][old_g as usize] =
                    self.channel_bins[1][old_g as usize].saturating_sub(1);
                self.channel_bins[2][old_b as usize] =
                    self.channel_bins[2][old_b as usize].saturating_sub(1);

                let region_idx = (row * update.width + col) as usize;
                let [new_r, new_g, new_b] = update.pixels[region_idx];

                // Add new values.
                self.channel_bins[0][new_r as usize] =
                    self.channel_bins[0][new_r as usize].saturating_add(1);
                self.channel_bins[1][new_g as usize] =
                    self.channel_bins[1][new_g as usize].saturating_add(1);
                self.channel_bins[2][new_b as usize] =
                    self.channel_bins[2][new_b as usize].saturating_add(1);

                // Write new pixel into cache.
                cache[cache_off] = new_r;
                cache[cache_off + 1] = new_g;
                cache[cache_off + 2] = new_b;
            }
        }

        Ok(())
    }

    /// Returns the current per-channel histogram accumulators built by
    /// [`Self::update_region`].  Index: `channel_bins()[channel][value]`.
    #[must_use]
    pub fn channel_bins(&self) -> &[[u32; 256]; 3] {
        &self.channel_bins
    }

    /// Seeds the running accumulators from a complete frame (RGB24).
    ///
    /// This initialises the cache and histogram so that subsequent
    /// [`Self::update_region`] calls produce accurate incremental results.
    ///
    /// # Errors
    ///
    /// Returns an error if the frame buffer is too small.
    pub fn seed_from_frame(&mut self, frame: &[u8], width: u32, height: u32) -> OxiResult<()> {
        let expected = (width * height * 3) as usize;
        if frame.len() < expected {
            return Err(oximedia_core::OxiError::InvalidData(format!(
                "Frame too small: need {expected}, got {}",
                frame.len()
            )));
        }

        self.cached_width = width;
        self.cached_height = height;
        self.cached_frame = Some(frame[..expected].to_vec());
        self.channel_bins = [[0u32; 256]; 3];

        let pixel_count = (width * height) as usize;
        for px in 0..pixel_count {
            let off = px * 3;
            self.channel_bins[0][frame[off] as usize] =
                self.channel_bins[0][frame[off] as usize].saturating_add(1);
            self.channel_bins[1][frame[off + 1] as usize] =
                self.channel_bins[1][frame[off + 1] as usize].saturating_add(1);
            self.channel_bins[2][frame[off + 2] as usize] =
                self.channel_bins[2][frame[off + 2] as usize].saturating_add(1);
        }

        Ok(())
    }

    /// Updates the configuration.
    pub fn set_config(&mut self, config: ScopeConfig) {
        self.config = config;
    }

    /// Gets the current configuration.
    #[must_use]
    pub const fn config(&self) -> &ScopeConfig {
        &self.config
    }
}

impl Default for VideoScopes {
    fn default() -> Self {
        Self::new(ScopeConfig::default())
    }
}

impl std::fmt::Debug for VideoScopes {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("VideoScopes")
            .field("config", &self.config)
            .field("cached_width", &self.cached_width)
            .field("cached_height", &self.cached_height)
            .finish_non_exhaustive()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_scope_config_default() {
        let config = ScopeConfig::default();
        assert_eq!(config.width, 512);
        assert_eq!(config.height, 512);
        assert!(config.show_graticule);
        assert!(config.show_labels);
    }

    #[test]
    fn test_video_scopes_new() {
        let scopes = VideoScopes::new(ScopeConfig::default());
        assert_eq!(scopes.config().width, 512);
    }

    #[test]
    fn test_video_scopes_default() {
        let scopes = VideoScopes::default();
        assert_eq!(scopes.config().width, 512);
    }

    // ── Helper ────────────────────────────────────────────────────────────────

    fn solid_frame(r: u8, g: u8, b: u8, w: u32, h: u32) -> Vec<u8> {
        let n = (w * h) as usize;
        let mut v = Vec::with_capacity(n * 3);
        for _ in 0..n {
            v.extend_from_slice(&[r, g, b]);
        }
        v
    }

    fn ramp_frame(w: u32, h: u32) -> Vec<u8> {
        (0..(w * h * 3) as usize).map(|i| (i % 256) as u8).collect()
    }

    // ── Item 2 tests (AnalysisConfig / analyze_downsampled) ──────────────────

    /// Downsampled analysis with factor 2 should produce a valid scope result
    /// with fewer pixels processed (implied by output having same scope dims).
    #[test]
    fn test_downsampled_analysis_valid_output() {
        let frame = ramp_frame(64, 64);
        let scopes = VideoScopes::new(ScopeConfig {
            width: 128,
            height: 128,
            show_graticule: false,
            show_labels: false,
            ..ScopeConfig::default()
        });
        let cfg = AnalysisConfig {
            downsample_factor: 2,
        };
        let result = scopes.analyze_downsampled(&frame, 64, 64, &cfg, ScopeType::HistogramRgb);
        assert!(result.is_ok(), "downsampled analysis should succeed");
        let data = result.expect("ok");
        // Scope output dimensions come from ScopeConfig, not frame size.
        assert_eq!(data.width, 128);
        assert_eq!(data.height, 128);
    }

    /// factor=1 must be identical to direct `analyze`.
    #[test]
    fn test_downsampled_result_within_tolerance() {
        let frame = solid_frame(100, 150, 200, 16, 16);
        let scopes = VideoScopes::new(ScopeConfig {
            width: 64,
            height: 64,
            show_graticule: false,
            show_labels: false,
            ..ScopeConfig::default()
        });
        let cfg_full = AnalysisConfig {
            downsample_factor: 1,
        };
        let full = scopes
            .analyze_downsampled(&frame, 16, 16, &cfg_full, ScopeType::WaveformLuma)
            .expect("full ok");
        let direct = scopes
            .analyze(&frame, 16, 16, ScopeType::WaveformLuma)
            .expect("direct ok");
        assert_eq!(full.data, direct.data, "factor=1 must equal direct analyze");
    }

    /// Small frame with large downsample factor should not panic.
    #[test]
    fn test_downsampled_large_factor_no_panic() {
        let frame = solid_frame(50, 60, 70, 4, 4);
        let scopes = VideoScopes::default();
        let cfg = AnalysisConfig {
            downsample_factor: 8,
        };
        let result = scopes.analyze_downsampled(&frame, 4, 4, &cfg, ScopeType::HistogramLuma);
        assert!(result.is_ok());
    }

    // ── Item 3 tests (RegionUpdate / update_region) ──────────────────────────

    /// After seeding from a full frame and applying no-op update (same pixels),
    /// the channel_bins must match a fresh seed of the same frame.
    #[test]
    fn test_incremental_update_matches_full_recompute() {
        let frame = ramp_frame(8, 8);
        let mut scopes = VideoScopes::new(ScopeConfig::default());
        scopes.seed_from_frame(&frame, 8, 8).expect("seed ok");
        let bins_before = *scopes.channel_bins();

        // Apply a no-op update (replace top-left pixel with its own values).
        let update = RegionUpdate {
            x: 0,
            y: 0,
            width: 1,
            height: 1,
            pixels: vec![[frame[0], frame[1], frame[2]]],
        };
        scopes.update_region(8, 8, &update).expect("update ok");

        // Bins should be unchanged.
        assert_eq!(
            *scopes.channel_bins(),
            bins_before,
            "no-op update should not change bins"
        );
    }

    /// Replace part of the frame with a new colour and verify that the running
    /// bins equal those from a fresh full-frame seed.
    #[test]
    fn test_incremental_update_partial_region() {
        // Start with a 4×4 frame, all red.
        let w = 4u32;
        let h = 4u32;
        let red = solid_frame(200, 0, 0, w, h);
        let mut scopes = VideoScopes::new(ScopeConfig::default());
        scopes.seed_from_frame(&red, w, h).expect("seed ok");

        // Replace the top-right 2×2 block with pure blue.
        let blue_pixels: Vec<[u8; 3]> = vec![[0, 0, 255]; 4];
        let update = RegionUpdate {
            x: 2,
            y: 0,
            width: 2,
            height: 2,
            pixels: blue_pixels,
        };
        scopes.update_region(w, h, &update).expect("update ok");

        // Build reference by seeding from the expected final frame.
        let mut expected_frame = red.clone();
        for row in 0..2u32 {
            for col in 0..2u32 {
                let off = ((row * w + (2 + col)) * 3) as usize;
                expected_frame[off] = 0;
                expected_frame[off + 1] = 0;
                expected_frame[off + 2] = 255;
            }
        }
        let mut ref_scopes = VideoScopes::new(ScopeConfig::default());
        ref_scopes
            .seed_from_frame(&expected_frame, w, h)
            .expect("ref seed ok");

        assert_eq!(
            *scopes.channel_bins(),
            *ref_scopes.channel_bins(),
            "incremental bins must match fresh full-frame bins"
        );
    }

    // ── Item 6 tests (ScopeResolution / generate_scopes) ─────────────────────

    /// Scope output must use the override resolution, not the frame dimensions.
    #[test]
    fn test_scope_resolution_independent_from_frame() {
        let frame = ramp_frame(64, 64);
        let scopes = VideoScopes::new(ScopeConfig {
            width: 512,
            height: 512,
            resolution: Some(ScopeResolution {
                width: 128,
                height: 64,
            }),
            show_graticule: false,
            show_labels: false,
            ..ScopeConfig::default()
        });
        let result = scopes
            .generate_scopes(&frame, 64, 64, ScopeType::WaveformLuma)
            .expect("generate_scopes ok");
        assert_eq!(
            result.width, 128,
            "scope width should equal resolution.width"
        );
        assert_eq!(
            result.height, 64,
            "scope height should equal resolution.height"
        );
        assert_eq!(result.data.len(), (128 * 64 * 4) as usize);
    }

    /// When resolution is None, generate_scopes must equal analyze.
    #[test]
    fn test_scope_resolution_none_equals_analyze() {
        let frame = solid_frame(128, 128, 128, 16, 16);
        let scopes = VideoScopes::new(ScopeConfig {
            width: 64,
            height: 64,
            resolution: None,
            show_graticule: false,
            show_labels: false,
            ..ScopeConfig::default()
        });
        let via_generate = scopes
            .generate_scopes(&frame, 16, 16, ScopeType::HistogramRgb)
            .expect("generate ok");
        let via_analyze = scopes
            .analyze(&frame, 16, 16, ScopeType::HistogramRgb)
            .expect("analyze ok");
        assert_eq!(via_generate.data, via_analyze.data);
    }

    /// Downscaled resolution must produce correct output dimensions.
    #[test]
    fn test_scope_resolution_downscale_correct_dims() {
        let frame = ramp_frame(320, 240);
        let scopes = VideoScopes::new(ScopeConfig {
            width: 512,
            height: 512,
            resolution: Some(ScopeResolution {
                width: 64,
                height: 64,
            }),
            show_graticule: false,
            show_labels: false,
            ..ScopeConfig::default()
        });
        let result = scopes
            .generate_scopes(&frame, 320, 240, ScopeType::HistogramRgb)
            .expect("generate ok");
        assert_eq!(result.width, 64);
        assert_eq!(result.height, 64);
    }
}
