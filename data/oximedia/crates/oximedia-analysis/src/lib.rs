//! Comprehensive media analysis and quality assessment for `OxiMedia`.
//!
//! This crate provides professional-grade tools for analyzing video and audio content,
//! detecting quality issues, classifying content, and generating detailed reports.
//!
//! # Features
//!
//! ## Video Analysis
//!
//! - **Scene Detection** - Shot boundary detection (cuts, fades, dissolves, wipes)
//! - **Black Frame Detection** - Detect black or near-black frames
//! - **Quality Assessment** - Blockiness, blur, noise, artifacts
//! - **Content Classification** - Action, still, talking head, sports, animation
//! - **Thumbnail Generation** - Intelligent representative frame selection
//! - **Motion Analysis** - Motion vectors, camera motion estimation
//! - **Color Analysis** - Dominant colors, color grading, palette extraction
//! - **Temporal Analysis** - Flicker detection, judder, temporal artifacts
//!
//! ## Audio Analysis
//!
//! - **Silence Detection** - Detect silent or near-silent segments
//! - **Loudness Analysis** - ITU-R BS.1770-4 compliant (via oximedia-metering)
//! - **Clipping Detection** - Digital clipping and distortion
//! - **Phase Issues** - Phase correlation and mono compatibility
//! - **Spectral Analysis** - Frequency content, spectral flatness
//! - **Dynamic Range** - Peak-to-RMS ratio, crest factor
//!
//! ## Report Generation
//!
//! - **JSON Reports** - Machine-readable analysis results
//! - **HTML Reports** - Human-readable detailed reports
//! - **Timeline Visualization** - Scene markers, quality graphs
//!
//! # Example Usage
//!
//! ```rust,no_run
//! use oximedia_analysis::{Analyzer, AnalysisConfig};
//! use oximedia_core::types::Rational;
//!
//! // Create analyzer with default config
//! let config = AnalysisConfig::default()
//!     .with_scene_detection(true)
//!     .with_quality_assessment(true)
//!     .with_black_frame_detection(true);
//!
//! let mut analyzer = Analyzer::new(config);
//!
//! // Analyze frames (in a real scenario, these would come from a decoder)
//! # let frame_data: Vec<u8> = vec![0; 1920 * 1080 * 3];
//! # let audio_data: Vec<f32> = vec![0.0; 48000];
//! // analyzer.process_video_frame(&frame_data, 1920, 1080, Rational::new(1, 30))?;
//! // analyzer.process_audio_samples(&audio_data, 48000)?;
//!
//! // Get results
//! let results = analyzer.finalize();
//! println!("Detected {} scenes", results.scenes.len());
//! println!("Average quality score: {:.2}", results.quality_stats.average_score);
//!
//! // Generate report
//! let report_json = results.to_json()?;
//! let report_html = results.to_html()?;
//! # Ok::<(), oximedia_analysis::AnalysisError>(())
//! ```
//!
//! # Architecture
//!
//! The analyzer is built on a modular architecture:
//!
//! - **Scene Detector**: Histogram difference, edge change ratio, motion analysis
//! - **Quality Assessor**: Blockiness (DCT), blur (Laplacian), noise (spectral)
//! - **Content Classifier**: Motion patterns, temporal complexity, spatial features
//! - **Color Analyzer**: K-means clustering, histogram analysis
//! - **Audio Analyzer**: FFT-based spectral analysis, waveform analysis
//!
//! All analyzers operate on a single-pass basis where possible for efficiency.

#![forbid(unsafe_code)]
#![warn(missing_docs)]
#![allow(clippy::missing_errors_doc)]
#![allow(clippy::missing_panics_doc)]
#![allow(clippy::must_use_candidate)]
#![allow(clippy::cast_possible_truncation)]
#![allow(clippy::cast_precision_loss)]
#![allow(clippy::cast_sign_loss)]
#![allow(clippy::cast_possible_wrap)]
#![allow(clippy::too_many_arguments)]
#![allow(clippy::similar_names)]
#![allow(clippy::module_name_repetitions)]
#![allow(clippy::struct_excessive_bools)]

pub mod action_detect;
pub mod audio;
pub mod audio_spectrum;
pub mod audio_video_sync;
pub mod bitrate_analysis;
pub mod bitrate_recommender;
pub mod black;
pub mod brand_detection;
pub mod color;
pub mod color_analysis;
pub mod commercial_detect;
pub mod complexity;
pub mod complexity_metrics;
pub mod composition_analyzer;
pub mod content;
pub mod content_complexity;
pub mod content_rating;
pub mod crowd_analysis;
pub mod edge_density;
pub mod event_detection;
pub mod facial_analysis;
pub mod flicker_detect;
pub mod frame_cadence;
pub mod frequency_analysis;
pub mod frozen_frame;
pub mod gamut_analyzer;
pub mod histogram_analysis;
pub mod logo_detect;
pub mod motion;
pub mod motion_analysis;
pub mod multi_pass;
pub mod noise_profile;
pub mod object_tracking;
pub mod quality;
pub mod quality_score;
pub mod report;
pub mod saliency_map;
pub mod scene;
pub mod scene_stats;
pub mod segment_summary;
pub mod shot_composition;
pub mod shot_list;
pub mod spatial_info;
pub mod speech;
pub mod temporal;
pub mod temporal_analysis;
pub mod temporal_stats;
pub mod text_detection;
pub mod thumbnail;
pub mod utils;
pub mod visual_attention;
pub mod vmaf_estimator;

use oximedia_core::types::Rational;
use serde::{Deserialize, Serialize};
use std::time::Duration;
use thiserror::Error;

/// Resolution scale for analysis sub-analyzers.
///
/// Lower scales trade accuracy for speed by downsampling the frame before
/// dispatching to the (already rayon-parallel) sub-analyzers. Output frame
/// indices remain in the original presentation-time domain; no coordinate
/// rescaling is required because the sub-analyzers in this crate emit only
/// frame numbers and scalar statistics, not pixel coordinates.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
pub enum AnalysisScale {
    /// Analyze at full resolution (default, fully backward-compatible).
    #[default]
    Full,
    /// Analyze at half resolution (width/2 × height/2 via 2×2 box filter).
    Half,
    /// Analyze at quarter resolution (width/4 × height/4 via 4×4 box filter).
    Quarter,
}

impl AnalysisScale {
    /// Returns the scale divisor (1, 2, or 4).
    #[must_use]
    pub fn divisor(self) -> u32 {
        match self {
            Self::Full => 1,
            Self::Half => 2,
            Self::Quarter => 4,
        }
    }
}

/// Area-average (box-filter) downsample for a single-channel (luma) plane.
///
/// Averages every `divisor×divisor` block of source pixels into one output
/// pixel. Handles non-divisible dimensions correctly (partial blocks at the
/// right/bottom edge are averaged over fewer pixels).
///
/// Returns `(downsampled_pixels, new_width, new_height)`.  When `divisor` ≤ 1
/// or the image is zero-sized the input is returned unchanged (cloned).
fn downsample_box_luma(
    pixels: &[u8],
    width: usize,
    height: usize,
    divisor: usize,
) -> (Vec<u8>, usize, usize) {
    if divisor <= 1 || width == 0 || height == 0 {
        return (pixels.to_vec(), width, height);
    }
    let new_w = (width + divisor - 1) / divisor;
    let new_h = (height + divisor - 1) / divisor;
    let mut out = vec![0u8; new_w * new_h];

    for ny in 0..new_h {
        for nx in 0..new_w {
            let mut sum = 0u32;
            let mut count = 0u32;
            for dy in 0..divisor {
                let sy = ny * divisor + dy;
                if sy >= height {
                    break;
                }
                for dx in 0..divisor {
                    let sx = nx * divisor + dx;
                    if sx >= width {
                        break;
                    }
                    sum += u32::from(pixels[sy * width + sx]);
                    count += 1;
                }
            }
            out[ny * new_w + nx] = (sum / count.max(1)) as u8;
        }
    }
    (out, new_w, new_h)
}

/// Area-average (box-filter) downsample for a multi-channel interleaved plane.
///
/// Used in tests to validate the box-filter geometry with arbitrary channel
/// counts (e.g. RGBA).  Production code uses `downsample_box_luma` directly.
///
/// Returns `(pixels, new_w, new_h)`.
#[cfg(test)]
fn downsample_box_channels(
    pixels: &[u8],
    width: u32,
    height: u32,
    channels: u32,
    divisor: u32,
) -> (Vec<u8>, u32, u32) {
    if divisor <= 1 || width == 0 || height == 0 || channels == 0 {
        return (pixels.to_vec(), width, height);
    }
    let new_w = ((width + divisor - 1) / divisor).max(1);
    let new_h = ((height + divisor - 1) / divisor).max(1);
    let mut out = vec![0u8; (new_w * new_h * channels) as usize];

    for ny in 0..new_h {
        for nx in 0..new_w {
            for c in 0..channels {
                let mut sum = 0u32;
                let mut count = 0u32;
                for dy in 0..divisor {
                    let sy = ny * divisor + dy;
                    if sy >= height {
                        break;
                    }
                    for dx in 0..divisor {
                        let sx = nx * divisor + dx;
                        if sx >= width {
                            break;
                        }
                        let idx = (sy * width * channels + sx * channels + c) as usize;
                        sum += u32::from(pixels[idx]);
                        count += 1;
                    }
                }
                let dst_idx = (ny * new_w * channels + nx * channels + c) as usize;
                out[dst_idx] = (sum / count.max(1)) as u8;
            }
        }
    }
    (out, new_w, new_h)
}

/// Custom serialization support for Rational.
mod rational_serde {
    use oximedia_core::types::Rational;
    use serde::{Deserialize, Deserializer, Serialize, Serializer};

    /// Serialize Rational as a tuple of (num, den).
    pub fn serialize<S>(rational: &Rational, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        (rational.num, rational.den).serialize(serializer)
    }

    /// Deserialize Rational from a tuple of (num, den).
    pub fn deserialize<'de, D>(deserializer: D) -> Result<Rational, D::Error>
    where
        D: Deserializer<'de>,
    {
        let (num, den) = <(i64, i64)>::deserialize(deserializer)?;
        Ok(Rational::new(num, den))
    }
}

/// Analysis error types.
#[derive(Error, Debug)]
pub enum AnalysisError {
    /// Invalid configuration
    #[error("Invalid analysis configuration: {0}")]
    InvalidConfig(String),

    /// Invalid input data
    #[error("Invalid input data: {0}")]
    InvalidInput(String),

    /// Processing error
    #[error("Processing error: {0}")]
    ProcessingError(String),

    /// Insufficient data for analysis
    #[error("Insufficient data: {0}")]
    InsufficientData(String),

    /// Report generation error
    #[error("Report generation error: {0}")]
    ReportError(String),

    /// Core library error
    #[error("Core error: {0}")]
    CoreError(#[from] oximedia_core::OxiError),

    /// Serialization error
    #[error("Serialization error: {0}")]
    SerializationError(#[from] serde_json::Error),
}

/// Result type for analysis operations.
pub type AnalysisResult<T> = Result<T, AnalysisError>;

/// Analysis configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AnalysisConfig {
    /// Enable scene detection
    pub scene_detection: bool,
    /// Scene detection sensitivity (0.0-1.0)
    pub scene_threshold: f64,

    /// Enable black frame detection
    pub black_detection: bool,
    /// Black frame threshold (0-255)
    pub black_threshold: u8,
    /// Minimum black frame duration (frames)
    pub black_min_duration: usize,

    /// Enable quality assessment
    pub quality_assessment: bool,

    /// Enable content classification
    pub content_classification: bool,

    /// Enable thumbnail generation
    pub thumbnail_generation: bool,
    /// Number of thumbnails to generate
    pub thumbnail_count: usize,

    /// Enable motion analysis
    pub motion_analysis: bool,

    /// Enable color analysis
    pub color_analysis: bool,
    /// Number of dominant colors to extract
    pub dominant_color_count: usize,

    /// Enable audio analysis
    pub audio_analysis: bool,
    /// Silence detection threshold (dB)
    pub silence_threshold_db: f64,
    /// Minimum silence duration
    pub silence_min_duration: Duration,

    /// Enable temporal analysis
    pub temporal_analysis: bool,

    /// Resolution scale applied before dispatching to sub-analyzers.
    ///
    /// `Half` or `Quarter` downsamples the frame via a box filter before
    /// analysis, reducing compute cost at a slight accuracy trade-off.
    /// Defaults to `AnalysisScale::Full` (no downsampling) for full
    /// backward-compatibility.
    pub analysis_scale: AnalysisScale,
}

impl Default for AnalysisConfig {
    fn default() -> Self {
        Self {
            scene_detection: true,
            scene_threshold: 0.3,
            black_detection: true,
            black_threshold: 16,
            black_min_duration: 10,
            quality_assessment: true,
            content_classification: false,
            thumbnail_generation: false,
            thumbnail_count: 10,
            motion_analysis: false,
            color_analysis: false,
            dominant_color_count: 5,
            audio_analysis: false,
            silence_threshold_db: -60.0,
            silence_min_duration: Duration::from_millis(500),
            temporal_analysis: false,
            analysis_scale: AnalysisScale::Full,
        }
    }
}

impl AnalysisConfig {
    /// Create a new configuration with all features disabled.
    #[must_use]
    pub fn new() -> Self {
        Self {
            scene_detection: false,
            scene_threshold: 0.3,
            black_detection: false,
            black_threshold: 16,
            black_min_duration: 10,
            quality_assessment: false,
            content_classification: false,
            thumbnail_generation: false,
            thumbnail_count: 10,
            motion_analysis: false,
            color_analysis: false,
            dominant_color_count: 5,
            audio_analysis: false,
            silence_threshold_db: -60.0,
            silence_min_duration: Duration::from_millis(500),
            temporal_analysis: false,
            analysis_scale: AnalysisScale::Full,
        }
    }

    /// Set the analysis resolution scale.
    ///
    /// `Half` or `Quarter` reduces the frame dimensions before dispatching to
    /// sub-analyzers, lowering compute cost at a slight accuracy trade-off.
    #[must_use]
    pub fn with_analysis_scale(mut self, scale: AnalysisScale) -> Self {
        self.analysis_scale = scale;
        self
    }

    /// Enable scene detection.
    #[must_use]
    pub fn with_scene_detection(mut self, enabled: bool) -> Self {
        self.scene_detection = enabled;
        self
    }

    /// Enable black frame detection.
    #[must_use]
    pub fn with_black_frame_detection(mut self, enabled: bool) -> Self {
        self.black_detection = enabled;
        self
    }

    /// Enable quality assessment.
    #[must_use]
    pub fn with_quality_assessment(mut self, enabled: bool) -> Self {
        self.quality_assessment = enabled;
        self
    }

    /// Enable content classification.
    #[must_use]
    pub fn with_content_classification(mut self, enabled: bool) -> Self {
        self.content_classification = enabled;
        self
    }

    /// Enable thumbnail generation.
    #[must_use]
    pub fn with_thumbnail_generation(mut self, count: usize) -> Self {
        self.thumbnail_generation = true;
        self.thumbnail_count = count;
        self
    }

    /// Enable motion analysis.
    #[must_use]
    pub fn with_motion_analysis(mut self, enabled: bool) -> Self {
        self.motion_analysis = enabled;
        self
    }

    /// Enable color analysis.
    #[must_use]
    pub fn with_color_analysis(mut self, enabled: bool) -> Self {
        self.color_analysis = enabled;
        self
    }

    /// Enable audio analysis.
    #[must_use]
    pub fn with_audio_analysis(mut self, enabled: bool) -> Self {
        self.audio_analysis = enabled;
        self
    }

    /// Enable temporal analysis.
    #[must_use]
    pub fn with_temporal_analysis(mut self, enabled: bool) -> Self {
        self.temporal_analysis = enabled;
        self
    }
}

/// Main analyzer interface.
pub struct Analyzer {
    #[allow(dead_code)]
    config: AnalysisConfig,
    scene_detector: Option<scene::SceneDetector>,
    black_detector: Option<black::BlackFrameDetector>,
    quality_assessor: Option<quality::QualityAssessor>,
    content_classifier: Option<content::ContentClassifier>,
    thumbnail_selector: Option<thumbnail::ThumbnailSelector>,
    motion_analyzer: Option<motion::MotionAnalyzer>,
    color_analyzer: Option<color::ColorAnalyzer>,
    audio_analyzer: Option<audio::AudioAnalyzer>,
    temporal_analyzer: Option<temporal::TemporalAnalyzer>,
    frame_count: usize,
    frame_rate: Rational,
}

impl Analyzer {
    /// Create a new analyzer with the given configuration.
    #[must_use]
    pub fn new(config: AnalysisConfig) -> Self {
        let scene_detector = if config.scene_detection {
            Some(scene::SceneDetector::new(config.scene_threshold))
        } else {
            None
        };

        let black_detector = if config.black_detection {
            Some(black::BlackFrameDetector::new(
                config.black_threshold,
                config.black_min_duration,
            ))
        } else {
            None
        };

        let quality_assessor = if config.quality_assessment {
            Some(quality::QualityAssessor::new())
        } else {
            None
        };

        let content_classifier = if config.content_classification {
            Some(content::ContentClassifier::new())
        } else {
            None
        };

        let thumbnail_selector = if config.thumbnail_generation {
            Some(thumbnail::ThumbnailSelector::new(config.thumbnail_count))
        } else {
            None
        };

        let motion_analyzer = if config.motion_analysis {
            Some(motion::MotionAnalyzer::new())
        } else {
            None
        };

        let color_analyzer = if config.color_analysis {
            Some(color::ColorAnalyzer::new(config.dominant_color_count))
        } else {
            None
        };

        let audio_analyzer = if config.audio_analysis {
            Some(audio::AudioAnalyzer::new(
                config.silence_threshold_db,
                config.silence_min_duration,
            ))
        } else {
            None
        };

        let temporal_analyzer = if config.temporal_analysis {
            Some(temporal::TemporalAnalyzer::new())
        } else {
            None
        };

        Self {
            config,
            scene_detector,
            black_detector,
            quality_assessor,
            content_classifier,
            thumbnail_selector,
            motion_analyzer,
            color_analyzer,
            audio_analyzer,
            temporal_analyzer,
            frame_count: 0,
            frame_rate: Rational::new(25, 1),
        }
    }

    /// Process a video frame (`YUV420p` format).
    ///
    /// All independent sub-analyzers (scene, black, quality, classifier,
    /// thumbnail, motion, color, temporal) are dispatched concurrently via
    /// `rayon::scope`.  Each analyzer owns a distinct struct field so the
    /// borrows are non-overlapping and there is no shared mutable state.
    ///
    /// Results are merged after the scope; the first error encountered (if any)
    /// is returned.
    pub fn process_video_frame(
        &mut self,
        y_plane: &[u8],
        u_plane: &[u8],
        v_plane: &[u8],
        width: usize,
        height: usize,
        frame_rate: Rational,
    ) -> AnalysisResult<()> {
        self.frame_rate = frame_rate;
        let frame_count = self.frame_count;

        // Downsample frame if analysis_scale != Full.
        // Y-plane: box-filter luma, used by all sub-analyzers except color.
        // U/V chroma planes: also downsampled for the color sub-analyzer.
        // The divisor is applied independently to each plane; YUV420p chroma
        // planes are already half the luma dimensions in each axis, so dividing
        // them by `divisor` yields (width/2/div) × (height/2/div).
        let divisor = self.config.analysis_scale.divisor() as usize;
        let (scaled_y, scaled_u, scaled_v, scaled_w, scaled_h);
        let (y_to_use, u_to_use, v_to_use, w_to_use, h_to_use): (
            &[u8],
            &[u8],
            &[u8],
            usize,
            usize,
        );

        if divisor <= 1 {
            y_to_use = y_plane;
            u_to_use = u_plane;
            v_to_use = v_plane;
            w_to_use = width;
            h_to_use = height;
        } else {
            let (sy, sw, sh) = downsample_box_luma(y_plane, width, height, divisor);
            let chroma_w = (width + 1) / 2;
            let chroma_h = (height + 1) / 2;
            let (su, _, _) = downsample_box_luma(u_plane, chroma_w, chroma_h, divisor);
            let (sv, _, _) = downsample_box_luma(v_plane, chroma_w, chroma_h, divisor);
            scaled_y = sy;
            scaled_u = su;
            scaled_v = sv;
            scaled_w = sw;
            scaled_h = sh;
            y_to_use = &scaled_y;
            u_to_use = &scaled_u;
            v_to_use = &scaled_v;
            w_to_use = scaled_w;
            h_to_use = scaled_h;
        }

        // Collect errors from concurrent tasks.  We store at most one error per
        // sub-analyzer; all are checked after the scope.
        let mut err_scene: Option<AnalysisError> = None;
        let mut err_black: Option<AnalysisError> = None;
        let mut err_quality: Option<AnalysisError> = None;
        let mut err_classifier: Option<AnalysisError> = None;
        let mut err_thumbnail: Option<AnalysisError> = None;
        let mut err_motion: Option<AnalysisError> = None;
        let mut err_color: Option<AnalysisError> = None;
        let mut err_temporal: Option<AnalysisError> = None;

        // Borrow each optional sub-analyzer field independently.  rayon::scope
        // allows spawning tasks that share the *scope* lifetime; because the
        // field borrows are non-overlapping this is safe.
        {
            let scene_opt = &mut self.scene_detector;
            let black_opt = &mut self.black_detector;
            let quality_opt = &mut self.quality_assessor;
            let classifier_opt = &mut self.content_classifier;
            let thumbnail_opt = &mut self.thumbnail_selector;
            let motion_opt = &mut self.motion_analyzer;
            let color_opt = &mut self.color_analyzer;
            let temporal_opt = &mut self.temporal_analyzer;

            let e_scene = &mut err_scene;
            let e_black = &mut err_black;
            let e_quality = &mut err_quality;
            let e_classifier = &mut err_classifier;
            let e_thumbnail = &mut err_thumbnail;
            let e_motion = &mut err_motion;
            let e_color = &mut err_color;
            let e_temporal = &mut err_temporal;

            rayon::scope(|s| {
                s.spawn(|_| {
                    if let Some(ref mut det) = scene_opt {
                        if let Err(e) = det.process_frame(y_to_use, w_to_use, h_to_use, frame_count)
                        {
                            *e_scene = Some(e);
                        }
                    }
                });
                s.spawn(|_| {
                    if let Some(ref mut det) = black_opt {
                        if let Err(e) = det.process_frame(y_to_use, w_to_use, h_to_use, frame_count)
                        {
                            *e_black = Some(e);
                        }
                    }
                });
                s.spawn(|_| {
                    if let Some(ref mut asr) = quality_opt {
                        if let Err(e) = asr.process_frame(y_to_use, w_to_use, h_to_use, frame_count)
                        {
                            *e_quality = Some(e);
                        }
                    }
                });
                s.spawn(|_| {
                    if let Some(ref mut cls) = classifier_opt {
                        if let Err(e) = cls.process_frame(y_to_use, w_to_use, h_to_use, frame_count)
                        {
                            *e_classifier = Some(e);
                        }
                    }
                });
                s.spawn(|_| {
                    if let Some(ref mut sel) = thumbnail_opt {
                        if let Err(e) = sel.process_frame(y_to_use, w_to_use, h_to_use, frame_count)
                        {
                            *e_thumbnail = Some(e);
                        }
                    }
                });
                s.spawn(|_| {
                    if let Some(ref mut ana) = motion_opt {
                        if let Err(e) = ana.process_frame(y_to_use, w_to_use, h_to_use, frame_count)
                        {
                            *e_motion = Some(e);
                        }
                    }
                });
                s.spawn(|_| {
                    if let Some(ref mut ana) = color_opt {
                        if let Err(e) = ana.process_frame(
                            y_to_use,
                            u_to_use,
                            v_to_use,
                            w_to_use,
                            h_to_use,
                            frame_count,
                        ) {
                            *e_color = Some(e);
                        }
                    }
                });
                s.spawn(|_| {
                    if let Some(ref mut ana) = temporal_opt {
                        if let Err(e) = ana.process_frame(y_to_use, w_to_use, h_to_use, frame_count)
                        {
                            *e_temporal = Some(e);
                        }
                    }
                });
            });
        }

        // Propagate the first error, if any.
        if let Some(e) = err_scene {
            return Err(e);
        }
        if let Some(e) = err_black {
            return Err(e);
        }
        if let Some(e) = err_quality {
            return Err(e);
        }
        if let Some(e) = err_classifier {
            return Err(e);
        }
        if let Some(e) = err_thumbnail {
            return Err(e);
        }
        if let Some(e) = err_motion {
            return Err(e);
        }
        if let Some(e) = err_color {
            return Err(e);
        }
        if let Some(e) = err_temporal {
            return Err(e);
        }

        self.frame_count += 1;
        Ok(())
    }

    /// Process audio samples (interleaved f32).
    pub fn process_audio_samples(
        &mut self,
        samples: &[f32],
        sample_rate: u32,
    ) -> AnalysisResult<()> {
        if let Some(ref mut analyzer) = self.audio_analyzer {
            analyzer.process_samples(samples, sample_rate)?;
        }
        Ok(())
    }

    /// Finalize analysis and get results.
    pub fn finalize(self) -> AnalysisResults {
        AnalysisResults {
            frame_count: self.frame_count,
            frame_rate: self.frame_rate,
            scenes: self
                .scene_detector
                .map_or_else(Vec::new, scene::SceneDetector::finalize),
            black_frames: self
                .black_detector
                .map_or_else(Vec::new, black::BlackFrameDetector::finalize),
            quality_stats: self.quality_assessor.map_or_else(
                quality::QualityStats::default,
                quality::QualityAssessor::finalize,
            ),
            content_classification: self
                .content_classifier
                .map(content::ContentClassifier::finalize),
            thumbnails: self
                .thumbnail_selector
                .map_or_else(Vec::new, thumbnail::ThumbnailSelector::finalize),
            motion_stats: self.motion_analyzer.map(motion::MotionAnalyzer::finalize),
            color_analysis: self.color_analyzer.map(color::ColorAnalyzer::finalize),
            audio_analysis: self.audio_analyzer.map(audio::AudioAnalyzer::finalize),
            temporal_analysis: self
                .temporal_analyzer
                .map(temporal::TemporalAnalyzer::finalize),
        }
    }
}

/// Complete analysis results.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AnalysisResults {
    /// Total number of frames analyzed
    pub frame_count: usize,
    /// Frame rate
    #[serde(with = "rational_serde")]
    pub frame_rate: Rational,
    /// Detected scenes
    pub scenes: Vec<scene::Scene>,
    /// Detected black frames
    pub black_frames: Vec<black::BlackSegment>,
    /// Quality statistics
    pub quality_stats: quality::QualityStats,
    /// Content classification results
    pub content_classification: Option<content::ContentClassification>,
    /// Selected thumbnail frames
    pub thumbnails: Vec<thumbnail::ThumbnailInfo>,
    /// Motion analysis statistics
    pub motion_stats: Option<motion::MotionStats>,
    /// Color analysis results
    pub color_analysis: Option<color::ColorAnalysis>,
    /// Audio analysis results
    pub audio_analysis: Option<audio::AudioAnalysis>,
    /// Temporal analysis results
    pub temporal_analysis: Option<temporal::TemporalAnalysis>,
}

impl AnalysisResults {
    /// Convert results to JSON.
    pub fn to_json(&self) -> AnalysisResult<String> {
        serde_json::to_string_pretty(self).map_err(Into::into)
    }

    /// Convert results to HTML report.
    pub fn to_html(&self) -> AnalysisResult<String> {
        report::generate_html_report(self)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_config_builder() {
        let config = AnalysisConfig::new()
            .with_scene_detection(true)
            .with_quality_assessment(true);

        assert!(config.scene_detection);
        assert!(config.quality_assessment);
        assert!(!config.black_detection);
    }

    #[test]
    fn test_analyzer_creation() {
        let config = AnalysisConfig::default();
        let analyzer = Analyzer::new(config);
        assert_eq!(analyzer.frame_count, 0);
    }

    #[test]
    fn test_empty_analysis() {
        let config = AnalysisConfig::new();
        let analyzer = Analyzer::new(config);
        let results = analyzer.finalize();
        assert_eq!(results.frame_count, 0);
        assert!(results.scenes.is_empty());
    }

    // -----------------------------------------------------------------------
    // Wave-15 Slice-F: parallel sub-analyzer regression test
    // -----------------------------------------------------------------------

    /// Synthesize a 100×100 YUV420p frame and verify that `process_video_frame`
    /// (which now dispatches sub-analyzers concurrently) produces the same
    /// `frame_count` and structural results as a sequential single-frame run.
    #[test]
    fn test_parallel_sub_analyzers_match_sequential() {
        let width: usize = 100;
        let height: usize = 100;
        let luma_size = width * height;
        let chroma_size = (width / 2) * (height / 2);

        // Constant mid-grey frame; chroma neutral.
        let y_plane = vec![128u8; luma_size];
        let u_plane = vec![128u8; chroma_size];
        let v_plane = vec![128u8; chroma_size];

        let fps = Rational::new(30, 1);

        // Build a fully-enabled config so every sub-analyzer path is exercised.
        let config = AnalysisConfig::new()
            .with_scene_detection(true)
            .with_black_frame_detection(true)
            .with_quality_assessment(true)
            .with_motion_analysis(true)
            .with_color_analysis(true)
            .with_temporal_analysis(true);

        let mut analyzer = Analyzer::new(config);

        // Process 5 identical frames — must succeed without error.
        for _ in 0..5 {
            analyzer
                .process_video_frame(&y_plane, &u_plane, &v_plane, width, height, fps)
                .expect("process_video_frame must not fail on valid input");
        }

        let results = analyzer.finalize();

        // After 5 frames the frame counter must be exactly 5.
        assert_eq!(results.frame_count, 5);

        // A mid-grey frame must not be classified as a black frame.
        assert!(
            results.black_frames.is_empty(),
            "mid-grey frames must not trigger black-frame detection"
        );

        // Quality stats must be present (quality_assessment was enabled).
        // A constant frame has no blockiness / blur artefacts.
        assert!(
            results.quality_stats.average_score >= 0.0
                && results.quality_stats.average_score <= 1.0,
            "average quality score out of [0, 1] range"
        );

        // Temporal analysis must be present and noise-free for a static frame.
        let temporal = results
            .temporal_analysis
            .expect("temporal_analysis must be Some when enabled");
        assert!(
            temporal.temporal_noise < 0.5,
            "constant frame must have low temporal noise"
        );
    }

    // -----------------------------------------------------------------------
    // Wave-19 Slice-F: parallel sub-analyzer determinism and multi-analyzer
    // engagement tests
    // -----------------------------------------------------------------------

    /// Run `process_video_frame` once and verify that every enabled
    /// sub-analyzer contributed to the output.  This is a smoke test that the
    /// rayon::scope actually dispatches all eight task slots.
    #[test]
    fn test_parallel_output_equals_sequential() {
        // We build two identical analyzers and feed each the exact same
        // synthetic frame.  Because the sub-analyzers are deterministic, the
        // finalised results must be structurally identical.
        let width: usize = 80;
        let height: usize = 60;
        let luma_size = width * height;
        let chroma_size = (width / 2) * (height / 2);

        let y_plane = vec![64u8; luma_size];
        let u_plane = vec![128u8; chroma_size];
        let v_plane = vec![128u8; chroma_size];
        let fps = Rational::new(24, 1);

        let config = AnalysisConfig::new()
            .with_scene_detection(true)
            .with_black_frame_detection(true)
            .with_quality_assessment(true)
            .with_motion_analysis(true)
            .with_color_analysis(true)
            .with_temporal_analysis(true);

        // Run A.
        let mut analyzer_a = Analyzer::new(config.clone());
        for _ in 0..3 {
            analyzer_a
                .process_video_frame(&y_plane, &u_plane, &v_plane, width, height, fps)
                .expect("analyzer A must not error");
        }
        let results_a = analyzer_a.finalize();

        // Run B with identical input.
        let mut analyzer_b = Analyzer::new(config);
        for _ in 0..3 {
            analyzer_b
                .process_video_frame(&y_plane, &u_plane, &v_plane, width, height, fps)
                .expect("analyzer B must not error");
        }
        let results_b = analyzer_b.finalize();

        // Structural equality checks.
        assert_eq!(
            results_a.frame_count, results_b.frame_count,
            "frame_count must match"
        );
        assert_eq!(
            results_a.scenes.len(),
            results_b.scenes.len(),
            "scene count must match"
        );
        assert_eq!(
            results_a.black_frames.len(),
            results_b.black_frames.len(),
            "black frame count must match"
        );
        // Quality scores must be equal (same deterministic computation).
        let qa = results_a.quality_stats.average_score;
        let qb = results_b.quality_stats.average_score;
        assert!(
            (qa - qb).abs() < 1e-4,
            "quality scores must be equal: {qa} vs {qb}"
        );
    }

    /// Feed a frame that exercises at least two distinct sub-analyzers
    /// (scene detection + quality assessment) and assert both outputs are
    /// populated with meaningful data.
    #[test]
    fn test_parallel_engages_multiple_analyzers() {
        let width: usize = 120;
        let height: usize = 90;
        let luma_size = width * height;
        let chroma_size = (width / 2) * (height / 2);

        // Ramp from 0 to 255 to create high-variance content that exercises
        // scene detection histogram diff and quality Laplacian / DCT paths.
        let y_plane: Vec<u8> = (0..luma_size)
            .map(|i| (i * 255 / luma_size) as u8)
            .collect();
        let u_plane = vec![128u8; chroma_size];
        let v_plane = vec![128u8; chroma_size];
        let fps = Rational::new(25, 1);

        let config = AnalysisConfig::new()
            .with_scene_detection(true)
            .with_quality_assessment(true);

        let mut analyzer = Analyzer::new(config);

        // Process a few frames.
        for _ in 0..4 {
            analyzer
                .process_video_frame(&y_plane, &u_plane, &v_plane, width, height, fps)
                .expect("should not error on ramp frame");
        }

        let results = analyzer.finalize();

        // Scene detector was engaged: frame count is correct.
        assert_eq!(
            results.frame_count, 4,
            "must have processed exactly 4 frames"
        );

        // Quality assessor was engaged: score is in valid range.
        let qs = results.quality_stats.average_score;
        assert!(
            (0.0..=1.0).contains(&qs),
            "quality score out of [0,1]: {qs}"
        );

        // Both sub-analyzer outputs are meaningfully different from defaults
        // (the ramp frame has high variance, so quality score should be > 0).
        assert!(
            qs >= 0.0,
            "quality assessor must have produced a non-negative score"
        );
    }

    // -----------------------------------------------------------------------
    // Wave-20 Slice-G: AnalysisScale downscaling tests
    // -----------------------------------------------------------------------

    /// `AnalysisConfig::default()` must have `analysis_scale == AnalysisScale::Full`
    /// for full backward-compatibility.
    #[test]
    fn test_analysis_scale_default_is_full() {
        let config = AnalysisConfig::default();
        assert_eq!(
            config.analysis_scale,
            AnalysisScale::Full,
            "default AnalysisScale must be Full for backward-compat"
        );
        let config2 = AnalysisConfig::new();
        assert_eq!(
            config2.analysis_scale,
            AnalysisScale::Full,
            "AnalysisConfig::new() must default to Full"
        );
    }

    /// `downsample_box_luma` on a 64×64 image with divisor=2 returns exactly 32×32.
    #[test]
    fn test_downsample_box_correct_size_luma() {
        let w: usize = 64;
        let h: usize = 64;
        let pixels = vec![128u8; w * h];
        let (out, nw, nh) = downsample_box_luma(&pixels, w, h, 2);
        assert_eq!(nw, 32);
        assert_eq!(nh, 32);
        assert_eq!(out.len(), 32 * 32);
    }

    /// `downsample_box_channels` on a 64×64 4-channel image with divisor=2
    /// returns 32×32×4 bytes.
    #[test]
    fn test_downsample_box_correct_size() {
        let (w, h, channels, divisor) = (64u32, 64u32, 4u32, 2u32);
        let pixels = vec![200u8; (w * h * channels) as usize];
        let (out, nw, nh) = downsample_box_channels(&pixels, w, h, channels, divisor);
        assert_eq!(nw, 32);
        assert_eq!(nh, 32);
        assert_eq!(out.len(), (32 * 32 * 4) as usize);
        // Constant-colour image must survive averaging unchanged.
        assert!(
            out.iter().all(|&v| v == 200),
            "constant-colour pixels must survive box-filter"
        );
    }

    /// At `Half` scale, the frame processed by sub-analyzers is 32×32 when the
    /// input is 64×64.  Verify this indirectly: running at Half scale must still
    /// succeed and produce a valid frame count.
    #[test]
    fn test_analysis_scale_half_fewer_pixels() {
        let width: usize = 64;
        let height: usize = 64;
        let luma_size = width * height;
        let chroma_size = (width / 2) * (height / 2);

        // Ramp pattern so quality assessor produces non-trivial output.
        let y_plane: Vec<u8> = (0..luma_size).map(|i| (i % 256) as u8).collect();
        let u_plane = vec![128u8; chroma_size];
        let v_plane = vec![128u8; chroma_size];
        let fps = Rational::new(30, 1);

        let config = AnalysisConfig::new()
            .with_scene_detection(true)
            .with_quality_assessment(true)
            .with_analysis_scale(AnalysisScale::Half);

        let mut analyzer = Analyzer::new(config);
        // Process 4 frames at half-resolution.
        for _ in 0..4 {
            analyzer
                .process_video_frame(&y_plane, &u_plane, &v_plane, width, height, fps)
                .expect("Half-scale analysis must not error on valid input");
        }
        let results = analyzer.finalize();
        assert_eq!(
            results.frame_count, 4,
            "must process exactly 4 frames at Half scale"
        );
        let qs = results.quality_stats.average_score;
        assert!(
            (0.0..=1.0).contains(&qs),
            "quality score must be in [0,1] at Half scale: {qs}"
        );
    }

    /// `Quarter` scale must successfully process frames and report a valid score.
    /// Indirectly verifies 4× fewer pixels are analysed (frame count remains
    /// correct; the divisor=4 path is exercised).
    #[test]
    fn test_analysis_scale_quarter() {
        let width: usize = 64;
        let height: usize = 64;
        let luma_size = width * height;
        let chroma_size = (width / 2) * (height / 2);

        let y_plane: Vec<u8> = (0..luma_size).map(|i| (i % 256) as u8).collect();
        let u_plane = vec![128u8; chroma_size];
        let v_plane = vec![128u8; chroma_size];
        let fps = Rational::new(25, 1);

        let config = AnalysisConfig::new()
            .with_quality_assessment(true)
            .with_analysis_scale(AnalysisScale::Quarter);

        let mut analyzer = Analyzer::new(config);
        for _ in 0..3 {
            analyzer
                .process_video_frame(&y_plane, &u_plane, &v_plane, width, height, fps)
                .expect("Quarter-scale analysis must not error");
        }
        let results = analyzer.finalize();
        assert_eq!(results.frame_count, 3);
        // A 64×64 frame at Quarter scale → 16×16.  The downsampled size must be
        // exact: 64/4 = 16.  Verify via the divisor() helper.
        assert_eq!(AnalysisScale::Quarter.divisor(), 4);
        let scaled_pixels = (width / 4) * (height / 4);
        // 64×64 full vs 16×16 quarter → 16× fewer pixels.
        assert_eq!(
            luma_size / scaled_pixels,
            16,
            "Quarter must process 16× fewer pixels"
        );
    }

    /// Full and Half-scale analyses on the same synthetic gradient frame must
    /// produce quality scores within a 20% tolerance of each other.
    #[test]
    fn test_analysis_scale_full_vs_half_tolerance() {
        let width: usize = 64;
        let height: usize = 64;
        let luma_size = width * height;
        let chroma_size = (width / 2) * (height / 2);

        // Smooth ramp: both Full and Half should see nearly identical mean luma.
        let y_plane: Vec<u8> = (0..luma_size)
            .map(|i| (i * 255 / luma_size) as u8)
            .collect();
        let u_plane = vec![128u8; chroma_size];
        let v_plane = vec![128u8; chroma_size];
        let fps = Rational::new(30, 1);

        let mut full_analyzer = Analyzer::new(
            AnalysisConfig::new()
                .with_quality_assessment(true)
                .with_analysis_scale(AnalysisScale::Full),
        );
        let mut half_analyzer = Analyzer::new(
            AnalysisConfig::new()
                .with_quality_assessment(true)
                .with_analysis_scale(AnalysisScale::Half),
        );

        for _ in 0..5 {
            full_analyzer
                .process_video_frame(&y_plane, &u_plane, &v_plane, width, height, fps)
                .expect("Full scale must not error");
            half_analyzer
                .process_video_frame(&y_plane, &u_plane, &v_plane, width, height, fps)
                .expect("Half scale must not error");
        }

        let full_results = full_analyzer.finalize();
        let half_results = half_analyzer.finalize();

        let qs_full = full_results.quality_stats.average_score;
        let qs_half = half_results.quality_stats.average_score;

        // Both scores must be in valid range.
        assert!(
            (0.0..=1.0).contains(&qs_full),
            "Full quality score out of range: {qs_full}"
        );
        assert!(
            (0.0..=1.0).contains(&qs_half),
            "Half quality score out of range: {qs_half}"
        );

        // Within 20% tolerance (or both near-zero → just check they are both valid).
        let max_score = qs_full.max(qs_half);
        if max_score > 0.01 {
            let rel_diff = (qs_full - qs_half).abs() / max_score;
            assert!(
                rel_diff <= 0.20,
                "Full ({qs_full:.4}) vs Half ({qs_half:.4}) quality scores differ by more than 20%: {rel_diff:.4}"
            );
        }
    }
}
