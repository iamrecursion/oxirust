//! Telecine detection for film-to-video transfer analysis.
//!
//! This module provides comprehensive telecine detection algorithms for
//! identifying various pulldown patterns used in converting film (24fps)
//! to video frame rates (29.97fps NTSC, 25fps PAL, etc.).

use crate::error::{CvError, CvResult};
use oximedia_codec::VideoFrame;

use super::field::{FieldAnalyzer, FieldParity};
use super::metrics::TelecineMetrics;
use super::pattern::{
    CadencePattern, FrameDifference, PatternMatcher, PatternValidator, PulldownPattern,
};

/// Configuration for telecine detection.
#[derive(Debug, Clone)]
pub struct TelecineDetectorConfig {
    /// Minimum confidence threshold for detection (0.0-1.0).
    pub confidence_threshold: f64,
    /// Minimum stability threshold for pattern (0.0-1.0).
    pub stability_threshold: f64,
    /// Window size for pattern analysis (number of frames).
    pub window_size: usize,
    /// Threshold for repeated field detection.
    pub repeat_threshold: f64,
    /// Enable mixed content detection.
    pub detect_mixed: bool,
}

impl TelecineDetectorConfig {
    /// Creates a new configuration with default values.
    #[must_use]
    pub const fn new() -> Self {
        Self {
            confidence_threshold: 0.6,
            stability_threshold: 0.7,
            window_size: 30,
            repeat_threshold: 0.1,
            detect_mixed: true,
        }
    }

    /// Creates a configuration optimized for sensitivity.
    #[must_use]
    pub const fn sensitive() -> Self {
        Self {
            confidence_threshold: 0.5,
            stability_threshold: 0.6,
            window_size: 40,
            repeat_threshold: 0.15,
            detect_mixed: true,
        }
    }

    /// Creates a configuration optimized for specificity.
    #[must_use]
    pub const fn specific() -> Self {
        Self {
            confidence_threshold: 0.75,
            stability_threshold: 0.8,
            window_size: 25,
            repeat_threshold: 0.08,
            detect_mixed: false,
        }
    }
}

impl Default for TelecineDetectorConfig {
    fn default() -> Self {
        Self::new()
    }
}

/// Telecine detector for identifying film-to-video transfer.
pub struct TelecineDetector {
    config: TelecineDetectorConfig,
    pattern_matcher: PatternMatcher,
    field_analyzer: FieldAnalyzer,
    validator: PatternValidator,
}

impl TelecineDetector {
    /// Creates a new telecine detector with the given configuration.
    #[must_use]
    pub fn new(config: TelecineDetectorConfig) -> Self {
        let pattern_matcher = PatternMatcher::new(config.window_size);
        let field_analyzer = FieldAnalyzer::new();
        let validator = PatternValidator::new(0.8, config.confidence_threshold);

        Self {
            config,
            pattern_matcher,
            field_analyzer,
            validator,
        }
    }

    /// Creates a telecine detector with default configuration.
    #[must_use]
    pub fn with_defaults() -> Self {
        Self::new(TelecineDetectorConfig::default())
    }

    /// Detects telecine patterns in a sequence of frames.
    ///
    /// Returns information about the detected telecine pattern.
    pub fn detect(&mut self, frames: &[VideoFrame]) -> CvResult<TelecineInfo> {
        if frames.is_empty() {
            return Err(CvError::insufficient_data(1, 0));
        }

        if frames.len() < 5 {
            return Ok(TelecineInfo::none());
        }

        // Calculate frame differences and add to pattern matcher
        self.update_pattern_matcher(frames)?;

        // Detect pattern
        let pattern = self.pattern_matcher.detect_pattern();

        if let Some(ref pat) = pattern {
            // Validate the pattern
            let history: Vec<_> = self.pattern_matcher.history().iter().copied().collect();
            let validation = self.validator.validate(pat, &history);

            if validation.is_valid {
                // Calculate detailed metrics
                let metrics = self.calculate_metrics(frames, pat)?;

                return Ok(TelecineInfo {
                    is_telecine: true,
                    pattern: pat.pattern_type,
                    confidence: pat.confidence,
                    phase: pat.phase,
                    metrics,
                    cadence_map: self.pattern_matcher.generate_cadence_map(),
                    validation_result: Some(validation),
                });
            }
        }

        Ok(TelecineInfo::none())
    }

    /// Updates the pattern matcher with new frame differences.
    fn update_pattern_matcher(&mut self, frames: &[VideoFrame]) -> CvResult<()> {
        if frames.len() < 2 {
            return Ok(());
        }

        for i in 0..frames.len() - 1 {
            let temporal_diff = self.calculate_temporal_difference(&frames[i], &frames[i + 1])?;
            let field_diff = self.calculate_field_difference(&frames[i])?;

            self.pattern_matcher
                .add_frame_difference(FrameDifference::new(temporal_diff, field_diff, i));
        }

        Ok(())
    }

    /// Calculates temporal difference between consecutive frames.
    fn calculate_temporal_difference(
        &self,
        frame1: &VideoFrame,
        frame2: &VideoFrame,
    ) -> CvResult<f64> {
        if frame1.planes.is_empty() || frame2.planes.is_empty() {
            return Ok(0.0);
        }

        if frame1.width != frame2.width || frame1.height != frame2.height {
            return Ok(0.0);
        }

        let width = frame1.width as usize;
        let height = frame1.height as usize;

        let plane1 = &frame1.planes[0];
        let plane2 = &frame2.planes[0];

        let mut diff_sum = 0i64;
        let mut count = 0;

        for y in 0..height {
            let row1 = plane1.row(y);
            let row2 = plane2.row(y);

            if row1.len() < width || row2.len() < width {
                continue;
            }

            for x in 0..width {
                let diff = (i32::from(row1[x]) - i32::from(row2[x])).abs();
                diff_sum += i64::from(diff);
                count += 1;
            }
        }

        if count == 0 {
            return Ok(0.0);
        }

        let avg_diff = diff_sum as f64 / count as f64;
        Ok((avg_diff / 255.0).clamp(0.0, 1.0))
    }

    /// Calculates field difference within a single frame.
    fn calculate_field_difference(&self, frame: &VideoFrame) -> CvResult<f64> {
        let (top, bottom) = self.field_analyzer.separate_fields(frame)?;
        self.field_analyzer
            .calculate_field_difference(&top, &bottom)
    }

    /// Calculates detailed telecine metrics.
    fn calculate_metrics(
        &self,
        frames: &[VideoFrame],
        pattern: &CadencePattern,
    ) -> CvResult<TelecineMetrics> {
        let pattern_confidence = pattern.confidence;
        let cadence_stability = pattern.stability;

        // Calculate frame variance
        let frame_variance = self.calculate_frame_variance(frames)?;

        // Calculate field match quality
        let field_match_quality = self.calculate_field_match_quality(frames, pattern)?;

        Ok(TelecineMetrics::from_components(
            pattern_confidence,
            cadence_stability,
            frame_variance,
            field_match_quality,
        ))
    }

    /// Calculates variance in frame differences (used to detect repeated fields).
    fn calculate_frame_variance(&self, frames: &[VideoFrame]) -> CvResult<f64> {
        if frames.len() < 3 {
            return Ok(0.0);
        }

        let mut diffs = Vec::with_capacity(frames.len() - 1);

        for i in 0..frames.len() - 1 {
            let diff = self.calculate_temporal_difference(&frames[i], &frames[i + 1])?;
            diffs.push(diff);
        }

        if diffs.is_empty() {
            return Ok(0.0);
        }

        let mean: f64 = diffs.iter().sum::<f64>() / diffs.len() as f64;
        let variance: f64 = diffs
            .iter()
            .map(|&d| {
                let diff = d - mean;
                diff * diff
            })
            .sum::<f64>()
            / diffs.len() as f64;

        Ok(variance.sqrt())
    }

    /// Calculates how well fields match the expected pattern.
    fn calculate_field_match_quality(
        &self,
        frames: &[VideoFrame],
        pattern: &CadencePattern,
    ) -> CvResult<f64> {
        if frames.len() < 2 {
            return Ok(0.0);
        }

        // Analyze field parity
        let parities = self.field_analyzer.analyze_field_parity(frames)?;

        let field_pattern = pattern.pattern_type.field_pattern();
        if field_pattern.is_empty() {
            return Ok(0.0);
        }

        let mut match_count = 0;
        let total_count = parities.len();

        for (i, parity) in parities.iter().enumerate() {
            let pattern_idx = (i + pattern.phase) % field_pattern.len();
            let expected_fields = field_pattern[pattern_idx];

            // In 3:2 pulldown, repeated fields should show "different" parity
            let expects_different = expected_fields == 3;
            let is_different = parity.is_different();

            if expects_different == is_different {
                match_count += 1;
            }
        }

        if total_count == 0 {
            return Ok(0.0);
        }

        Ok(match_count as f64 / total_count as f64)
    }

    /// Detects mixed content (combination of film and video).
    pub fn detect_mixed_content(&mut self, frames: &[VideoFrame]) -> CvResult<bool> {
        if !self.config.detect_mixed || frames.len() < 10 {
            return Ok(false);
        }

        // Analyze pattern consistency across the entire sequence
        let cadence_map = self.pattern_matcher.generate_cadence_map();

        if cadence_map.len() < 10 {
            return Ok(false);
        }

        // Look for breaks in the pattern
        let mut pattern_breaks = 0;
        let mut in_pattern = false;

        for entry in &cadence_map {
            if entry.matches_pattern {
                if !in_pattern {
                    pattern_breaks += 1;
                }
                in_pattern = true;
            } else {
                in_pattern = false;
            }
        }

        // If there are multiple pattern breaks, it's likely mixed content
        Ok(pattern_breaks > 2)
    }

    /// Resets the detector state.
    pub fn reset(&mut self) {
        self.pattern_matcher.reset();
    }

    /// Gets the current pattern matcher for analysis.
    #[must_use]
    pub const fn pattern_matcher(&self) -> &PatternMatcher {
        &self.pattern_matcher
    }

    /// Detects the specific type of pulldown being used.
    pub fn detect_pulldown_type(&mut self, frames: &[VideoFrame]) -> CvResult<PulldownPattern> {
        let info = self.detect(frames)?;

        if info.is_telecine && info.confidence > self.config.confidence_threshold {
            Ok(info.pattern)
        } else {
            Ok(PulldownPattern::None)
        }
    }

    /// Generates a recommendation for IVTC (inverse telecine) processing.
    pub fn recommend_ivtc(&mut self, frames: &[VideoFrame]) -> CvResult<IvtcRecommendation> {
        let info = self.detect(frames)?;

        if !info.is_telecine {
            return Ok(IvtcRecommendation {
                should_ivtc: false,
                pattern: PulldownPattern::None,
                confidence: 0.0,
                method: IvtcMethod::None,
            });
        }

        let method = match info.pattern {
            PulldownPattern::Pulldown32 => IvtcMethod::FieldMatch,
            PulldownPattern::Pulldown22 => IvtcMethod::FrameDuplicate,
            PulldownPattern::Pulldown2332 => IvtcMethod::AdvancedFieldMatch,
            PulldownPattern::EuroPulldown => IvtcMethod::SpeedChange,
            PulldownPattern::None => IvtcMethod::None,
        };

        Ok(IvtcRecommendation {
            should_ivtc: info.confidence > self.config.confidence_threshold,
            pattern: info.pattern,
            confidence: info.confidence,
            method,
        })
    }
}

impl Default for TelecineDetector {
    fn default() -> Self {
        Self::with_defaults()
    }
}

/// Information about detected telecine pattern.
#[derive(Debug, Clone)]
pub struct TelecineInfo {
    /// Whether telecine was detected.
    pub is_telecine: bool,
    /// The detected pulldown pattern.
    pub pattern: PulldownPattern,
    /// Detection confidence (0.0-1.0).
    pub confidence: f64,
    /// Phase offset within pattern cycle.
    pub phase: usize,
    /// Detailed metrics.
    pub metrics: TelecineMetrics,
    /// Cadence map showing pattern over time.
    pub cadence_map: Vec<super::pattern::CadenceMapEntry>,
    /// Pattern validation result.
    pub validation_result: Option<super::pattern::PatternValidation>,
}

impl TelecineInfo {
    /// Creates a TelecineInfo indicating no telecine detected.
    #[must_use]
    pub fn none() -> Self {
        Self {
            is_telecine: false,
            pattern: PulldownPattern::None,
            confidence: 0.0,
            phase: 0,
            metrics: TelecineMetrics::new(),
            cadence_map: Vec::new(),
            validation_result: None,
        }
    }

    /// Returns true if telecine was confidently detected.
    #[must_use]
    pub fn is_confident(&self, threshold: f64) -> bool {
        self.is_telecine && self.confidence >= threshold
    }

    /// Gets the film frame rate based on detected pattern.
    #[must_use]
    pub fn film_frame_rate(&self) -> f64 {
        match self.pattern {
            PulldownPattern::Pulldown32
            | PulldownPattern::Pulldown2332
            | PulldownPattern::EuroPulldown => 24.0,
            PulldownPattern::Pulldown22 => 25.0,
            PulldownPattern::None => 0.0,
        }
    }

    /// Gets the video frame rate based on detected pattern.
    #[must_use]
    pub fn video_frame_rate(&self) -> f64 {
        match self.pattern {
            PulldownPattern::Pulldown32 => 29.97,
            PulldownPattern::Pulldown22 => 50.0,
            PulldownPattern::Pulldown2332 => 30.0,
            PulldownPattern::EuroPulldown => 25.0,
            PulldownPattern::None => 0.0,
        }
    }
}

/// Recommendation for IVTC (inverse telecine) processing.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct IvtcRecommendation {
    /// Whether IVTC should be applied.
    pub should_ivtc: bool,
    /// The detected pulldown pattern.
    pub pattern: PulldownPattern,
    /// Confidence in the recommendation.
    pub confidence: f64,
    /// Recommended IVTC method.
    pub method: IvtcMethod,
}

/// IVTC processing method.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum IvtcMethod {
    /// No IVTC needed.
    None,
    /// Field matching for 3:2 pulldown.
    FieldMatch,
    /// Frame duplication removal for 2:2 pulldown.
    FrameDuplicate,
    /// Advanced field matching for 2:3:3:2.
    AdvancedFieldMatch,
    /// Speed change for Euro pulldown.
    SpeedChange,
}

/// Telecine statistics over a longer sequence.
#[derive(Debug, Clone)]
pub struct TelecineStatistics {
    /// Total number of frames analyzed.
    pub total_frames: usize,
    /// Number of frames in telecine pattern.
    pub telecine_frames: usize,
    /// Number of progressive frames.
    pub progressive_frames: usize,
    /// Number of interlaced frames (not telecine).
    pub interlaced_frames: usize,
    /// Overall telecine ratio (0.0-1.0).
    pub telecine_ratio: f64,
    /// Most common pattern detected.
    pub dominant_pattern: PulldownPattern,
    /// Average confidence across sequence.
    pub average_confidence: f64,
}

impl TelecineStatistics {
    /// Creates new statistics with default values.
    #[must_use]
    pub const fn new() -> Self {
        Self {
            total_frames: 0,
            telecine_frames: 0,
            progressive_frames: 0,
            interlaced_frames: 0,
            telecine_ratio: 0.0,
            dominant_pattern: PulldownPattern::None,
            average_confidence: 0.0,
        }
    }

    /// Returns true if the sequence is predominantly telecine.
    #[must_use]
    pub fn is_predominantly_telecine(&self, threshold: f64) -> bool {
        self.telecine_ratio >= threshold
    }
}

impl Default for TelecineStatistics {
    fn default() -> Self {
        Self::new()
    }
}
