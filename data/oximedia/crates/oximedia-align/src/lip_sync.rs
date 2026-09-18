//! Lip sync alignment for audio/video synchronization.
//!
//! Provides A/V offset detection, automatic correction, and tolerance checking
//! to ensure lips match audio in video content.

#![allow(dead_code)]
#![allow(clippy::cast_precision_loss)]
#![allow(clippy::too_many_arguments)]

use serde::{Deserialize, Serialize};

/// Standard lip sync tolerance window (ITU-R BT.1359)
pub const ITU_TOLERANCE_MS: f64 = 45.0; // ±45ms

/// Comfortable viewer tolerance (wider than ITU)
pub const COMFORTABLE_TOLERANCE_MS: f64 = 90.0;

/// A/V offset measurement between audio and video
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct AvOffset {
    /// Offset in milliseconds (positive = audio ahead, negative = video ahead)
    pub offset_ms: f64,
    /// Confidence in the measurement (0.0 to 1.0)
    pub confidence: f64,
    /// Detection method used
    pub method: DetectionMethod,
}

impl AvOffset {
    /// Create a new A/V offset
    #[must_use]
    pub fn new(offset_ms: f64, confidence: f64, method: DetectionMethod) -> Self {
        Self {
            offset_ms,
            confidence,
            method,
        }
    }

    /// Convert offset to samples at a given sample rate
    #[must_use]
    pub fn to_samples(&self, sample_rate: u32) -> i64 {
        (self.offset_ms * f64::from(sample_rate) / 1000.0).round() as i64
    }

    /// Convert offset to frames at a given frame rate
    #[must_use]
    pub fn to_frames(&self, fps: f64) -> f64 {
        self.offset_ms * fps / 1000.0
    }

    /// Is the offset within the ITU tolerance window?
    #[must_use]
    pub fn within_itu_tolerance(&self) -> bool {
        self.offset_ms.abs() <= ITU_TOLERANCE_MS
    }

    /// Is the offset within comfortable viewer tolerance?
    #[must_use]
    pub fn within_comfortable_tolerance(&self) -> bool {
        self.offset_ms.abs() <= COMFORTABLE_TOLERANCE_MS
    }
}

/// Method used to detect A/V offset
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum DetectionMethod {
    /// Audio cross-correlation with video motion
    AudioMotionCorrelation,
    /// Speech onset detection
    SpeechOnset,
    /// Visual mouth movement analysis
    MouthMovement,
    /// Clapper board detection
    ClapperBoard,
    /// Manual annotation
    Manual,
    /// Hybrid multi-method
    Hybrid,
}

/// Lip sync detector configuration
#[derive(Debug, Clone)]
pub struct LipSyncConfig {
    /// Analysis window in milliseconds
    pub window_ms: f64,
    /// Search range in milliseconds
    pub search_range_ms: f64,
    /// Minimum confidence to accept a detection
    pub min_confidence: f64,
    /// Sample rate of the audio
    pub sample_rate: u32,
    /// Frames per second of the video
    pub fps: f64,
}

impl Default for LipSyncConfig {
    fn default() -> Self {
        Self {
            window_ms: 500.0,
            search_range_ms: 500.0,
            min_confidence: 0.6,
            sample_rate: 48000,
            fps: 25.0,
        }
    }
}

impl LipSyncConfig {
    /// Create a new config with custom parameters
    #[must_use]
    pub fn new(window_ms: f64, search_range_ms: f64, sample_rate: u32, fps: f64) -> Self {
        Self {
            window_ms,
            search_range_ms,
            min_confidence: 0.6,
            sample_rate,
            fps,
        }
    }

    /// Convert window size to samples
    #[must_use]
    pub fn window_samples(&self) -> usize {
        (self.window_ms * f64::from(self.sample_rate) / 1000.0) as usize
    }

    /// Convert search range to samples
    #[must_use]
    pub fn search_range_samples(&self) -> usize {
        (self.search_range_ms * f64::from(self.sample_rate) / 1000.0) as usize
    }
}

/// Lip sync correction to be applied
#[derive(Debug, Clone, Copy)]
pub struct LipSyncCorrection {
    /// Delay to apply to audio (positive = delay, negative = advance)
    pub audio_delay_ms: f64,
    /// Delay to apply to video (positive = delay, negative = advance)
    pub video_delay_ms: f64,
    /// Whether correction is needed at all
    pub needs_correction: bool,
}

impl LipSyncCorrection {
    /// Create correction from an A/V offset
    /// Positive `offset_ms` means audio is ahead, so delay audio
    #[must_use]
    pub fn from_offset(offset: &AvOffset, tolerance_ms: f64) -> Self {
        if offset.offset_ms.abs() <= tolerance_ms {
            return Self {
                audio_delay_ms: 0.0,
                video_delay_ms: 0.0,
                needs_correction: false,
            };
        }

        // Prefer delaying one stream rather than advancing the other
        if offset.offset_ms > 0.0 {
            // Audio is ahead: delay audio by the offset amount
            Self {
                audio_delay_ms: offset.offset_ms,
                video_delay_ms: 0.0,
                needs_correction: true,
            }
        } else {
            // Video is ahead: delay video
            Self {
                audio_delay_ms: 0.0,
                video_delay_ms: -offset.offset_ms,
                needs_correction: true,
            }
        }
    }

    /// Total correction magnitude in ms
    #[must_use]
    pub fn magnitude_ms(&self) -> f64 {
        self.audio_delay_ms + self.video_delay_ms
    }
}

/// Lip sync analyzer
#[derive(Debug, Clone)]
pub struct LipSyncAnalyzer {
    config: LipSyncConfig,
    /// History of detected offsets
    offset_history: Vec<AvOffset>,
}

impl LipSyncAnalyzer {
    /// Create a new analyzer
    #[must_use]
    pub fn new(config: LipSyncConfig) -> Self {
        Self {
            config,
            offset_history: Vec::new(),
        }
    }

    /// Detect offset using cross-correlation of audio envelope and video activity
    pub fn detect_offset_from_envelopes(
        &mut self,
        audio_envelope: &[f32],
        video_activity: &[f32],
    ) -> Option<AvOffset> {
        if audio_envelope.is_empty() || video_activity.is_empty() {
            return None;
        }

        let max_lag = self
            .config
            .search_range_samples()
            .min(audio_envelope.len() / 2);
        let window = self.config.window_samples().min(audio_envelope.len());

        let mut best_lag = 0i64;
        let mut best_corr = f64::NEG_INFINITY;

        for lag in -(max_lag as i64)..=(max_lag as i64) {
            let corr = cross_correlate_at_lag(audio_envelope, video_activity, lag, window);
            if corr > best_corr {
                best_corr = corr;
                best_lag = lag;
            }
        }

        // Normalize correlation
        let audio_power: f64 = audio_envelope
            .iter()
            .map(|&x| f64::from(x) * f64::from(x))
            .sum::<f64>()
            / audio_envelope.len() as f64;
        let video_power: f64 = video_activity
            .iter()
            .map(|&x| f64::from(x) * f64::from(x))
            .sum::<f64>()
            / video_activity.len() as f64;

        let max_possible = (audio_power * video_power).sqrt() * window as f64;
        let confidence = if max_possible > 0.0 {
            (best_corr / max_possible).clamp(0.0, 1.0)
        } else {
            0.0
        };

        let offset_ms = best_lag as f64 / f64::from(self.config.sample_rate) * 1000.0;
        let offset = AvOffset::new(
            offset_ms,
            confidence,
            DetectionMethod::AudioMotionCorrelation,
        );

        if confidence >= self.config.min_confidence {
            self.offset_history.push(offset);
        }

        Some(offset)
    }

    /// Get the median offset from history (more robust than latest)
    #[must_use]
    pub fn median_offset(&self) -> Option<f64> {
        if self.offset_history.is_empty() {
            return None;
        }
        let mut offsets: Vec<f64> = self.offset_history.iter().map(|o| o.offset_ms).collect();
        offsets.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let mid = offsets.len() / 2;
        Some(if offsets.len() % 2 == 0 {
            (offsets[mid - 1] + offsets[mid]) / 2.0
        } else {
            offsets[mid]
        })
    }

    /// Get recommended correction based on history
    #[must_use]
    pub fn recommend_correction(&self, tolerance_ms: f64) -> Option<LipSyncCorrection> {
        let median = self.median_offset()?;
        let offset = AvOffset::new(median, 1.0, DetectionMethod::Hybrid);
        Some(LipSyncCorrection::from_offset(&offset, tolerance_ms))
    }

    /// Clear the offset history
    pub fn clear_history(&mut self) {
        self.offset_history.clear();
    }

    /// Number of measurements in history
    #[must_use]
    pub fn history_len(&self) -> usize {
        self.offset_history.len()
    }
}

/// Compute cross-correlation between two signals at a given lag
fn cross_correlate_at_lag(a: &[f32], b: &[f32], lag: i64, window: usize) -> f64 {
    let mut sum = 0.0_f64;
    let n = window.min(a.len()).min(b.len());
    for i in 0..n {
        let j = i as i64 + lag;
        if j >= 0 && (j as usize) < b.len() {
            sum += f64::from(a[i]) * f64::from(b[j as usize]);
        }
    }
    sum
}

/// Tolerance checker for lip sync
#[derive(Debug, Clone, Copy)]
pub struct ToleranceChecker {
    /// ITU-R BT.1359 tolerance in ms
    pub itu_tolerance_ms: f64,
    /// Custom tolerance in ms
    pub custom_tolerance_ms: f64,
}

impl ToleranceChecker {
    /// Create a new tolerance checker
    #[must_use]
    pub fn new(custom_tolerance_ms: f64) -> Self {
        Self {
            itu_tolerance_ms: ITU_TOLERANCE_MS,
            custom_tolerance_ms,
        }
    }

    /// Check if an offset passes ITU tolerance
    #[must_use]
    pub fn passes_itu(&self, offset_ms: f64) -> bool {
        offset_ms.abs() <= self.itu_tolerance_ms
    }

    /// Check if an offset passes custom tolerance
    #[must_use]
    pub fn passes_custom(&self, offset_ms: f64) -> bool {
        offset_ms.abs() <= self.custom_tolerance_ms
    }

    /// Rate the severity of the offset
    #[must_use]
    pub fn severity(&self, offset_ms: f64) -> SyncSeverity {
        let abs_ms = offset_ms.abs();
        if abs_ms <= ITU_TOLERANCE_MS {
            SyncSeverity::None
        } else if abs_ms <= COMFORTABLE_TOLERANCE_MS {
            SyncSeverity::Minor
        } else if abs_ms <= 200.0 {
            SyncSeverity::Moderate
        } else {
            SyncSeverity::Severe
        }
    }
}

impl Default for ToleranceChecker {
    fn default() -> Self {
        Self::new(ITU_TOLERANCE_MS)
    }
}

/// Severity of lip sync error
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SyncSeverity {
    /// Within tolerance, not noticeable
    None,
    /// Slightly outside tolerance, barely noticeable
    Minor,
    /// Clearly noticeable lip sync error
    Moderate,
    /// Severe lip sync error, very distracting
    Severe,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_av_offset_creation() {
        let offset = AvOffset::new(20.0, 0.9, DetectionMethod::Manual);
        assert!((offset.offset_ms - 20.0).abs() < f64::EPSILON);
        assert!((offset.confidence - 0.9).abs() < f64::EPSILON);
        assert_eq!(offset.method, DetectionMethod::Manual);
    }

    #[test]
    fn test_av_offset_to_samples() {
        let offset = AvOffset::new(100.0, 0.9, DetectionMethod::Manual);
        assert_eq!(offset.to_samples(48000), 4800);
    }

    #[test]
    fn test_av_offset_to_frames() {
        let offset = AvOffset::new(40.0, 0.9, DetectionMethod::Manual);
        let frames = offset.to_frames(25.0);
        assert!((frames - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_av_offset_itu_tolerance() {
        let within = AvOffset::new(40.0, 0.9, DetectionMethod::Manual);
        assert!(within.within_itu_tolerance());

        let outside = AvOffset::new(50.0, 0.9, DetectionMethod::Manual);
        assert!(!outside.within_itu_tolerance());
    }

    #[test]
    fn test_av_offset_comfortable_tolerance() {
        let within = AvOffset::new(80.0, 0.9, DetectionMethod::Manual);
        assert!(within.within_comfortable_tolerance());

        let outside = AvOffset::new(100.0, 0.9, DetectionMethod::Manual);
        assert!(!outside.within_comfortable_tolerance());
    }

    #[test]
    fn test_lip_sync_config_default() {
        let config = LipSyncConfig::default();
        assert_eq!(config.sample_rate, 48000);
        assert!((config.fps - 25.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_lip_sync_config_window_samples() {
        let config = LipSyncConfig::default(); // window_ms=500, sample_rate=48000
        assert_eq!(config.window_samples(), 24000);
    }

    #[test]
    fn test_lip_sync_correction_no_correction_needed() {
        let offset = AvOffset::new(10.0, 0.9, DetectionMethod::Manual);
        let correction = LipSyncCorrection::from_offset(&offset, 45.0);
        assert!(!correction.needs_correction);
    }

    #[test]
    fn test_lip_sync_correction_audio_ahead() {
        let offset = AvOffset::new(100.0, 0.9, DetectionMethod::Manual);
        let correction = LipSyncCorrection::from_offset(&offset, 45.0);
        assert!(correction.needs_correction);
        assert!(correction.audio_delay_ms > 0.0);
        assert_eq!(correction.video_delay_ms, 0.0);
    }

    #[test]
    fn test_lip_sync_correction_video_ahead() {
        let offset = AvOffset::new(-100.0, 0.9, DetectionMethod::Manual);
        let correction = LipSyncCorrection::from_offset(&offset, 45.0);
        assert!(correction.needs_correction);
        assert_eq!(correction.audio_delay_ms, 0.0);
        assert!(correction.video_delay_ms > 0.0);
    }

    #[test]
    fn test_lip_sync_correction_magnitude() {
        let offset = AvOffset::new(100.0, 0.9, DetectionMethod::Manual);
        let correction = LipSyncCorrection::from_offset(&offset, 45.0);
        assert!((correction.magnitude_ms() - 100.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_analyzer_detect_from_envelopes() {
        let config = LipSyncConfig::new(100.0, 200.0, 48000, 25.0);
        let mut analyzer = LipSyncAnalyzer::new(config);

        // Create simple test signals
        let n = 5000;
        let mut audio = vec![0.0f32; n];
        let mut video = vec![0.0f32; n];

        // Place a transient at position 1000 in audio and 1100 in video
        audio[1000] = 1.0;
        audio[1001] = 0.8;
        video[1100] = 1.0;
        video[1101] = 0.8;

        let result = analyzer.detect_offset_from_envelopes(&audio, &video);
        assert!(result.is_some());
    }

    #[test]
    fn test_analyzer_median_offset_empty() {
        let analyzer = LipSyncAnalyzer::new(LipSyncConfig::default());
        assert!(analyzer.median_offset().is_none());
    }

    #[test]
    fn test_analyzer_clear_history() {
        let config = LipSyncConfig::default();
        let mut analyzer = LipSyncAnalyzer::new(config);
        // Add manual entry to history
        analyzer
            .offset_history
            .push(AvOffset::new(10.0, 0.9, DetectionMethod::Manual));
        assert_eq!(analyzer.history_len(), 1);
        analyzer.clear_history();
        assert_eq!(analyzer.history_len(), 0);
    }

    #[test]
    fn test_tolerance_checker_itu() {
        let checker = ToleranceChecker::default();
        assert!(checker.passes_itu(44.9));
        assert!(!checker.passes_itu(45.1));
    }

    #[test]
    fn test_tolerance_checker_severity() {
        let checker = ToleranceChecker::default();
        assert_eq!(checker.severity(30.0), SyncSeverity::None);
        assert_eq!(checker.severity(70.0), SyncSeverity::Minor);
        assert_eq!(checker.severity(150.0), SyncSeverity::Moderate);
        assert_eq!(checker.severity(250.0), SyncSeverity::Severe);
    }

    #[test]
    fn test_cross_correlate_at_lag() {
        let a = vec![1.0f32, 0.0, 0.0, 0.0, 0.0];
        let b = vec![0.0f32, 1.0, 0.0, 0.0, 0.0];
        // At lag=1, a[0] aligns with b[1], should give 1.0
        let corr = cross_correlate_at_lag(&a, &b, 1, 5);
        assert!((corr - 1.0).abs() < 1e-6);
    }
}
