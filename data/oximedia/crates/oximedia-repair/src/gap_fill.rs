#![allow(dead_code)]
//! Gap filling and interpolation for corrupted or missing media segments.
//!
//! This module provides algorithms for detecting gaps in media streams and
//! filling them with interpolated data to maintain continuity. Supports
//! audio sample interpolation, video frame duplication/blending, and
//! adaptive fill strategies based on surrounding content analysis.

/// Strategy used for filling detected gaps.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FillStrategy {
    /// Insert silence (audio) or black frames (video).
    Silence,
    /// Repeat the last valid sample or frame.
    Repeat,
    /// Linearly interpolate between boundaries.
    LinearInterpolation,
    /// Use cubic spline interpolation for smoother results.
    CubicInterpolation,
    /// Cross-fade between boundary samples.
    Crossfade,
    /// Use spectral analysis to generate fill material.
    SpectralSynthesis,
}

impl Default for FillStrategy {
    fn default() -> Self {
        Self::LinearInterpolation
    }
}

/// Represents a detected gap in a media stream.
#[derive(Debug, Clone)]
pub struct Gap {
    /// Start position of the gap in samples or frames.
    pub start: u64,
    /// End position of the gap in samples or frames.
    pub end: u64,
    /// Duration of the gap in the same unit as start/end.
    pub duration: u64,
    /// Confidence that this is truly a gap (0.0 - 1.0).
    pub confidence: f64,
    /// Type of gap detected.
    pub gap_type: GapType,
}

impl Gap {
    /// Create a new gap descriptor.
    pub fn new(start: u64, end: u64, gap_type: GapType) -> Self {
        let duration = end.saturating_sub(start);
        Self {
            start,
            end,
            duration,
            confidence: 1.0,
            gap_type,
        }
    }

    /// Check whether the gap is within acceptable fill limits.
    pub fn is_fillable(&self, max_duration: u64) -> bool {
        self.duration <= max_duration && self.confidence > 0.5
    }
}

/// Classification of gap types.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GapType {
    /// Missing audio samples (digital silence or dropout).
    AudioDropout,
    /// Missing video frames.
    VideoFrameDrop,
    /// Corrupted data region that should be replaced.
    CorruptedRegion,
    /// Intentional gap from editing (splice point).
    SplicePoint,
    /// Transmission loss in streaming context.
    TransmissionLoss,
}

/// Configuration for the gap-fill engine.
#[derive(Debug, Clone)]
pub struct GapFillConfig {
    /// Strategy to use for filling gaps.
    pub strategy: FillStrategy,
    /// Maximum gap duration (in samples) to attempt filling.
    pub max_gap_samples: u64,
    /// Crossfade length in samples at gap boundaries.
    pub crossfade_len: usize,
    /// Minimum gap duration to consider (ignore very short gaps).
    pub min_gap_samples: u64,
    /// Whether to analyze surrounding content for adaptive fill.
    pub adaptive: bool,
    /// Sample rate of the audio stream (Hz).
    pub sample_rate: u32,
}

impl Default for GapFillConfig {
    fn default() -> Self {
        Self {
            strategy: FillStrategy::LinearInterpolation,
            max_gap_samples: 48000, // 1 second at 48 kHz
            crossfade_len: 256,
            min_gap_samples: 1,
            adaptive: false,
            sample_rate: 48000,
        }
    }
}

/// Result of a gap fill operation.
#[derive(Debug, Clone)]
pub struct FillResult {
    /// Number of gaps detected.
    pub gaps_detected: usize,
    /// Number of gaps successfully filled.
    pub gaps_filled: usize,
    /// Number of gaps that were too large to fill.
    pub gaps_skipped: usize,
    /// Total samples generated for fill.
    pub samples_generated: u64,
    /// Quality estimate of the fill (0.0 - 1.0).
    pub quality_estimate: f64,
}

/// Engine for detecting and filling gaps in audio streams.
#[derive(Debug)]
pub struct GapFillEngine {
    /// Configuration for gap filling.
    config: GapFillConfig,
    /// Detected gaps awaiting fill.
    pending_gaps: Vec<Gap>,
}

impl GapFillEngine {
    /// Create a new gap fill engine with the given configuration.
    pub fn new(config: GapFillConfig) -> Self {
        Self {
            config,
            pending_gaps: Vec::new(),
        }
    }

    /// Create a gap fill engine with default settings.
    pub fn default_engine() -> Self {
        Self::new(GapFillConfig::default())
    }

    /// Detect gaps in an audio buffer by scanning for silence or discontinuities.
    #[allow(clippy::cast_precision_loss)]
    pub fn detect_gaps(&mut self, samples: &[f32], threshold: f32) -> Vec<Gap> {
        let mut gaps = Vec::new();
        let mut gap_start: Option<usize> = None;

        for (i, &sample) in samples.iter().enumerate() {
            let is_silent = sample.abs() < threshold;
            match (is_silent, gap_start) {
                (true, None) => {
                    gap_start = Some(i);
                }
                (false, Some(start)) => {
                    let duration = i - start;
                    if duration as u64 >= self.config.min_gap_samples {
                        gaps.push(Gap::new(start as u64, i as u64, GapType::AudioDropout));
                    }
                    gap_start = None;
                }
                _ => {}
            }
        }

        // Handle trailing gap
        if let Some(start) = gap_start {
            let duration = samples.len() - start;
            if duration as u64 >= self.config.min_gap_samples {
                gaps.push(Gap::new(
                    start as u64,
                    samples.len() as u64,
                    GapType::AudioDropout,
                ));
            }
        }

        self.pending_gaps = gaps.clone();
        gaps
    }

    /// Fill detected gaps in the audio buffer using the configured strategy.
    #[allow(clippy::cast_precision_loss)]
    pub fn fill_gaps(&self, samples: &mut [f32], gaps: &[Gap]) -> FillResult {
        let mut filled = 0usize;
        let mut skipped = 0usize;
        let mut total_generated = 0u64;

        for gap in gaps {
            if !gap.is_fillable(self.config.max_gap_samples) {
                skipped += 1;
                continue;
            }

            let start = gap.start as usize;
            let end = gap.end as usize;
            if end > samples.len() || start >= end {
                skipped += 1;
                continue;
            }

            match self.config.strategy {
                FillStrategy::Silence => {
                    for s in &mut samples[start..end] {
                        *s = 0.0;
                    }
                }
                FillStrategy::Repeat => {
                    let fill_val = if start > 0 { samples[start - 1] } else { 0.0 };
                    for s in &mut samples[start..end] {
                        *s = fill_val;
                    }
                }
                FillStrategy::LinearInterpolation => {
                    let left = if start > 0 { samples[start - 1] } else { 0.0 };
                    let right = if end < samples.len() {
                        samples[end]
                    } else {
                        0.0
                    };
                    let len = (end - start) as f32;
                    for (i, s) in samples[start..end].iter_mut().enumerate() {
                        let t = (i as f32 + 1.0) / (len + 1.0);
                        *s = left + (right - left) * t;
                    }
                }
                FillStrategy::CubicInterpolation => {
                    let p0 = if start >= 2 { samples[start - 2] } else { 0.0 };
                    let p1 = if start > 0 { samples[start - 1] } else { 0.0 };
                    let p2 = if end < samples.len() {
                        samples[end]
                    } else {
                        0.0
                    };
                    let p3 = if end + 1 < samples.len() {
                        samples[end + 1]
                    } else {
                        p2
                    };
                    let len = (end - start) as f32;
                    for (i, s) in samples[start..end].iter_mut().enumerate() {
                        let t = (i as f32 + 1.0) / (len + 1.0);
                        *s = cubic_interpolate(p0, p1, p2, p3, t);
                    }
                }
                FillStrategy::Crossfade => {
                    let left = if start > 0 { samples[start - 1] } else { 0.0 };
                    let right = if end < samples.len() {
                        samples[end]
                    } else {
                        0.0
                    };
                    let len = (end - start) as f32;
                    for (i, s) in samples[start..end].iter_mut().enumerate() {
                        let t = i as f32 / len;
                        // Equal-power crossfade
                        let angle = t * std::f32::consts::FRAC_PI_2;
                        *s = left * angle.cos() + right * angle.sin();
                    }
                }
                FillStrategy::SpectralSynthesis => {
                    // Simplified: use linear interpolation as fallback
                    let left = if start > 0 { samples[start - 1] } else { 0.0 };
                    let right = if end < samples.len() {
                        samples[end]
                    } else {
                        0.0
                    };
                    let len = (end - start) as f32;
                    for (i, s) in samples[start..end].iter_mut().enumerate() {
                        let t = (i as f32 + 1.0) / (len + 1.0);
                        *s = left + (right - left) * t;
                    }
                }
            }

            total_generated += gap.duration;
            filled += 1;
        }

        let quality = if filled > 0 {
            1.0 - (skipped as f64 / (filled + skipped) as f64)
        } else if gaps.is_empty() {
            1.0
        } else {
            0.0
        };

        FillResult {
            gaps_detected: gaps.len(),
            gaps_filled: filled,
            gaps_skipped: skipped,
            samples_generated: total_generated,
            quality_estimate: quality,
        }
    }

    /// Get pending gaps that were detected but not yet filled.
    pub fn pending_gaps(&self) -> &[Gap] {
        &self.pending_gaps
    }

    /// Clear pending gaps.
    pub fn clear_pending(&mut self) {
        self.pending_gaps.clear();
    }

    /// Update the fill strategy.
    pub fn set_strategy(&mut self, strategy: FillStrategy) {
        self.config.strategy = strategy;
    }
}

/// Cubic Hermite interpolation between four control points.
#[allow(clippy::cast_precision_loss)]
fn cubic_interpolate(p0: f32, p1: f32, p2: f32, p3: f32, t: f32) -> f32 {
    let a = -0.5 * p0 + 1.5 * p1 - 1.5 * p2 + 0.5 * p3;
    let b = p0 - 2.5 * p1 + 2.0 * p2 - 0.5 * p3;
    let c = -0.5 * p0 + 0.5 * p2;
    let d = p1;
    ((a * t + b) * t + c) * t + d
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gap_new() {
        let gap = Gap::new(100, 200, GapType::AudioDropout);
        assert_eq!(gap.start, 100);
        assert_eq!(gap.end, 200);
        assert_eq!(gap.duration, 100);
        assert_eq!(gap.confidence, 1.0);
    }

    #[test]
    fn test_gap_is_fillable() {
        let gap = Gap::new(0, 100, GapType::AudioDropout);
        assert!(gap.is_fillable(200));
        assert!(gap.is_fillable(100));
        assert!(!gap.is_fillable(50));
    }

    #[test]
    fn test_gap_low_confidence_not_fillable() {
        let mut gap = Gap::new(0, 10, GapType::CorruptedRegion);
        gap.confidence = 0.3;
        assert!(!gap.is_fillable(1000));
    }

    #[test]
    fn test_fill_strategy_default() {
        assert_eq!(FillStrategy::default(), FillStrategy::LinearInterpolation);
    }

    #[test]
    fn test_gap_fill_config_default() {
        let cfg = GapFillConfig::default();
        assert_eq!(cfg.max_gap_samples, 48000);
        assert_eq!(cfg.crossfade_len, 256);
        assert_eq!(cfg.sample_rate, 48000);
    }

    #[test]
    fn test_detect_gaps_empty() {
        let mut engine = GapFillEngine::default_engine();
        let samples = [0.5f32; 100];
        let gaps = engine.detect_gaps(&samples, 0.01);
        assert!(gaps.is_empty());
    }

    #[test]
    fn test_detect_gaps_single() {
        let mut engine = GapFillEngine::default_engine();
        let mut samples = vec![0.5f32; 100];
        for s in &mut samples[30..50] {
            *s = 0.0;
        }
        let gaps = engine.detect_gaps(&samples, 0.01);
        assert_eq!(gaps.len(), 1);
        assert_eq!(gaps[0].start, 30);
        assert_eq!(gaps[0].end, 50);
    }

    #[test]
    fn test_fill_silence() {
        let engine = GapFillEngine::new(GapFillConfig {
            strategy: FillStrategy::Silence,
            ..Default::default()
        });
        let mut samples = vec![1.0f32; 100];
        let gaps = vec![Gap::new(10, 20, GapType::AudioDropout)];
        let result = engine.fill_gaps(&mut samples, &gaps);
        assert_eq!(result.gaps_filled, 1);
        for &s in &samples[10..20] {
            assert_eq!(s, 0.0);
        }
    }

    #[test]
    fn test_fill_repeat() {
        let engine = GapFillEngine::new(GapFillConfig {
            strategy: FillStrategy::Repeat,
            ..Default::default()
        });
        let mut samples = vec![0.0f32; 100];
        samples[9] = 0.75;
        let gaps = vec![Gap::new(10, 15, GapType::AudioDropout)];
        engine.fill_gaps(&mut samples, &gaps);
        for &s in &samples[10..15] {
            assert!((s - 0.75).abs() < 1e-6);
        }
    }

    #[test]
    fn test_fill_linear_interpolation() {
        let engine = GapFillEngine::new(GapFillConfig {
            strategy: FillStrategy::LinearInterpolation,
            ..Default::default()
        });
        let mut samples = vec![0.0f32; 20];
        samples[4] = 0.0;
        samples[10] = 1.0;
        let gaps = vec![Gap::new(5, 10, GapType::AudioDropout)];
        engine.fill_gaps(&mut samples, &gaps);
        // Values should smoothly increase from 0 toward 1
        for i in 5..10 {
            assert!(samples[i] >= -0.1 && samples[i] <= 1.1);
        }
    }

    #[test]
    fn test_fill_crossfade() {
        let engine = GapFillEngine::new(GapFillConfig {
            strategy: FillStrategy::Crossfade,
            ..Default::default()
        });
        let mut samples = vec![0.0f32; 30];
        samples[9] = 1.0;
        samples[20] = 0.5;
        let gaps = vec![Gap::new(10, 20, GapType::AudioDropout)];
        let result = engine.fill_gaps(&mut samples, &gaps);
        assert_eq!(result.gaps_filled, 1);
    }

    #[test]
    fn test_fill_skips_large_gaps() {
        let engine = GapFillEngine::new(GapFillConfig {
            max_gap_samples: 10,
            ..Default::default()
        });
        let mut samples = vec![0.0f32; 100];
        let gaps = vec![Gap::new(0, 50, GapType::AudioDropout)];
        let result = engine.fill_gaps(&mut samples, &gaps);
        assert_eq!(result.gaps_skipped, 1);
        assert_eq!(result.gaps_filled, 0);
    }

    #[test]
    fn test_cubic_interpolate() {
        let v = cubic_interpolate(0.0, 0.0, 1.0, 1.0, 0.5);
        assert!(v > 0.0 && v < 1.0, "Expected intermediate value, got {v}");
    }

    #[test]
    fn test_fill_result_quality() {
        let result = FillResult {
            gaps_detected: 4,
            gaps_filled: 3,
            gaps_skipped: 1,
            samples_generated: 300,
            quality_estimate: 0.75,
        };
        assert_eq!(result.gaps_detected, 4);
        assert!((result.quality_estimate - 0.75).abs() < 1e-6);
    }

    #[test]
    fn test_engine_set_strategy() {
        let mut engine = GapFillEngine::default_engine();
        engine.set_strategy(FillStrategy::CubicInterpolation);
        // Verify internal state changed by exercising fill
        let mut samples = vec![0.0f32; 20];
        samples[4] = 1.0;
        samples[10] = 0.0;
        let gaps = vec![Gap::new(5, 10, GapType::AudioDropout)];
        let result = engine.fill_gaps(&mut samples, &gaps);
        assert_eq!(result.gaps_filled, 1);
    }

    #[test]
    fn test_pending_gaps_lifecycle() {
        let mut engine = GapFillEngine::default_engine();
        assert!(engine.pending_gaps().is_empty());

        let mut samples = vec![0.0f32; 50];
        for s in &mut samples[10..20] {
            *s = 0.0;
        }
        // Set non-zero boundary so gap is detected
        for s in samples.iter_mut().take(10) {
            *s = 0.5;
        }
        for s in samples.iter_mut().skip(20) {
            *s = 0.5;
        }
        engine.detect_gaps(&samples, 0.01);
        assert!(!engine.pending_gaps().is_empty());
        engine.clear_pending();
        assert!(engine.pending_gaps().is_empty());
    }
}
