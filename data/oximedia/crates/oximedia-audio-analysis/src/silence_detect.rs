#![allow(dead_code)]
//! Silence detection and segmentation for audio streams.
//!
//! This module provides configurable silence detection with hysteresis
//! thresholds, minimum duration constraints, and segment labelling.
//! It can be used to strip silence from recordings, split audio at
//! silent passages, or detect speech/music activity.

/// Configuration for silence detection.
#[derive(Debug, Clone)]
pub struct SilenceDetectConfig {
    /// Threshold in dBFS below which audio is considered silent.
    pub threshold_dbfs: f64,
    /// Minimum duration of silence to be reported (seconds).
    pub min_silence_duration_s: f64,
    /// Minimum duration of non-silence to break a silence region (seconds).
    pub min_activity_duration_s: f64,
    /// Hysteresis margin in dB above the threshold for the "active" transition.
    pub hysteresis_db: f64,
    /// Pre-roll to keep before a silence region ends (seconds).
    pub pre_roll_s: f64,
    /// Post-roll to keep after a silence region starts (seconds).
    pub post_roll_s: f64,
}

impl Default for SilenceDetectConfig {
    fn default() -> Self {
        Self {
            threshold_dbfs: -50.0,
            min_silence_duration_s: 0.3,
            min_activity_duration_s: 0.05,
            hysteresis_db: 3.0,
            pre_roll_s: 0.0,
            post_roll_s: 0.0,
        }
    }
}

/// A detected region of silence or activity.
#[derive(Debug, Clone, PartialEq)]
pub struct SilenceRegion {
    /// Start time in seconds.
    pub start_s: f64,
    /// End time in seconds.
    pub end_s: f64,
    /// Whether this region is silent (`true`) or active (`false`).
    pub is_silent: bool,
    /// Average RMS level during this region (linear amplitude).
    pub avg_rms: f64,
    /// Peak level during this region (linear amplitude).
    pub peak_level: f64,
}

impl SilenceRegion {
    /// Duration in seconds.
    #[must_use]
    pub fn duration_s(&self) -> f64 {
        self.end_s - self.start_s
    }
}

/// Result of silence detection over a complete audio buffer.
#[derive(Debug, Clone)]
pub struct SilenceDetectResult {
    /// All detected regions (alternating silence / activity).
    pub regions: Vec<SilenceRegion>,
    /// Total silence duration in seconds.
    pub total_silence_s: f64,
    /// Total active duration in seconds.
    pub total_active_s: f64,
    /// Silence ratio (0.0 to 1.0).
    pub silence_ratio: f64,
    /// Number of silent segments.
    pub silence_count: usize,
}

/// Silence detector with hysteresis and duration constraints.
#[derive(Debug, Clone)]
pub struct SilenceDetector {
    /// Detector configuration.
    config: SilenceDetectConfig,
}

impl SilenceDetector {
    /// Create a new silence detector with the given configuration.
    #[must_use]
    pub fn new(config: SilenceDetectConfig) -> Self {
        Self { config }
    }

    /// Create a detector with default settings.
    #[must_use]
    pub fn with_defaults() -> Self {
        Self::new(SilenceDetectConfig::default())
    }

    /// Detect silence regions in mono audio samples at the given sample rate.
    #[allow(clippy::cast_precision_loss)]
    pub fn detect(&self, samples: &[f32], sample_rate: f64) -> SilenceDetectResult {
        if samples.is_empty() || sample_rate <= 0.0 {
            return SilenceDetectResult {
                regions: Vec::new(),
                total_silence_s: 0.0,
                total_active_s: 0.0,
                silence_ratio: 0.0,
                silence_count: 0,
            };
        }

        let frame_size = (0.01 * sample_rate) as usize; // 10 ms frames
        let frame_size = frame_size.max(1);
        let threshold_linear = dbfs_to_linear(self.config.threshold_dbfs);
        let hysteresis_linear =
            dbfs_to_linear(self.config.threshold_dbfs + self.config.hysteresis_db);

        let mut raw_regions: Vec<SilenceRegion> = Vec::new();
        let mut current_silent = true;
        let mut region_start = 0usize;
        let mut rms_accum = 0.0_f64;
        let mut peak = 0.0_f64;
        let mut frame_count_in_region = 0usize;

        let mut pos = 0;
        while pos < samples.len() {
            let end = (pos + frame_size).min(samples.len());
            let frame = &samples[pos..end];
            let rms = compute_rms_f64(frame);
            let frame_peak = frame
                .iter()
                .map(|s| f64::from(*s).abs())
                .fold(0.0_f64, f64::max);

            let is_silent_frame = if current_silent {
                rms < hysteresis_linear
            } else {
                rms < threshold_linear
            };

            if is_silent_frame != current_silent {
                // State transition: emit previous region.
                let start_s = region_start as f64 / sample_rate;
                let end_s = pos as f64 / sample_rate;
                let avg = if frame_count_in_region > 0 {
                    rms_accum / frame_count_in_region as f64
                } else {
                    0.0
                };
                raw_regions.push(SilenceRegion {
                    start_s,
                    end_s,
                    is_silent: current_silent,
                    avg_rms: avg,
                    peak_level: peak,
                });
                current_silent = is_silent_frame;
                region_start = pos;
                rms_accum = 0.0;
                peak = 0.0;
                frame_count_in_region = 0;
            }

            rms_accum += rms;
            if frame_peak > peak {
                peak = frame_peak;
            }
            frame_count_in_region += 1;
            pos = end;
        }

        // Emit final region.
        let start_s = region_start as f64 / sample_rate;
        let end_s = samples.len() as f64 / sample_rate;
        let avg = if frame_count_in_region > 0 {
            rms_accum / frame_count_in_region as f64
        } else {
            0.0
        };
        raw_regions.push(SilenceRegion {
            start_s,
            end_s,
            is_silent: current_silent,
            avg_rms: avg,
            peak_level: peak,
        });

        // Merge short regions that don't meet minimum duration.
        let regions = self.merge_short_regions(&raw_regions);

        let total_silence_s: f64 = regions
            .iter()
            .filter(|r| r.is_silent)
            .map(SilenceRegion::duration_s)
            .sum();
        let total_active_s: f64 = regions
            .iter()
            .filter(|r| !r.is_silent)
            .map(SilenceRegion::duration_s)
            .sum();
        let total = total_silence_s + total_active_s;
        let silence_ratio = if total > 0.0 {
            total_silence_s / total
        } else {
            0.0
        };
        let silence_count = regions.iter().filter(|r| r.is_silent).count();

        SilenceDetectResult {
            regions,
            total_silence_s,
            total_active_s,
            silence_ratio,
            silence_count,
        }
    }

    /// Merge regions shorter than minimum durations into their neighbours.
    fn merge_short_regions(&self, regions: &[SilenceRegion]) -> Vec<SilenceRegion> {
        if regions.is_empty() {
            return Vec::new();
        }

        let mut merged: Vec<SilenceRegion> = Vec::new();
        for region in regions {
            let too_short = if region.is_silent {
                region.duration_s() < self.config.min_silence_duration_s
            } else {
                region.duration_s() < self.config.min_activity_duration_s
            };

            if too_short {
                if let Some(last) = merged.last_mut() {
                    // Extend the previous region to cover this short one.
                    last.end_s = region.end_s;
                    last.peak_level = last.peak_level.max(region.peak_level);
                } else {
                    merged.push(region.clone());
                }
            } else {
                // If same type as previous, merge.
                if let Some(last) = merged.last_mut() {
                    if last.is_silent == region.is_silent {
                        last.end_s = region.end_s;
                        last.peak_level = last.peak_level.max(region.peak_level);
                        continue;
                    }
                }
                merged.push(region.clone());
            }
        }
        merged
    }
}

/// Convert dBFS to linear amplitude.
fn dbfs_to_linear(dbfs: f64) -> f64 {
    10.0_f64.powf(dbfs / 20.0)
}

/// Convert linear amplitude to dBFS.
#[must_use]
pub fn linear_to_dbfs(linear: f64) -> f64 {
    if linear <= 0.0 {
        -100.0
    } else {
        20.0 * linear.log10()
    }
}

/// Compute RMS of a block in f64 precision.
#[allow(clippy::cast_precision_loss)]
fn compute_rms_f64(samples: &[f32]) -> f64 {
    if samples.is_empty() {
        return 0.0;
    }
    let sum: f64 = samples.iter().map(|&s| f64::from(s) * f64::from(s)).sum();
    (sum / samples.len() as f64).sqrt()
}

/// Strip silence from the beginning and end of audio samples.
///
/// Returns a sub-slice of the original samples with leading and trailing
/// silence removed according to the given threshold.
#[allow(clippy::cast_precision_loss)]
#[must_use]
pub fn strip_silence(samples: &[f32], threshold_dbfs: f64) -> &[f32] {
    let threshold_linear = dbfs_to_linear(threshold_dbfs) as f32;
    let start = samples
        .iter()
        .position(|s| s.abs() >= threshold_linear)
        .unwrap_or(0);
    let end = samples
        .iter()
        .rposition(|s| s.abs() >= threshold_linear)
        .map_or(0, |p| p + 1);
    if start >= end {
        &[]
    } else {
        &samples[start..end]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sine_wave(freq: f64, sr: f64, dur: f64, amp: f32) -> Vec<f32> {
        let n = (sr * dur) as usize;
        (0..n)
            .map(|i| {
                let t = i as f64 / sr;
                (amp as f64 * (2.0 * std::f64::consts::PI * freq * t).sin()) as f32
            })
            .collect()
    }

    #[test]
    fn test_default_config() {
        let cfg = SilenceDetectConfig::default();
        assert!((cfg.threshold_dbfs - (-50.0)).abs() < f64::EPSILON);
        assert!((cfg.min_silence_duration_s - 0.3).abs() < f64::EPSILON);
    }

    #[test]
    fn test_empty_input() {
        let det = SilenceDetector::with_defaults();
        let r = det.detect(&[], 44100.0);
        assert!(r.regions.is_empty());
        assert_eq!(r.silence_count, 0);
    }

    #[test]
    fn test_pure_silence() {
        let det = SilenceDetector::with_defaults();
        let samples = vec![0.0f32; 44100];
        let r = det.detect(&samples, 44100.0);
        assert!(r.silence_ratio > 0.99);
        assert!(r.total_active_s < 0.01);
    }

    #[test]
    fn test_pure_tone_no_silence() {
        let config = SilenceDetectConfig {
            threshold_dbfs: -60.0,
            min_silence_duration_s: 0.05,
            min_activity_duration_s: 0.01,
            ..Default::default()
        };
        let det = SilenceDetector::new(config);
        let samples = sine_wave(440.0, 44100.0, 2.0, 0.5);
        let r = det.detect(&samples, 44100.0);
        assert!(
            r.silence_ratio < 0.1,
            "Tone should not be detected as silence"
        );
    }

    #[test]
    fn test_silence_then_tone() {
        let config = SilenceDetectConfig {
            threshold_dbfs: -50.0,
            min_silence_duration_s: 0.1,
            min_activity_duration_s: 0.01,
            hysteresis_db: 3.0,
            ..Default::default()
        };
        let det = SilenceDetector::new(config);
        let mut samples = vec![0.0f32; 44100]; // 1 second silence
        samples.extend(sine_wave(440.0, 44100.0, 1.0, 0.5)); // 1 second tone
        let r = det.detect(&samples, 44100.0);
        assert!(r.silence_count >= 1);
        assert!(r.total_silence_s > 0.5);
        assert!(r.total_active_s > 0.5);
    }

    #[test]
    fn test_strip_silence_both_ends() {
        let mut samples = vec![0.0f32; 1000];
        samples.extend(vec![0.5f32; 500]);
        samples.extend(vec![0.0f32; 1000]);
        let stripped = strip_silence(&samples, -20.0);
        assert_eq!(stripped.len(), 500);
    }

    #[test]
    fn test_strip_silence_all_silent() {
        let samples = vec![0.0f32; 1000];
        let stripped = strip_silence(&samples, -20.0);
        assert!(stripped.is_empty());
    }

    #[test]
    fn test_strip_silence_no_silence() {
        let samples = vec![0.5f32; 1000];
        let stripped = strip_silence(&samples, -20.0);
        assert_eq!(stripped.len(), 1000);
    }

    #[test]
    fn test_dbfs_to_linear_zero() {
        let lin = dbfs_to_linear(0.0);
        assert!((lin - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_dbfs_to_linear_minus6() {
        let lin = dbfs_to_linear(-6.0206);
        assert!((lin - 0.5).abs() < 0.01);
    }

    #[test]
    fn test_linear_to_dbfs_roundtrip() {
        let db = -23.0;
        let lin = dbfs_to_linear(db);
        let back = linear_to_dbfs(lin);
        assert!((db - back).abs() < 1e-6);
    }

    #[test]
    fn test_linear_to_dbfs_zero() {
        assert!((linear_to_dbfs(0.0) - (-100.0)).abs() < f64::EPSILON);
    }

    #[test]
    fn test_silence_region_duration() {
        let r = SilenceRegion {
            start_s: 1.0,
            end_s: 3.5,
            is_silent: true,
            avg_rms: 0.0,
            peak_level: 0.0,
        };
        assert!((r.duration_s() - 2.5).abs() < f64::EPSILON);
    }

    #[test]
    fn test_compute_rms_f64_empty() {
        assert!((compute_rms_f64(&[]) - 0.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_compute_rms_f64_unit() {
        let samples = vec![1.0f32, -1.0, 1.0, -1.0];
        let rms = compute_rms_f64(&samples);
        assert!((rms - 1.0).abs() < 1e-6);
    }
}
