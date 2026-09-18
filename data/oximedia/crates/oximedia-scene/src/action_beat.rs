#![allow(dead_code)]
//! Action beat detection and analysis for video scenes.
//!
//! An *action beat* is a discrete unit of dramatic or physical action within a
//! scene — a punch, a door opening, a character reaction, a camera cut timed
//! to music.  This module quantifies beats by analysing motion energy, audio
//! transients, and cut timing, making it useful for:
//!
//! - Pacing analysis (beats per minute of a scene)
//! - Highlight detection in sports / action footage
//! - Edit-point suggestion aligned with dramatic beats
//! - Synchronisation of cuts to music beats

use std::fmt;

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

/// Category of an action beat.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum BeatCategory {
    /// High-motion physical action (fight, chase, explosion).
    PhysicalAction,
    /// Dialogue emphasis or dramatic reaction.
    DramaticReaction,
    /// Camera movement beat (whip pan, zoom, crane).
    CameraMove,
    /// Edit cut synchronised to audio.
    RhythmicCut,
    /// Audio transient (loud hit, crash, music accent).
    AudioTransient,
    /// Generic / unclassified beat.
    Generic,
}

impl fmt::Display for BeatCategory {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::PhysicalAction => write!(f, "PhysicalAction"),
            Self::DramaticReaction => write!(f, "DramaticReaction"),
            Self::CameraMove => write!(f, "CameraMove"),
            Self::RhythmicCut => write!(f, "RhythmicCut"),
            Self::AudioTransient => write!(f, "AudioTransient"),
            Self::Generic => write!(f, "Generic"),
        }
    }
}

/// A detected action beat.
#[derive(Debug, Clone)]
pub struct ActionBeat {
    /// Timestamp in seconds.
    pub timestamp: f64,
    /// Duration of the beat in seconds.
    pub duration: f64,
    /// Intensity / energy of the beat (0..1).
    pub intensity: f64,
    /// Category of the beat.
    pub category: BeatCategory,
    /// Confidence of the detection (0..1).
    pub confidence: f64,
}

/// Aggregate statistics about beats in a segment.
#[derive(Debug, Clone)]
pub struct BeatStats {
    /// Total number of beats.
    pub count: usize,
    /// Average beats per minute.
    pub bpm: f64,
    /// Average beat intensity.
    pub avg_intensity: f64,
    /// Maximum beat intensity.
    pub max_intensity: f64,
    /// Standard deviation of inter-beat intervals.
    pub interval_std_dev: f64,
    /// Dominant beat category.
    pub dominant_category: BeatCategory,
}

/// Configuration for the beat detector.
#[derive(Debug, Clone)]
pub struct BeatDetectorConfig {
    /// Minimum motion energy to register a beat.
    pub motion_threshold: f64,
    /// Minimum audio energy to register a transient.
    pub audio_threshold: f64,
    /// Minimum time between consecutive beats (seconds).
    pub min_interval: f64,
    /// Look-ahead window (seconds) for peak picking.
    pub window_size: f64,
    /// Whether to merge nearby beats.
    pub merge_nearby: bool,
    /// Merge distance in seconds.
    pub merge_distance: f64,
}

/// A time-stamped energy sample used as detector input.
#[derive(Debug, Clone, Copy)]
pub struct EnergySample {
    /// Timestamp in seconds.
    pub time: f64,
    /// Motion energy (0..1).
    pub motion: f64,
    /// Audio energy (0..1).
    pub audio: f64,
}

/// Pacing profile derived from beat analysis.
#[derive(Debug, Clone)]
pub struct PacingProfile {
    /// Time segments and their BPM values.
    pub segments: Vec<PacingSegment>,
    /// Overall average BPM.
    pub overall_bpm: f64,
    /// Pacing trend over the scene (-1 = decelerating, 0 = steady, 1 = accelerating).
    pub trend: f64,
}

/// A single pacing segment.
#[derive(Debug, Clone)]
pub struct PacingSegment {
    /// Start time in seconds.
    pub start: f64,
    /// End time in seconds.
    pub end: f64,
    /// Beats per minute in this segment.
    pub bpm: f64,
}

// ---------------------------------------------------------------------------
// Implementations
// ---------------------------------------------------------------------------

impl Default for BeatDetectorConfig {
    fn default() -> Self {
        Self {
            motion_threshold: 0.25,
            audio_threshold: 0.3,
            min_interval: 0.15,
            window_size: 0.5,
            merge_nearby: true,
            merge_distance: 0.1,
        }
    }
}

/// Detect action beats from energy samples.
pub fn detect_beats(samples: &[EnergySample], config: &BeatDetectorConfig) -> Vec<ActionBeat> {
    if samples.is_empty() {
        return Vec::new();
    }

    let mut beats: Vec<ActionBeat> = Vec::new();

    for (i, s) in samples.iter().enumerate() {
        let combined = s.motion * 0.6 + s.audio * 0.4;
        let is_peak = combined > config.motion_threshold.min(config.audio_threshold);

        // Simple local-maximum check
        if is_peak {
            let prev_ok = if i > 0 {
                let pc = samples[i - 1].motion * 0.6 + samples[i - 1].audio * 0.4;
                combined >= pc
            } else {
                true
            };
            let next_ok = if i + 1 < samples.len() {
                let nc = samples[i + 1].motion * 0.6 + samples[i + 1].audio * 0.4;
                combined >= nc
            } else {
                true
            };
            if prev_ok && next_ok {
                // Enforce minimum interval
                let too_close = beats
                    .last()
                    .is_some_and(|b: &ActionBeat| s.time - b.timestamp < config.min_interval);
                if !too_close {
                    let category = classify_beat(s);
                    beats.push(ActionBeat {
                        timestamp: s.time,
                        duration: config.window_size,
                        intensity: combined.min(1.0),
                        category,
                        confidence: combined.min(1.0),
                    });
                }
            }
        }
    }

    if config.merge_nearby {
        merge_beats(&mut beats, config.merge_distance);
    }

    beats
}

/// Classify a beat based on motion vs audio dominance.
fn classify_beat(sample: &EnergySample) -> BeatCategory {
    if sample.motion > 0.7 {
        BeatCategory::PhysicalAction
    } else if sample.audio > 0.7 {
        BeatCategory::AudioTransient
    } else if sample.motion > 0.4 && sample.audio > 0.4 {
        BeatCategory::RhythmicCut
    } else if sample.motion > sample.audio {
        BeatCategory::CameraMove
    } else {
        BeatCategory::Generic
    }
}

/// Merge beats that are closer than `distance` seconds.
fn merge_beats(beats: &mut Vec<ActionBeat>, distance: f64) {
    if beats.len() < 2 {
        return;
    }
    let mut merged = vec![beats[0].clone()];
    for b in beats.iter().skip(1) {
        if let Some(last) = merged.last_mut() {
            if b.timestamp - last.timestamp < distance {
                last.intensity = last.intensity.max(b.intensity);
                last.duration = b.timestamp - last.timestamp + b.duration;
                continue;
            }
        }
        merged.push(b.clone());
    }
    *beats = merged;
}

/// Compute aggregate beat statistics.
#[allow(clippy::cast_precision_loss)]
pub fn compute_beat_stats(beats: &[ActionBeat], duration_secs: f64) -> BeatStats {
    if beats.is_empty() || duration_secs <= 0.0 {
        return BeatStats {
            count: 0,
            bpm: 0.0,
            avg_intensity: 0.0,
            max_intensity: 0.0,
            interval_std_dev: 0.0,
            dominant_category: BeatCategory::Generic,
        };
    }

    let count = beats.len();
    let bpm = count as f64 / duration_secs * 60.0;
    let avg_intensity = beats.iter().map(|b| b.intensity).sum::<f64>() / count as f64;
    let max_intensity = beats.iter().map(|b| b.intensity).fold(0.0f64, f64::max);

    // Inter-beat interval statistics
    let intervals: Vec<f64> = beats
        .windows(2)
        .map(|w| w[1].timestamp - w[0].timestamp)
        .collect();
    let interval_std_dev = if intervals.len() > 1 {
        let mean_iv = intervals.iter().sum::<f64>() / intervals.len() as f64;
        let var =
            intervals.iter().map(|v| (v - mean_iv).powi(2)).sum::<f64>() / intervals.len() as f64;
        var.sqrt()
    } else {
        0.0
    };

    // Dominant category by count
    let mut counts = [0usize; 6];
    for b in beats {
        let idx = match b.category {
            BeatCategory::PhysicalAction => 0,
            BeatCategory::DramaticReaction => 1,
            BeatCategory::CameraMove => 2,
            BeatCategory::RhythmicCut => 3,
            BeatCategory::AudioTransient => 4,
            BeatCategory::Generic => 5,
        };
        counts[idx] += 1;
    }
    let dominant_idx = counts
        .iter()
        .enumerate()
        .max_by_key(|(_, &c)| c)
        .map_or(5, |(i, _)| i);
    let dominant_category = match dominant_idx {
        0 => BeatCategory::PhysicalAction,
        1 => BeatCategory::DramaticReaction,
        2 => BeatCategory::CameraMove,
        3 => BeatCategory::RhythmicCut,
        4 => BeatCategory::AudioTransient,
        _ => BeatCategory::Generic,
    };

    BeatStats {
        count,
        bpm,
        avg_intensity,
        max_intensity,
        interval_std_dev,
        dominant_category,
    }
}

/// Compute a pacing profile by windowing the beats.
#[allow(clippy::cast_precision_loss)]
pub fn compute_pacing(
    beats: &[ActionBeat],
    total_duration: f64,
    window_secs: f64,
) -> PacingProfile {
    if beats.is_empty() || total_duration <= 0.0 || window_secs <= 0.0 {
        return PacingProfile {
            segments: Vec::new(),
            overall_bpm: 0.0,
            trend: 0.0,
        };
    }

    let n_windows = (total_duration / window_secs).ceil() as usize;
    let mut segments = Vec::with_capacity(n_windows);

    for w in 0..n_windows {
        let start = w as f64 * window_secs;
        let end = (start + window_secs).min(total_duration);
        let count = beats
            .iter()
            .filter(|b| b.timestamp >= start && b.timestamp < end)
            .count();
        let dur = end - start;
        let bpm = if dur > 0.0 {
            count as f64 / dur * 60.0
        } else {
            0.0
        };
        segments.push(PacingSegment { start, end, bpm });
    }

    let overall_bpm = beats.len() as f64 / total_duration * 60.0;

    // Trend: simple linear regression slope of BPM over segment index
    let trend = if segments.len() >= 2 {
        let n = segments.len() as f64;
        let sum_x: f64 = (0..segments.len()).map(|i| i as f64).sum();
        let sum_y: f64 = segments.iter().map(|s| s.bpm).sum();
        let sum_xy: f64 = segments
            .iter()
            .enumerate()
            .map(|(i, s)| i as f64 * s.bpm)
            .sum();
        let sum_xx: f64 = (0..segments.len()).map(|i| (i as f64).powi(2)).sum();
        let denom = n * sum_xx - sum_x * sum_x;
        if denom.abs() > 1e-12 {
            let slope = (n * sum_xy - sum_x * sum_y) / denom;
            slope.clamp(-1.0, 1.0)
        } else {
            0.0
        }
    } else {
        0.0
    };

    PacingProfile {
        segments,
        overall_bpm,
        trend,
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn sample(time: f64, motion: f64, audio: f64) -> EnergySample {
        EnergySample {
            time,
            motion,
            audio,
        }
    }

    #[test]
    fn test_detect_beats_empty() {
        let beats = detect_beats(&[], &BeatDetectorConfig::default());
        assert!(beats.is_empty());
    }

    #[test]
    fn test_detect_beats_single_peak() {
        let samples = vec![
            sample(0.0, 0.0, 0.0),
            sample(0.5, 0.8, 0.6),
            sample(1.0, 0.0, 0.0),
        ];
        let beats = detect_beats(&samples, &BeatDetectorConfig::default());
        assert_eq!(beats.len(), 1);
        assert!((beats[0].timestamp - 0.5).abs() < 1e-9);
    }

    #[test]
    fn test_detect_beats_multiple() {
        let samples = vec![
            sample(0.0, 0.0, 0.0),
            sample(1.0, 0.9, 0.5),
            sample(2.0, 0.1, 0.1),
            sample(3.0, 0.5, 0.8),
            sample(4.0, 0.0, 0.0),
        ];
        let beats = detect_beats(&samples, &BeatDetectorConfig::default());
        assert!(beats.len() >= 2);
    }

    #[test]
    fn test_detect_beats_min_interval() {
        let samples = vec![
            sample(0.0, 0.5, 0.5),
            sample(0.05, 0.6, 0.6),
            sample(0.1, 0.5, 0.5),
        ];
        let cfg = BeatDetectorConfig {
            min_interval: 0.2,
            ..BeatDetectorConfig::default()
        };
        let beats = detect_beats(&samples, &cfg);
        assert!(beats.len() <= 1);
    }

    #[test]
    fn test_classify_physical_action() {
        let s = sample(0.0, 0.9, 0.1);
        assert_eq!(classify_beat(&s), BeatCategory::PhysicalAction);
    }

    #[test]
    fn test_classify_audio_transient() {
        let s = sample(0.0, 0.1, 0.9);
        assert_eq!(classify_beat(&s), BeatCategory::AudioTransient);
    }

    #[test]
    fn test_classify_rhythmic_cut() {
        let s = sample(0.0, 0.5, 0.5);
        assert_eq!(classify_beat(&s), BeatCategory::RhythmicCut);
    }

    #[test]
    fn test_compute_beat_stats_empty() {
        let stats = compute_beat_stats(&[], 10.0);
        assert_eq!(stats.count, 0);
        assert_eq!(stats.bpm, 0.0);
    }

    #[test]
    fn test_compute_beat_stats_values() {
        let beats = vec![
            ActionBeat {
                timestamp: 1.0,
                duration: 0.1,
                intensity: 0.8,
                category: BeatCategory::PhysicalAction,
                confidence: 0.9,
            },
            ActionBeat {
                timestamp: 3.0,
                duration: 0.1,
                intensity: 0.6,
                category: BeatCategory::PhysicalAction,
                confidence: 0.7,
            },
            ActionBeat {
                timestamp: 5.0,
                duration: 0.1,
                intensity: 0.9,
                category: BeatCategory::AudioTransient,
                confidence: 0.8,
            },
        ];
        let stats = compute_beat_stats(&beats, 6.0);
        assert_eq!(stats.count, 3);
        assert!((stats.bpm - 30.0).abs() < 1e-9);
        assert!((stats.max_intensity - 0.9).abs() < 1e-9);
        assert_eq!(stats.dominant_category, BeatCategory::PhysicalAction);
    }

    #[test]
    fn test_pacing_profile_empty() {
        let p = compute_pacing(&[], 10.0, 5.0);
        assert!(p.segments.is_empty());
        assert_eq!(p.overall_bpm, 0.0);
    }

    #[test]
    fn test_pacing_profile_segments() {
        let beats = vec![
            ActionBeat {
                timestamp: 1.0,
                duration: 0.1,
                intensity: 0.5,
                category: BeatCategory::Generic,
                confidence: 0.5,
            },
            ActionBeat {
                timestamp: 2.0,
                duration: 0.1,
                intensity: 0.5,
                category: BeatCategory::Generic,
                confidence: 0.5,
            },
            ActionBeat {
                timestamp: 7.0,
                duration: 0.1,
                intensity: 0.5,
                category: BeatCategory::Generic,
                confidence: 0.5,
            },
        ];
        let p = compute_pacing(&beats, 10.0, 5.0);
        assert_eq!(p.segments.len(), 2);
    }

    #[test]
    fn test_beat_category_display() {
        assert_eq!(
            format!("{}", BeatCategory::PhysicalAction),
            "PhysicalAction"
        );
        assert_eq!(
            format!("{}", BeatCategory::AudioTransient),
            "AudioTransient"
        );
    }

    #[test]
    fn test_merge_beats() {
        let mut beats = vec![
            ActionBeat {
                timestamp: 1.0,
                duration: 0.1,
                intensity: 0.5,
                category: BeatCategory::Generic,
                confidence: 0.5,
            },
            ActionBeat {
                timestamp: 1.05,
                duration: 0.1,
                intensity: 0.7,
                category: BeatCategory::Generic,
                confidence: 0.5,
            },
            ActionBeat {
                timestamp: 5.0,
                duration: 0.1,
                intensity: 0.3,
                category: BeatCategory::Generic,
                confidence: 0.5,
            },
        ];
        merge_beats(&mut beats, 0.1);
        assert_eq!(beats.len(), 2);
        assert!((beats[0].intensity - 0.7).abs() < 1e-9);
    }
}
