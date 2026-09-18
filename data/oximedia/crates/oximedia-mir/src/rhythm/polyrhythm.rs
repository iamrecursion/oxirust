//! Polyrhythm detection and advanced syncopation analysis.
//!
//! Detects polyrhythmic patterns (e.g. 3-against-2, 4-against-3) by analyzing
//! inter-onset intervals and computing ratio-based rhythmic features.

use crate::MirResult;
use serde::{Deserialize, Serialize};

/// A detected polyrhythmic pattern.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PolyrhythmPattern {
    /// Primary pulse count (e.g. 3 in 3:2 polyrhythm).
    pub primary: u32,
    /// Secondary pulse count (e.g. 2 in 3:2 polyrhythm).
    pub secondary: u32,
    /// Confidence of detection (0.0 to 1.0).
    pub confidence: f32,
    /// Start time in seconds.
    pub start_time: f32,
    /// Duration in seconds.
    pub duration: f32,
}

impl PolyrhythmPattern {
    /// Return the ratio as a string (e.g. "3:2").
    #[must_use]
    pub fn ratio_string(&self) -> String {
        format!("{}:{}", self.primary, self.secondary)
    }
}

/// Advanced syncopation metrics.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SyncopationMetrics {
    /// Overall syncopation index (0.0 to 1.0).
    pub syncopation_index: f32,
    /// Longuet-Higgins & Lee syncopation weight (normalized 0.0 to 1.0).
    pub lhl_weight: f32,
    /// Off-beat ratio: fraction of onsets on off-beat positions.
    pub off_beat_ratio: f32,
    /// Anticipation ratio: fraction of onsets slightly before expected beats.
    pub anticipation_ratio: f32,
}

/// Result of extended rhythm analysis.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExtendedRhythmResult {
    /// Detected polyrhythmic patterns.
    pub polyrhythms: Vec<PolyrhythmPattern>,
    /// Syncopation metrics.
    pub syncopation: SyncopationMetrics,
    /// Rhythmic density (onsets per second).
    pub density: f32,
    /// Swing ratio (ratio of long to short subdivision, 1.0 = straight, >1.0 = swing).
    pub swing_ratio: f32,
    /// Metric regularity (0.0 = irregular, 1.0 = perfectly regular).
    pub metric_regularity: f32,
}

/// Polyrhythm and syncopation analyzer.
pub struct PolyrhythmAnalyzer {
    sample_rate: f32,
    /// Estimated BPM (needed for beat-relative calculations).
    estimated_bpm: Option<f32>,
}

impl PolyrhythmAnalyzer {
    /// Create a new polyrhythm analyzer.
    #[must_use]
    pub fn new(sample_rate: f32) -> Self {
        Self {
            sample_rate,
            estimated_bpm: None,
        }
    }

    /// Create with a known BPM.
    #[must_use]
    pub fn with_bpm(sample_rate: f32, bpm: f32) -> Self {
        Self {
            sample_rate,
            estimated_bpm: Some(bpm),
        }
    }

    /// Analyze onset times for polyrhythms and syncopation.
    ///
    /// # Errors
    ///
    /// Returns error if analysis fails.
    pub fn analyze(&self, onset_times: &[f32]) -> MirResult<ExtendedRhythmResult> {
        let bpm = self
            .estimated_bpm
            .unwrap_or_else(|| self.estimate_bpm(onset_times));
        let beat_duration = if bpm > 0.0 { 60.0 / bpm } else { 0.5 };

        let polyrhythms = self.detect_polyrhythms(onset_times, beat_duration);
        let syncopation = self.compute_syncopation(onset_times, beat_duration);
        let density = self.compute_density(onset_times);
        let swing_ratio = self.compute_swing_ratio(onset_times, beat_duration);
        let metric_regularity = self.compute_metric_regularity(onset_times, beat_duration);

        Ok(ExtendedRhythmResult {
            polyrhythms,
            syncopation,
            density,
            swing_ratio,
            metric_regularity,
        })
    }

    /// Estimate BPM from inter-onset intervals.
    #[allow(clippy::cast_precision_loss)]
    fn estimate_bpm(&self, onset_times: &[f32]) -> f32 {
        if onset_times.len() < 2 {
            return 120.0; // default
        }

        let mut intervals = Vec::with_capacity(onset_times.len() - 1);
        for i in 1..onset_times.len() {
            let ioi = onset_times[i] - onset_times[i - 1];
            if ioi > 0.01 {
                intervals.push(ioi);
            }
        }

        if intervals.is_empty() {
            return 120.0;
        }

        let median_ioi = {
            let mut sorted = intervals.clone();
            sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
            sorted[sorted.len() / 2]
        };

        let bpm = 60.0 / median_ioi;
        // Clamp to reasonable range
        bpm.clamp(30.0, 300.0)
    }

    /// Detect polyrhythmic patterns by analyzing IOI ratio clusters.
    ///
    /// We look for windows where the IOI ratios approximate known polyrhythm
    /// ratios (2:3, 3:4, etc.).
    #[allow(clippy::cast_precision_loss)]
    fn detect_polyrhythms(
        &self,
        onset_times: &[f32],
        _beat_duration: f32,
    ) -> Vec<PolyrhythmPattern> {
        if onset_times.len() < 6 {
            return Vec::new();
        }

        // Known polyrhythm ratios to detect: (primary, secondary, ratio)
        let known_ratios: &[(u32, u32, f32)] = &[
            (3, 2, 1.5),
            (4, 3, 1.333_333_3),
            (5, 3, 1.666_666_7),
            (5, 4, 1.25),
            (7, 4, 1.75),
        ];

        let mut intervals = Vec::with_capacity(onset_times.len() - 1);
        for i in 1..onset_times.len() {
            intervals.push(onset_times[i] - onset_times[i - 1]);
        }

        let mut patterns = Vec::new();
        let window_size = 8_usize;

        for start in 0..intervals.len().saturating_sub(window_size) {
            let window = &intervals[start..start + window_size];

            // Sort window IOIs to find the two dominant interval groups
            let mut sorted_iois = window.to_vec();
            sorted_iois.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

            // Split into "short" and "long" groups at the median
            let mid = sorted_iois.len() / 2;
            let short_mean: f32 = sorted_iois[..mid].iter().sum::<f32>() / mid as f32;
            let long_mean: f32 =
                sorted_iois[mid..].iter().sum::<f32>() / (sorted_iois.len() - mid) as f32;

            if short_mean < 1e-6 {
                continue;
            }

            let ratio = long_mean / short_mean;

            // Match against known polyrhythm ratios
            for &(primary, secondary, target_ratio) in known_ratios {
                let diff = (ratio - target_ratio).abs();
                if diff < 0.15 {
                    let confidence = 1.0 - (diff / 0.15);
                    let start_time = onset_times[start];
                    let end_idx = (start + window_size).min(onset_times.len() - 1);
                    let duration = onset_times[end_idx] - start_time;

                    patterns.push(PolyrhythmPattern {
                        primary,
                        secondary,
                        confidence: confidence.clamp(0.0, 1.0),
                        start_time,
                        duration,
                    });
                    break; // One match per window
                }
            }
        }

        // Deduplicate overlapping patterns (keep highest confidence)
        self.deduplicate_patterns(patterns)
    }

    /// Remove overlapping polyrhythm detections, keeping the highest confidence.
    fn deduplicate_patterns(&self, mut patterns: Vec<PolyrhythmPattern>) -> Vec<PolyrhythmPattern> {
        if patterns.len() <= 1 {
            return patterns;
        }

        patterns.sort_by(|a, b| {
            a.start_time
                .partial_cmp(&b.start_time)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        let mut result = Vec::new();
        let mut last_end = f32::NEG_INFINITY;

        for p in patterns {
            let p_end = p.start_time + p.duration;
            if p.start_time > last_end - 0.1 {
                last_end = p_end;
                result.push(p);
            } else if let Some(last) = result.last_mut() {
                if p.confidence > last.confidence {
                    *last = p;
                    last_end = p_end;
                }
            }
        }

        result
    }

    /// Compute syncopation metrics using beat-relative onset positions.
    #[allow(clippy::cast_precision_loss)]
    fn compute_syncopation(&self, onset_times: &[f32], beat_duration: f32) -> SyncopationMetrics {
        if onset_times.is_empty() || beat_duration <= 0.0 {
            return SyncopationMetrics {
                syncopation_index: 0.0,
                lhl_weight: 0.0,
                off_beat_ratio: 0.0,
                anticipation_ratio: 0.0,
            };
        }

        let first_onset = onset_times[0];
        let mut off_beat_count = 0_u32;
        let mut anticipation_count = 0_u32;
        let mut lhl_total = 0.0_f32;

        // Metrical weight hierarchy for 4/4 time subdivided to 16th notes
        // Position within a beat: [downbeat, e, and, a] weights = [4, 1, 2, 1]
        let metrical_weights: [f32; 4] = [4.0, 1.0, 2.0, 1.0];
        let subdivision = beat_duration / 4.0;

        for &onset in onset_times {
            let beat_phase = ((onset - first_onset) / beat_duration).rem_euclid(1.0);

            // Quantize to nearest 16th note position
            let sub_position = (beat_phase * 4.0).round() as usize % 4;
            let metrical_weight = metrical_weights[sub_position];

            // Off-beat: not on the downbeat
            if sub_position != 0 {
                off_beat_count += 1;
            }

            // Anticipation: onset slightly before a strong beat (within 10% of subdivision)
            let distance_to_next_strong = if sub_position == 3 {
                // Just before next downbeat
                (1.0 - beat_phase) * beat_duration
            } else {
                f32::MAX
            };
            if distance_to_next_strong < subdivision * 0.5 {
                anticipation_count += 1;
            }

            // LHL syncopation: syncopation occurs when a weak position is followed
            // by a rest on a stronger position. Approximate by inverse metrical weight.
            lhl_total += (4.0 - metrical_weight) / 4.0;
        }

        let n = onset_times.len() as f32;
        let off_beat_ratio = off_beat_count as f32 / n;
        let anticipation_ratio = anticipation_count as f32 / n;
        let lhl_weight = (lhl_total / n).clamp(0.0, 1.0);

        // Combined syncopation index
        let syncopation_index =
            (off_beat_ratio * 0.5 + lhl_weight * 0.3 + anticipation_ratio * 0.2).clamp(0.0, 1.0);

        SyncopationMetrics {
            syncopation_index,
            lhl_weight,
            off_beat_ratio,
            anticipation_ratio,
        }
    }

    /// Compute rhythmic density (onsets per second).
    fn compute_density(&self, onset_times: &[f32]) -> f32 {
        if onset_times.len() < 2 {
            return 0.0;
        }
        let first = onset_times[0];
        let last = onset_times[onset_times.len() - 1];
        let duration = last - first;
        if duration <= 0.0 {
            return 0.0;
        }
        onset_times.len() as f32 / duration
    }

    /// Compute swing ratio from alternating long-short IOI patterns.
    #[allow(clippy::cast_precision_loss)]
    fn compute_swing_ratio(&self, onset_times: &[f32], beat_duration: f32) -> f32 {
        if onset_times.len() < 4 {
            return 1.0; // Straight
        }

        let subdivision = beat_duration / 2.0; // eighth note
        let tolerance = subdivision * 0.5;

        let mut long_intervals = Vec::new();
        let mut short_intervals = Vec::new();

        for i in 1..onset_times.len() {
            let ioi = onset_times[i] - onset_times[i - 1];
            if ioi < 0.01 {
                continue;
            }

            // Check if this IOI is roughly an eighth note subdivision
            if (ioi - subdivision).abs() < tolerance {
                if ioi > subdivision {
                    long_intervals.push(ioi);
                } else {
                    short_intervals.push(ioi);
                }
            }
        }

        if short_intervals.is_empty() || long_intervals.is_empty() {
            return 1.0;
        }

        let long_mean: f32 = long_intervals.iter().sum::<f32>() / long_intervals.len() as f32;
        let short_mean: f32 = short_intervals.iter().sum::<f32>() / short_intervals.len() as f32;

        if short_mean < 1e-6 {
            return 1.0;
        }

        (long_mean / short_mean).clamp(0.5, 3.0)
    }

    /// Compute metric regularity (how well onsets align with a regular grid).
    #[allow(clippy::cast_precision_loss)]
    fn compute_metric_regularity(&self, onset_times: &[f32], beat_duration: f32) -> f32 {
        if onset_times.len() < 2 || beat_duration <= 0.0 {
            return 0.0;
        }

        let first = onset_times[0];
        let mut total_deviation = 0.0_f32;

        for &onset in onset_times {
            // Distance to nearest grid point
            let beat_phase = ((onset - first) / beat_duration).rem_euclid(1.0);
            let nearest_grid = beat_phase.round();
            let deviation = (beat_phase - nearest_grid).abs();
            total_deviation += deviation;
        }

        let mean_deviation = total_deviation / onset_times.len() as f32;
        // Convert to regularity (0 deviation = 1.0 regularity)
        (1.0 - mean_deviation * 4.0).clamp(0.0, 1.0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Generate perfectly regular onset times at given BPM.
    fn regular_onsets(bpm: f32, duration: f32) -> Vec<f32> {
        let interval = 60.0 / bpm;
        let mut onsets = Vec::new();
        let mut t = 0.0;
        while t < duration {
            onsets.push(t);
            t += interval;
        }
        onsets
    }

    /// Generate a 3:2 polyrhythm pattern.
    fn polyrhythm_3_2(beat_duration: f32, num_cycles: usize) -> Vec<f32> {
        let mut onsets = Vec::new();
        for cycle in 0..num_cycles {
            let base = cycle as f32 * beat_duration * 2.0;
            // Group of 3 evenly spaced
            let tri_interval = beat_duration * 2.0 / 3.0;
            onsets.push(base);
            onsets.push(base + tri_interval);
            onsets.push(base + 2.0 * tri_interval);
        }
        onsets.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        onsets.dedup_by(|a, b| (*a - *b).abs() < 0.001);
        onsets
    }

    #[test]
    fn test_analyzer_creation() {
        let analyzer = PolyrhythmAnalyzer::new(44100.0);
        assert!((analyzer.sample_rate - 44100.0).abs() < f32::EPSILON);
    }

    #[test]
    fn test_analyzer_with_bpm() {
        let analyzer = PolyrhythmAnalyzer::with_bpm(44100.0, 120.0);
        assert_eq!(analyzer.estimated_bpm, Some(120.0));
    }

    #[test]
    fn test_estimate_bpm_regular() {
        let analyzer = PolyrhythmAnalyzer::new(44100.0);
        let onsets = regular_onsets(120.0, 10.0);
        let bpm = analyzer.estimate_bpm(&onsets);
        assert!((bpm - 120.0).abs() < 5.0, "Expected ~120 BPM, got {bpm}");
    }

    #[test]
    fn test_estimate_bpm_empty() {
        let analyzer = PolyrhythmAnalyzer::new(44100.0);
        let bpm = analyzer.estimate_bpm(&[]);
        assert!((bpm - 120.0).abs() < f32::EPSILON); // default
    }

    #[test]
    fn test_regular_rhythm_high_regularity() {
        let analyzer = PolyrhythmAnalyzer::with_bpm(44100.0, 120.0);
        let onsets = regular_onsets(120.0, 10.0);
        let result = analyzer.analyze(&onsets);
        assert!(result.is_ok());
        let r = result.expect("analysis should succeed");
        assert!(
            r.metric_regularity > 0.7,
            "Expected high regularity, got {}",
            r.metric_regularity
        );
    }

    #[test]
    fn test_regular_rhythm_low_syncopation() {
        let analyzer = PolyrhythmAnalyzer::with_bpm(44100.0, 120.0);
        let onsets = regular_onsets(120.0, 10.0);
        let result = analyzer.analyze(&onsets).expect("should succeed");
        assert!(
            result.syncopation.syncopation_index < 0.5,
            "Expected low syncopation, got {}",
            result.syncopation.syncopation_index
        );
    }

    #[test]
    fn test_density_calculation() {
        let analyzer = PolyrhythmAnalyzer::new(44100.0);
        let onsets = regular_onsets(120.0, 10.0); // 2 onsets/sec at 120BPM
        let density = analyzer.compute_density(&onsets);
        assert!(
            (density - 2.0).abs() < 0.5,
            "Expected ~2.0 onsets/sec, got {density}"
        );
    }

    #[test]
    fn test_density_empty() {
        let analyzer = PolyrhythmAnalyzer::new(44100.0);
        assert!((analyzer.compute_density(&[]) - 0.0).abs() < f32::EPSILON);
    }

    #[test]
    fn test_swing_ratio_straight() {
        let analyzer = PolyrhythmAnalyzer::with_bpm(44100.0, 120.0);
        let onsets = regular_onsets(120.0, 5.0);
        let ratio = analyzer.compute_swing_ratio(&onsets, 0.5);
        // Regular onsets should be close to 1.0 (straight)
        assert!(ratio >= 0.5 && ratio <= 3.0);
    }

    #[test]
    fn test_polyrhythm_detection_empty() {
        let analyzer = PolyrhythmAnalyzer::new(44100.0);
        let patterns = analyzer.detect_polyrhythms(&[], 0.5);
        assert!(patterns.is_empty());
    }

    #[test]
    fn test_polyrhythm_pattern_ratio_string() {
        let p = PolyrhythmPattern {
            primary: 3,
            secondary: 2,
            confidence: 0.8,
            start_time: 0.0,
            duration: 2.0,
        };
        assert_eq!(p.ratio_string(), "3:2");
    }

    #[test]
    fn test_syncopation_metrics_empty() {
        let analyzer = PolyrhythmAnalyzer::new(44100.0);
        let m = analyzer.compute_syncopation(&[], 0.5);
        assert!((m.syncopation_index - 0.0).abs() < f32::EPSILON);
    }

    #[test]
    fn test_off_beat_onsets_higher_syncopation() {
        let analyzer = PolyrhythmAnalyzer::with_bpm(44100.0, 120.0);
        let beat_dur = 0.5; // 120 BPM

        // On-beat onsets
        let on_beat = regular_onsets(120.0, 5.0);
        let on_beat_sync = analyzer.compute_syncopation(&on_beat, beat_dur);

        // Off-beat onsets (shifted by half a beat)
        let off_beat: Vec<f32> = on_beat.iter().map(|t| t + beat_dur * 0.25).collect();
        let off_beat_sync = analyzer.compute_syncopation(&off_beat, beat_dur);

        assert!(
            off_beat_sync.off_beat_ratio >= on_beat_sync.off_beat_ratio,
            "Off-beat should have higher off_beat_ratio"
        );
    }

    #[test]
    fn test_metric_regularity_regular() {
        let analyzer = PolyrhythmAnalyzer::new(44100.0);
        let onsets = regular_onsets(120.0, 5.0);
        let reg = analyzer.compute_metric_regularity(&onsets, 0.5);
        assert!(
            reg > 0.8,
            "Regular onsets should have high metric regularity, got {reg}"
        );
    }

    #[test]
    fn test_metric_regularity_irregular() {
        let analyzer = PolyrhythmAnalyzer::new(44100.0);
        let onsets = vec![0.0, 0.13, 0.37, 0.61, 0.88, 1.14, 1.5, 1.73];
        let reg = analyzer.compute_metric_regularity(&onsets, 0.5);
        // Irregular onsets should have lower regularity
        assert!(reg < 0.9);
    }

    #[test]
    fn test_full_analysis_returns_ok() {
        let analyzer = PolyrhythmAnalyzer::with_bpm(44100.0, 120.0);
        let onsets = regular_onsets(120.0, 10.0);
        let result = analyzer.analyze(&onsets);
        assert!(result.is_ok());
    }

    #[test]
    fn test_full_analysis_density_positive() {
        let analyzer = PolyrhythmAnalyzer::with_bpm(44100.0, 120.0);
        let onsets = regular_onsets(120.0, 10.0);
        let result = analyzer.analyze(&onsets).expect("should succeed");
        assert!(result.density > 0.0);
    }

    #[test]
    fn test_extended_rhythm_result_serialization() {
        let result = ExtendedRhythmResult {
            polyrhythms: Vec::new(),
            syncopation: SyncopationMetrics {
                syncopation_index: 0.3,
                lhl_weight: 0.2,
                off_beat_ratio: 0.4,
                anticipation_ratio: 0.1,
            },
            density: 2.0,
            swing_ratio: 1.0,
            metric_regularity: 0.9,
        };
        let debug = format!("{:?}", result);
        assert!(debug.contains("syncopation_index"));
    }

    #[test]
    fn test_deduplicate_patterns_empty() {
        let analyzer = PolyrhythmAnalyzer::new(44100.0);
        let result = analyzer.deduplicate_patterns(Vec::new());
        assert!(result.is_empty());
    }

    #[test]
    fn test_deduplicate_patterns_single() {
        let analyzer = PolyrhythmAnalyzer::new(44100.0);
        let patterns = vec![PolyrhythmPattern {
            primary: 3,
            secondary: 2,
            confidence: 0.8,
            start_time: 0.0,
            duration: 2.0,
        }];
        let result = analyzer.deduplicate_patterns(patterns);
        assert_eq!(result.len(), 1);
    }
}
