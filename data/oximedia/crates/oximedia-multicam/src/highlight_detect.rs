//! Highlight detection combining cut density and audio energy peaks.
//!
//! This module provides a higher-level, timestamp-oriented API for detecting
//! exciting moments in a multi-camera recording.  It complements the
//! frame-level [`crate::highlight`] module by operating in millisecond time
//! rather than raw frame indices and exposing a richer [`HighlightReason`]
//! classification.
//!
//! # Algorithm
//!
//! 1. **Cut density window** — for each candidate moment, count how many
//!    `cut_timestamps` fall within a ±`WINDOW_MS` window.  A high density of
//!    cuts signals editorial excitement.
//! 2. **Audio energy peak** — the corresponding audio energy sample (indexed
//!    by `timestamp_ms * sample_rate / 1000`) is checked against an adaptive
//!    threshold derived from the mean energy of the entire signal.
//! 3. A [`HighlightMoment`] is emitted when either criterion exceeds its
//!    threshold.  The `score` combines both contributions with equal weighting.
//! 4. Adjacent moments within `MERGE_WINDOW_MS` milliseconds are merged,
//!    keeping the highest score.

// ── Constants ─────────────────────────────────────────────────────────────────

/// Half-width of the cut-density search window (milliseconds).
const WINDOW_MS: u64 = 500;

/// Moments closer than this are merged into a single highlight (milliseconds).
const MERGE_WINDOW_MS: u64 = 1000;

// ── HighlightReason ───────────────────────────────────────────────────────────

/// Why a particular moment was flagged as a highlight.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HighlightReason {
    /// Many editorial cuts occur in a short window — fast-paced action.
    HighCutDensity,
    /// Audio energy spike detected — crowd noise, impact, loud event.
    AudioEnergyPeak,
    /// Both cut density and audio energy are elevated simultaneously.
    Combined,
}

// ── HighlightMoment ───────────────────────────────────────────────────────────

/// A single detected highlight moment.
#[derive(Debug, Clone, PartialEq)]
pub struct HighlightMoment {
    /// Position in the timeline in milliseconds from the start.
    pub timestamp_ms: u64,
    /// Normalised highlight score in `[0.0, 1.0]`.
    pub score: f32,
    /// The primary reason this moment was detected.
    pub reason: HighlightReason,
}

// ── HighlightDetector ─────────────────────────────────────────────────────────

/// Detects highlight moments by combining cut density and audio energy peaks.
///
/// # Example
///
/// ```
/// use oximedia_multicam::highlight_detect::detect_highlights;
///
/// let cuts = vec![1_000_u64, 1_100, 1_200, 3_000];
/// let energy: Vec<f32> = (0..4800).map(|i| {
///     if i >= 100 && i < 200 { 1.0 } else { 0.05 }
/// }).collect();
/// let moments = detect_highlights(&cuts, &energy, 48_000);
/// assert!(!moments.is_empty());
/// ```
#[derive(Debug, Clone)]
pub struct HighlightDetector {
    /// Minimum number of cuts in a `WINDOW_MS` window to flag cut-density.
    pub cut_density_min: usize,
    /// Factor above mean energy required to flag an audio peak (e.g. 2.0 = 2×).
    pub energy_peak_factor: f32,
}

impl HighlightDetector {
    /// Create a new detector with default parameters.
    #[must_use]
    pub fn new() -> Self {
        Self {
            cut_density_min: 3,
            energy_peak_factor: 2.0,
        }
    }

    /// Set the minimum cut count in a window to trigger a density highlight.
    #[must_use]
    pub fn with_cut_density_min(mut self, n: usize) -> Self {
        self.cut_density_min = n.max(1);
        self
    }

    /// Set the energy peak factor (multiple of the signal mean).
    #[must_use]
    pub fn with_energy_peak_factor(mut self, f: f32) -> Self {
        self.energy_peak_factor = f.max(1.0);
        self
    }

    /// Detect highlight moments.
    ///
    /// See the module-level documentation for the algorithm description.
    #[must_use]
    pub fn detect(
        &self,
        cut_timestamps: &[u64],
        audio_energy: &[f32],
        sample_rate: u32,
    ) -> Vec<HighlightMoment> {
        detect_highlights_with(
            cut_timestamps,
            audio_energy,
            sample_rate,
            self.cut_density_min,
            self.energy_peak_factor,
        )
    }
}

impl Default for HighlightDetector {
    fn default() -> Self {
        Self::new()
    }
}

// ── Public free function ──────────────────────────────────────────────────────

/// Detect highlight moments using default thresholds.
///
/// Convenience wrapper around [`HighlightDetector`].
///
/// # Arguments
///
/// * `cut_timestamps` – Timestamps of editorial cut points in milliseconds.
/// * `audio_energy`   – Per-sample RMS energy values (non-negative).
/// * `sample_rate`    – Audio sample rate in Hz (e.g. `48_000`).
///
/// # Returns
///
/// A list of [`HighlightMoment`]s sorted by `timestamp_ms`, with nearby
/// moments merged within `MERGE_WINDOW_MS` milliseconds of each other.
#[must_use]
pub fn detect_highlights(
    cut_timestamps: &[u64],
    audio_energy: &[f32],
    sample_rate: u32,
) -> Vec<HighlightMoment> {
    HighlightDetector::new().detect(cut_timestamps, audio_energy, sample_rate)
}

// ── Implementation ────────────────────────────────────────────────────────────

#[allow(clippy::cast_precision_loss)]
fn detect_highlights_with(
    cut_timestamps: &[u64],
    audio_energy: &[f32],
    sample_rate: u32,
    cut_density_min: usize,
    energy_peak_factor: f32,
) -> Vec<HighlightMoment> {
    if cut_timestamps.is_empty() && audio_energy.is_empty() {
        return Vec::new();
    }

    // Pre-compute mean audio energy for adaptive threshold.
    let mean_energy = if audio_energy.is_empty() {
        0.0f32
    } else {
        audio_energy.iter().copied().sum::<f32>() / audio_energy.len() as f32
    };
    let energy_threshold = mean_energy * energy_peak_factor;

    // Build the candidate set: one candidate per cut point, and one per
    // audio energy peak.
    let mut candidates: Vec<u64> = Vec::new();

    for &ts in cut_timestamps {
        candidates.push(ts);
    }

    // Scan audio_energy for local maxima above threshold.
    if sample_rate > 0 && !audio_energy.is_empty() {
        // We scan in chunks of ~10 ms to avoid O(samples²) work.
        let chunk_size = (sample_rate / 100).max(1) as usize;
        let mut chunk_idx = 0usize;
        while chunk_idx < audio_energy.len() {
            let end = (chunk_idx + chunk_size).min(audio_energy.len());
            let chunk = &audio_energy[chunk_idx..end];
            let local_max = chunk.iter().copied().fold(0.0f32, f32::max);
            if local_max >= energy_threshold {
                // Timestamp of the midpoint of this chunk.
                let mid_sample = chunk_idx + chunk_size / 2;
                let ts_ms = (mid_sample as u64 * 1000) / u64::from(sample_rate);
                candidates.push(ts_ms);
            }
            chunk_idx += chunk_size;
        }
    }

    candidates.sort_unstable();
    candidates.dedup();

    if candidates.is_empty() {
        return Vec::new();
    }

    // Score each candidate.
    let mut moments: Vec<HighlightMoment> = candidates
        .iter()
        .filter_map(|&ts| {
            score_candidate(
                ts,
                cut_timestamps,
                audio_energy,
                sample_rate,
                mean_energy,
                cut_density_min,
                energy_peak_factor,
            )
        })
        .collect();

    moments.sort_by_key(|m| m.timestamp_ms);

    // Merge moments within MERGE_WINDOW_MS.
    merge_moments(moments)
}

/// Score a single candidate timestamp, returning `None` if it passes no threshold.
#[allow(clippy::cast_precision_loss, clippy::too_many_arguments)]
fn score_candidate(
    timestamp_ms: u64,
    cut_timestamps: &[u64],
    audio_energy: &[f32],
    sample_rate: u32,
    mean_energy: f32,
    cut_density_min: usize,
    energy_peak_factor: f32,
) -> Option<HighlightMoment> {
    // ── Cut density ───────────────────────────────────────────────────────────
    let lo = timestamp_ms.saturating_sub(WINDOW_MS);
    let hi = timestamp_ms.saturating_add(WINDOW_MS);
    let cut_count = cut_timestamps
        .iter()
        .filter(|&&t| t >= lo && t <= hi)
        .count();

    let density_score = if cut_density_min == 0 {
        0.0f32
    } else {
        (cut_count as f32 / cut_density_min as f32).clamp(0.0, 1.0)
    };
    let density_triggered = cut_count >= cut_density_min;

    // ── Audio energy at timestamp ─────────────────────────────────────────────
    let energy_score = if sample_rate > 0 && !audio_energy.is_empty() {
        let sample_idx = ((timestamp_ms as u64 * u64::from(sample_rate)) / 1000) as usize;
        let sample = if sample_idx < audio_energy.len() {
            audio_energy[sample_idx]
        } else {
            0.0
        };

        if mean_energy > 0.0 {
            (sample / (mean_energy * energy_peak_factor)).clamp(0.0, 1.0)
        } else {
            0.0
        }
    } else {
        0.0
    };
    let energy_triggered = energy_score >= 1.0 / energy_peak_factor.max(1.0);

    if !density_triggered && !energy_triggered {
        return None;
    }

    let score = (density_score + energy_score) / 2.0;
    let reason = match (density_triggered, energy_triggered) {
        (true, true) => HighlightReason::Combined,
        (true, false) => HighlightReason::HighCutDensity,
        (false, true) => HighlightReason::AudioEnergyPeak,
        (false, false) => return None,
    };

    Some(HighlightMoment {
        timestamp_ms,
        score: score.clamp(0.0, 1.0),
        reason,
    })
}

/// Merge nearby highlight moments, keeping the maximum score.
fn merge_moments(moments: Vec<HighlightMoment>) -> Vec<HighlightMoment> {
    if moments.is_empty() {
        return moments;
    }

    let mut merged: Vec<HighlightMoment> = Vec::with_capacity(moments.len());

    for m in moments {
        if let Some(last) = merged.last_mut() {
            if m.timestamp_ms.saturating_sub(last.timestamp_ms) <= MERGE_WINDOW_MS {
                // Upgrade reason to Combined when different non-Combined reasons merge.
                let new_reason = if last.reason != m.reason
                    && last.reason != HighlightReason::Combined
                    && m.reason != HighlightReason::Combined
                {
                    HighlightReason::Combined
                } else if m.reason == HighlightReason::Combined {
                    HighlightReason::Combined
                } else {
                    last.reason
                };

                // Keep the higher score.
                if m.score > last.score {
                    last.score = m.score;
                }
                last.reason = new_reason;
                continue;
            }
        }
        merged.push(m);
    }

    merged
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    /// Dense cuts within 500 ms window should produce a cut-density highlight.
    #[test]
    fn test_detect_highlights_cut_density() {
        let cuts = vec![1_000_u64, 1_100, 1_200, 1_300, 1_400];
        let energy = vec![0.05f32; 96_000]; // 2 s at 48 kHz, quiet
        let moments = detect_highlights(&cuts, &energy, 48_000);
        assert!(
            !moments.is_empty(),
            "Expected highlight from dense cuts; got none"
        );
        let has_density = moments.iter().any(|m| {
            matches!(
                m.reason,
                HighlightReason::HighCutDensity | HighlightReason::Combined
            )
        });
        assert!(
            has_density,
            "Expected HighCutDensity reason; got {:?}",
            moments
        );
    }

    /// High audio energy should produce an energy-peak highlight.
    #[test]
    fn test_detect_highlights_audio_peak() {
        // No cuts; energy spike at ~1 second (sample 48000).
        let cuts: Vec<u64> = Vec::new();
        let mut energy = vec![0.05f32; 96_000];
        for e in &mut energy[47_500..48_500] {
            *e = 2.0; // well above 2× mean
        }
        let moments = detect_highlights(&cuts, &energy, 48_000);
        assert!(
            !moments.is_empty(),
            "Expected highlight from audio peak; got none"
        );
        let has_energy = moments.iter().any(|m| {
            matches!(
                m.reason,
                HighlightReason::AudioEnergyPeak | HighlightReason::Combined
            )
        });
        assert!(
            has_energy,
            "Expected AudioEnergyPeak reason; got {:?}",
            moments
        );
    }

    /// Empty inputs should return an empty list.
    #[test]
    fn test_detect_highlights_empty() {
        let moments = detect_highlights(&[], &[], 48_000);
        assert!(moments.is_empty());
    }

    /// Moments close together should be merged.
    #[test]
    fn test_merge_moments_merges_nearby() {
        let m1 = HighlightMoment {
            timestamp_ms: 0,
            score: 0.6,
            reason: HighlightReason::HighCutDensity,
        };
        let m2 = HighlightMoment {
            timestamp_ms: 500,
            score: 0.8,
            reason: HighlightReason::AudioEnergyPeak,
        };
        let merged = merge_moments(vec![m1, m2]);
        assert_eq!(
            merged.len(),
            1,
            "Expected 1 merged moment, got {}",
            merged.len()
        );
        assert!((merged[0].score - 0.8).abs() < 1e-6);
        assert_eq!(merged[0].reason, HighlightReason::Combined);
    }

    /// Moments far apart should not be merged.
    #[test]
    fn test_merge_moments_keeps_separated() {
        let m1 = HighlightMoment {
            timestamp_ms: 0,
            score: 0.5,
            reason: HighlightReason::HighCutDensity,
        };
        let m2 = HighlightMoment {
            timestamp_ms: 5_000,
            score: 0.7,
            reason: HighlightReason::AudioEnergyPeak,
        };
        let merged = merge_moments(vec![m1, m2]);
        assert_eq!(merged.len(), 2);
    }

    /// Scores must be within [0.0, 1.0].
    #[test]
    fn test_scores_bounded() {
        let cuts: Vec<u64> = (0..100).map(|i| i * 10).collect();
        let energy: Vec<f32> = (0..96_000).map(|i| (i % 100) as f32 / 100.0).collect();
        let moments = detect_highlights(&cuts, &energy, 48_000);
        for m in &moments {
            assert!(
                m.score >= 0.0 && m.score <= 1.0,
                "score out of bounds: {}",
                m.score
            );
        }
    }

    /// HighlightDetector builder pattern works correctly.
    #[test]
    fn test_detector_builder() {
        let detector = HighlightDetector::new()
            .with_cut_density_min(2)
            .with_energy_peak_factor(3.0);
        assert_eq!(detector.cut_density_min, 2);
        assert!((detector.energy_peak_factor - 3.0).abs() < 1e-6);
    }
}
