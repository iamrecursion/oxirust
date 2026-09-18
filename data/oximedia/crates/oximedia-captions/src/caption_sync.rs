//! Caption synchronisation utilities for `OxiMedia`.
//!
//! Aligns caption tracks to audio/video references using anchor points,
//! detects drift, and produces detailed sync reports.
//!
//! # Audio Waveform Synchronisation
//!
//! The [`WaveformSync`] subsystem detects speech onsets from PCM audio energy
//! and aligns caption timestamps to those onsets, correcting drift that arises
//! from imprecise manual or ASR-generated timing.

/// A timing anchor that pins a caption timestamp to a reference timestamp.
#[derive(Debug, Clone, PartialEq)]
pub struct SyncAnchor {
    /// Caption timestamp in milliseconds.
    pub caption_ms: i64,
    /// Reference (audio/video) timestamp in milliseconds.
    pub reference_ms: i64,
    /// Confidence of this anchor (0.0 – 1.0).
    pub confidence: f32,
    /// Optional human-readable label (e.g. "shot cut", "speech onset").
    pub label: Option<String>,
}

impl SyncAnchor {
    /// Creates a new `SyncAnchor`.
    #[must_use]
    pub fn new(caption_ms: i64, reference_ms: i64, confidence: f32) -> Self {
        Self {
            caption_ms,
            reference_ms,
            confidence: confidence.clamp(0.0, 1.0),
            label: None,
        }
    }

    /// Attaches a label to this anchor.
    #[must_use]
    pub fn with_label(mut self, label: impl Into<String>) -> Self {
        self.label = Some(label.into());
        self
    }

    /// Returns the signed drift: `reference_ms - caption_ms`.
    ///
    /// A positive value means the caption is early; negative means it is late.
    #[must_use]
    pub fn drift_ms(&self) -> i64 {
        self.reference_ms - self.caption_ms
    }

    /// Returns `true` when the anchor has high confidence (≥ 0.8).
    #[must_use]
    pub fn is_high_confidence(&self) -> bool {
        self.confidence >= 0.8
    }
}

/// Configuration for a caption synchronisation pass.
#[derive(Debug, Clone)]
pub struct CaptionSyncConfig {
    /// Maximum acceptable drift in milliseconds before a cue is flagged.
    pub max_drift_ms: i64,
    /// Tolerance used when deciding whether an anchor aligns well enough.
    pub tolerance_ms: i64,
    /// Whether to apply a linear correction across the whole track.
    pub apply_linear_correction: bool,
    /// Minimum anchor confidence required for an anchor to be used.
    pub min_anchor_confidence: f32,
}

impl Default for CaptionSyncConfig {
    fn default() -> Self {
        Self {
            max_drift_ms: 500,
            tolerance_ms: 80,
            apply_linear_correction: true,
            min_anchor_confidence: 0.5,
        }
    }
}

impl CaptionSyncConfig {
    /// Creates a new sync configuration.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Sets the maximum drift threshold.
    #[must_use]
    pub fn with_max_drift(mut self, ms: i64) -> Self {
        self.max_drift_ms = ms;
        self
    }

    /// Returns the tolerance in milliseconds.
    #[must_use]
    pub fn tolerance_ms(&self) -> i64 {
        self.tolerance_ms
    }
}

/// A caption cue with mutable timing, used during synchronisation.
#[derive(Debug, Clone, PartialEq)]
pub struct SyncableCue {
    /// Cue index.
    pub index: usize,
    /// Start time in milliseconds.
    pub start_ms: i64,
    /// End time in milliseconds.
    pub end_ms: i64,
    /// Caption text content.
    pub text: String,
}

impl SyncableCue {
    /// Creates a new `SyncableCue`.
    #[must_use]
    pub fn new(index: usize, start_ms: i64, end_ms: i64, text: impl Into<String>) -> Self {
        Self {
            index,
            start_ms,
            end_ms,
            text: text.into(),
        }
    }

    /// Returns the duration of the cue in milliseconds.
    #[must_use]
    pub fn duration_ms(&self) -> i64 {
        self.end_ms - self.start_ms
    }

    /// Shifts the cue by `delta_ms` (positive = shift later).
    pub fn shift(&mut self, delta_ms: i64) {
        self.start_ms += delta_ms;
        self.end_ms += delta_ms;
    }
}

/// Aligns caption tracks to reference timing using anchor points.
#[derive(Debug, Default)]
pub struct CaptionSyncer;

impl CaptionSyncer {
    /// Creates a new `CaptionSyncer`.
    #[must_use]
    pub fn new() -> Self {
        Self
    }

    /// Aligns `cues` to reference timing using the provided `anchors` and `config`.
    ///
    /// Returns adjusted cues and a sync report.
    #[must_use]
    pub fn align(
        &self,
        mut cues: Vec<SyncableCue>,
        anchors: &[SyncAnchor],
        config: &CaptionSyncConfig,
    ) -> (Vec<SyncableCue>, SyncReport) {
        let valid_anchors: Vec<&SyncAnchor> = anchors
            .iter()
            .filter(|a| a.confidence >= config.min_anchor_confidence)
            .collect();

        if valid_anchors.is_empty() {
            let report = SyncReport::new(0, 0, 0, vec![], 0);
            return (cues, report);
        }

        // Compute global offset as the weighted mean drift of all valid anchors
        let (sum_w, sum_drift) = valid_anchors.iter().fold((0.0f64, 0.0f64), |(sw, sd), a| {
            let w = f64::from(a.confidence);
            (sw + w, sd + w * a.drift_ms() as f64)
        });
        let global_offset_ms = if sum_w > 0.0 {
            (sum_drift / sum_w).round() as i64
        } else {
            0
        };

        let mut drifts: Vec<i64> = Vec::new();
        let mut over_limit_cues: Vec<usize> = Vec::new();
        let mut corrected = 0usize;

        for cue in &mut cues {
            let effective_drift = global_offset_ms;
            drifts.push(effective_drift.abs());
            if effective_drift.abs() > config.tolerance_ms && config.apply_linear_correction {
                cue.shift(effective_drift);
                corrected += 1;
            }
            if effective_drift.abs() > config.max_drift_ms {
                over_limit_cues.push(cue.index);
            }
        }

        let max_drift = drifts.iter().copied().max().unwrap_or(0);
        let avg_drift = if drifts.is_empty() {
            0
        } else {
            (drifts.iter().sum::<i64>() as f64 / drifts.len() as f64).round() as i64
        };

        let report = SyncReport::new(
            max_drift,
            avg_drift,
            corrected,
            over_limit_cues,
            valid_anchors.len(),
        );
        (cues, report)
    }

    /// Returns the sync status description based on maximum drift.
    #[must_use]
    pub fn sync_status(&self, max_drift_ms: i64, config: &CaptionSyncConfig) -> SyncStatus {
        if max_drift_ms <= config.tolerance_ms {
            SyncStatus::Good
        } else if max_drift_ms <= config.max_drift_ms {
            SyncStatus::Acceptable
        } else {
            SyncStatus::OutOfSync
        }
    }
}

/// Describes the overall synchronisation quality of a caption track.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SyncStatus {
    /// All cues are within tolerance.
    Good,
    /// Some drift exists but is within the maximum threshold.
    Acceptable,
    /// Drift exceeds the configured threshold.
    OutOfSync,
}

impl SyncStatus {
    /// Human-readable label.
    #[must_use]
    pub fn label(self) -> &'static str {
        match self {
            Self::Good => "good",
            Self::Acceptable => "acceptable",
            Self::OutOfSync => "out-of-sync",
        }
    }
}

/// Summary of a synchronisation pass.
#[derive(Debug, Clone)]
pub struct SyncReport {
    /// Maximum absolute drift observed across all cues in milliseconds.
    pub max_drift_ms: i64,
    /// Average absolute drift in milliseconds.
    pub avg_drift_ms: i64,
    /// Number of cues that were corrected.
    pub corrected_cues: usize,
    /// Indices of cues whose drift exceeded the maximum threshold.
    pub over_limit_cue_indices: Vec<usize>,
    /// Number of anchors used in the alignment.
    pub anchors_used: usize,
}

impl SyncReport {
    /// Creates a new `SyncReport`.
    #[must_use]
    pub fn new(
        max_drift_ms: i64,
        avg_drift_ms: i64,
        corrected_cues: usize,
        over_limit_cue_indices: Vec<usize>,
        anchors_used: usize,
    ) -> Self {
        Self {
            max_drift_ms,
            avg_drift_ms,
            corrected_cues,
            over_limit_cue_indices,
            anchors_used,
        }
    }

    /// Returns the maximum absolute drift in milliseconds.
    #[must_use]
    pub fn max_drift_ms(&self) -> i64 {
        self.max_drift_ms
    }

    /// Returns `true` when no cues exceeded the maximum drift threshold.
    #[must_use]
    pub fn all_within_limit(&self) -> bool {
        self.over_limit_cue_indices.is_empty()
    }

    /// Returns the number of cues that exceeded the drift limit.
    #[must_use]
    pub fn over_limit_count(&self) -> usize {
        self.over_limit_cue_indices.len()
    }
}

// ============================================================================
// Audio Waveform-Based Synchronisation
// ============================================================================

/// Configuration for audio waveform-based caption synchronisation.
#[derive(Debug, Clone)]
pub struct WaveformSyncConfig {
    /// RMS energy threshold above which a window is considered to contain speech.
    /// Measured in normalised linear amplitude units (`0.0`–`1.0`); default `0.01`.
    pub energy_threshold: f32,
    /// Minimum gap between merged speech segments in milliseconds.  Onsets that
    /// fall within this window of a previous onset are merged; default `200`.
    pub min_gap_ms: u32,
    /// Maximum allowed timestamp shift in milliseconds.  Captions whose nearest
    /// onset lies further than this value are left unchanged; default `500`.
    pub max_shift_ms: i32,
}

impl Default for WaveformSyncConfig {
    fn default() -> Self {
        Self {
            energy_threshold: 0.01,
            min_gap_ms: 200,
            max_shift_ms: 500,
        }
    }
}

/// A detected speech onset together with the window's RMS energy.
#[derive(Debug, Clone, PartialEq)]
pub struct SpeechOnset {
    /// Time of the onset in milliseconds from the start of the audio stream.
    pub time_ms: u64,
    /// RMS energy of the detection window that triggered this onset.
    pub energy: f32,
}

impl SpeechOnset {
    /// Creates a new `SpeechOnset`.
    #[must_use]
    pub fn new(time_ms: u64, energy: f32) -> Self {
        Self { time_ms, energy }
    }
}

/// Detects speech onsets from PCM audio and aligns caption timestamps to them.
///
/// # Algorithm
///
/// **Onset detection** (`detect_onsets`):
/// 1. Slide a 20 ms analysis window with a 10 ms hop over the sample buffer.
/// 2. Compute the RMS energy of each window.
/// 3. Collect all windows whose energy rises above [`WaveformSyncConfig::energy_threshold`].
/// 4. Merge consecutive onsets that fall within [`WaveformSyncConfig::min_gap_ms`] of each
///    other, keeping the onset with the highest energy as the representative.
///
/// **Caption alignment** (`align_to_speech`):
/// For each caption, search the onset list for the nearest onset (by absolute
/// time distance from the caption's start).  If that onset lies within
/// [`WaveformSyncConfig::max_shift_ms`], shift both start and end timestamps by
/// the signed difference so the caption begins precisely at the onset.
#[derive(Debug)]
pub struct WaveformSync {
    config: WaveformSyncConfig,
}

impl WaveformSync {
    /// Creates a new `WaveformSync` with the given configuration.
    #[must_use]
    pub fn new(config: WaveformSyncConfig) -> Self {
        Self { config }
    }

    /// Detects speech onsets from a buffer of interleaved or mono PCM `f32` samples.
    ///
    /// `sample_rate` must be > 0.  An empty sample slice returns an empty onset list.
    #[must_use]
    pub fn detect_onsets(&self, samples: &[f32], sample_rate: u32) -> Vec<SpeechOnset> {
        if samples.is_empty() || sample_rate == 0 {
            return Vec::new();
        }

        let sr = sample_rate as f64;
        // Window length = 20 ms; hop = 10 ms — expressed in whole samples.
        let window_len = ((sr * 0.020).round() as usize).max(1);
        let hop_len = ((sr * 0.010).round() as usize).max(1);

        // Phase 1: collect all windows that exceed the energy threshold.
        let mut raw_onsets: Vec<SpeechOnset> = Vec::new();
        let mut window_start = 0usize;

        while window_start + window_len <= samples.len() {
            let window = &samples[window_start..window_start + window_len];
            let rms = compute_rms(window);
            if rms >= self.config.energy_threshold {
                // Convert window centre to milliseconds.
                let centre_sample = window_start + window_len / 2;
                let time_ms = ((centre_sample as f64 / sr) * 1000.0).round() as u64;
                raw_onsets.push(SpeechOnset::new(time_ms, rms));
            }
            window_start += hop_len;
        }

        if raw_onsets.is_empty() {
            return Vec::new();
        }

        // Phase 2: merge onsets within `min_gap_ms` of each other.
        let gap_ms = u64::from(self.config.min_gap_ms);
        let mut merged: Vec<SpeechOnset> = Vec::with_capacity(raw_onsets.len());
        // Seed with the first onset.
        let mut current = raw_onsets[0].clone();

        for next in raw_onsets.into_iter().skip(1) {
            if next.time_ms.saturating_sub(current.time_ms) <= gap_ms {
                // Keep the stronger representative within the same segment.
                if next.energy > current.energy {
                    current = next;
                }
            } else {
                merged.push(current);
                current = next;
            }
        }
        merged.push(current);
        merged
    }

    /// Aligns caption `(start_ms, end_ms)` pairs to the nearest detected speech onset.
    ///
    /// For each caption:
    /// - Find the onset whose `time_ms` is closest to the caption's `start_ms`.
    /// - If the signed shift `(onset.time_ms as i64 - start_ms as i64)` falls within
    ///   `[-max_shift_ms, +max_shift_ms]`, apply that shift to both endpoints.
    /// - Otherwise leave the caption unchanged.
    ///
    /// Timestamps are clamped to `u64::MIN` (0) on the lower bound; no upper clamp is
    /// applied as the total duration is not known at this layer.
    #[must_use]
    pub fn align_to_speech(
        &self,
        captions: &[(u64, u64)],
        onsets: &[SpeechOnset],
    ) -> Vec<(u64, u64)> {
        if onsets.is_empty() {
            return captions.to_vec();
        }

        let max_abs = i64::from(self.config.max_shift_ms.abs());

        captions
            .iter()
            .map(|&(start_ms, end_ms)| {
                // Find onset with minimum absolute distance to start_ms.
                let best = onsets.iter().min_by_key(|o| {
                    let onset_i = o.time_ms as i64;
                    let start_i = start_ms as i64;
                    (onset_i - start_i).unsigned_abs()
                });

                if let Some(onset) = best {
                    let shift: i64 = onset.time_ms as i64 - start_ms as i64;
                    if shift.abs() <= max_abs {
                        let new_start = (start_ms as i64 + shift).max(0) as u64;
                        let new_end = (end_ms as i64 + shift).max(0) as u64;
                        return (new_start, new_end);
                    }
                }
                (start_ms, end_ms)
            })
            .collect()
    }
}

/// Computes the root-mean-square energy of a sample window.
#[inline]
fn compute_rms(window: &[f32]) -> f32 {
    if window.is_empty() {
        return 0.0;
    }
    let sum_sq: f32 = window.iter().map(|s| s * s).sum();
    (sum_sq / window.len() as f32).sqrt()
}

// ============================================================================
// Unit Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    fn simple_cues() -> Vec<SyncableCue> {
        vec![
            SyncableCue::new(0, 0, 2000, "First cue"),
            SyncableCue::new(1, 2500, 5000, "Second cue"),
            SyncableCue::new(2, 5500, 8000, "Third cue"),
        ]
    }

    #[test]
    fn test_anchor_drift_positive() {
        let a = SyncAnchor::new(1000, 1200, 0.9);
        assert_eq!(a.drift_ms(), 200);
    }

    #[test]
    fn test_anchor_drift_negative() {
        let a = SyncAnchor::new(1000, 800, 0.9);
        assert_eq!(a.drift_ms(), -200);
    }

    #[test]
    fn test_anchor_high_confidence() {
        let high = SyncAnchor::new(0, 0, 0.9);
        let low = SyncAnchor::new(0, 0, 0.5);
        assert!(high.is_high_confidence());
        assert!(!low.is_high_confidence());
    }

    #[test]
    fn test_anchor_label() {
        let a = SyncAnchor::new(0, 0, 1.0).with_label("shot cut");
        assert_eq!(a.label.as_deref(), Some("shot cut"));
    }

    #[test]
    fn test_sync_config_tolerance() {
        let cfg = CaptionSyncConfig::new().with_max_drift(300);
        assert_eq!(cfg.max_drift_ms, 300);
        assert_eq!(cfg.tolerance_ms(), 80); // default
    }

    #[test]
    fn test_syncable_cue_duration() {
        let cue = SyncableCue::new(0, 1000, 4000, "text");
        assert_eq!(cue.duration_ms(), 3000);
    }

    #[test]
    fn test_syncable_cue_shift() {
        let mut cue = SyncableCue::new(0, 1000, 3000, "text");
        cue.shift(200);
        assert_eq!(cue.start_ms, 1200);
        assert_eq!(cue.end_ms, 3200);
    }

    #[test]
    fn test_syncer_no_anchors_returns_unchanged() {
        let cues = simple_cues();
        let config = CaptionSyncConfig::new();
        let (synced, report) = CaptionSyncer::new().align(cues.clone(), &[], &config);
        assert_eq!(synced.len(), cues.len());
        assert_eq!(report.anchors_used, 0);
    }

    #[test]
    fn test_syncer_applies_global_offset() {
        let cues = simple_cues();
        let anchors = vec![
            SyncAnchor::new(0, 200, 1.0),     // drift +200
            SyncAnchor::new(2500, 2700, 1.0), // drift +200
        ];
        let config = CaptionSyncConfig::new();
        let (synced, report) = CaptionSyncer::new().align(cues, &anchors, &config);
        // Offset of 200 ms applied
        assert_eq!(synced[0].start_ms, 200);
        assert_eq!(report.anchors_used, 2);
    }

    #[test]
    fn test_sync_report_all_within_limit() {
        let report = SyncReport::new(50, 25, 2, vec![], 3);
        assert!(report.all_within_limit());
        assert_eq!(report.over_limit_count(), 0);
    }

    #[test]
    fn test_sync_report_over_limit_count() {
        let report = SyncReport::new(600, 400, 0, vec![1, 3], 2);
        assert!(!report.all_within_limit());
        assert_eq!(report.over_limit_count(), 2);
    }

    #[test]
    fn test_sync_status_good() {
        let config = CaptionSyncConfig::new();
        let syncer = CaptionSyncer::new();
        assert_eq!(syncer.sync_status(40, &config), SyncStatus::Good);
    }

    #[test]
    fn test_sync_status_acceptable() {
        let config = CaptionSyncConfig::new();
        let syncer = CaptionSyncer::new();
        assert_eq!(syncer.sync_status(200, &config), SyncStatus::Acceptable);
    }

    #[test]
    fn test_sync_status_out_of_sync() {
        let config = CaptionSyncConfig::new();
        let syncer = CaptionSyncer::new();
        assert_eq!(syncer.sync_status(1000, &config), SyncStatus::OutOfSync);
    }

    #[test]
    fn test_sync_status_labels() {
        assert_eq!(SyncStatus::Good.label(), "good");
        assert_eq!(SyncStatus::Acceptable.label(), "acceptable");
        assert_eq!(SyncStatus::OutOfSync.label(), "out-of-sync");
    }

    // -----------------------------------------------------------------------
    // WaveformSync tests
    // -----------------------------------------------------------------------

    /// All-zero PCM input must yield no onsets.
    #[test]
    fn test_detect_onsets_silence() {
        let samples = vec![0.0f32; 16_000]; // 1 second at 16 kHz
        let sync = WaveformSync::new(WaveformSyncConfig::default());
        let onsets = sync.detect_onsets(&samples, 16_000);
        assert!(onsets.is_empty(), "silence should produce zero onsets");
    }

    /// A single burst of energy surrounded by silence should yield exactly one onset.
    #[test]
    fn test_detect_onsets_pulse() {
        let sample_rate: u32 = 16_000;
        let total = sample_rate as usize; // 1 second
        let mut samples = vec![0.0f32; total];
        // Place a 20 ms burst at t = 300 ms (4800 samples).
        let burst_start = (sample_rate as f32 * 0.300) as usize;
        let burst_len = (sample_rate as f32 * 0.020) as usize;
        for i in burst_start..burst_start + burst_len {
            samples[i] = 0.5; // energy = 0.5 >> threshold 0.01
        }
        let sync = WaveformSync::new(WaveformSyncConfig::default());
        let onsets = sync.detect_onsets(&samples, sample_rate);
        assert_eq!(onsets.len(), 1, "one burst → one onset; got: {:?}", onsets);
        // The onset should land near 300 ms (within 1 hop = 10 ms).
        let delta = (onsets[0].time_ms as i64 - 300).abs();
        assert!(
            delta <= 20,
            "onset should be near 300 ms, got {} ms",
            onsets[0].time_ms
        );
    }

    /// Caption already at the onset time → timestamps unchanged.
    #[test]
    fn test_align_to_speech_no_shift_needed() {
        let sync = WaveformSync::new(WaveformSyncConfig::default());
        let onsets = vec![SpeechOnset::new(1000, 0.1)];
        let captions = vec![(1000u64, 3000u64)];
        let aligned = sync.align_to_speech(&captions, &onsets);
        assert_eq!(aligned[0], (1000, 3000));
    }

    /// Caption 200 ms before the onset, which is within max_shift_ms (500) → shifted.
    #[test]
    fn test_align_to_speech_within_threshold() {
        let sync = WaveformSync::new(WaveformSyncConfig::default());
        // Onset at 1200 ms; caption starts at 1000 ms → shift = +200 ms.
        let onsets = vec![SpeechOnset::new(1200, 0.1)];
        let captions = vec![(1000u64, 3000u64)];
        let aligned = sync.align_to_speech(&captions, &onsets);
        assert_eq!(
            aligned[0],
            (1200, 3200),
            "both endpoints must shift by +200 ms"
        );
    }

    /// Onset outside max_shift_ms → caption unchanged.
    #[test]
    fn test_align_to_speech_beyond_max_shift() {
        let config = WaveformSyncConfig {
            max_shift_ms: 100,
            ..Default::default()
        };
        let sync = WaveformSync::new(config);
        let onsets = vec![SpeechOnset::new(1500, 0.1)];
        let captions = vec![(1000u64, 3000u64)]; // shift would be +500 ms > 100 ms limit
        let aligned = sync.align_to_speech(&captions, &onsets);
        assert_eq!(aligned[0], (1000, 3000), "exceeds max_shift_ms → no change");
    }

    /// Empty onset list → captions passed through unchanged.
    #[test]
    fn test_align_to_speech_no_onsets() {
        let sync = WaveformSync::new(WaveformSyncConfig::default());
        let captions = vec![(500u64, 2000u64), (3000u64, 5000u64)];
        let aligned = sync.align_to_speech(&captions, &[]);
        assert_eq!(aligned, captions);
    }

    /// Two bursts well separated (> min_gap_ms) should produce two onsets.
    #[test]
    fn test_detect_onsets_two_separate_bursts() {
        let sample_rate: u32 = 16_000;
        let total = 2 * sample_rate as usize; // 2 seconds
        let mut samples = vec![0.0f32; total];
        // Burst 1 at t = 200 ms; Burst 2 at t = 1200 ms (gap = 1000 ms >> min_gap_ms 200 ms).
        for i in 0..2 {
            let burst_start = (sample_rate as f32 * (0.200 + i as f32 * 1.0)) as usize;
            let burst_len = (sample_rate as f32 * 0.020) as usize;
            for j in burst_start..burst_start + burst_len {
                samples[j] = 0.5;
            }
        }
        let sync = WaveformSync::new(WaveformSyncConfig::default());
        let onsets = sync.detect_onsets(&samples, sample_rate);
        assert_eq!(
            onsets.len(),
            2,
            "two separate bursts → two onsets; got {:?}",
            onsets
        );
    }

    /// Two bursts close together (< min_gap_ms) should be merged into one onset.
    #[test]
    fn test_detect_onsets_merge_close_bursts() {
        let sample_rate: u32 = 16_000;
        let total = sample_rate as usize;
        let mut samples = vec![0.0f32; total];
        // Burst 1 at t = 300 ms; Burst 2 at t = 350 ms (gap = 50 ms < min_gap_ms 200 ms).
        for &start_ms in &[300u32, 350] {
            let burst_start = (sample_rate as f32 * start_ms as f32 / 1000.0) as usize;
            let burst_len = (sample_rate as f32 * 0.015) as usize;
            for j in burst_start..(burst_start + burst_len).min(total) {
                samples[j] = 0.5;
            }
        }
        let sync = WaveformSync::new(WaveformSyncConfig::default());
        let onsets = sync.detect_onsets(&samples, sample_rate);
        assert_eq!(
            onsets.len(),
            1,
            "close bursts should merge; got {:?}",
            onsets
        );
    }
}
