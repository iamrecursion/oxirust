//! Beat tracking using auto-correlation and onset detection.
//!
//! Implements a full beat-tracking pipeline:
//! 1. **Onset detection** – High Frequency Content (HFC) based onset strength function.
//! 2. **Tempo estimation** – Auto-correlation over the BPM-restricted lag range.
//! 3. **Beat picking** – Dynamic-programming beat grid optimisation.
//! 4. **Downbeat detection** – Every 4th beat is promoted to downbeat status.

#![allow(dead_code)]
#![allow(clippy::cast_precision_loss)]
#![allow(clippy::cast_possible_truncation)]
#![allow(clippy::cast_sign_loss)]

// ── Public data types ─────────────────────────────────────────────────────────

/// Configuration and runtime state for the beat tracker.
#[derive(Debug, Clone)]
pub struct BeatTracker {
    /// Audio sample rate in Hz.
    pub sample_rate: u32,
    /// Hop size in samples between analysis frames.
    pub hop_size: usize,
    /// Minimum detectable tempo (BPM).
    pub min_bpm: f32,
    /// Maximum detectable tempo (BPM).
    pub max_bpm: f32,
}

/// A single detected beat event.
#[derive(Debug, Clone)]
pub struct BeatEvent {
    /// Beat position in seconds from the start of the audio.
    pub time_secs: f64,
    /// Confidence score in the range [0.0, 1.0].
    pub confidence: f32,
    /// Whether this beat is a downbeat (first beat of a measure).
    pub is_downbeat: bool,
}

/// Full output of the beat-tracking pipeline.
#[derive(Debug, Clone)]
pub struct BeatTrackingResult {
    /// Estimated global tempo in beats per minute.
    pub tempo_bpm: f32,
    /// Confidence in the tempo estimate (0.0–1.0).
    pub tempo_confidence: f32,
    /// All detected beat events in chronological order.
    pub beats: Vec<BeatEvent>,
    /// Time-signature numerator (e.g. 4 for 4/4).
    pub time_signature_numerator: u32,
    /// Time-signature denominator (e.g. 4 for 4/4).
    pub time_signature_denominator: u32,
}

// ── BeatTracker implementation ────────────────────────────────────────────────

impl BeatTracker {
    /// Create a beat tracker with sensible defaults (hop 512, 60–200 BPM).
    #[must_use]
    pub fn new(sample_rate: u32) -> Self {
        Self {
            sample_rate,
            hop_size: 512,
            min_bpm: 60.0,
            max_bpm: 200.0,
        }
    }

    /// Create a beat tracker with explicit parameters.
    #[must_use]
    pub fn with_params(sample_rate: u32, hop_size: usize, min_bpm: f32, max_bpm: f32) -> Self {
        Self {
            sample_rate,
            hop_size,
            min_bpm,
            max_bpm,
        }
    }

    // ── Onset detection ───────────────────────────────────────────────────────

    /// Compute the onset strength function from raw audio samples.
    ///
    /// Uses High Frequency Content (HFC) approximated as the positive
    /// first-difference of the log-energy envelope.
    ///
    /// Returns one value per analysis frame (length = `ceil(samples.len() / hop_size)`).
    #[must_use]
    pub fn detect_onsets(&self, samples: &[f32]) -> Vec<f32> {
        if samples.is_empty() {
            return Vec::new();
        }

        let hop = self.hop_size.max(1);
        let n_frames = (samples.len() + hop - 1) / hop;

        // --- Frame-by-frame RMS energy ---
        let mut energy: Vec<f32> = (0..n_frames)
            .map(|i| {
                let start = i * hop;
                let end = (start + hop).min(samples.len());
                let frame = &samples[start..end];
                if frame.is_empty() {
                    return 0.0_f32;
                }
                let sum_sq: f32 = frame.iter().map(|x| x * x).sum();
                (sum_sq / frame.len() as f32).sqrt()
            })
            .collect();

        // --- Log energy (avoid log(0) with a small floor) ---
        const LOG_FLOOR: f32 = 1e-8;
        let log_energy: Vec<f32> = energy.iter().map(|&e| (e + LOG_FLOOR).ln()).collect();

        // --- Positive first-difference → onset strength ---
        let mut onset: Vec<f32> = vec![0.0; n_frames];
        for i in 1..n_frames {
            let diff = log_energy[i] - log_energy[i - 1];
            onset[i] = diff.max(0.0);
        }

        // --- Normalise to [0, 1] ---
        let max_val = onset.iter().cloned().fold(0.0_f32, f32::max);
        if max_val > 0.0 {
            for v in &mut onset {
                *v /= max_val;
            }
        }

        // Clear the energy vector explicitly (not needed but suppresses lint)
        energy.clear();

        onset
    }

    // ── Tempo estimation ──────────────────────────────────────────────────────

    /// Estimate tempo via auto-correlation of the onset strength function.
    ///
    /// Returns `(bpm, confidence)`.  Confidence is the normalised peak height.
    #[must_use]
    pub fn estimate_tempo(&self, onset_strength: &[f32]) -> (f32, f32) {
        if onset_strength.len() < 2 {
            return (self.min_bpm, 0.0);
        }

        let sr = self.sample_rate as f32;
        let hop = self.hop_size as f32;

        // Convert BPM bounds to frame-lag bounds
        let frames_per_beat_min = (60.0 * sr / (self.max_bpm * hop)).round() as usize;
        let frames_per_beat_max = (60.0 * sr / (self.min_bpm * hop)).round() as usize;
        let frames_per_beat_min = frames_per_beat_min.max(1);
        let frames_per_beat_max = frames_per_beat_max.min(onset_strength.len() - 1);

        if frames_per_beat_min >= frames_per_beat_max {
            return (self.min_bpm, 0.0);
        }

        let n = onset_strength.len();

        // --- Auto-correlation ---
        let mut best_lag = frames_per_beat_min;
        let mut best_corr = f32::NEG_INFINITY;

        for lag in frames_per_beat_min..=frames_per_beat_max {
            let mut corr = 0.0_f32;
            let mut count = 0_usize;
            for i in 0..(n - lag) {
                corr += onset_strength[i] * onset_strength[i + lag];
                count += 1;
            }
            if count > 0 {
                corr /= count as f32;
            }
            if corr > best_corr {
                best_corr = corr;
                best_lag = lag;
            }
        }

        // --- Confidence: normalise peak by total auto-correlation power ---
        let zero_lag: f32 =
            onset_strength.iter().map(|x| x * x).sum::<f32>() / onset_strength.len() as f32;
        let confidence = if zero_lag > 0.0 {
            (best_corr / zero_lag).clamp(0.0, 1.0)
        } else {
            0.0
        };

        let bpm = 60.0 * sr / (best_lag as f32 * hop);
        let bpm = bpm.clamp(self.min_bpm, self.max_bpm);

        (bpm, confidence)
    }

    // ── Beat picking ──────────────────────────────────────────────────────────

    /// Dynamic-programming beat picker.
    ///
    /// Score function: `onset_strength[t] − lambda * (deviation_from_period)^2 / period^2`
    ///
    /// Returns [`BeatEvent`] values without downbeat annotation.
    #[must_use]
    pub fn pick_beats(&self, onset_strength: &[f32], tempo_bpm: f32) -> Vec<BeatEvent> {
        if onset_strength.is_empty() || tempo_bpm <= 0.0 {
            return Vec::new();
        }

        let sr = self.sample_rate as f32;
        let hop = self.hop_size as f32;
        let period = 60.0 * sr / (tempo_bpm * hop); // in frames
        let period_frames = period.round() as usize;
        if period_frames == 0 {
            return Vec::new();
        }

        let n = onset_strength.len();
        let lambda = 100.0_f32; // DP penalty weight

        // DP arrays
        let mut score: Vec<f32> = vec![f32::NEG_INFINITY; n];
        let mut prev: Vec<i64> = vec![-1; n];

        // Seed: any frame can be the first beat
        for i in 0..n {
            score[i] = onset_strength[i];
        }

        // Fill DP table: for each frame t, consider all predecessor frames p
        // such that |t - p - period_frames| is small.
        let search_window = (period_frames / 2).max(1);
        for t in 1..n {
            let lo = if t > period_frames + search_window {
                t - period_frames - search_window
            } else {
                0
            };
            let hi = if t > period_frames {
                (t - period_frames + search_window).min(t - 1)
            } else {
                0
            };

            let mut best_pred_score = f32::NEG_INFINITY;
            let mut best_pred: i64 = -1;

            for p in lo..=hi {
                if p >= t {
                    continue;
                }
                let dev = (t as f32 - p as f32) - period;
                let penalty = lambda * (dev * dev) / (period * period);
                let candidate = score[p] - penalty;
                if candidate > best_pred_score {
                    best_pred_score = candidate;
                    best_pred = p as i64;
                }
            }

            if best_pred >= 0 {
                let new_score = onset_strength[t] + best_pred_score;
                if new_score > score[t] {
                    score[t] = new_score;
                    prev[t] = best_pred;
                }
            }
        }

        // --- Backtrack from best final state ---
        let end = score
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(i, _)| i)
            .unwrap_or(0);

        let mut beat_frames: Vec<usize> = Vec::new();
        let mut cur = end as i64;
        while cur >= 0 {
            beat_frames.push(cur as usize);
            let p = prev[cur as usize];
            if p < 0 || p == cur {
                break;
            }
            cur = p;
        }
        beat_frames.reverse();

        // --- Convert frames → BeatEvents ---
        let seconds_per_frame = hop / sr;
        let max_onset = onset_strength
            .iter()
            .cloned()
            .fold(0.0_f32, f32::max)
            .max(1e-8);

        beat_frames
            .into_iter()
            .map(|f| BeatEvent {
                time_secs: f as f64 * seconds_per_frame as f64,
                confidence: (onset_strength[f] / max_onset).clamp(0.0, 1.0),
                is_downbeat: false,
            })
            .collect()
    }

    // ── Downbeat detection ────────────────────────────────────────────────────

    /// Mark every 4th beat as a downbeat (simple heuristic for 4/4 time).
    ///
    /// Replaces the `is_downbeat` field in-place and returns an updated copy.
    #[must_use]
    pub fn detect_downbeats(&self, beats: &[BeatEvent], _onset_strength: &[f32]) -> Vec<BeatEvent> {
        beats
            .iter()
            .enumerate()
            .map(|(i, b)| BeatEvent {
                time_secs: b.time_secs,
                confidence: b.confidence,
                is_downbeat: i % 4 == 0,
            })
            .collect()
    }

    // ── Full pipeline ─────────────────────────────────────────────────────────

    /// Run the complete beat-tracking pipeline on raw audio samples.
    ///
    /// Pipeline: `detect_onsets` → `estimate_tempo` → `pick_beats` → `detect_downbeats`.
    #[must_use]
    pub fn analyze(&self, samples: &[f32]) -> BeatTrackingResult {
        let onset_strength = self.detect_onsets(samples);

        let (tempo_bpm, tempo_confidence) = if onset_strength.is_empty() {
            (120.0, 0.0)
        } else {
            self.estimate_tempo(&onset_strength)
        };

        let raw_beats = self.pick_beats(&onset_strength, tempo_bpm);
        let beats = self.detect_downbeats(&raw_beats, &onset_strength);

        BeatTrackingResult {
            tempo_bpm,
            tempo_confidence,
            beats,
            time_signature_numerator: 4,
            time_signature_denominator: 4,
        }
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── construction ──────────────────────────────────────────────────────────

    #[test]
    fn test_new_default_params() {
        let bt = BeatTracker::new(44100);
        assert_eq!(bt.sample_rate, 44100);
        assert_eq!(bt.hop_size, 512);
        assert!((bt.min_bpm - 60.0).abs() < 1e-4);
        assert!((bt.max_bpm - 200.0).abs() < 1e-4);
    }

    #[test]
    fn test_with_params() {
        let bt = BeatTracker::with_params(22050, 256, 80.0, 180.0);
        assert_eq!(bt.hop_size, 256);
        assert!((bt.min_bpm - 80.0).abs() < 1e-4);
    }

    // ── detect_onsets ─────────────────────────────────────────────────────────

    #[test]
    fn test_detect_onsets_empty() {
        let bt = BeatTracker::new(44100);
        let onset = bt.detect_onsets(&[]);
        assert!(onset.is_empty());
    }

    #[test]
    fn test_detect_onsets_silence_range() {
        let bt = BeatTracker::new(44100);
        let samples = vec![0.0_f32; 44100];
        let onset = bt.detect_onsets(&samples);
        // For silence all values must be in [0,1]
        for &v in &onset {
            assert!(v >= 0.0 && v <= 1.0, "out of range: {v}");
        }
    }

    #[test]
    fn test_detect_onsets_frame_count() {
        let bt = BeatTracker::new(44100);
        let samples = vec![0.1_f32; 4096];
        let onset = bt.detect_onsets(&samples);
        let expected_frames = (4096 + 511) / 512; // ceil(4096/512)
        assert_eq!(onset.len(), expected_frames);
    }

    #[test]
    fn test_detect_onsets_normalised_max_one() {
        let bt = BeatTracker::new(44100);
        // Create a spike followed by silence
        let mut samples = vec![0.0_f32; 4096];
        for s in &mut samples[512..1024] {
            *s = 1.0;
        }
        let onset = bt.detect_onsets(&samples);
        let max_val = onset.iter().cloned().fold(0.0_f32, f32::max);
        // max should be 1.0 when there is a non-trivial signal
        assert!((max_val - 1.0).abs() < 1e-5 || max_val == 0.0);
    }

    // ── estimate_tempo ────────────────────────────────────────────────────────

    #[test]
    fn test_estimate_tempo_empty() {
        let bt = BeatTracker::new(44100);
        let (bpm, conf) = bt.estimate_tempo(&[]);
        assert!(bpm >= bt.min_bpm);
        assert!((conf - 0.0).abs() < 1e-5);
    }

    #[test]
    fn test_estimate_tempo_periodic_signal_120bpm() {
        // Build synthetic onset-strength at 120 BPM with hop 512 @ 44100 Hz
        // period_frames ≈ 60 * 44100 / (120 * 512) ≈ 43.07 → 43 frames
        let bt = BeatTracker::new(44100);
        let period = 43_usize;
        let n = 512_usize;
        let mut onset = vec![0.0_f32; n];
        for i in (0..n).step_by(period) {
            onset[i] = 1.0;
        }
        let (bpm, _conf) = bt.estimate_tempo(&onset);
        // We expect something in the 60-200 BPM range
        assert!(bpm >= bt.min_bpm && bpm <= bt.max_bpm, "bpm={bpm}");
    }

    #[test]
    fn test_estimate_tempo_confidence_range() {
        let bt = BeatTracker::new(44100);
        let onset = vec![0.5_f32; 200];
        let (_bpm, conf) = bt.estimate_tempo(&onset);
        assert!(conf >= 0.0 && conf <= 1.0, "conf={conf}");
    }

    // ── pick_beats ────────────────────────────────────────────────────────────

    #[test]
    fn test_pick_beats_empty_onset() {
        let bt = BeatTracker::new(44100);
        let beats = bt.pick_beats(&[], 120.0);
        assert!(beats.is_empty());
    }

    #[test]
    fn test_pick_beats_zero_bpm() {
        let bt = BeatTracker::new(44100);
        let onset = vec![0.5_f32; 100];
        let beats = bt.pick_beats(&onset, 0.0);
        assert!(beats.is_empty());
    }

    #[test]
    fn test_pick_beats_produces_beats() {
        let bt = BeatTracker::new(44100);
        let onset = vec![0.5_f32; 300];
        let beats = bt.pick_beats(&onset, 120.0);
        // Should produce at least one beat
        assert!(!beats.is_empty());
    }

    #[test]
    fn test_pick_beats_confidence_in_range() {
        let bt = BeatTracker::new(44100);
        let onset = vec![0.8_f32; 300];
        let beats = bt.pick_beats(&onset, 120.0);
        for b in &beats {
            assert!(
                b.confidence >= 0.0 && b.confidence <= 1.0,
                "confidence={}",
                b.confidence
            );
        }
    }

    // ── detect_downbeats ──────────────────────────────────────────────────────

    #[test]
    fn test_downbeats_every_4th() {
        let bt = BeatTracker::new(44100);
        let beats: Vec<BeatEvent> = (0..12)
            .map(|i| BeatEvent {
                time_secs: i as f64 * 0.5,
                confidence: 0.9,
                is_downbeat: false,
            })
            .collect();
        let onset = vec![0.5_f32; 100];
        let result = bt.detect_downbeats(&beats, &onset);
        assert_eq!(result.len(), 12);
        for (i, b) in result.iter().enumerate() {
            if i % 4 == 0 {
                assert!(b.is_downbeat, "beat {i} should be downbeat");
            } else {
                assert!(!b.is_downbeat, "beat {i} should not be downbeat");
            }
        }
    }

    // ── analyze (full pipeline) ───────────────────────────────────────────────

    #[test]
    fn test_analyze_silence() {
        let bt = BeatTracker::new(44100);
        let samples = vec![0.0_f32; 44100];
        let result = bt.analyze(&samples);
        assert!(result.tempo_bpm >= bt.min_bpm && result.tempo_bpm <= bt.max_bpm);
        assert!(result.tempo_confidence >= 0.0 && result.tempo_confidence <= 1.0);
        assert_eq!(result.time_signature_numerator, 4);
        assert_eq!(result.time_signature_denominator, 4);
    }

    #[test]
    fn test_analyze_produces_time_signature_44() {
        let bt = BeatTracker::new(44100);
        let samples = vec![0.1_f32; 22050];
        let result = bt.analyze(&samples);
        assert_eq!(result.time_signature_numerator, 4);
        assert_eq!(result.time_signature_denominator, 4);
    }
}
