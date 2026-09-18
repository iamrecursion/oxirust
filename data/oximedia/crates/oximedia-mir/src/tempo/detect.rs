//! Tempo detection using autocorrelation and comb filtering.

use crate::tempo::utils::bounded_acf_with_early_exit;
use crate::types::TempoResult;
use crate::utils::{autocorrelation, find_peaks, mean};
use crate::{MirError, MirResult};

/// Tempo detector.
pub struct TempoDetector {
    sample_rate: f32,
    min_bpm: f32,
    max_bpm: f32,
}

impl TempoDetector {
    /// Create a new tempo detector.
    #[must_use]
    pub fn new(sample_rate: f32, min_bpm: f32, max_bpm: f32) -> Self {
        Self {
            sample_rate,
            min_bpm,
            max_bpm,
        }
    }

    /// Detect tempo from audio signal.
    ///
    /// Uses autocorrelation-based method with comb filtering for robustness.
    /// An incremental ACF scan with early-exit is performed first; if a
    /// high-confidence lag is found before exhausting the search range the
    /// more expensive full-scan and peak-finding steps are skipped.
    ///
    /// # Errors
    ///
    /// Returns error if signal is too short or tempo detection fails.
    #[allow(clippy::cast_possible_truncation)]
    #[allow(clippy::cast_sign_loss)]
    #[allow(clippy::cast_precision_loss)]
    pub fn detect(&self, signal: &[f32]) -> MirResult<TempoResult> {
        if signal.len() < self.sample_rate as usize {
            return Err(MirError::InsufficientData(
                "Signal too short for tempo detection".to_string(),
            ));
        }

        // Compute onset strength envelope
        let onset_env = self.compute_onset_envelope(signal)?;

        // Convert BPM range to lag range (in onset-envelope frames)
        let min_lag = self.bpm_to_lag(self.max_bpm);
        let max_lag_raw = self.bpm_to_lag(self.min_bpm);

        if min_lag == 0 || min_lag >= max_lag_raw {
            return Err(MirError::InvalidConfig(
                "Invalid BPM range for tempo detection".to_string(),
            ));
        }

        // --- Fast path: incremental ACF with early-exit -------------------------
        // Threshold 0.8 means we exit as soon as normalised ACF prominence ≥ 0.8.
        // We require at least 8 lags to have been scanned before triggering exit.
        let (early_best_lag, early_conf, _lags_scanned) =
            bounded_acf_with_early_exit(&onset_env, min_lag, max_lag_raw, 0.8, 8);

        if early_conf >= 0.8 {
            // High confidence: skip full ACF + peak-picking, go straight to refine.
            let primary_bpm = self.lag_to_bpm(early_best_lag);
            let refined_bpm = self.refine_with_comb_filter(&onset_env, primary_bpm)?;
            let stability = self.compute_stability(&onset_env, refined_bpm);
            return Ok(TempoResult {
                bpm: refined_bpm,
                confidence: early_conf.clamp(0.0, 1.0),
                stability,
                alternatives: vec![],
            });
        }

        // --- Slow path: full FFT-based autocorrelation + peak picking ------------
        let acf = autocorrelation(&onset_env)?;
        let max_lag = max_lag_raw.min(acf.len() - 1);

        if min_lag >= max_lag {
            return Err(MirError::InvalidConfig(
                "Invalid BPM range for tempo detection".to_string(),
            ));
        }

        // Find peaks in autocorrelation within valid lag range
        let peaks = find_peaks(&acf[min_lag..max_lag], 10);

        if peaks.is_empty() {
            return Err(MirError::AnalysisFailed("No tempo peaks found".to_string()));
        }

        // Convert peaks to BPM and sort by strength
        let mut candidates: Vec<(f32, f32)> = peaks
            .iter()
            .map(|&peak_idx| {
                let lag = peak_idx + min_lag;
                let bpm = self.lag_to_bpm(lag);
                let strength = acf[lag];
                (bpm, strength)
            })
            .collect();

        candidates.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        // Apply comb filtering to refine estimate
        let primary_bpm = candidates[0].0;
        let refined_bpm = self.refine_with_comb_filter(&onset_env, primary_bpm)?;

        // Compute confidence based on peak prominence
        let max_acf = acf[min_lag..max_lag]
            .iter()
            .fold(f32::NEG_INFINITY, |a, &b| a.max(b));
        let confidence = if max_acf > 0.0 {
            candidates[0].1 / max_acf
        } else {
            0.0
        }
        .clamp(0.0, 1.0);

        // Compute tempo stability
        let stability = self.compute_stability(&onset_env, refined_bpm);

        // Get alternative tempo estimates
        let alternatives: Vec<(f32, f32)> = candidates
            .iter()
            .skip(1)
            .take(3)
            .map(|&(bpm, strength)| {
                let alt_conf = if max_acf > 0.0 {
                    strength / max_acf
                } else {
                    0.0
                };
                (bpm, alt_conf)
            })
            .collect();

        Ok(TempoResult {
            bpm: refined_bpm,
            confidence,
            stability,
            alternatives,
        })
    }

    /// Compute onset strength envelope.
    fn compute_onset_envelope(&self, signal: &[f32]) -> MirResult<Vec<f32>> {
        let hop_size = 512;
        let window_size = 2048;

        let stft = crate::utils::stft(signal, window_size, hop_size)?;

        let mut onset_env = Vec::with_capacity(stft.len());
        let mut prev_mag = vec![0.0; window_size / 2 + 1];

        for frame in &stft {
            let mag = crate::utils::magnitude_spectrum(frame);

            // Spectral flux (positive differences)
            let flux: f32 = mag
                .iter()
                .zip(&prev_mag)
                .map(|(m, p)| (m - p).max(0.0))
                .sum();

            onset_env.push(flux);
            prev_mag = mag;
        }

        // Normalize onset envelope
        let max_val = onset_env.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
        if max_val > 0.0 {
            for val in &mut onset_env {
                *val /= max_val;
            }
        }

        Ok(onset_env)
    }

    /// Refine BPM estimate using comb filtering.
    #[allow(clippy::unnecessary_wraps)]
    fn refine_with_comb_filter(&self, onset_env: &[f32], initial_bpm: f32) -> MirResult<f32> {
        // Try small variations around the initial estimate
        let mut best_bpm = initial_bpm;
        let mut best_score = 0.0;

        for factor in [-0.05, -0.02, 0.0, 0.02, 0.05] {
            let test_bpm = initial_bpm * (1.0 + factor);
            let test_period = self.bpm_to_lag(test_bpm);

            // Compute comb filter response
            let mut score = 0.0;
            let mut count = 0;

            for i in 0..onset_env.len() {
                let mut comb_sum = 0.0;
                for harmonic in 1..=4 {
                    let offset = harmonic * test_period;
                    if i >= offset && i < onset_env.len() {
                        comb_sum += onset_env[i - offset];
                    }
                }
                score += onset_env[i] * comb_sum;
                count += 1;
            }

            if count > 0 {
                score /= count as f32;
            }

            if score > best_score {
                best_score = score;
                best_bpm = test_bpm;
            }
        }

        Ok(best_bpm)
    }

    /// Compute tempo stability.
    fn compute_stability(&self, onset_env: &[f32], bpm: f32) -> f32 {
        let period = self.bpm_to_lag(bpm);

        if period == 0 || onset_env.len() < period * 4 {
            return 0.5;
        }

        // Measure consistency of onset strength at regular intervals
        let mut intervals = Vec::new();
        for i in (period..onset_env.len()).step_by(period) {
            intervals.push(onset_env[i]);
        }

        if intervals.is_empty() {
            return 0.5;
        }

        // Stability is inversely related to variance
        let mean_val = mean(&intervals);
        if mean_val == 0.0 {
            return 0.5;
        }

        let variance: f32 = intervals
            .iter()
            .map(|v| (v - mean_val).powi(2))
            .sum::<f32>()
            / intervals.len() as f32;

        let cv = variance.sqrt() / mean_val; // Coefficient of variation

        // Convert to stability score (lower CV = higher stability)
        (1.0 - cv.min(1.0)).clamp(0.0, 1.0)
    }

    /// Convert BPM to lag in samples (for onset envelope at hop rate).
    #[allow(clippy::cast_possible_truncation)]
    #[allow(clippy::cast_sign_loss)]
    fn bpm_to_lag(&self, bpm: f32) -> usize {
        let hop_size = 512.0;
        let beats_per_second = bpm / 60.0;
        let samples_per_beat = self.sample_rate / beats_per_second;
        let frames_per_beat = samples_per_beat / hop_size;
        frames_per_beat.round() as usize
    }

    /// Convert lag to BPM.
    #[allow(clippy::cast_precision_loss)]
    fn lag_to_bpm(&self, lag: usize) -> f32 {
        let hop_size = 512.0;
        let samples = lag as f32 * hop_size;
        let beats_per_second = self.sample_rate / samples;
        beats_per_second * 60.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::structure_analysis::{novelty_curve, pick_boundaries, self_similarity_matrix};
    use crate::tempo::utils::bounded_acf_with_early_exit;

    // ─── existing tests ──────────────────────────────────────────────────────

    #[test]
    fn test_tempo_detector_creation() {
        let detector = TempoDetector::new(44100.0, 60.0, 200.0);
        assert_eq!(detector.sample_rate, 44100.0);
    }

    #[test]
    fn test_bpm_lag_conversion() {
        let detector = TempoDetector::new(44100.0, 60.0, 200.0);
        let bpm = 120.0;
        let lag = detector.bpm_to_lag(bpm);
        let recovered_bpm = detector.lag_to_bpm(lag);
        assert!((recovered_bpm - bpm).abs() < 1.0);
    }

    #[test]
    fn test_detect_insufficient_data() {
        let detector = TempoDetector::new(44100.0, 60.0, 200.0);
        let signal = vec![0.0; 1000]; // Too short
        let result = detector.detect(&signal);
        assert!(result.is_err());
    }

    // ─── Wave 18 Slice B tests ───────────────────────────────────────────────

    /// Build an onset-envelope that fires every `period` frames (in the
    /// hop-rate domain).  This simulates the normalised onset env that
    /// `compute_onset_envelope` would produce for a perfect click track.
    fn synthetic_onset_env(period: usize, n_frames: usize) -> Vec<f32> {
        (0..n_frames)
            .map(|i| if i % period == 0 { 1.0_f32 } else { 0.0 })
            .collect()
    }

    /// 120 BPM click track in onset-envelope space.
    ///
    /// At 44 100 Hz with hop=512:  frames_per_beat = 44100 / (120/60) / 512 ≈ 43.
    /// We use the bounded ACF helper directly on the onset envelope so the test
    /// does not depend on STFT processing.
    #[test]
    fn test_early_exit_clean_tempo() {
        // hop=512, sr=44100 → period for 120 BPM ≈ round(44100/2/512) = 43 frames
        let period = 43_usize;
        let n_frames = 512;
        let onset_env = synthetic_onset_env(period, n_frames);

        // min_lag = lag for 200 BPM ≈ round(44100/200*60/512) = round(25.9) = 26
        // max_lag = lag for 60 BPM  ≈ round(44100/60*60/512)  = round(86.1) = 86
        let min_lag = 26_usize;
        let max_lag = 86_usize;

        let total_range = max_lag - min_lag + 1; // 61 lags

        let (best_lag, conf, lags_scanned) =
            bounded_acf_with_early_exit(&onset_env, min_lag, max_lag, 0.8, 8);

        // The detected lag should correspond to ±1 frame of the true period (43)
        assert!(
            best_lag.abs_diff(period) <= 1,
            "best_lag={best_lag} should be close to {period}"
        );

        // Confidence must be high for a perfect signal
        assert!(conf >= 0.8, "confidence={conf} should be ≥ 0.8");

        // Early-exit: we should have scanned fewer lags than the full range
        assert!(
            lags_scanned < total_range,
            "lags_scanned={lags_scanned} should be < {total_range} (early exit expected)"
        );
    }

    /// For a nearly-white-noise signal the confidence never exceeds 0.99, so
    /// the full lag range is scanned.
    #[test]
    fn test_noisy_still_full_scan() {
        // Deterministic pseudo-noise: vary the signal such that no lag dominates.
        let n: usize = 256;
        let signal: Vec<f32> = (0..n)
            .map(|i| {
                // Combine three incommensurate frequencies so ACF stays low.
                ((i as f32 * 0.37).sin() + (i as f32 * 0.73).sin() + (i as f32 * 1.13).sin()) / 3.0
            })
            .collect();

        let min_lag = 5_usize;
        let max_lag = 50_usize;
        let total_range = max_lag - min_lag + 1;

        let (_lag, _conf, lags_scanned) =
            bounded_acf_with_early_exit(&signal, min_lag, max_lag, 0.99, 1);

        // Should scan the entire range (no early exit at 0.99 threshold)
        assert_eq!(
            lags_scanned, total_range,
            "expected full scan, got lags_scanned={lags_scanned}"
        );
    }

    /// A direct 120 BPM ACF test: synthetic onset env at period=43 frames,
    /// then check the best lag converts to a BPM within ±2 of 120.
    #[test]
    fn test_acf_120bpm_click_track() {
        let period = 43_usize; // ≈ 120 BPM at 44100 Hz, hop=512
        let onset_env = synthetic_onset_env(period, 512);

        let min_lag = 26_usize;
        let max_lag = 86_usize;

        let (best_lag, _conf, _scanned) =
            bounded_acf_with_early_exit(&onset_env, min_lag, max_lag, 0.8, 8);

        // Convert lag back to BPM: bpm = 44100 * 60 / (lag * 512)
        #[allow(clippy::cast_precision_loss)]
        let detected_bpm = 44100.0_f32 * 60.0 / (best_lag as f32 * 512.0);

        assert!(
            (detected_bpm - 120.0).abs() <= 2.0,
            "detected_bpm={detected_bpm:.1} should be within ±2 BPM of 120"
        );
    }

    // ─── Key detection smoke test ─────────────────────────────────────────────

    /// Instantiate KeyDetector and detect key on a simple signal.
    /// We don't assert a specific key, just that the detector returns `Ok`.
    #[test]
    fn test_key_detection() {
        use crate::key::detect::KeyDetector;

        let sample_rate = 22050.0_f32;
        let detector = KeyDetector::new(sample_rate, 4096);

        // C major chord tone frequencies (C4=261.6, E4=329.6, G4=392 Hz)
        let n_samples = 22050_usize; // 1 s
        let signal: Vec<f32> = (0..n_samples)
            .map(|i| {
                let t = i as f32 / sample_rate;
                (2.0 * std::f32::consts::PI * 261.63 * t).sin()
                    + 0.8 * (2.0 * std::f32::consts::PI * 329.63 * t).sin()
                    + 0.8 * (2.0 * std::f32::consts::PI * 392.00 * t).sin()
            })
            .collect();

        let result = detector.detect(&signal);
        assert!(
            result.is_ok(),
            "KeyDetector should succeed, got: {:?}",
            result.err()
        );
        let key = result.expect("checked above");
        assert!(!key.key.is_empty(), "key name must be non-empty");
        assert!(
            key.confidence >= 0.0 && key.confidence <= 1.0,
            "confidence={} out of range",
            key.confidence
        );
    }

    // ─── Chord recognition smoke test ────────────────────────────────────────

    /// ChordRecognizer on a synthetic C major chord signal → recognize returns
    /// Ok and detects at least one chord.
    #[test]
    fn test_chord_recognition() {
        use crate::chord::recognize::ChordRecognizer;

        let sample_rate = 22050.0_f32;
        let recognizer = ChordRecognizer::new(sample_rate, 2048, 512);

        // Build a C major chord: C4 (261.63 Hz) + E4 (329.63 Hz) + G4 (392.00 Hz)
        let n_samples = 22050_usize; // 1 s
        let signal: Vec<f32> = (0..n_samples)
            .map(|i| {
                let t = i as f32 / sample_rate;
                (2.0 * std::f32::consts::PI * 261.63 * t).sin()
                    + 0.8 * (2.0 * std::f32::consts::PI * 329.63 * t).sin()
                    + 0.8 * (2.0 * std::f32::consts::PI * 392.00 * t).sin()
            })
            .collect();

        let result = recognizer.recognize(&signal);
        assert!(
            result.is_ok(),
            "ChordRecognizer should succeed, got: {:?}",
            result.err()
        );
        let chord_result = result.expect("checked above");
        // At least one chord should be detected in 1 second of audio
        assert!(
            !chord_result.chords.is_empty(),
            "should detect at least one chord"
        );
        // The root chord should include "C" (or at minimum be non-empty)
        let root_label = &chord_result.chords[0].label;
        assert!(
            !root_label.is_empty(),
            "chord label must be non-empty, got '{root_label}'"
        );
    }

    // ─── Structure boundary test ──────────────────────────────────────────────

    /// self_similarity_matrix + novelty_curve + pick_boundaries on a 2-section
    /// synthetic feature sequence should detect at least 1 boundary near the
    /// midpoint where the two sections transition.
    ///
    /// Section A: first 10 frames — orthogonal to section B.
    /// Section B: last 10 frames — different direction in feature space.
    ///
    /// Note: the checkerboard kernel may also produce a peak at or near frame 0
    /// (the beginning) due to boundary effects; we only assert that exactly one
    /// boundary falls in the neighbourhood of the midpoint (the real transition).
    #[test]
    fn test_structure_boundaries() {
        // Section A: feature vector [1, 0, 0]
        // Section B: feature vector [0, 1, 0]
        let n = 20_usize;
        let half = n / 2; // 10
        let features: Vec<Vec<f64>> = (0..n)
            .map(|i| {
                if i < half {
                    vec![1.0, 0.0, 0.0]
                } else {
                    vec![0.0, 1.0, 0.0]
                }
            })
            .collect();

        let ssm = self_similarity_matrix(&features);
        assert_eq!(ssm.len(), n, "SSM must be {n}×{n}");

        // kernel_size=4 gives a wider checkerboard that clearly separates the two
        // uniform sections and concentrates novelty at the midpoint.
        let novelty = novelty_curve(&ssm, 4);
        assert_eq!(novelty.len(), n, "novelty curve length must equal n");

        // Require min_gap=3 so the frame-0 boundary-effect peak (if present) is
        // counted separately from the midpoint peak.
        let boundaries = pick_boundaries(&novelty, 0.01, 3);

        // At least one detected boundary must be near the midpoint.
        let near_mid = boundaries
            .iter()
            .any(|&b| b >= half.saturating_sub(3) && b <= half + 3);
        assert!(
            near_mid,
            "expected a boundary near midpoint {half}, detected boundaries: {boundaries:?}"
        );
    }
}
