//! Streaming / incremental MIR analysis for real-time use.
//!
//! [`StreamingAnalyzer`] processes audio chunk-by-chunk, maintaining an internal
//! ring-buffer and emitting updated analysis estimates as data accumulates.
//! It is designed for latency-sensitive pipelines where the full signal is not
//! available in advance (e.g. live broadcast, real-time DJ monitoring).
//!
//! # Design
//!
//! * A fixed-size internal overlap-save buffer is maintained.
//! * When the buffer has accumulated at least `min_analysis_samples` samples,
//!   lightweight feature estimates (spectral centroid, ZCR, onset strength)
//!   are updated.
//! * Full analysis (tempo, key, chord) is triggered only when
//!   `full_analysis_samples` have accumulated since the last full run.
//! * All state is stored as plain `Vec<f32>` — no ndarray.

use crate::utils::{hann_window, magnitude_spectrum, mean};
use crate::MirResult;

/// Lightweight per-chunk features updated every `min_analysis_samples`.
#[derive(Debug, Clone, Default)]
pub struct StreamingFrameFeatures {
    /// Spectral centroid estimate (normalised 0–1 relative to Nyquist).
    pub spectral_centroid: f32,
    /// Zero-crossing rate.
    pub zero_crossing_rate: f32,
    /// Onset strength (normalised).
    pub onset_strength: f32,
    /// RMS energy level.
    pub rms_energy: f32,
    /// Number of audio samples analysed so far.
    pub samples_processed: usize,
}

/// Full analysis summary emitted after enough audio has accumulated.
#[derive(Debug, Clone, Default)]
pub struct StreamingAnalysisSummary {
    /// Estimated BPM (0.0 if not yet determined).
    pub bpm: f32,
    /// BPM confidence (0–1).
    pub bpm_confidence: f32,
    /// Whether the performance appears to be rubato.
    pub is_rubato: bool,
    /// Dominant pitch class (0 = C … 11 = B), or 255 if unknown.
    pub dominant_pitch_class: u8,
    /// Onset times (seconds) detected so far.
    pub onset_times: Vec<f32>,
    /// Per-chunk spectral history (centroid, one value per chunk).
    pub centroid_history: Vec<f32>,
    /// Per-chunk RMS history.
    pub rms_history: Vec<f32>,
    /// Total duration analysed (seconds).
    pub duration_secs: f32,
}

/// Configuration for the streaming analyzer.
#[derive(Debug, Clone)]
pub struct StreamingConfig {
    /// Sample rate in Hz.
    pub sample_rate: f32,
    /// Minimum samples to accumulate before computing frame-level features.
    pub min_analysis_samples: usize,
    /// Samples to accumulate before running a full tempo/key analysis.
    pub full_analysis_samples: usize,
    /// FFT window size for spectral analysis.
    pub window_size: usize,
    /// Hop size.
    pub hop_size: usize,
    /// Minimum BPM for tempo estimation.
    pub min_bpm: f32,
    /// Maximum BPM for tempo estimation.
    pub max_bpm: f32,
}

impl Default for StreamingConfig {
    fn default() -> Self {
        Self {
            sample_rate: 44100.0,
            // ~93 ms chunks at 44.1 kHz
            min_analysis_samples: 4096,
            // ~3 seconds worth of audio before full analysis
            full_analysis_samples: 44100 * 3,
            window_size: 2048,
            hop_size: 512,
            min_bpm: 60.0,
            max_bpm: 200.0,
        }
    }
}

/// Incremental streaming MIR analyzer.
///
/// Call [`StreamingAnalyzer::push_chunk`] repeatedly with new audio blocks.
/// After each call, retrieve the lightweight [`StreamingFrameFeatures`] via
/// [`StreamingAnalyzer::frame_features`].  The heavier
/// [`StreamingAnalysisSummary`] is refreshed lazily via
/// [`StreamingAnalyzer::summary`] (it only re-runs when enough new audio
/// has arrived since the last full analysis).
pub struct StreamingAnalyzer {
    config: StreamingConfig,
    /// Internal ring-buffer holding the most recent samples.
    buffer: Vec<f32>,
    /// Total samples pushed.
    total_samples: usize,
    /// Samples at the time of the last full analysis run.
    last_full_analysis_at: usize,
    /// Latest per-frame feature estimates.
    frame_features: StreamingFrameFeatures,
    /// Latest full-analysis summary.
    summary: StreamingAnalysisSummary,
    /// Previous magnitude spectrum (for onset detection).
    prev_magnitude: Vec<f32>,
    /// Accumulated onset samples (for tempo estimation).
    onset_history: Vec<f32>,
    /// Centroid history per processed chunk.
    centroid_history: Vec<f32>,
    /// RMS history per processed chunk.
    rms_history: Vec<f32>,
}

impl StreamingAnalyzer {
    /// Create a new streaming analyzer with the given configuration.
    #[must_use]
    pub fn new(config: StreamingConfig) -> Self {
        let window_size = config.window_size;
        Self {
            config,
            buffer: Vec::with_capacity(window_size * 4),
            total_samples: 0,
            last_full_analysis_at: 0,
            frame_features: StreamingFrameFeatures::default(),
            summary: StreamingAnalysisSummary::default(),
            prev_magnitude: vec![0.0; window_size / 2 + 1],
            onset_history: Vec::new(),
            centroid_history: Vec::new(),
            rms_history: Vec::new(),
        }
    }

    /// Create a streaming analyzer with default config for the given sample rate.
    #[must_use]
    pub fn with_sample_rate(sample_rate: f32) -> Self {
        Self::new(StreamingConfig {
            sample_rate,
            ..StreamingConfig::default()
        })
    }

    /// Push a new chunk of mono audio samples into the analyzer.
    ///
    /// Frame-level features are recomputed on every call.  Full analysis is
    /// triggered automatically when enough samples have accumulated.
    ///
    /// # Errors
    ///
    /// Returns error if internal analysis fails.
    pub fn push_chunk(&mut self, chunk: &[f32]) -> MirResult<()> {
        if chunk.is_empty() {
            return Ok(());
        }

        self.buffer.extend_from_slice(chunk);
        self.total_samples += chunk.len();

        // Keep buffer bounded: retain the most recent `full_analysis_samples`
        // samples plus one extra window.
        let max_buffer = self.config.full_analysis_samples + self.config.window_size;
        if self.buffer.len() > max_buffer {
            let drop = self.buffer.len() - max_buffer;
            self.buffer.drain(..drop);
        }

        // Compute frame-level features on this chunk (even tiny chunks use
        // the RMS path; spectral path requires at least one full window).
        self.update_frame_features(chunk)?;

        // Run full analysis when enough new audio has accumulated.
        let new_samples_since_full = self.total_samples - self.last_full_analysis_at;
        if new_samples_since_full >= self.config.full_analysis_samples {
            self.run_full_analysis()?;
            self.last_full_analysis_at = self.total_samples;
        }

        Ok(())
    }

    /// Return the latest lightweight per-chunk features (updated every call to
    /// `push_chunk`).
    #[must_use]
    pub fn frame_features(&self) -> &StreamingFrameFeatures {
        &self.frame_features
    }

    /// Return the latest full-analysis summary.
    ///
    /// This is recomputed automatically inside `push_chunk` when sufficient
    /// audio has accumulated.
    #[must_use]
    pub fn summary(&self) -> &StreamingAnalysisSummary {
        &self.summary
    }

    /// Total number of samples pushed so far.
    #[must_use]
    pub fn samples_processed(&self) -> usize {
        self.total_samples
    }

    /// Duration analysed so far, in seconds.
    #[must_use]
    pub fn duration_secs(&self) -> f32 {
        self.total_samples as f32 / self.config.sample_rate
    }

    /// Reset all internal state.
    pub fn reset(&mut self) {
        self.buffer.clear();
        self.total_samples = 0;
        self.last_full_analysis_at = 0;
        self.frame_features = StreamingFrameFeatures::default();
        self.summary = StreamingAnalysisSummary::default();
        self.prev_magnitude = vec![0.0; self.config.window_size / 2 + 1];
        self.onset_history.clear();
        self.centroid_history.clear();
        self.rms_history.clear();
    }

    // ── Private helpers ───────────────────────────────────────────────────────

    /// Update lightweight per-chunk features from the latest `chunk`.
    #[allow(clippy::cast_precision_loss)]
    fn update_frame_features(&mut self, chunk: &[f32]) -> MirResult<()> {
        // RMS energy
        let rms = {
            let sq_sum: f32 = chunk.iter().map(|&s| s * s).sum();
            (sq_sum / chunk.len() as f32).sqrt()
        };

        // Zero-crossing rate
        let zcr = if chunk.len() >= 2 {
            let crossings = chunk
                .windows(2)
                .filter(|w| (w[0] >= 0.0) != (w[1] >= 0.0))
                .count();
            crossings as f32 / chunk.len() as f32
        } else {
            0.0
        };

        // Spectral centroid and onset strength — only when chunk is large enough
        let (centroid, onset_strength) = if chunk.len() >= self.config.window_size {
            self.compute_spectral_features(chunk)?
        } else {
            // For tiny chunks use previous values
            (self.frame_features.spectral_centroid, 0.0)
        };

        self.rms_history.push(rms);
        self.centroid_history.push(centroid);

        self.frame_features = StreamingFrameFeatures {
            spectral_centroid: centroid,
            zero_crossing_rate: zcr,
            onset_strength,
            rms_energy: rms,
            samples_processed: self.total_samples,
        };

        Ok(())
    }

    /// Compute spectral centroid and onset strength for a window of audio.
    #[allow(clippy::cast_precision_loss)]
    fn compute_spectral_features(&mut self, chunk: &[f32]) -> MirResult<(f32, f32)> {
        let win = self.config.window_size;
        let hop = self.config.hop_size;

        // Use the last `win` samples of the chunk (or pad with zeros if too short)
        let start = if chunk.len() >= win {
            chunk.len() - win
        } else {
            0
        };
        let frame_slice = &chunk[start..];

        // Apply Hann window
        let window = hann_window(win);
        let windowed: Vec<f32> = frame_slice
            .iter()
            .zip(window.iter().take(frame_slice.len()))
            .map(|(&s, &w)| s * w)
            .chain(std::iter::repeat_n(
                0.0_f32,
                win.saturating_sub(frame_slice.len()),
            ))
            .take(win)
            .collect();

        let fft_input: Vec<oxifft::Complex<f32>> = windowed
            .iter()
            .map(|&s| oxifft::Complex::new(s, 0.0))
            .collect();

        let spectrum = oxifft::fft(&fft_input);
        let mag = magnitude_spectrum(&spectrum);
        let n_bins = mag.len().min(win / 2 + 1);

        let sr = self.config.sample_rate;
        let freq_per_bin = sr / win as f32;

        // Spectral centroid (normalised to Nyquist)
        let (weighted_sum, total_mag) = mag[..n_bins]
            .iter()
            .enumerate()
            .fold((0.0_f32, 0.0_f32), |(ws, tm), (k, &m)| {
                (ws + k as f32 * freq_per_bin * m, tm + m)
            });
        let centroid_hz = if total_mag > 1e-9 {
            weighted_sum / total_mag
        } else {
            0.0
        };
        let centroid_norm = (centroid_hz / (sr * 0.5)).clamp(0.0, 1.0);

        // Onset strength: sum of positive spectral flux
        let prev = &self.prev_magnitude;
        let onset: f32 = mag[..n_bins]
            .iter()
            .zip(prev.iter())
            .map(|(&m, &p)| (m - p).max(0.0))
            .sum();
        let onset_norm = (onset / (n_bins as f32)).clamp(0.0, 1.0);

        // Update previous magnitude
        self.prev_magnitude = mag[..n_bins].to_vec();
        // Pad to expected length if needed
        if self.prev_magnitude.len() < win / 2 + 1 {
            self.prev_magnitude.resize(win / 2 + 1, 0.0);
        }

        // Accumulate onset for tempo estimation (use onset as scalar per frame)
        self.onset_history.push(onset_norm);

        // Keep onset history bounded to full_analysis_samples / hop frames
        let max_frames = self.config.full_analysis_samples / hop + 1;
        if self.onset_history.len() > max_frames {
            let drop = self.onset_history.len() - max_frames;
            self.onset_history.drain(..drop);
        }

        Ok((centroid_norm, onset_norm))
    }

    /// Run a full (heavyweight) tempo + chromagram analysis on the buffered audio.
    #[allow(clippy::cast_precision_loss)]
    fn run_full_analysis(&mut self) -> MirResult<()> {
        let sr = self.config.sample_rate;
        let buf_len = self.buffer.len();

        if buf_len < (sr as usize) {
            // Not enough audio yet for a meaningful full analysis
            return Ok(());
        }

        // ── Tempo from onset autocorrelation ──────────────────────────────
        let (bpm, bpm_confidence, is_rubato) = self.estimate_tempo()?;

        // ── Dominant pitch class from chromagram ──────────────────────────
        let dominant_pitch = self.estimate_dominant_pitch();

        // ── Onset times from onset history ────────────────────────────────
        let hop = self.config.hop_size;
        let onset_times: Vec<f32> = self
            .onset_history
            .iter()
            .enumerate()
            .filter(|(_, &v)| v > 0.1)
            .map(|(i, _)| {
                // Approximate onset time: buffer offset in seconds
                let sample_offset = (buf_len as isize
                    - (self.onset_history.len() as isize - i as isize) * hop as isize)
                    .max(0) as usize;
                (self.total_samples.saturating_sub(buf_len) + sample_offset) as f32 / sr
            })
            .collect();

        self.summary = StreamingAnalysisSummary {
            bpm,
            bpm_confidence,
            is_rubato,
            dominant_pitch_class: dominant_pitch,
            onset_times,
            centroid_history: self.centroid_history.clone(),
            rms_history: self.rms_history.clone(),
            duration_secs: self.total_samples as f32 / sr,
        };

        Ok(())
    }

    /// Estimate tempo from the onset envelope using autocorrelation.
    #[allow(clippy::cast_precision_loss)]
    fn estimate_tempo(&self) -> MirResult<(f32, f32, bool)> {
        if self.onset_history.len() < 16 {
            return Ok((0.0, 0.0, false));
        }

        let acf = crate::utils::autocorrelation(&self.onset_history)
            .unwrap_or_else(|_| vec![0.0; self.onset_history.len()]);

        let sr = self.config.sample_rate;
        let hop = self.config.hop_size as f32;
        let fps = sr / hop; // frames per second

        // Convert BPM range to lag range in frames
        let min_lag = ((fps * 60.0 / self.config.max_bpm) as usize).max(1);
        let max_lag =
            ((fps * 60.0 / self.config.min_bpm) as usize).min(acf.len().saturating_sub(1));

        if min_lag >= max_lag {
            return Ok((0.0, 0.0, false));
        }

        let peaks = crate::utils::find_peaks(&acf[min_lag..max_lag], 3);
        if peaks.is_empty() {
            return Ok((0.0, 0.0, false));
        }

        // Best peak
        let best_peak = peaks
            .iter()
            .copied()
            .max_by(|&a, &b| {
                acf[a + min_lag]
                    .partial_cmp(&acf[b + min_lag])
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
            .unwrap_or(0);

        let lag = best_peak + min_lag;
        let bpm = fps * 60.0 / lag as f32;

        let max_acf = acf[min_lag..max_lag]
            .iter()
            .fold(f32::NEG_INFINITY, |a, &b| a.max(b));
        let confidence = if max_acf > 0.0 {
            (acf[lag] / max_acf).clamp(0.0, 1.0)
        } else {
            0.0
        };

        // Stability: measure CV of inter-onset intervals
        let stability = self.measure_onset_stability(lag);
        let is_rubato = stability < 0.45;

        Ok((bpm, confidence, is_rubato))
    }

    /// Measure onset stability as inverse coefficient-of-variation at the detected period.
    #[allow(clippy::cast_precision_loss)]
    fn measure_onset_stability(&self, period_frames: usize) -> f32 {
        if period_frames == 0 || self.onset_history.len() < period_frames * 2 {
            return 0.5;
        }
        let samples: Vec<f32> = (period_frames..self.onset_history.len())
            .step_by(period_frames)
            .map(|i| self.onset_history[i])
            .collect();
        if samples.is_empty() {
            return 0.5;
        }
        let m = mean(&samples);
        if m < 1e-9 {
            return 0.5;
        }
        let variance: f32 =
            samples.iter().map(|v| (v - m).powi(2)).sum::<f32>() / samples.len() as f32;
        let cv = variance.sqrt() / m;
        (1.0 - cv.min(1.0)).clamp(0.0, 1.0)
    }

    /// Estimate dominant pitch class from the buffered audio chromagram.
    #[allow(clippy::cast_precision_loss)]
    fn estimate_dominant_pitch(&self) -> u8 {
        if self.buffer.len() < self.config.window_size {
            return 255;
        }

        // Use at most the last 2 × full_analysis_samples worth of audio
        let buf = &self.buffer;
        let win = self.config.window_size;
        let hop = self.config.hop_size;
        let sr = self.config.sample_rate as f64;

        // Accumulate chroma bins
        let mut chroma = [0.0_f64; 12];
        let n_frames = (buf.len().saturating_sub(win)) / hop + 1;

        for frame_idx in 0..n_frames {
            let start = frame_idx * hop;
            let end = start + win;
            if end > buf.len() {
                break;
            }
            let frame = &buf[start..end];

            for k in 1..(win / 2) {
                let freq = k as f64 * sr / win as f64;
                if !(65.0..=2093.0).contains(&freq) {
                    continue;
                }
                // Goertzel magnitude estimate
                let omega = 2.0 * std::f64::consts::PI * k as f64 / win as f64;
                let coeff = 2.0 * omega.cos();
                let (mut s1, mut s2) = (0.0_f64, 0.0_f64);
                for &sample in frame {
                    let s0 = f64::from(sample) + coeff * s1 - s2;
                    s2 = s1;
                    s1 = s0;
                }
                let mag = (s1 * s1 + s2 * s2 - coeff * s1 * s2).abs().sqrt();

                // Map to chroma bin
                let midi = 12.0 * (freq / 440.0).log2() + 69.0;
                let pc = (midi.round() as i64).rem_euclid(12) as usize;
                chroma[pc] += mag;
            }
        }

        // Find dominant pitch class
        chroma
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
            .map_or(255, |(i, _)| i as u8)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::f32::consts::TAU;

    fn make_sine(freq: f32, sr: f32, seconds: f32) -> Vec<f32> {
        let n = (sr * seconds) as usize;
        (0..n).map(|i| (TAU * freq * i as f32 / sr).sin()).collect()
    }

    #[test]
    fn test_streaming_analyzer_default() {
        let analyzer = StreamingAnalyzer::with_sample_rate(44100.0);
        assert_eq!(analyzer.samples_processed(), 0);
        assert!((analyzer.duration_secs() - 0.0).abs() < f32::EPSILON);
    }

    #[test]
    fn test_push_empty_chunk() {
        let mut analyzer = StreamingAnalyzer::with_sample_rate(44100.0);
        let result = analyzer.push_chunk(&[]);
        assert!(result.is_ok());
        assert_eq!(analyzer.samples_processed(), 0);
    }

    #[test]
    fn test_push_small_chunk_accumulates() {
        let mut analyzer = StreamingAnalyzer::with_sample_rate(44100.0);
        let chunk = vec![0.0f32; 512];
        let result = analyzer.push_chunk(&chunk);
        assert!(result.is_ok());
        assert_eq!(analyzer.samples_processed(), 512);
    }

    #[test]
    fn test_push_large_chunk_updates_features() {
        let mut analyzer = StreamingAnalyzer::with_sample_rate(44100.0);
        let sine = make_sine(440.0, 44100.0, 1.0);
        let result = analyzer.push_chunk(&sine);
        assert!(result.is_ok());
        assert_eq!(analyzer.samples_processed(), 44100);
        // After a full second of sine at 440 Hz, centroid should be non-zero
        assert!(analyzer.frame_features().spectral_centroid > 0.0);
    }

    #[test]
    fn test_reset_clears_state() {
        let mut analyzer = StreamingAnalyzer::with_sample_rate(44100.0);
        let sine = make_sine(440.0, 44100.0, 0.1);
        let _ = analyzer.push_chunk(&sine);
        assert!(analyzer.samples_processed() > 0);
        analyzer.reset();
        assert_eq!(analyzer.samples_processed(), 0);
        assert_eq!(analyzer.frame_features().samples_processed, 0);
    }

    #[test]
    fn test_streaming_multiple_chunks() {
        let mut analyzer = StreamingAnalyzer::with_sample_rate(44100.0);
        let chunk_size = 4096_usize;
        // Push 20 chunks × 4096 samples = ~80k samples of sine
        let sine = make_sine(220.0, 44100.0, 8.0);
        let mut offset = 0;
        while offset + chunk_size <= sine.len() {
            analyzer
                .push_chunk(&sine[offset..offset + chunk_size])
                .expect("push failed");
            offset += chunk_size;
        }
        assert!(analyzer.samples_processed() >= 20 * chunk_size);
        // Centroid history should have accumulated entries
        assert!(!analyzer.summary().centroid_history.is_empty());
    }

    #[test]
    fn test_full_analysis_triggers_on_threshold() {
        let config = StreamingConfig {
            sample_rate: 44100.0,
            // Require only 2 seconds of audio before full analysis
            full_analysis_samples: 44100 * 2,
            min_analysis_samples: 4096,
            window_size: 2048,
            hop_size: 512,
            min_bpm: 60.0,
            max_bpm: 200.0,
        };
        let mut analyzer = StreamingAnalyzer::new(config);
        let sine = make_sine(440.0, 44100.0, 3.0);
        analyzer.push_chunk(&sine).expect("push failed");
        // After 3 s the full analysis should have run and duration is set
        assert!(analyzer.summary().duration_secs > 0.0);
    }

    #[test]
    fn test_zcr_silent_signal() {
        let mut analyzer = StreamingAnalyzer::with_sample_rate(44100.0);
        let silence = vec![0.0f32; 8192];
        analyzer.push_chunk(&silence).expect("push failed");
        // ZCR for DC silence is 0
        assert!((analyzer.frame_features().zero_crossing_rate - 0.0).abs() < 1e-4);
    }
}
