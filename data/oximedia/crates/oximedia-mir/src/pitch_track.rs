//! Frame-by-frame fundamental-frequency (F0) tracking.
//!
//! Implements a simplified autocorrelation-based pitch tracker that operates
//! on overlapping frames of a mono audio signal and produces a time-series of
//! pitch estimates with voicing confidence.

#![allow(dead_code)]

// ---------------------------------------------------------------------------
// PitchEstimate
// ---------------------------------------------------------------------------

/// A single frame's pitch estimate.
#[derive(Debug, Clone, Copy)]
pub struct PitchEstimate {
    /// Estimated fundamental frequency in Hz, or 0.0 if unvoiced.
    pub frequency_hz: f32,
    /// Confidence in `[0.0, 1.0]` that the frame is voiced.
    pub confidence: f32,
    /// Time offset of the frame centre in seconds.
    pub time_sec: f32,
}

impl PitchEstimate {
    /// Whether this frame is considered voiced (confidence above threshold).
    #[must_use]
    pub fn is_voiced(&self, threshold: f32) -> bool {
        self.confidence >= threshold
    }

    /// MIDI note number (A4 = 69, 440 Hz).  Returns `None` if frequency is zero.
    #[must_use]
    #[allow(clippy::cast_precision_loss)]
    pub fn midi_note(&self) -> Option<f32> {
        if self.frequency_hz <= 0.0 {
            return None;
        }
        Some(69.0 + 12.0 * (self.frequency_hz / 440.0).log2())
    }
}

// ---------------------------------------------------------------------------
// PitchTrackerConfig
// ---------------------------------------------------------------------------

/// Configuration for the pitch tracker.
#[derive(Debug, Clone)]
pub struct PitchTrackerConfig {
    /// Sample rate in Hz.
    pub sample_rate: f32,
    /// Analysis window size in samples.
    pub window_size: usize,
    /// Hop size in samples.
    pub hop_size: usize,
    /// Minimum detectable frequency in Hz.
    pub min_freq: f32,
    /// Maximum detectable frequency in Hz.
    pub max_freq: f32,
    /// Voicing confidence threshold.
    pub voicing_threshold: f32,
}

impl PitchTrackerConfig {
    /// Create a config tuned for speech (80–600 Hz).
    #[must_use]
    pub fn speech() -> Self {
        Self {
            sample_rate: 44100.0,
            window_size: 2048,
            hop_size: 512,
            min_freq: 80.0,
            max_freq: 600.0,
            voicing_threshold: 0.3,
        }
    }

    /// Create a config tuned for music (50–4000 Hz).
    #[must_use]
    pub fn music() -> Self {
        Self {
            sample_rate: 44100.0,
            window_size: 2048,
            hop_size: 512,
            min_freq: 50.0,
            max_freq: 4000.0,
            voicing_threshold: 0.25,
        }
    }
}

impl Default for PitchTrackerConfig {
    fn default() -> Self {
        Self::music()
    }
}

// ---------------------------------------------------------------------------
// PitchTracker
// ---------------------------------------------------------------------------

/// Autocorrelation-based pitch tracker.
pub struct PitchTracker {
    config: PitchTrackerConfig,
}

impl PitchTracker {
    /// Create a new tracker.
    #[must_use]
    pub fn new(config: PitchTrackerConfig) -> Self {
        Self { config }
    }

    /// Run the tracker over a complete mono signal, returning one
    /// [`PitchEstimate`] per hop-sized frame.
    #[must_use]
    #[allow(clippy::cast_precision_loss)]
    pub fn track(&self, samples: &[f32]) -> Vec<PitchEstimate> {
        if samples.is_empty() || self.config.hop_size == 0 || self.config.window_size == 0 {
            return Vec::new();
        }
        let n_frames =
            samples.len().saturating_sub(self.config.window_size) / self.config.hop_size + 1;
        let mut estimates = Vec::with_capacity(n_frames);

        let min_lag = (self.config.sample_rate / self.config.max_freq).floor() as usize;
        let max_lag = (self.config.sample_rate / self.config.min_freq).ceil() as usize;

        for i in 0..n_frames {
            let start = i * self.config.hop_size;
            let end = (start + self.config.window_size).min(samples.len());
            let frame = &samples[start..end];
            let (freq, conf) = self.estimate_frame(frame, min_lag, max_lag);
            let time = (start + self.config.window_size / 2) as f32 / self.config.sample_rate;
            estimates.push(PitchEstimate {
                frequency_hz: freq,
                confidence: conf,
                time_sec: time,
            });
        }
        estimates
    }

    /// Estimate pitch for a single frame using normalised autocorrelation.
    #[allow(clippy::cast_precision_loss)]
    fn estimate_frame(&self, frame: &[f32], min_lag: usize, max_lag: usize) -> (f32, f32) {
        let n = frame.len();
        if n == 0 || max_lag >= n {
            return (0.0, 0.0);
        }

        // Energy of the frame
        let energy: f32 = frame.iter().map(|&s| s * s).sum();
        if energy < 1e-12 {
            return (0.0, 0.0);
        }

        // First pass: compute normalised autocorrelation for all lags
        let max_lag_clamped = max_lag.min(n - 1);
        let mut corr_values = vec![0.0_f32; max_lag_clamped + 1];

        for lag in min_lag..=max_lag_clamped {
            let mut num: f32 = 0.0;
            let mut den_a: f32 = 0.0;
            let mut den_b: f32 = 0.0;
            for j in 0..n - lag {
                num += frame[j] * frame[j + lag];
                den_a += frame[j] * frame[j];
                den_b += frame[j + lag] * frame[j + lag];
            }
            let denom = (den_a * den_b).sqrt();
            corr_values[lag] = if denom > 1e-12 { num / denom } else { 0.0 };
        }

        // Find global maximum correlation
        let best_corr: f32 = corr_values[min_lag..=max_lag_clamped]
            .iter()
            .copied()
            .fold(-1.0f32, f32::max);

        // Second pass: pick the *first* lag whose correlation is within 10%
        // of the global maximum, preferring the fundamental over sub-harmonics.
        let threshold = best_corr * 0.9;
        let mut best_lag = 0usize;
        let mut best_corr = best_corr;
        if let Some((offset, &val)) = corr_values[min_lag..=max_lag_clamped]
            .iter()
            .enumerate()
            .find(|(_, &v)| v >= threshold)
        {
            best_lag = min_lag + offset;
            best_corr = val;
        }

        let confidence = best_corr.max(0.0);
        if confidence < self.config.voicing_threshold || best_lag == 0 {
            return (0.0, confidence);
        }

        let freq = self.config.sample_rate / best_lag as f32;
        (freq, confidence)
    }

    /// Get a reference to the configuration.
    #[must_use]
    pub fn config(&self) -> &PitchTrackerConfig {
        &self.config
    }

    /// Average voiced frequency over all frames above the voicing threshold.
    /// Returns `None` if no frames are voiced.
    #[must_use]
    #[allow(clippy::cast_precision_loss)]
    pub fn average_pitch(estimates: &[PitchEstimate], threshold: f32) -> Option<f32> {
        let voiced: Vec<f32> = estimates
            .iter()
            .filter(|e| e.is_voiced(threshold) && e.frequency_hz > 0.0)
            .map(|e| e.frequency_hz)
            .collect();
        if voiced.is_empty() {
            return None;
        }
        Some(voiced.iter().sum::<f32>() / voiced.len() as f32)
    }

    /// Pitch range (min, max) among voiced frames.
    #[must_use]
    pub fn pitch_range(estimates: &[PitchEstimate], threshold: f32) -> Option<(f32, f32)> {
        let voiced: Vec<f32> = estimates
            .iter()
            .filter(|e| e.is_voiced(threshold) && e.frequency_hz > 0.0)
            .map(|e| e.frequency_hz)
            .collect();
        if voiced.is_empty() {
            return None;
        }
        let min = voiced.iter().copied().fold(f32::INFINITY, f32::min);
        let max = voiced.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        Some((min, max))
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    /// Generate a pure sine wave.
    #[allow(clippy::cast_precision_loss)]
    fn sine(freq: f32, sr: f32, n_samples: usize) -> Vec<f32> {
        (0..n_samples)
            .map(|i| (2.0 * std::f32::consts::PI * freq * i as f32 / sr).sin())
            .collect()
    }

    #[test]
    fn test_pitch_estimate_voiced() {
        let e = PitchEstimate {
            frequency_hz: 440.0,
            confidence: 0.9,
            time_sec: 0.0,
        };
        assert!(e.is_voiced(0.5));
        assert!(!e.is_voiced(0.95));
    }

    #[test]
    fn test_midi_note_a4() {
        let e = PitchEstimate {
            frequency_hz: 440.0,
            confidence: 1.0,
            time_sec: 0.0,
        };
        let note = e.midi_note().expect("should succeed in test");
        assert!((note - 69.0).abs() < 0.01);
    }

    #[test]
    fn test_midi_note_zero() {
        let e = PitchEstimate {
            frequency_hz: 0.0,
            confidence: 0.0,
            time_sec: 0.0,
        };
        assert!(e.midi_note().is_none());
    }

    #[test]
    fn test_config_speech_defaults() {
        let c = PitchTrackerConfig::speech();
        assert_eq!(c.min_freq, 80.0);
        assert_eq!(c.max_freq, 600.0);
    }

    #[test]
    fn test_config_music_defaults() {
        let c = PitchTrackerConfig::music();
        assert_eq!(c.min_freq, 50.0);
        assert_eq!(c.max_freq, 4000.0);
    }

    #[test]
    fn test_track_empty() {
        let t = PitchTracker::new(PitchTrackerConfig::default());
        assert!(t.track(&[]).is_empty());
    }

    #[test]
    fn test_track_silence() {
        let t = PitchTracker::new(PitchTrackerConfig::default());
        let silence = vec![0.0f32; 4096];
        let est = t.track(&silence);
        assert!(!est.is_empty());
        // All should be unvoiced (low confidence)
        for e in &est {
            assert!(e.frequency_hz < 1.0 || e.confidence < 0.25);
        }
    }

    #[test]
    fn test_track_sine_440() {
        // Use a lower sample rate and smaller window to reduce the O(N * lag_range)
        // autocorrelation cost while still detecting 440 Hz reliably.
        let sr = 8000.0;
        // 0.5 s at 8 kHz = 4000 samples; enough for several frames.
        let sig = sine(440.0, sr, 4000);
        let cfg = PitchTrackerConfig {
            sample_rate: sr,
            window_size: 512,
            hop_size: 128,
            min_freq: 200.0,
            max_freq: 1000.0,
            voicing_threshold: 0.3,
        };
        let t = PitchTracker::new(cfg);
        let est = t.track(&sig);
        // At least some frames should detect ~440 Hz
        let voiced: Vec<&PitchEstimate> = est.iter().filter(|e| e.confidence > 0.5).collect();
        assert!(!voiced.is_empty());
        for e in &voiced {
            assert!(
                (e.frequency_hz - 440.0).abs() < 50.0,
                "freq {} too far from 440",
                e.frequency_hz
            );
        }
    }

    #[test]
    fn test_average_pitch_none_on_silence() {
        let est = vec![PitchEstimate {
            frequency_hz: 0.0,
            confidence: 0.1,
            time_sec: 0.0,
        }];
        assert!(PitchTracker::average_pitch(&est, 0.5).is_none());
    }

    #[test]
    fn test_average_pitch_voiced() {
        let est = vec![
            PitchEstimate {
                frequency_hz: 440.0,
                confidence: 0.8,
                time_sec: 0.0,
            },
            PitchEstimate {
                frequency_hz: 460.0,
                confidence: 0.9,
                time_sec: 0.01,
            },
        ];
        let avg = PitchTracker::average_pitch(&est, 0.5).expect("should succeed in test");
        assert!((avg - 450.0).abs() < 1.0);
    }

    #[test]
    fn test_pitch_range() {
        let est = vec![
            PitchEstimate {
                frequency_hz: 200.0,
                confidence: 0.8,
                time_sec: 0.0,
            },
            PitchEstimate {
                frequency_hz: 400.0,
                confidence: 0.9,
                time_sec: 0.01,
            },
        ];
        let (lo, hi) = PitchTracker::pitch_range(&est, 0.5).expect("should succeed in test");
        assert!((lo - 200.0).abs() < 1e-3);
        assert!((hi - 400.0).abs() < 1e-3);
    }

    #[test]
    fn test_pitch_range_none() {
        let est: Vec<PitchEstimate> = Vec::new();
        assert!(PitchTracker::pitch_range(&est, 0.5).is_none());
    }

    #[test]
    fn test_time_progresses() {
        // Use a lower sample rate and smaller window/hop so autocorrelation stays fast.
        let sr = 8000.0;
        // 0.25 s at 8 kHz = 2000 samples; produces multiple frames with small window.
        let sig = sine(220.0, sr, 2000);
        let t = PitchTracker::new(PitchTrackerConfig {
            sample_rate: sr,
            window_size: 512,
            hop_size: 128,
            min_freq: 50.0,
            max_freq: 1000.0,
            voicing_threshold: 0.25,
        });
        let est = t.track(&sig);
        if est.len() >= 2 {
            assert!(est[1].time_sec > est[0].time_sec);
        }
    }
}
