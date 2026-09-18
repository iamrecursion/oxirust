//! Audio/video frequency synchronization using PLL and drift estimation.
//!
//! This module provides tools for measuring and correcting frequency errors
//! between audio and video clocks, implementing a simple Phase-Locked Loop (PLL)
//! for clock recovery, and computing A/V drift for re-sampling decisions.

/// A frequency error measurement.
#[derive(Debug, Clone, Copy)]
pub struct FrequencyError {
    /// Error expressed in parts-per-million (ppm).
    pub ppm: f64,
    /// Error expressed in hertz (absolute).
    pub hz: f64,
    /// Nominal sample rate from which this error was measured.
    pub sample_rate: u32,
}

impl FrequencyError {
    /// Returns `true` if the absolute error magnitude is at or below `max_ppm`.
    #[must_use]
    pub fn is_acceptable(&self, max_ppm: f64) -> bool {
        self.ppm.abs() <= max_ppm
    }

    /// Returns the multiplicative correction factor to apply to a clock.
    ///
    /// If you multiply your measured frequency by this factor you get the
    /// nominal frequency:  `nominal = measured * correction_factor`.
    #[must_use]
    pub fn correction_factor(&self) -> f64 {
        1.0 / (1.0 + self.ppm / 1_000_000.0)
    }
}

/// Estimate the frequency error between a measured signal frequency and its nominal value.
///
/// Returns a [`FrequencyError`] whose `ppm` field is positive when `measured_hz`
/// is higher than `nominal_hz` (clock is running fast).
#[must_use]
pub fn estimate_frequency_error(measured_hz: f64, nominal_hz: f64) -> FrequencyError {
    let hz_err = measured_hz - nominal_hz;
    let ppm = if nominal_hz.abs() > 1e-10 {
        hz_err / nominal_hz * 1_000_000.0
    } else {
        0.0
    };
    FrequencyError {
        ppm,
        hz: hz_err,
        sample_rate: nominal_hz as u32,
    }
}

/// A simple second-order digital Phase-Locked Loop for clock recovery.
///
/// The PLL integrates phase and frequency corrections to produce a stable
/// output frequency estimate that tracks an input reference.
pub struct SyncPll {
    /// Nominal (target) output frequency in Hz.
    pub nominal_hz: f64,
    /// Proportional gain (scales phase error into frequency correction).
    pub gain: f64,
    /// Accumulated phase (radians).
    pub phase_acc: f64,
    /// Accumulated frequency deviation from nominal (Hz).
    pub freq_acc: f64,
}

impl SyncPll {
    /// Create a new `SyncPll` with the given nominal frequency and gain.
    #[must_use]
    pub fn new(nominal_hz: f64, gain: f64) -> Self {
        Self {
            nominal_hz,
            gain,
            phase_acc: 0.0,
            freq_acc: 0.0,
        }
    }

    /// Update the PLL with a phase error measurement (in radians) and return the
    /// current locked output frequency in Hz.
    ///
    /// The PLL applies a proportional-integral correction:
    /// - `freq_acc += gain * phase_error`
    /// - Output: `nominal_hz + freq_acc`
    pub fn update(&mut self, phase_error: f64) -> f64 {
        // Proportional correction to phase accumulator
        self.phase_acc += phase_error;
        // Integral correction to frequency accumulator
        self.freq_acc += self.gain * phase_error;
        // Clamp frequency accumulator to ±10% of nominal
        let max_dev = self.nominal_hz * 0.1;
        self.freq_acc = self.freq_acc.clamp(-max_dev, max_dev);
        self.nominal_hz + self.freq_acc
    }

    /// Returns the ±lock range of this PLL in Hz (10% of nominal frequency).
    #[must_use]
    pub fn lock_range_hz(&self) -> f64 {
        self.nominal_hz * 0.1
    }

    /// Returns `true` if the given frequency error is within the PLL's lock range.
    #[must_use]
    pub fn is_locked(&self, err: &FrequencyError) -> bool {
        err.hz.abs() <= self.lock_range_hz()
    }
}

/// Audio/video synchronization helper that computes drift and re-sample ratios.
pub struct AudioVideoSync {
    /// Audio clock rate in Hz (e.g. 48000.0).
    pub audio_rate_hz: f64,
    /// Video clock rate in Hz (e.g. 25.0 for frames/s or the pixel clock).
    pub video_rate_hz: f64,
    /// Target latency between audio and video in milliseconds.
    pub target_latency_ms: f64,
}

impl AudioVideoSync {
    /// Create a new `AudioVideoSync`.
    #[must_use]
    pub fn new(audio_rate_hz: f64, video_rate_hz: f64, target_latency_ms: f64) -> Self {
        Self {
            audio_rate_hz,
            video_rate_hz,
            target_latency_ms,
        }
    }

    /// Compute the A/V drift in milliseconds.
    ///
    /// Positive drift means audio is ahead of video. Zero means perfectly
    /// synchronised.
    ///
    /// # Arguments
    /// * `elapsed_audio_s` – wall-clock seconds of audio consumed.
    /// * `elapsed_video_s` – wall-clock seconds of video consumed.
    #[must_use]
    pub fn compute_drift_ms(&self, elapsed_audio_s: f64, elapsed_video_s: f64) -> f64 {
        (elapsed_audio_s - elapsed_video_s) * 1_000.0 - self.target_latency_ms
    }

    /// Returns the audio re-sample ratio needed to correct A/V sync.
    ///
    /// A ratio > 1.0 means audio should be stretched (it is behind video).
    /// A ratio < 1.0 means audio should be compressed (it is ahead of video).
    #[must_use]
    pub fn resample_ratio(&self) -> f64 {
        if self.video_rate_hz.abs() < 1e-10 {
            return 1.0;
        }
        self.audio_rate_hz / (self.audio_rate_hz + self.video_rate_hz)
            * (1.0 + self.video_rate_hz / self.audio_rate_hz)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_frequency_error_is_acceptable_within() {
        let err = FrequencyError {
            ppm: 50.0,
            hz: 2.4,
            sample_rate: 48000,
        };
        assert!(err.is_acceptable(100.0));
    }

    #[test]
    fn test_frequency_error_is_acceptable_outside() {
        let err = FrequencyError {
            ppm: 150.0,
            hz: 7.2,
            sample_rate: 48000,
        };
        assert!(!err.is_acceptable(100.0));
    }

    #[test]
    fn test_frequency_error_is_acceptable_negative_ppm() {
        let err = FrequencyError {
            ppm: -80.0,
            hz: -3.84,
            sample_rate: 48000,
        };
        assert!(err.is_acceptable(100.0));
        assert!(!err.is_acceptable(50.0));
    }

    #[test]
    fn test_frequency_error_correction_factor_no_error() {
        let err = FrequencyError {
            ppm: 0.0,
            hz: 0.0,
            sample_rate: 48000,
        };
        assert!((err.correction_factor() - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_frequency_error_correction_factor_fast_clock() {
        // 1000 ppm fast → correction < 1.0
        let err = FrequencyError {
            ppm: 1000.0,
            hz: 48.0,
            sample_rate: 48000,
        };
        assert!(err.correction_factor() < 1.0);
    }

    #[test]
    fn test_estimate_frequency_error_zero() {
        let err = estimate_frequency_error(48000.0, 48000.0);
        assert!((err.ppm).abs() < 1e-10);
        assert!((err.hz).abs() < 1e-10);
    }

    #[test]
    fn test_estimate_frequency_error_positive() {
        let err = estimate_frequency_error(48048.0, 48000.0);
        // 48 Hz fast on 48 kHz = 1000 ppm
        assert!((err.ppm - 1000.0).abs() < 1.0);
        assert!((err.hz - 48.0).abs() < 0.01);
    }

    #[test]
    fn test_estimate_frequency_error_negative() {
        let err = estimate_frequency_error(47952.0, 48000.0);
        assert!(err.ppm < 0.0);
        assert!(err.hz < 0.0);
    }

    #[test]
    fn test_sync_pll_initial_output() {
        let mut pll = SyncPll::new(48000.0, 0.01);
        let freq = pll.update(0.0);
        assert!((freq - 48000.0).abs() < 1.0);
    }

    #[test]
    fn test_sync_pll_lock_range() {
        let pll = SyncPll::new(48000.0, 0.01);
        assert!((pll.lock_range_hz() - 4800.0).abs() < 0.01);
    }

    #[test]
    fn test_sync_pll_is_locked_within_range() {
        let pll = SyncPll::new(48000.0, 0.01);
        let small_err = FrequencyError {
            ppm: 10.0,
            hz: 0.5,
            sample_rate: 48000,
        };
        assert!(pll.is_locked(&small_err));
    }

    #[test]
    fn test_sync_pll_is_locked_outside_range() {
        let pll = SyncPll::new(48000.0, 0.01);
        // 5 kHz error is larger than 4.8 kHz lock range
        let big_err = FrequencyError {
            ppm: 100_000.0,
            hz: 5000.0,
            sample_rate: 48000,
        };
        assert!(!pll.is_locked(&big_err));
    }

    #[test]
    fn test_av_sync_compute_drift_ms_no_drift() {
        let sync = AudioVideoSync::new(48000.0, 25.0, 0.0);
        let drift = sync.compute_drift_ms(10.0, 10.0);
        assert!((drift).abs() < 0.01);
    }

    #[test]
    fn test_av_sync_compute_drift_ms_audio_ahead() {
        let sync = AudioVideoSync::new(48000.0, 25.0, 0.0);
        // Audio has processed 11 s worth, video 10 s
        let drift = sync.compute_drift_ms(11.0, 10.0);
        assert!((drift - 1000.0).abs() < 0.01);
    }

    #[test]
    fn test_av_sync_resample_ratio_is_positive() {
        let sync = AudioVideoSync::new(48000.0, 25.0, 40.0);
        assert!(sync.resample_ratio() > 0.0);
    }
}
