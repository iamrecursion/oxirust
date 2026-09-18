//! Meter ballistics implementation.
//!
//! Provides time-domain response characteristics for various meter types.

use std::collections::VecDeque;

/// Ballistics configuration for meters.
#[derive(Clone, Debug)]
pub struct BallisticsConfig {
    /// Integration time constant in seconds.
    pub integration_time: f64,
    /// Attack time constant in seconds.
    pub attack_time: f64,
    /// Release time constant in seconds.
    pub release_time: f64,
    /// Peak hold time in seconds (0.0 = no hold).
    pub peak_hold_time: f64,
    /// Return to zero time in seconds (0.0 = instant).
    pub return_time: f64,
    /// Sample rate in Hz.
    pub sample_rate: f64,
}

impl BallisticsConfig {
    /// Create VU meter ballistics (IEC 60268-10).
    ///
    /// - Integration time: 300ms
    /// - Attack time: 300ms
    /// - Release time: 300ms
    #[must_use]
    pub fn vu_meter(sample_rate: f64) -> Self {
        Self {
            integration_time: 0.300,
            attack_time: 0.300,
            release_time: 0.300,
            peak_hold_time: 0.0,
            return_time: 1.0,
            sample_rate,
        }
    }

    /// Create BBC PPM ballistics (BS.6840).
    ///
    /// - Integration time: 10ms
    /// - Attack time: 10ms (fast rise)
    /// - Release time: 2.8s (slow fall)
    #[must_use]
    pub fn bbc_ppm(sample_rate: f64) -> Self {
        Self {
            integration_time: 0.010,
            attack_time: 0.010,
            release_time: 2.8,
            peak_hold_time: 1.0,
            return_time: 0.0,
            sample_rate,
        }
    }

    /// Create EBU PPM ballistics (IEC 60268-10 Type IIa).
    ///
    /// - Integration time: 10ms
    /// - Attack time: 5ms
    /// - Release time: 1.7s (20dB in 1.7s)
    #[must_use]
    pub fn ebu_ppm(sample_rate: f64) -> Self {
        Self {
            integration_time: 0.010,
            attack_time: 0.005,
            release_time: 1.7,
            peak_hold_time: 0.0,
            return_time: 0.0,
            sample_rate,
        }
    }

    /// Create Nordic PPM ballistics (NRK/SR/DR/YLE).
    ///
    /// - Integration time: 5ms
    /// - Attack time: 5ms
    /// - Release time: 1.5s
    #[must_use]
    pub fn nordic_ppm(sample_rate: f64) -> Self {
        Self {
            integration_time: 0.005,
            attack_time: 0.005,
            release_time: 1.5,
            peak_hold_time: 0.0,
            return_time: 0.0,
            sample_rate,
        }
    }

    /// Create DIN PPM ballistics (IEC 60268-10 Type I).
    ///
    /// - Integration time: 10ms
    /// - Attack time: 10ms (instantaneous)
    /// - Release time: 1.5s (20dB in 1.5s)
    #[must_use]
    pub fn din_ppm(sample_rate: f64) -> Self {
        Self {
            integration_time: 0.010,
            attack_time: 0.010,
            release_time: 1.5,
            peak_hold_time: 0.0,
            return_time: 0.0,
            sample_rate,
        }
    }

    /// Create digital peak meter ballistics.
    ///
    /// - Integration time: 0ms (sample accurate)
    /// - Attack time: 0ms (instantaneous)
    /// - Release time: 0.0s (follows signal)
    /// - Peak hold: configurable
    #[must_use]
    pub fn digital_peak(sample_rate: f64, peak_hold_seconds: f64) -> Self {
        Self {
            integration_time: 0.0,
            attack_time: 0.0,
            release_time: 0.0,
            peak_hold_time: peak_hold_seconds,
            return_time: 0.0,
            sample_rate,
        }
    }

    /// Create RMS meter ballistics.
    ///
    /// - Integration time: configurable
    /// - Attack/release: instant with averaging
    #[must_use]
    pub fn rms(sample_rate: f64, window_seconds: f64) -> Self {
        Self {
            integration_time: window_seconds,
            attack_time: window_seconds,
            release_time: window_seconds,
            peak_hold_time: 0.0,
            return_time: 0.0,
            sample_rate,
        }
    }

    /// Get integration time coefficient.
    #[must_use]
    pub fn integration_coefficient(&self) -> f64 {
        if self.integration_time <= 0.0 {
            1.0
        } else {
            (-1.0 / (self.integration_time * self.sample_rate)).exp()
        }
    }

    /// Get attack time coefficient.
    #[must_use]
    pub fn attack_coefficient(&self) -> f64 {
        if self.attack_time <= 0.0 {
            1.0
        } else {
            (-1.0 / (self.attack_time * self.sample_rate)).exp()
        }
    }

    /// Get release time coefficient.
    #[must_use]
    pub fn release_coefficient(&self) -> f64 {
        if self.release_time <= 0.0 {
            1.0
        } else {
            (-1.0 / (self.release_time * self.sample_rate)).exp()
        }
    }

    /// Get peak hold samples.
    #[must_use]
    pub fn peak_hold_samples(&self) -> usize {
        (self.peak_hold_time * self.sample_rate) as usize
    }

    /// Get return time coefficient.
    #[must_use]
    pub fn return_coefficient(&self) -> f64 {
        if self.return_time <= 0.0 {
            0.0
        } else {
            (-1.0 / (self.return_time * self.sample_rate)).exp()
        }
    }
}

/// Ballistics processor for meter readings.
pub struct BallisticsProcessor {
    /// Configuration.
    config: BallisticsConfig,
    /// Current integrated value.
    integrated: f64,
    /// Current envelope value.
    envelope: f64,
    /// Peak hold value.
    peak_hold: f64,
    /// Peak hold counter (samples).
    peak_hold_counter: usize,
    /// Maximum peak value seen.
    max_peak: f64,
}

impl BallisticsProcessor {
    /// Create a new ballistics processor.
    #[must_use]
    pub fn new(config: BallisticsConfig) -> Self {
        Self {
            config,
            integrated: 0.0,
            envelope: 0.0,
            peak_hold: 0.0,
            peak_hold_counter: 0,
            max_peak: 0.0,
        }
    }

    /// Process a single sample value.
    ///
    /// # Arguments
    ///
    /// * `value` - Input value (linear or dB)
    ///
    /// # Returns
    ///
    /// Processed value with ballistics applied
    pub fn process(&mut self, value: f64) -> f64 {
        // Integration (smoothing)
        let integration_coeff = self.config.integration_coefficient();
        self.integrated = integration_coeff * self.integrated + (1.0 - integration_coeff) * value;

        // Attack/release envelope
        let target = self.integrated;
        let coeff = if target > self.envelope {
            self.config.attack_coefficient()
        } else {
            self.config.release_coefficient()
        };

        self.envelope = coeff * self.envelope + (1.0 - coeff) * target;

        // Peak hold
        if self.envelope > self.peak_hold {
            self.peak_hold = self.envelope;
            self.peak_hold_counter = self.config.peak_hold_samples();
            self.max_peak = self.max_peak.max(self.peak_hold);
        } else if self.peak_hold_counter > 0 {
            self.peak_hold_counter -= 1;
        } else if self.config.return_time > 0.0 {
            // Return to zero
            let return_coeff = self.config.return_coefficient();
            self.peak_hold *= return_coeff;
        } else {
            self.peak_hold = self.envelope;
        }

        self.envelope
    }

    /// Get current envelope value.
    #[must_use]
    pub fn envelope(&self) -> f64 {
        self.envelope
    }

    /// Get current peak hold value.
    #[must_use]
    pub fn peak_hold(&self) -> f64 {
        self.peak_hold
    }

    /// Get maximum peak value.
    #[must_use]
    pub fn max_peak(&self) -> f64 {
        self.max_peak
    }

    /// Reset the processor.
    pub fn reset(&mut self) {
        self.integrated = 0.0;
        self.envelope = 0.0;
        self.peak_hold = 0.0;
        self.peak_hold_counter = 0;
        self.max_peak = 0.0;
    }

    /// Reset peak hold only.
    pub fn reset_peak_hold(&mut self) {
        self.peak_hold = self.envelope;
        self.peak_hold_counter = 0;
    }

    /// Reset max peak only.
    pub fn reset_max_peak(&mut self) {
        self.max_peak = 0.0;
    }
}

/// RMS averaging window.
pub struct RmsWindow {
    /// Sample buffer for RMS calculation.
    buffer: VecDeque<f64>,
    /// Window size in samples.
    window_size: usize,
    /// Running sum of squares.
    sum_squares: f64,
}

impl RmsWindow {
    /// Create a new RMS window.
    ///
    /// # Arguments
    ///
    /// * `window_seconds` - Window duration in seconds
    /// * `sample_rate` - Sample rate in Hz
    #[must_use]
    pub fn new(window_seconds: f64, sample_rate: f64) -> Self {
        let window_size = (window_seconds * sample_rate) as usize;
        Self {
            buffer: VecDeque::with_capacity(window_size),
            window_size,
            sum_squares: 0.0,
        }
    }

    /// Add a sample and compute RMS.
    ///
    /// # Arguments
    ///
    /// * `sample` - Input sample value
    ///
    /// # Returns
    ///
    /// Current RMS value
    pub fn process(&mut self, sample: f64) -> f64 {
        let square = sample * sample;
        self.sum_squares += square;
        self.buffer.push_back(square);

        if self.buffer.len() > self.window_size {
            if let Some(old) = self.buffer.pop_front() {
                self.sum_squares -= old;
            }
        }

        if self.buffer.is_empty() {
            0.0
        } else {
            (self.sum_squares / self.buffer.len() as f64).sqrt()
        }
    }

    /// Get current RMS value without adding a sample.
    #[must_use]
    pub fn rms(&self) -> f64 {
        if self.buffer.is_empty() {
            0.0
        } else {
            (self.sum_squares / self.buffer.len() as f64).sqrt()
        }
    }

    /// Reset the window.
    pub fn reset(&mut self) {
        self.buffer.clear();
        self.sum_squares = 0.0;
    }
}

/// Peak detector with configurable hold time.
pub struct PeakDetector {
    /// Current peak value.
    peak: f64,
    /// Peak hold counter in samples.
    hold_counter: usize,
    /// Peak hold duration in samples.
    hold_samples: usize,
    /// Decay rate per sample (0.0 = instant, 1.0 = no decay).
    decay_rate: f64,
}

impl PeakDetector {
    /// Create a new peak detector.
    ///
    /// # Arguments
    ///
    /// * `hold_seconds` - Peak hold time in seconds
    /// * `decay_seconds` - Decay time in seconds (0.0 = instant)
    /// * `sample_rate` - Sample rate in Hz
    #[must_use]
    pub fn new(hold_seconds: f64, decay_seconds: f64, sample_rate: f64) -> Self {
        let hold_samples = (hold_seconds * sample_rate) as usize;
        let decay_rate = if decay_seconds > 0.0 {
            (-1.0 / (decay_seconds * sample_rate)).exp()
        } else {
            0.0
        };

        Self {
            peak: 0.0,
            hold_counter: 0,
            hold_samples,
            decay_rate,
        }
    }

    /// Process a sample and update peak.
    ///
    /// # Arguments
    ///
    /// * `value` - Input value (absolute)
    ///
    /// # Returns
    ///
    /// Current peak value
    pub fn process(&mut self, value: f64) -> f64 {
        if value > self.peak {
            self.peak = value;
            self.hold_counter = self.hold_samples;
        } else if self.hold_counter > 0 {
            self.hold_counter -= 1;
        } else {
            self.peak *= self.decay_rate;
        }

        self.peak
    }

    /// Get current peak value.
    #[must_use]
    pub fn peak(&self) -> f64 {
        self.peak
    }

    /// Reset peak value.
    pub fn reset(&mut self) {
        self.peak = 0.0;
        self.hold_counter = 0;
    }
}

/// Overload detector with hysteresis.
pub struct OverloadDetector {
    /// Overload threshold (linear).
    threshold: f64,
    /// Overload state.
    overload: bool,
    /// Overload counter (samples).
    overload_counter: usize,
    /// Minimum overload duration in samples.
    min_duration: usize,
    /// Reset delay in samples.
    reset_delay: usize,
    /// Reset counter.
    reset_counter: usize,
}

impl OverloadDetector {
    /// Create a new overload detector.
    ///
    /// # Arguments
    ///
    /// * `threshold_db` - Overload threshold in dB
    /// * `min_duration_ms` - Minimum overload duration in milliseconds
    /// * `reset_delay_ms` - Reset delay in milliseconds
    /// * `sample_rate` - Sample rate in Hz
    #[must_use]
    pub fn new(
        threshold_db: f64,
        min_duration_ms: f64,
        reset_delay_ms: f64,
        sample_rate: f64,
    ) -> Self {
        Self {
            threshold: db_to_linear(threshold_db),
            overload: false,
            overload_counter: 0,
            min_duration: ((min_duration_ms / 1000.0) * sample_rate) as usize,
            reset_delay: ((reset_delay_ms / 1000.0) * sample_rate) as usize,
            reset_counter: 0,
        }
    }

    /// Process a sample and check for overload.
    ///
    /// # Arguments
    ///
    /// * `value` - Input value (linear)
    ///
    /// # Returns
    ///
    /// `true` if overload is detected
    pub fn process(&mut self, value: f64) -> bool {
        if value.abs() >= self.threshold {
            self.overload_counter += 1;
            self.reset_counter = 0;

            if self.overload_counter >= self.min_duration {
                self.overload = true;
            }
        } else {
            self.overload_counter = 0;
            if self.overload {
                self.reset_counter += 1;
                if self.reset_counter >= self.reset_delay {
                    self.overload = false;
                    self.reset_counter = 0;
                }
            }
        }

        self.overload
    }

    /// Check if currently in overload state.
    #[must_use]
    pub fn is_overload(&self) -> bool {
        self.overload
    }

    /// Reset overload state.
    pub fn reset(&mut self) {
        self.overload = false;
        self.overload_counter = 0;
        self.reset_counter = 0;
    }
}

/// Convert decibels to linear amplitude.
#[must_use]
pub fn db_to_linear(db: f64) -> f64 {
    10.0_f64.powf(db / 20.0)
}

/// Convert linear amplitude to decibels.
#[must_use]
pub fn linear_to_db(linear: f64) -> f64 {
    if linear > 0.0 {
        20.0 * linear.log10()
    } else {
        f64::NEG_INFINITY
    }
}

/// Clamp a value to a range.
#[must_use]
pub fn clamp(value: f64, min: f64, max: f64) -> f64 {
    value.max(min).min(max)
}
