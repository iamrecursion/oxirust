//! PPM (Peak Programme Meter) implementation.

use serde::{Deserialize, Serialize};

/// PPM standard.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum PpmStandard {
    /// EBU PPM (IEC 60268-10 Type I).
    Ebu,

    /// Nordic PPM (IEC 60268-10 Type II).
    Nordic,

    /// DIN PPM (IEC 60268-10 Type IIa).
    Din,

    /// BBC PPM.
    Bbc,
}

/// PPM metrics.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct PpmMetrics {
    /// Per-channel PPM levels (0.0-1.0).
    pub channel_levels: Vec<f32>,

    /// Peak PPM level across all channels.
    pub peak_ppm: f32,

    /// Per-channel peak holds.
    pub peak_holds: Vec<f32>,
}

/// PPM (Peak Programme Meter).
///
/// PPM has fast attack (5-10ms) and slow decay (1.5-2.8s) depending on standard.
pub struct PpmMeter {
    sample_rate: f64,
    channels: usize,
    standard: PpmStandard,
    channel_states: Vec<f32>,
    peak_holds: Vec<f32>,
    attack_coeff: f32,
    release_coeff: f32,
    metrics: PpmMetrics,
}

impl PpmMeter {
    /// Create a new PPM meter.
    #[must_use]
    pub fn new(sample_rate: f64, channels: usize, standard: PpmStandard) -> Self {
        let (attack_time, release_time) = match standard {
            PpmStandard::Ebu => (0.010, 2.8),    // 10ms attack, 2.8s decay
            PpmStandard::Nordic => (0.005, 2.8), // 5ms attack, 2.8s decay
            PpmStandard::Din => (0.010, 1.5),    // 10ms attack, 1.5s decay
            PpmStandard::Bbc => (0.010, 2.4),    // 10ms attack, 2.4s decay
        };

        let attack_coeff = Self::calculate_coeff(attack_time, sample_rate);
        let release_coeff = Self::calculate_coeff(release_time, sample_rate);

        Self {
            sample_rate,
            channels,
            standard,
            channel_states: vec![0.0; channels],
            peak_holds: vec![0.0; channels],
            attack_coeff,
            release_coeff,
            metrics: PpmMetrics {
                channel_levels: vec![0.0; channels],
                peak_ppm: 0.0,
                peak_holds: vec![0.0; channels],
            },
        }
    }

    /// Process audio samples.
    pub fn process(&mut self, samples: &[f32]) {
        if samples.is_empty() || self.channels == 0 {
            return;
        }

        let frame_count = samples.len() / self.channels;

        for frame in 0..frame_count {
            for ch in 0..self.channels {
                let idx = frame * self.channels + ch;
                if idx < samples.len() {
                    let sample = samples[idx].abs();

                    // PPM uses fast attack, slow release
                    let coeff = if sample > self.channel_states[ch] {
                        self.attack_coeff
                    } else {
                        self.release_coeff
                    };

                    self.channel_states[ch] =
                        self.channel_states[ch] * (1.0 - coeff) + sample * coeff;

                    // Update peak hold
                    if self.channel_states[ch] > self.peak_holds[ch] {
                        self.peak_holds[ch] = self.channel_states[ch];
                    }
                }
            }
        }

        // Update metrics
        self.update_metrics();
    }

    /// Get current metrics.
    #[must_use]
    pub const fn metrics(&self) -> &PpmMetrics {
        &self.metrics
    }

    /// Reset meter state.
    pub fn reset(&mut self) {
        self.channel_states.fill(0.0);
        self.peak_holds.fill(0.0);
        self.metrics = PpmMetrics {
            channel_levels: vec![0.0; self.channels],
            peak_ppm: 0.0,
            peak_holds: vec![0.0; self.channels],
        };
    }

    /// Reset peak holds.
    pub fn reset_peaks(&mut self) {
        self.peak_holds.fill(0.0);
        // Sync metrics cache so callers see updated peak_holds immediately.
        self.metrics.peak_holds = self.peak_holds.clone();
    }

    fn update_metrics(&mut self) {
        self.metrics.channel_levels = self.channel_states.clone();
        self.metrics.peak_holds = self.peak_holds.clone();

        self.metrics.peak_ppm = self
            .channel_states
            .iter()
            .fold(0.0f32, |acc, &x| acc.max(x));
    }

    #[allow(clippy::cast_possible_truncation)]
    fn calculate_coeff(time_constant: f64, sample_rate: f64) -> f32 {
        (1.0 - (-1.0 / (time_constant * sample_rate)).exp()) as f32
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ppm_meter_creation() {
        let meter = PpmMeter::new(48000.0, 2, PpmStandard::Ebu);
        assert_eq!(meter.channels, 2);
        assert_eq!(meter.standard, PpmStandard::Ebu);
    }

    #[test]
    fn test_ppm_meter_process() {
        let mut meter = PpmMeter::new(48000.0, 2, PpmStandard::Ebu);

        let samples = vec![0.5f32; 1000];
        meter.process(&samples);

        let metrics = meter.metrics();
        assert!(metrics.peak_ppm > 0.0);
    }

    #[test]
    fn test_ppm_peak_hold() {
        let mut meter = PpmMeter::new(48000.0, 2, PpmStandard::Ebu);

        // Process loud signal
        let samples = vec![0.8f32; 1000];
        meter.process(&samples);

        let peak_before = meter.metrics().peak_ppm;

        // Process quiet signal
        let samples = vec![0.1f32; 1000];
        meter.process(&samples);

        // Peak hold should still be high
        assert!(meter.metrics().peak_holds[0] >= peak_before * 0.9);

        // Reset peak holds
        meter.reset_peaks();
        assert_eq!(meter.metrics().peak_holds[0], 0.0);
    }
}
