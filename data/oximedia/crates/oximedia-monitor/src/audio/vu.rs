//! VU (Volume Unit) meter implementation.

use serde::{Deserialize, Serialize};

/// VU meter metrics.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct VuMetrics {
    /// Per-channel VU levels (0.0-1.0, 0 VU = 0.707).
    pub channel_levels: Vec<f32>,

    /// Peak VU level across all channels.
    pub peak_vu: f32,

    /// Average VU level.
    pub avg_vu: f32,
}

/// VU meter with classic ballistics.
///
/// VU meters have a rise time of 300ms to 99% of a steady-state 0 VU tone (1 kHz sine wave at reference level).
pub struct VuMeter {
    sample_rate: f64,
    channels: usize,
    channel_states: Vec<f32>,
    attack_coeff: f32,
    release_coeff: f32,
    metrics: VuMetrics,
}

impl VuMeter {
    /// Create a new VU meter.
    ///
    /// # Arguments
    ///
    /// * `sample_rate` - Audio sample rate in Hz
    /// * `channels` - Number of audio channels
    #[must_use]
    pub fn new(sample_rate: f64, channels: usize) -> Self {
        // VU meter ballistics: 300ms integration time
        let integration_time = 0.3; // 300ms
        let attack_coeff = Self::calculate_coeff(integration_time, sample_rate);
        let release_coeff = attack_coeff; // VU has same attack/release

        Self {
            sample_rate,
            channels,
            channel_states: vec![0.0; channels],
            attack_coeff,
            release_coeff,
            metrics: VuMetrics {
                channel_levels: vec![0.0; channels],
                peak_vu: 0.0,
                avg_vu: 0.0,
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

                    // Apply ballistics
                    let coeff = if sample > self.channel_states[ch] {
                        self.attack_coeff
                    } else {
                        self.release_coeff
                    };

                    self.channel_states[ch] =
                        self.channel_states[ch] * (1.0 - coeff) + sample * coeff;
                }
            }
        }

        // Update metrics
        self.update_metrics();
    }

    /// Get current metrics.
    #[must_use]
    pub const fn metrics(&self) -> &VuMetrics {
        &self.metrics
    }

    /// Reset meter state.
    pub fn reset(&mut self) {
        self.channel_states.fill(0.0);
        self.metrics = VuMetrics {
            channel_levels: vec![0.0; self.channels],
            peak_vu: 0.0,
            avg_vu: 0.0,
        };
    }

    fn update_metrics(&mut self) {
        self.metrics.channel_levels.clear();

        let mut sum = 0.0f32;
        let mut peak = 0.0f32;

        for &state in &self.channel_states {
            // 0 VU = 0.707 (-3 dBFS for sine wave)
            let vu_level = state / 0.707;
            self.metrics.channel_levels.push(vu_level);

            sum += vu_level;
            peak = peak.max(vu_level);
        }

        self.metrics.peak_vu = peak;
        self.metrics.avg_vu = if self.channels > 0 {
            sum / self.channels as f32
        } else {
            0.0
        };
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
    fn test_vu_meter_creation() {
        let meter = VuMeter::new(48000.0, 2);
        assert_eq!(meter.channels, 2);
    }

    #[test]
    fn test_vu_meter_process() {
        let mut meter = VuMeter::new(48000.0, 2);

        // Process silence
        let samples = vec![0.0f32; 1000];
        meter.process(&samples);

        let metrics = meter.metrics();
        assert_eq!(metrics.peak_vu, 0.0);

        // Process signal
        let samples = vec![0.5f32; 1000];
        meter.process(&samples);

        let metrics = meter.metrics();
        assert!(metrics.peak_vu > 0.0);
    }

    #[test]
    fn test_vu_meter_reset() {
        let mut meter = VuMeter::new(48000.0, 2);

        let samples = vec![0.5f32; 1000];
        meter.process(&samples);

        meter.reset();

        let metrics = meter.metrics();
        assert_eq!(metrics.peak_vu, 0.0);
    }
}
