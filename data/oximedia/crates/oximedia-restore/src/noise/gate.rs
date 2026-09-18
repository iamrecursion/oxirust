//! Noise gate for removing low-level noise.

use crate::error::RestoreResult;

/// Noise gate configuration.
#[derive(Debug, Clone)]
pub struct NoiseGateConfig {
    /// Threshold in dB below which signal is gated.
    pub threshold_db: f32,
    /// Attack time in samples.
    pub attack_samples: usize,
    /// Release time in samples.
    pub release_samples: usize,
    /// Hold time in samples.
    pub hold_samples: usize,
    /// Reduction amount in dB.
    pub reduction_db: f32,
}

impl Default for NoiseGateConfig {
    fn default() -> Self {
        Self {
            threshold_db: -40.0,
            attack_samples: 100,
            release_samples: 500,
            hold_samples: 100,
            reduction_db: -60.0,
        }
    }
}

/// Noise gate state.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum GateState {
    /// Gate is open (signal passes through).
    Open,
    /// Gate is closed (signal is reduced).
    Closed,
    /// Gate is in attack phase (opening).
    Attack,
    /// Gate is in release phase (closing).
    Release,
    /// Gate is in hold phase.
    Hold,
}

/// Noise gate processor.
#[derive(Debug, Clone)]
pub struct NoiseGate {
    config: NoiseGateConfig,
    state: GateState,
    envelope: f32,
    hold_counter: usize,
    gain: f32,
}

impl NoiseGate {
    /// Create a new noise gate.
    #[must_use]
    pub fn new(config: NoiseGateConfig) -> Self {
        let initial_gain = db_to_linear(config.reduction_db);
        Self {
            config,
            state: GateState::Closed,
            envelope: 0.0,
            hold_counter: 0,
            gain: initial_gain,
        }
    }

    /// Process samples.
    pub fn process(&mut self, samples: &[f32]) -> RestoreResult<Vec<f32>> {
        let threshold = db_to_linear(self.config.threshold_db);
        let reduction = db_to_linear(self.config.reduction_db);

        #[allow(clippy::cast_precision_loss)]
        let attack_coeff = if self.config.attack_samples > 0 {
            1.0 / self.config.attack_samples as f32
        } else {
            1.0
        };

        #[allow(clippy::cast_precision_loss)]
        let release_coeff = if self.config.release_samples > 0 {
            1.0 / self.config.release_samples as f32
        } else {
            1.0
        };

        let mut output = Vec::with_capacity(samples.len());

        for &sample in samples {
            // Update envelope (simple peak detector)
            let sample_abs = sample.abs();
            if sample_abs > self.envelope {
                self.envelope = sample_abs;
            } else {
                self.envelope *= 0.9999; // Slow decay
            }

            // State machine
            match self.state {
                GateState::Closed => {
                    if self.envelope > threshold {
                        self.state = GateState::Attack;
                        self.hold_counter = 0;
                    } else {
                        self.gain = reduction;
                    }
                }
                GateState::Attack => {
                    self.gain += attack_coeff * (1.0 - self.gain);
                    if self.gain >= 0.99 {
                        self.gain = 1.0;
                        self.state = GateState::Open;
                    }
                }
                GateState::Open => {
                    self.gain = 1.0;
                    if self.envelope < threshold {
                        self.state = GateState::Hold;
                        self.hold_counter = self.config.hold_samples;
                    }
                }
                GateState::Hold => {
                    self.gain = 1.0;
                    if self.hold_counter > 0 {
                        self.hold_counter -= 1;
                    } else if self.envelope < threshold {
                        self.state = GateState::Release;
                    } else {
                        self.state = GateState::Open;
                    }
                }
                GateState::Release => {
                    self.gain -= release_coeff * (self.gain - reduction);
                    if self.gain <= reduction * 1.01 {
                        self.gain = reduction;
                        self.state = GateState::Closed;
                    } else if self.envelope > threshold {
                        self.state = GateState::Attack;
                    }
                }
            }

            output.push(sample * self.gain);
        }

        Ok(output)
    }

    /// Reset gate state.
    pub fn reset(&mut self) {
        self.state = GateState::Closed;
        self.envelope = 0.0;
        self.hold_counter = 0;
        self.gain = db_to_linear(self.config.reduction_db);
    }
}

/// Spectral gate using frequency-domain gating.
#[derive(Debug)]
pub struct SpectralGate {
    threshold_db: f32,
    reduction_db: f32,
}

impl SpectralGate {
    /// Create a new spectral gate.
    #[must_use]
    pub fn new(threshold_db: f32, reduction_db: f32) -> Self {
        Self {
            threshold_db,
            reduction_db,
        }
    }

    /// Apply gating to magnitude spectrum.
    ///
    /// # Arguments
    ///
    /// * `magnitude` - Input magnitude spectrum
    ///
    /// # Returns
    ///
    /// Gated magnitude spectrum.
    #[must_use]
    pub fn gate_spectrum(&self, magnitude: &[f32]) -> Vec<f32> {
        let threshold = db_to_linear(self.threshold_db);
        let reduction = db_to_linear(self.reduction_db);

        magnitude
            .iter()
            .map(|&mag| {
                if mag > threshold {
                    mag
                } else {
                    mag * reduction
                }
            })
            .collect()
    }

    /// Apply soft gating with smooth transition.
    #[must_use]
    pub fn gate_spectrum_soft(&self, magnitude: &[f32]) -> Vec<f32> {
        let threshold = db_to_linear(self.threshold_db);
        let reduction = db_to_linear(self.reduction_db);

        magnitude
            .iter()
            .map(|&mag| {
                if mag > threshold {
                    mag
                } else {
                    // Smooth transition
                    let ratio = mag / threshold;
                    let gain = reduction + (1.0 - reduction) * ratio * ratio;
                    mag * gain
                }
            })
            .collect()
    }
}

/// Convert dB to linear scale.
#[must_use]
fn db_to_linear(db: f32) -> f32 {
    10.0_f32.powf(db / 20.0)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_noise_gate() {
        let mut gate = NoiseGate::new(NoiseGateConfig::default());

        // Create signal with quiet and loud sections
        let mut samples = vec![0.01; 500]; // Quiet
        samples.extend(vec![0.5; 500]); // Loud
        samples.extend(vec![0.01; 500]); // Quiet again

        let output = gate.process(&samples).expect("should succeed in test");
        assert_eq!(output.len(), samples.len());

        // Check that quiet sections are reduced
        assert!(output[100].abs() < samples[100].abs());
        // Check that loud sections are mostly preserved (within attack/release envelope)
        assert!((output[700] - samples[700]).abs() < 0.2);
    }

    #[test]
    fn test_spectral_gate() {
        let gate = SpectralGate::new(-40.0, -60.0);

        let magnitude = vec![0.001, 0.1, 0.5, 0.01, 0.8];
        let gated = gate.gate_spectrum(&magnitude);

        assert_eq!(gated.len(), magnitude.len());
        // Values above threshold should be unchanged
        assert!((gated[2] - magnitude[2]).abs() < 1e-6);
    }

    #[test]
    fn test_spectral_gate_soft() {
        let gate = SpectralGate::new(-40.0, -60.0);

        let magnitude = vec![0.001, 0.1, 0.5, 0.01, 0.8];
        let gated = gate.gate_spectrum_soft(&magnitude);

        assert_eq!(gated.len(), magnitude.len());
    }

    #[test]
    fn test_reset() {
        let mut gate = NoiseGate::new(NoiseGateConfig::default());
        let samples = vec![0.5; 100];
        let _ = gate.process(&samples).expect("should succeed in test");

        gate.reset();
        assert_eq!(gate.state, GateState::Closed);
        assert_eq!(gate.envelope, 0.0);
    }

    #[test]
    fn test_db_to_linear() {
        assert!((db_to_linear(0.0) - 1.0).abs() < 1e-5);
        assert!((db_to_linear(-20.0) - 0.1).abs() < 1e-3);
    }
}
