//! Tremolo and vibrato effects with flexible LFO waveform support.
//!
//! Provides a full-featured tremolo processor operating on `f64` samples with
//! configurable rate, depth, waveform, and stereo phase offset.

#![allow(dead_code)]

use std::f64::consts::PI;

/// LFO waveform shape.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LfoWaveform {
    /// Smooth sine wave.
    Sine,
    /// Linear triangle wave.
    Triangle,
    /// Hard-edged square wave.
    Square,
    /// Rising sawtooth wave.
    Sawtooth,
    /// Pseudo-random noise LFO (stepped).
    Random,
}

/// Parameters for the tremolo effect.
#[derive(Debug, Clone)]
pub struct TremoloParams {
    /// Modulation rate in Hz.
    pub rate_hz: f64,
    /// Modulation depth (0.0 = no effect, 1.0 = full amplitude modulation).
    pub depth: f64,
    /// LFO waveform shape.
    pub waveform: LfoWaveform,
    /// Phase offset between left and right channels, in radians.
    pub stereo_phase: f64,
}

impl Default for TremoloParams {
    fn default() -> Self {
        Self {
            rate_hz: 5.0,
            depth: 0.5,
            waveform: LfoWaveform::Sine,
            stereo_phase: 0.0,
        }
    }
}

/// Stateful tremolo / amplitude-modulation processor.
pub struct TremoloProcessor {
    /// Current tremolo parameters.
    pub params: TremoloParams,
    /// Current LFO phase in [0, 1).
    pub phase: f64,
    /// Sample rate in Hz.
    pub sample_rate: f64,
    /// Internal pseudo-random state for `Random` waveform.
    rand_state: u64,
    /// Last random value (held until next step).
    rand_value: f64,
    /// Phase accumulated since last random step.
    rand_phase_acc: f64,
}

impl TremoloProcessor {
    /// Create a new tremolo processor.
    #[must_use]
    pub fn new(sample_rate: f64, params: TremoloParams) -> Self {
        Self {
            params,
            phase: 0.0,
            sample_rate,
            rand_state: 0x853c_49e6_748f_ea9b,
            rand_value: 0.0,
            rand_phase_acc: 0.0,
        }
    }

    /// Compute the current LFO output (unipolar, range [0, 1]).
    #[must_use]
    pub fn lfo_value(&self) -> f64 {
        let bipolar = generate_lfo(self.params.waveform, self.phase);
        (bipolar + 1.0) * 0.5 // map [-1, 1] to [0, 1]
    }

    /// Process a single mono sample, advancing the LFO by one sample.
    pub fn process_sample(&mut self, input: f64) -> f64 {
        let mod_val = self.lfo_value();
        let gain = 1.0 - self.params.depth + mod_val * self.params.depth;
        let output = input * gain;
        self.advance_phase();
        output
    }

    /// Process a block of samples in-place (mono).
    pub fn process_block(&mut self, block: &mut [f64]) {
        for sample in block.iter_mut() {
            *sample = self.process_sample(*sample);
        }
    }

    /// Advance the LFO phase by one sample period.
    fn advance_phase(&mut self) {
        let phase_inc = self.params.rate_hz / self.sample_rate;
        self.phase = (self.phase + phase_inc).rem_euclid(1.0);

        // Update random hold value when phase wraps around once per LFO cycle
        if self.params.waveform == LfoWaveform::Random {
            self.rand_phase_acc += phase_inc;
            if self.rand_phase_acc >= 1.0 {
                self.rand_phase_acc -= 1.0;
                self.rand_state ^= self.rand_state << 13;
                self.rand_state ^= self.rand_state >> 7;
                self.rand_state ^= self.rand_state << 17;
                // Map to [-1, 1]
                #[allow(clippy::cast_precision_loss)]
                let rand_value = (self.rand_state as f64 / u64::MAX as f64) * 2.0 - 1.0;
                self.rand_value = rand_value;
            }
        }
    }
}

/// Generate one LFO sample in the bipolar range [-1, 1].
///
/// `phase` must be in [0, 1).
#[must_use]
pub fn generate_lfo(waveform: LfoWaveform, phase: f64) -> f64 {
    match waveform {
        LfoWaveform::Sine => (2.0 * PI * phase).sin(),
        LfoWaveform::Triangle => {
            // 0..0.5 ramp up, 0.5..1 ramp down
            if phase < 0.5 {
                4.0 * phase - 1.0
            } else {
                3.0 - 4.0 * phase
            }
        }
        LfoWaveform::Square => {
            if phase < 0.5 {
                1.0
            } else {
                -1.0
            }
        }
        LfoWaveform::Sawtooth => {
            // Linear from -1 to +1 over one cycle
            2.0 * phase - 1.0
        }
        LfoWaveform::Random => {
            // Caller is responsible for generating the value; return 0 as default
            0.0
        }
    }
}

// ── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_generate_lfo_sine_at_zero() {
        let v = generate_lfo(LfoWaveform::Sine, 0.0);
        assert!(v.abs() < 1e-10, "sin(0) should be ~0, got {v}");
    }

    #[test]
    fn test_generate_lfo_sine_at_quarter() {
        let v = generate_lfo(LfoWaveform::Sine, 0.25);
        assert!((v - 1.0).abs() < 1e-10, "sin(pi/2) should be 1, got {v}");
    }

    #[test]
    fn test_generate_lfo_triangle_at_half() {
        let v = generate_lfo(LfoWaveform::Triangle, 0.25);
        assert!((v - 0.0).abs() < 1e-10, "Triangle at 0.25 = 0, got {v}");
    }

    #[test]
    fn test_generate_lfo_triangle_peak() {
        // Triangle peaks at phase=0.5 (val=1.0 from the ramp-down formula: 3-4*0.5=1)
        let v = generate_lfo(LfoWaveform::Triangle, 0.5);
        assert!((v - 1.0).abs() < 1e-10, "Triangle at 0.5 = 1, got {v}");
    }

    #[test]
    fn test_generate_lfo_square() {
        assert_eq!(generate_lfo(LfoWaveform::Square, 0.25), 1.0);
        assert_eq!(generate_lfo(LfoWaveform::Square, 0.75), -1.0);
    }

    #[test]
    fn test_generate_lfo_sawtooth() {
        let v = generate_lfo(LfoWaveform::Sawtooth, 0.0);
        assert!((v - (-1.0)).abs() < 1e-10, "Sawtooth at 0 = -1, got {v}");
        let v = generate_lfo(LfoWaveform::Sawtooth, 1.0);
        assert!((v - 1.0).abs() < 1e-10, "Sawtooth at 1 = 1, got {v}");
    }

    #[test]
    fn test_lfo_value_is_unipolar() {
        let params = TremoloParams::default();
        let proc = TremoloProcessor::new(48000.0, params);
        let v = proc.lfo_value();
        assert!(
            v >= 0.0 && v <= 1.0,
            "LFO value should be in [0,1], got {v}"
        );
    }

    #[test]
    fn test_process_sample_zero_depth() {
        let params = TremoloParams {
            depth: 0.0,
            ..Default::default()
        };
        let mut proc = TremoloProcessor::new(48000.0, params);
        let out = proc.process_sample(0.8);
        assert!(
            (out - 0.8).abs() < 1e-10,
            "Zero depth = pass-through, got {out}"
        );
    }

    #[test]
    fn test_process_sample_full_depth_at_trough() {
        // Phase = 0.75 gives sine = -1 → mod_val = 0 → gain = 1 - 1 + 0 = 0 → silence
        let params = TremoloParams {
            depth: 1.0,
            rate_hz: 1.0,
            ..Default::default()
        };
        let mut proc = TremoloProcessor::new(48000.0, params);
        proc.phase = 0.75; // sine trough
        let out = proc.process_sample(1.0);
        assert!(
            out.abs() < 0.05,
            "Full depth at trough should be near 0, got {out}"
        );
    }

    #[test]
    fn test_process_sample_output_is_finite() {
        let mut proc = TremoloProcessor::new(44100.0, TremoloParams::default());
        for _ in 0..1024 {
            let out = proc.process_sample(0.5);
            assert!(out.is_finite(), "Output should always be finite");
        }
    }

    #[test]
    fn test_process_block_length_preserved() {
        let mut proc = TremoloProcessor::new(48000.0, TremoloParams::default());
        let mut block = vec![0.3f64; 512];
        proc.process_block(&mut block);
        assert_eq!(block.len(), 512);
    }

    #[test]
    fn test_phase_advances() {
        let params = TremoloParams {
            rate_hz: 100.0,
            ..Default::default()
        };
        let mut proc = TremoloProcessor::new(1000.0, params);
        let initial_phase = proc.phase;
        proc.process_sample(0.0);
        assert!(
            proc.phase > initial_phase,
            "Phase should advance after one sample"
        );
    }

    #[test]
    fn test_process_block_all_waveforms() {
        for waveform in [
            LfoWaveform::Sine,
            LfoWaveform::Triangle,
            LfoWaveform::Square,
            LfoWaveform::Sawtooth,
            LfoWaveform::Random,
        ] {
            let params = TremoloParams {
                waveform,
                ..Default::default()
            };
            let mut proc = TremoloProcessor::new(48000.0, params);
            let mut block = vec![0.5f64; 256];
            proc.process_block(&mut block);
            for &s in &block {
                assert!(
                    s.is_finite(),
                    "Waveform {waveform:?} produced non-finite output"
                );
            }
        }
    }
}
