//! Chorus and flanger audio effect.
//!
//! Implements a multi-voice chorus effect in pure Rust with configurable rate,
//! depth, mix, voices, and feedback using internal circular delay lines.

#![allow(dead_code)]

use std::f64::consts::PI;

/// Configuration for the chorus/flanger effect.
#[derive(Debug, Clone)]
pub struct ChorusParams {
    /// LFO modulation rate in Hz.
    pub rate_hz: f64,
    /// Delay modulation depth in milliseconds.
    pub depth_ms: f64,
    /// Wet/dry mix (0.0 = fully dry, 1.0 = fully wet).
    pub mix: f64,
    /// Number of chorus voices (1–8).
    pub voices: usize,
    /// Feedback amount per delay line [-1.0, 1.0].
    pub feedback: f64,
}

impl Default for ChorusParams {
    fn default() -> Self {
        Self {
            rate_hz: 0.5,
            depth_ms: 5.0,
            mix: 0.5,
            voices: 3,
            feedback: 0.2,
        }
    }
}

/// Stateful chorus / flanger processor.
pub struct ChorusProcessor {
    /// Current chorus parameters.
    pub params: ChorusParams,
    /// One circular delay line per voice.
    pub delay_lines: Vec<Vec<f64>>,
    /// Write position in the delay buffer.
    pub write_pos: usize,
    /// Current LFO phase in [0, 1).
    pub phase: f64,
    /// Sample rate in Hz.
    pub sample_rate: f64,
}

impl ChorusProcessor {
    /// Create a new chorus processor.
    ///
    /// Delay lines are sized to hold at least `base_delay_ms + depth_ms` of audio.
    #[must_use]
    pub fn new(sample_rate: f64, params: ChorusParams) -> Self {
        let voices = params.voices.clamp(1, 8);

        // Maximum delay needed: base 30ms + depth
        let max_delay_ms = 30.0 + params.depth_ms;
        #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
        let max_delay_samples = ((max_delay_ms / 1000.0) * sample_rate).ceil() as usize + 2;

        let delay_lines = vec![vec![0.0f64; max_delay_samples]; voices];

        Self {
            params,
            delay_lines,
            write_pos: 0,
            phase: 0.0,
            sample_rate,
        }
    }

    /// Process a single sample through the chorus, returning the mixed output.
    pub fn process_sample(&mut self, input: f64) -> f64 {
        let num_voices = self.delay_lines.len();
        let buf_len = if num_voices > 0 {
            self.delay_lines[0].len()
        } else {
            1
        };

        #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
        let base_delay_samples = (30.0e-3 * self.sample_rate) as usize;
        let depth_samples = self.params.depth_ms * 1e-3 * self.sample_rate;

        let mut wet_sum = 0.0;

        for (i, delay_line) in self.delay_lines.iter_mut().enumerate() {
            // Each voice has a phase offset
            #[allow(clippy::cast_precision_loss)]
            let voice_phase = (self.phase + i as f64 / num_voices as f64).rem_euclid(1.0);
            let lfo = (2.0 * PI * voice_phase).sin();
            #[allow(clippy::cast_precision_loss)]
            let delay_s = base_delay_samples as f64 + lfo * depth_samples;
            #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
            let delay_int = delay_s.floor() as usize;
            let frac = delay_s - delay_s.floor();

            // Linear interpolation between two taps
            let read_pos_a = (self.write_pos + buf_len - delay_int) % buf_len;
            let read_pos_b = (self.write_pos + buf_len - delay_int.saturating_sub(1)) % buf_len;

            let delayed = delay_line[read_pos_a] * (1.0 - frac) + delay_line[read_pos_b] * frac;

            // Write input + feedback into delay line
            delay_line[self.write_pos] = input + delayed * self.params.feedback;

            wet_sum += delayed;
        }

        // Advance write position
        self.write_pos = (self.write_pos + 1) % buf_len;

        // Advance LFO phase
        self.phase = (self.phase + self.params.rate_hz / self.sample_rate).rem_euclid(1.0);

        // Mix dry and wet
        #[allow(clippy::cast_precision_loss)]
        let wet = wet_sum / num_voices as f64;
        input * (1.0 - self.params.mix) + wet * self.params.mix
    }

    /// Process a block of samples: `input` → `output` (both must be the same length).
    pub fn process_block(&mut self, input: &[f64], output: &mut [f64]) {
        let len = input.len().min(output.len());
        for i in 0..len {
            output[i] = self.process_sample(input[i]);
        }
    }
}

impl crate::AudioEffect for ChorusProcessor {
    const EFFECT_ID: &'static str = "chorus_processor";

    fn process_sample(&mut self, input: f32) -> f32 {
        #[allow(clippy::cast_possible_truncation)]
        let out = self.process_sample(input as f64);
        out as f32
    }

    fn reset(&mut self) {
        for dl in &mut self.delay_lines {
            for s in dl.iter_mut() {
                *s = 0.0;
            }
        }
        self.write_pos = 0;
        self.phase = 0.0;
    }

    fn set_sample_rate(&mut self, sample_rate: f32) {
        self.sample_rate = sample_rate as f64;
    }

    fn set_wet_dry(&mut self, wet: f32) {
        self.params.mix = wet.clamp(0.0, 1.0) as f64;
    }

    fn wet_dry(&self) -> f32 {
        self.params.mix as f32
    }
}

// ── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_chorus_default_params() {
        let p = ChorusParams::default();
        assert_eq!(p.voices, 3);
        assert!((p.rate_hz - 0.5).abs() < 1e-10);
    }

    #[test]
    fn test_chorus_process_sample_finite() {
        let mut proc = ChorusProcessor::new(48000.0, ChorusParams::default());
        let out = proc.process_sample(0.5);
        assert!(out.is_finite(), "Output should be finite, got {out}");
    }

    #[test]
    fn test_chorus_process_block_length() {
        let mut proc = ChorusProcessor::new(48000.0, ChorusParams::default());
        let input = vec![0.3f64; 128];
        let mut output = vec![0.0f64; 128];
        proc.process_block(&input, &mut output);
        assert_eq!(output.len(), 128);
    }

    #[test]
    fn test_chorus_process_block_all_finite() {
        let mut proc = ChorusProcessor::new(44100.0, ChorusParams::default());
        let input: Vec<f64> = (0..512).map(|i| (i as f64 * 0.01).sin() * 0.5).collect();
        let mut output = vec![0.0f64; 512];
        proc.process_block(&input, &mut output);
        for (i, &s) in output.iter().enumerate() {
            assert!(s.is_finite(), "output[{i}] is not finite: {s}");
        }
    }

    #[test]
    fn test_chorus_dry_mix() {
        // With mix=0, output should equal input (fully dry)
        let params = ChorusParams {
            mix: 0.0,
            ..Default::default()
        };
        let mut proc = ChorusProcessor::new(48000.0, params);
        let out = proc.process_sample(0.7);
        assert!(
            (out - 0.7).abs() < 1e-10,
            "Fully dry should pass through: {out}"
        );
    }

    #[test]
    fn test_chorus_delay_lines_count() {
        let params = ChorusParams {
            voices: 5,
            ..Default::default()
        };
        let proc = ChorusProcessor::new(48000.0, params);
        assert_eq!(proc.delay_lines.len(), 5);
    }

    #[test]
    fn test_chorus_voices_clamped() {
        let params = ChorusParams {
            voices: 0,
            ..Default::default()
        };
        let proc = ChorusProcessor::new(48000.0, params);
        // Should be clamped to at least 1
        assert!(proc.delay_lines.len() >= 1);

        let params_high = ChorusParams {
            voices: 100,
            ..Default::default()
        };
        let proc_high = ChorusProcessor::new(48000.0, params_high);
        assert!(proc_high.delay_lines.len() <= 8);
    }

    #[test]
    fn test_chorus_phase_advances() {
        let mut proc = ChorusProcessor::new(48000.0, ChorusParams::default());
        let initial = proc.phase;
        proc.process_sample(0.0);
        assert!(proc.phase >= 0.0 && proc.phase < 1.0);
        // Phase should have moved unless already wrapped
        assert!(proc.phase != initial || proc.params.rate_hz == 0.0);
    }

    #[test]
    fn test_chorus_write_pos_advances() {
        let mut proc = ChorusProcessor::new(48000.0, ChorusParams::default());
        let initial = proc.write_pos;
        proc.process_sample(0.0);
        assert_ne!(proc.write_pos, initial, "Write position should advance");
    }

    #[test]
    fn test_chorus_different_rates() {
        for rate in [0.1, 1.0, 5.0, 10.0] {
            let params = ChorusParams {
                rate_hz: rate,
                ..Default::default()
            };
            let mut proc = ChorusProcessor::new(48000.0, params);
            let mut out = vec![0.0f64; 256];
            proc.process_block(&vec![0.5; 256], &mut out);
            for &s in &out {
                assert!(s.is_finite(), "rate={rate}: non-finite output {s}");
            }
        }
    }

    #[test]
    fn test_chorus_process_block_mismatched_len() {
        // process_block should handle input shorter than output gracefully
        let mut proc = ChorusProcessor::new(48000.0, ChorusParams::default());
        let input = vec![0.5f64; 64];
        let mut output = vec![0.0f64; 128];
        proc.process_block(&input, &mut output);
        // First 64 samples should be processed
        for &s in &output[..64] {
            assert!(s.is_finite());
        }
        // Remaining samples should still be 0.0 (untouched)
        for &s in &output[64..] {
            assert_eq!(s, 0.0);
        }
    }
}
