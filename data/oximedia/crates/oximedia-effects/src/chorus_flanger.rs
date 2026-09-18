//! Modulation effects: chorus and flanger.
//!
//! Both effects use LFO-modulated delay lines to create the characteristic
//! thickening and sweeping sounds.
//!
//! - [`ChorusEffect`] — Multi-voice chorus with independent per-voice LFO phases.
//! - [`FlangerEffect`] — Single delay path with narrow LFO sweep and feedback.

use std::f32::consts::PI;

// ---------------------------------------------------------------------------
// ChorusEffect
// ---------------------------------------------------------------------------

/// Multi-voice chorus effect.
///
/// Each voice reads from the same circular delay buffer at a slightly
/// different modulation phase, producing a rich, de-tuned ensemble sound.
///
/// # Parameters
///
/// | Field | Default | Description |
/// |-------|---------|-------------|
/// | `depth_ms` | 5.0 | Modulation depth (half-swing) in milliseconds |
/// | `rate_hz` | 1.5 | LFO rate in Hz |
/// | `wet_mix` | 0.5 | Wet level |
/// | `dry_mix` | 0.5 | Dry level |
/// | `voices` | 3 | Number of chorus voices (1–8) |
/// | `stereo_spread` | 0.5 | Phase offset fraction between L/R LFOs |
pub struct ChorusEffect {
    /// Modulation depth in milliseconds.
    pub depth_ms: f32,
    /// LFO rate in Hz.
    pub rate_hz: f32,
    /// Wet (chorus) mix level.
    pub wet_mix: f32,
    /// Dry (direct) mix level.
    pub dry_mix: f32,
    /// Number of active chorus voices.
    pub voices: usize,
    /// Phase spread between left and right channels (fraction of 2π).
    pub stereo_spread: f32,

    // Internal state
    delay_buffers: Vec<Vec<f32>>,
    write_pos: usize,
    lfo_phase: f32,
    sample_rate: u32,
}

impl ChorusEffect {
    /// Maximum delay buffer size: base 30 ms + depth.  Allocated at creation.
    const BASE_DELAY_MS: f32 = 15.0;
    /// Hard maximum number of voices.
    const MAX_VOICES: usize = 8;

    /// Create a new chorus effect.
    ///
    /// `voices` is clamped to 1–8.
    #[must_use]
    pub fn new(sample_rate: u32, voices: usize) -> Self {
        let voices = voices.clamp(1, Self::MAX_VOICES);
        let sr = sample_rate.max(1) as f32;

        // Buffer large enough for base delay + maximum depth (default 5 ms + base).
        let max_delay_ms = Self::BASE_DELAY_MS + 5.0;
        #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
        let buf_len = ((max_delay_ms / 1000.0 * sr).ceil() as usize + 4).next_power_of_two();

        let delay_buffers = vec![vec![0.0_f32; buf_len]; voices];

        Self {
            depth_ms: 5.0,
            rate_hz: 1.5,
            wet_mix: 0.5,
            dry_mix: 0.5,
            voices,
            stereo_spread: 0.5,
            delay_buffers,
            write_pos: 0,
            lfo_phase: 0.0,
            sample_rate,
        }
    }

    /// Resize delay buffers when depth_ms changes.
    fn ensure_buffer_capacity(&mut self) {
        let sr = self.sample_rate.max(1) as f32;
        let max_delay_ms = Self::BASE_DELAY_MS + self.depth_ms;
        #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
        let needed = ((max_delay_ms / 1000.0 * sr).ceil() as usize + 4).next_power_of_two();
        let current = self.delay_buffers.first().map_or(0, |b| b.len());
        if needed > current {
            for buf in &mut self.delay_buffers {
                buf.resize(needed, 0.0);
            }
        }
    }

    /// Read a linearly-interpolated sample from a delay buffer.
    fn read_interp(buffer: &[f32], write_pos: usize, delay_samples: f32) -> f32 {
        let buf_len = buffer.len();
        let delay_samples = delay_samples.max(0.0);
        #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
        let delay_int = delay_samples as usize;
        let frac = delay_samples - delay_int as f32;

        let idx0 = (write_pos + buf_len - delay_int) % buf_len;
        let idx1 = (write_pos + buf_len - delay_int.saturating_sub(1)) % buf_len;
        buffer[idx0] * (1.0 - frac) + buffer[idx1] * frac
    }

    /// Process a single mono sample.
    ///
    /// All voices share the same write position but read at phase-offset delays.
    pub fn process_sample(&mut self, input: f32) -> f32 {
        self.ensure_buffer_capacity();
        let sr = self.sample_rate.max(1) as f32;
        let buf_len = self.delay_buffers.first().map_or(1, |b| b.len());

        #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
        let base_delay = (Self::BASE_DELAY_MS / 1000.0 * sr) as f32;
        let depth_samples = self.depth_ms / 1000.0 * sr;

        // Write input to all voice delay buffers.
        for buf in &mut self.delay_buffers {
            buf[self.write_pos] = input;
        }

        // Sum delayed signals from each voice.
        let mut wet_sum = 0.0_f32;
        let num_voices = self.voices;
        for v in 0..num_voices {
            #[allow(clippy::cast_precision_loss)]
            let voice_phase_offset = v as f32 / num_voices as f32;
            let phase = (self.lfo_phase + voice_phase_offset).rem_euclid(1.0);
            let lfo = (2.0 * PI * phase).sin(); // −1..+1
                                                // Map lfo to 0..1 swing then scale by depth.
            let delay_samples = base_delay + depth_samples * (lfo + 1.0) * 0.5;
            let delayed = Self::read_interp(&self.delay_buffers[v], self.write_pos, delay_samples);
            wet_sum += delayed;
        }

        // Advance write position.
        self.write_pos = (self.write_pos + 1) % buf_len;

        // Advance LFO phase.
        self.lfo_phase = (self.lfo_phase + self.rate_hz / sr).rem_euclid(1.0);

        #[allow(clippy::cast_precision_loss)]
        let wet = wet_sum / num_voices as f32;
        input * self.dry_mix + wet * self.wet_mix
    }

    /// Process a buffer of mono samples, returning a new `Vec<f32>`.
    #[must_use]
    pub fn process(&mut self, samples: &[f32]) -> Vec<f32> {
        samples.iter().map(|&s| self.process_sample(s)).collect()
    }

    /// Reset all internal state.
    pub fn reset(&mut self) {
        for buf in &mut self.delay_buffers {
            buf.fill(0.0);
        }
        self.write_pos = 0;
        self.lfo_phase = 0.0;
    }
}

// ---------------------------------------------------------------------------
// FlangerEffect
// ---------------------------------------------------------------------------

/// Flanger effect with sine LFO delay sweep and feedback.
///
/// The flanger sweeps a short delay (typically < 10 ms) with a sine LFO,
/// mixing the delayed signal back with the dry signal and feeding back into
/// the delay line for resonant comb-filter sweeps.
///
/// # Parameters
///
/// | Field | Default | Description |
/// |-------|---------|-------------|
/// | `min_delay_ms` | 0.1 | Minimum delay in milliseconds |
/// | `max_delay_ms` | 7.0 | Maximum delay in milliseconds |
/// | `rate_hz` | 0.5 | LFO sweep rate in Hz |
/// | `feedback` | 0.7 | Feedback amount (−0.95–0.95) |
/// | `wet_mix` | 0.5 | Wet level |
/// | `dry_mix` | 0.5 | Dry level |
pub struct FlangerEffect {
    /// Minimum delay in milliseconds.
    pub min_delay_ms: f32,
    /// Maximum delay in milliseconds.
    pub max_delay_ms: f32,
    /// LFO rate in Hz.
    pub rate_hz: f32,
    /// Feedback gain (clamped to −0.95–0.95 during processing).
    pub feedback: f32,
    /// Wet mix level.
    pub wet_mix: f32,
    /// Dry mix level.
    pub dry_mix: f32,

    // Internal state
    delay_buffer: Vec<f32>,
    write_pos: usize,
    lfo_phase: f32,
    feedback_sample: f32,
    sample_rate: u32,
}

impl FlangerEffect {
    /// Create a new flanger for the given sample rate.
    #[must_use]
    pub fn new(sample_rate: u32) -> Self {
        let sr = sample_rate.max(1) as f32;
        let max_delay_ms = 7.0_f32;
        #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
        let buf_len = ((max_delay_ms / 1000.0 * sr * 2.0).ceil() as usize + 4).next_power_of_two();

        Self {
            min_delay_ms: 0.1,
            max_delay_ms,
            rate_hz: 0.5,
            feedback: 0.7,
            wet_mix: 0.5,
            dry_mix: 0.5,
            delay_buffer: vec![0.0_f32; buf_len],
            write_pos: 0,
            lfo_phase: 0.0,
            feedback_sample: 0.0,
            sample_rate,
        }
    }

    /// Resize the delay buffer to accommodate the current max_delay_ms setting.
    fn ensure_capacity(&mut self) {
        let sr = self.sample_rate.max(1) as f32;
        #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
        let needed =
            ((self.max_delay_ms / 1000.0 * sr * 2.0).ceil() as usize + 4).next_power_of_two();
        if needed > self.delay_buffer.len() {
            self.delay_buffer.resize(needed, 0.0);
        }
    }

    /// Read a linearly-interpolated sample from the delay line.
    fn read_interp(&self, delay_samples: f32) -> f32 {
        let buf_len = self.delay_buffer.len();
        let delay_samples = delay_samples.max(0.0);
        #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
        let delay_int = delay_samples as usize;
        let frac = delay_samples - delay_int as f32;

        let idx0 = (self.write_pos + buf_len - delay_int) % buf_len;
        let idx1 = (self.write_pos + buf_len - delay_int.saturating_sub(1)) % buf_len;
        self.delay_buffer[idx0] * (1.0 - frac) + self.delay_buffer[idx1] * frac
    }

    /// Process a single mono sample.
    pub fn process_sample(&mut self, input: f32) -> f32 {
        self.ensure_capacity();
        let sr = self.sample_rate.max(1) as f32;
        let buf_len = self.delay_buffer.len();

        // Sine LFO in [0, 1].
        let lfo = (2.0 * PI * self.lfo_phase).sin() * 0.5 + 0.5;

        let min_d = self.min_delay_ms / 1000.0 * sr;
        let max_d = self.max_delay_ms / 1000.0 * sr;
        let delay_samples = min_d + lfo * (max_d - min_d);

        // Clamp feedback to safe range.
        let fb = self.feedback.clamp(-0.95, 0.95);

        // Write input + feedback into delay line.
        self.delay_buffer[self.write_pos] = input + self.feedback_sample * fb;
        self.write_pos = (self.write_pos + 1) % buf_len;

        // Read modulated tap.
        let delayed = self.read_interp(delay_samples);
        self.feedback_sample = delayed;

        // Advance LFO.
        self.lfo_phase = (self.lfo_phase + self.rate_hz / sr).rem_euclid(1.0);

        input * self.dry_mix + delayed * self.wet_mix
    }

    /// Process a buffer of mono samples, returning a new `Vec<f32>`.
    #[must_use]
    pub fn process(&mut self, samples: &[f32]) -> Vec<f32> {
        samples.iter().map(|&s| self.process_sample(s)).collect()
    }

    /// Reset all internal state.
    pub fn reset(&mut self) {
        self.delay_buffer.fill(0.0);
        self.write_pos = 0;
        self.lfo_phase = 0.0;
        self.feedback_sample = 0.0;
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // ---- ChorusEffect tests ----

    #[test]
    fn test_chorus_new_voices_clamped() {
        let c0 = ChorusEffect::new(44100, 0); // clamp to 1
        assert_eq!(c0.voices, 1);
        let c9 = ChorusEffect::new(44100, 9); // clamp to 8
        assert_eq!(c9.voices, 8);
    }

    #[test]
    fn test_chorus_output_length() {
        let mut c = ChorusEffect::new(44100, 3);
        let input = vec![0.5_f32; 256];
        let out = c.process(&input);
        assert_eq!(out.len(), 256);
    }

    #[test]
    fn test_chorus_output_finite() {
        let mut c = ChorusEffect::new(44100, 3);
        let input: Vec<f32> = (0..512).map(|i| (i as f32 * 0.02).sin()).collect();
        let out = c.process(&input);
        for (i, &s) in out.iter().enumerate() {
            assert!(s.is_finite(), "chorus out[{i}] not finite: {s}");
        }
    }

    #[test]
    fn test_chorus_dry_only() {
        let mut c = ChorusEffect::new(44100, 3);
        c.wet_mix = 0.0;
        c.dry_mix = 1.0;
        let out = c.process_sample(0.75);
        assert!((out - 0.75).abs() < 1e-5, "Dry-only chorus failed: {out}");
    }

    #[test]
    fn test_chorus_wet_only_not_zero() {
        // Wet-only chorus — after the buffer fills, output should not always be zero.
        let mut c = ChorusEffect::new(44100, 3);
        c.wet_mix = 1.0;
        c.dry_mix = 0.0;
        // Fill the buffer with constant signal.
        for _ in 0..2048 {
            c.process_sample(0.5);
        }
        let out = c.process_sample(0.5);
        // The delayed signal should be non-zero now.
        assert!(
            out.abs() > 0.0,
            "Wet-only chorus should produce non-zero output after fill"
        );
    }

    #[test]
    fn test_chorus_lfo_advances() {
        let mut c = ChorusEffect::new(44100, 3);
        let phase0 = c.lfo_phase;
        c.process_sample(0.0);
        assert_ne!(c.lfo_phase, phase0, "LFO phase should advance");
    }

    #[test]
    fn test_chorus_reset_clears_state() {
        let mut c = ChorusEffect::new(44100, 3);
        for _ in 0..512 {
            c.process_sample(1.0);
        }
        c.reset();
        for buf in &c.delay_buffers {
            for &s in buf {
                assert_eq!(s, 0.0);
            }
        }
        assert_eq!(c.write_pos, 0);
        assert_eq!(c.lfo_phase, 0.0);
    }

    #[test]
    fn test_chorus_multiple_voice_counts() {
        for v in [1, 2, 4, 8] {
            let mut c = ChorusEffect::new(44100, v);
            let out = c.process_sample(0.5);
            assert!(out.is_finite(), "voices={v}: output not finite: {out}");
        }
    }

    #[test]
    fn test_chorus_different_rates() {
        for rate in [0.1_f32, 1.5, 5.0, 10.0] {
            let mut c = ChorusEffect::new(44100, 3);
            c.rate_hz = rate;
            let input = vec![0.5_f32; 256];
            let out = c.process(&input);
            for &s in &out {
                assert!(s.is_finite(), "rate={rate}: output not finite: {s}");
            }
        }
    }

    #[test]
    fn test_chorus_different_depths() {
        for depth in [1.0_f32, 5.0, 15.0] {
            let mut c = ChorusEffect::new(44100, 3);
            c.depth_ms = depth;
            let input = vec![0.3_f32; 256];
            let out = c.process(&input);
            for &s in &out {
                assert!(s.is_finite(), "depth={depth}ms: output not finite: {s}");
            }
        }
    }

    #[test]
    fn test_chorus_write_pos_wraps() {
        let mut c = ChorusEffect::new(44100, 1);
        let buf_len = c.delay_buffers[0].len();
        for _ in 0..(buf_len + 10) {
            c.process_sample(0.5);
            assert!(
                c.write_pos < buf_len,
                "write_pos out of bounds: {}",
                c.write_pos
            );
        }
    }

    // ---- FlangerEffect tests ----

    #[test]
    fn test_flanger_new_buffer_power_of_two() {
        let f = FlangerEffect::new(44100);
        assert!(
            f.delay_buffer.len().is_power_of_two(),
            "Buffer length should be power-of-two"
        );
    }

    #[test]
    fn test_flanger_output_finite() {
        let mut f = FlangerEffect::new(44100);
        let input: Vec<f32> = (0..512).map(|i| (i as f32 * 0.02).sin()).collect();
        let out = f.process(&input);
        for (i, &s) in out.iter().enumerate() {
            assert!(s.is_finite(), "flanger out[{i}] not finite: {s}");
        }
    }

    #[test]
    fn test_flanger_silence_in_silence_out() {
        let mut f = FlangerEffect::new(44100);
        // All-zero input should produce all-zero output (no feedback buildup).
        for _ in 0..1024 {
            let out = f.process_sample(0.0);
            assert!(out.abs() < 1e-9, "Non-zero output for silence: {out}");
        }
    }

    #[test]
    fn test_flanger_dry_only() {
        let mut f = FlangerEffect::new(44100);
        f.wet_mix = 0.0;
        f.dry_mix = 1.0;
        f.feedback = 0.0;
        let out = f.process_sample(0.6);
        assert!((out - 0.6).abs() < 1e-5, "Dry-only flanger: {out}");
    }

    #[test]
    fn test_flanger_lfo_advances() {
        let mut f = FlangerEffect::new(44100);
        let p0 = f.lfo_phase;
        f.process_sample(0.0);
        assert_ne!(f.lfo_phase, p0, "LFO phase should advance");
    }

    #[test]
    fn test_flanger_reset_clears_state() {
        let mut f = FlangerEffect::new(44100);
        for _ in 0..512 {
            f.process_sample(1.0);
        }
        f.reset();
        for &s in &f.delay_buffer {
            assert_eq!(s, 0.0);
        }
        assert_eq!(f.write_pos, 0);
        assert_eq!(f.lfo_phase, 0.0);
        assert_eq!(f.feedback_sample, 0.0);
    }

    #[test]
    fn test_flanger_output_length() {
        let mut f = FlangerEffect::new(44100);
        let input = vec![0.3_f32; 512];
        let out = f.process(&input);
        assert_eq!(out.len(), 512);
    }

    #[test]
    fn test_flanger_feedback_clamped_safe() {
        // High feedback should not diverge.
        let mut f = FlangerEffect::new(44100);
        f.feedback = 0.95; // Will be clamped to 0.95 internally.
        let input = vec![0.5_f32; 2048];
        let out = f.process(&input);
        for (i, &s) in out.iter().enumerate() {
            assert!(
                s.is_finite(),
                "flanger (high feedback) out[{i}] not finite: {s}"
            );
        }
    }

    #[test]
    fn test_flanger_write_pos_wraps() {
        let mut f = FlangerEffect::new(44100);
        let buf_len = f.delay_buffer.len();
        for _ in 0..(buf_len + 10) {
            f.process_sample(0.5);
            assert!(
                f.write_pos < buf_len,
                "write_pos out of bounds: {}",
                f.write_pos
            );
        }
    }
}
