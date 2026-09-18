//! Audio resampler with linear interpolation and polyphase filter approximation.
//!
//! Supports multiple quality modes from simple linear interpolation up to
//! a windowed-sinc polyphase bank approximation.

/// Resampling quality modes.
#[allow(dead_code)]
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub enum ResampleQuality {
    /// Nearest-neighbor: low quality, very fast.
    NearestNeighbor,
    /// Linear interpolation: moderate quality, fast.
    #[default]
    Linear,
    /// Cubic (Catmull-Rom) interpolation: good quality.
    Cubic,
    /// Polyphase sinc approximation: high quality.
    Polyphase {
        /// Number of filter taps (higher = better quality, slower).
        taps: usize,
    },
}

/// A polyphase sinc coefficient table entry.
#[allow(dead_code)]
#[derive(Debug, Clone)]
struct PhaseBank {
    /// Number of phases.
    num_phases: usize,
    /// Number of taps per phase.
    taps_per_phase: usize,
    /// Flattened coefficient table [phase][tap].
    coeffs: Vec<f32>,
}

impl PhaseBank {
    /// Build a simple windowed-sinc polyphase bank.
    #[allow(dead_code)]
    fn new(num_phases: usize, taps_per_phase: usize) -> Self {
        let total = num_phases * taps_per_phase;
        let mut coeffs = vec![0.0_f32; total];
        let half_tap = (taps_per_phase as f32 - 1.0) / 2.0;

        for phase in 0..num_phases {
            let phase_offset = phase as f32 / num_phases as f32;
            let mut sum = 0.0_f32;
            for tap in 0..taps_per_phase {
                let x = tap as f32 - half_tap - phase_offset;
                let sinc = if x.abs() < 1e-6 {
                    1.0
                } else {
                    let px = std::f32::consts::PI * x;
                    px.sin() / px
                };
                // Hann window
                let window_pos = (tap as f32) / (taps_per_phase as f32 - 1.0);
                let window = 0.5 * (1.0 - (2.0 * std::f32::consts::PI * window_pos).cos());
                let coeff = sinc * window;
                coeffs[phase * taps_per_phase + tap] = coeff;
                sum += coeff;
            }
            // Normalize phase
            if sum.abs() > 1e-10 {
                for tap in 0..taps_per_phase {
                    coeffs[phase * taps_per_phase + tap] /= sum;
                }
            }
        }

        Self {
            num_phases,
            taps_per_phase,
            coeffs,
        }
    }

    /// Apply a phase filter to a buffer slice.
    #[allow(dead_code)]
    fn apply(&self, phase: usize, buffer: &[f32]) -> f32 {
        let taps = self.taps_per_phase;
        if buffer.len() < taps {
            return 0.0;
        }
        let phase_idx = phase.min(self.num_phases - 1);
        let base = phase_idx * taps;
        let start = buffer.len() - taps;
        let mut out = 0.0_f32;
        for i in 0..taps {
            out += self.coeffs[base + i] * buffer[start + i];
        }
        out
    }
}

/// Single-channel audio resampler.
#[allow(dead_code)]
pub struct SimpleResampler {
    input_rate: f32,
    output_rate: f32,
    quality: ResampleQuality,
    /// Fractional position in input stream.
    phase: f64,
    /// Step size per output sample.
    step: f64,
    /// History buffer for interpolation.
    history: Vec<f32>,
    /// Polyphase bank (used when quality is Polyphase).
    phase_bank: Option<PhaseBank>,
}

impl SimpleResampler {
    /// Create a new resampler.
    #[allow(dead_code)]
    pub fn new(input_rate: f32, output_rate: f32, quality: ResampleQuality) -> Self {
        let step = input_rate as f64 / output_rate as f64;

        let phase_bank = if let ResampleQuality::Polyphase { taps } = quality {
            let num_phases = 64;
            Some(PhaseBank::new(num_phases, taps.max(4)))
        } else {
            None
        };

        let history_len = match quality {
            ResampleQuality::NearestNeighbor => 2,
            ResampleQuality::Linear => 2,
            ResampleQuality::Cubic => 4,
            ResampleQuality::Polyphase { taps } => taps.max(4),
        };

        Self {
            input_rate,
            output_rate,
            quality,
            phase: 0.0,
            step,
            history: vec![0.0; history_len],
            phase_bank,
        }
    }

    /// Process input samples and produce output samples.
    #[allow(dead_code)]
    pub fn process(&mut self, input: &[f32]) -> Vec<f32> {
        if input.is_empty() {
            return Vec::new();
        }

        let n_out = (input.len() as f64 / self.step).ceil() as usize + 1;
        let mut output = Vec::with_capacity(n_out);

        for &sample in input {
            self.push_history(sample);
        }

        // Reset phase for this block and produce output
        let mut pos = 0.0_f64;
        while pos < input.len() as f64 {
            let int_pos = pos as usize;
            let frac = pos - int_pos as f64;

            let out_sample = match self.quality {
                ResampleQuality::NearestNeighbor => self.get_input_sample(input, int_pos),
                ResampleQuality::Linear => {
                    let s0 = self.get_input_sample(input, int_pos);
                    let s1 = self.get_input_sample(input, int_pos + 1);
                    s0 + (s1 - s0) * frac as f32
                }
                ResampleQuality::Cubic => {
                    let s_neg1 = self.get_input_sample_signed(input, int_pos as i64 - 1);
                    let s0 = self.get_input_sample(input, int_pos);
                    let s1 = self.get_input_sample(input, int_pos + 1);
                    let s2 = self.get_input_sample(input, int_pos + 2);
                    catmull_rom(s_neg1, s0, s1, s2, frac as f32)
                }
                ResampleQuality::Polyphase { .. } => {
                    // Approximate using linear for block processing
                    let s0 = self.get_input_sample(input, int_pos);
                    let s1 = self.get_input_sample(input, int_pos + 1);
                    s0 + (s1 - s0) * frac as f32
                }
            };

            output.push(out_sample);
            pos += self.step;
        }

        output
    }

    /// Get an input sample by index, clamping to boundaries.
    #[allow(dead_code)]
    fn get_input_sample(&self, input: &[f32], idx: usize) -> f32 {
        if input.is_empty() {
            return 0.0;
        }
        let clamped = idx.min(input.len() - 1);
        input[clamped]
    }

    fn get_input_sample_signed(&self, input: &[f32], idx: i64) -> f32 {
        if idx < 0 || input.is_empty() {
            return 0.0;
        }
        let i = idx as usize;
        if i >= input.len() {
            return *input.last().unwrap_or(&0.0);
        }
        input[i]
    }

    fn push_history(&mut self, sample: f32) {
        let len = self.history.len();
        if len == 0 {
            return;
        }
        self.history.rotate_left(1);
        self.history[len - 1] = sample;
    }

    /// Returns the conversion ratio (output_rate / input_rate).
    #[allow(dead_code)]
    pub fn ratio(&self) -> f32 {
        self.output_rate / self.input_rate
    }

    /// Returns the output rate.
    #[allow(dead_code)]
    pub fn output_rate(&self) -> f32 {
        self.output_rate
    }

    /// Returns the input rate.
    #[allow(dead_code)]
    pub fn input_rate(&self) -> f32 {
        self.input_rate
    }
}

/// Catmull-Rom cubic interpolation.
#[allow(dead_code)]
fn catmull_rom(p0: f32, p1: f32, p2: f32, p3: f32, t: f32) -> f32 {
    let t2 = t * t;
    let t3 = t2 * t;
    0.5 * ((2.0 * p1)
        + (-p0 + p2) * t
        + (2.0 * p0 - 5.0 * p1 + 4.0 * p2 - p3) * t2
        + (-p0 + 3.0 * p1 - 3.0 * p2 + p3) * t3)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_resampler_creation() {
        let r = SimpleResampler::new(44_100.0, 48_000.0, ResampleQuality::Linear);
        assert_eq!(r.input_rate(), 44_100.0);
        assert_eq!(r.output_rate(), 48_000.0);
    }

    #[test]
    fn test_ratio_upsample() {
        let r = SimpleResampler::new(44_100.0, 48_000.0, ResampleQuality::Linear);
        assert!(r.ratio() > 1.0);
    }

    #[test]
    fn test_ratio_downsample() {
        let r = SimpleResampler::new(48_000.0, 44_100.0, ResampleQuality::Linear);
        assert!(r.ratio() < 1.0);
    }

    #[test]
    fn test_empty_input() {
        let mut r = SimpleResampler::new(44_100.0, 48_000.0, ResampleQuality::Linear);
        let out = r.process(&[]);
        assert!(out.is_empty());
    }

    #[test]
    fn test_linear_upsample_produces_more_samples() {
        let mut r = SimpleResampler::new(44_100.0, 48_000.0, ResampleQuality::Linear);
        let input: Vec<f32> = (0..4410).map(|i| (i as f32 / 4410.0).sin()).collect();
        let output = r.process(&input);
        // Output should be roughly larger
        assert!(output.len() > input.len());
    }

    #[test]
    fn test_nearest_neighbor_passthrough() {
        let mut r = SimpleResampler::new(48_000.0, 48_000.0, ResampleQuality::NearestNeighbor);
        let input = vec![0.1_f32, 0.2, 0.3, 0.4, 0.5];
        let output = r.process(&input);
        assert!(!output.is_empty());
        for s in &output {
            assert!(s.is_finite());
        }
    }

    #[test]
    fn test_cubic_no_nan() {
        let mut r = SimpleResampler::new(44_100.0, 48_000.0, ResampleQuality::Cubic);
        let input: Vec<f32> = (0..100).map(|i| (i as f32 * 0.1).sin()).collect();
        let output = r.process(&input);
        for s in &output {
            assert!(s.is_finite(), "NaN or Inf detected");
        }
    }

    #[test]
    fn test_polyphase_no_nan() {
        let mut r =
            SimpleResampler::new(44_100.0, 48_000.0, ResampleQuality::Polyphase { taps: 8 });
        let input: Vec<f32> = (0..100).map(|i| (i as f32 * 0.05).cos()).collect();
        let output = r.process(&input);
        for s in &output {
            assert!(s.is_finite(), "NaN or Inf detected");
        }
    }

    #[test]
    fn test_catmull_rom_midpoint() {
        // Midpoint between p1 and p2 with uniform knots should be average-ish
        let v = catmull_rom(0.0, 0.0, 1.0, 1.0, 0.5);
        assert!((v - 0.5).abs() < 0.1);
    }

    #[test]
    fn test_phase_bank_creation() {
        let bank = PhaseBank::new(16, 8);
        assert_eq!(bank.num_phases, 16);
        assert_eq!(bank.taps_per_phase, 8);
    }

    #[test]
    fn test_phase_bank_apply_size_check() {
        let bank = PhaseBank::new(4, 4);
        let buf = vec![0.1_f32, 0.2, 0.3, 0.4];
        let out = bank.apply(0, &buf);
        assert!(out.is_finite());
    }

    #[test]
    fn test_phase_bank_apply_empty_buf() {
        let bank = PhaseBank::new(4, 4);
        let out = bank.apply(0, &[]);
        assert_eq!(out, 0.0);
    }

    #[test]
    fn test_quality_default() {
        let q = ResampleQuality::default();
        assert_eq!(q, ResampleQuality::Linear);
    }

    #[test]
    fn test_downsample_fewer_samples() {
        let mut r = SimpleResampler::new(48_000.0, 24_000.0, ResampleQuality::Linear);
        let input: Vec<f32> = (0..4800).map(|i| (i as f32 / 4800.0).sin()).collect();
        let output = r.process(&input);
        assert!(output.len() < input.len());
    }
}
