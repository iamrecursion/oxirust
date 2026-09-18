//! Audio mixing, bus management, and channel routing.
//!
//! Provides flexible mix-matrix routing and per-bus level/mute/solo control,
//! suited for DAW-style mixing pipelines.

/// A single named audio bus with level and mute/solo state.
pub struct AudioBus {
    /// Human-readable bus name.
    pub name: String,
    /// Number of audio channels on this bus.
    pub channels: u8,
    /// Fader level in dB (0.0 = unity, negative = attenuated).
    pub level_db: f64,
    /// Whether the bus is muted.
    pub muted: bool,
    /// Whether the bus is soloed.
    pub solo: bool,
}

impl AudioBus {
    /// Create a new bus at unity gain, unmuted and not soloed.
    pub fn new(name: &str, channels: u8) -> Self {
        Self {
            name: name.to_owned(),
            channels,
            level_db: 0.0,
            muted: false,
            solo: false,
        }
    }

    /// Convert the bus fader level from dB to a linear gain factor.
    pub fn gain_linear(&self) -> f64 {
        db_to_linear(self.level_db)
    }

    /// Return the effective level in dB, taking mute and solo into account.
    ///
    /// - If `any_solo` is `true` and this bus is not soloed, returns `None`
    ///   (bus is effectively silent).
    /// - If the bus is muted, returns `None`.
    /// - Otherwise returns `Some(level_db)`.
    pub fn effective_level_db(&self, any_solo: bool) -> Option<f64> {
        if self.muted {
            return None;
        }
        if any_solo && !self.solo {
            return None;
        }
        Some(self.level_db)
    }
}

// --- Mix matrix --------------------------------------------------------------

/// A gain matrix that routes `inputs` input channels to `outputs` output
/// channels with individual gain factors.
pub struct MixMatrix {
    /// Number of input channels.
    pub inputs: usize,
    /// Number of output channels.
    pub outputs: usize,
    /// Row-major gain table: `gains[input * outputs + output]`.
    gains: Vec<f64>,
}

impl MixMatrix {
    /// Create a zero-initialised mix matrix.
    pub fn new(inputs: usize, outputs: usize) -> Self {
        Self {
            inputs,
            outputs,
            gains: vec![0.0; inputs * outputs],
        }
    }

    /// Set the gain for the route from `input` to `output`.
    ///
    /// Panics if indices are out of range.
    pub fn set_gain(&mut self, input: usize, output: usize, gain: f64) {
        assert!(input < self.inputs, "input channel index out of range");
        assert!(output < self.outputs, "output channel index out of range");
        self.gains[input * self.outputs + output] = gain;
    }

    /// Get the gain for the route from `input` to `output`.
    pub fn get_gain(&self, input: usize, output: usize) -> f64 {
        assert!(input < self.inputs, "input channel index out of range");
        assert!(output < self.outputs, "output channel index out of range");
        self.gains[input * self.outputs + output]
    }

    /// Build a 1-input → 2-output (mono-to-stereo) matrix at unity gain.
    pub fn mono_to_stereo() -> Self {
        let mut m = Self::new(1, 2);
        m.set_gain(0, 0, 1.0);
        m.set_gain(0, 1, 1.0);
        m
    }

    /// Build a 2-input → 1-output (stereo-to-mono) matrix at -3 dB per channel.
    pub fn stereo_to_mono() -> Self {
        let g = db_to_linear(-3.0103); // ≈ 0.707 (equal power)
        let mut m = Self::new(2, 1);
        m.set_gain(0, 0, g);
        m.set_gain(1, 0, g);
        m
    }
}

// --- free functions ----------------------------------------------------------

/// Apply `matrix` to `inputs` and accumulate the result into `output`.
///
/// - `inputs`: slice of mono input channel buffers (each must be the same length).
/// - `matrix`: routing matrix with `inputs.len()` inputs and `output.len()` outputs.
/// - `output`: mutable slice of output channel buffers, pre-allocated to the
///   desired frame length (values are *accumulated*, not replaced).
///
/// # Panics
///
/// Panics if the number of inputs or outputs does not match the matrix dimensions,
/// or if buffer lengths differ.
pub fn mix_channels(inputs: &[&[f64]], matrix: &MixMatrix, output: &mut [Vec<f64>]) {
    assert_eq!(inputs.len(), matrix.inputs, "input count mismatch");
    assert_eq!(output.len(), matrix.outputs, "output count mismatch");

    if inputs.is_empty() || output.is_empty() {
        return;
    }

    let frame_len = inputs[0].len();

    for (in_idx, in_buf) in inputs.iter().enumerate() {
        assert_eq!(in_buf.len(), frame_len, "input buffer length mismatch");
        for out_idx in 0..matrix.outputs {
            let gain = matrix.get_gain(in_idx, out_idx);
            if gain == 0.0 {
                continue;
            }
            let out_buf = &mut output[out_idx];
            assert_eq!(out_buf.len(), frame_len, "output buffer length mismatch");
            for (s, &inp) in out_buf.iter_mut().zip(in_buf.iter()) {
                *s += inp * gain;
            }
        }
    }
}

/// Mix multiple audio sources together with individual linear gain factors.
///
/// Each entry in `buses` is a tuple of `(samples, gain)`.  All sample slices
/// must have the same length.  Returns a new `Vec<f64>` of that length.
pub fn sum_buses(buses: &[(&[f64], f64)]) -> Vec<f64> {
    if buses.is_empty() {
        return Vec::new();
    }
    let len = buses[0].0.len();
    let mut out = vec![0.0f64; len];
    for (buf, gain) in buses {
        assert_eq!(buf.len(), len, "bus buffer length mismatch");
        for (o, &s) in out.iter_mut().zip(buf.iter()) {
            *o += s * gain;
        }
    }
    out
}

// --- helpers -----------------------------------------------------------------

fn db_to_linear(db: f64) -> f64 {
    10.0_f64.powf(db / 20.0)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_bus_gain_linear_unity() {
        let bus = AudioBus::new("main", 2);
        let gain = bus.gain_linear();
        assert!((gain - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_bus_gain_linear_minus6db() {
        let mut bus = AudioBus::new("aux", 2);
        bus.level_db = -6.0;
        let gain = bus.gain_linear();
        // -6 dBFS ≈ 0.501
        assert!((gain - 0.501_187).abs() < 1e-4);
    }

    #[test]
    fn test_bus_effective_level_not_muted() {
        let bus = AudioBus::new("ch1", 1);
        assert_eq!(bus.effective_level_db(false), Some(0.0));
    }

    #[test]
    fn test_bus_effective_level_muted() {
        let mut bus = AudioBus::new("ch1", 1);
        bus.muted = true;
        assert_eq!(bus.effective_level_db(false), None);
    }

    #[test]
    fn test_bus_effective_level_solo_excluded() {
        let bus = AudioBus::new("ch2", 1); // solo = false
                                           // another bus is soloed
        assert_eq!(bus.effective_level_db(true), None);
    }

    #[test]
    fn test_bus_effective_level_solo_included() {
        let mut bus = AudioBus::new("ch2", 1);
        bus.solo = true;
        assert_eq!(bus.effective_level_db(true), Some(0.0));
    }

    #[test]
    fn test_mix_matrix_get_set() {
        let mut m = MixMatrix::new(2, 3);
        m.set_gain(1, 2, 0.5);
        assert!((m.get_gain(1, 2) - 0.5).abs() < f64::EPSILON);
        assert_eq!(m.get_gain(0, 0), 0.0);
    }

    #[test]
    fn test_mono_to_stereo_matrix() {
        let m = MixMatrix::mono_to_stereo();
        assert_eq!(m.inputs, 1);
        assert_eq!(m.outputs, 2);
        assert!((m.get_gain(0, 0) - 1.0).abs() < f64::EPSILON);
        assert!((m.get_gain(0, 1) - 1.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_stereo_to_mono_matrix() {
        let m = MixMatrix::stereo_to_mono();
        assert_eq!(m.inputs, 2);
        assert_eq!(m.outputs, 1);
        // Both channels should be around 0.707
        let g = m.get_gain(0, 0);
        assert!((g - 0.707).abs() < 0.001);
    }

    #[test]
    fn test_mix_channels_mono_to_stereo() {
        let m = MixMatrix::mono_to_stereo();
        let input = vec![1.0f64, 0.5, -0.5];
        let mut out = vec![vec![0.0f64; 3], vec![0.0f64; 3]];
        mix_channels(&[&input], &m, &mut out);
        assert!((out[0][0] - 1.0).abs() < f64::EPSILON);
        assert!((out[1][0] - 1.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_sum_buses_single() {
        let buf = vec![1.0f64, 2.0, 3.0];
        let out = sum_buses(&[(&buf, 0.5)]);
        assert!((out[0] - 0.5).abs() < f64::EPSILON);
        assert!((out[1] - 1.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_sum_buses_two() {
        let a = vec![1.0f64; 4];
        let b = vec![2.0f64; 4];
        let out = sum_buses(&[(&a, 1.0), (&b, 0.5)]);
        // 1.0*1 + 2.0*0.5 = 2.0
        for v in &out {
            assert!((*v - 2.0).abs() < f64::EPSILON);
        }
    }

    #[test]
    fn test_sum_buses_empty() {
        let out = sum_buses(&[]);
        assert!(out.is_empty());
    }

    #[test]
    fn test_db_to_linear_zero() {
        assert!((db_to_linear(0.0) - 1.0).abs() < 1e-9);
    }
}
