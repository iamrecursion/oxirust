//! Mixer routing matrix implementation.
//!
//! Provides a flexible gain-based routing matrix with preset configurations
//! and validation utilities.

/// A routing matrix that maps `inputs` audio channels to `outputs` channels via gain values.
///
/// `connections[input][output]` holds the gain (0.0 = silent, 1.0 = unity).
#[allow(dead_code)]
#[derive(Debug, Clone)]
pub struct RoutingMatrix {
    /// Number of input channels.
    pub inputs: u32,
    /// Number of output channels.
    pub outputs: u32,
    /// Gain matrix: `connections[input_idx][output_idx]`.
    pub connections: Vec<Vec<f32>>,
}

impl RoutingMatrix {
    /// Create a new routing matrix initialized to all zeros.
    #[must_use]
    #[allow(dead_code)]
    pub fn new(inputs: u32, outputs: u32) -> Self {
        let connections = vec![vec![0.0f32; outputs as usize]; inputs as usize];
        Self {
            inputs,
            outputs,
            connections,
        }
    }

    /// Set the gain from `input` to `output`.
    ///
    /// Does nothing if indices are out of range.
    #[allow(dead_code)]
    pub fn set_gain(&mut self, input: u32, output: u32, gain: f32) {
        let i = input as usize;
        let o = output as usize;
        if i < self.connections.len() && o < self.outputs as usize {
            self.connections[i][o] = gain;
        }
    }

    /// Get the gain from `input` to `output`, or 0.0 if out of range.
    #[must_use]
    #[allow(dead_code)]
    pub fn get_gain(&self, input: u32, output: u32) -> f32 {
        let i = input as usize;
        let o = output as usize;
        self.connections
            .get(i)
            .and_then(|row| row.get(o))
            .copied()
            .unwrap_or(0.0)
    }

    /// Process one frame of samples through the routing matrix.
    ///
    /// `input_samples[i]` is the sample buffer for input channel `i`.
    /// Returns output sample buffers; each output buffer has the same length as the
    /// shortest input buffer.
    #[must_use]
    #[allow(dead_code)]
    pub fn process(&self, input_samples: &[Vec<f32>]) -> Vec<Vec<f32>> {
        let num_out = self.outputs as usize;
        if input_samples.is_empty() || num_out == 0 {
            return vec![Vec::new(); num_out];
        }
        let buf_len = input_samples
            .iter()
            .map(std::vec::Vec::len)
            .min()
            .unwrap_or(0);
        let mut outputs = vec![vec![0.0f32; buf_len]; num_out];

        for (in_idx, in_buf) in input_samples.iter().enumerate() {
            if in_idx >= self.connections.len() {
                break;
            }
            for (out_idx, out_buf) in outputs.iter_mut().enumerate() {
                let gain = self.connections[in_idx][out_idx];
                if gain.abs() < 1e-10 {
                    continue;
                }
                for (s, &v) in out_buf.iter_mut().zip(in_buf.iter()) {
                    *s += v * gain;
                }
            }
        }
        outputs
    }
}

/// Preset routing configurations.
#[allow(dead_code)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RoutingPreset {
    /// Stereo passthrough: input 0 → L, input 1 → R.
    Stereo,
    /// 5.1 surround: L, R, C, LFE, Ls, Rs.
    Surround51,
    /// 7.1 surround: L, R, C, LFE, Lss, Rss, Lrs, Rrs.
    Surround71,
    /// Mono mix: all inputs summed to output 0.
    Mono,
    /// LFE only: input 3 → output 0.
    LfeOnly,
}

impl RoutingPreset {
    /// Build a routing matrix for this preset.
    ///
    /// `inputs` is the number of source channels to connect.
    #[must_use]
    #[allow(dead_code)]
    pub fn build_matrix(self, inputs: u32) -> RoutingMatrix {
        match self {
            Self::Stereo => {
                let mut m = RoutingMatrix::new(inputs.max(2), 2);
                m.set_gain(0, 0, 1.0); // L → L
                m.set_gain(1, 1, 1.0); // R → R
                m
            }
            Self::Surround51 => {
                let mut m = RoutingMatrix::new(inputs.max(6), 6);
                for i in 0..6u32 {
                    m.set_gain(i, i, 1.0);
                }
                m
            }
            Self::Surround71 => {
                let mut m = RoutingMatrix::new(inputs.max(8), 8);
                for i in 0..8u32 {
                    m.set_gain(i, i, 1.0);
                }
                m
            }
            Self::Mono => {
                let n = inputs.max(1);
                let mut m = RoutingMatrix::new(n, 1);
                let gain = 1.0 / n as f32;
                for i in 0..n {
                    m.set_gain(i, 0, gain);
                }
                m
            }
            Self::LfeOnly => {
                let mut m = RoutingMatrix::new(inputs.max(4), 1);
                m.set_gain(3, 0, 1.0); // LFE channel (index 3) → output
                m
            }
        }
    }
}

/// Validates a routing matrix for common issues.
pub struct RoutingValidator;

impl RoutingValidator {
    /// Check the matrix for warnings.
    ///
    /// Returns a list of human-readable warning strings.
    #[must_use]
    #[allow(dead_code)]
    pub fn check(matrix: &RoutingMatrix) -> Vec<String> {
        let mut warnings = Vec::new();
        let n_in = matrix.inputs as usize;
        let n_out = matrix.outputs as usize;

        // Check for unconnected inputs (all gains to all outputs are zero)
        for i in 0..n_in {
            let connected = (0..n_out).any(|o| matrix.connections[i][o].abs() > 1e-8);
            if !connected {
                warnings.push(format!("Input {i} is not connected to any output"));
            }
        }

        // Check for unconnected outputs (all inputs to this output are zero)
        for o in 0..n_out {
            let connected = (0..n_in).any(|i| matrix.connections[i][o].abs() > 1e-8);
            if !connected {
                warnings.push(format!("Output {o} receives no input"));
            }
        }

        // Check for unity gain exceeded (sum of gains into any output > 2.0)
        for o in 0..n_out {
            let total: f32 = (0..n_in).map(|i| matrix.connections[i][o].abs()).sum();
            if total > 2.0 {
                warnings.push(format!(
                    "Output {o} has combined gain {total:.2} which may cause clipping"
                ));
            }
        }

        warnings
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_routing_matrix_new_zeros() {
        let m = RoutingMatrix::new(4, 2);
        for i in 0..4 {
            for o in 0..2 {
                assert_eq!(m.get_gain(i, o), 0.0);
            }
        }
    }

    #[test]
    fn test_set_and_get_gain() {
        let mut m = RoutingMatrix::new(2, 2);
        m.set_gain(0, 1, 0.7);
        assert!((m.get_gain(0, 1) - 0.7).abs() < f32::EPSILON);
        assert_eq!(m.get_gain(0, 0), 0.0);
    }

    #[test]
    fn test_set_gain_out_of_range_no_panic() {
        let mut m = RoutingMatrix::new(2, 2);
        m.set_gain(10, 10, 1.0); // Should silently do nothing
        assert_eq!(m.get_gain(10, 10), 0.0);
    }

    #[test]
    fn test_process_passthrough() {
        let mut m = RoutingMatrix::new(2, 2);
        m.set_gain(0, 0, 1.0);
        m.set_gain(1, 1, 1.0);

        let inputs = vec![vec![1.0f32, 2.0], vec![3.0f32, 4.0]];
        let outputs = m.process(&inputs);
        assert_eq!(outputs.len(), 2);
        assert!((outputs[0][0] - 1.0).abs() < 1e-6);
        assert!((outputs[1][1] - 4.0).abs() < 1e-6);
    }

    #[test]
    fn test_process_mono_mix() {
        let mut m = RoutingMatrix::new(2, 1);
        m.set_gain(0, 0, 0.5);
        m.set_gain(1, 0, 0.5);

        let inputs = vec![vec![1.0f32], vec![1.0f32]];
        let outputs = m.process(&inputs);
        assert_eq!(outputs.len(), 1);
        assert!((outputs[0][0] - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_preset_stereo_matrix() {
        let m = RoutingPreset::Stereo.build_matrix(2);
        assert_eq!(m.inputs, 2);
        assert_eq!(m.outputs, 2);
        assert!((m.get_gain(0, 0) - 1.0).abs() < f32::EPSILON);
        assert!((m.get_gain(1, 1) - 1.0).abs() < f32::EPSILON);
        assert!(m.get_gain(0, 1).abs() < f32::EPSILON);
    }

    #[test]
    fn test_preset_surround51_matrix() {
        let m = RoutingPreset::Surround51.build_matrix(6);
        assert_eq!(m.outputs, 6);
        for i in 0..6u32 {
            assert!((m.get_gain(i, i) - 1.0).abs() < f32::EPSILON);
        }
    }

    #[test]
    fn test_preset_mono_matrix() {
        let m = RoutingPreset::Mono.build_matrix(4);
        assert_eq!(m.outputs, 1);
        let expected_gain = 0.25;
        for i in 0..4u32 {
            assert!((m.get_gain(i, 0) - expected_gain).abs() < 1e-6);
        }
    }

    #[test]
    fn test_preset_lfe_only() {
        let m = RoutingPreset::LfeOnly.build_matrix(6);
        assert!((m.get_gain(3, 0) - 1.0).abs() < f32::EPSILON);
        assert!(m.get_gain(0, 0).abs() < f32::EPSILON);
    }

    #[test]
    fn test_validator_unconnected_input() {
        let m = RoutingMatrix::new(2, 2); // all zeros
        let warnings = RoutingValidator::check(&m);
        assert!(warnings.iter().any(|w| w.contains("Input 0")));
        assert!(warnings.iter().any(|w| w.contains("Output 0")));
    }

    #[test]
    fn test_validator_no_warnings_passthrough() {
        let mut m = RoutingMatrix::new(2, 2);
        m.set_gain(0, 0, 1.0);
        m.set_gain(1, 1, 1.0);
        let warnings = RoutingValidator::check(&m);
        // Unconnected cross paths but not complete disconnection
        let unconnected_warnings: Vec<_> = warnings
            .iter()
            .filter(|w| w.contains("not connected"))
            .collect();
        assert!(unconnected_warnings.is_empty());
    }

    #[test]
    fn test_process_empty_inputs() {
        let m = RoutingMatrix::new(2, 2);
        let outputs = m.process(&[]);
        assert_eq!(outputs.len(), 2);
    }
}
