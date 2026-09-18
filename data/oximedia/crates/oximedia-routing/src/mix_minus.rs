#![allow(dead_code)]
//! Mix-minus routing for broadcast IFB (Interruptible Foldback) feeds.
//!
//! In broadcast audio production, mix-minus (also called "N-1") is the
//! technique of sending each contributor a mix of *all* inputs *minus* their
//! own feed.  This prevents speakers from hearing themselves with an
//! annoying delay, while still receiving everyone else clearly.
//!
//! ## Example
//!
//! ```rust
//! use oximedia_routing::mix_minus::{MixMinusConfig, MixMinusMatrix};
//!
//! let config = MixMinusConfig { num_inputs: 3, num_outputs: 3 };
//! let mut matrix = MixMinusMatrix::new(config);
//!
//! // Each output excludes its own "ear" input
//! matrix.route(0, 0, true).expect("valid indices");
//! matrix.route(1, 1, true).expect("valid indices");
//! matrix.route(2, 2, true).expect("valid indices");
//!
//! let a = vec![1.0_f32; 4];
//! let b = vec![1.0_f32; 4];
//! let c = vec![1.0_f32; 4];
//! let mixed = matrix.mix(&[&a, &b, &c]).expect("valid mix");
//! // Each output = sum of 2 inputs = 2.0 per sample
//! assert!((mixed[0][0] - 2.0).abs() < 1e-6);
//! ```

use std::collections::HashMap;

// ---------------------------------------------------------------------------
// Error type
// ---------------------------------------------------------------------------

/// Errors that can occur during mix-minus operations.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum MixMinusError {
    /// The provided input index exceeds `num_inputs - 1`.
    InputIndexOutOfBounds {
        /// The offending index.
        index: usize,
        /// Maximum valid index.
        max: usize,
    },
    /// The provided output index exceeds `num_outputs - 1`.
    OutputIndexOutOfBounds {
        /// The offending index.
        index: usize,
        /// Maximum valid index.
        max: usize,
    },
    /// Not all input sample slices have the same length.
    SampleLengthMismatch {
        /// Index of the mismatched slice.
        slice_index: usize,
        /// Expected length (length of slice 0).
        expected: usize,
        /// Actual length of the mismatched slice.
        actual: usize,
    },
    /// The number of input slices passed to `mix` does not match `num_inputs`.
    InputCountMismatch {
        /// Expected number of inputs.
        expected: usize,
        /// Actual number of inputs supplied.
        actual: usize,
    },
}

impl std::fmt::Display for MixMinusError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::InputIndexOutOfBounds { index, max } => {
                write!(f, "input index {index} out of bounds (max {max})")
            }
            Self::OutputIndexOutOfBounds { index, max } => {
                write!(f, "output index {index} out of bounds (max {max})")
            }
            Self::SampleLengthMismatch {
                slice_index,
                expected,
                actual,
            } => {
                write!(
                    f,
                    "input slice {slice_index} has length {actual}, expected {expected}"
                )
            }
            Self::InputCountMismatch { expected, actual } => {
                write!(f, "expected {expected} input slices, got {actual}")
            }
        }
    }
}

impl std::error::Error for MixMinusError {}

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

/// Configuration for a [`MixMinusMatrix`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MixMinusConfig {
    /// Number of input channels.
    pub num_inputs: usize,
    /// Number of output channels.
    pub num_outputs: usize,
}

// ---------------------------------------------------------------------------
// MixMinusMatrix
// ---------------------------------------------------------------------------

/// A mix-minus routing matrix for broadcast IFB feeds.
///
/// Each output produces `sum(all_inputs) - excluded_inputs`.  An input is
/// excluded from a specific output by calling [`Self::route`] with
/// `exclude = true`.
#[derive(Debug, Clone)]
pub struct MixMinusMatrix {
    /// Matrix configuration (dimensions).
    pub config: MixMinusConfig,
    /// Sparse exclusion map: key = `(output_idx, input_idx)`, value = `true` if excluded.
    exclusions: HashMap<(usize, usize), bool>,
}

impl MixMinusMatrix {
    /// Creates a new matrix with no exclusions.
    pub fn new(config: MixMinusConfig) -> Self {
        Self {
            config,
            exclusions: HashMap::new(),
        }
    }

    /// Sets or clears the exclusion of `input_idx` from `output_idx`.
    ///
    /// When `exclude` is `true`, that input is *subtracted* from the output
    /// mix (i.e. the output does not include that input).  When `false`, the
    /// exclusion is removed.
    ///
    /// Returns [`MixMinusError::InputIndexOutOfBounds`] or
    /// [`MixMinusError::OutputIndexOutOfBounds`] if either index is out of
    /// range.
    pub fn route(
        &mut self,
        input_idx: usize,
        output_idx: usize,
        exclude: bool,
    ) -> Result<(), MixMinusError> {
        if input_idx >= self.config.num_inputs {
            return Err(MixMinusError::InputIndexOutOfBounds {
                index: input_idx,
                max: self.config.num_inputs.saturating_sub(1),
            });
        }
        if output_idx >= self.config.num_outputs {
            return Err(MixMinusError::OutputIndexOutOfBounds {
                index: output_idx,
                max: self.config.num_outputs.saturating_sub(1),
            });
        }
        if exclude {
            self.exclusions.insert((output_idx, input_idx), true);
        } else {
            self.exclusions.remove(&(output_idx, input_idx));
        }
        Ok(())
    }

    /// Returns `true` if `input_idx` is excluded from `output_idx`.
    pub fn is_excluded(&self, output_idx: usize, input_idx: usize) -> bool {
        self.exclusions
            .get(&(output_idx, input_idx))
            .copied()
            .unwrap_or(false)
    }

    /// Returns the number of currently active exclusions.
    pub fn active_exclusions(&self) -> usize {
        self.exclusions.values().filter(|&&v| v).count()
    }

    /// Removes all exclusions, resetting the matrix to a full sum.
    pub fn clear_exclusions(&mut self) {
        self.exclusions.clear();
    }

    /// Computes the mix-minus output for all outputs.
    ///
    /// `input_samples` must have exactly `num_inputs` slices, all of the same
    /// length (the frame size).
    ///
    /// Returns a `Vec<Vec<f32>>` of length `num_outputs`; each inner `Vec` has
    /// the same length as the input slices.
    pub fn mix(&self, input_samples: &[&[f32]]) -> Result<Vec<Vec<f32>>, MixMinusError> {
        // Validate input count.
        if input_samples.len() != self.config.num_inputs {
            return Err(MixMinusError::InputCountMismatch {
                expected: self.config.num_inputs,
                actual: input_samples.len(),
            });
        }

        // Determine frame length from first slice (or 0 if no inputs).
        let frame_len = if self.config.num_inputs == 0 {
            0
        } else {
            input_samples[0].len()
        };

        // Validate all slices have the same length.
        for (i, slice) in input_samples.iter().enumerate().skip(1) {
            if slice.len() != frame_len {
                return Err(MixMinusError::SampleLengthMismatch {
                    slice_index: i,
                    expected: frame_len,
                    actual: slice.len(),
                });
            }
        }

        // Pre-compute the global sum across all inputs for each sample.
        let mut global_sum = vec![0.0_f32; frame_len];
        for slice in input_samples {
            for (s, sample) in global_sum.iter_mut().zip(slice.iter()) {
                *s += sample;
            }
        }

        // For each output, subtract excluded inputs from the global sum.
        let mut outputs = Vec::with_capacity(self.config.num_outputs);
        for out_idx in 0..self.config.num_outputs {
            let mut out_buf = global_sum.clone();
            for in_idx in 0..self.config.num_inputs {
                if self.is_excluded(out_idx, in_idx) {
                    for (s, sample) in out_buf.iter_mut().zip(input_samples[in_idx].iter()) {
                        *s -= sample;
                    }
                }
            }
            outputs.push(out_buf);
        }

        Ok(outputs)
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn make_3x3() -> MixMinusMatrix {
        let config = MixMinusConfig {
            num_inputs: 3,
            num_outputs: 3,
        };
        MixMinusMatrix::new(config)
    }

    // -----------------------------------------------------------------------
    // N-1 (each output excludes its own input)
    // -----------------------------------------------------------------------

    #[test]
    fn test_n_minus_1_three_inputs() {
        let mut m = make_3x3();
        m.route(0, 0, true).expect("valid");
        m.route(1, 1, true).expect("valid");
        m.route(2, 2, true).expect("valid");

        let a = vec![1.0_f32; 8];
        let b = vec![1.0_f32; 8];
        let c = vec![1.0_f32; 8];
        let mixed = m.mix(&[&a, &b, &c]).expect("valid mix");

        // Each output should have 3 inputs summed minus itself = 2
        for out in &mixed {
            for &s in out {
                assert!((s - 2.0).abs() < 1e-6, "expected 2.0, got {s}");
            }
        }
    }

    #[test]
    fn test_n_minus_1_asymmetric_values() {
        let mut m = make_3x3();
        m.route(0, 0, true).expect("valid");
        m.route(1, 1, true).expect("valid");
        m.route(2, 2, true).expect("valid");

        let a = vec![1.0_f32; 4];
        let b = vec![2.0_f32; 4];
        let c = vec![3.0_f32; 4];
        let mixed = m.mix(&[&a, &b, &c]).expect("valid mix");

        // out0 = b + c = 5
        assert!((mixed[0][0] - 5.0).abs() < 1e-6);
        // out1 = a + c = 4
        assert!((mixed[1][0] - 4.0).abs() < 1e-6);
        // out2 = a + b = 3
        assert!((mixed[2][0] - 3.0).abs() < 1e-6);
    }

    // -----------------------------------------------------------------------
    // Single exclusion
    // -----------------------------------------------------------------------

    #[test]
    fn test_single_exclusion_only_one_output_affected() {
        let mut m = make_3x3();
        // Only output 0 excludes input 0
        m.route(0, 0, true).expect("valid");

        let a = vec![1.0_f32; 4];
        let b = vec![1.0_f32; 4];
        let c = vec![1.0_f32; 4];
        let mixed = m.mix(&[&a, &b, &c]).expect("valid mix");

        // out0 = b + c = 2
        assert!((mixed[0][0] - 2.0).abs() < 1e-6);
        // out1 = full sum = 3
        assert!((mixed[1][0] - 3.0).abs() < 1e-6);
        // out2 = full sum = 3
        assert!((mixed[2][0] - 3.0).abs() < 1e-6);
    }

    // -----------------------------------------------------------------------
    // No exclusions
    // -----------------------------------------------------------------------

    #[test]
    fn test_no_exclusions_full_sum() {
        let m = make_3x3();
        let a = vec![1.0_f32; 4];
        let b = vec![2.0_f32; 4];
        let c = vec![3.0_f32; 4];
        let mixed = m.mix(&[&a, &b, &c]).expect("valid mix");

        for out in &mixed {
            for &s in out {
                assert!((s - 6.0).abs() < 1e-6, "expected 6.0, got {s}");
            }
        }
    }

    // -----------------------------------------------------------------------
    // Silence passthrough
    // -----------------------------------------------------------------------

    #[test]
    fn test_silence_passthrough() {
        let m = make_3x3();
        let silence = vec![0.0_f32; 16];
        let mixed = m.mix(&[&silence, &silence, &silence]).expect("valid mix");

        for out in &mixed {
            for &s in out {
                assert!(s.abs() < 1e-9, "expected silence, got {s}");
            }
        }
    }

    // -----------------------------------------------------------------------
    // Single-input edge cases
    // -----------------------------------------------------------------------

    #[test]
    fn test_single_input_excluded_produces_silence() {
        let config = MixMinusConfig {
            num_inputs: 1,
            num_outputs: 1,
        };
        let mut m = MixMinusMatrix::new(config);
        m.route(0, 0, true).expect("valid");

        let input = vec![0.5_f32; 4];
        let mixed = m.mix(&[&input]).expect("valid mix");
        for &s in &mixed[0] {
            assert!(s.abs() < 1e-6, "expected silence, got {s}");
        }
    }

    #[test]
    fn test_single_input_not_excluded_passes_through() {
        let config = MixMinusConfig {
            num_inputs: 1,
            num_outputs: 1,
        };
        let m = MixMinusMatrix::new(config);
        let input = vec![0.75_f32; 4];
        let mixed = m.mix(&[&input]).expect("valid mix");
        for &s in &mixed[0] {
            assert!((s - 0.75).abs() < 1e-6, "expected 0.75, got {s}");
        }
    }

    // -----------------------------------------------------------------------
    // Error cases
    // -----------------------------------------------------------------------

    #[test]
    fn test_route_input_out_of_bounds() {
        let mut m = make_3x3();
        let err = m.route(5, 0, true);
        assert!(matches!(
            err,
            Err(MixMinusError::InputIndexOutOfBounds { index: 5, .. })
        ));
    }

    #[test]
    fn test_route_output_out_of_bounds() {
        let mut m = make_3x3();
        let err = m.route(0, 5, true);
        assert!(matches!(
            err,
            Err(MixMinusError::OutputIndexOutOfBounds { index: 5, .. })
        ));
    }

    #[test]
    fn test_mix_input_count_mismatch() {
        let m = make_3x3();
        let a = vec![0.0_f32; 4];
        // Only 2 slices instead of 3
        let err = m.mix(&[&a, &a]);
        assert!(matches!(
            err,
            Err(MixMinusError::InputCountMismatch {
                expected: 3,
                actual: 2
            })
        ));
    }

    #[test]
    fn test_mix_sample_length_mismatch() {
        let m = make_3x3();
        let a = vec![0.0_f32; 4];
        let b = vec![0.0_f32; 8]; // different length
        let c = vec![0.0_f32; 4];
        let err = m.mix(&[&a, &b, &c]);
        assert!(matches!(
            err,
            Err(MixMinusError::SampleLengthMismatch {
                slice_index: 1,
                expected: 4,
                actual: 8
            })
        ));
    }

    // -----------------------------------------------------------------------
    // active_exclusions
    // -----------------------------------------------------------------------

    #[test]
    fn test_active_exclusions_count() {
        let mut m = make_3x3();
        assert_eq!(m.active_exclusions(), 0);

        m.route(0, 0, true).expect("valid");
        assert_eq!(m.active_exclusions(), 1);

        m.route(1, 1, true).expect("valid");
        assert_eq!(m.active_exclusions(), 2);

        m.route(2, 2, true).expect("valid");
        assert_eq!(m.active_exclusions(), 3);
    }

    // -----------------------------------------------------------------------
    // clear_exclusions
    // -----------------------------------------------------------------------

    #[test]
    fn test_clear_exclusions() {
        let mut m = make_3x3();
        m.route(0, 0, true).expect("valid");
        m.route(1, 1, true).expect("valid");
        assert_eq!(m.active_exclusions(), 2);

        m.clear_exclusions();
        assert_eq!(m.active_exclusions(), 0);

        // After clear, full sum should be produced.
        let a = vec![1.0_f32; 4];
        let mixed = m.mix(&[&a, &a, &a]).expect("valid mix");
        for out in &mixed {
            for &s in out {
                assert!((s - 3.0).abs() < 1e-6);
            }
        }
    }

    // -----------------------------------------------------------------------
    // is_excluded
    // -----------------------------------------------------------------------

    #[test]
    fn test_is_excluded_true_and_false() {
        let mut m = make_3x3();
        assert!(!m.is_excluded(0, 0));
        m.route(0, 0, true).expect("valid");
        assert!(m.is_excluded(0, 0));
        m.route(0, 0, false).expect("valid");
        assert!(!m.is_excluded(0, 0));
    }

    // -----------------------------------------------------------------------
    // Multiple outputs excluding same input
    // -----------------------------------------------------------------------

    #[test]
    fn test_multiple_outputs_exclude_same_input() {
        let mut m = make_3x3();
        // All outputs exclude input 0
        m.route(0, 0, true).expect("valid");
        m.route(0, 1, true).expect("valid");
        m.route(0, 2, true).expect("valid");

        let a = vec![10.0_f32; 4];
        let b = vec![1.0_f32; 4];
        let c = vec![1.0_f32; 4];
        let mixed = m.mix(&[&a, &b, &c]).expect("valid mix");

        // Each output excludes input 0 (value 10) → b + c = 2
        for out in &mixed {
            for &s in out {
                assert!((s - 2.0).abs() < 1e-6, "expected 2.0, got {s}");
            }
        }
    }

    // -----------------------------------------------------------------------
    // Gain accumulation correctness
    // -----------------------------------------------------------------------

    #[test]
    fn test_gain_accumulation_known_values() {
        let config = MixMinusConfig {
            num_inputs: 4,
            num_outputs: 2,
        };
        let mut m = MixMinusMatrix::new(config);
        // out0 excludes in1 and in3
        m.route(1, 0, true).expect("valid");
        m.route(3, 0, true).expect("valid");
        // out1 excludes in0
        m.route(0, 1, true).expect("valid");

        let samples: Vec<Vec<f32>> = (0..4).map(|i| vec![(i + 1) as f32; 2]).collect();
        // in0=1, in1=2, in2=3, in3=4  global sum = 10
        let refs: Vec<&[f32]> = samples.iter().map(|v| v.as_slice()).collect();
        let mixed = m.mix(&refs).expect("valid mix");

        // out0 = 10 - in1(2) - in3(4) = 4
        assert!((mixed[0][0] - 4.0).abs() < 1e-6);
        // out1 = 10 - in0(1) = 9
        assert!((mixed[1][0] - 9.0).abs() < 1e-6);
    }

    // -----------------------------------------------------------------------
    // Error display
    // -----------------------------------------------------------------------

    #[test]
    fn test_error_display() {
        let e = MixMinusError::InputIndexOutOfBounds { index: 5, max: 2 };
        let s = format!("{e}");
        assert!(s.contains("5"));

        let e2 = MixMinusError::InputCountMismatch {
            expected: 3,
            actual: 1,
        };
        let s2 = format!("{e2}");
        assert!(s2.contains("3") && s2.contains("1"));
    }
}
