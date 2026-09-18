//! Audio crossfade utilities.
//!
//! This module provides configurable crossfade curves for smoothly
//! transitioning between two audio sources. Several curve shapes are
//! supported, and the [`CrossfadeProcessor`] can apply a crossfade to
//! a pair of buffers in a single pass.
//!
//! # Example
//!
//! ```
//! use oximedia_audio::crossfade::{CrossfadeShape, CrossfadeProcessor};
//!
//! let proc = CrossfadeProcessor::new(CrossfadeShape::EqualPower, 256);
//! let a = vec![1.0_f32; 256];
//! let b = vec![0.0_f32; 256];
//! let mixed = proc.apply(&a, &b);
//! assert_eq!(mixed.len(), 256);
//! ```

#![allow(dead_code)]

use std::f32::consts::PI;

// ---------------------------------------------------------------------------
// Curve shapes
// ---------------------------------------------------------------------------

/// Shape of the crossfade curve.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum CrossfadeShape {
    /// Linear ramp (simplest but can dip in the middle).
    Linear,
    /// Equal-power (constant-power) crossfade -- no loudness dip.
    EqualPower,
    /// Sine-curve crossfade.
    Sine,
    /// Cosine S-curve (slow start/end, fast middle).
    SCurve,
    /// Square-root curve.
    SquareRoot,
    /// Logarithmic fade (fast start, slow tail).
    Logarithmic,
}

impl CrossfadeShape {
    /// Compute the fade-in gain at normalised position `t` in `[0, 1]`.
    ///
    /// `t == 0` means fully source A, `t == 1` means fully source B.
    pub fn fade_in(&self, t: f32) -> f32 {
        let t = t.clamp(0.0, 1.0);
        match self {
            Self::Linear => t,
            Self::EqualPower => (t * PI * 0.5).sin(),
            Self::Sine => (t * PI * 0.5).sin(),
            Self::SCurve => {
                let s = t * PI;
                (1.0 - s.cos()) * 0.5
            }
            Self::SquareRoot => t.sqrt(),
            Self::Logarithmic => {
                if t <= 0.0 {
                    0.0
                } else {
                    #[allow(clippy::cast_precision_loss)]
                    let v = (1.0 + t * 99.0).log10() / 2.0_f32.log10();
                    v.clamp(0.0, 1.0)
                }
            }
        }
    }

    /// Compute the fade-out gain at normalised position `t` in `[0, 1]`.
    pub fn fade_out(&self, t: f32) -> f32 {
        let t = t.clamp(0.0, 1.0);
        match self {
            Self::Linear => 1.0 - t,
            Self::EqualPower => ((1.0 - t) * PI * 0.5).sin(),
            Self::Sine => ((1.0 - t) * PI * 0.5).sin(),
            Self::SCurve => {
                let s = t * PI;
                (1.0 + s.cos()) * 0.5
            }
            Self::SquareRoot => (1.0 - t).sqrt(),
            Self::Logarithmic => {
                let inv = 1.0 - t;
                if inv <= 0.0 {
                    0.0
                } else {
                    #[allow(clippy::cast_precision_loss)]
                    let v = (1.0 + inv * 99.0).log10() / 2.0_f32.log10();
                    v.clamp(0.0, 1.0)
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Processor
// ---------------------------------------------------------------------------

/// Applies a crossfade between two audio buffers.
#[derive(Debug, Clone)]
pub struct CrossfadeProcessor {
    /// The curve shape.
    pub shape: CrossfadeShape,
    /// Crossfade length in samples.
    pub length: usize,
}

impl CrossfadeProcessor {
    /// Create a new crossfade processor.
    pub fn new(shape: CrossfadeShape, length: usize) -> Self {
        Self {
            shape,
            length: length.max(1),
        }
    }

    /// Apply the crossfade, blending `source_a` (fading out) with
    /// `source_b` (fading in). Returns a new buffer of `self.length` samples.
    ///
    /// Both sources must be at least `self.length` samples long.
    pub fn apply(&self, source_a: &[f32], source_b: &[f32]) -> Vec<f32> {
        let len = self.length.min(source_a.len()).min(source_b.len());
        let mut output = Vec::with_capacity(len);
        for i in 0..len {
            #[allow(clippy::cast_precision_loss)]
            let t = if len <= 1 {
                1.0
            } else {
                i as f32 / (len - 1) as f32
            };
            let a_gain = self.shape.fade_out(t);
            let b_gain = self.shape.fade_in(t);
            output.push(source_a[i] * a_gain + source_b[i] * b_gain);
        }
        output
    }

    /// Apply the crossfade in-place, writing the result into `dest`.
    pub fn apply_into(&self, source_a: &[f32], source_b: &[f32], dest: &mut [f32]) {
        let len = self
            .length
            .min(source_a.len())
            .min(source_b.len())
            .min(dest.len());
        for i in 0..len {
            #[allow(clippy::cast_precision_loss)]
            let t = if len <= 1 {
                1.0
            } else {
                i as f32 / (len - 1) as f32
            };
            dest[i] = source_a[i] * self.shape.fade_out(t) + source_b[i] * self.shape.fade_in(t);
        }
    }

    /// Return the crossfade length in samples.
    pub fn length_samples(&self) -> usize {
        self.length
    }

    /// Compute the gain pair `(fade_out, fade_in)` at a given position.
    pub fn gains_at(&self, position: usize) -> (f32, f32) {
        #[allow(clippy::cast_precision_loss)]
        let t = if self.length <= 1 {
            1.0
        } else {
            position as f32 / (self.length - 1) as f32
        };
        (self.shape.fade_out(t), self.shape.fade_in(t))
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_linear_fade_endpoints() {
        let shape = CrossfadeShape::Linear;
        assert!((shape.fade_in(0.0)).abs() < 1e-6);
        assert!((shape.fade_in(1.0) - 1.0).abs() < 1e-6);
        assert!((shape.fade_out(0.0) - 1.0).abs() < 1e-6);
        assert!((shape.fade_out(1.0)).abs() < 1e-6);
    }

    #[test]
    fn test_equal_power_endpoints() {
        let shape = CrossfadeShape::EqualPower;
        assert!((shape.fade_in(0.0)).abs() < 1e-6);
        assert!((shape.fade_in(1.0) - 1.0).abs() < 1e-6);
        assert!((shape.fade_out(0.0) - 1.0).abs() < 1e-6);
        assert!((shape.fade_out(1.0)).abs() < 1e-6);
    }

    #[test]
    fn test_equal_power_midpoint_no_dip() {
        let shape = CrossfadeShape::EqualPower;
        let a = shape.fade_out(0.5);
        let b = shape.fade_in(0.5);
        // Equal-power: a^2 + b^2 should be ~1
        let power = a * a + b * b;
        assert!((power - 1.0).abs() < 1e-4);
    }

    #[test]
    fn test_sine_endpoints() {
        let shape = CrossfadeShape::Sine;
        assert!(shape.fade_in(0.0) < 1e-6);
        assert!((shape.fade_in(1.0) - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_s_curve_endpoints() {
        let shape = CrossfadeShape::SCurve;
        assert!(shape.fade_in(0.0).abs() < 1e-6);
        assert!((shape.fade_in(1.0) - 1.0).abs() < 1e-6);
        assert!((shape.fade_out(0.0) - 1.0).abs() < 1e-6);
        assert!(shape.fade_out(1.0).abs() < 1e-6);
    }

    #[test]
    fn test_square_root_monotonic() {
        let shape = CrossfadeShape::SquareRoot;
        let mut prev = 0.0;
        for i in 0..=10 {
            #[allow(clippy::cast_precision_loss)]
            let t = i as f32 / 10.0;
            let v = shape.fade_in(t);
            assert!(v >= prev - 1e-6);
            prev = v;
        }
    }

    #[test]
    fn test_logarithmic_endpoints() {
        let shape = CrossfadeShape::Logarithmic;
        assert!(shape.fade_in(0.0).abs() < 1e-6);
        assert!((shape.fade_in(1.0) - 1.0).abs() < 0.01);
    }

    #[test]
    fn test_processor_apply() {
        let proc = CrossfadeProcessor::new(CrossfadeShape::Linear, 5);
        let a = vec![1.0; 5];
        let b = vec![0.0; 5];
        let mixed = proc.apply(&a, &b);
        assert_eq!(mixed.len(), 5);
        // first sample: fully A
        assert!((mixed[0] - 1.0).abs() < 1e-6);
        // last sample: fully B
        assert!(mixed[4].abs() < 1e-6);
    }

    #[test]
    fn test_processor_apply_into() {
        let proc = CrossfadeProcessor::new(CrossfadeShape::EqualPower, 4);
        let a = vec![1.0; 4];
        let b = vec![1.0; 4];
        let mut dest = vec![0.0; 4];
        proc.apply_into(&a, &b, &mut dest);
        // all values should be >= 1.0 because of equal-power curve
        for &v in &dest {
            assert!(v >= 0.99);
        }
    }

    #[test]
    fn test_processor_length() {
        let proc = CrossfadeProcessor::new(CrossfadeShape::Linear, 128);
        assert_eq!(proc.length_samples(), 128);
    }

    #[test]
    fn test_gains_at_endpoints() {
        let proc = CrossfadeProcessor::new(CrossfadeShape::Linear, 100);
        let (fo, fi) = proc.gains_at(0);
        assert!((fo - 1.0).abs() < 1e-6);
        assert!(fi.abs() < 1e-6);
        let (fo, fi) = proc.gains_at(99);
        assert!(fo.abs() < 1e-6);
        assert!((fi - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_clamp_out_of_range() {
        let shape = CrossfadeShape::Linear;
        assert!(shape.fade_in(-0.5).abs() < 1e-6);
        assert!((shape.fade_in(2.0) - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_minimum_length_is_one() {
        let proc = CrossfadeProcessor::new(CrossfadeShape::Linear, 0);
        assert_eq!(proc.length_samples(), 1);
    }

    #[test]
    fn test_all_shapes_compile_and_run() {
        let shapes = [
            CrossfadeShape::Linear,
            CrossfadeShape::EqualPower,
            CrossfadeShape::Sine,
            CrossfadeShape::SCurve,
            CrossfadeShape::SquareRoot,
            CrossfadeShape::Logarithmic,
        ];
        for shape in &shapes {
            let proc = CrossfadeProcessor::new(*shape, 16);
            let a = vec![1.0; 16];
            let b = vec![0.5; 16];
            let mixed = proc.apply(&a, &b);
            assert_eq!(mixed.len(), 16);
        }
    }
}
