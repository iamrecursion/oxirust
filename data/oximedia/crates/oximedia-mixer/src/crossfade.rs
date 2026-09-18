//! Audio/video crossfade transitions for the `OxiMedia` mixer.
//!
//! Provides crossfade shapes, pairs, and generator utilities for creating
//! smooth transitions between two audio or video sources.

#![allow(dead_code)]

/// Shape of the crossfade gain curve applied to the fading sources.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CrossfadeShape {
    /// Gain changes linearly with position.
    Linear,
    /// Equal-power (constant-power) crossfade — energy is preserved.
    EqualPower,
    /// S-curve (smooth-step) crossfade — slow start and end.
    SCurve,
    /// Logarithmic crossfade — fast initial fade.
    Logarithmic,
}

impl CrossfadeShape {
    /// Compute the fade-*in* gain at normalised position `t` (0.0 = start, 1.0 = end).
    ///
    /// The fade-*out* gain is always `1.0 - gain_at(t)` for `Linear`, or the
    /// complementary curve for power-preserving shapes.
    #[allow(clippy::cast_precision_loss)]
    #[must_use]
    pub fn gain_at(self, t: f32) -> f32 {
        let t = t.clamp(0.0, 1.0);
        match self {
            Self::Linear => t,
            Self::EqualPower => (t * std::f32::consts::FRAC_PI_2).sin(),
            Self::SCurve => t * t * (3.0 - 2.0 * t),
            Self::Logarithmic => {
                // Map [0,1] → [-60 dB, 0 dB] and convert to linear.
                if t == 0.0 {
                    0.0
                } else {
                    10_f32.powf(-6.0 * (1.0 - t))
                }
            }
        }
    }
}

/// A pair of gain values for one frame of a crossfade.
///
/// `fade_out_gain` is applied to the outgoing source;
/// `fade_in_gain` is applied to the incoming source.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CrossfadePair {
    /// Gain for the outgoing (fading-out) signal.
    pub fade_out_gain: f32,
    /// Gain for the incoming (fading-in) signal.
    pub fade_in_gain: f32,
}

impl CrossfadePair {
    /// Create a new crossfade pair.
    #[must_use]
    pub fn new(fade_out_gain: f32, fade_in_gain: f32) -> Self {
        Self {
            fade_out_gain,
            fade_in_gain,
        }
    }

    /// Mix two samples using the stored gains.
    ///
    /// Returns `sample_a * fade_out_gain + sample_b * fade_in_gain`.
    #[must_use]
    pub fn mix(self, sample_a: f32, sample_b: f32) -> f32 {
        sample_a * self.fade_out_gain + sample_b * self.fade_in_gain
    }
}

/// Generates a sequence of `CrossfadePair` values over a fixed number of frames.
#[derive(Debug, Clone)]
pub struct CrossfadeGenerator {
    /// The gain curve to use.
    pub shape: CrossfadeShape,
    /// Total number of frames in the crossfade.
    pub duration_frames: u32,
}

impl CrossfadeGenerator {
    /// Create a new crossfade generator.
    #[must_use]
    pub fn new(shape: CrossfadeShape, duration_frames: u32) -> Self {
        Self {
            shape,
            duration_frames,
        }
    }

    /// Generate the `CrossfadePair` for the given frame index.
    ///
    /// Frame 0 returns full fade-out / zero fade-in;
    /// the last frame returns zero fade-out / full fade-in.
    #[allow(clippy::cast_precision_loss)]
    #[must_use]
    pub fn generate(&self, frame: u32) -> CrossfadePair {
        let t = if self.duration_frames == 0 {
            1.0_f32
        } else {
            (frame as f32 / self.duration_frames as f32).clamp(0.0, 1.0)
        };

        let fade_in = self.shape.gain_at(t);

        // For equal-power, fade_out uses the complementary angle.
        let fade_out = match self.shape {
            CrossfadeShape::EqualPower => ((1.0 - t) * std::f32::consts::FRAC_PI_2).sin(),
            _ => 1.0 - self.shape.gain_at(t),
        };

        CrossfadePair::new(fade_out, fade_in)
    }

    /// Returns `true` once `frame` has reached or exceeded `duration_frames`.
    #[must_use]
    pub fn is_complete(&self, frame: u32) -> bool {
        frame >= self.duration_frames
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // ---- CrossfadeShape::gain_at ----

    #[test]
    fn test_linear_gain_at_zero() {
        assert!((CrossfadeShape::Linear.gain_at(0.0)).abs() < f32::EPSILON);
    }

    #[test]
    fn test_linear_gain_at_one() {
        assert!((CrossfadeShape::Linear.gain_at(1.0) - 1.0).abs() < f32::EPSILON);
    }

    #[test]
    fn test_linear_gain_at_half() {
        assert!((CrossfadeShape::Linear.gain_at(0.5) - 0.5).abs() < 1e-6);
    }

    #[test]
    fn test_equal_power_gain_at_zero() {
        assert!(CrossfadeShape::EqualPower.gain_at(0.0) < 1e-6);
    }

    #[test]
    fn test_equal_power_gain_at_one() {
        assert!((CrossfadeShape::EqualPower.gain_at(1.0) - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_equal_power_midpoint_is_sqrt2_over_2() {
        let mid = CrossfadeShape::EqualPower.gain_at(0.5);
        let expected = std::f32::consts::FRAC_1_SQRT_2;
        assert!(
            (mid - expected).abs() < 1e-5,
            "mid={mid} expected={expected}"
        );
    }

    #[test]
    fn test_scurve_gain_is_monotonic() {
        let prev = CrossfadeShape::SCurve.gain_at(0.3);
        let next = CrossfadeShape::SCurve.gain_at(0.7);
        assert!(next > prev);
    }

    #[test]
    fn test_logarithmic_gain_at_zero_is_zero() {
        assert!(CrossfadeShape::Logarithmic.gain_at(0.0) < 1e-6);
    }

    #[test]
    fn test_logarithmic_gain_at_one_is_one() {
        assert!((CrossfadeShape::Logarithmic.gain_at(1.0) - 1.0).abs() < 1e-5);
    }

    #[test]
    fn test_gain_clamps_above_one() {
        assert!((CrossfadeShape::Linear.gain_at(2.0) - 1.0).abs() < f32::EPSILON);
    }

    // ---- CrossfadePair::mix ----

    #[test]
    fn test_mix_full_a() {
        let pair = CrossfadePair::new(1.0, 0.0);
        assert!((pair.mix(0.8, 0.4) - 0.8).abs() < f32::EPSILON);
    }

    #[test]
    fn test_mix_full_b() {
        let pair = CrossfadePair::new(0.0, 1.0);
        assert!((pair.mix(0.8, 0.4) - 0.4).abs() < f32::EPSILON);
    }

    #[test]
    fn test_mix_equal_blend() {
        let pair = CrossfadePair::new(0.5, 0.5);
        let expected = 0.5 * 1.0 + -0.5;
        assert!((pair.mix(1.0, -1.0) - expected).abs() < f32::EPSILON);
    }

    // ---- CrossfadeGenerator ----

    #[test]
    fn test_generator_first_frame_is_full_fade_out() {
        let gen = CrossfadeGenerator::new(CrossfadeShape::Linear, 100);
        let pair = gen.generate(0);
        assert!((pair.fade_out_gain - 1.0).abs() < f32::EPSILON);
        assert!(pair.fade_in_gain.abs() < f32::EPSILON);
    }

    #[test]
    fn test_generator_last_frame_is_full_fade_in() {
        let gen = CrossfadeGenerator::new(CrossfadeShape::Linear, 100);
        let pair = gen.generate(100);
        assert!((pair.fade_in_gain - 1.0).abs() < f32::EPSILON);
        assert!(pair.fade_out_gain.abs() < f32::EPSILON);
    }

    #[test]
    fn test_generator_is_complete() {
        let gen = CrossfadeGenerator::new(CrossfadeShape::Linear, 50);
        assert!(!gen.is_complete(49));
        assert!(gen.is_complete(50));
        assert!(gen.is_complete(100));
    }

    #[test]
    fn test_generator_zero_duration() {
        let gen = CrossfadeGenerator::new(CrossfadeShape::Linear, 0);
        let pair = gen.generate(0);
        // t=1.0 → fade_in=1, fade_out=0
        assert!((pair.fade_in_gain - 1.0).abs() < f32::EPSILON);
    }

    #[test]
    fn test_equal_power_energy_preserved_at_midpoint() {
        let gen = CrossfadeGenerator::new(CrossfadeShape::EqualPower, 100);
        let pair = gen.generate(50);
        let energy =
            pair.fade_out_gain * pair.fade_out_gain + pair.fade_in_gain * pair.fade_in_gain;
        // For equal-power crossfade, energy should be ~1.0
        assert!((energy - 1.0).abs() < 0.01, "energy={energy}");
    }
}
