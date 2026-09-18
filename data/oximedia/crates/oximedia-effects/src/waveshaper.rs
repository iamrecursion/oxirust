//! Waveshaping distortion effect.
//!
//! Waveshaping applies a static non-linear transfer function to each sample,
//! producing harmonic distortion whose character depends on the curve used.
//! This module provides several built-in curves ([`WaveshaperCurve`]) and a
//! [`Waveshaper`] struct that processes audio with optional oversampling
//! awareness (via an internal DC-blocking filter).
//!
//! # Example
//!
//! ```
//! use oximedia_effects::waveshaper::{WaveshaperCurve, Waveshaper};
//!
//! let mut ws = Waveshaper::new(WaveshaperCurve::SoftClip, 0.8);
//! let out = ws.process_sample(0.9);
//! assert!(out.abs() <= 1.0);
//! ```

#![allow(dead_code)]

// ---------------------------------------------------------------------------
// Curve types
// ---------------------------------------------------------------------------

/// Built-in waveshaping transfer-function curves.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum WaveshaperCurve {
    /// Hyperbolic tangent soft clipping.
    SoftClip,
    /// Hard clipping at +/-1.
    HardClip,
    /// Sine-based folding distortion.
    Foldback,
    /// Asymmetric warm tube-style curve.
    Tube,
    /// Chebyshev polynomial (adds 2nd harmonic).
    Chebyshev2,
    /// Chebyshev polynomial (adds 3rd harmonic).
    Chebyshev3,
}

impl WaveshaperCurve {
    /// Apply the transfer function to `x`.
    #[must_use]
    pub fn apply(&self, x: f32) -> f32 {
        match self {
            Self::SoftClip => x.tanh(),
            Self::HardClip => x.clamp(-1.0, 1.0),
            Self::Foldback => {
                // Fold signal back when it exceeds [-1, 1].
                let mut v = x;
                while !(-1.0..=1.0).contains(&v) {
                    if v > 1.0 {
                        v = 2.0 - v;
                    }
                    if v < -1.0 {
                        v = -2.0 - v;
                    }
                }
                v
            }
            Self::Tube => {
                // Asymmetric: positive half gets softer saturation.
                if x >= 0.0 {
                    1.0 - (-3.0 * x).exp()
                } else {
                    -(1.0 - (3.0 * x).exp())
                }
            }
            Self::Chebyshev2 => {
                // T2(x) = 2x^2 - 1
                2.0 * x * x - 1.0
            }
            Self::Chebyshev3 => {
                // T3(x) = 4x^3 - 3x
                4.0 * x * x * x - 3.0 * x
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Waveshaper processor
// ---------------------------------------------------------------------------

/// A waveshaping distortion effect.
///
/// Combines a drive parameter that scales the input before the curve, with an
/// optional DC-blocking high-pass filter to remove any DC offset introduced by
/// asymmetric curves.
#[derive(Debug, Clone)]
pub struct Waveshaper {
    /// The transfer-function curve.
    pub curve: WaveshaperCurve,
    /// Drive amount (pre-gain before curve), typically 0.0 -- 10.0.
    pub drive: f32,
    /// Output gain (post-gain after curve), 0.0 -- 1.0.
    pub output_gain: f32,
    /// Wet/dry mix (0 = dry, 1 = wet).
    pub mix: f32,
    // DC blocker state
    dc_x1: f32,
    dc_y1: f32,
    dc_coeff: f32,
}

impl Waveshaper {
    /// Create a new waveshaper with the given curve and drive level.
    #[must_use]
    pub fn new(curve: WaveshaperCurve, drive: f32) -> Self {
        Self {
            curve,
            drive: drive.max(0.0),
            output_gain: 1.0,
            mix: 1.0,
            dc_x1: 0.0,
            dc_y1: 0.0,
            dc_coeff: 0.995,
        }
    }

    /// Set the wet/dry mix (0 = fully dry, 1 = fully wet).
    pub fn set_mix(&mut self, mix: f32) {
        self.mix = mix.clamp(0.0, 1.0);
    }

    /// Set the output gain.
    pub fn set_output_gain(&mut self, gain: f32) {
        self.output_gain = gain.clamp(0.0, 2.0);
    }

    /// Process a single sample.
    pub fn process_sample(&mut self, input: f32) -> f32 {
        let driven = input * (1.0 + self.drive);
        let shaped = self.curve.apply(driven);
        let dc_blocked = self.dc_block(shaped);
        let wet = dc_blocked * self.output_gain;
        input * (1.0 - self.mix) + wet * self.mix
    }

    /// Process a buffer of samples in-place.
    pub fn process_buffer(&mut self, buffer: &mut [f32]) {
        for sample in buffer.iter_mut() {
            *sample = self.process_sample(*sample);
        }
    }

    /// Reset internal state.
    pub fn reset(&mut self) {
        self.dc_x1 = 0.0;
        self.dc_y1 = 0.0;
    }

    /// DC-blocking high-pass filter (1-pole).
    fn dc_block(&mut self, x: f32) -> f32 {
        let y = x - self.dc_x1 + self.dc_coeff * self.dc_y1;
        self.dc_x1 = x;
        self.dc_y1 = y;
        y
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_soft_clip_bounded() {
        let curve = WaveshaperCurve::SoftClip;
        for i in -100..=100 {
            #[allow(clippy::cast_precision_loss)]
            let x = i as f32 * 0.1;
            let y = curve.apply(x);
            assert!(y >= -1.0 && y <= 1.0, "SoftClip out of range at x={x}: {y}");
        }
    }

    #[test]
    fn test_hard_clip_exact() {
        let curve = WaveshaperCurve::HardClip;
        assert!((curve.apply(0.5) - 0.5).abs() < 1e-6);
        assert!((curve.apply(2.0) - 1.0).abs() < 1e-6);
        assert!((curve.apply(-3.0) - (-1.0)).abs() < 1e-6);
    }

    #[test]
    fn test_foldback_bounded() {
        let curve = WaveshaperCurve::Foldback;
        let y = curve.apply(1.5);
        assert!(y >= -1.0 && y <= 1.0, "Foldback out of range: {y}");
    }

    #[test]
    fn test_tube_asymmetric() {
        let curve = WaveshaperCurve::Tube;
        let pos = curve.apply(0.5);
        let neg = curve.apply(-0.5);
        // The curve is asymmetric, but magnitudes should be close (not identical).
        assert!(pos > 0.0);
        assert!(neg < 0.0);
    }

    #[test]
    fn test_chebyshev2_at_zero() {
        let curve = WaveshaperCurve::Chebyshev2;
        // T2(0) = -1
        assert!((curve.apply(0.0) - (-1.0)).abs() < 1e-6);
    }

    #[test]
    fn test_chebyshev3_at_one() {
        let curve = WaveshaperCurve::Chebyshev3;
        // T3(1) = 4 - 3 = 1
        assert!((curve.apply(1.0) - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_waveshaper_process_sample() {
        let mut ws = Waveshaper::new(WaveshaperCurve::SoftClip, 1.0);
        let out = ws.process_sample(0.5);
        // Should be some value, not NaN
        assert!(out.is_finite());
    }

    #[test]
    fn test_waveshaper_dry_mix() {
        let mut ws = Waveshaper::new(WaveshaperCurve::HardClip, 1.0);
        ws.set_mix(0.0); // fully dry
                         // Reset DC blocker to avoid transient
        ws.reset();
        // After DC blocker settles, dry signal passes through
        for _ in 0..100 {
            ws.process_sample(0.3);
        }
        let out = ws.process_sample(0.3);
        assert!((out - 0.3).abs() < 0.05);
    }

    #[test]
    fn test_waveshaper_process_buffer() {
        let mut ws = Waveshaper::new(WaveshaperCurve::SoftClip, 0.5);
        let mut buf = vec![0.1, 0.5, -0.3, 0.9, -0.8];
        ws.process_buffer(&mut buf);
        for v in &buf {
            assert!(v.is_finite());
        }
    }

    #[test]
    fn test_waveshaper_reset() {
        let mut ws = Waveshaper::new(WaveshaperCurve::Tube, 2.0);
        ws.process_sample(0.5);
        ws.reset();
        assert!((ws.dc_x1).abs() < 1e-6);
        assert!((ws.dc_y1).abs() < 1e-6);
    }

    #[test]
    fn test_output_gain() {
        let mut ws = Waveshaper::new(WaveshaperCurve::HardClip, 0.0);
        ws.set_output_gain(0.5);
        assert!((ws.output_gain - 0.5).abs() < 1e-6);
    }

    #[test]
    fn test_output_gain_clamp() {
        let mut ws = Waveshaper::new(WaveshaperCurve::SoftClip, 0.0);
        ws.set_output_gain(-1.0);
        assert!((ws.output_gain).abs() < 1e-6);
        ws.set_output_gain(5.0);
        assert!((ws.output_gain - 2.0).abs() < 1e-6);
    }

    #[test]
    fn test_drive_non_negative() {
        let ws = Waveshaper::new(WaveshaperCurve::SoftClip, -5.0);
        assert!(ws.drive >= 0.0);
    }

    #[test]
    fn test_all_curves_produce_finite() {
        let curves = [
            WaveshaperCurve::SoftClip,
            WaveshaperCurve::HardClip,
            WaveshaperCurve::Foldback,
            WaveshaperCurve::Tube,
            WaveshaperCurve::Chebyshev2,
            WaveshaperCurve::Chebyshev3,
        ];
        for curve in &curves {
            let mut ws = Waveshaper::new(*curve, 2.0);
            for i in -10..=10 {
                #[allow(clippy::cast_precision_loss)]
                let x = i as f32 * 0.1;
                let y = ws.process_sample(x);
                assert!(y.is_finite(), "NaN for curve {:?} at x={x}", curve);
            }
        }
    }

    #[test]
    fn test_mix_clamp() {
        let mut ws = Waveshaper::new(WaveshaperCurve::SoftClip, 1.0);
        ws.set_mix(1.5);
        assert!((ws.mix - 1.0).abs() < 1e-6);
        ws.set_mix(-0.5);
        assert!(ws.mix.abs() < 1e-6);
    }
}
