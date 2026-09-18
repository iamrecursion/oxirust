//! Stereo panning laws and pan processors.
#![allow(dead_code)]

/// Supported panning laws for stereo positioning.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum PanLaw {
    /// -3 dB centre: sum of squares of gains equals 1.
    Minus3dB,
    /// -6 dB centre: gains sum linearly to 1.
    Minus6dB,
    /// Linear (same as -6 dB, explicit alias).
    Linear,
    /// Constant power (same as -3 dB, equal-power).
    ConstantPower,
}

impl PanLaw {
    /// Apply the pan law to a position and return `(left_gain, right_gain)`.
    ///
    /// `pan_pos` is in \[-1.0, 1.0\]: -1 = full left, 0 = centre, +1 = full right.
    pub fn apply(&self, pan_pos: f32) -> (f32, f32) {
        let p = pan_pos.clamp(-1.0, 1.0);
        // Map to [0, 1]: 0 = full left, 0.5 = centre, 1 = full right.
        let t = (p + 1.0) * 0.5;
        match self {
            Self::Minus3dB | Self::ConstantPower => {
                // Equal-power: use quarter-wave sine/cosine.
                use std::f32::consts::FRAC_PI_2;
                let angle = t * FRAC_PI_2;
                // Clamp to [0,1] to guard against floating-point rounding
                // near the endpoints (e.g. cos(π/2) ≈ -4.37e-8).
                (angle.cos().clamp(0.0, 1.0), angle.sin().clamp(0.0, 1.0))
            }
            Self::Minus6dB | Self::Linear => {
                // Linear crossfade.
                (1.0 - t, t)
            }
        }
    }

    /// Returns a human-readable name for this law.
    pub fn name(&self) -> &'static str {
        match self {
            Self::Minus3dB => "-3 dB",
            Self::Minus6dB => "-6 dB",
            Self::Linear => "Linear",
            Self::ConstantPower => "Constant Power",
        }
    }
}

/// Represents a pan position with pre-computed gains.
#[derive(Debug, Clone, Copy)]
pub struct PanPosition {
    /// Raw position in \[-1.0, 1.0\].
    pub position: f32,
    left: f32,
    right: f32,
}

impl PanPosition {
    /// Compute gains for `position` using the given `law`.
    pub fn new(position: f32, law: PanLaw) -> Self {
        let (left, right) = law.apply(position);
        Self {
            position,
            left,
            right,
        }
    }

    /// Pre-computed left channel gain.
    pub fn left_gain(&self) -> f32 {
        self.left
    }

    /// Pre-computed right channel gain.
    pub fn right_gain(&self) -> f32 {
        self.right
    }

    /// Returns `true` when the signal is panned hard left.
    pub fn is_hard_left(&self) -> bool {
        self.position <= -1.0
    }

    /// Returns `true` when the signal is panned hard right.
    pub fn is_hard_right(&self) -> bool {
        self.position >= 1.0
    }

    /// Returns `true` when the signal is centred (within ±0.01).
    pub fn is_centre(&self) -> bool {
        self.position.abs() < 0.01
    }
}

/// Stateful stereo panner applying a configurable pan law.
pub struct PanProcessor {
    law: PanLaw,
    position: PanPosition,
}

impl PanProcessor {
    /// Create a new processor centred (pan = 0.0).
    pub fn new(law: PanLaw) -> Self {
        let position = PanPosition::new(0.0, law);
        Self { law, position }
    }

    /// Update the pan position.
    pub fn set_position(&mut self, pan: f32) {
        self.position = PanPosition::new(pan, self.law);
    }

    /// Return a copy of the current position.
    pub fn position(&self) -> PanPosition {
        self.position
    }

    /// Process a mono input buffer into interleaved stereo output.
    ///
    /// `input` is mono samples; `output` receives L,R pairs.
    pub fn process_stereo(&self, input: &[f32], output: &mut Vec<f32>) {
        output.clear();
        let lg = self.position.left_gain();
        let rg = self.position.right_gain();
        for &s in input {
            output.push(s * lg);
            output.push(s * rg);
        }
    }

    /// Returns the current law.
    pub fn law(&self) -> PanLaw {
        self.law
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // --- PanLaw::apply ---

    #[test]
    fn test_constant_power_centre_equal_gains() {
        let (l, r) = PanLaw::ConstantPower.apply(0.0);
        // At centre, gains should be equal (cos(π/4) = sin(π/4)).
        assert!((l - r).abs() < 1e-6);
    }

    #[test]
    fn test_constant_power_sum_of_squares_equals_one() {
        for pos in [-0.5f32, 0.0, 0.5] {
            let (l, r) = PanLaw::ConstantPower.apply(pos);
            let sum_sq = l * l + r * r;
            assert!((sum_sq - 1.0).abs() < 1e-5, "sum_sq={sum_sq} for pos={pos}");
        }
    }

    #[test]
    fn test_linear_centre_equal_gains() {
        let (l, r) = PanLaw::Linear.apply(0.0);
        assert!((l - 0.5).abs() < 1e-6);
        assert!((r - 0.5).abs() < 1e-6);
    }

    #[test]
    fn test_linear_hard_left() {
        let (l, r) = PanLaw::Linear.apply(-1.0);
        assert!((l - 1.0).abs() < 1e-6);
        assert!(r.abs() < 1e-6);
    }

    #[test]
    fn test_linear_hard_right() {
        let (l, r) = PanLaw::Linear.apply(1.0);
        assert!(l.abs() < 1e-6);
        assert!((r - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_minus6db_same_as_linear() {
        let (l6, r6) = PanLaw::Minus6dB.apply(0.3);
        let (ll, rl) = PanLaw::Linear.apply(0.3);
        assert!((l6 - ll).abs() < 1e-6);
        assert!((r6 - rl).abs() < 1e-6);
    }

    #[test]
    fn test_pan_law_clamps_out_of_range() {
        let (l, r) = PanLaw::ConstantPower.apply(5.0);
        assert!(l >= 0.0 && r >= 0.0);
    }

    #[test]
    fn test_pan_law_name_not_empty() {
        for law in [
            PanLaw::Minus3dB,
            PanLaw::Minus6dB,
            PanLaw::Linear,
            PanLaw::ConstantPower,
        ] {
            assert!(!law.name().is_empty());
        }
    }

    // --- PanPosition ---

    #[test]
    fn test_pan_position_left_gain_hard_left() {
        let p = PanPosition::new(-1.0, PanLaw::Linear);
        assert!((p.left_gain() - 1.0).abs() < 1e-6);
        assert!(p.is_hard_left());
    }

    #[test]
    fn test_pan_position_right_gain_hard_right() {
        let p = PanPosition::new(1.0, PanLaw::Linear);
        assert!((p.right_gain() - 1.0).abs() < 1e-6);
        assert!(p.is_hard_right());
    }

    #[test]
    fn test_pan_position_is_centre() {
        let p = PanPosition::new(0.005, PanLaw::ConstantPower);
        assert!(p.is_centre());
    }

    #[test]
    fn test_pan_position_not_centre() {
        let p = PanPosition::new(0.5, PanLaw::ConstantPower);
        assert!(!p.is_centre());
    }

    // --- PanProcessor ---

    #[test]
    fn test_processor_process_stereo_length() {
        let proc = PanProcessor::new(PanLaw::ConstantPower);
        let input = vec![0.5f32; 4];
        let mut out = Vec::new();
        proc.process_stereo(&input, &mut out);
        assert_eq!(out.len(), 8); // 4 mono → 4 L/R pairs
    }

    #[test]
    fn test_processor_hard_left_mutes_right() {
        let mut proc = PanProcessor::new(PanLaw::Linear);
        proc.set_position(-1.0);
        let input = vec![1.0f32; 2];
        let mut out = Vec::new();
        proc.process_stereo(&input, &mut out);
        // out[0]=L, out[1]=R for first sample
        assert!(out[1].abs() < 1e-6);
    }

    #[test]
    fn test_processor_law_accessor() {
        let proc = PanProcessor::new(PanLaw::Minus3dB);
        assert_eq!(proc.law(), PanLaw::Minus3dB);
    }
}
