//! Blend mode effects for audio and visual processing.
//!
//! Provides implementations of standard blend modes: multiply, screen,
//! overlay, hard light, and soft light, operating on normalized [0.0, 1.0] samples.

#![allow(dead_code)]
#![allow(clippy::cast_precision_loss)]

/// Apply multiply blend: `a * b`.
///
/// Both inputs in [0.0, 1.0]. Output is darkened or equal to the darker input.
#[must_use]
pub fn multiply(a: f32, b: f32) -> f32 {
    (a * b).clamp(0.0, 1.0)
}

/// Apply screen blend: `1 - (1 - a) * (1 - b)`.
///
/// Output is brightened; always >= either input.
#[must_use]
pub fn screen(a: f32, b: f32) -> f32 {
    let result = 1.0 - (1.0 - a) * (1.0 - b);
    result.clamp(0.0, 1.0)
}

/// Apply overlay blend.
///
/// Combines multiply and screen: dark areas darken, bright areas brighten.
#[must_use]
pub fn overlay(base: f32, blend: f32) -> f32 {
    let result = if base < 0.5 {
        2.0 * base * blend
    } else {
        1.0 - 2.0 * (1.0 - base) * (1.0 - blend)
    };
    result.clamp(0.0, 1.0)
}

/// Apply hard light blend.
///
/// Like overlay but driven by the blend layer instead of the base layer.
#[must_use]
pub fn hard_light(base: f32, blend: f32) -> f32 {
    overlay(blend, base)
}

/// Apply soft light blend.
///
/// Gentler version of overlay; uses a smooth curve to darken/lighten.
#[must_use]
pub fn soft_light(base: f32, blend: f32) -> f32 {
    let result = if blend < 0.5 {
        base - (1.0 - 2.0 * blend) * base * (1.0 - base)
    } else {
        let d = if base < 0.25 {
            ((16.0 * base - 12.0) * base + 4.0) * base
        } else {
            base.sqrt()
        };
        base + (2.0 * blend - 1.0) * (d - base)
    };
    result.clamp(0.0, 1.0)
}

/// Apply difference blend: `|a - b|`.
#[must_use]
pub fn difference(a: f32, b: f32) -> f32 {
    (a - b).abs().clamp(0.0, 1.0)
}

/// Apply exclusion blend: `a + b - 2 * a * b`.
#[must_use]
pub fn exclusion(a: f32, b: f32) -> f32 {
    (a + b - 2.0 * a * b).clamp(0.0, 1.0)
}

/// Apply linear dodge (add) blend: `a + b`.
#[must_use]
pub fn linear_dodge(a: f32, b: f32) -> f32 {
    (a + b).clamp(0.0, 1.0)
}

/// Apply linear burn blend: `a + b - 1`.
#[must_use]
pub fn linear_burn(a: f32, b: f32) -> f32 {
    (a + b - 1.0).clamp(0.0, 1.0)
}

/// Blend two buffers together using a named blend mode at the given mix ratio.
///
/// `mix` is in [0.0, 1.0] where 0.0 = 100% `a`, 1.0 = 100% blended result.
pub fn blend_buffers(a: &[f32], b: &[f32], output: &mut [f32], mode: BlendMode, mix: f32) {
    let mix = mix.clamp(0.0, 1.0);
    let len = a.len().min(b.len()).min(output.len());
    for i in 0..len {
        let blended = mode.apply(a[i], b[i]);
        output[i] = a[i] * (1.0 - mix) + blended * mix;
    }
}

/// Available blend modes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BlendMode {
    /// Multiply blend.
    Multiply,
    /// Screen blend.
    Screen,
    /// Overlay blend.
    Overlay,
    /// Hard light blend.
    HardLight,
    /// Soft light blend.
    SoftLight,
    /// Difference blend.
    Difference,
    /// Exclusion blend.
    Exclusion,
    /// Linear dodge (add) blend.
    LinearDodge,
    /// Linear burn blend.
    LinearBurn,
}

impl BlendMode {
    /// Apply this blend mode to two normalized sample values.
    #[must_use]
    pub fn apply(self, a: f32, b: f32) -> f32 {
        match self {
            Self::Multiply => multiply(a, b),
            Self::Screen => screen(a, b),
            Self::Overlay => overlay(a, b),
            Self::HardLight => hard_light(a, b),
            Self::SoftLight => soft_light(a, b),
            Self::Difference => difference(a, b),
            Self::Exclusion => exclusion(a, b),
            Self::LinearDodge => linear_dodge(a, b),
            Self::LinearBurn => linear_burn(a, b),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_multiply_extremes() {
        assert_eq!(multiply(0.0, 1.0), 0.0);
        assert_eq!(multiply(1.0, 1.0), 1.0);
        assert_eq!(multiply(0.0, 0.0), 0.0);
    }

    #[test]
    fn test_multiply_midpoint() {
        let result = multiply(0.5, 0.5);
        assert!((result - 0.25).abs() < 1e-6);
    }

    #[test]
    fn test_screen_extremes() {
        assert_eq!(screen(0.0, 0.0), 0.0);
        assert_eq!(screen(1.0, 0.0), 1.0);
        assert_eq!(screen(1.0, 1.0), 1.0);
    }

    #[test]
    fn test_screen_midpoint() {
        let result = screen(0.5, 0.5);
        assert!((result - 0.75).abs() < 1e-6);
    }

    #[test]
    fn test_screen_always_brighter_than_inputs() {
        let a = 0.3_f32;
        let b = 0.4_f32;
        let result = screen(a, b);
        assert!(result >= a);
        assert!(result >= b);
    }

    #[test]
    fn test_multiply_always_darker_than_inputs() {
        let a = 0.6_f32;
        let b = 0.7_f32;
        let result = multiply(a, b);
        assert!(result <= a);
        assert!(result <= b);
    }

    #[test]
    fn test_overlay_dark_region() {
        // base < 0.5, so overlay = 2 * base * blend
        let result = overlay(0.25, 0.5);
        assert!((result - 0.25).abs() < 1e-6);
    }

    #[test]
    fn test_overlay_bright_region() {
        // base >= 0.5
        let result = overlay(0.75, 0.5);
        assert!((result - 0.75).abs() < 1e-6);
    }

    #[test]
    fn test_hard_light_is_overlay_swapped() {
        let a = 0.3_f32;
        let b = 0.6_f32;
        assert!((hard_light(a, b) - overlay(b, a)).abs() < 1e-6);
    }

    #[test]
    fn test_soft_light_neutral_at_half() {
        // soft_light(base, 0.5) should return base (neutral)
        let base = 0.4_f32;
        let result = soft_light(base, 0.5);
        assert!((result - base).abs() < 1e-6);
    }

    #[test]
    fn test_difference_symmetry() {
        assert!((difference(0.3, 0.7) - difference(0.7, 0.3)).abs() < 1e-6);
    }

    #[test]
    fn test_exclusion_neutral_at_zero() {
        assert_eq!(exclusion(0.5, 0.0), 0.5);
    }

    #[test]
    fn test_linear_dodge_clamped() {
        assert_eq!(linear_dodge(0.8, 0.8), 1.0);
    }

    #[test]
    fn test_linear_burn_clamped() {
        assert_eq!(linear_burn(0.1, 0.1), 0.0);
    }

    #[test]
    fn test_blend_mode_enum_multiply() {
        let result = BlendMode::Multiply.apply(0.5, 0.5);
        assert!((result - 0.25).abs() < 1e-6);
    }

    #[test]
    fn test_blend_buffers() {
        let a = vec![0.0, 0.5, 1.0];
        let b = vec![1.0, 0.5, 0.0];
        let mut out = vec![0.0; 3];
        blend_buffers(&a, &b, &mut out, BlendMode::Screen, 1.0);
        // screen(0.0, 1.0) = 1.0; screen(0.5, 0.5) = 0.75; screen(1.0, 0.0) = 1.0
        assert!((out[0] - 1.0).abs() < 1e-6);
        assert!((out[1] - 0.75).abs() < 1e-6);
        assert!((out[2] - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_blend_buffers_mix_zero_passthrough() {
        let a = vec![0.3, 0.6, 0.9];
        let b = vec![0.1, 0.2, 0.3];
        let mut out = vec![0.0; 3];
        blend_buffers(&a, &b, &mut out, BlendMode::Multiply, 0.0);
        for i in 0..3 {
            assert!((out[i] - a[i]).abs() < 1e-6);
        }
    }

    #[test]
    fn test_all_modes_clamp() {
        let modes = [
            BlendMode::Multiply,
            BlendMode::Screen,
            BlendMode::Overlay,
            BlendMode::HardLight,
            BlendMode::SoftLight,
            BlendMode::Difference,
            BlendMode::Exclusion,
            BlendMode::LinearDodge,
            BlendMode::LinearBurn,
        ];
        for mode in modes {
            let result = mode.apply(1.2, -0.2);
            assert!(
                result >= 0.0 && result <= 1.0,
                "Mode {mode:?} out of range: {result}"
            );
        }
    }
}
