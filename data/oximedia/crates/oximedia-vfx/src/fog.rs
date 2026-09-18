//! Atmospheric fog effects for `OxiMedia` VFX.
//!
//! Provides depth-based fog, volumetric approximation, and colour-tinted
//! atmospheric haze effects for video frames.

use crate::{Color, Frame};

/// Configuration for depth-based fog.
#[derive(Debug, Clone)]
#[allow(dead_code)]
pub struct FogConfig {
    /// Fog colour.
    pub color: Color,
    /// Fog density (0 = no fog, 1 = fully opaque).
    pub density: f32,
    /// Depth at which fog starts (normalised 0–1, where 0 = top of frame).
    pub start_depth: f32,
    /// Depth at which fog reaches full opacity (normalised 0–1).
    pub end_depth: f32,
    /// Type of fog falloff.
    pub falloff: FogFalloff,
}

impl Default for FogConfig {
    fn default() -> Self {
        Self {
            color: Color::new(200, 210, 220, 255),
            density: 0.5,
            start_depth: 0.4,
            end_depth: 1.0,
            falloff: FogFalloff::Linear,
        }
    }
}

/// Fog falloff function.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[allow(dead_code)]
pub enum FogFalloff {
    /// Linear falloff.
    Linear,
    /// Exponential falloff (more realistic).
    Exponential,
    /// Squared-exponential (volumetric approximation).
    ExponentialSquared,
}

impl FogFalloff {
    /// Compute fog factor for a normalised depth value (0–1).
    ///
    /// Returns a value in [0, 1] where 0 = no fog and 1 = fully fogged.
    #[allow(dead_code)]
    pub fn factor(self, depth: f32, density: f32) -> f32 {
        let d = depth.clamp(0.0, 1.0);
        match self {
            Self::Linear => (d * density).clamp(0.0, 1.0),
            Self::Exponential => {
                let e = (-density * d).exp();
                (1.0 - e).clamp(0.0, 1.0)
            }
            Self::ExponentialSquared => {
                let e = (-(density * d) * (density * d)).exp();
                (1.0 - e).clamp(0.0, 1.0)
            }
        }
    }
}

/// Compute the normalised depth for a pixel row `y` in a frame of `height` rows.
///
/// `start` and `end` map the fog start/end to the row range.
#[allow(dead_code)]
#[allow(clippy::cast_precision_loss)]
pub fn row_depth(y: u32, height: u32, start: f32, end: f32) -> f32 {
    if height == 0 || end <= start {
        return 0.0;
    }
    let normalised = y as f32 / height.saturating_sub(1).max(1) as f32;
    ((normalised - start) / (end - start)).clamp(0.0, 1.0)
}

/// Apply depth-based fog to an entire frame.
///
/// Fog is applied row-by-row using the `start_depth` / `end_depth` mapping.
#[allow(dead_code)]
pub fn apply_fog(frame: &mut Frame, config: &FogConfig) {
    for y in 0..frame.height {
        let depth = row_depth(y, frame.height, config.start_depth, config.end_depth);
        let fog_factor = config.falloff.factor(depth, config.density);
        if fog_factor <= 0.0 {
            continue;
        }
        for x in 0..frame.width {
            if let Some(pixel) = frame.get_pixel(x, y) {
                let original = Color::from_rgba(pixel);
                let fogged = original.lerp(config.color, fog_factor);
                frame.set_pixel(x, y, fogged.to_rgba());
            }
        }
    }
}

/// Apply a radial volumetric fog glow centred on a point.
///
/// Produces a soft haze emanating from `(cx, cy)` with the given `radius`.
#[allow(dead_code)]
#[allow(clippy::cast_precision_loss)]
#[allow(clippy::cast_possible_truncation)]
#[allow(clippy::cast_sign_loss)]
pub fn apply_radial_fog(
    frame: &mut Frame,
    cx: f32,
    cy: f32,
    radius: f32,
    fog_color: Color,
    density: f32,
) {
    let density = density.clamp(0.0, 1.0);
    for y in 0..frame.height {
        for x in 0..frame.width {
            let dx = x as f32 - cx;
            let dy = y as f32 - cy;
            let dist = (dx * dx + dy * dy).sqrt();
            if dist > radius {
                continue;
            }
            let t = 1.0 - dist / radius;
            let fog_factor = t * t * density;
            if let Some(pixel) = frame.get_pixel(x, y) {
                let original = Color::from_rgba(pixel);
                let fogged = original.lerp(fog_color, fog_factor);
                frame.set_pixel(x, y, fogged.to_rgba());
            }
        }
    }
}

/// Add a thin colour-tinted atmospheric haze layer.
///
/// Blends `haze_color` uniformly across the entire frame at `strength`.
#[allow(dead_code)]
pub fn apply_haze(frame: &mut Frame, haze_color: Color, strength: f32) {
    let strength = strength.clamp(0.0, 1.0);
    for y in 0..frame.height {
        for x in 0..frame.width {
            if let Some(pixel) = frame.get_pixel(x, y) {
                let original = Color::from_rgba(pixel);
                let hazed = original.lerp(haze_color, strength);
                frame.set_pixel(x, y, hazed.to_rgba());
            }
        }
    }
}

/// Volumetric light shaft approximation.
///
/// Brightens pixels along a direction from source `(sx, sy)` by scattering
/// a fraction `scatter` of the fog colour into them.
#[allow(dead_code)]
#[allow(clippy::cast_precision_loss)]
#[allow(clippy::cast_possible_truncation)]
#[allow(clippy::cast_sign_loss)]
pub fn apply_light_shafts(
    frame: &mut Frame,
    sx: f32,
    sy: f32,
    shaft_color: Color,
    scatter: f32,
    steps: u32,
) {
    let scatter = scatter.clamp(0.0, 1.0);
    let steps = steps.max(1);
    for y in 0..frame.height {
        for x in 0..frame.width {
            let dx = x as f32 - sx;
            let dy = y as f32 - sy;
            // Accumulate shaft contribution along ray
            let mut accumulated = 0.0_f32;
            for s in 0..steps {
                let t = s as f32 / steps as f32;
                let rx = sx + dx * t;
                let ry = sy + dy * t;
                if rx >= 0.0 && ry >= 0.0 && rx < frame.width as f32 && ry < frame.height as f32 {
                    accumulated += 1.0 / steps as f32;
                }
            }
            let shaft_factor = accumulated * scatter;
            if shaft_factor <= 0.0 {
                continue;
            }
            if let Some(pixel) = frame.get_pixel(x, y) {
                let original = Color::from_rgba(pixel);
                let lit = original.lerp(shaft_color, shaft_factor * 0.3);
                frame.set_pixel(x, y, lit.to_rgba());
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_config_density_in_range() {
        let config = FogConfig::default();
        assert!(config.density >= 0.0 && config.density <= 1.0);
    }

    #[test]
    fn test_falloff_linear_zero_at_zero() {
        let f = FogFalloff::Linear.factor(0.0, 0.8);
        assert_eq!(f, 0.0);
    }

    #[test]
    fn test_falloff_linear_proportional() {
        let f = FogFalloff::Linear.factor(0.5, 1.0);
        assert!((f - 0.5).abs() < 1e-5);
    }

    #[test]
    fn test_falloff_exponential_starts_near_zero() {
        let f = FogFalloff::Exponential.factor(0.0, 5.0);
        assert!(f < 0.01);
    }

    #[test]
    fn test_falloff_exponential_reaches_one() {
        let f = FogFalloff::Exponential.factor(1.0, 10.0);
        assert!(f > 0.99);
    }

    #[test]
    fn test_falloff_exp_squared_slower_than_exp() {
        let exp = FogFalloff::Exponential.factor(0.3, 2.0);
        let exp2 = FogFalloff::ExponentialSquared.factor(0.3, 2.0);
        assert!(exp2 < exp, "exp2={exp2} should be less than exp={exp}");
    }

    #[test]
    fn test_row_depth_at_start() {
        let depth = row_depth(40, 100, 0.4, 1.0);
        assert!(depth < 0.02, "depth={depth}");
    }

    #[test]
    fn test_row_depth_before_start_is_zero() {
        let depth = row_depth(10, 100, 0.4, 1.0);
        assert_eq!(depth, 0.0);
    }

    #[test]
    fn test_row_depth_at_end() {
        let depth = row_depth(99, 100, 0.4, 1.0);
        assert!(depth > 0.98, "depth={depth}");
    }

    #[test]
    fn test_apply_fog_modifies_lower_rows() {
        let mut frame = Frame::new(32, 32).expect("should succeed in test");
        frame.clear([100, 100, 100, 255]);
        let config = FogConfig {
            density: 1.0,
            start_depth: 0.0,
            end_depth: 1.0,
            color: Color::new(255, 255, 255, 255),
            falloff: FogFalloff::Linear,
        };
        apply_fog(&mut frame, &config);
        // Bottom rows should be whiter
        let bottom_pixel = frame.get_pixel(0, 31).expect("should succeed in test");
        assert!(bottom_pixel[0] > 100, "Bottom row should be lighter");
    }

    #[test]
    fn test_apply_fog_no_density_noop() {
        let mut frame = Frame::new(16, 16).expect("should succeed in test");
        frame.clear([50, 100, 150, 255]);
        let config = FogConfig {
            density: 0.0,
            ..Default::default()
        };
        let original = frame.data.clone();
        apply_fog(&mut frame, &config);
        assert_eq!(frame.data, original);
    }

    #[test]
    fn test_apply_haze_uniform() {
        let mut frame = Frame::new(8, 8).expect("should succeed in test");
        frame.clear([0, 0, 0, 255]);
        apply_haze(&mut frame, Color::rgb(255, 255, 255), 0.5);
        // All pixels should be gray (~128)
        for chunk in frame.data.chunks(4) {
            assert!(chunk[0] > 100 && chunk[0] < 200, "r={}", chunk[0]);
        }
    }

    #[test]
    fn test_apply_haze_zero_strength_noop() {
        let mut frame = Frame::new(8, 8).expect("should succeed in test");
        frame.clear([200, 100, 50, 255]);
        let original = frame.data.clone();
        apply_haze(&mut frame, Color::white(), 0.0);
        assert_eq!(frame.data, original);
    }

    #[test]
    fn test_apply_radial_fog_does_not_panic() {
        let mut frame = Frame::new(64, 64).expect("should succeed in test");
        apply_radial_fog(
            &mut frame,
            32.0,
            32.0,
            20.0,
            Color::new(200, 210, 220, 255),
            0.6,
        );
    }

    #[test]
    fn test_apply_light_shafts_does_not_panic() {
        let mut frame = Frame::new(64, 64).expect("should succeed in test");
        apply_light_shafts(
            &mut frame,
            32.0,
            0.0,
            Color::new(255, 255, 200, 255),
            0.3,
            8,
        );
    }
}
