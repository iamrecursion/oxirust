//! Vignette effect with customizable shape and falloff.
//!
//! Provides a configurable vignetting effect commonly used in post-production
//! to darken frame edges. Supports circular, elliptical, and rectangular shapes
//! with smooth falloff curves and tinting options.

use crate::{Color, EffectParams, Frame, VfxError, VfxResult, VideoEffect};
use serde::{Deserialize, Serialize};

/// Shape of the vignette mask.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum VignetteShape {
    /// Circular vignette (uniform radius).
    Circular,
    /// Elliptical vignette (follows frame aspect ratio).
    Elliptical,
    /// Rectangular vignette with rounded corners.
    Rectangular,
}

/// Falloff curve type controlling how the vignette darkens toward edges.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum FalloffCurve {
    /// Linear falloff: `1 - t`.
    Linear,
    /// Smooth (Hermite) falloff: `3t^2 - 2t^3` (smoothstep).
    Smooth,
    /// Quadratic falloff: `1 - t^2`.
    Quadratic,
    /// Cosine falloff: `cos(t * PI/2)`.
    Cosine,
    /// Exponential falloff: `exp(-k * t^2)`.
    Exponential,
}

impl FalloffCurve {
    /// Evaluate the falloff at parameter `t` in [0, 1].
    ///
    /// Returns a value in [0, 1] where 1 = fully lit, 0 = fully dark.
    #[must_use]
    pub fn evaluate(self, t: f32) -> f32 {
        let t = t.clamp(0.0, 1.0);
        match self {
            Self::Linear => 1.0 - t,
            Self::Smooth => {
                // Smoothstep inverse: bright at t=0, dark at t=1
                let t2 = t * t;
                1.0 - (3.0 * t2 - 2.0 * t2 * t)
            }
            Self::Quadratic => 1.0 - t * t,
            Self::Cosine => (t * std::f32::consts::FRAC_PI_2).cos(),
            Self::Exponential => (-4.0 * t * t).exp(),
        }
    }
}

/// Configuration for the vignette effect.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VignetteConfig {
    /// Shape of the vignette.
    pub shape: VignetteShape,
    /// Falloff curve type.
    pub falloff: FalloffCurve,
    /// Inner radius where darkening begins (0.0-1.0 normalised).
    /// 0.0 = darkening from center, 1.0 = no darkening.
    pub inner_radius: f32,
    /// Outer radius where full darkening is reached (0.0-1.0 normalised).
    /// Must be >= inner_radius.
    pub outer_radius: f32,
    /// Overall strength of the vignette effect (0.0 = no effect, 1.0 = full).
    pub strength: f32,
    /// Center X position (normalised 0.0-1.0, 0.5 = center).
    pub center_x: f32,
    /// Center Y position (normalised 0.0-1.0, 0.5 = center).
    pub center_y: f32,
    /// Roundness for rectangular shape (0.0 = sharp corners, 1.0 = fully rounded).
    pub roundness: f32,
    /// Optional tint color applied to darkened regions.
    /// When `None`, edges are darkened toward black.
    pub tint: Option<Color>,
    /// Aspect ratio override for elliptical mode.
    /// When `None`, uses the frame's native aspect ratio.
    pub aspect_ratio: Option<f32>,
}

impl Default for VignetteConfig {
    fn default() -> Self {
        Self {
            shape: VignetteShape::Elliptical,
            falloff: FalloffCurve::Smooth,
            inner_radius: 0.4,
            outer_radius: 1.0,
            strength: 0.7,
            center_x: 0.5,
            center_y: 0.5,
            roundness: 0.5,
            tint: None,
            aspect_ratio: None,
        }
    }
}

/// Vignette video effect.
///
/// Darkens the edges of each frame according to a configurable shape,
/// falloff curve, strength, and optional tint color.
pub struct VignetteEffect {
    config: VignetteConfig,
}

impl VignetteEffect {
    /// Create a new vignette effect with the given configuration.
    #[must_use]
    pub fn new(config: VignetteConfig) -> Self {
        Self { config }
    }

    /// Create a default elliptical vignette.
    #[must_use]
    pub fn default_elliptical() -> Self {
        Self::new(VignetteConfig::default())
    }

    /// Create a circular vignette.
    #[must_use]
    pub fn circular() -> Self {
        Self::new(VignetteConfig {
            shape: VignetteShape::Circular,
            ..VignetteConfig::default()
        })
    }

    /// Create a rectangular vignette.
    #[must_use]
    pub fn rectangular() -> Self {
        Self::new(VignetteConfig {
            shape: VignetteShape::Rectangular,
            ..VignetteConfig::default()
        })
    }

    /// Set the inner radius.
    #[must_use]
    pub fn with_inner_radius(mut self, r: f32) -> Self {
        self.config.inner_radius = r.clamp(0.0, 1.0);
        self
    }

    /// Set the outer radius.
    #[must_use]
    pub fn with_outer_radius(mut self, r: f32) -> Self {
        self.config.outer_radius = r.clamp(0.0, 2.0);
        self
    }

    /// Set the effect strength.
    #[must_use]
    pub fn with_strength(mut self, s: f32) -> Self {
        self.config.strength = s.clamp(0.0, 1.0);
        self
    }

    /// Set center position.
    #[must_use]
    pub fn with_center(mut self, x: f32, y: f32) -> Self {
        self.config.center_x = x.clamp(0.0, 1.0);
        self.config.center_y = y.clamp(0.0, 1.0);
        self
    }

    /// Set a tint color for the vignette.
    #[must_use]
    pub fn with_tint(mut self, color: Color) -> Self {
        self.config.tint = Some(color);
        self
    }

    /// Set the falloff curve.
    #[must_use]
    pub fn with_falloff(mut self, falloff: FalloffCurve) -> Self {
        self.config.falloff = falloff;
        self
    }

    /// Set roundness for rectangular shape.
    #[must_use]
    pub fn with_roundness(mut self, roundness: f32) -> Self {
        self.config.roundness = roundness.clamp(0.0, 1.0);
        self
    }

    /// Compute the normalised distance from the vignette center for a given pixel.
    ///
    /// Returns a value where 0.0 = at center, 1.0 = at the edge/corner
    /// according to the configured shape.
    fn compute_distance(&self, x: u32, y: u32, width: u32, height: u32) -> f32 {
        let w = width as f32;
        let h = height as f32;

        // Normalised pixel coordinates
        let nx = x as f32 / w;
        let ny = y as f32 / h;

        // Distance from vignette center
        let dx = nx - self.config.center_x;
        let dy = ny - self.config.center_y;

        match self.config.shape {
            VignetteShape::Circular => {
                // Uniform circular distance
                (dx * dx + dy * dy).sqrt() * 2.0 // Scale so corner ~ 1.0
            }
            VignetteShape::Elliptical => {
                // Account for aspect ratio
                let aspect = self.config.aspect_ratio.unwrap_or(w / h.max(1.0));
                let scaled_dx = dx * aspect.max(0.01);
                let scaled_dy = dy;
                (scaled_dx * scaled_dx + scaled_dy * scaled_dy).sqrt() * 2.0
            }
            VignetteShape::Rectangular => {
                // Chebyshev distance blended with Euclidean via roundness
                let abs_dx = dx.abs() * 2.0;
                let abs_dy = dy.abs() * 2.0;

                let chebyshev = abs_dx.max(abs_dy);
                let euclidean = (abs_dx * abs_dx + abs_dy * abs_dy).sqrt();

                // Blend: roundness=0 => rectangular, roundness=1 => circular
                let r = self.config.roundness;
                chebyshev * (1.0 - r) + euclidean * r
            }
        }
    }

    /// Compute vignette factor (brightness multiplier) for a pixel.
    ///
    /// Returns a value in [0, 1] where 1 = no darkening, 0 = fully dark.
    fn vignette_factor(&self, dist: f32) -> f32 {
        let inner = self.config.inner_radius;
        let outer = self.config.outer_radius;

        if dist <= inner {
            1.0
        } else if dist >= outer || outer <= inner {
            let base = self.config.falloff.evaluate(1.0);
            1.0 - self.config.strength * (1.0 - base)
        } else {
            let t = (dist - inner) / (outer - inner);
            let falloff = self.config.falloff.evaluate(t);
            // Blend between 1.0 (no effect) and falloff based on strength
            1.0 - self.config.strength * (1.0 - falloff)
        }
    }
}

impl VideoEffect for VignetteEffect {
    fn name(&self) -> &str {
        "Vignette"
    }

    fn description(&self) -> &'static str {
        "Vignette effect with customizable shape, falloff, and tint"
    }

    fn apply(
        &mut self,
        input: &Frame,
        output: &mut Frame,
        _params: &EffectParams,
    ) -> VfxResult<()> {
        if input.width != output.width || input.height != output.height {
            return Err(VfxError::InvalidDimensions {
                width: output.width,
                height: output.height,
            });
        }

        let tint_r = self.config.tint.map_or(0.0, |c| f32::from(c.r));
        let tint_g = self.config.tint.map_or(0.0, |c| f32::from(c.g));
        let tint_b = self.config.tint.map_or(0.0, |c| f32::from(c.b));
        let has_tint = self.config.tint.is_some();

        for y in 0..input.height {
            for x in 0..input.width {
                let pixel = input.get_pixel(x, y).unwrap_or([0, 0, 0, 0]);
                let dist = self.compute_distance(x, y, input.width, input.height);
                let factor = self.vignette_factor(dist);

                let (r, g, b) = if has_tint {
                    // Blend toward tint color as factor decreases
                    let inv = 1.0 - factor;
                    (
                        (f32::from(pixel[0]) * factor + tint_r * inv).clamp(0.0, 255.0) as u8,
                        (f32::from(pixel[1]) * factor + tint_g * inv).clamp(0.0, 255.0) as u8,
                        (f32::from(pixel[2]) * factor + tint_b * inv).clamp(0.0, 255.0) as u8,
                    )
                } else {
                    // Darken toward black
                    (
                        (f32::from(pixel[0]) * factor) as u8,
                        (f32::from(pixel[1]) * factor) as u8,
                        (f32::from(pixel[2]) * factor) as u8,
                    )
                };

                output.set_pixel(x, y, [r, g, b, pixel[3]]);
            }
        }

        Ok(())
    }

    fn supports_gpu(&self) -> bool {
        true
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn white_frame(w: u32, h: u32) -> Frame {
        let mut f = Frame::new(w, h).expect("test frame");
        f.clear([255, 255, 255, 255]);
        f
    }

    #[test]
    fn test_vignette_center_brighter_than_edge() {
        let mut effect = VignetteEffect::default_elliptical()
            .with_strength(1.0)
            .with_inner_radius(0.2)
            .with_outer_radius(0.8);
        let input = white_frame(100, 100);
        let mut output = Frame::new(100, 100).expect("test frame");
        let params = EffectParams::new();
        effect
            .apply(&input, &mut output, &params)
            .expect("vignette");

        let center = output.get_pixel(50, 50).expect("center");
        let corner = output.get_pixel(0, 0).expect("corner");

        // Center should be brighter than corner
        assert!(
            center[0] > corner[0],
            "center R={} should be > corner R={}",
            center[0],
            corner[0]
        );
    }

    #[test]
    fn test_vignette_zero_strength_preserves_input() {
        let mut effect = VignetteEffect::default_elliptical().with_strength(0.0);
        let input = white_frame(50, 50);
        let mut output = Frame::new(50, 50).expect("test frame");
        let params = EffectParams::new();
        effect
            .apply(&input, &mut output, &params)
            .expect("vignette");

        for y in 0..50 {
            for x in 0..50 {
                let inp = input.get_pixel(x, y).expect("pixel");
                let out = output.get_pixel(x, y).expect("pixel");
                assert_eq!(inp, out, "zero strength should preserve pixel at ({x},{y})");
            }
        }
    }

    #[test]
    fn test_vignette_circular() {
        let mut effect = VignetteEffect::circular().with_strength(0.8);
        let input = white_frame(80, 80);
        let mut output = Frame::new(80, 80).expect("test frame");
        let params = EffectParams::new();
        effect
            .apply(&input, &mut output, &params)
            .expect("vignette");

        // Symmetry: pixels at equal distance from center should have same brightness
        let p1 = output.get_pixel(40, 0).expect("top");
        let p2 = output.get_pixel(40, 79).expect("bottom");
        // Due to circular symmetry, these should be very close
        assert!((i32::from(p1[0]) - i32::from(p2[0])).abs() < 5);
    }

    #[test]
    fn test_vignette_rectangular() {
        let mut effect = VignetteEffect::rectangular()
            .with_roundness(0.0)
            .with_strength(1.0);
        let input = white_frame(100, 100);
        let mut output = Frame::new(100, 100).expect("test frame");
        let params = EffectParams::new();
        effect
            .apply(&input, &mut output, &params)
            .expect("vignette");

        // With rectangular shape, midpoints of edges should have more darkening
        // than center
        let center = output.get_pixel(50, 50).expect("center");
        let edge = output.get_pixel(0, 50).expect("left edge");
        assert!(center[0] >= edge[0]);
    }

    #[test]
    fn test_vignette_with_tint() {
        let tint = Color::new(0, 0, 128, 255); // dark blue
        let mut effect = VignetteEffect::default_elliptical()
            .with_strength(1.0)
            .with_inner_radius(0.0)
            .with_outer_radius(0.5)
            .with_tint(tint);
        let input = white_frame(100, 100);
        let mut output = Frame::new(100, 100).expect("test frame");
        let params = EffectParams::new();
        effect
            .apply(&input, &mut output, &params)
            .expect("vignette");

        // Corner pixel should have blue tint influence
        let corner = output.get_pixel(0, 0).expect("corner");
        // Blue should be higher relative to red/green due to tint
        assert!(
            corner[2] >= corner[0],
            "blue={} should be >= red={} due to tint",
            corner[2],
            corner[0]
        );
    }

    #[test]
    fn test_vignette_custom_center() {
        let mut effect = VignetteEffect::default_elliptical()
            .with_center(0.0, 0.0)
            .with_strength(1.0)
            .with_inner_radius(0.1)
            .with_outer_radius(0.5);
        let input = white_frame(100, 100);
        let mut output = Frame::new(100, 100).expect("test frame");
        let params = EffectParams::new();
        effect
            .apply(&input, &mut output, &params)
            .expect("vignette");

        // Top-left corner should be brightest now
        let top_left = output.get_pixel(0, 0).expect("top-left");
        let center = output.get_pixel(50, 50).expect("center");
        assert!(
            top_left[0] >= center[0],
            "top-left should be >= center brightness when vignette centered at (0,0)"
        );
    }

    #[test]
    fn test_falloff_curve_linear() {
        assert_eq!(FalloffCurve::Linear.evaluate(0.0), 1.0);
        assert_eq!(FalloffCurve::Linear.evaluate(1.0), 0.0);
        assert!((FalloffCurve::Linear.evaluate(0.5) - 0.5).abs() < 0.01);
    }

    #[test]
    fn test_falloff_curve_smooth() {
        assert_eq!(FalloffCurve::Smooth.evaluate(0.0), 1.0);
        assert!((FalloffCurve::Smooth.evaluate(1.0)).abs() < 0.01);
        // Smoothstep should be close to 0.5 at midpoint
        let mid = FalloffCurve::Smooth.evaluate(0.5);
        assert!((mid - 0.5).abs() < 0.1, "smooth mid={mid}");
    }

    #[test]
    fn test_falloff_curve_quadratic() {
        assert_eq!(FalloffCurve::Quadratic.evaluate(0.0), 1.0);
        assert_eq!(FalloffCurve::Quadratic.evaluate(1.0), 0.0);
        assert!((FalloffCurve::Quadratic.evaluate(0.5) - 0.75).abs() < 0.01);
    }

    #[test]
    fn test_falloff_curve_cosine() {
        assert!((FalloffCurve::Cosine.evaluate(0.0) - 1.0).abs() < 0.01);
        assert!(FalloffCurve::Cosine.evaluate(1.0).abs() < 0.01);
    }

    #[test]
    fn test_falloff_curve_exponential() {
        assert!((FalloffCurve::Exponential.evaluate(0.0) - 1.0).abs() < 0.01);
        // At t=1.0, exp(-4) ~ 0.018
        let edge = FalloffCurve::Exponential.evaluate(1.0);
        assert!(edge < 0.1, "exponential at edge={edge}");
    }

    #[test]
    fn test_falloff_curves_monotonically_decrease() {
        let curves = [
            FalloffCurve::Linear,
            FalloffCurve::Smooth,
            FalloffCurve::Quadratic,
            FalloffCurve::Cosine,
            FalloffCurve::Exponential,
        ];
        for curve in curves {
            let mut prev = curve.evaluate(0.0);
            for i in 1..=20 {
                let t = i as f32 / 20.0;
                let val = curve.evaluate(t);
                assert!(
                    val <= prev + 0.001,
                    "{curve:?} not monotonically decreasing at t={t}: {val} > {prev}"
                );
                prev = val;
            }
        }
    }

    #[test]
    fn test_vignette_dimension_mismatch() {
        let mut effect = VignetteEffect::default_elliptical();
        let input = Frame::new(100, 100).expect("test frame");
        let mut output = Frame::new(50, 50).expect("test frame");
        let params = EffectParams::new();
        assert!(effect.apply(&input, &mut output, &params).is_err());
    }

    #[test]
    fn test_vignette_1x1_frame() {
        let mut effect = VignetteEffect::default_elliptical();
        let input = white_frame(1, 1);
        let mut output = Frame::new(1, 1).expect("test frame");
        let params = EffectParams::new();
        effect.apply(&input, &mut output, &params).expect("1x1");
    }

    #[test]
    fn test_vignette_preserves_alpha() {
        let mut effect = VignetteEffect::default_elliptical().with_strength(1.0);
        let mut input = Frame::new(50, 50).expect("test frame");
        input.clear([255, 255, 255, 128]); // Semi-transparent
        let mut output = Frame::new(50, 50).expect("test frame");
        let params = EffectParams::new();
        effect
            .apply(&input, &mut output, &params)
            .expect("vignette");

        // Alpha should be preserved
        for y in 0..50 {
            for x in 0..50 {
                let out = output.get_pixel(x, y).expect("pixel");
                assert_eq!(out[3], 128, "alpha should be preserved at ({x},{y})");
            }
        }
    }

    #[test]
    fn test_vignette_name() {
        let effect = VignetteEffect::default_elliptical();
        assert_eq!(effect.name(), "Vignette");
        assert!(!effect.description().is_empty());
    }

    #[test]
    fn test_vignette_config_default() {
        let cfg = VignetteConfig::default();
        assert_eq!(cfg.shape, VignetteShape::Elliptical);
        assert_eq!(cfg.falloff, FalloffCurve::Smooth);
        assert!(cfg.inner_radius < cfg.outer_radius);
    }
}
