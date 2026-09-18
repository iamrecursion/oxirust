//! Edge refinement for keyed footage.

use crate::{EffectParams, Frame, VfxResult, VideoEffect};
use serde::{Deserialize, Serialize};

/// Edge refinement method.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum EdgeRefinementMethod {
    /// Erode edges.
    Erode,
    /// Dilate edges.
    Dilate,
    /// Smooth edges.
    Smooth,
    /// Sharpen edges.
    Sharpen,
}

/// Falloff curve used for feathered edge blending.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum FeatherFalloff {
    /// Linear falloff: alpha decreases linearly with distance.
    Linear,
    /// Smooth (Hermite/smoothstep) falloff for natural-looking edges.
    Smooth,
    /// Gaussian falloff: bell-curve shaped feathering.
    Gaussian,
    /// Cosine falloff: cos-based transition.
    Cosine,
}

impl FeatherFalloff {
    /// Evaluate the falloff at normalised distance `t` in [0, 1].
    ///
    /// Returns a value in [0, 1] where 1 = fully opaque, 0 = fully transparent.
    #[must_use]
    pub fn evaluate(self, t: f32) -> f32 {
        let t = t.clamp(0.0, 1.0);
        match self {
            Self::Linear => 1.0 - t,
            Self::Smooth => {
                // Smoothstep: 1 - (3t^2 - 2t^3)
                let t2 = t * t;
                1.0 - (3.0 * t2 - 2.0 * t2 * t)
            }
            Self::Gaussian => {
                // Gaussian: exp(-4 * t^2), gives a nice bell-curve falloff
                (-4.0 * t * t).exp()
            }
            Self::Cosine => (t * std::f32::consts::FRAC_PI_2).cos(),
        }
    }
}

/// Configuration for feathered edge support.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FeatherConfig {
    /// Enable feathered edges.
    pub enabled: bool,
    /// Feather radius in pixels (how far the soft edge extends).
    pub radius: u32,
    /// Falloff curve type.
    pub falloff: FeatherFalloff,
    /// Inner shrink: how many pixels to erode the matte before feathering.
    /// This prevents fringing on the key edge.
    pub inner_shrink: u32,
    /// Choke: tightens or loosens the matte edge before feathering.
    /// Positive values shrink the matte, negative values expand.
    pub choke: f32,
}

impl Default for FeatherConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            radius: 3,
            falloff: FeatherFalloff::Smooth,
            inner_shrink: 0,
            choke: 0.0,
        }
    }
}

/// Edge refinement effect.
pub struct EdgeRefine {
    method: EdgeRefinementMethod,
    amount: f32,
    radius: u32,
    feather: FeatherConfig,
}

impl EdgeRefine {
    /// Create a new edge refine effect.
    #[must_use]
    pub const fn new(method: EdgeRefinementMethod) -> Self {
        Self {
            method,
            amount: 1.0,
            radius: 1,
            feather: FeatherConfig {
                enabled: false,
                radius: 3,
                falloff: FeatherFalloff::Smooth,
                inner_shrink: 0,
                choke: 0.0,
            },
        }
    }

    /// Set refinement amount (0.0 - 1.0).
    #[must_use]
    pub fn with_amount(mut self, amount: f32) -> Self {
        self.amount = amount.clamp(0.0, 1.0);
        self
    }

    /// Set processing radius.
    #[must_use]
    pub fn with_radius(mut self, radius: u32) -> Self {
        self.radius = radius.max(1);
        self
    }

    /// Enable feathered edges with the given configuration.
    #[must_use]
    pub fn with_feather(mut self, config: FeatherConfig) -> Self {
        self.feather = config;
        self
    }

    /// Convenience: enable feathered edges with default settings and a given radius.
    #[must_use]
    pub fn with_feather_radius(mut self, radius: u32) -> Self {
        self.feather.enabled = true;
        self.feather.radius = radius.max(1);
        self
    }

    /// Set the feather falloff curve.
    #[must_use]
    pub fn with_feather_falloff(mut self, falloff: FeatherFalloff) -> Self {
        self.feather.falloff = falloff;
        self
    }

    /// Compute minimum distance from pixel (x, y) to the nearest edge pixel.
    ///
    /// An edge pixel is defined as one where the alpha transitions between
    /// opaque (>= 128) and transparent (< 128) regions.
    fn compute_edge_distance(
        &self,
        alpha_buffer: &[u8],
        x: u32,
        y: u32,
        width: u32,
        height: u32,
    ) -> f32 {
        let search_radius = self.feather.radius as i32;
        let current_alpha = alpha_buffer[(y * width + x) as usize];
        let current_opaque = current_alpha >= 128;

        let mut min_dist_sq = f32::MAX;

        for dy in -search_radius..=search_radius {
            for dx in -search_radius..=search_radius {
                let nx = x as i32 + dx;
                let ny = y as i32 + dy;

                if nx < 0 || nx >= width as i32 || ny < 0 || ny >= height as i32 {
                    continue;
                }

                let neighbor_alpha = alpha_buffer[(ny as u32 * width + nx as u32) as usize];
                let neighbor_opaque = neighbor_alpha >= 128;

                // If neighbor is on the other side of the edge
                if current_opaque != neighbor_opaque {
                    let dist_sq = (dx * dx + dy * dy) as f32;
                    if dist_sq < min_dist_sq {
                        min_dist_sq = dist_sq;
                    }
                }
            }
        }

        min_dist_sq.sqrt()
    }

    /// Apply feathered edge processing to an alpha buffer.
    ///
    /// Modifies alpha values near edges to create smooth, natural-looking
    /// key edges with configurable falloff curves.
    fn apply_feather(&self, input: &Frame, output: &mut Frame) {
        let width = input.width;
        let height = input.height;
        let feather_radius = self.feather.radius as f32;

        // Extract alpha channel from current output (after initial edge refinement)
        let mut alpha_buffer = Vec::with_capacity((width * height) as usize);
        for y in 0..height {
            for x in 0..width {
                alpha_buffer.push(output.get_pixel(x, y).unwrap_or([0, 0, 0, 0])[3]);
            }
        }

        // Apply choke: shift the alpha threshold
        let choke_offset = (self.feather.choke * 255.0).clamp(-255.0, 255.0) as i32;

        for y in 0..height {
            for x in 0..width {
                let pixel = output.get_pixel(x, y).unwrap_or([0, 0, 0, 0]);
                let alpha_idx = (y * width + x) as usize;
                let raw_alpha = alpha_buffer[alpha_idx];

                // Apply choke to the alpha threshold
                let choked_alpha = (i32::from(raw_alpha) + choke_offset).clamp(0, 255) as u8;

                // Compute distance to nearest edge
                let edge_dist = self.compute_edge_distance(&alpha_buffer, x, y, width, height);

                // If we're near an edge (within feather radius), apply falloff
                let final_alpha = if edge_dist < feather_radius && edge_dist > 0.0 {
                    let normalised = edge_dist / feather_radius;

                    // If we're on the opaque side, feather outward
                    if choked_alpha >= 128 {
                        let falloff = self.feather.falloff.evaluate(1.0 - normalised);
                        (f32::from(choked_alpha) * falloff).clamp(0.0, 255.0) as u8
                    } else {
                        // On the transparent side, feather inward
                        let falloff = self.feather.falloff.evaluate(normalised);
                        (f32::from(choked_alpha) * (1.0 - falloff)).clamp(0.0, 255.0) as u8
                    }
                } else if self.feather.inner_shrink > 0 && choked_alpha >= 128 {
                    // Apply inner shrink: check if pixel is too close to edge
                    if edge_dist <= self.feather.inner_shrink as f32 {
                        0
                    } else {
                        choked_alpha
                    }
                } else {
                    choked_alpha
                };

                output.set_pixel(x, y, [pixel[0], pixel[1], pixel[2], final_alpha]);
            }
        }
    }

    fn get_neighbor_alphas(&self, input: &Frame, x: u32, y: u32) -> Vec<u8> {
        let mut alphas = Vec::new();
        let r = self.radius as i32;

        for dy in -r..=r {
            for dx in -r..=r {
                let nx = (x as i32 + dx).max(0).min(input.width as i32 - 1) as u32;
                let ny = (y as i32 + dy).max(0).min(input.height as i32 - 1) as u32;

                if let Some(pixel) = input.get_pixel(nx, ny) {
                    alphas.push(pixel[3]);
                }
            }
        }

        alphas
    }
}

impl VideoEffect for EdgeRefine {
    fn name(&self) -> &'static str {
        "Edge Refine"
    }

    fn description(&self) -> &'static str {
        "Refine edges of keyed footage"
    }

    fn apply(
        &mut self,
        input: &Frame,
        output: &mut Frame,
        _params: &EffectParams,
    ) -> VfxResult<()> {
        for y in 0..output.height {
            for x in 0..output.width {
                let pixel = input.get_pixel(x, y).unwrap_or([0, 0, 0, 0]);
                let alphas = self.get_neighbor_alphas(input, x, y);

                let new_alpha = match self.method {
                    EdgeRefinementMethod::Erode => {
                        let min_alpha = *alphas.iter().min().unwrap_or(&255);
                        let current = pixel[3];
                        ((1.0 - self.amount) * f32::from(current)
                            + self.amount * f32::from(min_alpha)) as u8
                    }
                    EdgeRefinementMethod::Dilate => {
                        let max_alpha = *alphas.iter().max().unwrap_or(&0);
                        let current = pixel[3];
                        ((1.0 - self.amount) * f32::from(current)
                            + self.amount * f32::from(max_alpha)) as u8
                    }
                    EdgeRefinementMethod::Smooth => {
                        let avg_alpha =
                            alphas.iter().map(|&a| u32::from(a)).sum::<u32>() / alphas.len() as u32;
                        let current = pixel[3];
                        ((1.0 - self.amount) * f32::from(current) + self.amount * avg_alpha as f32)
                            as u8
                    }
                    EdgeRefinementMethod::Sharpen => {
                        let avg_alpha =
                            alphas.iter().map(|&a| u32::from(a)).sum::<u32>() / alphas.len() as u32;
                        let current = i32::from(pixel[3]);
                        let sharpened =
                            current + ((current - avg_alpha as i32) as f32 * self.amount) as i32;
                        sharpened.clamp(0, 255) as u8
                    }
                };

                output.set_pixel(x, y, [pixel[0], pixel[1], pixel[2], new_alpha]);
            }
        }

        // Apply feathered edge processing if enabled
        if self.feather.enabled {
            self.apply_feather(input, output);
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

    /// Create a test frame with a hard-edged circle of opaque pixels.
    fn create_keyed_frame(width: u32, height: u32) -> Frame {
        let mut frame = Frame::new(width, height).expect("test frame");
        let cx = width as f32 / 2.0;
        let cy = height as f32 / 2.0;
        let radius = (width.min(height) as f32) / 3.0;

        for y in 0..height {
            for x in 0..width {
                let dx = x as f32 - cx;
                let dy = y as f32 - cy;
                let dist = (dx * dx + dy * dy).sqrt();
                let alpha = if dist < radius { 255 } else { 0 };
                frame.set_pixel(x, y, [128, 200, 100, alpha]);
            }
        }
        frame
    }

    #[test]
    fn test_edge_refine() {
        let methods = [
            EdgeRefinementMethod::Erode,
            EdgeRefinementMethod::Dilate,
            EdgeRefinementMethod::Smooth,
        ];

        for method in methods {
            let mut refine = EdgeRefine::new(method);
            let input = Frame::new(100, 100).expect("should succeed in test");
            let mut output = Frame::new(100, 100).expect("should succeed in test");
            let params = EffectParams::new();
            refine
                .apply(&input, &mut output, &params)
                .expect("should succeed in test");
        }
    }

    #[test]
    fn test_feather_falloff_linear() {
        assert!((FeatherFalloff::Linear.evaluate(0.0) - 1.0).abs() < 0.01);
        assert!(FeatherFalloff::Linear.evaluate(1.0).abs() < 0.01);
        assert!((FeatherFalloff::Linear.evaluate(0.5) - 0.5).abs() < 0.01);
    }

    #[test]
    fn test_feather_falloff_smooth() {
        assert!((FeatherFalloff::Smooth.evaluate(0.0) - 1.0).abs() < 0.01);
        assert!(FeatherFalloff::Smooth.evaluate(1.0).abs() < 0.01);
    }

    #[test]
    fn test_feather_falloff_gaussian() {
        assert!((FeatherFalloff::Gaussian.evaluate(0.0) - 1.0).abs() < 0.01);
        let edge = FeatherFalloff::Gaussian.evaluate(1.0);
        assert!(edge < 0.1, "gaussian at edge should be near 0, got {edge}");
    }

    #[test]
    fn test_feather_falloff_cosine() {
        assert!((FeatherFalloff::Cosine.evaluate(0.0) - 1.0).abs() < 0.01);
        assert!(FeatherFalloff::Cosine.evaluate(1.0).abs() < 0.01);
    }

    #[test]
    fn test_feather_falloff_monotonic() {
        let falloffs = [
            FeatherFalloff::Linear,
            FeatherFalloff::Smooth,
            FeatherFalloff::Gaussian,
            FeatherFalloff::Cosine,
        ];
        for falloff in falloffs {
            let mut prev = falloff.evaluate(0.0);
            for i in 1..=20 {
                let t = i as f32 / 20.0;
                let val = falloff.evaluate(t);
                assert!(
                    val <= prev + 0.001,
                    "{falloff:?} not monotonically decreasing at t={t}"
                );
                prev = val;
            }
        }
    }

    #[test]
    fn test_feathered_edge_produces_gradient() {
        let input = create_keyed_frame(100, 100);
        let mut output = Frame::new(100, 100).expect("test frame");
        let params = EffectParams::new();

        let mut refine = EdgeRefine::new(EdgeRefinementMethod::Smooth)
            .with_feather_radius(5)
            .with_feather_falloff(FeatherFalloff::Smooth);

        refine.apply(&input, &mut output, &params).expect("feather");

        // Check that near the edge boundary, we get intermediate alpha values
        // (not just 0 or 255)
        let mut has_intermediate = false;
        for y in 0..100 {
            for x in 0..100 {
                let pixel = output.get_pixel(x, y).unwrap_or([0, 0, 0, 0]);
                if pixel[3] > 5 && pixel[3] < 250 {
                    has_intermediate = true;
                }
            }
        }
        assert!(
            has_intermediate,
            "feathered edge should produce intermediate alpha values"
        );
    }

    #[test]
    fn test_feathered_edge_disabled_by_default() {
        let refine = EdgeRefine::new(EdgeRefinementMethod::Smooth);
        assert!(!refine.feather.enabled, "feather should be off by default");
    }

    #[test]
    fn test_feathered_edge_with_config() {
        let config = FeatherConfig {
            enabled: true,
            radius: 8,
            falloff: FeatherFalloff::Gaussian,
            inner_shrink: 1,
            choke: 0.1,
        };
        let mut refine = EdgeRefine::new(EdgeRefinementMethod::Smooth).with_feather(config);
        let input = create_keyed_frame(80, 80);
        let mut output = Frame::new(80, 80).expect("test frame");
        let params = EffectParams::new();
        refine
            .apply(&input, &mut output, &params)
            .expect("feather with config");
    }

    #[test]
    fn test_feathered_edge_preserves_rgb() {
        let input = create_keyed_frame(60, 60);
        let mut output = Frame::new(60, 60).expect("test frame");
        let params = EffectParams::new();

        let mut refine = EdgeRefine::new(EdgeRefinementMethod::Smooth).with_feather_radius(3);

        refine.apply(&input, &mut output, &params).expect("feather");

        // RGB values should be preserved (feathering only affects alpha)
        for y in 0..60 {
            for x in 0..60 {
                let inp = input.get_pixel(x, y).unwrap_or([0, 0, 0, 0]);
                let out = output.get_pixel(x, y).unwrap_or([0, 0, 0, 0]);
                assert_eq!(inp[0], out[0], "R preserved at ({x},{y})");
                assert_eq!(inp[1], out[1], "G preserved at ({x},{y})");
                assert_eq!(inp[2], out[2], "B preserved at ({x},{y})");
            }
        }
    }

    #[test]
    fn test_feathered_edge_all_falloff_types() {
        let falloffs = [
            FeatherFalloff::Linear,
            FeatherFalloff::Smooth,
            FeatherFalloff::Gaussian,
            FeatherFalloff::Cosine,
        ];

        for falloff in falloffs {
            let input = create_keyed_frame(50, 50);
            let mut output = Frame::new(50, 50).expect("test frame");
            let params = EffectParams::new();

            let mut refine = EdgeRefine::new(EdgeRefinementMethod::Smooth)
                .with_feather_radius(4)
                .with_feather_falloff(falloff);

            refine
                .apply(&input, &mut output, &params)
                .unwrap_or_else(|_| panic!("feather with {falloff:?}"));
        }
    }

    #[test]
    fn test_feathered_edge_sharpen_method() {
        let input = create_keyed_frame(50, 50);
        let mut output = Frame::new(50, 50).expect("test frame");
        let params = EffectParams::new();

        let mut refine = EdgeRefine::new(EdgeRefinementMethod::Sharpen).with_feather_radius(3);

        refine
            .apply(&input, &mut output, &params)
            .expect("sharpen + feather");
    }

    #[test]
    fn test_feather_config_default() {
        let cfg = FeatherConfig::default();
        assert!(!cfg.enabled);
        assert_eq!(cfg.radius, 3);
        assert_eq!(cfg.falloff, FeatherFalloff::Smooth);
        assert_eq!(cfg.inner_shrink, 0);
        assert!((cfg.choke).abs() < 0.01);
    }
}
