//! Photoshop-style blend modes for compositing image layers.
#![allow(dead_code)]

/// Supported blend modes (mirrors Photoshop / CSS naming).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BlendMode {
    /// Source replaces destination.
    Normal,
    /// Multiplies colour values (darkens).
    Multiply,
    /// Inverse multiply (lightens).
    Screen,
    /// Multiply or screen depending on destination.
    Overlay,
    /// Softer version of Overlay.
    SoftLight,
    /// Harsher version of Overlay.
    HardLight,
}

impl BlendMode {
    /// Apply the blend mode to a single normalised channel pair.
    ///
    /// Both `src` and `dst` must be in \[0.0, 1.0\].
    /// Returns the blended value in \[0.0, 1.0\].
    pub fn apply(&self, src: f32, dst: f32) -> f32 {
        let s = src.clamp(0.0, 1.0);
        let d = dst.clamp(0.0, 1.0);
        let result = match self {
            Self::Normal => s,
            Self::Multiply => s * d,
            Self::Screen => 1.0 - (1.0 - s) * (1.0 - d),
            Self::Overlay => {
                if d < 0.5 {
                    2.0 * s * d
                } else {
                    1.0 - 2.0 * (1.0 - s) * (1.0 - d)
                }
            }
            Self::SoftLight => {
                if s <= 0.5 {
                    d - (1.0 - 2.0 * s) * d * (1.0 - d)
                } else {
                    let g = if d <= 0.25 {
                        ((16.0 * d - 12.0) * d + 4.0) * d
                    } else {
                        d.sqrt()
                    };
                    d + (2.0 * s - 1.0) * (g - d)
                }
            }
            Self::HardLight => {
                if s < 0.5 {
                    2.0 * s * d
                } else {
                    1.0 - 2.0 * (1.0 - s) * (1.0 - d)
                }
            }
        };
        result.clamp(0.0, 1.0)
    }

    /// Returns a human-readable label for this blend mode.
    pub fn label(&self) -> &'static str {
        match self {
            Self::Normal => "Normal",
            Self::Multiply => "Multiply",
            Self::Screen => "Screen",
            Self::Overlay => "Overlay",
            Self::SoftLight => "Soft Light",
            Self::HardLight => "Hard Light",
        }
    }
}

/// A single compositing layer (RGBA f32).
#[derive(Debug, Clone)]
pub struct BlendLayer {
    /// RGBA pixel data in row-major order (each element is `[r, g, b, a]`).
    pub pixels: Vec<[f32; 4]>,
    /// Width of this layer in pixels.
    pub width: u32,
    /// Height of this layer in pixels.
    pub height: u32,
    /// Blending mode used when compositing this layer.
    pub mode: BlendMode,
    /// Opacity of this layer in the range \[0.0, 1.0\].
    pub opacity: f32,
}

impl BlendLayer {
    /// Create a new layer filled with a constant colour.
    pub fn solid(width: u32, height: u32, colour: [f32; 4], mode: BlendMode) -> Self {
        let count = (width * height) as usize;
        Self {
            pixels: vec![colour; count],
            width,
            height,
            mode,
            opacity: 1.0,
        }
    }

    /// Blend this layer onto `dst` in place. Both must be the same dimensions.
    pub fn blend_onto(&self, dst: &mut [[f32; 4]]) {
        let alpha = self.opacity.clamp(0.0, 1.0);
        for (d, s) in dst.iter_mut().zip(self.pixels.iter()) {
            let src_a = s[3] * alpha;
            d[0] = lerp(d[0], self.mode.apply(s[0], d[0]), src_a);
            d[1] = lerp(d[1], self.mode.apply(s[1], d[1]), src_a);
            d[2] = lerp(d[2], self.mode.apply(s[2], d[2]), src_a);
            d[3] = (d[3] + src_a * (1.0 - d[3])).clamp(0.0, 1.0);
        }
    }
}

fn lerp(a: f32, b: f32, t: f32) -> f32 {
    a + t * (b - a)
}

/// A stack of layers that can be flattened into a single RGBA f32 image.
#[derive(Debug, Default)]
pub struct BlendStack {
    layers: Vec<BlendLayer>,
}

impl BlendStack {
    /// Create an empty stack.
    pub fn new() -> Self {
        Self::default()
    }

    /// Push a layer onto the top of the stack.
    pub fn push(&mut self, layer: BlendLayer) {
        self.layers.push(layer);
    }

    /// Number of layers in the stack.
    pub fn len(&self) -> usize {
        self.layers.len()
    }

    /// Returns `true` if the stack has no layers.
    pub fn is_empty(&self) -> bool {
        self.layers.is_empty()
    }

    /// Flatten all layers from bottom to top.
    ///
    /// Returns `None` if the stack is empty.
    pub fn flatten(&self) -> Option<Vec<[f32; 4]>> {
        let first = self.layers.first()?;
        let w = first.width;
        let h = first.height;
        let count = (w * h) as usize;
        let mut canvas: Vec<[f32; 4]> = vec![[0.0, 0.0, 0.0, 0.0]; count];
        for layer in &self.layers {
            if layer.width == w && layer.height == h {
                layer.blend_onto(&mut canvas);
            }
        }
        Some(canvas)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // --- BlendMode::apply ---

    #[test]
    fn test_normal_returns_src() {
        let v = BlendMode::Normal.apply(0.7, 0.3);
        assert!((v - 0.7).abs() < 1e-6);
    }

    #[test]
    fn test_multiply_darkens() {
        let v = BlendMode::Multiply.apply(0.5, 0.5);
        assert!((v - 0.25).abs() < 1e-6);
    }

    #[test]
    fn test_multiply_identity_with_one() {
        let v = BlendMode::Multiply.apply(0.8, 1.0);
        assert!((v - 0.8).abs() < 1e-6);
    }

    #[test]
    fn test_screen_lightens() {
        let v = BlendMode::Screen.apply(0.5, 0.5);
        // 1 - 0.5*0.5 = 0.75
        assert!((v - 0.75).abs() < 1e-6);
    }

    #[test]
    fn test_overlay_dark_dst() {
        // dst < 0.5 → 2*s*d
        let v = BlendMode::Overlay.apply(0.5, 0.4);
        assert!((v - 2.0 * 0.5 * 0.4).abs() < 1e-6);
    }

    #[test]
    fn test_overlay_bright_dst() {
        // dst >= 0.5 → 1 - 2*(1-s)*(1-d)
        let v = BlendMode::Overlay.apply(0.5, 0.6);
        let expected = 1.0 - 2.0 * 0.5 * 0.4;
        assert!((v - expected).abs() < 1e-6);
    }

    #[test]
    fn test_hard_light_is_overlay_with_swapped_args() {
        // HardLight(s,d) == Overlay(d,s) by definition
        let hl = BlendMode::HardLight.apply(0.3, 0.7);
        let ov = BlendMode::Overlay.apply(0.7, 0.3);
        assert!((hl - ov).abs() < 1e-6);
    }

    #[test]
    fn test_soft_light_mid_values() {
        let v = BlendMode::SoftLight.apply(0.5, 0.5);
        // s=0.5 → d - (1-1)*d*(1-d) = d = 0.5
        assert!((v - 0.5).abs() < 1e-6);
    }

    #[test]
    fn test_blend_mode_label() {
        assert_eq!(BlendMode::Normal.label(), "Normal");
        assert_eq!(BlendMode::SoftLight.label(), "Soft Light");
    }

    #[test]
    fn test_apply_clamps_output() {
        // Even with out-of-range input the result stays in [0,1].
        let v = BlendMode::Screen.apply(2.0, 2.0);
        assert!(v <= 1.0);
    }

    // --- BlendLayer ---

    #[test]
    fn test_blend_layer_solid_dimensions() {
        let layer = BlendLayer::solid(4, 3, [1.0, 0.0, 0.0, 1.0], BlendMode::Normal);
        assert_eq!(layer.pixels.len(), 12);
    }

    #[test]
    fn test_blend_onto_normal_full_alpha() {
        let layer = BlendLayer::solid(1, 1, [0.5, 0.5, 0.5, 1.0], BlendMode::Normal);
        let mut dst = vec![[0.0f32; 4]];
        layer.blend_onto(&mut dst);
        assert!((dst[0][0] - 0.5).abs() < 1e-5);
    }

    // --- BlendStack ---

    #[test]
    fn test_stack_empty_flatten_returns_none() {
        let stack = BlendStack::new();
        assert!(stack.flatten().is_none());
    }

    #[test]
    fn test_stack_single_layer_flatten() {
        let mut stack = BlendStack::new();
        stack.push(BlendLayer::solid(
            2,
            2,
            [1.0, 0.0, 0.0, 1.0],
            BlendMode::Normal,
        ));
        let flat = stack.flatten().expect("should succeed in test");
        assert_eq!(flat.len(), 4);
        assert!((flat[0][0] - 1.0).abs() < 1e-5);
    }

    #[test]
    fn test_stack_len_and_is_empty() {
        let mut stack = BlendStack::new();
        assert!(stack.is_empty());
        stack.push(BlendLayer::solid(1, 1, [0.0; 4], BlendMode::Normal));
        assert_eq!(stack.len(), 1);
        assert!(!stack.is_empty());
    }
}
