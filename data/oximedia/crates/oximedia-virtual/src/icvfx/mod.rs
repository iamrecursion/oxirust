//! In-Camera VFX (ICVFX) compositing subsystem
//!
//! Provides real-time compositing of foreground and background elements
//! with depth-based blending for in-camera visual effects.

pub mod background;
pub mod composite;
pub mod depth;
pub mod foreground;
pub mod frustum_tracking;
pub mod tiled;

pub use tiled::TiledCompositor;

use serde::{Deserialize, Serialize};

/// Composite layer
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CompositeLayer {
    /// Layer name
    pub name: String,
    /// Layer opacity (0.0 - 1.0)
    pub opacity: f32,
    /// Layer enabled
    pub enabled: bool,
    /// Blend mode
    pub blend_mode: BlendMode,
}

impl CompositeLayer {
    /// Create new composite layer
    #[must_use]
    pub fn new(name: String) -> Self {
        Self {
            name,
            opacity: 1.0,
            enabled: true,
            blend_mode: BlendMode::Normal,
        }
    }
}

/// Blend mode for compositing
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum BlendMode {
    /// Normal alpha blending
    Normal,
    /// Additive blending
    Add,
    /// Multiply blending
    Multiply,
    /// Screen blending
    Screen,
    /// Overlay blending
    Overlay,
}

impl BlendMode {
    /// Apply blend mode to two colors
    #[must_use]
    pub fn blend(&self, base: [f32; 3], blend: [f32; 3], alpha: f32) -> [f32; 3] {
        match self {
            BlendMode::Normal => Self::blend_normal(base, blend, alpha),
            BlendMode::Add => Self::blend_add(base, blend, alpha),
            BlendMode::Multiply => Self::blend_multiply(base, blend, alpha),
            BlendMode::Screen => Self::blend_screen(base, blend, alpha),
            BlendMode::Overlay => Self::blend_overlay(base, blend, alpha),
        }
    }

    fn blend_normal(base: [f32; 3], blend: [f32; 3], alpha: f32) -> [f32; 3] {
        [
            base[0] * (1.0 - alpha) + blend[0] * alpha,
            base[1] * (1.0 - alpha) + blend[1] * alpha,
            base[2] * (1.0 - alpha) + blend[2] * alpha,
        ]
    }

    fn blend_add(base: [f32; 3], blend: [f32; 3], alpha: f32) -> [f32; 3] {
        [
            (base[0] + blend[0] * alpha).min(1.0),
            (base[1] + blend[1] * alpha).min(1.0),
            (base[2] + blend[2] * alpha).min(1.0),
        ]
    }

    fn blend_multiply(base: [f32; 3], blend: [f32; 3], _alpha: f32) -> [f32; 3] {
        [base[0] * blend[0], base[1] * blend[1], base[2] * blend[2]]
    }

    fn blend_screen(base: [f32; 3], blend: [f32; 3], _alpha: f32) -> [f32; 3] {
        [
            1.0 - (1.0 - base[0]) * (1.0 - blend[0]),
            1.0 - (1.0 - base[1]) * (1.0 - blend[1]),
            1.0 - (1.0 - base[2]) * (1.0 - blend[2]),
        ]
    }

    fn blend_overlay(base: [f32; 3], blend: [f32; 3], _alpha: f32) -> [f32; 3] {
        let overlay = |b: f32, s: f32| {
            if b < 0.5 {
                2.0 * b * s
            } else {
                1.0 - 2.0 * (1.0 - b) * (1.0 - s)
            }
        };

        [
            overlay(base[0], blend[0]),
            overlay(base[1], blend[1]),
            overlay(base[2], blend[2]),
        ]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_composite_layer() {
        let layer = CompositeLayer::new("Test Layer".to_string());
        assert_eq!(layer.name, "Test Layer");
        assert_eq!(layer.opacity, 1.0);
        assert!(layer.enabled);
    }

    #[test]
    fn test_blend_normal() {
        let base = [1.0, 0.5, 0.0];
        let blend = [0.0, 0.5, 1.0];
        let result = BlendMode::Normal.blend(base, blend, 0.5);

        assert!((result[0] - 0.5).abs() < 1e-6);
        assert!((result[1] - 0.5).abs() < 1e-6);
        assert!((result[2] - 0.5).abs() < 1e-6);
    }

    #[test]
    fn test_blend_add() {
        let base = [0.5, 0.5, 0.5];
        let blend = [0.5, 0.5, 0.5];
        let result = BlendMode::Add.blend(base, blend, 1.0);

        assert!((result[0] - 1.0).abs() < 1e-6);
        assert!((result[1] - 1.0).abs() < 1e-6);
        assert!((result[2] - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_blend_multiply() {
        let base = [1.0, 0.5, 0.25];
        let blend = [0.5, 0.5, 0.5];
        let result = BlendMode::Multiply.blend(base, blend, 1.0);

        assert!((result[0] - 0.5).abs() < 1e-6);
        assert!((result[1] - 0.25).abs() < 1e-6);
        assert!((result[2] - 0.125).abs() < 1e-6);
    }
}
