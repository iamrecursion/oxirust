//! Wipe transitions with 30+ patterns.

use crate::{EffectParams, Frame, TransitionEffect, VfxResult};
use serde::{Deserialize, Serialize};

/// Wipe pattern types.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum WipePattern {
    /// Wipe from left to right.
    LeftToRight,
    /// Wipe from right to left.
    RightToLeft,
    /// Wipe from top to bottom.
    TopToBottom,
    /// Wipe from bottom to top.
    BottomToTop,
    /// Diagonal wipe top-left to bottom-right.
    DiagonalTLBR,
    /// Diagonal wipe top-right to bottom-left.
    DiagonalTRBL,
    /// Diagonal wipe bottom-left to top-right.
    DiagonalBLTR,
    /// Diagonal wipe bottom-right to top-left.
    DiagonalBRTL,
    /// Circular wipe from center.
    CircleIn,
    /// Circular wipe to center.
    CircleOut,
    /// Horizontal barn door (split from center).
    BarnDoorHorizontal,
    /// Vertical barn door (split from center).
    BarnDoorVertical,
    /// Box wipe from center outward.
    BoxIn,
    /// Box wipe from edges inward.
    BoxOut,
    /// Clock wipe (12 o'clock).
    ClockWipe,
    /// Wedge wipe.
    Wedge,
    /// Iris round.
    IrisRound,
    /// Iris diamond.
    IrisDiamond,
    /// Checkerboard pattern.
    Checkerboard,
    /// Horizontal blinds.
    BlindsHorizontal,
    /// Vertical blinds.
    BlindsVertical,
    /// Random blocks.
    RandomBlocks,
    /// Radial wipe.
    Radial,
    /// Spiral wipe.
    Spiral,
    /// Heart shape.
    Heart,
    /// Star shape.
    Star,
    /// Diamond shape.
    Diamond,
    /// Cross shape.
    Cross,
    /// Arrow right.
    ArrowRight,
    /// Arrow left.
    ArrowLeft,
    /// Wave horizontal.
    WaveHorizontal,
    /// Wave vertical.
    WaveVertical,
}

/// Wipe transition.
pub struct Wipe {
    pattern: WipePattern,
    feather: f32,
    reverse: bool,
}

impl Wipe {
    /// Create a new wipe transition.
    #[must_use]
    pub const fn new(pattern: WipePattern) -> Self {
        Self {
            pattern,
            feather: 0.05,
            reverse: false,
        }
    }

    /// Set edge feathering amount (0.0 - 1.0).
    #[must_use]
    pub fn with_feather(mut self, feather: f32) -> Self {
        self.feather = feather.clamp(0.0, 1.0);
        self
    }

    /// Reverse wipe direction.
    #[must_use]
    pub const fn with_reverse(mut self, reverse: bool) -> Self {
        self.reverse = reverse;
        self
    }

    fn calculate_wipe_value(&self, x: f32, y: f32, width: f32, height: f32, _progress: f32) -> f32 {
        let nx = x / width;
        let ny = y / height;
        let cx = 0.5;
        let cy = 0.5;

        let value = match self.pattern {
            WipePattern::LeftToRight => nx,
            WipePattern::RightToLeft => 1.0 - nx,
            WipePattern::TopToBottom => ny,
            WipePattern::BottomToTop => 1.0 - ny,
            WipePattern::DiagonalTLBR => (nx + ny) * 0.5,
            WipePattern::DiagonalTRBL => (1.0 - nx + ny) * 0.5,
            WipePattern::DiagonalBLTR => (nx + 1.0 - ny) * 0.5,
            WipePattern::DiagonalBRTL => (2.0 - nx - ny) * 0.5,
            WipePattern::CircleIn => {
                let dx = nx - cx;
                let dy = ny - cy;
                1.0 - (dx * dx + dy * dy).sqrt() * 1.414
            }
            WipePattern::CircleOut => {
                let dx = nx - cx;
                let dy = ny - cy;
                (dx * dx + dy * dy).sqrt() * 1.414
            }
            WipePattern::BarnDoorHorizontal => 1.0 - (nx - 0.5).abs() * 2.0,
            WipePattern::BarnDoorVertical => 1.0 - (ny - 0.5).abs() * 2.0,
            WipePattern::BoxIn => {
                let dx = (nx - cx).abs();
                let dy = (ny - cy).abs();
                1.0 - dx.max(dy) * 2.0
            }
            WipePattern::BoxOut => {
                let dx = (nx - cx).abs();
                let dy = (ny - cy).abs();
                dx.max(dy) * 2.0
            }
            WipePattern::ClockWipe => {
                let angle = ((nx - cx).atan2(ny - cy) / std::f32::consts::PI + 1.0) * 0.5;
                angle
            }
            WipePattern::Wedge => {
                let angle = ((nx - cx).atan2(ny - cy) / std::f32::consts::PI + 1.0) * 0.5;
                let dist = ((nx - cx).powi(2) + (ny - cy).powi(2)).sqrt();
                (angle + dist) * 0.5
            }
            WipePattern::IrisRound => {
                let dx = nx - cx;
                let dy = ny - cy;
                (dx * dx + dy * dy).sqrt() * 1.414
            }
            WipePattern::IrisDiamond => {
                let dx = (nx - cx).abs();
                let dy = (ny - cy).abs();
                (dx + dy) * 1.414
            }
            WipePattern::Checkerboard => {
                let grid_size = 8.0;
                let gx = (nx * grid_size) as i32;
                let gy = (ny * grid_size) as i32;
                if (gx + gy) % 2 == 0 {
                    nx
                } else {
                    1.0 - nx
                }
            }
            WipePattern::BlindsHorizontal => {
                let blinds = 10.0;
                let blind_idx = (ny * blinds).floor();
                let blind_pos = (ny * blinds) - blind_idx;
                blind_pos
            }
            WipePattern::BlindsVertical => {
                let blinds = 10.0;
                let blind_idx = (nx * blinds).floor();
                let blind_pos = (nx * blinds) - blind_idx;
                blind_pos
            }
            WipePattern::RandomBlocks => {
                let hash = ((nx * 12.9898 + ny * 78.233).sin() * 43_758.547).fract();
                hash
            }
            WipePattern::Radial => {
                let angle = (ny - cy).atan2(nx - cx);
                (angle / std::f32::consts::PI + 1.0) * 0.5
            }
            WipePattern::Spiral => {
                let angle = (ny - cy).atan2(nx - cx);
                let dist = ((nx - cx).powi(2) + (ny - cy).powi(2)).sqrt();
                ((angle / std::f32::consts::PI + 1.0) * 0.5 + dist) % 1.0
            }
            WipePattern::Heart => {
                // Parametric heart shape
                let dx = nx - cx;
                let dy = ny - cy + 0.1;
                let heart = (dx.powi(2) + dy.powi(2) - 0.1).powi(3) - dx.powi(2) * dy.powi(3);
                if heart < 0.0 {
                    ((dx.powi(2) + dy.powi(2)).sqrt() * 2.0).min(1.0)
                } else {
                    1.0
                }
            }
            WipePattern::Star => {
                let angle = (ny - cy).atan2(nx - cx);
                let points = 5.0;
                let star = ((angle * points).cos() * 0.2 + 0.8).abs();
                let dist = ((nx - cx).powi(2) + (ny - cy).powi(2)).sqrt();
                (dist / star).min(1.0)
            }
            WipePattern::Diamond => ((nx - cx).abs() + (ny - cy).abs()) * 1.414,
            WipePattern::Cross => {
                let horiz = (nx - cx).abs();
                let vert = (ny - cy).abs();
                horiz.min(vert) * 2.0
            }
            WipePattern::ArrowRight => {
                let tip = nx;
                let edge = (ny - cy).abs();
                (tip - edge).max(0.0)
            }
            WipePattern::ArrowLeft => {
                let tip = 1.0 - nx;
                let edge = (ny - cy).abs();
                (tip - edge).max(0.0)
            }
            WipePattern::WaveHorizontal => {
                let wave = ((nx * std::f32::consts::PI * 4.0).sin() * 0.1 + ny).clamp(0.0, 1.0);
                wave
            }
            WipePattern::WaveVertical => {
                let wave = ((ny * std::f32::consts::PI * 4.0).sin() * 0.1 + nx).clamp(0.0, 1.0);
                wave
            }
        };

        if self.reverse {
            1.0 - value
        } else {
            value
        }
    }

    fn blend_pixel(from: [u8; 4], to: [u8; 4], t: f32) -> [u8; 4] {
        let t = t.clamp(0.0, 1.0);
        let inv_t = 1.0 - t;

        [
            (f32::from(from[0]) * inv_t + f32::from(to[0]) * t) as u8,
            (f32::from(from[1]) * inv_t + f32::from(to[1]) * t) as u8,
            (f32::from(from[2]) * inv_t + f32::from(to[2]) * t) as u8,
            (f32::from(from[3]) * inv_t + f32::from(to[3]) * t) as u8,
        ]
    }
}

impl TransitionEffect for Wipe {
    fn name(&self) -> &'static str {
        "Wipe"
    }

    fn description(&self) -> &'static str {
        "Wipe transition with 30+ patterns"
    }

    fn apply(
        &mut self,
        from: &Frame,
        to: &Frame,
        output: &mut Frame,
        params: &EffectParams,
    ) -> VfxResult<()> {
        let width = output.width as f32;
        let height = output.height as f32;

        for y in 0..output.height {
            for x in 0..output.width {
                let wipe_val =
                    self.calculate_wipe_value(x as f32, y as f32, width, height, params.progress);

                // Apply feathering
                let t = if self.feather > 0.0 {
                    let edge = (wipe_val - params.progress) / self.feather;
                    edge.clamp(0.0, 1.0)
                } else if wipe_val < params.progress {
                    1.0
                } else {
                    0.0
                };

                let from_pixel = from.get_pixel(x, y).unwrap_or([0, 0, 0, 0]);
                let to_pixel = to.get_pixel(x, y).unwrap_or([0, 0, 0, 0]);
                let pixel = Self::blend_pixel(from_pixel, to_pixel, t);

                output.set_pixel(x, y, pixel);
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

    #[test]
    fn test_wipe_patterns() {
        let patterns = [
            WipePattern::LeftToRight,
            WipePattern::CircleIn,
            WipePattern::ClockWipe,
        ];

        for pattern in patterns {
            let mut wipe = Wipe::new(pattern);
            let from = Frame::new(100, 100).expect("should succeed in test");
            let to = Frame::new(100, 100).expect("should succeed in test");
            let mut output = Frame::new(100, 100).expect("should succeed in test");

            let params = EffectParams::new().with_progress(0.5);
            wipe.apply(&from, &to, &mut output, &params)
                .expect("should succeed in test");
        }
    }

    #[test]
    fn test_wipe_feather() {
        let wipe = Wipe::new(WipePattern::LeftToRight).with_feather(0.1);
        assert_eq!(wipe.feather, 0.1);
    }

    #[test]
    fn test_wipe_reverse() {
        let wipe = Wipe::new(WipePattern::LeftToRight).with_reverse(true);
        assert!(wipe.reverse);
    }
}
