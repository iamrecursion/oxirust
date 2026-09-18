//! Wipe transition effects for broadcast graphics.
//!
//! Implements directional wipe transitions commonly used in live production
//! and video editing: left-to-right, top-to-bottom, diagonal, radial, and
//! barn-door variants.

#![allow(dead_code)]

/// Direction of a linear wipe.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum WipeDirection {
    /// Wipe from left to right.
    LeftToRight,
    /// Wipe from right to left.
    RightToLeft,
    /// Wipe from top to bottom.
    TopToBottom,
    /// Wipe from bottom to top.
    BottomToTop,
    /// Diagonal wipe from top-left to bottom-right.
    DiagonalDown,
    /// Diagonal wipe from bottom-left to top-right.
    DiagonalUp,
    /// Radial wipe from centre outward.
    RadialOut,
    /// Barn-door: two edges meet at centre.
    BarnDoor,
}

impl WipeDirection {
    /// Human-readable label.
    #[must_use]
    pub fn label(&self) -> &'static str {
        match self {
            Self::LeftToRight => "Left to Right",
            Self::RightToLeft => "Right to Left",
            Self::TopToBottom => "Top to Bottom",
            Self::BottomToTop => "Bottom to Top",
            Self::DiagonalDown => "Diagonal Down",
            Self::DiagonalUp => "Diagonal Up",
            Self::RadialOut => "Radial Out",
            Self::BarnDoor => "Barn Door",
        }
    }

    /// Returns `true` for the horizontal variants.
    #[must_use]
    pub fn is_horizontal(&self) -> bool {
        matches!(self, Self::LeftToRight | Self::RightToLeft)
    }

    /// Returns `true` for the vertical variants.
    #[must_use]
    pub fn is_vertical(&self) -> bool {
        matches!(self, Self::TopToBottom | Self::BottomToTop)
    }
}

/// Easing function applied to the wipe progress.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WipeEasing {
    /// Linear (no easing).
    Linear,
    /// Ease-in (slow start).
    EaseIn,
    /// Ease-out (slow end).
    EaseOut,
    /// Ease-in-out (slow start and end).
    EaseInOut,
}

impl WipeEasing {
    /// Apply the easing curve to a normalised value `t` in \[0, 1\].
    #[must_use]
    pub fn apply(&self, t: f32) -> f32 {
        let t = t.clamp(0.0, 1.0);
        match self {
            Self::Linear => t,
            Self::EaseIn => t * t,
            Self::EaseOut => t * (2.0 - t),
            Self::EaseInOut => {
                if t < 0.5 {
                    2.0 * t * t
                } else {
                    -1.0 + (4.0 - 2.0 * t) * t
                }
            }
        }
    }
}

/// Configuration for a wipe transition.
#[derive(Debug, Clone)]
pub struct WipeConfig {
    /// Wipe direction.
    pub direction: WipeDirection,
    /// Duration in seconds.
    pub duration_s: f32,
    /// Easing curve.
    pub easing: WipeEasing,
    /// Softness of the wipe edge in normalised units \[0, 1\].
    pub softness: f32,
    /// Frame width.
    pub frame_width: u32,
    /// Frame height.
    pub frame_height: u32,
}

impl WipeConfig {
    /// Create a new [`WipeConfig`] with sensible defaults for a given direction.
    #[must_use]
    pub fn new(
        direction: WipeDirection,
        duration_s: f32,
        frame_width: u32,
        frame_height: u32,
    ) -> Self {
        Self {
            direction,
            duration_s: duration_s.max(0.001),
            easing: WipeEasing::Linear,
            softness: 0.02,
            frame_width,
            frame_height,
        }
    }

    /// Builder: set easing.
    #[must_use]
    pub fn with_easing(mut self, easing: WipeEasing) -> Self {
        self.easing = easing;
        self
    }

    /// Builder: set softness.
    #[must_use]
    pub fn with_softness(mut self, softness: f32) -> Self {
        self.softness = softness.clamp(0.0, 0.5);
        self
    }
}

/// Runtime state for a wipe transition.
#[derive(Debug, Clone)]
pub struct WipeTransition {
    config: WipeConfig,
    elapsed: f32,
}

impl WipeTransition {
    /// Create a new [`WipeTransition`] from the given config.
    #[must_use]
    pub fn new(config: WipeConfig) -> Self {
        Self {
            config,
            elapsed: 0.0,
        }
    }

    /// Advance the transition clock by `dt` seconds.
    pub fn advance(&mut self, dt: f32) {
        self.elapsed = (self.elapsed + dt).min(self.config.duration_s);
    }

    /// Reset the transition to the beginning.
    pub fn reset(&mut self) {
        self.elapsed = 0.0;
    }

    /// Raw progress \[0, 1\] (before easing).
    #[must_use]
    pub fn raw_progress(&self) -> f32 {
        (self.elapsed / self.config.duration_s).clamp(0.0, 1.0)
    }

    /// Eased progress \[0, 1\].
    #[must_use]
    pub fn progress_at(&self) -> f32 {
        self.config.easing.apply(self.raw_progress())
    }

    /// Returns `true` when the transition has completed.
    #[must_use]
    pub fn is_complete(&self) -> bool {
        self.elapsed >= self.config.duration_s
    }

    /// Compute the mix factor for a pixel at normalised coordinates `(nx, ny)`.
    ///
    /// Returns 0.0 when the pixel shows source A, 1.0 when it shows source B,
    /// and intermediate values within the soft edge.
    #[must_use]
    pub fn mix_at(&self, nx: f32, ny: f32) -> f32 {
        let p = self.progress_at();
        let edge = match self.config.direction {
            WipeDirection::LeftToRight => nx,
            WipeDirection::RightToLeft => 1.0 - nx,
            WipeDirection::TopToBottom => ny,
            WipeDirection::BottomToTop => 1.0 - ny,
            WipeDirection::DiagonalDown => (nx + ny) * 0.5,
            WipeDirection::DiagonalUp => ((1.0 - ny) + nx) * 0.5,
            WipeDirection::RadialOut => {
                let dx = nx - 0.5;
                let dy = ny - 0.5;
                (dx * dx + dy * dy).sqrt() * 2.0_f32.sqrt()
            }
            WipeDirection::BarnDoor => {
                let d = (nx - 0.5).abs() * 2.0;
                1.0 - d
            }
        };

        let half_soft = self.config.softness * 0.5;
        if half_soft < 1e-6 {
            if edge < p {
                1.0
            } else {
                0.0
            }
        } else {
            ((edge - (p - half_soft)) / (2.0 * half_soft)).clamp(0.0, 1.0)
        }
    }

    /// Reference to the underlying config.
    #[must_use]
    pub fn config(&self) -> &WipeConfig {
        &self.config
    }
}

// -- unit tests --

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_wipe_direction_labels() {
        assert_eq!(WipeDirection::LeftToRight.label(), "Left to Right");
        assert_eq!(WipeDirection::BarnDoor.label(), "Barn Door");
    }

    #[test]
    fn test_direction_horizontal() {
        assert!(WipeDirection::LeftToRight.is_horizontal());
        assert!(WipeDirection::RightToLeft.is_horizontal());
        assert!(!WipeDirection::TopToBottom.is_horizontal());
    }

    #[test]
    fn test_direction_vertical() {
        assert!(WipeDirection::TopToBottom.is_vertical());
        assert!(!WipeDirection::LeftToRight.is_vertical());
    }

    #[test]
    fn test_easing_linear() {
        assert!((WipeEasing::Linear.apply(0.5) - 0.5).abs() < 0.001);
    }

    #[test]
    fn test_easing_ease_in() {
        let v = WipeEasing::EaseIn.apply(0.5);
        assert!(v < 0.5); // ease-in is slower at the start
    }

    #[test]
    fn test_easing_clamped() {
        assert_eq!(WipeEasing::Linear.apply(-1.0), 0.0);
        assert_eq!(WipeEasing::Linear.apply(2.0), 1.0);
    }

    #[test]
    fn test_wipe_config_defaults() {
        let cfg = WipeConfig::new(WipeDirection::LeftToRight, 1.0, 1920, 1080);
        assert_eq!(cfg.direction, WipeDirection::LeftToRight);
        assert_eq!(cfg.easing, WipeEasing::Linear);
    }

    #[test]
    fn test_wipe_config_builder() {
        let cfg = WipeConfig::new(WipeDirection::TopToBottom, 2.0, 1920, 1080)
            .with_easing(WipeEasing::EaseInOut)
            .with_softness(0.1);
        assert_eq!(cfg.easing, WipeEasing::EaseInOut);
        assert!((cfg.softness - 0.1).abs() < 0.001);
    }

    #[test]
    fn test_transition_progress() {
        let cfg = WipeConfig::new(WipeDirection::LeftToRight, 1.0, 1920, 1080);
        let mut t = WipeTransition::new(cfg);
        assert_eq!(t.progress_at(), 0.0);
        t.advance(0.5);
        assert!((t.progress_at() - 0.5).abs() < 0.01);
        t.advance(0.5);
        assert!((t.progress_at() - 1.0).abs() < 0.01);
        assert!(t.is_complete());
    }

    #[test]
    fn test_transition_reset() {
        let cfg = WipeConfig::new(WipeDirection::LeftToRight, 1.0, 1920, 1080);
        let mut t = WipeTransition::new(cfg);
        t.advance(0.7);
        t.reset();
        assert_eq!(t.raw_progress(), 0.0);
    }

    #[test]
    fn test_mix_at_left_to_right() {
        let cfg = WipeConfig::new(WipeDirection::LeftToRight, 1.0, 1920, 1080).with_softness(0.0);
        let mut t = WipeTransition::new(cfg);
        t.advance(0.5); // progress = 0.5
                        // Left half (nx=0.25) should be fully revealed (1.0)
        assert!((t.mix_at(0.25, 0.5) - 1.0).abs() < 0.01);
        // Right half (nx=0.75) should still show source A (0.0)
        assert!((t.mix_at(0.75, 0.5) - 0.0).abs() < 0.01);
    }

    #[test]
    fn test_mix_at_with_softness() {
        let cfg = WipeConfig::new(WipeDirection::LeftToRight, 1.0, 1920, 1080).with_softness(0.2);
        let mut t = WipeTransition::new(cfg);
        t.advance(0.5);
        // At the wipe edge, value should be intermediate
        let v = t.mix_at(0.5, 0.5);
        assert!(v > 0.0 && v < 1.0);
    }

    #[test]
    fn test_duration_clamp() {
        let cfg = WipeConfig::new(WipeDirection::TopToBottom, 0.0, 1920, 1080);
        assert!(cfg.duration_s > 0.0);
    }

    #[test]
    fn test_softness_clamp() {
        let cfg = WipeConfig::new(WipeDirection::TopToBottom, 1.0, 1920, 1080).with_softness(10.0);
        assert!(cfg.softness <= 0.5);
    }

    #[test]
    fn test_config_accessor() {
        let cfg = WipeConfig::new(WipeDirection::RadialOut, 2.0, 1920, 1080);
        let t = WipeTransition::new(cfg);
        assert_eq!(t.config().direction, WipeDirection::RadialOut);
    }
}
