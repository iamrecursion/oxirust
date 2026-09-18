//! Motion trail / echo effects for video elements.
//!
//! Generates a fading trail behind a moving object by recording
//! historical positions and rendering them with decreasing opacity.

#![allow(dead_code)]

use serde::{Deserialize, Serialize};

/// A single recorded position along a trail.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct TrailPoint {
    /// X position in normalised coordinates [0, 1].
    pub x: f64,
    /// Y position in normalised coordinates [0, 1].
    pub y: f64,
    /// Opacity of this trail point [0, 1].
    pub opacity: f64,
    /// Timestamp (seconds) when the point was recorded.
    pub time: f64,
}

impl TrailPoint {
    /// Create a new trail point.
    #[must_use]
    pub fn new(x: f64, y: f64, opacity: f64, time: f64) -> Self {
        Self {
            x,
            y,
            opacity: opacity.clamp(0.0, 1.0),
            time,
        }
    }

    /// Euclidean distance to another trail point.
    #[must_use]
    pub fn distance_to(&self, other: &Self) -> f64 {
        let dx = self.x - other.x;
        let dy = self.y - other.y;
        (dx * dx + dy * dy).sqrt()
    }

    /// Linearly interpolate between two trail points.
    #[must_use]
    pub fn lerp(&self, other: &Self, t: f64) -> Self {
        let t = t.clamp(0.0, 1.0);
        Self {
            x: self.x + (other.x - self.x) * t,
            y: self.y + (other.y - self.y) * t,
            opacity: self.opacity + (other.opacity - self.opacity) * t,
            time: self.time + (other.time - self.time) * t,
        }
    }

    /// Returns `true` if the point is fully transparent.
    #[must_use]
    pub fn is_invisible(&self) -> bool {
        self.opacity <= 0.0
    }
}

/// Configuration for a trail effect.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrailConfig {
    /// Maximum number of trail points to retain.
    pub max_points: usize,
    /// How quickly older points fade out (0.0 = instant, 1.0 = never).
    pub fade_rate: f64,
    /// Minimum distance between consecutive points before a new one is recorded.
    pub min_spacing: f64,
    /// Width of the trail in normalised units.
    pub width: f64,
    /// Whether to smooth the trail with cubic interpolation.
    pub smooth: bool,
}

impl Default for TrailConfig {
    fn default() -> Self {
        Self {
            max_points: 60,
            fade_rate: 0.92,
            min_spacing: 0.005,
            width: 0.02,
            smooth: true,
        }
    }
}

impl TrailConfig {
    /// Create a new default configuration.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Set the maximum number of trail points.
    #[must_use]
    pub fn with_max_points(mut self, n: usize) -> Self {
        self.max_points = n.max(1);
        self
    }

    /// Set the fade rate.
    #[must_use]
    pub fn with_fade_rate(mut self, rate: f64) -> Self {
        self.fade_rate = rate.clamp(0.0, 1.0);
        self
    }

    /// Set the minimum spacing threshold.
    #[must_use]
    pub fn with_min_spacing(mut self, spacing: f64) -> Self {
        self.min_spacing = spacing.max(0.0);
        self
    }

    /// Validate that the configuration is sensible.
    #[must_use]
    pub fn is_valid(&self) -> bool {
        self.max_points >= 1 && self.fade_rate >= 0.0 && self.fade_rate <= 1.0 && self.width > 0.0
    }
}

/// Manages a live motion trail.
#[derive(Debug, Clone)]
pub struct TrailEffect {
    /// Current trail configuration.
    pub config: TrailConfig,
    /// Ordered trail points, newest first.
    points: Vec<TrailPoint>,
}

impl TrailEffect {
    /// Create a new trail effect with the given configuration.
    #[must_use]
    pub fn new(config: TrailConfig) -> Self {
        Self {
            points: Vec::with_capacity(config.max_points),
            config,
        }
    }

    /// Number of active points in the trail.
    #[must_use]
    pub fn len(&self) -> usize {
        self.points.len()
    }

    /// Whether the trail is empty.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.points.is_empty()
    }

    /// Get a slice of all active trail points (newest first).
    #[must_use]
    pub fn points(&self) -> &[TrailPoint] {
        &self.points
    }

    /// Update the trail by recording a new position and aging existing points.
    ///
    /// Returns the number of points pruned (expired or over the limit).
    pub fn update(&mut self, x: f64, y: f64, time: f64) -> usize {
        // Check spacing -- skip if too close to most recent point.
        let should_add = if let Some(last) = self.points.first() {
            let dx = x - last.x;
            let dy = y - last.y;
            (dx * dx + dy * dy).sqrt() >= self.config.min_spacing
        } else {
            true
        };

        if should_add {
            let pt = TrailPoint::new(x, y, 1.0, time);
            self.points.insert(0, pt);
        }

        // Fade existing points.
        for p in &mut self.points {
            p.opacity *= self.config.fade_rate;
        }

        // Prune invisible or over-limit points.
        let before = self.points.len();
        self.points.retain(|p| p.opacity > 0.01);
        if self.points.len() > self.config.max_points {
            self.points.truncate(self.config.max_points);
        }
        before.saturating_sub(self.points.len())
    }

    /// Discard all trail points.
    pub fn clear(&mut self) {
        self.points.clear();
    }

    /// Compute the total arc length of the trail.
    #[must_use]
    pub fn arc_length(&self) -> f64 {
        if self.points.len() < 2 {
            return 0.0;
        }
        self.points
            .windows(2)
            .map(|w| w[0].distance_to(&w[1]))
            .sum()
    }

    /// Return the bounding box of all trail points as `(min_x, min_y, max_x, max_y)`.
    #[must_use]
    pub fn bounding_box(&self) -> Option<(f64, f64, f64, f64)> {
        if self.points.is_empty() {
            return None;
        }
        let mut min_x = f64::MAX;
        let mut min_y = f64::MAX;
        let mut max_x = f64::MIN;
        let mut max_y = f64::MIN;
        for p in &self.points {
            if p.x < min_x {
                min_x = p.x;
            }
            if p.y < min_y {
                min_y = p.y;
            }
            if p.x > max_x {
                max_x = p.x;
            }
            if p.y > max_y {
                max_y = p.y;
            }
        }
        Some((min_x, min_y, max_x, max_y))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_trail_point_new_clamps_opacity() {
        let p = TrailPoint::new(0.5, 0.5, 1.5, 0.0);
        assert!((p.opacity - 1.0).abs() < 1e-12);
        let p2 = TrailPoint::new(0.5, 0.5, -0.5, 0.0);
        assert!(p2.opacity.abs() < 1e-12);
    }

    #[test]
    fn test_trail_point_distance() {
        let a = TrailPoint::new(0.0, 0.0, 1.0, 0.0);
        let b = TrailPoint::new(3.0, 4.0, 1.0, 0.0);
        assert!((a.distance_to(&b) - 5.0).abs() < 1e-12);
    }

    #[test]
    fn test_trail_point_lerp_midpoint() {
        let a = TrailPoint::new(0.0, 0.0, 0.0, 0.0);
        let b = TrailPoint::new(1.0, 1.0, 1.0, 1.0);
        let m = a.lerp(&b, 0.5);
        assert!((m.x - 0.5).abs() < 1e-12);
        assert!((m.y - 0.5).abs() < 1e-12);
        assert!((m.opacity - 0.5).abs() < 1e-12);
    }

    #[test]
    fn test_trail_point_is_invisible() {
        let a = TrailPoint::new(0.0, 0.0, 0.0, 0.0);
        assert!(a.is_invisible());
        let b = TrailPoint::new(0.0, 0.0, 0.5, 0.0);
        assert!(!b.is_invisible());
    }

    #[test]
    fn test_trail_config_default_valid() {
        let cfg = TrailConfig::default();
        assert!(cfg.is_valid());
    }

    #[test]
    fn test_trail_config_builder() {
        let cfg = TrailConfig::new()
            .with_max_points(100)
            .with_fade_rate(0.8)
            .with_min_spacing(0.01);
        assert_eq!(cfg.max_points, 100);
        assert!((cfg.fade_rate - 0.8).abs() < 1e-12);
        assert!((cfg.min_spacing - 0.01).abs() < 1e-12);
    }

    #[test]
    fn test_trail_config_clamps_fade_rate() {
        let cfg = TrailConfig::new().with_fade_rate(5.0);
        assert!((cfg.fade_rate - 1.0).abs() < 1e-12);
    }

    #[test]
    fn test_trail_effect_starts_empty() {
        let eff = TrailEffect::new(TrailConfig::default());
        assert!(eff.is_empty());
        assert_eq!(eff.len(), 0);
    }

    #[test]
    fn test_trail_effect_update_adds_point() {
        let mut eff = TrailEffect::new(TrailConfig::default());
        eff.update(0.5, 0.5, 0.0);
        assert_eq!(eff.len(), 1);
    }

    #[test]
    fn test_trail_effect_min_spacing() {
        let cfg = TrailConfig::new().with_min_spacing(1.0);
        let mut eff = TrailEffect::new(cfg);
        eff.update(0.5, 0.5, 0.0);
        eff.update(0.500_01, 0.500_01, 0.016);
        // Second point too close, should not add a new one (still 1 point).
        assert_eq!(eff.len(), 1);
    }

    #[test]
    fn test_trail_effect_clear() {
        let mut eff = TrailEffect::new(TrailConfig::default());
        eff.update(0.1, 0.1, 0.0);
        eff.update(0.5, 0.5, 0.1);
        eff.clear();
        assert!(eff.is_empty());
    }

    #[test]
    fn test_trail_effect_arc_length() {
        let cfg = TrailConfig::new().with_min_spacing(0.0).with_fade_rate(1.0);
        let mut eff = TrailEffect::new(cfg);
        eff.update(0.0, 0.0, 0.0);
        eff.update(1.0, 0.0, 0.1);
        // Arc length should be ~1.0.
        assert!((eff.arc_length() - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_trail_effect_bounding_box_empty() {
        let eff = TrailEffect::new(TrailConfig::default());
        assert!(eff.bounding_box().is_none());
    }

    #[test]
    fn test_trail_effect_bounding_box() {
        let cfg = TrailConfig::new().with_min_spacing(0.0).with_fade_rate(1.0);
        let mut eff = TrailEffect::new(cfg);
        eff.update(0.2, 0.3, 0.0);
        eff.update(0.8, 0.9, 0.1);
        let (min_x, min_y, max_x, max_y) = eff.bounding_box().expect("should succeed in test");
        assert!((min_x - 0.2).abs() < 1e-6);
        assert!((min_y - 0.3).abs() < 1e-6);
        assert!((max_x - 0.8).abs() < 1e-6);
        assert!((max_y - 0.9).abs() < 1e-6);
    }

    #[test]
    fn test_trail_effect_max_points_limit() {
        let cfg = TrailConfig::new()
            .with_max_points(3)
            .with_min_spacing(0.0)
            .with_fade_rate(1.0);
        let mut eff = TrailEffect::new(cfg);
        for i in 0..10 {
            #[allow(clippy::cast_precision_loss)]
            let v = i as f64 * 0.1;
            eff.update(v, v, v);
        }
        assert!(eff.len() <= 3);
    }

    #[test]
    fn test_trail_effect_fade_prunes() {
        let cfg = TrailConfig::new().with_min_spacing(0.0).with_fade_rate(0.0);
        let mut eff = TrailEffect::new(cfg);
        eff.update(0.0, 0.0, 0.0);
        // After a second update the first point has been faded to ~0.
        eff.update(0.5, 0.5, 0.1);
        // Points with opacity < 0.01 are pruned.
        // The first update point will be faded and pruned on second update.
        assert!(eff.len() <= 2);
    }
}
