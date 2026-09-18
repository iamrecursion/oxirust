//! DAW-style automation lanes with sample-accurate timing and multiple curve types.
//!
//! This module provides a self-contained automation lane implementation that uses
//! sample-based timestamps (`u64`) rather than seconds.  It is distinct from the
//! `automation` module, which offers a richer but more coupled data model.
//!
//! # Overview
//!
//! An `AutomationLane` contains a time-ordered sequence of `AutomationPoint`s.
//! Each point carries a `time_samples` position, a `value`, and a [`CurveType`]
//! that controls how the value transitions *from that point to the next*.
//!
//! ```
//! use oximedia_mixer::daw_automation::{AutomationLane, AutomationParam, AutomationPoint, CurveType};
//!
//! let mut lane = AutomationLane::new(AutomationParam::Volume, 1.0);
//! lane.add_point(AutomationPoint { time_samples: 0,     value: 0.0, curve: CurveType::Linear });
//! lane.add_point(AutomationPoint { time_samples: 48000, value: 1.0, curve: CurveType::Linear });
//! assert!((lane.get_value_at(24000) - 0.5).abs() < 1e-5);
//! ```

#![allow(clippy::cast_precision_loss)]

// ---------------------------------------------------------------------------
// AutomationParam
// ---------------------------------------------------------------------------

/// Identifies the mixer parameter being automated by a lane.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum AutomationParam {
    /// Channel volume (linear gain, 0.0 – 2.0).
    Volume,
    /// Stereo pan position (-1.0 left … +1.0 right).
    Pan,
    /// Aux send level for the given send-slot index.
    Send(usize),
    /// Effect parameter identified by slot index and parameter index.
    EffectParam {
        /// Effect insert slot (0-based).
        slot: usize,
        /// Parameter index within the effect.
        param_idx: usize,
    },
    /// Boolean mute (0.0 = unmuted, 1.0 = muted; Step curve recommended).
    Mute,
    /// Master bus gain.
    MasterGain,
    /// Custom parameter identified by a string tag.
    Custom(String),
}

// ---------------------------------------------------------------------------
// CurveType
// ---------------------------------------------------------------------------

/// The interpolation shape used to transition from one [`AutomationPoint`] to the next.
///
/// The curve is defined in terms of a normalised time parameter `t ∈ [0, 1]`
/// where `t = 0` is the *start* point and `t = 1` is the *end* point.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CurveType {
    /// Straight line between the two values: `v = v0 + (v1 - v0) * t`.
    Linear,
    /// Ease-in exponential: `v = v0 + (v1 - v0) * t²`.
    Exponential,
    /// Ease-out logarithmic: `v = v0 + (v1 - v0) * sqrt(t)`.
    Logarithmic,
    /// Hold the start value until the next point: no interpolation.
    Step,
}

// ---------------------------------------------------------------------------
// AutomationPoint
// ---------------------------------------------------------------------------

/// A single automation breakpoint.
#[derive(Debug, Clone, PartialEq)]
pub struct AutomationPoint {
    /// Position in the timeline, measured in audio samples.
    pub time_samples: u64,
    /// Parameter value at this position.
    pub value: f32,
    /// Curve type applied from this point *to the next* point in the lane.
    pub curve: CurveType,
}

impl AutomationPoint {
    /// Create a new point with [`CurveType::Linear`].
    #[must_use]
    pub fn linear(time_samples: u64, value: f32) -> Self {
        Self {
            time_samples,
            value,
            curve: CurveType::Linear,
        }
    }

    /// Create a new point with an explicit curve type.
    #[must_use]
    pub fn with_curve(time_samples: u64, value: f32, curve: CurveType) -> Self {
        Self {
            time_samples,
            value,
            curve,
        }
    }

    /// Interpolate from `self` to `next` at `query_samples` using `self.curve`.
    ///
    /// `query_samples` must be in `[self.time_samples, next.time_samples]`.
    /// Values outside that range are clamped to the nearest endpoint.
    #[must_use]
    pub fn interpolate_to(&self, next: &Self, query_samples: u64) -> f32 {
        if next.time_samples <= self.time_samples {
            return self.value;
        }
        let span = (next.time_samples - self.time_samples) as f64;
        let offset = query_samples.saturating_sub(self.time_samples) as f64;
        let t = (offset / span).clamp(0.0, 1.0) as f32;

        let shaped = match self.curve {
            CurveType::Linear => t,
            CurveType::Exponential => t * t,
            CurveType::Logarithmic => t.sqrt(),
            CurveType::Step => 0.0, // hold self.value until next point
        };

        self.value + (next.value - self.value) * shaped
    }
}

// ---------------------------------------------------------------------------
// AutomationLane
// ---------------------------------------------------------------------------

/// A DAW-style automation lane: an ordered collection of [`AutomationPoint`]s
/// that can be queried for a parameter value at any sample position.
#[derive(Debug, Clone)]
pub struct AutomationLane {
    /// The parameter this lane controls.
    pub parameter: AutomationParam,
    /// Breakpoints, always kept sorted by `time_samples` ascending.
    pub points: Vec<AutomationPoint>,
    /// Value returned when the lane is empty or queried outside all points.
    pub default_value: f32,
    /// Whether the lane is active; an inactive lane always returns `default_value`.
    pub enabled: bool,
}

impl AutomationLane {
    /// Create an empty lane for `parameter` with the given `default_value`.
    #[must_use]
    pub fn new(parameter: AutomationParam, default_value: f32) -> Self {
        Self {
            parameter,
            points: Vec::new(),
            default_value,
            enabled: true,
        }
    }

    /// Insert `point` into the lane, keeping the internal list sorted by
    /// `time_samples`.  If a point already exists at the same `time_samples`
    /// it is replaced.
    pub fn add_point(&mut self, point: AutomationPoint) {
        // Remove any existing point at the same time.
        self.points.retain(|p| p.time_samples != point.time_samples);
        // Insert at sorted position.
        let pos = self
            .points
            .partition_point(|p| p.time_samples < point.time_samples);
        self.points.insert(pos, point);
    }

    /// Remove the point at exactly `time_samples`.
    ///
    /// Returns `true` if a point was found and removed.
    pub fn remove_at(&mut self, time_samples: u64) -> bool {
        let len_before = self.points.len();
        self.points.retain(|p| p.time_samples != time_samples);
        self.points.len() < len_before
    }

    /// Remove all breakpoints.
    pub fn clear(&mut self) {
        self.points.clear();
    }

    /// Evaluate the parameter value at `time_samples`.
    ///
    /// - If the lane is disabled: returns `default_value`.
    /// - If the lane has no points: returns `default_value`.
    /// - If `time_samples` is before the first point: returns the first point's value.
    /// - If `time_samples` is at or after the last point: returns the last point's value.
    /// - Otherwise: interpolates between the surrounding points using the
    ///   preceding point's [`CurveType`].
    #[must_use]
    pub fn get_value_at(&self, time_samples: u64) -> f32 {
        if !self.enabled {
            return self.default_value;
        }
        match self.points.len() {
            0 => self.default_value,
            1 => self.points[0].value,
            _ => {
                let first = &self.points[0];
                if time_samples <= first.time_samples {
                    return first.value;
                }
                let last = match self.points.last() {
                    Some(p) => p,
                    None => return self.default_value,
                };
                if time_samples >= last.time_samples {
                    return last.value;
                }
                // Binary-search for the segment containing `time_samples`.
                let idx = self
                    .points
                    .partition_point(|p| p.time_samples <= time_samples);
                // `idx` is the index of the first point strictly AFTER query.
                // The preceding point is at `idx - 1`.
                let lo = &self.points[idx - 1];
                let hi = &self.points[idx];
                lo.interpolate_to(hi, time_samples)
            }
        }
    }

    /// Number of breakpoints in the lane.
    #[must_use]
    pub fn len(&self) -> usize {
        self.points.len()
    }

    /// Returns `true` if the lane contains no breakpoints.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.points.is_empty()
    }

    /// Returns the sample range `(first_sample, last_sample)` covered by this lane,
    /// or `None` if the lane is empty.
    #[must_use]
    pub fn time_span(&self) -> Option<(u64, u64)> {
        let first = self.points.first()?;
        let last = self.points.last()?;
        Some((first.time_samples, last.time_samples))
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // Helper: build a lane with Linear curve points at given (time, value) pairs.
    fn lane_with_points(pairs: &[(u64, f32)]) -> AutomationLane {
        let mut lane = AutomationLane::new(AutomationParam::Volume, 0.0);
        for &(t, v) in pairs {
            lane.add_point(AutomationPoint::linear(t, v));
        }
        lane
    }

    // 1. Empty lane returns default value.
    #[test]
    fn test_empty_lane_returns_default() {
        let lane = AutomationLane::new(AutomationParam::Volume, 0.75);
        assert!((lane.get_value_at(0) - 0.75).abs() < 1e-6);
        assert!((lane.get_value_at(99999) - 0.75).abs() < 1e-6);
    }

    // 2. Single point returns that value everywhere.
    #[test]
    fn test_single_point_returns_everywhere() {
        let mut lane = AutomationLane::new(AutomationParam::Pan, 0.0);
        lane.add_point(AutomationPoint::linear(1000, 0.4));
        assert!((lane.get_value_at(0) - 0.4).abs() < 1e-6);
        assert!((lane.get_value_at(1000) - 0.4).abs() < 1e-6);
        assert!((lane.get_value_at(2000) - 0.4).abs() < 1e-6);
    }

    // 3. Linear interpolation at midpoint.
    #[test]
    fn test_linear_interpolation_midpoint() {
        let lane = lane_with_points(&[(0, 0.0), (48000, 1.0)]);
        let v = lane.get_value_at(24000);
        assert!((v - 0.5).abs() < 1e-5, "expected 0.5, got {v}");
    }

    // 4. Linear at start clamps to first value.
    #[test]
    fn test_linear_before_first_point() {
        let lane = lane_with_points(&[(1000, 0.2), (2000, 0.8)]);
        assert!((lane.get_value_at(0) - 0.2).abs() < 1e-6);
    }

    // 5. Linear at or after last point clamps to last value.
    #[test]
    fn test_linear_after_last_point() {
        let lane = lane_with_points(&[(0, 0.1), (1000, 0.9)]);
        assert!((lane.get_value_at(1000) - 0.9).abs() < 1e-6);
        assert!((lane.get_value_at(99999) - 0.9).abs() < 1e-6);
    }

    // 6. Step curve holds the preceding value (no interpolation).
    #[test]
    fn test_step_curve_holds_value() {
        let mut lane = AutomationLane::new(AutomationParam::Mute, 0.0);
        lane.add_point(AutomationPoint::with_curve(0, 0.0, CurveType::Step));
        lane.add_point(AutomationPoint::with_curve(1000, 1.0, CurveType::Step));
        // Between 0 and 1000 the Step curve holds 0.0.
        let mid = lane.get_value_at(500);
        assert!(
            (mid - 0.0).abs() < 1e-6,
            "Step curve should hold 0.0, got {mid}"
        );
        // At and after 1000 we get 1.0.
        assert!((lane.get_value_at(1000) - 1.0).abs() < 1e-6);
    }

    // 7. Exponential curve: midpoint value is less than linear midpoint (ease-in).
    #[test]
    fn test_exponential_curve_ease_in() {
        let mut lane = AutomationLane::new(AutomationParam::Volume, 0.0);
        lane.add_point(AutomationPoint::with_curve(0, 0.0, CurveType::Exponential));
        lane.add_point(AutomationPoint::linear(100, 1.0));
        let v = lane.get_value_at(50);
        // t=0.5, Exponential: t² = 0.25  (less than linear 0.5)
        assert!(
            v < 0.5,
            "Exponential mid should be < 0.5 (ease-in), got {v}"
        );
        assert!((v - 0.25).abs() < 1e-5, "expected 0.25, got {v}");
    }

    // 8. Logarithmic curve: midpoint value is more than linear midpoint (ease-out).
    #[test]
    fn test_logarithmic_curve_ease_out() {
        let mut lane = AutomationLane::new(AutomationParam::Volume, 0.0);
        lane.add_point(AutomationPoint::with_curve(0, 0.0, CurveType::Logarithmic));
        lane.add_point(AutomationPoint::linear(100, 1.0));
        let v = lane.get_value_at(50);
        // t=0.5, Logarithmic: sqrt(0.5) ≈ 0.707 (more than linear 0.5)
        assert!(
            v > 0.5,
            "Logarithmic mid should be > 0.5 (ease-out), got {v}"
        );
        let expected = 0.5_f32.sqrt();
        assert!((v - expected).abs() < 1e-5, "expected {expected}, got {v}");
    }

    // 9. add_point keeps sorted order regardless of insertion order.
    #[test]
    fn test_add_point_maintains_sorted_order() {
        let mut lane = AutomationLane::new(AutomationParam::Volume, 0.0);
        lane.add_point(AutomationPoint::linear(3000, 0.9));
        lane.add_point(AutomationPoint::linear(1000, 0.1));
        lane.add_point(AutomationPoint::linear(2000, 0.5));
        assert_eq!(lane.len(), 3);
        assert!(lane.points[0].time_samples < lane.points[1].time_samples);
        assert!(lane.points[1].time_samples < lane.points[2].time_samples);
    }

    // 10. add_point replaces an existing point at the same time_samples.
    #[test]
    fn test_add_point_replaces_duplicate() {
        let mut lane = AutomationLane::new(AutomationParam::Volume, 0.0);
        lane.add_point(AutomationPoint::linear(1000, 0.3));
        lane.add_point(AutomationPoint::linear(1000, 0.7)); // replaces
        assert_eq!(lane.len(), 1);
        assert!((lane.get_value_at(1000) - 0.7).abs() < 1e-6);
    }

    // 11. remove_at returns true when a point is found and removed.
    #[test]
    fn test_remove_at_existing() {
        let mut lane = lane_with_points(&[(0, 0.0), (1000, 1.0)]);
        let removed = lane.remove_at(1000);
        assert!(removed);
        assert_eq!(lane.len(), 1);
    }

    // 12. remove_at returns false when no point at that time.
    #[test]
    fn test_remove_at_nonexistent() {
        let mut lane = lane_with_points(&[(0, 0.0)]);
        let removed = lane.remove_at(9999);
        assert!(!removed);
        assert_eq!(lane.len(), 1);
    }

    // 13. clear empties the lane.
    #[test]
    fn test_clear_empties_lane() {
        let mut lane = lane_with_points(&[(0, 0.0), (1000, 1.0), (2000, 0.5)]);
        lane.clear();
        assert!(lane.is_empty());
        assert_eq!(lane.len(), 0);
    }

    // 14. time_span returns None for empty lane.
    #[test]
    fn test_time_span_empty() {
        let lane = AutomationLane::new(AutomationParam::Volume, 0.0);
        assert!(lane.time_span().is_none());
    }

    // 15. time_span returns correct (first, last) for non-empty lane.
    #[test]
    fn test_time_span_nonempty() {
        let lane = lane_with_points(&[(500, 0.1), (1000, 0.5), (3000, 0.9)]);
        let (start, end) = lane.time_span().expect("should have span");
        assert_eq!(start, 500);
        assert_eq!(end, 3000);
    }

    // 16. is_empty / len work correctly.
    #[test]
    fn test_is_empty_and_len() {
        let mut lane = AutomationLane::new(AutomationParam::Pan, 0.0);
        assert!(lane.is_empty());
        assert_eq!(lane.len(), 0);
        lane.add_point(AutomationPoint::linear(0, 0.0));
        assert!(!lane.is_empty());
        assert_eq!(lane.len(), 1);
    }

    // 17. Multiple segments with different curve types in the same lane.
    #[test]
    fn test_mixed_curve_types_in_one_lane() {
        let mut lane = AutomationLane::new(AutomationParam::Volume, 0.0);
        // Segment 1: 0→100 Linear
        lane.add_point(AutomationPoint::with_curve(0, 0.0, CurveType::Linear));
        // Segment 2: 100→200 Step
        lane.add_point(AutomationPoint::with_curve(100, 1.0, CurveType::Step));
        // Segment 3: end marker
        lane.add_point(AutomationPoint::linear(200, 0.5));

        // Midpoint of linear segment: t=0.5 → 0.5
        let v_linear = lane.get_value_at(50);
        assert!((v_linear - 0.5).abs() < 1e-5, "linear mid={v_linear}");

        // Midpoint of step segment: holds 1.0
        let v_step = lane.get_value_at(150);
        assert!((v_step - 1.0).abs() < 1e-6, "step mid={v_step}");
    }

    // 18. AutomationParam variants are distinct (equality check).
    #[test]
    fn test_automation_param_variants_distinct() {
        assert_ne!(AutomationParam::Volume, AutomationParam::Pan);
        assert_ne!(AutomationParam::Send(0), AutomationParam::Send(1));
        assert_eq!(AutomationParam::Send(2), AutomationParam::Send(2));
        assert_ne!(
            AutomationParam::EffectParam {
                slot: 0,
                param_idx: 0
            },
            AutomationParam::EffectParam {
                slot: 0,
                param_idx: 1
            }
        );
    }

    // 19. Disabled lane returns default_value regardless of points.
    #[test]
    fn test_disabled_lane_returns_default() {
        let mut lane = lane_with_points(&[(0, 0.0), (1000, 1.0)]);
        lane.enabled = false;
        lane.default_value = 0.42;
        assert!((lane.get_value_at(500) - 0.42).abs() < 1e-6);
    }

    // 20. Interpolate_to at exact start and end of segment.
    #[test]
    fn test_interpolate_to_at_endpoints() {
        let p0 = AutomationPoint::linear(0, 0.1);
        let p1 = AutomationPoint::linear(100, 0.9);
        let at_start = p0.interpolate_to(&p1, 0);
        let at_end = p0.interpolate_to(&p1, 100);
        assert!((at_start - 0.1).abs() < 1e-6);
        assert!((at_end - 0.9).abs() < 1e-6);
    }
}
