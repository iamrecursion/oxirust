//! Keyframe animation system for timeline parameters.
//!
//! This module provides easing-based keyframe animation with tracks and curves.

#![allow(dead_code)]
#![allow(clippy::cast_precision_loss)]

/// Easing function type for keyframe interpolation.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum EasingType {
    /// Constant speed interpolation.
    Linear,
    /// Slow start, fast end.
    EaseIn,
    /// Fast start, slow end.
    EaseOut,
    /// Slow start and end.
    EaseInOut,
    /// Bouncing effect at the end.
    Bounce,
    /// Spring oscillation effect.
    Spring,
    /// Cubic bezier curve with two control points.
    ///
    /// The control points `(x1, y1)` and `(x2, y2)` define the shape of the
    /// easing curve, following the CSS `cubic-bezier(x1, y1, x2, y2)` convention.
    /// `x1` and `x2` must be in `[0, 1]`. `y1` and `y2` can exceed `[0, 1]`
    /// to create overshoot effects.
    CubicBezier {
        /// X coordinate of the first control point (0.0-1.0).
        x1: f32,
        /// Y coordinate of the first control point.
        y1: f32,
        /// X coordinate of the second control point (0.0-1.0).
        x2: f32,
        /// Y coordinate of the second control point.
        y2: f32,
    },
}

impl EasingType {
    /// Evaluates the easing function at `t` (0.0–1.0), returning a remapped value.
    #[must_use]
    pub fn evaluate(self, t: f32) -> f32 {
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
            Self::Bounce => {
                // Simple bounce: decelerates and bounces back once
                if t < 0.727_272_7 {
                    7.5625 * t * t
                } else if t < 0.909_090_9 {
                    let t2 = t - 0.818_181_8;
                    7.5625 * t2 * t2 + 0.75
                } else if t < 0.969_696_9 {
                    let t2 = t - 0.939_393_9;
                    7.5625 * t2 * t2 + 0.9375
                } else {
                    let t2 = t - 0.984_848_4;
                    7.5625 * t2 * t2 + 0.984_375
                }
            }
            Self::Spring => {
                // Damped spring approximation
                let freq = 2.0 * std::f32::consts::PI;
                1.0 - ((-6.0 * t).exp() * (freq * t).cos())
            }
            Self::CubicBezier { x1, y1, x2, y2 } => cubic_bezier_evaluate(t, x1, y1, x2, y2),
        }
    }
}

/// Evaluates a cubic bezier curve at parameter `t`.
///
/// The curve is defined by four points:
/// - P0 = (0, 0) (implicit start)
/// - P1 = (x1, y1) (first control point)
/// - P2 = (x2, y2) (second control point)
/// - P3 = (1, 1) (implicit end)
///
/// Given an input `t` (time, 0-1), we need to find the parameter `u` such
/// that `bezier_x(u) = t`, then return `bezier_y(u)`.
///
/// This uses Newton's method to solve for `u`, falling back to bisection
/// for robustness.
fn cubic_bezier_evaluate(t: f32, x1: f32, y1: f32, x2: f32, y2: f32) -> f32 {
    if t <= 0.0 {
        return 0.0;
    }
    if t >= 1.0 {
        return 1.0;
    }

    // Find u such that bezier_x(u) = t using Newton's method
    let u = solve_bezier_t(t, x1, x2);

    // Evaluate bezier_y(u)
    bezier_component(u, y1, y2)
}

/// Computes one component (x or y) of a cubic bezier at parameter `u`.
///
/// B(u) = 3*(1-u)^2*u*c1 + 3*(1-u)*u^2*c2 + u^3
///
/// where c1 and c2 are the corresponding component of the control points.
fn bezier_component(u: f32, c1: f32, c2: f32) -> f32 {
    let u2 = u * u;
    let u3 = u2 * u;
    let inv = 1.0 - u;
    let inv2 = inv * inv;

    3.0 * inv2 * u * c1 + 3.0 * inv * u2 * c2 + u3
}

/// Computes the derivative of one bezier component with respect to `u`.
///
/// B'(u) = 3*(1-u)^2*c1 + 6*(1-u)*u*(c2-c1) + 3*u^2*(1-c2)
fn bezier_component_derivative(u: f32, c1: f32, c2: f32) -> f32 {
    let inv = 1.0 - u;
    3.0 * inv * inv * c1 + 6.0 * inv * u * (c2 - c1) + 3.0 * u * u * (1.0 - c2)
}

/// Solves for the bezier parameter `u` such that `bezier_x(u) = target_x`.
///
/// Uses Newton's method with bisection fallback for robustness.
fn solve_bezier_t(target_x: f32, x1: f32, x2: f32) -> f32 {
    const EPSILON: f32 = 1e-6;
    const MAX_ITERATIONS: u32 = 8;

    // Initial guess: t itself is a reasonable starting point
    let mut u = target_x;

    // Newton's method
    for _ in 0..MAX_ITERATIONS {
        let x = bezier_component(u, x1, x2) - target_x;
        if x.abs() < EPSILON {
            return u;
        }
        let dx = bezier_component_derivative(u, x1, x2);
        if dx.abs() < EPSILON {
            break; // Derivative too small, fall back to bisection
        }
        u -= x / dx;
        u = u.clamp(0.0, 1.0);
    }

    // Bisection fallback for robustness
    let mut lo = 0.0_f32;
    let mut hi = 1.0_f32;

    for _ in 0..20 {
        let mid = (lo + hi) * 0.5;
        let x = bezier_component(mid, x1, x2);
        if (x - target_x).abs() < EPSILON {
            return mid;
        }
        if x < target_x {
            lo = mid;
        } else {
            hi = mid;
        }
    }

    (lo + hi) * 0.5
}

/// A single keyframe holding a value at a specific frame position.
#[derive(Debug, Clone)]
pub struct Keyframe {
    /// Frame number of this keyframe on the timeline.
    pub time_frames: u64,
    /// The animated value at this keyframe.
    pub value: f64,
    /// Easing applied when interpolating *from* this keyframe to the next.
    pub easing: EasingType,
}

impl Keyframe {
    /// Creates a new keyframe.
    #[must_use]
    pub fn new(time_frames: u64, value: f64, easing: EasingType) -> Self {
        Self {
            time_frames,
            value,
            easing,
        }
    }

    /// Returns `true` if this keyframe's time is strictly before `other_frames`.
    #[must_use]
    pub fn is_before(&self, other_frames: u64) -> bool {
        self.time_frames < other_frames
    }
}

/// A named track containing an ordered sequence of keyframes.
#[derive(Debug, Clone)]
pub struct KeyframeTrack {
    /// Keyframes in this track (sorted by `time_frames` after calling `sort()`).
    pub keyframes: Vec<Keyframe>,
    /// Human-readable name for the parameter being animated.
    pub name: String,
}

impl KeyframeTrack {
    /// Creates a new empty keyframe track.
    #[must_use]
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            keyframes: Vec::new(),
            name: name.into(),
        }
    }

    /// Appends a keyframe to the track.
    pub fn add(&mut self, keyframe: Keyframe) {
        self.keyframes.push(keyframe);
    }

    /// Sorts keyframes by ascending frame number.
    pub fn sort(&mut self) {
        self.keyframes.sort_by_key(|k| k.time_frames);
    }

    /// Interpolates the animated value at `time_frames`.
    ///
    /// - Before first keyframe: returns first keyframe value.
    /// - After last keyframe: returns last keyframe value.
    /// - Between two keyframes: linearly interpolates with easing from the left keyframe.
    #[must_use]
    pub fn interpolate(&self, time_frames: u64) -> f64 {
        if self.keyframes.is_empty() {
            return 0.0;
        }
        if time_frames <= self.keyframes[0].time_frames {
            return self.keyframes[0].value;
        }
        // SAFETY: we checked `self.keyframes.is_empty()` above, so `last()` is Some
        let Some(last) = self.keyframes.last() else {
            return 0.0;
        };
        if time_frames >= last.time_frames {
            return last.value;
        }
        // Find bracket
        for i in 0..self.keyframes.len() - 1 {
            let a = &self.keyframes[i];
            let b = &self.keyframes[i + 1];
            if time_frames >= a.time_frames && time_frames < b.time_frames {
                let range = (b.time_frames - a.time_frames) as f32;
                let offset = (time_frames - a.time_frames) as f32;
                let t_raw = offset / range;
                let t_eased = f64::from(a.easing.evaluate(t_raw));
                return a.value + (b.value - a.value) * t_eased;
            }
        }
        last.value
    }

    /// Returns the frame number of the last keyframe, or `0` if the track is empty.
    #[must_use]
    pub fn duration_frames(&self) -> u64 {
        self.keyframes.last().map_or(0, |k| k.time_frames)
    }

    /// Returns the number of keyframes in this track.
    #[must_use]
    pub fn keyframe_count(&self) -> usize {
        self.keyframes.len()
    }
}

/// A collection of named `KeyframeTrack`s forming a full animation curve.
#[derive(Debug, Clone, Default)]
pub struct AnimationCurve {
    /// Tracks indexed by name order.
    pub tracks: Vec<KeyframeTrack>,
}

impl AnimationCurve {
    /// Creates a new empty animation curve.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Appends a track to the curve.
    pub fn add_track(&mut self, track: KeyframeTrack) {
        self.tracks.push(track);
    }

    /// Returns a reference to the track with the given name, if it exists.
    #[must_use]
    pub fn get_track(&self, name: &str) -> Option<&KeyframeTrack> {
        self.tracks.iter().find(|t| t.name == name)
    }

    /// Returns the number of tracks in this curve.
    #[must_use]
    pub fn track_count(&self) -> usize {
        self.tracks.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // --- EasingType tests ---

    #[test]
    fn test_easing_linear_endpoints() {
        assert!((EasingType::Linear.evaluate(0.0) - 0.0).abs() < 1e-6);
        assert!((EasingType::Linear.evaluate(1.0) - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_easing_linear_midpoint() {
        assert!((EasingType::Linear.evaluate(0.5) - 0.5).abs() < 1e-6);
    }

    #[test]
    fn test_easing_ease_in_midpoint() {
        // t*t at 0.5 => 0.25
        assert!((EasingType::EaseIn.evaluate(0.5) - 0.25).abs() < 1e-6);
    }

    #[test]
    fn test_easing_ease_out_midpoint() {
        // 0.5*(2-0.5) = 0.75
        assert!((EasingType::EaseOut.evaluate(0.5) - 0.75).abs() < 1e-6);
    }

    #[test]
    fn test_easing_ease_in_out_midpoint() {
        // 2*0.5*0.5 = 0.5
        assert!((EasingType::EaseInOut.evaluate(0.5) - 0.5).abs() < 1e-6);
    }

    #[test]
    fn test_easing_clamp_below_zero() {
        assert!((EasingType::Linear.evaluate(-1.0) - 0.0).abs() < 1e-6);
    }

    #[test]
    fn test_easing_clamp_above_one() {
        assert!((EasingType::Linear.evaluate(2.0) - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_easing_bounce_at_one() {
        // At t=1.0 bounce should be close to 1.0
        assert!(EasingType::Bounce.evaluate(1.0) > 0.9);
    }

    #[test]
    fn test_easing_spring_at_zero() {
        // cos(0)=1, exp(0)=1 => 1-(1*1) = 0
        assert!((EasingType::Spring.evaluate(0.0) - 0.0).abs() < 1e-5);
    }

    // --- Keyframe tests ---

    #[test]
    fn test_keyframe_is_before_true() {
        let kf = Keyframe::new(10, 1.0, EasingType::Linear);
        assert!(kf.is_before(20));
    }

    #[test]
    fn test_keyframe_is_before_false_equal() {
        let kf = Keyframe::new(10, 1.0, EasingType::Linear);
        assert!(!kf.is_before(10));
    }

    #[test]
    fn test_keyframe_is_before_false_less() {
        let kf = Keyframe::new(10, 1.0, EasingType::Linear);
        assert!(!kf.is_before(5));
    }

    // --- KeyframeTrack tests ---

    #[test]
    fn test_track_empty_interpolate() {
        let track = KeyframeTrack::new("opacity");
        assert!((track.interpolate(50) - 0.0).abs() < 1e-9);
    }

    #[test]
    fn test_track_before_first_keyframe() {
        let mut track = KeyframeTrack::new("x");
        track.add(Keyframe::new(100, 5.0, EasingType::Linear));
        assert!((track.interpolate(0) - 5.0).abs() < 1e-9);
    }

    #[test]
    fn test_track_after_last_keyframe() {
        let mut track = KeyframeTrack::new("x");
        track.add(Keyframe::new(0, 0.0, EasingType::Linear));
        track.add(Keyframe::new(100, 10.0, EasingType::Linear));
        track.sort();
        assert!((track.interpolate(200) - 10.0).abs() < 1e-9);
    }

    #[test]
    fn test_track_linear_interpolation_midpoint() {
        let mut track = KeyframeTrack::new("x");
        track.add(Keyframe::new(0, 0.0, EasingType::Linear));
        track.add(Keyframe::new(100, 10.0, EasingType::Linear));
        track.sort();
        assert!((track.interpolate(50) - 5.0).abs() < 1e-6);
    }

    #[test]
    fn test_track_sort() {
        let mut track = KeyframeTrack::new("x");
        track.add(Keyframe::new(100, 10.0, EasingType::Linear));
        track.add(Keyframe::new(0, 0.0, EasingType::Linear));
        track.sort();
        assert_eq!(track.keyframes[0].time_frames, 0);
        assert_eq!(track.keyframes[1].time_frames, 100);
    }

    #[test]
    fn test_track_duration_frames() {
        let mut track = KeyframeTrack::new("x");
        track.add(Keyframe::new(0, 0.0, EasingType::Linear));
        track.add(Keyframe::new(240, 1.0, EasingType::Linear));
        assert_eq!(track.duration_frames(), 240);
    }

    #[test]
    fn test_track_keyframe_count() {
        let mut track = KeyframeTrack::new("x");
        track.add(Keyframe::new(0, 0.0, EasingType::Linear));
        track.add(Keyframe::new(50, 0.5, EasingType::EaseIn));
        track.add(Keyframe::new(100, 1.0, EasingType::Linear));
        assert_eq!(track.keyframe_count(), 3);
    }

    #[test]
    fn test_track_ease_in_interpolation() {
        let mut track = KeyframeTrack::new("x");
        track.add(Keyframe::new(0, 0.0, EasingType::EaseIn));
        track.add(Keyframe::new(100, 10.0, EasingType::Linear));
        track.sort();
        // At t=0.5, EaseIn => 0.25 => value = 2.5
        let v = track.interpolate(50);
        assert!((v - 2.5).abs() < 1e-4);
    }

    // --- AnimationCurve tests ---

    #[test]
    fn test_curve_add_and_get_track() {
        let mut curve = AnimationCurve::new();
        let track = KeyframeTrack::new("scale");
        curve.add_track(track);
        assert!(curve.get_track("scale").is_some());
        assert!(curve.get_track("missing").is_none());
    }

    #[test]
    fn test_curve_track_count() {
        let mut curve = AnimationCurve::new();
        curve.add_track(KeyframeTrack::new("a"));
        curve.add_track(KeyframeTrack::new("b"));
        assert_eq!(curve.track_count(), 2);
    }

    #[test]
    fn test_curve_default_is_empty() {
        let curve = AnimationCurve::default();
        assert_eq!(curve.track_count(), 0);
    }

    // --- CubicBezier easing tests ---

    #[test]
    fn test_cubic_bezier_linear() {
        // CubicBezier(0.0, 0.0, 1.0, 1.0) should approximate linear
        let easing = EasingType::CubicBezier {
            x1: 0.0,
            y1: 0.0,
            x2: 1.0,
            y2: 1.0,
        };
        assert!((easing.evaluate(0.0) - 0.0).abs() < 1e-3);
        assert!((easing.evaluate(0.5) - 0.5).abs() < 1e-2);
        assert!((easing.evaluate(1.0) - 1.0).abs() < 1e-3);
    }

    #[test]
    fn test_cubic_bezier_ease() {
        // CSS "ease": cubic-bezier(0.25, 0.1, 0.25, 1.0)
        let easing = EasingType::CubicBezier {
            x1: 0.25,
            y1: 0.1,
            x2: 0.25,
            y2: 1.0,
        };
        let v0 = easing.evaluate(0.0);
        let v_mid = easing.evaluate(0.5);
        let v1 = easing.evaluate(1.0);
        assert!((v0 - 0.0).abs() < 1e-3, "Start should be 0, got {v0}");
        assert!((v1 - 1.0).abs() < 1e-3, "End should be 1, got {v1}");
        // "ease" is slower at start, faster through middle
        assert!(v_mid > 0.5, "At t=0.5, ease should be > 0.5, got {v_mid}");
    }

    #[test]
    fn test_cubic_bezier_ease_in_out() {
        // CSS "ease-in-out": cubic-bezier(0.42, 0.0, 0.58, 1.0)
        let easing = EasingType::CubicBezier {
            x1: 0.42,
            y1: 0.0,
            x2: 0.58,
            y2: 1.0,
        };
        let v_mid = easing.evaluate(0.5);
        // Should be close to 0.5 due to symmetry
        assert!(
            (v_mid - 0.5).abs() < 0.05,
            "Midpoint should be ~0.5, got {v_mid}"
        );
    }

    #[test]
    fn test_cubic_bezier_endpoints() {
        let easing = EasingType::CubicBezier {
            x1: 0.42,
            y1: 0.0,
            x2: 0.58,
            y2: 1.0,
        };
        assert!((easing.evaluate(0.0) - 0.0).abs() < 1e-6);
        assert!((easing.evaluate(1.0) - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_cubic_bezier_clamping() {
        let easing = EasingType::CubicBezier {
            x1: 0.25,
            y1: 0.1,
            x2: 0.25,
            y2: 1.0,
        };
        // Values outside [0,1] should be clamped
        assert!((easing.evaluate(-1.0) - 0.0).abs() < 1e-3);
        assert!((easing.evaluate(2.0) - 1.0).abs() < 1e-3);
    }

    #[test]
    fn test_cubic_bezier_monotonic_for_standard_curves() {
        // Standard CSS ease should be monotonically increasing
        let easing = EasingType::CubicBezier {
            x1: 0.25,
            y1: 0.1,
            x2: 0.25,
            y2: 1.0,
        };
        let mut prev = 0.0_f32;
        for i in 0..=20 {
            let t = i as f32 / 20.0;
            let v = easing.evaluate(t);
            assert!(
                v >= prev - 1e-4,
                "Should be monotonic: t={t}, v={v}, prev={prev}"
            );
            prev = v;
        }
    }

    #[test]
    fn test_cubic_bezier_interpolation_in_track() {
        let mut track = KeyframeTrack::new("opacity");
        track.add(Keyframe::new(
            0,
            0.0,
            EasingType::CubicBezier {
                x1: 0.42,
                y1: 0.0,
                x2: 0.58,
                y2: 1.0,
            },
        ));
        track.add(Keyframe::new(100, 1.0, EasingType::Linear));
        track.sort();

        // At midpoint
        let v = track.interpolate(50);
        // With ease-in-out bezier, midpoint should be close to 0.5
        assert!(
            (v - 0.5).abs() < 0.1,
            "Expected ~0.5 at midpoint with ease-in-out, got {v}"
        );

        // At endpoints
        assert!((track.interpolate(0) - 0.0).abs() < 1e-6);
        assert!((track.interpolate(100) - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_cubic_bezier_extreme_control_points() {
        // Extreme ease-in: very slow start, fast end
        let easing = EasingType::CubicBezier {
            x1: 0.9,
            y1: 0.0,
            x2: 1.0,
            y2: 1.0,
        };
        let v_quarter = easing.evaluate(0.25);
        let v_three_quarter = easing.evaluate(0.75);
        // Should be very slow at start
        assert!(
            v_quarter < 0.15,
            "Extreme ease-in at 0.25 should be very small, got {v_quarter}"
        );
        // And catching up by 0.75
        assert!(
            v_three_quarter > 0.3,
            "Extreme ease-in at 0.75 should be moderate, got {v_three_quarter}"
        );
    }

    #[test]
    fn test_bezier_component_at_boundaries() {
        assert!((bezier_component(0.0, 0.25, 0.75) - 0.0).abs() < 1e-6);
        assert!((bezier_component(1.0, 0.25, 0.75) - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_solve_bezier_t_linear() {
        // For a linear bezier (x1=0, x2=1), solve_bezier_t should return t
        let u = solve_bezier_t(0.5, 0.0, 1.0);
        assert!(
            (u - 0.5).abs() < 1e-3,
            "Linear bezier solve: expected 0.5, got {u}"
        );
    }
}
