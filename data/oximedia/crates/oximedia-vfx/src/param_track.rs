//! Multi-channel parameter tracks for Vec2, Vec3, and Color keyframe animation.
//!
//! Extends the scalar [`crate::ParameterTrack`] with tracks that carry
//! structured values: 2D/3D vectors and RGBA colours.  Each track stores
//! keyframes sorted by time and performs per-component eased interpolation.

use crate::{Color, EasingFunction, Vec2, Vec3};
use serde::{Deserialize, Serialize};

// ─────────────────────────────────────────────────────────────────────────────
// Helpers
// ─────────────────────────────────────────────────────────────────────────────

/// Binary-search insertion position, replacing an existing keyframe at the same
/// time if one already exists.
///
/// Returns `(index, replaced)` where `replaced` is `true` when an existing
/// keyframe at `time` was overwritten.
fn sorted_insert_index<T>(
    keyframes: &[T],
    time: f64,
    get_time: impl Fn(&T) -> f64,
) -> (usize, bool) {
    match keyframes.binary_search_by(|k| {
        get_time(k)
            .partial_cmp(&time)
            .unwrap_or(std::cmp::Ordering::Equal)
    }) {
        Ok(idx) => (idx, true),
        Err(idx) => (idx, false),
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Vec2Track
// ─────────────────────────────────────────────────────────────────────────────

/// A single keyframe carrying a [`Vec2`] value.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Vec2Keyframe {
    /// Time in seconds.
    pub time: f64,
    /// 2D value at this keyframe.
    pub value: Vec2,
    /// Easing function applied when interpolating *from* this keyframe.
    pub easing: EasingFunction,
}

/// Animatable parameter track that stores [`Vec2`] keyframes.
///
/// Each component (x, y) is interpolated independently using the easing
/// function of the preceding keyframe.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct Vec2Track {
    keyframes: Vec<Vec2Keyframe>,
}

impl Vec2Track {
    /// Create an empty track.
    #[must_use]
    pub fn new() -> Self {
        Self {
            keyframes: Vec::new(),
        }
    }

    /// Add or replace a keyframe at `time`.
    pub fn add_keyframe(&mut self, time: f64, value: Vec2, easing: EasingFunction) {
        let kf = Vec2Keyframe {
            time,
            value,
            easing,
        };
        let (idx, replaced) = sorted_insert_index(&self.keyframes, time, |k| k.time);
        if replaced {
            self.keyframes[idx] = kf;
        } else {
            self.keyframes.insert(idx, kf);
        }
    }

    /// Evaluate the interpolated [`Vec2`] at `time`.
    ///
    /// Returns `None` if the track is empty.
    #[must_use]
    pub fn evaluate(&self, time: f64) -> Option<Vec2> {
        let kfs = &self.keyframes;
        if kfs.is_empty() {
            return None;
        }
        if kfs.len() == 1 {
            return Some(kfs[0].value);
        }
        let idx = match kfs.binary_search_by(|k| {
            k.time
                .partial_cmp(&time)
                .unwrap_or(std::cmp::Ordering::Equal)
        }) {
            Ok(i) => return Some(kfs[i].value),
            Err(i) => i,
        };
        if idx == 0 {
            return Some(kfs[0].value);
        }
        if idx >= kfs.len() {
            return Some(kfs[kfs.len() - 1].value);
        }
        let k1 = &kfs[idx - 1];
        let k2 = &kfs[idx];
        let dt = k2.time - k1.time;
        if dt <= 0.0 {
            return Some(k1.value);
        }
        let t = ((time - k1.time) / dt) as f32;
        let eased = k1.easing.apply(t);
        Some(Vec2::new(
            k1.value.x + (k2.value.x - k1.value.x) * eased,
            k1.value.y + (k2.value.y - k1.value.y) * eased,
        ))
    }

    /// Number of keyframes.
    #[must_use]
    pub fn len(&self) -> usize {
        self.keyframes.len()
    }

    /// Returns `true` if the track has no keyframes.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.keyframes.is_empty()
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Vec3Track
// ─────────────────────────────────────────────────────────────────────────────

/// A single keyframe carrying a [`Vec3`] value.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Vec3Keyframe {
    /// Time in seconds.
    pub time: f64,
    /// 3D value at this keyframe.
    pub value: Vec3,
    /// Easing function applied when interpolating *from* this keyframe.
    pub easing: EasingFunction,
}

/// Animatable parameter track that stores [`Vec3`] keyframes.
///
/// Each component (x, y, z) is interpolated independently using the easing
/// function of the preceding keyframe.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct Vec3Track {
    keyframes: Vec<Vec3Keyframe>,
}

impl Vec3Track {
    /// Create an empty track.
    #[must_use]
    pub fn new() -> Self {
        Self {
            keyframes: Vec::new(),
        }
    }

    /// Add or replace a keyframe at `time`.
    pub fn add_keyframe(&mut self, time: f64, value: Vec3, easing: EasingFunction) {
        let kf = Vec3Keyframe {
            time,
            value,
            easing,
        };
        let (idx, replaced) = sorted_insert_index(&self.keyframes, time, |k| k.time);
        if replaced {
            self.keyframes[idx] = kf;
        } else {
            self.keyframes.insert(idx, kf);
        }
    }

    /// Evaluate the interpolated [`Vec3`] at `time`.
    ///
    /// Returns `None` if the track is empty.
    #[must_use]
    pub fn evaluate(&self, time: f64) -> Option<Vec3> {
        let kfs = &self.keyframes;
        if kfs.is_empty() {
            return None;
        }
        if kfs.len() == 1 {
            return Some(kfs[0].value);
        }
        let idx = match kfs.binary_search_by(|k| {
            k.time
                .partial_cmp(&time)
                .unwrap_or(std::cmp::Ordering::Equal)
        }) {
            Ok(i) => return Some(kfs[i].value),
            Err(i) => i,
        };
        if idx == 0 {
            return Some(kfs[0].value);
        }
        if idx >= kfs.len() {
            return Some(kfs[kfs.len() - 1].value);
        }
        let k1 = &kfs[idx - 1];
        let k2 = &kfs[idx];
        let dt = k2.time - k1.time;
        if dt <= 0.0 {
            return Some(k1.value);
        }
        let t = ((time - k1.time) / dt) as f32;
        let eased = k1.easing.apply(t);
        Some(k1.value.lerp(&k2.value, eased))
    }

    /// Number of keyframes.
    #[must_use]
    pub fn len(&self) -> usize {
        self.keyframes.len()
    }

    /// Returns `true` if the track has no keyframes.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.keyframes.is_empty()
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// ColorTrack
// ─────────────────────────────────────────────────────────────────────────────

/// A single keyframe carrying a [`Color`] (RGBA) value.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ColorKeyframe {
    /// Time in seconds.
    pub time: f64,
    /// RGBA colour at this keyframe.
    pub value: Color,
    /// Easing function applied when interpolating *from* this keyframe.
    pub easing: EasingFunction,
}

/// Animatable parameter track that stores [`Color`] (RGBA) keyframes.
///
/// Each channel (r, g, b, a) is interpolated as `f32` in [0, 255] and then
/// clamped back to `u8`.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct ColorTrack {
    keyframes: Vec<ColorKeyframe>,
}

impl ColorTrack {
    /// Create an empty track.
    #[must_use]
    pub fn new() -> Self {
        Self {
            keyframes: Vec::new(),
        }
    }

    /// Add or replace a keyframe at `time`.
    pub fn add_keyframe(&mut self, time: f64, value: Color, easing: EasingFunction) {
        let kf = ColorKeyframe {
            time,
            value,
            easing,
        };
        let (idx, replaced) = sorted_insert_index(&self.keyframes, time, |k| k.time);
        if replaced {
            self.keyframes[idx] = kf;
        } else {
            self.keyframes.insert(idx, kf);
        }
    }

    /// Evaluate the interpolated [`Color`] at `time`.
    ///
    /// Returns `None` if the track is empty.
    #[must_use]
    pub fn evaluate(&self, time: f64) -> Option<Color> {
        let kfs = &self.keyframes;
        if kfs.is_empty() {
            return None;
        }
        if kfs.len() == 1 {
            return Some(kfs[0].value);
        }
        let idx = match kfs.binary_search_by(|k| {
            k.time
                .partial_cmp(&time)
                .unwrap_or(std::cmp::Ordering::Equal)
        }) {
            Ok(i) => return Some(kfs[i].value),
            Err(i) => i,
        };
        if idx == 0 {
            return Some(kfs[0].value);
        }
        if idx >= kfs.len() {
            return Some(kfs[kfs.len() - 1].value);
        }
        let k1 = &kfs[idx - 1];
        let k2 = &kfs[idx];
        let dt = k2.time - k1.time;
        if dt <= 0.0 {
            return Some(k1.value);
        }
        let t = ((time - k1.time) / dt) as f32;
        let eased = k1.easing.apply(t);
        Some(k1.value.lerp(k2.value, eased))
    }

    /// Number of keyframes.
    #[must_use]
    pub fn len(&self) -> usize {
        self.keyframes.len()
    }

    /// Returns `true` if the track has no keyframes.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.keyframes.is_empty()
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── Vec2Track ──────────────────────────────────────────────────────────

    #[test]
    fn test_vec2_track_empty_returns_none() {
        let track = Vec2Track::new();
        assert_eq!(track.evaluate(0.5), None);
    }

    #[test]
    fn test_vec2_track_single_keyframe_constant() {
        let mut track = Vec2Track::new();
        track.add_keyframe(0.0, Vec2::new(3.0, 7.0), EasingFunction::Linear);
        let v = track.evaluate(99.0).expect("single kf");
        assert!((v.x - 3.0).abs() < 1e-5);
        assert!((v.y - 7.0).abs() < 1e-5);
    }

    #[test]
    fn test_vec2_track_linear_mid() {
        let mut track = Vec2Track::new();
        track.add_keyframe(0.0, Vec2::new(0.0, 0.0), EasingFunction::Linear);
        track.add_keyframe(1.0, Vec2::new(10.0, 20.0), EasingFunction::Linear);
        let v = track.evaluate(0.5).expect("mid");
        assert!((v.x - 5.0).abs() < 0.01);
        assert!((v.y - 10.0).abs() < 0.01);
    }

    #[test]
    fn test_vec2_track_before_first_kf() {
        let mut track = Vec2Track::new();
        track.add_keyframe(1.0, Vec2::new(5.0, 5.0), EasingFunction::Linear);
        track.add_keyframe(2.0, Vec2::new(10.0, 10.0), EasingFunction::Linear);
        let v = track.evaluate(0.0).expect("before first");
        assert!((v.x - 5.0).abs() < 1e-5);
    }

    #[test]
    fn test_vec2_track_after_last_kf() {
        let mut track = Vec2Track::new();
        track.add_keyframe(0.0, Vec2::new(0.0, 0.0), EasingFunction::Linear);
        track.add_keyframe(1.0, Vec2::new(1.0, 1.0), EasingFunction::Linear);
        let v = track.evaluate(5.0).expect("after last");
        assert!((v.x - 1.0).abs() < 1e-5);
    }

    #[test]
    fn test_vec2_track_unsorted_insertion() {
        let mut track = Vec2Track::new();
        track.add_keyframe(2.0, Vec2::new(2.0, 2.0), EasingFunction::Linear);
        track.add_keyframe(0.0, Vec2::new(0.0, 0.0), EasingFunction::Linear);
        track.add_keyframe(1.0, Vec2::new(1.0, 1.0), EasingFunction::Linear);
        let v = track.evaluate(0.5).expect("mid unsorted");
        assert!((v.x - 0.5).abs() < 0.01);
    }

    #[test]
    fn test_vec2_track_replace_keyframe() {
        let mut track = Vec2Track::new();
        track.add_keyframe(0.5, Vec2::new(1.0, 1.0), EasingFunction::Linear);
        track.add_keyframe(0.5, Vec2::new(99.0, 99.0), EasingFunction::Linear);
        assert_eq!(track.len(), 1);
        let v = track.evaluate(0.5).expect("replaced");
        assert!((v.x - 99.0).abs() < 1e-5);
    }

    // ── Vec3Track ──────────────────────────────────────────────────────────

    #[test]
    fn test_vec3_track_empty_returns_none() {
        let track = Vec3Track::new();
        assert_eq!(track.evaluate(0.5), None);
    }

    #[test]
    fn test_vec3_track_single_keyframe() {
        let mut track = Vec3Track::new();
        track.add_keyframe(0.0, Vec3::new(1.0, 2.0, 3.0), EasingFunction::Linear);
        let v = track.evaluate(10.0).expect("single");
        assert!((v.z - 3.0).abs() < 1e-5);
    }

    #[test]
    fn test_vec3_track_linear_mid() {
        let mut track = Vec3Track::new();
        track.add_keyframe(0.0, Vec3::new(0.0, 0.0, 0.0), EasingFunction::Linear);
        track.add_keyframe(1.0, Vec3::new(4.0, 6.0, 8.0), EasingFunction::Linear);
        let v = track.evaluate(0.5).expect("mid");
        assert!((v.x - 2.0).abs() < 0.01);
        assert!((v.y - 3.0).abs() < 0.01);
        assert!((v.z - 4.0).abs() < 0.01);
    }

    #[test]
    fn test_vec3_track_ease_in_slower() {
        let mut track = Vec3Track::new();
        track.add_keyframe(0.0, Vec3::new(0.0, 0.0, 0.0), EasingFunction::EaseIn);
        track.add_keyframe(1.0, Vec3::new(10.0, 10.0, 10.0), EasingFunction::EaseIn);
        let v_mid = track.evaluate(0.5).expect("ease-in mid");
        // EaseIn at t=0.5: t*t = 0.25, so value should be < 5.0
        assert!(v_mid.x < 5.0, "ease-in should be below linear at mid");
    }

    #[test]
    fn test_vec3_track_boundary() {
        let mut track = Vec3Track::new();
        track.add_keyframe(0.0, Vec3::new(1.0, 1.0, 1.0), EasingFunction::Linear);
        track.add_keyframe(2.0, Vec3::new(3.0, 3.0, 3.0), EasingFunction::Linear);
        let before = track.evaluate(-1.0).expect("before");
        let after = track.evaluate(5.0).expect("after");
        assert!((before.x - 1.0).abs() < 1e-5);
        assert!((after.x - 3.0).abs() < 1e-5);
    }

    // ── ColorTrack ─────────────────────────────────────────────────────────

    #[test]
    fn test_color_track_empty_returns_none() {
        let track = ColorTrack::new();
        assert_eq!(track.evaluate(0.5), None);
    }

    #[test]
    fn test_color_track_single_keyframe() {
        let mut track = ColorTrack::new();
        let col = Color::new(100, 150, 200, 255);
        track.add_keyframe(0.0, col, EasingFunction::Linear);
        let result = track.evaluate(5.0).expect("single");
        assert_eq!(result.r, 100);
        assert_eq!(result.g, 150);
    }

    #[test]
    fn test_color_track_linear_mid() {
        let mut track = ColorTrack::new();
        track.add_keyframe(0.0, Color::new(0, 0, 0, 0), EasingFunction::Linear);
        track.add_keyframe(1.0, Color::new(200, 100, 50, 255), EasingFunction::Linear);
        let mid = track.evaluate(0.5).expect("mid");
        assert!((mid.r as i32 - 100).abs() <= 2, "r={}", mid.r);
        assert!((mid.g as i32 - 50).abs() <= 2, "g={}", mid.g);
        assert!((mid.a as i32 - 127).abs() <= 3, "a={}", mid.a);
    }

    #[test]
    fn test_color_track_unsorted_insertion() {
        let mut track = ColorTrack::new();
        track.add_keyframe(1.0, Color::rgb(100, 100, 100), EasingFunction::Linear);
        track.add_keyframe(0.0, Color::rgb(0, 0, 0), EasingFunction::Linear);
        // At time 0.0 exact match
        let v = track.evaluate(0.0).expect("at 0");
        assert_eq!(v.r, 0);
    }

    #[test]
    fn test_color_track_is_empty_and_len() {
        let mut track = ColorTrack::new();
        assert!(track.is_empty());
        track.add_keyframe(0.0, Color::black(), EasingFunction::Linear);
        assert_eq!(track.len(), 1);
        assert!(!track.is_empty());
    }

    #[test]
    fn test_color_track_replace_at_same_time() {
        let mut track = ColorTrack::new();
        track.add_keyframe(1.0, Color::rgb(10, 10, 10), EasingFunction::Linear);
        track.add_keyframe(1.0, Color::rgb(200, 200, 200), EasingFunction::Linear);
        assert_eq!(track.len(), 1);
        let v = track.evaluate(1.0).expect("exact");
        assert_eq!(v.r, 200);
    }
}
