//! Keyframe animation system.

use serde::{Deserialize, Serialize};

use crate::types::Position;

/// Interpolation method for keyframes.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum KeyframeInterpolation {
    /// Linear interpolation.
    Linear,
    /// Bezier curve interpolation.
    Bezier,
    /// Ease-in (slow start).
    EaseIn,
    /// Ease-out (slow end).
    EaseOut,
    /// Ease-in-ease-out (slow start and end).
    EaseInOut,
    /// Hold (no interpolation).
    Hold,
}

/// Value stored in a keyframe.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub enum KeyframeValue {
    /// Single floating-point value.
    Float(f64),
    /// 2D vector value.
    Vec2([f64; 2]),
    /// 3D vector value.
    Vec3([f64; 3]),
    /// 4D vector value (e.g., RGBA color).
    Vec4([f64; 4]),
    /// Boolean value.
    Bool(bool),
}

impl KeyframeValue {
    /// Creates a float keyframe value.
    #[must_use]
    pub const fn float(value: f64) -> Self {
        Self::Float(value)
    }

    /// Creates a 2D vector keyframe value.
    #[must_use]
    pub const fn vec2(x: f64, y: f64) -> Self {
        Self::Vec2([x, y])
    }

    /// Creates a 3D vector keyframe value.
    #[must_use]
    pub const fn vec3(x: f64, y: f64, z: f64) -> Self {
        Self::Vec3([x, y, z])
    }

    /// Creates a 4D vector keyframe value.
    #[must_use]
    pub const fn vec4(x: f64, y: f64, z: f64, w: f64) -> Self {
        Self::Vec4([x, y, z, w])
    }

    /// Creates a boolean keyframe value.
    #[must_use]
    pub const fn bool(value: bool) -> Self {
        Self::Bool(value)
    }

    /// Interpolates between two keyframe values.
    ///
    /// # Arguments
    ///
    /// * `other` - The other keyframe value
    /// * `t` - Interpolation factor (0.0-1.0)
    /// * `interpolation` - Interpolation method
    ///
    /// Returns `None` if values are incompatible types.
    #[must_use]
    pub fn interpolate(
        &self,
        other: &Self,
        t: f64,
        interpolation: KeyframeInterpolation,
    ) -> Option<Self> {
        let t = Self::apply_interpolation(t, interpolation);

        match (self, other) {
            (Self::Float(a), Self::Float(b)) => Some(Self::Float(a + (b - a) * t)),
            (Self::Vec2([ax, ay]), Self::Vec2([bx, by])) => {
                Some(Self::Vec2([ax + (bx - ax) * t, ay + (by - ay) * t]))
            }
            (Self::Vec3([ax, ay, az]), Self::Vec3([bx, by, bz])) => Some(Self::Vec3([
                ax + (bx - ax) * t,
                ay + (by - ay) * t,
                az + (bz - az) * t,
            ])),
            (Self::Vec4([ax, ay, az, aw]), Self::Vec4([bx, by, bz, bw])) => Some(Self::Vec4([
                ax + (bx - ax) * t,
                ay + (by - ay) * t,
                az + (bz - az) * t,
                aw + (bw - aw) * t,
            ])),
            (Self::Bool(a), Self::Bool(_)) => {
                // Boolean uses hold interpolation
                Some(Self::Bool(*a))
            }
            _ => None,
        }
    }

    /// Applies interpolation curve to t value.
    #[must_use]
    fn apply_interpolation(t: f64, interpolation: KeyframeInterpolation) -> f64 {
        match interpolation {
            KeyframeInterpolation::Linear => t,
            KeyframeInterpolation::Bezier => {
                // Simple cubic bezier approximation
                t * t * (3.0 - 2.0 * t)
            }
            KeyframeInterpolation::EaseIn => t * t,
            KeyframeInterpolation::EaseOut => t * (2.0 - t),
            KeyframeInterpolation::EaseInOut => {
                if t < 0.5 {
                    2.0 * t * t
                } else {
                    -1.0 + (4.0 - 2.0 * t) * t
                }
            }
            KeyframeInterpolation::Hold => 0.0,
        }
    }
}

/// A keyframe at a specific time.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Keyframe {
    /// Position in timeline.
    pub position: Position,
    /// Value at this keyframe.
    pub value: KeyframeValue,
    /// Interpolation to next keyframe.
    pub interpolation: KeyframeInterpolation,
    /// Parameter name this keyframe affects.
    pub parameter: String,
}

impl Keyframe {
    /// Creates a new keyframe.
    #[must_use]
    pub fn new(position: Position, value: KeyframeValue, parameter: String) -> Self {
        Self {
            position,
            value,
            interpolation: KeyframeInterpolation::Linear,
            parameter,
        }
    }

    /// Creates a new keyframe with interpolation.
    #[must_use]
    pub fn with_interpolation(
        position: Position,
        value: KeyframeValue,
        parameter: String,
        interpolation: KeyframeInterpolation,
    ) -> Self {
        Self {
            position,
            value,
            interpolation,
            parameter,
        }
    }

    /// Sets the interpolation method.
    pub fn set_interpolation(&mut self, interpolation: KeyframeInterpolation) {
        self.interpolation = interpolation;
    }
}

/// Evaluates a parameter value at a given position using keyframes.
///
/// # Arguments
///
/// * `keyframes` - List of keyframes for the parameter
/// * `position` - Timeline position to evaluate
/// * `default_value` - Default value if no keyframes exist
///
/// Returns the interpolated value at the given position.
#[must_use]
#[allow(clippy::cast_precision_loss)]
pub fn evaluate_keyframes(
    keyframes: &[Keyframe],
    position: Position,
    default_value: &KeyframeValue,
) -> KeyframeValue {
    if keyframes.is_empty() {
        return default_value.clone();
    }

    // Find surrounding keyframes
    let mut before = None;
    let mut after = None;

    for keyframe in keyframes {
        if keyframe.position <= position {
            before = Some(keyframe);
        }
        if keyframe.position > position && after.is_none() {
            after = Some(keyframe);
            break;
        }
    }

    match (before, after) {
        (Some(before_kf), Some(after_kf)) => {
            // Interpolate between keyframes
            let range = (after_kf.position.value() - before_kf.position.value()) as f64;
            let offset = (position.value() - before_kf.position.value()) as f64;
            let t = offset / range;

            before_kf
                .value
                .interpolate(&after_kf.value, t, before_kf.interpolation)
                .unwrap_or_else(|| before_kf.value.clone())
        }
        (Some(before_kf), None) => {
            // Use last keyframe value
            before_kf.value.clone()
        }
        (None, Some(after_kf)) => {
            // Use first keyframe value
            after_kf.value.clone()
        }
        (None, None) => default_value.clone(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_keyframe_value_float() {
        let val = KeyframeValue::float(1.5);
        assert!(matches!(val, KeyframeValue::Float(v) if (v - 1.5).abs() < f64::EPSILON));
    }

    #[test]
    fn test_keyframe_value_vec2() {
        let val = KeyframeValue::vec2(1.0, 2.0);
        assert!(
            matches!(val, KeyframeValue::Vec2([x, y]) if (x - 1.0).abs() < f64::EPSILON && (y - 2.0).abs() < f64::EPSILON)
        );
    }

    #[test]
    fn test_keyframe_value_interpolate_float() {
        let a = KeyframeValue::float(0.0);
        let b = KeyframeValue::float(10.0);
        let result = a
            .interpolate(&b, 0.5, KeyframeInterpolation::Linear)
            .expect("should succeed in test");
        assert!(matches!(result, KeyframeValue::Float(v) if (v - 5.0).abs() < f64::EPSILON));
    }

    #[test]
    fn test_keyframe_value_interpolate_vec2() {
        let a = KeyframeValue::vec2(0.0, 0.0);
        let b = KeyframeValue::vec2(10.0, 20.0);
        let result = a
            .interpolate(&b, 0.5, KeyframeInterpolation::Linear)
            .expect("should succeed in test");
        match result {
            KeyframeValue::Vec2([x, y]) => {
                assert!((x - 5.0).abs() < f64::EPSILON);
                assert!((y - 10.0).abs() < f64::EPSILON);
            }
            _ => panic!("Expected Vec2"),
        }
    }

    #[test]
    fn test_keyframe_value_interpolate_incompatible() {
        let a = KeyframeValue::float(0.0);
        let b = KeyframeValue::vec2(10.0, 20.0);
        let result = a.interpolate(&b, 0.5, KeyframeInterpolation::Linear);
        assert!(result.is_none());
    }

    #[test]
    fn test_keyframe_value_interpolate_bool() {
        let a = KeyframeValue::bool(true);
        let b = KeyframeValue::bool(false);
        let result = a
            .interpolate(&b, 0.5, KeyframeInterpolation::Linear)
            .expect("should succeed in test");
        assert!(matches!(result, KeyframeValue::Bool(true)));
    }

    #[test]
    fn test_keyframe_creation() {
        let kf = Keyframe::new(
            Position::new(100),
            KeyframeValue::float(1.5),
            "opacity".to_string(),
        );
        assert_eq!(kf.position.value(), 100);
        assert_eq!(kf.parameter, "opacity");
        assert_eq!(kf.interpolation, KeyframeInterpolation::Linear);
    }

    #[test]
    fn test_keyframe_with_interpolation() {
        let kf = Keyframe::with_interpolation(
            Position::new(100),
            KeyframeValue::float(1.5),
            "opacity".to_string(),
            KeyframeInterpolation::EaseIn,
        );
        assert_eq!(kf.interpolation, KeyframeInterpolation::EaseIn);
    }

    #[test]
    fn test_evaluate_keyframes_empty() {
        let keyframes: Vec<Keyframe> = vec![];
        let default = KeyframeValue::float(1.0);
        let result = evaluate_keyframes(&keyframes, Position::new(50), &default);
        assert!(matches!(result, KeyframeValue::Float(v) if (v - 1.0).abs() < f64::EPSILON));
    }

    #[test]
    fn test_evaluate_keyframes_single() {
        let keyframes = vec![Keyframe::new(
            Position::new(0),
            KeyframeValue::float(5.0),
            "test".to_string(),
        )];
        let default = KeyframeValue::float(1.0);
        let result = evaluate_keyframes(&keyframes, Position::new(100), &default);
        assert!(matches!(result, KeyframeValue::Float(v) if (v - 5.0).abs() < f64::EPSILON));
    }

    #[test]
    fn test_evaluate_keyframes_interpolate() {
        let keyframes = vec![
            Keyframe::new(
                Position::new(0),
                KeyframeValue::float(0.0),
                "test".to_string(),
            ),
            Keyframe::new(
                Position::new(100),
                KeyframeValue::float(10.0),
                "test".to_string(),
            ),
        ];
        let default = KeyframeValue::float(1.0);
        let result = evaluate_keyframes(&keyframes, Position::new(50), &default);
        assert!(matches!(result, KeyframeValue::Float(v) if (v - 5.0).abs() < f64::EPSILON));
    }

    #[test]
    fn test_evaluate_keyframes_before_first() {
        let keyframes = vec![Keyframe::new(
            Position::new(100),
            KeyframeValue::float(5.0),
            "test".to_string(),
        )];
        let default = KeyframeValue::float(1.0);
        let result = evaluate_keyframes(&keyframes, Position::new(50), &default);
        assert!(matches!(result, KeyframeValue::Float(v) if (v - 5.0).abs() < f64::EPSILON));
    }

    #[test]
    fn test_evaluate_keyframes_after_last() {
        let keyframes = vec![Keyframe::new(
            Position::new(0),
            KeyframeValue::float(5.0),
            "test".to_string(),
        )];
        let default = KeyframeValue::float(1.0);
        let result = evaluate_keyframes(&keyframes, Position::new(100), &default);
        assert!(matches!(result, KeyframeValue::Float(v) if (v - 5.0).abs() < f64::EPSILON));
    }

    #[test]
    fn test_interpolation_ease_in() {
        let t = KeyframeValue::apply_interpolation(0.5, KeyframeInterpolation::EaseIn);
        assert!((t - 0.25).abs() < f64::EPSILON);
    }

    #[test]
    fn test_interpolation_ease_out() {
        let t = KeyframeValue::apply_interpolation(0.5, KeyframeInterpolation::EaseOut);
        assert!((t - 0.75).abs() < f64::EPSILON);
    }

    #[test]
    fn test_interpolation_hold() {
        let t = KeyframeValue::apply_interpolation(0.5, KeyframeInterpolation::Hold);
        assert!((t - 0.0).abs() < f64::EPSILON);
    }
}
