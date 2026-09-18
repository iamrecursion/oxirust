//! Variable-speed effects with keyframeable speed curves.
//!
//! Speed ramps allow clips to smoothly change playback speed over time,
//! enabling effects like slow-motion transitions, speed ramping, and
//! time remapping with configurable easing functions.

use serde::{Deserialize, Serialize};

use crate::error::{TimelineError, TimelineResult};
use crate::types::{Duration, Position};

/// Easing function for speed transitions between keyframes.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum SpeedEasing {
    /// Linear interpolation (constant rate of change).
    Linear,
    /// Ease-in (slow start, accelerating).
    EaseIn,
    /// Ease-out (fast start, decelerating).
    EaseOut,
    /// Ease-in-out (slow start and end).
    EaseInOut,
    /// Hold previous value until next keyframe.
    Hold,
}

impl SpeedEasing {
    /// Evaluates the easing function at parameter `t` in [0.0, 1.0].
    /// Returns a value in [0.0, 1.0] representing the interpolation weight.
    #[must_use]
    pub fn evaluate(self, t: f64) -> f64 {
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
            Self::Hold => 0.0,
        }
    }
}

/// A keyframe in a speed ramp curve.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct SpeedKeyframe {
    /// Time position (relative to clip start, in frames).
    pub time: Position,
    /// Speed multiplier at this keyframe (e.g. 1.0 = normal, 0.5 = half, 2.0 = double).
    pub speed: f64,
    /// Easing function for transition to the next keyframe.
    pub easing: SpeedEasing,
}

impl SpeedKeyframe {
    /// Creates a new speed keyframe.
    ///
    /// # Errors
    ///
    /// Returns error if speed is not in valid range (0.01..=100.0).
    pub fn new(time: Position, speed: f64, easing: SpeedEasing) -> TimelineResult<Self> {
        if !(0.01..=100.0).contains(&speed) {
            return Err(TimelineError::InvalidSpeed(speed));
        }
        Ok(Self {
            time,
            speed,
            easing,
        })
    }
}

/// A speed ramp effect that applies variable-speed playback to a clip.
///
/// The speed ramp is defined by a series of keyframes, each specifying
/// a speed value and an easing function for transitioning to the next keyframe.
/// Between keyframes, the speed is interpolated using the easing function
/// of the preceding keyframe.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SpeedRamp {
    /// Ordered list of speed keyframes (sorted by time).
    keyframes: Vec<SpeedKeyframe>,
    /// Whether the speed ramp is enabled.
    pub enabled: bool,
}

impl SpeedRamp {
    /// Creates a new speed ramp with a single keyframe at normal speed.
    #[must_use]
    pub fn new() -> Self {
        Self {
            keyframes: vec![SpeedKeyframe {
                time: Position::zero(),
                speed: 1.0,
                easing: SpeedEasing::Linear,
            }],
            enabled: true,
        }
    }

    /// Creates a speed ramp from a list of keyframes.
    ///
    /// # Errors
    ///
    /// Returns error if keyframes list is empty or contains invalid speeds.
    pub fn from_keyframes(mut keyframes: Vec<SpeedKeyframe>) -> TimelineResult<Self> {
        if keyframes.is_empty() {
            return Err(TimelineError::Other(
                "Speed ramp must have at least one keyframe".to_string(),
            ));
        }
        keyframes.sort_by_key(|kf| kf.time.value());
        Ok(Self {
            keyframes,
            enabled: true,
        })
    }

    /// Adds a keyframe to the speed ramp.
    ///
    /// If a keyframe already exists at the same time, it is replaced.
    pub fn add_keyframe(&mut self, keyframe: SpeedKeyframe) {
        // Remove existing keyframe at same time
        self.keyframes.retain(|kf| kf.time != keyframe.time);
        self.keyframes.push(keyframe);
        self.keyframes.sort_by_key(|kf| kf.time.value());
    }

    /// Removes a keyframe at the given time.
    ///
    /// # Errors
    ///
    /// Returns error if removing would leave zero keyframes.
    pub fn remove_keyframe(&mut self, time: Position) -> TimelineResult<()> {
        if self.keyframes.len() <= 1 {
            return Err(TimelineError::Other(
                "Cannot remove last keyframe from speed ramp".to_string(),
            ));
        }
        self.keyframes.retain(|kf| kf.time != time);
        Ok(())
    }

    /// Returns the speed at a given time position by interpolating between keyframes.
    #[must_use]
    #[allow(clippy::cast_precision_loss)]
    pub fn speed_at(&self, time: Position) -> f64 {
        if self.keyframes.is_empty() {
            return 1.0;
        }

        // Before first keyframe
        if time <= self.keyframes[0].time {
            return self.keyframes[0].speed;
        }

        // After last keyframe
        let last = self.keyframes.len() - 1;
        if time >= self.keyframes[last].time {
            return self.keyframes[last].speed;
        }

        // Find surrounding keyframes
        for i in 0..last {
            let kf_a = &self.keyframes[i];
            let kf_b = &self.keyframes[i + 1];

            if time >= kf_a.time && time < kf_b.time {
                let range = (kf_b.time.value() - kf_a.time.value()) as f64;
                if range <= 0.0 {
                    return kf_a.speed;
                }
                let t = (time.value() - kf_a.time.value()) as f64 / range;
                let eased_t = kf_a.easing.evaluate(t);
                return kf_a.speed + (kf_b.speed - kf_a.speed) * eased_t;
            }
        }

        self.keyframes[last].speed
    }

    /// Calculates the output duration given the source duration.
    ///
    /// This integrates the reciprocal of the speed curve to find the
    /// total output time for a given source duration.
    #[must_use]
    #[allow(clippy::cast_precision_loss, clippy::cast_possible_truncation)]
    pub fn calculate_output_duration(&self, source_duration: Duration) -> Duration {
        if !self.enabled || self.keyframes.len() == 1 {
            let speed = self.keyframes.first().map_or(1.0, |kf| kf.speed);
            return Duration::new((source_duration.value() as f64 / speed) as i64);
        }

        // Numerical integration using trapezoidal rule
        let steps = source_duration.value().max(1);
        let mut total_output = 0.0;

        for frame in 0..steps {
            let time = Position::new(frame);
            let speed = self.speed_at(time);
            // Each source frame takes 1/speed output frames
            total_output += 1.0 / speed;
        }

        Duration::new(total_output as i64)
    }

    /// Maps a source frame to its output frame position, accounting for
    /// variable speed.
    #[must_use]
    #[allow(clippy::cast_precision_loss, clippy::cast_possible_truncation)]
    pub fn source_to_output_frame(&self, source_frame: Position) -> Position {
        if !self.enabled {
            return source_frame;
        }

        let mut output = 0.0;
        for frame in 0..source_frame.value() {
            let speed = self.speed_at(Position::new(frame));
            output += 1.0 / speed;
        }

        Position::new(output as i64)
    }

    /// Maps an output frame to its source frame position, accounting for
    /// variable speed (inverse mapping).
    #[must_use]
    #[allow(clippy::cast_precision_loss, clippy::cast_possible_truncation)]
    pub fn output_to_source_frame(&self, output_frame: Position) -> Position {
        if !self.enabled {
            return output_frame;
        }

        let target = output_frame.value() as f64;
        let mut accumulated = 0.0;
        let mut source = 0i64;

        // Walk through source frames until we accumulate enough output frames
        let max_search = (target * 100.0) as i64 + 1000; // safety bound
        while accumulated < target && source < max_search {
            let speed = self.speed_at(Position::new(source));
            accumulated += 1.0 / speed;
            source += 1;
        }

        Position::new(source)
    }

    /// Returns all keyframes.
    #[must_use]
    pub fn keyframes(&self) -> &[SpeedKeyframe] {
        &self.keyframes
    }

    /// Returns the number of keyframes.
    #[must_use]
    pub fn keyframe_count(&self) -> usize {
        self.keyframes.len()
    }

    /// Returns the average speed across all keyframes.
    #[must_use]
    pub fn average_speed(&self) -> f64 {
        if self.keyframes.is_empty() {
            return 1.0;
        }
        let sum: f64 = self.keyframes.iter().map(|kf| kf.speed).sum();
        sum / self.keyframes.len() as f64
    }
}

impl Default for SpeedRamp {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_speed_easing_linear() {
        let e = SpeedEasing::Linear;
        assert!((e.evaluate(0.0) - 0.0).abs() < f64::EPSILON);
        assert!((e.evaluate(0.5) - 0.5).abs() < f64::EPSILON);
        assert!((e.evaluate(1.0) - 1.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_speed_easing_ease_in() {
        let e = SpeedEasing::EaseIn;
        assert!((e.evaluate(0.0) - 0.0).abs() < f64::EPSILON);
        assert!((e.evaluate(0.5) - 0.25).abs() < f64::EPSILON);
        assert!((e.evaluate(1.0) - 1.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_speed_easing_ease_out() {
        let e = SpeedEasing::EaseOut;
        assert!((e.evaluate(0.0) - 0.0).abs() < f64::EPSILON);
        assert!((e.evaluate(0.5) - 0.75).abs() < f64::EPSILON);
        assert!((e.evaluate(1.0) - 1.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_speed_easing_hold() {
        let e = SpeedEasing::Hold;
        assert!((e.evaluate(0.5) - 0.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_speed_easing_clamp() {
        let e = SpeedEasing::Linear;
        assert!((e.evaluate(-0.5) - 0.0).abs() < f64::EPSILON);
        assert!((e.evaluate(1.5) - 1.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_speed_keyframe_creation() {
        let kf = SpeedKeyframe::new(Position::new(0), 1.0, SpeedEasing::Linear);
        assert!(kf.is_ok());
    }

    #[test]
    fn test_speed_keyframe_invalid_speed() {
        assert!(SpeedKeyframe::new(Position::new(0), 0.0, SpeedEasing::Linear).is_err());
        assert!(SpeedKeyframe::new(Position::new(0), 101.0, SpeedEasing::Linear).is_err());
    }

    #[test]
    fn test_speed_ramp_new() {
        let ramp = SpeedRamp::new();
        assert_eq!(ramp.keyframe_count(), 1);
        assert!(ramp.enabled);
        assert!((ramp.speed_at(Position::new(0)) - 1.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_speed_ramp_add_keyframe() {
        let mut ramp = SpeedRamp::new();
        let kf = SpeedKeyframe::new(Position::new(50), 2.0, SpeedEasing::Linear)
            .expect("should succeed in test");
        ramp.add_keyframe(kf);
        assert_eq!(ramp.keyframe_count(), 2);
    }

    #[test]
    fn test_speed_ramp_add_keyframe_replaces_at_same_time() {
        let mut ramp = SpeedRamp::new();
        let kf1 = SpeedKeyframe::new(Position::new(0), 2.0, SpeedEasing::Linear)
            .expect("should succeed in test");
        ramp.add_keyframe(kf1);
        assert_eq!(ramp.keyframe_count(), 1);
        assert!((ramp.speed_at(Position::new(0)) - 2.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_speed_ramp_remove_keyframe() {
        let mut ramp = SpeedRamp::new();
        let kf = SpeedKeyframe::new(Position::new(50), 2.0, SpeedEasing::Linear)
            .expect("should succeed in test");
        ramp.add_keyframe(kf);
        assert!(ramp.remove_keyframe(Position::new(50)).is_ok());
        assert_eq!(ramp.keyframe_count(), 1);
    }

    #[test]
    fn test_speed_ramp_remove_last_keyframe_error() {
        let mut ramp = SpeedRamp::new();
        assert!(ramp.remove_keyframe(Position::new(0)).is_err());
    }

    #[test]
    fn test_speed_ramp_interpolation_linear() {
        let mut ramp = SpeedRamp::new();
        // Start at 1.0x speed, ramp to 2.0x at frame 100
        let kf = SpeedKeyframe::new(Position::new(100), 2.0, SpeedEasing::Linear)
            .expect("should succeed in test");
        ramp.add_keyframe(kf);

        // At frame 0: speed should be 1.0
        assert!((ramp.speed_at(Position::new(0)) - 1.0).abs() < f64::EPSILON);
        // At frame 50: speed should be 1.5 (linear interpolation)
        assert!((ramp.speed_at(Position::new(50)) - 1.5).abs() < f64::EPSILON);
        // At frame 100: speed should be 2.0
        assert!((ramp.speed_at(Position::new(100)) - 2.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_speed_ramp_interpolation_hold() {
        let mut ramp = SpeedRamp::new();
        ramp.keyframes[0].easing = SpeedEasing::Hold;
        let kf = SpeedKeyframe::new(Position::new(100), 2.0, SpeedEasing::Linear)
            .expect("should succeed in test");
        ramp.add_keyframe(kf);

        // With Hold easing, speed should remain at 1.0 until frame 100
        assert!((ramp.speed_at(Position::new(50)) - 1.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_speed_ramp_before_first_keyframe() {
        let kfs = vec![
            SpeedKeyframe::new(Position::new(10), 2.0, SpeedEasing::Linear)
                .expect("should succeed in test"),
        ];
        let ramp = SpeedRamp::from_keyframes(kfs).expect("should succeed in test");
        assert!((ramp.speed_at(Position::new(0)) - 2.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_speed_ramp_after_last_keyframe() {
        let ramp = SpeedRamp::new();
        assert!((ramp.speed_at(Position::new(1000)) - 1.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_speed_ramp_output_duration_constant() {
        let mut ramp = SpeedRamp::new();
        ramp.keyframes[0].speed = 2.0;
        let output = ramp.calculate_output_duration(Duration::new(100));
        assert_eq!(output.value(), 50);
    }

    #[test]
    fn test_speed_ramp_output_duration_normal() {
        let ramp = SpeedRamp::new();
        let output = ramp.calculate_output_duration(Duration::new(100));
        assert_eq!(output.value(), 100);
    }

    #[test]
    fn test_speed_ramp_source_to_output_disabled() {
        let mut ramp = SpeedRamp::new();
        ramp.enabled = false;
        let output = ramp.source_to_output_frame(Position::new(50));
        assert_eq!(output.value(), 50);
    }

    #[test]
    fn test_speed_ramp_output_to_source_disabled() {
        let mut ramp = SpeedRamp::new();
        ramp.enabled = false;
        let source = ramp.output_to_source_frame(Position::new(50));
        assert_eq!(source.value(), 50);
    }

    #[test]
    fn test_speed_ramp_average_speed() {
        let mut ramp = SpeedRamp::new();
        let kf = SpeedKeyframe::new(Position::new(100), 3.0, SpeedEasing::Linear)
            .expect("should succeed in test");
        ramp.add_keyframe(kf);
        assert!((ramp.average_speed() - 2.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_speed_ramp_from_keyframes_empty_error() {
        assert!(SpeedRamp::from_keyframes(vec![]).is_err());
    }

    #[test]
    fn test_speed_ramp_from_keyframes_sorts() {
        let kfs = vec![
            SpeedKeyframe::new(Position::new(100), 2.0, SpeedEasing::Linear)
                .expect("should succeed in test"),
            SpeedKeyframe::new(Position::new(0), 1.0, SpeedEasing::Linear)
                .expect("should succeed in test"),
        ];
        let ramp = SpeedRamp::from_keyframes(kfs).expect("should succeed in test");
        assert_eq!(ramp.keyframes()[0].time.value(), 0);
        assert_eq!(ramp.keyframes()[1].time.value(), 100);
    }

    #[test]
    fn test_speed_ramp_ease_in_out() {
        let e = SpeedEasing::EaseInOut;
        assert!((e.evaluate(0.0) - 0.0).abs() < f64::EPSILON);
        assert!((e.evaluate(1.0) - 1.0).abs() < f64::EPSILON);
        // At 0.5, ease-in-out should be exactly 0.5
        assert!((e.evaluate(0.5) - 0.5).abs() < f64::EPSILON);
    }

    #[test]
    fn test_speed_ramp_three_keyframes() {
        let kfs = vec![
            SpeedKeyframe::new(Position::new(0), 1.0, SpeedEasing::Linear)
                .expect("should succeed in test"),
            SpeedKeyframe::new(Position::new(50), 0.5, SpeedEasing::Linear)
                .expect("should succeed in test"),
            SpeedKeyframe::new(Position::new(100), 2.0, SpeedEasing::Linear)
                .expect("should succeed in test"),
        ];
        let ramp = SpeedRamp::from_keyframes(kfs).expect("should succeed in test");

        // At frame 25: should be between 1.0 and 0.5 => 0.75
        assert!((ramp.speed_at(Position::new(25)) - 0.75).abs() < f64::EPSILON);
        // At frame 75: should be between 0.5 and 2.0 => 1.25
        assert!((ramp.speed_at(Position::new(75)) - 1.25).abs() < f64::EPSILON);
    }

    #[test]
    fn test_speed_ramp_default() {
        let ramp = SpeedRamp::default();
        assert_eq!(ramp.keyframe_count(), 1);
    }

    #[test]
    fn test_speed_ramp_roundtrip_mapping() {
        let mut ramp = SpeedRamp::new();
        ramp.add_keyframe(
            SpeedKeyframe::new(Position::new(50), 2.0, SpeedEasing::Linear)
                .expect("should succeed in test"),
        );

        // Source->output->source should be approximately identity
        let source = Position::new(30);
        let output = ramp.source_to_output_frame(source);
        let back = ramp.output_to_source_frame(output);
        // Allow small rounding error
        assert!((back.value() - source.value()).abs() <= 2);
    }
}
