//! Basic types for timeline operations.

use oximedia_core::Rational;
use serde::{Deserialize, Serialize};
use std::fmt;
use std::ops::{Add, Sub};

/// Position in the timeline (in frames or samples).
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub struct Position(pub i64);

impl Position {
    /// Creates a new position.
    #[must_use]
    pub const fn new(value: i64) -> Self {
        Self(value)
    }

    /// Creates a position at zero.
    #[must_use]
    pub const fn zero() -> Self {
        Self(0)
    }

    /// Returns the underlying value.
    #[must_use]
    pub const fn value(self) -> i64 {
        self.0
    }

    /// Converts position to frames given a frame rate.
    #[must_use]
    #[allow(clippy::cast_possible_truncation)]
    pub fn to_frames(self, _frame_rate: Rational) -> i64 {
        self.0
    }

    /// Converts frames to position.
    #[must_use]
    pub const fn from_frames(frames: i64) -> Self {
        Self(frames)
    }

    /// Converts position to seconds.
    #[must_use]
    #[allow(clippy::cast_precision_loss)]
    pub fn to_seconds(self, frame_rate: Rational) -> f64 {
        self.0 as f64 * frame_rate.to_f64()
    }

    /// Creates position from seconds.
    #[must_use]
    #[allow(clippy::cast_possible_truncation)]
    pub fn from_seconds(seconds: f64, frame_rate: Rational) -> Self {
        Self((seconds / frame_rate.to_f64()) as i64)
    }
}

impl Add for Position {
    type Output = Self;

    fn add(self, rhs: Self) -> Self::Output {
        Self(self.0 + rhs.0)
    }
}

impl Sub for Position {
    type Output = Self;

    fn sub(self, rhs: Self) -> Self::Output {
        Self(self.0 - rhs.0)
    }
}

impl Add<Duration> for Position {
    type Output = Self;

    fn add(self, rhs: Duration) -> Self::Output {
        Self(self.0 + rhs.0)
    }
}

impl Sub<Duration> for Position {
    type Output = Self;

    fn sub(self, rhs: Duration) -> Self::Output {
        Self(self.0 - rhs.0)
    }
}

impl fmt::Display for Position {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.0)
    }
}

impl Default for Position {
    fn default() -> Self {
        Self::zero()
    }
}

/// Duration in the timeline (in frames or samples).
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub struct Duration(pub i64);

impl Duration {
    /// Creates a new duration.
    #[must_use]
    pub const fn new(value: i64) -> Self {
        Self(value)
    }

    /// Creates a zero duration.
    #[must_use]
    pub const fn zero() -> Self {
        Self(0)
    }

    /// Returns the underlying value.
    #[must_use]
    pub const fn value(self) -> i64 {
        self.0
    }

    /// Converts duration to frames.
    #[must_use]
    pub const fn to_frames(self) -> i64 {
        self.0
    }

    /// Creates duration from frames.
    #[must_use]
    pub const fn from_frames(frames: i64) -> Self {
        Self(frames)
    }

    /// Converts duration to seconds.
    #[must_use]
    #[allow(clippy::cast_precision_loss)]
    pub fn to_seconds(self, frame_rate: Rational) -> f64 {
        self.0 as f64 * frame_rate.to_f64()
    }

    /// Creates duration from seconds.
    #[must_use]
    #[allow(clippy::cast_possible_truncation)]
    pub fn from_seconds(seconds: f64, frame_rate: Rational) -> Self {
        Self((seconds / frame_rate.to_f64()) as i64)
    }

    /// Checks if duration is zero.
    #[must_use]
    pub const fn is_zero(self) -> bool {
        self.0 == 0
    }

    /// Returns the absolute value.
    #[must_use]
    pub const fn abs(self) -> Self {
        Self(self.0.abs())
    }
}

impl Add for Duration {
    type Output = Self;

    fn add(self, rhs: Self) -> Self::Output {
        Self(self.0 + rhs.0)
    }
}

impl Sub for Duration {
    type Output = Self;

    fn sub(self, rhs: Self) -> Self::Output {
        Self(self.0 - rhs.0)
    }
}

impl fmt::Display for Duration {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.0)
    }
}

impl Default for Duration {
    fn default() -> Self {
        Self::zero()
    }
}

/// Playback speed multiplier.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct Speed(pub f64);

impl Speed {
    /// Creates a new speed.
    ///
    /// # Errors
    ///
    /// Returns error if speed is not between 0.25 and 4.0.
    pub fn new(value: f64) -> crate::error::TimelineResult<Self> {
        if !(0.25..=4.0).contains(&value.abs()) {
            return Err(crate::error::TimelineError::InvalidSpeed(value));
        }
        Ok(Self(value))
    }

    /// Normal playback speed (1.0x).
    #[must_use]
    pub const fn normal() -> Self {
        Self(1.0)
    }

    /// Returns the underlying value.
    #[must_use]
    pub const fn value(self) -> f64 {
        self.0
    }

    /// Checks if this is normal speed.
    #[must_use]
    pub fn is_normal(self) -> bool {
        (self.0 - 1.0).abs() < f64::EPSILON
    }

    /// Checks if this is reverse playback.
    #[must_use]
    pub const fn is_reverse(self) -> bool {
        self.0 < 0.0
    }

    /// Applies speed to a duration.
    #[must_use]
    #[allow(clippy::cast_possible_truncation)]
    pub fn apply_to_duration(self, duration: Duration) -> Duration {
        Duration::new((duration.0 as f64 / self.0) as i64)
    }
}

impl Default for Speed {
    fn default() -> Self {
        Self::normal()
    }
}

impl fmt::Display for Speed {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}x", self.0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_position_basic() {
        let pos = Position::new(100);
        assert_eq!(pos.value(), 100);
        assert_eq!(pos.to_frames(Rational::new(1, 1)), 100);
    }

    #[test]
    fn test_position_arithmetic() {
        let pos1 = Position::new(100);
        let pos2 = Position::new(50);
        assert_eq!((pos1 + pos2).value(), 150);
        assert_eq!((pos1 - pos2).value(), 50);
    }

    #[test]
    fn test_position_with_duration() {
        let pos = Position::new(100);
        let dur = Duration::new(50);
        assert_eq!((pos + dur).value(), 150);
        assert_eq!((pos - dur).value(), 50);
    }

    #[test]
    fn test_position_to_seconds() {
        let pos = Position::new(24);
        let fps = Rational::new(1, 24);
        assert!((pos.to_seconds(fps) - 1.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_position_from_seconds() {
        let fps = Rational::new(1, 24);
        let pos = Position::from_seconds(1.0, fps);
        assert_eq!(pos.value(), 24);
    }

    #[test]
    fn test_duration_basic() {
        let dur = Duration::new(100);
        assert_eq!(dur.value(), 100);
        assert_eq!(dur.to_frames(), 100);
        assert!(!dur.is_zero());
    }

    #[test]
    fn test_duration_zero() {
        let dur = Duration::zero();
        assert!(dur.is_zero());
        assert_eq!(dur.value(), 0);
    }

    #[test]
    fn test_duration_arithmetic() {
        let dur1 = Duration::new(100);
        let dur2 = Duration::new(50);
        assert_eq!((dur1 + dur2).value(), 150);
        assert_eq!((dur1 - dur2).value(), 50);
    }

    #[test]
    fn test_duration_abs() {
        let dur = Duration::new(-100);
        assert_eq!(dur.abs().value(), 100);
    }

    #[test]
    fn test_duration_to_seconds() {
        let dur = Duration::new(48);
        let fps = Rational::new(1, 24);
        assert!((dur.to_seconds(fps) - 2.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_speed_normal() {
        let speed = Speed::normal();
        assert!(speed.is_normal());
        assert!(!speed.is_reverse());
        assert_eq!(speed.value(), 1.0);
    }

    #[test]
    fn test_speed_valid_range() {
        assert!(Speed::new(0.25).is_ok());
        assert!(Speed::new(1.0).is_ok());
        assert!(Speed::new(4.0).is_ok());
    }

    #[test]
    fn test_speed_invalid_range() {
        assert!(Speed::new(0.1).is_err());
        assert!(Speed::new(5.0).is_err());
        assert!(Speed::new(0.0).is_err());
    }

    #[test]
    fn test_speed_apply_to_duration() {
        let speed = Speed::new(2.0).expect("should succeed in test");
        let dur = Duration::new(100);
        let result = speed.apply_to_duration(dur);
        assert_eq!(result.value(), 50);
    }

    #[test]
    fn test_speed_reverse() {
        let speed = Speed::new(-1.0).expect("should succeed in test");
        assert!(speed.is_reverse());
        assert!(!speed.is_normal());
    }
}
