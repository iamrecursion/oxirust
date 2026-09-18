//! Timecode handling for timeline operations.

use oximedia_core::Rational;
use serde::{Deserialize, Serialize};
use std::fmt;

use crate::error::{TimelineError, TimelineResult};
use crate::types::Position;

/// Timecode format.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum TimecodeFormat {
    /// Non-drop-frame timecode (e.g., 24fps, 25fps, 30fps).
    NonDropFrame,
    /// Drop-frame timecode (e.g., 29.97fps, 59.94fps).
    /// Drops frame numbers 0 and 1 every minute except every 10th minute.
    DropFrame,
}

impl TimecodeFormat {
    /// Checks if a frame rate requires drop-frame timecode.
    #[must_use]
    pub fn from_frame_rate(frame_rate: Rational) -> Self {
        // Check if this is a 29.97 or 59.94 frame rate
        let fps = frame_rate.to_f64();
        if (fps - 29.97).abs() < 0.01 || (fps - 59.94).abs() < 0.01 {
            Self::DropFrame
        } else {
            Self::NonDropFrame
        }
    }

    /// Checks if this format is drop-frame.
    #[must_use]
    pub const fn is_drop_frame(self) -> bool {
        matches!(self, Self::DropFrame)
    }
}

/// Timecode value (hours:minutes:seconds:frames).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct TimecodeValue {
    /// Hours component (0-23).
    pub hours: u32,
    /// Minutes component (0-59).
    pub minutes: u32,
    /// Seconds component (0-59).
    pub seconds: u32,
    /// Frames component (0-fps).
    pub frames: u32,
}

impl TimecodeValue {
    /// Creates a new timecode value.
    #[must_use]
    pub const fn new(hours: u32, minutes: u32, seconds: u32, frames: u32) -> Self {
        Self {
            hours,
            minutes,
            seconds,
            frames,
        }
    }

    /// Creates timecode from frame number.
    #[must_use]
    #[allow(clippy::cast_possible_truncation)]
    pub fn from_frames(frame_number: i64, frame_rate: Rational, format: TimecodeFormat) -> Self {
        if frame_number < 0 {
            return Self::new(0, 0, 0, 0);
        }

        let fps = frame_rate.to_f64().round() as u32;
        let frame_number = frame_number as u32;

        if format.is_drop_frame() {
            Self::from_frames_drop(frame_number, fps)
        } else {
            Self::from_frames_non_drop(frame_number, fps)
        }
    }

    /// Converts non-drop-frame timecode.
    #[must_use]
    fn from_frames_non_drop(frame_number: u32, fps: u32) -> Self {
        let frames_per_minute = fps * 60;
        let frames_per_hour = frames_per_minute * 60;

        let hours = frame_number / frames_per_hour;
        let remainder = frame_number % frames_per_hour;

        let minutes = remainder / frames_per_minute;
        let remainder = remainder % frames_per_minute;

        let seconds = remainder / fps;
        let frames = remainder % fps;

        Self::new(hours, minutes, seconds, frames)
    }

    /// Converts drop-frame timecode.
    #[must_use]
    #[allow(clippy::cast_possible_truncation)]
    fn from_frames_drop(frame_number: u32, fps: u32) -> Self {
        // Drop-frame timecode: skip frames 0 and 1 every minute except every 10th minute
        let drop_frames = if fps > 30 { 4 } else { 2 };
        let frames_per_minute = fps * 60 - drop_frames;
        let frames_per_10min = frames_per_minute * 10 + drop_frames;
        let frames_per_hour = frames_per_10min * 6;

        let hours = frame_number / frames_per_hour;
        let mut remainder = frame_number % frames_per_hour;

        let ten_minutes = remainder / frames_per_10min;
        remainder %= frames_per_10min;

        if remainder >= drop_frames {
            remainder += drop_frames;
        }

        let minutes = remainder / (fps * 60);
        remainder %= fps * 60;

        let seconds = remainder / fps;
        let frames = remainder % fps;

        Self::new(hours, ten_minutes * 10 + minutes, seconds, frames)
    }

    /// Converts timecode to frame number.
    #[must_use]
    #[allow(clippy::cast_possible_wrap)]
    pub fn to_frames(self, frame_rate: Rational, format: TimecodeFormat) -> i64 {
        let fps = frame_rate.to_f64().round() as u32;

        if format.is_drop_frame() {
            i64::from(self.to_frames_drop(fps))
        } else {
            i64::from(self.to_frames_non_drop(fps))
        }
    }

    /// Converts to frames using non-drop-frame.
    #[must_use]
    fn to_frames_non_drop(self, fps: u32) -> u32 {
        let frames_per_minute = fps * 60;
        let frames_per_hour = frames_per_minute * 60;

        self.hours * frames_per_hour
            + self.minutes * frames_per_minute
            + self.seconds * fps
            + self.frames
    }

    /// Converts to frames using drop-frame.
    #[must_use]
    fn to_frames_drop(self, fps: u32) -> u32 {
        let drop_frames = if fps > 30 { 4 } else { 2 };
        let frames_per_minute = fps * 60;

        let total_minutes = self.hours * 60 + self.minutes;
        let dropped = drop_frames * (total_minutes - total_minutes / 10);

        self.hours * frames_per_minute * 60
            + self.minutes * frames_per_minute
            + self.seconds * fps
            + self.frames
            - dropped
    }

    /// Parses timecode from string (HH:MM:SS:FF or HH:MM:SS;FF for drop-frame).
    ///
    /// # Errors
    ///
    /// Returns error if the string format is invalid.
    pub fn parse(s: &str) -> TimelineResult<(Self, TimecodeFormat)> {
        let is_drop = s.contains(';');
        let format = if is_drop {
            TimecodeFormat::DropFrame
        } else {
            TimecodeFormat::NonDropFrame
        };

        // For drop-frame "HH:MM:SS;FF", normalize by replacing ';' with ':' then split on ':'
        let normalized = s.replace(';', ":");
        let parts: Vec<&str> = normalized.split(':').collect();
        if parts.len() != 4 {
            return Err(TimelineError::InvalidTimecode(format!(
                "Expected HH:MM:SS:FF format, got: {s}"
            )));
        }

        let hours = parts[0]
            .parse()
            .map_err(|_| TimelineError::InvalidTimecode(format!("Invalid hours: {}", parts[0])))?;
        let minutes = parts[1].parse().map_err(|_| {
            TimelineError::InvalidTimecode(format!("Invalid minutes: {}", parts[1]))
        })?;
        let seconds = parts[2].parse().map_err(|_| {
            TimelineError::InvalidTimecode(format!("Invalid seconds: {}", parts[2]))
        })?;
        let frames = parts[3]
            .parse()
            .map_err(|_| TimelineError::InvalidTimecode(format!("Invalid frames: {}", parts[3])))?;

        Ok((Self::new(hours, minutes, seconds, frames), format))
    }

    /// Formats timecode as string.
    #[must_use]
    pub fn format(self, format: TimecodeFormat) -> String {
        let separator = if format.is_drop_frame() { ';' } else { ':' };
        format!(
            "{:02}:{:02}:{:02}{}{:02}",
            self.hours, self.minutes, self.seconds, separator, self.frames
        )
    }
}

impl fmt::Display for TimecodeValue {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "{:02}:{:02}:{:02}:{:02}",
            self.hours, self.minutes, self.seconds, self.frames
        )
    }
}

/// Converts position to timecode.
#[must_use]
pub fn position_to_timecode(
    position: Position,
    frame_rate: Rational,
    format: TimecodeFormat,
) -> TimecodeValue {
    TimecodeValue::from_frames(position.value(), frame_rate, format)
}

/// Converts timecode to position.
#[must_use]
pub fn timecode_to_position(
    timecode: TimecodeValue,
    frame_rate: Rational,
    format: TimecodeFormat,
) -> Position {
    Position::new(timecode.to_frames(frame_rate, format))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_timecode_format_from_frame_rate() {
        let fps_24 = Rational::new(24, 1);
        assert_eq!(
            TimecodeFormat::from_frame_rate(fps_24),
            TimecodeFormat::NonDropFrame
        );

        let fps_2997 = Rational::new(30000, 1001);
        assert_eq!(
            TimecodeFormat::from_frame_rate(fps_2997),
            TimecodeFormat::DropFrame
        );
    }

    #[test]
    fn test_timecode_value_non_drop() {
        let fps = Rational::new(24, 1);
        let tc = TimecodeValue::from_frames(100, fps, TimecodeFormat::NonDropFrame);
        assert_eq!(tc.hours, 0);
        assert_eq!(tc.minutes, 0);
        assert_eq!(tc.seconds, 4);
        assert_eq!(tc.frames, 4);
    }

    #[test]
    fn test_timecode_value_to_frames() {
        let tc = TimecodeValue::new(0, 1, 30, 15);
        let fps = Rational::new(24, 1);
        let frames = tc.to_frames(fps, TimecodeFormat::NonDropFrame);
        assert_eq!(frames, 24 * 90 + 15);
    }

    #[test]
    fn test_timecode_roundtrip() {
        let fps = Rational::new(30, 1);
        let original_frames = 5000;
        let tc = TimecodeValue::from_frames(original_frames, fps, TimecodeFormat::NonDropFrame);
        let converted_frames = tc.to_frames(fps, TimecodeFormat::NonDropFrame);
        assert_eq!(original_frames, converted_frames);
    }

    #[test]
    fn test_timecode_parse_non_drop() {
        let (tc, format) = TimecodeValue::parse("01:23:45:12").expect("should succeed in test");
        assert_eq!(tc.hours, 1);
        assert_eq!(tc.minutes, 23);
        assert_eq!(tc.seconds, 45);
        assert_eq!(tc.frames, 12);
        assert_eq!(format, TimecodeFormat::NonDropFrame);
    }

    #[test]
    fn test_timecode_parse_drop() {
        let (tc, format) = TimecodeValue::parse("01:23:45;12").expect("should succeed in test");
        assert_eq!(tc.hours, 1);
        assert_eq!(tc.minutes, 23);
        assert_eq!(tc.seconds, 45);
        assert_eq!(tc.frames, 12);
        assert_eq!(format, TimecodeFormat::DropFrame);
    }

    #[test]
    fn test_timecode_parse_invalid() {
        assert!(TimecodeValue::parse("invalid").is_err());
        assert!(TimecodeValue::parse("12:34").is_err());
        assert!(TimecodeValue::parse("aa:bb:cc:dd").is_err());
    }

    #[test]
    fn test_timecode_format_non_drop() {
        let tc = TimecodeValue::new(1, 23, 45, 12);
        let formatted = tc.format(TimecodeFormat::NonDropFrame);
        assert_eq!(formatted, "01:23:45:12");
    }

    #[test]
    fn test_timecode_format_drop() {
        let tc = TimecodeValue::new(1, 23, 45, 12);
        let formatted = tc.format(TimecodeFormat::DropFrame);
        assert_eq!(formatted, "01:23:45;12");
    }

    #[test]
    fn test_position_to_timecode() {
        let pos = Position::new(100);
        let fps = Rational::new(24, 1);
        let tc = position_to_timecode(pos, fps, TimecodeFormat::NonDropFrame);
        assert_eq!(tc.seconds, 4);
        assert_eq!(tc.frames, 4);
    }

    #[test]
    fn test_timecode_to_position() {
        let tc = TimecodeValue::new(0, 0, 1, 0);
        let fps = Rational::new(24, 1);
        let pos = timecode_to_position(tc, fps, TimecodeFormat::NonDropFrame);
        assert_eq!(pos.value(), 24);
    }

    #[test]
    fn test_timecode_display() {
        let tc = TimecodeValue::new(1, 2, 3, 4);
        assert_eq!(format!("{tc}"), "01:02:03:04");
    }

    #[test]
    fn test_drop_frame_calculation() {
        // Test 29.97 fps drop-frame
        let fps = Rational::new(30000, 1001);
        let tc = TimecodeValue::new(0, 1, 0, 2);
        let frames = tc.to_frames(fps, TimecodeFormat::DropFrame);
        // First minute drops frames 0 and 1, so frame 2 of minute 1 is actually frame 1800-2+2 = 1800
        assert_eq!(frames, 1800);
    }

    #[test]
    fn test_negative_frames() {
        let fps = Rational::new(24, 1);
        let tc = TimecodeValue::from_frames(-100, fps, TimecodeFormat::NonDropFrame);
        assert_eq!(tc, TimecodeValue::new(0, 0, 0, 0));
    }
}
