#![allow(dead_code)]
//! Frame rate conversion and timecode adaptation for conform workflows.
//!
//! Handles converting timecodes and durations when source and target frame
//! rates differ, including pull-down, pull-up, and drop-frame adjustments.

/// Standard frame rates used in production and broadcast.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum StandardRate {
    /// 23.976 fps (NTSC film).
    Fps23_976,
    /// 24 fps (true cinema).
    Fps24,
    /// 25 fps (PAL / SECAM).
    Fps25,
    /// 29.97 fps (NTSC video).
    Fps29_97,
    /// 30 fps (progressive NTSC).
    Fps30,
    /// 48 fps (HFR cinema).
    Fps48,
    /// 50 fps (PAL progressive / HFR).
    Fps50,
    /// 59.94 fps (NTSC progressive).
    Fps59_94,
    /// 60 fps (progressive).
    Fps60,
}

impl StandardRate {
    /// Return the frame rate as a floating-point value.
    #[must_use]
    #[allow(clippy::cast_precision_loss)]
    pub fn as_f64(self) -> f64 {
        match self {
            Self::Fps23_976 => 24000.0 / 1001.0,
            Self::Fps24 => 24.0,
            Self::Fps25 => 25.0,
            Self::Fps29_97 => 30000.0 / 1001.0,
            Self::Fps30 => 30.0,
            Self::Fps48 => 48.0,
            Self::Fps50 => 50.0,
            Self::Fps59_94 => 60000.0 / 1001.0,
            Self::Fps60 => 60.0,
        }
    }

    /// Whether this rate uses drop-frame timecode convention.
    #[must_use]
    pub const fn is_drop_frame(self) -> bool {
        matches!(self, Self::Fps29_97 | Self::Fps59_94)
    }

    /// Frame duration in seconds.
    #[must_use]
    pub fn frame_duration_secs(self) -> f64 {
        1.0 / self.as_f64()
    }
}

/// A timecode value with hours, minutes, seconds, and frames.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SimpleTimecode {
    /// Hours component.
    pub hours: u32,
    /// Minutes component.
    pub minutes: u32,
    /// Seconds component.
    pub seconds: u32,
    /// Frames component.
    pub frames: u32,
}

impl SimpleTimecode {
    /// Create a new timecode.
    #[must_use]
    pub const fn new(hours: u32, minutes: u32, seconds: u32, frames: u32) -> Self {
        Self {
            hours,
            minutes,
            seconds,
            frames,
        }
    }

    /// Convert this timecode to a total frame count at the given rate.
    #[must_use]
    #[allow(clippy::cast_precision_loss)]
    #[allow(clippy::cast_possible_truncation)]
    #[allow(clippy::cast_sign_loss)]
    pub fn to_frame_count(self, rate: StandardRate) -> u64 {
        let fps = rate.as_f64().round() as u64;
        let total_secs =
            u64::from(self.hours) * 3600 + u64::from(self.minutes) * 60 + u64::from(self.seconds);
        let raw_frames = total_secs * fps + u64::from(self.frames);

        if rate.is_drop_frame() {
            // Drop-frame: skip 2 frames every minute except every 10th minute
            let total_minutes = u64::from(self.hours) * 60 + u64::from(self.minutes);
            let drop_count = 2 * (total_minutes - total_minutes / 10);
            raw_frames.saturating_sub(drop_count)
        } else {
            raw_frames
        }
    }

    /// Create a timecode from a total frame count at the given rate.
    #[must_use]
    #[allow(clippy::cast_precision_loss)]
    #[allow(clippy::cast_possible_truncation)]
    pub fn from_frame_count(mut frame_count: u64, rate: StandardRate) -> Self {
        let fps = rate.as_f64().round() as u64;

        if rate.is_drop_frame() {
            // Reverse drop-frame calculation
            let frames_per_10min = fps * 60 * 10 - 2 * 9;
            let d = frame_count / frames_per_10min;
            let m = frame_count % frames_per_10min;

            let extra = if m > 2 { (m - 2) / (fps * 60 - 2) } else { 0 };
            frame_count += 2 * (9 * d + extra);
        }

        let total_secs = frame_count / fps;
        let frames = (frame_count % fps) as u32;
        let seconds = (total_secs % 60) as u32;
        let total_mins = total_secs / 60;
        let minutes = (total_mins % 60) as u32;
        let hours = (total_mins / 60) as u32;

        Self {
            hours,
            minutes,
            seconds,
            frames,
        }
    }
}

impl std::fmt::Display for SimpleTimecode {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "{:02}:{:02}:{:02}:{:02}",
            self.hours, self.minutes, self.seconds, self.frames
        )
    }
}

/// A frame-rate conversion specification.
#[derive(Debug, Clone, Copy)]
pub struct RateConversion {
    /// Source frame rate.
    pub source: StandardRate,
    /// Target frame rate.
    pub target: StandardRate,
}

impl RateConversion {
    /// Create a new rate conversion.
    #[must_use]
    pub const fn new(source: StandardRate, target: StandardRate) -> Self {
        Self { source, target }
    }

    /// The speed factor: `target_fps` / `source_fps`.
    #[must_use]
    pub fn speed_factor(&self) -> f64 {
        self.target.as_f64() / self.source.as_f64()
    }

    /// Convert a frame number from source rate to target rate.
    #[must_use]
    #[allow(clippy::cast_precision_loss)]
    #[allow(clippy::cast_possible_truncation)]
    #[allow(clippy::cast_sign_loss)]
    pub fn convert_frame(&self, source_frame: u64) -> u64 {
        let source_time_sec = source_frame as f64 / self.source.as_f64();
        let target_frame = source_time_sec * self.target.as_f64();
        target_frame.round() as u64
    }

    /// Convert a duration in milliseconds from source to target rate.
    #[must_use]
    pub fn convert_duration_ms(&self, source_ms: u64) -> u64 {
        // Duration stays the same in real time; only frame numbers change.
        source_ms
    }

    /// Convert a timecode from source to target rate.
    #[must_use]
    pub fn convert_timecode(&self, tc: SimpleTimecode) -> SimpleTimecode {
        let source_frames = tc.to_frame_count(self.source);
        let target_frames = self.convert_frame(source_frames);
        SimpleTimecode::from_frame_count(target_frames, self.target)
    }

    /// Whether this conversion is a pull-down (increasing frame rate).
    #[must_use]
    pub fn is_pulldown(&self) -> bool {
        self.target.as_f64() > self.source.as_f64()
    }

    /// Whether this conversion is a pull-up (decreasing frame rate).
    #[must_use]
    pub fn is_pullup(&self) -> bool {
        self.target.as_f64() < self.source.as_f64()
    }

    /// Whether source and target rates are identical.
    #[must_use]
    pub fn is_identity(&self) -> bool {
        (self.source.as_f64() - self.target.as_f64()).abs() < 0.001
    }
}

/// Convert a frame count to a time in seconds.
#[must_use]
#[allow(clippy::cast_precision_loss)]
pub fn frames_to_seconds(frame_count: u64, rate: StandardRate) -> f64 {
    frame_count as f64 / rate.as_f64()
}

/// Convert a time in seconds to a frame count.
#[must_use]
#[allow(clippy::cast_precision_loss)]
#[allow(clippy::cast_possible_truncation)]
#[allow(clippy::cast_sign_loss)]
pub fn seconds_to_frames(seconds: f64, rate: StandardRate) -> u64 {
    (seconds * rate.as_f64()).round() as u64
}

/// Calculate the number of frames that would be duplicated or dropped
/// in a simple nearest-frame conversion.
#[must_use]
#[allow(clippy::cast_precision_loss)]
#[allow(clippy::cast_possible_truncation)]
#[allow(clippy::cast_sign_loss)]
pub fn frame_diff_count(duration_secs: f64, source: StandardRate, target: StandardRate) -> i64 {
    let source_frames = (duration_secs * source.as_f64()).round() as i64;
    let target_frames = (duration_secs * target.as_f64()).round() as i64;
    target_frames - source_frames
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_standard_rate_values() {
        assert!((StandardRate::Fps24.as_f64() - 24.0).abs() < 0.001);
        assert!((StandardRate::Fps25.as_f64() - 25.0).abs() < 0.001);
        assert!((StandardRate::Fps23_976.as_f64() - 23.976).abs() < 0.001);
        assert!((StandardRate::Fps29_97.as_f64() - 29.97).abs() < 0.01);
    }

    #[test]
    fn test_drop_frame_flag() {
        assert!(StandardRate::Fps29_97.is_drop_frame());
        assert!(StandardRate::Fps59_94.is_drop_frame());
        assert!(!StandardRate::Fps24.is_drop_frame());
        assert!(!StandardRate::Fps25.is_drop_frame());
    }

    #[test]
    fn test_timecode_display() {
        let tc = SimpleTimecode::new(1, 2, 3, 4);
        assert_eq!(format!("{tc}"), "01:02:03:04");
    }

    #[test]
    fn test_timecode_to_frames_non_drop() {
        let tc = SimpleTimecode::new(0, 1, 0, 0);
        let frames = tc.to_frame_count(StandardRate::Fps24);
        assert_eq!(frames, 24 * 60);
    }

    #[test]
    fn test_timecode_roundtrip() {
        let original = SimpleTimecode::new(1, 30, 45, 12);
        let frames = original.to_frame_count(StandardRate::Fps24);
        let recovered = SimpleTimecode::from_frame_count(frames, StandardRate::Fps24);
        assert_eq!(original, recovered);
    }

    #[test]
    fn test_conversion_identity() {
        let conv = RateConversion::new(StandardRate::Fps25, StandardRate::Fps25);
        assert!(conv.is_identity());
        assert_eq!(conv.convert_frame(100), 100);
    }

    #[test]
    fn test_conversion_pulldown() {
        let conv = RateConversion::new(StandardRate::Fps24, StandardRate::Fps25);
        assert!(conv.is_pulldown());
        assert!(!conv.is_pullup());
    }

    #[test]
    fn test_conversion_pullup() {
        let conv = RateConversion::new(StandardRate::Fps25, StandardRate::Fps24);
        assert!(conv.is_pullup());
        assert!(!conv.is_pulldown());
    }

    #[test]
    fn test_convert_frame_24_to_25() {
        let conv = RateConversion::new(StandardRate::Fps24, StandardRate::Fps25);
        // Frame 24 at 24fps = 1.0 second. At 25fps, 1.0 second = frame 25.
        let result = conv.convert_frame(24);
        assert_eq!(result, 25);
    }

    #[test]
    fn test_frames_to_seconds() {
        let secs = frames_to_seconds(48, StandardRate::Fps24);
        assert!((secs - 2.0).abs() < 0.001);
    }

    #[test]
    fn test_seconds_to_frames() {
        let frames = seconds_to_frames(2.0, StandardRate::Fps24);
        assert_eq!(frames, 48);
    }

    #[test]
    fn test_frame_diff_count() {
        // 10 seconds: 240 frames @24fps, 250 frames @25fps => diff = 10
        let diff = frame_diff_count(10.0, StandardRate::Fps24, StandardRate::Fps25);
        assert_eq!(diff, 10);
    }

    #[test]
    fn test_speed_factor() {
        let conv = RateConversion::new(StandardRate::Fps24, StandardRate::Fps48);
        assert!((conv.speed_factor() - 2.0).abs() < 0.001);
    }

    #[test]
    fn test_frame_duration() {
        let dur = StandardRate::Fps25.frame_duration_secs();
        assert!((dur - 0.04).abs() < 0.001);
    }
}
