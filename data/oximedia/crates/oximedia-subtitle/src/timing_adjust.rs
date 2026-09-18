//! Subtitle timing adjustment utilities.
//!
//! Provides global time-shift, playback-speed scaling, and frame-rate conversion
//! for subtitle cue timestamps.

#![allow(dead_code)]
#![allow(clippy::cast_precision_loss)]

/// Direction and magnitude of a global time shift.
#[derive(Clone, Debug, PartialEq)]
pub struct TimeShift {
    /// Millisecond offset applied to every cue (positive = later, negative = earlier).
    pub offset_ms: i64,
}

impl TimeShift {
    /// Create a new `TimeShift`.
    #[must_use]
    pub fn new(offset_ms: i64) -> Self {
        Self { offset_ms }
    }

    /// Apply this shift to a single timestamp.
    #[must_use]
    pub fn apply(&self, timestamp_ms: i64) -> i64 {
        timestamp_ms + self.offset_ms
    }

    /// Apply this shift, clamping the result to zero or above.
    #[must_use]
    pub fn apply_clamped(&self, timestamp_ms: i64) -> i64 {
        (timestamp_ms + self.offset_ms).max(0)
    }
}

/// Adjust timing by a speed factor (e.g. 1.25 = 25% faster playback).
///
/// Both start and end times are scaled from the origin so relative
/// durations are preserved.
#[derive(Clone, Debug, PartialEq)]
pub struct SpeedAdjust {
    /// Playback speed factor (> 0).  1.0 = no change.
    pub factor: f64,
    /// Origin (ms) that remains fixed during the scale.  Usually 0.
    pub origin_ms: i64,
}

impl SpeedAdjust {
    /// Create a new `SpeedAdjust`.
    ///
    /// # Panics
    ///
    /// Panics if `factor` is not finite or is <= 0.
    #[must_use]
    pub fn new(factor: f64, origin_ms: i64) -> Self {
        assert!(
            factor > 0.0 && factor.is_finite(),
            "factor must be positive and finite"
        );
        Self { factor, origin_ms }
    }

    /// Scale a single timestamp around `origin_ms`.
    #[must_use]
    pub fn apply(&self, timestamp_ms: i64) -> i64 {
        let relative = (timestamp_ms - self.origin_ms) as f64;
        let scaled = relative / self.factor;
        self.origin_ms + scaled.round() as i64
    }
}

/// Frame-rate conversion parameters.
///
/// Converts timestamps from one frame-rate to another when the video is
/// re-encoded at a different frame rate.
#[derive(Clone, Debug, PartialEq)]
pub struct FrameRateConvert {
    /// Source frames per second.
    pub src_fps: f64,
    /// Destination frames per second.
    pub dst_fps: f64,
}

impl FrameRateConvert {
    /// Create a new `FrameRateConvert`.
    ///
    /// # Panics
    ///
    /// Panics if either fps value is not finite or is <= 0.
    #[must_use]
    pub fn new(src_fps: f64, dst_fps: f64) -> Self {
        assert!(
            src_fps > 0.0 && src_fps.is_finite(),
            "src_fps must be positive and finite"
        );
        assert!(
            dst_fps > 0.0 && dst_fps.is_finite(),
            "dst_fps must be positive and finite"
        );
        Self { src_fps, dst_fps }
    }

    /// Convert a single timestamp (ms) from source to destination frame rate.
    #[must_use]
    pub fn convert(&self, timestamp_ms: i64) -> i64 {
        let ratio = self.dst_fps / self.src_fps;
        let scaled = timestamp_ms as f64 * ratio;
        scaled.round() as i64
    }

    /// Convert a timestamp measured in *frames* to milliseconds at the
    /// destination frame rate.
    #[must_use]
    pub fn frames_to_ms(&self, frame_number: u64) -> i64 {
        let ms = frame_number as f64 / self.dst_fps * 1000.0;
        ms.round() as i64
    }

    /// Convert a millisecond timestamp to a frame number at the destination
    /// frame rate.
    #[must_use]
    pub fn ms_to_frames(&self, timestamp_ms: i64) -> u64 {
        let frames = timestamp_ms as f64 / 1000.0 * self.dst_fps;
        frames.floor() as u64
    }
}

/// Apply a [`TimeShift`] to a list of (start, end) timestamp pairs.
#[allow(clippy::module_name_repetitions)]
#[must_use]
pub fn shift_timestamps(timestamps: &[(i64, i64)], shift: &TimeShift) -> Vec<(i64, i64)> {
    timestamps
        .iter()
        .map(|(s, e)| (shift.apply(*s), shift.apply(*e)))
        .collect()
}

/// Apply a [`SpeedAdjust`] to a list of (start, end) timestamp pairs.
#[must_use]
pub fn speed_adjust_timestamps(timestamps: &[(i64, i64)], adj: &SpeedAdjust) -> Vec<(i64, i64)> {
    timestamps
        .iter()
        .map(|(s, e)| (adj.apply(*s), adj.apply(*e)))
        .collect()
}

/// Apply a [`FrameRateConvert`] to a list of (start, end) timestamp pairs.
#[must_use]
pub fn convert_timestamps(timestamps: &[(i64, i64)], conv: &FrameRateConvert) -> Vec<(i64, i64)> {
    timestamps
        .iter()
        .map(|(s, e)| (conv.convert(*s), conv.convert(*e)))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_time_shift_positive() {
        let shift = TimeShift::new(500);
        assert_eq!(shift.apply(1000), 1500);
    }

    #[test]
    fn test_time_shift_negative() {
        let shift = TimeShift::new(-300);
        assert_eq!(shift.apply(1000), 700);
    }

    #[test]
    fn test_time_shift_clamped_no_underflow() {
        let shift = TimeShift::new(-5000);
        assert_eq!(shift.apply_clamped(1000), 0);
    }

    #[test]
    fn test_time_shift_clamped_positive() {
        let shift = TimeShift::new(200);
        assert_eq!(shift.apply_clamped(800), 1000);
    }

    #[test]
    fn test_shift_timestamps_list() {
        let pairs = vec![(1000, 4000), (5000, 8000)];
        let shift = TimeShift::new(500);
        let result = shift_timestamps(&pairs, &shift);
        assert_eq!(result, vec![(1500, 4500), (5500, 8500)]);
    }

    #[test]
    fn test_speed_adjust_double() {
        // Double speed: timestamps halve.
        let adj = SpeedAdjust::new(2.0, 0);
        assert_eq!(adj.apply(2000), 1000);
    }

    #[test]
    fn test_speed_adjust_half_speed() {
        // Half speed: timestamps double.
        let adj = SpeedAdjust::new(0.5, 0);
        assert_eq!(adj.apply(1000), 2000);
    }

    #[test]
    fn test_speed_adjust_identity() {
        let adj = SpeedAdjust::new(1.0, 0);
        assert_eq!(adj.apply(1234), 1234);
    }

    #[test]
    fn test_speed_adjust_with_origin() {
        let adj = SpeedAdjust::new(2.0, 1000);
        // 2000 - 1000 = 1000 relative; 1000 / 2 = 500; 1000 + 500 = 1500
        assert_eq!(adj.apply(2000), 1500);
    }

    #[test]
    fn test_frame_rate_convert_same_rate() {
        let conv = FrameRateConvert::new(25.0, 25.0);
        assert_eq!(conv.convert(1000), 1000);
    }

    #[test]
    fn test_frame_rate_convert_24_to_25() {
        let conv = FrameRateConvert::new(24.0, 25.0);
        // 1000 ms * (25/24) ≈ 1042 ms
        let result = conv.convert(1000);
        assert!((result - 1042).abs() <= 1);
    }

    #[test]
    fn test_frame_rate_frames_to_ms() {
        let conv = FrameRateConvert::new(25.0, 25.0);
        // Frame 25 at 25 fps = 1000 ms
        assert_eq!(conv.frames_to_ms(25), 1000);
    }

    #[test]
    fn test_frame_rate_ms_to_frames() {
        let conv = FrameRateConvert::new(25.0, 25.0);
        // 1000 ms at 25 fps = frame 25
        assert_eq!(conv.ms_to_frames(1000), 25);
    }

    #[test]
    fn test_convert_timestamps_list() {
        let pairs = vec![(0, 1000), (2000, 3000)];
        let conv = FrameRateConvert::new(25.0, 25.0);
        let result = convert_timestamps(&pairs, &conv);
        assert_eq!(result, vec![(0, 1000), (2000, 3000)]);
    }

    #[test]
    fn test_speed_adjust_timestamps_list() {
        let pairs = vec![(0, 2000), (4000, 6000)];
        let adj = SpeedAdjust::new(2.0, 0);
        let result = speed_adjust_timestamps(&pairs, &adj);
        assert_eq!(result, vec![(0, 1000), (2000, 3000)]);
    }
}
