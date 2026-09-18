//! Audio clock and timestamp utilities.
//!
//! Provides sample-accurate clock tracking, drift detection/compensation between
//! two clock sources, and frame-to-PTS conversion helpers.

#![allow(dead_code)]
#![allow(clippy::cast_precision_loss)]

/// A sample-accurate audio clock.
///
/// Tracks playback position in samples, converting to presentation timestamps
/// (PTS) in microseconds on demand.
#[derive(Debug, Clone)]
pub struct AudioClock {
    sample_rate: u32,
    position: u64,
}

impl AudioClock {
    /// Create a new clock at position 0.
    ///
    /// # Parameters
    /// - `sample_rate`: samples per second (e.g. 48_000).
    pub fn new(sample_rate: u32) -> Self {
        Self {
            sample_rate,
            position: 0,
        }
    }

    /// Advance the clock by `frames` samples.
    pub fn tick(&mut self, frames: u64) {
        self.position += frames;
    }

    /// Returns the current sample position.
    pub fn sample_position(&self) -> u64 {
        self.position
    }

    /// Returns the current presentation timestamp in microseconds.
    #[allow(clippy::cast_precision_loss)]
    pub fn pts_us(&self) -> i64 {
        if self.sample_rate == 0 {
            return 0;
        }
        // position / sample_rate * 1_000_000
        let us = (self.position as f64 / self.sample_rate as f64) * 1_000_000.0;
        us as i64
    }

    /// Reset the clock to position 0.
    pub fn reset(&mut self) {
        self.position = 0;
    }

    /// Returns the sample rate this clock was created with.
    pub fn sample_rate(&self) -> u32 {
        self.sample_rate
    }
}

/// Synchronises two [`AudioClock`]s and reports drift.
///
/// Used for detecting when a hardware clock drifts from the software clock,
/// enabling sample-rate conversion or stuffing/dropping of frames to compensate.
#[derive(Debug, Clone)]
pub struct AudioClockSync {
    reference_rate: u32,
    measured_rate: u32,
    /// Accumulated sample count from the reference clock.
    ref_samples: u64,
    /// Accumulated sample count from the measured clock.
    meas_samples: u64,
}

impl AudioClockSync {
    /// Create a sync tracker comparing a reference rate and measured rate.
    pub fn new(reference_rate: u32, measured_rate: u32) -> Self {
        Self {
            reference_rate,
            measured_rate,
            ref_samples: 0,
            meas_samples: 0,
        }
    }

    /// Record that `frames` samples elapsed on the reference clock.
    pub fn advance_reference(&mut self, frames: u64) {
        self.ref_samples += frames;
    }

    /// Record that `frames` samples elapsed on the measured clock.
    pub fn advance_measured(&mut self, frames: u64) {
        self.meas_samples += frames;
    }

    /// Returns the drift in samples: positive means the measured clock is ahead,
    /// negative means it is behind the reference.
    #[allow(clippy::cast_precision_loss)]
    pub fn drift_samples(&self) -> i64 {
        if self.reference_rate == 0 || self.measured_rate == 0 {
            return 0;
        }
        // Convert both accumulators to the reference timeline
        let ref_us = (self.ref_samples as f64 / self.reference_rate as f64) * 1_000_000.0;
        let meas_us = (self.meas_samples as f64 / self.measured_rate as f64) * 1_000_000.0;
        let drift_us = meas_us - ref_us;
        // Convert drift_us back to reference-rate samples
        ((drift_us / 1_000_000.0) * self.reference_rate as f64) as i64
    }

    /// Returns a compensation factor in PPM (parts per million) to apply to the
    /// measured clock so it matches the reference.
    ///
    /// Positive PPM means speed up the measured clock.
    #[allow(clippy::cast_precision_loss)]
    pub fn compensate(&self) -> f64 {
        if self.meas_samples == 0 || self.measured_rate == 0 || self.reference_rate == 0 {
            return 0.0;
        }
        // Ideal measured samples for the reference duration
        let ref_dur_s = self.ref_samples as f64 / self.reference_rate as f64;
        let ideal_meas = ref_dur_s * self.measured_rate as f64;
        let actual_meas = self.meas_samples as f64;
        if actual_meas == 0.0 {
            return 0.0;
        }
        // PPM = (ideal - actual) / actual * 1_000_000
        (ideal_meas - actual_meas) / actual_meas * 1_000_000.0
    }

    /// Reset accumulated counters.
    pub fn reset(&mut self) {
        self.ref_samples = 0;
        self.meas_samples = 0;
    }
}

/// Utility for converting between frame counts and PTS in various units.
#[derive(Debug, Clone, Copy)]
pub struct AudioTimestampCalc {
    sample_rate: u32,
}

impl AudioTimestampCalc {
    /// Create a calculator for a given sample rate.
    pub fn new(sample_rate: u32) -> Self {
        Self { sample_rate }
    }

    /// Convert a frame (sample) count to a PTS value in microseconds.
    #[allow(clippy::cast_precision_loss)]
    pub fn frames_to_pts_us(&self, frames: u64) -> i64 {
        if self.sample_rate == 0 {
            return 0;
        }
        let us = (frames as f64 / self.sample_rate as f64) * 1_000_000.0;
        us as i64
    }

    /// Convert a PTS in microseconds to the nearest frame count.
    #[allow(clippy::cast_precision_loss)]
    pub fn pts_us_to_frames(&self, pts_us: i64) -> u64 {
        if self.sample_rate == 0 {
            return 0;
        }
        let frames = (pts_us as f64 / 1_000_000.0) * self.sample_rate as f64;
        frames.round() as u64
    }

    /// Convert frame count to milliseconds.
    #[allow(clippy::cast_precision_loss)]
    pub fn frames_to_ms(&self, frames: u64) -> f64 {
        if self.sample_rate == 0 {
            return 0.0;
        }
        (frames as f64 / self.sample_rate as f64) * 1_000.0
    }

    /// Returns the sample rate used by this calculator.
    pub fn sample_rate(&self) -> u32 {
        self.sample_rate
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_clock_initial_position() {
        let clock = AudioClock::new(48_000);
        assert_eq!(clock.sample_position(), 0);
    }

    #[test]
    fn test_clock_tick_advances_position() {
        let mut clock = AudioClock::new(48_000);
        clock.tick(1024);
        assert_eq!(clock.sample_position(), 1024);
    }

    #[test]
    fn test_clock_pts_us_zero_at_start() {
        let clock = AudioClock::new(48_000);
        assert_eq!(clock.pts_us(), 0);
    }

    #[test]
    fn test_clock_pts_us_one_second() {
        let mut clock = AudioClock::new(48_000);
        clock.tick(48_000);
        // Should be 1_000_000 µs = 1 second
        assert!((clock.pts_us() - 1_000_000).abs() < 10);
    }

    #[test]
    fn test_clock_reset() {
        let mut clock = AudioClock::new(48_000);
        clock.tick(9999);
        clock.reset();
        assert_eq!(clock.sample_position(), 0);
    }

    #[test]
    fn test_clock_sample_rate() {
        let clock = AudioClock::new(44_100);
        assert_eq!(clock.sample_rate(), 44_100);
    }

    #[test]
    fn test_sync_no_drift_when_equal() {
        let mut sync = AudioClockSync::new(48_000, 48_000);
        sync.advance_reference(48_000);
        sync.advance_measured(48_000);
        assert_eq!(sync.drift_samples(), 0);
    }

    #[test]
    fn test_sync_drift_positive_when_measured_ahead() {
        let mut sync = AudioClockSync::new(48_000, 48_000);
        sync.advance_reference(48_000);
        // Measured has one extra frame
        sync.advance_measured(48_001);
        let drift = sync.drift_samples();
        assert!(drift > 0, "Expected positive drift, got {drift}");
    }

    #[test]
    fn test_sync_compensate_zero_when_equal() {
        let mut sync = AudioClockSync::new(48_000, 48_000);
        sync.advance_reference(48_000);
        sync.advance_measured(48_000);
        let ppm = sync.compensate();
        assert!(ppm.abs() < 1.0, "Expected near-zero PPM, got {ppm}");
    }

    #[test]
    fn test_sync_reset() {
        let mut sync = AudioClockSync::new(48_000, 48_000);
        sync.advance_reference(1000);
        sync.advance_measured(1000);
        sync.reset();
        assert_eq!(sync.drift_samples(), 0);
    }

    #[test]
    fn test_timestamp_calc_frames_to_pts_us() {
        let calc = AudioTimestampCalc::new(48_000);
        // 48000 frames = 1 second = 1_000_000 µs
        assert!((calc.frames_to_pts_us(48_000) - 1_000_000).abs() < 10);
    }

    #[test]
    fn test_timestamp_calc_pts_us_to_frames() {
        let calc = AudioTimestampCalc::new(48_000);
        assert_eq!(calc.pts_us_to_frames(1_000_000), 48_000);
    }

    #[test]
    fn test_timestamp_calc_frames_to_ms() {
        let calc = AudioTimestampCalc::new(48_000);
        let ms = calc.frames_to_ms(48_000);
        assert!((ms - 1000.0).abs() < 0.01);
    }

    #[test]
    fn test_timestamp_calc_zero_sample_rate() {
        let calc = AudioTimestampCalc::new(0);
        assert_eq!(calc.frames_to_pts_us(1000), 0);
        assert_eq!(calc.pts_us_to_frames(1_000_000), 0);
    }
}
