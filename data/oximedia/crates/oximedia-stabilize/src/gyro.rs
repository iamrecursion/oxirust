//! Gyroscope-based stabilization.
//!
//! This module provides types and functions for processing raw gyroscope data
//! into smoothed orientation streams that can drive video stabilization.

#![allow(dead_code)]

/// A single gyroscope measurement.
#[derive(Debug, Clone, PartialEq)]
pub struct GyroSample {
    /// Timestamp of this measurement in microseconds.
    pub timestamp_us: u64,
    /// Roll angular velocity in degrees per second.
    pub roll: f64,
    /// Pitch angular velocity in degrees per second.
    pub pitch: f64,
    /// Yaw angular velocity in degrees per second.
    pub yaw: f64,
}

impl GyroSample {
    /// Creates a new `GyroSample`.
    #[must_use]
    pub const fn new(timestamp_us: u64, roll: f64, pitch: f64, yaw: f64) -> Self {
        Self {
            timestamp_us,
            roll,
            pitch,
            yaw,
        }
    }

    /// Returns the magnitude of the angular velocity vector (deg/s).
    #[must_use]
    pub fn magnitude(&self) -> f64 {
        (self.roll * self.roll + self.pitch * self.pitch + self.yaw * self.yaw).sqrt()
    }
}

/// A time-ordered stream of gyroscope samples.
#[derive(Debug, Clone)]
pub struct GyroStream {
    /// Samples ordered by `timestamp_us` ascending.
    pub samples: Vec<GyroSample>,
    /// The nominal sample rate in Hz.
    pub sample_rate_hz: f64,
}

impl GyroStream {
    /// Creates a new, empty `GyroStream` with the given sample rate.
    #[must_use]
    pub fn new(sample_rate_hz: f64) -> Self {
        Self {
            samples: Vec::new(),
            sample_rate_hz,
        }
    }

    /// Appends a sample to the stream.  The stream is kept sorted by timestamp.
    pub fn add_sample(&mut self, sample: GyroSample) {
        let pos = self
            .samples
            .partition_point(|s| s.timestamp_us <= sample.timestamp_us);
        self.samples.insert(pos, sample);
    }

    /// Linearly interpolates (or extrapolates at boundaries) a sample at the
    /// given timestamp.
    ///
    /// Returns `None` if the stream is empty.
    #[must_use]
    pub fn interpolate_at(&self, timestamp_us: u64) -> Option<GyroSample> {
        if self.samples.is_empty() {
            return None;
        }
        if self.samples.len() == 1 {
            let s = &self.samples[0];
            return Some(GyroSample::new(timestamp_us, s.roll, s.pitch, s.yaw));
        }

        // Find the bracket.
        let pos = self
            .samples
            .partition_point(|s| s.timestamp_us <= timestamp_us);

        if pos == 0 {
            // Before first sample – return first sample values.
            let s = &self.samples[0];
            return Some(GyroSample::new(timestamp_us, s.roll, s.pitch, s.yaw));
        }
        if pos >= self.samples.len() {
            // After last sample – return last sample values.
            let s = &self.samples[self.samples.len() - 1];
            return Some(GyroSample::new(timestamp_us, s.roll, s.pitch, s.yaw));
        }

        let a = &self.samples[pos - 1];
        let b = &self.samples[pos];
        let span = (b.timestamp_us - a.timestamp_us) as f64;
        let t = if span > 0.0 {
            (timestamp_us - a.timestamp_us) as f64 / span
        } else {
            0.0
        };

        Some(GyroSample::new(
            timestamp_us,
            a.roll + (b.roll - a.roll) * t,
            a.pitch + (b.pitch - a.pitch) * t,
            a.yaw + (b.yaw - a.yaw) * t,
        ))
    }

    /// Applies a box-filter smoothing over a window of `window_ms` milliseconds
    /// and returns a new `GyroStream`.
    #[must_use]
    pub fn smooth(self, window_ms: u64) -> Self {
        if self.samples.len() < 2 {
            return self;
        }
        let half_window_us = window_ms * 500; // half window in microseconds
        let mut smoothed = Vec::with_capacity(self.samples.len());

        for sample in &self.samples {
            let t = sample.timestamp_us;
            let lo = t.saturating_sub(half_window_us);
            let hi = t.saturating_add(half_window_us);

            let in_window: Vec<&GyroSample> = self
                .samples
                .iter()
                .filter(|s| s.timestamp_us >= lo && s.timestamp_us <= hi)
                .collect();

            let n = in_window.len() as f64;
            let roll = in_window.iter().map(|s| s.roll).sum::<f64>() / n;
            let pitch = in_window.iter().map(|s| s.pitch).sum::<f64>() / n;
            let yaw = in_window.iter().map(|s| s.yaw).sum::<f64>() / n;

            smoothed.push(GyroSample::new(t, roll, pitch, yaw));
        }

        Self {
            samples: smoothed,
            sample_rate_hz: self.sample_rate_hz,
        }
    }

    /// Numerically integrates the angular velocities to produce cumulative
    /// orientation angles (roll, pitch, yaw) in degrees.
    ///
    /// Returns a vector of `(timestamp_us, roll_deg, pitch_deg, yaw_deg)`.
    #[must_use]
    pub fn integrate_angles(&self) -> Vec<(u64, f64, f64, f64)> {
        if self.samples.is_empty() {
            return Vec::new();
        }

        let mut result = Vec::with_capacity(self.samples.len());
        let mut acc_roll = 0.0_f64;
        let mut acc_pitch = 0.0_f64;
        let mut acc_yaw = 0.0_f64;

        result.push((self.samples[0].timestamp_us, 0.0, 0.0, 0.0));

        for window in self.samples.windows(2) {
            let a = &window[0];
            let b = &window[1];
            let dt_s = (b.timestamp_us - a.timestamp_us) as f64 * 1e-6;
            // Trapezoidal integration.
            acc_roll += (a.roll + b.roll) * 0.5 * dt_s;
            acc_pitch += (a.pitch + b.pitch) * 0.5 * dt_s;
            acc_yaw += (a.yaw + b.yaw) * 0.5 * dt_s;
            result.push((b.timestamp_us, acc_roll, acc_pitch, acc_yaw));
        }

        result
    }
}

/// Gyroscope calibration parameters (bias and scale).
#[derive(Debug, Clone, PartialEq)]
pub struct GyroCalibration {
    /// Roll axis bias in deg/s.
    pub bias_roll: f64,
    /// Pitch axis bias in deg/s.
    pub bias_pitch: f64,
    /// Yaw axis bias in deg/s.
    pub bias_yaw: f64,
    /// Uniform scale factor applied to all axes.
    pub scale: f64,
}

impl Default for GyroCalibration {
    fn default() -> Self {
        Self {
            bias_roll: 0.0,
            bias_pitch: 0.0,
            bias_yaw: 0.0,
            scale: 1.0,
        }
    }
}

/// Estimates calibration parameters from a set of samples that were collected
/// while the sensor was stationary.
///
/// The biases are estimated as the mean of each axis. The scale is kept at 1.0
/// (external scale calibration requires a reference).
#[must_use]
pub fn calibrate_gyro(samples: &[GyroSample]) -> GyroCalibration {
    if samples.is_empty() {
        return GyroCalibration::default();
    }
    let n = samples.len() as f64;
    let bias_roll = samples.iter().map(|s| s.roll).sum::<f64>() / n;
    let bias_pitch = samples.iter().map(|s| s.pitch).sum::<f64>() / n;
    let bias_yaw = samples.iter().map(|s| s.yaw).sum::<f64>() / n;
    GyroCalibration {
        bias_roll,
        bias_pitch,
        bias_yaw,
        scale: 1.0,
    }
}

/// Applies calibration parameters to every sample in the stream in-place.
pub fn apply_calibration(stream: &mut GyroStream, cal: &GyroCalibration) {
    for sample in stream.samples.iter_mut() {
        sample.roll = (sample.roll - cal.bias_roll) * cal.scale;
        sample.pitch = (sample.pitch - cal.bias_pitch) * cal.scale;
        sample.yaw = (sample.yaw - cal.bias_yaw) * cal.scale;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample(ts: u64, r: f64, p: f64, y: f64) -> GyroSample {
        GyroSample::new(ts, r, p, y)
    }

    #[test]
    fn test_gyro_sample_magnitude() {
        let s = sample(0, 3.0, 4.0, 0.0);
        assert!((s.magnitude() - 5.0).abs() < 1e-10);
    }

    #[test]
    fn test_add_sample_keeps_sorted() {
        let mut stream = GyroStream::new(200.0);
        stream.add_sample(sample(3000, 1.0, 0.0, 0.0));
        stream.add_sample(sample(1000, 2.0, 0.0, 0.0));
        stream.add_sample(sample(2000, 3.0, 0.0, 0.0));
        assert_eq!(stream.samples[0].timestamp_us, 1000);
        assert_eq!(stream.samples[1].timestamp_us, 2000);
        assert_eq!(stream.samples[2].timestamp_us, 3000);
    }

    #[test]
    fn test_interpolate_at_exact_timestamp() {
        let mut stream = GyroStream::new(200.0);
        stream.add_sample(sample(0, 1.0, 2.0, 3.0));
        stream.add_sample(sample(1_000_000, 2.0, 4.0, 6.0));
        let s = stream.interpolate_at(0).expect("should succeed in test");
        assert!((s.roll - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_interpolate_at_midpoint() {
        let mut stream = GyroStream::new(200.0);
        stream.add_sample(sample(0, 0.0, 0.0, 0.0));
        stream.add_sample(sample(1_000_000, 2.0, 4.0, 6.0));
        let s = stream
            .interpolate_at(500_000)
            .expect("should succeed in test");
        assert!((s.roll - 1.0).abs() < 1e-10);
        assert!((s.pitch - 2.0).abs() < 1e-10);
    }

    #[test]
    fn test_interpolate_at_empty_returns_none() {
        let stream = GyroStream::new(200.0);
        assert!(stream.interpolate_at(0).is_none());
    }

    #[test]
    fn test_interpolate_before_first_sample() {
        let mut stream = GyroStream::new(200.0);
        stream.add_sample(sample(1000, 5.0, 0.0, 0.0));
        let s = stream.interpolate_at(0).expect("should succeed in test");
        assert!((s.roll - 5.0).abs() < 1e-10);
    }

    #[test]
    fn test_interpolate_after_last_sample() {
        let mut stream = GyroStream::new(200.0);
        stream.add_sample(sample(0, 5.0, 0.0, 0.0));
        let s = stream
            .interpolate_at(999_999)
            .expect("should succeed in test");
        assert!((s.roll - 5.0).abs() < 1e-10);
    }

    #[test]
    fn test_smooth_reduces_variance() {
        let mut stream = GyroStream::new(1000.0);
        // Alternating high/low values
        for i in 0..10u64 {
            let v = if i % 2 == 0 { 10.0 } else { -10.0 };
            stream.add_sample(sample(i * 1000, v, 0.0, 0.0));
        }
        let smoothed = stream.clone().smooth(5);
        // After smoothing the extremes should be reduced
        let max_raw = stream
            .samples
            .iter()
            .map(|s| s.roll.abs())
            .fold(0.0_f64, f64::max);
        let max_smoothed = smoothed
            .samples
            .iter()
            .map(|s| s.roll.abs())
            .fold(0.0_f64, f64::max);
        assert!(max_smoothed < max_raw);
    }

    #[test]
    fn test_integrate_angles_zero_velocity() {
        let mut stream = GyroStream::new(1000.0);
        stream.add_sample(sample(0, 0.0, 0.0, 0.0));
        stream.add_sample(sample(1_000_000, 0.0, 0.0, 0.0));
        let integrated = stream.integrate_angles();
        assert_eq!(integrated.len(), 2);
        let (_, r, p, y) = integrated[1];
        assert!(r.abs() < 1e-10);
        assert!(p.abs() < 1e-10);
        assert!(y.abs() < 1e-10);
    }

    #[test]
    fn test_calibrate_gyro_estimates_bias() {
        let samples = vec![
            sample(0, 0.5, 1.0, -0.5),
            sample(1000, 0.5, 1.0, -0.5),
            sample(2000, 0.5, 1.0, -0.5),
        ];
        let cal = calibrate_gyro(&samples);
        assert!((cal.bias_roll - 0.5).abs() < 1e-10);
        assert!((cal.bias_pitch - 1.0).abs() < 1e-10);
        assert!((cal.bias_yaw - (-0.5)).abs() < 1e-10);
    }

    #[test]
    fn test_apply_calibration_removes_bias() {
        let mut stream = GyroStream::new(1000.0);
        stream.add_sample(sample(0, 1.0, 2.0, 3.0));
        let cal = GyroCalibration {
            bias_roll: 1.0,
            bias_pitch: 2.0,
            bias_yaw: 3.0,
            scale: 1.0,
        };
        apply_calibration(&mut stream, &cal);
        let s = &stream.samples[0];
        assert!(s.roll.abs() < 1e-10);
        assert!(s.pitch.abs() < 1e-10);
        assert!(s.yaw.abs() < 1e-10);
    }

    #[test]
    fn test_calibrate_empty_samples() {
        let cal = calibrate_gyro(&[]);
        assert!((cal.bias_roll).abs() < 1e-10);
        assert!((cal.scale - 1.0).abs() < 1e-10);
    }
}
