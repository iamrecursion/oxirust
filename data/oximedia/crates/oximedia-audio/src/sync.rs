//! Audio synchronisation and alignment utilities.
//!
//! Provides tools to track clock drift between audio and video streams,
//! compute corrected timestamps, and map between sample indices and
//! wall-clock time.

/// A single synchronisation anchor: a known relationship between a
/// wall-clock timestamp and a sample index.
#[derive(Debug, Clone, Copy)]
pub struct AudioSyncPoint {
    /// Wall-clock timestamp in microseconds.
    pub timestamp_us: u64,
    /// Corresponding sample index (0-based).
    pub sample_index: u64,
}

/// Tracks clock drift by accumulating [`AudioSyncPoint`]s and computing
/// a linear drift estimate.
pub struct AudioSyncTracker {
    /// Collected synchronisation anchors.
    pub sync_points: Vec<AudioSyncPoint>,
    /// Audio sample rate in Hz.
    pub sample_rate: u32,
    /// Manually applied drift correction in microseconds.
    pub drift_us: i64,
}

impl AudioSyncTracker {
    /// Create a new tracker for the given sample rate.
    pub fn new(sample_rate: u32) -> Self {
        Self {
            sync_points: Vec::new(),
            sample_rate,
            drift_us: 0,
        }
    }

    /// Add a synchronisation point.
    pub fn add_sync_point(&mut self, sp: AudioSyncPoint) {
        self.sync_points.push(sp);
    }

    /// Estimate the current drift in microseconds by comparing the
    /// expected timestamp (derived from sample index) with the measured
    /// wall-clock timestamp across all sync points.
    ///
    /// Returns the mean error (measured minus expected) in microseconds.
    /// Returns `0` when fewer than two sync points are available.
    pub fn estimate_drift_us(&self) -> i64 {
        if self.sync_points.len() < 2 {
            return 0;
        }
        let first = &self.sync_points[0];
        let mut total_error: i64 = 0;
        let count = (self.sync_points.len() - 1) as i64;

        for sp in self.sync_points.iter().skip(1) {
            let elapsed_samples = sp.sample_index.saturating_sub(first.sample_index);
            let expected_us = first.timestamp_us + samples_to_us(elapsed_samples, self.sample_rate);
            let measured_us = sp.timestamp_us;
            let error = measured_us as i64 - expected_us as i64;
            total_error += error;
        }
        total_error / count
    }

    /// Compute the corrected wall-clock timestamp for a given sample index,
    /// applying the measured drift and any manual `drift_us` correction.
    pub fn corrected_timestamp(&self, sample_index: u64) -> u64 {
        let base_us = if let Some(first) = self.sync_points.first() {
            let elapsed = sample_index.saturating_sub(first.sample_index);
            first.timestamp_us + samples_to_us(elapsed, self.sample_rate)
        } else {
            samples_to_us(sample_index, self.sample_rate)
        };
        let drift = self.estimate_drift_us() + self.drift_us;
        if drift >= 0 {
            base_us.saturating_sub(drift as u64)
        } else {
            base_us.saturating_add((-drift) as u64)
        }
    }

    /// Return the sample index that corresponds to the given wall-clock
    /// timestamp, using the nearest sync point as an anchor.
    pub fn sample_for_time(&self, timestamp_us: u64) -> u64 {
        if let Some(first) = self.sync_points.first() {
            let elapsed_us = timestamp_us.saturating_sub(first.timestamp_us);
            first.sample_index + us_to_samples(elapsed_us, self.sample_rate)
        } else {
            us_to_samples(timestamp_us, self.sample_rate)
        }
    }
}

// --- free functions ----------------------------------------------------------

/// Compute the sample offset in stream B that corresponds to frame `a_frame`
/// in stream A, given their respective sample rates.
pub fn compute_sample_offset(a_rate: u32, b_rate: u32, a_frame: u64) -> u64 {
    if a_rate == 0 {
        return 0;
    }
    // a_frame / a_rate * b_rate  (done in u128 to avoid overflow)
    let result = a_frame as u128 * b_rate as u128 / a_rate as u128;
    result as u64
}

/// Return the audio/video synchronisation error in milliseconds.
///
/// A positive value means audio is ahead of video; negative means audio is late.
pub fn audio_video_sync_error_ms(audio_pts_us: u64, video_pts_us: u64) -> f64 {
    let diff = audio_pts_us as i64 - video_pts_us as i64;
    diff as f64 / 1000.0
}

// --- helpers -----------------------------------------------------------------

/// Convert a sample count to microseconds at the given sample rate.
fn samples_to_us(samples: u64, sample_rate: u32) -> u64 {
    if sample_rate == 0 {
        return 0;
    }
    samples * 1_000_000 / sample_rate as u64
}

/// Convert microseconds to a sample count at the given sample rate.
fn us_to_samples(us: u64, sample_rate: u32) -> u64 {
    us * sample_rate as u64 / 1_000_000
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_samples_to_us_48k() {
        // 48 000 samples at 48 kHz = 1 second = 1 000 000 µs
        assert_eq!(samples_to_us(48_000, 48_000), 1_000_000);
    }

    #[test]
    fn test_us_to_samples_48k() {
        assert_eq!(us_to_samples(1_000_000, 48_000), 48_000);
    }

    #[test]
    fn test_tracker_new() {
        let t = AudioSyncTracker::new(44_100);
        assert_eq!(t.sample_rate, 44_100);
        assert!(t.sync_points.is_empty());
    }

    #[test]
    fn test_estimate_drift_no_points() {
        let t = AudioSyncTracker::new(48_000);
        assert_eq!(t.estimate_drift_us(), 0);
    }

    #[test]
    fn test_estimate_drift_one_point() {
        let mut t = AudioSyncTracker::new(48_000);
        t.add_sync_point(AudioSyncPoint {
            timestamp_us: 0,
            sample_index: 0,
        });
        assert_eq!(t.estimate_drift_us(), 0);
    }

    #[test]
    fn test_estimate_drift_perfect_sync() {
        let mut t = AudioSyncTracker::new(48_000);
        t.add_sync_point(AudioSyncPoint {
            timestamp_us: 0,
            sample_index: 0,
        });
        // Exactly 1 s later
        t.add_sync_point(AudioSyncPoint {
            timestamp_us: 1_000_000,
            sample_index: 48_000,
        });
        assert_eq!(t.estimate_drift_us(), 0);
    }

    #[test]
    fn test_estimate_drift_with_drift() {
        let mut t = AudioSyncTracker::new(48_000);
        t.add_sync_point(AudioSyncPoint {
            timestamp_us: 0,
            sample_index: 0,
        });
        // Clock is 500 µs fast
        t.add_sync_point(AudioSyncPoint {
            timestamp_us: 1_000_500,
            sample_index: 48_000,
        });
        assert_eq!(t.estimate_drift_us(), 500);
    }

    #[test]
    fn test_corrected_timestamp_no_sync() {
        let t = AudioSyncTracker::new(48_000);
        // 48 000 samples => 1 000 000 µs, no drift
        assert_eq!(t.corrected_timestamp(48_000), 1_000_000);
    }

    #[test]
    fn test_sample_for_time_no_sync() {
        let t = AudioSyncTracker::new(48_000);
        assert_eq!(t.sample_for_time(1_000_000), 48_000);
    }

    #[test]
    fn test_sample_for_time_with_anchor() {
        let mut t = AudioSyncTracker::new(48_000);
        t.add_sync_point(AudioSyncPoint {
            timestamp_us: 1_000_000,
            sample_index: 48_000,
        });
        // 2 seconds from stream start = 2 s from epoch => sample 96 000
        assert_eq!(t.sample_for_time(2_000_000), 96_000);
    }

    #[test]
    fn test_compute_sample_offset_same_rate() {
        assert_eq!(compute_sample_offset(48_000, 48_000, 1000), 1000);
    }

    #[test]
    fn test_compute_sample_offset_rate_conversion() {
        // Frame 44100 at 44.1 kHz => 1 s => 48 000 samples at 48 kHz
        assert_eq!(compute_sample_offset(44_100, 48_000, 44_100), 48_000);
    }

    #[test]
    fn test_audio_video_sync_error_ms_ahead() {
        // Audio 10 ms ahead of video
        let err = audio_video_sync_error_ms(1_010_000, 1_000_000);
        assert!((err - 10.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_audio_video_sync_error_ms_late() {
        let err = audio_video_sync_error_ms(990_000, 1_000_000);
        assert!((err + 10.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_audio_video_sync_error_ms_zero() {
        let err = audio_video_sync_error_ms(1_000_000, 1_000_000);
        assert_eq!(err, 0.0);
    }
}
