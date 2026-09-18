//! Sample-accurate audio synchronization.

use std::time::{Duration, Instant};

/// Audio sync state.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AudioSyncState {
    /// Not synchronized
    Unlocked,
    /// Synchronizing
    Locking,
    /// Locked to reference
    Locked,
}

/// Audio synchronizer.
pub struct AudioSync {
    /// Sample rate (Hz)
    sample_rate: u32,
    /// Sync state
    state: AudioSyncState,
    /// Start time
    start_time: Instant,
    /// Samples elapsed
    samples_elapsed: u64,
    /// Sample rate adjustment (ppm - parts per million)
    rate_adjust_ppm: f64,
}

impl AudioSync {
    /// Create a new audio synchronizer.
    #[must_use]
    pub fn new(sample_rate: u32) -> Self {
        Self {
            sample_rate,
            state: AudioSyncState::Unlocked,
            start_time: Instant::now(),
            samples_elapsed: 0,
            rate_adjust_ppm: 0.0,
        }
    }

    /// Get the time for a given sample number.
    #[must_use]
    pub fn sample_to_time(&self, sample: u64) -> Instant {
        // Base duration
        let base_duration = Duration::from_secs_f64(sample as f64 / f64::from(self.sample_rate));

        // Apply rate adjustment
        let adjusted_duration = if self.rate_adjust_ppm == 0.0 {
            base_duration
        } else {
            let adjustment = 1.0 + (self.rate_adjust_ppm / 1_000_000.0);
            Duration::from_secs_f64(base_duration.as_secs_f64() / adjustment)
        };

        self.start_time + adjusted_duration
    }

    /// Get the sample number for a given time.
    #[must_use]
    pub fn time_to_sample(&self, time: Instant) -> u64 {
        let elapsed = if time > self.start_time {
            time.duration_since(self.start_time)
        } else {
            Duration::ZERO
        };

        // Apply rate adjustment
        let adjusted_elapsed = if self.rate_adjust_ppm == 0.0 {
            elapsed
        } else {
            let adjustment = 1.0 + (self.rate_adjust_ppm / 1_000_000.0);
            Duration::from_secs_f64(elapsed.as_secs_f64() * adjustment)
        };

        (adjusted_elapsed.as_secs_f64() * f64::from(self.sample_rate)) as u64
    }

    /// Advance by a number of samples.
    pub fn advance_samples(&mut self, samples: u64) {
        self.samples_elapsed += samples;
    }

    /// Synchronize to a target time.
    pub fn sync_to(&mut self, target_sample: u64, target_time: Instant) {
        let current_time = self.sample_to_time(target_sample);
        let now = Instant::now();

        // Calculate error
        let error_ns = if target_time > current_time {
            target_time.duration_since(current_time).as_nanos() as i64
        } else {
            -(current_time.duration_since(target_time).as_nanos() as i64)
        };

        // Convert to ppm adjustment
        // ppm = (error / current_time) * 1_000_000
        let elapsed = now.duration_since(self.start_time).as_secs_f64();
        if elapsed > 0.0 {
            let error_frac = error_ns as f64 / (elapsed * 1e9);
            self.rate_adjust_ppm = error_frac * 1_000_000.0;

            // Limit adjustment to ±1000 ppm (0.1%)
            if self.rate_adjust_ppm > 1000.0 {
                self.rate_adjust_ppm = 1000.0;
            } else if self.rate_adjust_ppm < -1000.0 {
                self.rate_adjust_ppm = -1000.0;
            }
        }

        self.state = AudioSyncState::Locking;

        // Check if within tolerance
        if error_ns.abs() < 1_000_000 {
            // Within 1ms
            self.state = AudioSyncState::Locked;
        }
    }

    /// Get sync state.
    #[must_use]
    pub fn state(&self) -> AudioSyncState {
        self.state
    }

    /// Check if locked.
    #[must_use]
    pub fn is_locked(&self) -> bool {
        self.state == AudioSyncState::Locked
    }

    /// Reset synchronization.
    pub fn reset(&mut self) {
        self.start_time = Instant::now();
        self.samples_elapsed = 0;
        self.rate_adjust_ppm = 0.0;
        self.state = AudioSyncState::Unlocked;
    }

    /// Get current sample rate adjustment (ppm).
    #[must_use]
    pub fn rate_adjustment(&self) -> f64 {
        self.rate_adjust_ppm
    }

    /// Get samples elapsed.
    #[must_use]
    pub fn samples_elapsed(&self) -> u64 {
        self.samples_elapsed
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_audio_sync_creation() {
        let sync = AudioSync::new(48000);
        assert_eq!(sync.state(), AudioSyncState::Unlocked);
        assert!(!sync.is_locked());
    }

    #[test]
    fn test_sample_time_conversion() {
        let sync = AudioSync::new(48000);

        // 48000 samples at 48kHz = 1 second
        let time = sync.sample_to_time(48000);
        let elapsed = time.duration_since(sync.start_time);
        assert_eq!(elapsed.as_secs(), 1);

        // Reverse conversion
        let sample = sync.time_to_sample(time);
        assert_eq!(sample, 48000);
    }

    #[test]
    fn test_advance_samples() {
        let mut sync = AudioSync::new(48000);
        sync.advance_samples(1000);
        assert_eq!(sync.samples_elapsed(), 1000);

        sync.advance_samples(500);
        assert_eq!(sync.samples_elapsed(), 1500);
    }
}
