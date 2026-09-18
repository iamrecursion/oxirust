//! Frame timing and synchronization
//!
//! Provides accurate frame timing for smooth playback and recording.

use super::SyncTimestamp;
use std::time::{Duration, Instant};

/// Frame timer
pub struct FrameTimer {
    frame_rate: f64,
    start_time: Instant,
    frame_count: u64,
}

impl FrameTimer {
    /// Create new frame timer
    #[must_use]
    pub fn new(frame_rate: f64) -> Self {
        Self {
            frame_rate,
            start_time: Instant::now(),
            frame_count: 0,
        }
    }

    /// Wait for next frame
    pub fn wait_for_next_frame(&mut self) -> SyncTimestamp {
        let frame_duration = Duration::from_secs_f64(1.0 / self.frame_rate);
        let target_time = self.start_time + frame_duration * self.frame_count as u32;

        let now = Instant::now();
        if now < target_time {
            std::thread::sleep(target_time - now);
        }

        let timestamp = SyncTimestamp::new(
            Instant::now().duration_since(self.start_time).as_nanos() as u64,
            self.frame_count,
        );

        self.frame_count += 1;
        timestamp
    }

    /// Reset timer
    pub fn reset(&mut self) {
        self.start_time = Instant::now();
        self.frame_count = 0;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_frame_timer() {
        let timer = FrameTimer::new(60.0);
        assert_eq!(timer.frame_count, 0);
    }
}
