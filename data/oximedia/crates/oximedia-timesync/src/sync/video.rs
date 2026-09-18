//! Frame-accurate video synchronization.

use super::genlock::{GenlockFrameRate, GenlockGenerator};
use std::time::{Duration, Instant};

/// Video sync state.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VideoSyncState {
    /// Not synchronized
    Unlocked,
    /// Synchronizing
    Locking,
    /// Locked to reference
    Locked,
}

/// Video synchronizer.
pub struct VideoSync {
    /// Genlock generator
    genlock: GenlockGenerator,
    /// Sync state
    state: VideoSyncState,
    /// Target frame time
    target_frame_time: Option<Instant>,
    /// Tolerance for frame time matching (nanoseconds)
    tolerance_ns: i64,
}

impl VideoSync {
    /// Create a new video synchronizer.
    #[must_use]
    pub fn new(frame_rate: GenlockFrameRate) -> Self {
        Self {
            genlock: GenlockGenerator::new(frame_rate),
            state: VideoSyncState::Unlocked,
            target_frame_time: None,
            tolerance_ns: 100_000, // 100 microseconds
        }
    }

    /// Synchronize to a target time.
    pub fn sync_to(&mut self, target_time: Instant) {
        self.target_frame_time = Some(target_time);
        self.state = VideoSyncState::Locking;
    }

    /// Get next frame time with synchronization.
    pub fn next_frame(&mut self) -> Instant {
        let frame_time = self.genlock.next_frame_time();

        // Check synchronization
        if let Some(target) = self.target_frame_time {
            let now = Instant::now();
            let error = if target > now {
                target.duration_since(now).as_nanos() as i64
            } else {
                -(now.duration_since(target).as_nanos() as i64)
            };

            if error.abs() < self.tolerance_ns {
                // Within tolerance - locked
                self.state = VideoSyncState::Locked;
            } else {
                // Out of tolerance - adjust phase
                let adjustment = error / 10; // Gradual adjustment
                self.genlock.adjust_phase(adjustment);
                self.state = VideoSyncState::Locking;
            }
        }

        frame_time
    }

    /// Get sync state.
    #[must_use]
    pub fn state(&self) -> VideoSyncState {
        self.state
    }

    /// Check if locked.
    #[must_use]
    pub fn is_locked(&self) -> bool {
        self.state == VideoSyncState::Locked
    }

    /// Reset synchronization.
    pub fn reset(&mut self) {
        self.genlock.reset();
        self.state = VideoSyncState::Unlocked;
        self.target_frame_time = None;
    }

    /// Get current frame number.
    #[must_use]
    pub fn frame_number(&self) -> u64 {
        self.genlock.frame_number()
    }
}

/// Frame-accurate video player sync.
pub struct FrameAccurateSync {
    /// Video synchronizer
    video_sync: VideoSync,
    /// Expected frame interval
    #[allow(dead_code)]
    frame_interval: Duration,
}

impl FrameAccurateSync {
    /// Create a new frame-accurate sync.
    #[must_use]
    pub fn new(frame_rate: GenlockFrameRate) -> Self {
        Self {
            video_sync: VideoSync::new(frame_rate),
            frame_interval: frame_rate.frame_duration(),
        }
    }

    /// Wait until next frame should be displayed.
    #[cfg(not(target_arch = "wasm32"))]
    pub async fn wait_for_next_frame(&mut self) {
        let next_time = self.video_sync.next_frame();
        let now = Instant::now();

        if next_time > now {
            tokio::time::sleep_until(tokio::time::Instant::from_std(next_time)).await;
        }
    }

    /// Get current frame number.
    #[must_use]
    pub fn frame_number(&self) -> u64 {
        self.video_sync.frame_number()
    }

    /// Sync to external reference.
    pub fn sync_to(&mut self, target_time: Instant) {
        self.video_sync.sync_to(target_time);
    }

    /// Check if synchronized.
    #[must_use]
    pub fn is_synced(&self) -> bool {
        self.video_sync.is_locked()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_video_sync_creation() {
        let sync = VideoSync::new(GenlockFrameRate::FPS_25);
        assert_eq!(sync.state(), VideoSyncState::Unlocked);
        assert!(!sync.is_locked());
    }

    #[test]
    fn test_video_sync_next_frame() {
        let mut sync = VideoSync::new(GenlockFrameRate::FPS_25);

        let t1 = sync.next_frame();
        let t2 = sync.next_frame();

        let diff = t2.duration_since(t1);
        assert!((diff.as_millis() as i64 - 40).abs() < 2);
    }

    #[test]
    fn test_frame_accurate_sync() {
        let sync = FrameAccurateSync::new(GenlockFrameRate::FPS_25);
        assert_eq!(sync.frame_number(), 0);
    }
}
