//! Jam sync for timecode synchronization.

use crate::error::TimeSyncResult;
use oximedia_timecode::{FrameRate, Timecode};
use std::time::{Duration, Instant};

/// Jam sync state.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum JamSyncState {
    /// Unlocked - no external timecode
    Unlocked,
    /// Syncing - receiving timecode but not locked
    Syncing,
    /// Locked - synchronized to external timecode
    Locked,
    /// Freewheel - was locked but lost external timecode
    Freewheel,
}

/// Jam sync configuration.
#[derive(Debug, Clone)]
pub struct JamSyncConfig {
    /// Minimum consecutive matches to lock
    pub lock_threshold: u32,
    /// Maximum time without updates before freewheeling
    pub freewheel_timeout: Duration,
    /// Maximum time in freewheel before unlocking
    pub unlock_timeout: Duration,
}

impl Default for JamSyncConfig {
    fn default() -> Self {
        Self {
            lock_threshold: 5,
            freewheel_timeout: Duration::from_millis(100),
            unlock_timeout: Duration::from_secs(5),
        }
    }
}

/// Jam sync controller.
pub struct JamSync {
    /// Configuration
    config: JamSyncConfig,
    /// Current state
    state: JamSyncState,
    /// Current timecode
    current_timecode: Option<Timecode>,
    /// Frame rate
    #[allow(dead_code)]
    frame_rate: FrameRate,
    /// Consecutive matches
    consecutive_matches: u32,
    /// Last update time
    last_update: Option<Instant>,
    /// Freewheel start time
    freewheel_start: Option<Instant>,
}

impl JamSync {
    /// Create a new jam sync controller.
    #[must_use]
    pub fn new(frame_rate: FrameRate, config: JamSyncConfig) -> Self {
        Self {
            config,
            state: JamSyncState::Unlocked,
            current_timecode: None,
            frame_rate,
            consecutive_matches: 0,
            last_update: None,
            freewheel_start: None,
        }
    }

    /// Update with external timecode.
    pub fn update(&mut self, timecode: Timecode) -> TimeSyncResult<()> {
        let now = Instant::now();

        match self.current_timecode {
            None => {
                // First timecode received
                self.current_timecode = Some(timecode);
                self.state = JamSyncState::Syncing;
                self.consecutive_matches = 1;
            }
            Some(mut current) => {
                // Check if timecode is consecutive
                current
                    .increment()
                    .map_err(|e| crate::error::TimeSyncError::Timecode(e.to_string()))?;
                if current == timecode {
                    self.consecutive_matches += 1;

                    // Check if we should lock
                    if self.consecutive_matches >= self.config.lock_threshold
                        && self.state != JamSyncState::Locked
                    {
                        self.state = JamSyncState::Locked;
                        self.freewheel_start = None;
                    }
                } else {
                    // Non-consecutive, reset
                    self.consecutive_matches = 1;
                    if self.state == JamSyncState::Locked {
                        self.state = JamSyncState::Syncing;
                    }
                }

                self.current_timecode = Some(timecode);
            }
        }

        self.last_update = Some(now);
        Ok(())
    }

    /// Tick the jam sync (call regularly to maintain freewheel).
    pub fn tick(&mut self) -> TimeSyncResult<()> {
        let now = Instant::now();

        if let Some(last_update) = self.last_update {
            let elapsed = now.duration_since(last_update);

            match self.state {
                JamSyncState::Locked => {
                    if elapsed > self.config.freewheel_timeout {
                        // Enter freewheel mode
                        self.state = JamSyncState::Freewheel;
                        self.freewheel_start = Some(now);
                    }
                }
                JamSyncState::Freewheel => {
                    if let Some(freewheel_start) = self.freewheel_start {
                        let freewheel_time = now.duration_since(freewheel_start);
                        if freewheel_time > self.config.unlock_timeout {
                            // Lost lock
                            self.state = JamSyncState::Unlocked;
                            self.current_timecode = None;
                            self.consecutive_matches = 0;
                            self.freewheel_start = None;
                        }
                    }
                }
                _ => {}
            }
        }

        Ok(())
    }

    /// Get current state.
    #[must_use]
    pub fn state(&self) -> JamSyncState {
        self.state
    }

    /// Get current timecode.
    #[must_use]
    pub fn current_timecode(&self) -> Option<&Timecode> {
        self.current_timecode.as_ref()
    }

    /// Check if locked.
    #[must_use]
    pub fn is_locked(&self) -> bool {
        self.state == JamSyncState::Locked
    }

    /// Check if in freewheel.
    #[must_use]
    pub fn is_freewheel(&self) -> bool {
        self.state == JamSyncState::Freewheel
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_jam_sync_creation() {
        let config = JamSyncConfig::default();
        let js = JamSync::new(FrameRate::Fps25, config);
        assert_eq!(js.state(), JamSyncState::Unlocked);
        assert!(!js.is_locked());
    }

    #[test]
    fn test_jam_sync_lock() {
        let config = JamSyncConfig {
            lock_threshold: 3,
            ..Default::default()
        };
        let mut js = JamSync::new(FrameRate::Fps25, config);

        // Send consecutive timecodes
        let mut tc = Timecode::new(0, 0, 0, 0, FrameRate::Fps25).expect("should succeed in test");
        for _ in 0..3 {
            tc.increment().expect("should succeed in test");
            js.update(tc).expect("should succeed in test");
        }

        assert_eq!(js.state(), JamSyncState::Locked);
        assert!(js.is_locked());
    }

    #[test]
    fn test_jam_sync_freewheel() {
        let config = JamSyncConfig {
            lock_threshold: 2,
            freewheel_timeout: Duration::from_millis(50),
            ..Default::default()
        };
        let mut js = JamSync::new(FrameRate::Fps25, config);

        // Lock
        let mut tc = Timecode::new(0, 0, 0, 0, FrameRate::Fps25).expect("should succeed in test");
        for _ in 0..2 {
            tc.increment().expect("should succeed in test");
            js.update(tc).expect("should succeed in test");
        }
        assert!(js.is_locked());

        // Wait for freewheel
        std::thread::sleep(Duration::from_millis(60));
        js.tick().expect("should succeed in test");
        assert!(js.is_freewheel());
    }
}
