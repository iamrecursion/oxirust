//! House sync / genlock timecode distribution for broadcast automation.
//!
//! This module manages a facility-wide timecode master that distributes
//! reference timecode to all automation subsystems.  It supports:
//!
//! - SMPTE LTC (Linear Timecode) — 30 fps, 29.97 df, 25 fps, 24 fps.
//! - SMPTE VITC (Vertical Interval Timecode) — same frame rates.
//! - ATC (Ancillary Timecode) embedded in SDI.
//!
//! The [`TimecodeDistributor`] maintains a table of registered subscribers
//! and emits timecode updates whenever the master advances a frame.  In a
//! real facility this would be driven by a genlock reference signal; here the
//! master is advanced manually via `advance_frame` for testability.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use tracing::{debug, info, warn};

// ─────────────────────────────────────────────────────────────────────────────
// Timecode types
// ─────────────────────────────────────────────────────────────────────────────

/// Supported timecode standards.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum TimecodeStandard {
    /// SMPTE LTC 30 fps (integer).
    Ltc30,
    /// SMPTE LTC 29.97 fps drop-frame.
    Ltc2997Df,
    /// SMPTE LTC 25 fps (EBU PAL).
    Ltc25,
    /// SMPTE LTC 24 fps (cinema).
    Ltc24,
    /// SMPTE VITC 25 fps embedded in vertical interval.
    Vitc25,
    /// SMPTE VITC 30 fps embedded in vertical interval.
    Vitc30,
    /// Ancillary timecode in SDI HANC space.
    Atc,
}

impl TimecodeStandard {
    /// Nominal frames-per-second for this standard.
    pub fn fps(self) -> u32 {
        match self {
            Self::Ltc30 | Self::Vitc30 | Self::Atc => 30,
            Self::Ltc2997Df => 30, // nominal; actual is 29.97
            Self::Ltc25 | Self::Vitc25 => 25,
            Self::Ltc24 => 24,
        }
    }

    /// Returns `true` for drop-frame standards.
    pub fn is_drop_frame(self) -> bool {
        matches!(self, Self::Ltc2997Df)
    }

    /// Human-readable label.
    pub fn label(self) -> &'static str {
        match self {
            Self::Ltc30 => "LTC 30fps",
            Self::Ltc2997Df => "LTC 29.97df",
            Self::Ltc25 => "LTC 25fps",
            Self::Ltc24 => "LTC 24fps",
            Self::Vitc25 => "VITC 25fps",
            Self::Vitc30 => "VITC 30fps",
            Self::Atc => "ATC",
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Timecode value
// ─────────────────────────────────────────────────────────────────────────────

/// A SMPTE timecode value (HH:MM:SS:FF).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct SmpteTimecode {
    /// Hours (0–23).
    pub hours: u8,
    /// Minutes (0–59).
    pub minutes: u8,
    /// Seconds (0–59).
    pub seconds: u8,
    /// Frames (0 .. fps-1).
    pub frames: u8,
    /// Whether this timecode uses drop-frame counting.
    pub drop_frame: bool,
}

impl SmpteTimecode {
    /// Create a timecode at midnight / start-of-day.
    pub fn zero(drop_frame: bool) -> Self {
        Self {
            hours: 0,
            minutes: 0,
            seconds: 0,
            frames: 0,
            drop_frame,
        }
    }

    /// Advance by one frame at the given `fps`, wrapping at 24-hour boundary.
    pub fn advance(&mut self, fps: u32) {
        let fps = fps.max(1) as u8;
        self.frames += 1;
        if self.frames >= fps {
            self.frames = 0;
            self.seconds += 1;
            if self.seconds >= 60 {
                self.seconds = 0;
                self.minutes += 1;
                if self.minutes >= 60 {
                    self.minutes = 0;
                    self.hours += 1;
                    if self.hours >= 24 {
                        self.hours = 0;
                    }
                }
            }
        }
    }

    /// Convert to total frame count from 00:00:00:00 (no drop-frame
    /// compensation applied).
    pub fn to_frame_count(&self, fps: u32) -> u64 {
        let fps = fps as u64;
        let h = self.hours as u64;
        let m = self.minutes as u64;
        let s = self.seconds as u64;
        let f = self.frames as u64;
        ((h * 3600 + m * 60 + s) * fps) + f
    }
}

impl std::fmt::Display for SmpteTimecode {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let sep = if self.drop_frame { ';' } else { ':' };
        write!(
            f,
            "{:02}:{:02}:{:02}{}{:02}",
            self.hours, self.minutes, self.seconds, sep, self.frames
        )
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Source lock state
// ─────────────────────────────────────────────────────────────────────────────

/// Lock state of the timecode source.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum LockState {
    /// Locked to the reference signal.
    Locked,
    /// Searching for / attempting to lock to the reference.
    Searching,
    /// Lock has been lost; coasting on internal oscillator.
    Lost,
}

// ─────────────────────────────────────────────────────────────────────────────
// Subscriber
// ─────────────────────────────────────────────────────────────────────────────

/// A subscriber to the timecode distribution.
#[derive(Debug, Clone)]
pub struct TimecodeSubscriber {
    /// Unique subscriber ID.
    pub id: String,
    /// Human-readable name (e.g. `"Channel 1 Playout"`, `"Monitor Wall"`).
    pub name: String,
    /// Last timecode delivered to this subscriber.
    pub last_tc: Option<SmpteTimecode>,
    /// How many frames this subscriber has received.
    pub frames_received: u64,
}

impl TimecodeSubscriber {
    /// Create a new subscriber.
    pub fn new(id: impl Into<String>, name: impl Into<String>) -> Self {
        Self {
            id: id.into(),
            name: name.into(),
            last_tc: None,
            frames_received: 0,
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Distributor
// ─────────────────────────────────────────────────────────────────────────────

/// House timecode distributor.
///
/// Maintains a master timecode value and distributes it to all registered
/// subscribers on every call to [`Self::advance_frame`].
pub struct TimecodeDistributor {
    standard: TimecodeStandard,
    master: SmpteTimecode,
    lock_state: LockState,
    subscribers: HashMap<String, TimecodeSubscriber>,
    frame_count: u64,
}

impl TimecodeDistributor {
    /// Create a new distributor locked to the given timecode standard.
    pub fn new(standard: TimecodeStandard) -> Self {
        info!("Creating timecode distributor ({})", standard.label());
        Self {
            standard,
            master: SmpteTimecode::zero(standard.is_drop_frame()),
            lock_state: LockState::Searching,
            subscribers: HashMap::new(),
            frame_count: 0,
        }
    }

    // ── Lock state ────────────────────────────────────────────────────────────

    /// Signal that the distributor has locked to the reference signal.
    pub fn set_locked(&mut self) {
        info!("Timecode distributor locked ({})", self.standard.label());
        self.lock_state = LockState::Locked;
    }

    /// Signal that lock has been lost.
    pub fn set_lost(&mut self) {
        warn!("Timecode distributor lost lock");
        self.lock_state = LockState::Lost;
    }

    /// Return the current lock state.
    pub fn lock_state(&self) -> LockState {
        self.lock_state
    }

    // ── Master timecode ───────────────────────────────────────────────────────

    /// Return the current master timecode.
    pub fn master(&self) -> SmpteTimecode {
        self.master
    }

    /// Set the master timecode directly (e.g. for jam sync).
    pub fn jam_sync(&mut self, tc: SmpteTimecode) {
        info!("Jam sync: master timecode set to {}", tc);
        self.master = tc;
    }

    /// Advance the master by one frame and distribute to all subscribers.
    ///
    /// Returns the new timecode after the advance.
    pub fn advance_frame(&mut self) -> SmpteTimecode {
        self.master.advance(self.standard.fps());
        self.frame_count += 1;
        debug!("Master TC: {} (frame {})", self.master, self.frame_count);

        // Distribute to all subscribers.
        for sub in self.subscribers.values_mut() {
            sub.last_tc = Some(self.master);
            sub.frames_received += 1;
        }

        self.master
    }

    /// Total number of frames advanced since creation.
    pub fn frame_count(&self) -> u64 {
        self.frame_count
    }

    /// The timecode standard in use.
    pub fn standard(&self) -> TimecodeStandard {
        self.standard
    }

    // ── Subscribers ───────────────────────────────────────────────────────────

    /// Register a subscriber.  If a subscriber with the same ID already exists
    /// it is replaced.
    pub fn subscribe(&mut self, sub: TimecodeSubscriber) {
        info!("Timecode subscriber added: '{}' ({})", sub.id, sub.name);
        self.subscribers.insert(sub.id.clone(), sub);
    }

    /// Unsubscribe by ID.  Returns the removed subscriber, if any.
    pub fn unsubscribe(&mut self, id: &str) -> Option<TimecodeSubscriber> {
        self.subscribers.remove(id)
    }

    /// Return the last timecode delivered to a subscriber, if any.
    pub fn last_tc_for(&self, id: &str) -> Option<SmpteTimecode> {
        self.subscribers.get(id)?.last_tc
    }

    /// Number of registered subscribers.
    pub fn subscriber_count(&self) -> usize {
        self.subscribers.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_timecode_advance_basic() {
        let mut tc = SmpteTimecode::zero(false);
        tc.advance(25);
        assert_eq!(tc.frames, 1);
    }

    #[test]
    fn test_timecode_advance_wraps_seconds() {
        let mut tc = SmpteTimecode {
            hours: 0,
            minutes: 0,
            seconds: 0,
            frames: 24,
            drop_frame: false,
        };
        tc.advance(25);
        assert_eq!(tc.frames, 0);
        assert_eq!(tc.seconds, 1);
    }

    #[test]
    fn test_timecode_display_non_dropframe() {
        let tc = SmpteTimecode {
            hours: 1,
            minutes: 2,
            seconds: 3,
            frames: 4,
            drop_frame: false,
        };
        assert_eq!(tc.to_string(), "01:02:03:04");
    }

    #[test]
    fn test_timecode_display_dropframe() {
        let tc = SmpteTimecode {
            hours: 0,
            minutes: 0,
            seconds: 10,
            frames: 5,
            drop_frame: true,
        };
        assert_eq!(tc.to_string(), "00:00:10;05");
    }

    #[test]
    fn test_timecode_to_frame_count() {
        // 1 minute at 25 fps = 1500 frames
        let tc = SmpteTimecode {
            hours: 0,
            minutes: 1,
            seconds: 0,
            frames: 0,
            drop_frame: false,
        };
        assert_eq!(tc.to_frame_count(25), 1500);
    }

    #[test]
    fn test_distributor_creation() {
        let d = TimecodeDistributor::new(TimecodeStandard::Ltc25);
        assert_eq!(d.standard().fps(), 25);
        assert_eq!(d.lock_state(), LockState::Searching);
    }

    #[test]
    fn test_distributor_advance_distributes_to_subscribers() {
        let mut d = TimecodeDistributor::new(TimecodeStandard::Ltc25);
        d.set_locked();
        d.subscribe(TimecodeSubscriber::new("ch1", "Channel 1"));
        d.advance_frame();
        let tc = d.last_tc_for("ch1");
        assert!(tc.is_some());
        assert_eq!(tc.expect("should have tc").frames, 1);
    }

    #[test]
    fn test_distributor_frame_count() {
        let mut d = TimecodeDistributor::new(TimecodeStandard::Ltc30);
        for _ in 0..30 {
            d.advance_frame();
        }
        assert_eq!(d.frame_count(), 30);
        // After 30 frames at 30fps, timecode should be 00:00:01:00
        assert_eq!(d.master().seconds, 1);
        assert_eq!(d.master().frames, 0);
    }

    #[test]
    fn test_jam_sync() {
        let mut d = TimecodeDistributor::new(TimecodeStandard::Ltc25);
        let target = SmpteTimecode {
            hours: 10,
            minutes: 0,
            seconds: 0,
            frames: 0,
            drop_frame: false,
        };
        d.jam_sync(target);
        assert_eq!(d.master().hours, 10);
    }

    #[test]
    fn test_unsubscribe() {
        let mut d = TimecodeDistributor::new(TimecodeStandard::Ltc25);
        d.subscribe(TimecodeSubscriber::new("x", "X"));
        assert_eq!(d.subscriber_count(), 1);
        d.unsubscribe("x");
        assert_eq!(d.subscriber_count(), 0);
    }

    #[test]
    fn test_ltc_standards_fps() {
        assert_eq!(TimecodeStandard::Ltc30.fps(), 30);
        assert_eq!(TimecodeStandard::Ltc25.fps(), 25);
        assert_eq!(TimecodeStandard::Ltc24.fps(), 24);
        assert!(TimecodeStandard::Ltc2997Df.is_drop_frame());
    }
}
