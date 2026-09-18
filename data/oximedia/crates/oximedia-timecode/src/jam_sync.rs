#![allow(dead_code)]
//! Jam-sync controller for locking a local timecode generator to an external
//! timecode reference (LTC/MTC).
//!
//! # Overview
//!
//! A [`JamSyncController`] ingests external timecode frames via
//! [`feed_reference`](JamSyncController::feed_reference).  Once
//! `lock_threshold` consecutive frames arrive that are exactly one frame apart
//! (within `tolerance` frames), the controller transitions to
//! [`Locked`](JamSyncState::Locked) and slaves its internal
//! [`TimecodeGenerator`] to the reference.
//!
//! If the reference disappears (no call to `feed_reference` for more than
//! `holdover_budget` increments), the controller transitions to
//! [`Holdover`](JamSyncState::Holdover): the local generator keeps running
//! from the last known-good position.  Calls to [`output`](JamSyncController::output)
//! continue to return valid, incrementing timecodes during holdover.

use crate::{timecode_generator::TimecodeGenerator, FrameRate, Timecode, TimecodeError};

/// The synchronisation state of a [`JamSyncController`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum JamSyncState {
    /// No reference has been received yet (or the controller was just reset).
    WaitingForReference,
    /// Candidate frames are arriving and being evaluated; the controller has
    /// not yet accumulated `lock_threshold` consecutive matching frames.
    Locking,
    /// The controller is locked to the external reference.
    Locked,
    /// The reference has been lost; the local generator is free-running from
    /// the last locked position.
    Holdover,
}

/// Configuration for a [`JamSyncController`].
#[derive(Debug, Clone, Copy)]
pub struct JamSyncConfig {
    /// Number of consecutive reference frames that must be exactly sequential
    /// before the controller transitions from `Locking` to `Locked`.
    pub lock_threshold: usize,
    /// Maximum difference in frames between consecutive reference inputs
    /// that is still considered "sequential" (for tolerance of jittery inputs).
    pub tolerance_frames: u64,
    /// Number of `output()` calls (frames) that can elapse without a reference
    /// before the controller transitions from `Locked` to `Holdover`.
    pub holdover_budget: u64,
}

impl Default for JamSyncConfig {
    fn default() -> Self {
        Self {
            lock_threshold: 5,
            tolerance_frames: 2,
            holdover_budget: 25, // ≈ 1 second at 25 fps
        }
    }
}

/// Controller that synchronises a local timecode generator to an external
/// timecode reference.
///
/// # State machine
///
/// ```text
/// WaitingForReference ──(first feed_reference)──► Locking
///       ▲                                              │
///       │              ┌───────────────────────────────┘
///       │              │  N consecutive sequential frames
///       │              ▼
///       │           Locked ◄─────────────────────────────┐
///       │              │                                  │
///       │   (reference lost > holdover_budget)    (reference resumes)
///       │              ▼                                  │
///       │          Holdover ──────────────────────────────┘
///       │              │
///       └──(reset())───┘
/// ```
pub struct JamSyncController {
    /// Current synchronisation state.
    state: JamSyncState,
    /// Configuration.
    config: JamSyncConfig,
    /// Local free-running generator (used in Locked/Holdover).
    generator: TimecodeGenerator,
    /// Frame rate of the reference (and local generator).
    frame_rate: FrameRate,
    /// Candidate window: recent reference frames used during `Locking`.
    candidate_window: Vec<Timecode>,
    /// Last reference frame received (used to detect holdover).
    last_reference: Option<Timecode>,
    /// Number of output() calls since the last feed_reference() call.
    frames_since_ref: u64,
    /// Number of consecutive in-tolerance reference frames seen so far.
    consecutive_count: usize,
}

impl JamSyncController {
    /// Create a new controller for the given `frame_rate` using the provided
    /// `config`.
    ///
    /// # Errors
    ///
    /// Returns an error if `TimecodeGenerator::at_midnight` fails (should not
    /// occur for well-defined frame rates).
    pub fn new(frame_rate: FrameRate, config: JamSyncConfig) -> Result<Self, TimecodeError> {
        let generator = TimecodeGenerator::at_midnight(frame_rate)?;
        Ok(Self {
            state: JamSyncState::WaitingForReference,
            config,
            generator,
            frame_rate,
            candidate_window: Vec::new(),
            last_reference: None,
            frames_since_ref: 0,
            consecutive_count: 0,
        })
    }

    /// Create a controller with default configuration.
    ///
    /// # Errors
    ///
    /// See [`new`](Self::new).
    pub fn with_default_config(frame_rate: FrameRate) -> Result<Self, TimecodeError> {
        Self::new(frame_rate, JamSyncConfig::default())
    }

    /// Current synchronisation state.
    pub fn state(&self) -> JamSyncState {
        self.state
    }

    /// Feed an incoming reference timecode frame.
    ///
    /// This drives the state machine forward:
    /// * `WaitingForReference` → `Locking` on the first call.
    /// * `Locking` → `Locked` after `lock_threshold` consecutive sequential
    ///   frames.
    /// * `Holdover` → `Locked` (immediate re-lock) when a sequential frame
    ///   arrives.
    pub fn feed_reference(&mut self, tc: Timecode) {
        self.frames_since_ref = 0;

        match self.state {
            JamSyncState::WaitingForReference => {
                self.last_reference = Some(tc);
                self.consecutive_count = 1;
                self.state = JamSyncState::Locking;
            }

            JamSyncState::Locking => {
                if self.is_sequential(tc) {
                    self.consecutive_count += 1;
                    if self.consecutive_count >= self.config.lock_threshold {
                        // Lock acquired — jam the local generator to this position
                        self.generator.reset_to(tc);
                        // Advance by one so the *next* output() is already ahead
                        let _ = self.generator.next();
                        self.state = JamSyncState::Locked;
                    }
                } else {
                    // Non-sequential → restart counting
                    self.consecutive_count = 1;
                }
                self.last_reference = Some(tc);
            }

            JamSyncState::Locked => {
                // Re-jam if the reference drifts more than tolerance
                if !self.is_sequential(tc) {
                    // Slip — re-jam immediately to stay frame-accurate
                    self.generator.reset_to(tc);
                    let _ = self.generator.next();
                }
                self.last_reference = Some(tc);
            }

            JamSyncState::Holdover => {
                // Any sequential or near-sequential frame re-locks
                self.generator.reset_to(tc);
                let _ = self.generator.next();
                self.last_reference = Some(tc);
                self.consecutive_count = 1;
                self.state = JamSyncState::Locked;
            }
        }
    }

    /// Return the current output timecode.
    ///
    /// In `Locked` or `Holdover` state the local generator advances by one
    /// frame each call.  In `WaitingForReference` or `Locking` the generator
    /// is frozen until lock is acquired.
    ///
    /// Calling `output()` also updates the holdover counter: if
    /// `frames_since_ref > holdover_budget` while in `Locked` state the
    /// controller transitions to `Holdover`.
    pub fn output(&mut self) -> Timecode {
        // Update holdover counter
        self.frames_since_ref += 1;

        match self.state {
            JamSyncState::Locked => {
                if self.frames_since_ref > self.config.holdover_budget {
                    self.state = JamSyncState::Holdover;
                }
                self.generator.next()
            }
            JamSyncState::Holdover => self.generator.next(),
            // Not yet locked: peek without advancing
            _ => self.generator.peek(),
        }
    }

    /// Reset the controller to `WaitingForReference` state.
    ///
    /// The local generator is reset to midnight.
    ///
    /// # Errors
    ///
    /// Returns an error if resetting the generator fails.
    pub fn reset(&mut self) -> Result<(), TimecodeError> {
        self.state = JamSyncState::WaitingForReference;
        self.last_reference = None;
        self.frames_since_ref = 0;
        self.consecutive_count = 0;
        self.candidate_window.clear();
        self.generator.reset()
    }

    /// Force the controller into `Holdover` state (e.g. reference cable
    /// disconnected).
    pub fn enter_holdover(&mut self) {
        if self.state == JamSyncState::Locked {
            self.state = JamSyncState::Holdover;
        }
    }

    /// Number of `output()` calls since the last `feed_reference()` call.
    pub fn frames_since_reference(&self) -> u64 {
        self.frames_since_ref
    }

    // ------------------------------------------------------------------
    // Internal helpers
    // ------------------------------------------------------------------

    /// Return `true` if `incoming` is "sequential" relative to the last
    /// reference, i.e. `|incoming.to_frames() - last.to_frames() - 1| <=
    /// tolerance`.
    fn is_sequential(&self, incoming: Timecode) -> bool {
        match self.last_reference {
            None => false,
            Some(last) => {
                let expected = last.to_frames() + 1;
                let actual = incoming.to_frames();
                // Use saturating arithmetic to avoid underflow
                let diff = if actual >= expected {
                    actual - expected
                } else {
                    expected - actual
                };
                diff <= self.config.tolerance_frames
            }
        }
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    fn make_ctrl() -> JamSyncController {
        JamSyncController::with_default_config(FrameRate::Fps25).expect("ok")
    }

    /// Build a sequence of `n` consecutive timecodes starting at `start`.
    fn seq(start: Timecode, n: usize) -> Vec<Timecode> {
        let mut v = Vec::with_capacity(n);
        let mut cur = start;
        for _ in 0..n {
            v.push(cur);
            let _ = cur.increment();
        }
        v
    }

    #[test]
    fn test_initial_state_is_waiting() {
        let ctrl = make_ctrl();
        assert_eq!(ctrl.state(), JamSyncState::WaitingForReference);
    }

    #[test]
    fn test_first_feed_transitions_to_locking() {
        let mut ctrl = make_ctrl();
        let tc = Timecode::new(0, 0, 0, 0, FrameRate::Fps25).expect("valid");
        ctrl.feed_reference(tc);
        assert_eq!(ctrl.state(), JamSyncState::Locking);
    }

    #[test]
    fn test_lock_acquired_after_threshold() {
        let mut ctrl = make_ctrl();
        let start = Timecode::new(0, 0, 0, 0, FrameRate::Fps25).expect("valid");
        for tc in seq(start, ctrl.config.lock_threshold) {
            ctrl.feed_reference(tc);
        }
        assert_eq!(ctrl.state(), JamSyncState::Locked);
    }

    #[test]
    fn test_lock_not_acquired_before_threshold() {
        let mut ctrl = make_ctrl();
        let start = Timecode::new(0, 0, 0, 0, FrameRate::Fps25).expect("valid");
        // Feed one fewer than threshold
        for tc in seq(start, ctrl.config.lock_threshold - 1) {
            ctrl.feed_reference(tc);
        }
        assert_eq!(ctrl.state(), JamSyncState::Locking);
    }

    #[test]
    fn test_non_sequential_resets_lock_count() {
        let mut ctrl = make_ctrl();
        let start = Timecode::new(0, 0, 0, 0, FrameRate::Fps25).expect("valid");
        // Feed threshold - 1 sequential frames
        for tc in seq(start, ctrl.config.lock_threshold - 1) {
            ctrl.feed_reference(tc);
        }
        // Feed a non-sequential frame (jump by 100)
        let jump = Timecode::new(0, 0, 10, 0, FrameRate::Fps25).expect("valid");
        ctrl.feed_reference(jump);
        assert_eq!(ctrl.state(), JamSyncState::Locking);
        // consecutive_count should be back to 1
        assert_eq!(ctrl.consecutive_count, 1);
    }

    #[test]
    fn test_output_tracks_reference_after_lock() {
        let mut ctrl = make_ctrl();
        let start = Timecode::new(1, 0, 0, 0, FrameRate::Fps25).expect("valid");
        for tc in seq(start, ctrl.config.lock_threshold) {
            ctrl.feed_reference(tc);
        }
        assert_eq!(ctrl.state(), JamSyncState::Locked);
        // After lock, output should be advancing from (start + threshold)
        let out = ctrl.output();
        let expected_frames = start.to_frames() + ctrl.config.lock_threshold as u64;
        assert_eq!(out.to_frames(), expected_frames);
    }

    #[test]
    fn test_holdover_triggered_after_budget_exceeded() {
        let budget = 5u64;
        let config = JamSyncConfig {
            lock_threshold: 3,
            tolerance_frames: 2,
            holdover_budget: budget,
        };
        let mut ctrl = JamSyncController::new(FrameRate::Fps25, config).expect("ok");
        let start = Timecode::new(0, 0, 0, 0, FrameRate::Fps25).expect("valid");
        for tc in seq(start, ctrl.config.lock_threshold) {
            ctrl.feed_reference(tc);
        }
        assert_eq!(ctrl.state(), JamSyncState::Locked);
        // Exhaust the holdover budget by calling output() without feeding
        for _ in 0..=budget {
            ctrl.output();
        }
        assert_eq!(ctrl.state(), JamSyncState::Holdover);
    }

    #[test]
    fn test_holdover_keeps_advancing() {
        let mut ctrl = make_ctrl();
        let start = Timecode::new(0, 0, 0, 0, FrameRate::Fps25).expect("valid");
        for tc in seq(start, ctrl.config.lock_threshold) {
            ctrl.feed_reference(tc);
        }
        ctrl.enter_holdover();
        assert_eq!(ctrl.state(), JamSyncState::Holdover);
        let f0 = ctrl.output().to_frames();
        let f1 = ctrl.output().to_frames();
        assert_eq!(f1, f0 + 1, "generator must keep advancing in holdover");
    }

    #[test]
    fn test_re_lock_from_holdover() {
        let mut ctrl = make_ctrl();
        let start = Timecode::new(0, 0, 0, 0, FrameRate::Fps25).expect("valid");
        for tc in seq(start, ctrl.config.lock_threshold) {
            ctrl.feed_reference(tc);
        }
        ctrl.enter_holdover();
        // Feed one new reference frame → immediate re-lock
        let new_ref = Timecode::new(0, 1, 0, 0, FrameRate::Fps25).expect("valid");
        ctrl.feed_reference(new_ref);
        assert_eq!(ctrl.state(), JamSyncState::Locked);
    }

    #[test]
    fn test_reset_returns_to_waiting() {
        let mut ctrl = make_ctrl();
        let start = Timecode::new(0, 0, 0, 0, FrameRate::Fps25).expect("valid");
        for tc in seq(start, ctrl.config.lock_threshold) {
            ctrl.feed_reference(tc);
        }
        ctrl.reset().expect("reset ok");
        assert_eq!(ctrl.state(), JamSyncState::WaitingForReference);
    }

    #[test]
    fn test_output_frozen_while_locking() {
        let mut ctrl = make_ctrl();
        let tc = Timecode::new(0, 0, 0, 0, FrameRate::Fps25).expect("valid");
        ctrl.feed_reference(tc);
        assert_eq!(ctrl.state(), JamSyncState::Locking);
        // output() should not advance while locking
        let o1 = ctrl.output();
        let o2 = ctrl.output();
        assert_eq!(o1, o2, "output must be frozen during locking");
    }

    #[test]
    fn test_frames_since_reference_counter() {
        let mut ctrl = make_ctrl();
        let start = Timecode::new(0, 0, 0, 0, FrameRate::Fps25).expect("valid");
        for tc in seq(start, ctrl.config.lock_threshold) {
            ctrl.feed_reference(tc);
        }
        // After lock, call output 3 times without feeding
        for _ in 0..3 {
            ctrl.output();
        }
        assert_eq!(ctrl.frames_since_reference(), 3);
        // Feeding a new frame resets it
        let new_tc = Timecode::new(0, 0, 5, 0, FrameRate::Fps25).expect("valid");
        ctrl.feed_reference(new_tc);
        assert_eq!(ctrl.frames_since_reference(), 0);
    }
}
