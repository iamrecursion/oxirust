//! High-level transition orchestration for video switchers.
//!
//! This module provides an orchestrator layer on top of the low-level
//! transition engine in `transition.rs`. It tracks multi-step transition
//! sequences, holds the requested style and duration, and exposes a simple
//! state machine for controlling the progression of a transition.

#![allow(dead_code)]

/// Style of a switcher transition.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum TransitionStyle {
    /// Instantaneous switch with no blend.
    Cut,
    /// Linear cross-dissolve.
    Mix,
    /// Horizontal wipe.
    WipeHorizontal,
    /// Vertical wipe.
    WipeVertical,
    /// Diagonal wipe.
    WipeDiagonal,
    /// Digital Video Effects (DVE) push.
    DvePush,
    /// Stinger (animated clip-based) transition.
    Stinger,
}

impl TransitionStyle {
    /// Returns `true` if the style requires frame-by-frame rendering.
    #[must_use]
    pub fn requires_rendering(self) -> bool {
        !matches!(self, Self::Cut)
    }

    /// Default duration in frames for this transition style.
    #[must_use]
    pub fn default_duration_frames(self) -> u32 {
        match self {
            Self::Cut => 0,
            Self::Mix => 25,
            Self::WipeHorizontal | Self::WipeVertical | Self::WipeDiagonal => 20,
            Self::DvePush => 30,
            Self::Stinger => 50,
        }
    }
}

impl std::fmt::Display for TransitionStyle {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let s = match self {
            Self::Cut => "Cut",
            Self::Mix => "Mix",
            Self::WipeHorizontal => "Wipe (H)",
            Self::WipeVertical => "Wipe (V)",
            Self::WipeDiagonal => "Wipe (Diagonal)",
            Self::DvePush => "DVE Push",
            Self::Stinger => "Stinger",
        };
        write!(f, "{s}")
    }
}

/// Current state of the transition state machine.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TransitionPhase {
    /// Idle — no transition in progress.
    Idle,
    /// Transition has been armed but not yet started.
    Armed,
    /// Transition is actively running.
    Running,
    /// Transition finished but the result has not been committed.
    Complete,
}

/// Orchestrates a single transition between two sources.
#[derive(Debug, Clone)]
pub struct TransitionOrchestrator {
    style: TransitionStyle,
    duration_frames: u32,
    elapsed_frames: u32,
    phase: TransitionPhase,
    /// Source that is currently on program.
    from_source: u32,
    /// Source that will be taken to program after the transition.
    to_source: u32,
}

impl TransitionOrchestrator {
    /// Create a new orchestrator with the given style.
    #[must_use]
    pub fn new(style: TransitionStyle) -> Self {
        let duration = style.default_duration_frames();
        Self {
            style,
            duration_frames: duration,
            elapsed_frames: 0,
            phase: TransitionPhase::Idle,
            from_source: 0,
            to_source: 0,
        }
    }

    /// Override the transition duration.
    pub fn set_duration(&mut self, frames: u32) {
        self.duration_frames = frames;
    }

    /// Arm the transition for sources `from` → `to`.
    ///
    /// Can be called while idle or when a previous transition has completed.
    pub fn arm(&mut self, from_source: u32, to_source: u32) {
        self.from_source = from_source;
        self.to_source = to_source;
        self.elapsed_frames = 0;
        self.phase = TransitionPhase::Armed;
    }

    /// Start the transition (move from Armed to Running).
    ///
    /// Returns `false` if not in the Armed state.
    pub fn start(&mut self) -> bool {
        if self.phase == TransitionPhase::Armed {
            self.phase = TransitionPhase::Running;
            // A Cut completes immediately.
            if self.style == TransitionStyle::Cut {
                self.phase = TransitionPhase::Complete;
            }
            true
        } else {
            false
        }
    }

    /// Advance the transition by one frame.
    ///
    /// Returns `true` if the transition just completed on this call.
    pub fn tick(&mut self) -> bool {
        if self.phase != TransitionPhase::Running {
            return false;
        }
        self.elapsed_frames += 1;
        if self.elapsed_frames >= self.duration_frames {
            self.phase = TransitionPhase::Complete;
            return true;
        }
        false
    }

    /// Commit the completed transition and return to Idle.
    ///
    /// Returns the new program source ID, or `None` if not complete.
    pub fn commit(&mut self) -> Option<u32> {
        if self.phase == TransitionPhase::Complete {
            self.phase = TransitionPhase::Idle;
            self.elapsed_frames = 0;
            Some(self.to_source)
        } else {
            None
        }
    }

    /// Abort a running or armed transition without committing it.
    pub fn abort(&mut self) {
        self.phase = TransitionPhase::Idle;
        self.elapsed_frames = 0;
    }

    /// Fractional progress in [0.0, 1.0].
    #[allow(clippy::cast_precision_loss)]
    #[must_use]
    pub fn progress(&self) -> f32 {
        if self.duration_frames == 0 {
            return 1.0;
        }
        (self.elapsed_frames as f32 / self.duration_frames as f32).min(1.0)
    }

    /// Current phase.
    #[must_use]
    pub fn phase(&self) -> TransitionPhase {
        self.phase
    }

    /// Current style.
    #[must_use]
    pub fn style(&self) -> TransitionStyle {
        self.style
    }

    /// Duration in frames.
    #[must_use]
    pub fn duration_frames(&self) -> u32 {
        self.duration_frames
    }

    /// Frames elapsed so far.
    #[must_use]
    pub fn elapsed_frames(&self) -> u32 {
        self.elapsed_frames
    }

    /// Remaining frames until completion (0 if idle or complete).
    #[must_use]
    pub fn remaining_frames(&self) -> u32 {
        self.duration_frames.saturating_sub(self.elapsed_frames)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_new_orchestrator_is_idle() {
        let orch = TransitionOrchestrator::new(TransitionStyle::Mix);
        assert_eq!(orch.phase(), TransitionPhase::Idle);
    }

    #[test]
    fn test_arm_moves_to_armed() {
        let mut orch = TransitionOrchestrator::new(TransitionStyle::Mix);
        orch.arm(1, 2);
        assert_eq!(orch.phase(), TransitionPhase::Armed);
    }

    #[test]
    fn test_start_moves_to_running() {
        let mut orch = TransitionOrchestrator::new(TransitionStyle::Mix);
        orch.arm(1, 2);
        let ok = orch.start();
        assert!(ok);
        assert_eq!(orch.phase(), TransitionPhase::Running);
    }

    #[test]
    fn test_cut_completes_immediately_on_start() {
        let mut orch = TransitionOrchestrator::new(TransitionStyle::Cut);
        orch.arm(1, 2);
        orch.start();
        assert_eq!(orch.phase(), TransitionPhase::Complete);
    }

    #[test]
    fn test_tick_advances_frames() {
        let mut orch = TransitionOrchestrator::new(TransitionStyle::Mix);
        orch.set_duration(5);
        orch.arm(1, 2);
        orch.start();
        for _ in 0..4 {
            assert!(!orch.tick());
        }
        let done = orch.tick();
        assert!(done);
        assert_eq!(orch.phase(), TransitionPhase::Complete);
    }

    #[test]
    fn test_commit_returns_new_source() {
        let mut orch = TransitionOrchestrator::new(TransitionStyle::Cut);
        orch.arm(1, 5);
        orch.start();
        let new_prog = orch.commit();
        assert_eq!(new_prog, Some(5));
        assert_eq!(orch.phase(), TransitionPhase::Idle);
    }

    #[test]
    fn test_commit_before_complete_returns_none() {
        let mut orch = TransitionOrchestrator::new(TransitionStyle::Mix);
        orch.arm(1, 2);
        orch.start();
        assert_eq!(orch.commit(), None);
    }

    #[test]
    fn test_abort_resets_to_idle() {
        let mut orch = TransitionOrchestrator::new(TransitionStyle::Mix);
        orch.arm(1, 2);
        orch.start();
        orch.abort();
        assert_eq!(orch.phase(), TransitionPhase::Idle);
        assert_eq!(orch.elapsed_frames(), 0);
    }

    #[test]
    fn test_progress_zero_when_idle() {
        let orch = TransitionOrchestrator::new(TransitionStyle::Mix);
        assert!((orch.progress() - 0.0_f32).abs() < f32::EPSILON);
    }

    #[test]
    fn test_progress_increases_with_ticks() {
        let mut orch = TransitionOrchestrator::new(TransitionStyle::Mix);
        orch.set_duration(10);
        orch.arm(1, 2);
        orch.start();
        orch.tick();
        orch.tick();
        orch.tick();
        assert!((orch.progress() - 0.3_f32).abs() < 0.01);
    }

    #[test]
    fn test_remaining_frames() {
        let mut orch = TransitionOrchestrator::new(TransitionStyle::WipeHorizontal);
        orch.set_duration(10);
        orch.arm(1, 2);
        orch.start();
        orch.tick();
        assert_eq!(orch.remaining_frames(), 9);
    }

    #[test]
    fn test_transition_style_display() {
        assert_eq!(TransitionStyle::Cut.to_string(), "Cut");
        assert_eq!(TransitionStyle::Mix.to_string(), "Mix");
        assert_eq!(TransitionStyle::WipeHorizontal.to_string(), "Wipe (H)");
    }

    #[test]
    fn test_transition_style_requires_rendering() {
        assert!(!TransitionStyle::Cut.requires_rendering());
        assert!(TransitionStyle::Mix.requires_rendering());
        assert!(TransitionStyle::DvePush.requires_rendering());
    }

    #[test]
    fn test_default_duration_frames() {
        assert_eq!(TransitionStyle::Cut.default_duration_frames(), 0);
        assert_eq!(TransitionStyle::Mix.default_duration_frames(), 25);
        assert_eq!(TransitionStyle::Stinger.default_duration_frames(), 50);
    }

    #[test]
    fn test_start_without_arm_returns_false() {
        let mut orch = TransitionOrchestrator::new(TransitionStyle::Mix);
        let ok = orch.start();
        assert!(!ok);
        assert_eq!(orch.phase(), TransitionPhase::Idle);
    }
}
