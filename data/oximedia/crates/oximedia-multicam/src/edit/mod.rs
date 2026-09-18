//! Multi-angle editing for multi-camera production.

pub mod lazy_frame;
pub mod switch;
pub mod timeline;
pub mod transition;

use crate::{AngleId, FrameNumber};

/// Edit decision for camera switching
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct EditDecision {
    /// Frame number where switch occurs
    pub frame: FrameNumber,
    /// Target camera angle
    pub angle: AngleId,
    /// Transition type
    pub transition: TransitionType,
    /// Transition duration in frames
    pub duration: u32,
}

/// Type of transition between angles
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TransitionType {
    /// Immediate cut
    Cut,
    /// Cross-fade/dissolve
    Dissolve,
    /// Wipe transition
    Wipe,
    /// Dip to black
    DipToBlack,
}

impl EditDecision {
    /// Create a new edit decision with cut transition
    #[must_use]
    pub fn cut(frame: FrameNumber, angle: AngleId) -> Self {
        Self {
            frame,
            angle,
            transition: TransitionType::Cut,
            duration: 0,
        }
    }

    /// Create a new edit decision with dissolve transition
    #[must_use]
    pub fn dissolve(frame: FrameNumber, angle: AngleId, duration: u32) -> Self {
        Self {
            frame,
            angle,
            transition: TransitionType::Dissolve,
            duration,
        }
    }

    /// Create a new edit decision with wipe transition
    #[must_use]
    pub fn wipe(frame: FrameNumber, angle: AngleId, duration: u32) -> Self {
        Self {
            frame,
            angle,
            transition: TransitionType::Wipe,
            duration,
        }
    }
}

/// Multi-camera timeline
pub use timeline::MultiCamTimeline;

/// Camera switcher
pub use switch::AngleSwitcher;

/// Transition engine
pub use transition::TransitionEngine;

/// Lazy frame reference
pub use lazy_frame::LazyFrameRef;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_edit_decision_cut() {
        let decision = EditDecision::cut(100, 1);
        assert_eq!(decision.frame, 100);
        assert_eq!(decision.angle, 1);
        assert_eq!(decision.transition, TransitionType::Cut);
        assert_eq!(decision.duration, 0);
    }

    #[test]
    fn test_edit_decision_dissolve() {
        let decision = EditDecision::dissolve(200, 2, 10);
        assert_eq!(decision.frame, 200);
        assert_eq!(decision.angle, 2);
        assert_eq!(decision.transition, TransitionType::Dissolve);
        assert_eq!(decision.duration, 10);
    }
}
