//! Transition types for timeline clips.

use crate::types::Timecode;
use serde::{Deserialize, Serialize};

/// Transition type.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum TransitionType {
    /// Cut (no transition).
    Cut,
    /// Dissolve/crossfade.
    Dissolve,
    /// Wipe.
    Wipe,
}

/// A transition between clips.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Transition {
    /// Transition type.
    pub transition_type: TransitionType,
    /// Transition start timecode.
    pub start: Timecode,
    /// Duration in frames.
    pub duration_frames: u32,
    /// Wipe pattern (if applicable).
    pub wipe_pattern: Option<u16>,
}

impl Transition {
    /// Create a new transition.
    #[must_use]
    pub const fn new(
        transition_type: TransitionType,
        start: Timecode,
        duration_frames: u32,
    ) -> Self {
        Self {
            transition_type,
            start,
            duration_frames,
            wipe_pattern: None,
        }
    }

    /// Create a dissolve transition.
    #[must_use]
    pub const fn dissolve(start: Timecode, duration_frames: u32) -> Self {
        Self::new(TransitionType::Dissolve, start, duration_frames)
    }

    /// Create a cut (no transition).
    #[must_use]
    pub const fn cut(start: Timecode) -> Self {
        Self::new(TransitionType::Cut, start, 0)
    }

    /// Create a wipe transition.
    #[must_use]
    pub fn wipe(start: Timecode, duration_frames: u32, pattern: u16) -> Self {
        Self {
            transition_type: TransitionType::Wipe,
            start,
            duration_frames,
            wipe_pattern: Some(pattern),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dissolve_creation() {
        let tc = Timecode::new(1, 0, 0, 0);
        let transition = Transition::dissolve(tc, 30);
        assert_eq!(transition.transition_type, TransitionType::Dissolve);
        assert_eq!(transition.duration_frames, 30);
    }

    #[test]
    fn test_cut_creation() {
        let tc = Timecode::new(1, 0, 0, 0);
        let transition = Transition::cut(tc);
        assert_eq!(transition.transition_type, TransitionType::Cut);
        assert_eq!(transition.duration_frames, 0);
    }

    #[test]
    fn test_wipe_creation() {
        let tc = Timecode::new(1, 0, 0, 0);
        let transition = Transition::wipe(tc, 30, 1);
        assert_eq!(transition.transition_type, TransitionType::Wipe);
        assert_eq!(transition.wipe_pattern, Some(1));
    }
}
