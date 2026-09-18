//! Transition management between items.

use serde::{Deserialize, Serialize};
use std::time::Duration;

/// Type of transition between items.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub enum TransitionType {
    /// Hard cut (no transition).
    #[default]
    Cut,

    /// Crossfade transition.
    Crossfade {
        /// Duration of the crossfade.
        duration: Duration,
    },

    /// Fade to black then fade up.
    FadeToBlack {
        /// Fade out duration.
        fade_out: Duration,
        /// Hold black duration.
        hold: Duration,
        /// Fade in duration.
        fade_in: Duration,
    },

    /// Wipe transition.
    Wipe {
        /// Duration of the wipe.
        duration: Duration,
        /// Direction (0-360 degrees).
        direction: f32,
    },

    /// Dissolve transition.
    Dissolve {
        /// Duration of the dissolve.
        duration: Duration,
    },

    /// Custom transition.
    Custom {
        /// Name of the custom transition.
        name: String,
        /// Duration of the transition.
        duration: Duration,
    },
}

impl TransitionType {
    /// Returns the total duration of the transition.
    #[must_use]
    pub const fn duration(&self) -> Duration {
        match self {
            Self::Cut => Duration::ZERO,
            Self::Crossfade { duration }
            | Self::Dissolve { duration }
            | Self::Wipe { duration, .. }
            | Self::Custom { duration, .. } => *duration,
            Self::FadeToBlack {
                fade_out,
                hold,
                fade_in,
            } => Duration::from_millis(
                fade_out.as_millis() as u64 + hold.as_millis() as u64 + fade_in.as_millis() as u64,
            ),
        }
    }
}

/// Transition configuration between two items.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Transition {
    /// From item index.
    pub from_item: usize,

    /// To item index.
    pub to_item: usize,

    /// Type of transition.
    pub transition_type: TransitionType,

    /// Whether this transition is enabled.
    pub enabled: bool,
}

impl Transition {
    /// Creates a new transition.
    #[must_use]
    pub const fn new(from_item: usize, to_item: usize, transition_type: TransitionType) -> Self {
        Self {
            from_item,
            to_item,
            transition_type,
            enabled: true,
        }
    }

    /// Creates a simple crossfade transition.
    #[must_use]
    pub const fn crossfade(from_item: usize, to_item: usize, duration: Duration) -> Self {
        Self::new(from_item, to_item, TransitionType::Crossfade { duration })
    }

    /// Creates a fade to black transition.
    #[must_use]
    #[allow(clippy::too_many_arguments)]
    pub const fn fade_to_black(
        from_item: usize,
        to_item: usize,
        fade_out: Duration,
        hold: Duration,
        fade_in: Duration,
    ) -> Self {
        Self::new(
            from_item,
            to_item,
            TransitionType::FadeToBlack {
                fade_out,
                hold,
                fade_in,
            },
        )
    }
}

/// Manager for transitions between playlist items.
#[derive(Debug, Default)]
pub struct TransitionManager {
    transitions: Vec<Transition>,
    default_transition: TransitionType,
}

impl TransitionManager {
    /// Creates a new transition manager.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Creates a new transition manager with a default transition type.
    #[must_use]
    pub const fn with_default(default_transition: TransitionType) -> Self {
        Self {
            transitions: Vec::new(),
            default_transition,
        }
    }

    /// Adds a transition.
    pub fn add_transition(&mut self, transition: Transition) {
        self.transitions.push(transition);
    }

    /// Removes a transition between two items.
    pub fn remove_transition(&mut self, from_item: usize, to_item: usize) {
        self.transitions
            .retain(|t| !(t.from_item == from_item && t.to_item == to_item));
    }

    /// Gets the transition between two items.
    #[must_use]
    pub fn get_transition(&self, from_item: usize, to_item: usize) -> TransitionType {
        self.transitions
            .iter()
            .find(|t| t.enabled && t.from_item == from_item && t.to_item == to_item)
            .map_or(self.default_transition.clone(), |t| {
                t.transition_type.clone()
            })
    }

    /// Sets the default transition type.
    pub fn set_default_transition(&mut self, transition_type: TransitionType) {
        self.default_transition = transition_type;
    }

    /// Clears all transitions.
    pub fn clear(&mut self) {
        self.transitions.clear();
    }

    /// Returns the number of transitions.
    #[must_use]
    pub fn len(&self) -> usize {
        self.transitions.len()
    }

    /// Returns true if there are no transitions.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.transitions.is_empty()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_transition_type_duration() {
        let transition = TransitionType::Crossfade {
            duration: Duration::from_secs(1),
        };
        assert_eq!(transition.duration(), Duration::from_secs(1));

        let transition = TransitionType::FadeToBlack {
            fade_out: Duration::from_millis(500),
            hold: Duration::from_millis(200),
            fade_in: Duration::from_millis(500),
        };
        assert_eq!(transition.duration(), Duration::from_millis(1200));
    }

    #[test]
    fn test_transition_manager() {
        let mut manager = TransitionManager::new();
        let transition = Transition::crossfade(0, 1, Duration::from_secs(1));

        manager.add_transition(transition);
        assert_eq!(manager.len(), 1);

        let trans_type = manager.get_transition(0, 1);
        assert!(matches!(trans_type, TransitionType::Crossfade { .. }));
    }

    #[test]
    fn test_default_transition() {
        let manager = TransitionManager::with_default(TransitionType::Crossfade {
            duration: Duration::from_millis(500),
        });

        let trans_type = manager.get_transition(0, 1);
        assert!(matches!(trans_type, TransitionType::Crossfade { .. }));
    }
}
