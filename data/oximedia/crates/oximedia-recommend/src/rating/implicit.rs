//! Implicit signal processing.

use serde::{Deserialize, Serialize};
use uuid::Uuid;

/// Implicit signal from user behavior
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ImplicitSignal {
    /// User ID
    pub user_id: Uuid,
    /// Content ID
    pub content_id: Uuid,
    /// Signal type
    pub signal_type: SignalType,
    /// Signal strength (0-1)
    pub strength: f32,
    /// Timestamp
    pub timestamp: i64,
}

/// Type of implicit signal
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum SignalType {
    /// View event
    View,
    /// Completion
    Completion,
    /// Repeat view
    RepeatView,
    /// Skip
    Skip,
    /// Fast forward
    FastForward,
    /// Rewind
    Rewind,
    /// Pause
    Pause,
    /// Share
    Share,
    /// Bookmark
    Bookmark,
}

impl ImplicitSignal {
    /// Create a new implicit signal
    #[must_use]
    pub fn new(user_id: Uuid, content_id: Uuid, signal_type: SignalType, strength: f32) -> Self {
        Self {
            user_id,
            content_id,
            signal_type,
            strength: strength.clamp(0.0, 1.0),
            timestamp: chrono::Utc::now().timestamp(),
        }
    }

    /// Convert signal to rating contribution
    #[must_use]
    pub fn to_rating(&self) -> f32 {
        let base = match self.signal_type {
            SignalType::View => 2.0,
            SignalType::Completion => 4.0,
            SignalType::RepeatView => 4.5,
            SignalType::Skip => 1.0,
            SignalType::FastForward => 1.5,
            SignalType::Rewind => 3.0,
            SignalType::Pause => 2.5,
            SignalType::Share => 5.0,
            SignalType::Bookmark => 4.5,
        };

        (base * self.strength).min(5.0)
    }
}

/// Implicit signal processor
pub struct ImplicitSignalProcessor {
    /// Signal weights
    weights: SignalWeights,
}

/// Weights for different signal types
#[derive(Debug, Clone)]
pub struct SignalWeights {
    /// View weight
    pub view: f32,
    /// Completion weight
    pub completion: f32,
    /// Repeat view weight
    pub repeat_view: f32,
    /// Skip weight (negative)
    pub skip: f32,
    /// Share weight
    pub share: f32,
}

impl Default for SignalWeights {
    fn default() -> Self {
        Self {
            view: 0.5,
            completion: 1.0,
            repeat_view: 0.9,
            skip: -0.3,
            share: 1.2,
        }
    }
}

impl ImplicitSignalProcessor {
    /// Create a new implicit signal processor
    #[must_use]
    pub fn new() -> Self {
        Self {
            weights: SignalWeights::default(),
        }
    }

    /// Process multiple signals into a rating
    #[must_use]
    pub fn process_signals(&self, signals: &[ImplicitSignal]) -> f32 {
        if signals.is_empty() {
            return 0.0;
        }

        let mut weighted_sum = 0.0;
        let mut total_weight = 0.0;

        for signal in signals {
            let weight = self.get_signal_weight(&signal.signal_type);
            weighted_sum += signal.to_rating() * weight;
            total_weight += weight.abs();
        }

        if total_weight > 0.0 {
            (weighted_sum / total_weight).clamp(0.0, 5.0)
        } else {
            0.0
        }
    }

    /// Get weight for signal type
    fn get_signal_weight(&self, signal_type: &SignalType) -> f32 {
        match signal_type {
            SignalType::View => self.weights.view,
            SignalType::Completion => self.weights.completion,
            SignalType::RepeatView => self.weights.repeat_view,
            SignalType::Skip => self.weights.skip,
            SignalType::Share => self.weights.share,
            SignalType::FastForward => 0.3,
            SignalType::Rewind => 0.4,
            SignalType::Pause => 0.2,
            SignalType::Bookmark => 0.8,
        }
    }
}

impl Default for ImplicitSignalProcessor {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_implicit_signal_creation() {
        let signal = ImplicitSignal::new(Uuid::new_v4(), Uuid::new_v4(), SignalType::View, 0.8);
        assert_eq!(signal.strength, 0.8);
    }

    #[test]
    fn test_signal_to_rating() {
        let signal =
            ImplicitSignal::new(Uuid::new_v4(), Uuid::new_v4(), SignalType::Completion, 1.0);
        let rating = signal.to_rating();
        assert!((rating - 4.0).abs() < f32::EPSILON);
    }

    #[test]
    fn test_process_signals() {
        let processor = ImplicitSignalProcessor::new();
        let user_id = Uuid::new_v4();
        let content_id = Uuid::new_v4();

        let signals = vec![
            ImplicitSignal::new(user_id, content_id, SignalType::View, 1.0),
            ImplicitSignal::new(user_id, content_id, SignalType::Completion, 1.0),
        ];

        let rating = processor.process_signals(&signals);
        assert!(rating > 0.0 && rating <= 5.0);
    }
}
