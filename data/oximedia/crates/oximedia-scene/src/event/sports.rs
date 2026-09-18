//! Sports-specific event detection.

use crate::common::Confidence;
use crate::error::SceneResult;
use serde::{Deserialize, Serialize};

/// Sports event types.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum SportsEventType {
    /// Goal scored.
    Goal,
    /// Shot on goal.
    Shot,
    /// Foul or penalty.
    Foul,
    /// Celebration.
    Celebration,
    /// Unknown sports event.
    Unknown,
}

/// Sports event.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SportsEvent {
    /// Event type.
    pub event_type: SportsEventType,
    /// Frame number.
    pub frame_number: usize,
    /// Detection confidence.
    pub confidence: Confidence,
}

/// Sports event detector.
pub struct SportsEventDetector;

impl SportsEventDetector {
    /// Create a new sports event detector.
    #[must_use]
    pub fn new() -> Self {
        Self
    }

    /// Detect sports events.
    ///
    /// # Errors
    ///
    /// Returns error if detection fails.
    pub fn detect(
        &self,
        _frames: &[&[u8]],
        _width: usize,
        _height: usize,
    ) -> SceneResult<Vec<SportsEvent>> {
        // Simplified implementation
        Ok(Vec::new())
    }
}

impl Default for SportsEventDetector {
    fn default() -> Self {
        Self::new()
    }
}
