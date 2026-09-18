//! Green screen alternative using LED walls
//!
//! Provides LED wall as green screen alternative with spill suppression.

pub mod alternative;
pub mod spill;

use serde::{Deserialize, Serialize};

/// Green screen configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GreenScreenConfig {
    /// Key color (RGB)
    pub key_color: [u8; 3],
    /// Threshold
    pub threshold: f32,
    /// Enable spill suppression
    pub spill_suppression: bool,
}

impl Default for GreenScreenConfig {
    fn default() -> Self {
        Self {
            key_color: [0, 255, 0], // Green
            threshold: 0.5,
            spill_suppression: true,
        }
    }
}
