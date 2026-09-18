//! Real-time chroma keying
//!
//! Provides real-time chroma key with quality modes.

pub mod quality;
pub mod realtime;

use serde::{Deserialize, Serialize};

/// Keying parameters
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KeyingParams {
    /// Key color
    pub key_color: [f32; 3],
    /// Similarity threshold
    pub similarity: f32,
    /// Smoothness
    pub smoothness: f32,
    /// Spill removal
    pub spill: f32,
}

impl Default for KeyingParams {
    fn default() -> Self {
        Self {
            key_color: [0.0, 1.0, 0.0],
            similarity: 0.5,
            smoothness: 0.1,
            spill: 0.5,
        }
    }
}
