//! Preview system for operators
//!
//! Provides operator preview with guides and multi-output support.

pub mod guides;
pub mod operator;
pub mod output;

use serde::{Deserialize, Serialize};

/// Preview configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PreviewConfig {
    /// Preview resolution
    pub resolution: (usize, usize),
    /// Enable guides
    pub guides: bool,
    /// Preview quality
    pub quality: f32,
}

impl Default for PreviewConfig {
    fn default() -> Self {
        Self {
            resolution: (1920, 1080),
            guides: true,
            quality: 0.8,
        }
    }
}
