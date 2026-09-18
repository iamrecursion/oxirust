//! Sign language video overlay support.

pub mod border;
pub mod overlay;
pub mod position;
pub mod quality;

pub use border::{SignBorder, SignBorderStyle};
pub use overlay::SignLanguageOverlay;
pub use position::{SignPosition, SignSize};

use serde::{Deserialize, Serialize};

/// Sign language video configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SignConfig {
    /// Position of sign language video.
    pub position: SignPosition,
    /// Size configuration.
    pub size: SignSize,
    /// Border style.
    pub border: Option<SignBorder>,
    /// Opacity (0.0 to 1.0).
    pub opacity: f32,
}

impl Default for SignConfig {
    fn default() -> Self {
        Self {
            position: SignPosition::BottomRight,
            size: SignSize::Medium,
            border: Some(SignBorder::default()),
            opacity: 1.0,
        }
    }
}
