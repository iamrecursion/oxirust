//! Framing guides and overlays

use serde::{Deserialize, Serialize};

/// Guide type
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum GuideType {
    /// Rule of thirds
    RuleOfThirds,
    /// Safe area
    SafeArea,
    /// Center cross
    CenterCross,
    /// Aspect ratio
    AspectRatio,
}

/// Guide overlay
pub struct GuideOverlay {
    #[allow(dead_code)]
    guide_type: GuideType,
}

impl GuideOverlay {
    /// Create new guide overlay
    #[must_use]
    pub fn new(guide_type: GuideType) -> Self {
        Self { guide_type }
    }

    /// Draw guide on frame
    pub fn draw(&self, _frame: &mut [u8], _width: usize, _height: usize) {
        // Guide drawing implementation
    }
}
