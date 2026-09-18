//! Visual enhancements for accessibility.

pub mod color;
pub mod contrast;
pub mod size;

pub use color::{ColorBlindnessAdapter, ColorBlindnessType};
pub use contrast::{ContrastEnhancer, DynamicRangeAnalysis, EnhancementLevel, EnhancementParams};
pub use size::TextSizeAdjuster;

use serde::{Deserialize, Serialize};

/// Visual enhancement configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VisualConfig {
    /// Contrast enhancement level (0.0 to 1.0).
    pub contrast_level: f32,
    /// Color blindness adaptation type.
    pub color_adaptation: Option<ColorBlindnessType>,
    /// Text size multiplier.
    pub text_size_multiplier: f32,
}

impl Default for VisualConfig {
    fn default() -> Self {
        Self {
            contrast_level: 0.0,
            color_adaptation: None,
            text_size_multiplier: 1.0,
        }
    }
}
