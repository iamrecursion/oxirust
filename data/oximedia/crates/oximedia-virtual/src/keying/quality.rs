//! Keying quality modes

use crate::QualityMode;

/// Quality settings for keying
pub struct KeyingQuality {
    mode: QualityMode,
}

impl KeyingQuality {
    /// Create new quality settings
    #[must_use]
    pub fn new(mode: QualityMode) -> Self {
        Self { mode }
    }

    /// Get sample count for quality mode
    #[must_use]
    pub fn sample_count(&self) -> usize {
        match self.mode {
            QualityMode::Draft => 1,
            QualityMode::Preview => 4,
            QualityMode::Final => 16,
        }
    }
}
