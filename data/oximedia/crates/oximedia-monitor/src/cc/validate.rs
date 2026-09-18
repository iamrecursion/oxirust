//! Caption validation.

use super::CaptionValidation;

/// Caption validator.
pub struct CaptionValidator {
    validation: CaptionValidation,
}

impl CaptionValidator {
    /// Create a new caption validator.
    #[must_use]
    pub fn new() -> Self {
        Self {
            validation: CaptionValidation::default(),
        }
    }

    /// Get validation result.
    #[must_use]
    pub const fn validation(&self) -> &CaptionValidation {
        &self.validation
    }

    /// Reset validator.
    pub fn reset(&mut self) {
        self.validation = CaptionValidation::default();
    }
}

impl Default for CaptionValidator {
    fn default() -> Self {
        Self::new()
    }
}
