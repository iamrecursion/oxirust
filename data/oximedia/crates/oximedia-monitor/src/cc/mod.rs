//! Closed caption monitoring.

pub mod display;
pub mod validate;

use crate::MonitorResult;
use serde::{Deserialize, Serialize};

pub use display::CaptionDisplay;
pub use validate::CaptionValidator;

/// Caption validation result.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct CaptionValidation {
    /// Is valid?
    pub is_valid: bool,

    /// Validation errors.
    pub errors: Vec<String>,

    /// Caption count.
    pub caption_count: u64,
}

/// Caption monitor.
pub struct CaptionMonitor {
    display: CaptionDisplay,
    validator: CaptionValidator,
}

impl CaptionMonitor {
    /// Create a new caption monitor.
    ///
    /// # Errors
    ///
    /// Returns an error if initialization fails.
    pub fn new() -> MonitorResult<Self> {
        Ok(Self {
            display: CaptionDisplay::new(),
            validator: CaptionValidator::new(),
        })
    }

    /// Reset monitor.
    pub fn reset(&mut self) {
        self.display.reset();
        self.validator.reset();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_caption_monitor() {
        let result = CaptionMonitor::new();
        assert!(result.is_ok());
    }
}
