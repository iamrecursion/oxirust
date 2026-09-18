//! Timecode monitoring and validation.

pub mod display;
pub mod validate;

use crate::MonitorResult;
use oximedia_timecode::{FrameRate, Timecode};
use serde::{Deserialize, Serialize};

pub use display::TimecodeDisplay;
pub use validate::TimecodeValidator;

/// Timecode validation result.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct TimecodeValidation {
    /// Is continuous?
    pub is_continuous: bool,

    /// Discontinuities detected.
    pub discontinuities: u64,

    /// Last timecode.
    pub last_timecode: Option<String>,
}

/// Timecode monitor.
pub struct TimecodeMonitor {
    frame_rate: FrameRate,
    display: TimecodeDisplay,
    validator: TimecodeValidator,
}

impl TimecodeMonitor {
    /// Create a new timecode monitor.
    ///
    /// # Errors
    ///
    /// Returns an error if initialization fails.
    pub fn new(frame_rate: FrameRate) -> MonitorResult<Self> {
        Ok(Self {
            frame_rate,
            display: TimecodeDisplay::new(),
            validator: TimecodeValidator::new(frame_rate),
        })
    }

    /// Process timecode.
    ///
    /// # Errors
    ///
    /// Returns an error if processing fails.
    pub fn process_timecode(&mut self, timecode: &Timecode) -> MonitorResult<()> {
        self.validator.validate(timecode)?;
        Ok(())
    }

    /// Get validation result.
    #[must_use]
    pub fn validation(&self) -> &TimecodeValidation {
        self.validator.validation()
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
    fn test_timecode_monitor() {
        let result = TimecodeMonitor::new(FrameRate::Fps25);
        assert!(result.is_ok());
    }
}
