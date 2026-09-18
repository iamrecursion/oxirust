//! Timecode continuity validation.

use super::TimecodeValidation;
use crate::MonitorResult;
use oximedia_timecode::{FrameRate, Timecode};

/// Timecode validator.
pub struct TimecodeValidator {
    frame_rate: FrameRate,
    validation: TimecodeValidation,
    last_timecode: Option<Timecode>,
}

impl TimecodeValidator {
    /// Create a new timecode validator.
    #[must_use]
    pub fn new(frame_rate: FrameRate) -> Self {
        Self {
            frame_rate,
            validation: TimecodeValidation::default(),
            last_timecode: None,
        }
    }

    /// Validate timecode continuity.
    ///
    /// # Errors
    ///
    /// Returns an error if validation fails.
    pub fn validate(&mut self, timecode: &Timecode) -> MonitorResult<()> {
        self.validation.last_timecode = Some(timecode.to_string());

        if let Some(ref last_tc) = self.last_timecode {
            // Check for continuity (simplified)
            let expected_frame = last_tc.frames + 1;
            if timecode.frames != expected_frame && timecode.frames != 0 {
                self.validation.is_continuous = false;
                self.validation.discontinuities += 1;
            }
        }

        self.last_timecode = Some(*timecode);

        Ok(())
    }

    /// Get validation result.
    #[must_use]
    pub const fn validation(&self) -> &TimecodeValidation {
        &self.validation
    }

    /// Reset validator.
    pub fn reset(&mut self) {
        self.validation = TimecodeValidation::default();
        self.last_timecode = None;
    }
}
