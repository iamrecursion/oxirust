//! Timecode display overlay.

/// Timecode display.
pub struct TimecodeDisplay;

impl TimecodeDisplay {
    /// Create a new timecode display.
    #[must_use]
    pub fn new() -> Self {
        Self
    }

    /// Reset display.
    pub fn reset(&mut self) {}
}

impl Default for TimecodeDisplay {
    fn default() -> Self {
        Self::new()
    }
}
