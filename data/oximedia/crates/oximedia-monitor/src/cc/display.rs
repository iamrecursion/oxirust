//! Caption display rendering.

/// Caption display.
pub struct CaptionDisplay;

impl CaptionDisplay {
    /// Create a new caption display.
    #[must_use]
    pub fn new() -> Self {
        Self
    }

    /// Reset display.
    pub fn reset(&mut self) {}
}

impl Default for CaptionDisplay {
    fn default() -> Self {
        Self::new()
    }
}
