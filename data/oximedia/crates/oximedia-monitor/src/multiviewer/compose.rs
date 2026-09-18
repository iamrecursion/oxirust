//! View composition for multi-viewer.

use super::ViewPane;

/// View composer.
pub struct ViewComposer;

impl ViewComposer {
    /// Create a new view composer.
    #[must_use]
    pub fn new() -> Self {
        Self
    }

    /// Compose views into a single output.
    #[must_use]
    pub fn compose(&self, _panes: &[ViewPane]) -> Vec<u8> {
        Vec::new()
    }
}

impl Default for ViewComposer {
    fn default() -> Self {
        Self::new()
    }
}
