//! Layout management for multi-viewer.

use super::ViewerLayout;

/// Layout manager.
pub struct LayoutManager {
    current_layout: ViewerLayout,
}

impl LayoutManager {
    /// Create a new layout manager.
    #[must_use]
    pub fn new() -> Self {
        Self {
            current_layout: ViewerLayout::Single,
        }
    }

    /// Get current layout.
    #[must_use]
    pub const fn current_layout(&self) -> ViewerLayout {
        self.current_layout
    }

    /// Set layout.
    pub fn set_layout(&mut self, layout: ViewerLayout) {
        self.current_layout = layout;
    }
}

impl Default for LayoutManager {
    fn default() -> Self {
        Self::new()
    }
}
