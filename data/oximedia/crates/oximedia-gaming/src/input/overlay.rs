//! Input visualization overlay.

/// Input overlay for visualizing inputs on stream.
#[allow(dead_code)]
pub struct InputOverlay {
    style: OverlayStyle,
    enabled: bool,
}

/// Overlay style configuration.
#[derive(Debug, Clone)]
#[allow(dead_code)]
pub struct OverlayStyle {
    /// Show keyboard inputs
    pub show_keyboard: bool,
    /// Show mouse inputs
    pub show_mouse: bool,
    /// Show click positions
    pub show_clicks: bool,
    /// Fade duration in seconds
    pub fade_duration: f32,
}

impl InputOverlay {
    /// Create a new input overlay.
    #[must_use]
    pub fn new(style: OverlayStyle) -> Self {
        Self {
            style,
            enabled: true,
        }
    }

    /// Enable overlay.
    pub fn enable(&mut self) {
        self.enabled = true;
    }

    /// Disable overlay.
    pub fn disable(&mut self) {
        self.enabled = false;
    }

    /// Check if overlay is enabled.
    #[must_use]
    pub fn is_enabled(&self) -> bool {
        self.enabled
    }
}

impl Default for OverlayStyle {
    fn default() -> Self {
        Self {
            show_keyboard: true,
            show_mouse: true,
            show_clicks: true,
            fade_duration: 1.0,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_overlay_creation() {
        let overlay = InputOverlay::new(OverlayStyle::default());
        assert!(overlay.is_enabled());
    }
}
