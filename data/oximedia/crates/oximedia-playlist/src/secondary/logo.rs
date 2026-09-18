//! Station logo/bug management.

use serde::{Deserialize, Serialize};
use std::path::PathBuf;

/// Position of the station logo.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum LogoPosition {
    /// Top left corner.
    TopLeft,
    /// Top right corner.
    TopRight,
    /// Bottom left corner.
    BottomLeft,
    /// Bottom right corner.
    BottomRight,
    /// Custom position (x, y in pixels).
    Custom {
        /// X position in pixels.
        x: i32,
        /// Y position in pixels.
        y: i32,
    },
}

/// Station logo/bug configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StationLogo {
    /// Path to logo image file.
    pub path: PathBuf,

    /// Position of the logo.
    pub position: LogoPosition,

    /// Margin from edge (pixels).
    pub margin: u32,

    /// Opacity (0.0 - 1.0).
    pub opacity: f32,

    /// Width in pixels (None = source width).
    pub width: Option<u32>,

    /// Height in pixels (None = source height).
    pub height: Option<u32>,

    /// Whether the logo is currently visible.
    pub visible: bool,
}

impl StationLogo {
    /// Creates a new station logo.
    #[must_use]
    pub fn new<P: Into<PathBuf>>(path: P, position: LogoPosition) -> Self {
        Self {
            path: path.into(),
            position,
            margin: 20,
            opacity: 1.0,
            width: None,
            height: None,
            visible: true,
        }
    }

    /// Sets the margin from edge.
    #[must_use]
    pub const fn with_margin(mut self, margin: u32) -> Self {
        self.margin = margin;
        self
    }

    /// Sets the opacity.
    #[must_use]
    pub const fn with_opacity(mut self, opacity: f32) -> Self {
        self.opacity = opacity;
        self
    }

    /// Sets the size.
    #[must_use]
    pub const fn with_size(mut self, width: u32, height: u32) -> Self {
        self.width = Some(width);
        self.height = Some(height);
        self
    }

    /// Shows the logo.
    pub fn show(&mut self) {
        self.visible = true;
    }

    /// Hides the logo.
    pub fn hide(&mut self) {
        self.visible = false;
    }

    /// Calculates the actual pixel position based on the screen size.
    #[must_use]
    pub fn calculate_position(&self, screen_width: u32, screen_height: u32) -> (i32, i32) {
        let margin = i32::try_from(self.margin).unwrap_or(20);
        let logo_width = self.width.map_or(0, |w| i32::try_from(w).unwrap_or(0));
        let logo_height = self.height.map_or(0, |h| i32::try_from(h).unwrap_or(0));

        match self.position {
            LogoPosition::TopLeft => (margin, margin),
            LogoPosition::TopRight => (
                i32::try_from(screen_width).unwrap_or(0) - logo_width - margin,
                margin,
            ),
            LogoPosition::BottomLeft => (
                margin,
                i32::try_from(screen_height).unwrap_or(0) - logo_height - margin,
            ),
            LogoPosition::BottomRight => (
                i32::try_from(screen_width).unwrap_or(0) - logo_width - margin,
                i32::try_from(screen_height).unwrap_or(0) - logo_height - margin,
            ),
            LogoPosition::Custom { x, y } => (x, y),
        }
    }
}

/// Manager for station logos.
#[derive(Debug, Default)]
pub struct LogoManager {
    logos: Vec<StationLogo>,
    active_logo_index: Option<usize>,
}

impl LogoManager {
    /// Creates a new logo manager.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Adds a logo.
    pub fn add_logo(&mut self, logo: StationLogo) {
        self.logos.push(logo);
        if self.active_logo_index.is_none() {
            self.active_logo_index = Some(0);
        }
    }

    /// Sets the active logo by index.
    pub fn set_active_logo(&mut self, index: usize) -> Result<(), String> {
        if index >= self.logos.len() {
            return Err("Logo index out of bounds".to_string());
        }
        self.active_logo_index = Some(index);
        Ok(())
    }

    /// Gets the active logo.
    #[must_use]
    pub fn get_active_logo(&self) -> Option<&StationLogo> {
        self.active_logo_index.and_then(|i| self.logos.get(i))
    }

    /// Gets a mutable reference to the active logo.
    pub fn get_active_logo_mut(&mut self) -> Option<&mut StationLogo> {
        self.active_logo_index.and_then(|i| self.logos.get_mut(i))
    }

    /// Shows the active logo.
    pub fn show(&mut self) {
        if let Some(logo) = self.get_active_logo_mut() {
            logo.show();
        }
    }

    /// Hides the active logo.
    pub fn hide(&mut self) {
        if let Some(logo) = self.get_active_logo_mut() {
            logo.hide();
        }
    }

    /// Returns the number of logos.
    #[must_use]
    pub fn len(&self) -> usize {
        self.logos.len()
    }

    /// Returns true if there are no logos.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.logos.is_empty()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_station_logo() {
        let mut logo = StationLogo::new("logo.png", LogoPosition::TopRight)
            .with_margin(30)
            .with_opacity(0.8);

        assert!(logo.visible);
        logo.hide();
        assert!(!logo.visible);
    }

    #[test]
    fn test_logo_position() {
        let logo = StationLogo::new("logo.png", LogoPosition::TopLeft)
            .with_size(100, 50)
            .with_margin(20);

        let (x, y) = logo.calculate_position(1920, 1080);
        assert_eq!(x, 20);
        assert_eq!(y, 20);
    }

    #[test]
    fn test_logo_manager() {
        let mut manager = LogoManager::new();
        let logo = StationLogo::new("logo.png", LogoPosition::TopRight);

        manager.add_logo(logo);
        assert_eq!(manager.len(), 1);
        assert!(manager.get_active_logo().is_some());
    }
}
