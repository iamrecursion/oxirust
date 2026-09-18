//! Graphics overlay management.

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::path::PathBuf;
use std::time::Duration;

/// Graphics overlay layer.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GraphicsOverlay {
    /// Unique identifier.
    pub id: String,

    /// Display name.
    pub name: String,

    /// Path to graphics file (PNG with alpha, etc.).
    pub path: PathBuf,

    /// X position (pixels from left).
    pub x: i32,

    /// Y position (pixels from top).
    pub y: i32,

    /// Width in pixels (None = source width).
    pub width: Option<u32>,

    /// Height in pixels (None = source height).
    pub height: Option<u32>,

    /// Opacity (0.0 - 1.0).
    pub opacity: f32,

    /// Z-index for layering.
    pub z_index: i32,

    /// Whether the overlay is currently visible.
    pub visible: bool,

    /// Start time for the overlay (None = manual control).
    pub start_time: Option<DateTime<Utc>>,

    /// Duration to show (None = infinite).
    pub duration: Option<Duration>,

    /// Fade in duration.
    pub fade_in: Option<Duration>,

    /// Fade out duration.
    pub fade_out: Option<Duration>,
}

impl GraphicsOverlay {
    /// Creates a new graphics overlay.
    #[must_use]
    pub fn new<P: Into<PathBuf>, S: Into<String>>(name: S, path: P, x: i32, y: i32) -> Self {
        Self {
            id: generate_id(),
            name: name.into(),
            path: path.into(),
            x,
            y,
            width: None,
            height: None,
            opacity: 1.0,
            z_index: 0,
            visible: false,
            start_time: None,
            duration: None,
            fade_in: None,
            fade_out: None,
        }
    }

    /// Sets the size of the overlay.
    #[must_use]
    pub const fn with_size(mut self, width: u32, height: u32) -> Self {
        self.width = Some(width);
        self.height = Some(height);
        self
    }

    /// Sets the opacity.
    #[must_use]
    pub const fn with_opacity(mut self, opacity: f32) -> Self {
        self.opacity = opacity;
        self
    }

    /// Sets the z-index.
    #[must_use]
    pub const fn with_z_index(mut self, z_index: i32) -> Self {
        self.z_index = z_index;
        self
    }

    /// Sets the duration.
    #[must_use]
    pub const fn with_duration(mut self, duration: Duration) -> Self {
        self.duration = Some(duration);
        self
    }

    /// Sets fade in/out durations.
    #[must_use]
    pub const fn with_fades(mut self, fade_in: Duration, fade_out: Duration) -> Self {
        self.fade_in = Some(fade_in);
        self.fade_out = Some(fade_out);
        self
    }

    /// Shows the overlay.
    pub fn show(&mut self) {
        self.visible = true;
    }

    /// Hides the overlay.
    pub fn hide(&mut self) {
        self.visible = false;
    }

    /// Checks if the overlay should be visible at a given time.
    #[must_use]
    pub fn is_active_at(&self, time: &DateTime<Utc>) -> bool {
        if !self.visible {
            return false;
        }

        if let Some(start) = self.start_time {
            if time < &start {
                return false;
            }

            if let Some(dur) = self.duration {
                let end = start
                    + chrono::Duration::from_std(dur).unwrap_or_else(|_| chrono::Duration::zero());
                if time >= &end {
                    return false;
                }
            }
        }

        true
    }
}

/// Manager for graphics overlays.
#[derive(Debug, Default)]
pub struct GraphicsManager {
    overlays: Vec<GraphicsOverlay>,
}

impl GraphicsManager {
    /// Creates a new graphics manager.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Adds a graphics overlay.
    pub fn add_overlay(&mut self, overlay: GraphicsOverlay) {
        self.overlays.push(overlay);
        self.sort_overlays();
    }

    /// Removes an overlay by ID.
    pub fn remove_overlay(&mut self, overlay_id: &str) {
        self.overlays.retain(|o| o.id != overlay_id);
    }

    /// Shows an overlay by ID.
    pub fn show_overlay(&mut self, overlay_id: &str) {
        if let Some(overlay) = self.overlays.iter_mut().find(|o| o.id == overlay_id) {
            overlay.show();
        }
    }

    /// Hides an overlay by ID.
    pub fn hide_overlay(&mut self, overlay_id: &str) {
        if let Some(overlay) = self.overlays.iter_mut().find(|o| o.id == overlay_id) {
            overlay.hide();
        }
    }

    /// Gets all visible overlays.
    #[must_use]
    pub fn get_visible_overlays(&self) -> Vec<&GraphicsOverlay> {
        self.overlays.iter().filter(|o| o.visible).collect()
    }

    /// Gets active overlays at a specific time.
    #[must_use]
    pub fn get_active_overlays(&self, time: &DateTime<Utc>) -> Vec<&GraphicsOverlay> {
        self.overlays
            .iter()
            .filter(|o| o.is_active_at(time))
            .collect()
    }

    /// Sorts overlays by z-index.
    fn sort_overlays(&mut self) {
        self.overlays.sort_by_key(|o| o.z_index);
    }

    /// Returns the number of overlays.
    #[must_use]
    pub fn len(&self) -> usize {
        self.overlays.len()
    }

    /// Returns true if there are no overlays.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.overlays.is_empty()
    }
}

fn generate_id() -> String {
    use std::time::{SystemTime, UNIX_EPOCH};
    let timestamp = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_nanos();
    format!("graphics_{timestamp}")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_graphics_overlay() {
        let mut overlay = GraphicsOverlay::new("lower_third", "lower_third.png", 100, 500)
            .with_opacity(0.9)
            .with_z_index(10);

        assert!(!overlay.visible);
        overlay.show();
        assert!(overlay.visible);
    }

    #[test]
    fn test_graphics_manager() {
        let mut manager = GraphicsManager::new();
        let overlay = GraphicsOverlay::new("test", "test.png", 0, 0);
        let overlay_id = overlay.id.clone();

        manager.add_overlay(overlay);
        assert_eq!(manager.len(), 1);

        manager.show_overlay(&overlay_id);
        assert_eq!(manager.get_visible_overlays().len(), 1);

        manager.hide_overlay(&overlay_id);
        assert_eq!(manager.get_visible_overlays().len(), 0);
    }
}
