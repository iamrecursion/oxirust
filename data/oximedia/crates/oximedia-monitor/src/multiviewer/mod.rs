//! Multi-viewer layout and composition.

pub mod compose;
pub mod layout;

use serde::{Deserialize, Serialize};

pub use compose::ViewComposer;
pub use layout::LayoutManager;

/// Viewer layout type.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ViewerLayout {
    /// Single view (full screen).
    Single,

    /// 2x2 grid.
    Grid2x2,

    /// 3x3 grid.
    Grid3x3,

    /// 4x4 grid.
    Grid4x4,

    /// Picture-in-picture.
    PictureInPicture,

    /// Custom layout.
    Custom,
}

/// View pane.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ViewPane {
    /// Pane ID.
    pub id: String,

    /// X position (0.0-1.0).
    pub x: f32,

    /// Y position (0.0-1.0).
    pub y: f32,

    /// Width (0.0-1.0).
    pub width: f32,

    /// Height (0.0-1.0).
    pub height: f32,

    /// Content type.
    pub content_type: ViewContentType,
}

/// View content type.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum ViewContentType {
    /// Video feed.
    Video,

    /// Waveform scope.
    Waveform,

    /// Vectorscope.
    Vectorscope,

    /// Histogram.
    Histogram,

    /// Audio meters.
    AudioMeters,

    /// Status display.
    Status,
}

/// Multi-viewer.
pub struct MultiViewer {
    layout: ViewerLayout,
    panes: Vec<ViewPane>,
}

impl MultiViewer {
    /// Create a new multi-viewer.
    #[must_use]
    pub fn new(layout: ViewerLayout) -> Self {
        let panes = Self::create_layout_panes(&layout);

        Self { layout, panes }
    }

    /// Get current layout.
    #[must_use]
    pub const fn layout(&self) -> ViewerLayout {
        self.layout
    }

    /// Get panes.
    #[must_use]
    pub fn panes(&self) -> &[ViewPane] {
        &self.panes
    }

    /// Set layout.
    pub fn set_layout(&mut self, layout: ViewerLayout) {
        self.layout = layout;
        self.panes = Self::create_layout_panes(&layout);
    }

    fn create_layout_panes(layout: &ViewerLayout) -> Vec<ViewPane> {
        match layout {
            ViewerLayout::Single => {
                vec![ViewPane {
                    id: "main".to_string(),
                    x: 0.0,
                    y: 0.0,
                    width: 1.0,
                    height: 1.0,
                    content_type: ViewContentType::Video,
                }]
            }
            ViewerLayout::Grid2x2 => {
                vec![
                    ViewPane {
                        id: "pane1".to_string(),
                        x: 0.0,
                        y: 0.0,
                        width: 0.5,
                        height: 0.5,
                        content_type: ViewContentType::Video,
                    },
                    ViewPane {
                        id: "pane2".to_string(),
                        x: 0.5,
                        y: 0.0,
                        width: 0.5,
                        height: 0.5,
                        content_type: ViewContentType::Waveform,
                    },
                    ViewPane {
                        id: "pane3".to_string(),
                        x: 0.0,
                        y: 0.5,
                        width: 0.5,
                        height: 0.5,
                        content_type: ViewContentType::Vectorscope,
                    },
                    ViewPane {
                        id: "pane4".to_string(),
                        x: 0.5,
                        y: 0.5,
                        width: 0.5,
                        height: 0.5,
                        content_type: ViewContentType::AudioMeters,
                    },
                ]
            }
            _ => Vec::new(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_multiviewer() {
        let viewer = MultiViewer::new(ViewerLayout::Single);
        assert_eq!(viewer.panes().len(), 1);

        let viewer = MultiViewer::new(ViewerLayout::Grid2x2);
        assert_eq!(viewer.panes().len(), 4);
    }
}
