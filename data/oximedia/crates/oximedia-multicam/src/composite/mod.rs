//! Multi-view composition for multi-camera production.

pub mod grid;
pub mod pip;
pub mod split;

pub use grid::GridCompositor;
pub use pip::PictureInPicture;
pub use split::SplitScreen;

use crate::{AngleId, Result};

/// Composition layout
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Layout {
    /// Single angle (no composition)
    Single {
        /// Angle to display
        angle: AngleId,
    },

    /// Picture-in-picture
    PictureInPicture {
        /// Main angle
        main: AngleId,
        /// Inset angle
        inset: AngleId,
    },

    /// Side-by-side split
    SplitScreen {
        /// Left angle
        left: AngleId,
        /// Right angle
        right: AngleId,
    },

    /// Quad split (2x2)
    QuadSplit {
        /// Top-left angle
        top_left: AngleId,
        /// Top-right angle
        top_right: AngleId,
        /// Bottom-left angle
        bottom_left: AngleId,
        /// Bottom-right angle
        bottom_right: AngleId,
    },

    /// Grid layout
    Grid {
        /// Grid dimensions (rows, cols)
        dimensions: (usize, usize),
        /// Angle mapping
        angles: Vec<AngleId>,
    },
}

/// Compositor trait
pub trait Compositor {
    /// Set output dimensions
    fn set_dimensions(&mut self, width: u32, height: u32);

    /// Get output dimensions
    fn dimensions(&self) -> (u32, u32);

    /// Set layout
    ///
    /// # Errors
    ///
    /// Returns an error if layout is invalid
    fn set_layout(&mut self, layout: Layout) -> Result<()>;

    /// Get current layout
    fn layout(&self) -> &Layout;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_layout_single() {
        let layout = Layout::Single { angle: 0 };
        if let Layout::Single { angle } = layout {
            assert_eq!(angle, 0);
        } else {
            panic!("Wrong layout type");
        }
    }

    #[test]
    fn test_layout_pip() {
        let layout = Layout::PictureInPicture { main: 0, inset: 1 };
        if let Layout::PictureInPicture { main, inset } = layout {
            assert_eq!(main, 0);
            assert_eq!(inset, 1);
        } else {
            panic!("Wrong layout type");
        }
    }
}
