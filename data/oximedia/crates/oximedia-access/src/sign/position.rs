//! Sign language video positioning.

use serde::{Deserialize, Serialize};

/// Position for sign language video (picture-in-picture).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum SignPosition {
    /// Top left corner.
    TopLeft,
    /// Top right corner.
    TopRight,
    /// Bottom left corner.
    BottomLeft,
    /// Bottom right corner (default).
    BottomRight,
    /// Custom position (x%, y%).
    Custom(u8, u8),
}

/// Size preset for sign language video.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum SignSize {
    /// Small (15% of screen).
    Small,
    /// Medium (25% of screen).
    Medium,
    /// Large (35% of screen).
    Large,
    /// Custom size (percentage of screen width).
    Custom(u8),
}

impl SignSize {
    /// Get size as percentage of screen width.
    #[must_use]
    pub const fn as_percent(&self) -> u8 {
        match self {
            Self::Small => 15,
            Self::Medium => 25,
            Self::Large => 35,
            Self::Custom(p) => *p,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sign_size() {
        assert_eq!(SignSize::Small.as_percent(), 15);
        assert_eq!(SignSize::Medium.as_percent(), 25);
        assert_eq!(SignSize::Custom(40).as_percent(), 40);
    }
}
