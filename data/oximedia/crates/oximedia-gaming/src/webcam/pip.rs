//! Picture-in-picture composition.

/// Picture-in-picture compositor.
pub struct PictureInPicture {
    position: PipPosition,
    scale: f32,
}

/// `PiP` position on screen.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PipPosition {
    /// Top left corner
    TopLeft,
    /// Top right corner
    TopRight,
    /// Bottom left corner
    BottomLeft,
    /// Bottom right corner
    BottomRight,
    /// Custom position (x, y)
    Custom(i32, i32),
}

impl PictureInPicture {
    /// Create a new `PiP` compositor.
    #[must_use]
    pub fn new(position: PipPosition, scale: f32) -> Self {
        Self { position, scale }
    }

    /// Set position.
    pub fn set_position(&mut self, position: PipPosition) {
        self.position = position;
    }

    /// Set scale.
    pub fn set_scale(&mut self, scale: f32) {
        self.scale = scale;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pip_creation() {
        let pip = PictureInPicture::new(PipPosition::BottomRight, 0.25);
        assert_eq!(pip.position, PipPosition::BottomRight);
    }
}
