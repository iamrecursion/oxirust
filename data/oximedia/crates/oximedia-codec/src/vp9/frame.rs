//! VP9 frame types and structures.

#![allow(dead_code)]

/// VP9 frame type enumeration.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum FrameType {
    /// Keyframe (intra-only, random access point).
    #[default]
    Key,
    /// Inter frame (requires reference frames).
    Inter,
}

impl FrameType {
    /// Returns true if this is a keyframe.
    #[must_use]
    pub const fn is_keyframe(&self) -> bool {
        matches!(self, Self::Key)
    }
}

/// VP9 decoded frame placeholder.
#[derive(Clone, Debug, Default)]
pub struct Vp9Frame {
    /// Frame type.
    pub frame_type: FrameType,
    /// Frame width in pixels.
    pub width: u32,
    /// Frame height in pixels.
    pub height: u32,
}

impl Vp9Frame {
    /// Creates a new VP9 frame.
    #[must_use]
    pub const fn new(frame_type: FrameType, width: u32, height: u32) -> Self {
        Self {
            frame_type,
            width,
            height,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_frame_type() {
        assert!(FrameType::Key.is_keyframe());
        assert!(!FrameType::Inter.is_keyframe());
    }

    #[test]
    fn test_vp9_frame() {
        let frame = Vp9Frame::new(FrameType::Key, 1920, 1080);
        assert!(frame.frame_type.is_keyframe());
        assert_eq!(frame.width, 1920);
        assert_eq!(frame.height, 1080);
    }
}
