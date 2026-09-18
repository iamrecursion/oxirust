//! AV1 frame-level processing.
//!
//! This module handles frame-level operations including frame header parsing,
//! reference frame management, and frame buffer allocation.

#![allow(dead_code)]

/// AV1 frame types as defined in the specification.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum Av1FrameType {
    /// Keyframe - independently decodable.
    #[default]
    Key = 0,
    /// Inter frame - references previous frames.
    Inter = 1,
    /// Intra-only frame - not a random access point.
    IntraOnly = 2,
    /// Switch frame - for stream switching.
    Switch = 3,
}

impl From<u8> for Av1FrameType {
    fn from(value: u8) -> Self {
        match value {
            0 | 4.. => Self::Key,
            1 => Self::Inter,
            2 => Self::IntraOnly,
            3 => Self::Switch,
        }
    }
}

/// Reference frame indices.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ReferenceFrame {
    /// No reference.
    None = 0,
    /// Intra prediction.
    Intra = 1,
    /// Last frame reference.
    Last = 2,
    /// Last2 frame reference.
    Last2 = 3,
    /// Last3 frame reference.
    Last3 = 4,
    /// Golden frame reference.
    Golden = 5,
    /// Backward reference.
    BwdRef = 6,
    /// Alternate reference 2.
    AltRef2 = 7,
    /// Alternate reference.
    AltRef = 8,
}

/// Frame header information.
#[derive(Clone, Debug, Default)]
pub struct FrameHeader {
    /// Frame type.
    pub frame_type: Av1FrameType,
    /// Show frame flag.
    pub show_frame: bool,
    /// Error resilient mode.
    pub error_resilient_mode: bool,
    /// Frame width.
    pub frame_width: u32,
    /// Frame height.
    pub frame_height: u32,
    /// Refresh frame flags.
    pub refresh_frame_flags: u8,
}

/// Reference frame buffer.
#[derive(Clone, Debug, Default)]
pub struct RefFrameBuffer {
    /// Buffer slots (up to 8).
    slots: [Option<RefFrameSlot>; 8],
}

/// Single reference frame slot.
#[derive(Clone, Debug)]
pub struct RefFrameSlot {
    /// Frame width.
    pub width: u32,
    /// Frame height.
    pub height: u32,
    /// Order hint.
    pub order_hint: u8,
}

impl RefFrameBuffer {
    /// Create a new reference frame buffer.
    #[must_use]
    pub const fn new() -> Self {
        Self {
            slots: [const { None }; 8],
        }
    }

    /// Update reference frames based on refresh flags.
    pub fn update(&mut self, refresh_flags: u8, slot: &RefFrameSlot) {
        for i in 0..8 {
            if refresh_flags & (1 << i) != 0 {
                self.slots[i] = Some(slot.clone());
            }
        }
    }

    /// Get a reference frame slot.
    #[must_use]
    pub fn get(&self, index: usize) -> Option<&RefFrameSlot> {
        self.slots.get(index).and_then(|s| s.as_ref())
    }

    /// Clear all reference frames.
    pub fn clear(&mut self) {
        self.slots = [const { None }; 8];
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_frame_type_from_u8() {
        assert_eq!(Av1FrameType::from(0), Av1FrameType::Key);
        assert_eq!(Av1FrameType::from(1), Av1FrameType::Inter);
    }

    #[test]
    fn test_ref_frame_buffer() {
        let mut buffer = RefFrameBuffer::new();
        assert!(buffer.get(0).is_none());

        let slot = RefFrameSlot {
            width: 1920,
            height: 1080,
            order_hint: 0,
        };
        buffer.update(0b0000_0101, &slot);

        assert!(buffer.get(0).is_some());
        assert!(buffer.get(1).is_none());
        assert!(buffer.get(2).is_some());
    }
}
