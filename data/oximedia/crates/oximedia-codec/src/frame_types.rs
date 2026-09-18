//! Frame type management for video encoding.
//!
//! Covers I/P/B frame type decisions, GOP (Group Of Pictures) structure,
//! reference frame lists, and frame ordering utilities.

#![allow(dead_code)]

/// The coding type of a video frame.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum CodingFrameType {
    /// Intra-coded frame – no reference to other frames; also called a keyframe.
    I,
    /// Predictively coded frame – references one past frame.
    P,
    /// Bi-directionally coded frame – references past and future frames.
    B,
    /// An IDR (Instantaneous Decoder Refresh) I-frame that clears the DPB.
    Idr,
}

impl CodingFrameType {
    /// Returns `true` if this frame can be used as a reference by subsequent frames.
    #[must_use]
    pub fn is_reference(self) -> bool {
        !matches!(self, Self::B)
    }

    /// Returns `true` if this frame is intra-coded (no inter dependencies).
    #[must_use]
    pub fn is_intra(self) -> bool {
        matches!(self, Self::I | Self::Idr)
    }

    /// Returns a short ASCII label for this frame type.
    #[must_use]
    pub fn label(self) -> &'static str {
        match self {
            Self::I => "I",
            Self::P => "P",
            Self::B => "B",
            Self::Idr => "IDR",
        }
    }
}

/// A display-order frame descriptor with its assigned coding type.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FrameDescriptor {
    /// Zero-based display-order index.
    pub display_index: u64,
    /// Assigned coding type for this frame.
    pub frame_type: CodingFrameType,
    /// Decode-order index (may differ from `display_index` for B-frames).
    pub decode_index: u64,
    /// Quantizer parameter override, if any (0 = use default).
    pub qp_override: Option<u8>,
}

impl FrameDescriptor {
    /// Creates a new frame descriptor with matching display and decode indices.
    #[must_use]
    pub fn new(index: u64, frame_type: CodingFrameType) -> Self {
        Self {
            display_index: index,
            frame_type,
            decode_index: index,
            qp_override: None,
        }
    }

    /// Sets an explicit decode-order index.
    #[must_use]
    pub fn with_decode_index(mut self, decode_index: u64) -> Self {
        self.decode_index = decode_index;
        self
    }

    /// Attaches a QP override.
    #[must_use]
    pub fn with_qp(mut self, qp: u8) -> Self {
        self.qp_override = Some(qp);
        self
    }
}

/// Configuration for a Group Of Pictures (GOP).
#[derive(Debug, Clone)]
pub struct GopConfig {
    /// Maximum GOP length (number of frames between I-frames).
    pub max_gop_size: u32,
    /// Number of consecutive B-frames between each pair of reference frames.
    pub b_frames: u32,
    /// Whether closed GOPs are used (each GOP is independently decodable).
    pub closed_gop: bool,
    /// Whether adaptive scene-change detection may insert extra I-frames.
    pub adaptive_keyframes: bool,
}

impl GopConfig {
    /// Creates a config for a simple all-I-frame stream.
    #[must_use]
    pub fn intra_only() -> Self {
        Self {
            max_gop_size: 1,
            b_frames: 0,
            closed_gop: true,
            adaptive_keyframes: false,
        }
    }

    /// Creates a standard IP-only GOP (no B-frames).
    #[must_use]
    pub fn ip_only(gop_size: u32) -> Self {
        Self {
            max_gop_size: gop_size,
            b_frames: 0,
            closed_gop: true,
            adaptive_keyframes: true,
        }
    }

    /// Creates a standard IBP GOP with the given B-frame count.
    #[must_use]
    pub fn with_b_frames(gop_size: u32, b_frames: u32) -> Self {
        Self {
            max_gop_size: gop_size,
            b_frames,
            closed_gop: false,
            adaptive_keyframes: true,
        }
    }
}

/// Generates a sequence of [`FrameDescriptor`] entries for `num_frames` frames
/// given a [`GopConfig`].
///
/// This produces a simplified IBBBP…P pattern: one IDR at position 0, then
/// I-frames at every `max_gop_size` boundary, B-frames filling the gaps, and
/// P-frames at sub-GOP boundaries.
#[must_use]
pub fn generate_gop_sequence(config: &GopConfig, num_frames: u64) -> Vec<FrameDescriptor> {
    let mut descriptors = Vec::with_capacity(num_frames as usize);
    let gop = config.max_gop_size as u64;
    let b = config.b_frames as u64;

    for i in 0..num_frames {
        let frame_type = if i == 0 {
            CodingFrameType::Idr
        } else if gop <= 1 {
            // Intra-only: every frame after the first IDR is I
            CodingFrameType::I
        } else if i % gop == 0 {
            CodingFrameType::I
        } else if b > 0 {
            // Frames just before a P-frame anchor.
            let pos_in_gop = i % gop;
            let sub_period = b + 1;
            if pos_in_gop % sub_period == 0 {
                CodingFrameType::P
            } else {
                CodingFrameType::B
            }
        } else {
            CodingFrameType::P
        };
        descriptors.push(FrameDescriptor::new(i, frame_type));
    }
    descriptors
}

/// A pool of decoded reference frames available for inter-prediction.
#[derive(Debug, Default)]
pub struct ReferenceFramePool {
    frames: Vec<u64>, // display indices of currently held reference frames
    /// Maximum number of references to retain simultaneously.
    pub capacity: usize,
}

impl ReferenceFramePool {
    /// Creates a new pool with the given capacity.
    #[must_use]
    pub fn new(capacity: usize) -> Self {
        Self {
            frames: Vec::with_capacity(capacity),
            capacity,
        }
    }

    /// Inserts a reference frame by its display index.
    /// If the pool is at capacity, the oldest entry is evicted (FIFO).
    pub fn insert(&mut self, display_index: u64) {
        if self.frames.len() == self.capacity {
            self.frames.remove(0);
        }
        self.frames.push(display_index);
    }

    /// Returns `true` if `display_index` is currently in the pool.
    #[must_use]
    pub fn contains(&self, display_index: u64) -> bool {
        self.frames.contains(&display_index)
    }

    /// Returns the number of frames currently in the pool.
    #[must_use]
    pub fn len(&self) -> usize {
        self.frames.len()
    }

    /// Returns `true` if the pool is empty.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.frames.is_empty()
    }

    /// Clears all reference frames (used on IDR boundaries).
    pub fn clear(&mut self) {
        self.frames.clear();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_i_frame_is_intra() {
        assert!(CodingFrameType::I.is_intra());
        assert!(CodingFrameType::Idr.is_intra());
    }

    #[test]
    fn test_p_frame_not_intra() {
        assert!(!CodingFrameType::P.is_intra());
    }

    #[test]
    fn test_b_frame_not_reference() {
        assert!(!CodingFrameType::B.is_reference());
    }

    #[test]
    fn test_i_and_p_are_references() {
        assert!(CodingFrameType::I.is_reference());
        assert!(CodingFrameType::P.is_reference());
        assert!(CodingFrameType::Idr.is_reference());
    }

    #[test]
    fn test_frame_type_labels() {
        assert_eq!(CodingFrameType::I.label(), "I");
        assert_eq!(CodingFrameType::P.label(), "P");
        assert_eq!(CodingFrameType::B.label(), "B");
        assert_eq!(CodingFrameType::Idr.label(), "IDR");
    }

    #[test]
    fn test_frame_descriptor_defaults() {
        let fd = FrameDescriptor::new(5, CodingFrameType::P);
        assert_eq!(fd.display_index, 5);
        assert_eq!(fd.decode_index, 5);
        assert_eq!(fd.qp_override, None);
    }

    #[test]
    fn test_frame_descriptor_with_qp() {
        let fd = FrameDescriptor::new(0, CodingFrameType::I).with_qp(22);
        assert_eq!(fd.qp_override, Some(22));
    }

    #[test]
    fn test_gop_sequence_starts_with_idr() {
        let cfg = GopConfig::ip_only(30);
        let seq = generate_gop_sequence(&cfg, 10);
        assert_eq!(seq[0].frame_type, CodingFrameType::Idr);
    }

    #[test]
    fn test_intra_only_all_idr_or_i() {
        let cfg = GopConfig::intra_only();
        let seq = generate_gop_sequence(&cfg, 5);
        for (i, fd) in seq.iter().enumerate() {
            if i == 0 {
                assert_eq!(fd.frame_type, CodingFrameType::Idr);
            } else {
                assert!(fd.frame_type.is_intra());
            }
        }
    }

    #[test]
    fn test_ip_sequence_no_b_frames() {
        let cfg = GopConfig::ip_only(8);
        let seq = generate_gop_sequence(&cfg, 16);
        for fd in &seq {
            assert!(!matches!(fd.frame_type, CodingFrameType::B));
        }
    }

    #[test]
    fn test_reference_pool_capacity_eviction() {
        let mut pool = ReferenceFramePool::new(3);
        pool.insert(0);
        pool.insert(1);
        pool.insert(2);
        pool.insert(3); // should evict 0
        assert!(!pool.contains(0));
        assert!(pool.contains(3));
    }

    #[test]
    fn test_reference_pool_len() {
        let mut pool = ReferenceFramePool::new(4);
        assert!(pool.is_empty());
        pool.insert(10);
        pool.insert(11);
        assert_eq!(pool.len(), 2);
    }

    #[test]
    fn test_reference_pool_clear_on_idr() {
        let mut pool = ReferenceFramePool::new(4);
        pool.insert(0);
        pool.insert(1);
        pool.clear();
        assert!(pool.is_empty());
    }
}
