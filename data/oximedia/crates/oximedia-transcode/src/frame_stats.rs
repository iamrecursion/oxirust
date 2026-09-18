#![allow(dead_code)]
//! Per-frame statistics collection for transcoding analysis.

/// The type of a compressed video frame.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum FrameType {
    /// Intra-coded frame — self-contained, can be decoded independently.
    I,
    /// Predictive frame — encoded relative to a prior reference frame.
    P,
    /// Bi-directional frame — uses both past and future references.
    B,
    /// Switching P frame (H.264 SP).
    SP,
    /// Switching I frame (H.264 SI).
    SI,
}

impl FrameType {
    /// Returns `true` if this frame can be used as a reference by other frames.
    #[must_use]
    pub fn is_reference(&self) -> bool {
        matches!(self, Self::I | Self::P | Self::SP | Self::SI)
    }

    /// Returns `true` for intra-coded frame types.
    #[must_use]
    pub fn is_intra(&self) -> bool {
        matches!(self, Self::I | Self::SI)
    }

    /// Returns `true` for inter-coded frame types.
    #[must_use]
    pub fn is_inter(&self) -> bool {
        matches!(self, Self::P | Self::B | Self::SP)
    }

    /// Returns a short ASCII tag for logging.
    #[must_use]
    pub fn tag(&self) -> &'static str {
        match self {
            Self::I => "I",
            Self::P => "P",
            Self::B => "B",
            Self::SP => "SP",
            Self::SI => "SI",
        }
    }
}

/// Statistics for a single encoded frame.
#[derive(Debug, Clone)]
pub struct FrameStat {
    /// Sequential frame index (0-based).
    pub index: u64,
    /// Frame type.
    pub frame_type: FrameType,
    /// Encoded size in bytes.
    pub size_bytes: u64,
    /// Frame width in pixels.
    pub width: u32,
    /// Frame height in pixels.
    pub height: u32,
    /// Quantisation parameter used by the encoder.
    pub qp: u8,
}

impl FrameStat {
    /// Create a new frame stat entry.
    #[must_use]
    pub fn new(
        index: u64,
        frame_type: FrameType,
        size_bytes: u64,
        width: u32,
        height: u32,
        qp: u8,
    ) -> Self {
        Self {
            index,
            frame_type,
            size_bytes,
            width,
            height,
            qp,
        }
    }

    /// Bits per pixel for this frame.
    #[allow(clippy::cast_precision_loss)]
    #[must_use]
    pub fn bits_per_pixel(&self) -> f64 {
        let pixels = u64::from(self.width) * u64::from(self.height);
        if pixels == 0 {
            return 0.0;
        }
        (self.size_bytes * 8) as f64 / pixels as f64
    }

    /// Returns `true` when the frame is an I-frame (IDR or SI).
    #[must_use]
    pub fn is_keyframe(&self) -> bool {
        self.frame_type.is_intra()
    }
}

/// Collects and aggregates per-frame statistics across a transcode session.
#[derive(Debug, Default)]
pub struct FrameStatsCollector {
    frames: Vec<FrameStat>,
}

impl FrameStatsCollector {
    /// Create an empty collector.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Record a frame's statistics.
    pub fn record(&mut self, stat: FrameStat) {
        self.frames.push(stat);
    }

    /// Number of recorded frames.
    #[must_use]
    pub fn frame_count(&self) -> usize {
        self.frames.len()
    }

    /// Number of I-frames recorded.
    #[must_use]
    pub fn i_frame_count(&self) -> usize {
        self.frames
            .iter()
            .filter(|f| f.frame_type.is_intra())
            .count()
    }

    /// Number of P-frames recorded.
    #[must_use]
    pub fn p_frame_count(&self) -> usize {
        self.frames
            .iter()
            .filter(|f| matches!(f.frame_type, FrameType::P | FrameType::SP))
            .count()
    }

    /// Number of B-frames recorded.
    #[must_use]
    pub fn b_frame_count(&self) -> usize {
        self.frames
            .iter()
            .filter(|f| f.frame_type == FrameType::B)
            .count()
    }

    /// Average encoded size in bits across all recorded frames.
    #[allow(clippy::cast_precision_loss)]
    #[must_use]
    pub fn avg_bits_per_frame(&self) -> f64 {
        if self.frames.is_empty() {
            return 0.0;
        }
        let total_bytes: u64 = self.frames.iter().map(|f| f.size_bytes).sum();
        (total_bytes * 8) as f64 / self.frames.len() as f64
    }

    /// Total encoded bytes across all frames.
    #[must_use]
    pub fn total_bytes(&self) -> u64 {
        self.frames.iter().map(|f| f.size_bytes).sum()
    }

    /// Largest frame (by size) in the recording.
    #[must_use]
    pub fn largest_frame(&self) -> Option<&FrameStat> {
        self.frames.iter().max_by_key(|f| f.size_bytes)
    }

    /// Average QP across all frames.
    #[allow(clippy::cast_precision_loss)]
    #[must_use]
    pub fn avg_qp(&self) -> f64 {
        if self.frames.is_empty() {
            return 0.0;
        }
        let sum: u64 = self.frames.iter().map(|f| u64::from(f.qp)).sum();
        sum as f64 / self.frames.len() as f64
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_stat(idx: u64, ft: FrameType, size: u64) -> FrameStat {
        FrameStat::new(idx, ft, size, 1920, 1080, 28)
    }

    #[test]
    fn test_frame_type_is_reference_i() {
        assert!(FrameType::I.is_reference());
    }

    #[test]
    fn test_frame_type_is_reference_b() {
        assert!(!FrameType::B.is_reference());
    }

    #[test]
    fn test_frame_type_is_intra() {
        assert!(FrameType::I.is_intra());
        assert!(FrameType::SI.is_intra());
        assert!(!FrameType::P.is_intra());
    }

    #[test]
    fn test_frame_type_is_inter() {
        assert!(FrameType::P.is_inter());
        assert!(FrameType::B.is_inter());
        assert!(!FrameType::I.is_inter());
    }

    #[test]
    fn test_frame_type_tag() {
        assert_eq!(FrameType::I.tag(), "I");
        assert_eq!(FrameType::B.tag(), "B");
        assert_eq!(FrameType::SP.tag(), "SP");
    }

    #[test]
    fn test_frame_stat_bits_per_pixel() {
        // 1920 * 1080 = 2_073_600 pixels; 100_000 bytes = 800_000 bits
        let s = FrameStat::new(0, FrameType::I, 100_000, 1920, 1080, 20);
        let bpp = s.bits_per_pixel();
        assert!((bpp - 800_000.0 / 2_073_600.0).abs() < 1e-6);
    }

    #[test]
    fn test_frame_stat_bits_per_pixel_zero_dimension() {
        let s = FrameStat::new(0, FrameType::I, 1000, 0, 0, 20);
        assert_eq!(s.bits_per_pixel(), 0.0);
    }

    #[test]
    fn test_frame_stat_is_keyframe() {
        let i = FrameStat::new(0, FrameType::I, 0, 100, 100, 1);
        assert!(i.is_keyframe());
        let b = FrameStat::new(1, FrameType::B, 0, 100, 100, 1);
        assert!(!b.is_keyframe());
    }

    #[test]
    fn test_collector_i_frame_count() {
        let mut c = FrameStatsCollector::new();
        c.record(make_stat(0, FrameType::I, 50_000));
        c.record(make_stat(1, FrameType::P, 10_000));
        c.record(make_stat(2, FrameType::B, 5_000));
        assert_eq!(c.i_frame_count(), 1);
    }

    #[test]
    fn test_collector_b_frame_count() {
        let mut c = FrameStatsCollector::new();
        c.record(make_stat(0, FrameType::B, 5_000));
        c.record(make_stat(1, FrameType::B, 6_000));
        assert_eq!(c.b_frame_count(), 2);
    }

    #[test]
    fn test_collector_avg_bits_per_frame() {
        let mut c = FrameStatsCollector::new();
        c.record(make_stat(0, FrameType::I, 100)); // 800 bits
        c.record(make_stat(1, FrameType::P, 100)); // 800 bits
        assert!((c.avg_bits_per_frame() - 800.0).abs() < 1e-9);
    }

    #[test]
    fn test_collector_total_bytes() {
        let mut c = FrameStatsCollector::new();
        c.record(make_stat(0, FrameType::I, 1000));
        c.record(make_stat(1, FrameType::P, 2000));
        assert_eq!(c.total_bytes(), 3000);
    }

    #[test]
    fn test_collector_largest_frame() {
        let mut c = FrameStatsCollector::new();
        c.record(make_stat(0, FrameType::I, 100_000));
        c.record(make_stat(1, FrameType::P, 10_000));
        let largest = c.largest_frame().expect("should succeed in test");
        assert_eq!(largest.index, 0);
    }

    #[test]
    fn test_collector_avg_qp() {
        let mut c = FrameStatsCollector::new();
        let mut f1 = make_stat(0, FrameType::I, 0);
        f1.qp = 20;
        let mut f2 = make_stat(1, FrameType::P, 0);
        f2.qp = 30;
        c.record(f1);
        c.record(f2);
        assert!((c.avg_qp() - 25.0).abs() < 1e-9);
    }
}
