//! Dolby Vision display management metadata
//!
//! Per-frame and per-scene display management (DM) metadata used for
//! color volume transform and content-adaptive tone mapping. Covers
//! the DM data payload structures specified in the Dolby Vision RPU format.

#![allow(dead_code)]
#![allow(clippy::cast_precision_loss)]

/// Signal range for DM metadata values.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum SignalRange {
    /// Narrow (limited/broadcast) range: 16–235 for 8-bit.
    Narrow,
    /// Full (data) range: 0–255 for 8-bit.
    Full,
}

impl SignalRange {
    /// Black level for 10-bit representation.
    #[must_use]
    pub const fn black_10bit(&self) -> u16 {
        match self {
            Self::Narrow => 64,
            Self::Full => 0,
        }
    }

    /// White level for 10-bit representation.
    #[must_use]
    pub const fn white_10bit(&self) -> u16 {
        match self {
            Self::Narrow => 940,
            Self::Full => 1023,
        }
    }

    /// Usable code value range for 10-bit.
    #[must_use]
    pub const fn range_10bit(&self) -> u16 {
        self.white_10bit() - self.black_10bit()
    }
}

/// Color primaries identification for the source content.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum DmColorPrimaries {
    /// BT.709 (sRGB / Rec.709).
    Bt709,
    /// BT.2020.
    Bt2020,
    /// DCI-P3.
    DciP3,
    /// Display P3 (P3-D65).
    DisplayP3,
}

impl DmColorPrimaries {
    /// CIE 1931 xy coordinates for the red primary.
    #[must_use]
    pub const fn red_xy(&self) -> (f32, f32) {
        match self {
            Self::Bt709 => (0.640, 0.330),
            Self::Bt2020 => (0.708, 0.292),
            Self::DciP3 => (0.680, 0.320),
            Self::DisplayP3 => (0.680, 0.320),
        }
    }

    /// CIE 1931 xy coordinates for the green primary.
    #[must_use]
    pub const fn green_xy(&self) -> (f32, f32) {
        match self {
            Self::Bt709 => (0.300, 0.600),
            Self::Bt2020 => (0.170, 0.797),
            Self::DciP3 => (0.265, 0.690),
            Self::DisplayP3 => (0.265, 0.690),
        }
    }

    /// CIE 1931 xy coordinates for the blue primary.
    #[must_use]
    pub const fn blue_xy(&self) -> (f32, f32) {
        match self {
            Self::Bt709 => (0.150, 0.060),
            Self::Bt2020 => (0.131, 0.046),
            Self::DciP3 => (0.150, 0.060),
            Self::DisplayP3 => (0.150, 0.060),
        }
    }

    /// CIE 1931 xy coordinates for the white point.
    #[must_use]
    pub const fn white_point_xy(&self) -> (f32, f32) {
        match self {
            Self::Bt709 | Self::Bt2020 | Self::DisplayP3 => (0.3127, 0.3290),
            Self::DciP3 => (0.314, 0.351),
        }
    }
}

/// Source content mastering display parameters embedded in DM metadata.
#[derive(Debug, Clone, PartialEq)]
pub struct DmSourceDisplay {
    /// Minimum display mastering luminance in nits.
    pub min_luminance: f32,
    /// Maximum display mastering luminance in nits.
    pub max_luminance: f32,
    /// Color primaries of the source mastering display.
    pub primaries: DmColorPrimaries,
    /// Signal range.
    pub signal_range: SignalRange,
}

impl DmSourceDisplay {
    /// Create a new source display descriptor.
    #[must_use]
    pub fn new(
        min_luminance: f32,
        max_luminance: f32,
        primaries: DmColorPrimaries,
        signal_range: SignalRange,
    ) -> Self {
        Self {
            min_luminance,
            max_luminance,
            primaries,
            signal_range,
        }
    }

    /// Typical HDR mastering display (0.005–1000 nits, BT.2020, narrow).
    #[must_use]
    pub fn hdr_1000() -> Self {
        Self::new(0.005, 1000.0, DmColorPrimaries::Bt2020, SignalRange::Narrow)
    }

    /// Typical cinema mastering display (0.005–4000 nits, P3, narrow).
    #[must_use]
    pub fn cinema_4000() -> Self {
        Self::new(0.005, 4000.0, DmColorPrimaries::DciP3, SignalRange::Narrow)
    }

    /// Dynamic range in stops (log2(max/min)).
    #[must_use]
    pub fn dynamic_range_stops(&self) -> f32 {
        if self.min_luminance <= 0.0 || self.max_luminance <= 0.0 {
            return 0.0;
        }
        (self.max_luminance / self.min_luminance).log2()
    }

    /// Validate that luminance values are sensible.
    #[must_use]
    pub fn validate(&self) -> Vec<String> {
        let mut errors = Vec::new();
        if self.min_luminance < 0.0 {
            errors.push(format!(
                "min_luminance must be >= 0, got {}",
                self.min_luminance
            ));
        }
        if self.max_luminance <= self.min_luminance {
            errors.push(format!(
                "max_luminance ({}) must be > min_luminance ({})",
                self.max_luminance, self.min_luminance
            ));
        }
        if self.max_luminance > 10000.0 {
            errors.push(format!(
                "max_luminance ({}) exceeds PQ limit of 10000 nits",
                self.max_luminance
            ));
        }
        errors
    }
}

/// Per-frame DM metadata containing color volume transform parameters.
#[derive(Debug, Clone, PartialEq)]
pub struct FrameDmMetadata {
    /// Frame index (0-based).
    pub frame_index: u64,
    /// Scene-level min PQ code value (12-bit, 0–4095).
    pub min_pq: u16,
    /// Scene-level max PQ code value (12-bit, 0–4095).
    pub max_pq: u16,
    /// Mid-tone PQ anchor value (12-bit).
    pub mid_pq: u16,
    /// Source display parameters.
    pub source_display: DmSourceDisplay,
}

impl FrameDmMetadata {
    /// Create new frame DM metadata.
    #[must_use]
    pub fn new(
        frame_index: u64,
        min_pq: u16,
        max_pq: u16,
        source_display: DmSourceDisplay,
    ) -> Self {
        let mid_pq = (min_pq + max_pq) / 2;
        Self {
            frame_index,
            min_pq,
            max_pq,
            mid_pq,
            source_display,
        }
    }

    /// PQ code-value range for this frame.
    #[must_use]
    pub fn pq_range(&self) -> u16 {
        self.max_pq.saturating_sub(self.min_pq)
    }

    /// Normalised min PQ (0.0–1.0 over 12-bit range).
    #[must_use]
    pub fn min_pq_normalized(&self) -> f32 {
        self.min_pq as f32 / 4095.0
    }

    /// Normalised max PQ (0.0–1.0 over 12-bit range).
    #[must_use]
    pub fn max_pq_normalized(&self) -> f32 {
        self.max_pq as f32 / 4095.0
    }

    /// Validate PQ code values.
    #[must_use]
    pub fn validate(&self) -> Vec<String> {
        let mut errors = Vec::new();
        if self.min_pq > 4095 {
            errors.push(format!("min_pq ({}) exceeds 12-bit range", self.min_pq));
        }
        if self.max_pq > 4095 {
            errors.push(format!("max_pq ({}) exceeds 12-bit range", self.max_pq));
        }
        if self.max_pq < self.min_pq {
            errors.push(format!(
                "max_pq ({}) < min_pq ({})",
                self.max_pq, self.min_pq
            ));
        }
        errors
    }
}

/// A sequence of per-frame DM metadata entries.
#[derive(Debug, Default)]
pub struct DmMetadataSequence {
    /// All frame metadata entries, sorted by frame index.
    frames: Vec<FrameDmMetadata>,
}

impl DmMetadataSequence {
    /// Create an empty sequence.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Add a frame's DM metadata.
    pub fn add(&mut self, frame: FrameDmMetadata) {
        self.frames.push(frame);
    }

    /// Number of frames in the sequence.
    #[must_use]
    pub fn len(&self) -> usize {
        self.frames.len()
    }

    /// Whether the sequence is empty.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.frames.is_empty()
    }

    /// Get metadata for a specific frame index.
    #[must_use]
    pub fn get_frame(&self, frame_index: u64) -> Option<&FrameDmMetadata> {
        self.frames.iter().find(|f| f.frame_index == frame_index)
    }

    /// Average max PQ across all frames (useful for content analysis).
    #[must_use]
    pub fn average_max_pq(&self) -> f32 {
        if self.frames.is_empty() {
            return 0.0;
        }
        let sum: f32 = self.frames.iter().map(|f| f.max_pq as f32).sum();
        sum / self.frames.len() as f32
    }

    /// Peak max PQ across all frames.
    #[must_use]
    pub fn peak_max_pq(&self) -> u16 {
        self.frames.iter().map(|f| f.max_pq).max().unwrap_or(0)
    }

    /// Validate all frames in the sequence.
    #[must_use]
    pub fn validate_all(&self) -> Vec<String> {
        let mut errors = Vec::new();
        for frame in &self.frames {
            for err in frame.validate() {
                errors.push(format!("Frame {}: {}", frame.frame_index, err));
            }
        }
        errors
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_signal_range_narrow() {
        let r = SignalRange::Narrow;
        assert_eq!(r.black_10bit(), 64);
        assert_eq!(r.white_10bit(), 940);
        assert_eq!(r.range_10bit(), 876);
    }

    #[test]
    fn test_signal_range_full() {
        let r = SignalRange::Full;
        assert_eq!(r.black_10bit(), 0);
        assert_eq!(r.white_10bit(), 1023);
        assert_eq!(r.range_10bit(), 1023);
    }

    #[test]
    fn test_dm_color_primaries_bt709() {
        let p = DmColorPrimaries::Bt709;
        let (rx, ry) = p.red_xy();
        assert!((rx - 0.640).abs() < 0.001);
        assert!((ry - 0.330).abs() < 0.001);
    }

    #[test]
    fn test_dm_color_primaries_bt2020() {
        let p = DmColorPrimaries::Bt2020;
        let (rx, _ry) = p.red_xy();
        assert!((rx - 0.708).abs() < 0.001);
    }

    #[test]
    fn test_dm_source_display_hdr_1000() {
        let d = DmSourceDisplay::hdr_1000();
        assert!((d.min_luminance - 0.005).abs() < 0.001);
        assert!((d.max_luminance - 1000.0).abs() < 0.1);
        assert_eq!(d.primaries, DmColorPrimaries::Bt2020);
    }

    #[test]
    fn test_dm_source_display_dynamic_range() {
        let d = DmSourceDisplay::hdr_1000();
        let stops = d.dynamic_range_stops();
        // log2(1000/0.005) ~ 17.6
        assert!(stops > 17.0 && stops < 18.0, "stops={stops}");
    }

    #[test]
    fn test_dm_source_display_validate_ok() {
        let d = DmSourceDisplay::hdr_1000();
        assert!(d.validate().is_empty());
    }

    #[test]
    fn test_dm_source_display_validate_bad_luminance() {
        let d = DmSourceDisplay::new(-1.0, 1000.0, DmColorPrimaries::Bt2020, SignalRange::Narrow);
        let errs = d.validate();
        assert!(!errs.is_empty());
        assert!(errs[0].contains("min_luminance"));
    }

    #[test]
    fn test_dm_source_display_validate_max_exceeds_pq() {
        let d = DmSourceDisplay::new(
            0.005,
            15000.0,
            DmColorPrimaries::Bt2020,
            SignalRange::Narrow,
        );
        let errs = d.validate();
        assert!(errs.iter().any(|e| e.contains("10000")));
    }

    #[test]
    fn test_frame_dm_metadata_creation() {
        let src = DmSourceDisplay::hdr_1000();
        let f = FrameDmMetadata::new(0, 100, 3000, src);
        assert_eq!(f.frame_index, 0);
        assert_eq!(f.min_pq, 100);
        assert_eq!(f.max_pq, 3000);
        assert_eq!(f.mid_pq, 1550);
    }

    #[test]
    fn test_frame_dm_metadata_pq_range() {
        let src = DmSourceDisplay::hdr_1000();
        let f = FrameDmMetadata::new(0, 200, 3500, src);
        assert_eq!(f.pq_range(), 3300);
    }

    #[test]
    fn test_frame_dm_metadata_normalized() {
        let src = DmSourceDisplay::hdr_1000();
        let f = FrameDmMetadata::new(0, 0, 4095, src);
        assert!((f.min_pq_normalized() - 0.0).abs() < 0.001);
        assert!((f.max_pq_normalized() - 1.0).abs() < 0.001);
    }

    #[test]
    fn test_frame_dm_metadata_validate_ok() {
        let src = DmSourceDisplay::hdr_1000();
        let f = FrameDmMetadata::new(0, 100, 3000, src);
        assert!(f.validate().is_empty());
    }

    #[test]
    fn test_frame_dm_metadata_validate_inverted_pq() {
        let src = DmSourceDisplay::hdr_1000();
        let mut f = FrameDmMetadata::new(0, 3000, 100, src);
        f.min_pq = 3000;
        f.max_pq = 100;
        let errs = f.validate();
        assert!(!errs.is_empty());
    }

    #[test]
    fn test_dm_metadata_sequence_basic() {
        let mut seq = DmMetadataSequence::new();
        assert!(seq.is_empty());

        let src = DmSourceDisplay::hdr_1000();
        seq.add(FrameDmMetadata::new(0, 100, 3000, src.clone()));
        seq.add(FrameDmMetadata::new(1, 110, 3100, src));

        assert_eq!(seq.len(), 2);
        assert!(seq.get_frame(0).is_some());
        assert!(seq.get_frame(99).is_none());
    }

    #[test]
    fn test_dm_metadata_sequence_average_max_pq() {
        let mut seq = DmMetadataSequence::new();
        let src = DmSourceDisplay::hdr_1000();
        seq.add(FrameDmMetadata::new(0, 0, 2000, src.clone()));
        seq.add(FrameDmMetadata::new(1, 0, 4000, src));
        assert!((seq.average_max_pq() - 3000.0).abs() < 0.1);
    }

    #[test]
    fn test_dm_metadata_sequence_peak_max_pq() {
        let mut seq = DmMetadataSequence::new();
        let src = DmSourceDisplay::hdr_1000();
        seq.add(FrameDmMetadata::new(0, 0, 2000, src.clone()));
        seq.add(FrameDmMetadata::new(1, 0, 3500, src));
        assert_eq!(seq.peak_max_pq(), 3500);
    }

    #[test]
    fn test_dm_metadata_sequence_validate_all() {
        let mut seq = DmMetadataSequence::new();
        let src = DmSourceDisplay::hdr_1000();
        seq.add(FrameDmMetadata::new(0, 100, 3000, src.clone()));
        assert!(seq.validate_all().is_empty());

        // Add invalid frame
        let mut bad = FrameDmMetadata::new(1, 0, 0, src);
        bad.min_pq = 5000; // exceeds 12-bit
        seq.add(bad);
        let errs = seq.validate_all();
        assert!(!errs.is_empty());
    }

    #[test]
    fn test_dm_color_primaries_white_point() {
        let wp = DmColorPrimaries::Bt709.white_point_xy();
        assert!((wp.0 - 0.3127).abs() < 0.001);
        assert!((wp.1 - 0.3290).abs() < 0.001);
    }
}
