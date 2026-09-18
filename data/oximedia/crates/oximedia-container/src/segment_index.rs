#![allow(dead_code)]
//! Segment index (`sidx`) and random-access point (SAP) index for fragmented
//! ISO BMFF containers.
//!
//! Provides [`SegmentIndexEntry`], [`SegmentIndex`], and [`SapType`] for
//! building, querying, and serializing the `sidx` box used by DASH and
//! fragmented MP4.
//!
//! Re-exports [`build_sidx`] from `mux::mp4::writer` so downstream packagers
//! (e.g. `oximedia-stream::fmp4`) can emit DASH-compatible `sidx` blocks
//! without depending on internal MP4 writer paths.

pub use crate::mux::mp4::build_sidx;

use std::fmt;

/// Stream Access Point type as defined in ISO 14496-12.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum SapType {
    /// Type 1 — IDR / closed GOP start.
    Type1,
    /// Type 2 — open GOP start.
    Type2,
    /// Type 3 — gradual decoder refresh.
    Type3,
    /// No SAP (non-random-access segment).
    None,
}

impl SapType {
    /// Returns the numeric SAP type value (0 = none, 1-3 as defined).
    #[must_use]
    pub const fn as_u8(self) -> u8 {
        match self {
            Self::Type1 => 1,
            Self::Type2 => 2,
            Self::Type3 => 3,
            Self::None => 0,
        }
    }

    /// Constructs a `SapType` from its numeric code.
    #[must_use]
    pub const fn from_u8(v: u8) -> Self {
        match v {
            1 => Self::Type1,
            2 => Self::Type2,
            3 => Self::Type3,
            _ => Self::None,
        }
    }
}

impl fmt::Display for SapType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Type1 => f.write_str("SAP-1 (IDR)"),
            Self::Type2 => f.write_str("SAP-2 (Open GOP)"),
            Self::Type3 => f.write_str("SAP-3 (GDR)"),
            Self::None => f.write_str("No SAP"),
        }
    }
}

/// A single reference inside a `sidx` box.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SegmentIndexEntry {
    /// Byte size of the referenced subsegment.
    pub referenced_size: u32,
    /// Duration of the subsegment in timescale units.
    pub subsegment_duration: u32,
    /// `true` if this reference points to another `sidx` (hierarchical).
    pub reference_type: bool,
    /// Stream Access Point type for this subsegment.
    pub sap_type: SapType,
    /// Delta PTS to the first SAP in this subsegment (timescale units).
    pub sap_delta_time: u32,
}

impl SegmentIndexEntry {
    /// Creates a new entry.
    #[must_use]
    pub const fn new(referenced_size: u32, subsegment_duration: u32, sap_type: SapType) -> Self {
        Self {
            referenced_size,
            subsegment_duration,
            reference_type: false,
            sap_type,
            sap_delta_time: 0,
        }
    }

    /// Builder: marks this reference as pointing to another `sidx`.
    #[must_use]
    pub const fn with_reference_type(mut self, is_sidx: bool) -> Self {
        self.reference_type = is_sidx;
        self
    }

    /// Builder: sets the SAP delta time.
    #[must_use]
    pub const fn with_sap_delta(mut self, delta: u32) -> Self {
        self.sap_delta_time = delta;
        self
    }

    /// Returns `true` if this subsegment starts with a SAP.
    #[must_use]
    pub fn starts_with_sap(&self) -> bool {
        self.sap_type != SapType::None
    }
}

/// A complete segment index representing a `sidx` box.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SegmentIndex {
    /// Reference track ID.
    pub reference_id: u32,
    /// Timescale (ticks per second) for durations in this index.
    pub timescale: u32,
    /// Earliest presentation time of the first subsegment.
    pub earliest_presentation_time: u64,
    /// Byte offset of the first subsegment from the anchor point.
    pub first_offset: u64,
    /// The ordered list of references.
    pub entries: Vec<SegmentIndexEntry>,
}

impl SegmentIndex {
    /// Creates a new, empty segment index.
    #[must_use]
    pub fn new(reference_id: u32, timescale: u32) -> Self {
        Self {
            reference_id,
            timescale,
            earliest_presentation_time: 0,
            first_offset: 0,
            entries: Vec::new(),
        }
    }

    /// Appends an entry to the index.
    pub fn push(&mut self, entry: SegmentIndexEntry) {
        self.entries.push(entry);
    }

    /// Returns the number of entries.
    #[must_use]
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Returns `true` if the index has no entries.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Total referenced size in bytes across all entries.
    #[must_use]
    pub fn total_size(&self) -> u64 {
        self.entries
            .iter()
            .map(|e| u64::from(e.referenced_size))
            .sum()
    }

    /// Total duration in timescale units across all entries.
    #[must_use]
    pub fn total_duration_ticks(&self) -> u64 {
        self.entries
            .iter()
            .map(|e| u64::from(e.subsegment_duration))
            .sum()
    }

    /// Total duration converted to seconds.
    #[allow(clippy::cast_precision_loss)]
    #[must_use]
    pub fn total_duration_secs(&self) -> f64 {
        if self.timescale == 0 {
            return 0.0;
        }
        self.total_duration_ticks() as f64 / f64::from(self.timescale)
    }

    /// Finds the entry index containing the given presentation time (in
    /// timescale units relative to `earliest_presentation_time`).
    ///
    /// Returns `None` if the time exceeds the total duration.
    #[must_use]
    pub fn find_entry_at(&self, time_ticks: u64) -> Option<usize> {
        let mut accumulated = 0u64;
        for (i, entry) in self.entries.iter().enumerate() {
            accumulated += u64::from(entry.subsegment_duration);
            if time_ticks < accumulated {
                return Some(i);
            }
        }
        None
    }

    /// Returns the byte offset (relative to `first_offset`) for entry `index`.
    #[must_use]
    pub fn byte_offset_of(&self, index: usize) -> u64 {
        self.entries
            .iter()
            .take(index)
            .map(|e| u64::from(e.referenced_size))
            .sum()
    }

    /// Returns all indices that start with a SAP.
    #[must_use]
    pub fn sap_indices(&self) -> Vec<usize> {
        self.entries
            .iter()
            .enumerate()
            .filter(|(_, e)| e.starts_with_sap())
            .map(|(i, _)| i)
            .collect()
    }
}

// ─── Unit tests ───────────────────────────────────────────────────────────────
#[cfg(test)]
mod tests {
    use super::*;

    fn sample_entry(size: u32, dur: u32, sap: SapType) -> SegmentIndexEntry {
        SegmentIndexEntry::new(size, dur, sap)
    }

    // 1. SapType round-trip
    #[test]
    fn test_sap_type_roundtrip() {
        for v in 0..=3 {
            assert_eq!(SapType::from_u8(v).as_u8(), v);
        }
    }

    // 2. SapType display
    #[test]
    fn test_sap_type_display() {
        assert_eq!(format!("{}", SapType::Type1), "SAP-1 (IDR)");
        assert_eq!(format!("{}", SapType::None), "No SAP");
    }

    // 3. entry starts_with_sap
    #[test]
    fn test_starts_with_sap() {
        assert!(sample_entry(100, 48000, SapType::Type1).starts_with_sap());
        assert!(!sample_entry(100, 48000, SapType::None).starts_with_sap());
    }

    // 4. entry builder methods
    #[test]
    fn test_entry_builders() {
        let e = SegmentIndexEntry::new(200, 90000, SapType::Type2)
            .with_reference_type(true)
            .with_sap_delta(1000);
        assert!(e.reference_type);
        assert_eq!(e.sap_delta_time, 1000);
    }

    // 5. new index is empty
    #[test]
    fn test_new_index_empty() {
        let idx = SegmentIndex::new(1, 90000);
        assert!(idx.is_empty());
        assert_eq!(idx.len(), 0);
    }

    // 6. push increases len
    #[test]
    fn test_push() {
        let mut idx = SegmentIndex::new(1, 90000);
        idx.push(sample_entry(1000, 90000, SapType::Type1));
        assert_eq!(idx.len(), 1);
    }

    // 7. total_size
    #[test]
    fn test_total_size() {
        let mut idx = SegmentIndex::new(1, 90000);
        idx.push(sample_entry(1000, 90000, SapType::Type1));
        idx.push(sample_entry(2000, 90000, SapType::None));
        assert_eq!(idx.total_size(), 3000);
    }

    // 8. total_duration_ticks
    #[test]
    fn test_total_duration_ticks() {
        let mut idx = SegmentIndex::new(1, 90000);
        idx.push(sample_entry(1000, 90000, SapType::Type1));
        idx.push(sample_entry(2000, 45000, SapType::None));
        assert_eq!(idx.total_duration_ticks(), 135000);
    }

    // 9. total_duration_secs
    #[test]
    fn test_total_duration_secs() {
        let mut idx = SegmentIndex::new(1, 90000);
        idx.push(sample_entry(1000, 90000, SapType::Type1));
        let dur = idx.total_duration_secs();
        assert!((dur - 1.0).abs() < 1e-9);
    }

    // 10. total_duration_secs zero timescale
    #[test]
    fn test_total_duration_secs_zero_timescale() {
        let mut idx = SegmentIndex::new(1, 0);
        idx.push(sample_entry(100, 500, SapType::None));
        assert_eq!(idx.total_duration_secs(), 0.0);
    }

    // 11. find_entry_at within range
    #[test]
    fn test_find_entry_at() {
        let mut idx = SegmentIndex::new(1, 90000);
        idx.push(sample_entry(1000, 90000, SapType::Type1));
        idx.push(sample_entry(2000, 90000, SapType::None));
        assert_eq!(idx.find_entry_at(0), Some(0));
        assert_eq!(idx.find_entry_at(89999), Some(0));
        assert_eq!(idx.find_entry_at(90000), Some(1));
    }

    // 12. find_entry_at out of range
    #[test]
    fn test_find_entry_at_out_of_range() {
        let mut idx = SegmentIndex::new(1, 90000);
        idx.push(sample_entry(1000, 90000, SapType::Type1));
        assert!(idx.find_entry_at(90000).is_none());
    }

    // 13. byte_offset_of
    #[test]
    fn test_byte_offset_of() {
        let mut idx = SegmentIndex::new(1, 90000);
        idx.push(sample_entry(1000, 90000, SapType::Type1));
        idx.push(sample_entry(2000, 90000, SapType::None));
        idx.push(sample_entry(3000, 90000, SapType::Type1));
        assert_eq!(idx.byte_offset_of(0), 0);
        assert_eq!(idx.byte_offset_of(1), 1000);
        assert_eq!(idx.byte_offset_of(2), 3000);
    }

    // 14. sap_indices
    #[test]
    fn test_sap_indices() {
        let mut idx = SegmentIndex::new(1, 90000);
        idx.push(sample_entry(1000, 90000, SapType::Type1));
        idx.push(sample_entry(2000, 90000, SapType::None));
        idx.push(sample_entry(3000, 90000, SapType::Type2));
        assert_eq!(idx.sap_indices(), vec![0, 2]);
    }

    // 15. reference_id and timescale stored
    #[test]
    fn test_index_fields() {
        let idx = SegmentIndex::new(42, 48000);
        assert_eq!(idx.reference_id, 42);
        assert_eq!(idx.timescale, 48000);
    }

    // 16. SapType unknown maps to None
    #[test]
    fn test_sap_type_unknown() {
        assert_eq!(SapType::from_u8(255), SapType::None);
    }
}
