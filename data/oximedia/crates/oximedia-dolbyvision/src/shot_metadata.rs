//! Dolby Vision shot-level metadata management.
//!
//! Each "shot" in a DV stream can carry per-shot trim data. This module
//! provides types for representing, validating and querying that data.

#![allow(dead_code)]

/// Categorises the kind of trim applied at the shot level.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ShotTrimType {
    /// No trim – passes source values unchanged.
    Identity,
    /// Linear lift/gain applied to all pixels.
    Linear,
    /// Nonlinear (S-curve) tone mapping.
    Nonlinear,
    /// Saturation-adjusted trim.
    Saturation,
    /// Combined tone and saturation adjustment.
    Combined,
}

impl ShotTrimType {
    /// Returns `true` when this trim type modifies playback appearance.
    #[must_use]
    pub fn affects_playback(self) -> bool {
        !matches!(self, Self::Identity)
    }

    /// Short label used in logs and reports.
    #[must_use]
    pub fn label(self) -> &'static str {
        match self {
            Self::Identity => "identity",
            Self::Linear => "linear",
            Self::Nonlinear => "nonlinear",
            Self::Saturation => "saturation",
            Self::Combined => "combined",
        }
    }
}

// ---------------------------------------------------------------------------

/// Metadata associated with a single DV shot.
#[derive(Debug, Clone)]
pub struct DvShotMeta {
    /// Start frame index (inclusive).
    pub start_frame: u32,
    /// End frame index (inclusive).
    pub end_frame: u32,
    /// Trim type applied to this shot.
    pub trim_type: ShotTrimType,
    /// Target display peak luminance in nits (must be > 0).
    pub target_peak_nits: f32,
}

impl DvShotMeta {
    /// Create a new shot metadata entry.
    #[must_use]
    pub fn new(
        start_frame: u32,
        end_frame: u32,
        trim_type: ShotTrimType,
        target_peak_nits: f32,
    ) -> Self {
        Self {
            start_frame,
            end_frame,
            trim_type,
            target_peak_nits,
        }
    }

    /// Returns `true` when the entry passes basic sanity checks.
    #[must_use]
    pub fn is_valid(&self) -> bool {
        self.end_frame >= self.start_frame && self.target_peak_nits > 0.0
    }

    /// Duration in frames (inclusive range, so +1).
    #[must_use]
    pub fn duration_frames(&self) -> u32 {
        self.end_frame.saturating_sub(self.start_frame) + 1
    }

    /// Returns `true` if `frame` falls within this shot.
    #[must_use]
    pub fn contains_frame(&self, frame: u32) -> bool {
        frame >= self.start_frame && frame <= self.end_frame
    }
}

// ---------------------------------------------------------------------------

/// An ordered list of [`DvShotMeta`] entries.
#[derive(Debug, Clone, Default)]
pub struct ShotMetaList {
    shots: Vec<DvShotMeta>,
}

impl ShotMetaList {
    /// Create an empty list.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Append a shot entry.
    pub fn add(&mut self, shot: DvShotMeta) {
        self.shots.push(shot);
    }

    /// Total number of shots in the list.
    #[must_use]
    pub fn count(&self) -> usize {
        self.shots.len()
    }

    /// Return all shots whose range overlaps `[from_frame, to_frame]`.
    #[must_use]
    pub fn shots_in_range(&self, from_frame: u32, to_frame: u32) -> Vec<&DvShotMeta> {
        self.shots
            .iter()
            .filter(|s| s.start_frame <= to_frame && s.end_frame >= from_frame)
            .collect()
    }

    /// Returns `true` if all entries pass [`DvShotMeta::is_valid`].
    #[must_use]
    pub fn all_valid(&self) -> bool {
        self.shots.iter().all(|s| s.is_valid())
    }

    /// Iterate over all shots.
    pub fn iter(&self) -> impl Iterator<Item = &DvShotMeta> {
        self.shots.iter()
    }
}

// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn make_shot(start: u32, end: u32) -> DvShotMeta {
        DvShotMeta::new(start, end, ShotTrimType::Linear, 1000.0)
    }

    #[test]
    fn test_trim_type_identity_does_not_affect_playback() {
        assert!(!ShotTrimType::Identity.affects_playback());
    }

    #[test]
    fn test_trim_type_linear_affects_playback() {
        assert!(ShotTrimType::Linear.affects_playback());
    }

    #[test]
    fn test_trim_type_combined_affects_playback() {
        assert!(ShotTrimType::Combined.affects_playback());
    }

    #[test]
    fn test_trim_type_label() {
        assert_eq!(ShotTrimType::Identity.label(), "identity");
        assert_eq!(ShotTrimType::Nonlinear.label(), "nonlinear");
    }

    #[test]
    fn test_shot_meta_is_valid() {
        let s = make_shot(0, 23);
        assert!(s.is_valid());
    }

    #[test]
    fn test_shot_meta_invalid_when_end_before_start() {
        let s = DvShotMeta::new(10, 5, ShotTrimType::Identity, 100.0);
        assert!(!s.is_valid());
    }

    #[test]
    fn test_shot_meta_invalid_when_zero_nits() {
        let s = DvShotMeta::new(0, 23, ShotTrimType::Linear, 0.0);
        assert!(!s.is_valid());
    }

    #[test]
    fn test_shot_meta_duration() {
        let s = make_shot(0, 23);
        assert_eq!(s.duration_frames(), 24);
    }

    #[test]
    fn test_shot_meta_contains_frame() {
        let s = make_shot(10, 20);
        assert!(s.contains_frame(15));
        assert!(!s.contains_frame(9));
        assert!(!s.contains_frame(21));
    }

    #[test]
    fn test_list_count() {
        let mut list = ShotMetaList::new();
        list.add(make_shot(0, 23));
        list.add(make_shot(24, 47));
        assert_eq!(list.count(), 2);
    }

    #[test]
    fn test_shots_in_range_overlap() {
        let mut list = ShotMetaList::new();
        list.add(make_shot(0, 23));
        list.add(make_shot(24, 47));
        list.add(make_shot(100, 123));
        let hits = list.shots_in_range(20, 30);
        assert_eq!(hits.len(), 2);
    }

    #[test]
    fn test_shots_in_range_no_overlap() {
        let mut list = ShotMetaList::new();
        list.add(make_shot(0, 23));
        let hits = list.shots_in_range(50, 100);
        assert!(hits.is_empty());
    }

    #[test]
    fn test_list_all_valid() {
        let mut list = ShotMetaList::new();
        list.add(make_shot(0, 23));
        assert!(list.all_valid());
    }

    #[test]
    fn test_list_not_all_valid() {
        let mut list = ShotMetaList::new();
        list.add(make_shot(0, 23));
        list.add(DvShotMeta::new(50, 40, ShotTrimType::Linear, 100.0)); // invalid
        assert!(!list.all_valid());
    }
}
