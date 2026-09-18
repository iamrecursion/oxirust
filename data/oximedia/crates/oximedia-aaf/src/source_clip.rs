#![allow(dead_code)]
//! Source clip definitions for AAF compositions.
//!
//! A source clip represents a reference to a section of media within an AAF file.
//! Source clips can point to video, audio, timecode, or other essence data
//! stored in master mobs or file source mobs.

use std::collections::HashMap;
use uuid::Uuid;

/// The kind of media that a source clip references.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum SourceClipKind {
    /// Video essence data.
    Video,
    /// Audio essence data (mono, stereo, or multi-channel).
    Audio,
    /// Timecode track reference.
    Timecode,
    /// Edge-code or film key-code reference.
    Edgecode,
    /// Auxiliary / descriptive metadata track.
    Auxiliary,
    /// Data essence (subtitles, captions, ancillary data).
    Data,
}

impl SourceClipKind {
    /// Returns a human-readable label for this kind.
    #[must_use]
    pub const fn label(&self) -> &'static str {
        match self {
            Self::Video => "Video",
            Self::Audio => "Audio",
            Self::Timecode => "Timecode",
            Self::Edgecode => "Edgecode",
            Self::Auxiliary => "Auxiliary",
            Self::Data => "Data",
        }
    }

    /// Returns `true` when the kind carries time-based essence.
    #[must_use]
    pub const fn is_time_based(&self) -> bool {
        matches!(self, Self::Video | Self::Audio)
    }
}

impl std::fmt::Display for SourceClipKind {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.label())
    }
}

/// A single source clip that references a span of media inside an AAF mob.
///
/// Each source clip points at a specific mob via `source_mob_id`, an optional
/// `slot_id` within that mob, and a start-offset plus length measured in edit
/// units at the given `edit_rate_num / edit_rate_den`.
#[derive(Debug, Clone, PartialEq)]
pub struct SourceClip {
    /// Unique identifier for this clip instance.
    id: Uuid,
    /// Kind of essence this clip references.
    kind: SourceClipKind,
    /// Mob ID of the source mob that contains the actual essence.
    source_mob_id: Uuid,
    /// Slot within the source mob (0 means unspecified).
    slot_id: u32,
    /// Start offset in edit units.
    start_offset: i64,
    /// Length in edit units.
    length: i64,
    /// Edit rate numerator.
    edit_rate_num: u32,
    /// Edit rate denominator.
    edit_rate_den: u32,
    /// Optional human-readable name.
    name: Option<String>,
}

impl SourceClip {
    /// Create a new source clip.
    #[must_use]
    pub fn new(
        kind: SourceClipKind,
        source_mob_id: Uuid,
        slot_id: u32,
        start_offset: i64,
        length: i64,
        edit_rate_num: u32,
        edit_rate_den: u32,
    ) -> Self {
        Self {
            id: Uuid::new_v4(),
            kind,
            source_mob_id,
            slot_id,
            start_offset,
            length,
            edit_rate_num,
            edit_rate_den,
            name: None,
        }
    }

    /// Create a source clip with a specific id (for deserialization).
    #[must_use]
    pub fn with_id(mut self, id: Uuid) -> Self {
        self.id = id;
        self
    }

    /// Attach a human-readable name.
    #[must_use]
    pub fn with_name(mut self, name: impl Into<String>) -> Self {
        self.name = Some(name.into());
        self
    }

    /// Unique identifier.
    #[must_use]
    pub fn id(&self) -> Uuid {
        self.id
    }

    /// Kind of essence.
    #[must_use]
    pub fn kind(&self) -> SourceClipKind {
        self.kind
    }

    /// Mob ID of the referenced source mob.
    #[must_use]
    pub fn source_mob_id(&self) -> Uuid {
        self.source_mob_id
    }

    /// Slot within the source mob.
    #[must_use]
    pub fn slot_id(&self) -> u32 {
        self.slot_id
    }

    /// Start offset in edit units.
    #[must_use]
    pub fn start_offset(&self) -> i64 {
        self.start_offset
    }

    /// Length in edit units.
    #[must_use]
    pub fn length(&self) -> i64 {
        self.length
    }

    /// Edit rate as `(numerator, denominator)`.
    #[must_use]
    pub fn edit_rate(&self) -> (u32, u32) {
        (self.edit_rate_num, self.edit_rate_den)
    }

    /// Optional human-readable name.
    #[must_use]
    pub fn name(&self) -> Option<&str> {
        self.name.as_deref()
    }

    /// Duration in seconds (floating point).
    #[must_use]
    #[allow(clippy::cast_precision_loss)]
    pub fn duration_seconds(&self) -> f64 {
        if self.edit_rate_den == 0 || self.edit_rate_num == 0 {
            return 0.0;
        }
        let rate = self.edit_rate_num as f64 / self.edit_rate_den as f64;
        self.length as f64 / rate
    }

    /// End offset (start + length) in edit units.
    #[must_use]
    pub fn end_offset(&self) -> i64 {
        self.start_offset.saturating_add(self.length)
    }

    /// Returns `true` when this clip overlaps with another clip (same mob, same slot).
    #[must_use]
    pub fn overlaps(&self, other: &Self) -> bool {
        if self.source_mob_id != other.source_mob_id || self.slot_id != other.slot_id {
            return false;
        }
        self.start_offset < other.end_offset() && other.start_offset < self.end_offset()
    }
}

/// A pool that holds multiple [`SourceClip`] instances, indexed by their id.
///
/// The pool provides efficient lookup, insertion, and query by mob id.
#[derive(Debug, Clone)]
pub struct SourceClipPool {
    clips: HashMap<Uuid, SourceClip>,
}

impl SourceClipPool {
    /// Create an empty pool.
    #[must_use]
    pub fn new() -> Self {
        Self {
            clips: HashMap::new(),
        }
    }

    /// Insert a clip and return the previous value if one existed for the same id.
    pub fn insert(&mut self, clip: SourceClip) -> Option<SourceClip> {
        self.clips.insert(clip.id(), clip)
    }

    /// Remove a clip by id.
    pub fn remove(&mut self, id: &Uuid) -> Option<SourceClip> {
        self.clips.remove(id)
    }

    /// Look up a clip by id.
    #[must_use]
    pub fn get(&self, id: &Uuid) -> Option<&SourceClip> {
        self.clips.get(id)
    }

    /// Number of clips in the pool.
    #[must_use]
    pub fn len(&self) -> usize {
        self.clips.len()
    }

    /// Returns `true` when the pool is empty.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.clips.is_empty()
    }

    /// Return all clips that reference a given source mob id.
    #[must_use]
    pub fn clips_for_mob(&self, mob_id: &Uuid) -> Vec<&SourceClip> {
        self.clips
            .values()
            .filter(|c| &c.source_mob_id == mob_id)
            .collect()
    }

    /// Return all clips of a given kind.
    #[must_use]
    pub fn clips_of_kind(&self, kind: SourceClipKind) -> Vec<&SourceClip> {
        self.clips.values().filter(|c| c.kind == kind).collect()
    }

    /// Total duration of all clips in seconds.
    #[must_use]
    pub fn total_duration_seconds(&self) -> f64 {
        self.clips.values().map(|c| c.duration_seconds()).sum()
    }

    /// Iterator over all clips.
    pub fn iter(&self) -> impl Iterator<Item = &SourceClip> {
        self.clips.values()
    }
}

impl Default for SourceClipPool {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_clip(kind: SourceClipKind, mob: Uuid, start: i64, len: i64) -> SourceClip {
        SourceClip::new(kind, mob, 1, start, len, 25, 1)
    }

    #[test]
    fn test_source_clip_kind_label() {
        assert_eq!(SourceClipKind::Video.label(), "Video");
        assert_eq!(SourceClipKind::Audio.label(), "Audio");
        assert_eq!(SourceClipKind::Timecode.label(), "Timecode");
        assert_eq!(SourceClipKind::Data.label(), "Data");
    }

    #[test]
    fn test_source_clip_kind_is_time_based() {
        assert!(SourceClipKind::Video.is_time_based());
        assert!(SourceClipKind::Audio.is_time_based());
        assert!(!SourceClipKind::Timecode.is_time_based());
        assert!(!SourceClipKind::Edgecode.is_time_based());
    }

    #[test]
    fn test_source_clip_kind_display() {
        assert_eq!(format!("{}", SourceClipKind::Auxiliary), "Auxiliary");
    }

    #[test]
    fn test_source_clip_creation() {
        let mob = Uuid::new_v4();
        let clip = SourceClip::new(SourceClipKind::Video, mob, 1, 0, 100, 25, 1);
        assert_eq!(clip.kind(), SourceClipKind::Video);
        assert_eq!(clip.source_mob_id(), mob);
        assert_eq!(clip.slot_id(), 1);
        assert_eq!(clip.start_offset(), 0);
        assert_eq!(clip.length(), 100);
        assert_eq!(clip.edit_rate(), (25, 1));
        assert!(clip.name().is_none());
    }

    #[test]
    fn test_source_clip_with_name() {
        let mob = Uuid::new_v4();
        let clip = SourceClip::new(SourceClipKind::Audio, mob, 2, 10, 50, 48000, 1)
            .with_name("dialogue_01");
        assert_eq!(clip.name(), Some("dialogue_01"));
    }

    #[test]
    fn test_source_clip_with_id() {
        let mob = Uuid::new_v4();
        let fixed_id = Uuid::new_v4();
        let clip = SourceClip::new(SourceClipKind::Video, mob, 1, 0, 50, 25, 1).with_id(fixed_id);
        assert_eq!(clip.id(), fixed_id);
    }

    #[test]
    fn test_source_clip_duration_seconds() {
        let mob = Uuid::new_v4();
        let clip = SourceClip::new(SourceClipKind::Video, mob, 1, 0, 75, 25, 1);
        let dur = clip.duration_seconds();
        assert!((dur - 3.0).abs() < 1e-9);
    }

    #[test]
    fn test_source_clip_duration_zero_rate() {
        let mob = Uuid::new_v4();
        let clip = SourceClip::new(SourceClipKind::Video, mob, 1, 0, 100, 0, 1);
        assert_eq!(clip.duration_seconds(), 0.0);
    }

    #[test]
    fn test_source_clip_end_offset() {
        let mob = Uuid::new_v4();
        let clip = SourceClip::new(SourceClipKind::Video, mob, 1, 10, 40, 25, 1);
        assert_eq!(clip.end_offset(), 50);
    }

    #[test]
    fn test_source_clip_overlaps_true() {
        let mob = Uuid::new_v4();
        let a = make_clip(SourceClipKind::Video, mob, 0, 100);
        let b = SourceClip::new(SourceClipKind::Video, mob, 1, 50, 100, 25, 1);
        assert!(a.overlaps(&b));
        assert!(b.overlaps(&a));
    }

    #[test]
    fn test_source_clip_overlaps_false_different_mob() {
        let a = make_clip(SourceClipKind::Video, Uuid::new_v4(), 0, 100);
        let b = make_clip(SourceClipKind::Video, Uuid::new_v4(), 50, 100);
        assert!(!a.overlaps(&b));
    }

    #[test]
    fn test_source_clip_overlaps_adjacent() {
        let mob = Uuid::new_v4();
        let a = make_clip(SourceClipKind::Video, mob, 0, 50);
        let b = SourceClip::new(SourceClipKind::Video, mob, 1, 50, 50, 25, 1);
        assert!(!a.overlaps(&b));
    }

    #[test]
    fn test_pool_insert_and_get() {
        let mut pool = SourceClipPool::new();
        let mob = Uuid::new_v4();
        let clip = make_clip(SourceClipKind::Video, mob, 0, 100);
        let id = clip.id();
        pool.insert(clip);
        assert_eq!(pool.len(), 1);
        assert!(!pool.is_empty());
        assert!(pool.get(&id).is_some());
    }

    #[test]
    fn test_pool_remove() {
        let mut pool = SourceClipPool::new();
        let clip = make_clip(SourceClipKind::Audio, Uuid::new_v4(), 0, 50);
        let id = clip.id();
        pool.insert(clip);
        assert!(pool.remove(&id).is_some());
        assert!(pool.is_empty());
    }

    #[test]
    fn test_pool_clips_for_mob() {
        let mut pool = SourceClipPool::new();
        let mob_a = Uuid::new_v4();
        let mob_b = Uuid::new_v4();
        pool.insert(make_clip(SourceClipKind::Video, mob_a, 0, 50));
        pool.insert(make_clip(SourceClipKind::Audio, mob_a, 0, 50));
        pool.insert(make_clip(SourceClipKind::Video, mob_b, 0, 100));
        assert_eq!(pool.clips_for_mob(&mob_a).len(), 2);
        assert_eq!(pool.clips_for_mob(&mob_b).len(), 1);
    }

    #[test]
    fn test_pool_clips_of_kind() {
        let mut pool = SourceClipPool::new();
        let mob = Uuid::new_v4();
        pool.insert(make_clip(SourceClipKind::Video, mob, 0, 50));
        pool.insert(make_clip(SourceClipKind::Audio, mob, 0, 50));
        pool.insert(make_clip(SourceClipKind::Video, mob, 50, 50));
        assert_eq!(pool.clips_of_kind(SourceClipKind::Video).len(), 2);
        assert_eq!(pool.clips_of_kind(SourceClipKind::Audio).len(), 1);
    }

    #[test]
    fn test_pool_total_duration() {
        let mut pool = SourceClipPool::new();
        let mob = Uuid::new_v4();
        pool.insert(make_clip(SourceClipKind::Video, mob, 0, 25)); // 1 sec
        pool.insert(make_clip(SourceClipKind::Video, mob, 25, 50)); // 2 sec
        let total = pool.total_duration_seconds();
        assert!((total - 3.0).abs() < 1e-9);
    }
}
