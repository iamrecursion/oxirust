//! AAF timeline mob management
//!
//! Provides types for managing timeline mob objects in AAF files, including
//! timeline mob creation, track layout, and edit-rate-aware position calculations
//! per SMPTE ST 377-1 Section 11.

#![allow(dead_code)]
#![allow(clippy::cast_precision_loss)]

use std::collections::HashMap;

/// Edit rate as a rational number (numerator / denominator).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct MobEditRate {
    /// Numerator of the rate fraction (e.g. 24000).
    pub numerator: u32,
    /// Denominator of the rate fraction (e.g. 1001).
    pub denominator: u32,
}

impl MobEditRate {
    /// Create a new edit rate.
    #[must_use]
    pub const fn new(numerator: u32, denominator: u32) -> Self {
        Self {
            numerator,
            denominator,
        }
    }

    /// The rate expressed as a floating-point value.
    #[must_use]
    pub fn as_f64(&self) -> f64 {
        if self.denominator == 0 {
            return 0.0;
        }
        f64::from(self.numerator) / f64::from(self.denominator)
    }

    /// Standard 23.976 fps (24000/1001).
    #[must_use]
    pub const fn fps_23_976() -> Self {
        Self::new(24000, 1001)
    }

    /// Standard 24 fps.
    #[must_use]
    pub const fn fps_24() -> Self {
        Self::new(24, 1)
    }

    /// Standard 25 fps (PAL).
    #[must_use]
    pub const fn fps_25() -> Self {
        Self::new(25, 1)
    }

    /// Standard 29.97 fps (NTSC).
    #[must_use]
    pub const fn fps_29_97() -> Self {
        Self::new(30000, 1001)
    }

    /// Standard 30 fps.
    #[must_use]
    pub const fn fps_30() -> Self {
        Self::new(30, 1)
    }

    /// Standard 48 kHz audio sample rate.
    #[must_use]
    pub const fn audio_48k() -> Self {
        Self::new(48000, 1)
    }

    /// Convert a position from this edit rate to another.
    #[must_use]
    pub fn convert_position(&self, position: i64, target: &MobEditRate) -> i64 {
        if self.denominator == 0 || target.denominator == 0 || self.numerator == 0 {
            return 0;
        }
        let source_rate = self.as_f64();
        let target_rate = target.as_f64();
        if source_rate == 0.0 {
            return 0;
        }
        let seconds = position as f64 / source_rate;
        (seconds * target_rate).round() as i64
    }

    /// Check if this rate is drop-frame (NTSC).
    #[must_use]
    pub fn is_drop_frame(&self) -> bool {
        self.denominator == 1001
    }
}

impl Default for MobEditRate {
    fn default() -> Self {
        Self::fps_24()
    }
}

/// The kind of a mob in AAF.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum MobKind {
    /// Composition mob (top-level edit sequence).
    Composition,
    /// Master mob (links composition to source).
    Master,
    /// Source mob (references physical essence).
    Source,
}

impl MobKind {
    /// Human-readable label.
    #[must_use]
    pub const fn label(&self) -> &'static str {
        match self {
            Self::Composition => "CompositionMob",
            Self::Master => "MasterMob",
            Self::Source => "SourceMob",
        }
    }
}

/// A single timeline slot within a mob.
#[derive(Debug, Clone)]
pub struct TimelineSlot {
    /// Slot identifier (unique within the mob).
    pub slot_id: u32,
    /// Human-readable slot name.
    pub name: String,
    /// Edit rate of this slot.
    pub edit_rate: MobEditRate,
    /// Origin offset in edit units.
    pub origin: i64,
    /// Segment length in edit units (0 = unknown).
    pub length: i64,
    /// Physical track number (for audio channel mapping).
    pub physical_track: Option<u32>,
}

impl TimelineSlot {
    /// Create a new timeline slot.
    #[must_use]
    pub fn new(slot_id: u32, name: impl Into<String>, edit_rate: MobEditRate) -> Self {
        Self {
            slot_id,
            name: name.into(),
            edit_rate,
            origin: 0,
            length: 0,
            physical_track: None,
        }
    }

    /// Set the origin offset.
    pub fn with_origin(mut self, origin: i64) -> Self {
        self.origin = origin;
        self
    }

    /// Set the segment length.
    pub fn with_length(mut self, length: i64) -> Self {
        self.length = length;
        self
    }

    /// Set the physical track number.
    pub fn with_physical_track(mut self, track: u32) -> Self {
        self.physical_track = Some(track);
        self
    }

    /// Duration of this slot in seconds.
    #[must_use]
    pub fn duration_seconds(&self) -> f64 {
        let rate = self.edit_rate.as_f64();
        if rate == 0.0 {
            return 0.0;
        }
        self.length as f64 / rate
    }

    /// End position (origin + length).
    #[must_use]
    pub fn end_position(&self) -> i64 {
        self.origin + self.length
    }
}

/// A timeline mob representing a top-level composition, master, or source mob.
#[derive(Debug, Clone)]
pub struct TimelineMob {
    /// Mob unique identifier string.
    pub mob_id: String,
    /// Human-readable mob name.
    pub name: String,
    /// Kind of this mob.
    pub kind: MobKind,
    /// Timeline slots within this mob, keyed by slot_id.
    pub slots: Vec<TimelineSlot>,
    /// User comments / metadata.
    pub comments: HashMap<String, String>,
    /// Creation date as ISO 8601 string.
    pub creation_date: Option<String>,
    /// Last modification date as ISO 8601 string.
    pub modification_date: Option<String>,
}

impl TimelineMob {
    /// Create a new timeline mob.
    #[must_use]
    pub fn new(mob_id: impl Into<String>, name: impl Into<String>, kind: MobKind) -> Self {
        Self {
            mob_id: mob_id.into(),
            name: name.into(),
            kind,
            slots: Vec::new(),
            comments: HashMap::new(),
            creation_date: None,
            modification_date: None,
        }
    }

    /// Add a timeline slot.
    pub fn add_slot(&mut self, slot: TimelineSlot) {
        self.slots.push(slot);
    }

    /// Find a slot by its ID.
    #[must_use]
    pub fn find_slot(&self, slot_id: u32) -> Option<&TimelineSlot> {
        self.slots.iter().find(|s| s.slot_id == slot_id)
    }

    /// Find a slot by its ID (mutable).
    pub fn find_slot_mut(&mut self, slot_id: u32) -> Option<&mut TimelineSlot> {
        self.slots.iter_mut().find(|s| s.slot_id == slot_id)
    }

    /// Number of slots.
    #[must_use]
    pub fn slot_count(&self) -> usize {
        self.slots.len()
    }

    /// Duration of the longest slot in seconds.
    #[must_use]
    pub fn max_duration_seconds(&self) -> f64 {
        self.slots
            .iter()
            .map(|s| s.duration_seconds())
            .fold(0.0_f64, f64::max)
    }

    /// Add a user comment.
    pub fn add_comment(&mut self, key: impl Into<String>, value: impl Into<String>) {
        self.comments.insert(key.into(), value.into());
    }

    /// Get a user comment.
    #[must_use]
    pub fn get_comment(&self, key: &str) -> Option<&str> {
        self.comments.get(key).map(String::as_str)
    }

    /// All video slots (slots whose name contains "Video" or "V").
    #[must_use]
    pub fn video_slots(&self) -> Vec<&TimelineSlot> {
        self.slots
            .iter()
            .filter(|s| {
                let n = s.name.to_lowercase();
                n.contains("video") || n.starts_with('v')
            })
            .collect()
    }

    /// All audio slots (slots whose name contains "Audio" or "A").
    #[must_use]
    pub fn audio_slots(&self) -> Vec<&TimelineSlot> {
        self.slots
            .iter()
            .filter(|s| {
                let n = s.name.to_lowercase();
                n.contains("audio") || n.starts_with('a')
            })
            .collect()
    }
}

/// A collection of timeline mobs forming a complete AAF content storage view.
#[derive(Debug, Default)]
pub struct TimelineMobCollection {
    /// All mobs indexed by mob_id.
    mobs: Vec<TimelineMob>,
}

impl TimelineMobCollection {
    /// Create an empty collection.
    #[must_use]
    pub fn new() -> Self {
        Self { mobs: Vec::new() }
    }

    /// Add a mob to the collection.
    pub fn add(&mut self, mob: TimelineMob) {
        self.mobs.push(mob);
    }

    /// Number of mobs.
    #[must_use]
    pub fn len(&self) -> usize {
        self.mobs.len()
    }

    /// Whether the collection is empty.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.mobs.is_empty()
    }

    /// Find a mob by its ID string.
    #[must_use]
    pub fn find_by_id(&self, mob_id: &str) -> Option<&TimelineMob> {
        self.mobs.iter().find(|m| m.mob_id == mob_id)
    }

    /// All composition mobs.
    #[must_use]
    pub fn compositions(&self) -> Vec<&TimelineMob> {
        self.mobs
            .iter()
            .filter(|m| m.kind == MobKind::Composition)
            .collect()
    }

    /// All master mobs.
    #[must_use]
    pub fn masters(&self) -> Vec<&TimelineMob> {
        self.mobs
            .iter()
            .filter(|m| m.kind == MobKind::Master)
            .collect()
    }

    /// All source mobs.
    #[must_use]
    pub fn sources(&self) -> Vec<&TimelineMob> {
        self.mobs
            .iter()
            .filter(|m| m.kind == MobKind::Source)
            .collect()
    }

    /// Total number of timeline slots across all mobs.
    #[must_use]
    pub fn total_slot_count(&self) -> usize {
        self.mobs.iter().map(|m| m.slot_count()).sum()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mob_edit_rate_as_f64() {
        let rate = MobEditRate::fps_24();
        assert!((rate.as_f64() - 24.0).abs() < 1e-9);
    }

    #[test]
    fn test_mob_edit_rate_23_976() {
        let rate = MobEditRate::fps_23_976();
        assert!((rate.as_f64() - 23.976).abs() < 0.001);
    }

    #[test]
    fn test_mob_edit_rate_drop_frame() {
        assert!(MobEditRate::fps_29_97().is_drop_frame());
        assert!(!MobEditRate::fps_30().is_drop_frame());
    }

    #[test]
    fn test_convert_position_same_rate() {
        let rate = MobEditRate::fps_24();
        assert_eq!(rate.convert_position(240, &rate), 240);
    }

    #[test]
    fn test_convert_position_video_to_audio() {
        let video = MobEditRate::fps_24();
        let audio = MobEditRate::audio_48k();
        let audio_pos = video.convert_position(24, &audio);
        assert_eq!(audio_pos, 48000);
    }

    #[test]
    fn test_convert_position_zero_rate() {
        let zero = MobEditRate::new(0, 1);
        let target = MobEditRate::fps_24();
        assert_eq!(zero.convert_position(100, &target), 0);
    }

    #[test]
    fn test_timeline_slot_creation() {
        let slot = TimelineSlot::new(1, "V1", MobEditRate::fps_24())
            .with_origin(0)
            .with_length(240)
            .with_physical_track(1);
        assert_eq!(slot.slot_id, 1);
        assert_eq!(slot.name, "V1");
        assert_eq!(slot.length, 240);
        assert_eq!(slot.physical_track, Some(1));
    }

    #[test]
    fn test_timeline_slot_duration_seconds() {
        let slot = TimelineSlot::new(1, "V1", MobEditRate::fps_24()).with_length(48);
        assert!((slot.duration_seconds() - 2.0).abs() < 1e-9);
    }

    #[test]
    fn test_timeline_slot_end_position() {
        let slot = TimelineSlot::new(1, "V1", MobEditRate::fps_24())
            .with_origin(10)
            .with_length(100);
        assert_eq!(slot.end_position(), 110);
    }

    #[test]
    fn test_timeline_mob_basic() {
        let mut mob = TimelineMob::new("mob-001", "Scene1", MobKind::Composition);
        mob.add_slot(TimelineSlot::new(1, "Video1", MobEditRate::fps_24()).with_length(240));
        mob.add_slot(TimelineSlot::new(2, "Audio1", MobEditRate::audio_48k()).with_length(480000));
        assert_eq!(mob.slot_count(), 2);
        assert!(mob.find_slot(1).is_some());
        assert!(mob.find_slot(99).is_none());
    }

    #[test]
    fn test_timeline_mob_max_duration() {
        let mut mob = TimelineMob::new("mob-002", "Scene2", MobKind::Master);
        mob.add_slot(TimelineSlot::new(1, "V1", MobEditRate::fps_24()).with_length(48));
        mob.add_slot(TimelineSlot::new(2, "V2", MobEditRate::fps_24()).with_length(240));
        assert!((mob.max_duration_seconds() - 10.0).abs() < 1e-9);
    }

    #[test]
    fn test_timeline_mob_comments() {
        let mut mob = TimelineMob::new("mob-003", "S", MobKind::Source);
        mob.add_comment("Director", "Jane");
        assert_eq!(mob.get_comment("Director"), Some("Jane"));
        assert!(mob.get_comment("Missing").is_none());
    }

    #[test]
    fn test_timeline_mob_collection() {
        let mut coll = TimelineMobCollection::new();
        assert!(coll.is_empty());

        coll.add(TimelineMob::new("c1", "Comp", MobKind::Composition));
        coll.add(TimelineMob::new("m1", "Master", MobKind::Master));
        coll.add(TimelineMob::new("s1", "Source", MobKind::Source));

        assert_eq!(coll.len(), 3);
        assert_eq!(coll.compositions().len(), 1);
        assert_eq!(coll.masters().len(), 1);
        assert_eq!(coll.sources().len(), 1);
    }

    #[test]
    fn test_timeline_mob_collection_find_by_id() {
        let mut coll = TimelineMobCollection::new();
        coll.add(TimelineMob::new("abc", "Test", MobKind::Composition));
        assert!(coll.find_by_id("abc").is_some());
        assert!(coll.find_by_id("xyz").is_none());
    }

    #[test]
    fn test_mob_kind_label() {
        assert_eq!(MobKind::Composition.label(), "CompositionMob");
        assert_eq!(MobKind::Master.label(), "MasterMob");
        assert_eq!(MobKind::Source.label(), "SourceMob");
    }

    #[test]
    fn test_mob_edit_rate_default() {
        let rate = MobEditRate::default();
        assert_eq!(rate, MobEditRate::fps_24());
    }

    #[test]
    fn test_total_slot_count() {
        let mut coll = TimelineMobCollection::new();
        let mut m1 = TimelineMob::new("a", "A", MobKind::Composition);
        m1.add_slot(TimelineSlot::new(1, "V1", MobEditRate::fps_24()));
        m1.add_slot(TimelineSlot::new(2, "A1", MobEditRate::audio_48k()));
        let mut m2 = TimelineMob::new("b", "B", MobKind::Master);
        m2.add_slot(TimelineSlot::new(1, "V1", MobEditRate::fps_24()));
        coll.add(m1);
        coll.add(m2);
        assert_eq!(coll.total_slot_count(), 3);
    }
}
