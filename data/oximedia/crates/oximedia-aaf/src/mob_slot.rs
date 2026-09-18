//! AAF MobSlot extended module
//!
//! Provides higher-level SlotKind, PhysicalTrack, MobSlotDef and MobSlotCollection
//! abstractions for working with AAF track slots.

#[allow(dead_code)]
/// Kind of a mob slot
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SlotKind {
    /// Timeline-based slot
    TimelineSlot,
    /// Static (non-time-varying) slot
    StaticSlot,
    /// Event slot
    EventSlot,
}

impl SlotKind {
    /// Returns `true` if this is a timeline slot
    #[must_use]
    pub fn is_timeline(&self) -> bool {
        matches!(self, SlotKind::TimelineSlot)
    }
}

#[allow(dead_code)]
/// Physical track associated with a mob slot
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PhysicalTrack {
    /// Video track
    Video,
    /// Audio track with channel index
    Audio(u8),
    /// Timecode track
    Timecode,
    /// Edge code track
    EdgeCode,
}

impl PhysicalTrack {
    /// Returns the audio channel number if this is an audio track
    #[must_use]
    pub fn channel(&self) -> Option<u8> {
        if let PhysicalTrack::Audio(ch) = self {
            Some(*ch)
        } else {
            None
        }
    }
}

#[allow(dead_code)]
/// A single mob slot definition (distinct from `object_model::MobSlot`)
#[derive(Debug, Clone)]
pub struct MobSlotDef {
    /// Slot identifier
    pub slot_id: u32,
    /// Human-readable slot name
    pub name: String,
    /// Kind of slot
    pub kind: SlotKind,
    /// Physical track
    pub physical_track: PhysicalTrack,
    /// Edit rate numerator
    pub edit_rate_num: u32,
    /// Edit rate denominator
    pub edit_rate_den: u32,
}

impl MobSlotDef {
    /// Create a new `MobSlotDef`
    #[must_use]
    pub fn new(
        slot_id: u32,
        name: String,
        kind: SlotKind,
        physical_track: PhysicalTrack,
        edit_rate_num: u32,
        edit_rate_den: u32,
    ) -> Self {
        Self {
            slot_id,
            name,
            kind,
            physical_track,
            edit_rate_num,
            edit_rate_den,
        }
    }

    /// Compute the edit rate in frames per second
    #[allow(clippy::cast_precision_loss)]
    #[must_use]
    pub fn edit_rate_fps(&self) -> f64 {
        if self.edit_rate_den == 0 {
            return 0.0;
        }
        self.edit_rate_num as f64 / self.edit_rate_den as f64
    }

    /// Returns `true` if the slot carries video
    #[must_use]
    pub fn is_video(&self) -> bool {
        matches!(self.physical_track, PhysicalTrack::Video)
    }

    /// Returns `true` if the slot carries audio
    #[must_use]
    pub fn is_audio(&self) -> bool {
        matches!(self.physical_track, PhysicalTrack::Audio(_))
    }
}

#[allow(dead_code)]
/// A collection of `MobSlotDef` entries
#[derive(Debug, Clone, Default)]
pub struct MobSlotCollection {
    /// All slots in this collection
    pub slots: Vec<MobSlotDef>,
}

impl MobSlotCollection {
    /// Create an empty collection
    #[must_use]
    pub fn new() -> Self {
        Self { slots: Vec::new() }
    }

    /// Add a slot to the collection
    pub fn add(&mut self, slot: MobSlotDef) {
        self.slots.push(slot);
    }

    /// Find a slot by its identifier
    #[must_use]
    pub fn find_by_id(&self, id: u32) -> Option<&MobSlotDef> {
        self.slots.iter().find(|s| s.slot_id == id)
    }

    /// Return all video slots
    #[must_use]
    pub fn video_slots(&self) -> Vec<&MobSlotDef> {
        self.slots.iter().filter(|s| s.is_video()).collect()
    }

    /// Return all audio slots
    #[must_use]
    pub fn audio_slots(&self) -> Vec<&MobSlotDef> {
        self.slots.iter().filter(|s| s.is_audio()).collect()
    }

    /// Return the total number of slots
    #[must_use]
    pub fn slot_count(&self) -> usize {
        self.slots.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_video_slot(id: u32) -> MobSlotDef {
        MobSlotDef::new(
            id,
            format!("V{}", id),
            SlotKind::TimelineSlot,
            PhysicalTrack::Video,
            24,
            1,
        )
    }

    fn make_audio_slot(id: u32, ch: u8) -> MobSlotDef {
        MobSlotDef::new(
            id,
            format!("A{}", id),
            SlotKind::TimelineSlot,
            PhysicalTrack::Audio(ch),
            48000,
            1,
        )
    }

    #[test]
    fn test_slot_kind_is_timeline() {
        assert!(SlotKind::TimelineSlot.is_timeline());
        assert!(!SlotKind::StaticSlot.is_timeline());
        assert!(!SlotKind::EventSlot.is_timeline());
    }

    #[test]
    fn test_physical_track_channel_video() {
        assert_eq!(PhysicalTrack::Video.channel(), None);
    }

    #[test]
    fn test_physical_track_channel_audio() {
        assert_eq!(PhysicalTrack::Audio(2).channel(), Some(2));
    }

    #[test]
    fn test_physical_track_channel_timecode() {
        assert_eq!(PhysicalTrack::Timecode.channel(), None);
    }

    #[test]
    fn test_physical_track_channel_edgecode() {
        assert_eq!(PhysicalTrack::EdgeCode.channel(), None);
    }

    #[test]
    fn test_mob_slot_def_edit_rate_fps() {
        let slot = make_video_slot(1);
        assert!((slot.edit_rate_fps() - 24.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_mob_slot_def_edit_rate_zero_den() {
        let slot = MobSlotDef::new(
            1,
            "bad".into(),
            SlotKind::StaticSlot,
            PhysicalTrack::Video,
            25,
            0,
        );
        assert_eq!(slot.edit_rate_fps(), 0.0);
    }

    #[test]
    fn test_mob_slot_def_is_video() {
        let slot = make_video_slot(1);
        assert!(slot.is_video());
        assert!(!slot.is_audio());
    }

    #[test]
    fn test_mob_slot_def_is_audio() {
        let slot = make_audio_slot(2, 1);
        assert!(slot.is_audio());
        assert!(!slot.is_video());
    }

    #[test]
    fn test_mob_slot_collection_add_and_count() {
        let mut col = MobSlotCollection::new();
        col.add(make_video_slot(1));
        col.add(make_audio_slot(2, 1));
        assert_eq!(col.slot_count(), 2);
    }

    #[test]
    fn test_mob_slot_collection_find_by_id() {
        let mut col = MobSlotCollection::new();
        col.add(make_video_slot(1));
        col.add(make_audio_slot(2, 1));
        assert!(col.find_by_id(1).is_some());
        assert!(col.find_by_id(99).is_none());
    }

    #[test]
    fn test_mob_slot_collection_video_slots() {
        let mut col = MobSlotCollection::new();
        col.add(make_video_slot(1));
        col.add(make_audio_slot(2, 1));
        col.add(make_audio_slot(3, 2));
        let video = col.video_slots();
        assert_eq!(video.len(), 1);
        assert_eq!(video[0].slot_id, 1);
    }

    #[test]
    fn test_mob_slot_collection_audio_slots() {
        let mut col = MobSlotCollection::new();
        col.add(make_video_slot(1));
        col.add(make_audio_slot(2, 1));
        col.add(make_audio_slot(3, 2));
        let audio = col.audio_slots();
        assert_eq!(audio.len(), 2);
    }

    #[test]
    fn test_mob_slot_collection_default_is_empty() {
        let col = MobSlotCollection::default();
        assert_eq!(col.slot_count(), 0);
    }
}
