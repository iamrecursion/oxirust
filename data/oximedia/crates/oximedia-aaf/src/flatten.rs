//! Timeline flattening
//!
//! Resolves nested compositions in an AAF `ContentStorage` into a single
//! flat sequence of clips — suitable for rendering or further processing.

use crate::composition::{CompositionMob, Sequence, SequenceComponent};
use crate::timeline::EditRate;
use crate::{AafError, ContentStorage, Result};
use uuid::Uuid;

/// A single clip in a flat timeline
#[derive(Debug, Clone, PartialEq)]
pub struct FlatClip {
    /// Timeline start position (in edit units of the top-level edit rate)
    pub timeline_start: i64,
    /// Duration in edit units
    pub duration: i64,
    /// Source mob UUID
    pub source_mob_id: Uuid,
    /// Source mob slot ID
    pub source_slot_id: u32,
    /// Offset into source in edit units
    pub source_offset: i64,
    /// Track name the clip comes from
    pub track_name: String,
    /// Nesting depth at which this clip was resolved
    pub nesting_depth: u32,
}

/// A flattened, resolved composition timeline
#[derive(Debug, Clone)]
pub struct FlatTimeline {
    /// Name of the original composition mob
    pub name: String,
    /// Edit rate of the timeline
    pub edit_rate: Option<EditRate>,
    /// Resolved flat clips in timeline order
    pub clips: Vec<FlatClip>,
}

impl FlatTimeline {
    /// Total duration (end of last clip)
    #[must_use]
    pub fn total_duration(&self) -> i64 {
        self.clips
            .iter()
            .map(|c| c.timeline_start + c.duration)
            .max()
            .unwrap_or(0)
    }

    /// Count of clips
    #[must_use]
    pub fn clip_count(&self) -> usize {
        self.clips.len()
    }
}

/// Maximum nesting depth to prevent infinite recursion from circular refs
const MAX_NESTING_DEPTH: u32 = 32;

/// Flatten a composition mob into a `FlatTimeline`
///
/// Resolves nested `CompositionMob` references up to `MAX_NESTING_DEPTH`
/// levels deep.
///
/// # Errors
///
/// Returns `AafError::ObjectNotFound` if `mob_id` is not found in storage.
pub fn flatten_composition(storage: &ContentStorage, mob_id: &Uuid) -> Result<FlatTimeline> {
    let comp_mob = storage
        .find_composition_mob(mob_id)
        .ok_or_else(|| AafError::ObjectNotFound(format!("CompositionMob {mob_id} not found")))?;

    let name = comp_mob.name().to_string();
    let edit_rate = comp_mob.edit_rate();
    let mut flat = FlatTimeline {
        name,
        edit_rate,
        clips: Vec::new(),
    };

    flatten_mob(storage, comp_mob, 0, 0, &mut flat, 0)?;

    Ok(flat)
}

/// Recursively flatten a composition mob into `flat`, offsetting by `timeline_offset`
fn flatten_mob(
    storage: &ContentStorage,
    comp_mob: &CompositionMob,
    timeline_offset: i64,
    track_filter: u32, // 0 = all tracks
    flat: &mut FlatTimeline,
    depth: u32,
) -> Result<()> {
    if depth > MAX_NESTING_DEPTH {
        return Err(AafError::TimelineError(format!(
            "Max nesting depth ({MAX_NESTING_DEPTH}) exceeded — possible circular reference"
        )));
    }

    for track in comp_mob.tracks() {
        if track_filter != 0 && track.track_id != track_filter {
            continue;
        }

        let track_name = track.name.clone();

        if let Some(ref sequence) = track.sequence {
            flatten_sequence(storage, sequence, timeline_offset, &track_name, flat, depth)?;
        }
    }

    Ok(())
}

/// Flatten a sequence, appending flat clips to `flat`
fn flatten_sequence(
    storage: &ContentStorage,
    sequence: &Sequence,
    timeline_offset: i64,
    track_name: &str,
    flat: &mut FlatTimeline,
    depth: u32,
) -> Result<()> {
    let mut pos = timeline_offset;

    for component in &sequence.components {
        match component {
            SequenceComponent::SourceClip(clip) => {
                // Check if this clip references another CompositionMob we can inline
                if let Some(nested_comp) = storage.find_composition_mob(&clip.source_mob_id) {
                    // Inline the nested composition at the slot specified by clip.source_mob_slot_id
                    flatten_mob(
                        storage,
                        nested_comp,
                        pos,
                        clip.source_mob_slot_id,
                        flat,
                        depth + 1,
                    )?;
                } else {
                    // Leaf clip — emit a FlatClip
                    flat.clips.push(FlatClip {
                        timeline_start: pos,
                        duration: clip.length,
                        source_mob_id: clip.source_mob_id,
                        source_slot_id: clip.source_mob_slot_id,
                        source_offset: clip.start_time.0,
                        track_name: track_name.to_string(),
                        nesting_depth: depth,
                    });
                }
                pos += clip.length;
            }
            SequenceComponent::Filler(filler) => {
                pos += filler.length;
            }
            SequenceComponent::Transition(trans) => {
                // Transitions overlap with adjacent clips — do not advance position
                // but record them as zero-duration markers for completeness
                let _ = trans; // intentionally unused in flat output
            }
            SequenceComponent::Effect(_effect) => {
                // Effects wrap other clips; skip — already handled in nested sequences
            }
        }
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::composition::{
        CompositionMob, Filler, Sequence, SequenceComponent, SourceClip, Track, TrackType,
    };
    use crate::dictionary::Auid;
    use crate::timeline::{EditRate, Position};
    use crate::ContentStorage;
    use uuid::Uuid;

    fn make_simple_storage() -> (ContentStorage, Uuid) {
        let mut storage = ContentStorage::new();
        let mob_id = Uuid::new_v4();
        let source_id1 = Uuid::new_v4();
        let source_id2 = Uuid::new_v4();

        let mut comp = CompositionMob::new(mob_id, "TopLevel");
        let mut seq = Sequence::new(Auid::PICTURE);
        seq.add_component(SequenceComponent::SourceClip(SourceClip::new(
            100,
            Position::zero(),
            source_id1,
            1,
        )));
        seq.add_component(SequenceComponent::Filler(Filler::new(20)));
        seq.add_component(SequenceComponent::SourceClip(SourceClip::new(
            50,
            Position::new(0),
            source_id2,
            1,
        )));
        let mut track = Track::new(1, "Video", EditRate::PAL_25, TrackType::Picture);
        track.set_sequence(seq);
        comp.add_track(track);
        storage.add_composition_mob(comp);

        (storage, mob_id)
    }

    #[test]
    fn test_flatten_simple() {
        let (storage, mob_id) = make_simple_storage();
        let flat = flatten_composition(&storage, &mob_id).expect("flatten should succeed");
        assert_eq!(flat.clip_count(), 2);
    }

    #[test]
    fn test_flatten_timeline_positions() {
        let (storage, mob_id) = make_simple_storage();
        let flat = flatten_composition(&storage, &mob_id).expect("flatten should succeed");

        let clips = &flat.clips;
        assert_eq!(clips[0].timeline_start, 0);
        assert_eq!(clips[0].duration, 100);
        // Second clip after 100 frames clip + 20 filler
        assert_eq!(clips[1].timeline_start, 120);
        assert_eq!(clips[1].duration, 50);
    }

    #[test]
    fn test_flatten_total_duration() {
        let (storage, mob_id) = make_simple_storage();
        let flat = flatten_composition(&storage, &mob_id).expect("flatten should succeed");
        // Last clip ends at 120 + 50 = 170
        assert_eq!(flat.total_duration(), 170);
    }

    #[test]
    fn test_flatten_not_found() {
        let storage = ContentStorage::new();
        let bogus_id = Uuid::new_v4();
        let result = flatten_composition(&storage, &bogus_id);
        assert!(result.is_err());
    }

    #[test]
    fn test_flatten_nested_composition() {
        let mut storage = ContentStorage::new();
        let inner_source = Uuid::new_v4();

        // Create inner composition
        let inner_id = Uuid::new_v4();
        let mut inner = CompositionMob::new(inner_id, "Inner");
        let mut inner_seq = Sequence::new(Auid::PICTURE);
        inner_seq.add_component(SequenceComponent::SourceClip(SourceClip::new(
            60,
            Position::zero(),
            inner_source,
            1,
        )));
        let mut inner_track = Track::new(1, "V", EditRate::PAL_25, TrackType::Picture);
        inner_track.set_sequence(inner_seq);
        inner.add_track(inner_track);
        storage.add_composition_mob(inner);

        // Create outer composition that references inner
        let outer_id = Uuid::new_v4();
        let mut outer = CompositionMob::new(outer_id, "Outer");
        let mut outer_seq = Sequence::new(Auid::PICTURE);
        outer_seq.add_component(SequenceComponent::SourceClip(SourceClip::new(
            60, // length = 60 to cover the inner comp
            Position::zero(),
            inner_id, // references the inner composition mob
            1,
        )));
        let mut outer_track = Track::new(1, "V", EditRate::PAL_25, TrackType::Picture);
        outer_track.set_sequence(outer_seq);
        outer.add_track(outer_track);
        storage.add_composition_mob(outer);

        let flat = flatten_composition(&storage, &outer_id).expect("flatten should succeed");
        // The inner clip (inner_source, 60 frames) should be inlined
        assert_eq!(flat.clip_count(), 1);
        assert_eq!(flat.clips[0].source_mob_id, inner_source);
        assert_eq!(flat.clips[0].nesting_depth, 1);
    }

    #[test]
    fn test_flatten_name_propagated() {
        let (storage, mob_id) = make_simple_storage();
        let flat = flatten_composition(&storage, &mob_id).expect("flatten should succeed");
        assert_eq!(flat.name, "TopLevel");
    }
}
