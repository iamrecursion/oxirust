//! Composition structures
//!
//! This module implements AAF composition objects:
//! - `CompositionMob`: Represents an edit/sequence
//! - Track: Individual timeline track
//! - Sequence: Collection of segments
//! - `SourceClip`: Reference to media
//! - Composition hierarchy and navigation

use crate::dictionary::Auid;
use crate::object_model::{
    Component, FillerSegment, Mob, MobSlot, MobType, OperationGroupSegment, Segment,
    SequenceSegment, SourceClipSegment, TransitionSegment,
};
use crate::timeline::{EditRate, Position};
use std::collections::HashMap;
use uuid::Uuid;

/// Composition Mob - represents an edit/sequence
#[derive(Debug, Clone)]
pub struct CompositionMob {
    /// Underlying mob
    mob: Mob,
    /// Default fade length
    pub default_fade_length: Option<i64>,
    /// Default fade type
    pub default_fade_type: Option<FadeType>,
    /// Usage code
    pub usage_code: Option<UsageCode>,
}

impl CompositionMob {
    /// Create a new composition mob
    pub fn new(mob_id: Uuid, name: impl Into<String>) -> Self {
        Self {
            mob: Mob::new(mob_id, name.into(), MobType::Composition),
            default_fade_length: None,
            default_fade_type: None,
            usage_code: None,
        }
    }

    /// Get mob ID
    #[must_use]
    pub fn mob_id(&self) -> Uuid {
        self.mob.mob_id()
    }

    /// Get name
    #[must_use]
    pub fn name(&self) -> &str {
        self.mob.name()
    }

    /// Get tracks
    #[must_use]
    pub fn tracks(&self) -> Vec<Track> {
        self.mob
            .slots()
            .iter()
            .map(|slot| Track::from_mob_slot(slot.clone()))
            .collect()
    }

    /// Get track by slot ID
    #[must_use]
    pub fn get_track(&self, slot_id: u32) -> Option<Track> {
        self.mob
            .get_slot(slot_id)
            .map(|slot| Track::from_mob_slot(slot.clone()))
    }

    /// Add a track
    pub fn add_track(&mut self, track: Track) {
        self.mob.add_slot(track.into_mob_slot());
    }

    /// Get edit rate (from first track)
    #[must_use]
    pub fn edit_rate(&self) -> Option<EditRate> {
        self.tracks().first().map(|t| t.edit_rate)
    }

    /// Get duration (from longest track)
    #[must_use]
    pub fn duration(&self) -> Option<i64> {
        self.tracks().iter().filter_map(Track::duration).max()
    }

    /// Get all picture tracks
    #[must_use]
    pub fn picture_tracks(&self) -> Vec<Track> {
        self.tracks()
            .into_iter()
            .filter(Track::is_picture)
            .collect()
    }

    /// Get all sound tracks
    #[must_use]
    pub fn sound_tracks(&self) -> Vec<Track> {
        self.tracks().into_iter().filter(Track::is_sound).collect()
    }

    /// Get all timecode tracks
    #[must_use]
    pub fn timecode_tracks(&self) -> Vec<Track> {
        self.tracks()
            .into_iter()
            .filter(Track::is_timecode)
            .collect()
    }

    /// Get the underlying mob
    #[must_use]
    pub fn mob(&self) -> &Mob {
        &self.mob
    }

    /// Get mutable reference to underlying mob
    pub fn mob_mut(&mut self) -> &mut Mob {
        &mut self.mob
    }

    /// Set default fade
    pub fn set_default_fade(&mut self, length: i64, fade_type: FadeType) {
        self.default_fade_length = Some(length);
        self.default_fade_type = Some(fade_type);
    }

    /// Set usage code
    pub fn set_usage_code(&mut self, code: UsageCode) {
        self.usage_code = Some(code);
    }

    // ─── Track re-ordering and insertion APIs ─────────────────────────────────

    /// Return mutable `Track` objects — changes are applied to the underlying slots.
    ///
    /// Converts each slot to a `Track`, applies `f`, then writes back.
    pub fn tracks_mut(&mut self) -> Vec<&mut MobSlot> {
        self.mob.slots.iter_mut().collect()
    }

    /// Insert a new track at the given position (0-based index).
    ///
    /// All existing tracks at or after `index` are shifted right.
    /// If `index` is greater than the current track count the track is appended.
    pub fn insert_track_at(&mut self, index: usize, track: Track) {
        let slot = track.into_mob_slot();
        let len = self.mob.slots.len();
        let idx = index.min(len);
        self.mob.slots.insert(idx, slot);
    }

    /// Remove and return the track at the given slot index (position in the internal list).
    ///
    /// Returns `None` if `index` is out of bounds.
    pub fn remove_track_at(&mut self, index: usize) -> Option<Track> {
        if index >= self.mob.slots.len() {
            None
        } else {
            Some(Track::from_mob_slot(self.mob.slots.remove(index)))
        }
    }

    /// Move the track currently at `from_index` to `to_index`.
    ///
    /// Both indices are positions in the internal slot list.
    /// If either index is out of bounds the operation is a no-op and
    /// `false` is returned; otherwise `true`.
    pub fn move_track(&mut self, from_index: usize, to_index: usize) -> bool {
        let len = self.mob.slots.len();
        if from_index >= len || to_index >= len || from_index == to_index {
            return false;
        }
        let slot = self.mob.slots.remove(from_index);
        // After removal the target index may have shifted
        let adjusted_to = if to_index > from_index {
            to_index - 1
        } else {
            to_index
        };
        self.mob.slots.insert(adjusted_to, slot);
        true
    }

    /// Re-order all tracks according to a permutation vector.
    ///
    /// `order` must be a permutation of `0..track_count()`.  Each element
    /// gives the *current* index of the track that should occupy that output
    /// position.
    ///
    /// Returns `Ok(())` on success, or `AafError::TimelineError` if `order`
    /// is not a valid permutation.
    pub fn reorder_tracks(&mut self, order: &[usize]) -> crate::Result<()> {
        let len = self.mob.slots.len();
        if order.len() != len {
            return Err(crate::AafError::TimelineError(format!(
                "reorder_tracks: order length {} != track count {}",
                order.len(),
                len
            )));
        }
        // Validate permutation
        let mut seen = vec![false; len];
        for &idx in order {
            if idx >= len {
                return Err(crate::AafError::TimelineError(format!(
                    "reorder_tracks: index {idx} out of bounds (len={len})"
                )));
            }
            if seen[idx] {
                return Err(crate::AafError::TimelineError(format!(
                    "reorder_tracks: duplicate index {idx}"
                )));
            }
            seen[idx] = true;
        }
        let old_slots = self.mob.slots.clone();
        for (pos, &src_idx) in order.iter().enumerate() {
            self.mob.slots[pos] = old_slots[src_idx].clone();
        }
        Ok(())
    }

    /// Get the number of tracks (slots)
    #[must_use]
    pub fn track_count(&self) -> usize {
        self.mob.slots.len()
    }
}

/// Fade type
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FadeType {
    /// Linear fade
    Linear,
    /// Logarithmic fade
    Logarithmic,
    /// Exponential fade
    Exponential,
    /// S-curve fade
    SCurve,
}

/// Usage code for composition mobs
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum UsageCode {
    /// Top level composition
    TopLevel,
    /// Lower level composition
    LowerLevel,
    /// Sub-clip
    SubClip,
    /// Adjusted clip
    AdjustedClip,
    /// Template
    Template,
}

/// Track - represents a timeline track
#[derive(Debug, Clone)]
pub struct Track {
    /// Track ID
    pub track_id: u32,
    /// Track name
    pub name: String,
    /// Edit rate
    pub edit_rate: EditRate,
    /// Origin (start position)
    pub origin: Position,
    /// Physical track number
    pub physical_track_number: Option<u32>,
    /// Sequence
    pub sequence: Option<Sequence>,
    /// Track type
    pub track_type: TrackType,
}

impl Track {
    /// Create a new track
    pub fn new(
        track_id: u32,
        name: impl Into<String>,
        edit_rate: EditRate,
        track_type: TrackType,
    ) -> Self {
        Self {
            track_id,
            name: name.into(),
            edit_rate,
            origin: Position::zero(),
            physical_track_number: None,
            sequence: None,
            track_type,
        }
    }

    /// Create from a mob slot
    #[must_use]
    pub fn from_mob_slot(slot: MobSlot) -> Self {
        let track_type = if let Some(ref segment) = slot.segment {
            determine_track_type(segment.as_ref())
        } else {
            TrackType::Unknown
        };

        let sequence = if let Some(ref segment) = slot.segment {
            extract_sequence(segment.as_ref())
        } else {
            None
        };

        Self {
            track_id: slot.slot_id,
            name: slot.name,
            edit_rate: slot.edit_rate,
            origin: slot.origin,
            physical_track_number: slot.physical_track_number,
            sequence,
            track_type,
        }
    }

    /// Convert to mob slot
    #[must_use]
    pub fn into_mob_slot(self) -> MobSlot {
        let segment = self
            .sequence
            .map(|sequence| Box::new(Segment::Sequence(sequence.into_segment())));

        MobSlot {
            slot_id: self.track_id,
            name: self.name,
            physical_track_number: self.physical_track_number,
            edit_rate: self.edit_rate,
            origin: self.origin,
            segment,
            slot_type: crate::object_model::SlotType::Timeline,
        }
    }

    /// Get duration
    #[must_use]
    pub fn duration(&self) -> Option<i64> {
        self.sequence.as_ref().and_then(Sequence::duration)
    }

    /// Check if this is a picture track
    #[must_use]
    pub fn is_picture(&self) -> bool {
        matches!(self.track_type, TrackType::Picture)
    }

    /// Check if this is a sound track
    #[must_use]
    pub fn is_sound(&self) -> bool {
        matches!(self.track_type, TrackType::Sound)
    }

    /// Check if this is a timecode track
    #[must_use]
    pub fn is_timecode(&self) -> bool {
        matches!(self.track_type, TrackType::Timecode)
    }

    /// Set sequence
    pub fn set_sequence(&mut self, sequence: Sequence) {
        self.sequence = Some(sequence);
    }

    /// Get all source clips in this track
    #[must_use]
    pub fn source_clips(&self) -> Vec<&SourceClip> {
        if let Some(ref sequence) = self.sequence {
            sequence.source_clips()
        } else {
            Vec::new()
        }
    }
}

/// Track type
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TrackType {
    /// Picture/video track
    Picture,
    /// Sound/audio track
    Sound,
    /// Timecode track
    Timecode,
    /// Data track
    Data,
    /// Unknown track
    Unknown,
}

/// Sequence - collection of segments in a track
#[derive(Debug, Clone)]
pub struct Sequence {
    /// Components
    pub components: Vec<SequenceComponent>,
    /// Data definition
    pub data_definition: Auid,
}

impl Sequence {
    /// Create a new sequence
    #[must_use]
    pub fn new(data_definition: Auid) -> Self {
        Self {
            components: Vec::new(),
            data_definition,
        }
    }

    /// Add a component
    pub fn add_component(&mut self, component: SequenceComponent) {
        self.components.push(component);
    }

    /// Get duration
    #[must_use]
    pub fn duration(&self) -> Option<i64> {
        let mut total = 0i64;
        for component in &self.components {
            total += component.length()?;
        }
        Some(total)
    }

    /// Get all source clips
    #[must_use]
    pub fn source_clips(&self) -> Vec<&SourceClip> {
        let mut clips = Vec::new();
        for component in &self.components {
            if let SequenceComponent::SourceClip(clip) = component {
                clips.push(clip);
            }
        }
        clips
    }

    /// Convert to segment
    #[must_use]
    pub fn into_segment(self) -> SequenceSegment {
        let components = self
            .components
            .into_iter()
            .map(|c| c.into_component(self.data_definition))
            .collect();

        SequenceSegment {
            components,
            length: None,
        }
    }

    /// Check if this is a picture sequence
    #[must_use]
    pub fn is_picture(&self) -> bool {
        self.data_definition.is_picture()
    }

    /// Check if this is a sound sequence
    #[must_use]
    pub fn is_sound(&self) -> bool {
        self.data_definition.is_sound()
    }
}

/// Sequence component types
#[derive(Debug, Clone)]
pub enum SequenceComponent {
    /// Source clip
    SourceClip(SourceClip),
    /// Filler
    Filler(Filler),
    /// Transition
    Transition(Transition),
    /// Effect
    Effect(Effect),
}

impl SequenceComponent {
    /// Get length
    #[must_use]
    pub fn length(&self) -> Option<i64> {
        match self {
            SequenceComponent::SourceClip(clip) => Some(clip.length),
            SequenceComponent::Filler(filler) => Some(filler.length),
            SequenceComponent::Transition(transition) => Some(transition.length),
            SequenceComponent::Effect(effect) => effect.length,
        }
    }

    /// Convert to component
    #[must_use]
    pub fn into_component(self, data_definition: Auid) -> Component {
        let segment = match self {
            SequenceComponent::SourceClip(clip) => Segment::SourceClip(clip.into_segment()),
            SequenceComponent::Filler(filler) => Segment::Filler(filler.into_segment()),
            SequenceComponent::Transition(transition) => {
                Segment::Transition(transition.into_segment())
            }
            SequenceComponent::Effect(effect) => Segment::OperationGroup(effect.into_segment()),
        };

        Component::new(data_definition, segment)
    }
}

/// Source clip - reference to media
#[derive(Debug, Clone)]
pub struct SourceClip {
    /// Length
    pub length: i64,
    /// Start time in source
    pub start_time: Position,
    /// Source mob ID
    pub source_mob_id: Uuid,
    /// Source mob slot ID
    pub source_mob_slot_id: u32,
    /// Source track ID
    pub source_track_id: Option<u32>,
}

impl SourceClip {
    /// Create a new source clip
    #[must_use]
    pub fn new(
        length: i64,
        start_time: Position,
        source_mob_id: Uuid,
        source_mob_slot_id: u32,
    ) -> Self {
        Self {
            length,
            start_time,
            source_mob_id,
            source_mob_slot_id,
            source_track_id: None,
        }
    }

    /// Set source track ID
    #[must_use]
    pub fn with_source_track_id(mut self, track_id: u32) -> Self {
        self.source_track_id = Some(track_id);
        self
    }

    /// Convert to segment
    #[must_use]
    pub fn into_segment(self) -> SourceClipSegment {
        SourceClipSegment::new(
            self.length,
            self.start_time,
            self.source_mob_id,
            self.source_mob_slot_id,
        )
    }

    /// Get end time in source
    #[must_use]
    pub fn end_time(&self) -> Position {
        Position(self.start_time.0 + self.length)
    }
}

/// Filler - gap in the timeline
#[derive(Debug, Clone)]
pub struct Filler {
    /// Length
    pub length: i64,
}

impl Filler {
    /// Create a new filler
    #[must_use]
    pub fn new(length: i64) -> Self {
        Self { length }
    }

    /// Convert to segment
    #[must_use]
    pub fn into_segment(self) -> FillerSegment {
        FillerSegment::new(self.length)
    }
}

/// Transition - dissolve, wipe, etc.
#[derive(Debug, Clone)]
pub struct Transition {
    /// Length
    pub length: i64,
    /// Cut point
    pub cut_point: Position,
    /// Effect (optional)
    pub effect: Option<Effect>,
}

impl Transition {
    /// Create a new transition
    #[must_use]
    pub fn new(length: i64, cut_point: Position) -> Self {
        Self {
            length,
            cut_point,
            effect: None,
        }
    }

    /// Set effect
    #[must_use]
    pub fn with_effect(mut self, effect: Effect) -> Self {
        self.effect = Some(effect);
        self
    }

    /// Convert to segment
    #[must_use]
    pub fn into_segment(self) -> TransitionSegment {
        let effect = self.effect.map(|e| Box::new(e.into_segment()));

        TransitionSegment {
            length: self.length,
            cut_point: self.cut_point,
            effect,
        }
    }
}

/// Effect - operation applied to segments
#[derive(Debug, Clone)]
pub struct Effect {
    /// Operation ID
    pub operation_id: Auid,
    /// Input segments
    pub inputs: Vec<SequenceComponent>,
    /// Parameters
    pub parameters: HashMap<String, EffectParameter>,
    /// Length
    pub length: Option<i64>,
}

impl Effect {
    /// Create a new effect
    #[must_use]
    pub fn new(operation_id: Auid) -> Self {
        Self {
            operation_id,
            inputs: Vec::new(),
            parameters: HashMap::new(),
            length: None,
        }
    }

    /// Add input
    pub fn add_input(&mut self, input: SequenceComponent) {
        self.inputs.push(input);
    }

    /// Add parameter
    pub fn add_parameter(&mut self, name: impl Into<String>, parameter: EffectParameter) {
        self.parameters.insert(name.into(), parameter);
    }

    /// Set length
    pub fn set_length(&mut self, length: i64) {
        self.length = Some(length);
    }

    /// Convert to segment
    #[must_use]
    pub fn into_segment(self) -> OperationGroupSegment {
        OperationGroupSegment {
            operation_id: self.operation_id,
            input_segments: Vec::new(), // Would need conversion
            parameters: Vec::new(),     // Would need conversion
            length: self.length,
        }
    }
}

/// Effect parameter
#[derive(Debug, Clone)]
pub enum EffectParameter {
    /// Constant value
    Constant(f64),
    /// Varying value (keyframes)
    Varying(Vec<Keyframe>),
}

/// Keyframe for effect parameters
#[derive(Debug, Clone)]
pub struct Keyframe {
    /// Time position
    pub time: Position,
    /// Value
    pub value: f64,
    /// Interpolation
    pub interpolation: InterpolationType,
}

impl Keyframe {
    /// Create a new keyframe
    #[must_use]
    pub fn new(time: Position, value: f64, interpolation: InterpolationType) -> Self {
        Self {
            time,
            value,
            interpolation,
        }
    }
}

/// Interpolation type
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum InterpolationType {
    /// No interpolation (step)
    None,
    /// Linear interpolation
    Linear,
    /// Bezier interpolation
    Bezier,
    /// Cubic interpolation
    Cubic,
}

/// Helper to determine track type from segment
fn determine_track_type(segment: &Segment) -> TrackType {
    match segment {
        Segment::Sequence(seq) => {
            if let Some(component) = seq.components.first() {
                if component.is_picture() {
                    TrackType::Picture
                } else if component.is_sound() {
                    TrackType::Sound
                } else if component.is_timecode() {
                    TrackType::Timecode
                } else {
                    TrackType::Unknown
                }
            } else {
                TrackType::Unknown
            }
        }
        Segment::SourceClip(_) => TrackType::Unknown,
        Segment::Filler(_) => TrackType::Unknown,
        _ => TrackType::Unknown,
    }
}

/// Helper to extract sequence from segment
fn extract_sequence(segment: &Segment) -> Option<Sequence> {
    match segment {
        Segment::Sequence(seq) => {
            let data_def = seq
                .components
                .first()
                .map_or_else(Auid::null, |c| c.data_definition);

            let mut sequence = Sequence::new(data_def);

            for component in &seq.components {
                if let Some(seq_component) = convert_component_to_sequence_component(component) {
                    sequence.add_component(seq_component);
                }
            }

            Some(sequence)
        }
        _ => None,
    }
}

/// Helper to convert Component to `SequenceComponent`
fn convert_component_to_sequence_component(component: &Component) -> Option<SequenceComponent> {
    match &component.segment {
        Segment::SourceClip(clip) => Some(SequenceComponent::SourceClip(SourceClip {
            length: clip.length,
            start_time: clip.start_time,
            source_mob_id: clip.source_mob_id,
            source_mob_slot_id: clip.source_mob_slot_id,
            source_track_id: None,
        })),
        Segment::Filler(filler) => Some(SequenceComponent::Filler(Filler {
            length: filler.length,
        })),
        Segment::Transition(trans) => Some(SequenceComponent::Transition(Transition {
            length: trans.length,
            cut_point: trans.cut_point,
            effect: None,
        })),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_composition_mob_creation() {
        let mob_id = Uuid::new_v4();
        let comp = CompositionMob::new(mob_id, "Test Composition");
        assert_eq!(comp.mob_id(), mob_id);
        assert_eq!(comp.name(), "Test Composition");
    }

    #[test]
    fn test_track_creation() {
        let track = Track::new(1, "Video", EditRate::PAL_25, TrackType::Picture);
        assert_eq!(track.track_id, 1);
        assert_eq!(track.name, "Video");
        assert!(track.is_picture());
    }

    #[test]
    fn test_sequence_creation() {
        let seq = Sequence::new(Auid::PICTURE);
        assert!(seq.is_picture());
        assert_eq!(seq.duration(), Some(0));
    }

    #[test]
    fn test_sequence_with_clips() {
        let mut seq = Sequence::new(Auid::PICTURE);

        let clip1 = SourceClip::new(100, Position::zero(), Uuid::new_v4(), 1);
        let clip2 = SourceClip::new(50, Position::new(100), Uuid::new_v4(), 1);

        seq.add_component(SequenceComponent::SourceClip(clip1));
        seq.add_component(SequenceComponent::SourceClip(clip2));

        assert_eq!(seq.duration(), Some(150));
        assert_eq!(seq.source_clips().len(), 2);
    }

    #[test]
    fn test_source_clip() {
        let clip = SourceClip::new(100, Position::new(50), Uuid::new_v4(), 1);
        assert_eq!(clip.length, 100);
        assert_eq!(clip.start_time.0, 50);
        assert_eq!(clip.end_time().0, 150);
    }

    #[test]
    fn test_filler() {
        let filler = Filler::new(25);
        assert_eq!(filler.length, 25);
    }

    #[test]
    fn test_transition() {
        let trans = Transition::new(10, Position::new(5));
        assert_eq!(trans.length, 10);
        assert_eq!(trans.cut_point.0, 5);
    }

    #[test]
    fn test_effect() {
        let mut effect = Effect::new(Auid::null());
        effect.set_length(50);
        assert_eq!(effect.length, Some(50));
    }

    #[test]
    fn test_keyframe() {
        let kf = Keyframe::new(Position::new(10), 0.5, InterpolationType::Linear);
        assert_eq!(kf.time.0, 10);
        assert_eq!(kf.value, 0.5);
        assert_eq!(kf.interpolation, InterpolationType::Linear);
    }
}
