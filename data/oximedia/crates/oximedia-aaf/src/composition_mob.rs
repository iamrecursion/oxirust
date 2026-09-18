//! AAF composition mob objects
//!
//! Sequence composition, selector, and transition objects for the AAF
//! composition model (SMPTE ST 377-1 Section 13).

#![allow(dead_code)]
#![allow(clippy::cast_precision_loss)]

use std::collections::HashMap;

/// A sequence of components forming a track segment
#[derive(Debug, Clone)]
pub struct SequenceComposition {
    pub name: String,
    pub components: Vec<CompositionComponent>,
    pub data_def: String,
}

impl SequenceComposition {
    /// Create a new empty sequence
    #[must_use]
    pub fn new(name: impl Into<String>, data_def: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            components: Vec::new(),
            data_def: data_def.into(),
        }
    }

    /// Add a component to the sequence
    pub fn add_component(&mut self, component: CompositionComponent) {
        self.components.push(component);
    }

    /// Total duration of the sequence in edit units
    #[must_use]
    pub fn duration(&self) -> i64 {
        self.components.iter().map(|c| c.duration()).sum()
    }

    /// Number of components
    #[must_use]
    pub fn len(&self) -> usize {
        self.components.len()
    }

    /// Whether the sequence is empty
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.components.is_empty()
    }
}

/// A component within a sequence composition
#[derive(Debug, Clone)]
pub enum CompositionComponent {
    /// A source clip reference
    SourceClipRef(SourceClipRef),
    /// A filler (gap)
    Filler(FillerComponent),
    /// A transition between clips
    Transition(TransitionComponent),
    /// A nested sequence
    Nested(Box<SequenceComposition>),
}

impl CompositionComponent {
    /// Duration of this component in edit units
    #[must_use]
    pub fn duration(&self) -> i64 {
        match self {
            Self::SourceClipRef(s) => s.length,
            Self::Filler(f) => f.length,
            Self::Transition(t) => t.length,
            Self::Nested(n) => n.duration(),
        }
    }
}

/// Reference to a source clip in the composition
#[derive(Debug, Clone)]
pub struct SourceClipRef {
    pub source_id: String,
    pub source_track_id: u32,
    pub start_time: i64,
    pub length: i64,
}

impl SourceClipRef {
    /// Create a new source clip reference
    #[must_use]
    pub fn new(
        source_id: impl Into<String>,
        source_track_id: u32,
        start_time: i64,
        length: i64,
    ) -> Self {
        Self {
            source_id: source_id.into(),
            source_track_id,
            start_time,
            length,
        }
    }
}

/// A filler component (silence/black)
#[derive(Debug, Clone)]
pub struct FillerComponent {
    pub length: i64,
    pub data_def: String,
}

impl FillerComponent {
    /// Create a new filler
    #[must_use]
    pub fn new(length: i64, data_def: impl Into<String>) -> Self {
        Self {
            length,
            data_def: data_def.into(),
        }
    }
}

/// A transition between two adjacent clips
#[derive(Debug, Clone)]
pub struct TransitionComponent {
    pub length: i64,
    pub cut_point: i64,
    pub effect_name: String,
}

impl TransitionComponent {
    /// Create a new transition
    #[must_use]
    pub fn new(length: i64, cut_point: i64, effect_name: impl Into<String>) -> Self {
        Self {
            length,
            cut_point,
            effect_name: effect_name.into(),
        }
    }
}

/// A selector that chooses between alternative segments at runtime
#[derive(Debug, Clone)]
pub struct Selector {
    pub selected: usize,
    pub alternatives: Vec<SequenceComposition>,
}

impl Selector {
    /// Create a new selector with a default selected index
    #[must_use]
    pub fn new(selected: usize) -> Self {
        Self {
            selected,
            alternatives: Vec::new(),
        }
    }

    /// Add an alternative
    pub fn add_alternative(&mut self, seq: SequenceComposition) {
        self.alternatives.push(seq);
    }

    /// Get the currently selected sequence
    #[must_use]
    pub fn selected_sequence(&self) -> Option<&SequenceComposition> {
        self.alternatives.get(self.selected)
    }

    /// Duration of selected alternative
    #[must_use]
    pub fn duration(&self) -> i64 {
        self.selected_sequence().map_or(0, |s| s.duration())
    }
}

/// A composition mob track with slot info
#[derive(Debug, Clone)]
pub struct CompositionTrack {
    pub slot_id: u32,
    pub name: String,
    pub segment: SequenceComposition,
    pub attributes: HashMap<String, String>,
}

impl CompositionTrack {
    /// Create a new composition track
    #[must_use]
    pub fn new(slot_id: u32, name: impl Into<String>, segment: SequenceComposition) -> Self {
        Self {
            slot_id,
            name: name.into(),
            segment,
            attributes: HashMap::new(),
        }
    }

    /// Set an attribute
    pub fn set_attribute(&mut self, key: impl Into<String>, value: impl Into<String>) {
        self.attributes.insert(key.into(), value.into());
    }

    /// Get an attribute
    #[must_use]
    pub fn get_attribute(&self, key: &str) -> Option<&str> {
        self.attributes.get(key).map(String::as_str)
    }
}

/// A full composition mob containing multiple tracks
#[derive(Debug, Clone)]
pub struct CompositionMobDef {
    pub mob_id: String,
    pub name: String,
    pub tracks: Vec<CompositionTrack>,
    pub user_comments: HashMap<String, String>,
}

impl CompositionMobDef {
    /// Create a new composition mob definition
    #[must_use]
    pub fn new(mob_id: impl Into<String>, name: impl Into<String>) -> Self {
        Self {
            mob_id: mob_id.into(),
            name: name.into(),
            tracks: Vec::new(),
            user_comments: HashMap::new(),
        }
    }

    /// Add a track
    pub fn add_track(&mut self, track: CompositionTrack) {
        self.tracks.push(track);
    }

    /// Find a track by slot ID
    #[must_use]
    pub fn find_track(&self, slot_id: u32) -> Option<&CompositionTrack> {
        self.tracks.iter().find(|t| t.slot_id == slot_id)
    }

    /// Add a user comment
    pub fn add_comment(&mut self, key: impl Into<String>, value: impl Into<String>) {
        self.user_comments.insert(key.into(), value.into());
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sequence_composition_new() {
        let seq = SequenceComposition::new("MySeq", "Picture");
        assert_eq!(seq.name, "MySeq");
        assert_eq!(seq.data_def, "Picture");
        assert!(seq.is_empty());
        assert_eq!(seq.duration(), 0);
    }

    #[test]
    fn test_sequence_add_source_clip() {
        let mut seq = SequenceComposition::new("Test", "Picture");
        let clip = SourceClipRef::new("mob-001", 1, 0, 100);
        seq.add_component(CompositionComponent::SourceClipRef(clip));
        assert_eq!(seq.len(), 1);
        assert_eq!(seq.duration(), 100);
    }

    #[test]
    fn test_sequence_duration_sum() {
        let mut seq = SequenceComposition::new("Test", "Picture");
        seq.add_component(CompositionComponent::SourceClipRef(SourceClipRef::new(
            "a", 1, 0, 50,
        )));
        seq.add_component(CompositionComponent::Filler(FillerComponent::new(
            25, "Picture",
        )));
        seq.add_component(CompositionComponent::SourceClipRef(SourceClipRef::new(
            "b", 1, 0, 75,
        )));
        assert_eq!(seq.duration(), 150);
    }

    #[test]
    fn test_filler_component() {
        let f = FillerComponent::new(30, "Sound");
        assert_eq!(f.length, 30);
        assert_eq!(f.data_def, "Sound");
    }

    #[test]
    fn test_transition_component() {
        let t = TransitionComponent::new(20, 10, "Dissolve");
        assert_eq!(t.length, 20);
        assert_eq!(t.cut_point, 10);
        assert_eq!(t.effect_name, "Dissolve");
    }

    #[test]
    fn test_selector_basic() {
        let mut sel = Selector::new(0);
        let mut seq1 = SequenceComposition::new("Alt1", "Picture");
        seq1.add_component(CompositionComponent::SourceClipRef(SourceClipRef::new(
            "x", 1, 0, 100,
        )));
        let seq2 = SequenceComposition::new("Alt2", "Picture");
        sel.add_alternative(seq1);
        sel.add_alternative(seq2);
        assert_eq!(sel.selected, 0);
        assert_eq!(sel.duration(), 100);
    }

    #[test]
    fn test_selector_no_alternatives() {
        let sel = Selector::new(0);
        assert!(sel.selected_sequence().is_none());
        assert_eq!(sel.duration(), 0);
    }

    #[test]
    fn test_composition_track_attributes() {
        let seq = SequenceComposition::new("S", "Picture");
        let mut track = CompositionTrack::new(1, "Video", seq);
        track.set_attribute("color", "red");
        assert_eq!(track.get_attribute("color"), Some("red"));
        assert!(track.get_attribute("missing").is_none());
    }

    #[test]
    fn test_composition_mob_def_add_track() {
        let mut mob = CompositionMobDef::new("mob-xyz", "Scene1");
        let seq = SequenceComposition::new("S", "Picture");
        let track = CompositionTrack::new(1, "V1", seq);
        mob.add_track(track);
        assert_eq!(mob.tracks.len(), 1);
        assert!(mob.find_track(1).is_some());
        assert!(mob.find_track(99).is_none());
    }

    #[test]
    fn test_composition_mob_def_comments() {
        let mut mob = CompositionMobDef::new("mob-1", "Scene");
        mob.add_comment("Director", "Alice");
        assert_eq!(
            mob.user_comments.get("Director").map(String::as_str),
            Some("Alice")
        );
    }

    #[test]
    fn test_nested_sequence_duration() {
        let mut inner = SequenceComposition::new("Inner", "Picture");
        inner.add_component(CompositionComponent::SourceClipRef(SourceClipRef::new(
            "c", 1, 0, 40,
        )));
        let mut outer = SequenceComposition::new("Outer", "Picture");
        outer.add_component(CompositionComponent::Nested(Box::new(inner)));
        outer.add_component(CompositionComponent::Filler(FillerComponent::new(
            10, "Picture",
        )));
        assert_eq!(outer.duration(), 50);
    }

    #[test]
    fn test_source_clip_ref_fields() {
        let clip = SourceClipRef::new("mob-007", 3, 48000, 96000);
        assert_eq!(clip.source_id, "mob-007");
        assert_eq!(clip.source_track_id, 3);
        assert_eq!(clip.start_time, 48000);
        assert_eq!(clip.length, 96000);
    }

    #[test]
    fn test_component_duration_dispatch() {
        let s = CompositionComponent::SourceClipRef(SourceClipRef::new("a", 1, 0, 100));
        let f = CompositionComponent::Filler(FillerComponent::new(50, "Picture"));
        let t = CompositionComponent::Transition(TransitionComponent::new(10, 5, "Wipe"));
        assert_eq!(s.duration(), 100);
        assert_eq!(f.duration(), 50);
        assert_eq!(t.duration(), 10);
    }
}
