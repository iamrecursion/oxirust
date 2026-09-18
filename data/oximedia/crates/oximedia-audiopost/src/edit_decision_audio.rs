#![allow(dead_code)]
//! Audio Edit Decision List (EDL) types for audio post-production.
//!
//! Models individual [`AudioEdit`] points, an [`AudioEditList`] of edits on a
//! single track, and a multi-track [`AudioEDL`] that aggregates them.

// ---------------------------------------------------------------------------
// EditType
// ---------------------------------------------------------------------------

/// The transition type at an edit point.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum EditType {
    /// Hard cut with no transition.
    Cut,
    /// Gradual fade from one clip to the next.
    Crossfade,
    /// Fade-in at the start of a clip from silence.
    FadeIn,
    /// Fade-out at the end of a clip to silence.
    FadeOut,
    /// Dissolve (same as crossfade but semantically for music transitions).
    Dissolve,
}

impl EditType {
    /// Human-readable label.
    #[must_use]
    pub fn label(self) -> &'static str {
        match self {
            Self::Cut => "cut",
            Self::Crossfade => "crossfade",
            Self::FadeIn => "fade_in",
            Self::FadeOut => "fade_out",
            Self::Dissolve => "dissolve",
        }
    }
}

// ---------------------------------------------------------------------------
// AudioEdit
// ---------------------------------------------------------------------------

/// A single edit decision on an audio track.
#[derive(Debug, Clone)]
pub struct AudioEdit {
    /// Edit point position in frames (relative to sequence start).
    pub position_frames: u64,
    /// Source clip identifier.
    pub clip_id: String,
    /// Source in-point in frames.
    pub source_in_frames: u64,
    /// Source out-point in frames.
    pub source_out_frames: u64,
    /// Transition type at this edit point.
    pub edit_type: EditType,
    /// Duration of the crossfade / fade in frames (0 for hard cuts).
    pub transition_frames: u32,
}

impl AudioEdit {
    /// Create a new audio edit.
    #[must_use]
    pub fn new(
        position_frames: u64,
        clip_id: impl Into<String>,
        source_in_frames: u64,
        source_out_frames: u64,
        edit_type: EditType,
        transition_frames: u32,
    ) -> Self {
        Self {
            position_frames,
            clip_id: clip_id.into(),
            source_in_frames,
            source_out_frames,
            edit_type,
            transition_frames,
        }
    }

    /// Returns `true` if the edit type involves blending between two clips
    /// (crossfade or dissolve).
    #[must_use]
    pub fn is_crossfade(&self) -> bool {
        matches!(self.edit_type, EditType::Crossfade | EditType::Dissolve)
    }

    /// Duration of the edited clip in frames.
    #[must_use]
    pub fn clip_duration_frames(&self) -> u64 {
        self.source_out_frames.saturating_sub(self.source_in_frames)
    }
}

// ---------------------------------------------------------------------------
// AudioEditList
// ---------------------------------------------------------------------------

/// An ordered list of [`AudioEdit`]s on a single audio track.
#[derive(Debug, Clone, Default)]
pub struct AudioEditList {
    track_name: String,
    edits: Vec<AudioEdit>,
}

impl AudioEditList {
    /// Create an empty edit list for `track_name`.
    #[must_use]
    pub fn new(track_name: impl Into<String>) -> Self {
        Self {
            track_name: track_name.into(),
            edits: Vec::new(),
        }
    }

    /// Track this edit list belongs to.
    #[must_use]
    pub fn track_name(&self) -> &str {
        &self.track_name
    }

    /// Append an edit (kept in insertion order; callers should insert in
    /// timeline order for correct `total_duration_frames` results).
    pub fn add(&mut self, edit: AudioEdit) {
        self.edits.push(edit);
    }

    /// Total number of edits.
    #[must_use]
    pub fn total_edits(&self) -> usize {
        self.edits.len()
    }

    /// Collect references to all crossfade/dissolve edits.
    #[must_use]
    pub fn crossfades(&self) -> Vec<&AudioEdit> {
        self.edits.iter().filter(|e| e.is_crossfade()).collect()
    }

    /// Iterate over all edits in order.
    pub fn iter(&self) -> impl Iterator<Item = &AudioEdit> {
        self.edits.iter()
    }

    /// Approximate total duration in frames: last edit's position plus its
    /// clip duration.  Returns 0 for an empty list.
    #[must_use]
    pub fn total_duration_frames(&self) -> u64 {
        self.edits
            .iter()
            .map(|e| e.position_frames + e.clip_duration_frames())
            .max()
            .unwrap_or(0)
    }
}

// ---------------------------------------------------------------------------
// AudioEDL
// ---------------------------------------------------------------------------

/// A multi-track Audio Edit Decision List.
#[derive(Debug, Clone, Default)]
pub struct AudioEDL {
    title: String,
    frame_rate: f32,
    tracks: Vec<AudioEditList>,
}

impl AudioEDL {
    /// Create an empty EDL.
    #[must_use]
    pub fn new(title: impl Into<String>, frame_rate: f32) -> Self {
        Self {
            title: title.into(),
            frame_rate,
            tracks: Vec::new(),
        }
    }

    /// EDL title.
    #[must_use]
    pub fn title(&self) -> &str {
        &self.title
    }

    /// Frame rate (e.g. 24.0, 25.0, 29.97, 30.0).
    #[must_use]
    pub fn frame_rate(&self) -> f32 {
        self.frame_rate
    }

    /// Add a track edit list.
    pub fn add_track(&mut self, track: AudioEditList) {
        self.tracks.push(track);
    }

    /// Number of tracks in the EDL.
    #[must_use]
    pub fn track_count(&self) -> usize {
        self.tracks.len()
    }

    /// Total duration in frames: the maximum `total_duration_frames` across
    /// all tracks.
    #[must_use]
    pub fn total_duration_frames(&self) -> u64 {
        self.tracks
            .iter()
            .map(|t| t.total_duration_frames())
            .max()
            .unwrap_or(0)
    }

    /// Iterate over track edit lists.
    pub fn tracks(&self) -> impl Iterator<Item = &AudioEditList> {
        self.tracks.iter()
    }

    /// Total number of crossfade edits across all tracks.
    #[must_use]
    pub fn total_crossfades(&self) -> usize {
        self.tracks.iter().map(|t| t.crossfades().len()).sum()
    }
}

// ---------------------------------------------------------------------------
// Unit tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn make_cut_edit(pos: u64) -> AudioEdit {
        AudioEdit::new(pos, "clip_a", 0, 240, EditType::Cut, 0)
    }

    fn make_xfade_edit(pos: u64) -> AudioEdit {
        AudioEdit::new(pos, "clip_b", 0, 480, EditType::Crossfade, 48)
    }

    #[test]
    fn test_edit_type_labels() {
        assert_eq!(EditType::Cut.label(), "cut");
        assert_eq!(EditType::Crossfade.label(), "crossfade");
        assert_eq!(EditType::FadeIn.label(), "fade_in");
        assert_eq!(EditType::FadeOut.label(), "fade_out");
        assert_eq!(EditType::Dissolve.label(), "dissolve");
    }

    #[test]
    fn test_audio_edit_is_crossfade_true() {
        let e = make_xfade_edit(100);
        assert!(e.is_crossfade());
    }

    #[test]
    fn test_audio_edit_is_crossfade_false_for_cut() {
        let e = make_cut_edit(0);
        assert!(!e.is_crossfade());
    }

    #[test]
    fn test_audio_edit_dissolve_is_crossfade() {
        let e = AudioEdit::new(0, "c", 0, 100, EditType::Dissolve, 24);
        assert!(e.is_crossfade());
    }

    #[test]
    fn test_audio_edit_clip_duration() {
        let e = AudioEdit::new(0, "c", 100, 340, EditType::Cut, 0);
        assert_eq!(e.clip_duration_frames(), 240);
    }

    #[test]
    fn test_audio_edit_clip_duration_saturating() {
        // out < in should not panic
        let e = AudioEdit::new(0, "c", 500, 100, EditType::Cut, 0);
        assert_eq!(e.clip_duration_frames(), 0);
    }

    #[test]
    fn test_edit_list_add_and_total_edits() {
        let mut list = AudioEditList::new("Dialogue");
        assert_eq!(list.total_edits(), 0);
        list.add(make_cut_edit(0));
        list.add(make_xfade_edit(240));
        assert_eq!(list.total_edits(), 2);
    }

    #[test]
    fn test_edit_list_crossfades() {
        let mut list = AudioEditList::new("DX");
        list.add(make_cut_edit(0));
        list.add(make_xfade_edit(240));
        list.add(make_cut_edit(720));
        let xf = list.crossfades();
        assert_eq!(xf.len(), 1);
    }

    #[test]
    fn test_edit_list_total_duration() {
        let mut list = AudioEditList::new("MX");
        // position 0, clip duration 240 → ends at 240
        list.add(make_cut_edit(0));
        // position 240, clip duration 480 → ends at 720
        list.add(make_xfade_edit(240));
        assert_eq!(list.total_duration_frames(), 720);
    }

    #[test]
    fn test_edit_list_total_duration_empty() {
        let list = AudioEditList::new("Empty");
        assert_eq!(list.total_duration_frames(), 0);
    }

    #[test]
    fn test_edit_list_track_name() {
        let list = AudioEditList::new("FX Bed");
        assert_eq!(list.track_name(), "FX Bed");
    }

    #[test]
    fn test_edl_track_count() {
        let mut edl = AudioEDL::new("My Mix", 24.0);
        assert_eq!(edl.track_count(), 0);
        edl.add_track(AudioEditList::new("DX"));
        edl.add_track(AudioEditList::new("MX"));
        assert_eq!(edl.track_count(), 2);
    }

    #[test]
    fn test_edl_total_duration_frames() {
        let mut edl = AudioEDL::new("Film", 24.0);
        let mut t1 = AudioEditList::new("DX");
        t1.add(make_cut_edit(0)); // ends at 240
        let mut t2 = AudioEditList::new("MX");
        t2.add(make_xfade_edit(240)); // ends at 720
        edl.add_track(t1);
        edl.add_track(t2);
        assert_eq!(edl.total_duration_frames(), 720);
    }

    #[test]
    fn test_edl_total_crossfades() {
        let mut edl = AudioEDL::new("Film", 24.0);
        let mut t1 = AudioEditList::new("DX");
        t1.add(make_xfade_edit(0));
        t1.add(make_cut_edit(480));
        let mut t2 = AudioEditList::new("MX");
        t2.add(make_xfade_edit(0));
        edl.add_track(t1);
        edl.add_track(t2);
        assert_eq!(edl.total_crossfades(), 2);
    }

    #[test]
    fn test_edl_title_and_frame_rate() {
        let edl = AudioEDL::new("Promo Cut", 29.97);
        assert_eq!(edl.title(), "Promo Cut");
        assert!((edl.frame_rate() - 29.97).abs() < 0.001);
    }
}
