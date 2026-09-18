//! Multi-camera EDL support.
//!
//! This module provides structures and utilities for working with multi-camera
//! edit decision lists, enabling synchronisation and switching between multiple
//! camera angles in a timeline.

use crate::error::EdlResult;
use crate::event::EdlEvent;
use crate::timecode::{EdlFrameRate, EdlTimecode};

/// Identifier for a camera angle in a multi-camera group.
#[derive(Debug, Clone, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
pub struct CameraAngleId(pub String);

impl CameraAngleId {
    /// Create a new camera angle identifier.
    #[must_use]
    pub fn new(id: impl Into<String>) -> Self {
        Self(id.into())
    }

    /// Return the identifier as a string slice.
    #[must_use]
    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl std::fmt::Display for CameraAngleId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0)
    }
}

/// A single camera angle entry within a multi-camera group.
#[derive(Debug, Clone)]
pub struct CameraAngle {
    /// Unique identifier for this angle (e.g. "CAM_A", "CAM_B").
    pub id: CameraAngleId,

    /// Reel name or source identifier for the footage from this angle.
    pub reel: String,

    /// Optional human-readable label.
    pub label: Option<String>,
}

impl CameraAngle {
    /// Create a new camera angle.
    #[must_use]
    pub fn new(id: impl Into<String>, reel: impl Into<String>) -> Self {
        Self {
            id: CameraAngleId::new(id),
            reel: reel.into(),
            label: None,
        }
    }

    /// Attach a human-readable label.
    #[must_use]
    pub fn with_label(mut self, label: impl Into<String>) -> Self {
        self.label = Some(label.into());
        self
    }
}

/// A group of synchronised camera angles that can be switched between.
#[derive(Debug, Clone)]
pub struct MulticamGroup {
    /// Group name.
    pub name: String,

    /// All camera angles in this group.
    pub angles: Vec<CameraAngle>,

    /// Synchronisation timecode (the "sync point" common to all angles).
    pub sync_timecode: Option<EdlTimecode>,
}

impl MulticamGroup {
    /// Create a new multi-camera group.
    #[must_use]
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            angles: Vec::new(),
            sync_timecode: None,
        }
    }

    /// Add a camera angle to this group.
    pub fn add_angle(&mut self, angle: CameraAngle) {
        self.angles.push(angle);
    }

    /// Set the synchronisation timecode.
    pub fn set_sync_timecode(&mut self, tc: EdlTimecode) {
        self.sync_timecode = Some(tc);
    }

    /// Look up an angle by its identifier.
    #[must_use]
    pub fn find_angle(&self, id: &CameraAngleId) -> Option<&CameraAngle> {
        self.angles.iter().find(|a| &a.id == id)
    }
}

/// A multi-camera cut decision: at what timecode to switch to which angle.
#[derive(Debug, Clone)]
pub struct MulticamCut {
    /// The record timecode at which the cut occurs.
    pub record_in: EdlTimecode,

    /// The angle to switch to.
    pub angle_id: CameraAngleId,
}

impl MulticamCut {
    /// Create a new multi-camera cut.
    #[must_use]
    pub fn new(record_in: EdlTimecode, angle_id: CameraAngleId) -> Self {
        Self {
            record_in,
            angle_id,
        }
    }
}

/// An EDL sequence that represents a multi-camera edit.
///
/// The sequence holds a [`MulticamGroup`] (defining the available angles) and
/// an ordered list of [`MulticamCut`]s describing when to switch angles.
#[derive(Debug, Clone)]
pub struct MulticamSequence {
    /// The camera group whose angles are used in this sequence.
    pub group: MulticamGroup,

    /// Ordered list of cut decisions (sorted by `record_in`).
    pub cuts: Vec<MulticamCut>,
}

impl MulticamSequence {
    /// Create a new multi-camera sequence from a group.
    #[must_use]
    pub fn new(group: MulticamGroup) -> Self {
        Self {
            group,
            cuts: Vec::new(),
        }
    }

    /// Add a cut decision. The list is kept sorted by `record_in`.
    pub fn add_cut(&mut self, cut: MulticamCut) {
        self.cuts.push(cut);
        self.cuts.sort_by_key(|c| c.record_in.to_frames());
    }

    /// Determine the active angle at a given timecode (in frames).
    #[must_use]
    pub fn active_angle_at(&self, frame: u64) -> Option<&CameraAngleId> {
        // The last cut whose record_in ≤ frame is the active one.
        self.cuts
            .iter()
            .rev()
            .find(|c| c.record_in.to_frames() <= frame)
            .map(|c| &c.angle_id)
    }

    /// Flatten the multi-camera sequence into a list of single-angle EDL events.
    ///
    /// Each resulting event covers the span between consecutive cuts (or to the
    /// end of the last event when provided). The `fps` parameter specifies the
    /// frame rate to use for the output timecodes.
    ///
    /// # Errors
    ///
    /// Returns an error if the sequence cannot be resolved (e.g. an angle id
    /// does not exist in the group, or timecodes cannot be constructed).
    pub fn flatten(&self, end_frame: u64, fps: EdlFrameRate) -> EdlResult<Vec<EdlEvent>> {
        use crate::audio::AudioChannel;
        use crate::event::{EditType, TrackType};

        if self.cuts.is_empty() {
            return Ok(Vec::new());
        }

        let mut events: Vec<EdlEvent> = Vec::new();
        let mut event_number: u32 = 1;

        for (i, cut) in self.cuts.iter().enumerate() {
            let rec_in = cut.record_in;
            let rec_out_frame = self
                .cuts
                .get(i + 1)
                .map(|next| next.record_in.to_frames())
                .unwrap_or(end_frame);

            // Skip zero-length spans.
            if rec_out_frame <= rec_in.to_frames() {
                continue;
            }

            let angle = self
                .group
                .find_angle(&cut.angle_id)
                .ok_or_else(|| crate::error::EdlError::InvalidTrackType(cut.angle_id.0.clone()))?;

            let rec_out = EdlTimecode::from_frames(rec_out_frame, fps)?;

            // Source timecode mirrors the record timecode for a simple flatten.
            let src_in = rec_in;
            let src_out = rec_out;

            let event = EdlEvent::new(
                event_number,
                angle.reel.clone(),
                TrackType::Audio(AudioChannel::A1),
                EditType::Cut,
                src_in,
                src_out,
                rec_in,
                rec_out,
            );

            events.push(event);
            event_number += 1;
        }

        Ok(events)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::timecode::{EdlFrameRate, EdlTimecode};

    fn tc(h: u8, m: u8, s: u8, f: u8) -> EdlTimecode {
        EdlTimecode::new(h, m, s, f, EdlFrameRate::Fps25).expect("valid timecode")
    }

    #[test]
    fn test_camera_angle_id_display() {
        let id = CameraAngleId::new("CAM_A");
        assert_eq!(id.to_string(), "CAM_A");
    }

    #[test]
    fn test_multicam_group_find_angle() {
        let mut group = MulticamGroup::new("Scene1");
        group.add_angle(CameraAngle::new("CAM_A", "REEL_A"));
        group.add_angle(CameraAngle::new("CAM_B", "REEL_B"));

        let id = CameraAngleId::new("CAM_A");
        let found = group.find_angle(&id).expect("should find angle");
        assert_eq!(found.reel, "REEL_A");

        let missing = CameraAngleId::new("CAM_Z");
        assert!(group.find_angle(&missing).is_none());
    }

    #[test]
    fn test_multicam_sequence_active_angle() {
        let mut group = MulticamGroup::new("Scene1");
        group.add_angle(CameraAngle::new("CAM_A", "REEL_A"));
        group.add_angle(CameraAngle::new("CAM_B", "REEL_B"));

        let mut seq = MulticamSequence::new(group);
        seq.add_cut(MulticamCut::new(
            tc(1, 0, 0, 0),
            CameraAngleId::new("CAM_A"),
        ));
        seq.add_cut(MulticamCut::new(
            tc(1, 0, 5, 0),
            CameraAngleId::new("CAM_B"),
        ));

        let frame_in_a = tc(1, 0, 2, 0).to_frames();
        let frame_in_b = tc(1, 0, 7, 0).to_frames();

        assert_eq!(
            seq.active_angle_at(frame_in_a).map(|a| a.as_str()),
            Some("CAM_A")
        );
        assert_eq!(
            seq.active_angle_at(frame_in_b).map(|a| a.as_str()),
            Some("CAM_B")
        );
    }

    #[test]
    fn test_multicam_sequence_flatten() {
        let mut group = MulticamGroup::new("Scene1");
        group.add_angle(CameraAngle::new("CAM_A", "REEL_A"));
        group.add_angle(CameraAngle::new("CAM_B", "REEL_B"));

        let mut seq = MulticamSequence::new(group);
        seq.add_cut(MulticamCut::new(
            tc(1, 0, 0, 0),
            CameraAngleId::new("CAM_A"),
        ));
        seq.add_cut(MulticamCut::new(
            tc(1, 0, 5, 0),
            CameraAngleId::new("CAM_B"),
        ));

        let end = tc(1, 0, 10, 0).to_frames();
        let events = seq
            .flatten(end, EdlFrameRate::Fps25)
            .expect("flatten should succeed");

        assert_eq!(events.len(), 2);
        assert_eq!(events[0].reel, "REEL_A");
        assert_eq!(events[1].reel, "REEL_B");
    }
}
