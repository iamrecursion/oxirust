//! Python-exposed timeline types.
//!
//! Provides plain Rust structs representing clips, tracks, and timelines.
//! These can be wrapped with PyO3 annotations later if needed.

#![allow(dead_code)]

// ─────────────────────────────────────────────────────────────
//  PyTrackType
// ─────────────────────────────────────────────────────────────

/// The kind of media a track carries.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PyTrackType {
    /// Video track.
    Video,
    /// Audio track.
    Audio,
    /// Subtitle / caption track.
    Subtitle,
    /// Effect / graphics track.
    Effect,
}

// ─────────────────────────────────────────────────────────────
//  PyClip
// ─────────────────────────────────────────────────────────────

/// A single media clip placed on the timeline.
#[derive(Debug, Clone)]
pub struct PyClip {
    /// Unique clip identifier.
    pub id: u64,
    /// Absolute path (or URI) to the source media file.
    pub source_path: String,
    /// In-point within the source file, in milliseconds.
    pub start_ms: u64,
    /// Out-point within the source file, in milliseconds.
    pub end_ms: u64,
    /// Record-in position on the timeline, in milliseconds.
    pub record_in_ms: u64,
    /// Record-out position on the timeline, in milliseconds.
    pub record_out_ms: u64,
    /// Zero-based track index this clip is placed on.
    pub track: u32,
}

impl PyClip {
    /// Construct a new `PyClip`.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        id: u64,
        source_path: &str,
        start_ms: u64,
        end_ms: u64,
        record_in_ms: u64,
        record_out_ms: u64,
        track: u32,
    ) -> Self {
        Self {
            id,
            source_path: source_path.to_string(),
            start_ms,
            end_ms,
            record_in_ms,
            record_out_ms,
            track,
        }
    }

    /// Duration of the clip on the timeline in milliseconds.
    pub fn duration_ms(&self) -> u64 {
        self.record_out_ms.saturating_sub(self.record_in_ms)
    }

    /// Returns `true` if this clip overlaps `other` on the same track.
    ///
    /// Two clips overlap when their timeline ranges intersect and they share
    /// the same track.
    pub fn overlaps(&self, other: &PyClip) -> bool {
        if self.track != other.track {
            return false;
        }
        // Overlap when neither clip ends before the other starts.
        self.record_in_ms < other.record_out_ms && other.record_in_ms < self.record_out_ms
    }

    /// Returns `true` if this clip is placed on `track`.
    pub fn is_on_track(&self, track: u32) -> bool {
        self.track == track
    }
}

// ─────────────────────────────────────────────────────────────
//  PyTrack
// ─────────────────────────────────────────────────────────────

/// A single timeline track that holds ordered clips.
#[derive(Debug, Clone)]
pub struct PyTrack {
    /// Unique track identifier.
    pub id: u32,
    /// Human-readable track name.
    pub name: String,
    /// What kind of media this track carries.
    pub track_type: PyTrackType,
    /// Clips placed on this track.
    pub clips: Vec<PyClip>,
}

impl PyTrack {
    /// Construct a new empty track.
    pub fn new(id: u32, name: &str, track_type: PyTrackType) -> Self {
        Self {
            id,
            name: name.to_string(),
            track_type,
            clips: Vec::new(),
        }
    }

    /// Append a clip to the track.
    pub fn add_clip(&mut self, clip: PyClip) {
        self.clips.push(clip);
    }

    /// Remove the clip with the given `id`.
    ///
    /// Returns `true` if a clip was removed.
    pub fn remove_clip(&mut self, id: u64) -> bool {
        let before = self.clips.len();
        self.clips.retain(|c| c.id != id);
        self.clips.len() < before
    }

    /// Total track duration = record-out of the last clip (by record-out time).
    ///
    /// Returns `0` when the track is empty.
    pub fn duration_ms(&self) -> u64 {
        self.clips
            .iter()
            .map(|c| c.record_out_ms)
            .max()
            .unwrap_or(0)
    }

    /// Find a clip by its unique `id`, or `None`.
    pub fn find_clip(&self, id: u64) -> Option<&PyClip> {
        self.clips.iter().find(|c| c.id == id)
    }

    /// Return all clips active at `time_ms` on the timeline.
    ///
    /// A clip is considered active when `record_in_ms <= time_ms < record_out_ms`.
    pub fn clips_at_time(&self, time_ms: u64) -> Vec<&PyClip> {
        self.clips
            .iter()
            .filter(|c| c.record_in_ms <= time_ms && time_ms < c.record_out_ms)
            .collect()
    }
}

// ─────────────────────────────────────────────────────────────
//  PyTimeline
// ─────────────────────────────────────────────────────────────

/// A complete timeline composed of multiple tracks.
#[derive(Debug, Clone)]
pub struct PyTimeline {
    /// All tracks in this timeline.
    pub tracks: Vec<PyTrack>,
    /// Nominal frame rate for this timeline.
    pub frame_rate: f64,
}

impl PyTimeline {
    /// Construct a new empty timeline with the given frame rate.
    pub fn new(fps: f64) -> Self {
        Self {
            tracks: Vec::new(),
            frame_rate: fps,
        }
    }

    /// Append a track to the timeline.
    pub fn add_track(&mut self, t: PyTrack) {
        self.tracks.push(t);
    }

    /// Overall duration = maximum `duration_ms` across all tracks.
    pub fn total_duration_ms(&self) -> u64 {
        self.tracks
            .iter()
            .map(|t| t.duration_ms())
            .max()
            .unwrap_or(0)
    }

    /// Number of tracks in the timeline.
    pub fn track_count(&self) -> usize {
        self.tracks.len()
    }
}

// ─────────────────────────────────────────────────────────────
//  Unit tests
// ─────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn make_clip(id: u64, rin: u64, rout: u64, track: u32) -> PyClip {
        PyClip::new(id, "source.mp4", 0, rout - rin, rin, rout, track)
    }

    // ── PyClip ───────────────────────────────────────────────

    #[test]
    fn test_clip_duration_ms() {
        let clip = make_clip(1, 1000, 5000, 0);
        assert_eq!(clip.duration_ms(), 4000);
    }

    #[test]
    fn test_clip_is_on_track() {
        let clip = make_clip(1, 0, 1000, 2);
        assert!(clip.is_on_track(2));
        assert!(!clip.is_on_track(3));
    }

    #[test]
    fn test_clip_overlaps_same_track() {
        let a = make_clip(1, 0, 3000, 0);
        let b = make_clip(2, 2000, 5000, 0);
        assert!(a.overlaps(&b));
        assert!(b.overlaps(&a));
    }

    #[test]
    fn test_clip_no_overlap_adjacent() {
        // End of a == start of b: touching but not overlapping.
        let a = make_clip(1, 0, 2000, 0);
        let b = make_clip(2, 2000, 4000, 0);
        assert!(!a.overlaps(&b));
    }

    #[test]
    fn test_clip_no_overlap_different_track() {
        let a = make_clip(1, 0, 3000, 0);
        let b = make_clip(2, 0, 3000, 1);
        assert!(!a.overlaps(&b));
    }

    // ── PyTrack ──────────────────────────────────────────────

    #[test]
    fn test_track_add_and_find_clip() {
        let mut track = PyTrack::new(0, "Video 1", PyTrackType::Video);
        track.add_clip(make_clip(10, 0, 1000, 0));
        track.add_clip(make_clip(20, 2000, 4000, 0));
        assert_eq!(track.clips.len(), 2);
        assert!(track.find_clip(10).is_some());
        assert!(track.find_clip(99).is_none());
    }

    #[test]
    fn test_track_remove_clip() {
        let mut track = PyTrack::new(0, "Video 1", PyTrackType::Video);
        track.add_clip(make_clip(1, 0, 1000, 0));
        track.add_clip(make_clip(2, 1000, 2000, 0));
        let removed = track.remove_clip(1);
        assert!(removed);
        assert_eq!(track.clips.len(), 1);
        // Removing non-existent returns false.
        assert!(!track.remove_clip(999));
    }

    #[test]
    fn test_track_duration_ms() {
        let mut track = PyTrack::new(0, "Audio 1", PyTrackType::Audio);
        assert_eq!(track.duration_ms(), 0);
        track.add_clip(make_clip(1, 0, 5000, 0));
        track.add_clip(make_clip(2, 3000, 8000, 0));
        assert_eq!(track.duration_ms(), 8000);
    }

    #[test]
    fn test_track_clips_at_time() {
        let mut track = PyTrack::new(0, "Video 1", PyTrackType::Video);
        track.add_clip(make_clip(1, 0, 3000, 0));
        track.add_clip(make_clip(2, 2000, 5000, 0));
        track.add_clip(make_clip(3, 6000, 8000, 0));
        let at_2500 = track.clips_at_time(2500);
        assert_eq!(at_2500.len(), 2);
        let at_1000 = track.clips_at_time(1000);
        assert_eq!(at_1000.len(), 1);
        assert_eq!(at_1000[0].id, 1);
        let at_7000 = track.clips_at_time(7000);
        assert_eq!(at_7000.len(), 1);
        assert_eq!(at_7000[0].id, 3);
    }

    // ── PyTimeline ───────────────────────────────────────────

    #[test]
    fn test_timeline_new() {
        let tl = PyTimeline::new(25.0);
        assert!((tl.frame_rate - 25.0).abs() < f64::EPSILON);
        assert_eq!(tl.track_count(), 0);
    }

    #[test]
    fn test_timeline_add_track() {
        let mut tl = PyTimeline::new(30.0);
        tl.add_track(PyTrack::new(0, "Video 1", PyTrackType::Video));
        tl.add_track(PyTrack::new(1, "Audio 1", PyTrackType::Audio));
        assert_eq!(tl.track_count(), 2);
    }

    #[test]
    fn test_timeline_total_duration_ms() {
        let mut tl = PyTimeline::new(24.0);
        assert_eq!(tl.total_duration_ms(), 0);

        let mut v_track = PyTrack::new(0, "V1", PyTrackType::Video);
        v_track.add_clip(make_clip(1, 0, 10_000, 0));
        let mut a_track = PyTrack::new(1, "A1", PyTrackType::Audio);
        a_track.add_clip(make_clip(2, 0, 12_000, 0));

        tl.add_track(v_track);
        tl.add_track(a_track);
        assert_eq!(tl.total_duration_ms(), 12_000);
    }

    #[test]
    fn test_clip_saturation_on_duration() {
        // record_out < record_in would saturate to 0 (not panic).
        let clip = PyClip::new(99, "x.mp4", 0, 100, 5000, 3000, 0);
        assert_eq!(clip.duration_ms(), 0);
    }
}
