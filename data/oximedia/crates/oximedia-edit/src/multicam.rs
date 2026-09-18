//! Multi-camera editing support with sync point alignment.
//!
//! Enables editing footage from multiple camera angles that were
//! recording the same event. Cameras are synchronized via sync points
//! (timecode, audio waveform, or manual markers), allowing the editor
//! to switch between angles on a single output track.

#![allow(dead_code)]

use std::collections::HashMap;

use crate::clip::ClipId;
use crate::error::{EditError, EditResult};

/// Unique identifier for a camera angle.
pub type AngleId = u64;

/// Unique identifier for a multi-cam clip group.
pub type MultiCamId = u64;

/// Method used to synchronize camera angles.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SyncMethod {
    /// Synchronize by matching embedded timecode.
    Timecode,
    /// Synchronize by audio waveform cross-correlation.
    AudioWaveform,
    /// Synchronize by a manually placed marker at a known event.
    ManualMarker,
    /// Synchronize by a common start point (slate clap, flash, etc.).
    CommonStart,
}

impl SyncMethod {
    /// Returns a human-readable label for this sync method.
    #[must_use]
    pub fn label(self) -> &'static str {
        match self {
            Self::Timecode => "Timecode",
            Self::AudioWaveform => "Audio Waveform",
            Self::ManualMarker => "Manual Marker",
            Self::CommonStart => "Common Start",
        }
    }
}

/// A sync point that aligns a camera angle to the master timeline.
///
/// The `source_offset` is the position in the camera's source media
/// (in timebase units) that corresponds to `timeline_position` on the
/// master timeline.
#[derive(Debug, Clone, PartialEq)]
pub struct SyncPoint {
    /// Position on the master timeline (timebase units).
    pub timeline_position: i64,
    /// Corresponding position in the camera source (timebase units).
    pub source_offset: i64,
    /// Confidence score for automatic sync (0.0 to 1.0).
    pub confidence: f64,
    /// Sync method that produced this point.
    pub method: SyncMethod,
}

impl SyncPoint {
    /// Create a new sync point.
    #[must_use]
    pub fn new(
        timeline_position: i64,
        source_offset: i64,
        confidence: f64,
        method: SyncMethod,
    ) -> Self {
        Self {
            timeline_position,
            source_offset,
            confidence: confidence.clamp(0.0, 1.0),
            method,
        }
    }

    /// Compute the time delta between timeline and source.
    #[must_use]
    pub fn offset_delta(&self) -> i64 {
        self.timeline_position - self.source_offset
    }

    /// Returns `true` when the confidence exceeds a threshold.
    #[must_use]
    pub fn is_confident(&self, threshold: f64) -> bool {
        self.confidence >= threshold
    }
}

/// A single camera angle within a multi-cam group.
#[derive(Debug, Clone)]
pub struct CameraAngle {
    /// Unique angle identifier.
    pub id: AngleId,
    /// Human-readable label (e.g. "Camera A", "Wide Shot").
    pub label: String,
    /// Clip IDs that belong to this angle (video + audio).
    pub clips: Vec<ClipId>,
    /// Sync points for this angle.
    pub sync_points: Vec<SyncPoint>,
    /// Whether this angle is currently active (selected for output).
    pub active: bool,
    /// The computed offset to align this angle to the master timeline.
    /// Positive means the source starts *after* the master zero point.
    pub alignment_offset: i64,
}

impl CameraAngle {
    /// Create a new camera angle.
    #[must_use]
    pub fn new(id: AngleId, label: String) -> Self {
        Self {
            id,
            label,
            clips: Vec::new(),
            sync_points: Vec::new(),
            active: false,
            alignment_offset: 0,
        }
    }

    /// Add a clip to this angle.
    pub fn add_clip(&mut self, clip_id: ClipId) {
        if !self.clips.contains(&clip_id) {
            self.clips.push(clip_id);
        }
    }

    /// Remove a clip from this angle.
    pub fn remove_clip(&mut self, clip_id: ClipId) -> bool {
        if let Some(pos) = self.clips.iter().position(|&id| id == clip_id) {
            self.clips.remove(pos);
            true
        } else {
            false
        }
    }

    /// Add a sync point.
    pub fn add_sync_point(&mut self, point: SyncPoint) {
        self.sync_points.push(point);
    }

    /// Compute alignment offset from sync points.
    ///
    /// Uses a weighted average of sync point deltas, weighted by confidence.
    /// Returns `None` if there are no sync points.
    #[must_use]
    #[allow(clippy::cast_possible_truncation, clippy::cast_precision_loss)]
    pub fn compute_alignment(&self) -> Option<i64> {
        if self.sync_points.is_empty() {
            return None;
        }
        let total_weight: f64 = self.sync_points.iter().map(|sp| sp.confidence).sum();
        if total_weight < 1e-12 {
            // All zero-confidence: fall back to simple average
            let sum: i64 = self.sync_points.iter().map(SyncPoint::offset_delta).sum();
            return Some(sum / self.sync_points.len() as i64);
        }
        let weighted_sum: f64 = self
            .sync_points
            .iter()
            .map(|sp| sp.offset_delta() as f64 * sp.confidence)
            .sum();
        Some((weighted_sum / total_weight).round() as i64)
    }
}

/// A switch point on the output track indicating when to cut to a
/// different camera angle.
#[derive(Debug, Clone, PartialEq)]
pub struct AngleSwitch {
    /// Timeline position where the switch occurs.
    pub position: i64,
    /// The angle to switch to.
    pub angle_id: AngleId,
}

impl AngleSwitch {
    /// Create a new angle switch.
    #[must_use]
    pub fn new(position: i64, angle_id: AngleId) -> Self {
        Self { position, angle_id }
    }
}

/// A multi-camera editing group.
///
/// Contains multiple [`CameraAngle`]s that cover the same event, plus
/// a list of [`AngleSwitch`]es that define which angle is on-screen at
/// each point in the output timeline.
#[derive(Debug, Clone)]
pub struct MultiCamGroup {
    /// Unique multi-cam ID.
    pub id: MultiCamId,
    /// Human-readable name.
    pub name: String,
    /// Camera angles in this group.
    pub angles: Vec<CameraAngle>,
    /// Switch list (sorted by position).
    pub switches: Vec<AngleSwitch>,
    /// Next angle ID counter.
    next_angle_id: AngleId,
}

impl MultiCamGroup {
    /// Create a new multi-cam group.
    #[must_use]
    pub fn new(id: MultiCamId, name: String) -> Self {
        Self {
            id,
            name,
            angles: Vec::new(),
            switches: Vec::new(),
            next_angle_id: 1,
        }
    }

    /// Add a new camera angle and return its ID.
    pub fn add_angle(&mut self, label: String) -> AngleId {
        let id = self.next_angle_id;
        self.next_angle_id += 1;
        self.angles.push(CameraAngle::new(id, label));
        id
    }

    /// Remove a camera angle by ID.
    pub fn remove_angle(&mut self, angle_id: AngleId) -> Option<CameraAngle> {
        if let Some(pos) = self.angles.iter().position(|a| a.id == angle_id) {
            let angle = self.angles.remove(pos);
            // Remove switches that reference this angle
            self.switches.retain(|s| s.angle_id != angle_id);
            Some(angle)
        } else {
            None
        }
    }

    /// Get an angle by ID.
    #[must_use]
    pub fn get_angle(&self, angle_id: AngleId) -> Option<&CameraAngle> {
        self.angles.iter().find(|a| a.id == angle_id)
    }

    /// Get a mutable angle by ID.
    pub fn get_angle_mut(&mut self, angle_id: AngleId) -> Option<&mut CameraAngle> {
        self.angles.iter_mut().find(|a| a.id == angle_id)
    }

    /// Add a switch point (cut to a different angle).
    pub fn add_switch(&mut self, position: i64, angle_id: AngleId) -> EditResult<()> {
        if !self.angles.iter().any(|a| a.id == angle_id) {
            return Err(EditError::InvalidEdit(format!(
                "Angle {angle_id} not found in multi-cam group"
            )));
        }
        self.switches.push(AngleSwitch::new(position, angle_id));
        self.switches.sort_by_key(|s| s.position);
        Ok(())
    }

    /// Remove the switch at a given position.
    pub fn remove_switch_at(&mut self, position: i64) -> Option<AngleSwitch> {
        if let Some(pos) = self.switches.iter().position(|s| s.position == position) {
            Some(self.switches.remove(pos))
        } else {
            None
        }
    }

    /// Get the active angle at a given timeline position.
    ///
    /// Finds the most recent switch at or before `position`.
    #[must_use]
    pub fn active_angle_at(&self, position: i64) -> Option<AngleId> {
        self.switches
            .iter()
            .rev()
            .find(|s| s.position <= position)
            .map(|s| s.angle_id)
    }

    /// Synchronize all angles by computing their alignment offsets.
    pub fn sync_all_angles(&mut self) {
        for angle in &mut self.angles {
            if let Some(offset) = angle.compute_alignment() {
                angle.alignment_offset = offset;
            }
        }
    }

    /// Get the total number of angles.
    #[must_use]
    pub fn angle_count(&self) -> usize {
        self.angles.len()
    }

    /// Get the total number of switches.
    #[must_use]
    pub fn switch_count(&self) -> usize {
        self.switches.len()
    }
}

/// Manager for all multi-cam groups in a project.
#[derive(Debug, Default)]
pub struct MultiCamManager {
    /// All multi-cam groups.
    groups: HashMap<MultiCamId, MultiCamGroup>,
    /// Next group ID.
    next_id: MultiCamId,
}

impl MultiCamManager {
    /// Create a new manager.
    #[must_use]
    pub fn new() -> Self {
        Self {
            groups: HashMap::new(),
            next_id: 1,
        }
    }

    /// Create a new multi-cam group.
    pub fn create_group(&mut self, name: String) -> MultiCamId {
        let id = self.next_id;
        self.next_id += 1;
        self.groups.insert(id, MultiCamGroup::new(id, name));
        id
    }

    /// Delete a multi-cam group.
    pub fn delete_group(&mut self, id: MultiCamId) -> Option<MultiCamGroup> {
        self.groups.remove(&id)
    }

    /// Get a group by ID.
    #[must_use]
    pub fn get_group(&self, id: MultiCamId) -> Option<&MultiCamGroup> {
        self.groups.get(&id)
    }

    /// Get a mutable group by ID.
    pub fn get_group_mut(&mut self, id: MultiCamId) -> Option<&mut MultiCamGroup> {
        self.groups.get_mut(&id)
    }

    /// Get all groups.
    #[must_use]
    pub fn all_groups(&self) -> Vec<&MultiCamGroup> {
        self.groups.values().collect()
    }

    /// Clear all groups.
    pub fn clear(&mut self) {
        self.groups.clear();
    }

    /// Get total group count.
    #[must_use]
    pub fn len(&self) -> usize {
        self.groups.len()
    }

    /// Check if there are no groups.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.groups.is_empty()
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// MultiCamSession / MultiCamEditor API
// ─────────────────────────────────────────────────────────────────────────────

/// A lightweight reference to a clip within a camera track,
/// expressed as a millisecond-based time range.
#[derive(Debug, Clone, PartialEq)]
pub struct ClipRef {
    /// The underlying clip identifier.
    pub clip_id: ClipId,
    /// Start of the clip reference on the camera timeline, in milliseconds.
    pub start_ms: i64,
    /// Duration of this reference, in milliseconds.
    pub duration_ms: i64,
}

impl ClipRef {
    /// Create a new clip reference.
    #[must_use]
    pub fn new(clip_id: ClipId, start_ms: i64, duration_ms: i64) -> Self {
        Self {
            clip_id,
            start_ms,
            duration_ms,
        }
    }

    /// Inclusive end position of this reference (start_ms + duration_ms).
    #[must_use]
    pub fn end_ms(&self) -> i64 {
        self.start_ms + self.duration_ms
    }
}

/// A single camera track holding an ordered list of [`ClipRef`]s and a
/// source timecode anchor.
#[derive(Debug, Clone)]
pub struct CameraTrack {
    /// Human-readable or machine-generated camera identifier.
    pub id: String,
    /// Ordered list of clip references on this track.
    pub clips: Vec<ClipRef>,
    /// The timeline start position of this camera, in milliseconds.
    /// All clip start_ms values are relative to this anchor.
    pub timecode_start: i64,
}

impl CameraTrack {
    /// Create a new empty camera track.
    #[must_use]
    pub fn new(id: impl Into<String>, timecode_start: i64) -> Self {
        Self {
            id: id.into(),
            clips: Vec::new(),
            timecode_start,
        }
    }

    /// Append a clip reference to the track.
    pub fn add_clip(&mut self, clip: ClipRef) {
        self.clips.push(clip);
    }

    /// Return the clip whose time range contains `ms` (timeline-relative).
    ///
    /// The lookup accounts for the track's `timecode_start` offset so that
    /// `ms` should be given in the same coordinate system as `timecode_start`.
    #[must_use]
    pub fn clip_at_ms(&self, ms: i64) -> Option<&ClipRef> {
        let local_ms = ms - self.timecode_start;
        self.clips
            .iter()
            .find(|c| local_ms >= c.start_ms && local_ms < c.start_ms + c.duration_ms)
    }

    /// Total number of clips on this track.
    #[must_use]
    pub fn clip_count(&self) -> usize {
        self.clips.len()
    }
}

/// A cut from one camera to another at a specific timeline position.
#[derive(Debug, Clone, PartialEq)]
pub struct MultiCamCut {
    /// Position at which the cut occurs, in milliseconds.
    pub timestamp_ms: u64,
    /// Index (into `MultiCamSession::cameras`) of the outgoing camera.
    pub from_camera: usize,
    /// Index (into `MultiCamSession::cameras`) of the incoming camera.
    pub to_camera: usize,
}

impl MultiCamCut {
    /// Create a new multi-cam cut.
    #[must_use]
    pub fn new(timestamp_ms: u64, from_camera: usize, to_camera: usize) -> Self {
        Self {
            timestamp_ms,
            from_camera,
            to_camera,
        }
    }
}

/// A multi-camera editing session that groups camera tracks and their sync offsets.
#[derive(Debug, Clone, Default)]
pub struct MultiCamSession {
    /// All camera tracks in this session, in order.
    pub cameras: Vec<CameraTrack>,
    /// Per-camera sync offset in milliseconds (positive = camera starts later).
    /// The vector is grown automatically when cameras are added.
    pub sync_offset_ms: Vec<i64>,
}

impl MultiCamSession {
    /// Create an empty session.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Add a camera track to the session. A zero sync offset is appended.
    pub fn add_camera(&mut self, track: CameraTrack) {
        self.cameras.push(track);
        self.sync_offset_ms.push(0);
    }

    /// Override the sync offset for `camera_idx`.
    ///
    /// # Errors
    ///
    /// Returns [`EditError::InvalidEdit`] when `camera_idx` is out of range.
    pub fn set_sync_offset(&mut self, camera_idx: usize, offset_ms: i64) -> EditResult<()> {
        if camera_idx >= self.cameras.len() {
            return Err(EditError::InvalidEdit(format!(
                "Camera index {camera_idx} out of range (session has {} cameras)",
                self.cameras.len()
            )));
        }
        self.sync_offset_ms[camera_idx] = offset_ms;
        Ok(())
    }

    /// Number of cameras in this session.
    #[must_use]
    pub fn camera_count(&self) -> usize {
        self.cameras.len()
    }

    /// Return the sync-adjusted timeline position for a given camera at absolute
    /// position `abs_ms`.  The result is the position within that camera's own
    /// coordinate system after accounting for its sync offset.
    #[must_use]
    pub fn camera_local_ms(&self, camera_idx: usize, abs_ms: i64) -> i64 {
        let offset = self.sync_offset_ms.get(camera_idx).copied().unwrap_or(0);
        abs_ms - offset
    }
}

/// High-level multi-camera editing controller.
///
/// Manages a [`MultiCamSession`] together with a sorted list of
/// [`MultiCamCut`]s and provides timeline generation.
#[derive(Debug)]
pub struct MultiCamEditor {
    /// The underlying session (cameras + sync offsets).
    pub session: MultiCamSession,
    /// Sorted list of cuts.
    pub cuts: Vec<MultiCamCut>,
}

impl MultiCamEditor {
    /// Create a new editor wrapping the given session.
    #[must_use]
    pub fn new(session: MultiCamSession) -> Self {
        Self {
            session,
            cuts: Vec::new(),
        }
    }

    /// Add a cut, keeping the cut list sorted by `timestamp_ms`.
    ///
    /// # Errors
    ///
    /// Returns [`EditError::InvalidEdit`] if either `from_camera` or `to_camera`
    /// is not a valid camera index in the underlying session.
    pub fn add_cut(&mut self, cut: MultiCamCut) -> EditResult<()> {
        let n = self.session.camera_count();
        if cut.from_camera >= n {
            return Err(EditError::InvalidEdit(format!(
                "from_camera {} is out of range (session has {n} cameras)",
                cut.from_camera
            )));
        }
        if cut.to_camera >= n {
            return Err(EditError::InvalidEdit(format!(
                "to_camera {} is out of range (session has {n} cameras)",
                cut.to_camera
            )));
        }
        self.cuts.push(cut);
        self.cuts.sort_by_key(|c| c.timestamp_ms);
        Ok(())
    }

    /// Remove the first cut whose `timestamp_ms` equals `timestamp_ms`.
    pub fn remove_cut_at(&mut self, timestamp_ms: u64) -> Option<MultiCamCut> {
        if let Some(pos) = self
            .cuts
            .iter()
            .position(|c| c.timestamp_ms == timestamp_ms)
        {
            Some(self.cuts.remove(pos))
        } else {
            None
        }
    }

    /// Number of cuts currently registered.
    #[must_use]
    pub fn cut_count(&self) -> usize {
        self.cuts.len()
    }

    /// Generate a flat timeline as a sequence of [`ClipRef`]s.
    ///
    /// For each pair of consecutive cuts (or the single interval defined by the
    /// only cut), the active camera at the *midpoint* of that interval is queried.
    /// The resulting [`ClipRef`] spans the full interval.
    ///
    /// If there are no cuts, an empty vec is returned.
    #[must_use]
    pub fn generate_timeline(&self) -> Vec<ClipRef> {
        if self.cuts.is_empty() {
            return Vec::new();
        }

        let mut result = Vec::with_capacity(self.cuts.len());

        for (i, cut) in self.cuts.iter().enumerate() {
            let interval_start_ms = cut.timestamp_ms as i64;
            let interval_end_ms = self
                .cuts
                .get(i + 1)
                .map_or(interval_start_ms + 5000, |next| next.timestamp_ms as i64);

            let midpoint_ms = (interval_start_ms + interval_end_ms) / 2;
            let cam_idx = cut.to_camera;

            // Apply sync offset for this camera.
            let local_ms = self.session.camera_local_ms(cam_idx, midpoint_ms);

            if let Some(cam) = self.session.cameras.get(cam_idx) {
                // Attempt to find the clip covering local_ms (track-relative).
                let clip_ref = cam.clip_at_ms(local_ms + cam.timecode_start);
                let duration_ms = interval_end_ms - interval_start_ms;

                if let Some(cr) = clip_ref {
                    result.push(ClipRef::new(cr.clip_id, interval_start_ms, duration_ms));
                } else if let Some(first_clip) = cam.clips.first() {
                    // Fallback: use the first clip of the camera track.
                    result.push(ClipRef::new(
                        first_clip.clip_id,
                        interval_start_ms,
                        duration_ms,
                    ));
                }
                // If the camera has no clips, we skip this interval.
            }
        }

        result
    }
}

/// Compute the cross-correlation offset (in samples) between two audio
/// fingerprint arrays.
///
/// The function slides `camera_b` over `camera_a` in the lag range
/// `[-max_lag, +max_lag]` where `max_lag = len / 4` (at least 1), and
/// returns the lag that maximises the dot-product.
///
/// A positive returned lag means `camera_b` is shifted forward relative to
/// `camera_a` by that many samples; a negative lag means it starts earlier.
///
/// Returns `0` when either slice is empty.
#[must_use]
pub fn sync_by_audio_fingerprint(camera_a: &[f32], camera_b: &[f32]) -> i64 {
    if camera_a.is_empty() || camera_b.is_empty() {
        return 0;
    }

    let len_a = camera_a.len();
    let len_b = camera_b.len();
    let window = len_a.min(len_b);
    let max_lag = (window / 4).max(1) as i64;

    let mut best_lag: i64 = 0;
    let mut best_corr = f64::NEG_INFINITY;

    for lag in -max_lag..=max_lag {
        let mut dot: f64 = 0.0;
        for j in 0..window {
            let b_idx = j as i64 + lag;
            if b_idx >= 0 && (b_idx as usize) < len_b {
                dot += f64::from(camera_a[j]) * f64::from(camera_b[b_idx as usize]);
            }
        }
        if dot > best_corr {
            best_corr = dot;
            best_lag = lag;
        }
    }

    best_lag
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sync_method_label() {
        assert_eq!(SyncMethod::Timecode.label(), "Timecode");
        assert_eq!(SyncMethod::AudioWaveform.label(), "Audio Waveform");
        assert_eq!(SyncMethod::ManualMarker.label(), "Manual Marker");
        assert_eq!(SyncMethod::CommonStart.label(), "Common Start");
    }

    #[test]
    fn test_sync_point_creation() {
        let sp = SyncPoint::new(1000, 500, 0.95, SyncMethod::Timecode);
        assert_eq!(sp.timeline_position, 1000);
        assert_eq!(sp.source_offset, 500);
        assert!((sp.confidence - 0.95).abs() < 1e-9);
        assert_eq!(sp.offset_delta(), 500);
    }

    #[test]
    fn test_sync_point_confidence_clamped() {
        let sp = SyncPoint::new(0, 0, 1.5, SyncMethod::Timecode);
        assert!((sp.confidence - 1.0).abs() < 1e-9);
        let sp2 = SyncPoint::new(0, 0, -0.5, SyncMethod::Timecode);
        assert!((sp2.confidence).abs() < 1e-9);
    }

    #[test]
    fn test_sync_point_is_confident() {
        let sp = SyncPoint::new(0, 0, 0.8, SyncMethod::Timecode);
        assert!(sp.is_confident(0.5));
        assert!(sp.is_confident(0.8));
        assert!(!sp.is_confident(0.9));
    }

    #[test]
    fn test_camera_angle_clips() {
        let mut angle = CameraAngle::new(1, "Camera A".to_string());
        angle.add_clip(10);
        angle.add_clip(20);
        angle.add_clip(10); // duplicate, should not add
        assert_eq!(angle.clips.len(), 2);
        assert!(angle.remove_clip(10));
        assert!(!angle.remove_clip(999));
        assert_eq!(angle.clips.len(), 1);
    }

    #[test]
    fn test_camera_angle_compute_alignment() {
        let mut angle = CameraAngle::new(1, "A".to_string());
        assert!(angle.compute_alignment().is_none());

        angle.add_sync_point(SyncPoint::new(1000, 500, 1.0, SyncMethod::Timecode));
        angle.add_sync_point(SyncPoint::new(2000, 1500, 1.0, SyncMethod::Timecode));
        // Both deltas are 500, so alignment should be 500
        let offset = angle.compute_alignment();
        assert_eq!(offset, Some(500));
    }

    #[test]
    fn test_camera_angle_compute_alignment_weighted() {
        let mut angle = CameraAngle::new(1, "A".to_string());
        // delta=100 at high confidence, delta=200 at low confidence
        angle.add_sync_point(SyncPoint::new(100, 0, 0.9, SyncMethod::Timecode));
        angle.add_sync_point(SyncPoint::new(200, 0, 0.1, SyncMethod::AudioWaveform));
        let offset = angle.compute_alignment();
        // weighted = (100*0.9 + 200*0.1) / (0.9+0.1) = (90+20)/1.0 = 110
        assert_eq!(offset, Some(110));
    }

    #[test]
    fn test_multicam_group_add_remove_angle() {
        let mut group = MultiCamGroup::new(1, "Test".to_string());
        let a1 = group.add_angle("Camera A".to_string());
        let a2 = group.add_angle("Camera B".to_string());
        assert_eq!(group.angle_count(), 2);

        assert!(group.get_angle(a1).is_some());
        assert!(group.remove_angle(a2).is_some());
        assert_eq!(group.angle_count(), 1);
        assert!(group.remove_angle(999).is_none());
    }

    #[test]
    fn test_multicam_group_switches() {
        let mut group = MultiCamGroup::new(1, "Test".to_string());
        let a1 = group.add_angle("Camera A".to_string());
        let a2 = group.add_angle("Camera B".to_string());

        assert!(group.add_switch(0, a1).is_ok());
        assert!(group.add_switch(5000, a2).is_ok());
        assert!(group.add_switch(10000, a1).is_ok());
        assert_eq!(group.switch_count(), 3);

        // Invalid angle
        assert!(group.add_switch(15000, 999).is_err());

        // Active angle lookup
        assert_eq!(group.active_angle_at(2500), Some(a1));
        assert_eq!(group.active_angle_at(5000), Some(a2));
        assert_eq!(group.active_angle_at(7500), Some(a2));
        assert_eq!(group.active_angle_at(10000), Some(a1));
        assert!(group.active_angle_at(-100).is_none());
    }

    #[test]
    fn test_multicam_group_remove_switch() {
        let mut group = MultiCamGroup::new(1, "Test".to_string());
        let a1 = group.add_angle("A".to_string());
        let _ = group.add_switch(0, a1);
        assert!(group.remove_switch_at(0).is_some());
        assert!(group.remove_switch_at(0).is_none());
        assert_eq!(group.switch_count(), 0);
    }

    #[test]
    fn test_multicam_group_remove_angle_removes_switches() {
        let mut group = MultiCamGroup::new(1, "Test".to_string());
        let a1 = group.add_angle("A".to_string());
        let a2 = group.add_angle("B".to_string());
        let _ = group.add_switch(0, a1);
        let _ = group.add_switch(5000, a2);
        group.remove_angle(a1);
        // Only switches for a2 should remain
        assert_eq!(group.switch_count(), 1);
        assert_eq!(group.switches[0].angle_id, a2);
    }

    #[test]
    fn test_multicam_group_sync_all_angles() {
        let mut group = MultiCamGroup::new(1, "Test".to_string());
        let a1 = group.add_angle("A".to_string());
        let angle = group.get_angle_mut(a1).expect("angle should exist");
        angle.add_sync_point(SyncPoint::new(1000, 800, 1.0, SyncMethod::Timecode));
        group.sync_all_angles();
        assert_eq!(
            group
                .get_angle(a1)
                .expect("angle should exist")
                .alignment_offset,
            200
        );
    }

    #[test]
    fn test_multicam_manager() {
        let mut mgr = MultiCamManager::new();
        assert!(mgr.is_empty());

        let g1 = mgr.create_group("Group 1".to_string());
        let g2 = mgr.create_group("Group 2".to_string());
        assert_eq!(mgr.len(), 2);

        assert!(mgr.get_group(g1).is_some());
        assert!(mgr.delete_group(g2).is_some());
        assert_eq!(mgr.len(), 1);

        mgr.clear();
        assert!(mgr.is_empty());
    }

    #[test]
    fn test_multicam_manager_all_groups() {
        let mut mgr = MultiCamManager::new();
        let _ = mgr.create_group("A".to_string());
        let _ = mgr.create_group("B".to_string());
        assert_eq!(mgr.all_groups().len(), 2);
    }

    #[test]
    fn test_angle_switch_creation() {
        let sw = AngleSwitch::new(5000, 2);
        assert_eq!(sw.position, 5000);
        assert_eq!(sw.angle_id, 2);
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// MultiCamSession / MultiCamEditor / sync_by_audio_fingerprint tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod multicam_session_tests {
    use super::*;

    fn make_clip_ref(id: ClipId, start_ms: i64, duration_ms: i64) -> ClipRef {
        ClipRef::new(id, start_ms, duration_ms)
    }

    fn make_track(id: &str, timecode_start: i64, clips: Vec<ClipRef>) -> CameraTrack {
        let mut t = CameraTrack::new(id, timecode_start);
        for c in clips {
            t.add_clip(c);
        }
        t
    }

    // ── ClipRef ──────────────────────────────────────────────────────────────

    #[test]
    fn test_clip_ref_creation() {
        let cr = make_clip_ref(42, 1000, 500);
        assert_eq!(cr.clip_id, 42);
        assert_eq!(cr.start_ms, 1000);
        assert_eq!(cr.duration_ms, 500);
    }

    #[test]
    fn test_clip_ref_end_ms() {
        let cr = make_clip_ref(1, 200, 300);
        assert_eq!(cr.end_ms(), 500);
    }

    // ── CameraTrack ──────────────────────────────────────────────────────────

    #[test]
    fn test_camera_track_add_and_count() {
        let mut t = CameraTrack::new("cam-a", 0);
        assert_eq!(t.clip_count(), 0);
        t.add_clip(make_clip_ref(10, 0, 1000));
        t.add_clip(make_clip_ref(11, 1000, 1000));
        assert_eq!(t.clip_count(), 2);
    }

    #[test]
    fn test_camera_track_clip_at_ms_found() {
        let clips = vec![make_clip_ref(10, 0, 1000), make_clip_ref(11, 1000, 1000)];
        let track = make_track("cam-a", 0, clips);
        let found = track.clip_at_ms(500);
        assert!(found.is_some());
        assert_eq!(found.expect("clip should exist").clip_id, 10);
    }

    #[test]
    fn test_camera_track_clip_at_ms_second_clip() {
        let clips = vec![make_clip_ref(10, 0, 1000), make_clip_ref(11, 1000, 1000)];
        let track = make_track("cam-a", 0, clips);
        let found = track.clip_at_ms(1500);
        assert!(found.is_some());
        assert_eq!(found.expect("clip should exist").clip_id, 11);
    }

    #[test]
    fn test_camera_track_clip_at_ms_not_found() {
        let clips = vec![make_clip_ref(10, 0, 1000)];
        let track = make_track("cam-a", 0, clips);
        assert!(track.clip_at_ms(5000).is_none());
    }

    #[test]
    fn test_camera_track_timecode_start_offset() {
        // Track starts at 10_000 ms. Clip covers [0, 1000) local.
        // Absolute query at 10_500 should find the clip.
        let clips = vec![make_clip_ref(99, 0, 1000)];
        let track = make_track("cam-b", 10_000, clips);
        let found = track.clip_at_ms(10_500);
        assert!(found.is_some(), "Should find clip at abs 10_500");
    }

    // ── MultiCamCut ──────────────────────────────────────────────────────────

    #[test]
    fn test_multicam_cut_creation() {
        let cut = MultiCamCut::new(3000, 0, 1);
        assert_eq!(cut.timestamp_ms, 3000);
        assert_eq!(cut.from_camera, 0);
        assert_eq!(cut.to_camera, 1);
    }

    // ── MultiCamSession ──────────────────────────────────────────────────────

    #[test]
    fn test_session_add_camera() {
        let mut session = MultiCamSession::new();
        assert_eq!(session.camera_count(), 0);
        session.add_camera(CameraTrack::new("A", 0));
        session.add_camera(CameraTrack::new("B", 0));
        assert_eq!(session.camera_count(), 2);
        assert_eq!(session.sync_offset_ms.len(), 2);
    }

    #[test]
    fn test_session_set_sync_offset_valid() {
        let mut session = MultiCamSession::new();
        session.add_camera(CameraTrack::new("A", 0));
        let result = session.set_sync_offset(0, 250);
        assert!(result.is_ok());
        assert_eq!(session.sync_offset_ms[0], 250);
    }

    #[test]
    fn test_session_set_sync_offset_out_of_bounds() {
        let mut session = MultiCamSession::new();
        session.add_camera(CameraTrack::new("A", 0));
        let result = session.set_sync_offset(5, 100);
        assert!(result.is_err());
    }

    #[test]
    fn test_session_camera_local_ms() {
        let mut session = MultiCamSession::new();
        session.add_camera(CameraTrack::new("A", 0));
        session.set_sync_offset(0, 500).expect("set ok");
        // abs_ms=2000, offset=500 → local_ms = 2000 - 500 = 1500
        assert_eq!(session.camera_local_ms(0, 2000), 1500);
    }

    // ── MultiCamEditor ───────────────────────────────────────────────────────

    #[test]
    fn test_editor_add_cut_valid() {
        let mut session = MultiCamSession::new();
        session.add_camera(CameraTrack::new("A", 0));
        session.add_camera(CameraTrack::new("B", 0));
        let mut editor = MultiCamEditor::new(session);
        let result = editor.add_cut(MultiCamCut::new(1000, 0, 1));
        assert!(result.is_ok());
        assert_eq!(editor.cut_count(), 1);
    }

    #[test]
    fn test_editor_add_cut_invalid_from_camera() {
        let mut session = MultiCamSession::new();
        session.add_camera(CameraTrack::new("A", 0));
        let mut editor = MultiCamEditor::new(session);
        let result = editor.add_cut(MultiCamCut::new(1000, 5, 0));
        assert!(result.is_err());
    }

    #[test]
    fn test_editor_add_cut_invalid_to_camera() {
        let mut session = MultiCamSession::new();
        session.add_camera(CameraTrack::new("A", 0));
        let mut editor = MultiCamEditor::new(session);
        let result = editor.add_cut(MultiCamCut::new(1000, 0, 5));
        assert!(result.is_err());
    }

    #[test]
    fn test_editor_cuts_sorted() {
        let mut session = MultiCamSession::new();
        session.add_camera(CameraTrack::new("A", 0));
        session.add_camera(CameraTrack::new("B", 0));
        let mut editor = MultiCamEditor::new(session);
        editor.add_cut(MultiCamCut::new(5000, 0, 1)).expect("ok");
        editor.add_cut(MultiCamCut::new(1000, 1, 0)).expect("ok");
        editor.add_cut(MultiCamCut::new(3000, 0, 1)).expect("ok");
        assert_eq!(editor.cuts[0].timestamp_ms, 1000);
        assert_eq!(editor.cuts[1].timestamp_ms, 3000);
        assert_eq!(editor.cuts[2].timestamp_ms, 5000);
    }

    #[test]
    fn test_editor_remove_cut_at_found() {
        let mut session = MultiCamSession::new();
        session.add_camera(CameraTrack::new("A", 0));
        session.add_camera(CameraTrack::new("B", 0));
        let mut editor = MultiCamEditor::new(session);
        editor.add_cut(MultiCamCut::new(1000, 0, 1)).expect("ok");
        let removed = editor.remove_cut_at(1000);
        assert!(removed.is_some());
        assert_eq!(editor.cut_count(), 0);
    }

    #[test]
    fn test_editor_remove_cut_at_not_found() {
        let mut session = MultiCamSession::new();
        session.add_camera(CameraTrack::new("A", 0));
        let mut editor = MultiCamEditor::new(session);
        assert!(editor.remove_cut_at(9999).is_none());
    }

    #[test]
    fn test_editor_generate_timeline_empty_cuts() {
        let session = MultiCamSession::new();
        let editor = MultiCamEditor::new(session);
        assert!(editor.generate_timeline().is_empty());
    }

    #[test]
    fn test_editor_generate_timeline_single_cut() {
        let mut session = MultiCamSession::new();
        let mut cam = CameraTrack::new("A", 0);
        cam.add_clip(ClipRef::new(7, 0, 10_000));
        session.add_camera(cam);
        session.add_camera(CameraTrack::new("B", 0)); // dummy second cam

        let mut editor = MultiCamEditor::new(session);
        editor.add_cut(MultiCamCut::new(0, 0, 0)).expect("ok");

        let tl = editor.generate_timeline();
        // One cut → one output clip reference
        assert_eq!(tl.len(), 1);
        assert_eq!(tl[0].clip_id, 7);
    }

    // ── sync_by_audio_fingerprint ─────────────────────────────────────────────

    #[test]
    fn test_sync_empty_a() {
        assert_eq!(sync_by_audio_fingerprint(&[], &[1.0, 2.0, 3.0]), 0);
    }

    #[test]
    fn test_sync_empty_b() {
        assert_eq!(sync_by_audio_fingerprint(&[1.0, 2.0, 3.0], &[]), 0);
    }

    #[test]
    fn test_sync_both_empty() {
        assert_eq!(sync_by_audio_fingerprint(&[], &[]), 0);
    }

    #[test]
    fn test_sync_identical_signals_zero_lag() {
        let signal: Vec<f32> = (0..64).map(|i| (i as f32).sin()).collect();
        let lag = sync_by_audio_fingerprint(&signal, &signal);
        assert_eq!(lag, 0, "Identical signals should have zero lag");
    }

    #[test]
    fn test_sync_shifted_signal() {
        // camera_b is camera_a shifted right by 4 samples.
        let len = 64usize;
        let shift = 4i64;
        let base: Vec<f32> = (0..len).map(|i| ((i as f32) * 0.3).sin()).collect();
        let mut shifted = vec![0.0f32; len];
        for i in shift as usize..len {
            shifted[i] = base[i - shift as usize];
        }
        let lag = sync_by_audio_fingerprint(&base, &shifted);
        // lag == shift means camera_b is shifted forward by `shift` samples
        assert_eq!(lag, shift, "Expected lag={shift}, got {lag}");
    }
}
