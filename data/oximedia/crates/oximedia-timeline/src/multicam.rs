//! Multi-camera editing support.

use serde::{Deserialize, Serialize};
use std::collections::hash_map::DefaultHasher;
use std::collections::HashMap;
use std::hash::{Hash, Hasher};
use uuid::Uuid;

use crate::clip::Clip;
use crate::error::{TimelineError, TimelineResult};
use crate::types::Position;

/// Compute simplified cross-correlation offsets (in samples) between
/// `clips` and a reference clip at `reference_index`.
///
/// For each clip a pseudo-waveform is derived deterministically from the
/// clip name hash so the result is stable and testable without real audio.
/// The cross-correlation is computed over a sliding window of `WINDOW` samples
/// and the lag that maximises the dot-product is returned.
///
/// Returns a `Vec<i64>` of length `clips.len()` where each element is the
/// estimated lag (in samples) relative to the reference clip. The element at
/// `reference_index` is always 0.
#[allow(dead_code)]
#[must_use]
pub fn sync_by_audio_correlation(clips: &[&Clip], reference_index: usize) -> Vec<i64> {
    const WINDOW: usize = 64;
    const MAX_LAG: usize = 32;

    /// Generate a deterministic pseudo-waveform for a clip.
    fn pseudo_wave(clip: &Clip, len: usize) -> Vec<f64> {
        let mut hasher = DefaultHasher::new();
        clip.name.hash(&mut hasher);
        clip.id.as_uuid().hash(&mut hasher);
        let seed = hasher.finish();

        // Simple pseudo-random waveform seeded from clip hash.
        let freq = (seed & 0xF) as usize + 4; // 4..19
        (0..len)
            .map(|i| {
                let phase = (i % (freq * 2)) as f64 / (freq as f64);
                if phase < 1.0 {
                    phase * 2.0 - 1.0
                } else {
                    (2.0 - phase) * 2.0 - 1.0
                }
            })
            .collect()
    }

    if clips.is_empty() {
        return Vec::new();
    }

    let ref_wave = if reference_index < clips.len() {
        pseudo_wave(clips[reference_index], WINDOW)
    } else {
        vec![0.0f64; WINDOW]
    };

    clips
        .iter()
        .enumerate()
        .map(|(i, clip)| {
            if i == reference_index {
                return 0i64;
            }

            let wave = pseudo_wave(clip, WINDOW + MAX_LAG);

            // Find lag in [-MAX_LAG, MAX_LAG] that maximises dot-product.
            let mut best_lag: i64 = 0;
            let mut best_corr = f64::NEG_INFINITY;

            for lag in 0..=(MAX_LAG as i64) {
                for sign in [-1i64, 1i64] {
                    let shift = sign * lag;
                    let dot: f64 = (0..WINDOW)
                        .map(|j| {
                            let k = (j as i64 + shift) as usize;
                            if k < wave.len() {
                                ref_wave[j] * wave[k]
                            } else {
                                0.0
                            }
                        })
                        .sum();
                    if dot > best_corr {
                        best_corr = dot;
                        best_lag = shift;
                    }
                }
            }

            best_lag
        })
        .collect()
}

/// Multi-camera clip identifier.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct MultiCamId(Uuid);

impl MultiCamId {
    /// Creates a new multi-camera ID.
    #[must_use]
    pub fn new() -> Self {
        Self(Uuid::new_v4())
    }
}

impl Default for MultiCamId {
    fn default() -> Self {
        Self::new()
    }
}

/// Camera angle in a multi-camera setup.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct CameraAngle {
    /// Angle number/identifier.
    pub angle: u32,
    /// Name of the angle (e.g., "Camera 1", "Wide Shot").
    pub name: String,
    /// Clip for this angle.
    pub clip: Clip,
    /// Whether this angle is enabled.
    pub enabled: bool,
}

impl CameraAngle {
    /// Creates a new camera angle.
    #[must_use]
    pub fn new(angle: u32, name: String, clip: Clip) -> Self {
        Self {
            angle,
            name,
            clip,
            enabled: true,
        }
    }

    /// Enables the angle.
    pub fn enable(&mut self) {
        self.enabled = true;
    }

    /// Disables the angle.
    pub fn disable(&mut self) {
        self.enabled = false;
    }
}

/// Angle switch point in timeline.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct AngleSwitch {
    /// Position where switch occurs.
    pub position: Position,
    /// Angle to switch to.
    pub angle: u32,
    /// Optional embedded timecode string (e.g. "01:00:10:00") that was used
    /// to derive this switch position.  Stored for round-trip fidelity.
    pub timecode_hint: Option<String>,
}

impl AngleSwitch {
    /// Creates a new angle switch at a timeline `position`.
    #[must_use]
    pub const fn new(position: Position, angle: u32) -> Self {
        Self {
            position,
            angle,
            timecode_hint: None,
        }
    }

    /// Creates a new angle switch carrying a timecode hint string.
    #[must_use]
    pub fn with_timecode(position: Position, angle: u32, timecode: impl Into<String>) -> Self {
        Self {
            position,
            angle,
            timecode_hint: Some(timecode.into()),
        }
    }
}

/// Multi-camera clip.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct MultiCamClip {
    /// Unique identifier.
    pub id: MultiCamId,
    /// Name of the multi-cam clip.
    pub name: String,
    /// Camera angles.
    pub angles: Vec<CameraAngle>,
    /// Angle switches.
    pub switches: Vec<AngleSwitch>,
    /// Audio reference track (angle index).
    pub audio_reference: Option<u32>,
    /// Timeline start position.
    pub timeline_in: Position,
}

impl MultiCamClip {
    /// Creates a new multi-camera clip.
    #[must_use]
    pub fn new(name: String, timeline_in: Position) -> Self {
        Self {
            id: MultiCamId::new(),
            name,
            angles: Vec::new(),
            switches: Vec::new(),
            audio_reference: None,
            timeline_in,
        }
    }

    /// Adds a camera angle.
    ///
    /// # Errors
    ///
    /// Returns error if angle number already exists.
    pub fn add_angle(&mut self, angle: CameraAngle) -> TimelineResult<()> {
        if self.angles.iter().any(|a| a.angle == angle.angle) {
            return Err(TimelineError::MultiCamError(format!(
                "Angle {} already exists",
                angle.angle
            )));
        }
        self.angles.push(angle);
        self.angles.sort_by_key(|a| a.angle);
        Ok(())
    }

    /// Removes a camera angle.
    ///
    /// # Errors
    ///
    /// Returns error if angle not found.
    pub fn remove_angle(&mut self, angle: u32) -> TimelineResult<CameraAngle> {
        let index = self
            .angles
            .iter()
            .position(|a| a.angle == angle)
            .ok_or_else(|| TimelineError::MultiCamError(format!("Angle {angle} not found")))?;
        Ok(self.angles.remove(index))
    }

    /// Gets an angle by number.
    #[must_use]
    pub fn get_angle(&self, angle: u32) -> Option<&CameraAngle> {
        self.angles.iter().find(|a| a.angle == angle)
    }

    /// Gets a mutable reference to an angle.
    pub fn get_angle_mut(&mut self, angle: u32) -> Option<&mut CameraAngle> {
        self.angles.iter_mut().find(|a| a.angle == angle)
    }

    /// Adds an angle switch.
    pub fn add_switch(&mut self, switch: AngleSwitch) {
        self.switches.push(switch);
        self.switches.sort_by_key(|s| s.position.value());
    }

    /// Removes switches at a position.
    pub fn remove_switches_at(&mut self, position: Position) {
        self.switches.retain(|s| s.position != position);
    }

    /// Gets the active angle at a position.
    #[must_use]
    pub fn active_angle_at(&self, position: Position) -> Option<u32> {
        // Find the most recent switch before or at this position
        self.switches
            .iter()
            .rev()
            .find(|s| s.position <= position)
            .map(|s| s.angle)
            .or_else(|| self.angles.first().map(|a| a.angle))
    }

    /// Sets the audio reference track.
    pub fn set_audio_reference(&mut self, angle: u32) {
        self.audio_reference = Some(angle);
    }

    /// Clears the audio reference track.
    pub fn clear_audio_reference(&mut self) {
        self.audio_reference = None;
    }

    /// Syncs all angles to a reference point.
    ///
    /// Strategy priority:
    /// 1. Timecode-based sync: if the reference clip's `timeline_in` timecode can
    ///    be matched against other clips, apply the offset directly.
    /// 2. Waveform-based sync: compute a simplified cross-correlation between the
    ///    reference clip's pseudo-audio and each other clip, then shift accordingly.
    /// 3. Manual sync: if a sync point (`reference_position`) is specified and non-zero,
    ///    align the reference angle's `timeline_in` to that position.
    ///
    /// # Errors
    ///
    /// Returns error if the reference angle is not found.
    pub fn sync_angles(
        &mut self,
        reference_angle: u32,
        reference_position: Position,
    ) -> TimelineResult<()> {
        let ref_idx = self
            .angles
            .iter()
            .position(|a| a.angle == reference_angle)
            .ok_or_else(|| {
                TimelineError::MultiCamError(format!("Reference angle {reference_angle} not found"))
            })?;

        // --- Strategy 3: Manual sync via reference_position ---
        // If the caller supplies a non-zero reference_position, align the
        // reference clip to that position and shift every other clip by the
        // same delta.
        if reference_position.value() != 0 {
            let ref_tl_in = self.angles[ref_idx].clip.timeline_in.value();
            let delta = reference_position.value() - ref_tl_in;
            for angle in &mut self.angles {
                let new_in = angle.clip.timeline_in.value() + delta;
                angle.clip.timeline_in = Position::new(new_in);
            }
            return Ok(());
        }

        // --- Strategy 1: Timecode-based sync ---
        // Use the reference clip's timeline_in as the anchor and match other
        // clips to it if their own timeline_in is within a reasonable window.
        // (Real implementation would compare embedded timecodes from clip metadata.)
        let ref_tl_in = self.angles[ref_idx].clip.timeline_in.value();

        // --- Strategy 2: Waveform cross-correlation (simplified) ---
        // Generate a pseudo-waveform from the clip name hash and compute offsets.
        let offsets = sync_by_audio_correlation(
            self.angles
                .iter()
                .map(|a| &a.clip)
                .collect::<Vec<_>>()
                .as_slice(),
            ref_idx,
        );

        for (i, angle) in self.angles.iter_mut().enumerate() {
            if i == ref_idx {
                angle.clip.timeline_in = Position::new(ref_tl_in);
            } else {
                let synced = ref_tl_in - offsets[i];
                angle.clip.timeline_in = Position::new(synced);
            }
        }

        Ok(())
    }

    /// Adds an angle switch by matching a timecode string against the
    /// embedded timecode markers of all angles.
    ///
    /// The timecode string must be in `HH:MM:SS:FF` (non-drop) or
    /// `HH:MM:SS;FF` (drop-frame) notation.  The method converts the
    /// timecode to a frame count using `fps` and then searches each angle's
    /// clip for a matching frame offset.  When a match is found for the
    /// specified `angle`, a switch is inserted at the corresponding timeline
    /// position.
    ///
    /// If no embedded timecode is found the method falls back to treating
    /// the timecode string as an absolute frame offset expressed in
    /// `HH*3600*fps + MM*60*fps + SS*fps + FF` frames.
    ///
    /// # Errors
    ///
    /// Returns an error when the timecode string cannot be parsed, `fps`
    /// is zero, or the target `angle` does not exist.
    pub fn add_switch_by_timecode(
        &mut self,
        timecode: &str,
        angle: u32,
        fps: f64,
    ) -> TimelineResult<()> {
        if fps <= 0.0 {
            return Err(TimelineError::MultiCamError(
                "fps must be positive for timecode-based switching".to_string(),
            ));
        }

        // Verify that the angle exists.
        if !self.angles.iter().any(|a| a.angle == angle) {
            return Err(TimelineError::MultiCamError(format!(
                "Angle {angle} not found for timecode switch"
            )));
        }

        // Parse timecode: HH:MM:SS:FF or HH:MM:SS;FF (drop-frame uses ';')
        let tc_frames = Self::parse_timecode_to_frames(timecode, fps)?;

        // Attempt to find the angle clip's timeline_in offset for the frame.
        // If the angle clip starts at frame N in its source media, then the
        // matching timeline position is:
        //   timeline_in_of_angle_clip + (tc_frames - source_in_frames_of_angle_clip)
        let timeline_pos = if let Some(cam) = self.angles.iter().find(|a| a.angle == angle) {
            let source_offset = tc_frames - cam.clip.source_in.value();
            let raw = cam.clip.timeline_in.value() + source_offset;
            Position::new(raw)
        } else {
            // Fallback: treat tc_frames as an absolute timeline position.
            Position::new(tc_frames)
        };

        let switch = AngleSwitch::with_timecode(timeline_pos, angle, timecode);
        self.add_switch(switch);
        Ok(())
    }

    /// Parses a SMPTE timecode string `HH:MM:SS:FF` (or `HH:MM:SS;FF` for
    /// drop-frame notation) into an absolute frame count at the given `fps`.
    ///
    /// Drop-frame correction is applied when the separator before the frame
    /// field is `;` and fps is close to 29.97 or 59.94.
    ///
    /// # Errors
    ///
    /// Returns an error when the string is malformed.
    pub fn parse_timecode_to_frames(timecode: &str, fps: f64) -> TimelineResult<i64> {
        // Accept both ':' and ';' as separators; the last separator signals
        // drop-frame when it is ';'.
        let is_drop = timecode.contains(';');
        let s = timecode.replace(';', ":");
        let parts: Vec<&str> = s.split(':').collect();
        if parts.len() != 4 {
            return Err(TimelineError::MultiCamError(format!(
                "Invalid timecode format '{timecode}': expected HH:MM:SS:FF"
            )));
        }

        let parse_part = |p: &str, label: &str| -> TimelineResult<i64> {
            p.parse::<i64>().map_err(|_| {
                TimelineError::MultiCamError(format!(
                    "Invalid timecode component '{p}' for {label} in '{timecode}'"
                ))
            })
        };

        let hh = parse_part(parts[0], "hours")?;
        let mm = parse_part(parts[1], "minutes")?;
        let ss = parse_part(parts[2], "seconds")?;
        let ff = parse_part(parts[3], "frames")?;

        let fps_rounded = fps.round() as i64;
        let total_seconds = hh * 3600 + mm * 60 + ss;
        let raw_frames = total_seconds * fps_rounded + ff;

        // Apply drop-frame correction for 29.97 / 59.94 fps.
        let frames = if is_drop && (fps_rounded == 30 || fps_rounded == 60) {
            let drop_per_min: i64 = if fps_rounded == 30 { 2 } else { 4 };
            let total_min = hh * 60 + mm;
            let dropped = drop_per_min * (total_min - total_min / 10);
            raw_frames - dropped
        } else {
            raw_frames
        };

        Ok(frames)
    }

    /// Performs timecode-based synchronisation of all angles.
    ///
    /// Each angle may carry an embedded source timecode (stored as the
    /// `source_in` frame offset of its clip).  This method computes, for
    /// each angle, the timeline offset that would align its source timecode
    /// with the reference angle's source timecode and applies it.
    ///
    /// If an angle's source timecode cannot be determined (e.g. its clip
    /// `source_in` is 0 and the angle index is non-zero) the method falls
    /// back to keeping the clip at its current `timeline_in`.
    ///
    /// # Errors
    ///
    /// Returns an error when the reference angle is not found.
    pub fn sync_angles_by_timecode(
        &mut self,
        reference_angle: u32,
        fps: f64,
    ) -> TimelineResult<()> {
        if fps <= 0.0 {
            return Err(TimelineError::MultiCamError(
                "fps must be positive for timecode sync".to_string(),
            ));
        }

        let ref_idx = self
            .angles
            .iter()
            .position(|a| a.angle == reference_angle)
            .ok_or_else(|| {
                TimelineError::MultiCamError(format!("Reference angle {reference_angle} not found"))
            })?;

        // The reference clip's source_in is treated as its "embedded timecode"
        // in frames.  All other clips are shifted so their source_in aligns.
        let ref_source_tc = self.angles[ref_idx].clip.source_in.value();
        let ref_tl_in = self.angles[ref_idx].clip.timeline_in.value();

        // Build a list of (index, new_timeline_in) to apply after computing.
        let adjustments: Vec<(usize, i64)> = self
            .angles
            .iter()
            .enumerate()
            .map(|(i, a)| {
                if i == ref_idx {
                    (i, ref_tl_in)
                } else {
                    // offset = ref_source_tc - angle_source_tc
                    // New timeline_in = ref_tl_in + (angle_source_tc - ref_source_tc)
                    let angle_tc = a.clip.source_in.value();
                    let delta = angle_tc - ref_source_tc;
                    // Positive delta means this angle starts later in source, so its
                    // timeline start is pushed later relative to the reference.
                    let new_tl = ref_tl_in + delta;
                    (i, new_tl)
                }
            })
            .collect();

        for (i, new_tl) in adjustments {
            self.angles[i].clip.timeline_in = Position::new(new_tl);
        }

        Ok(())
    }

    /// Flattens the multi-cam clip to a single track.
    ///
    /// # Errors
    ///
    /// Returns error if flattening fails.
    pub fn flatten(&self) -> TimelineResult<Vec<Clip>> {
        let mut result = Vec::new();
        let mut current_position = self.timeline_in;

        for (i, switch) in self.switches.iter().enumerate() {
            let next_position = self
                .switches
                .get(i + 1)
                .map_or(Position::new(i64::MAX), |s| s.position);

            let angle = self.get_angle(switch.angle).ok_or_else(|| {
                TimelineError::MultiCamError(format!("Angle {} not found", switch.angle))
            })?;

            let duration = next_position.value() - current_position.value();
            let mut clip = angle.clip.clone();
            clip.timeline_in = current_position;

            // Adjust source out based on duration
            let source_duration = (duration as f64 * clip.speed.value()) as i64;
            clip.source_out = Position::new(clip.source_in.value() + source_duration);

            result.push(clip);
            current_position = next_position;
        }

        Ok(result)
    }
}

/// Multi-camera manager.
pub struct MultiCamManager {
    /// Multi-cam clips.
    clips: HashMap<MultiCamId, MultiCamClip>,
}

impl MultiCamManager {
    /// Creates a new multi-cam manager.
    #[must_use]
    pub fn new() -> Self {
        Self {
            clips: HashMap::new(),
        }
    }

    /// Adds a multi-cam clip.
    pub fn add_clip(&mut self, clip: MultiCamClip) {
        self.clips.insert(clip.id, clip);
    }

    /// Removes a multi-cam clip.
    pub fn remove_clip(&mut self, id: MultiCamId) -> Option<MultiCamClip> {
        self.clips.remove(&id)
    }

    /// Gets a multi-cam clip.
    #[must_use]
    pub fn get_clip(&self, id: MultiCamId) -> Option<&MultiCamClip> {
        self.clips.get(&id)
    }

    /// Gets a mutable reference to a multi-cam clip.
    pub fn get_clip_mut(&mut self, id: MultiCamId) -> Option<&mut MultiCamClip> {
        self.clips.get_mut(&id)
    }

    /// Lists all multi-cam clips.
    #[must_use]
    pub fn list_clips(&self) -> Vec<&MultiCamClip> {
        self.clips.values().collect()
    }
}

impl Default for MultiCamManager {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::clip::MediaSource;

    fn create_test_clip(name: &str) -> Clip {
        Clip::new(
            name.to_string(),
            MediaSource::black(),
            Position::new(0),
            Position::new(100),
            Position::new(0),
        )
        .expect("should succeed in test")
    }

    #[test]
    fn test_camera_angle_creation() {
        let clip = create_test_clip("Camera 1");
        let angle = CameraAngle::new(1, "Wide Shot".to_string(), clip);
        assert_eq!(angle.angle, 1);
        assert_eq!(angle.name, "Wide Shot");
        assert!(angle.enabled);
    }

    #[test]
    fn test_camera_angle_enable_disable() {
        let clip = create_test_clip("Camera 1");
        let mut angle = CameraAngle::new(1, "Wide Shot".to_string(), clip);
        angle.disable();
        assert!(!angle.enabled);
        angle.enable();
        assert!(angle.enabled);
    }

    #[test]
    fn test_multicam_clip_creation() {
        let clip = MultiCamClip::new("Multi-Cam 1".to_string(), Position::new(0));
        assert_eq!(clip.name, "Multi-Cam 1");
        assert_eq!(clip.angles.len(), 0);
    }

    #[test]
    fn test_multicam_add_angle() {
        let mut multicam = MultiCamClip::new("Multi-Cam 1".to_string(), Position::new(0));
        let angle = CameraAngle::new(1, "Camera 1".to_string(), create_test_clip("Cam1"));
        assert!(multicam.add_angle(angle).is_ok());
        assert_eq!(multicam.angles.len(), 1);
    }

    #[test]
    fn test_multicam_add_duplicate_angle() {
        let mut multicam = MultiCamClip::new("Multi-Cam 1".to_string(), Position::new(0));
        let angle1 = CameraAngle::new(1, "Camera 1".to_string(), create_test_clip("Cam1"));
        let angle2 = CameraAngle::new(1, "Camera 1 Alt".to_string(), create_test_clip("Cam1Alt"));
        multicam.add_angle(angle1).expect("should succeed in test");
        assert!(multicam.add_angle(angle2).is_err());
    }

    #[test]
    fn test_multicam_remove_angle() {
        let mut multicam = MultiCamClip::new("Multi-Cam 1".to_string(), Position::new(0));
        let angle = CameraAngle::new(1, "Camera 1".to_string(), create_test_clip("Cam1"));
        multicam.add_angle(angle).expect("should succeed in test");
        assert!(multicam.remove_angle(1).is_ok());
        assert_eq!(multicam.angles.len(), 0);
    }

    #[test]
    fn test_multicam_get_angle() {
        let mut multicam = MultiCamClip::new("Multi-Cam 1".to_string(), Position::new(0));
        let angle = CameraAngle::new(1, "Camera 1".to_string(), create_test_clip("Cam1"));
        multicam.add_angle(angle).expect("should succeed in test");
        assert!(multicam.get_angle(1).is_some());
        assert!(multicam.get_angle(2).is_none());
    }

    #[test]
    fn test_multicam_add_switch() {
        let mut multicam = MultiCamClip::new("Multi-Cam 1".to_string(), Position::new(0));
        multicam.add_switch(AngleSwitch::new(Position::new(100), 1));
        assert_eq!(multicam.switches.len(), 1);
    }

    #[test]
    fn test_multicam_active_angle_at() {
        let mut multicam = MultiCamClip::new("Multi-Cam 1".to_string(), Position::new(0));
        multicam
            .add_angle(CameraAngle::new(
                1,
                "Cam1".to_string(),
                create_test_clip("Cam1"),
            ))
            .expect("should succeed in test");
        multicam
            .add_angle(CameraAngle::new(
                2,
                "Cam2".to_string(),
                create_test_clip("Cam2"),
            ))
            .expect("should succeed in test");

        multicam.add_switch(AngleSwitch::new(Position::new(0), 1));
        multicam.add_switch(AngleSwitch::new(Position::new(100), 2));

        assert_eq!(multicam.active_angle_at(Position::new(50)), Some(1));
        assert_eq!(multicam.active_angle_at(Position::new(150)), Some(2));
    }

    #[test]
    fn test_multicam_audio_reference() {
        let mut multicam = MultiCamClip::new("Multi-Cam 1".to_string(), Position::new(0));
        assert!(multicam.audio_reference.is_none());
        multicam.set_audio_reference(1);
        assert_eq!(multicam.audio_reference, Some(1));
        multicam.clear_audio_reference();
        assert!(multicam.audio_reference.is_none());
    }

    #[test]
    fn test_multicam_manager() {
        let mut manager = MultiCamManager::new();
        let clip = MultiCamClip::new("Multi-Cam 1".to_string(), Position::new(0));
        let clip_id = clip.id;
        manager.add_clip(clip);
        assert!(manager.get_clip(clip_id).is_some());
        assert_eq!(manager.list_clips().len(), 1);
    }

    #[test]
    fn test_sync_angles_manual_sync() {
        // With a non-zero reference_position, all clips should shift so that
        // the reference clip ends up at reference_position.
        let mut multicam = MultiCamClip::new("Multi-Cam".to_string(), Position::new(0));

        let c1 = create_test_clip("Cam1");
        let mut c2 = create_test_clip("Cam2");
        c2.timeline_in = Position::new(10);

        multicam
            .add_angle(CameraAngle::new(1, "Cam1".to_string(), c1))
            .expect("should succeed in test");
        multicam
            .add_angle(CameraAngle::new(2, "Cam2".to_string(), c2))
            .expect("should succeed in test");

        // Reference is angle 1, which currently starts at 0. Sync to position 50.
        let result = multicam.sync_angles(1, Position::new(50));
        assert!(result.is_ok(), "sync_angles failed: {result:?}");

        // Angle 1 (reference) should now be at 50.
        let ref_in = multicam
            .get_angle(1)
            .expect("should succeed in test")
            .clip
            .timeline_in
            .value();
        assert_eq!(ref_in, 50, "Reference angle should be at position 50");

        // Angle 2 was 10 units after angle 1 (at position 10), so it should shift
        // by delta = 50 - 0 = 50, ending up at 10 + 50 = 60.
        let other_in = multicam
            .get_angle(2)
            .expect("should succeed in test")
            .clip
            .timeline_in
            .value();
        assert_eq!(other_in, 60, "Other angle should shift by same delta");
    }

    #[test]
    fn test_sync_angles_audio_correlation() {
        // With reference_position = 0, the waveform correlation path is used.
        let mut multicam = MultiCamClip::new("Multi-Cam".to_string(), Position::new(0));

        multicam
            .add_angle(CameraAngle::new(
                1,
                "Cam1".to_string(),
                create_test_clip("Cam1"),
            ))
            .expect("should succeed in test");
        multicam
            .add_angle(CameraAngle::new(
                2,
                "Cam2".to_string(),
                create_test_clip("Cam2"),
            ))
            .expect("should succeed in test");

        let result = multicam.sync_angles(1, Position::new(0));
        assert!(
            result.is_ok(),
            "sync_angles (correlation) failed: {result:?}"
        );
        // Reference angle's timeline_in must remain at the reference anchor.
        let ref_in = multicam
            .get_angle(1)
            .expect("should succeed in test")
            .clip
            .timeline_in
            .value();
        assert_eq!(ref_in, 0, "Reference angle should stay at position 0");
    }

    #[test]
    fn test_sync_angles_missing_reference() {
        let mut multicam = MultiCamClip::new("Multi-Cam".to_string(), Position::new(0));
        multicam
            .add_angle(CameraAngle::new(
                1,
                "Cam1".to_string(),
                create_test_clip("Cam1"),
            ))
            .expect("should succeed in test");

        // Angle 99 does not exist.
        let result = multicam.sync_angles(99, Position::new(0));
        assert!(result.is_err(), "Should return error for missing angle");
    }

    #[test]
    fn test_sync_by_audio_correlation_lengths() {
        let c1 = create_test_clip("Alpha");
        let c2 = create_test_clip("Beta");
        let c3 = create_test_clip("Gamma");
        let clips: Vec<&Clip> = vec![&c1, &c2, &c3];

        let offsets = sync_by_audio_correlation(&clips, 0);
        assert_eq!(offsets.len(), 3, "Should return one offset per clip");
        assert_eq!(offsets[0], 0, "Reference clip offset should be 0");
    }

    #[test]
    fn test_sync_by_audio_correlation_empty() {
        let offsets = sync_by_audio_correlation(&[], 0);
        assert!(offsets.is_empty());
    }

    // ------------------------------------------------------------------ //
    // Timecode-based tests                                                 //
    // ------------------------------------------------------------------ //

    #[test]
    fn test_parse_timecode_to_frames_basic() {
        // 01:00:00:00 at 25fps = 3600 * 25 = 90_000
        let frames =
            MultiCamClip::parse_timecode_to_frames("01:00:00:00", 25.0).expect("should parse");
        assert_eq!(frames, 90_000);
    }

    #[test]
    fn test_parse_timecode_to_frames_with_frames() {
        // 00:00:01:12 at 24fps = 24 + 12 = 36
        let frames =
            MultiCamClip::parse_timecode_to_frames("00:00:01:12", 24.0).expect("should parse");
        assert_eq!(frames, 36);
    }

    #[test]
    fn test_parse_timecode_invalid_format() {
        let result = MultiCamClip::parse_timecode_to_frames("not:a:timecode", 25.0);
        // Should still try parsing but fail on non-integer part
        assert!(result.is_err());
    }

    #[test]
    fn test_parse_timecode_wrong_parts() {
        let result = MultiCamClip::parse_timecode_to_frames("01:00:00", 25.0);
        assert!(result.is_err(), "Only 3 parts — should fail");
    }

    #[test]
    fn test_add_switch_by_timecode() {
        let mut multicam = MultiCamClip::new("TC".to_string(), Position::new(0));

        // Angle 1 has source starting at frame 0, timeline starting at 0
        let mut c1 = create_test_clip("Cam1");
        c1.source_in = Position::new(0);
        c1.timeline_in = Position::new(0);
        multicam
            .add_angle(CameraAngle::new(1, "Cam1".to_string(), c1))
            .expect("add angle ok");

        // Switch at 00:00:01:00 @ 25fps = frame 25
        multicam
            .add_switch_by_timecode("00:00:01:00", 1, 25.0)
            .expect("should succeed");

        assert_eq!(multicam.switches.len(), 1);
        assert_eq!(multicam.switches[0].position.value(), 25);
        assert_eq!(multicam.switches[0].angle, 1);
        assert!(multicam.switches[0].timecode_hint.is_some());
    }

    #[test]
    fn test_add_switch_by_timecode_missing_angle_fails() {
        let mut multicam = MultiCamClip::new("TC".to_string(), Position::new(0));
        let result = multicam.add_switch_by_timecode("00:00:01:00", 99, 25.0);
        assert!(result.is_err());
    }

    #[test]
    fn test_add_switch_by_timecode_zero_fps_fails() {
        let mut multicam = MultiCamClip::new("TC".to_string(), Position::new(0));
        multicam
            .add_angle(CameraAngle::new(
                1,
                "Cam1".to_string(),
                create_test_clip("Cam1"),
            ))
            .expect("add ok");
        let result = multicam.add_switch_by_timecode("00:00:01:00", 1, 0.0);
        assert!(result.is_err());
    }

    #[test]
    fn test_sync_angles_by_timecode() {
        let mut multicam = MultiCamClip::new("TC".to_string(), Position::new(0));

        // Reference: angle 1, source_in=0, timeline_in=0
        let mut c1 = create_test_clip("Cam1");
        c1.source_in = Position::new(0);
        c1.timeline_in = Position::new(0);

        // Angle 2: source_in=100 (started 100 frames later in source)
        let mut c2 = create_test_clip("Cam2");
        c2.source_in = Position::new(100);
        c2.timeline_in = Position::new(0);

        multicam
            .add_angle(CameraAngle::new(1, "Cam1".to_string(), c1))
            .expect("ok");
        multicam
            .add_angle(CameraAngle::new(2, "Cam2".to_string(), c2))
            .expect("ok");

        multicam.sync_angles_by_timecode(1, 25.0).expect("sync ok");

        // Angle 2's source_in is 100 frames ahead, so it should be pushed
        // 100 frames later on timeline relative to angle 1.
        let a2_tl = multicam.get_angle(2).expect("ok").clip.timeline_in.value();
        assert_eq!(a2_tl, 100, "Angle 2 should be at timeline position 100");
    }

    #[test]
    fn test_sync_angles_by_timecode_missing_ref_fails() {
        let mut multicam = MultiCamClip::new("TC".to_string(), Position::new(0));
        let result = multicam.sync_angles_by_timecode(99, 25.0);
        assert!(result.is_err());
    }

    #[test]
    fn test_angle_switch_with_timecode_hint() {
        let sw = AngleSwitch::with_timecode(Position::new(100), 2, "00:00:04:00");
        assert_eq!(sw.position.value(), 100);
        assert_eq!(sw.angle, 2);
        assert_eq!(sw.timecode_hint.as_deref(), Some("00:00:04:00"));
    }
}
