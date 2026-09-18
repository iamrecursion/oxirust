//! Extended multi-angle synchronization primitives.
//!
//! Provides [`SyncMethodExt`], [`AngleSyncPoint`], [`MultiAngleSyncResult`],
//! and [`SyncValidator`] for reliable multi-camera synchronization workflows.

#![allow(dead_code)]

// ── SyncMethodExt ─────────────────────────────────────────────────────────────

/// Method used to establish the synchronization point for a camera angle.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SyncMethodExt {
    /// SMPTE/LTC/VITC timecode embedded in the signal.
    Timecode,
    /// Audio transient (loudest peak) detection.
    AudioPeak,
    /// Slate/clapper-board flash detection.
    ClapperDetect,
    /// Manually entered frame offset.
    Manual,
}

impl SyncMethodExt {
    /// Returns `true` for methods that don't require human data entry.
    #[must_use]
    pub fn is_automatic(&self) -> bool {
        matches!(self, Self::Timecode | Self::AudioPeak | Self::ClapperDetect)
    }
}

// ── AngleSyncPoint ────────────────────────────────────────────────────────────

/// Synchronization data for a single camera angle.
#[derive(Debug, Clone)]
pub struct AngleSyncPoint {
    /// Camera identifier.
    pub camera_id: u32,
    /// The frame number within this angle's timeline that acts as the sync
    /// reference.
    pub sync_frame: u64,
    /// How the sync frame was determined.
    pub sync_method: SyncMethodExt,
    /// Confidence score in [0.0, 1.0].
    pub confidence: f32,
}

impl AngleSyncPoint {
    /// Returns `true` when the confidence exceeds 0.8.
    #[must_use]
    pub fn is_reliable(&self) -> bool {
        self.confidence > 0.8
    }
}

// ── MultiAngleSyncResult ──────────────────────────────────────────────────────

/// Aggregate synchronisation result across all camera angles.
#[derive(Debug, Clone)]
pub struct MultiAngleSyncResult {
    /// The camera that acts as the time reference (offset = 0).
    pub reference_camera: u32,
    /// Per-camera frame offsets: `(camera_id, frame_offset)`.
    ///
    /// A positive offset means the camera is *behind* the reference and frames
    /// should be advanced by that amount.
    pub offsets: Vec<(u32, i64)>,
}

impl MultiAngleSyncResult {
    /// Return the frame offset for `camera_id`, or `None` if not present.
    #[must_use]
    pub fn offset_for_camera(&self, camera_id: u32) -> Option<i64> {
        self.offsets
            .iter()
            .find(|(id, _)| *id == camera_id)
            .map(|(_, offset)| *offset)
    }

    /// Number of cameras that have a recorded offset.
    #[must_use]
    pub fn synchronized_camera_count(&self) -> usize {
        self.offsets.len()
    }
}

// ── SyncValidator ─────────────────────────────────────────────────────────────

/// Validates a [`MultiAngleSyncResult`] and reports anomalies.
pub struct SyncValidator;

impl SyncValidator {
    /// Return a list of warning messages for cameras whose absolute frame
    /// offset exceeds `max_offset_frames`.
    #[must_use]
    pub fn check(sync_result: &MultiAngleSyncResult, max_offset_frames: i64) -> Vec<String> {
        sync_result
            .offsets
            .iter()
            .filter(|(_, offset)| offset.abs() > max_offset_frames)
            .map(|(cam, offset)| {
                format!("Camera {cam} has offset {offset} frames (limit: {max_offset_frames})")
            })
            .collect()
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // SyncMethodExt ────────────────────────────────────────────────────────────

    #[test]
    fn test_timecode_is_automatic() {
        assert!(SyncMethodExt::Timecode.is_automatic());
    }

    #[test]
    fn test_audio_peak_is_automatic() {
        assert!(SyncMethodExt::AudioPeak.is_automatic());
    }

    #[test]
    fn test_clapper_detect_is_automatic() {
        assert!(SyncMethodExt::ClapperDetect.is_automatic());
    }

    #[test]
    fn test_manual_not_automatic() {
        assert!(!SyncMethodExt::Manual.is_automatic());
    }

    // AngleSyncPoint ───────────────────────────────────────────────────────────

    #[test]
    fn test_is_reliable_high_confidence() {
        let p = AngleSyncPoint {
            camera_id: 1,
            sync_frame: 100,
            sync_method: SyncMethodExt::Timecode,
            confidence: 0.95,
        };
        assert!(p.is_reliable());
    }

    #[test]
    fn test_is_reliable_low_confidence() {
        let p = AngleSyncPoint {
            camera_id: 2,
            sync_frame: 50,
            sync_method: SyncMethodExt::Manual,
            confidence: 0.5,
        };
        assert!(!p.is_reliable());
    }

    #[test]
    fn test_is_reliable_exactly_threshold_not_reliable() {
        // threshold is > 0.8, so 0.8 is NOT reliable
        let p = AngleSyncPoint {
            camera_id: 3,
            sync_frame: 0,
            sync_method: SyncMethodExt::AudioPeak,
            confidence: 0.8,
        };
        assert!(!p.is_reliable());
    }

    #[test]
    fn test_is_reliable_just_above_threshold() {
        let p = AngleSyncPoint {
            camera_id: 4,
            sync_frame: 0,
            sync_method: SyncMethodExt::ClapperDetect,
            confidence: 0.801,
        };
        assert!(p.is_reliable());
    }

    // MultiAngleSyncResult ─────────────────────────────────────────────────────

    fn make_result() -> MultiAngleSyncResult {
        MultiAngleSyncResult {
            reference_camera: 0,
            offsets: vec![(0, 0), (1, 10), (2, -5), (3, 25)],
        }
    }

    #[test]
    fn test_offset_for_camera_found() {
        let r = make_result();
        assert_eq!(r.offset_for_camera(1), Some(10));
    }

    #[test]
    fn test_offset_for_camera_negative() {
        let r = make_result();
        assert_eq!(r.offset_for_camera(2), Some(-5));
    }

    #[test]
    fn test_offset_for_camera_not_found() {
        let r = make_result();
        assert!(r.offset_for_camera(99).is_none());
    }

    #[test]
    fn test_synchronized_camera_count() {
        let r = make_result();
        assert_eq!(r.synchronized_camera_count(), 4);
    }

    #[test]
    fn test_synchronized_camera_count_empty() {
        let r = MultiAngleSyncResult {
            reference_camera: 0,
            offsets: vec![],
        };
        assert_eq!(r.synchronized_camera_count(), 0);
    }

    // SyncValidator ────────────────────────────────────────────────────────────

    #[test]
    fn test_check_no_violations() {
        let r = make_result(); // offsets: 0, 10, -5, 25
        let warnings = SyncValidator::check(&r, 30);
        assert!(warnings.is_empty());
    }

    #[test]
    fn test_check_detects_large_offset() {
        let r = make_result(); // camera 3 has offset 25
        let warnings = SyncValidator::check(&r, 20);
        assert_eq!(warnings.len(), 1);
        assert!(warnings[0].contains("Camera 3"));
    }

    #[test]
    fn test_check_detects_negative_large_offset() {
        let r = MultiAngleSyncResult {
            reference_camera: 0,
            offsets: vec![(0, 0), (1, -50)],
        };
        let warnings = SyncValidator::check(&r, 10);
        assert_eq!(warnings.len(), 1);
        assert!(warnings[0].contains("Camera 1"));
    }
}
