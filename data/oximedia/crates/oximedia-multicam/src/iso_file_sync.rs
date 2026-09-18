//! ISO file-based synchronisation and EDL export.
//!
//! Provides [`IsoFile`], [`IsoFileSyncSession`], and [`IsoEdlExport`] for
//! managing and exporting synchronised ISO recordings from multi-camera shoots.

#![allow(dead_code)]
#![allow(clippy::cast_precision_loss)]

use std::collections::HashMap;

// ── IsoFile ───────────────────────────────────────────────────────────────────

/// Metadata for a single ISO (isolated) camera recording file.
#[derive(Debug, Clone)]
pub struct IsoFile {
    /// Camera source identifier.
    pub camera_id: u32,
    /// Absolute path or URI to the media file.
    pub file_path: String,
    /// SMPTE start timecode as a string (HH:MM:SS:FF).
    pub start_timecode: String,
    /// Total duration in frames.
    pub duration_frames: u64,
    /// Frames per second of this recording.
    pub frame_rate: f32,
}

impl IsoFile {
    /// Frame index of the last frame (exclusive end).
    #[must_use]
    pub fn end_timecode_frames(&self) -> u64 {
        self.duration_frames
    }

    /// Duration in seconds.
    #[must_use]
    pub fn duration_seconds(&self) -> f64 {
        if self.frame_rate <= 0.0 {
            return 0.0;
        }
        self.duration_frames as f64 / f64::from(self.frame_rate)
    }
}

// ── IsoFileSyncSession ────────────────────────────────────────────────────────

/// A synchronisation session grouping multiple ISO files with per-camera frame
/// offsets.
#[derive(Debug, Default)]
pub struct IsoFileSyncSession {
    /// Unique session label.
    pub session_id: String,
    /// Registered ISO files.
    pub files: Vec<IsoFile>,
    /// Per-camera sync offsets in frames (`camera_id` → signed offset).
    sync_offsets: HashMap<u32, i64>,
}

impl IsoFileSyncSession {
    /// Create a new session with the given `session_id`.
    #[must_use]
    pub fn new(session_id: impl Into<String>) -> Self {
        Self {
            session_id: session_id.into(),
            ..Self::default()
        }
    }

    /// Register an ISO file with the session.
    pub fn add_iso(&mut self, file: IsoFile) {
        self.files.push(file);
    }

    /// Set the frame offset for `camera_id`.
    pub fn set_offset(&mut self, camera_id: u32, offset: i64) {
        self.sync_offsets.insert(camera_id, offset);
    }

    /// Number of cameras that have a recorded offset.
    #[must_use]
    pub fn synced_count(&self) -> usize {
        self.sync_offsets.len()
    }

    /// Convert a session-relative frame number to a file-relative frame
    /// number for the given camera.
    ///
    /// Returns `None` if the camera has no registered file or the computed
    /// frame would be negative.
    #[must_use]
    pub fn frame_at_session_offset(&self, camera_id: u32, session_frame: u64) -> Option<u64> {
        // Ensure the camera has a registered file; propagate None otherwise.
        if !self.files.iter().any(|f| f.camera_id == camera_id) {
            return None;
        }
        let offset = self.sync_offsets.get(&camera_id).copied().unwrap_or(0);
        let raw = session_frame as i64 + offset;
        if raw < 0 {
            None
        } else {
            Some(raw as u64)
        }
    }
}

// ── IsoEdlExport ─────────────────────────────────────────────────────────────

/// Generates simplified CMX 3600-style EDL lines from an [`IsoFileSyncSession`].
pub struct IsoEdlExport;

impl IsoEdlExport {
    /// Produce one EDL event line per ISO file in the session.
    ///
    /// Each line has the format:
    /// `<event> <reel> V C <src_in> <src_out> <rec_in> <rec_out>`
    ///
    /// Timecodes are formatted as `HH:MM:SS:FF` using integer arithmetic.
    #[must_use]
    pub fn to_edl_lines(session: &IsoFileSyncSession, frame_rate: f32) -> Vec<String> {
        let fps = frame_rate.max(1.0) as u64;
        session
            .files
            .iter()
            .enumerate()
            .map(|(idx, file)| {
                let event = idx + 1;
                let reel = format!("CAM{:03}", file.camera_id);
                let src_in = Self::frames_to_tc(0, fps);
                let src_out = Self::frames_to_tc(file.duration_frames, fps);
                let offset = session
                    .sync_offsets
                    .get(&file.camera_id)
                    .copied()
                    .unwrap_or(0);
                let rec_in_frames = if offset < 0 { 0u64 } else { offset as u64 };
                let rec_out_frames = rec_in_frames + file.duration_frames;
                let rec_in = Self::frames_to_tc(rec_in_frames, fps);
                let rec_out = Self::frames_to_tc(rec_out_frames, fps);
                format!("{event:03}  {reel}  V  C  {src_in} {src_out} {rec_in} {rec_out}")
            })
            .collect()
    }

    fn frames_to_tc(frames: u64, fps: u64) -> String {
        let fps = fps.max(1);
        let ff = frames % fps;
        let total_seconds = frames / fps;
        let ss = total_seconds % 60;
        let total_minutes = total_seconds / 60;
        let mm = total_minutes % 60;
        let hh = total_minutes / 60;
        format!("{hh:02}:{mm:02}:{ss:02}:{ff:02}")
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn make_file(camera_id: u32, duration_frames: u64) -> IsoFile {
        IsoFile {
            camera_id,
            file_path: format!("/media/cam{camera_id}.mxf"),
            start_timecode: "00:00:00:00".into(),
            duration_frames,
            frame_rate: 25.0,
        }
    }

    // IsoFile ──────────────────────────────────────────────────────────────────

    #[test]
    fn test_duration_seconds() {
        let f = make_file(1, 2500);
        assert!((f.duration_seconds() - 100.0).abs() < 1e-9);
    }

    #[test]
    fn test_duration_seconds_zero_fps() {
        let f = IsoFile {
            camera_id: 1,
            file_path: String::new(),
            start_timecode: String::new(),
            duration_frames: 100,
            frame_rate: 0.0,
        };
        assert_eq!(f.duration_seconds(), 0.0);
    }

    #[test]
    fn test_end_timecode_frames() {
        let f = make_file(2, 750);
        assert_eq!(f.end_timecode_frames(), 750);
    }

    // IsoFileSyncSession ───────────────────────────────────────────────────────

    #[test]
    fn test_add_iso_increases_file_count() {
        let mut session = IsoFileSyncSession::new("S1");
        session.add_iso(make_file(1, 500));
        session.add_iso(make_file(2, 600));
        assert_eq!(session.files.len(), 2);
    }

    #[test]
    fn test_set_offset_and_synced_count() {
        let mut session = IsoFileSyncSession::new("S2");
        session.add_iso(make_file(1, 100));
        session.set_offset(1, 10);
        assert_eq!(session.synced_count(), 1);
    }

    #[test]
    fn test_frame_at_session_offset_no_file_returns_none() {
        let session = IsoFileSyncSession::new("S3");
        assert!(session.frame_at_session_offset(99, 0).is_none());
    }

    #[test]
    fn test_frame_at_session_offset_positive_offset() {
        let mut session = IsoFileSyncSession::new("S4");
        session.add_iso(make_file(1, 1000));
        session.set_offset(1, 50); // camera lags 50 frames
                                   // session frame 0 → file frame 50
        assert_eq!(session.frame_at_session_offset(1, 0), Some(50));
    }

    #[test]
    fn test_frame_at_session_offset_negative_offset_in_range() {
        let mut session = IsoFileSyncSession::new("S5");
        session.add_iso(make_file(1, 1000));
        session.set_offset(1, -20);
        // session frame 100 → file frame 80
        assert_eq!(session.frame_at_session_offset(1, 100), Some(80));
    }

    #[test]
    fn test_frame_at_session_offset_negative_yields_none() {
        let mut session = IsoFileSyncSession::new("S6");
        session.add_iso(make_file(1, 1000));
        session.set_offset(1, -50);
        // session frame 0 → raw = -50 < 0 → None
        assert!(session.frame_at_session_offset(1, 0).is_none());
    }

    #[test]
    fn test_frame_at_session_offset_no_offset_recorded() {
        let mut session = IsoFileSyncSession::new("S7");
        session.add_iso(make_file(3, 500));
        // No offset set → defaults to 0
        assert_eq!(session.frame_at_session_offset(3, 100), Some(100));
    }

    // IsoEdlExport ─────────────────────────────────────────────────────────────

    #[test]
    fn test_to_edl_lines_count() {
        let mut session = IsoFileSyncSession::new("E1");
        session.add_iso(make_file(1, 100));
        session.add_iso(make_file(2, 200));
        let lines = IsoEdlExport::to_edl_lines(&session, 25.0);
        assert_eq!(lines.len(), 2);
    }

    #[test]
    fn test_to_edl_line_contains_reel() {
        let mut session = IsoFileSyncSession::new("E2");
        session.add_iso(make_file(5, 50));
        let lines = IsoEdlExport::to_edl_lines(&session, 25.0);
        assert!(lines[0].contains("CAM005"));
    }

    #[test]
    fn test_to_edl_line_starts_with_event_number() {
        let mut session = IsoFileSyncSession::new("E3");
        session.add_iso(make_file(1, 25));
        let lines = IsoEdlExport::to_edl_lines(&session, 25.0);
        assert!(lines[0].starts_with("001"));
    }

    #[test]
    fn test_to_edl_empty_session() {
        let session = IsoFileSyncSession::new("E4");
        let lines = IsoEdlExport::to_edl_lines(&session, 25.0);
        assert!(lines.is_empty());
    }

    #[test]
    fn test_to_edl_timecode_format() {
        let mut session = IsoFileSyncSession::new("E5");
        session.add_iso(make_file(1, 25)); // 1 second at 25 fps
        let lines = IsoEdlExport::to_edl_lines(&session, 25.0);
        // src_out should be 00:00:01:00
        assert!(lines[0].contains("00:00:01:00"));
    }
}
