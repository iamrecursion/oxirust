//! Quality checker for media files and conformed output.

use crate::error::{ConformError, ConformResult};
use crate::types::{ClipMatch, FrameRate};
use std::path::Path;

/// Quality checker for conform sessions.
pub struct QualityChecker;

impl QualityChecker {
    /// Create a new quality checker.
    #[must_use]
    pub const fn new() -> Self {
        Self
    }

    /// Check source format consistency.
    pub fn check_format_consistency(&self, matches: &[ClipMatch]) -> ConformResult<()> {
        if matches.is_empty() {
            return Ok(());
        }

        let first_fps = matches[0].clip.fps;
        for clip_match in matches {
            if clip_match.clip.fps.as_f64() != first_fps.as_f64() {
                return Err(ConformError::Validation(format!(
                    "Inconsistent frame rates: {} vs {}",
                    first_fps.as_f64(),
                    clip_match.clip.fps.as_f64()
                )));
            }
        }

        Ok(())
    }

    /// Check timecode continuity.
    pub fn check_timecode_continuity(&self, matches: &[ClipMatch]) -> ConformResult<()> {
        for window in matches.windows(2) {
            let prev = &window[0];
            let next = &window[1];

            if prev.clip.record_out != next.clip.record_in {
                return Err(ConformError::Validation(format!(
                    "Timecode gap between {} and {}: {} -> {}",
                    prev.clip.id, next.clip.id, prev.clip.record_out, next.clip.record_in
                )));
            }
        }

        Ok(())
    }

    /// Check audio sync.
    pub fn check_audio_sync(&self, _matches: &[ClipMatch]) -> ConformResult<()> {
        // Placeholder: would verify audio/video sync
        Ok(())
    }

    /// Verify file integrity.
    pub fn verify_file_integrity<P: AsRef<Path>>(&self, _path: P) -> ConformResult<bool> {
        // Placeholder: would check file integrity
        Ok(true)
    }

    /// Check frame rate consistency across all clips.
    pub fn check_frame_rate_consistency(&self, matches: &[ClipMatch]) -> ConformResult<FrameRate> {
        if matches.is_empty() {
            return Err(ConformError::Validation("No matches to check".to_string()));
        }

        let first_fps = matches[0].clip.fps;
        for clip_match in matches.iter().skip(1) {
            if (clip_match.clip.fps.as_f64() - first_fps.as_f64()).abs() > 0.01 {
                return Err(ConformError::Validation(format!(
                    "Frame rate mismatch: expected {}, found {}",
                    first_fps.as_f64(),
                    clip_match.clip.fps.as_f64()
                )));
            }
        }

        Ok(first_fps)
    }
}

impl Default for QualityChecker {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::{ClipReference, MatchMethod, MediaFile, Timecode, TrackType};
    use std::path::PathBuf;

    fn create_test_match(fps: FrameRate, record_in: Timecode, record_out: Timecode) -> ClipMatch {
        let clip = ClipReference {
            id: "test".to_string(),
            source_file: Some("test.mov".to_string()),
            source_in: Timecode::new(1, 0, 0, 0),
            source_out: Timecode::new(1, 0, 10, 0),
            record_in,
            record_out,
            track: TrackType::Video,
            fps,
            metadata: std::collections::HashMap::new(),
        };

        ClipMatch {
            clip,
            media: MediaFile::new(PathBuf::from("/test/file.mov")),
            score: 1.0,
            method: MatchMethod::ExactFilename,
            details: String::new(),
        }
    }

    #[test]
    fn test_format_consistency_pass() {
        let checker = QualityChecker::new();
        let matches = vec![
            create_test_match(
                FrameRate::Fps25,
                Timecode::new(1, 0, 0, 0),
                Timecode::new(1, 0, 10, 0),
            ),
            create_test_match(
                FrameRate::Fps25,
                Timecode::new(1, 0, 10, 0),
                Timecode::new(1, 0, 20, 0),
            ),
        ];

        assert!(checker.check_format_consistency(&matches).is_ok());
    }

    #[test]
    fn test_format_consistency_fail() {
        let checker = QualityChecker::new();
        let matches = vec![
            create_test_match(
                FrameRate::Fps25,
                Timecode::new(1, 0, 0, 0),
                Timecode::new(1, 0, 10, 0),
            ),
            create_test_match(
                FrameRate::Fps30,
                Timecode::new(1, 0, 10, 0),
                Timecode::new(1, 0, 20, 0),
            ),
        ];

        assert!(checker.check_format_consistency(&matches).is_err());
    }

    #[test]
    fn test_timecode_continuity_pass() {
        let checker = QualityChecker::new();
        let matches = vec![
            create_test_match(
                FrameRate::Fps25,
                Timecode::new(1, 0, 0, 0),
                Timecode::new(1, 0, 10, 0),
            ),
            create_test_match(
                FrameRate::Fps25,
                Timecode::new(1, 0, 10, 0),
                Timecode::new(1, 0, 20, 0),
            ),
        ];

        assert!(checker.check_timecode_continuity(&matches).is_ok());
    }

    #[test]
    fn test_timecode_continuity_fail() {
        let checker = QualityChecker::new();
        let matches = vec![
            create_test_match(
                FrameRate::Fps25,
                Timecode::new(1, 0, 0, 0),
                Timecode::new(1, 0, 10, 0),
            ),
            create_test_match(
                FrameRate::Fps25,
                Timecode::new(1, 0, 15, 0), // Gap!
                Timecode::new(1, 0, 20, 0),
            ),
        ];

        assert!(checker.check_timecode_continuity(&matches).is_err());
    }
}
