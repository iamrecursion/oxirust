//! EDL export with shot metadata.

use crate::error::{ShotError, ShotResult};
use crate::types::Shot;
// use oximedia_timecode::Timecode; // unused
use std::io::Write;

/// EDL exporter with shot metadata.
pub struct EdlExporter {
    /// Frame rate for timecode conversion.
    framerate: f64,
}

impl EdlExporter {
    /// Create a new EDL exporter.
    #[must_use]
    pub const fn new(framerate: f64) -> Self {
        Self { framerate }
    }

    /// Export shots to CMX 3600 EDL format with shot metadata in comments.
    ///
    /// # Errors
    ///
    /// Returns error if writing fails.
    pub fn export_cmx3600<W: Write>(&self, shots: &[Shot], writer: &mut W) -> ShotResult<()> {
        // Write EDL header
        writeln!(writer, "TITLE: Shot List").map_err(|e| ShotError::ExportFailed(e.to_string()))?;
        writeln!(writer, "FCM: NON-DROP FRAME")
            .map_err(|e| ShotError::ExportFailed(e.to_string()))?;
        writeln!(writer).map_err(|e| ShotError::ExportFailed(e.to_string()))?;

        // Write each shot as an EDL event
        for (i, shot) in shots.iter().enumerate() {
            let event_num = i + 1;

            // Write shot metadata as comment
            writeln!(
                writer,
                "* SHOT {} - {} ({}) - {} - Confidence: {:.2}",
                shot.id,
                shot.shot_type.name(),
                shot.coverage.name(),
                shot.transition.name(),
                shot.confidence
            )
            .map_err(|e| ShotError::ExportFailed(e.to_string()))?;

            // Convert timestamps to timecodes
            let src_in = self.pts_to_timecode(shot.start.pts);
            let src_out = self.pts_to_timecode(shot.end.pts);
            let rec_in = self.pts_to_timecode(shot.start.pts);
            let rec_out = self.pts_to_timecode(shot.end.pts);

            // Write EDL event line
            writeln!(
                writer,
                "{:03}  AX       V     C        {} {} {} {}",
                event_num, src_in, src_out, rec_in, rec_out
            )
            .map_err(|e| ShotError::ExportFailed(e.to_string()))?;

            // Add composition metadata
            writeln!(
                writer,
                "* COMPOSITION: RoT={:.2} Sym={:.2} Bal={:.2}",
                shot.composition.rule_of_thirds,
                shot.composition.symmetry,
                shot.composition.balance
            )
            .map_err(|e| ShotError::ExportFailed(e.to_string()))?;

            writeln!(writer).map_err(|e| ShotError::ExportFailed(e.to_string()))?;
        }

        Ok(())
    }

    /// Convert PTS to timecode string.
    fn pts_to_timecode(&self, pts: i64) -> String {
        let total_frames = pts;
        let fps = self.framerate as i64;

        let hours = total_frames / (fps * 3600);
        let minutes = (total_frames % (fps * 3600)) / (fps * 60);
        let seconds = (total_frames % (fps * 60)) / fps;
        let frames = total_frames % fps;

        format!("{:02}:{:02}:{:02}:{:02}", hours, minutes, seconds, frames)
    }
}

impl Default for EdlExporter {
    fn default() -> Self {
        Self::new(30.0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::{CameraAngle, CompositionAnalysis, CoverageType, ShotType, TransitionType};
    use oximedia_core::types::{Rational, Timestamp};

    #[test]
    fn test_edl_exporter_creation() {
        let _exporter = EdlExporter::new(30.0);
    }

    #[test]
    fn test_export_cmx3600() {
        let exporter = EdlExporter::new(30.0);
        let shot = Shot {
            id: 1,
            start: Timestamp::new(0, Rational::new(1, 30)),
            end: Timestamp::new(60, Rational::new(1, 30)),
            shot_type: ShotType::MediumShot,
            angle: CameraAngle::EyeLevel,
            movements: Vec::new(),
            composition: CompositionAnalysis {
                rule_of_thirds: 0.5,
                symmetry: 0.5,
                balance: 0.5,
                leading_lines: 0.5,
                depth: 0.5,
            },
            coverage: CoverageType::Master,
            confidence: 0.8,
            transition: TransitionType::Cut,
        };

        let mut output = Vec::new();
        let result = exporter.export_cmx3600(&[shot], &mut output);
        assert!(result.is_ok());
        assert!(!output.is_empty());
    }

    #[test]
    fn test_pts_to_timecode() {
        let exporter = EdlExporter::new(30.0);
        let tc = exporter.pts_to_timecode(90); // 3 seconds at 30fps
        assert_eq!(tc, "00:00:03:00");
    }
}
