//! Shot list export functionality.
//!
//! Supports the following output formats:
//! - **CSV** – simple spreadsheet-compatible format
//! - **JSON** – machine-readable serde serialisation
//! - **Detailed text** – human-readable report
//! - **FCP XML** – Final Cut Pro 7 / X compatible XML interchange format
//! - **Resolve Markers XML** – DaVinci Resolve marker import XML format

use crate::error::{ShotError, ShotResult};
use crate::types::{Scene, Shot};
use std::io::Write;

/// Shot list exporter.
pub struct ShotListExporter;

impl ShotListExporter {
    /// Create a new shot list exporter.
    #[must_use]
    pub const fn new() -> Self {
        Self
    }

    /// Export shots to CSV format.
    ///
    /// # Errors
    ///
    /// Returns error if writing fails.
    pub fn export_csv<W: Write>(&self, shots: &[Shot], writer: &mut W) -> ShotResult<()> {
        // Write header
        writeln!(
            writer,
            "Shot,Start,End,Duration,Type,Angle,Coverage,Transition,Confidence"
        )
        .map_err(|e| ShotError::ExportFailed(e.to_string()))?;

        // Write each shot
        for shot in shots {
            writeln!(
                writer,
                "{},{},{},{:.2},{},{},{},{},{:.2}",
                shot.id,
                shot.start.pts,
                shot.end.pts,
                shot.duration_seconds(),
                shot.shot_type.abbreviation(),
                shot.angle.name(),
                shot.coverage.name(),
                shot.transition.name(),
                shot.confidence
            )
            .map_err(|e| ShotError::ExportFailed(e.to_string()))?;
        }

        Ok(())
    }

    /// Export shots to JSON format.
    ///
    /// # Errors
    ///
    /// Returns error if serialization fails.
    pub fn export_json(&self, shots: &[Shot]) -> ShotResult<String> {
        serde_json::to_string_pretty(&shots)
            .map_err(|e| ShotError::SerializationError(e.to_string()))
    }

    /// Export scenes to JSON format.
    ///
    /// # Errors
    ///
    /// Returns error if serialization fails.
    pub fn export_scenes_json(&self, scenes: &[Scene]) -> ShotResult<String> {
        serde_json::to_string_pretty(&scenes)
            .map_err(|e| ShotError::SerializationError(e.to_string()))
    }

    /// Export comprehensive shot list with all metadata.
    ///
    /// # Errors
    ///
    /// Returns error if writing fails.
    pub fn export_detailed<W: Write>(&self, shots: &[Shot], writer: &mut W) -> ShotResult<()> {
        writeln!(writer, "SHOT LIST").map_err(|e| ShotError::ExportFailed(e.to_string()))?;
        writeln!(writer, "=========\n").map_err(|e| ShotError::ExportFailed(e.to_string()))?;

        for shot in shots {
            writeln!(writer, "Shot #{}", shot.id)
                .map_err(|e| ShotError::ExportFailed(e.to_string()))?;
            writeln!(
                writer,
                "  Time: {:.2}s - {:.2}s (Duration: {:.2}s)",
                shot.start.to_seconds(),
                shot.end.to_seconds(),
                shot.duration_seconds()
            )
            .map_err(|e| ShotError::ExportFailed(e.to_string()))?;
            writeln!(writer, "  Type: {}", shot.shot_type.name())
                .map_err(|e| ShotError::ExportFailed(e.to_string()))?;
            writeln!(writer, "  Angle: {}", shot.angle.name())
                .map_err(|e| ShotError::ExportFailed(e.to_string()))?;
            writeln!(writer, "  Coverage: {}", shot.coverage.name())
                .map_err(|e| ShotError::ExportFailed(e.to_string()))?;
            writeln!(writer, "  Transition: {}", shot.transition.name())
                .map_err(|e| ShotError::ExportFailed(e.to_string()))?;

            if !shot.movements.is_empty() {
                writeln!(writer, "  Movements:")
                    .map_err(|e| ShotError::ExportFailed(e.to_string()))?;
                for movement in &shot.movements {
                    writeln!(
                        writer,
                        "    - {} ({:.2}s - {:.2}s)",
                        movement.movement_type.name(),
                        movement.start,
                        movement.end
                    )
                    .map_err(|e| ShotError::ExportFailed(e.to_string()))?;
                }
            }

            writeln!(
                writer,
                "  Composition: Rule of Thirds: {:.2}, Symmetry: {:.2}, Balance: {:.2}",
                shot.composition.rule_of_thirds,
                shot.composition.symmetry,
                shot.composition.balance
            )
            .map_err(|e| ShotError::ExportFailed(e.to_string()))?;
            writeln!(writer, "  Confidence: {:.2}\n", shot.confidence)
                .map_err(|e| ShotError::ExportFailed(e.to_string()))?;
        }

        Ok(())
    }

    /// Export shots to FCP XML (Final Cut Pro XML interchange format).
    ///
    /// Produces a minimal but valid FCP XML document that can be imported by
    /// Final Cut Pro 7 and compatible applications.  Each shot is represented
    /// as a `<clip>` element inside a `<sequence>`.
    ///
    /// # Errors
    ///
    /// Returns error if writing fails.
    pub fn export_fcp_xml<W: Write>(&self, shots: &[Shot], writer: &mut W) -> ShotResult<()> {
        writeln!(writer, r#"<?xml version="1.0" encoding="UTF-8"?>"#)
            .map_err(|e| ShotError::ExportFailed(e.to_string()))?;
        writeln!(writer, r#"<!DOCTYPE xmeml>"#)
            .map_err(|e| ShotError::ExportFailed(e.to_string()))?;
        writeln!(writer, r#"<xmeml version="4">"#)
            .map_err(|e| ShotError::ExportFailed(e.to_string()))?;
        writeln!(writer, "  <sequence>").map_err(|e| ShotError::ExportFailed(e.to_string()))?;
        writeln!(writer, "    <name>Shot List</name>")
            .map_err(|e| ShotError::ExportFailed(e.to_string()))?;
        writeln!(writer, "    <media>").map_err(|e| ShotError::ExportFailed(e.to_string()))?;
        writeln!(writer, "      <video>").map_err(|e| ShotError::ExportFailed(e.to_string()))?;
        writeln!(writer, "        <track>").map_err(|e| ShotError::ExportFailed(e.to_string()))?;

        for shot in shots {
            let start_frames = shot.start.pts;
            let end_frames = shot.end.pts;
            let duration = (end_frames - start_frames).max(0);

            writeln!(writer, "          <clipitem id=\"shot-{}\">", shot.id)
                .map_err(|e| ShotError::ExportFailed(e.to_string()))?;
            writeln!(writer, "            <name>{}</name>", shot.shot_type.name())
                .map_err(|e| ShotError::ExportFailed(e.to_string()))?;
            writeln!(writer, "            <start>{start_frames}</start>")
                .map_err(|e| ShotError::ExportFailed(e.to_string()))?;
            writeln!(writer, "            <end>{end_frames}</end>")
                .map_err(|e| ShotError::ExportFailed(e.to_string()))?;
            writeln!(writer, "            <in>{start_frames}</in>")
                .map_err(|e| ShotError::ExportFailed(e.to_string()))?;
            writeln!(writer, "            <out>{end_frames}</out>")
                .map_err(|e| ShotError::ExportFailed(e.to_string()))?;
            writeln!(writer, "            <duration>{duration}</duration>")
                .map_err(|e| ShotError::ExportFailed(e.to_string()))?;
            writeln!(
                writer,
                "            <comments><comment>Type: {} | Angle: {} | Coverage: {} | Confidence: {:.2}</comment></comments>",
                shot.shot_type.name(),
                shot.angle.name(),
                shot.coverage.name(),
                shot.confidence
            )
            .map_err(|e| ShotError::ExportFailed(e.to_string()))?;
            writeln!(writer, "          </clipitem>")
                .map_err(|e| ShotError::ExportFailed(e.to_string()))?;
        }

        writeln!(writer, "        </track>").map_err(|e| ShotError::ExportFailed(e.to_string()))?;
        writeln!(writer, "      </video>").map_err(|e| ShotError::ExportFailed(e.to_string()))?;
        writeln!(writer, "    </media>").map_err(|e| ShotError::ExportFailed(e.to_string()))?;
        writeln!(writer, "  </sequence>").map_err(|e| ShotError::ExportFailed(e.to_string()))?;
        writeln!(writer, "</xmeml>").map_err(|e| ShotError::ExportFailed(e.to_string()))?;

        Ok(())
    }

    /// Export shots as DaVinci Resolve markers XML.
    ///
    /// Generates a Resolve-compatible marker list that can be imported via
    /// the *Markers* menu.  Each shot boundary becomes a marker with the shot
    /// type, angle, and coverage encoded in the note field.
    ///
    /// The `frame_rate` parameter is used to convert PTS values to timecode.
    /// Pass `30` for 30 fps, `25` for PAL, etc.
    ///
    /// # Errors
    ///
    /// Returns error if writing fails.
    pub fn export_resolve_markers_xml<W: Write>(
        &self,
        shots: &[Shot],
        frame_rate: u32,
        writer: &mut W,
    ) -> ShotResult<()> {
        let fps = frame_rate.max(1);

        writeln!(writer, r#"<?xml version="1.0" encoding="UTF-8"?>"#)
            .map_err(|e| ShotError::ExportFailed(e.to_string()))?;
        writeln!(writer, "<markers>").map_err(|e| ShotError::ExportFailed(e.to_string()))?;

        for shot in shots {
            let frame = shot.start.pts.max(0) as u64;
            let total_seconds = frame / fps as u64;
            let hours = total_seconds / 3600;
            let minutes = (total_seconds % 3600) / 60;
            let seconds = total_seconds % 60;
            let frames = frame % fps as u64;
            let timecode = format!("{hours:02}:{minutes:02}:{seconds:02}:{frames:02}");

            // Colour-code markers by shot type for visual organisation:
            // ECU/CU = Red (detail), MS/MCU = Green (dialogue), LS/ELS = Blue (wide)
            let color = match shot.shot_type {
                crate::types::ShotType::ExtremeCloseUp | crate::types::ShotType::CloseUp => "Red",
                crate::types::ShotType::MediumCloseUp | crate::types::ShotType::MediumShot => {
                    "Green"
                }
                crate::types::ShotType::MediumLongShot => "Cyan",
                crate::types::ShotType::LongShot | crate::types::ShotType::ExtremeLongShot => {
                    "Blue"
                }
                crate::types::ShotType::Unknown => "Yellow",
            };

            writeln!(writer, "  <marker>").map_err(|e| ShotError::ExportFailed(e.to_string()))?;
            writeln!(writer, "    <name>Shot {}</name>", shot.id)
                .map_err(|e| ShotError::ExportFailed(e.to_string()))?;
            writeln!(writer, "    <timecode>{timecode}</timecode>")
                .map_err(|e| ShotError::ExportFailed(e.to_string()))?;
            writeln!(writer, "    <frameNumber>{frame}</frameNumber>")
                .map_err(|e| ShotError::ExportFailed(e.to_string()))?;
            writeln!(writer, "    <color>{color}</color>")
                .map_err(|e| ShotError::ExportFailed(e.to_string()))?;
            writeln!(
                writer,
                "    <note>{} | {} | {} | conf={:.2}</note>",
                shot.shot_type.name(),
                shot.angle.name(),
                shot.coverage.name(),
                shot.confidence
            )
            .map_err(|e| ShotError::ExportFailed(e.to_string()))?;
            writeln!(writer, "  </marker>").map_err(|e| ShotError::ExportFailed(e.to_string()))?;
        }

        writeln!(writer, "</markers>").map_err(|e| ShotError::ExportFailed(e.to_string()))?;

        Ok(())
    }
}

impl Default for ShotListExporter {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::{CameraAngle, CompositionAnalysis, CoverageType, ShotType, TransitionType};
    use oximedia_core::types::{Rational, Timestamp};

    #[test]
    fn test_exporter_creation() {
        let _exporter = ShotListExporter::new();
    }

    #[test]
    fn test_export_csv() {
        let exporter = ShotListExporter::new();
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
        let result = exporter.export_csv(&[shot], &mut output);
        assert!(result.is_ok());
        assert!(!output.is_empty());
    }

    #[test]
    fn test_export_json() {
        let exporter = ShotListExporter::new();
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

        let result = exporter.export_json(&[shot]);
        assert!(result.is_ok());
    }

    fn make_test_shot(id: u64, start_pts: i64, end_pts: i64) -> Shot {
        Shot {
            id,
            start: Timestamp::new(start_pts, Rational::new(1, 30)),
            end: Timestamp::new(end_pts, Rational::new(1, 30)),
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
        }
    }

    // ---- FCP XML export tests ----

    #[test]
    fn test_export_fcp_xml_empty() {
        let exporter = ShotListExporter::new();
        let mut output = Vec::new();
        let result = exporter.export_fcp_xml(&[], &mut output);
        assert!(result.is_ok());
        let xml = String::from_utf8(output).expect("valid utf8");
        assert!(xml.contains("<xmeml"));
        assert!(xml.contains("</xmeml>"));
        assert!(xml.contains("<sequence>"));
    }

    #[test]
    fn test_export_fcp_xml_single_shot() {
        let exporter = ShotListExporter::new();
        let shot = make_test_shot(1, 0, 90);
        let mut output = Vec::new();
        let result = exporter.export_fcp_xml(&[shot], &mut output);
        assert!(result.is_ok());
        let xml = String::from_utf8(output).expect("valid utf8");
        assert!(xml.contains(r#"id="shot-1""#));
        assert!(xml.contains("<start>0</start>"));
        assert!(xml.contains("<end>90</end>"));
        assert!(xml.contains("<duration>90</duration>"));
        assert!(xml.contains("Medium Shot"));
    }

    #[test]
    fn test_export_fcp_xml_multiple_shots() {
        let exporter = ShotListExporter::new();
        let shots = vec![
            make_test_shot(1, 0, 30),
            make_test_shot(2, 30, 90),
            make_test_shot(3, 90, 150),
        ];
        let mut output = Vec::new();
        let result = exporter.export_fcp_xml(&shots, &mut output);
        assert!(result.is_ok());
        let xml = String::from_utf8(output).expect("valid utf8");
        assert!(xml.contains(r#"id="shot-1""#));
        assert!(xml.contains(r#"id="shot-2""#));
        assert!(xml.contains(r#"id="shot-3""#));
    }

    #[test]
    fn test_export_fcp_xml_contains_shot_metadata() {
        let exporter = ShotListExporter::new();
        let shot = Shot {
            id: 5,
            start: Timestamp::new(10, Rational::new(1, 25)),
            end: Timestamp::new(50, Rational::new(1, 25)),
            shot_type: ShotType::CloseUp,
            angle: CameraAngle::Low,
            movements: Vec::new(),
            composition: CompositionAnalysis {
                rule_of_thirds: 0.6,
                symmetry: 0.4,
                balance: 0.7,
                leading_lines: 0.3,
                depth: 0.5,
            },
            coverage: CoverageType::Single,
            confidence: 0.9,
            transition: TransitionType::Dissolve,
        };
        let mut output = Vec::new();
        exporter
            .export_fcp_xml(&[shot], &mut output)
            .expect("export should succeed");
        let xml = String::from_utf8(output).expect("valid utf8");
        // Shot metadata should appear in the comment node
        assert!(xml.contains("Close-up"));
        assert!(xml.contains("Low Angle"));
        assert!(xml.contains("conf=0.90") || xml.contains("Confidence: 0.90"));
    }

    // ---- Resolve markers XML export tests ----

    #[test]
    fn test_export_resolve_markers_xml_empty() {
        let exporter = ShotListExporter::new();
        let mut output = Vec::new();
        let result = exporter.export_resolve_markers_xml(&[], 30, &mut output);
        assert!(result.is_ok());
        let xml = String::from_utf8(output).expect("valid utf8");
        assert!(xml.contains("<markers>"));
        assert!(xml.contains("</markers>"));
    }

    #[test]
    fn test_export_resolve_markers_xml_single_shot() {
        let exporter = ShotListExporter::new();
        let shot = make_test_shot(1, 0, 90);
        let mut output = Vec::new();
        let result = exporter.export_resolve_markers_xml(&[shot], 30, &mut output);
        assert!(result.is_ok());
        let xml = String::from_utf8(output).expect("valid utf8");
        assert!(xml.contains("<marker>"));
        assert!(xml.contains("<name>Shot 1</name>"));
        assert!(xml.contains("<timecode>00:00:00:00</timecode>"));
        assert!(xml.contains("<frameNumber>0</frameNumber>"));
        assert!(xml.contains("<color>"));
        assert!(xml.contains("<note>"));
    }

    #[test]
    fn test_export_resolve_markers_xml_timecode() {
        let exporter = ShotListExporter::new();
        // Shot starting at frame 3600 with 30fps = 2:00:00
        let shot = make_test_shot(2, 3600, 3690);
        let mut output = Vec::new();
        exporter
            .export_resolve_markers_xml(&[shot], 30, &mut output)
            .expect("export ok");
        let xml = String::from_utf8(output).expect("valid utf8");
        assert!(xml.contains("<timecode>00:02:00:00</timecode>"));
    }

    #[test]
    fn test_export_resolve_markers_xml_color_coding() {
        let exporter = ShotListExporter::new();
        let mut ecu_shot = make_test_shot(1, 0, 30);
        ecu_shot.shot_type = ShotType::ExtremeCloseUp;
        let mut ls_shot = make_test_shot(2, 30, 60);
        ls_shot.shot_type = ShotType::LongShot;
        let mut output = Vec::new();
        exporter
            .export_resolve_markers_xml(&[ecu_shot, ls_shot], 25, &mut output)
            .expect("export ok");
        let xml = String::from_utf8(output).expect("valid utf8");
        assert!(xml.contains("<color>Red</color>"));
        assert!(xml.contains("<color>Blue</color>"));
    }

    #[test]
    fn test_export_resolve_markers_xml_multiple_shots() {
        let exporter = ShotListExporter::new();
        let shots: Vec<Shot> = (0..5i64)
            .map(|i| make_test_shot(i as u64, i * 30, (i + 1) * 30))
            .collect();
        let mut output = Vec::new();
        let result = exporter.export_resolve_markers_xml(&shots, 30, &mut output);
        assert!(result.is_ok());
        let xml = String::from_utf8(output).expect("valid utf8");
        // Should have 5 markers
        let marker_count = xml.matches("<marker>").count();
        assert_eq!(marker_count, 5);
    }
}
