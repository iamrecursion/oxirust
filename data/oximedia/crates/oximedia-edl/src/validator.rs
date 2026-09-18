//! EDL validation and compliance checking.
//!
//! This module provides validation functionality to ensure EDLs conform
//! to CMX 3600 and other EDL format specifications.

use crate::error::{EdlError, EdlResult};
use crate::event::{EditType, EdlEvent};
use crate::Edl;
use std::collections::HashSet;

/// Validation level for EDL validation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ValidationLevel {
    /// Strict validation (CMX 3600 compliance).
    Strict,
    /// Standard validation (common issues).
    Standard,
    /// Lenient validation (minimal checks).
    Lenient,
}

/// EDL validator for checking compliance and consistency.
#[derive(Debug)]
pub struct EdlValidator {
    /// Validation level.
    pub level: ValidationLevel,
    /// Check for event overlaps.
    pub check_overlaps: bool,
    /// Check for timeline gaps.
    pub check_gaps: bool,
    /// Check timecode validity.
    pub check_timecodes: bool,
    /// Check event numbering.
    pub check_event_numbers: bool,
    /// Maximum allowed gap in frames (0 = no gaps allowed).
    pub max_gap_frames: u64,
}

impl EdlValidator {
    /// Create a new EDL validator with strict settings.
    #[must_use]
    pub const fn strict() -> Self {
        Self {
            level: ValidationLevel::Strict,
            check_overlaps: true,
            check_gaps: true,
            check_timecodes: true,
            check_event_numbers: true,
            max_gap_frames: 0,
        }
    }

    /// Create a new EDL validator with standard settings.
    #[must_use]
    pub const fn standard() -> Self {
        Self {
            level: ValidationLevel::Standard,
            check_overlaps: true,
            check_gaps: false,
            check_timecodes: true,
            check_event_numbers: true,
            max_gap_frames: 1,
        }
    }

    /// Create a new EDL validator with lenient settings.
    #[must_use]
    pub const fn lenient() -> Self {
        Self {
            level: ValidationLevel::Lenient,
            check_overlaps: false,
            check_gaps: false,
            check_timecodes: true,
            check_event_numbers: false,
            max_gap_frames: 100,
        }
    }

    /// Validate an EDL.
    ///
    /// # Errors
    ///
    /// Returns an error if the EDL fails validation.
    pub fn validate(&self, edl: &Edl) -> EdlResult<ValidationReport> {
        let mut report = ValidationReport::new();

        // Check if EDL has events
        if edl.events.is_empty() {
            report.add_warning("EDL has no events".to_string());
        }

        // Validate each event
        for event in &edl.events {
            if let Err(e) = self.validate_event(event) {
                report.add_error(format!("Event {}: {e}", event.number));
            }
        }

        // Check event numbering
        if self.check_event_numbers {
            if let Err(e) = self.check_event_numbering(edl) {
                report.add_error(format!("Event numbering: {e}"));
            }
        }

        // Check for overlaps
        if self.check_overlaps {
            if let Err(e) = self.check_event_overlaps(edl) {
                report.add_error(format!("Overlap detection: {e}"));
            }
        }

        // Check for gaps
        if self.check_gaps {
            if let Err(e) = self.check_timeline_gaps(edl) {
                report.add_error(format!("Gap detection: {e}"));
            }
        }

        // Validate reel table
        if let Err(e) = edl.reel_table.validate() {
            report.add_error(format!("Reel table: {e}"));
        }

        if report.has_errors() {
            Err(EdlError::ValidationError(format!(
                "Validation failed with {} errors",
                report.errors.len()
            )))
        } else {
            Ok(report)
        }
    }

    /// Validate a single event.
    fn validate_event(&self, event: &EdlEvent) -> EdlResult<()> {
        // Basic event validation
        event.validate()?;

        if self.check_timecodes {
            // Check that timecodes are in valid ranges
            if event.source_in.hours() > 23 {
                return Err(EdlError::InvalidTimecode {
                    line: 0,
                    message: format!("Invalid source in hours: {}", event.source_in.hours()),
                });
            }

            if event.record_in.hours() > 23 {
                return Err(EdlError::InvalidTimecode {
                    line: 0,
                    message: format!("Invalid record in hours: {}", event.record_in.hours()),
                });
            }
        }

        // Strict mode checks
        if self.level == ValidationLevel::Strict {
            // Check that reel names are <= 8 characters
            if event.reel.len() > 8 {
                return Err(EdlError::InvalidReelName(format!(
                    "Reel name too long (max 8 characters): {}",
                    event.reel
                )));
            }

            // Check that dissolves and wipes have transition durations
            if matches!(event.edit_type, EditType::Dissolve | EditType::Wipe)
                && event.transition_duration.is_none()
            {
                return Err(EdlError::MissingField(format!(
                    "Event {} missing transition duration",
                    event.number
                )));
            }
        }

        Ok(())
    }

    /// Check event numbering is sequential.
    fn check_event_numbering(&self, edl: &Edl) -> EdlResult<()> {
        let mut expected_num = 1;
        let mut seen_numbers = HashSet::new();

        for event in &edl.events {
            // Check for duplicates
            if !seen_numbers.insert(event.number) {
                return Err(EdlError::ValidationError(format!(
                    "Duplicate event number: {}",
                    event.number
                )));
            }

            // Check for sequential numbering (strict mode only)
            if self.level == ValidationLevel::Strict {
                if event.number != expected_num {
                    return Err(EdlError::ValidationError(format!(
                        "Non-sequential event numbering: expected {expected_num}, got {}",
                        event.number
                    )));
                }
                expected_num += 1;
            }
        }

        Ok(())
    }

    /// Check for event overlaps.
    fn check_event_overlaps(&self, edl: &Edl) -> EdlResult<()> {
        for i in 0..edl.events.len() {
            for j in (i + 1)..edl.events.len() {
                if edl.events[i].overlaps_with(&edl.events[j]) {
                    return Err(EdlError::event_overlap(
                        edl.events[i].number,
                        edl.events[j].number,
                    ));
                }
            }
        }
        Ok(())
    }

    /// Check for timeline gaps.
    fn check_timeline_gaps(&self, edl: &Edl) -> EdlResult<()> {
        // Sort events by record in timecode
        let mut sorted_events: Vec<&EdlEvent> = edl.events.iter().collect();
        sorted_events.sort_by_key(|e| e.record_in.to_frames());

        for i in 0..(sorted_events.len().saturating_sub(1)) {
            let current = sorted_events[i];
            let next = sorted_events[i + 1];

            // Check if same track
            if !current.track.overlaps_with(&next.track) {
                continue;
            }

            let gap = next.record_in.to_frames() as i64 - current.record_out.to_frames() as i64;

            if gap > self.max_gap_frames as i64 {
                return Err(EdlError::timeline_gap(current.number, next.number));
            } else if gap < 0 {
                return Err(EdlError::event_overlap(current.number, next.number));
            }
        }

        Ok(())
    }
}

impl Default for EdlValidator {
    fn default() -> Self {
        Self::standard()
    }
}

// ────────────────────────────────────────────────────────────────────────────
// Format-specific validation
// ────────────────────────────────────────────────────────────────────────────

/// Format-specific validation rules that differ between EDL format specs.
#[derive(Debug)]
pub struct FormatValidator {
    /// The target EDL format.
    pub format: crate::EdlFormat,
    /// Base validator settings.
    pub base: EdlValidator,
}

impl FormatValidator {
    /// Create a format-specific validator for the given format.
    ///
    /// Uses a base validator appropriate for the format:
    /// - CMX formats use strict base validation
    /// - GVG and Sony use standard base validation (more permissive reel names)
    #[must_use]
    pub fn new(format: crate::EdlFormat) -> Self {
        let base = match format {
            // GVG supports longer reel names, so don't use strict which checks 8-char limit
            crate::EdlFormat::Gvg => EdlValidator::standard(),
            _ => EdlValidator::strict(),
        };
        Self { format, base }
    }

    /// Create a format-specific validator with a custom base validator.
    #[must_use]
    pub fn with_base(format: crate::EdlFormat, base: EdlValidator) -> Self {
        Self { format, base }
    }

    /// Validate an EDL against format-specific rules.
    ///
    /// This runs the base validator first, then applies additional rules
    /// that are specific to the target format (CMX 3600, CMX 3400, GVG, Sony BVE-9000).
    ///
    /// # Errors
    ///
    /// Returns an error if the EDL fails format-specific validation.
    pub fn validate(&self, edl: &Edl) -> EdlResult<FormatValidationReport> {
        let mut report = FormatValidationReport {
            format: self.format,
            base_report: ValidationReport::new(),
            format_errors: Vec::new(),
            format_warnings: Vec::new(),
        };

        // Run base validation (collect errors/warnings rather than failing immediately)
        match self.base.validate(edl) {
            Ok(base) => report.base_report = base,
            Err(e) => report.base_report.add_error(format!("{e}")),
        }

        // Apply format-specific rules
        match self.format {
            crate::EdlFormat::Cmx3600 => self.validate_cmx3600(edl, &mut report),
            crate::EdlFormat::Cmx3400 => self.validate_cmx3400(edl, &mut report),
            crate::EdlFormat::Cmx340 => self.validate_cmx340(edl, &mut report),
            crate::EdlFormat::Gvg => self.validate_gvg(edl, &mut report),
            crate::EdlFormat::SonyBve9000 => self.validate_sony_bve9000(edl, &mut report),
        }

        if report.has_errors() {
            Err(EdlError::ValidationError(format!(
                "Format validation ({}) failed with {} errors",
                self.format,
                report.total_error_count()
            )))
        } else {
            Ok(report)
        }
    }

    /// CMX 3600 format-specific validation.
    ///
    /// Rules:
    /// - Reel names must be exactly 1-8 uppercase alphanumeric characters (or "AX"/"BL").
    /// - Event numbers must be in range 001-999.
    /// - Maximum of 999 events.
    /// - Only supports 24, 25, 29.97, 30 fps.
    fn validate_cmx3600(&self, edl: &Edl, report: &mut FormatValidationReport) {
        // Max 999 events
        if edl.events.len() > 999 {
            report.add_format_error(format!(
                "CMX 3600 supports max 999 events, EDL has {}",
                edl.events.len()
            ));
        }

        // Validate supported frame rates
        let supported_rates = [
            crate::timecode::EdlFrameRate::Fps24,
            crate::timecode::EdlFrameRate::Fps25,
            crate::timecode::EdlFrameRate::Fps2997DF,
            crate::timecode::EdlFrameRate::Fps2997NDF,
            crate::timecode::EdlFrameRate::Fps30,
        ];
        if !supported_rates.contains(&edl.frame_rate) {
            report.add_format_warning(format!(
                "CMX 3600 typically uses 24/25/29.97/30 fps, EDL uses {}",
                edl.frame_rate
            ));
        }

        // Validate each event
        for event in &edl.events {
            // Event number range
            if event.number > 999 {
                report.add_format_error(format!(
                    "CMX 3600 event number must be 001-999, got {}",
                    event.number
                ));
            }

            // Reel name: 1-8 chars, alphanumeric or "AX"/"BL"
            self.validate_cmx_reel_name(&event.reel, event.number, report);

            // Title max length: 72 characters (line width limit)
            if let Some(title) = &edl.title {
                if title.len() > 72 {
                    report.add_format_warning(format!(
                        "CMX 3600 title should be <= 72 characters, got {}",
                        title.len()
                    ));
                }
            }
        }
    }

    /// CMX 3400 format-specific validation.
    ///
    /// Rules (older, more restrictive):
    /// - Maximum of 400 events.
    /// - Reel names max 4 characters.
    /// - No wipe support (only Cut and Dissolve).
    fn validate_cmx3400(&self, edl: &Edl, report: &mut FormatValidationReport) {
        if edl.events.len() > 400 {
            report.add_format_error(format!(
                "CMX 3400 supports max 400 events, EDL has {}",
                edl.events.len()
            ));
        }

        for event in &edl.events {
            // Reel names max 4 chars
            if event.reel.len() > 4 {
                report.add_format_error(format!(
                    "CMX 3400 reel name max 4 chars: event {} has '{}'",
                    event.number, event.reel
                ));
            }

            // No wipe or key support
            if matches!(event.edit_type, EditType::Wipe | EditType::Key) {
                report.add_format_error(format!(
                    "CMX 3400 does not support {} transitions (event {})",
                    event.edit_type, event.number
                ));
            }
        }
    }

    /// CMX 340 format-specific validation (very restrictive).
    fn validate_cmx340(&self, edl: &Edl, report: &mut FormatValidationReport) {
        // Same as CMX 3400 but even more restrictive
        self.validate_cmx3400(edl, report);

        if edl.events.len() > 340 {
            report.add_format_error(format!(
                "CMX 340 supports max 340 events, EDL has {}",
                edl.events.len()
            ));
        }
    }

    /// GVG (Grass Valley Group) format-specific validation.
    ///
    /// Rules:
    /// - Supports extended reel names (up to 32 characters).
    /// - Supports additional transition types.
    /// - Max 9999 events.
    fn validate_gvg(&self, edl: &Edl, report: &mut FormatValidationReport) {
        if edl.events.len() > 9999 {
            report.add_format_error(format!(
                "GVG supports max 9999 events, EDL has {}",
                edl.events.len()
            ));
        }

        for event in &edl.events {
            if event.reel.len() > 32 {
                report.add_format_error(format!(
                    "GVG reel name max 32 chars: event {} has '{}' ({})",
                    event.number,
                    event.reel,
                    event.reel.len()
                ));
            }

            if event.number > 9999 {
                report.add_format_error(format!(
                    "GVG event number must be 0001-9999, got {}",
                    event.number
                ));
            }
        }
    }

    /// Sony BVE-9000 format-specific validation.
    ///
    /// Rules:
    /// - Reel names up to 8 characters.
    /// - Supports all transition types.
    /// - Max 999 events.
    /// - Requires drop-frame or non-drop-frame specification.
    fn validate_sony_bve9000(&self, edl: &Edl, report: &mut FormatValidationReport) {
        if edl.events.len() > 999 {
            report.add_format_error(format!(
                "Sony BVE-9000 supports max 999 events, EDL has {}",
                edl.events.len()
            ));
        }

        for event in &edl.events {
            if event.reel.len() > 8 {
                report.add_format_error(format!(
                    "Sony BVE-9000 reel name max 8 chars: event {} has '{}'",
                    event.number, event.reel
                ));
            }

            if event.number > 999 {
                report.add_format_error(format!(
                    "Sony BVE-9000 event number must be 001-999, got {}",
                    event.number
                ));
            }
        }
    }

    /// Validate a reel name for CMX format compliance.
    fn validate_cmx_reel_name(
        &self,
        reel: &str,
        event_number: u32,
        report: &mut FormatValidationReport,
    ) {
        if reel.is_empty() {
            report.add_format_error(format!(
                "Reel name cannot be empty (event {})",
                event_number
            ));
            return;
        }

        if reel.len() > 8 {
            report.add_format_error(format!(
                "CMX reel name max 8 chars: event {} has '{}' ({})",
                event_number,
                reel,
                reel.len()
            ));
        }

        // Check for valid characters (alphanumeric and underscore)
        let special_reels = ["AX", "BL", "AUX"];
        if !special_reels.contains(&reel) {
            for ch in reel.chars() {
                if !ch.is_ascii_alphanumeric() && ch != '_' && ch != '-' {
                    report.add_format_warning(format!(
                        "CMX reel name should be alphanumeric: event {} has '{}' (char '{}')",
                        event_number, reel, ch
                    ));
                    break;
                }
            }
        }
    }
}

/// Report from format-specific validation.
#[derive(Debug, Clone)]
pub struct FormatValidationReport {
    /// The target EDL format that was validated against.
    pub format: crate::EdlFormat,
    /// Results from base validation.
    pub base_report: ValidationReport,
    /// Format-specific errors.
    pub format_errors: Vec<String>,
    /// Format-specific warnings.
    pub format_warnings: Vec<String>,
}

impl FormatValidationReport {
    /// Add a format-specific error.
    pub fn add_format_error(&mut self, error: String) {
        self.format_errors.push(error);
    }

    /// Add a format-specific warning.
    pub fn add_format_warning(&mut self, warning: String) {
        self.format_warnings.push(warning);
    }

    /// Whether any errors exist (base or format-specific).
    #[must_use]
    pub fn has_errors(&self) -> bool {
        self.base_report.has_errors() || !self.format_errors.is_empty()
    }

    /// Whether any warnings exist (base or format-specific).
    #[must_use]
    pub fn has_warnings(&self) -> bool {
        self.base_report.has_warnings() || !self.format_warnings.is_empty()
    }

    /// Total number of errors.
    #[must_use]
    pub fn total_error_count(&self) -> usize {
        self.base_report.error_count() + self.format_errors.len()
    }

    /// Total number of warnings.
    #[must_use]
    pub fn total_warning_count(&self) -> usize {
        self.base_report.warning_count() + self.format_warnings.len()
    }

    /// Generate a human-readable report.
    #[must_use]
    pub fn to_report_string(&self) -> String {
        let mut lines = Vec::new();
        lines.push(format!("Format Validation Report ({})", self.format));
        lines.push(format!(
            "Errors: {} | Warnings: {}",
            self.total_error_count(),
            self.total_warning_count()
        ));

        if !self.base_report.errors.is_empty() {
            lines.push(String::new());
            lines.push("Base validation errors:".to_string());
            for err in &self.base_report.errors {
                lines.push(format!("  [E] {err}"));
            }
        }

        if !self.format_errors.is_empty() {
            lines.push(String::new());
            lines.push("Format-specific errors:".to_string());
            for err in &self.format_errors {
                lines.push(format!("  [E] {err}"));
            }
        }

        if !self.base_report.warnings.is_empty() {
            lines.push(String::new());
            lines.push("Base validation warnings:".to_string());
            for warn in &self.base_report.warnings {
                lines.push(format!("  [W] {warn}"));
            }
        }

        if !self.format_warnings.is_empty() {
            lines.push(String::new());
            lines.push("Format-specific warnings:".to_string());
            for warn in &self.format_warnings {
                lines.push(format!("  [W] {warn}"));
            }
        }

        lines.join("\n")
    }
}

/// Validation report containing errors and warnings.
#[derive(Debug, Clone, Default)]
pub struct ValidationReport {
    /// Validation errors.
    pub errors: Vec<String>,
    /// Validation warnings.
    pub warnings: Vec<String>,
}

impl ValidationReport {
    /// Create a new empty validation report.
    #[must_use]
    pub const fn new() -> Self {
        Self {
            errors: Vec::new(),
            warnings: Vec::new(),
        }
    }

    /// Add an error to the report.
    pub fn add_error(&mut self, error: String) {
        self.errors.push(error);
    }

    /// Add a warning to the report.
    pub fn add_warning(&mut self, warning: String) {
        self.warnings.push(warning);
    }

    /// Check if the report has any errors.
    #[must_use]
    pub fn has_errors(&self) -> bool {
        !self.errors.is_empty()
    }

    /// Check if the report has any warnings.
    #[must_use]
    pub fn has_warnings(&self) -> bool {
        !self.warnings.is_empty()
    }

    /// Get the number of errors.
    #[must_use]
    pub fn error_count(&self) -> usize {
        self.errors.len()
    }

    /// Get the number of warnings.
    #[must_use]
    pub fn warning_count(&self) -> usize {
        self.warnings.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::event::TrackType;
    use crate::timecode::{EdlFrameRate, EdlTimecode};
    use crate::EdlFormat;

    #[test]
    fn test_validate_empty_edl() {
        let edl = Edl::new(EdlFormat::Cmx3600);
        let validator = EdlValidator::lenient();
        let report = validator.validate(&edl).expect("validation should succeed");
        assert!(report.has_warnings());
    }

    #[test]
    fn test_validate_simple_edl() {
        let mut edl = Edl::new(EdlFormat::Cmx3600);
        edl.set_frame_rate(EdlFrameRate::Fps25);

        let tc1 = EdlTimecode::new(1, 0, 0, 0, EdlFrameRate::Fps25).expect("failed to create");
        let tc2 = EdlTimecode::new(1, 0, 5, 0, EdlFrameRate::Fps25).expect("failed to create");

        let event = EdlEvent::new(
            1,
            "A001".to_string(),
            TrackType::Video,
            EditType::Cut,
            tc1,
            tc2,
            tc1,
            tc2,
        );

        edl.add_event(event).expect("add_event should succeed");

        let validator = EdlValidator::standard();
        let report = validator.validate(&edl).expect("validation should succeed");
        assert!(!report.has_errors());
    }

    #[test]
    fn test_detect_overlap() {
        let mut edl = Edl::new(EdlFormat::Cmx3600);
        edl.set_frame_rate(EdlFrameRate::Fps25);

        let tc1 = EdlTimecode::new(1, 0, 0, 0, EdlFrameRate::Fps25).expect("failed to create");
        let tc2 = EdlTimecode::new(1, 0, 10, 0, EdlFrameRate::Fps25).expect("failed to create");
        let tc3 = EdlTimecode::new(1, 0, 5, 0, EdlFrameRate::Fps25).expect("failed to create");
        let tc4 = EdlTimecode::new(1, 0, 15, 0, EdlFrameRate::Fps25).expect("failed to create");

        let event1 = EdlEvent::new(
            1,
            "A001".to_string(),
            TrackType::Video,
            EditType::Cut,
            tc1,
            tc2,
            tc1,
            tc2,
        );

        let event2 = EdlEvent::new(
            2,
            "A002".to_string(),
            TrackType::Video,
            EditType::Cut,
            tc3,
            tc4,
            tc3,
            tc4,
        );

        edl.add_event(event1).expect("add_event should succeed");
        edl.add_event(event2).expect("add_event should succeed");

        let validator = EdlValidator::strict();
        assert!(validator.validate(&edl).is_err());
    }

    #[test]
    fn test_detect_gap() {
        let mut edl = Edl::new(EdlFormat::Cmx3600);
        edl.set_frame_rate(EdlFrameRate::Fps25);

        let tc1 = EdlTimecode::new(1, 0, 0, 0, EdlFrameRate::Fps25).expect("failed to create");
        let tc2 = EdlTimecode::new(1, 0, 5, 0, EdlFrameRate::Fps25).expect("failed to create");
        let tc3 = EdlTimecode::new(1, 0, 10, 0, EdlFrameRate::Fps25).expect("failed to create");
        let tc4 = EdlTimecode::new(1, 0, 15, 0, EdlFrameRate::Fps25).expect("failed to create");

        let event1 = EdlEvent::new(
            1,
            "A001".to_string(),
            TrackType::Video,
            EditType::Cut,
            tc1,
            tc2,
            tc1,
            tc2,
        );

        let event2 = EdlEvent::new(
            2,
            "A002".to_string(),
            TrackType::Video,
            EditType::Cut,
            tc3,
            tc4,
            tc3,
            tc4,
        );

        edl.add_event(event1).expect("add_event should succeed");
        edl.add_event(event2).expect("add_event should succeed");

        let validator = EdlValidator::strict();
        assert!(validator.validate(&edl).is_err());
    }

    #[test]
    fn test_event_numbering() {
        let mut edl = Edl::new(EdlFormat::Cmx3600);
        edl.set_frame_rate(EdlFrameRate::Fps25);

        let tc1 = EdlTimecode::new(1, 0, 0, 0, EdlFrameRate::Fps25).expect("failed to create");
        let tc2 = EdlTimecode::new(1, 0, 5, 0, EdlFrameRate::Fps25).expect("failed to create");

        // Non-sequential numbering
        let event1 = EdlEvent::new(
            1,
            "A001".to_string(),
            TrackType::Video,
            EditType::Cut,
            tc1,
            tc2,
            tc1,
            tc2,
        );

        let event2 = EdlEvent::new(
            3,
            "A002".to_string(),
            TrackType::Video,
            EditType::Cut,
            tc1,
            tc2,
            tc1,
            tc2,
        );

        edl.add_event(event1).expect("add_event should succeed");
        edl.add_event(event2).expect("add_event should succeed");

        let validator = EdlValidator::strict();
        assert!(validator.validate(&edl).is_err());

        // Lenient mode should allow this
        let validator = EdlValidator::lenient();
        assert!(validator.validate(&edl).is_ok());
    }

    // ── Format-specific validation tests ──

    #[test]
    fn test_format_validator_cmx3600_valid() {
        let mut edl = Edl::new(EdlFormat::Cmx3600);
        edl.set_frame_rate(EdlFrameRate::Fps25);
        let tc1 = EdlTimecode::new(1, 0, 0, 0, EdlFrameRate::Fps25).expect("failed to create");
        let tc2 = EdlTimecode::new(1, 0, 5, 0, EdlFrameRate::Fps25).expect("failed to create");
        let event = EdlEvent::new(
            1,
            "A001".to_string(),
            TrackType::Video,
            EditType::Cut,
            tc1,
            tc2,
            tc1,
            tc2,
        );
        edl.events.push(event);

        let fv = FormatValidator::new(EdlFormat::Cmx3600);
        let report = fv.validate(&edl).expect("validation should succeed");
        assert!(!report.has_errors());
    }

    #[test]
    fn test_format_validator_cmx3600_reel_too_long() {
        let mut edl = Edl::new(EdlFormat::Cmx3600);
        edl.set_frame_rate(EdlFrameRate::Fps25);
        let tc1 = EdlTimecode::new(1, 0, 0, 0, EdlFrameRate::Fps25).expect("failed to create");
        let tc2 = EdlTimecode::new(1, 0, 5, 0, EdlFrameRate::Fps25).expect("failed to create");
        let event = EdlEvent::new(
            1,
            "VERY_LONG_REEL_NAME".to_string(), // > 8 chars
            TrackType::Video,
            EditType::Cut,
            tc1,
            tc2,
            tc1,
            tc2,
        );
        edl.events.push(event);

        let fv = FormatValidator::new(EdlFormat::Cmx3600);
        let result = fv.validate(&edl);
        assert!(result.is_err());
    }

    #[test]
    fn test_format_validator_cmx3400_no_wipe() {
        let mut edl = Edl::new(EdlFormat::Cmx3400);
        edl.set_frame_rate(EdlFrameRate::Fps25);
        let tc1 = EdlTimecode::new(1, 0, 0, 0, EdlFrameRate::Fps25).expect("failed to create");
        let tc2 = EdlTimecode::new(1, 0, 5, 0, EdlFrameRate::Fps25).expect("failed to create");
        let mut event = EdlEvent::new(
            1,
            "R1".to_string(),
            TrackType::Video,
            EditType::Wipe,
            tc1,
            tc2,
            tc1,
            tc2,
        );
        event.set_transition_duration(15);
        event.set_wipe_pattern(crate::event::WipePattern::Horizontal);
        edl.events.push(event);

        let fv = FormatValidator::new(EdlFormat::Cmx3400);
        let result = fv.validate(&edl);
        assert!(result.is_err());
    }

    #[test]
    fn test_format_validator_cmx3400_reel_max_4() {
        let mut edl = Edl::new(EdlFormat::Cmx3400);
        edl.set_frame_rate(EdlFrameRate::Fps25);
        let tc1 = EdlTimecode::new(1, 0, 0, 0, EdlFrameRate::Fps25).expect("failed to create");
        let tc2 = EdlTimecode::new(1, 0, 5, 0, EdlFrameRate::Fps25).expect("failed to create");
        let event = EdlEvent::new(
            1,
            "ABCDE".to_string(), // > 4 chars
            TrackType::Video,
            EditType::Cut,
            tc1,
            tc2,
            tc1,
            tc2,
        );
        edl.events.push(event);

        let fv = FormatValidator::new(EdlFormat::Cmx3400);
        let result = fv.validate(&edl);
        assert!(result.is_err());
    }

    #[test]
    fn test_format_validator_gvg_long_reel_ok() {
        let mut edl = Edl::new(EdlFormat::Gvg);
        edl.set_frame_rate(EdlFrameRate::Fps25);
        let tc1 = EdlTimecode::new(1, 0, 0, 0, EdlFrameRate::Fps25).expect("failed to create");
        let tc2 = EdlTimecode::new(1, 0, 5, 0, EdlFrameRate::Fps25).expect("failed to create");
        let event = EdlEvent::new(
            1,
            "LONG_REEL_NAME_OK".to_string(), // up to 32 chars is fine for GVG
            TrackType::Video,
            EditType::Cut,
            tc1,
            tc2,
            tc1,
            tc2,
        );
        edl.events.push(event);

        let fv = FormatValidator::new(EdlFormat::Gvg);
        let report = fv.validate(&edl).expect("validation should succeed");
        assert!(!report.has_errors());
    }

    #[test]
    fn test_format_validator_gvg_too_long_reel() {
        let mut edl = Edl::new(EdlFormat::Gvg);
        edl.set_frame_rate(EdlFrameRate::Fps25);
        let tc1 = EdlTimecode::new(1, 0, 0, 0, EdlFrameRate::Fps25).expect("failed to create");
        let tc2 = EdlTimecode::new(1, 0, 5, 0, EdlFrameRate::Fps25).expect("failed to create");
        let event = EdlEvent::new(
            1,
            "THIS_REEL_NAME_IS_MUCH_TOO_LONG_FOR_GVG_FORMAT".to_string(),
            TrackType::Video,
            EditType::Cut,
            tc1,
            tc2,
            tc1,
            tc2,
        );
        edl.events.push(event);

        let fv = FormatValidator::new(EdlFormat::Gvg);
        let result = fv.validate(&edl);
        assert!(result.is_err());
    }

    #[test]
    fn test_format_validator_sony_valid() {
        let mut edl = Edl::new(EdlFormat::SonyBve9000);
        edl.set_frame_rate(EdlFrameRate::Fps25);
        let tc1 = EdlTimecode::new(1, 0, 0, 0, EdlFrameRate::Fps25).expect("failed to create");
        let tc2 = EdlTimecode::new(1, 0, 5, 0, EdlFrameRate::Fps25).expect("failed to create");
        let event = EdlEvent::new(
            1,
            "R001".to_string(),
            TrackType::Video,
            EditType::Cut,
            tc1,
            tc2,
            tc1,
            tc2,
        );
        edl.events.push(event);

        let fv = FormatValidator::new(EdlFormat::SonyBve9000);
        let report = fv.validate(&edl).expect("validation should succeed");
        assert!(!report.has_errors());
    }

    #[test]
    fn test_format_validator_report_string() {
        let mut edl = Edl::new(EdlFormat::Cmx3600);
        edl.set_frame_rate(EdlFrameRate::Fps25);
        let tc1 = EdlTimecode::new(1, 0, 0, 0, EdlFrameRate::Fps25).expect("failed to create");
        let tc2 = EdlTimecode::new(1, 0, 5, 0, EdlFrameRate::Fps25).expect("failed to create");
        let event = EdlEvent::new(
            1,
            "A001".to_string(),
            TrackType::Video,
            EditType::Cut,
            tc1,
            tc2,
            tc1,
            tc2,
        );
        edl.events.push(event);

        let fv = FormatValidator::new(EdlFormat::Cmx3600);
        let report = fv.validate(&edl).expect("validation should succeed");
        let report_str = report.to_report_string();
        assert!(report_str.contains("CMX 3600"));
        assert!(report_str.contains("Errors: 0"));
    }

    #[test]
    fn test_format_validator_cmx3600_unsupported_fps() {
        let mut edl = Edl::new(EdlFormat::Cmx3600);
        edl.set_frame_rate(EdlFrameRate::Fps60); // Non-standard for CMX
        let tc1 = EdlTimecode::new(1, 0, 0, 0, EdlFrameRate::Fps60).expect("failed to create");
        let tc2 = EdlTimecode::new(1, 0, 5, 0, EdlFrameRate::Fps60).expect("failed to create");
        let event = EdlEvent::new(
            1,
            "A001".to_string(),
            TrackType::Video,
            EditType::Cut,
            tc1,
            tc2,
            tc1,
            tc2,
        );
        edl.events.push(event);

        let fv = FormatValidator::new(EdlFormat::Cmx3600);
        let report = fv.validate(&edl).expect("validation should succeed");
        assert!(report.has_warnings());
        assert!(report.format_warnings.iter().any(|w| w.contains("60")));
    }

    #[test]
    fn test_format_validator_empty_edl() {
        let edl = Edl::new(EdlFormat::Cmx3600);
        let fv = FormatValidator::new(EdlFormat::Cmx3600);
        // Empty EDL should pass format validation but have base warnings
        let report = fv.validate(&edl).expect("validation should succeed");
        assert!(report.has_warnings());
    }

    #[test]
    fn test_format_validator_with_base() {
        let base = EdlValidator::lenient();
        let fv = FormatValidator::with_base(EdlFormat::Cmx3600, base);
        assert_eq!(fv.base.level, ValidationLevel::Lenient);
    }

    #[test]
    fn test_format_validation_report_counts() {
        let mut report = FormatValidationReport {
            format: EdlFormat::Cmx3600,
            base_report: ValidationReport::new(),
            format_errors: Vec::new(),
            format_warnings: Vec::new(),
        };
        report.base_report.add_error("base error".to_string());
        report.add_format_error("format error".to_string());
        report.add_format_warning("format warning".to_string());

        assert_eq!(report.total_error_count(), 2);
        assert_eq!(report.total_warning_count(), 1);
        assert!(report.has_errors());
        assert!(report.has_warnings());
    }

    #[test]
    fn test_format_validator_cmx_special_reels() {
        let mut edl = Edl::new(EdlFormat::Cmx3600);
        edl.set_frame_rate(EdlFrameRate::Fps25);
        let tc1 = EdlTimecode::new(1, 0, 0, 0, EdlFrameRate::Fps25).expect("failed to create");
        let tc2 = EdlTimecode::new(1, 0, 5, 0, EdlFrameRate::Fps25).expect("failed to create");

        // "AX" and "BL" are special reserved reel names
        let event = EdlEvent::new(
            1,
            "AX".to_string(),
            TrackType::Video,
            EditType::Cut,
            tc1,
            tc2,
            tc1,
            tc2,
        );
        edl.events.push(event);

        let fv = FormatValidator::new(EdlFormat::Cmx3600);
        let report = fv.validate(&edl).expect("validation should succeed");
        assert!(!report.has_errors());
    }
}
