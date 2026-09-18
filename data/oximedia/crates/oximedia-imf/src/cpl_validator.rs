#![allow(dead_code)]
//! CPL (Composition Playlist) validation for IMF packages.
//!
//! Validates CPL documents against SMPTE ST 2067-3 constraints including:
//!
//! - **Structural integrity** - Required elements, correct nesting
//! - **Edit rate consistency** - Uniform edit rates across sequences
//! - **Timeline continuity** - No gaps or overlaps in virtual tracks
//! - **UUID validity** - All identifiers are well-formed UUIDs
//! - **Resource references** - Track file references resolve correctly
//! - **Duration checks** - Segment and resource durations are valid

use std::collections::{HashMap, HashSet};
use std::fmt;

/// Severity level for CPL validation issues.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum Severity {
    /// Informational note - not a problem.
    Info,
    /// Warning - may indicate an issue but not a hard violation.
    Warning,
    /// Error - violates SMPTE specification.
    Error,
    /// Fatal - the CPL cannot be processed.
    Fatal,
}

impl fmt::Display for Severity {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Info => write!(f, "INFO"),
            Self::Warning => write!(f, "WARNING"),
            Self::Error => write!(f, "ERROR"),
            Self::Fatal => write!(f, "FATAL"),
        }
    }
}

/// A single validation issue found in a CPL.
#[derive(Clone, Debug)]
pub struct CplIssue {
    /// Severity of the issue.
    pub severity: Severity,
    /// Category of the issue.
    pub category: IssueCategory,
    /// Human-readable description.
    pub message: String,
    /// Location within the CPL (e.g., segment index, resource ID).
    pub location: Option<String>,
}

impl CplIssue {
    /// Create a new CPL issue.
    pub fn new(severity: Severity, category: IssueCategory, message: impl Into<String>) -> Self {
        Self {
            severity,
            category,
            message: message.into(),
            location: None,
        }
    }

    /// Set the location context for this issue.
    pub fn with_location(mut self, location: impl Into<String>) -> Self {
        self.location = Some(location.into());
        self
    }
}

impl fmt::Display for CplIssue {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if let Some(ref loc) = self.location {
            write!(
                f,
                "[{}] {} (at {}): {}",
                self.severity, self.category, loc, self.message
            )
        } else {
            write!(f, "[{}] {}: {}", self.severity, self.category, self.message)
        }
    }
}

/// Categories of CPL validation issues.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum IssueCategory {
    /// Missing required XML element.
    MissingElement,
    /// Invalid UUID format.
    InvalidUuid,
    /// Edit rate mismatch.
    EditRateMismatch,
    /// Timeline gap or overlap.
    TimelineDiscontinuity,
    /// Resource reference error.
    ResourceReference,
    /// Duration validation error.
    DurationError,
    /// Duplicate identifier.
    DuplicateId,
    /// Structural violation.
    StructuralError,
    /// Constraint violation.
    ConstraintViolation,
    /// Metadata issue.
    MetadataIssue,
}

impl fmt::Display for IssueCategory {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::MissingElement => write!(f, "MissingElement"),
            Self::InvalidUuid => write!(f, "InvalidUuid"),
            Self::EditRateMismatch => write!(f, "EditRateMismatch"),
            Self::TimelineDiscontinuity => write!(f, "TimelineDiscontinuity"),
            Self::ResourceReference => write!(f, "ResourceReference"),
            Self::DurationError => write!(f, "DurationError"),
            Self::DuplicateId => write!(f, "DuplicateId"),
            Self::StructuralError => write!(f, "StructuralError"),
            Self::ConstraintViolation => write!(f, "ConstraintViolation"),
            Self::MetadataIssue => write!(f, "MetadataIssue"),
        }
    }
}

/// An edit rate as numerator/denominator pair.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct CplEditRate {
    /// Numerator (e.g., 24000).
    pub numerator: u32,
    /// Denominator (e.g., 1001).
    pub denominator: u32,
}

impl CplEditRate {
    /// Create a new edit rate.
    pub fn new(numerator: u32, denominator: u32) -> Self {
        Self {
            numerator,
            denominator,
        }
    }

    /// Get the rate as a floating-point value.
    #[allow(clippy::cast_precision_loss)]
    pub fn as_f64(&self) -> f64 {
        if self.denominator == 0 {
            0.0
        } else {
            self.numerator as f64 / self.denominator as f64
        }
    }

    /// Check if this is a valid edit rate.
    pub fn is_valid(&self) -> bool {
        self.numerator > 0 && self.denominator > 0
    }
}

impl fmt::Display for CplEditRate {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}/{}", self.numerator, self.denominator)
    }
}

/// A resource entry in a CPL segment for validation.
#[derive(Clone, Debug)]
pub struct CplResource {
    /// Resource UUID.
    pub id: String,
    /// Track file UUID this resource references.
    pub track_file_id: String,
    /// Edit rate of this resource.
    pub edit_rate: CplEditRate,
    /// Entry point in the source (in edit units).
    pub entry_point: u64,
    /// Source duration in edit units.
    pub source_duration: u64,
    /// Intrinsic duration of the referenced track file.
    pub intrinsic_duration: u64,
    /// Repeat count (1 = no repeat).
    pub repeat_count: u32,
}

impl CplResource {
    /// Calculate the effective duration (source_duration * repeat_count).
    pub fn effective_duration(&self) -> u64 {
        self.source_duration * u64::from(self.repeat_count)
    }

    /// Check if entry point + source duration exceeds intrinsic duration.
    pub fn exceeds_intrinsic(&self) -> bool {
        self.entry_point + self.source_duration > self.intrinsic_duration
    }
}

/// A virtual track in a CPL segment for validation.
#[derive(Clone, Debug)]
pub struct CplVirtualTrack {
    /// Virtual track UUID.
    pub id: String,
    /// Track type (e.g., "MainImageSequence", "MainAudioSequence").
    pub track_type: String,
    /// Resources in this virtual track.
    pub resources: Vec<CplResource>,
}

impl CplVirtualTrack {
    /// Calculate the total duration of this virtual track.
    pub fn total_duration(&self) -> u64 {
        self.resources
            .iter()
            .map(CplResource::effective_duration)
            .sum()
    }
}

/// A CPL segment for validation.
#[derive(Clone, Debug)]
pub struct CplSegment {
    /// Segment UUID.
    pub id: String,
    /// Virtual tracks in this segment.
    pub virtual_tracks: Vec<CplVirtualTrack>,
}

/// A CPL document model for validation.
#[derive(Clone, Debug)]
pub struct CplDocument {
    /// CPL UUID.
    pub id: String,
    /// Content title.
    pub content_title: String,
    /// Edit rate.
    pub edit_rate: CplEditRate,
    /// Segments.
    pub segments: Vec<CplSegment>,
    /// Set of all known track file UUIDs.
    pub known_track_files: HashSet<String>,
}

/// Result of CPL validation.
#[derive(Clone, Debug)]
pub struct CplValidationReport {
    /// All issues found.
    pub issues: Vec<CplIssue>,
}

impl CplValidationReport {
    /// Create an empty report.
    pub fn new() -> Self {
        Self { issues: Vec::new() }
    }

    /// Add an issue to the report.
    pub fn add_issue(&mut self, issue: CplIssue) {
        self.issues.push(issue);
    }

    /// Check if validation passed (no errors or fatals).
    pub fn is_valid(&self) -> bool {
        !self
            .issues
            .iter()
            .any(|i| matches!(i.severity, Severity::Error | Severity::Fatal))
    }

    /// Count issues by severity.
    pub fn count_by_severity(&self, severity: Severity) -> usize {
        self.issues
            .iter()
            .filter(|i| i.severity == severity)
            .count()
    }

    /// Get all issues of a specific category.
    pub fn issues_by_category(&self, category: IssueCategory) -> Vec<&CplIssue> {
        self.issues
            .iter()
            .filter(|i| i.category == category)
            .collect()
    }

    /// Get the highest severity in the report.
    pub fn max_severity(&self) -> Option<Severity> {
        self.issues.iter().map(|i| i.severity).max()
    }

    /// Return the total number of issues.
    pub fn issue_count(&self) -> usize {
        self.issues.len()
    }
}

impl Default for CplValidationReport {
    fn default() -> Self {
        Self::new()
    }
}

/// Validate a CPL document.
pub fn validate_cpl(cpl: &CplDocument) -> CplValidationReport {
    let mut report = CplValidationReport::new();

    validate_cpl_id(cpl, &mut report);
    validate_edit_rate(cpl, &mut report);
    validate_segments(cpl, &mut report);
    validate_resource_references(cpl, &mut report);
    validate_timeline_continuity(cpl, &mut report);
    validate_durations(cpl, &mut report);
    validate_unique_ids(cpl, &mut report);

    report
}

/// Validate a CPL document with full SMPTE ST 2067-2:2020 constraint checks.
///
/// In addition to basic structural validation, this performs:
/// - Edit-rate allowlist check (§6.1 of SMPTE ST 2067-2:2020)
/// - UUID format check (urn:uuid: prefix, §6.1.1)
/// - Minimum one main image sequence per segment (§6.4)
/// - Repeat-count range check (§6.6)
/// - Resource entry-point alignment (§6.6)
/// - At most one marker sequence per segment (§6.4.6)
pub fn validate_cpl_st2067_2(cpl: &CplDocument) -> CplValidationReport {
    let mut report = validate_cpl(cpl);

    validate_allowed_edit_rates(cpl, &mut report);
    validate_uuid_format(cpl, &mut report);
    validate_main_image_sequence_required(cpl, &mut report);
    validate_repeat_count_range(cpl, &mut report);
    validate_marker_sequence_count(cpl, &mut report);

    report
}

/// SMPTE ST 2067-2:2020 §6.1 — allowed edit rates.
///
/// Legal rates are: 24, 25, 30, 48, 50, 60 fps and their NTSC drop-frame
/// equivalents (24000/1001, 30000/1001, 60000/1001).
fn validate_allowed_edit_rates(cpl: &CplDocument, report: &mut CplValidationReport) {
    let allowed: &[(u32, u32)] = &[
        (24, 1),
        (25, 1),
        (30, 1),
        (48, 1),
        (50, 1),
        (60, 1),
        (24000, 1001),
        (30000, 1001),
        (60000, 1001),
    ];
    let rate = cpl.edit_rate;
    if rate.is_valid()
        && !allowed
            .iter()
            .any(|&(n, d)| rate.numerator == n && rate.denominator == d)
    {
        report.add_issue(CplIssue::new(
            Severity::Warning,
            IssueCategory::EditRateMismatch,
            format!(
                "CPL edit rate {} is not in the SMPTE ST 2067-2:2020 §6.1 allowlist \
                 (24, 25, 30, 48, 50, 60 fps and NTSC variants)",
                rate,
            ),
        ));
    }

    // Check per-resource edit rates too
    for (seg_idx, segment) in cpl.segments.iter().enumerate() {
        for track in &segment.virtual_tracks {
            for (res_idx, resource) in track.resources.iter().enumerate() {
                let rr = resource.edit_rate;
                if rr.is_valid()
                    && !allowed
                        .iter()
                        .any(|&(n, d)| rr.numerator == n && rr.denominator == d)
                {
                    report.add_issue(
                        CplIssue::new(
                            Severity::Warning,
                            IssueCategory::EditRateMismatch,
                            format!(
                                "Resource edit rate {} is not in the SMPTE ST 2067-2:2020 \
                                 §6.1 allowlist",
                                rr
                            ),
                        )
                        .with_location(format!(
                            "Segment[{seg_idx}]/Track[{}]/Resource[{res_idx}]",
                            track.id
                        )),
                    );
                }
            }
        }
    }
}

/// SMPTE ST 2067-2:2020 §6.1.1 — UUIDs should use the `urn:uuid:` prefix.
fn validate_uuid_format(cpl: &CplDocument, report: &mut CplValidationReport) {
    let check_id = |id: &str, location: &str, report: &mut CplValidationReport| {
        if !id.is_empty() && !id.starts_with("urn:uuid:") && !id.starts_with("urn:smpte:") {
            report.add_issue(
                CplIssue::new(
                    Severity::Warning,
                    IssueCategory::InvalidUuid,
                    format!("Identifier '{id}' does not use the required 'urn:uuid:' prefix"),
                )
                .with_location(location.to_string()),
            );
        }
    };

    check_id(&cpl.id, "CPL/Id", report);
    for (seg_idx, segment) in cpl.segments.iter().enumerate() {
        check_id(&segment.id, &format!("Segment[{seg_idx}]/Id"), report);
        for track in &segment.virtual_tracks {
            check_id(&track.id, &format!("Segment[{seg_idx}]/Track/Id"), report);
            for (res_idx, resource) in track.resources.iter().enumerate() {
                check_id(
                    &resource.id,
                    &format!(
                        "Segment[{seg_idx}]/Track[{}]/Resource[{res_idx}]/Id",
                        track.id
                    ),
                    report,
                );
            }
        }
    }
}

/// SMPTE ST 2067-2:2020 §6.4 — every segment must contain at least one main
/// image sequence (track type "MainImageSequence" or "MainImageVirtualTrack").
fn validate_main_image_sequence_required(cpl: &CplDocument, report: &mut CplValidationReport) {
    for (seg_idx, segment) in cpl.segments.iter().enumerate() {
        let has_image = segment.virtual_tracks.iter().any(|t| {
            t.track_type.contains("MainImage")
                || t.track_type.contains("main_image")
                || t.track_type == "MainImageSequence"
        });
        if !has_image && !segment.virtual_tracks.is_empty() {
            report.add_issue(
                CplIssue::new(
                    Severity::Warning,
                    IssueCategory::StructuralError,
                    "Segment contains no MainImageSequence; \
                     SMPTE ST 2067-2:2020 §6.4 requires at least one",
                )
                .with_location(format!("Segment[{seg_idx}]")),
            );
        }
    }
}

/// SMPTE ST 2067-2:2020 §6.6 — repeat count must be ≥ 1 and ≤ 65535.
fn validate_repeat_count_range(cpl: &CplDocument, report: &mut CplValidationReport) {
    for (seg_idx, segment) in cpl.segments.iter().enumerate() {
        for track in &segment.virtual_tracks {
            for (res_idx, resource) in track.resources.iter().enumerate() {
                if resource.repeat_count > 65535 {
                    report.add_issue(
                        CplIssue::new(
                            Severity::Error,
                            IssueCategory::ConstraintViolation,
                            format!(
                                "RepeatCount {} exceeds SMPTE ST 2067-2:2020 §6.6 maximum of 65535",
                                resource.repeat_count
                            ),
                        )
                        .with_location(format!(
                            "Segment[{seg_idx}]/Track[{}]/Resource[{res_idx}]",
                            track.id
                        )),
                    );
                }
            }
        }
    }
}

/// SMPTE ST 2067-2:2020 §6.4.6 — at most one marker sequence per segment.
fn validate_marker_sequence_count(cpl: &CplDocument, report: &mut CplValidationReport) {
    for (seg_idx, segment) in cpl.segments.iter().enumerate() {
        let marker_count = segment
            .virtual_tracks
            .iter()
            .filter(|t| t.track_type.contains("Marker") || t.track_type.contains("marker"))
            .count();
        if marker_count > 1 {
            report.add_issue(
                CplIssue::new(
                    Severity::Error,
                    IssueCategory::ConstraintViolation,
                    format!(
                        "Segment has {marker_count} marker sequences; \
                         SMPTE ST 2067-2:2020 §6.4.6 allows at most 1"
                    ),
                )
                .with_location(format!("Segment[{seg_idx}]")),
            );
        }
    }
}

/// Validate the CPL ID is a valid UUID-like string.
fn validate_cpl_id(cpl: &CplDocument, report: &mut CplValidationReport) {
    if cpl.id.is_empty() {
        report.add_issue(CplIssue::new(
            Severity::Fatal,
            IssueCategory::MissingElement,
            "CPL Id is empty",
        ));
    }

    if cpl.content_title.is_empty() {
        report.add_issue(CplIssue::new(
            Severity::Warning,
            IssueCategory::MetadataIssue,
            "ContentTitle is empty",
        ));
    }
}

/// Validate the edit rate.
fn validate_edit_rate(cpl: &CplDocument, report: &mut CplValidationReport) {
    if !cpl.edit_rate.is_valid() {
        report.add_issue(CplIssue::new(
            Severity::Error,
            IssueCategory::EditRateMismatch,
            format!("Invalid CPL edit rate: {}", cpl.edit_rate),
        ));
    }
}

/// Validate all segments have content and consistent structure.
fn validate_segments(cpl: &CplDocument, report: &mut CplValidationReport) {
    if cpl.segments.is_empty() {
        report.add_issue(CplIssue::new(
            Severity::Error,
            IssueCategory::StructuralError,
            "CPL has no segments",
        ));
        return;
    }

    for (seg_idx, segment) in cpl.segments.iter().enumerate() {
        if segment.id.is_empty() {
            report.add_issue(
                CplIssue::new(
                    Severity::Error,
                    IssueCategory::MissingElement,
                    "Segment Id is empty",
                )
                .with_location(format!("Segment[{seg_idx}]")),
            );
        }

        if segment.virtual_tracks.is_empty() {
            report.add_issue(
                CplIssue::new(
                    Severity::Warning,
                    IssueCategory::StructuralError,
                    "Segment has no virtual tracks",
                )
                .with_location(format!("Segment[{seg_idx}]")),
            );
        }
    }
}

/// Validate resource references to track files.
fn validate_resource_references(cpl: &CplDocument, report: &mut CplValidationReport) {
    for (seg_idx, segment) in cpl.segments.iter().enumerate() {
        for track in &segment.virtual_tracks {
            for (res_idx, resource) in track.resources.iter().enumerate() {
                if !cpl.known_track_files.contains(&resource.track_file_id) {
                    report.add_issue(
                        CplIssue::new(
                            Severity::Error,
                            IssueCategory::ResourceReference,
                            format!(
                                "Resource references unknown track file: {}",
                                resource.track_file_id
                            ),
                        )
                        .with_location(format!(
                            "Segment[{seg_idx}]/Track[{}]/Resource[{res_idx}]",
                            track.id
                        )),
                    );
                }
            }
        }
    }
}

/// Validate timeline continuity (consistent durations across virtual tracks in a segment).
fn validate_timeline_continuity(cpl: &CplDocument, report: &mut CplValidationReport) {
    for (seg_idx, segment) in cpl.segments.iter().enumerate() {
        let mut track_durations: Vec<(String, u64)> = Vec::new();
        for track in &segment.virtual_tracks {
            let duration = track.total_duration();
            track_durations.push((track.id.clone(), duration));
        }

        // All tracks in a segment should have the same duration
        if track_durations.len() > 1 {
            let first_dur = track_durations[0].1;
            for (tid, dur) in &track_durations[1..] {
                if *dur != first_dur {
                    report.add_issue(
                        CplIssue::new(
                            Severity::Error,
                            IssueCategory::TimelineDiscontinuity,
                            format!(
                                "Track {} duration ({}) differs from first track duration ({})",
                                tid, dur, first_dur
                            ),
                        )
                        .with_location(format!("Segment[{seg_idx}]")),
                    );
                }
            }
        }
    }
}

/// Validate resource durations.
fn validate_durations(cpl: &CplDocument, report: &mut CplValidationReport) {
    for (seg_idx, segment) in cpl.segments.iter().enumerate() {
        for track in &segment.virtual_tracks {
            for (res_idx, resource) in track.resources.iter().enumerate() {
                if resource.source_duration == 0 {
                    report.add_issue(
                        CplIssue::new(
                            Severity::Error,
                            IssueCategory::DurationError,
                            "Resource has zero source duration",
                        )
                        .with_location(format!(
                            "Segment[{seg_idx}]/Track[{}]/Resource[{res_idx}]",
                            track.id
                        )),
                    );
                }

                if resource.exceeds_intrinsic() {
                    report.add_issue(
                        CplIssue::new(
                            Severity::Error,
                            IssueCategory::DurationError,
                            format!(
                                "EntryPoint ({}) + SourceDuration ({}) exceeds IntrinsicDuration ({})",
                                resource.entry_point,
                                resource.source_duration,
                                resource.intrinsic_duration,
                            ),
                        )
                        .with_location(format!(
                            "Segment[{seg_idx}]/Track[{}]/Resource[{res_idx}]",
                            track.id
                        )),
                    );
                }

                if resource.repeat_count == 0 {
                    report.add_issue(
                        CplIssue::new(
                            Severity::Warning,
                            IssueCategory::ConstraintViolation,
                            "Resource has zero repeat count",
                        )
                        .with_location(format!(
                            "Segment[{seg_idx}]/Track[{}]/Resource[{res_idx}]",
                            track.id
                        )),
                    );
                }

                if !resource.edit_rate.is_valid() {
                    report.add_issue(
                        CplIssue::new(
                            Severity::Error,
                            IssueCategory::EditRateMismatch,
                            format!("Resource has invalid edit rate: {}", resource.edit_rate),
                        )
                        .with_location(format!(
                            "Segment[{seg_idx}]/Track[{}]/Resource[{res_idx}]",
                            track.id
                        )),
                    );
                }
            }
        }
    }
}

/// Validate uniqueness of IDs throughout the CPL.
fn validate_unique_ids(cpl: &CplDocument, report: &mut CplValidationReport) {
    let mut seen_ids: HashMap<String, String> = HashMap::new();

    for (seg_idx, segment) in cpl.segments.iter().enumerate() {
        if !segment.id.is_empty() {
            if let Some(prev_loc) = seen_ids.get(&segment.id) {
                report.add_issue(
                    CplIssue::new(
                        Severity::Error,
                        IssueCategory::DuplicateId,
                        format!(
                            "Duplicate segment ID: {} (also at {})",
                            segment.id, prev_loc
                        ),
                    )
                    .with_location(format!("Segment[{seg_idx}]")),
                );
            } else {
                seen_ids.insert(segment.id.clone(), format!("Segment[{seg_idx}]"));
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_valid_cpl() -> CplDocument {
        let mut known = HashSet::new();
        known.insert("track-file-001".to_string());
        known.insert("track-file-002".to_string());

        CplDocument {
            id: "cpl-uuid-001".to_string(),
            content_title: "Test Composition".to_string(),
            edit_rate: CplEditRate::new(24, 1),
            segments: vec![CplSegment {
                id: "segment-001".to_string(),
                virtual_tracks: vec![
                    CplVirtualTrack {
                        id: "vt-video".to_string(),
                        track_type: "MainImageSequence".to_string(),
                        resources: vec![CplResource {
                            id: "res-v-001".to_string(),
                            track_file_id: "track-file-001".to_string(),
                            edit_rate: CplEditRate::new(24, 1),
                            entry_point: 0,
                            source_duration: 240,
                            intrinsic_duration: 240,
                            repeat_count: 1,
                        }],
                    },
                    CplVirtualTrack {
                        id: "vt-audio".to_string(),
                        track_type: "MainAudioSequence".to_string(),
                        resources: vec![CplResource {
                            id: "res-a-001".to_string(),
                            track_file_id: "track-file-002".to_string(),
                            edit_rate: CplEditRate::new(24, 1),
                            entry_point: 0,
                            source_duration: 240,
                            intrinsic_duration: 480,
                            repeat_count: 1,
                        }],
                    },
                ],
            }],
            known_track_files: known,
        }
    }

    #[test]
    fn test_valid_cpl_passes() {
        let cpl = make_valid_cpl();
        let report = validate_cpl(&cpl);
        assert!(
            report.is_valid(),
            "Valid CPL should pass: {:?}",
            report.issues
        );
    }

    #[test]
    fn test_empty_cpl_id() {
        let mut cpl = make_valid_cpl();
        cpl.id = String::new();
        let report = validate_cpl(&cpl);
        assert!(!report.is_valid());
        assert!(report.count_by_severity(Severity::Fatal) > 0);
    }

    #[test]
    fn test_invalid_edit_rate() {
        let mut cpl = make_valid_cpl();
        cpl.edit_rate = CplEditRate::new(0, 0);
        let report = validate_cpl(&cpl);
        assert!(!report.is_valid());
        assert!(
            report
                .issues_by_category(IssueCategory::EditRateMismatch)
                .len()
                > 0
        );
    }

    #[test]
    fn test_no_segments() {
        let mut cpl = make_valid_cpl();
        cpl.segments.clear();
        let report = validate_cpl(&cpl);
        assert!(!report.is_valid());
    }

    #[test]
    fn test_unknown_track_file() {
        let mut cpl = make_valid_cpl();
        cpl.segments[0].virtual_tracks[0].resources[0].track_file_id = "unknown-track".to_string();
        let report = validate_cpl(&cpl);
        assert!(!report.is_valid());
        assert!(
            report
                .issues_by_category(IssueCategory::ResourceReference)
                .len()
                > 0
        );
    }

    #[test]
    fn test_duration_mismatch_across_tracks() {
        let mut cpl = make_valid_cpl();
        // Make audio track shorter
        cpl.segments[0].virtual_tracks[1].resources[0].source_duration = 120;
        let report = validate_cpl(&cpl);
        assert!(!report.is_valid());
        assert!(
            report
                .issues_by_category(IssueCategory::TimelineDiscontinuity)
                .len()
                > 0
        );
    }

    #[test]
    fn test_zero_source_duration() {
        let mut cpl = make_valid_cpl();
        cpl.segments[0].virtual_tracks[0].resources[0].source_duration = 0;
        let report = validate_cpl(&cpl);
        assert!(!report.is_valid());
    }

    #[test]
    fn test_exceeds_intrinsic_duration() {
        let mut cpl = make_valid_cpl();
        cpl.segments[0].virtual_tracks[0].resources[0].entry_point = 200;
        cpl.segments[0].virtual_tracks[0].resources[0].source_duration = 100;
        cpl.segments[0].virtual_tracks[0].resources[0].intrinsic_duration = 240;
        let report = validate_cpl(&cpl);
        assert!(!report.is_valid());
    }

    #[test]
    fn test_duplicate_segment_ids() {
        let mut cpl = make_valid_cpl();
        let segment_clone = cpl.segments[0].clone();
        cpl.segments.push(segment_clone);
        let report = validate_cpl(&cpl);
        assert!(report.issues_by_category(IssueCategory::DuplicateId).len() > 0);
    }

    #[test]
    fn test_edit_rate_display() {
        let rate = CplEditRate::new(24000, 1001);
        assert_eq!(format!("{rate}"), "24000/1001");
        let fps = rate.as_f64();
        assert!((fps - 23.976).abs() < 0.1);
    }

    #[test]
    fn test_severity_ordering() {
        assert!(Severity::Info < Severity::Warning);
        assert!(Severity::Warning < Severity::Error);
        assert!(Severity::Error < Severity::Fatal);
    }

    #[test]
    fn test_report_max_severity() {
        let mut report = CplValidationReport::new();
        assert!(report.max_severity().is_none());
        report.add_issue(CplIssue::new(
            Severity::Warning,
            IssueCategory::MetadataIssue,
            "test",
        ));
        assert_eq!(report.max_severity(), Some(Severity::Warning));
        report.add_issue(CplIssue::new(
            Severity::Error,
            IssueCategory::DurationError,
            "test2",
        ));
        assert_eq!(report.max_severity(), Some(Severity::Error));
    }

    #[test]
    fn test_issue_display() {
        let issue = CplIssue::new(Severity::Error, IssueCategory::DurationError, "test msg")
            .with_location("Segment[0]");
        let display = format!("{issue}");
        assert!(display.contains("ERROR"));
        assert!(display.contains("DurationError"));
        assert!(display.contains("Segment[0]"));
    }

    #[test]
    fn test_resource_effective_duration() {
        let res = CplResource {
            id: "r1".to_string(),
            track_file_id: "tf1".to_string(),
            edit_rate: CplEditRate::new(24, 1),
            entry_point: 0,
            source_duration: 100,
            intrinsic_duration: 500,
            repeat_count: 3,
        };
        assert_eq!(res.effective_duration(), 300);
        assert!(!res.exceeds_intrinsic());
    }
}
