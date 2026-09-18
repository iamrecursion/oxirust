//! Format conformance checking for professional media deliverables.
//!
//! Validates video format specifications against target delivery requirements,
//! reporting issues at various severity levels.

#![allow(dead_code)]

/// Severity of a conformance issue.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub enum IssueSeverity {
    /// Informational warning; delivery may still succeed.
    Warning,
    /// Non-fatal error; should be corrected before delivery.
    Error,
    /// Critical blocker; delivery will fail without correction.
    Critical,
}

impl IssueSeverity {
    /// Returns a numeric priority (higher = more urgent).
    #[must_use]
    pub fn priority(&self) -> u32 {
        match self {
            Self::Warning => 1,
            Self::Error => 2,
            Self::Critical => 3,
        }
    }
}

/// A single conformance issue found in a media file.
#[derive(Debug, Clone)]
pub struct ConformIssue {
    /// The field/parameter that failed conformance.
    pub field: String,
    /// The value that was expected.
    pub expected: String,
    /// The value that was actually found.
    pub found: String,
    /// Severity of the issue.
    pub severity: IssueSeverity,
}

impl ConformIssue {
    /// Create a new conformance issue.
    #[must_use]
    pub fn new(
        field: impl Into<String>,
        expected: impl Into<String>,
        found: impl Into<String>,
        severity: IssueSeverity,
    ) -> Self {
        Self {
            field: field.into(),
            expected: expected.into(),
            found: found.into(),
            severity,
        }
    }

    /// Returns `true` if this issue would block delivery.
    #[must_use]
    pub fn is_blocking(&self) -> bool {
        matches!(
            self.severity,
            IssueSeverity::Error | IssueSeverity::Critical
        )
    }
}

/// Target format specification for conformance checking.
#[derive(Debug, Clone)]
pub struct FormatSpec {
    /// Width in pixels.
    pub width: u32,
    /// Height in pixels.
    pub height: u32,
    /// Frame rate in frames per second.
    pub frame_rate: f64,
    /// Color space identifier (e.g. "BT.709", "BT.2020").
    pub color_space: String,
    /// Bit depth per channel.
    pub bit_depth: u8,
    /// Codec identifier (e.g. "H.264", "`ProRes` 422").
    pub codec: String,
}

impl FormatSpec {
    /// Returns `true` if the spec is HD (1280×720 or 1920×1080).
    #[must_use]
    pub fn is_hd(&self) -> bool {
        (self.width == 1280 && self.height == 720) || (self.width == 1920 && self.height == 1080)
    }

    /// Returns `true` if the spec is UHD 4K (3840×2160).
    #[must_use]
    pub fn is_4k(&self) -> bool {
        self.width == 3840 && self.height == 2160
    }

    /// Returns the pixel aspect ratio (width / height).
    ///
    /// Returns `0.0` when height is zero.
    #[must_use]
    pub fn aspect_ratio(&self) -> f64 {
        if self.height == 0 {
            return 0.0;
        }
        f64::from(self.width) / f64::from(self.height)
    }
}

/// Tolerance parameters used when comparing two [`FormatSpec`] values.
#[derive(Debug, Clone)]
pub struct FormatTolerance {
    /// Maximum allowed frame-rate difference (in fps) before raising an issue.
    pub frame_rate_tolerance: f64,
    /// Whether upscaling (actual resolution smaller than target) is acceptable.
    pub allow_upscale: bool,
}

impl FormatTolerance {
    /// Strict tolerance: exact frame rate match, no upscaling.
    #[must_use]
    pub fn strict() -> Self {
        Self {
            frame_rate_tolerance: 0.001,
            allow_upscale: false,
        }
    }

    /// Loose tolerance: 0.5 fps margin, upscaling allowed.
    #[must_use]
    pub fn loose() -> Self {
        Self {
            frame_rate_tolerance: 0.5,
            allow_upscale: true,
        }
    }
}

/// Aggregated conformance report for a media file against a [`FormatSpec`].
#[derive(Debug, Default)]
pub struct FormatConformReport {
    /// All conformance issues found.
    pub issues: Vec<ConformIssue>,
}

impl FormatConformReport {
    /// Create a new empty report.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Returns `true` when there are no blocking issues.
    #[must_use]
    pub fn is_conformant(&self) -> bool {
        !self.issues.iter().any(ConformIssue::is_blocking)
    }

    /// Returns only the critical issues.
    #[must_use]
    pub fn critical_issues(&self) -> Vec<&ConformIssue> {
        self.issues
            .iter()
            .filter(|i| i.severity == IssueSeverity::Critical)
            .collect()
    }

    /// Returns only the warning-level issues.
    #[must_use]
    pub fn warnings(&self) -> Vec<&ConformIssue> {
        self.issues
            .iter()
            .filter(|i| i.severity == IssueSeverity::Warning)
            .collect()
    }
}

/// Check a media `actual` format spec against a `target`, using the given
/// `tolerance`, and return a [`FormatConformReport`].
#[must_use]
pub fn check_format(
    target: &FormatSpec,
    actual: &FormatSpec,
    tolerance: &FormatTolerance,
) -> FormatConformReport {
    let mut report = FormatConformReport::new();

    // Width
    if actual.width != target.width {
        let severity = if !tolerance.allow_upscale || actual.width < target.width {
            IssueSeverity::Error
        } else {
            IssueSeverity::Warning
        };
        report.issues.push(ConformIssue::new(
            "width",
            target.width.to_string(),
            actual.width.to_string(),
            severity,
        ));
    }

    // Height
    if actual.height != target.height {
        let severity = if !tolerance.allow_upscale || actual.height < target.height {
            IssueSeverity::Error
        } else {
            IssueSeverity::Warning
        };
        report.issues.push(ConformIssue::new(
            "height",
            target.height.to_string(),
            actual.height.to_string(),
            severity,
        ));
    }

    // Frame rate
    if (actual.frame_rate - target.frame_rate).abs() > tolerance.frame_rate_tolerance {
        report.issues.push(ConformIssue::new(
            "frame_rate",
            format!("{:.3}", target.frame_rate),
            format!("{:.3}", actual.frame_rate),
            IssueSeverity::Error,
        ));
    }

    // Color space
    if actual.color_space != target.color_space {
        report.issues.push(ConformIssue::new(
            "color_space",
            target.color_space.clone(),
            actual.color_space.clone(),
            IssueSeverity::Warning,
        ));
    }

    // Bit depth
    if actual.bit_depth != target.bit_depth {
        let severity = if actual.bit_depth < target.bit_depth {
            IssueSeverity::Error
        } else {
            IssueSeverity::Warning
        };
        report.issues.push(ConformIssue::new(
            "bit_depth",
            target.bit_depth.to_string(),
            actual.bit_depth.to_string(),
            severity,
        ));
    }

    // Codec
    if actual.codec != target.codec {
        report.issues.push(ConformIssue::new(
            "codec",
            target.codec.clone(),
            actual.codec.clone(),
            IssueSeverity::Critical,
        ));
    }

    report
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn hd_spec() -> FormatSpec {
        FormatSpec {
            width: 1920,
            height: 1080,
            frame_rate: 25.0,
            color_space: "BT.709".to_string(),
            bit_depth: 8,
            codec: "H.264".to_string(),
        }
    }

    #[test]
    fn test_is_hd_1080() {
        let spec = hd_spec();
        assert!(spec.is_hd());
    }

    #[test]
    fn test_is_hd_720() {
        let spec = FormatSpec {
            width: 1280,
            height: 720,
            frame_rate: 25.0,
            color_space: "BT.709".to_string(),
            bit_depth: 8,
            codec: "H.264".to_string(),
        };
        assert!(spec.is_hd());
    }

    #[test]
    fn test_is_not_hd() {
        let spec = FormatSpec {
            width: 720,
            height: 576,
            frame_rate: 25.0,
            color_space: "BT.601".to_string(),
            bit_depth: 8,
            codec: "MPEG-2".to_string(),
        };
        assert!(!spec.is_hd());
    }

    #[test]
    fn test_is_4k() {
        let spec = FormatSpec {
            width: 3840,
            height: 2160,
            frame_rate: 24.0,
            color_space: "BT.2020".to_string(),
            bit_depth: 10,
            codec: "H.265".to_string(),
        };
        assert!(spec.is_4k());
    }

    #[test]
    fn test_aspect_ratio() {
        let spec = hd_spec();
        let ratio = spec.aspect_ratio();
        assert!((ratio - 16.0 / 9.0).abs() < 0.001, "ratio: {ratio}");
    }

    #[test]
    fn test_aspect_ratio_zero_height() {
        let spec = FormatSpec {
            width: 1920,
            height: 0,
            frame_rate: 25.0,
            color_space: "BT.709".to_string(),
            bit_depth: 8,
            codec: "H.264".to_string(),
        };
        assert_eq!(spec.aspect_ratio(), 0.0);
    }

    #[test]
    fn test_issue_severity_priority_order() {
        assert!(IssueSeverity::Critical.priority() > IssueSeverity::Error.priority());
        assert!(IssueSeverity::Error.priority() > IssueSeverity::Warning.priority());
    }

    #[test]
    fn test_conform_issue_blocking() {
        let err = ConformIssue::new("codec", "H.264", "H.265", IssueSeverity::Critical);
        assert!(err.is_blocking());
        let warn = ConformIssue::new("color_space", "BT.709", "BT.601", IssueSeverity::Warning);
        assert!(!warn.is_blocking());
    }

    #[test]
    fn test_format_tolerance_strict() {
        let t = FormatTolerance::strict();
        assert!(!t.allow_upscale);
        assert!(t.frame_rate_tolerance < 0.01);
    }

    #[test]
    fn test_format_tolerance_loose() {
        let t = FormatTolerance::loose();
        assert!(t.allow_upscale);
        assert!(t.frame_rate_tolerance >= 0.5);
    }

    #[test]
    fn test_conformant_when_identical() {
        let spec = hd_spec();
        let report = check_format(&spec, &spec, &FormatTolerance::strict());
        assert!(report.is_conformant());
        assert!(report.issues.is_empty());
    }

    #[test]
    fn test_codec_mismatch_is_critical() {
        let target = hd_spec();
        let actual = FormatSpec {
            codec: "H.265".to_string(),
            ..hd_spec()
        };
        let report = check_format(&target, &actual, &FormatTolerance::strict());
        assert!(!report.is_conformant());
        assert!(!report.critical_issues().is_empty());
    }

    #[test]
    fn test_frame_rate_mismatch_strict() {
        let target = hd_spec();
        let actual = FormatSpec {
            frame_rate: 29.97,
            ..hd_spec()
        };
        let report = check_format(&target, &actual, &FormatTolerance::strict());
        let fps_issue = report.issues.iter().find(|i| i.field == "frame_rate");
        assert!(fps_issue.is_some());
    }

    #[test]
    fn test_frame_rate_within_loose_tolerance() {
        let target = hd_spec();
        let actual = FormatSpec {
            frame_rate: 25.1,
            ..hd_spec()
        };
        let report = check_format(&target, &actual, &FormatTolerance::loose());
        let fps_issue = report.issues.iter().find(|i| i.field == "frame_rate");
        assert!(fps_issue.is_none());
    }

    #[test]
    fn test_warnings_vs_critical_split() {
        let target = hd_spec();
        let actual = FormatSpec {
            color_space: "BT.601".to_string(),
            codec: "H.265".to_string(),
            ..hd_spec()
        };
        let report = check_format(&target, &actual, &FormatTolerance::strict());
        assert!(!report.warnings().is_empty());
        assert!(!report.critical_issues().is_empty());
    }
}
