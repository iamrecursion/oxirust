//! IMF delivery specification and compliance checking.
//!
//! This module provides structures for defining delivery specifications
//! used by broadcasters and streaming platforms, along with compliance
//! checking utilities.

#![allow(dead_code)]

/// The kind of content being delivered.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ContentKind {
    /// Feature-length film.
    Feature,
    /// Theatrical or promotional trailer.
    Trailer,
    /// Television episode.
    TelevisionEpisode,
    /// Short film.
    Short,
    /// Documentary.
    Documentary,
    /// Animated content.
    Animation,
    /// Advertisement or commercial.
    Advertisement,
}

impl ContentKind {
    /// Returns a human-readable description of the content kind.
    pub fn description(&self) -> &'static str {
        match self {
            ContentKind::Feature => "Feature Film",
            ContentKind::Trailer => "Trailer",
            ContentKind::TelevisionEpisode => "Television Episode",
            ContentKind::Short => "Short Film",
            ContentKind::Documentary => "Documentary",
            ContentKind::Animation => "Animation",
            ContentKind::Advertisement => "Advertisement",
        }
    }
}

/// Severity level for a compliance issue.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub enum IssueSeverity {
    /// A non-critical advisory note.
    Warning,
    /// A significant problem that should be resolved.
    Error,
    /// A showstopper that makes delivery unacceptable.
    Critical,
}

impl IssueSeverity {
    /// Returns a human-readable label for the severity.
    pub fn label(&self) -> &'static str {
        match self {
            IssueSeverity::Warning => "WARNING",
            IssueSeverity::Error => "ERROR",
            IssueSeverity::Critical => "CRITICAL",
        }
    }
}

/// A single compliance issue found during delivery checking.
#[derive(Debug, Clone)]
pub struct ComplianceIssue {
    /// The field or aspect that has the issue.
    pub field: String,
    /// Human-readable description of the issue.
    pub description: String,
    /// How severe the issue is.
    pub severity: IssueSeverity,
}

impl ComplianceIssue {
    /// Creates a new compliance issue.
    pub fn new(
        field: impl Into<String>,
        description: impl Into<String>,
        severity: IssueSeverity,
    ) -> Self {
        Self {
            field: field.into(),
            description: description.into(),
            severity,
        }
    }

    /// Creates a warning-level issue.
    pub fn warning(field: impl Into<String>, description: impl Into<String>) -> Self {
        Self::new(field, description, IssueSeverity::Warning)
    }

    /// Creates an error-level issue.
    pub fn error(field: impl Into<String>, description: impl Into<String>) -> Self {
        Self::new(field, description, IssueSeverity::Error)
    }

    /// Creates a critical-level issue.
    pub fn critical(field: impl Into<String>, description: impl Into<String>) -> Self {
        Self::new(field, description, IssueSeverity::Critical)
    }
}

/// A delivery specification describing what is required for a given platform or context.
#[derive(Debug, Clone)]
pub struct DeliverySpec {
    /// Name of this delivery specification.
    pub name: String,
    /// Kind of content this spec targets.
    pub content_kind: ContentKind,
    /// Minimum required resolution (width, height).
    pub min_resolution: (u32, u32),
    /// Required audio language codes (BCP-47).
    pub required_audio_languages: Vec<String>,
    /// Required subtitle language codes (BCP-47).
    pub required_subtitle_languages: Vec<String>,
    /// Whether HDR (High Dynamic Range) content is required.
    pub hdr_required: bool,
}

impl DeliverySpec {
    /// Creates a new delivery spec.
    pub fn new(name: impl Into<String>, content_kind: ContentKind) -> Self {
        Self {
            name: name.into(),
            content_kind,
            min_resolution: (1920, 1080),
            required_audio_languages: Vec::new(),
            required_subtitle_languages: Vec::new(),
            hdr_required: false,
        }
    }

    /// Standard theatrical delivery specification (4K HDR, English audio/subtitles required).
    pub fn standard_theatrical() -> Self {
        Self {
            name: "Standard Theatrical".to_string(),
            content_kind: ContentKind::Feature,
            min_resolution: (3840, 2160),
            required_audio_languages: vec!["en".to_string()],
            required_subtitle_languages: vec!["en".to_string()],
            hdr_required: true,
        }
    }

    /// Streaming delivery specification (1080p, English audio required).
    pub fn streaming() -> Self {
        Self {
            name: "Streaming".to_string(),
            content_kind: ContentKind::Feature,
            min_resolution: (1920, 1080),
            required_audio_languages: vec!["en".to_string()],
            required_subtitle_languages: Vec::new(),
            hdr_required: false,
        }
    }

    /// Broadcast delivery specification (1080i, multiple languages).
    pub fn broadcast() -> Self {
        Self {
            name: "Broadcast".to_string(),
            content_kind: ContentKind::TelevisionEpisode,
            min_resolution: (1920, 1080),
            required_audio_languages: vec!["en".to_string()],
            required_subtitle_languages: vec!["en".to_string()],
            hdr_required: false,
        }
    }

    /// Sets the minimum resolution requirement.
    pub fn with_min_resolution(mut self, width: u32, height: u32) -> Self {
        self.min_resolution = (width, height);
        self
    }

    /// Adds a required audio language.
    pub fn with_required_audio_language(mut self, lang: impl Into<String>) -> Self {
        self.required_audio_languages.push(lang.into());
        self
    }

    /// Adds a required subtitle language.
    pub fn with_required_subtitle_language(mut self, lang: impl Into<String>) -> Self {
        self.required_subtitle_languages.push(lang.into());
        self
    }

    /// Sets whether HDR is required.
    pub fn with_hdr_required(mut self, required: bool) -> Self {
        self.hdr_required = required;
        self
    }
}

/// Result of a delivery compliance check.
#[derive(Debug, Clone)]
pub struct DeliveryCompliance {
    /// Name of the spec checked against.
    pub spec: String,
    /// All issues found during the check.
    pub issues: Vec<ComplianceIssue>,
    /// Whether the check passed (no Critical or Error issues).
    pub passes: bool,
}

impl DeliveryCompliance {
    /// Returns only issues at or above the given severity.
    pub fn issues_at_severity(&self, min_severity: &IssueSeverity) -> Vec<&ComplianceIssue> {
        self.issues
            .iter()
            .filter(|i| &i.severity >= min_severity)
            .collect()
    }

    /// Returns all critical issues.
    pub fn critical_issues(&self) -> Vec<&ComplianceIssue> {
        self.issues_at_severity(&IssueSeverity::Critical)
    }

    /// Returns all error-level issues (including critical).
    pub fn error_issues(&self) -> Vec<&ComplianceIssue> {
        self.issues_at_severity(&IssueSeverity::Error)
    }

    /// Returns the total count of issues.
    pub fn issue_count(&self) -> usize {
        self.issues.len()
    }
}

/// Checks whether a delivery meets the given spec's requirements.
///
/// # Arguments
///
/// * `spec` - The delivery specification to check against.
/// * `provided_audio_langs` - Language codes of provided audio tracks.
/// * `provided_subtitle_langs` - Language codes of provided subtitle tracks.
/// * `has_hdr` - Whether the content includes HDR metadata.
/// * `resolution` - The actual resolution (width, height) of the content.
pub fn check_delivery_compliance(
    spec: &DeliverySpec,
    provided_audio_langs: &[String],
    provided_subtitle_langs: &[String],
    has_hdr: bool,
    resolution: (u32, u32),
) -> DeliveryCompliance {
    let mut issues = Vec::new();

    // Check resolution
    let (min_w, min_h) = spec.min_resolution;
    let (actual_w, actual_h) = resolution;
    if actual_w < min_w || actual_h < min_h {
        issues.push(ComplianceIssue::error(
            "resolution",
            format!(
                "Resolution {}x{} is below minimum {}x{}",
                actual_w, actual_h, min_w, min_h
            ),
        ));
    }

    // Check required audio languages
    for lang in &spec.required_audio_languages {
        if !provided_audio_langs.contains(lang) {
            issues.push(ComplianceIssue::error(
                "audio_languages",
                format!("Required audio language '{}' is missing", lang),
            ));
        }
    }

    // Check required subtitle languages
    for lang in &spec.required_subtitle_languages {
        if !provided_subtitle_langs.contains(lang) {
            issues.push(ComplianceIssue::warning(
                "subtitle_languages",
                format!("Required subtitle language '{}' is missing", lang),
            ));
        }
    }

    // Check HDR requirement
    if spec.hdr_required && !has_hdr {
        issues.push(ComplianceIssue::critical(
            "hdr",
            "HDR metadata is required but not present",
        ));
    }

    let has_blocking = issues.iter().any(|i| i.severity >= IssueSeverity::Error);

    DeliveryCompliance {
        spec: spec.name.clone(),
        issues,
        passes: !has_blocking,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_content_kind_description() {
        assert_eq!(ContentKind::Feature.description(), "Feature Film");
        assert_eq!(ContentKind::Trailer.description(), "Trailer");
        assert_eq!(
            ContentKind::TelevisionEpisode.description(),
            "Television Episode"
        );
        assert_eq!(ContentKind::Short.description(), "Short Film");
        assert_eq!(ContentKind::Documentary.description(), "Documentary");
        assert_eq!(ContentKind::Animation.description(), "Animation");
        assert_eq!(ContentKind::Advertisement.description(), "Advertisement");
    }

    #[test]
    fn test_issue_severity_ordering() {
        assert!(IssueSeverity::Warning < IssueSeverity::Error);
        assert!(IssueSeverity::Error < IssueSeverity::Critical);
        assert!(IssueSeverity::Warning < IssueSeverity::Critical);
    }

    #[test]
    fn test_issue_severity_label() {
        assert_eq!(IssueSeverity::Warning.label(), "WARNING");
        assert_eq!(IssueSeverity::Error.label(), "ERROR");
        assert_eq!(IssueSeverity::Critical.label(), "CRITICAL");
    }

    #[test]
    fn test_compliance_issue_constructors() {
        let w = ComplianceIssue::warning("field", "desc");
        assert_eq!(w.severity, IssueSeverity::Warning);
        assert_eq!(w.field, "field");
        assert_eq!(w.description, "desc");

        let e = ComplianceIssue::error("field2", "desc2");
        assert_eq!(e.severity, IssueSeverity::Error);

        let c = ComplianceIssue::critical("field3", "desc3");
        assert_eq!(c.severity, IssueSeverity::Critical);
    }

    #[test]
    fn test_standard_theatrical_spec() {
        let spec = DeliverySpec::standard_theatrical();
        assert_eq!(spec.name, "Standard Theatrical");
        assert_eq!(spec.min_resolution, (3840, 2160));
        assert!(spec.hdr_required);
        assert!(spec.required_audio_languages.contains(&"en".to_string()));
        assert!(spec.required_subtitle_languages.contains(&"en".to_string()));
    }

    #[test]
    fn test_streaming_spec() {
        let spec = DeliverySpec::streaming();
        assert_eq!(spec.name, "Streaming");
        assert_eq!(spec.min_resolution, (1920, 1080));
        assert!(!spec.hdr_required);
    }

    #[test]
    fn test_broadcast_spec() {
        let spec = DeliverySpec::broadcast();
        assert_eq!(spec.name, "Broadcast");
        assert_eq!(spec.content_kind, ContentKind::TelevisionEpisode);
    }

    #[test]
    fn test_compliance_passes_for_valid_delivery() {
        let spec = DeliverySpec::streaming();
        let result =
            check_delivery_compliance(&spec, &["en".to_string()], &[], false, (1920, 1080));
        assert!(result.passes);
        assert_eq!(result.issue_count(), 0);
    }

    #[test]
    fn test_compliance_fails_low_resolution() {
        let spec = DeliverySpec::streaming();
        let result = check_delivery_compliance(&spec, &["en".to_string()], &[], false, (1280, 720));
        assert!(!result.passes);
        assert!(result.issues.iter().any(|i| i.field == "resolution"));
    }

    #[test]
    fn test_compliance_fails_missing_audio_lang() {
        let spec = DeliverySpec::streaming();
        let result =
            check_delivery_compliance(&spec, &["fr".to_string()], &[], false, (1920, 1080));
        assert!(!result.passes);
        assert!(result.issues.iter().any(|i| i.field == "audio_languages"));
    }

    #[test]
    fn test_compliance_critical_for_missing_hdr() {
        let spec = DeliverySpec::standard_theatrical();
        let result = check_delivery_compliance(
            &spec,
            &["en".to_string()],
            &["en".to_string()],
            false,
            (3840, 2160),
        );
        assert!(!result.passes);
        let critical: Vec<_> = result.critical_issues();
        assert!(!critical.is_empty());
        assert!(critical.iter().any(|i| i.field == "hdr"));
    }

    #[test]
    fn test_compliance_subtitle_is_warning() {
        let spec = DeliverySpec::broadcast();
        let result = check_delivery_compliance(
            &spec,
            &["en".to_string()],
            &[], // missing subtitles
            false,
            (1920, 1080),
        );
        // Missing subtitles generate a Warning, not an Error, so should still pass
        assert!(result.passes);
        let warnings: Vec<_> = result
            .issues
            .iter()
            .filter(|i| i.severity == IssueSeverity::Warning)
            .collect();
        assert!(!warnings.is_empty());
    }

    #[test]
    fn test_delivery_spec_builder_methods() {
        let spec = DeliverySpec::new("Custom Spec", ContentKind::Documentary)
            .with_min_resolution(2560, 1440)
            .with_required_audio_language("de")
            .with_required_subtitle_language("fr")
            .with_hdr_required(true);

        assert_eq!(spec.name, "Custom Spec");
        assert_eq!(spec.min_resolution, (2560, 1440));
        assert!(spec.required_audio_languages.contains(&"de".to_string()));
        assert!(spec.required_subtitle_languages.contains(&"fr".to_string()));
        assert!(spec.hdr_required);
    }
}
