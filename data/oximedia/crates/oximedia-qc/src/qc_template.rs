//! QC template system: reusable check configurations for quality control workflows.
//!
//! Allows building named templates of QC checks that can be saved to a library
//! and instantiated per-job.  Templates support **inheritance**: a custom template
//! can extend a built-in preset via [`crate::qc_template::QcTemplateRef`], overriding only the fields
//! that differ from the parent.
//!
//! # Template Inheritance
//!
//! ```rust
//! use oximedia_qc::qc_template::{
//!     QcTemplateLibrary, QcTemplateRef, QcTemplateOverrides, resolve_template,
//! };
//!
//! let mut library = QcTemplateLibrary::new();
//! // register built-in presets …
//! // library.register(make_broadcast_hd_template());
//!
//! let child = QcTemplateRef {
//!     base: Some("broadcast_hd".to_string()),
//!     name: "my_hdr_broadcast".to_string(),
//!     overrides: QcTemplateOverrides {
//!         max_peak_luma: Some(4000.0),
//!         ..Default::default()
//!     },
//! };
//! // let resolved = resolve_template(&child, &library).unwrap();
//! // resolved.max_peak_luma == 4000.0
//! ```

#![allow(dead_code)]

use std::collections::HashMap;
use std::fmt;

/// Categories of QC checks with associated severity levels.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum QcCheckType {
    /// Video signal level checks (luma, chroma, clipping).
    VideoLevels,
    /// Audio loudness and level checks (EBU R128, ATSC A/85).
    AudioLoudness,
    /// Codec/container structural validation.
    CodecContainer,
    /// Closed-caption/subtitle presence and validity.
    CaptionCompliance,
    /// Timecode continuity and accuracy.
    TimecodeIntegrity,
    /// Colour space and transfer function compliance.
    ColourSpace,
    /// Black frames and freeze detection.
    BlackFreeze,
    /// Custom check defined by external rule name.
    Custom(String),
}

impl QcCheckType {
    /// Return the default severity string for this check type.
    ///
    /// Severity: `"error"`, `"warning"`, or `"info"`.
    pub fn severity(&self) -> &'static str {
        match self {
            QcCheckType::VideoLevels => "error",
            QcCheckType::AudioLoudness => "error",
            QcCheckType::CodecContainer => "error",
            QcCheckType::CaptionCompliance => "warning",
            QcCheckType::TimecodeIntegrity => "warning",
            QcCheckType::ColourSpace => "warning",
            QcCheckType::BlackFreeze => "info",
            QcCheckType::Custom(_) => "info",
        }
    }

    /// Returns `true` if the severity is `"error"`.
    pub fn is_error_level(&self) -> bool {
        self.severity() == "error"
    }
}

/// A single check entry within a QC template.
#[derive(Debug, Clone)]
pub struct QcTemplateEntry {
    /// The type of check.
    pub check_type: QcCheckType,
    /// Human-readable description of what is being checked.
    pub description: String,
    /// Whether this check must pass for delivery to be allowed.
    pub required: bool,
    /// Optional override severity (if `None`, uses `QcCheckType::severity()`).
    pub severity_override: Option<String>,
}

impl QcTemplateEntry {
    /// Create a new template entry.
    pub fn new(check_type: QcCheckType, description: impl Into<String>, required: bool) -> Self {
        Self {
            check_type,
            description: description.into(),
            required,
            severity_override: None,
        }
    }

    /// Returns `true` if this check is required for delivery.
    pub fn is_required(&self) -> bool {
        self.required
    }

    /// Effective severity (override takes precedence).
    pub fn effective_severity(&self) -> &str {
        self.severity_override
            .as_deref()
            .unwrap_or_else(|| self.check_type.severity())
    }
}

/// A named collection of QC checks forming a reusable template.
///
/// In addition to the ordered list of [`QcTemplateEntry`] items, a template
/// carries *parameter fields* (luma limits, allowed codecs, loudness target)
/// that drive threshold-based validation rules.  These parameters are the
/// fields that are eligible for override in the template inheritance system —
/// see [`QcTemplateRef`] and [`resolve_template`].
#[derive(Debug, Clone)]
pub struct QcTemplate {
    /// Template name, e.g. `"broadcast_hd"`.
    pub name: String,
    /// Optional description of this template's intended use.
    pub description: String,
    /// Ordered list of check entries.
    entries: Vec<QcTemplateEntry>,

    // ---- parameter fields ----
    /// Maximum allowed peak luma in nits (e.g. 1000.0 for HDR10, 100.0 for SDR).
    ///
    /// Default: 1000.0 nits (PQ / HDR10 reference peak).
    pub max_peak_luma: f32,
    /// Minimum allowed peak luma in nits.
    ///
    /// Default: 0.005 nits (mastering display black level).
    pub min_peak_luma: f32,
    /// Codec identifiers permitted by this template (e.g. `["av1", "vp9"]`).
    ///
    /// An empty list means *all patent-free codecs are accepted*.
    pub allowed_codecs: Vec<String>,
    /// Maximum integrated loudness target in LUFS (negative, e.g. -23.0).
    ///
    /// Default: -23.0 LUFS (EBU R128).
    pub max_loudness_lufs: f32,
}

impl QcTemplate {
    /// Create an empty template with sensible parameter defaults.
    pub fn new(name: impl Into<String>, description: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            description: description.into(),
            entries: Vec::new(),
            max_peak_luma: 1000.0,
            min_peak_luma: 0.005,
            allowed_codecs: Vec::new(),
            max_loudness_lufs: -23.0,
        }
    }

    /// Add a check entry to the template.
    pub fn add_check(&mut self, entry: QcTemplateEntry) {
        self.entries.push(entry);
    }

    /// Return only the required check entries.
    pub fn required_checks(&self) -> Vec<&QcTemplateEntry> {
        self.entries.iter().filter(|e| e.is_required()).collect()
    }

    /// Return all check entries.
    pub fn all_checks(&self) -> &[QcTemplateEntry] {
        &self.entries
    }

    /// Total number of checks in the template.
    pub fn check_count(&self) -> usize {
        self.entries.len()
    }
}

/// Library of named QC templates.
#[derive(Debug, Default)]
pub struct QcTemplateLibrary {
    templates: HashMap<String, QcTemplate>,
}

impl QcTemplateLibrary {
    /// Create an empty library.
    pub fn new() -> Self {
        Self::default()
    }

    /// Register a template, replacing any existing template with the same name.
    pub fn register(&mut self, template: QcTemplate) {
        self.templates.insert(template.name.clone(), template);
    }

    /// Retrieve a template by name.
    pub fn get(&self, name: &str) -> Option<&QcTemplate> {
        self.templates.get(name)
    }

    /// Number of templates in the library.
    pub fn count(&self) -> usize {
        self.templates.len()
    }

    /// Return the names of all registered templates.
    pub fn names(&self) -> Vec<&str> {
        self.templates
            .keys()
            .map(std::string::String::as_str)
            .collect()
    }
}

// ============================================================================
// Template inheritance
// ============================================================================

/// Errors that can occur when resolving a [`QcTemplateRef`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum QcError {
    /// The named base template does not exist in the registry.
    TemplateNotFound(String),
}

impl fmt::Display for QcError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::TemplateNotFound(name) => {
                write!(f, "QC template not found: {name:?}")
            }
        }
    }
}

impl std::error::Error for QcError {}

/// Partial overrides applied when resolving a child template.
///
/// Every field is wrapped in `Option<T>`.  A `Some(value)` replaces the
/// parent template's value; `None` means "keep parent value".  When there
/// is no parent (i.e. [`QcTemplateRef::base`] is `None`), `None` fields fall
/// back to the defaults defined in [`QcTemplate::new`].
#[derive(Debug, Clone, Default)]
pub struct QcTemplateOverrides {
    /// Override for [`QcTemplate::max_peak_luma`].
    pub max_peak_luma: Option<f32>,
    /// Override for [`QcTemplate::min_peak_luma`].
    pub min_peak_luma: Option<f32>,
    /// Override for [`QcTemplate::allowed_codecs`].
    pub allowed_codecs: Option<Vec<String>>,
    /// Override for [`QcTemplate::max_loudness_lufs`].
    pub max_loudness_lufs: Option<f32>,
    /// Override for [`QcTemplate::description`].
    pub description: Option<String>,
}

/// A reference to a [`QcTemplate`] that supports single-level inheritance.
///
/// When resolved via [`resolve_template`]:
/// - If [`QcTemplateRef::base`] is `Some(name)`, the named template is looked
///   up in the registry and used as the starting point; the child's
///   [`overrides`](QcTemplateRef::overrides) are then applied on top.
/// - If [`QcTemplateRef::base`] is `None`, a fresh template is built using the
///   defaults from [`QcTemplate::new`] with overrides applied.
///
/// Check entries (the [`Vec<QcTemplateEntry>`] part of the template) are
/// *inherited as-is* from the base template; the override mechanism only
/// affects parameter fields.
#[derive(Debug, Clone)]
pub struct QcTemplateRef {
    /// Optional name of the parent template to inherit from.
    pub base: Option<String>,
    /// Name for the resolved template.
    pub name: String,
    /// Field-level overrides that take priority over the parent template.
    pub overrides: QcTemplateOverrides,
}

impl QcTemplateRef {
    /// Create a new template reference with no base (standalone).
    #[must_use]
    pub fn standalone(name: impl Into<String>) -> Self {
        Self {
            base: None,
            name: name.into(),
            overrides: QcTemplateOverrides::default(),
        }
    }

    /// Create a new template reference that extends an existing template.
    #[must_use]
    pub fn extending(name: impl Into<String>, base: impl Into<String>) -> Self {
        Self {
            base: Some(base.into()),
            name: name.into(),
            overrides: QcTemplateOverrides::default(),
        }
    }
}

/// Resolve a [`QcTemplateRef`] into a concrete [`QcTemplate`].
///
/// # Algorithm
///
/// 1. If `template_ref.base` is `None`, start from [`QcTemplate::new`] defaults.
/// 2. If `template_ref.base` is `Some(base_name)`, look up `base_name` in
///    `registry`.  Returns [`QcError::TemplateNotFound`] if not found.
/// 3. Apply each field in `template_ref.overrides`: a `Some(v)` replaces the
///    corresponding field; `None` leaves it unchanged.
/// 4. Set the template `name` to `template_ref.name`.
///
/// Check entries ([`QcTemplateEntry`] items) are cloned from the base
/// template and are *not* affected by the overrides.
///
/// # Errors
///
/// Returns [`QcError::TemplateNotFound`] when the base template name does not
/// exist in the registry.
pub fn resolve_template(
    template_ref: &QcTemplateRef,
    registry: &QcTemplateLibrary,
) -> Result<QcTemplate, QcError> {
    // Step 1 / 2: obtain base template (or default)
    let mut resolved = match &template_ref.base {
        None => QcTemplate::new(&template_ref.name, ""),
        Some(base_name) => {
            let parent = registry
                .get(base_name)
                .ok_or_else(|| QcError::TemplateNotFound(base_name.clone()))?;
            parent.clone()
        }
    };

    // Step 3: apply overrides
    let ov = &template_ref.overrides;
    if let Some(v) = ov.max_peak_luma {
        resolved.max_peak_luma = v;
    }
    if let Some(v) = ov.min_peak_luma {
        resolved.min_peak_luma = v;
    }
    if let Some(ref codecs) = ov.allowed_codecs {
        resolved.allowed_codecs = codecs.clone();
    }
    if let Some(v) = ov.max_loudness_lufs {
        resolved.max_loudness_lufs = v;
    }
    if let Some(ref desc) = ov.description {
        resolved.description = desc.clone();
    }

    // Step 4: always use the child's name
    resolved.name = template_ref.name.clone();

    Ok(resolved)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_entry(check_type: QcCheckType, required: bool) -> QcTemplateEntry {
        QcTemplateEntry::new(check_type, "desc", required)
    }

    #[test]
    fn test_check_type_severity_video_levels() {
        assert_eq!(QcCheckType::VideoLevels.severity(), "error");
    }

    #[test]
    fn test_check_type_severity_caption() {
        assert_eq!(QcCheckType::CaptionCompliance.severity(), "warning");
    }

    #[test]
    fn test_check_type_severity_black_freeze() {
        assert_eq!(QcCheckType::BlackFreeze.severity(), "info");
    }

    #[test]
    fn test_check_type_is_error_level_true() {
        assert!(QcCheckType::CodecContainer.is_error_level());
    }

    #[test]
    fn test_check_type_is_error_level_false() {
        assert!(!QcCheckType::ColourSpace.is_error_level());
    }

    #[test]
    fn test_check_type_custom_severity() {
        let ct = QcCheckType::Custom("my_check".to_string());
        assert_eq!(ct.severity(), "info");
    }

    #[test]
    fn test_template_entry_is_required() {
        let e = make_entry(QcCheckType::VideoLevels, true);
        assert!(e.is_required());
    }

    #[test]
    fn test_template_entry_not_required() {
        let e = make_entry(QcCheckType::BlackFreeze, false);
        assert!(!e.is_required());
    }

    #[test]
    fn test_template_entry_effective_severity_default() {
        let e = make_entry(QcCheckType::AudioLoudness, true);
        assert_eq!(e.effective_severity(), "error");
    }

    #[test]
    fn test_template_entry_effective_severity_override() {
        let mut e = make_entry(QcCheckType::VideoLevels, true);
        e.severity_override = Some("warning".to_string());
        assert_eq!(e.effective_severity(), "warning");
    }

    #[test]
    fn test_qc_template_add_check_and_count() {
        let mut t = QcTemplate::new("broadcast", "Standard broadcast checks");
        t.add_check(make_entry(QcCheckType::VideoLevels, true));
        t.add_check(make_entry(QcCheckType::AudioLoudness, true));
        assert_eq!(t.check_count(), 2);
    }

    #[test]
    fn test_qc_template_required_checks() {
        let mut t = QcTemplate::new("test", "");
        t.add_check(make_entry(QcCheckType::VideoLevels, true));
        t.add_check(make_entry(QcCheckType::BlackFreeze, false));
        let req = t.required_checks();
        assert_eq!(req.len(), 1);
        assert_eq!(req[0].check_type, QcCheckType::VideoLevels);
    }

    #[test]
    fn test_template_library_register_and_get() {
        let mut lib = QcTemplateLibrary::new();
        lib.register(QcTemplate::new("t1", "first"));
        assert!(lib.get("t1").is_some());
    }

    #[test]
    fn test_template_library_get_missing() {
        let lib = QcTemplateLibrary::new();
        assert!(lib.get("nope").is_none());
    }

    #[test]
    fn test_template_library_count() {
        let mut lib = QcTemplateLibrary::new();
        lib.register(QcTemplate::new("a", ""));
        lib.register(QcTemplate::new("b", ""));
        assert_eq!(lib.count(), 2);
    }

    #[test]
    fn test_template_library_names() {
        let mut lib = QcTemplateLibrary::new();
        lib.register(QcTemplate::new("alpha", ""));
        let names = lib.names();
        assert!(names.contains(&"alpha"));
    }

    // ---- Template inheritance tests ----

    /// Build a registry with a single "broadcast_hd" parent template.
    fn make_registry_with_broadcast() -> QcTemplateLibrary {
        let mut lib = QcTemplateLibrary::new();
        let mut parent = QcTemplate::new("broadcast_hd", "Standard broadcast HD");
        // Set non-default values so we can verify fallthrough.
        parent.max_peak_luma = 1000.0;
        parent.min_peak_luma = 0.05;
        parent.max_loudness_lufs = -23.0;
        parent.allowed_codecs = vec!["av1".to_string(), "vp9".to_string()];
        parent.add_check(make_entry(QcCheckType::VideoLevels, true));
        lib.register(parent);
        lib
    }

    /// Child overrides parent's `max_peak_luma`; resolved template uses child value.
    #[test]
    fn test_template_inheritance_override() {
        let registry = make_registry_with_broadcast();
        let child = QcTemplateRef {
            base: Some("broadcast_hd".to_string()),
            name: "hdr_broadcast".to_string(),
            overrides: QcTemplateOverrides {
                max_peak_luma: Some(4000.0),
                ..Default::default()
            },
        };
        let resolved = resolve_template(&child, &registry).expect("should resolve");
        assert_eq!(resolved.name, "hdr_broadcast");
        assert!(
            (resolved.max_peak_luma - 4000.0).abs() < f32::EPSILON,
            "child override should win: expected 4000.0, got {}",
            resolved.max_peak_luma
        );
    }

    /// Field NOT overridden in child → resolved template uses parent value.
    #[test]
    fn test_template_inheritance_fallthrough() {
        let registry = make_registry_with_broadcast();
        let child = QcTemplateRef {
            base: Some("broadcast_hd".to_string()),
            name: "slight_variant".to_string(),
            overrides: QcTemplateOverrides {
                // Only override loudness; luma should fall through from parent.
                max_loudness_lufs: Some(-24.0),
                ..Default::default()
            },
        };
        let resolved = resolve_template(&child, &registry).expect("should resolve");
        // max_peak_luma was NOT overridden → keeps parent value 1000.0
        assert!(
            (resolved.max_peak_luma - 1000.0).abs() < f32::EPSILON,
            "should fall through to parent 1000.0, got {}",
            resolved.max_peak_luma
        );
        // min_peak_luma also falls through
        assert!(
            (resolved.min_peak_luma - 0.05).abs() < f32::EPSILON,
            "should fall through to parent 0.05, got {}",
            resolved.min_peak_luma
        );
        // loudness override applied
        assert!(
            (resolved.max_loudness_lufs - (-24.0)).abs() < f32::EPSILON,
            "loudness override should apply: expected -24.0, got {}",
            resolved.max_loudness_lufs
        );
        // check entries inherited from parent
        assert_eq!(
            resolved.check_count(),
            1,
            "check entries should be inherited"
        );
    }

    /// None base → builds from overrides/defaults.
    #[test]
    fn test_template_inheritance_no_base() {
        let registry = QcTemplateLibrary::new(); // empty
        let child = QcTemplateRef {
            base: None,
            name: "custom_sdr".to_string(),
            overrides: QcTemplateOverrides {
                max_peak_luma: Some(100.0),
                allowed_codecs: Some(vec!["av1".to_string()]),
                ..Default::default()
            },
        };
        let resolved = resolve_template(&child, &registry).expect("should resolve");
        assert_eq!(resolved.name, "custom_sdr");
        assert!(
            (resolved.max_peak_luma - 100.0).abs() < f32::EPSILON,
            "override applied, got {}",
            resolved.max_peak_luma
        );
        assert_eq!(resolved.allowed_codecs, vec!["av1".to_string()]);
        // Non-overridden fields use defaults from QcTemplate::new
        assert!(
            (resolved.min_peak_luma - 0.005).abs() < 1e-6_f32,
            "default min_peak_luma, got {}",
            resolved.min_peak_luma
        );
        assert!(
            (resolved.max_loudness_lufs - (-23.0)).abs() < f32::EPSILON,
            "default loudness, got {}",
            resolved.max_loudness_lufs
        );
    }

    /// Err when base template not found in registry.
    #[test]
    fn test_template_inheritance_unknown_base() {
        let registry = QcTemplateLibrary::new(); // empty
        let child = QcTemplateRef {
            base: Some("nonexistent_template".to_string()),
            name: "orphan".to_string(),
            overrides: QcTemplateOverrides::default(),
        };
        let result = resolve_template(&child, &registry);
        assert!(result.is_err(), "should fail for unknown base");
        assert_eq!(
            result.err(),
            Some(QcError::TemplateNotFound(
                "nonexistent_template".to_string()
            ))
        );
    }
}
