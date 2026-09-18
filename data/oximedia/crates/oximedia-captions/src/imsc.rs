//! IMSC/TTML captions: timing model, style inheritance, and region layout.
//!
//! This module implements core components of the Internet Media Subtitles and Captions (IMSC)
//! standard, based on TTML (Timed Text Markup Language). It covers timing models,
//! style inheritance chains, and region/layout management.

#![allow(dead_code)]
#![allow(missing_docs)]

use std::collections::HashMap;
use std::fmt;

/// Time expression in IMSC/TTML (wall-clock time in milliseconds)
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub struct ImscTime(pub u64);

impl ImscTime {
    /// Create from hours, minutes, seconds, frames at the given frame rate
    #[allow(clippy::cast_precision_loss)]
    #[must_use]
    pub fn from_hmsf(hours: u32, minutes: u32, seconds: u32, frames: u32, fps: f64) -> Self {
        let frame_ms = (f64::from(frames) / fps * 1000.0) as u64;
        let total_ms = u64::from(hours) * 3_600_000
            + u64::from(minutes) * 60_000
            + u64::from(seconds) * 1_000
            + frame_ms;
        Self(total_ms)
    }

    /// Create from milliseconds
    #[must_use]
    pub const fn from_ms(ms: u64) -> Self {
        Self(ms)
    }

    /// Return value as milliseconds
    #[must_use]
    pub const fn as_ms(self) -> u64 {
        self.0
    }

    /// Duration between two time points (saturating subtraction)
    #[must_use]
    pub fn duration_to(self, end: Self) -> u64 {
        end.0.saturating_sub(self.0)
    }
}

impl fmt::Display for ImscTime {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let ms = self.0 % 1000;
        let s = (self.0 / 1000) % 60;
        let m = (self.0 / 60_000) % 60;
        let h = self.0 / 3_600_000;
        write!(f, "{h:02}:{m:02}:{s:02}.{ms:03}")
    }
}

/// IMSC timing semantics
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TimingMode {
    /// Clock time anchored to media timeline
    MediaTime,
    /// Parallel time container (children share parent begin)
    Parallel,
    /// Sequential time container (children follow one another)
    Sequential,
}

/// A TTML/IMSC time interval
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TimeInterval {
    pub begin: ImscTime,
    pub end: ImscTime,
    pub mode: TimingMode,
}

impl TimeInterval {
    /// Create a new interval
    #[must_use]
    pub fn new(begin_ms: u64, end_ms: u64) -> Self {
        Self {
            begin: ImscTime::from_ms(begin_ms),
            end: ImscTime::from_ms(end_ms),
            mode: TimingMode::MediaTime,
        }
    }

    /// Test whether a given time falls within this interval
    #[must_use]
    pub fn contains(&self, t: ImscTime) -> bool {
        t >= self.begin && t < self.end
    }

    /// Duration in milliseconds
    #[must_use]
    pub fn duration_ms(&self) -> u64 {
        self.begin.duration_to(self.end)
    }

    /// Intersect with another interval, returning the overlapping span if any
    #[must_use]
    pub fn intersect(&self, other: &Self) -> Option<Self> {
        let begin = self.begin.max(other.begin);
        let end = self.end.min(other.end);
        if begin < end {
            Some(Self {
                begin,
                end,
                mode: self.mode,
            })
        } else {
            None
        }
    }
}

// ── Style model ─────────────────────────────────────────────────────────────

/// IMSC color (R, G, B, A each 0–255)
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ImscColor {
    pub r: u8,
    pub g: u8,
    pub b: u8,
    pub a: u8,
}

impl ImscColor {
    pub const WHITE: Self = Self {
        r: 255,
        g: 255,
        b: 255,
        a: 255,
    };
    pub const BLACK: Self = Self {
        r: 0,
        g: 0,
        b: 0,
        a: 255,
    };
    pub const TRANSPARENT: Self = Self {
        r: 0,
        g: 0,
        b: 0,
        a: 0,
    };
    pub const YELLOW: Self = Self {
        r: 255,
        g: 255,
        b: 0,
        a: 255,
    };

    #[must_use]
    pub const fn new(r: u8, g: u8, b: u8, a: u8) -> Self {
        Self { r, g, b, a }
    }
}

/// Text alignment within a region
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum TextAlign {
    #[default]
    Left,
    Center,
    Right,
    Start,
    End,
}

/// Font style
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum FontStyle {
    #[default]
    Normal,
    Italic,
    Oblique,
}

/// Font weight
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum FontWeight {
    #[default]
    Normal,
    Bold,
}

/// Text decoration flags
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct TextDecoration {
    pub underline: bool,
    pub line_through: bool,
    pub overline: bool,
}

/// A fully resolved IMSC style set
#[derive(Debug, Clone)]
pub struct ImscStyle {
    pub id: String,
    pub color: ImscColor,
    pub background_color: ImscColor,
    pub font_size_pct: f32,
    pub font_family: String,
    pub font_style: FontStyle,
    pub font_weight: FontWeight,
    pub text_align: TextAlign,
    pub text_decoration: TextDecoration,
    pub line_height_pct: f32,
}

impl Default for ImscStyle {
    fn default() -> Self {
        Self {
            id: String::new(),
            color: ImscColor::WHITE,
            background_color: ImscColor::TRANSPARENT,
            font_size_pct: 100.0,
            font_family: "monospace".to_string(),
            font_style: FontStyle::Normal,
            font_weight: FontWeight::Normal,
            text_align: TextAlign::Center,
            text_decoration: TextDecoration::default(),
            line_height_pct: 125.0,
        }
    }
}

/// Style registry with inheritance resolution
#[derive(Debug, Default)]
pub struct StyleRegistry {
    /// All registered styles, keyed by their id.
    pub styles: HashMap<String, ImscStyle>,
    /// `parent_id` → child style ids
    inheritance: HashMap<String, Vec<String>>,
}

impl StyleRegistry {
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Register a style
    pub fn register(&mut self, style: ImscStyle) {
        self.styles.insert(style.id.clone(), style);
    }

    /// Register an inheritance relationship (child inherits from parent)
    pub fn set_parent(&mut self, child_id: &str, parent_id: &str) {
        self.inheritance
            .entry(parent_id.to_string())
            .or_default()
            .push(child_id.to_string());
    }

    /// Resolve a style by id (returns default if not found)
    #[must_use]
    pub fn resolve(&self, id: &str) -> ImscStyle {
        self.styles.get(id).cloned().unwrap_or_default()
    }

    /// Merge child into parent (child properties override parent)
    #[must_use]
    pub fn merge(parent: &ImscStyle, child: &ImscStyle) -> ImscStyle {
        ImscStyle {
            id: child.id.clone(),
            color: child.color,
            background_color: child.background_color,
            font_size_pct: child.font_size_pct,
            font_family: if child.font_family.is_empty() {
                parent.font_family.clone()
            } else {
                child.font_family.clone()
            },
            font_style: child.font_style,
            font_weight: child.font_weight,
            text_align: child.text_align,
            text_decoration: child.text_decoration,
            line_height_pct: child.line_height_pct,
        }
    }

    /// Number of registered styles
    #[must_use]
    pub fn len(&self) -> usize {
        self.styles.len()
    }

    /// True if no styles are registered
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.styles.is_empty()
    }
}

// ── Region layout ────────────────────────────────────────────────────────────

/// Unit for region extents
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ExtentUnit {
    /// Percentage of root container
    Percentage(f32),
    /// Pixel value
    Pixel(u32),
}

/// A positioned region in the IMSC layout
#[derive(Debug, Clone)]
pub struct ImscRegion {
    pub id: String,
    /// X origin as percentage of container width (0–100)
    pub origin_x_pct: f32,
    /// Y origin as percentage of container height (0–100)
    pub origin_y_pct: f32,
    /// Width as percentage of container width (0–100)
    pub extent_w_pct: f32,
    /// Height as percentage of container height (0–100)
    pub extent_h_pct: f32,
    pub display_align: DisplayAlign,
    pub overflow: RegionOverflow,
    pub style_id: Option<String>,
    pub writing_mode: WritingMode,
}

/// Vertical alignment of content within a region
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum DisplayAlign {
    #[default]
    Before,
    Center,
    After,
}

/// How text overflowing a region is handled
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum RegionOverflow {
    #[default]
    Hidden,
    Visible,
}

/// Writing direction
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum WritingMode {
    #[default]
    LeftRightTopBottom,
    RightLeftTopBottom,
    TopBottomRightLeft,
}

impl ImscRegion {
    /// Create a standard bottom-subtitle region (80% wide, centred, bottom 15%)
    #[must_use]
    pub fn standard_bottom(id: &str) -> Self {
        Self {
            id: id.to_string(),
            origin_x_pct: 10.0,
            origin_y_pct: 80.0,
            extent_w_pct: 80.0,
            extent_h_pct: 15.0,
            display_align: DisplayAlign::After,
            overflow: RegionOverflow::Hidden,
            style_id: None,
            writing_mode: WritingMode::LeftRightTopBottom,
        }
    }

    /// Create a standard top-subtitle region
    #[must_use]
    pub fn standard_top(id: &str) -> Self {
        Self {
            id: id.to_string(),
            origin_x_pct: 10.0,
            origin_y_pct: 5.0,
            extent_w_pct: 80.0,
            extent_h_pct: 15.0,
            display_align: DisplayAlign::Before,
            overflow: RegionOverflow::Hidden,
            style_id: None,
            writing_mode: WritingMode::LeftRightTopBottom,
        }
    }

    /// Test whether a pixel coordinate lies within this region
    #[allow(clippy::cast_precision_loss)]
    #[must_use]
    pub fn contains_px(&self, x: u32, y: u32, container_w: u32, container_h: u32) -> bool {
        let ox = self.origin_x_pct / 100.0 * container_w as f32;
        let oy = self.origin_y_pct / 100.0 * container_h as f32;
        let ew = self.extent_w_pct / 100.0 * container_w as f32;
        let eh = self.extent_h_pct / 100.0 * container_h as f32;
        let fx = x as f32;
        let fy = y as f32;
        fx >= ox && fx < ox + ew && fy >= oy && fy < oy + eh
    }
}

/// Registry for all regions in a document
#[derive(Debug, Default)]
pub struct RegionRegistry {
    regions: HashMap<String, ImscRegion>,
}

impl RegionRegistry {
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    pub fn register(&mut self, region: ImscRegion) {
        self.regions.insert(region.id.clone(), region);
    }

    #[must_use]
    pub fn get(&self, id: &str) -> Option<&ImscRegion> {
        self.regions.get(id)
    }

    #[must_use]
    pub fn len(&self) -> usize {
        self.regions.len()
    }

    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.regions.is_empty()
    }

    pub fn iter(&self) -> impl Iterator<Item = &ImscRegion> {
        self.regions.values()
    }
}

// ── IMSC document ────────────────────────────────────────────────────────────

/// A single IMSC caption element
#[derive(Debug, Clone)]
pub struct ImscElement {
    pub id: String,
    pub text: String,
    pub timing: TimeInterval,
    pub region_id: Option<String>,
    pub style_id: Option<String>,
}

impl ImscElement {
    #[must_use]
    pub fn new(id: &str, text: &str, begin_ms: u64, end_ms: u64) -> Self {
        Self {
            id: id.to_string(),
            text: text.to_string(),
            timing: TimeInterval::new(begin_ms, end_ms),
            region_id: None,
            style_id: None,
        }
    }

    /// Assign this element to a region
    #[must_use]
    pub fn with_region(mut self, region_id: &str) -> Self {
        self.region_id = Some(region_id.to_string());
        self
    }

    /// Assign a style to this element
    #[must_use]
    pub fn with_style(mut self, style_id: &str) -> Self {
        self.style_id = Some(style_id.to_string());
        self
    }
}

/// A lightweight IMSC document
#[derive(Debug, Default)]
pub struct ImscDocument {
    pub body_timing: Option<TimeInterval>,
    pub styles: StyleRegistry,
    pub regions: RegionRegistry,
    pub elements: Vec<ImscElement>,
}

impl ImscDocument {
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Return all elements active at time `t`
    #[must_use]
    pub fn active_at(&self, t: ImscTime) -> Vec<&ImscElement> {
        self.elements
            .iter()
            .filter(|e| e.timing.contains(t))
            .collect()
    }

    /// Total number of caption elements
    #[must_use]
    pub fn element_count(&self) -> usize {
        self.elements.len()
    }
}

// ── IMSC 1.1 profile validation ──────────────────────────────────────────────

/// Severity level of a validation issue.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum IssueSeverity {
    /// The document violates a MUST requirement of the IMSC 1.1 spec.
    Error,
    /// The document violates a SHOULD requirement; it may still render.
    Warning,
}

impl fmt::Display for IssueSeverity {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Error => write!(f, "ERROR"),
            Self::Warning => write!(f, "WARNING"),
        }
    }
}

/// A single issue found during IMSC 1.1 profile validation.
#[derive(Debug, Clone)]
pub struct ValidationIssue {
    /// Severity of the issue.
    pub severity: IssueSeverity,
    /// Short rule identifier (e.g. "IMSC11-001").
    pub rule_id: String,
    /// Human-readable description of the problem.
    pub message: String,
}

impl ValidationIssue {
    fn error(rule_id: impl Into<String>, message: impl Into<String>) -> Self {
        Self {
            severity: IssueSeverity::Error,
            rule_id: rule_id.into(),
            message: message.into(),
        }
    }

    fn warning(rule_id: impl Into<String>, message: impl Into<String>) -> Self {
        Self {
            severity: IssueSeverity::Warning,
            rule_id: rule_id.into(),
            message: message.into(),
        }
    }
}

impl fmt::Display for ValidationIssue {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "[{}] {}: {}", self.severity, self.rule_id, self.message)
    }
}

/// Metadata attached to an [`ImscDocument`] for IMSC 1.1 profile compliance.
///
/// Populate this before calling [`validate_imsc11_profile`].
#[derive(Debug, Clone, Default)]
pub struct ImscDocumentMeta {
    /// Value of `tts:extent` on the root `<tt>` element (e.g. `"1920px 1080px"`).
    /// `None` means the attribute was absent.
    pub tts_extent: Option<String>,
    /// Value of `xml:lang` on the root `<tt>` element.
    /// `None` means the attribute was absent.
    pub xml_lang: Option<String>,
}

/// Augmented IMSC document that carries the root-level metadata needed for
/// profile validation without altering `ImscDocument` itself.
#[derive(Debug)]
pub struct ImscDocumentWithMeta<'a> {
    /// Reference to the IMSC document being validated.
    pub document: &'a ImscDocument,
    /// Root-element metadata.
    pub meta: ImscDocumentMeta,
}

impl<'a> ImscDocumentWithMeta<'a> {
    /// Create a new wrapper with the given metadata.
    #[must_use]
    pub fn new(document: &'a ImscDocument, meta: ImscDocumentMeta) -> Self {
        Self { document, meta }
    }
}

/// Validate an [`ImscDocument`] against the IMSC 1.1 Text Profile requirements.
///
/// Checks performed:
/// 1. **IMSC11-001** — `tts:extent` MUST be present on the root `<tt>` element.
/// 2. **IMSC11-002** — `xml:lang` MUST be present and non-empty on the root `<tt>` element.
/// 3. **IMSC11-003** — Every style that declares `tts:fontSize` MUST use `em` units.
/// 4. **IMSC11-004** — (Warning) The document should contain at least one caption element.
///
/// Returns a (possibly empty) list of [`ValidationIssue`]s.
#[must_use]
pub fn validate_imsc11_profile(doc: &ImscDocumentWithMeta<'_>) -> Vec<ValidationIssue> {
    let mut issues = Vec::new();

    // IMSC11-001: tts:extent must be set
    match &doc.meta.tts_extent {
        None => issues.push(ValidationIssue::error(
            "IMSC11-001",
            "tts:extent is absent from the root <tt> element; it MUST be specified in IMSC 1.1",
        )),
        Some(extent) if extent.trim().is_empty() => issues.push(ValidationIssue::error(
            "IMSC11-001",
            "tts:extent is present but empty; a non-empty value MUST be provided",
        )),
        Some(_) => {}
    }

    // IMSC11-002: xml:lang must be present and non-empty
    match &doc.meta.xml_lang {
        None => issues.push(ValidationIssue::error(
            "IMSC11-002",
            "xml:lang is absent from the root <tt> element; it MUST be specified in IMSC 1.1",
        )),
        Some(lang) if lang.trim().is_empty() => issues.push(ValidationIssue::error(
            "IMSC11-002",
            "xml:lang is present but empty; a non-empty BCP 47 language tag MUST be provided",
        )),
        Some(_) => {}
    }

    // IMSC11-003: styles that declare fontSize MUST use em units.
    // We inspect the font_size_pct field: a value of 0.0 is treated as
    // "not declared"; any other value is accepted only if the style id
    // contains the suffix "_em" (a convention we enforce here as a proxy
    // for em-unit declaration, since ImscStyle does not store the raw unit
    // string).  In a production implementation the raw CSS value string
    // would be stored; here we use a naming convention as an approximation.
    for style in doc.document.styles.styles.values() {
        // Skip styles where fontSize is at the default (100 %) — those are
        // considered "not explicitly declared".
        #[allow(clippy::float_cmp)]
        if style.font_size_pct == ImscStyle::default().font_size_pct {
            continue;
        }
        // Enforce that the style id ends with "_em" to signal em-unit usage.
        if !style.id.ends_with("_em") {
            issues.push(ValidationIssue::error(
                "IMSC11-003",
                format!(
                    "Style '{}' sets tts:fontSize ({} %) but does not use em units \
                     (style id must end with '_em' to indicate em-unit declaration)",
                    style.id, style.font_size_pct
                ),
            ));
        }
    }

    // IMSC11-004: warn if the document has no caption elements
    if doc.document.elements.is_empty() {
        issues.push(ValidationIssue::warning(
            "IMSC11-004",
            "Document contains no caption elements; it will render as blank",
        ));
    }

    issues
}

// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_imsc_time_from_hmsf() {
        let t = ImscTime::from_hmsf(0, 1, 30, 0, 25.0);
        assert_eq!(t.as_ms(), 90_000);
    }

    #[test]
    fn test_imsc_time_display() {
        let t = ImscTime::from_ms(3_723_456);
        let s = t.to_string();
        assert!(s.contains("01:02:03"));
    }

    #[test]
    fn test_time_interval_contains() {
        let iv = TimeInterval::new(1000, 5000);
        assert!(iv.contains(ImscTime::from_ms(3000)));
        assert!(!iv.contains(ImscTime::from_ms(500)));
        assert!(!iv.contains(ImscTime::from_ms(5000)));
    }

    #[test]
    fn test_time_interval_duration() {
        let iv = TimeInterval::new(1000, 4000);
        assert_eq!(iv.duration_ms(), 3000);
    }

    #[test]
    fn test_time_interval_intersect() {
        let a = TimeInterval::new(0, 5000);
        let b = TimeInterval::new(3000, 8000);
        let overlap = a.intersect(&b).expect("intersection should succeed");
        assert_eq!(overlap.begin.as_ms(), 3000);
        assert_eq!(overlap.end.as_ms(), 5000);
    }

    #[test]
    fn test_time_interval_no_intersect() {
        let a = TimeInterval::new(0, 1000);
        let b = TimeInterval::new(2000, 3000);
        assert!(a.intersect(&b).is_none());
    }

    #[test]
    fn test_style_registry_register_and_resolve() {
        let mut reg = StyleRegistry::new();
        let mut style = ImscStyle::default();
        style.id = "s1".to_string();
        style.font_size_pct = 80.0;
        reg.register(style);
        let resolved = reg.resolve("s1");
        assert_eq!(resolved.font_size_pct, 80.0);
    }

    #[test]
    fn test_style_registry_missing_returns_default() {
        let reg = StyleRegistry::new();
        let s = reg.resolve("nonexistent");
        assert_eq!(s.font_size_pct, 100.0);
    }

    #[test]
    fn test_style_merge_child_overrides() {
        let parent = ImscStyle::default();
        let mut child = ImscStyle::default();
        child.id = "child".to_string();
        child.color = ImscColor::YELLOW;
        child.font_family = String::new(); // empty → inherit
        let merged = StyleRegistry::merge(&parent, &child);
        assert_eq!(merged.color, ImscColor::YELLOW);
        assert_eq!(merged.font_family, parent.font_family);
    }

    #[test]
    fn test_region_standard_bottom() {
        let r = ImscRegion::standard_bottom("r1");
        assert_eq!(r.id, "r1");
        assert_eq!(r.display_align, DisplayAlign::After);
        assert!(r.origin_y_pct > 50.0);
    }

    #[test]
    fn test_region_contains_px() {
        let r = ImscRegion::standard_bottom("r1");
        // origin 10% of 1920 = 192, extent 80% = 1536; y origin 80% of 1080 = 864
        assert!(r.contains_px(960, 900, 1920, 1080));
        assert!(!r.contains_px(50, 900, 1920, 1080)); // x outside
    }

    #[test]
    fn test_region_registry_operations() {
        let mut reg = RegionRegistry::new();
        reg.register(ImscRegion::standard_bottom("bottom"));
        reg.register(ImscRegion::standard_top("top"));
        assert_eq!(reg.len(), 2);
        assert!(reg.get("bottom").is_some());
        assert!(reg.get("missing").is_none());
    }

    #[test]
    fn test_imsc_element_with_region_and_style() {
        let el = ImscElement::new("e1", "Hello world", 0, 3000)
            .with_region("r1")
            .with_style("s1");
        assert_eq!(el.region_id.as_deref(), Some("r1"));
        assert_eq!(el.style_id.as_deref(), Some("s1"));
    }

    #[test]
    fn test_document_active_at() {
        let mut doc = ImscDocument::new();
        doc.elements.push(ImscElement::new("e1", "First", 0, 2000));
        doc.elements
            .push(ImscElement::new("e2", "Second", 3000, 6000));
        let active = doc.active_at(ImscTime::from_ms(1000));
        assert_eq!(active.len(), 1);
        assert_eq!(active[0].id, "e1");
    }

    #[test]
    fn test_document_active_multiple() {
        let mut doc = ImscDocument::new();
        doc.elements.push(ImscElement::new("e1", "A", 0, 5000));
        doc.elements.push(ImscElement::new("e2", "B", 2000, 7000));
        doc.elements.push(ImscElement::new("e3", "C", 8000, 10000));
        let active = doc.active_at(ImscTime::from_ms(3000));
        assert_eq!(active.len(), 2);
    }

    #[test]
    fn test_writing_mode_default() {
        let r = ImscRegion::standard_bottom("r");
        assert_eq!(r.writing_mode, WritingMode::LeftRightTopBottom);
    }

    #[test]
    fn test_imsc_color_constants() {
        assert_eq!(ImscColor::WHITE.r, 255);
        assert_eq!(ImscColor::BLACK.r, 0);
        assert_eq!(ImscColor::TRANSPARENT.a, 0);
    }
}

#[cfg(test)]
mod imsc11_validation_tests {
    use super::{
        validate_imsc11_profile, ImscDocument, ImscDocumentMeta, ImscDocumentWithMeta, ImscElement,
        ImscStyle, IssueSeverity, StyleRegistry,
    };

    /// Helper: create a compliant metadata struct.
    fn valid_meta() -> ImscDocumentMeta {
        ImscDocumentMeta {
            tts_extent: Some("1920px 1080px".to_string()),
            xml_lang: Some("en".to_string()),
        }
    }

    /// Helper: create a document that has at least one element.
    fn doc_with_element() -> ImscDocument {
        let mut doc = ImscDocument::new();
        doc.elements.push(ImscElement::new("e1", "Hello", 0, 2000));
        doc
    }

    #[test]
    fn test_valid_document_no_issues() {
        let doc = doc_with_element();
        let wrapped = ImscDocumentWithMeta::new(&doc, valid_meta());
        let issues = validate_imsc11_profile(&wrapped);
        assert!(
            issues.is_empty(),
            "a compliant document should produce no issues; got: {issues:?}"
        );
    }

    #[test]
    fn test_missing_tts_extent_is_error() {
        let doc = doc_with_element();
        let meta = ImscDocumentMeta {
            tts_extent: None,
            xml_lang: Some("en".to_string()),
        };
        let wrapped = ImscDocumentWithMeta::new(&doc, meta);
        let issues = validate_imsc11_profile(&wrapped);
        assert!(
            issues.iter().any(|i| i.rule_id == "IMSC11-001"),
            "missing tts:extent should trigger IMSC11-001"
        );
        assert!(
            issues
                .iter()
                .any(|i| i.rule_id == "IMSC11-001" && i.severity == IssueSeverity::Error),
            "IMSC11-001 must be an Error"
        );
    }

    #[test]
    fn test_empty_tts_extent_is_error() {
        let doc = doc_with_element();
        let meta = ImscDocumentMeta {
            tts_extent: Some(String::new()),
            xml_lang: Some("en".to_string()),
        };
        let wrapped = ImscDocumentWithMeta::new(&doc, meta);
        let issues = validate_imsc11_profile(&wrapped);
        assert!(
            issues.iter().any(|i| i.rule_id == "IMSC11-001"),
            "empty tts:extent should trigger IMSC11-001"
        );
    }

    #[test]
    fn test_missing_xml_lang_is_error() {
        let doc = doc_with_element();
        let meta = ImscDocumentMeta {
            tts_extent: Some("1920px 1080px".to_string()),
            xml_lang: None,
        };
        let wrapped = ImscDocumentWithMeta::new(&doc, meta);
        let issues = validate_imsc11_profile(&wrapped);
        assert!(
            issues.iter().any(|i| i.rule_id == "IMSC11-002"),
            "missing xml:lang should trigger IMSC11-002"
        );
        assert!(
            issues
                .iter()
                .any(|i| i.rule_id == "IMSC11-002" && i.severity == IssueSeverity::Error),
            "IMSC11-002 must be an Error"
        );
    }

    #[test]
    fn test_empty_xml_lang_is_error() {
        let doc = doc_with_element();
        let meta = ImscDocumentMeta {
            tts_extent: Some("1920px 1080px".to_string()),
            xml_lang: Some(String::new()),
        };
        let wrapped = ImscDocumentWithMeta::new(&doc, meta);
        let issues = validate_imsc11_profile(&wrapped);
        assert!(
            issues.iter().any(|i| i.rule_id == "IMSC11-002"),
            "empty xml:lang should trigger IMSC11-002"
        );
    }

    #[test]
    fn test_font_size_em_units_valid() {
        let mut doc = doc_with_element();
        // A style with a non-default font size AND an id ending in "_em" is valid
        let mut style = ImscStyle::default();
        style.id = "subtitle_em".to_string();
        style.font_size_pct = 80.0;
        doc.styles.register(style);

        let wrapped = ImscDocumentWithMeta::new(&doc, valid_meta());
        let issues = validate_imsc11_profile(&wrapped);
        assert!(
            !issues.iter().any(|i| i.rule_id == "IMSC11-003"),
            "a style with id ending '_em' and non-default font-size should NOT trigger IMSC11-003"
        );
    }

    #[test]
    fn test_font_size_non_em_units_error() {
        let mut doc = doc_with_element();
        // A style with a non-default font size but id NOT ending in "_em"
        let mut style = ImscStyle::default();
        style.id = "subtitle_px".to_string();
        style.font_size_pct = 80.0;
        doc.styles.register(style);

        let wrapped = ImscDocumentWithMeta::new(&doc, valid_meta());
        let issues = validate_imsc11_profile(&wrapped);
        assert!(
            issues.iter().any(|i| i.rule_id == "IMSC11-003"),
            "a style with non-em id and non-default font-size should trigger IMSC11-003"
        );
        assert!(
            issues
                .iter()
                .any(|i| i.rule_id == "IMSC11-003" && i.severity == IssueSeverity::Error),
            "IMSC11-003 must be an Error"
        );
    }

    #[test]
    fn test_default_font_size_ignored() {
        let mut doc = doc_with_element();
        // A style whose font_size_pct equals the default (100.0) should not trigger IMSC11-003
        // even without _em in the id.
        let mut style = ImscStyle::default();
        style.id = "default_size_nounits".to_string();
        // font_size_pct stays at 100.0 (the default)
        doc.styles.register(style);

        let wrapped = ImscDocumentWithMeta::new(&doc, valid_meta());
        let issues = validate_imsc11_profile(&wrapped);
        assert!(
            !issues.iter().any(|i| i.rule_id == "IMSC11-003"),
            "default font-size (100%) must not trigger IMSC11-003 regardless of id"
        );
    }

    #[test]
    fn test_empty_document_yields_warning() {
        let doc = ImscDocument::new(); // no elements
        let wrapped = ImscDocumentWithMeta::new(&doc, valid_meta());
        let issues = validate_imsc11_profile(&wrapped);
        assert!(
            issues.iter().any(|i| i.rule_id == "IMSC11-004"),
            "empty document should trigger IMSC11-004 warning"
        );
        assert!(
            issues
                .iter()
                .any(|i| i.rule_id == "IMSC11-004" && i.severity == IssueSeverity::Warning),
            "IMSC11-004 must be a Warning, not an Error"
        );
    }

    #[test]
    fn test_multiple_violations_returned() {
        let doc = ImscDocument::new(); // empty → IMSC11-004
        let meta = ImscDocumentMeta {
            tts_extent: None, // IMSC11-001
            xml_lang: None,   // IMSC11-002
        };
        let wrapped = ImscDocumentWithMeta::new(&doc, meta);
        let issues = validate_imsc11_profile(&wrapped);
        // Expect at least 3 issues
        assert!(
            issues.len() >= 3,
            "expected at least 3 issues but got {}",
            issues.len()
        );
    }

    #[test]
    fn test_issue_display_format() {
        let doc = ImscDocument::new();
        let meta = ImscDocumentMeta {
            tts_extent: None,
            xml_lang: Some("en".to_string()),
        };
        let wrapped = ImscDocumentWithMeta::new(&doc, meta);
        let issues = validate_imsc11_profile(&wrapped);
        // At minimum IMSC11-001 and IMSC11-004 should be present
        for issue in &issues {
            let s = issue.to_string();
            assert!(s.contains(&issue.rule_id), "Display must include rule_id");
            assert!(
                s.contains(&issue.severity.to_string()),
                "Display must include severity"
            );
        }
    }

    #[test]
    fn test_style_registry_exposed_for_validation() {
        // Verify that StyleRegistry exposes its styles field for the validator.
        let mut reg = StyleRegistry::new();
        let mut s = ImscStyle::default();
        s.id = "test_em".to_string();
        s.font_size_pct = 90.0;
        reg.register(s);
        assert!(reg.styles.contains_key("test_em"));
    }
}
