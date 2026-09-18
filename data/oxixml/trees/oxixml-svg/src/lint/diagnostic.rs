//! Lint diagnostics: [`Severity`], [`Diagnostic`] itself, the [`codes`] a
//! [`crate::lint::rules::Linter`] can report, and the [`LintOptions`] that
//! choose which rule families run.
//!
//! This module only defines the *vocabulary* of linting (what a diagnostic
//! looks like, what its possible codes are, and what knobs a caller can
//! turn); [`crate::lint::rules`] is where the rules that actually produce
//! [`Diagnostic`] values live.

use std::fmt;

use oxixml_dom::NodeId;

/// How serious a [`Diagnostic`] is.
///
/// Ordered from least to most severe: [`Self::Info`] `<` [`Self::Warning`]
/// `<` [`Self::Error`]. The derived [`Ord`]/[`PartialOrd`] rely on this
/// declaration order, so a list of diagnostics can be sorted or filtered by
/// severity (`diagnostics.iter().filter(|d| d.severity() >= Severity::Warning)`)
/// without a hand-written comparison.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum Severity {
    /// Informational: worth surfacing (e.g. in a "modernize this document"
    /// report) but not evidence of a mistake. Used for the ubiquitous,
    /// harmless `xlink:href` deprecation and for the linter's own
    /// truncation notice.
    Info,
    /// Spec-legal but discouraged, non-conformant, or very likely a
    /// mistake: an unknown element/attribute, a content-model violation, a
    /// dangling reference. The document still parses and (mostly) still
    /// renders; something in it should probably be fixed.
    Warning,
    /// A conformance failure serious enough that the affected node (or, for
    /// [`codes::NOT_SVG`], the whole document) cannot be treated as valid
    /// SVG: a malformed grammar the specification calls "in error", a
    /// duplicate `id`, or a reference cycle.
    Error,
}

impl fmt::Display for Severity {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Self::Info => "info",
            Self::Warning => "warning",
            Self::Error => "error",
        })
    }
}

/// One finding from [`crate::lint::rules::Linter::lint`]: a rule [`code`](Diagnostic::code),
/// a [`severity`](Diagnostic::severity), a human-readable
/// [`message`](Diagnostic::message), and the [`node`](Diagnostic::node) the
/// finding is about.
///
/// `node` is always a node of the [`oxixml_dom::Document`] the linter was
/// given — usually the element or attribute the rule fired on, occasionally
/// (for [`codes::NOT_SVG`] when the document has no element at all, and for
/// [`codes::TRUNCATED`]) a less specific anchor documented on the code
/// itself.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Diagnostic {
    code: &'static str,
    severity: Severity,
    message: String,
    node: NodeId,
}

impl Diagnostic {
    /// Creates a diagnostic from its four parts.
    ///
    /// `code` is expected to be one of the [`codes`] constants, but this is
    /// not enforced — a caller building a custom lint pass on top of this
    /// crate may mint its own codes.
    #[must_use]
    pub fn new(
        code: &'static str,
        severity: Severity,
        message: impl Into<String>,
        node: NodeId,
    ) -> Self {
        Self {
            code,
            severity,
            message: message.into(),
            node,
        }
    }

    /// Creates an [`Severity::Error`] diagnostic.
    #[must_use]
    pub fn error(code: &'static str, message: impl Into<String>, node: NodeId) -> Self {
        Self::new(code, Severity::Error, message, node)
    }

    /// Creates a [`Severity::Warning`] diagnostic.
    #[must_use]
    pub fn warning(code: &'static str, message: impl Into<String>, node: NodeId) -> Self {
        Self::new(code, Severity::Warning, message, node)
    }

    /// Creates a [`Severity::Info`] diagnostic.
    #[must_use]
    pub fn info(code: &'static str, message: impl Into<String>, node: NodeId) -> Self {
        Self::new(code, Severity::Info, message, node)
    }

    /// The rule code that produced this diagnostic, e.g. `"duplicate-id"`.
    ///
    /// See [`codes`] for the full catalog of codes this crate's own linter
    /// produces.
    #[must_use]
    pub const fn code(&self) -> &'static str {
        self.code
    }

    /// How serious this finding is.
    #[must_use]
    pub const fn severity(&self) -> Severity {
        self.severity
    }

    /// The human-readable explanation of this finding.
    #[must_use]
    pub fn message(&self) -> &str {
        &self.message
    }

    /// The node this finding is about.
    #[must_use]
    pub const fn node(&self) -> NodeId {
        self.node
    }
}

impl fmt::Display for Diagnostic {
    /// Formats as `"{severity}: {message} [{code}]"`, e.g.
    /// `"warning: unknown attribute \`foo\` [unknown-attribute]"`.
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}: {} [{}]", self.severity, self.message, self.code)
    }
}

/// The rule codes [`crate::lint::rules::Linter`] can report, one `const`
/// per rule.
///
/// Every code is a stable, kebab-case string suitable for machine matching
/// (`diagnostic.code() == codes::DANGLING_REFERENCE`) as well as display.
/// The doc comment on each constant names the exact condition that rule
/// fires under and the [`Severity`] it fires at; see
/// [`crate::lint::rules`]'s module documentation for the handful of
/// attribute/element families that are deliberately **not** covered by any
/// code (no grammar implemented, or not part of this release's scope).
pub mod codes {
    /// The document has no element, or its document element is not an
    /// `svg` element in [`crate::ns::SVG_NAMESPACE`].
    ///
    /// [`super::Severity::Error`]. Fires at most once, and when it fires it is the
    /// *only* diagnostic [`crate::lint::rules::Linter::lint`] returns — a
    /// document that fails this check is not SVG at all, so nothing else
    /// about it is meaningfully lintable. Attached to
    /// [`oxixml_dom::NodeId::DOCUMENT`] when the document has no element,
    /// otherwise to the (non-`svg`) document element that was found.
    pub const NOT_SVG: &str = "not-svg";

    /// An element in [`crate::ns::SVG_NAMESPACE`] whose local name is not
    /// one [`crate::names::elements::ElementId::from_name`] recognizes.
    ///
    /// [`super::Severity::Warning`]. The element's attributes and content are not
    /// further checked (see [`crate::lint::rules`] module docs), but its
    /// `id` is still collected and its children are still walked.
    pub const UNKNOWN_ELEMENT: &str = "unknown-element";

    /// An authored attribute (in no namespace, `xml:`, or `xlink:`) whose
    /// name [`crate::names::attrs::AttrId::from_name`] does not recognize.
    ///
    /// [`super::Severity::Warning`]. `xmlns`/`xmlns:*` declarations and the
    /// open-ended `data-*`/`aria-*` prefixes are never reported under this
    /// code — see [`crate::lint::rules`] module docs.
    pub const UNKNOWN_ATTRIBUTE: &str = "unknown-attribute";

    /// A recognized attribute is authored on an element
    /// [`crate::names::content::attribute_applies`] says it does not apply
    /// to (e.g. `cx` on a `rect`).
    ///
    /// [`super::Severity::Warning`].
    pub const ATTRIBUTE_NOT_APPLICABLE: &str = "attribute-not-applicable";

    /// A presentation or geometry attribute's value fails the microsyntax
    /// grammar this crate implements for it.
    ///
    /// [`super::Severity::Warning`]. The message includes the underlying
    /// [`crate::error::ValueError`]'s [`std::fmt::Display`] text, which
    /// carries a byte offset into the attribute value. See
    /// [`INVALID_VIEW_BOX`] and [`NEGATIVE_DIMENSION`] for the two grammar
    /// families that are reported under a more specific code instead of
    /// this one.
    pub const INVALID_ATTRIBUTE_VALUE: &str = "invalid-attribute-value";

    /// The `style` attribute is not a well-formed CSS declaration list per
    /// [`crate::style::attr::parse_style_attribute_strict`].
    ///
    /// [`super::Severity::Warning`].
    pub const INVALID_STYLE_ATTRIBUTE: &str = "invalid-style-attribute";

    /// An `id` value that has already been seen earlier in document order.
    ///
    /// [`super::Severity::Error`]. XML "first wins": the earlier element keeps the
    /// id for reference-resolution purposes, and this diagnostic is
    /// attached to each *later* element repeating it.
    pub const DUPLICATE_ID: &str = "duplicate-id";

    /// A `url(#x)`/`href="#x"`-shaped local reference whose target `id`
    /// does not exist anywhere in the document.
    ///
    /// [`super::Severity::Warning`]. Only fragment-only (`#id`) references are
    /// checked; an absolute or relative external reference is silently not
    /// checked (this crate does not resolve external documents).
    pub const DANGLING_REFERENCE: &str = "dangling-reference";

    /// A `use`/gradient/pattern/filter `href` (or `xlink:href`) chain that
    /// loops back on itself.
    ///
    /// [`super::Severity::Error`]. One diagnostic per cycle, attached to the entry
    /// node (the first `use`/gradient/pattern/filter whose expansion was
    /// found to recur) rather than to every node on the cycle.
    pub const CIRCULAR_REFERENCE: &str = "circular-reference";

    /// A `viewBox` value with a negative `width` or `height`.
    ///
    /// [`super::Severity::Warning`]. A more specific code than
    /// [`INVALID_ATTRIBUTE_VALUE`] for exactly this
    /// [`crate::error::ValueError::OutOfRange`] case; any other `viewBox`
    /// grammar violation is still reported as
    /// [`INVALID_ATTRIBUTE_VALUE`]. A `viewBox` of exactly zero width or
    /// height is grammar-valid (it disables rendering, per the
    /// specification) and is not reported under either code.
    pub const INVALID_VIEW_BOX: &str = "invalid-view-box";

    /// A `width`, `height`, `r`, `rx`, or `ry` value that parses as a valid
    /// [`crate::values::length::Length`] but is negative.
    ///
    /// [`super::Severity::Error`]. These five attributes are the ones the
    /// specification defines as accepting no negative magnitude; other
    /// signed geometry attributes (`x`, `y`, `dx`, `dy`, ...) are not
    /// checked for sign at all.
    pub const NEGATIVE_DIMENSION: &str = "negative-dimension";

    /// An element whose [`crate::names::elements::ElementId::status`] is
    /// [`crate::names::elements::ElementStatus::DeprecatedSvg2`].
    ///
    /// [`super::Severity::Warning`]. The message names a suggested replacement
    /// when the specification catalog names an obvious one (`tref` →
    /// `use`, `animateColor` → `animate`, the SVG-fonts family → CSS
    /// Fonts/WOFF, `altGlyph`/... → plain text runs). Draft (`meshgradient`,
    /// `hatch`, ...) and experimental (`discard`) elements are a different
    /// [`crate::names::elements::ElementStatus`] and are **not** reported
    /// under this code.
    pub const DEPRECATED_ELEMENT: &str = "deprecated-element";

    /// An attribute [`crate::names::attrs::AttrId::deprecated_in_svg2`]
    /// reports `true` for.
    ///
    /// [`super::Severity::Info`], not `Warning` — deliberately quieter than
    /// [`DEPRECATED_ELEMENT`] because the most common member of this set,
    /// `xlink:href`, remains valid indefinitely and is ubiquitous in
    /// perfectly ordinary SVG 1.1 content.
    pub const DEPRECATED_ATTRIBUTE: &str = "deprecated-attribute";

    /// A child element that
    /// [`crate::names::content::allows_child`] says its parent's content
    /// model does not permit.
    ///
    /// [`super::Severity::Warning`]. Attached to the *child* node.
    pub const CONTENT_MODEL_VIOLATION: &str = "content-model-violation";

    /// A non-whitespace text (or CDATA) node inside an element
    /// [`crate::names::content::allows_character_data`] reports `false`
    /// for.
    ///
    /// [`super::Severity::Warning`]. Whitespace-only text is never reported.
    /// Comments are never reported (a comment is not "character data
    /// content" in the content-model sense, regardless of what element it
    /// is inside).
    pub const CHARACTER_DATA_NOT_ALLOWED: &str = "character-data-not-allowed";

    /// An element carries both a bare `href` and an `xlink:href`.
    ///
    /// [`super::Severity::Warning`]. Not an error — SVG 2 defines `href` as
    /// winning in this case — but almost always an authoring accident (a
    /// tool that "modernized" a reference without removing the old one).
    pub const HREF_DUPLICATED: &str = "href-duplicated";

    /// A `use` element with no `href`/`xlink:href` at all, or one whose
    /// (whitespace-trimmed) value is empty.
    ///
    /// [`super::Severity::Warning`]. Distinguished from [`DANGLING_REFERENCE`]
    /// because "this `use` cannot possibly instantiate anything" is a more
    /// specific and usually more obviously-a-mistake condition than "this
    /// `use` points at an id that happens not to exist".
    pub const EMPTY_USE_REFERENCE: &str = "empty-use-reference";

    /// The linter found more diagnostics than
    /// [`super::LintOptions::max_diagnostics_limit`] allows.
    ///
    /// [`super::Severity::Info`]. Emitted at most once, as the *last* diagnostic
    /// in the returned list (after document-order sorting), with a message
    /// reporting how many further diagnostics were suppressed. Attached to
    /// the SVG root element. This diagnostic is additional to the
    /// configured limit — the returned vector may hold up to
    /// `max_diagnostics + 1` entries, the last being this one.
    pub const TRUNCATED: &str = "truncated";
}

/// Which rule families [`crate::lint::rules::Linter::lint`] runs, and how
/// many diagnostics it reports before giving up.
///
/// Every check defaults to on. Each `check_*` builder method documents
/// exactly which [`codes`] constants it gates; [`codes::NOT_SVG`],
/// [`codes::DUPLICATE_ID`], and [`codes::TRUNCATED`] are not gated by any
/// toggle here — they are structural/document-level facts the linter
/// always reports regardless of which rule families are enabled.
///
/// # Example
/// ```
/// use oxixml_svg::lint::diagnostic::LintOptions;
///
/// let options = LintOptions::new()
///     .check_deprecated(false)
///     .max_diagnostics(500);
/// assert!(!options.checks_deprecated());
/// assert!(options.checks_values(), "everything else stays on by default");
/// assert_eq!(options.max_diagnostics_limit(), 500);
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct LintOptions {
    check_values: bool,
    check_content_model: bool,
    check_references: bool,
    check_deprecated: bool,
    check_unknown: bool,
    max_diagnostics: usize,
}

/// The default cap on the number of diagnostics a single [`crate::lint::rules::Linter::lint`]
/// call reports, before [`codes::TRUNCATED`] is appended.
pub const DEFAULT_MAX_DIAGNOSTICS: usize = 10_000;

impl Default for LintOptions {
    fn default() -> Self {
        Self::new()
    }
}

impl LintOptions {
    /// Every check enabled, [`DEFAULT_MAX_DIAGNOSTICS`] as the cap. Same as
    /// [`Default::default`].
    #[must_use]
    pub const fn new() -> Self {
        Self {
            check_values: true,
            check_content_model: true,
            check_references: true,
            check_deprecated: true,
            check_unknown: true,
            max_diagnostics: DEFAULT_MAX_DIAGNOSTICS,
        }
    }

    /// Enables or disables microsyntax value validation:
    /// [`codes::INVALID_ATTRIBUTE_VALUE`], [`codes::INVALID_STYLE_ATTRIBUTE`],
    /// [`codes::INVALID_VIEW_BOX`], and [`codes::NEGATIVE_DIMENSION`].
    #[must_use]
    pub const fn check_values(mut self, enabled: bool) -> Self {
        self.check_values = enabled;
        self
    }

    /// Enables or disables content-model checking:
    /// [`codes::CONTENT_MODEL_VIOLATION`] and [`codes::CHARACTER_DATA_NOT_ALLOWED`].
    #[must_use]
    pub const fn check_content_model(mut self, enabled: bool) -> Self {
        self.check_content_model = enabled;
        self
    }

    /// Enables or disables reference-resolution checking:
    /// [`codes::DANGLING_REFERENCE`], [`codes::CIRCULAR_REFERENCE`],
    /// [`codes::HREF_DUPLICATED`], and [`codes::EMPTY_USE_REFERENCE`].
    #[must_use]
    pub const fn check_references(mut self, enabled: bool) -> Self {
        self.check_references = enabled;
        self
    }

    /// Enables or disables deprecation checking: [`codes::DEPRECATED_ELEMENT`]
    /// and [`codes::DEPRECATED_ATTRIBUTE`].
    #[must_use]
    pub const fn check_deprecated(mut self, enabled: bool) -> Self {
        self.check_deprecated = enabled;
        self
    }

    /// Enables or disables unknown/inapplicable-name checking:
    /// [`codes::UNKNOWN_ELEMENT`], [`codes::UNKNOWN_ATTRIBUTE`], and
    /// [`codes::ATTRIBUTE_NOT_APPLICABLE`].
    #[must_use]
    pub const fn check_unknown(mut self, enabled: bool) -> Self {
        self.check_unknown = enabled;
        self
    }

    /// Sets the cap on the number of diagnostics returned before
    /// [`codes::TRUNCATED`] is appended (see that code's documentation for
    /// the exact truncation behavior, including the `0` case).
    #[must_use]
    pub const fn max_diagnostics(mut self, limit: usize) -> Self {
        self.max_diagnostics = limit;
        self
    }

    /// Whether microsyntax value validation is enabled.
    #[must_use]
    pub const fn checks_values(&self) -> bool {
        self.check_values
    }

    /// Whether content-model checking is enabled.
    #[must_use]
    pub const fn checks_content_model(&self) -> bool {
        self.check_content_model
    }

    /// Whether reference-resolution checking is enabled.
    #[must_use]
    pub const fn checks_references(&self) -> bool {
        self.check_references
    }

    /// Whether deprecation checking is enabled.
    #[must_use]
    pub const fn checks_deprecated(&self) -> bool {
        self.check_deprecated
    }

    /// Whether unknown/inapplicable-name checking is enabled.
    #[must_use]
    pub const fn checks_unknown(&self) -> bool {
        self.check_unknown
    }

    /// The current cap on the number of diagnostics returned before
    /// [`codes::TRUNCATED`] is appended.
    #[must_use]
    pub const fn max_diagnostics_limit(&self) -> usize {
        self.max_diagnostics
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // ---- Severity ----

    #[test]
    fn severity_orders_info_below_warning_below_error() {
        assert!(Severity::Info < Severity::Warning);
        assert!(Severity::Warning < Severity::Error);
        assert!(Severity::Info < Severity::Error);
        assert_eq!(
            Severity::Info.cmp(&Severity::Info),
            std::cmp::Ordering::Equal
        );
    }

    #[test]
    fn severity_display_is_lowercase() {
        assert_eq!(Severity::Info.to_string(), "info");
        assert_eq!(Severity::Warning.to_string(), "warning");
        assert_eq!(Severity::Error.to_string(), "error");
    }

    // ---- Diagnostic ----

    #[test]
    fn constructors_set_the_expected_severity() {
        let node = NodeId::from_index(3);
        assert_eq!(
            Diagnostic::error(codes::DUPLICATE_ID, "boom", node).severity(),
            Severity::Error
        );
        assert_eq!(
            Diagnostic::warning(codes::UNKNOWN_ATTRIBUTE, "boom", node).severity(),
            Severity::Warning
        );
        assert_eq!(
            Diagnostic::info(codes::DEPRECATED_ATTRIBUTE, "boom", node).severity(),
            Severity::Info
        );
    }

    #[test]
    fn getters_report_back_what_was_constructed() {
        let node = NodeId::from_index(7);
        let diagnostic = Diagnostic::new(
            codes::DANGLING_REFERENCE,
            Severity::Warning,
            "reference to nonexistent id",
            node,
        );
        assert_eq!(diagnostic.code(), codes::DANGLING_REFERENCE);
        assert_eq!(diagnostic.severity(), Severity::Warning);
        assert_eq!(diagnostic.message(), "reference to nonexistent id");
        assert_eq!(diagnostic.node(), node);
    }

    #[test]
    fn display_matches_the_documented_format() {
        let diagnostic = Diagnostic::new(
            codes::UNKNOWN_ATTRIBUTE,
            Severity::Warning,
            "unknown attribute `foo`",
            NodeId::from_index(1),
        );
        assert_eq!(
            diagnostic.to_string(),
            "warning: unknown attribute `foo` [unknown-attribute]"
        );
    }

    // ---- codes ----

    #[test]
    fn every_code_is_a_distinct_kebab_case_string() {
        let all = [
            codes::NOT_SVG,
            codes::UNKNOWN_ELEMENT,
            codes::UNKNOWN_ATTRIBUTE,
            codes::ATTRIBUTE_NOT_APPLICABLE,
            codes::INVALID_ATTRIBUTE_VALUE,
            codes::INVALID_STYLE_ATTRIBUTE,
            codes::DUPLICATE_ID,
            codes::DANGLING_REFERENCE,
            codes::CIRCULAR_REFERENCE,
            codes::INVALID_VIEW_BOX,
            codes::NEGATIVE_DIMENSION,
            codes::DEPRECATED_ELEMENT,
            codes::DEPRECATED_ATTRIBUTE,
            codes::CONTENT_MODEL_VIOLATION,
            codes::CHARACTER_DATA_NOT_ALLOWED,
            codes::HREF_DUPLICATED,
            codes::EMPTY_USE_REFERENCE,
            codes::TRUNCATED,
        ];
        assert_eq!(all.len(), 18);
        let unique: std::collections::HashSet<_> = all.iter().collect();
        assert_eq!(unique.len(), all.len(), "every code must be distinct");
        for code in all {
            assert!(
                code.chars().all(|c| c.is_ascii_lowercase() || c == '-'),
                "{code:?} must be kebab-case"
            );
            assert!(!code.starts_with('-') && !code.ends_with('-'));
        }
    }

    // ---- LintOptions ----

    #[test]
    fn default_enables_every_check() {
        let options = LintOptions::new();
        assert!(options.checks_values());
        assert!(options.checks_content_model());
        assert!(options.checks_references());
        assert!(options.checks_deprecated());
        assert!(options.checks_unknown());
        assert_eq!(options.max_diagnostics_limit(), DEFAULT_MAX_DIAGNOSTICS);
        assert_eq!(options, LintOptions::default());
    }

    #[test]
    fn builder_methods_toggle_one_flag_at_a_time() {
        let options = LintOptions::new().check_values(false);
        assert!(!options.checks_values());
        assert!(options.checks_content_model(), "unrelated flags stay on");

        let options = LintOptions::new().check_content_model(false);
        assert!(!options.checks_content_model());

        let options = LintOptions::new().check_references(false);
        assert!(!options.checks_references());

        let options = LintOptions::new().check_deprecated(false);
        assert!(!options.checks_deprecated());

        let options = LintOptions::new().check_unknown(false);
        assert!(!options.checks_unknown());
    }

    #[test]
    fn max_diagnostics_is_configurable() {
        let options = LintOptions::new().max_diagnostics(5);
        assert_eq!(options.max_diagnostics_limit(), 5);
    }

    #[test]
    fn builder_methods_chain() {
        let options = LintOptions::new()
            .check_values(false)
            .check_content_model(false)
            .check_references(false)
            .check_deprecated(false)
            .check_unknown(false)
            .max_diagnostics(1);
        assert!(!options.checks_values());
        assert!(!options.checks_content_model());
        assert!(!options.checks_references());
        assert!(!options.checks_deprecated());
        assert!(!options.checks_unknown());
        assert_eq!(options.max_diagnostics_limit(), 1);
    }
}
