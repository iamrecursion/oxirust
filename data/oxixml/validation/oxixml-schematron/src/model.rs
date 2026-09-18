//! The raw, unresolved object model a Schematron schema parses into.
//!
//! Everything here is a direct transcription of [ISO/IEC 19757-3](https://www.iso.org/standard/55982.html)
//! element content — no `is-a` instantiation, no `extends` flattening, no phase filtering and no
//! query-binding compilation has happened yet. [`crate::compile`] turns a [`RawSchema`] into a
//! [`crate::Compiled`] schema ready to validate documents.

/// The Schematron ([ISO/IEC 19757-3](https://www.iso.org/standard/55982.html)) namespace.
pub const SCHEMATRON_NS: &str = "http://purl.oclc.org/dsdl/schematron";

/// The pre-ISO Schematron 1.5 namespace, still seen in the wild.
pub const SCHEMATRON_1_5_NS: &str = "http://www.ascc.net/xml/schematron";

/// The [SVRL](https://schematron.com/document/321.html) output namespace.
pub const SVRL_NS: &str = "http://purl.oclc.org/dsdl/svrl";

/// Whether a check is a `sch:assert` (fires when its test is *false*) or a `sch:report` (fires
/// when its test is *true*).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum CheckKind {
    /// `sch:assert` — a requirement; it is reported when the test does *not* hold.
    Assert,
    /// `sch:report` — an observation; it is reported when the test *does* hold.
    Report,
}

/// One piece of an interpolated message: `sch:assert`/`sch:report`/`sch:diagnostic` content is
/// mixed text with `sch:name` and `sch:value-of` substitutions.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum MessagePart {
    /// Literal text, copied verbatim.
    Text(String),
    /// `sch:value-of select="EXPR"` — the string value of an XPath expression, evaluated with
    /// the rule's context node in focus.
    ValueOf(String),
    /// `sch:name` or `sch:name path="EXPR"` — the (possibly computed) name of a node. `None`
    /// selects the context node itself.
    Name(Option<String>),
}

/// `sch:ns` — one namespace binding in scope for every XPath expression in the schema.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct NsBinding {
    /// The bound prefix.
    pub prefix: String,
    /// The namespace URI.
    pub uri: String,
}

/// `sch:let` — one variable binding, in scope for the expressions of its container and every
/// descendant.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LetDecl {
    /// The variable's unprefixed name.
    pub name: String,
    /// The XPath expression its value comes from.
    pub value: String,
}

/// `sch:param` — one textual substitution supplied when a concrete pattern instantiates an
/// abstract one through `is-a`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ParamDecl {
    /// The parameter name, referenced in the abstract pattern's text as `$name`.
    pub name: String,
    /// The literal replacement text.
    pub value: String,
}

/// `sch:diagnostic` — supplementary, non-normative text attached to a check by id.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DiagnosticDecl {
    /// The id other elements reference through `assert/@diagnostics` or `report/@diagnostics`.
    pub id: String,
    /// The diagnostic's interpolated message.
    pub message: Vec<MessagePart>,
}

/// `sch:assert` or `sch:report`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CheckDecl {
    /// Whether this is an assertion or a report.
    pub kind: CheckKind,
    /// The `test` XPath expression, evaluated with the rule's context node in focus.
    pub test: String,
    /// `id`, if present.
    pub id: Option<String>,
    /// `role`, a free-form severity/category annotation (e.g. `"error"`, `"warning"`).
    pub role: Option<String>,
    /// `flag`, the older severity annotation mechanism.
    pub flag: Option<String>,
    /// `subject`, an XPath expression that relocates the SVRL `location` away from the rule's
    /// context node.
    pub subject: Option<String>,
    /// The `diagnostics` attribute, split on whitespace: ids of `sch:diagnostic` elements to
    /// attach when this check fires.
    pub diagnostics: Vec<String>,
    /// The interpolated message content.
    pub message: Vec<MessagePart>,
}

/// `sch:rule`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RuleDecl {
    /// `id`, if present. Required for a rule to be an `extends` target.
    pub id: Option<String>,
    /// The context pattern nodes must match for this rule to fire on them. Absent only when
    /// `abstract="true"`.
    pub context: Option<String>,
    /// `abstract="true"`: this rule exists only to be `extend`ed and never fires on its own.
    pub is_abstract: bool,
    /// `role`, applied to every check that does not set its own.
    pub role: Option<String>,
    /// `flag`, applied to every check that does not set its own.
    pub flag: Option<String>,
    /// `sch:extends` children, in document order, interleaved with `lets` and `checks` in the
    /// position they occurred (tracked separately here because [`crate::compile`] flattens them
    /// by re-walking the original order via [`RuleDecl::body`]).
    pub body: Vec<RuleItem>,
}

/// One child of `sch:rule`, in document order — the shape [`crate::compile`] needs to flatten
/// `extends` at exactly the position it occurred.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RuleItem {
    /// A `sch:let`.
    Let(LetDecl),
    /// A `sch:assert` or `sch:report`.
    Check(CheckDecl),
    /// A `sch:extends rule="id"` — splice the referenced rule's own flattened lets and checks in
    /// at this point.
    Extends(String),
    /// A `sch:extends href="…"` — the fetched fragment, already parsed, spliced in at this
    /// point instead of being looked up in the schema's rule registry.
    InlineExtends(Box<RuleDecl>),
}

/// `sch:pattern`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PatternDecl {
    /// `id`, if present.
    pub id: Option<String>,
    /// `abstract="true"`: this pattern is a template, only usable through another pattern's
    /// `is-a`.
    pub is_abstract: bool,
    /// `is-a="id"`: instantiate the named abstract pattern, substituting [`PatternDecl::params`].
    pub is_a: Option<String>,
    /// `sch:title`, if present.
    pub title: Option<String>,
    /// The pattern's own `sch:let` declarations, in scope for every rule in it.
    pub lets: Vec<LetDecl>,
    /// The pattern's rules, in document order — the order that decides which rule fires first
    /// when more than one rule's context matches the same node.
    pub rules: Vec<RuleDecl>,
    /// `sch:param` children, used only when [`PatternDecl::is_a`] is set.
    pub params: Vec<ParamDecl>,
}

/// `sch:active` — one pattern activated by a phase.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PhaseDecl {
    /// `id`.
    pub id: String,
    /// The `pattern` ids named by this phase's `sch:active` children.
    pub active: Vec<String>,
    /// The phase's own `sch:let` declarations.
    pub lets: Vec<LetDecl>,
}

/// A fully parsed `sch:schema`, before `is-a`/`extends`/phase resolution.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RawSchema {
    /// `sch:title`, if present.
    pub title: Option<String>,
    /// `schemaVersion`, if present.
    pub schema_version: Option<String>,
    /// `defaultPhase`, if present — the phase [`crate::Schematron::compile`] uses when the
    /// caller does not name one.
    pub default_phase: Option<String>,
    /// `queryBinding`, if present — selects the [`oxixml_xpath::XPathVersion`] every expression
    /// in the schema compiles under. See [`crate::compile::query_binding_version`].
    pub query_binding: Option<String>,
    /// `sch:ns` bindings, in scope for every XPath expression in the schema.
    pub namespaces: Vec<NsBinding>,
    /// The schema's own, top-level `sch:let` declarations.
    pub lets: Vec<LetDecl>,
    /// `sch:phase` declarations.
    pub phases: Vec<PhaseDecl>,
    /// `sch:pattern` declarations, in document order.
    pub patterns: Vec<PatternDecl>,
    /// `sch:diagnostics/sch:diagnostic` declarations.
    pub diagnostics: Vec<DiagnosticDecl>,
}
