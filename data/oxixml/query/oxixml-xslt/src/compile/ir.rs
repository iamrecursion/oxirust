//! The compiled form of a stylesheet: instructions, declarations and the tables the engine
//! looks them up in.
//!
//! Nothing here is public API. A [`CompiledStylesheet`](crate::CompiledStylesheet) owns one
//! [`StylesheetIr`] and the engine walks it; keeping the two apart means the engine never has
//! to re-inspect the source tree.

use crate::compile::env::{Avt, XsltVersion};
use crate::output::OutputProperties;
use crate::pattern::Pattern;
use crate::typing::RequiredType;
use oxixml_infoset::{ExpandedName, SourcePosition};
use oxixml_xpath::{CompiledXPath, SequenceType};
use std::collections::HashMap;
use std::sync::Arc;

/// Which mode an `xsl:apply-templates` selects.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub(crate) enum ModeRef {
    /// `#default`, or no `mode` attribute at all.
    Default,
    /// `#current` — keep the mode already in force.
    Current,
    /// A named mode.
    Named(ExpandedName),
}

/// One `xsl:catch` inside an `xsl:try`.
#[derive(Debug)]
pub(crate) struct CatchClause {
    /// The `errors` attribute, as name tests; empty means every error is caught.
    pub(crate) errors: Vec<CatchTest>,
    pub(crate) select: Option<CompiledXPath>,
    pub(crate) content: Vec<Instruction>,
}

/// One token of an `xsl:catch/@errors` list.
#[derive(Debug)]
pub(crate) enum CatchTest {
    /// `*` — any error.
    Any,
    /// `prefix:*` — any error in this namespace.
    AnyLocal(Option<String>),
    /// `*:local` — this local name in any namespace.
    AnyNamespace(String),
    /// A fully resolved name.
    Name(ExpandedName),
}

impl CatchTest {
    /// Whether the test admits an error with this name.
    pub(crate) fn admits(&self, namespace: Option<&str>, local: &str) -> bool {
        match self {
            Self::Any => true,
            Self::AnyLocal(uri) => uri.as_deref() == namespace,
            Self::AnyNamespace(name) => name == local,
            Self::Name(name) => name.namespace() == namespace && name.local_name() == local,
        }
    }
}

/// The `xsl:result-document` attributes that only settle when the instruction runs.
///
/// [XSLT 3.0 §23.1](https://www.w3.org/TR/xslt-30/#element-result-document) makes `format` and
/// every serialization parameter on the instruction an *attribute value template*, so a value
/// carrying an embedded expression is not there to be judged while the stylesheet is read. Only
/// those attributes are held here — a constant is folded into the instruction's
/// [`OutputProperties`] at compile time, and the whole record is absent when every attribute is
/// constant, which is the ordinary case.
#[derive(Debug)]
pub(crate) struct DeferredOutput {
    /// `format`, naming an output definition to start from.
    pub(crate) format: Option<Avt>,
    /// The serialization parameters, each paired with the attribute's local name.
    pub(crate) parameters: Vec<(String, Avt)>,
    /// The code point map the instruction's own `use-character-maps` builds, which supplements
    /// whatever the output definition already names.
    pub(crate) character_map: std::collections::BTreeMap<char, String>,
    /// The namespace bindings in scope where the instruction is written, which is what a
    /// `QName` in `cdata-section-elements` or in `format` is expanded against.
    pub(crate) namespaces: Arc<Vec<(String, String)>>,
    /// The XSLT version in force there, which decides how wide a boolean parameter's lexical
    /// space is.
    pub(crate) version: XsltVersion,
}

/// One `xsl:with-param`.
#[derive(Debug)]
pub(crate) struct WithParam {
    pub(crate) name: ExpandedName,
    pub(crate) select: Option<CompiledXPath>,
    pub(crate) content: Vec<Instruction>,
    pub(crate) tunnel: bool,
    pub(crate) as_type: Option<Arc<RequiredType>>,
    /// The static base URI of the binding element — see [`VariableDecl::base_uri`].
    pub(crate) base_uri: Option<Arc<str>>,
}

/// One `xsl:param` on a template, function or `xsl:iterate`.
#[derive(Debug)]
pub(crate) struct ParamDecl {
    pub(crate) name: ExpandedName,
    pub(crate) select: Option<CompiledXPath>,
    pub(crate) content: Vec<Instruction>,
    pub(crate) required: bool,
    pub(crate) tunnel: bool,
    pub(crate) as_type: Option<Arc<RequiredType>>,
    /// The static base URI of the binding element — see [`VariableDecl::base_uri`].
    pub(crate) base_uri: Option<Arc<str>>,
}

impl ParamDecl {
    /// Whether the parameter is *implicitly mandatory* —
    /// [XSLT 3.0 §9.3](https://www.w3.org/TR/xslt-30/#parameters): "If a parameter that is not
    /// explicitly mandatory has no explicit default value, then it has an implicit default
    /// value, which is the empty sequence if there is an `as` attribute … If a parameter has an
    /// implicit default value which cannot be converted to the required type (that is, if it
    /// has an `as` attribute which does not permit the empty sequence), then the parameter is
    /// implicitly mandatory."
    ///
    /// Such a parameter behaves exactly as if `required="yes"` had been written, so a call that
    /// supplies no value is `XTDE0700` rather than a failed conversion of the default.
    pub(crate) fn is_implicitly_mandatory(&self) -> bool {
        !self.required
            && self.select.is_none()
            && self.content.is_empty()
            && self
                .as_type
                .as_ref()
                .is_some_and(|as_type| !as_type.permits_empty())
    }
}

/// One `xsl:sort`.
#[derive(Debug)]
pub(crate) struct SortKey {
    pub(crate) select: Option<CompiledXPath>,
    pub(crate) content: Vec<Instruction>,
    pub(crate) order: Avt,
    /// `data-type`, kept as an `Option` so that the runtime can tell an explicit `text` from the
    /// attribute being absent. The two differ from XSLT 2.0 on: `data-type="text"` forces the
    /// key to be compared as a string, where an absent `data-type` compares the atomized key with
    /// the `lt` operator of its own type. XSLT 1.0's schema defaults the attribute to `text`, so
    /// a 1.0 module is compiled as though it had written it.
    pub(crate) data_type: Option<Avt>,
    /// `case-order`, kept as an `Option` so that the runtime can tell an explicit
    /// `upper-first` from the attribute being absent.
    pub(crate) case_order: Option<Avt>,
    pub(crate) language: Option<Avt>,
    pub(crate) collation: Option<Avt>,
    /// The default collation in force where the `xsl:sort` was written, which XSLT 3.0 §13.1
    /// falls back on once neither `collation` nor `lang` has named one. It is a static property
    /// (§5.4.4), so the nearest `[xsl:]default-collation` ancestor settles it here rather than
    /// the stylesheet-wide one the runtime carries.
    pub(crate) default_collation: Option<String>,
    pub(crate) stable: bool,
}

/// How `xsl:for-each-group` forms its groups.
#[derive(Debug)]
pub(crate) enum Grouping {
    /// `group-by`.
    By(CompiledXPath),
    /// `group-adjacent`.
    Adjacent(CompiledXPath),
    /// `group-starting-with`.
    StartingWith(Pattern),
    /// `group-ending-with`.
    EndingWith(Pattern),
}

/// The `level` of an `xsl:number`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub(crate) enum NumberLevel {
    #[default]
    Single,
    Multiple,
    Any,
}

/// A compiled `xsl:number`.
#[derive(Debug)]
pub(crate) struct NumberSpec {
    pub(crate) value: Option<CompiledXPath>,
    pub(crate) select: Option<CompiledXPath>,
    pub(crate) level: NumberLevel,
    pub(crate) count: Option<Pattern>,
    pub(crate) from: Option<Pattern>,
    pub(crate) format: Avt,
    pub(crate) letter_value: Option<Avt>,
    pub(crate) grouping_separator: Option<Avt>,
    pub(crate) grouping_size: Option<Avt>,
    pub(crate) ordinal: Option<Avt>,
    /// `start-at` (XSLT 3.0): the value the first number in the sequence takes.
    pub(crate) start_at: Option<Avt>,
    /// `lang`, kept only so that its value can be checked against `xs:language` at run time.
    pub(crate) language: Option<Avt>,
}

/// A literal result element.
#[derive(Debug)]
pub(crate) struct LiteralElement {
    pub(crate) name: ExpandedName,
    pub(crate) prefix: Option<String>,
    /// The namespace bindings to copy onto the result element.
    pub(crate) namespaces: Vec<(Option<String>, String)>,
    pub(crate) attributes: Vec<(ExpandedName, Option<String>, Avt)>,
    pub(crate) use_attribute_sets: Vec<ExpandedName>,
    /// `xsl:inherit-namespaces`: whether the elements built inside the content see this
    /// element's namespaces — [XSLT 3.0 §11.7.1](https://www.w3.org/TR/xslt-30/#copy-namespaces).
    /// The default is `yes`.
    pub(crate) inherit_namespaces: bool,
    pub(crate) content: Vec<Instruction>,
}

/// A variable or parameter binding written inside a sequence constructor.
#[derive(Debug)]
pub(crate) struct VariableDecl {
    pub(crate) name: ExpandedName,
    pub(crate) select: Option<CompiledXPath>,
    pub(crate) content: Vec<Instruction>,
    pub(crate) as_type: Option<Arc<RequiredType>>,
    /// The static base URI of the binding element, which is the base URI of the temporary tree
    /// its sequence constructor builds — [XSLT 3.0 §9.2](https://www.w3.org/TR/xslt-30/#temporary-trees).
    pub(crate) base_uri: Option<Arc<str>>,
}

/// One executable instruction.
#[derive(Debug)]
#[allow(
    clippy::enum_variant_names,
    reason = "the names are the XSLT element names"
)]
pub(crate) enum Instruction {
    /// A literal result element.
    Literal(Box<LiteralElement>),
    /// Literal character data, or `xsl:text`.
    Text {
        text: String,
        disable_escaping: bool,
    },
    /// A *text value template* — a text node compiled under `expand-text="yes"`, whose
    /// `{…}` parts are expressions ([XSLT 3.0 §5.7](https://www.w3.org/TR/xslt-30/#text-value-templates)).
    TextTemplate {
        template: Avt,
        disable_escaping: bool,
    },
    /// `xsl:value-of`.
    ValueOf {
        select: Option<CompiledXPath>,
        content: Vec<Instruction>,
        separator: Option<Avt>,
        disable_escaping: bool,
    },
    /// `xsl:sequence`.
    Sequence {
        select: Option<CompiledXPath>,
        content: Vec<Instruction>,
    },
    /// `xsl:copy-of`.
    CopyOf {
        select: CompiledXPath,
        copy_namespaces: bool,
    },
    /// `xsl:copy`.
    Copy {
        /// `select` (XSLT 3.0): the item to copy, in place of the context item.
        select: Option<CompiledXPath>,
        content: Vec<Instruction>,
        use_attribute_sets: Vec<ExpandedName>,
        copy_namespaces: bool,
        /// `inherit-namespaces` — [XSLT 3.0 §11.7.1](https://www.w3.org/TR/xslt-30/#copy-namespaces).
        inherit_namespaces: bool,
    },
    /// `xsl:apply-templates`.
    ApplyTemplates {
        select: Option<CompiledXPath>,
        mode: ModeRef,
        sort: Vec<SortKey>,
        params: Vec<WithParam>,
    },
    /// `xsl:apply-imports`.
    ApplyImports { params: Vec<WithParam> },
    /// `xsl:next-match`.
    NextMatch { params: Vec<WithParam> },
    /// `xsl:call-template`.
    CallTemplate {
        name: ExpandedName,
        params: Vec<WithParam>,
    },
    /// `xsl:for-each`.
    ForEach {
        select: CompiledXPath,
        sort: Vec<SortKey>,
        content: Vec<Instruction>,
    },
    /// `xsl:for-each-group`.
    ForEachGroup {
        select: CompiledXPath,
        grouping: Grouping,
        /// `composite="yes"`: the whole atomized key sequence is one grouping key.
        composite: bool,
        collation: Option<Avt>,
        sort: Vec<SortKey>,
        content: Vec<Instruction>,
    },
    /// `xsl:if`.
    If {
        test: CompiledXPath,
        content: Vec<Instruction>,
    },
    /// `xsl:choose`. The final branch has no test when an `xsl:otherwise` was written.
    Choose {
        branches: Vec<(Option<CompiledXPath>, Vec<Instruction>)>,
    },
    /// `xsl:variable` inside a sequence constructor.
    Variable(Box<VariableDecl>),
    /// `xsl:element`.
    Element {
        name: Avt,
        namespace: Option<Avt>,
        /// The namespace bindings in scope on the instruction, for resolving the computed name.
        prefixes: Arc<Vec<(String, String)>>,
        use_attribute_sets: Vec<ExpandedName>,
        /// `inherit-namespaces` — [XSLT 3.0 §11.7.1](https://www.w3.org/TR/xslt-30/#copy-namespaces).
        inherit_namespaces: bool,
        content: Vec<Instruction>,
    },
    /// `xsl:attribute`.
    Attribute {
        name: Avt,
        namespace: Option<Avt>,
        /// The namespace bindings in scope on the instruction.
        prefixes: Arc<Vec<(String, String)>>,
        select: Option<CompiledXPath>,
        separator: Option<Avt>,
        content: Vec<Instruction>,
    },
    /// `xsl:namespace`.
    Namespace {
        name: Avt,
        select: Option<CompiledXPath>,
        content: Vec<Instruction>,
    },
    /// `xsl:comment`.
    Comment {
        select: Option<CompiledXPath>,
        content: Vec<Instruction>,
    },
    /// `xsl:processing-instruction`.
    ProcessingInstruction {
        name: Avt,
        select: Option<CompiledXPath>,
        content: Vec<Instruction>,
    },
    /// `xsl:document`.
    Document { content: Vec<Instruction> },
    /// `xsl:number`.
    Number(Box<NumberSpec>),
    /// `xsl:message`.
    Message {
        select: Option<CompiledXPath>,
        content: Vec<Instruction>,
        terminate: Avt,
        error_code: Option<ErrorCode>,
    },
    /// `xsl:analyze-string`.
    AnalyzeString {
        select: CompiledXPath,
        regex: Avt,
        flags: Avt,
        matching: Option<Vec<Instruction>>,
        non_matching: Option<Vec<Instruction>>,
    },
    /// `xsl:perform-sort`.
    PerformSort {
        select: Option<CompiledXPath>,
        sort: Vec<SortKey>,
        content: Vec<Instruction>,
    },
    /// `xsl:result-document`.
    ResultDocument {
        href: Option<Avt>,
        output: Arc<OutputProperties>,
        deferred: Option<Box<DeferredOutput>>,
        content: Vec<Instruction>,
    },
    /// `xsl:iterate`.
    Iterate {
        select: CompiledXPath,
        params: Vec<ParamDecl>,
        content: Vec<Instruction>,
        on_completion: Vec<Instruction>,
    },
    /// `xsl:try` with its `xsl:catch` handlers.
    Try {
        select: Option<CompiledXPath>,
        content: Vec<Instruction>,
        catches: Vec<CatchClause>,
    },
    /// `xsl:assert`.
    Assert {
        test: CompiledXPath,
        select: Option<CompiledXPath>,
        content: Vec<Instruction>,
        error_code: Option<ErrorCode>,
    },
    /// `xsl:next-iteration`.
    NextIteration { params: Vec<WithParam> },
    /// `xsl:break`.
    Break {
        select: Option<CompiledXPath>,
        content: Vec<Instruction>,
    },
    /// `xsl:on-empty` and `xsl:on-non-empty`.
    OnEmpty {
        expect_empty: bool,
        select: Option<CompiledXPath>,
        content: Vec<Instruction>,
    },
    /// `xsl:where-populated`.
    WherePopulated {
        select: Option<CompiledXPath>,
        content: Vec<Instruction>,
    },
    /// `xsl:map`: the content is evaluated to a sequence of maps
    /// ([`Instruction::MapEntry`], or any other map-valued expression), which are combined
    /// into the single map this instruction evaluates to.
    Map { content: Vec<Instruction> },
    /// `xsl:map-entry`: a single-entry map, built from `key` and — using exactly the rules
    /// [XSLT 3.0 §9.3](https://www.w3.org/TR/xslt-30/#creating-values) gives a variable's value
    /// — `select` or `content`.
    MapEntry {
        key: CompiledXPath,
        select: Option<CompiledXPath>,
        content: Vec<Instruction>,
    },
    /// An element whose *effective version* is 1.0, which
    /// [XSLT 3.0 §3.9](https://www.w3.org/TR/xslt-30/#backwards) only a processor claiming the
    /// XSLT 1.0 compatibility feature may evaluate. Compiling it into an instruction, rather
    /// than reporting the fault while the stylesheet is read, is what makes `XTDE0160` the
    /// dynamic error the specification asks for: a 1.0 branch that is never taken raises
    /// nothing (`initial-template-080`), while one that is evaluated fails (`error-0160a`,
    /// `initial-template-081`, `version-031`).
    BackwardsCompatible { name: String },
    /// An instruction whose place in the stylesheet text is remembered, so that a failure
    /// raised while it runs can say where it was written.
    ///
    /// [XSLT 3.0 §8.3](https://www.w3.org/TR/xslt-30/#element-catch) gives an `xsl:catch` body
    /// `$err:line-number` and `$err:column-number`, and only an instruction that carries its
    /// own provenance can answer them. Wrapping the few instructions that raise from their own
    /// evaluation — rather than giving every variant a field — keeps the provenance out of the
    /// instructions that never need it, and the wrapper is built only when the stylesheet was
    /// parsed with [`oxixml_infoset::ParseOptions::record_source_positions`].
    Positioned {
        position: SourcePosition,
        inner: Box<Instruction>,
    },
    /// `xsl:fallback`, which runs only when its parent instruction is unknown.
    Fallback,
    /// An instruction this processor does not implement: its `xsl:fallback` children run, and
    /// it is an error when the element has no `xsl:fallback` child at all.
    ///
    /// The two conditions are not the same one. [XSLT 3.0 §3.9](https://www.w3.org/TR/xslt-30/#fallback)
    /// instantiates "the `xsl:fallback` children of the instruction", which an `<xsl:fallback/>`
    /// with no content satisfies by instantiating nothing — the stylesheet has said what to do
    /// instead, and the answer is "nothing" (`extension-functions-0106`).
    Unknown {
        name: ExpandedName,
        /// Whether the element carries at least one `xsl:fallback` child.
        has_fallback: bool,
        fallback: Vec<Instruction>,
    },
}

/// What `xsl:context-item/@use` asks of the caller —
/// [XSLT 3.0 §10.1.1](https://www.w3.org/TR/xslt-30/#context-item).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub(crate) enum ContextItemUse {
    /// `use="required"`: "there must be a context item".
    Required,
    /// `use="optional"`, the default: "there may or may not be a context item".
    #[default]
    Optional,
    /// `use="absent"`: "the contained sequence constructor, and any `xsl:param` elements, are
    /// evaluated with an absent focus".
    Absent,
}

/// A compiled `xsl:context-item` child of `xsl:template`.
#[derive(Debug, Default)]
pub(crate) struct ContextItemDecl {
    /// The `as` attribute. Absent is equivalent to `item()`, which every item matches, so it
    /// is stored as `None` rather than as a type that always succeeds.
    pub(crate) as_type: Option<Arc<RequiredType>>,
    pub(crate) usage: ContextItemUse,
}

/// A compiled `xsl:template`.
#[derive(Debug)]
pub(crate) struct Template {
    pub(crate) pattern: Option<Arc<Pattern>>,
    /// The `as` attribute, which the body's result is converted to (`XTTE0505`).
    pub(crate) as_type: Option<Arc<RequiredType>>,
    /// The index of the union alternative this rule was derived from.
    pub(crate) alternative: usize,
    pub(crate) priority: f64,
    pub(crate) import_precedence: usize,
    /// The position of the template within its module, which breaks remaining ties.
    pub(crate) declaration_order: usize,
    pub(crate) params: Arc<Vec<ParamDecl>>,
    /// The `xsl:context-item` child, when the template declares one.
    pub(crate) context_item: Option<Arc<ContextItemDecl>>,
    pub(crate) content: Arc<Vec<Instruction>>,
}

/// What the compiler publishes about one `xsl:function` before any expression is compiled.
///
/// A call must compile wherever it is written, so the name and arity are needed early; a
/// *reference* to the function — `local:f#2` — additionally has to be judged against the types
/// the declaration gives, because [XPath 3.1 §3.6.1](https://www.w3.org/TR/xpath-31/#id-sequencetype-syntax)
/// makes `instance of function(…) as …` compare parameter and result types and not only arity
/// (`higher-order-functions-032` … `-034`). The declared types travel here so that
/// [`oxixml_xpath::StaticContext::with_host_signatures`] can answer for them.
#[derive(Debug, Clone)]
pub(crate) struct FunctionSignature {
    pub(crate) name: ExpandedName,
    pub(crate) arity: usize,
    /// The declared parameter and result types, when this processor understands every one of
    /// them; `None` leaves the reference judged by arity alone, which is what an unrecognized
    /// type name (forwards-compatible processing) has to fall back on.
    pub(crate) declared: Option<(Vec<SequenceType>, SequenceType)>,
}

/// A compiled `xsl:function`.
#[derive(Debug)]
pub(crate) struct UserFunction {
    /// The `as` attribute, which the body's result is converted to (`XTTE0780`).
    pub(crate) as_type: Option<Arc<RequiredType>>,
    pub(crate) params: Vec<ParamDecl>,
    pub(crate) content: Vec<Instruction>,
    pub(crate) import_precedence: usize,
    /// `cache="yes"`: [XSLT 3.0 §10.3](https://www.w3.org/TR/xslt-30/#stylesheet-functions-caching)
    /// invites the processor to memoize the function's results. It is only a hint, so the engine
    /// honours it just for calls it can prove observe nothing but their arguments — a call whose
    /// arguments are all atomic values, memoized within a single transformation so that every
    /// context-dependent value the body might read is stable while the cache is alive.
    pub(crate) cache: bool,
}

impl UserFunction {
    /// The declared parameter and result types, in the shape a static context publishes them.
    ///
    /// `None` when any one of them is a sequence type this processor does not recognize; an
    /// absent `as` is `item()*` on both sides ([XSLT 3.0 §10.3](https://www.w3.org/TR/xslt-30/#stylesheet-functions)).
    pub(crate) fn declared_signature(&self) -> Option<(Vec<SequenceType>, SequenceType)> {
        let mut parameters = Vec::with_capacity(self.params.len());
        for parameter in &self.params {
            parameters.push(declared_or_any(parameter.as_type.as_deref())?);
        }
        Some((parameters, declared_or_any(self.as_type.as_deref())?))
    }
}

/// The sequence type an optional `as` attribute denotes: the parsed type, `item()*` when the
/// attribute is absent, and `None` when it named something unrecognized.
fn declared_or_any(declared: Option<&RequiredType>) -> Option<SequenceType> {
    match declared {
        None => Some(RequiredType::any_sequence()),
        Some(required) => required.sequence_type().cloned(),
    }
}

/// A compiled `xsl:key`.
#[derive(Debug)]
pub(crate) struct KeyDefinition {
    pub(crate) name: ExpandedName,
    pub(crate) pattern: Pattern,
    pub(crate) use_expression: Option<CompiledXPath>,
    pub(crate) content: Vec<Instruction>,
    /// The effective collation, which every declaration of one key name must share.
    pub(crate) collation: Option<String>,
    /// `composite="yes"`, which every declaration of one key name must share.
    pub(crate) composite: bool,
}

/// A compiled `xsl:attribute-set`.
#[derive(Debug)]
pub(crate) struct AttributeSet {
    pub(crate) uses: Vec<ExpandedName>,
    pub(crate) content: Vec<Instruction>,
    pub(crate) import_precedence: usize,
}

/// A global `xsl:variable` or `xsl:param`.
#[derive(Debug)]
pub(crate) struct GlobalVariable {
    pub(crate) name: ExpandedName,
    pub(crate) select: Option<CompiledXPath>,
    pub(crate) content: Vec<Instruction>,
    pub(crate) is_parameter: bool,
    /// `static="yes"`: the declaration is evaluated while the stylesheet is compiled, so a
    /// forward reference to it is the static error `XPST0008` rather than a circularity.
    pub(crate) is_static: bool,
    pub(crate) required: bool,
    pub(crate) as_type: Option<Arc<RequiredType>>,
    pub(crate) import_precedence: usize,
    /// The static base URI of the declaration — see [`VariableDecl::base_uri`].
    pub(crate) base_uri: Option<Arc<str>>,
}

impl GlobalVariable {
    /// Whether the declaration is an *implicitly mandatory* stylesheet parameter — the same
    /// [XSLT 3.0 §9.3](https://www.w3.org/TR/xslt-30/#parameters) rule [`ParamDecl`] applies,
    /// read for a top-level `xsl:param`.
    pub(crate) fn is_implicitly_mandatory(&self) -> bool {
        self.is_parameter
            && !self.required
            && self.select.is_none()
            && self.content.is_empty()
            && self
                .as_type
                .as_ref()
                .is_some_and(|as_type| !as_type.permits_empty())
    }
}

/// A whitespace-control declaration.
#[derive(Debug)]
pub(crate) struct SpaceControl {
    /// The name test, in the form `*`, `prefix:*`, `*:local` or a resolved name.
    pub(crate) test: SpaceTest,
    pub(crate) strip: bool,
    pub(crate) priority: f64,
    pub(crate) import_precedence: usize,
}

/// The name test of an `xsl:strip-space` or `xsl:preserve-space` token.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum SpaceTest {
    /// `*`.
    Any,
    /// `prefix:*`.
    AnyLocal(String),
    /// `*:local`.
    AnyNamespace(String),
    /// A resolved name.
    Name(ExpandedName),
}

/// A declared `xsl:mode`.
#[derive(Debug, Default)]
pub(crate) struct ModeDeclaration {
    /// `on-no-match="deep-skip"` and friends.
    pub(crate) on_no_match: NoMatchPolicy,
    /// `on-multiple-match="fail"`: an ambiguous match is `XTDE0540` rather than the default
    /// recovery, which keeps the last of the tied rules.
    pub(crate) on_multiple_match_fails: bool,
    /// `typed="yes"`: the nodes processed in this mode are expected to carry a type annotation,
    /// so processing an untyped element or attribute is `XTTE3100`.
    ///
    /// `typed="no"` states the converse expectation, which this processor cannot violate: it
    /// never annotates a node with anything but `xs:untyped`, so the flag is only recorded when
    /// the declaration asks for typed nodes.
    pub(crate) typed: bool,
    /// `visibility="private"`, which shuts the mode out of the entry points: an invocation that
    /// names it as the initial mode is `XTDE0045`
    /// ([XSLT 3.0 §6.6.1](https://www.w3.org/TR/xslt-30/#declaring-modes)).
    ///
    /// Only a declaration that *states* `private` sets this. A mode nothing declares, and one
    /// whose `xsl:mode` leaves the attribute out, stays reachable — that is what the whole
    /// `initial-mode` half of the W3C suite relies on, and shutting them out would make an
    /// entry point of a mode a stylesheet plainly means to offer unreachable.
    pub(crate) private: bool,
}

/// What the built-in template rule does when nothing matches.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub(crate) enum NoMatchPolicy {
    /// Recurse into children, copying text — the XSLT 1.0 built-in rules.
    #[default]
    DeepCopyText,
    /// Recurse into children but copy nothing.
    DeepSkip,
    /// Copy the node shallowly and recurse.
    ShallowCopy,
    /// Do nothing at all.
    ShallowSkip,
    /// Raise `XTDE0555`.
    Fail,
    /// Copy the node deeply.
    DeepCopy,
    /// Copy text nodes, but do not recurse into elements — the XSLT 3.0 default.
    TextOnlyCopy,
}

impl NoMatchPolicy {
    /// Parses the `on-no-match` attribute.
    pub(crate) fn parse(value: &str) -> Option<Self> {
        match value {
            "text-only-copy" => Some(Self::TextOnlyCopy),
            "shallow-copy" => Some(Self::ShallowCopy),
            "shallow-skip" => Some(Self::ShallowSkip),
            "deep-copy" => Some(Self::DeepCopy),
            "deep-skip" => Some(Self::DeepSkip),
            "fail" => Some(Self::Fail),
            _ => None,
        }
    }
}

/// One mode's template rules.
#[derive(Debug, Default)]
pub(crate) struct Mode {
    pub(crate) rules: Vec<usize>,
    pub(crate) declaration: ModeDeclaration,
}

/// An `error-code` attribute: the attribute value template that produces the lexical name, and
/// the namespace bindings in scope where it was written, which resolve its prefix.
#[derive(Debug)]
pub(crate) struct ErrorCode {
    pub(crate) template: Avt,
    pub(crate) namespaces: Vec<(String, String)>,
}

/// Everything a compiled stylesheet holds.
#[derive(Debug, Default)]
pub(crate) struct StylesheetIr {
    pub(crate) version: XsltVersion,
    pub(crate) templates: Vec<Template>,
    pub(crate) named_templates: HashMap<ExpandedName, usize>,
    pub(crate) modes: HashMap<ModeKey, Mode>,
    pub(crate) functions: HashMap<(ExpandedName, usize), UserFunction>,
    pub(crate) keys: Vec<KeyDefinition>,
    pub(crate) attribute_sets: HashMap<ExpandedName, Vec<AttributeSet>>,
    pub(crate) globals: Vec<GlobalVariable>,
    pub(crate) space_controls: Vec<SpaceControl>,
    pub(crate) output: OutputProperties,
    pub(crate) named_outputs: HashMap<ExpandedName, Arc<OutputProperties>>,
    pub(crate) decimal_formats: HashMap<Option<ExpandedName>, oxixml_fn::numeric::DecimalFormat>,
    pub(crate) namespace_aliases: Vec<(String, String, usize)>,
    pub(crate) strips_any_space: bool,
    pub(crate) default_collation: Option<String>,
    /// The `xsl:global-context-item` declaration, which constrains the item the global
    /// variables and parameters are evaluated with.
    pub(crate) global_context_item: Option<Arc<ContextItemDecl>>,
    /// Every namespace prefix any module of the stylesheet declares, which the availability
    /// functions need to tell an unbound prefix from a bound one at run time.
    pub(crate) declared_prefixes: std::collections::HashSet<String>,
    /// For each stylesheet level, indexed by its import precedence, the lowest import precedence
    /// in its own import subtree.
    ///
    /// Modules are numbered in post-order — a module is given its precedence only once every
    /// module it imports has one — so a module's import subtree is the contiguous band
    /// `[import_subtree_floors[p], p]`. That band, and not "every lower precedence", is what
    /// [XSLT 3.0 §6.8](https://www.w3.org/TR/xslt-30/#apply-imports) makes `xsl:apply-imports`
    /// search.
    pub(crate) import_subtree_floors: Vec<usize>,
}

/// The key a mode is looked up under.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub(crate) enum ModeKey {
    /// The unnamed mode.
    Default,
    /// A named mode.
    Named(ExpandedName),
}

impl StylesheetIr {
    /// The lowest import precedence in the import subtree of the stylesheet level numbered
    /// `precedence`.
    ///
    /// A stylesheet the compiler never recorded a floor for — which only an IR assembled by hand
    /// can be — behaves as though its subtree started at the very bottom, which is the reading
    /// `xsl:apply-imports` had before the import tree was recorded.
    pub(crate) fn import_subtree_floor(&self, precedence: usize) -> usize {
        self.import_subtree_floors
            .get(precedence)
            .copied()
            .unwrap_or(0)
    }

    /// The mode a reference selects, given the mode currently in force.
    pub(crate) fn resolve_mode(&self, reference: &ModeRef, current: &ModeKey) -> ModeKey {
        match reference {
            ModeRef::Default => ModeKey::Default,
            ModeRef::Current => current.clone(),
            ModeRef::Named(name) => ModeKey::Named(name.clone()),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mode_references_resolve() {
        let ir = StylesheetIr::default();
        let named = ModeKey::Named(ExpandedName::new(None::<&str>, "m"));
        assert_eq!(ir.resolve_mode(&ModeRef::Default, &named), ModeKey::Default);
        assert_eq!(ir.resolve_mode(&ModeRef::Current, &named), named);
        assert_eq!(
            ir.resolve_mode(
                &ModeRef::Named(ExpandedName::new(None::<&str>, "x")),
                &named
            ),
            ModeKey::Named(ExpandedName::new(None::<&str>, "x"))
        );
    }

    #[test]
    fn no_match_policies_parse() {
        assert_eq!(
            NoMatchPolicy::parse("shallow-copy"),
            Some(NoMatchPolicy::ShallowCopy)
        );
        assert_eq!(NoMatchPolicy::parse("nonsense"), None);
        assert_eq!(NoMatchPolicy::default(), NoMatchPolicy::DeepCopyText);
    }
}
