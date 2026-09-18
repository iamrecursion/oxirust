//! Turns a `RawSchema` into a [`Compiled`] schema: `is-a` pattern instantiation, `extends`
//! rule flattening, phase filtering and query-binding compilation all happen here, once, so that
//! the validator only ever walks already-compiled [`oxixml_xpath::CompiledXPath`]s.

use std::collections::{BTreeMap, HashMap};

use oxixml_xpath::{CompiledXPath, StaticContext, XPath, XPathVersion};

use crate::error::{SchematronError, SchematronResult};
use crate::model::{
    CheckDecl, CheckKind, LetDecl, MessagePart, PatternDecl, RawSchema, RuleDecl, RuleItem,
};
use crate::parse::{IncludeResolver, parse_schema};

/// The entry point: compiles a Schematron schema document.
///
/// # Examples
///
/// ```
/// use oxixml_schematron::Schematron;
///
/// let compiled = Schematron::compile(
///     r#"<schema xmlns="http://purl.oclc.org/dsdl/schematron">
///          <pattern id="p1">
///            <rule context="doc">
///              <assert test="item">a doc must have an item</assert>
///            </rule>
///          </pattern>
///        </schema>"#,
///     None,
/// )?;
/// let report = compiled.validate("<doc/>")?;
/// assert!(!report.is_valid());
/// # Ok::<(), oxixml_schematron::SchematronError>(())
/// ```
#[derive(Debug, Clone, Copy, Default)]
pub struct Schematron;

impl Schematron {
    /// Compiles `schema`, activating `phase` (or the schema's `defaultPhase`, or every pattern
    /// when neither is set — the same as naming the reserved phase `#ALL`).
    ///
    /// `sch:include` in `schema` is left unresolved; use [`Schematron::compile_with`] to supply
    /// an [`IncludeResolver`].
    ///
    /// # Errors
    ///
    /// [`SchematronError`] for any well-formedness, structural, reference-resolution or
    /// XPath-compilation failure. See [`SchematronError`]'s variants.
    pub fn compile(schema: &str, phase: Option<&str>) -> SchematronResult<Compiled> {
        Self::compile_with(schema, phase, None)
    }

    /// As [`Schematron::compile`], resolving `sch:include` (and `sch:extends/@href`) through
    /// `resolver`.
    ///
    /// # Errors
    ///
    /// As [`Schematron::compile`], plus [`SchematronError::UnresolvedInclude`] when the schema
    /// contains an `include` and `resolver` cannot answer it.
    pub fn compile_with(
        schema: &str,
        phase: Option<&str>,
        resolver: Option<&dyn IncludeResolver>,
    ) -> SchematronResult<Compiled> {
        let raw = parse_schema(schema, resolver)?;
        compile_raw(raw, phase)
    }
}

/// The [`oxixml_xpath::XPathVersion`] a schema's `queryBinding` selects.
///
/// The ISO default query binding, `"xslt"` (and its historical spellings `"xslt1"`), matches
/// the XPath 1.0 semantics XSLT 1.0 evaluates `select`/`test`/`context` under. `"xslt2"` and
/// `"xslt3"` select the corresponding XDM-sequence semantics of XPath 2.0 and XPath 3.0/3.1.
/// Executing an embedded XSLT sequence constructor — as opposed to compiling its query-binding
/// expressions as XPath — is out of scope for this crate; see the crate documentation.
#[must_use]
pub fn query_binding_version(binding: Option<&str>) -> XPathVersion {
    match binding.map(str::trim) {
        Some("xslt2") | Some("xpath2") | Some("xpath20") => XPathVersion::XPath20,
        Some("xslt3") | Some("xpath3") | Some("xpath30") => XPathVersion::XPath30,
        Some("xpath31") | Some("xpath3.1") => XPathVersion::XPath31,
        _ => XPathVersion::XPath10,
    }
}

/// A compiled `sch:let`, `sch:param`-substituted or not: a name paired with the compiled
/// expression its value comes from.
type CompiledLet = (String, CompiledXPath);

/// A registry of rule declarations by id, used to resolve `sch:extends`. Borrowed from the
/// original, pre-`is-a` [`RawSchema::patterns`] so that `extends` can reach rules inside
/// abstract patterns.
type RuleRegistry<'a> = HashMap<&'a str, &'a RuleDecl>;

/// One interpolated message part, compiled.
#[derive(Debug, Clone)]
pub(crate) enum CompiledMessagePart {
    /// Literal text.
    Text(String),
    /// `sch:value-of`.
    ValueOf(CompiledXPath),
    /// `sch:name`, with its optional `path`.
    Name(Option<CompiledXPath>),
}

/// A compiled `sch:diagnostic`.
#[derive(Debug, Clone)]
pub(crate) struct CompiledDiagnostic {
    pub(crate) message: Vec<CompiledMessagePart>,
}

/// A compiled `sch:assert` or `sch:report`, with `sch:extends`, role/flag inheritance and its
/// visible `let` bindings all already resolved.
#[derive(Debug, Clone)]
pub(crate) struct CompiledCheck {
    pub(crate) kind: CheckKind,
    pub(crate) id: Option<String>,
    pub(crate) role: Option<String>,
    pub(crate) flag: Option<String>,
    pub(crate) subject: Option<CompiledXPath>,
    pub(crate) test_source: String,
    pub(crate) test: CompiledXPath,
    pub(crate) diagnostics: Vec<String>,
    pub(crate) message: Vec<CompiledMessagePart>,
    /// Every `let` visible at this check's position: schema, phase, pattern and rule bindings,
    /// in the order they must be bound so that later ones shadow earlier ones of the same name.
    pub(crate) lets: Vec<CompiledLet>,
}

/// A compiled `sch:rule`: a concrete rule, with every `sch:extends` already spliced in.
#[derive(Debug, Clone)]
pub(crate) struct CompiledRule {
    pub(crate) id: Option<String>,
    pub(crate) role: Option<String>,
    pub(crate) flag: Option<String>,
    pub(crate) context_source: String,
    /// The rule's context, rewritten to a selecting expression evaluated from the document
    /// root. See [`context_selecting_expression`].
    pub(crate) context_expr: CompiledXPath,
    pub(crate) checks: Vec<CompiledCheck>,
}

/// A compiled `sch:pattern`, `is-a`-instantiated if it was abstract.
#[derive(Debug, Clone)]
pub(crate) struct CompiledPattern {
    pub(crate) id: Option<String>,
    pub(crate) title: Option<String>,
    pub(crate) rules: Vec<CompiledRule>,
}

/// A schema, compiled and ready to validate any number of documents.
///
/// Build one with [`Schematron::compile`]; validate a document with [`Compiled::validate`] or
/// [`Compiled::validate_tree`].
#[derive(Debug, Clone)]
pub struct Compiled {
    pub(crate) title: Option<String>,
    pub(crate) schema_version: Option<String>,
    pub(crate) phase: Option<String>,
    pub(crate) diagnostics: HashMap<String, CompiledDiagnostic>,
    /// The schema-level and phase-level `sch:let` declarations, in that order.
    ///
    /// [ISO/IEC 19757-3 §5.4.5](https://www.iso.org/standard/55982.html) makes these *global*
    /// XSLT variables, so — unlike a pattern's or a rule's `let` — they are evaluated once per
    /// validation run, with the document root as the context node, and every rule firing then
    /// sees the same value. They are therefore kept out of [`CompiledRule::checks`] and bound by
    /// [`Compiled::validate_tree`] instead.
    pub(crate) globals: Vec<CompiledLet>,
    pub(crate) patterns: Vec<CompiledPattern>,
}

impl Compiled {
    /// The schema's `sch:title`, if it had one.
    #[must_use]
    pub fn title(&self) -> Option<&str> {
        self.title.as_deref()
    }

    /// The `schemaVersion` attribute, if it had one.
    #[must_use]
    pub fn schema_version(&self) -> Option<&str> {
        self.schema_version.as_deref()
    }

    /// The phase this schema was compiled with — `None` for `#ALL`/every pattern.
    #[must_use]
    pub fn phase(&self) -> Option<&str> {
        self.phase.as_deref()
    }

    /// The ids of the patterns active in this compilation, in the order they will be validated.
    pub fn active_pattern_ids(&self) -> impl Iterator<Item = Option<&str>> {
        self.patterns.iter().map(|pattern| pattern.id.as_deref())
    }
}

fn compile_raw(raw: RawSchema, phase: Option<&str>) -> SchematronResult<Compiled> {
    let static_context = build_static_context(&raw);
    let version = query_binding_version(raw.query_binding.as_deref());

    let mut rule_registry: RuleRegistry<'_> = HashMap::new();
    let mut pattern_registry: HashMap<&str, &PatternDecl> = HashMap::new();
    for pattern in &raw.patterns {
        if let Some(id) = pattern.id.as_deref() {
            pattern_registry.insert(id, pattern);
        }
        for rule in &pattern.rules {
            if let Some(id) = rule.id.as_deref() {
                rule_registry.insert(id, rule);
            }
        }
    }

    let mut resolved_patterns: Vec<PatternDecl> = Vec::new();
    for pattern in &raw.patterns {
        if pattern.is_a.is_some() {
            resolved_patterns.push(instantiate_pattern(pattern, &pattern_registry)?);
        } else if !pattern.is_abstract {
            resolved_patterns.push(pattern.clone());
        }
    }

    let phase_name = phase.or(raw.default_phase.as_deref()).map(str::trim);
    let (active_ids, phase_lets_raw): (Option<Vec<String>>, Vec<LetDecl>) = match phase_name {
        None | Some("#ALL") | Some("") => (None, Vec::new()),
        Some(name) => {
            let decl = raw
                .phases
                .iter()
                .find(|candidate| candidate.id == name)
                .ok_or_else(|| SchematronError::UnknownReference {
                    kind: "phase",
                    id: name.to_owned(),
                })?;
            (Some(decl.active.clone()), decl.lets.clone())
        }
    };

    let active_patterns_raw: Vec<PatternDecl> = match &active_ids {
        None => resolved_patterns,
        Some(ids) => ids
            .iter()
            .map(|id| {
                resolved_patterns
                    .iter()
                    .find(|pattern| pattern.id.as_deref() == Some(id.as_str()))
                    .cloned()
                    .ok_or_else(|| SchematronError::UnknownReference {
                        kind: "pattern",
                        id: id.clone(),
                    })
            })
            .collect::<SchematronResult<Vec<_>>>()?,
    };

    // Schema- and phase-level `let`s are global variables (§5.4.5): they are *not* folded into
    // the per-pattern/per-rule bindings, because those are evaluated with the firing rule's
    // context node in focus and once per firing.
    let mut globals = compile_lets(&raw.lets, &static_context, version)?;
    globals.extend(compile_lets(&phase_lets_raw, &static_context, version)?);

    let mut diagnostics = HashMap::new();
    for diagnostic in &raw.diagnostics {
        diagnostics.insert(
            diagnostic.id.clone(),
            CompiledDiagnostic {
                message: compile_message(&diagnostic.message, &static_context, version)?,
            },
        );
    }

    let mut patterns = Vec::new();
    for pattern in &active_patterns_raw {
        patterns.push(compile_pattern(
            pattern,
            &rule_registry,
            &static_context,
            version,
        )?);
    }

    Ok(Compiled {
        title: raw.title,
        schema_version: raw.schema_version,
        phase: phase_name.filter(|name| *name != "#ALL").map(str::to_owned),
        diagnostics,
        globals,
        patterns,
    })
}

fn build_static_context(raw: &RawSchema) -> StaticContext {
    let mut context = StaticContext::new().with_undeclared_variables();
    for binding in &raw.namespaces {
        context = context.with_namespace(binding.prefix.clone(), binding.uri.clone());
    }
    context
}

fn compile_lets(
    lets: &[LetDecl],
    context: &StaticContext,
    version: XPathVersion,
) -> SchematronResult<Vec<CompiledLet>> {
    lets.iter()
        .map(|decl| {
            let expr = compile_expression(&decl.value, context, version)?;
            Ok((decl.name.clone(), expr))
        })
        .collect()
}

/// Compiles `source`, wrapping any failure with the source text for a useful error message.
fn compile_expression(
    source: &str,
    context: &StaticContext,
    version: XPathVersion,
) -> SchematronResult<CompiledXPath> {
    XPath::compile(source, version, context.clone()).map_err(|error| SchematronError::XPath {
        expression: source.to_owned(),
        source: Box::new(error),
    })
}

fn compile_message(
    parts: &[MessagePart],
    context: &StaticContext,
    version: XPathVersion,
) -> SchematronResult<Vec<CompiledMessagePart>> {
    parts
        .iter()
        .map(|part| {
            Ok(match part {
                MessagePart::Text(text) => CompiledMessagePart::Text(text.clone()),
                MessagePart::ValueOf(select) => {
                    CompiledMessagePart::ValueOf(compile_expression(select, context, version)?)
                }
                MessagePart::Name(path) => CompiledMessagePart::Name(
                    path.as_deref()
                        .map(|path| compile_expression(path, context, version))
                        .transpose()?,
                ),
            })
        })
        .collect()
}

fn compile_pattern(
    pattern: &PatternDecl,
    registry: &RuleRegistry<'_>,
    context: &StaticContext,
    version: XPathVersion,
) -> SchematronResult<CompiledPattern> {
    let mut pattern_lets: Vec<CompiledLet> = Vec::new();
    for decl in &pattern.lets {
        pattern_lets.push((
            decl.name.clone(),
            compile_expression(&decl.value, context, version)?,
        ));
    }

    let mut rules = Vec::new();
    for rule in &pattern.rules {
        if rule.is_abstract {
            continue;
        }
        let context_source =
            rule.context
                .clone()
                .ok_or_else(|| SchematronError::MissingAttribute {
                    name: "context",
                    context: format!("rule id={:?}", rule.id.as_deref().unwrap_or("?")),
                })?;
        let context_expr = compile_expression(
            &context_selecting_expression(&context_source),
            context,
            version,
        )?;

        let mut running = pattern_lets.clone();
        let mut checks = Vec::new();
        let mut chain = Vec::new();
        flatten_rule_body(
            &rule.body,
            rule.role.as_deref(),
            rule.flag.as_deref(),
            registry,
            context,
            version,
            &mut running,
            &mut checks,
            &mut chain,
            0,
        )?;

        rules.push(CompiledRule {
            id: rule.id.clone(),
            role: rule.role.clone(),
            flag: rule.flag.clone(),
            context_source,
            context_expr,
            checks,
        });
    }

    Ok(CompiledPattern {
        id: pattern.id.clone(),
        title: pattern.title.clone(),
        rules,
    })
}

/// The greatest depth of `extends`/inline-`extends` nesting [`flatten_rule_body`] will follow
/// before reporting a clean error instead of recursing without bound.
///
/// `chain` detects a rule that `extends` back into itself (directly or through
/// intermediaries) and rejects it as [`SchematronError::ExtendsCycle`], but that alone does
/// nothing to bound a chain of `extends` references that only ever move on to a *new* rule id —
/// a schema with `r0` extending `r1` extending `r2` … thousands of rules deep, none of them
/// repeated, would otherwise recurse the native call stack without bound. This mirrors the
/// depth guard [`crate::parse`] already places on nested `sch:include`, for the same reason.
const MAX_EXTENDS_DEPTH: usize = 256;

#[allow(clippy::too_many_arguments)]
fn flatten_rule_body(
    body: &[RuleItem],
    default_role: Option<&str>,
    default_flag: Option<&str>,
    registry: &RuleRegistry<'_>,
    context: &StaticContext,
    version: XPathVersion,
    running: &mut Vec<CompiledLet>,
    checks_out: &mut Vec<CompiledCheck>,
    chain: &mut Vec<String>,
    depth: usize,
) -> SchematronResult<()> {
    if depth >= MAX_EXTENDS_DEPTH {
        return Err(SchematronError::ExtendsDepth {
            limit: MAX_EXTENDS_DEPTH,
        });
    }
    for item in body {
        match item {
            RuleItem::Let(decl) => {
                running.push((
                    decl.name.clone(),
                    compile_expression(&decl.value, context, version)?,
                ));
            }
            RuleItem::Check(decl) => {
                checks_out.push(compile_check(
                    decl,
                    default_role,
                    default_flag,
                    running.clone(),
                    context,
                    version,
                )?);
            }
            RuleItem::Extends(id) => {
                if chain.iter().any(|open| open == id) {
                    return Err(SchematronError::ExtendsCycle { id: id.clone() });
                }
                let target: &RuleDecl = *registry.get(id.as_str()).ok_or_else(|| {
                    SchematronError::UnknownReference {
                        kind: "rule",
                        id: id.clone(),
                    }
                })?;
                chain.push(id.clone());
                flatten_rule_body(
                    &target.body,
                    target.role.as_deref(),
                    target.flag.as_deref(),
                    registry,
                    context,
                    version,
                    running,
                    checks_out,
                    chain,
                    depth + 1,
                )?;
                chain.pop();
            }
            RuleItem::InlineExtends(inline) => {
                flatten_rule_body(
                    &inline.body,
                    inline.role.as_deref(),
                    inline.flag.as_deref(),
                    registry,
                    context,
                    version,
                    running,
                    checks_out,
                    chain,
                    depth + 1,
                )?;
            }
        }
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn compile_check(
    decl: &CheckDecl,
    default_role: Option<&str>,
    default_flag: Option<&str>,
    lets: Vec<CompiledLet>,
    context: &StaticContext,
    version: XPathVersion,
) -> SchematronResult<CompiledCheck> {
    Ok(CompiledCheck {
        kind: decl.kind,
        id: decl.id.clone(),
        role: decl
            .role
            .clone()
            .or_else(|| default_role.map(str::to_owned)),
        flag: decl
            .flag
            .clone()
            .or_else(|| default_flag.map(str::to_owned)),
        subject: decl
            .subject
            .as_deref()
            .map(|s| compile_expression(s, context, version))
            .transpose()?,
        test_source: decl.test.clone(),
        test: compile_expression(&decl.test, context, version)?,
        diagnostics: decl.diagnostics.clone(),
        message: compile_message(&decl.message, context, version)?,
        lets,
    })
}

/// Rewrites an XSLT-pattern-style `context` into a path expression selecting every matching
/// node from the document root: each top-level `|`-separated alternative is prefixed with `//`
/// unless it already starts with `/` or `.`.
///
/// This is the same practical translation essentially every non-XSLT Schematron engine uses in
/// place of a full XSLT match-pattern implementation — see the crate documentation for why an
/// engine built only on [`oxixml_xpath`] cannot do better without also being an XSLT processor.
#[must_use]
pub fn context_selecting_expression(context: &str) -> String {
    split_top_level_union(context)
        .into_iter()
        .map(|alternative| {
            let alternative = alternative.trim();
            if alternative.is_empty()
                || alternative.starts_with('/')
                || alternative.starts_with('.')
            {
                alternative.to_owned()
            } else {
                format!("//{alternative}")
            }
        })
        .collect::<Vec<_>>()
        .join(" | ")
}

/// Splits `pattern` on every top-level `|` — one not nested inside `[...]`, `(...)` or a string
/// literal.
fn split_top_level_union(pattern: &str) -> Vec<&str> {
    let mut parts = Vec::new();
    let mut depth = 0i32;
    let mut quote: Option<char> = None;
    let mut start = 0usize;
    for (index, ch) in pattern.char_indices() {
        if let Some(open) = quote {
            if ch == open {
                quote = None;
            }
            continue;
        }
        match ch {
            '\'' | '"' => quote = Some(ch),
            '[' | '(' => depth += 1,
            ']' | ')' => depth -= 1,
            '|' if depth == 0 => {
                parts.push(&pattern[start..index]);
                start = index + ch.len_utf8();
            }
            _ => {}
        }
    }
    parts.push(&pattern[start..]);
    parts
}

fn instantiate_pattern(
    instantiating: &PatternDecl,
    patterns: &HashMap<&str, &PatternDecl>,
) -> SchematronResult<PatternDecl> {
    let base_id = instantiating
        .is_a
        .as_deref()
        .expect("caller only calls this when `is_a` is `Some`");
    let base = *patterns
        .get(base_id)
        .ok_or_else(|| SchematronError::UnknownReference {
            kind: "pattern",
            id: base_id.to_owned(),
        })?;
    let params: BTreeMap<String, String> = instantiating
        .params
        .iter()
        .map(|param| (param.name.clone(), param.value.clone()))
        .collect();

    Ok(PatternDecl {
        id: instantiating.id.clone(),
        is_abstract: false,
        is_a: None,
        title: instantiating.title.clone().or_else(|| base.title.clone()),
        lets: base
            .lets
            .iter()
            .map(|decl| substitute_let(decl, &params))
            .collect(),
        rules: base
            .rules
            .iter()
            .map(|rule| substitute_rule(rule, &params))
            .collect(),
        params: Vec::new(),
    })
}

fn substitute_let(decl: &LetDecl, params: &BTreeMap<String, String>) -> LetDecl {
    LetDecl {
        name: decl.name.clone(),
        value: substitute(&decl.value, params),
    }
}

fn substitute_rule(rule: &RuleDecl, params: &BTreeMap<String, String>) -> RuleDecl {
    RuleDecl {
        id: rule.id.clone(),
        context: rule
            .context
            .as_deref()
            .map(|context| substitute(context, params)),
        is_abstract: rule.is_abstract,
        role: rule.role.clone(),
        flag: rule.flag.clone(),
        body: rule
            .body
            .iter()
            .map(|item| substitute_rule_item(item, params))
            .collect(),
    }
}

fn substitute_rule_item(item: &RuleItem, params: &BTreeMap<String, String>) -> RuleItem {
    match item {
        RuleItem::Let(decl) => RuleItem::Let(substitute_let(decl, params)),
        RuleItem::Check(decl) => RuleItem::Check(substitute_check(decl, params)),
        RuleItem::Extends(id) => RuleItem::Extends(id.clone()),
        RuleItem::InlineExtends(rule) => {
            RuleItem::InlineExtends(Box::new(substitute_rule(rule, params)))
        }
    }
}

fn substitute_check(decl: &CheckDecl, params: &BTreeMap<String, String>) -> CheckDecl {
    CheckDecl {
        kind: decl.kind,
        test: substitute(&decl.test, params),
        id: decl.id.clone(),
        role: decl.role.clone(),
        flag: decl.flag.clone(),
        subject: decl
            .subject
            .as_deref()
            .map(|subject| substitute(subject, params)),
        diagnostics: decl.diagnostics.clone(),
        message: decl
            .message
            .iter()
            .map(|part| substitute_message_part(part, params))
            .collect(),
    }
}

fn substitute_message_part(part: &MessagePart, params: &BTreeMap<String, String>) -> MessagePart {
    match part {
        MessagePart::Text(text) => MessagePart::Text(text.clone()),
        MessagePart::ValueOf(select) => MessagePart::ValueOf(substitute(select, params)),
        MessagePart::Name(path) => {
            MessagePart::Name(path.as_deref().map(|path| substitute(path, params)))
        }
    }
}

/// Replaces every `$name` token in `text` whose name is a key of `params` with its literal
/// value — the textual macro-substitution [ISO/IEC 19757-3 §5.4](https://www.iso.org/standard/55982.html)
/// abstract patterns use, distinct from (and evaluated before) XPath variable binding.
fn substitute(text: &str, params: &BTreeMap<String, String>) -> String {
    if params.is_empty() || !text.contains('$') {
        return text.to_owned();
    }
    let mut out = String::with_capacity(text.len());
    let mut chars = text.char_indices().peekable();
    while let Some((index, ch)) = chars.next() {
        if ch != '$' {
            out.push(ch);
            continue;
        }
        let name_start = index + 1;
        let mut name_end = name_start;
        while let Some(&(next_index, next_char)) = chars.peek() {
            if next_char.is_alphanumeric()
                || next_char == '_'
                || next_char == '-'
                || next_char == '.'
            {
                name_end = next_index + next_char.len_utf8();
                chars.next();
            } else {
                break;
            }
        }
        let name = &text[name_start..name_end];
        match params.get(name) {
            Some(value) => out.push_str(value),
            None => {
                out.push('$');
                out.push_str(name);
            }
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn query_binding_selects_the_right_version() {
        assert_eq!(query_binding_version(None), XPathVersion::XPath10);
        assert_eq!(query_binding_version(Some("xslt")), XPathVersion::XPath10);
        assert_eq!(query_binding_version(Some("xslt2")), XPathVersion::XPath20);
        assert_eq!(query_binding_version(Some("xslt3")), XPathVersion::XPath30);
    }

    #[test]
    fn context_becomes_a_selecting_union() {
        assert_eq!(context_selecting_expression("book"), "//book");
        assert_eq!(context_selecting_expression("book | cd"), "//book | //cd");
        assert_eq!(context_selecting_expression("/root"), "/root");
        assert_eq!(context_selecting_expression("a[b='|']"), "//a[b='|']");
    }

    #[test]
    fn param_substitution_respects_word_boundaries() {
        let mut params = BTreeMap::new();
        params.insert("name".to_owned(), "title".to_owned());
        assert_eq!(substitute("book/$name", &params), "book/title");
        assert_eq!(substitute("$namespace", &params), "$namespace");
        assert_eq!(substitute("no dollars here", &params), "no dollars here");
    }
}
