//! JSON-LD context generation and document building for TensorLogic knowledge graph serialization.
//!
//! This module provides rich JSON-LD support for publishing TensorLogic schemas and predicate
//! definitions as Linked Data. It enables generating JSON-LD `@context` blocks, typed `@id`/`@type`
//! mappings, and complete JSON-LD documents suitable for semantic-web publication.
//!
//! # Overview
//!
//! The core workflow is:
//! 1. Build a [`TlJsonLdContext`] with prefixes, base IRI, and per-term type coercions.
//! 2. Construct [`TlJsonLdNode`] instances for each entity (or generate them from
//!    [`tensorlogic_ir::TLExpr`] via [`expr_to_json_ld_node`]).
//! 3. Wrap everything in a [`TlJsonLdDocument`] and serialize with
//!    [`TlJsonLdDocument::to_json_string`] or [`TlJsonLdDocument::to_pretty_string`].
//!
//! # Example
//!
//! ```rust
//! use tensorlogic_oxirs_bridge::json_ld::{
//!     TlJsonLdContext, TlJsonLdNode, TlJsonLdDocument, JsonLdValue,
//!     ContextTerm,
//! };
//!
//! let mut ctx = TlJsonLdContext::new()
//!     .with_base("http://example.org/")
//!     .with_vocab("http://example.org/vocab#");
//! ctx.add_prefix("ex", "http://example.org/");
//!
//! let term = ContextTerm::new("knows", "http://example.org/knows")
//!     .with_type("@id");
//! ctx.add_term(term);
//!
//! let mut node = TlJsonLdNode::new()
//!     .with_id("http://example.org/alice")
//!     .with_type("http://example.org/Person");
//! node.add_property("ex:knows", JsonLdValue::id("http://example.org/bob"));
//!
//! let mut doc = TlJsonLdDocument::new(ctx);
//! doc.add_node(node);
//!
//! let json = doc.to_json_string();
//! assert!(json.contains("@context"));
//! ```

use std::collections::HashSet;
use tensorlogic_ir::TLExpr;

// ── Error ─────────────────────────────────────────────────────────────────────

/// Errors that can arise during JSON-LD generation or validation.
#[derive(Debug)]
pub enum JsonLdError {
    /// An IRI is syntactically invalid (e.g., relative without base).
    InvalidIri(String),
    /// A compact IRI uses an undefined prefix.
    UndefinedPrefix(String),
    /// Two nodes share the same `@id` value.
    DuplicateId(String),
    /// The document contains no nodes and cannot be serialized meaningfully.
    EmptyDocument,
    /// A generic serialization failure.
    SerializationError(String),
}

impl std::fmt::Display for JsonLdError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            JsonLdError::InvalidIri(iri) => write!(f, "Invalid IRI: {iri}"),
            JsonLdError::UndefinedPrefix(prefix) => write!(f, "Undefined prefix: {prefix}"),
            JsonLdError::DuplicateId(id) => write!(f, "Duplicate @id in document: {id}"),
            JsonLdError::EmptyDocument => write!(f, "JSON-LD document is empty"),
            JsonLdError::SerializationError(msg) => write!(f, "Serialization error: {msg}"),
        }
    }
}

impl std::error::Error for JsonLdError {}

// ── JsonLdValue ───────────────────────────────────────────────────────────────

/// A JSON-LD value: scalars, objects and arrays, built with pure-Rust string rendering.
///
/// Ordering of keys in `Object` variants is preserved (insertion order), which
/// matters for `@id`/`@type` appearing first in output.
#[derive(Debug, Clone)]
pub enum JsonLdValue {
    /// JSON null literal.
    Null,
    /// JSON boolean.
    Bool(bool),
    /// JSON number (stored as `f64`).
    Number(f64),
    /// JSON string.
    Str(String),
    /// JSON array.
    Array(Vec<JsonLdValue>),
    /// JSON object — ordered key-value pairs.
    Object(Vec<(String, JsonLdValue)>),
}

impl JsonLdValue {
    /// Construct a `Str` variant.
    pub fn string(s: impl Into<String>) -> Self {
        JsonLdValue::Str(s.into())
    }

    /// Construct a `Number` variant.
    pub fn number(n: f64) -> Self {
        JsonLdValue::Number(n)
    }

    /// Construct an `{"@id": iri}` object.
    pub fn id(iri: impl Into<String>) -> Self {
        JsonLdValue::Object(vec![("@id".to_string(), JsonLdValue::Str(iri.into()))])
    }

    /// Construct a `{"@value": value, "@type": datatype}` typed-literal object.
    pub fn typed_value(value: impl Into<String>, datatype: impl Into<String>) -> Self {
        JsonLdValue::Object(vec![
            ("@value".to_string(), JsonLdValue::Str(value.into())),
            ("@type".to_string(), JsonLdValue::Str(datatype.into())),
        ])
    }

    /// Render to compact (single-line) JSON.
    pub fn to_json_string(&self) -> String {
        self.render(0, false)
    }

    /// Render to pretty-printed JSON with the given base indentation level.
    pub fn to_pretty_string(&self, indent: usize) -> String {
        self.render(indent, true)
    }

    fn render(&self, depth: usize, pretty: bool) -> String {
        match self {
            JsonLdValue::Null => "null".to_string(),
            JsonLdValue::Bool(b) => if *b { "true" } else { "false" }.to_string(),
            JsonLdValue::Number(n) => {
                // Emit integer form when there is no fractional component.
                if n.fract() == 0.0 && n.is_finite() {
                    format!("{}", *n as i64)
                } else {
                    format!("{n}")
                }
            }
            JsonLdValue::Str(s) => format!("\"{}\"", json_escape(s)),
            JsonLdValue::Array(items) => {
                if items.is_empty() {
                    return "[]".to_string();
                }
                if !pretty {
                    let inner: Vec<String> = items.iter().map(|v| v.render(depth, false)).collect();
                    return format!("[{}]", inner.join(","));
                }
                let indent_str = "  ".repeat(depth + 1);
                let closing_str = "  ".repeat(depth);
                let inner: Vec<String> = items
                    .iter()
                    .map(|v| format!("{}{}", indent_str, v.render(depth + 1, true)))
                    .collect();
                format!("[\n{}\n{}]", inner.join(",\n"), closing_str)
            }
            JsonLdValue::Object(pairs) => {
                if pairs.is_empty() {
                    return "{}".to_string();
                }
                if !pretty {
                    let inner: Vec<String> = pairs
                        .iter()
                        .map(|(k, v)| format!("\"{}\":{}", json_escape(k), v.render(depth, false)))
                        .collect();
                    return format!("{{{}}}", inner.join(","));
                }
                let indent_str = "  ".repeat(depth + 1);
                let closing_str = "  ".repeat(depth);
                let inner: Vec<String> = pairs
                    .iter()
                    .map(|(k, v)| {
                        format!(
                            "{}\"{}\": {}",
                            indent_str,
                            json_escape(k),
                            v.render(depth + 1, true)
                        )
                    })
                    .collect();
                format!("{{\n{}\n{}}}", inner.join(",\n"), closing_str)
            }
        }
    }
}

// ── ContextTerm ───────────────────────────────────────────────────────────────

/// A single term definition inside a JSON-LD `@context` block.
///
/// Each `ContextTerm` maps a short name to a full IRI, optionally specifying
/// how JSON values should be coerced (`@type`) and how arrays should be treated
/// (`@container`).
#[derive(Debug, Clone)]
pub struct ContextTerm {
    /// The short name used in JSON-LD documents (e.g., `"knows"`).
    pub name: String,
    /// The expanded IRI this term maps to.
    pub iri: String,
    /// Optional `@type` coercion: `"@id"`, `"@vocab"`, `"xsd:string"`, etc.
    pub term_type: Option<String>,
    /// Optional `@container` annotation: `"@set"`, `"@list"`, `"@language"`, etc.
    pub container: Option<String>,
    /// If `true` the property is declared as an `@reverse` mapping.
    pub reverse: bool,
}

impl ContextTerm {
    /// Create a new term with a name and its expanded IRI.
    pub fn new(name: impl Into<String>, iri: impl Into<String>) -> Self {
        ContextTerm {
            name: name.into(),
            iri: iri.into(),
            term_type: None,
            container: None,
            reverse: false,
        }
    }

    /// Attach a `@type` coercion annotation (builder-style).
    pub fn with_type(mut self, ty: impl Into<String>) -> Self {
        self.term_type = Some(ty.into());
        self
    }

    /// Attach a `@container` annotation (builder-style).
    pub fn with_container(mut self, container: impl Into<String>) -> Self {
        self.container = Some(container.into());
        self
    }

    /// Mark this term as an `@reverse` property (builder-style).
    pub fn as_reverse(mut self) -> Self {
        self.reverse = true;
        self
    }

    /// Render the term definition as a [`JsonLdValue`].
    ///
    /// If the term only maps a name to an IRI (no extra annotations), the value is
    /// a plain string. Otherwise it is an object with `@id`, `@type`, etc.
    pub fn to_json_value(&self) -> JsonLdValue {
        let needs_object = self.term_type.is_some() || self.container.is_some() || self.reverse;
        if !needs_object {
            return JsonLdValue::Str(self.iri.clone());
        }

        let mut pairs: Vec<(String, JsonLdValue)> = Vec::new();
        if self.reverse {
            pairs.push(("@reverse".to_string(), JsonLdValue::Str(self.iri.clone())));
        } else {
            pairs.push(("@id".to_string(), JsonLdValue::Str(self.iri.clone())));
        }
        if let Some(ty) = &self.term_type {
            pairs.push(("@type".to_string(), JsonLdValue::Str(ty.clone())));
        }
        if let Some(container) = &self.container {
            pairs.push((
                "@container".to_string(),
                JsonLdValue::Str(container.clone()),
            ));
        }
        JsonLdValue::Object(pairs)
    }
}

// ── JsonLdContext ─────────────────────────────────────────────────────────────

/// A JSON-LD `@context` block: base IRI, vocab, prefix mappings, and term definitions.
///
/// This is distinct from [`crate::schema::jsonld::JsonLdContext`] which is an
/// RDF-schema-level utility used by [`crate::SchemaAnalyzer`].  This struct lives
/// in the top-level `json_ld` module and is intended for programmatic construction
/// of richly annotated contexts.
#[derive(Debug, Clone, Default)]
pub struct TlJsonLdContext {
    /// The `@base` IRI (relative IRI resolution root).
    pub base: Option<String>,
    /// The `@vocab` IRI (default vocabulary expansion).
    pub vocab: Option<String>,
    /// Named term definitions with type coercions and container annotations.
    pub terms: Vec<ContextTerm>,
    /// `(prefix, IRI)` pairs for compact IRI expansion (e.g., `"rdf"` → full IRI).
    pub prefixes: Vec<(String, String)>,
}

impl TlJsonLdContext {
    /// Create an empty context.
    pub fn new() -> Self {
        TlJsonLdContext::default()
    }

    /// Set `@base` IRI (builder-style).
    pub fn with_base(mut self, base: impl Into<String>) -> Self {
        self.base = Some(base.into());
        self
    }

    /// Set `@vocab` IRI (builder-style).
    pub fn with_vocab(mut self, vocab: impl Into<String>) -> Self {
        self.vocab = Some(vocab.into());
        self
    }

    /// Register a prefix–IRI mapping (e.g., `"rdf"` → `"http://www.w3.org/1999/…#"`).
    pub fn add_prefix(&mut self, prefix: impl Into<String>, iri: impl Into<String>) {
        self.prefixes.push((prefix.into(), iri.into()));
    }

    /// Add a term definition to this context.
    pub fn add_term(&mut self, term: ContextTerm) {
        self.terms.push(term);
    }

    /// Expand a compact IRI like `"rdf:type"` using registered prefixes.
    ///
    /// If the string already looks like an absolute IRI (contains `"://"`) it is
    /// returned unchanged.  If no matching prefix is found the input is returned
    /// as-is.
    pub fn expand_iri(&self, compact: &str) -> String {
        // Already absolute.
        if compact.contains("://") {
            return compact.to_string();
        }
        if let Some(colon_pos) = compact.find(':') {
            let prefix = &compact[..colon_pos];
            let local = &compact[colon_pos + 1..];
            for (p, ns) in &self.prefixes {
                if p == prefix {
                    return format!("{}{}", ns, local);
                }
            }
        }
        compact.to_string()
    }

    /// Total number of named term definitions (excludes raw prefix mappings).
    pub fn term_count(&self) -> usize {
        self.terms.len()
    }

    /// Render the entire `@context` as a JSON string (the object value, not the
    /// surrounding document).
    pub fn to_json_string(&self) -> String {
        self.to_json_value().to_json_string()
    }

    /// Build the context as a [`JsonLdValue`] object.
    pub(crate) fn to_json_value(&self) -> JsonLdValue {
        let mut pairs: Vec<(String, JsonLdValue)> = Vec::new();

        if let Some(base) = &self.base {
            pairs.push(("@base".to_string(), JsonLdValue::Str(base.clone())));
        }
        if let Some(vocab) = &self.vocab {
            pairs.push(("@vocab".to_string(), JsonLdValue::Str(vocab.clone())));
        }
        // Prefix shorthands
        for (prefix, iri) in &self.prefixes {
            pairs.push((prefix.clone(), JsonLdValue::Str(iri.clone())));
        }
        // Named term definitions
        for term in &self.terms {
            pairs.push((term.name.clone(), term.to_json_value()));
        }

        JsonLdValue::Object(pairs)
    }
}

// ── JsonLdNode ────────────────────────────────────────────────────────────────

/// A JSON-LD node: an entity with an optional `@id`, one or more `@type`s, and
/// an ordered list of property–value pairs.
#[derive(Debug, Clone)]
pub struct TlJsonLdNode {
    /// The node's IRI identifier (`@id`).
    pub id: Option<String>,
    /// The node's RDF types (`@type`).
    pub types: Vec<String>,
    /// Property–value pairs in insertion order.
    pub properties: Vec<(String, JsonLdValue)>,
}

impl Default for TlJsonLdNode {
    fn default() -> Self {
        TlJsonLdNode::new()
    }
}

impl TlJsonLdNode {
    /// Create an empty node.
    pub fn new() -> Self {
        TlJsonLdNode {
            id: None,
            types: Vec::new(),
            properties: Vec::new(),
        }
    }

    /// Attach an `@id` IRI (builder-style).
    pub fn with_id(mut self, id: impl Into<String>) -> Self {
        self.id = Some(id.into());
        self
    }

    /// Append an `@type` IRI (builder-style).
    pub fn with_type(mut self, ty: impl Into<String>) -> Self {
        self.types.push(ty.into());
        self
    }

    /// Append a property–value pair.
    pub fn add_property(&mut self, key: impl Into<String>, value: JsonLdValue) {
        self.properties.push((key.into(), value));
    }

    /// Render the node as a [`JsonLdValue`] object.
    pub fn to_json_value(&self) -> JsonLdValue {
        let mut pairs: Vec<(String, JsonLdValue)> = Vec::new();

        if let Some(id) = &self.id {
            pairs.push(("@id".to_string(), JsonLdValue::Str(id.clone())));
        }

        match self.types.len() {
            0 => {}
            1 => {
                pairs.push(("@type".to_string(), JsonLdValue::Str(self.types[0].clone())));
            }
            _ => {
                let type_arr = self
                    .types
                    .iter()
                    .map(|t| JsonLdValue::Str(t.clone()))
                    .collect();
                pairs.push(("@type".to_string(), JsonLdValue::Array(type_arr)));
            }
        }

        for (k, v) in &self.properties {
            pairs.push((k.clone(), v.clone()));
        }

        JsonLdValue::Object(pairs)
    }
}

// ── JsonLdDocument ────────────────────────────────────────────────────────────

/// A complete JSON-LD document: one shared `@context` and a graph of
/// [`TlJsonLdNode`] instances.
#[derive(Debug, Clone)]
pub struct TlJsonLdDocument {
    /// The shared context for all nodes in this document.
    pub context: TlJsonLdContext,
    /// All nodes in the `@graph`.
    pub nodes: Vec<TlJsonLdNode>,
}

impl TlJsonLdDocument {
    /// Create a new document with the given context.
    pub fn new(context: TlJsonLdContext) -> Self {
        TlJsonLdDocument {
            context,
            nodes: Vec::new(),
        }
    }

    /// Append a node to the document.
    pub fn add_node(&mut self, node: TlJsonLdNode) {
        self.nodes.push(node);
    }

    /// Number of nodes in the document.
    pub fn node_count(&self) -> usize {
        self.nodes.len()
    }

    /// Render the document as compact JSON.
    pub fn to_json_string(&self) -> String {
        self.build_document_value().to_json_string()
    }

    /// Render the document as pretty-printed JSON.
    pub fn to_pretty_string(&self) -> String {
        self.build_document_value().to_pretty_string(0)
    }

    fn build_document_value(&self) -> JsonLdValue {
        let ctx_value = self.context.to_json_value();

        let graph_items: Vec<JsonLdValue> = self.nodes.iter().map(|n| n.to_json_value()).collect();

        JsonLdValue::Object(vec![
            ("@context".to_string(), ctx_value),
            ("@graph".to_string(), JsonLdValue::Array(graph_items)),
        ])
    }
}

// ── Context generators ────────────────────────────────────────────────────────

/// Generate a [`TlJsonLdContext`] from a list of predicate names and their arities.
///
/// - Unary predicates (arity == 1) are given `@type: "@vocab"` (they denote a class
///   membership assertion).
/// - Binary predicates (arity == 2) are given `@type: "@id"` (they denote object
///   properties pointing to named resources).
/// - Higher-arity predicates get a plain IRI expansion with no coercion.
///
/// An optional `vocab_iri` is set as `@vocab` on the returned context.
pub fn context_from_predicates(
    predicates: &[(&str, usize)],
    base_iri: &str,
    vocab_iri: Option<&str>,
) -> Result<TlJsonLdContext, JsonLdError> {
    if base_iri.is_empty() {
        return Err(JsonLdError::InvalidIri(
            "base_iri must not be empty".to_string(),
        ));
    }

    let mut ctx = TlJsonLdContext::new().with_base(base_iri.to_string());
    if let Some(vocab) = vocab_iri {
        ctx = ctx.with_vocab(vocab.to_string());
    }

    for (name, arity) in predicates {
        let iri = format!("{}{}", base_iri, name);
        let term_type: Option<&str> = match arity {
            1 => Some("@vocab"),
            2 => Some("@id"),
            _ => None,
        };
        let mut term = ContextTerm::new(*name, iri);
        if let Some(ty) = term_type {
            term = term.with_type(ty);
        }
        ctx.add_term(term);
    }

    Ok(ctx)
}

/// Build a [`TlJsonLdContext`] pre-loaded with standard semantic-web prefixes.
///
/// Includes: `rdf`, `rdfs`, `owl`, `xsd`, `skos`, `schema` (schema.org) and
/// `dc` (Dublin Core elements).
pub fn standard_prefixes_context() -> TlJsonLdContext {
    let mut ctx = TlJsonLdContext::new();
    ctx.add_prefix("rdf", "http://www.w3.org/1999/02/22-rdf-syntax-ns#");
    ctx.add_prefix("rdfs", "http://www.w3.org/2000/01/rdf-schema#");
    ctx.add_prefix("owl", "http://www.w3.org/2002/07/owl#");
    ctx.add_prefix("xsd", "http://www.w3.org/2001/XMLSchema#");
    ctx.add_prefix("skos", "http://www.w3.org/2004/02/skos/core#");
    ctx.add_prefix("schema", "https://schema.org/");
    ctx.add_prefix("dc", "http://purl.org/dc/elements/1.1/");
    ctx
}

/// Generate a [`TlJsonLdContext`] by scanning all [`TLExpr::Pred`] nodes in `expr`
/// and creating a term for each unique predicate name found.
///
/// The resulting context uses `base_iri` as both `@base` and the IRI prefix for
/// each term.
pub fn context_from_expr(expr: &TLExpr, base_iri: &str) -> Result<TlJsonLdContext, JsonLdError> {
    if base_iri.is_empty() {
        return Err(JsonLdError::InvalidIri(
            "base_iri must not be empty".to_string(),
        ));
    }

    let mut seen: HashSet<String> = HashSet::new();
    let mut pred_arities: Vec<(String, usize)> = Vec::new();
    collect_preds(expr, &mut seen, &mut pred_arities);

    let pairs: Vec<(&str, usize)> = pred_arities
        .iter()
        .map(|(name, arity)| (name.as_str(), *arity))
        .collect();

    context_from_predicates(&pairs, base_iri, None)
}

/// Recursively collect predicate names and arities from a [`TLExpr`] tree.
fn collect_preds(expr: &TLExpr, seen: &mut HashSet<String>, out: &mut Vec<(String, usize)>) {
    match expr {
        TLExpr::Pred { name, args } if seen.insert(name.clone()) => {
            out.push((name.clone(), args.len()));
        }
        TLExpr::Pred { .. } => {}
        TLExpr::And(l, r)
        | TLExpr::Or(l, r)
        | TLExpr::Imply(l, r)
        | TLExpr::Add(l, r)
        | TLExpr::Sub(l, r)
        | TLExpr::Mul(l, r)
        | TLExpr::Div(l, r)
        | TLExpr::Pow(l, r)
        | TLExpr::Mod(l, r)
        | TLExpr::Min(l, r)
        | TLExpr::Max(l, r)
        | TLExpr::Eq(l, r)
        | TLExpr::Lt(l, r)
        | TLExpr::Gt(l, r)
        | TLExpr::Lte(l, r)
        | TLExpr::Gte(l, r) => {
            collect_preds(l, seen, out);
            collect_preds(r, seen, out);
        }
        TLExpr::Not(inner)
        | TLExpr::Score(inner)
        | TLExpr::Abs(inner)
        | TLExpr::Floor(inner)
        | TLExpr::Ceil(inner)
        | TLExpr::Round(inner)
        | TLExpr::Sqrt(inner)
        | TLExpr::Exp(inner)
        | TLExpr::Log(inner)
        | TLExpr::Sin(inner)
        | TLExpr::Cos(inner)
        | TLExpr::Tan(inner)
        | TLExpr::Box(inner)
        | TLExpr::Diamond(inner)
        | TLExpr::Next(inner)
        | TLExpr::Eventually(inner)
        | TLExpr::Always(inner) => {
            collect_preds(inner, seen, out);
        }
        TLExpr::Exists { body, .. }
        | TLExpr::ForAll { body, .. }
        | TLExpr::SoftExists { body, .. }
        | TLExpr::SoftForAll { body, .. }
        | TLExpr::Aggregate { body, .. }
        | TLExpr::WeightedRule { rule: body, .. } => {
            collect_preds(body, seen, out);
        }
        TLExpr::IfThenElse {
            condition,
            then_branch,
            else_branch,
        } => {
            collect_preds(condition, seen, out);
            collect_preds(then_branch, seen, out);
            collect_preds(else_branch, seen, out);
        }
        TLExpr::Let { value, body, .. } => {
            collect_preds(value, seen, out);
            collect_preds(body, seen, out);
        }
        TLExpr::Until { before, after }
        | TLExpr::WeakUntil { before, after }
        | TLExpr::Release {
            released: before,
            releaser: after,
        }
        | TLExpr::StrongRelease {
            released: before,
            releaser: after,
        } => {
            collect_preds(before, seen, out);
            collect_preds(after, seen, out);
        }
        TLExpr::TNorm { left, right, .. } | TLExpr::TCoNorm { left, right, .. } => {
            collect_preds(left, seen, out);
            collect_preds(right, seen, out);
        }
        TLExpr::FuzzyNot { expr: inner, .. } => {
            collect_preds(inner, seen, out);
        }
        TLExpr::FuzzyImplication {
            premise,
            conclusion,
            ..
        } => {
            collect_preds(premise, seen, out);
            collect_preds(conclusion, seen, out);
        }
        TLExpr::ProbabilisticChoice { alternatives } => {
            for (_, alt_expr) in alternatives {
                collect_preds(alt_expr, seen, out);
            }
        }
        // Lambda, Apply, Var, etc. — recurse where possible
        TLExpr::Constant(_) => {}
        // Catch-all for any new variants: treat as leaf.
        _ => {}
    }
}

// ── Document builders ─────────────────────────────────────────────────────────

/// Construct a [`TlJsonLdNode`] for a named entity with a given type and properties.
pub fn build_entity_node(
    id: &str,
    type_iri: &str,
    properties: &[(&str, JsonLdValue)],
) -> TlJsonLdNode {
    let mut node = TlJsonLdNode::new().with_id(id).with_type(type_iri);
    for (key, value) in properties {
        node.add_property(*key, value.clone());
    }
    node
}

/// Serialize a [`TLExpr`] as a JSON-LD node representing the logic expression
/// as a linked data resource.
///
/// The mapping follows these conventions:
/// - `Pred(name, args)` → node with `@type = name`, `rdf:subject` = args list.
/// - `And`/`Or`/`Not` → compound nodes with `tl:operator` and child references.
/// - `Exists`/`ForAll` → nodes with quantifier type, variable and domain.
/// - Arithmetic / comparison leaves → nodes with operator type and operand values.
///
/// The caller provides `base_iri` (for IRI construction) and `node_id` (the
/// `@id` of the root node produced).
pub fn expr_to_json_ld_node(expr: &TLExpr, base_iri: &str, node_id: &str) -> TlJsonLdNode {
    let tl = format!("{}tl#", base_iri);
    let mut node = TlJsonLdNode::new().with_id(node_id);

    match expr {
        TLExpr::Pred { name, args } => {
            node = node.with_type(format!("{}{}", tl, name));
            let arg_values: Vec<JsonLdValue> = args
                .iter()
                .enumerate()
                .map(|(i, term)| {
                    let term_str = format!("{:?}", term);
                    JsonLdValue::Object(vec![
                        (format!("{}argIndex", tl), JsonLdValue::Number(i as f64)),
                        (format!("{}term", tl), JsonLdValue::Str(term_str)),
                    ])
                })
                .collect();
            node.add_property(format!("{}arguments", tl), JsonLdValue::Array(arg_values));
        }
        TLExpr::And(left, right) => {
            node = node.with_type(format!("{}AndExpression", tl));
            node.add_property(
                format!("{}operator", tl),
                JsonLdValue::Str("AND".to_string()),
            );
            let left_node = expr_to_json_ld_node(left, base_iri, &format!("{}_left", node_id));
            let right_node = expr_to_json_ld_node(right, base_iri, &format!("{}_right", node_id));
            node.add_property(format!("{}left", tl), left_node.to_json_value());
            node.add_property(format!("{}right", tl), right_node.to_json_value());
        }
        TLExpr::Or(left, right) => {
            node = node.with_type(format!("{}OrExpression", tl));
            node.add_property(
                format!("{}operator", tl),
                JsonLdValue::Str("OR".to_string()),
            );
            let left_node = expr_to_json_ld_node(left, base_iri, &format!("{}_left", node_id));
            let right_node = expr_to_json_ld_node(right, base_iri, &format!("{}_right", node_id));
            node.add_property(format!("{}left", tl), left_node.to_json_value());
            node.add_property(format!("{}right", tl), right_node.to_json_value());
        }
        TLExpr::Not(inner) => {
            node = node.with_type(format!("{}NotExpression", tl));
            node.add_property(
                format!("{}operator", tl),
                JsonLdValue::Str("NOT".to_string()),
            );
            let inner_node = expr_to_json_ld_node(inner, base_iri, &format!("{}_inner", node_id));
            node.add_property(format!("{}operand", tl), inner_node.to_json_value());
        }
        TLExpr::Exists { var, domain, body } => {
            node = node.with_type(format!("{}ExistsExpression", tl));
            node.add_property(
                format!("{}quantifier", tl),
                JsonLdValue::Str("EXISTS".to_string()),
            );
            node.add_property(format!("{}variable", tl), JsonLdValue::Str(var.clone()));
            node.add_property(format!("{}domain", tl), JsonLdValue::Str(domain.clone()));
            let body_node = expr_to_json_ld_node(body, base_iri, &format!("{}_body", node_id));
            node.add_property(format!("{}body", tl), body_node.to_json_value());
        }
        TLExpr::ForAll { var, domain, body } => {
            node = node.with_type(format!("{}ForAllExpression", tl));
            node.add_property(
                format!("{}quantifier", tl),
                JsonLdValue::Str("FORALL".to_string()),
            );
            node.add_property(format!("{}variable", tl), JsonLdValue::Str(var.clone()));
            node.add_property(format!("{}domain", tl), JsonLdValue::Str(domain.clone()));
            let body_node = expr_to_json_ld_node(body, base_iri, &format!("{}_body", node_id));
            node.add_property(format!("{}body", tl), body_node.to_json_value());
        }
        TLExpr::Imply(premise, conclusion) => {
            node = node.with_type(format!("{}ImplicationExpression", tl));
            node.add_property(
                format!("{}operator", tl),
                JsonLdValue::Str("IMPLIES".to_string()),
            );
            let prem_node =
                expr_to_json_ld_node(premise, base_iri, &format!("{}_premise", node_id));
            let conc_node =
                expr_to_json_ld_node(conclusion, base_iri, &format!("{}_conclusion", node_id));
            node.add_property(format!("{}premise", tl), prem_node.to_json_value());
            node.add_property(format!("{}conclusion", tl), conc_node.to_json_value());
        }
        TLExpr::Constant(n) => {
            node = node.with_type(format!("{}NumericConstant", tl));
            node.add_property(format!("{}value", tl), JsonLdValue::Number(*n));
        }
        TLExpr::WeightedRule { weight, rule } => {
            node = node.with_type(format!("{}WeightedRule", tl));
            node.add_property(format!("{}weight", tl), JsonLdValue::Number(*weight));
            let rule_node = expr_to_json_ld_node(rule, base_iri, &format!("{}_rule", node_id));
            node.add_property(format!("{}rule", tl), rule_node.to_json_value());
        }
        _ => {
            // Generic fallback: emit the debug representation as a string literal.
            node = node.with_type(format!("{}Expression", tl));
            node.add_property(
                format!("{}repr", tl),
                JsonLdValue::Str(format!("{:?}", expr)),
            );
        }
    }

    node
}

// ── Validation ────────────────────────────────────────────────────────────────

/// Validate a [`TlJsonLdDocument`] for common structural problems.
///
/// Returns a (possibly empty) list of errors. Checks performed:
/// 1. Duplicate `@id` values across all nodes.
/// 2. Compact IRI usage in term definitions where the prefix is not registered.
/// 3. Nodes that have an empty `@type` list when one is expected (currently
///    informational — not checked here to avoid false positives).
pub fn validate_document(doc: &TlJsonLdDocument) -> Vec<JsonLdError> {
    let mut errors: Vec<JsonLdError> = Vec::new();

    // 1. Duplicate @id detection.
    let mut seen_ids: HashSet<String> = HashSet::new();
    for node in &doc.nodes {
        if let Some(id) = &node.id {
            if !seen_ids.insert(id.clone()) {
                errors.push(JsonLdError::DuplicateId(id.clone()));
            }
        }
    }

    // 2. Undefined prefix detection in term IRIs.
    let registered_prefixes: HashSet<&str> = doc
        .context
        .prefixes
        .iter()
        .map(|(p, _)| p.as_str())
        .collect();

    for term in &doc.context.terms {
        // Check compact IRIs in the term's IRI field (e.g. "ex:Person")
        if !term.iri.contains("://") {
            if let Some(colon_pos) = term.iri.find(':') {
                let prefix = &term.iri[..colon_pos];
                if !registered_prefixes.contains(prefix) {
                    errors.push(JsonLdError::UndefinedPrefix(prefix.to_string()));
                }
            }
        }
    }

    errors
}

// ── Internal helpers ──────────────────────────────────────────────────────────

/// Escape a string for safe embedding inside a JSON double-quoted string.
fn json_escape(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    for ch in s.chars() {
        match ch {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            c => out.push(c),
        }
    }
    out
}

// ── Re-exports with canonical aliases ─────────────────────────────────────────

// The public type aliases used from lib.rs are defined here so external
// consumers can write e.g.  `use tensorlogic_oxirs_bridge::json_ld::TlContextTerm`.
/// Alias for [`ContextTerm`] — the canonical public name in this module.
pub type TlContextTerm = ContextTerm;

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use tensorlogic_ir::{TLExpr, Term};

    // 1. ContextTerm::new creates term with correct name and IRI
    #[test]
    fn test_context_term_new() {
        let term = ContextTerm::new("knows", "http://example.org/knows");
        assert_eq!(term.name, "knows");
        assert_eq!(term.iri, "http://example.org/knows");
        assert!(term.term_type.is_none());
        assert!(term.container.is_none());
        assert!(!term.reverse);
    }

    // 2. ContextTerm::with_type sets @type
    #[test]
    fn test_context_term_with_type() {
        let term = ContextTerm::new("knows", "http://example.org/knows").with_type("@id");
        assert_eq!(term.term_type.as_deref(), Some("@id"));
    }

    // 3. ContextTerm::to_json_value renders correctly
    #[test]
    fn test_context_term_to_json_value_plain() {
        let term = ContextTerm::new("knows", "http://example.org/knows");
        let val = term.to_json_value();
        // Simple IRI: should be a plain string
        match val {
            JsonLdValue::Str(s) => assert_eq!(s, "http://example.org/knows"),
            other => panic!("Expected Str, got {:?}", other),
        }
    }

    #[test]
    fn test_context_term_to_json_value_with_type() {
        let term = ContextTerm::new("knows", "http://example.org/knows").with_type("@id");
        let val = term.to_json_value();
        let s = val.to_json_string();
        assert!(s.contains("@id"));
        assert!(s.contains("http://example.org/knows"));
    }

    // 4. JsonLdContext::expand_iri expands "rdf:type"
    #[test]
    fn test_expand_iri_rdf_type() {
        let ctx = standard_prefixes_context();
        let expanded = ctx.expand_iri("rdf:type");
        assert_eq!(expanded, "http://www.w3.org/1999/02/22-rdf-syntax-ns#type");
    }

    // 5. JsonLdContext::expand_iri passes through absolute IRIs
    #[test]
    fn test_expand_iri_absolute() {
        let ctx = TlJsonLdContext::new();
        let iri = "http://example.org/Person";
        assert_eq!(ctx.expand_iri(iri), iri);
    }

    // 6. JsonLdContext::to_json_string contains "@context" key indirectly via document
    #[test]
    fn test_context_to_json_string_has_content() {
        let mut ctx = TlJsonLdContext::new();
        ctx.add_prefix("rdf", "http://www.w3.org/1999/02/22-rdf-syntax-ns#");
        let s = ctx.to_json_string();
        assert!(s.contains("rdf"));
        assert!(s.contains("http://www.w3.org/1999/02/22-rdf-syntax-ns#"));
    }

    // 7. context_from_predicates creates terms for each predicate
    #[test]
    fn test_context_from_predicates() {
        let preds = vec![("Person", 1usize), ("knows", 2usize)];
        let ctx =
            context_from_predicates(&preds, "http://example.org/", None).expect("should succeed");
        assert_eq!(ctx.term_count(), 2);
        let names: Vec<&str> = ctx.terms.iter().map(|t| t.name.as_str()).collect();
        assert!(names.contains(&"Person"));
        assert!(names.contains(&"knows"));
    }

    // 8. standard_prefixes_context contains "rdf" prefix
    #[test]
    fn test_standard_prefixes_has_rdf() {
        let ctx = standard_prefixes_context();
        let has_rdf = ctx.prefixes.iter().any(|(p, _)| p == "rdf");
        assert!(has_rdf);
    }

    // 9. JsonLdValue::to_json_string on Str wraps in quotes
    #[test]
    fn test_jsonld_value_str_quoted() {
        let val = JsonLdValue::string("hello");
        assert_eq!(val.to_json_string(), "\"hello\"");
    }

    // 10. JsonLdValue::id creates {"@id": "..."}
    #[test]
    fn test_jsonld_value_id() {
        let val = JsonLdValue::id("http://example.org/alice");
        let s = val.to_json_string();
        assert!(s.contains("@id"));
        assert!(s.contains("http://example.org/alice"));
    }

    // 11. JsonLdValue::typed_value creates {"@value": ..., "@type": ...}
    #[test]
    fn test_jsonld_value_typed() {
        let val = JsonLdValue::typed_value("42", "xsd:integer");
        let s = val.to_json_string();
        assert!(s.contains("@value"));
        assert!(s.contains("@type"));
        assert!(s.contains("xsd:integer"));
    }

    // 12. JsonLdValue::to_json_string on Array renders [...]
    #[test]
    fn test_jsonld_value_array() {
        let val = JsonLdValue::Array(vec![
            JsonLdValue::Str("a".to_string()),
            JsonLdValue::Str("b".to_string()),
        ]);
        let s = val.to_json_string();
        assert!(s.starts_with('['));
        assert!(s.ends_with(']'));
        assert!(s.contains("\"a\""));
        assert!(s.contains("\"b\""));
    }

    // 13. JsonLdValue::to_json_string on Object renders {...}
    #[test]
    fn test_jsonld_value_object() {
        let val = JsonLdValue::Object(vec![(
            "key".to_string(),
            JsonLdValue::Str("val".to_string()),
        )]);
        let s = val.to_json_string();
        assert!(s.starts_with('{'));
        assert!(s.ends_with('}'));
        assert!(s.contains("\"key\""));
        assert!(s.contains("\"val\""));
    }

    // 14. JsonLdValue::to_pretty_string has indentation
    #[test]
    fn test_jsonld_value_pretty_has_newlines() {
        let val = JsonLdValue::Object(vec![
            ("a".to_string(), JsonLdValue::Str("b".to_string())),
            ("c".to_string(), JsonLdValue::Str("d".to_string())),
        ]);
        let s = val.to_pretty_string(0);
        assert!(s.contains('\n'), "pretty string should contain newlines");
    }

    // 15. JsonLdNode::with_type adds to types list
    #[test]
    fn test_node_with_type() {
        let node = TlJsonLdNode::new().with_type("http://example.org/Person");
        assert_eq!(node.types.len(), 1);
        assert_eq!(node.types[0], "http://example.org/Person");
    }

    // 16. JsonLdNode::add_property adds to properties
    #[test]
    fn test_node_add_property() {
        let mut node = TlJsonLdNode::new();
        node.add_property("ex:name", JsonLdValue::string("Alice"));
        assert_eq!(node.properties.len(), 1);
        assert_eq!(node.properties[0].0, "ex:name");
    }

    // 17. JsonLdNode::to_json_value includes "@id" when set
    #[test]
    fn test_node_to_json_value_with_id() {
        let node = TlJsonLdNode::new().with_id("http://example.org/alice");
        let v = node.to_json_value().to_json_string();
        assert!(v.contains("@id"));
        assert!(v.contains("http://example.org/alice"));
    }

    // 18. JsonLdNode::to_json_value includes "@type" array
    #[test]
    fn test_node_to_json_value_multiple_types() {
        let node = TlJsonLdNode::new()
            .with_type("http://example.org/Person")
            .with_type("http://example.org/Agent");
        let v = node.to_json_value().to_json_string();
        assert!(v.contains("@type"));
        assert!(v.contains("Person"));
        assert!(v.contains("Agent"));
    }

    // 19. JsonLdDocument::to_json_string contains "@context" and node IDs
    #[test]
    fn test_document_to_json_string() {
        let ctx = standard_prefixes_context();
        let mut doc = TlJsonLdDocument::new(ctx);
        doc.add_node(
            TlJsonLdNode::new()
                .with_id("http://example.org/alice")
                .with_type("http://example.org/Person"),
        );
        let s = doc.to_json_string();
        assert!(s.contains("@context"));
        assert!(s.contains("http://example.org/alice"));
    }

    // 20. JsonLdDocument::to_pretty_string is multiline
    #[test]
    fn test_document_to_pretty_string_multiline() {
        let ctx = standard_prefixes_context();
        let mut doc = TlJsonLdDocument::new(ctx);
        doc.add_node(TlJsonLdNode::new().with_id("http://example.org/x"));
        let s = doc.to_pretty_string();
        assert!(s.contains('\n'), "to_pretty_string should be multiline");
    }

    // 21. validate_document detects duplicate @id
    #[test]
    fn test_validate_duplicate_id() {
        let ctx = TlJsonLdContext::new();
        let mut doc = TlJsonLdDocument::new(ctx);
        doc.add_node(TlJsonLdNode::new().with_id("http://example.org/x"));
        doc.add_node(TlJsonLdNode::new().with_id("http://example.org/x"));
        let errors = validate_document(&doc);
        assert!(
            errors
                .iter()
                .any(|e| matches!(e, JsonLdError::DuplicateId(_))),
            "should detect duplicate @id"
        );
    }

    // 22. build_entity_node creates node with correct type and properties
    #[test]
    fn test_build_entity_node() {
        let node = build_entity_node(
            "http://example.org/alice",
            "http://example.org/Person",
            &[("ex:name", JsonLdValue::string("Alice"))],
        );
        assert_eq!(node.id.as_deref(), Some("http://example.org/alice"));
        assert!(node
            .types
            .contains(&"http://example.org/Person".to_string()));
        assert_eq!(node.properties.len(), 1);
        assert_eq!(node.properties[0].0, "ex:name");
    }

    // 23. context_from_expr scans Pred names from expression
    #[test]
    fn test_context_from_expr() {
        let expr = TLExpr::and(
            TLExpr::pred("Person", vec![Term::var("x")]),
            TLExpr::pred("knows", vec![Term::var("x"), Term::var("y")]),
        );
        let ctx = context_from_expr(&expr, "http://example.org/").expect("should succeed");
        let names: Vec<&str> = ctx.terms.iter().map(|t| t.name.as_str()).collect();
        assert!(names.contains(&"Person"), "should contain Person");
        assert!(names.contains(&"knows"), "should contain knows");
    }

    // 24. JsonLdError Display shows meaningful message
    #[test]
    fn test_error_display() {
        let e = JsonLdError::InvalidIri("bad-iri".to_string());
        let s = e.to_string();
        assert!(
            s.contains("bad-iri"),
            "error display should mention the IRI"
        );

        let e2 = JsonLdError::DuplicateId("http://example.org/x".to_string());
        let s2 = e2.to_string();
        assert!(s2.contains("http://example.org/x"));

        let e3 = JsonLdError::UndefinedPrefix("unknown".to_string());
        let s3 = e3.to_string();
        assert!(s3.contains("unknown"));

        let e4 = JsonLdError::EmptyDocument;
        let s4 = e4.to_string();
        assert!(!s4.is_empty());

        let e5 = JsonLdError::SerializationError("oops".to_string());
        let s5 = e5.to_string();
        assert!(s5.contains("oops"));
    }

    // Extra: validate_document detects undefined prefix
    #[test]
    fn test_validate_undefined_prefix() {
        let ctx = TlJsonLdContext::new(); // no prefixes registered
        let mut doc_ctx = ctx;
        // Add a term with a compact IRI whose prefix isn't registered
        doc_ctx.add_term(ContextTerm::new("thing", "ex:Thing"));
        let doc = TlJsonLdDocument::new(doc_ctx);
        let errors = validate_document(&doc);
        assert!(
            errors
                .iter()
                .any(|e| matches!(e, JsonLdError::UndefinedPrefix(_))),
            "should detect undefined prefix 'ex'"
        );
    }

    // Extra: expr_to_json_ld_node for Pred produces correct type
    #[test]
    fn test_expr_to_json_ld_node_pred() {
        let expr = TLExpr::pred("Person", vec![Term::var("x")]);
        let node = expr_to_json_ld_node(&expr, "http://example.org/", "node_1");
        assert_eq!(node.id.as_deref(), Some("node_1"));
        assert!(
            node.types.iter().any(|t| t.contains("Person")),
            "type should contain predicate name"
        );
    }

    // Extra: TlJsonLdContext term_count
    #[test]
    fn test_context_term_count() {
        let mut ctx = TlJsonLdContext::new();
        assert_eq!(ctx.term_count(), 0);
        ctx.add_term(ContextTerm::new("a", "http://example.org/a"));
        ctx.add_term(ContextTerm::new("b", "http://example.org/b"));
        assert_eq!(ctx.term_count(), 2);
    }

    // Extra: TlJsonLdDocument node_count
    #[test]
    fn test_document_node_count() {
        let ctx = TlJsonLdContext::new();
        let mut doc = TlJsonLdDocument::new(ctx);
        assert_eq!(doc.node_count(), 0);
        doc.add_node(TlJsonLdNode::new());
        assert_eq!(doc.node_count(), 1);
    }

    // Extra: context_from_predicates with invalid base_iri
    #[test]
    fn test_context_from_predicates_empty_base() {
        let result = context_from_predicates(&[("foo", 1)], "", None);
        assert!(result.is_err());
    }

    // Extra: ContextTerm as_reverse
    #[test]
    fn test_context_term_as_reverse() {
        let term = ContextTerm::new("isMemberOf", "http://example.org/member").as_reverse();
        assert!(term.reverse);
        let s = term.to_json_value().to_json_string();
        assert!(s.contains("@reverse"));
    }

    // Extra: ContextTerm with_container
    #[test]
    fn test_context_term_with_container() {
        let term = ContextTerm::new("tags", "http://example.org/tag").with_container("@set");
        assert_eq!(term.container.as_deref(), Some("@set"));
        let s = term.to_json_value().to_json_string();
        assert!(s.contains("@container"));
        assert!(s.contains("@set"));
    }
}
