//! The high-level JSON-LD 1.1 API: [`expand`], [`to_rdf`], [`from_rdf`], [`compact`], and
//! [`flatten`], plus the [`JsonLdOptions`] that configure them.
//!
//! These functions operate on the crate's [`JsonValue`] tree and mirror the
//! [JSON-LD 1.1 API](https://www.w3.org/TR/json-ld11-api/) processing algorithms. Framing is a
//! documented gap (see the crate root).

use crate::compaction::compact_document;
use crate::context::{Context, ContextProcessor};
use crate::error::JsonLdSyntaxError;
use crate::expansion::Expander;
use crate::flattening::flatten_expanded;
use crate::from_rdf::from_rdf_quads;
use crate::loader::{DocumentLoader, NoLoader};
use crate::node_map::{BlankNodeNamer, generate_node_map};
use crate::profile::{JsonLdProcessingMode, JsonLdProfile, JsonLdProfileSet};
use crate::to_rdf::{RdfConverter, RdfDirection, ToRdfOptions};
use crate::value::{JsonObject, JsonValue};
use oxixml_model::Quad;

static DEFAULT_LOADER: NoLoader = NoLoader;

/// Options shared by the JSON-LD API operations.
///
/// Construct with [`JsonLdOptions::new`] and adjust fields as needed. The `loader` defaults to a
/// no-op [`NoLoader`]; supply your own [`DocumentLoader`] to resolve remote contexts.
pub struct JsonLdOptions<'a> {
    /// The base IRI used for relative reference resolution.
    pub base: Option<String>,
    /// The processing mode.
    pub processing_mode: JsonLdProcessingMode,
    /// Whether to skip strict IRI validation.
    pub lenient: bool,
    /// Whether to sort map entries for deterministic output.
    pub ordered: bool,
    /// A context to apply before expanding the document (`expandContext`).
    pub expand_context: Option<JsonValue>,
    /// Whether to emit generalized RDF (blank node predicates).
    pub produce_generalized_rdf: bool,
    /// How to represent `@direction` in RDF.
    pub rdf_direction: RdfDirection,
    /// Whether `fromRdf` should use native JSON scalars for `xsd:integer`/`double`/`boolean`.
    pub use_native_types: bool,
    /// Whether `fromRdf` should keep `rdf:type` as a property rather than `@type`.
    pub use_rdf_type: bool,
    /// Whether compaction collapses single-element arrays.
    pub compact_arrays: bool,
    /// Whether compaction produces relative IRIs against the base.
    pub compact_to_relative: bool,
    /// The document loader used to resolve remote references.
    pub loader: &'a dyn DocumentLoader,
    /// The maximum JSON nesting depth the algorithms will descend into.
    ///
    /// The JSON-LD algorithms are specified recursively, so one native stack frame is consumed
    /// per nesting level; without a bound a modestly sized untrusted document would abort the
    /// process with a stack overflow. Defaults to
    /// [`DEFAULT_MAX_NESTING_DEPTH`](crate::DEFAULT_MAX_NESTING_DEPTH).
    pub max_nesting_depth: usize,
    /// The profile(s) the input document is assumed to follow.
    ///
    /// Including [`JsonLdProfile::Streaming`] makes [`to_rdf`] additionally enforce the
    /// [streaming document form](https://www.w3.org/TR/json-ld11-streaming/)'s key-order rules.
    pub profile: JsonLdProfileSet,
}

impl Default for JsonLdOptions<'_> {
    fn default() -> Self {
        Self::new()
    }
}

impl<'a> JsonLdOptions<'a> {
    /// Builds default options (JSON-LD 1.1, `compactArrays` on, no remote loading).
    #[must_use]
    pub fn new() -> Self {
        Self {
            base: None,
            processing_mode: JsonLdProcessingMode::JsonLd1_1,
            lenient: false,
            ordered: false,
            expand_context: None,
            produce_generalized_rdf: false,
            rdf_direction: RdfDirection::None,
            use_native_types: false,
            use_rdf_type: false,
            compact_arrays: true,
            compact_to_relative: true,
            loader: &DEFAULT_LOADER,
            max_nesting_depth: crate::value::DEFAULT_MAX_NESTING_DEPTH,
            profile: JsonLdProfileSet::empty(),
        }
    }

    /// Sets the base IRI.
    #[must_use]
    pub fn with_base(mut self, base: impl Into<String>) -> Self {
        self.base = Some(base.into());
        self
    }

    /// Sets the maximum JSON nesting depth (see
    /// [`max_nesting_depth`](Self::max_nesting_depth)).
    #[must_use]
    pub fn with_max_nesting_depth(mut self, max_nesting_depth: usize) -> Self {
        self.max_nesting_depth = max_nesting_depth;
        self
    }

    /// Sets the document loader.
    #[must_use]
    pub fn with_loader(mut self, loader: &'a dyn DocumentLoader) -> Self {
        self.loader = loader;
        self
    }

    fn processor(&self) -> ContextProcessor<'a> {
        ContextProcessor::new(self.processing_mode, self.lenient, self.loader)
    }

    fn initial_context(
        &self,
        processor: &ContextProcessor<'_>,
    ) -> Result<Context, JsonLdSyntaxError> {
        let mut context = Context::new_empty(self.base.clone());
        if let Some(expand_context) = &self.expand_context {
            let expand_context = match expand_context {
                JsonValue::Object(o) if o.contains_key("@context") => {
                    o.get("@context").cloned().unwrap_or(JsonValue::Null)
                }
                other => other.clone(),
            };
            let mut errors = Vec::new();
            context = processor.process_context(
                &context,
                expand_context,
                self.base.as_deref(),
                &mut Vec::new(),
                false,
                true,
                true,
                &mut errors,
            );
            if let Some(error) = errors.into_iter().next() {
                return Err(error);
            }
        }
        Ok(context)
    }
}

/// The prefixes and base IRI a document's top-level `@context` declares.
///
/// Only the *document's own* context is inspected — remote contexts pulled in through the loader
/// are processed too, so the prefixes reported are those in effect after the whole top-level
/// context has been applied. A term counts as a prefix when it can introduce a compact IRI.
pub(crate) fn document_prefixes(
    document: &JsonValue,
    options: &JsonLdOptions<'_>,
) -> (Vec<(String, String)>, Option<String>) {
    let Some(local_context) = document
        .as_object()
        .and_then(|object| object.get("@context"))
    else {
        return (Vec::new(), None);
    };
    let processor = options.processor();
    let base = Context::new_empty(options.base.clone());
    let context = processor.process_context(
        &base,
        local_context.clone(),
        options.base.as_deref(),
        &mut Vec::new(),
        false,
        true,
        true,
        &mut Vec::new(),
    );
    let mut prefixes: Vec<(String, String)> = context
        .term_definitions
        .iter()
        .filter(|(_, definition)| definition.prefix_flag)
        .filter_map(|(term, definition)| definition.iri().map(|iri| (term.clone(), iri.to_owned())))
        .collect();
    prefixes.sort_unstable();
    (prefixes, context.base_iri)
}

/// [Expands](https://www.w3.org/TR/json-ld11-api/#expansion) `input`.
pub fn expand(
    input: JsonValue,
    options: &JsonLdOptions<'_>,
) -> Result<JsonValue, JsonLdSyntaxError> {
    let processor = options.processor();
    let active_context = options.initial_context(&processor)?;
    let mut expander = Expander::new(&processor, options.ordered)
        .with_max_depth(options.max_nesting_depth)
        .with_streaming_profile(options.profile.contains(JsonLdProfile::Streaming));
    let result = expander.expand_document(&active_context, input, options.base.as_deref());
    if let Some(error) = expander.errors.into_iter().next() {
        return Err(error);
    }
    Ok(result)
}

/// [Serializes](https://www.w3.org/TR/json-ld11-api/#rdf-serialization-deserialization-algorithms)
/// `input` to a set of RDF [`Quad`]s (the *toRdf* operation).
pub fn to_rdf(
    input: JsonValue,
    options: &JsonLdOptions<'_>,
) -> Result<Vec<Quad>, JsonLdSyntaxError> {
    let expanded = expand(input, options)?;
    let mut node_map = JsonObject::new();
    let mut namer = BlankNodeNamer::new();
    let mut errors = Vec::new();
    generate_node_map(
        expanded,
        &mut node_map,
        "@default",
        None,
        None,
        None,
        &mut namer,
        &mut errors,
    );
    if let Some(error) = errors.into_iter().next() {
        return Err(error);
    }
    let rdf_options = ToRdfOptions {
        produce_generalized_rdf: options.produce_generalized_rdf,
        rdf_direction: options.rdf_direction,
    };
    Ok(RdfConverter::new(rdf_options).convert(&node_map))
}

/// [Deserializes](https://www.w3.org/TR/json-ld11-api/#serialize-rdf-as-json-ld-algorithm) a set
/// of RDF quads into an expanded JSON-LD document (the *fromRdf* operation).
pub fn from_rdf(
    quads: &[Quad],
    options: &JsonLdOptions<'_>,
) -> Result<JsonValue, JsonLdSyntaxError> {
    from_rdf_quads(quads, options)
}

/// [Flattens](https://www.w3.org/TR/json-ld11-api/#flattening) `input`.
///
/// When `context` is provided, the flattened result is compacted with it.
pub fn flatten(
    input: JsonValue,
    context: Option<JsonValue>,
    options: &JsonLdOptions<'_>,
) -> Result<JsonValue, JsonLdSyntaxError> {
    let expanded = expand(input, options)?;
    let flattened = flatten_expanded(expanded, options.ordered)?;
    match context {
        Some(context) => compact(flattened, context, options),
        None => Ok(flattened),
    }
}

/// [Compacts](https://www.w3.org/TR/json-ld11-api/#compaction) `input` using `context`.
pub fn compact(
    input: JsonValue,
    context: JsonValue,
    options: &JsonLdOptions<'_>,
) -> Result<JsonValue, JsonLdSyntaxError> {
    let expanded = expand(input, options)?;
    let processor = options.processor();
    let context_value = match &context {
        JsonValue::Object(o) if o.contains_key("@context") => {
            o.get("@context").cloned().unwrap_or(JsonValue::Null)
        }
        other => other.clone(),
    };
    compact_document(&processor, expanded, context_value, options)
}
