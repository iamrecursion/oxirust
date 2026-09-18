//! [Context processing](https://www.w3.org/TR/json-ld-api/#context-processing-algorithms):
//! the active context, term definitions, and the IRI-expansion helper shared with expansion.
//!
//! This module implements the JSON-LD 1.1 API's *Context Processing Algorithm*, *Create Term
//! Definition Algorithm*, and (the term-definition-time flavour of) *IRI Expansion*, operating
//! over the crate's internal [`JsonValue`] tree. Remote `@context` and `@import` references are
//! resolved through the caller-supplied [`DocumentLoader`]; the default loader performs no I/O.

use crate::error::{JsonLdErrorCode, JsonLdSyntaxError};
use crate::loader::{DocumentLoader, LoadDocumentOptions};
use crate::profile::{JsonLdProcessingMode, JsonLdProfile};
use crate::value::{JsonObject, JsonValue};
use oxixml_iri::{Iri, IriRef};
use std::cell::RefCell;
use std::collections::HashMap;

/// The maximum number of nested remote contexts allowed before a
/// [`JsonLdErrorCode::ContextOverflow`] is raised.
pub const MAX_CONTEXT_RECURSION: usize = 16;

/// A base text direction (`@direction`).
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Direction {
    /// Left-to-right (`"ltr"`).
    Ltr,
    /// Right-to-left (`"rtl"`).
    Rtl,
}

impl Direction {
    /// The `"ltr"` / `"rtl"` string form.
    #[must_use]
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Ltr => "ltr",
            Self::Rtl => "rtl",
        }
    }

    /// Parses `"ltr"` / `"rtl"`.
    #[must_use]
    pub fn parse(value: &str) -> Option<Self> {
        match value {
            "ltr" => Some(Self::Ltr),
            "rtl" => Some(Self::Rtl),
            _ => None,
        }
    }
}

/// A single term definition in an active context.
///
/// `iri_mapping` follows the specification's tri-state convention: `None` means "unset",
/// `Some(None)` means "explicitly mapped to null" (so the term is dropped during expansion), and
/// `Some(Some(iri))` means the term maps to `iri`.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct TermDefinition {
    /// The IRI (or keyword) this term maps to. See the type documentation for the tri-state.
    pub iri_mapping: Option<Option<String>>,
    /// Whether this term may be used as a prefix in a compact IRI.
    pub prefix_flag: bool,
    /// Whether this term definition is protected.
    pub protected: bool,
    /// Whether this term is a reverse property.
    pub reverse_property: bool,
    /// The base URL used to resolve the term's scoped context, if any.
    pub base_url: Option<String>,
    /// The term's scoped `@context`, kept in raw form for later processing. `Some(JsonValue::Null)`
    /// represents an explicit null scoped context.
    pub context: Option<JsonValue>,
    /// The `@container` mapping, as a sorted set of keyword strings.
    pub container_mapping: Vec<String>,
    /// The `@direction` mapping: `None` unset, `Some(None)` null, `Some(Some(dir))` a value.
    pub direction_mapping: Option<Option<Direction>>,
    /// The `@index` mapping.
    pub index_mapping: Option<String>,
    /// The `@language` mapping: `None` unset, `Some(None)` null, `Some(Some(lang))` a value.
    pub language_mapping: Option<Option<String>>,
    /// The `@nest` value.
    pub nest_value: Option<String>,
    /// The `@type` mapping.
    pub type_mapping: Option<String>,
}

impl TermDefinition {
    /// The resolved IRI/keyword this term maps to, if it is non-null.
    #[must_use]
    pub fn iri(&self) -> Option<&str> {
        self.iri_mapping.as_ref().and_then(|m| m.as_deref())
    }

    /// Whether the container mapping contains `keyword`.
    #[must_use]
    pub fn has_container(&self, keyword: &str) -> bool {
        self.container_mapping.iter().any(|c| c == keyword)
    }
}

/// An active context.
#[derive(Clone, Debug, Default)]
pub struct Context {
    /// The current base IRI (`@base`).
    pub base_iri: Option<String>,
    /// The document's original base URL, restored when the context is nullified.
    pub original_base_url: Option<String>,
    /// The vocabulary mapping (`@vocab`).
    pub vocabulary_mapping: Option<String>,
    /// The default language (`@language`).
    pub default_language: Option<String>,
    /// The default base direction (`@direction`).
    pub default_direction: Option<Direction>,
    /// The term definitions, keyed by term.
    pub term_definitions: HashMap<String, TermDefinition>,
    /// The previous context, used when a context is not propagated.
    pub previous_context: Option<Box<Context>>,
}

impl Context {
    /// Creates an empty context whose base IRI is the given original base URL.
    #[must_use]
    pub fn new_empty(original_base_url: Option<String>) -> Self {
        Self {
            base_iri: original_base_url.clone(),
            original_base_url,
            vocabulary_mapping: None,
            default_language: None,
            default_direction: None,
            term_definitions: HashMap::new(),
            previous_context: None,
        }
    }

    /// Looks up a term definition.
    #[must_use]
    pub fn term(&self, term: &str) -> Option<&TermDefinition> {
        self.term_definitions.get(term)
    }
}

/// A cached remote context: its final URL (as a base) plus the raw `@context` value.
type RemoteContextEntry = (Option<String>, JsonValue);

/// Runs the context-processing and IRI-expansion algorithms.
pub struct ContextProcessor<'a> {
    /// The active processing mode.
    pub processing_mode: JsonLdProcessingMode,
    /// Whether to skip strict IRI validation (custom leniency option).
    pub lenient: bool,
    /// The maximum remote-context recursion depth.
    pub max_context_recursion: usize,
    /// The document loader used to resolve remote references.
    pub loader: &'a dyn DocumentLoader,
    /// Cache of already-loaded remote contexts.
    pub remote_context_cache: RefCell<HashMap<String, RemoteContextEntry>>,
}

impl<'a> ContextProcessor<'a> {
    /// Creates a new processor with the default recursion limit.
    #[must_use]
    pub fn new(
        processing_mode: JsonLdProcessingMode,
        lenient: bool,
        loader: &'a dyn DocumentLoader,
    ) -> Self {
        Self {
            processing_mode,
            lenient,
            max_context_recursion: MAX_CONTEXT_RECURSION,
            loader,
            remote_context_cache: RefCell::new(HashMap::new()),
        }
    }

    /// [Context Processing Algorithm](https://www.w3.org/TR/json-ld-api/#algorithm).
    #[allow(clippy::too_many_arguments)]
    pub fn process_context(
        &self,
        active_context: &Context,
        local_context: JsonValue,
        base_url: Option<&str>,
        remote_contexts: &mut Vec<String>,
        override_protected: bool,
        mut propagate: bool,
        validate_scoped_context: bool,
        errors: &mut Vec<JsonLdSyntaxError>,
    ) -> Context {
        // 1)
        let mut result = active_context.clone();
        // 2)
        if let JsonValue::Object(object) = &local_context {
            if let Some(node) = object.get("@propagate") {
                if let JsonValue::Bool(new) = node {
                    propagate = *new;
                } else {
                    errors.push(JsonLdSyntaxError::msg(
                        "@propagate value must be a boolean",
                        JsonLdErrorCode::InvalidPropagateValue,
                    ));
                }
            }
        }
        // 3)
        if !propagate && result.previous_context.is_none() {
            result.previous_context = Some(Box::new(active_context.clone()));
        }
        // 4)
        let local_context = match local_context {
            JsonValue::Array(items) => items,
            other => vec![other],
        };
        // 5)
        for context in local_context {
            let mut context = match context {
                // 5.1)
                JsonValue::Null => {
                    if !override_protected {
                        for (name, def) in &result.term_definitions {
                            if def.protected {
                                errors.push(JsonLdSyntaxError::msg(
                                    format!(
                                        "Definition of {name} would be overridden by a null context even though it is protected"
                                    ),
                                    JsonLdErrorCode::InvalidContextNullification,
                                ));
                            }
                        }
                    }
                    let mut new_result =
                        Context::new_empty(active_context.original_base_url.clone());
                    if !propagate {
                        new_result.previous_context = Some(Box::new(result));
                    }
                    result = new_result;
                    continue;
                }
                // 5.2)
                JsonValue::String(context) => {
                    let context = match self.resolve_iri(
                        &context,
                        base_url,
                        JsonLdErrorCode::LoadingDocumentFailed,
                    ) {
                        Ok(url) => url,
                        Err(e) => {
                            errors.push(e);
                            continue;
                        }
                    };
                    if !validate_scoped_context && remote_contexts.contains(&context) {
                        continue;
                    }
                    if remote_contexts.len() >= self.max_context_recursion {
                        errors.push(JsonLdSyntaxError::msg(
                            format!(
                                "This processor only allows {} nested remote contexts",
                                self.max_context_recursion
                            ),
                            JsonLdErrorCode::ContextOverflow,
                        ));
                        continue;
                    }
                    remote_contexts.push(context.clone());
                    let (loaded_base, loaded_content) = match self.load_remote_context(&context) {
                        Ok(r) => r,
                        Err(e) => {
                            errors.push(e);
                            remote_contexts.pop();
                            continue;
                        }
                    };
                    result = self.process_context(
                        &result,
                        loaded_content,
                        loaded_base.as_deref(),
                        remote_contexts,
                        false,
                        true,
                        validate_scoped_context,
                        errors,
                    );
                    remote_contexts.pop();
                    continue;
                }
                // 5.3)
                JsonValue::Array(_) | JsonValue::Number(_) | JsonValue::Bool(_) => {
                    errors.push(JsonLdSyntaxError::msg(
                        "@context value must be null, a string or an object",
                        JsonLdErrorCode::InvalidLocalContext,
                    ));
                    continue;
                }
                // 5.4)
                JsonValue::Object(context) => context,
            };
            // 5.5)
            if let Some(value) = context.remove("@version") {
                if let JsonValue::Number(version) = &value {
                    if version != "1.1" {
                        errors.push(JsonLdSyntaxError::msg(
                            format!("The only supported @version value is 1.1, found {version}"),
                            JsonLdErrorCode::InvalidVersionValue,
                        ));
                    }
                } else {
                    errors.push(JsonLdSyntaxError::msg(
                        "@version value must be a number",
                        JsonLdErrorCode::InvalidVersionValue,
                    ));
                }
                if self.processing_mode == JsonLdProcessingMode::JsonLd1_0 {
                    errors.push(JsonLdSyntaxError::msg(
                        "@version is only supported in JSON-LD 1.1",
                        JsonLdErrorCode::ProcessingModeConflict,
                    ));
                }
            }
            // 5.6)
            if let Some(value) = context.remove("@import") {
                if self.processing_mode == JsonLdProcessingMode::JsonLd1_0 {
                    errors.push(JsonLdSyntaxError::msg(
                        "@import is only supported in JSON-LD 1.1",
                        JsonLdErrorCode::InvalidContextEntry,
                    ));
                }
                let JsonValue::String(import) = value else {
                    errors.push(JsonLdSyntaxError::msg(
                        "@import must be a string",
                        JsonLdErrorCode::InvalidImportValue,
                    ));
                    continue;
                };
                let import = match self.resolve_iri(
                    &import,
                    base_url,
                    JsonLdErrorCode::InvalidImportValue,
                ) {
                    Ok(import) => import,
                    Err(e) => {
                        errors.push(e);
                        continue;
                    }
                };
                let (_, loaded_content) = match self.load_remote_context(&import) {
                    Ok(r) => r,
                    Err(e) => {
                        errors.push(e);
                        continue;
                    }
                };
                let JsonValue::Object(loaded_content) = loaded_content else {
                    errors.push(JsonLdSyntaxError::msg(
                        format!("Imported context {import} must be an object"),
                        JsonLdErrorCode::InvalidRemoteContext,
                    ));
                    continue;
                };
                if loaded_content.contains_key("@import") {
                    errors.push(JsonLdSyntaxError::msg(
                        format!("Imported context {import} must not contain an @import key"),
                        JsonLdErrorCode::InvalidContextEntry,
                    ));
                    continue;
                }
                for (key, value) in loaded_content.into_entries() {
                    if !context.contains_key(&key) {
                        context.insert(key, value);
                    }
                }
            }
            // 5.7)
            if let Some(value) = context.remove("@base") {
                if remote_contexts.is_empty() {
                    match value {
                        JsonValue::Null => result.base_iri = None,
                        JsonValue::String(value) => {
                            // An absolute `@base` is adopted verbatim: resolving it against the
                            // current base would silently normalise away dot segments the document
                            // asked for. A relative one is resolved; a base that cannot be
                            // resolved at all is kept as written, leaving relative references
                            // unresolved rather than raising an error.
                            result.base_iri = Some(if Iri::parse(value.as_str()).is_ok() {
                                value
                            } else {
                                self.resolve_iri(
                                    &value,
                                    result.base_iri.as_deref(),
                                    JsonLdErrorCode::InvalidBaseIri,
                                )
                                .unwrap_or(value)
                            });
                        }
                        _ => errors.push(JsonLdSyntaxError::msg(
                            "@base value must be a string",
                            JsonLdErrorCode::InvalidBaseIri,
                        )),
                    }
                }
            }
            // 5.8)
            if let Some(value) = context.remove("@vocab") {
                match value {
                    JsonValue::Null => result.vocabulary_mapping = None,
                    JsonValue::String(value) => {
                        match self.expand_iri(
                            &mut result,
                            value.clone(),
                            true,
                            true,
                            None,
                            &mut HashMap::new(),
                            errors,
                        ) {
                            Some(vocab) if !has_keyword_form(&vocab) || is_keyword(&vocab) => {
                                result.vocabulary_mapping = Some(vocab);
                            }
                            _ => errors.push(JsonLdSyntaxError::msg(
                                format!("Invalid @vocab '{value}'"),
                                JsonLdErrorCode::InvalidVocabMapping,
                            )),
                        }
                    }
                    _ => errors.push(JsonLdSyntaxError::msg(
                        "@vocab value must be a string",
                        JsonLdErrorCode::InvalidVocabMapping,
                    )),
                }
            }
            // 5.9)
            if let Some(value) = context.remove("@language") {
                match value {
                    JsonValue::Null => result.default_language = None,
                    JsonValue::String(value) => result.default_language = Some(value),
                    _ => errors.push(JsonLdSyntaxError::msg(
                        "@language value must be a string or null",
                        JsonLdErrorCode::InvalidDefaultLanguage,
                    )),
                }
            }
            // 5.10)
            if let Some(value) = context.remove("@direction") {
                if self.processing_mode == JsonLdProcessingMode::JsonLd1_0 {
                    errors.push(JsonLdSyntaxError::msg(
                        "@direction is only supported in JSON-LD 1.1",
                        JsonLdErrorCode::InvalidContextEntry,
                    ));
                }
                match value {
                    JsonValue::Null => result.default_direction = None,
                    JsonValue::String(value) => match Direction::parse(&value) {
                        Some(dir) => result.default_direction = Some(dir),
                        None => errors.push(JsonLdSyntaxError::msg(
                            format!("@direction value must be 'ltr' or 'rtl', found '{value}'"),
                            JsonLdErrorCode::InvalidBaseDirection,
                        )),
                    },
                    _ => errors.push(JsonLdSyntaxError::msg(
                        "@direction value must be a string or null",
                        JsonLdErrorCode::InvalidBaseDirection,
                    )),
                }
            }
            // 5.11)
            if let Some(value) = context.remove("@propagate") {
                if self.processing_mode == JsonLdProcessingMode::JsonLd1_0 {
                    errors.push(JsonLdSyntaxError::msg(
                        "@propagate is only supported in JSON-LD 1.1",
                        JsonLdErrorCode::InvalidContextEntry,
                    ));
                }
                if !matches!(value, JsonValue::Bool(_)) {
                    errors.push(JsonLdSyntaxError::msg(
                        "@propagate value must be a boolean",
                        JsonLdErrorCode::InvalidPropagateValue,
                    ));
                    continue;
                }
            }
            // 5.13)
            let mut protected = false;
            if let Some(value) = context.remove("@protected") {
                if self.processing_mode == JsonLdProcessingMode::JsonLd1_0 {
                    errors.push(JsonLdSyntaxError::msg(
                        "@protected is only supported in JSON-LD 1.1",
                        JsonLdErrorCode::InvalidContextEntry,
                    ));
                }
                if let JsonValue::Bool(value) = value {
                    protected = value;
                } else {
                    errors.push(JsonLdSyntaxError::msg(
                        "@protected value must be a boolean",
                        JsonLdErrorCode::InvalidProtectedValue,
                    ));
                }
            }
            // 5.12 / 5.13)
            let mut defined = HashMap::new();
            let terms: Vec<String> = context.keys().map(str::to_owned).collect();
            for term in terms {
                self.create_term_definition(
                    &mut result,
                    &context,
                    term,
                    &mut defined,
                    base_url,
                    protected,
                    override_protected,
                    remote_contexts,
                    errors,
                );
            }
        }
        // 6)
        result
    }

    /// [Create Term Definition](https://www.w3.org/TR/json-ld-api/#create-term-definition).
    #[allow(clippy::too_many_arguments, clippy::too_many_lines)]
    fn create_term_definition(
        &self,
        active_context: &mut Context,
        local_context: &JsonObject,
        term: String,
        defined: &mut HashMap<String, bool>,
        base_url: Option<&str>,
        protected: bool,
        override_protected: bool,
        remote_contexts: &mut Vec<String>,
        errors: &mut Vec<JsonLdSyntaxError>,
    ) {
        // 1)
        if let Some(defined_value) = defined.get(&term) {
            if !defined_value {
                errors.push(JsonLdSyntaxError::msg(
                    format!("Cyclic IRI mapping for term {term}"),
                    JsonLdErrorCode::CyclicIriMapping,
                ));
            }
            return;
        }
        // 2)
        if term.is_empty() {
            errors.push(JsonLdSyntaxError::msg(
                "@context terms must not be the empty string",
                JsonLdErrorCode::InvalidTermDefinition,
            ));
            return;
        }
        defined.insert(term.clone(), false);
        // 3)
        let Some(value) = local_context.get(&term) else {
            return;
        };
        // 4)
        if term == "@type" {
            if self.processing_mode == JsonLdProcessingMode::JsonLd1_0 {
                errors.push(JsonLdSyntaxError::msg(
                    "@type keyword can't be redefined in a JSON-LD 1.0 @context",
                    JsonLdErrorCode::KeywordRedefinition,
                ));
            }
            if let JsonValue::Object(value) = value {
                if value.is_empty() {
                    errors.push(JsonLdSyntaxError::msg(
                        "@type keyword definition can't be empty",
                        JsonLdErrorCode::KeywordRedefinition,
                    ));
                    return;
                }
                for (key, key_value) in value.iter() {
                    match key {
                        "@protected" => {}
                        "@container" => {
                            let ok = match key_value {
                                JsonValue::String(s) => s == "@set",
                                JsonValue::Array(items) => items
                                    .iter()
                                    .all(|v| matches!(v, JsonValue::String(s) if s == "@set")),
                                _ => false,
                            };
                            if !ok {
                                errors.push(JsonLdSyntaxError::msg(
                                    "@type definition only allows @set as @container",
                                    JsonLdErrorCode::KeywordRedefinition,
                                ));
                                return;
                            }
                        }
                        _ => {
                            errors.push(JsonLdSyntaxError::msg(
                                format!(
                                    "@type definition can only contain @protected and @container, found {key}"
                                ),
                                JsonLdErrorCode::KeywordRedefinition,
                            ));
                            return;
                        }
                    }
                }
            } else {
                errors.push(JsonLdSyntaxError::msg(
                    "@type definition must be an object",
                    JsonLdErrorCode::KeywordRedefinition,
                ));
                return;
            }
        } else if has_keyword_form(&term) {
            // 5)
            if is_keyword(&term) {
                errors.push(JsonLdSyntaxError::msg(
                    format!("{term} keyword can't be redefined in a context"),
                    JsonLdErrorCode::KeywordRedefinition,
                ));
            }
            return;
        }
        // 6)
        let previous_definition = active_context.term_definitions.remove(&term);
        // 7-9) `simple term` records whether the definition was written as a bare string; only
        // those may later become prefixes.
        let mut simple_term = true;
        let value = match value {
            JsonValue::Null => {
                let mut object = JsonObject::new();
                object.insert("@id", JsonValue::Null);
                object
            }
            JsonValue::String(id) => {
                let mut object = JsonObject::new();
                object.insert("@id", JsonValue::String(id.clone()));
                object
            }
            JsonValue::Object(map) => {
                simple_term = false;
                map.clone()
            }
            _ => {
                errors.push(JsonLdSyntaxError::msg(
                    "Term definition value must be null, a string or a map",
                    JsonLdErrorCode::InvalidTermDefinition,
                ));
                return;
            }
        };
        // 10)
        let mut definition = TermDefinition {
            protected,
            ..TermDefinition::default()
        };
        // 11)
        if let Some(key_value) = value.get("@protected") {
            if self.processing_mode == JsonLdProcessingMode::JsonLd1_0 {
                errors.push(JsonLdSyntaxError::msg(
                    "@protected can't be used in a JSON-LD 1.0 @context",
                    JsonLdErrorCode::InvalidTermDefinition,
                ));
            }
            let JsonValue::Bool(key_value) = key_value else {
                errors.push(JsonLdSyntaxError::msg(
                    "@protected value must be a boolean",
                    JsonLdErrorCode::InvalidProtectedValue,
                ));
                return;
            };
            definition.protected = *key_value;
        }
        // 12)
        if let Some(key_value) = value.get("@type") {
            let JsonValue::String(type_value) = key_value else {
                errors.push(JsonLdSyntaxError::msg(
                    "The value of @type in a context must be a string",
                    JsonLdErrorCode::InvalidTypeMapping,
                ));
                return;
            };
            let Some(type_iri) = self.expand_iri(
                active_context,
                type_value.clone(),
                false,
                true,
                Some(local_context),
                defined,
                errors,
            ) else {
                errors.push(JsonLdSyntaxError::msg(
                    format!("Invalid @type value in context: {type_value}"),
                    JsonLdErrorCode::InvalidTypeMapping,
                ));
                return;
            };
            if matches!(type_iri.as_str(), "@json" | "@none")
                && self.processing_mode == JsonLdProcessingMode::JsonLd1_0
            {
                errors.push(JsonLdSyntaxError::msg(
                    format!("@type value {type_iri} is only supported in JSON-LD 1.1"),
                    JsonLdErrorCode::InvalidTypeMapping,
                ));
            }
            let is_keyword_type = has_keyword_form(&type_iri);
            if (is_keyword_type
                && !matches!(type_iri.as_str(), "@id" | "@json" | "@none" | "@vocab"))
                || type_iri.starts_with("_:")
            {
                errors.push(JsonLdSyntaxError::msg(
                    format!("Invalid @type value in context: {type_iri}"),
                    JsonLdErrorCode::InvalidTypeMapping,
                ));
            }
            if !self.lenient && !is_keyword_type {
                if let Err(e) = Iri::parse(type_iri.as_str()) {
                    errors.push(JsonLdSyntaxError::msg(
                        format!("Invalid @type IRI '{type_iri}': {e}"),
                        JsonLdErrorCode::InvalidTypeMapping,
                    ));
                }
            }
            definition.type_mapping = Some(type_iri);
        }
        // 13)
        if let Some(key_value) = value.get("@reverse") {
            if value.contains_key("@id") {
                errors.push(JsonLdSyntaxError::msg(
                    "@reverse and @id can't be used together in a context",
                    JsonLdErrorCode::InvalidReverseProperty,
                ));
                return;
            }
            if value.contains_key("@nest") {
                errors.push(JsonLdSyntaxError::msg(
                    "@reverse and @nest can't be used together in a context",
                    JsonLdErrorCode::InvalidReverseProperty,
                ));
                return;
            }
            let JsonValue::String(reverse) = key_value else {
                errors.push(JsonLdSyntaxError::msg(
                    "@reverse value must be a string in a context",
                    JsonLdErrorCode::InvalidIriMapping,
                ));
                return;
            };
            if has_keyword_form(reverse) {
                return;
            }
            // 13.5) the container mapping of a reverse property may only be @index or @set
            match self.expand_iri(
                active_context,
                reverse.clone(),
                false,
                true,
                Some(local_context),
                defined,
                errors,
            ) {
                Some(iri) if iri.starts_with("_:") || self.iri_ok(&iri) => {
                    definition.iri_mapping = Some(Some(iri));
                }
                other => {
                    if let Some(iri) = other {
                        errors.push(JsonLdSyntaxError::msg(
                            format!("{iri} is not a valid IRI or blank node"),
                            JsonLdErrorCode::InvalidIriMapping,
                        ));
                    }
                    definition.iri_mapping = Some(None);
                }
            }
            if let Some(container_entry) = value.get("@container") {
                match container_entry {
                    JsonValue::Null => {}
                    JsonValue::String(container) => {
                        if matches!(container.as_str(), "@index" | "@set") {
                            definition.container_mapping = vec![container.clone()];
                        } else {
                            errors.push(JsonLdSyntaxError::msg(
                                "@reverse is only compatible with @index and @set containers",
                                JsonLdErrorCode::InvalidReverseProperty,
                            ));
                        }
                    }
                    _ => errors.push(JsonLdSyntaxError::msg(
                        "@container value must be a string or null",
                        JsonLdErrorCode::InvalidReverseProperty,
                    )),
                }
            }
            // 20) an @index mapping is honoured for reverse properties too
            if let Some(JsonValue::String(index)) = value.get("@index") {
                definition.index_mapping = Some(index.clone());
            }
            definition.reverse_property = true;
            active_context
                .term_definitions
                .insert(term.clone(), definition);
            defined.insert(term, true);
            return;
        }
        // 14)
        let id_differs = value
            .get("@id")
            .map(|v| !matches!(v, JsonValue::String(s) if *s == term))
            .unwrap_or(false);
        if id_differs {
            match value.get("@id") {
                Some(JsonValue::Null) => definition.iri_mapping = Some(None),
                Some(JsonValue::String(id)) => {
                    // 14.2.2) a keyword-shaped @id that is not a keyword makes the whole term
                    // definition be ignored.
                    if has_keyword_form(id) && !is_keyword(id) {
                        defined.insert(term, true);
                        return;
                    }
                    let Some(expanded) = self.expand_iri(
                        active_context,
                        id.clone(),
                        false,
                        true,
                        Some(local_context),
                        defined,
                        errors,
                    ) else {
                        definition.iri_mapping = Some(None);
                        active_context
                            .term_definitions
                            .insert(term.clone(), definition);
                        defined.insert(term, true);
                        return;
                    };
                    if expanded == "@context" {
                        errors.push(JsonLdSyntaxError::msg(
                            "@context can't be aliased with @id",
                            JsonLdErrorCode::InvalidKeywordAlias,
                        ));
                        return;
                    }
                    definition.iri_mapping = Some(Some(expanded));
                    // 14.2.4)
                    let inner_colon = term
                        .get(1..term.len().saturating_sub(1))
                        .is_some_and(|t| t.contains(':'));
                    if inner_colon || term.contains('/') {
                        defined.insert(term.clone(), true);
                        let expanded_term = self.expand_iri(
                            active_context,
                            term.clone(),
                            false,
                            true,
                            Some(local_context),
                            defined,
                            errors,
                        );
                        if expanded_term.as_deref() != definition.iri() {
                            errors.push(JsonLdSyntaxError::msg(
                                format!("Inconsistent expansion of {term}"),
                                JsonLdErrorCode::InvalidIriMapping,
                            ));
                        }
                    }
                    // 14.2.5)
                    if simple_term
                        && !term.contains(':')
                        && !term.contains('/')
                        && definition.iri().is_some_and(|iri| {
                            iri.ends_with([':', '/', '?', '#', '[', ']', '@'])
                                || iri.starts_with("_:")
                        })
                    {
                        definition.prefix_flag = true;
                    }
                }
                _ => {
                    definition.iri_mapping = Some(None);
                    errors.push(JsonLdSyntaxError::msg(
                        "@id value must be a string",
                        JsonLdErrorCode::InvalidIriMapping,
                    ));
                }
            }
        } else if let Some((prefix, suffix)) = split_after_first_char(&term) {
            // 15)
            if local_context.contains_key(prefix) {
                self.create_term_definition(
                    active_context,
                    local_context,
                    prefix.to_owned(),
                    defined,
                    base_url,
                    false,
                    false,
                    remote_contexts,
                    errors,
                );
            }
            if let Some(term_definition) = active_context.term_definitions.get(prefix) {
                if let Some(iri_mapping) = term_definition.iri() {
                    definition.iri_mapping = Some(Some(format!("{iri_mapping}{suffix}")));
                }
            } else {
                definition.iri_mapping = Some(Some(term.clone()));
            }
        } else if term.contains('/') {
            // 16) the term is a relative IRI reference, expanded like any other value
            // The expansion is done against the active context alone: consulting the local
            // context here would re-enter this very term and be reported as a cycle.
            let expanded = self.expand_iri(
                active_context,
                term.clone(),
                false,
                true,
                None,
                &mut HashMap::new(),
                errors,
            );
            match expanded {
                Some(iri) if self.iri_ok(&iri) => definition.iri_mapping = Some(Some(iri)),
                _ => {
                    errors.push(JsonLdSyntaxError::msg(
                        format!("Term {term} does not expand to an IRI"),
                        JsonLdErrorCode::InvalidIriMapping,
                    ));
                    return;
                }
            }
        } else if term == "@type" {
            // 17)
            definition.iri_mapping = Some(Some("@type".to_owned()));
        } else {
            // 18)
            if let Some(vocab) = &active_context.vocabulary_mapping {
                definition.iri_mapping = Some(Some(format!("{vocab}{term}")));
            } else {
                errors.push(JsonLdSyntaxError::msg(
                    format!("No @vocab mapping to build an IRI for term {term}"),
                    JsonLdErrorCode::InvalidIriMapping,
                ));
            }
        }
        // 19)
        if let Some(key_value) = value.get("@container") {
            if let Some(container_mapping) =
                self.parse_container(key_value, &term, &mut definition, errors)
            {
                definition.container_mapping = container_mapping;
            } else {
                return;
            }
        }
        // 20)
        if let Some(key_value) = value.get("@index") {
            if self.processing_mode == JsonLdProcessingMode::JsonLd1_0 {
                errors.push(JsonLdSyntaxError::msg(
                    "@index in term definitions is only supported in JSON-LD 1.1",
                    JsonLdErrorCode::InvalidTermDefinition,
                ));
            }
            if !definition.has_container("@index") {
                errors.push(JsonLdSyntaxError::msg(
                    "@index in a term definition requires an @index container",
                    JsonLdErrorCode::InvalidTermDefinition,
                ));
            }
            let JsonValue::String(index) = key_value else {
                errors.push(JsonLdSyntaxError::msg(
                    "@index value must be a string",
                    JsonLdErrorCode::InvalidTermDefinition,
                ));
                return;
            };
            let expanded_index = self.expand_iri(
                active_context,
                index.clone(),
                false,
                true,
                Some(local_context),
                defined,
                errors,
            );
            let valid = expanded_index
                .as_deref()
                .is_some_and(|i| self.iri_ok(i) && !has_keyword_form(i) && !i.starts_with("_:"));
            if !valid {
                errors.push(JsonLdSyntaxError::msg(
                    "@index value must be a valid IRI",
                    JsonLdErrorCode::InvalidTermDefinition,
                ));
                return;
            }
            definition.index_mapping = Some(index.clone());
        }
        // 21)
        if let Some(context) = value.get("@context") {
            if self.processing_mode == JsonLdProcessingMode::JsonLd1_0 {
                errors.push(JsonLdSyntaxError::msg(
                    "@context in term definitions is only supported in JSON-LD 1.1",
                    JsonLdErrorCode::InvalidTermDefinition,
                ));
            }
            let error_count = errors.len();
            self.process_context(
                active_context,
                context.clone(),
                base_url,
                remote_contexts,
                true,
                true,
                false,
                errors,
            );
            let scoped: Vec<_> = errors.drain(error_count..).collect();
            for error in scoped {
                errors.push(JsonLdSyntaxError::msg(
                    format!("Invalid scoped context: {error}"),
                    JsonLdErrorCode::InvalidScopedContext,
                ));
            }
            definition.context = Some(context.clone());
            definition.base_url = base_url.map(str::to_owned);
        }
        // 22)
        if let Some(key_value) = value.get("@language") {
            if value.contains_key("@type") {
                errors.push(JsonLdSyntaxError::msg(
                    "@language and @type can't be set at the same time",
                    JsonLdErrorCode::InvalidLanguageMapping,
                ));
            }
            definition.language_mapping = Some(match key_value {
                JsonValue::String(language) => Some(language.clone()),
                JsonValue::Null => None,
                _ => {
                    errors.push(JsonLdSyntaxError::msg(
                        "@language value must be a string or null",
                        JsonLdErrorCode::InvalidLanguageMapping,
                    ));
                    return;
                }
            });
        }
        // 23)
        if let Some(key_value) = value.get("@direction") {
            if self.processing_mode == JsonLdProcessingMode::JsonLd1_0 {
                errors.push(JsonLdSyntaxError::msg(
                    "@direction is only supported in JSON-LD 1.1",
                    JsonLdErrorCode::InvalidTermDefinition,
                ));
            }
            match key_value {
                JsonValue::Null => definition.direction_mapping = Some(None),
                JsonValue::String(value) => match Direction::parse(value) {
                    Some(dir) => definition.direction_mapping = Some(Some(dir)),
                    None => errors.push(JsonLdSyntaxError::msg(
                        format!("@direction value must be 'ltr' or 'rtl', found '{value}'"),
                        JsonLdErrorCode::InvalidBaseDirection,
                    )),
                },
                _ => errors.push(JsonLdSyntaxError::msg(
                    "@direction value must be a string or null",
                    JsonLdErrorCode::InvalidBaseDirection,
                )),
            }
        }
        // 24)
        if let Some(key_value) = value.get("@nest") {
            if self.processing_mode == JsonLdProcessingMode::JsonLd1_0 {
                errors.push(JsonLdSyntaxError::msg(
                    "@nest is only supported in JSON-LD 1.1",
                    JsonLdErrorCode::InvalidTermDefinition,
                ));
            }
            let JsonValue::String(nest) = key_value else {
                errors.push(JsonLdSyntaxError::msg(
                    "@nest value must be a string",
                    JsonLdErrorCode::InvalidNestValue,
                ));
                return;
            };
            if is_keyword(nest) && nest != "@nest" {
                errors.push(JsonLdSyntaxError::msg(
                    "@nest value must not be a keyword other than @nest",
                    JsonLdErrorCode::InvalidNestValue,
                ));
                return;
            }
            definition.nest_value = Some(nest.clone());
        }
        // 25)
        if let Some(key_value) = value.get("@prefix") {
            if self.processing_mode == JsonLdProcessingMode::JsonLd1_0 {
                errors.push(JsonLdSyntaxError::msg(
                    "@prefix is only supported in JSON-LD 1.1",
                    JsonLdErrorCode::InvalidTermDefinition,
                ));
            }
            if term.contains(':') || term.contains('/') {
                errors.push(JsonLdSyntaxError::msg(
                    format!("@prefix can't be set on term {term} containing ':' or '/'"),
                    JsonLdErrorCode::InvalidTermDefinition,
                ));
                return;
            }
            let JsonValue::Bool(prefix_flag) = key_value else {
                errors.push(JsonLdSyntaxError::msg(
                    "@prefix value must be a boolean",
                    JsonLdErrorCode::InvalidPrefixValue,
                ));
                return;
            };
            definition.prefix_flag = *prefix_flag;
            if definition.prefix_flag && definition.iri().is_some_and(is_keyword) {
                errors.push(JsonLdSyntaxError::msg(
                    format!("@prefix can't be set on term {term} that maps to a keyword"),
                    JsonLdErrorCode::InvalidTermDefinition,
                ));
                return;
            }
        }
        // 26)
        if let Some(key) = value.keys().find(|k| {
            !matches!(
                *k,
                "@id"
                    | "@reverse"
                    | "@container"
                    | "@context"
                    | "@direction"
                    | "@index"
                    | "@language"
                    | "@nest"
                    | "@prefix"
                    | "@protected"
                    | "@type"
            )
        }) {
            errors.push(JsonLdSyntaxError::msg(
                format!("Unexpected key in term definition: {key}"),
                JsonLdErrorCode::InvalidTermDefinition,
            ));
        }
        // 27)
        if !override_protected {
            if let Some(previous_definition) = previous_definition {
                if previous_definition.protected {
                    if definitions_differ(&definition, &previous_definition) {
                        errors.push(JsonLdSyntaxError::msg(
                            format!("Overriding the protected term {term}"),
                            JsonLdErrorCode::ProtectedTermRedefinition,
                        ));
                    }
                    definition = previous_definition;
                }
            }
        }
        // 28)
        active_context
            .term_definitions
            .insert(term.clone(), definition);
        defined.insert(term, true);
    }

    fn parse_container(
        &self,
        key_value: &JsonValue,
        _term: &str,
        definition: &mut TermDefinition,
        errors: &mut Vec<JsonLdSyntaxError>,
    ) -> Option<Vec<String>> {
        const ALLOWED: &[&[&str]] = &[
            &["@index"],
            &["@index", "@set"],
            &["@language"],
            &["@language", "@set"],
            &["@graph"],
            &["@graph", "@set"],
            &["@graph", "@id"],
            &["@graph", "@id", "@set"],
            &["@graph", "@index"],
            &["@graph", "@index", "@set"],
            &["@id"],
            &["@id", "@set"],
            &["@list"],
            &["@set"],
            &["@set", "@type"],
            &["@type"],
        ];
        let mut container_mapping = Vec::new();
        let items: Vec<&JsonValue> = match key_value {
            JsonValue::Array(items) => {
                if self.processing_mode == JsonLdProcessingMode::JsonLd1_0 {
                    errors.push(JsonLdSyntaxError::msg(
                        "@container with multiple values is not supported in JSON-LD 1.0",
                        JsonLdErrorCode::InvalidContainerMapping,
                    ));
                }
                items.iter().collect()
            }
            other => vec![other],
        };
        for item in items {
            if let JsonValue::String(container) = item {
                container_mapping.push(container.clone());
            } else {
                errors.push(JsonLdSyntaxError::msg(
                    "@container value must be a string or an array of strings",
                    JsonLdErrorCode::InvalidContainerMapping,
                ));
            }
        }
        let mut sorted = container_mapping.clone();
        sorted.sort_unstable();
        let matched = ALLOWED.iter().find(|c| {
            c.len() == sorted.len() && c.iter().zip(&sorted).all(|(a, b)| *a == b.as_str())
        });
        let Some(matched) = matched else {
            errors.push(JsonLdSyntaxError::msg(
                "Unsupported @container value combination",
                JsonLdErrorCode::InvalidContainerMapping,
            ));
            return None;
        };
        if self.processing_mode == JsonLdProcessingMode::JsonLd1_0 {
            if let Some(bad) = ["@graph", "@id", "@type"]
                .into_iter()
                .find(|k| matched.contains(k))
            {
                errors.push(JsonLdSyntaxError::msg(
                    format!("{bad} container is not supported in JSON-LD 1.0"),
                    JsonLdErrorCode::InvalidContainerMapping,
                ));
            }
        }
        let result: Vec<String> = matched.iter().map(|s| (*s).to_owned()).collect();
        // 19.4)
        if result.iter().any(|c| c == "@type") {
            if let Some(type_mapping) = &definition.type_mapping {
                if !matches!(type_mapping.as_str(), "@id" | "@vocab") {
                    errors.push(JsonLdSyntaxError::msg(
                        format!(
                            "Type mapping must be @id or @vocab, not {type_mapping} with @type container"
                        ),
                        JsonLdErrorCode::InvalidTypeMapping,
                    ));
                }
            } else {
                definition.type_mapping = Some("@id".to_owned());
            }
        }
        Some(result)
    }

    /// [IRI Expansion](https://www.w3.org/TR/json-ld-api/#iri-expansion) (term-definition flavour).
    #[allow(clippy::too_many_arguments)]
    pub fn expand_iri(
        &self,
        active_context: &mut Context,
        value: String,
        document_relative: bool,
        vocab: bool,
        local_context: Option<&JsonObject>,
        defined: &mut HashMap<String, bool>,
        errors: &mut Vec<JsonLdSyntaxError>,
    ) -> Option<String> {
        // 1)
        if has_keyword_form(&value) {
            return is_keyword(&value).then_some(value);
        }
        // 3)
        if let Some(local_context) = local_context {
            if local_context.contains_key(&value) && defined.get(&value) != Some(&true) {
                self.create_term_definition(
                    active_context,
                    local_context,
                    value.clone(),
                    defined,
                    None,
                    false,
                    false,
                    &mut Vec::new(),
                    errors,
                );
            }
        }
        // 4-5)
        if let Some(term_definition) = active_context.term_definitions.get(&value) {
            if let Some(Some(iri_mapping)) = &term_definition.iri_mapping {
                if is_keyword(iri_mapping) {
                    return Some(iri_mapping.clone());
                }
            }
            if vocab {
                return term_definition.iri_mapping.clone().flatten();
            }
        }
        // 6)
        if let Some((prefix, suffix)) = split_after_first_char(&value) {
            if prefix == "_" || suffix.starts_with("//") {
                return Some(value);
            }
            if let Some(local_context) = local_context {
                if local_context.contains_key(prefix) && defined.get(prefix) != Some(&true) {
                    self.create_term_definition(
                        active_context,
                        local_context,
                        prefix.to_owned(),
                        defined,
                        None,
                        false,
                        false,
                        &mut Vec::new(),
                        errors,
                    );
                }
            }
            if let Some(term_definition) = active_context.term_definitions.get(prefix) {
                if term_definition.prefix_flag {
                    if let Some(iri_mapping) = term_definition.iri() {
                        return Some(format!("{iri_mapping}{suffix}"));
                    }
                }
            }
            if self.iri_ok(&value) {
                return Some(value);
            }
        }
        // 7)
        if vocab {
            if let Some(vocab_mapping) = &active_context.vocabulary_mapping {
                return Some(format!("{vocab_mapping}{value}"));
            }
        }
        // 8)
        if document_relative {
            if let Some(base_iri) = &active_context.base_iri {
                if let Ok(result) =
                    self.resolve_iri(&value, Some(base_iri), JsonLdErrorCode::InvalidBaseIri)
                {
                    return Some(result);
                }
            }
        }
        Some(value)
    }

    /// [IRI Expansion](https://www.w3.org/TR/json-ld-api/#iri-expansion) against an already-built
    /// active context (no term definitions are created).
    ///
    /// Returns `None` when the value must be dropped: a keyword-shaped string that is not a
    /// keyword, or a term whose IRI mapping is explicitly `null`.
    #[must_use]
    pub fn expand_iri_in_context(
        &self,
        active_context: &Context,
        value: &str,
        document_relative: bool,
        vocab: bool,
    ) -> Option<String> {
        // 1-2)
        if has_keyword_form(value) {
            return is_keyword(value).then(|| value.to_owned());
        }
        // 4-5)
        if let Some(term_definition) = active_context.term(value) {
            if let Some(Some(iri_mapping)) = &term_definition.iri_mapping {
                if is_keyword(iri_mapping) {
                    return Some(iri_mapping.clone());
                }
            }
            if vocab {
                return term_definition.iri_mapping.clone().flatten();
            }
        }
        // 6)
        if let Some((prefix, suffix)) = split_after_first_char(value) {
            if prefix == "_" || suffix.starts_with("//") {
                return Some(value.to_owned());
            }
            if let Some(term_definition) = active_context.term(prefix) {
                if term_definition.prefix_flag {
                    if let Some(iri_mapping) = term_definition.iri() {
                        return Some(format!("{iri_mapping}{suffix}"));
                    }
                }
            }
            if self.iri_ok(value) {
                return Some(value.to_owned());
            }
        }
        // 7)
        if vocab {
            if let Some(vocabulary_mapping) = &active_context.vocabulary_mapping {
                return Some(format!("{vocabulary_mapping}{value}"));
            }
        }
        // 8)
        if document_relative {
            if let Some(base_iri) = &active_context.base_iri {
                if let Ok(resolved) =
                    self.resolve_iri(value, Some(base_iri), JsonLdErrorCode::InvalidBaseIri)
                {
                    return Some(resolved);
                }
            }
        }
        Some(value.to_owned())
    }

    /// Resolves `iri` against `base_iri` (if any), applying the leniency policy.
    pub fn resolve_iri(
        &self,
        iri: &str,
        base_iri: Option<&str>,
        error_code: JsonLdErrorCode,
    ) -> Result<String, JsonLdSyntaxError> {
        // RFC 3986 §5.3: when the reference is empty or carries only a query and/or a fragment,
        // the target path is the *verbatim* base path — dot segments in the base must survive.
        if let Some(base) = base_iri {
            if let Some(resolved) = resolve_same_document(base, iri) {
                if self.lenient || Iri::parse(resolved.as_str()).is_ok() {
                    return Ok(resolved);
                }
            }
        }
        if self.lenient {
            return Ok(match base_iri {
                Some(base) => {
                    let base = IriRef::parse_unchecked(base);
                    base.resolve_unchecked(&IriRef::parse_unchecked(iri))
                        .into_inner()
                }
                None => iri.to_owned(),
            });
        }
        match base_iri {
            Some(base) => {
                let base = Iri::parse(base).map_err(|e| {
                    JsonLdSyntaxError::msg(format!("Invalid base IRI '{base}': {e}"), error_code)
                })?;
                let reference = IriRef::parse(iri).map_err(|e| {
                    JsonLdSyntaxError::msg(format!("Invalid IRI '{iri}': {e}"), error_code)
                })?;
                base.resolve(&reference)
                    .map(|resolved| resolved.into_inner())
                    .map_err(|e| {
                        JsonLdSyntaxError::msg(format!("Invalid IRI '{iri}': {e}"), error_code)
                    })
            }
            None => Iri::parse(iri.to_owned())
                .map(|resolved| resolved.into_inner())
                .map_err(|e| {
                    JsonLdSyntaxError::msg(format!("Invalid IRI '{iri}': {e}"), error_code)
                }),
        }
    }

    fn iri_ok(&self, iri: &str) -> bool {
        self.lenient || Iri::parse(iri).is_ok()
    }

    /// Whether `iri` is an absolute IRI (always true when the processor runs in lenient mode).
    #[must_use]
    pub fn iri_is_valid(&self, iri: &str) -> bool {
        self.iri_ok(iri)
    }

    fn load_remote_context(&self, url: &str) -> Result<RemoteContextEntry, JsonLdSyntaxError> {
        if let Some(cached) = self.remote_context_cache.borrow().get(url) {
            return Ok(cached.clone());
        }
        let document = self
            .loader
            .load(
                url,
                &LoadDocumentOptions {
                    request_profile: JsonLdProfile::Context.into(),
                },
            )
            .map_err(|e| {
                JsonLdSyntaxError::msg(
                    format!("Failed to load remote context {url}: {e}"),
                    JsonLdErrorCode::LoadingRemoteContextFailed,
                )
            })?;
        let parsed = JsonValue::from_slice(&document.document).map_err(|e| {
            JsonLdSyntaxError::msg(
                format!("Remote context {url} is not valid JSON: {e}"),
                JsonLdErrorCode::LoadingRemoteContextFailed,
            )
        })?;
        let JsonValue::Object(object) = parsed else {
            return Err(JsonLdSyntaxError::msg(
                format!("Remote context {url} must be a map"),
                JsonLdErrorCode::InvalidRemoteContext,
            ));
        };
        let Some(loaded_context) = object
            .into_entries()
            .into_iter()
            .find_map(|(k, v)| (k == "@context").then_some(v))
        else {
            return Err(JsonLdSyntaxError::msg(
                format!("Remote context {url} must contain a @context key"),
                JsonLdErrorCode::InvalidRemoteContext,
            ));
        };
        let document_url = Iri::parse(document.document_url)
            .ok()
            .map(|resolved| resolved.into_inner());
        let entry = (document_url, loaded_context);
        self.remote_context_cache
            .borrow_mut()
            .insert(url.to_owned(), entry.clone());
        Ok(entry)
    }
}

/// Resolves a same-document reference (empty, `#fragment`, or `?query`) against `base` by string
/// surgery, as RFC 3986 §5.3 requires: the base path is copied verbatim.
///
/// Returns `None` when `reference` is not a same-document reference.
fn resolve_same_document(base: &str, reference: &str) -> Option<String> {
    let base_without_fragment = base.split('#').next().unwrap_or(base);
    if reference.is_empty() {
        return Some(base_without_fragment.to_owned());
    }
    if let Some(fragment) = reference.strip_prefix('#') {
        return Some(format!("{base_without_fragment}#{fragment}"));
    }
    if reference.starts_with('?') {
        let base_without_query = base_without_fragment
            .split('?')
            .next()
            .unwrap_or(base_without_fragment);
        return Some(format!("{base_without_query}{reference}"));
    }
    None
}

/// Splits `value` at the first colon that occurs *after* the first character, as the JSON-LD
/// compact-IRI production requires (`:suffix` is therefore not a compact IRI).
#[must_use]
pub fn split_after_first_char(value: &str) -> Option<(&str, &str)> {
    let start = value.char_indices().nth(1)?.0;
    let colon = value.get(start..)?.find(':')? + start;
    Some((value.get(..colon)?, value.get(colon + 1..)?))
}

fn definitions_differ(a: &TermDefinition, b: &TermDefinition) -> bool {
    a.iri_mapping != b.iri_mapping
        || a.prefix_flag != b.prefix_flag
        || a.reverse_property != b.reverse_property
        || a.base_url != b.base_url
        || a.context != b.context
        || a.container_mapping != b.container_mapping
        || a.direction_mapping != b.direction_mapping
        || a.index_mapping != b.index_mapping
        || a.language_mapping != b.language_mapping
        || a.nest_value != b.nest_value
        || a.type_mapping != b.type_mapping
}

/// Whether `value` has the lexical form of a keyword: `@` followed by one or more ASCII letters.
#[must_use]
pub fn has_keyword_form(value: &str) -> bool {
    value
        .strip_prefix('@')
        .is_some_and(|suffix| !suffix.is_empty() && suffix.bytes().all(|b| b.is_ascii_alphabetic()))
}

/// Whether `value` is one of the JSON-LD keywords.
#[must_use]
pub fn is_keyword(value: &str) -> bool {
    matches!(
        value,
        "@base"
            | "@container"
            | "@context"
            | "@direction"
            | "@graph"
            | "@id"
            | "@import"
            | "@included"
            | "@index"
            | "@json"
            | "@language"
            | "@list"
            | "@nest"
            | "@none"
            | "@prefix"
            | "@propagate"
            | "@protected"
            | "@reverse"
            | "@set"
            | "@type"
            | "@value"
            | "@version"
            | "@vocab"
    )
}
