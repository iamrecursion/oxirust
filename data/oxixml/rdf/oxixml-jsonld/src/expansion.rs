//! The [Expansion Algorithm](https://www.w3.org/TR/json-ld11-api/#expansion-algorithms).
//!
//! Expansion is the JSON-LD normal form every other algorithm builds on: it removes the context,
//! turns every property into an absolute IRI or a keyword, and represents every value uniformly.
//! This module implements the specification's *Expansion Algorithm* and *Value Expansion
//! Algorithm* over the crate's [`JsonValue`] tree; step numbers in the comments refer to the
//! [JSON-LD 1.1 API](https://www.w3.org/TR/json-ld11-api/#expansion-algorithm) recommendation.
//!
//! Errors are accumulated in [`Expander::errors`] rather than short-circuiting, so a document can
//! report the *first* specification error it violates while the algorithm still terminates.

use crate::context::{Context, ContextProcessor, Direction, is_keyword};
use crate::error::{JsonLdErrorCode, JsonLdSyntaxError};
use crate::profile::JsonLdProcessingMode;
use crate::value::{DEFAULT_MAX_NESTING_DEPTH, JsonObject, JsonValue};

/// Runs the expansion algorithm using a shared [`ContextProcessor`].
pub struct Expander<'a> {
    processor: &'a ContextProcessor<'a>,
    ordered: bool,
    streaming: bool,
    /// The deepest JSON nesting [`expand`](Self::expand) will descend into before erroring out.
    max_depth: usize,
    /// The nesting depth of the [`expand`](Self::expand) call currently in progress.
    depth: usize,
    /// Accumulated specification errors; a non-empty list makes the top-level operation fail.
    pub errors: Vec<JsonLdSyntaxError>,
}

/// The active property of an expansion step: absent (the document root), `@graph`, or a term.
type ActiveProperty<'p> = Option<&'p str>;

impl<'a> Expander<'a> {
    /// Creates a new expander.
    #[must_use]
    pub fn new(processor: &'a ContextProcessor<'a>, ordered: bool) -> Self {
        Self {
            processor,
            ordered,
            streaming: false,
            max_depth: DEFAULT_MAX_NESTING_DEPTH,
            depth: 0,
            errors: Vec::new(),
        }
    }

    /// Sets the maximum JSON nesting depth expansion will descend into.
    ///
    /// The expansion algorithm is specified recursively and this implementation follows it, so
    /// one native stack frame is consumed per nesting level: without a bound, a modestly sized
    /// untrusted document aborts the process with a stack overflow instead of reporting an error.
    #[must_use]
    pub fn with_max_depth(mut self, max_depth: usize) -> Self {
        self.max_depth = max_depth;
        self
    }

    /// Enables the [streaming profile](https://www.w3.org/TR/json-ld11-streaming/)'s key-order
    /// validation.
    #[must_use]
    pub fn with_streaming_profile(mut self, streaming: bool) -> Self {
        self.streaming = streaming;
        self
    }

    /// Rejects a node object whose entries are ordered in a way a streaming processor could not
    /// honour (see [`crate::streaming`]).
    fn validate_streaming_order(&mut self, active_context: &Context, keys: &[String]) {
        // Value objects impose no order; the check applies to node and graph objects.
        if keys
            .iter()
            .any(|key| self.key_is(active_context, key, "@value"))
        {
            return;
        }
        let mut highest = 0;
        for key in keys {
            let expanded = self.iri(active_context, key, false, true);
            let current = crate::streaming::key_rank(key, expanded.as_deref());
            if current < highest {
                self.errors.push(crate::streaming::key_order_error(key));
                return;
            }
            highest = highest.max(current);
        }
    }

    /// Runs the [Expansion API operation](https://www.w3.org/TR/json-ld11-api/#dom-jsonldprocessor-expand):
    /// expands `element` and normalizes the top-level result to an array.
    pub fn expand_document(
        &mut self,
        active_context: &Context,
        element: JsonValue,
        base_url: Option<&str>,
    ) -> JsonValue {
        let expanded = self.expand(active_context, None, element, base_url, false);
        // If the result is a map containing only `@graph`, set it to the value of `@graph`.
        let expanded = match expanded {
            Some(JsonValue::Object(object))
                if object.len() == 1 && object.contains_key("@graph") =>
            {
                object.get("@graph").cloned()
            }
            other => other,
        };
        match expanded {
            None | Some(JsonValue::Null) => JsonValue::Array(Vec::new()),
            Some(JsonValue::Array(items)) => JsonValue::Array(items),
            Some(other) => JsonValue::Array(vec![other]),
        }
    }

    fn error(&mut self, message: impl Into<String>, code: JsonLdErrorCode) {
        self.errors.push(JsonLdSyntaxError::msg(message, code));
    }

    fn is_json_ld_1_0(&self) -> bool {
        self.processor.processing_mode == JsonLdProcessingMode::JsonLd1_0
    }

    /// IRI expansion against an active context, without creating term definitions.
    fn iri(
        &self,
        active_context: &Context,
        value: &str,
        document_relative: bool,
        vocab: bool,
    ) -> Option<String> {
        self.processor
            .expand_iri_in_context(active_context, value, document_relative, vocab)
    }

    /// Whether `key` expands to `keyword` in `active_context`.
    fn key_is(&self, active_context: &Context, key: &str, keyword: &str) -> bool {
        self.iri(active_context, key, false, true).as_deref() == Some(keyword)
    }

    fn process_context(
        &mut self,
        active_context: &Context,
        local_context: JsonValue,
        base_url: Option<&str>,
        override_protected: bool,
        propagate: bool,
    ) -> Context {
        let mut errors = Vec::new();
        let result = self.processor.process_context(
            active_context,
            local_context,
            base_url,
            &mut Vec::new(),
            override_protected,
            propagate,
            true,
            &mut errors,
        );
        self.errors.append(&mut errors);
        result
    }

    /// The recursive [Expansion Algorithm](https://www.w3.org/TR/json-ld11-api/#expansion-algorithm).
    ///
    /// `None` stands for the specification's `null` return value.
    fn expand(
        &mut self,
        active_context: &Context,
        active_property: ActiveProperty<'_>,
        element: JsonValue,
        base_url: Option<&str>,
        from_map: bool,
    ) -> Option<JsonValue> {
        // 1)
        if matches!(element, JsonValue::Null) {
            return None;
        }
        // Not a specification step: bound the recursion this algorithm is defined in terms of.
        if self.depth >= self.max_depth {
            self.error(
                format!(
                    "the document nests more than {} arrays/objects deep",
                    self.max_depth
                ),
                JsonLdErrorCode::DocumentNestingTooDeep,
            );
            return None;
        }
        self.depth += 1;
        let expanded =
            self.expand_bounded(active_context, active_property, element, base_url, from_map);
        self.depth -= 1;
        expanded
    }

    /// The body of [`expand`](Self::expand), called only with the depth counter already
    /// incremented and checked.
    fn expand_bounded(
        &mut self,
        active_context: &Context,
        active_property: ActiveProperty<'_>,
        element: JsonValue,
        base_url: Option<&str>,
        from_map: bool,
    ) -> Option<JsonValue> {
        // 3) property-scoped context
        let property_scoped = active_property
            .and_then(|property| active_context.term(property))
            .and_then(|definition| {
                definition
                    .context
                    .clone()
                    .map(|context| (context, definition.base_url.clone()))
            });
        match element {
            // 4) scalar
            JsonValue::Bool(_) | JsonValue::Number(_) | JsonValue::String(_) => {
                // 4.1)
                let active_property = match active_property {
                    None | Some("@graph") => return None,
                    Some(property) => property,
                };
                // 4.2)
                let context = match property_scoped {
                    Some((context, scoped_base)) => self.process_context(
                        active_context,
                        context,
                        scoped_base.as_deref().or(base_url),
                        false,
                        true,
                    ),
                    None => active_context.clone(),
                };
                // 4.3)
                Some(self.expand_value(&context, active_property, element))
            }
            // 5) array
            JsonValue::Array(items) => {
                let is_list_container = active_property
                    .and_then(|property| active_context.term(property))
                    .is_some_and(|definition| definition.has_container("@list"));
                let mut result = Vec::new();
                for item in items {
                    let mut expanded =
                        self.expand(active_context, active_property, item, base_url, from_map);
                    // 5.2.2)
                    if is_list_container {
                        if let Some(JsonValue::Array(_)) = &expanded {
                            let mut object = JsonObject::new();
                            object.insert("@list", expanded.unwrap_or(JsonValue::Null));
                            expanded = Some(JsonValue::Object(object));
                        }
                    }
                    // 5.2.3)
                    match expanded {
                        None | Some(JsonValue::Null) => {}
                        Some(JsonValue::Array(inner)) => result.extend(inner),
                        Some(other) => result.push(other),
                    }
                }
                Some(JsonValue::Array(result))
            }
            // 6) map
            JsonValue::Object(object) => self.expand_object(
                active_context,
                active_property,
                object,
                base_url,
                from_map,
                property_scoped,
            ),
            JsonValue::Null => None,
        }
    }

    #[allow(clippy::too_many_lines)]
    fn expand_object(
        &mut self,
        active_context: &Context,
        active_property: ActiveProperty<'_>,
        mut element: JsonObject,
        base_url: Option<&str>,
        from_map: bool,
        property_scoped: Option<(JsonValue, Option<String>)>,
    ) -> Option<JsonValue> {
        let mut active_context = active_context.clone();
        let original_keys: Vec<String> = if self.streaming {
            element.keys().map(str::to_owned).collect()
        } else {
            Vec::new()
        };
        // 7) revert a non-propagated context unless the map is a value object or a bare node
        // reference.
        if active_context.previous_context.is_some() && !from_map {
            let has_value = element
                .keys()
                .any(|key| self.key_is(&active_context, key, "@value"));
            let single_id = element.len() == 1
                && element
                    .keys()
                    .next()
                    .is_some_and(|key| self.key_is(&active_context, key, "@id"));
            if !has_value && !single_id {
                if let Some(previous) = active_context.previous_context.clone() {
                    active_context = *previous;
                }
            }
        }
        // 8) property-scoped context
        if let Some((context, scoped_base)) = property_scoped {
            active_context = self.process_context(
                &active_context,
                context,
                scoped_base.as_deref().or(base_url),
                true,
                true,
            );
        }
        // 9) local @context
        if let Some(context) = element.remove("@context") {
            active_context = self.process_context(&active_context, context, base_url, false, true);
        }
        if self.streaming {
            self.validate_streaming_order(&active_context, &original_keys);
        }
        // 10)
        let type_scoped_context = active_context.clone();
        // 11) type-scoped contexts, keyed by every alias of @type, in lexicographic key order
        let mut type_keys: Vec<String> = element
            .keys()
            .filter(|key| self.key_is(&active_context, key, "@type"))
            .map(str::to_owned)
            .collect();
        type_keys.sort_unstable();
        for type_key in &type_keys {
            let mut terms = element
                .get(type_key)
                .map(to_string_array)
                .unwrap_or_default();
            terms.sort_unstable();
            for term in terms {
                let scoped = type_scoped_context.term(&term).and_then(|definition| {
                    definition
                        .context
                        .clone()
                        .map(|context| (context, definition.base_url.clone()))
                });
                if let Some((context, scoped_base)) = scoped {
                    active_context = self.process_context(
                        &active_context,
                        context,
                        scoped_base.as_deref().or(base_url),
                        false,
                        false,
                    );
                }
            }
        }
        // 12) input type: the last value of the first @type-expanding entry
        let input_type = type_keys.first().and_then(|type_key| {
            let value = element.get(type_key)?;
            let last = match value {
                JsonValue::Array(items) => items.iter().rev().find_map(JsonValue::as_str),
                JsonValue::String(s) => Some(s.as_str()),
                _ => None,
            }?;
            self.iri(&active_context, last, false, true)
        });

        let mut result = JsonObject::new();
        let mut nests: Vec<String> = Vec::new();
        // 13-14)
        self.expand_entries(
            &active_context,
            &type_scoped_context,
            active_property,
            &element,
            base_url,
            input_type.as_deref(),
            &mut result,
            &mut nests,
        );
        let mut visited_nests = Vec::new();
        self.expand_nests(
            &active_context,
            &type_scoped_context,
            active_property,
            &element,
            base_url,
            input_type.as_deref(),
            &mut result,
            &mut nests,
            &mut visited_nests,
        );

        self.finalize(active_property, result)
    }

    /// Step 13: expands every entry of `element` into `result`.
    #[allow(clippy::too_many_arguments, clippy::too_many_lines)]
    fn expand_entries(
        &mut self,
        active_context: &Context,
        type_scoped_context: &Context,
        active_property: ActiveProperty<'_>,
        element: &JsonObject,
        base_url: Option<&str>,
        input_type: Option<&str>,
        result: &mut JsonObject,
        nests: &mut Vec<String>,
    ) {
        let keys: Vec<String> = if self.ordered {
            element.sorted_keys()
        } else {
            element.keys().map(str::to_owned).collect()
        };
        for key in keys {
            // 13.1)
            if key == "@context" {
                continue;
            }
            // 13.2)
            let Some(expanded_property) = self.iri(active_context, &key, false, true) else {
                continue;
            };
            // 13.3)
            if !expanded_property.contains(':') && !is_keyword(&expanded_property) {
                continue;
            }
            let value = element.get(&key).cloned().unwrap_or(JsonValue::Null);
            // 13.4) keywords
            if is_keyword(&expanded_property) {
                self.expand_keyword_entry(
                    active_context,
                    type_scoped_context,
                    active_property,
                    &expanded_property,
                    value,
                    base_url,
                    input_type,
                    result,
                    nests,
                    &key,
                );
                continue;
            }
            // 13.5-13.14) properties
            self.expand_property_entry(
                active_context,
                &key,
                &expanded_property,
                value,
                base_url,
                result,
            );
        }
    }

    /// Step 14: `@nest` entries repeat steps 13 and 14 with the nested map as the element.
    #[allow(clippy::too_many_arguments)]
    fn expand_nests(
        &mut self,
        active_context: &Context,
        type_scoped_context: &Context,
        active_property: ActiveProperty<'_>,
        element: &JsonObject,
        base_url: Option<&str>,
        input_type: Option<&str>,
        result: &mut JsonObject,
        nests: &mut Vec<String>,
        visited: &mut Vec<String>,
    ) {
        let mut pending = std::mem::take(nests);
        if self.ordered {
            pending.sort_unstable();
        }
        for nesting_key in pending {
            if visited.contains(&nesting_key) {
                continue;
            }
            visited.push(nesting_key.clone());
            // A term that aliases `@nest` may carry a property-scoped context; it applies to the
            // nested map exactly as it would to the value of an ordinary property.
            let scoped = active_context.term(&nesting_key).and_then(|definition| {
                definition
                    .context
                    .clone()
                    .map(|context| (context, definition.base_url.clone()))
            });
            let nest_context = match scoped {
                Some((context, scoped_base)) => self.process_context(
                    active_context,
                    context,
                    scoped_base.as_deref().or(base_url),
                    true,
                    true,
                ),
                None => active_context.clone(),
            };
            let active_context = &nest_context;
            let nested_values = match element.get(&nesting_key) {
                Some(JsonValue::Array(items)) => items.clone(),
                Some(other) => vec![other.clone()],
                None => Vec::new(),
            };
            for nested_value in nested_values {
                // 14.2.1)
                let JsonValue::Object(nested_value) = nested_value else {
                    self.error(
                        "@nest value must be a node object",
                        JsonLdErrorCode::InvalidNestValue,
                    );
                    continue;
                };
                if nested_value
                    .keys()
                    .any(|key| self.key_is(active_context, key, "@value"))
                {
                    self.error(
                        "@nest value must not be a value object",
                        JsonLdErrorCode::InvalidNestValue,
                    );
                    continue;
                }
                // 14.2.2)
                let mut inner_nests = Vec::new();
                self.expand_entries(
                    active_context,
                    type_scoped_context,
                    active_property,
                    &nested_value,
                    base_url,
                    input_type,
                    result,
                    &mut inner_nests,
                );
                let mut inner_visited = Vec::new();
                self.expand_nests(
                    active_context,
                    type_scoped_context,
                    active_property,
                    &nested_value,
                    base_url,
                    input_type,
                    result,
                    &mut inner_nests,
                    &mut inner_visited,
                );
            }
        }
    }

    /// Step 13.4: expands a keyword entry.
    #[allow(clippy::too_many_arguments, clippy::too_many_lines)]
    fn expand_keyword_entry(
        &mut self,
        active_context: &Context,
        type_scoped_context: &Context,
        active_property: ActiveProperty<'_>,
        expanded_property: &str,
        value: JsonValue,
        base_url: Option<&str>,
        input_type: Option<&str>,
        result: &mut JsonObject,
        nests: &mut Vec<String>,
        key: &str,
    ) {
        // 13.4.1)
        if active_property == Some("@reverse") {
            self.error(
                "keywords are not allowed inside a @reverse map",
                JsonLdErrorCode::InvalidReversePropertyMap,
            );
            return;
        }
        // 13.4.2)
        if result.contains_key(expanded_property)
            && !matches!(expanded_property, "@included" | "@type")
        {
            self.error(
                format!("colliding keyword {expanded_property}"),
                JsonLdErrorCode::CollidingKeywords,
            );
            return;
        }
        let expanded_value: Option<JsonValue> = match expanded_property {
            // 13.4.3)
            "@id" => {
                let JsonValue::String(id) = &value else {
                    self.error(
                        "@id value must be a string",
                        JsonLdErrorCode::InvalidIdValue,
                    );
                    return;
                };
                // A keyword-shaped identifier expands to null; the entry is still recorded so
                // that the node object is recognisably identifier-less downstream.
                Some(
                    self.iri(active_context, id, true, false)
                        .map_or(JsonValue::Null, JsonValue::String),
                )
            }
            // 13.4.4)
            "@type" => {
                let types = match &value {
                    JsonValue::String(s) => vec![s.clone()],
                    JsonValue::Array(items) => {
                        if !items.iter().all(JsonValue::is_string) {
                            self.error(
                                "@type value must be a string or an array of strings",
                                JsonLdErrorCode::InvalidTypeValue,
                            );
                            return;
                        }
                        items
                            .iter()
                            .filter_map(|item| item.as_str().map(str::to_owned))
                            .collect()
                    }
                    JsonValue::Object(object) if object.is_empty() => {
                        // A frame's empty map is passed through unchanged.
                        result.insert("@type", value);
                        return;
                    }
                    _ => {
                        self.error(
                            "@type value must be a string or an array of strings",
                            JsonLdErrorCode::InvalidTypeValue,
                        );
                        return;
                    }
                };
                // 13.4.4.4) IRI expansion preserves the shape of `value`: a lone string stays a
                // string, which is what keeps `@type` scalar inside a value object.
                let mut expanded_types: Vec<JsonValue> = Vec::new();
                for term in &types {
                    if let Some(expanded) = self.iri(type_scoped_context, term, true, true) {
                        expanded_types.push(JsonValue::String(expanded));
                    }
                }
                let single = value.is_string() && expanded_types.len() == 1;
                // 13.4.4.5) prepend anything already collected from another @type alias
                match result.remove("@type") {
                    Some(existing) => {
                        let mut all = match existing {
                            JsonValue::Array(items) => items,
                            other => vec![other],
                        };
                        all.append(&mut expanded_types);
                        Some(JsonValue::Array(all))
                    }
                    None if single => expanded_types.into_iter().next(),
                    None => Some(JsonValue::Array(expanded_types)),
                }
            }
            // 13.4.5)
            "@graph" => {
                let expanded = self.expand(active_context, Some("@graph"), value, base_url, false);
                Some(ensure_array(expanded))
            }
            // 13.4.6)
            "@included" => {
                if self.is_json_ld_1_0() {
                    return;
                }
                let expanded = self.expand(active_context, active_property, value, base_url, false);
                // "ensuring that the result is an array": a dropped (null) value is kept as a
                // single null entry so that the node-object check below rejects it.
                let items = match expanded {
                    Some(JsonValue::Array(items)) => items,
                    Some(other) => vec![other],
                    None => vec![JsonValue::Null],
                };
                if !items.iter().all(is_node_object) {
                    self.error(
                        "@included value must be an array of node objects",
                        JsonLdErrorCode::InvalidIncludedValue,
                    );
                    return;
                }
                let mut all = Vec::new();
                if let Some(JsonValue::Array(existing)) = result.remove("@included") {
                    all.extend(existing);
                }
                all.extend(items);
                Some(JsonValue::Array(all))
            }
            // 13.4.7)
            "@value" => {
                if input_type == Some("@json") {
                    if self.is_json_ld_1_0() {
                        self.error(
                            "@json literals are only supported in JSON-LD 1.1",
                            JsonLdErrorCode::InvalidValueObjectValue,
                        );
                        return;
                    }
                    result.insert("@value", value);
                    return;
                }
                if !value.is_scalar() && !matches!(value, JsonValue::Null) {
                    self.error(
                        "@value value must be a scalar or null",
                        JsonLdErrorCode::InvalidValueObjectValue,
                    );
                    return;
                }
                // 13.4.7.4)
                result.insert("@value", value);
                return;
            }
            // 13.4.8)
            "@language" => {
                let JsonValue::String(language) = value else {
                    self.error(
                        "@language value must be a string",
                        JsonLdErrorCode::InvalidLanguageTaggedString,
                    );
                    return;
                };
                Some(JsonValue::String(language))
            }
            // 13.4.9)
            "@direction" => {
                if self.is_json_ld_1_0() {
                    return;
                }
                match &value {
                    JsonValue::String(s) if Direction::parse(s).is_some() => Some(value),
                    _ => {
                        self.error(
                            "@direction value must be 'ltr' or 'rtl'",
                            JsonLdErrorCode::InvalidBaseDirection,
                        );
                        return;
                    }
                }
            }
            // 13.4.10)
            "@index" => {
                if !value.is_string() {
                    self.error(
                        "@index value must be a string",
                        JsonLdErrorCode::InvalidIndexValue,
                    );
                    return;
                }
                Some(value)
            }
            // 13.4.11)
            "@list" => {
                if active_property.is_none() || active_property == Some("@graph") {
                    return;
                }
                let expanded = self.expand(active_context, active_property, value, base_url, false);
                Some(ensure_array(expanded))
            }
            // 13.4.12)
            "@set" => self.expand(active_context, active_property, value, base_url, false),
            // 13.4.13)
            "@reverse" => {
                if !value.is_object() {
                    self.error(
                        "@reverse value must be a map",
                        JsonLdErrorCode::InvalidReverseValue,
                    );
                    return;
                }
                let expanded =
                    self.expand(active_context, Some("@reverse"), value, base_url, false);
                if let Some(JsonValue::Object(expanded)) = expanded {
                    for (property, item) in expanded.into_entries() {
                        if property == "@reverse" {
                            // 13.4.13.3) doubly-reversed properties become forward properties
                            if let JsonValue::Object(inner) = item {
                                for (inner_property, inner_value) in inner.into_entries() {
                                    add_value(result, &inner_property, inner_value, true);
                                }
                            }
                            continue;
                        }
                        // 13.4.13.4)
                        let items = match item {
                            JsonValue::Array(items) => items,
                            other => vec![other],
                        };
                        for item in items {
                            if is_value_or_list_object(&item) {
                                self.error(
                                    "reverse property values must be node objects",
                                    JsonLdErrorCode::InvalidReversePropertyValue,
                                );
                                continue;
                            }
                            let reverse_map = reverse_map_mut(result);
                            add_value(reverse_map, &property, item, true);
                        }
                    }
                }
                return;
            }
            // 13.4.14)
            "@nest" => {
                if !nests.contains(&key.to_owned()) {
                    nests.push(key.to_owned());
                }
                return;
            }
            _ => return,
        };
        // 13.4.16)
        if let Some(expanded_value) = expanded_value {
            result.insert(expanded_property, expanded_value);
        }
    }

    /// Steps 13.5 to 13.14: expands a non-keyword property entry.
    #[allow(clippy::too_many_arguments, clippy::too_many_lines)]
    fn expand_property_entry(
        &mut self,
        active_context: &Context,
        key: &str,
        expanded_property: &str,
        value: JsonValue,
        base_url: Option<&str>,
        result: &mut JsonObject,
    ) {
        let definition = active_context.term(key).cloned().unwrap_or_default();
        let container = definition.container_mapping.clone();
        let has = |keyword: &str| container.iter().any(|c| c == keyword);

        // 13.6 / 13.7 / 13.8 / 13.9)
        let mut expanded_value = if definition.type_mapping.as_deref() == Some("@json") {
            let mut object = JsonObject::new();
            object.insert("@value", value);
            object.insert("@type", JsonValue::String("@json".to_owned()));
            Some(JsonValue::Object(object))
        } else if has("@language") && value.is_object() {
            Some(self.expand_language_map(active_context, &definition, value))
        } else if (has("@index") || has("@type") || has("@id")) && value.is_object() {
            Some(self.expand_index_map(
                active_context,
                key,
                &definition,
                &container,
                value,
                base_url,
            ))
        } else {
            self.expand(active_context, Some(key), value, base_url, false)
        };

        // 13.10)
        let Some(mut value) = expanded_value.take() else {
            return;
        };
        // 13.11) @list container
        if has("@list") && !is_list_object(&value) {
            let mut object = JsonObject::new();
            object.insert("@list", ensure_array(Some(value)));
            value = JsonValue::Object(object);
        }
        // 13.12) @graph container
        if has("@graph") && !has("@id") && !has("@index") {
            let items = match ensure_array(Some(value)) {
                JsonValue::Array(items) => items,
                other => vec![other],
            };
            // Every value becomes a graph object, even one that already is one: nesting a graph
            // inside a `@graph` container yields a graph of a graph.
            value = JsonValue::Array(
                items
                    .into_iter()
                    .map(|item| {
                        let mut object = JsonObject::new();
                        object.insert("@graph", JsonValue::Array(vec![item]));
                        JsonValue::Object(object)
                    })
                    .collect(),
            );
        }
        // 13.13) reverse property
        if definition.reverse_property {
            let items = match ensure_array(Some(value)) {
                JsonValue::Array(items) => items,
                other => vec![other],
            };
            for item in items {
                if is_value_or_list_object(&item) {
                    self.error(
                        "reverse property values must be node objects",
                        JsonLdErrorCode::InvalidReversePropertyValue,
                    );
                    continue;
                }
                let reverse_map = reverse_map_mut(result);
                add_value(reverse_map, expanded_property, item, true);
            }
            return;
        }
        // 13.14)
        add_value(result, expanded_property, value, true);
    }

    /// Step 13.7: expands a language map.
    fn expand_language_map(
        &mut self,
        active_context: &Context,
        definition: &crate::context::TermDefinition,
        value: JsonValue,
    ) -> JsonValue {
        let JsonValue::Object(map) = value else {
            return JsonValue::Array(Vec::new());
        };
        let direction = definition
            .direction_mapping
            .unwrap_or(active_context.default_direction);
        let mut entries = map.into_entries();
        if self.ordered {
            entries.sort_by(|a, b| a.0.cmp(&b.0));
        }
        let mut result = Vec::new();
        for (language, language_value) in entries {
            let items = match language_value {
                JsonValue::Array(items) => items,
                other => vec![other],
            };
            for item in items {
                if matches!(item, JsonValue::Null) {
                    continue;
                }
                let JsonValue::String(item) = item else {
                    self.error(
                        "language map values must be strings",
                        JsonLdErrorCode::InvalidLanguageMapValue,
                    );
                    continue;
                };
                let mut object = JsonObject::new();
                object.insert("@value", JsonValue::String(item));
                let is_none = language == "@none"
                    || self.iri(active_context, &language, false, true).as_deref() == Some("@none");
                if !is_none {
                    object.insert("@language", JsonValue::String(language.clone()));
                }
                if let Some(direction) = direction {
                    object.insert(
                        "@direction",
                        JsonValue::String(direction.as_str().to_owned()),
                    );
                }
                result.push(JsonValue::Object(object));
            }
        }
        JsonValue::Array(result)
    }

    /// Step 13.8: expands an index, id, or type map.
    #[allow(clippy::too_many_lines)]
    fn expand_index_map(
        &mut self,
        active_context: &Context,
        key: &str,
        definition: &crate::context::TermDefinition,
        container: &[String],
        value: JsonValue,
        base_url: Option<&str>,
    ) -> JsonValue {
        let JsonValue::Object(map) = value else {
            return JsonValue::Array(Vec::new());
        };
        let has = |keyword: &str| container.iter().any(|c| c == keyword);
        let index_key = definition
            .index_mapping
            .clone()
            .unwrap_or_else(|| "@index".to_owned());
        let mut entries = map.into_entries();
        if self.ordered {
            entries.sort_by(|a, b| a.0.cmp(&b.0));
        }
        let mut result = Vec::new();
        for (index, index_value) in entries {
            // 13.8.3.1-13.8.3.3)
            let mut map_context = if has("@id") || has("@type") {
                active_context
                    .previous_context
                    .as_deref()
                    .cloned()
                    .unwrap_or_else(|| active_context.clone())
            } else {
                active_context.clone()
            };
            if has("@type") {
                let scoped = map_context.term(&index).and_then(|td| {
                    td.context
                        .clone()
                        .map(|context| (context, td.base_url.clone()))
                });
                if let Some((context, scoped_base)) = scoped {
                    map_context = self.process_context(
                        &map_context,
                        context,
                        scoped_base.as_deref().or(base_url),
                        false,
                        true,
                    );
                } else {
                    map_context = active_context.clone();
                }
            } else if !has("@id") {
                map_context = active_context.clone();
            }
            // 13.8.3.4)
            let expanded_index = self
                .iri(active_context, &index, false, true)
                .unwrap_or_else(|| index.clone());
            // 13.8.3.5-13.8.3.6)
            let index_value = ensure_array(Some(index_value));
            let expanded = self.expand(&map_context, Some(key), index_value, base_url, true);
            let JsonValue::Array(items) = ensure_array(expanded) else {
                continue;
            };
            for mut item in items {
                // 13.8.3.7.1)
                if has("@graph") && !is_graph_object(&item) {
                    let mut object = JsonObject::new();
                    object.insert("@graph", JsonValue::Array(vec![item]));
                    item = JsonValue::Object(object);
                }
                let Some(object) = item.as_object_mut() else {
                    result.push(item);
                    continue;
                };
                if has("@index") && index_key != "@index" && expanded_index != "@none" {
                    // 13.8.3.7.2) property-valued index
                    let re_expanded = self.expand_value(
                        active_context,
                        &index_key,
                        JsonValue::String(index.clone()),
                    );
                    let Some(expanded_index_key) =
                        self.iri(active_context, &index_key, false, true)
                    else {
                        result.push(item);
                        continue;
                    };
                    let mut values = vec![re_expanded];
                    if let Some(existing) = object.remove(&expanded_index_key) {
                        match existing {
                            JsonValue::Array(existing) => values.extend(existing),
                            other => values.push(other),
                        }
                    }
                    object.insert(expanded_index_key, JsonValue::Array(values));
                    if object.contains_key("@value") && object.len() > 1 {
                        self.error(
                            "a value object with a property-valued index must not have extra entries",
                            JsonLdErrorCode::InvalidValueObject,
                        );
                    }
                } else if has("@index")
                    && !object.contains_key("@index")
                    && expanded_index != "@none"
                {
                    // 13.8.3.7.3)
                    object.insert("@index", JsonValue::String(index.clone()));
                } else if has("@id") && !object.contains_key("@id") && expanded_index != "@none" {
                    // 13.8.3.7.4)
                    if let Some(expanded_index) = self.iri(active_context, &index, true, false) {
                        object.insert("@id", JsonValue::String(expanded_index));
                    }
                } else if has("@type") && expanded_index != "@none" {
                    // 13.8.3.7.5)
                    let mut types = vec![JsonValue::String(expanded_index.clone())];
                    if let Some(existing) = object.remove("@type") {
                        match existing {
                            JsonValue::Array(existing) => types.extend(existing),
                            other => types.push(other),
                        }
                    }
                    object.insert("@type", JsonValue::Array(types));
                }
                result.push(item);
            }
        }
        JsonValue::Array(result)
    }

    /// The [Value Expansion Algorithm](https://www.w3.org/TR/json-ld11-api/#value-expansion).
    fn expand_value(
        &mut self,
        active_context: &Context,
        active_property: &str,
        value: JsonValue,
    ) -> JsonValue {
        let definition = active_context.term(active_property);
        let type_mapping = definition.and_then(|d| d.type_mapping.clone());
        // 1-2)
        if let (Some(type_mapping), JsonValue::String(text)) = (type_mapping.as_deref(), &value) {
            if matches!(type_mapping, "@id" | "@vocab") {
                let vocab = type_mapping == "@vocab";
                if let Some(expanded) = self.iri(active_context, text, true, vocab) {
                    let mut object = JsonObject::new();
                    object.insert("@id", JsonValue::String(expanded));
                    return JsonValue::Object(object);
                }
            }
        }
        // 3)
        let mut object = JsonObject::new();
        object.insert("@value", value.clone());
        // 4)
        if let Some(type_mapping) = &type_mapping {
            if !matches!(type_mapping.as_str(), "@id" | "@vocab" | "@none") {
                object.insert("@type", JsonValue::String(type_mapping.clone()));
                return JsonValue::Object(object);
            }
        }
        // 5)
        if matches!(value, JsonValue::String(_)) {
            let language = match definition.and_then(|d| d.language_mapping.clone()) {
                Some(language) => language,
                None => active_context.default_language.clone(),
            };
            let direction = match definition.and_then(|d| d.direction_mapping) {
                Some(direction) => direction,
                None => active_context.default_direction,
            };
            if let Some(language) = language {
                object.insert("@language", JsonValue::String(language));
            }
            if let Some(direction) = direction {
                object.insert(
                    "@direction",
                    JsonValue::String(direction.as_str().to_owned()),
                );
            }
        }
        JsonValue::Object(object)
    }

    /// Steps 15 to 20: validates and cleans up an expanded map.
    fn finalize(
        &mut self,
        active_property: ActiveProperty<'_>,
        mut result: JsonObject,
    ) -> Option<JsonValue> {
        // 15) value object
        if result.contains_key("@value") {
            const ALLOWED: [&str; 5] = ["@value", "@type", "@language", "@direction", "@index"];
            if let Some(unexpected) = result.keys().find(|key| !ALLOWED.contains(key)) {
                self.error(
                    format!("a value object may not contain {unexpected}"),
                    JsonLdErrorCode::InvalidValueObject,
                );
                return None;
            }
            if result.contains_key("@type")
                && (result.contains_key("@language") || result.contains_key("@direction"))
            {
                self.error(
                    "a value object may not combine @type with @language or @direction",
                    JsonLdErrorCode::InvalidValueObject,
                );
                return None;
            }
            let is_json = matches!(result.get("@type"), Some(JsonValue::String(s)) if s == "@json");
            if !is_json {
                // 15.3)
                match result.get("@value") {
                    Some(JsonValue::Null) => return None,
                    Some(JsonValue::Array(items)) if items.is_empty() => return None,
                    _ => {}
                }
                // 15.4)
                if result.contains_key("@language")
                    && !result.get("@value").is_some_and(JsonValue::is_string)
                {
                    self.error(
                        "only strings may be language-tagged",
                        JsonLdErrorCode::InvalidLanguageTaggedValue,
                    );
                    return None;
                }
                // 15.5)
                if let Some(datatype) = result.get("@type") {
                    let valid = datatype.as_str().is_some_and(|iri| {
                        !iri.starts_with("_:") && self.processor.iri_is_valid(iri)
                    });
                    if !valid {
                        self.error(
                            "the @type of a value object must be an IRI",
                            JsonLdErrorCode::InvalidTypedValue,
                        );
                        return None;
                    }
                }
            }
        } else if let Some(types) = result.get_mut("@type") {
            // 16)
            if !types.is_array() {
                let old = std::mem::replace(types, JsonValue::Null);
                *types = JsonValue::Array(vec![old]);
            }
        } else if result.contains_key("@set") || result.contains_key("@list") {
            // 17)
            let extra = result
                .keys()
                .find(|key| !matches!(*key, "@set" | "@list" | "@index"));
            if let Some(extra) = extra {
                self.error(
                    format!("a set or list object may not contain {extra}"),
                    JsonLdErrorCode::InvalidSetOrListObject,
                );
                return None;
            }
            // 17.2)
            if let Some(set) = result.remove("@set") {
                return match set {
                    JsonValue::Null => None,
                    other => Some(other),
                };
            }
        }
        // 18)
        if result.len() == 1 && result.contains_key("@language") {
            return None;
        }
        // 19) free-floating values
        if active_property.is_none() || active_property == Some("@graph") {
            let drop = result.is_empty()
                || result.contains_key("@value")
                || result.contains_key("@list")
                || (result.len() == 1 && result.contains_key("@id"));
            if drop {
                return None;
            }
        }
        // 20)
        Some(JsonValue::Object(result))
    }
}

fn to_string_array(value: &JsonValue) -> Vec<String> {
    match value {
        JsonValue::String(s) => vec![s.clone()],
        JsonValue::Array(items) => items
            .iter()
            .filter_map(|item| item.as_str().map(str::to_owned))
            .collect(),
        _ => Vec::new(),
    }
}

fn ensure_array(value: Option<JsonValue>) -> JsonValue {
    match value {
        None | Some(JsonValue::Null) => JsonValue::Array(Vec::new()),
        Some(JsonValue::Array(items)) => JsonValue::Array(items),
        Some(other) => JsonValue::Array(vec![other]),
    }
}

fn is_list_object(value: &JsonValue) -> bool {
    matches!(value, JsonValue::Object(object) if object.contains_key("@list"))
}

fn is_graph_object(value: &JsonValue) -> bool {
    let JsonValue::Object(object) = value else {
        return false;
    };
    object.contains_key("@graph")
        && object
            .keys()
            .all(|key| matches!(key, "@graph" | "@id" | "@index" | "@context"))
}

fn is_value_or_list_object(value: &JsonValue) -> bool {
    matches!(value, JsonValue::Object(object) if object.contains_key("@value") || object.contains_key("@list"))
}

fn is_node_object(value: &JsonValue) -> bool {
    matches!(value, JsonValue::Object(object) if !object.contains_key("@value")
        && !object.contains_key("@list")
        && !object.contains_key("@set"))
}

fn reverse_map_mut(result: &mut JsonObject) -> &mut JsonObject {
    if !matches!(result.get("@reverse"), Some(JsonValue::Object(_))) {
        result.insert("@reverse", JsonValue::Object(JsonObject::new()));
    }
    match result
        .get_mut("@reverse")
        .and_then(JsonValue::as_object_mut)
    {
        Some(map) => map,
        None => unreachable!("the @reverse entry was just ensured to be a map"),
    }
}

/// The JSON-LD [add value](https://www.w3.org/TR/json-ld11-api/#dfn-add-value) helper.
pub(crate) fn add_value(map: &mut JsonObject, key: &str, value: JsonValue, as_array: bool) {
    if as_array && !map.get(key).is_some_and(JsonValue::is_array) {
        let existing = map.remove(key);
        let seed = match existing {
            None => Vec::new(),
            Some(existing) => vec![existing],
        };
        map.insert(key, JsonValue::Array(seed));
    }
    if let JsonValue::Array(items) = value {
        for item in items {
            add_value(map, key, item, as_array);
        }
        return;
    }
    match map.get_mut(key) {
        Some(JsonValue::Array(existing)) => existing.push(value),
        Some(slot) => {
            let old = std::mem::replace(slot, JsonValue::Null);
            *slot = JsonValue::Array(vec![old, value]);
        }
        None => {
            map.insert(key, value);
        }
    }
}
