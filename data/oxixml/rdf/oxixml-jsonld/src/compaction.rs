//! [Compaction](https://www.w3.org/TR/json-ld11-api/#compaction-algorithms): re-applies a context
//! to an expanded document, shortening IRIs to terms and compact IRIs, keywords to keyword
//! aliases, and expanded values to plain JSON scalars.
//!
//! This implements the specification's *Compaction Algorithm*, *IRI Compaction*, and *Value
//! Compaction*, on top of the [inverse context](crate::inverse) that drives term selection. Step
//! numbers in the comments refer to
//! [§6 of the JSON-LD 1.1 API](https://www.w3.org/TR/json-ld11-api/#compaction-algorithms).

use crate::api::JsonLdOptions;
use crate::context::{Context, ContextProcessor, Direction, has_keyword_form};
use crate::error::{JsonLdErrorCode, JsonLdSyntaxError};
use crate::expansion::add_value;
use crate::inverse::{ANY, InverseContext, NONE, NULL};
use crate::profile::JsonLdProcessingMode;
use crate::value::{JsonObject, JsonValue};

/// An active context paired with the inverse context derived from it.
///
/// Compaction consults the inverse context constantly, so it is built once per active context
/// rather than once per lookup.
pub struct ActiveContext {
    context: Context,
    inverse: InverseContext,
}

impl ActiveContext {
    /// Pairs `context` with its freshly built inverse context.
    #[must_use]
    pub fn new(context: Context) -> Self {
        let inverse = InverseContext::build(&context);
        Self { context, inverse }
    }
}

/// Compacts `expanded` using `context`, returning a compacted document with the context attached.
pub fn compact_document(
    processor: &ContextProcessor<'_>,
    expanded: JsonValue,
    context: JsonValue,
    options: &JsonLdOptions<'_>,
) -> Result<JsonValue, JsonLdSyntaxError> {
    let mut errors = Vec::new();
    let base = Context::new_empty(options.base.clone());
    let active_context = processor.process_context(
        &base,
        context.clone(),
        options.base.as_deref(),
        &mut Vec::new(),
        false,
        true,
        true,
        &mut errors,
    );
    if let Some(error) = errors.into_iter().next() {
        return Err(error);
    }
    let active_context = ActiveContext::new(active_context);
    let mut compactor: Compactor<'_> = Compactor {
        processor,
        options,
        errors: Vec::new(),
    };
    let compacted = compactor.compact(&active_context, None, expanded);
    if !compactor.errors.is_empty() {
        return Err(compactor.errors.remove(0));
    }
    // API step 6: an array result becomes a `@graph` map.
    let mut result = match compacted {
        JsonValue::Array(items) if items.is_empty() => JsonObject::new(),
        JsonValue::Array(items) => {
            let alias = compactor.compact_iri(&active_context, "@graph", None, true, false);
            let mut object = JsonObject::new();
            object.insert(alias, JsonValue::Array(items));
            object
        }
        JsonValue::Object(object) => object,
        other => {
            let mut object = JsonObject::new();
            object.insert("@graph", JsonValue::Array(vec![other]));
            object
        }
    };
    // API step 7: re-attach the caller's context, in front.
    if !context_is_empty(&context) {
        let mut with_context = JsonObject::new();
        with_context.insert("@context", context);
        for (key, value) in result.into_entries() {
            with_context.insert(key, value);
        }
        result = with_context;
    }
    Ok(JsonValue::Object(result))
}

fn context_is_empty(context: &JsonValue) -> bool {
    match context {
        JsonValue::Null => true,
        JsonValue::Array(items) => items.is_empty(),
        JsonValue::Object(object) => object.is_empty(),
        _ => false,
    }
}

struct Compactor<'a> {
    processor: &'a ContextProcessor<'a>,
    options: &'a JsonLdOptions<'a>,
    errors: Vec<JsonLdSyntaxError>,
}

impl Compactor<'_> {
    fn is_json_ld_1_1(&self) -> bool {
        self.options.processing_mode == JsonLdProcessingMode::JsonLd1_1
    }

    fn error(&mut self, message: impl Into<String>, code: JsonLdErrorCode) {
        self.errors.push(JsonLdSyntaxError::msg(message, code));
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

    /// The recursive [Compaction Algorithm](https://www.w3.org/TR/json-ld11-api/#compaction-algorithm).
    fn compact(
        &mut self,
        active_context: &ActiveContext,
        active_property: Option<&str>,
        element: JsonValue,
    ) -> JsonValue {
        match element {
            // 2)
            JsonValue::Null | JsonValue::Bool(_) | JsonValue::Number(_) | JsonValue::String(_) => {
                element
            }
            // 3)
            JsonValue::Array(items) => {
                let mut result = Vec::new();
                for item in items {
                    let compacted = self.compact(active_context, active_property, item);
                    if !matches!(compacted, JsonValue::Null) {
                        result.push(compacted);
                    }
                }
                let container = active_property
                    .and_then(|property| active_context.context.term(property))
                    .map(|definition| definition.container_mapping.clone())
                    .unwrap_or_default();
                let keep_array = result.len() != 1
                    || !self.options.compact_arrays
                    || matches!(active_property, Some("@graph" | "@set"))
                    || container.iter().any(|c| c == "@list" || c == "@set");
                if keep_array {
                    JsonValue::Array(result)
                } else {
                    result.into_iter().next().unwrap_or(JsonValue::Null)
                }
            }
            // 4)
            JsonValue::Object(object) => {
                self.compact_object(active_context, active_property, object)
            }
        }
    }

    #[allow(clippy::too_many_lines)]
    fn compact_object(
        &mut self,
        outer_context: &ActiveContext,
        active_property: Option<&str>,
        element: JsonObject,
    ) -> JsonValue {
        // 1) the context as it was handed to this invocation, before any scoped context applies
        let input_context = outer_context.context.clone();
        let mut context = input_context.clone();
        // 5) a non-propagated context is reverted for new node objects
        if context.previous_context.is_some()
            && !element.contains_key("@value")
            && !(element.len() == 1 && element.contains_key("@id"))
        {
            if let Some(previous) = context.previous_context.clone() {
                context = *previous;
            }
        }
        // 6) the property-scoped context is looked up in the *input* context and applied on top of
        // the reverted one, so a term-scoped definition still governs its own values.
        if let Some(property) = active_property {
            let scoped = input_context.term(property).and_then(|definition| {
                definition
                    .context
                    .clone()
                    .map(|local| (local, definition.base_url.clone()))
            });
            if let Some((local, base_url)) = scoped {
                context = self.process_context(&context, local, base_url.as_deref(), true, true);
            }
        }
        let mut active = ActiveContext::new(context);
        // 7) scalar-compactable values
        if element.contains_key("@value") || element.contains_key("@id") {
            let compacted = self.compact_value(&active, active_property, &element);
            let is_json_typed = active_property
                .and_then(|property| active.context.term(property))
                .and_then(|definition| definition.type_mapping.clone())
                .as_deref()
                == Some("@json");
            if compacted.is_scalar() || is_json_typed {
                return compacted;
            }
        }
        // 8) a list object inside a `@list` container compacts to its bare contents
        if element.contains_key("@list")
            && active_property
                .and_then(|property| active.context.term(property))
                .is_some_and(|definition| definition.has_container("@list"))
        {
            let list = element.get("@list").cloned().unwrap_or(JsonValue::Null);
            return self.compact(&active, active_property, list);
        }
        // 9)
        let inside_reverse = active_property == Some("@reverse");
        // 10)
        let mut result = JsonObject::new();
        // 11) type-scoped contexts, in lexicographic order of the *compacted* types
        if let Some(types) = element.get("@type") {
            let mut compacted_types: Vec<String> = to_array(types.clone())
                .into_iter()
                .filter_map(|item| item.as_str().map(str::to_owned))
                .map(|expanded_type| self.compact_iri(&active, &expanded_type, None, true, false))
                .collect();
            compacted_types.sort_unstable();
            let mut context = active.context.clone();
            let type_scoped_context = &input_context;
            for term in &compacted_types {
                let scoped = type_scoped_context.term(term).and_then(|definition| {
                    definition
                        .context
                        .clone()
                        .map(|local| (local, definition.base_url.clone()))
                });
                if let Some((local, base_url)) = scoped {
                    context =
                        self.process_context(&context, local, base_url.as_deref(), false, false);
                }
            }
            active = ActiveContext::new(context);
        }
        // 12) every entry of the expanded map
        let keys: Vec<String> = if self.options.ordered {
            element.sorted_keys()
        } else {
            element.keys().map(str::to_owned).collect()
        };
        for expanded_property in keys {
            let expanded_value = element
                .get(&expanded_property)
                .cloned()
                .unwrap_or(JsonValue::Null);
            match expanded_property.as_str() {
                // 12.1)
                "@id" => {
                    let compacted_value = match &expanded_value {
                        JsonValue::String(id) => {
                            JsonValue::String(self.compact_iri(&active, id, None, false, false))
                        }
                        other => other.clone(),
                    };
                    let alias = self.compact_iri(&active, "@id", None, true, false);
                    result.insert(alias, compacted_value);
                }
                // 12.2)
                "@type" => {
                    let compacted_value = match &expanded_value {
                        JsonValue::String(expanded_type) => JsonValue::String(self.compact_iri(
                            &ActiveContext::new(input_context.clone()),
                            expanded_type,
                            None,
                            true,
                            false,
                        )),
                        JsonValue::Array(items) => JsonValue::Array(
                            items
                                .iter()
                                .filter_map(|item| item.as_str())
                                .map(|expanded_type| {
                                    JsonValue::String(self.compact_iri(
                                        &ActiveContext::new(input_context.clone()),
                                        expanded_type,
                                        None,
                                        true,
                                        false,
                                    ))
                                })
                                .collect(),
                        ),
                        other => other.clone(),
                    };
                    let alias = self.compact_iri(&active, "@type", None, true, false);
                    let as_array = (self.is_json_ld_1_1()
                        && active
                            .context
                            .term(&alias)
                            .is_some_and(|definition| definition.has_container("@set")))
                        || !self.options.compact_arrays;
                    add_value(&mut result, &alias, compacted_value, as_array);
                }
                // 12.3)
                "@reverse" => {
                    let compacted_value =
                        self.compact(&active, Some("@reverse"), expanded_value.clone());
                    let JsonValue::Object(compacted_value) = compacted_value else {
                        continue;
                    };
                    let mut remaining = JsonObject::new();
                    for (property, value) in compacted_value.into_entries() {
                        let definition = active.context.term(&property);
                        if definition.is_some_and(|definition| definition.reverse_property) {
                            let as_array = definition
                                .is_some_and(|definition| definition.has_container("@set"))
                                || !self.options.compact_arrays;
                            add_value(&mut result, &property, value, as_array);
                        } else {
                            remaining.insert(property, value);
                        }
                    }
                    if !remaining.is_empty() {
                        let alias = self.compact_iri(&active, "@reverse", None, true, false);
                        result.insert(alias, JsonValue::Object(remaining));
                    }
                }
                // 12.5) an `@index` inside an `@index` container is implied by the map key
                "@index"
                    if active_property
                        .and_then(|property| active.context.term(property))
                        .is_some_and(|definition| definition.has_container("@index")) => {}
                // 12.6)
                "@direction" | "@index" | "@language" | "@value" => {
                    let alias = self.compact_iri(&active, &expanded_property, None, true, false);
                    result.insert(alias, expanded_value);
                }
                _ => {
                    self.compact_entry(
                        &active,
                        &expanded_property,
                        expanded_value,
                        inside_reverse,
                        &mut result,
                    );
                }
            }
        }
        // 13)
        JsonValue::Object(result)
    }

    /// Steps 12.7 and 12.8: compacts a property entry (a non-keyword, `@graph`, or `@included`).
    #[allow(clippy::too_many_lines)]
    fn compact_entry(
        &mut self,
        active: &ActiveContext,
        expanded_property: &str,
        expanded_value: JsonValue,
        inside_reverse: bool,
        result: &mut JsonObject,
    ) {
        let items = to_array(expanded_value);
        // 12.7) an empty array still declares the property
        if items.is_empty() {
            let item_active_property = self.compact_iri(
                active,
                expanded_property,
                Some(&JsonValue::Array(Vec::new())),
                true,
                inside_reverse,
            );
            let nest_result = self.nest_result(active, &item_active_property, result);
            add_value(
                nest_result,
                &item_active_property,
                JsonValue::Array(Vec::new()),
                true,
            );
            return;
        }
        // 12.8)
        for expanded_item in items {
            let item_active_property = self.compact_iri(
                active,
                expanded_property,
                Some(&expanded_item),
                true,
                inside_reverse,
            );
            let definition = active.context.term(&item_active_property).cloned();
            let container = definition
                .as_ref()
                .map(|definition| definition.container_mapping.clone())
                .unwrap_or_default();
            let has = |keyword: &str| container.iter().any(|c| c == keyword);
            let as_array = has("@set")
                || item_active_property == "@graph"
                || item_active_property == "@list"
                || !self.options.compact_arrays;
            // 12.8.5) a list or graph object is compacted through its contents
            let inner = match &expanded_item {
                JsonValue::Object(object) if object.contains_key("@list") => {
                    object.get("@list").cloned().unwrap_or(JsonValue::Null)
                }
                JsonValue::Object(object) if is_graph_object(&expanded_item) => {
                    object.get("@graph").cloned().unwrap_or(JsonValue::Null)
                }
                other => other.clone(),
            };
            let mut compacted_item = self.compact(active, Some(&item_active_property), inner);

            if is_list_object(&expanded_item) {
                // 12.8.6)
                if !compacted_item.is_array() {
                    compacted_item = JsonValue::Array(vec![compacted_item]);
                }
                if has("@list") {
                    let nest_result = self.nest_result(active, &item_active_property, result);
                    nest_result.insert(item_active_property.clone(), compacted_item);
                } else {
                    let list_alias = self.compact_iri(active, "@list", None, true, false);
                    let mut list_object = JsonObject::new();
                    list_object.insert(list_alias, compacted_item);
                    if let JsonValue::Object(object) = &expanded_item {
                        if let Some(index) = object.get("@index") {
                            let index_alias = self.compact_iri(active, "@index", None, true, false);
                            list_object.insert(index_alias, index.clone());
                        }
                    }
                    let nest_result = self.nest_result(active, &item_active_property, result);
                    add_value(
                        nest_result,
                        &item_active_property,
                        JsonValue::Object(list_object),
                        as_array,
                    );
                }
                continue;
            }

            if is_graph_object(&expanded_item) {
                self.compact_graph_object(
                    active,
                    &item_active_property,
                    &container,
                    &expanded_item,
                    compacted_item,
                    as_array,
                    result,
                );
                continue;
            }

            if (has("@language") || has("@index") || has("@id") || has("@type")) && !has("@graph") {
                self.compact_into_map(
                    active,
                    &item_active_property,
                    &container,
                    definition.as_ref(),
                    &expanded_item,
                    compacted_item,
                    as_array,
                    result,
                );
                continue;
            }
            // 12.8.9)
            let nest_result = self.nest_result(active, &item_active_property, result);
            add_value(nest_result, &item_active_property, compacted_item, as_array);
        }
    }

    /// Step 12.8.7: a graph object becomes a graph map, an `@included` block, or a `@graph` entry.
    #[allow(clippy::too_many_arguments)]
    fn compact_graph_object(
        &mut self,
        active: &ActiveContext,
        item_active_property: &str,
        container: &[String],
        expanded_item: &JsonValue,
        compacted_item: JsonValue,
        as_array: bool,
        result: &mut JsonObject,
    ) {
        let has = |keyword: &str| container.iter().any(|c| c == keyword);
        let object = expanded_item.as_object();
        let has_id = object.is_some_and(|object| object.contains_key("@id"));
        let has_index = object.is_some_and(|object| object.contains_key("@index"));
        if has("@graph") && has("@id") {
            let map_key = match object.and_then(|object| object.get("@id")) {
                Some(JsonValue::String(id)) => self.compact_iri(active, id, None, false, false),
                _ => self.compact_iri(active, NONE, None, true, false),
            };
            let nest_result = self.nest_result(active, item_active_property, result);
            let map_object = map_entry(nest_result, item_active_property);
            add_value(map_object, &map_key, compacted_item, as_array);
        } else if has("@graph") && has("@index") && !has_id {
            let _ = has_index;
            let map_key = object
                .and_then(|object| object.get("@index"))
                .and_then(JsonValue::as_str)
                .map_or_else(|| NONE.to_owned(), str::to_owned);
            let nest_result = self.nest_result(active, item_active_property, result);
            let map_object = map_entry(nest_result, item_active_property);
            add_value(map_object, &map_key, compacted_item, as_array);
        } else if has("@graph") && !has_id {
            // A simple graph object in a plain `@graph` container.
            let compacted_item = match compacted_item {
                JsonValue::Array(items) if items.len() > 1 => {
                    let included_alias = self.compact_iri(active, "@included", None, true, false);
                    let mut object = JsonObject::new();
                    object.insert(included_alias, JsonValue::Array(items));
                    JsonValue::Object(object)
                }
                other => other,
            };
            let nest_result = self.nest_result(active, item_active_property, result);
            add_value(nest_result, item_active_property, compacted_item, as_array);
        } else {
            let graph_alias = self.compact_iri(active, "@graph", None, true, false);
            let mut wrapper = JsonObject::new();
            wrapper.insert(graph_alias, compacted_item);
            if let Some(JsonValue::String(id)) = object.and_then(|object| object.get("@id")) {
                let id_alias = self.compact_iri(active, "@id", None, true, false);
                let compacted_id = self.compact_iri(active, id, None, false, false);
                wrapper.insert(id_alias, JsonValue::String(compacted_id));
            }
            if let Some(index) = object.and_then(|object| object.get("@index")) {
                let index_alias = self.compact_iri(active, "@index", None, true, false);
                wrapper.insert(index_alias, index.clone());
            }
            let nest_result = self.nest_result(active, item_active_property, result);
            add_value(
                nest_result,
                item_active_property,
                JsonValue::Object(wrapper),
                as_array,
            );
        }
    }

    /// Step 12.8.8: a `@language`, `@index`, `@id`, or `@type` container becomes a map.
    #[allow(clippy::too_many_arguments)]
    fn compact_into_map(
        &mut self,
        active: &ActiveContext,
        item_active_property: &str,
        container: &[String],
        definition: Option<&crate::context::TermDefinition>,
        expanded_item: &JsonValue,
        compacted_item: JsonValue,
        as_array: bool,
        result: &mut JsonObject,
    ) {
        let has = |keyword: &str| container.iter().any(|c| c == keyword);
        let expanded_object = expanded_item.as_object();
        let mut compacted_item = compacted_item;
        let mut container_key = if has("@language") {
            self.compact_iri(active, "@language", None, true, false)
        } else if has("@index") {
            self.compact_iri(active, "@index", None, true, false)
        } else if has("@id") {
            self.compact_iri(active, "@id", None, true, false)
        } else {
            self.compact_iri(active, "@type", None, true, false)
        };
        let index_key = definition
            .and_then(|definition| definition.index_mapping.clone())
            .unwrap_or_else(|| "@index".to_owned());
        let mut map_key: Option<String> = None;
        if has("@language") {
            if let Some(object) = compacted_item.as_object() {
                if let Some(value) = object.get("@value") {
                    compacted_item = value.clone();
                }
            }
            map_key = expanded_object
                .and_then(|object| object.get("@language"))
                .and_then(JsonValue::as_str)
                .map(str::to_owned);
        } else if has("@index") && index_key == "@index" {
            map_key = expanded_object
                .and_then(|object| object.get("@index"))
                .and_then(JsonValue::as_str)
                .map(str::to_owned);
        } else if has("@index") {
            // A property-valued index: the map key is the property's own first value, and only if
            // that value compacted to a string. Otherwise the entry stays put and `@none` is used.
            container_key = self.compact_iri(active, &index_key, None, true, false);
            if let Some(object) = compacted_item.as_object_mut() {
                map_key = take_first_string(object, &container_key);
            }
        } else if has("@id") {
            if let Some(object) = compacted_item.as_object_mut() {
                map_key = object
                    .remove(&container_key)
                    .and_then(|value| value.as_str().map(str::to_owned));
            }
        } else if has("@type") {
            if let Some(object) = compacted_item.as_object_mut() {
                map_key = take_first_string(object, &container_key);
            }
            // A residual single `@id` entry compacts as a plain node reference.
            let single_id = compacted_item.as_object().is_some_and(|object| {
                object.len() == 1
                    && object.keys().next().is_some_and(|key| {
                        self.processor
                            .expand_iri_in_context(&active.context, key, false, true)
                            .as_deref()
                            == Some("@id")
                    })
            });
            if single_id {
                if let Some(JsonValue::String(id)) =
                    expanded_object.and_then(|object| object.get("@id"))
                {
                    let mut single = JsonObject::new();
                    single.insert("@id", JsonValue::String(id.clone()));
                    compacted_item = self.compact(
                        active,
                        Some(item_active_property),
                        JsonValue::Object(single),
                    );
                }
            }
        }
        let map_key = match map_key {
            Some(map_key) => map_key,
            None => self.compact_iri(active, NONE, None, true, false),
        };
        let nest_result = self.nest_result(active, item_active_property, result);
        let map_object = map_entry(nest_result, item_active_property);
        add_value(map_object, &map_key, compacted_item, as_array);
    }

    /// Resolves the `@nest` indirection of a term: entries of a nesting term land in a sub-map.
    fn nest_result<'r>(
        &mut self,
        active: &ActiveContext,
        item_active_property: &str,
        result: &'r mut JsonObject,
    ) -> &'r mut JsonObject {
        let nest_term = active
            .context
            .term(item_active_property)
            .and_then(|definition| definition.nest_value.clone());
        let Some(nest_term) = nest_term else {
            return result;
        };
        let expands_to_nest = nest_term == "@nest"
            || self
                .processor
                .expand_iri_in_context(&active.context, &nest_term, false, true)
                .as_deref()
                == Some("@nest");
        if !expands_to_nest {
            self.error(
                format!("{nest_term} does not expand to @nest"),
                JsonLdErrorCode::InvalidNestValue,
            );
            return result;
        }
        map_entry(result, &nest_term)
    }

    /// The [Value Compaction](https://www.w3.org/TR/json-ld11-api/#value-compaction) algorithm.
    #[allow(clippy::too_many_lines)]
    fn compact_value(
        &mut self,
        active: &ActiveContext,
        active_property: Option<&str>,
        value: &JsonObject,
    ) -> JsonValue {
        let definition = active_property.and_then(|property| active.context.term(property));
        let type_mapping = definition.and_then(|definition| definition.type_mapping.clone());
        let container = definition
            .map(|definition| definition.container_mapping.clone())
            .unwrap_or_default();
        // 4-5)
        let language = match definition.and_then(|definition| definition.language_mapping.clone()) {
            Some(language) => language,
            None => active.context.default_language.clone(),
        };
        let direction = match definition.and_then(|definition| definition.direction_mapping) {
            Some(direction) => direction,
            None => active.context.default_direction,
        };
        let value_language = value.get("@language").and_then(JsonValue::as_str);
        let value_direction = value.get("@direction").and_then(JsonValue::as_str);
        let value_type = value.get("@type").and_then(JsonValue::as_str);
        let has_index = value.contains_key("@index");
        let index_ok = !has_index || container.iter().any(|c| c == "@index");

        // 6) a bare node reference
        if value.contains_key("@id") && value.keys().all(|key| matches!(key, "@id" | "@index")) {
            if let Some(JsonValue::String(id)) = value.get("@id") {
                match type_mapping.as_deref() {
                    Some("@id") => {
                        return JsonValue::String(self.compact_iri(active, id, None, false, false));
                    }
                    Some("@vocab") => {
                        return JsonValue::String(self.compact_iri(active, id, None, true, false));
                    }
                    _ => {}
                }
            }
        } else if value_type.is_some() && value_type == type_mapping.as_deref() {
            // 7)
            if let Some(inner) = value.get("@value") {
                return inner.clone();
            }
        } else if type_mapping.as_deref() == Some(NONE)
            || (value_type.is_some() && value_type != type_mapping.as_deref())
        {
            // 8) value compaction is disabled; only the datatype IRI is compacted
        } else if value.get("@value").is_some_and(|inner| !inner.is_string()) {
            // 9)
            if index_ok {
                if let Some(inner) = value.get("@value") {
                    return inner.clone();
                }
            }
        } else if language_matches(value_language, language.as_deref())
            && direction_matches(value_direction, direction)
        {
            // 10)
            if index_ok {
                if let Some(inner) = value.get("@value") {
                    return inner.clone();
                }
            }
        }
        // 11) compact the remaining keys (and any `@type` / `@id` IRI)
        let mut result = JsonObject::new();
        for (key, entry) in value.iter() {
            let alias = self.compact_iri(active, key, None, true, false);
            let entry = match key {
                "@type" => match entry {
                    JsonValue::String(datatype) => {
                        JsonValue::String(self.compact_iri(active, datatype, None, true, false))
                    }
                    other => other.clone(),
                },
                "@id" => match entry {
                    JsonValue::String(id) => {
                        JsonValue::String(self.compact_iri(active, id, None, false, false))
                    }
                    other => other.clone(),
                },
                _ => entry.clone(),
            };
            result.insert(alias, entry);
        }
        JsonValue::Object(result)
    }

    /// The [IRI Compaction](https://www.w3.org/TR/json-ld11-api/#iri-compaction) algorithm.
    #[allow(clippy::too_many_lines)]
    fn compact_iri(
        &mut self,
        active: &ActiveContext,
        var: &str,
        value: Option<&JsonValue>,
        vocab: bool,
        reverse: bool,
    ) -> String {
        let context = &active.context;
        // 4)
        if vocab && active.inverse.contains(var) {
            // 4.1)
            let default_language = match (&context.default_language, context.default_direction) {
                (language, Some(direction)) => format!(
                    "{}_{}",
                    language.clone().unwrap_or_default(),
                    direction.as_str()
                )
                .to_lowercase(),
                (Some(language), None) => language.to_lowercase(),
                (None, None) => NONE.to_owned(),
            };
            let mut containers: Vec<String> = Vec::new();
            let mut type_language = "@language".to_owned();
            let mut type_language_value = NULL.to_owned();
            let value_object = value.and_then(JsonValue::as_object);
            let is_graph = value.is_some_and(is_graph_object);
            let is_list = value.is_some_and(is_list_object);
            let has_index = value_object.is_some_and(|object| object.contains_key("@index"));
            // 4.5)
            if has_index && !is_graph {
                containers.push("@index".to_owned());
                containers.push("@index@set".to_owned());
            }
            if reverse {
                // 4.6)
                type_language = "@type".to_owned();
                type_language_value = "@reverse".to_owned();
                containers.push("@set".to_owned());
            } else if is_list {
                // 4.7)
                if !has_index {
                    containers.push("@list".to_owned());
                }
                let list = value_object
                    .and_then(|object| object.get("@list"))
                    .map(|list| to_array(list.clone()))
                    .unwrap_or_default();
                let mut common_type: Option<String> = None;
                let mut common_language: Option<String> = if list.is_empty() {
                    Some(default_language.clone())
                } else {
                    None
                };
                for item in &list {
                    let mut item_language = NONE.to_owned();
                    let mut item_type = NONE.to_owned();
                    let item_object = item.as_object();
                    let is_value = item_object.is_some_and(|object| object.contains_key("@value"));
                    if is_value {
                        let object = item_object.unwrap_or(&EMPTY_OBJECT);
                        if let Some(direction) =
                            object.get("@direction").and_then(JsonValue::as_str)
                        {
                            let language = object
                                .get("@language")
                                .and_then(JsonValue::as_str)
                                .unwrap_or_default();
                            item_language = format!("{language}_{direction}").to_lowercase();
                        } else if let Some(language) =
                            object.get("@language").and_then(JsonValue::as_str)
                        {
                            item_language = language.to_lowercase();
                        } else if let Some(datatype) =
                            object.get("@type").and_then(JsonValue::as_str)
                        {
                            item_type = datatype.to_owned();
                        } else {
                            item_language = NULL.to_owned();
                        }
                    } else {
                        item_type = "@id".to_owned();
                    }
                    match &common_language {
                        None => common_language = Some(item_language.clone()),
                        Some(existing) if *existing != item_language && is_value => {
                            common_language = Some(NONE.to_owned());
                        }
                        Some(_) => {}
                    }
                    match &common_type {
                        None => common_type = Some(item_type.clone()),
                        Some(existing) if *existing != item_type => {
                            common_type = Some(NONE.to_owned());
                        }
                        Some(_) => {}
                    }
                    if common_language.as_deref() == Some(NONE)
                        && common_type.as_deref() == Some(NONE)
                    {
                        break;
                    }
                }
                let common_language = common_language.unwrap_or_else(|| NONE.to_owned());
                let common_type = common_type.unwrap_or_else(|| NONE.to_owned());
                if common_type != NONE {
                    type_language = "@type".to_owned();
                    type_language_value = common_type;
                } else {
                    type_language_value = common_language;
                }
            } else if is_graph {
                // 4.8)
                let has_id = value_object.is_some_and(|object| object.contains_key("@id"));
                if has_index {
                    containers.push("@graph@index".to_owned());
                    containers.push("@graph@index@set".to_owned());
                }
                if has_id {
                    containers.push("@graph@id".to_owned());
                    containers.push("@graph@id@set".to_owned());
                }
                containers.push("@graph".to_owned());
                containers.push("@graph@set".to_owned());
                containers.push("@set".to_owned());
                if !has_index {
                    containers.push("@graph@index".to_owned());
                    containers.push("@graph@index@set".to_owned());
                }
                if !has_id {
                    containers.push("@graph@id".to_owned());
                    containers.push("@graph@id@set".to_owned());
                }
                containers.push("@index".to_owned());
                containers.push("@index@set".to_owned());
                type_language = "@type".to_owned();
                type_language_value = "@id".to_owned();
            } else {
                // 4.9)
                let is_value_object =
                    value_object.is_some_and(|object| object.contains_key("@value"));
                if is_value_object {
                    let object = value_object.unwrap_or(&EMPTY_OBJECT);
                    if let (Some(direction), false) = (
                        object.get("@direction").and_then(JsonValue::as_str),
                        has_index,
                    ) {
                        let language = object
                            .get("@language")
                            .and_then(JsonValue::as_str)
                            .unwrap_or_default();
                        type_language_value = format!("{language}_{direction}").to_lowercase();
                        containers.push("@language".to_owned());
                        containers.push("@language@set".to_owned());
                    } else if let (Some(language), false) = (
                        object.get("@language").and_then(JsonValue::as_str),
                        has_index,
                    ) {
                        type_language_value = language.to_lowercase();
                        containers.push("@language".to_owned());
                        containers.push("@language@set".to_owned());
                    } else if let Some(datatype) = object.get("@type").and_then(JsonValue::as_str) {
                        type_language_value = datatype.to_owned();
                        type_language = "@type".to_owned();
                    }
                } else {
                    type_language = "@type".to_owned();
                    type_language_value = "@id".to_owned();
                    containers.push("@id".to_owned());
                    containers.push("@id@set".to_owned());
                    containers.push("@type".to_owned());
                    containers.push("@set@type".to_owned());
                }
                containers.push("@set".to_owned());
            }
            // 4.10-4.12)
            containers.push(NONE.to_owned());
            if self.is_json_ld_1_1() && !has_index {
                containers.push("@index".to_owned());
                containers.push("@index@set".to_owned());
            }
            if self.is_json_ld_1_1()
                && value_object
                    .is_some_and(|object| object.len() == 1 && object.contains_key("@value"))
            {
                containers.push("@language".to_owned());
                containers.push("@language@set".to_owned());
            }
            // 4.14-4.19)
            let mut preferred_values: Vec<String> = Vec::new();
            if type_language_value == "@reverse" {
                preferred_values.push("@reverse".to_owned());
            }
            let value_id = value_object
                .filter(|object| !object.contains_key("@value"))
                .and_then(|object| object.get("@id"))
                .and_then(JsonValue::as_str);
            if matches!(type_language_value.as_str(), "@id" | "@reverse") && value_id.is_some() {
                let id = value_id.unwrap_or_default();
                let compacted = self.compact_iri(active, id, None, true, false);
                let round_trips = active
                    .context
                    .term(&compacted)
                    .and_then(|definition| definition.iri())
                    == Some(id);
                if round_trips {
                    preferred_values.push("@vocab".to_owned());
                    preferred_values.push("@id".to_owned());
                    preferred_values.push(NONE.to_owned());
                } else {
                    preferred_values.push("@id".to_owned());
                    preferred_values.push("@vocab".to_owned());
                    preferred_values.push(NONE.to_owned());
                }
            } else {
                preferred_values.push(type_language_value.clone());
                preferred_values.push(NONE.to_owned());
                let empty_list = value_object
                    .and_then(|object| object.get("@list"))
                    .and_then(JsonValue::as_array)
                    .is_some_and(<[JsonValue]>::is_empty);
                if empty_list {
                    type_language = ANY.to_owned();
                }
            }
            preferred_values.push(ANY.to_owned());
            let underscored: Vec<String> = preferred_values
                .iter()
                .filter_map(|entry| entry.find('_').map(|index| entry[index..].to_owned()))
                .collect();
            preferred_values.extend(underscored);
            // 4.20-4.21)
            if let Some(term) =
                active
                    .inverse
                    .select_term(var, &containers, &type_language, &preferred_values)
            {
                return term;
            }
        }
        // 5) relative to the vocabulary mapping
        if vocab {
            if let Some(vocabulary_mapping) = &context.vocabulary_mapping {
                if var.len() > vocabulary_mapping.len()
                    && var.starts_with(vocabulary_mapping.as_str())
                {
                    if let Some(suffix) = var.get(vocabulary_mapping.len()..) {
                        if context.term(suffix).is_none() {
                            return suffix.to_owned();
                        }
                    }
                }
            }
        }
        // 6-8) a compact IRI
        let mut compact_iri: Option<String> = None;
        for (term, definition) in &context.term_definitions {
            let Some(iri_mapping) = definition.iri() else {
                continue;
            };
            if iri_mapping == var || !var.starts_with(iri_mapping) || !definition.prefix_flag {
                continue;
            }
            let Some(suffix) = var.get(iri_mapping.len()..) else {
                continue;
            };
            let candidate = format!("{term}:{suffix}");
            let shorter = match &compact_iri {
                None => true,
                Some(existing) => {
                    candidate.len() < existing.len()
                        || (candidate.len() == existing.len() && candidate < *existing)
                }
            };
            let usable = match context.term(&candidate) {
                None => true,
                Some(existing) => existing.iri() == Some(var) && value.is_none(),
            };
            if shorter && usable {
                compact_iri = Some(candidate);
            }
        }
        if let Some(compact_iri) = compact_iri {
            return compact_iri;
        }
        // 9) an IRI that could be read back as a compact IRI is unrepresentable.
        //
        // The check only applies to an IRI that carries a value, i.e. one taken from the expanded
        // document. Bare lookups — keyword aliases, `@type` values, and the index key of a
        // property-valued index container — legitimately pass strings that are already written in
        // compact form and must survive unchanged.
        if let (Some((scheme, rest)), true) = (var.split_once(':'), value.is_some()) {
            if !rest.starts_with("//")
                && context
                    .term(scheme)
                    .is_some_and(|definition| definition.prefix_flag)
            {
                self.error(
                    format!("<{var}> would be confused with the compact IRI {scheme}:…"),
                    JsonLdErrorCode::IriConfusedWithPrefix,
                );
            }
        }
        // 10) a relative reference against the base
        if !vocab && self.options.compact_to_relative {
            if let Some(base) = &context.base_iri {
                if let Some(relative) = relativize(base, var) {
                    return relative;
                }
            }
        }
        // 11)
        var.to_owned()
    }
}

static EMPTY_OBJECT: JsonObject = JsonObject::empty();

fn language_matches(value_language: Option<&str>, language: Option<&str>) -> bool {
    match (value_language, language) {
        (None, None) => true,
        (Some(left), Some(right)) => left.eq_ignore_ascii_case(right),
        _ => false,
    }
}

fn direction_matches(value_direction: Option<&str>, direction: Option<Direction>) -> bool {
    match (value_direction, direction) {
        (None, None) => true,
        (Some(left), Some(right)) => left == right.as_str(),
        _ => false,
    }
}

fn to_array(value: JsonValue) -> Vec<JsonValue> {
    match value {
        JsonValue::Array(items) => items,
        JsonValue::Null => Vec::new(),
        other => vec![other],
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

/// Borrows (creating if needed) the map stored under `key`.
fn map_entry<'m>(result: &'m mut JsonObject, key: &str) -> &'m mut JsonObject {
    if !matches!(result.get(key), Some(JsonValue::Object(_))) {
        result.insert(key, JsonValue::Object(JsonObject::new()));
    }
    match result.get_mut(key).and_then(JsonValue::as_object_mut) {
        Some(object) => object,
        None => unreachable!("the entry was just ensured to be a map"),
    }
}

/// Removes `base` from `iri`, producing the relative IRI reference JSON-LD compaction expects.
///
/// This follows RFC 3986 §5.3 in reverse the way the specification's test suite requires: matching
/// path segments are dropped, unmatched leading base segments become `../`, and an IRI equal to the
/// base keeps its last segment (so `…/things/1` against `…/things/1` compacts to `1`, not to the
/// empty string, which would not round-trip).
///
/// Returns `None` when `iri` is not below `base`.
fn relativize(base: &str, iri: &str) -> Option<String> {
    let (root, base_rest) = split_root(base)?;
    let rest = iri.strip_prefix(root.as_str())?;
    let (relative_path, query, fragment) = split_parts(rest);
    let (base_path, _, _) = split_parts(base_rest);
    let base_path = remove_dot_segments(base_path);
    let relative_path = remove_dot_segments(relative_path);
    let mut base_segments: Vec<&str> = base_path.split('/').collect();
    let mut iri_segments: Vec<&str> = relative_path.split('/').collect();
    // The last segment is only dropped when there is no query or fragment to carry it.
    let last = usize::from(query.is_none() && fragment.is_none());
    while !base_segments.is_empty() && iri_segments.len() > last {
        if base_segments.first() != iri_segments.first() {
            break;
        }
        base_segments.remove(0);
        iri_segments.remove(0);
    }
    let mut result = String::new();
    if !base_segments.is_empty() {
        base_segments.pop();
        for _ in &base_segments {
            result.push_str("../");
        }
    }
    result.push_str(&iri_segments.join("/"));
    if let Some(query) = query {
        result.push('?');
        result.push_str(query);
    }
    if let Some(fragment) = fragment {
        result.push('#');
        result.push_str(fragment);
    }
    if result.is_empty() {
        result = "./".to_owned();
    }
    // A relative reference shaped like a keyword would be dropped when read back.
    if has_keyword_form(&result) {
        result = format!("./{result}");
    }
    Some(result)
}

/// Splits an absolute IRI into its `scheme://authority` root and the remainder.
fn split_root(url: &str) -> Option<(String, &str)> {
    let scheme_end = url.find("://")? + 3;
    let rest = url.get(scheme_end..)?;
    let authority_end = rest.find(['/', '?', '#']).unwrap_or(rest.len());
    let split = scheme_end + authority_end;
    Some((url.get(..split)?.to_owned(), url.get(split..)?))
}

/// Splits `rest` (an IRI with its root removed) into path, query, and fragment.
fn split_parts(rest: &str) -> (&str, Option<&str>, Option<&str>) {
    let (without_fragment, fragment) = match rest.split_once('#') {
        Some((head, fragment)) => (head, Some(fragment)),
        None => (rest, None),
    };
    let (path, query) = match without_fragment.split_once('?') {
        Some((head, query)) => (head, Some(query)),
        None => (without_fragment, None),
    };
    (path, query, fragment)
}

/// RFC 3986 §5.2.4 `remove_dot_segments`.
fn remove_dot_segments(path: &str) -> String {
    if path.is_empty() {
        return String::new();
    }
    let mut input: Vec<&str> = path.split('/').collect();
    let mut output: Vec<&str> = Vec::new();
    while !input.is_empty() {
        let next = input.remove(0);
        let done = input.is_empty();
        if next == "." {
            if done {
                output.push("");
            }
            continue;
        }
        if next == ".." {
            output.pop();
            if done {
                output.push("");
            }
            continue;
        }
        output.push(next);
    }
    if path.starts_with('/') && output.first().is_some_and(|first| !first.is_empty()) {
        output.insert(0, "");
    }
    if output.len() == 1 && output.first() == Some(&"") {
        return "/".to_owned();
    }
    output.join("/")
}

/// Takes the first value of `key` when it is a string, leaving any remaining values behind.
fn take_first_string(object: &mut JsonObject, key: &str) -> Option<String> {
    let values = to_array(object.get(key).cloned().unwrap_or(JsonValue::Null));
    let first = values.first().and_then(JsonValue::as_str)?.to_owned();
    match values.len() {
        0 | 1 => {
            object.remove(key);
        }
        2 => {
            if let Some(second) = values.get(1) {
                object.insert(key, second.clone());
            }
        }
        _ => {
            object.insert(key, JsonValue::Array(values.get(1..)?.to_vec()));
        }
    }
    Some(first)
}
