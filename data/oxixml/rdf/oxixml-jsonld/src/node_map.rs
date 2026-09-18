//! [Node Map Generation](https://www.w3.org/TR/json-ld-api/#node-map-generation).
//!
//! The node map is the intermediate structure shared by flattening and RDF serialization: it
//! collects every node object of an expanded document into a per-graph, per-subject map, assigning
//! fresh blank node identifiers where needed. Both [`crate::flattening`] and [`crate::to_rdf`]
//! build on it.

use crate::error::{JsonLdErrorCode, JsonLdSyntaxError};
use crate::value::{JsonObject, JsonValue};
use std::collections::HashMap;

/// Allocates and remembers blank node identifiers.
#[derive(Default)]
pub struct BlankNodeNamer {
    counter: usize,
    map: HashMap<String, String>,
}

impl BlankNodeNamer {
    /// Creates a fresh namer.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Returns the canonical blank node identifier for `identifier`, allocating a new one the
    /// first time a given identifier (or an anonymous node, when `identifier` is `None`) is seen.
    pub fn get(&mut self, identifier: Option<&str>) -> String {
        if let Some(identifier) = identifier {
            if let Some(existing) = self.map.get(identifier) {
                return existing.clone();
            }
        }
        let name = format!("_:b{}", self.counter);
        self.counter += 1;
        if let Some(identifier) = identifier {
            self.map.insert(identifier.to_owned(), name.clone());
        }
        name
    }
}

/// The active subject of a node-map generation step: either a subject IRI/blank id, or a node
/// reference to attach (for reverse properties).
#[derive(Clone)]
pub enum Subject {
    /// A subject identifier.
    Id(String),
    /// A node reference `{"@id": ...}` to add as a value (reverse-property case).
    Reference(JsonObject),
}

/// Runs [node map generation](https://www.w3.org/TR/json-ld-api/#node-map-generation) for
/// `element`, accumulating into `node_map` (a graph-name → subject-id → node object structure).
#[allow(clippy::too_many_arguments)]
pub fn generate_node_map(
    element: JsonValue,
    node_map: &mut JsonObject,
    active_graph: &str,
    active_subject: Option<&Subject>,
    active_property: Option<&str>,
    list: Option<&mut Vec<JsonValue>>,
    namer: &mut BlankNodeNamer,
    errors: &mut Vec<JsonLdSyntaxError>,
) {
    // 1) array
    if let JsonValue::Array(items) = element {
        let mut list = list;
        for item in items {
            generate_node_map(
                item,
                node_map,
                active_graph,
                active_subject,
                active_property,
                list.as_deref_mut(),
                namer,
                errors,
            );
        }
        return;
    }
    let JsonValue::Object(mut element) = element else {
        return;
    };
    graph_entry(node_map, active_graph);

    // 2) relabel blank node types
    if let Some(types) = element.get("@type") {
        let relabeled = relabel_types(types, namer);
        element.insert("@type", relabeled);
    }

    // 3) value object
    if element.contains_key("@value") {
        let value = JsonValue::Object(element);
        if let Some(list) = list {
            list.push(value);
        } else if let (Some(Subject::Id(subject)), Some(property)) =
            (active_subject, active_property)
        {
            let graph = graph_entry(node_map, active_graph);
            let node = node_entry(graph, subject);
            add_value_no_dup(node, property, value);
        }
        return;
    }

    // 4) list object
    if let Some(list_value) = element.remove("@list") {
        let mut result_list = Vec::new();
        generate_node_map(
            list_value,
            node_map,
            active_graph,
            active_subject,
            active_property,
            Some(&mut result_list),
            namer,
            errors,
        );
        let mut list_object = JsonObject::new();
        list_object.insert("@list", JsonValue::Array(result_list));
        let value = JsonValue::Object(list_object);
        if let Some(list) = list {
            list.push(value);
        } else if let (Some(Subject::Id(subject)), Some(property)) =
            (active_subject, active_property)
        {
            let graph = graph_entry(node_map, active_graph);
            let node = node_entry(graph, subject);
            // A list object is *appended*, never deduplicated: two structurally equal `@list`
            // objects are distinct RDF collections.
            append_value(node, property, value);
        }
        return;
    }

    // 5) node object
    let id = match element.remove("@id") {
        Some(JsonValue::String(id)) => {
            if let Some(blank) = id.strip_prefix("_:") {
                namer.get(Some(&format!("_:{blank}")))
            } else {
                id
            }
        }
        Some(_) => return, // invalid @id, drop (frame expansion only)
        None => namer.get(None),
    };

    {
        let graph = graph_entry(node_map, active_graph);
        if graph.get(&id).is_none() {
            let mut node = JsonObject::new();
            node.insert("@id", JsonValue::String(id.clone()));
            graph.insert(id.clone(), JsonValue::Object(node));
        }
    }

    // 6) subject handling
    match active_subject {
        Some(Subject::Reference(reference)) => {
            let graph = graph_entry(node_map, active_graph);
            let node = node_entry(graph, &id);
            if let Some(property) = active_property {
                add_value_no_dup(node, property, JsonValue::Object(reference.clone()));
            }
        }
        Some(Subject::Id(subject)) => {
            if let Some(property) = active_property {
                let mut reference = JsonObject::new();
                reference.insert("@id", JsonValue::String(id.clone()));
                let reference = JsonValue::Object(reference);
                if let Some(list) = list {
                    list.push(reference);
                } else {
                    let graph = graph_entry(node_map, active_graph);
                    let node = node_entry(graph, subject);
                    add_value_no_dup(node, property, reference);
                }
            }
        }
        None => {}
    }

    // 7) @type
    if let Some(types) = element.remove("@type") {
        let graph = graph_entry(node_map, active_graph);
        let node = node_entry(graph, &id);
        for t in to_array(types) {
            add_value_no_dup(node, "@type", t);
        }
    }

    // 8) @index
    if let Some(index) = element.remove("@index") {
        let graph = graph_entry(node_map, active_graph);
        let node = node_entry(graph, &id);
        if let Some(existing) = node.get("@index") {
            if *existing != index {
                errors.push(JsonLdSyntaxError::msg(
                    "conflicting @index values",
                    JsonLdErrorCode::ConflictingIndexes,
                ));
            }
        }
        node.insert("@index", index);
    }

    // 9) @reverse
    if let Some(JsonValue::Object(reverse_map)) = element.remove("@reverse") {
        let mut reference = JsonObject::new();
        reference.insert("@id", JsonValue::String(id.clone()));
        let referenced = Subject::Reference(reference);
        for (property, values) in reverse_map.into_entries() {
            for value in to_array(values) {
                generate_node_map(
                    value,
                    node_map,
                    active_graph,
                    Some(&referenced),
                    Some(&property),
                    None,
                    namer,
                    errors,
                );
            }
        }
    }

    // 10) @graph
    if let Some(graph_value) = element.remove("@graph") {
        generate_node_map(graph_value, node_map, &id, None, None, None, namer, errors);
    }

    // 11) @included
    if let Some(included) = element.remove("@included") {
        generate_node_map(
            included,
            node_map,
            active_graph,
            None,
            None,
            None,
            namer,
            errors,
        );
    }

    // 12) remaining properties
    let mut keys: Vec<String> = element.keys().map(str::to_owned).collect();
    keys.sort_unstable();
    for property in keys {
        let value = element.get(&property).cloned().unwrap_or(JsonValue::Null);
        let property = if let Some(blank) = property.strip_prefix("_:") {
            namer.get(Some(&format!("_:{blank}")))
        } else {
            property
        };
        {
            let graph = graph_entry(node_map, active_graph);
            let node = node_entry(graph, &id);
            if !node.contains_key(&property) {
                node.insert(&property, JsonValue::Array(Vec::new()));
            }
        }
        let subject = Subject::Id(id.clone());
        generate_node_map(
            value,
            node_map,
            active_graph,
            Some(&subject),
            Some(&property),
            None,
            namer,
            errors,
        );
    }
}

fn relabel_types(types: &JsonValue, namer: &mut BlankNodeNamer) -> JsonValue {
    match types {
        JsonValue::String(s) => JsonValue::String(relabel_one(s, namer)),
        JsonValue::Array(items) => JsonValue::Array(
            items
                .iter()
                .map(|item| match item {
                    JsonValue::String(s) => JsonValue::String(relabel_one(s, namer)),
                    other => other.clone(),
                })
                .collect(),
        ),
        other => other.clone(),
    }
}

fn relabel_one(value: &str, namer: &mut BlankNodeNamer) -> String {
    if value.starts_with("_:") {
        namer.get(Some(value))
    } else {
        value.to_owned()
    }
}

fn to_array(value: JsonValue) -> Vec<JsonValue> {
    match value {
        JsonValue::Array(items) => items,
        JsonValue::Null => Vec::new(),
        other => vec![other],
    }
}

fn graph_entry<'m>(node_map: &'m mut JsonObject, graph: &str) -> &'m mut JsonObject {
    if node_map.get(graph).and_then(JsonValue::as_object).is_none() {
        node_map.insert(graph.to_owned(), JsonValue::Object(JsonObject::new()));
    }
    match node_map.get_mut(graph).and_then(JsonValue::as_object_mut) {
        Some(object) => object,
        None => unreachable!("graph entry was just inserted"),
    }
}

fn node_entry<'m>(graph: &'m mut JsonObject, subject: &str) -> &'m mut JsonObject {
    if graph.get(subject).and_then(JsonValue::as_object).is_none() {
        let mut node = JsonObject::new();
        node.insert("@id", JsonValue::String(subject.to_owned()));
        graph.insert(subject.to_owned(), JsonValue::Object(node));
    }
    match graph.get_mut(subject).and_then(JsonValue::as_object_mut) {
        Some(object) => object,
        None => unreachable!("node entry was just inserted"),
    }
}

fn append_value(node: &mut JsonObject, key: &str, value: JsonValue) {
    match node.get_mut(key) {
        Some(JsonValue::Array(items)) => items.push(value),
        Some(slot) => {
            let old = std::mem::replace(slot, JsonValue::Null);
            *slot = JsonValue::Array(vec![old, value]);
        }
        None => {
            node.insert(key, JsonValue::Array(vec![value]));
        }
    }
}

fn add_value_no_dup(node: &mut JsonObject, key: &str, value: JsonValue) {
    match node.get_mut(key) {
        Some(JsonValue::Array(items)) => {
            if !items.contains(&value) {
                items.push(value);
            }
        }
        Some(slot) => {
            if *slot != value {
                let old = std::mem::replace(slot, JsonValue::Null);
                *slot = JsonValue::Array(vec![old, value]);
            }
        }
        None => {
            node.insert(key, JsonValue::Array(vec![value]));
        }
    }
}
