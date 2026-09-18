//! The [Flattening Algorithm](https://www.w3.org/TR/json-ld11-api/#flattening-algorithm):
//! collects an expanded document into a single flat array of node objects.
//!
//! Flattening builds a [node map](crate::node_map) — one entry per graph, one entry per subject —
//! and then serialises it back: named graphs become `@graph` entries of the node that names them,
//! and nodes that carry nothing but their identifier are dropped.

use crate::error::JsonLdSyntaxError;
use crate::node_map::{BlankNodeNamer, generate_node_map};
use crate::value::{JsonObject, JsonValue};

/// Flattens an already-expanded document into the expanded flattened form (an array of node
/// objects).
///
/// # Errors
///
/// Returns the first node-map generation error, such as
/// [`JsonLdErrorCode::ConflictingIndexes`](crate::JsonLdErrorCode::ConflictingIndexes).
pub fn flatten_expanded(
    expanded: JsonValue,
    ordered: bool,
) -> Result<JsonValue, JsonLdSyntaxError> {
    // 1-2)
    let mut node_map = JsonObject::new();
    node_map.insert("@default", JsonValue::Object(JsonObject::new()));
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
    if !errors.is_empty() {
        return Err(errors.remove(0));
    }

    // 3)
    let mut default_graph = node_map
        .get("@default")
        .and_then(JsonValue::as_object)
        .cloned()
        .unwrap_or_default();

    // 4) every named graph becomes a `@graph` entry of the node that names it
    let mut graph_names: Vec<String> = node_map
        .keys()
        .filter(|name| *name != "@default")
        .map(str::to_owned)
        .collect();
    if ordered {
        graph_names.sort_unstable();
    }
    for graph_name in graph_names {
        if !default_graph.contains_key(&graph_name) {
            let mut entry = JsonObject::new();
            entry.insert("@id", JsonValue::String(graph_name.clone()));
            default_graph.insert(graph_name.clone(), JsonValue::Object(entry));
        }
        let graph = node_map
            .get(&graph_name)
            .and_then(JsonValue::as_object)
            .cloned()
            .unwrap_or_default();
        let mut subjects: Vec<String> = graph.keys().map(str::to_owned).collect();
        if ordered {
            subjects.sort_unstable();
        }
        let mut nodes = Vec::new();
        for subject in subjects {
            if let Some(node) = graph.get(&subject).and_then(JsonValue::as_object) {
                if !is_bare_reference(node) {
                    nodes.push(JsonValue::Object(node.clone()));
                }
            }
        }
        if let Some(JsonValue::Object(entry)) = default_graph.get_mut(&graph_name) {
            entry.insert("@graph", JsonValue::Array(nodes));
        }
    }

    // 5-7)
    let mut subjects: Vec<String> = default_graph.keys().map(str::to_owned).collect();
    if ordered {
        subjects.sort_unstable();
    }
    let mut result = Vec::new();
    for subject in subjects {
        if let Some(node) = default_graph.get(&subject).and_then(JsonValue::as_object) {
            if !is_bare_reference(node) {
                result.push(JsonValue::Object(node.clone()));
            }
        }
    }
    Ok(JsonValue::Array(result))
}

/// A node that carries nothing but its identifier contributes no statement.
fn is_bare_reference(node: &JsonObject) -> bool {
    node.len() == 1 && node.contains_key("@id")
}
