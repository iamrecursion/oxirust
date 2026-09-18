//! [Serialize RDF as JSON-LD](https://www.w3.org/TR/json-ld11-api/#serialize-rdf-as-json-ld-algorithm)
//! (the *fromRdf* operation) and the streaming [`JsonLdSerializer`].
//!
//! The algorithm turns a dataset back into an expanded JSON-LD document: quads are collected into a
//! per-graph node map, `rdf:first`/`rdf:rest` chains are folded back into `@list` objects, and
//! named graphs become `@graph` entries of the node that names them. Step numbers in the comments
//! refer to [§8.4](https://www.w3.org/TR/json-ld11-api/#serialize-rdf-as-json-ld-algorithm) and
//! [§8.5](https://www.w3.org/TR/json-ld11-api/#rdf-to-object-conversion) of the JSON-LD 1.1 API.

use crate::api::JsonLdOptions;
use crate::context::has_keyword_form;
use crate::error::{JsonLdErrorCode, JsonLdParseError, JsonLdSyntaxError};
use crate::to_rdf::RdfDirection;
use crate::value::{JsonObject, JsonValue};
use oxixml_iri::Iri;
use oxixml_model::vocab::xsd;
use oxixml_model::{
    GraphName, GraphNameRef, Literal, NamedOrBlankNode, Quad, QuadRef, Term, TripleRef,
};
use std::collections::HashMap;
use std::io::{self, Write};
#[cfg(feature = "async-tokio")]
use tokio::io::{AsyncWrite, AsyncWriteExt};

const RDF_FIRST: &str = "http://www.w3.org/1999/02/22-rdf-syntax-ns#first";
const RDF_REST: &str = "http://www.w3.org/1999/02/22-rdf-syntax-ns#rest";
const RDF_NIL: &str = "http://www.w3.org/1999/02/22-rdf-syntax-ns#nil";
const RDF_TYPE: &str = "http://www.w3.org/1999/02/22-rdf-syntax-ns#type";
const RDF_LIST: &str = "http://www.w3.org/1999/02/22-rdf-syntax-ns#List";
const RDF_JSON: &str = "http://www.w3.org/1999/02/22-rdf-syntax-ns#JSON";
const RDF_VALUE: &str = "http://www.w3.org/1999/02/22-rdf-syntax-ns#value";
const RDF_LANGUAGE: &str = "http://www.w3.org/1999/02/22-rdf-syntax-ns#language";
const RDF_DIRECTION: &str = "http://www.w3.org/1999/02/22-rdf-syntax-ns#direction";
const I18N_PREFIX: &str = "https://www.w3.org/ns/i18n#";

/// Where a node reference sits inside the node map: `node[property][value_index]`.
#[derive(Clone, Debug)]
struct Usage {
    node: String,
    property: String,
    value_index: usize,
}

/// Serializes the RDF dataset `quads` as an expanded JSON-LD document.
///
/// # Errors
///
/// Returns [`JsonLdErrorCode::InvalidJsonLiteral`] for a malformed `rdf:JSON` literal, and the
/// language/direction errors of the `compound-literal` representation.
pub fn from_rdf_quads(
    quads: &[Quad],
    options: &JsonLdOptions<'_>,
) -> Result<JsonValue, JsonLdSyntaxError> {
    // 1-4)
    let mut graph_map = JsonObject::new();
    graph_map.insert("@default", JsonValue::Object(JsonObject::new()));
    let mut usages: HashMap<String, Vec<Usage>> = HashMap::new();
    // `referenced once` is deliberately *not* per graph: a node reachable from two graphs is
    // referenced more than once and can therefore never be a list node.
    let mut referenced_once: HashMap<String, Option<Usage>> = HashMap::new();
    let mut compound_literal_subjects: HashMap<String, Vec<String>> = HashMap::new();

    // A dataset is a set: a repeated statement must not look like a second reference.
    let mut seen = std::collections::HashSet::new();
    let quads: Vec<&Quad> = quads.iter().filter(|quad| seen.insert(*quad)).collect();

    // 5) collect every quad into the node map
    for quad in quads {
        let graph_name = match quad.graph_name.as_ref() {
            GraphNameRef::DefaultGraph => "@default".to_owned(),
            GraphNameRef::NamedNode(node) => node.as_str().to_owned(),
            GraphNameRef::BlankNode(node) => format!("_:{}", node.as_str()),
        };
        ensure_graph(&mut graph_map, &graph_name);
        if graph_name != "@default" {
            ensure_node(&mut graph_map, "@default", &graph_name);
        }
        let subject = named_or_blank_id(&quad.subject);
        let property = quad.predicate.as_str().to_owned();
        ensure_node(&mut graph_map, &graph_name, &subject);
        // 5.7.3)
        if options.rdf_direction == RdfDirection::CompoundLiteral && property == RDF_DIRECTION {
            let subjects = compound_literal_subjects
                .entry(graph_name.clone())
                .or_default();
            if !subjects.contains(&subject) {
                subjects.push(subject.clone());
            }
        }
        let object_id = term_id(&quad.object);
        // 5.7.4)
        if let Some(object_id) = &object_id {
            ensure_node(&mut graph_map, &graph_name, object_id);
        }
        // 5.7.5)
        if property == RDF_TYPE && !options.use_rdf_type {
            if let Some(object_id) = &object_id {
                let node = node_mut(&mut graph_map, &graph_name, &subject);
                add_unique(node, "@type", JsonValue::String(object_id.clone()));
                continue;
            }
        }
        // 5.7.6-5.7.8)
        let value = rdf_to_object(&quad.object, options)?;
        let value_index = {
            let node = node_mut(&mut graph_map, &graph_name, &subject);
            if !node.get(&property).is_some_and(JsonValue::is_array) {
                node.insert(property.clone(), JsonValue::Array(Vec::new()));
            }
            match node.get_mut(&property) {
                Some(JsonValue::Array(items)) => match items.iter().position(|item| *item == value)
                {
                    Some(index) => index,
                    None => {
                        items.push(value);
                        items.len() - 1
                    }
                },
                _ => 0,
            }
        };
        let usage = Usage {
            node: subject.clone(),
            property: property.clone(),
            value_index,
        };
        // 5.7.9-5.7.11)
        match object_id.as_deref() {
            Some(RDF_NIL) => usages.entry(graph_name.clone()).or_default().push(usage),
            Some(object_id) => match referenced_once.entry(object_id.to_owned()) {
                std::collections::hash_map::Entry::Occupied(mut entry) => {
                    entry.insert(None);
                }
                std::collections::hash_map::Entry::Vacant(entry) => {
                    if object_id.starts_with("_:") {
                        entry.insert(Some(usage));
                    }
                }
            },
            None => {}
        }
    }

    // 6) per-graph clean-up: compound literals, then list reconstruction
    let graph_names: Vec<String> = graph_map.keys().map(str::to_owned).collect();
    for graph_name in graph_names {
        if let Some(subjects) = compound_literal_subjects.get(&graph_name) {
            for subject in subjects {
                rebuild_compound_literal(&mut graph_map, &graph_name, subject, &referenced_once)?;
            }
        }
        rebuild_lists(&mut graph_map, &graph_name, &usages, &referenced_once);
    }

    // 7-8) assemble the flattened document
    let mut result = Vec::new();
    let default_graph = graph_map
        .get("@default")
        .and_then(JsonValue::as_object)
        .cloned()
        .unwrap_or_default();
    let mut subjects: Vec<String> = default_graph.keys().map(str::to_owned).collect();
    if options.ordered {
        subjects.sort_unstable();
    }
    for subject in subjects {
        let Some(mut node) = default_graph
            .get(&subject)
            .and_then(JsonValue::as_object)
            .cloned()
        else {
            continue;
        };
        if let Some(graph) = graph_map.get(&subject).and_then(JsonValue::as_object) {
            let mut inner_subjects: Vec<String> = graph.keys().map(str::to_owned).collect();
            if options.ordered {
                inner_subjects.sort_unstable();
            }
            let mut nodes = Vec::new();
            for inner in inner_subjects {
                if let Some(inner_node) = graph.get(&inner).and_then(JsonValue::as_object) {
                    if !is_bare_reference(inner_node) {
                        nodes.push(JsonValue::Object(inner_node.clone()));
                    }
                }
            }
            node.insert("@graph", JsonValue::Array(nodes));
        }
        if !is_bare_reference(&node) {
            result.push(JsonValue::Object(node));
        }
    }
    Ok(JsonValue::Array(result))
}

/// A node that carries nothing but its identifier contributes no statement.
fn is_bare_reference(node: &JsonObject) -> bool {
    node.len() == 1 && node.contains_key("@id")
}

/// One list to rebuild: where its head reference sits, and which nodes formed its spine.
struct ListConversion {
    head: Usage,
    nodes: Vec<String>,
}

/// Step 6.4: folds `rdf:first`/`rdf:rest` chains back into `@list` objects.
///
/// Conversions are collected first and then applied innermost-first: the head of a nested list is
/// the `rdf:first` value of an enclosing list node, so the inner `@list` must be in place before
/// the outer list reads its items.
fn rebuild_lists(
    graph_map: &mut JsonObject,
    graph_name: &str,
    usages: &HashMap<String, Vec<Usage>>,
    referenced_once: &HashMap<String, Option<Usage>>,
) {
    let Some(nil_usages) = usages.get(graph_name) else {
        return;
    };
    let mut conversions = Vec::new();
    for usage in nil_usages {
        let mut node_id = usage.node.clone();
        let mut property = usage.property.clone();
        let mut head = usage.clone();
        let mut nodes: Vec<String> = Vec::new();
        // 6.4.3) walk the spine backwards towards the head of the collection
        loop {
            if property != RDF_REST || !node_id.starts_with("_:") {
                break;
            }
            let Some(Some(node_usage)) = referenced_once.get(&node_id) else {
                break;
            };
            let Some(node) = get_node(graph_map, graph_name, &node_id) else {
                break;
            };
            if !is_well_formed_list_node(node) {
                break;
            }
            nodes.push(node_id.clone());
            head = node_usage.clone();
            node_id = node_usage.node.clone();
            property = node_usage.property.clone();
            if !node_id.starts_with("_:") {
                break;
            }
        }
        conversions.push(ListConversion { head, nodes });
    }

    let order = innermost_first(&conversions);
    let mut removals: Vec<String> = Vec::new();
    for index in order {
        let Some(conversion) = conversions.get(index) else {
            continue;
        };
        // 6.4.1-6.4.6) collect the items now, so that any nested list already rebuilt is picked up
        let mut list = Vec::new();
        for node_id in &conversion.nodes {
            let item = get_node(graph_map, graph_name, node_id)
                .and_then(|node| node.get(RDF_FIRST))
                .and_then(JsonValue::as_array)
                .and_then(<[JsonValue]>::first)
                .cloned();
            match item {
                Some(item) => list.push(item),
                None => break,
            }
        }
        list.reverse();
        let mut list_object = JsonObject::new();
        list_object.insert("@list", JsonValue::Array(list));
        let head = conversion.head.clone();
        if let Some(node) = get_node_mut(graph_map, graph_name, &head.node) {
            if let Some(JsonValue::Array(values)) = node.get_mut(&head.property) {
                if let Some(slot) = values.get_mut(head.value_index) {
                    *slot = JsonValue::Object(list_object);
                }
            }
        }
        removals.extend(conversion.nodes.iter().cloned());
    }
    // 6.4.7) the spine nodes are no longer part of the graph
    if let Some(graph) = graph_map
        .get_mut(graph_name)
        .and_then(JsonValue::as_object_mut)
    {
        for node_id in removals {
            graph.remove(&node_id);
        }
    }
}

/// Orders conversions so that a list whose head lives inside another list's spine comes first.
fn innermost_first(conversions: &[ListConversion]) -> Vec<usize> {
    let mut order = Vec::with_capacity(conversions.len());
    let mut pending: Vec<usize> = (0..conversions.len()).collect();
    while !pending.is_empty() {
        let ready: Vec<usize> = pending
            .iter()
            .copied()
            .filter(|index| {
                let Some(conversion) = conversions.get(*index) else {
                    return true;
                };
                !pending.iter().any(|other| {
                    *other != *index
                        && conversions
                            .get(*other)
                            .is_some_and(|other| conversion.nodes.contains(&other.head.node))
                })
            })
            .collect();
        if ready.is_empty() {
            // A cycle among list spines cannot occur in well-formed data; fall back to input order.
            order.append(&mut pending);
            break;
        }
        pending.retain(|index| !ready.contains(index));
        order.extend(ready);
    }
    order
}

fn is_well_formed_list_node(node: &JsonObject) -> bool {
    let single =
        |key: &str| matches!(node.get(key), Some(JsonValue::Array(items)) if items.len() == 1);
    if !single(RDF_FIRST) || !single(RDF_REST) {
        return false;
    }
    for key in node.keys() {
        if !matches!(key, "@id" | "@type") && key != RDF_FIRST && key != RDF_REST {
            return false;
        }
    }
    match node.get("@type") {
        None => true,
        Some(JsonValue::Array(types)) => {
            types.len() == 1 && types.first().and_then(JsonValue::as_str) == Some(RDF_LIST)
        }
        Some(_) => false,
    }
}

/// Step 6.1: rebuilds a `compound-literal` blank node into a value object.
fn rebuild_compound_literal(
    graph_map: &mut JsonObject,
    graph_name: &str,
    subject: &str,
    referenced_once: &HashMap<String, Option<Usage>>,
) -> Result<(), JsonLdSyntaxError> {
    let Some(Some(usage)) = referenced_once.get(subject) else {
        return Ok(());
    };
    let Some(compound_node) = graph_map
        .get_mut(graph_name)
        .and_then(JsonValue::as_object_mut)
        .and_then(|graph| graph.remove(subject))
    else {
        return Ok(());
    };
    let Some(compound_node) = compound_node.as_object() else {
        return Ok(());
    };
    let literal = first_value(compound_node, RDF_VALUE);
    let language = first_value(compound_node, RDF_LANGUAGE);
    let direction = first_value(compound_node, RDF_DIRECTION);
    if let Some(direction) = &direction {
        if !matches!(direction.as_str(), "ltr" | "rtl") {
            return Err(JsonLdSyntaxError::msg(
                format!("{direction} is not a valid base direction"),
                JsonLdErrorCode::InvalidBaseDirection,
            ));
        }
    }
    if let Some(language) = &language {
        if Literal::new_language_tagged_literal("x", language.clone()).is_err() {
            return Err(JsonLdSyntaxError::msg(
                format!("{language} is not a well-formed language tag"),
                JsonLdErrorCode::InvalidLanguageTaggedString,
            ));
        }
    }
    let Some(node) = get_node_mut(graph_map, graph_name, &usage.node) else {
        return Ok(());
    };
    let Some(JsonValue::Array(values)) = node.get_mut(&usage.property) else {
        return Ok(());
    };
    for value in values.iter_mut() {
        let Some(reference) = value.as_object_mut() else {
            continue;
        };
        if reference.get("@id").and_then(JsonValue::as_str) != Some(subject) {
            continue;
        }
        reference.remove("@id");
        reference.insert(
            "@value",
            JsonValue::String(literal.clone().unwrap_or_default()),
        );
        if let Some(language) = &language {
            reference.insert("@language", JsonValue::String(language.clone()));
        }
        if let Some(direction) = &direction {
            reference.insert("@direction", JsonValue::String(direction.clone()));
        }
    }
    Ok(())
}

fn first_value(node: &JsonObject, key: &str) -> Option<String> {
    let JsonValue::Array(items) = node.get(key)? else {
        return None;
    };
    items
        .first()?
        .as_object()?
        .get("@value")?
        .as_str()
        .map(str::to_owned)
}

/// The [RDF to Object Conversion](https://www.w3.org/TR/json-ld11-api/#rdf-to-object-conversion)
/// algorithm.
fn rdf_to_object(term: &Term, options: &JsonLdOptions<'_>) -> Result<JsonValue, JsonLdSyntaxError> {
    match term {
        Term::NamedNode(node) => Ok(node_reference(node.as_str())),
        Term::BlankNode(node) => Ok(node_reference(&format!("_:{}", node.as_str()))),
        Term::Literal(literal) => literal_to_object(literal, options),
        // `Term::Triple` only exists when `oxixml-model` is built with its `rdf-12` feature, which
        // another crate of the build may switch on. JSON-LD 1.1 has no syntax for an RDF 1.2
        // triple term, so the statement is rejected rather than silently dropped.
        #[allow(
            unreachable_patterns,
            reason = "only reachable when oxixml-model/rdf-12 is enabled elsewhere in the build"
        )]
        _ => Err(JsonLdSyntaxError::msg(
            "JSON-LD 1.1 cannot represent an RDF 1.2 triple term",
            JsonLdErrorCode::UnsupportedTripleTerm,
        )),
    }
}

fn node_reference(id: &str) -> JsonValue {
    let mut object = JsonObject::new();
    object.insert("@id", JsonValue::String(id.to_owned()));
    JsonValue::Object(object)
}

fn literal_to_object(
    literal: &Literal,
    options: &JsonLdOptions<'_>,
) -> Result<JsonValue, JsonLdSyntaxError> {
    let mut result = JsonObject::new();
    let datatype = literal.datatype().as_str().to_owned();
    let lexical = literal.value().to_owned();
    let mut converted = JsonValue::String(lexical.clone());
    let mut value_type: Option<String> = None;

    // 2.4) native types are only used when the lexical form really is in the datatype's lexical
    // space; anything else falls through to the generic handling below.
    let mut natively_converted = false;
    if options.use_native_types {
        if datatype == xsd::STRING.as_str() {
            natively_converted = true;
        } else if datatype == xsd::BOOLEAN.as_str() {
            match lexical.as_str() {
                "true" | "1" => {
                    converted = JsonValue::Bool(true);
                    natively_converted = true;
                }
                "false" | "0" => {
                    converted = JsonValue::Bool(false);
                    natively_converted = true;
                }
                _ => {}
            }
        } else if datatype == xsd::INTEGER.as_str() && is_xsd_integer(&lexical) {
            converted = JsonValue::Number(canonical_integer(&lexical));
            natively_converted = true;
        } else if datatype == xsd::DOUBLE.as_str() && is_xsd_double(&lexical) {
            converted = JsonValue::Number(json_number(&lexical));
            natively_converted = true;
        }
    }

    if natively_converted {
        // nothing more to determine: the value is a native JSON scalar
    } else if datatype == RDF_JSON {
        converted = JsonValue::from_slice(lexical.as_bytes()).map_err(|e| {
            JsonLdSyntaxError::msg(
                format!("{lexical} is not valid JSON: {e}"),
                JsonLdErrorCode::InvalidJsonLiteral,
            )
        })?;
        value_type = Some("@json".to_owned());
    } else if options.rdf_direction == RdfDirection::I18nDatatype
        && datatype.starts_with(I18N_PREFIX)
    {
        converted = JsonValue::String(lexical);
        value_type = None;
        if let Some(fragment) = datatype.strip_prefix(I18N_PREFIX) {
            let (language, direction) = match fragment.split_once('_') {
                Some((language, direction)) => (language, direction),
                None => ("", fragment),
            };
            if !language.is_empty() {
                result.insert("@language", JsonValue::String(language.to_owned()));
            }
            result.insert("@direction", JsonValue::String(direction.to_owned()));
        }
    } else if let Some(language) = literal.language() {
        result.insert("@language", JsonValue::String(language.to_owned()));
        value_type = None;
    } else if datatype != xsd::STRING.as_str() {
        value_type = Some(datatype);
    }

    result.insert("@value", converted);
    if let Some(value_type) = value_type {
        result.insert("@type", JsonValue::String(value_type));
    }
    Ok(JsonValue::Object(result))
}

fn is_xsd_integer(lexical: &str) -> bool {
    let digits = lexical.strip_prefix(['-', '+']).unwrap_or(lexical);
    !digits.is_empty() && digits.bytes().all(|byte| byte.is_ascii_digit())
}

/// Whether `lexical` is an `xsd:double` that JSON can represent: `INF`, `-INF`, `NaN`, and values
/// that overflow to infinity have no JSON number form and stay typed literals.
fn is_xsd_double(lexical: &str) -> bool {
    lexical.parse::<f64>().is_ok_and(f64::is_finite)
}

fn canonical_integer(lexical: &str) -> String {
    match lexical.parse::<i128>() {
        Ok(number) => number.to_string(),
        Err(_) => lexical.to_owned(),
    }
}

/// Renders an `xsd:double` lexical form as the JSON number JSON-LD round-trips to.
fn json_number(lexical: &str) -> String {
    let Ok(number) = lexical.parse::<f64>() else {
        return lexical.to_owned();
    };
    if number.fract() == 0.0 && number.abs() < 1e21 {
        format!("{}E0", number as i128)
    } else {
        crate::to_rdf::canonical_double(number)
    }
}

fn named_or_blank_id(node: &NamedOrBlankNode) -> String {
    match node {
        NamedOrBlankNode::NamedNode(node) => node.as_str().to_owned(),
        NamedOrBlankNode::BlankNode(node) => format!("_:{}", node.as_str()),
    }
}

fn term_id(term: &Term) -> Option<String> {
    match term {
        Term::NamedNode(node) => Some(node.as_str().to_owned()),
        Term::BlankNode(node) => Some(format!("_:{}", node.as_str())),
        // Literals and RDF 1.2 triple terms never name a node in the node map.
        Term::Literal(_) => None,
        #[allow(
            unreachable_patterns,
            reason = "only reachable when oxixml-model/rdf-12 is enabled elsewhere in the build"
        )]
        _ => None,
    }
}

fn ensure_graph(graph_map: &mut JsonObject, graph: &str) {
    if !graph_map.contains_key(graph) {
        graph_map.insert(graph.to_owned(), JsonValue::Object(JsonObject::new()));
    }
}

fn ensure_node(graph_map: &mut JsonObject, graph: &str, subject: &str) {
    ensure_graph(graph_map, graph);
    if let Some(JsonValue::Object(nodes)) = graph_map.get_mut(graph) {
        if !nodes.contains_key(subject) {
            let mut node = JsonObject::new();
            node.insert("@id", JsonValue::String(subject.to_owned()));
            nodes.insert(subject.to_owned(), JsonValue::Object(node));
        }
    }
}

fn node_mut<'m>(graph_map: &'m mut JsonObject, graph: &str, subject: &str) -> &'m mut JsonObject {
    ensure_node(graph_map, graph, subject);
    match get_node_mut(graph_map, graph, subject) {
        Some(node) => node,
        None => unreachable!("the node was just ensured"),
    }
}

fn get_node<'m>(graph_map: &'m JsonObject, graph: &str, subject: &str) -> Option<&'m JsonObject> {
    graph_map
        .get(graph)
        .and_then(JsonValue::as_object)
        .and_then(|nodes| nodes.get(subject))
        .and_then(JsonValue::as_object)
}

fn get_node_mut<'m>(
    graph_map: &'m mut JsonObject,
    graph: &str,
    subject: &str,
) -> Option<&'m mut JsonObject> {
    graph_map
        .get_mut(graph)
        .and_then(JsonValue::as_object_mut)
        .and_then(|nodes| nodes.get_mut(subject))
        .and_then(JsonValue::as_object_mut)
}

fn add_unique(node: &mut JsonObject, key: &str, value: JsonValue) {
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

/// Builds the `@context` object a serializer's options call for, or `None` when they call for
/// none.
///
/// The layout is `oxjsonld`'s: `@base` first, then one entry per prefix in name order, with the
/// empty prefix name written as `@vocab`. `oxjsonld` holds its prefixes in a `BTreeMap`, so a name
/// declared twice keeps only its last IRI; the same happens here.
fn build_context(base: Option<&str>, prefixes: &[(String, String)]) -> Option<JsonObject> {
    if base.is_none() && prefixes.is_empty() {
        return None;
    }
    let mut context = JsonObject::new();
    if let Some(base) = base {
        context.insert("@base", JsonValue::String(base.to_owned()));
    }
    let mut declared: Vec<(&str, &str)> = Vec::with_capacity(prefixes.len());
    for (name, iri) in prefixes {
        match declared.iter_mut().find(|(seen, _)| *seen == name.as_str()) {
            Some(slot) => slot.1 = iri.as_str(),
            None => declared.push((name.as_str(), iri.as_str())),
        }
    }
    declared.sort_unstable_by(|left, right| left.0.cmp(right.0));
    for (name, iri) in declared {
        context.insert(
            if name.is_empty() { "@vocab" } else { name },
            JsonValue::String(iri.to_owned()),
        );
    }
    Some(context)
}

/// Rewrites `iri` relative to `base`, or returns `None` when the shorter form would not read back
/// as the same IRI.
///
/// The guards are `oxjsonld`'s: a relative reference that expansion would take for a blank node
/// (`_:…`), for an absolute IRI (`scheme://…`) or for a keyword (`@id`) has to stay absolute.
fn relativize_iri(base: &Iri<String>, iri: &str) -> Option<String> {
    let target = Iri::parse(iri.to_owned()).ok()?;
    let relative = base.relativize(&target).ok()?.into_inner();
    if relative
        .split_once(':')
        .is_some_and(|(prefix, suffix)| prefix == "_" || suffix.starts_with("//"))
        || has_keyword_form(&relative)
    {
        return None;
    }
    Some(relative)
}

/// Makes every node identifier in `value` relative to `base`.
///
/// Only `@id` is rewritten. Blank node labels are left alone, and so is `@type`: expansion resolves
/// a type against `@vocab` rather than `@base`, so a relative type would not survive a round trip.
/// `oxjsonld` draws the same line between its `id_value` and `type_value` helpers.
fn relativize_ids(value: &mut JsonValue, base: &Iri<String>) {
    match value {
        JsonValue::Array(items) => {
            for item in items {
                relativize_ids(item, base);
            }
        }
        JsonValue::Object(object) => {
            for (key, child) in object.iter_mut() {
                match key {
                    "@id" => {
                        if let JsonValue::String(iri) = child
                            && !iri.starts_with("_:")
                            && let Some(relative) = relativize_iri(base, iri)
                        {
                            *iri = relative;
                        }
                    }
                    "@value" | "@type" | "@language" | "@direction" | "@index" => {}
                    _ => relativize_ids(child, base),
                }
            }
        }
        _ => {}
    }
}

/// Runs *fromRdf* over `quads` and lays the result out as the document a serializer writes.
///
/// With neither a base IRI nor a prefix declared, this is the bare expanded array the *fromRdf*
/// algorithm produces. As soon as either is set, the array is wrapped in the
/// `{"@context": …, "@graph": […]}` envelope `oxjsonld` writes, and node identifiers are made
/// relative to the base.
fn serialized_document(
    quads: &[Quad],
    base: Option<&str>,
    prefixes: &[(String, String)],
) -> Result<JsonValue, JsonLdSyntaxError> {
    let options = JsonLdOptions {
        base: base.map(str::to_owned),
        ..JsonLdOptions::new()
    };
    let mut document = from_rdf_quads(quads, &options)?;
    let Some(context) = build_context(base, prefixes) else {
        return Ok(document);
    };
    if let Some(base) = base
        && let Ok(base) = Iri::parse(base.to_owned())
    {
        relativize_ids(&mut document, &base);
    }
    let mut envelope = JsonObject::new();
    envelope.insert("@context", JsonValue::Object(context));
    envelope.insert("@graph", document);
    Ok(JsonValue::Object(envelope))
}

/// A streaming [JSON-LD](https://www.w3.org/TR/json-ld/) serializer.
///
/// It buffers the fed quads, runs the *fromRdf* algorithm, and writes an expanded JSON-LD document
/// on [`finish`](WriterJsonLdSerializer::finish).
///
/// [`with_prefix`](Self::with_prefix) and [`with_base_iri`](Self::with_base_iri) put an
/// `@context` on that document and wrap it in a `@graph`, matching `oxjsonld`; see those methods
/// for the exact shape.
#[derive(Default)]
pub struct JsonLdSerializer {
    prefixes: Vec<(String, String)>,
    base: Option<String>,
}

impl JsonLdSerializer {
    /// Creates a new serializer.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Registers a prefix.
    ///
    /// Declaring a prefix puts an `@context` on the written document and wraps the nodes in a
    /// `@graph`, so the output becomes `{"@context": {…}, "@graph": […]}`. The prefix appears as a
    /// term definition; property keys themselves stay expanded IRIs, which is what `oxjsonld`
    /// writes too. The empty prefix name is written as `@vocab`.
    ///
    /// ```
    /// use oxixml_jsonld::JsonLdSerializer;
    /// use oxixml_model::{NamedNodeRef, QuadRef};
    ///
    /// let mut serializer = JsonLdSerializer::new()
    ///     .with_prefix("schema", "http://schema.org/")?
    ///     .for_writer(Vec::new());
    /// serializer.serialize_quad(QuadRef::new(
    ///     NamedNodeRef::new("http://example.com#me")?,
    ///     NamedNodeRef::new("http://schema.org/name")?,
    ///     NamedNodeRef::new("http://example.com#name")?,
    ///     oxixml_model::GraphNameRef::DefaultGraph,
    /// ))?;
    /// assert_eq!(
    ///     serializer.finish()?,
    ///     br#"{"@context":{"schema":"http://schema.org/"},"@graph":[{"@id":"http://example.com#me","http://schema.org/name":[{"@id":"http://example.com#name"}]}]}"#
    /// );
    /// # Result::<_, Box<dyn std::error::Error>>::Ok(())
    /// ```
    ///
    /// # Errors
    ///
    /// Returns [`oxixml_iri::IriParseError`] if `iri` is not a valid IRI.
    pub fn with_prefix(
        mut self,
        prefix: impl Into<String>,
        iri: impl Into<String>,
    ) -> Result<Self, oxixml_iri::IriParseError> {
        let iri = iri.into();
        oxixml_iri::Iri::parse(iri.as_str())?;
        self.prefixes.push((prefix.into(), iri));
        Ok(self)
    }

    /// The prefixes registered with [`with_prefix`](JsonLdSerializer::with_prefix).
    #[must_use]
    pub fn prefixes(&self) -> &[(String, String)] {
        &self.prefixes
    }

    /// Sets the base IRI.
    ///
    /// The base is written as `@base` in the document's `@context`, and node identifiers are made
    /// relative to it, so the output becomes `{"@context": {"@base": …}, "@graph": […]}`. An
    /// identifier that would read back as something else — a blank node, an absolute IRI or a
    /// keyword — is left absolute. `@type` is never made relative: expansion resolves a type
    /// against `@vocab`, not against `@base`.
    ///
    /// ```
    /// use oxixml_jsonld::JsonLdSerializer;
    /// use oxixml_model::{GraphNameRef, NamedNodeRef, QuadRef};
    ///
    /// let mut serializer = JsonLdSerializer::new()
    ///     .with_base_iri("http://example.com/")?
    ///     .for_writer(Vec::new());
    /// serializer.serialize_quad(QuadRef::new(
    ///     NamedNodeRef::new("http://example.com/s")?,
    ///     NamedNodeRef::new("http://example.com/p")?,
    ///     NamedNodeRef::new("http://example.com/o")?,
    ///     GraphNameRef::DefaultGraph,
    /// ))?;
    /// assert_eq!(
    ///     serializer.finish()?,
    ///     br#"{"@context":{"@base":"http://example.com/"},"@graph":[{"@id":"s","http://example.com/p":[{"@id":"o"}]}]}"#
    /// );
    /// # Result::<_, Box<dyn std::error::Error>>::Ok(())
    /// ```
    ///
    /// # Errors
    ///
    /// Returns [`oxixml_iri::IriParseError`] if `base_iri` is not a valid IRI.
    pub fn with_base_iri(
        mut self,
        base_iri: impl Into<String>,
    ) -> Result<Self, oxixml_iri::IriParseError> {
        let base_iri = base_iri.into();
        oxixml_iri::Iri::parse(base_iri.as_str())?;
        self.base = Some(base_iri);
        Ok(self)
    }

    /// Begins serializing to `writer`.
    pub fn for_writer<W: Write>(self, writer: W) -> WriterJsonLdSerializer<W> {
        WriterJsonLdSerializer {
            writer,
            quads: Vec::new(),
            base: self.base,
            prefixes: self.prefixes,
        }
    }

    /// Begins serializing to an asynchronous `writer`.
    ///
    /// This is the asynchronous counterpart of [`for_writer`](Self::for_writer). The *fromRdf*
    /// algorithm needs the whole dataset before it can lay out the document, so — exactly as in
    /// the blocking path — quads are buffered and the JSON is produced by the same core on
    /// [`finish`](TokioAsyncWriterJsonLdSerializer::finish); only the final write is awaited.
    #[cfg(feature = "async-tokio")]
    pub fn for_tokio_async_writer<W: AsyncWrite + Unpin>(
        self,
        writer: W,
    ) -> TokioAsyncWriterJsonLdSerializer<W> {
        TokioAsyncWriterJsonLdSerializer {
            writer,
            quads: Vec::new(),
            base: self.base,
            prefixes: self.prefixes,
        }
    }
}

/// A [`JsonLdSerializer`] bound to an [`AsyncWrite`] target.
#[cfg(feature = "async-tokio")]
#[must_use]
pub struct TokioAsyncWriterJsonLdSerializer<W: AsyncWrite + Unpin> {
    writer: W,
    quads: Vec<Quad>,
    base: Option<String>,
    prefixes: Vec<(String, String)>,
}

#[cfg(feature = "async-tokio")]
impl<W: AsyncWrite + Unpin> TokioAsyncWriterJsonLdSerializer<W> {
    /// Buffers a quad for serialization.
    ///
    /// # Errors
    ///
    /// Never fails; the signature mirrors the other OxiXML serializers.
    #[expect(
        clippy::unused_async,
        reason = "mirrors the upstream async serializer API"
    )]
    pub async fn serialize_quad<'a>(&mut self, quad: impl Into<QuadRef<'a>>) -> io::Result<()> {
        self.quads.push(quad.into().into_owned());
        Ok(())
    }

    /// Buffers a triple (in the default graph) for serialization.
    ///
    /// # Errors
    ///
    /// Never fails; the signature mirrors the other OxiXML serializers.
    #[expect(
        clippy::unused_async,
        reason = "mirrors the upstream async serializer API"
    )]
    pub async fn serialize_triple<'a>(
        &mut self,
        triple: impl Into<TripleRef<'a>>,
    ) -> io::Result<()> {
        self.quads
            .push(triple.into().into_owned().in_graph(GraphName::DefaultGraph));
        Ok(())
    }

    /// Writes the buffered dataset as JSON-LD and returns the underlying writer.
    ///
    /// # Errors
    ///
    /// Returns any I/O error raised by the writer, and an [`io::ErrorKind::InvalidData`] error if
    /// the dataset cannot be represented (for example, a malformed `rdf:JSON` literal).
    pub async fn finish(mut self) -> io::Result<W> {
        let document = serialized_document(&self.quads, self.base.as_deref(), &self.prefixes)
            .map_err(JsonLdParseError::from)?;
        let mut buffer = Vec::new();
        document
            .write_to(&mut buffer)
            .map_err(JsonLdParseError::from)?;
        self.writer.write_all(&buffer).await?;
        Ok(self.writer)
    }
}

/// A [`JsonLdSerializer`] bound to a [`Write`] target.
pub struct WriterJsonLdSerializer<W: Write> {
    writer: W,
    quads: Vec<Quad>,
    base: Option<String>,
    prefixes: Vec<(String, String)>,
}

impl<W: Write> WriterJsonLdSerializer<W> {
    /// Buffers a quad for serialization.
    ///
    /// # Errors
    ///
    /// Never fails; the signature mirrors the other OxiXML serializers.
    pub fn serialize_quad<'a>(&mut self, quad: impl Into<QuadRef<'a>>) -> io::Result<()> {
        self.quads.push(quad.into().into_owned());
        Ok(())
    }

    /// Buffers a triple (in the default graph) for serialization.
    ///
    /// # Errors
    ///
    /// Never fails; the signature mirrors the other OxiXML serializers.
    pub fn serialize_triple<'a>(&mut self, triple: impl Into<TripleRef<'a>>) -> io::Result<()> {
        self.quads
            .push(triple.into().into_owned().in_graph(GraphName::DefaultGraph));
        Ok(())
    }

    /// Writes the buffered dataset as JSON-LD and returns the underlying writer.
    ///
    /// # Errors
    ///
    /// Returns any I/O error raised by the writer, and an [`io::ErrorKind::InvalidData`] error if
    /// the dataset cannot be represented (for example, a malformed `rdf:JSON` literal).
    pub fn finish(mut self) -> io::Result<W> {
        let document = serialized_document(&self.quads, self.base.as_deref(), &self.prefixes)
            .map_err(JsonLdParseError::from)?;
        document
            .write_to(&mut self.writer)
            .map_err(JsonLdParseError::from)?;
        Ok(self.writer)
    }
}
