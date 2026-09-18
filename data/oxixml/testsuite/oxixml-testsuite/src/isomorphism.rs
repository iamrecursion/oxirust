//! Blank-node-bijection isomorphism checking, shared by every eval-style RDF runner
//! ([`crate::rdf_tests`] and [`crate::json_ld`]'s `toRdf`/`fromRdf` entries).
//!
//! Two graphs (or datasets) are *isomorphic* when one can be turned into the other by
//! consistently renaming its blank nodes — comparing the parsed and expected N-Triples/N-Quads
//! output of an eval test verbatim would fail on the (permitted, and common) case where a
//! conforming parser mints different blank-node labels than the reference output happened to
//! use.
//!
//! A term position is either a *ground* value (compared by its lexical/canonical string), a
//! *blank node* (compared up to renaming), or — when this crate's `rdf-12` feature forwards
//! `oxixml-model/rdf-12` — an *embedded triple*, which is compared structurally so that blank
//! nodes nested inside it take part in the same bijection as top-level ones.

use oxixml_model::{GraphName, NamedOrBlankNode, Quad, Term, Triple};
use std::collections::{HashMap, HashSet};

/// One term position, reduced to what isomorphism checking needs: its ground value, or the
/// label of a blank node still to be matched up.
#[derive(Clone, PartialEq, Eq, Hash)]
enum Node {
    Ground(String),
    Blank(String),
    /// An RDF 1.2 embedded triple, reduced positionally to subject, predicate, object.
    #[cfg(feature = "rdf-12")]
    Triple(Box<[Node; 3]>),
}

/// A statement reduced to four positional nodes: subject, predicate, object, graph. A triple's
/// graph position is always `Node::Ground(String::new())` — the default graph, spelled the same
/// way for every triple, so it never accidentally unifies with a *named* graph.
type Statement = [Node; 4];

fn subject_node(subject: &NamedOrBlankNode) -> Node {
    match subject {
        NamedOrBlankNode::NamedNode(node) => Node::Ground(node.to_string()),
        NamedOrBlankNode::BlankNode(node) => Node::Blank(node.as_str().to_owned()),
    }
}

fn object_node(object: &Term) -> Node {
    match object {
        Term::NamedNode(node) => Node::Ground(node.to_string()),
        Term::Literal(literal) => Node::Ground(literal.to_string()),
        Term::BlankNode(node) => Node::Blank(node.as_str().to_owned()),
        #[cfg(feature = "rdf-12")]
        Term::Triple(triple) => Node::Triple(Box::new([
            subject_node(&triple.subject),
            Node::Ground(triple.predicate.to_string()),
            object_node(&triple.object),
        ])),
        // `Term::Triple` exists as soon as `oxixml-model` is built with its own `rdf-12` feature,
        // which any other crate of the build may switch on through Cargo feature unification even
        // when this crate's `rdf-12` feature is off. Without this arm the match would then stop
        // being exhaustive and the crate would fail to compile. There is no lossless `Node` to
        // fall back to here: collapsing an embedded triple to `Node::Ground` would silently
        // compare it by the wrong criterion and could make isomorphism checking report a wrong
        // answer, which is worse than failing loudly in a correctness-checking test harness. Every
        // RDF-1.2-aware runner that could feed a triple term into this function is itself gated on
        // this crate's own `rdf-12` (see the module docs), so this is unreachable in practice.
        #[allow(
            unreachable_patterns,
            reason = "only reachable when oxixml-model/rdf-12 is enabled elsewhere in the build without this crate's own rdf-12"
        )]
        _ => unreachable!(
            "a Term::Triple value requires this crate's `rdf-12` feature (which implies \
             `oxixml-model/rdf-12`) to be reduced to a Node for isomorphism checking; enable it \
             wherever RDF 1.2 triple terms may reach this conversion"
        ),
    }
}

fn graph_node(graph: &GraphName) -> Node {
    match graph {
        GraphName::DefaultGraph => Node::Ground(String::new()),
        GraphName::NamedNode(node) => Node::Ground(node.to_string()),
        GraphName::BlankNode(node) => Node::Blank(node.as_str().to_owned()),
    }
}

fn triple_statement(triple: &Triple) -> Statement {
    [
        subject_node(&triple.subject),
        Node::Ground(triple.predicate.to_string()),
        object_node(&triple.object),
        Node::Ground(String::new()),
    ]
}

fn quad_statement(quad: &Quad) -> Statement {
    [
        subject_node(&quad.subject),
        Node::Ground(quad.predicate.to_string()),
        object_node(&quad.object),
        graph_node(&quad.graph_name),
    ]
}

/// Whether `left` and `right` describe the same graph up to a bijective renaming of blank nodes.
#[must_use]
pub fn triples_isomorphic(left: &[Triple], right: &[Triple]) -> bool {
    let l: Vec<Statement> = left.iter().map(triple_statement).collect();
    let r: Vec<Statement> = right.iter().map(triple_statement).collect();
    statements_isomorphic(&l, &r)
}

/// Whether `left` and `right` describe the same dataset up to a bijective renaming of blank
/// nodes (blank graph names included).
#[must_use]
pub fn quads_isomorphic(left: &[Quad], right: &[Quad]) -> bool {
    let l: Vec<Statement> = left.iter().map(quad_statement).collect();
    let r: Vec<Statement> = right.iter().map(quad_statement).collect();
    statements_isomorphic(&l, &r)
}

fn is_ground(node: &Node) -> bool {
    match node {
        Node::Ground(_) => true,
        Node::Blank(_) => false,
        #[cfg(feature = "rdf-12")]
        Node::Triple(inner) => inner.iter().all(is_ground),
    }
}

fn collect_blanks(node: &Node, set: &mut HashSet<String>) {
    match node {
        Node::Ground(_) => {}
        Node::Blank(id) => {
            set.insert(id.clone());
        }
        #[cfg(feature = "rdf-12")]
        Node::Triple(inner) => {
            for nested in inner.iter() {
                collect_blanks(nested, set);
            }
        }
    }
}

fn blank_ids(statements: &[Statement]) -> Vec<String> {
    let mut set = HashSet::new();
    for statement in statements {
        for node in statement {
            collect_blanks(node, &mut set);
        }
    }
    let mut ids: Vec<String> = set.into_iter().collect();
    ids.sort();
    ids
}

/// Renders a statement into a canonical string key under `mapping`; `None` if a blank node in it
/// is not yet mapped.
fn render(statement: &Statement, mapping: &HashMap<String, String>) -> Option<String> {
    let mut key = String::new();
    for node in statement {
        render_node(node, mapping, &mut key)?;
        key.push('\u{1}');
    }
    Some(key)
}

/// Appends one node's canonical rendering to `key`; `None` if it (or a nested embedded triple)
/// mentions a blank node that is not yet mapped.
fn render_node(node: &Node, mapping: &HashMap<String, String>, key: &mut String) -> Option<()> {
    match node {
        Node::Ground(value) => {
            key.push('=');
            key.push_str(value);
        }
        Node::Blank(id) => {
            key.push('~');
            key.push_str(mapping.get(id)?);
        }
        #[cfg(feature = "rdf-12")]
        Node::Triple(inner) => {
            key.push('<');
            for nested in inner.iter() {
                render_node(nested, mapping, key)?;
                key.push('\u{2}');
            }
            key.push('>');
        }
    }
    Some(())
}

/// Renders a statement under the identity mapping (every blank node mapped to itself) — the key
/// space the *target* side is compared against.
fn render_identity(statement: &Statement) -> String {
    let mut ids = HashSet::new();
    for node in statement {
        collect_blanks(node, &mut ids);
    }
    let identity: HashMap<String, String> = ids.into_iter().map(|id| (id.clone(), id)).collect();
    render(statement, &identity).unwrap_or_default()
}

fn statements_isomorphic(left: &[Statement], right: &[Statement]) -> bool {
    let left_ids = blank_ids(left);
    let right_ids = blank_ids(right);
    if left_ids.len() != right_ids.len() {
        return false;
    }

    // Ground statements (no blank node at all) must coincide exactly; they constrain nothing
    // about the blank-node mapping, so checking them up front prunes the obviously-wrong case
    // cheaply.
    let left_ground: HashSet<&Statement> =
        left.iter().filter(|s| s.iter().all(is_ground)).collect();
    let right_ground: HashSet<&Statement> =
        right.iter().filter(|s| s.iter().all(is_ground)).collect();
    if left_ground != right_ground {
        return false;
    }

    if left_ids.is_empty() {
        return left.len() == right.len();
    }

    let right_keys: HashSet<String> = right.iter().map(render_identity).collect();
    let mut mapping = HashMap::new();
    let mut used = HashSet::new();
    backtrack(
        left,
        &left_ids,
        &right_ids,
        &right_keys,
        &mut mapping,
        &mut used,
    )
}

fn backtrack(
    left: &[Statement],
    left_ids: &[String],
    right_ids: &[String],
    right_keys: &HashSet<String>,
    mapping: &mut HashMap<String, String>,
    used: &mut HashSet<String>,
) -> bool {
    if mapping.len() == left_ids.len() {
        let left_keys: HashSet<String> = left
            .iter()
            .map(|s| render(s, mapping).unwrap_or_default())
            .collect();
        return left_keys == *right_keys && left_keys.len() == right_keys.len();
    }

    let Some(next) = left_ids.iter().find(|id| !mapping.contains_key(*id)) else {
        return false;
    };

    for candidate in right_ids {
        if used.contains(candidate) {
            continue;
        }
        mapping.insert(next.clone(), candidate.clone());
        used.insert(candidate.clone());
        if partial_consistent(left, mapping, right_keys)
            && backtrack(left, left_ids, right_ids, right_keys, mapping, used)
        {
            return true;
        }
        used.remove(candidate);
        mapping.remove(next);
    }
    false
}

/// Every fully-determined `left` statement (every blank node in it already mapped) must exist in
/// `right` — the pruning check that keeps the backtracking search tractable.
fn partial_consistent(
    left: &[Statement],
    mapping: &HashMap<String, String>,
    right_keys: &HashSet<String>,
) -> bool {
    left.iter()
        .filter_map(|statement| render(statement, mapping))
        .all(|key| right_keys.contains(&key))
}

#[cfg(test)]
mod tests {
    use super::*;
    use oxixml_turtle::NTriplesParser;

    fn triples(text: &str) -> Vec<Triple> {
        NTriplesParser::new()
            .for_slice(text.as_bytes())
            .collect::<Result<Vec<_>, _>>()
            .expect("valid N-Triples fixture")
    }

    #[test]
    fn isomorphic_graphs_with_differently_labelled_blanks_match() {
        let left = triples(
            "_:a <http://example.com/p> _:b .\n_:b <http://example.com/p> <http://example.com/o> .",
        );
        let right = triples(
            "_:x <http://example.com/p> _:y .\n_:y <http://example.com/p> <http://example.com/o> .",
        );
        assert!(triples_isomorphic(&left, &right));
    }

    #[test]
    fn non_isomorphic_graphs_do_not_match() {
        let left = triples(
            "_:a <http://example.com/p> _:b .\n_:b <http://example.com/p> <http://example.com/o> .",
        );
        let different = triples(
            "_:x <http://example.com/p> _:y .\n_:x <http://example.com/p> <http://example.com/o> .",
        );
        assert!(!triples_isomorphic(&left, &different));
    }

    #[test]
    fn ground_only_graphs_compare_by_equality() {
        let left =
            triples("<http://example.com/s> <http://example.com/p> <http://example.com/o> .");
        let same =
            triples("<http://example.com/s> <http://example.com/p> <http://example.com/o> .");
        let different =
            triples("<http://example.com/s> <http://example.com/p> <http://example.com/o2> .");
        assert!(triples_isomorphic(&left, &same));
        assert!(!triples_isomorphic(&left, &different));
    }

    #[test]
    fn different_sizes_never_match() {
        let left =
            triples("<http://example.com/s> <http://example.com/p> <http://example.com/o> .");
        let right = triples(
            "<http://example.com/s> <http://example.com/p> <http://example.com/o> .\n\
             <http://example.com/s> <http://example.com/p> <http://example.com/o2> .",
        );
        assert!(!triples_isomorphic(&left, &right));
    }
}
