//! Helpers for walking and relabeling the blank nodes of [`oxixml_model`] terms.
//!
//! With the `rdf-12` feature enabled these helpers also descend into
//! [triple terms](https://www.w3.org/TR/rdf12-concepts/#dfn-triple-term), so datasets containing
//! RDF 1.2 embedded triples are canonicalized as well.

use oxixml_model::{BlankNode, GraphName, NamedOrBlankNode, Quad, Term};

/// Calls `visit` with the identifier of every blank node occurring anywhere in `quad`.
///
/// The same identifier may be visited more than once if it occupies several positions.
pub(crate) fn for_each_blank_node_in_quad(quad: &Quad, visit: &mut impl FnMut(&str)) {
    if let NamedOrBlankNode::BlankNode(node) = &quad.subject {
        visit(node.as_str());
    }
    for_each_blank_node_in_term(&quad.object, visit);
    if let GraphName::BlankNode(node) = &quad.graph_name {
        visit(node.as_str());
    }
}

/// Calls `visit` with the identifier of every blank node occurring anywhere in `term`.
pub(crate) fn for_each_blank_node_in_term(term: &Term, visit: &mut impl FnMut(&str)) {
    match term {
        Term::BlankNode(node) => visit(node.as_str()),
        #[cfg(feature = "rdf-12")]
        Term::Triple(triple) => {
            if let NamedOrBlankNode::BlankNode(node) = &triple.subject {
                visit(node.as_str());
            }
            for_each_blank_node_in_term(&triple.object, visit);
        }
        Term::NamedNode(_) | Term::Literal(_) => {}
        // `Term::Triple` exists as soon as `oxixml-model` is built with its own `rdf-12` feature,
        // which any other crate of the build may switch on through Cargo feature unification even
        // when this crate's `rdf-12` feature is off. Without this arm the match would then stop
        // being exhaustive and the crate would fail to compile.
        #[allow(
            unreachable_patterns,
            reason = "only reachable when oxixml-model/rdf-12 is enabled elsewhere in the build"
        )]
        _ => {}
    }
}

/// Returns a copy of `quad` in which every blank-node identifier has been rewritten by `map`.
pub(crate) fn relabel_quad(quad: &Quad, map: &impl Fn(&str) -> String) -> Quad {
    Quad {
        subject: relabel_named_or_blank_node(&quad.subject, map),
        predicate: quad.predicate.clone(),
        object: relabel_term(&quad.object, map),
        graph_name: relabel_graph_name(&quad.graph_name, map),
    }
}

fn relabel_named_or_blank_node(
    node: &NamedOrBlankNode,
    map: &impl Fn(&str) -> String,
) -> NamedOrBlankNode {
    match node {
        NamedOrBlankNode::NamedNode(node) => NamedOrBlankNode::NamedNode(node.clone()),
        NamedOrBlankNode::BlankNode(node) => {
            NamedOrBlankNode::BlankNode(BlankNode::new_unchecked(map(node.as_str())))
        }
    }
}

fn relabel_term(term: &Term, map: &impl Fn(&str) -> String) -> Term {
    match term {
        Term::NamedNode(node) => Term::NamedNode(node.clone()),
        Term::Literal(literal) => Term::Literal(literal.clone()),
        Term::BlankNode(node) => Term::BlankNode(BlankNode::new_unchecked(map(node.as_str()))),
        #[cfg(feature = "rdf-12")]
        Term::Triple(triple) => Term::Triple(Box::new(oxixml_model::Triple::new(
            relabel_named_or_blank_node(&triple.subject, map),
            triple.predicate.clone(),
            relabel_term(&triple.object, map),
        ))),
        // See `for_each_blank_node_in_term`: keeps the match exhaustive when
        // `oxixml-model/rdf-12` is unified on without this crate's own `rdf-12`.
        #[allow(
            unreachable_patterns,
            reason = "only reachable when oxixml-model/rdf-12 is enabled elsewhere in the build"
        )]
        other => other.clone(),
    }
}

fn relabel_graph_name(graph_name: &GraphName, map: &impl Fn(&str) -> String) -> GraphName {
    match graph_name {
        GraphName::NamedNode(node) => GraphName::NamedNode(node.clone()),
        GraphName::BlankNode(node) => {
            GraphName::BlankNode(BlankNode::new_unchecked(map(node.as_str())))
        }
        GraphName::DefaultGraph => GraphName::DefaultGraph,
    }
}

#[cfg(test)]
mod tests {
    use super::{for_each_blank_node_in_quad, relabel_quad};
    use oxixml_model::{BlankNode, GraphName, NamedNode, Quad};

    fn sample_quad() -> Result<Quad, Box<dyn std::error::Error>> {
        Ok(Quad::new(
            BlankNode::new("e0")?,
            NamedNode::new("http://example.com/p")?,
            BlankNode::new("e1")?,
            GraphName::BlankNode(BlankNode::new("g0")?),
        ))
    }

    #[test]
    fn visits_every_blank_node_position() -> Result<(), Box<dyn std::error::Error>> {
        let quad = sample_quad()?;
        let mut seen = Vec::new();
        for_each_blank_node_in_quad(&quad, &mut |id| seen.push(id.to_owned()));
        assert_eq!(seen, ["e0".to_owned(), "e1".to_owned(), "g0".to_owned()]);
        Ok(())
    }

    #[test]
    fn relabels_every_blank_node_position() -> Result<(), Box<dyn std::error::Error>> {
        let quad = sample_quad()?;
        let relabeled = relabel_quad(&quad, &|id| format!("x{id}"));
        assert_eq!(
            relabeled.to_string(),
            "_:xe0 <http://example.com/p> _:xe1 _:xg0"
        );
        Ok(())
    }
}
