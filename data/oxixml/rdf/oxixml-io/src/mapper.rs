//! [`QuadMapper`]: applies [`RdfParser::with_default_graph`](crate::RdfParser::with_default_graph),
//! [`without_named_graphs`](crate::RdfParser::without_named_graphs), and
//! [`rename_blank_nodes`](crate::RdfParser::rename_blank_nodes) to every parsed statement,
//! uniformly across backends.

use crate::error::RdfSyntaxError;
use oxixml_model::{BlankNode, GraphName, NamedOrBlankNode, Quad, Term, Triple};
use oxixml_turtle::n3::{N3Quad, N3Term};
use std::collections::HashMap;

/// Rewrites parsed triples/quads according to the [`RdfParser`](crate::RdfParser) options in
/// effect.
#[derive(Clone)]
pub(crate) struct QuadMapper {
    pub(crate) default_graph: GraphName,
    pub(crate) without_named_graphs: bool,
    pub(crate) blank_node_map: Option<HashMap<BlankNode, BlankNode>>,
}

impl QuadMapper {
    fn map_blank_node(&mut self, node: BlankNode) -> BlankNode {
        if let Some(blank_node_map) = &mut self.blank_node_map {
            blank_node_map
                .entry(node)
                .or_insert_with(BlankNode::default)
                .clone()
        } else {
            node
        }
    }

    fn map_subject(&mut self, node: NamedOrBlankNode) -> NamedOrBlankNode {
        match node {
            NamedOrBlankNode::NamedNode(node) => node.into(),
            NamedOrBlankNode::BlankNode(node) => self.map_blank_node(node).into(),
        }
    }

    fn map_term(&mut self, term: Term) -> Term {
        match term {
            Term::NamedNode(node) => node.into(),
            Term::BlankNode(node) => self.map_blank_node(node).into(),
            Term::Literal(literal) => literal.into(),
            // A triple term carries its own blank nodes: they must be renamed too.
            #[cfg(feature = "rdf-12")]
            Term::Triple(triple) => Term::Triple(Box::new(self.map_triple(*triple))),
            // `Term::Triple` exists as soon as `oxixml-model` is built with its own `rdf-12`
            // feature, which any other crate of the build may switch on through Cargo feature
            // unification even when this crate's `rdf-12` feature is off. Without this arm the
            // match would then stop being exhaustive and the crate would fail to compile.
            #[allow(
                unreachable_patterns,
                reason = "only reachable when oxixml-model/rdf-12 is enabled elsewhere in the build"
            )]
            other => other,
        }
    }

    fn map_triple(&mut self, triple: Triple) -> Triple {
        Triple {
            subject: self.map_subject(triple.subject),
            predicate: triple.predicate,
            object: self.map_term(triple.object),
        }
    }

    fn map_graph_name(&mut self, graph_name: GraphName) -> Result<GraphName, RdfSyntaxError> {
        match graph_name {
            GraphName::NamedNode(node) => {
                if self.without_named_graphs {
                    Err(RdfSyntaxError::msg("named graphs are not allowed"))
                } else {
                    Ok(node.into())
                }
            }
            GraphName::BlankNode(node) => {
                if self.without_named_graphs {
                    Err(RdfSyntaxError::msg("named graphs are not allowed"))
                } else {
                    Ok(self.map_blank_node(node).into())
                }
            }
            GraphName::DefaultGraph => Ok(self.default_graph.clone()),
        }
    }

    /// Rewrites a quad already produced by a dataset-capable backend (blank-node renaming, the
    /// [`without_named_graphs`](crate::RdfParser::without_named_graphs) check, and default-graph
    /// substitution).
    pub(crate) fn map_quad(&mut self, quad: Quad) -> Result<Quad, RdfSyntaxError> {
        Ok(Quad {
            subject: self.map_subject(quad.subject),
            predicate: quad.predicate,
            object: self.map_term(quad.object),
            graph_name: self.map_graph_name(quad.graph_name)?,
        })
    }

    /// Rewrites a triple produced by a graph-only backend and places it in
    /// [`default_graph`](Self::default_graph) (never fails: a bare triple can never violate
    /// [`without_named_graphs`](crate::RdfParser::without_named_graphs)).
    pub(crate) fn map_triple_to_quad(&mut self, triple: Triple) -> Quad {
        self.map_triple(triple).in_graph(self.default_graph.clone())
    }

    /// Rewrites an [`N3Quad`], rejecting the N3-only term kinds (SPARQL-style variables, and
    /// literals or quoted triples outside of object position) that have no place in regular RDF.
    pub(crate) fn map_n3_quad(&mut self, quad: N3Quad) -> Result<Quad, RdfSyntaxError> {
        Ok(Quad {
            subject: match quad.subject {
                N3Term::NamedNode(node) => Ok(node.into()),
                N3Term::BlankNode(node) => Ok(self.map_blank_node(node).into()),
                N3Term::Literal(_) => Err(RdfSyntaxError::msg(
                    "literals are not allowed in regular RDF subjects",
                )),
                #[cfg(feature = "rdf-12")]
                N3Term::Triple(_) => Err(RdfSyntaxError::msg(
                    "triple terms are not allowed in regular RDF subjects",
                )),
                N3Term::Variable(_) => Err(RdfSyntaxError::msg(
                    "variables are not allowed in regular RDF subjects",
                )),
                #[allow(
                    unreachable_patterns,
                    reason = "only reachable when oxixml-model/rdf-12 is enabled elsewhere in the build"
                )]
                _ => Err(RdfSyntaxError::msg(
                    "triple terms are not allowed in regular RDF subjects",
                )),
            }?,
            predicate: match quad.predicate {
                N3Term::NamedNode(node) => Ok(node),
                N3Term::BlankNode(_) => Err(RdfSyntaxError::msg(
                    "blank nodes are not allowed in regular RDF predicates",
                )),
                N3Term::Literal(_) => Err(RdfSyntaxError::msg(
                    "literals are not allowed in regular RDF predicates",
                )),
                #[cfg(feature = "rdf-12")]
                N3Term::Triple(_) => Err(RdfSyntaxError::msg(
                    "triple terms are not allowed in regular RDF predicates",
                )),
                N3Term::Variable(_) => Err(RdfSyntaxError::msg(
                    "variables are not allowed in regular RDF predicates",
                )),
                #[allow(
                    unreachable_patterns,
                    reason = "only reachable when oxixml-model/rdf-12 is enabled elsewhere in the build"
                )]
                _ => Err(RdfSyntaxError::msg(
                    "triple terms are not allowed in regular RDF predicates",
                )),
            }?,
            object: match quad.object {
                N3Term::NamedNode(node) => Ok(node.into()),
                N3Term::BlankNode(node) => Ok(self.map_blank_node(node).into()),
                N3Term::Literal(literal) => Ok(literal.into()),
                #[cfg(feature = "rdf-12")]
                N3Term::Triple(triple) => Ok(self.map_triple(*triple).into()),
                N3Term::Variable(_) => Err(RdfSyntaxError::msg(
                    "variables are not allowed in regular RDF objects",
                )),
                #[allow(
                    unreachable_patterns,
                    reason = "only reachable when oxixml-model/rdf-12 is enabled elsewhere in the build"
                )]
                _ => Err(RdfSyntaxError::msg(
                    "triple terms require the `rdf-12` feature of this crate",
                )),
            }?,
            graph_name: self.map_graph_name(quad.graph_name)?,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use oxixml_model::NamedNode;

    fn named(iri: &str) -> NamedOrBlankNode {
        NamedNode::new_unchecked(iri.to_owned()).into()
    }

    #[test]
    fn default_graph_substitution() -> Result<(), RdfSyntaxError> {
        let mut mapper = QuadMapper {
            default_graph: NamedNode::new_unchecked("http://example.com/g").into(),
            without_named_graphs: false,
            blank_node_map: None,
        };
        let quad = Quad::new(
            named("http://example.com/s"),
            NamedNode::new_unchecked("http://example.com/p"),
            named("http://example.com/o"),
            GraphName::DefaultGraph,
        );
        let mapped = mapper.map_quad(quad)?;
        assert_eq!(
            mapped.graph_name,
            GraphName::from(NamedNode::new_unchecked("http://example.com/g"))
        );
        Ok(())
    }

    #[test]
    fn without_named_graphs_rejects_named_graph() {
        let mut mapper = QuadMapper {
            default_graph: GraphName::DefaultGraph,
            without_named_graphs: true,
            blank_node_map: None,
        };
        let quad = Quad::new(
            named("http://example.com/s"),
            NamedNode::new_unchecked("http://example.com/p"),
            named("http://example.com/o"),
            GraphName::from(NamedNode::new_unchecked("http://example.com/g")),
        );
        assert!(mapper.map_quad(quad).is_err());
    }

    #[test]
    fn rename_blank_nodes_is_consistent_within_one_mapper() {
        let mut mapper = QuadMapper {
            default_graph: GraphName::DefaultGraph,
            without_named_graphs: false,
            blank_node_map: Some(HashMap::new()),
        };
        let original = BlankNode::new_unchecked("a");
        let triple1 = Triple::new(
            original.clone(),
            NamedNode::new_unchecked("http://example.com/p"),
            NamedNode::new_unchecked("http://example.com/o"),
        );
        let triple2 = Triple::new(
            original.clone(),
            NamedNode::new_unchecked("http://example.com/p2"),
            NamedNode::new_unchecked("http://example.com/o2"),
        );
        let quad1 = mapper.map_triple_to_quad(triple1);
        let quad2 = mapper.map_triple_to_quad(triple2);
        // Consistent within one mapper: the same source blank node maps to the same fresh id.
        assert_eq!(quad1.subject, quad2.subject);
        // But renamed away from the original id.
        assert_ne!(quad1.subject, NamedOrBlankNode::from(original));
    }
}
