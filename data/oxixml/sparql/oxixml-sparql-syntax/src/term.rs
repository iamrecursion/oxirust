//! Data structures for basic concepts like IRI, literal or triples used in SPARQL.
//!
//! These mirror `spargebra::term`, reusing the RDF term types from
//! [`oxixml_model`] exactly like `spargebra` reuses `oxrdf`.

pub use oxixml_model::{BlankNode, Literal, NamedNode, NamedOrBlankNode, Term, Triple, Variable};
use std::fmt;
use std::fmt::Write;

/// The union of [IRIs](https://www.w3.org/TR/rdf11-concepts/#dfn-iri) and
/// [literals](https://www.w3.org/TR/rdf11-concepts/#dfn-literal).
///
/// The default string formatter returns an N-Triples, Turtle, and SPARQL compatible representation.
#[derive(Eq, PartialEq, Debug, Clone, Hash)]
pub enum GroundTerm {
    /// An RDF IRI.
    NamedNode(NamedNode),
    /// An RDF literal.
    Literal(Literal),
    /// An RDF 1.2 triple term.
    #[cfg(feature = "sparql-12")]
    #[cfg_attr(docsrs, doc(cfg(feature = "sparql-12")))]
    Triple(Box<GroundTriple>),
}

impl fmt::Display for GroundTerm {
    #[inline]
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::NamedNode(node) => node.fmt(f),
            Self::Literal(literal) => literal.fmt(f),
            #[cfg(feature = "sparql-12")]
            Self::Triple(triple) => write!(f, "<<( {triple} )>>"),
        }
    }
}

#[cfg(feature = "sparql-12")]
#[cfg_attr(docsrs, doc(cfg(feature = "sparql-12")))]
impl From<GroundTriple> for GroundTerm {
    #[inline]
    fn from(triple: GroundTriple) -> Self {
        Self::Triple(Box::new(triple))
    }
}

impl From<NamedNode> for GroundTerm {
    #[inline]
    fn from(node: NamedNode) -> Self {
        Self::NamedNode(node)
    }
}

impl From<Literal> for GroundTerm {
    #[inline]
    fn from(literal: Literal) -> Self {
        Self::Literal(literal)
    }
}

impl TryFrom<Term> for GroundTerm {
    type Error = ();

    #[inline]
    fn try_from(term: Term) -> Result<Self, Self::Error> {
        match term {
            Term::NamedNode(t) => Ok(t.into()),
            Term::BlankNode(_) => Err(()),
            Term::Literal(t) => Ok(t.into()),
            #[cfg(feature = "sparql-12")]
            Term::Triple(t) => Ok(GroundTriple::try_from(*t)?.into()),
            // `Term::Triple` exists as soon as `oxixml-model` is built with its own `rdf-12`
            // feature, which any other crate of the build may switch on through Cargo feature
            // unification even when this crate's `sparql-12` feature is off. Without this arm the
            // match would then stop being exhaustive and the crate would fail to compile. Treated
            // exactly like `Term::BlankNode` above: a ground term structurally cannot hold a
            // triple term in this build configuration either.
            #[allow(
                unreachable_patterns,
                reason = "only reachable when oxixml-model/rdf-12 is enabled elsewhere in the build without this crate's own sparql-12"
            )]
            _ => Err(()),
        }
    }
}

impl From<GroundTerm> for Term {
    #[inline]
    fn from(term: GroundTerm) -> Self {
        match term {
            GroundTerm::NamedNode(t) => t.into(),
            GroundTerm::Literal(l) => l.into(),
            #[cfg(feature = "sparql-12")]
            GroundTerm::Triple(t) => Triple::from(*t).into(),
        }
    }
}

/// A [RDF triple](https://www.w3.org/TR/rdf11-concepts/#dfn-rdf-triple) without blank nodes.
///
/// The default string formatter returns an N-Quads representation.
#[derive(Eq, PartialEq, Debug, Clone, Hash)]
pub struct GroundTriple {
    /// The subject of this triple.
    pub subject: NamedNode,
    /// The predicate of this triple.
    pub predicate: NamedNode,
    /// The object of this triple.
    pub object: GroundTerm,
}

impl fmt::Display for GroundTriple {
    #[inline]
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{} {} {}", self.subject, self.predicate, self.object)
    }
}

impl TryFrom<Triple> for GroundTriple {
    type Error = ();

    #[inline]
    fn try_from(triple: Triple) -> Result<Self, Self::Error> {
        Ok(Self {
            subject: if let NamedOrBlankNode::NamedNode(s) = triple.subject {
                s
            } else {
                return Err(());
            },
            predicate: triple.predicate,
            object: triple.object.try_into()?,
        })
    }
}

impl From<GroundTriple> for Triple {
    #[inline]
    fn from(triple: GroundTriple) -> Self {
        Self {
            subject: triple.subject.into(),
            predicate: triple.predicate,
            object: triple.object.into(),
        }
    }
}

/// A possible graph name.
///
/// It is the union of [IRIs](https://www.w3.org/TR/rdf11-concepts/#dfn-iri) and the
/// [default graph name](https://www.w3.org/TR/rdf11-concepts/#dfn-default-graph).
#[derive(Eq, PartialEq, Debug, Clone, Hash, Default)]
pub enum GraphName {
    /// An RDF IRI naming a graph.
    NamedNode(NamedNode),
    /// The default graph.
    #[default]
    DefaultGraph,
}

impl GraphName {
    /// Formats using the [SPARQL S-Expression syntax](https://jena.apache.org/documentation/notes/sse.html).
    pub(crate) fn fmt_sse(&self, f: &mut impl Write) -> fmt::Result {
        match self {
            Self::NamedNode(node) => write!(f, "{node}"),
            Self::DefaultGraph => f.write_str("default"),
        }
    }
}

impl fmt::Display for GraphName {
    #[inline]
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::NamedNode(node) => node.fmt(f),
            Self::DefaultGraph => f.write_str("DEFAULT"),
        }
    }
}

impl From<NamedNode> for GraphName {
    #[inline]
    fn from(node: NamedNode) -> Self {
        Self::NamedNode(node)
    }
}

impl TryFrom<GraphNamePattern> for GraphName {
    type Error = ();

    #[inline]
    fn try_from(pattern: GraphNamePattern) -> Result<Self, Self::Error> {
        match pattern {
            GraphNamePattern::NamedNode(t) => Ok(t.into()),
            GraphNamePattern::DefaultGraph => Ok(Self::DefaultGraph),
            GraphNamePattern::Variable(_) => Err(()),
        }
    }
}

/// A [RDF triple](https://www.w3.org/TR/rdf11-concepts/#dfn-rdf-triple) in an
/// [RDF dataset](https://www.w3.org/TR/rdf11-concepts/#dfn-rdf-dataset).
///
/// The default string formatter returns an N-Quads representation.
#[derive(Eq, PartialEq, Debug, Clone, Hash)]
pub struct Quad {
    /// The subject of this quad.
    pub subject: NamedOrBlankNode,
    /// The predicate of this quad.
    pub predicate: NamedNode,
    /// The object of this quad.
    pub object: Term,
    /// The graph name of this quad.
    pub graph_name: GraphName,
}

impl Quad {
    /// Formats using the [SPARQL S-Expression syntax](https://jena.apache.org/documentation/notes/sse.html).
    pub(crate) fn fmt_sse(&self, f: &mut impl Write) -> fmt::Result {
        if self.graph_name != GraphName::DefaultGraph {
            f.write_str("(graph ")?;
            self.graph_name.fmt_sse(f)?;
            f.write_str(" (")?;
        }
        write!(
            f,
            "(triple {} {} {})",
            self.subject, self.predicate, self.object
        )?;
        if self.graph_name != GraphName::DefaultGraph {
            f.write_str("))")?;
        }
        Ok(())
    }
}

impl fmt::Display for Quad {
    #[inline]
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.graph_name == GraphName::DefaultGraph {
            write!(f, "{} {} {}", self.subject, self.predicate, self.object)
        } else {
            write!(
                f,
                "{} {} {} {}",
                self.subject, self.predicate, self.object, self.graph_name
            )
        }
    }
}

impl TryFrom<QuadPattern> for Quad {
    type Error = ();

    #[inline]
    fn try_from(quad: QuadPattern) -> Result<Self, Self::Error> {
        Ok(Self {
            subject: quad.subject.try_into()?,
            predicate: quad.predicate.try_into()?,
            object: quad.object.try_into()?,
            graph_name: quad.graph_name.try_into()?,
        })
    }
}

/// A [RDF triple](https://www.w3.org/TR/rdf11-concepts/#dfn-rdf-triple) in an
/// [RDF dataset](https://www.w3.org/TR/rdf11-concepts/#dfn-rdf-dataset) without blank nodes.
///
/// The default string formatter returns an N-Quads representation.
#[derive(Eq, PartialEq, Debug, Clone, Hash)]
pub struct GroundQuad {
    /// The subject of this quad.
    pub subject: NamedNode,
    /// The predicate of this quad.
    pub predicate: NamedNode,
    /// The object of this quad.
    pub object: GroundTerm,
    /// The graph name of this quad.
    pub graph_name: GraphName,
}

impl GroundQuad {
    /// Formats using the [SPARQL S-Expression syntax](https://jena.apache.org/documentation/notes/sse.html).
    pub(crate) fn fmt_sse(&self, f: &mut impl Write) -> fmt::Result {
        if self.graph_name != GraphName::DefaultGraph {
            f.write_str("(graph ")?;
            self.graph_name.fmt_sse(f)?;
            f.write_str(" (")?;
        }
        write!(
            f,
            "(triple {} {} {})",
            self.subject, self.predicate, self.object
        )?;
        if self.graph_name != GraphName::DefaultGraph {
            f.write_str("))")?;
        }
        Ok(())
    }
}

impl fmt::Display for GroundQuad {
    #[inline]
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.graph_name == GraphName::DefaultGraph {
            write!(f, "{} {} {}", self.subject, self.predicate, self.object)
        } else {
            write!(
                f,
                "{} {} {} {}",
                self.subject, self.predicate, self.object, self.graph_name
            )
        }
    }
}

impl TryFrom<Quad> for GroundQuad {
    type Error = ();

    #[inline]
    fn try_from(quad: Quad) -> Result<Self, Self::Error> {
        Ok(Self {
            subject: if let NamedOrBlankNode::NamedNode(s) = quad.subject {
                s
            } else {
                return Err(());
            },
            predicate: quad.predicate,
            object: quad.object.try_into()?,
            graph_name: quad.graph_name,
        })
    }
}

/// The union of [IRIs](https://www.w3.org/TR/rdf11-concepts/#dfn-iri) and
/// [variables](https://www.w3.org/TR/sparql11-query/#sparqlQueryVariables).
#[derive(Eq, PartialEq, Debug, Clone, Hash)]
pub enum NamedNodePattern {
    /// An RDF IRI.
    NamedNode(NamedNode),
    /// A SPARQL variable.
    Variable(Variable),
}

impl NamedNodePattern {
    /// Formats using the [SPARQL S-Expression syntax](https://jena.apache.org/documentation/notes/sse.html).
    pub(crate) fn fmt_sse(&self, f: &mut impl Write) -> fmt::Result {
        match self {
            Self::NamedNode(node) => write!(f, "{node}"),
            Self::Variable(var) => write!(f, "{var}"),
        }
    }
}

impl fmt::Display for NamedNodePattern {
    #[inline]
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::NamedNode(node) => node.fmt(f),
            Self::Variable(var) => var.fmt(f),
        }
    }
}

impl From<NamedNode> for NamedNodePattern {
    #[inline]
    fn from(node: NamedNode) -> Self {
        Self::NamedNode(node)
    }
}

impl From<Variable> for NamedNodePattern {
    #[inline]
    fn from(var: Variable) -> Self {
        Self::Variable(var)
    }
}

impl TryFrom<NamedNodePattern> for NamedNode {
    type Error = ();

    #[inline]
    fn try_from(pattern: NamedNodePattern) -> Result<Self, Self::Error> {
        match pattern {
            NamedNodePattern::NamedNode(t) => Ok(t),
            NamedNodePattern::Variable(_) => Err(()),
        }
    }
}

/// The union of [terms](https://www.w3.org/TR/rdf11-concepts/#dfn-rdf-term) and
/// [variables](https://www.w3.org/TR/sparql11-query/#sparqlQueryVariables).
#[derive(Eq, PartialEq, Debug, Clone, Hash)]
pub enum TermPattern {
    /// An RDF IRI.
    NamedNode(NamedNode),
    /// An RDF blank node.
    BlankNode(BlankNode),
    /// An RDF literal.
    Literal(Literal),
    /// An RDF 1.2 triple term, possibly containing variables.
    #[cfg(feature = "sparql-12")]
    #[cfg_attr(docsrs, doc(cfg(feature = "sparql-12")))]
    Triple(Box<TriplePattern>),
    /// A SPARQL variable.
    Variable(Variable),
}

impl TermPattern {
    /// Formats using the [SPARQL S-Expression syntax](https://jena.apache.org/documentation/notes/sse.html).
    pub(crate) fn fmt_sse(&self, f: &mut impl Write) -> fmt::Result {
        match self {
            Self::NamedNode(term) => write!(f, "{term}"),
            Self::BlankNode(term) => write!(f, "{term}"),
            Self::Literal(term) => write!(f, "{term}"),
            #[cfg(feature = "sparql-12")]
            Self::Triple(triple) => triple.fmt_sse(f),
            Self::Variable(var) => write!(f, "{var}"),
        }
    }
}

impl fmt::Display for TermPattern {
    #[inline]
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::NamedNode(term) => term.fmt(f),
            Self::BlankNode(term) => term.fmt(f),
            Self::Literal(term) => term.fmt(f),
            #[cfg(feature = "sparql-12")]
            Self::Triple(triple) => write!(f, "<<( {triple} )>>"),
            Self::Variable(var) => var.fmt(f),
        }
    }
}

impl From<NamedNode> for TermPattern {
    #[inline]
    fn from(node: NamedNode) -> Self {
        Self::NamedNode(node)
    }
}

impl From<BlankNode> for TermPattern {
    #[inline]
    fn from(node: BlankNode) -> Self {
        Self::BlankNode(node)
    }
}

impl From<Literal> for TermPattern {
    #[inline]
    fn from(literal: Literal) -> Self {
        Self::Literal(literal)
    }
}

#[cfg(feature = "sparql-12")]
#[cfg_attr(docsrs, doc(cfg(feature = "sparql-12")))]
impl From<TriplePattern> for TermPattern {
    #[inline]
    fn from(triple: TriplePattern) -> Self {
        Self::Triple(Box::new(triple))
    }
}

impl From<Variable> for TermPattern {
    fn from(var: Variable) -> Self {
        Self::Variable(var)
    }
}

impl From<NamedOrBlankNode> for TermPattern {
    #[inline]
    fn from(subject: NamedOrBlankNode) -> Self {
        match subject {
            NamedOrBlankNode::NamedNode(node) => node.into(),
            NamedOrBlankNode::BlankNode(node) => node.into(),
        }
    }
}

/// Converts a [`Term`] into a [`TermPattern`].
///
/// This is fallible — rather than the `From` every other lossless conversion in this file
/// implements — because a `Term::Triple` value has no lossless `TermPattern` representation
/// unless this crate's own `sparql-12` feature is enabled: see the `Err` arm below. Every other
/// `Term` variant always converts, so matching on `Ok`/`Err` is only ever meaningful once
/// `sparql-12` is off and the input might hold a triple term (reachable through
/// `oxixml-model/rdf-12`, see below).
impl TryFrom<Term> for TermPattern {
    type Error = ();

    #[inline]
    fn try_from(term: Term) -> Result<Self, Self::Error> {
        match term {
            Term::NamedNode(node) => Ok(node.into()),
            Term::BlankNode(node) => Ok(node.into()),
            Term::Literal(literal) => Ok(literal.into()),
            #[cfg(feature = "sparql-12")]
            Term::Triple(t) => Ok(TriplePattern::try_from(*t)?.into()),
            // `Term::Triple` exists as soon as `oxixml-model` is built with its own `rdf-12`
            // feature, which any other crate of the build may switch on through Cargo feature
            // unification even when this crate's `sparql-12` feature is off. Without this arm the
            // match would then stop being exhaustive and the crate would fail to compile.
            // `TermPattern`'s own `Triple` variant is *also* gated on this crate's `sparql-12` (no
            // unification risk for it specifically), so there is no lossless `TermPattern` value
            // to represent a triple term in this build configuration — reported as `Err`, exactly
            // like `TryFrom<Term> for GroundTerm` above already treats the structurally analogous
            // `Term::BlankNode` case for `GroundTerm` (which likewise cannot represent every
            // `Term`).
            #[allow(
                unreachable_patterns,
                reason = "only reachable when oxixml-model/rdf-12 is enabled elsewhere in the build without this crate's own sparql-12; there is no lossless TermPattern for a triple term in that configuration"
            )]
            _ => Err(()),
        }
    }
}

impl From<NamedNodePattern> for TermPattern {
    #[inline]
    fn from(element: NamedNodePattern) -> Self {
        match element {
            NamedNodePattern::NamedNode(node) => node.into(),
            NamedNodePattern::Variable(var) => var.into(),
        }
    }
}

impl From<GroundTermPattern> for TermPattern {
    #[inline]
    fn from(element: GroundTermPattern) -> Self {
        match element {
            GroundTermPattern::NamedNode(node) => node.into(),
            GroundTermPattern::Literal(literal) => literal.into(),
            #[cfg(feature = "sparql-12")]
            GroundTermPattern::Triple(t) => TriplePattern::from(*t).into(),
            GroundTermPattern::Variable(variable) => variable.into(),
        }
    }
}

impl TryFrom<TermPattern> for NamedOrBlankNode {
    type Error = ();

    #[inline]
    fn try_from(term: TermPattern) -> Result<Self, Self::Error> {
        match term {
            TermPattern::NamedNode(t) => Ok(t.into()),
            TermPattern::BlankNode(t) => Ok(t.into()),
            #[cfg(feature = "sparql-12")]
            TermPattern::Triple(_) => Err(()),
            TermPattern::Literal(_) | TermPattern::Variable(_) => Err(()),
        }
    }
}

impl TryFrom<TermPattern> for Term {
    type Error = ();

    #[inline]
    fn try_from(pattern: TermPattern) -> Result<Self, Self::Error> {
        match pattern {
            TermPattern::NamedNode(t) => Ok(t.into()),
            TermPattern::BlankNode(t) => Ok(t.into()),
            TermPattern::Literal(t) => Ok(t.into()),
            #[cfg(feature = "sparql-12")]
            TermPattern::Triple(t) => Ok(Triple::try_from(*t)?.into()),
            TermPattern::Variable(_) => Err(()),
        }
    }
}

/// The union of [terms](https://www.w3.org/TR/rdf11-concepts/#dfn-rdf-term) and
/// [variables](https://www.w3.org/TR/sparql11-query/#sparqlQueryVariables) without blank nodes.
#[derive(Eq, PartialEq, Debug, Clone, Hash)]
pub enum GroundTermPattern {
    /// An RDF IRI.
    NamedNode(NamedNode),
    /// An RDF literal.
    Literal(Literal),
    /// An RDF 1.2 triple term, possibly containing variables.
    #[cfg(feature = "sparql-12")]
    #[cfg_attr(docsrs, doc(cfg(feature = "sparql-12")))]
    Triple(Box<GroundTriplePattern>),
    /// A SPARQL variable.
    Variable(Variable),
}

impl GroundTermPattern {
    /// Formats using the [SPARQL S-Expression syntax](https://jena.apache.org/documentation/notes/sse.html).
    pub(crate) fn fmt_sse(&self, f: &mut impl Write) -> fmt::Result {
        match self {
            Self::NamedNode(term) => write!(f, "{term}"),
            Self::Literal(term) => write!(f, "{term}"),
            #[cfg(feature = "sparql-12")]
            Self::Triple(triple) => triple.fmt_sse(f),
            Self::Variable(var) => write!(f, "{var}"),
        }
    }
}

impl fmt::Display for GroundTermPattern {
    #[inline]
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::NamedNode(term) => term.fmt(f),
            Self::Literal(term) => term.fmt(f),
            #[cfg(feature = "sparql-12")]
            Self::Triple(triple) => write!(f, "<<( {triple} )>>"),
            Self::Variable(var) => var.fmt(f),
        }
    }
}

impl From<NamedNode> for GroundTermPattern {
    #[inline]
    fn from(node: NamedNode) -> Self {
        Self::NamedNode(node)
    }
}

impl From<Literal> for GroundTermPattern {
    #[inline]
    fn from(literal: Literal) -> Self {
        Self::Literal(literal)
    }
}

#[cfg(feature = "sparql-12")]
#[cfg_attr(docsrs, doc(cfg(feature = "sparql-12")))]
impl From<GroundTriplePattern> for GroundTermPattern {
    #[inline]
    fn from(triple: GroundTriplePattern) -> Self {
        Self::Triple(Box::new(triple))
    }
}

impl From<Variable> for GroundTermPattern {
    #[inline]
    fn from(var: Variable) -> Self {
        Self::Variable(var)
    }
}

impl From<GroundTerm> for GroundTermPattern {
    #[inline]
    fn from(term: GroundTerm) -> Self {
        match term {
            GroundTerm::NamedNode(node) => node.into(),
            GroundTerm::Literal(literal) => literal.into(),
            #[cfg(feature = "sparql-12")]
            GroundTerm::Triple(t) => GroundTriplePattern::from(*t).into(),
        }
    }
}

impl From<NamedNodePattern> for GroundTermPattern {
    #[inline]
    fn from(element: NamedNodePattern) -> Self {
        match element {
            NamedNodePattern::NamedNode(node) => node.into(),
            NamedNodePattern::Variable(var) => var.into(),
        }
    }
}

impl TryFrom<TermPattern> for GroundTermPattern {
    type Error = ();

    #[inline]
    fn try_from(pattern: TermPattern) -> Result<Self, Self::Error> {
        Ok(match pattern {
            TermPattern::NamedNode(named_node) => named_node.into(),
            TermPattern::BlankNode(_) => return Err(()),
            TermPattern::Literal(literal) => literal.into(),
            #[cfg(feature = "sparql-12")]
            TermPattern::Triple(triple) => GroundTriplePattern::try_from(*triple)?.into(),
            TermPattern::Variable(variable) => variable.into(),
        })
    }
}

/// The union of [IRIs](https://www.w3.org/TR/rdf11-concepts/#dfn-iri),
/// [default graph name](https://www.w3.org/TR/rdf11-concepts/#dfn-default-graph) and
/// [variables](https://www.w3.org/TR/sparql11-query/#sparqlQueryVariables).
#[derive(Eq, PartialEq, Debug, Clone, Hash)]
pub enum GraphNamePattern {
    /// An RDF IRI naming a graph.
    NamedNode(NamedNode),
    /// The default graph.
    DefaultGraph,
    /// A SPARQL variable.
    Variable(Variable),
}

impl GraphNamePattern {
    /// Formats using the [SPARQL S-Expression syntax](https://jena.apache.org/documentation/notes/sse.html).
    pub(crate) fn fmt_sse(&self, f: &mut impl Write) -> fmt::Result {
        match self {
            Self::NamedNode(node) => write!(f, "{node}"),
            Self::DefaultGraph => f.write_str("default"),
            Self::Variable(var) => write!(f, "{var}"),
        }
    }
}

impl fmt::Display for GraphNamePattern {
    #[inline]
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::NamedNode(node) => node.fmt(f),
            Self::DefaultGraph => f.write_str("DEFAULT"),
            Self::Variable(var) => var.fmt(f),
        }
    }
}

impl From<NamedNode> for GraphNamePattern {
    #[inline]
    fn from(node: NamedNode) -> Self {
        Self::NamedNode(node)
    }
}

impl From<Variable> for GraphNamePattern {
    #[inline]
    fn from(var: Variable) -> Self {
        Self::Variable(var)
    }
}

impl From<GraphName> for GraphNamePattern {
    #[inline]
    fn from(graph_name: GraphName) -> Self {
        match graph_name {
            GraphName::NamedNode(node) => node.into(),
            GraphName::DefaultGraph => Self::DefaultGraph,
        }
    }
}

impl From<NamedNodePattern> for GraphNamePattern {
    #[inline]
    fn from(graph_name: NamedNodePattern) -> Self {
        match graph_name {
            NamedNodePattern::NamedNode(node) => node.into(),
            NamedNodePattern::Variable(var) => var.into(),
        }
    }
}

/// A [triple pattern](https://www.w3.org/TR/sparql11-query/#defn_TriplePattern).
#[derive(Eq, PartialEq, Debug, Clone, Hash)]
pub struct TriplePattern {
    /// The subject of this triple pattern.
    pub subject: TermPattern,
    /// The predicate of this triple pattern.
    pub predicate: NamedNodePattern,
    /// The object of this triple pattern.
    pub object: TermPattern,
}

impl TriplePattern {
    pub(crate) fn new(
        subject: impl Into<TermPattern>,
        predicate: impl Into<NamedNodePattern>,
        object: impl Into<TermPattern>,
    ) -> Self {
        Self {
            subject: subject.into(),
            predicate: predicate.into(),
            object: object.into(),
        }
    }

    /// Formats using the [SPARQL S-Expression syntax](https://jena.apache.org/documentation/notes/sse.html).
    pub(crate) fn fmt_sse(&self, f: &mut impl Write) -> fmt::Result {
        f.write_str("(triple ")?;
        self.subject.fmt_sse(f)?;
        f.write_str(" ")?;
        self.predicate.fmt_sse(f)?;
        f.write_str(" ")?;
        self.object.fmt_sse(f)?;
        f.write_str(")")
    }
}

impl fmt::Display for TriplePattern {
    #[inline]
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{} {} {}", self.subject, self.predicate, self.object)
    }
}

impl TryFrom<Triple> for TriplePattern {
    type Error = ();

    #[inline]
    fn try_from(triple: Triple) -> Result<Self, Self::Error> {
        Ok(Self {
            subject: triple.subject.into(),
            predicate: triple.predicate.into(),
            object: triple.object.try_into()?,
        })
    }
}

impl TryFrom<TriplePattern> for Triple {
    type Error = ();

    #[inline]
    fn try_from(triple: TriplePattern) -> Result<Self, Self::Error> {
        Ok(Self {
            subject: triple.subject.try_into()?,
            predicate: triple.predicate.try_into()?,
            object: triple.object.try_into()?,
        })
    }
}

/// A [triple pattern](https://www.w3.org/TR/sparql11-query/#defn_TriplePattern) without blank nodes.
#[derive(Eq, PartialEq, Debug, Clone, Hash)]
pub struct GroundTriplePattern {
    /// The subject of this triple pattern.
    pub subject: GroundTermPattern,
    /// The predicate of this triple pattern.
    pub predicate: NamedNodePattern,
    /// The object of this triple pattern.
    pub object: GroundTermPattern,
}

impl GroundTriplePattern {
    /// Formats using the [SPARQL S-Expression syntax](https://jena.apache.org/documentation/notes/sse.html).
    #[cfg(feature = "sparql-12")]
    pub(crate) fn fmt_sse(&self, f: &mut impl Write) -> fmt::Result {
        f.write_str("(triple ")?;
        self.subject.fmt_sse(f)?;
        f.write_str(" ")?;
        self.predicate.fmt_sse(f)?;
        f.write_str(" ")?;
        self.object.fmt_sse(f)?;
        f.write_str(")")
    }
}

impl fmt::Display for GroundTriplePattern {
    #[inline]
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{} {} {}", self.subject, self.predicate, self.object)
    }
}

#[cfg(feature = "sparql-12")]
#[cfg_attr(docsrs, doc(cfg(feature = "sparql-12")))]
impl From<GroundTriple> for GroundTriplePattern {
    #[inline]
    fn from(triple: GroundTriple) -> Self {
        Self {
            subject: triple.subject.into(),
            predicate: triple.predicate.into(),
            object: triple.object.into(),
        }
    }
}

#[cfg(feature = "sparql-12")]
#[cfg_attr(docsrs, doc(cfg(feature = "sparql-12")))]
impl From<GroundTriplePattern> for TriplePattern {
    #[inline]
    fn from(triple: GroundTriplePattern) -> Self {
        Self {
            subject: triple.subject.into(),
            predicate: triple.predicate,
            object: triple.object.into(),
        }
    }
}

impl TryFrom<TriplePattern> for GroundTriplePattern {
    type Error = ();

    #[inline]
    fn try_from(triple: TriplePattern) -> Result<Self, Self::Error> {
        Ok(Self {
            subject: triple.subject.try_into()?,
            predicate: triple.predicate,
            object: triple.object.try_into()?,
        })
    }
}

/// A [triple pattern](https://www.w3.org/TR/sparql11-query/#defn_TriplePattern) in a specific graph.
#[derive(Eq, PartialEq, Debug, Clone, Hash)]
pub struct QuadPattern {
    /// The subject of this quad pattern.
    pub subject: TermPattern,
    /// The predicate of this quad pattern.
    pub predicate: NamedNodePattern,
    /// The object of this quad pattern.
    pub object: TermPattern,
    /// The graph name of this quad pattern.
    pub graph_name: GraphNamePattern,
}

impl QuadPattern {
    pub(crate) fn new(
        subject: impl Into<TermPattern>,
        predicate: impl Into<NamedNodePattern>,
        object: impl Into<TermPattern>,
        graph_name: impl Into<GraphNamePattern>,
    ) -> Self {
        Self {
            subject: subject.into(),
            predicate: predicate.into(),
            object: object.into(),
            graph_name: graph_name.into(),
        }
    }

    /// Formats using the [SPARQL S-Expression syntax](https://jena.apache.org/documentation/notes/sse.html).
    pub(crate) fn fmt_sse(&self, f: &mut impl Write) -> fmt::Result {
        if self.graph_name != GraphNamePattern::DefaultGraph {
            f.write_str("(graph ")?;
            self.graph_name.fmt_sse(f)?;
            f.write_str(" (")?;
        }
        f.write_str("(triple ")?;
        self.subject.fmt_sse(f)?;
        f.write_str(" ")?;
        self.predicate.fmt_sse(f)?;
        f.write_str(" ")?;
        self.object.fmt_sse(f)?;
        f.write_str(")")?;
        if self.graph_name != GraphNamePattern::DefaultGraph {
            f.write_str("))")?;
        }
        Ok(())
    }
}

impl fmt::Display for QuadPattern {
    #[inline]
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.graph_name == GraphNamePattern::DefaultGraph {
            write!(f, "{} {} {}", self.subject, self.predicate, self.object)
        } else {
            write!(
                f,
                "GRAPH {} {{ {} {} {} }}",
                self.graph_name, self.subject, self.predicate, self.object
            )
        }
    }
}

/// A [triple pattern](https://www.w3.org/TR/sparql11-query/#defn_TriplePattern) in a specific graph
/// without blank nodes.
#[derive(Eq, PartialEq, Debug, Clone, Hash)]
pub struct GroundQuadPattern {
    /// The subject of this quad pattern.
    pub subject: GroundTermPattern,
    /// The predicate of this quad pattern.
    pub predicate: NamedNodePattern,
    /// The object of this quad pattern.
    pub object: GroundTermPattern,
    /// The graph name of this quad pattern.
    pub graph_name: GraphNamePattern,
}

impl GroundQuadPattern {
    /// Formats using the [SPARQL S-Expression syntax](https://jena.apache.org/documentation/notes/sse.html).
    pub(crate) fn fmt_sse(&self, f: &mut impl Write) -> fmt::Result {
        if self.graph_name != GraphNamePattern::DefaultGraph {
            f.write_str("(graph ")?;
            self.graph_name.fmt_sse(f)?;
            f.write_str(" (")?;
        }
        f.write_str("(triple ")?;
        self.subject.fmt_sse(f)?;
        f.write_str(" ")?;
        self.predicate.fmt_sse(f)?;
        f.write_str(" ")?;
        self.object.fmt_sse(f)?;
        f.write_str(")")?;
        if self.graph_name != GraphNamePattern::DefaultGraph {
            f.write_str("))")?;
        }
        Ok(())
    }
}

impl fmt::Display for GroundQuadPattern {
    #[inline]
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.graph_name == GraphNamePattern::DefaultGraph {
            write!(f, "{} {} {}", self.subject, self.predicate, self.object)
        } else {
            write!(
                f,
                "GRAPH {} {{ {} {} {} }}",
                self.graph_name, self.subject, self.predicate, self.object
            )
        }
    }
}

impl TryFrom<QuadPattern> for GroundQuadPattern {
    type Error = ();

    #[inline]
    fn try_from(pattern: QuadPattern) -> Result<Self, Self::Error> {
        Ok(Self {
            subject: pattern.subject.try_into()?,
            predicate: pattern.predicate,
            object: pattern.object.try_into()?,
            graph_name: pattern.graph_name,
        })
    }
}
