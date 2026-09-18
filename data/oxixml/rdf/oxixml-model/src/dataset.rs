//! [In-memory implementation](Dataset) of [RDF datasets](https://www.w3.org/TR/rdf11-concepts/#dfn-rdf-dataset).
//!
//! Usage example:
//! ```
//! use oxixml_model::*;
//!
//! let mut dataset = Dataset::default();
//!
//! // insertion
//! let ex = NamedNodeRef::new("http://example.com")?;
//! let quad = QuadRef::new(ex, ex, ex, ex);
//! dataset.insert(quad);
//!
//! // simple filter
//! let results: Vec<_> = dataset.quads_for_subject(ex).collect();
//! assert_eq!(vec![quad], results);
//!
//! // direct access to a dataset graph
//! let results: Vec<_> = dataset.graph(ex).iter().collect();
//! assert_eq!(vec![TripleRef::new(ex, ex, ex)], results);
//!
//! // Print
//! assert_eq!(
//!     dataset.to_string(),
//!     "<http://example.com> <http://example.com> <http://example.com> <http://example.com> .\n"
//! );
//! # Result::<_, Box<dyn std::error::Error>>::Ok(())
//! ```
//!
//! See also [`Graph`] if you only care about plain triples.
#![allow(clippy::type_complexity)]

mod canon;

pub use self::canon::{
    CanonicalizationAlgorithm, CanonicalizationError, CanonicalizationHashAlgorithm,
    DEFAULT_CANONICALIZATION_WORK_BUDGET,
};

use crate::interning::*;
use crate::*;
use std::collections::BTreeSet;
use std::fmt;

/// An in-memory [RDF dataset](https://www.w3.org/TR/rdf11-concepts/#dfn-rdf-dataset).
///
/// It can accommodate a fairly large number of quads (in the few millions).
///
/// <div class="warning">It interns the strings and does not do any garbage collection yet:
/// if you insert and remove a lot of different terms, memory will grow without any reduction.</div>
///
/// Usage example:
/// ```
/// use oxixml_model::*;
///
/// let mut dataset = Dataset::default();
///
/// // insertion
/// let ex = NamedNodeRef::new("http://example.com")?;
/// let quad = QuadRef::new(ex, ex, ex, ex);
/// dataset.insert(quad);
///
/// // simple filter
/// let results: Vec<_> = dataset.quads_for_subject(ex).collect();
/// assert_eq!(vec![quad], results);
///
/// // direct access to a dataset graph
/// let results: Vec<_> = dataset.graph(ex).iter().collect();
/// assert_eq!(vec![TripleRef::new(ex, ex, ex)], results);
/// # Result::<_, Box<dyn std::error::Error>>::Ok(())
/// ```
#[derive(Debug, Default, Clone)]
pub struct Dataset {
    interner: Interner,
    gspo: BTreeSet<(
        InternedGraphName,
        InternedNamedOrBlankNode,
        InternedNamedNode,
        InternedTerm,
    )>,
    gpos: BTreeSet<(
        InternedGraphName,
        InternedNamedNode,
        InternedTerm,
        InternedNamedOrBlankNode,
    )>,
    gosp: BTreeSet<(
        InternedGraphName,
        InternedTerm,
        InternedNamedOrBlankNode,
        InternedNamedNode,
    )>,
    spog: BTreeSet<(
        InternedNamedOrBlankNode,
        InternedNamedNode,
        InternedTerm,
        InternedGraphName,
    )>,
    posg: BTreeSet<(
        InternedNamedNode,
        InternedTerm,
        InternedNamedOrBlankNode,
        InternedGraphName,
    )>,
    ospg: BTreeSet<(
        InternedTerm,
        InternedNamedOrBlankNode,
        InternedNamedNode,
        InternedGraphName,
    )>,
}

impl Dataset {
    /// Creates a new dataset.
    pub fn new() -> Self {
        Self::default()
    }

    /// Provides a read-only view on an
    /// [RDF graph](https://www.w3.org/TR/rdf11-concepts/#dfn-rdf-graph) contained in this dataset.
    ///
    /// ```
    /// use oxixml_model::*;
    ///
    /// let mut dataset = Dataset::default();
    /// let ex = NamedNodeRef::new("http://example.com")?;
    /// dataset.insert(QuadRef::new(ex, ex, ex, ex));
    ///
    /// let results: Vec<_> = dataset.graph(ex).iter().collect();
    /// assert_eq!(vec![TripleRef::new(ex, ex, ex)], results);
    /// # Result::<_, Box<dyn std::error::Error>>::Ok(())
    /// ```
    pub fn graph<'a, 'b>(&'a self, graph_name: impl Into<GraphNameRef<'b>>) -> GraphView<'a> {
        let graph_name = self
            .encoded_graph_name(graph_name)
            .unwrap_or_else(InternedGraphName::impossible);
        GraphView {
            dataset: self,
            graph_name,
        }
    }

    /// Provides a read/write view on an
    /// [RDF graph](https://www.w3.org/TR/rdf11-concepts/#dfn-rdf-graph) contained in this dataset.
    ///
    /// ```
    /// use oxixml_model::*;
    ///
    /// let mut dataset = Dataset::default();
    /// let ex = NamedNodeRef::new("http://example.com")?;
    ///
    /// // We edit and query the dataset http://example.com graph
    /// {
    ///     let mut graph = dataset.graph_mut(ex);
    ///     graph.insert(TripleRef::new(ex, ex, ex));
    ///     let results: Vec<_> = graph.iter().collect();
    ///     assert_eq!(vec![TripleRef::new(ex, ex, ex)], results);
    /// }
    ///
    /// // We have also changed the dataset itself
    /// let results: Vec<_> = dataset.iter().collect();
    /// assert_eq!(vec![QuadRef::new(ex, ex, ex, ex)], results);
    /// # Result::<_, Box<dyn std::error::Error>>::Ok(())
    /// ```
    pub fn graph_mut<'a, 'b>(
        &'a mut self,
        graph_name: impl Into<GraphNameRef<'b>>,
    ) -> GraphViewMut<'a> {
        let graph_name = InternedGraphName::encoded_into(graph_name.into(), &mut self.interner);
        GraphViewMut {
            dataset: self,
            graph_name,
        }
    }

    /// Returns all the quads contained by the dataset.
    pub fn iter(&self) -> Iter<'_> {
        let iter = self.spog.iter();
        Iter {
            dataset: self,
            inner: iter,
        }
    }

    /// Returns all the quads with the given subject.
    pub fn quads_for_subject<'a, 'b>(
        &'a self,
        subject: impl Into<NamedOrBlankNodeRef<'b>>,
    ) -> impl Iterator<Item = QuadRef<'a>> + 'a {
        let subject = self
            .encoded_named_or_blank_node(subject)
            .unwrap_or_else(InternedNamedOrBlankNode::impossible);
        self.interned_quads_for_subject(&subject)
            .map(move |q| self.decode_spog(q))
    }

    fn interned_quads_for_subject<'a>(
        &'a self,
        subject: &InternedNamedOrBlankNode,
    ) -> impl Iterator<
        Item = (
            &'a InternedNamedOrBlankNode,
            &'a InternedNamedNode,
            &'a InternedTerm,
            &'a InternedGraphName,
        ),
    > + use<'a> {
        self.spog
            .range(
                &(
                    *subject,
                    InternedNamedNode::first(),
                    InternedTerm::first(),
                    InternedGraphName::first(),
                )
                    ..&(
                        subject.next(),
                        InternedNamedNode::first(),
                        InternedTerm::first(),
                        InternedGraphName::first(),
                    ),
            )
            .map(|(s, p, o, g)| (s, p, o, g))
    }

    /// Returns all the quads with the given predicate.
    pub fn quads_for_predicate<'a, 'b>(
        &'a self,
        predicate: impl Into<NamedNodeRef<'b>>,
    ) -> impl Iterator<Item = QuadRef<'a>> + 'a {
        let predicate = self
            .encoded_named_node(predicate)
            .unwrap_or_else(InternedNamedNode::impossible);
        self.interned_quads_for_predicate(predicate)
            .map(move |q| self.decode_spog(q))
    }

    fn interned_quads_for_predicate(
        &self,
        predicate: InternedNamedNode,
    ) -> impl Iterator<
        Item = (
            &InternedNamedOrBlankNode,
            &InternedNamedNode,
            &InternedTerm,
            &InternedGraphName,
        ),
    > + '_ {
        self.posg
            .range(
                &(
                    predicate,
                    InternedTerm::first(),
                    InternedNamedOrBlankNode::first(),
                    InternedGraphName::first(),
                )
                    ..&(
                        predicate.next(),
                        InternedTerm::first(),
                        InternedNamedOrBlankNode::first(),
                        InternedGraphName::first(),
                    ),
            )
            .map(|(p, o, s, g)| (s, p, o, g))
    }

    /// Returns all the quads with the given object.
    pub fn quads_for_object<'a, 'b>(
        &'a self,
        object: impl Into<TermRef<'b>>,
    ) -> impl Iterator<Item = QuadRef<'a>> + 'a {
        let object = self
            .encoded_term(object)
            .unwrap_or_else(InternedTerm::impossible);

        self.interned_quads_for_object(&object)
            .map(move |q| self.decode_spog(q))
    }

    fn interned_quads_for_object<'a>(
        &'a self,
        object: &InternedTerm,
    ) -> impl Iterator<
        Item = (
            &'a InternedNamedOrBlankNode,
            &'a InternedNamedNode,
            &'a InternedTerm,
            &'a InternedGraphName,
        ),
    > + use<'a> {
        self.ospg
            .range(
                &(
                    object.clone(),
                    InternedNamedOrBlankNode::first(),
                    InternedNamedNode::first(),
                    InternedGraphName::first(),
                )
                    ..&(
                        object.next(),
                        InternedNamedOrBlankNode::first(),
                        InternedNamedNode::first(),
                        InternedGraphName::first(),
                    ),
            )
            .map(|(o, s, p, g)| (s, p, o, g))
    }

    /// Returns all the quads with the given graph name.
    pub fn quads_for_graph_name<'a, 'b>(
        &'a self,
        graph_name: impl Into<GraphNameRef<'b>>,
    ) -> impl Iterator<Item = QuadRef<'a>> + 'a {
        let graph_name = self
            .encoded_graph_name(graph_name)
            .unwrap_or_else(InternedGraphName::impossible);

        self.interned_quads_for_graph_name(&graph_name)
            .map(move |q| self.decode_spog(q))
    }

    fn interned_quads_for_graph_name<'a>(
        &'a self,
        graph_name: &InternedGraphName,
    ) -> impl Iterator<
        Item = (
            &'a InternedNamedOrBlankNode,
            &'a InternedNamedNode,
            &'a InternedTerm,
            &'a InternedGraphName,
        ),
    > + use<'a> {
        self.gspo
            .range(
                &(
                    *graph_name,
                    InternedNamedOrBlankNode::first(),
                    InternedNamedNode::first(),
                    InternedTerm::first(),
                )
                    ..&(
                        graph_name.next(),
                        InternedNamedOrBlankNode::first(),
                        InternedNamedNode::first(),
                        InternedTerm::first(),
                    ),
            )
            .map(|(g, s, p, o)| (s, p, o, g))
    }

    /// Retrieves quads with a filter on each quad component.
    pub fn quads_for_pattern<'a>(
        &'a self,
        subject: Option<NamedOrBlankNodeRef<'a>>,
        predicate: Option<NamedNodeRef<'a>>,
        object: Option<TermRef<'a>>,
        graph_name: Option<GraphNameRef<'a>>,
    ) -> impl Iterator<Item = QuadRef<'a>> + 'a {
        let iter: Box<dyn Iterator<Item = _>> = if let Some(subject) = subject {
            Box::new(self.quads_for_subject(subject).filter(move |q| {
                predicate.as_ref().is_none_or(|p| *p == q.predicate)
                    && object.as_ref().is_none_or(|o| *o == q.object)
                    && graph_name.as_ref().is_none_or(|g| *g == q.graph_name)
            }))
        } else if let Some(object) = object {
            Box::new(self.quads_for_object(object).filter(move |q| {
                predicate.as_ref().is_none_or(|p| *p == q.predicate)
                    && graph_name.as_ref().is_none_or(|g| *g == q.graph_name)
            }))
        } else if let Some(predicate) = predicate {
            Box::new(
                self.quads_for_predicate(predicate)
                    .filter(move |q| graph_name.as_ref().is_none_or(|g| *g == q.graph_name)),
            )
        } else if let Some(graph_name) = graph_name {
            Box::new(self.quads_for_graph_name(graph_name))
        } else {
            Box::new(self.iter())
        };
        iter
    }

    /// Checks if the dataset contains the given quad.
    pub fn contains<'a>(&self, quad: impl Into<QuadRef<'a>>) -> bool {
        if let Some(q) = self.encoded_quad(quad.into()) {
            self.spog.contains(&q)
        } else {
            false
        }
    }

    /// Returns the number of quads in this dataset.
    pub fn len(&self) -> usize {
        self.gspo.len()
    }

    /// Checks if this dataset is empty.
    pub fn is_empty(&self) -> bool {
        self.gspo.is_empty()
    }

    /// Adds a quad to the dataset.
    pub fn insert<'a>(&mut self, quad: impl Into<QuadRef<'a>>) -> bool {
        let quad = self.encode_quad(quad.into());
        self.insert_encoded(quad)
    }

    fn insert_encoded(
        &mut self,
        quad: (
            InternedNamedOrBlankNode,
            InternedNamedNode,
            InternedTerm,
            InternedGraphName,
        ),
    ) -> bool {
        let (s, p, o, g) = quad;
        self.gspo.insert((g, s, p, o.clone()));
        self.gpos.insert((g, p, o.clone(), s));
        self.gosp.insert((g, o.clone(), s, p));
        self.spog.insert((s, p, o.clone(), g));
        self.posg.insert((p, o.clone(), s, g));
        self.ospg.insert((o, s, p, g))
    }

    /// Removes a concrete quad from the dataset.
    pub fn remove<'a>(&mut self, quad: impl Into<QuadRef<'a>>) -> bool {
        if let Some(quad) = self.encoded_quad(quad.into()) {
            self.remove_encoded(quad)
        } else {
            false
        }
    }

    fn remove_encoded(
        &mut self,
        quad: (
            InternedNamedOrBlankNode,
            InternedNamedNode,
            InternedTerm,
            InternedGraphName,
        ),
    ) -> bool {
        let (s, p, o, g) = quad;
        self.gspo.remove(&(g, s, p, o.clone()));
        self.gpos.remove(&(g, p, o.clone(), s));
        self.gosp.remove(&(g, o.clone(), s, p));
        self.spog.remove(&(s, p, o.clone(), g));
        self.posg.remove(&(p, o.clone(), s, g));
        self.ospg.remove(&(o, s, p, g))
    }

    /// Clears the dataset.
    pub fn clear(&mut self) {
        self.gspo.clear();
        self.gpos.clear();
        self.gosp.clear();
        self.spog.clear();
        self.posg.clear();
        self.ospg.clear();
    }

    fn encode_quad(
        &mut self,
        quad: QuadRef<'_>,
    ) -> (
        InternedNamedOrBlankNode,
        InternedNamedNode,
        InternedTerm,
        InternedGraphName,
    ) {
        (
            InternedNamedOrBlankNode::encoded_into(quad.subject, &mut self.interner),
            InternedNamedNode::encoded_into(quad.predicate, &mut self.interner),
            InternedTerm::encoded_into(quad.object, &mut self.interner),
            InternedGraphName::encoded_into(quad.graph_name, &mut self.interner),
        )
    }

    fn encoded_quad(
        &self,
        quad: QuadRef<'_>,
    ) -> Option<(
        InternedNamedOrBlankNode,
        InternedNamedNode,
        InternedTerm,
        InternedGraphName,
    )> {
        Some((
            self.encoded_named_or_blank_node(quad.subject)?,
            self.encoded_named_node(quad.predicate)?,
            self.encoded_term(quad.object)?,
            self.encoded_graph_name(quad.graph_name)?,
        ))
    }

    pub(super) fn encoded_named_node<'a>(
        &self,
        node: impl Into<NamedNodeRef<'a>>,
    ) -> Option<InternedNamedNode> {
        InternedNamedNode::encoded_from(node.into(), &self.interner)
    }

    pub(super) fn encoded_named_or_blank_node<'a>(
        &self,
        node: impl Into<NamedOrBlankNodeRef<'a>>,
    ) -> Option<InternedNamedOrBlankNode> {
        InternedNamedOrBlankNode::encoded_from(node.into(), &self.interner)
    }

    pub(super) fn encoded_term<'a>(&self, term: impl Into<TermRef<'a>>) -> Option<InternedTerm> {
        InternedTerm::encoded_from(term.into(), &self.interner)
    }

    pub(super) fn encoded_graph_name<'a>(
        &self,
        graph_name: impl Into<GraphNameRef<'a>>,
    ) -> Option<InternedGraphName> {
        InternedGraphName::encoded_from(graph_name.into(), &self.interner)
    }

    fn decode_spog(
        &self,
        quad: (
            &InternedNamedOrBlankNode,
            &InternedNamedNode,
            &InternedTerm,
            &InternedGraphName,
        ),
    ) -> QuadRef<'_> {
        QuadRef {
            subject: quad.0.decode_from(&self.interner),
            predicate: quad.1.decode_from(&self.interner),
            object: quad.2.decode_from(&self.interner),
            graph_name: quad.3.decode_from(&self.interner),
        }
    }

    fn decode_spo(
        &self,
        triple: (&InternedNamedOrBlankNode, &InternedNamedNode, &InternedTerm),
    ) -> TripleRef<'_> {
        TripleRef {
            subject: triple.0.decode_from(&self.interner),
            predicate: triple.1.decode_from(&self.interner),
            object: triple.2.decode_from(&self.interner),
        }
    }
}

impl PartialEq for Dataset {
    fn eq(&self, other: &Self) -> bool {
        if self.len() != other.len() {
            return false;
        }
        for q in self {
            if !other.contains(q) {
                return false;
            }
        }
        true
    }
}

impl Eq for Dataset {}

impl<'a> IntoIterator for &'a Dataset {
    type Item = QuadRef<'a>;
    type IntoIter = Iter<'a>;

    fn into_iter(self) -> Self::IntoIter {
        self.iter()
    }
}

impl FromIterator<Quad> for Dataset {
    fn from_iter<I: IntoIterator<Item = Quad>>(iter: I) -> Self {
        let mut g = Self::new();
        g.extend(iter);
        g
    }
}

impl<'a, Q: Into<QuadRef<'a>>> FromIterator<Q> for Dataset {
    fn from_iter<I: IntoIterator<Item = Q>>(iter: I) -> Self {
        let mut g = Self::new();
        g.extend(iter);
        g
    }
}

impl Extend<Quad> for Dataset {
    fn extend<I: IntoIterator<Item = Quad>>(&mut self, iter: I) {
        for q in iter {
            self.insert(&q);
        }
    }
}

impl<'a, Q: Into<QuadRef<'a>>> Extend<Q> for Dataset {
    fn extend<I: IntoIterator<Item = Q>>(&mut self, iter: I) {
        for q in iter {
            self.insert(q);
        }
    }
}

impl fmt::Display for Dataset {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        for t in self {
            writeln!(f, "{t} .")?;
        }
        Ok(())
    }
}

/// A read-only view on an [RDF graph](https://www.w3.org/TR/rdf11-concepts/#dfn-rdf-graph)
/// contained in a [`Dataset`].
///
/// It is built using the [`Dataset::graph`] method.
///
/// Usage example:
/// ```
/// use oxixml_model::*;
///
/// let mut dataset = Dataset::default();
/// let ex = NamedNodeRef::new("http://example.com")?;
/// dataset.insert(QuadRef::new(ex, ex, ex, ex));
///
/// let results: Vec<_> = dataset.graph(ex).iter().collect();
/// assert_eq!(vec![TripleRef::new(ex, ex, ex)], results);
/// # Result::<_, Box<dyn std::error::Error>>::Ok(())
/// ```
#[derive(Clone, Debug)]
pub struct GraphView<'a> {
    dataset: &'a Dataset,
    graph_name: InternedGraphName,
}

impl<'a> GraphView<'a> {
    /// Returns all the triples contained by the graph.
    pub fn iter(&self) -> GraphViewIter<'a> {
        let iter = self.dataset.gspo.range(
            &(
                self.graph_name,
                InternedNamedOrBlankNode::first(),
                InternedNamedNode::first(),
                InternedTerm::first(),
            )
                ..&(
                    self.graph_name.next(),
                    InternedNamedOrBlankNode::first(),
                    InternedNamedNode::first(),
                    InternedTerm::first(),
                ),
        );
        GraphViewIter {
            dataset: self.dataset,
            inner: iter,
        }
    }

    /// Returns all the triples of the graph with the given subject.
    pub fn triples_for_subject<'b>(
        &self,
        subject: impl Into<NamedOrBlankNodeRef<'b>>,
    ) -> impl Iterator<Item = TripleRef<'a>> + 'a {
        self.triples_for_interned_subject(self.dataset.encoded_named_or_blank_node(subject))
    }

    pub(super) fn triples_for_interned_subject(
        &self,
        subject: Option<InternedNamedOrBlankNode>,
    ) -> impl Iterator<Item = TripleRef<'a>> + use<'a> {
        let subject = subject.unwrap_or_else(InternedNamedOrBlankNode::impossible);
        let ds = self.dataset;
        self.dataset
            .gspo
            .range(
                &(
                    self.graph_name,
                    subject,
                    InternedNamedNode::first(),
                    InternedTerm::first(),
                )
                    ..&(
                        self.graph_name,
                        subject.next(),
                        InternedNamedNode::first(),
                        InternedTerm::first(),
                    ),
            )
            .map(move |q| {
                let (_, s, p, o) = q;
                ds.decode_spo((s, p, o))
            })
    }

    /// Returns all the objects of the graph with the given subject and predicate.
    pub fn objects_for_subject_predicate<'b>(
        &self,
        subject: impl Into<NamedOrBlankNodeRef<'b>>,
        predicate: impl Into<NamedNodeRef<'b>>,
    ) -> impl Iterator<Item = TermRef<'a>> + 'a {
        self.objects_for_interned_subject_predicate(
            self.dataset.encoded_named_or_blank_node(subject),
            self.dataset.encoded_named_node(predicate),
        )
    }

    pub(super) fn objects_for_interned_subject_predicate(
        &self,
        subject: Option<InternedNamedOrBlankNode>,
        predicate: Option<InternedNamedNode>,
    ) -> impl Iterator<Item = TermRef<'a>> + use<'a> {
        let subject = subject.unwrap_or_else(InternedNamedOrBlankNode::impossible);
        let predicate = predicate.unwrap_or_else(InternedNamedNode::impossible);
        let ds = self.dataset;
        self.dataset
            .gspo
            .range(
                &(self.graph_name, subject, predicate, InternedTerm::first())
                    ..&(
                        self.graph_name,
                        subject,
                        predicate.next(),
                        InternedTerm::first(),
                    ),
            )
            .map(move |q| q.3.decode_from(&ds.interner))
    }

    /// Returns the first object of the graph with the given subject and predicate, if any.
    pub fn object_for_subject_predicate<'b>(
        &self,
        subject: impl Into<NamedOrBlankNodeRef<'b>>,
        predicate: impl Into<NamedNodeRef<'b>>,
    ) -> Option<TermRef<'a>> {
        self.objects_for_subject_predicate(subject, predicate)
            .next()
    }

    /// Returns all the predicates of the graph with the given subject and object.
    pub fn predicates_for_subject_object<'b>(
        &self,
        subject: impl Into<NamedOrBlankNodeRef<'b>>,
        object: impl Into<TermRef<'b>>,
    ) -> impl Iterator<Item = NamedNodeRef<'a>> + 'a {
        self.predicates_for_interned_subject_object(
            self.dataset.encoded_named_or_blank_node(subject),
            self.dataset.encoded_term(object),
        )
    }

    pub(super) fn predicates_for_interned_subject_object(
        &self,
        subject: Option<InternedNamedOrBlankNode>,
        object: Option<InternedTerm>,
    ) -> impl Iterator<Item = NamedNodeRef<'a>> + use<'a> {
        let subject = subject.unwrap_or_else(InternedNamedOrBlankNode::impossible);
        let object = object.unwrap_or_else(InternedTerm::impossible);
        let ds = self.dataset;
        self.dataset
            .gosp
            .range(
                &(
                    self.graph_name,
                    object.clone(),
                    subject,
                    InternedNamedNode::first(),
                )
                    ..&(
                        self.graph_name,
                        object,
                        subject.next(),
                        InternedNamedNode::first(),
                    ),
            )
            .map(move |q| q.3.decode_from(&ds.interner))
    }

    /// Returns all the triples of the graph with the given predicate.
    pub fn triples_for_predicate<'b>(
        &self,
        predicate: impl Into<NamedNodeRef<'b>>,
    ) -> impl Iterator<Item = TripleRef<'a>> + 'a {
        self.triples_for_interned_predicate(self.dataset.encoded_named_node(predicate))
    }

    pub(super) fn triples_for_interned_predicate(
        &self,
        predicate: Option<InternedNamedNode>,
    ) -> impl Iterator<Item = TripleRef<'a>> + use<'a> {
        let predicate = predicate.unwrap_or_else(InternedNamedNode::impossible);
        let ds = self.dataset;
        self.dataset
            .gpos
            .range(
                &(
                    self.graph_name,
                    predicate,
                    InternedTerm::first(),
                    InternedNamedOrBlankNode::first(),
                )
                    ..&(
                        self.graph_name,
                        predicate.next(),
                        InternedTerm::first(),
                        InternedNamedOrBlankNode::first(),
                    ),
            )
            .map(move |(_, p, o, s)| ds.decode_spo((s, p, o)))
    }

    /// Returns all the subjects of the graph with the given predicate and object.
    pub fn subjects_for_predicate_object<'b>(
        &self,
        predicate: impl Into<NamedNodeRef<'b>>,
        object: impl Into<TermRef<'b>>,
    ) -> impl Iterator<Item = NamedOrBlankNodeRef<'a>> + 'a {
        self.subjects_for_interned_predicate_object(
            self.dataset.encoded_named_node(predicate),
            self.dataset.encoded_term(object),
        )
    }

    pub(super) fn subjects_for_interned_predicate_object(
        &self,
        predicate: Option<InternedNamedNode>,
        object: Option<InternedTerm>,
    ) -> impl Iterator<Item = NamedOrBlankNodeRef<'a>> + use<'a> {
        let predicate = predicate.unwrap_or_else(InternedNamedNode::impossible);
        let object = object.unwrap_or_else(InternedTerm::impossible);
        let ds = self.dataset;
        self.dataset
            .gpos
            .range(
                &(
                    self.graph_name,
                    predicate,
                    object.clone(),
                    InternedNamedOrBlankNode::first(),
                )
                    ..&(
                        self.graph_name,
                        predicate,
                        object.next(),
                        InternedNamedOrBlankNode::first(),
                    ),
            )
            .map(move |q| q.3.decode_from(&ds.interner))
    }

    /// Returns the first subject of the graph with the given predicate and object, if any.
    pub fn subject_for_predicate_object<'b>(
        &self,
        predicate: impl Into<NamedNodeRef<'b>>,
        object: impl Into<TermRef<'b>>,
    ) -> Option<NamedOrBlankNodeRef<'a>> {
        self.subjects_for_predicate_object(predicate, object).next()
    }

    /// Returns all the triples of the graph with the given object.
    pub fn triples_for_object<'b>(
        &self,
        object: impl Into<TermRef<'b>>,
    ) -> impl Iterator<Item = TripleRef<'a>> + 'a {
        self.triples_for_interned_object(self.dataset.encoded_term(object))
    }

    pub(super) fn triples_for_interned_object(
        &self,
        object: Option<InternedTerm>,
    ) -> impl Iterator<Item = TripleRef<'a>> + use<'a> {
        let object = object.unwrap_or_else(InternedTerm::impossible);
        let ds = self.dataset;
        self.dataset
            .gosp
            .range(
                &(
                    self.graph_name,
                    object.clone(),
                    InternedNamedOrBlankNode::first(),
                    InternedNamedNode::first(),
                )
                    ..&(
                        self.graph_name,
                        object.next(),
                        InternedNamedOrBlankNode::first(),
                        InternedNamedNode::first(),
                    ),
            )
            .map(move |(_, o, s, p)| ds.decode_spo((s, p, o)))
    }

    /// Retrieves triples with a filter on each triple component.
    pub fn triples_for_pattern(
        &self,
        subject: Option<NamedOrBlankNodeRef<'a>>,
        predicate: Option<NamedNodeRef<'a>>,
        object: Option<TermRef<'a>>,
    ) -> impl Iterator<Item = TripleRef<'a>> + 'a {
        let iter: Box<dyn Iterator<Item = _>> = match (subject, predicate, object) {
            (Some(subject), Some(predicate), Some(object)) => {
                let triple = TripleRef::new(subject, predicate, object);
                Box::new(self.contains(triple).then_some(triple).into_iter())
            }
            (Some(subject), Some(predicate), None) => Box::new(
                self.objects_for_subject_predicate(subject, predicate)
                    .map(move |object| TripleRef::new(subject, predicate, object)),
            ),
            (Some(subject), None, Some(object)) => Box::new(
                self.predicates_for_subject_object(subject, object)
                    .map(move |predicate| TripleRef::new(subject, predicate, object)),
            ),
            (Some(subject), None, None) => Box::new(self.triples_for_subject(subject)),
            (None, Some(predicate), Some(object)) => Box::new(
                self.subjects_for_predicate_object(predicate, object)
                    .map(move |subject| TripleRef::new(subject, predicate, object)),
            ),
            (None, Some(predicate), None) => Box::new(self.triples_for_predicate(predicate)),
            (None, None, Some(object)) => Box::new(self.triples_for_object(object)),
            (None, None, None) => Box::new(self.iter()),
        };
        iter
    }

    /// Checks if the graph contains the given triple.
    pub fn contains<'b>(&self, triple: impl Into<TripleRef<'b>>) -> bool {
        if let Some(triple) = self.encoded_triple(triple.into()) {
            self.dataset.gspo.contains(&(
                self.graph_name,
                triple.subject,
                triple.predicate,
                triple.object,
            ))
        } else {
            false
        }
    }

    /// Returns the number of triples in this graph.
    pub fn len(&self) -> usize {
        self.iter().count()
    }

    /// Checks if this graph is empty.
    pub fn is_empty(&self) -> bool {
        self.iter().next().is_none()
    }

    fn encoded_triple(&self, triple: TripleRef<'_>) -> Option<InternedTriple> {
        Some(InternedTriple {
            subject: self.dataset.encoded_named_or_blank_node(triple.subject)?,
            predicate: self.dataset.encoded_named_node(triple.predicate)?,
            object: self.dataset.encoded_term(triple.object)?,
        })
    }
}

impl<'a> IntoIterator for GraphView<'a> {
    type Item = TripleRef<'a>;
    type IntoIter = GraphViewIter<'a>;

    fn into_iter(self) -> Self::IntoIter {
        self.iter()
    }
}

impl<'a> IntoIterator for &GraphView<'a> {
    type Item = TripleRef<'a>;
    type IntoIter = GraphViewIter<'a>;

    fn into_iter(self) -> Self::IntoIter {
        self.iter()
    }
}

impl fmt::Display for GraphView<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        for t in self {
            writeln!(f, "{t} .")?;
        }
        Ok(())
    }
}

/// A read/write view on an [RDF graph](https://www.w3.org/TR/rdf11-concepts/#dfn-rdf-graph)
/// contained in a [`Dataset`].
///
/// It is built using the [`Dataset::graph_mut`] method.
///
/// Usage example:
/// ```
/// use oxixml_model::*;
///
/// let mut dataset = Dataset::default();
/// let ex = NamedNodeRef::new("http://example.com")?;
///
/// // We edit and query the dataset http://example.com graph
/// {
///     let mut graph = dataset.graph_mut(ex);
///     graph.insert(TripleRef::new(ex, ex, ex));
///     let results: Vec<_> = graph.iter().collect();
///     assert_eq!(vec![TripleRef::new(ex, ex, ex)], results);
/// }
///
/// // We have also changed the dataset itself
/// let results: Vec<_> = dataset.iter().collect();
/// assert_eq!(vec![QuadRef::new(ex, ex, ex, ex)], results);
/// # Result::<_, Box<dyn std::error::Error>>::Ok(())
/// ```
#[derive(Debug)]
pub struct GraphViewMut<'a> {
    dataset: &'a mut Dataset,
    graph_name: InternedGraphName,
}

impl<'a> GraphViewMut<'a> {
    fn read(&self) -> GraphView<'_> {
        GraphView {
            dataset: self.dataset,
            graph_name: self.graph_name,
        }
    }

    /// Adds a triple to the graph.
    pub fn insert<'b>(&mut self, triple: impl Into<TripleRef<'b>>) -> bool {
        let triple = self.encode_triple(triple.into());
        self.dataset.insert_encoded((
            triple.subject,
            triple.predicate,
            triple.object,
            self.graph_name,
        ))
    }

    /// Removes a concrete triple from the graph.
    pub fn remove<'b>(&mut self, triple: impl Into<TripleRef<'b>>) -> bool {
        if let Some(triple) = self.read().encoded_triple(triple.into()) {
            self.dataset.remove_encoded((
                triple.subject,
                triple.predicate,
                triple.object,
                self.graph_name,
            ))
        } else {
            false
        }
    }

    fn encode_triple(&mut self, triple: TripleRef<'_>) -> InternedTriple {
        InternedTriple {
            subject: InternedNamedOrBlankNode::encoded_into(
                triple.subject,
                &mut self.dataset.interner,
            ),
            predicate: InternedNamedNode::encoded_into(
                triple.predicate,
                &mut self.dataset.interner,
            ),
            object: InternedTerm::encoded_into(triple.object, &mut self.dataset.interner),
        }
    }

    /// Returns all the triples contained by the graph.
    pub fn iter(&'a self) -> GraphViewIter<'a> {
        self.read().iter()
    }

    /// Returns all the triples of the graph with the given subject.
    pub fn triples_for_subject<'b>(
        &'a self,
        subject: impl Into<NamedOrBlankNodeRef<'b>>,
    ) -> impl Iterator<Item = TripleRef<'a>> + 'a {
        self.read()
            .triples_for_interned_subject(self.dataset.encoded_named_or_blank_node(subject))
    }

    /// Returns all the objects of the graph with the given subject and predicate.
    pub fn objects_for_subject_predicate<'b>(
        &'a self,
        subject: impl Into<NamedOrBlankNodeRef<'b>>,
        predicate: impl Into<NamedNodeRef<'b>>,
    ) -> impl Iterator<Item = TermRef<'a>> + 'a {
        self.read().objects_for_interned_subject_predicate(
            self.dataset.encoded_named_or_blank_node(subject),
            self.dataset.encoded_named_node(predicate),
        )
    }

    /// Returns the first object of the graph with the given subject and predicate, if any.
    pub fn object_for_subject_predicate<'b>(
        &'a self,
        subject: impl Into<NamedOrBlankNodeRef<'b>>,
        predicate: impl Into<NamedNodeRef<'b>>,
    ) -> Option<TermRef<'a>> {
        self.read().object_for_subject_predicate(subject, predicate)
    }

    /// Returns all the predicates of the graph with the given subject and object.
    pub fn predicates_for_subject_object<'b>(
        &'a self,
        subject: impl Into<NamedOrBlankNodeRef<'b>>,
        object: impl Into<TermRef<'b>>,
    ) -> impl Iterator<Item = NamedNodeRef<'a>> + 'a {
        self.read().predicates_for_interned_subject_object(
            self.dataset.encoded_named_or_blank_node(subject),
            self.dataset.encoded_term(object),
        )
    }

    /// Returns all the triples of the graph with the given predicate.
    pub fn triples_for_predicate<'b>(
        &'a self,
        predicate: impl Into<NamedNodeRef<'b>>,
    ) -> impl Iterator<Item = TripleRef<'a>> + 'a {
        self.read()
            .triples_for_interned_predicate(self.dataset.encoded_named_node(predicate))
    }

    /// Returns all the subjects of the graph with the given predicate and object.
    pub fn subjects_for_predicate_object<'b>(
        &'a self,
        predicate: impl Into<NamedNodeRef<'b>>,
        object: impl Into<TermRef<'b>>,
    ) -> impl Iterator<Item = NamedOrBlankNodeRef<'a>> + 'a {
        self.read().subjects_for_interned_predicate_object(
            self.dataset.encoded_named_node(predicate),
            self.dataset.encoded_term(object),
        )
    }

    /// Returns the first subject of the graph with the given predicate and object, if any.
    pub fn subject_for_predicate_object<'b>(
        &'a self,
        predicate: impl Into<NamedNodeRef<'b>>,
        object: impl Into<TermRef<'b>>,
    ) -> Option<NamedOrBlankNodeRef<'a>> {
        self.read().subject_for_predicate_object(predicate, object)
    }

    /// Returns all the triples of the graph with the given object.
    pub fn triples_for_object<'b>(
        &'a self,
        object: TermRef<'b>,
    ) -> impl Iterator<Item = TripleRef<'a>> + 'a {
        self.read()
            .triples_for_interned_object(self.dataset.encoded_term(object))
    }

    /// Checks if the graph contains the given triple.
    pub fn contains<'b>(&self, triple: impl Into<TripleRef<'b>>) -> bool {
        self.read().contains(triple)
    }

    /// Returns the number of triples in this graph.
    pub fn len(&self) -> usize {
        self.read().len()
    }

    /// Checks if this graph is empty.
    pub fn is_empty(&self) -> bool {
        self.read().is_empty()
    }
}

impl Extend<Triple> for GraphViewMut<'_> {
    fn extend<I: IntoIterator<Item = Triple>>(&mut self, iter: I) {
        for t in iter {
            self.insert(&t);
        }
    }
}

impl<'b, T: Into<TripleRef<'b>>> Extend<T> for GraphViewMut<'_> {
    fn extend<I: IntoIterator<Item = T>>(&mut self, iter: I) {
        for t in iter {
            self.insert(t);
        }
    }
}

impl<'a> IntoIterator for &'a GraphViewMut<'a> {
    type Item = TripleRef<'a>;
    type IntoIter = GraphViewIter<'a>;

    fn into_iter(self) -> Self::IntoIter {
        self.iter()
    }
}

impl fmt::Display for GraphViewMut<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        for t in self {
            writeln!(f, "{t}")?;
        }
        Ok(())
    }
}

/// Iterator returned by [`Dataset::iter`].
pub struct Iter<'a> {
    dataset: &'a Dataset,
    inner: std::collections::btree_set::Iter<
        'a,
        (
            InternedNamedOrBlankNode,
            InternedNamedNode,
            InternedTerm,
            InternedGraphName,
        ),
    >,
}

impl<'a> Iterator for Iter<'a> {
    type Item = QuadRef<'a>;

    fn next(&mut self) -> Option<Self::Item> {
        self.inner
            .next()
            .map(|(s, p, o, g)| self.dataset.decode_spog((s, p, o, g)))
    }
}

/// Iterator returned by [`GraphView::iter`].
pub struct GraphViewIter<'a> {
    dataset: &'a Dataset,
    inner: std::collections::btree_set::Range<
        'a,
        (
            InternedGraphName,
            InternedNamedOrBlankNode,
            InternedNamedNode,
            InternedTerm,
        ),
    >,
}

impl<'a> Iterator for GraphViewIter<'a> {
    type Item = TripleRef<'a>;

    fn next(&mut self) -> Option<Self::Item> {
        self.inner
            .next()
            .map(|(_, s, p, o)| self.dataset.decode_spo((s, p, o)))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ex(name: &str) -> NamedNode {
        NamedNode::new_unchecked(format!("http://example.com/{name}"))
    }

    #[test]
    fn insert_len_contains_remove() {
        let mut dataset = Dataset::new();
        let quad = Quad::new(ex("s"), ex("p"), ex("o"), ex("g"));
        assert!(dataset.insert(&quad));
        assert!(!dataset.insert(&quad));
        assert_eq!(dataset.len(), 1);
        assert!(dataset.contains(&quad));
        assert!(dataset.remove(&quad));
        assert!(dataset.is_empty());
        assert!(!dataset.contains(&quad));
    }

    #[test]
    fn graph_view_round_trip() {
        let mut dataset = Dataset::default();
        let g = ex("g");
        dataset.insert(&Quad::new(ex("s"), ex("p"), ex("o"), g.clone()));
        let view = dataset.graph(&g);
        let triples: Vec<_> = view.iter().collect();
        let expected = Triple::new(ex("s"), ex("p"), ex("o"));
        assert_eq!(triples, vec![expected.as_ref()]);
        assert_eq!(view.len(), 1);
        assert!(dataset.graph(GraphNameRef::DefaultGraph).is_empty());
    }

    #[test]
    fn pattern_iteration() {
        let mut dataset = Dataset::new();
        dataset.insert(&Quad::new(
            ex("s"),
            ex("p"),
            ex("o1"),
            GraphName::DefaultGraph,
        ));
        dataset.insert(&Quad::new(
            ex("s"),
            ex("p"),
            ex("o2"),
            GraphName::DefaultGraph,
        ));
        let objects: Vec<_> = dataset
            .graph(GraphNameRef::DefaultGraph)
            .objects_for_subject_predicate(&ex("s"), &ex("p"))
            .collect();
        assert_eq!(objects.len(), 2);
        assert_eq!(dataset.quads_for_subject(&ex("s")).count(), 2);
    }

    #[test]
    fn display_is_nquads() {
        let mut dataset = Dataset::new();
        dataset.insert(&Quad::new(ex("s"), ex("p"), ex("o"), ex("g")));
        assert_eq!(
            dataset.to_string(),
            "<http://example.com/s> <http://example.com/p> <http://example.com/o> \
             <http://example.com/g> .\n"
        );
    }
}
