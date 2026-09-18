//! [In-memory implementation](Graph) of [RDF graphs](https://www.w3.org/TR/rdf11-concepts/#dfn-rdf-graph).
//!
//! Usage example:
//! ```
//! use oxixml_model::*;
//!
//! let mut graph = Graph::default();
//!
//! // insertion
//! let ex = NamedNodeRef::new("http://example.com")?;
//! let triple = TripleRef::new(ex, ex, ex);
//! graph.insert(triple);
//!
//! // simple filter
//! let results: Vec<_> = graph.triples_for_subject(ex).collect();
//! assert_eq!(vec![triple], results);
//!
//! // Print
//! assert_eq!(
//!     graph.to_string(),
//!     "<http://example.com> <http://example.com> <http://example.com> .\n"
//! );
//! # Result::<_, Box<dyn std::error::Error>>::Ok(())
//! ```
//!
//! See also [`Dataset`] if you want support for multiple RDF graphs at the same
//! time.

use crate::dataset::*;
pub use crate::dataset::{
    CanonicalizationAlgorithm, CanonicalizationError, CanonicalizationHashAlgorithm,
    DEFAULT_CANONICALIZATION_WORK_BUDGET,
};
use crate::*;
use std::fmt;

/// An in-memory [RDF graph](https://www.w3.org/TR/rdf11-concepts/#dfn-rdf-graph).
///
/// It can accommodate a fairly large number of triples (in the few millions).
///
/// <div class="warning">It interns the strings and does not do any garbage collection yet:
/// if you insert and remove a lot of different terms, memory will grow without any reduction.</div>
///
/// Usage example:
/// ```
/// use oxixml_model::*;
///
/// let mut graph = Graph::default();
///
/// // insertion
/// let ex = NamedNodeRef::new("http://example.com")?;
/// let triple = TripleRef::new(ex, ex, ex);
/// graph.insert(triple);
///
/// // simple filter
/// let results: Vec<_> = graph.triples_for_subject(ex).collect();
/// assert_eq!(vec![triple], results);
/// # Result::<_, Box<dyn std::error::Error>>::Ok(())
/// ```
#[derive(Debug, Default, Clone)]
pub struct Graph {
    dataset: Dataset,
}

impl Graph {
    /// Creates a new graph.
    pub fn new() -> Self {
        Self::default()
    }

    fn graph(&self) -> GraphView<'_> {
        self.dataset.graph(GraphNameRef::DefaultGraph)
    }

    fn graph_mut(&mut self) -> GraphViewMut<'_> {
        self.dataset.graph_mut(GraphNameRef::DefaultGraph)
    }

    /// Returns all the triples contained by the graph.
    pub fn iter(&self) -> Iter<'_> {
        Iter {
            inner: self.graph().iter(),
        }
    }

    /// Returns all the triples with the given subject.
    pub fn triples_for_subject<'a, 'b>(
        &'a self,
        subject: impl Into<NamedOrBlankNodeRef<'b>>,
    ) -> impl Iterator<Item = TripleRef<'a>> + 'a {
        self.graph()
            .triples_for_interned_subject(self.dataset.encoded_named_or_blank_node(subject))
    }

    /// Returns all the objects with the given subject and predicate.
    pub fn objects_for_subject_predicate<'a, 'b>(
        &'a self,
        subject: impl Into<NamedOrBlankNodeRef<'b>>,
        predicate: impl Into<NamedNodeRef<'b>>,
    ) -> impl Iterator<Item = TermRef<'a>> + 'a {
        self.graph().objects_for_interned_subject_predicate(
            self.dataset.encoded_named_or_blank_node(subject),
            self.dataset.encoded_named_node(predicate),
        )
    }

    /// Returns the first object with the given subject and predicate, if any.
    pub fn object_for_subject_predicate<'a, 'b>(
        &'a self,
        subject: impl Into<NamedOrBlankNodeRef<'b>>,
        predicate: impl Into<NamedNodeRef<'b>>,
    ) -> Option<TermRef<'a>> {
        self.graph()
            .objects_for_subject_predicate(subject, predicate)
            .next()
    }

    /// Returns all the predicates with the given subject and object.
    pub fn predicates_for_subject_object<'a, 'b>(
        &'a self,
        subject: impl Into<NamedOrBlankNodeRef<'b>>,
        object: impl Into<TermRef<'b>>,
    ) -> impl Iterator<Item = NamedNodeRef<'a>> + 'a {
        self.graph().predicates_for_interned_subject_object(
            self.dataset.encoded_named_or_blank_node(subject),
            self.dataset.encoded_term(object),
        )
    }

    /// Returns all the triples with the given predicate.
    pub fn triples_for_predicate<'a, 'b>(
        &'a self,
        predicate: impl Into<NamedNodeRef<'b>>,
    ) -> impl Iterator<Item = TripleRef<'a>> + 'a {
        self.graph()
            .triples_for_interned_predicate(self.dataset.encoded_named_node(predicate))
    }

    /// Returns all the subjects with the given predicate and object.
    pub fn subjects_for_predicate_object<'a, 'b>(
        &'a self,
        predicate: impl Into<NamedNodeRef<'b>>,
        object: impl Into<TermRef<'b>>,
    ) -> impl Iterator<Item = NamedOrBlankNodeRef<'a>> + 'a {
        self.graph().subjects_for_interned_predicate_object(
            self.dataset.encoded_named_node(predicate),
            self.dataset.encoded_term(object),
        )
    }

    /// Returns the first subject with the given predicate and object, if any.
    pub fn subject_for_predicate_object<'a, 'b>(
        &'a self,
        predicate: impl Into<NamedNodeRef<'b>>,
        object: impl Into<TermRef<'b>>,
    ) -> Option<NamedOrBlankNodeRef<'a>> {
        self.graph().subject_for_predicate_object(predicate, object)
    }

    /// Returns all the triples with the given object.
    pub fn triples_for_object<'a, 'b>(
        &'a self,
        object: impl Into<TermRef<'b>>,
    ) -> impl Iterator<Item = TripleRef<'a>> + 'a {
        self.graph()
            .triples_for_interned_object(self.dataset.encoded_term(object))
    }

    /// Checks if the graph contains the given triple.
    pub fn contains<'a>(&self, triple: impl Into<TripleRef<'a>>) -> bool {
        self.graph().contains(triple)
    }

    /// Returns the number of triples in this graph.
    pub fn len(&self) -> usize {
        self.dataset.len()
    }

    /// Checks if this graph is empty.
    pub fn is_empty(&self) -> bool {
        self.dataset.is_empty()
    }

    /// Adds a triple to the graph.
    pub fn insert<'a>(&mut self, triple: impl Into<TripleRef<'a>>) -> bool {
        self.graph_mut().insert(triple)
    }

    /// Removes a concrete triple from the graph.
    pub fn remove<'a>(&mut self, triple: impl Into<TripleRef<'a>>) -> bool {
        self.graph_mut().remove(triple)
    }

    /// Clears the graph.
    pub fn clear(&mut self) {
        self.dataset.clear()
    }

    /// Canonicalizes the graph by renaming blank nodes.
    ///
    /// Usage example ([Graph isomorphism](https://www.w3.org/TR/rdf11-concepts/#dfn-graph-isomorphism)):
    /// ```
    /// use oxixml_model::graph::CanonicalizationAlgorithm;
    /// use oxixml_model::*;
    ///
    /// let iri = NamedNodeRef::new("http://example.com")?;
    ///
    /// let mut graph1 = Graph::new();
    /// let bnode1 = BlankNode::default();
    /// graph1.insert(TripleRef::new(iri, iri, &bnode1));
    /// graph1.insert(TripleRef::new(&bnode1, iri, iri));
    ///
    /// let mut graph2 = Graph::new();
    /// let bnode2 = BlankNode::default();
    /// graph2.insert(TripleRef::new(iri, iri, &bnode2));
    /// graph2.insert(TripleRef::new(&bnode2, iri, iri));
    ///
    /// assert_ne!(graph1, graph2);
    /// graph1.canonicalize(CanonicalizationAlgorithm::Unstable);
    /// graph2.canonicalize(CanonicalizationAlgorithm::Unstable);
    /// assert_eq!(graph1, graph2);
    /// # Result::<_, Box<dyn std::error::Error>>::Ok(())
    /// ```
    ///
    /// <div class="warning">Blank node ids depend on the current shape of the graph. Adding a new
    /// quad might change the ids of a lot of blank nodes. Hence, this canonicalization might not
    /// be suitable for diffs.</div>
    ///
    /// <div class="warning">This implementation's worst-case complexity is exponential with
    /// respect to the number of blank nodes in the input graph: an adversarially crafted graph
    /// can make this method run for a very long time. Prefer
    /// [`try_canonicalize`](Self::try_canonicalize), which runs under a bounded work budget and
    /// reports [`CanonicalizationError::WorkBudgetExceeded`] instead, when the input is not
    /// trusted.</div>
    pub fn canonicalize(&mut self, algorithm: CanonicalizationAlgorithm) {
        self.dataset.canonicalize(algorithm)
    }

    /// Canonicalizes the graph by renaming blank nodes, like [`canonicalize`](Self::canonicalize),
    /// but rejects the input instead of running unboundedly if doing so would exceed the
    /// [`DEFAULT_CANONICALIZATION_WORK_BUDGET`].
    ///
    /// Prefer this over [`canonicalize`](Self::canonicalize) when the input graph is not
    /// trusted. See [`try_canonicalize_with_budget`](Self::try_canonicalize_with_budget) to use a
    /// different budget.
    ///
    /// # Errors
    ///
    /// Returns [`CanonicalizationError::WorkBudgetExceeded`] if the work budget is exhausted
    /// before a canonical form could be produced. The graph is left unmodified in that case.
    pub fn try_canonicalize(
        &mut self,
        algorithm: CanonicalizationAlgorithm,
    ) -> Result<(), CanonicalizationError> {
        self.dataset.try_canonicalize(algorithm)
    }

    /// Like [`try_canonicalize`](Self::try_canonicalize), with an explicit work budget (in
    /// abstract work units) instead of [`DEFAULT_CANONICALIZATION_WORK_BUDGET`]. Pass `u64::MAX`
    /// for an effectively unlimited budget (as [`canonicalize`](Self::canonicalize) does).
    ///
    /// # Errors
    ///
    /// Returns [`CanonicalizationError::WorkBudgetExceeded`] if `work_budget` is exhausted before
    /// a canonical form could be produced. The graph is left unmodified in that case.
    pub fn try_canonicalize_with_budget(
        &mut self,
        algorithm: CanonicalizationAlgorithm,
        work_budget: u64,
    ) -> Result<(), CanonicalizationError> {
        self.dataset
            .try_canonicalize_with_budget(algorithm, work_budget)
    }
}

impl PartialEq for Graph {
    fn eq(&self, other: &Self) -> bool {
        self.dataset == other.dataset
    }
}

impl Eq for Graph {}

impl<'a> IntoIterator for &'a Graph {
    type Item = TripleRef<'a>;
    type IntoIter = Iter<'a>;

    fn into_iter(self) -> Self::IntoIter {
        self.iter()
    }
}

impl FromIterator<Triple> for Graph {
    fn from_iter<I: IntoIterator<Item = Triple>>(iter: I) -> Self {
        let mut g = Self::new();
        g.extend(iter);
        g
    }
}

impl<'a, T: Into<TripleRef<'a>>> FromIterator<T> for Graph {
    fn from_iter<I: IntoIterator<Item = T>>(iter: I) -> Self {
        let mut g = Self::new();
        g.extend(iter);
        g
    }
}

impl Extend<Triple> for Graph {
    fn extend<I: IntoIterator<Item = Triple>>(&mut self, iter: I) {
        self.graph_mut().extend(iter)
    }
}

impl<'a, T: Into<TripleRef<'a>>> Extend<T> for Graph {
    fn extend<I: IntoIterator<Item = T>>(&mut self, iter: I) {
        self.graph_mut().extend(iter)
    }
}

impl fmt::Display for Graph {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.graph().fmt(f)
    }
}

/// Iterator returned by [`Graph::iter`].
pub struct Iter<'a> {
    inner: GraphViewIter<'a>,
}

impl<'a> Iterator for Iter<'a> {
    type Item = TripleRef<'a>;

    fn next(&mut self) -> Option<Self::Item> {
        self.inner.next()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ex(name: &str) -> NamedNode {
        NamedNode::new_unchecked(format!("http://example.com/{name}"))
    }

    #[test]
    fn insert_iter_remove() {
        let mut graph = Graph::new();
        let triple = Triple::new(ex("s"), ex("p"), ex("o"));
        assert!(graph.insert(&triple));
        assert!(!graph.insert(&triple));
        assert_eq!(graph.len(), 1);
        assert!(graph.contains(&triple));
        let collected: Vec<_> = graph.iter().collect();
        assert_eq!(collected, vec![triple.as_ref()]);
        assert!(graph.remove(&triple));
        assert!(graph.is_empty());
    }

    #[test]
    fn from_iter_and_display() {
        let graph: Graph = [Triple::new(ex("s"), ex("p"), ex("o"))]
            .into_iter()
            .collect();
        assert_eq!(
            graph.to_string(),
            "<http://example.com/s> <http://example.com/p> <http://example.com/o> .\n"
        );
    }

    #[test]
    fn canonicalize_isomorphic_graphs_equal() {
        let iri = ex("i");
        let mut graph1 = Graph::new();
        let bnode1 = BlankNode::default();
        graph1.insert(TripleRef::new(&iri, &iri, &bnode1));
        graph1.insert(TripleRef::new(&bnode1, &iri, &iri));

        let mut graph2 = Graph::new();
        let bnode2 = BlankNode::default();
        graph2.insert(TripleRef::new(&iri, &iri, &bnode2));
        graph2.insert(TripleRef::new(&bnode2, &iri, &iri));

        assert_ne!(graph1, graph2);
        graph1.canonicalize(CanonicalizationAlgorithm::Unstable);
        graph2.canonicalize(CanonicalizationAlgorithm::Unstable);
        assert_eq!(graph1, graph2);
    }
}
