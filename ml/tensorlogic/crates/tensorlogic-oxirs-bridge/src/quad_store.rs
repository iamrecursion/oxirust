//! Multi-named-graph storage with per-graph query API.
//!
//! `QuadStore` holds one [`TripleStore`] per named graph (including the default graph
//! represented as `None`). It exposes query methods that mirror what `TripleStore`
//! already provides, plus a subject-level lookup backed by an internal index because
//! `TripleStore` only supports predicate/object-anchored access.
//!
//! # Example
//!
//! ```
//! use tensorlogic_oxirs_bridge::QuadStore;
//! use tensorlogic_oxirs_bridge::schema::nquads::Quad;
//!
//! let mut qs = QuadStore::new();
//! qs.insert_quad(Quad::new(
//!     "http://example.org/Alice".into(),
//!     "http://example.org/type".into(),
//!     "http://example.org/Person".into(),
//!     None,
//! ));
//! qs.insert_quad(Quad::new(
//!     "http://example.org/Bob".into(),
//!     "http://example.org/type".into(),
//!     "http://example.org/Agent".into(),
//!     Some("http://g1".into()),
//! ));
//!
//! let results = qs.query_subject(None, "http://example.org/Alice");
//! assert!(!results.is_empty());
//!
//! let empty = qs.query_subject(Some("http://g1"), "http://example.org/Alice");
//! assert!(empty.is_empty());
//! ```

use std::collections::HashMap;

use crate::property_path::TripleStore;
use crate::schema::nquads::Quad;

/// Key type for per-graph maps: `None` = default graph, `Some(iri)` = named graph.
type GraphKey = Option<String>;
/// Subject index for a single graph: subject IRI → `(predicate, object)` pairs.
type SubjectIndex = HashMap<String, Vec<(String, String)>>;

/// Multi-named-graph store.
///
/// `None` represents the default (unnamed) graph; `Some(iri)` represents a named graph.
///
/// Internally each graph is stored as a [`TripleStore`] (for the property-path query
/// methods `objects_from`, `subjects_to`, `triples_with_predicate`) plus a parallel
/// subject index (`HashMap<String, Vec<(String, String)>>`) because `TripleStore` does
/// not expose a public subject-anchored lookup.
pub struct QuadStore {
    /// Per-graph triple stores, keyed by graph IRI (None = default graph).
    stores: HashMap<GraphKey, TripleStore>,
    /// Per-graph subject index: subject → list of (predicate, object) pairs.
    subject_index: HashMap<GraphKey, SubjectIndex>,
}

impl QuadStore {
    /// Create an empty `QuadStore`.
    pub fn new() -> Self {
        Self {
            stores: HashMap::new(),
            subject_index: HashMap::new(),
        }
    }

    /// Insert a quad into the appropriate named graph.
    ///
    /// Both the underlying [`TripleStore`] and the subject index are updated.
    pub fn insert_quad(&mut self, q: Quad) {
        let graph_key = q.graph.clone();

        // Update the TripleStore for this graph.
        let store = self.stores.entry(graph_key.clone()).or_default();
        store.add(q.subject.clone(), q.predicate.clone(), q.object.clone());

        // Update the subject index for this graph.
        self.subject_index
            .entry(graph_key)
            .or_default()
            .entry(q.subject)
            .or_default()
            .push((q.predicate, q.object));
    }

    // -------------------------------------------------------------------------
    // Subject-anchored queries (backed by the parallel index)
    // -------------------------------------------------------------------------

    /// Return all `(predicate, object)` pairs for `subject` in the given graph.
    ///
    /// Pass `None` for the default graph, `Some(iri)` for a named graph.
    pub fn query_subject(&self, graph: Option<&str>, subject: &str) -> Vec<(String, String)> {
        let key = graph.map(str::to_string);
        self.subject_index
            .get(&key)
            .and_then(|idx| idx.get(subject))
            .cloned()
            .unwrap_or_default()
    }

    // -------------------------------------------------------------------------
    // Predicate-anchored queries (delegate to TripleStore)
    // -------------------------------------------------------------------------

    /// Return all `(subject, object)` pairs for `predicate` in the given graph.
    pub fn query_predicate(&self, graph: Option<&str>, predicate: &str) -> Vec<(String, String)> {
        let key = graph.map(str::to_string);
        self.stores
            .get(&key)
            .map(|store| {
                store
                    .triples_with_predicate(predicate)
                    .into_iter()
                    .map(|(s, o)| (s.to_string(), o.to_string()))
                    .collect()
            })
            .unwrap_or_default()
    }

    // -------------------------------------------------------------------------
    // Object-anchored queries (delegate to TripleStore via subjects_to)
    // -------------------------------------------------------------------------

    /// Return all `(subject, predicate)` pairs whose object equals `object` in the given graph.
    ///
    /// This iterates over all predicates stored in the graph and gathers matching triples.
    /// For large graphs the predicate-first scan in `TripleStore` is unavoidable because
    /// `TripleStore` does not expose a public object index; for the typical quad-store
    /// sizes targeted here this is acceptable.
    pub fn query_object(&self, graph: Option<&str>, object: &str) -> Vec<(String, String)> {
        let key = graph.map(str::to_string);
        self.stores
            .get(&key)
            .map(|store| {
                // Collect all unique predicates from the subject index, then use
                // subjects_to() which is already provided by TripleStore.
                let graph_idx = self.subject_index.get(&key);
                let predicates: std::collections::HashSet<String> = graph_idx
                    .map(|idx| {
                        idx.values()
                            .flat_map(|pairs| pairs.iter().map(|(p, _)| p.clone()))
                            .collect()
                    })
                    .unwrap_or_default();

                predicates
                    .into_iter()
                    .flat_map(|pred| {
                        store
                            .subjects_to(object, &pred)
                            .into_iter()
                            .map(|s| (s.to_string(), pred.clone()))
                            .collect::<Vec<_>>()
                    })
                    .collect()
            })
            .unwrap_or_default()
    }

    // -------------------------------------------------------------------------
    // Graph-level accessors
    // -------------------------------------------------------------------------

    /// Iterate over all graph keys (including `None` for the default graph if populated).
    pub fn iter_graphs(&self) -> impl Iterator<Item = &Option<String>> {
        self.stores.keys()
    }

    /// Number of distinct graphs (including the default graph if it contains any quads).
    pub fn graph_count(&self) -> usize {
        self.stores.len()
    }

    /// Total number of quads across all graphs.
    pub fn total_quads(&self) -> usize {
        self.stores.values().map(|s| s.len()).sum()
    }

    /// Return a reference to the underlying [`TripleStore`] for a graph, if it exists.
    pub fn get_store(&self, graph: Option<&str>) -> Option<&TripleStore> {
        let key = graph.map(str::to_string);
        self.stores.get(&key)
    }

    /// Whether the store contains any quads at all.
    pub fn is_empty(&self) -> bool {
        self.stores.values().all(|s| s.is_empty())
    }
}

impl Default for QuadStore {
    fn default() -> Self {
        Self::new()
    }
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    fn make_quad(s: &str, p: &str, o: &str, g: Option<&str>) -> Quad {
        Quad {
            subject: s.to_string(),
            predicate: p.to_string(),
            object: o.to_string(),
            graph: g.map(|x| x.to_string()),
        }
    }

    #[test]
    fn test_per_graph_isolation() {
        let mut qs = QuadStore::new();
        qs.insert_quad(make_quad("Alice", "type", "Person", None));
        qs.insert_quad(make_quad("Bob", "type", "Agent", Some("http://g1")));
        qs.insert_quad(make_quad(
            "Carol",
            "type",
            "Organization",
            Some("http://g2"),
        ));

        // Default graph contains Alice but not Bob or Carol.
        let default_results = qs.query_subject(None, "Alice");
        assert!(!default_results.is_empty());
        let g1_results = qs.query_subject(Some("http://g1"), "Alice");
        assert!(g1_results.is_empty()); // Alice not in g1

        // g1 contains Bob.
        let bob_results = qs.query_subject(Some("http://g1"), "Bob");
        assert!(!bob_results.is_empty());
    }

    #[test]
    fn test_iter_graphs() {
        let mut qs = QuadStore::new();
        qs.insert_quad(make_quad("x", "p", "o", None));
        qs.insert_quad(make_quad("x", "p", "o", Some("http://g1")));
        qs.insert_quad(make_quad("x", "p", "o", Some("http://g2")));
        let graphs: Vec<_> = qs.iter_graphs().collect();
        assert_eq!(graphs.len(), 3);
    }

    #[test]
    fn test_query_predicate() {
        let mut qs = QuadStore::new();
        qs.insert_quad(make_quad("Alice", "knows", "Bob", None));
        qs.insert_quad(make_quad("Alice", "knows", "Carol", None));
        qs.insert_quad(make_quad("Dave", "knows", "Eve", Some("http://g1")));

        let pairs = qs.query_predicate(None, "knows");
        assert_eq!(pairs.len(), 2);

        // g1 "knows" should not bleed into default graph
        let g1_pairs = qs.query_predicate(Some("http://g1"), "knows");
        assert_eq!(g1_pairs.len(), 1);

        // Missing graph returns empty
        let missing = qs.query_predicate(Some("http://missing"), "knows");
        assert!(missing.is_empty());
    }

    #[test]
    fn test_query_object() {
        let mut qs = QuadStore::new();
        qs.insert_quad(make_quad("Alice", "type", "Person", None));
        qs.insert_quad(make_quad("Bob", "type", "Person", None));
        qs.insert_quad(make_quad("Carol", "type", "Agent", None));

        let persons = qs.query_object(None, "Person");
        assert_eq!(persons.len(), 2);

        let agents = qs.query_object(None, "Agent");
        assert_eq!(agents.len(), 1);
    }

    #[test]
    fn test_default_graph_isolation() {
        let mut qs = QuadStore::new();
        qs.insert_quad(make_quad("X", "p", "Y", None));
        qs.insert_quad(make_quad("X", "p", "Z", Some("http://named")));

        // Default graph only has one triple
        assert_eq!(qs.query_subject(None, "X").len(), 1);
        // Named graph only has one triple
        assert_eq!(qs.query_subject(Some("http://named"), "X").len(), 1);
    }

    #[test]
    fn test_graph_count_and_total() {
        let mut qs = QuadStore::new();
        assert_eq!(qs.graph_count(), 0);
        assert_eq!(qs.total_quads(), 0);
        assert!(qs.is_empty());

        qs.insert_quad(make_quad("a", "b", "c", None));
        qs.insert_quad(make_quad("a", "b", "c", Some("http://g1")));
        assert_eq!(qs.graph_count(), 2);
        assert_eq!(qs.total_quads(), 2);
        assert!(!qs.is_empty());
    }

    #[test]
    fn test_subject_returns_correct_predicate_object_pairs() {
        let mut qs = QuadStore::new();
        qs.insert_quad(make_quad("Alice", "knows", "Bob", None));
        qs.insert_quad(make_quad("Alice", "age", "30", None));

        let pairs = qs.query_subject(None, "Alice");
        assert_eq!(pairs.len(), 2);

        let knows_pair = pairs.iter().find(|(p, _)| p == "knows");
        assert!(knows_pair.is_some());
        assert_eq!(knows_pair.map(|(_, o)| o.as_str()), Some("Bob"));
    }

    #[test]
    fn test_into_quad_store_via_nquads_processor() {
        use crate::schema::nquads::NQuadsProcessor;

        let nquads = r#"<http://example.org/Alice> <http://example.org/knows> <http://example.org/Bob> <http://example.org/g1> .
<http://example.org/Bob> <http://example.org/knows> <http://example.org/Charlie> .
<http://example.org/Dave> <http://example.org/age> "42" <http://example.org/g2> .
"#;

        let mut processor = NQuadsProcessor::new();
        processor.load_nquads(nquads).expect("load_nquads failed");

        let qs = processor.into_quad_store();
        assert_eq!(qs.graph_count(), 3); // default + g1 + g2
        assert_eq!(qs.total_quads(), 3);

        // Alice is in g1
        let alice = qs.query_subject(Some("http://example.org/g1"), "http://example.org/Alice");
        assert_eq!(alice.len(), 1);

        // Bob is in default graph
        let bob = qs.query_subject(None, "http://example.org/Bob");
        assert_eq!(bob.len(), 1);
    }
}
