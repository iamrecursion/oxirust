//! Intern-based term dictionary and triple store for the tensor SPARQL evaluator.
//!
//! [`InternedGraph`] maps every RDF term (IRI or literal string) to a compact `u32`
//! identifier and stores triples as `(subject_id, predicate_id, object_id)` tuples.
//! Lookups are O(1) via two complementary indexes:
//! - `dict`: term string → id
//! - `by_predicate`: predicate_id → list of (subject_id, object_id) pairs
//!
//! # Parallel bulk load
//!
//! When the input contains ≤1_000_000 triples the constructor spawns one thread
//! per CPU (using `std::thread::scope`) to parse chunks in parallel.  Each thread
//! builds a local `HashMap<&str, ()>` of unique terms; the main thread then merges
//! all unique strings into one global dictionary before filling the adjacency
//! indexes in a single sequential pass.  For inputs larger than the threshold the
//! whole process is sequential.
//!
//! In `#[cfg(test)]` the threshold is overridden to 4 so that even small inputs
//! exercise the parallel path.

use std::collections::HashMap;
use std::collections::HashSet;

use crate::quad_store::QuadStore;
use crate::rdf_bulk_io::{BulkIoError, RdfBulkImporter, RdfTriple};
use crate::schema::nquads::Quad;

// ─── threshold ───────────────────────────────────────────────────────────────

#[cfg(not(test))]
const PARALLEL_THRESHOLD: usize = 1_000_000;

/// Override for tests: use parallel path even for tiny inputs.
#[cfg(test)]
const PARALLEL_THRESHOLD: usize = 4;

// ─── InternedGraph ────────────────────────────────────────────────────────────

/// Term dictionary + triple store backed by interned `u32` IDs.
pub struct InternedGraph {
    /// term string → compact u32 ID
    dict: HashMap<String, u32>,
    /// compact u32 ID → term string (reverse lookup)
    terms: Vec<String>,
    /// All triples as (subject_id, predicate_id, object_id)
    pub(crate) triples: Vec<(u32, u32, u32)>,
    /// Predicate-anchored index: pred_id → [(subj_id, obj_id)]
    by_predicate: HashMap<u32, Vec<(u32, u32)>>,
}

impl InternedGraph {
    /// Create an empty graph.
    pub fn new() -> Self {
        Self {
            dict: HashMap::new(),
            terms: Vec::new(),
            triples: Vec::new(),
            by_predicate: HashMap::new(),
        }
    }

    /// Intern `term`: return existing ID if present, otherwise allocate a new one.
    pub fn intern(&mut self, term: &str) -> u32 {
        if let Some(&id) = self.dict.get(term) {
            return id;
        }
        let id = self.dict.len() as u32;
        self.dict.insert(term.to_string(), id);
        self.terms.push(term.to_string());
        id
    }

    /// Intern without inserting.  Returns `None` when the term is unknown.
    pub fn intern_or_none(&self, term: &str) -> Option<u32> {
        self.dict.get(term).copied()
    }

    /// Resolve an ID back to the original term string.
    pub fn term(&self, id: u32) -> Option<&str> {
        self.terms.get(id as usize).map(String::as_str)
    }

    /// Add a triple, interning each component as needed.
    pub fn add_triple(&mut self, s: &str, p: &str, o: &str) {
        let s_id = self.intern(s);
        let p_id = self.intern(p);
        let o_id = self.intern(o);
        self.triples.push((s_id, p_id, o_id));
        self.by_predicate
            .entry(p_id)
            .or_default()
            .push((s_id, o_id));
    }

    /// Return all `(subject_id, object_id)` pairs for the given predicate ID.
    /// Returns an empty slice when the predicate has no triples.
    pub fn predicate_pairs(&self, pred_id: u32) -> &[(u32, u32)] {
        match self.by_predicate.get(&pred_id) {
            Some(pairs) => pairs.as_slice(),
            None => &[],
        }
    }

    /// Number of distinct terms in the dictionary.
    pub fn num_entities(&self) -> usize {
        self.dict.len()
    }

    /// Total number of triples stored.
    pub fn num_triples(&self) -> usize {
        self.triples.len()
    }

    /// Build an [`InternedGraph`] from a collection of [`RdfTriple`]s.
    ///
    /// Uses parallel term-collection when `triples.len() <= PARALLEL_THRESHOLD`,
    /// single-threaded otherwise.
    pub fn from_rdf_triples(triples: Vec<RdfTriple>) -> Self {
        if triples.is_empty() {
            return Self::new();
        }

        let num_triples = triples.len();
        let use_parallel = num_triples <= PARALLEL_THRESHOLD;

        if use_parallel {
            Self::from_rdf_triples_parallel(triples)
        } else {
            Self::from_rdf_triples_sequential(triples)
        }
    }

    // ── Sequential (fallback) ─────────────────────────────────────────────────

    fn from_rdf_triples_sequential(triples: Vec<RdfTriple>) -> Self {
        let mut g = Self::new();
        for t in &triples {
            g.add_triple(&t.subject, &t.predicate, &t.object);
        }
        g
    }

    // ── Parallel ──────────────────────────────────────────────────────────────

    /// Parallel build strategy:
    /// 1. Split input into chunks (one per logical CPU, capped at 8).
    /// 2. Each thread collects the set of unique term strings from its chunk.
    /// 3. Main thread merges all unique strings → builds global dict.
    /// 4. Sequential pass over all triples to resolve IDs and fill indexes.
    fn from_rdf_triples_parallel(triples: Vec<RdfTriple>) -> Self {
        let raw_cpus = std::thread::available_parallelism()
            .map(|n| n.get())
            .unwrap_or(2);
        let num_cpus = raw_cpus.clamp(1, 8);
        let num_chunks = num_cpus.min(triples.len()).max(1);

        eprintln!(
            "[InternedGraph] parallel bulk-load: {} triples, {} chunks",
            triples.len(),
            num_chunks
        );

        let chunk_size = triples.len().div_ceil(num_chunks);
        let chunks: Vec<&[RdfTriple]> = triples.chunks(chunk_size).collect();

        // Phase 1: parallel term collection — gather unique strings per chunk
        let mut per_chunk_terms: Vec<HashSet<String>> =
            (0..chunks.len()).map(|_| HashSet::new()).collect();

        std::thread::scope(|scope| {
            let mut handles = Vec::with_capacity(chunks.len());
            for chunk in &chunks {
                let handle = scope.spawn(|| {
                    let mut local: HashSet<String> = HashSet::new();
                    for t in chunk.iter() {
                        local.insert(t.subject.clone());
                        local.insert(t.predicate.clone());
                        local.insert(t.object.clone());
                    }
                    local
                });
                handles.push(handle);
            }
            for (i, handle) in handles.into_iter().enumerate() {
                // Scope threads are guaranteed to finish before scope exits,
                // so joining them here is always valid.
                per_chunk_terms[i] = handle.join().unwrap_or_default();
            }
        });

        // Phase 2: merge per-chunk term sets into a sorted global list
        // (sorting is deterministic regardless of HashMap iteration order)
        let mut all_terms: HashSet<String> = HashSet::new();
        for chunk_terms in per_chunk_terms {
            all_terms.extend(chunk_terms);
        }
        let mut sorted_terms: Vec<String> = all_terms.into_iter().collect();
        sorted_terms.sort_unstable();

        // Phase 3: build dict/terms from sorted list
        let mut dict: HashMap<String, u32> = HashMap::with_capacity(sorted_terms.len());
        let mut terms_vec: Vec<String> = Vec::with_capacity(sorted_terms.len());
        for (idx, term) in sorted_terms.into_iter().enumerate() {
            dict.insert(term.clone(), idx as u32);
            terms_vec.push(term);
        }

        // Phase 4: single-pass to build adjacency using resolved IDs
        let num_triples = triples.len();
        let mut stored_triples: Vec<(u32, u32, u32)> = Vec::with_capacity(num_triples);
        let mut by_predicate: HashMap<u32, Vec<(u32, u32)>> = HashMap::new();

        for t in &triples {
            // SAFETY: all terms were collected in phase 1; every lookup succeeds.
            let s_id = *dict.get(t.subject.as_str()).unwrap_or(&0);
            let p_id = *dict.get(t.predicate.as_str()).unwrap_or(&0);
            let o_id = *dict.get(t.object.as_str()).unwrap_or(&0);
            stored_triples.push((s_id, p_id, o_id));
            by_predicate.entry(p_id).or_default().push((s_id, o_id));
        }

        Self {
            dict,
            terms: terms_vec,
            triples: stored_triples,
            by_predicate,
        }
    }

    // ── Conversion helpers ────────────────────────────────────────────────────

    /// Emit a [`QuadStore`] (default graph only) containing all stored triples.
    pub fn into_quad_store(&self) -> QuadStore {
        let mut qs = QuadStore::new();
        for (s_id, p_id, o_id) in &self.triples {
            let s = self.terms[*s_id as usize].clone();
            let p = self.terms[*p_id as usize].clone();
            let o = self.terms[*o_id as usize].clone();
            qs.insert_quad(Quad {
                subject: s,
                predicate: p,
                object: o,
                graph: None,
            });
        }
        qs
    }
}

impl Default for InternedGraph {
    fn default() -> Self {
        Self::new()
    }
}

// ─── Free function ────────────────────────────────────────────────────────────

/// Parse `input` with `importer` (auto-detecting format) and return an
/// [`InternedGraph`].
pub fn rdf_bulk_importer_into_interned(
    importer: &RdfBulkImporter,
    input: &str,
) -> Result<InternedGraph, BulkIoError> {
    let (triples, _stats) = importer.parse_auto(input)?;
    Ok(InternedGraph::from_rdf_triples(triples))
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── intern / term round-trip ──────────────────────────────────────────────

    #[test]
    fn test_intern_idempotent() {
        let mut g = InternedGraph::new();
        let id1 = g.intern("Alice");
        let id2 = g.intern("Alice");
        assert_eq!(id1, id2, "intern must be idempotent");
        let id3 = g.intern("Bob");
        assert_ne!(id1, id3, "different terms must have different IDs");
    }

    #[test]
    fn test_term_round_trip() {
        let mut g = InternedGraph::new();
        let id = g.intern("http://example.org/Alice");
        assert_eq!(g.term(id), Some("http://example.org/Alice"));
    }

    #[test]
    fn test_intern_or_none_known_and_unknown() {
        let mut g = InternedGraph::new();
        g.intern("known");
        assert!(g.intern_or_none("known").is_some());
        assert!(g.intern_or_none("unknown").is_none());
    }

    // ── add_triple / predicate_pairs ──────────────────────────────────────────

    #[test]
    fn test_add_triple_and_predicate_pairs() {
        let mut g = InternedGraph::new();
        g.add_triple("Alice", "knows", "Bob");
        g.add_triple("Alice", "knows", "Carol");
        g.add_triple("Bob", "age", "30");

        let knows_id = g.intern_or_none("knows").expect("knows must be interned");
        // Collect pairs into a Vec to release the borrow before calling intern()
        let pairs: Vec<(u32, u32)> = g.predicate_pairs(knows_id).to_vec();
        assert_eq!(pairs.len(), 2, "should have 2 knows pairs");

        let alice_id = g.intern("Alice");
        let bob_id = g.intern("Bob");
        let carol_id = g.intern("Carol");
        assert!(
            pairs.contains(&(alice_id, bob_id)),
            "Alice knows Bob should be present"
        );
        assert!(
            pairs.contains(&(alice_id, carol_id)),
            "Alice knows Carol should be present"
        );
    }

    #[test]
    fn test_predicate_pairs_absent_returns_empty() {
        let g = InternedGraph::new();
        assert_eq!(g.predicate_pairs(99), &[]);
    }

    // ── num_entities / num_triples ────────────────────────────────────────────

    #[test]
    fn test_num_entities_and_triples() {
        let mut g = InternedGraph::new();
        assert_eq!(g.num_entities(), 0);
        assert_eq!(g.num_triples(), 0);

        g.add_triple("A", "p", "B");
        // 3 unique terms: A, p, B
        assert_eq!(g.num_entities(), 3);
        assert_eq!(g.num_triples(), 1);

        // Adding same terms again should not create new entities
        g.add_triple("A", "p", "B");
        assert_eq!(g.num_entities(), 3);
        assert_eq!(g.num_triples(), 2);
    }

    // ── from_rdf_triples (sequential vs parallel) ─────────────────────────────

    #[test]
    fn test_from_rdf_triples_bulk() {
        let triples = vec![
            RdfTriple::new("Alice", "knows", "Bob"),
            RdfTriple::new("Bob", "knows", "Carol"),
            RdfTriple::new("Carol", "age", "25"),
        ];
        let g = InternedGraph::from_rdf_triples(triples);
        assert_eq!(g.num_triples(), 3);
        // terms: Alice, knows, Bob, Carol, age, 25 → 6 unique
        assert_eq!(g.num_entities(), 6);
    }

    #[test]
    fn test_parallel_equivalent_to_sequential() {
        // With PARALLEL_THRESHOLD = 4 (test override), inputs of len <= 4
        // take the parallel path.
        let triples: Vec<RdfTriple> = vec![
            RdfTriple::new("s1", "p", "o1"),
            RdfTriple::new("s2", "p", "o2"),
            RdfTriple::new("s3", "p", "o3"),
        ];

        let parallel = InternedGraph::from_rdf_triples(triples.clone());
        let sequential = InternedGraph::from_rdf_triples_sequential(triples);

        assert_eq!(parallel.num_triples(), sequential.num_triples());
        assert_eq!(parallel.num_entities(), sequential.num_entities());

        let p_id = parallel.intern_or_none("p").expect("p must be interned");
        let pairs_p = parallel.predicate_pairs(p_id);
        assert_eq!(pairs_p.len(), 3);

        let q_id = sequential.intern_or_none("p").expect("p must be interned");
        let pairs_q = sequential.predicate_pairs(q_id);
        assert_eq!(pairs_q.len(), 3);
    }

    // ── into_quad_store ───────────────────────────────────────────────────────

    #[test]
    fn test_into_quad_store_contains_correct_triples() {
        let triples = vec![
            RdfTriple::new("Alice", "knows", "Bob"),
            RdfTriple::new("Bob", "knows", "Carol"),
        ];
        let g = InternedGraph::from_rdf_triples(triples);
        let qs = g.into_quad_store();
        // default graph should have 2 triples
        assert_eq!(qs.total_quads(), 2);
        let pairs = qs.query_predicate(None, "knows");
        assert_eq!(pairs.len(), 2);
    }

    // ── rdf_bulk_importer_into_interned ───────────────────────────────────────

    #[test]
    fn test_bulk_importer_into_interned_ntriples() {
        let input = "<Alice> <knows> <Bob> .\n<Bob> <knows> <Carol> .\n";
        let importer = RdfBulkImporter::new();
        let g =
            rdf_bulk_importer_into_interned(&importer, input).expect("bulk import should succeed");
        assert_eq!(g.num_triples(), 2);
    }
}
