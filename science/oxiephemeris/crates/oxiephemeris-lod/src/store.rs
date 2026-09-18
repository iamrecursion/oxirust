//! A file-backed, pure-Rust SPARQL store.
//!
//! # Why a whole-file dump instead of `Store::open`
//!
//! oxigraph ships a durable, crash-safe, transactional backend — but it is
//! `RocksDB`, which is C++. The COOLJAPAN pure-Rust policy rules that out, so
//! this crate builds oxigraph with `default-features = false`, leaving only
//! the in-memory [`Store`]. [`FileBackedStore`] adds the thinnest possible
//! persistence on top of it: on [`save`](FileBackedStore::save) the *entire*
//! dataset is serialized to N-Quads and atomically renamed over the target
//! file.
//!
//! ## Honest trade-offs
//!
//! * **No write-ahead log, no incremental writes.** Every `save` rewrites
//!   the whole file. Cost is `O(triples)` per save, so this does not scale
//!   to very large datasets.
//! * **Not crash-safe *between* saves.** Updates live only in memory until
//!   `save` is called; a crash before then loses them. The `save` itself
//!   *is* atomic on a POSIX filesystem: we write a sibling `.tmp` file and
//!   [`rename`](std::fs::rename) it into place, so a reader never observes a
//!   half-written store and a crash mid-save leaves the previous good file
//!   intact.
//! * **Same-directory temp file.** The temp file is created next to the
//!   target (not in `$TMPDIR`) precisely so the rename stays on one
//!   filesystem, where it is guaranteed atomic.
//!
//! For the vocabulary-hosting and small-dataset workloads this endpoint is
//! built for, that is an acceptable price for zero C dependencies.

use std::fs::File;
use std::io::{BufReader, BufWriter, Read};
use std::path::{Path, PathBuf};

use oxigraph::io::RdfFormat;
use oxigraph::sparql::{QueryResults, SparqlEvaluator};
use oxigraph::store::Store;

// This crate straddles two RDF term models. `oxiephemeris_rdf::oxrdf` is
// the model the published documents are built in; `oxigraph::model` —
// aliased `store_model` below — is the store's own. They are distinct
// types with the same names, so the store side is always spelled out.
use oxiephemeris_rdf::oxrdf::{Graph, NamedOrBlankNode, Term};
use oxigraph::model as store_model;

use crate::error::LodError;

/// Copies a subject from the document model into the store model.
fn store_subject(subject: &NamedOrBlankNode) -> store_model::NamedOrBlankNode {
    match subject {
        NamedOrBlankNode::NamedNode(node) => {
            store_model::NamedNode::new_unchecked(node.as_str()).into()
        }
        NamedOrBlankNode::BlankNode(node) => {
            store_model::BlankNode::new_unchecked(node.as_str()).into()
        }
    }
}

/// Copies an object term from the document model into the store model.
///
/// A literal is rebuilt from the three components that define its identity
/// in RDF 1.1 — lexical form, datatype IRI, and language tag — so the copy
/// is the same literal, not merely a similar one. A simple literal has
/// datatype `xsd:string` and no language tag, which
/// [`Literal::new_typed_literal`](store_model::Literal::new_typed_literal)
/// reproduces exactly.
fn store_term(object: &Term) -> store_model::Term {
    match object {
        Term::NamedNode(node) => store_model::NamedNode::new_unchecked(node.as_str()).into(),
        Term::BlankNode(node) => store_model::BlankNode::new_unchecked(node.as_str()).into(),
        Term::Literal(literal) => match literal.language() {
            // The source literal already carries a validated, lowercase
            // language tag, which is what makes `_unchecked` sound here.
            Some(tag) => {
                store_model::Literal::new_language_tagged_literal_unchecked(literal.value(), tag)
            }
            None => store_model::Literal::new_typed_literal(
                literal.value(),
                store_model::NamedNode::new_unchecked(literal.datatype().as_str()),
            ),
        }
        .into(),
    }
}

/// An in-memory oxigraph [`Store`] with optional whole-file persistence.
///
/// Construct with [`FileBackedStore::open`]; persist with
/// [`FileBackedStore::save`]. When opened without a path the store is purely
/// in-memory and `save` is a no-op.
pub struct FileBackedStore {
    store: Store,
    path: Option<PathBuf>,
}

impl FileBackedStore {
    /// Opens a store, loading `path` if it is given and already exists.
    ///
    /// The on-disk format is N-Quads. A `None` path (or a path that does not
    /// yet exist) yields an empty store; in the latter case the path is
    /// remembered so a later [`save`](Self::save) creates it.
    ///
    /// # Errors
    ///
    /// [`LodError::Storage`] if the in-memory store cannot be created,
    /// [`LodError::Io`] if the file cannot be opened, or
    /// [`LodError::Loader`] if its contents are not valid N-Quads.
    pub fn open(path: Option<&Path>) -> Result<Self, LodError> {
        let store = Store::new()?;
        if let Some(p) = path {
            if p.exists() {
                let reader = BufReader::new(File::open(p)?);
                store.load_from_reader(RdfFormat::NQuads, reader)?;
            }
        }
        Ok(Self {
            store,
            path: path.map(Path::to_path_buf),
        })
    }

    /// Atomically writes the whole dataset to the configured path.
    ///
    /// Serializes to a sibling `<name>.tmp` file, then renames it over the
    /// target so the replacement is atomic on a POSIX filesystem. Does
    /// nothing (and succeeds) if no path was configured.
    ///
    /// # Errors
    ///
    /// [`LodError::Io`] if the temp file cannot be written or the rename
    /// fails, or [`LodError::Serializer`] if serialization fails.
    pub fn save(&self) -> Result<(), LodError> {
        let Some(path) = &self.path else {
            return Ok(());
        };
        let Some(file_name) = path.file_name() else {
            return Err(LodError::Io(std::io::Error::new(
                std::io::ErrorKind::InvalidInput,
                "store path has no file name",
            )));
        };
        // Temp file next to the target: same filesystem => atomic rename.
        let mut tmp_name = file_name.to_os_string();
        tmp_name.push(".tmp");
        let tmp_path = path.with_file_name(tmp_name);

        let writer = BufWriter::new(File::create(&tmp_path)?);
        let writer = self.store.dump_to_writer(RdfFormat::NQuads, writer)?;
        // Flush the BufWriter down to the file before the rename.
        writer
            .into_inner()
            .map_err(|e| LodError::Io(e.into_error()))?
            .sync_all()?;
        std::fs::rename(&tmp_path, path)?;
        Ok(())
    }

    /// Runs a SPARQL query, distinguishing syntax from evaluation errors.
    ///
    /// The query is parsed first so a malformed query surfaces as
    /// [`LodError::Syntax`] (which the endpoint maps to `400`) rather than
    /// being conflated with a genuine evaluation failure.
    ///
    /// # Errors
    ///
    /// [`LodError::Syntax`] if the query does not parse, or
    /// [`LodError::QueryEval`] if it fails during evaluation.
    pub fn query(&self, sparql: &str) -> Result<QueryResults<'static>, LodError> {
        let prepared = SparqlEvaluator::new().parse_query(sparql)?;
        Ok(prepared.on_store(&self.store).execute()?)
    }

    /// Runs a SPARQL update against the in-memory store.
    ///
    /// This does **not** persist the change; call [`save`](Self::save)
    /// afterwards to write it to disk. Takes `&self` because oxigraph's
    /// [`Store`] uses interior mutability.
    ///
    /// # Errors
    ///
    /// [`LodError::Syntax`] if the update does not parse, or
    /// [`LodError::UpdateEval`] if it fails during evaluation.
    pub fn update(&self, sparql: &str) -> Result<(), LodError> {
        let prepared = SparqlEvaluator::new().parse_update(sparql)?;
        prepared.on_store(&self.store).execute()?;
        Ok(())
    }

    /// Loads RDF from `reader` in the given `format` into the default graph.
    ///
    /// Used by the binary's `--load` option. Blank nodes are renamed to
    /// avoid clashing with data already in the store.
    ///
    /// # Errors
    ///
    /// [`LodError::Loader`] if the input is not valid RDF in `format` or the
    /// insert fails.
    pub fn load_from_reader(&self, format: RdfFormat, reader: impl Read) -> Result<(), LodError> {
        self.store.load_from_reader(format, reader)?;
        Ok(())
    }

    /// Inserts every triple of an in-memory [`Graph`] as a quad.
    ///
    /// With `name = None` the triples go into the default graph; with
    /// `Some(g)` they go into the named graph `g`.
    ///
    /// The graph is built in the document model and the store keeps its
    /// own, so each term is copied across. The copy is structural rather
    /// than a serialize/parse round trip, which is what keeps blank-node
    /// identity intact.
    ///
    /// # Errors
    ///
    /// [`LodError::Storage`] if any insert fails.
    pub fn load_graph(
        &self,
        graph: &Graph,
        name: Option<store_model::NamedNodeRef<'_>>,
    ) -> Result<(), LodError> {
        let graph_name = name.map_or(store_model::GraphName::DefaultGraph, |g| {
            store_model::GraphName::NamedNode(g.into_owned())
        });
        // Iterating a `Graph` yields borrows into it; the copy helpers take
        // the owned document-model terms, so each borrowed term is owned
        // out for the length of the statement that copies it across.
        for triple in graph {
            let subject = store_subject(&triple.subject.into_owned());
            let predicate = store_model::NamedNode::new_unchecked(triple.predicate.as_str());
            let object = store_term(&triple.object.into_owned());
            self.store.insert(store_model::QuadRef::new(
                subject.as_ref(),
                predicate.as_ref(),
                object.as_ref(),
                graph_name.as_ref(),
            ))?;
        }
        Ok(())
    }

    /// The number of quads in the store.
    ///
    /// # Errors
    ///
    /// [`LodError::Storage`] if the store cannot be read.
    pub fn len(&self) -> Result<usize, LodError> {
        Ok(self.store.len()?)
    }

    /// Whether the store holds no quads.
    ///
    /// # Errors
    ///
    /// [`LodError::Storage`] if the store cannot be read.
    pub fn is_empty(&self) -> Result<bool, LodError> {
        Ok(self.store.is_empty()?)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use oxiephemeris_rdf::concept_scheme_graph;
    // Iterating a document-model `Graph` yields borrowed terms.
    use oxiephemeris_rdf::oxrdf::TermRef;
    use std::sync::atomic::{AtomicU64, Ordering};

    /// A per-call unique temp directory, distinct across threads and runs.
    fn temp_dir() -> PathBuf {
        static COUNTER: AtomicU64 = AtomicU64::new(0);
        let seq = COUNTER.fetch_add(1, Ordering::Relaxed);
        let mut dir = std::env::temp_dir();
        dir.push(format!("oxieph-lod-store-{}-{seq}", std::process::id()));
        let Ok(()) = std::fs::create_dir_all(&dir) else {
            panic!("cannot create temp dir {}", dir.display());
        };
        dir
    }

    #[test]
    fn load_save_round_trip_leaves_no_temp_behind() {
        let dir = temp_dir();
        let path = dir.join("store.nq");

        let vocab = concept_scheme_graph();
        let vocab_len = vocab.len();

        {
            let Ok(store) = FileBackedStore::open(Some(&path)) else {
                panic!("open must succeed");
            };
            let Ok(()) = store.load_graph(&vocab, None) else {
                panic!("load_graph must succeed");
            };
            let Ok(len) = store.len() else {
                panic!("len must succeed");
            };
            assert_eq!(len, vocab_len, "loaded triple count must match graph");
            let Ok(()) = store.save() else {
                panic!("save must succeed");
            };
        }

        // The atomic rename must not leave a `.tmp` sibling behind.
        let tmp = dir.join("store.nq.tmp");
        assert!(!tmp.exists(), "temp file must be renamed away");
        assert!(path.exists(), "store file must exist after save");

        // Reopening must reload exactly the same data.
        let Ok(reopened) = FileBackedStore::open(Some(&path)) else {
            panic!("reopen must succeed");
        };
        let Ok(len) = reopened.len() else {
            panic!("len must succeed");
        };
        assert_eq!(len, vocab_len, "reloaded triple count must match");

        let _ = std::fs::remove_dir_all(&dir);
    }

    /// `load_graph` copies terms between two RDF models, so counting the
    /// quads it produced would not notice a literal that arrived with its
    /// language tag dropped or its datatype replaced. This censuses the
    /// object terms by kind on both sides — the source graph in the
    /// document model, the loaded quads in the store model — and requires
    /// the three tallies to agree.
    #[test]
    fn load_graph_preserves_every_object_term_kind() {
        let vocab = concept_scheme_graph();

        let (mut want_iri, mut want_lang, mut want_typed) = (0_usize, 0_usize, 0_usize);
        for triple in &vocab {
            match triple.object {
                TermRef::NamedNode(_) => want_iri += 1,
                TermRef::BlankNode(_) => panic!("the vocabulary must be blank-node free"),
                TermRef::Literal(literal) => {
                    if literal.language().is_some() {
                        want_lang += 1;
                    } else {
                        want_typed += 1;
                    }
                }
            }
        }
        assert!(
            want_iri > 0 && want_lang > 0 && want_typed > 0,
            "thin sample"
        );

        let Ok(store) = FileBackedStore::open(None) else {
            panic!("open must succeed");
        };
        let Ok(()) = store.load_graph(&vocab, None) else {
            panic!("load_graph must succeed");
        };

        let (mut got_iri, mut got_lang, mut got_typed) = (0_usize, 0_usize, 0_usize);
        for quad in &store.store {
            let Ok(quad) = quad else {
                panic!("the store must be readable");
            };
            match &quad.object {
                store_model::Term::NamedNode(_) => got_iri += 1,
                store_model::Term::BlankNode(_) => panic!("no blank node may appear"),
                store_model::Term::Literal(literal) => {
                    if literal.language().is_some() {
                        got_lang += 1;
                    } else {
                        got_typed += 1;
                    }
                }
            }
        }

        assert_eq!(got_iri, want_iri, "IRI objects lost or invented");
        assert_eq!(got_lang, want_lang, "a language tag did not survive");
        assert_eq!(got_typed, want_typed, "a typed literal did not survive");
    }

    /// A spot check on the actual terms, built with the *store's* own
    /// constructors so it cannot agree with `load_graph` by sharing its
    /// logic: one IRI object, one language-tagged literal, one typed
    /// literal — the three shapes `store_term` distinguishes.
    #[test]
    fn load_graph_reproduces_representative_terms() {
        let vocab = concept_scheme_graph();
        let Ok(store) = FileBackedStore::open(None) else {
            panic!("open must succeed");
        };
        let Ok(()) = store.load_graph(&vocab, None) else {
            panic!("load_graph must succeed");
        };

        // The published Scorpio concept IRI. Spelled out rather than
        // re-derived so this also pins that the copy lands on the very IRI
        // the service dereferences.
        let subject = store_model::NamedNode::new_unchecked(
            "https://cooljapan.tech/ns/oxiephemeris/concept/sign/Scorpio",
        );

        let expected = [
            store_model::Quad::new(
                subject.clone(),
                store_model::NamedNode::new_unchecked(
                    "http://www.w3.org/1999/02/22-rdf-syntax-ns#type",
                ),
                store_model::NamedNode::new_unchecked(
                    "http://www.w3.org/2004/02/skos/core#Concept",
                ),
                store_model::GraphName::DefaultGraph,
            ),
            store_model::Quad::new(
                subject.clone(),
                store_model::NamedNode::new_unchecked(
                    "http://www.w3.org/2004/02/skos/core#prefLabel",
                ),
                store_model::Literal::new_language_tagged_literal_unchecked("Scorpio", "en"),
                store_model::GraphName::DefaultGraph,
            ),
            store_model::Quad::new(
                subject,
                store_model::NamedNode::new_unchecked(
                    "https://cooljapan.tech/ns/oxiephemeris/astro#signIndex",
                ),
                // `Sign::index` is 0-based, so Scorpio is 7.
                store_model::Literal::new_typed_literal(
                    "7",
                    store_model::NamedNode::new_unchecked(
                        "http://www.w3.org/2001/XMLSchema#integer",
                    ),
                ),
                store_model::GraphName::DefaultGraph,
            ),
        ];

        for quad in &expected {
            let Ok(found) = store.store.contains(quad) else {
                panic!("the store must be readable");
            };
            assert!(found, "term did not survive the model copy: {quad}");
        }
    }

    #[test]
    fn missing_path_opens_empty_and_save_creates_file() {
        let dir = temp_dir();
        let path = dir.join("fresh.nq");
        assert!(!path.exists());

        let Ok(store) = FileBackedStore::open(Some(&path)) else {
            panic!("open must succeed");
        };
        let Ok(empty) = store.is_empty() else {
            panic!("is_empty must succeed");
        };
        assert!(empty, "a fresh store must be empty");
        let Ok(()) = store.save() else {
            panic!("save must succeed");
        };
        assert!(path.exists(), "save must create the file");

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn in_memory_store_save_is_noop() {
        let Ok(store) = FileBackedStore::open(None) else {
            panic!("open must succeed");
        };
        let Ok(()) = store.save() else {
            panic!("save on an in-memory store must be a no-op success");
        };
    }
}
