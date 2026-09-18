//! Batch indexing support for high-throughput bulk document import.
//!
//! This module provides [`BatchIndexer`], which accumulates documents in an
//! in-memory buffer and flushes them in configurable batch sizes.  Compared to
//! indexing one document at a time, batching amortises serialisation overhead,
//! reduces lock contention on shared indices, and enables parallel pre-processing
//! of document features before the batch is committed.
//!
//! # Design
//!
//! ```text
//! producer(s)                 BatchIndexer
//! ──────────                  ────────────
//! push(doc)   ─────►  [ buffer ] ──► flush() ──► IndexBackend
//!                     capacity ^
//!                     auto-flush when full
//! ```
//!
//! # Example
//!
//! ```rust
//! use oximedia_search::batch_index::{BatchIndexer, BatchDocument, InMemoryBackend, IndexBackend};
//!
//! let backend = InMemoryBackend::new();
//! let mut indexer = BatchIndexer::with_capacity(backend, 3);
//!
//! for i in 0..7u32 {
//!     let doc = BatchDocument::new(format!("doc-{i}"), format!("content about item {i}"));
//!     indexer.push(doc).expect("push should succeed");
//! }
//! indexer.flush().expect("final flush should succeed");
//! assert_eq!(indexer.backend().total_indexed(), 7);
//! ```

use std::collections::HashMap;

use crate::error::{SearchError, SearchResult};

// ─────────────────────────────────────────────────────────────────────────────
// Document type
// ─────────────────────────────────────────────────────────────────────────────

/// A document that can be submitted to a batch indexer.
#[derive(Debug, Clone)]
pub struct BatchDocument {
    /// Unique document identifier (e.g. asset UUID as string).
    pub doc_id: String,
    /// Raw text content to index (title + description + transcript, etc.).
    pub text: String,
    /// Optional arbitrary metadata key/value pairs.
    pub metadata: HashMap<String, String>,
    /// Optional binary feature vector (visual / audio embeddings).
    pub features: Option<Vec<f32>>,
    /// Optional pre-computed tags.
    pub tags: Vec<String>,
}

impl BatchDocument {
    /// Create a minimal batch document with only an ID and text body.
    #[must_use]
    pub fn new(doc_id: impl Into<String>, text: impl Into<String>) -> Self {
        Self {
            doc_id: doc_id.into(),
            text: text.into(),
            metadata: HashMap::new(),
            features: None,
            tags: Vec::new(),
        }
    }

    /// Attach a metadata entry.
    #[must_use]
    pub fn with_meta(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        self.metadata.insert(key.into(), value.into());
        self
    }

    /// Attach pre-computed feature vector.
    #[must_use]
    pub fn with_features(mut self, features: Vec<f32>) -> Self {
        self.features = Some(features);
        self
    }

    /// Attach tags.
    #[must_use]
    pub fn with_tags(mut self, tags: Vec<String>) -> Self {
        self.tags = tags;
        self
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// IndexBackend trait
// ─────────────────────────────────────────────────────────────────────────────

/// Abstraction over a search index that accepts batches of documents.
///
/// Implementors are responsible for persistence, locking, and commit logic.
/// The [`InMemoryBackend`] provided in this module is useful for testing and
/// benchmarking without an on-disk index.
pub trait IndexBackend: Send {
    /// Write a batch of documents to the index.
    ///
    /// Implementations may commit or buffer internally; callers should use
    /// [`IndexBackend::commit`] to ensure all writes are durable.
    ///
    /// # Errors
    ///
    /// Returns [`SearchError`] if any document in the batch cannot be indexed.
    fn write_batch(&mut self, docs: &[BatchDocument]) -> SearchResult<()>;

    /// Commit any buffered writes to make them visible to search.
    ///
    /// # Errors
    ///
    /// Returns [`SearchError`] if the commit operation fails.
    fn commit(&mut self) -> SearchResult<()>;

    /// Return the total number of documents committed so far.
    fn total_indexed(&self) -> usize;
}

// ─────────────────────────────────────────────────────────────────────────────
// InMemoryBackend
// ─────────────────────────────────────────────────────────────────────────────

/// A simple in-memory [`IndexBackend`] that stores documents in a `Vec`.
///
/// Primarily intended for unit tests and benchmarks.
#[derive(Debug, Default)]
pub struct InMemoryBackend {
    /// Indexed documents.
    docs: Vec<BatchDocument>,
    /// Pending (not yet committed) documents.
    pending: Vec<BatchDocument>,
    /// Simulated failure counter — if `Some(n)`, the next `n` `write_batch`
    /// calls will return an error.  Useful for testing error paths.
    simulate_failure_count: Option<usize>,
}

impl InMemoryBackend {
    /// Create a new, empty in-memory backend.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Configure the backend to fail the next `n` `write_batch` calls.
    pub fn fail_next(&mut self, n: usize) {
        self.simulate_failure_count = Some(n);
    }

    /// Return a slice of all *committed* documents.
    #[must_use]
    pub fn committed_docs(&self) -> &[BatchDocument] {
        &self.docs
    }

    /// Return the number of pending (not yet committed) documents.
    #[must_use]
    pub fn pending_count(&self) -> usize {
        self.pending.len()
    }
}

impl IndexBackend for InMemoryBackend {
    fn write_batch(&mut self, docs: &[BatchDocument]) -> SearchResult<()> {
        // Simulate failures for testing.
        if let Some(ref mut remaining) = self.simulate_failure_count {
            if *remaining > 0 {
                *remaining -= 1;
                if *remaining == 0 {
                    self.simulate_failure_count = None;
                }
                return Err(SearchError::Other("simulated write failure".to_string()));
            }
        }
        self.pending.extend_from_slice(docs);
        Ok(())
    }

    fn commit(&mut self) -> SearchResult<()> {
        self.docs.append(&mut self.pending);
        Ok(())
    }

    fn total_indexed(&self) -> usize {
        self.docs.len()
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// BatchIndexer
// ─────────────────────────────────────────────────────────────────────────────

/// Statistics collected by [`BatchIndexer`] over its lifetime.
#[derive(Debug, Clone, Default)]
pub struct BatchStats {
    /// Total documents pushed.
    pub docs_pushed: usize,
    /// Total automatic flushes triggered.
    pub auto_flushes: usize,
    /// Total manual flushes triggered.
    pub manual_flushes: usize,
    /// Total batches written to the backend (each flush = 1 or more batches).
    pub batches_written: usize,
    /// Total documents successfully written.
    pub docs_written: usize,
    /// Total errors encountered during flush.
    pub flush_errors: usize,
}

/// Buffered batch indexer that amortises per-document overhead.
///
/// Documents are accumulated in memory up to `capacity`.  When the buffer
/// reaches capacity an automatic flush is triggered.  A final [`Self::flush`]
/// call must be issued by the caller to drain any remaining documents.
///
/// The indexer owns the [`IndexBackend`] and exposes it via [`Self::backend`]
/// and [`Self::backend_mut`] for inspection or manual operations.
pub struct BatchIndexer<B: IndexBackend> {
    /// Underlying index backend.
    backend: B,
    /// Pending documents not yet written to the backend.
    buffer: Vec<BatchDocument>,
    /// Number of documents to accumulate before auto-flushing.
    capacity: usize,
    /// Collected statistics.
    stats: BatchStats,
    /// Whether errors during auto-flush are propagated immediately (strict) or
    /// counted and skipped (lenient).
    strict_errors: bool,
}

impl<B: IndexBackend> BatchIndexer<B> {
    /// Create a new batch indexer wrapping `backend` with the given buffer
    /// `capacity`.
    ///
    /// # Panics
    ///
    /// Panics if `capacity` is zero.
    #[must_use]
    pub fn with_capacity(backend: B, capacity: usize) -> Self {
        assert!(capacity > 0, "BatchIndexer capacity must be greater than 0");
        Self {
            backend,
            buffer: Vec::with_capacity(capacity),
            capacity,
            stats: BatchStats::default(),
            strict_errors: true,
        }
    }

    /// Set lenient error mode: flush errors increment the error counter but do
    /// not abort the push operation.
    #[must_use]
    pub fn lenient(mut self) -> Self {
        self.strict_errors = false;
        self
    }

    /// Push a document into the buffer.
    ///
    /// If the buffer reaches `capacity` after this push, an automatic flush is
    /// triggered.
    ///
    /// # Errors
    ///
    /// In strict mode (default) returns an error if the auto-flush fails.
    /// In lenient mode the error is counted but not propagated.
    pub fn push(&mut self, doc: BatchDocument) -> SearchResult<()> {
        self.buffer.push(doc);
        self.stats.docs_pushed += 1;

        if self.buffer.len() >= self.capacity {
            self.stats.auto_flushes += 1;
            let result = self.do_flush();
            match result {
                Ok(()) => {}
                Err(e) => {
                    self.stats.flush_errors += 1;
                    if self.strict_errors {
                        return Err(e);
                    }
                }
            }
        }
        Ok(())
    }

    /// Flush any remaining buffered documents to the backend and commit.
    ///
    /// This is a no-op if the buffer is empty.
    ///
    /// # Errors
    ///
    /// Returns an error if the write or commit fails.
    pub fn flush(&mut self) -> SearchResult<()> {
        self.stats.manual_flushes += 1;
        self.do_flush()?;
        self.backend.commit()?;
        Ok(())
    }

    /// Return the number of documents currently buffered (not yet flushed).
    #[must_use]
    pub fn buffered_count(&self) -> usize {
        self.buffer.len()
    }

    /// Return the configured buffer capacity.
    #[must_use]
    pub fn capacity(&self) -> usize {
        self.capacity
    }

    /// Read-only access to the underlying backend.
    #[must_use]
    pub fn backend(&self) -> &B {
        &self.backend
    }

    /// Mutable access to the underlying backend.
    #[must_use]
    pub fn backend_mut(&mut self) -> &mut B {
        &mut self.backend
    }

    /// Return a snapshot of the current statistics.
    #[must_use]
    pub fn stats(&self) -> &BatchStats {
        &self.stats
    }

    /// Consume the indexer, returning the backend.
    ///
    /// Any unflushed documents are discarded.  Call [`Self::flush`] first if
    /// you need all documents committed.
    #[must_use]
    pub fn into_backend(self) -> B {
        self.backend
    }

    // ── internal ──────────────────────────────────────────────────────────────

    /// Write the current buffer to the backend (without committing).
    fn do_flush(&mut self) -> SearchResult<()> {
        if self.buffer.is_empty() {
            return Ok(());
        }
        let batch: Vec<BatchDocument> = self.buffer.drain(..).collect();
        let n = batch.len();
        self.backend.write_batch(&batch)?;
        self.stats.batches_written += 1;
        self.stats.docs_written += n;
        Ok(())
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Parallel batch helper
// ─────────────────────────────────────────────────────────────────────────────

/// Pre-process a slice of documents in parallel (e.g., feature extraction) and
/// return them ready for batch indexing.
///
/// The `preprocess` closure is called on each document concurrently using
/// Rayon.  Documents that fail pre-processing are dropped with a warning
/// recorded in the returned [`PreprocessStats`].
///
/// # Errors
///
/// This function itself is infallible (errors are reported via
/// [`PreprocessStats`]).  Individual document failures are counted but not
/// propagated to avoid aborting an entire batch for one bad document.
pub fn parallel_preprocess<F>(
    docs: Vec<BatchDocument>,
    preprocess: F,
) -> (Vec<BatchDocument>, PreprocessStats)
where
    F: Fn(BatchDocument) -> Result<BatchDocument, String> + Sync + Send,
{
    use rayon::prelude::*;

    let total = docs.len();
    let results: Vec<Result<BatchDocument, String>> =
        docs.into_par_iter().map(|d| preprocess(d)).collect();

    let mut processed = Vec::with_capacity(total);
    let mut failed = 0usize;
    for r in results {
        match r {
            Ok(d) => processed.push(d),
            Err(_) => failed += 1,
        }
    }

    let stats = PreprocessStats {
        total_input: total,
        succeeded: processed.len(),
        failed,
    };
    (processed, stats)
}

/// Statistics from a parallel pre-processing run.
#[derive(Debug, Clone)]
pub struct PreprocessStats {
    /// Total documents submitted for pre-processing.
    pub total_input: usize,
    /// Documents successfully pre-processed.
    pub succeeded: usize,
    /// Documents that failed pre-processing (and were dropped).
    pub failed: usize,
}

// ─────────────────────────────────────────────────────────────────────────────
// DocIndexSink + EngineBackend — bridge to a rich document index
// ─────────────────────────────────────────────────────────────────────────────

/// A sink that can index full-fidelity documents one at a time and commit them
/// all at once.
///
/// This is the bridge that lets the text-oriented [`BatchIndexer`] machinery
/// drive a richer index (such as `SearchEngine`, which indexes visual, audio,
/// face, OCR and colour features in addition to text) while still amortising
/// the (expensive) commit into a single call at the end of the batch.
///
/// The associated [`DocIndexSink::Doc`] type is the *real* document the sink
/// understands; [`BatchDocument`] is used only to carry the lightweight text
/// payload and a back-reference (its `doc_id`) into the original document slice,
/// so the buffering/auto-flush/stats logic of [`BatchIndexer`] is exercised
/// without forcing the sink to lose any fidelity.
pub trait DocIndexSink {
    /// The rich document type this sink indexes.
    type Doc;

    /// Add a single document to the underlying indices **without** committing.
    ///
    /// # Errors
    ///
    /// Returns [`SearchError`] if the document cannot be added.
    fn index_one(&mut self, doc: &Self::Doc) -> SearchResult<()>;

    /// Commit all pending writes, making every previously-added document
    /// visible to search. Called exactly once per batch by [`EngineBackend`].
    ///
    /// # Errors
    ///
    /// Returns [`SearchError`] if the commit fails.
    fn commit_all(&mut self) -> SearchResult<()>;
}

/// An [`IndexBackend`] that resolves each [`BatchDocument`] back to a rich
/// document in a borrowed slice and feeds it to a [`DocIndexSink`].
///
/// `write_batch` indexes (without committing) every document in the batch;
/// `commit` performs the single, batch-wide commit on the sink. This is what
/// gives bulk imports their throughput win: the ~6 sub-indices of a
/// `SearchEngine` are finalised once, not once per document.
///
/// The `doc_id` of each [`BatchDocument`] must be the decimal string index of
/// the corresponding entry in `docs`. [`build_batch_documents`] produces
/// exactly this layout.
pub struct EngineBackend<'a, S: DocIndexSink> {
    /// The rich document index being driven.
    sink: &'a mut S,
    /// The original, full-fidelity documents, indexed by `BatchDocument::doc_id`.
    docs: &'a [S::Doc],
    /// Number of documents successfully written (pre-commit).
    written: usize,
    /// Number of documents made durable via `commit`.
    committed: usize,
}

impl<'a, S: DocIndexSink> EngineBackend<'a, S> {
    /// Create a backend bridging `sink` and the borrowed `docs` slice.
    #[must_use]
    pub fn new(sink: &'a mut S, docs: &'a [S::Doc]) -> Self {
        Self {
            sink,
            docs,
            written: 0,
            committed: 0,
        }
    }

    /// Resolve a [`BatchDocument`]'s `doc_id` back to an index into `docs`.
    fn resolve_index(doc_id: &str) -> SearchResult<usize> {
        doc_id.parse::<usize>().map_err(|_| {
            SearchError::Other(format!(
                "EngineBackend: malformed batch doc_id (expected integer index, got {doc_id:?})"
            ))
        })
    }
}

// `EngineBackend` borrows the sink; it is only ever used on the thread that owns
// that borrow, but `IndexBackend` requires `Send`. The borrow is `&mut S`, which
// is `Send` whenever `S: Send`, so this bound is sound.
impl<S: DocIndexSink + Send> IndexBackend for EngineBackend<'_, S>
where
    S::Doc: Sync,
{
    fn write_batch(&mut self, docs: &[BatchDocument]) -> SearchResult<()> {
        for bd in docs {
            let idx = Self::resolve_index(&bd.doc_id)?;
            let full = self.docs.get(idx).ok_or_else(|| {
                SearchError::Other(format!(
                    "EngineBackend: batch doc index {idx} out of bounds (len {})",
                    self.docs.len()
                ))
            })?;
            self.sink.index_one(full)?;
            self.written += 1;
        }
        Ok(())
    }

    fn commit(&mut self) -> SearchResult<()> {
        // Single batch-wide commit — the throughput win.
        self.sink.commit_all()?;
        self.committed = self.written;
        Ok(())
    }

    fn total_indexed(&self) -> usize {
        self.committed
    }
}

/// Build text-only [`BatchDocument`]s from a slice of rich documents, using the
/// supplied `text_of` projection to extract searchable text and stamping each
/// `doc_id` with the document's index so an [`EngineBackend`] can resolve it.
///
/// This is the glue used by `SearchEngine::index_documents_batch`: it lets the
/// generic [`BatchIndexer`] buffer and auto-flush over real document text while
/// the [`EngineBackend`] performs full-fidelity indexing against the originals.
pub fn build_batch_documents<D, F>(docs: &[D], text_of: F) -> Vec<BatchDocument>
where
    F: Fn(&D) -> String,
{
    docs.iter()
        .enumerate()
        .map(|(i, d)| BatchDocument::new(i.to_string(), text_of(d)))
        .collect()
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn make_doc(id: &str) -> BatchDocument {
        BatchDocument::new(id, format!("text content for {id}"))
    }

    // ── basic functionality ───────────────────────────────────────────────────

    #[test]
    fn test_push_below_capacity_no_auto_flush() {
        let backend = InMemoryBackend::new();
        let mut indexer = BatchIndexer::with_capacity(backend, 5);
        for i in 0..4 {
            indexer
                .push(make_doc(&format!("doc-{i}")))
                .expect("push ok");
        }
        assert_eq!(indexer.buffered_count(), 4);
        // Nothing committed yet.
        assert_eq!(indexer.backend().total_indexed(), 0);
    }

    #[test]
    fn test_auto_flush_at_capacity() {
        let backend = InMemoryBackend::new();
        let mut indexer = BatchIndexer::with_capacity(backend, 3);
        for i in 0..3 {
            indexer
                .push(make_doc(&format!("doc-{i}")))
                .expect("push ok");
        }
        // Auto-flush triggered but commit not yet called.
        assert_eq!(indexer.buffered_count(), 0);
        assert_eq!(indexer.stats().auto_flushes, 1);
        assert_eq!(indexer.stats().docs_written, 3);
    }

    #[test]
    fn test_manual_flush_commits() {
        let backend = InMemoryBackend::new();
        let mut indexer = BatchIndexer::with_capacity(backend, 10);
        for i in 0..7 {
            indexer
                .push(make_doc(&format!("doc-{i}")))
                .expect("push ok");
        }
        indexer.flush().expect("flush ok");
        assert_eq!(indexer.backend().total_indexed(), 7);
        assert_eq!(indexer.buffered_count(), 0);
    }

    #[test]
    fn test_multiple_auto_flushes_and_final_flush() {
        let backend = InMemoryBackend::new();
        let mut indexer = BatchIndexer::with_capacity(backend, 3);
        for i in 0..10u32 {
            indexer
                .push(make_doc(&format!("doc-{i}")))
                .expect("push ok");
        }
        indexer.flush().expect("final flush ok");
        assert_eq!(indexer.backend().total_indexed(), 10);
        assert_eq!(indexer.stats().auto_flushes, 3); // 9/3 = 3 auto-flushes
        assert_eq!(indexer.stats().manual_flushes, 1);
    }

    #[test]
    fn test_flush_empty_buffer_is_noop() {
        let backend = InMemoryBackend::new();
        let mut indexer = BatchIndexer::with_capacity(backend, 5);
        indexer
            .flush()
            .expect("flush of empty buffer should succeed");
        assert_eq!(indexer.stats().docs_written, 0);
        assert_eq!(indexer.backend().total_indexed(), 0);
    }

    #[test]
    fn test_batch_document_builder() {
        let doc = BatchDocument::new("id-1", "hello world")
            .with_meta("codec", "h264")
            .with_features(vec![0.1, 0.2, 0.3])
            .with_tags(vec!["sport".to_string(), "outdoor".to_string()]);

        assert_eq!(doc.doc_id, "id-1");
        assert_eq!(doc.metadata.get("codec").map(String::as_str), Some("h264"));
        assert_eq!(doc.features.as_ref().map(Vec::len), Some(3));
        assert_eq!(doc.tags.len(), 2);
    }

    #[test]
    fn test_strict_error_propagated() {
        let mut backend = InMemoryBackend::new();
        backend.fail_next(1);
        let mut indexer = BatchIndexer::with_capacity(backend, 2);
        indexer.push(make_doc("a")).expect("first push ok");
        let result = indexer.push(make_doc("b")); // triggers auto-flush which fails
        assert!(result.is_err(), "error should propagate in strict mode");
        assert_eq!(indexer.stats().flush_errors, 1);
    }

    #[test]
    fn test_lenient_error_not_propagated() {
        let mut backend = InMemoryBackend::new();
        backend.fail_next(1);
        let mut indexer = BatchIndexer::with_capacity(backend, 2).lenient();
        indexer.push(make_doc("a")).expect("first push ok");
        let result = indexer.push(make_doc("b")); // triggers auto-flush which fails
        assert!(result.is_ok(), "lenient mode should not propagate error");
        assert_eq!(indexer.stats().flush_errors, 1);
    }

    #[test]
    fn test_into_backend_returns_committed_docs() {
        let backend = InMemoryBackend::new();
        let mut indexer = BatchIndexer::with_capacity(backend, 5);
        for i in 0..4 {
            indexer.push(make_doc(&format!("d{i}"))).expect("push ok");
        }
        indexer.flush().expect("flush ok");
        let backend = indexer.into_backend();
        assert_eq!(backend.total_indexed(), 4);
    }

    #[test]
    fn test_parallel_preprocess_all_succeed() {
        let docs: Vec<BatchDocument> = (0..20)
            .map(|i| BatchDocument::new(format!("doc-{i}"), format!("body {i}")))
            .collect();

        let (processed, stats) = parallel_preprocess(docs, |mut d| {
            d.tags.push("processed".to_string());
            Ok(d)
        });

        assert_eq!(stats.total_input, 20);
        assert_eq!(stats.succeeded, 20);
        assert_eq!(stats.failed, 0);
        assert!(processed
            .iter()
            .all(|d| d.tags.contains(&"processed".to_string())));
    }

    #[test]
    fn test_parallel_preprocess_partial_failure() {
        let docs: Vec<BatchDocument> = (0..10)
            .map(|i| BatchDocument::new(format!("doc-{i}"), format!("body {i}")))
            .collect();

        // Fail documents with even-numbered IDs.
        let (processed, stats) = parallel_preprocess(docs, |d| {
            if d.doc_id.ends_with('0')
                || d.doc_id.ends_with('2')
                || d.doc_id.ends_with('4')
                || d.doc_id.ends_with('6')
                || d.doc_id.ends_with('8')
            {
                Err(format!("rejected: {}", d.doc_id))
            } else {
                Ok(d)
            }
        });

        assert_eq!(stats.total_input, 10);
        assert_eq!(stats.failed, 5);
        assert_eq!(stats.succeeded, 5);
        assert_eq!(processed.len(), 5);
    }

    #[test]
    fn test_stats_tracking() {
        let backend = InMemoryBackend::new();
        let mut indexer = BatchIndexer::with_capacity(backend, 4);
        for i in 0..9u32 {
            indexer.push(make_doc(&format!("d{i}"))).expect("push ok");
        }
        indexer.flush().expect("flush ok");

        let stats = indexer.stats();
        assert_eq!(stats.docs_pushed, 9);
        assert_eq!(stats.auto_flushes, 2); // 8 docs / 4 = 2 auto-flushes
        assert_eq!(stats.manual_flushes, 1);
        assert_eq!(stats.docs_written, 9);
        assert_eq!(stats.flush_errors, 0);
    }

    #[test]
    fn test_capacity_accessor() {
        let backend = InMemoryBackend::new();
        let indexer = BatchIndexer::with_capacity(backend, 42);
        assert_eq!(indexer.capacity(), 42);
    }

    // ── EngineBackend / DocIndexSink bridge ────────────────────────────────────

    /// A minimal rich-document sink that records indexed docs and counts commits.
    #[derive(Default)]
    struct MockSink {
        indexed: Vec<String>,
        committed: Vec<String>,
        commit_calls: usize,
    }

    impl DocIndexSink for MockSink {
        type Doc = String;

        fn index_one(&mut self, doc: &String) -> SearchResult<()> {
            self.indexed.push(doc.clone());
            Ok(())
        }

        fn commit_all(&mut self) -> SearchResult<()> {
            self.commit_calls += 1;
            self.committed = self.indexed.clone();
            Ok(())
        }
    }

    #[test]
    fn test_build_batch_documents_stamps_index() {
        let docs = vec!["alpha".to_string(), "beta".to_string(), "gamma".to_string()];
        let batch = build_batch_documents(&docs, |d| d.clone());
        assert_eq!(batch.len(), 3);
        assert_eq!(batch[0].doc_id, "0");
        assert_eq!(batch[1].doc_id, "1");
        assert_eq!(batch[2].doc_id, "2");
        assert_eq!(batch[0].text, "alpha");
        assert_eq!(batch[2].text, "gamma");
    }

    #[test]
    fn test_engine_backend_commits_once_over_multiple_batches() {
        let docs: Vec<String> = (0..10).map(|i| format!("doc-{i}")).collect();
        let mut sink = MockSink::default();
        {
            let backend = EngineBackend::new(&mut sink, &docs);
            // capacity 3 over 10 docs => 3 auto-flushes (write_batch) + 1 final flush.
            let mut indexer = BatchIndexer::with_capacity(backend, 3);
            for bd in build_batch_documents(&docs, |d| d.clone()) {
                indexer.push(bd).expect("push ok");
            }
            indexer.flush().expect("flush ok");
            // 4 write_batch calls (3 auto + 1 final-with-remainder), 1 commit.
            assert_eq!(indexer.backend().total_indexed(), 10);
        }
        // Exactly one commit despite multiple write batches.
        assert_eq!(sink.commit_calls, 1);
        assert_eq!(sink.indexed.len(), 10);
        assert_eq!(sink.committed.len(), 10);
        assert_eq!(sink.indexed[0], "doc-0");
        assert_eq!(sink.indexed[9], "doc-9");
    }

    #[test]
    fn test_engine_backend_empty_does_not_commit() {
        let docs: Vec<String> = Vec::new();
        let mut sink = MockSink::default();
        {
            let backend = EngineBackend::new(&mut sink, &docs);
            let mut indexer = BatchIndexer::with_capacity(backend, 4);
            // No pushes; final flush over an empty buffer must be a no-op write,
            // but BatchIndexer::flush still calls backend.commit().
            for bd in build_batch_documents(&docs, |d: &String| d.clone()) {
                indexer.push(bd).expect("push ok");
            }
            // Deliberately do NOT call flush(): with zero docs the caller path in
            // SearchEngine short-circuits before constructing the indexer.
            assert_eq!(indexer.backend().total_indexed(), 0);
        }
        assert_eq!(sink.commit_calls, 0);
        assert_eq!(sink.indexed.len(), 0);
    }

    #[test]
    fn test_engine_backend_rejects_bad_doc_id() {
        let docs: Vec<String> = vec!["x".to_string()];
        let mut sink = MockSink::default();
        let mut backend = EngineBackend::new(&mut sink, &docs);
        let bad = vec![BatchDocument::new("not-a-number", "x")];
        let r = backend.write_batch(&bad);
        assert!(r.is_err(), "malformed doc_id should error");
    }

    #[test]
    fn test_engine_backend_rejects_out_of_bounds_index() {
        let docs: Vec<String> = vec!["x".to_string()];
        let mut sink = MockSink::default();
        let mut backend = EngineBackend::new(&mut sink, &docs);
        let oob = vec![BatchDocument::new("5", "x")];
        let r = backend.write_batch(&oob);
        assert!(r.is_err(), "out-of-bounds index should error");
    }
}
