# oximedia-search TODO

## Current Status
- 42 modules (16 subdirectory modules) covering full-text search, visual similarity (perceptual hashing), audio fingerprinting, faceted search, boolean queries, range queries, reverse search, color search, face search, OCR search, relevance scoring, query parsing, search analytics, clustering, federation, filtering, pipelines, suggestions, spell correction, semantic search, temporal search
- Core types: SearchEngine (feature-gated behind `search-engine`), SearchQuery, SearchFilters, SearchResults, SearchResultItem, SortOptions
- Feature gates: `search-engine` (tantivy dependency)
- Dependencies: oximedia-core, oximedia-cv, oximedia-scene, tantivy (optional), rayon, serde, uuid, chrono (ndarray removed)

## Enhancements
- [x] Remove `unwrap_or` calls in `sort_results` and replace with proper `Ordering` handling — now uses `f32::total_cmp` (NaN-safe, no unwrap needed)
- [x] Add `modified_at` field to `SearchResultItem` instead of using `created_at` as proxy for ModifiedAt sorting
- [x] Add `file_size` field to `SearchResultItem` to enable FileSize sorting
- [x] Extend `search_filter` with codec-specific filters (bit depth, sample rate, frame rate, color space) — `CodecFilters` struct in lib.rs
- [x] Implement `search_suggest` auto-complete with frequency-weighted suggestions from indexed terms — `search_suggest.rs` with `TermTrie` + `SearchSuggestor`
- [x] Add `search_rewrite` query expansion using synonyms (e.g., "audio" -> "audio OR sound OR music") — `search_rewrite.rs` with `SynonymDictionary::media_defaults()`
- [x] Extend `facet::aggregation` with hierarchical facets (e.g., format -> codec -> profile) (verified 2026-05-16; src/facet/aggregation.rs:324 HierarchicalFacet)
- [x] Implement `search_analytics` query logging with click-through tracking for relevance tuning

## New Features
- [x] Add `embedding_search` module for vector similarity search using learned embeddings (CLIP-like, patent-free)
- [x] Implement `transcript_search` module indexing speech-to-text transcripts with timestamp alignment — TF-IDF + phrase search + millisecond timestamps
- [x] Add `geo_search` module for location-based media search using GPS metadata — Haversine radius/bbox/KNN
- [x] Implement `duplicate_detection` module combining visual hash + audio fingerprint for finding duplicates — Hamming distance fusion with configurable weights
- [x] Add `scene_search` integration with oximedia-scene for searching by detected objects, scenes, or activities (verified 2026-05-16; src/scene_search.rs:464 lines)
- [x] Implement `search_export` for exporting search results as CSV, JSON, or XML with custom field selection
- [x] Add `saved_search` module for persisting and re-executing named search queries
- [x] Implement `search_ab_test` for A/B testing different ranking algorithms with metrics collection
- [x] Add `related_content` module for "more like this" recommendations from a seed result

## Performance
- [x] Remove ndarray dependency (was unused in source; removed from Cargo.toml) — COOLJAPAN policy compliant
- [x] Add index sharding — `search_shard.rs` with `ShardedIndex` (FNV-1a hash assignment, parallel Bloom pre-check + rayon-ready)
- [x] Implement Bloom filter pre-check in shards — counting Bloom filter with 3 FNV hash functions supports add/remove
- [x] Add result caching in `search_pipeline` with TTL-based invalidation on index updates
- [x] Optimize `visual::index::VisualIndex` with VP-tree or ball-tree for sub-linear similarity search — VisualIndex now routes through FloatVpTree for N >= 8
- [x] Add batch indexing in `SearchEngine::index_document` for bulk import throughput — `SearchEngine::index_documents_batch` (src/lib.rs:530) routes docs through `batch_index::BatchIndexer` + `EngineBackend` adapter (src/batch_index.rs:466) so the six sub-indices commit ONCE per batch instead of per document

## Testing
- [x] Add precision/recall benchmarks for `text::search` with standard IR test collections — Wave 30, 2026-06-08: new reusable `eval.rs` (set-based `precision_at_k`/`recall_at_k`/`average_precision`/`mean_average_precision`, generic over `Id: Eq+Hash`, re-exported from lib.rs) + golden `tests/ir_eval.rs` (hand-computed AP/P@k/R@k/MAP known answers to 1e-9 + 20-doc 4-cluster in-test corpus run through the real `SearchEngine` text search computing P@10/R@20/MAP against calibrated thresholds; note: Tantivy returns only matching docs so cluster P@10 clamps to 1.0)
- [x] Test `bool_query` with complex nested AND/OR/NOT expressions — Wave 30, 2026-06-08: `tests/bool_query_nested.rs`, 13 cases over a 10-doc term-set corpus with exact match counts ((A OR B) AND (C OR D), …NOT E, all-NOT NOT A AND NOT B, phrase-AND, 3-level nesting, double-negation, OR-of-NOTs, and empty-result queries) via the real `BoolQuery::matches` evaluator
- [ ] Add tests for `facet::aggregation` with multi-value faceted fields
- [ ] Test `search_federation` merging results from multiple remote indices
- [ ] Add tests for `SearchEngine` full lifecycle: create index, add documents, search, delete, re-search

## Documentation
- [ ] Document the query language syntax supported by `query_parser` with examples
- [ ] Add architecture diagram showing the search pipeline: parse -> rewrite -> execute -> rank -> filter -> paginate
- [ ] Document the visual search algorithm (perceptual hashing variant) and matching threshold selection
- [ ] Add guide for tuning relevance scoring weights in `ranking` module
