# oximedia-dedup TODO

## Current Status
- 34 source files; media deduplication and duplicate detection
- Hashing: BLAKE3 cryptographic hashing for exact duplicates, rolling hash, frame hash, perceptual hash (pHash)
- Visual similarity: SSIM, histogram comparison, feature vector cosine similarity
- Audio: audio fingerprinting with bit-level Hamming distance comparison
- Metadata: weighted metadata comparison (duration, resolution, codec, container)
- Storage: SQLite-based indexing (feature-gated), bloom filter, LSH index, hash store, similarity index
- Detection strategies: ExactHash, PerceptualHash, SSIM, Histogram, FeatureMatch, AudioFingerprint, Metadata, All, VisualAll, Fast
- Modules: hash, visual, audio, metadata, database, report, bloom_filter, cluster, content_id, content_signature, dedup_cache, dedup_index, dedup_policy, dedup_report, dedup_stats, frame_hash, fuzzy_match, lsh_index, merge_strategy, near_duplicate, perceptual_hash, phash, rolling_hash, segment_dedup, similarity_index, video_dedup

## Enhancements
- [x] Replace O(n^2) pairwise comparison in `find_perceptual_duplicates()` with LSH-based approximate nearest neighbor via `lsh_index.rs`
- [x] Extend `visual.rs` with dHash and wHash alongside pHash for robustness against different transformations (verified 2026-05-16; src/visual.rs:180 compute_dhash, compute_wHash wavelet:632)
- [x] Add configurable thumbnail resolution in `find_ssim_duplicates()` (implemented 2026-05-15; SsimConfig struct + find_ssim_duplicates_with_config() in visual.rs, 5 tests)
- [x] Improve `metadata.rs` comparison with normalized title/filename matching using `fuzzy_match.rs` (implemented 2026-05-15; title field in MediaMetadata, title_fuzzy_score in MetadataSimilarity, compare_titles() function, 6 tests)
- [x] Extend `dedup_policy.rs` with configurable actions per duplicate group (keep newest, keep highest quality, prompt user) (implemented 2026-05-15; GroupAction enum + select_keeper() function, 7 tests)
- [x] Add progress reporting callbacks to `DuplicateDetector::find_duplicates()` for large libraries
- [x] Improve `audio.rs` fingerprinting with chromagram-based features for better music matching (verified 2026-05-16; src/chromagram.rs:550 lines)
- [x] Extend `dedup_report.rs` with disk space savings estimation per duplicate group (implemented 2026-05-15; GroupSavings::potential_savings_bytes() + SavingsSummary::total_potential_savings_bytes(), 4 tests)

## New Features
- [x] Implement incremental deduplication in `dedup_index.rs` (only scan new/modified files)
- [x] Add video segment deduplication in `segment_dedup.rs` (detect shared clips within different videos) (implemented 2026-05-15; SharedClipMatch, find_shared_clips/find_shared_clips_with_hashes with BTreeMap O(log n) index, extension loop, confidence scoring; 2 new tests)
- [x] Implement cross-format duplicate detection (same content in different containers/codecs)
- [x] Add `merge_strategy.rs` implementation for automatic duplicate resolution (symlinks, hardlinks, deletion) (implemented 2026-05-16)
  - **Implemented:** `MergeExecutor` with `apply()` / `dry_run()` / `apply_resolution()` for real FS mutations. `AppliedAction { Symlinked, Hardlinked, Deleted, Kept, Skipped }` + `MergeReport`. Unix symlinks via `std::os::unix::fs::symlink`; hard links via `std::fs::hard_link` with EXDEV-safe `Skipped` fallback. Never silently destroys — primary existence guard on Delete; self-dedup guard via `same_path()`.
  - **Tests:** 9 tests in `executor_tests` module — symlink/hardlink/delete/dry_run/cross-fs/self-dedup/modified_count/apply_resolution — all pass (647 total). Zero clippy warnings.
- [x] Implement hierarchical deduplication: fast pass (hash) -> medium pass (perceptual) -> slow pass (SSIM) (verified 2026-05-16; src/hierarchical.rs:744 lines)
- [x] Add `cluster.rs` transitive closure grouping (if A~B and B~C, group {A,B,C} together) (verified 2026-05-16; src/cluster.rs:253 transitive_closure_groups fn, test:603)
- [x] Implement network-aware deduplication for distributed media libraries (verified 2026-05-16; src/network_dedup.rs:697 lines)
- [x] Add `content_signature.rs` robust signature that survives transcoding, cropping, and watermarking

## Performance
- [x] Replace `rustfft` with OxiFFT per COOLJAPAN policy for audio fingerprinting FFT
- [x] Replace `ndarray` with native implementations per COOLJAPAN policy (implemented 2026-05-15; dedup_index.rs confirmed ndarray-free; visual.rs already used flat Vec<f64> row-major layout; no ndarray dep exists)
- [x] Parallelize `add_files()` with rayon for bulk indexing (currently sequential) (implemented 2026-05-15; add_files() in dedup_index.rs uses rayon par_iter for feature extraction, sequential insertion; FileFeatures + compute_file_features(); 2 tests)
- [x] Implement batch database insertions in `database.rs` for faster indexing (implemented 2026-05-15; BatchFileEntry struct + insert_batch() async fn wrapping all inserts in a single SQLite transaction; 2 tests: test_batch_insert_100_entries, test_batch_insert_vs_individual_same_results)
- [x] Add bloom filter pre-screening in `bloom_filter.rs` before expensive pairwise comparisons (verified 2026-05-16; src/bloom_prescreen.rs:408 lines)
- [x] Optimize `rolling_hash.rs` for streaming duplicate detection without loading entire files (implemented 2026-05-15; RollingHashStream<R: Read> Rabin-fingerprint iterator with 64 KiB ring buffer, precomputed pow_table, rolling_hash_slice reference impl; 2 tests)
- [x] Cache decoded thumbnails and fingerprints in `dedup_cache.rs` across sessions (implemented 2026-05-15; DedupSessionCache with FNV-1a keyed entries, mtime invalidation, LRU eviction, save/load JSON; 4 tests)

## Testing
- [x] Add end-to-end dedup test: index files -> find duplicates -> verify correct grouping (implemented 2026-05-15; tests/it_dedup_e2e.rs: 3 identical + 2 unique files, ExactHash strategy, group membership asserted)
- [x] Test `bloom_filter.rs` false positive rate at different fill levels (implemented 2026-05-15; tests/it_bloom_filter.rs: 50%/75%/90% fill, FPR ≤ 2×/5× configured threshold, no-false-negatives, clear() reset)
- [x] Test `lsh_index.rs` recall accuracy for near-duplicate queries (implemented 2026-05-15; tests/it_lsh_index.rs: 100 hashes, ≥70% recall at hamming≤3, far-duplicate rejection, end-to-end find_near_duplicates)
- [ ] Add benchmark for 10K+ file library deduplication throughput
- [x] Test `perceptual_hash.rs` robustness against common transformations (resize, crop, brightness) (implemented 2026-05-15; tests/it_phash_robustness.rs: 64×64 gradient, resize/brightness±10/centre-crop, hamming≤15; identical=0, inverted≥20)
- [x] Test `audio.rs` fingerprint matching across different encodings of the same audio (implemented 2026-05-15; tests/it_audio_fingerprint.rs: 440Hz sine vs noisy≥0.75, vs white-noise≤0.70, identical=1.0)
- [x] Add integration test with `sqlite` feature enabled for full `DuplicateDetector` workflow (implemented 2026-05-15; tests/it_sqlite_integration.rs: full workflow + batch API + incremental add, 3 tests)

## Documentation
- [x] Document detection strategy selection guide (when to use Fast vs All vs specific strategies) (implemented 2026-05-15; Strategy Selection Guide table in src/lib.rs module docs)
- [x] Add accuracy/performance trade-off analysis for each detection method (implemented 2026-05-15; Detection Method Trade-offs table in src/lib.rs module docs)
- [x] Document database schema and migration strategy for `database.rs` (implemented 2026-05-15; full schema docs in module-level //! with tables, columns, indices, migration notes)
