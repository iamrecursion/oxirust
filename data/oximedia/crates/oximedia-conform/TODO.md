# oximedia-conform TODO

## Current Status
- 60 source files; professional media conforming system
- Import formats: EDL (CMX 3600/3400), XML (FCP/Premiere/DaVinci), AAF (Avid)
- Matching strategies: filename, timecode, content hash, duration
- Features: SQLite media catalog, QC validation, timeline reconstruction (multi-track), batch processing, export (MP4, MKV, EDL, XML, AAF, frame sequences)
- Modules: importers (edl, xml, aaf), matching (filename, timecode, content, strategies), exporters (report, project, sequence), media (catalog, scanner, fingerprint), timeline (clip, track, transition), qc (checker, validator), session, batch, database, reconstruction, etc.

## Enhancements
- [x] Extend `importers/edl.rs` with support for CMX 340 and File128 EDL variants (implemented 2026-06-01)
  - **Goal:** Parse CMX 340 (40-char narrow format) and File128 variants alongside the existing CMX 3600/3400 paths; dispatch on header detection.
  - **Design:** Column-offset table per format variant (event#, reel name, transition, src in/out/dur, rec in/out/dur differ in width between variants); `detect_edl_variant(header_line) -> EdlVariant`; `parse_event_line(line, variant)` using per-variant column offsets. Round-trip test with a known fixture per variant.
  - **Files:** `src/importers/edl.rs`, `TODO.md` L11
  - **Tests:** CMX 340 parse round-trips a known fixture; File128 parse fixture.
  - **Risk:** Column offsets are subtle — derive from format specs, not guesses; add a fixture-driven test.
- [x] Add OTIO (OpenTimelineIO) import/export support in `importers/` and `exporters/` (verified 2026-05-16; src/importers/otio.rs:656 lines; src/exporters/otio.rs:428 OTIO JSON exporter)
- [x] Improve `matching/timecode.rs` with sub-frame accuracy matching for high frame rate content (verified 2026-05-16; src/matching/timecode.rs:11 SubFrameTimecode, sub_frame:20, 80 sub-frames per frame)
- [x] Extend `matching/content.rs` with perceptual hash-based fuzzy matching for re-encoded sources
- [x] Add confidence scoring to `matching/strategies.rs` with weighted multi-strategy combination
- [x] Improve `media_relink.rs` with recursive directory search and fuzzy path matching (verified 2026-05-16; src/media_relink.rs:126 RelinkCandidate, RelinkStrategy, confidence:141, recursive search support)
- [x] Extend `qc/validator.rs` with codec-specific validation rules (AV1 levels, Opus bitrate ranges) (implemented 2026-05-31: CodecValidator::validate_av1 — Av1Level enum with max_picture_size/max_luma_sample_rate/max_bitrate_main_tier_bps per Annex A; profile/chroma/bitdepth rules; CodecValidator::validate_opus — RFC 6716/7845: channel count, 6–510 kbps, native sample rates, frame durations, complexity, per-mode advisories; 19 tests)
- [x] Add `conform_diff.rs` comparison between two conform sessions for change tracking (verified 2026-05-16; src/conform_diff.rs 596 lines)

## New Features
- [x] Implement watch folder mode in `session.rs` for automatic re-conform on source changes
- [x] Add partial conform (selected clips only) support in `batch.rs` (implemented `partial_conform.rs`: `PartialConformSelector` enum with ByEventNumbers/ByReelNames/ByClipIds/All2/Any2; `PartialBatchProcessor` with filter/select/partition/build_plan; `PartialConformPlan` + `PartialConformStats` 2026-05-31)
- [x] Implement proxy/offline-to-online conform workflow with resolution scaling (implemented `proxy_conform.rs`: `ProxyResolution` Half/Quarter/Eighth/Custom; `ProxyRelinkStrategy` DirectorySwap/SuffixReplace/DirectoryComponent/Identity; `ProxyConformTranslator` with frame-rate scaling; `ProxyConformReport`; 15 tests 2026-05-31)
- [x] Add color space conforming rules in `format_conform.rs` (ensure consistent color space across clips) (verified 2026-05-16; src/format_conform.rs 435 lines — color space conforming rules)
- [x] Implement audio loudness normalization during conform in `loudness_conform.rs` (EBU R128) (verified 2026-05-16; src/loudness_conform.rs 464 lines)
- [x] Add `delivery_map.rs` deliverable generation from a single conform session (multiple output specs) (verified 2026-05-16; src/delivery_map.rs 336 lines)
- [x] Implement frame rate conversion during conform in `frame_rate_convert.rs` with pulldown detection (verified 2026-05-16; src/frame_rate_convert.rs 345 lines)
- [x] Add `test_card.rs` offline placeholder generation for missing source media (verified 2026-05-16; src/test_card.rs 363 lines)

## Performance
- [x] Parallelize `media/scanner.rs` directory scanning using rayon
- [x] Add incremental database updates in `database.rs` (skip unchanged files on re-scan) (implemented 2026-06-01)
  - **Goal:** On re-scan, skip re-ingesting files whose (path, size, mtime) triple is unchanged from the catalog; only re-ingest changed/new rows.
  - **Design:** Add `file_mtime` and `file_size` columns to the catalog table (migration); on scan, load existing (path, size, mtime) set; skip files whose triple matches; remove rows for deleted files.
  - **Files:** `src/database.rs`, `TODO.md` L32
  - **Tests:** incremental scan skips unchanged file (no re-ingest on second pass); re-ingests on mtime bump.
  - **Risk:** mtime resolution varies across filesystems — use u64 Unix seconds; accept false re-ingests on same-second changes.
- [x] Cache fingerprint computation results in `media/fingerprint.rs` with file modification time checks (implemented 2026-06-01)
  - **Goal:** Memoize fingerprint by (path, mtime) so unchanged files skip recomputation on repeated calls; recompute when mtime advances.
  - **Design:** `FingerprintCache { map: HashMap<(PathBuf, u64), FingerprintResult> }`; `get_or_compute(path, mtime, compute_fn)` → cache hit returns stored result; cache miss calls `compute_fn`, stores, and returns result.
  - **Files:** `src/media/fingerprint.rs`, `TODO.md` L33
  - **Tests:** cache hit returns identical fingerprint + skips recompute (verify via call counter); miss on mtime change triggers recompute.
  - **Risk:** cache must be bounded — add an LRU eviction or size cap to prevent unbounded growth on large media libraries.
- [x] Optimize `matching/` strategies to use bloom filters for initial candidate filtering (implemented 2026-06-01)
  - **Goal:** Build a Bloom filter over catalog fingerprint/key features; probe it before the expensive per-strategy match to drop obvious non-candidates, cutting the O(sources×catalog) comparison set.
  - **Design:** `CandidateBloom { bits: Vec<u64>, k: usize, n_expected: usize }` (pure-Rust bitset, k hashes via DefaultHasher with distinct seeds); `insert(feature_key)` at catalog-load time; `might_contain(feature_key)` as pre-filter gate; size from expected n + target FPR (≤1%). Pre-filter is a gate only — every surviving candidate still runs the full matcher (no false negatives allowed).
  - **Files:** `src/matching/bloom.rs` (new), `src/matching/mod.rs`, `src/matching/strategies.rs`, `TODO.md` L34
  - **Tests:** Bloom never produces false negatives on known-present set; Bloom pre-filter + full match == full-match-only (correctness invariant); FPR ≤ expected bound on a sized filter.
  - **Risk:** Bloom must be a pre-filter only — false positives waste time (acceptable), false negatives break correctness. The correctness-invariant test is load-bearing.
- [x] Profile and optimize `reconstruction.rs` for timelines with 1000+ clips (implemented 2026-06-05: precompute sort keys once per build_video_tracks/build_audio_tracks, sort indices instead of full clone; 4 new tests: 1000-clip sort correctness, reference match, stable sort for duplicates, 1000-clip < 100ms perf bound)

## Testing
- [ ] Add end-to-end conform test with sample EDL, source media, and expected output verification
- [ ] Test `importers/xml.rs` with real FCP X, Premiere Pro, and DaVinci Resolve XML exports
- [ ] Test `importers/aaf.rs` with Avid Media Composer AAF exports
- [x] Add round-trip test: import EDL -> conform -> export EDL -> verify identical timeline
- [ ] Test `batch.rs` with 100+ clip conform jobs for throughput and correctness
- [ ] Test `timecode_conform.rs` with drop-frame and non-drop-frame timecode edge cases

## Documentation
- [ ] Document supported EDL/XML/AAF format variants and their limitations
- [ ] Add conform workflow tutorial from import to export
- [ ] Document matching strategy selection guidelines for different source material types
