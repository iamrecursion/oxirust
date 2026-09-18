# oximedia-repair TODO

## Current Status
- 35 modules (15 subdirectory modules) covering corruption detection, header repair, index rebuilding, timestamp correction, packet recovery, sync fixes, truncation recovery, metadata repair, frame reordering, and error concealment
- Core types: RepairEngine, RepairOptions, RepairMode (Safe/Balanced/Aggressive/Extract), Issue/IssueType/Severity
- Batch repair support via `repair_batch`
- Dependencies: oximedia-core, oximedia-container, oximedia-codec, thiserror

## Enhancements
- [x] Wire dormant modules (conceal, partial, codec_probe, container_migrate) into `RepairEngine::fix_issue` dispatcher (implemented 2026-05-11; lib.rs:~600)
- [x] Add progress callback to `repair_file` and `repair_batch` for UI integration (verified 2026-05-11; lib.rs:~200)
- [x] Extend `corruption_map` to visualize byte-level corruption regions with exportable heat maps (verified 2026-05-16; src/corruption_heatmap.rs:263 to_json fn, 501 lines; src/corruption_map.rs:62 CorruptionMap, 285 lines)
- [x] Add confidence scores to `detect::corruption` results indicating certainty of corruption detection (verified 2026-05-16; src/detect/corruption.rs:18 detect_corruption_with_confidence)
- [x] Implement incremental `verify::integrity` that can resume verification from a checkpoint (implemented 2026-05-15; VerifyCheckpoint/CheckpointIssue/VerifyReport types + verify_from_checkpoint/save_checkpoint/load_checkpoint in verify/integrity.rs; serde JSON serialisation via temp_dir uuid filenames)
- [x] Extend `repair_log` with structured JSON output for machine-parseable repair history (verified 2026-05-16; src/repair_log.rs:10 serde::Serialize on RepairLog entries, JSON-structured)
- [x] Add `RepairMode::Custom` variant allowing per-issue-type aggressiveness configuration (verified 2026-05-11; lib.rs:~139)
- [x] Improve `frame_concealment` with motion-compensated interpolation from adjacent good frames (verified 2026-05-11; conceal/frame.rs:~166)

## New Features
- [x] Add a `codec_probe` module that identifies codec parameters from raw bitstream when headers are damaged (implemented 2026-05-11; codec_probe.rs — wired into CorruptedHeader fallback)
- [x] Implement `container_migrate` for re-muxing repaired streams into a clean container without re-encoding (implemented 2026-05-11; container_migrate.rs — wired into CorruptedHeader fallback)
- [x] Add `repair_profile` for saving and loading repair configuration presets per container format (verified 2026-05-16; src/repair_profile.rs:209 to_json, 440 lines)
- [x] Implement `parallel_repair` using rayon for batch repair of independent files (verified 2026-05-16; src/parallel_repair.rs:499 lines)
- [x] Add `corruption_simulator` for testing: inject controlled corruption (bit-flip, truncation, header wipe) (verified 2026-05-16; src/corruption_simulator.rs:591 lines)
- [x] Implement `stream_splice` for extracting playable segments from partially corrupt files (verified 2026-05-16; src/stream_splice.rs:694 lines)
- [x] Add `repair_diff` module to generate before/after comparison reports with frame-by-frame quality metrics (verified 2026-05-16; src/repair_diff.rs:405 lines)
- [x] Implement `progressive_repair` that yields partially repaired output as each issue is fixed (verified 2026-05-16; src/progressive_repair.rs:311 lines)

## Performance
- [x] Use memory-mapped I/O in `detect::scan::deep_scan` for large files (>4 MiB) to avoid full reads (implemented 2026-05-11; detect/scan.rs — deep_scan dispatches to deep_scan_mmap/deep_scan_streaming)
- [x] Add sector-aligned reads in `bitstream_repair` for SSD-friendly I/O patterns (implemented 2026-05-15; SectorAlignedReader<R: Read+Seek> with configurable sector_size, aligned internal buffer, Read+Seek impls; 2 tests: test_sector_aligned_reader_matches_direct_read, test_sector_aligned_reader_alignment_correctness)
- [x] Cache detected issues in `RepairEngine` to avoid re-analysis when repairing the same file (implemented 2026-05-11; lib.rs — mtime-keyed detection_cache with parking_lot::RwLock)
- [x] Parallelize analyze_file sub-passes via rayon (planned 2026-05-14)
  - **Goal:** Run analyze_container_structure, analyze_timestamps, analyze_indices, analyze_metadata concurrently via rayon::join
  - **Files:** crates/oximedia-repair/src/detect/analyze.rs (analyze_file lines 78-114), Cargo.toml (rayon dep)
  - **Tests:** test_analyze_file_parallel_deterministic_order
- [x] Optimize detect_patterns from O(n²) to O(n)-amortized using memchr + bounded early exit (planned 2026-05-14)
  - **Goal:** Replace byte-by-byte j-loop with memchr SIMD skip; cap count at MAX_PATTERN_COUNT=32; fixes 177s test
  - **Files:** crates/oximedia-repair/src/detect/analyze.rs (lines 1316-1353), Cargo.toml (memchr dep)
  - **Tests:** test_detect_patterns_all_zeros_fast_path, test_detect_patterns_sparse_sync_bytes, test_detect_patterns_count_bounded

## Testing
- [x] Add tests for `repair_file` with actual corrupted test fixtures (truncated MP4, damaged WebM headers) (implemented 2026-05-15; test_repair_file_corrupted_header uses CorruptionSimulator to inject header wipe + bit-flips)
- [x] Test `repair_batch` error handling when some files in the batch are unrecoverable (implemented 2026-05-15; test_repair_batch_partial_unrecoverable verifies repair_batch_all returns one entry per input, no panics for empty file)
- [x] Add roundtrip tests: corrupt a valid file, repair it, verify playback (implemented 2026-05-15; test_roundtrip_corrupt_repair_verify injects bit-flips then asserts issue count does not increase after repair)
- [x] Test `backup` creation and skip-backup threshold logic with files of varying sizes (implemented 2026-05-15; test_backup_creation_and_skip checks .bak created when create_backup=true and skipped when file > skip_backup_threshold)
- [x] Add fuzz tests for `header` repair modules with random byte sequences (implemented 2026-05-31; src/header/repair.rs `fuzz_tests` module: 7 test groups — repair_header random sequences, mp4 corrupted ftyp/size/truncated/random, avi bad RIFF/fourcc/random/tiny, matroska corrupted EBML/zeros/valid/random, repair_magic_number boundary cases, repair_size_field LE/BE/zero/saturate, copy_with_repair output integrity; deterministic LCG PRNG; no new deps)
- [x] Regression test for deep-scan runtime budget (planned 2026-05-14)
  - **Goal:** Assert 5 MiB deep_scan completes in <30s debug / <5s release; locks in the algorithmic speedup
  - **Files:** crates/oximedia-repair/tests/perf_deep_scan_budget.rs (new file)

## Documentation
- [x] Document the detection pipeline: corruption -> analyze -> deep_scan flow (implemented 2026-05-15; ASCII diagram + stage descriptions in detect/mod.rs module-level //! doc)
- [x] Add repair strategy guide explaining when to use Safe vs Balanced vs Aggressive modes (implemented 2026-05-15; "# Repair Modes" table in lib.rs //! block listing Safe/Balanced/Aggressive/Extract/Custom with when-to-use guidance)
- [x] Document each IssueType with example symptoms and expected repair outcomes (implemented 2026-05-15; /// doc comments on each IssueType variant: symptoms + repair outcome)
- [x] Add troubleshooting guide for files that fail the TooCorrupted check (implemented 2026-05-15; "# Troubleshooting" //! section in lib.rs: what TooCorrupted means, how to lower aggressiveness, how to try ExtractMode/stream_splice)
