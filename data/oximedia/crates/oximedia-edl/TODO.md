# oximedia-edl TODO

## Current Status
- 34 source files covering CMX 3600/3400, GVG, Sony BVE-9000 format parsing and generation
- Full timecode support: drop-frame, non-drop-frame at 24/25/30/60 fps
- Event types: Cut, Dissolve, Wipe, Key with motion effects
- Audio channel mapping and routing in `audio.rs`
- Validation, comparison, merging, filtering, optimization, and batch export
- Dependencies: oximedia-core, oximedia-timecode, nom, serde, bitflags

## Enhancements
- [x] Add ALE (Avid Log Exchange) format parsing in addition to CMX formats
- [x] Implement AAF (Advanced Authoring Format) round-trip support (verified 2026-05-16; src/aaf.rs:130 AafComposition, AafSourceClip:57, 858 lines)
- [x] Enhance `edl_compare.rs` with diff visualization showing added/removed/modified events
- [x] Add `edl_merge.rs` conflict resolution strategies (prefer source A, prefer source B, manual)
- [x] Implement `optimizer.rs` event consolidation (merge adjacent cuts from same reel)
- [x] Enhance `validator.rs` with format-specific validation rules per EDL format
- [x] Add `edl_filter.rs` filtering by reel name patterns and timecode ranges (verified 2026-05-16; src/edl_filter.rs:170 EventFilter, FilterableEvent:12, 502 lines)
- [x] Improve `timecode.rs` with 23.976 fps and 59.94 fps drop-frame timecode support (`Fps23_976`, `Fps59_94`)
- [x] Add `conform_report.rs` report generation comparing EDL against actual media files (verified 2026-05-16; src/conform_report.rs:102 ConformReport, 356 lines)
- [x] Implement `roundtrip.rs` format conversion between CMX 3600 and GVG/Sony formats (verified 2026-05-16; src/roundtrip.rs:57 RoundtripReport, RoundtripValidator:84, 416 lines)

## New Features
- [x] Add XML-based EDL format support (Final Cut Pro XML, DaVinci Resolve XML) (verified 2026-05-16; src/fcpxml.rs:17 FcpXmlSequence, 566 lines)
- [x] Implement OTIO (OpenTimelineIO) format import/export (verified 2026-05-16; src/otio.rs:36 OtioRationalTime, OtioExternalReference:156, 1321 lines)
- [x] Add EDL-to-timeline conversion producing oximedia-edit Timeline structures (verified 2026-05-16; src/roundtrip.rs:84 RoundtripValidator with format conversion)
- [x] Implement multi-camera EDL support with camera angle switching events (verified 2026-05-16; src/multicam.rs:13 CameraAngleId, 316 lines)
- [x] Add EDL visualization as ASCII timeline for debugging
- [x] Implement `batch_export.rs` parallel EDL generation via `BatchEdlExporter::export_parallel` (rayon)
- [x] Add EDL change tracking with version history (verified 2026-05-16; src/version_history.rs:32 VersionHistory, 410 lines)
- [x] Implement sub-frame timecode precision for high frame rate content (120fps+) (verified 2026-05-16; src/subframe.rs:7 SubFrameTimecode, 120fps support)

## Performance
- [x] Optimize `parser.rs` nom combinators to reduce allocations during parsing (implemented `parser_opt.rs`: `EdlHeaderScan` zero-copy pre-scan, `CompactEventRef` byte-offset event refs, `detect_frame_rate_extended` for 24/25/30/50/60fps + DF/NDF variants 2026-05-31)
- [x] Add lazy parsing mode that only parses event headers until details are accessed (implemented 2026-06-01)
  - **Goal:** Parse event number, reel name, and in/out timecodes eagerly; defer comment, motion-effect, and extended-field parsing until the caller accesses those fields.
  - **Design:** `LazyEvent { header: EventHeader, detail_raw: &str, detail: RefCell<Option<EventDetail>> }`; `EventDetail` resolved on first `detail()` call and cached; mirror the `oximedia-jobs` `LazyPayload<T>` `RefCell<Option<T>>` pattern.
  - **Files:** `src/parser.rs` (add `LazyEvent` type and lazy-parse path), `TODO.md` L35
  - **Tests:** lazy mode parses headers without touching details (verify via counter on detail parser); lazy detail resolves correctly on first access and caches (second access returns same value without re-parsing).
  - **Risk:** borrow-checker challenge with `&str` lifetime — may need `String` raw storage instead; use `String` if needed, document the tradeoff.
- [x] Cache timecode-to-frames conversions in `timecode.rs` for repeated lookups (implemented `tc_cache.rs`: `TimecodeCache` open-addressing 1024-slot hash table + `with_tc_cache` thread-local singleton 2026-05-31)
- [x] Optimize `edl_statistics.rs` to compute all statistics in a single pass (implemented 2026-06-01)
  - **Goal:** Fold all statistics (event count, per-reel counts, duration totals, transition histogram, etc.) in one iteration over events instead of multiple passes; one accumulator struct updated per event.
  - **Design:** `StatsAccumulator` struct with all counters/maps; `accumulate(event: &EdlEvent)` updates all fields in one call; `finalize() -> EdlStatistics` computes any derived fields (e.g. averages). Keep existing `EdlStatistics` public API identical.
  - **Files:** `src/edl_statistics.rs`, `TODO.md` L37
  - **Tests:** single-pass stats == prior multi-pass stats on a fixture (exact agreement on every field); single-pass visits each event exactly once (via counter).
  - **Risk:** single-pass refactor must preserve every existing statistic exactly — the agreement test is mandatory.
- [x] Use `SmallVec` for event comments to avoid heap allocation for common cases (implemented 2026-06-01)
  - **Goal:** Replace `Vec<String>` comments field with `SmallVec<[String; 2]>` so the common 0–2 comment case is stack-allocated.
  - **Design:** Add `smallvec` to workspace deps if absent (latest crates.io); change the comments field type; update all construction sites.
  - **Files:** `src/event.rs` (or wherever comments live), `Cargo.toml` (add smallvec if absent), root `Cargo.toml` workspace deps, `TODO.md` L38
  - **Tests:** SmallVec inline path for ≤2 comments (capacity stays at inline); SmallVec spills correctly for many comments (>2 items stored correctly).
  - **Risk:** `smallvec` may already be a workspace dep — check before adding.

## Testing
- [x] Add conformance tests with real-world EDL files from major NLE systems (Avid, Premiere, Resolve) — hand-authored CMX 3600 strings pin cut/dissolve/wipe/key classification, reel extraction, serialize round-trip, and the duration-column parser limitation (`tests/cmx3600_conformance.rs:25` `test_cmx_cut_then_dissolve`, `:90` `test_cmx_serialize_roundtrip`, `:122` `test_cmx_rejects_duration_column`)
- [x] Test `cmx3600.rs` parser with EDLs containing complex wipe patterns and key events — wipe + key classification via `parse_cmx`, and wipe-without-pattern rejection via the main parser (`tests/cmx3600_conformance.rs:66` `test_cmx_wipe_and_key`, `tests/parser_fuzz.rs:81` `test_wipe_without_pattern_is_rejected`)
- [x] Add fuzz testing for `parser.rs` with malformed EDL inputs — seeded xorshift PRNG feeds ~1200 random ASCII lines/blobs (never panics/hangs) plus the silent-skip and inverted-timecode contracts (`tests/parser_fuzz.rs:142` `test_parse_edl_fuzz_never_panics`, `:43` `test_parse_edl_silently_skips_malformed`, `:62` `test_parse_edl_inverted_timecode_errors`)
- [x] Test `motion.rs` speed effect parsing with reverse playback and freeze frames — reverse/freeze constructors, effective speed, validate() rejection rules, and the M2 comment round-trip (`tests/motion_timeline.rs:14` `test_motion_effect_constructors_and_validation`, `:60` `test_motion_m2_comment_roundtrip`)
- [x] Test `edl_timeline.rs` conversion accuracy with overlapping events — TimelineAnalyzer overlap + merged-coverage accounting, plus the EDL→timeline bridge clip-frame mapping (`tests/motion_timeline.rs:191` `test_timeline_analyzer_overlap_and_coverage`, `:82` `test_convert_edl_to_timeline_clip_frames`)
- [x] Add round-trip tests for all supported formats — `test_cmx3600_roundtrip` (3-event EDL), `test_ale_parse_basic`
- [ ] Test `reel_registry.rs` with duplicate reel name handling

## Documentation
- [ ] Document CMX 3600 format specification with field-by-field breakdown
- [ ] Add format comparison table (CMX 3600 vs GVG vs Sony BVE-9000 capabilities)
- [ ] Document the event numbering and timecode conventions used across formats
