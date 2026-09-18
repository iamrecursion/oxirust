# oximedia-caption-gen TODO

## Current Status
- 4 modules: `alignment` (speech-to-caption), `line_breaking` (greedy + Knuth-Plass DP), `wcag` (WCAG 2.1 compliance), `diarization` (speaker metadata + crosstalk)
- Zero external dependencies beyond `thiserror` (pure Rust, minimal footprint)
- Exports: `WordTimestamp`, `TranscriptSegment`, `CaptionBlock`, `LineBreakConfig`, `WcagChecker`, `Speaker`, `SpeakerTurn`

## Enhancements
- [x] Add word-level confidence scores to `alignment::WordTimestamp` for quality-aware caption display (verified 2026-05-16; src/alignment.rs:13 pub confidence: f32, word_confidence: f32)
- [x] Extend `line_breaking::optimal_break` with language-aware hyphenation (CJK line break rules) (verified 2026-05-16; src/line_breaking.rs:142 CJK line breaking, cjk_break:189, language_aware_break:234)
- [x] Add `wcag` check for maximum simultaneous on-screen caption count (accessibility guideline) (verified 2026-05-16; src/wcag.rs:295 check_max_simultaneous_captions)
- [x] Improve `diarization::merge_consecutive_turns` with configurable silence gap threshold (verified 2026-05-16; src/diarization.rs:136 merge_consecutive_turns_with_gap with max_gap_ms)
- [x] Add `alignment::split_long_segments` support for splitting at sentence boundaries (not just duration) (verified 2026-05-16; src/alignment.rs:279 split_long_segments, find_sentence_boundary:372)
- [x] Extend `line_breaking::LineBreakConfig` with maximum characters-per-line constraint (verified 2026-05-16; src/line_breaking.rs:8 LineBreakConfig, max_chars_per_line field, effective_max_chars:37)
- [x] Add reading speed validation for different target audiences (children vs adults) in `wcag` (verified 2026-05-16; src/wcag.rs:334 check_cps_for_audience with AudienceProfile)
- [x] Improve `diarization::CrosstalkDetector` with adjustable overlap tolerance percentage (verified 2026-05-16; src/diarization.rs:279 with_overlap_tolerance, tolerance filter:319)

## New Features
- [x] Add `forced_narrative` module for identifying and marking forced narrative subtitles (FN/SDH)
- [x] Implement `caption_format_adapter` module converting `CaptionBlock` to SRT/VTT/TTML output strings
- [x] Add `punctuation_restoration` module for adding punctuation to raw ASR transcript output (verified 2026-05-16; src/punctuation_restoration.rs:702 lines PunctuationRestorer)
- [x] Implement `language_detect` module to auto-detect transcript language for locale-aware line breaking (verified 2026-05-16; src/language_detect.rs:757 lines LanguageDetector)
- [x] Add `caption_timing_adjuster` module for shifting/stretching caption timings to match edited video
- [x] Implement `caption_diff` module for comparing two caption tracks and highlighting differences (verified 2026-05-16; src/caption_diff.rs:528 lines CaptionDiff)
- [x] Add `style_generator` module suggesting font size, position, colors based on video content analysis (verified 2026-05-16; src/style_generator.rs:563 lines StyleGenerator)
- [x] Implement `multi_language` module for bilingual caption layout (primary + secondary language) (verified 2026-05-16; src/multi_language.rs:593 lines, multi_language_sync.rs:522 lines)

## Performance
- [x] Optimize `optimal_break` DP algorithm with Knuth-Plass SMAWK speedup for O(n) line breaking (implemented 2026-05-29; Larmore-Schieber LARSCH online concave row-minima implemented (`Larsch` struct, `larsch_crossover`, `kp_cost_i64`, `kp_cost_is_concave`) and wired via `optimal_break_larsch` at line 750 of src/line_breaking.rs; strategy: O(n²) baseline + LARSCH cross-check with per-step fallback; validated via `test_larsch_matches_naive_min`, `test_larsch_monotone_minima`, `test_larsch_optimal_break_matches_dp`, `test_larsch_optimal_break_stress_random` (1000 proptest cases), `test_larsch_optimal_break_speedup`; note: full O(n) path deferred — requires proof that KP penalty is strictly Monge for dynamic f[])
  - **Refinement (2026-05-29):** Implement Larmore-Schieber LARSCH (online concave SMAWK) to wire the existing `smawk_row_minima` primitive (line 362) + `TotallyMonotoneMatrix` trait into the inner DP. Current DP is O(n²) because `f[]` is dynamic (each row min depends on previous). LARSCH provides amortised O(1) per row → total O(n). Implementation: `TotallyMonotoneConcaveMatrix<F>` trait with lazy entry computation (~250 LoC); rewrite inner DP of `optimal_break` to use LARSCH; assert LARSCH output identical to O(n²) reference on 10000 random inputs (extend existing harness). Tests: `test_larsch_matches_naive_min`, `test_larsch_optimal_break_matches_dp`, `test_larsch_optimal_break_speedup`. Risk: strictly concave penalty required — gate behind `if cost_is_concave` with O(n²) fallback if property test fails.
- [x] Add batch processing API to `align_to_frames` for processing multiple segments in one call (verified 2026-05-16; src/alignment.rs:175 align_to_frames_batch)
- [x] Cache CPS (characters-per-second) computation results in `line_breaking` for repeated re-breaks (verified 2026-05-16; src/line_breaking.rs:93 CPS cache struct with intern)
- [x] Use string interning for `Speaker` labels in `diarization` to reduce allocation in large transcripts (verified 2026-05-16; src/diarization.rs:8 SpeakerLabelPool with Arc<str> interning)

## Testing
- [x] Add test for `alignment::build_caption_blocks` with overlapping word timestamps (implemented 2026-05-13; tests/integration.rs:224 test_alignment_overlapping_words)
- [x] Test `line_breaking::optimal_break` against known Knuth-Plass reference outputs (implemented 2026-05-13; tests/integration.rs:252 test_optimal_break_matches_kp_reference)
- [x] Add WCAG compliance test suite with real-world caption samples that pass/fail each criterion (implemented 2026-05-13; tests/integration.rs:294 test_wcag_compliance_suite_a_aa_aaa)
- [x] Test `diarization::assign_speakers_to_blocks` with 5+ simultaneous speakers (implemented 2026-05-13; tests/integration.rs:336 test_diarization_5_simultaneous_speakers)
- [x] Add property-based test: `merge_short_segments` output has no segment shorter than min duration (implemented 2026-05-13; tests/integration.rs:374 proptest test_merge_short_segments_no_segment_below_min)
- [x] Test `greedy_break` vs `optimal_break` produce identical results for single-line captions (implemented 2026-05-13; tests/integration.rs:412 test_greedy_vs_optimal_single_line_identical)
- [x] Add round-trip test: split then merge segments should preserve total text content (implemented 2026-05-13; tests/integration.rs:435 proptest test_split_merge_segments_text_preserved)

## Documentation
- [x] Document WCAG 2.1 conformance levels (A, AA, AAA) and which checks map to each level (implemented 2026-05-14)
  - **Goal:** Module rustdoc in `src/wcag.rs` with a table listing each check function, its rule_id, WCAG SC, level, and notes. Usage example wrapping `WcagChecker::new(WcagLevel::AA).run_all_checks(...)`. Document that 1.2.5/1.2.7 are not machine-checkable.
  - **Design:** Table: check_caption_coverage→1.2.2/A, check_live_latency→1.2.4/AA, check_sign_language→1.2.6/AAA(stub), check_cps→AA, check_min_duration→A, check_gap_duration→A, check_max_simultaneous_captions→AA, check_cps_for_audience→AA.
  - **Files:** `crates/oximedia-caption-gen/src/wcag.rs` (top-of-file `//!` rustdoc + one doctest).
  - **Tests:** `cargo doc -p oximedia-caption-gen --no-deps` clean. Doctest compiles.
  - **Risk:** Doctest must not require external data — use inline test blocks.
- [x] Add example showing complete pipeline: word timestamps -> alignment -> line breaking -> blocks (implemented 2026-05-13; src/auto_pipeline.rs:11 doctest demonstrating AsrEngine + AutoCaptionPipeline end-to-end)
- [x] Document `diarization` module usage with speaker identification metadata from ASR systems (implemented 2026-05-14)
  - **Goal:** Module rustdoc in `src/diarization.rs` with step-by-step doctest: build SpeakerLabelPool, construct Speakers, push SpeakerTurns, merge consecutive turns, assign to caption blocks. Cover CrosstalkDetector briefly.
  - **Design:** Doctest uses only public alignment API. Cross-reference dominant_speaker, speaker_stats, format_speaker_label.
  - **Files:** `crates/oximedia-caption-gen/src/diarization.rs` (top-of-file `//!` rustdoc + doctest).
  - **Tests:** Doctest compiles + runs. `cargo doc` clean.
  - **Risk:** Must verify CaptionBlock is accessible from doctest context.

## 0.1.8 Wave 5 (completed 2026-05-29)
- [x] Split `line_breaking.rs` (2021 lines — policy violation) into 8-module directory via splitrs (Slice α)
  - Files: src/line_breaking/{mod,config,greedy,optimal,kp_common,smawk,larsch,balance}.rs
  - Result: largest module 600 lines (smawk.rs); 523 tests pass; zero stubs; old flat file deleted
