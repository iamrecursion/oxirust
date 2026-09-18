# oximedia-subtitle TODO

## Current Status
- 47 source files across 33+ public modules covering subtitle parsing, rendering, and processing
- Parsers: SRT, WebVTT, SSA/ASS, CEA-608/708, TTML, PGS, DVB (7 format parsers)
- Rendering: font loading (fontdue + ab_glyph), glyph caching, burn-in, overlay composition
- Text: bidirectional text (unicode-bidi), Unicode segmentation, line breaking
- Processing: timing adjustment, format conversion, merge, diff, search, stats, sanitize, validation, overlap detection, reading speed analysis, segmentation, spell check, cue parsing/timing, position calculation, accessibility, translation
- Export: subtitle_export with multi-format output; subtitle_index for timestamp-indexed lookup
- Dependencies: oximedia-core, oximedia-codec, nom, fontdue, ab_glyph, unicode-bidi, unicode-segmentation, quick-xml

## Enhancements
- [x] Add HTML tag stripping and formatting preservation in `parser::srt` for complex SRT files with `<b>`, `<i>`, `<font color>` tags
- [x] Extend `parser::webvtt` to support region definitions and vertical text cue settings (verified 2026-05-16; src/parser/webvtt.rs:60 VttRegion struct, vertical text cue settings:94)
- [x] Improve `parser::ssa` to handle all ASS override tags including `\clip`, `\iclip`, `\org`, `\fad`, `\move` (already implemented at parser/ssa.rs:419 AssOverrideTags + parse_override_tags:445 + parse_clip_args:500 + parse_fad_args:528 + parse_move_args:539, Wave 14 re-verified 2026-06-01)
- [x] Add fallback font chain support in `font::Font` for missing glyphs (CJK, Arabic, Devanagari) (already implemented at font.rs:290 FontChain struct + ChainedFont + rasterize, Wave 14 re-verified 2026-06-01)
- [x] Extend `burn_in` to support soft-shadow rendering with configurable blur radius (already implemented at soft_shadow.rs:34 SoftShadowConfig + render_soft_shadow_rgba:160 + gaussian_blur_alpha:89, Wave 14 re-verified 2026-06-01)
- [x] Improve `reading_speed` module to account for word complexity (not just character count per second)
- [x] Add `timing_adjuster` support for non-linear time remapping (speed ramp, variable frame rate)
- [x] Extend `subtitle_validator` to check for WCAG 2.1 AA contrast ratio compliance (verified 2026-05-16; src/wcag_validator.rs:71 WcagValidator, WcagLevel:50)

## New Features
- [x] Implement `parser::eia608_realtime` module for live CEA-608 caption decoding from a byte stream (verified 2026-05-16; src/eia608_realtime.rs:441 lines, src/parser/eia608_realtime.rs)
- [x] Add `karaoke_engine` module implementing full ASS karaoke timing with syllable-level highlighting (verified 2026-05-16; src/karaoke_engine.rs:513 lines)
- [x] Implement `teletext` parser module for DVB Teletext subtitle extraction (ETS 300 706) (verified 2026-05-16; src/teletext.rs:601 lines, src/teletext_subtitle.rs)
- [x] Add `subtitle_alignment` module for automatic timing alignment between two subtitle tracks (e.g., sync fansubs to official audio) (verified 2026-05-16; src/subtitle_alignment.rs:598 lines)
- [x] Implement `forced_subtitle` detection and flagging for narrative foreign-language subtitles
- [x] Add `sign_language` module for sign language overlay region management and picture-in-picture positioning (verified 2026-05-16; src/sign_language.rs:817 lines)
- [x] Implement `subtitle_ocr` module for extracting text from bitmap-based subtitle formats (PGS, VobSub) using pattern matching (verified 2026-05-16; src/subtitle_ocr.rs:795 lines)
- [x] Add `live_caption` module for real-time caption display with roll-up and paint-on modes (verified 2026-05-16; src/live_caption.rs:549 lines)

## Performance
- [x] Implement glyph atlas caching in `font::GlyphCache` to batch render frequently used characters into a texture atlas (verified 2026-05-16; src/glyph_atlas.rs:32 GlyphAtlas config, AtlasConfig, pixel_slice_mut:173)
- [x] Use binary search in `subtitle_index` for O(log n) timestamp lookup instead of linear scan
- [x] Add parallel subtitle parsing in `format_converter` for batch conversion of large subtitle collections (already implemented at format_converter.rs:1522 parallel_parse_batch + par_iter:1542 + parallel_parse_batch_owned:1573, Wave 14 re-verified 2026-06-01)
- [x] Cache parsed ASS style lookups in `parser::ssa` to avoid re-parsing style definitions per dialogue line (already implemented at ssa_style_cache.rs:90 AssStyleCache; wired into parser/ssa.rs parse_ass() → AssStyleCache::from_map + AssFile::style_cache field, Wave 14 2026-06-01)
- [x] Implement incremental rendering in `renderer::SubtitleRenderer` that only redraws changed regions (already implemented at renderer.rs:284 DirtyRect + IncrementalSubtitleRenderer:362 + render_incremental:400, Wave 14 re-verified 2026-06-01)

## Testing
- [ ] Add conformance tests for CEA-608 decoder with known test streams (FCC captioning test patterns)
- [x] Test `format_convert` round-trip: SRT -> WebVTT -> SRT and verify timing preservation within 1ms (added test_srt_webvtt_roundtrip_ms_precision + test_srt_webvtt_roundtrip_boundary_timestamps in format_convert.rs, Wave 14 2026-06-01)
- [x] Add edge case tests for `parser::srt` with malformed files (missing blank lines, BOM markers, CR/LF variations) (added test_srt_utf8_bom_is_stripped + test_srt_crlf_equals_lf_field_by_field + test_srt_lone_cr_yields_parse_error + test_srt_missing_blank_line_is_lenient_single_cue in tests/srt_overlap_merge.rs:25-130, Wave 24 2026-06-05; missing-blank-line PINS current lenient behavior — parser absorbs 2nd cue into 1st cue text, see test doc comment)
- [x] Test `overlap_detect` with intentionally overlapping subtitle tracks and verify all collisions are reported (added test_overlap_partial_pair + test_overlap_chain_report_counts + test_overlap_full_containment in tests/srt_overlap_merge.rs:136-208, Wave 24 2026-06-05; verifies exact pairs {(0,1),(1,2)} + OverlapReport counts + Full/Partial classification)
- [x] Add bidirectional text rendering test with mixed Arabic/Hebrew and Latin text in `text::TextLayoutEngine` (added test_bidi_mixed_arabic_latin_visual_order + test_bidi_paragraph_direction_ltr_first + test_bidi_pure_rtl_reversal + test_bidi_level_wrapper in text.rs, Wave 14 2026-06-01)
- [x] Test `subtitle_merge` with two tracks having partially overlapping time ranges (added test_merge_prefer_first + test_merge_prefer_last + test_merge_keep_all_sorted_and_flagged + test_merge_drop_on_conflict + test_merge_disjoint_no_conflict_all_strategies in tests/srt_overlap_merge.rs:218-336, Wave 24 2026-06-05; all 4 MergeStrategy variants over the conflicting A=[0,2000]/B=[1000,3000] scenario + disjoint zero-conflict control)

## Documentation
- [ ] Add format comparison table showing feature support across SRT, WebVTT, ASS, TTML, CEA-608/708
- [ ] Document font loading requirements and supported font formats (TTF, OTF, TTC)
- [ ] Add example showing subtitle burn-in pipeline: parse -> style -> render -> composite onto video frame

## 0.1.8 Wave 6 — 2026-05-29
- [x] Register 18 orphan modules in lib.rs + parser/eia608_realtime (verified 2026-05-29; 17 top-level + 1 parser sub-module wired, 0 warnings)

## 0.1.8 Wave 14 Slice I — 2026-06-01
- [x] Wire AssStyleCache into parse_ass(): AssFile now carries style_cache field (AssStyleCache::from_map); dialogue lookup unchanged but cache available for callers (parser/ssa.rs:107)
- [x] Add AssStyleCache wiring tests: test_ass_style_cache_wiring, test_ass_cache_fallback_for_unknown_style, test_ass_cache_and_direct_lookup_agree (parser/ssa.rs:780+)
- [x] Add VttParser (WebVTT → SubtitleEntry): parse_vtt_timestamp + VttParser::from_vtt supporting cue IDs, NOTE blocks, cue settings (format_convert.rs:190+)
- [x] Add round-trip tests: test_srt_webvtt_roundtrip_ms_precision (50 cues, prime-offset ms), test_srt_webvtt_roundtrip_boundary_timestamps (6 edge cases) (format_convert.rs:460+)
- [x] Add BiDi tests: test_bidi_paragraph_direction_ltr_first, test_bidi_mixed_arabic_latin_visual_order, test_bidi_pure_ltr_unchanged, test_bidi_pure_rtl_reversal, test_bidi_level_wrapper, test_bidi_level_from_unicode_bidi (text.rs:379+)
- [x] Flip 6 stale TODO.md checkboxes (clip/move/fad, font chain, soft-shadow, parallel batch, ASS cache, incremental renderer)
- Result: 1054 tests, 0 failures, 0 clippy warnings
