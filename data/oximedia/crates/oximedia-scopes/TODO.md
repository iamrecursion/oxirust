# oximedia-scopes TODO

## Current Status
- 33 modules covering waveform monitors (luma, RGB parade, RGB overlay, YCbCr), vectorscope (circular/rectangular), histogram (RGB/luma), parade displays, false color exposure, CIE 1931 chromaticity, focus assist, HDR waveform (PQ/HLG/nits), audio scopes, bit depth analysis, clipping detection, gamut scope, Lissajous, loudness scope, motion vector scope, overlay rendering, peaking, zebra patterns, signal stats
- Core types: VideoScopes, ScopeConfig, ScopeType (14 variants), ScopeData, WaveformMode, VectorscopeMode, HistogramMode, GamutColorspace
- Dependencies: oximedia-core, rayon, thiserror

## Enhancements
- [x] Add 10-bit and 12-bit precision paths in `waveform` (currently implied but not explicit in ScopeConfig)
- [x] Extend `vectorscope` with zoom/pan controls and IQ vs UV display mode selection
- [x] Add `histogram_stats` percentile markers (1%, 5%, 95%, 99%) overlay on histogram display
- [x] Implement `clipping_detector` with configurable head/foot room thresholds per broadcast standard
- [x] Extend `false_color_mapping` with custom user-defined color LUT support
- [x] Add `gamut_scope` with triangle outline overlay for Rec.709/P3/Rec.2020 boundaries on CIE diagram (verified 2026-05-16; src/gamut_scope_overlay.rs:4 GamutBoundaryOverlay, GamutTriangle:27 rec2020/dci_p3)
- [x] Implement `audio_scope` stereo phase correlation meter alongside existing Lissajous display (verified 2026-05-16; src/audio_scope.rs:1049 lines)
- [x] Add `scope_layout` multi-scope grid arrangement with configurable scope combinations (verified 2026-05-16; src/scope_layout.rs:299 lines)

## New Features
- [x] Add `timecode_overlay` module for burning timecode and frame number onto scope displays (verified 2026-05-16; src/timecode_overlay.rs:625 lines)
- [x] Implement `scope_comparison` for side-by-side before/after scope display of color-graded footage (verified 2026-05-16; src/scope_comparison.rs:558 lines)
- [x] Add `safe_area_overlay` module showing title-safe and action-safe boundaries per broadcast standard (verified 2026-05-16; src/safe_area_overlay.rs:452 lines)
- [x] Implement `scope_recording` for capturing scope data over time as time-series for analysis (verified 2026-05-16; src/scope_recording.rs:523 lines)
- [x] Add `exposure_histogram` module with zone system (Ansel Adams) overlay for cinematography (verified 2026-05-16; src/exposure_histogram.rs:556 lines)
- [x] Implement `chroma_level_scope` dedicated Cb/Cr level display with broadcast-legal limits (verified 2026-05-16; src/chroma_level_scope.rs:602 lines)
- [x] Add `scope_snapshot` for exporting scope displays as PNG images with metadata annotations (verified 2026-05-16; src/scope_snapshot.rs:448 lines)
- [x] Implement `noise_floor_scope` for measuring and displaying signal-to-noise ratio per channel (verified 2026-05-16; src/noise_floor_scope.rs:529 lines)
- [x] Add `color_checker_scope` for analyzing captured color checker charts against reference values (verified 2026-05-16; src/color_checker_scope.rs:768 lines)

## Performance
- [x] Add SIMD-optimized RGB-to-YCbCr conversion in `waveform` and `vectorscope` generation (implemented 2026-05-15; src/simd_convert.rs: `rgb_to_ycbcr_simd`, `convert_batch_bt601_simd` — BT.601 coefficients, auto-vectorizable 8-pixel batch loop, wired into `generate_ycbcr_waveform` and `generate_vectorscope`)
- [x] Implement downsampled analysis in `analyze` for real-time preview at reduced accuracy (implemented 2026-05-15; src/lib.rs: `AnalysisConfig { downsample_factor: usize }`, `VideoScopes::analyze_downsampled()`)
- [x] Add incremental scope update in `VideoScopes` (update from changed regions only) (implemented 2026-05-15; src/lib.rs: `RegionUpdate`, `VideoScopes::update_region()`, `seed_from_frame()`, per-channel running `channel_bins` accumulators)
- [x] Use rayon parallel column processing in `parade::generate_rgb_parade` (implemented 2026-05-15; src/parade.rs: parallel row-fold + reduce pattern via rayon replacing serial nested loop)
- [x] Optimize `histogram` bin accumulation with thread-local bins merged at end (implemented 2026-05-15; src/histogram.rs: `par_chunks(3).fold().reduce()` in both `generate_rgb_histogram` and `generate_luma_histogram`)
- [x] Add configurable scope resolution independent of output resolution for faster rendering (implemented 2026-05-15; src/lib.rs: `ScopeResolution { width, height }` added to `ScopeConfig`, `VideoScopes::generate_scopes()`)

## Testing
- [x] Add tests for `waveform` with known test patterns (color bars, ramp, flat field) (implemented 2026-05-31; src/waveform.rs: test_waveform_color_bars_dimensions_and_non_empty, test_waveform_color_bars_rgb_parade_has_data, test_waveform_luma_ramp_shape, test_waveform_flat_black_field_pixels_near_bottom, test_waveform_flat_white_field_pixels_near_top, test_waveform_10bit_ramp_via_hd_api, test_waveform_stats_10bit_flat_grey — 7 tests)
- [x] Test `vectorscope` with SMPTE color bar input and verify target dot positions (implemented 2026-05-31; src/vectorscope.rs: test_vectorscope_smpte_75_produces_valid_output, test_vectorscope_smpte_75_target_positions_in_bounds, test_vectorscope_pure_red_in_red_quadrant, test_vectorscope_pure_cyan_opposite_red, test_vectorscope_gray_near_center, test_smpte_target_list_contract — 6 tests)
- [ ] Add tests for `false_color` mapping consistency across Rec.709 and HLG inputs
- [ ] Test `histogram` with uniform-distribution input and verify flat histogram output
- [ ] Add roundtrip tests for `ScopeConfig` serialization/deserialization

## Documentation
- [ ] Add visual reference guide showing each scope type with example output images
- [ ] Document broadcast compliance checking workflow using `compliance` module
- [ ] Add guide for interpreting vectorscope displays (skin tone line, gamut boundaries, color balance)
- [ ] Document HDR scope usage with PQ, HLG, and nits scale interpretation
