# oximedia-auto TODO

## Current Status
- 55 modules providing automated video editing: highlights, cuts, assembly, rules, scoring, pacing curves, subject tracking, reframing, and more
- Core modules: `smart_crop`, `smart_reframe` (+ `SubjectTracker`, `VerticalToHorizontalParams`), `smart_trim`, `music_sync`, `tempo_detect`, `narrative`, `scene_classifier`, `color_match`, `subtitle_sync`, `tag_suggest`, `visual_theme`
- Pacing: `pacing_curve` (`PacingCurve` with 7 `CurveShape` variants, `CurveAnalyser`, `CurveStats`)
- Complete `AutoEditor` pipeline with `auto_edit()` orchestrating detect-score-cut-assemble workflow
- All numerical operations use plain Rust primitives (f32/f64/Vec/fixed arrays); no ndarray dependency

## Enhancements
- [x] Replace `ndarray` dependency with SciRS2-Core per SCIRS2 policy (confirmed: ndarray was never present in Cargo.toml or source; all array ops use plain Rust)
- [x] Improve `highlights::HighlightDetector` with configurable multi-pass analysis (coarse then fine) (verified 2026-05-16; src/highlights.rs:253 MultiPassConfig, coarse-to-fine detection:267)
- [x] Add confidence scores to `cuts::CutPoint` for user review prioritization (verified 2026-05-16; src/cuts.rs:80 CutPoint struct, confidence field:85)
- [x] Extend `assembly::AssemblyType` with `Recap` variant for episode/series recaps (verified 2026-05-16; src/assembly.rs:39 Recap variant)
- [x] Add `rules::PacingPreset::Custom` with user-defined shot duration curves (`pacing_curve` module: `PacingCurve`, `CurveShape` 7 variants, `CurveKeyframe`, `CurveAnalyser`; `distribute_clips`/`compute_cut_positions`; 27 tests)
- [x] Improve `scoring::SceneScorer` with temporal context (score relative to neighbors) (verified 2026-05-16; src/scoring.rs:509 TemporalContextConfig, neighbor bonus:516)
- [x] Extend `smart_reframe` with subject tracking across multiple frames for smooth panning (`SubjectTracker` + `SubjectBounds` with EMA; `generate_sequence`/`generate_smooth_sequence`; 10 tests)
- [x] Add vertical-to-horizontal reframing in `smart_reframe` (`VerticalToHorizontalParams`, `VerticalToHorizontalStrategy` 5 variants, `FrameOrientation`; `primary_placement`/`side_regions`/`saliency_crop_window`; 7 tests)
- [x] Improve `music_sync` with downbeat vs upbeat distinction for edit point selection (verified 2026-05-16; src/music_sync.rs:12 BeatGrid, downbeats_ms:16)

## New Features
- [x] Add `auto_thumbnail` module for automatic thumbnail selection from best frames (verified 2026-05-16; src/auto_thumbnail.rs:903 lines)
- [x] Implement `auto_chaptering` module to generate chapter points from scene analysis (verified 2026-05-16; src/auto_chaptering.rs:1099 lines)
- [x] Add `content_warning` module for automatic content classification (violence, language) (verified 2026-05-16; src/content_warning.rs:1064 lines)
- [x] Implement `engagement_predictor` module using interest curve analysis for audience retention (verified 2026-05-16; src/engagement_predictor.rs:853 lines)
- [x] Add `a_b_roll` module for automatic B-roll insertion suggestions based on dialogue content (verified 2026-05-16; src/a_b_roll.rs:1131 lines)
- [x] Implement `color_continuity` checker that flags jarring color shifts between assembled clips (verified 2026-05-16; src/color_continuity.rs:767 lines)
- [x] Add platform-specific export presets for YouTube Shorts, Instagram Reels, TikTok in `assembly` (verified 2026-05-16; src/assembly.rs:79 PlatformPreset enum, YouTubeShorts:81, InstagramReels:83, TikTok:85)

## Performance
- [x] Parallelize `highlights::detect_highlights()` across frame batches using rayon
- [x] Cache scene features in `scoring::SceneScorer` to avoid recomputation on config changes (completed 2026-06-01)
  - `SceneId`, `SceneComponentScores`, `feature_cache: HashMap<SceneId, SceneComponentScores>` added to `SceneScorer`; `score_scene` / `score_scene_with_context` take `&mut self`; `invalidate_cache` / `clear_scene` for manual invalidation; `batch_score_scenes` updated to `&mut SceneScorer`; bench file updated. 4 new tests.
- [x] Add early termination to `cuts::detect_cuts()` when sufficient cut points found for target duration (completed 2026-06-01)
  - `DetectCutsOptions { target_duration_ms, max_cuts }` added; `CutDetector::detect_cuts_with_options` selects prefix after full sort to guarantee consistency with full scan. 4 new tests.
- [x] Use downscaled frames for initial `smart_crop` pass, refine on full resolution only for final crops (completed 2026-06-01)
  - `SmartCropConfig.coarse_scale: f32` (default 0.25); `suggest_crop_from_frame(frame, w, h, ch)` with box-average downscale, variance-based synthetic saliency, ±10 % margin fine pass; `extract_saliency_regions` 4×4 luma-variance grid. 3 new tests.
- [x] Implement lazy evaluation for `AutoEditResult` fields that may not be needed by caller (completed 2026-06-24; `LazyAutoEditResult` with `OnceLock`-cached `interest_curve()` + `try_assembled()`; `auto_edit_lazy()` defers steps 5 & 10; `into_eager()` forces both)

## Testing
- [x] Add test for `auto_edit()` end-to-end with synthetic video frames and audio (completed 2026-06-24; 3 tests in `src/lib.rs`: `test_auto_edit_end_to_end_synthetic`, `test_lazy_auto_edit_defers_assembly`, `test_lazy_into_eager_matches_eager`)
- [x] Test `rules::RulesEngine::apply_rules()` enforces minimum shot duration constraints (already implemented: `test_enforce_min_shot_duration` in `src/rules.rs`)
- [x] Add regression test for `assembly::generate_social_clip()` ensuring output within duration tolerance (already implemented: `test_generate_social_clip_15s` / `test_generate_social_clip_30s` in `src/assembly.rs`)
- [x] Test `smart_trim` preserves dialogue boundaries when trimming (completed 2026-06-24; `test_trim_preserves_dialogue_boundaries` in `src/smart_trim.rs`)
- [x] Add benchmark comparing `scoring` performance across different `ScoringConfig` settings (already implemented: `benches/scoring_bench.rs` — `bench_score_scene_configs` with 6 config variants)
- [x] Test `color_match` produces consistent results for identical input pairs (completed 2026-06-24; `test_color_transfer_identical_inputs_deterministic` in `src/color_match.rs`)

## Documentation
- [x] Document use-case presets ("trailer", "highlights", "social") with expected behavior (completed 2026-06-24; expanded doc on `AutoEditorConfig::for_use_case` and `AutoEditor::for_use_case` with preset table)
- [x] Add flowchart for `auto_edit()` pipeline showing data flow between stages (completed 2026-06-24; numbered 10-step pipeline doc added to `auto_edit()` method)
- [x] Document `scoring::FeatureWeights` tuning guidelines for different content types (completed 2026-06-24; added `# Tuning Guidelines` section with 4 content-type presets: sports, interview, music video, nature/B-roll)
