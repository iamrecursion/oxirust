# oximedia-analysis TODO

## Current Status
- 43 source files across modules: `scene`, `black`, `quality`, `content`, `thumbnail`, `motion`, `color`, `audio`, `temporal`, `report`, `utils`
- Advanced modules: `action_detect`, `brand_detection`, `logo_detect`, `facial_analysis`, `crowd_analysis`, `text_detection`, `object_tracking`, `flicker_detect`, `frame_cadence`, `saliency_map`, `visual_attention`, `speech`, `audio_spectrum`, `audio_video_sync`, `noise_profile`, `edge_density`, `histogram_analysis`, `color_analysis`, `content_complexity`, `content_rating`, `motion_analysis`, `scene_stats`, `shot_list`, `temporal_analysis`, `temporal_stats`, `quality_score`
- Single-pass modular `Analyzer` with config builder pattern
- JSON and HTML report generation
- Dependencies: oximedia-core, oximedia-audio, oximedia-metering, oxifft, rayon, serde (ndarray removed)

## Enhancements
- [x] Add adaptive scene detection threshold in `scene` based on content type (high-motion vs static)
- [x] Implement dissolve and wipe detection in `scene` (currently focused on hard cuts) (verified 2026-05-16; src/scene.rs:66 dissolve_window, wipe_gradients:68, compute_wipe_score:295, detect_scene_change:183 dissolve/wipe branches)
- [x] Add per-scene quality scoring in `quality` (aggregate blockiness/blur/noise per scene segment)
- [x] Implement VMAF-like perceptual quality metric in `quality_score` module (verified 2026-05-16; src/vmaf_estimator.rs:18 VmafEstimator, estimate combining PSNR+SSIM:8, 222 lines)
- [x] Add face blur/obscurity detection in `facial_analysis` for privacy compliance (implemented 2026-06-01)
  - `BlurScore { variance_of_laplacian: f32, is_blurred: bool }` struct added; `BlurScore::compute(pixels, width, height, threshold)` uses Laplacian kernel normalised by pixel count.
  - `FacePrivacyAssessment.blur_score_detail: BlurScore` field added; populated in `assess_single_face`.
  - 3 tests: `test_sharp_face_scores_high`, `test_blurred_face_scores_low`, `test_blur_privacy_assessment`.
- [x] Implement letterbox/pillarbox detection in `black` module (partial black frame patterns)
- [x] Add interlacing detection in `frame_cadence` (detect combing artifacts)
- [x] Implement cadence pattern recognition in `frame_cadence` (3:2 pulldown, 2:2 pulldown)
- [x] Add configurable saliency model weights in `saliency_map` for different content types (verified 2026-05-16; src/saliency_map.rs:225 SaliencyModelWeights, ContentTypeHint:195 General/Face/Sports/Documentary/ScreenCapture/Nature, weights_for:309, 574 lines)
- [x] Implement multi-pass analysis mode for higher accuracy (second pass with refined parameters) (verified 2026-05-16; src/multi_pass.rs:98 MultiPassAnalyzer, analyze_first_pass:255, analyze_second_pass:255, 406 lines)

## New Features
- [x] Add BRISQUE blind image quality assessment in `quality` module
- [x] Implement commercial/advertisement break detection using `brand_detection` + `audio` cues (verified 2026-05-16; src/commercial_detect.rs:70 CommercialDetector, detect:102, CommercialBreak:35, 249 lines)
- [x] Add subtitle/burned-in text region detection in `text_detection` with OCR readiness score (verified 2026-05-16; src/text_detection.rs:11 TextRegion, OCR confidence score:20, TextFrame:68, TextTimeline:123, 302 lines)
- [x] Implement camera motion classification in `motion_analysis` (pan, tilt, zoom, static, handheld)
- [x] Add duplicate frame detection (frozen video) as separate detector (verified 2026-05-16; src/frozen_frame.rs:73 FrozenFrameDetector, FrozenRange:49, detect:103, 260 lines)
- [x] Implement bandwidth estimation module for adaptive streaming bitrate recommendations (verified 2026-05-16; src/bitrate_recommender.rs:41 BitrateRecommender, recommend:77, BitrateRecommendation:11, 186 lines)
- [x] Add color gamut analysis: detect out-of-gamut pixels for SDR/HDR content (verified 2026-05-16; src/gamut_analyzer.rs:1 GamutAnalyzer, GamutDescriptor Rec709/DCI-P3/Rec2020:6, out_of_gamut_ratio:69, 275 lines)
- [x] Implement shot composition analysis (rule of thirds, symmetry, leading lines) (verified 2026-05-16; src/shot_composition.rs:128 rule_of_thirds, compute_rule_of_thirds:238, RuleOfThirdsResult, 704 lines)
- [x] Add audio-video sync drift detection over time in `audio_video_sync`

## Performance
- [x] Replace ndarray with pure-Rust matrix operations to satisfy COOLJAPAN Pure Rust Policy
- [x] Parallelize `Analyzer::process_video_frame` sub-analyzers with rayon (implemented 2026-06-01)
  - `process_video_frame` in `src/lib.rs` dispatches all 8 sub-analyzers via `rayon::scope` with non-overlapping field borrows.
  - 2 tests: `test_parallel_output_equals_sequential` (determinism), `test_parallel_engages_multiple_analyzers` (multi-analyzer coverage).
- [x] Add downscaling option: analyze at half/quarter resolution for faster processing (implemented 2026-06-02)
  - `AnalysisScale { Full, Half, Quarter }` enum added to `src/lib.rs`; `AnalysisScale::Full` is the `#[default]` for full backward-compatibility.
  - `AnalysisConfig.analysis_scale: AnalysisScale` field added; `with_analysis_scale()` builder added.
  - `downsample_box_luma` (single-channel box filter) and `downsample_box_channels` (multi-channel) helpers added in `src/lib.rs`.
  - `process_video_frame` downsamples Y/U/V planes via box filter before dispatching to all rayon-parallel sub-analyzers when scale ≠ Full.
  - No coordinate rescaling required: all sub-analyzers in this crate emit frame indices and scalar statistics, not pixel coordinates.
  - 5 tests: `test_analysis_scale_default_is_full`, `test_downsample_box_correct_size_luma`, `test_downsample_box_correct_size`, `test_analysis_scale_half_fewer_pixels`, `test_analysis_scale_quarter`, `test_analysis_scale_full_vs_half_tolerance`.
- [x] Implement ring buffer for temporal analysis to bound memory usage for long videos (done — bounded VecDeque ring-buffers in src/temporal_analysis.rs)
- [x] Cache histogram computations across analyzers that share the same frame data (implemented 2026-06-01)
  - `FrameHistogramCache`, `HistogramData`, and `frame_hash_key` added to `src/histogram_analysis.rs`.
  - LRU eviction via `VecDeque<u64>` insertion-order queue; `get_or_compute` closure called at most once per key.
  - 2 tests: `test_histogram_cache_hit_skips_recompute`, `test_histogram_cache_miss_on_new_frame`.
- [x] Use integer histograms instead of float arrays in `color` and `scene` for speed (verified 2026-06-06: `type Histogram = [usize; 256];` at src/scene.rs:225)

## Testing
- [x] Add test with synthetic gradient image for `scene` detector (known scene boundary)
- [x] Test `black` detector with various near-black thresholds (implemented 2026-06-06; Wave 28 Slice 3)
  - New `tests/black_detection.rs` (64×64 Y-planes): `near_black_threshold_boundary` (threshold=16: 15u8 black via strict `<` / ratio, 16u8 closes the run), `full_black_all_zero_is_black` (all-0 run bounded by a non-black frame), `gray_253_not_black` (uniform 253 → empty).
  - Letterbox/pillarbox/windowbox is the separate `BlackBarDetector`, already covered by in-file unit tests; intentionally not duplicated.
- [x] Fix `BlackFrameDetector` segment-duration / `end_frame` divergence between close paths (fixed 2026-06-06; Wave 28 Slice 3)
  - ROOT CAUSE: `process_frame` close used `end_frame = frame_number` / `duration = frame_number - start` (FIRST non-black frame index) while `finalize` used `end_frame = start + lums.len()` / `duration = lums.len()` (accumulated sample count). Disagreed for non-contiguous frame numbers (every-Nth sampling), e.g. black at 0,2,4,6 → 8 vs 4.
  - FIX: extended `current_segment` to track `last_seen_frame`; both paths now route through one private `close_segment()` helper setting `end_frame = last_seen_frame + 1` (preserves documented exclusive semantics, so existing contiguous in-file tests stay green) and `duration = lums.len()`.
  - Regression test `non_contiguous_frame_numbers_consistent_duration` proves the mid-stream and `finalize` close paths yield byte-identical `BlackSegment`s for the same run; existing `test_black_frame_detection` / `test_black_frame_min_duration` remain green.
  - Also documented the deliberate per-pixel `<` (BlackFrameDetector) vs `<=` (BlackBarDetector) seam in `src/black.rs` (behaviour unchanged).
- [ ] Add `quality` assessor accuracy test with reference VQEG test sequences
- [ ] Test `thumbnail` selector diversity: verify selected frames span different scenes
- [ ] Test `report` HTML output renders valid HTML (basic tag matching validation)
- [x] Add integration test: process 30fps 10-second synthetic video through full analyzer pipeline

## Documentation
- [ ] Document each analyzer's algorithm and computational complexity
- [ ] Add example showing custom `AnalysisConfig` for broadcast QC workflow
- [ ] Document `report` output schema for JSON format (field descriptions and value ranges)
