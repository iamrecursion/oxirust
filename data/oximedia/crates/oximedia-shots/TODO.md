# oximedia-shots TODO

## Current Status
- 38 public modules + lib.rs (~70 source files total) covering shot detection, classification, camera movement, composition analysis, continuity checking, pattern analysis, and export
- Key features: cut/dissolve/fade/wipe detection, 7 shot types (ECU to ELS), camera angle/movement classification, scene detection, coverage analysis, EDL/CSV/JSON export
- Dependencies: oximedia-core, oximedia-cv, oximedia-scene, oximedia-edl, oximedia-timecode

## Enhancements
- [x] Add adaptive threshold tuning in `detect::CutDetector` based on content complexity (e.g., action vs. dialogue scenes)
- [x] Extend `classify::ShotTypeClassifier` to support insert shots and cutaway classification
- [x] Add confidence calibration to `classify::AngleClassifier` using histogram-based features (verified 2026-05-16; src/confidence_calibration.rs:359 ConfidenceCalibrator with histogram similarity)
- [x] Improve `composition::CompositionAnalyzer` with golden ratio and phi grid analysis in addition to rule of thirds
- [x] Add multi-threaded frame processing in `ShotDetector::detect_shots` using rayon parallel iterators (verified 2026-05-16; src/lib.rs:226 rayon par_iter boundary detection, src/detector.rs:46)
- [x] Extend `export::shotlist` to support XML-based shot list formats (FCP XML, Resolve markers) (verified 2026-05-16; src/export/shotlist.rs:137 FCP XML, :213 DaVinci Resolve markers XML)
- [x] Add configurable wipe direction patterns in `detect::WipeDetector` (radial, iris, clock) (already implemented at wipe_patterns.rs:48 WipePatternKind::Radial/Iris/Clock + WipePatternDetector:145, Wave 14 verified)
- [x] Improve `continuity::ContinuityChecker` to detect 180-degree rule violations using optical flow direction (verified 2026-05-16; src/continuity/continuity_errors.rs:168 AxisViolation struct, 180-degree rule check:166)

## New Features
- [x] Add ML-based shot boundary detection module using lightweight convolutional features (no external ML runtime) (verified 2026-05-16; src/ml_boundary.rs:1 MlBoundaryDetector using hand-crafted conv features)
- [x] Implement `shot_similarity` module for finding visually similar shots across a project using perceptual hashing (verified 2026-05-16; src/shot_similarity.rs:91 ShotSimilarity, 849 lines)
- [x] Add `color_continuity` module to detect color temperature/grading inconsistencies between consecutive shots (verified 2026-05-16; src/color_continuity.rs:747 lines)
- [x] Implement `depth_of_field` estimation module to classify shallow vs. deep focus shots (verified 2026-05-16; src/depth_of_field.rs:57 DofEstimate, DofEstimator:113, 691 lines)
- [x] Add `audio_scene_boundary` integration with audio energy envelope for scene boundary refinement (verified 2026-05-16; src/audio_scene_boundary.rs:493 lines)
- [x] Implement real-time shot detection mode in `ShotDetector` with streaming frame input (already implemented at realtime.rs:107 RealtimeShotDetector + push_frame:160, Wave 14 verified)
- [x] Add `shot_annotation` module for attaching free-form metadata and tags to detected shots (verified 2026-05-16; src/shot_annotation.rs:922 lines)

## Performance
- [x] Cache histogram computations in `detect::CutDetector` to avoid recomputing for overlapping frame pairs (already implemented at detect/cut.rs:63 histogram_cache Mutex<HashMap<u64,...>>, Wave 14 verified)
- [x] Use downscaled proxy frames (e.g., 160x90) in `camera::MovementDetector` for faster optical flow estimation (implemented at camera/movement.rs: box_downsample_to_proxy + GrayImageProxy, PROXY_WIDTH=160, Wave 14 Slice G)
- [x] Implement block-based SAD comparison in `detect::DissolveDetector` instead of full-frame pixel difference (implemented at detect/dissolve.rs: detect_dissolve_block_sad, DISSOLVE_BLOCK_SIZE=16, DISSOLVE_SAD_THRESHOLD=2048, Wave 14 Slice G)
- [x] Add early termination in `classify::ShotTypeClassifier` when confidence exceeds 0.95 threshold (implemented at classify/shottype.rs: HIGH_CONFIDENCE_THRESHOLD=0.95, Wave 14 Slice G)
- [x] Pool `FrameBuffer` allocations in `detect_shots` to reduce allocation pressure on long sequences (implemented at frame_buffer.rs: FrameBufferPool + acquire/release, used in lib.rs detect_shots, Wave 14 Slice G)

## Testing
- [x] Add integration tests for `detect_shots` with synthetic frame sequences containing known cuts and dissolves (tests/detect_synthetic.rs:60 cut_sequence_yields_three_shots_at_expected_starts, :122 identical_frames_collapse_to_single_shot, :155 dissolve_ramp_scores_higher_than_abrupt_triplet, :194 dissolve_detector_below_minimum_returns_sentinel)
- [x] Test `continuity::ContinuityChecker` with crafted jump-cut and axis-crossing scenarios (tests/structure_analysis.rs:67 jump_cut_detected_for_identical_consecutive_shots, :103 axis_crossing_detected_for_opposed_pans_only, :144 distinct_shots_without_movement_have_no_issues)
- [x] Add round-trip tests for `export::edl` and `export::shotlist` (export then re-import) (tests/export_roundtrip.rs:139 shots_json_roundtrip_is_identity, :160 scenes_json_roundtrip_is_identity; note: EDL/CSV/FCP-XML/Resolve exporters are export-only with no re-importer in this crate, so the lossless round-trip is pinned via the JSON interchange path)
- [x] Test `scene::SceneDetector` with multi-scene sequences having distinct visual characteristics (tests/structure_analysis.rs:180 fade_transitions_partition_shots_into_scenes, :211 scene_boundary_requires_both_coverage_and_type_change)
- [x] Add property-based tests for `pattern::RhythmAnalyzer` with varying shot duration distributions (tests/structure_analysis.rs:241 rhythm_known_durations_produce_exact_metrics exact closed-form, :330 rhythm_invariants_hold_for_random_durations seeded SplitMix64 over 500 random vectors)

## Documentation
- [ ] Add architecture diagram showing the full detection pipeline (tracking -> estimation -> classification -> export)
- [ ] Document threshold tuning guidelines for different content types (documentary, action, interview)
- [ ] Add inline examples to `ShotDetector::detect_shots` showing how to iterate results
