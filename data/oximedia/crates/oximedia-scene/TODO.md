# oximedia-scene TODO

## Current Status
- 33 modules (9 subdirectory modules) covering scene classification, object/face/logo detection, activity recognition, shot composition analysis, semantic segmentation, saliency detection, aesthetic scoring, event detection, feature extraction, camera motion, crowd density, depth of field, mood analysis, pacing, storyboard generation, summarization, transitions, visual rhythm
- Common types: Point, Rect (with IoU), Confidence
- All algorithms patent-free (HOG, Haar cascades, color histograms, motion histograms, spectral saliency, graph-based segmentation)
- Dependencies: oximedia-core, oximedia-cv, scirs2-core, rayon, serde, thiserror

## Enhancements
- [x] Add multi-scale detection in `detect::face::FaceDetector` for improved accuracy across face sizes (verified 2026-05-16; src/detect/face.rs:95 scale_factor for multi-scale, NMS across scales:211)
- [x] Extend `classify::scene::SceneClassifier` with temporal smoothing to avoid flickering classifications
- [x] Add `composition::rules::CompositionAnalyzer` support for golden ratio and phi grid in addition to rule of thirds (verified 2026-05-16; src/composition/rules.rs:15 golden_ratio+phi_grid fields, phi_grid fn:32)
- [x] Implement non-maximum suppression (NMS) in `detect` for overlapping detections across all detector types
- [x] Extend `saliency` with temporal saliency for video (motion-weighted attention maps) (implemented 2026-05-31: `temporal_saliency()` free fn, `TemporalSaliencyFnConfig`, `TemporalSaliencyAccumulator` with Gaussian smoothing + pixel-level diff; 4 tests in saliency/temporal.rs fn_api_tests)
- [x] Add `scene_boundary` configurable threshold sensitivity with automatic threshold estimation
- [x] Improve `camera_motion` estimation with robust RANSAC-based homography fitting (verified 2026-05-16; src/camera_motion.rs:238 estimate_ransac, RANSAC iterations:244)
- [x] Extend `aesthetic` scoring with content-type-specific models (landscape, portrait, action, still life) (verified 2026-05-16; src/aesthetic/score.rs:184 content-type-specific models, AutoScorer:371)

## New Features
- [x] Add `text_detection` module for detecting and localizing text regions in frames (scene text, overlays, subtitles) (verified 2026-05-16; src/text_detect.rs:1231 lines, src/detect/text.rs)
- [x] Implement `emotion_recognition` module analyzing facial expressions in detected faces (verified 2026-05-16; src/emotion_recognition.rs:691 lines EmotionRecognizer)
- [x] Add `object_tracking` module for tracking detected objects across frames with Kalman filtering (verified 2026-05-16; src/object_tracker.rs:751 lines ObjectTracker Kalman filtering)
- [x] Implement `scene_captioning` module generating natural-language descriptions from scene features (verified 2026-05-16; src/scene_captioning.rs:665 lines SceneCaptioner)
- [x] Add `content_moderation` module for detecting sensitive content categories (violence, NSFW) (verified 2026-05-16; src/content_moderation.rs:827 lines ContentModerator)
- [x] Implement `temporal_graph` module connecting scene analysis results across time for narrative structure (verified 2026-05-16; src/temporal_graph.rs:677 lines TemporalGraph)
- [x] Add `thumbnail_selector` module choosing the most visually representative frame per scene (verified 2026-05-16; src/thumbnail_selector.rs:446 lines ThumbnailSelector)
- [x] Implement `motion_energy` module for quantifying overall motion intensity per scene segment (verified 2026-05-16; src/motion_energy.rs:536 lines MotionEnergy)
- [x] Add `audio_visual_correlation` module detecting sync between audio events and visual changes (verified 2026-05-16; src/audio_visual_correlation.rs:629 lines AudioVisualCorrelation)

## Performance
- [x] Add rayon parallel processing in `segment::Segmenter` for graph-based segmentation on large frames (implemented 2026-05-31: rayon par_iter in segment/mod.rs `segment_foreground_parallel`; `test_segmenter_parallel_matches_sequential` test added)
- [x] Implement multi-resolution pyramid processing in `detect` to avoid full-resolution scans (verified 2026-05-16; src/detect/pyramid.rs:1 multi-resolution image pyramid, levels:18)
- [x] Add batch frame processing in `classify` to amortize model initialization across frames (verified 2026-05-16; src/classify/batch.rs:75 BatchClassifier amortizes init, classify_batch:177)
- [x] Cache feature descriptors in `features` module to avoid recomputation when multiple analyzers use them (verified 2026-05-16; src/features/cache.rs:12 FeatureCache, get_cached/set_cached)
- [x] Optimize `saliency::spectral` FFT-based saliency with pre-allocated FFT buffers (implemented 2026-05-31: `SpectralSaliencyDetector` owns pre-alloc'd gray/blur/saliency buffers; `SpectralSaliencyComputer` simple grayscale wrapper; 2 new tests `test_spectral_saliency_computer_reuse` + `test_spectral_saliency_computer_dimensions`)
- [x] Add frame decimation in `summarization` to process every Nth frame for long videos (VideoSummarizer + FrameDecimationConfig done)

## Testing
- [ ] Add accuracy benchmarks for `detect::face` using standard face detection datasets (FDDB-like metrics)
- [ ] Test `scene_boundary` detection against known scene-change timestamps in reference videos
- [x] Add tests for `composition::rules` with synthetically generated images with known compositions
- [x] Test classification consistency across sequential frames of the same scene (done 2026-06-01; `test_classification_temporal_consistency` in `src/classify/batch.rs` — 10 near-identical 32×32 frames, dominant label ≥9/10)
- [x] Add regression tests for `aesthetic` scoring to ensure score stability across versions

## Documentation
- [ ] Document all patent-free algorithms used with academic references and complexity analysis
- [ ] Add guide for combining multiple analysis modules into a complete video analysis pipeline
- [ ] Document confidence threshold selection guidelines for each detector type
- [ ] Add performance benchmarks table (frames/second) for each analysis module at common resolutions
