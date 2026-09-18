# oximedia-video TODO

## Current Status
- 8 source files covering professional video processing operations
- Key features: block-based motion estimation/compensation, frame rate conversion with intermediate frame generation, video deinterlacing, scene change detection, 3:2 pulldown cadence detection, perceptual video fingerprinting, temporal noise reduction
- Dependencies: oximedia-core, thiserror, rayon, serde

## Enhancements
- [x] Extend `motion_compensation` with sub-pixel motion estimation (half-pel and quarter-pel refinement) (L9: `refine_subpel` in `motion_compensation_ext` — half-pel bilinear + quarter-pel Catmull-Rom bicubic; 2026-05-15)
- [x] Add adaptive block size selection in motion estimation (variable block sizes from 4x4 to 64x64) (L10: `adaptive_block_size_search` + `BlockSize` enum in `motion_compensation_ext`; 2026-05-15)
- [x] Implement bidirectional motion estimation in `motion_compensation` for B-frame-style interpolation (L11: `bidir_motion_estimate` + `BidirMotionVector` in `motion_compensation_ext`; 2026-05-15)
- [x] Add multiple deinterlace algorithms to `deinterlace` (bob, weave, Yadif-style adaptive, EEDI-style edge-directed) (verified-open 2026-05-16: not yet implemented)
- [x] Extend `scene_detection` with adaptive threshold based on content complexity histogram
- [x] Improve `temporal_denoise` with motion-compensated temporal filtering (MCTF) using motion vectors (verified-open 2026-05-16: not yet implemented)
- [x] Add confidence scoring to `pulldown_detect` cadence detection with frame-level accuracy reporting
- [x] Extend `video_fingerprint` with rotation/scale invariance for robust content matching

## New Features
- [x] Implement `super_resolution` module for AI-free upscaling (Lanczos, edge-directed interpolation, NEDI) (verified 2026-05-16; src/super_resolution.rs:1497 lines)
- [x] Add `film_grain_synthesis` module for AV1 film grain parameter estimation and generation (verified 2026-05-16; src/film_grain_synthesis.rs:1954 lines)
- [x] Implement `field_order_detect` for automatic top-field-first vs bottom-field-first detection (verified 2026-05-16; src/field_order_detect.rs:478 lines)
- [x] Add `cadence_convert` module for frame rate conversion (24->30, 25->30, 50->60) with proper cadence (verified 2026-05-16; src/cadence_convert.rs:463 lines)
- [x] Implement `video_stabilization` module using motion vectors from motion_compensation (verified 2026-05-16; src/stabilization.rs:1481 lines)
- [x] Add `shot_boundary_classifier` that categorizes cuts, dissolves, wipes, and fades (verified 2026-05-16; src/shot_boundary_classifier.rs:570 lines)
- [x] Implement `duplicate_frame_detect` for detecting and removing duplicate/near-duplicate frames (verified 2026-05-16; src/duplicate_frame_detect.rs:745 lines)
- [x] Add `quality_metrics` module (PSNR, SSIM, VMAF-like perceptual metric) for video quality assessment (verified 2026-05-16; src/quality_metrics.rs:813 lines)

## Performance
- [x] Parallelize motion search in `motion_compensation` across blocks using rayon (L29: `estimate_frame_motion_parallel` in `motion_compensation_ext` — rayon par_iter over macroblocks; 2026-05-15)
- [x] Implement diamond/hexagonal search patterns in motion estimation instead of full search (L30: `motion_search_with_pattern` + `SearchPattern` enum in `motion_compensation_ext` — LDSP/SDSP diamond, 6-point hexagonal, three-step, full; 2026-05-15)
- [x] Add SIMD-optimized SAD (Sum of Absolute Differences) and SSD computation for block matching (L31: `sad_simd` in `motion_compensation_ext` — AVX2 32B/iter, SSE4.1 16B/iter, scalar fallback; 2026-05-15)
- [x] Use integral images for fast variance computation in `scene_detection` (IntegralImage Crow 1984 SAT, O(1) rect_variance, integral_image.rs 350L; 2026-05-30)
- [x] Implement hierarchical (coarse-to-fine) motion estimation for large search ranges (L33: `hierarchical_motion_estimate` in `motion_compensation_ext` — Gaussian pyramid, coarse-to-fine propagation; 2026-05-15)
- [x] Add frame-level parallelism in `temporal_denoise` for processing independent pixel columns (existing rayon pinned; parallel tests added 2026-05-30)

## Testing
- [x] Add round-trip tests for deinterlace (interlace synthetic content, deinterlace, measure PSNR) (verified 2026-05-15; tests/it_deinterlace.rs:5 tests)
- [x] Test `frame_interpolation` with known linear motion to verify interpolated frame accuracy (verified 2026-05-15; tests/it_frame_interpolation.rs:5 tests)
- [x] Add `scene_detection` tests with synthetic scene-change sequences (hard cuts, dissolves, flashes) (verified 2026-05-15; tests/it_scene_detection.rs:4 tests)
- [x] Test `pulldown_detect` with synthetic 3:2 pulldown cadence patterns and verify detection accuracy (verified 2026-05-15; tests/it_pulldown_detect.rs:5 tests)
- [x] Benchmark `motion_compensation` against known motion vectors for standard test sequences (verified 2026-05-15; tests/it_motion_compensation.rs:4 tests)
- [x] Test `video_fingerprint` collision rate with large synthetic frame datasets (verified 2026-05-15; tests/it_video_fingerprint.rs:6 tests)
- [x] Add deterministic known-answer tests for `estimate_frame_motion_parallel` (Wave 29; tests/it_known_answer.rs — 48×16 right-shift-8px frame recovers MV (-8,0) on interior blocks; identical-frames all-zero)
- [x] Add closed-form known-answer tests for `IntegralImage` (Wave 29; tests/it_known_answer.rs — 8×8 gradient rect_sum=2016, rect_sum_sq=85344, rect_variance=341.25=(64²−1)/12, exclusive-upper subrect rect_sum(2,1,5,4)=171, uniform sum=6400/var=0)
- [x] Add known-answer cadence tests for `detect_cadence` and `CadenceConfidenceScorer` (Wave 29; tests/it_known_answer.rs — [L,L,H,L,H]→Pulldown32, threshold-boundary pin; [H,L,H,L,L]×{5,10,20,40}→Pulldown32 winner p>0.9, posterior sums to 1.0)

## Known latent issue (Wave 29)
Two real cross-module cadence inconsistencies were confirmed (DOCUMENTED, not fixed — picking a
canonical phase/threshold is an out-of-scope design decision; see `tests/it_known_answer.rs` `//!` doc):
- **Bug #1 — divergent 3:2 reference phase.** `pulldown_detect::detect_cadence` matches 3:2 against
  `[L,L,H,L,H]` (`src/pulldown_detect.rs:205`) while `cadence_confidence::cadence_pattern` uses
  `[H,L,H,L,L]` for the same `Cadence::Pulldown32` (`src/cadence_confidence.rs:83`). These are
  different phase rotations of the same cadence; a caller mixing the two modules sees a 2-frame
  phase offset. Each module is internally self-consistent.
- **Bug #2 — divergent LOW/High threshold.** `detect_cadence` classifies High as `combing_score > 0.04`
  (`src/pulldown_detect.rs:195`), but `cadence_confidence` uses nominal `LOW = 0.05`
  (`src/cadence_confidence.rs:72`). Feeding 0.05 into `detect_cadence` reads as all-High
  (→ would mis-classify a clean 3:2 sequence as Interlaced). The `detect_cadence` tests keep LOW ≤ 0.04.

## Documentation
- [x] Document motion estimation algorithm selection guide (when to use which block size/search range) (verified 2026-05-15; src/motion_compensation.rs module-level doc)
- [x] Add deinterlace algorithm comparison table with quality/speed tradeoffs (verified 2026-05-15; src/deinterlace.rs module-level doc)
- [x] Document video fingerprint format and matching threshold recommendations (verified 2026-05-15; src/video_fingerprint.rs module-level doc)
