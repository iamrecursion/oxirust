# oximedia-align TODO

## Current Status
- 36 source files across modules: `temporal`, `spatial`, `features`, `distortion`, `color`, `markers`, `rolling_shutter`, `optical_flow`, `phase_correlate`, `affine`, `transform`, `warp`, `icp`, `lip_sync`, `drift_correct`, `drift_correction`, `gradient_flow`, `sync_score`, `stereo_rectify`, `beat_align`, `elastic_align`, `frame_matcher`, `frequency_align`, `multicam_sync`, `multitrack_align`, `motion_compensate`, `multi_stream`, `audio_align`, `subframe_interp`, `tempo_align`, `temporal_align`, `align_report`
- Audio cross-correlation sync, timecode sync, visual marker detection
- Homography estimation with RANSAC, perspective correction, feature matching (FAST, BRIEF, ORB)
- Brown-Conrady and fisheye lens distortion correction
- Color transfer, histogram matching, white balance
- Rolling shutter correction
- Dependencies: oximedia-core, oximedia-audio, nalgebra, rayon, serde

## Enhancements
- [x] Add KLT (Kanade-Lucas-Tomasi) tracker to `optical_flow` for sparse flow tracking
  — `KltTracker::track_features(prev, curr, width, height, &[(f32,f32)]) -> Vec<Option<(f32,f32)>>` added to `klt_tracker`
- [x] Implement multi-scale feature matching in `features` (pyramid ORB for scale invariance) (verified 2026-05-16; src/features.rs:1014 FeaturePyramid, build_pyramid:1043, detect_features_at_all_levels:1109, num_levels:1516, 1757 lines)
- [x] Add PROSAC as alternative to RANSAC in `spatial` for faster convergence with sorted matches
  — `ProsacEstimator` in `prosac` module (Affine + Homography models)
- [x] Implement weighted least squares in `HomographyEstimator` for refined estimates after RANSAC
- [x] Add sub-pixel refinement for FAST corner detection in `features`
- [x] Implement exposure-invariant feature descriptors in `features` for mixed-lighting scenarios (verified 2026-05-16; src/illumination_invariant.rs:25 IlluminationInvariantConfig, IlluminationInvariantDescriptor:48, 334 lines)
- [x] Add temporal smoothing to `rolling_shutter` correction to prevent frame-to-frame jitter
- [x] Implement spectral domain audio alignment in `audio_align` using phase correlation
- [x] Add genlock drift estimation in `drift_correct` for long-duration multi-camera recordings
- [x] Implement bundle adjustment in `multicam_sync` for globally consistent multi-camera calibration (verified 2026-05-16; src/bundle_adjust.rs:155 BundleAdjuster, BundleAdjustConfig:32, adjust:192, BundleAdjustResult:141, 799 lines)

## New Features
- [x] Add dense optical flow (Farneback method) to `optical_flow` module
  — `compute_dense_flow(prev: &[f32], curr: &[f32], width: u32, height: u32) -> Vec<(f32,f32)>`
  — Full Farneback implementation in `farneback_flow`; convenience bridge in `optical_flow`
- [x] Implement image stitching pipeline: detect features -> match -> estimate homography -> blend (verified 2026-05-16; src/image_stitch.rs:125 ImageStitcher, StitchConfig:36, chain_homographies:564, bilinear_sample:550, 595 lines)
- [x] Add automatic lens calibration from checkerboard/ArUco marker detection (verified 2026-05-16; src/lens_calibration.rs:37 CheckerboardDetector, ArUco-style:8, CameraCalibrator:349, 955 lines)
- [x] Implement depth-from-stereo using `stereo_rectify` + block matching (implemented 2026-05-15: src/stereo_depth.rs — StereoDepthEstimator::compute_depth_map using SAD block matching + depth=f*B/d formula; tests: test_depth_map_uniform_disparity, test_depth_map_zero_disparity_infinite_depth, test_depth_map_known_shift; 650/650 tests pass)
- [x] Add motion-compensated temporal interpolation for frame rate conversion (verified 2026-05-16; src/motion_interp.rs:36 MotionInterpolator, 202 lines)
- [x] Implement video stabilization pipeline combining `optical_flow` + `affine` + `warp` (verified 2026-05-16; src/stabilize.rs:286 stabilize_pipeline, StabilizeConfig:36, compute_corrections:202, 799 lines)
- [x] Add network-based time sync (PTP/NTP) for distributed camera systems in `temporal` (implemented 2026-05-15: NtpConfig/TimeDelta/NtpClient::query_offset added to temporal.rs — SNTP RFC 4330 single-packet UDP exchange; tests: test_ntp_packet_parse_known_bytes, test_ntp_offset_computation_formula, test_ntp_unix_roundtrip; 650/650 tests pass)
- [x] Implement scene-based alignment: detect scene changes and align segments independently (implemented 2026-05-31: src/scene_align.rs — SceneAligner::detect_scenes (luma histogram L1 diff, O(n)), SceneAligner::align (DP monotone segment matching χ² cost, O(n×m)), AlignedSegment with frame_offset/confidence; 22 tests)

## Performance
- [x] Add SIMD-accelerated Hamming distance for BRIEF descriptor matching in `features`
  — `hamming_distance_simd(a: &[u8], b: &[u8]) -> u32` using u64 popcount batching
- [x] Implement parallel RANSAC with early termination across rayon threads (already done — verified 2026-06-05: src/parallel_ransac.rs ParallelRansac::estimate runs batches via `.into_par_iter()` (line ~139), adaptive `log(1-p)/log(1-w^s)` iteration bound via `prosac::adaptive_max_iterations` (line ~226) with `confidence: 0.99`, and `early_termination_ratio` short-circuit setting `early_terminated` (lines ~182, ~221); src/spatial.rs HomographyEstimator::estimate (line ~120) is also rayon-parallel via `.into_par_iter()` (line ~151) with a shared AtomicUsize best-inlier tracker)
- [x] Use integral images for fast feature detection in `features` (box filter acceleration) (done — SAT integral image at src/features.rs)
- [x] Optimize `phase_correlate` FFT computation with real-valued FFT (half the complex ops)
  - **Goal:** Switch `src/phase_correlate.rs` forward transforms from full-complex `oxifft::fft/ifft` (current: real input converted to Complex<f64>) to OxiFFT real-valued `rfft`/`irfft` (N/2+1 bins, half the complex ops), per COOLJAPAN OxiFFT policy. Cross-power spectrum + inverse transform + peak-find stay identical. Result must match full-complex path within fp tolerance. (Wave 20 Slice F, 2026-06-02)
  - **Design:** Replace the two forward FFTs of real reference/target with rfft (N/2+1 bins); compute normalized cross-power spectrum on half-spectrum; irfft back to real correlation surface. If OxiFFT lacks 2-D rFFT, use rfft row-wise + complex FFT column-wise (standard real-2D decomposition). Sub-pixel parabolic refine unchanged.
  - **Files:** `src/phase_correlate.rs`, `TODO.md`
  - **Tests:** rFFT estimate == full-complex estimate within tolerance on known integer shift; sub-pixel shift ±0.1px; zero-shift → peak at origin; Gaussian-noise robustness; assert against full-complex result on same input.
  - **Done:** `oxifft::rfft`/`irfft` confirmed available (N/2+1 bins); `phase_correlate_1d` now uses rFFT; `phase_correlate_1d_full_complex` kept as private regression reference; 4 new tests added (integer-shift, sub-pixel, zero-shift, noise-robustness).
- [ ] Add GPU acceleration path for dense optical flow in `optical_flow`
- [x] Cache feature descriptors across frames for temporal feature tracking (Wave 30, 2026-06-08: `DescriptorCache` in src/feature_cache.rs — coarse spatial-grid + FIFO ring over the last `max_frames` frames; `detect_and_compute_cached` reuses a recent keypoint's BRIEF descriptor when a fresh keypoint lands within `tol_px`, else recomputes via the exact uncached path so a MISS is BIT-IDENTICAL to `OrbDetector::detect_and_compute`. Factored a public `OrbDetector::detect_keypoints` (detection half of `detect_and_compute`) + `OrbDetector::brief()` accessor so the cached keypoint set and per-miss descriptor bytes are provably identical. Exposes `hit_ratio()`/`hits()`/`misses()`; default `max_frames=10`, `tol_px=1.5`. 11 tests (6 integration in tests/feature_cache_temporal.rs + 5 unit): same-frame re-query = 100% hits & bit-exact, slow 8 px pan = 0.998 reuse (≥80% asserted), 11-frame eviction with capacity respected, miss path bit-exact vs uncached. features.rs 1630→1668 L (under 2000); clippy --all-targets -D warnings clean; 786 crate tests pass)

## Testing
- [x] Add synthetic test: generate known homography, project points, recover and compare
  — `test_homography_roundtrip` in `spatial`: forward-inverse roundtrip + RANSAC recovery
  — Also fixed latent bug: `estimate_from_4_points` now uses A^T*A normal equations
    (9×9 SVD) instead of the broken thin-SVD path that panicked for 8×9 A matrices
- [ ] Test `audio_align` with known offsets using time-shifted synthetic audio
- [ ] Add accuracy tests for `distortion` using synthetic radially distorted checkerboard images
- [ ] Test `rolling_shutter` correction with synthetic rolling shutter simulation
- [x] Benchmark `features` (FAST+ORB) against varying image sizes (VGA to 4K) (implemented 2026-06-05, hardened 2026-06-06: tests/fast_orb_resolution.rs — public-API resolution sweep with a LOCALLY-UNIQUE synthetic texture (fixed 64px feature lattice, per-cell blob clusters seeded by an inline splitmix/LCG so corner density ∝ area AND BRIEF descriptors are discriminative — a purely periodic texture made every descriptor ambiguous and the ratio test returned 0 matches). 7 tests: keypoint_count_scales_with_resolution (strict-monotone VGA<HD<2K, none clamped, OrbDetector::new(50_000)), descriptor_count_matches_keypoint_count (VGA/HD/2K), matcher_recovers_known_translation (FeatureMatcher::new(50,0.8), 427 matches, median displacement == (12,8) exactly), self_match_is_identity (428/428 zero-distance & zero-displacement), fast_count_ge_orb_retained_count (FAST(25)=1080 ≥ ORB-retained=1057), timing_liveness_vga_hd_2k (Instant + eprintln, asserts only <60s liveness), timing_liveness_4k (#[ignore]d — slow in debug; carries the 2K→4K monotonicity + un-clamped 4K coverage). Full crate: 776 passed / 0 failed / 1 skipped, clippy --all-targets -D warnings clean. Also fixed a production robustness bug: OrbDetector::detect_and_compute now drops keypoints inside BRIEF's patch_size/2 border margin (a single edge corner used to make extract_batch fail the whole frame). features.rs split — unit tests moved to src/features_tests.rs via `#[cfg(test)] #[path] mod tests;` to stay under the 2000-line limit (1630 lines))
- [ ] Test `color` transfer with known color chart images across different illuminants

## Documentation
- [ ] Add coordinate system convention diagram (image vs camera vs world)
- [ ] Document multi-camera workflow: calibrate -> sync -> align -> composite
- [ ] Add accuracy/performance comparison table for different feature detectors
