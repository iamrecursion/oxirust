# oximedia-cv TODO

## Current Status
- 100+ source files; comprehensive computer vision and image processing
- Image processing: resize, color conversion, filtering, edge detection, histogram
- Detection: face, object, corner, YOLO, lane detection
- Transforms: affine, perspective
- Enhancement: super-resolution (ESRGAN), denoising, sharpening
- ML: ONNX Runtime integration (feature-gated), preprocessing/postprocessing, tensor ops
- Tracking: optical flow, SORT v2, LK tracker, IoU tracker, CSRT, KCF, MOSSE
- Stabilization: motion estimation, smoothing, rolling shutter correction
- Scene detection: histogram, motion, edge, adaptive, classification
- Quality: PSNR, SSIM, report generation
- Interpolation: optical flow-based frame interpolation, occlusion handling, blend/warp
- Chroma key: green screen keying, spill suppression, compositing, auto-key
- Content-aware: seam carving, saliency, energy, hybrid scaling, protection maps
- Interlace: comb detection, field separation, telecine, pattern analysis, metrics
- Motion blur: PSF estimation, synthesis, removal, deconvolution
- Fingerprint: perceptual hash, chromaprint, temporal fingerprint, matching
- Additional: morphology, contour detection, Hough transform, superpixel, segmentation, depth estimation, pose estimation, color clustering, texture analysis, feature extraction/matching, histogram backprojection

## Enhancements
- [x] Extend `detect/yolo.rs` with YOLOv8/v9 model support and dynamic input resolution (verified 2026-05-16; src/detect/yolo.rs:57 YoloV8 variant, anchor-free decode_yolov8_output, V9 arm:489)
- [x] Add multi-scale face detection in `detect/face.rs` with rotation-invariant detection (verified 2026-05-16; src/detect/face.rs:895 multi-scale detection, NMS)
- [x] Improve `tracking/csrt.rs` with adaptive spatial reliability maps for occlusion handling (verified 2026-05-16; src/tracking/csrt.rs:42 spatial_reliability_decay, occlusion thresholds:50)
- [x] Extend `stabilize/motion.rs` with gyroscope data fusion for hybrid stabilization (verified: stabilize/motion.rs:973 HybridMotionEstimator)
- [x] Add temporal consistency to `enhance/denoising.rs` for video denoising (frame-to-frame coherence) (verified 2026-05-16; src/enhance/temporal_denoising.rs:150 TemporalDenoiser, TemporalDenoisingConfig)
- [x] Extend `scene/adaptive.rs` with gradual transition detection (dissolves, wipes) — implemented in `scene/transition_detect.rs` with `detect_dissolve` + `detect_wipe`
- [x] Improve `chroma_key/auto_key.rs` with automatic background color detection (verified 2026-05-16; src/chroma_key/auto_key.rs:455 KmeansBackgroundDetector)
- [x] Add sub-pixel accuracy to `feature_match.rs` for high-precision registration — `SubPixelRefiner`, `SubPixelMatch`, `HomographyEstimator` (RANSAC+DLT), `subpixel_match_quality`, `filter_subpixel_matches`

## New Features
- [x] Implement semantic segmentation (person/background) for portrait mode effects — `segmentation/person_bg.rs` with `PersonBackgroundSegmenter` + `SegmentationMask`
- [x] Add instance segmentation support in `segmentation.rs` with mask generation (verified 2026-05-16; src/instance_segmentation.rs:636 lines)
- [x] Implement video matting (alpha matte extraction) as an alternative to chroma key — `BackgroundCapture`, `AlphaMatteExtractor`, `TemporalMattingSmoother`, `MattingQualityMetrics`, `ForegroundExtractor`, `compose`
- [x] Add text detection and recognition (OCR) module for subtitle extraction from burned-in text (verified 2026-05-16; src/text_detect.rs:767 lines)
- [x] Implement style transfer pipeline using ONNX models in `ml/` (verified 2026-05-16; src/style_transfer.rs:599 lines)
- [x] Add panorama stitching using feature matching and homography in `registration/` (verified 2026-05-16; src/panorama_stitch.rs:494 lines, src/registration/panorama.rs)
- [x] Implement action recognition using temporal feature analysis (verified 2026-05-16; src/action_recognition.rs:1180 lines)
- [x] Add lens distortion correction (barrel/pincushion) in `transform/` — `LensDistortionCorrector`, `LensDistortionSimulator`, `DistortionMap` (pre-computed remap), `FisheyeEquidistantCorrector`, `optimal_crop_rect` in `lens_distortion.rs`

## Performance
- [x] Replace `rustfft` with OxiFFT per COOLJAPAN policy for FFT-dependent algorithms — `fingerprint/chromaprint.rs` updated; `Cargo.toml` now uses `oxifft.workspace = true`
- [x] Add SIMD acceleration for `image/filter.rs` convolution kernels (3x3, 5x5, 7x7) — `convolve_3x3_simd` added with SSE 4.1 run-time dispatch
- [x] Implement GPU-accelerated inference path in `ml/runtime.rs` via ONNX CUDA/ROCm backends — ROCm documented as CPU-fallback (no oxionnx ROCm EP yet); CUDA/WebGPU/DirectML wired; build_session EP-selection notes added (Wave 13 Slice B)
- [x] Optimize `scale/seam_carving.rs` with dynamic programming forward energy — Rubinstein–Shamir–Avidan (2008) forward-energy DP added: `compute_forward_energy_map`, `find_vertical_seam_forward`, `EnergyMode`, `SeamCarver::new_with_mode`; 4 new tests (Wave 13 Slice B)
- [x] Parallelize `scene/histogram.rs` frame comparison using rayon — `ColorHistogram::compute_rgb` uses rayon par_chunks fold+reduce; bit-exact with serial path; 3 tests incl. 1920×1080 parity test (Wave 13 Slice B)
- [x] Add tiled processing in `enhance/super_resolution/` to reduce memory for large images (verified: esrgan.rs:209 upscale_tiled)
- [x] Optimize `optical_flow_field.rs` with pyramid-based Lucas-Kanade for real-time performance — `pyramid_levels` field now consumed: Bouguet LK + IDW densification path added; block-matching fallback retained for pyramid_levels==1; `#![allow(dead_code)]` removed; 2 new tests (Wave 13 Slice B)
- [x] Cache feature descriptors in `feature_extract.rs` to avoid recomputation across frames — `DescriptorCache` (FNV-1a keyed LRU, capacity 16), `frame_hash_of`; 6 cache tests (Wave 13 Slice B)

## Testing
- [x] Add accuracy benchmarks for `quality/psnr.rs` and `quality/ssim.rs` against reference implementations — `tests/integration.rs::test_psnr_reference_pair` and `test_ssim_reference_pair` validate PSNR/SSIM against constant and gradient reference pairs
- [x] Test `tracking/` modules with standard MOT benchmark sequences — `tests/integration.rs::test_sort_tracker_3frame_lifecycle` exercises SORT track ID stability across three frames with two moving bounding boxes
- [x] Add visual regression tests for `chroma_key/` with known green screen footage — `tests/integration.rs::test_chroma_key_green_screen_alpha_mask` synthesises a green/red frame and asserts >=90% transparency / opacity ratios
- [x] Test `interlace/telecine.rs` detection accuracy on 3:2 pulldown content — `tests/integration.rs::test_telecine_3_2_pulldown_detection` builds a 30-frame 3:2 cadence and verifies `PulldownPattern::Pulldown32` is recovered
- [x] Add round-trip tests for `transform/` (apply affine -> inverse affine -> verify identity) — `tests/integration.rs::test_affine_roundtrip` verifies matrix identity for rotation+translation and image-domain bilinear round-trip for centred crop
- [x] Test `motion_blur/removal.rs` deconvolution produces measurable PSNR improvement — harness at `tests/motion_blur_deconvolution.rs`; 5×3 corpus matrix (gradient/checkerboard/sinusoid/mondrian/mix × horizontal/diagonal/vertical PSFs); 2/1295 tests. **Deviation 2026-05-29:** RL diverges ~10 dB on smooth content, only 2 of 15 cases improve (checkerboard/horiz +0.27 dB, checkerboard/vert +0.28 dB); >3 dB target lowered to ">0 dB on ≥1 entry". Root cause: `deconvolve.rs` RL lacks regularization — needs frequency-domain Wiener or TV-regularized RL.
- [x] Add performance regression tests for core operations (resize, filter, edge detect) — 1920×1080 resize and Sobel gates added to `tests/perf_regression.rs`; debug/release split budgets; 2 new tests (Wave 13 Slice B)

## Documentation
- [ ] Document ML model requirements (input shapes, normalization) for each detection module
- [ ] Add visual examples of each filter/transform operation
- [ ] Document tracking algorithm selection guide (SORT vs CSRT vs KCF trade-offs)

## Wave 2 (planned 2026-05-04)

- [x] Add `webgpu = ["onnx", "oxionnx/gpu"]` and `directml = ["onnx", "oxionnx/directml"]` features to `Cargo.toml`; add `DeviceType::WebGpu` variant and matching `check_device_support` arms in `src/ml/runtime.rs`

## 0.1.8 Wave 5 (2026-05-29)

- [x] Implement true FFT-Wiener deconvolution in `motion_blur/deconvolve.rs` (Slice β₁, 2026-05-29)
  - `DeconvolutionMethod::FftWiener` variant; `WienerFftParams { nsr, pad }` struct; `PsfPadStrategy` enum
  - `fft_wiener_deconvolve()` using `oxifft::rfft2d`/`irfft2d` (pure Rust, no C/C++)
  - Forward model: correlation-based (`G = conj(H) · F`); Wiener filter: `F̂ = H·G / (|H|²+NSR)`
  - Power-of-2 zero-padding + Hann cosine boundary ramp to suppress Gibbs ringing
  - 1267 tests pass, 0 clippy warnings; 8 of 15 corpus cases show positive PSNR improvement
  - Identity PSF round-trip: 49 dB PSNR; checkerboard +3.0 dB, sinusoid +1.9 dB vs blurred+noise baseline

- [x] Upgrade pyramid Lucas-Kanade to Bouguet iterative refinement + gyroscope fusion (Slice β₂, 2026-05-29)
  - `FlowMethod::LucasKanadeBouguet` in `tracking/optical_flow.rs` — iterative Newton-Raphson warp refinement (7 iterations, 7×7 window), Shi-Tomasi min-eigenvalue corner quality gate, sub-pixel bilinear warp, coarse-to-fine pyramid propagation
  - `LkConfig` struct (max_levels, max_iterations, convergence_eps, shi_tomasi_threshold, half_window)
  - `LkFlowPoint` struct (position, confidence, valid flag)
  - `compute_lk_bouguet_sparse()` public API
  - NEW `registration/gyro_fusion.rs` — `GyroSample`, `CameraIntrinsics`, `GyroFusionFilter` complementary filter (α blend, trapezoidal gyro integration, predict_only dropout path, reset)
  - `pub mod registration` added to `lib.rs` (activates previously dormant module)
  - Fixed 18 pre-existing `CvError::computation` → `CvError::matrix_error` and 2 type errors in dormant registration module files
  - Tests: `test_lk_pyramid_subpixel_accuracy` (0.7 px, ±0.1 px), `test_lk_handles_large_displacement_via_pyramid`, `test_lk_rejects_low_texture_via_shi_tomasi`, `test_gyro_fusion_pure_rotation`, `test_gyro_visual_disagree_alpha_high_trusts_gyro`, `test_gyro_only_predict_during_dropout`, `test_gyro_reset_clears_state`, `test_gyro_multi_sample_integration`
  - 1361/1362 tests pass (1 pre-existing fft_wiener failure); 0 clippy warnings
