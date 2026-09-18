# oximedia-quality TODO

## Current Status
- 36 modules for video quality assessment and objective metrics
- Full-reference metrics: PSNR, SSIM, MS-SSIM, VMAF, VIF, FSIM
- No-reference metrics: NIQE, BRISQUE, Blockiness, Blur, Noise
- Advanced modules: aggregate_score, artifact_score, bitrate_quality, codec_quality, color_fidelity, compression_artifacts, histogram_quality, metrics, perceptual, perceptual_model, quality_gate, quality_preset, quality_report, reference_free, scene_quality, sharpness_score, spatial_quality, temporal_quality, vmaf_score
- Key types: QualityAssessor, Frame, MetricType, QualityScore, PoolingMethod, BatchAssessment
- Dependencies: oximedia-core, ndarray, rayon, serde

## Enhancements
- [x] Replace `unreachable!()` in `QualityAssessor::assess_no_reference` with proper error return (verified 2026-05-16; src/lib.rs:450 returns Err(OxiError::InvalidData) for non-NR metrics)
- [x] Add per-region quality assessment in SSIM/PSNR (compute metrics for specific frame regions)
- [x] Implement configurable VMAF model selection in `VmafCalculator` (phone model, 4K model) (verified 2026-05-16; src/vmaf_like.rs:50 VmafModel enum Phone/Hdtv/FourK/Custom, VmafModelWeights:63)
- [x] Extend `temporal_quality` with scene-aware temporal pooling (reset stats at scene cuts) (TemporalPsnrAccumulator + SceneAwareQualityAccumulator done)
- [x] Add chroma plane quality assessment in SSIM (currently luma-only in many implementations) (verified 2026-05-16; src/ssim.rs:34 chroma_weight, cb/cr SSIM computation:115-140)
- [x] Implement quality metric confidence intervals based on frame count (verified 2026-05-16; src/confidence.rs:81 ConfidenceInterval, ConfidenceCalculator:161)
- [x] Extend `quality_gate` with multi-metric composite gates (pass only if SSIM > X AND VMAF > Y)
- [x] Add `quality_report` export to CSV format for spreadsheet analysis

## New Features
- [x] Implement LPIPS (Learned Perceptual Image Patch Similarity) metric using pre-trained weights (verified 2026-05-16; src/lpips.rs:128 LpipsCalculator, LpipsConfig:30, LpipsResult:88)
- [x] Add video quality assessment for HDR content (PQ/HLG-aware metrics) (verified 2026-05-16; src/hdr_quality.rs:104 HdrQualityAssessor, analyze_frame:174)
- [x] Implement real-time quality monitoring (running average with configurable window) (verified 2026-05-16; src/realtime_monitor.rs:29 MonitorConfig sliding window, RealtimeMonitor:600 lines)
- [x] Add A/B quality comparison tool — rank multiple encodes by perceptual quality (verified 2026-05-16; src/ab_compare.rs:121 ranked CandidateMetrics, AbComparator)
- [x] Implement quality-bitrate curve generation (encode at multiple CRF values, plot VMAF vs bitrate) (verified 2026-05-16; src/quality_bitrate_curve.rs:150 QualityBitrateCurve, QualityBitrateCurveBuilder:481)
- [x] Add motion-compensated temporal quality analysis (account for frame motion in temporal metrics) (verified 2026-05-16; src/motion_compensated.rs:117 MotionCompensatedResult, MotionCompensatedAnalyzer:71)
- [x] Implement CIEDE2000 color difference metric for color fidelity assessment (verified 2026-05-16; src/ciede2000.rs:105 Ciede2000Calculator, delta_e_2000:130)
- [x] Add quality heatmap generation — spatial map of per-pixel quality for visualization (verified 2026-05-16; src/quality_heatmap.rs:102 QualityHeatmap, HeatmapGenerator:162)

## Performance
- [x] Implement SIMD-optimized SSIM computation using portable_simd or manual intrinsics (verified 2026-05-16; src/ssim_simd.rs:70 SsimSIMD SIMD_LANES=8 f32 chunks, portable_simd-style accumulation)
- [x] Add SSIM with subsampling for fast large-frame quality assessment (ssim_subsampled done)
- [x] Implement VMAF feature extraction module (VmafFeatures done)
- [ ] Add GPU-accelerated PSNR/SSIM computation via compute shaders (verified-open 2026-05-16: no wgpu/compute shader in quality crate)
- [x] Parallelize `BatchAssessment` across frames using rayon with configurable thread count (verified 2026-05-16; src/batch.rs:228 par_iter()+zip par_iter())
- [x] Optimize `VifCalculator` steerable pyramid decomposition with FFT-based filtering (done 2026-06-01: vif_pyramid.rs PyramidVifCalculator with gaussian_bandpass_filter via oxifft::fft2d/ifft2d; separable_gaussian_conv for local stats; GSM VIF formula faithful to Sheikh-Bovik 2006)
- [x] Cache Gaussian kernel weights in SSIM calculator to avoid recomputation per frame (verified 2026-05-16; src/ssim_simd.rs:366 test_gaussian_kernel_cache_sum_to_one, kernel cached at construction)
- [x] Use pre-allocated buffers in `Frame` operations to reduce allocation in tight loops (done 2026-06-01: frame_pool.rs FramePool with acquire/release; zero-copy resize; exported as oximedia_quality::FramePool)

## Testing
- [x] Add golden reference tests — compare metric outputs against known-correct values from published papers
- [x] Test metric monotonicity — verify PSNR/SSIM increase as distortion decreases
- [x] Add tests for 10-bit and 12-bit Frame data (HDR content)
- [x] Test `PoolingMethod` with edge cases (single frame, all-identical scores, NaN values)
- [x] Benchmark quality metrics against reference implementations (compare speed and accuracy) — speed: `benches/metrics_bench.rs` criterion groups `bench_psnr`/`bench_ssim`/`bench_vmaf` over [640×360, 1280×720] (gradient ref vs +5-luma distorted); accuracy: already covered by `src/golden_tests.rs` (20+ analytic known-answer tests incl. the all-plane-Δ exact `20·log10(255/8)` identity) — 0.1.9 Wave 27

## Documentation
- [ ] Add metric interpretation guide — what PSNR/SSIM/VMAF scores mean in practice
- [ ] Document quality assessment workflow from raw frames to aggregated report
- [ ] Add visual examples showing metric behavior on different distortion types (blur, blocking, noise)
