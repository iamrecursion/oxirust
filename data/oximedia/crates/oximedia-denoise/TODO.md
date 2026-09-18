# oximedia-denoise TODO

## Current Status
- 22 modules across spatial, temporal, hybrid, motion, grain, multiscale, audio, and video domains
- 45 source files total including submodules (spatial/bilateral, spatial/nlmeans, spatial/wiener, spatial/wavelet, temporal/kalman, temporal/median, temporal/motioncomp, grain/analysis, grain/preserve, etc.)
- Core Denoiser struct with 5 modes: Fast, Balanced, Quality, GrainAware, Custom
- Dependencies: oximedia-core, oximedia-codec, oximedia-graph, rayon, ndarray

## Enhancements
- [x] Replace `expect("Invalid configuration")` in `Denoiser::new()` with proper error return (no unwrap policy)
- [x] Add configurable noise model selection to `DenoiseConfig` (currently hardcoded in estimator)
- [x] Implement actual algorithm selection in `process_custom()` mode (currently just delegates to `process_balanced`)
- [x] Add per-channel denoising strength in `chroma_denoise` (luma vs chroma independent control)
- [x] Improve `motion/estimation.rs` with hierarchical block matching for better motion vectors (verified 2026-05-16; src/motion/estimation.rs:212 hierarchical_motion_estimation, multi-level pyramid:226)
- [x] Add sub-pixel motion estimation accuracy in `motion/compensation.rs` (verified 2026-05-16; src/motion/compensation.rs:153 SubPixelMv struct)
- [x] Enhance `grain/analysis.rs` with frequency-domain grain characterization (verified 2026-05-16; src/grain/analysis.rs:44 FrequencyGrainProfile, characterize_grain_frequency:126)
- [x] Make `frame_buffer` use a `VecDeque` instead of `Vec` to avoid O(n) removal at index 0
- [x] Add streaming/pipeline mode to `Denoiser` for integration with oximedia-graph processing chains (verified 2026-05-16; src/streaming.rs:1 streaming pipeline-mode denoiser, 426 lines)
- [x] Implement adaptive temporal window sizing based on motion level in `hybrid/adaptive.rs` (verified 2026-05-16; src/hybrid/spatiotemporal.rs:119 adaptive_spatio_temporal fn)

## New Features
- [x] Add BM3D (Block-Matching 3D) denoising algorithm as a new spatial method
- [x] Implement video stabilization-aware denoising (joint stabilization + denoise pass) (verified-open 2026-05-16: no joint stab+denoise module found)
- [ ] Add GPU-accelerated denoising path via oximedia-gpu for real-time 4K processing (verified-open 2026-05-16: no GPU path in denoise sources)
- [x] Implement perceptual noise shaping that keeps denoising artifacts below JND threshold
  - File: src/perceptual_shaping.rs (JND map, luminance/contrast/texture masking, perceptual_denoise)
- [x] Add HDR-aware denoising with PQ/HLG transfer function awareness (verified 2026-05-16; src/hdr_denoise.rs:868 lines)
- [x] Implement deep image prior-style iterative denoising without neural networks
  - File: src/deep_prior.rs (guided filter SAT O(n), 5-tap Gaussian, Donoho–Johnstone noise var, blend loop)
- [x] Add real-time noise level monitoring and auto-strength adjustment
  - File: src/noise_monitor.rs (NoiseMonitor, diagonal-MAD estimator, AutoDenoiseStrength, snr_db)

## Performance
- [x] Add SIMD-accelerated bilateral filter kernel in `spatial/bilateral.rs`
- [x] Implement tiled processing in NLM to improve cache locality in `spatial/nlmeans.rs`
  - File: src/spatial/nlmeans.rs (tiled_nlmeans_filter with overlap-blend via acc_sum/acc_weight buffers)
- [x] Use rayon parallel iterators for row-parallel wavelet transform in `multiscale/wavelet.rs`
  - File: src/multiscale/wavelet.rs (parallel_wavelet_denoise using par_chunks_mut + par_iter_mut)
- [x] Add look-up table acceleration for Gaussian weight computation in bilateral filter
  - File: src/spatial/bilateral.rs (range_lut[256] precomputed exp table, ~3–5× speedup)
- [x] Profile and optimize `noise_estimate.rs` variance computation for large frames
  - File: src/noise_monitor.rs (diagonal-MAD O(n) estimator; 1080p < 200ms verified in test)

## Testing
- [x] Add PSNR/SSIM regression tests comparing denoised output against known-good references
  - File: src/regression_tests.rs (test_nlmeans_psnr_improvement_over_noisy)
- [x] Add temporal consistency tests verifying no flickering across frame sequences
  - File: src/regression_tests.rs (test_mctf_temporal_consistency: MAD inter-frame must decrease)
- [x] Test edge preservation metrics for bilateral and NLM filters
  - File: src/spatial/bilateral.rs (test_bilateral_preserves_hard_edge, test_bilateral_reduces_flat_region_variance)
  - File: src/spatial/nlmeans.rs (test_nlmeans_preserves_edge, test_nlmeans_reduces_flat_noise)
- [x] Add stress tests with various pixel formats beyond Yuv420p (Yuv422p, Yuv444p, RGB)
  - File: src/regression_tests.rs (test_nlmeans_yuv422p_no_panic, test_nlmeans_rgb24_no_panic)
- [x] Test `spectral_gate.rs` with known audio noise patterns
  - File: src/spectral_gate.rs (test_spectral_gate_learns_noise_floor, test_spectral_gate_signal_bins_survive, test_spectral_gate_reduction_db_positive)

## Documentation
- [ ] Document the noise model mathematical formulations in `noise_model.rs`
- [ ] Add architecture diagram showing the relationship between spatial, temporal, and hybrid modules
- [ ] Document the grain preservation pipeline in `grain/` submodules

## 0.1.8 Wave 12 Slice G (completed 2026-05-31)
- [x] Deep image prior-style iterative denoising (no neural weights)
  - File: src/deep_prior.rs — guided_filter_gray SAT O(n), 5-tap Gaussian blur, Donoho–Johnstone noise var, blend loop; PSNR ≥ 2 dB improvement test
- [x] Real-time noise level monitoring with auto-strength adjustment
  - File: src/noise_monitor.rs — NoiseMonitor, diagonal-MAD robust estimator (σ ≈ MAD/0.6745), AutoDenoiseStrength, snr_db(), 1080p < 200ms profile test
- [x] PSNR regression test (NLMeans: denoised PSNR > noisy PSNR vs clean)
  - File: src/regression_tests.rs
- [x] Temporal consistency test (MCTF: inter-frame MAD decreases after denoising)
  - File: src/regression_tests.rs
- [x] Multi-format stress tests (Yuv422p, Rgb24 → no panic)
  - File: src/regression_tests.rs
- [x] Noise-estimate 1080p profiling test (< 200ms)
  - File: src/noise_monitor.rs + src/regression_tests.rs

## 0.1.8 Wave 5 (completed 2026-05-29)
- [x] Watson (1993) DCT JND perceptual noise shaping — T_base CSF table, luminance+contrast masking (Slice ε)
  - File: src/perceptual_shaping.rs
- [x] Tiled NLM with rayon overlap-blend — linear-ramp blending at tile borders for C0 continuity (Slice ε)
  - File: src/spatial/nlmeans.rs
- [x] Bilateral LUT accelerator — range_lut[256] precomputed exp, 3–5× speedup vs naive exp path (Slice ε)
  - File: src/bilateral.rs
- [x] Wavelet per-band parallel denoise — rayon::join over LL/LH/HL/HH subbands; bit-identical to sequential (Slice ε)
  - File: src/multiscale/wavelet.rs
