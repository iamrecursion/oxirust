# oximedia-optimize TODO

## Current Status
- 43 source files/directories covering codec optimization and encoding tuning
- Core: Optimizer with RdoEngine, PsychoAnalyzer, MotionOptimizer, AqEngine
- RDO: rate-distortion optimization, lambda calculation, RDOQ (rate-distortion optimized quantization)
- Psychovisual: contrast sensitivity, visual masking models, perceptual optimization
- Motion: TZSearch, EPZS, UMH algorithms, bidirectional optimization, subpel refinement, MV prediction
- Encoding tools: partition selection, intra mode optimization, transform selection, loop filter tuning
- Presets: Fast/Medium/Slow/Placebo levels, content-type adaptation (Animation/Film/Screen/Generic)
- Advanced: lookahead analysis, GOP optimizer, two-pass encoding, CRF sweep, adaptive ladder, bitrate control

## Enhancements
- [x] Implement vmaf_predict module -- currently a file exists but needs actual VMAF score prediction from features (verified 2026-05-16; src/vmaf_predict.rs:491 fn predict, :523 predict_from_stats, :532 predict_from_pixels, 1249 total lines)
- [x] Wire roi_encode into the main Optimizer pipeline for region-of-interest based quality allocation (verified 2026-05-16; src/roi_encode.rs:756 lines, no stubs)
- [x] Connect temporal_aq to the AqEngine for frame-level QP adaptation based on temporal complexity (verified 2026-05-16; src/temporal_aq.rs:674 lines)
- [x] Improve scene_encode to use lookahead data for scene-cut-aware QP adjustment (completed 2026-05-29: scene_cut_aware_qp_curve in lookahead.rs)
- [x] Add content-adaptive GOP structure selection in gop_optimizer (longer GOPs for static, shorter for action) (verified 2026-05-16; src/gop_optimizer.rs:237 fn content_type_base_rules, ContentAdaptiveGop)
- [x] Implement actual CABAC context optimization in entropy module (currently may be structural) (verified 2026-05-16; src/entropy/cabac.rs:192 CabacContext with LPS/MPS probability update)
- [x] Extend crf_sweep to output Pareto-optimal bitrate/quality curves for automated quality targeting (verified 2026-05-16; src/crf_sweep.rs:604 lines)
- [x] Add grain synthesis detection in psycho module to preserve film grain without wasting bits (verified 2026-05-16; src/psycho/grain.rs:1140 lines)

## New Features
- [x] Implement AV1 tile/frame parallel optimization -- select tile partitioning based on content complexity
- [x] Add per-frame bitrate allocation using Viterbi algorithm for optimal constant-quality distribution
- [x] Implement machine-learning-based mode decision using oximedia-neural for fast RDO approximation (completed 2026-05-31 Wave 13: MlModeDecider 3-layer MLP in decision.rs, feature-gated ml-decision, PipelineModeDecider pipeline integration, shortlist_modes invariant tests)
- [x] Add encoding quality estimation without full encode (fast VMAF/SSIM prediction from frame features) (verified 2026-05-16; src/vmaf_predict.rs:5 fast quality score estimator without full VMAF, predict fn:391)
- [x] Implement denoising-aware optimization -- coordinate with pre-filter to avoid encoding noise
- [x] Add HDR-aware psychovisual model using PQ/HLG transfer function aware masking thresholds (completed 2026-05-29: hdr_masking.rs + compute_hdr_qp in perceptual_optimization.rs)
- [x] Implement multi-pass encoding with rate redistribution between passes in two_pass module (verified 2026-05-16; src/multi_pass.rs:520 RateRedistributor, redistribute:569)

## Performance
- [x] Use rayon for parallel RDO evaluation across partition candidates (when parallel_rdo=true) (PartitionRdo trait done)
- [x] Implement SIMD-accelerated SAD/SATD computation in motion search (completed 2026-05-29: satd_8x8, satd_16x16, sad_block_avx2 in simd_metrics.rs)
- [x] Add block-level caching in RdoEngine -- cache cost estimates for repeated partition evaluations (CachedRdoEngine done)
- [x] Implement early termination in partition search when cost clearly exceeds parent split (rdo_with_early_termination done)
- [x] Add complexity-based QP adjustment per frame (estimate_frame_complexity done)
- [x] Profile cache_optimizer and prefetch modules -- ensure they actually improve memory access patterns (completed 2026-05-31 Wave 13: benches/cache_prefetch.rs criterion benchmarks; test_prefetch_reduces_simulated_misses passes)
- [x] Use thread-local storage for per-thread scratch buffers in parallel motion search (completed 2026-05-31 Wave 13: MOTION_SCRATCH thread_local! in motion/search.rs, parallel_motion_search uses padded reference + sad_with_mv_ref)

## Testing
- [x] Add RDO cost monotonicity test: higher lambda should prefer lower-rate modes (completed 2026-05-31 Wave 13: test_rdo_cost_monotonicity_lambda in wave13_tests.rs)
- [x] Test motion search accuracy: known displacement input should find correct motion vector (completed 2026-05-31 Wave 13: test_motion_search_exact_displacement, test_parallel_full_search_exact_shift, test_parallel_search_with_noise)
- [x] Verify AQ produces higher QP for flat regions and lower QP for detailed regions (completed 2026-05-31 Wave 13: test_aq_flat_vs_detailed in wave13_tests.rs)
- [x] Test partition decision against brute-force: verify faster presets don't miss optimal split by >5% RD cost (completed 2026-05-31 Wave 13: test_partition_vs_brute_force_gap in wave13_tests.rs)
- [x] Add benchmark comparing Fast/Medium/Slow/Placebo encoding speed and quality metrics (completed 2026-05-31 Wave 13: benches/presets.rs with preset_construction, preset_aq_64frames, preset_quality_proxy)
- [x] Test psychovisual masking: verify high-texture regions tolerate more quantization noise (completed 2026-05-31 Wave 13: test_psychovisual_masking_texture_tolerance in wave13_tests.rs)

## Documentation
- [ ] Document the optimization level trade-offs with encoding speed and quality impact measurements
- [ ] Add guide for tuning custom OptimizerConfig for specific content types
- [ ] Document the RDO pipeline with mathematical formulation of cost = distortion + lambda * rate

## 0.1.8 Wave 13 (completed 2026-05-31)

- [x] Neural-assisted RDO mode decision (ml-decision feature gate)
  - `MlModeDecider`: 3-layer MLP (6→32→16→n_modes), deterministic LCG weights, ReLU + softmax
  - `extract_features`: variance, mean_abs_grad, edge_energy, mean_sad_to_dc, neighbour_depth_avg, QP
  - `shortlist_modes`: top-k shortlisting with critical invariant: exhaustive best always in shortlist
  - `PipelineModeDecider`: pipeline integration; uses ML shortlister when feature is on
  - Feature: `ml-decision = ["oximedia-neural"]` in Cargo.toml
  - Tests: shortlist invariant on flat/edge/texture blocks; softmax sum; feature range; size clamp

- [x] Thread-local motion scratch (parallel_motion_search)
  - `MOTION_SCRATCH: RefCell<MotionScratch>` const thread_local with sad/cost/candidate buffers
  - Padded reference API: `ref_size = block_size + 2*search_range` eliminates OOB returns
  - `sad_with_mv_ref`: separate src/ref dimensions for padded-frame SAD
  - Tests: exact shift, noisy shift (±2 tolerance), multi-block thread-local reuse

- [x] Cache/prefetch benchmarks (`benches/cache_prefetch.rs`)
  - Cold scan 1000 blocks, warm scan 1000 blocks, prefetch add 200, top-N priority, evict 500 stale
  - `test_prefetch_reduces_simulated_misses`: sliding-window simulation; with_pf < without

- [x] Preset benchmarks (`benches/presets.rs`)
  - preset_construction (Optimizer::new per level), preset_aq_64frames, preset_quality_proxy (VMAF)

- [x] Five integration tests in `wave13_tests.rs`:
  - (a) `test_rdo_cost_monotonicity_lambda`: small λ → mode B wins; large λ → mode A wins
  - (b) `test_motion_search_exact_displacement`: full parallel search finds exact integer shift
  - (b2) `test_motion_search_noisy_within_tolerance`: noisy block MV within ±2
  - (c) `test_aq_flat_vs_detailed`: flat block gets HIGHER QP offset than detailed
  - (d) `test_partition_vs_brute_force_gap`: ≤5% RD cost gap vs exhaustive for Fast/Medium/Slow
  - (e) `test_psychovisual_masking_texture_tolerance`: high-texture σ²>1000 gets higher JND

## 0.1.8 Wave 5 (completed 2026-05-29)

- [x] SIMD SAD/SATD 8x8/16x16 + HDR PQ/HLG psychovisual masking + scene-cut-aware QP lookahead
  - satd_8x8 / satd_16x16 (separable 8-point Hadamard butterfly, scalar + platform dispatch)
  - sad_block_avx2 (variable-block, _mm256_sad_epu8, x86_64-gated)
  - hdr_masking.rs (new): TransferFunction, HdrMaskingConfig, hdr_qp_offset, hdr_qp_map (PQ/HLG/SDR)
  - PsychovisualQp::compute_hdr_qp composing psychovisual + HDR deltas
  - scene_cut_aware_qp_curve in lookahead.rs (SAD-based cut detection, pre-cut ramp +3, post-cut +2/+1)
