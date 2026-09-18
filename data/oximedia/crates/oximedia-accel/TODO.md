# oximedia-accel TODO

## Current Status
- 39 source files across modules: `vulkan`, `cpu_fallback`, `cpu_simd`, `device`, `buffer`, `cache`, `pool`, `dispatch`, `kernels`, `ops`, `shaders`, `traits`, `task_graph`, `task_scheduler`, `fence_timeline`, `memory_arena`, `memory_bandwidth`, `accel_profile`, `accel_stats`, `pipeline_accel`, `prefetch`, `workgroup`, `device_caps`, `subgroup`, `stress_tests`
- `HardwareAccel` trait with GPU (Vulkan) and CPU backends
- Operations: image scaling (bilinear AVX2 SIMD), color conversion (YUV→RGB SSE4.2), motion estimation
- Shader modules for scale/color/motion compute shaders
- Dependencies: vulkano 0.35, oximedia-core, rayon, bytemuck
- Metal backend: feature-gated (`metal-backend`) for macOS; MSL kernels for YUV↔RGB and bilinear scale
- 409 tests pass (355 original + 54 new in Wave 4)

## Enhancements
- [x] Add `deinterlace` operation to `HardwareAccel` trait (bob, weave, motion-adaptive)
- [x] Implement Lanczos and bicubic resampling in `ops/scale` alongside bilinear
- [x] Add YUV<->RGB color space support to `ops/color` (currently RGB-centric)
- [x] Implement HDR tone mapping operations (PQ to SDR, HLG to SDR) in `ops/color`
- [x] Add multi-GPU support: distribute work across multiple Vulkan devices in `dispatch` (`multi_gpu.rs`, 7 tests)
- [x] Implement async compute queue submission in `vulkan` for overlapped CPU/GPU work (`async_compute.rs`)
- [x] Add pipeline caching (VkPipelineCache) in `pipeline_accel` to speed up shader compilation (`pipeline_cache.rs` with LRU and export/import)
- [x] Implement double/triple buffering strategy in `buffer` for streaming frame processing (`buffer_ring.rs` — Single/Double/Triple/Custom modes + StagingRing)
- [x] Add GPU memory pressure monitoring in `memory_arena` with automatic eviction
- [x] Implement `workgroup` auto-tuning based on `device_caps` (optimal local size per device)

## New Features
- [x] Add Metal backend for macOS/iOS acceleration (`metal_backend.rs`, feature-gated `metal-backend`; MSL YUV↔RGB + bilinear scale kernels; motion estimation CPU fallback planned)
- [x] Implement WebGPU backend for WASM target compatibility (still stub — `webgpu_backend.rs`)
- [x] Add GPU-accelerated histogram computation for color grading pipelines
- [x] Implement GPU-based 2D convolution for filter kernels (blur, sharpen, edge detect)
- [x] Add GPU-accelerated alpha blending and compositing operations
- [x] Implement GPU-based image rotation and affine transform
- [x] Add compute shader for DCT/IDCT to accelerate codec operations
- [x] Implement GPU-accelerated noise reduction (temporal and spatial NR) — 2026-05-13 Wave 4 Slice 5: WGSL bilateral filter with 32×32 shared-memory tiling (16-thread workgroup + 8-px halo, `workgroupBarrier` cooperative load) in `shaders/bilateral.rs` + dispatcher `bilateral_gpu.rs`; WGSL motion-adaptive temporal NR with up to 2 prev frames in `shaders/temporal_nr.rs` + ring-buffer dispatcher `temporal_nr.rs` (VecDeque bounded capacity, persistent zero-buffer placeholder for unused slots); CPU references + GPU-vs-CPU parity tests gated behind `webgpu` feature
- [x] Add profiling/timing overlay for measuring per-operation GPU times in `accel_stats`

## Performance
- [x] Implement descriptor set pooling in Vulkan backend to reduce allocation overhead
- [x] Add staging buffer ring for efficient CPU-to-GPU transfers in `buffer` (`buffer_ring.rs` — StagingRing)
- [x] Profile and optimize `cpu_fallback` paths with SIMD intrinsics (SSE4.2/AVX2/NEON) — `cpu_simd.rs`: AVX2 bilinear scale (8-pixel batches, prefetch hints), SSE4.2 YUV→RGB (4-pixel batches, fixed-point BT.601), runtime dispatch via `is_x86_feature_detected!`
- [x] Implement subgroup operations in compute shaders where supported (Vulkan 1.1+) — `subgroup.rs`: `SubgroupCapabilities`, `SubgroupOperations`, `SubgroupStages` with Vulkan 1.1 baseline; GLSL reduce-sum and ballot-compaction snippets; `recommend_subgroup_size()` API
- [x] Add shared memory tiling in scale/color shaders for reduced global memory bandwidth (verified 2026-05-16; src/shaders/scale.rs + src/shaders/color.rs — 18×18 bilinear tile with 1-pixel halo in scale, 16×16 Y/RGB tiles with barrier() in both color shaders; 15 structural tests added)
- [x] Benchmark and tune `prefetch` hints for different GPU architectures (AMD/NVIDIA/Intel) — `_mm_prefetch(_MM_HINT_T0)` added in AVX2 bilinear and SSE4.2 YUV→RGB inner loops with `// PERF:` comments

## Testing
- [x] Add visual regression tests: GPU vs CPU output comparison within tolerance — `cpu_simd::tests::visual_regression_scale_dispatch_vs_scalar` and `visual_regression_yuv_dispatch_vs_scalar` (scalar vs SIMD dispatch, max diff ≤ 1 pixel)
- [x] Test device selection fallback when no Vulkan device is available — `stress_tests::device_fallback_cpu_only_is_available`, `device_fallback_new_does_not_panic`, `device_fallback_cpu_can_scale`, `device_fallback_cpu_can_convert_color`, `device_fallback_cpu_motion_estimation`
- [x] Add stress test for concurrent `scale_image` + `convert_color` operations — `stress_tests::concurrent_scale_and_convert_no_panic` (8 threads), `concurrent_large_scale_no_panic` (4 threads, determinism check)
- [x] Test `memory_arena` allocation/deallocation patterns for leak detection — `stress_tests::memory_arena_concurrent_alloc_reset_no_panic` (4 threads × 100 allocs via Mutex), `memory_arena_alloc_reset_no_leak` (50 cycles), `memory_arena_stats_peak_correct`, `memory_arena_many_small_allocs`
- [x] Benchmark `task_graph` scheduling overhead with large dependency chains — `task_graph::task_graph_100_node_chain_schedules_quickly` (#[ignore] slow), `task_graph_diamond_dag_correct_order`, `task_graph_50_node_fan_out_in`, `task_graph_cycle_detection_returns_error`, `task_graph_single_task`, `task_graph_two_task_dependency_order`
- [x] Test `fence_timeline` synchronization correctness under high contention — `fence_timeline::fence_pool_concurrent_acquire_release_via_mutex` (4 threads × 25 cycles via Mutex), `timeline_registry_concurrent_timeline_updates_via_mutex`, plus 8 additional signal/monotone/history/reset tests

## Documentation
- [x] Document device selection heuristics and how to override in `DeviceSelector` (planned 2026-05-14)
  - **Goal:** rustdoc on `DeviceSelector` (`src/device.rs` L29) lists the exact scoring rules (preference score + universal +100 discrete bonus) and shows code examples for overriding via `with_preference` / `with_min_api_version`.
  - **Design:** Module-level rustdoc table mapping each `DevicePreference` variant to its score function. Inline code block showing `DeviceSelector::new().with_preference(DevicePreference::MostMemory).select()?`. Cross-reference to `score_device` (private but the heuristic is now documented in prose).
  - **Files:** `crates/oximedia-accel/src/device.rs` (rustdoc only — no behavior changes).
  - **Tests:** `cargo doc -p oximedia-accel --no-deps` must succeed with zero broken_intra_doc_links warnings.
  - **Risk:** None — rustdoc-only change.
- [x] Add architecture diagram showing data flow: CPU buffer → staging → GPU → staging → CPU (planned 2026-05-14)
  - **Goal:** ASCII diagram in `crates/oximedia-accel/src/lib.rs` module rustdoc showing canonical GPU data flow for discrete (PCIe staging required) and integrated (UMA direct-map) paths.
  - **Design:** Two diagrams in text rustdoc blocks; prose explaining each path.
  - **Files:** `crates/oximedia-accel/src/lib.rs` (top-of-file `//!` rustdoc).
  - **Tests:** `cargo doc -p oximedia-accel --no-deps` clean.
  - **Risk:** None.
- [x] Document supported Vulkan extensions and minimum device requirements (planned 2026-05-14)
  - **Goal:** Rustdoc table in `src/device.rs` listing required Vulkan API version (≥ V1_2), required device extensions, and recommended optional extensions. Sourced from actual `DeviceExtensions` in `device.rs` and subgroup logic in `subgroup.rs`.
  - **Design:** Two tables: Required and Optional/Recommended. Cross-reference `SubgroupCapabilities` in `subgroup.rs`.
  - **Files:** `crates/oximedia-accel/src/device.rs` + paragraph in `src/lib.rs`.
  - **Tests:** `cargo doc -p oximedia-accel --no-deps` clean.
  - **Risk:** None.
