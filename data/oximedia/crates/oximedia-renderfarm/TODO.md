# oximedia-renderfarm TODO

## Current Status
- 62 modules covering job management, worker pools, scheduling, cloud integration, cost tracking, tile rendering, and fault tolerance
- Key features: Coordinator, Scheduler, Worker management, Cloud bursting, Cost optimization, Multi-site support
- Dependencies: axum, tokio, rusqlite, prometheus, sysinfo, blake3, zstd, lz4

## Enhancements
- [x] Add GPU resource tracking to `node_capability` (VRAM, CUDA/Vulkan compute units, GPU temperature) (verified 2026-05-16; src/node_capability.rs:92 vram_total_mib, compute_units:109)
- [x] Extend `cost_optimizer` with spot instance pricing models and preemption handling for cloud workers (verified 2026-05-16; src/cost_optimizer.rs:18 AwsSpot/AzureSpot variants)
- [x] Add weighted fair-share scheduling to `scheduler` alongside existing algorithms (verified 2026-05-16; src/scheduler.rs:25 WeightedFairShare variant)
- [x] Implement render job dependency DAG validation in `job_dependency_graph` (cycle detection, unreachable node warnings) (verified 2026-05-16; src/job_dependency_graph.rs:61 acyclic flag, unreachable_nodes:68)
- [x] Add render progress ETA prediction in `progress` using historical completion rates per worker class (verified 2026-05-31: EtaPredictor+WorkerClass fully implemented in src/progress_eta.rs; per-class rate tracking confirmed)
- [x] Extend `elastic_scaling` with scale-down cooldown timers and min/max node constraints (verified 2026-05-16; src/elastic_scaling.rs:277 scale_down_cooldown_ms, min/max workers:551-562)
- [x] Add job preemption support in `render_job_queue` for higher-priority jobs arriving mid-render (verified 2026-05-16; src/render_job_queue.rs:40 is_preemptive, JobUrgency::Critical)
- [x] Implement chunk-level retry in `failure_recovery` instead of full-frame retry on transient errors (verified 2026-05-31: ChunkId+ChunkRetryState+ChunkRetryManager added; RecoveryAction::RetryChunk variant; 7 tests)

## New Features
- [x] Add a `render_template` module for reusable render configuration presets (resolution, codec, quality, frame range)
- [x] Add a `worker_benchmark` module to auto-profile worker performance and assign capability scores
- [x] Implement a `render_cache` module for caching intermediate render outputs (e.g., lighting passes) across jobs (verified 2026-05-16; src/render_cache.rs:295 lines)
- [x] Add an `alert_rule` module with configurable alert thresholds (queue depth, idle workers, budget overrun)
- [x] Implement a `resource_reservation` module for reserving worker capacity for scheduled high-priority jobs (verified 2026-05-16; src/resource_reservation.rs:334 lines)
- [x] Add a `render_artifact` module for managing output files (checksums, storage locations, lifecycle policies) (verified 2026-05-16; src/render_artifact.rs:379 lines)
- [x] Implement `job_template` inheritance so child jobs inherit parent settings with overrides (verified 2026-05-16; src/job_template.rs:216 job type inherited from template, 634 lines)

## Performance
- [ ] Add connection pooling for the `api` axum handlers to reduce per-request overhead
- [x] Implement batch insert for `render_log` entries instead of per-frame writes (2026-06-22: added `BatchedRenderLogWriter` in src/render_log.rs — in-memory buffer, configurable `batch_size` threshold auto-flush, explicit `flush`/`flush_and_sync`, flush-on-drop, single JSONL append write per batch instead of per-frame; durability trade-off documented; 13 new tests)
- [x] Add LRU eviction policy to `cache` module with configurable max memory usage (verified 2026-05-31: EvictionPolicy::LRU+max_size in src/cache.rs:32,123)
- [x] Profile and optimize `tile_rendering` merge step for large frame resolutions (8K+) (verified 2026-05-31: TileCompositor::merge with rayon par_chunks_mut parallelism; benches/tile_merge.rs)
- [x] Use `crossbeam-channel` bounded channels in `frame_distribution` to apply backpressure when workers are saturated (verified 2026-05-31: BackpressureDistributor in src/frame_distribution.rs; dispatch+try_dispatch; 5 tests)
- [x] Implement zero-copy frame data transfer in `frame_merge` using memory-mapped files (verified 2026-05-31: MmapFrameMerger in src/frame_merge.rs; anon mmap + Vec fallback; 4 tests)

## Testing
- [x] Add integration tests for `multi_site` failover scenarios (primary site down, secondary takeover) (verified 2026-05-31: test_multi_site_failover_and_rebalance in tests_wave13.rs)
- [x] Add load tests for `scheduler` with 1000+ concurrent jobs and 100+ workers (verified 2026-05-31: test_scheduler_load_1000_jobs_100_workers in tests_wave13.rs)
- [x] Test `elastic_scaling` scale-up/scale-down timing under variable load (verified 2026-05-31: test_elastic_scaling_timing_mock_clock in tests_wave13.rs)
- [x] Add property-based tests for `priority_queue` ordering guarantees (verified 2026-05-31: proptest_priority_queue_ordering in tests_wave13.rs)
- [x] Test `render_checkpoint` resume after simulated crash mid-frame (verified 2026-05-31: test_checkpoint_resume_partial_tiles in tests_wave13.rs)

## Documentation
- [ ] Add architecture diagram showing Coordinator -> Scheduler -> Worker data flow
- [ ] Document the job lifecycle states in `job` module (Submitted -> Queued -> Running -> Complete/Failed)
- [ ] Add examples for `cloud` module showing hybrid on-prem + cloud bursting configuration
- [ ] Document `tile_rendering` strategy selection criteria (frame size, worker count, network bandwidth)
