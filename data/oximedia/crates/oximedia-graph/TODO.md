# oximedia-graph TODO

## Current Status
- 30+ modules implementing a filter graph pipeline for media processing
- Core: graph builder, nodes, ports, connections, frame pool
- Filters: video (passthrough, null sink) and audio filters
- Advanced: topological sort, cycle detection, graph merge, partitioning, subgraphs
- Scheduling, profiling, serialization, visualization, optimization
- Dependencies: oximedia-core, oximedia-codec, oximedia-audio, fontdue

## Enhancements
- [x] Add parallel execution support to `scheduler.rs` using rayon for independent node branches (verified 2026-05-16; src/scheduler.rs:33 parallelizable_groups computed by Kahn wave algorithm)
- [x] Extend `optimization.rs` with automatic filter fusion (merge adjacent compatible nodes) (verified 2026-05-16; src/optimization.rs:225 NodeFusionPass)
- [x] Add backpressure mechanism to `data_flow.rs` to prevent memory exhaustion on slow sinks (verified 2026-05-16; src/data_flow.rs:120 BackpressurePolicy enum, coordinator:138)
- [x] Implement frame format negotiation between connected ports in `port.rs` (verified 2026-05-16; src/port.rs:47 PortFormat enum with Video/Audio/Data variants)
- [x] Add dynamic graph reconfiguration (hot-swap nodes) without rebuilding the entire graph (hot_swap.rs done)
- [x] Extend `graph_stats.rs` with latency histograms per node and per edge (LatencyHistogram 32 log₂ buckets, O(1) record; 2026-05-30)
- [x] Add error recovery / retry semantics to `processing_graph.rs` for transient failures (RetryPolicy exponential backoff, TransientError trait; 2026-05-30)
- [x] Implement `node_cache.rs` with LRU eviction policy and configurable cache size limits (verified 2026-05-16; src/node_cache.rs:115 NodeCache, lru fn:144)
- [x] Add graph snapshot/restore for checkpoint-based processing in `serialize.rs` (verified 2026-05-16; src/graph_evaluator.rs:405 GraphStatsSnapshot)

## New Features
- [x] Add a `SplitFilter` node that duplicates frames to multiple output ports (tee/fanout) (verified 2026-05-16; src/filters/video/split.rs:56 SplitFilter)
- [x] Add a `MergeFilter` node that combines multiple input streams (e.g., picture-in-picture) (verified 2026-05-16; src/filters/video/merge.rs:164 MergeFilter)
- [x] Implement a `RateLimitFilter` to throttle frame throughput for real-time playback (verified 2026-05-16; src/filters/video/rate_limit.rs:77 RateLimitFilter)
- [x] Add a `TimecodeFilter` node that stamps/modifies frame timestamps (verified 2026-05-16; src/filters/video/mod.rs:123 TimecodeFilter)
- [x] Implement a graph DSL parser (text-based graph description) complementing `serialize.rs` (verified 2026-05-16; src/dsl.rs)
- [x] Add `filters/video/scale.rs` for resolution scaling within the graph pipeline (verified 2026-05-16; src/filters/video/scale.rs:802 lines)
- [x] Add `filters/video/crop.rs` for frame cropping within the graph pipeline (verified 2026-05-16; src/filters/video/crop.rs:699 lines)
- [x] Add `filters/audio/resample.rs` for sample rate conversion within the graph pipeline (verified 2026-05-16; src/filters/audio/resample.rs:910 lines)
- [x] Implement async graph execution mode using tokio tasks for I/O-bound source/sink nodes (verified 2026-05-16; src/async_exec.rs:65 AsyncExecutor with tokio tasks)

## Performance
- [x] Add SIMD-accelerated frame copy in `frame.rs` for large video buffers (AVX2 auto-vectorization; 2026-05-30)
- [x] Implement zero-copy frame passing between compatible adjacent nodes (SharedFrame done)
- [x] Add memory pool pre-allocation in `FramePool` based on graph topology analysis (FramePoolConfig done)
- [x] Profile and optimize `topological.rs` sort for large graphs (>1000 nodes) (FastTopoSorter done)
- [x] Add lock-free ring buffers for inter-node frame passing in `port_buffer.rs` (SpscRingBuffer; AtomicUsize head/tail; lock_free_ring.rs 287L; 2026-05-30)

## Testing
- [x] Add integration tests for complex multi-branch graph topologies (diamond, fan-out/fan-in) (tests/graph_topology.rs: diamond + fan-out/fan-in topo-order validity + cycle rejection; 2026-06-06)
- [x] Add stress tests for `graph_merge.rs` with overlapping node IDs (tests/cycle_and_merge.rs: 50/50 full-overlap remap + mixed-overlap bug-trip; FIXED Remap node-collision where a non-conflicting `other` id was silently overwritten by a remap id; 2026-06-06)
- [x] Test `cycle_detect.rs` with self-loops and multi-edge cycles (tests/cycle_and_merge.rs: self-loop, 2-node cycle, parallel-edge collapse, mixed cycle+tail; 2026-06-06)
- [x] Add benchmarks for graph execution throughput at various node counts (benches/graph_exec_bench.rs: ProcessingGraph::execution_order at 16/64/256/1024 nodes, seeded forward-only DAG; 2026-06-06)

## Documentation
- [ ] Add architecture diagram in module-level docs showing node/port/connection relationships
- [ ] Document thread-safety guarantees for concurrent graph execution
- [ ] Add examples for building common filter chains (transcode, overlay, split/merge)
