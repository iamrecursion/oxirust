# oximedia-pipeline TODO

## Current Status
- 4 modules: `builder`, `execution_plan`, `graph`, `node`
- Provides declarative media processing pipeline DSL with typed filter graph
- Supports source/sink/filter nodes, pipeline validation, topological sort, execution planning
- Minimal dependencies (thiserror, uuid)

## Enhancements
- [x] Add parallel execution support in `ExecutionPlan` for independent pipeline branches
- [x] Implement pipeline serialization/deserialization (serde support for `PipelineGraph`)
- [x] Add pipeline merging — combine two `PipelineGraph` instances with shared nodes
- [x] Extend `FilterConfig` with parametric filter configuration (key-value properties)
- [x] Add `PipelineBuilder::branch()` for splitting a pipeline into multiple output paths
- [x] Implement cycle detection with detailed path reporting in `PipelineError::CycleDetected` (verified 2026-05-16; src/lib.rs:46 CycleDetected{path}, src/topo_tests.rs:290 tests cycle reporting)
- [x] Add pipeline graph visualization export (DOT/Graphviz format from `PipelineGraph`) (verified 2026-05-16; src/dot.rs:1 DotExporter, fn export)
- [x] Support dynamic pipeline reconfiguration (add/remove nodes at runtime) (verified 2026-05-16; src/dynamic_reconfig.rs:55 reconfiguration operations, DynamicPipeline::commit)

## New Features
- [x] Add `PipelineProfiler` for timing each node's execution within a pipeline (verified 2026-05-16; src/pipeline_profiler.rs:14 PipelineProfiler ring-buffer, NodeTiming, PipelineExecutionProfile)
- [x] Implement pipeline templates — predefined pipeline patterns (transcode, ABR, thumbnail) (verified 2026-05-16; src/templates.rs:212 PipelineTemplate::Transcode enum, TemplateConfig)
- [x] Add a `PipelineValidator` that checks hardware resource availability before execution (verified 2026-05-16; src/hardware_validator.rs:46 AvailableResources, src/validation.rs:33 PipelineValidator)
- [x] Implement pipeline checkpointing — save/resume interrupted pipelines
- [x] Add conditional nodes (`IfNode`) for branching based on stream properties (verified 2026-05-16; src/conditional.rs:331 ConditionalBranch, conditional pipeline description:369)
- [x] Implement pipeline composition — nest sub-pipelines as single nodes (verified 2026-05-16; src/composition.rs:1 ComposedNode, BoundaryMap nested sub-pipelines)
- [x] Add `PipelineMetrics` for collecting throughput, latency, and buffer stats per node (verified 2026-05-16; src/metrics.rs:3 PipelineMetrics, NodeMetrics per-node stats)

## Performance
- [x] Implement node fusion optimization in `PipelineOptimizer` (merge adjacent scale+crop) (verified 2026-05-16; src/optimizer.rs:415 fn apply_node_fusion, scale+crop/pad fusion at :174/:197)
- [x] Add zero-copy frame passing between adjacent nodes in the same thread (verified 2026-05-16; src/zero_copy.rs:9 ZeroCopyFrame Arc<[u8]>, ZeroCopyChannel MPSC)
- [x] Implement memory pool allocation for `ResourceEstimate` to reduce allocation overhead (verified 2026-06-01; memory_pool.rs:75 PoolConfig::from_estimate, lib.rs:35 pub mod memory_pool)
- [x] Add SIMD-aware node scheduling in `ExecutionPlanner` for data-parallel filters (verified 2026-05-16; src/simd_scheduler.rs:36 SimdScheduler, CpuCapabilities, SimdTier)

## Testing
- [x] Add property-based tests for topological sort correctness with random graph shapes (verified 2026-06-01; tests/wave15_tests.rs test_pipeline_topo_sort_random_graph: 5 seeded xorshift32 DAGs 8-12 nodes, forward-edge invariant)
- [x] Test pipeline validation with malformed graphs (disconnected nodes, dangling pads) (verified 2026-06-01; tests/wave15_tests.rs test_pipeline_validation_disconnected_nodes + validation.rs existing tests)
- [x] Add benchmark tests for `ExecutionPlanner` with large pipeline graphs (1000+ nodes) (verified 2026-06-06; tests/wave15_tests.rs:387 test_planner_large_dag_valid_topo_order, :420 test_planner_large_dag_all_scheduled_once, :455 test_planner_fan_in_fan_out_heavy, :542 test_planner_chain_and_wide_layer_structure — 1000+ node layered/chain/wide/fan-in-out DAGs, valid topo order + every node scheduled once + level ordering)
- [ ] Test `PipelineBuilder` chain correctness for all filter types (scale, flip, crop, etc.)

## Documentation
- [ ] Add architecture diagram showing node/edge/pad relationship model
- [ ] Document thread safety guarantees for concurrent pipeline execution
- [ ] Add examples for multi-input pipelines (picture-in-picture, audio mixing)
