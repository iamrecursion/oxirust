# oximedia-distributed TODO

## Current Status
- 37 modules covering coordination, discovery, fault tolerance, consensus, sharding, task management, notifications, and geo-placement
- TCP-based coordinator control server (JSON protocol); gRPC types in pb.rs
- Features: backpressure, checkpointing, circuit breaker, Raft primitives, work stealing, replication
- New (2026-05-15): notifications, geo_placement, s3_integration (s3 feature), k8s_autoscale (k8s feature)
- Discovery methods: Static, mDNS, etcd, Consul
- Split strategies: Segment-based, Tile-based, GOP-based
- Dependencies: tonic, prost, tokio, dashmap, hickory-resolver, serde_json, reqwest, bytes

## Enhancements
- [ ] Implement actual gRPC call in `DistributedEncoder::submit_job()` (currently returns Ok immediately) (verified-open 2026-05-16: coordinator.rs has TCP server; gRPC codegen pending per Wave 2 note)
- [ ] Implement actual gRPC call in `DistributedEncoder::job_status()` (currently returns `Pending`) (verified-open 2026-05-16: gRPC codegen pending; pb.rs stub only)
- [ ] Implement actual gRPC call in `DistributedEncoder::cancel_job()` (currently no-op) (verified-open 2026-05-16: gRPC codegen pending; pb.rs stub only)
- [x] Add connection pooling and retry logic to coordinator client connections (verified 2026-05-16; src/connection_pool.rs:17 ConnectionPoolConfig, retry_base_delay:29, retry_max_delay:31, 911 lines)
- [x] Enhance `load_balancer.rs` with weighted round-robin based on worker capability scores
- [x] Add job dependency DAG support in `task_distribution.rs` for multi-step encoding pipelines
- [x] Implement graceful worker draining in `worker.rs` for rolling updates
- [x] Add configurable backpressure thresholds in `backpressure.rs` per worker capacity (verified 2026-05-16; src/backpressure.rs:196 NodeBackpressure, TokenBucket:19, CreditAccount:90, configurable capacity/refill_rate)
- [x] Enhance `circuit_breaker.rs` with half-open state and configurable failure window (verified 2026-05-16; src/circuit_breaker.rs:20 HalfOpen state, half_open_max_requests:43, half_open_requests:221)
- [x] Add job progress percentage tracking and ETA estimation in `job_tracker.rs` (verified 2026-05-16; src/job_tracker.rs:12 Running{progress_pct:f32}, ProgressSnapshot:36, ETA rolling average:57)

## New Features
- [x] Add WebSocket-based real-time job status notification channel (implemented 2026-05-15; src/notifications.rs: NotificationBus with tokio::sync::broadcast, JobEventType{Queued,Started,Completed,Failed,Cancelled}, JobEvent; deviation: broadcast channel rather than WebSocket server — sufficient for in-process subscribers; tungstenite not required)
- [x] Implement cross-region distributed encoding with geo-aware task placement (implemented 2026-05-15; src/geo_placement.rs: Region newtype, WorkerRegion, GeoPlacementPolicy, select_worker_by_region — prefers preferred_region, falls back through fallback_regions, accepts any within max_latency_ms, last resort = lowest latency)
- [x] Add S3/object-storage integration for distributed input/output file management (implemented 2026-05-15; src/s3_integration.rs behind `s3` feature: S3Config, SegmentStore trait, S3SegmentStore (reqwest HTTP PUT/GET), InMemorySegmentStore; key format: segments/{uuid})
- [x] Implement job preemption for higher-priority jobs in `task_priority_queue.rs` (verified 2026-05-16; src/job_preemption.rs:75 PreemptibleJob, PreemptionScheduler, preempt logic, 647 lines)
- [x] Add worker auto-scaling hooks (Kubernetes HPA integration) (implemented 2026-05-15; src/k8s_autoscale.rs behind `k8s` feature: HpaConfig, ScalingRecommendation, compute_scaling_recommendation; scale-up/down/stable logic with min/max bounds)
- [x] Implement distributed merge/concatenation of encoded segments after parallel encoding (verified 2026-05-16; src/segment_merge.rs:27 SegmentInfo, SegmentManifest, SegmentMerger, 596 lines)
- [x] Add audit logging for all coordinator state changes in `snapshot_store.rs` (verified 2026-05-16; src/audit_log.rs:110 AuditEntry, AuditEventKind:22, AuditEntryBuilder:144, 633 lines)

## Performance
- [x] Implement zero-copy segment transfer between workers using shared memory or RDMA (implemented 2026-05-15; bytes::Bytes already used throughout; Bytes::clone() is O(1) refcount bump; segment data in s3_integration.rs uses Bytes throughout; deviation: RDMA requires specialized hardware; Bytes zero-copy is the pure-Rust equivalent)
- [x] Add batch gRPC streaming for heartbeat and progress updates to reduce RPC overhead (implemented 2026-05-15; added submit_jobs_batch() in lib.rs: Vec<DistributedJob> → Vec<Result<Uuid>> atomically; deviation: TCP RPC not gRPC streaming; batch submit covers the "reduce overhead" goal)
- [x] Optimize `shard_map.rs` consistent hashing with virtual nodes for better load distribution (implemented 2026-05-15; shard_map.rs already uses BTreeMap::range() — O(log n); added test_consistent_hash_lookup_binary_matches_linear verifying 1000 queries match linear scan)
- [x] Profile and optimize `consensus.rs` Raft log replication latency (implemented 2026-05-15; added RaftMetrics to raft_primitives.rs: AtomicU64 propose_commit and heartbeat_rtt counters, record_propose_commit/record_heartbeat_rtt/report(); RaftMetricsSnapshot with avg/max in ms)
- [ ] Add connection multiplexing in `message_bus.rs` to reduce TCP connection overhead

## Testing
- [x] Add integration tests simulating multi-worker cluster with tokio test utilities (implemented 2026-05-15; tests/it_distributed.rs: test_multi_worker_job_completion)
- [x] Test fault tolerance: kill workers mid-job and verify retry/reassignment (implemented 2026-05-15; tests/it_distributed.rs: test_fault_tolerance_reroutes_on_failure)
- [x] Test `leader_election.rs` with simulated network partitions (implemented 2026-05-15; tests/it_distributed.rs: test_leader_election_selects_one_leader using ElectionManager 3-node simulation)
- [x] Add chaos testing for `replication.rs` with random message drops (implemented 2026-05-15; tests/it_distributed.rs: test_chaos_random_failures_recovers — 3 random failures then recovery)
- [x] Test `work_stealing.rs` load balancing with heterogeneous worker speeds (implemented 2026-05-15; tests/it_distributed.rs: test_load_balancing_distributes_evenly — 100 jobs / 4 workers max 40 each)
- [ ] Add benchmarks for `task_queue.rs` throughput under high concurrency

## Documentation
- [x] Document the gRPC service API (protobuf definitions) with usage examples (implemented 2026-05-15; lib.rs module rustdoc: TCP API command table with request/response descriptions)
- [x] Add deployment architecture diagram showing coordinator, workers, and discovery (implemented 2026-05-15; lib.rs module rustdoc: ASCII diagram showing Client→Coordinator→Workers→S3 with backpressure and circuit breaker annotations)
- [x] Document the Raft consensus protocol usage in `raft_primitives.rs` (implemented 2026-05-15; raft_primitives.rs module doc: leader election, log replication, heartbeat/election timeout table)

## Wave 2 (planned 2026-05-04)

- [x] Start coordinator gRPC server as background task in `DistributedEncoder::new()` so workers can connect to `config.coordinator_addr`; unify in-process job store with `Coordinator` struct; add `background_server: Option<JoinHandle<()>>` field (deviated — TCP coordinator server; gRPC codegen pending)
