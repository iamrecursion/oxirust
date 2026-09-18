# oximedia-batch TODO

## Current Status
- 30+ modules providing batch processing: job queuing, worker pool, templates, watch folders, distributed processing, REST API, CLI
- Feature-gated: `sqlite` (database, API, CLI, execution, watch), `scripting` (Lua via mlua)
- Sub-directories: `api/`, `cli/`, `database/`, `execution/`, `examples/`, `metrics/`, `monitoring/`, `notifications/`, `operations/`, `presets/`, `queue/`, `script/`, `template/`, `utils/`, `watch/`
- `BatchEngine` is the main entry point wrapping `JobQueue`, `ExecutionEngine`, and `Database`

## Enhancements
- [x] Add graceful shutdown support to `BatchEngine::stop()` that waits for in-progress jobs to complete (verified 2026-05-16; src/graceful_shutdown.rs:519 lines)
- [x] Extend `retry_policy` with exponential backoff jitter to prevent thundering herd on retries (verified 2026-05-16; src/retry_policy.rs:25 ExponentialWithJitter variant, full jitter algorithm)
- [x] Add job dependency chaining in `dep_graph` so jobs can declare predecessor requirements (verified 2026-05-16; src/dep_graph.rs:1259 lines)
- [x] Implement `checkpointing` module with periodic state snapshots for crash recovery (verified 2026-05-16; src/checkpointing.rs:658 lines)
- [x] Extend `notifications` with webhook callback support on job state transitions (verified 2026-05-16; src/notifications/mod.rs:57 Webhook variant, send_webhook:144)
- [x] Add `rate_limiter` integration with `execution` to enforce per-user or per-project job limits (verified 2026-05-16; src/rate_limiter.rs:597 lines)
- [x] Improve `priority_queue` with fair scheduling to prevent starvation of low-priority jobs
- [x] Add `batch_schedule` support for cron-like recurring job schedules (verified 2026-05-16; src/batch_schedule.rs:303 lines)
- [x] Extend `pipeline_validator` to check for circular dependencies in DAG workflows (verified 2026-05-16; src/pipeline_validator.rs:341 detect_cycle DFS, test_circular_dependency:524)

## New Features
- [x] Add `job_migration` module for upgrading job schemas when template format changes
- [x] Implement `cost_estimator` module predicting job duration and resource usage from historical data (verified 2026-05-16; src/cost_estimator.rs:797 lines)
- [x] Add `dead_letter_queue` for permanently failed jobs with configurable retention (verified 2026-05-16; src/dead_letter_queue.rs:663 lines)
- [x] Implement `job_splitting` module to automatically partition large transcode jobs across workers
- [x] Add `audit_log` module tracking who submitted/modified/cancelled each job (verified 2026-05-16; src/audit_log.rs:636 lines)
- [x] Implement `cluster_discovery` module for auto-detecting batch workers on the network (verified 2026-05-16; src/cluster_discovery.rs:861 lines)
- [x] Add `resource_reservation` module for pre-allocating GPU/CPU cores for high-priority jobs

## Performance
- [x] Use connection pooling for SQLite database access in `database` module (completed 2026-05-15 — `DatabasePool::new(path, max_connections)` added to `src/database/mod.rs` wrapping `r2d2::Pool<SqliteConnectionManager>`; WAL-mode concurrent test `test_pool_concurrent_access` (4 threads × 10 inserts = 40 rows verified); existing `Database::new` retains backward-compatible default pool)
- [x] Implement work-stealing scheduler in `execution` for better load balancing across workers
- [x] Add memory-mapped file I/O for large batch input files in `operations` (completed 2026-05-15 — `MmapReader` and `open_smart` in `src/operations/mmap_reader.rs`; 4 MiB threshold, falls back to `BufReader` for small files; `unsafe` block isolated with SAFETY comment; 4 tests including large-file mmap path verification)
- [x] Cache template parsing results in `template` to avoid re-parsing on repeated job submissions (completed 2026-06-05: `parse_template_to_segments` + `parsed_cache: Mutex<HashMap<String, Arc<Vec<Segment>>>>` in engine.rs; 6 new cache-correctness tests; 974 tests pass, 0 warnings)
- [ ] Use zero-copy deserialization for job payloads in `queue` using serde `borrow` (verified-open 2026-05-16: not yet implemented)

## Testing
- [ ] Add integration test for full job lifecycle: submit, queue, execute, complete, query status
- [ ] Test `watch` folder monitoring with rapid file creation/deletion
- [ ] Add test for `conditional_dag` with branching execution paths
- [ ] Test `timeout_enforcer` correctly cancels jobs exceeding time limits
- [ ] Add stress test submitting 10,000 jobs concurrently to verify queue stability
- [ ] Test `task_group` parallel execution with mixed success/failure outcomes

## Documentation
- [ ] Document REST API endpoints in `api` module with request/response examples
- [ ] Add guide for creating custom `template` configurations for common transcode workflows
- [ ] Document `watch` folder setup including supported file patterns and polling intervals

## Wave 4 Progress (2026-04-18)
- [x] wasm-mio-fix: cfg-gate tokio deps for wasm32 target compatibility — Wave 4 Slice A
