# oximedia-workflow TODO

## Current Status
- 46 source files implementing comprehensive workflow orchestration engine
- Key features: DAG-based workflow definition, task dependencies and parallel execution, SQLite persistence (feature-gated), cron-style scheduling, REST API (axum), WebSocket real-time monitoring, WorkflowEngine main entry point, multi-pass encoding patterns, SLA tracking, cost tracking, approval gates, audit logging, notification system, retry policies, resource pools, workflow versioning/migration/snapshots/checkpoints
- Modules: api, approval_gate, audit_log, builder, cli, cost_tracking, dag, executor, monitoring, notification_system, patterns, persistence, queue, resource_pool, retry_policy, scheduler, sla/sla_tracking, state_machine, step_condition/step_conditions/step_result, task/task_dependency/task_graph/task_priority_queue/task_template, templates, triggers, validation (ComplexityAnalyzer), websocket, workflow/workflow_audit/workflow_checkpoint/workflow_log/workflow_metrics/workflow_migration/workflow_retry/workflow_snapshot/workflow_template/workflow_throttle/workflow_version
- Dependencies: tokio, axum, rusqlite (optional), serde, chrono, cron, uuid, clap, dashmap, parking_lot

## Enhancements
- [x] Add workflow branching/conditional paths in `dag` (if-else nodes that choose next task based on previous task output)
- [x] Implement parallel fan-out/fan-in pattern in `executor` for tasks with shared dependencies (verified 2026-05-16; src/fanout.rs:552 lines)
- [x] Extend `retry_policy` with circuit breaker pattern (stop retrying after N consecutive failures across workflows) (verified 2026-05-16; src/circuit_breaker.rs:751 lines)
- [x] Add `workflow_migration` actual schema migration logic (currently likely scaffolding -- implement versioned DB migrations) (verified 2026-05-16; src/workflow_migration.rs:123 MigrationStep, MigrationError enum)
- [x] Extend `triggers` with webhook triggers (HTTP POST starts workflow) beyond cron and file-watch (verified 2026-05-16; src/webhook_trigger.rs:532 lines)
- [x] Implement workflow pause/resume in `executor` with checkpoint serialization for long-running workflows (verified 2026-05-16; src/pause_resume.rs:684 lines)
- [x] Add dynamic resource scaling in `resource_pool` based on queue depth (auto-allocate more workers under load)
- [x] Extend `notification_system` with Slack, email, and PagerDuty integration via webhook URLs
- [ ] Improve `cost_tracking` with actual cloud cost API integration (estimate compute cost per task based on duration and resource type) (verified-open 2026-05-16: no cloud_api/external API integration in cost_tracking.rs)

## New Features
- [x] Implement `workflow_compose` module for composing smaller workflows into larger meta-workflows
- [x] Add `workflow_import_export` for importing/exporting workflows as portable YAML/JSON bundles (verified 2026-05-16; src/workflow_import_export.rs:815 lines)
- [x] Implement `workflow_diff` module for comparing two workflow versions and showing added/removed/changed tasks
- [x] Add `workflow_simulation` dry-run mode that traces execution path without actually running tasks (verified 2026-05-16; src/workflow_simulation.rs:1182 lines)
- [x] Implement `workflow_marketplace` module for sharing and discovering reusable workflow templates
- [x] Add `event_bus` module for publish/subscribe event-driven communication between workflow tasks (verified 2026-05-16; src/event_bus.rs:825 lines)
- [x] Implement `workflow_dashboard` data provider module that aggregates metrics for web UI consumption
- [x] Add `workflow_health_check` module for periodic validation of workflow engine health (DB connectivity, queue depth, stuck tasks)

## Performance
- [x] Optimize `task_priority_queue` with binary heap instead of sorted Vec for O(log n) insert/extract (verified: task_priority_queue.rs:BinaryHeap<PriorityEntry>, :54:PriorityEntry)
- [x] Add connection pooling in `persistence` for SQLite (using r2d2 pool with configurable size) (verified: persistence.rs:r2d2::Pool+SqliteConnectionManager)
- [x] Implement batch task status updates in `executor` to reduce database write frequency (Wave 15: StatusUpdate buffer, buffer_flush_threshold=20, buffer_status_update/flush_status_buffer/flush methods, auto-flush on threshold reached)
- [x] Cache workflow DAG topology in `dag` after first computation to avoid recomputation on each execution (Wave 15: RefCell<Option<Vec<NodeId>>> topo_cache, invalidated in add_node/add_edge, returned from topological_sort on cache hit)
- [x] Optimize `monitoring` metric collection with lock-free counters instead of mutex-guarded HashMap (Wave 15: completed_tasks/failed_tasks/running_tasks converted to Arc<AtomicU64>, DashMap already in use for MonitoringService, concurrent test passes N=8 threads × M=100 tasks)
- [x] Add lazy deserialization in `persistence::load_workflow` to skip parsing task configs until accessed (Wave 15: LazyWorkflowConfig with Mutex<Option<WorkflowConfig>> cache, get_cloned() API, raw() for zero-cost access)

## Testing
- [x] Add integration test for full workflow lifecycle: create -> submit -> execute -> complete with SQLite persistence (Wave 15: test_sqlite_lifecycle in tests/wave15_tests.rs, sqlite feature-gated)
- [x] Test `dag` cycle detection with intentionally cyclic graphs and verify proper error reporting (Wave 15: test_dag_cycle_detection_negative verifies add_edge returns Err(DagError::CycleDetected))
- [x] Add stress test for `queue` with 1000+ concurrent task submissions and verify ordering correctness (Wave 27: tests/wave27_queue_stress.rs — 1000 concurrent enqueues no-loss/no-dup HashSet drain, FIFO-within-priority 100-task order, priority-bucket monotone non-increasing drain)
- [x] Test `scheduler` cron trigger firing accuracy with mock clock (Wave 27: injectable `Clock` trait + `SystemClock` default in scheduler.rs, `WorkflowScheduler::with_clock`; tests/wave27_scheduler_clock.rs FakeClock drives per-minute cron — fires exactly when now>=next_execution then empties, next_execution advances one period)
- [x] Add `approval_gate` test verifying workflow blocks until approval is granted and resumes correctly (Wave 27: tests/wave27_approval_block.rs — Notify+Mutex<ApprovalGate> harness, deterministic ["blocked","approved","resumed"] order, no sleeps, timeout only as safety net)
- [ ] Test `workflow_checkpoint` save/restore across process restarts (serialize state, reload, continue execution)
- [ ] Test `sla_tracking` with workflows that exceed SLA and verify breach notification

## Documentation
- [ ] Document workflow YAML/JSON schema with annotated examples for common media processing workflows
- [ ] Add REST API endpoint reference with request/response examples for `api` module
- [ ] Document task type catalog with required/optional parameters for each TaskType variant
