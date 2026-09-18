//! Wave 15 integration tests for oximedia-workflow.
//!
//! Tests:
//!   1. `test_dag_topo_cache_matches_recompute` — cache correctness
//!   2. `test_dag_cycle_detection_negative` — cycle → error
//!   3. `test_atomic_counters_concurrent` — N threads × M tasks
//!   4. `test_executor_batch_flush` — threshold-based flushing
//!   5. `test_lazy_deser_parses_on_access` — LazyWorkflowConfig (sqlite feature)
//!   6. `test_sqlite_lifecycle` — create → save → load → assert (sqlite feature)

use oximedia_workflow::dag::{DagError, WorkflowDag, WorkflowEdge, WorkflowNode};
use oximedia_workflow::executor::{DefaultTaskExecutor, StatusUpdate, WorkflowExecutor};
use oximedia_workflow::monitoring::MonitoringService;
use oximedia_workflow::task::{TaskId, TaskState};
use oximedia_workflow::workflow::WorkflowId;
use std::sync::Arc;

// ---------------------------------------------------------------------------
// Helper
// ---------------------------------------------------------------------------

fn wait_node(name: &str) -> WorkflowNode {
    WorkflowNode::new(name)
}

// ---------------------------------------------------------------------------
// Item 1 — DAG topology cache
// ---------------------------------------------------------------------------

#[test]
fn test_dag_topo_cache_matches_recompute() {
    let mut dag = WorkflowDag::new();
    let a = dag.add_node(wait_node("a")).expect("add node a");
    let b = dag.add_node(wait_node("b")).expect("add node b");
    let c = dag.add_node(wait_node("c")).expect("add node c");

    dag.add_edge(WorkflowEdge::new(a, b, "x"))
        .expect("edge a→b");
    dag.add_edge(WorkflowEdge::new(b, c, "x"))
        .expect("edge b→c");

    // First call — computes and caches.
    let order1 = dag.topological_sort().expect("topo sort 1");
    // Second call — returns cached copy (same length and content).
    let order2 = dag.topological_sort().expect("topo sort 2");

    assert_eq!(order1, order2, "cached order must match recomputed order");

    let pos_a = order1.iter().position(|&x| x == a).expect("a in order");
    let pos_b = order1.iter().position(|&x| x == b).expect("b in order");
    let pos_c = order1.iter().position(|&x| x == c).expect("c in order");
    assert!(pos_a < pos_b, "a must precede b");
    assert!(pos_b < pos_c, "b must precede c");

    // Add a new node + edge — cache must be invalidated.
    let d = dag.add_node(wait_node("d")).expect("add node d");
    dag.add_edge(WorkflowEdge::new(c, d, "x"))
        .expect("edge c→d");

    // The new order must include d and respect c → d.
    let order3 = dag.topological_sort().expect("topo sort 3");
    assert_eq!(order3.len(), 4, "order should now include d");

    let pos_c2 = order3.iter().position(|&x| x == c).expect("c in order3");
    let pos_d = order3.iter().position(|&x| x == d).expect("d in order3");
    assert!(pos_c2 < pos_d, "c must precede d after cache invalidation");
}

// ---------------------------------------------------------------------------
// Item 1 — DAG cycle detection
// ---------------------------------------------------------------------------

#[test]
fn test_dag_cycle_detection_negative() {
    let mut dag = WorkflowDag::new();
    let a = dag.add_node(wait_node("a")).expect("add a");
    let b = dag.add_node(wait_node("b")).expect("add b");
    let c = dag.add_node(wait_node("c")).expect("add c");

    dag.add_edge(WorkflowEdge::new(a, b, "x")).expect("a→b ok");
    dag.add_edge(WorkflowEdge::new(b, c, "x")).expect("b→c ok");

    // Creating a back-edge (c → a) must be rejected.
    let result = dag.add_edge(WorkflowEdge::new(c, a, "x"));
    assert!(
        matches!(result, Err(DagError::CycleDetected)),
        "cycle must be detected: got {:?}",
        result
    );

    // topological_sort on a fresh cyclic dag (two nodes that reference each other).
    let mut dag2 = WorkflowDag::new();
    let x = dag2.add_node(wait_node("x")).expect("add x");
    let y = dag2.add_node(wait_node("y")).expect("add y");
    // First edge is fine.
    dag2.add_edge(WorkflowEdge::new(x, y, "xy"))
        .expect("x→y ok");
    // Back edge is rejected.
    let back = dag2.add_edge(WorkflowEdge::new(y, x, "yx"));
    assert!(
        matches!(back, Err(DagError::CycleDetected)),
        "back-edge must be rejected"
    );
}

// ---------------------------------------------------------------------------
// Item 2 — Atomic counters: concurrent increment
// ---------------------------------------------------------------------------

#[test]
fn test_atomic_counters_concurrent() {
    use oximedia_workflow::task::TaskId;
    use std::thread;

    const THREADS: u64 = 8;
    const TASKS_PER_THREAD: u64 = 100;

    let service = Arc::new(MonitoringService::new());
    let workflow_id = WorkflowId::new();

    // Register the workflow with enough task slots.
    service.start_workflow(
        workflow_id,
        "concurrent-test".to_string(),
        (THREADS * TASKS_PER_THREAD) as usize,
    );

    let mut handles = Vec::with_capacity(THREADS as usize);
    for _ in 0..THREADS {
        let svc = Arc::clone(&service);
        handles.push(thread::spawn(move || {
            for _ in 0..TASKS_PER_THREAD {
                let task_id = TaskId::new();
                // Running → Completed lifecycle.
                svc.update_task(
                    workflow_id,
                    task_id,
                    "worker-task".to_string(),
                    oximedia_workflow::task::TaskState::Running,
                    None,
                );
                svc.update_task(
                    workflow_id,
                    task_id,
                    "worker-task".to_string(),
                    oximedia_workflow::task::TaskState::Completed,
                    None,
                );
            }
        }));
    }

    for h in handles {
        h.join().expect("thread panicked");
    }

    let metrics = service
        .get_workflow_metrics(&workflow_id)
        .expect("metrics must exist");

    let completed = metrics.completed_tasks_count();
    assert_eq!(
        completed,
        THREADS * TASKS_PER_THREAD,
        "all tasks must be counted as completed: got {} expected {}",
        completed,
        THREADS * TASKS_PER_THREAD
    );
    assert_eq!(
        metrics.running_tasks_count(),
        0,
        "no tasks should still be running"
    );
}

// ---------------------------------------------------------------------------
// Item 3 — Executor batch status flush
// ---------------------------------------------------------------------------

#[test]
fn test_executor_batch_flush() {
    let threshold = 5_usize;
    let executor =
        WorkflowExecutor::new(Arc::new(DefaultTaskExecutor)).with_flush_threshold(threshold);

    // Push threshold-1 updates — buffer should NOT be flushed yet.
    for i in 0..(threshold - 1) {
        let update = StatusUpdate::new(
            TaskId::new(),
            if i % 2 == 0 {
                TaskState::Completed
            } else {
                TaskState::Failed
            },
        );
        executor
            .buffer_status_update(update)
            .expect("buffer_status_update must not error");
    }

    let buffered = executor
        .buffered_update_count()
        .expect("buffered_update_count must not error");
    assert_eq!(
        buffered,
        threshold - 1,
        "buffer should hold {} updates before threshold",
        threshold - 1
    );

    // Push one more update to reach threshold — auto-flush should clear buffer.
    executor
        .buffer_status_update(StatusUpdate::new(TaskId::new(), TaskState::Completed))
        .expect("final update must not error");

    let after_flush = executor
        .buffered_update_count()
        .expect("buffered_update_count after flush");
    assert_eq!(
        after_flush, 0,
        "buffer must be empty after reaching threshold"
    );

    // Push a few more and explicit flush.
    for _ in 0..3 {
        executor
            .buffer_status_update(StatusUpdate::new(TaskId::new(), TaskState::Completed))
            .expect("buffer");
    }
    executor.flush().expect("explicit flush must not error");
    assert_eq!(
        executor
            .buffered_update_count()
            .expect("count after explicit flush"),
        0,
        "buffer must be empty after explicit flush"
    );
}

// ---------------------------------------------------------------------------
// Item 4 — Lazy deserialization of WorkflowConfig (sqlite feature required
// because LazyWorkflowConfig lives in the persistence module which is
// feature-gated)
// ---------------------------------------------------------------------------

#[cfg(feature = "sqlite")]
#[test]
fn test_lazy_deser_parses_on_access() {
    use oximedia_workflow::persistence::LazyWorkflowConfig;
    use oximedia_workflow::workflow::WorkflowConfig;

    let config = WorkflowConfig {
        max_concurrent_tasks: 8,
        fail_fast: true,
        ..Default::default()
    };
    let json = serde_json::to_string(&config).expect("serialize config");

    // Construct lazy wrapper — no parsing yet.
    let lazy = LazyWorkflowConfig::new(json.clone());

    // Raw JSON is always available without parsing.
    assert_eq!(lazy.raw(), json.as_str(), "raw JSON must be preserved");

    // First access triggers parsing and caches the result.
    let parsed = lazy
        .get_cloned()
        .expect("first access must parse correctly");
    assert_eq!(parsed.max_concurrent_tasks, 8, "max_concurrent_tasks");
    assert!(parsed.fail_fast, "fail_fast");

    // Second access returns a clone of the cached result — same values.
    let parsed2 = lazy
        .get_cloned()
        .expect("second access must return cached value");
    assert_eq!(
        parsed.max_concurrent_tasks, parsed2.max_concurrent_tasks,
        "cached value must equal original"
    );
    assert_eq!(
        parsed.fail_fast, parsed2.fail_fast,
        "cached fail_fast must match"
    );
}

// ---------------------------------------------------------------------------
// Item 6 — SQLite lifecycle (feature-gated)
// ---------------------------------------------------------------------------

#[cfg(feature = "sqlite")]
#[test]
fn test_sqlite_lifecycle() {
    use oximedia_workflow::persistence::PersistenceManager;
    use oximedia_workflow::task::{Task, TaskType};
    use oximedia_workflow::workflow::{Workflow, WorkflowState};
    use std::time::Duration;

    let persistence = PersistenceManager::in_memory().expect("in-memory db");

    // Create and save a workflow.
    let mut workflow = Workflow::new("sqlite-lifecycle-test");
    let task = Task::new(
        "step-1",
        TaskType::Wait {
            duration: Duration::from_millis(1),
        },
    );
    workflow.add_task(task);

    persistence
        .save_workflow(&workflow)
        .expect("save_workflow must succeed");

    // Load it back.
    let loaded = persistence
        .load_workflow(workflow.id)
        .expect("load_workflow must succeed");

    assert_eq!(loaded.id, workflow.id, "workflow ID must match");
    assert_eq!(loaded.name, "sqlite-lifecycle-test", "name must match");
    assert_eq!(loaded.tasks.len(), 1, "one task must be present");
    // State defaults to Created on load.
    assert_eq!(loaded.state, WorkflowState::Created, "initial state");
}
