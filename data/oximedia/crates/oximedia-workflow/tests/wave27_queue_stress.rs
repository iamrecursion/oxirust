//! Wave 27 — `TaskQueue` concurrency and ordering conformance.
//!
//! These tests pin the queue's contract under concurrent load and verify the
//! priority-desc / FIFO-within-priority ordering guarantees:
//!
//! * `test_queue_1000_concurrent_no_loss` — 1000 concurrent enqueues onto a
//!   single Arc-shared queue lose and duplicate nothing.
//! * `test_queue_fifo_within_priority_sequential` — same-priority tasks dequeue
//!   in submission order (FIFO via the monotonic sequence counter).
//! * `test_queue_priority_buckets_monotone` — every higher-priority task is
//!   dequeued before any lower-priority task, regardless of submission order.

use std::collections::HashSet;
use std::time::Duration;

use futures::future::join_all;
use oximedia_workflow::{Task, TaskId, TaskPriority, TaskQueue, TaskType};

/// Build a trivial `Wait` task at the given priority.
fn make_task(name: &str, priority: TaskPriority) -> Task {
    Task::new(
        name,
        TaskType::Wait {
            duration: Duration::from_secs(1),
        },
    )
    .with_priority(priority)
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn test_queue_1000_concurrent_no_loss() {
    let queue = TaskQueue::new();

    // Spawn 1000 concurrent producers, each enqueuing one task onto a clone of
    // the Arc-shared queue. Collect the IDs we *intended* to enqueue.
    let mut handles = Vec::with_capacity(1000);
    for i in 0..1000 {
        let q = queue.clone();
        handles.push(tokio::spawn(async move {
            let task = make_task(&format!("task-{i}"), TaskPriority::Normal);
            let id = task.id;
            q.enqueue(task)
                .await
                .expect("enqueue should succeed in test");
            id
        }));
    }

    let mut submitted: HashSet<TaskId> = HashSet::with_capacity(1000);
    for handle in join_all(handles).await {
        let id = handle.expect("producer task should not panic");
        // No duplicate IDs were minted.
        assert!(submitted.insert(id), "TaskId collision among producers");
    }
    assert_eq!(submitted.len(), 1000, "all 1000 IDs should be unique");

    // Drain everything via try_dequeue; every dequeued ID must be one we
    // submitted, with no loss and no duplication.
    let mut drained: HashSet<TaskId> = HashSet::with_capacity(1000);
    while let Some(task) = queue.try_dequeue().await {
        assert!(
            drained.insert(task.id),
            "the same task was dequeued twice: {}",
            task.id
        );
    }

    assert_eq!(
        drained.len(),
        1000,
        "exactly 1000 tasks must be drained — no loss"
    );
    assert_eq!(
        drained, submitted,
        "the drained set must equal the submitted set"
    );
}

#[tokio::test]
async fn test_queue_fifo_within_priority_sequential() {
    let queue = TaskQueue::new();

    // Enqueue 100 same-priority tasks in a deterministic order.
    let mut order: Vec<TaskId> = Vec::with_capacity(100);
    for i in 0..100 {
        let task = make_task(&format!("fifo-{i}"), TaskPriority::Normal);
        order.push(task.id);
        queue
            .enqueue(task)
            .await
            .expect("enqueue should succeed in test");
    }

    // Dequeue order must match submission order for equal priority (FIFO).
    let mut dequeued: Vec<TaskId> = Vec::with_capacity(100);
    while let Some(task) = queue.try_dequeue().await {
        dequeued.push(task.id);
    }

    assert_eq!(dequeued.len(), 100);
    assert_eq!(
        dequeued, order,
        "same-priority tasks must dequeue in submission (FIFO) order"
    );
}

#[tokio::test]
async fn test_queue_priority_buckets_monotone() {
    let queue = TaskQueue::new();

    // Submit a deliberately interleaved mix of all four priorities so that
    // submission order does NOT match priority order.
    let mix = [
        TaskPriority::Low,
        TaskPriority::Critical,
        TaskPriority::Normal,
        TaskPriority::High,
        TaskPriority::Low,
        TaskPriority::High,
        TaskPriority::Critical,
        TaskPriority::Normal,
        TaskPriority::Low,
        TaskPriority::Critical,
    ];
    for (i, &priority) in mix.iter().enumerate() {
        let task = make_task(&format!("prio-{i}"), priority);
        queue
            .enqueue(task)
            .await
            .expect("enqueue should succeed in test");
    }

    // Dequeue all and record the priority sequence.
    let mut priorities: Vec<TaskPriority> = Vec::with_capacity(mix.len());
    while let Some(task) = queue.try_dequeue().await {
        priorities.push(task.priority);
    }
    assert_eq!(priorities.len(), mix.len());

    // The dequeued priority sequence must be non-increasing: every higher
    // priority is fully drained before any lower priority appears.
    for window in priorities.windows(2) {
        assert!(
            window[0] >= window[1],
            "priority order violated: {:?} dequeued before {:?}",
            window[0],
            window[1]
        );
    }

    // Sanity: Critical first, Low last given the mix above.
    assert_eq!(priorities[0], TaskPriority::Critical);
    assert_eq!(
        priorities[priorities.len() - 1],
        TaskPriority::Low,
        "lowest priority must be drained last"
    );
}
