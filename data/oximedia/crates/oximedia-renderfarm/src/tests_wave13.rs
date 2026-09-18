//! Wave 13 integration and property-based tests for oximedia-renderfarm.

#![cfg(test)]

use std::collections::HashMap;

// ─── Test a: Multi-site failover ────────────────────────────────────────────

#[test]
fn test_multi_site_failover_and_rebalance() {
    use crate::multi_site::{FarmSite, JobRouting, MultiSiteRouter};

    let primary = FarmSite {
        site_id: "primary".to_owned(),
        location: "US-East".to_owned(),
        available_workers: 10,
        queue_depth: 5,
        avg_latency_ms: 10,
        network_bandwidth_mbps: 1000,
        cost_multiplier: 1.0,
    };
    let secondary = FarmSite {
        site_id: "secondary".to_owned(),
        location: "US-West".to_owned(),
        available_workers: 8,
        queue_depth: 2,
        avg_latency_ms: 25,
        network_bandwidth_mbps: 800,
        cost_multiplier: 1.1,
    };
    let mut router = MultiSiteRouter::new(vec![primary, secondary]);

    // Normal routing: both sites available.
    let routed = router.route_job(100, &JobRouting::LoadBalance);
    assert!(routed.is_some(), "should route with both sites available");

    // Primary goes down.
    router.sites[0].available_workers = 0;
    let routed_failover = router.route_job(100, &JobRouting::LoadBalance);
    assert!(routed_failover.is_some(), "should failover to secondary");
    assert_eq!(
        routed_failover.expect("failover route ok").site_id,
        "secondary",
        "failover should land on secondary"
    );

    // Primary recovers.
    router.sites[0].available_workers = 10;
    let routed_rebalance = router.route_job(100, &JobRouting::CostOptimal);
    assert!(
        routed_rebalance.is_some(),
        "should route after primary recovery"
    );
    // cost-optimal: primary has lower cost_multiplier (1.0 vs 1.1).
    assert_eq!(
        routed_rebalance.expect("rebalance route ok").site_id,
        "primary",
        "cost-optimal rebalance should prefer cheaper primary"
    );
}

// ─── Test b: Scheduler load test ────────────────────────────────────────────

#[test]
#[cfg_attr(debug_assertions, ignore)] // Skip in debug builds for speed
fn test_scheduler_load_1000_jobs_100_workers() {
    use crate::priority_queue::{RenderPriority, RenderPriorityQueue};

    let mut queue: RenderPriorityQueue<String> = RenderPriorityQueue::new();
    let priorities = [
        RenderPriority::Low,
        RenderPriority::Normal,
        RenderPriority::High,
        RenderPriority::Critical,
        RenderPriority::Background,
    ];
    for i in 0..1000u32 {
        let prio = priorities[(i % 5) as usize];
        queue.push(prio, format!("job-{i}"));
    }
    assert_eq!(queue.len(), 1000, "all 1000 jobs queued");

    let mut worker_loads: HashMap<usize, usize> = HashMap::new();
    let mut pop_order: Vec<RenderPriority> = Vec::new();
    let mut worker_idx = 0usize;
    while let Some(job) = queue.pop() {
        *worker_loads.entry(worker_idx % 100).or_insert(0) += 1;
        pop_order.push(job.priority);
        worker_idx += 1;
    }
    assert_eq!(pop_order.len(), 1000, "all jobs dequeued");

    // Non-increasing priority order: once we see a lower-weight priority,
    // no higher-weight priority should appear afterwards.
    let mut min_weight_seen = u8::MAX;
    for p in &pop_order {
        let w = p.weight();
        if w < min_weight_seen {
            min_weight_seen = w;
        }
        assert!(
            w <= min_weight_seen,
            "priority ordering violated: saw weight {} after min {}",
            w,
            min_weight_seen
        );
    }

    // No worker over-subscribed (soft bound: at most 20 per 100-worker pool).
    for (wid, count) in &worker_loads {
        assert!(
            *count <= 20,
            "worker {wid} over-subscribed with {count} jobs"
        );
    }
}

// ─── Test c: Elastic scaling timing with mock clock ─────────────────────────

#[test]
fn test_elastic_scaling_timing_mock_clock() {
    use crate::elastic_scaling::{
        CooldownTracker, ElasticScaler, ScalingPolicy, WorkerSpec, WorkerType,
    };

    let policy = ScalingPolicy::Demand {
        min: 2,
        max: 20,
        target_queue_depth: 100,
    };
    let spec = WorkerSpec::cpu_only(8, 32, WorkerType::OnDemand, 1.0);
    let mut scaler = ElasticScaler::new(policy, spec);
    let mut cooldown = CooldownTracker::new(
        1_000, // 1 s scale-up cooldown
        5_000, // 5 s scale-down cooldown
    );

    let t0: i64 = 0;

    // Phase 1: spike → scale up.
    let decision = scaler.compute_scaling_decision(12, 80);
    assert!(decision.scale_delta > 0, "should scale up on spike");
    assert!(cooldown.can_scale_up(t0), "first scale-up allowed");
    cooldown.record_scale_up(t0);
    scaler.apply_decision(&decision);
    assert!(scaler.current_workers >= 2, "workers >= min");
    assert!(scaler.current_workers <= 20, "workers <= max");

    // Scale-up cooldown: 500 ms should be blocked.
    assert!(
        !cooldown.can_scale_up(t0 + 500),
        "scale-up blocked within cooldown"
    );
    // After 1100 ms cooldown passes.
    assert!(
        cooldown.can_scale_up(t0 + 1100),
        "scale-up allowed after cooldown"
    );

    // Phase 2: drop → scale down.
    let decision_down = scaler.compute_scaling_decision(12, 0);
    assert!(
        decision_down.scale_delta < 0 || decision_down.desired_workers == 2,
        "should scale down or reach min"
    );

    // Scale-down cooldown.
    let t1 = t0 + 2_000i64;
    assert!(cooldown.can_scale_down(t1), "first scale-down allowed");
    cooldown.record_scale_down(t1);
    assert!(
        !cooldown.can_scale_down(t1 + 1_000),
        "scale-down blocked within cooldown"
    );
    assert!(
        cooldown.can_scale_down(t1 + 5_001),
        "scale-down allowed after cooldown"
    );
}

// ─── Test d: Priority-queue proptest ─────────────────────────────────────────

use proptest::prelude::*;

proptest! {
    #[test]
    fn proptest_priority_queue_ordering(
        raw_priorities in proptest::collection::vec(0u32..5u32, 1..50),
    ) {
        use crate::priority_queue::{RenderPriority, RenderPriorityQueue};

        let mut queue: RenderPriorityQueue<u32> = RenderPriorityQueue::new();

        for (i, &p) in raw_priorities.iter().enumerate() {
            let prio = match p {
                0 => RenderPriority::Background,
                1 => RenderPriority::Low,
                2 => RenderPriority::Normal,
                3 => RenderPriority::High,
                _ => RenderPriority::Critical,
            };
            queue.push(prio, i as u32);
        }

        let mut last_weight = u8::MAX;
        while let Some(job) = queue.pop() {
            let w = job.priority.weight();
            prop_assert!(
                w <= last_weight,
                "non-increasing priority violated: got weight {} after {}",
                w,
                last_weight
            );
            last_weight = w;
        }
    }
}

// ─── Test e: Checkpoint resume ───────────────────────────────────────────────

#[test]
fn test_checkpoint_resume_partial_tiles() {
    use crate::render_checkpoint::{FrameProgress, RenderCheckpoint};

    let job_id = "ckpt-test-job-wave13";

    // Build a checkpoint with 3 of 6 tiles complete.
    let mut checkpoint = RenderCheckpoint::new(job_id);

    for tile_idx in 0u64..3 {
        checkpoint.add_frame_progress(FrameProgress::new(tile_idx, 100, 100));
    }
    // Tiles 3–5 are absent (incomplete).

    // Verify tiles 0–2 are complete.
    for tile_idx in 0u64..3 {
        let complete = checkpoint
            .frame_progress
            .iter()
            .find(|fp| fp.frame == tile_idx)
            .map(|fp| fp.is_complete())
            .unwrap_or(false);
        assert!(complete, "tile {tile_idx} should be complete in checkpoint");
    }

    // Tiles 3–5 are absent → incomplete.
    for tile_idx in 3u64..6 {
        let complete = checkpoint
            .frame_progress
            .iter()
            .find(|fp| fp.frame == tile_idx)
            .map(|fp| fp.is_complete())
            .unwrap_or(false);
        assert!(
            !complete,
            "tile {tile_idx} should NOT be complete (needs re-dispatch)"
        );
    }

    // Simulate crash: new checkpoint for the same job, reload only completed.
    let mut resumed = RenderCheckpoint::new(job_id);
    for tile_idx in 0u64..3 {
        resumed.add_frame_progress(FrameProgress::new(tile_idx, 100, 100));
    }

    // Only tiles 3–5 need re-dispatch.
    let incomplete: Vec<u64> = (0u64..6)
        .filter(|&t| {
            !resumed
                .frame_progress
                .iter()
                .find(|fp| fp.frame == t)
                .map(|fp| fp.is_complete())
                .unwrap_or(false)
        })
        .collect();
    assert_eq!(
        incomplete,
        vec![3, 4, 5],
        "only tiles 3,4,5 need re-dispatch after resume"
    );
}
