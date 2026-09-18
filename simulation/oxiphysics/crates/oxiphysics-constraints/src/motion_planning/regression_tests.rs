// Copyright 2026 COOLJAPAN OU (Team KitaSan)
// SPDX-License-Identifier: Apache-2.0

//! Regression tests for the sampling-based planners.
//!
//! They pin the audit findings F0403 / F0404 (a PRM build that never
//! terminates and links nodes through obstacles), F0406 / F0407 (NaN panics in
//! the nearest-neighbour search, CBiRRT paths that run goal-to-start and are
//! never collision checked between their nodes).

use super::*;
use rand::SeedableRng;
use rand::rngs::StdRng;

/// Rejects every configuration.
struct NeverValidChecker;

impl ValidityChecker for NeverValidChecker {
    fn is_valid(&self, _q: &[f64]) -> bool {
        false
    }
}

/// Accepts a configuration only when its first coordinate is at least `min_x`.
struct HalfSpaceChecker {
    min_x: f64,
}

impl ValidityChecker for HalfSpaceChecker {
    fn is_valid(&self, q: &[f64]) -> bool {
        q[0] >= self.min_x
    }
}

/// Smallest distance from `obstacle` to the polyline through `points`,
/// sampled `samples_per_segment` times per segment (independent of the
/// planners' own collision checking).
fn polyline_clearance(
    points: &[Vec<f64>],
    obstacle: &SphereObstacle,
    samples_per_segment: usize,
) -> f64 {
    let mut clearance = f64::INFINITY;
    for pair in points.windows(2) {
        for k in 0..=samples_per_segment {
            let q = config_lerp(&pair[0], &pair[1], k as f64 / samples_per_segment as f64);
            clearance = clearance.min(config_distance(&q, &obstacle.center) - obstacle.radius);
        }
    }
    clearance
}

// ── PRM ──────────────────────────────────────────────────────────────────────

#[test]
fn prm_build_terminates_when_every_sample_is_rejected() {
    let cs = ConfigurationSpace::uniform(2, 1.0, 0.1);
    let mut prm = PrmPlanner::new(cs, 0.5, 10);
    assert_eq!(prm.max_attempts, 1000, "default budget is 100 * max_nodes");
    let mut rng = StdRng::seed_from_u64(1);
    let err = prm
        .build(&NeverValidChecker, &mut rng)
        .expect_err("a fully obstructed C-space cannot be filled");
    match err {
        Error::ConvergenceFailure {
            residual,
            iterations,
            ..
        } => {
            assert_eq!(iterations, 1000, "every attempt of the budget was used");
            assert!((residual - 10.0).abs() < 1e-12, "all 10 nodes are missing");
        }
        other => panic!("unexpected error: {other:?}"),
    }
    assert!(prm.nodes.is_empty());
}

#[test]
fn prm_build_reports_missing_nodes_and_keeps_partial_roadmap() {
    let cs = ConfigurationSpace::uniform(2, 1.0, 0.1);
    let mut prm = PrmPlanner::new(cs, 0.5, 50);
    prm.max_attempts = 200;
    // Only the strip x >= 0.9 (5 % of the box) is free: 200 samples cannot
    // yield 50 nodes.
    let checker = HalfSpaceChecker { min_x: 0.9 };
    let mut rng = StdRng::seed_from_u64(2);
    let err = prm
        .build(&checker, &mut rng)
        .expect_err("the attempt budget is too small for 50 nodes");
    match err {
        Error::ConvergenceFailure {
            residual,
            iterations,
            ..
        } => {
            assert_eq!(iterations, 200);
            assert!(!prm.nodes.is_empty(), "accepted samples are kept");
            assert!(prm.nodes.len() < 50);
            assert!((residual - (50 - prm.nodes.len()) as f64).abs() < 1e-12);
        }
        other => panic!("unexpected error: {other:?}"),
    }
    assert!(prm.nodes.iter().all(|n| n.config[0] >= 0.9));
}

#[test]
fn prm_build_returns_number_of_added_nodes_and_resumes() {
    let cs = ConfigurationSpace::uniform(2, 1.0, 0.1);
    let mut prm = PrmPlanner::new(cs, 0.6, 12);
    let mut rng = StdRng::seed_from_u64(3);
    assert_eq!(prm.build(&AlwaysValidChecker, &mut rng).ok(), Some(12));
    // The roadmap is already full: a second build adds nothing.
    assert_eq!(prm.build(&AlwaysValidChecker, &mut rng).ok(), Some(0));
    // Raising the target resumes from the existing nodes.
    prm.max_nodes = 20;
    assert_eq!(prm.build(&AlwaysValidChecker, &mut rng).ok(), Some(8));
    assert_eq!(prm.nodes.len(), 20);
}

#[test]
fn prm_default_attempt_budget_saturates() {
    let cs = ConfigurationSpace::uniform(2, 1.0, 0.1);
    let prm = PrmPlanner::new(cs, 0.5, usize::MAX);
    assert_eq!(prm.max_attempts, usize::MAX);
}

#[test]
fn prm_edges_never_cross_an_obstacle() {
    let cs = ConfigurationSpace::uniform(2, 1.0, 0.05);
    let mut prm = PrmPlanner::new(cs, 2.0, 0);
    let obstacle = SphereObstacle::new(vec![0.0, 0.0], 0.3);
    let left = prm.add_and_connect(vec![-0.6, 0.0], &obstacle);
    let right = prm.add_and_connect(vec![0.6, 0.0], &obstacle);
    // Both nodes are valid and 1.2 <= connection_radius apart, but the
    // straight line between them runs through the obstacle.
    assert!(prm.nodes[left].neighbours.is_empty());
    assert!(prm.nodes[right].neighbours.is_empty());
    assert!(prm.query(left, right).is_none());

    // A node above the obstacle sees both of them (the segments clear the
    // obstacle by 0.48 > 0.3) and reconnects the roadmap.
    let top = prm.add_and_connect(vec![0.0, 0.8], &obstacle);
    let neighbour_ids: Vec<usize> = prm.nodes[top].neighbours.iter().map(|&(i, _)| i).collect();
    assert_eq!(neighbour_ids, vec![left, right]);
    for &(_, cost) in &prm.nodes[top].neighbours {
        assert!(
            (cost - 1.0).abs() < 1e-12,
            "edge cost is the segment length"
        );
    }
    assert_eq!(prm.query(left, right), Some(vec![left, top, right]));
}

#[test]
fn prm_isolates_an_invalid_configuration() {
    let cs = ConfigurationSpace::uniform(2, 1.0, 0.05);
    let mut prm = PrmPlanner::new(cs, 2.0, 0);
    let obstacle = SphereObstacle::new(vec![0.0, 0.0], 0.3);
    let free = prm.add_and_connect(vec![0.8, 0.8], &obstacle);
    let inside = prm.add_and_connect(vec![0.1, 0.0], &obstacle);
    assert_eq!(
        prm.nodes.len(),
        2,
        "the node is inserted so its index is usable"
    );
    assert!(prm.nodes[inside].neighbours.is_empty());
    assert!(prm.nodes[free].neighbours.is_empty());
    assert!(prm.query(free, inside).is_none());
}

#[test]
fn prm_rejects_out_of_range_configuration_edges() {
    let cs = ConfigurationSpace::uniform(2, 1.0, 0.1);
    let mut prm = PrmPlanner::new(cs, 5.0, 0);
    let inside = prm.add_and_connect(vec![0.5, 0.5], &AlwaysValidChecker);
    let outside = prm.add_and_connect(vec![1.5, 0.5], &AlwaysValidChecker);
    assert!(prm.nodes[inside].neighbours.is_empty());
    assert!(prm.nodes[outside].neighbours.is_empty());
}

#[test]
fn prm_query_with_unknown_indices_returns_none() {
    let cs = ConfigurationSpace::uniform(2, 1.0, 0.1);
    let mut prm = PrmPlanner::new(cs, 1.0, 0);
    assert!(prm.query(0, 0).is_none(), "empty roadmap");
    let only = prm.add_and_connect(vec![0.0, 0.0], &AlwaysValidChecker);
    assert_eq!(prm.query(only, only), Some(vec![only]));
    assert!(prm.query(only, 7).is_none());
    assert!(prm.query(7, only).is_none());
}

#[test]
fn prm_roadmap_edges_are_collision_free_and_paths_detour() {
    let cs = ConfigurationSpace::uniform(2, 1.0, 0.05);
    let mut prm = PrmPlanner::new(cs, 0.35, 300);
    let obstacle = SphereObstacle::new(vec![0.0, 0.0], 0.4);
    let mut rng = StdRng::seed_from_u64(11);
    prm.build(&obstacle, &mut rng)
        .expect("a mostly free C-space fills quickly");
    assert_eq!(prm.nodes.len(), 300);

    // Every edge, sampled much more finely than the planner's resolution,
    // stays out of the obstacle (up to the discretisation depth ~1e-3).
    let mut edge_count = 0;
    for (i, node) in prm.nodes.iter().enumerate() {
        for &(j, cost) in &node.neighbours {
            edge_count += 1;
            assert!(cost <= 0.35 + 1e-12);
            let pair = [node.config.clone(), prm.nodes[j].config.clone()];
            let clearance = polyline_clearance(&pair, &obstacle, 400);
            assert!(
                clearance > -0.01,
                "edge {i}-{j} cuts {clearance} into the obstacle"
            );
        }
    }
    assert!(edge_count > 0, "the roadmap has edges");

    // A start and goal on opposite sides of the obstacle: the path detours.
    let start = prm.add_and_connect(vec![-0.9, 0.0], &obstacle);
    let goal = prm.add_and_connect(vec![0.9, 0.0], &obstacle);
    let ids = prm
        .query(start, goal)
        .expect("the free space around the obstacle is connected");
    let path: Vec<Vec<f64>> = ids.iter().map(|&i| prm.nodes[i].config.clone()).collect();
    assert!(polyline_clearance(&path, &obstacle, 400) > -0.01);
    let length: f64 = path.windows(2).map(|w| config_distance(&w[0], &w[1])).sum();
    assert!(
        length > 1.9,
        "straight line is 1.8 long but blocked; path length {length}"
    );
}

// ── RRT ──────────────────────────────────────────────────────────────────────

#[test]
fn rrt_edges_never_cut_through_an_obstacle() {
    // The extension step (0.5) exceeds the obstacle diameter (0.3): without a
    // segment check the tree hops straight across the obstacle.
    let cs = ConfigurationSpace::uniform(2, 1.0, 0.02);
    let obstacle = SphereObstacle::new(vec![0.0, 0.0], 0.15);
    let planner = RrtPlanner::new(cs, 0.5, 0.1, 5000, 0.1);
    let start = vec![-0.8, 0.0];
    let goal = vec![0.8, 0.0];
    let mut found = 0;
    for seed in 0..10 {
        let mut rng = StdRng::seed_from_u64(seed);
        if let Some(path) = planner.plan(start.clone(), goal.clone(), &obstacle, &mut rng) {
            found += 1;
            assert_eq!(path.first(), Some(&start));
            let last = path.last().expect("a path has a last waypoint");
            assert!(config_distance(last, &goal) < 0.1);
            let clearance = polyline_clearance(&path, &obstacle, 200);
            assert!(
                clearance > -0.01,
                "seed {seed}: path cuts {clearance} into the obstacle"
            );
        }
    }
    assert!(found >= 5, "RRT found a path for only {found}/10 seeds");
}

#[test]
fn rrt_rejects_non_finite_endpoints() {
    // A NaN goal used to be sampled (goal_bias = 1), pushed into the tree as a
    // NaN node and then made the nearest-node search panic on
    // `partial_cmp(NaN).expect(..)`.
    let cs = ConfigurationSpace::uniform(2, 1.0, 0.1);
    let planner = RrtPlanner::new(cs, 0.3, 1.0, 50, 0.1);
    let mut rng = StdRng::seed_from_u64(5);
    let ok = vec![0.0, 0.0];
    for bad in [
        vec![f64::NAN, 0.0],
        vec![0.0, f64::INFINITY],
        vec![f64::NEG_INFINITY, 0.0],
    ] {
        assert!(
            planner
                .plan(ok.clone(), bad.clone(), &AlwaysValidChecker, &mut rng)
                .is_none(),
            "goal {bad:?}"
        );
        assert!(
            planner
                .plan(bad.clone(), ok.clone(), &AlwaysValidChecker, &mut rng)
                .is_none(),
            "start {bad:?}"
        );
    }
}

#[test]
fn nearest_node_never_panics_on_nan_configurations() {
    let tree = vec![
        RrtNode::new(vec![f64::NAN, 0.0], usize::MAX, 0.0),
        RrtNode::new(vec![0.5, 0.0], 0, 0.0),
        RrtNode::new(vec![-0.1, 0.0], 1, 0.0),
    ];
    assert_eq!(nearest_node(&tree, &[0.0, 0.0]), Some(2));
    assert_eq!(nearest_node(&tree, &[1.0, 0.0]), Some(1));
    // NaN query: every distance is NaN, the first node wins deterministically.
    assert_eq!(nearest_node(&tree, &[f64::NAN, 0.0]), Some(0));
    assert_eq!(nearest_node(&[], &[0.0, 0.0]), None);
}

#[test]
fn rrt_rejects_an_invalid_start() {
    let cs = ConfigurationSpace::uniform(2, 1.0, 0.1);
    let planner = RrtPlanner::new(cs, 0.3, 0.2, 500, 0.1);
    let obstacle = SphereObstacle::new(vec![0.0, 0.0], 0.3);
    let mut rng = StdRng::seed_from_u64(6);
    // Start inside the obstacle.
    assert!(
        planner
            .plan(vec![0.0, 0.0], vec![0.8, 0.8], &obstacle, &mut rng)
            .is_none()
    );
    // Start outside the joint limits.
    assert!(
        planner
            .plan(
                vec![2.0, 0.0],
                vec![0.8, 0.8],
                &AlwaysValidChecker,
                &mut rng
            )
            .is_none()
    );
}

// ── C-space helpers ──────────────────────────────────────────────────────────

#[test]
fn sample_random_handles_a_locked_joint() {
    let cs = ConfigurationSpace::new(vec![0.5, -1.0], vec![0.5, 1.0], vec![0.1, 0.1]);
    let mut rng = StdRng::seed_from_u64(7);
    for _ in 0..20 {
        let q = cs.sample_random(&mut rng);
        assert!((q[0] - 0.5).abs() < 1e-15, "locked joint keeps its value");
        assert!((-1.0..1.0).contains(&q[1]));
    }
}

#[test]
fn segment_is_valid_checks_every_waypoint() {
    let cs = ConfigurationSpace::uniform(2, 1.0, 0.1);
    let obstacle = SphereObstacle::new(vec![0.0, 0.0], 0.3);
    // Both end points are free, the middle is blocked.
    assert!(!cs.segment_is_valid(&[-0.8, 0.0], &[0.8, 0.0], &obstacle));
    // A parallel segment clearing the obstacle is free.
    assert!(cs.segment_is_valid(&[-0.8, 0.6], &[0.8, 0.6], &obstacle));
    // An end point inside the obstacle invalidates the segment.
    assert!(!cs.segment_is_valid(&[0.1, 0.0], &[0.8, 0.8], &obstacle));
    assert!(!cs.segment_is_valid(&[0.8, 0.8], &[0.1, 0.0], &obstacle));
    // Out-of-range and NaN end points are never feasible.
    assert!(!cs.segment_is_valid(&[0.0, 0.5], &[1.5, 0.5], &AlwaysValidChecker));
    assert!(!cs.segment_is_valid(&[f64::NAN, 0.5], &[0.5, 0.5], &AlwaysValidChecker));
    // A degenerate segment is just its single configuration.
    assert!(cs.segment_is_valid(&[0.7, 0.7], &[0.7, 0.7], &obstacle));
    assert!(!cs.segment_is_valid(&[0.0, 0.0], &[0.0, 0.0], &obstacle));
}

// ── CBiRRT ───────────────────────────────────────────────────────────────────

fn unit_circle(q: &[f64]) -> (f64, Vec<f64>) {
    (
        q[0] * q[0] + q[1] * q[1] - 1.0,
        vec![2.0 * q[0], 2.0 * q[1]],
    )
}

fn circle_planner(step: f64, cspace_step: f64, max_iter: usize, goal_tol: f64) -> CbiRrtPlanner {
    CbiRrtPlanner::new(
        ConfigurationSpace::uniform(2, 2.0, cspace_step),
        step,
        max_iter,
        goal_tol,
        ManifoldProjector::new(1.0, 50, 1e-9),
    )
}

#[test]
fn cbirrt_assemble_path_runs_start_to_goal_for_either_active_tree() {
    let node = |x: f64, parent: usize| RrtNode::new(vec![x, 0.0], parent, 0.0);
    // Start tree: 0 -> 1.  Goal tree: 5 -> 4.
    let start_tree = vec![node(0.0, usize::MAX), node(1.0, 0)];
    let goal_tree = vec![node(5.0, usize::MAX), node(4.0, 0)];
    let bridge = vec![vec![2.5, 0.0]];
    let expected: Vec<Vec<f64>> = vec![
        vec![0.0, 0.0],
        vec![1.0, 0.0],
        vec![2.5, 0.0],
        vec![4.0, 0.0],
        vec![5.0, 0.0],
    ];

    // The start tree is the active one: the new node (1) is bridged to the
    // goal tree's node 1.
    let path = CbiRrtPlanner::assemble_path(&start_tree, 1, &bridge, &goal_tree, 1, true);
    assert_eq!(path, expected);

    // The goal tree is the active one: the bridge runs from the goal-side node
    // (4) towards the start-side node (1), and the raw walk goal -> start is
    // reversed.
    let bridge_rev = vec![vec![2.5, 0.0]];
    let path = CbiRrtPlanner::assemble_path(&goal_tree, 1, &bridge_rev, &start_tree, 1, false);
    assert_eq!(path, expected);
}

#[test]
fn cbirrt_paths_run_start_to_goal_on_the_manifold() {
    let planner = circle_planner(0.3, 0.05, 3000, 0.25);
    let start = vec![1.0, 0.0];
    let goal = vec![0.0, 1.0];
    let mut found = 0;
    for seed in 0..50 {
        let mut rng = StdRng::seed_from_u64(seed);
        let Some(path) = planner.plan(
            start.clone(),
            goal.clone(),
            &AlwaysValidChecker,
            unit_circle,
            &mut rng,
        ) else {
            continue;
        };
        found += 1;
        assert_eq!(
            path.first(),
            Some(&start),
            "seed {seed}: path must start at start"
        );
        assert_eq!(
            path.last(),
            Some(&goal),
            "seed {seed}: path must end at goal"
        );
        for q in &path {
            assert!(
                unit_circle(q).0.abs() < 1e-8,
                "seed {seed}: waypoint {q:?} is off the manifold"
            );
            assert!(planner.cspace.in_bounds(q));
        }
        for pair in path.windows(2) {
            let gap = config_distance(&pair[0], &pair[1]);
            assert!(
                gap <= 0.25 + 1e-9,
                "seed {seed}: consecutive waypoints {gap} apart"
            );
        }
    }
    assert!(found >= 40, "CBiRRT found a path for only {found}/50 seeds");
}

#[test]
fn cbirrt_paths_never_cut_through_an_obstacle() {
    // A small obstacle sits on the short arc between start and goal.  The
    // extension step (0.4) is far larger than the obstacle (radius 0.05), so
    // without segment checks a chord hops across it.
    let angle = std::f64::consts::FRAC_PI_4;
    let obstacle = SphereObstacle::new(vec![angle.cos(), angle.sin()], 0.05);
    let planner = circle_planner(0.4, 0.02, 4000, 0.2);
    let start = vec![1.0, 0.0];
    let goal = vec![0.0, 1.0];
    let mut found = 0;
    for seed in 0..10 {
        let mut rng = StdRng::seed_from_u64(100 + seed);
        let Some(path) = planner.plan(
            start.clone(),
            goal.clone(),
            &obstacle,
            unit_circle,
            &mut rng,
        ) else {
            continue;
        };
        found += 1;
        assert_eq!(path.first(), Some(&start));
        assert_eq!(path.last(), Some(&goal));
        let clearance = polyline_clearance(&path, &obstacle, 100);
        assert!(
            clearance > -0.005,
            "seed {seed}: path cuts {clearance} into the obstacle"
        );
    }
    assert!(found >= 3, "CBiRRT found a path for only {found}/10 seeds");
}

#[test]
fn cbirrt_rejects_infeasible_endpoints() {
    let planner = circle_planner(0.3, 0.05, 200, 0.25);
    let mut rng = StdRng::seed_from_u64(9);
    let obstacle = SphereObstacle::new(vec![1.0, 0.0], 0.1);
    // Start inside the obstacle.
    assert!(
        planner
            .plan(
                vec![1.0, 0.0],
                vec![0.0, 1.0],
                &obstacle,
                unit_circle,
                &mut rng
            )
            .is_none()
    );
    // Start off the manifold.
    assert!(
        planner
            .plan(
                vec![1.5, 0.0],
                vec![0.0, 1.0],
                &AlwaysValidChecker,
                unit_circle,
                &mut rng
            )
            .is_none()
    );
    // Goal outside the joint limits.
    assert!(
        planner
            .plan(
                vec![1.0, 0.0],
                vec![0.0, 2.5],
                &AlwaysValidChecker,
                unit_circle,
                &mut rng
            )
            .is_none()
    );
}

#[test]
fn cbirrt_nearest_idx_ranks_nan_nodes_farthest() {
    let tree = vec![
        RrtNode::new(vec![f64::NAN, 0.0], usize::MAX, 0.0),
        RrtNode::new(vec![3.0, 0.0], 0, 0.0),
        RrtNode::new(vec![1.0, 0.0], 1, 0.0),
    ];
    assert_eq!(CbiRrtPlanner::nearest_idx(&tree, &[0.0, 0.0]), 2);
    // A NaN query makes every distance NaN: no panic, deterministic answer.
    assert_eq!(CbiRrtPlanner::nearest_idx(&tree, &[f64::NAN, 0.0]), 0);
    assert_eq!(CbiRrtPlanner::nearest_idx(&[], &[0.0, 0.0]), 0);
}
