// Copyright 2026 COOLJAPAN OU (Team KitaSan)
// SPDX-License-Identifier: Apache-2.0

//! Tests of the RRT / RRT* planner of [`MotionPlanning`].
//!
//! The parent selection and rewiring are tested on hand-built trees with exact
//! numbers.  The planner itself is run with seeded generators so that every
//! test is reproducible; its paths are checked against exact segment–sphere
//! geometry, not only against the planner's own sampled collision check.

use super::*;
use rand::SeedableRng;
use rand::rngs::StdRng;

/// Fraction of an obstacle's radius that a path edge must keep clear.
///
/// The planner samples an edge at [`EDGE_CHECK_STEPS`] + 1 points, so an edge
/// may graze an obstacle between two samples by a fraction of a percent of the
/// radius; an edge cut through the obstacle penetrates by a large fraction.
const CLEARANCE_FRACTION: f64 = 0.97;

fn sphere(x: f64, y: f64, radius: f64) -> PlanningObstacle {
    PlanningObstacle {
        center: vec![x, y],
        radius,
    }
}

/// Planner in the unit square.
fn unit_square_planner(
    step: f64,
    max_iters: usize,
    goal_tol: f64,
    use_rrt_star: bool,
    obstacles: Vec<PlanningObstacle>,
) -> MotionPlanning {
    MotionPlanning::new(
        2,
        vec![0.0; 2],
        vec![1.0; 2],
        step,
        max_iters,
        goal_tol,
        use_rrt_star,
    )
    .with_obstacles(obstacles)
}

/// The scenario of the original defect: a small sphere sits on the straight
/// line between start and goal.  Its diameter (0.12) is smaller than the
/// rewiring radius (0.15), so neighbours on opposite sides of the sphere are
/// candidate parents of each other.
fn sphere_between_start_and_goal(use_rrt_star: bool) -> MotionPlanning {
    unit_square_planner(
        0.05,
        1_500,
        0.08,
        use_rrt_star,
        vec![sphere(0.5, 0.5, 0.06)],
    )
}

const START: [f64; 2] = [0.1, 0.5];
const GOAL: [f64; 2] = [0.9, 0.5];

/// Distance from `center` to the segment `a`–`b`.
fn distance_to_segment(a: &[f64], b: &[f64], center: &[f64]) -> f64 {
    let ab: Vec<f64> = b.iter().zip(a).map(|(q, p)| q - p).collect();
    let ac: Vec<f64> = center.iter().zip(a).map(|(c, p)| c - p).collect();
    let length_sq: f64 = ab.iter().map(|x| x * x).sum();
    let t = if length_sq > 0.0 {
        let along: f64 = ab.iter().zip(&ac).map(|(x, y)| x * y).sum();
        (along / length_sq).clamp(0.0, 1.0)
    } else {
        0.0
    };
    let closest: Vec<f64> = a.iter().zip(&ab).map(|(p, d)| p + t * d).collect();
    config_dist(&closest, center)
}

/// Require a complete path: it starts at `start`, ends at `goal` and every
/// edge is collision-free, both for the planner's sampled check and for the
/// exact distance to every obstacle.
fn assert_valid_path(
    label: &str,
    planner: &MotionPlanning,
    path: &[Vec<f64>],
    start: &[f64],
    goal: &[f64],
) {
    assert!(path.len() >= 2, "{label}: no path found");
    assert_eq!(path[0], start, "{label}: path does not start at the start");
    assert_eq!(
        path[path.len() - 1],
        goal,
        "{label}: path does not end at the goal"
    );
    for pair in path.windows(2) {
        assert!(
            planner.is_segment_collision_free(&pair[0], &pair[1], EDGE_CHECK_STEPS),
            "{label}: edge {:?} -> {:?} fails the planner's collision check",
            pair[0],
            pair[1]
        );
        for obstacle in &planner.obstacles {
            let clearance = distance_to_segment(&pair[0], &pair[1], &obstacle.center);
            assert!(
                clearance >= CLEARANCE_FRACTION * obstacle.radius,
                "{label}: edge {:?} -> {:?} passes {clearance} from the centre of an obstacle of radius {}",
                pair[0],
                pair[1],
                obstacle.radius
            );
        }
    }
}

fn path_length(path: &[Vec<f64>]) -> f64 {
    path.windows(2).map(|p| config_dist(&p[0], &p[1])).sum()
}

// ---- parent selection and rewiring on hand-built trees ---------------------

#[test]
fn choose_parent_skips_a_cheaper_parent_behind_an_obstacle() {
    // Reaching q_new = (0.5, 0) from the root is cheapest (0.5) but the edge
    // crosses the obstacle at (0.25, 0); the other candidates are free.
    let blocked = unit_square_planner(0.5, 10, 0.05, true, vec![sphere(0.25, 0.0, 0.05)]);
    let open = unit_square_planner(0.5, 10, 0.05, true, vec![]);
    let q_new = [0.5, 0.0];

    let mut tree = PlannerTree::new(vec![0.0, 0.0]);
    let p = tree.add_node(vec![0.25, 0.25], 0); // cost through it: 0.7071
    let q = tree.add_node(vec![0.3, 0.2], 0); // cost through it: 0.6434
    let near = [0, p, q];

    assert_eq!(open.choose_parent(&tree, &near, p, &q_new), 0);
    assert!(!blocked.is_segment_collision_free(&[0.0, 0.0], &q_new, EDGE_CHECK_STEPS));
    assert_eq!(
        blocked.choose_parent(&tree, &near, p, &q_new),
        q,
        "the cheapest collision-free candidate must win"
    );
    // With only the nearest node as a candidate it is used as is.
    assert_eq!(blocked.choose_parent(&tree, &[p], p, &q_new), p);
}

/// Tree for the rewiring tests: the root, a detour node `w` (cost 0.5), a node
/// `m` reached through it (cost 1.0) and `m`'s child `d` (cost 1.3).  A new node
/// at (0.3, 0.3) could reach `m` for 0.7071.
fn detour_tree() -> (PlannerTree, [usize; 3]) {
    let mut tree = PlannerTree::new(vec![0.0, 0.0]);
    let w = tree.add_node(vec![0.0, 0.5], 0);
    let m = tree.add_node(vec![0.5, 0.5], w);
    let d = tree.add_node(vec![0.5, 0.8], m);
    (tree, [w, m, d])
}

#[test]
fn rewire_reroutes_a_neighbour_and_lowers_its_descendants() {
    let mut planner = unit_square_planner(0.2, 10, 0.05, true, vec![]);
    planner.rewire_radius = 0.5;
    let (mut tree, [w, m, d]) = detour_tree();
    assert!((tree.nodes[m].cost - 1.0).abs() < 1e-12);
    assert!((tree.nodes[d].cost - 1.3).abs() < 1e-12);

    let q_new = vec![0.3, 0.3];
    let near = planner.near_indices(&tree.nodes, &q_new);
    // d is 0.5385 away, outside the radius: it can only profit by propagation.
    assert_eq!(near, vec![0, w, m]);
    let x = tree.add_node(q_new, 0);
    planner.rewire(&mut tree, &near, x);

    let through_x = 0.3_f64.hypot(0.3) + 0.2_f64.hypot(0.2);
    assert_eq!(tree.nodes[m].parent, x);
    assert!((tree.nodes[m].cost - through_x).abs() < 1e-12);
    assert!(
        (tree.nodes[d].cost - (through_x + 0.3)).abs() < 1e-12,
        "the child of a rewired node must follow its cost, got {}",
        tree.nodes[d].cost
    );
    assert_eq!(tree.nodes[d].parent, m);
    // The detour node keeps its (cheaper) direct parent and loses its child.
    assert_eq!(tree.nodes[w].parent, 0);
    assert!((tree.nodes[w].cost - 0.5).abs() < 1e-12);
    assert!(tree.children[w].is_empty());
    assert_eq!(tree.children[x], vec![m]);
    assert_eq!(tree.children[m], vec![d]);
}

#[test]
fn rewire_does_not_cut_through_an_obstacle() {
    let mut planner = unit_square_planner(0.2, 10, 0.05, true, vec![sphere(0.4, 0.4, 0.03)]);
    planner.rewire_radius = 0.5;
    let (mut tree, [w, m, d]) = detour_tree();

    let q_new = vec![0.3, 0.3];
    let near = planner.near_indices(&tree.nodes, &q_new);
    let x = tree.add_node(q_new, 0);
    assert!(
        !planner.is_segment_collision_free(&tree.nodes[x].config, &tree.nodes[m].config, 10),
        "the obstacle blocks the shortcut"
    );
    planner.rewire(&mut tree, &near, x);

    assert_eq!(
        tree.nodes[m].parent, w,
        "m must keep its collision-free parent"
    );
    assert!((tree.nodes[m].cost - 1.0).abs() < 1e-12);
    assert!((tree.nodes[d].cost - 1.3).abs() < 1e-12);
    assert!(tree.children[x].is_empty());
}

#[test]
fn rewire_never_moves_an_ancestor_below_its_descendant() {
    // Chain root -> a -> b -> n.  Inflate a's cost so that the cost test alone
    // would accept rewiring a through n; that would close a cycle.
    let planner = unit_square_planner(0.2, 10, 0.05, true, vec![]);
    let mut tree = PlannerTree::new(vec![0.0, 0.0]);
    let a = tree.add_node(vec![0.3, 0.0], 0);
    let b = tree.add_node(vec![0.3, 0.1], a);
    let n = tree.add_node(vec![0.32, 0.1], b);
    assert!(tree.is_ancestor(a, n) && tree.is_ancestor(0, n) && tree.is_ancestor(b, n));
    assert!(!tree.is_ancestor(n, a) && !tree.is_ancestor(n, n));
    tree.nodes[a].cost = 10.0;

    planner.rewire(&mut tree, &[a, b], n);

    assert_eq!(tree.nodes[a].parent, 0, "an ancestor must keep its parent");
    assert_eq!(tree.nodes[b].parent, a);
    let mut steps = 0;
    let mut node = n;
    while node != usize::MAX {
        node = tree.nodes[node].parent;
        steps += 1;
        assert!(steps <= tree.nodes.len(), "the parent chain of n loops");
    }
}

#[test]
fn reparent_keeps_the_child_lists_and_costs_consistent() {
    let (mut tree, [w, m, d]) = detour_tree();
    // Deeper descendants follow too: d gets a child of its own.
    let leaf = tree.add_node(vec![0.9, 0.8], d);
    assert!((tree.nodes[leaf].cost - 1.7).abs() < 1e-12);

    tree.reparent(m, 0);

    assert_eq!(tree.nodes[m].parent, 0);
    assert!(tree.children[w].is_empty());
    assert_eq!(tree.children[0], vec![w, m]);
    let m_cost = 0.5_f64.hypot(0.5);
    assert!((tree.nodes[m].cost - m_cost).abs() < 1e-12);
    assert!((tree.nodes[d].cost - (m_cost + 0.3)).abs() < 1e-12);
    assert!((tree.nodes[leaf].cost - (m_cost + 0.7)).abs() < 1e-12);
}

// ---- complete planner runs --------------------------------------------------

#[test]
fn rrt_star_paths_never_cross_the_obstacle_between_start_and_goal() {
    let planner = sphere_between_start_and_goal(true);
    for seed in 0..12 {
        let mut rng = StdRng::seed_from_u64(seed);
        let path = planner.rrt_with_rng(&START, &GOAL, &mut rng);
        assert_valid_path(&format!("rrt* seed {seed}"), &planner, &path, &START, &GOAL);
    }
}

#[test]
fn rrt_paths_never_cross_the_obstacle_between_start_and_goal() {
    let planner = sphere_between_start_and_goal(false);
    for seed in 0..12 {
        let mut rng = StdRng::seed_from_u64(seed);
        let path = planner.rrt_with_rng(&START, &GOAL, &mut rng);
        assert_valid_path(&format!("rrt seed {seed}"), &planner, &path, &START, &GOAL);
    }
}

#[test]
fn the_edge_to_the_goal_is_collision_checked_too() {
    // The goal tolerance (0.35) is wide enough for a node to the left of the
    // obstacle to "reach" the goal across it.  Only nodes that can actually
    // see the goal may end the search.
    for use_rrt_star in [false, true] {
        let planner = unit_square_planner(
            0.05,
            1_500,
            0.35,
            use_rrt_star,
            vec![sphere(0.78, 0.5, 0.1)],
        );
        for seed in 0..12 {
            let mut rng = StdRng::seed_from_u64(seed);
            let path = planner.rrt_with_rng(&START, &GOAL, &mut rng);
            assert_valid_path(
                &format!("goal hop, rrt*={use_rrt_star}, seed {seed}"),
                &planner,
                &path,
                &START,
                &GOAL,
            );
        }
    }
}

#[test]
fn rrt_star_tree_stays_a_consistent_collision_free_tree() {
    let planner = sphere_between_start_and_goal(true);
    let mut rewired_runs = 0;
    for seed in 0..6 {
        let mut rng = StdRng::seed_from_u64(100 + seed);
        let tree = planner
            .grow_tree(&START, &GOAL, &mut rng)
            .expect("start and goal are free");
        let nodes = &tree.nodes;
        assert_eq!(nodes[0].parent, usize::MAX);
        assert_eq!(nodes[0].cost, 0.0);
        let mut child_links = 0;
        let mut rewired = 0;
        for (index, node) in nodes.iter().enumerate().skip(1) {
            assert!(
                node.parent < nodes.len(),
                "node {index} has no valid parent"
            );
            let parent = &nodes[node.parent];
            let expected = parent.cost + config_dist(&parent.config, &node.config);
            assert!(
                (node.cost - expected).abs() < 1e-9,
                "seed {seed}: node {index} costs {} but its parent chain gives {expected}",
                node.cost
            );
            assert!(
                planner.is_segment_collision_free(&parent.config, &node.config, EDGE_CHECK_STEPS),
                "seed {seed}: the edge into node {index} is blocked"
            );
            assert_eq!(
                tree.children[node.parent]
                    .iter()
                    .filter(|&&c| c == index)
                    .count(),
                1,
                "node {index} must be listed once in its parent's children"
            );
            child_links += 1;
            if node.parent > index {
                rewired += 1;
            }
            // Walking up must reach the root.
            let mut current = index;
            let mut steps = 0;
            while current != 0 {
                current = nodes[current].parent;
                steps += 1;
                assert!(
                    steps <= nodes.len(),
                    "seed {seed}: cycle above node {index}"
                );
            }
        }
        assert_eq!(
            tree.children.iter().map(Vec::len).sum::<usize>(),
            child_links,
            "every child link must belong to a node"
        );
        for &candidate in &tree.goal_candidates {
            assert!(config_dist(&nodes[candidate].config, &GOAL) < planner.goal_tol);
            assert!(planner.is_segment_collision_free(
                &nodes[candidate].config,
                &GOAL,
                EDGE_CHECK_STEPS
            ));
        }
        if rewired > 0 {
            rewired_runs += 1;
        }
    }
    assert!(
        rewired_runs > 0,
        "no run rewired a node: the rewiring step is not exercised"
    );
}

#[test]
fn plain_rrt_never_rewires() {
    let planner = sphere_between_start_and_goal(false);
    let mut rng = StdRng::seed_from_u64(7);
    let tree = planner
        .grow_tree(&START, &GOAL, &mut rng)
        .expect("start and goal are free");
    assert!(
        tree.nodes
            .iter()
            .enumerate()
            .skip(1)
            .all(|(index, node)| node.parent < index),
        "plain RRT attaches every node to an older one"
    );
    assert_eq!(
        tree.goal_candidates.len(),
        1,
        "plain RRT stops at the first hit"
    );
}

#[test]
fn rrt_star_returns_the_cheapest_goal_connection() {
    let planner = unit_square_planner(0.05, 1_500, 0.15, true, vec![sphere(0.5, 0.5, 0.06)]);
    let mut rng = StdRng::seed_from_u64(3);
    let tree = planner
        .grow_tree(&START, &GOAL, &mut rng)
        .expect("start and goal are free");
    assert!(
        tree.goal_candidates.len() > 1,
        "the wide goal region collects several connections"
    );
    let total = |node: usize| tree.nodes[node].cost + config_dist(&tree.nodes[node].config, &GOAL);
    let best = tree
        .best_goal_node(&GOAL)
        .expect("a goal connection exists");
    for &candidate in &tree.goal_candidates {
        assert!(total(best) <= total(candidate) + 1e-12);
    }
    let path = tree.path_to(best, &GOAL);
    assert!((path_length(&path) - total(best)).abs() < 1e-9);
}

#[test]
fn rrt_star_paths_are_shorter_than_plain_rrt_paths() {
    let mut rrt_total = 0.0;
    let mut star_total = 0.0;
    let seeds = 10;
    for seed in 0..seeds {
        for (use_rrt_star, total) in [(false, &mut rrt_total), (true, &mut star_total)] {
            let planner = unit_square_planner(0.05, 1_500, 0.08, use_rrt_star, vec![]);
            let mut rng = StdRng::seed_from_u64(200 + seed);
            let path = planner.rrt_with_rng(&[0.05, 0.05], &[0.95, 0.95], &mut rng);
            assert_valid_path("free space", &planner, &path, &[0.05, 0.05], &[0.95, 0.95]);
            *total += path_length(&path);
        }
    }
    let direct = 0.9_f64.hypot(0.9);
    let mean_rrt = rrt_total / seeds as f64;
    let mean_star = star_total / seeds as f64;
    assert!(
        mean_star < 0.95 * mean_rrt,
        "RRT* mean length {mean_star} should clearly beat RRT's {mean_rrt}"
    );
    assert!(
        mean_star < 1.15 * direct,
        "RRT* mean length {mean_star} should approach the straight line {direct}"
    );
}

#[test]
fn start_that_sees_the_goal_gives_the_direct_path() {
    for use_rrt_star in [false, true] {
        let planner = unit_square_planner(0.05, 100, 0.1, use_rrt_star, vec![]);
        let mut rng = StdRng::seed_from_u64(1);
        let path = planner.rrt_with_rng(&[0.5, 0.5], &[0.55, 0.5], &mut rng);
        assert_eq!(path, vec![vec![0.5, 0.5], vec![0.55, 0.5]]);
    }
    // An obstacle between start and goal forbids the shortcut.
    let blocked = unit_square_planner(0.05, 100, 0.2, false, vec![sphere(0.525, 0.5, 0.02)]);
    let mut rng = StdRng::seed_from_u64(1);
    let path = blocked.rrt_with_rng(&[0.5, 0.5], &[0.55, 0.5], &mut rng);
    assert_ne!(path, vec![vec![0.5, 0.5], vec![0.55, 0.5]]);
}

#[test]
fn unusable_endpoints_give_no_path() {
    for use_rrt_star in [false, true] {
        let planner =
            unit_square_planner(0.05, 200, 0.08, use_rrt_star, vec![sphere(0.5, 0.5, 0.1)]);
        let mut rng = StdRng::seed_from_u64(5);
        // Start inside the obstacle.
        assert!(
            planner
                .rrt_with_rng(&[0.5, 0.5], &GOAL, &mut rng)
                .is_empty()
        );
        // Goal inside the obstacle.
        assert!(
            planner
                .rrt_with_rng(&START, &[0.52, 0.5], &mut rng)
                .is_empty()
        );
        // Wrong dimension and non-finite components.
        assert!(planner.rrt_with_rng(&[0.1], &GOAL, &mut rng).is_empty());
        assert!(
            planner
                .rrt_with_rng(&START, &[0.9, 0.5, 0.0], &mut rng)
                .is_empty()
        );
        assert!(
            planner
                .rrt_with_rng(&[f64::NAN, 0.5], &GOAL, &mut rng)
                .is_empty()
        );
        assert!(
            planner
                .rrt_with_rng(&START, &[f64::INFINITY, 0.5], &mut rng)
                .is_empty()
        );
    }
}

#[test]
fn a_sealed_goal_gives_no_path() {
    // A ring of overlapping spheres around the goal: no edge can reach it.
    let mut ring = Vec::new();
    for k in 0..16 {
        let angle = std::f64::consts::TAU * f64::from(k) / 16.0;
        ring.push(sphere(
            0.8 + 0.12 * angle.cos(),
            0.5 + 0.12 * angle.sin(),
            0.06,
        ));
    }
    for use_rrt_star in [false, true] {
        let planner = unit_square_planner(0.05, 400, 0.2, use_rrt_star, ring.clone());
        let mut rng = StdRng::seed_from_u64(9);
        assert!(
            planner
                .rrt_with_rng(&START, &[0.8, 0.5], &mut rng)
                .is_empty()
        );
    }
}

#[test]
fn path_to_appends_the_goal_unless_the_node_already_is_the_goal() {
    let mut tree = PlannerTree::new(vec![0.0, 0.0]);
    let a = tree.add_node(vec![0.1, 0.0], 0);
    let b = tree.add_node(vec![0.2, 0.0], a);
    assert_eq!(
        tree.path_to(b, &[0.25, 0.0]),
        vec![
            vec![0.0, 0.0],
            vec![0.1, 0.0],
            vec![0.2, 0.0],
            vec![0.25, 0.0]
        ]
    );
    // A node steered exactly onto the goal must not be repeated.
    assert_eq!(
        tree.path_to(b, &[0.2, 0.0]),
        vec![vec![0.0, 0.0], vec![0.1, 0.0], vec![0.2, 0.0]]
    );
    assert_eq!(tree.path_to(0, &[0.0, 0.0]), vec![vec![0.0, 0.0]]);
}

#[test]
fn seeded_plans_are_reproducible_and_have_no_repeated_waypoints() {
    for use_rrt_star in [false, true] {
        let planner = sphere_between_start_and_goal(use_rrt_star);
        let first = planner.rrt_with_rng(&START, &GOAL, &mut StdRng::seed_from_u64(42));
        let second = planner.rrt_with_rng(&START, &GOAL, &mut StdRng::seed_from_u64(42));
        assert!(!first.is_empty());
        assert_eq!(first, second);
        for seed in 0..12 {
            let path = planner.rrt_with_rng(&START, &GOAL, &mut StdRng::seed_from_u64(seed));
            for pair in path.windows(2) {
                assert_ne!(
                    pair[0], pair[1],
                    "seed {seed}: repeated waypoint {:?}",
                    pair[0]
                );
            }
        }
    }
}

#[test]
fn the_unseeded_entry_point_finds_a_collision_free_path() {
    let planner = sphere_between_start_and_goal(true);
    let path = planner.rrt(&START, &GOAL);
    assert_valid_path("rrt()", &planner, &path, &START, &GOAL);
}
