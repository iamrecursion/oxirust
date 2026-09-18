// Copyright 2026 COOLJAPAN OU (Team KitaSan)
// SPDX-License-Identifier: Apache-2.0

//! Unit tests of the exact and the Barnes-Hut repulsion kernels.
//!
//! The reference is a plain double loop over ordered pairs that is written out
//! in this file and shares no code with the kernels.

use super::{
    MAX_DEPTH, MIN_DISTANCE, NONE, QuadTree, ROOT, approximate_repulsion, exact_repulsion,
};

/// Strength `repulsion * k^2` of the default configuration.
const STRENGTH: f64 = 2500.0;

/// Deterministic pseudo-random positions in `[-extent, extent]^2`.
fn random_positions(count: usize, seed: u64, extent: f64) -> Vec<[f64; 2]> {
    let mut state = seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1;
    let mut unit = move || {
        state = state
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        (state >> 11) as f64 / (1u64 << 53) as f64
    };
    (0..count)
        .map(|_| {
            let x = (2.0 * unit() - 1.0) * extent;
            let y = (2.0 * unit() - 1.0) * extent;
            [x, y]
        })
        .collect()
}

/// Reference: for every node the sum over all *other* nodes, each pair
/// evaluated in both directions.
fn reference_repulsion(positions: &[[f64; 2]], strength: f64) -> Vec<[f64; 2]> {
    positions
        .iter()
        .enumerate()
        .map(|(i, at)| {
            let mut force = [0.0_f64; 2];
            for (j, other) in positions.iter().enumerate() {
                if i == j {
                    continue;
                }
                let dx = at[0] - other[0];
                let dy = at[1] - other[1];
                let dist = dx.hypot(dy).max(MIN_DISTANCE);
                force[0] += strength / dist * (dx / dist);
                force[1] += strength / dist * (dy / dist);
            }
            force
        })
        .collect()
}

/// Largest absolute component difference between two force fields.
fn max_difference(a: &[[f64; 2]], b: &[[f64; 2]]) -> f64 {
    assert_eq!(a.len(), b.len());
    a.iter()
        .zip(b)
        .flat_map(|(x, y)| [(x[0] - y[0]).abs(), (x[1] - y[1]).abs()])
        .fold(0.0, f64::max)
}

/// Relative L2 error `|approx - exact| / |exact|` over all nodes together.
fn relative_error(approx: &[[f64; 2]], exact: &[[f64; 2]]) -> f64 {
    let mut error = 0.0_f64;
    let mut norm = 0.0_f64;
    for (a, e) in approx.iter().zip(exact) {
        error += (a[0] - e[0]).powi(2) + (a[1] - e[1]).powi(2);
        norm += e[0].powi(2) + e[1].powi(2);
    }
    (error / norm).sqrt()
}

// ---------------------------------------------------------------------------
// Exact kernel
// ---------------------------------------------------------------------------

#[test]
fn exact_kernel_matches_the_double_sum_and_balances_forces() {
    let positions = random_positions(60, 1, 150.0);
    let exact = exact_repulsion(&positions, STRENGTH);
    let reference = reference_repulsion(&positions, STRENGTH);
    assert!(max_difference(&exact, &reference) < 1e-9);
    // Newton's third law: the repulsions cancel.
    let total_x: f64 = exact.iter().map(|f| f[0]).sum();
    let total_y: f64 = exact.iter().map(|f| f[1]).sum();
    assert!(total_x.abs() < 1e-9 && total_y.abs() < 1e-9);
}

#[test]
fn exact_kernel_of_two_nodes_is_strength_over_distance() {
    let forces = exact_repulsion(&[[0.0, 0.0], [10.0, 0.0]], STRENGTH);
    // d = 10: magnitude 2500 / 10, pointing away from the other node.
    assert!((forces[0][0] + 250.0).abs() < 1e-12);
    assert!((forces[1][0] - 250.0).abs() < 1e-12);
    assert!(forces[0][1].abs() < 1e-12 && forces[1][1].abs() < 1e-12);
}

#[test]
fn distances_below_the_minimum_are_clamped() {
    // d = 0.001 is clamped to 0.01: magnitude strength / 0.01, whose direction
    // is still (dx, dy) / dist with the clamped dist, i.e. scaled by d / 0.01.
    let forces = exact_repulsion(&[[0.0, 0.0], [0.001, 0.0]], STRENGTH);
    let expected = STRENGTH / MIN_DISTANCE * (0.001 / MIN_DISTANCE);
    assert!((forces[1][0] - expected).abs() < 1e-9);
}

// ---------------------------------------------------------------------------
// Barnes-Hut: accuracy
// ---------------------------------------------------------------------------

#[test]
fn theta_zero_reproduces_the_exact_sum_on_random_nodes() {
    for seed in [3_u64, 4, 5] {
        let positions = random_positions(50, seed, 100.0);
        let tree = approximate_repulsion(&positions, 0.0, STRENGTH);
        let exact = exact_repulsion(&positions, STRENGTH);
        let reference = reference_repulsion(&positions, STRENGTH);
        assert!(max_difference(&tree, &exact) < 1e-9, "seed {seed}");
        assert!(max_difference(&tree, &reference) < 1e-9, "seed {seed}");
    }
}

#[test]
fn a_theta_that_is_not_positive_opens_every_cell() {
    let positions = random_positions(40, 8, 100.0);
    let exact = exact_repulsion(&positions, STRENGTH);
    for theta in [-0.5, -1.0e9, f64::NAN] {
        let tree = approximate_repulsion(&positions, theta, STRENGTH);
        assert!(max_difference(&tree, &exact) < 1e-9, "theta {theta}");
    }
}

#[test]
fn theta_half_is_within_five_percent_of_the_exact_sum() {
    for (count, seed) in [(50, 11_u64), (400, 12), (1000, 13)] {
        let positions = random_positions(count, seed, 200.0);
        let tree = approximate_repulsion(&positions, 0.5, STRENGTH);
        let exact = exact_repulsion(&positions, STRENGTH);
        let error = relative_error(&tree, &exact);
        assert!(error < 0.05, "{count} nodes: relative error {error}");
        // It really is an approximation, not a disguised exact sum.
        if count >= 400 {
            assert!(max_difference(&tree, &exact) > 1e-9, "{count} nodes");
        }
    }
}

#[test]
fn a_smaller_opening_angle_is_more_accurate() {
    let positions = random_positions(600, 21, 200.0);
    let exact = exact_repulsion(&positions, STRENGTH);
    let coarse = relative_error(&approximate_repulsion(&positions, 1.0, STRENGTH), &exact);
    let medium = relative_error(&approximate_repulsion(&positions, 0.5, STRENGTH), &exact);
    let fine = relative_error(&approximate_repulsion(&positions, 0.2, STRENGTH), &exact);
    assert!(fine < medium && medium < coarse, "{fine} {medium} {coarse}");
    assert!(fine < 0.01, "{fine}");
}

#[test]
fn clustered_layouts_stay_accurate() {
    // Three tight clusters far apart: the aggregates replace whole clusters.
    let mut positions = Vec::new();
    for (cx, cy, seed) in [(-500.0, 0.0, 31_u64), (500.0, 100.0, 32), (0.0, -700.0, 33)] {
        positions.extend(
            random_positions(80, seed, 20.0)
                .into_iter()
                .map(|p| [p[0] + cx, p[1] + cy]),
        );
    }
    let tree = approximate_repulsion(&positions, 0.5, STRENGTH);
    let exact = exact_repulsion(&positions, STRENGTH);
    let error = relative_error(&tree, &exact);
    assert!(error < 0.05, "relative error {error}");
}

#[test]
fn a_cell_that_holds_the_node_is_never_replaced_by_its_aggregate() {
    // Two nodes: the root holds both, and for theta = 100 the distance test
    // alone would accept it (width 1 < 100 * 0.5).  Replacing it would repel
    // the node with the mass of both nodes from the middle point.
    let positions = [[0.0, 0.0], [1.0, 0.0]];
    let forces = approximate_repulsion(&positions, 100.0, STRENGTH);
    // d = 1: magnitude 2500, away from the other node.  The aggregate of both
    // nodes would give 2 * 2500 / 0.5 = 10000 towards the wrong side.
    assert!((forces[0][0] + STRENGTH).abs() < 1e-9);
    assert!((forces[1][0] - STRENGTH).abs() < 1e-9);
    assert!(forces[0][1].abs() < 1e-12 && forces[1][1].abs() < 1e-12);
}

// ---------------------------------------------------------------------------
// Barnes-Hut: degenerate inputs
// ---------------------------------------------------------------------------

#[test]
fn empty_and_single_node_inputs() {
    assert!(approximate_repulsion(&[], 0.5, STRENGTH).is_empty());
    assert!(exact_repulsion(&[], STRENGTH).is_empty());
    assert_eq!(
        approximate_repulsion(&[[3.0, -4.0]], 0.5, STRENGTH),
        vec![[0.0, 0.0]]
    );
}

#[test]
fn coincident_nodes_exert_no_force_and_do_not_hang() {
    let origin = vec![[0.0, 0.0]; 5];
    assert_eq!(
        approximate_repulsion(&origin, 0.5, STRENGTH),
        vec![[0.0, 0.0]; 5]
    );
    // A bucket of coincident nodes next to other nodes behaves like the exact
    // sum, including the forces *on* the bucket's members.
    let mut positions = vec![[5.0, 5.0]; 4];
    positions.extend([[40.0, -10.0], [-30.0, 25.0], [12.0, 60.0]]);
    let tree = approximate_repulsion(&positions, 0.0, STRENGTH);
    let exact = exact_repulsion(&positions, STRENGTH);
    assert!(max_difference(&tree, &exact) < 1e-9);
    let coarse = approximate_repulsion(&positions, 0.5, STRENGTH);
    assert!(relative_error(&coarse, &exact) < 0.05);
}

#[test]
fn nodes_closer_than_the_depth_limit_share_a_leaf() {
    // 64 distinct nodes within 1e-12 of each other (far below 2^-48 of the
    // 1000-unit extent) and one far away.
    let mut positions: Vec<[f64; 2]> = (0..64)
        .map(|i| [f64::from(i) * 1e-14, f64::from(i % 7) * 1e-14])
        .collect();
    positions.push([1000.0, 1000.0]);
    let tree = approximate_repulsion(&positions, 0.0, STRENGTH);
    let exact = exact_repulsion(&positions, STRENGTH);
    assert!(max_difference(&tree, &exact) < 1e-6);
    // One chain of cells down to the depth limit for the cluster, not a cell
    // per node and level.
    let built = QuadTree::build(&positions);
    assert!(
        built.cells.len() < 2 * MAX_DEPTH,
        "{} cells",
        built.cells.len()
    );
}

#[test]
fn non_finite_positions_neither_panic_nor_hang() {
    let positions = [
        [0.0, 0.0],
        [f64::NAN, 1.0],
        [2.0, f64::INFINITY],
        [f64::NEG_INFINITY, f64::NEG_INFINITY],
        [3.0, 4.0],
        [f64::NAN, f64::NAN],
    ];
    for theta in [0.0, 0.5, 2.0] {
        assert_eq!(
            approximate_repulsion(&positions, theta, STRENGTH).len(),
            positions.len()
        );
    }
    // The finite nodes still repel each other when the others are not around.
    let finite = approximate_repulsion(&[[0.0, 0.0], [3.0, 4.0]], 0.5, STRENGTH);
    assert!((finite[1][0] - STRENGTH / 5.0 * 0.6).abs() < 1e-9);
}

#[test]
fn results_are_deterministic() {
    let positions = random_positions(300, 41, 120.0);
    let first = approximate_repulsion(&positions, 0.5, STRENGTH);
    let second = approximate_repulsion(&positions, 0.5, STRENGTH);
    assert_eq!(first, second);
}

// ---------------------------------------------------------------------------
// Tree structure
// ---------------------------------------------------------------------------

/// Indices of the nodes stored in the subtree below `cell`, by walking it.
fn bodies_below(tree: &QuadTree<'_>, cell: usize) -> Vec<usize> {
    let mut found = Vec::new();
    let mut pending = vec![cell];
    while let Some(index) = pending.pop() {
        let cell = &tree.cells[index];
        if cell.is_leaf() {
            let mut body = cell.first_body;
            while body != NONE {
                found.push(body);
                body = tree.next_body[body];
            }
        } else {
            pending.extend(cell.children.iter().copied().filter(|&c| c != NONE));
        }
    }
    found.sort_unstable();
    found
}

#[test]
fn every_node_is_stored_exactly_once_and_masses_add_up() {
    let positions = random_positions(500, 51, 300.0);
    let tree = QuadTree::build(&positions);
    let all: Vec<usize> = (0..positions.len()).collect();
    assert_eq!(bodies_below(&tree, ROOT), all);
    let root = &tree.cells[ROOT];
    assert_eq!(root.mass, 500.0);
    let sum_x: f64 = positions.iter().map(|p| p[0]).sum();
    let sum_y: f64 = positions.iter().map(|p| p[1]).sum();
    assert!((root.moment[0] - sum_x).abs() < 1e-8);
    assert!((root.moment[1] - sum_y).abs() < 1e-8);
    for (index, cell) in tree.cells.iter().enumerate() {
        let bodies = bodies_below(&tree, index);
        assert_eq!(cell.mass, bodies.len() as f64, "cell {index}");
        // Every node lies inside its cell (up to rounding of the geometry).
        for &body in &bodies {
            let p = positions[body];
            assert!(
                (p[0] - cell.center[0]).abs() <= cell.half * (1.0 + 1e-9),
                "{index}"
            );
            assert!(
                (p[1] - cell.center[1]).abs() <= cell.half * (1.0 + 1e-9),
                "{index}"
            );
        }
        // Children are created after their parent.
        for &child in cell.children.iter().filter(|&&c| c != NONE) {
            assert!(child > index);
        }
    }
}

#[test]
fn leaves_hold_a_single_node_unless_positions_coincide() {
    let mut positions = random_positions(100, 61, 80.0);
    positions.push(positions[7]);
    positions.push(positions[7]);
    let tree = QuadTree::build(&positions);
    let mut buckets = 0;
    for cell in tree.cells.iter().filter(|cell| cell.is_leaf()) {
        let mut body = cell.first_body;
        let mut size = 0;
        while body != NONE {
            size += 1;
            body = tree.next_body[body];
        }
        assert!(size <= 3);
        if size > 1 {
            buckets += 1;
            assert_eq!(size, 3, "only the three coincident nodes may share a leaf");
        }
    }
    assert_eq!(buckets, 1);
}
