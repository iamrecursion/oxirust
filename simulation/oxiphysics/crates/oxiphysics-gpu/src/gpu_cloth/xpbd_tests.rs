// Copyright 2026 COOLJAPAN OU (Team KitaSan)
// SPDX-License-Identifier: Apache-2.0

//! Tests of the XPBD cloth solver: accumulated Lagrange multipliers (the stretch
//! of a hanging cloth must not depend on the iteration count), the configurable
//! gravity, the edge damping, and the dihedral-angle bending constraints with
//! their analytic gradients.

use super::*;
use std::f64::consts::PI;

/// Deterministic uniform samples in `[0, 1)` (splitmix64).
struct Samples(u64);

impl Samples {
    fn next_f64(&mut self) -> f64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^= z >> 31;
        (z >> 11) as f64 / (1u64 << 53) as f64
    }
}

/// Four random hinge vertices in `[-1, 1]³`, optionally mirrored in `z` (which
/// flips the side of the fold).
fn random_hinge(samples: &mut Samples, mirror: bool) -> [[f64; 3]; 4] {
    let mut points = [[0.0; 3]; 4];
    for point in &mut points {
        for component in point.iter_mut() {
            *component = 2.0 * samples.next_f64() - 1.0;
        }
        if mirror {
            point[2] = -point[2];
        }
    }
    points
}

fn vertices(points: &[[f64; 3]; 4]) -> Vec<ClothVertex> {
    points
        .iter()
        .map(|&p| ClothVertex::new(p, 1.0, false))
        .collect()
}

// ── dihedral angle and its gradient ───────────────────────────────────────

/// The hinge with edge `(−1,0,0)–(1,0,0)`, a wing at `(0, 1, 0)` and the other
/// wing folded out of the plane by `fold` radians (`fold = 0` is flat).
fn folded_hinge(fold: f64) -> [[f64; 3]; 4] {
    [
        [0.0, 1.0, 0.0],
        [-1.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, -fold.cos(), fold.sin()],
    ]
}

#[test]
fn dihedral_angle_equals_the_fold_angle_with_full_precision() {
    for fold in [0.0, 1.0e-9, 1.0e-6, 0.3, PI / 2.0, 2.5, PI - 1.0e-6] {
        let [p0, p1, p2, p3] = folded_hinge(fold);
        let angle = dihedral_angle(p0, p1, p2, p3);
        assert!(
            (angle - fold).abs() < 1.0e-12,
            "fold {fold}: dihedral angle {angle}"
        );
        let (theta, _) = dihedral_angle_gradient(p0, p1, p2, p3).expect("non-degenerate hinge");
        assert!((theta - fold).abs() < 1.0e-12, "fold {fold}: theta {theta}");
    }
}

#[test]
fn dihedral_angle_gradient_matches_finite_differences() {
    let mut samples = Samples(1);
    let step = 1.0e-6;
    let mut checked = 0;
    for trial in 0..300 {
        let points = random_hinge(&mut samples, trial % 2 == 1);
        let [p0, p1, p2, p3] = points;
        let (theta, gradients) = dihedral_angle_gradient(p0, p1, p2, p3).expect("hinge");
        assert!(
            (theta - dihedral_angle(p0, p1, p2, p3)).abs() < 1.0e-12,
            "trial {trial}"
        );
        // The unsigned angle has a kink at 0 and π, and a wing vertex close to the
        // edge line makes the gradient large and the finite difference poor.
        if !(0.05..PI - 0.05).contains(&theta) || gradients.iter().any(|g| length3(*g) > 10.0) {
            continue;
        }
        checked += 1;
        for vertex in 0..4 {
            for axis in 0..3 {
                let mut plus = points;
                let mut minus = points;
                plus[vertex][axis] += step;
                minus[vertex][axis] -= step;
                let numeric = (dihedral_angle(plus[0], plus[1], plus[2], plus[3])
                    - dihedral_angle(minus[0], minus[1], minus[2], minus[3]))
                    / (2.0 * step);
                let analytic = gradients[vertex][axis];
                assert!(
                    (numeric - analytic).abs() < 1.0e-6 * (1.0 + analytic.abs()),
                    "trial {trial}, vertex {vertex}, axis {axis}: {numeric} vs {analytic}"
                );
            }
        }
    }
    assert!(checked > 150, "only {checked} regular hinges");
}

#[test]
fn dihedral_angle_gradient_has_no_net_force_or_torque() {
    let mut samples = Samples(2);
    for trial in 0..100 {
        let points = random_hinge(&mut samples, trial % 2 == 0);
        let [p0, p1, p2, p3] = points;
        let (_, gradients) = dihedral_angle_gradient(p0, p1, p2, p3).expect("hinge");
        let mut force = [0.0; 3];
        let mut torque = [0.0; 3];
        for (point, gradient) in points.iter().zip(&gradients) {
            force = add3(force, *gradient);
            torque = add3(torque, cross3(*point, *gradient));
        }
        let scale: f64 = points
            .iter()
            .zip(&gradients)
            .map(|(point, gradient)| length3(*point) * length3(*gradient))
            .sum();
        assert!(
            length3(force) < 1.0e-11 * (1.0 + scale),
            "trial {trial}: net {force:?}"
        );
        assert!(
            length3(torque) < 1.0e-11 * (1.0 + scale),
            "trial {trial}: torque {torque:?}"
        );
    }
}

#[test]
fn dihedral_angle_gradient_of_a_degenerate_hinge_is_none() {
    // Coincident edge vertices.
    assert!(
        dihedral_angle_gradient(
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            [0.0, -1.0, 0.0]
        )
        .is_none()
    );
    // A wing vertex on the line of the edge.
    assert!(
        dihedral_angle_gradient(
            [0.5, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.5, -1.0, 0.3]
        )
        .is_none()
    );
}

// ── bending force ─────────────────────────────────────────────────────────

#[test]
fn bending_force_is_the_negative_gradient_of_the_bending_energy() {
    let mut samples = Samples(3);
    let step = 1.0e-6;
    let (stiffness, rest) = (7.5, 0.4);
    let energy = |points: &[[f64; 3]; 4]| {
        let theta = dihedral_angle(points[0], points[1], points[2], points[3]);
        0.5 * stiffness * (theta - rest) * (theta - rest)
    };
    let constraint = BendingConstraint::new(0, 1, 2, 3, rest, stiffness);
    for trial in 0..60 {
        let points = random_hinge(&mut samples, trial % 2 == 1);
        let theta = dihedral_angle(points[0], points[1], points[2], points[3]);
        if !(0.05..PI - 0.05).contains(&theta) {
            continue;
        }
        let forces = constraint.compute_force(&vertices(&points));
        assert_eq!(forces.len(), 4);
        if forces.iter().any(|f| length3(*f) > 200.0) {
            continue;
        }
        let mut net = [0.0; 3];
        let mut torque = [0.0; 3];
        for vertex in 0..4 {
            net = add3(net, forces[vertex]);
            torque = add3(torque, cross3(points[vertex], forces[vertex]));
            for axis in 0..3 {
                let mut plus = points;
                let mut minus = points;
                plus[vertex][axis] += step;
                minus[vertex][axis] -= step;
                let numeric = -(energy(&plus) - energy(&minus)) / (2.0 * step);
                assert!(
                    (numeric - forces[vertex][axis]).abs() < 1.0e-5 * (1.0 + numeric.abs()),
                    "trial {trial}, vertex {vertex}, axis {axis}: {numeric} vs {}",
                    forces[vertex][axis]
                );
            }
        }
        assert!(length3(net) < 1.0e-9, "net force {net:?}");
        assert!(length3(torque) < 1.0e-9, "net torque {torque:?}");
    }
}

#[test]
fn bending_force_pushes_a_folded_hinge_back_towards_its_rest_angle() {
    // The hinge is folded by 90 degrees: wing 3 sits at (0, 0, 1) above the
    // plane of the edge and wing 0 at (0, 1, 0).
    let points = folded_hinge(PI / 2.0);
    let verts = vertices(&points);
    let flatten = BendingConstraint::new(0, 1, 2, 3, 0.0, 10.0).compute_force(&verts);
    // Flattening swings wing 3 back down towards -y and lifts wing 0's side
    // out of the fold, towards -z.
    assert!(flatten[3][1] < -1.0, "{:?}", flatten[3]);
    assert!(flatten[0][2] < -1.0, "{:?}", flatten[0]);
    // A rest angle as far above the current angle as 0 is below reverses every
    // force.
    let fold_more = BendingConstraint::new(0, 1, 2, 3, PI, 10.0).compute_force(&verts);
    for vertex in 0..4 {
        for axis in 0..3 {
            assert!(
                (flatten[vertex][axis] + fold_more[vertex][axis]).abs() < 1.0e-12,
                "vertex {vertex} axis {axis}"
            );
        }
    }
}

#[test]
fn bending_constraint_built_from_vertices_has_no_force_at_rest() {
    let mut samples = Samples(4);
    let points = random_hinge(&mut samples, false);
    let verts = vertices(&points);
    let constraint =
        BendingConstraint::from_vertices(&verts, 0, 1, 2, 3, 50.0).expect("valid indices");
    assert!(
        (constraint.rest_angle - dihedral_angle(points[0], points[1], points[2], points[3])).abs()
            < 1.0e-15
    );
    for force in constraint.compute_force(&verts) {
        assert!(length3(force) < 1.0e-12, "{force:?}");
    }
    assert!(BendingConstraint::from_vertices(&verts, 0, 1, 2, 4, 50.0).is_none());
}

#[test]
fn bending_force_of_a_degenerate_hinge_or_missing_vertex_is_zero() {
    let collinear = vertices(&[
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [2.0, 0.0, 0.0],
        [3.0, 0.0, 0.0],
    ]);
    let forces = BendingConstraint::new(0, 1, 2, 3, 0.5, 10.0).compute_force(&collinear);
    assert_eq!(forces, vec![[0.0; 3]; 4]);
    let forces = BendingConstraint::new(0, 1, 2, 9, 0.5, 10.0).compute_force(&collinear);
    assert_eq!(forces, vec![[0.0; 3]; 4]);
}

// ── accumulated multipliers: the stretch of a hanging cloth ───────────────

const HANG_GRAVITY: [f64; 3] = [0.0, 0.0, 9.81];

/// A grid hanging from its pinned first row (the grid lies in the xz plane and
/// its rows extend along +z), pulled along +z by gravity.
fn hanging_cloth(rows: usize, cols: usize, spacing: f64, iterations: usize) -> GpuClothSolver {
    let mut mesh = ClothMesh::new();
    mesh.build_grid(rows, cols, spacing);
    let mut solver = GpuClothSolver::new(mesh);
    solver.gravity = HANG_GRAVITY;
    solver.xpbd_iterations = iterations;
    solver
}

/// Mean `z` displacement of the last row from its rest position after
/// `steps` steps of `dt`.
fn bottom_row_stretch(
    solver: &mut GpuClothSolver,
    rows: usize,
    cols: usize,
    dt: f64,
    steps: usize,
) -> f64 {
    let spacing_rest = solver.mesh.vertices[cols].position[2];
    for _ in 0..steps {
        solver.solve(dt);
    }
    let last = &solver.mesh.vertices[(rows - 1) * cols..];
    let mean_z = last.iter().map(|v| v.position[2]).sum::<f64>() / cols as f64;
    mean_z - (rows - 1) as f64 * spacing_rest
}

#[test]
fn hanging_vertex_stretches_by_weight_over_stiffness_for_any_iteration_count() {
    // One free unit mass on one spring of stiffness 1000 N/m: the static stretch
    // is m g / k = 9.81 mm.  Without the −α̃ λ term the stretch shrinks as
    // iterations are added (about 0.9 mm for 8 and 0.05 mm for 32).
    let expected = 9.81 / 1000.0;
    for dt in [0.01, 0.005] {
        for iterations in [1, 8, 32] {
            let mut solver = hanging_cloth(2, 1, 1.0, iterations);
            let stretch = bottom_row_stretch(&mut solver, 2, 1, dt, 600);
            assert!(
                (stretch / expected - 1.0).abs() < 0.01,
                "dt {dt}, {iterations} iterations: stretch {stretch}, weight/stiffness {expected}"
            );
        }
    }
}

#[test]
fn hanging_cloth_stretch_agrees_between_8_and_32_iterations() {
    let (rows, cols, dt, steps) = (3, 3, 0.01, 300);
    let mut coarse = hanging_cloth(rows, cols, 0.1, 8);
    let mut fine = hanging_cloth(rows, cols, 0.1, 32);
    let stretch_coarse = bottom_row_stretch(&mut coarse, rows, cols, dt, steps);
    let stretch_fine = bottom_row_stretch(&mut fine, rows, cols, dt, steps);
    assert!(
        (stretch_coarse / stretch_fine - 1.0).abs() < 0.01,
        "8 iterations: {stretch_coarse}, 32 iterations: {stretch_fine}"
    );
    // The two free rows carry 2 and 1 particle weights: the bottom row hangs at
    // least one weight over the stiffness below the pinned row.
    assert!(stretch_fine > 0.015, "stretch {stretch_fine}");
    // The pinned row has not moved.
    for v in &fine.mesh.vertices[..cols] {
        assert_eq!(v.position[1], 0.0);
        assert!(v.position[2] == 0.0);
    }
}

// ── gravity ───────────────────────────────────────────────────────────────

#[test]
fn gravity_field_defaults_to_standard_gravity_and_is_used() {
    let lone = |gravity: Option<[f64; 3]>| {
        let mesh = ClothMesh {
            vertices: vec![ClothVertex::new([1.0, 2.0, 3.0], 2.0, false)],
            edges: Vec::new(),
        };
        let mut solver = GpuClothSolver::new(mesh);
        if let Some(g) = gravity {
            solver.gravity = g;
        }
        solver.solve(0.1);
        solver.mesh.vertices[0].clone()
    };
    let default = GpuClothSolver::new(ClothMesh::new());
    assert_eq!(default.gravity, [0.0, -9.81, 0.0]);
    assert!(default.bending.is_empty());

    let fallen = lone(None);
    assert!((fallen.position[1] - (2.0 - 9.81 * 0.01)).abs() < 1.0e-12);
    assert!((fallen.velocity[1] + 9.81 * 0.1).abs() < 1.0e-12);

    let sideways = lone(Some([3.0, 0.0, -1.0]));
    assert!((sideways.position[0] - (1.0 + 3.0 * 0.01)).abs() < 1.0e-12);
    assert!((sideways.position[1] - 2.0).abs() < 1.0e-12);
    assert!((sideways.position[2] - (3.0 - 1.0 * 0.01)).abs() < 1.0e-12);

    let weightless = lone(Some([0.0; 3]));
    assert_eq!(weightless.position, [1.0, 2.0, 3.0]);
    assert_eq!(weightless.velocity, [0.0; 3]);
}

// ── edge damping ──────────────────────────────────────────────────────────

/// Two free unit masses on a spring of stiffness 100 N/m and rest length 1,
/// released at the rest length with opposite velocities along the edge.
fn energy_of_a_released_spring(damping: f64) -> f64 {
    let mut v0 = ClothVertex::new([0.0, 0.0, 0.0], 1.0, false);
    let mut v1 = ClothVertex::new([1.0, 0.0, 0.0], 1.0, false);
    v0.velocity = [-0.5, 0.0, 0.0];
    v1.velocity = [0.5, 0.0, 0.0];
    let mesh = ClothMesh {
        vertices: vec![v0, v1],
        edges: vec![ClothEdge::new(0, 1, 1.0, 100.0, damping)],
    };
    let mut solver = GpuClothSolver::new(mesh);
    solver.gravity = [0.0; 3];
    for _ in 0..200 {
        solver.solve(0.005);
    }
    let [a, b] = [&solver.mesh.vertices[0], &solver.mesh.vertices[1]];
    let kinetic = 0.5 * (dot3(a.velocity, a.velocity) + dot3(b.velocity, b.velocity));
    let stretch = length3(sub3(b.position, a.position)) - 1.0;
    kinetic + 0.5 * 100.0 * stretch * stretch
}

#[test]
fn edge_damping_dissipates_energy_monotonically() {
    let undamped = energy_of_a_released_spring(0.0);
    let light = energy_of_a_released_spring(1.0);
    let heavy = energy_of_a_released_spring(5.0);
    assert!(undamped < 0.25, "implicit integration never adds energy");
    assert!(light < 0.5 * undamped, "{light} vs {undamped}");
    assert!(heavy < 0.1 * light, "{heavy} vs {light}");
    // A negative damping coefficient is not an energy source.
    assert!((energy_of_a_released_spring(-3.0) - undamped).abs() < 1.0e-12);
}

// ── robustness ────────────────────────────────────────────────────────────

#[test]
fn solve_with_unusable_time_step_changes_nothing() {
    for dt in [0.0, -0.01, f64::NAN, f64::INFINITY] {
        let mut solver = hanging_cloth(3, 3, 0.1, 8);
        solver.mesh.vertices[4].velocity = [1.0, 2.0, 3.0];
        let before = solver.mesh.vertices.clone();
        solver.solve(dt);
        for (now, then) in solver.mesh.vertices.iter().zip(&before) {
            assert_eq!(now.position, then.position, "dt = {dt}");
            assert_eq!(now.velocity, then.velocity, "dt = {dt}");
        }
    }
}

#[test]
fn edges_with_bad_indices_or_stiffness_and_vertices_without_mass_are_handled() {
    let mut mesh = ClothMesh::new();
    mesh.build_grid(2, 2, 0.5);
    // An edge to a vertex that does not exist, a zero-stiffness edge and a
    // negative-stiffness edge.
    mesh.edges.push(ClothEdge::new(0, 99, 1.0, 1000.0, 0.1));
    mesh.edges.push(ClothEdge::new(2, 3, 0.5, 0.0, 0.1));
    mesh.edges.push(ClothEdge::new(2, 3, 0.5, -5.0, 0.1));
    // A vertex with no mass cannot be moved.
    mesh.vertices[3].mass = 0.0;
    let fixed = mesh.vertices[3].position;
    let mut solver = GpuClothSolver::new(mesh);
    solver
        .bending
        .push(BendingConstraint::new(0, 1, 2, 77, 0.0, 10.0));
    for _ in 0..20 {
        solver.solve(0.01);
    }
    for v in &solver.mesh.vertices {
        assert!(v.position.iter().all(|c| c.is_finite()), "{:?}", v.position);
        assert!(v.velocity.iter().all(|c| c.is_finite()), "{:?}", v.velocity);
    }
    assert_eq!(solver.mesh.vertices[3].position, fixed);
}

// ── bending constraints in the solver ─────────────────────────────────────

/// A flap hinged on the x axis between two pinned vertices, with a pinned wing
/// at `+y` and a free wing of unit mass at `−y` that is held at its distance
/// from both edge vertices by stiff springs.  Gravity pulls it out of the plane
/// along `−z`, folding the hinge.
fn flap(bending_stiffness: Option<f64>) -> GpuClothSolver {
    let mesh = ClothMesh {
        vertices: vec![
            ClothVertex::new([0.0, 0.5, 0.0], 1.0, true),
            ClothVertex::new([-0.5, 0.0, 0.0], 1.0, true),
            ClothVertex::new([0.5, 0.0, 0.0], 1.0, true),
            ClothVertex::new([0.0, -0.5, 0.0], 1.0, false),
        ],
        edges: vec![
            ClothEdge::new(1, 3, 0.5 * std::f64::consts::SQRT_2, 1.0e5, 0.0),
            ClothEdge::new(2, 3, 0.5 * std::f64::consts::SQRT_2, 1.0e5, 0.0),
        ],
    };
    let mut solver = GpuClothSolver::new(mesh);
    solver.gravity = [0.0, 0.0, -9.81];
    if let Some(stiffness) = bending_stiffness {
        solver
            .bending
            .push(BendingConstraint::new(0, 1, 2, 3, 0.0, stiffness));
    }
    solver
}

fn flap_angle(solver: &GpuClothSolver) -> f64 {
    let p = |i: usize| solver.mesh.vertices[i].position;
    dihedral_angle(p(0), p(1), p(2), p(3))
}

#[test]
fn bending_stiffness_balances_the_weight_of_a_flap_like_a_torsion_spring() {
    // Energy ½ k φ² + m g z with z = −0.5 sin φ gives k φ = 0.5 m g cos φ.
    let stiffness = 100.0;
    let mut phi = 0.0_f64;
    for _ in 0..50 {
        phi = 0.5 * 9.81 * phi.cos() / stiffness;
    }
    let mut solver = flap(Some(stiffness));
    let mut peak = 0.0_f64;
    for _ in 0..600 {
        solver.solve(0.01);
        peak = peak.max(flap_angle(&solver));
    }
    let settled = flap_angle(&solver);
    assert!(
        (settled / phi - 1.0).abs() < 0.01,
        "settled fold {settled}, static equilibrium {phi}"
    );
    assert!(peak < 0.15, "the stiff hinge never folds far: {peak}");
}

#[test]
fn stiffer_bending_folds_the_flap_less_and_no_bending_lets_it_swing_through() {
    let peak_of = |stiffness: Option<f64>| {
        let mut solver = flap(stiffness);
        let mut peak = 0.0_f64;
        for _ in 0..600 {
            solver.solve(0.01);
            peak = peak.max(flap_angle(&solver));
        }
        peak
    };
    let free = peak_of(None);
    let soft = peak_of(Some(10.0));
    let stiff = peak_of(Some(1000.0));
    assert!(free > 1.2, "a free flap swings down: {free}");
    assert!(soft < free && stiff < soft, "{free} > {soft} > {stiff}");
    assert!(stiff < 0.02, "{stiff}");
}

#[test]
fn bending_projection_drives_a_folded_hinge_to_its_rest_angle() {
    // Stiff bending (nearly rigid compliance), edge vertices pinned, no gravity.
    for rest in [0.0, PI / 4.0] {
        let points = folded_hinge(PI / 2.0);
        let mut verts = vertices(&points);
        verts[1].pinned = true;
        verts[2].pinned = true;
        let mesh = ClothMesh {
            vertices: verts,
            edges: Vec::new(),
        };
        let mut solver = GpuClothSolver::new(mesh);
        solver.gravity = [0.0; 3];
        solver.xpbd_iterations = 40;
        solver
            .bending
            .push(BendingConstraint::new(0, 1, 2, 3, rest, 1.0e8));
        solver.solve(0.01);
        let angle = flap_angle(&solver);
        assert!(
            (angle - rest).abs() < 1.0e-3,
            "rest {rest}: folded hinge relaxed to {angle}"
        );
    }
}

#[test]
fn hinge_without_a_bending_constraint_is_not_bent_back() {
    let mut verts = vertices(&folded_hinge(PI / 2.0));
    verts[1].pinned = true;
    verts[2].pinned = true;
    let mesh = ClothMesh {
        vertices: verts,
        edges: Vec::new(),
    };
    let mut solver = GpuClothSolver::new(mesh);
    solver.gravity = [0.0; 3];
    let before: Vec<[f64; 3]> = solver.mesh.vertices.iter().map(|v| v.position).collect();
    solver.solve(0.01);
    for (v, p) in solver.mesh.vertices.iter().zip(&before) {
        assert_eq!(v.position, *p);
    }
}

/// A hinge of four free unit masses at rest, folded by 2 rad, with one bending
/// constraint of rest angle 0.3.
fn free_hinge(iterations: usize) -> GpuClothSolver {
    let mesh = ClothMesh {
        vertices: vertices(&folded_hinge(2.0)),
        edges: Vec::new(),
    };
    let mut solver = GpuClothSolver::new(mesh);
    solver.gravity = [0.0; 3];
    solver.xpbd_iterations = iterations;
    solver
        .bending
        .push(BendingConstraint::new(0, 1, 2, 3, 0.3, 50.0));
    solver
}

#[test]
fn bending_projection_conserves_linear_momentum_of_free_vertices() {
    let mut solver = free_hinge(12);
    let centre = |solver: &GpuClothSolver| {
        solver
            .mesh
            .vertices
            .iter()
            .fold([0.0; 3], |acc, v| add3(acc, scale3(v.position, 0.25)))
    };
    let before = centre(&solver);
    solver.solve(0.01);
    assert!(length3(sub3(centre(&solver), before)) < 1.0e-12);
    let momentum = solver
        .mesh
        .vertices
        .iter()
        .fold([0.0; 3], |acc, v| add3(acc, v.velocity));
    assert!(length3(momentum) < 1.0e-9, "momentum {momentum:?}");
    // The constraint did act: the fold moved towards the rest angle.
    let angle = flap_angle(&solver);
    assert!(angle < 2.0 - 1.0e-4 && angle > 0.3, "angle {angle}");
}

#[test]
fn single_bending_projection_conserves_angular_momentum() {
    // One iteration displaces the vertices along the torque-free gradient of the
    // dihedral angle taken at the initial positions, so the angular momentum
    // about the origin stays zero.
    let mut solver = free_hinge(1);
    solver.solve(0.01);
    let angular = solver
        .mesh
        .vertices
        .iter()
        .fold([0.0; 3], |acc, v| add3(acc, cross3(v.position, v.velocity)));
    assert!(length3(angular) < 1.0e-9, "angular momentum {angular:?}");
    assert!((flap_angle(&solver) - 2.0).abs() > 1.0e-4);
}
