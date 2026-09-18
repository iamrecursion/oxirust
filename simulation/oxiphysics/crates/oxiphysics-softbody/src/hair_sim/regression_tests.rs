// Copyright 2026 COOLJAPAN OU (Team KitaSan)
// SPDX-License-Identifier: Apache-2.0

//! Regression tests for the twist model and the analytic forces of
//! [`HairStrand`].
//!
//! * The twist energy used to be `((t1 - t0) . (t0 + t1))² ...`, which is
//!   identically zero for unit tangents, so the twist stiffness had no effect
//!   at all. The strand now carries per-segment material angles and a
//!   parallel-transported reference frame (Bergou et al. 2008).
//! * The elastic forces used to be a forward finite-difference gradient that
//!   cloned the whole strand three times per node (`O(n²)` per step). They
//!   are analytic now, and the finite differences live here, in the tests,
//!   where they serve as the reference.

use super::*;
use std::f64::consts::{FRAC_PI_2, FRAC_PI_4, PI};
use std::time::{Duration, Instant};

const FD_STEP: f64 = 1.0e-6;

/// Strand along +y from the origin (pinned root) with the given stiffnesses.
fn vertical_strand(segments: usize, bend: f64, twist: f64, stretch: f64) -> HairStrand {
    HairStrand::new_straight(
        [0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        segments,
        1.0e-3,
        bend,
        twist,
        stretch,
    )
}

/// A deterministic, clearly non-planar curve with varying edge lengths.
fn helix_point(i: usize) -> [f64; 3] {
    let s = i as f64;
    [
        0.15 * (0.8 * s).sin(),
        0.2 * s + 0.02 * (1.7 * s).sin(),
        0.15 * ((0.8 * s).cos() - 1.0),
    ]
}

/// A strand moved onto the helix by the shortest rotation of every segment
/// (so the reference frames carry a non-trivial holonomy), with prescribed
/// material angles, rest curvature and stretched, non-uniform rest lengths.
fn bent_twisted_strand(segments: usize, bend: f64, twist: f64, stretch: f64) -> HairStrand {
    let mut strand = vertical_strand(segments, bend, twist, stretch);
    for (i, node) in strand.nodes.iter_mut().enumerate() {
        node.position = helix_point(i);
    }
    sync_reference_frames(&mut strand);
    for j in 0..strand.num_segments() {
        let length = v3_norm(v3_sub(
            strand.nodes[j + 1].position,
            strand.nodes[j].position,
        ));
        strand.rest_lengths[j] = 0.9 * length * (1.0 + 0.05 * j as f64);
        strand.theta[j] = 0.3 * (1.1 * j as f64 + 0.4).sin();
    }
    for i in 1..strand.num_nodes() - 1 {
        strand.rest_curvature[i] = [0.3, -0.2 + 0.05 * i as f64, 0.1];
    }
    strand
}

/// Assert `forces == -grad(energy)` for every node and axis, comparing against
/// a central finite difference of the energy.
fn assert_forces_match_energy_gradient(
    strand: &HairStrand,
    forces: &[[f64; 3]],
    energy: &dyn Fn(&HairStrand) -> f64,
    label: &str,
) {
    let scale = forces
        .iter()
        .flat_map(|f| f.iter())
        .fold(0.0_f64, |acc, c| acc.max(c.abs()));
    assert!(
        scale > 1.0e-6,
        "{label}: the configuration must produce non-trivial forces, largest component {scale}"
    );
    for (node, force) in forces.iter().enumerate() {
        for (axis, analytic) in force.iter().enumerate() {
            let mut plus = strand.clone();
            plus.nodes[node].position[axis] += FD_STEP;
            let mut minus = strand.clone();
            minus.nodes[node].position[axis] -= FD_STEP;
            let gradient = (energy(&plus) - energy(&minus)) / (2.0 * FD_STEP);
            assert!(
                (analytic + gradient).abs() <= 1.0e-5 * scale,
                "{label}: node {node} axis {axis}: analytic force {analytic}, \
                 finite-difference force {}",
                -gradient
            );
        }
    }
}

fn total_energy(strand: &HairStrand) -> f64 {
    stretch_energy(strand) + bending_energy(strand) + twisting_energy(strand)
}

/// Stretch + bending + twisting forces on every node, pinned ones included.
fn raw_elastic_forces(strand: &HairStrand) -> Vec<[f64; 3]> {
    let mut sum = vec![[0.0_f64; 3]; strand.num_nodes()];
    for part in [
        stretch_forces(strand),
        bending_forces(strand),
        twisting_forces(strand),
    ] {
        for (acc, f) in sum.iter_mut().zip(part.iter()) {
            *acc = v3_add(*acc, *f);
        }
    }
    sum
}

// ---------------------------------------------------------------------------
// F4149: the twist energy is a real function of the material angles
// ---------------------------------------------------------------------------

#[test]
fn twist_energy_of_a_material_angle_difference_matches_the_closed_form() {
    let twist = 2.5e-3;
    let mut strand = HairStrand::new_straight(
        [0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        2,
        1.0e-3,
        1.0e-3,
        twist,
        1.0e3,
    );
    assert_eq!(strand.num_nodes(), 3);
    strand.theta[0] = 0.2;
    strand.theta[1] = 0.2 + FRAC_PI_4;
    // Two segments of 0.5 m: l_bar = 0.5.
    let expected = 0.5 * twist * FRAC_PI_4 * FRAC_PI_4 / 0.5;
    let energy = twisting_energy(&strand);
    assert!(
        (energy - expected).abs() <= 1.0e-12 * expected,
        "E_twist {energy} should be 1/2 k_t (pi/4)^2 / l_bar = {expected}"
    );
    // Linear in the stiffness, even in the sign of the angle difference.
    strand.twist_stiffness = 2.0 * twist;
    assert!((twisting_energy(&strand) - 2.0 * expected).abs() <= 2.0e-12 * expected);
    strand.theta[1] = 0.2 - FRAC_PI_4;
    assert!((twisting_energy(&strand) - 2.0 * expected).abs() <= 2.0e-12 * expected);
    // Rotating both material frames together costs nothing.
    strand.theta[1] = strand.theta[0];
    assert_eq!(twisting_energy(&strand), 0.0);
}

#[test]
fn twist_energy_uses_the_voronoi_length_of_the_node() {
    let twist = 1.0e-2;
    let mut strand = vertical_strand(2, 1.0e-3, twist, 1.0e3);
    strand.rest_lengths = vec![0.2, 0.6];
    strand.theta[1] = 0.5;
    // l_bar = (0.2 + 0.6) / 2 = 0.4
    let expected = 0.5 * twist * 0.5 * 0.5 / 0.4;
    let energy = twisting_energy(&strand);
    assert!(
        (energy - expected).abs() <= 1.0e-12 * expected,
        "E_twist {energy} should use l_bar = 0.4: expected {expected}"
    );
}

#[test]
fn planar_bending_induces_no_reference_twist() {
    let mut strand = vertical_strand(6, 1.0e-3, 1.0e-2, 1.0e3);
    // A director that is neither in nor out of the bending plane.
    let director = v3_normalize([1.0, 0.0, 1.0]);
    strand.ref_d1 = vec![director; 6];
    // Bend into a circular arc in the x-y plane without synchronising.
    for (i, node) in strand.nodes.iter_mut().enumerate() {
        let phi = 0.25 * i as f64;
        node.position = [0.8 * (1.0 - phi.cos()), 0.8 * phi.sin(), 0.0];
    }
    assert!(bending_energy(&strand) > 0.0);
    for (i, alpha) in reference_twists(&strand).iter().enumerate() {
        assert!(alpha.abs() < 1.0e-12, "node {i}: reference twist {alpha}");
    }
    assert!(twisting_energy(&strand).abs() < 1.0e-24);
    assert!(
        twisting_forces(&strand)
            .iter()
            .all(|f| v3_norm(*f) < 1.0e-12),
        "a planar bend must not produce twist forces"
    );
    sync_reference_frames(&mut strand);
    for alpha in reference_twists(&strand) {
        assert!(alpha.abs() < 1.0e-12);
    }
}

#[test]
fn closing_a_solid_angle_loop_gives_the_holonomy_of_an_octant() {
    // Tangents y -> x -> z: the geodesic triangle (y, x, z) is one octant of
    // the sphere, whose solid angle (the holonomy) is pi / 2.
    let mut strand = vertical_strand(3, 1.0e-3, 1.0e-2, 1.0e3);
    strand.ref_d1 = vec![[0.0, 0.0, 1.0]; 3];
    strand.nodes[0].position = [0.0, 0.0, 0.0];
    strand.nodes[1].position = [0.0, 1.0, 0.0];
    strand.nodes[2].position = [1.0, 1.0, 0.0];
    strand.nodes[3].position = [1.0, 1.0, 1.0];
    let alpha = reference_twists(&strand);
    assert!(alpha[0].abs() < 1.0e-14 && alpha[3].abs() < 1.0e-14);
    assert!(
        alpha[1].abs() < 1.0e-12,
        "rotation about the common axis z does not twist: {}",
        alpha[1]
    );
    assert!(
        (alpha[2] - FRAC_PI_2).abs() < 1.0e-12,
        "holonomy of the octant should be pi/2, got {}",
        alpha[2]
    );
    // The mirror image winds the other way.
    strand.nodes[2].position = [-1.0, 1.0, 0.0];
    strand.nodes[3].position = [-1.0, 1.0, 1.0];
    let mirrored = reference_twists(&strand);
    assert!(
        (mirrored[2] + FRAC_PI_2).abs() < 1.0e-12,
        "the mirrored loop should give -pi/2, got {}",
        mirrored[2]
    );
    // The twist energy follows: theta = 0, so m = alpha.
    let l_bar = 1.0 / 3.0;
    let expected = 0.5 * strand.twist_stiffness * FRAC_PI_2 * FRAC_PI_2 / l_bar;
    assert!((twisting_energy(&strand) - expected).abs() < 1.0e-12 * expected);
}

#[test]
fn reference_twist_is_tracked_continuously_beyond_pi() {
    // Sweep the second segment once around the first one on a cone of half
    // angle 75 degrees. The reference frame ends up rotated by the solid
    // angle of the cap, 2 pi (1 - cos 75 deg) = 4.66 rad, which is more than
    // pi and therefore must not be wrapped back into [-pi, pi].
    let half_angle = 75.0_f64.to_radians();
    let steps = 720;
    let place = |strand: &mut HairStrand, psi: f64| {
        let tangent = [
            half_angle.sin() * psi.cos(),
            half_angle.cos(),
            half_angle.sin() * psi.sin(),
        ];
        strand.nodes[0].position = [0.0, 0.0, 0.0];
        strand.nodes[1].position = [0.0, 1.0, 0.0];
        strand.nodes[2].position = v3_add([0.0, 1.0, 0.0], tangent);
    };
    let mut strand = vertical_strand(2, 1.0e-3, 1.0e-2, 1.0e3);
    place(&mut strand, 0.0);
    strand.reset_reference_frames();
    assert!(reference_twists(&strand)[1].abs() < 1.0e-14);
    for k in 1..=steps {
        place(&mut strand, 2.0 * PI * k as f64 / steps as f64);
        sync_reference_frames(&mut strand);
    }
    let alpha = reference_twists(&strand)[1];
    let solid_angle = 2.0 * PI * (1.0 - half_angle.cos());
    assert!(
        alpha.abs() > PI,
        "the accumulated twist {alpha} must exceed pi"
    );
    assert!(
        (alpha.abs() - solid_angle).abs() < 5.0e-3,
        "accumulated twist {alpha} should match the cap solid angle {solid_angle}"
    );
    assert!(
        (alpha - wrap_to_pi(alpha)).abs() > 1.0,
        "the stored value must not be a wrapped angle"
    );
    assert!((strand.ref_twist[1] - alpha).abs() < 1.0e-9);
}

#[test]
fn twist_stiffness_changes_the_forces_and_the_motion() {
    let soft = bent_twisted_strand(6, 1.0e-3, 0.0, 50.0);
    let mut stiff = soft.clone();
    stiff.twist_stiffness = 1.0e-2;
    let f_soft = discrete_elastic_rod(&soft);
    let f_stiff = discrete_elastic_rod(&stiff);
    let twist = twisting_forces(&stiff);
    let largest_difference = f_soft
        .iter()
        .zip(f_stiff.iter())
        .map(|(a, b)| v3_norm(v3_sub(*b, *a)))
        .fold(0.0_f64, f64::max);
    assert!(
        largest_difference > 1.0e-4,
        "raising the twist stiffness must change the elastic forces, largest change {largest_difference}"
    );
    // The change is exactly the twist force on the free nodes.
    for (i, ((a, b), t)) in f_soft
        .iter()
        .zip(f_stiff.iter())
        .zip(twist.iter())
        .enumerate()
    {
        if stiff.nodes[i].is_static() {
            continue;
        }
        let residual = v3_norm(v3_sub(v3_sub(*b, *a), *t));
        assert!(residual < 1.0e-12, "node {i}: residual {residual}");
    }
    // ... and the motion differs after a few steps.
    let mut soft_run = soft;
    let mut stiff_run = stiff;
    for _ in 0..20 {
        step_strand(&mut soft_run, [0.0; 3], 5.0e-4);
        step_strand(&mut stiff_run, [0.0; 3], 5.0e-4);
    }
    let separation = v3_norm(v3_sub(
        soft_run.nodes[6].position,
        stiff_run.nodes[6].position,
    ));
    assert!(
        separation > 1.0e-6,
        "twist stiffness must affect the trajectory, tip separation {separation}"
    );
}

// ---------------------------------------------------------------------------
// Quasi-static relaxation of the interior material angles
// ---------------------------------------------------------------------------

#[test]
fn quasistatic_twist_is_uniform_on_a_uniform_strand() {
    let twist = 4.0e-3;
    let mut strand = vertical_strand(5, 1.0e-3, twist, 1.0e3);
    strand.theta[0] = 0.2;
    strand.theta[4] = 1.2;
    solve_twist_quasistatic(&mut strand);
    for j in 0..5 {
        assert!(
            (strand.theta[j] - (0.2 + 0.25 * j as f64)).abs() < 1.0e-12,
            "theta[{j}] = {}",
            strand.theta[j]
        );
    }
    assert_eq!(strand.theta[0], 0.2);
    assert_eq!(strand.theta[4], 1.2);
    // One radian spread over four nodes of 0.2 m each.
    let expected = 0.5 * twist / (4.0 * 0.2);
    let energy = twisting_energy(&strand);
    assert!((energy - expected).abs() < 1.0e-12 * expected);
    // The relaxed state is the minimum for the prescribed boundary values.
    for j in 1..4 {
        for delta in [-1.0e-3, 1.0e-3] {
            let mut perturbed = strand.clone();
            perturbed.theta[j] += delta;
            assert!(twisting_energy(&perturbed) > energy);
        }
    }
}

#[test]
fn quasistatic_twist_balances_the_torque_on_a_non_uniform_strand() {
    let mut strand = vertical_strand(5, 1.0e-3, 3.0e-3, 1.0e3);
    strand.rest_lengths = vec![0.1, 0.3, 0.2, 0.4, 0.1];
    strand.theta[0] = -0.4;
    strand.theta[4] = 0.9;
    solve_twist_quasistatic(&mut strand);
    let twists = node_twists(&strand);
    let torque: Vec<f64> = (1..5)
        .map(|i| twists[i] / strand.voronoi_length(i))
        .collect();
    for t in &torque {
        assert!(
            (t - torque[0]).abs() < 1.0e-12 * torque[0].abs(),
            "torque must be the same at every node: {torque:?}"
        );
    }
    let total: f64 = twists[1..5].iter().sum();
    assert!((total - 1.3).abs() < 1.0e-12, "sum of twists {total}");
}

#[test]
fn quasistatic_solve_includes_the_reference_twist() {
    // On a bent strand the relaxed state is the one where the twist
    // m_i = theta_i - theta_(i-1) + alpha_i obeys the torque balance.
    let mut strand = bent_twisted_strand(7, 1.0e-3, 5.0e-3, 50.0);
    solve_twist_quasistatic(&mut strand);
    let twists = node_twists(&strand);
    let n = strand.num_nodes();
    let torque: Vec<f64> = (1..n - 1)
        .map(|i| twists[i] / strand.voronoi_length(i))
        .collect();
    let scale = torque.iter().fold(0.0_f64, |acc, t| acc.max(t.abs()));
    assert!(scale > 1.0e-3, "the test needs a non-trivial twist");
    for t in &torque {
        assert!((t - torque[0]).abs() < 1.0e-9 * scale, "torque {torque:?}");
    }
}

#[test]
fn short_strands_have_no_interior_angle_to_solve() {
    let mut strand = vertical_strand(2, 1.0e-3, 1.0e-2, 1.0e3);
    strand.theta = vec![0.3, -0.7];
    solve_twist_quasistatic(&mut strand);
    assert_eq!(strand.theta, vec![0.3, -0.7]);
    let mut single = vertical_strand(1, 1.0e-3, 1.0e-2, 1.0e3);
    single.theta = vec![0.5];
    solve_twist_quasistatic(&mut single);
    assert_eq!(single.theta, vec![0.5]);
    assert_eq!(twisting_energy(&single), 0.0);
}

#[test]
fn step_strand_keeps_the_boundary_angles_and_relaxes_the_interior() {
    let mut strand = bent_twisted_strand(6, 1.0e-3, 5.0e-3, 50.0);
    let first = strand.theta[0];
    let last = strand.theta[5];
    step_strand(&mut strand, [0.0, -9.81, 0.0], 5.0e-4);
    assert_eq!(strand.theta[0], first);
    assert_eq!(strand.theta[5], last);
    let twists = node_twists(&strand);
    let torque: Vec<f64> = (1..6)
        .map(|i| twists[i] / strand.voronoi_length(i))
        .collect();
    let scale = torque.iter().fold(0.0_f64, |acc, t| acc.max(t.abs()));
    assert!(scale > 1.0e-3);
    for t in &torque {
        assert!((t - torque[0]).abs() < 1.0e-9 * scale, "torque {torque:?}");
    }
    for j in 0..strand.num_segments() {
        let tangent = strand.segment_tangent(j).expect("non-degenerate segment");
        assert!(v3_norm(v3_sub(tangent, strand.ref_tangent[j])) < 1.0e-14);
    }
}

#[test]
fn reference_frames_stay_orthonormal_while_the_strand_swings_in_3d() {
    let mut strand = vertical_strand(8, 2.0e-3, 1.0e-3, 40.0);
    strand.drag_coeff = 0.02;
    let mut largest_twist = 0.0_f64;
    for step in 0..1200 {
        wind_force(
            &mut strand,
            [0.0, 0.0, 3.0 + 2.0 * (0.05 * step as f64).sin()],
        );
        step_strand(&mut strand, [2.0, -9.81, 0.0], 5.0e-4);
        for j in 0..strand.num_segments() {
            let tangent = strand.segment_tangent(j).expect("non-degenerate segment");
            let d1 = strand.ref_d1[j];
            assert!(
                (v3_norm(d1) - 1.0).abs() < 1.0e-9,
                "step {step} segment {j}"
            );
            assert!(
                v3_dot(d1, tangent).abs() < 1.0e-9,
                "step {step} segment {j}"
            );
            assert!(v3_norm(v3_sub(tangent, strand.ref_tangent[j])) < 1.0e-12);
        }
        for alpha in &strand.ref_twist {
            assert!(alpha.is_finite());
            largest_twist = largest_twist.max(alpha.abs());
        }
    }
    assert!(
        largest_twist > 1.0e-4,
        "the 3D swing must accumulate some reference twist, largest {largest_twist}"
    );
}

// ---------------------------------------------------------------------------
// F4150: analytic forces are the gradient of the energies
// ---------------------------------------------------------------------------

#[test]
fn stretch_forces_match_the_energy_gradient() {
    let strand = bent_twisted_strand(7, 1.0e-3, 5.0e-3, 80.0);
    let forces = stretch_forces(&strand);
    assert_forces_match_energy_gradient(&strand, &forces, &stretch_energy, "stretch");
}

#[test]
fn bending_forces_match_the_energy_gradient() {
    let strand = bent_twisted_strand(7, 3.0e-3, 5.0e-3, 80.0);
    let forces = bending_forces(&strand);
    assert_forces_match_energy_gradient(&strand, &forces, &bending_energy, "bending");
}

#[test]
fn twisting_forces_match_the_energy_gradient() {
    let strand = bent_twisted_strand(7, 1.0e-3, 4.0e-2, 80.0);
    let twists = node_twists(&strand);
    assert!(
        twists.iter().any(|m| m.abs() > 0.05),
        "the configuration must carry twist: {twists:?}"
    );
    let forces = twisting_forces(&strand);
    assert_forces_match_energy_gradient(&strand, &forces, &twisting_energy, "twisting");
}

#[test]
fn elastic_forces_match_the_total_energy_gradient() {
    let strand = bent_twisted_strand(7, 3.0e-3, 4.0e-2, 80.0);
    let raw = raw_elastic_forces(&strand);
    assert_forces_match_energy_gradient(&strand, &raw, &total_energy, "total");
    // The solver force is the same, except that the pinned root receives none.
    let forces = discrete_elastic_rod(&strand);
    assert_eq!(forces[0], [0.0; 3]);
    assert!(v3_norm(raw[0]) > 1.0e-6, "the root is loaded in this setup");
    for (i, (f, r)) in forces.iter().zip(raw.iter()).enumerate().skip(1) {
        assert!(v3_norm(v3_sub(*f, *r)) < 1.0e-14, "node {i}");
    }
}

#[test]
fn internal_forces_do_not_move_the_centre_of_mass() {
    let strand = bent_twisted_strand(9, 3.0e-3, 4.0e-2, 80.0);
    for (label, forces) in [
        ("stretch", stretch_forces(&strand)),
        ("bending", bending_forces(&strand)),
        ("twisting", twisting_forces(&strand)),
    ] {
        let scale = forces.iter().map(|f| v3_norm(*f)).fold(0.0_f64, f64::max);
        let net = forces.iter().fold([0.0_f64; 3], |acc, f| v3_add(acc, *f));
        assert!(scale > 1.0e-6, "{label}: trivial configuration");
        assert!(
            v3_norm(net) < 1.0e-12 * scale.max(1.0),
            "{label}: net internal force {net:?}"
        );
    }
}

#[test]
fn bending_force_reduces_the_bending_energy() {
    let mut strand = bent_twisted_strand(7, 3.0e-3, 0.0, 80.0);
    strand.rest_curvature = vec![[0.0; 3]; strand.num_nodes()];
    let before = bending_energy(&strand);
    let forces = bending_forces(&strand);
    let step = 1.0e-3;
    for (node, force) in strand.nodes.iter_mut().zip(forces.iter()) {
        node.position = v3_add(node.position, v3_scale(*force, step));
    }
    let after = bending_energy(&strand);
    assert!(
        after < before,
        "moving along the bending force must straighten the strand: {before} -> {after}"
    );
}

#[test]
fn stretch_forces_pull_a_stretched_segment_back() {
    let mut strand = vertical_strand(1, 1.0e-3, 1.0e-3, 100.0);
    strand.nodes[1].position = [0.0, 1.5, 0.0];
    let forces = stretch_forces(&strand);
    // E = k/2 (1.5 - 1)^2 / 1 -> |F| = k (1.5 - 1) / 1 = 50 N, towards the other node.
    assert!((forces[1][1] + 50.0).abs() < 1.0e-12, "{:?}", forces[1]);
    assert!((forces[0][1] - 50.0).abs() < 1.0e-12, "{:?}", forces[0]);
    // A single segment has no bending or twist; the elastic force still acts
    // on the free node and the pinned root receives none.
    let elastic = discrete_elastic_rod(&strand);
    assert!((elastic[1][1] + 50.0).abs() < 1.0e-12);
    assert_eq!(elastic[0], [0.0; 3]);
}

#[test]
fn elastic_forces_are_linear_in_the_number_of_nodes() {
    // The previous finite-difference gradient cloned the whole strand three
    // times per node: this size would take minutes there.
    let segments = 20_000;
    let mut strand = HairStrand::new_straight(
        [0.0, 0.0, 0.0],
        [0.0, 20.0, 0.0],
        segments,
        1.0e-3,
        1.0e-3,
        1.0e-3,
        1.0e3,
    );
    for (i, node) in strand.nodes.iter_mut().enumerate() {
        node.position[0] += 2.0e-3 * (0.37 * i as f64).sin();
        node.position[2] += 2.0e-3 * (0.23 * i as f64).cos();
    }
    sync_reference_frames(&mut strand);
    for (j, theta) in strand.theta.iter_mut().enumerate() {
        *theta = 0.01 * (0.11 * j as f64).sin();
    }
    let start = Instant::now();
    let forces = discrete_elastic_rod(&strand);
    let elapsed = start.elapsed();
    assert_eq!(forces.len(), segments + 1);
    assert!(forces.iter().all(|f| f.iter().all(|c| c.is_finite())));
    assert!(
        forces.iter().any(|f| v3_norm(*f) > 0.0),
        "the perturbed strand must be loaded"
    );
    assert!(
        elapsed < Duration::from_secs(30),
        "one force evaluation of {segments} segments took {elapsed:?}"
    );
}

// ---------------------------------------------------------------------------
// Frame algebra and degenerate input
// ---------------------------------------------------------------------------

#[test]
fn transport_rotates_from_onto_to_without_spin() {
    let from = v3_normalize([0.3, -0.5, 0.8]);
    let to = v3_normalize([-0.7, 0.2, 0.4]);
    let director = perpendicular_unit(from);
    assert!(v3_dot(director, from).abs() < 1.0e-15);
    assert!(v3_norm(v3_sub(transport(from, from, to), to)) < 1.0e-14);
    let moved = transport(director, from, to);
    assert!((v3_norm(moved) - 1.0).abs() < 1.0e-14);
    assert!(v3_dot(moved, to).abs() < 1.0e-14);
    // A rotation preserves the handedness: (from, director, from x director)
    // maps to (to, moved, to x moved).
    let third = v3_cross(from, director);
    let third_moved = transport(third, from, to);
    assert!(v3_norm(v3_sub(third_moved, v3_cross(to, moved))) < 1.0e-14);
    // No rotation when the tangents agree.
    assert!(v3_norm(v3_sub(transport(director, from, from), director)) < 1.0e-14);
    // Half turn when they are opposite.
    let opposite = v3_scale(from, -1.0);
    assert!(v3_norm(v3_sub(transport(from, from, opposite), opposite)) < 1.0e-14);
    let flipped = transport(director, from, opposite);
    assert!((v3_norm(flipped) - 1.0).abs() < 1.0e-14);
    assert!(v3_dot(flipped, from).abs() < 1.0e-14);
}

#[test]
fn wrap_to_pi_folds_into_the_principal_branch() {
    assert!((wrap_to_pi(0.3) - 0.3).abs() < 1.0e-15);
    assert!((wrap_to_pi(0.3 + 2.0 * PI) - 0.3).abs() < 1.0e-12);
    assert!((wrap_to_pi(0.3 - 6.0 * PI) - 0.3).abs() < 1.0e-12);
    assert!(wrap_to_pi(PI + 0.1) < 0.0);
    assert!(wrap_to_pi(-PI - 0.1) > 0.0);
}

#[test]
fn coincident_nodes_do_not_poison_the_forces() {
    let mut strand = bent_twisted_strand(6, 3.0e-3, 4.0e-2, 80.0);
    strand.nodes[3].position = strand.nodes[2].position;
    for energy in [
        stretch_energy(&strand),
        bending_energy(&strand),
        twisting_energy(&strand),
    ] {
        assert!(energy.is_finite());
    }
    assert!(
        discrete_elastic_rod(&strand)
            .iter()
            .all(|f| f.iter().all(|c| c.is_finite()))
    );
    sync_reference_frames(&mut strand);
    solve_twist_quasistatic(&mut strand);
    step_strand(&mut strand, [0.0, -9.81, 0.0], 1.0e-4);
    assert!(
        strand
            .nodes
            .iter()
            .all(|n| n.position.iter().all(|c| c.is_finite()))
    );
    assert!(strand.theta.iter().all(|t| t.is_finite()));
}

#[test]
fn inconsistent_frame_state_disables_twist_instead_of_panicking() {
    let mut strand = bent_twisted_strand(5, 1.0e-3, 4.0e-2, 80.0);
    assert!(twisting_energy(&strand) > 0.0);
    strand.theta.pop();
    assert_eq!(twisting_energy(&strand), 0.0);
    assert!(node_twists(&strand).iter().all(|m| *m == 0.0));
    assert!(twisting_forces(&strand).iter().all(|f| *f == [0.0; 3]));
    solve_twist_quasistatic(&mut strand);
    sync_reference_frames(&mut strand);
    assert!(
        discrete_elastic_rod(&strand)
            .iter()
            .all(|f| f.iter().all(|c| c.is_finite()))
    );
    strand.reset_reference_frames();
    assert_eq!(strand.theta.len(), strand.num_segments());
    assert!(reference_twists(&strand).iter().all(|a| a.abs() < 1.0e-12));
    assert!(twisting_energy(&strand).abs() < 1.0e-24);
}

#[test]
fn reset_reference_frames_builds_a_bishop_frame_on_a_curved_strand() {
    let mut strand = vertical_strand(6, 1.0e-3, 1.0e-2, 1.0e3);
    for (i, node) in strand.nodes.iter_mut().enumerate() {
        node.position = helix_point(i);
    }
    strand.theta = vec![0.4; 6];
    strand.reset_reference_frames();
    assert_eq!(strand.theta, vec![0.0; 6]);
    for j in 0..6 {
        let tangent = strand.segment_tangent(j).expect("non-degenerate segment");
        assert!(v3_dot(strand.ref_d1[j], tangent).abs() < 1.0e-14);
        assert!((v3_norm(strand.ref_d1[j]) - 1.0).abs() < 1.0e-14);
        assert!(v3_norm(v3_sub(strand.ref_tangent[j], tangent)) < 1.0e-15);
    }
    // Space-parallel transport along the strand is, by construction, twist free.
    for alpha in reference_twists(&strand) {
        assert!(alpha.abs() < 1.0e-12, "reference twist {alpha}");
    }
    assert!(twisting_energy(&strand).abs() < 1.0e-24);
}
