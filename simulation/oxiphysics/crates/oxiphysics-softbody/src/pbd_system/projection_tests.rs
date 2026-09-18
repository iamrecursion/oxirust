//! Regression tests for the constraint projections of the PBD system.
//!
//! They cover the exact tetrahedron-volume and dihedral-angle gradients, the
//! sphere collision constraint and the shared [`project_constraint`] that
//! [`PbdSystem::step`] and [`xpbd_step_colored`] use.

use crate::pbd_system::functions::*;
use crate::pbd_system::types::*;
use std::f64::consts::PI;

const DT: f64 = 1.0 / 60.0;
const FD_STEP: f64 = 1.0e-6;

/// Deterministic pseudo-random numbers in `[-1, 1)`.
struct Lcg(u64);

impl Lcg {
    fn next_unit(&mut self) -> f64 {
        self.0 = self
            .0
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        (self.0 >> 11) as f64 / (1_u64 << 53) as f64 * 2.0 - 1.0
    }

    /// `base` with every coordinate moved by up to `amplitude`.
    fn perturbed(&mut self, base: &[[f64; 3]; 4], amplitude: f64) -> [[f64; 3]; 4] {
        base.map(|point| point.map(|c| c + amplitude * self.next_unit()))
    }
}

fn unit_tet() -> [[f64; 3]; 4] {
    [
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
    ]
}

/// Flat kite hinge: shared edge `(0,0,0)-(1,0,0)`, wings at `(0.5, +-1, 0)`.
fn flat_kite() -> [[f64; 3]; 4] {
    lifted_kite(0.0)
}

/// Kite hinge whose wings are lifted by `lift` along `+z`.
fn lifted_kite(lift: f64) -> [[f64; 3]; 4] {
    [
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.5, 1.0, lift],
        [0.5, -1.0, lift],
    ]
}

/// Interior angle of the symmetric kite whose wings are lifted by `lift`.
fn lifted_kite_angle(lift: f64) -> f64 {
    PI - 2.0 * lift.atan()
}

fn max_abs_diff(a: [f64; 3], b: [f64; 3]) -> f64 {
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| (x - y).abs())
        .fold(0.0, f64::max)
}

fn norm(v: [f64; 3]) -> f64 {
    dist3(v, [0.0; 3])
}

fn system_from(points: &[[f64; 3]], masses: &[f64]) -> PbdSystem {
    let mut system = PbdSystem::new([0.0; 3], 1);
    for (point, mass) in points.iter().zip(masses.iter()) {
        system.add_particle(*point, *mass);
    }
    system
}

fn corners(particles: &[PbdParticle]) -> [[f64; 3]; 4] {
    std::array::from_fn(|k| particles[k].predicted)
}

fn volume_of(particles: &[PbdParticle]) -> f64 {
    let [p0, p1, p2, p3] = corners(particles);
    compute_tetrahedron_volume(p0, p1, p2, p3)
}

fn angle_of(particles: &[PbdParticle]) -> f64 {
    let [p0, p1, p2, p3] = corners(particles);
    compute_dihedral_angle(p0, p1, p2, p3).expect("non-degenerate hinge")
}

/// Wrapped difference of two angles in `(-pi, pi]`.
fn angle_error(angle: f64, target: f64) -> f64 {
    let mut difference = (angle - target) % (2.0 * PI);
    if difference > PI {
        difference -= 2.0 * PI;
    } else if difference <= -PI {
        difference += 2.0 * PI;
    }
    difference
}

/// Central finite-difference gradient of `f` with respect to the four points.
fn numeric_gradients(f: impl Fn(&[[f64; 3]; 4]) -> f64, points: &[[f64; 3]; 4]) -> [[f64; 3]; 4] {
    let mut gradients = [[0.0; 3]; 4];
    for (corner, gradient) in gradients.iter_mut().enumerate() {
        for (axis, component) in gradient.iter_mut().enumerate() {
            let mut plus = *points;
            let mut minus = *points;
            plus[corner][axis] += FD_STEP;
            minus[corner][axis] -= FD_STEP;
            *component = (f(&plus) - f(&minus)) / (2.0 * FD_STEP);
        }
    }
    gradients
}

fn volume_constraint(compliance: f64, rest_volume: f64) -> PbdConstraint {
    PbdConstraint {
        constraint_type: PbdConstraintType::VolumeConservation {
            rest_volume,
            compliance,
        },
        particles: vec![0, 1, 2, 3],
    }
}

fn bending_constraint(compliance: f64, rest_angle: f64) -> PbdConstraint {
    PbdConstraint {
        constraint_type: PbdConstraintType::BendingDihedral {
            rest_angle,
            compliance,
        },
        particles: vec![0, 1, 2, 3],
    }
}

/// Linear momentum `sum m_i dp_i` and angular momentum `sum m_i p_i x dp_i` of a
/// position correction from `before` to the current predicted positions.
fn correction_momenta(before: &[PbdParticle], after: &[PbdParticle]) -> ([f64; 3], [f64; 3]) {
    let mut linear = [0.0; 3];
    let mut angular = [0.0; 3];
    for (b, a) in before.iter().zip(after.iter()) {
        let mass = 1.0 / b.inv_mass;
        let delta = sub3(a.predicted, b.predicted);
        let torque = cross3(b.predicted, delta);
        linear = add3(linear, scale3(delta, mass));
        angular = add3(angular, scale3(torque, mass));
    }
    (linear, angular)
}

// ---------------------------------------------------------------------------
// Tetrahedron volume gradients
// ---------------------------------------------------------------------------

#[test]
fn tet_volume_gradients_of_the_unit_tet_are_exact() {
    let [p0, p1, p2, p3] = unit_tet();
    let g = compute_tetrahedron_volume_gradients(p0, p1, p2, p3);
    let sixth = 1.0 / 6.0;
    let expected = [
        [-sixth, -sixth, -sixth],
        [sixth, 0.0, 0.0],
        [0.0, sixth, 0.0],
        [0.0, 0.0, sixth],
    ];
    for (got, want) in g.iter().zip(expected.iter()) {
        assert!(max_abs_diff(*got, *want) < 1e-15, "{got:?} vs {want:?}");
    }
    // The gradient of p0 is (p3 - p1) x (p2 - p1) / 6.
    let g0 = scale3(cross3(sub3(p3, p1), sub3(p2, p1)), sixth);
    assert!(max_abs_diff(g[0], g0) < 1e-15);
}

#[test]
fn tet_volume_gradients_match_finite_differences() {
    let mut rng = Lcg(7);
    for _ in 0..25 {
        let tet = rng.perturbed(&unit_tet(), 0.3);
        let analytic = compute_tetrahedron_volume_gradients(tet[0], tet[1], tet[2], tet[3]);
        let numeric =
            numeric_gradients(|p| compute_tetrahedron_volume(p[0], p[1], p[2], p[3]), &tet);
        for (a, n) in analytic.iter().zip(numeric.iter()) {
            assert!(max_abs_diff(*a, *n) < 1e-8, "{a:?} vs {n:?}");
        }
        let sum = analytic.iter().fold([0.0; 3], |acc, g| add3(acc, *g));
        assert!(norm(sum) < 1e-14, "gradients must add up to zero: {sum:?}");
    }
}

#[test]
fn tet_volume_gradients_flip_sign_with_the_winding() {
    // Swapping two corners flips the sign of the volume and of every gradient.
    let [p0, p1, p2, p3] = unit_tet();
    let straight = compute_tetrahedron_volume_gradients(p0, p1, p2, p3);
    let swapped = compute_tetrahedron_volume_gradients(p0, p2, p1, p3);
    assert!(max_abs_diff(straight[0], scale3(swapped[0], -1.0)) < 1e-15);
    assert!(max_abs_diff(straight[3], scale3(swapped[3], -1.0)) < 1e-15);
    assert!((compute_tetrahedron_volume(p0, p2, p1, p3) + 1.0 / 6.0).abs() < 1e-15);
}

#[test]
fn volume_projection_restores_a_compressed_tet_monotonically() {
    let rest = unit_tet();
    let rest_volume = compute_tetrahedron_volume(rest[0], rest[1], rest[2], rest[3]);
    let centroid = [0.25; 3];
    // Compress the unit tetrahedron by 10 % about its centroid.
    let compressed = rest.map(|p| add3(centroid, scale3(sub3(p, centroid), 0.9)));
    let mut system = system_from(&compressed, &[1.0; 4]);
    let constraint = volume_constraint(0.0, rest_volume);
    let mut error = (volume_of(&system.particles) - rest_volume).abs();
    assert!(error > 0.25 * rest_volume, "scenario must violate: {error}");
    for _ in 0..12 {
        project_constraint(&mut system.particles, &constraint, DT);
        let now = (volume_of(&system.particles) - rest_volume).abs();
        assert!(
            now < error || now < 1e-15,
            "|V - V0| must shrink: {error} -> {now}"
        );
        error = now;
    }
    assert!(error < 1e-12, "volume must be restored, error {error}");
    let new_centroid = system
        .particles
        .iter()
        .fold([0.0; 3], |acc, p| add3(acc, p.predicted));
    assert!(
        max_abs_diff(scale3(new_centroid, 0.25), centroid) < 1e-12,
        "equal masses keep the centroid"
    );
}

#[test]
fn volume_projection_shrinks_an_expanded_tet() {
    let rest = unit_tet();
    let rest_volume = compute_tetrahedron_volume(rest[0], rest[1], rest[2], rest[3]);
    let expanded = rest.map(|p| scale3(p, 1.2));
    let mut system = system_from(&expanded, &[1.0; 4]);
    let constraint = volume_constraint(0.0, rest_volume);
    for _ in 0..12 {
        project_constraint(&mut system.particles, &constraint, DT);
    }
    let error = (volume_of(&system.particles) - rest_volume).abs();
    assert!(error < 1e-12, "volume must be restored, error {error}");
}

#[test]
fn volume_projection_conserves_momentum_for_unequal_masses() {
    let mut rng = Lcg(11);
    let tet = rng.perturbed(&unit_tet(), 0.1);
    let mut system = system_from(&tet, &[1.0, 2.0, 0.5, 3.0]);
    let rest_volume = volume_of(&system.particles) * 1.3;
    let before = system.particles.clone();
    project_constraint(
        &mut system.particles,
        &volume_constraint(0.0, rest_volume),
        DT,
    );
    let (linear, angular) = correction_momenta(&before, &system.particles);
    assert!(norm(linear) < 1e-13, "linear momentum {linear:?}");
    assert!(norm(angular) < 1e-12, "angular momentum {angular:?}");
    assert!(
        (volume_of(&system.particles) - rest_volume).abs()
            < (volume_of(&before) - rest_volume).abs() * 0.5,
        "one projection must remove at least half of the volume error"
    );
}

#[test]
fn volume_compliance_scales_the_correction() {
    let rest = unit_tet();
    let rest_volume = compute_tetrahedron_volume(rest[0], rest[1], rest[2], rest[3]);
    let squeezed = rest.map(|p| scale3(p, 0.99));
    let squeezed_volume =
        compute_tetrahedron_volume(squeezed[0], squeezed[1], squeezed[2], squeezed[3]);
    let gradients =
        compute_tetrahedron_volume_gradients(squeezed[0], squeezed[1], squeezed[2], squeezed[3]);
    // S = sum w |g|^2 with unit inverse masses.
    let weighted_norm: f64 = gradients.iter().map(|g| dot3(*g, *g)).sum();
    // alpha = compliance / dt^2 = 4 S, so the linearised correction is
    // C * S / (S + alpha) = C / 5.
    let compliance = 4.0 * weighted_norm * DT * DT;
    let mut rigid = system_from(&squeezed, &[1.0; 4]);
    let mut soft = system_from(&squeezed, &[1.0; 4]);
    project_constraint(
        &mut rigid.particles,
        &volume_constraint(0.0, rest_volume),
        DT,
    );
    project_constraint(
        &mut soft.particles,
        &volume_constraint(compliance, rest_volume),
        DT,
    );
    let violation = rest_volume - squeezed_volume;
    let gained_rigid = volume_of(&rigid.particles) - squeezed_volume;
    let gained_soft = volume_of(&soft.particles) - squeezed_volume;
    // The volume is cubic in the positions, so one step is exact only to first
    // order in the (1 %) correction.
    assert!(
        (gained_rigid - violation).abs() < 0.02 * violation,
        "a rigid projection removes the violation: {gained_rigid} vs {violation}"
    );
    assert!(
        (gained_soft / gained_rigid - 0.2).abs() < 0.02,
        "compliance must scale the correction: {}",
        gained_soft / gained_rigid
    );
}

#[test]
fn reversed_winding_tet_is_at_rest_when_the_constraint_is_added() {
    let [p0, p1, p2, p3] = unit_tet();
    // Corners 1 and 2 swapped: the signed volume is -1/6.
    let mut system = system_from(&[p0, p2, p1, p3], &[1.0; 4]);
    system.add_volume_constraint(0, 1, 2, 3, 0.0);
    match system.constraints[0].constraint_type {
        PbdConstraintType::VolumeConservation { rest_volume, .. } => {
            assert!((rest_volume + 1.0 / 6.0).abs() < 1e-15, "{rest_volume}");
        }
        ref other => panic!("expected a volume constraint, got {other:?}"),
    }
    let before = corners(&system.particles);
    project_constraint(&mut system.particles, &system.constraints[0], DT);
    assert_eq!(corners(&system.particles), before, "C = 0: nothing moves");
    // The same holds for a full step without gravity.
    for _ in 0..10 {
        system.step(DT);
    }
    for (point, start) in corners(&system.particles).iter().zip(before.iter()) {
        assert!(max_abs_diff(*point, *start) < 1e-12);
    }
}

#[test]
fn positively_wound_tet_keeps_a_positive_rest_volume() {
    let mut system = system_from(&unit_tet(), &[1.0; 4]);
    system.add_volume_constraint(0, 1, 2, 3, 0.0);
    match system.constraints[0].constraint_type {
        PbdConstraintType::VolumeConservation { rest_volume, .. } => {
            assert!((rest_volume - 1.0 / 6.0).abs() < 1e-15, "{rest_volume}");
        }
        ref other => panic!("expected a volume constraint, got {other:?}"),
    }
}

#[test]
fn volume_projection_does_not_move_pinned_particles() {
    let mut system = system_from(&unit_tet(), &[1.0; 4]);
    system.particles[0] = PbdParticle::new_fixed([0.0; 3]);
    system.particles[3].predicted = [0.0, 0.0, 0.5];
    let rest_volume = 1.0 / 6.0;
    project_constraint(
        &mut system.particles,
        &volume_constraint(0.0, rest_volume),
        DT,
    );
    assert_eq!(system.particles[0].predicted, [0.0; 3]);
    assert!(
        system.particles[3].predicted[2] > 0.5,
        "the free corner grows"
    );
}

// ---------------------------------------------------------------------------
// Dihedral angle and its gradients
// ---------------------------------------------------------------------------

#[test]
fn flat_hinge_has_the_interior_angle_pi() {
    let [p0, p1, p2, p3] = flat_kite();
    let angle = compute_dihedral_angle(p0, p1, p2, p3).expect("flat hinge");
    assert!((angle - PI).abs() < 1e-14, "angle {angle}");
}

#[test]
fn fold_direction_decides_whether_the_angle_is_below_or_above_pi() {
    for lift in [0.3, -0.3, 1.0, -2.0, 0.01] {
        let [p0, p1, p2, p3] = lifted_kite(lift);
        let angle = compute_dihedral_angle(p0, p1, p2, p3).expect("hinge");
        assert!(
            (angle - lifted_kite_angle(lift)).abs() < 1e-12,
            "lift {lift}: angle {angle}, expected {}",
            lifted_kite_angle(lift)
        );
        assert_eq!(
            angle < PI,
            lift > 0.0,
            "wings folded towards the normal (p1-p0)x(p2-p0) give angle < pi"
        );
    }
}

#[test]
fn dihedral_angle_is_invariant_under_rigid_motion() {
    let hinge = lifted_kite(0.4);
    let reference = compute_dihedral_angle(hinge[0], hinge[1], hinge[2], hinge[3]).unwrap();
    // Rotate by 0.9 rad about the axis (1, 2, 2) / 3 and translate.
    let axis = [1.0 / 3.0, 2.0 / 3.0, 2.0 / 3.0];
    let (s, c) = 0.9_f64.sin_cos();
    let moved = hinge.map(|x| {
        let rotated = add3(
            add3(scale3(x, c), scale3(cross3(axis, x), s)),
            scale3(axis, dot3(axis, x) * (1.0 - c)),
        );
        add3(rotated, [3.0, -2.0, 5.0])
    });
    let angle = compute_dihedral_angle(moved[0], moved[1], moved[2], moved[3]).unwrap();
    assert!((angle - reference).abs() < 1e-12, "{angle} vs {reference}");
}

#[test]
fn dihedral_gradients_match_finite_differences() {
    let mut rng = Lcg(3);
    for lift in [0.2, -0.35, 0.0, 0.8, -1.2] {
        for _ in 0..8 {
            let hinge = rng.perturbed(&lifted_kite(lift), 0.15);
            let (_, analytic) =
                dihedral_angle_with_gradients(hinge[0], hinge[1], hinge[2], hinge[3]).unwrap();
            let numeric = numeric_gradients(
                |p| compute_dihedral_angle(p[0], p[1], p[2], p[3]).unwrap(),
                &hinge,
            );
            for (a, n) in analytic.iter().zip(numeric.iter()) {
                assert!(
                    max_abs_diff(*a, *n) < 2e-6 * (1.0 + norm(*n)),
                    "lift {lift}: {a:?} vs {n:?}"
                );
            }
        }
    }
}

#[test]
fn dihedral_gradients_add_up_to_zero_and_carry_no_torque() {
    let mut rng = Lcg(5);
    for _ in 0..20 {
        let hinge = rng.perturbed(&lifted_kite(0.3), 0.2);
        let (_, g) = dihedral_angle_with_gradients(hinge[0], hinge[1], hinge[2], hinge[3]).unwrap();
        let force = g.iter().fold([0.0; 3], |acc, g| add3(acc, *g));
        assert!(norm(force) < 1e-12, "net force {force:?}");
        let torque = hinge
            .iter()
            .zip(g.iter())
            .fold([0.0; 3], |acc, (p, g)| add3(acc, cross3(*p, *g)));
        assert!(norm(torque) < 1e-11, "net torque {torque:?}");
    }
}

#[test]
fn dihedral_gradient_of_the_wings_is_the_inverse_wing_height() {
    // |d theta / d p2| = |e| / |N1| = 1 / (distance of p2 from the edge line).
    let hinge = lifted_kite(0.0);
    let (_, g) = dihedral_angle_with_gradients(hinge[0], hinge[1], hinge[2], hinge[3]).unwrap();
    // Both wings are one unit from the edge line and the normals point along +z;
    // raising a wing lowers the interior angle.
    assert!(max_abs_diff(g[2], [0.0, 0.0, -1.0]) < 1e-14, "{:?}", g[2]);
    assert!(max_abs_diff(g[3], [0.0, 0.0, -1.0]) < 1e-14, "{:?}", g[3]);
    // The edge ends move the opposite way and balance the wings.
    assert!(max_abs_diff(g[0], [0.0, 0.0, 1.0]) < 1e-14, "{:?}", g[0]);
    assert!(max_abs_diff(g[1], [0.0, 0.0, 1.0]) < 1e-14, "{:?}", g[1]);
}

#[test]
fn degenerate_hinges_have_no_dihedral_angle() {
    let flat = flat_kite();
    // Shared edge of zero length.
    assert!(compute_dihedral_angle(flat[0], flat[0], flat[2], flat[3]).is_none());
    // A wing on the line through the shared edge.
    assert!(compute_dihedral_angle(flat[0], flat[1], [0.5, 0.0, 0.0], flat[3]).is_none());
    assert!(compute_dihedral_angle(flat[0], flat[1], flat[2], [2.0, 0.0, 0.0]).is_none());
    // Non-finite input.
    assert!(compute_dihedral_angle(flat[0], flat[1], [f64::NAN, 1.0, 0.0], flat[3]).is_none());
}

// ---------------------------------------------------------------------------
// Dihedral bending constraint
// ---------------------------------------------------------------------------

#[test]
fn bent_hinge_relaxes_to_the_flat_angle_for_both_fold_directions() {
    for lift in [0.5, -0.5, 1.5, -1.5] {
        let mut system = system_from(&lifted_kite(lift), &[1.0; 4]);
        let constraint = bending_constraint(0.0, PI);
        let mut error = angle_error(angle_of(&system.particles), PI).abs();
        assert!(error > 0.5, "scenario must be bent: {error}");
        for _ in 0..8 {
            project_constraint(&mut system.particles, &constraint, DT);
            let now = angle_error(angle_of(&system.particles), PI).abs();
            assert!(
                now < error || now < 1e-12,
                "lift {lift}: |theta - pi| {error} -> {now}"
            );
            error = now;
        }
        assert!(error < 1e-9, "lift {lift}: residual angle error {error}");
    }
}

#[test]
fn bending_moves_all_four_particles_and_conserves_momentum() {
    let mut rng = Lcg(21);
    let hinge = rng.perturbed(&lifted_kite(0.4), 0.1);
    let mut system = system_from(&hinge, &[1.0, 2.0, 0.5, 3.0]);
    let before = system.particles.clone();
    project_constraint(&mut system.particles, &bending_constraint(0.0, PI), DT);
    for (k, (b, a)) in before.iter().zip(system.particles.iter()).enumerate() {
        assert!(
            dist3(a.predicted, b.predicted) > 1e-6,
            "particle {k} (hinge edge: 0 and 1) must take part in the correction"
        );
    }
    let (linear, angular) = correction_momenta(&before, &system.particles);
    assert!(norm(linear) < 1e-13, "linear momentum {linear:?}");
    assert!(norm(angular) < 1e-12, "angular momentum {angular:?}");
}

#[test]
fn bending_correction_follows_the_inverse_mass_weights() {
    // Heavy wings barely move, light edge ends take the correction.
    let mut system = system_from(&lifted_kite(0.4), &[0.1, 0.1, 100.0, 100.0]);
    let before = system.particles.clone();
    project_constraint(&mut system.particles, &bending_constraint(0.0, PI), DT);
    let edge_move = dist3(system.particles[0].predicted, before[0].predicted);
    let wing_move = dist3(system.particles[2].predicted, before[2].predicted);
    assert!(edge_move > 100.0 * wing_move, "{edge_move} vs {wing_move}");
}

#[test]
fn non_flat_rest_angle_is_reached_from_either_side() {
    let rest_angles = [2.0, 2.0 * PI - 2.0];
    for rest_angle in rest_angles {
        for lift in [0.0, -0.8, 0.8] {
            let mut system = system_from(&lifted_kite(lift), &[1.0; 4]);
            let constraint = bending_constraint(0.0, rest_angle);
            for _ in 0..40 {
                project_constraint(&mut system.particles, &constraint, DT);
            }
            let error = angle_error(angle_of(&system.particles), rest_angle).abs();
            assert!(
                error < 1e-9,
                "rest {rest_angle}, start lift {lift}: angle error {error}"
            );
        }
    }
}

#[test]
fn equivalent_rest_angles_give_the_same_projection() {
    let start = lifted_kite(0.7);
    let mut a = system_from(&start, &[1.0; 4]);
    let mut b = system_from(&start, &[1.0; 4]);
    project_constraint(&mut a.particles, &bending_constraint(0.0, 2.5), DT);
    project_constraint(
        &mut b.particles,
        &bending_constraint(0.0, 2.5 + 2.0 * PI),
        DT,
    );
    for (pa, pb) in a.particles.iter().zip(b.particles.iter()) {
        assert!(max_abs_diff(pa.predicted, pb.predicted) < 1e-12);
    }
}

#[test]
fn bending_compliance_softens_the_correction() {
    let start = lifted_kite(0.4);
    let mut rigid = system_from(&start, &[1.0; 4]);
    let mut soft = system_from(&start, &[1.0; 4]);
    project_constraint(&mut rigid.particles, &bending_constraint(0.0, PI), DT);
    project_constraint(&mut soft.particles, &bending_constraint(1.0e-3, PI), DT);
    let rigid_error = angle_error(angle_of(&rigid.particles), PI).abs();
    let soft_error = angle_error(angle_of(&soft.particles), PI).abs();
    let initial = angle_error(lifted_kite_angle(0.4), PI).abs();
    assert!(
        soft_error < initial,
        "compliant projection still bends back"
    );
    assert!(
        soft_error > 10.0 * rigid_error + 1e-6,
        "compliance must limit the correction: {soft_error} vs {rigid_error}"
    );
}

#[test]
fn pinned_hinge_edge_stays_put_and_the_wings_flatten() {
    let mut system = system_from(&lifted_kite(0.6), &[1.0; 4]);
    system.particles[0] = PbdParticle::new_fixed([0.0, 0.0, 0.0]);
    system.particles[1] = PbdParticle::new_fixed([1.0, 0.0, 0.0]);
    let constraint = bending_constraint(0.0, PI);
    for _ in 0..30 {
        project_constraint(&mut system.particles, &constraint, DT);
    }
    assert_eq!(system.particles[0].predicted, [0.0, 0.0, 0.0]);
    assert_eq!(system.particles[1].predicted, [1.0, 0.0, 0.0]);
    let error = angle_error(angle_of(&system.particles), PI).abs();
    assert!(error < 1e-9, "wings must flatten: {error}");
}

#[test]
fn degenerate_or_malformed_bending_constraints_leave_the_particles_alone() {
    let flat = flat_kite();
    // Collapsed hinge: both wings on the shared-edge line.
    let collapsed = [flat[0], flat[1], [0.5, 0.0, 0.0], [0.25, 0.0, 0.0]];
    let mut system = system_from(&collapsed, &[1.0; 4]);
    let before = corners(&system.particles);
    project_constraint(&mut system.particles, &bending_constraint(0.0, 2.0), DT);
    assert_eq!(corners(&system.particles), before);

    let mut system = system_from(&lifted_kite(0.5), &[1.0; 4]);
    let before = corners(&system.particles);
    let too_few = PbdConstraint {
        constraint_type: PbdConstraintType::BendingDihedral {
            rest_angle: PI,
            compliance: 0.0,
        },
        particles: vec![0, 1, 2],
    };
    let out_of_range = PbdConstraint {
        particles: vec![0, 1, 2, 9],
        ..too_few.clone()
    };
    let repeated = PbdConstraint {
        particles: vec![0, 1, 2, 2],
        ..too_few.clone()
    };
    for constraint in [&too_few, &out_of_range, &repeated] {
        project_constraint(&mut system.particles, constraint, DT);
        assert_eq!(corners(&system.particles), before, "{constraint:?}");
    }
}

#[test]
fn bending_system_step_flattens_a_free_hinge() {
    let mut system = PbdSystem::new([0.0; 3], 4);
    for point in lifted_kite(0.8) {
        system.add_particle(point, 1.0);
    }
    system.add_bending_constraint(0, 1, 2, 3, PI, 0.0);
    for _ in 0..30 {
        system.step(DT);
    }
    let error = angle_error(
        compute_dihedral_angle(
            system.particles[0].position,
            system.particles[1].position,
            system.particles[2].position,
            system.particles[3].position,
        )
        .unwrap(),
        PI,
    )
    .abs();
    assert!(error < 1e-6, "angle error after the steps: {error}");
}

// ---------------------------------------------------------------------------
// Sphere collision constraint
// ---------------------------------------------------------------------------

#[test]
fn sphere_helper_adds_a_sphere_constraint_not_a_floor() {
    let mut system = PbdSystem::new([0.0; 3], 1);
    let i = system.add_particle([5.0, 0.0, 0.0], 1.0);
    add_sphere_collision_constraint(&mut system, i, [1.0, 2.0, 3.0], 0.75);
    assert_eq!(system.constraints.len(), 1);
    assert_eq!(system.constraints[0].particles, vec![i]);
    match system.constraints[0].constraint_type {
        PbdConstraintType::CollisionSphere {
            center,
            radius,
            restitution,
        } => {
            assert_eq!(center, [1.0, 2.0, 3.0]);
            assert_eq!(radius, 0.75);
            assert_eq!(restitution, 0.0);
        }
        ref other => panic!("expected a sphere constraint, got {other:?}"),
    }
    system.add_sphere_constraint(i, [0.0; 3], 2.0, 0.4);
    match system.constraints[1].constraint_type {
        PbdConstraintType::CollisionSphere { restitution, .. } => assert_eq!(restitution, 0.4),
        ref other => panic!("expected a sphere constraint, got {other:?}"),
    }
}

#[test]
fn particle_inside_a_sphere_is_pushed_onto_its_surface() {
    let center = [1.0, 2.0, 3.0];
    let radius = 2.0;
    let mut system = PbdSystem::new([0.0; 3], 1);
    let i = system.add_particle([center[0] + 0.5 * radius, center[1], center[2]], 1.0);
    add_sphere_collision_constraint(&mut system, i, center, radius);
    project_constraint(&mut system.particles, &system.constraints[0], DT);
    let p = system.particles[i].predicted;
    assert!((dist3(p, center) - radius).abs() < 1e-12, "{p:?}");
    assert!(max_abs_diff(p, [center[0] + radius, center[1], center[2]]) < 1e-12);
}

#[test]
fn sphere_pushes_along_the_radial_direction() {
    let center = [0.0, 0.0, 0.0];
    let mut system = PbdSystem::new([0.0; 3], 1);
    let i = system.add_particle([0.3, -0.4, 0.0], 1.0);
    add_sphere_collision_constraint(&mut system, i, center, 2.0);
    project_constraint(&mut system.particles, &system.constraints[0], DT);
    let p = system.particles[i].predicted;
    assert!(max_abs_diff(p, [1.2, -1.6, 0.0]) < 1e-12, "{p:?}");
}

#[test]
fn particle_outside_a_sphere_is_untouched_wherever_it_is() {
    let mut system = PbdSystem::new([0.0; 3], 1);
    // Far away; just below the height of the sphere top but far to the side (the
    // old implementation lifted it onto the plane y = top); and underneath.
    let outside = [
        [10.0, -5.0, 0.0],
        [5.0, 0.5, 0.0],
        [0.0, -3.0, 0.0],
        [0.0, 0.0, 1.5],
    ];
    for point in outside {
        let i = system.add_particle(point, 1.0);
        add_sphere_collision_constraint(&mut system, i, [0.0; 3], 1.0);
    }
    for constraint in &system.constraints {
        project_constraint(&mut system.particles, constraint, DT);
    }
    for (particle, point) in system.particles.iter().zip(outside.iter()) {
        assert_eq!(particle.predicted, *point);
    }
}

#[test]
fn pinned_particle_is_not_pushed_out_of_a_sphere() {
    let mut system = PbdSystem::new([0.0; 3], 1);
    let i = system.add_fixed_particle([0.2, 0.0, 0.0]);
    add_sphere_collision_constraint(&mut system, i, [0.0; 3], 1.0);
    project_constraint(&mut system.particles, &system.constraints[0], DT);
    assert_eq!(system.particles[i].predicted, [0.2, 0.0, 0.0]);
}

#[test]
fn particle_at_the_sphere_centre_leaves_along_its_previous_direction() {
    let center = [1.0, 1.0, 1.0];
    let mut particle = PbdParticle::new([1.0, 1.0, 0.5], 1.0);
    particle.predicted = center;
    solve_sphere_collision(&mut particle, center, 2.0, 0.0);
    assert!(max_abs_diff(particle.predicted, [1.0, 1.0, -1.0]) < 1e-12);

    // Without a usable previous direction it leaves along +y.
    let mut particle = PbdParticle::new(center, 1.0);
    solve_sphere_collision(&mut particle, center, 2.0, 0.0);
    assert!(max_abs_diff(particle.predicted, [1.0, 3.0, 1.0]) < 1e-12);
}

#[test]
fn sphere_restitution_reflects_the_inward_velocity() {
    let mut particle = PbdParticle::new([0.5, 0.0, 0.0], 1.0);
    particle.velocity = [-4.0, 1.0, 0.0];
    solve_sphere_collision(&mut particle, [0.0; 3], 1.0, 0.5);
    assert!((particle.predicted[0] - 1.0).abs() < 1e-12);
    assert!(
        (particle.velocity[0] - 2.0).abs() < 1e-12,
        "{:?}",
        particle.velocity
    );
    assert!(
        (particle.velocity[1] - 1.0).abs() < 1e-12,
        "tangential part is kept"
    );
}

#[test]
fn falling_particle_slides_off_a_sphere_instead_of_resting_on_a_plane() {
    let radius = 1.0;
    let mut system = PbdSystem::new([0.0, -9.81, 0.0], 10);
    let i = system.add_particle([0.3, 1.5, 0.0], 1.0);
    add_sphere_collision_constraint(&mut system, i, [0.0; 3], radius);
    let mut closest = f64::INFINITY;
    for _ in 0..150 {
        system.step(DT);
        let distance = norm(system.particles[i].position);
        assert!(
            distance >= radius - 1e-9,
            "penetration: distance {distance}"
        );
        closest = closest.min(distance);
    }
    assert!(
        closest < radius + 1e-6,
        "the particle must touch the sphere: {closest}"
    );
    let p = system.particles[i].position;
    // A plane at the sphere top would have held the particle at y = 1; the
    // sphere lets it slide off the side and fall.
    assert!(p[1] < -5.0, "particle must fall past the sphere: {p:?}");
    assert!(p[0] > 0.3, "particle must have slid sideways: {p:?}");
}

// ---------------------------------------------------------------------------
// Shared projection: step vs. colored step
// ---------------------------------------------------------------------------

/// A tetrahedron with every constraint type; each constraint contains particle
/// 0, so the greedy coloring puts every constraint into a group of its own, in
/// index order.
fn busy_tet() -> PbdSystem {
    let mut system = PbdSystem::new([0.0, -9.81, 0.0], 3);
    let masses = [1.0, 2.0, 0.5, 3.0];
    let tet = Lcg(13).perturbed(&unit_tet(), 0.1);
    for (point, mass) in tet.iter().zip(masses.iter()) {
        system.add_particle(*point, *mass);
    }
    system.add_volume_constraint(0, 1, 2, 3, 1.0e-6);
    let angle = angle_of(&system.particles);
    system.add_bending_constraint(0, 1, 2, 3, angle, 1.0e-5);
    system.add_distance_constraint(0, 1, 0.0);
    system.add_distance_constraint(0, 2, 1.0e-6);
    system.add_distance_constraint(0, 3, 0.0);
    system.add_floor_constraint(0, -1.0, 0.0);
    system.add_sphere_constraint(0, [0.2, -0.3, 0.1], 0.5, 0.0);
    // Compress the tetrahedron so the volume and bending constraints are active.
    for particle in system.particles.iter_mut() {
        particle.position = scale3(particle.position, 0.8);
        particle.predicted = particle.position;
    }
    system
}

#[test]
fn colored_step_is_identical_to_the_plain_step_for_every_constraint_type() {
    let mut plain = busy_tet();
    let mut colored = busy_tet();
    let coloring = ConstraintColoring::build(&colored.constraints, colored.particles.len());
    assert_eq!(coloring.num_colors(), colored.constraints.len());
    for group in &coloring.groups {
        assert_eq!(group.len(), 1, "every constraint shares particle 0");
    }
    for step in 0..40 {
        plain.step(DT);
        xpbd_step_colored(&mut colored, DT, &coloring);
        for (a, b) in plain.particles.iter().zip(colored.particles.iter()) {
            assert_eq!(a.position, b.position, "positions differ in step {step}");
            assert_eq!(a.velocity, b.velocity, "velocities differ in step {step}");
        }
    }
    for particle in &plain.particles {
        assert!(particle.position.iter().all(|c| c.is_finite()));
    }
    // The scenario really exercises the bending, volume and sphere projections.
    let violation: f64 = plain
        .constraints
        .iter()
        .map(|c| constraint_violation(&plain.particles, c))
        .sum();
    assert!(violation.is_finite());
    let moved = dist3(
        plain.particles[0].position,
        busy_tet().particles[0].position,
    );
    assert!(moved > 0.5, "the tetrahedron must have fallen: {moved}");
}

#[test]
fn colored_step_enforces_bending_and_volume_constraints() {
    let mut system = PbdSystem::new([0.0; 3], 4);
    for point in lifted_kite(0.8) {
        system.add_particle(point, 1.0);
    }
    system.add_bending_constraint(0, 1, 2, 3, PI, 0.0);
    let coloring = ConstraintColoring::build(&system.constraints, system.particles.len());
    for _ in 0..30 {
        xpbd_step_colored(&mut system, DT, &coloring);
    }
    let [p0, p1, p2, p3] = std::array::from_fn(|k| system.particles[k].position);
    let angle = compute_dihedral_angle(p0, p1, p2, p3).unwrap();
    assert!(
        angle_error(angle, PI).abs() < 1e-6,
        "the colored step must flatten the hinge: angle {angle}"
    );

    let mut system = PbdSystem::new([0.0; 3], 4);
    for point in unit_tet() {
        system.add_particle(point, 1.0);
    }
    system.add_volume_constraint(0, 1, 2, 3, 0.0);
    for particle in system.particles.iter_mut() {
        particle.position = scale3(particle.position, 1.25);
        particle.predicted = particle.position;
    }
    let coloring = ConstraintColoring::build(&system.constraints, system.particles.len());
    for _ in 0..30 {
        xpbd_step_colored(&mut system, DT, &coloring);
    }
    let volume = compute_tetrahedron_volume(
        system.particles[0].position,
        system.particles[1].position,
        system.particles[2].position,
        system.particles[3].position,
    );
    assert!(
        (volume - 1.0 / 6.0).abs() < 1e-9,
        "the colored step must restore the volume: {volume}"
    );
}

#[test]
fn colored_step_skips_color_indices_without_a_constraint() {
    let mut system = PbdSystem::new([0.0, -9.81, 0.0], 1);
    system.add_particle([0.0, 1.0, 0.0], 1.0);
    let coloring = ConstraintColoring {
        groups: vec![vec![0, 7, 99]],
    };
    xpbd_step_colored(&mut system, DT, &coloring);
    assert!(system.particles[0].position[1] < 1.0);
}

#[test]
fn malformed_constraints_are_ignored_by_step_and_projection() {
    let mut system = PbdSystem::new([0.0; 3], 2);
    for point in unit_tet() {
        system.add_particle(point, 1.0);
    }
    // (constraint, number of particles it needs) with deliberately wrong index
    // lists: empty, too short, and with an index outside the system.
    let kinds = [
        (
            PbdConstraintType::Distance {
                rest_length: 2.0,
                compliance: 0.0,
            },
            2,
        ),
        (
            PbdConstraintType::BendingDihedral {
                rest_angle: PI,
                compliance: 0.0,
            },
            4,
        ),
        (
            PbdConstraintType::VolumeConservation {
                rest_volume: 1.0,
                compliance: 0.0,
            },
            4,
        ),
        (
            PbdConstraintType::CollisionFloor {
                y_level: 10.0,
                restitution: 0.0,
            },
            1,
        ),
        (
            PbdConstraintType::CollisionSphere {
                center: [0.0; 3],
                radius: 10.0,
                restitution: 0.0,
            },
            1,
        ),
    ];
    for (kind, needed) in kinds {
        let valid: Vec<usize> = (0..needed).collect();
        let too_short = valid[..needed - 1].to_vec();
        let mut out_of_range = valid.clone();
        out_of_range[needed - 1] = 40;
        for particles in [Vec::new(), too_short, out_of_range] {
            system.constraints.push(PbdConstraint {
                constraint_type: kind.clone(),
                particles,
            });
        }
    }
    let before = corners(&system.particles);
    for constraint in &system.constraints {
        project_constraint(&mut system.particles, constraint, DT);
        assert_eq!(corners(&system.particles), before, "{constraint:?}");
    }
    let coloring = ConstraintColoring::build(&system.constraints, system.particles.len());
    system.step(DT);
    xpbd_step_colored(&mut system, DT, &coloring);
    for (particle, start) in system.particles.iter().zip(before.iter()) {
        assert!(max_abs_diff(particle.position, *start) < 1e-12);
    }
}

// ---------------------------------------------------------------------------
// Constraint violation and the profiled residual
// ---------------------------------------------------------------------------

#[test]
fn constraint_violation_reports_every_constraint_type() {
    let tolerance = 1e-12;
    // Distance.
    let mut system = system_from(&[[0.0; 3], [3.0, 0.0, 0.0]], &[1.0, 1.0]);
    system.add_distance_constraint(0, 1, 0.0);
    system.particles[1].position = [4.0, 0.0, 0.0];
    assert!(
        (constraint_violation(&system.particles, &system.constraints[0]) - 1.0).abs() < tolerance
    );

    // Bending: wings lifted by 0.5 against a flat rest angle.
    let bent = system_from(&lifted_kite(0.5), &[1.0; 4]);
    let violation = constraint_violation(&bent.particles, &bending_constraint(0.0, PI));
    assert!(
        (violation - 2.0 * 0.5_f64.atan()).abs() < tolerance,
        "{violation}"
    );

    // Volume.
    let scaled = system_from(&unit_tet().map(|p| scale3(p, 1.1)), &[1.0; 4]);
    let violation = constraint_violation(&scaled.particles, &volume_constraint(0.0, 1.0 / 6.0));
    assert!(
        (violation - (1.331 - 1.0) / 6.0).abs() < tolerance,
        "{violation}"
    );

    // Floor and sphere penetration depth.
    let mut system = system_from(&[[0.0, -0.25, 0.0], [0.4, 0.0, 0.0]], &[1.0, 1.0]);
    system.add_floor_constraint(0, 0.0, 0.0);
    system.add_sphere_constraint(1, [0.0; 3], 1.0, 0.0);
    assert!(
        (constraint_violation(&system.particles, &system.constraints[0]) - 0.25).abs() < tolerance
    );
    assert!(
        (constraint_violation(&system.particles, &system.constraints[1]) - 0.6).abs() < tolerance
    );
    // No violation above the floor or outside the sphere.
    system.particles[0].position = [0.0, 1.0, 0.0];
    system.particles[1].position = [3.0, 0.0, 0.0];
    assert_eq!(
        constraint_violation(&system.particles, &system.constraints[0]),
        0.0
    );
    assert_eq!(
        constraint_violation(&system.particles, &system.constraints[1]),
        0.0
    );

    // A constraint that cannot be evaluated reports no violation.
    let malformed = PbdConstraint {
        constraint_type: PbdConstraintType::VolumeConservation {
            rest_volume: 1.0,
            compliance: 0.0,
        },
        particles: vec![0, 1],
    };
    assert_eq!(constraint_violation(&system.particles, &malformed), 0.0);
}

#[test]
fn profiled_residual_includes_bending_volume_and_sphere_constraints() {
    // Compliances so large that a step does not change the violated state.
    let mut profiled = ProfiledPbdSystem::new([0.0; 3], 1);
    for point in lifted_kite(0.5) {
        profiled.system.add_particle(point, 1.0);
    }
    profiled
        .system
        .add_bending_constraint(0, 1, 2, 3, PI, 1.0e12);
    profiled.step(DT);
    let bending_only = profiled.last_profile.residual;
    assert!(
        (bending_only - 2.0 * 0.5_f64.atan()).abs() < 1e-6,
        "bending residual {bending_only}"
    );

    // A volume constraint whose rest volume is twice the current one.
    profiled.system.add_volume_constraint(0, 1, 2, 3, 1.0e12);
    let doubled = match profiled.system.constraints.last_mut() {
        Some(PbdConstraint {
            constraint_type: PbdConstraintType::VolumeConservation { rest_volume, .. },
            ..
        }) => {
            *rest_volume *= 2.0;
            *rest_volume
        }
        other => panic!("expected the volume constraint, got {other:?}"),
    };
    profiled.step(DT);
    let with_volume = profiled.last_profile.residual;
    assert!(
        (with_volume - bending_only - doubled / 2.0).abs() < 1e-6,
        "volume residual {} vs {}",
        with_volume - bending_only,
        doubled / 2.0
    );

    // A pinned particle inside a sphere stays inside: its penetration counts.
    let pinned = profiled.system.add_fixed_particle([0.0, 0.0, 0.0]);
    profiled
        .system
        .add_sphere_constraint(pinned, [0.0, 0.0, 0.1], 0.3, 0.0);
    profiled.step(DT);
    assert!(
        (profiled.last_profile.residual - with_volume - 0.2).abs() < 1e-6,
        "sphere residual {}",
        profiled.last_profile.residual - with_volume
    );
}

// ---------------------------------------------------------------------------
// A strip with bending constraints behaves like a plate, not like a chain
// ---------------------------------------------------------------------------

/// A two-particle-wide strip along `+x` pinned at its first column and falling
/// under gravity, together with the hinges of its triangulation: the diagonal
/// of every quad and every cross edge between two quads.  Bending constraints
/// with the given compliance are added when `bending` is `Some`.
fn hanging_strip(columns: usize, bending: Option<f64>) -> (PbdSystem, Vec<[usize; 4]>) {
    let index = |column: usize, row: usize| column * 2 + row;
    let mut system = PbdSystem::new([0.0, -9.81, 0.0], 10);
    for column in 0..columns {
        for row in 0..2 {
            let position = [column as f64, 0.0, row as f64];
            if column == 0 {
                system.add_fixed_particle(position);
            } else {
                system.add_particle(position, 0.1);
            }
        }
    }
    for column in 0..columns {
        system.add_distance_constraint(index(column, 0), index(column, 1), 0.0);
        if column + 1 < columns {
            system.add_distance_constraint(index(column, 0), index(column + 1, 0), 0.0);
            system.add_distance_constraint(index(column, 1), index(column + 1, 1), 0.0);
            system.add_distance_constraint(index(column, 1), index(column + 1, 0), 0.0);
        }
    }
    let mut hinges = Vec::new();
    for column in 0..columns - 1 {
        hinges.push([
            index(column, 1),
            index(column + 1, 0),
            index(column, 0),
            index(column + 1, 1),
        ]);
    }
    for column in 1..columns - 1 {
        hinges.push([
            index(column, 0),
            index(column, 1),
            index(column - 1, 0),
            index(column + 1, 1),
        ]);
    }
    if let Some(compliance) = bending {
        for hinge in &hinges {
            system.add_bending_constraint(hinge[0], hinge[1], hinge[2], hinge[3], PI, compliance);
        }
    }
    (system, hinges)
}

/// Runs the strip for four seconds with the plain or the colored solver and
/// returns the largest deviation of a hinge angle from flat that occurred.
fn worst_hinge_angle_error(bending: Option<f64>, colored: bool) -> f64 {
    let (mut system, hinges) = hanging_strip(8, bending);
    let coloring = ConstraintColoring::build(&system.constraints, system.particles.len());
    let mut worst: f64 = 0.0;
    for _ in 0..240 {
        if colored {
            xpbd_step_colored(&mut system, DT, &coloring);
        } else {
            system.step(DT);
        }
        for hinge in &hinges {
            let [p0, p1, p2, p3] = hinge.map(|k| system.particles[k].position);
            let angle = compute_dihedral_angle(p0, p1, p2, p3).expect("strip hinge");
            worst = worst.max(angle_error(angle, PI).abs());
        }
    }
    worst
}

#[test]
fn strip_with_bending_constraints_swings_like_a_plate() {
    // Pinned along one edge, the strip swings like a pendulum.  Without bending
    // constraints it folds up at the cross edges; with them it stays flat, and
    // a compliant hinge sits in between.
    for colored in [false, true] {
        let free = worst_hinge_angle_error(None, colored);
        let rigid = worst_hinge_angle_error(Some(0.0), colored);
        let soft = worst_hinge_angle_error(Some(1.0e-3), colored);
        assert!(free > 0.5, "colored {colored}: free strip folds, {free}");
        assert!(rigid < 5e-3, "colored {colored}: rigid hinges, {rigid}");
        assert!(
            rigid < soft && soft < 0.05,
            "colored {colored}: compliant hinges: {rigid} < {soft} < 0.05"
        );
    }
}

#[test]
fn strip_with_rigid_bending_keeps_its_length_while_swinging() {
    let (mut system, _) = hanging_strip(8, Some(0.0));
    let tip = 2 * 7;
    for step in 0..240 {
        system.step(DT);
        // The strip is a straight rod of length 7 about the pinned edge.
        let distance = norm(system.particles[tip].position);
        assert!(
            (distance - 7.0).abs() < 0.05,
            "step {step}: the tip is {distance} from the pivot"
        );
    }
}
