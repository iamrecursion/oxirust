// Copyright 2026 COOLJAPAN OU (Team KitaSan)
// SPDX-License-Identifier: Apache-2.0

//! Regression tests for the isometric bending, CST strain and Neo-Hookean
//! constraints of [`crate::pbd_constraints`].
//!
//! The old bending matrix was a made-up diagonal form (not translation
//! invariant, non-zero for a flat sheet), the CST deformation gradient applied
//! the transpose of `Dm⁻¹` (so a resting equilateral triangle violated its own
//! constraint) and the Neo-Hookean constraint took columns instead of rows of
//! `Dm⁻¹` and ignored the Lamé parameters.  Every scenario below uses a hinge,
//! triangle or tetrahedron whose `Dm⁻¹` is *not* symmetric, because the right
//! triangle and the unit tetrahedron hide transposition mistakes.

use super::{CSTStrainConstraint, IsometricBendingConstraint, NeoHookeanConstraint};
use crate::constraint::SoftConstraint;
use crate::particle::SoftParticle;
use oxiphysics_core::math::{Mat3, Real, Vec3};

const DT: Real = 1.0 / 60.0;

/// Deterministic pseudo-random numbers in `[0, 1)`.
struct Lcg(u64);

impl Lcg {
    fn next_unit(&mut self) -> Real {
        self.0 = self
            .0
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        (self.0 >> 11) as Real / (1_u64 << 53) as Real
    }
}

/// Rodrigues rotation of `x` about the unit `axis` by `angle`.
fn rotate_about(axis: Vec3, angle: Real, x: Vec3) -> Vec3 {
    let (s, c) = angle.sin_cos();
    x * c + axis.cross(&x) * s + axis * (axis.dot(&x) * (1.0 - c))
}

fn particles_with_masses(points: &[[Real; 3]], masses: &[Real]) -> Vec<SoftParticle> {
    points
        .iter()
        .zip(masses.iter())
        .map(|(p, &mass)| SoftParticle::new(Vec3::new(p[0], p[1], p[2]), mass))
        .collect()
}

fn unit_masses(points: &[[Real; 3]]) -> Vec<SoftParticle> {
    let masses = vec![1.0; points.len()];
    particles_with_masses(points, &masses)
}

/// Rotates and translates the particles rigidly (axis `(1, 2, 2) / 3`).
fn rigidly_moved(particles: &[SoftParticle], angle: Real, shift: Vec3) -> Vec<SoftParticle> {
    let axis = Vec3::new(1.0, 2.0, 2.0) / 3.0;
    let (s, c) = angle.sin_cos();
    particles
        .iter()
        .map(|p| {
            let x = p.position;
            let rotated = x * c + axis.cross(&x) * s + axis * (axis.dot(&x) * (1.0 - c));
            SoftParticle::new(rotated + shift, 1.0 / p.inverse_mass)
        })
        .collect()
}

/// Linear and angular momentum `sum m dx`, `sum m x x dx` of a position
/// correction from `before` to `after`.
fn correction_momenta(before: &[SoftParticle], after: &[SoftParticle]) -> (Vec3, Vec3) {
    let mut linear = Vec3::zeros();
    let mut angular = Vec3::zeros();
    for (b, a) in before.iter().zip(after.iter()) {
        let mass = 1.0 / b.inverse_mass;
        let delta = a.position - b.position;
        linear += delta * mass;
        angular += b.position.cross(&delta) * mass;
    }
    (linear, angular)
}

fn max_deviation(a: &[SoftParticle], b: &[SoftParticle]) -> Real {
    a.iter()
        .zip(b.iter())
        .map(|(pa, pb)| (pa.position - pb.position).norm())
        .fold(0.0, Real::max)
}

/// Central-difference gradient of a scalar function of the particle positions.
fn numeric_gradient(
    particles: &[SoftParticle],
    value: impl Fn(&[SoftParticle]) -> Real,
) -> Vec<Vec3> {
    let h = 1e-6;
    let mut work = particles.to_vec();
    let mut gradient = vec![Vec3::zeros(); particles.len()];
    for (k, grad) in gradient.iter_mut().enumerate() {
        for axis in 0..3 {
            let original = work[k].position[axis];
            work[k].position[axis] = original + h;
            let plus = value(&work);
            work[k].position[axis] = original - h;
            let minus = value(&work);
            work[k].position[axis] = original;
            grad[axis] = (plus - minus) / (2.0 * h);
        }
    }
    gradient
}

// ---------------------------------------------------------------------------
// IsometricBendingConstraint
// ---------------------------------------------------------------------------

/// Symmetric kite: shared edge `(0,0,0)-(1,0,0)`, wings at `(0.5, +-1, 0)`.
fn kite() -> Vec<SoftParticle> {
    unit_masses(&[
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.5, 1.0, 0.0],
        [0.5, -1.0, 0.0],
    ])
}

/// Planar hinge with scalene triangles and wings on opposite sides of the edge.
fn skewed_flat() -> Vec<SoftParticle> {
    unit_masses(&[
        [0.0, 0.0, 0.0],
        [1.3, 0.2, 0.0],
        [0.4, 1.1, 0.0],
        [1.0, -0.8, 0.0],
    ])
}

/// The kite whose wings are turned about the shared edge so that the angle
/// between the two face planes deviates from flat by `fold` radians.  The
/// triangles keep their shape.
fn folded_kite(fold: Real) -> Vec<SoftParticle> {
    let (sin_half, cos_half) = (0.5 * fold).sin_cos();
    unit_masses(&[
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.5, cos_half, sin_half],
        [0.5, -cos_half, sin_half],
    ])
}

fn hinge_weights_of(particles: &[SoftParticle]) -> ([Real; 4], Real) {
    IsometricBendingConstraint::hinge_weights(
        particles[0].position,
        particles[1].position,
        particles[2].position,
        particles[3].position,
    )
    .expect("non-degenerate hinge")
}

#[test]
fn hinge_weights_of_the_symmetric_kite() {
    // All base angles have the cotangent 1/2: K = [1, 1, -1, -1]; the areas add
    // up to 1.
    let (k, scale) = hinge_weights_of(&kite());
    for (got, want) in k.iter().zip([1.0, 1.0, -1.0, -1.0]) {
        assert!((got - want).abs() < 1e-12, "K = {k:?}");
    }
    assert!((scale - 3.0).abs() < 1e-12, "3 / (A0 + A1) = {scale}");
}

#[test]
fn hinge_weights_sum_to_zero_and_annihilate_planar_positions() {
    let moved = rigidly_moved(&skewed_flat(), 1.1, Vec3::new(3.0, -2.0, 5.0));
    for hinge in [kite(), skewed_flat(), moved] {
        let (k, _) = hinge_weights_of(&hinge);
        assert!(k.iter().sum::<Real>().abs() < 1e-12, "sum(K) = {k:?}");
        let weighted = k
            .iter()
            .zip(hinge.iter())
            .fold(Vec3::zeros(), |acc, (k_a, p)| acc + p.position * *k_a);
        assert!(weighted.norm() < 1e-11, "K . p = {weighted:?}");
    }
}

#[test]
fn q_matrix_is_symmetric_and_has_zero_row_sums() {
    for hinge in [kite(), skewed_flat(), folded_kite(0.7)] {
        let c = IsometricBendingConstraint::new([0, 1, 2, 3], &hinge, 0.0);
        let q = c.q_matrix();
        for row in 0..4 {
            let sum: Real = (0..4).map(|col| q[row * 4 + col]).sum();
            assert!(sum.abs() < 1e-11, "row {row} of Q sums to {sum}");
            for col in 0..4 {
                assert!((q[row * 4 + col] - q[col * 4 + row]).abs() < 1e-12);
            }
        }
    }
}

#[test]
fn flat_hinges_carry_no_bending_energy_wherever_they_are() {
    for hinge in [kite(), skewed_flat()] {
        let c = IsometricBendingConstraint::new([0, 1, 2, 3], &hinge, 0.0);
        assert!(c.bending_energy(&hinge).abs() < 1e-12);
        // Translated by (5, 5, 5) and moved rigidly: still no energy.
        let shifted = rigidly_moved(&hinge, 0.0, Vec3::new(5.0, 5.0, 5.0));
        assert!(
            c.bending_energy(&shifted).abs() < 1e-10,
            "a translated flat hinge must stay at zero energy: {}",
            c.bending_energy(&shifted)
        );
        let moved = rigidly_moved(&hinge, 1.234, Vec3::new(-4.0, 1.5, 0.25));
        assert!(c.bending_energy(&moved).abs() < 1e-10);
    }
}

#[test]
fn bending_energy_is_invariant_under_rigid_motion() {
    let c = IsometricBendingConstraint::new([0, 1, 2, 3], &kite(), 0.0);
    let folded = folded_kite(0.6);
    let reference = c.bending_energy(&folded);
    assert!(
        reference > 1e-2,
        "a folded hinge carries energy: {reference}"
    );
    for (angle, shift) in [
        (0.0, Vec3::new(5.0, 5.0, 5.0)),
        (1.234, Vec3::new(-4.0, 1.5, 0.25)),
    ] {
        let moved = rigidly_moved(&folded, angle, shift);
        assert!(
            (c.bending_energy(&moved) - reference).abs() < 1e-9 * reference,
            "rigid motion changed the energy: {} vs {reference}",
            c.bending_energy(&moved)
        );
    }
}

#[test]
fn folded_kite_energy_has_the_closed_form() {
    // K = [1, 1, -1, -1] and 3 / (A0 + A1) = 3 stay valid while the wings turn
    // rigidly about the edge, and K . p = p0 + p1 - p2 - p3 = (0, 0, -2 sin(fold/2)),
    // so E = 1/2 * 3 * 4 sin^2(fold/2) = 6 sin^2(fold/2).
    let c = IsometricBendingConstraint::new([0, 1, 2, 3], &kite(), 0.0);
    for fold_degrees in [15.0_f64, 30.0, 60.0, -30.0] {
        let fold = fold_degrees.to_radians();
        let expected = 6.0 * (0.5 * fold).sin().powi(2);
        let energy = c.bending_energy(&folded_kite(fold));
        assert!(
            (energy - expected).abs() < 1e-12,
            "{fold_degrees} deg: E = {energy}, expected {expected}"
        );
    }
}

#[test]
fn flat_hinge_is_a_fixed_point_of_the_projection() {
    for hinge in [kite(), skewed_flat()] {
        let mut particles = hinge.clone();
        let mut c = IsometricBendingConstraint::new([0, 1, 2, 3], &hinge, 1e-4);
        for _ in 0..3 {
            c.project(&mut particles, DT);
        }
        assert!(
            max_deviation(&particles, &hinge) < 1e-12,
            "a flat hinge must stay put"
        );
    }
}

/// A planar hinge with generic coordinates: the wings lie on opposite sides of
/// the shared edge, the hinge is about `size` long, turned into a random
/// orientation and moved by up to `offset` in every direction.
fn random_flat_hinge(rng: &mut Lcg, size: Real, offset: Real) -> Vec<SoftParticle> {
    let edge_end = Vec3::new(1.0 + rng.next_unit(), 0.3 * rng.next_unit(), 0.0);
    let edge_dir = edge_end / edge_end.norm();
    let side = Vec3::new(-edge_dir.y, edge_dir.x, 0.0);
    let wing_a = edge_end * rng.next_unit() + side * (0.2 + rng.next_unit());
    let wing_b = edge_end * rng.next_unit() - side * (0.2 + rng.next_unit());
    let axis_1 = Vec3::new(1.0, 2.0, 2.0) / 3.0;
    let axis_2 = Vec3::new(2.0, -1.0, 2.0) / 3.0;
    let (angle_1, angle_2) = (6.0 * rng.next_unit(), 6.0 * rng.next_unit());
    let shift = Vec3::new(rng.next_unit(), rng.next_unit(), rng.next_unit()) * offset;
    [Vec3::zeros(), edge_end, wing_a, wing_b]
        .iter()
        .map(|&p| {
            let turned = rotate_about(axis_2, angle_2, rotate_about(axis_1, angle_1, p));
            SoftParticle::new(turned * size + shift, 1.0)
        })
        .collect()
}

#[test]
fn flat_hinges_stay_put_whatever_their_size_and_position() {
    // The energy and its gradient are both zero up to round-off for a flat
    // hinge.  Evaluated independently (from the absolute positions and the
    // full matrix Q), the two roundings used to be divided by each other and
    // moved a flat hinge of a few centimetres, one metre from the origin, by many
    // times its own size.
    let mut rng = Lcg(2024);
    for size in [1.0, 1e-2, 1e-3, 1e-4] {
        for offset in [0.0, 1.0, 100.0] {
            for _ in 0..100 {
                let hinge = random_flat_hinge(&mut rng, size, offset);
                let mut c = IsometricBendingConstraint::new([0, 1, 2, 3], &hinge, 0.0);
                let mut particles = hinge.clone();
                c.project(&mut particles, DT);
                let moved = max_deviation(&particles, &hinge);
                assert!(
                    moved < 1e-9 * size,
                    "size {size}, offset {offset}: a flat hinge moved by {moved}"
                );
                assert!(
                    c.bending_energy(&hinge) < 1e-12 * size.powi(2) * c.scale,
                    "size {size}, offset {offset}: flat energy {}",
                    c.bending_energy(&hinge)
                );
            }
        }
    }
}

#[test]
fn projection_quarters_the_energy_whatever_the_masses() {
    // E = 1/2 s |y|^2 with y = K . x.  The Newton step of E moves the particles
    // by -w_i K_i y / (2 sum w K^2), i.e. it halves y and quarters E exactly.
    let masses = [1.0, 2.0, 0.5, 3.0];
    let flat = [
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.5, 1.0, 0.0],
        [0.5, -1.0, 0.0],
    ];
    let folded = [
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.5, 1.0, 0.4],
        [0.5, -1.0, 0.1],
    ];
    let rest = particles_with_masses(&flat, &masses);
    let mut particles = particles_with_masses(&folded, &masses);
    let mut c = IsometricBendingConstraint::new([0, 1, 2, 3], &rest, 0.0);
    for _ in 0..4 {
        let before = particles.clone();
        let energy_before = c.bending_energy(&particles);
        c.reset_lambda();
        c.project(&mut particles, DT);
        let energy_after = c.bending_energy(&particles);
        assert!(
            (energy_after - 0.25 * energy_before).abs() < 1e-9 * energy_before,
            "E {energy_before} -> {energy_after}"
        );
        let (linear, angular) = correction_momenta(&before, &particles);
        assert!(linear.norm() < 1e-13, "linear momentum {linear:?}");
        assert!(angular.norm() < 1e-12, "angular momentum {angular:?}");
    }
}

#[test]
fn projection_moves_the_hinge_edge_too_and_flattens_the_hinge() {
    let rest = kite();
    let mut particles = folded_kite(0.8);
    let start = particles.clone();
    let mut c = IsometricBendingConstraint::new([0, 1, 2, 3], &rest, 0.0);
    let initial = c.bending_energy(&particles);
    c.project(&mut particles, DT);
    for k in 0..4 {
        assert!(
            (particles[k].position - start[k].position).norm() > 1e-6,
            "particle {k} must take part in the correction"
        );
    }
    for _ in 0..40 {
        c.reset_lambda();
        c.project(&mut particles, DT);
    }
    assert!(
        c.bending_energy(&particles) < 1e-12 * initial,
        "the hinge must flatten: {initial} -> {}",
        c.bending_energy(&particles)
    );
}

#[test]
fn compliance_limits_the_bending_correction() {
    let rest = kite();
    let folded = folded_kite(0.8);
    let mut rigid = folded.clone();
    let mut soft = folded.clone();
    IsometricBendingConstraint::new([0, 1, 2, 3], &rest, 0.0).project(&mut rigid, DT);
    IsometricBendingConstraint::new([0, 1, 2, 3], &rest, 1e-2).project(&mut soft, DT);
    let reference = IsometricBendingConstraint::new([0, 1, 2, 3], &rest, 0.0);
    let initial = reference.bending_energy(&folded);
    let e_rigid = reference.bending_energy(&rigid);
    let e_soft = reference.bending_energy(&soft);
    assert!(
        e_rigid < e_soft && e_soft < initial,
        "{e_rigid} {e_soft} {initial}"
    );
}

#[test]
fn degenerate_hinge_has_no_bending_resistance() {
    // Both wings on the shared-edge line: the triangles have no area.
    let collapsed = unit_masses(&[
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.5, 0.0, 0.0],
        [0.25, 0.0, 0.0],
    ]);
    let mut c = IsometricBendingConstraint::new([0, 1, 2, 3], &collapsed, 1e-4);
    assert!(c.q_matrix().iter().all(|&q| q == 0.0));
    let mut particles = folded_kite(0.5);
    let before = particles.clone();
    c.project(&mut particles, DT);
    assert!(max_deviation(&particles, &before) < 1e-15);
    assert_eq!(c.bending_energy(&particles), 0.0);
}

// ---------------------------------------------------------------------------
// CSTStrainConstraint
// ---------------------------------------------------------------------------

/// Equilateral triangle in the xy-plane: `Dm⁻¹` is not symmetric.
fn equilateral() -> Vec<SoftParticle> {
    unit_masses(&[
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.5, 0.75_f64.sqrt(), 0.0],
    ])
}

/// Scalene triangle in a general 3D orientation.
fn scalene() -> Vec<SoftParticle> {
    unit_masses(&[[0.2, -0.1, 0.3], [1.4, 0.3, 0.1], [0.6, 1.2, 0.9]])
}

/// The triangle scaled by `factor` about its centroid.
fn scaled_about_centroid(particles: &[SoftParticle], factor: Real) -> Vec<SoftParticle> {
    let centroid = particles
        .iter()
        .fold(Vec3::zeros(), |acc, p| acc + p.position)
        / particles.len() as Real;
    particles
        .iter()
        .map(|p| {
            SoftParticle::new(
                centroid + (p.position - centroid) * factor,
                1.0 / p.inverse_mass,
            )
        })
        .collect()
}

#[test]
fn rest_triangles_have_unit_strain_invariants() {
    for triangle in [equilateral(), scalene()] {
        let c = CSTStrainConstraint::new([0, 1, 2], &triangle, 0.0);
        let (i1, area_ratio) = c.strain_invariants(&triangle);
        assert!((i1 - 2.0).abs() < 1e-12, "I1 = {i1} must be 2 at rest");
        assert!((area_ratio - 1.0).abs() < 1e-12, "J = {area_ratio}");
    }
}

#[test]
fn rigid_motion_does_not_change_the_strain() {
    for triangle in [equilateral(), scalene()] {
        let c = CSTStrainConstraint::new([0, 1, 2], &triangle, 0.0);
        let moved = rigidly_moved(&triangle, 0.77, Vec3::new(2.0, -3.0, 1.0));
        let (i1, area_ratio) = c.strain_invariants(&moved);
        assert!((i1 - 2.0).abs() < 1e-12, "I1 = {i1}");
        assert!((area_ratio - 1.0).abs() < 1e-12, "J = {area_ratio}");
    }
}

#[test]
fn rest_and_rigidly_moved_triangles_are_fixed_points_of_the_projection() {
    for triangle in [equilateral(), scalene()] {
        let mut c = CSTStrainConstraint::new([0, 1, 2], &triangle, 1e-4);
        for start in [
            triangle.clone(),
            rigidly_moved(&triangle, 1.9, Vec3::new(-1.0, 4.0, 0.5)),
        ] {
            let mut particles = start.clone();
            for _ in 0..3 {
                c.project(&mut particles, DT);
            }
            assert!(
                max_deviation(&particles, &start) < 1e-12,
                "undeformed cloth must not be deformed by the strain constraint"
            );
        }
    }
}

#[test]
fn uniform_stretch_scales_the_trace_and_the_area() {
    let c = CSTStrainConstraint::new([0, 1, 2], &scalene(), 0.0);
    let stretched = scaled_about_centroid(&scalene(), 1.3);
    let (i1, area_ratio) = c.strain_invariants(&stretched);
    assert!((i1 - 2.0 * 1.69).abs() < 1e-12, "I1 = {i1}");
    assert!((area_ratio - 1.69).abs() < 1e-12, "J = {area_ratio}");
}

#[test]
fn strain_gradient_matches_finite_differences() {
    // A deformed copy of the scalene triangle, so that every entry of F and Dm⁻¹
    // matters.
    let rest = scalene();
    let c = CSTStrainConstraint::new([0, 1, 2], &rest, 0.0);
    let deformed = unit_masses(&[[0.3, -0.2, 0.35], [1.6, 0.1, 0.0], [0.5, 1.5, 1.1]]);
    let analytic = c.strain_gradient(&deformed);
    let numeric = numeric_gradient(&deformed, |p| c.strain_invariants(p).0);
    for (k, (a, n)) in analytic.iter().zip(numeric.iter()).enumerate() {
        assert!((a - n).norm() < 1e-6, "vertex {k}: {a:?} vs {n:?}");
    }
    let sum = analytic.iter().fold(Vec3::zeros(), |acc, g| acc + g);
    assert!(sum.norm() < 1e-12, "gradients must add up to zero: {sum:?}");
}

#[test]
fn projection_restores_a_stretched_triangle_and_conserves_momentum() {
    let masses = [1.0, 2.0, 0.5];
    let points = [[0.2, -0.1, 0.3], [1.4, 0.3, 0.1], [0.6, 1.2, 0.9]];
    let rest = particles_with_masses(&points, &masses);
    let mut particles = scaled_about_centroid(&rest, 1.3);
    let mut c = CSTStrainConstraint::new([0, 1, 2], &rest, 0.0);
    let mut violation = (c.strain_invariants(&particles).0 - 2.0).abs();
    assert!(violation > 1.0, "scenario must be stretched: {violation}");
    for _ in 0..20 {
        let before = particles.clone();
        c.reset_lambda();
        c.project(&mut particles, DT);
        let now = (c.strain_invariants(&particles).0 - 2.0).abs();
        assert!(
            now < violation || now < 1e-12,
            "|I1 - 2| must shrink: {violation} -> {now}"
        );
        violation = now;
        let (linear, angular) = correction_momenta(&before, &particles);
        assert!(linear.norm() < 1e-13, "linear momentum {linear:?}");
        assert!(angular.norm() < 1e-12, "angular momentum {angular:?}");
    }
    assert!(violation < 1e-8, "I1 must return to 2: {violation}");
}

// ---------------------------------------------------------------------------
// NeoHookeanConstraint
// ---------------------------------------------------------------------------

/// A tetrahedron that is not aligned with the axes and whose `Dm⁻¹` is not
/// symmetric.
fn general_tet() -> Vec<SoftParticle> {
    unit_masses(&[
        [0.3, 0.1, -0.2],
        [1.4, 0.2, 0.1],
        [0.2, 1.1, 0.3],
        [-0.1, 0.2, 1.3],
    ])
}

fn signed_volume(particles: &[SoftParticle]) -> Real {
    let e1 = particles[1].position - particles[0].position;
    let e2 = particles[2].position - particles[0].position;
    let e3 = particles[3].position - particles[0].position;
    e1.dot(&e2.cross(&e3)) / 6.0
}

fn centroid(particles: &[SoftParticle]) -> Vec3 {
    particles
        .iter()
        .fold(Vec3::zeros(), |acc, p| acc + p.position)
        / particles.len() as Real
}

/// Deform `rest` by `x -> map * (x - c) + c`, which gives `F = map` and keeps
/// the centroid `c` fixed.
fn deform(rest: &[SoftParticle], map: &Mat3) -> Vec<SoftParticle> {
    let c = centroid(rest);
    rest.iter()
        .map(|p| SoftParticle::new(map * (p.position - c) + c, 1.0 / p.inverse_mass))
        .collect()
}

fn lame(youngs: Real, poisson: Real) -> (Real, Real) {
    let mu = youngs / (2.0 * (1.0 + poisson));
    let lambda = youngs * poisson / ((1.0 + poisson) * (1.0 - 2.0 * poisson));
    (mu, lambda)
}

/// `max |F^T F - I|` of the deformation gradient: zero for a pure rotation.
fn deviation_from_rotation(c: &NeoHookeanConstraint, particles: &[SoftParticle]) -> Real {
    let f = Mat3::from_column_slice(&c.current_element().deformation_gradient(particles));
    (f.transpose() * f - Mat3::identity())
        .iter()
        .fold(0.0, |worst: Real, entry| worst.max(entry.abs()))
}

fn relax(c: &mut NeoHookeanConstraint, particles: &mut [SoftParticle], iterations: usize) {
    for _ in 0..iterations {
        c.reset_lambda();
        c.project(particles, DT);
    }
}

#[test]
fn lame_parameters_follow_young_modulus_and_poisson_ratio() {
    let rest = general_tet();
    let c = NeoHookeanConstraint::new([0, 1, 2, 3], &rest, 2.0e5, 0.3, 0.0);
    let (mu, lambda) = lame(2.0e5, 0.3);
    assert!((c.lame_mu - mu).abs() < 1e-6 * mu);
    assert!((c.lame_lambda - lambda).abs() < 1e-6 * lambda);
    assert!((c.rest_volume - signed_volume(&rest).abs()).abs() < 1e-12);
    let k = lambda + mu;
    assert!((c.hydrostatic_target() - (1.0 + mu / k)).abs() < 1e-12);
}

#[test]
fn constraint_values_at_rest_for_a_general_tet() {
    let rest = general_tet();
    let c = NeoHookeanConstraint::new([0, 1, 2, 3], &rest, 1.0e5, 0.3, 0.0);
    let k = c.lame_lambda + c.lame_mu;
    assert!(c.volume_strain(&rest).abs() < 1e-12);
    assert!((c.deviatoric_constraint(&rest) - 3.0_f64.sqrt()).abs() < 1e-12);
    assert!((c.hydrostatic_constraint(&rest) + c.lame_mu / k).abs() < 1e-12);
    // The two forces balance: the rest state is stress free and carries no energy.
    let element = c.current_element();
    for entry in element.first_piola_stress(&rest) {
        assert!(entry.abs() < 1e-8 * c.lame_mu, "rest stress {entry}");
    }
    assert!(element.elastic_energy(&rest).abs() < 1e-9);
}

#[test]
fn rest_state_is_a_fixed_point_for_every_poisson_ratio_and_time_step() {
    // nu = 0 (lambda = 0) is the case in which a hydrostatic target 1 + mu / lambda
    // would be singular.
    for rest in [
        general_tet(),
        rigidly_moved(&general_tet(), 0.6, Vec3::new(1.0, 2.0, 3.0)),
    ] {
        for youngs in [1.0e3, 1.0e6, 1.0e9] {
            for poisson in [-0.5, 0.0, 0.3, 0.45, 0.5] {
                for dt in [DT / 10.0, DT, 10.0 * DT] {
                    let mut c =
                        NeoHookeanConstraint::new([0, 1, 2, 3], &rest, youngs, poisson, 0.0);
                    let mut particles = rest.clone();
                    c.project(&mut particles, dt);
                    assert!(
                        max_deviation(&particles, &rest) < 1e-7,
                        "E = {youngs}, nu = {poisson}, dt = {dt}: the rest state moved by {}",
                        max_deviation(&particles, &rest)
                    );
                }
            }
        }
    }
}

#[test]
fn small_strain_stress_is_hookes_law_with_the_lame_parameters() {
    let rest = general_tet();
    let c = NeoHookeanConstraint::new([0, 1, 2, 3], &rest, 2.0e5, 0.3, 0.0);
    let element = c.current_element();
    let (mu, lambda) = (c.lame_mu, c.lame_lambda);
    let eps = 1e-6;
    // Uniform dilation F = (1 + eps) I: sigma = (3 lambda + 2 mu) eps I.
    let dilated = deform(&rest, &(Mat3::identity() * (1.0 + eps)));
    let p = Mat3::from_column_slice(&element.first_piola_stress(&dilated));
    let expected = (3.0 * lambda + 2.0 * mu) * eps;
    for i in 0..3 {
        assert!(
            (p[(i, i)] - expected).abs() < 1e-3 * expected,
            "P[{i},{i}] = {} vs {expected}",
            p[(i, i)]
        );
    }
    // Simple shear F = I + gamma e_x e_y^T: sigma_xy = mu gamma.
    let mut shear = Mat3::identity();
    shear[(0, 1)] = eps;
    let sheared = deform(&rest, &shear);
    let p = Mat3::from_column_slice(&element.first_piola_stress(&sheared));
    assert!(
        (p[(0, 1)] - mu * eps).abs() < 1e-3 * mu * eps,
        "P_xy = {}",
        p[(0, 1)]
    );
    assert!(
        (p[(1, 0)] - mu * eps).abs() < 1e-3 * mu * eps,
        "P_yx = {}",
        p[(1, 0)]
    );
    assert!(p[(2, 2)].abs() < 1e-3 * mu * eps, "P_zz = {}", p[(2, 2)]);
}

#[test]
fn elastic_forces_equal_the_energy_gradient_with_the_inverse_transpose() {
    // dE/dx_(k+1) = V0 * (P Dm^-T) column k, dE/dx_0 = minus the sum: the rows of
    // Dm^-1, not its columns, carry the vertex gradients.
    let rest = general_tet();
    let c = NeoHookeanConstraint::new([0, 1, 2, 3], &rest, 1.0e5, 0.3, 0.0);
    let element = c.current_element();
    let map = Mat3::new(1.2, 0.3, -0.1, 0.05, 0.8, 0.2, -0.15, 0.1, 1.1);
    let deformed = deform(&rest, &map);
    let p = Mat3::from_column_slice(&element.first_piola_stress(&deformed));
    let dm = Mat3::from_columns(&[
        rest[1].position - rest[0].position,
        rest[2].position - rest[0].position,
        rest[3].position - rest[0].position,
    ]);
    let dm_inv = dm.try_inverse().expect("the rest tetrahedron has volume");
    assert!(
        (dm_inv - dm_inv.transpose()).norm() > 0.1,
        "Dm^-1 must not be symmetric for this check to discriminate"
    );
    let forces = (p * dm_inv.transpose()) * element.rest_volume;
    let numeric = numeric_gradient(&deformed, |ps| element.elastic_energy(ps));
    let mut vertex_0 = Vec3::zeros();
    for k in 0..3 {
        let analytic = forces.column(k).into_owned();
        vertex_0 -= analytic;
        assert!(
            (analytic - numeric[k + 1]).norm() < 1e-5 * analytic.norm().max(1.0),
            "vertex {}: {analytic:?} vs {:?}",
            k + 1,
            numeric[k + 1]
        );
    }
    assert!((vertex_0 - numeric[0]).norm() < 1e-5 * vertex_0.norm().max(1.0));
}

#[test]
fn sheared_non_axis_aligned_tet_relaxes_to_the_rest_shape() {
    // A stiff element would need sub-steps (see `crate::constraint::NeoHookeanConstraint`);
    // E = 1e4 converges with one projection per iteration.
    let rest = general_tet();
    let mut c = NeoHookeanConstraint::new([0, 1, 2, 3], &rest, 1.0e4, 0.3, 0.0);
    let map = Mat3::new(1.25, 0.35, -0.1, 0.05, 0.8, 0.25, -0.15, 0.1, 1.1);
    let mut particles = deform(&rest, &map);
    let centroid_before = centroid(&particles);
    assert!(
        deviation_from_rotation(&c, &particles) > 0.3,
        "scenario must be strained"
    );
    relax(&mut c, &mut particles, 100);
    assert!(
        deviation_from_rotation(&c, &particles) < 1e-6,
        "F must become a rotation, deviation {}",
        deviation_from_rotation(&c, &particles)
    );
    assert!(
        (signed_volume(&particles) - signed_volume(&rest)).abs()
            < 1e-6 * signed_volume(&rest).abs(),
        "the rest volume must be restored"
    );
    assert!(
        (centroid(&particles) - centroid_before).norm() < 1e-10,
        "internal forces must not move the centroid"
    );
}

#[test]
fn compressed_tet_recovers_its_volume() {
    let rest = general_tet();
    let mut c = NeoHookeanConstraint::new([0, 1, 2, 3], &rest, 1.0e4, 0.3, 0.0);
    let map = Mat3::new(1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.4);
    let mut particles = deform(&rest, &map);
    let before = c.volume_strain(&particles).abs();
    assert!(before > 0.5, "scenario must be compressed: {before}");
    relax(&mut c, &mut particles, 100);
    assert!(c.volume_strain(&particles).abs() < 1e-6);
}

#[test]
fn stiffness_and_poisson_ratio_change_the_projection() {
    let rest = general_tet();
    let map = Mat3::new(1.2, 0.3, -0.1, 0.05, 0.8, 0.2, -0.15, 0.1, 1.1);
    let project_with = |youngs: Real, poisson: Real| {
        let mut c = NeoHookeanConstraint::new([0, 1, 2, 3], &rest, youngs, poisson, 0.0);
        let mut particles = deform(&rest, &map);
        c.project(&mut particles, DT);
        particles
    };
    let soft = project_with(1.0e3, 0.3);
    let stiff = project_with(1.0e7, 0.3);
    assert!(
        max_deviation(&soft, &stiff) > 1e-4,
        "Young's modulus is ignored"
    );
    let compressible = project_with(1.0e5, 0.05);
    let nearly_incompressible = project_with(1.0e5, 0.49);
    assert!(
        max_deviation(&compressible, &nearly_incompressible) > 1e-4,
        "Poisson's ratio is ignored"
    );
}

#[test]
fn public_lame_parameters_are_read_by_the_projection() {
    // A volume-preserving shear: with mu = 0 only the hydrostatic constraint is
    // left and it is satisfied (det F = 1 = gamma), so nothing moves; with a shear
    // modulus the deviatoric constraint pulls the tet back.
    let rest = general_tet();
    let shear = Mat3::new(1.0, 0.4, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0);
    let sheared = deform(&rest, &shear);
    let mut c = NeoHookeanConstraint::new([0, 1, 2, 3], &rest, 1.0e6, 0.3, 0.0);
    let mut moved = sheared.clone();
    c.project(&mut moved, DT);
    assert!(
        max_deviation(&moved, &sheared) > 1e-4,
        "shear must be resisted"
    );

    c.reset_lambda();
    c.lame_mu = 0.0;
    let mut frozen = sheared.clone();
    c.project(&mut frozen, DT);
    assert!(
        max_deviation(&frozen, &sheared) < 1e-12,
        "without a shear modulus a volume-preserving shear costs nothing, moved by {}",
        max_deviation(&frozen, &sheared)
    );
}

#[test]
fn compliance_softens_the_volume_response() {
    // Pure volume material (no shear modulus, so only the hydrostatic constraint
    // acts): one projection of a 20 % compressed tet.  The extra compliance is
    // added in series to the physical compliance 1 / (K V0).
    let rest = general_tet();
    let rest_volume = signed_volume(&rest);
    let map = Mat3::new(0.8, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0);
    let volume_error_after_one_step = |compliance: Real| -> Real {
        let mut c = NeoHookeanConstraint::new([0, 1, 2, 3], &rest, 1.0e6, 0.3, compliance);
        c.lame_mu = 0.0;
        let mut particles = deform(&rest, &map);
        c.project(&mut particles, DT / 10.0);
        (signed_volume(&particles) / rest_volume - 1.0).abs()
    };
    let stiff = volume_error_after_one_step(0.0);
    let medium = volume_error_after_one_step(1e-5);
    let soft = volume_error_after_one_step(1e-3);
    assert!(
        stiff < medium && medium < soft && soft < 0.2 - 1e-6,
        "more compliance must mean a smaller correction: {stiff} < {medium} < {soft} < 0.2"
    );
}

#[test]
fn incompressible_poisson_ratio_keeps_the_lame_parameters_finite() {
    let rest = general_tet();
    for poisson in [0.5, 0.75, -1.0, -3.0] {
        let mut c = NeoHookeanConstraint::new([0, 1, 2, 3], &rest, 1.0e5, poisson, 0.0);
        assert!(c.lame_lambda.is_finite(), "lambda = {}", c.lame_lambda);
        assert!(
            c.lame_mu.is_finite() && c.lame_mu > 0.0,
            "mu = {}",
            c.lame_mu
        );
        let mut particles = deform(
            &rest,
            &Mat3::new(1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.6),
        );
        c.project(&mut particles, DT);
        assert!(
            particles
                .iter()
                .all(|p| p.position.iter().all(|x| x.is_finite())),
            "nu = {poisson}: non-finite positions"
        );
    }
}
