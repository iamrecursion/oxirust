//! Constraint projections of the XPBD solvers.
//!
//! [`XPBDGpuSolver`](super::types::XPBDGpuSolver) and
//! [`XPBDGpuKernel`](super::types::XPBDGpuKernel) share these projections.
//!
//! Distance and volume constraints are *compliant* constraints solved in the
//! XPBD form of Macklin, Müller and Chentanez (2016).  A constraint `C(x) = 0`
//! with compliance `α` (inverse stiffness) and Lagrange multiplier `λ` is
//! projected with
//!
//! ```text
//! α̃  = α / Δt²
//! Δλ = (−C(x) − α̃ λ) / (Σₖ wₖ |∇ₖC|² + α̃)
//! xₖ += wₖ Δλ ∇ₖC,        λ += Δλ
//! ```
//!
//! where `wₖ` are the inverse masses.  `λ` starts at zero for every time step and
//! is *accumulated* over the solver iterations, so that the result approaches the
//! solution of the implicit (compliant) problem instead of a stiffer and stiffer
//! one as the iteration count grows.  Zero compliance recovers the classic
//! position-based projection.
//!
//! Shape matching (Müller et al. 2005) is a stiffness-blended goal-position
//! constraint and has no multiplier.

use super::corotational::{Mat3, polar_rotation};
use super::functions::{add3, cross3, dot3, length3, scale3, sub3};
use super::types::{ShapeMatchConstraint, VolumeConstraint, XpbdDistConstraint};

/// Lengths and denominators below this are treated as zero.
const EPSILON: f64 = 1.0e-15;

/// Weight of a pinned particle in a shape-matching fit, relative to the total
/// mass of the movable particles of that shape.  A pinned particle has infinite
/// mass; this large finite weight lets it dominate the fitted centre without
/// making the fit singular.
const PINNED_WEIGHT_FACTOR: f64 = 1.0e6;

/// `1 / Δt²` for a usable time step, `None` for a time step that is not a
/// positive finite number (or whose square is not representable).
pub(super) fn inverse_dt_squared(dt: f64) -> Option<f64> {
    if !(dt.is_finite() && dt > 0.0) {
        return None;
    }
    let inverse = 1.0 / (dt * dt);
    inverse.is_finite().then_some(inverse)
}

/// Project one distance constraint `C = |xᵢ − xⱼ| − rest_len`.
///
/// `inv_dt_sq` is `1 / Δt²` (see [`inverse_dt_squared`]) and `lambda` the
/// accumulated multiplier of this constraint for the current time step.
/// Constraints that reference a particle outside `positions` or `inv_masses`,
/// whose particles coincide, or whose particles are all fixed are left alone.
pub(super) fn project_distance(
    positions: &mut [[f64; 3]],
    inv_masses: &[f64],
    constraint: &XpbdDistConstraint,
    inv_dt_sq: f64,
    lambda: &mut f64,
) {
    let (i, j) = (constraint.i, constraint.j);
    let (Some(&pi), Some(&pj)) = (positions.get(i), positions.get(j)) else {
        return;
    };
    let (Some(&wi), Some(&wj)) = (inv_masses.get(i), inv_masses.get(j)) else {
        return;
    };
    let offset = sub3(pi, pj);
    let dist = length3(offset);
    if dist < EPSILON {
        return;
    }
    let alpha_tilde = constraint.compliance.max(0.0) * inv_dt_sq;
    let denominator = wi + wj + alpha_tilde;
    if denominator < EPSILON {
        return;
    }
    let value = dist - constraint.rest_len;
    let delta_lambda = (-value - alpha_tilde * *lambda) / denominator;
    *lambda += delta_lambda;
    let gradient = scale3(offset, 1.0 / dist);
    positions[i] = add3(pi, scale3(gradient, delta_lambda * wi));
    positions[j] = sub3(pj, scale3(gradient, delta_lambda * wj));
}

/// Signed volume `det[b − a | c − a | d − a] / 6` of the tetrahedron `a b c d`:
/// positive when `a b c d` is right-handed.
pub(super) fn signed_volume(a: [f64; 3], b: [f64; 3], c: [f64; 3], d: [f64; 3]) -> f64 {
    dot3(sub3(b, a), cross3(sub3(c, a), sub3(d, a))) / 6.0
}

/// Gradients `[∇ₐV, ∇ᵦV, ∇꜀V, ∇ᵈV]` of the signed volume of the tetrahedron
/// `a b c d` with respect to its four vertices.
///
/// `∇ᵦV = (c − a) × (d − a) / 6`, `∇꜀V = (d − a) × (b − a) / 6`,
/// `∇ᵈV = (b − a) × (c − a) / 6` and `∇ₐV = −(∇ᵦV + ∇꜀V + ∇ᵈV)`.
pub(super) fn volume_gradients(
    a: [f64; 3],
    b: [f64; 3],
    c: [f64; 3],
    d: [f64; 3],
) -> [[f64; 3]; 4] {
    let ab = sub3(b, a);
    let ac = sub3(c, a);
    let ad = sub3(d, a);
    let grad_b = scale3(cross3(ac, ad), 1.0 / 6.0);
    let grad_c = scale3(cross3(ad, ab), 1.0 / 6.0);
    let grad_d = scale3(cross3(ab, ac), 1.0 / 6.0);
    let grad_a = scale3(add3(add3(grad_b, grad_c), grad_d), -1.0);
    [grad_a, grad_b, grad_c, grad_d]
}

/// Project one volume constraint `C = V − V₀` with the *signed* tetrahedron
/// volume `V`, so that an inverted tetrahedron (`V < 0`) is turned inside out
/// again.
///
/// Constraints that reference a particle outside `positions` or `inv_masses`,
/// list a particle twice, or have no movable particle are left alone.
pub(super) fn project_volume(
    positions: &mut [[f64; 3]],
    inv_masses: &[f64],
    constraint: &VolumeConstraint,
    inv_dt_sq: f64,
    lambda: &mut f64,
) {
    let indices = constraint.indices;
    for (k, &index) in indices.iter().enumerate() {
        if index >= positions.len() || index >= inv_masses.len() || indices[..k].contains(&index) {
            return;
        }
    }
    let [pa, pb, pc, pd] = indices.map(|index| positions[index]);
    let weights = indices.map(|index| inv_masses[index]);
    let gradients = volume_gradients(pa, pb, pc, pd);
    let weighted_norm_sq: f64 = gradients
        .iter()
        .zip(&weights)
        .map(|(gradient, &weight)| weight * dot3(*gradient, *gradient))
        .sum();
    let alpha_tilde = constraint.compliance.max(0.0) * inv_dt_sq;
    let denominator = weighted_norm_sq + alpha_tilde;
    if denominator < EPSILON {
        return;
    }
    let value = signed_volume(pa, pb, pc, pd) - constraint.rest_volume;
    let delta_lambda = (-value - alpha_tilde * *lambda) / denominator;
    *lambda += delta_lambda;
    for ((&index, gradient), &weight) in indices.iter().zip(&gradients).zip(&weights) {
        positions[index] = add3(positions[index], scale3(*gradient, weight * delta_lambda));
    }
}

/// Matrix–vector product `m · v`.
fn mat_vec(m: &Mat3, v: [f64; 3]) -> [f64; 3] {
    [dot3(m[0], v), dot3(m[1], v), dot3(m[2], v)]
}

/// A particle of a shape-matching constraint: its index, rest position and
/// weight in the fit.
struct ShapeMember {
    index: usize,
    rest: [f64; 3],
    weight: f64,
    movable: bool,
}

/// Project one shape-matching constraint (Müller et al. 2005).
///
/// With the weights `mₖ` (the particle masses, see below), the weighted centres
/// `c` of the current and `c₀` of the rest positions, and
/// `A = Σ mₖ (xₖ − c)(qₖ − c₀)ᵀ`, the best rigid fit is the rotation
/// `R` of the polar decomposition `A = R S`, and the goal position of particle
/// `k` is `gₖ = c + R (qₖ − c₀)`.  Every movable particle moves the fraction
/// `stiffness` of the way towards its goal, so a rigid motion of the whole shape
/// (rotation included) is left untouched.
///
/// The mass of a movable particle is `1 / inverse_mass`.  A pinned particle
/// (inverse mass zero) never moves but belongs to the shape; its weight is
/// `PINNED_WEIGHT_FACTOR` times the total movable mass, which makes the fit
/// follow the pin.  Entries of `constraint.indices` that are outside `positions`
/// or `inv_masses`, or that have no rest position, are ignored.
pub(super) fn project_shape(
    positions: &mut [[f64; 3]],
    inv_masses: &[f64],
    constraint: &ShapeMatchConstraint,
) {
    if !constraint.stiffness.is_finite() {
        return;
    }
    let mut members: Vec<ShapeMember> = constraint
        .indices
        .iter()
        .zip(&constraint.rest_positions)
        .filter_map(|(&index, &rest)| {
            if index >= positions.len() {
                return None;
            }
            let inverse_mass = *inv_masses.get(index)?;
            let movable = inverse_mass.is_finite() && inverse_mass > 0.0;
            Some(ShapeMember {
                index,
                rest,
                weight: if movable { 1.0 / inverse_mass } else { 0.0 },
                movable,
            })
        })
        .collect();
    let movable_mass: f64 = members
        .iter()
        .filter(|member| member.movable)
        .map(|member| member.weight)
        .sum();
    if !(movable_mass.is_finite() && movable_mass > 0.0) {
        return;
    }
    for member in members.iter_mut().filter(|member| !member.movable) {
        member.weight = PINNED_WEIGHT_FACTOR * movable_mass;
    }
    let total_weight: f64 = members.iter().map(|member| member.weight).sum();
    let mut centre = [0.0; 3];
    let mut rest_centre = [0.0; 3];
    for member in &members {
        centre = add3(centre, scale3(positions[member.index], member.weight));
        rest_centre = add3(rest_centre, scale3(member.rest, member.weight));
    }
    centre = scale3(centre, 1.0 / total_weight);
    rest_centre = scale3(rest_centre, 1.0 / total_weight);

    let mut covariance: Mat3 = [[0.0; 3]; 3];
    for member in &members {
        let current = sub3(positions[member.index], centre);
        let rest = sub3(member.rest, rest_centre);
        for (row, &current_component) in covariance.iter_mut().zip(&current) {
            for (entry, &rest_component) in row.iter_mut().zip(&rest) {
                *entry += member.weight * current_component * rest_component;
            }
        }
    }
    let rotation = polar_rotation(&covariance);

    for member in members.iter().filter(|member| member.movable) {
        let goal = add3(centre, mat_vec(&rotation, sub3(member.rest, rest_centre)));
        let position = positions[member.index];
        positions[member.index] =
            add3(position, scale3(sub3(goal, position), constraint.stiffness));
    }
}

#[cfg(test)]
mod tests {
    use super::super::functions::normalize3;
    use super::super::types::{DeformableGpuBuffer, XPBDGpuKernel, XPBDGpuSolver};
    use super::*;

    const DT: f64 = 0.01;

    /// Rotation matrix about `axis` by `angle` (Rodrigues).
    fn rotation(axis: [f64; 3], angle: f64) -> Mat3 {
        let k = normalize3(axis);
        let (s, c) = angle.sin_cos();
        let cross = [[0.0, -k[2], k[1]], [k[2], 0.0, -k[0]], [-k[1], k[0], 0.0]];
        let mut r = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];
        for i in 0..3 {
            for j in 0..3 {
                let cross_sq: f64 = (0..3).map(|m| cross[i][m] * cross[m][j]).sum();
                r[i][j] += s * cross[i][j] + (1.0 - c) * cross_sq;
            }
        }
        r
    }

    /// Unit cube corners centred at the origin.
    fn cube() -> Vec<[f64; 3]> {
        let mut corners = Vec::new();
        for &x in &[-0.5, 0.5] {
            for &y in &[-0.5, 0.5] {
                for &z in &[-0.5, 0.5] {
                    corners.push([x, y, z]);
                }
            }
        }
        corners
    }

    fn rigid(points: &[[f64; 3]], r: &Mat3, shift: [f64; 3]) -> Vec<[f64; 3]> {
        points.iter().map(|&p| add3(mat_vec(r, p), shift)).collect()
    }

    fn buffer(points: &[[f64; 3]], mass: f64) -> DeformableGpuBuffer {
        let mut buf = DeformableGpuBuffer::new();
        for &p in points {
            buf.push(p, [0.0; 3], mass);
        }
        buf
    }

    fn max_distance(a: &[[f64; 3]], b: &[[f64; 3]]) -> f64 {
        a.iter()
            .zip(b)
            .map(|(&p, &q)| length3(sub3(p, q)))
            .fold(0.0, f64::max)
    }

    /// The unit corner tetrahedron scaled by `scale`; its volume is `scale³ / 6`.
    fn corner_tet(scale: f64) -> [[f64; 3]; 4] {
        [
            [0.0, 0.0, 0.0],
            [scale, 0.0, 0.0],
            [0.0, scale, 0.0],
            [0.0, 0.0, scale],
        ]
    }

    fn tet_volume(buf: &DeformableGpuBuffer) -> f64 {
        signed_volume(
            buf.positions[0],
            buf.positions[1],
            buf.positions[2],
            buf.positions[3],
        )
    }

    fn tet_buffer(points: [[f64; 3]; 4]) -> DeformableGpuBuffer {
        buffer(&points, 1.0)
    }

    // ── helpers ──────────────────────────────────────────────────────────

    #[test]
    fn inverse_dt_squared_accepts_only_positive_finite_steps() {
        assert_eq!(inverse_dt_squared(0.5), Some(4.0));
        assert_eq!(inverse_dt_squared(0.0), None);
        assert_eq!(inverse_dt_squared(-0.1), None);
        assert_eq!(inverse_dt_squared(f64::NAN), None);
        assert_eq!(inverse_dt_squared(f64::INFINITY), None);
        assert_eq!(inverse_dt_squared(1.0e-200), None);
    }

    // ── distance constraints ─────────────────────────────────────────────

    #[test]
    fn distance_with_compliance_reaches_the_xpbd_fixed_point_for_any_iteration_count() {
        // Two unit masses, stretched by C0 = 2.  Compliance 1e-3 at dt = 0.1 gives
        // α̃ = 0.1; the accumulated multiplier makes the projection converge to
        // C = α̃ C0 / (w_i + w_j + α̃) = 0.2 / 2.1 after the first iteration and stay
        // there.  Without the −α̃ λ term the residual would keep shrinking towards
        // zero as iterations are added.
        let expected_stretch = 2.0 * 0.1 / 2.1;
        let dt = 0.1;
        let mut reference: Option<[[f64; 3]; 2]> = None;
        for iterations in [1, 2, 8, 64] {
            let mut kernel = XPBDGpuKernel::new(iterations);
            kernel.add_constraint(XpbdDistConstraint::new(0, 1, 1.0, 1.0e-3));
            let mut positions = [[0.0; 3], [3.0, 0.0, 0.0]];
            kernel.solve_step(&mut positions, &[1.0, 1.0], dt);
            let stretch = length3(sub3(positions[1], positions[0])) - 1.0;
            assert!(
                (stretch - expected_stretch).abs() < 1.0e-12,
                "iterations {iterations}: stretch {stretch}"
            );
            match reference {
                None => reference = Some(positions),
                Some(first) => {
                    assert!(max_distance(&first, &positions) < 1.0e-12);
                }
            }
        }
    }

    #[test]
    fn distance_with_zero_compliance_is_the_rigid_projection() {
        let mut kernel = XPBDGpuKernel::new(1);
        kernel.add_constraint(XpbdDistConstraint::new(0, 1, 1.0, 0.0));
        let mut positions = [[0.0; 3], [3.0, 0.0, 0.0]];
        kernel.solve_step(&mut positions, &[1.0, 3.0], DT);
        // The lighter particle (inverse mass 3) moves three times as far as the
        // other one and the rest length is restored.
        assert!((positions[0][0] - 0.5).abs() < 1.0e-12, "{positions:?}");
        assert!((positions[1][0] - 1.5).abs() < 1.0e-12, "{positions:?}");
    }

    #[test]
    fn distance_ignores_missing_particles_and_masses() {
        let mut kernel = XPBDGpuKernel::new(4);
        kernel.add_constraint(XpbdDistConstraint::new(0, 5, 1.0, 0.0));
        kernel.add_constraint(XpbdDistConstraint::new(0, 1, 1.0, 0.0));
        let mut positions = [[0.0; 3], [3.0, 0.0, 0.0]];
        // `inv_masses` is shorter than `positions`: particle 1 has no mass entry.
        kernel.solve_step(&mut positions, &[1.0], DT);
        assert_eq!(positions, [[0.0; 3], [3.0, 0.0, 0.0]]);
    }

    #[test]
    fn solve_with_unusable_time_step_changes_nothing() {
        let mut kernel = XPBDGpuKernel::new(4);
        kernel.add_constraint(XpbdDistConstraint::new(0, 1, 1.0, 0.0));
        for dt in [0.0, -1.0, f64::NAN, f64::INFINITY] {
            let mut positions = [[0.0; 3], [3.0, 0.0, 0.0]];
            kernel.solve_step(&mut positions, &[1.0, 1.0], dt);
            assert_eq!(positions, [[0.0; 3], [3.0, 0.0, 0.0]], "dt = {dt}");
        }
    }

    // ── volume constraints ───────────────────────────────────────────────

    #[test]
    fn signed_volume_has_the_orientation_sign() {
        let [a, b, c, d] = corner_tet(1.0);
        assert!((signed_volume(a, b, c, d) - 1.0 / 6.0).abs() < 1.0e-15);
        assert!((signed_volume(a, c, b, d) + 1.0 / 6.0).abs() < 1.0e-15);
    }

    #[test]
    fn volume_gradients_match_finite_differences() {
        let points = [
            [0.1, -0.2, 0.3],
            [1.2, 0.1, -0.1],
            [-0.3, 0.9, 0.2],
            [0.2, 0.3, 1.1],
        ];
        let analytic = volume_gradients(points[0], points[1], points[2], points[3]);
        let step = 1.0e-6;
        for vertex in 0..4 {
            for axis in 0..3 {
                let mut plus = points;
                let mut minus = points;
                plus[vertex][axis] += step;
                minus[vertex][axis] -= step;
                let numeric = (signed_volume(plus[0], plus[1], plus[2], plus[3])
                    - signed_volume(minus[0], minus[1], minus[2], minus[3]))
                    / (2.0 * step);
                assert!(
                    (numeric - analytic[vertex][axis]).abs() < 1.0e-8,
                    "vertex {vertex} axis {axis}: {numeric} vs {}",
                    analytic[vertex][axis]
                );
            }
        }
    }

    #[test]
    fn volume_gradients_sum_to_zero() {
        let g = volume_gradients(
            [0.0, 0.1, 0.2],
            [1.0, 0.0, 0.3],
            [0.2, 1.0, 0.0],
            [0.1, 0.4, 1.3],
        );
        for axis in 0..3 {
            let sum: f64 = g.iter().map(|v| v[axis]).sum();
            assert!(sum.abs() < 1.0e-15, "axis {axis}: {sum}");
        }
    }

    #[test]
    fn compressed_tetrahedron_reaches_its_rest_volume_with_zero_compliance() {
        let rest_volume = 1.0 / 6.0;
        let mut solver = XPBDGpuSolver::new(30);
        solver.add_volume(VolumeConstraint::new([0, 1, 2, 3], rest_volume, 0.0));
        let mut buf = tet_buffer(corner_tet(0.7));
        let centroid_before = scale3(
            buf.positions.iter().fold([0.0; 3], |acc, &p| add3(acc, p)),
            0.25,
        );
        assert!((tet_volume(&buf) - 0.343 / 6.0).abs() < 1.0e-15);
        solver.solve(&mut buf, DT);
        assert!(
            (tet_volume(&buf) - rest_volume).abs() < 1.0e-6,
            "volume {}",
            tet_volume(&buf)
        );
        // Equal masses: the volume projection moves no net momentum.
        let centroid_after = scale3(
            buf.positions.iter().fold([0.0; 3], |acc, &p| add3(acc, p)),
            0.25,
        );
        assert!(length3(sub3(centroid_after, centroid_before)) < 1.0e-12);
    }

    #[test]
    fn stretched_tetrahedron_is_compressed_back() {
        let rest_volume = 1.0 / 6.0;
        let mut solver = XPBDGpuSolver::new(30);
        solver.add_volume(VolumeConstraint::new([0, 1, 2, 3], rest_volume, 0.0));
        let mut buf = tet_buffer(corner_tet(1.3));
        solver.solve(&mut buf, DT);
        assert!((tet_volume(&buf) - rest_volume).abs() < 1.0e-6);
    }

    #[test]
    fn inverted_tetrahedron_is_turned_back_to_positive_orientation() {
        // Swapping two vertices keeps |V| = V0 but makes the signed volume
        // negative: an unsigned volume constraint would call this satisfied.
        let rest_volume = 1.0 / 6.0;
        let [a, b, c, d] = corner_tet(1.0);
        let mut solver = XPBDGpuSolver::new(60);
        solver.add_volume(VolumeConstraint::new([0, 1, 2, 3], rest_volume, 0.0));
        let mut buf = tet_buffer([a, c, b, d]);
        assert!((tet_volume(&buf) + rest_volume).abs() < 1.0e-15);
        solver.solve(&mut buf, DT);
        assert!(
            (tet_volume(&buf) - rest_volume).abs() < 1.0e-6,
            "volume {}",
            tet_volume(&buf)
        );
    }

    #[test]
    fn volume_compliance_leaves_the_xpbd_residual() {
        // For a small violation the constraint is linear, so the converged XPBD
        // residual is C = α̃ C0 / (S + α̃) with S = Σ w |∇V|².
        let rest_volume = 1.0 / 6.0;
        let scale = 0.98;
        let points = corner_tet(scale);
        let initial = tet_volume(&tet_buffer(points)) - rest_volume;
        let gradients = volume_gradients(points[0], points[1], points[2], points[3]);
        let s: f64 = gradients.iter().map(|g| dot3(*g, *g)).sum();
        let dt = 0.1;
        let mut previous_residual = 0.0;
        for compliance in [0.0, 1.0e-3, 1.0e-2, 1.0e-1] {
            let alpha_tilde = compliance / (dt * dt);
            let mut solver = XPBDGpuSolver::new(40);
            solver.add_volume(VolumeConstraint::new([0, 1, 2, 3], rest_volume, compliance));
            let mut buf = tet_buffer(points);
            solver.solve(&mut buf, dt);
            let residual = tet_volume(&buf) - rest_volume;
            let expected = initial * alpha_tilde / (s + alpha_tilde);
            assert!(
                (residual - expected).abs() < 0.02 * initial.abs() + 1.0e-9,
                "compliance {compliance}: residual {residual}, expected {expected}"
            );
            // A softer (more compliant) constraint corrects less.
            assert!(residual.abs() >= previous_residual - 1.0e-12);
            previous_residual = residual.abs();
        }
        assert!(previous_residual > 0.5 * initial.abs());
    }

    #[test]
    fn volume_correction_does_not_depend_on_the_iteration_count_for_compliant_constraints() {
        let rest_volume = 1.0 / 6.0;
        let points = corner_tet(0.98);
        let mut volumes = Vec::new();
        for iterations in [1, 5, 40] {
            let mut solver = XPBDGpuSolver::new(iterations);
            solver.add_volume(VolumeConstraint::new([0, 1, 2, 3], rest_volume, 1.0e-2));
            let mut buf = tet_buffer(points);
            solver.solve(&mut buf, 0.1);
            volumes.push(tet_volume(&buf));
        }
        let spread = volumes.iter().cloned().fold(f64::NEG_INFINITY, f64::max)
            - volumes.iter().cloned().fold(f64::INFINITY, f64::min);
        assert!(spread < 2.0e-4, "volumes {volumes:?}");
    }

    #[test]
    fn volume_projection_never_moves_pinned_particles() {
        let rest_volume = 1.0 / 6.0;
        let mut solver = XPBDGpuSolver::new(40);
        solver.add_volume(VolumeConstraint::new([0, 1, 2, 3], rest_volume, 0.0));
        let mut buf = DeformableGpuBuffer::new();
        let points = corner_tet(0.7);
        buf.push(points[0], [0.0; 3], 0.0);
        for &p in &points[1..] {
            buf.push(p, [0.0; 3], 1.0);
        }
        solver.solve(&mut buf, DT);
        assert_eq!(buf.positions[0], points[0]);
        assert!((tet_volume(&buf) - rest_volume).abs() < 1.0e-6);
    }

    #[test]
    fn volume_constraints_with_bad_indices_are_ignored() {
        let mut solver = XPBDGpuSolver::new(5);
        solver.add_volume(VolumeConstraint::new([0, 1, 2, 9], 1.0, 0.0));
        solver.add_volume(VolumeConstraint::new([0, 1, 1, 3], 1.0, 0.0));
        let mut buf = tet_buffer(corner_tet(0.7));
        let before = buf.positions.clone();
        solver.solve(&mut buf, DT);
        assert_eq!(buf.positions, before);
    }

    #[test]
    fn all_fixed_tetrahedron_is_left_alone() {
        let mut solver = XPBDGpuSolver::new(5);
        solver.add_volume(VolumeConstraint::new([0, 1, 2, 3], 1.0, 0.0));
        let mut buf = buffer(&corner_tet(0.7), 0.0);
        let before = buf.positions.clone();
        solver.solve(&mut buf, DT);
        assert_eq!(buf.positions, before);
    }

    // ── shape matching ───────────────────────────────────────────────────

    fn cube_solver(stiffness: f64, indices: Vec<usize>) -> XPBDGpuSolver {
        let rest = cube();
        let rest_positions = indices.iter().map(|&i| rest[i]).collect();
        let mut solver = XPBDGpuSolver::new(1);
        solver.add_shape(ShapeMatchConstraint::new(
            indices,
            rest_positions,
            stiffness,
        ));
        solver
    }

    #[test]
    fn rigidly_rotated_cube_is_a_fixed_point_of_shape_matching() {
        let r = rotation([1.0, 2.0, 3.0], 0.9);
        let moved = rigid(&cube(), &r, [0.3, -1.2, 2.0]);
        let mut buf = buffer(&moved, 1.0);
        cube_solver(1.0, (0..8).collect()).solve(&mut buf, DT);
        assert!(
            max_distance(&buf.positions, &moved) < 1.0e-12,
            "a rigid rotation must not be pulled back to the rest orientation"
        );
    }

    #[test]
    fn quarter_turn_of_a_cube_is_not_undone() {
        // The rotated cube occupies the same corner set, but every particle has
        // moved to another corner: a translation-only goal would drag each one back.
        let r = rotation([0.0, 0.0, 1.0], std::f64::consts::FRAC_PI_2);
        let moved = rigid(&cube(), &r, [0.0; 3]);
        let mut buf = buffer(&moved, 1.0);
        cube_solver(1.0, (0..8).collect()).solve(&mut buf, DT);
        assert!(max_distance(&buf.positions, &moved) < 1.0e-12);
    }

    #[test]
    fn deformed_rotated_cube_becomes_a_rigid_copy_of_the_rest_shape() {
        let r = rotation([0.3, -1.0, 0.5], 2.1);
        let mut moved = rigid(&cube(), &r, [1.0, 2.0, 3.0]);
        moved[3] = add3(moved[3], [0.2, -0.1, 0.15]);
        let mut buf = buffer(&moved, 1.0);
        cube_solver(1.0, (0..8).collect()).solve(&mut buf, DT);
        let rest = cube();
        for a in 0..8 {
            for b in (a + 1)..8 {
                let now = length3(sub3(buf.positions[a], buf.positions[b]));
                let then = length3(sub3(rest[a], rest[b]));
                assert!(
                    (now - then).abs() < 1.0e-12,
                    "pair ({a}, {b}): {now} vs {then}"
                );
            }
        }
    }

    #[test]
    fn shape_matching_conserves_the_mass_weighted_centre() {
        let r = rotation([0.0, 1.0, 1.0], 0.4);
        let mut moved = rigid(&cube(), &r, [0.5, 0.0, -0.5]);
        moved[6] = add3(moved[6], [0.3, 0.2, -0.1]);
        let mut buf = DeformableGpuBuffer::new();
        let masses = [1.0, 2.0, 3.0, 1.5, 0.5, 4.0, 1.0, 2.5];
        for (&p, &m) in moved.iter().zip(&masses) {
            buf.push(p, [0.0; 3], m);
        }
        let weighted_centre = |buf: &DeformableGpuBuffer| {
            let total: f64 = masses.iter().sum();
            let sum = buf
                .positions
                .iter()
                .zip(&masses)
                .fold([0.0; 3], |acc, (&p, &m)| add3(acc, scale3(p, m)));
            scale3(sum, 1.0 / total)
        };
        let before = weighted_centre(&buf);
        cube_solver(1.0, (0..8).collect()).solve(&mut buf, DT);
        let after = weighted_centre(&buf);
        assert!(length3(sub3(after, before)) < 1.0e-12);
    }

    #[test]
    fn partial_stiffness_moves_a_fraction_of_the_way_to_the_goal() {
        let r = rotation([1.0, 0.0, 0.0], 0.5);
        let mut moved = rigid(&cube(), &r, [0.0; 3]);
        moved[0] = add3(moved[0], [0.4, 0.0, 0.0]);
        let mut full = buffer(&moved, 1.0);
        let mut half = buffer(&moved, 1.0);
        cube_solver(1.0, (0..8).collect()).solve(&mut full, DT);
        cube_solver(0.5, (0..8).collect()).solve(&mut half, DT);
        for (k, ((&start, &goal), &halfway)) in moved
            .iter()
            .zip(&full.positions)
            .zip(&half.positions)
            .enumerate()
        {
            let expected = scale3(add3(start, goal), 0.5);
            assert!(length3(sub3(halfway, expected)) < 1.0e-12, "particle {k}");
        }
    }

    #[test]
    fn pinned_particle_anchors_the_shape_and_never_moves() {
        // Swing the cube about its pinned corner 0 (at the origin of the rest pose
        // shifted to the pin) and disturb a corner.
        let pin = [0.5, 0.5, 0.5];
        let rest: Vec<[f64; 3]> = cube().iter().map(|&p| add3(p, pin)).collect();
        let r = rotation([0.0, 0.0, 1.0], 0.5);
        let mut moved: Vec<[f64; 3]> = rest
            .iter()
            .map(|&p| add3(mat_vec(&r, sub3(p, rest[0])), rest[0]))
            .collect();
        moved[7] = add3(moved[7], [0.1, -0.2, 0.05]);
        let mut buf = DeformableGpuBuffer::new();
        buf.push(moved[0], [0.0; 3], 0.0);
        for &p in &moved[1..] {
            buf.push(p, [0.0; 3], 1.0);
        }
        let mut solver = XPBDGpuSolver::new(1);
        solver.add_shape(ShapeMatchConstraint::new(
            (0..8).collect(),
            rest.clone(),
            1.0,
        ));
        solver.solve(&mut buf, DT);
        assert_eq!(
            buf.positions[0], moved[0],
            "a pinned particle must not move"
        );
        for k in 1..8 {
            let now = length3(sub3(buf.positions[k], buf.positions[0]));
            let then = length3(sub3(rest[k], rest[0]));
            assert!((now - then).abs() < 1.0e-5, "particle {k}: {now} vs {then}");
        }
    }

    #[test]
    fn shape_matching_of_a_planar_shape_keeps_a_rotated_square() {
        let square = [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [1.0, 1.0, 0.0],
            [0.0, 1.0, 0.0],
        ];
        let r = rotation([1.0, 0.0, 0.0], std::f64::consts::FRAC_PI_2);
        let moved = rigid(&square, &r, [0.0, 0.0, 1.0]);
        let mut buf = buffer(&moved, 1.0);
        let mut solver = XPBDGpuSolver::new(1);
        solver.add_shape(ShapeMatchConstraint::new(
            (0..4).collect(),
            square.to_vec(),
            1.0,
        ));
        solver.solve(&mut buf, DT);
        assert!(max_distance(&buf.positions, &moved) < 1.0e-12);
    }

    #[test]
    fn shape_matching_uses_only_particles_that_exist() {
        let r = rotation([0.0, 1.0, 0.0], 0.8);
        let moved = rigid(&cube(), &r, [0.0; 3]);
        let mut buf = buffer(&moved[..4], 1.0);
        // Indices 4..8 do not exist in the buffer, and the rest pose lists only
        // seven positions for eight indices.
        let mut solver = XPBDGpuSolver::new(1);
        solver.add_shape(ShapeMatchConstraint::new(
            (0..8).collect(),
            cube()[..7].to_vec(),
            1.0,
        ));
        solver.solve(&mut buf, DT);
        assert!(buf.positions.iter().flatten().all(|v| v.is_finite()));
        // The four existing particles keep the shape of the corresponding rest
        // positions (their distances are those of the rest cube corners).
        let rest = cube();
        for a in 0..4 {
            for b in (a + 1)..4 {
                let now = length3(sub3(buf.positions[a], buf.positions[b]));
                let then = length3(sub3(rest[a], rest[b]));
                assert!((now - then).abs() < 1.0e-9, "pair ({a}, {b})");
            }
        }
    }

    #[test]
    fn shape_matching_of_nothing_or_a_single_particle_is_a_no_op() {
        let mut solver = XPBDGpuSolver::new(3);
        solver.add_shape(ShapeMatchConstraint::new(Vec::new(), Vec::new(), 1.0));
        solver.add_shape(ShapeMatchConstraint::new(
            vec![0],
            vec![[9.0, 9.0, 9.0]],
            1.0,
        ));
        let mut buf = buffer(&[[1.0, 2.0, 3.0]], 1.0);
        solver.solve(&mut buf, DT);
        assert_eq!(buf.positions, vec![[1.0, 2.0, 3.0]]);
    }

    #[test]
    fn shape_matching_with_a_non_finite_stiffness_changes_nothing() {
        let moved = rigid(&cube(), &rotation([1.0, 1.0, 0.0], 0.3), [0.0; 3]);
        let mut buf = buffer(&moved, 1.0);
        cube_solver(f64::NAN, (0..8).collect()).solve(&mut buf, DT);
        assert_eq!(buf.positions, moved);
    }

    // ── all constraint kinds together ────────────────────────────────────

    #[test]
    fn rotated_rest_tetrahedron_satisfies_every_constraint_kind() {
        let rest = corner_tet(1.0);
        let r = rotation([0.2, 0.7, -0.4], 1.1);
        let moved = rigid(&rest, &r, [0.0, 1.0, 0.0]);
        let mut solver = XPBDGpuSolver::new(10);
        for a in 0..4 {
            for b in (a + 1)..4 {
                solver.add_distance(a, b, length3(sub3(rest[a], rest[b])), 1.0e-6);
            }
        }
        solver.add_volume(VolumeConstraint::from_rest_positions(
            [0, 1, 2, 3],
            &rest,
            1.0e-6,
        ));
        solver.add_shape(ShapeMatchConstraint::new(
            (0..4).collect(),
            rest.to_vec(),
            1.0,
        ));
        let mut buf = buffer(&moved, 1.0);
        solver.solve(&mut buf, DT);
        assert!(max_distance(&buf.positions, &moved) < 1.0e-9);
    }

    #[test]
    fn volume_constraint_from_rest_positions_records_the_signed_volume() {
        let [a, b, c, d] = corner_tet(2.0);
        let right_handed = VolumeConstraint::from_rest_positions([0, 1, 2, 3], &[a, b, c, d], 0.0);
        assert!((right_handed.rest_volume - 8.0 / 6.0).abs() < 1.0e-15);
        let left_handed = VolumeConstraint::from_rest_positions([0, 1, 2, 3], &[a, c, b, d], 0.5);
        assert!((left_handed.rest_volume + 8.0 / 6.0).abs() < 1.0e-15);
        assert_eq!(left_handed.compliance, 0.5);
        assert_eq!(left_handed.indices, [0, 1, 2, 3]);
    }
}
