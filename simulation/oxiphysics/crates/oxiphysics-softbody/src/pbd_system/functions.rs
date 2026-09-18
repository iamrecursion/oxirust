//! Auto-generated module
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use super::types::{ConstraintColoring, PbdConstraint, PbdConstraintType, PbdParticle, PbdSystem};

/// Euclidean distance between two 3-D points.
pub fn dist3(a: [f64; 3], b: [f64; 3]) -> f64 {
    let d = sub3(a, b);
    (d[0] * d[0] + d[1] * d[1] + d[2] * d[2]).sqrt()
}
/// Component-wise subtraction.
pub fn sub3(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}
/// Component-wise addition.
pub fn add3(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [a[0] + b[0], a[1] + b[1], a[2] + b[2]]
}
/// Scale a vector by a scalar.
pub fn scale3(v: [f64; 3], s: f64) -> [f64; 3] {
    [v[0] * s, v[1] * s, v[2] * s]
}
pub(super) fn dot3(a: [f64; 3], b: [f64; 3]) -> f64 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}
pub(super) fn len3(v: [f64; 3]) -> f64 {
    (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt()
}
pub(super) fn cross3(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}
/// XPBD distance-constraint solver.
///
/// Adjusts the predicted positions of particles `i` and `j` so that their
/// separation approaches `rest`.
pub fn solve_distance_xpbd(
    particles: &mut [PbdParticle],
    i: usize,
    j: usize,
    rest: f64,
    compliance: f64,
    dt: f64,
) {
    let wi = particles[i].inv_mass;
    let wj = particles[j].inv_mass;
    let w_sum = wi + wj;
    if w_sum < 1e-30 {
        return;
    }
    let pi = particles[i].predicted;
    let pj = particles[j].predicted;
    let diff = sub3(pi, pj);
    let cur_len = len3(diff);
    if cur_len < 1e-12 {
        return;
    }
    let alpha = compliance / (dt * dt);
    let c = cur_len - rest;
    let lambda = -c / (w_sum + alpha);
    let n = scale3(diff, 1.0 / cur_len);
    particles[i].predicted = add3(pi, scale3(n, lambda * wi));
    particles[j].predicted = add3(pj, scale3(n, -lambda * wj));
}
/// XPBD floor-collision solver (unilateral).
///
/// If the predicted position is below `y_level`, push it back up and
/// apply a velocity correction according to `restitution`.
pub fn solve_floor_xpbd(particle: &mut PbdParticle, y_level: f64, restitution: f64) {
    if particle.predicted[1] < y_level {
        let pen = y_level - particle.predicted[1];
        particle.predicted[1] += pen;
        let vy = particle.velocity[1];
        if vy < 0.0 {
            particle.velocity[1] = -restitution * vy;
        }
    }
}
/// Signed volume of a tetrahedron defined by four points.
pub fn compute_tetrahedron_volume(p0: [f64; 3], p1: [f64; 3], p2: [f64; 3], p3: [f64; 3]) -> f64 {
    let a = sub3(p1, p0);
    let b = sub3(p2, p0);
    let c = sub3(p3, p0);
    dot3(a, cross3(b, c)) / 6.0
}
/// Gradient of the signed tetrahedron volume
/// `V = (p1 - p0) . ((p2 - p0) x (p3 - p0)) / 6` with respect to the four corner
/// positions `[p0, p1, p2, p3]`.
///
/// With `a = p1 - p0`, `b = p2 - p0` and `c = p3 - p0` the gradients of the
/// last three corners are `b x c / 6`, `c x a / 6` and `a x b / 6`; the gradient
/// of `p0` is minus their sum, so the four gradients add up to zero (moving the
/// whole tetrahedron does not change its volume).
pub fn compute_tetrahedron_volume_gradients(
    p0: [f64; 3],
    p1: [f64; 3],
    p2: [f64; 3],
    p3: [f64; 3],
) -> [[f64; 3]; 4] {
    let a = sub3(p1, p0);
    let b = sub3(p2, p0);
    let c = sub3(p3, p0);
    let g1 = scale3(cross3(b, c), 1.0 / 6.0);
    let g2 = scale3(cross3(c, a), 1.0 / 6.0);
    let g3 = scale3(cross3(a, b), 1.0 / 6.0);
    let g0 = scale3(add3(add3(g1, g2), g3), -1.0);
    [g0, g1, g2, g3]
}
/// Smallest distance of a hinge wing from the hinge line, relative to the hinge
/// length, for which the triangle still defines a dihedral angle.
const MIN_HINGE_ASPECT: f64 = 1.0e-9;
/// Interior dihedral angle of the hinge `(p0, p1 | p2, p3)` together with its
/// gradient with respect to the four positions, or `None` when the hinge is
/// degenerate (zero-length shared edge or a collapsed triangle).
///
/// `p0`-`p1` is the edge shared by the triangles `(p0, p1, p2)` and
/// `(p0, p1, p3)`.  The angle `theta` in `[0, 2 pi]` is `pi` for a flat hinge,
/// smaller than `pi` when the wings `p2`, `p3` are folded towards the side that
/// the normal `(p1 - p0) x (p2 - p0)` points to and larger than `pi` when they are
/// folded to the other side.  It is the supplement of the signed angle `phi` from
/// the normal `N2 = (p3 - p1) x (p3 - p0)` of the second triangle to the normal
/// `N1 = (p2 - p0) x (p2 - p1)` of the first one, measured about the shared edge.
///
/// The gradient follows Bridson et al. 2003: with `e = p1 - p0`,
/// `e_hat = e / |e|` and the area vectors `N1`, `N2`,
///
/// ```text
/// d phi / d p2 =  |e| N1 / |N1|^2
/// d phi / d p3 =  |e| N2 / |N2|^2
/// d phi / d p0 =  ((p2 - p1) . e_hat) N1 / |N1|^2 + ((p3 - p1) . e_hat) N2 / |N2|^2
/// d phi / d p1 = -((p2 - p0) . e_hat) N1 / |N1|^2 - ((p3 - p0) . e_hat) N2 / |N2|^2
/// ```
///
/// and `d theta = -d phi`.  The four gradients add up to zero and carry no net
/// torque, so a correction along them conserves linear and angular momentum.
pub fn dihedral_angle_with_gradients(
    p0: [f64; 3],
    p1: [f64; 3],
    p2: [f64; 3],
    p3: [f64; 3],
) -> Option<(f64, [[f64; 3]; 4])> {
    let edge = sub3(p1, p0);
    let edge_len = len3(edge);
    if !edge_len.is_finite() || edge_len <= 0.0 {
        return None;
    }
    let e_hat = scale3(edge, 1.0 / edge_len);
    let n1 = cross3(sub3(p2, p0), sub3(p2, p1));
    let n2 = cross3(sub3(p3, p1), sub3(p3, p0));
    let n1_sq = dot3(n1, n1);
    let n2_sq = dot3(n2, n2);
    if !n1_sq.is_finite() || !n2_sq.is_finite() {
        return None;
    }
    // |N| = |e| * (distance of the wing from the hinge line).
    let min_sq = (MIN_HINGE_ASPECT * edge_len * edge_len).powi(2);
    if n1_sq <= min_sq || n2_sq <= min_sq {
        return None;
    }
    let phi = dot3(cross3(n2, n1), e_hat).atan2(dot3(n2, n1));
    let a1 = scale3(n1, 1.0 / n1_sq);
    let a2 = scale3(n2, 1.0 / n2_sq);
    let along = |from: [f64; 3], wing_1: [f64; 3], wing_2: [f64; 3]| -> [f64; 3] {
        add3(
            scale3(a1, dot3(sub3(wing_1, from), e_hat)),
            scale3(a2, dot3(sub3(wing_2, from), e_hat)),
        )
    };
    let dphi_dp0 = along(p1, p2, p3);
    let dphi_dp1 = scale3(along(p0, p2, p3), -1.0);
    let dphi_dp2 = scale3(a1, edge_len);
    let dphi_dp3 = scale3(a2, edge_len);
    let gradients = [dphi_dp0, dphi_dp1, dphi_dp2, dphi_dp3].map(|g| scale3(g, -1.0));
    Some((std::f64::consts::PI - phi, gradients))
}
/// Interior dihedral angle of the hinge `(p0, p1 | p2, p3)` in `[0, 2 pi]`
/// (`pi` for a flat hinge), or `None` for a degenerate hinge.
///
/// See [`dihedral_angle_with_gradients`] for the orientation convention.
pub fn compute_dihedral_angle(
    p0: [f64; 3],
    p1: [f64; 3],
    p2: [f64; 3],
    p3: [f64; 3],
) -> Option<f64> {
    dihedral_angle_with_gradients(p0, p1, p2, p3).map(|(angle, _)| angle)
}
/// Difference of two angles wrapped into `(-pi, pi]`, the shortest rotation that
/// carries one onto the other.
fn wrap_angle(difference: f64) -> f64 {
    use std::f64::consts::{PI, TAU};
    let wrapped = difference % TAU;
    if wrapped > PI {
        wrapped - TAU
    } else if wrapped <= -PI {
        wrapped + TAU
    } else {
        wrapped
    }
}
/// The first `N` particle indices of `constraint`, or `None` when it lists fewer
/// or one of them is not a particle of a system with `particle_count` particles.
fn constraint_indices<const N: usize>(
    constraint: &PbdConstraint,
    particle_count: usize,
) -> Option<[usize; N]> {
    let indices: [usize; N] = constraint.particles.get(..N)?.try_into().ok()?;
    indices
        .iter()
        .all(|&index| index < particle_count)
        .then_some(indices)
}
/// `true` when no index occurs twice.
fn all_distinct<const N: usize>(indices: &[usize; N]) -> bool {
    indices
        .iter()
        .enumerate()
        .all(|(k, index)| !indices[k + 1..].contains(index))
}
/// One XPBD projection of a scalar constraint with value `c` whose gradients with
/// respect to the predicted positions of `indices` are `gradients`.
///
/// `lambda = -c / (sum_i w_i |g_i|^2 + alpha)` with `alpha = compliance / dt^2`,
/// and every predicted position moves by `w_i g_i lambda`.  Because the
/// gradients of a translation- and rotation-invariant constraint add up to zero
/// and carry no torque, the correction conserves linear and angular momentum.
fn apply_scalar_constraint<const N: usize>(
    particles: &mut [PbdParticle],
    indices: [usize; N],
    gradients: [[f64; 3]; N],
    c: f64,
    compliance: f64,
    sub_dt: f64,
) {
    let weights = indices.map(|index| particles[index].inv_mass);
    let weighted_norm: f64 = gradients
        .iter()
        .zip(weights.iter())
        .map(|(g, w)| w * dot3(*g, *g))
        .sum();
    let alpha = compliance / (sub_dt * sub_dt);
    let denominator = weighted_norm + alpha;
    if !denominator.is_finite() || denominator <= 1e-30 || !c.is_finite() {
        return;
    }
    let lambda = -c / denominator;
    for ((&index, g), w) in indices.iter().zip(gradients.iter()).zip(weights.iter()) {
        let particle = &mut particles[index];
        particle.predicted = add3(particle.predicted, scale3(*g, lambda * w));
    }
}
/// Project one constraint onto the predicted positions of `particles`.
///
/// This is the single place that knows how to solve each constraint type; both
/// [`PbdSystem::step`] and [`xpbd_step_colored`] call it, so the two solvers
/// always handle the same set of constraints.  `sub_dt` is the sub-step length
/// used to scale the XPBD compliance.
///
/// - `Distance`: [`solve_distance_xpbd`].
/// - `BendingDihedral`: XPBD on `C = theta - rest_angle` (the angle difference is
///   wrapped into `(-pi, pi]`) with the exact angle gradients of
///   [`dihedral_angle_with_gradients`] on all four particles.
/// - `VolumeConservation`: XPBD on `C = V - rest_volume` with the exact signed
///   volume gradients of [`compute_tetrahedron_volume_gradients`].
/// - `CollisionFloor`: [`solve_floor_xpbd`].
/// - `CollisionSphere`: [`solve_sphere_collision`] for particles that are not
///   fixed.
///
/// A constraint that lists too few particles, an index outside `particles`, or
/// (for bending and volume) the same particle twice is ignored, as is a
/// degenerate hinge.
pub fn project_constraint(particles: &mut [PbdParticle], constraint: &PbdConstraint, sub_dt: f64) {
    let count = particles.len();
    match constraint.constraint_type {
        PbdConstraintType::Distance {
            rest_length,
            compliance,
        } => {
            if let Some([i, j]) = constraint_indices::<2>(constraint, count) {
                solve_distance_xpbd(particles, i, j, rest_length, compliance, sub_dt);
            }
        }
        PbdConstraintType::BendingDihedral {
            rest_angle,
            compliance,
        } => {
            let Some(indices) = constraint_indices::<4>(constraint, count) else {
                return;
            };
            if !all_distinct(&indices) {
                return;
            }
            let [p0, p1, p2, p3] = indices.map(|index| particles[index].predicted);
            let Some((angle, gradients)) = dihedral_angle_with_gradients(p0, p1, p2, p3) else {
                return;
            };
            let c = wrap_angle(angle - rest_angle);
            apply_scalar_constraint(particles, indices, gradients, c, compliance, sub_dt);
        }
        PbdConstraintType::VolumeConservation {
            rest_volume,
            compliance,
        } => {
            let Some(indices) = constraint_indices::<4>(constraint, count) else {
                return;
            };
            if !all_distinct(&indices) {
                return;
            }
            let [p0, p1, p2, p3] = indices.map(|index| particles[index].predicted);
            let c = compute_tetrahedron_volume(p0, p1, p2, p3) - rest_volume;
            let gradients = compute_tetrahedron_volume_gradients(p0, p1, p2, p3);
            apply_scalar_constraint(particles, indices, gradients, c, compliance, sub_dt);
        }
        PbdConstraintType::CollisionFloor {
            y_level,
            restitution,
        } => {
            if let Some([i]) = constraint_indices::<1>(constraint, count) {
                solve_floor_xpbd(&mut particles[i], y_level, restitution);
            }
        }
        PbdConstraintType::CollisionSphere {
            center,
            radius,
            restitution,
        } => {
            if let Some([i]) = constraint_indices::<1>(constraint, count) {
                let particle = &mut particles[i];
                if !particle.fixed {
                    solve_sphere_collision(particle, center, radius, restitution);
                }
            }
        }
    }
}
/// Magnitude `|C|` of the violation of `constraint` at the current particle
/// positions: the distance error, the angle error (wrapped into `(-pi, pi]`), the
/// volume error, or the penetration depth below the floor / into the sphere.
///
/// A constraint that [`project_constraint`] would ignore reports zero.
pub fn constraint_violation(particles: &[PbdParticle], constraint: &PbdConstraint) -> f64 {
    let count = particles.len();
    let positions = |indices: [usize; 4]| indices.map(|index| particles[index].position);
    match constraint.constraint_type {
        PbdConstraintType::Distance { rest_length, .. } => {
            constraint_indices::<2>(constraint, count).map_or(0.0, |[i, j]| {
                (dist3(particles[i].position, particles[j].position) - rest_length).abs()
            })
        }
        PbdConstraintType::BendingDihedral { rest_angle, .. } => {
            constraint_indices::<4>(constraint, count)
                .filter(all_distinct)
                .and_then(|indices| {
                    let [p0, p1, p2, p3] = positions(indices);
                    compute_dihedral_angle(p0, p1, p2, p3)
                })
                .map_or(0.0, |angle| wrap_angle(angle - rest_angle).abs())
        }
        PbdConstraintType::VolumeConservation { rest_volume, .. } => {
            constraint_indices::<4>(constraint, count)
                .filter(all_distinct)
                .map_or(0.0, |indices| {
                    let [p0, p1, p2, p3] = positions(indices);
                    (compute_tetrahedron_volume(p0, p1, p2, p3) - rest_volume).abs()
                })
        }
        PbdConstraintType::CollisionFloor { y_level, .. } => {
            constraint_indices::<1>(constraint, count)
                .map_or(0.0, |[i]| (y_level - particles[i].position[1]).max(0.0))
        }
        PbdConstraintType::CollisionSphere { center, radius, .. } => {
            constraint_indices::<1>(constraint, count).map_or(0.0, |[i]| {
                (radius - dist3(particles[i].position, center)).max(0.0)
            })
        }
    }
}
/// Build a vertical chain of `n` particles.
///
/// The top particle is fixed; the rest hang down with spacing `spacing`.
pub fn build_chain(n: usize, spacing: f64, mass: f64, compliance: f64) -> PbdSystem {
    let mut sys = PbdSystem::new([0.0, -9.81, 0.0], 10);
    for i in 0..n {
        let y = -(i as f64) * spacing;
        if i == 0 {
            sys.add_fixed_particle([0.0, y, 0.0]);
        } else {
            sys.add_particle([0.0, y, 0.0], mass);
        }
    }
    for i in 0..n.saturating_sub(1) {
        sys.add_distance_constraint(i, i + 1, compliance);
    }
    sys
}
/// Build a flat 2-D grid cloth in the XZ plane.
///
/// Corner (0, 0) is at the origin.  Structural (horizontal + vertical)
/// distance constraints are added for every edge.  The top row is **not**
/// pinned by default; callers can pin particles manually.
pub fn build_grid_cloth(
    rows: usize,
    cols: usize,
    spacing: f64,
    mass: f64,
    compliance: f64,
) -> PbdSystem {
    let mut sys = PbdSystem::new([0.0, -9.81, 0.0], 10);
    for r in 0..rows {
        for c in 0..cols {
            let x = c as f64 * spacing;
            let z = r as f64 * spacing;
            sys.add_particle([x, 0.0, z], mass);
        }
    }
    let idx = |r: usize, c: usize| r * cols + c;
    for r in 0..rows {
        for c in 0..cols - 1 {
            sys.add_distance_constraint(idx(r, c), idx(r, c + 1), compliance);
        }
    }
    for r in 0..rows - 1 {
        for c in 0..cols {
            sys.add_distance_constraint(idx(r, c), idx(r + 1, c), compliance);
        }
    }
    sys
}
/// Sphere collision constraint (unilateral).
///
/// Pushes a particle outside the sphere if it penetrates: the predicted position
/// moves along the outward normal to the sphere surface, and an inward velocity
/// component is reflected according to `restitution`.
///
/// A particle exactly at the centre has no outward normal; it leaves along the
/// direction from the centre to its position at the start of the step, or along
/// `+y` when that position is the centre too.
pub fn solve_sphere_collision(
    particle: &mut PbdParticle,
    center: [f64; 3],
    radius: f64,
    restitution: f64,
) {
    let d = sub3(particle.predicted, center);
    let dist = len3(d);
    if dist < radius {
        let n = if dist > 1e-12 {
            scale3(d, 1.0 / dist)
        } else {
            let back = sub3(particle.position, center);
            let back_len = len3(back);
            if back_len > 1e-12 {
                scale3(back, 1.0 / back_len)
            } else {
                [0.0, 1.0, 0.0]
            }
        };
        let pen = radius - dist;
        particle.predicted = add3(particle.predicted, scale3(n, pen));
        let vn = dot3(particle.velocity, n);
        if vn < 0.0 {
            particle.velocity = add3(particle.velocity, scale3(n, (-1.0 - restitution) * vn));
        }
    }
}
/// AABB collision constraint (unilateral, axis-aligned bounding box).
///
/// Pushes a particle back inside the box if it escapes.
pub fn solve_aabb_collision(
    particle: &mut PbdParticle,
    aabb_min: [f64; 3],
    aabb_max: [f64; 3],
    restitution: f64,
) {
    for axis in 0..3 {
        if particle.predicted[axis] < aabb_min[axis] {
            let pen = aabb_min[axis] - particle.predicted[axis];
            particle.predicted[axis] += pen;
            let v = particle.velocity[axis];
            if v < 0.0 {
                particle.velocity[axis] = -restitution * v;
            }
        } else if particle.predicted[axis] > aabb_max[axis] {
            let pen = particle.predicted[axis] - aabb_max[axis];
            particle.predicted[axis] -= pen;
            let v = particle.velocity[axis];
            if v > 0.0 {
                particle.velocity[axis] = -restitution * v;
            }
        }
    }
}
/// Triangle collision constraint.
///
/// If a particle is below the plane defined by triangle (a, b, c), project it back.
pub fn solve_triangle_collision(
    particle: &mut PbdParticle,
    ta: [f64; 3],
    tb: [f64; 3],
    tc: [f64; 3],
    restitution: f64,
) {
    let n_raw = cross3(sub3(tb, ta), sub3(tc, ta));
    let n_len = len3(n_raw);
    if n_len < 1e-12 {
        return;
    }
    let n = scale3(n_raw, 1.0 / n_len);
    let signed_dist = dot3(n, sub3(particle.predicted, ta));
    if signed_dist < 0.0 {
        particle.predicted = add3(particle.predicted, scale3(n, -signed_dist));
        let vn = dot3(particle.velocity, n);
        if vn < 0.0 {
            particle.velocity = add3(particle.velocity, scale3(n, (-1.0 - restitution) * vn));
        }
    }
}
/// Enforce a minimum separation distance between all particle pairs.
///
/// This is O(n²) and only suitable for small systems.
pub fn solve_self_collision_all_pairs(
    particles: &mut [PbdParticle],
    min_distance: f64,
    stiffness: f64,
) {
    let n = particles.len();
    for i in 0..n {
        for j in (i + 1)..n {
            let pi = particles[i].predicted;
            let pj = particles[j].predicted;
            let d = sub3(pi, pj);
            let dist = len3(d);
            if dist < min_distance && dist > 1e-12 {
                let pen = min_distance - dist;
                let n_hat = scale3(d, 1.0 / dist);
                let wi = particles[i].inv_mass;
                let wj = particles[j].inv_mass;
                let w_sum = wi + wj;
                if w_sum < 1e-30 {
                    continue;
                }
                let corr = pen * stiffness / w_sum;
                particles[i].predicted = add3(pi, scale3(n_hat, corr * wi));
                particles[j].predicted = add3(pj, scale3(n_hat, -corr * wj));
            }
        }
    }
}
/// Self-collision hash-grid accelerated (approximate).
///
/// Divides space into a grid and only checks nearby particles.
pub fn solve_self_collision_grid(
    particles: &mut [PbdParticle],
    min_distance: f64,
    stiffness: f64,
    cell_size: f64,
) {
    use std::collections::HashMap;
    let n = particles.len();
    if n < 2 {
        return;
    }
    let inv_cell = 1.0 / cell_size.max(1e-12);
    let cell_key = |p: [f64; 3]| -> (i64, i64, i64) {
        (
            (p[0] * inv_cell).floor() as i64,
            (p[1] * inv_cell).floor() as i64,
            (p[2] * inv_cell).floor() as i64,
        )
    };
    let mut grid: HashMap<(i64, i64, i64), Vec<usize>> = HashMap::new();
    for (i, p) in particles.iter().enumerate() {
        let key = cell_key(p.predicted);
        grid.entry(key).or_default().push(i);
    }
    let mut pairs: Vec<(usize, usize)> = Vec::new();
    for (&(cx, cy, cz), cell_indices) in &grid {
        for dx in -1i64..=1 {
            for dy in -1i64..=1 {
                for dz in -1i64..=1 {
                    let nb_key = (cx + dx, cy + dy, cz + dz);
                    if let Some(nb_indices) = grid.get(&nb_key) {
                        for &i in cell_indices {
                            for &j in nb_indices {
                                if i < j {
                                    pairs.push((i, j));
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    pairs.sort_unstable();
    pairs.dedup();
    for (i, j) in pairs {
        let pi = particles[i].predicted;
        let pj = particles[j].predicted;
        let d = sub3(pi, pj);
        let dist = len3(d);
        if dist < min_distance && dist > 1e-12 {
            let pen = min_distance - dist;
            let n_hat = scale3(d, 1.0 / dist);
            let wi = particles[i].inv_mass;
            let wj = particles[j].inv_mass;
            let w_sum = wi + wj;
            if w_sum < 1e-30 {
                continue;
            }
            let corr = pen * stiffness / w_sum;
            particles[i].predicted = add3(pi, scale3(n_hat, corr * wi));
            particles[j].predicted = add3(pj, scale3(n_hat, -corr * wj));
        }
    }
}
/// Attachment constraint: pin a cloth particle to a point on the rigid body.
///
/// Implemented as a stiff spring pulling the particle toward `target`.
pub fn solve_cloth_rigid_attachment(
    particles: &mut [PbdParticle],
    particle_idx: usize,
    target: [f64; 3],
    stiffness: f64,
    dt: f64,
) {
    if particle_idx >= particles.len() {
        return;
    }
    let p = particles[particle_idx].predicted;
    let delta = sub3(target, p);
    let alpha = (1.0 - stiffness) / (dt * dt);
    let w = particles[particle_idx].inv_mass;
    let denom = w + alpha;
    if denom < 1e-30 {
        return;
    }
    let lambda = dot3(delta, [1.0, 1.0, 1.0]) / denom;
    let disp = scale3(delta, stiffness);
    particles[particle_idx].predicted = add3(p, disp);
    let _ = lambda;
}
/// Compute the number of substeps adaptively based on the current maximum
/// particle velocity and a target CFL number.
///
/// Returns a substep count in `[min_substeps, max_substeps]`.
pub fn adaptive_substep_count(
    particles: &[PbdParticle],
    dt: f64,
    target_cfl: f64,
    min_substeps: u32,
    max_substeps: u32,
) -> u32 {
    let max_speed = particles
        .iter()
        .map(|p| len3(p.velocity))
        .fold(0.0_f64, f64::max);
    if max_speed < 1e-12 {
        return min_substeps;
    }
    let needed = (max_speed * dt / target_cfl.max(1e-12)).ceil() as u32;
    needed.clamp(min_substeps, max_substeps)
}
/// Add a (fully inelastic) sphere collision constraint for particle
/// `particle_idx` in a system.
///
/// The particle is kept outside the sphere with the given `center` and `radius`;
/// see [`PbdSystem::add_sphere_constraint`] to choose a restitution.
pub fn add_sphere_collision_constraint(
    system: &mut PbdSystem,
    particle_idx: usize,
    center: [f64; 3],
    radius: f64,
) {
    system.add_sphere_constraint(particle_idx, center, radius, 0.0);
}
/// XPBD global step: solve all constraint groups sequentially (color-safe).
///
/// Within each color group the constraints are independent so could be solved
/// in parallel; here we solve them sequentially for single-threaded safety.
/// Every constraint type is projected by [`project_constraint`], exactly as in
/// [`PbdSystem::step`]; indices of `coloring` that are not constraints of the
/// system are skipped.
pub fn xpbd_step_colored(system: &mut PbdSystem, dt: f64, coloring: &ConstraintColoring) {
    let sub_dt = dt / system.substeps as f64;
    for _ in 0..system.substeps {
        for p in system.particles.iter_mut() {
            if p.fixed {
                p.predicted = p.position;
                continue;
            }
            let v = add3(p.velocity, scale3(system.gravity, sub_dt));
            p.predicted = add3(p.position, scale3(v, sub_dt));
        }
        for group in &coloring.groups {
            for &ci in group {
                if let Some(constraint) = system.constraints.get(ci) {
                    project_constraint(&mut system.particles, constraint, sub_dt);
                }
            }
        }
        for p in system.particles.iter_mut() {
            if p.fixed {
                continue;
            }
            p.velocity = scale3(sub3(p.predicted, p.position), 1.0 / sub_dt);
            p.position = p.predicted;
        }
    }
}
/// Solve distance constraints using the Jacobi style (all corrections computed
/// before any are applied).
pub fn solve_distance_jacobi(
    particles: &mut [PbdParticle],
    constraints: &[PbdConstraint],
    dt: f64,
) {
    let n = particles.len();
    let mut corrections: Vec<[f64; 3]> = vec![[0.0; 3]; n];
    let mut counts = vec![0u32; n];
    for c in constraints {
        if let PbdConstraintType::Distance {
            rest_length,
            compliance,
        } = c.constraint_type
        {
            if c.particles.len() < 2 {
                continue;
            }
            let (i, j) = (c.particles[0], c.particles[1]);
            let wi = particles[i].inv_mass;
            let wj = particles[j].inv_mass;
            let w_sum = wi + wj;
            if w_sum < 1e-30 {
                continue;
            }
            let pi = particles[i].predicted;
            let pj = particles[j].predicted;
            let diff = sub3(pi, pj);
            let cur_len = len3(diff);
            if cur_len < 1e-12 {
                continue;
            }
            let alpha = compliance / (dt * dt);
            let lambda_val = -(cur_len - rest_length) / (w_sum + alpha);
            let n_hat = scale3(diff, 1.0 / cur_len);
            let corr_i = scale3(n_hat, lambda_val * wi);
            let corr_j = scale3(n_hat, -lambda_val * wj);
            for k in 0..3 {
                corrections[i][k] += corr_i[k];
            }
            for k in 0..3 {
                corrections[j][k] += corr_j[k];
            }
            counts[i] += 1;
            counts[j] += 1;
        }
    }
    for i in 0..n {
        if counts[i] > 0 {
            let scale = 1.0 / counts[i] as f64;
            particles[i].predicted = add3(particles[i].predicted, scale3(corrections[i], scale));
        }
    }
}
/// Apply position-level damping by blending each particle's velocity
/// toward the center-of-mass velocity.
///
/// `alpha` in \[0,1\]: 0 = no damping, 1 = full damping to COM velocity.
pub fn apply_position_damping(particles: &mut [PbdParticle], alpha: f64) {
    let n = particles.len();
    if n == 0 {
        return;
    }
    let mut total_m = 0.0;
    let mut com_v = [0.0f64; 3];
    for p in particles.iter() {
        if p.inv_mass < 1e-30 {
            continue;
        }
        let m = 1.0 / p.inv_mass;
        total_m += m;
        for (cv, pv) in com_v.iter_mut().zip(p.velocity.iter()) {
            *cv += m * pv;
        }
    }
    if total_m < 1e-30 {
        return;
    }
    for cv in com_v.iter_mut() {
        *cv /= total_m;
    }
    for p in particles.iter_mut() {
        if p.fixed || p.inv_mass < 1e-30 {
            continue;
        }
        for (pv, cv) in p.velocity.iter_mut().zip(com_v.iter()) {
            *pv = *pv * (1.0 - alpha) + cv * alpha;
        }
    }
}
/// Apply per-axis velocity damping: scale velocity by `(1 - damping_coeff)`.
pub fn apply_velocity_damping(particles: &mut [PbdParticle], damping_coeff: f64) {
    let scale = (1.0 - damping_coeff).max(0.0);
    for p in particles.iter_mut() {
        if p.fixed {
            continue;
        }
        for k in 0..3 {
            p.velocity[k] *= scale;
        }
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    use crate::pbd_system::AdaptivePbdSystem;
    use crate::pbd_system::ProfiledPbdSystem;
    use crate::pbd_system::RigidBodyProxy;
    pub(super) const EPS: f64 = 1e-9;
    #[test]
    fn test_dist3_zero() {
        assert!((dist3([1.0, 2.0, 3.0], [1.0, 2.0, 3.0])).abs() < EPS);
    }
    #[test]
    fn test_dist3_unit_x() {
        assert!((dist3([0.0, 0.0, 0.0], [1.0, 0.0, 0.0]) - 1.0).abs() < EPS);
    }
    #[test]
    fn test_dist3_3d() {
        let d = dist3([0.0, 0.0, 0.0], [1.0, 1.0, 1.0]);
        assert!((d - 3.0_f64.sqrt()).abs() < EPS);
    }
    #[test]
    fn test_sub3() {
        let r = sub3([3.0, 5.0, 7.0], [1.0, 2.0, 3.0]);
        assert_eq!(r, [2.0, 3.0, 4.0]);
    }
    #[test]
    fn test_add3() {
        let r = add3([1.0, 0.0, -1.0], [0.0, 1.0, 2.0]);
        assert_eq!(r, [1.0, 1.0, 1.0]);
    }
    #[test]
    fn test_scale3() {
        let r = scale3([1.0, 2.0, 3.0], 2.0);
        assert_eq!(r, [2.0, 4.0, 6.0]);
    }
    #[test]
    fn test_particle_inv_mass() {
        let p = PbdParticle::new([0.0; 3], 2.0);
        assert!((p.inv_mass - 0.5).abs() < EPS);
    }
    #[test]
    fn test_fixed_particle_zero_inv_mass() {
        let p = PbdParticle::new_fixed([0.0; 3]);
        assert!(p.inv_mass.abs() < EPS);
        assert!(p.fixed);
    }
    #[test]
    fn test_system_particle_count() {
        let mut sys = PbdSystem::new([0.0, -9.81, 0.0], 1);
        sys.add_particle([0.0; 3], 1.0);
        sys.add_particle([1.0, 0.0, 0.0], 1.0);
        assert_eq!(sys.particle_count(), 2);
    }
    #[test]
    fn test_gravity_fall() {
        let mut sys = PbdSystem::new([0.0, -9.81, 0.0], 1);
        sys.add_particle([0.0, 10.0, 0.0], 1.0);
        let dt = 1.0 / 60.0;
        for _ in 0..120 {
            sys.step(dt);
        }
        assert!(
            sys.particles[0].position[1] < 9.0,
            "Particle should fall under gravity"
        );
    }
    #[test]
    fn test_fixed_particle_does_not_move() {
        let mut sys = PbdSystem::new([0.0, -9.81, 0.0], 1);
        sys.add_fixed_particle([0.0, 5.0, 0.0]);
        let dt = 1.0 / 60.0;
        for _ in 0..60 {
            sys.step(dt);
        }
        let y = sys.particles[0].position[1];
        assert!(
            (y - 5.0).abs() < EPS,
            "Fixed particle must not move, got y={y}"
        );
    }
    #[test]
    fn test_distance_constraint_converges() {
        let mut sys = PbdSystem::new([0.0; 3], 20);
        let a = sys.add_particle([0.0, 0.0, 0.0], 1.0);
        let b = sys.add_particle([3.0, 0.0, 0.0], 1.0);
        sys.add_distance_constraint(a, b, 0.0);
        sys.particles[b].position = [6.0, 0.0, 0.0];
        sys.particles[b].predicted = [6.0, 0.0, 0.0];
        let dt = 1.0 / 60.0;
        for _ in 0..200 {
            sys.step(dt);
        }
        let rest = 3.0;
        let d = dist3(sys.particles[a].position, sys.particles[b].position);
        assert!(
            (d - rest).abs() < 0.3,
            "Distance should approach rest={rest}, got {d}"
        );
    }
    #[test]
    fn test_distance_constraint_rest_length_auto() {
        let mut sys = PbdSystem::new([0.0; 3], 1);
        let a = sys.add_particle([0.0, 0.0, 0.0], 1.0);
        let b = sys.add_particle([2.5, 0.0, 0.0], 1.0);
        sys.add_distance_constraint(a, b, 0.0);
        if let PbdConstraintType::Distance { rest_length, .. } = sys.constraints[0].constraint_type
        {
            assert!((rest_length - 2.5).abs() < EPS);
        } else {
            panic!("Expected Distance constraint");
        }
    }
    #[test]
    fn test_floor_constraint_stops_particle() {
        let mut sys = PbdSystem::new([0.0, -9.81, 0.0], 5);
        let i = sys.add_particle([0.0, 5.0, 0.0], 1.0);
        sys.add_floor_constraint(i, 0.0, 0.0);
        let dt = 1.0 / 60.0;
        for _ in 0..300 {
            sys.step(dt);
        }
        assert!(
            sys.particles[i].position[1] >= -0.01,
            "Particle should not fall below floor, got y={}",
            sys.particles[i].position[1]
        );
    }
    #[test]
    fn test_floor_restitution_zero() {
        let mut sys = PbdSystem::new([0.0, -9.81, 0.0], 5);
        let i = sys.add_particle([0.0, 1.0, 0.0], 1.0);
        sys.add_floor_constraint(i, 0.0, 0.0);
        let dt = 1.0 / 60.0;
        for _ in 0..600 {
            sys.step(dt);
        }
        assert!(
            sys.particles[i].position[1] >= -0.05,
            "Particle should rest on floor"
        );
    }
    #[test]
    fn test_kinetic_energy_non_negative() {
        let mut sys = PbdSystem::new([0.0, -9.81, 0.0], 1);
        sys.add_particle([0.0, 5.0, 0.0], 1.0);
        sys.step(1.0 / 60.0);
        assert!(sys.total_kinetic_energy() >= 0.0);
    }
    #[test]
    fn test_kinetic_energy_stationary_zero() {
        let mut sys = PbdSystem::new([0.0; 3], 1);
        sys.add_particle([0.0; 3], 1.0);
        assert!((sys.total_kinetic_energy()).abs() < EPS);
    }
    #[test]
    fn test_tet_volume_unit() {
        let v = compute_tetrahedron_volume(
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        );
        assert!(
            (v - 1.0 / 6.0).abs() < EPS,
            "Unit tet volume should be 1/6, got {v}"
        );
    }
    #[test]
    fn test_tet_volume_degenerate() {
        let v = compute_tetrahedron_volume(
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.5, 0.5, 0.0],
        );
        assert!(v.abs() < EPS, "Degenerate tet volume should be 0, got {v}");
    }
    #[test]
    fn test_build_chain_count() {
        let sys = build_chain(5, 1.0, 1.0, 0.0);
        assert_eq!(sys.particle_count(), 5);
        assert_eq!(sys.constraints.len(), 4);
    }
    #[test]
    fn test_build_chain_top_fixed() {
        let sys = build_chain(4, 1.0, 1.0, 0.0);
        assert!(sys.particles[0].fixed, "Top of chain must be fixed");
        assert!(!sys.particles[1].fixed, "Other chain particles are dynamic");
    }
    #[test]
    fn test_build_grid_cloth_count() {
        let sys = build_grid_cloth(3, 4, 1.0, 1.0, 0.0);
        assert_eq!(sys.particle_count(), 12);
        assert_eq!(sys.constraints.len(), 17);
    }
    #[test]
    fn test_chain_hangs_under_gravity() {
        let mut sys = build_chain(5, 1.0, 1.0, 1e-5);
        let dt = 1.0 / 60.0;
        for _ in 0..300 {
            sys.step(dt);
        }
        let y_initial = -4.0;
        let y_final = sys.particles[4].position[1];
        assert!(
            y_final <= y_initial + 0.1,
            "Bottom chain particle should sag, got y={y_final}"
        );
    }
    #[test]
    fn test_solve_distance_standalone() {
        let mut pts = vec![
            PbdParticle::new([0.0, 0.0, 0.0], 1.0),
            PbdParticle::new([4.0, 0.0, 0.0], 1.0),
        ];
        for _ in 0..100 {
            solve_distance_xpbd(&mut pts, 0, 1, 2.0, 0.0, 1.0 / 60.0);
            for p in pts.iter_mut() {
                p.position = p.predicted;
            }
        }
        let d = dist3(pts[0].position, pts[1].position);
        assert!((d - 2.0).abs() < 0.01, "Standalone solver: d={d}");
    }
    #[test]
    fn test_substep_count_enforced() {
        let sys = PbdSystem::new([0.0; 3], 0);
        assert_eq!(sys.substeps, 1);
    }
    #[test]
    fn test_sphere_collision_pushes_out() {
        let mut p = PbdParticle::new([0.0, 0.0, 0.0], 1.0);
        p.predicted = [0.1, 0.0, 0.0];
        solve_sphere_collision(&mut p, [0.0, 0.0, 0.0], 1.0, 0.0);
        let d = dist3(p.predicted, [0.0, 0.0, 0.0]);
        assert!(
            d >= 1.0 - 1e-9,
            "Particle should be pushed outside sphere, d={d}"
        );
    }
    #[test]
    fn test_sphere_collision_outside_unchanged() {
        let mut p = PbdParticle::new([2.0, 0.0, 0.0], 1.0);
        p.predicted = [2.0, 0.0, 0.0];
        let before = p.predicted;
        solve_sphere_collision(&mut p, [0.0, 0.0, 0.0], 1.0, 0.0);
        let after = p.predicted;
        assert_eq!(before, after, "Outside particle should not be moved");
    }
    #[test]
    fn test_aabb_collision_clamps_particle() {
        let mut p = PbdParticle::new([5.0, 0.0, 0.0], 1.0);
        p.predicted = [5.0, 0.0, 0.0];
        p.velocity = [2.0, 0.0, 0.0];
        solve_aabb_collision(&mut p, [0.0, 0.0, 0.0], [1.0, 1.0, 1.0], 0.0);
        assert!(
            p.predicted[0] <= 1.0 + 1e-9,
            "Particle should be clamped inside AABB"
        );
        assert!(p.velocity[0] <= 0.0 + 1e-9, "Velocity should be reversed");
    }
    #[test]
    fn test_aabb_collision_negative_side() {
        let mut p = PbdParticle::new([-0.5, 0.0, 0.0], 1.0);
        p.predicted = [-0.5, 0.0, 0.0];
        p.velocity = [-1.0, 0.0, 0.0];
        solve_aabb_collision(&mut p, [0.0, 0.0, 0.0], [1.0, 1.0, 1.0], 0.5);
        assert!(
            p.predicted[0] >= 0.0 - 1e-9,
            "Particle should be on +x side of min"
        );
        assert!(p.velocity[0] >= 0.0 - 1e-9, "Velocity should be reflected");
    }
    #[test]
    fn test_triangle_collision_below_plane() {
        let mut p = PbdParticle::new([0.5, -0.5, 0.5], 1.0);
        p.predicted = [0.5, -0.5, 0.5];
        let ta = [0.0, 0.0, 0.0];
        let tb = [0.0, 0.0, 1.0];
        let tc = [1.0, 0.0, 0.0];
        solve_triangle_collision(&mut p, ta, tb, tc, 0.0);
        assert!(
            p.predicted[1] >= -1e-9,
            "Particle should be pushed above triangle plane, got y={}",
            p.predicted[1]
        );
    }
    #[test]
    fn test_self_collision_all_pairs_separation() {
        let mut particles = vec![
            PbdParticle::new([0.0, 0.0, 0.0], 1.0),
            PbdParticle::new([0.05, 0.0, 0.0], 1.0),
        ];
        particles[0].predicted = [0.0, 0.0, 0.0];
        particles[1].predicted = [0.05, 0.0, 0.0];
        solve_self_collision_all_pairs(&mut particles, 0.1, 1.0);
        let d = dist3(particles[0].predicted, particles[1].predicted);
        assert!(
            d >= 0.1 - 1e-9,
            "Particles should be separated to at least min_distance: d={d}"
        );
    }
    #[test]
    fn test_self_collision_grid_separation() {
        let mut particles = vec![
            PbdParticle::new([0.0, 0.0, 0.0], 1.0),
            PbdParticle::new([0.05, 0.0, 0.0], 1.0),
            PbdParticle::new([0.5, 0.5, 0.5], 1.0),
        ];
        for p in &mut particles {
            p.predicted = p.position;
        }
        solve_self_collision_grid(&mut particles, 0.1, 1.0, 0.2);
        let d01 = dist3(particles[0].predicted, particles[1].predicted);
        assert!(d01 >= 0.1 - 1e-9, "Grid self-collision: d01={d01}");
    }
    #[test]
    fn test_rigid_proxy_collision() {
        let proxy = RigidBodyProxy::new([0.0, 0.0, 0.0], 1.0);
        let mut p = PbdParticle::new([0.2, 0.0, 0.0], 1.0);
        p.predicted = [0.2, 0.0, 0.0];
        proxy.apply_collision(&mut p, 0.0);
        let d = dist3(p.predicted, [0.0, 0.0, 0.0]);
        assert!(
            d >= 1.0 - 1e-9,
            "Rigid proxy should push particle outside radius: d={d}"
        );
    }
    #[test]
    fn test_rigid_proxy_step() {
        let mut proxy = RigidBodyProxy::new([0.0, 0.0, 0.0], 1.0);
        proxy.velocity = [1.0, 0.0, 0.0];
        proxy.step(0.1);
        assert!((proxy.center[0] - 0.1).abs() < 1e-12);
    }
    #[test]
    fn test_cloth_rigid_attachment() {
        let mut particles = vec![PbdParticle::new([0.0, 0.0, 0.0], 1.0)];
        particles[0].predicted = [0.0, 0.0, 0.0];
        let target = [2.0, 0.0, 0.0];
        solve_cloth_rigid_attachment(&mut particles, 0, target, 0.5, 0.016);
        let after = particles[0].predicted;
        assert!(
            after[0] > 0.5,
            "Cloth particle should move toward target: x={}",
            after[0]
        );
    }
    #[test]
    fn test_adaptive_substeps_stationary() {
        let particles = vec![PbdParticle::new([0.0; 3], 1.0)];
        let n = adaptive_substep_count(&particles, 1.0 / 60.0, 0.5, 1, 10);
        assert_eq!(n, 1, "Stationary particles should use minimum substeps");
    }
    #[test]
    fn test_adaptive_substeps_fast_particle() {
        let mut p = PbdParticle::new([0.0; 3], 1.0);
        p.velocity = [100.0, 0.0, 0.0];
        let particles = vec![p];
        let n = adaptive_substep_count(&particles, 1.0 / 60.0, 0.1, 1, 32);
        assert!(n > 1, "Fast particle should require more substeps: n={n}");
    }
    #[test]
    fn test_adaptive_substeps_clamped_max() {
        let mut p = PbdParticle::new([0.0; 3], 1.0);
        p.velocity = [1_000_000.0, 0.0, 0.0];
        let particles = vec![p];
        let n = adaptive_substep_count(&particles, 1.0 / 60.0, 0.1, 1, 8);
        assert!(n <= 8, "Substeps should be clamped to max: n={n}");
    }
    #[test]
    fn test_profiled_system_records_stats() {
        let mut sys = ProfiledPbdSystem::new([0.0, -9.81, 0.0], 4);
        sys.system.add_particle([0.0, 5.0, 0.0], 1.0);
        sys.step(1.0 / 60.0);
        assert_eq!(sys.last_profile.substeps, 4);
        assert!(sys.last_profile.total_constraint_solves == 0);
    }
    #[test]
    fn test_profiled_system_constraint_residual() {
        let mut sys = ProfiledPbdSystem::new([0.0; 3], 4);
        let a = sys.system.add_particle([0.0, 0.0, 0.0], 1.0);
        let b = sys.system.add_particle([3.0, 0.0, 0.0], 1.0);
        sys.system.add_distance_constraint(a, b, 0.0);
        sys.step(1.0 / 60.0);
        assert!(
            sys.last_profile.residual >= 0.0,
            "Residual must be non-negative"
        );
    }
    #[test]
    fn test_adaptive_pbd_system_step() {
        let mut sys = AdaptivePbdSystem::new([0.0, -9.81, 0.0], 1, 8);
        sys.system.add_particle([0.0, 10.0, 0.0], 1.0);
        let dt = 1.0 / 60.0;
        for _ in 0..10 {
            sys.step(dt);
        }
        assert!(
            sys.system.particles[0].position[1] < 10.0,
            "Particle should fall under gravity"
        );
    }
}
