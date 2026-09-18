// Copyright 2026 COOLJAPAN OU (Team KitaSan)
// SPDX-License-Identifier: Apache-2.0

//! Strain limiting through [`XpbdBody::step`]: the 100x gravity cloth hang.
//!
//! A 5 x 5 cloth hangs from its pinned top row under 980 m/s^2 acting
//! perpendicular to its rest plane, stepped with the public [`XpbdBody`] API
//! only.  Without a strain limiter the cloth stretches without bound; with the
//! per-triangle limiter enabled on the body, the singular values of every
//! triangle's deformation gradient stay inside the configured range.

use oxiphysics_core::math::Vec3;
use oxiphysics_softbody::StrainLimitConfig;
use oxiphysics_softbody::xpbd::XpbdBody;

const NX: usize = 5;
const NY: usize = 5;
/// Particle spacing of the cloth (m).
const SPACING: f64 = 0.2;
/// 100 x g, along -y.
const GRAVITY: f64 = 980.0;
/// Frame time (s) of one [`XpbdBody::step`] call.
const FRAME: f64 = 1.0 / 60.0;
/// Substeps per frame: 10 substeps of 1/600 s.
const SUBSTEPS: usize = 10;
/// Frames of the hang: 1/3 s in all.
const FRAMES: usize = 20;
/// Constraint-projection iterations per substep.
const CONSTRAINT_ITERATIONS: usize = 4;
/// Compliance (m^2/N) of the distance constraints along every edge of the
/// triangulation: soft enough that the cloth stretches by about 75% under this
/// load without a strain limiter.
const COMPLIANCE: f64 = 1e-4;
const MAX_STRETCH: f64 = 1.05;
const MIN_STRETCH: f64 = 0.95;
/// Gauss-Seidel sweeps of the limiter per substep.  A hanging cloth needs many:
/// the pinned row's influence travels down one row of triangles per sweep, and
/// triangles that share vertices keep undoing each other's corrections.  With
/// this many sweeps the worst stretch of a substep's limiting pass is within
/// about 1e-14 of the limit.
const SWEEPS: u32 = 384;
/// Allowed excess over the limits: the residual of the Gauss-Seidel sweeps, far
/// below what 12 sweeps leave (a few 1e-3).
const TOLERANCE: f64 = 1e-9;

struct Cloth {
    body: XpbdBody,
    rest_positions: Vec<[f64; 3]>,
    triangles: Vec<[usize; 3]>,
}

/// The cloth of the module documentation, in its flat rest shape.  With
/// `compliance` the edges of the triangulation carry distance constraints,
/// without it nothing but a strain limiter can hold the cloth together.
fn hanging_cloth(compliance: Option<f64>) -> Cloth {
    let mut body = XpbdBody::new(NX * NY);
    let mut rest_positions = Vec::with_capacity(NX * NY);
    for row in 0..NY {
        for column in 0..NX {
            let position = Vec3::new(column as f64 * SPACING, 0.0, row as f64 * SPACING);
            let mass = if row == 0 { 0.0 } else { 1.0 };
            body.add_particle(position, mass);
            rest_positions.push([position.x, position.y, position.z]);
        }
    }
    let mut triangles = Vec::new();
    for row in 0..NY - 1 {
        for column in 0..NX - 1 {
            let v00 = row * NX + column;
            let v10 = v00 + 1;
            let v01 = v00 + NX;
            let v11 = v01 + 1;
            triangles.push([v00, v10, v01]);
            triangles.push([v10, v11, v01]);
        }
    }
    if let Some(compliance) = compliance {
        let mut edges = std::collections::BTreeSet::new();
        for &[a, b, c] in &triangles {
            for (p, q) in [(a, b), (b, c), (c, a)] {
                edges.insert((p.min(q), p.max(q)));
            }
        }
        for (a, b) in edges {
            body.add_distance_constraint(a, b, compliance);
        }
    }
    Cloth {
        body,
        rest_positions,
        triangles,
    }
}

fn limiter_config(iterations: u32) -> StrainLimitConfig {
    StrainLimitConfig {
        max_stretch: MAX_STRETCH,
        min_stretch: MIN_STRETCH,
        iterations,
    }
}

/// Step `body` through the whole hang.
fn hang(body: &mut XpbdBody) {
    let gravity = Vec3::new(0.0, -GRAVITY, 0.0);
    for _ in 0..FRAMES {
        body.step(FRAME, gravity, SUBSTEPS, CONSTRAINT_ITERATIONS);
    }
}

fn dot3(a: [f64; 3], b: [f64; 3]) -> f64 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

fn norm3(a: [f64; 3]) -> f64 {
    dot3(a, a).sqrt()
}

fn sub3(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}

fn scale3(a: [f64; 3], s: f64) -> [f64; 3] {
    [a[0] * s, a[1] * s, a[2] * s]
}

/// Singular values `[sigma_max, sigma_min]` of the 3x2 deformation gradient of
/// one triangle, evaluated in world space (out-of-plane motion included):
/// `F = Ds * Dm^-1`, where `Ds = [f0 f1]` holds the current edges as columns and
/// `Dm = [[|e0|, e1.t1], [0, e1.t2]]` the rest edges in an orthonormal frame
/// `(t1, t2)` of the rest plane.
fn singular_values(
    positions: &[[f64; 3]],
    rest_positions: &[[f64; 3]],
    triangle: [usize; 3],
) -> [f64; 2] {
    let [i0, i1, i2] = triangle;
    let rest_edge_0 = sub3(rest_positions[i1], rest_positions[i0]);
    let rest_edge_1 = sub3(rest_positions[i2], rest_positions[i0]);
    let length_0 = norm3(rest_edge_0);
    let t1 = scale3(rest_edge_0, 1.0 / length_0);
    let d12 = dot3(rest_edge_1, t1);
    let d22 = norm3(sub3(rest_edge_1, scale3(t1, d12)));
    let dm_inv = [[1.0 / length_0, -d12 / (length_0 * d22)], [0.0, 1.0 / d22]];
    let edges = [
        sub3(positions[i1], positions[i0]),
        sub3(positions[i2], positions[i0]),
    ];
    // F[row][col] = sum_k Ds[row][k] * Dm^-1[k][col].
    let f: [[f64; 2]; 3] = std::array::from_fn(|row| {
        std::array::from_fn(|col| edges[0][row] * dm_inv[0][col] + edges[1][row] * dm_inv[1][col])
    });
    // Singular values from the eigenvalues of the symmetric 2x2 matrix F^T F.
    let a: f64 = f.iter().map(|row| row[0] * row[0]).sum();
    let b: f64 = f.iter().map(|row| row[0] * row[1]).sum();
    let d: f64 = f.iter().map(|row| row[1] * row[1]).sum();
    let mean = 0.5 * (a + d);
    let radius = (0.25 * (a - d) * (a - d) + b * b).sqrt();
    [
        (mean + radius).max(0.0).sqrt(),
        (mean - radius).max(0.0).sqrt(),
    ]
}

/// Largest and smallest singular value of any triangle of the cloth.
fn stretch_range(cloth: &Cloth) -> (f64, f64) {
    let positions: Vec<[f64; 3]> = cloth
        .body
        .positions
        .iter()
        .map(|p| [p.x, p.y, p.z])
        .collect();
    cloth
        .triangles
        .iter()
        .map(|&triangle| singular_values(&positions, &cloth.rest_positions, triangle))
        .fold((0.0_f64, f64::INFINITY), |(worst, least), [max, min]| {
            (worst.max(max), least.min(min))
        })
}

fn lowest_y(body: &XpbdBody) -> f64 {
    body.positions
        .iter()
        .map(|p| p.y)
        .fold(f64::INFINITY, f64::min)
}

#[test]
fn hang_through_xpbd_body_step_stays_within_the_stretch_limits() {
    // Control: plain XPBD, which is what XpbdBody::step has always done.
    let mut control = hanging_cloth(Some(COMPLIANCE));
    assert!(control.body.strain_limiter.is_none(), "off by default");
    hang(&mut control.body);
    let (control_worst, _) = stretch_range(&control);
    assert!(
        control_worst > 1.5,
        "without a limiter the soft cloth overstretches: worst {control_worst}"
    );

    let mut cloth = hanging_cloth(Some(COMPLIANCE));
    cloth
        .body
        .enable_strain_limiting(cloth.triangles.clone(), limiter_config(SWEEPS))
        .unwrap();
    hang(&mut cloth.body);

    let (worst, least) = stretch_range(&cloth);
    assert!(
        worst <= MAX_STRETCH + TOLERANCE,
        "worst triangle stretch {worst} exceeds {MAX_STRETCH}"
    );
    assert!(
        least >= MIN_STRETCH - TOLERANCE,
        "smallest triangle stretch {least} is below {MIN_STRETCH}"
    );
    // The load was real: the cloth hangs well below its pinned row, only not as
    // far down as the overstretched control.
    let lowest = lowest_y(&cloth.body);
    assert!(
        lowest < -0.5,
        "cloth should hang below its pinned row, lowest y {lowest}"
    );
    assert!(
        lowest > lowest_y(&control.body),
        "the limited cloth hangs higher than the overstretched one"
    );
    for (index, position) in cloth.body.positions.iter().enumerate() {
        assert!(
            position.iter().all(|c| c.is_finite()),
            "non-finite position {index}: {position:?}"
        );
    }
    // The pinned row never moved.
    for (index, rest) in cloth.rest_positions.iter().enumerate().take(NX) {
        let position = cloth.body.positions[index];
        assert_eq!(
            [position.x, position.y, position.z],
            *rest,
            "pinned particle {index} moved"
        );
    }
    // The limiter ran before the velocity update: the velocities of the last
    // substep include its correction.
    let h = FRAME / SUBSTEPS as f64;
    for index in NX..NX * NY {
        let body = &cloth.body;
        let expected = (body.positions[index] - body.prev_positions[index]) / h;
        assert!(
            (body.velocities[index] - expected).norm() < 1e-12,
            "particle {index}: velocity {:?} differs from the position change {expected:?}",
            body.velocities[index]
        );
    }
}

#[test]
fn limiter_alone_holds_a_cloth_without_distance_constraints() {
    let mut control = hanging_cloth(None);
    hang(&mut control.body);
    let (control_worst, _) = stretch_range(&control);
    assert!(
        control_worst > 50.0,
        "with nothing holding it the cloth falls apart: worst {control_worst}"
    );

    let mut cloth = hanging_cloth(None);
    cloth
        .body
        .enable_strain_limiting(cloth.triangles.clone(), limiter_config(SWEEPS))
        .unwrap();
    hang(&mut cloth.body);
    let (worst, least) = stretch_range(&cloth);
    assert!(
        worst <= MAX_STRETCH + TOLERANCE,
        "worst triangle stretch {worst} exceeds {MAX_STRETCH}"
    );
    assert!(
        least >= MIN_STRETCH - TOLERANCE,
        "smallest triangle stretch {least} is below {MIN_STRETCH}"
    );
    assert!(
        lowest_y(&cloth.body) < -0.5,
        "the cloth hangs below the pinned row"
    );
}

#[test]
fn step_with_stats_hang_matches_step_and_counts_the_limiter() {
    const SWEEPS_PER_SUBSTEP: u32 = 24;
    let gravity = Vec3::new(0.0, -GRAVITY, 0.0);
    let mut stepped = hanging_cloth(Some(COMPLIANCE));
    let mut counted = hanging_cloth(Some(COMPLIANCE));
    for cloth in [&mut stepped, &mut counted] {
        cloth
            .body
            .enable_strain_limiting(cloth.triangles.clone(), limiter_config(SWEEPS_PER_SUBSTEP))
            .unwrap();
    }
    let distance_constraints = stepped.body.distance_constraints.len();
    let triangles = stepped.triangles.len();
    for _ in 0..FRAMES {
        stepped
            .body
            .step(FRAME, gravity, SUBSTEPS, CONSTRAINT_ITERATIONS);
        let stats = counted
            .body
            .step_with_stats(FRAME, gravity, SUBSTEPS, CONSTRAINT_ITERATIONS);
        assert_eq!(
            stats.total_projections,
            SUBSTEPS
                * (CONSTRAINT_ITERATIONS * distance_constraints
                    + SWEEPS_PER_SUBSTEP as usize * triangles)
        );
    }
    assert_eq!(stepped.body.positions, counted.body.positions);
    assert_eq!(stepped.body.velocities, counted.body.velocities);
    let (worst, _) = stretch_range(&counted);
    assert!(
        worst < 1.1,
        "even {SWEEPS_PER_SUBSTEP} sweeps keep the hang near the limit: worst {worst}"
    );
}
