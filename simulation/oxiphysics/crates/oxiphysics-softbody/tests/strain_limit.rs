// Copyright 2026 COOLJAPAN OU (Team KitaSan)
// SPDX-License-Identifier: Apache-2.0

//! Integration tests for SVD-based per-triangle strain limiting.

use oxiphysics_softbody::xpbd::strain_limit::{StrainLimitConfig, apply_strain_limiting};

// ─────────────────────────── helper ──────────────────────────────────────────

fn dot3(a: [f64; 3], b: [f64; 3]) -> f64 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

fn norm3(a: [f64; 3]) -> f64 {
    dot3(a, a).sqrt()
}

fn sub3(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}

fn add3(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [a[0] + b[0], a[1] + b[1], a[2] + b[2]]
}

fn scale3(a: [f64; 3], s: f64) -> [f64; 3] {
    [a[0] * s, a[1] * s, a[2] * s]
}

fn cross3(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

/// Singular values `[sigma_max, sigma_min]` of the 3x2 deformation gradient of
/// one triangle, evaluated in world space (out-of-plane motion included):
/// `F = Ds * Dm^-1`, where `Ds = [f0 f1]` holds the current edges as columns and
/// `Dm = [[|e0|, e1.t1], [0, e1.t2]]` the rest edges in an orthonormal frame
/// `(t1, t2)` of the rest plane.
fn singular_values(
    positions: &[[f64; 3]],
    rest_positions: &[[f64; 3]],
    tri: [usize; 3],
) -> [f64; 2] {
    let [i0, i1, i2] = tri;
    let e0r = sub3(rest_positions[i1], rest_positions[i0]);
    let e1r = sub3(rest_positions[i2], rest_positions[i0]);
    let len0 = norm3(e0r);
    let t1 = scale3(e0r, 1.0 / len0);
    let d12 = dot3(e1r, t1);
    let d22 = norm3(sub3(e1r, scale3(t1, d12)));
    let dm_inv = [[1.0 / len0, -d12 / (len0 * d22)], [0.0, 1.0 / d22]];

    let f0 = sub3(positions[i1], positions[i0]);
    let f1 = sub3(positions[i2], positions[i0]);
    let ds = [f0, f1];
    // F[row][col] = sum_k Ds[row][k] * Dm^-1[k][col].
    let f: [[f64; 2]; 3] = std::array::from_fn(|row| {
        std::array::from_fn(|col| ds[0][row] * dm_inv[0][col] + ds[1][row] * dm_inv[1][col])
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

/// Largest singular value of the deformation gradient of one triangle.
fn max_stretch(positions: &[[f64; 3]], rest_positions: &[[f64; 3]], tri: [usize; 3]) -> f64 {
    singular_values(positions, rest_positions, tri)[0]
}

/// Smallest singular value of the deformation gradient of one triangle.
fn min_stretch(positions: &[[f64; 3]], rest_positions: &[[f64; 3]], tri: [usize; 3]) -> f64 {
    singular_values(positions, rest_positions, tri)[1]
}

/// Rotate `v` about the axis direction `axis` by `angle` radians (Rodrigues).
fn rotate(v: [f64; 3], axis: [f64; 3], angle: f64) -> [f64; 3] {
    let k = scale3(axis, 1.0 / norm3(axis));
    let (sin, cos) = angle.sin_cos();
    add3(
        add3(scale3(v, cos), scale3(cross3(k, v), sin)),
        scale3(k, dot3(k, v) * (1.0 - cos)),
    )
}

/// Apply the linear map `m` (row-major) to `v`.
fn apply_linear(m: [[f64; 3]; 3], v: [f64; 3]) -> [f64; 3] {
    [dot3(m[0], v), dot3(m[1], v), dot3(m[2], v)]
}

fn assert_close(a: [f64; 3], b: [f64; 3], tolerance: f64, what: &str) {
    let error = norm3(sub3(a, b));
    assert!(
        error <= tolerance,
        "{what}: {a:?} differs from {b:?} by {error} (> {tolerance})"
    );
}

/// A scalene rest triangle in a generic plane.
fn skew_rest() -> Vec<[f64; 3]> {
    vec![[0.2, -0.1, 0.4], [1.4, 0.3, 0.1], [0.7, 1.2, 0.6]]
}

/// A generic linear map that stretches the rest plane along one direction and
/// compresses it along another, both well outside `[0.95, 1.05]`.
const SQUASH_AND_STRETCH: [[f64; 3]; 3] = [[1.5, 0.3, -0.2], [0.1, 0.4, 0.3], [-0.3, 0.2, 0.7]];

// ─────────────────────────── tests ───────────────────────────────────────────

/// Equilateral triangle at rest with side length 1 in the XZ plane.
fn equilateral_rest() -> ([[f64; 3]; 3], [[usize; 3]; 1]) {
    let p0 = [0.0_f64, 0.0, 0.0];
    let p1 = [1.0_f64, 0.0, 0.0];
    let p2 = [0.5_f64, 0.0, 3_f64.sqrt() * 0.5];
    ([p0, p1, p2], [[0, 1, 2]])
}

// Test 1: Pre-stretched triangle gets clamped to <= max_stretch.
#[test]
fn test_strain_limiting_clamps_stretch() {
    let (rest, tris) = equilateral_rest();
    let rest_pos: Vec<[f64; 3]> = rest.to_vec();
    // Scale positions by 2x to create 2x stretch.
    let mut positions: Vec<[f64; 3]> = rest_pos
        .iter()
        .map(|p| [p[0] * 2.0, p[1] * 2.0, p[2] * 2.0])
        .collect();
    let inv_masses = vec![1.0_f64, 1.0, 1.0];
    let config = StrainLimitConfig::default();

    // Run several iterations to converge.
    for _ in 0..10 {
        apply_strain_limiting(&mut positions, &rest_pos, &tris, &inv_masses, &config);
    }

    let ms = max_stretch(&positions, &rest_pos, [0, 1, 2]);
    assert!(
        ms <= config.max_stretch + 1e-9,
        "max_stretch after limiting = {ms}, expected <= {}",
        config.max_stretch
    );
    let mn = min_stretch(&positions, &rest_pos, [0, 1, 2]);
    assert!(
        mn >= config.min_stretch - 1e-9,
        "min_stretch after limiting = {mn}, expected >= {}",
        config.min_stretch
    );
}

// Test 2: No-op when max_stretch = infinity (effectively disabled).
#[test]
fn test_strain_limiting_noop_when_disabled() {
    let (rest, tris) = equilateral_rest();
    let rest_pos: Vec<[f64; 3]> = rest.to_vec();
    let mut positions: Vec<[f64; 3]> = rest_pos
        .iter()
        .map(|p| [p[0] * 2.0, p[1] * 2.0, p[2] * 2.0])
        .collect();
    let positions_before = positions.clone();
    let inv_masses = vec![1.0_f64, 1.0, 1.0];

    let config = StrainLimitConfig {
        max_stretch: f64::INFINITY,
        min_stretch: 0.0,
        iterations: 1,
    };
    apply_strain_limiting(&mut positions, &rest_pos, &tris, &inv_masses, &config);

    for (i, (a, b)) in positions.iter().zip(positions_before.iter()).enumerate() {
        for k in 0..3 {
            assert!(
                (a[k] - b[k]).abs() < 1e-12,
                "Vertex {i} coord {k} changed when strain limiting should be disabled"
            );
        }
    }
}

// Test 3: Pinned vertex (inv_mass=0) does not move.
#[test]
fn test_strain_limiting_pinned_vertex_unchanged() {
    let (rest, tris) = equilateral_rest();
    let rest_pos: Vec<[f64; 3]> = rest.to_vec();
    // 3x stretch.
    let mut positions: Vec<[f64; 3]> = rest_pos
        .iter()
        .map(|p| [p[0] * 3.0, p[1] * 3.0, p[2] * 3.0])
        .collect();
    let pin_pos = positions[0];
    let inv_masses = vec![0.0_f64, 1.0, 1.0]; // i0 is pinned.
    let config = StrainLimitConfig {
        iterations: 5,
        ..Default::default()
    };

    apply_strain_limiting(&mut positions, &rest_pos, &tris, &inv_masses, &config);

    for k in 0..3 {
        assert!(
            (positions[0][k] - pin_pos[k]).abs() < 1e-12,
            "Pinned vertex 0 coord {k} changed: was {}, now {}",
            pin_pos[k],
            positions[0][k]
        );
    }
}

// Test 4: Triangle at rest -- no correction applied.
#[test]
fn test_strain_limiting_rest_triangle_noop() {
    let (rest, tris) = equilateral_rest();
    let rest_pos: Vec<[f64; 3]> = rest.to_vec();
    let mut positions = rest_pos.clone();
    let inv_masses = vec![1.0_f64, 1.0, 1.0];
    let config = StrainLimitConfig::default();

    apply_strain_limiting(&mut positions, &rest_pos, &tris, &inv_masses, &config);

    for (i, (a, b)) in positions.iter().zip(rest_pos.iter()).enumerate() {
        for k in 0..3 {
            assert!(
                (a[k] - b[k]).abs() < 1e-12,
                "Rest triangle: vertex {i} coord {k} changed (delta={})",
                (a[k] - b[k]).abs()
            );
        }
    }
}

// Test 5: 100x gravity hang test -- 5x5 grid cloth, gravity=980 m/s^2, 200 substeps.
//
// The cloth starts flat in the XZ plane and gravity acts along -y, perpendicular
// to the rest plane, so the whole fall is out-of-plane motion: a limiter (or a
// stretch measure) that only sees the in-plane projection cannot notice it.
// The limiter's corrections feed back into the velocities (position-based
// velocity update), as in a PBD/XPBD loop.
#[test]
fn test_100x_gravity_cloth_hang_strain_limit() {
    const NX: usize = 5;
    const NY: usize = 5;
    const DX: f64 = 0.2;
    const DY: f64 = 0.2;
    const GRAVITY: f64 = 980.0; // 100x g
    const DT: f64 = 1.0 / 600.0;
    const N_SUBSTEPS: usize = 200;

    let n = NX * NY;
    // Flat cloth in XZ plane: x = i*DX, z = j*DY, y = 0.
    let mut rest_pos: Vec<[f64; 3]> = Vec::with_capacity(n);
    for j in 0..NY {
        for i in 0..NX {
            rest_pos.push([i as f64 * DX, 0.0, j as f64 * DY]);
        }
    }
    let mut positions = rest_pos.clone();
    let mut velocities: Vec<[f64; 3]> = vec![[0.0; 3]; n];

    // Triangulate: for each quad (i,j)->(i+1,j)->(i,j+1) and (i+1,j)->(i+1,j+1)->(i,j+1).
    let mut triangles: Vec<[usize; 3]> = Vec::new();
    for j in 0..NY - 1 {
        for i in 0..NX - 1 {
            let v00 = j * NX + i;
            let v10 = j * NX + i + 1;
            let v01 = (j + 1) * NX + i;
            let v11 = (j + 1) * NX + i + 1;
            triangles.push([v00, v10, v01]);
            triangles.push([v10, v11, v01]);
        }
    }

    // Pin top row (j=0).
    let mut inv_masses: Vec<f64> = vec![1.0; n];
    for w in inv_masses.iter_mut().take(NX) {
        *w = 0.0;
    }

    // Gauss-Seidel carries the pinned row's influence down one row of triangles
    // per sweep, so a 5-row cloth needs a few sweeps per sub-step to settle under
    // this load: 12 = 3 * (NY - 1).  With two sweeps the worst triangle ends at a
    // stretch of about 1.10, with eight at about 1.06 and with twelve at about
    // 1.05 (limit 1.05).
    let config = StrainLimitConfig {
        max_stretch: 1.05,
        min_stretch: 0.95,
        iterations: 12,
    };

    for _ in 0..N_SUBSTEPS {
        let previous = positions.clone();
        // Symplectic Euler: gravity into the velocity, then the velocity into
        // the position.
        for ((position, velocity), &w) in positions
            .iter_mut()
            .zip(velocities.iter_mut())
            .zip(&inv_masses)
        {
            if w > 0.0 {
                velocity[1] -= GRAVITY * DT;
                for (x, v) in position.iter_mut().zip(velocity.iter()) {
                    *x += v * DT;
                }
            }
        }
        apply_strain_limiting(&mut positions, &rest_pos, &triangles, &inv_masses, &config);
        // Position-based velocity update.
        for (((velocity, position), before), &w) in velocities
            .iter_mut()
            .zip(&positions)
            .zip(&previous)
            .zip(&inv_masses)
        {
            if w > 0.0 {
                *velocity = scale3(sub3(*position, *before), 1.0 / DT);
            }
        }
    }

    // Assert no NaN.
    for (idx, p) in positions.iter().enumerate() {
        for (k, &coord) in p.iter().enumerate() {
            assert!(
                coord.is_finite(),
                "NaN/Inf in positions[{idx}][{k}] = {}",
                coord
            );
        }
    }

    // The pinned top row has not moved.
    for (idx, (p, q)) in positions.iter().zip(&rest_pos).enumerate().take(NX) {
        assert_eq!(p, q, "pinned vertex {idx} moved");
    }

    // The cloth really fell out of its rest plane and hangs below the pinned row.
    let lowest = positions.iter().map(|p| p[1]).fold(f64::INFINITY, f64::min);
    assert!(
        lowest < -0.5,
        "cloth should hang below the pinned row, lowest y = {lowest}"
    );

    // Assert the true 3-D stretch of every triangle respects the limit.
    let mut worst = 0.0_f64;
    for &tri in &triangles {
        let s = max_stretch(&positions, &rest_pos, tri);
        if s > worst {
            worst = s;
        }
    }
    assert!(
        worst <= config.max_stretch + 0.01,
        "Worst triangle stretch = {worst}, expected <= {}",
        config.max_stretch
    );
    let least = triangles
        .iter()
        .map(|&tri| min_stretch(&positions, &rest_pos, tri))
        .fold(f64::INFINITY, f64::min);
    assert!(
        least >= config.min_stretch - 0.01,
        "Smallest triangle stretch = {least}, expected >= {}",
        config.min_stretch
    );
}

// Test 6: a rest triangle rigidly rotated by 90 degrees about its first edge
// leaves the rest plane completely, yet it is not strained at all.  The old
// limiter projected the current edges onto the rest plane, saw a collapsed
// triangle (sigma_2 = 0) and dragged it back into the plane.
#[test]
fn test_rest_triangle_rotated_90_degrees_about_its_first_edge_is_unchanged() {
    let (rest, tris) = equilateral_rest();
    let rest_pos: Vec<[f64; 3]> = rest.to_vec();
    let first_edge = sub3(rest_pos[1], rest_pos[0]);
    let rotated: Vec<[f64; 3]> = rest_pos
        .iter()
        .map(|&p| {
            rotate(
                sub3(p, rest_pos[0]),
                first_edge,
                std::f64::consts::FRAC_PI_2,
            )
        })
        .map(|p| add3(p, rest_pos[0]))
        .collect();
    // The apex really left the rest plane (y was 0).
    assert!(rotated[2][1].abs() > 0.8, "apex = {:?}", rotated[2]);

    for inv_masses in [[1.0, 1.0, 1.0], [0.0, 1.0, 1.0], [0.0, 0.0, 1.0]] {
        let mut positions = rotated.clone();
        apply_strain_limiting(
            &mut positions,
            &rest_pos,
            &tris,
            &inv_masses,
            &StrainLimitConfig::default(),
        );
        for (i, (a, b)) in positions.iter().zip(&rotated).enumerate() {
            assert_close(
                *a,
                *b,
                1e-12,
                &format!("vertex {i}, inv masses {inv_masses:?}"),
            );
        }
    }
}

// Test 7: any rigid motion leaves both singular values at 1, so even a razor
// thin admissible range does not move the triangle.
#[test]
fn test_rigid_motions_of_any_kind_are_left_alone() {
    let rest_pos = skew_rest();
    let tris = [[0usize, 1, 2]];
    let inv_masses = [1.0, 2.0, 0.5];
    let thin = StrainLimitConfig {
        max_stretch: 1.0 + 1e-9,
        min_stretch: 1.0 - 1e-9,
        iterations: 3,
    };
    let motions = [
        (
            [1.0, 0.0, 0.0],
            std::f64::consts::FRAC_PI_2,
            [0.0, 0.0, 0.0],
        ),
        ([1.0, 0.0, 0.0], std::f64::consts::PI, [0.0, 0.0, 0.0]),
        ([1.0, 1.0, 1.0], 0.3, [2.0, -1.0, 4.0]),
        ([0.0, 1.0, 0.0], 2.5, [-10.0, 0.5, 3.0]),
        ([1.0, -2.0, 0.5], -1.1, [0.0, 0.0, 0.0]),
    ];
    for (axis, angle, shift) in motions {
        let moved: Vec<[f64; 3]> = rest_pos
            .iter()
            .map(|&p| add3(rotate(p, axis, angle), shift))
            .collect();
        for config in [StrainLimitConfig::default(), thin.clone()] {
            let mut positions = moved.clone();
            apply_strain_limiting(&mut positions, &rest_pos, &tris, &inv_masses, &config);
            for (i, (a, b)) in positions.iter().zip(&moved).enumerate() {
                assert_close(*a, *b, 1e-12, &format!("vertex {i}, axis {axis:?}"));
            }
        }
    }
}

// Test 8: the limiter is frame indifferent -- limiting a rigidly moved copy of a
// strained triangle gives the rigidly moved copy of the limited triangle.
#[test]
fn test_limiting_commutes_with_rigid_motions() {
    let rest_pos = skew_rest();
    let tris = [[0usize, 1, 2]];
    let inv_masses = [1.0, 2.0, 0.5];
    let strained: Vec<[f64; 3]> = rest_pos
        .iter()
        .map(|&p| apply_linear(SQUASH_AND_STRETCH, p))
        .collect();
    let sigma = singular_values(&strained, &rest_pos, [0, 1, 2]);
    assert!(sigma[0] > 1.3 && sigma[1] < 0.8, "sigma = {sigma:?}");

    let config = StrainLimitConfig::default();
    let mut limited = strained.clone();
    apply_strain_limiting(&mut limited, &rest_pos, &tris, &inv_masses, &config);
    assert!(
        limited
            .iter()
            .zip(&strained)
            .any(|(a, b)| norm3(sub3(*a, *b)) > 0.05),
        "the strained triangle must be corrected"
    );

    let axis = [1.0, 2.0, -0.5];
    let angle = 0.9;
    let shift = [3.0, -2.0, 5.0];
    let rigid = |p: [f64; 3]| add3(rotate(p, axis, angle), shift);
    let mut moved: Vec<[f64; 3]> = strained.iter().map(|&p| rigid(p)).collect();
    apply_strain_limiting(&mut moved, &rest_pos, &tris, &inv_masses, &config);
    for (i, (a, b)) in moved.iter().zip(&limited).enumerate() {
        assert_close(*a, rigid(*b), 1e-12, &format!("vertex {i}"));
    }
}

// Test 9: out-of-plane stretch.  The rest triangle lies in the XZ plane; the
// current one is rotated into the XY plane and stretched by 2.  The 3-D
// strain is 2, so the limiter must scale it back to 1.05 within its own plane.
#[test]
fn test_out_of_plane_stretch_is_limited_within_the_rotated_plane() {
    let (rest, tris) = equilateral_rest();
    let rest_pos: Vec<[f64; 3]> = rest.to_vec();
    let mut positions: Vec<[f64; 3]> = rest_pos
        .iter()
        .map(|&p| scale3(rotate(p, [1.0, 0.0, 0.0], std::f64::consts::FRAC_PI_2), 2.0))
        .collect();
    let centroid_before = scale3(
        positions.iter().fold([0.0; 3], |sum, &p| add3(sum, p)),
        1.0 / 3.0,
    );
    let sigma = singular_values(&positions, &rest_pos, [0, 1, 2]);
    assert!((sigma[0] - 2.0).abs() < 1e-12 && (sigma[1] - 2.0).abs() < 1e-12);

    let config = StrainLimitConfig::default();
    apply_strain_limiting(&mut positions, &rest_pos, &tris, &[1.0; 3], &config);

    let after = singular_values(&positions, &rest_pos, [0, 1, 2]);
    assert!(
        (after[0] - config.max_stretch).abs() < 1e-9
            && (after[1] - config.max_stretch).abs() < 1e-9,
        "singular values after limiting: {after:?}"
    );
    // The triangle stays in its rotated (XY) plane, and with equal masses it
    // shrinks about its centroid.
    for p in &positions {
        assert!(p[2].abs() < 1e-12, "left the XY plane: {p:?}");
    }
    let centroid_after = scale3(
        positions.iter().fold([0.0; 3], |sum, &p| add3(sum, p)),
        1.0 / 3.0,
    );
    assert_close(centroid_after, centroid_before, 1e-12, "centroid");
}

// Test 10: with two or three free vertices one sweep meets the clamped shape
// exactly, whatever the masses and however the triangle is deformed in 3-D.
#[test]
fn test_one_sweep_reaches_the_limits_when_two_vertices_are_free() {
    let rest_pos = skew_rest();
    let tris = [[0usize, 1, 2]];
    let config = StrainLimitConfig::default();
    let strained: Vec<[f64; 3]> = rest_pos
        .iter()
        .map(|&p| apply_linear(SQUASH_AND_STRETCH, p))
        .collect();
    let before = singular_values(&strained, &rest_pos, [0, 1, 2]);
    assert!(before[0] > 1.3 && before[1] < 0.8, "sigma = {before:?}");

    for inv_masses in [
        [1.0, 1.0, 1.0],
        [0.5, 2.0, 4.0],
        [0.0, 1.0, 1.0],
        [1.0, 0.0, 1.0],
        [1.0, 1.0, 0.0],
    ] {
        let mut positions = strained.clone();
        apply_strain_limiting(&mut positions, &rest_pos, &tris, &inv_masses, &config);
        let after = singular_values(&positions, &rest_pos, [0, 1, 2]);
        assert!(
            (after[0] - config.max_stretch).abs() < 1e-9
                && (after[1] - config.min_stretch).abs() < 1e-9,
            "inv masses {inv_masses:?}: sigma {after:?}"
        );
        for ((p, q), &w) in positions.iter().zip(&strained).zip(&inv_masses) {
            if w == 0.0 {
                assert_eq!(p, q, "pinned vertex moved for {inv_masses:?}");
            }
        }
    }
}

// Test 11: the corrections conserve linear momentum, sum(m_i dp_i) = 0.
#[test]
fn test_corrections_conserve_linear_momentum() {
    let rest_pos = skew_rest();
    let tris = [[0usize, 1, 2]];
    let masses = [1.0, 2.0, 0.5];
    let inv_masses = masses.map(|m| 1.0 / m);
    let strained: Vec<[f64; 3]> = rest_pos
        .iter()
        .map(|&p| apply_linear(SQUASH_AND_STRETCH, p))
        .collect();
    let mut positions = strained.clone();
    apply_strain_limiting(
        &mut positions,
        &rest_pos,
        &tris,
        &inv_masses,
        &StrainLimitConfig::default(),
    );
    let impulse = positions
        .iter()
        .zip(&strained)
        .zip(&masses)
        .fold([0.0; 3], |sum, ((after, before), m)| {
            add3(sum, scale3(sub3(*after, *before), *m))
        });
    let moved = positions
        .iter()
        .zip(&strained)
        .map(|(a, b)| norm3(sub3(*a, *b)))
        .fold(0.0_f64, f64::max);
    assert!(moved > 0.05, "the limiter must act (moved {moved})");
    assert!(norm3(impulse) < 1e-12, "net impulse {impulse:?}");
}

// Test 12: compression is limited from below, in 3-D as well.
#[test]
fn test_compression_is_limited_to_min_stretch() {
    let rest_pos = skew_rest();
    let tris = [[0usize, 1, 2]];
    let centre = scale3(
        rest_pos.iter().fold([0.0; 3], |sum, &p| add3(sum, p)),
        1.0 / 3.0,
    );
    // Shrink to half size about the centroid and tilt out of the rest plane.
    let mut positions: Vec<[f64; 3]> = rest_pos
        .iter()
        .map(|&p| {
            add3(
                rotate(scale3(sub3(p, centre), 0.5), [1.0, 0.2, 0.0], 0.7),
                centre,
            )
        })
        .collect();
    let config = StrainLimitConfig::default();
    assert!(max_stretch(&positions, &rest_pos, [0, 1, 2]) < 0.51);
    apply_strain_limiting(&mut positions, &rest_pos, &tris, &[1.0; 3], &config);
    let sigma = singular_values(&positions, &rest_pos, [0, 1, 2]);
    assert!(
        (sigma[0] - config.min_stretch).abs() < 1e-9
            && (sigma[1] - config.min_stretch).abs() < 1e-9,
        "sigma = {sigma:?}"
    );
}

// Test 13: a triangle with two pinned vertices can only move the free one.
#[test]
fn test_two_pinned_vertices_only_the_free_one_moves() {
    let (rest, tris) = equilateral_rest();
    let rest_pos: Vec<[f64; 3]> = rest.to_vec();
    // Stretch the second edge to 1.6 times its length, out of the rest plane.
    let mut positions = rest_pos.clone();
    positions[2] = add3(
        rest_pos[0],
        scale3(
            rotate(sub3(rest_pos[2], rest_pos[0]), [1.0, 0.0, 0.0], 0.8),
            1.6,
        ),
    );
    let start = positions.clone();
    let config = StrainLimitConfig::default();
    apply_strain_limiting(&mut positions, &rest_pos, &tris, &[0.0, 0.0, 1.0], &config);
    assert_eq!(positions[0], start[0]);
    assert_eq!(positions[1], start[1]);
    assert!(norm3(sub3(positions[2], start[2])) > 0.1);
    // The free vertex takes the whole change of its own edge, so that edge
    // ends up inside the admissible range (the other edge cannot change).
    let edge_ratio =
        norm3(sub3(positions[2], positions[0])) / norm3(sub3(rest_pos[2], rest_pos[0]));
    assert!(
        edge_ratio >= config.min_stretch - 1e-9 && edge_ratio <= config.max_stretch + 1e-9,
        "edge ratio {edge_ratio}"
    );
}

// Test 14: a collapsed triangle (a point, or a segment) is re-expanded with its
// rest orientation instead of producing NaNs or flipping.
#[test]
fn test_collapsed_triangles_regain_their_rest_orientation() {
    let (rest, tris) = equilateral_rest();
    let rest_pos: Vec<[f64; 3]> = rest.to_vec();
    let rest_normal = cross3(
        sub3(rest_pos[1], rest_pos[0]),
        sub3(rest_pos[2], rest_pos[0]),
    );
    let config = StrainLimitConfig::default();
    for collapsed in [
        // A point.
        vec![[0.3, -0.2, 0.1]; 3],
        // Segments along the first rest edge, along the rest plane's other
        // axis, and along a direction that is neither in nor normal to it.
        vec![[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.5, 0.0, 0.0]],
        vec![[0.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, 0.0, 0.4]],
        vec![[0.0, 0.0, 0.0], [1.0, 1.0, 1.0], [0.4, 0.4, 0.4]],
    ] {
        let mut positions = collapsed.clone();
        apply_strain_limiting(&mut positions, &rest_pos, &tris, &[1.0; 3], &config);
        for p in &positions {
            assert!(p.iter().all(|c| c.is_finite()), "non-finite {p:?}");
        }
        let sigma = singular_values(&positions, &rest_pos, [0, 1, 2]);
        assert!(
            sigma[0] <= config.max_stretch + 1e-9 && sigma[1] >= config.min_stretch - 1e-9,
            "collapsed {collapsed:?} -> sigma {sigma:?}"
        );
        let normal = cross3(
            sub3(positions[1], positions[0]),
            sub3(positions[2], positions[0]),
        );
        assert!(
            dot3(normal, rest_normal) > 0.0,
            "orientation flipped for {collapsed:?}: normal {normal:?}"
        );
    }
}

// Test 15: the degenerate-triangle guard is relative, so micrometre-sized
// meshes are limited like any other (the old absolute area threshold skipped
// them).
#[test]
fn test_tiny_triangles_are_limited_too() {
    let (rest, tris) = equilateral_rest();
    let scale = 1e-6;
    let rest_pos: Vec<[f64; 3]> = rest.iter().map(|&p| scale3(p, scale)).collect();
    let mut positions: Vec<[f64; 3]> = rest_pos.iter().map(|&p| scale3(p, 2.0)).collect();
    let config = StrainLimitConfig::default();
    apply_strain_limiting(&mut positions, &rest_pos, &tris, &[1.0; 3], &config);
    let sigma = singular_values(&positions, &rest_pos, [0, 1, 2]);
    assert!(
        (sigma[0] - config.max_stretch).abs() < 1e-9
            && (sigma[1] - config.max_stretch).abs() < 1e-9,
        "sigma = {sigma:?}"
    );
}

// Test 16: bad configurations must not panic (`f64::clamp` panics on an
// inverted or NaN range).
#[test]
fn test_invalid_configurations_do_not_panic() {
    let (rest, tris) = equilateral_rest();
    let rest_pos: Vec<[f64; 3]> = rest.to_vec();
    let stretched: Vec<[f64; 3]> = rest_pos.iter().map(|&p| scale3(p, 2.0)).collect();

    // Inverted range: the upper bound wins, so the triangle is driven to sigma 1.
    let mut positions = stretched.clone();
    let inverted = StrainLimitConfig {
        max_stretch: 1.0,
        min_stretch: 2.0,
        iterations: 1,
    };
    apply_strain_limiting(&mut positions, &rest_pos, &tris, &[1.0; 3], &inverted);
    let sigma = singular_values(&positions, &rest_pos, [0, 1, 2]);
    assert!((sigma[0] - 1.0).abs() < 1e-9 && (sigma[1] - 1.0).abs() < 1e-9);

    // NaN bounds do not limit anything.
    let mut positions = stretched.clone();
    let not_a_number = StrainLimitConfig {
        max_stretch: f64::NAN,
        min_stretch: f64::NAN,
        iterations: 2,
    };
    apply_strain_limiting(&mut positions, &rest_pos, &tris, &[1.0; 3], &not_a_number);
    assert_eq!(positions, stretched);

    // Zero sweeps do nothing.
    let mut positions = stretched.clone();
    let none = StrainLimitConfig {
        iterations: 0,
        ..Default::default()
    };
    apply_strain_limiting(&mut positions, &rest_pos, &tris, &[1.0; 3], &none);
    assert_eq!(positions, stretched);
}

// Test 17: degenerate rest triangles and non-finite positions are skipped and
// never poison their neighbours.
#[test]
fn test_degenerate_input_is_skipped() {
    // Collinear rest triangle.
    let rest_pos = vec![[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]];
    let stretched = vec![[0.0, 0.0, 0.0], [3.0, 0.0, 0.0], [6.0, 1.0, 0.0]];
    let mut positions = stretched.clone();
    apply_strain_limiting(
        &mut positions,
        &rest_pos,
        &[[0, 1, 2]],
        &[1.0; 3],
        &StrainLimitConfig::default(),
    );
    assert_eq!(positions, stretched);

    // A NaN coordinate: that triangle is skipped, the healthy one is limited.
    let (rest, _) = equilateral_rest();
    let mut rest_pos: Vec<[f64; 3]> = rest.to_vec();
    rest_pos.push([1.5, 0.0, 0.866_025_403_784_438_6]);
    let mut positions: Vec<[f64; 3]> = rest_pos.iter().map(|&p| scale3(p, 2.0)).collect();
    positions[3] = [f64::NAN, 0.0, 0.0];
    let before = positions.clone();
    let triangles = [[0usize, 1, 3], [0, 1, 2]];
    apply_strain_limiting(
        &mut positions,
        &rest_pos,
        &triangles,
        &[1.0; 4],
        &StrainLimitConfig::default(),
    );
    assert!(positions[3][0].is_nan());
    assert_eq!(positions[3][1..], before[3][1..]);
    for p in &positions[..3] {
        assert!(p.iter().all(|c| c.is_finite()), "non-finite {p:?}");
    }
    let sigma = singular_values(&positions, &rest_pos, [0, 1, 2]);
    assert!(
        sigma[0] <= 1.05 + 1e-9,
        "healthy triangle not limited: {sigma:?}"
    );
}

// Test 18: Gauss-Seidel sweeps over triangles that share vertices converge to a
// mesh that respects the limits.
#[test]
fn test_shared_vertices_converge_over_several_sweeps() {
    let rest_pos = vec![
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0],
        [1.0, 0.0, 1.0],
    ];
    let triangles = [[0usize, 1, 2], [1, 3, 2]];
    let mut positions: Vec<[f64; 3]> = rest_pos
        .iter()
        .map(|&p| scale3(rotate(p, [0.3, 1.0, 0.1], 0.4), 1.5))
        .collect();
    let config = StrainLimitConfig {
        iterations: 300,
        ..Default::default()
    };
    apply_strain_limiting(&mut positions, &rest_pos, &triangles, &[1.0; 4], &config);
    for &tri in &triangles {
        let sigma = singular_values(&positions, &rest_pos, tri);
        assert!(
            sigma[0] <= config.max_stretch + 1e-3 && sigma[1] >= config.min_stretch - 1e-3,
            "triangle {tri:?}: sigma {sigma:?}"
        );
    }
    // min_stretch is exercised too: no triangle collapsed.
    assert!(
        triangles
            .iter()
            .all(|&tri| min_stretch(&positions, &rest_pos, tri) > 0.9)
    );
}

// Test 19: a projected triangle is a fixed point -- limiting it again does not
// move it (the rounding noise of the first projection is below the threshold).
#[test]
fn test_projected_triangle_is_a_fixed_point() {
    let rest_pos = skew_rest();
    let tris = [[0usize, 1, 2]];
    let inv_masses = [1.0, 2.0, 0.5];
    let config = StrainLimitConfig::default();
    let mut positions: Vec<[f64; 3]> = rest_pos
        .iter()
        .map(|&p| apply_linear(SQUASH_AND_STRETCH, p))
        .collect();
    apply_strain_limiting(&mut positions, &rest_pos, &tris, &inv_masses, &config);
    let limited = positions.clone();
    apply_strain_limiting(&mut positions, &rest_pos, &tris, &inv_masses, &config);
    for (i, (a, b)) in positions.iter().zip(&limited).enumerate() {
        assert_close(*a, *b, 1e-13, &format!("vertex {i}"));
    }
}

/// Deterministic xorshift64* generator for the randomized test.
struct Rng(u64);

impl Rng {
    /// Uniform in `[0, 1)`.
    fn unit(&mut self) -> f64 {
        self.0 ^= self.0 >> 12;
        self.0 ^= self.0 << 25;
        self.0 ^= self.0 >> 27;
        let bits = self.0.wrapping_mul(0x2545_F491_4F6C_DD1D);
        (bits >> 11) as f64 / (1_u64 << 53) as f64
    }

    /// Uniform in `[lo, hi)`.
    fn range(&mut self, lo: f64, hi: f64) -> f64 {
        lo + (hi - lo) * self.unit()
    }

    fn vector(&mut self, lo: f64, hi: f64) -> [f64; 3] {
        [self.range(lo, hi), self.range(lo, hi), self.range(lo, hi)]
    }
}

// Test 20: randomized triangles, deformations, masses and pins.  With two or
// three free vertices one sweep must land inside the limits; pinned vertices
// never move; a triangle without pins gets no net momentum; and rigidly moving
// the input rigidly moves the output.
#[test]
fn test_random_triangles_meet_the_contract() {
    let mut rng = Rng(0x2545_F491_4F6C_DD1D);
    let config = StrainLimitConfig::default();
    let tris = [[0usize, 1, 2]];
    let mut checked = 0;
    while checked < 600 {
        let rest_pos: Vec<[f64; 3]> = (0..3).map(|_| rng.vector(-1.0, 1.0)).collect();
        let e0 = sub3(rest_pos[1], rest_pos[0]);
        let e1 = sub3(rest_pos[2], rest_pos[0]);
        let area = 0.5 * norm3(cross3(e0, e1));
        let longest = norm3(e0)
            .max(norm3(e1))
            .max(norm3(sub3(rest_pos[2], rest_pos[1])));
        if area < 0.05 || longest > 2.5 {
            continue;
        }
        // A random affine deformation, possibly collapsing or inflating.
        let map = [
            rng.vector(-1.6, 1.6),
            rng.vector(-1.6, 1.6),
            rng.vector(-1.6, 1.6),
        ];
        let shift = rng.vector(-1.0, 1.0);
        let strained: Vec<[f64; 3]> = rest_pos
            .iter()
            .map(|&p| add3(apply_linear(map, p), shift))
            .collect();
        let masses: Vec<f64> = (0..3)
            .map(|_| {
                if rng.unit() < 0.3 {
                    0.0
                } else {
                    rng.range(0.1, 4.0)
                }
            })
            .collect();
        let free = masses.iter().filter(|&&w| w > 0.0).count();

        let mut positions = strained.clone();
        apply_strain_limiting(&mut positions, &rest_pos, &tris, &masses, &config);
        checked += 1;

        for p in &positions {
            assert!(p.iter().all(|c| c.is_finite()), "non-finite {p:?}");
        }
        for ((after, before), &w) in positions.iter().zip(&strained).zip(&masses) {
            if w == 0.0 {
                assert_eq!(after, before, "pinned vertex moved");
            }
        }
        let sigma_before = singular_values(&strained, &rest_pos, [0, 1, 2]);
        let sigma_after = singular_values(&positions, &rest_pos, [0, 1, 2]);
        if free >= 2 {
            assert!(
                sigma_after[0] <= config.max_stretch + 1e-7
                    && sigma_after[1] >= config.min_stretch - 1e-7,
                "free vertices {free}: {sigma_before:?} -> {sigma_after:?}"
            );
        }
        if free == 3 {
            let impulse = positions
                .iter()
                .zip(&strained)
                .zip(&masses)
                .fold([0.0; 3], |sum, ((after, before), &w)| {
                    add3(sum, scale3(sub3(*after, *before), 1.0 / w))
                });
            let moved = positions
                .iter()
                .zip(&strained)
                .map(|(a, b)| norm3(sub3(*a, *b)))
                .fold(0.0_f64, f64::max);
            assert!(
                norm3(impulse) <= 1e-9 * (1.0 + moved),
                "net impulse {impulse:?} for moves of {moved}"
            );
        }

        // Frame indifference.
        let axis = rng.vector(-1.0, 1.0);
        if norm3(axis) < 0.1 {
            continue;
        }
        let angle = rng.range(-3.0, 3.0);
        let offset = rng.vector(-2.0, 2.0);
        let rigid = |p: [f64; 3]| add3(rotate(p, axis, angle), offset);
        let mut moved_positions: Vec<[f64; 3]> = strained.iter().map(|&p| rigid(p)).collect();
        apply_strain_limiting(&mut moved_positions, &rest_pos, &tris, &masses, &config);
        for (i, (a, b)) in moved_positions.iter().zip(&positions).enumerate() {
            assert_close(
                *a,
                rigid(*b),
                1e-8,
                &format!("vertex {i}, frame indifference"),
            );
        }
    }
}
