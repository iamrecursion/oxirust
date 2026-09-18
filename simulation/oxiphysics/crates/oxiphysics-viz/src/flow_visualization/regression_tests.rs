//! Regression tests for the finite-difference derivatives of [`VectorField`]
//! and the vortex criteria built on them.
//!
//! The clamped central difference must divide by the physical distance between
//! the two neighbours (`2 dx` in the interior, `dx` at the boundary); fields
//! that are linear in the coordinates therefore have exact derivatives at
//! every node.

use super::*;

/// Fill a 2-D field with `f(x, y)` sampled at the node coordinates.
fn field_2d(
    nx: usize,
    ny: usize,
    dx: f64,
    dy: f64,
    f: impl Fn(f64, f64) -> (f64, f64),
) -> VectorField {
    let mut field = VectorField::new_2d(nx, ny, dx, dy, Vec2::zero());
    for j in 0..ny {
        for i in 0..nx {
            let (u, v) = f(i as f64 * dx, j as f64 * dy);
            field.set(i, j, 0, Vec3::new(u, v, 0.0));
        }
    }
    field
}

/// Fill a 3-D field with `f(x, y, z)` sampled at the node coordinates.
fn field_3d(
    dims: (usize, usize, usize),
    spacing: (f64, f64, f64),
    f: impl Fn(f64, f64, f64) -> Vec3,
) -> VectorField {
    let (nx, ny, nz) = dims;
    let (dx, dy, dz) = spacing;
    let mut field = VectorField::new_3d(nx, ny, nz, dx, dy, dz, Vec3::zero());
    for k in 0..nz {
        for j in 0..ny {
            for i in 0..nx {
                field.set(i, j, k, f(i as f64 * dx, j as f64 * dy, k as f64 * dz));
            }
        }
    }
    field
}

#[test]
fn divergence_of_a_linear_expansion_is_two_at_every_node() {
    let field = field_2d(8, 8, 0.5, 0.5, |x, y| (x, y));
    for (idx, d) in field.divergence_2d().iter().enumerate() {
        assert!((d - 2.0).abs() < 1e-12, "node {idx}: divergence {d}");
    }
}

#[test]
fn divergence_uses_the_spacing_of_each_axis() {
    // u = 3x, v = -5y on an anisotropic grid: div = 3 - 5 = -2.
    let field = field_2d(9, 7, 0.5, 0.25, |x, y| (3.0 * x, -5.0 * y));
    for (idx, d) in field.divergence_2d().iter().enumerate() {
        assert!((d + 2.0).abs() < 1e-12, "node {idx}: divergence {d}");
    }
}

#[test]
fn central_difference_is_exact_inside_and_one_sided_on_the_boundary() {
    // u = x^2: interior nodes give (x+dx)^2 - (x-dx)^2 over 2dx = 2x exactly,
    // the boundary nodes the one-sided differences x0 + x1 and x_{n-2} + x_{n-1}.
    let (n, dx) = (8usize, 0.5);
    let field = field_2d(n, 4, dx, dx, |x, _| (x * x, 0.0));
    let div = field.divergence_2d();
    for j in 0..4 {
        for i in 0..n {
            let x = i as f64 * dx;
            let expected = if i == 0 {
                x + (x + dx)
            } else if i == n - 1 {
                x + (x - dx)
            } else {
                2.0 * x
            };
            let d = div[field.index(i, j, 0)];
            assert!((d - expected).abs() < 1e-12, "i = {i}: {d} vs {expected}");
        }
    }
}

#[test]
fn curl_and_vorticity_of_solid_body_rotation_are_two() {
    // u = (-y, x, 0) rotates rigidly with angular velocity 1, so omega = 2.
    let field = field_2d(8, 8, 0.5, 0.5, |x, y| (-y, x));
    for (idx, c) in field.curl_2d().iter().enumerate() {
        assert!((c - 2.0).abs() < 1e-12, "node {idx}: curl {c}");
    }

    let volume = field_3d((8, 8, 4), (0.5, 0.5, 0.5), |x, y, _| Vec3::new(-y, x, 0.0));
    for (idx, w) in volume.vorticity_3d().iter().enumerate() {
        assert!(w.x.abs() < 1e-12 && w.y.abs() < 1e-12, "node {idx}: {w:?}");
        assert!((w.z - 2.0).abs() < 1e-12, "node {idx}: {w:?}");
    }
    for (idx, m) in VorticityVisualization::vorticity_magnitude(&volume)
        .iter()
        .enumerate()
    {
        assert!((m - 2.0).abs() < 1e-12, "node {idx}: |omega| = {m}");
    }
}

#[test]
fn curl_of_rotation_on_an_anisotropic_grid_is_still_two() {
    let field = field_2d(12, 6, 0.5, 0.25, |x, y| (-y, x));
    for (idx, c) in field.curl_2d().iter().enumerate() {
        assert!((c - 2.0).abs() < 1e-12, "node {idx}: curl {c}");
    }
}

#[test]
fn q_criterion_and_swirling_strength_of_solid_body_rotation_are_one() {
    let planar = field_2d(8, 8, 0.5, 0.5, |x, y| (-y, x));
    for (idx, s) in VorticityVisualization::swirling_strength_2d(&planar)
        .iter()
        .enumerate()
    {
        assert!((s - 1.0).abs() < 1e-12, "node {idx}: swirling strength {s}");
    }

    let volume = field_3d((8, 8, 4), (0.5, 0.5, 0.5), |x, y, _| Vec3::new(-y, x, 0.0));
    for (idx, q) in VorticityVisualization::q_criterion(&volume)
        .iter()
        .enumerate()
    {
        assert!((q - 1.0).abs() < 1e-12, "node {idx}: Q = {q}");
    }
}

#[test]
fn pure_strain_has_zero_swirling_strength_and_negative_q() {
    // u = (x, -y, 0): div = 0, S = diag(1, -1, 0), Omega = 0, so Q = -1.
    let planar = field_2d(8, 8, 0.5, 0.5, |x, y| (x, -y));
    for s in VorticityVisualization::swirling_strength_2d(&planar) {
        assert_eq!(s, 0.0);
    }
    let volume = field_3d((6, 6, 3), (0.5, 0.5, 0.5), |x, y, _| Vec3::new(x, -y, 0.0));
    for (idx, q) in VorticityVisualization::q_criterion(&volume)
        .iter()
        .enumerate()
    {
        assert!((q + 1.0).abs() < 1e-12, "node {idx}: Q = {q}");
    }
}

#[test]
fn linear_field_has_exact_vorticity_and_q_on_an_anisotropic_grid() {
    // u = A x with
    //   A = [[ 0, 2, -1],
    //        [ 3, 0,  5],
    //        [-4, 7,  0]].
    // omega = (A32 - A23, A13 - A31, A21 - A12) = (2, 3, 1); the symmetric
    // part has |S|^2 = 97 and the antisymmetric part |Omega|^2 = 7, so
    // Q = (7 - 97) / 2 = -45.
    let field = field_3d((6, 5, 4), (0.5, 0.25, 0.4), |x, y, z| {
        Vec3::new(2.0 * y - z, 3.0 * x + 5.0 * z, -4.0 * x + 7.0 * y)
    });
    for (idx, w) in field.vorticity_3d().iter().enumerate() {
        assert!(
            (w.x - 2.0).abs() < 1e-12 && (w.y - 3.0).abs() < 1e-12 && (w.z - 1.0).abs() < 1e-12,
            "node {idx}: {w:?}"
        );
    }
    for (idx, q) in VorticityVisualization::q_criterion(&field)
        .iter()
        .enumerate()
    {
        assert!((q + 45.0).abs() < 1e-10, "node {idx}: Q = {q}");
    }
}

#[test]
fn single_node_axes_have_zero_derivatives_instead_of_nan() {
    // One column of nodes: nothing to difference along x.
    let column = field_2d(1, 5, 0.5, 0.5, |_, y| (7.0, 2.0 * y));
    let div = column.divergence_2d();
    assert_eq!(div.len(), 5);
    for d in &div {
        assert!((d - 2.0).abs() < 1e-12, "divergence {d}");
    }
    for c in column.curl_2d() {
        assert!(c.is_finite() && c.abs() < 1e-12, "curl {c}");
    }

    // A planar 3-D field (nz = 1) has no z-derivatives.
    let slab = field_3d((4, 4, 1), (0.5, 0.5, 0.5), |x, y, _| Vec3::new(-y, x, 0.0));
    for w in slab.vorticity_3d() {
        assert!(w.x.abs() < 1e-12 && w.y.abs() < 1e-12 && (w.z - 2.0).abs() < 1e-12);
    }
}
