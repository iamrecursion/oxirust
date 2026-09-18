// Copyright 2026 COOLJAPAN OU (Team KitaSan)
// SPDX-License-Identifier: Apache-2.0

//! Regression tests for the voxel collision audit findings: fast sweeping
//! (F0302), mesh voxelization (F0301) and voxel ray casting (F0303), together
//! with the grid sampling changes they rely on.

use super::fast_sweeping::godunov_update;
use super::mesh_voxelizer::PreparedMesh;
use super::*;
use crate::Error;

fn approx(a: f64, b: f64, tol: f64) -> bool {
    (a - b).abs() <= tol
}

fn flat(nx: usize, ny: usize, ix: usize, iy: usize, iz: usize) -> usize {
    ix + nx * (iy + ny * iz)
}

// ─────────────────────────────────────────────────────────────────────────────
// FastSweeping (F0302)
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn godunov_two_axis_update_is_the_root_of_the_quadratic() {
    let h = 1.0;
    // (u - 0)^2 + (u - 0)^2 = h^2  =>  u = h / sqrt(2); the old formula gave 0.866 h.
    let u = godunov_update([0.0, 0.0, f64::INFINITY], h);
    assert!(approx(u, 1.0 / 2.0_f64.sqrt(), 1e-12), "u = {u}");
    // The input order does not matter.
    let u = godunov_update([f64::INFINITY, 0.0, 0.0], h);
    assert!(approx(u, 1.0 / 2.0_f64.sqrt(), 1e-12), "u = {u}");
    // a = 0, b = h: the one-axis value h already satisfies u <= b (old: 1.5 h).
    let u = godunov_update([0.0, h, f64::INFINITY], h);
    assert!(approx(u, h, 1e-12), "u = {u}");
    // a = 0, b = 1.5 h: still one axis only.
    let u = godunov_update([1.5 * h, 0.0, f64::INFINITY], h);
    assert!(approx(u, h, 1e-12), "u = {u}");
}

#[test]
fn godunov_three_axis_update_is_the_root_of_the_quadratic() {
    // 3 u^2 = h^2  =>  u = h / sqrt(3); the old discriminant gave a different value.
    let u = godunov_update([0.0, 0.0, 0.0], 1.0);
    assert!(approx(u, 1.0 / 3.0_f64.sqrt(), 1e-12), "u = {u}");
    let u = godunov_update([0.0, 0.0, 0.0], 0.25);
    assert!(approx(u, 0.25 / 3.0_f64.sqrt(), 1e-12), "u = {u}");
}

#[test]
fn godunov_update_solves_the_active_eikonal_equation() {
    let h = 0.7;
    let cases: [[f64; 3]; 5] = [
        [0.0, 0.3, 0.4],
        [1.0, 1.2, 1.1],
        [2.0, 2.0, 2.2],
        [0.0, 0.5, 5.0],
        [3.0, 3.1, f64::INFINITY],
    ];
    for case in cases {
        let u = godunov_update(case, h);
        let mut sorted = case;
        sorted.sort_by(f64::total_cmp);
        // Residual of the equation over the axes that are upwind of `u`.
        let residual: f64 = sorted
            .iter()
            .filter(|v| v.is_finite() && **v < u)
            .map(|v| (u - v) * (u - v))
            .sum();
        assert!(
            approx(residual, h * h, 1e-12),
            "case {case:?}: u = {u}, residual = {residual}"
        );
        // Causality: the update never lies below the smallest neighbour.
        assert!(u >= sorted[0], "case {case:?}");
    }
}

#[test]
fn godunov_update_without_known_neighbours_stays_infinite() {
    assert!(godunov_update([f64::INFINITY; 3], 1.0).is_infinite());
}

#[test]
fn fast_sweeping_diagonal_voxel_uses_the_two_axis_update() {
    let grid = SdfVoxelGrid::new([0.0; 3], 1.0, 2, 2, 1);
    let seeds = [
        (VoxelCoord::new(1, 0, 0), 0.0),
        (VoxelCoord::new(0, 1, 0), 0.0),
    ];
    let phi = FastSweeping::compute(&grid, &seeds);
    let corner = phi[flat(2, 2, 1, 1, 0)];
    assert!(
        approx(corner, 1.0 / 2.0_f64.sqrt(), 1e-12),
        "corner = {corner}"
    );
}

#[test]
fn fast_sweeping_corner_voxel_uses_the_three_axis_update() {
    let grid = SdfVoxelGrid::new([0.0; 3], 0.5, 2, 2, 2);
    let seeds = [
        (VoxelCoord::new(0, 1, 1), 0.0),
        (VoxelCoord::new(1, 0, 1), 0.0),
        (VoxelCoord::new(1, 1, 0), 0.0),
    ];
    let phi = FastSweeping::compute(&grid, &seeds);
    let corner = phi[flat(2, 2, 1, 1, 1)];
    assert!(
        approx(corner, 0.5 / 3.0_f64.sqrt(), 1e-12),
        "corner = {corner}"
    );
}

#[test]
fn fast_sweeping_falls_back_to_one_axis_when_the_other_is_far() {
    let h = 0.5;
    let grid = SdfVoxelGrid::new([0.0; 3], h, 2, 2, 1);
    // Voxel (1, 0): x neighbour 0, y neighbour 1.5 h  =>  plain a + h = h.
    let seeds = [
        (VoxelCoord::new(0, 0, 0), 0.0),
        (VoxelCoord::new(1, 1, 0), 1.5 * h),
    ];
    let phi = FastSweeping::compute(&grid, &seeds);
    assert!(approx(phi[flat(2, 2, 1, 0, 0)], h, 1e-12), "{phi:?}");
    assert!(approx(phi[flat(2, 2, 0, 1, 0)], h, 1e-12), "{phi:?}");
    assert!(approx(phi[flat(2, 2, 1, 1, 0)], 1.5 * h, 1e-12), "{phi:?}");
}

#[test]
fn fast_sweeping_single_centre_seed_is_exact_on_the_axes() {
    let n = 7;
    let h = 0.5;
    let grid = SdfVoxelGrid::new([0.0; 3], h, n, n, n);
    let c = 3;
    let phi = FastSweeping::compute(&grid, &[(VoxelCoord::new(3, 3, 3), 0.0)]);
    for k in 1..=3_usize {
        let expected = k as f64 * h;
        for (dx, dy, dz) in [(k, 0, 0), (0, k, 0), (0, 0, k)] {
            for sign in [-1_isize, 1] {
                let ix = (c as isize + sign * dx as isize) as usize;
                let iy = (c as isize + sign * dy as isize) as usize;
                let iz = (c as isize + sign * dz as isize) as usize;
                let value = phi[flat(n, n, ix, iy, iz)];
                assert!(
                    approx(value, expected, 1e-12),
                    "({ix},{iy},{iz}): {value} != {expected}"
                );
            }
        }
    }
    // Everywhere else the first-order scheme stays within one cell of the
    // Euclidean distance.
    for iz in 0..n {
        for iy in 0..n {
            for ix in 0..n {
                let dx = ix as f64 - c as f64;
                let dy = iy as f64 - c as f64;
                let dz = iz as f64 - c as f64;
                let euclid = (dx * dx + dy * dy + dz * dz).sqrt() * h;
                let value = phi[flat(n, n, ix, iy, iz)];
                assert!(
                    (value - euclid).abs() <= h,
                    "({ix},{iy},{iz}): {value} vs Euclidean {euclid}"
                );
            }
        }
    }
}

#[test]
fn fast_sweeping_two_seeds_are_exact_between_them() {
    let h = 0.25;
    let grid = SdfVoxelGrid::new([0.0; 3], h, 5, 1, 1);
    let phi = FastSweeping::compute(
        &grid,
        &[
            (VoxelCoord::new(0, 0, 0), 0.0),
            (VoxelCoord::new(4, 0, 0), 0.0),
        ],
    );
    let expected = [0.0, 1.0, 2.0, 1.0, 0.0];
    for (value, factor) in phi.iter().zip(expected) {
        assert!(approx(*value, factor * h, 1e-12), "{phi:?}");
    }
}

#[test]
fn fast_sweeping_keeps_seed_values() {
    let h = 1.0;
    let grid = SdfVoxelGrid::new([0.0; 3], h, 4, 1, 1);
    // The seed at x = 1 is far above what its neighbour would give it.
    let seeds = [
        (VoxelCoord::new(0, 0, 0), 0.0),
        (VoxelCoord::new(1, 0, 0), 5.0),
        (VoxelCoord::new(1, 0, 0), 7.0),
    ];
    let phi = FastSweeping::compute(&grid, &seeds);
    assert!(approx(phi[0], 0.0, 1e-12));
    assert!(
        approx(phi[1], 5.0, 1e-12),
        "seed must not be lowered: {phi:?}"
    );
    assert!(approx(phi[2], 6.0, 1e-12));
    assert!(approx(phi[3], 7.0, 1e-12));
}

#[test]
fn fast_sweeping_ignores_invalid_seeds() {
    let grid = SdfVoxelGrid::new([0.0; 3], 1.0, 3, 3, 1);
    let seeds = [
        (VoxelCoord::new(-1, 0, 0), 0.0),
        (VoxelCoord::new(0, 7, 0), 0.0),
        (VoxelCoord::new(0, 0, 4), 0.0),
        (VoxelCoord::new(2, 2, 0), f64::NAN),
        (VoxelCoord::new(2, 2, 0), f64::INFINITY),
        (VoxelCoord::new(0, 0, 0), 0.0),
    ];
    let phi = FastSweeping::compute(&grid, &seeds);
    assert!(approx(phi[flat(3, 3, 0, 0, 0)], 0.0, 1e-12));
    assert!(approx(phi[flat(3, 3, 2, 0, 0)], 2.0, 1e-12));
    assert!(phi.iter().all(|v| v.is_finite()), "{phi:?}");
}

#[test]
fn fast_sweeping_without_valid_seeds_or_cell_size_stays_infinite() {
    let grid = SdfVoxelGrid::new([0.0; 3], 1.0, 3, 3, 3);
    let phi = FastSweeping::compute(&grid, &[]);
    assert!(phi.iter().all(|v| v.is_infinite()));

    let degenerate = SdfVoxelGrid::new([0.0; 3], 0.0, 3, 3, 3);
    let phi = FastSweeping::compute(&degenerate, &[(VoxelCoord::new(1, 1, 1), 0.0)]);
    assert!(approx(phi[flat(3, 3, 1, 1, 1)], 0.0, 0.0));
    assert!(phi.iter().filter(|v| v.is_infinite()).count() == 26);

    let empty = SdfVoxelGrid::new([0.0; 3], 1.0, 0, 3, 3);
    assert!(FastSweeping::compute(&empty, &[(VoxelCoord::new(0, 0, 0), 0.0)]).is_empty());
}

#[test]
fn fast_sweeping_result_is_a_fixed_point_of_the_eikonal_update() {
    // Irregular seeds with different values: after convergence every free voxel
    // must equal the Godunov update of its own neighbours.
    let (nx, ny, nz) = (9, 8, 7);
    let h = 0.3;
    let grid = SdfVoxelGrid::new([0.0; 3], h, nx, ny, nz);
    let seeds = [
        (VoxelCoord::new(1, 6, 2), 0.2),
        (VoxelCoord::new(7, 1, 5), 0.0),
        (VoxelCoord::new(4, 4, 0), 0.9),
        (VoxelCoord::new(8, 7, 6), 0.1),
        (VoxelCoord::new(0, 0, 3), 0.0),
    ];
    let phi = FastSweeping::compute(&grid, &seeds);
    let is_seed = |ix: usize, iy: usize, iz: usize| {
        seeds
            .iter()
            .any(|(c, _)| (c.ix as usize, c.iy as usize, c.iz as usize) == (ix, iy, iz))
    };
    let at = |ix: usize, iy: usize, iz: usize| phi[flat(nx, ny, ix, iy, iz)];
    for iz in 0..nz {
        for iy in 0..ny {
            for ix in 0..nx {
                if is_seed(ix, iy, iz) {
                    continue;
                }
                let along = |lo: Option<f64>, hi: Option<f64>| {
                    lo.unwrap_or(f64::INFINITY).min(hi.unwrap_or(f64::INFINITY))
                };
                let ax = along(
                    (ix > 0).then(|| at(ix - 1, iy, iz)),
                    (ix + 1 < nx).then(|| at(ix + 1, iy, iz)),
                );
                let ay = along(
                    (iy > 0).then(|| at(ix, iy - 1, iz)),
                    (iy + 1 < ny).then(|| at(ix, iy + 1, iz)),
                );
                let az = along(
                    (iz > 0).then(|| at(ix, iy, iz - 1)),
                    (iz + 1 < nz).then(|| at(ix, iy, iz + 1)),
                );
                let again = godunov_update([ax, ay, az], h);
                assert!(
                    approx(again, at(ix, iy, iz), 1e-9),
                    "({ix},{iy},{iz}): {} is not a fixed point (update gives {again})",
                    at(ix, iy, iz)
                );
            }
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// MeshVoxelizer (F0301)
// ─────────────────────────────────────────────────────────────────────────────

/// Closed axis-aligned box, triangles wound counter-clockwise seen from outside.
fn box_mesh(lo: [f64; 3], hi: [f64; 3]) -> (Vec<[f64; 3]>, Vec<[usize; 3]>) {
    let vertices = vec![
        [lo[0], lo[1], lo[2]],
        [hi[0], lo[1], lo[2]],
        [hi[0], hi[1], lo[2]],
        [lo[0], hi[1], lo[2]],
        [lo[0], lo[1], hi[2]],
        [hi[0], lo[1], hi[2]],
        [hi[0], hi[1], hi[2]],
        [lo[0], hi[1], hi[2]],
    ];
    let triangles = vec![
        [0, 3, 2],
        [0, 2, 1], // -z
        [4, 5, 6],
        [4, 6, 7], // +z
        [0, 1, 5],
        [0, 5, 4], // -y
        [3, 7, 6],
        [3, 6, 2], // +y
        [0, 4, 7],
        [0, 7, 3], // -x
        [1, 2, 6],
        [1, 6, 5], // +x
    ];
    (vertices, triangles)
}

/// Exact signed distance to an axis-aligned box (negative inside).
fn box_sdf(p: [f64; 3], lo: [f64; 3], hi: [f64; 3]) -> f64 {
    let q = [
        (lo[0] - p[0]).max(p[0] - hi[0]),
        (lo[1] - p[1]).max(p[1] - hi[1]),
        (lo[2] - p[2]).max(p[2] - hi[2]),
    ];
    let outside = (q[0].max(0.0).powi(2) + q[1].max(0.0).powi(2) + q[2].max(0.0).powi(2)).sqrt();
    outside + q[0].max(q[1]).max(q[2]).min(0.0)
}

/// Explode an indexed mesh into triangle soup (three private vertices per triangle).
fn to_soup(vertices: &[[f64; 3]], triangles: &[[usize; 3]]) -> (Vec<[f64; 3]>, Vec<[usize; 3]>) {
    let mut soup_vertices = Vec::new();
    let mut soup_triangles = Vec::new();
    for tri in triangles {
        let base = soup_vertices.len();
        for &i in tri {
            soup_vertices.push(vertices[i]);
        }
        soup_triangles.push([base, base + 1, base + 2]);
    }
    (soup_vertices, soup_triangles)
}

/// Check a voxelized box against the analytic distance: exact inside the
/// 2-cell band, correct sign and within 1.5 cells farther away.
fn assert_matches_box(grid: &SdfVoxelGrid, lo: [f64; 3], hi: [f64; 3]) {
    let cell = grid.cell_size;
    for iz in 0..grid.nz {
        for iy in 0..grid.ny {
            for ix in 0..grid.nx {
                let center = grid.voxel_center(ix, iy, iz);
                let exact = box_sdf(center, lo, hi);
                let got = grid.data[flat(grid.nx, grid.ny, ix, iy, iz)];
                assert!(
                    exact.abs() > 1e-9,
                    "test voxel on the surface at {center:?}"
                );
                if (exact.abs() - 2.0 * cell).abs() < 1e-6 {
                    // Exactly on the edge of the exact band: either treatment is valid.
                    continue;
                }
                if exact.abs() <= 2.0 * cell {
                    assert!(
                        approx(got, exact, 1e-9),
                        "({ix},{iy},{iz}): band value {got} != exact {exact}"
                    );
                } else {
                    assert_eq!(
                        got < 0.0,
                        exact < 0.0,
                        "({ix},{iy},{iz}): wrong sign, got {got}, exact {exact}"
                    );
                    assert!(
                        (got.abs() - exact.abs()).abs() <= 1.5 * cell,
                        "({ix},{iy},{iz}): far value {got} vs exact {exact}"
                    );
                }
            }
        }
    }
}

#[test]
fn voxelize_box_matches_the_analytic_signed_distance() {
    let lo = [-0.9, -0.7, -1.1];
    let hi = [1.3, 0.8, 1.05];
    let (vertices, triangles) = box_mesh(lo, hi);
    let mut grid = SdfVoxelGrid::new([-2.4; 3], 0.4, 12, 12, 12);
    MeshVoxelizer::voxelize(&mut grid, &vertices, &triangles).unwrap();
    assert_matches_box(&grid, lo, hi);
    assert!(grid.count_inside(0.0) > 0);
}

#[test]
fn voxelize_large_box_extends_the_field_inside_and_outside() {
    // The box is deeper than the exact band, so both sides use the sweeping.
    let lo = [-1.45, -1.35, -1.25];
    let hi = [1.35, 1.4, 1.3];
    let (vertices, triangles) = box_mesh(lo, hi);
    let mut grid = SdfVoxelGrid::new([-2.0; 3], 0.25, 16, 16, 16);
    MeshVoxelizer::voxelize(&mut grid, &vertices, &triangles).unwrap();
    assert_matches_box(&grid, lo, hi);
    let deepest = grid.data.iter().copied().fold(f64::INFINITY, f64::min);
    assert!(deepest < -0.9, "deepest inside value {deepest}");
}

#[test]
fn voxelize_triangle_soup_matches_the_indexed_mesh() {
    let lo = [-0.9, -0.7, -1.1];
    let hi = [1.3, 0.8, 1.05];
    let (vertices, triangles) = box_mesh(lo, hi);
    let (soup_vertices, soup_triangles) = to_soup(&vertices, &triangles);
    let mut indexed = SdfVoxelGrid::new([-2.4; 3], 0.4, 12, 12, 12);
    let mut soup = SdfVoxelGrid::new([-2.4; 3], 0.4, 12, 12, 12);
    MeshVoxelizer::voxelize(&mut indexed, &vertices, &triangles).unwrap();
    MeshVoxelizer::voxelize(&mut soup, &soup_vertices, &soup_triangles).unwrap();
    assert_eq!(indexed.data, soup.data);
}

/// A closed prism whose cross-section is a thin isosceles triangle: the apex
/// edge runs along z from the origin, the two long faces meet there at an
/// interior angle of 10 degrees, so their normals are 170 degrees apart.
fn blade_mesh(length: f64) -> (Vec<[f64; 3]>, Vec<[usize; 3]>) {
    let w = length * 5.0_f64.to_radians().tan();
    let vertices = vec![
        [0.0, 0.0, 0.0],   // 0: apex, z = 0
        [0.0, 0.0, 1.0],   // 1: apex, z = 1
        [length, w, 0.0],  // 2
        [length, w, 1.0],  // 3
        [length, -w, 0.0], // 4
        [length, -w, 1.0], // 5
    ];
    // The -y face comes first on purpose: for points beside the apex on the +y
    // side its plane is the wrong one.
    let triangles = vec![
        [0, 4, 5],
        [0, 5, 1], // face towards -y
        [0, 3, 2],
        [0, 1, 3], // face towards +y
        [0, 2, 4], // cap z = 0
        [1, 5, 3], // cap z = 1
        [2, 3, 5],
        [2, 5, 4], // end face x = length
    ];
    (vertices, triangles)
}

fn blade_contains(p: [f64; 3], length: f64) -> bool {
    let slope = 5.0_f64.to_radians().tan();
    p[0] > 0.0 && p[0] < length && p[2] > 0.0 && p[2] < 1.0 && p[1].abs() < p[0] * slope
}

#[test]
fn sharp_ridge_point_is_classified_outside() {
    let (vertices, triangles) = blade_mesh(2.0);
    // Beside the apex edge, on the +y side: closest to the apex edge, outside.
    let p = [-0.1, 0.05, 0.5];
    // The plane of the first (-y) face puts the point on its negative side, so a
    // classification by the nearest face plane would call it inside.
    let wrong_plane = MeshVoxelizer::sign_winding(p, vertices[0], vertices[5], vertices[1]);
    assert!(wrong_plane < 0.0, "premise: plane sign is {wrong_plane}");
    // The triangles of both faces that contain the apex edge are equally close.
    let near_minus =
        MeshVoxelizer::point_triangle_dist_sq(p, vertices[0], vertices[5], vertices[1]);
    let near_plus = MeshVoxelizer::point_triangle_dist_sq(p, vertices[0], vertices[1], vertices[3]);
    assert!(approx(near_minus, near_plus, 1e-15));
    assert!(approx(near_minus, 0.0125, 1e-12));

    let mesh = PreparedMesh::new(&vertices, &triangles).unwrap();
    let d = mesh.signed_distance(p);
    assert!(approx(d, 0.0125_f64.sqrt(), 1e-12), "signed distance {d}");
    // The mirror image point on the -y side is outside as well.
    let d = mesh.signed_distance([-0.1, -0.05, 0.5]);
    assert!(approx(d, 0.0125_f64.sqrt(), 1e-12), "signed distance {d}");
}

#[test]
fn voxelize_sharp_ridge_gives_positive_voxel_outside() {
    let length = 2.0;
    let (vertices, triangles) = blade_mesh(length);
    // Voxel (0, 0, 0) is centred on the witness point (-0.1, 0.05, 0.5).
    let mut grid = SdfVoxelGrid::new([-0.15, 0.0, 0.45], 0.1, 10, 10, 10);
    MeshVoxelizer::voxelize(&mut grid, &vertices, &triangles).unwrap();
    let witness = grid.data[0];
    assert!(
        approx(witness, 0.0125_f64.sqrt(), 1e-12),
        "voxel beside the ridge: {witness}"
    );

    // Every voxel: sign from the analytic prism, magnitude from brute force.
    let mesh = PreparedMesh::new(&vertices, &triangles).unwrap();
    let mut inside = 0;
    for iz in 0..10 {
        for iy in 0..10 {
            for ix in 0..10 {
                let center = grid.voxel_center(ix, iy, iz);
                let got = grid.data[flat(10, 10, ix, iy, iz)];
                let brute = mesh.signed_distance(center);
                if brute.abs() < 1e-6 {
                    continue;
                }
                let is_inside = blade_contains(center, length);
                inside += usize::from(is_inside);
                assert_eq!(
                    got < 0.0,
                    is_inside,
                    "({ix},{iy},{iz}) at {center:?}: {got}"
                );
                if brute.abs() <= 0.19 {
                    assert!(
                        approx(got, brute, 1e-9),
                        "({ix},{iy},{iz}): {got} vs {brute}"
                    );
                }
            }
        }
    }
    assert!(inside > 0, "the grid must contain voxels inside the blade");
}

#[test]
fn sharp_ridge_is_classified_correctly_for_triangle_soup() {
    let (vertices, triangles) = blade_mesh(2.0);
    let (soup_vertices, soup_triangles) = to_soup(&vertices, &triangles);
    let mesh = PreparedMesh::new(&soup_vertices, &soup_triangles).unwrap();
    // Welding coincident vertices restores the edge adjacency.
    let d = mesh.signed_distance([-0.1, 0.05, 0.5]);
    assert!(approx(d, 0.0125_f64.sqrt(), 1e-12), "signed distance {d}");
}

#[test]
fn voxelize_octahedron_sphere_signs_match_the_radius() {
    // Octahedron subdivided three times and projected onto the unit sphere.
    fn subdivide(tris: Vec<[[f64; 3]; 3]>) -> Vec<[[f64; 3]; 3]> {
        let mid = |a: [f64; 3], b: [f64; 3]| {
            let m = [a[0] + b[0], a[1] + b[1], a[2] + b[2]];
            let l = (m[0] * m[0] + m[1] * m[1] + m[2] * m[2]).sqrt();
            [m[0] / l, m[1] / l, m[2] / l]
        };
        let mut out = Vec::with_capacity(tris.len() * 4);
        for [a, b, c] in tris {
            let (ab, bc, ca) = (mid(a, b), mid(b, c), mid(c, a));
            out.push([a, ab, ca]);
            out.push([ab, b, bc]);
            out.push([ca, bc, c]);
            out.push([ab, bc, ca]);
        }
        out
    }
    let axes = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];
    let mut tris: Vec<[[f64; 3]; 3]> = Vec::new();
    for sx in [1.0, -1.0] {
        for sy in [1.0, -1.0] {
            for sz in [1.0, -1.0] {
                let corner = |axis: usize, s: f64| axes[axis].map(|v| v * s);
                let (x, y, z) = (corner(0, sx), corner(1, sy), corner(2, sz));
                // Counter-clockwise from outside when sx * sy * sz > 0.
                if sx * sy * sz > 0.0 {
                    tris.push([x, y, z]);
                } else {
                    tris.push([x, z, y]);
                }
            }
        }
    }
    for _ in 0..3 {
        tris = subdivide(tris);
    }
    let mut vertices = Vec::new();
    let mut triangles = Vec::new();
    for tri in &tris {
        let base = vertices.len();
        vertices.extend_from_slice(tri);
        triangles.push([base, base + 1, base + 2]);
    }

    let cell = 0.2;
    let mut grid = SdfVoxelGrid::new([-1.6; 3], cell, 16, 16, 16);
    MeshVoxelizer::voxelize(&mut grid, &vertices, &triangles).unwrap();
    let mut inside = 0;
    for iz in 0..16 {
        for iy in 0..16 {
            for ix in 0..16 {
                let c = grid.voxel_center(ix, iy, iz);
                let r = (c[0] * c[0] + c[1] * c[1] + c[2] * c[2]).sqrt();
                let got = grid.data[flat(16, 16, ix, iy, iz)];
                if (r - 1.0).abs() > 0.05 {
                    assert_eq!(got < 0.0, r < 1.0, "({ix},{iy},{iz}): r = {r}, got {got}");
                    inside += usize::from(r < 1.0);
                }
                if (r - 1.0).abs() < 0.35 {
                    assert!(
                        approx(got, r - 1.0, 0.02),
                        "({ix},{iy},{iz}): r = {r}, got {got}"
                    );
                }
            }
        }
    }
    assert!(inside > 100, "only {inside} voxels inside the sphere");
}

/// Deterministic pseudo random numbers (64-bit LCG).
struct Lcg(u64);

impl Lcg {
    fn next_f64(&mut self) -> f64 {
        self.0 = self
            .0
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        (self.0 >> 11) as f64 / (1_u64 << 53) as f64
    }

    fn point(&mut self, half: f64) -> [f64; 3] {
        [
            (self.next_f64() * 2.0 - 1.0) * half,
            (self.next_f64() * 2.0 - 1.0) * half,
            (self.next_f64() * 2.0 - 1.0) * half,
        ]
    }
}

/// Brute-force distance from `p` to the nearest triangle (no sign).
fn nearest_triangle_distance(p: [f64; 3], vertices: &[[f64; 3]], triangles: &[[usize; 3]]) -> f64 {
    triangles
        .iter()
        .map(|t| {
            MeshVoxelizer::point_triangle_dist_sq(p, vertices[t[0]], vertices[t[1]], vertices[t[2]])
        })
        .fold(f64::INFINITY, f64::min)
        .sqrt()
}

#[test]
fn voxelize_random_tetrahedra_signs_match_the_analytic_test() {
    // Tetrahedra put every kind of closest feature (faces, edges, vertices) in
    // front of the grid voxels, with sharp and flat dihedral angles alike.
    let mut rng = Lcg(0x5DEE_CE66_D1CE_4E5B);
    let mut tested = 0;
    let mut round = 0;
    while tested < 12 {
        round += 1;
        assert!(
            round < 200,
            "could not draw enough non-degenerate tetrahedra"
        );
        let mut p = [
            rng.point(1.0),
            rng.point(1.0),
            rng.point(1.0),
            rng.point(1.0),
        ];
        let volume = {
            let e = |i: usize| [p[i][0] - p[0][0], p[i][1] - p[0][1], p[i][2] - p[0][2]];
            let (b, c, d) = (e(1), e(2), e(3));
            b[0] * (c[1] * d[2] - c[2] * d[1]) - b[1] * (c[0] * d[2] - c[2] * d[0])
                + b[2] * (c[0] * d[1] - c[1] * d[0])
        };
        if volume.abs() < 0.15 {
            continue;
        }
        if volume < 0.0 {
            p.swap(1, 2);
        }
        tested += 1;
        // Counter-clockwise from outside for a right-handed tetrahedron.
        let vertices = p.to_vec();
        let triangles = vec![[0, 2, 1], [0, 1, 3], [0, 3, 2], [1, 2, 3]];
        let mut grid = SdfVoxelGrid::new([-1.6; 3], 0.2, 16, 16, 16);
        MeshVoxelizer::voxelize(&mut grid, &vertices, &triangles).unwrap();
        let mut inside_count = 0;
        for iz in 0..16 {
            for iy in 0..16 {
                for ix in 0..16 {
                    let center = grid.voxel_center(ix, iy, iz);
                    let got = grid.data[flat(16, 16, ix, iy, iz)];
                    let distance = nearest_triangle_distance(center, &vertices, &triangles);
                    if distance < 1e-9 {
                        continue;
                    }
                    let inside = triangles.iter().all(|t| {
                        let (a, b, c) = (vertices[t[0]], vertices[t[1]], vertices[t[2]]);
                        let n = cross3(sub3(b, a), sub3(c, a));
                        dot3(n, sub3(center, a)) < 0.0
                    });
                    inside_count += usize::from(inside);
                    assert_eq!(
                        got < 0.0,
                        inside,
                        "round {round} voxel ({ix},{iy},{iz}) at {center:?}: got {got}, tetra {p:?}"
                    );
                    if distance <= 0.39 {
                        assert!(
                            approx(got.abs(), distance, 1e-9),
                            "round {round} voxel ({ix},{iy},{iz}): {got} vs distance {distance}"
                        );
                    }
                }
            }
        }
        assert!(
            inside_count > 0,
            "round {round}: no voxel inside the tetrahedron"
        );
    }
}

#[test]
fn voxelize_concave_l_prism_signs_match_the_analytic_test() {
    // L-shaped prism: the plan view (counter-clockwise) has a reflex corner at
    // (1, 1); extruded over z in [0, 1].
    let plan = [
        [0.0, 0.0],
        [2.0, 0.0],
        [2.0, 1.0],
        [1.0, 1.0],
        [1.0, 2.0],
        [0.0, 2.0],
    ];
    let mut vertices = Vec::new();
    for z in [0.0, 1.0] {
        for p in plan {
            vertices.push([p[0], p[1], z]);
        }
    }
    let mut triangles: Vec<[usize; 3]> = Vec::new();
    for i in 0..6 {
        let j = (i + 1) % 6;
        // Side wall, outward = to the right of the counter-clockwise edge.
        triangles.push([i, j, 6 + j]);
        triangles.push([i, 6 + j, 6 + i]);
    }
    for k in 2..6 {
        triangles.push([6, 6 + k - 1, 6 + k]); // top, +z
        triangles.push([0, k, k - 1]); // bottom, -z
    }
    let analytic_inside = |p: [f64; 3]| {
        p[0] > 0.0
            && p[0] < 2.0
            && p[1] > 0.0
            && p[1] < 2.0
            && p[2] > 0.0
            && p[2] < 1.0
            && !(p[0] > 1.0 && p[1] > 1.0)
    };
    let mut grid = SdfVoxelGrid::new([-0.65, -0.65, -0.65], 0.1, 33, 33, 23);
    MeshVoxelizer::voxelize(&mut grid, &vertices, &triangles).unwrap();
    let mut inside_count = 0;
    for iz in 0..23 {
        for iy in 0..33 {
            for ix in 0..33 {
                let center = grid.voxel_center(ix, iy, iz);
                let got = grid.data[flat(33, 33, ix, iy, iz)];
                let distance = nearest_triangle_distance(center, &vertices, &triangles);
                if distance < 1e-9 {
                    continue;
                }
                let inside = analytic_inside(center);
                inside_count += usize::from(inside);
                assert_eq!(
                    got < 0.0,
                    inside,
                    "({ix},{iy},{iz}) at {center:?}: got {got}"
                );
                if distance <= 0.19 {
                    assert!(
                        approx(got.abs(), distance, 1e-9),
                        "({ix},{iy},{iz}): {got} vs distance {distance}"
                    );
                }
            }
        }
    }
    assert!(
        inside_count > 500,
        "only {inside_count} voxels inside the prism"
    );
}

#[test]
fn voxelize_rejects_out_of_range_indices_without_touching_the_grid() {
    let (vertices, mut triangles) = box_mesh([0.0; 3], [1.0; 3]);
    triangles.push([0, 1, 99]);
    let mut grid = SdfVoxelGrid::new([-1.0; 3], 0.5, 6, 6, 6);
    grid.fill(7.0);
    let err = MeshVoxelizer::voxelize(&mut grid, &vertices, &triangles).unwrap_err();
    match err {
        Error::IndexOutOfRange { index, len } => {
            assert_eq!(index, 99);
            assert_eq!(len, 8);
        }
        other => panic!("unexpected error {other:?}"),
    }
    assert!(
        grid.data.iter().all(|&v| v == 7.0),
        "grid must be unchanged"
    );
}

#[test]
fn voxelize_rejects_non_finite_vertices_and_bad_cell_sizes() {
    let (mut vertices, triangles) = box_mesh([0.0; 3], [1.0; 3]);
    let mut grid = SdfVoxelGrid::new([-1.0; 3], 0.5, 4, 4, 4);
    vertices[3][1] = f64::NAN;
    let err = MeshVoxelizer::voxelize(&mut grid, &vertices, &triangles).unwrap_err();
    assert!(matches!(err, Error::NonFinite { .. }), "{err:?}");

    let (vertices, triangles) = box_mesh([0.0; 3], [1.0; 3]);
    for bad in [0.0, -1.0, f64::NAN, f64::INFINITY] {
        let mut grid = SdfVoxelGrid::new([-1.0; 3], bad, 4, 4, 4);
        let err = MeshVoxelizer::voxelize(&mut grid, &vertices, &triangles).unwrap_err();
        assert!(
            matches!(err, Error::DegenerateShape { .. }),
            "{bad}: {err:?}"
        );
    }

    let mut grid = SdfVoxelGrid::new([-1.0; 3], 0.5, 4, 4, 4);
    grid.data.pop();
    let err = MeshVoxelizer::voxelize(&mut grid, &vertices, &triangles).unwrap_err();
    assert!(matches!(err, Error::General(_)), "{err:?}");
}

#[test]
fn voxelize_without_triangles_or_with_degenerate_ones_fills_infinity() {
    let vertices = vec![[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]];
    let mut grid = SdfVoxelGrid::new([-1.0; 3], 0.5, 4, 4, 4);
    grid.fill(1.0);
    MeshVoxelizer::voxelize(&mut grid, &vertices, &[]).unwrap();
    assert!(grid.data.iter().all(|v| v.is_infinite()));

    grid.fill(1.0);
    // Collinear and repeated-vertex triangles carry no normal.
    MeshVoxelizer::voxelize(&mut grid, &vertices, &[[0, 1, 2], [0, 0, 1]]).unwrap();
    assert!(grid.data.iter().all(|v| v.is_infinite()));
}

#[test]
fn voxelize_ignores_degenerate_triangles_in_a_valid_mesh() {
    let lo = [-0.9, -0.7, -1.1];
    let hi = [1.3, 0.8, 1.05];
    let (mut vertices, mut triangles) = box_mesh(lo, hi);
    let mut baseline = SdfVoxelGrid::new([-2.4; 3], 0.4, 12, 12, 12);
    MeshVoxelizer::voxelize(&mut baseline, &vertices, &triangles).unwrap();

    // A sliver whose three vertices are collinear, and a repeated-index triangle.
    vertices.push([0.0, 0.0, 1.05]);
    vertices.push([0.5, 0.0, 1.05]);
    vertices.push([1.0, 0.0, 1.05]);
    triangles.push([8, 9, 10]);
    triangles.push([8, 8, 9]);
    let mut with_degenerate = SdfVoxelGrid::new([-2.4; 3], 0.4, 12, 12, 12);
    MeshVoxelizer::voxelize(&mut with_degenerate, &vertices, &triangles).unwrap();
    assert_eq!(baseline.data, with_degenerate.data);
}

#[test]
fn voxelize_mesh_far_from_the_grid_uses_exact_distances() {
    // The unit cube is much farther than the narrow band from every voxel.
    let (vertices, triangles) = box_mesh([0.0; 3], [1.0; 3]);
    let mut grid = SdfVoxelGrid::new([10.0, 0.0, 0.0], 0.5, 4, 4, 4);
    MeshVoxelizer::voxelize(&mut grid, &vertices, &triangles).unwrap();
    for iz in 0..4 {
        for iy in 0..4 {
            for ix in 0..4 {
                let c = grid.voxel_center(ix, iy, iz);
                let exact = box_sdf(c, [0.0; 3], [1.0; 3]);
                let got = grid.data[flat(4, 4, ix, iy, iz)];
                assert!(exact > 8.0);
                assert!(
                    approx(got, exact, 1e-9),
                    "({ix},{iy},{iz}): {got} vs {exact}"
                );
            }
        }
    }
}

#[test]
fn voxelize_grid_enclosed_by_a_large_mesh_is_negative() {
    let (vertices, triangles) = box_mesh([-50.0; 3], [50.0; 3]);
    let mut grid = SdfVoxelGrid::new([-1.0; 3], 0.5, 4, 4, 4);
    MeshVoxelizer::voxelize(&mut grid, &vertices, &triangles).unwrap();
    for &value in &grid.data {
        assert!(value < -40.0, "{value}");
    }
}

#[test]
fn closest_point_on_triangle_reports_the_feature() {
    let a = [0.0, 0.0, 0.0];
    let b = [2.0, 0.0, 0.0];
    let c = [0.0, 2.0, 0.0];
    let case = |p: [f64; 3]| MeshVoxelizer::closest_point_on_triangle(p, a, b, c);

    let face = case([0.5, 0.5, 1.0]);
    assert_eq!(face.feature, TriangleFeature::Face);
    assert!(approx(face.dist_sq, 1.0, 1e-12));
    assert!(approx(face.point[0], 0.5, 1e-12) && approx(face.point[1], 0.5, 1e-12));

    let edge_ab = case([1.0, -1.0, 0.0]);
    assert_eq!(edge_ab.feature, TriangleFeature::Edge(0));
    assert!(approx(edge_ab.point[0], 1.0, 1e-12));

    let edge_bc = case([2.0, 2.0, 0.0]);
    assert_eq!(edge_bc.feature, TriangleFeature::Edge(1));
    assert!(approx(edge_bc.point[0], 1.0, 1e-12) && approx(edge_bc.point[1], 1.0, 1e-12));

    let edge_ca = case([-1.0, 1.0, 0.0]);
    assert_eq!(edge_ca.feature, TriangleFeature::Edge(2));
    assert!(approx(edge_ca.point[1], 1.0, 1e-12));

    assert_eq!(case([-1.0, -1.0, 0.0]).feature, TriangleFeature::Vertex(0));
    assert_eq!(case([3.0, -1.0, 0.0]).feature, TriangleFeature::Vertex(1));
    assert_eq!(case([-1.0, 3.0, 0.0]).feature, TriangleFeature::Vertex(2));

    // A closest point exactly at a corner is a vertex, not an edge: the point
    // projects onto the end of edge a-b.
    let at_corner = case([0.0, -1.0, 0.0]);
    assert_eq!(at_corner.feature, TriangleFeature::Vertex(0));
    assert!(approx(at_corner.dist_sq, 1.0, 1e-12));
}

#[test]
fn closest_point_on_degenerate_triangle_is_finite() {
    let a = [0.0, 0.0, 0.0];
    let b = [1.0, 0.0, 0.0];
    let c = [2.0, 0.0, 0.0];
    for p in [
        [1.0, 1.0, 0.0],
        [-1.0, 0.0, 0.0],
        [5.0, 1.0, 1.0],
        [0.5, 0.0, 0.0],
    ] {
        let closest = MeshVoxelizer::closest_point_on_triangle(p, a, b, c);
        assert!(closest.dist_sq.is_finite() && closest.point.iter().all(|x| x.is_finite()));
    }
    let same = MeshVoxelizer::closest_point_on_triangle([1.0, 1.0, 0.0], a, a, a);
    assert!(approx(same.dist_sq, 2.0, 1e-12));
}

// ─────────────────────────────────────────────────────────────────────────────
// Grid sampling (interpolate / gradient / clamped_voxel)
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn interpolate_is_defined_up_to_the_grid_box() {
    let mut g = SdfVoxelGrid::new([0.0; 3], 1.0, 4, 4, 4);
    g.fill(-1.0);
    // The outer half cell was infinite before the stencil was clamped.
    for p in [
        [0.1, 0.1, 0.1],
        [0.0, 2.0, 2.0],
        [4.0, 4.0, 4.0],
        [3.9, 2.0, 0.05],
    ] {
        assert!(approx(g.interpolate(p), -1.0, 1e-12), "{p:?}");
    }
    // Outside the box there is no field.
    for p in [
        [-0.1, 2.0, 2.0],
        [4.1, 2.0, 2.0],
        [2.0, 2.0, -3.0],
        [f64::NAN, 2.0, 2.0],
    ] {
        assert!(g.interpolate(p).is_infinite(), "{p:?}");
    }
}

#[test]
fn interpolate_clamped_extends_the_border_values() {
    let mut g = SdfVoxelGrid::new([0.0; 3], 1.0, 4, 4, 4);
    for iz in 0..4 {
        for iy in 0..4 {
            for ix in 0..4 {
                g.set(ix, iy, iz, ix as f64);
            }
        }
    }
    assert!(approx(g.interpolate_clamped([-7.0, 2.0, 2.0]), 0.0, 1e-12));
    assert!(approx(g.interpolate_clamped([40.0, 2.0, 2.0]), 3.0, 1e-12));
    assert!(approx(g.interpolate_clamped([2.0, -9.0, 99.0]), 1.5, 1e-12));
    assert!(g.interpolate_clamped([f64::NAN, 0.0, 0.0]).is_nan());
    let empty = SdfVoxelGrid::new([0.0; 3], 1.0, 0, 4, 4);
    assert!(empty.interpolate_clamped([0.0; 3]).is_infinite());
}

#[test]
fn interpolate_skips_zero_weight_infinite_samples() {
    let mut g = SdfVoxelGrid::new([0.0; 3], 1.0, 4, 4, 4);
    g.set(1, 1, 1, -1.0);
    // Exactly at the centre every neighbour has weight zero: used to be NaN.
    let at_center = g.interpolate([1.5, 1.5, 1.5]);
    assert!(approx(at_center, -1.0, 1e-12), "{at_center}");
    // Anywhere else in the cell an infinite neighbour carries weight.
    assert!(g.interpolate([1.6, 1.5, 1.5]).is_infinite());
    // Along an axis with zero fractional offset only the aligned samples count.
    // With a finite neighbour along x the other (infinite) samples still have
    // weight zero: 0.75 * -1 + 0.25 * 3 = 0.
    g.set(2, 1, 1, 3.0);
    let along_x = g.interpolate([1.75, 1.5, 1.5]);
    assert!(approx(along_x, 0.0, 1e-12), "{along_x}");
}

#[test]
fn clamped_voxel_covers_the_max_faces_and_the_outside() {
    let g = SdfVoxelGrid::new([0.0; 3], 1.0, 4, 3, 2);
    assert_eq!(g.clamped_voxel([4.0, 3.0, 2.0]), Some([3, 2, 1]));
    assert_eq!(g.clamped_voxel([0.0, 0.0, 0.0]), Some([0, 0, 0]));
    assert_eq!(g.clamped_voxel([1.5, 2.5, 0.5]), Some([1, 2, 0]));
    assert_eq!(g.clamped_voxel([-9.0, 50.0, 1.2]), Some([0, 2, 1]));
    assert_eq!(g.clamped_voxel([f64::NAN, 0.0, 0.0]), None);
    assert_eq!(
        SdfVoxelGrid::new([0.0; 3], 1.0, 0, 1, 1).clamped_voxel([0.0; 3]),
        None
    );
}

#[test]
fn gradient_uses_one_sided_differences_on_the_boundary() {
    let mut g = SdfVoxelGrid::new([0.0; 3], 0.5, 4, 3, 1);
    for iy in 0..3 {
        for ix in 0..4 {
            g.set(ix, iy, 0, 2.0 * ix as f64 + 0.5 * iy as f64);
        }
    }
    let expected = {
        let len = (2.0_f64 * 2.0 + 0.5 * 0.5).sqrt();
        [2.0 / len, 0.5 / len, 0.0]
    };
    // Interior, edge, corner, and a size-one axis (z has a single layer).
    for (ix, iy) in [(1, 1), (0, 1), (3, 1), (0, 0), (3, 2), (2, 0)] {
        let grad = g.gradient(ix, iy, 0);
        for k in 0..3 {
            assert!(approx(grad[k], expected[k], 1e-12), "({ix},{iy}): {grad:?}");
        }
    }
    assert_eq!(g.gradient(9, 0, 0), [0.0; 3]);
}

#[test]
fn sphere_query_reports_a_contact_in_the_outer_half_cell() {
    // The field x_center - 0.9 crosses zero at x = 0.9; the sphere centre lies
    // in the outer half cell, where the SDF used to be infinite.
    let mut g = SdfVoxelGrid::new([0.0; 3], 1.0, 4, 4, 4);
    for iz in 0..4 {
        for iy in 0..4 {
            for ix in 0..4 {
                g.set(ix, iy, iz, ix as f64 - 0.4);
            }
        }
    }
    let contact = VoxelSphereQuery::query(&g, [0.2, 2.0, 2.0], 1.0).expect("contact");
    assert!(approx(contact.depth, 1.4, 1e-12), "{contact:?}");
    assert!(approx(contact.normal[0], 1.0, 1e-12), "{contact:?}");
    // A centre outside the grid box still yields no contact.
    assert!(VoxelSphereQuery::query(&g, [-3.0, 2.0, 2.0], 1.0).is_none());
    // On the max face the lookup no longer falls back to an arbitrary normal.
    let at_max = VoxelSphereQuery::query(&g, [4.0, 2.0, 2.0], 5.0).expect("contact");
    assert!(approx(at_max.normal[0], 1.0, 1e-12), "{at_max:?}");
}

// ─────────────────────────────────────────────────────────────────────────────
// VoxelRayQuery (F0303)
// ─────────────────────────────────────────────────────────────────────────────

fn solid_grid() -> SdfVoxelGrid {
    let mut g = SdfVoxelGrid::new([0.0; 3], 1.0, 4, 4, 4);
    g.fill(-1.0);
    g
}

#[test]
fn ray_from_outside_hits_the_min_face() {
    let g = solid_grid();
    let hit = VoxelRayQuery::cast(&g, [-1.0, 2.0, 2.0], [1.0, 0.0, 0.0], 20.0, 0.0).expect("hit");
    assert!(approx(hit.t, 1.0, 1e-9), "t = {}", hit.t);
    assert!(approx(hit.position[0], 0.0, 1e-9), "{:?}", hit.position);
    assert!(approx(hit.position[1], 2.0, 1e-9) && approx(hit.position[2], 2.0, 1e-9));
    assert_eq!(hit.coord, [0, 2, 2]);
    assert!(approx(hit.sdf, -1.0, 1e-12));
}

#[test]
fn ray_from_outside_hits_the_max_faces() {
    let g = solid_grid();
    // (origin, direction, expected t, expected voxel)
    type RayCase = ([f64; 3], [f64; 3], f64, [usize; 3]);
    let cases: [RayCase; 6] = [
        ([5.0, 2.0, 2.0], [-1.0, 0.0, 0.0], 1.0, [3, 2, 2]),
        ([2.0, 5.0, 2.0], [0.0, -1.0, 0.0], 1.0, [2, 3, 2]),
        ([2.0, 2.0, 7.0], [0.0, 0.0, -1.0], 3.0, [2, 2, 3]),
        ([2.0, -2.0, 2.0], [0.0, 1.0, 0.0], 2.0, [2, 0, 2]),
        ([2.0, 2.0, -3.0], [0.0, 0.0, 1.0], 3.0, [2, 2, 0]),
        (
            [-1.0, -1.0, -1.0],
            [1.0, 1.0, 1.0],
            3.0_f64.sqrt(),
            [0, 0, 0],
        ),
    ];
    for (origin, dir, t_expected, coord) in cases {
        let hit = VoxelRayQuery::cast(&g, origin, dir, 50.0, 0.0)
            .unwrap_or_else(|| panic!("no hit from {origin:?} along {dir:?}"));
        assert!(approx(hit.t, t_expected, 1e-9), "{origin:?}: t = {}", hit.t);
        assert_eq!(hit.coord, coord, "{origin:?}");
    }
}

#[test]
fn ray_direction_need_not_be_normalised() {
    let g = solid_grid();
    let hit = VoxelRayQuery::cast(&g, [-1.0, 2.0, 2.0], [7.5, 0.0, 0.0], 20.0, 0.0).expect("hit");
    assert!(approx(hit.t, 1.0, 1e-9), "t is a distance, got {}", hit.t);
}

#[test]
fn ray_starting_inside_the_solid_hits_immediately() {
    let g = solid_grid();
    let hit = VoxelRayQuery::cast(&g, [1.5, 1.5, 1.5], [0.0, 1.0, 0.0], 20.0, 0.0).expect("hit");
    assert!(approx(hit.t, 0.0, 1e-12));
    assert!(approx(hit.position[1], 1.5, 1e-12));
}

#[test]
fn ray_misses_and_respects_max_t() {
    let g = solid_grid();
    // Passing beside the grid.
    assert!(VoxelRayQuery::cast(&g, [-1.0, 9.0, 2.0], [1.0, 0.0, 0.0], 50.0, 0.0).is_none());
    // Pointing away from the grid.
    assert!(VoxelRayQuery::cast(&g, [-1.0, 2.0, 2.0], [-1.0, 0.0, 0.0], 50.0, 0.0).is_none());
    // Too short to reach the grid face.
    assert!(VoxelRayQuery::cast(&g, [-1.0, 2.0, 2.0], [1.0, 0.0, 0.0], 0.5, 0.0).is_none());
    assert!(VoxelRayQuery::cast(&g, [-1.0, 2.0, 2.0], [1.0, 0.0, 0.0], 1.5, 0.0).is_some());
    // Invalid input.
    assert!(VoxelRayQuery::cast(&g, [-1.0, 2.0, 2.0], [0.0, 0.0, 0.0], 5.0, 0.0).is_none());
    assert!(VoxelRayQuery::cast(&g, [f64::NAN, 2.0, 2.0], [1.0, 0.0, 0.0], 5.0, 0.0).is_none());
    assert!(VoxelRayQuery::cast(&g, [-1.0, 2.0, 2.0], [1.0, 0.0, 0.0], f64::NAN, 0.0).is_none());
    assert!(VoxelRayQuery::cast(&g, [-1.0, 2.0, 2.0], [1.0, 0.0, 0.0], -1.0, 0.0).is_none());
    // An unlimited ray is fine.
    assert!(
        VoxelRayQuery::cast(&g, [-1.0, 2.0, 2.0], [1.0, 0.0, 0.0], f64::INFINITY, 0.0).is_some()
    );
}

#[test]
fn ray_against_an_empty_grid_finds_nothing() {
    let g = SdfVoxelGrid::new([0.0; 3], 1.0, 4, 4, 4);
    assert!(VoxelRayQuery::cast(&g, [-1.0, 2.0, 2.0], [1.0, 0.0, 0.0], 20.0, 0.0).is_none());
    let none = SdfVoxelGrid::new([0.0; 3], 1.0, 0, 4, 4);
    assert!(VoxelRayQuery::cast(&none, [-1.0, 2.0, 2.0], [1.0, 0.0, 0.0], 20.0, 0.0).is_none());
}

/// SDF of a sphere sampled at the voxel centres.
fn sphere_grid(center: [f64; 3], radius: f64) -> SdfVoxelGrid {
    let mut g = SdfVoxelGrid::new([0.0; 3], 0.25, 24, 24, 24);
    for iz in 0..24 {
        for iy in 0..24 {
            for ix in 0..24 {
                let c = g.voxel_center(ix, iy, iz);
                let d = ((c[0] - center[0]).powi(2)
                    + (c[1] - center[1]).powi(2)
                    + (c[2] - center[2]).powi(2))
                .sqrt();
                g.set(ix, iy, iz, d - radius);
            }
        }
    }
    g
}

#[test]
fn ray_hits_the_sphere_surface_from_outside_the_grid() {
    let g = sphere_grid([3.0; 3], 1.5);
    // Along +x from outside the grid: the surface is at x = 1.5, t = 3.5.
    let hit = VoxelRayQuery::cast(&g, [-2.0, 3.0, 3.0], [1.0, 0.0, 0.0], 50.0, 0.0).expect("hit");
    assert!(approx(hit.t, 3.5, 0.03), "t = {}", hit.t);
    assert!(
        approx(g.interpolate(hit.position), 0.0, 1e-6),
        "{:?}",
        hit.position
    );
    // Along -y from above: the surface is at y = 4.5, t = 5.5 from y = 10.
    let hit = VoxelRayQuery::cast(&g, [3.0, 10.0, 3.0], [0.0, -1.0, 0.0], 50.0, 0.0).expect("hit");
    assert!(approx(hit.t, 5.5, 0.03), "t = {}", hit.t);
    // Oblique ray towards the centre from a corner direction.
    let dir = [1.0, 1.0, 1.0];
    let hit = VoxelRayQuery::cast(&g, [-1.0, -1.0, -1.0], dir, 50.0, 0.0).expect("hit");
    let reach = 4.0 * 3.0_f64.sqrt() - 1.5;
    assert!(
        approx(hit.t, reach, 0.05),
        "t = {}, expected {reach}",
        hit.t
    );
    // Starting inside the grid but outside the sphere.
    let hit = VoxelRayQuery::cast(&g, [0.3, 3.0, 3.0], [1.0, 0.0, 0.0], 50.0, 0.0).expect("hit");
    assert!(approx(hit.position[0], 1.5, 0.03), "{:?}", hit.position);
}

#[test]
fn ray_passing_beside_the_sphere_misses() {
    let g = sphere_grid([3.0; 3], 1.5);
    assert!(VoxelRayQuery::cast(&g, [-2.0, 3.0, 5.0], [1.0, 0.0, 0.0], 50.0, 0.0).is_none());
    assert!(VoxelRayQuery::cast(&g, [-2.0, 3.0, 3.0], [-1.0, 0.0, 0.0], 50.0, 0.0).is_none());
    // The sphere is out of reach for a short ray, and reached by a longer one.
    assert!(VoxelRayQuery::cast(&g, [-2.0, 3.0, 3.0], [1.0, 0.0, 0.0], 3.0, 0.0).is_none());
    assert!(VoxelRayQuery::cast(&g, [-2.0, 3.0, 3.0], [1.0, 0.0, 0.0], 3.6, 0.0).is_some());
}

#[test]
fn ray_threshold_selects_the_level_set() {
    // Linear field 2 - x on a grid spanning [0, 6] (solid for x > 2): the level
    // 2 - x = 0.5 is at x = 1.5.
    let mut g = SdfVoxelGrid::new([0.0; 3], 0.25, 24, 24, 24);
    for iz in 0..24 {
        for iy in 0..24 {
            for ix in 0..24 {
                let c = g.voxel_center(ix, iy, iz);
                g.set(ix, iy, iz, 2.0 - c[0]);
            }
        }
    }
    let hit = VoxelRayQuery::cast(&g, [-1.0, 3.0, 3.0], [1.0, 0.0, 0.0], 50.0, 0.5).expect("hit");
    assert!(approx(hit.position[0], 1.5, 1e-6), "{:?}", hit.position);
    assert!(approx(hit.t, 2.5, 1e-6), "t = {}", hit.t);
    let hit = VoxelRayQuery::cast(&g, [-1.0, 3.0, 3.0], [1.0, 0.0, 0.0], 50.0, 0.0).expect("hit");
    assert!(approx(hit.position[0], 2.0, 1e-6), "{:?}", hit.position);
    // A ray that approaches the level set at a grazing angle does not stall.
    let grazing = VoxelRayQuery::cast(&g, [-1.0, 1.0, 3.0], [1.0, 0.001, 0.0], 50.0, 0.5);
    assert!(grazing.is_some());
}

#[test]
fn ray_finds_an_isolated_solid_voxel_in_an_empty_grid() {
    let mut g = SdfVoxelGrid::new([0.0; 3], 1.0, 8, 8, 8);
    g.set(3, 4, 4, -0.5);
    // Through the voxel along +x: enters its face at x = 3.
    let hit = VoxelRayQuery::cast(&g, [-1.0, 4.5, 4.5], [1.0, 0.0, 0.0], 50.0, 0.0).expect("hit");
    assert_eq!(hit.coord, [3, 4, 4]);
    assert!(approx(hit.position[0], 3.0, 1e-3), "{:?}", hit.position);
    assert!(approx(hit.t, 4.0, 1e-3), "t = {}", hit.t);
    assert!(approx(hit.sdf, -0.5, 1e-12));
    // Diagonal through the voxel from outside the grid.
    let hit = VoxelRayQuery::cast(&g, [1.0, 2.0, 2.0], [1.0, 1.0, 1.0], 50.0, 0.0).expect("hit");
    assert_eq!(hit.coord, [3, 4, 4]);
    // One voxel to the side: nothing.
    assert!(VoxelRayQuery::cast(&g, [-1.0, 6.5, 4.5], [1.0, 0.0, 0.0], 50.0, 0.0).is_none());
    assert!(VoxelRayQuery::cast(&g, [3.5, -1.0, 6.5], [0.0, 1.0, 0.0], 50.0, 0.0).is_none());
}

#[test]
fn ray_intersects_agrees_with_cast() {
    let g = solid_grid();
    assert!(VoxelRayQuery::intersects(
        &g,
        [-1.0, 2.0, 2.0],
        [1.0, 0.0, 0.0],
        20.0,
        0.0
    ));
    assert!(!VoxelRayQuery::intersects(
        &g,
        [-1.0, 9.0, 2.0],
        [1.0, 0.0, 0.0],
        20.0,
        0.0
    ));
}
