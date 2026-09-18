// Copyright 2026 COOLJAPAN OU (Team KitaSan)
// SPDX-License-Identifier: Apache-2.0

//! Unit tests of the triangle mesh processing functions.

use super::*;

fn unit_triangle() -> GpuMesh {
    let mut m = GpuMesh::new();
    m.add_vertex([0.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, 0.0]);
    m.add_vertex([1.0, 0.0, 0.0], [0.0, 0.0, 1.0], [1.0, 0.0]);
    m.add_vertex([0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [0.0, 1.0]);
    m.add_triangle(0, 1, 2);
    m
}

fn tetrahedron() -> GpuMesh {
    let mut m = GpuMesh::new();
    m.add_vertex([1.0, 1.0, 1.0], [0.0, 0.0, 1.0], [0.0, 0.0]);
    m.add_vertex([-1.0, -1.0, 1.0], [0.0, 0.0, 1.0], [1.0, 0.0]);
    m.add_vertex([-1.0, 1.0, -1.0], [0.0, 0.0, 1.0], [0.0, 1.0]);
    m.add_vertex([1.0, -1.0, -1.0], [0.0, 0.0, 1.0], [1.0, 1.0]);
    m.add_triangle(0, 1, 2);
    m.add_triangle(0, 1, 3);
    m.add_triangle(0, 2, 3);
    m.add_triangle(1, 2, 3);
    m
}

// ── add_vertex / add_triangle counts ────────────────────────────────────

#[test]
fn test_add_vertex_count() {
    let m = unit_triangle();
    assert_eq!(m.vertex_count(), 3);
}

#[test]
fn test_add_triangle_count() {
    let m = unit_triangle();
    assert_eq!(m.triangle_count(), 1);
}

#[test]
fn test_empty_mesh_counts() {
    let m = GpuMesh::new();
    assert_eq!(m.vertex_count(), 0);
    assert_eq!(m.triangle_count(), 0);
}

// ── triangle_area ────────────────────────────────────────────────────────

#[test]
fn test_triangle_area_unit_right_triangle() {
    let area = triangle_area([0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]);
    assert!((area - 0.5).abs() < 1e-6);
}

#[test]
fn test_triangle_area_degenerate_is_zero() {
    let area = triangle_area([0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]);
    assert!(area < 1e-6);
}

#[test]
fn test_triangle_area_equilateral() {
    // Equilateral with side 2: area = √3
    let area = triangle_area([0.0, 0.0, 0.0], [2.0, 0.0, 0.0], [1.0, 3.0f32.sqrt(), 0.0]);
    let expected = 3.0f32.sqrt();
    assert!(
        (area - expected).abs() < 1e-5,
        "area={area} expected={expected}"
    );
}

// ── triangle_normal ──────────────────────────────────────────────────────

#[test]
fn test_triangle_normal_unit_length() {
    let n = triangle_normal([0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]);
    let len = length3f(n);
    assert!((len - 1.0).abs() < 1e-6, "normal length={len}");
}

#[test]
fn test_triangle_normal_xy_plane_is_z() {
    let n = triangle_normal([0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]);
    assert!(n[2].abs() > 0.99);
}

#[test]
fn test_triangle_normal_degenerate_zero() {
    let n = triangle_normal([0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]);
    assert_eq!(n, [0.0; 3]);
}

// ── compute_aabb ─────────────────────────────────────────────────────────

#[test]
fn test_aabb_bounds_all_vertices() {
    let m = unit_triangle();
    let (mn, mx) = gpu_compute_aabb(&m);
    assert!(mn[0] <= 0.0 && mn[1] <= 0.0);
    assert!(mx[0] >= 1.0 && mx[1] >= 1.0);
}

#[test]
fn test_aabb_min_lt_max_nonempty() {
    let m = unit_triangle();
    let (mn, mx) = gpu_compute_aabb(&m);
    // At least one axis should have min < max.
    let any_spread = (0..3).any(|a| mx[a] > mn[a]);
    assert!(any_spread);
}

#[test]
fn test_aabb_empty_mesh() {
    let m = GpuMesh::new();
    let (mn, mx) = gpu_compute_aabb(&m);
    for i in 0..3 {
        assert_eq!(mn[i], mx[i]);
    }
}

// ── surface_area ─────────────────────────────────────────────────────────

#[test]
fn test_surface_area_positive_for_triangle() {
    let m = unit_triangle();
    let area = gpu_compute_surface_area(&m);
    assert!(area > 0.0);
}

#[test]
fn test_surface_area_unit_right_triangle() {
    let m = unit_triangle();
    let area = gpu_compute_surface_area(&m);
    assert!((area - 0.5).abs() < 1e-6);
}

#[test]
fn test_surface_area_empty_mesh() {
    let m = GpuMesh::new();
    assert!((gpu_compute_surface_area(&m)).abs() < 1e-10);
}

// ── compute_normals ──────────────────────────────────────────────────────

#[test]
fn test_compute_normals_unit_length() {
    let mut m = unit_triangle();
    gpu_compute_normals(&mut m);
    for n in &m.normals {
        let len = length3f(*n);
        assert!((len - 1.0).abs() < 1e-5 || len < 1e-9, "len={len}");
    }
}

#[test]
fn test_compute_normals_xy_plane() {
    let mut m = unit_triangle();
    gpu_compute_normals(&mut m);
    for n in &m.normals {
        assert!(n[2].abs() > 0.9, "expected z-dominant normal, got {:?}", n);
    }
}

// ── smooth_normals ────────────────────────────────────────────────────────

#[test]
fn test_smooth_normals_preserves_unit_length() {
    let mut m = unit_triangle();
    gpu_compute_normals(&mut m);
    gpu_smooth_normals(&mut m, 2);
    for n in &m.normals {
        let len = length3f(*n);
        assert!((len - 1.0).abs() < 0.01 || len < 1e-9);
    }
}

#[test]
fn test_smooth_normals_zero_iter_unchanged() {
    let mut m = unit_triangle();
    gpu_compute_normals(&mut m);
    let before = m.normals.clone();
    gpu_smooth_normals(&mut m, 0);
    assert_eq!(m.normals, before);
}

// ── loop_subdivision ─────────────────────────────────────────────────────

#[test]
fn test_loop_subdivision_quadruples_triangles() {
    let m = unit_triangle();
    let sub = gpu_loop_subdivision(&m);
    assert_eq!(sub.triangle_count(), m.triangle_count() * 4);
}

#[test]
fn test_loop_subdivision_increases_vertex_count() {
    let m = unit_triangle();
    let sub = gpu_loop_subdivision(&m);
    assert!(sub.vertex_count() > m.vertex_count());
}

#[test]
fn test_loop_subdivision_tetrahedron() {
    let m = tetrahedron();
    let sub = gpu_loop_subdivision(&m);
    assert_eq!(sub.triangle_count(), m.triangle_count() * 4);
}

// ── mesh_decimate: see the dedicated section at the end of the file ─

// ── edge_collapse ─────────────────────────────────────────────────────────

#[test]
fn test_edge_collapse_large_threshold_reduces_triangles() {
    let mut m = tetrahedron();
    let before = m.triangle_count();
    let _removed = gpu_edge_collapse(&mut m, 100.0);
    assert!(m.triangle_count() <= before);
}

#[test]
fn test_edge_collapse_zero_threshold_no_change() {
    let mut m = unit_triangle();
    let before = m.triangle_count();
    let removed = gpu_edge_collapse(&mut m, 0.0);
    assert_eq!(removed, 0);
    assert_eq!(m.triangle_count(), before);
}

// ── weld_vertices ─────────────────────────────────────────────────────────

#[test]
fn test_weld_vertices_no_duplicates() {
    let mut m = unit_triangle();
    let merged = gpu_weld_vertices(&mut m, 0.01);
    assert_eq!(merged, 0);
    assert_eq!(m.vertex_count(), 3);
}

#[test]
fn test_weld_vertices_all_same() {
    let mut m = GpuMesh::new();
    for _ in 0..3 {
        m.add_vertex([0.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, 0.0]);
    }
    m.add_triangle(0, 1, 2);
    let merged = gpu_weld_vertices(&mut m, 0.1);
    assert!(merged > 0);
}

// ── volume ────────────────────────────────────────────────────────────────

#[test]
fn test_volume_tetrahedron_nonzero() {
    // Use a simple axis-aligned tetrahedron with known positive volume.
    // Vertices: origin + three unit axis points.  Consistent CCW winding.
    let mut m = GpuMesh::new();
    m.add_vertex([0.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, 0.0]);
    m.add_vertex([1.0, 0.0, 0.0], [0.0, 0.0, 1.0], [1.0, 0.0]);
    m.add_vertex([0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [0.0, 1.0]);
    m.add_vertex([0.0, 0.0, 1.0], [0.0, 0.0, 1.0], [0.0, 0.0]);
    // Four faces with consistent outward winding:
    m.add_triangle(0, 2, 1); // base
    m.add_triangle(0, 1, 3);
    m.add_triangle(1, 2, 3);
    m.add_triangle(0, 3, 2);
    let vol = gpu_compute_volume(&m);
    // Volume of unit tetrahedron = 1/6
    assert!(vol.abs() > 0.01, "vol={vol}");
}

#[test]
fn test_volume_empty_mesh_zero() {
    let m = GpuMesh::new();
    assert!((gpu_compute_volume(&m)).abs() < 1e-10);
}

// ═══════════════════════════════════════════════════════════════════════════
// Fixtures and structural checks
// ═══════════════════════════════════════════════════════════════════════════

use std::collections::HashMap;

/// A tetrahedron with consistent outward winding (signed volume `+1/6`).
fn oriented_tetrahedron() -> GpuMesh {
    let mut m = GpuMesh::new();
    m.add_vertex([0.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, 0.0]);
    m.add_vertex([1.0, 0.0, 0.0], [0.0, 0.0, 1.0], [1.0, 0.0]);
    m.add_vertex([0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [0.0, 1.0]);
    m.add_vertex([0.0, 0.0, 1.0], [0.0, 0.0, 1.0], [1.0, 1.0]);
    m.add_triangle(0, 2, 1);
    m.add_triangle(0, 1, 3);
    m.add_triangle(1, 2, 3);
    m.add_triangle(0, 3, 2);
    m
}

/// Counts that characterise a triangle mesh's connectivity.
#[derive(Debug)]
struct Topology {
    vertices: usize,
    edges: usize,
    faces: usize,
    /// Edges with exactly one incident triangle.
    boundary_edges: usize,
    /// Edges with more than two incident triangles.
    non_manifold_edges: usize,
    /// Triangles with a repeated index or (almost) zero area.
    degenerate: usize,
    /// Vertices that no triangle uses.
    unreferenced: usize,
    /// Triangle corners that name a vertex that does not exist.
    out_of_range: usize,
}

impl Topology {
    fn euler(&self) -> i64 {
        self.vertices as i64 - self.edges as i64 + self.faces as i64
    }
}

fn topology(mesh: &GpuMesh) -> Topology {
    let nv = mesh.vertex_count();
    let mut edge_faces: HashMap<(u32, u32), usize> = HashMap::new();
    let mut used = vec![false; nv];
    let mut degenerate = 0;
    let mut out_of_range = 0;
    for tri in &mesh.indices {
        let mut valid = true;
        for (slot, &v) in tri.iter().enumerate() {
            if (v as usize) < nv {
                used[v as usize] = true;
            } else {
                out_of_range += 1;
                valid = false;
            }
            let w = tri[(slot + 1) % 3];
            *edge_faces
                .entry(if v < w { (v, w) } else { (w, v) })
                .or_insert(0) += 1;
        }
        let repeated = tri[0] == tri[1] || tri[1] == tri[2] || tri[0] == tri[2];
        if repeated
            || (valid
                && triangle_area(
                    mesh.vertices[tri[0] as usize],
                    mesh.vertices[tri[1] as usize],
                    mesh.vertices[tri[2] as usize],
                ) <= 1e-12)
        {
            degenerate += 1;
        }
    }
    Topology {
        vertices: nv,
        edges: edge_faces.len(),
        faces: mesh.triangle_count(),
        boundary_edges: edge_faces.values().filter(|&&n| n == 1).count(),
        non_manifold_edges: edge_faces.values().filter(|&&n| n > 2).count(),
        degenerate,
        unreferenced: used.iter().filter(|&&u| !u).count(),
        out_of_range,
    }
}

/// Assert that `mesh` is a clean closed 2-manifold with the given Euler
/// characteristic: no degenerate triangles, no unreferenced vertices, every
/// edge shared by exactly two triangles.
fn assert_clean_closed_manifold(mesh: &GpuMesh, euler: i64) {
    let t = topology(mesh);
    assert_eq!(t.out_of_range, 0, "{t:?}");
    assert_eq!(t.degenerate, 0, "{t:?}");
    assert_eq!(t.unreferenced, 0, "{t:?}");
    assert_eq!(t.boundary_edges, 0, "{t:?}");
    assert_eq!(t.non_manifold_edges, 0, "{t:?}");
    assert_eq!(t.euler(), euler, "{t:?}");
    assert_eq!(mesh.normals.len(), mesh.vertex_count());
    assert_eq!(mesh.tex_coords.len(), mesh.vertex_count());
}

/// Unit icosphere with `20 * 4^level` triangles (midpoint subdivision of an
/// icosahedron, every vertex projected onto the unit sphere).
fn icosphere(level: usize) -> GpuMesh {
    let t = (1.0 + 5.0_f32.sqrt()) / 2.0;
    let unit = |v: [f32; 3]| {
        let l = length3f(v);
        [v[0] / l, v[1] / l, v[2] / l]
    };
    let mut vertices: Vec<[f32; 3]> = [
        [-1.0, t, 0.0],
        [1.0, t, 0.0],
        [-1.0, -t, 0.0],
        [1.0, -t, 0.0],
        [0.0, -1.0, t],
        [0.0, 1.0, t],
        [0.0, -1.0, -t],
        [0.0, 1.0, -t],
        [t, 0.0, -1.0],
        [t, 0.0, 1.0],
        [-t, 0.0, -1.0],
        [-t, 0.0, 1.0],
    ]
    .into_iter()
    .map(unit)
    .collect();
    let mut triangles: Vec<[u32; 3]> = vec![
        [0, 11, 5],
        [0, 5, 1],
        [0, 1, 7],
        [0, 7, 10],
        [0, 10, 11],
        [1, 5, 9],
        [5, 11, 4],
        [11, 10, 2],
        [10, 7, 6],
        [7, 1, 8],
        [3, 9, 4],
        [3, 4, 2],
        [3, 2, 6],
        [3, 6, 8],
        [3, 8, 9],
        [4, 9, 5],
        [2, 4, 11],
        [6, 2, 10],
        [8, 6, 7],
        [9, 8, 1],
    ];
    for _ in 0..level {
        let mut midpoints: HashMap<(u32, u32), u32> = HashMap::new();
        let mut refined = Vec::with_capacity(triangles.len() * 4);
        for tri in &triangles {
            let mut mid = [0_u32; 3];
            for (slot, m) in mid.iter_mut().enumerate() {
                let (a, b) = (tri[slot], tri[(slot + 1) % 3]);
                let key = if a < b { (a, b) } else { (b, a) };
                *m = *midpoints.entry(key).or_insert_with(|| {
                    let (pa, pb) = (vertices[a as usize], vertices[b as usize]);
                    vertices.push(unit(midpoint3f(pa, pb)));
                    (vertices.len() - 1) as u32
                });
            }
            refined.push([tri[0], mid[0], mid[2]]);
            refined.push([tri[1], mid[1], mid[0]]);
            refined.push([tri[2], mid[2], mid[1]]);
            refined.push([mid[0], mid[1], mid[2]]);
        }
        triangles = refined;
    }
    let mut mesh = GpuMesh::new();
    for v in &vertices {
        mesh.add_vertex(*v, *v, [0.0, 0.0]);
    }
    for t in &triangles {
        mesh.add_triangle(t[0], t[1], t[2]);
    }
    mesh
}

/// Flat `n x n` grid of unit squares in the plane `z = 0` (two triangles per
/// square, one diagonal direction: interior valence six).  The texture
/// coordinate of a vertex is its position divided by `n`.
fn flat_grid(n: usize) -> GpuMesh {
    let mut m = GpuMesh::new();
    for j in 0..=n {
        for i in 0..=n {
            let uv = [i as f32 / n as f32, j as f32 / n as f32];
            m.add_vertex([i as f32, j as f32, 0.0], [0.0, 0.0, 1.0], uv);
        }
    }
    let row = (n + 1) as u32;
    for j in 0..n as u32 {
        for i in 0..n as u32 {
            let a = j * row + i;
            let (b, c, d) = (a + 1, a + row, a + row + 1);
            m.add_triangle(a, b, d);
            m.add_triangle(a, d, c);
        }
    }
    m
}

/// Torus with major radius 2 and minor radius 1, `nu x nv` quads.
fn torus(nu: usize, nv: usize) -> GpuMesh {
    let mut m = GpuMesh::new();
    let tau = std::f32::consts::TAU;
    for j in 0..nv {
        for i in 0..nu {
            let (u, v) = (tau * i as f32 / nu as f32, tau * j as f32 / nv as f32);
            let ring = 2.0 + v.cos();
            m.add_vertex(
                [ring * u.cos(), ring * u.sin(), v.sin()],
                [0.0, 0.0, 1.0],
                [0.0, 0.0],
            );
        }
    }
    for j in 0..nv {
        for i in 0..nu {
            let (i1, j1) = ((i + 1) % nu, (j + 1) % nv);
            let a = (j * nu + i) as u32;
            let b = (j * nu + i1) as u32;
            let c = (j1 * nu + i) as u32;
            let d = (j1 * nu + i1) as u32;
            m.add_triangle(a, b, d);
            m.add_triangle(a, d, c);
        }
    }
    m
}

/// Latitude/longitude sphere: `nu` segments, `nv` rings, a pole vertex at
/// each end (so the poles have valence `nu`).
fn uv_sphere(nu: usize, nv: usize) -> GpuMesh {
    let mut m = GpuMesh::new();
    let pi = std::f32::consts::PI;
    m.add_vertex([0.0, 1.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0]);
    for j in 1..nv {
        let phi = pi * j as f32 / nv as f32;
        for i in 0..nu {
            let theta = 2.0 * pi * i as f32 / nu as f32;
            let p = [phi.sin() * theta.cos(), phi.cos(), phi.sin() * theta.sin()];
            m.add_vertex(p, p, [0.0, 0.0]);
        }
    }
    m.add_vertex([0.0, -1.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0]);
    let south = (m.vertex_count() - 1) as u32;
    for i in 0..nu {
        m.add_triangle(0, 1 + ((i + 1) % nu) as u32, 1 + i as u32);
    }
    for j in 0..nv - 2 {
        for i in 0..nu {
            let a = 1 + (j * nu + i) as u32;
            let b = 1 + (j * nu + (i + 1) % nu) as u32;
            let (c, d) = (a + nu as u32, b + nu as u32);
            m.add_triangle(a, b, d);
            m.add_triangle(a, d, c);
        }
    }
    for i in 0..nu {
        let a = 1 + ((nv - 2) * nu + i) as u32;
        let b = 1 + ((nv - 2) * nu + (i + 1) % nu) as u32;
        m.add_triangle(south, a, b);
    }
    m
}

fn assert_close3(a: [f32; 3], b: [f32; 3], tol: f32, what: &str) {
    for (x, y) in a.iter().zip(b.iter()) {
        assert!((x - y).abs() <= tol, "{what}: {a:?} vs {b:?}");
    }
}

/// `sum(weight_i * v_i)` of weighted points.
fn combination(terms: &[(f32, [f32; 3])]) -> [f32; 3] {
    let mut sum = [0.0_f32; 3];
    for (weight, v) in terms {
        for (slot, component) in sum.iter_mut().zip(v.iter()) {
            *slot += weight * component;
        }
    }
    sum
}

/// Deterministic uniform values in `[0, 1)` (SplitMix64).
struct UnitRng(u64);

impl UnitRng {
    fn next(&mut self) -> f32 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^= z >> 31;
        ((z >> 40) as f32) / (1_u64 << 24) as f32
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Loop subdivision
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn test_loop_beta_matches_the_closed_form() {
    assert!((loop_beta(3) - 3.0 / 16.0).abs() < 1e-12);
    assert!((loop_beta(6) - 1.0 / 16.0).abs() < 1e-12);
    // n = 4: (1/4)(5/8 - (3/8)^2).
    assert!((loop_beta(4) - 0.25 * (0.625 - 0.375 * 0.375)).abs() < 1e-12);
    // The vertex stencil is an affine combination for every valence.
    for n in 1..40_usize {
        assert!(loop_beta(n).is_finite());
    }
}

/// Loop's stencils on a tetrahedron (every vertex has valence 3, every edge is
/// interior): odd vertices at `3/8 (a + b) + 1/8 (c + d)`, even vertices at
/// `(1 - 3 beta) v + beta sum(neighbours)` with `beta = 3/16`.
#[test]
fn test_loop_subdivision_tetrahedron_uses_loop_stencils() {
    let m = tetrahedron();
    let sub = gpu_loop_subdivision(&m);
    assert_eq!(sub.vertex_count(), 10);
    assert_eq!(sub.triangle_count(), 16);

    let position = |v: usize| m.vertices[v];
    // Even vertices.
    let beta = 3.0 / 16.0_f32;
    for v in 0..4 {
        let mut terms = vec![(1.0 - 3.0 * beta, position(v))];
        terms.extend((0..4).filter(|&o| o != v).map(|o| (beta, position(o))));
        assert_close3(sub.vertices[v], combination(&terms), 1e-6, "even vertex");
        // The fixture's vertices sum to zero: the stencil reduces to v / 4.
        assert_close3(
            sub.vertices[v],
            combination(&[(0.25, position(v))]),
            1e-6,
            "even vertex = v/4",
        );
    }
    // Odd vertices: triangle k owns output triangles 4k..4k+4 =
    // [a, ab, ca], [b, bc, ab], [c, ca, bc], [ab, bc, ca].
    for (k, tri) in m.indices.iter().enumerate() {
        let corner = &sub.indices[4 * k];
        let next = &sub.indices[4 * k + 1];
        let (a, b, c) = (tri[0] as usize, tri[1] as usize, tri[2] as usize);
        let odd = [
            (a, b, corner[1] as usize),
            (b, c, next[1] as usize),
            (c, a, corner[2] as usize),
        ];
        for (u, w, index) in odd {
            assert!(index >= 4, "odd vertices follow the original ones");
            let others: Vec<usize> = (0..4).filter(|&o| o != u && o != w).collect();
            let expected = combination(&[
                (0.375, position(u)),
                (0.375, position(w)),
                (0.125, position(others[0])),
                (0.125, position(others[1])),
            ]);
            assert_close3(sub.vertices[index], expected, 1e-6, "odd vertex");
            assert_close3(
                sub.vertices[index],
                combination(&[(0.25, position(u)), (0.25, position(w))]),
                1e-6,
                "odd vertex = (a + b) / 4",
            );
        }
    }
}

#[test]
fn test_loop_subdivision_closed_mesh_counts_and_topology() {
    // V' = V + E and F' = 4F; the Euler characteristic is preserved.
    let m = oriented_tetrahedron();
    let once = gpu_loop_subdivision(&m);
    assert_clean_closed_manifold(&once, 2);
    assert_eq!((once.vertex_count(), once.triangle_count()), (10, 16));
    let twice = gpu_loop_subdivision(&once);
    assert_clean_closed_manifold(&twice, 2);
    assert_eq!((twice.vertex_count(), twice.triangle_count()), (34, 64));
    // A genus-1 surface keeps its Euler characteristic as well.
    let ring = torus(12, 6);
    assert_clean_closed_manifold(&ring, 0);
    assert_clean_closed_manifold(&gpu_loop_subdivision(&ring), 0);
}

#[test]
fn test_loop_subdivision_preserves_winding_and_shrinks_convex_surfaces() {
    let m = oriented_tetrahedron();
    let before = gpu_compute_volume(&m);
    assert!((before - 1.0 / 6.0).abs() < 1e-6);
    let mut mesh = m;
    let mut last = before;
    for _ in 0..3 {
        mesh = gpu_loop_subdivision(&mesh);
        let volume = gpu_compute_volume(&mesh);
        // Positive: the outward winding survived.  Smaller: Loop subdivision
        // approximates, it does not interpolate.
        assert!(
            volume > 0.0 && volume < last,
            "volume {volume} after {last}"
        );
        last = volume;
    }
}

/// The boundary rules on a lone triangle: odd vertices are edge midpoints and
/// even vertices move to `3/4 v + 1/8 (n0 + n1)`.
#[test]
fn test_loop_subdivision_boundary_rules_on_a_single_triangle() {
    let m = unit_triangle();
    let sub = gpu_loop_subdivision(&m);
    assert_eq!((sub.vertex_count(), sub.triangle_count()), (6, 4));
    assert_close3(sub.vertices[0], [0.125, 0.125, 0.0], 1e-7, "corner 0");
    // v1 = (1,0,0): 3/4 (1,0,0) + 1/8 ((0,0,0) + (0,1,0)).
    assert_close3(sub.vertices[1], [0.75, 0.125, 0.0], 1e-7, "corner 1");
    assert_close3(sub.vertices[2], [0.125, 0.75, 0.0], 1e-7, "corner 2");
    assert_close3(sub.vertices[3], [0.5, 0.0, 0.0], 1e-7, "edge (0,1)");
    assert_close3(sub.vertices[4], [0.5, 0.5, 0.0], 1e-7, "edge (1,2)");
    assert_close3(sub.vertices[5], [0.0, 0.5, 0.0], 1e-7, "edge (2,0)");
}

/// On a regular planar patch Loop subdivision reproduces the refined lattice:
/// interior and straight-boundary vertices stay put, every edge vertex is the
/// midpoint, and only the corners of the boundary move inwards.
#[test]
fn test_loop_subdivision_regular_planar_grid_gives_the_refined_lattice() {
    let n = 6;
    let grid = flat_grid(n);
    let sub = gpu_loop_subdivision(&grid);
    assert_eq!(sub.triangle_count(), 4 * grid.triangle_count());
    let corners = [
        [0.0, 0.0],
        [n as f32, 0.0],
        [0.0, n as f32],
        [n as f32, n as f32],
    ];
    for (index, p) in sub.vertices.iter().enumerate() {
        assert!(p[2].abs() < 1e-6, "vertex {index} left the plane: {p:?}");
        // Original vertices (first block) other than the corners are fixed.
        if index < grid.vertex_count() {
            let q = grid.vertices[index];
            let is_corner = corners.iter().any(|c| c[0] == q[0] && c[1] == q[1]);
            if !is_corner {
                assert_close3(*p, q, 1e-5, "regular vertex moved");
            }
        }
        // Everything lies on the half-spaced lattice, except next to a corner.
        let near_corner = corners
            .iter()
            .any(|c| (p[0] - c[0]).abs() < 0.3 && (p[1] - c[1]).abs() < 0.3);
        if !near_corner {
            for coordinate in &p[..2] {
                let doubled = coordinate * 2.0;
                assert!((doubled - doubled.round()).abs() < 1e-5, "{p:?}");
            }
        }
    }
    let t = topology(&sub);
    assert_eq!(t.euler(), 1, "a disk stays a disk: {t:?}");
    assert_eq!(t.degenerate + t.unreferenced + t.non_manifold_edges, 0);
}

/// Every Loop weight set sums to one, so an affine function of position is
/// reproduced: vertices of an irregular, bounded planar mesh (mixed valences,
/// boundary and interior) stay on their (tilted) plane.
#[test]
fn test_loop_subdivision_weights_sum_to_one_on_an_irregular_mesh() {
    let mut rng = UnitRng(42);
    let plane_z = |x: f32, y: f32| 2.0 * x - 3.0 * y + 1.0;
    let n = 7;
    let mut m = GpuMesh::new();
    for j in 0..=n {
        for i in 0..=n {
            let x = i as f32 + 0.3 * (rng.next() - 0.5);
            let y = j as f32 + 0.3 * (rng.next() - 0.5);
            m.add_vertex([x, y, plane_z(x, y)], [0.0, 0.0, 1.0], [0.0, 0.0]);
        }
    }
    let row = (n + 1) as u32;
    for j in 0..n as u32 {
        for i in 0..n as u32 {
            let a = j * row + i;
            let (b, c, d) = (a + 1, a + row, a + row + 1);
            // Alternate the diagonal: valences 4..8 appear.
            if (i + j) % 2 == 0 {
                m.add_triangle(a, b, d);
                m.add_triangle(a, d, c);
            } else {
                m.add_triangle(a, b, c);
                m.add_triangle(b, d, c);
            }
        }
    }
    let sub = gpu_loop_subdivision(&gpu_loop_subdivision(&m));
    for p in &sub.vertices {
        assert!(
            (p[2] - plane_z(p[0], p[1])).abs() < 2e-4,
            "off the plane: {p:?}"
        );
    }
}

#[test]
fn test_loop_subdivision_non_manifold_edge_is_a_crease() {
    // Three triangles share the edge (0, 1): there is no opposite-vertex pair,
    // so the edge vertex is the plain midpoint.
    let mut m = GpuMesh::new();
    m.add_vertex([0.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, 0.0]);
    m.add_vertex([2.0, 0.0, 0.0], [0.0, 0.0, 1.0], [1.0, 0.0]);
    m.add_vertex([1.0, 1.0, 0.0], [0.0, 0.0, 1.0], [0.0, 1.0]);
    m.add_vertex([1.0, -1.0, 0.0], [0.0, 0.0, 1.0], [0.0, 1.0]);
    m.add_vertex([1.0, 0.0, 1.0], [0.0, 0.0, 1.0], [0.0, 1.0]);
    m.add_triangle(0, 1, 2);
    m.add_triangle(0, 1, 3);
    m.add_triangle(0, 1, 4);
    let sub = gpu_loop_subdivision(&m);
    assert_eq!(sub.triangle_count(), 12);
    // The shared edge is the first edge of triangle 0: its vertex follows the
    // five original ones.
    let shared = sub.indices[0][1] as usize;
    assert_eq!(shared, 5);
    assert_close3(sub.vertices[shared], [1.0, 0.0, 0.0], 1e-7, "shared edge");
    assert!(sub.vertices.iter().all(|p| p.iter().all(|c| c.is_finite())));
}

#[test]
fn test_loop_subdivision_skips_invalid_triangles() {
    let mut m = oriented_tetrahedron();
    let valid = m.triangle_count();
    // Out-of-range indices (the old code panicked on these) and a triangle with
    // a repeated vertex.
    m.add_triangle(0, 1, 99);
    m.add_triangle(100, 101, 102);
    m.add_triangle(2, 2, 3);
    let sub = gpu_loop_subdivision(&m);
    assert_eq!(sub.triangle_count(), 4 * valid);
    assert_clean_closed_manifold(&sub, 2);
}

#[test]
fn test_loop_subdivision_texture_coordinates_and_short_attribute_arrays() {
    let mut m = unit_triangle();
    m.tex_coords[1] = [1.0, 0.5];
    let sub = gpu_loop_subdivision(&m);
    // Originals keep their coordinates, edge vertices average the end points.
    assert_eq!(&sub.tex_coords[..3], &m.tex_coords[..]);
    assert_eq!(sub.tex_coords[3], [0.5, 0.25]);
    assert_eq!(sub.tex_coords[4], [0.5, 0.75]);
    assert_eq!(sub.tex_coords[5], [0.0, 0.5]);
    assert_eq!(sub.normals.len(), sub.vertex_count());
    assert_eq!(sub.tex_coords.len(), sub.vertex_count());

    // The public arrays may be shorter than the vertex list: no panic, missing
    // texture coordinates read as zero.
    let mut ragged = oriented_tetrahedron();
    ragged.tex_coords.truncate(1);
    ragged.normals.truncate(2);
    let sub = gpu_loop_subdivision(&ragged);
    assert_eq!(sub.vertex_count(), 10);
    assert_eq!(sub.tex_coords.len(), 10);
    assert_eq!(sub.tex_coords[0], ragged.tex_coords[0]);
    assert_eq!(sub.tex_coords[3], [0.0, 0.0]);
}

#[test]
fn test_loop_subdivision_recomputes_normals() {
    let sub = gpu_loop_subdivision(&oriented_tetrahedron());
    let centre = [0.25_f32, 0.25, 0.25];
    for (p, n) in sub.vertices.iter().zip(sub.normals.iter()) {
        assert!((length3f(*n) - 1.0).abs() < 1e-5, "normal {n:?}");
        // Outward on a convex solid: away from the centroid.
        let out = [p[0] - centre[0], p[1] - centre[1], p[2] - centre[2]];
        assert!(n[0] * out[0] + n[1] * out[1] + n[2] * out[2] > 0.0);
    }
    // The input normals (all +z) were not just copied.
    assert!(sub.normals.iter().any(|n| n[2] < 0.0));
}

#[test]
fn test_loop_subdivision_keeps_isolated_vertices_and_handles_empty_meshes() {
    // A vertex no triangle uses keeps its position, normal and UV.
    let mut m = oriented_tetrahedron();
    m.add_vertex([5.0, 6.0, 7.0], [0.0, 1.0, 0.0], [0.25, 0.75]);
    let sub = gpu_loop_subdivision(&m);
    assert_eq!(sub.vertices[4], [5.0, 6.0, 7.0]);
    assert_eq!(sub.normals[4], [0.0, 1.0, 0.0]);
    assert_eq!(sub.tex_coords[4], [0.25, 0.75]);
    assert_eq!(sub.triangle_count(), 16);

    // Vertices only: nothing moves.
    let mut points = GpuMesh::new();
    points.add_vertex([1.0, 2.0, 3.0], [1.0, 0.0, 0.0], [0.5, 0.5]);
    let same = gpu_loop_subdivision(&points);
    assert_eq!(same.vertices, points.vertices);
    assert_eq!(same.normals, points.normals);
    assert_eq!(same.triangle_count(), 0);

    let empty = gpu_loop_subdivision(&GpuMesh::new());
    assert_eq!((empty.vertex_count(), empty.triangle_count()), (0, 0));
}

// ═══════════════════════════════════════════════════════════════════════════
// Decimation
// ═══════════════════════════════════════════════════════════════════════════

/// The headline regression: a 20k-triangle sphere is decimated to 2k without
/// degenerate triangles, orphaned vertices or topology changes (the old code
/// rescanned and remapped every triangle per collapse and left the removed
/// vertices in the array).
#[test]
fn test_decimate_20k_triangle_sphere_to_2k_gives_a_clean_closed_manifold() {
    let sphere = icosphere(5);
    assert_eq!(sphere.triangle_count(), 20_480);
    assert_eq!(sphere.vertex_count(), 10_242);
    let reduced = gpu_mesh_decimate(&sphere, 2_000);

    assert_eq!(reduced.triangle_count(), 2_000);
    // A closed genus-0 triangle mesh has V = F/2 + 2.
    assert_eq!(reduced.vertex_count(), 1_002);
    assert_clean_closed_manifold(&reduced, 2);

    // Still a sphere: radii and volume close to the unit ball's, normals
    // outward and unit length, and the winding survived (volume > 0).
    let ball = 4.0 / 3.0 * std::f32::consts::PI;
    let volume = gpu_compute_volume(&reduced);
    assert!(volume > 0.97 * ball && volume < ball, "volume {volume}");
    for (p, n) in reduced.vertices.iter().zip(reduced.normals.iter()) {
        let radius = length3f(*p);
        assert!((0.93..1.0001).contains(&radius), "radius {radius}");
        assert!((length3f(*n) - 1.0).abs() < 1e-4);
        assert!((n[0] * p[0] + n[1] * p[1] + n[2] * p[2]) / radius > 0.9);
    }
    // No sliver triangles (quality 4 sqrt(3) A / sum of squared edges).
    let mut worst = f32::MAX;
    for tri in &reduced.indices {
        let [a, b, c] = tri.map(|v| reduced.vertices[v as usize]);
        let edges_sq = dist2_3f(a, b) + dist2_3f(b, c) + dist2_3f(c, a);
        worst = worst.min(4.0 * 3.0_f32.sqrt() * triangle_area(a, b, c) / edges_sq);
    }
    assert!(worst > 0.2, "worst triangle quality {worst}");
}

#[test]
fn test_decimate_is_deterministic() {
    let sphere = icosphere(4);
    let first = gpu_mesh_decimate(&sphere, 500);
    let second = gpu_mesh_decimate(&sphere, 500);
    assert_eq!(first.vertices, second.vertices);
    assert_eq!(first.indices, second.indices);
    assert_eq!(first.tex_coords, second.tex_coords);
}

#[test]
fn test_decimate_returns_the_input_when_it_is_small_enough() {
    let mut m = oriented_tetrahedron();
    m.add_vertex([9.0, 9.0, 9.0], [0.0, 0.0, 1.0], [0.0, 0.0]); // unused
    m.add_triangle(0, 0, 1); // degenerate
    for target in [m.triangle_count(), m.triangle_count() + 10, usize::MAX] {
        let same = gpu_mesh_decimate(&m, target);
        assert_eq!(same.vertices, m.vertices);
        assert_eq!(same.indices, m.indices);
        assert_eq!(same.normals, m.normals);
        assert_eq!(same.tex_coords, m.tex_coords);
    }
}

#[test]
fn test_decimate_never_collapses_a_closed_mesh_below_a_tetrahedron() {
    for target in [0, 1, 3] {
        let tetra = gpu_mesh_decimate(&oriented_tetrahedron(), target);
        assert_clean_closed_manifold(&tetra, 2);
        assert_eq!(tetra.triangle_count(), 4);
        assert!((gpu_compute_volume(&tetra) - 1.0 / 6.0).abs() < 1e-6);

        let small = gpu_mesh_decimate(&icosphere(2), target);
        assert_clean_closed_manifold(&small, 2);
        assert_eq!(small.triangle_count(), 4, "target {target}");
    }
}

/// Collapsing an edge whose end points have a common neighbour that is not
/// opposite the edge would pinch the surface.  In a triangular bipyramid the
/// three shortest edges are the equator edges, all of which violate the link
/// condition; the valid collapse merges an apex into an equator vertex.
#[test]
fn test_decimate_respects_the_link_condition() {
    let mut m = GpuMesh::new();
    // Equator (short edges), apexes far away.
    let radius = 0.3_f32;
    for k in 0..3 {
        let angle = k as f32 * std::f32::consts::TAU / 3.0;
        m.add_vertex(
            [radius * angle.cos(), radius * angle.sin(), 0.0],
            [0.0; 3],
            [0.0, 0.0],
        );
    }
    m.add_vertex([0.0, 0.0, 1.5], [0.0; 3], [0.0, 0.0]);
    m.add_vertex([0.0, 0.0, -1.5], [0.0; 3], [0.0, 0.0]);
    for k in 0..3_u32 {
        m.add_triangle(k, (k + 1) % 3, 3);
        m.add_triangle((k + 1) % 3, k, 4);
    }
    assert_clean_closed_manifold(&m, 2);

    let reduced = gpu_mesh_decimate(&m, 4);
    assert_clean_closed_manifold(&reduced, 2);
    assert_eq!(reduced.triangle_count(), 4);
    // An equator collapse would have left two coincident triangles instead.
    assert!(gpu_compute_volume(&reduced) > 0.0);
}

#[test]
fn test_decimate_planar_grid_stays_flat_and_keeps_its_boundary() {
    let n = 20;
    let grid = flat_grid(n);
    let reduced = gpu_mesh_decimate(&grid, 100);
    assert!(reduced.triangle_count() <= 100);
    let t = topology(&reduced);
    assert_eq!(t.out_of_range + t.degenerate + t.unreferenced, 0, "{t:?}");
    assert_eq!(t.non_manifold_edges, 0);
    assert_eq!(t.euler(), 1, "a disk must stay a disk: {t:?}");

    // Flat stays flat; the texture coordinates follow the merges (they are an
    // affine function of position, preserved by midpoint merging).
    let n = n as f32;
    for (p, uv) in reduced.vertices.iter().zip(reduced.tex_coords.iter()) {
        assert_eq!(p[2], 0.0);
        assert!((uv[0] - p[0] / n).abs() < 1e-5 && (uv[1] - p[1] / n).abs() < 1e-5);
    }
    // Interior vertices only ever merge with interior vertices, so they stay
    // in the inner square [1, n-1]^2; boundary vertices stay in [0, n]^2.
    let mut on_boundary = vec![false; reduced.vertex_count()];
    let mut edge_faces: HashMap<(u32, u32), usize> = HashMap::new();
    for tri in &reduced.indices {
        for slot in 0..3 {
            let (a, b) = (tri[slot], tri[(slot + 1) % 3]);
            *edge_faces.entry((a.min(b), a.max(b))).or_insert(0) += 1;
        }
    }
    for (&(a, b), &faces) in &edge_faces {
        if faces == 1 {
            on_boundary[a as usize] = true;
            on_boundary[b as usize] = true;
        }
    }
    assert!(on_boundary.iter().any(|&b| b), "the patch keeps a boundary");
    for (p, &boundary) in reduced.vertices.iter().zip(on_boundary.iter()) {
        let (low, high) = if boundary { (0.0, n) } else { (1.0, n - 1.0) };
        for coordinate in &p[..2] {
            assert!(
                *coordinate >= low - 1e-4 && *coordinate <= high + 1e-4,
                "vertex {p:?} (boundary: {boundary})"
            );
        }
    }
    // Collapsing a boundary edge merges at its midpoint, which rounds off the
    // corners of the outline a little but keeps every side of it in place: the
    // mid-side vertices are still on the sides.
    let on_a_side = |p: &[f32; 3]| {
        p[0].abs() < 1e-4 || (p[0] - n).abs() < 1e-4 || p[1].abs() < 1e-4 || (p[1] - n).abs() < 1e-4
    };
    let side_vertices = reduced
        .vertices
        .iter()
        .zip(on_boundary.iter())
        .filter(|&(p, &boundary)| boundary && on_a_side(p))
        .count();
    assert!(
        side_vertices >= 8,
        "only {side_vertices} boundary vertices left on the sides"
    );
    // The covered area hardly changes (midpoint merging on the outline eats at
    // most a few percent).
    assert!(gpu_compute_surface_area(&reduced) > 0.9 * n * n);
}

#[test]
fn test_decimate_preserves_genus() {
    let ring = torus(40, 20);
    let reduced = gpu_mesh_decimate(&ring, 600);
    assert!(reduced.triangle_count() <= 600);
    assert_clean_closed_manifold(&reduced, 0);
}

#[test]
fn test_decimate_sphere_with_high_valence_poles() {
    let sphere = uv_sphere(48, 24);
    let total = sphere.triangle_count();
    assert_clean_closed_manifold(&sphere, 2);
    for target in [total / 2, total / 6, 64] {
        let reduced = gpu_mesh_decimate(&sphere, target);
        assert!(reduced.triangle_count() <= target);
        assert_clean_closed_manifold(&reduced, 2);
        assert!(gpu_compute_volume(&reduced) > 0.0);
    }
}

#[test]
fn test_decimate_drops_invalid_triangles_and_compacts() {
    let mut m = icosphere(2);
    m.add_vertex([7.0, 7.0, 7.0], [0.0, 0.0, 1.0], [0.0, 0.0]); // unreferenced
    m.add_triangle(0, 1, 5000); // out of range: used to panic
    m.add_triangle(3, 3, 4); // degenerate
    let reduced = gpu_mesh_decimate(&m, 100);
    assert!(reduced.triangle_count() <= 100);
    assert_clean_closed_manifold(&reduced, 2);
}

#[test]
fn test_decimate_texture_coordinates_follow_the_merged_vertices() {
    let reduced = gpu_mesh_decimate(&icosphere(3), 200);
    assert_eq!(reduced.tex_coords.len(), reduced.vertex_count());
    assert_eq!(reduced.normals.len(), reduced.vertex_count());
    assert_clean_closed_manifold(&reduced, 2);
}

// ═══════════════════════════════════════════════════════════════════════════
// Collapse rules (`Decimator::plan_collapse`)
// ═══════════════════════════════════════════════════════════════════════════

/// The collapse plan for merging `remove` into `keep` in `mesh`, if allowed.
fn plan(mesh: &GpuMesh, keep: u32, remove: u32) -> Option<CollapsePlan> {
    Decimator::new(mesh).plan_collapse(keep, remove)
}

/// Triangular bipyramid: apexes `p = 0` (top) and `q = 4` (bottom), equator
/// `a = 1`, `b = 2`, `c = 3`.  `a` is configurable so that collapsing the apex
/// edge `(p, a)` can be made harmless or fold the triangle `(p, b, c)`.
fn bipyramid(a: [f32; 3]) -> GpuMesh {
    let mut m = GpuMesh::new();
    for p in [
        [0.0, 0.0, 1.0],
        a,
        [1.0, -1.0, 0.0],
        [1.0, 1.0, 0.0],
        [0.0, 0.0, -1.0],
    ] {
        m.add_vertex(p, [0.0; 3], [0.0, 0.0]);
    }
    for (x, y, z) in [
        (0, 1, 2),
        (0, 2, 3),
        (0, 3, 1),
        (4, 2, 1),
        (4, 3, 2),
        (4, 1, 3),
    ] {
        m.add_triangle(x, y, z);
    }
    m
}

#[test]
fn test_triangle_survives_collapse_rejects_flips_and_slivers() {
    let old = [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0_f32]];
    assert!(triangle_survives_collapse(old, old));
    // Turning the normal by 30 degrees (about the x axis) is fine, 80 is not.
    let tilt = |angle: f32| {
        let (s, c) = angle.sin_cos();
        [old[0], old[1], [0.0, c, s]]
    };
    assert!(triangle_survives_collapse(old, tilt(30.0_f32.to_radians())));
    assert!(!triangle_survives_collapse(
        old,
        tilt(80.0_f32.to_radians())
    ));
    // A reflected (inside-out) triangle.
    let flipped = [old[0], old[2], old[1]];
    assert!(!triangle_survives_collapse(old, flipped));
    // Collapsed to a line, to a point, or not finite.
    let sliver = [old[0], old[1], [2.0, 0.0, 0.0]];
    assert!(!triangle_survives_collapse(old, sliver));
    assert!(!triangle_survives_collapse(old, [old[0]; 3]));
    let broken = [old[0], old[1], [f32::NAN, 0.0, 0.0]];
    assert!(!triangle_survives_collapse(old, broken));
    // An input sliver has no orientation to preserve.
    assert!(triangle_survives_collapse(sliver, old));
    assert!(triangle_survives_collapse(sliver, sliver));
}

#[test]
fn test_collapse_refuses_a_flipping_triangle() {
    // Apex edge (p, a) with `a` far out: the midpoint lies beyond the edge
    // (b, c), which would turn the triangle (p, b, c) inside out.
    let folded = bipyramid([5.0, 0.0, 0.0]);
    assert_clean_closed_manifold(&folded, 2);
    assert!(plan(&folded, 0, 1).is_none());
    // The same topology with `a` close by merges fine, at the midpoint.
    let gentle = bipyramid([1.5, 0.0, 0.0]);
    let merged = plan(&gentle, 0, 1).expect("harmless collapse");
    assert_close3(merged.position, [0.75, 0.0, 0.5], 1e-6, "midpoint");
}

#[test]
fn test_collapse_refuses_the_equator_of_a_bipyramid() {
    // Link condition: the equator vertices share the third equator vertex,
    // which is not opposite the edge.
    let m = bipyramid([1.5, 0.0, 0.0]);
    for (keep, remove) in [(1, 2), (2, 3), (1, 3)] {
        assert!(
            plan(&m, keep, remove).is_none(),
            "equator edge {keep}-{remove}"
        );
    }
    // The apex edges are fine.
    for (keep, remove) in [(0, 1), (0, 2), (0, 3), (1, 4), (2, 4), (3, 4)] {
        assert!(
            plan(&m, keep, remove).is_some(),
            "apex edge {keep}-{remove}"
        );
    }
}

#[test]
fn test_collapse_refuses_every_edge_of_a_closed_tetrahedron() {
    let tetra = oriented_tetrahedron();
    for keep in 0..4_u32 {
        for remove in (keep + 1)..4 {
            assert!(plan(&tetra, keep, remove).is_none(), "edge {keep}-{remove}");
        }
    }
}

#[test]
fn test_collapse_refuses_non_manifold_edges_and_missing_edges() {
    // Three triangles share the edge (0, 1).
    let mut book = GpuMesh::new();
    for p in [
        [0.0, 0.0, 0.0],
        [2.0, 0.0, 0.0],
        [1.0, 1.0, 0.0],
        [1.0, -1.0, 0.0],
        [1.0, 0.0, 1.0_f32],
    ] {
        book.add_vertex(p, [0.0; 3], [0.0, 0.0]);
    }
    for apex in 2..5 {
        book.add_triangle(0, 1, apex);
    }
    assert!(plan(&book, 0, 1).is_none());
    // Vertices 2 and 3 are not joined by an edge.
    assert!(plan(&book, 2, 3).is_none());
}

#[test]
fn test_collapse_joining_two_boundary_vertices_needs_a_boundary_edge() {
    // One square with a diagonal: all four vertices are on the boundary, so
    // collapsing the interior diagonal would pinch the square flat; collapsing
    // a side is fine.
    let square = flat_grid(1);
    let diagonal = plan(&square, 0, 3);
    assert!(diagonal.is_none(), "the diagonal (0, 3) must be refused");
    let side = plan(&square, 0, 1).expect("a boundary edge may collapse");
    assert_close3(side.position, [0.5, 0.0, 0.0], 1e-6, "side midpoint");
    assert_eq!(side.uv, [0.5, 0.0]);
}

#[test]
fn test_collapse_places_the_merged_vertex_on_a_boundary_endpoint() {
    let grid = flat_grid(3); // 4 x 4 vertices, index = i + 4 j
    // boundary (1, 0) = 1 absorbs interior (1, 1) = 5: stays at (1, 0).
    let onto_boundary = plan(&grid, 1, 5).expect("valid collapse");
    assert_eq!(onto_boundary.position, [1.0, 0.0, 0.0]);
    assert_eq!(onto_boundary.uv, grid.tex_coords[1]);
    // Interior (2, 2) = 10 with boundary (3, 2) = 11: the boundary vertex wins
    // although it has the larger index.
    let boundary_second = plan(&grid, 10, 11).expect("valid collapse");
    assert_eq!(boundary_second.position, [3.0, 2.0, 0.0]);
    assert_eq!(boundary_second.uv, grid.tex_coords[11]);
    // Two interior vertices: the midpoint, with the averaged UV.
    let interior = plan(&grid, 5, 6).expect("valid collapse");
    assert_close3(
        interior.position,
        [1.5, 1.0, 0.0],
        1e-6,
        "interior midpoint",
    );
    assert!((interior.uv[0] - 0.5).abs() < 1e-6 && (interior.uv[1] - 1.0 / 3.0).abs() < 1e-6);
}

// ═══════════════════════════════════════════════════════════════════════════
// Edge collapse and union-find
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn test_find_root_compresses_paths_by_halving() {
    // A chain 0 <- 1 <- 2 <- ... <- 8 (parent of k is k - 1).
    let mut parent: Vec<usize> = (0..9_usize).map(|k| k.saturating_sub(1)).collect();
    assert_eq!(find_root(&mut parent, 8), 0);
    // Path halving re-points visited nodes at their grandparents.
    assert_eq!(parent[8], 6);
    assert_eq!(parent[6], 4);
    assert_eq!(parent[4], 2);
    assert_eq!(parent[2], 0);
    // Roots map to themselves; a second lookup halves the shortened path again.
    assert_eq!(find_root(&mut parent, 0), 0);
    assert_eq!(find_root(&mut parent, 8), 0);
    assert_eq!(parent[8], 4);
    assert_eq!(parent[4], 0);
    assert_eq!(find_root(&mut parent, 8), 0);
    assert_eq!(parent[8], 0);
    // Singletons.
    let mut lone = vec![0_usize, 1, 2];
    assert_eq!(find_root(&mut lone, 2), 2);
}

#[test]
fn test_find_root_on_a_deep_chain_is_cheap() {
    let n = 1_000_000_usize;
    let mut parent: Vec<usize> = (0..n).map(|k: usize| k.saturating_sub(1)).collect();
    // Repeated lookups from the far end: with halving the work drops
    // geometrically instead of repeating n steps each time.
    for _ in 0..40 {
        assert_eq!(find_root(&mut parent, n - 1), 0);
    }
}

#[test]
fn test_edge_collapse_merges_a_long_strip() {
    // A strip of 20k triangles whose vertices are all within the threshold.
    let m = 10_000;
    let mut mesh = GpuMesh::new();
    for k in 0..m {
        let x = k as f32 * 1.0e-3;
        mesh.add_vertex([x, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, 0.0]);
        mesh.add_vertex([x, 1.0e-3, 0.0], [0.0, 0.0, 1.0], [0.0, 0.0]);
    }
    for k in 0..(m - 1) as u32 {
        mesh.add_triangle(2 * k, 2 * k + 1, 2 * k + 2);
        mesh.add_triangle(2 * k + 1, 2 * k + 3, 2 * k + 2);
    }
    let removed = gpu_edge_collapse(&mut mesh, 1.0);
    assert_eq!(removed, 2 * m - 1);
    assert_eq!(mesh.triangle_count(), 0);
}

#[test]
fn test_edge_collapse_drops_triangles_with_missing_vertices() {
    let mut m = unit_triangle();
    m.add_triangle(0, 1, 40); // used to panic
    m.add_triangle(7, 8, 9);
    let removed = gpu_edge_collapse(&mut m, 0.5);
    assert_eq!(removed, 0);
    assert_eq!(m.indices, vec![[0, 1, 2]]);
}

#[test]
fn test_edge_collapse_remaps_and_removes_degenerate_triangles() {
    // Two triangles sharing an edge shorter than the threshold.
    let mut m = GpuMesh::new();
    m.add_vertex([0.0, 0.0, 0.0], [0.0; 3], [0.0, 0.0]);
    m.add_vertex([0.01, 0.0, 0.0], [0.0; 3], [0.0, 0.0]);
    m.add_vertex([0.0, 1.0, 0.0], [0.0; 3], [0.0, 0.0]);
    m.add_vertex([1.0, 1.0, 0.0], [0.0; 3], [0.0, 0.0]);
    m.add_triangle(0, 1, 2);
    m.add_triangle(1, 3, 2);
    let removed = gpu_edge_collapse(&mut m, 0.1);
    assert_eq!(removed, 1);
    // Vertex 1 merged into 0: the first triangle degenerated, the second now
    // starts at vertex 0.
    assert_eq!(m.indices, vec![[0, 3, 2]]);
}

// ═══════════════════════════════════════════════════════════════════════════
// Vertex welding
// ═══════════════════════════════════════════════════════════════════════════

/// The original quadratic algorithm (same `distance² < tol²` predicate): every
/// vertex that is still unmerged absorbs the later unmerged vertices closer
/// than `tol`.
fn reference_weld(vertices: &[[f32; 3]], tol: f32) -> Vec<usize> {
    let n = vertices.len();
    let tol_sq = tol * tol;
    let mut remap: Vec<usize> = (0..n).collect();
    for i in 0..n {
        if remap[i] != i {
            continue;
        }
        for j in (i + 1)..n {
            if remap[j] == j && dist2_3f(vertices[i], vertices[j]) < tol_sq {
                remap[j] = i;
            }
        }
    }
    remap
}

/// Jittered duplicates: `bases` random points, each copied `copies` times with
/// a displacement of at most `jitter` per axis, plus random triangles.
fn duplicated_cloud(bases: usize, copies: usize, jitter: f32, seed: u64) -> GpuMesh {
    let mut rng = UnitRng(seed);
    let mut m = GpuMesh::new();
    for _ in 0..bases {
        let base = [rng.next() * 10.0, rng.next() * 10.0, rng.next() * 10.0];
        for _ in 0..copies {
            let p = [
                base[0] + (rng.next() * 2.0 - 1.0) * jitter,
                base[1] + (rng.next() * 2.0 - 1.0) * jitter,
                base[2] + (rng.next() * 2.0 - 1.0) * jitter,
            ];
            m.add_vertex(p, [0.0, 0.0, 1.0], [0.0, 0.0]);
        }
    }
    let nv = m.vertex_count() as f32;
    for _ in 0..(bases * copies) {
        let pick = |r: &mut UnitRng| (r.next() * nv) as u32;
        let (a, b, c) = (pick(&mut rng), pick(&mut rng), pick(&mut rng));
        m.add_triangle(a, b, c);
    }
    m
}

fn reference_welded_indices(m: &GpuMesh, remap: &[usize]) -> Vec<[u32; 3]> {
    m.indices
        .iter()
        .map(|t| t.map(|v| remap[v as usize] as u32))
        .filter(|t| t[0] != t[1] && t[1] != t[2] && t[0] != t[2])
        .collect()
}

/// 10k jittered duplicates weld exactly like the O(V²) reference.
#[test]
fn test_weld_matches_the_quadratic_reference() {
    // Jitter 0.2 tol per axis: copies of one point are within 0.7 tol.
    let tol = 0.05;
    let mut m = duplicated_cloud(2_500, 4, 0.2 * tol, 7);
    assert_eq!(m.vertex_count(), 10_000);
    let remap = reference_weld(&m.vertices, tol);
    let expected_merged = remap.iter().enumerate().filter(|&(i, &r)| r != i).count();
    let expected_indices = reference_welded_indices(&m, &remap);
    let merged = gpu_weld_vertices(&mut m, tol);
    assert_eq!(merged, expected_merged);
    assert!(merged >= 7_000, "most duplicates must weld: {merged}");
    assert_eq!(m.indices, expected_indices);
    // Welded vertices stay in the arrays but are unreferenced.
    assert_eq!(m.vertex_count(), 10_000);
}

#[test]
fn test_weld_matches_the_reference_on_clusters_chains_and_boundaries() {
    // Dense random cloud (many near-ties), then a collinear chain whose
    // greedy grouping depends on the visiting order.
    let tol = 0.5;
    let mut rng = UnitRng(99);
    let mut m = GpuMesh::new();
    for _ in 0..1_500 {
        m.add_vertex(
            [rng.next() * 6.0, rng.next() * 6.0, rng.next() * 6.0],
            [0.0; 3],
            [0.0, 0.0],
        );
    }
    // Chain with spacing 0.6 tol: 0 absorbs 1, but 2 (1.2 tol away) survives.
    for k in 0..6 {
        m.add_vertex([20.0 + 0.3 * k as f32, 0.0, 0.0], [0.0; 3], [0.0, 0.0]);
    }
    // Points exactly tol apart do not weld (strictly below).
    m.add_vertex([40.0, 0.0, 0.0], [0.0; 3], [0.0, 0.0]);
    m.add_vertex([40.5, 0.0, 0.0], [0.0; 3], [0.0, 0.0]);
    // Cell-boundary straddlers at negative coordinates.
    m.add_vertex([-0.001, -0.001, -0.001], [0.0; 3], [0.0, 0.0]);
    m.add_vertex([0.001, 0.001, 0.001], [0.0; 3], [0.0, 0.0]);
    let remap = reference_weld(&m.vertices, tol);
    let expected_merged = remap.iter().enumerate().filter(|&(i, &r)| r != i).count();
    let mut welded = m.clone();
    assert_eq!(gpu_weld_vertices(&mut welded, tol), expected_merged);
    // Recompute the chain/boundary expectations explicitly.
    let n = m.vertex_count();
    assert_eq!(remap[n - 4], n - 4, "points exactly tol apart stay apart");
    assert_eq!(remap[n - 3], n - 3);
    assert_eq!(remap[n - 1], n - 2, "straddlers across the origin weld");
}

#[test]
fn test_weld_tolerance_must_be_positive_and_finite() {
    for tol in [0.0, -1.0, f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
        let mut m = GpuMesh::new();
        for _ in 0..3 {
            m.add_vertex([0.0; 3], [0.0; 3], [0.0, 0.0]);
        }
        m.add_triangle(0, 1, 2);
        assert_eq!(gpu_weld_vertices(&mut m, tol), 0, "tol {tol}");
        assert_eq!(m.indices, vec![[0, 1, 2]]);
    }
}

#[test]
fn test_weld_handles_huge_and_non_finite_coordinates() {
    let mut m = GpuMesh::new();
    for p in [
        [1.0e30, 1.0e30, 1.0e30],
        [1.0e30, 1.0e30, 1.0e30],
        [-1.0e30, 0.0, 0.0],
        [f32::INFINITY, 0.0, 0.0],
        [f32::INFINITY, 0.0, 0.0],
        [f32::NAN, 0.0, 0.0],
        [f32::NAN, 0.0, 0.0],
        [0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0],
    ] {
        m.add_vertex(p, [0.0; 3], [0.0, 0.0]);
    }
    // Tiny tolerance: cell coordinates saturate instead of overflowing.
    let merged = gpu_weld_vertices(&mut m, 1.0e-6);
    // Only the two identical huge points and the two origins weld: infinities
    // and NaNs never compare as close.
    assert_eq!(merged, 2);
}

#[test]
fn test_weld_remaps_triangles_and_drops_degenerate_and_invalid_ones() {
    let mut m = GpuMesh::new();
    m.add_vertex([0.0, 0.0, 0.0], [0.0; 3], [0.0, 0.0]);
    m.add_vertex([0.001, 0.0, 0.0], [0.0; 3], [0.0, 0.0]); // welds into 0
    m.add_vertex([1.0, 0.0, 0.0], [0.0; 3], [0.0, 0.0]);
    m.add_vertex([0.0, 1.0, 0.0], [0.0; 3], [0.0, 0.0]);
    m.add_triangle(1, 2, 3);
    m.add_triangle(0, 1, 2); // degenerate after welding
    m.add_triangle(0, 2, 50); // missing vertex: dropped
    let merged = gpu_weld_vertices(&mut m, 0.01);
    assert_eq!(merged, 1);
    assert_eq!(m.indices, vec![[0, 2, 3]]);
}

/// 100k vertices: the spatial hash makes this instantaneous (the all-pairs
/// loop needed about 5e9 distance tests).
#[test]
fn test_weld_scales_to_large_vertex_counts() {
    let tol = 0.02;
    let mut m = duplicated_cloud(25_000, 4, 0.1 * tol, 3);
    assert_eq!(m.vertex_count(), 100_000);
    let merged = gpu_weld_vertices(&mut m, tol);
    // Copies of one point are within 0.35 tol of each other: roughly three
    // quarters of the vertices are duplicates.
    // At most three of four copies merge away; a few unrelated points happen
    // to land within the tolerance of each other as well.
    assert!(merged >= 70_000, "merged {merged}");
    assert!(merged <= 76_000, "merged {merged}");
    let t = topology(&m);
    assert_eq!(t.out_of_range + t.degenerate, 0);
}
