// Copyright 2026 COOLJAPAN OU (Team KitaSan)
// SPDX-License-Identifier: Apache-2.0

//! Regression tests for the terrain collision audit findings: marching cubes
//! isosurface extraction (F0291) and exact heightfield ray casting (F0295).

use std::collections::HashMap;

use super::marching_cubes::{
    MC_CORNER_OFFSETS, MC_EDGE_ASCENDING, MC_EDGE_CORNERS, MC_EDGE_TABLE, MC_TRI_TABLE,
};
use super::*;

fn approx(a: f64, b: f64, tol: f64) -> bool {
    (a - b).abs() <= tol
}

fn approx3(a: [f64; 3], b: [f64; 3], tol: f64) -> bool {
    (0..3).all(|k| approx(a[k], b[k], tol))
}

/// Deterministic pseudo random numbers (64-bit LCG), so the tests need no RNG crate.
struct Lcg(u64);

impl Lcg {
    fn next_f64(&mut self) -> f64 {
        self.0 = self
            .0
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        (self.0 >> 11) as f64 / (1_u64 << 53) as f64
    }

    fn range(&mut self, lo: f64, hi: f64) -> f64 {
        lo + (hi - lo) * self.next_f64()
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Marching cubes tables (F0291)
// ─────────────────────────────────────────────────────────────────────────────

/// A cube face: `(axis, side)` with `side` in `{0, 1}`.
type Face = (usize, usize);

const ALL_FACES: [Face; 6] = [(0, 0), (0, 1), (1, 0), (1, 1), (2, 0), (2, 1)];

/// The triangles of `case` parsed from the raw table row.
fn case_triangles(case: usize) -> Vec<[usize; 3]> {
    let row = &MC_TRI_TABLE[case];
    let used = row.iter().take_while(|&&e| e != -1).count();
    assert_eq!(used % 3, 0, "case {case}: edge list is not a multiple of 3");
    assert!(
        row[used..].iter().all(|&e| e == -1),
        "case {case}: entries after the terminator must all be -1"
    );
    row[..used]
        .chunks(3)
        .map(|t| {
            let edge = |k: usize| usize::try_from(t[k]).expect("edge index is non-negative");
            [edge(0), edge(1), edge(2)]
        })
        .collect()
}

/// The two faces of the cube that contain edge `e`.
fn faces_of_edge(e: usize) -> Vec<Face> {
    let [a, b] = MC_EDGE_CORNERS[e];
    ALL_FACES
        .into_iter()
        .filter(|&(axis, side)| {
            MC_CORNER_OFFSETS[a][axis] == side && MC_CORNER_OFFSETS[b][axis] == side
        })
        .collect()
}

fn common_face(e1: usize, e2: usize) -> Option<Face> {
    let second = faces_of_edge(e2);
    faces_of_edge(e1).into_iter().find(|f| second.contains(f))
}

/// Edge mask derived from the cube topology alone.
fn derived_edge_mask(case: usize) -> u16 {
    let mut mask = 0u16;
    for (e, [a, b]) in MC_EDGE_CORNERS.iter().enumerate() {
        if ((case >> a) & 1) != ((case >> b) & 1) {
            mask |= 1 << e;
        }
    }
    mask
}

/// Directed boundary sides of the patch of `case`, each with the cube face it
/// lies on.  Also checks that every other side is shared by exactly two
/// triangles that traverse it in opposite directions.
fn boundary_sides(case: usize) -> Vec<(usize, usize, Face)> {
    let mut sides: Vec<(usize, usize)> = Vec::new();
    for tri in case_triangles(case) {
        for s in 0..3 {
            sides.push((tri[s], tri[(s + 1) % 3]));
        }
    }
    let count = |a: usize, b: usize| sides.iter().filter(|&&s| s == (a, b)).count();
    let mut boundary = Vec::new();
    for &(a, b) in &sides {
        let (forward, reverse) = (count(a, b), count(b, a));
        if reverse > 0 {
            assert_eq!(
                (forward, reverse),
                (1, 1),
                "case {case}: side {a}->{b} is not shared by exactly two opposite triangles"
            );
        } else {
            assert_eq!(forward, 1, "case {case}: boundary side {a}->{b} repeated");
            let face = common_face(a, b).unwrap_or_else(|| {
                panic!("case {case}: boundary side {a}->{b} does not lie on a cube face")
            });
            boundary.push((a, b, face));
        }
    }
    boundary
}

#[test]
fn edge_table_matches_the_cube_topology() {
    for (case, &mask) in MC_EDGE_TABLE.iter().enumerate() {
        assert_eq!(
            mask,
            derived_edge_mask(case),
            "case {case}: edge mask disagrees with the corner signs"
        );
        assert_eq!(
            mask,
            MC_EDGE_TABLE[255 - case],
            "case {case}: a case and its complement cross the same edges"
        );
    }
}

#[test]
fn triangle_rows_use_exactly_the_crossed_edges() {
    for (case, &mask) in MC_EDGE_TABLE.iter().enumerate() {
        let triangles = case_triangles(case);
        assert!(
            triangles.len() <= 5,
            "case {case}: {} triangles",
            triangles.len()
        );
        let mut used = 0u16;
        for tri in &triangles {
            assert!(
                tri[0] != tri[1] && tri[1] != tri[2] && tri[0] != tri[2],
                "case {case}: degenerate triangle {tri:?}"
            );
            for &e in tri {
                assert!(e < 12, "case {case}: edge {e} out of range");
                used |= 1 << e;
            }
        }
        assert_eq!(
            used, mask,
            "case {case}: triangles use other edges than the edge table"
        );
    }
    assert!(case_triangles(0).is_empty() && case_triangles(255).is_empty());
    for corner in 0..8 {
        assert_eq!(case_triangles(1 << corner).len(), 1, "corner {corner}");
        assert_eq!(
            case_triangles(255 ^ (1 << corner)).len(),
            1,
            "corner {corner}"
        );
    }
}

#[test]
fn ascending_edge_corners_run_from_the_lower_to_the_upper_lattice_end() {
    for (e, &[lo, hi]) in MC_EDGE_ASCENDING.iter().enumerate() {
        let [a, b] = MC_EDGE_CORNERS[e];
        let mut joined = [a, b];
        joined.sort_unstable();
        let mut ascending = [lo, hi];
        ascending.sort_unstable();
        assert_eq!(joined, ascending, "edge {e} must join the same two corners");
        let differing: Vec<usize> = (0..3)
            .filter(|&k| MC_CORNER_OFFSETS[lo][k] != MC_CORNER_OFFSETS[hi][k])
            .collect();
        assert_eq!(differing.len(), 1, "edge {e} runs along one axis");
        let axis = differing[0];
        assert_eq!(MC_CORNER_OFFSETS[lo][axis], 0, "edge {e} lower end");
        assert_eq!(MC_CORNER_OFFSETS[hi][axis], 1, "edge {e} upper end");
    }
}

/// Every case is an oriented patch whose boundary runs along the cube faces:
/// on each face the boundary sides pair up exactly the crossed edges of that face.
#[test]
fn every_case_is_a_patch_bounded_by_cube_faces() {
    for (case, &mask) in MC_EDGE_TABLE.iter().enumerate() {
        let boundary = boundary_sides(case);
        for face in ALL_FACES {
            let crossed: Vec<usize> = (0..12)
                .filter(|&e| mask & (1 << e) != 0 && faces_of_edge(e).contains(&face))
                .collect();
            for &e in &crossed {
                let touching = boundary
                    .iter()
                    .filter(|&&(a, b, f)| f == face && (a == e || b == e))
                    .count();
                assert_eq!(
                    touching, 1,
                    "case {case}: crossed edge {e} meets {touching} boundary sides on face {face:?}"
                );
            }
            let on_face = boundary.iter().filter(|&&(_, _, f)| f == face).count();
            assert_eq!(on_face * 2, crossed.len(), "case {case}: face {face:?}");
        }
    }
}

/// What a cube face looks like from the outside: the segments of the isosurface
/// on it, as unordered pairs of the face edges they join.  Edges are named by
/// their two end points with the face axis dropped, so opposite faces of
/// neighbouring cubes use the same names.
type FaceSegments = Vec<[[[usize; 2]; 2]; 2]>;

fn face_segments(case: usize, face: Face) -> FaceSegments {
    let (axis, _) = face;
    let name = |e: usize| -> [[usize; 2]; 2] {
        let [a, b] = MC_EDGE_CORNERS[e];
        let drop_axis = |c: usize| {
            let o = MC_CORNER_OFFSETS[c];
            match axis {
                0 => [o[1], o[2]],
                1 => [o[0], o[2]],
                _ => [o[0], o[1]],
            }
        };
        let (pa, pb) = (drop_axis(a), drop_axis(b));
        if pa <= pb { [pa, pb] } else { [pb, pa] }
    };
    let mut segments: FaceSegments = boundary_sides(case)
        .into_iter()
        .filter(|&(_, _, f)| f == face)
        .map(|(a, b, _)| {
            let (na, nb) = (name(a), name(b));
            if na <= nb { [na, nb] } else { [nb, na] }
        })
        .collect();
    segments.sort_unstable();
    segments
}

/// The four corner bits of a face as a pattern index, in face-local order.
fn face_pattern(case: usize, face: Face) -> usize {
    let (axis, side) = face;
    let mut pattern = 0usize;
    let mut slot = 0;
    for u in 0..2 {
        for v in 0..2 {
            let corner = (0..8)
                .find(|&c| {
                    let o = MC_CORNER_OFFSETS[c];
                    let (first, second) = match axis {
                        0 => (o[1], o[2]),
                        1 => (o[0], o[2]),
                        _ => (o[0], o[1]),
                    };
                    o[axis] == side && first == u && second == v
                })
                .expect("every face has four corners");
            pattern |= ((case >> corner) & 1) << slot;
            slot += 1;
        }
    }
    pattern
}

/// Two cubes that share a face must triangulate it identically, otherwise
/// cracks open between them.  The segments on a face may therefore only depend
/// on the four corner signs of the face, not on the rest of the cube or on which
/// side of the face the cube is on.
#[test]
fn neighbouring_cubes_agree_on_every_shared_face() {
    let mut seen: HashMap<(usize, usize), (usize, Face, FaceSegments)> = HashMap::new();
    for case in 0..256 {
        for face in ALL_FACES {
            let axis = face.0;
            let key = (axis, face_pattern(case, face));
            let segments = face_segments(case, face);
            match seen.get(&key) {
                Some((other_case, other_face, other)) => assert_eq!(
                    other, &segments,
                    "case {case} (face {face:?}) and case {other_case} (face {other_face:?}) \
                     triangulate the same face pattern differently"
                ),
                None => {
                    seen.insert(key, (case, face, segments));
                }
            }
        }
    }
    // 3 axes x 16 corner patterns.
    assert_eq!(seen.len(), 48);
}

/// A face whose diagonal corners are empty is ambiguous.  Every case resolves it
/// the same way: each empty corner is cut off by a segment of its own.
#[test]
fn checkerboard_faces_cut_off_every_empty_corner() {
    let mut checked = 0;
    for (case, &mask) in MC_EDGE_TABLE.iter().enumerate() {
        for face in ALL_FACES {
            let (axis, side) = face;
            let empty: Vec<usize> = (0..8)
                .filter(|&c| MC_CORNER_OFFSETS[c][axis] == side && (case >> c) & 1 == 1)
                .collect();
            let adjacent = MC_EDGE_CORNERS
                .iter()
                .any(|&[a, b]| empty.contains(&a) && empty.contains(&b));
            if empty.len() != 2 || adjacent {
                continue;
            }
            checked += 1;
            let boundary = boundary_sides(case);
            for &corner in &empty {
                let at_corner: Vec<usize> = (0..12)
                    .filter(|&e| {
                        mask & (1 << e) != 0
                            && faces_of_edge(e).contains(&face)
                            && MC_EDGE_CORNERS[e].contains(&corner)
                    })
                    .collect();
                assert_eq!(at_corner.len(), 2, "case {case}: corner {corner}");
                let cut_off = boundary.iter().any(|&(a, b, f)| {
                    f == face
                        && ((a == at_corner[0] && b == at_corner[1])
                            || (a == at_corner[1] && b == at_corner[0]))
                });
                assert!(
                    cut_off,
                    "case {case}: face {face:?} must cut off corner {corner}"
                );
            }
        }
    }
    assert!(checked > 0);
}

/// The right-hand normals of the table triangles point towards decreasing
/// density (into the empty corners), for every connected patch of every case.
#[test]
fn table_triangles_wind_out_of_the_solid() {
    let corner_pos = |c: usize| -> [f64; 3] {
        let o = MC_CORNER_OFFSETS[c];
        [o[0] as f64, o[1] as f64, o[2] as f64]
    };
    let midpoint = |e: usize| -> [f64; 3] {
        let [a, b] = MC_EDGE_CORNERS[e];
        let (pa, pb) = (corner_pos(a), corner_pos(b));
        [
            0.5 * (pa[0] + pb[0]),
            0.5 * (pa[1] + pb[1]),
            0.5 * (pa[2] + pb[2]),
        ]
    };
    for case in 1..255 {
        // Density -1 at empty corners and +1 at solid ones: every crossing is the
        // edge midpoint, and the trilinear gradient points into the solid.
        let value = |c: usize| if (case >> c) & 1 == 1 { -1.0 } else { 1.0 };
        let triangles = case_triangles(case);

        // Connected patches: triangles joined by a shared side.
        let mut patch: Vec<usize> = (0..triangles.len()).collect();
        for i in 0..triangles.len() {
            for j in (i + 1)..triangles.len() {
                let shares_side = (0..3).any(|s| {
                    let (a, b) = (triangles[i][s], triangles[i][(s + 1) % 3]);
                    (0..3).any(|r| {
                        let (c, d) = (triangles[j][r], triangles[j][(r + 1) % 3]);
                        (a, b) == (d, c)
                    })
                });
                if shares_side {
                    let (keep, drop) = (patch[i], patch[j]);
                    for p in patch.iter_mut() {
                        if *p == drop {
                            *p = keep;
                        }
                    }
                }
            }
        }

        let mut flux: HashMap<usize, f64> = HashMap::new();
        for (tri, &id) in triangles.iter().zip(&patch) {
            let (p0, p1, p2) = (midpoint(tri[0]), midpoint(tri[1]), midpoint(tri[2]));
            let u = [p1[0] - p0[0], p1[1] - p0[1], p1[2] - p0[2]];
            let v = [p2[0] - p0[0], p2[1] - p0[1], p2[2] - p0[2]];
            // Twice the area times the unit normal.
            let n = [
                u[1] * v[2] - u[2] * v[1],
                u[2] * v[0] - u[0] * v[2],
                u[0] * v[1] - u[1] * v[0],
            ];
            let g = [
                (p0[0] + p1[0] + p2[0]) / 3.0,
                (p0[1] + p1[1] + p2[1]) / 3.0,
                (p0[2] + p1[2] + p2[2]) / 3.0,
            ];
            // Gradient of the trilinear interpolant at the centroid.
            let mut grad = [0.0; 3];
            for (c, o) in MC_CORNER_OFFSETS.iter().enumerate() {
                let w = |axis: usize| if o[axis] == 1 { g[axis] } else { 1.0 - g[axis] };
                let dw = |axis: usize| if o[axis] == 1 { 1.0 } else { -1.0 };
                grad[0] += value(c) * dw(0) * w(1) * w(2);
                grad[1] += value(c) * w(0) * dw(1) * w(2);
                grad[2] += value(c) * w(0) * w(1) * dw(2);
            }
            *flux.entry(id).or_insert(0.0) += n[0] * grad[0] + n[1] * grad[1] + n[2] * grad[2];
        }
        for (id, total) in flux {
            assert!(
                total < 0.0,
                "case {case}: patch {id} faces towards increasing density (flux {total})"
            );
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Marching cubes extraction (F0291)
// ─────────────────────────────────────────────────────────────────────────────

/// Weld a triangle soup by the exact bit pattern of its coordinates.
fn weld(soup: &[[[f64; 3]; 3]]) -> (Vec<[f64; 3]>, Vec<[usize; 3]>) {
    let mut ids: HashMap<[u64; 3], usize> = HashMap::new();
    let mut vertices: Vec<[f64; 3]> = Vec::new();
    let mut triangles = Vec::with_capacity(soup.len());
    for tri in soup {
        let mut idx = [0usize; 3];
        for (k, p) in tri.iter().enumerate() {
            let key = [
                (p[0] + 0.0).to_bits(),
                (p[1] + 0.0).to_bits(),
                (p[2] + 0.0).to_bits(),
            ];
            idx[k] = *ids.entry(key).or_insert_with(|| {
                vertices.push(*p);
                vertices.len() - 1
            });
        }
        triangles.push(idx);
    }
    (vertices, triangles)
}

/// `true` if every directed edge occurs once and so does its reverse, i.e. the
/// triangles form a closed, consistently oriented surface.
fn is_closed_and_oriented(triangles: &[[usize; 3]]) -> bool {
    let mut directed: HashMap<(usize, usize), usize> = HashMap::new();
    for t in triangles {
        for k in 0..3 {
            *directed.entry((t[k], t[(k + 1) % 3])).or_insert(0) += 1;
        }
    }
    directed
        .iter()
        .all(|(&(a, b), &count)| count == 1 && directed.get(&(b, a)) == Some(&1))
}

/// Number of triangles incident to every undirected edge, as a histogram.
fn edge_valences(triangles: &[[usize; 3]]) -> HashMap<usize, usize> {
    let mut valence: HashMap<(usize, usize), usize> = HashMap::new();
    for t in triangles {
        for k in 0..3 {
            let (a, b) = (t[k], t[(k + 1) % 3]);
            *valence.entry((a.min(b), a.max(b))).or_insert(0) += 1;
        }
    }
    let mut histogram = HashMap::new();
    for &v in valence.values() {
        *histogram.entry(v).or_insert(0) += 1;
    }
    histogram
}

fn cross(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

fn sub(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}

fn dot(a: [f64; 3], b: [f64; 3]) -> f64 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

/// Right-hand normal of a triangle, scaled by twice its area.
fn area_normal(tri: &[[f64; 3]; 3]) -> [f64; 3] {
    cross(sub(tri[1], tri[0]), sub(tri[2], tri[0]))
}

/// Total area and signed volume (divergence theorem) of a triangle soup; the
/// volume is positive when the triangles face outwards.
fn area_and_volume(soup: &[[[f64; 3]; 3]]) -> (f64, f64) {
    let mut area = 0.0;
    let mut volume = 0.0;
    for tri in soup {
        let n = area_normal(tri);
        area += 0.5 * dot(n, n).sqrt();
        volume += dot(tri[0], cross(tri[1], tri[2])) / 6.0;
    }
    (area, volume)
}

fn sphere_terrain(center: [f64; 3], radius: f64, cells: usize, voxel_size: f64) -> VoxelTerrain {
    let mut terrain = VoxelTerrain::new(cells, cells, cells, voxel_size, [0.0; 3]);
    for iz in 0..cells {
        for iy in 0..cells {
            for ix in 0..cells {
                let p = [
                    ix as f64 * voxel_size,
                    iy as f64 * voxel_size,
                    iz as f64 * voxel_size,
                ];
                let d = dot(sub(p, center), sub(p, center)).sqrt();
                terrain.set(ix, iy, iz, radius - d);
            }
        }
    }
    terrain
}

#[test]
fn isosurface_of_a_sphere_lies_on_the_sphere_and_is_closed() {
    let voxel_size = 0.25;
    // Off the lattice, so no sample has a density of exactly zero.
    let center = [3.07, 3.11, 3.13];
    let radius = 2.0;
    let terrain = sphere_terrain(center, radius, 24, voxel_size);
    let soup = terrain.extract_isosurface();
    assert!(soup.len() > 1000, "only {} triangles", soup.len());

    // Every vertex is within half a voxel of the sphere (it is far closer).
    for tri in &soup {
        for p in tri {
            let d = dot(sub(*p, center), sub(*p, center)).sqrt();
            assert!(
                (d - radius).abs() <= 0.5 * voxel_size,
                "vertex {p:?} is {} from the sphere",
                d - radius
            );
        }
    }

    // A closed surface: every edge is shared by exactly two triangles, with
    // opposite orientation.
    let (vertices, triangles) = weld(&soup);
    let histogram = edge_valences(&triangles);
    assert_eq!(histogram.len(), 1, "edge valences {histogram:?}");
    assert!(histogram.contains_key(&2), "edge valences {histogram:?}");
    assert!(is_closed_and_oriented(&triangles));
    // Euler characteristic of a sphere: V - E + F = 2.
    let edges = triangles.len() * 3 / 2;
    assert_eq!(vertices.len() + triangles.len() - edges, 2);

    // Outward facing, and the enclosed volume / area match the sphere.
    for tri in &soup {
        let centroid = [
            (tri[0][0] + tri[1][0] + tri[2][0]) / 3.0,
            (tri[0][1] + tri[1][1] + tri[2][1]) / 3.0,
            (tri[0][2] + tri[1][2] + tri[2][2]) / 3.0,
        ];
        assert!(
            dot(area_normal(tri), sub(centroid, center)) > 0.0,
            "triangle {tri:?} faces into the solid"
        );
    }
    let (area, volume) = area_and_volume(&soup);
    let pi = std::f64::consts::PI;
    assert!(
        approx(
            volume,
            4.0 / 3.0 * pi * radius.powi(3),
            0.02 * 4.0 / 3.0 * pi * radius.powi(3)
        ),
        "volume {volume}"
    );
    assert!(
        approx(
            area,
            4.0 * pi * radius * radius,
            0.03 * 4.0 * pi * radius * radius
        ),
        "area {area}"
    );
}

#[test]
fn isosurface_through_exact_zero_samples_has_no_degenerate_triangles() {
    // Centre and radius on the lattice: many samples have a density of exactly 0,
    // so the crossings of their incident edges coincide.
    let center = [4.0; 3];
    let radius = 3.0;
    let terrain = sphere_terrain(center, radius, 9, 1.0);
    assert!(terrain.densities.contains(&0.0));
    let soup = terrain.extract_isosurface();
    assert!(!soup.is_empty());
    for tri in &soup {
        let n = area_normal(tri);
        assert!(dot(n, n) > 1e-12, "degenerate triangle {tri:?}");
        let centroid = [
            (tri[0][0] + tri[1][0] + tri[2][0]) / 3.0,
            (tri[0][1] + tri[1][1] + tri[2][1]) / 3.0,
            (tri[0][2] + tri[1][2] + tri[2][2]) / 3.0,
        ];
        assert!(
            dot(n, sub(centroid, center)) > 0.0,
            "triangle {tri:?} faces into the solid"
        );
        for p in tri {
            let d = dot(sub(*p, center), sub(*p, center)).sqrt();
            assert!(
                (d - radius).abs() <= 0.5,
                "vertex {p:?} is {} from the sphere",
                d - radius
            );
        }
    }
    let (vertices, triangles) = weld(&soup);
    assert!(
        is_closed_and_oriented(&triangles),
        "surface through zero samples has cracks"
    );
    let edges = triangles.len() * 3 / 2;
    assert_eq!(
        vertices.len() + triangles.len() - edges,
        2,
        "not a topological sphere"
    );
}

#[test]
fn isosurface_vertices_follow_the_density_values() {
    // The plane y = h inside a grid with samples at y = 2.0, 2.5, 3.0, ...
    for height in [3.05, 3.3, 3.45] {
        let mut terrain = VoxelTerrain::new(6, 6, 6, 0.5, [1.0, 2.0, 3.0]);
        for iz in 0..6 {
            for iy in 0..6 {
                for ix in 0..6 {
                    terrain.set(ix, iy, iz, height - (2.0 + iy as f64 * 0.5));
                }
            }
        }
        let soup = terrain.extract_isosurface();
        // One layer of 5 x 5 cells, two triangles per cell, on the plane.
        assert_eq!(soup.len(), 50, "height {height}");
        for tri in &soup {
            for p in tri {
                assert!(approx(p[1], height, 1e-12), "height {height}: vertex {p:?}");
                assert!((1.0..=3.5).contains(&p[0]) && (3.0..=5.5).contains(&p[2]));
            }
            // Solid below, empty above: the normals point up.
            let n = area_normal(tri);
            assert!(
                n[1] > 0.0 && n[0].abs() < 1e-12 && n[2].abs() < 1e-12,
                "{n:?}"
            );
        }
        let (area, _) = area_and_volume(&soup);
        assert!(approx(area, 2.5 * 2.5, 1e-9), "area {area}");
    }
}

#[test]
fn isosurface_samples_all_eight_corners_of_every_cell() {
    // One cell with a single solid corner: one triangle that cuts the three edges
    // at that corner, at the interpolated position.
    for (solid_corner, &offset) in MC_CORNER_OFFSETS.iter().enumerate() {
        let mut terrain = VoxelTerrain::new(2, 2, 2, 2.0, [10.0, 20.0, 30.0]);
        for iz in 0..2 {
            for iy in 0..2 {
                for ix in 0..2 {
                    let is_corner = [ix, iy, iz] == offset;
                    terrain.set(ix, iy, iz, if is_corner { 1.0 } else { -3.0 });
                }
            }
        }
        let soup = terrain.extract_isosurface();
        assert_eq!(soup.len(), 1, "solid corner {solid_corner}");
        let corner_world = [
            10.0 + 2.0 * offset[0] as f64,
            20.0 + 2.0 * offset[1] as f64,
            30.0 + 2.0 * offset[2] as f64,
        ];
        // d_solid = 1, d_empty = -3: the zero is a quarter of the way along each
        // edge, i.e. 0.25 * 2.0 = 0.5 away from the solid corner.
        let mut seen = [false; 3];
        for p in &soup[0] {
            let delta = sub(*p, corner_world);
            let axis = (0..3)
                .find(|&k| delta[k] != 0.0)
                .expect("a vertex differs from the corner");
            assert!(
                (0..3).filter(|&k| delta[k] != 0.0).count() == 1,
                "vertex {p:?} must lie on an edge of corner {solid_corner}"
            );
            let toward = if offset[axis] == 0 { 0.5 } else { -0.5 };
            assert!(approx(delta[axis], toward, 1e-12), "{delta:?}");
            seen[axis] = true;
        }
        assert_eq!(seen, [true; 3]);
        // The normal points away from the solid corner.
        let tri = &soup[0];
        let centroid = [
            (tri[0][0] + tri[1][0] + tri[2][0]) / 3.0,
            (tri[0][1] + tri[1][1] + tri[2][1]) / 3.0,
            (tri[0][2] + tri[1][2] + tri[2][2]) / 3.0,
        ];
        assert!(dot(area_normal(tri), sub(centroid, corner_world)) > 0.0);
    }
}

#[test]
fn isosurface_of_a_single_solid_sample_is_an_octahedron() {
    let mut terrain = VoxelTerrain::new(3, 3, 3, 1.0, [0.0; 3]);
    terrain.densities.fill(-1.0);
    terrain.set(1, 1, 1, 1.0);
    let soup = terrain.extract_isosurface();
    assert_eq!(soup.len(), 8);
    let (vertices, triangles) = weld(&soup);
    assert_eq!(vertices.len(), 6);
    assert!(is_closed_and_oriented(&triangles));
    let center = [1.0; 3];
    for v in &vertices {
        let d = sub(*v, center);
        let along: Vec<f64> = d.iter().copied().filter(|x| x.abs() > 1e-12).collect();
        assert_eq!(along.len(), 1, "{v:?}");
        assert!(approx(along[0].abs(), 0.5, 1e-12), "{v:?}");
    }
    for tri in &soup {
        let centroid = [
            (tri[0][0] + tri[1][0] + tri[2][0]) / 3.0,
            (tri[0][1] + tri[1][1] + tri[2][1]) / 3.0,
            (tri[0][2] + tri[1][2] + tri[2][2]) / 3.0,
        ];
        assert!(dot(area_normal(tri), sub(centroid, center)) > 0.0);
    }
    let (_, volume) = area_and_volume(&soup);
    // Octahedron with circumradius 0.5: volume 4/3 r^3.
    assert!(approx(volume, 4.0 / 3.0 * 0.125, 1e-12), "volume {volume}");
}

#[test]
fn isosurface_without_sign_change_is_empty() {
    for fill in [0.0, 1.0, 1e-300, -1.0, -1e-300] {
        let mut terrain = VoxelTerrain::new(4, 5, 3, 0.7, [1.0, -2.0, 0.5]);
        terrain.densities.fill(fill);
        assert!(terrain.extract_isosurface().is_empty(), "fill {fill}");
    }
}

#[test]
fn isosurface_of_random_densities_is_a_closed_oriented_surface() {
    // Random signs make every ambiguous configuration occur; with a solid border
    // layer the surface cannot reach the edge of the grid, so it must be closed.
    for seed in 0..12_u64 {
        let mut rng = Lcg(0x9E37_79B9_7F4A_7C15 ^ seed.wrapping_mul(0xD1B5_4A32_D192_ED03));
        let n = 8;
        let mut terrain = VoxelTerrain::new(n, n, n, 0.5, [-1.0, 0.0, 2.0]);
        for iz in 0..n {
            for iy in 0..n {
                for ix in 0..n {
                    let border = [ix, iy, iz].iter().any(|&i| i == 0 || i == n - 1);
                    let value = if border {
                        1.0
                    } else if seed % 3 == 0 {
                        // Checkerboard: the most ambiguous field.
                        if (ix + iy + iz) % 2 == 0 { -1.0 } else { 1.0 }
                    } else {
                        rng.range(-1.0, 1.0)
                    };
                    terrain.set(ix, iy, iz, value);
                }
            }
        }
        let soup = terrain.extract_isosurface();
        assert!(!soup.is_empty(), "seed {seed}");
        let (_, triangles) = weld(&soup);
        assert!(
            is_closed_and_oriented(&triangles),
            "seed {seed}: surface has cracks"
        );
        for tri in &soup {
            for p in tri {
                assert!(p.iter().all(|x| x.is_finite()), "seed {seed}: {p:?}");
            }
        }
    }
}

#[test]
fn isosurface_ignores_invalid_terrains_and_non_finite_densities() {
    let mut terrain = VoxelTerrain::new(3, 3, 3, 1.0, [0.0; 3]);
    terrain.densities.fill(-1.0);
    terrain.set(1, 1, 1, 1.0);
    assert_eq!(terrain.extract_isosurface().len(), 8);

    let mut flat = VoxelTerrain::new(3, 3, 1, 1.0, [0.0; 3]);
    flat.densities.fill(-1.0);
    flat.set(1, 1, 0, 1.0);
    assert!(
        flat.extract_isosurface().is_empty(),
        "needs two samples along every axis"
    );

    for bad in [0.0, -1.0, f64::NAN, f64::INFINITY] {
        let mut t = VoxelTerrain::new(3, 3, 3, bad, [0.0; 3]);
        t.densities.fill(-1.0);
        t.set(1, 1, 1, 1.0);
        assert!(t.extract_isosurface().is_empty(), "voxel size {bad}");
    }

    let mut truncated = VoxelTerrain::new(3, 3, 3, 1.0, [0.0; 3]);
    truncated.densities.fill(-1.0);
    truncated.densities.pop();
    assert!(truncated.extract_isosurface().is_empty());

    // Non-finite densities never produce NaN vertices.
    let mut messy = VoxelTerrain::new(3, 3, 3, 1.0, [0.0; 3]);
    messy.densities.fill(-1.0);
    messy.set(1, 1, 1, f64::INFINITY);
    messy.set(0, 0, 0, f64::NAN);
    messy.set(2, 2, 2, f64::NEG_INFINITY);
    for tri in messy.extract_isosurface() {
        for p in tri {
            assert!(p.iter().all(|x| x.is_finite()), "{p:?}");
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// TerrainRaycast (F0295)
// ─────────────────────────────────────────────────────────────────────────────

fn flat_field(nx: usize, nz: usize, h: f64) -> HeightfieldShape {
    HeightfieldShape::new(nx, nz, 1.0, 1.0, vec![h; nx * nz])
}

/// Heights `f(ix, iz)` on an `nx x nz` grid with unit spacing.
fn field_from(nx: usize, nz: usize, mut f: impl FnMut(usize, usize) -> f64) -> HeightfieldShape {
    let mut heights = Vec::with_capacity(nx * nz);
    for iz in 0..nz {
        for ix in 0..nx {
            heights.push(f(ix, iz));
        }
    }
    HeightfieldShape::new(nx, nz, 1.0, 1.0, heights)
}

#[test]
fn raycast_vertical_hits_flat_terrain_exactly() {
    let hf = flat_field(10, 10, 2.0);
    let hit = TerrainRaycast::new(&hf)
        .cast([4.5, 10.0, 4.5], [0.0, -1.0, 0.0])
        .expect("ray from above must hit");
    assert!(
        approx3(hit.point, [4.5, 2.0, 4.5], 1e-12),
        "{:?}",
        hit.point
    );
    assert!(
        approx3(hit.normal, [0.0, 1.0, 0.0], 1e-12),
        "{:?}",
        hit.normal
    );
    assert!(approx(hit.t, 8.0, 1e-12));
    assert_eq!((hit.cell_x, hit.cell_z), (4, 4));
}

#[test]
fn raycast_horizontal_ray_hits_a_raised_wall() {
    // Heights 0 for ix <= 5 and 5 for ix >= 6: a steep ramp across cell column 5.
    let hf = field_from(12, 4, |ix, _| if ix >= 6 { 5.0 } else { 0.0 });
    let caster = TerrainRaycast::new(&hf);
    let hit = caster
        .cast([0.5, 2.0, 1.5], [1.0, 0.0, 0.0])
        .expect("a horizontal ray must hit the wall");
    // The ramp is the plane y = 5 (x - 5); at y = 2 that is x = 5.4.
    assert!(approx3(hit.point, [5.4, 2.0, 1.5], 1e-9), "{:?}", hit.point);
    assert!(approx(hit.t, 4.9, 1e-9), "t = {}", hit.t);
    let len = 26.0_f64.sqrt();
    assert!(
        approx3(hit.normal, [-5.0 / len, 1.0 / len, 0.0], 1e-9),
        "{:?}",
        hit.normal
    );
    assert_eq!(hit.cell_x, 5);

    // The same ray from the other side hits the other face of the ramp, with the
    // same upward normal.
    let back = caster
        .cast([11.5, 2.0, 1.5], [-1.0, 0.0, 0.0])
        .expect("a horizontal ray must hit the wall");
    assert!(
        approx3(back.point, [5.4, 2.0, 1.5], 1e-9),
        "{:?}",
        back.point
    );
    assert!(approx3(back.normal, hit.normal, 1e-9), "{:?}", back.normal);

    // A horizontal ray above the wall top or inside a flat part passes over.
    assert!(caster.cast([0.5, 6.0, 1.5], [1.0, 0.0, 0.0]).is_none());
    // A horizontal ray buried inside the plateau never crosses the surface, but it
    // has to walk every cell of its column to find that out.
    assert!(caster.cast([8.5, 2.0, -3.0], [0.0, 0.0, 1.0]).is_none());
}

#[test]
fn raycast_hits_a_45_degree_slope_at_the_analytic_point() {
    // Heights equal to ix: the surface is the plane y = x.
    let hf = field_from(8, 4, |ix, _| ix as f64);
    let caster = TerrainRaycast::new(&hf);
    let s = 0.5_f64.sqrt();

    // Straight down onto the slope.
    let down = caster
        .cast([2.25, 9.0, 1.5], [0.0, -1.0, 0.0])
        .expect("hit");
    assert!(
        approx3(down.point, [2.25, 2.25, 1.5], 1e-12),
        "{:?}",
        down.point
    );
    assert!(
        approx3(down.normal, [-s, s, 0.0], 1e-12),
        "{:?}",
        down.normal
    );
    assert_eq!(down.cell_x, 2);

    // Oblique: x = 6.5 - 2u, y = 4 - u hits y = x at u = 2.5, i.e. (1.5, 1.5, 1.5).
    // The direction is not normalised, so t is the parameter u.
    let oblique = caster
        .cast([6.5, 4.0, 1.5], [-2.0, -1.0, 0.0])
        .expect("hit");
    assert!(
        approx3(oblique.point, [1.5, 1.5, 1.5], 1e-12),
        "{:?}",
        oblique.point
    );
    assert!(approx(oblique.t, 2.5, 1e-12), "t = {}", oblique.t);
    assert!(approx3(oblique.normal, [-s, s, 0.0], 1e-12));
    assert_eq!(oblique.cell_x, 1);

    // Slanted in the xz plane as well: x = 6.25 - u, y = 7 - 2u, z = 0.25 + 0.5u
    // meets y = x at u = 0.75, i.e. (5.5, 5.5, 0.625).
    let diag = caster
        .cast([6.25, 7.0, 0.25], [-1.0, -2.0, 0.5])
        .expect("hit");
    assert!(
        approx3(diag.point, [5.5, 5.5, 0.625], 1e-12),
        "{:?}",
        diag.point
    );
    assert!(approx(diag.t, 0.75, 1e-12), "t = {}", diag.t);
    assert!(approx3(diag.normal, [-s, s, 0.0], 1e-12));
    assert_eq!((diag.cell_x, diag.cell_z), (5, 0));
}

#[test]
fn raycast_from_below_and_from_outside_the_footprint() {
    let hf = flat_field(8, 8, 2.0);
    let caster = TerrainRaycast::new(&hf);

    // From below, going up: hits the underside, with the upward normal.
    let up = caster.cast([3.5, -2.0, 1.5], [0.0, 1.0, 0.0]).expect("hit");
    assert!(approx3(up.point, [3.5, 2.0, 1.5], 1e-12));
    assert!(approx(up.t, 4.0, 1e-12));
    assert!(approx3(up.normal, [0.0, 1.0, 0.0], 1e-12));

    // From outside the footprint, on the negative side of x.
    let side = caster
        .cast([-3.0, 4.0, 2.5], [1.0, -0.4, 0.0])
        .expect("hit");
    assert!(
        approx3(side.point, [2.0, 2.0, 2.5], 1e-9),
        "{:?}",
        side.point
    );
    assert!(approx(side.t, 5.0, 1e-9), "t = {}", side.t);

    // From outside on the positive side of z, running towards -z at a slant:
    // y = 3 - 0.25 u reaches 2 at u = 4, where z = 9 - 4 = 5.
    let far = caster
        .cast([3.0, 3.0, 9.0], [0.0, -0.25, -1.0])
        .expect("hit");
    assert!(approx3(far.point, [3.0, 2.0, 5.0], 1e-9), "{:?}", far.point);
    assert!(approx(far.t, 4.0, 1e-9), "t = {}", far.t);
    // The same slant meets the ground plane at z = 8, past the footprint (z <= 7).
    assert!(caster.cast([3.0, 3.0, 12.0], [0.0, -0.25, -1.0]).is_none());

    // A ray that starts exactly on the surface hits at t = 0.
    let on = caster.cast([2.5, 2.0, 2.5], [0.0, -1.0, 0.0]).expect("hit");
    assert!(approx(on.t, 0.0, 1e-9), "t = {}", on.t);

    // Misses: pointing up from above, away from the footprint, parallel to the ground.
    assert!(caster.cast([3.5, 5.0, 1.5], [0.0, 1.0, 0.0]).is_none());
    assert!(caster.cast([-3.0, 4.0, 2.5], [-1.0, -0.4, 0.0]).is_none());
    assert!(caster.cast([20.0, 4.0, 2.5], [1.0, -0.4, 0.0]).is_none());
    assert!(caster.cast([3.5, 5.0, 1.5], [1.0, 0.0, 0.3]).is_none());
    // Beside the footprint.
    assert!(caster.cast([-1.0, 5.0, 2.5], [0.0, -1.0, 0.0]).is_none());
}

#[test]
fn raycast_invalid_input_and_degenerate_fields_return_none() {
    let hf = flat_field(8, 8, 2.0);
    let caster = TerrainRaycast::new(&hf);
    assert!(caster.cast([3.5, 5.0, 1.5], [0.0, 0.0, 0.0]).is_none());
    assert!(
        caster
            .cast([f64::NAN, 5.0, 1.5], [0.0, -1.0, 0.0])
            .is_none()
    );
    assert!(
        caster
            .cast([3.5, 5.0, 1.5], [0.0, f64::INFINITY, 0.0])
            .is_none()
    );

    // Fewer than two samples along an axis: no surface.
    let thin_x = HeightfieldShape::new(1, 6, 1.0, 1.0, vec![0.0; 6]);
    assert!(
        TerrainRaycast::new(&thin_x)
            .cast([0.0, 5.0, 2.5], [0.0, -1.0, 0.0])
            .is_none()
    );
    let thin_z = HeightfieldShape::new(6, 1, 1.0, 1.0, vec![0.0; 6]);
    assert!(
        TerrainRaycast::new(&thin_z)
            .cast([2.5, 5.0, 0.0], [0.0, -1.0, 0.0])
            .is_none()
    );
    let empty = HeightfieldShape::new(0, 0, 1.0, 1.0, Vec::new());
    assert!(
        TerrainRaycast::new(&empty)
            .cast([0.0, 5.0, 0.0], [0.0, -1.0, 0.0])
            .is_none()
    );

    let mut bad_scale = flat_field(4, 4, 0.0);
    bad_scale.scale_x = 0.0;
    assert!(
        TerrainRaycast::new(&bad_scale)
            .cast([1.0, 5.0, 1.0], [0.0, -1.0, 0.0])
            .is_none()
    );
}

#[test]
fn raycast_respects_scale_and_offset() {
    let mut hf = HeightfieldShape::new(5, 4, 2.0, 0.5, vec![1.0; 20]);
    hf.y_offset = 3.0;
    let caster = TerrainRaycast::new(&hf);
    let hit = caster
        .cast([5.0, 10.0, 1.2], [0.0, -2.0, 0.0])
        .expect("hit");
    assert!(
        approx3(hit.point, [5.0, 4.0, 1.2], 1e-12),
        "{:?}",
        hit.point
    );
    assert!(approx(hit.t, 3.0, 1e-12), "t = {}", hit.t);
    assert_eq!((hit.cell_x, hit.cell_z), (2, 2));
    // Beyond the footprint (x extent is 8, z extent 1.5).
    assert!(caster.cast([8.5, 10.0, 1.0], [0.0, -1.0, 0.0]).is_none());
    assert!(caster.cast([5.0, 10.0, 1.6], [0.0, -1.0, 0.0]).is_none());
}

/// Brute force reference: nearest hit over every triangle of the field.
fn brute_force_hit(hf: &HeightfieldShape, origin: [f64; 3], dir: [f64; 3]) -> Option<f64> {
    let len = dot(dir, dir).sqrt();
    let inv = 1.0 / len;
    let unit = [dir[0] * inv, dir[1] * inv, dir[2] * inv];
    let mut best: Option<f64> = None;
    for cz in 0..hf.nz - 1 {
        for cx in 0..hf.nx - 1 {
            let (lower, upper) = TerrainTriangle::cell_triangles(hf, cx, cz);
            for tri in [lower, upper] {
                if let Some(h) = crate::sweep::ray_triangle(
                    origin,
                    unit,
                    tri[0],
                    tri[1],
                    tri[2],
                    0.0,
                    f64::INFINITY,
                ) && best.is_none_or(|t| h.t < t)
                {
                    best = Some(h.t);
                }
            }
        }
    }
    best.map(|t| t / len)
}

#[test]
fn raycast_matches_brute_force_on_random_terrain() {
    let mut rng = Lcg(0xC0FF_EE12_3456_789A);
    let (nx, nz) = (11, 9);
    let hf = field_from(nx, nz, |_, _| rng.range(0.0, 3.0));
    let caster = TerrainRaycast::new(&hf);
    let mut hits = 0;
    let mut misses = 0;
    for i in 0..900 {
        // Two thirds of the rays aim at the ground from above, the rest are arbitrary.
        let downward = i % 3 != 0;
        let origin = if downward {
            [
                rng.range(-2.0, 12.0),
                rng.range(3.2, 9.0),
                rng.range(-2.0, 10.0),
            ]
        } else {
            [
                rng.range(-4.0, 14.0),
                rng.range(-2.0, 7.0),
                rng.range(-4.0, 12.0),
            ]
        };
        let mut dir = if downward {
            [
                rng.range(-0.5, 0.5),
                -rng.range(0.4, 1.0),
                rng.range(-0.5, 0.5),
            ]
        } else {
            [
                rng.range(-1.0, 1.0),
                rng.range(-1.0, 1.0),
                rng.range(-1.0, 1.0),
            ]
        };
        if downward {
            // Some straight down, some along an axis plane.
            match i % 7 {
                0 => {
                    dir[0] = 0.0;
                    dir[2] = 0.0;
                }
                1 => dir[0] = 0.0,
                2 => dir[2] = 0.0,
                _ => {}
            }
        } else {
            // A share of the rays is horizontal or axis aligned.
            match (rng.next_f64() * 6.0) as usize {
                0 => dir[1] = 0.0,
                1 => dir[0] = 0.0,
                2 => dir[2] = 0.0,
                3 => {
                    dir[0] = 0.0;
                    dir[2] = 0.0;
                }
                _ => {}
            }
        }
        if dot(dir, dir) < 1e-6 {
            continue;
        }
        let expected = brute_force_hit(&hf, origin, dir);
        let got = caster.cast(origin, dir);
        match (expected, got) {
            (Some(t), Some(hit)) => {
                hits += 1;
                assert!(
                    approx(hit.t, t, 1e-9 * (1.0 + t)),
                    "origin {origin:?} dir {dir:?}: t {} vs brute force {t}",
                    hit.t
                );
                let on_ray = [
                    origin[0] + dir[0] * hit.t,
                    origin[1] + dir[1] * hit.t,
                    origin[2] + dir[2] * hit.t,
                ];
                assert!(approx3(hit.point, on_ray, 1e-9), "point is not on the ray");
                assert!(hit.normal[1] > 0.0 && approx(dot(hit.normal, hit.normal), 1.0, 1e-12));
            }
            (None, None) => misses += 1,
            (expected, got) => panic!(
                "origin {origin:?} dir {dir:?}: brute force {expected:?}, cast {:?}",
                got.map(|h| h.t)
            ),
        }
    }
    assert!(hits > 300 && misses > 50, "hits {hits}, misses {misses}");
}

#[test]
fn raycast_hits_lattice_vertices_and_cell_edges() {
    let hf = field_from(6, 6, |ix, iz| 0.3 * ix as f64 + 0.1 * iz as f64);
    let caster = TerrainRaycast::new(&hf);
    // Straight down onto lattice vertices (shared by up to six triangles).
    for (ix, iz) in [(0, 0), (3, 2), (5, 5), (0, 4), (4, 0)] {
        let expected = hf.height_at_cell(ix, iz);
        let hit = caster
            .cast([ix as f64, 9.0, iz as f64], [0.0, -1.0, 0.0])
            .unwrap_or_else(|| panic!("no hit above vertex ({ix}, {iz})"));
        assert!(
            approx(hit.point[1], expected, 1e-9),
            "vertex ({ix}, {iz}): {:?}",
            hit.point
        );
    }
    // Along a cell edge on the diagonal of a cell.
    let hit = caster.cast([2.5, 9.0, 1.5], [0.0, -1.0, 0.0]).expect("hit");
    assert!(
        approx(hit.point[1], 0.3 * 2.5 + 0.1 * 1.5, 1e-9),
        "{:?}",
        hit.point
    );
}
