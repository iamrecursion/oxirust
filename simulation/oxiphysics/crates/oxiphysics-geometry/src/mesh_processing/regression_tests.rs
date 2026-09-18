// Copyright 2026 COOLJAPAN OU (Team KitaSan)
// SPDX-License-Identifier: Apache-2.0

//! Regression tests for the quadric-error-metric decimator ([`qem_decimate`])
//! and the pinch-vertex repair ([`fix_non_manifold_vertices`]).
//!
//! The decimation tests check the *topology* and the *geometry* of the result
//! (a valid oriented manifold, the boundary kept in place, no flipped or
//! degenerate triangle, the covered area unchanged), not just the face count.

use super::*;
use std::collections::{HashMap, HashSet};

/// Triangulated `cells x cells` grid over the unit square in the plane `z = 0`,
/// counter-clockwise seen from `+z`.
fn planar_grid(cells: usize) -> ProcessMesh {
    height_field_grid(cells, |_, _| 0.0)
}

/// Triangulated `cells x cells` grid over the unit square with `z = height(x, y)`.
fn height_field_grid(cells: usize, height: impl Fn(f64, f64) -> f64) -> ProcessMesh {
    let stride = cells + 1;
    let mut verts = Vec::with_capacity(stride * stride);
    for j in 0..=cells {
        for i in 0..=cells {
            let x = i as f64 / cells as f64;
            let y = j as f64 / cells as f64;
            verts.push([x, y, height(x, y)]);
        }
    }
    let mut faces = Vec::with_capacity(2 * cells * cells);
    for j in 0..cells {
        for i in 0..cells {
            let v00 = j * stride + i;
            let v10 = v00 + 1;
            let v01 = v00 + stride;
            let v11 = v01 + 1;
            faces.push([v00, v10, v11]);
            faces.push([v00, v11, v01]);
        }
    }
    ProcessMesh::new(verts, faces)
}

/// Twice-area vector of a face.
fn area_vector_of(mesh: &ProcessMesh, face: Face) -> Vertex {
    let [a, b, c] = face;
    vec3_cross(
        vec3_sub(mesh.verts[b], mesh.verts[a]),
        vec3_sub(mesh.verts[c], mesh.verts[a]),
    )
}

/// Sum of the triangle areas.
fn total_area(mesh: &ProcessMesh) -> f64 {
    mesh.faces
        .iter()
        .map(|&face| 0.5 * vec3_len(area_vector_of(mesh, face)))
        .sum()
}

/// Volume enclosed by a closed, outward-oriented mesh.
fn enclosed_volume(mesh: &ProcessMesh) -> f64 {
    mesh.faces
        .iter()
        .map(|&[a, b, c]| vec3_dot(mesh.verts[a], vec3_cross(mesh.verts[b], mesh.verts[c])))
        .sum::<f64>()
        / 6.0
}

/// How often every directed edge is used.
fn directed_edge_uses(mesh: &ProcessMesh) -> HashMap<(usize, usize), usize> {
    let mut uses = HashMap::new();
    for &[a, b, c] in &mesh.faces {
        for edge in [(a, b), (b, c), (c, a)] {
            *uses.entry(edge).or_insert(0) += 1;
        }
    }
    uses
}

/// How often every undirected edge is used.
fn undirected_edge_uses(mesh: &ProcessMesh) -> HashMap<(usize, usize), usize> {
    let mut uses = HashMap::new();
    for &[a, b, c] in &mesh.faces {
        for (u, v) in [(a, b), (b, c), (c, a)] {
            *uses.entry((u.min(v), u.max(v))).or_insert(0) += 1;
        }
    }
    uses
}

/// Panics unless the mesh is an oriented 2-manifold with boundary: no face with a
/// repeated or out-of-range index, no directed edge used twice (which would be
/// an orientation conflict or a duplicate face) and no edge shared by more than
/// two faces.
fn assert_oriented_manifold(mesh: &ProcessMesh) {
    for (fi, &[a, b, c]) in mesh.faces.iter().enumerate() {
        assert!(
            a != b && b != c && a != c,
            "face {fi} repeats a vertex: {:?}",
            mesh.faces[fi]
        );
        assert!(
            a < mesh.verts.len() && b < mesh.verts.len() && c < mesh.verts.len(),
            "face {fi} references a missing vertex: {:?}",
            mesh.faces[fi]
        );
    }
    for (edge, uses) in directed_edge_uses(mesh) {
        assert_eq!(uses, 1, "directed edge {edge:?} is used {uses} times");
    }
    for (edge, uses) in undirected_edge_uses(mesh) {
        assert!(uses <= 2, "edge {edge:?} is shared by {uses} faces");
    }
}

/// Number of vertices that at least one face references.
fn referenced_vertex_count(mesh: &ProcessMesh) -> usize {
    mesh.faces
        .iter()
        .flatten()
        .copied()
        .collect::<HashSet<usize>>()
        .len()
}

/// Euler characteristic `V - E + F` over the referenced vertices.
fn euler_characteristic(mesh: &ProcessMesh) -> i64 {
    referenced_vertex_count(mesh) as i64 - undirected_edge_uses(mesh).len() as i64
        + mesh.faces.len() as i64
}

/// Boundary edges (used by exactly one face).
fn boundary_edges(mesh: &ProcessMesh) -> Vec<(usize, usize)> {
    let mut edges: Vec<(usize, usize)> = undirected_edge_uses(mesh)
        .into_iter()
        .filter(|&(_, uses)| uses == 1)
        .map(|(edge, _)| edge)
        .collect();
    edges.sort_unstable();
    edges
}

/// Whether both end points of the edge lie on the same side of the unit square.
fn edge_on_unit_square_boundary(mesh: &ProcessMesh, (u, v): (usize, usize)) -> bool {
    let (p, q) = (mesh.verts[u], mesh.verts[v]);
    let on_side = |axis: usize, value: f64| {
        (p[axis] - value).abs() < 1e-12 && (q[axis] - value).abs() < 1e-12
    };
    on_side(0, 0.0) || on_side(0, 1.0) || on_side(1, 0.0) || on_side(1, 1.0)
}

/// Asserts that a decimated unit-square patch is still a valid triangulation of
/// the square: oriented manifold, one boundary loop that runs along the square,
/// all four corners present, nothing flipped or flattened, area one.
fn assert_valid_square_patch(mesh: &ProcessMesh) {
    assert_oriented_manifold(mesh);
    for &face in &mesh.faces {
        let normal = area_vector_of(mesh, face);
        assert!(
            normal[2] > 1e-9,
            "face {face:?} is flipped or degenerate: area vector {normal:?}"
        );
    }
    assert!(
        (total_area(mesh) - 1.0).abs() < 1e-9,
        "covered area {} differs from the unit square",
        total_area(mesh)
    );
    assert_eq!(
        referenced_vertex_count(mesh),
        mesh.verts.len(),
        "the output must not keep unreferenced vertices"
    );
    assert_eq!(euler_characteristic(mesh), 1, "the patch must stay a disc");
    for edge in boundary_edges(mesh) {
        assert!(
            edge_on_unit_square_boundary(mesh, edge),
            "boundary edge {edge:?} left the boundary of the square: {:?} - {:?}",
            mesh.verts[edge.0],
            mesh.verts[edge.1]
        );
    }
    for corner in [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]] {
        assert!(
            mesh.verts
                .iter()
                .any(|v| (v[0] - corner[0]).abs() < 1e-12 && (v[1] - corner[1]).abs() < 1e-12),
            "corner {corner:?} was lost"
        );
    }
}

#[test]
fn planar_grid_decimated_to_100_faces_is_a_valid_patch() {
    // 20 x 20 cells = 800 triangles.  On a plane every collapse costs zero, which
    // used to make the decimator collapse the self-edge of an already removed
    // face and remap the surviving faces onto vertex 0.
    let grid = planar_grid(20);
    assert_eq!(grid.num_faces(), 800);
    let decimated = qem_decimate(&grid, 100);
    assert!(
        decimated.num_faces() <= 100 && decimated.num_faces() >= 2,
        "faces = {}",
        decimated.num_faces()
    );
    assert!(
        decimated.verts.iter().all(|v| v[2] == 0.0),
        "left the plane"
    );
    assert_valid_square_patch(&decimated);
}

#[test]
fn planar_grid_decimation_is_valid_for_many_sizes_and_targets() {
    for cells in [2usize, 3, 5, 8, 13] {
        let grid = planar_grid(cells);
        for target in [grid.num_faces() / 2, grid.num_faces() / 4, 6, 2] {
            let decimated = qem_decimate(&grid, target);
            assert!(
                decimated.num_faces() <= target.max(2),
                "cells {cells}, target {target}: {} faces",
                decimated.num_faces()
            );
            assert_valid_square_patch(&decimated);
        }
    }
}

#[test]
fn bumpy_height_field_decimation_never_flips_or_overlaps_triangles() {
    let surface = height_field_grid(16, |x, y| 0.2 * (3.0 * x).sin() * (2.0 * y).cos());
    for target in [200usize, 100, 40] {
        let decimated = qem_decimate(&surface, target);
        assert!(decimated.num_faces() <= target);
        // The surface is a height field, so a valid decimation keeps every
        // triangle facing up, covers the square exactly once and keeps its
        // outline.
        assert_oriented_manifold(&decimated);
        let projected_area: f64 = decimated
            .faces
            .iter()
            .map(|&face| {
                let normal = area_vector_of(&decimated, face);
                assert!(normal[2] > 0.0, "face {face:?} points down");
                0.5 * normal[2]
            })
            .sum();
        assert!(
            (projected_area - 1.0).abs() < 1e-9,
            "target {target}: projected area {projected_area}"
        );
        assert_eq!(euler_characteristic(&decimated), 1);
        for edge in boundary_edges(&decimated) {
            assert!(edge_on_unit_square_boundary(&decimated, edge));
        }
    }
}

#[test]
fn icosphere_decimation_stays_a_closed_outward_manifold() {
    let sphere = make_icosphere(2);
    assert_eq!(sphere.num_faces(), 320);
    let decimated = qem_decimate(&sphere, 100);
    assert!(
        decimated.num_faces() <= 100 && decimated.num_faces() >= 4,
        "faces = {}",
        decimated.num_faces()
    );
    assert_oriented_manifold(&decimated);
    assert!(
        boundary_edges(&decimated).is_empty(),
        "the sphere must stay closed"
    );
    assert_eq!(
        euler_characteristic(&decimated),
        2,
        "the sphere must stay a sphere"
    );
    for &face in &decimated.faces {
        let [a, b, c] = face;
        let centroid = vec3_scale(
            vec3_add(
                vec3_add(decimated.verts[a], decimated.verts[b]),
                decimated.verts[c],
            ),
            1.0 / 3.0,
        );
        assert!(
            vec3_dot(area_vector_of(&decimated, face), centroid) > 0.0,
            "face {face:?} points inwards"
        );
    }
    for v in &decimated.verts {
        let radius = vec3_len(*v);
        assert!(
            (0.75..=1.05).contains(&radius),
            "vertex off the sphere: radius {radius}"
        );
    }
    let unit_ball = 4.0 / 3.0 * std::f64::consts::PI;
    let volume = enclosed_volume(&decimated);
    assert!(
        volume > 0.5 * unit_ball && volume < unit_ball,
        "volume {volume} of a decimated unit sphere"
    );
}

#[test]
fn decimation_is_deterministic() {
    let sphere = make_icosphere(2);
    let first = qem_decimate(&sphere, 120);
    let second = qem_decimate(&sphere, 120);
    assert_eq!(first.verts, second.verts);
    assert_eq!(first.faces, second.faces);
    let grid = planar_grid(9);
    assert_eq!(qem_decimate(&grid, 30).faces, qem_decimate(&grid, 30).faces);
}

#[test]
fn a_tetrahedron_cannot_be_decimated_any_further() {
    let tetrahedron = make_tetrahedron();
    let decimated = qem_decimate(&tetrahedron, 0);
    assert_eq!(decimated.num_faces(), 4);
    assert_eq!(decimated.num_verts(), 4);
    assert_oriented_manifold(&decimated);
    assert!(boundary_edges(&decimated).is_empty());
}

#[test]
fn a_single_triangle_is_kept() {
    let triangle = ProcessMesh::new(
        vec![[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
        vec![[0, 1, 2]],
    );
    let decimated = qem_decimate(&triangle, 0);
    assert_eq!(decimated.faces, vec![[0, 1, 2]]);
    assert_eq!(decimated.verts, triangle.verts);
}

#[test]
fn meshes_that_are_small_enough_or_invalid_are_returned_unchanged() {
    let tetrahedron = make_tetrahedron();
    let same = qem_decimate(&tetrahedron, 4);
    assert_eq!(same.verts, tetrahedron.verts);
    assert_eq!(same.faces, tetrahedron.faces);

    let broken = ProcessMesh::new(
        vec![[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
        vec![[0, 1, 2], [0, 2, 7]],
    );
    let unchanged = qem_decimate(&broken, 1);
    assert_eq!(unchanged.verts, broken.verts);
    assert_eq!(unchanged.faces, broken.faces);
}

#[test]
fn degenerate_input_faces_are_dropped_not_collapsed_into_the_mesh() {
    let mut grid = planar_grid(4);
    grid.faces.push([3, 3, 4]);
    grid.faces.push([0, 5, 5]);
    let decimated = qem_decimate(&grid, 10);
    assert!(decimated.num_faces() <= 10);
    assert_valid_square_patch(&decimated);
}

#[test]
fn two_disconnected_patches_are_decimated_independently() {
    let mut mesh = planar_grid(6);
    let second = planar_grid(6);
    let offset = mesh.verts.len();
    for v in &second.verts {
        mesh.verts.push([v[0] + 3.0, v[1], v[2]]);
    }
    for f in &second.faces {
        mesh.faces
            .push([f[0] + offset, f[1] + offset, f[2] + offset]);
    }
    let decimated = qem_decimate(&mesh, 20);
    assert!(decimated.num_faces() <= 20);
    assert_oriented_manifold(&decimated);
    // Both squares are still covered exactly once.
    assert!((total_area(&decimated) - 2.0).abs() < 1e-9);
    assert_eq!(euler_characteristic(&decimated), 2);
}

/// Number of connected boundary loops; panics when a boundary vertex is pinched
/// (it does not have exactly two boundary edges).
fn boundary_loop_count(mesh: &ProcessMesh) -> usize {
    let mut adjacency: HashMap<usize, Vec<usize>> = HashMap::new();
    for (u, v) in boundary_edges(mesh) {
        adjacency.entry(u).or_default().push(v);
        adjacency.entry(v).or_default().push(u);
    }
    for (vertex, neighbours) in &adjacency {
        assert_eq!(
            neighbours.len(),
            2,
            "boundary vertex {vertex} is pinched: {neighbours:?}"
        );
    }
    let mut starts: Vec<usize> = adjacency.keys().copied().collect();
    starts.sort_unstable();
    let mut seen: HashSet<usize> = HashSet::new();
    let mut loops = 0;
    for start in starts {
        if !seen.insert(start) {
            continue;
        }
        loops += 1;
        let mut stack = vec![start];
        while let Some(v) = stack.pop() {
            for &w in &adjacency[&v] {
                if seen.insert(w) {
                    stack.push(w);
                }
            }
        }
    }
    loops
}

/// Flat ring with outer radius 2 and inner radius 1, `segments` quads around.
fn annulus(segments: usize) -> ProcessMesh {
    let mut verts = Vec::new();
    for radius in [2.0_f64, 1.0] {
        for i in 0..segments {
            let angle = std::f64::consts::TAU * i as f64 / segments as f64;
            verts.push([radius * angle.cos(), radius * angle.sin(), 0.0]);
        }
    }
    let mut faces = Vec::new();
    for i in 0..segments {
        let next = (i + 1) % segments;
        let (outer, outer_next) = (i, next);
        let (inner, inner_next) = (segments + i, segments + next);
        faces.push([outer, outer_next, inner_next]);
        faces.push([outer, inner_next, inner]);
    }
    ProcessMesh::new(verts, faces)
}

/// Open cylinder of radius 1 and height 2 with `rings` vertex rings of `around`
/// vertices each; the triangles face outwards.
fn open_tube(around: usize, rings: usize) -> ProcessMesh {
    let mut verts = Vec::new();
    for j in 0..rings {
        for i in 0..around {
            let angle = std::f64::consts::TAU * i as f64 / around as f64;
            verts.push([
                angle.cos(),
                angle.sin(),
                2.0 * j as f64 / (rings - 1) as f64,
            ]);
        }
    }
    let mut faces = Vec::new();
    for j in 0..rings - 1 {
        for i in 0..around {
            let next = (i + 1) % around;
            let v00 = j * around + i;
            let v10 = j * around + next;
            let v01 = (j + 1) * around + i;
            let v11 = (j + 1) * around + next;
            faces.push([v00, v10, v11]);
            faces.push([v00, v11, v01]);
        }
    }
    ProcessMesh::new(verts, faces)
}

/// Octahedron with the vertices `+-x, +-y, +-z` and outward triangles.
fn octahedron() -> ProcessMesh {
    ProcessMesh::new(
        vec![
            [1.0, 0.0, 0.0],
            [-1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, -1.0, 0.0],
            [0.0, 0.0, 1.0],
            [0.0, 0.0, -1.0],
        ],
        vec![
            [0, 2, 4],
            [2, 1, 4],
            [1, 3, 4],
            [3, 0, 4],
            [2, 0, 5],
            [1, 2, 5],
            [3, 1, 5],
            [0, 3, 5],
        ],
    )
}

/// Asserts that every face of a closed mesh around the origin points outwards.
fn assert_faces_point_outwards(mesh: &ProcessMesh) {
    for &face in &mesh.faces {
        let [a, b, c] = face;
        let centroid = vec3_scale(
            vec3_add(vec3_add(mesh.verts[a], mesh.verts[b]), mesh.verts[c]),
            1.0 / 3.0,
        );
        assert!(
            vec3_dot(area_vector_of(mesh, face), centroid) > 0.0,
            "face {face:?} points inwards"
        );
    }
}

#[test]
fn annulus_decimation_keeps_both_boundary_loops() {
    // Every vertex of the ring is a boundary vertex and every rung joins two of
    // them: collapsing a rung would weld the two rims and destroy the hole.
    let ring = annulus(12);
    assert_eq!(ring.num_faces(), 24);
    assert_eq!(boundary_loop_count(&ring), 2);
    for target in [20usize, 12, 8] {
        let decimated = qem_decimate(&ring, target);
        assert!(
            decimated.num_faces() <= target,
            "target {target}: {} faces",
            decimated.num_faces()
        );
        assert_oriented_manifold(&decimated);
        assert_eq!(
            euler_characteristic(&decimated),
            0,
            "target {target}: the ring must keep its hole"
        );
        assert_eq!(boundary_loop_count(&decimated), 2, "target {target}");
        for &face in &decimated.faces {
            assert!(area_vector_of(&decimated, face)[2] > 0.0, "face {face:?}");
        }
        // Boundary vertices are never given a new position.
        for v in &decimated.verts {
            let radius = v[0].hypot(v[1]);
            assert!(
                (radius - 1.0).abs() < 1e-12 || (radius - 2.0).abs() < 1e-12,
                "vertex {v:?} left the rims"
            );
        }
    }
}

#[test]
fn open_tube_decimation_keeps_its_rims_and_outward_normals() {
    let tube = open_tube(12, 6);
    assert_eq!(tube.num_faces(), 120);
    assert_eq!(boundary_loop_count(&tube), 2);
    for target in [80usize, 48, 24] {
        let decimated = qem_decimate(&tube, target);
        assert!(
            decimated.num_faces() <= target,
            "target {target}: {} faces",
            decimated.num_faces()
        );
        assert_oriented_manifold(&decimated);
        assert_eq!(euler_characteristic(&decimated), 0, "target {target}");
        assert_eq!(boundary_loop_count(&decimated), 2, "target {target}");
        for &face in &decimated.faces {
            let [a, b, c] = face;
            let centroid = vec3_scale(
                vec3_add(
                    vec3_add(decimated.verts[a], decimated.verts[b]),
                    decimated.verts[c],
                ),
                1.0 / 3.0,
            );
            let outward = [centroid[0], centroid[1], 0.0];
            assert!(
                vec3_dot(area_vector_of(&decimated, face), outward) > 0.0,
                "target {target}: face {face:?} points inwards"
            );
        }
        // The rims stay planar circles at the two ends of the tube.
        for (u, v) in boundary_edges(&decimated) {
            let (p, q) = (decimated.verts[u], decimated.verts[v]);
            assert!((p[2] - q[2]).abs() < 1e-12, "rim edge {u}-{v} is slanted");
            assert!(p[2].abs() < 1e-12 || (p[2] - 2.0).abs() < 1e-12);
        }
    }
}

#[test]
fn octahedron_decimation_stops_at_a_valid_closed_mesh() {
    let octahedron = octahedron();
    assert_eq!(octahedron.num_faces(), 8);
    assert!((enclosed_volume(&octahedron) - 4.0 / 3.0).abs() < 1e-12);
    for target in [7usize, 6, 5, 4, 0] {
        let decimated = qem_decimate(&octahedron, target);
        assert!(
            decimated.num_faces() == 4 || decimated.num_faces() == 6,
            "target {target}: {} faces",
            decimated.num_faces()
        );
        if target >= 4 {
            assert!(decimated.num_faces() <= target);
        }
        assert_oriented_manifold(&decimated);
        assert!(boundary_edges(&decimated).is_empty(), "target {target}");
        assert_eq!(euler_characteristic(&decimated), 2, "target {target}");
        assert_faces_point_outwards(&decimated);
        assert!(enclosed_volume(&decimated) > 0.0);
    }
}

#[test]
fn icosphere_decimation_is_valid_for_every_target_size() {
    let sphere = make_icosphere(2);
    for target in [
        300usize, 250, 200, 150, 120, 80, 60, 40, 30, 20, 12, 8, 6, 4,
    ] {
        let decimated = qem_decimate(&sphere, target);
        assert!(
            decimated.num_faces() <= target.max(4),
            "target {target}: {} faces",
            decimated.num_faces()
        );
        assert_oriented_manifold(&decimated);
        assert!(boundary_edges(&decimated).is_empty(), "target {target}");
        assert_eq!(euler_characteristic(&decimated), 2, "target {target}");
        assert_faces_point_outwards(&decimated);
    }
}

// ---------------------------------------------------------------------------
// fix_non_manifold_vertices
// ---------------------------------------------------------------------------

/// Number of edge-connected fans of faces around every vertex (a manifold vertex
/// has at most one).  Faces with a repeated vertex index have no area and are
/// ignored, like the repair does.
fn fan_counts(mesh: &ProcessMesh) -> Vec<usize> {
    (0..mesh.verts.len())
        .map(|v| {
            let faces: Vec<Face> = mesh
                .faces
                .iter()
                .copied()
                .filter(|f| f.contains(&v) && f[0] != f[1] && f[1] != f[2] && f[0] != f[2])
                .collect();
            let mut label: Vec<usize> = (0..faces.len()).collect();
            loop {
                let mut changed = false;
                for i in 0..faces.len() {
                    for j in (i + 1)..faces.len() {
                        let adjacent = faces[i].iter().any(|&w| w != v && faces[j].contains(&w));
                        if adjacent && label[i] != label[j] {
                            let (keep, drop) = (label[i].min(label[j]), label[i].max(label[j]));
                            for l in label.iter_mut() {
                                if *l == drop {
                                    *l = keep;
                                }
                            }
                            changed = true;
                        }
                    }
                }
                if !changed {
                    break;
                }
            }
            label.sort_unstable();
            label.dedup();
            label.len()
        })
        .collect()
}

/// Apex vertex 0 at the origin with a fan of `count` triangles around the `z`
/// axis (a cone), vertices `first..first + count + 1` on the rim.
fn cone_fan(verts: &mut Vec<Vertex>, faces: &mut Vec<Face>, apex: usize, count: usize, z: f64) {
    let first = verts.len();
    for k in 0..=count {
        let angle = k as f64 * 0.5;
        verts.push([angle.cos(), angle.sin(), z]);
    }
    for k in 0..count {
        faces.push([apex, first + k, first + k + 1]);
    }
}

#[test]
fn two_triangles_sharing_only_a_vertex_get_one_extra_vertex() {
    let mesh = ProcessMesh::new(
        vec![
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [-1.0, 0.0, 0.0],
            [0.0, -1.0, 0.0],
        ],
        vec![[0, 1, 2], [0, 3, 4]],
    );
    assert_eq!(fan_counts(&mesh)[0], 2);
    let repaired = fix_non_manifold_vertices(&mesh);
    assert_eq!(repaired.num_verts(), 6, "exactly one vertex must be added");
    assert_eq!(
        repaired.verts[5], mesh.verts[0],
        "the copy sits at the pinch vertex"
    );
    assert_eq!(repaired.faces, vec![[0, 1, 2], [5, 3, 4]]);
    assert!(fan_counts(&repaired).iter().all(|&n| n <= 1));
}

#[test]
fn a_bow_tie_of_three_fans_gets_two_copies() {
    let mut verts = vec![[0.0, 0.0, 0.0]];
    let mut faces = Vec::new();
    cone_fan(&mut verts, &mut faces, 0, 1, 0.0);
    cone_fan(&mut verts, &mut faces, 0, 1, 1.0);
    cone_fan(&mut verts, &mut faces, 0, 1, 2.0);
    let mesh = ProcessMesh::new(verts, faces);
    assert_eq!(fan_counts(&mesh)[0], 3);
    let repaired = fix_non_manifold_vertices(&mesh);
    assert_eq!(repaired.num_verts(), mesh.num_verts() + 2);
    assert_eq!(repaired.faces[0][0], 0);
    assert_eq!(repaired.faces[1][0], mesh.num_verts());
    assert_eq!(repaired.faces[2][0], mesh.num_verts() + 1);
    assert_ne!(repaired.faces[1][0], repaired.faces[2][0]);
    assert!(fan_counts(&repaired).iter().all(|&n| n <= 1));
}

#[test]
fn edge_connected_triangles_stay_in_one_fan() {
    // Vertex 0 is the apex of two separate cones of three triangles each; only
    // the second cone may move to the copy, and it must move as a whole.
    let mut verts = vec![[0.0, 0.0, 0.0]];
    let mut faces = Vec::new();
    cone_fan(&mut verts, &mut faces, 0, 3, 0.0);
    cone_fan(&mut verts, &mut faces, 0, 3, 5.0);
    let mesh = ProcessMesh::new(verts, faces);
    assert_eq!(fan_counts(&mesh)[0], 2);
    let repaired = fix_non_manifold_vertices(&mesh);
    let copy = mesh.num_verts();
    assert_eq!(repaired.num_verts(), copy + 1);
    for face in &repaired.faces[..3] {
        assert_eq!(face[0], 0);
    }
    for face in &repaired.faces[3..] {
        assert_eq!(face[0], copy);
    }
    assert_eq!(repaired.verts[copy], [0.0, 0.0, 0.0]);
    assert!(fan_counts(&repaired).iter().all(|&n| n <= 1));
}

#[test]
fn manifold_meshes_are_returned_unchanged() {
    for mesh in [
        make_tetrahedron(),
        make_icosphere(1),
        make_quad(),
        planar_grid(4),
    ] {
        assert!(fan_counts(&mesh).iter().all(|&n| n <= 1));
        let repaired = fix_non_manifold_vertices(&mesh);
        assert_eq!(repaired.verts, mesh.verts);
        assert_eq!(repaired.faces, mesh.faces);
    }
}

#[test]
fn two_tetrahedra_touching_in_a_vertex_become_two_closed_shells() {
    let mut tetra_a = make_tetrahedron();
    let tetra_b = make_tetrahedron();
    // Move the second tetrahedron so that its vertex 0 coincides with vertex 3 of
    // the first one, and merge the two vertices into a single (pinch) vertex.
    let anchor = tetra_a.verts[3];
    let shift = vec3_sub(anchor, tetra_b.verts[0]);
    let offset = tetra_a.verts.len();
    let mut remap = vec![3usize];
    for (i, v) in tetra_b.verts.iter().enumerate().skip(1) {
        tetra_a.verts.push(vec3_add(*v, shift));
        remap.push(offset + i - 1);
    }
    for f in &tetra_b.faces {
        tetra_a.faces.push([remap[f[0]], remap[f[1]], remap[f[2]]]);
    }
    assert_eq!(fan_counts(&tetra_a)[3], 2, "vertex 3 is the pinch vertex");
    let repaired = fix_non_manifold_vertices(&tetra_a);
    assert_eq!(repaired.num_verts(), tetra_a.num_verts() + 1);
    assert!(fan_counts(&repaired).iter().all(|&n| n <= 1));
    assert_oriented_manifold(&repaired);
    assert!(boundary_edges(&repaired).is_empty());
    // Two separate closed shells: V - E + F = 2 + 2.
    assert_eq!(euler_characteristic(&repaired), 4);
}

#[test]
fn duplicated_vertices_carry_their_normals_and_uvs() {
    let mut mesh = ProcessMesh::new(
        vec![
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [-1.0, 0.0, 0.0],
            [0.0, -1.0, 0.0],
        ],
        vec![[0, 1, 2], [0, 3, 4]],
    );
    mesh.normals = Some(vec![[0.0, 0.0, 1.0]; 5]);
    mesh.uvs = Some(vec![
        [0.0, 0.0],
        [1.0, 0.0],
        [0.0, 1.0],
        [0.5, 0.5],
        [0.25, 0.75],
    ]);
    let repaired = fix_non_manifold_vertices(&mesh);
    let normals = repaired.normals.as_ref().map(|n| n.len());
    let uvs = repaired.uvs.as_ref();
    assert_eq!(normals, Some(6));
    assert_eq!(uvs.map(|u| u.len()), Some(6));
    assert_eq!(
        uvs.map(|u| u[5]),
        Some([0.0, 0.0]),
        "the copy repeats the UV of vertex 0"
    );

    // An attribute array that does not match the vertex count is dropped.
    mesh.normals = Some(vec![[0.0, 0.0, 1.0]; 2]);
    assert!(fix_non_manifold_vertices(&mesh).normals.is_none());
}

#[test]
fn faces_with_missing_vertices_do_not_panic_the_pinch_repair() {
    let mesh = ProcessMesh::new(
        vec![[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
        vec![[0, 1, 2], [0, 2, 9], [1, 1, 2]],
    );
    let repaired = fix_non_manifold_vertices(&mesh);
    assert_eq!(repaired.faces, mesh.faces);
    assert_eq!(repaired.verts, mesh.verts);
}

/// Deterministic pseudo-random numbers (64-bit LCG).
struct Lcg(u64);

impl Lcg {
    fn below(&mut self, n: usize) -> usize {
        self.0 = self
            .0
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        ((self.0 >> 33) % n as u64) as usize
    }
}

#[test]
fn random_triangle_soups_survive_pinch_repair_and_decimation() {
    let mut rng = Lcg(0xDEAD_BEEF_1234_5678);
    for case in 0..300usize {
        let vertex_count = 1 + rng.below(14);
        let verts: Vec<Vertex> = (0..vertex_count)
            .map(|i| {
                [
                    (i * 5 % 11) as f64 * 0.37,
                    (i * 7 % 13) as f64 * 0.29,
                    (i % 3) as f64 * 0.11,
                ]
            })
            .collect();
        let face_count = rng.below(24);
        let faces: Vec<Face> = (0..face_count)
            .map(|_| {
                [
                    rng.below(vertex_count),
                    rng.below(vertex_count),
                    rng.below(vertex_count),
                ]
            })
            .collect();
        let mesh = ProcessMesh::new(verts, faces);

        // Pinch repair: same faces, every vertex has a single fan afterwards.
        let repaired = fix_non_manifold_vertices(&mesh);
        assert_eq!(repaired.faces.len(), mesh.faces.len(), "case {case}");
        assert!(
            repaired
                .faces
                .iter()
                .flatten()
                .all(|&v| v < repaired.verts.len()),
            "case {case}"
        );
        assert!(
            fan_counts(&repaired).iter().all(|&n| n <= 1),
            "case {case}: a vertex still has several fans"
        );

        // Decimation: valid indices, no degenerate faces, never more faces.
        let target = rng.below(face_count + 1);
        let decimated = qem_decimate(&mesh, target);
        assert!(decimated.faces.len() <= mesh.faces.len(), "case {case}");
        assert!(
            decimated
                .faces
                .iter()
                .flatten()
                .all(|&v| v < decimated.verts.len()),
            "case {case}"
        );
        if decimated.faces.len() != mesh.faces.len() {
            for f in &decimated.faces {
                assert!(
                    f[0] != f[1] && f[1] != f[2] && f[0] != f[2],
                    "case {case}: degenerate face {f:?}"
                );
            }
        }
    }
}
