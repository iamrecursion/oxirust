// Copyright 2026 COOLJAPAN OU (Team KitaSan)
// SPDX-License-Identifier: Apache-2.0

//! Regression tests for the uniform remeshing operators: crack-free edge
//! splitting, rejection of unusable target lengths and randomised meshes.

use super::*;

// ---- Uniform remeshing: edge splitting ----

/// How often every undirected edge is used by a face.
fn edge_uses(faces: &[Face]) -> HashMap<(usize, usize), usize> {
    let mut uses = HashMap::new();
    for face in faces {
        for k in 0..3 {
            let a = face[k];
            let b = face[(k + 1) % 3];
            *uses.entry((a.min(b), a.max(b))).or_insert(0) += 1;
        }
    }
    uses
}

/// Length of the longest edge of the mesh.
fn longest_edge(vertices: &[Vertex], faces: &[Face]) -> f64 {
    edge_uses(faces)
        .keys()
        .map(|&(a, b)| edge_length(&vertices[a], &vertices[b]))
        .fold(0.0, f64::max)
}

/// Sum of the triangle areas.
fn mesh_area(vertices: &[Vertex], faces: &[Face]) -> f64 {
    faces
        .iter()
        .map(|f| triangle_area(&vertices[f[0]], &vertices[f[1]], &vertices[f[2]]))
        .sum()
}

/// Whether every face of the mesh faces `+z`.
fn all_faces_face_up(vertices: &[Vertex], faces: &[Face]) -> bool {
    faces
        .iter()
        .all(|f| face_normal(&vertices[f[0]], &vertices[f[1]], &vertices[f[2]])[2] > 0.5)
}

/// Checks that the mesh covers the rectangle `[0, w] x [0, h]` without cracks:
/// no edge is shared by more than two faces, and the edges used only once are
/// exactly the ones along the outline (their total length is the perimeter).
fn assert_crack_free_rectangle(vertices: &[Vertex], faces: &[Face], w: f64, h: f64) {
    let uses = edge_uses(faces);
    let mut outline_length = 0.0;
    for (&(a, b), &count) in &uses {
        assert!(count <= 2, "edge {a}-{b} is shared by {count} faces");
        if count == 1 {
            let (p, q) = (vertices[a], vertices[b]);
            let on_side = |axis: usize, value: f64| {
                (p[axis] - value).abs() < 1e-12 && (q[axis] - value).abs() < 1e-12
            };
            assert!(
                on_side(0, 0.0) || on_side(0, w) || on_side(1, 0.0) || on_side(1, h),
                "crack: boundary edge {a}-{b} ({p:?} - {q:?}) is not on the outline"
            );
            outline_length += edge_length(&p, &q);
        }
    }
    assert!(
        (outline_length - 2.0 * (w + h)).abs() < 1e-9,
        "outline length {outline_length}"
    );
    assert!((mesh_area(vertices, faces) - w * h).abs() < 1e-9);
    assert!(
        all_faces_face_up(vertices, faces),
        "a face lost its winding"
    );
}

#[test]
fn test_split_long_edges_leaves_no_cracks() {
    let (mut verts, mut faces) = unit_quad();
    let remesher = RemeshingUniform::new(0.3, 1);
    remesher.split_long_edges(&mut verts, &mut faces);
    assert!(
        longest_edge(&verts, &faces) <= 0.4 + 1e-12,
        "longest edge {}",
        longest_edge(&verts, &faces)
    );
    assert_crack_free_rectangle(&verts, &faces, 1.0, 1.0);
    // The shared diagonal is split once, so it stays interior everywhere; the
    // four sides are bisected twice (four segments each).
    let boundary = edge_uses(&faces).values().filter(|&&c| c == 1).count();
    assert_eq!(boundary, 16, "boundary segments");
    for face in &faces {
        for &idx in face {
            assert!(idx < verts.len());
        }
    }
}

#[test]
fn test_split_long_edges_with_unusable_target_returns_immediately() {
    for length in [0.0, -1.0, f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
        let (verts0, faces0) = unit_quad();
        let remesher = RemeshingUniform::new(length, 3);
        assert!(!remesher.has_valid_target(), "length {length}");

        let (mut verts, mut faces) = (verts0.clone(), faces0.clone());
        remesher.split_long_edges(&mut verts, &mut faces);
        assert_eq!((&verts, &faces), (&verts0, &faces0), "split, L = {length}");

        remesher.collapse_short_edges(&mut verts, &mut faces);
        assert_eq!(
            (&verts, &faces),
            (&verts0, &faces0),
            "collapse, L = {length}"
        );

        remesher.remesh_pass(&mut verts, &mut faces);
        assert_eq!((&verts, &faces), (&verts0, &faces0), "pass, L = {length}");

        IsotropicRemeshing::new(length, 3).remesh(&mut verts, &mut faces);
        assert_eq!(
            (&verts, &faces),
            (&verts0, &faces0),
            "pipeline, L = {length}"
        );
    }
    assert!(RemeshingUniform::new(0.5, 1).has_valid_target());
}

#[test]
fn test_split_one_edge_gives_two_triangles() {
    // Only the base (length 1) exceeds 4/3 * 0.5.
    let mut verts = vec![[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.5, 0.3, 0.0]];
    let mut faces = vec![[0, 1, 2]];
    RemeshingUniform::new(0.5, 1).split_long_edges(&mut verts, &mut faces);
    assert_eq!(verts.len(), 4);
    assert_eq!(verts[3], [0.5, 0.0, 0.0]);
    assert_eq!(faces, vec![[0, 3, 2], [3, 1, 2]]);
    assert!((mesh_area(&verts, &faces) - 0.15).abs() < 1e-12);
    assert!(all_faces_face_up(&verts, &faces));
}

#[test]
fn test_split_two_edges_gives_three_triangles_along_the_shorter_diagonal() {
    // Edges 2 and 2.5 exceed 4/3 * 1.35 = 1.8, the edge 1.5 does not.
    let mut verts = vec![[0.0, 0.0, 0.0], [2.0, 0.0, 0.0], [0.0, 1.5, 0.0]];
    let mut faces = vec![[0, 1, 2]];
    RemeshingUniform::new(1.35, 1).split_long_edges(&mut verts, &mut faces);
    assert_eq!(verts.len(), 5);
    assert_eq!(verts[3], [1.0, 0.0, 0.0]);
    assert_eq!(verts[4], [1.0, 0.75, 0.0]);
    // Corner triangle plus the quadrilateral cut along 0-4 (length 1.25), not
    // along 3-2 (length 1.80).
    assert_eq!(faces, vec![[3, 1, 4], [0, 3, 4], [0, 4, 2]]);
    assert!((mesh_area(&verts, &faces) - 1.5).abs() < 1e-12);
    assert!(all_faces_face_up(&verts, &faces));
    assert!(longest_edge(&verts, &faces) <= 1.8);
}

#[test]
fn test_split_three_edges_gives_four_triangles() {
    let h = 3.0_f64.sqrt() / 2.0;
    let mut verts = vec![[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.5, h, 0.0]];
    let mut faces = vec![[0, 1, 2]];
    RemeshingUniform::new(0.5, 1).split_long_edges(&mut verts, &mut faces);
    assert_eq!(verts.len(), 6);
    assert_eq!(faces, vec![[0, 3, 5], [3, 1, 4], [5, 4, 2], [3, 4, 5]]);
    assert!((mesh_area(&verts, &faces) - 3.0_f64.sqrt() / 4.0).abs() < 1e-12);
    assert!(all_faces_face_up(&verts, &faces));
}

#[test]
fn test_split_long_edges_repeats_until_every_edge_is_short() {
    // A thin strip, 10 long: needs several bisection rounds.
    let mut verts = vec![
        [0.0, 0.0, 0.0],
        [10.0, 0.0, 0.0],
        [10.0, 0.1, 0.0],
        [0.0, 0.1, 0.0],
    ];
    let mut faces = vec![[0, 1, 2], [0, 2, 3]];
    RemeshingUniform::new(0.3, 1).split_long_edges(&mut verts, &mut faces);
    assert!(longest_edge(&verts, &faces) <= 0.4 + 1e-12);
    assert_crack_free_rectangle(&verts, &faces, 10.0, 0.1);
}

#[test]
fn test_split_long_edges_is_crack_free_with_mixed_split_patterns() {
    // A grid with very different cell sizes, so that neighbouring faces split
    // one, two or three of their edges and share some of them.
    let xs = [0.0, 0.2, 0.5, 1.1, 2.0];
    let ys = [0.0, 0.4, 0.6, 1.5, 2.0];
    let stride = xs.len();
    let mut verts: Vec<Vertex> = Vec::new();
    for &y in &ys {
        for &x in &xs {
            verts.push([x, y, 0.0]);
        }
    }
    let mut faces: Vec<Face> = Vec::new();
    for j in 0..ys.len() - 1 {
        for i in 0..xs.len() - 1 {
            let v00 = j * stride + i;
            let (v10, v01, v11) = (v00 + 1, v00 + stride, v00 + stride + 1);
            if (i + j) % 2 == 0 {
                faces.push([v00, v10, v11]);
                faces.push([v00, v11, v01]);
            } else {
                faces.push([v00, v10, v01]);
                faces.push([v10, v11, v01]);
            }
        }
    }
    assert_crack_free_rectangle(&verts, &faces, 2.0, 2.0);
    RemeshingUniform::new(0.3, 1).split_long_edges(&mut verts, &mut faces);
    assert!(longest_edge(&verts, &faces) <= 0.4 + 1e-12);
    assert_crack_free_rectangle(&verts, &faces, 2.0, 2.0);
    // Interior edges are shared by exactly two faces.
    let uses = edge_uses(&faces);
    let interior = uses.values().filter(|&&c| c == 2).count();
    let boundary = uses.values().filter(|&&c| c == 1).count();
    assert_eq!(2 * interior + boundary, 3 * faces.len());
}

#[test]
fn test_split_long_edges_never_splits_edges_of_non_finite_length() {
    let mut verts = vec![[0.0, 0.0, 0.0], [f64::INFINITY, 0.0, 0.0], [0.0, 1.0, 0.0]];
    let mut faces = vec![[0, 1, 2]];
    RemeshingUniform::new(0.1, 1).split_long_edges(&mut verts, &mut faces);
    // Only the finite edge 2-0 is split; the infinite ones are left alone and
    // the bisection terminates.
    assert!(faces.len() >= 2);
    assert!(verts.len() < 100);
}

#[test]
fn test_random_meshes_are_split_down_to_four_thirds_of_the_target_length() {
    let mut state = 0xABCD_EF01_2345_6789_u64;
    let mut next_unit = || {
        state = state
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        (state >> 11) as f64 / (1u64 << 53) as f64
    };
    for case in 0..300usize {
        let vertex_count = 3 + (next_unit() * 9.0) as usize;
        let verts: Vec<Vertex> = (0..vertex_count)
            .map(|_| {
                [
                    next_unit() * 4.0 - 2.0,
                    next_unit() * 4.0 - 2.0,
                    next_unit(),
                ]
            })
            .collect();
        let faces: Vec<Face> = (0..(next_unit() * 20.0) as usize)
            .map(|_| {
                [
                    (next_unit() * vertex_count as f64) as usize,
                    (next_unit() * vertex_count as f64) as usize,
                    (next_unit() * vertex_count as f64) as usize,
                ]
            })
            .collect();
        let length = 0.5 + next_unit() * 2.0;
        let remesher = RemeshingUniform::new(length, 1);

        let (mut split_verts, mut split_faces) = (verts.clone(), faces.clone());
        remesher.split_long_edges(&mut split_verts, &mut split_faces);
        assert!(
            split_faces
                .iter()
                .flatten()
                .all(|&idx| idx < split_verts.len()),
            "case {case}"
        );
        assert!(
            longest_edge(&split_verts, &split_faces) <= 4.0 / 3.0 * length + 1e-9,
            "case {case}: longest edge {}",
            longest_edge(&split_verts, &split_faces)
        );

        let (mut piped_verts, mut piped_faces) = (verts.clone(), faces.clone());
        IsotropicRemeshing::new(length, 1).remesh(&mut piped_verts, &mut piped_faces);
        assert!(
            piped_faces
                .iter()
                .flatten()
                .all(|&idx| idx < piped_verts.len()),
            "case {case}"
        );
    }
}
