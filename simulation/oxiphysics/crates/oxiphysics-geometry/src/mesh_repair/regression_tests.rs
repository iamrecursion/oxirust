// Copyright 2026 COOLJAPAN OU (Team KitaSan)
// SPDX-License-Identifier: Apache-2.0

//! Regression tests for hole filling (winding of the fill, deterministic
//! boundary loops) and for the combinatorial orientation check and repair.

use super::*;
use std::collections::HashMap;

/// The unit cube `[0, 1]^3`; vertex `i` has the coordinates of the bits of `i`
/// (x = bit 0, y = bit 1, z = bit 2).
fn cube_vertices() -> Vec<[f64; 3]> {
    (0..8usize)
        .map(|i| [(i & 1) as f64, ((i >> 1) & 1) as f64, ((i >> 2) & 1) as f64])
        .collect()
}

/// Twelve outward-oriented triangles of the unit cube, two per face in the order
/// bottom, top, front, back, left, right.
fn outward_cube_triangles() -> Vec<[usize; 3]> {
    vec![
        [0, 2, 1],
        [1, 2, 3],
        [4, 5, 6],
        [5, 7, 6],
        [0, 1, 4],
        [1, 5, 4],
        [2, 6, 3],
        [3, 6, 7],
        [0, 4, 2],
        [2, 4, 6],
        [1, 3, 5],
        [3, 7, 5],
    ]
}

/// Outward-oriented regular-ish tetrahedron; the winding is derived from the
/// geometry (normal against the direction away from the centroid) so that it
/// does not depend on the functions under test.
fn outward_tetrahedron() -> (Vec<[f64; 3]>, Vec<[usize; 3]>) {
    let vertices = vec![
        [1.0, 1.0, 1.0],
        [-1.0, -1.0, 1.0],
        [-1.0, 1.0, -1.0],
        [1.0, -1.0, -1.0],
    ];
    let triangles = [[0, 1, 2], [0, 1, 3], [0, 2, 3], [1, 2, 3]]
        .into_iter()
        .map(|[a, b, c]| {
            let normal = cross(sub(vertices[b], vertices[a]), sub(vertices[c], vertices[a]));
            // The centroid of this tetrahedron is the origin.
            let away = add(add(vertices[a], vertices[b]), vertices[c]);
            if dot(normal, away) > 0.0 {
                [a, b, c]
            } else {
                [a, c, b]
            }
        })
        .collect();
    (vertices, triangles)
}

/// Volume enclosed by a closed oriented triangle mesh.
fn signed_volume(vertices: &[[f64; 3]], triangles: &[[usize; 3]]) -> f64 {
    triangles
        .iter()
        .map(|t| dot(vertices[t[0]], cross(vertices[t[1]], vertices[t[2]])))
        .sum::<f64>()
        / 6.0
}

/// Whether every directed edge is used exactly once and so is its reverse, i.e.
/// the mesh is closed and consistently oriented.
fn is_closed_and_oriented(triangles: &[[usize; 3]]) -> bool {
    let mut directed: HashMap<(usize, usize), usize> = HashMap::new();
    for t in triangles {
        for k in 0..3 {
            *directed.entry((t[k], t[(k + 1) % 3])).or_insert(0) += 1;
        }
    }
    directed
        .iter()
        .all(|(&(a, b), &n)| n == 1 && directed.get(&(b, a)) == Some(&1))
}

/// `true` when `a` and `b` are the same triangle up to a cyclic rotation.
fn same_cyclic_triangle(a: [usize; 3], b: [usize; 3]) -> bool {
    (0..3).any(|k| a[k] == b[0] && a[(k + 1) % 3] == b[1] && a[(k + 2) % 3] == b[2])
}

#[test]
fn test_meshes_used_by_the_regression_tests_are_well_formed() {
    let cube = outward_cube_triangles();
    assert!(is_closed_and_oriented(&cube));
    assert!((signed_volume(&cube_vertices(), &cube) - 1.0).abs() < 1e-12);
    let (vertices, tetrahedron) = outward_tetrahedron();
    assert!(is_closed_and_oriented(&tetrahedron));
    assert!(signed_volume(&vertices, &tetrahedron) > 0.0);
}

// ---------------------------------------------------------------------------
// Hole filling: winding of the fill triangles
// ---------------------------------------------------------------------------

#[test]
fn refilling_a_removed_triangle_restores_its_winding() {
    let vertices = cube_vertices();
    let cube = outward_cube_triangles();
    for removed in 0..cube.len() {
        let mut triangles = cube.clone();
        let hole = triangles.remove(removed);
        let loops = find_boundary_loops(&triangles);
        assert_eq!(loops.len(), 1, "removing face {removed}");
        assert_eq!(loops[0].len(), 3);
        assert_eq!(fill_boundary_holes(&vertices, &mut triangles), 1);
        assert_eq!(triangles.len(), cube.len());
        let fill = triangles[triangles.len() - 1];
        assert!(
            same_cyclic_triangle(fill, hole),
            "face {removed}: removed {hole:?}, refilled {fill:?}"
        );
        assert!(is_closed_and_oriented(&triangles), "face {removed}");
        assert!((signed_volume(&vertices, &triangles) - 1.0).abs() < 1e-12);
    }
}

#[test]
fn a_quadrilateral_hole_is_filled_with_outward_triangles() {
    let vertices = cube_vertices();
    // Drop the two triangles of the top face.
    let mut triangles = outward_cube_triangles();
    triangles.retain(|t| !(t.iter().all(|&v| v >= 4)));
    assert_eq!(triangles.len(), 10);
    let loops = find_boundary_loops(&triangles);
    assert_eq!(loops.len(), 1);
    assert_eq!(loops[0].len(), 4);
    assert_eq!(fill_boundary_holes(&vertices, &mut triangles), 1);
    assert_eq!(triangles.len(), 12);
    assert!(is_closed_and_oriented(&triangles));
    assert!((signed_volume(&vertices, &triangles) - 1.0).abs() < 1e-12);
    // The new top face points up.
    for t in &triangles[10..] {
        let normal = cross(
            sub(vertices[t[1]], vertices[t[0]]),
            sub(vertices[t[2]], vertices[t[0]]),
        );
        assert!(normal[2] > 0.0, "fill triangle {t:?} points down");
    }
}

#[test]
fn ear_clipping_a_concave_loop_follows_the_surrounding_winding() {
    // An L-shaped hole (area 5, one reflex corner) in a plate whose triangles face
    // +z.  The boundary loop runs the way the plate's triangles run along their
    // edges, i.e. clockwise around the hole, and the fill has to face +z as well.
    let vertices = vec![
        [0.0, 3.0, 0.0],
        [1.0, 3.0, 0.0],
        [1.0, 1.0, 0.0],
        [3.0, 1.0, 0.0],
        [3.0, 0.0, 0.0],
        [0.0, 0.0, 0.0],
    ];
    let boundary_loop = [0usize, 1, 2, 3, 4, 5];
    let fill = fill_hole_ear_clipping(&vertices, &boundary_loop);
    assert_eq!(fill.len(), 4);
    let mut area = 0.0;
    for t in &fill {
        let normal = cross(
            sub(vertices[t[1]], vertices[t[0]]),
            sub(vertices[t[2]], vertices[t[0]]),
        );
        assert!(normal[2] > 0.0, "fill triangle {t:?} faces down");
        area += 0.5 * normal[2];
    }
    assert!((area - 5.0).abs() < 1e-12, "area {area}");
    // Every loop edge a -> b is used as b -> a by exactly one fill triangle.
    for k in 0..boundary_loop.len() {
        let a = boundary_loop[k];
        let b = boundary_loop[(k + 1) % boundary_loop.len()];
        let uses = fill
            .iter()
            .filter(|t| (0..3).any(|j| t[j] == b && t[(j + 1) % 3] == a))
            .count();
        assert_eq!(uses, 1, "loop edge {a}->{b}");
    }
}

#[test]
fn mesh_repair_fills_holes_with_consistent_winding() {
    let vertices = cube_vertices();
    let mut open_cube = outward_cube_triangles();
    open_cube.retain(|t| !(t.iter().all(|&v| v < 4)));
    assert_eq!(open_cube.len(), 10);
    for ear in [false, true] {
        let mut repair = MeshRepair::new(vertices.clone(), open_cube.clone());
        let filled = if ear {
            repair.fill_holes_ear()
        } else {
            repair.fill_holes()
        };
        assert_eq!(filled, 1);
        assert_eq!(repair.triangles.len(), 12, "ear clipping = {ear}");
        assert!(
            is_closed_and_oriented(&repair.triangles),
            "ear clipping = {ear}"
        );
        assert!((signed_volume(&vertices, &repair.triangles) - 1.0).abs() < 1e-12);
    }
}

// ---------------------------------------------------------------------------
// Boundary loops
// ---------------------------------------------------------------------------

#[test]
fn boundary_loops_are_canonical_and_deterministic() {
    // Remove two disjoint triangles, so that the cube has two holes.
    let mut triangles = outward_cube_triangles();
    let second = triangles.remove(11);
    let first = triangles.remove(0);
    let expected = {
        let mut loops: Vec<Vec<usize>> = [first, second]
            .iter()
            .map(|t| {
                // The loop runs against the winding of the removed triangle.
                let mut reversed = vec![t[2], t[1], t[0]];
                let min_pos = (0..3).min_by_key(|&k| reversed[k]).unwrap_or(0);
                reversed.rotate_left(min_pos);
                reversed
            })
            .collect();
        loops.sort();
        loops
    };
    for _ in 0..25 {
        assert_eq!(find_boundary_loops(&triangles), expected);
    }
}

#[test]
fn a_boundary_with_a_pinch_vertex_is_split_into_simple_loops() {
    // Two triangles that touch in vertex 0 only.
    let triangles = vec![[0usize, 1, 2], [0, 3, 4]];
    let loops = find_boundary_loops(&triangles);
    assert_eq!(loops, vec![vec![0, 1, 2], vec![0, 3, 4]]);
}

#[test]
fn boundary_chains_that_do_not_close_are_not_reported() {
    // The two triangles run their shared edge in the same direction, so the
    // boundary edges 1->2, 2->0, 1->3, 3->0 form no closed loop.
    let triangles = vec![[0usize, 1, 2], [0, 1, 3]];
    assert!(find_boundary_loops(&triangles).is_empty());
}

// ---------------------------------------------------------------------------
// Orientation check and repair
// ---------------------------------------------------------------------------

#[test]
fn correctly_wound_closed_meshes_are_not_flagged_and_not_changed() {
    let cube_vertices = cube_vertices();
    let cube = outward_cube_triangles();
    let (tet_vertices, tetrahedron) = outward_tetrahedron();
    for (vertices, triangles) in [(cube_vertices, cube), (tet_vertices, tetrahedron)] {
        assert!(inconsistent_normal_faces(&vertices, &triangles).is_empty());
        let mut repaired = triangles.clone();
        assert_eq!(fix_face_normals(&vertices, &mut repaired), 0);
        assert_eq!(repaired, triangles);
        assert_eq!(fix_normal_consistency(&vertices, &triangles), triangles);
    }
}

#[test]
fn a_single_reversed_face_is_flagged_and_repaired() {
    let cube_vertices = cube_vertices();
    let cube = outward_cube_triangles();
    let (tet_vertices, tetrahedron) = outward_tetrahedron();
    for (vertices, triangles) in [(cube_vertices, cube), (tet_vertices, tetrahedron)] {
        for reversed in 0..triangles.len() {
            let mut damaged = triangles.clone();
            damaged[reversed].swap(1, 2);
            assert_eq!(
                inconsistent_normal_faces(&vertices, &damaged),
                vec![reversed],
                "reversed face {reversed}"
            );
            let mut repaired = damaged.clone();
            assert_eq!(
                fix_face_normals(&vertices, &mut repaired),
                1,
                "reversed face {reversed}"
            );
            assert_eq!(repaired, triangles, "reversed face {reversed}");
            assert_eq!(fix_normal_consistency(&vertices, &damaged), triangles);
        }
    }
}

#[test]
fn several_reversed_faces_of_a_closed_mesh_are_repaired() {
    let vertices = cube_vertices();
    let cube = outward_cube_triangles();
    // Three neighbouring faces around vertex 0 and one far away.
    let reversed = [0usize, 4, 8, 11];
    let mut damaged = cube.clone();
    for &i in &reversed {
        damaged[i].swap(1, 2);
    }
    let mut repaired = damaged.clone();
    assert_eq!(fix_face_normals(&vertices, &mut repaired), reversed.len());
    assert_eq!(repaired, cube);
    assert!(inconsistent_normal_faces(&vertices, &repaired).is_empty());
}

#[test]
fn an_inside_out_closed_mesh_is_turned_outward() {
    let vertices = cube_vertices();
    let cube = outward_cube_triangles();
    let inside_out: Vec<[usize; 3]> = cube.iter().map(|t| [t[0], t[2], t[1]]).collect();
    // Consistently wound, so nothing is flagged ...
    assert!(inconsistent_normal_faces(&vertices, &inside_out).is_empty());
    assert!(signed_volume(&vertices, &inside_out) < 0.0);
    // ... but the repair still turns the normals outward.
    let mut repaired = inside_out.clone();
    assert_eq!(fix_face_normals(&vertices, &mut repaired), cube.len());
    assert_eq!(repaired, cube);
}

#[test]
fn open_surfaces_keep_the_orientation_of_the_majority() {
    // A 3 x 3 grid of unit squares in the z = 0 plane, 18 triangles facing +z.
    let mut vertices = Vec::new();
    for j in 0..4 {
        for i in 0..4 {
            vertices.push([i as f64, j as f64, 0.0]);
        }
    }
    let mut triangles = Vec::new();
    for j in 0..3 {
        for i in 0..3 {
            let v00 = j * 4 + i;
            triangles.push([v00, v00 + 1, v00 + 5]);
            triangles.push([v00, v00 + 5, v00 + 4]);
        }
    }
    let mut damaged = triangles.clone();
    // The first triangle is the seed of the flood fill; it is the odd one out.
    damaged[0].swap(1, 2);
    let mut repaired = damaged.clone();
    assert_eq!(fix_face_normals(&vertices, &mut repaired), 1);
    assert_eq!(repaired, triangles);

    // The same repair with the damage elsewhere.
    let mut damaged = triangles.clone();
    damaged[7].swap(1, 2);
    damaged[8].swap(1, 2);
    let mut repaired = damaged.clone();
    assert_eq!(fix_face_normals(&vertices, &mut repaired), 2);
    assert_eq!(repaired, triangles);
}

#[test]
fn two_isolated_triangles_with_opposite_windings_are_reported_together() {
    let vertices = vec![
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.5, 1.0, 0.0],
        [0.5, -1.0, 0.0],
    ];
    let triangles = vec![[0usize, 1, 2], [0, 1, 3]];
    assert_eq!(inconsistent_normal_faces(&vertices, &triangles), vec![0, 1]);
    let mut repaired = triangles.clone();
    assert_eq!(fix_face_normals(&vertices, &mut repaired), 1);
    assert_eq!(repaired, vec![[0, 1, 2], [0, 3, 1]]);
    assert!(inconsistent_normal_faces(&vertices, &repaired).is_empty());
}

#[test]
fn separate_components_are_oriented_independently() {
    let mut vertices = cube_vertices();
    let cube = outward_cube_triangles();
    let (tet_vertices, tetrahedron) = outward_tetrahedron();
    let offset = vertices.len();
    for v in &tet_vertices {
        vertices.push([v[0] + 10.0, v[1], v[2]]);
    }
    let mut triangles = cube.clone();
    // The tetrahedron comes in inside out.
    for t in &tetrahedron {
        triangles.push([t[0] + offset, t[2] + offset, t[1] + offset]);
    }
    let mut repaired = triangles.clone();
    assert_eq!(
        fix_face_normals(&vertices, &mut repaired),
        tetrahedron.len()
    );
    assert_eq!(&repaired[..cube.len()], &cube[..]);
    for (fixed, original) in repaired[cube.len()..].iter().zip(tetrahedron.iter()) {
        assert_eq!(
            *fixed,
            [
                original[0] + offset,
                original[1] + offset,
                original[2] + offset
            ]
        );
    }
}

#[test]
fn degenerate_triangles_and_missing_vertices_are_tolerated() {
    let vertices = vec![[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]];
    // A triangle with a repeated vertex is ignored, and the orientation of an open
    // pair is purely combinatorial, so a vertex outside the array does not matter.
    let triangles = vec![[0usize, 0, 1], [0, 1, 2], [0, 1, 7]];
    assert!(inconsistent_normal_faces(&vertices, &triangles).contains(&1));
    let mut repaired = triangles.clone();
    let flipped = fix_face_normals(&vertices, &mut repaired);
    assert_eq!(flipped, 1);
    assert_eq!(repaired[0], [0, 0, 1]);
    assert_eq!(repaired[1], [0, 1, 2]);
    assert_eq!(repaired[2], [0, 7, 1]);
    assert!(inconsistent_normal_faces(&vertices, &repaired).is_empty());

    // A closed tetrahedron whose vertex array is too short cannot be volume-checked;
    // it is still made consistent and keeps the orientation of its first triangle.
    let tetrahedron = vec![[0usize, 2, 1], [0, 1, 3], [1, 2, 3], [0, 3, 2]];
    let short_vertices = vec![[0.0; 3]; 3];
    let mut damaged = tetrahedron.clone();
    damaged[2].swap(1, 2);
    assert_eq!(fix_face_normals(&short_vertices, &mut damaged), 1);
    assert_eq!(damaged, tetrahedron);
}

#[test]
fn closed_meshes_far_from_the_origin_keep_a_reliable_orientation() {
    // At a distance of 1e7 the volume terms measured from the origin are around
    // 1e21 while the volume itself is 1, so the sign has to be taken relative to
    // the mesh.
    let cube = outward_cube_triangles();
    let vertices: Vec<[f64; 3]> = cube_vertices()
        .iter()
        .map(|v| [v[0] + 1.0e7, v[1] - 2.0e7, v[2] + 3.0e7])
        .collect();
    let mut outward = cube.clone();
    assert_eq!(fix_face_normals(&vertices, &mut outward), 0);
    assert_eq!(outward, cube);
    let mut inside_out: Vec<[usize; 3]> = cube.iter().map(|t| [t[0], t[2], t[1]]).collect();
    assert_eq!(fix_face_normals(&vertices, &mut inside_out), cube.len());
    assert_eq!(inside_out, cube);
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
fn random_triangle_soups_keep_boundary_loops_and_orientation_repair_consistent() {
    let mut rng = Lcg(0x5EED_0BAD_CAFE_F00D);
    for case in 0..400usize {
        let vertex_count = 1 + rng.below(12);
        // Repeated and (every fourth case) out-of-range indices are allowed: none of
        // these routines may panic on them.
        let index_range = vertex_count + if case % 4 == 0 { 3 } else { 0 };
        let vertices: Vec<[f64; 3]> = (0..vertex_count)
            .map(|i| [i as f64, (i * i % 7) as f64, (i % 3) as f64])
            .collect();
        let face_count = rng.below(24);
        let triangles: Vec<[usize; 3]> = (0..face_count)
            .map(|_| {
                [
                    rng.below(index_range),
                    rng.below(index_range),
                    rng.below(index_range),
                ]
            })
            .collect();

        // Every reported loop is a simple cycle made of boundary edges.
        let mut directed: HashMap<(usize, usize), usize> = HashMap::new();
        let mut undirected: HashMap<(usize, usize), usize> = HashMap::new();
        for t in &triangles {
            for k in 0..3 {
                let (a, b) = (t[k], t[(k + 1) % 3]);
                *directed.entry((a, b)).or_insert(0) += 1;
                *undirected.entry((a.min(b), a.max(b))).or_insert(0) += 1;
            }
        }
        for boundary_loop in find_boundary_loops(&triangles) {
            assert!(boundary_loop.len() >= 3, "case {case}");
            let mut distinct = boundary_loop.clone();
            distinct.sort_unstable();
            distinct.dedup();
            assert_eq!(distinct.len(), boundary_loop.len(), "case {case}");
            for k in 0..boundary_loop.len() {
                let a = boundary_loop[k];
                let b = boundary_loop[(k + 1) % boundary_loop.len()];
                assert!(directed.contains_key(&(a, b)), "case {case}: edge {a}->{b}");
                assert_eq!(undirected.get(&(a.min(b), a.max(b))), Some(&1));
            }
        }

        // The orientation repair only ever swaps the last two indices of a face.
        let flagged = inconsistent_normal_faces(&vertices, &triangles);
        assert!(flagged.windows(2).all(|w| w[0] < w[1]), "case {case}");
        let mut repaired = triangles.clone();
        let flips = fix_face_normals(&vertices, &mut repaired);
        assert!(flips <= triangles.len(), "case {case}");
        let mut visibly_changed = 0;
        for (fixed, original) in repaired.iter().zip(triangles.iter()) {
            if fixed != original {
                visibly_changed += 1;
                assert_eq!(*fixed, [original[0], original[2], original[1]]);
            }
        }
        assert!(visibly_changed <= flips, "case {case}");
        assert_eq!(fix_normal_consistency(&vertices, &triangles), repaired);
    }
}

#[test]
fn random_soups_with_valid_indices_can_be_hole_filled_without_panicking() {
    let mut rng = Lcg(0x0DDB_A115_5EED_1234);
    for case in 0..300usize {
        let vertex_count = 3 + rng.below(10);
        let vertices: Vec<[f64; 3]> = (0..vertex_count)
            .map(|i| [(i * 5 % 11) as f64, (i * 7 % 13) as f64, (i % 2) as f64])
            .collect();
        let triangles: Vec<[usize; 3]> = (0..rng.below(20))
            .map(|_| {
                [
                    rng.below(vertex_count),
                    rng.below(vertex_count),
                    rng.below(vertex_count),
                ]
            })
            .collect();
        let mut filled = triangles.clone();
        let holes = fill_boundary_holes(&vertices, &mut filled);
        assert_eq!(holes, find_boundary_loops(&triangles).len(), "case {case}");
        assert!(filled.len() >= triangles.len());
        assert_eq!(&filled[..triangles.len()], &triangles[..]);
        assert!(filled.iter().flatten().all(|&v| v < vertex_count));
    }
}
