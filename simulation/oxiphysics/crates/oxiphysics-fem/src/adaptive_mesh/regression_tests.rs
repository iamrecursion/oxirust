//! Regression tests for the tetrahedral [`AdaptiveMesh`]: conforming refinement
//! (no hanging nodes, no duplicated midpoints, one pass over the mesh) and
//! boundary-preserving smoothing.

use super::*;
use std::collections::{BTreeSet, HashMap};

/// Unit cube split into `6·n³` positively oriented tetrahedra by the Kuhn
/// (Freudenthal) triangulation, which is conforming across cells.
fn kuhn_cube(n: usize) -> AdaptiveMesh {
    let stride = n + 1;
    let index = |i: usize, j: usize, k: usize| k * stride * stride + j * stride + i;
    let mut nodes = Vec::with_capacity(stride * stride * stride);
    for k in 0..=n {
        for j in 0..=n {
            for i in 0..=n {
                nodes.push([
                    i as f64 / n as f64,
                    j as f64 / n as f64,
                    k as f64 / n as f64,
                ]);
            }
        }
    }
    let axis_orders: [[usize; 3]; 6] = [
        [0, 1, 2],
        [0, 2, 1],
        [1, 0, 2],
        [1, 2, 0],
        [2, 0, 1],
        [2, 1, 0],
    ];
    let mut tets = Vec::new();
    for k in 0..n {
        for j in 0..n {
            for i in 0..n {
                for order in axis_orders {
                    let mut corner = [i, j, k];
                    let mut path = [index(i, j, k); 4];
                    for (step, axis) in order.into_iter().enumerate() {
                        corner[axis] += 1;
                        path[step + 1] = index(corner[0], corner[1], corner[2]);
                    }
                    let mut tet = path;
                    if tet_volume(&nodes, &tet) < 0.0 {
                        tet.swap(1, 2);
                    }
                    tets.push(tet);
                }
            }
        }
    }
    AdaptiveMesh::new(nodes, tets)
}

fn tet_volume(nodes: &[[f64; 3]], tet: &[usize; 4]) -> f64 {
    MeshQualityMetric::volume([nodes[tet[0]], nodes[tet[1]], nodes[tet[2]], nodes[tet[3]]])
}

fn total_volume(mesh: &AdaptiveMesh) -> f64 {
    mesh.tets
        .iter()
        .map(|tet| tet_volume(&mesh.nodes, tet))
        .sum()
}

/// Number of tetrahedra that contain each (sorted) triangular face.
fn face_counts(tets: &[[usize; 4]]) -> HashMap<[usize; 3], usize> {
    let mut counts = HashMap::new();
    for tet in tets {
        for skipped in 0..4 {
            let mut face: Vec<usize> = (0..4).filter(|&p| p != skipped).map(|p| tet[p]).collect();
            face.sort_unstable();
            *counts.entry([face[0], face[1], face[2]]).or_insert(0) += 1;
        }
    }
    counts
}

fn triangle_area(nodes: &[[f64; 3]], face: &[usize; 3]) -> f64 {
    let ab = [
        nodes[face[1]][0] - nodes[face[0]][0],
        nodes[face[1]][1] - nodes[face[0]][1],
        nodes[face[1]][2] - nodes[face[0]][2],
    ];
    let ac = [
        nodes[face[2]][0] - nodes[face[0]][0],
        nodes[face[2]][1] - nodes[face[0]][1],
        nodes[face[2]][2] - nodes[face[0]][2],
    ];
    let cross = [
        ab[1] * ac[2] - ab[2] * ac[1],
        ab[2] * ac[0] - ab[0] * ac[2],
        ab[0] * ac[1] - ab[1] * ac[0],
    ];
    0.5 * (cross[0] * cross[0] + cross[1] * cross[1] + cross[2] * cross[2]).sqrt()
}

/// `true` when the face lies in one of the six planes of the unit cube.
fn lies_on_cube_surface(nodes: &[[f64; 3]], face: &[usize; 3]) -> bool {
    (0..3).any(|axis| {
        [0.0, 1.0]
            .iter()
            .any(|&plane| face.iter().all(|&node| nodes[node][axis] == plane))
    })
}

/// Total area of the faces that belong to a single tetrahedron (the open boundary).
fn open_face_area(mesh: &AdaptiveMesh) -> f64 {
    face_counts(&mesh.tets)
        .iter()
        .filter(|&(_, &count)| count == 1)
        .map(|(face, _)| triangle_area(&mesh.nodes, face))
        .sum()
}

/// Length of the longest edge of a tetrahedron.
fn longest_edge_length(nodes: &[[f64; 3]], tet: &[usize; 4]) -> f64 {
    let corners: Vec<[f64; 3]> = tet.iter().map(|&node| nodes[node]).collect();
    let mut longest = 0.0_f64;
    for (i, a) in corners.iter().enumerate() {
        for b in &corners[i + 1..] {
            let squared = (a[0] - b[0]).powi(2) + (a[1] - b[1]).powi(2) + (a[2] - b[2]).powi(2);
            longest = longest.max(squared.sqrt());
        }
    }
    longest
}

/// Assert that `mesh` is a conforming tetrahedral mesh of the unit cube:
///
/// * no face belongs to more than two tetrahedra;
/// * every face that belongs to a single tetrahedron lies on the cube surface, so
///   there is no hanging node or open interior face;
/// * the volumes of the tetrahedra add up to one and none is inverted;
/// * no two nodes coincide.
fn assert_conforming_unit_cube(mesh: &AdaptiveMesh) {
    let counts = face_counts(&mesh.tets);
    assert!(
        counts.values().all(|&count| count <= 2),
        "a face is shared by more than two tetrahedra"
    );
    let mut surface_area = 0.0;
    for (face, &count) in &counts {
        if count == 1 {
            assert!(
                lies_on_cube_surface(&mesh.nodes, face),
                "open interior face {face:?}: hanging node"
            );
            surface_area += triangle_area(&mesh.nodes, face);
        }
    }
    assert!(
        (surface_area - 6.0).abs() < 1e-12,
        "boundary area {surface_area}, expected 6"
    );
    assert!(
        (total_volume(mesh) - 1.0).abs() < 1e-12,
        "total volume {}",
        total_volume(mesh)
    );
    assert!(
        mesh.tets
            .iter()
            .all(|tet| tet_volume(&mesh.nodes, tet) > 0.0),
        "an element is inverted or degenerate"
    );
    assert_no_coincident_nodes(&mesh.nodes);
    assert_eq!(mesh.refinement_level.len(), mesh.tets.len());
}

fn assert_no_coincident_nodes(nodes: &[[f64; 3]]) {
    for (i, a) in nodes.iter().enumerate() {
        for (j, b) in nodes.iter().enumerate().skip(i + 1) {
            assert!(a != b, "nodes {i} and {j} coincide at {a:?}");
        }
    }
}

fn sorted(tet: [usize; 4]) -> [usize; 4] {
    let mut copy = tet;
    copy.sort_unstable();
    copy
}

/// Deterministic pseudo-random numbers in `[0, 1)`.
fn lcg_stream(seed: u64) -> impl FnMut() -> f64 {
    let mut state = seed;
    move || {
        state = state
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        (state >> 11) as f64 / (1_u64 << 53) as f64
    }
}

/// Two tetrahedra on opposite sides of the face `(0, 1, 2)`, which contains the
/// longest edge `(0, 1)` of both.
fn two_tets_sharing_their_longest_edge() -> AdaptiveMesh {
    let nodes = vec![
        [0.0, 0.0, 0.0],
        [2.0, 0.0, 0.0],
        [1.0, 1.0, 0.0],
        [1.0, 0.3, 0.8],
        [1.0, 0.3, -0.8],
    ];
    AdaptiveMesh::new(nodes, vec![[0, 1, 2, 3], [0, 1, 4, 2]])
}

#[test]
fn the_kuhn_test_mesh_is_a_valid_conforming_cube() {
    for n in [1, 2, 3] {
        let mesh = kuhn_cube(n);
        assert_eq!(mesh.tets.len(), 6 * n * n * n);
        assert_conforming_unit_cube(&mesh);
    }
}

#[test]
fn uniform_refinement_of_two_tets_shares_one_midpoint() {
    // The old code split each tetrahedron on its own: two coincident midpoint nodes
    // and an open interior face.
    let mut mesh = two_tets_sharing_their_longest_edge();
    let volume = total_volume(&mesh);
    let errors = vec![0.0; 2];
    mesh.refine_to_criterion(&RefinementCriterion::Uniform, &errors);

    assert_eq!(mesh.nodes.len(), 6, "exactly one midpoint node");
    assert_eq!(mesh.nodes[5], [1.0, 0.0, 0.0]);
    assert_no_coincident_nodes(&mesh.nodes);
    assert_eq!(mesh.tets.len(), 4);
    assert!(mesh.refinement_level.iter().all(|&level| level == 1));
    assert!((total_volume(&mesh) - volume).abs() < 1e-14);

    // Every interior face is shared by exactly two tetrahedra: the four faces
    // through the midpoint and the original common face halves.
    let counts = face_counts(&mesh.tets);
    assert!(counts.values().all(|&count| count <= 2));
    let interior = counts.values().filter(|&&count| count == 2).count();
    assert_eq!(interior, 4, "{counts:?}");
    assert_eq!(counts.len(), 12);
    // The boundary is unchanged by refining: the two tetrahedra still bound the
    // same surface.
    let before = two_tets_sharing_their_longest_edge();
    assert!((open_face_area(&mesh) - open_face_area(&before)).abs() < 1e-14);
}

#[test]
fn refining_one_tet_also_splits_the_neighbour_across_its_longest_edge() {
    let mut mesh = two_tets_sharing_their_longest_edge();
    mesh.refine_element(0);
    // The shared edge (0, 1) is the longest edge of both tetrahedra: both are cut.
    assert_eq!(mesh.tets.len(), 4);
    assert_eq!(mesh.nodes.len(), 6);
    assert_eq!(mesh.refinement_level, vec![1, 1, 1, 1]);
    let counts = face_counts(&mesh.tets);
    assert_eq!(counts.values().filter(|&&count| count == 2).count(), 4);
    // The first child keeps the slot of the parent.
    assert!(mesh.tets[0].contains(&0) && !mesh.tets[0].contains(&1));
}

#[test]
fn an_edge_that_no_neighbour_shares_splits_only_its_own_tetrahedron() {
    // The longest edge (2, 3) of tetrahedron 0 belongs to no other tetrahedron, so
    // only tetrahedron 0 is split, although its neighbour has far longer edges: the
    // bisection leaves the shared face (0, 1, 2) intact and the mesh stays conforming.
    let nodes = vec![
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.5, 1.0, 0.0],
        [0.5, 0.333, 1.0],
        [0.5, 0.333, -3.0],
    ];
    let mut mesh = AdaptiveMesh::new(nodes, vec![[0, 1, 2, 3], [0, 1, 4, 2]]);
    mesh.refine_element(0);
    assert_eq!(mesh.tets.len(), 3);
    assert_eq!(mesh.nodes.len(), 6);
    assert_eq!(mesh.refinement_level, vec![1, 0, 1]);
    // The neighbour kept its connectivity, only tetrahedron 0 was split.
    assert_eq!(mesh.tets[1], [0, 1, 4, 2]);
    assert!(face_counts(&mesh.tets).values().all(|&count| count <= 2));
}

#[test]
fn a_neighbour_with_a_longer_edge_is_refined_first() {
    // Tetrahedra 0 and 1 share the edge (0, 1), the longest edge of tetrahedron 0.
    // Tetrahedron 1 has a longer edge (1, 4): refining tetrahedron 0 must bisect that
    // edge first and then the shared one, never leaving a hanging node.
    let nodes = vec![
        [0.0, 0.0, 0.0],
        [2.0, 0.0, 0.0],
        [1.0, 0.9, 0.0],
        [1.0, 0.3, 0.8],
        [1.0, 0.3, -2.5],
    ];
    let original = AdaptiveMesh::new(nodes, vec![[0, 1, 2, 3], [0, 1, 4, 2]]);
    let mut mesh = AdaptiveMesh::new(original.nodes.clone(), original.tets.clone());
    mesh.refine_element(0);
    // Tetrahedron 1 is cut through (1, 4), then both tetrahedra through (0, 1).
    assert_eq!(mesh.tets.len(), 5);
    assert_eq!(mesh.nodes.len(), 7);
    assert!(face_counts(&mesh.tets).values().all(|&count| count <= 2));
    // The open (boundary) faces are pieces of the original ones: same total area.
    assert!(
        (open_face_area(&mesh) - open_face_area(&original)).abs() < 1e-12,
        "open area {} vs original {}",
        open_face_area(&mesh),
        open_face_area(&original)
    );
    assert!((total_volume(&mesh) - total_volume(&original)).abs() < 1e-12);
    assert_no_coincident_nodes(&mesh.nodes);
    // The propagation refined the neighbour although only element 0 was requested.
    assert!(mesh.refinement_level.iter().all(|&level| level >= 1));
}

#[test]
fn refinement_preserves_the_orientation_of_both_children() {
    // One child of the old code was inverted.
    for flip in [false, true] {
        let mut mesh = two_tets_sharing_their_longest_edge();
        if flip {
            for tet in &mut mesh.tets {
                tet.swap(0, 1);
            }
        }
        let parent_sign = tet_volume(&mesh.nodes, &mesh.tets[0]).signum();
        mesh.refine_element(0);
        for tet in &mesh.tets {
            let volume = tet_volume(&mesh.nodes, tet);
            assert_eq!(volume.signum(), parent_sign, "flip = {flip}: {volume}");
            assert!(volume.abs() > 1e-6);
        }
    }
}

#[test]
fn refinement_levels_saturate_instead_of_overflowing() {
    let mut mesh = two_tets_sharing_their_longest_edge();
    mesh.refinement_level = vec![255, 3];
    mesh.refine_element(0);
    assert_eq!(mesh.refinement_level.len(), mesh.tets.len());
    // Both tetrahedra share the edge, so all four children descend from them.
    assert_eq!(
        mesh.refinement_level.iter().filter(|&&l| l == 255).count(),
        2
    );
    assert_eq!(mesh.refinement_level.iter().filter(|&&l| l == 4).count(), 2);
}

#[test]
fn out_of_range_and_degenerate_requests_leave_the_mesh_alone() {
    let mut mesh = two_tets_sharing_their_longest_edge();
    let before = (mesh.nodes.clone(), mesh.tets.clone());
    mesh.refine_element(2);
    mesh.refine_element(usize::MAX);
    assert_eq!((mesh.nodes.clone(), mesh.tets.clone()), before);

    // A tetrahedron with coincident nodes cannot be bisected.
    let nodes = vec![
        [0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
    ];
    let mut degenerate = AdaptiveMesh::new(nodes, vec![[0, 1, 2, 3]]);
    degenerate.refine_element(0);
    assert_eq!(degenerate.tets, vec![[0, 1, 2, 3]]);
    assert_eq!(degenerate.nodes.len(), 4);
}

#[test]
fn uniform_refinement_of_a_cube_mesh_stays_conforming() {
    let mut mesh = kuhn_cube(2);
    let before = mesh.tets.len();
    let originals: Vec<[usize; 4]> = mesh.tets.iter().copied().map(sorted).collect();
    let errors = vec![0.0; before];
    mesh.refine_to_criterion(&RefinementCriterion::Uniform, &errors);
    assert_conforming_unit_cube(&mesh);
    assert!(mesh.tets.len() >= 2 * before);
    // Every original tetrahedron was bisected: none of them survives.
    for tet in &mesh.tets {
        assert!(
            !originals.contains(&sorted(*tet)),
            "{tet:?} was not refined"
        );
    }
    assert!(mesh.refinement_level.iter().all(|&level| level >= 1));
}

#[test]
fn randomly_selected_elements_are_refined_without_hanging_nodes() {
    let mut mesh = kuhn_cube(3);
    let count = mesh.tets.len();
    let originals: Vec<[usize; 4]> = mesh.tets.iter().copied().map(sorted).collect();
    let mut next = lcg_stream(20260622);
    let errors: Vec<f64> = (0..count).map(|_| next()).collect();
    let threshold = 0.8;
    mesh.refine_to_criterion(&RefinementCriterion::ErrorBased { threshold }, &errors);
    assert_conforming_unit_cube(&mesh);
    assert!(mesh.tets.len() > count);
    // Every selected tetrahedron was bisected...
    for (index, error) in errors.iter().enumerate() {
        if *error > threshold {
            assert!(
                !originals_present(&mesh, originals[index]),
                "selected tetrahedron {index} survived"
            );
        }
    }
    // ...and the refinement stayed local: far fewer elements than a uniform pass.
    assert!(mesh.tets.len() < 3 * count);
}

/// `true` when a tetrahedron with the (sorted) vertex set `original` is still present.
fn originals_present(mesh: &AdaptiveMesh, original: [usize; 4]) -> bool {
    mesh.tets.iter().any(|tet| sorted(*tet) == original)
}

#[test]
fn repeated_refinement_passes_keep_the_mesh_conforming() {
    let mut mesh = kuhn_cube(2);
    let mut next = lcg_stream(7);
    for _ in 0..4 {
        let errors: Vec<f64> = (0..mesh.tets.len()).map(|_| next()).collect();
        mesh.refine_to_criterion(&RefinementCriterion::ErrorBased { threshold: 0.6 }, &errors);
        assert_conforming_unit_cube(&mesh);
    }
    assert!(mesh.refinement_level.iter().any(|&level| level >= 2));
}

#[test]
fn irregular_meshes_stay_conforming_over_many_refinement_passes() {
    // Perturbed interior nodes remove the equal edge lengths of the Kuhn mesh, so
    // the propagation paths become long and irregular; every pass must still leave a
    // conforming mesh of the same volume and boundary, with nothing inverted.
    let mut mesh = perturbed_kuhn_cube(2, 0.04);
    assert_conforming_unit_cube(&mesh);
    let mut next = lcg_stream(31415);
    for pass in 0..5 {
        let errors: Vec<f64> = (0..mesh.tets.len()).map(|_| next()).collect();
        let selected: Vec<[usize; 4]> = errors
            .iter()
            .zip(&mesh.tets)
            .filter(|&(&error, _)| error > 0.75)
            .map(|(_, tet)| sorted(*tet))
            .collect();
        mesh.refine_to_criterion(
            &RefinementCriterion::ErrorBased { threshold: 0.75 },
            &errors,
        );
        assert_conforming_unit_cube(&mesh);
        for tet in &selected {
            assert!(
                !originals_present(&mesh, *tet),
                "pass {pass}: a selected tetrahedron survived"
            );
        }
    }
    assert!(mesh.refinement_level.iter().any(|&level| level >= 2));
}

#[test]
fn geometric_criterion_selects_by_edge_length_and_stays_conforming() {
    // Every Kuhn tetrahedron of the 2x2x2 cube contains a cell diagonal of length
    // 0.866 as its longest edge and the six tetrahedra of a cell share it, so one
    // pass bisects each cell's tetrahedra together: 8 cells x 12 children.
    let mut mesh = kuhn_cube(2);
    assert!(
        mesh.tets
            .iter()
            .all(|tet| longest_edge_length(&mesh.nodes, tet) > 0.8)
    );
    mesh.refine_to_criterion(
        &RefinementCriterion::GeometricBased { target_size: 0.8 },
        &[],
    );
    assert_conforming_unit_cube(&mesh);
    assert_eq!(mesh.tets.len(), 96);
    assert!(
        mesh.tets
            .iter()
            .all(|tet| longest_edge_length(&mesh.nodes, tet) <= 0.8)
    );
    // A second pass has to bisect face diagonals that neighbouring cells share.
    mesh.refine_to_criterion(
        &RefinementCriterion::GeometricBased { target_size: 0.6 },
        &[],
    );
    assert_conforming_unit_cube(&mesh);
    assert!(mesh.tets.len() > 96);
}

#[test]
fn errors_beyond_the_element_count_and_nan_errors_select_nothing_extra() {
    let mut mesh = kuhn_cube(1);
    let before = mesh.tets.len();
    let errors = vec![f64::NAN; before + 5];
    mesh.refine_to_criterion(&RefinementCriterion::ErrorBased { threshold: 0.1 }, &errors);
    assert_eq!(mesh.tets.len(), before);
    let mut errors = vec![0.0; before + 5];
    errors[before + 2] = 10.0;
    mesh.refine_to_criterion(&RefinementCriterion::ErrorBased { threshold: 0.1 }, &errors);
    assert_eq!(mesh.tets.len(), before);
}

#[test]
fn a_larger_mesh_is_refined_in_one_conforming_pass() {
    // 3072 tetrahedra; the former element-by-element rebuild shifted the whole
    // element list once per refined tetrahedron.
    let mut mesh = kuhn_cube(8);
    let count = mesh.tets.len();
    assert_eq!(count, 3072);
    let errors = vec![0.0; count];
    mesh.refine_to_criterion(&RefinementCriterion::Uniform, &errors);
    assert_conforming_unit_cube(&mesh);
    assert!(mesh.tets.len() >= 2 * count);
}

#[test]
fn quality_of_refined_children_is_bounded_below() {
    // Longest-edge bisection of a regular-ish mesh must not produce slivers.
    let mut mesh = kuhn_cube(2);
    let errors = vec![0.0; mesh.tets.len()];
    let initial_quality = mesh.min_quality();
    mesh.refine_to_criterion(&RefinementCriterion::Uniform, &errors);
    assert!(mesh.min_quality() > 0.5 * initial_quality);
}

// ---- smoothing -----------------------------------------------------------

/// Kuhn cube whose interior nodes are displaced by a deterministic amount.
fn perturbed_kuhn_cube(n: usize, amplitude: f64) -> AdaptiveMesh {
    let mut mesh = kuhn_cube(n);
    let mut next = lcg_stream(99);
    for node in &mut mesh.nodes {
        let interior = node.iter().all(|&c| c > 0.0 && c < 1.0);
        if interior {
            for c in node.iter_mut() {
                *c += amplitude * (2.0 * next() - 1.0);
            }
        }
    }
    mesh
}

fn is_surface_node(node: &[f64; 3]) -> bool {
    node.iter().any(|&c| c == 0.0 || c == 1.0)
}

#[test]
fn smoothing_keeps_the_boundary_nodes_bit_identical() {
    let mut mesh = perturbed_kuhn_cube(3, 0.05);
    assert_conforming_unit_cube(&mesh);
    let before = mesh.nodes.clone();
    mesh.smooth(5);
    let mut moved_interior = 0;
    for (old, new) in before.iter().zip(&mesh.nodes) {
        if is_surface_node(old) {
            assert_eq!(old, new, "a boundary node moved");
        } else if old != new {
            moved_interior += 1;
        }
    }
    assert!(moved_interior > 0, "smoothing moved no interior node");
    // With the boundary fixed the enclosed volume cannot change.
    assert!(
        (total_volume(&mesh) - 1.0).abs() < 1e-12,
        "volume {}",
        total_volume(&mesh)
    );
    assert!(
        mesh.tets
            .iter()
            .all(|tet| tet_volume(&mesh.nodes, tet) > 0.0)
    );
}

#[test]
fn smoothing_does_not_shrink_a_single_tetrahedron() {
    // Every node of a lone tetrahedron is a boundary node; the former all-false
    // mask collapsed it towards its centroid.
    let nodes = vec![
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
    ];
    let mut mesh = AdaptiveMesh::new(nodes.clone(), vec![[0, 1, 2, 3]]);
    mesh.smooth(10);
    assert_eq!(mesh.nodes, nodes);
}

#[test]
fn smoothing_moves_an_interior_node_to_the_average_of_its_neighbours() {
    let mut mesh = perturbed_kuhn_cube(2, 0.1);
    // The only interior node of the 2x2x2 cube is its centre.
    let centre = mesh
        .nodes
        .iter()
        .position(|node| !is_surface_node(node))
        .expect("the 2x2x2 cube has one interior node");
    let neighbours: BTreeSet<usize> = mesh
        .tets
        .iter()
        .filter(|tet| tet.contains(&centre))
        .flat_map(|tet| tet.iter().copied())
        .filter(|&node| node != centre)
        .collect();
    let mut expected = [0.0_f64; 3];
    for &node in &neighbours {
        for (sum, &coordinate) in expected.iter_mut().zip(&mesh.nodes[node]) {
            *sum += coordinate;
        }
    }
    for value in &mut expected {
        *value /= neighbours.len() as f64;
    }
    mesh.smooth(1);
    for (got, want) in mesh.nodes[centre].iter().zip(&expected) {
        assert!((got - want).abs() < 1e-15, "{got} vs {want}");
    }
}

#[test]
fn smoothing_is_reproducible_and_ignores_unused_nodes() {
    let build = || {
        let mut mesh = perturbed_kuhn_cube(3, 0.05);
        // A node that belongs to no tetrahedron has no neighbours and stays put.
        mesh.nodes.push([5.0, 5.0, 5.0]);
        mesh
    };
    let mut first = build();
    let mut second = build();
    first.smooth(4);
    second.smooth(4);
    assert_eq!(first.nodes, second.nodes);
    assert_eq!(first.nodes.last(), Some(&[5.0, 5.0, 5.0]));
}

#[test]
fn smoothing_tolerates_tets_that_reference_missing_nodes() {
    let nodes = vec![[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]];
    let mut mesh = AdaptiveMesh::new(nodes.clone(), vec![[0, 1, 2, 7]]);
    mesh.smooth(2);
    assert_eq!(mesh.nodes, nodes);
}
