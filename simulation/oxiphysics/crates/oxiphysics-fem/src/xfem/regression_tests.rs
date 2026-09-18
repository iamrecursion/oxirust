//! Regression tests for the XFEM module.
//!
//! Covers the polyline level set / Heaviside function of kinked cracks, the
//! sub-triangulation of cut triangles, the K_I extraction from crack-face
//! displacements and the Heaviside node flags of a mesh.

use super::*;
use std::f64::consts::PI;

// ── helpers ──────────────────────────────────────────────────────────────────

/// Crack (0,0) -> (1,0) that then turns left by 90 degrees to (1,1).
fn left_kink_crack() -> Crack {
    let mut crack = Crack::new([0.0, 0.0, 0.0], [1.0, 0.0, 0.0]);
    crack.propagate(1.0, [0.0, 1.0, 0.0]);
    crack
}

/// Crack (0,0) -> (1,0) that then turns right by 90 degrees to (1,-1).
fn right_kink_crack() -> Crack {
    let mut crack = Crack::new([0.0, 0.0, 0.0], [1.0, 0.0, 0.0]);
    crack.propagate(1.0, [0.0, -1.0, 0.0]);
    crack
}

/// Zig-zag crack (0,0) -> (1,0) -> (1,1) -> (2,1).
fn zigzag_crack() -> Crack {
    let mut crack = left_kink_crack();
    crack.propagate(1.0, [1.0, 0.0, 0.0]);
    crack
}

/// Nodes of a regular `n x n` element grid on the unit square (index `j*(n+1)+i`).
fn grid_nodes(n: usize) -> Vec<[f64; 3]> {
    let h = 1.0 / n as f64;
    let mut nodes = Vec::with_capacity((n + 1) * (n + 1));
    for j in 0..=n {
        for i in 0..=n {
            nodes.push([i as f64 * h, j as f64 * h, 0.0]);
        }
    }
    nodes
}

/// Node index of grid node `(i, j)`.
fn node_id(n: usize, i: usize, j: usize) -> usize {
    j * (n + 1) + i
}

/// Quadrilateral connectivity of the `n x n` grid, element `(i, j)` at `j*n+i`.
fn grid_elements(n: usize) -> Vec<Vec<usize>> {
    let mut elements = Vec::with_capacity(n * n);
    for j in 0..n {
        for i in 0..n {
            elements.push(vec![
                node_id(n, i, j),
                node_id(n, i + 1, j),
                node_id(n, i + 1, j + 1),
                node_id(n, i, j + 1),
            ]);
        }
    }
    elements
}

/// Crack-tip flags: nodes within `radius` of the tip in both `psi` and `phi`.
fn tip_flags(ls: &LevelSet, radius: f64) -> Vec<bool> {
    (0..ls.phi.len())
        .map(|node| ls.needs_tip_enrichment(node, radius))
        .collect()
}

/// Deterministic pseudo-random numbers in `[0, 1)` (64-bit LCG).
struct Lcg(u64);
impl Lcg {
    fn next_f64(&mut self) -> f64 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        (self.0 >> 11) as f64 / (1u64 << 53) as f64
    }
}

// ── F1412: polyline level set and Heaviside function ────────────────────────

#[test]
fn heaviside_separates_both_faces_of_a_kinked_crack() {
    let crack = left_kink_crack();
    // The faces of the first segment: the old code used the infinite line of
    // the last segment (x = 1) and returned +1 for both points.
    assert_eq!(heaviside_enrichment([0.5, -0.5, 0.0], &crack), -1.0);
    assert_eq!(heaviside_enrichment([0.5, 0.5, 0.0], &crack), 1.0);
    // The faces of the second segment.
    assert_eq!(heaviside_enrichment([0.9, 0.5, 0.0], &crack), 1.0);
    assert_eq!(heaviside_enrichment([1.5, 0.5, 0.0], &crack), -1.0);
    // The outer fan of the kink vertex and the extensions beyond both ends.
    assert_eq!(heaviside_enrichment([1.5, -0.2, 0.0], &crack), -1.0);
    assert_eq!(heaviside_enrichment([1.5, 1.5, 0.0], &crack), -1.0);
    assert_eq!(heaviside_enrichment([0.5, 1.5, 0.0], &crack), 1.0);
    assert_eq!(heaviside_enrichment([-0.5, 0.5, 0.0], &crack), 1.0);
    assert_eq!(heaviside_enrichment([-0.5, -0.5, 0.0], &crack), -1.0);
    // A point on the crack counts as the left side.
    assert_eq!(heaviside_enrichment([0.5, 0.0, 0.0], &crack), 1.0);
}

#[test]
fn signed_distance_to_a_kinked_crack_has_the_expected_values() {
    let left = left_kink_crack();
    let cases = [
        ([0.5, -0.5, 0.0], -0.5),
        ([0.5, 0.5, 0.0], 0.5),
        ([0.9, 0.5, 0.0], 0.1),
        ([1.5, 0.5, 0.0], -0.5),
        ([1.5, -0.2, 0.0], -(0.5_f64 * 0.5 + 0.2 * 0.2).sqrt()),
        ([1.5, 1.5, 0.0], -0.5),
        ([0.5, 1.5, 0.0], 0.5),
        ([0.5, 0.5, 7.0], 0.5),
    ];
    for (point, expected) in cases {
        let phi = signed_distance_to_crack(point, &left);
        assert!(
            (phi - expected).abs() < 1e-12,
            "phi{point:?} = {phi} vs {expected}"
        );
    }
    // Right turn: the outer fan of the vertex is on the left (positive) side.
    let right = right_kink_crack();
    let phi = signed_distance_to_crack([1.5, 0.2, 0.0], &right);
    assert!(
        (phi - (0.5_f64 * 0.5 + 0.2 * 0.2).sqrt()).abs() < 1e-12,
        "phi = {phi}"
    );
    let phi = signed_distance_to_crack([0.5, -0.5, 0.0], &right);
    assert!((phi + 0.5).abs() < 1e-12, "phi = {phi}");
    // The left of the downward second segment is the +x side.
    assert_eq!(heaviside_enrichment([1.5, 0.2, 0.0], &right), 1.0);
    assert_eq!(heaviside_enrichment([1.5, -0.5, 0.0], &right), 1.0);
    assert_eq!(heaviside_enrichment([0.5, -0.5, 0.0], &right), -1.0);
}

#[test]
fn signed_distance_to_a_single_segment_matches_the_line_distance() {
    let crack = Crack::new([0.0, 0.0, 0.0], [2.0, 1.0, 0.0]);
    let mut rng = Lcg(7);
    for _ in 0..200 {
        let point = [
            -1.0 + 5.0 * rng.next_f64(),
            -2.0 + 4.0 * rng.next_f64(),
            0.0,
        ];
        let expected = signed_distance_to_line(point, [0.0, 0.0, 0.0], [2.0, 1.0, 0.0]);
        let got = signed_distance_to_crack(point, &crack);
        assert!(
            (got - expected).abs() < 1e-12,
            "point {point:?}: {got} vs {expected}"
        );
    }
}

#[test]
fn signed_distance_is_one_lipschitz_around_every_vertex() {
    // Sampling a circle around a vertex: a wrong side at the kink would show up
    // as a jump of 2 |phi| between neighbouring samples.
    for (name, crack, vertices) in [
        ("left", left_kink_crack(), vec![[1.0, 0.0]]),
        ("right", right_kink_crack(), vec![[1.0, 0.0]]),
        ("zigzag", zigzag_crack(), vec![[1.0, 0.0], [1.0, 1.0]]),
    ] {
        for [vx, vy] in vertices {
            for radius in [0.05, 0.3] {
                let samples = 3600;
                let step = 2.0 * PI * radius / samples as f64;
                let phi: Vec<f64> = (0..samples)
                    .map(|index| {
                        let angle = 2.0 * PI * index as f64 / samples as f64;
                        signed_distance_to_crack(
                            [vx + radius * angle.cos(), vy + radius * angle.sin(), 0.0],
                            &crack,
                        )
                    })
                    .collect();
                for index in 0..samples {
                    let (a, b) = (phi[index], phi[(index + 1) % samples]);
                    assert!(
                        (b - a).abs() <= step * (1.0 + 1e-9),
                        "{name} kink at ({vx},{vy}), r = {radius}: jump {a} -> {b} at sample {index}"
                    );
                }
                // Samples that sit on the crack up to rounding carry no sign.
                let signs: Vec<bool> = phi
                    .iter()
                    .filter(|value| value.abs() > 1e-9)
                    .map(|&value| value > 0.0)
                    .collect();
                let sign_changes = (0..signs.len())
                    .filter(|&index| signs[index] != signs[(index + 1) % signs.len()])
                    .count();
                assert_eq!(
                    sign_changes, 2,
                    "{name} kink at ({vx},{vy}), r = {radius}: the circle crosses the crack twice"
                );
            }
        }
    }
}

#[test]
fn signed_distance_ignores_zero_length_segments() {
    // propagate(0, ..) inserts a zero-length segment in the middle of the polyline.
    let mut crack = Crack::new([0.0, 0.0, 0.0], [1.0, 0.0, 0.0]);
    crack.propagate(0.0, [0.0, 1.0, 0.0]);
    crack.propagate(1.0, [0.0, 1.0, 0.0]);
    assert_eq!(crack.segments.len(), 3);
    let reference = left_kink_crack();
    for point in [
        [0.5, -0.5, 0.0],
        [0.5, 0.5, 0.0],
        [1.5, -0.2, 0.0],
        [1.5, 1.5, 0.0],
        [0.9, 0.5, 0.0],
    ] {
        let a = signed_distance_to_crack(point, &crack);
        let b = signed_distance_to_crack(point, &reference);
        assert!((a - b).abs() < 1e-12, "{point:?}: {a} vs {b}");
    }
}

#[test]
fn heaviside_and_level_set_survive_a_crack_without_segments() {
    let crack = Crack {
        segments: Vec::new(),
        tip: [1.0, 0.0, 0.0],
        growth_direction: [1.0, 0.0, 0.0],
    };
    assert_eq!(heaviside_enrichment([0.5, 0.5, 0.0], &crack), 1.0);
    assert_eq!(heaviside_enrichment([0.5, -0.5, 0.0], &crack), -1.0);
    let nodes = vec![[0.5, 1.0, 0.0], [0.5, -1.0, 0.0], [2.0, 0.0, 0.0]];
    let ls = LevelSet::from_crack(&nodes, &crack);
    assert!((ls.phi[0] - 1.0).abs() < 1e-12 && (ls.phi[1] + 1.0).abs() < 1e-12);
    assert!(ls.phi[2].abs() < 1e-12);
    assert!((ls.psi[2] - 1.0).abs() < 1e-12);
    // A crack with a zero growth direction has no side: phi = 0, H = +1.
    let degenerate = Crack::new([0.0, 0.0, 0.0], [0.0, 0.0, 0.0]);
    assert_eq!(signed_distance_to_crack([1.0, 1.0, 0.0], &degenerate), 0.0);
    assert_eq!(heaviside_enrichment([1.0, -1.0, 0.0], &degenerate), 1.0);
}

#[test]
fn level_set_of_a_kinked_crack_uses_the_whole_polyline() {
    let crack = left_kink_crack();
    let nodes = vec![
        [0.5, -0.5, 0.0], // below the first segment
        [0.5, 0.5, 0.0],  // above the first segment, left of the second
        [1.5, 0.5, 0.0],  // right of the second segment
    ];
    let ls = LevelSet::from_crack(&nodes, &crack);
    assert!((ls.phi[0] + 0.5).abs() < 1e-12, "phi = {:?}", ls.phi);
    assert!((ls.phi[1] - 0.5).abs() < 1e-12, "phi = {:?}", ls.phi);
    assert!((ls.phi[2] + 0.5).abs() < 1e-12, "phi = {:?}", ls.phi);
    // psi measures the distance from the tip (1,1) along the growth direction (0,1).
    for (node, psi) in nodes.iter().zip(ls.psi.iter()) {
        assert!((psi - (node[1] - 1.0)).abs() < 1e-12);
    }
}

// ── F1418: sub-triangulation of cut triangles ───────────────────────────────

const UNIT_TRIANGLE: [[f64; 2]; 3] = [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]];

fn side_area(tris: &[SubTriangle], side: i8) -> f64 {
    tris.iter()
        .filter(|tri| tri.side == side)
        .map(|tri| tri.area())
        .sum()
}

#[test]
fn sub_triangulate_cut_between_vertices_gives_ordered_polygons() {
    // phi = [1, -1, 3]: crossings at (0.5, 0) and (0.75, 0.25).
    let tris = sub_triangulate_triangle(&UNIT_TRIANGLE, &[1.0, -1.0, 3.0]);
    assert_eq!(
        tris.len(),
        3,
        "one triangle on the lone side, two on the other"
    );
    assert!((side_area(&tris, 1) - 0.4375).abs() < 1e-12);
    assert!((side_area(&tris, -1) - 0.0625).abs() < 1e-12);
    assert!((side_area(&tris, 1) + side_area(&tris, -1) - 0.5).abs() < 1e-12);
    assert_eq!(tris.iter().filter(|tri| tri.side == -1).count(), 1);
    let negative = tris
        .iter()
        .find(|tri| tri.side == -1)
        .expect("negative piece");
    let has_vertex = |target: [f64; 2]| {
        negative
            .vertices
            .iter()
            .any(|v| (v[0] - target[0]).abs() < 1e-12 && (v[1] - target[1]).abs() < 1e-12)
    };
    assert!(has_vertex([1.0, 0.0]) && has_vertex([0.5, 0.0]) && has_vertex([0.75, 0.25]));
}

#[test]
fn sub_triangulate_interface_through_a_vertex_gives_two_pieces() {
    // phi = [-1, 1, 0]: the interface runs from vertex 2 to the midpoint of edge 0-1.
    let tris = sub_triangulate_triangle(&UNIT_TRIANGLE, &[-1.0, 1.0, 0.0]);
    assert_eq!(tris.len(), 2);
    assert!((side_area(&tris, 1) - 0.25).abs() < 1e-12);
    assert!((side_area(&tris, -1) - 0.25).abs() < 1e-12);
    // Same through vertex 0 and 1.
    for phi in [[0.0, 1.0, -1.0], [1.0, 0.0, -1.0], [-1.0, 0.0, 1.0]] {
        let tris = sub_triangulate_triangle(&UNIT_TRIANGLE, &phi);
        assert_eq!(tris.len(), 2, "phi = {phi:?}");
        assert!(
            side_area(&tris, 1) > 0.0 && side_area(&tris, -1) > 0.0,
            "phi = {phi:?}"
        );
        assert!((side_area(&tris, 1) + side_area(&tris, -1) - 0.5).abs() < 1e-12);
    }
}

#[test]
fn sub_triangulate_touching_interface_keeps_a_single_side() {
    for (phi, side) in [
        ([0.0, -1.0, -1.0], -1i8),
        ([0.0, 1.0, 1.0], 1),
        ([-1.0, 0.0, -1.0], -1),
        ([0.0, 0.0, 1.0], 1),
        ([0.0, 0.0, -1.0], -1),
        ([1.0, 2.0, 3.0], 1),
        ([-1.0, -2.0, -3.0], -1),
    ] {
        let tris = sub_triangulate_triangle(&UNIT_TRIANGLE, &phi);
        assert_eq!(tris.len(), 1, "phi = {phi:?}");
        assert_eq!(tris[0].side, side, "phi = {phi:?}");
        assert!((tris[0].area() - 0.5).abs() < 1e-12);
    }
    let tris = sub_triangulate_triangle(&UNIT_TRIANGLE, &[0.0, 0.0, 0.0]);
    assert_eq!(tris.len(), 1);
    assert_eq!(tris[0].side, 1);
}

#[test]
fn sub_triangulate_conserves_area_and_respects_the_interface() {
    let mut rng = Lcg(2024);
    for _ in 0..500 {
        let vertices = [
            [rng.next_f64(), rng.next_f64()],
            [2.0 + rng.next_f64(), rng.next_f64()],
            [rng.next_f64(), 2.0 + rng.next_f64()],
        ];
        // Occasionally mirror the triangle to get clockwise orientations too.
        let vertices = if rng.next_f64() < 0.5 {
            vertices
        } else {
            [vertices[0], vertices[2], vertices[1]]
        };
        let mut phi = [
            rng.next_f64() - 0.5,
            rng.next_f64() - 0.5,
            rng.next_f64() - 0.5,
        ];
        if rng.next_f64() < 0.3 {
            phi[(rng.next_f64() * 3.0) as usize % 3] = 0.0;
        }
        let parent = SubTriangle::new(vertices, 1);
        let tris = sub_triangulate_triangle(&vertices, &phi);
        assert!(!tris.is_empty() && tris.len() <= 3, "phi = {phi:?}");
        let total: f64 = tris.iter().map(|tri| tri.area()).sum();
        assert!(
            (total - parent.area()).abs() < 1e-12 * parent.area(),
            "phi = {phi:?}: {total} vs {}",
            parent.area()
        );
        // The interpolated level set at every sub-triangle centroid has the
        // sub-triangle's side.
        let [v0, v1, v2] = vertices;
        let det = (v1[0] - v0[0]) * (v2[1] - v0[1]) - (v2[0] - v0[0]) * (v1[1] - v0[1]);
        for tri in &tris {
            let c = tri.centroid();
            let l1 = ((c[0] - v0[0]) * (v2[1] - v0[1]) - (v2[0] - v0[0]) * (c[1] - v0[1])) / det;
            let l2 = ((v1[0] - v0[0]) * (c[1] - v0[1]) - (c[0] - v0[0]) * (v1[1] - v0[1])) / det;
            let phi_c = (1.0 - l1 - l2) * phi[0] + l1 * phi[1] + l2 * phi[2];
            assert!(
                phi_c * f64::from(tri.side) >= -1e-12,
                "phi = {phi:?}: centroid phi {phi_c} on side {}",
                tri.side
            );
        }
    }
}

// ── F1427: K_I from the crack-face displacement ─────────────────────────────

#[test]
fn ki_from_displacement_agrees_with_the_crack_opening_extractor() {
    let (e, nu) = (70e9, 0.33);
    let sif = SifCalculator::new(e, nu);
    let extractor = XfemSifExtractor::new(e, nu);
    for (u_y, r) in [(1.0e-6, 1.0e-3), (3.5e-7, 2.0e-4), (2.0e-5, 0.05)] {
        let from_face = sif.ki_from_displacement(u_y, r, true);
        let from_cod = extractor.ki_from_cod(2.0 * u_y, r);
        assert!(
            (from_face / from_cod - 1.0).abs() < 1e-12,
            "{from_face:e} vs {from_cod:e}"
        );
    }
}

#[test]
fn ki_from_displacement_plane_stress_is_e_over_four() {
    let (u_y, r) = (1.0e-6, 1.0e-3);
    for nu in [0.0, 0.25, 0.45] {
        let sif = SifCalculator::new(100e9, nu);
        let ki = sif.ki_from_displacement(u_y, r, false);
        let expected = 100e9 / 4.0 * u_y * (2.0 * PI / r).sqrt();
        assert!(
            (ki / expected - 1.0).abs() < 1e-12,
            "nu = {nu}: {ki:e} vs {expected:e}"
        );
    }
}

#[test]
fn ki_from_displacement_inverts_the_williams_crack_face_field() {
    let (e, nu, k_i, r) = (200e9, 0.3, 30.0e6, 5.0e-4);
    let mu = e / (2.0 * (1.0 + nu));
    for plane_strain in [true, false] {
        let kappa = if plane_strain {
            3.0 - 4.0 * nu
        } else {
            (3.0 - nu) / (1.0 + nu)
        };
        // u_y(r, theta = pi) = (kappa + 1) K_I / (2 mu) * sqrt(r / (2 pi))
        let u_y = (kappa + 1.0) * k_i / (2.0 * mu) * (r / (2.0 * PI)).sqrt();
        let recovered = SifCalculator::new(e, nu).ki_from_displacement(u_y, r, plane_strain);
        assert!(
            (recovered / k_i - 1.0).abs() < 1e-12,
            "plane_strain = {plane_strain}: {recovered:e} vs {k_i:e}"
        );
    }
}

#[test]
fn ki_from_displacement_rejects_non_positive_distance() {
    let sif = SifCalculator::new(200e9, 0.3);
    assert!(sif.ki_from_displacement(1.0e-6, 0.0, true).is_nan());
    assert!(sif.ki_from_displacement(1.0e-6, -1.0e-3, false).is_nan());
    assert!(sif.ki_from_displacement(1.0e-6, f64::NAN, true).is_nan());
}

// ── F1429: Heaviside node flags ─────────────────────────────────────────────

fn flagged(flags: &[bool]) -> Vec<usize> {
    flags
        .iter()
        .enumerate()
        .filter_map(|(node, &flag)| flag.then_some(node))
        .collect()
}

#[test]
fn heaviside_flags_select_only_the_cut_element_row_behind_the_tip() {
    let n = 10;
    let nodes = grid_nodes(n);
    let elements = grid_elements(n);
    // Crack along y = 0.45 (inside the element row 4) from the left edge to x = 0.55.
    let crack = Crack::new([0.0, 0.45, 0.0], [0.55, 0.45, 0.0]);
    let ls = LevelSet::from_crack(&nodes, &crack);
    let tips = tip_flags(&ls, 0.08);
    // The four corners of the tip element (column 5, row 4) are tip enriched.
    let expected_tips: Vec<usize> = vec![
        node_id(n, 5, 4),
        node_id(n, 6, 4),
        node_id(n, 5, 5),
        node_id(n, 6, 5),
    ];
    assert_eq!(flagged(&tips), {
        let mut sorted = expected_tips.clone();
        sorted.sort_unstable();
        sorted
    });
    let flags = ls.heaviside_node_flags(&elements, &tips);
    assert_eq!(flags.len(), nodes.len());
    // Columns 0..=4 of row 4 are cut and fully behind the tip: nodes x = 0 .. 0.4
    // of the node rows 4 and 5 (the nodes at x = 0.5 belong to the tip element).
    let mut expected: Vec<usize> = Vec::new();
    for j in [4, 5] {
        for i in 0..=4 {
            expected.push(node_id(n, i, j));
        }
    }
    expected.sort_unstable();
    assert_eq!(flagged(&flags), expected);
    // Without tip flags the nodes at x = 0.5 are flagged as well.
    let without_tips = ls.heaviside_node_flags(&elements, &[]);
    assert_eq!(flagged(&without_tips).len(), 12);
    assert!(without_tips[node_id(n, 5, 4)] && without_tips[node_id(n, 5, 5)]);
}

#[test]
fn heaviside_flags_do_not_enrich_nodes_far_from_the_crack() {
    let n = 10;
    let nodes = grid_nodes(n);
    let elements = grid_elements(n);
    let crack = Crack::new([0.0, 0.45, 0.0], [0.55, 0.45, 0.0]);
    let ls = LevelSet::from_crack(&nodes, &crack);
    // Every node behind the tip has psi < 0, but most have nothing to do with the crack.
    let behind_tip = (0..nodes.len()).filter(|&node| ls.psi[node] < 0.0).count();
    assert!(behind_tip > 50, "{behind_tip} nodes lie behind the tip");
    let flags = ls.heaviside_node_flags(&elements, &[]);
    for (node, &flag) in flags.iter().enumerate() {
        if flag {
            assert!(
                ls.phi[node].abs() < 0.0999,
                "node {node} with phi = {} must not be enriched",
                ls.phi[node]
            );
            assert!(ls.psi[node] <= 0.0, "node {node} lies ahead of the tip");
        }
    }
    assert!(!flags[node_id(n, 2, 9)] && !flags[node_id(n, 8, 0)] && !flags[node_id(n, 0, 0)]);
}

#[test]
fn heaviside_flags_for_a_crack_through_nodes_flag_only_the_nodes_on_it() {
    let n = 10;
    let nodes = grid_nodes(n);
    let elements = grid_elements(n);
    // The crack runs exactly along the node row y = 0.5, phi = 0 on that row.
    let crack = Crack::new([0.0, 0.5, 0.0], [0.55, 0.5, 0.0]);
    let ls = LevelSet::from_crack(&nodes, &crack);
    assert!(ls.phi[node_id(n, 2, 5)].abs() < 1e-15);
    let tips = tip_flags(&ls, 0.08);
    let flags = ls.heaviside_node_flags(&elements, &tips);
    // H = +1 on the crack, so only the on-crack nodes see the negative side of the
    // elements below; every other shifted enrichment vanishes identically.
    let expected: Vec<usize> = (0..=4).map(|i| node_id(n, i, 5)).collect();
    assert_eq!(flagged(&flags), expected);
}

#[test]
fn heaviside_flags_follow_a_kinked_crack() {
    let n = 10;
    let nodes = grid_nodes(n);
    let elements = grid_elements(n);
    // Crack along y = 0.45 up to (0.45, 0.45), then straight up to the tip (0.45, 0.85).
    let mut crack = Crack::new([0.0, 0.45, 0.0], [0.45, 0.45, 0.0]);
    crack.propagate(0.4, [0.0, 1.0, 0.0]);
    let ls = LevelSet::from_crack(&nodes, &crack);
    let tips = tip_flags(&ls, 0.08);
    assert_eq!(flagged(&tips), {
        let mut tip_nodes = vec![
            node_id(n, 4, 8),
            node_id(n, 5, 8),
            node_id(n, 4, 9),
            node_id(n, 5, 9),
        ];
        tip_nodes.sort_unstable();
        tip_nodes
    });
    let flags = ls.heaviside_node_flags(&elements, &tips);
    let mut expected: Vec<usize> = Vec::new();
    // Horizontal part: element row 4, columns 0..=4.
    for j in [4, 5] {
        for i in 0..=5 {
            expected.push(node_id(n, i, j));
        }
    }
    // Vertical part: element column 4, rows 5..=7 (the nodes at y = 0.8 are tip nodes).
    for j in [6, 7] {
        for i in [4, 5] {
            expected.push(node_id(n, i, j));
        }
    }
    expected.sort_unstable();
    assert_eq!(flagged(&flags), expected);
}

#[test]
fn heaviside_flags_tolerate_malformed_input() {
    let n = 4;
    let nodes = grid_nodes(n);
    let crack = Crack::new([0.0, 0.3, 0.0], [0.7, 0.3, 0.0]);
    let ls = LevelSet::from_crack(&nodes, &crack);
    let mut elements = grid_elements(n);
    let reference = ls.heaviside_node_flags(&elements, &[]);
    assert!(reference.iter().any(|&flag| flag));
    // Out-of-range node indices and empty elements are skipped, short tip flags are fine.
    elements.push(vec![0, 1, 2, nodes.len()]);
    elements.push(Vec::new());
    elements.push(vec![usize::MAX]);
    let flags = ls.heaviside_node_flags(&elements, &[false, false]);
    assert_eq!(flags, reference);
    // Surplus tip flags are ignored.
    let long_tips = vec![false; nodes.len() + 10];
    assert_eq!(ls.heaviside_node_flags(&elements, &long_tips), reference);
    // No elements: nothing flagged.
    assert!(ls.heaviside_node_flags(&[], &[]).iter().all(|&flag| !flag));
}
