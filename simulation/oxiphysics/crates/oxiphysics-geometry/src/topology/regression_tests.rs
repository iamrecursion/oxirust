//! Regression tests for the half-edge mesh, the polygon offset and the
//! 0-dimensional persistence routines of the topology module.

use std::collections::BTreeSet;

use super::fixtures::{
    Fixture, grid, grid_torus, icosahedron, octahedron, seven_vertex_torus, tetrahedron, tube,
};
use super::functions::{
    classify_morse_critical_points, compute_0d_persistence, count_connected_components,
    trace_descending_manifold,
};
use super::types::{HalfEdgeMesh, MorseCriticalType, PersistenceDiagram, PolygonOffset};

/// Deterministic pseudo-random numbers.
struct Lcg(u64);

impl Lcg {
    fn next_f64(&mut self) -> f64 {
        self.0 = self
            .0
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        (self.0 >> 11) as f64 / (1u64 << 53) as f64
    }

    fn below(&mut self, n: usize) -> usize {
        (self.next_f64() * n as f64) as usize % n
    }
}

fn build(fixture: &Fixture) -> HalfEdgeMesh {
    HalfEdgeMesh::from_triangles(fixture.0.clone(), &fixture.1)
}

/// Unit cube with 8 vertices and 12 outward-oriented triangles.
fn cube() -> Fixture {
    let vertices = vec![
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [1.0, 1.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
        [1.0, 0.0, 1.0],
        [1.0, 1.0, 1.0],
        [0.0, 1.0, 1.0],
    ];
    let triangles = vec![
        [0, 3, 2],
        [0, 2, 1],
        [4, 5, 6],
        [4, 6, 7],
        [0, 1, 5],
        [0, 5, 4],
        [3, 7, 6],
        [3, 6, 2],
        [0, 4, 7],
        [0, 7, 3],
        [1, 2, 6],
        [1, 6, 5],
    ];
    (vertices, triangles)
}

// ---------------------------------------------------------------------------
// HalfEdgeMesh::num_edges
// ---------------------------------------------------------------------------

#[test]
fn num_edges_counts_every_edge_once() {
    let tet = build(&tetrahedron());
    assert_eq!(tet.num_edges(), 6);
    let cube = build(&cube());
    assert_eq!(cube.num_edges(), 18);
    let single = HalfEdgeMesh::from_triangles(
        vec![[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
        &[[0, 1, 2]],
    );
    assert_eq!(single.num_edges(), 3);
    assert_eq!(HalfEdgeMesh::new().num_edges(), 0);
}

#[test]
fn num_edges_agrees_with_the_unique_edge_count_and_euler_formula() {
    for (name, fixture) in [
        ("tetrahedron", tetrahedron()),
        ("cube", cube()),
        ("octahedron", octahedron()),
        ("icosahedron", icosahedron()),
        ("seven-vertex torus", seven_vertex_torus()),
        ("grid torus", grid_torus(5, 4)),
        ("tube", tube(4, 6)),
        ("grid", grid(4, 5, false)),
        ("alternating grid", grid(5, 5, true)),
    ] {
        let mesh = build(&fixture);
        let edges = mesh.num_edges();
        assert_eq!(edges, mesh.count_unique_edges(), "{name}");
        let chi = mesh.num_vertices() as i64 - edges as i64 + mesh.num_faces() as i64;
        assert_eq!(chi, mesh.euler_characteristic(), "{name}");
        if mesh.is_closed() {
            // Closed triangulated surface: every face has three edges, each shared.
            assert_eq!(2 * edges, 3 * mesh.num_faces(), "{name}");
        }
    }
}

#[test]
fn num_edges_of_the_expected_closed_surfaces() {
    assert_eq!(build(&octahedron()).num_edges(), 12);
    assert_eq!(build(&icosahedron()).num_edges(), 30);
    assert_eq!(build(&seven_vertex_torus()).num_edges(), 21);
    assert_eq!(build(&grid_torus(4, 4)).num_edges(), 48);
}

// ---------------------------------------------------------------------------
// HalfEdgeMesh::vertex_neighbors
// ---------------------------------------------------------------------------

fn sorted(mut list: Vec<usize>) -> Vec<usize> {
    list.sort_unstable();
    list
}

fn quad_mesh() -> HalfEdgeMesh {
    HalfEdgeMesh::from_triangles(
        vec![
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [1.0, 1.0, 0.0],
            [0.0, 1.0, 0.0],
        ],
        &[[0, 1, 2], [0, 2, 3]],
    )
}

#[test]
fn boundary_vertices_keep_all_of_their_neighbours() {
    let mesh = quad_mesh();
    assert_eq!(sorted(mesh.vertex_neighbors(0)), vec![1, 2, 3]);
    assert_eq!(sorted(mesh.vertex_neighbors(1)), vec![0, 2]);
    assert_eq!(sorted(mesh.vertex_neighbors(2)), vec![0, 1, 3]);
    assert_eq!(sorted(mesh.vertex_neighbors(3)), vec![0, 2]);
}

#[test]
fn boundary_fans_are_returned_as_a_chain_in_walk_order() {
    let mesh = quad_mesh();
    // Clockwise around vertex 0 of the unit square: 3 (90 deg), 2 (45), 1 (0).
    assert_eq!(mesh.vertex_neighbors(0), vec![3, 2, 1]);
    assert_eq!(mesh.vertex_neighbors(2), vec![1, 0, 3]);
    assert_eq!(mesh.vertex_neighbors(1), vec![0, 2]);
    assert_eq!(mesh.vertex_neighbors(3), vec![2, 0]);
    // The walk order is the same for every start: it does not depend on which
    // outgoing half-edge `vertex_he` happens to store.
    let mut other_start = quad_mesh();
    other_start.vertex_he[0] = 3;
    assert_eq!(other_start.vertex_neighbors(0), vec![3, 2, 1]);
}

#[test]
fn vertex_neighbors_of_unused_or_missing_vertices_is_empty() {
    let mut mesh = quad_mesh();
    mesh.vertices.push([5.0, 5.0, 0.0]);
    mesh.vertex_he.push(usize::MAX);
    assert!(mesh.vertex_neighbors(4).is_empty());
    assert!(mesh.vertex_neighbors(99).is_empty());
    assert!(HalfEdgeMesh::new().vertex_neighbors(0).is_empty());
}

/// Vertices that share a triangle with `v`.
fn adjacent_vertices(triangles: &[[usize; 3]], v: usize) -> BTreeSet<usize> {
    triangles
        .iter()
        .filter(|t| t.contains(&v))
        .flat_map(|t| t.iter().copied())
        .filter(|&w| w != v)
        .collect()
}

/// Whether `v` is an endpoint of an edge with a single incident triangle.
fn is_boundary_vertex(triangles: &[[usize; 3]], v: usize) -> bool {
    let directed: BTreeSet<(usize, usize)> = triangles
        .iter()
        .flat_map(|t| (0..3).map(move |k| (t[k], t[(k + 1) % 3])))
        .collect();
    directed
        .iter()
        .any(|&(a, b)| (a == v || b == v) && !directed.contains(&(b, a)))
}

/// Check that `vertex_neighbors(v)` is the complete, duplicate-free fan of `v`
/// in which consecutive entries share a triangle with `v`.
fn assert_fan(mesh: &HalfEdgeMesh, triangles: &[[usize; 3]], v: usize, name: &str) {
    let fan = mesh.vertex_neighbors(v);
    let as_set: BTreeSet<usize> = fan.iter().copied().collect();
    assert_eq!(as_set.len(), fan.len(), "{name}: duplicates around {v}");
    assert_eq!(
        as_set,
        adjacent_vertices(triangles, v),
        "{name}: neighbours of {v}"
    );
    let share_triangle = |a: usize, b: usize| {
        triangles
            .iter()
            .any(|t| t.contains(&v) && t.contains(&a) && t.contains(&b))
    };
    for pair in fan.windows(2) {
        assert!(share_triangle(pair[0], pair[1]), "{name}: fan of {v}");
    }
    if is_boundary_vertex(triangles, v) {
        // An open chain: both ends are joined to `v` by a boundary edge.
        let directed: BTreeSet<(usize, usize)> = triangles
            .iter()
            .flat_map(|t| (0..3).map(move |k| (t[k], t[(k + 1) % 3])))
            .collect();
        for end in [fan[0], fan[fan.len() - 1]] {
            let forward = directed.contains(&(v, end));
            let backward = directed.contains(&(end, v));
            assert!(forward != backward, "{name}: chain end {end} of {v}");
        }
    } else {
        assert!(
            share_triangle(fan[fan.len() - 1], fan[0]),
            "{name}: ring of {v} is not closed"
        );
    }
}

#[test]
fn every_fan_is_complete_on_closed_and_open_meshes() {
    for (name, fixture) in [
        ("tetrahedron", tetrahedron()),
        ("cube", cube()),
        ("octahedron", octahedron()),
        ("icosahedron", icosahedron()),
        ("seven-vertex torus", seven_vertex_torus()),
        ("grid torus", grid_torus(4, 5)),
        ("tube", tube(4, 6)),
        ("grid", grid(4, 5, false)),
        ("alternating grid", grid(5, 4, true)),
    ] {
        let mesh = build(&fixture);
        for v in 0..mesh.num_vertices() {
            assert_fan(&mesh, &fixture.1, v, name);
        }
    }
}

#[test]
fn closed_fans_have_one_neighbour_per_incident_edge() {
    let tet = build(&tetrahedron());
    for v in 0..4 {
        assert_eq!(sorted(tet.vertex_neighbors(v)).len(), 3);
    }
    let icosa = build(&icosahedron());
    for v in 0..12 {
        assert_eq!(icosa.vertex_neighbors(v).len(), 5);
    }
    let torus = build(&seven_vertex_torus());
    for v in 0..7 {
        assert_eq!(torus.vertex_neighbors(v).len(), 6);
    }
}

#[test]
fn morse_classification_sees_the_whole_fan_of_boundary_vertices() {
    let mesh = quad_mesh();
    // Vertex 0 has value 1 with neighbours 1 (below), 2 and 3 (above): regular.
    // A truncated fan [1] made it look like a maximum.
    let values = [1.0, 0.0, 3.0, 2.0];
    let classes = classify_morse_critical_points(&mesh, &values);
    assert_eq!(classes[0].critical_type, MorseCriticalType::Regular);
    assert_eq!(classes[1].critical_type, MorseCriticalType::Minimum);
    assert_eq!(classes[2].critical_type, MorseCriticalType::Maximum);
    assert_eq!(classes[3].critical_type, MorseCriticalType::Regular);
}

#[test]
fn morse_classification_of_a_bowl_on_an_open_grid() {
    let fixture = grid(5, 5, false);
    let mesh = build(&fixture);
    let value = |v: usize| {
        let (r, c) = ((v / 5) as f64, (v % 5) as f64);
        (r - 2.0).powi(2) + (c - 2.0).powi(2)
    };
    let values: Vec<f64> = (0..25).map(value).collect();
    let classes = classify_morse_critical_points(&mesh, &values);
    assert_eq!(classes[12].critical_type, MorseCriticalType::Minimum);
    let maxima: Vec<usize> = classes
        .iter()
        .filter(|m| m.critical_type == MorseCriticalType::Maximum)
        .map(|m| m.idx)
        .collect();
    assert_eq!(maxima, vec![0, 4, 20, 24], "the four corners");
    // The midpoints of the sides sit on the slope of the bowl.
    for v in [2, 10, 14, 22] {
        assert_eq!(classes[v].critical_type, MorseCriticalType::Regular, "{v}");
    }
}

#[test]
fn descending_path_uses_all_neighbours_of_a_boundary_vertex() {
    let mesh = quad_mesh();
    let values = [1.0, 0.0, 3.0, 2.0];
    // Steepest descent from vertex 2 (value 3): the lowest neighbour is vertex 1.
    assert_eq!(trace_descending_manifold(&mesh, &values, 2, 10), vec![2, 1]);
    assert_eq!(
        trace_descending_manifold(&mesh, &values, 3, 10),
        vec![3, 0, 1]
    );
}

// ---------------------------------------------------------------------------
// PolygonOffset
// ---------------------------------------------------------------------------

fn unit_square() -> Vec<[f64; 2]> {
    vec![[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]]
}

fn close(a: [f64; 2], b: [f64; 2]) -> bool {
    (a[0] - b[0]).abs() < 1e-12 && (a[1] - b[1]).abs() < 1e-12
}

#[test]
fn square_offset_outward_is_the_larger_square() {
    let out = PolygonOffset::offset_polygon_2d(&unit_square(), 0.1);
    let expected = [[-0.1, -0.1], [1.1, -0.1], [1.1, 1.1], [-0.1, 1.1]];
    assert_eq!(out.len(), 4);
    for (got, want) in out.iter().zip(expected) {
        assert!(close(*got, want), "{got:?} != {want:?}");
    }
    assert!((PolygonOffset::signed_area(&out) - 1.44).abs() < 1e-12);
}

#[test]
fn square_offset_inward_is_the_smaller_square() {
    let out = PolygonOffset::offset_polygon_2d(&unit_square(), -0.1);
    let expected = [[0.1, 0.1], [0.9, 0.1], [0.9, 0.9], [0.1, 0.9]];
    for (got, want) in out.iter().zip(expected) {
        assert!(close(*got, want), "{got:?} != {want:?}");
    }
    assert!((PolygonOffset::signed_area(&out) - 0.64).abs() < 1e-12);
}

#[test]
fn clockwise_polygons_are_offset_outward_for_positive_distance() {
    let clockwise: Vec<[f64; 2]> = unit_square().into_iter().rev().collect();
    assert!(!PolygonOffset::is_ccw(&clockwise));
    let out = PolygonOffset::offset_polygon_2d(&clockwise, 0.1);
    assert_eq!(out.len(), 4);
    for p in &out {
        assert!(
            ((p[0] + 0.1).abs() < 1e-12 || (p[0] - 1.1).abs() < 1e-12)
                && ((p[1] + 0.1).abs() < 1e-12 || (p[1] - 1.1).abs() < 1e-12),
            "{p:?}"
        );
    }
    assert!((PolygonOffset::signed_area(&out) + 1.44).abs() < 1e-12);
    let inward = PolygonOffset::offset_polygon_2d(&clockwise, -0.25);
    assert!((PolygonOffset::signed_area(&inward) + 0.25).abs() < 1e-12);
}

#[test]
fn collinear_vertices_move_by_exactly_the_distance() {
    let rectangle = [[0.0, 0.0], [1.0, 0.0], [2.0, 0.0], [2.0, 1.0], [0.0, 1.0]];
    let out = PolygonOffset::offset_polygon_2d(&rectangle, 0.25);
    assert_eq!(out.len(), 5);
    assert!(close(out[1], [1.0, -0.25]), "{:?}", out[1]);
    assert!(close(out[0], [-0.25, -0.25]));
    assert!(close(out[2], [2.25, -0.25]));
}

/// Signed distance of `p` from the line through `a` and `b`, positive on the
/// right-hand side of `a -> b` (the outside of a counter-clockwise polygon).
fn right_distance(a: [f64; 2], b: [f64; 2], p: [f64; 2]) -> f64 {
    let (dx, dy) = (b[0] - a[0], b[1] - a[1]);
    ((p[0] - a[0]) * dy - (p[1] - a[1]) * dx) / dx.hypot(dy)
}

#[test]
fn every_offset_edge_is_at_the_requested_distance() {
    let hexagon: Vec<[f64; 2]> = (0..6)
        .map(|k| {
            let angle = std::f64::consts::TAU * k as f64 / 6.0;
            [2.0 * angle.cos(), 2.0 * angle.sin()]
        })
        .collect();
    let pentagon = vec![[0.0, 0.0], [3.0, 0.5], [3.5, 2.0], [1.5, 3.0], [-0.5, 1.5]];
    for (name, polygon) in [("hexagon", hexagon), ("pentagon", pentagon)] {
        for dist in [0.15, -0.15] {
            let out = PolygonOffset::offset_polygon_2d(&polygon, dist);
            assert_eq!(out.len(), polygon.len(), "{name}");
            let n = polygon.len();
            for i in 0..n {
                // Offset vertex i is the meeting point of the displaced edges
                // i - 1 -> i and i -> i + 1: it is `dist` away from both lines
                // (a counter-clockwise polygon has its outside on the right).
                let before = polygon[(i + n - 1) % n];
                let after = polygon[(i + 1) % n];
                let d_prev = right_distance(before, polygon[i], out[i]);
                let d_next = right_distance(polygon[i], after, out[i]);
                assert!((d_prev - dist).abs() < 1e-12, "{name} {i}: {d_prev}");
                assert!((d_next - dist).abs() < 1e-12, "{name} {i}: {d_next}");
            }
        }
    }
}

/// Area of a miter-joined offset: `A + P d + d^2 * sum tan(turn / 2)`, where
/// `turn` is the signed turning angle at each vertex.
fn expected_offset_area(polygon: &[[f64; 2]], dist: f64) -> f64 {
    let n = polygon.len();
    let area = PolygonOffset::signed_area(polygon);
    let mut perimeter = 0.0;
    let mut tangents = 0.0;
    for i in 0..n {
        let before = polygon[(i + n - 1) % n];
        let here = polygon[i];
        let after = polygon[(i + 1) % n];
        let e_in = [here[0] - before[0], here[1] - before[1]];
        let e_out = [after[0] - here[0], after[1] - here[1]];
        perimeter += e_out[0].hypot(e_out[1]);
        let cross = e_in[0] * e_out[1] - e_in[1] * e_out[0];
        let dot = e_in[0] * e_out[0] + e_in[1] * e_out[1];
        tangents += (0.5 * cross.atan2(dot)).tan();
    }
    area + perimeter * dist + dist * dist * tangents
}

#[test]
fn reflex_corners_use_the_inner_intersection() {
    let l_shape = [
        [0.0, 0.0],
        [2.0, 0.0],
        [2.0, 1.0],
        [1.0, 1.0],
        [1.0, 2.0],
        [0.0, 2.0],
    ];
    let out = PolygonOffset::offset_polygon_2d(&l_shape, 0.1);
    assert_eq!(out.len(), 6);
    // The inner corner moves diagonally into the notch by 0.1 * sqrt(2).
    assert!(close(out[3], [1.1, 1.1]), "{:?}", out[3]);
    assert!((PolygonOffset::signed_area(&out) - 3.84).abs() < 1e-12);
    assert!((expected_offset_area(&l_shape, 0.1) - 3.84).abs() < 1e-12);
    let inward = PolygonOffset::offset_polygon_2d(&l_shape, -0.1);
    assert!(close(inward[3], [0.9, 0.9]), "{:?}", inward[3]);
    assert!((PolygonOffset::signed_area(&inward) - 2.24).abs() < 1e-12);
}

#[test]
fn offset_area_follows_the_steiner_formula_on_a_concave_polygon() {
    let polygon = [
        [0.0, 0.0],
        [4.0, 0.0],
        [4.0, 3.0],
        [2.5, 1.5],
        [1.0, 3.0],
        [0.0, 2.0],
    ];
    for dist in [0.05, 0.1, -0.05, -0.1] {
        let out = PolygonOffset::offset_polygon_2d(&polygon, dist);
        assert_eq!(
            out.len(),
            polygon.len(),
            "no corner is sharp enough to bevel"
        );
        let area = PolygonOffset::signed_area(&out);
        let expected = expected_offset_area(&polygon, dist);
        assert!(
            (area - expected).abs() < 1e-10,
            "{dist}: {area} vs {expected}"
        );
    }
}

#[test]
fn sharp_outer_corners_are_bevelled_instead_of_spiking() {
    // The corner at the origin has an interior angle of about 2.9 degrees: its
    // miter would reach 40 times the offset distance.
    let sliver = [[0.0, 0.0], [10.0, 0.0], [10.0, 0.5]];
    let dist = 0.1;
    let out = PolygonOffset::offset_polygon_2d(&sliver, dist);
    assert_eq!(out.len(), 4, "the sharp corner contributes two points");
    let len = 10.0_f64.hypot(0.5);
    let expected_first = [-0.5 / len * dist, 10.0 / len * dist];
    assert!(close(out[0], expected_first), "{:?}", out[0]);
    assert!(close(out[1], [0.0, -dist]), "{:?}", out[1]);
    // The two bevel points are exactly `dist` from the vertex.
    assert!((out[0][0].hypot(out[0][1]) - dist).abs() < 1e-12);
    assert!((out[1][0].hypot(out[1][1]) - dist).abs() < 1e-12);
    // The other two corners keep their single miter point, within four offsets
    // of their vertex.
    assert!(close(out[2], [10.0 + dist, -dist]), "{:?}", out[2]);
    for (point, vertex) in [(out[2], sliver[1]), (out[3], sliver[2])] {
        let reach = (point[0] - vertex[0]).hypot(point[1] - vertex[1]);
        assert!(reach <= 4.0 * dist, "{point:?} is {reach} from {vertex:?}");
    }
}

#[test]
fn sharp_inner_corners_keep_the_exact_intersection() {
    // Shrinking a sliver triangle: the inner polygon is the similar triangle
    // reached by the exact intersection of the displaced edges, not a bevel.
    let sliver = [[0.0, 0.0], [10.0, 0.0], [10.0, 1.0]];
    let out = PolygonOffset::offset_polygon_2d(&sliver, -0.05);
    assert_eq!(out.len(), 3);
    let area = PolygonOffset::signed_area(&out);
    let expected = expected_offset_area(&sliver, -0.05);
    assert!((area - expected).abs() < 1e-10, "{area} vs {expected}");
    // Each vertex is 0.05 inside both of its edges.
    let n = sliver.len();
    for i in 0..n {
        let d_prev = right_distance(sliver[(i + n - 1) % n], sliver[i], out[i]);
        let d_next = right_distance(sliver[i], sliver[(i + 1) % n], out[i]);
        assert!((d_prev + 0.05).abs() < 1e-12 && (d_next + 0.05).abs() < 1e-12);
    }
}

#[test]
fn degenerate_polygons_do_not_blow_up() {
    // Coincident consecutive vertices receive the same offset point.
    let doubled = [[0.0, 0.0], [1.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]];
    let out = PolygonOffset::offset_polygon_2d(&doubled, 0.1);
    assert_eq!(out.len(), 5);
    assert!(close(out[1], [1.1, -0.1]) && close(out[2], [1.1, -0.1]));
    assert!((PolygonOffset::signed_area(&out) - 1.44).abs() < 1e-12);
    // A polygon folded onto a line (a hairpin at every end) stays finite.
    let flat = [[0.0, 0.0], [2.0, 0.0], [1.0, 0.0]];
    let out = PolygonOffset::offset_polygon_2d(&flat, 0.1);
    assert!(out.iter().all(|p| p[0].is_finite() && p[1].is_finite()));
    // All vertices equal.
    let point = [[1.0, 1.0]; 4];
    assert_eq!(
        PolygonOffset::offset_polygon_2d(&point, 0.3),
        point.to_vec()
    );
}

#[test]
fn trivial_offsets_return_the_input() {
    let square = unit_square();
    assert_eq!(PolygonOffset::offset_polygon_2d(&square, 0.0), square);
    let segment = [[0.0, 0.0], [1.0, 1.0]];
    assert_eq!(
        PolygonOffset::offset_polygon_2d(&segment, 0.5),
        segment.to_vec()
    );
}

// ---------------------------------------------------------------------------
// compute_0d_persistence
// ---------------------------------------------------------------------------

fn finite_pairs(diagram: &PersistenceDiagram) -> Vec<(f64, f64)> {
    let mut pairs: Vec<(f64, f64)> = diagram
        .pairs
        .iter()
        .filter(|p| !p.is_essential())
        .map(|p| (p.birth, p.death))
        .collect();
    pairs.sort_by(|a, b| a.partial_cmp(b).expect("finite pairs"));
    pairs
}

fn essential_births(diagram: &PersistenceDiagram) -> Vec<f64> {
    let mut births: Vec<f64> = diagram
        .pairs
        .iter()
        .filter(|p| p.is_essential())
        .map(|p| p.birth)
        .collect();
    births.sort_by(|a, b| a.partial_cmp(b).expect("finite births"));
    births
}

#[test]
fn the_elder_birth_survives_a_merge() {
    // Union by rank may keep the younger root; the surviving component must
    // still carry the elder birth. The old code reported (1, 3) here.
    let values = [0.0, 1.0, 0.5, 3.0];
    let edges = [(1, 0), (2, 3), (3, 1)];
    let diagram = compute_0d_persistence(&values, &edges);
    assert_eq!(finite_pairs(&diagram), vec![(0.5, 3.0)]);
    assert_eq!(essential_births(&diagram), vec![0.0]);
    assert_eq!(diagram.pairs.len(), 2);
}

#[test]
fn two_minima_die_at_the_saddles() {
    // Path 0-1-2-3-4: minima 0, 1 and 2 separated by two ridges of height 4.
    let values = [0.0, 4.0, 1.0, 4.0, 2.0];
    let edges = [(0, 1), (1, 2), (2, 3), (3, 4)];
    let diagram = compute_0d_persistence(&values, &edges);
    assert_eq!(finite_pairs(&diagram), vec![(1.0, 4.0), (2.0, 4.0)]);
    assert_eq!(essential_births(&diagram), vec![0.0]);
}

#[test]
fn every_component_has_its_own_essential_class() {
    let diagram = compute_0d_persistence(&[0.0, 1.0], &[]);
    assert_eq!(finite_pairs(&diagram), Vec::<(f64, f64)>::new());
    assert_eq!(essential_births(&diagram), vec![0.0, 1.0]);
    assert!(diagram.pairs.iter().all(|p| p.dim == 0));

    // Two components: {0, 1} and {2, 3}; no merge across them.
    let values = [0.0, 5.0, 2.0, 3.0];
    let diagram = compute_0d_persistence(&values, &[(0, 1), (2, 3)]);
    assert_eq!(finite_pairs(&diagram), Vec::<(f64, f64)>::new());
    assert_eq!(essential_births(&diagram), vec![0.0, 2.0]);
    // An isolated third component joins them.
    let diagram = compute_0d_persistence(&[0.0, 5.0, 2.0, 3.0, 1.5], &[(0, 1), (2, 3)]);
    assert_eq!(essential_births(&diagram), vec![0.0, 1.5, 2.0]);
}

#[test]
fn monotone_path_has_only_the_essential_class() {
    // Every merge happens when the younger vertex is born: zero persistence.
    let values = [0.0, 1.0, 2.0, 3.0];
    let diagram = compute_0d_persistence(&values, &[(0, 1), (1, 2), (2, 3)]);
    assert_eq!(diagram.pairs.len(), 1);
    assert_eq!(finite_pairs(&diagram), Vec::<(f64, f64)>::new());
    assert_eq!(essential_births(&diagram), vec![0.0]);
    assert!(diagram.pairs[0].is_essential());
}

#[test]
fn empty_input_gives_an_empty_diagram() {
    assert!(compute_0d_persistence(&[], &[]).pairs.is_empty());
    // Edges that point outside the vertex range are ignored.
    let diagram = compute_0d_persistence(&[0.0, 2.0], &[(0, 7), (9, 1)]);
    assert_eq!(essential_births(&diagram), vec![0.0, 2.0]);
}

#[test]
fn nan_values_do_not_panic() {
    let values = [0.0, f64::NAN, 1.0, 2.0];
    let diagram = compute_0d_persistence(&values, &[(0, 1), (1, 2), (2, 3), (3, 0)]);
    assert!(!diagram.pairs.is_empty());
}

/// Components of the sublevel graph at level `t`, counted from scratch.
fn sublevel_components(values: &[f64], edges: &[(usize, usize)], t: f64) -> usize {
    let alive: Vec<usize> = (0..values.len()).filter(|&v| values[v] <= t).collect();
    let mut index = vec![usize::MAX; values.len()];
    for (k, &v) in alive.iter().enumerate() {
        index[v] = k;
    }
    let kept: Vec<(usize, usize)> = edges
        .iter()
        .filter(|&&(a, b)| values[a] <= t && values[b] <= t)
        .map(|&(a, b)| (index[a], index[b]))
        .collect();
    count_connected_components(alive.len(), &kept)
}

fn random_graph(rng: &mut Lcg, n: usize, edge_count: usize) -> Vec<(usize, usize)> {
    (0..edge_count)
        .map(|_| (rng.below(n), rng.below(n)))
        .filter(|&(a, b)| a != b)
        .collect()
}

#[test]
fn betti_numbers_of_the_diagram_match_the_sublevel_components() {
    let mut rng = Lcg(2026);
    for trial in 0..60 {
        let n = 3 + trial % 12;
        let values: Vec<f64> = (0..n).map(|_| rng.next_f64()).collect();
        let edges = random_graph(&mut rng, n, n + trial % 7);
        let diagram = compute_0d_persistence(&values, &edges);
        let mut levels = values.clone();
        levels.sort_by(f64::total_cmp);
        for &t in &levels {
            assert_eq!(
                diagram.betti_at(0, t),
                sublevel_components(&values, &edges, t),
                "trial {trial}, level {t}"
            );
        }
        // One essential class per connected component of the whole graph.
        let essentials = diagram.pairs.iter().filter(|p| p.is_essential()).count();
        assert_eq!(essentials, count_connected_components(n, &edges));
    }
}

#[test]
fn perturbing_the_function_moves_the_diagram_by_at_most_the_perturbation() {
    // Stability of persistence: d_B(PD(f), PD(g)) <= max |f - g|.
    let mut rng = Lcg(7);
    let delta = 0.05;
    for trial in 0..40 {
        let n = 4 + trial % 10;
        let values: Vec<f64> = (0..n).map(|_| rng.next_f64()).collect();
        let perturbed: Vec<f64> = values
            .iter()
            .map(|v| v + delta * (2.0 * rng.next_f64() - 1.0))
            .collect();
        let edges = random_graph(&mut rng, n, 2 * n);
        let a = compute_0d_persistence(&values, &edges);
        let b = compute_0d_persistence(&perturbed, &edges);
        let distance = a.bottleneck_distance(&b);
        assert!(distance <= delta + 1e-12, "trial {trial}: {distance}");
        assert!((distance - b.bottleneck_distance(&a)).abs() < 1e-15);
    }
}

// ---------------------------------------------------------------------------
// PersistenceDiagram::bottleneck_distance
// ---------------------------------------------------------------------------

fn diagram(pairs: &[(usize, f64, f64)]) -> PersistenceDiagram {
    let mut diagram = PersistenceDiagram::new();
    for &(dim, birth, death) in pairs {
        diagram.add_pair(dim, birth, death);
    }
    diagram
}

#[test]
fn bottleneck_against_an_empty_diagram_is_half_the_persistence() {
    // The old one-sided nearest match returned 0 here.
    let a = diagram(&[(0, 0.0, 4.0)]);
    let empty = PersistenceDiagram::new();
    assert_eq!(a.bottleneck_distance(&empty), 2.0);
    assert_eq!(empty.bottleneck_distance(&a), 2.0);
    assert_eq!(empty.bottleneck_distance(&PersistenceDiagram::new()), 0.0);
}

#[test]
fn bottleneck_is_a_one_to_one_matching() {
    let two = diagram(&[(1, 0.0, 10.0), (1, 0.0, 10.0)]);
    let one = diagram(&[(1, 0.0, 10.0)]);
    assert_eq!(two.bottleneck_distance(&one), 5.0);
    assert_eq!(one.bottleneck_distance(&two), 5.0);
    let shifted = diagram(&[(0, 0.0, 4.0)]);
    let other = diagram(&[(0, 1.0, 5.0)]);
    assert_eq!(shifted.bottleneck_distance(&other), 1.0);
    assert_eq!(other.bottleneck_distance(&shifted), 1.0);
    assert_eq!(shifted.bottleneck_distance(&shifted), 0.0);
}

#[test]
fn bottleneck_never_matches_across_dimensions() {
    let a = diagram(&[(0, 0.0, 4.0)]);
    let b = diagram(&[(1, 0.0, 4.0)]);
    assert_eq!(a.bottleneck_distance(&b), 2.0);
}

#[test]
fn bottleneck_matches_essential_classes_by_birth() {
    let a = diagram(&[(0, 0.0, f64::MAX), (0, 1.0, 3.0)]);
    let b = diagram(&[(0, 0.5, f64::MAX), (0, 1.0, 3.25)]);
    assert_eq!(a.bottleneck_distance(&b), 0.5);
    let c = diagram(&[(0, 0.0, f64::MAX), (0, 0.0, f64::MAX)]);
    assert_eq!(a.bottleneck_distance(&c), f64::INFINITY);
}
