//! Regression tests for the table-driven marching cubes of
//! [`extract_isosurface`].

use super::*;
use crate::isosurface::{EDGE_TABLE, TRI_TABLE};
use std::collections::{BTreeMap, BTreeSet};

/// Bit pattern of a vertex: vertices of neighbouring cells must agree exactly.
type VertexKey = [u64; 3];

fn vertex_key(p: [f64; 3]) -> VertexKey {
    p.map(f64::to_bits)
}

/// Number of times each undirected edge of the welded mesh is traversed in the
/// direction of increasing key (`.0`) and of decreasing key (`.1`).
fn edge_usage(tris: &[IsoTriangle]) -> BTreeMap<(VertexKey, VertexKey), (usize, usize)> {
    let mut usage = BTreeMap::new();
    for tri in tris {
        let k = tri.vertices.map(vertex_key);
        for (a, b) in [(k[0], k[1]), (k[1], k[2]), (k[2], k[0])] {
            let entry = usage.entry((a.min(b), a.max(b))).or_insert((0, 0));
            if a < b {
                entry.0 += 1;
            } else {
                entry.1 += 1;
            }
        }
    }
    usage
}

/// Every edge is shared by exactly two triangles that traverse it in opposite
/// directions: a closed, consistently oriented mesh.
fn assert_closed_and_oriented(tris: &[IsoTriangle]) {
    assert!(!tris.is_empty());
    for (edge, &(forward, backward)) in &edge_usage(tris) {
        assert_eq!(
            (forward, backward),
            (1, 1),
            "edge {edge:?} is not shared by two oppositely oriented triangles"
        );
    }
}

/// Euler characteristic `V - E + F` of the welded mesh.
fn euler_characteristic(tris: &[IsoTriangle]) -> i64 {
    let vertices: BTreeSet<VertexKey> = tris
        .iter()
        .flat_map(|t| t.vertices)
        .map(vertex_key)
        .collect();
    vertices.len() as i64 - edge_usage(tris).len() as i64 + tris.len() as i64
}

fn cross(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

fn dot(a: [f64; 3], b: [f64; 3]) -> f64 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

fn sub(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}

fn centroid(t: &IsoTriangle) -> [f64; 3] {
    let [a, b, c] = t.vertices;
    [
        (a[0] + b[0] + c[0]) / 3.0,
        (a[1] + b[1] + c[1]) / 3.0,
        (a[2] + b[2] + c[2]) / 3.0,
    ]
}

fn area(t: &IsoTriangle) -> f64 {
    let [a, b, c] = t.vertices;
    let n = cross(sub(b, a), sub(c, a));
    0.5 * dot(n, n).sqrt()
}

/// Volume enclosed by a closed outward-oriented mesh (sum of the signed
/// volumes of the tetrahedra spanned with the origin).
fn enclosed_volume(tris: &[IsoTriangle]) -> f64 {
    tris.iter()
        .map(|t| {
            let [a, b, c] = t.vertices;
            dot(a, cross(b, c)) / 6.0
        })
        .sum()
}

/// A unit-strength ball at the origin: the field is `1 / r^2`.
fn unit_ball() -> MetaballField {
    let mut field = MetaballField::new();
    field.add(Metaball::new([0.0; 3], 1.0));
    field
}

#[test]
fn metaball_uses_the_shared_marching_cubes_tables() {
    // The local copy of the edge table had a corrupt row (0x835, 0xb3f, 0xa36
    // instead of 0x83f, 0xb35, 0xa3c) and broke the complement symmetry.
    for case in 0..256usize {
        assert_eq!(EDGE_TABLE[case], EDGE_TABLE[255 - case], "case {case}");
    }
    assert_eq!(EDGE_TABLE[213], 0x83f);
    assert_eq!(EDGE_TABLE[214], 0xb35);
    assert_eq!(EDGE_TABLE[215], 0xa3c);
    // No case needs more than five triangles (15 of the 16 row entries).
    assert!(TRI_TABLE.iter().all(|row| row[15] == -1));
}

#[test]
fn vertices_lie_on_the_isosurface() {
    let field = unit_ball();
    let threshold = 0.3;
    let tris = extract_isosurface(&field, [-2.0; 3], [2.0; 3], GridResolution::Fine, threshold);
    assert!(tris.len() > 1000, "{} triangles", tris.len());
    for tri in &tris {
        for &v in &tri.vertices {
            let err = (field.evaluate(v) - threshold).abs();
            assert!(
                err < 0.05 * threshold,
                "vertex {v:?}: field {} vs {threshold}",
                field.evaluate(v)
            );
        }
    }
}

#[test]
fn single_ball_gives_a_closed_oriented_topological_sphere() {
    let field = unit_ball();
    for resolution in [
        GridResolution::Medium,
        GridResolution::Fine,
        GridResolution::Custom(24, 20, 28),
    ] {
        let tris = extract_isosurface(&field, [-2.0; 3], [2.0; 3], resolution, 0.3);
        assert_closed_and_oriented(&tris);
        assert_eq!(euler_characteristic(&tris), 2, "{resolution:?}");
    }
}

#[test]
fn surface_through_grid_vertices_is_still_watertight() {
    // With the threshold 0.5 the sphere r = sqrt(2) passes exactly through the
    // grid vertices (+-1, +-1, 0) and permutations, whose field value is
    // exactly 0.5: crossings land on grid vertices and some triangles
    // degenerate.  The vertices must still agree bit for bit between cells.
    let field = unit_ball();
    let tris = extract_isosurface(&field, [-2.0; 3], [2.0; 3], GridResolution::Fine, 0.5);
    assert_closed_and_oriented(&tris);
    for tri in &tris {
        assert!(area(tri) > 0.0, "degenerate triangles are not emitted");
        for &v in &tri.vertices {
            assert!((field.evaluate(v) - 0.5).abs() < 0.025, "vertex {v:?}");
        }
    }
}

#[test]
fn normals_point_out_of_the_blob() {
    let field = unit_ball();
    let tris = extract_isosurface(&field, [-2.0; 3], [2.0; 3], GridResolution::Fine, 0.3);
    for tri in &tris {
        let c = centroid(tri);
        let radial = dot(c, c).sqrt();
        let outward = dot(tri.normal, c) / radial;
        assert!(
            outward > 0.9,
            "normal {:?} at {c:?} does not point outwards (cos {outward})",
            tri.normal
        );
        // The stored normal is the unit normal of the winding order.
        let [a, b, v] = tri.vertices;
        let n = cross(sub(b, a), sub(v, a));
        let len = dot(n, n).sqrt();
        assert!((dot(tri.normal, n) / len - 1.0).abs() < 1e-9);
    }
}

#[test]
fn enclosed_area_and_volume_match_the_sphere() {
    // The level set 1 / r^2 = 0.3 is a sphere of radius sqrt(1 / 0.3).
    let field = unit_ball();
    let radius = (1.0_f64 / 0.3).sqrt();
    let tris = extract_isosurface(&field, [-2.0; 3], [2.0; 3], GridResolution::Fine, 0.3);
    let surface: f64 = tris.iter().map(area).sum();
    let exact_surface = 4.0 * std::f64::consts::PI * radius * radius;
    assert!(
        (surface / exact_surface - 1.0).abs() < 0.02,
        "area {surface} vs {exact_surface}"
    );
    let volume = enclosed_volume(&tris);
    let exact_volume = 4.0 / 3.0 * std::f64::consts::PI * radius.powi(3);
    assert!(
        (volume / exact_volume - 1.0).abs() < 0.02,
        "volume {volume} vs {exact_volume}"
    );
}

#[test]
fn merged_balls_give_one_closed_surface() {
    let mut field = MetaballField::new();
    field.add(Metaball::new([-0.6, 0.0, 0.0], 1.0));
    field.add(Metaball::new([0.6, 0.0, 0.0], 1.0));
    let threshold = 0.3;
    let tris = extract_isosurface(
        &field,
        [-3.0; 3],
        [3.0; 3],
        GridResolution::Custom(48, 48, 48),
        threshold,
    );
    assert_closed_and_oriented(&tris);
    assert_eq!(euler_characteristic(&tris), 2, "one merged blob");
    for tri in &tris {
        for &v in &tri.vertices {
            let err = (field.evaluate(v) - threshold).abs();
            assert!(err < 0.05 * threshold, "vertex {v:?}: error {err}");
        }
    }
}

#[test]
fn separated_balls_give_two_closed_surfaces() {
    let mut field = MetaballField::new();
    field.add(Metaball::new([-3.5, 0.0, 0.0], 1.0));
    field.add(Metaball::new([3.5, 0.0, 0.0], 1.0));
    let tris = extract_isosurface(
        &field,
        [-6.0, -3.0, -3.0],
        [6.0, 3.0, 3.0],
        GridResolution::Custom(96, 48, 48),
        0.3,
    );
    assert_closed_and_oriented(&tris);
    assert_eq!(euler_characteristic(&tris), 4, "two separate spheres");
}

#[test]
fn degenerate_inputs_give_no_triangles() {
    let field = unit_ball();
    // Inverted, empty and non-finite boxes.
    assert!(
        extract_isosurface(&field, [2.0; 3], [-2.0; 3], GridResolution::Medium, 0.3).is_empty()
    );
    assert!(extract_isosurface(&field, [0.0; 3], [0.0; 3], GridResolution::Medium, 0.3).is_empty());
    assert!(
        extract_isosurface(&field, [f64::NAN; 3], [2.0; 3], GridResolution::Medium, 0.3).is_empty()
    );
    // A field with non-finite samples.
    let mut broken = MetaballField::new();
    broken.add(Metaball::with_strength([0.0; 3], 1.0, f64::NAN));
    assert!(
        extract_isosurface(&broken, [-2.0; 3], [2.0; 3], GridResolution::Medium, 0.3).is_empty()
    );
    // A threshold above the whole field.
    assert!(
        extract_isosurface(&field, [-2.0; 3], [2.0; 3], GridResolution::Medium, 1e13).is_empty()
    );
}

#[test]
fn interpolation_reproduces_the_end_points_exactly() {
    // 0.1 + (-0.3 - 0.1) is -0.30000000000000004, not -0.3.
    let p0 = [0.1, 0.0, 0.0];
    let p1 = [-0.3, 0.0, 0.0];
    assert_eq!(interpolate_edge(p0, p1, 0.0, 1.0, 0.0), p0);
    assert_eq!(interpolate_edge(p0, p1, 0.0, 1.0, 1.0), p1);
    // The threshold outside the end values is clamped to the nearer end.
    assert_eq!(interpolate_edge(p0, p1, 0.0, 1.0, 5.0), p1);
    assert_eq!(interpolate_edge(p0, p1, 0.0, 1.0, -5.0), p0);
    // A crossing of a field with tiny values is interpolated, not centred.
    let tiny = interpolate_edge([0.0; 3], [1.0, 0.0, 0.0], 0.0, 4e-16, 1e-16);
    assert!((tiny[0] - 0.25).abs() < 1e-12, "{tiny:?}");
}

#[test]
fn triangle_normal_is_scale_independent() {
    let tri = |s: f64| [[0.0, 0.0, 0.0], [s, 0.0, 0.0], [0.0, s, 0.0]];
    for scale in [1e-12, 1e-3, 1.0, 1e6, 1e120] {
        let n = IsoTriangle::compute_normal(&tri(scale)).expect("a proper triangle");
        assert!(
            (n[2] - 1.0).abs() < 1e-12 && n[0].abs() < 1e-12 && n[1].abs() < 1e-12,
            "scale {scale}: {n:?}"
        );
    }
    assert!(IsoTriangle::compute_normal(&[[1.0, 2.0, 3.0]; 3]).is_none());
    let collinear = [[0.0; 3], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]];
    assert!(IsoTriangle::compute_normal(&collinear).is_none());
    let non_finite = [[0.0; 3], [f64::NAN, 0.0, 0.0], [0.0, 1.0, 0.0]];
    assert!(IsoTriangle::compute_normal(&non_finite).is_none());
}
