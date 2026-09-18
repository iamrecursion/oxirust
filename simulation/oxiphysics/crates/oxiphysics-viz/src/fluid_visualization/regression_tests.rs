//! Regression tests for the table-driven [`marching_cubes`].

use super::*;
use std::collections::{BTreeMap, BTreeSet, HashMap};

/// Sample `f(x, y, z)` at the nodes of an `n`-cubed grid of spacing `h`
/// (x-major layout, as expected by [`marching_cubes`]).
fn sample_grid(n: usize, h: f64, f: impl Fn(f64, f64, f64) -> f64) -> Vec<f64> {
    let mut field = Vec::with_capacity(n * n * n);
    for ix in 0..n {
        for iy in 0..n {
            for iz in 0..n {
                field.push(f(ix as f64 * h, iy as f64 * h, iz as f64 * h));
            }
        }
    }
    field
}

fn triangle_area(t: &Triangle) -> f64 {
    let e1 = sub3(t.vertices[1], t.vertices[0]);
    let e2 = sub3(t.vertices[2], t.vertices[0]);
    0.5 * length3(cross3(e1, e2))
}

fn centroid(t: &Triangle) -> Vec3 {
    scale3(
        add3(add3(t.vertices[0], t.vertices[1]), t.vertices[2]),
        1.0 / 3.0,
    )
}

type VertexKey = [u64; 3];

fn vertex_key(p: Vec3) -> VertexKey {
    [p[0].to_bits(), p[1].to_bits(), p[2].to_bits()]
}

/// Asserts that the triangles form a closed, consistently oriented
/// 2-manifold (every directed edge occurs once, next to its reverse) and
/// returns the Euler characteristic `V - E + F`.
fn closed_manifold_euler_characteristic(tris: &[Triangle]) -> i64 {
    let mut directed: HashMap<(VertexKey, VertexKey), usize> = HashMap::new();
    let mut vertices: BTreeSet<VertexKey> = BTreeSet::new();
    for t in tris {
        let keys = t.vertices.map(vertex_key);
        assert!(
            keys[0] != keys[1] && keys[1] != keys[2] && keys[2] != keys[0],
            "degenerate triangle {t:?}"
        );
        vertices.extend(keys);
        for (a, b) in [(keys[0], keys[1]), (keys[1], keys[2]), (keys[2], keys[0])] {
            *directed.entry((a, b)).or_insert(0) += 1;
        }
    }
    for (&(a, b), &count) in &directed {
        assert_eq!(count, 1, "a directed edge is used {count} times");
        assert_eq!(
            directed.get(&(b, a)).copied(),
            Some(1),
            "an edge has no oppositely oriented partner"
        );
    }
    let edges = directed.len() / 2;
    vertices.len() as i64 - edges as i64 + tris.len() as i64
}

#[test]
fn sphere_sdf_gives_a_closed_oriented_sphere_of_the_right_area() {
    let (n, radius) = (32usize, 0.35);
    let h = 1.0 / (n - 1) as f64;
    let field = sample_grid(n, h, |x, y, z| {
        ((x - 0.5).powi(2) + (y - 0.5).powi(2) + (z - 0.5).powi(2)).sqrt() - radius
    });
    let tris = marching_cubes(&field, n, n, n, h, 0.0);
    assert!(!tris.is_empty());

    assert_eq!(
        closed_manifold_euler_characteristic(&tris),
        2,
        "a closed surface of genus 0 has Euler characteristic 2"
    );

    let area: f64 = tris.iter().map(triangle_area).sum();
    let exact = 4.0 * PI * radius * radius;
    assert!(
        (area - exact).abs() / exact < 0.03,
        "area {area} vs 4 pi r^2 = {exact}"
    );

    // Vertices sit on the sphere and the normals point towards decreasing
    // field values, i.e. into the sphere of a signed distance field.
    let mut mean_alignment = 0.0;
    for t in &tris {
        for v in t.vertices {
            let r = length3(sub3(v, [0.5; 3]));
            assert!((r - radius).abs() < 0.01 * radius, "vertex at radius {r}");
        }
        let radial = normalize3(sub3(centroid(t), [0.5; 3]));
        let alignment = dot3(t.normal, radial);
        assert!(alignment < -0.5, "normal {:?} is not inward", t.normal);
        assert!((length3(t.normal) - 1.0).abs() < 1e-12);
        mean_alignment += alignment / tris.len() as f64;
    }
    assert!(mean_alignment < -0.95, "mean alignment {mean_alignment}");
}

#[test]
fn flipping_the_field_flips_the_normals_but_not_the_surface() {
    let (n, radius) = (16usize, 0.3);
    let h = 1.0 / (n - 1) as f64;
    let sphere = |x: f64, y: f64, z: f64| {
        radius - ((x - 0.5).powi(2) + (y - 0.5).powi(2) + (z - 0.5).powi(2)).sqrt()
    };
    // `sphere` is high inside, so the normals leave the region above the
    // iso-value and point away from the centre.
    let field = sample_grid(n, h, sphere);
    let tris = marching_cubes(&field, n, n, n, h, 0.0);
    assert!(!tris.is_empty());
    for t in &tris {
        let radial = normalize3(sub3(centroid(t), [0.5; 3]));
        assert!(
            dot3(t.normal, radial) > 0.5,
            "normal {:?} is not outward",
            t.normal
        );
    }
    assert_eq!(closed_manifold_euler_characteristic(&tris), 2);
}

#[test]
fn axis_aligned_planes_give_flat_sheets_of_exact_area() {
    let n = 8usize;
    let h = 1.0 / (n - 1) as f64;
    let level = 0.4;
    for axis in 0..3 {
        let field = sample_grid(n, h, |x, y, z| [x, y, z][axis]);
        let tris = marching_cubes(&field, n, n, n, h, level);
        assert!(!tris.is_empty(), "axis {axis}");
        let area: f64 = tris.iter().map(triangle_area).sum();
        assert!((area - 1.0).abs() < 1e-12, "axis {axis}: area {area}");
        let mut expected_normal = [0.0; 3];
        expected_normal[axis] = -1.0;
        for t in &tris {
            for v in t.vertices {
                assert!((v[axis] - level).abs() < 1e-12, "axis {axis}: {v:?}");
            }
            for (component, expected) in t.normal.iter().zip(expected_normal) {
                assert!(
                    (component - expected).abs() < 1e-12,
                    "axis {axis}: normal {:?}",
                    t.normal
                );
            }
        }
    }
}

#[test]
fn z_plane_over_a_cell_face_uses_the_vertical_edges() {
    // Corners 0-3 (the z = 0 face) below the iso-value is case 15; its
    // surface crosses the four vertical edges 8-11 at mid-height.
    let mut field = [1.0_f64; 8];
    for (ix, iy) in [(0, 0), (1, 0), (1, 1), (0, 1)] {
        field[ix * 4 + iy * 2] = 0.0;
    }
    let tris = marching_cubes(&field, 2, 2, 2, 1.0, 0.5);
    assert_eq!(tris.len(), 2);
    let area: f64 = tris.iter().map(triangle_area).sum();
    assert!((area - 1.0).abs() < 1e-12, "area {area}");
    for t in &tris {
        for v in t.vertices {
            assert!((v[2] - 0.5).abs() < 1e-12, "vertex {v:?}");
        }
    }
}

/// Gradient of the trilinear interpolant of the corner `values` (standard
/// marching-cubes order) over the unit cube.
fn trilinear_gradient(values: &[f64; 8], p: Vec3) -> Vec3 {
    let mut g = [0.0; 3];
    for (corner, offset) in MC_CORNERS.iter().enumerate() {
        let weight = |axis: usize| {
            if offset[axis] == 1 {
                p[axis]
            } else {
                1.0 - p[axis]
            }
        };
        let slope = |axis: usize| if offset[axis] == 1 { 1.0 } else { -1.0 };
        g[0] += values[corner] * slope(0) * weight(1) * weight(2);
        g[1] += values[corner] * weight(0) * slope(1) * weight(2);
        g[2] += values[corner] * weight(0) * weight(1) * slope(2);
    }
    g
}

#[test]
fn every_corner_configuration_is_a_consistent_surface_patch() {
    // Mid-edge vertex keys on the doubled integer lattice of the unit cube.
    let key = |p: Vec3| {
        [
            (p[0] * 2.0) as i32,
            (p[1] * 2.0) as i32,
            (p[2] * 2.0) as i32,
        ]
    };
    for case in 1usize..255 {
        // One-cell grid; a set bit means the corner is below the iso-value.
        let mut field = [1.0_f64; 8];
        let mut corner_values = [1.0_f64; 8];
        for (corner, offset) in MC_CORNERS.iter().enumerate() {
            if case & (1 << corner) != 0 {
                field[offset[0] * 4 + offset[1] * 2 + offset[2]] = 0.0;
                corner_values[corner] = 0.0;
            }
        }
        let tris = marching_cubes(&field, 2, 2, 2, 1.0, 0.5);
        assert!(
            !tris.is_empty() && tris.len() <= 5,
            "case {case}: {} triangles",
            tris.len()
        );

        // The vertices are exactly the midpoints of the crossed edges.
        let expected: BTreeSet<[i32; 3]> = MC_EDGES
            .iter()
            .filter(|&&(a, b)| (case >> a) & 1 != (case >> b) & 1)
            .map(|&(a, b)| {
                let (pa, pb) = (MC_CORNERS[a], MC_CORNERS[b]);
                [
                    (pa[0] + pb[0]) as i32,
                    (pa[1] + pb[1]) as i32,
                    (pa[2] + pb[2]) as i32,
                ]
            })
            .collect();
        let used: BTreeSet<[i32; 3]> = tris.iter().flat_map(|t| t.vertices.map(key)).collect();
        assert_eq!(
            used, expected,
            "case {case}: vertices must be the crossed edges"
        );

        // A side lying in a cube face occurs once (the face contour); every
        // other side is shared by two triangles that traverse it in opposite
        // directions.
        let mut sides: BTreeMap<([i32; 3], [i32; 3]), (usize, usize)> = BTreeMap::new();
        for t in &tris {
            let k = t.vertices.map(key);
            assert!(
                k[0] != k[1] && k[1] != k[2] && k[2] != k[0],
                "case {case}: degenerate"
            );
            for (a, b) in [(k[0], k[1]), (k[1], k[2]), (k[2], k[0])] {
                let entry = sides.entry((a.min(b), a.max(b))).or_insert((0, 0));
                if a < b {
                    entry.0 += 1;
                } else {
                    entry.1 += 1;
                }
            }
            // The normal points towards decreasing field values.
            let g = trilinear_gradient(&corner_values, centroid(t));
            assert!(
                dot3(t.normal, g) < 0.0,
                "case {case}: normal {:?} does not follow -grad {g:?}",
                t.normal
            );
        }
        for (&(a, b), &(forward, backward)) in &sides {
            let on_face = (0..3).any(|axis| a[axis] == b[axis] && (a[axis] == 0 || a[axis] == 2));
            if on_face {
                assert_eq!(forward + backward, 1, "case {case}: face side {a:?}-{b:?}");
            } else {
                assert_eq!(
                    (forward, backward),
                    (1, 1),
                    "case {case}: inner side {a:?}-{b:?}"
                );
            }
        }
    }
}

#[test]
fn tilted_plane_is_a_watertight_disc_with_shared_vertices() {
    // The plane 0.7x + 0.5y + 0.3z = 0.93 crosses many cells and grid edges
    // at generic positions (the left-hand side is an even multiple of 0.01 at
    // every node, so no node lies on the plane).
    let n = 6usize;
    let h = 0.2;
    let field = sample_grid(n, h, |x, y, z| 0.7 * x + 0.5 * y + 0.3 * z);
    let tris = marching_cubes(&field, n, n, n, h, 0.93);
    assert!(tris.len() > 10, "{} triangles", tris.len());

    let mut directed: HashMap<(VertexKey, VertexKey), usize> = HashMap::new();
    let mut vertices: BTreeSet<VertexKey> = BTreeSet::new();
    for t in &tris {
        let keys = t.vertices.map(vertex_key);
        assert!(keys[0] != keys[1] && keys[1] != keys[2] && keys[2] != keys[0]);
        vertices.extend(keys);
        for (a, b) in [(keys[0], keys[1]), (keys[1], keys[2]), (keys[2], keys[0])] {
            *directed.entry((a, b)).or_insert(0) += 1;
        }
    }
    assert!(directed.values().all(|&count| count == 1));
    // Interior edges occur in both directions, boundary edges in one; sharing
    // bit-identical vertices makes the mesh a single disc: V - E + F = 1.
    let boundary = directed
        .keys()
        .filter(|&&(a, b)| !directed.contains_key(&(b, a)))
        .count();
    let edges = (directed.len() - boundary) / 2 + boundary;
    assert_eq!(vertices.len() as i64 - edges as i64 + tris.len() as i64, 1);

    // All triangles lie in the plane and face towards -grad f.
    let gradient = normalize3([0.7, 0.5, 0.3]);
    for t in &tris {
        for v in t.vertices {
            let value = 0.7 * v[0] + 0.5 * v[1] + 0.3 * v[2];
            assert!((value - 0.93).abs() < 1e-12, "vertex {v:?} off the plane");
        }
        assert!(
            dot3(t.normal, gradient) < -0.999999,
            "normal {:?}",
            t.normal
        );
    }
}

#[test]
fn degenerate_inputs_return_no_triangles() {
    let field = sample_grid(3, 1.0, |x, _, _| x - 0.5);
    assert!(!marching_cubes(&field, 3, 3, 3, 1.0, 0.0).is_empty());
    assert!(marching_cubes(&field[..26], 3, 3, 3, 1.0, 0.0).is_empty());
    assert!(marching_cubes(&field, 1, 3, 3, 1.0, 0.0).is_empty());
    assert!(marching_cubes(&[], 0, 0, 0, 1.0, 0.0).is_empty());
    assert!(marching_cubes(&field, usize::MAX, 2, 2, 1.0, 0.0).is_empty());
}

#[test]
fn cells_with_non_finite_corners_are_skipped() {
    let mut field = sample_grid(3, 1.0, |x, _, _| x - 0.5);
    field[13] = f64::NAN; // the centre node touches all eight cells
    assert!(marching_cubes(&field, 3, 3, 3, 1.0, 0.0).is_empty());
    let mut field = sample_grid(4, 1.0, |x, _, _| x - 0.5);
    field[0] = f64::INFINITY;
    let tris = marching_cubes(&field, 4, 4, 4, 1.0, 0.0);
    assert!(!tris.is_empty());
    for t in &tris {
        assert!(t.vertices.iter().flatten().all(|c| c.is_finite()));
    }
}

#[test]
fn pressure_isosurface_extracts_the_pressure_level_set() {
    let n = 16usize;
    let h = 0.1;
    let mut snapshot = FluidGridSnapshot::zeros(n, n, n, h);
    let centre = 0.5 * (n - 1) as f64 * h;
    snapshot.pressure = sample_grid(n, h, |x, y, z| {
        let r = ((x - centre).powi(2) + (y - centre).powi(2) + (z - centre).powi(2)).sqrt();
        100.0 - 40.0 * r
    });
    // p = 70 at r = 0.75.
    let tris = snapshot.pressure_isosurface(70.0);
    assert!(!tris.is_empty());
    let area: f64 = tris.iter().map(triangle_area).sum();
    let exact = 4.0 * PI * 0.75_f64.powi(2);
    assert!(
        (area - exact).abs() / exact < 0.05,
        "area {area} vs {exact}"
    );
    assert_eq!(closed_manifold_euler_characteristic(&tris), 2);
}
