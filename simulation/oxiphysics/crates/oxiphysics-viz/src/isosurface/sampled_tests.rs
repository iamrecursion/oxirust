//! Tests of the node-sampled marching-cubes core that `marching_cubes`, the
//! voxel-grid extractor and the volume iso-surface routines share.

use super::*;
use std::collections::{BTreeMap, BTreeSet};

/// Deterministic xorshift generator returning values in `[0, 1)`.
struct TestRng(u64);

impl TestRng {
    fn next(&mut self) -> f64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        (self.0 >> 11) as f64 / (1u64 << 53) as f64
    }
}

fn bits(p: [f64; 3]) -> [u64; 3] {
    p.map(f64::to_bits)
}

/// Directed edges of the welded mesh and how often each is used.
fn directed_edge_counts(tris: &[Triangle]) -> BTreeMap<([u64; 3], [u64; 3]), usize> {
    let mut counts = BTreeMap::new();
    for tri in tris {
        for k in 0..3 {
            let from = bits(tri.vertices[k]);
            let to = bits(tri.vertices[(k + 1) % 3]);
            *counts.entry((from, to)).or_insert(0) += 1;
        }
    }
    counts
}

fn sphere_values(n: usize, centre: f64, radius: f64) -> Vec<f64> {
    let mut values = Vec::with_capacity(n * n * n);
    for i in 0..n {
        for j in 0..n {
            for k in 0..n {
                values.push(sphere_sdf(
                    [centre; 3],
                    radius,
                    i as f64,
                    j as f64,
                    k as f64,
                ));
            }
        }
    }
    values
}

#[test]
fn test_sampled_marching_cubes_is_independent_of_the_storage_layout() {
    let n = 9;
    let mut field = ScalarField::new(n, n, n, 0.5, 0.25, 2.0);
    for i in 0..n {
        for j in 0..n {
            for k in 0..n {
                field.set(
                    i,
                    j,
                    k,
                    sphere_sdf([4.0; 3], 2.6, i as f64, j as f64, k as f64),
                );
            }
        }
    }
    let from_field = marching_cubes(&field, 0.0);
    // x-fastest storage instead of the z-fastest `ScalarField` layout.
    let mut x_fastest = vec![0.0; n * n * n];
    for i in 0..n {
        for j in 0..n {
            for k in 0..n {
                x_fastest[k * n * n + j * n + i] = field.get(i, j, k);
            }
        }
    }
    let from_closure = marching_cubes_sampled(
        [n; 3],
        [0.5, 0.25, 2.0],
        |i, j, k| x_fastest[k * n * n + j * n + i],
        0.0,
    );
    assert!(!from_field.is_empty());
    assert_eq!(from_field.len(), from_closure.len());
    for (a, b) in from_field.iter().zip(&from_closure) {
        assert_eq!(a.vertices, b.vertices);
        assert_eq!(a.normal, b.normal);
    }
}

#[test]
fn test_sampled_marching_cubes_places_nodes_at_index_times_spacing() {
    // Field = x, iso 1.25 on a 5x2x2 lattice with spacing (0.5, 3, 7): the
    // plane x = 1.25 crosses the cells between nodes 2 (x = 1.0) and 3 (1.5).
    let tris = marching_cubes_sampled([5, 2, 2], [0.5, 3.0, 7.0], |i, _, _| i as f64 * 0.5, 1.25);
    assert_eq!(tris.len(), 2);
    for tri in &tris {
        for v in tri.vertices {
            assert!((v[0] - 1.25).abs() < 1e-12, "x = {}", v[0]);
            assert!((0.0..=3.0).contains(&v[1]), "y = {}", v[1]);
            assert!((0.0..=7.0).contains(&v[2]), "z = {}", v[2]);
        }
    }
}

#[test]
fn test_sampled_marching_cubes_needs_two_nodes_along_every_axis() {
    for dims in [[1, 4, 4], [4, 1, 4], [4, 4, 1], [0, 4, 4], [1, 1, 1]] {
        let tris = marching_cubes_sampled(dims, [1.0; 3], |i, j, k| (i + j + k) as f64, 3.5);
        assert!(tris.is_empty(), "dims {dims:?}");
    }
}

#[test]
fn test_sampled_marching_cubes_skips_cells_with_non_finite_corners() {
    let n = 8;
    let values = sphere_values(n, 3.5, 2.4);
    let nan_node = [3usize, 3, 1];
    let tris = marching_cubes_sampled(
        [n; 3],
        [1.0; 3],
        |i, j, k| {
            if [i, j, k] == nan_node {
                f64::NAN
            } else {
                values[(i * n + j) * n + k]
            }
        },
        0.0,
    );
    assert!(!tris.is_empty());
    for tri in &tris {
        for v in tri.vertices {
            assert!(v.iter().all(|c| c.is_finite()), "vertex {v:?}");
        }
        assert!(tri.normal.iter().all(|c| c.is_finite()));
    }
    // No vertex lies on an edge of a cell that touches the NaN node, i.e.
    // inside the unit box around it (vertices on the faces of that box
    // belong to the neighbouring cells).
    let touched = |v: &[f64; 3]| (0..3).all(|a| (v[a] - nan_node[a] as f64).abs() < 1.0 - 1e-9);
    assert!(tris.iter().all(|t| t.vertices.iter().all(|v| !touched(v))));

    let infinite = marching_cubes_sampled(
        [n; 3],
        [1.0; 3],
        |i, j, k| {
            if [i, j, k] == nan_node {
                f64::INFINITY
            } else {
                values[(i * n + j) * n + k]
            }
        },
        0.0,
    );
    for tri in &infinite {
        for v in tri.vertices {
            assert!(v.iter().all(|c| c.is_finite()), "vertex {v:?}");
        }
    }
}

#[test]
fn test_sampled_marching_cubes_shared_edge_vertices_are_bit_identical() {
    // Random node values make `interpolate_edge` direction-sensitive in the
    // last bit.  Every vertex must equal the vertex interpolated from the
    // lower-index to the higher-index node of its lattice edge.
    let n = 7;
    let h = [0.1, 0.3, 0.7];
    let iso = 0.5;
    let mut rng = TestRng(0x9e37_79b9_7f4a_7c15);
    let values: Vec<f64> = (0..n * n * n).map(|_| rng.next()).collect();
    let value = |i: usize, j: usize, k: usize| values[(i * n + j) * n + k];
    let position =
        |i: usize, j: usize, k: usize| [i as f64 * h[0], j as f64 * h[1], k as f64 * h[2]];

    let mut canonical: BTreeSet<[u64; 3]> = BTreeSet::new();
    for i in 0..n {
        for j in 0..n {
            for k in 0..n {
                for (di, dj, dk) in [(1, 0, 0), (0, 1, 0), (0, 0, 1)] {
                    let (i2, j2, k2) = (i + di, j + dj, k + dk);
                    if i2 >= n || j2 >= n || k2 >= n {
                        continue;
                    }
                    let (v0, v1) = (value(i, j, k), value(i2, j2, k2));
                    if (v0 < iso) != (v1 < iso) {
                        canonical.insert(bits(interpolate_edge(
                            position(i, j, k),
                            v0,
                            position(i2, j2, k2),
                            v1,
                            iso,
                        )));
                    }
                }
            }
        }
    }

    let tris = marching_cubes_sampled([n; 3], h, value, iso);
    assert!(tris.len() > 100, "only {} triangles", tris.len());
    for tri in &tris {
        for v in tri.vertices {
            assert!(
                canonical.contains(&bits(v)),
                "vertex {v:?} is not the canonical crossing of any lattice edge"
            );
        }
    }
}

#[test]
fn test_sampled_marching_cubes_sphere_is_a_closed_consistently_oriented_mesh() {
    let n = 12;
    let values = sphere_values(n, 5.5, 3.3);
    let tris = marching_cubes_sampled([n; 3], [1.0; 3], |i, j, k| values[(i * n + j) * n + k], 0.0);
    assert!(tris.len() > 50);

    let counts = directed_edge_counts(&tris);
    for (&(from, to), &count) in &counts {
        assert_eq!(
            count, 1,
            "directed edge {from:?}->{to:?} used {count} times"
        );
        assert_eq!(
            counts.get(&(to, from)),
            Some(&1),
            "edge {from:?}->{to:?} has no opposite twin (hole or flipped triangle)"
        );
    }
}
