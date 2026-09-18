// Copyright 2026 COOLJAPAN OU (Team KitaSan)
// SPDX-License-Identifier: Apache-2.0

//! Tests of [`HullTopology`] and of the topology cache of
//! [`ConvexHull`](crate::convex_hull::types::ConvexHull).

use std::borrow::Cow;

use oxiphysics_core::math::Vec3;

use super::*;
use crate::convex_hull::types::{ConvexHull, ConvexHull3D};

/// Corners of an axis-aligned cube with half-extent `half` centred at the origin.
fn cube_points(half: f64) -> Vec<[f64; 3]> {
    let mut points = Vec::new();
    for sx in [-1.0, 1.0] {
        for sy in [-1.0, 1.0] {
            for sz in [-1.0, 1.0] {
                points.push([sx * half, sy * half, sz * half]);
            }
        }
    }
    points
}

/// A hexagonal prism with axis `y`, circumradius `r` and half-height `hh`.
fn hex_prism_points(r: f64, hh: f64) -> Vec<[f64; 3]> {
    let mut points = Vec::new();
    for k in 0..6 {
        let angle = std::f64::consts::TAU * f64::from(k) / 6.0;
        let (x, z) = (r * angle.cos(), r * angle.sin());
        points.push([x, hh, z]);
        points.push([x, -hh, z]);
    }
    points
}

/// The 27 points of a 3x3x3 lattice: corners, edge midpoints, face centres and
/// the centre (many coplanar and collinear points, one interior point).
fn lattice_points() -> Vec<[f64; 3]> {
    let mut points = Vec::new();
    for x in [-0.5, 0.0, 0.5] {
        for y in [-0.5, 0.0, 0.5] {
            for z in [-0.5, 0.0, 0.5] {
                points.push([x, y, z]);
            }
        }
    }
    points
}

/// `count` points spread evenly over a sphere (Fibonacci lattice): all of them
/// are hull vertices and no four are coplanar.
fn sphere_points(count: usize, radius: f64) -> Vec<[f64; 3]> {
    let golden_angle = std::f64::consts::PI * (3.0 - 5.0_f64.sqrt());
    (0..count)
        .map(|i| {
            let y = 1.0 - 2.0 * (i as f64 + 0.5) / count as f64;
            let ring = (1.0 - y * y).sqrt();
            let phi = golden_angle * i as f64;
            [
                radius * ring * phi.cos(),
                radius * y,
                radius * ring * phi.sin(),
            ]
        })
        .collect()
}

/// Deterministic pseudo-random numbers in `[-1, 1]`.
struct Lcg(u64);

impl Lcg {
    fn next(&mut self) -> f64 {
        self.0 = self
            .0
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        2.0 * ((self.0 >> 11) as f64 / (1u64 << 53) as f64) - 1.0
    }
}

/// Rotate about the axis `(1, 2, 3)` by 0.7 rad (Rodrigues).
fn rotated(p: [f64; 3]) -> [f64; 3] {
    let axis = normalize([1.0, 2.0, 3.0]);
    let (sin, cos) = 0.7_f64.sin_cos();
    add(
        add(scale(p, cos), scale(cross(axis, p), sin)),
        scale(axis, dot(axis, p) * (1.0 - cos)),
    )
}

/// Rotate as [`rotated`] and translate.
fn turned(points: &[[f64; 3]]) -> Vec<[f64; 3]> {
    points
        .iter()
        .map(|&p| add(rotated(p), [3.0, -2.0, 1.5]))
        .collect()
}

fn topology_of(points: &[[f64; 3]]) -> HullTopology {
    HullTopology::build(points).expect("non-degenerate point set")
}

#[test]
fn cube_merges_to_six_quads_and_twelve_edges() {
    // A cube point cloud triangulates into 12 triangles; the coplanar merge must
    // recover the 6 square faces.
    let topology = topology_of(&cube_points(0.5));
    assert_eq!(topology.vertices().len(), 8);
    assert_eq!(topology.faces().len(), 6, "cube should merge to 6 quads");
    assert_eq!(topology.normals().len(), 6);
    for face in topology.faces() {
        assert_eq!(face.len(), 4, "each merged face is a quad");
    }
    assert_eq!(topology.edges().len(), 12, "cube has 12 unique edges");
}

#[test]
fn cube_normals_are_unit_and_axis_aligned() {
    let topology = topology_of(&cube_points(0.5));
    for normal in topology.normals() {
        assert!((norm(*normal) - 1.0).abs() < 1e-9, "unit normal");
        let big = normal.iter().filter(|c| c.abs() > 0.9).count();
        assert_eq!(big, 1, "axis-aligned face normal {normal:?}");
    }
}

#[test]
fn hex_prism_extracts_eight_faces_with_caps() {
    // Two hexagonal caps (6 vertices each) plus 6 rectangular side faces: the
    // case a triangle-soup hull mangles.
    let topology = topology_of(&hex_prism_points(1.0, 0.5));
    assert_eq!(topology.faces().len(), 8, "hex prism has 8 faces");
    let top = topology
        .normals()
        .iter()
        .position(|n| n[1] > 0.99)
        .expect("a +y cap face exists");
    assert_eq!(topology.faces()[top].len(), 6, "top cap is a hexagon");
    let bottom = topology
        .normals()
        .iter()
        .position(|n| n[1] < -0.99)
        .expect("a -y cap face exists");
    assert_eq!(topology.faces()[bottom].len(), 6, "bottom cap is a hexagon");
    let sides = topology
        .normals()
        .iter()
        .filter(|n| n[1].abs() < 0.01)
        .count();
    assert_eq!(sides, 6, "six vertical side faces");
}

#[test]
fn lattice_faces_gather_every_coplanar_point() {
    let points = lattice_points();
    let topology = topology_of(&points);
    assert_eq!(topology.faces().len(), 6);
    for face in topology.faces() {
        // Corners, edge midpoints and the face centre: nine points per face, and
        // the interior centre point belongs to none.
        assert_eq!(face.len(), 9);
        assert!(
            face.iter()
                .all(|&vi| topology.vertices()[vi] != [0.0, 0.0, 0.0])
        );
    }
}

#[test]
fn round_hull_is_a_valid_triangulated_polyhedron() {
    // 300 points: the old cubic triple enumeration needed minutes here.
    let count = 300;
    let topology = topology_of(&sphere_points(count, 1.0));
    // All points are extreme and in general position: every face is a triangle
    // and Euler's formula gives F = 2 V - 4 and E = 3 V - 6.
    assert_eq!(topology.faces().len(), 2 * count - 4);
    assert!(topology.faces().iter().all(|f| f.len() == 3));
    assert_eq!(topology.edges().len(), 3 * count - 6);
    for (face, normal) in topology.faces().iter().zip(topology.normals()) {
        assert!((norm(*normal) - 1.0).abs() < 1e-9);
        let offset = dot(topology.vertices()[face[0]], *normal);
        // Supporting plane, outward: every vertex is below it, the origin
        // (inside) is strictly below.
        assert!(
            topology
                .vertices()
                .iter()
                .all(|v| dot(*v, *normal) <= offset + 1e-7)
        );
        assert!(offset > 0.0);
    }
}

#[test]
fn every_edge_borders_exactly_two_faces() {
    for (name, points) in [
        ("cube", cube_points(0.5)),
        ("hex prism", hex_prism_points(1.0, 0.5)),
        ("sphere", sphere_points(64, 1.0)),
    ] {
        let topology = topology_of(&points);
        for &[a, b] in topology.edges() {
            let sharing = topology
                .faces()
                .iter()
                .filter(|face| {
                    (0..face.len()).any(|i| {
                        let (p, q) = (face[i], face[(i + 1) % face.len()]);
                        (p == a && q == b) || (p == b && q == a)
                    })
                })
                .count();
            assert_eq!(sharing, 2, "{name}: edge {a}-{b}");
        }
    }
}

#[test]
fn edges_are_listed_once_in_the_direction_they_are_first_met() {
    for points in [cube_points(0.5), hex_prism_points(1.0, 0.5)] {
        let topology = topology_of(&points);
        let mut expected: Vec<[usize; 2]> = Vec::new();
        for face in topology.faces() {
            for i in 0..face.len() {
                let (a, b) = (face[i], face[(i + 1) % face.len()]);
                let known = expected
                    .iter()
                    .any(|e| (e[0] == a && e[1] == b) || (e[0] == b && e[1] == a));
                if !known {
                    expected.push([a, b]);
                }
            }
        }
        assert_eq!(topology.edges(), expected.as_slice());
    }
}

#[test]
fn degenerate_point_sets_have_no_topology() {
    let square = vec![
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [1.0, 0.0, 1.0],
        [0.0, 0.0, 1.0],
    ];
    let collinear = vec![[0.0; 3], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0], [3.0, 0.0, 0.0]];
    let mut with_nan = cube_points(0.5);
    with_nan[3][1] = f64::NAN;
    let mut with_inf = cube_points(0.5);
    with_inf[5][2] = f64::INFINITY;
    // Eight points, but only three distinct ones.
    let repeated: Vec<[f64; 3]> = (0..8)
        .map(|i| [[0.0; 3], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]][i % 3])
        .collect();
    assert!(HullTopology::build(&[]).is_none());
    assert!(HullTopology::build(&cube_points(0.5)[..3]).is_none());
    assert!(HullTopology::build(&square).is_none(), "coplanar");
    assert!(HullTopology::build(&collinear).is_none(), "collinear");
    assert!(HullTopology::build(&with_nan).is_none(), "NaN coordinate");
    assert!(
        HullTopology::build(&with_inf).is_none(),
        "infinite coordinate"
    );
    assert!(HullTopology::build(&repeated).is_none(), "three distinct");
}

#[test]
fn near_duplicates_collapse_and_keep_the_first_occurrence() {
    let mut points = cube_points(0.5);
    let original = points.clone();
    // Repeat two corners, once exactly and once within the duplicate distance.
    points.push(original[0]);
    points.push(add(original[6], [1e-11, -1e-11, 0.0]));
    let topology = topology_of(&points);
    assert_eq!(topology.vertices(), original.as_slice());
    assert_eq!(topology.faces().len(), 6);
}

#[test]
fn into_parts_hands_out_the_accessor_data() {
    let topology = topology_of(&cube_points(0.5));
    let (vertices, faces, normals, edges) = topology.clone().into_parts();
    assert_eq!(vertices.as_slice(), topology.vertices());
    assert_eq!(faces.as_slice(), topology.faces());
    assert_eq!(normals.as_slice(), topology.normals());
    assert_eq!(edges.as_slice(), topology.edges());
}

/// Whether `b` is `a` read from another starting vertex.
fn same_cycle(a: &[usize], b: &[usize]) -> bool {
    a.len() == b.len()
        && (0..a.len().max(1)).any(|s| (0..a.len()).all(|i| a[(i + s) % a.len()] == b[i]))
}

/// A rigid motion maps the topology of a cloud onto the topology of the moved
/// cloud: the faces are the same loops of vertex indices (up to the vertex a
/// loop starts at, which depends on rounding when a face has a vertex exactly
/// opposite the first one), the edges are the same set and the normals turn with
/// the body.  This is what makes it valid to build the topology once in the
/// local frame of a hull and to pose it per query.
#[test]
fn rigid_motion_preserves_the_topology() {
    let mut rng = Lcg(0x5EED_1234_ABCD_0001);
    let random: Vec<[f64; 3]> = (0..14)
        .map(|_| [rng.next(), rng.next(), rng.next()])
        .collect();
    let sets: Vec<(&str, Vec<[f64; 3]>)> = vec![
        ("cube", cube_points(0.5)),
        ("hex prism", hex_prism_points(1.0, 0.5)),
        ("sphere", sphere_points(64, 0.7)),
        ("random cloud", random),
    ];
    for (name, points) in sets {
        let local = topology_of(&points);
        let world = topology_of(&turned(&points));
        assert_eq!(local.faces().len(), world.faces().len(), "{name}: faces");
        for (a, b) in local.faces().iter().zip(world.faces()) {
            assert!(same_cycle(a, b), "{name}: loop {a:?} vs {b:?}");
        }
        let sorted = |edges: &[[usize; 2]]| {
            let mut list = edges.to_vec();
            list.sort_unstable();
            list
        };
        assert_eq!(
            sorted(local.edges()),
            sorted(world.edges()),
            "{name}: edges"
        );
        assert_eq!(local.vertices().len(), world.vertices().len(), "{name}");
        for (local_normal, world_normal) in local.normals().iter().zip(world.normals()) {
            let expected = rotated(*local_normal);
            assert!(
                norm(sub(expected, *world_normal)) < 1e-9,
                "{name}: normal {world_normal:?} != rotated {expected:?}"
            );
        }
    }
}

// ── oracle ──────────────────────────────────────────────────────────────────

/// The first implementation, kept as an oracle: every plane through three points
/// that leaves all other points on one side is a face.  `O(n^4)`, so only for
/// small sets.
fn reference_faces(vertices: &[[f64; 3]]) -> Option<FaceSet> {
    let n = vertices.len();
    let mut centroid = [0.0_f64; 3];
    for v in vertices {
        centroid = add(centroid, *v);
    }
    centroid = scale(centroid, 1.0 / n as f64);
    let extent = vertices
        .iter()
        .map(|v| norm(sub(*v, centroid)))
        .fold(0.0_f64, f64::max)
        .max(1.0);
    let plane_eps = 1e-7 * extent;
    let mut faces: Vec<Vec<usize>> = Vec::new();
    let mut normals: Vec<[f64; 3]> = Vec::new();
    for i in 0..n {
        for j in (i + 1)..n {
            for k in (j + 1)..n {
                let a = vertices[i];
                let raw = cross(sub(vertices[j], a), sub(vertices[k], a));
                if norm(raw) < 1e-12 {
                    continue;
                }
                let mut normal = normalize(raw);
                if dot(sub(centroid, a), normal) > 0.0 {
                    normal = [-normal[0], -normal[1], -normal[2]];
                }
                let d = dot(a, normal);
                if vertices.iter().any(|v| dot(*v, normal) > d + plane_eps) {
                    continue;
                }
                let already = normals.iter().enumerate().any(|(fi, m)| {
                    dot(*m, normal) > COPLANAR_DOT && {
                        let fv = faces[fi][0];
                        (dot(vertices[fv], normal) - d).abs() < plane_eps
                    }
                });
                if already {
                    continue;
                }
                let on_plane: Vec<usize> = (0..n)
                    .filter(|&vi| (dot(vertices[vi], normal) - d).abs() < plane_eps)
                    .collect();
                if on_plane.len() < 3 {
                    continue;
                }
                faces.push(order_face_ccw(vertices, &on_plane, normal));
                normals.push(normal);
            }
        }
    }
    if faces.is_empty() {
        None
    } else {
        Some((faces, normals))
    }
}

/// Point sets of every kind the extraction has to handle, by name.
fn shape_families() -> Vec<(&'static str, Vec<[f64; 3]>)> {
    let mut rng = Lcg(0x00C0_FFEE_1234_5678);
    let random: Vec<[f64; 3]> = (0..14)
        .map(|_| [rng.next(), rng.next(), rng.next()])
        .collect();
    let tetra = vec![
        [1.0, 1.0, 1.0],
        [-1.0, -1.0, 1.0],
        [-1.0, 1.0, -1.0],
        [1.0, -1.0, -1.0],
    ];
    let octahedron = vec![
        [1.0, 0.0, 0.0],
        [-1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, -1.0, 0.0],
        [0.0, 0.0, 1.0],
        [0.0, 0.0, -1.0],
    ];
    vec![
        ("cube", cube_points(0.5)),
        ("lattice", lattice_points()),
        ("turned lattice", turned(&lattice_points())),
        ("hex prism", hex_prism_points(1.0, 0.5)),
        ("tetrahedron", tetra),
        ("octahedron", octahedron),
        ("random cloud", random),
        ("sphere", sphere_points(18, 0.7)),
        ("turned sphere", turned(&sphere_points(18, 0.7))),
    ]
}

#[test]
fn extraction_matches_the_triple_enumeration_oracle() {
    for (name, points) in shape_families() {
        let points = dedup_vertices(&points);
        let (ref_faces, ref_normals) = reference_faces(&points).expect(name);
        let (faces, normals) = extract_convex_faces(&points).expect(name);
        assert_eq!(faces, ref_faces, "{name}: face loops");
        assert_eq!(normals.len(), ref_normals.len(), "{name}");
        for (got, want) in normals.iter().zip(&ref_normals) {
            assert!(
                norm(sub(*got, *want)) < 1e-9,
                "{name}: normal {got:?} != {want:?}"
            );
        }
    }
}

#[test]
fn closed_surface_check_accepts_only_closed_triangulations() {
    let tetrahedron = [[0, 2, 1], [0, 1, 3], [1, 2, 3], [0, 3, 2]];
    assert!(is_closed_surface(&tetrahedron));
    assert!(!is_closed_surface(&[]), "no triangles");
    assert!(
        !is_closed_surface(&tetrahedron[..3]),
        "a triangle is missing"
    );
    let mut doubled = tetrahedron.to_vec();
    doubled.push(tetrahedron[0]);
    assert!(!is_closed_surface(&doubled), "a triangle is listed twice");
    let mut dangling = tetrahedron.to_vec();
    dangling.push([1, 3, 4]);
    assert!(
        !is_closed_surface(&dangling),
        "an extra triangle hangs off an edge"
    );
}

#[test]
fn the_hull_builder_returns_a_closed_surface_for_every_shape_family() {
    for (name, points) in shape_families() {
        let hull = ConvexHull3D::build(&dedup_vertices(&points)).expect(name);
        assert!(is_closed_surface(&hull.faces), "{name}");
    }
}

#[test]
fn a_hull_surface_with_a_hole_gives_no_facets() {
    let points = cube_points(0.5);
    let mut hull = ConvexHull3D::build(&points).expect("cube hull");
    assert!(
        facets_from_hull(&points, &hull).is_some(),
        "the intact surface is accepted"
    );
    hull.faces.pop();
    assert!(
        facets_from_hull(&points, &hull).is_none(),
        "a missing triangle leaves a hole"
    );
}

#[test]
fn a_hull_that_left_a_point_out_gives_no_facets() {
    let cube = cube_points(0.5);
    let hull = ConvexHull3D::build(&cube).expect("cube hull");
    // A point inside the cube changes nothing.
    let mut inside = cube.clone();
    inside.push([0.1, -0.2, 0.3]);
    assert!(facets_from_hull(&inside, &hull).is_some());
    // A point far above the top face is in front of the plane of the top facet,
    // which the stale hull does not know.
    let mut outside = cube;
    outside.push([0.0, 2.0, 0.0]);
    assert!(facets_from_hull(&outside, &hull).is_none());
}

// ── the cache on ConvexHull ─────────────────────────────────────────────────

fn hull_of(points: &[[f64; 3]]) -> ConvexHull {
    ConvexHull::new(points.iter().map(|p| Vec3::new(p[0], p[1], p[2])).collect())
}

#[test]
fn hull_topology_matches_a_direct_build_and_is_borrowed_from_the_cache() {
    let points = sphere_points(64, 0.5);
    let hull = hull_of(&points);
    let first = hull.topology().expect("hull spans a volume");
    let second = hull.topology().expect("hull spans a volume");
    assert!(matches!(first, Cow::Borrowed(_)), "served from the cache");
    assert!(matches!(second, Cow::Borrowed(_)), "served from the cache");
    assert!(
        std::ptr::eq(&*first, &*second),
        "every query sees the one cached topology"
    );
    assert_eq!(*first, topology_of(&points));
}

#[test]
fn hull_topology_cache_is_filled_by_the_first_query_only() {
    let hull = hull_of(&cube_points(0.5));
    assert!(hull.topology.get().is_none(), "nothing is built eagerly");
    let _ = hull.topology();
    assert!(hull.topology.get().is_some_and(|c| c.topology.is_some()));
}

#[test]
fn changed_vertices_invalidate_the_cached_topology() {
    let mut hull = hull_of(&cube_points(0.5));
    assert_eq!(hull.topology().expect("cube").faces().len(), 6);
    // Raise a pyramid on the cube: the hull now has a different topology, and the
    // stale cache must not be served.
    hull.vertices.push(Vec3::new(0.0, 2.0, 0.0));
    let changed = hull.topology().expect("pyramid-capped cube");
    assert!(matches!(changed, Cow::Owned(_)), "rebuilt, not cached");
    let direct = {
        let points: Vec<[f64; 3]> = hull.vertices.iter().map(|v| [v.x, v.y, v.z]).collect();
        topology_of(&points)
    };
    assert_eq!(*changed, direct);
    assert_ne!(changed.vertices().len(), 8);
    // Shrinking the vertex list is detected too.
    hull.vertices.truncate(8);
    let restored = hull.topology().expect("cube again");
    assert!(matches!(restored, Cow::Borrowed(_)));
    assert_eq!(restored.faces().len(), 6);
}

#[test]
fn hull_without_volume_has_no_topology_and_the_miss_is_cached() {
    let hull = hull_of(&[
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [1.0, 0.0, 1.0],
        [0.0, 0.0, 1.0],
    ]);
    assert!(hull.topology().is_none());
    assert!(
        hull.topology.get().is_some_and(|c| c.topology.is_none()),
        "the missing topology is remembered"
    );
    assert!(hull.topology().is_none());
}

#[test]
fn cloned_hull_serves_an_equal_topology() {
    let hull = hull_of(&hex_prism_points(1.0, 0.5));
    let original = hull.topology().expect("prism").into_owned();
    let copy = hull.clone();
    assert!(matches!(copy.topology(), Some(Cow::Borrowed(_))));
    assert_eq!(copy.topology().expect("prism").into_owned(), original);
}

#[test]
fn threads_share_one_cached_topology() {
    let hull = hull_of(&sphere_points(128, 1.0));
    let addresses: Vec<usize> = std::thread::scope(|scope| {
        let handles: Vec<_> = (0..4)
            .map(|_| {
                scope.spawn(|| {
                    let topology = hull.topology().expect("sphere hull");
                    assert!(matches!(topology, Cow::Borrowed(_)));
                    let cached: &HullTopology = &topology;
                    std::ptr::from_ref(cached) as usize
                })
            })
            .collect();
        handles
            .into_iter()
            .map(|h| h.join().expect("worker finished"))
            .collect()
    });
    assert!(
        addresses.windows(2).all(|w| w[0] == w[1]),
        "workers saw different topologies: {addresses:?}"
    );
    let topology = hull.topology().expect("sphere hull");
    let cached: &HullTopology = &topology;
    assert_eq!(std::ptr::from_ref(cached) as usize, addresses[0]);
}
