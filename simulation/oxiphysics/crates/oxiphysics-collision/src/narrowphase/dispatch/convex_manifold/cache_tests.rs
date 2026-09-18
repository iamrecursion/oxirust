// Copyright 2026 COOLJAPAN OU (Team KitaSan)
// SPDX-License-Identifier: Apache-2.0

//! The manifold of a hull pair built from the topology cached on the shapes
//! ([`convex_shape_manifold`]) is the manifold of the world-space extraction
//! ([`convex_hull_manifold_from_vertices`]): identical contacts at the identity
//! pose, the same contacts within rounding under a rigid motion, and a cache that
//! is built once per hull and notices edits of the public vertex list.

use oxiphysics_geometry::Sphere;

use super::super::functions::gjk_epa;
use super::*;

const PAIR: CollisionPair = CollisionPair { a: 0, b: 1 };

fn hull_of(points: &[[f64; 3]]) -> ConvexHull {
    ConvexHull::new(points.iter().map(|&p| Vec3::from(p)).collect())
}

fn shifted(points: &[[f64; 3]], offset: [f64; 3]) -> Vec<[f64; 3]> {
    points.iter().map(|&p| add3(p, offset)).collect()
}

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

/// The 27 points of a 3x3x3 lattice (coplanar and collinear points, one interior
/// point).
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

/// `count` points spread evenly over a sphere (Fibonacci lattice).
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

/// Deterministic pseudo-random points in the cube `[-1, 1]^3`.
fn random_points(count: usize, seed: u64) -> Vec<[f64; 3]> {
    let mut state = seed;
    let mut next = move || {
        state = state
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        2.0 * ((state >> 11) as f64 / (1u64 << 53) as f64) - 1.0
    };
    (0..count).map(|_| [next(), next(), next()]).collect()
}

/// A cube rotated by `angle` about the unit `axis`, as a vertex list (so that it
/// can be posed with the identity transform).
fn tilted_cube(axis: [f64; 3], angle: f64, offset: [f64; 3]) -> Vec<[f64; 3]> {
    let rotation = Transform::from_axis_angle(Vec3::zeros(), Vec3::from(axis), angle);
    cube_points(0.5)
        .iter()
        .map(|&p| {
            let q = rotation.transform_point(&Vec3::from(p));
            add3([q.x, q.y, q.z], offset)
        })
        .collect()
}

fn identity() -> Transform {
    Transform::from_position(Vec3::zeros())
}

/// A rotation of 0.7 rad about `(1, 2, 3)` plus a translation.
fn motion() -> Transform {
    Transform::from_axis_angle(Vec3::new(3.0, -2.0, 1.5), Vec3::new(1.0, 2.0, 3.0), 0.7)
}

fn world_vertices(hull: &ConvexHull, transform: &Transform) -> Vec<[f64; 3]> {
    hull.vertices
        .iter()
        .map(|v| {
            let p = transform.transform_point(v);
            [p.x, p.y, p.z]
        })
        .collect()
}

fn separation(normal: [f64; 3], depth: f64) -> ConvexSeparation {
    ConvexSeparation {
        normal,
        depth,
        point_a: [0.1, 0.2, 0.3],
        point_b: [0.1, 0.2, 0.3],
    }
}

/// Whether `b` is `a` read from another starting vertex.
fn same_cycle(a: &[usize], b: &[usize]) -> bool {
    a.len() == b.len()
        && (0..a.len().max(1)).any(|s| (0..a.len()).all(|i| a[(i + s) % a.len()] == b[i]))
}

fn sorted_pairs(mut pairs: Vec<[usize; 2]>) -> Vec<[usize; 2]> {
    for pair in &mut pairs {
        pair.sort_unstable();
    }
    pairs.sort_unstable();
    pairs
}

/// Contact lists equal field by field (exactly).
fn assert_identical(got: &ContactManifold, want: &ContactManifold, label: &str) {
    assert_eq!(got.contacts.len(), want.contacts.len(), "{label}: count");
    for (i, (g, w)) in got.contacts.iter().zip(&want.contacts).enumerate() {
        assert_eq!(g.point_a, w.point_a, "{label}: contact {i} point_a");
        assert_eq!(g.point_b, w.point_b, "{label}: contact {i} point_b");
        assert_eq!(g.normal, w.normal, "{label}: contact {i} normal");
        assert_eq!(g.depth, w.depth, "{label}: contact {i} depth");
    }
}

/// Contact lists equal within `tol`, in any order.
fn assert_same_contacts(got: &ContactManifold, want: &ContactManifold, tol: f64, label: &str) {
    assert_eq!(got.contacts.len(), want.contacts.len(), "{label}: count");
    for g in &got.contacts {
        let matched = want.contacts.iter().any(|w| {
            (g.point_a - w.point_a).norm() < tol
                && (g.point_b - w.point_b).norm() < tol
                && (g.normal - w.normal).norm() < tol
                && (g.depth - w.depth).abs() < tol
        });
        assert!(matched, "{label}: no counterpart for {g:?} in {want:?}");
    }
}

#[test]
fn posed_hull_equals_the_world_space_extraction_at_the_identity_pose() {
    let sets: Vec<(&str, Vec<[f64; 3]>)> = vec![
        ("cube", cube_points(0.5)),
        ("hex prism", hex_prism_points(1.0, 0.5)),
        ("lattice", lattice_points()),
        ("sphere", sphere_points(256, 0.5)),
        ("random cloud", random_points(14, 0xC0FF_EE01)),
    ];
    for (name, points) in sets {
        let hull = hull_of(&points);
        let posed = PosedHull::new(&hull, &identity()).expect(name);
        let poly = Polyhedron::from_hull_vertices(&world_vertices(&hull, &identity())).expect(name);
        let (cached, direct) = (posed.view(), poly.view());
        assert_eq!(cached.vertices, direct.vertices, "{name}: vertices");
        assert_eq!(cached.faces, direct.faces, "{name}: faces");
        assert_eq!(cached.normals, direct.normals, "{name}: normals");
        assert_eq!(
            cached.edge_segments(),
            direct.edge_segments(),
            "{name}: edges"
        );
    }
}

#[test]
fn posed_hull_follows_a_rigid_motion() {
    let sets: Vec<(&str, Vec<[f64; 3]>)> = vec![
        ("cube", cube_points(0.5)),
        ("hex prism", hex_prism_points(1.0, 0.5)),
        ("sphere", sphere_points(256, 0.5)),
        ("random cloud", random_points(14, 0xC0FF_EE02)),
    ];
    let transform = motion();
    for (name, points) in sets {
        let hull = hull_of(&points);
        let posed = PosedHull::new(&hull, &transform).expect(name);
        let poly = Polyhedron::from_hull_vertices(&world_vertices(&hull, &transform)).expect(name);
        let (cached, direct) = (posed.view(), poly.view());
        assert_eq!(cached.vertices.len(), direct.vertices.len(), "{name}");
        for (c, d) in cached.vertices.iter().zip(direct.vertices) {
            assert!(norm3(sub3(*c, *d)) < 1e-9, "{name}: vertex {c:?} vs {d:?}");
        }
        assert_eq!(cached.faces.len(), direct.faces.len(), "{name}: faces");
        for (c, d) in cached.faces.iter().zip(direct.faces) {
            assert!(same_cycle(c, d), "{name}: loop {c:?} vs {d:?}");
        }
        for (c, d) in cached.normals.iter().zip(direct.normals) {
            assert!(norm3(sub3(*c, *d)) < 1e-9, "{name}: normal {c:?} vs {d:?}");
        }
        let cached_edges = cached
            .edges
            .map(<[[usize; 2]]>::to_vec)
            .expect("cached edges");
        let direct_edges: Vec<[usize; 2]> = {
            let mut seen = HashSet::new();
            let mut edges = Vec::new();
            for face in direct.faces {
                for (i, &a) in face.iter().enumerate() {
                    let b = face[(i + 1) % face.len()];
                    if seen.insert((a.min(b), a.max(b))) {
                        edges.push([a, b]);
                    }
                }
            }
            edges
        };
        assert_eq!(
            sorted_pairs(cached_edges),
            sorted_pairs(direct_edges),
            "{name}: edge set"
        );
    }
}

#[test]
fn polyhedron_from_a_hull_shape_matches_the_world_space_extraction() {
    let hull = hull_of(&hex_prism_points(1.0, 0.5));
    let from_shape = Polyhedron::from_hull(&hull, &identity()).expect("hex prism");
    let from_vertices =
        Polyhedron::from_hull_vertices(&world_vertices(&hull, &identity())).expect("hex prism");
    assert_eq!(from_shape.vertices, from_vertices.vertices);
    assert_eq!(from_shape.faces, from_vertices.faces);
    assert_eq!(from_shape.normals, from_vertices.normals);

    let moved = Polyhedron::from_hull(&hull, &motion()).expect("hex prism");
    let moved_vertices =
        Polyhedron::from_hull_vertices(&world_vertices(&hull, &motion())).expect("hex prism");
    assert_eq!(moved.faces.len(), moved_vertices.faces.len());
    for (a, b) in moved.vertices.iter().zip(&moved_vertices.vertices) {
        assert!(norm3(sub3(*a, *b)) < 1e-9, "vertex {a:?} vs {b:?}");
    }

    let flat = hull_of(&[
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [1.0, 0.0, 1.0],
        [0.0, 0.0, 1.0],
    ]);
    assert!(Polyhedron::from_hull(&flat, &identity()).is_none());
}

#[test]
fn posed_hulls_share_the_cached_topology() {
    let hull = hull_of(&sphere_points(64, 0.5));
    let one = PosedHull::new(&hull, &identity()).expect("sphere hull");
    let two = PosedHull::new(&hull, &motion()).expect("sphere hull");
    assert!(matches!(one.topology, Cow::Borrowed(_)));
    assert!(matches!(two.topology, Cow::Borrowed(_)));
    let (a, b) = (one.view(), two.view());
    assert!(
        std::ptr::eq(a.faces, b.faces),
        "faces are shared, not copied"
    );
    assert!(
        std::ptr::eq(
            a.edges.expect("cached edges"),
            b.edges.expect("cached edges")
        ),
        "edges are shared, not copied"
    );
    // The posed vertices and normals are per query.
    assert!(!std::ptr::eq(a.vertices, b.vertices));
    assert_ne!(a.vertices, b.vertices);
}

/// The face-contact normal that points B -> A when face `k` of A is the
/// reference face.
fn face_contact_normal(hull: &ConvexHull, transform: &Transform, k: usize) -> [f64; 3] {
    let posed = PosedHull::new(hull, transform).expect("hull");
    let n = posed.view().normals[k];
    [-n[0], -n[1], -n[2]]
}

/// A named pair of hull vertex lists (A above B) and the separation axis (B -> A) their
/// contact is built for.
type PairCase = (&'static str, Vec<[f64; 3]>, Vec<[f64; 3]>, [f64; 3]);

#[test]
fn cached_manifold_equals_the_world_space_manifold_at_the_identity_pose() {
    let tall = |r: f64| hex_prism_points(r, 0.5);
    let pairs: Vec<PairCase> = vec![
        (
            "cube on cube",
            shifted(&cube_points(0.5), [0.0, 0.99, 0.0]),
            cube_points(0.5),
            [0.0, 1.0, 0.0],
        ),
        (
            "hex prism on hex prism",
            shifted(&tall(1.0), [0.0, 0.99, 0.0]),
            tall(1.0),
            [0.0, 1.0, 0.0],
        ),
        (
            "cube on hex prism",
            shifted(&cube_points(0.4), [0.1, 0.9, 0.05]),
            tall(1.0),
            [0.0, 1.0, 0.0],
        ),
        (
            "sphere on sphere",
            shifted(&sphere_points(256, 0.5), [0.0, 0.9, 0.0]),
            sphere_points(256, 0.5),
            [0.0, 1.0, 0.0],
        ),
        (
            "lattice on lattice",
            shifted(&lattice_points(), [0.02, 0.98, 0.01]),
            lattice_points(),
            [0.0, 1.0, 0.0],
        ),
        (
            "crossed cubes (edge-edge)",
            tilted_cube(
                [0.0, 0.0, 1.0],
                std::f64::consts::FRAC_PI_4,
                [0.0, 1.2, 0.0],
            ),
            tilted_cube(
                [1.0, 0.0, 0.0],
                std::f64::consts::FRAC_PI_4,
                [0.0, 0.0, 0.0],
            ),
            [0.0, 1.0, 0.0],
        ),
        (
            "random clouds",
            shifted(&random_points(14, 0xC0FF_EE03), [0.0, 1.5, 0.0]),
            random_points(14, 0xC0FF_EE04),
            normalize3([0.3, 0.9, -0.2]),
        ),
    ];
    for (name, points_a, points_b, normal) in pairs {
        let (hull_a, hull_b) = (hull_of(&points_a), hull_of(&points_b));
        let sep = separation(normal, 0.01);
        let cached = convex_shape_manifold(&hull_a, &identity(), &hull_b, &identity(), sep, PAIR)
            .expect("hulls are polyhedral");
        let direct = convex_hull_manifold_from_vertices(
            &world_vertices(&hull_a, &identity()),
            &world_vertices(&hull_b, &identity()),
            sep,
            PAIR,
        );
        assert!(!direct.is_empty(), "{name}: the fixture produces contacts");
        assert_identical(&cached, &direct, name);
    }
}

#[test]
fn both_manifold_paths_are_exercised() {
    // Random hulls with a separation axis that is a face normal of A (a face
    // contact) and one that is oblique to every face of both hulls (an edge-edge
    // contact): the cached path agrees with the world-space path on each of them.
    let (mut faces, mut edges) = (0, 0);
    for seed in 0..6_u64 {
        let hull_a = hull_of(&random_points(14, 0xBEEF_0000 + seed));
        let hull_b = hull_of(&random_points(14, 0xFEED_0000 + seed));
        let (ta, tb) = (
            identity(),
            Transform::from_position(Vec3::new(0.1, 1.2, -0.2)),
        );
        let poly_a = Polyhedron::from_hull_vertices(&world_vertices(&hull_a, &ta));
        let poly_b = Polyhedron::from_hull_vertices(&world_vertices(&hull_b, &tb));
        let (Some(poly_a), Some(poly_b)) = (poly_a, poly_b) else {
            panic!("random hulls span a volume");
        };
        let aligned = face_contact_normal(&hull_a, &ta, 0);
        let oblique = sphere_points(200, 1.0)
            .into_iter()
            .map(normalize3)
            .find(|&axis| !is_face_contact(poly_a.view(), poly_b.view(), axis))
            .expect("an axis oblique to every face of both hulls exists");
        for normal in [aligned, oblique] {
            let sep = separation(normal, 0.02);
            let cached = convex_shape_manifold(&hull_a, &ta, &hull_b, &tb, sep, PAIR)
                .expect("hulls are polyhedral");
            let direct = convex_hull_manifold_from_vertices(
                &world_vertices(&hull_a, &ta),
                &world_vertices(&hull_b, &tb),
                sep,
                PAIR,
            );
            assert_same_contacts(&cached, &direct, 1e-9, "random hulls");
            if is_face_contact(poly_a.view(), poly_b.view(), normal) {
                faces += 1;
            } else {
                edges += 1;
            }
        }
    }
    assert_eq!(faces, 6, "every aligned axis is a face contact");
    assert_eq!(edges, 6, "every oblique axis is an edge-edge contact");
}

#[test]
fn cached_manifold_matches_under_a_rigid_motion() {
    // Asymmetric hulls: no two clip points tie, so the manifold does not depend on
    // the vertex a face loop starts at and agrees with the world-space path.
    let hull_a = hull_of(&random_points(18, 0xA11C_E001));
    let hull_b = hull_of(&random_points(18, 0xA11C_E002));
    let ta = motion();
    let tb = Transform::from_axis_angle(Vec3::new(3.2, 0.2, 1.4), Vec3::new(-1.0, 0.5, 2.0), 1.1);
    for k in 0..4 {
        let normal = face_contact_normal(&hull_a, &ta, k);
        let sep = separation(normal, 0.03);
        let cached = convex_shape_manifold(&hull_a, &ta, &hull_b, &tb, sep, PAIR)
            .expect("hulls are polyhedral");
        let direct = convex_hull_manifold_from_vertices(
            &world_vertices(&hull_a, &ta),
            &world_vertices(&hull_b, &tb),
            sep,
            PAIR,
        );
        assert_same_contacts(&cached, &direct, 1e-9, "rigid motion");
    }
}

#[test]
fn box_and_hull_pair_matches_the_world_space_manifold() {
    let cube = BoxShape::new(Vec3::new(0.5, 0.5, 0.5));
    let hull = hull_of(&shifted(&hex_prism_points(0.6, 0.5), [0.0, -0.99, 0.0]));
    let corners: Vec<[f64; 3]> = cube
        .vertex_list()
        .iter()
        .map(|&v| {
            let p = identity().transform_point(&Vec3::from(v));
            [p.x, p.y, p.z]
        })
        .collect();
    let sep = separation([0.0, 1.0, 0.0], 0.01);
    let cached = convex_shape_manifold(&cube, &identity(), &hull, &identity(), sep, PAIR)
        .expect("box and hull are polyhedral");
    let direct = convex_hull_manifold_from_vertices(
        &corners,
        &world_vertices(&hull, &identity()),
        sep,
        PAIR,
    );
    assert!(!direct.is_empty());
    assert_identical(&cached, &direct, "box on hull");
    let swapped = convex_shape_manifold(&hull, &identity(), &cube, &identity(), sep, PAIR)
        .expect("hull and box are polyhedral");
    let swapped_direct = convex_hull_manifold_from_vertices(
        &world_vertices(&hull, &identity()),
        &corners,
        sep,
        PAIR,
    );
    assert_identical(&swapped, &swapped_direct, "hull on box");
}

#[test]
fn a_hull_resting_on_a_box_gets_the_four_corner_patch_for_either_argument_order() {
    // The seed is stated explicitly: the patch below depends on the cached topology
    // and the clipping only, not on the GJK/EPA solver that seeds the dispatcher.
    let hull = hull_of(&cube_points(0.5));
    let slab = BoxShape::new(Vec3::new(0.5, 0.5, 0.5));
    let hull_pose = Transform::from_position(Vec3::new(0.0, 0.95, 0.0));
    let manifold = convex_shape_manifold(
        &hull,
        &hull_pose,
        &slab,
        &identity(),
        separation([0.0, 1.0, 0.0], 0.05),
        PAIR,
    )
    .expect("a hull and a box are polyhedral");
    assert_eq!(manifold.contacts.len(), 4, "face-on-face patch");
    for contact in &manifold.contacts {
        assert!(
            contact.normal.y > 0.99,
            "B -> A normal, got {:?}",
            contact.normal
        );
        assert!(
            (contact.depth - 0.05).abs() < 1e-9,
            "depth {}",
            contact.depth
        );
        assert!((contact.point_a.y - 0.45).abs() < 1e-9, "A's bottom face");
        assert!((contact.point_b.y - 0.5).abs() < 1e-9, "B's top face");
    }
    // The four contacts are the four distinct corners of the 1 x 1 face.
    for (sx, sz) in [(-0.5, -0.5), (-0.5, 0.5), (0.5, -0.5), (0.5, 0.5)] {
        let corners = manifold
            .contacts
            .iter()
            .filter(|c| (c.point_a.x - sx).abs() < 1e-9 && (c.point_a.z - sz).abs() < 1e-9)
            .count();
        assert_eq!(corners, 1, "corner ({sx}, {sz})");
    }
    // The other argument order flips the normal and keeps the patch.
    let flipped = convex_shape_manifold(
        &slab,
        &identity(),
        &hull,
        &hull_pose,
        separation([0.0, -1.0, 0.0], 0.05),
        PAIR,
    )
    .expect("a box and a hull are polyhedral");
    assert_eq!(flipped.contacts.len(), 4, "face-on-face patch");
    for contact in &flipped.contacts {
        assert!(
            contact.normal.y < -0.99,
            "B -> A normal, got {:?}",
            contact.normal
        );
        assert!(
            (contact.depth - 0.05).abs() < 1e-9,
            "depth {}",
            contact.depth
        );
        assert!((contact.point_a.y - 0.5).abs() < 1e-9, "A's top face");
        assert!((contact.point_b.y - 0.45).abs() < 1e-9, "B's bottom face");
    }
}

#[test]
fn edited_hull_vertices_are_picked_up_by_the_next_query() {
    let mut hull_a = hull_of(&shifted(&cube_points(0.5), [0.0, 0.99, 0.0]));
    let hull_b = hull_of(&cube_points(0.5));
    let sep = separation([0.0, 1.0, 0.0], 0.01);
    let before = convex_shape_manifold(&hull_a, &identity(), &hull_b, &identity(), sep, PAIR)
        .expect("polyhedral");
    assert_eq!(before.contacts.len(), 4);
    // Shrink the cube to a quarter of the footprint: the reference face of the
    // contact becomes smaller than the incident one, so a stale cache would show
    // in the contact points.
    hull_a.vertices = shifted(&cube_points(0.5), [0.0, 0.99, 0.0])
        .into_iter()
        .map(|[x, y, z]| Vec3::new(0.5 * x, y, 0.5 * z))
        .collect();
    let after = convex_shape_manifold(&hull_a, &identity(), &hull_b, &identity(), sep, PAIR)
        .expect("polyhedral");
    let direct = convex_hull_manifold_from_vertices(
        &world_vertices(&hull_a, &identity()),
        &world_vertices(&hull_b, &identity()),
        sep,
        PAIR,
    );
    assert_identical(&after, &direct, "after the edit");
    assert!(
        before
            .contacts
            .iter()
            .zip(&after.contacts)
            .any(|(b, a)| (b.point_a - a.point_a).norm() > 1e-6)
            || before.contacts.len() != after.contacts.len(),
        "the edit changed the manifold"
    );
}

#[test]
fn a_hull_without_volume_keeps_the_epa_witness() {
    let flat = hull_of(&[
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [1.0, 0.0, 1.0],
        [0.0, 0.0, 1.0],
    ]);
    let cube = hull_of(&cube_points(0.5));
    let sep = separation([0.0, 1.0, 0.0], 0.25);
    for (a, b) in [(&flat, &cube), (&cube, &flat), (&flat, &flat)] {
        let manifold =
            convex_shape_manifold(a, &identity(), b, &identity(), sep, PAIR).expect("polyhedral");
        assert_eq!(manifold.contacts.len(), 1);
        let contact = &manifold.contacts[0];
        assert_eq!(contact.point_a, Vec3::from(sep.point_a));
        assert_eq!(contact.point_b, Vec3::from(sep.point_b));
        assert_eq!(contact.normal, Vec3::new(0.0, 1.0, 0.0));
        assert_eq!(contact.depth, 0.25);
    }
}

#[test]
fn non_polyhedral_shapes_are_left_to_the_epa_contact() {
    let cube = hull_of(&cube_points(0.5));
    let ball = Sphere::new(0.5);
    let sep = separation([0.0, 1.0, 0.0], 0.01);
    assert!(convex_shape_manifold(&ball, &identity(), &cube, &identity(), sep, PAIR).is_none());
    assert!(convex_shape_manifold(&cube, &identity(), &ball, &identity(), sep, PAIR).is_none());
    assert!(convex_shape_manifold(&ball, &identity(), &ball, &identity(), sep, PAIR).is_none());
}

#[test]
fn the_dispatcher_returns_the_world_space_manifold_of_its_epa_seed() {
    let hull_a = hull_of(&shifted(&hex_prism_points(1.0, 0.5), [0.0, 0.99, 0.0]));
    let hull_b = hull_of(&hex_prism_points(1.0, 0.5));
    let pair = CollisionPair::new(0, 1);
    let seed = gjk_epa(&hull_a, &identity(), &hull_b, &identity(), pair).expect("overlap");
    let epa = &seed.contacts[0];
    let sep = ConvexSeparation {
        normal: [epa.normal.x, epa.normal.y, epa.normal.z],
        depth: epa.depth,
        point_a: [epa.point_a.x, epa.point_a.y, epa.point_a.z],
        point_b: [epa.point_b.x, epa.point_b.y, epa.point_b.z],
    };
    let direct = convex_hull_manifold_from_vertices(
        &world_vertices(&hull_a, &identity()),
        &world_vertices(&hull_b, &identity()),
        sep,
        pair,
    );
    assert!(!direct.is_empty());
    // Repeated queries (the second and later ones are served from the cache) agree
    // with each other and with the world-space extraction.
    for _ in 0..3 {
        let result = super::super::functions::convex_manifold_dispatch(
            &hull_a,
            &identity(),
            &hull_b,
            &identity(),
            pair,
        );
        let manifold = result.manifold.expect("overlapping hulls give a manifold");
        assert_identical(&manifold, &direct, "dispatch");
    }
}
