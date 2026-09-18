// Copyright 2026 COOLJAPAN OU (Team KitaSan)
// SPDX-License-Identifier: Apache-2.0

//! `NarrowPhaseDispatcher::generate_contacts` is the production narrow-phase
//! entry point. It used to classify shapes by formatting them with `Debug` (twice
//! per pair, a cost proportional to the vertex count of hulls and meshes) and to
//! bypass the registered manifold algorithms; it now classifies with
//! [`shape_type_of`] and runs the default dispatcher table.

use std::any::Any;
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

use oxiphysics_collision::narrowphase::dispatch::{ShapeType, shape_type_of};
use oxiphysics_collision::{CollisionPair, Contact, NarrowPhaseDispatcher};
use oxiphysics_core::math::{Mat3, Real, Vec3};
use oxiphysics_core::{Aabb, Transform};
use oxiphysics_geometry::{
    BoxShape, Capsule, Compound, Cone, ConvexHull, Cylinder, HeightField, RayHit, Shape, Sphere,
    Torus, TriangleMesh,
};

const PAIR: CollisionPair = CollisionPair { a: 0, b: 1 };

fn at(x: f64, y: f64, z: f64) -> Transform {
    Transform::from_position(Vec3::new(x, y, z))
}

/// A user-defined shape (so it has no dedicated `ShapeType`) that counts how often
/// it is formatted with `Debug`.
struct Probe {
    inner: Sphere,
    debug_calls: Arc<AtomicUsize>,
}

impl Probe {
    fn new(radius: f64, debug_calls: &Arc<AtomicUsize>) -> Self {
        Self {
            inner: Sphere::new(radius),
            debug_calls: Arc::clone(debug_calls),
        }
    }
}

impl std::fmt::Debug for Probe {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.debug_calls.fetch_add(1, Ordering::SeqCst);
        write!(f, "Probe")
    }
}

impl Shape for Probe {
    fn bounding_box(&self) -> Aabb {
        self.inner.bounding_box()
    }

    fn as_any(&self) -> &dyn Any {
        self
    }

    fn support_point(&self, direction: &Vec3) -> Vec3 {
        self.inner.support_point(direction)
    }

    fn volume(&self) -> Real {
        self.inner.volume()
    }

    fn center_of_mass(&self) -> Vec3 {
        self.inner.center_of_mass()
    }

    fn inertia_tensor(&self, mass: Real) -> Mat3 {
        self.inner.inertia_tensor(mass)
    }

    fn ray_cast(&self, origin: &Vec3, direction: &Vec3, max_toi: Real) -> Option<RayHit> {
        self.inner.ray_cast(origin, direction, max_toi)
    }
}

fn cube_hull(half: f64) -> ConvexHull {
    let mut vertices = Vec::new();
    for sx in [-half, half] {
        for sy in [-half, half] {
            for sz in [-half, half] {
                vertices.push(Vec3::new(sx, sy, sz));
            }
        }
    }
    ConvexHull::new(vertices)
}

#[test]
fn generate_contacts_never_formats_a_shape_with_debug() {
    let calls = Arc::new(AtomicUsize::new(0));
    let a = Probe::new(1.0, &calls);
    let b = Probe::new(1.0, &calls);

    // The counter really counts.
    let _ = format!("{a:?}");
    assert_eq!(calls.load(Ordering::SeqCst), 1);
    calls.store(0, Ordering::SeqCst);

    let manifold = NarrowPhaseDispatcher::generate_contacts(
        &a,
        &at(0.0, 0.0, 0.0),
        &b,
        &at(1.5, 0.0, 0.0),
        PAIR,
    )
    .expect("unit spheres 1.5 apart overlap by 0.5");
    assert_eq!(
        calls.load(Ordering::SeqCst),
        0,
        "classifying the pair must not format the shapes"
    );
    // The pair fell through to GJK/EPA with the B -> A convention.
    let contact = &manifold.contacts[0];
    assert!(contact.normal.x < -0.9, "normal {:?}", contact.normal);
    assert!(
        (contact.depth - 0.5).abs() < 1e-3,
        "depth {}",
        contact.depth
    );
}

#[test]
fn shape_type_of_classifies_every_dedicated_shape() {
    assert_eq!(shape_type_of(&Sphere::new(1.0)), ShapeType::Sphere);
    assert_eq!(
        shape_type_of(&BoxShape::new(Vec3::new(1.0, 1.0, 1.0))),
        ShapeType::Box
    );
    assert_eq!(shape_type_of(&Capsule::new(0.5, 1.0)), ShapeType::Capsule);
    assert_eq!(shape_type_of(&Cylinder::new(0.5, 1.0)), ShapeType::Cylinder);
    assert_eq!(shape_type_of(&Cone::new(0.5, 1.0)), ShapeType::Cone);
    assert_eq!(shape_type_of(&cube_hull(0.5)), ShapeType::ConvexHull);
    let mesh = TriangleMesh::new(
        vec![
            Vec3::new(0.0, 0.0, 0.0),
            Vec3::new(1.0, 0.0, 0.0),
            Vec3::new(0.0, 1.0, 0.0),
        ],
        vec![[0, 1, 2]],
    );
    assert_eq!(shape_type_of(&mesh), ShapeType::TriangleMesh);
    let compound = Compound::new(vec![(
        Transform::default(),
        Arc::new(Sphere::new(1.0)) as Arc<dyn Shape>,
    )]);
    assert_eq!(shape_type_of(&compound), ShapeType::Compound);
    let field = HeightField::new(vec![0.0; 4], 2, 2, 1.0, 1.0);
    assert_eq!(shape_type_of(&field), ShapeType::HeightField);
    // No dedicated type: user-defined and torus shapes are `Custom(0)`.
    assert_eq!(shape_type_of(&Torus::new(1.0, 0.25)), ShapeType::Custom(0));
    let calls = Arc::new(AtomicUsize::new(0));
    assert_eq!(
        shape_type_of(&Probe::new(1.0, &calls)),
        ShapeType::Custom(0)
    );
}

#[test]
fn a_hull_resting_on_a_box_gets_the_four_point_face_manifold() {
    // The registered (Box, ConvexHull) algorithm clips the faces; the old
    // `generate_contacts` returned the single EPA witness for this pair.
    let hull = cube_hull(0.5);
    let slab = BoxShape::new(Vec3::new(0.5, 0.5, 0.5));
    let manifold = NarrowPhaseDispatcher::generate_contacts(
        &hull,
        &at(0.0, 0.95, 0.0),
        &slab,
        &at(0.0, 0.0, 0.0),
        PAIR,
    )
    .expect("the cube overlaps the box by 0.05");
    assert_eq!(manifold.contacts.len(), 4, "face-on-face patch");
    for c in &manifold.contacts {
        assert!(c.normal.y > 0.99, "B -> A normal, got {:?}", c.normal);
        assert!((c.depth - 0.05).abs() < 1e-4, "depth {}", c.depth);
        assert!((c.point_a.y - 0.45).abs() < 1e-6, "A's bottom face");
        assert!((c.point_b.y - 0.5).abs() < 1e-6, "B's top face");
    }
    // The other argument order flips the normal and keeps the patch.
    let flipped = NarrowPhaseDispatcher::generate_contacts(
        &slab,
        &at(0.0, 0.0, 0.0),
        &hull,
        &at(0.0, 0.95, 0.0),
        PAIR,
    )
    .expect("same overlap");
    assert_eq!(flipped.contacts.len(), 4);
    assert!(flipped.contacts.iter().all(|c| c.normal.y < -0.99));
}

#[test]
fn two_hulls_get_a_multi_point_manifold() {
    let hull = cube_hull(0.5);
    let manifold = NarrowPhaseDispatcher::generate_contacts(
        &hull,
        &at(0.0, 0.95, 0.0),
        &hull,
        &at(0.0, 0.0, 0.0),
        PAIR,
    )
    .expect("stacked cubes overlap");
    assert_eq!(manifold.contacts.len(), 4);
    assert!(manifold.contacts.iter().all(|c| c.normal.y > 0.99));
}

/// `count` points spread evenly over a sphere (Fibonacci lattice): the vertex set
/// of a round convex hull.
fn sphere_hull(count: usize, radius: f64) -> ConvexHull {
    let golden_angle = std::f64::consts::PI * (3.0 - 5.0_f64.sqrt());
    let vertices = (0..count)
        .map(|i| {
            let y = 1.0 - 2.0 * (i as f64 + 0.5) / count as f64;
            let ring = (1.0 - y * y).sqrt();
            let phi = golden_angle * i as f64;
            Vec3::new(
                radius * ring * phi.cos(),
                radius * y,
                radius * ring * phi.sin(),
            )
        })
        .collect();
    ConvexHull::new(vertices)
}

#[test]
fn thousand_vertex_hulls_get_a_b_to_a_manifold_of_the_true_depth() {
    // Two round hulls of radius 0.5 whose centres are 0.9 apart overlap by 0.1
    // (less the facet sag of the 1000-point approximation).
    let upper = sphere_hull(1000, 0.5);
    let lower = sphere_hull(1000, 0.5);
    let manifold = NarrowPhaseDispatcher::generate_contacts(
        &upper,
        &at(0.0, 0.9, 0.0),
        &lower,
        &at(0.0, 0.0, 0.0),
        PAIR,
    )
    .expect("the hulls overlap");
    assert!(!manifold.contacts.is_empty());
    for c in &manifold.contacts {
        assert!(c.normal.y > 0.9, "B -> A normal, got {:?}", c.normal);
        assert!(
            (c.depth - 0.1).abs() < 0.03,
            "depth {} should be about 0.1",
            c.depth
        );
    }
    // Separated hulls have no contact.
    assert!(
        NarrowPhaseDispatcher::generate_contacts(
            &upper,
            &at(0.0, 1.2, 0.0),
            &lower,
            &at(0.0, 0.0, 0.0),
            PAIR,
        )
        .is_none()
    );
}

fn make(kind: ShapeType) -> Box<dyn Shape> {
    match kind {
        ShapeType::Sphere => Box::new(Sphere::new(0.5)),
        ShapeType::Box => Box::new(BoxShape::new(Vec3::new(0.5, 0.5, 0.5))),
        ShapeType::Capsule => Box::new(Capsule::new(0.25, 0.5)),
        ShapeType::Cylinder => Box::new(Cylinder::new(0.5, 0.5)),
        ShapeType::Cone => Box::new(Cone::new(0.5, 0.5)),
        _ => Box::new(cube_hull(0.5)),
    }
}

fn same_contact(x: &Contact, y: &Contact) -> bool {
    (x.normal - y.normal).norm() < 1e-12
        && (x.point_a - y.point_a).norm() < 1e-12
        && (x.point_b - y.point_b).norm() < 1e-12
        && (x.depth - y.depth).abs() < 1e-12
}

#[test]
fn generate_contacts_is_the_default_dispatcher_for_every_primitive_pair() {
    let kinds = [
        ShapeType::Sphere,
        ShapeType::Box,
        ShapeType::Capsule,
        ShapeType::Cylinder,
        ShapeType::Cone,
        ShapeType::ConvexHull,
    ];
    let dispatcher = NarrowPhaseDispatcher::default();
    for &kind_a in &kinds {
        for &kind_b in &kinds {
            let a = make(kind_a);
            let b = make(kind_b);
            // A a little above B so that every pair overlaps by a few centimetres.
            let ta = at(0.03, 0.9, -0.02);
            let tb = at(0.0, 0.0, 0.0);
            let direct =
                NarrowPhaseDispatcher::generate_contacts(a.as_ref(), &ta, b.as_ref(), &tb, PAIR);
            let routed = dispatcher
                .dispatch(
                    a.as_ref(),
                    shape_type_of(a.as_ref()),
                    &ta,
                    b.as_ref(),
                    shape_type_of(b.as_ref()),
                    &tb,
                    PAIR,
                )
                .manifold;
            match (direct, routed) {
                (None, None) => {}
                (Some(x), Some(y)) => {
                    assert_eq!(
                        x.contacts.len(),
                        y.contacts.len(),
                        "{kind_a:?}/{kind_b:?}: contact count"
                    );
                    for (cx, cy) in x.contacts.iter().zip(&y.contacts) {
                        assert!(same_contact(cx, cy), "{kind_a:?}/{kind_b:?}");
                    }
                }
                _ => panic!("{kind_a:?}/{kind_b:?}: one route found a contact, the other did not"),
            }
        }
    }
}
