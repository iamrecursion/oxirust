// Copyright 2026 COOLJAPAN OU (Team KitaSan)
// SPDX-License-Identifier: Apache-2.0

//! Benchmarks of `NarrowPhaseDispatcher::generate_contacts`, the production
//! narrow-phase entry point, on convex hulls with many vertices.
//!
//! It used to classify every shape by formatting it with `Debug` (twice per pair)
//! and comparing the text, a cost proportional to the vertex count, and then
//! returned the single GJK/EPA contact for a convex hull pair. The groups below
//! measure the pieces of that old path (`debug_format`, paid twice per pair, and
//! `single_contact_gjk_epa`) next to the current one (`hull_pair`):
//! `generate_contacts` classifies with `shape_type_of` (a few `TypeId`
//! comparisons) and runs the registered pair algorithm, which builds the full
//! face-clip manifold of the two hulls.
//!
//! The polygon faces and edges of a hull do not depend on its pose, so they are
//! built once per hull and cached on the shape (`ConvexHull::topology`); a query
//! only poses the cached vertices and normals. The `manifold_*` groups isolate the
//! manifold construction (no GJK/EPA) for the same separation seed:
//! `manifold_cached_topology` is the production path (`convex_shape_manifold`),
//! `manifold_world_space_extraction` is the former per-query path that extracted
//! the faces from the world-space vertices of both hulls on every call.

use std::hint::black_box;

use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};
use oxiphysics_collision::narrowphase::epa::Epa;
use oxiphysics_collision::narrowphase::gjk::Gjk;
use oxiphysics_collision::narrowphase::{
    ConvexSeparation, convex_hull_manifold_from_vertices, convex_shape_manifold,
};
use oxiphysics_collision::{CollisionPair, NarrowPhaseDispatcher};
use oxiphysics_core::Transform;
use oxiphysics_core::math::Vec3;
use oxiphysics_geometry::{BoxShape, ConvexHull, Sphere};

/// `count` points spread evenly over a sphere of the given radius (Fibonacci
/// lattice), the vertex set of a round convex hull.
fn sphere_points(count: usize, radius: f64) -> Vec<Vec3> {
    let golden_angle = std::f64::consts::PI * (3.0 - 5.0_f64.sqrt());
    (0..count)
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
        .collect()
}

fn bench_hull_pairs(c: &mut Criterion) {
    let pair = CollisionPair::new(0, 1);
    let upper = Transform::from_position(Vec3::new(0.0, 0.9, 0.0));
    let lower = Transform::from_position(Vec3::zeros());

    let mut group = c.benchmark_group("generate_contacts/hull_pair");
    for &vertices in &[64_usize, 256, 1000] {
        let hull_a = ConvexHull::new(sphere_points(vertices, 0.5));
        let hull_b = ConvexHull::new(sphere_points(vertices, 0.5));
        group.bench_with_input(BenchmarkId::from_parameter(vertices), &vertices, |b, _| {
            b.iter(|| {
                NarrowPhaseDispatcher::generate_contacts(
                    black_box(&hull_a),
                    &upper,
                    black_box(&hull_b),
                    &lower,
                    pair,
                )
            })
        });
    }
    group.finish();

    let mut group = c.benchmark_group("generate_contacts/single_contact_gjk_epa");
    for &vertices in &[64_usize, 256, 1000] {
        let hull_a = ConvexHull::new(sphere_points(vertices, 0.5));
        let hull_b = ConvexHull::new(sphere_points(vertices, 0.5));
        group.bench_with_input(BenchmarkId::from_parameter(vertices), &vertices, |b, _| {
            b.iter(|| {
                Gjk::overlap_simplex(black_box(&hull_a), &upper, black_box(&hull_b), &lower)
                    .and_then(|simplex| {
                        Epa::penetration_depth(&hull_a, &upper, &hull_b, &lower, &simplex)
                    })
            })
        });
    }
    group.finish();

    let mut group = c.benchmark_group("generate_contacts/debug_format");
    for &vertices in &[64_usize, 256, 1000] {
        let hull = ConvexHull::new(sphere_points(vertices, 0.5));
        group.bench_with_input(BenchmarkId::from_parameter(vertices), &vertices, |b, _| {
            b.iter(|| format!("{:?}", black_box(&hull)).len())
        });
    }
    group.finish();
}

/// World-space vertices of `hull` posed with `transform`.
fn world_vertices(hull: &ConvexHull, transform: &Transform) -> Vec<[f64; 3]> {
    hull.vertices
        .iter()
        .map(|v| {
            let p = transform.transform_point(v);
            [p.x, p.y, p.z]
        })
        .collect()
}

fn bench_manifold_topology(c: &mut Criterion) {
    let pair = CollisionPair::new(0, 1);
    let upper = Transform::from_position(Vec3::new(0.0, 0.9, 0.0));
    let lower = Transform::from_position(Vec3::zeros());

    // The hull pairs and their separation seed, as the dispatcher takes it from
    // GJK/EPA.
    let fixtures: Vec<(usize, ConvexHull, ConvexHull, ConvexSeparation)> = [64_usize, 256, 1000]
        .into_iter()
        .filter_map(|vertices| {
            let hull_a = ConvexHull::new(sphere_points(vertices, 0.5));
            let hull_b = ConvexHull::new(sphere_points(vertices, 0.5));
            let contact =
                Gjk::overlap_simplex(&hull_a, &upper, &hull_b, &lower).and_then(|simplex| {
                    Epa::penetration_depth(&hull_a, &upper, &hull_b, &lower, &simplex)
                })?;
            let sep = ConvexSeparation {
                normal: [contact.normal.x, contact.normal.y, contact.normal.z],
                depth: contact.depth,
                point_a: [contact.point_a.x, contact.point_a.y, contact.point_a.z],
                point_b: [contact.point_b.x, contact.point_b.y, contact.point_b.z],
            };
            Some((vertices, hull_a, hull_b, sep))
        })
        .collect();

    let mut group = c.benchmark_group("generate_contacts/manifold_cached_topology");
    for (vertices, hull_a, hull_b, sep) in &fixtures {
        group.bench_with_input(BenchmarkId::from_parameter(vertices), vertices, |b, _| {
            b.iter(|| {
                convex_shape_manifold(
                    black_box(hull_a),
                    &upper,
                    black_box(hull_b),
                    &lower,
                    *sep,
                    pair,
                )
            })
        });
    }
    group.finish();

    let mut group = c.benchmark_group("generate_contacts/manifold_world_space_extraction");
    for (vertices, hull_a, hull_b, sep) in &fixtures {
        group.bench_with_input(BenchmarkId::from_parameter(vertices), vertices, |b, _| {
            b.iter(|| {
                let world_a = world_vertices(black_box(hull_a), &upper);
                let world_b = world_vertices(black_box(hull_b), &lower);
                convex_hull_manifold_from_vertices(&world_a, &world_b, *sep, pair)
            })
        });
    }
    group.finish();
}

fn bench_primitive_pairs(c: &mut Criterion) {
    let pair = CollisionPair::new(0, 1);
    let upper = Transform::from_position(Vec3::new(0.0, 0.9, 0.0));
    let lower = Transform::from_position(Vec3::zeros());
    let sphere = Sphere::new(0.5);
    let cube = BoxShape::new(Vec3::new(0.5, 0.5, 0.5));

    let mut group = c.benchmark_group("generate_contacts/primitives");
    group.bench_function("sphere_sphere", |b| {
        b.iter(|| {
            NarrowPhaseDispatcher::generate_contacts(
                black_box(&sphere),
                &upper,
                black_box(&sphere),
                &lower,
                pair,
            )
        })
    });
    group.bench_function("box_box", |b| {
        b.iter(|| {
            NarrowPhaseDispatcher::generate_contacts(
                black_box(&cube),
                &upper,
                black_box(&cube),
                &lower,
                pair,
            )
        })
    });
    group.finish();
}

criterion_group!(
    benches,
    bench_hull_pairs,
    bench_manifold_topology,
    bench_primitive_pairs
);
criterion_main!(benches);
