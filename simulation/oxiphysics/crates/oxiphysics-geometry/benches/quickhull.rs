// Copyright 2026 COOLJAPAN OU (Team KitaSan)
// SPDX-License-Identifier: Apache-2.0

//! Benchmarks of the Quickhull convex hull builders.
//!
//! `ConvexHull3D::build` and `ConvexHull3DVec::build` share one Quickhull core:
//! every face keeps the points in front of it, the hull grows by the point
//! farthest in front of a face, and the faces a point sees are found over the
//! face adjacency.  The cost per point is therefore proportional to the size of
//! its visible region instead of the number of hull faces, which is what the
//! groups below measure:
//!
//! * `sphere_points`: random points on a sphere, so every point is a hull vertex
//!   and the hull has about `2 n` faces (the worst case for an algorithm that
//!   tests every point against every face);
//! * `ball_points`: random points in a ball, where only a handful of points are
//!   hull vertices and almost every point is discarded after one test per face of
//!   the seed.

use std::hint::black_box;

use criterion::{BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};
use oxiphysics_core::math::Vec3;
use oxiphysics_geometry::{ConvexHull3D, ConvexHull3DVec};

/// Deterministic xorshift generator returning values in `[0, 1)`.
struct Rng(u64);

impl Rng {
    fn unit(&mut self) -> f64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        (self.0 >> 11) as f64 / (1_u64 << 53) as f64
    }

    fn signed(&mut self) -> f64 {
        2.0 * self.unit() - 1.0
    }
}

/// `count` random points on the unit sphere.
fn sphere_points(count: usize) -> Vec<[f64; 3]> {
    let mut rng = Rng(0x2545_F491_4F6C_DD1D);
    let mut points = Vec::with_capacity(count);
    while points.len() < count {
        let p = [rng.signed(), rng.signed(), rng.signed()];
        let length = (p[0] * p[0] + p[1] * p[1] + p[2] * p[2]).sqrt();
        if (0.1..=1.0).contains(&length) {
            points.push([p[0] / length, p[1] / length, p[2] / length]);
        }
    }
    points
}

/// `count` random points in the unit ball.
fn ball_points(count: usize) -> Vec<[f64; 3]> {
    let mut rng = Rng(0x9E37_79B9_7F4A_7C15);
    let mut points = Vec::with_capacity(count);
    while points.len() < count {
        let p = [rng.signed(), rng.signed(), rng.signed()];
        if p[0] * p[0] + p[1] * p[1] + p[2] * p[2] <= 1.0 {
            points.push(p);
        }
    }
    points
}

fn to_vectors(points: &[[f64; 3]]) -> Vec<Vec3> {
    points.iter().map(|p| Vec3::new(p[0], p[1], p[2])).collect()
}

fn bench_cloud(c: &mut Criterion, group_name: &str, cloud: fn(usize) -> Vec<[f64; 3]>) {
    let mut group = c.benchmark_group(group_name);
    group.sample_size(10);
    for &count in &[1_000_usize, 10_000, 100_000] {
        let points = cloud(count);
        let vectors = to_vectors(&points);
        group.throughput(Throughput::Elements(count as u64));
        group.bench_with_input(BenchmarkId::new("ConvexHull3D", count), &points, |b, p| {
            b.iter(|| ConvexHull3D::build(black_box(p)))
        });
        group.bench_with_input(
            BenchmarkId::new("ConvexHull3DVec", count),
            &vectors,
            |b, p| b.iter(|| ConvexHull3DVec::build(black_box(p))),
        );
    }
    group.finish();
}

fn bench_sphere_points(c: &mut Criterion) {
    bench_cloud(c, "quickhull/sphere_points", sphere_points);
}

fn bench_ball_points(c: &mut Criterion) {
    bench_cloud(c, "quickhull/ball_points", ball_points);
}

criterion_group!(benches, bench_sphere_points, bench_ball_points);
criterion_main!(benches);
