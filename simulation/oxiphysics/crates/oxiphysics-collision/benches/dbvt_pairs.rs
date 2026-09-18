// Copyright 2026 COOLJAPAN OU (Team KitaSan)
// SPDX-License-Identifier: Apache-2.0

//! Benchmarks of `DynamicBvh::query_overlapping_pairs`.
//!
//! The pair query used to test every leaf against every other leaf, so the
//! hierarchy pruned nothing and 10 000 bodies cost about 5e7 box tests. It now
//! walks the hierarchy with a pruned self-collision traversal. Each group runs
//! the tree query next to the brute-force scan over the same fat leaf boxes (the
//! old cost), for 1 000 and 10 000 leaves scattered at a fixed density, so the
//! number of overlapping pairs grows linearly with the leaf count.

use std::hint::black_box;

use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};
use oxiphysics_collision::{BvhAabb, DynamicBvh};
use oxiphysics_core::math::Vec3;

/// Deterministic linear congruential generator.
struct Lcg(u64);

impl Lcg {
    /// Uniform sample in `[0, 1)`.
    fn unit(&mut self) -> f64 {
        self.0 = self
            .0
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        (self.0 >> 33) as f64 / (1u64 << 31) as f64
    }
}

/// `count` boxes of half extent 0.3 to 0.8 scattered over a cube whose side
/// grows with the cube root of `count`, which keeps the density fixed.
fn scene(count: u32) -> DynamicBvh {
    let mut rng = Lcg(0x0DB7_5EED);
    let side = f64::from(count).cbrt() * 2.0;
    let mut tree = DynamicBvh::new();
    for data in 0..count {
        let center = Vec3::new(rng.unit() * side, rng.unit() * side, rng.unit() * side);
        let half = 0.3 + 0.5 * rng.unit();
        let extent = Vec3::new(half, half, half);
        tree.insert(BvhAabb::new(center - extent, center + extent), data);
    }
    tree
}

/// The pair count by testing every leaf against every other leaf.
fn brute_force_pair_count(tree: &DynamicBvh) -> usize {
    let leaves = tree.leaf_snapshot();
    let mut pairs = 0;
    for (i, (_, aabb_a)) in leaves.iter().enumerate() {
        for (_, aabb_b) in &leaves[(i + 1)..] {
            if aabb_a.intersects(aabb_b) {
                pairs += 1;
            }
        }
    }
    pairs
}

fn bench_pair_query(c: &mut Criterion) {
    let mut group = c.benchmark_group("dbvt/query_overlapping_pairs");
    group.sample_size(10);
    for count in [1_000_u32, 10_000] {
        let tree = scene(count);
        // Both strategies must find the same pairs before they are compared.
        assert_eq!(
            tree.query_overlapping_pairs().len(),
            brute_force_pair_count(&tree)
        );
        group.bench_with_input(BenchmarkId::new("tree", count), &tree, |b, tree| {
            b.iter(|| black_box(tree.query_overlapping_pairs()));
        });
        group.bench_with_input(BenchmarkId::new("brute_force", count), &tree, |b, tree| {
            b.iter(|| black_box(brute_force_pair_count(tree)));
        });
    }
    group.finish();
}

criterion_group!(benches, bench_pair_query);
criterion_main!(benches);
