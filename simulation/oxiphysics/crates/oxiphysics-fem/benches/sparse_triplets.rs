// Copyright 2026 COOLJAPAN OU (Team KitaSan)
// SPDX-License-Identifier: Apache-2.0

//! Criterion benchmark: triplet (COO) to CSR/CSC compression with one million
//! entries, the hot path of every global stiffness assembly.
//! Run with: cargo bench -p oxiphysics-fem --bench sparse_triplets
//!
//! `from_triplets` is a hash-free counting sort (`O(nnz + Σ k log k)`).  The
//! `hashmap_reference` case reproduces the previous implementation (a
//! `HashMap<(row, col), f64>` accumulated and then sorted) so the speed-up stays
//! measurable.

use criterion::{BatchSize, BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};
use oxiphysics_fem::assembly::CooMatrix;
use oxiphysics_fem::sparse::{CscMatrix, CsrMatrix};
use std::collections::HashMap;
use std::hint::black_box;

/// Number of triplets in the benchmark systems.
const TRIPLET_COUNT: usize = 1_000_000;

/// Linear-congruential generator: deterministic input without extra crates.
struct Lcg(u64);

impl Lcg {
    fn next(&mut self) -> u64 {
        self.0 = self
            .0
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        self.0 >> 33
    }
}

/// Uniformly scattered triplets of an `n × n` matrix (random order, with
/// duplicates once `count` exceeds the number of distinct positions per row).
fn scattered(n: usize, count: usize) -> Vec<(usize, usize, f64)> {
    let mut rng = Lcg(0x5EED);
    (0..count)
        .map(|_| {
            let row = (rng.next() % n as u64) as usize;
            let col = (rng.next() % n as u64) as usize;
            (row, col, (rng.next() % 1000) as f64 * 1e-3)
        })
        .collect()
}

/// Element-wise assembly pattern of a structured mesh: every element scatters a
/// dense `dofs × dofs` block, so interior DOFs receive many duplicate
/// contributions and rows are short (the realistic FEM case).
fn element_blocks(elements: usize, dofs: usize) -> Vec<(usize, usize, f64)> {
    let mut triplets = Vec::with_capacity(elements * dofs * dofs);
    for e in 0..elements {
        for i in 0..dofs {
            for j in 0..dofs {
                triplets.push((e + i, e + j, 1.0 + (i + j) as f64));
            }
        }
    }
    triplets
}

/// The pre-rewrite algorithm: hash accumulation followed by a sort.
fn hashmap_reference(
    nrows: usize,
    triplets: &[(usize, usize, f64)],
) -> (Vec<usize>, Vec<usize>, Vec<f64>) {
    let mut map: HashMap<(usize, usize), f64> = HashMap::new();
    for &(r, c, v) in triplets {
        *map.entry((r, c)).or_insert(0.0) += v;
    }
    let mut entries: Vec<((usize, usize), f64)> = map.into_iter().collect();
    entries.sort_by_key(|&((r, c), _)| (r, c));
    let mut row_ptr = vec![0usize; nrows + 1];
    let mut cols = Vec::with_capacity(entries.len());
    let mut values = Vec::with_capacity(entries.len());
    for &((r, c), v) in &entries {
        row_ptr[r + 1] += 1;
        cols.push(c);
        values.push(v);
    }
    for i in 1..=nrows {
        row_ptr[i] += row_ptr[i - 1];
    }
    (row_ptr, cols, values)
}

fn bench_compression(c: &mut Criterion) {
    let mut group = c.benchmark_group("triplets_to_sparse");
    group.sample_size(10);
    group.throughput(Throughput::Elements(TRIPLET_COUNT as u64));

    let n = 100_000;
    let random = scattered(n, TRIPLET_COUNT);
    group.bench_with_input(
        BenchmarkId::new("csr_from_triplets", "1e6_scattered"),
        &random,
        |bench, triplets| bench.iter(|| black_box(CsrMatrix::from_triplets(n, n, triplets))),
    );
    group.bench_with_input(
        BenchmarkId::new("csc_from_triplets", "1e6_scattered"),
        &random,
        |bench, triplets| bench.iter(|| black_box(CscMatrix::from_triplets(n, n, triplets))),
    );
    group.bench_with_input(
        BenchmarkId::new("hashmap_reference", "1e6_scattered"),
        &random,
        |bench, triplets| bench.iter(|| black_box(hashmap_reference(n, triplets))),
    );

    // 1e6 / 144 element blocks of 12 DOFs (a linear tetrahedron's 12 x 12 matrix).
    let blocks = element_blocks(TRIPLET_COUNT / 144, 12);
    let ndof = TRIPLET_COUNT / 144 + 12;
    group.bench_with_input(
        BenchmarkId::new("csr_from_triplets", "1e6_element_blocks"),
        &blocks,
        |bench, triplets| bench.iter(|| black_box(CsrMatrix::from_triplets(ndof, ndof, triplets))),
    );
    group.bench_with_input(
        BenchmarkId::new("hashmap_reference", "1e6_element_blocks"),
        &blocks,
        |bench, triplets| bench.iter(|| black_box(hashmap_reference(ndof, triplets))),
    );

    // The assembler path: COO -> CSR without cloning the entry list.
    group.bench_function(BenchmarkId::new("coo_to_csr", "1e6_scattered"), |bench| {
        bench.iter_batched(
            || {
                let mut coo = CooMatrix::new(n, n);
                coo.entries = random.clone();
                coo
            },
            |coo| black_box(coo.to_csr()),
            BatchSize::LargeInput,
        )
    });
    group.finish();
}

criterion_group!(benches, bench_compression);
criterion_main!(benches);
