// Copyright 2026 COOLJAPAN OU (Team KitaSan)
// SPDX-License-Identifier: Apache-2.0

//! Reproducibility tests of the parallel reductions and Krylov solvers.
//!
//! Rayon's `par_iter().sum()` combines partial sums in a tree that depends on work
//! stealing and on the thread count, so a CG or GMRES iteration built on it rounds
//! differently from run to run.  These tests run the same solve on pools of different
//! sizes and require *bitwise* equal iterates.

use super::*;

fn pool(threads: usize) -> rayon::ThreadPool {
    rayon::ThreadPoolBuilder::new()
        .num_threads(threads)
        .build()
        .expect("a rayon pool can be built")
}

fn bits(values: &[f64]) -> Vec<u64> {
    values.iter().map(|v| v.to_bits()).collect()
}

/// Order-sensitive test vector: values spread over several decades, no short binary
/// expansion.
fn awkward_vector(n: usize, salt: f64) -> Vec<f64> {
    (0..n)
        .map(|i| {
            let fraction = (i as f64 * 0.618_033_988_749_894_9 + salt).fract();
            let decade = 10.0_f64.powi((i % 7) as i32 - 3);
            decade * (fraction - 0.5)
        })
        .collect()
}

/// Five-point Laplacian on an `m x m` grid, optionally with an advection term that makes
/// it non-symmetric.
fn grid_operator(m: usize, advection: f64) -> CsrMatrix {
    let n = m * m;
    let mut row_offsets = Vec::with_capacity(n + 1);
    let mut col_indices = Vec::new();
    let mut values = Vec::new();
    row_offsets.push(0);
    for j in 0..m {
        for i in 0..m {
            let row = j * m + i;
            if j > 0 {
                col_indices.push(row - m);
                values.push(-1.0);
            }
            if i > 0 {
                col_indices.push(row - 1);
                values.push(-1.0 - advection);
            }
            col_indices.push(row);
            values.push(4.0 + 0.2 * advection);
            if i + 1 < m {
                col_indices.push(row + 1);
                values.push(-1.0 + advection);
            }
            if j + 1 < m {
                col_indices.push(row + m);
                values.push(-1.0);
            }
            row_offsets.push(col_indices.len());
        }
    }
    CsrMatrix {
        nrows: n,
        ncols: n,
        row_offsets,
        col_indices,
        values,
    }
}

fn smooth_rhs(n: usize) -> Vec<f64> {
    (0..n).map(|i| (0.37 * i as f64).sin() + 0.1).collect()
}

/// Neumaier-compensated dot product: an accurate value to judge `dot_par` against.
fn compensated_dot(a: &[f64], b: &[f64]) -> f64 {
    let mut sum = 0.0_f64;
    let mut compensation = 0.0_f64;
    for (x, y) in a.iter().zip(b) {
        let term = x * y;
        let t = sum + term;
        compensation += if sum.abs() >= term.abs() {
            (sum - t) + term
        } else {
            (term - t) + sum
        };
        sum = t;
    }
    sum + compensation
}

#[test]
fn dot_par_uses_the_documented_fixed_chunk_order() {
    let n = 5 * DOT_CHUNK_LEN + 123;
    let a = awkward_vector(n, 0.11);
    let b = awkward_vector(n, 0.73);

    // Reference: sequential chunk sums, added in chunk order.
    let reference = a
        .chunks(DOT_CHUNK_LEN)
        .zip(b.chunks(DOT_CHUNK_LEN))
        .map(|(x, y)| x.iter().zip(y).map(|(p, q)| p * q).sum::<f64>())
        .collect::<Vec<f64>>()
        .iter()
        .sum::<f64>();

    for threads in [1, 2, 3, 5, 8] {
        let pool = pool(threads);
        for _ in 0..5 {
            let value = pool.install(|| dot_par(&a, &b));
            assert_eq!(value.to_bits(), reference.to_bits(), "{threads} threads");
        }
    }

    // And it is a correct dot product.
    let exact = compensated_dot(&a, &b);
    let scale: f64 = a.iter().zip(&b).map(|(x, y)| (x * y).abs()).sum();
    assert!(
        (reference - exact).abs() <= 1e-12 * scale,
        "dot_par {reference} vs compensated {exact}"
    );
}

#[test]
fn dot_par_handles_short_empty_and_mismatched_vectors() {
    assert_eq!(dot_par(&[], &[]), 0.0);
    assert_eq!(dot_par(&[2.0], &[3.0]), 6.0);

    let a = awkward_vector(DOT_CHUNK_LEN, 0.2);
    let b = awkward_vector(DOT_CHUNK_LEN, 0.9);
    let expected: f64 = a.iter().zip(&b).map(|(x, y)| x * y).sum();
    assert_eq!(dot_par(&a, &b).to_bits(), expected.to_bits());

    // The common prefix is used, as with `zip`.
    let long = awkward_vector(DOT_CHUNK_LEN + 17, 0.4);
    let prefix: f64 = long.iter().zip(&a).map(|(x, y)| x * y).sum();
    assert_eq!(dot_par(&long, &a).to_bits(), prefix.to_bits());
    assert_eq!(dot_par(&a, &long).to_bits(), prefix.to_bits());
}

#[test]
fn pcg_iterates_are_bitwise_identical_for_one_and_four_threads() {
    // 150 x 150 unknowns: several dot-product chunks, many iterations that amplify any
    // difference in rounding.
    let a = grid_operator(150, 0.0);
    let b = smooth_rhs(a.nrows);
    let solver = ParallelPcgSolver::new(60, 1e-30);

    let solve = |threads: usize| {
        let mut x = vec![0.0; a.nrows];
        let stats = pool(threads).install(|| solver.solve(&a, &b, &mut x));
        (x, stats)
    };
    let (x1, stats1) = solve(1);
    assert_eq!(stats1.iterations, 60);
    for threads in [2, 4, 7] {
        let (x, stats) = solve(threads);
        assert_eq!(bits(&x), bits(&x1), "{threads} threads");
        assert_eq!(stats.iterations, stats1.iterations);
        assert_eq!(
            stats.residual_norm.to_bits(),
            stats1.residual_norm.to_bits()
        );
    }
    // The repeated 4-thread solves agree with each other as well.
    let (again, _) = solve(4);
    assert_eq!(bits(&again), bits(&x1));
}

#[test]
fn pcg_still_converges_on_the_grid_problem() {
    let a = grid_operator(60, 0.0);
    let b = smooth_rhs(a.nrows);
    let mut x = vec![0.0; a.nrows];
    let stats = ParallelPcgSolver::new(500, 1e-10).solve(&a, &b, &mut x);
    assert!(stats.converged, "{stats:?}");
    let mut ax = vec![0.0; a.nrows];
    a.spmv(&x, &mut ax);
    let residual: f64 = ax
        .iter()
        .zip(&b)
        .map(|(l, r)| (l - r) * (l - r))
        .sum::<f64>()
        .sqrt();
    let rhs_norm: f64 = b.iter().map(|v| v * v).sum::<f64>().sqrt();
    assert!(residual <= 1e-8 * rhs_norm, "residual {residual}");
}

#[test]
fn gmres_iterates_are_bitwise_identical_for_one_and_four_threads() {
    // Non-symmetric advection-diffusion operator with 100 x 100 unknowns.
    let a = grid_operator(100, 0.35);
    let b = smooth_rhs(a.nrows);
    let solver = ParallelGmresSolver {
        krylov_dim: 10,
        max_restarts: 3,
        tolerance: 1e-30,
    };

    let solve = |threads: usize| {
        let mut x = vec![0.0; a.nrows];
        let stats = pool(threads).install(|| solver.solve(&a, &b, &mut x));
        (x, stats)
    };
    let (x1, stats1) = solve(1);
    assert!(stats1.iterations > 0);
    for threads in [2, 4, 7] {
        let (x, stats) = solve(threads);
        assert_eq!(bits(&x), bits(&x1), "{threads} threads");
        assert_eq!(stats.iterations, stats1.iterations);
        assert_eq!(
            stats.residual_norm.to_bits(),
            stats1.residual_norm.to_bits()
        );
    }
}

#[test]
fn gmres_converges_on_the_non_symmetric_grid_problem() {
    let a = grid_operator(40, 0.35);
    let b = smooth_rhs(a.nrows);
    let mut x = vec![0.0; a.nrows];
    let solver = ParallelGmresSolver {
        krylov_dim: 30,
        max_restarts: 20,
        tolerance: 1e-10,
    };
    let stats = solver.solve(&a, &b, &mut x);
    assert!(stats.converged, "{stats:?}");
    let mut ax = vec![0.0; a.nrows];
    a.spmv(&x, &mut ax);
    let residual: f64 = ax
        .iter()
        .zip(&b)
        .map(|(l, r)| (l - r) * (l - r))
        .sum::<f64>()
        .sqrt();
    let rhs_norm: f64 = b.iter().map(|v| v * v).sum::<f64>().sqrt();
    assert!(residual <= 1e-7 * rhs_norm, "residual {residual}");
}
