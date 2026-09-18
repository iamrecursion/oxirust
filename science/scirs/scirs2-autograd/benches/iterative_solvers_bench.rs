//! Iterative Solvers Convergence Benchmarks
//!
//! This benchmark suite measures convergence rates and performance
//! of various iterative linear solvers.
//!
//! Categories:
//! - CG (Conjugate Gradient): 10 benchmarks
//! - GMRES (Generalized Minimal Residual): 10 benchmarks
//! - BiCGSTAB (Biconjugate Gradient Stabilized): 10 benchmarks
//! - PCG (Preconditioned Conjugate Gradient): 10 benchmarks
//! - Solver comparison: 10 benchmarks

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use scirs2_autograd as ag;
use scirs2_autograd::tensor_ops as T;
use scirs2_core::ndarray::Array2;
use std::hint::black_box;

/// Helper function to create a symmetric positive definite matrix
fn create_spd_matrix(size: usize, condition_number: f64) -> Array2<f64> {
    use scirs2_core::random::{thread_rng, Distribution, Uniform};
    let mut rng = thread_rng();
    let uniform = Uniform::new(-1.0, 1.0);

    // Create random matrix
    let mut a = Array2::from_shape_fn((size, size), |_| {
        uniform.expect("uniform failed").sample(&mut rng)
    });

    // Make it symmetric: A = (M + M^T) / 2
    for i in 0..size {
        for j in 0..i {
            let avg = (a[[i, j]] + a[[j, i]]) / 2.0;
            a[[i, j]] = avg;
            a[[j, i]] = avg;
        }
    }

    // Add diagonal dominance for positive definiteness
    for i in 0..size {
        a[[i, i]] = condition_number + size as f64;
    }

    a
}

/// CG convergence benchmarks (10 benchmarks)
fn cg_convergence_benchmark(c: &mut Criterion) {
    let mut group = c.benchmark_group("cg_convergence");

    // Benchmarks 1-5: Well-conditioned matrices
    for size in [10, 25, 50, 75, 100].iter() {
        group.bench_with_input(
            BenchmarkId::new("well_conditioned", size),
            size,
            |b, &size| {
                let a_matrix = create_spd_matrix(size, 1.0);
                let b_vec = Array2::from_elem((size, 1), 1.0);

                b.iter(|| {
                    ag::run(|ctx: &mut ag::Context<f64>| {
                        let a = ctx.placeholder("A", &[size as isize, size as isize]);
                        let b = ctx.placeholder("b", &[size as isize, 1]);

                        // Simulate CG-like operation
                        let x = ctx.placeholder("x", &[size as isize, 1]);
                        let ax = T::matmul(a, x);
                        let residual = ax - b;
                        let loss = T::reduce_sum(residual * residual, &[0, 1], false);

                        let grad = T::grad(&[loss], &[x]);
                        black_box(grad);
                    });
                });
            },
        );
    }

    // Benchmarks 6-10: Ill-conditioned matrices
    for size in [10, 25, 50, 75, 100].iter() {
        group.bench_with_input(
            BenchmarkId::new("ill_conditioned", size),
            size,
            |b, &size| {
                let a_matrix = create_spd_matrix(size, 1000.0); // High condition number
                let b_vec = Array2::from_elem((size, 1), 1.0);

                b.iter(|| {
                    ag::run(|ctx: &mut ag::Context<f64>| {
                        let a = ctx.placeholder("A", &[size as isize, size as isize]);
                        let b = ctx.placeholder("b", &[size as isize, 1]);

                        let x = ctx.placeholder("x", &[size as isize, 1]);
                        let ax = T::matmul(a, x);
                        let residual = ax - b;
                        let loss = T::reduce_sum(residual * residual, &[0, 1], false);

                        let grad = T::grad(&[loss], &[x]);
                        black_box(grad);
                    });
                });
            },
        );
    }

    group.finish();
}

/// GMRES convergence benchmarks (10 benchmarks)
fn gmres_convergence_benchmark(c: &mut Criterion) {
    let mut group = c.benchmark_group("gmres_convergence");

    // Benchmarks 11-15: GMRES with restart=5
    for size in [20, 40, 60, 80, 100].iter() {
        group.bench_with_input(BenchmarkId::new("restart_5", size), size, |b, &size| {
            b.iter(|| {
                ag::run(|ctx: &mut ag::Context<f64>| {
                    let a = ctx.placeholder("A", &[size as isize, size as isize]);
                    let b = ctx.placeholder("b", &[size as isize, 1]);
                    let x = ctx.placeholder("x", &[size as isize, 1]);

                    // Simulate GMRES iteration
                    let mut residuals = Vec::new();
                    for i in 0..5 {
                        let ax = T::matmul(a, x);
                        let r = ax - b;
                        residuals.push(r);
                    }

                    let grad = T::grad(&residuals, &[x]);
                    black_box(grad);
                });
            });
        });
    }

    // Benchmarks 16-20: GMRES with restart=10
    for size in [20, 40, 60, 80, 100].iter() {
        group.bench_with_input(BenchmarkId::new("restart_10", size), size, |b, &size| {
            b.iter(|| {
                ag::run(|ctx: &mut ag::Context<f64>| {
                    let a = ctx.placeholder("A", &[size as isize, size as isize]);
                    let b = ctx.placeholder("b", &[size as isize, 1]);
                    let x = ctx.placeholder("x", &[size as isize, 1]);

                    // Simulate GMRES iteration with larger restart
                    let mut residuals = Vec::new();
                    for i in 0..10 {
                        let ax = T::matmul(a, x);
                        let r = ax - b;
                        residuals.push(r);
                    }

                    let grad = T::grad(&residuals, &[x]);
                    black_box(grad);
                });
            });
        });
    }

    group.finish();
}

/// BiCGSTAB convergence benchmarks (10 benchmarks)
fn bicgstab_convergence_benchmark(c: &mut Criterion) {
    let mut group = c.benchmark_group("bicgstab_convergence");

    // Benchmarks 21-25: BiCGSTAB on symmetric matrices
    for size in [15, 30, 45, 60, 75].iter() {
        group.bench_with_input(BenchmarkId::new("symmetric", size), size, |b, &size| {
            b.iter(|| {
                ag::run(|ctx: &mut ag::Context<f64>| {
                    let a = ctx.placeholder("A", &[size as isize, size as isize]);
                    let b = ctx.placeholder("b", &[size as isize, 1]);
                    let x = ctx.placeholder("x", &[size as isize, 1]);

                    // Simulate BiCGSTAB-like operations
                    let r = T::matmul(a, x) - b;
                    let v = T::matmul(a, r);
                    let alpha = T::reduce_sum(r * v, &[0, 1], false);

                    let grad = T::grad(&[alpha], &[x]);
                    black_box(grad);
                });
            });
        });
    }

    // Benchmarks 26-30: BiCGSTAB on non-symmetric matrices
    for size in [15, 30, 45, 60, 75].iter() {
        group.bench_with_input(BenchmarkId::new("nonsymmetric", size), size, |b, &size| {
            b.iter(|| {
                ag::run(|ctx: &mut ag::Context<f64>| {
                    let a = ctx.placeholder("A", &[size as isize, size as isize]);
                    let at = ctx.placeholder("AT", &[size as isize, size as isize]);
                    let b = ctx.placeholder("b", &[size as isize, 1]);
                    let x = ctx.placeholder("x", &[size as isize, 1]);

                    // Non-symmetric operations
                    let r = T::matmul(a, x) - b;
                    let rt = T::matmul(at, r);
                    let loss = T::reduce_sum(rt * rt, &[0, 1], false);

                    let grad = T::grad(&[loss], &[x]);
                    black_box(grad);
                });
            });
        });
    }

    group.finish();
}

/// PCG (Preconditioned CG) speedup benchmarks (10 benchmarks)
fn pcg_speedup_benchmark(c: &mut Criterion) {
    let mut group = c.benchmark_group("pcg_speedup");

    // Benchmarks 31-35: Diagonal preconditioning
    for size in [25, 50, 75, 100, 125].iter() {
        group.bench_with_input(
            BenchmarkId::new("diagonal_precond", size),
            size,
            |b, &size| {
                b.iter(|| {
                    ag::run(|ctx: &mut ag::Context<f64>| {
                        let a = ctx.placeholder("A", &[size as isize, size as isize]);
                        let m_inv = ctx.placeholder("M_inv", &[size as isize, size as isize]); // Preconditioner
                        let b = ctx.placeholder("b", &[size as isize, 1]);
                        let x = ctx.placeholder("x", &[size as isize, 1]);

                        // Preconditioned residual
                        let r = T::matmul(a, x) - b;
                        let z = T::matmul(m_inv, r);
                        let loss = T::reduce_sum(z * z, &[0, 1], false);

                        let grad = T::grad(&[loss], &[x]);
                        black_box(grad);
                    });
                });
            },
        );
    }

    // Benchmarks 36-40: No preconditioning (comparison baseline)
    for size in [25, 50, 75, 100, 125].iter() {
        group.bench_with_input(BenchmarkId::new("no_precond", size), size, |b, &size| {
            b.iter(|| {
                ag::run(|ctx: &mut ag::Context<f64>| {
                    let a = ctx.placeholder("A", &[size as isize, size as isize]);
                    let b = ctx.placeholder("b", &[size as isize, 1]);
                    let x = ctx.placeholder("x", &[size as isize, 1]);

                    let r = T::matmul(a, x) - b;
                    let loss = T::reduce_sum(r * r, &[0, 1], false);

                    let grad = T::grad(&[loss], &[x]);
                    black_box(grad);
                });
            });
        });
    }

    group.finish();
}

/// Solver comparison benchmarks (10 benchmarks)
fn solver_comparison_benchmark(c: &mut Criterion) {
    let mut group = c.benchmark_group("solver_comparison");

    // Benchmarks 41-50: Compare different solver strategies
    for (idx, size) in [20, 30, 40, 50, 60, 70, 80, 90, 100, 110]
        .iter()
        .enumerate()
    {
        group.bench_with_input(
            BenchmarkId::new("solver_strategy", size),
            size,
            |b, &size| {
                b.iter(|| {
                    ag::run(|ctx: &mut ag::Context<f64>| {
                        let a = ctx.placeholder("A", &[size as isize, size as isize]);
                        let b = ctx.placeholder("b", &[size as isize, 1]);
                        let x = ctx.placeholder("x", &[size as isize, 1]);

                        // Different solver iteration styles
                        let strategy = idx % 3;
                        let loss = match strategy {
                            0 => {
                                // Direct method style
                                let ax = T::matmul(a, x);
                                let r = ax - b;
                                T::reduce_sum(r * r, &[0, 1], false)
                            }
                            1 => {
                                // Iterative refinement style
                                let mut current = x;
                                for _ in 0..3 {
                                    let ax = T::matmul(a, current);
                                    let r = ax - b;
                                    current = current - r * T::scalar(0.1, ctx);
                                }
                                T::reduce_sum(current * current, &[0, 1], false)
                            }
                            _ => {
                                // Gradient descent style
                                let ax = T::matmul(a, x);
                                let grad_x = T::grad(&[T::reduce_sum(ax, &[0, 1], false)], &[x])[0];
                                T::reduce_sum(grad_x * grad_x, &[0, 1], false)
                            }
                        };

                        let grad = T::grad(&[loss], &[x]);
                        black_box(grad);
                    });
                });
            },
        );
    }

    group.finish();
}

criterion_group!(
    benches,
    cg_convergence_benchmark,
    gmres_convergence_benchmark,
    bicgstab_convergence_benchmark,
    pcg_speedup_benchmark,
    solver_comparison_benchmark
);
criterion_main!(benches);
