// Copyright 2026 COOLJAPAN OU (Team KitaSan)
// SPDX-License-Identifier: Apache-2.0

//! Tests of the CP-ALS start values ([`CpInit`]): the HOSVD start must hold the
//! leading singular vectors of the unfoldings, be deterministic, and make
//! [`cp_als`] converge in far fewer sweeps than the fixed ramp start it replaced.

use super::*;
use crate::numerical_linear_algebra::TensorDecomposition;

/// Deterministic pseudo-random numbers in `[-1, 1)`.
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

type Factor = Vec<Vec<f64>>;

fn random_factor(rng: &mut Lcg, rows: usize, rank: usize) -> Factor {
    (0..rows)
        .map(|_| (0..rank).map(|_| rng.next()).collect())
        .collect()
}

/// `Σ_r λ_r a_r ∘ b_r ∘ c_r`, built element by element and independently of
/// `cp_reconstruct`.
fn cp_tensor(a: &Factor, b: &Factor, c: &Factor, lambdas: &[f64]) -> DenseTensor {
    let mut tensor = DenseTensor::zeros(&[a.len(), b.len(), c.len()]);
    for (i, ai) in a.iter().enumerate() {
        for (j, bj) in b.iter().enumerate() {
            for (k, ck) in c.iter().enumerate() {
                let value: f64 = lambdas
                    .iter()
                    .enumerate()
                    .map(|(r, lam)| lam * ai[r] * bj[r] * ck[r])
                    .sum();
                tensor.set(&[i, j, k], value);
            }
        }
    }
    tensor
}

/// A 3x4x5 tensor of CP rank two with different factors in every mode, and the
/// factors it was built from.
fn rank_two_fixture() -> (DenseTensor, [Factor; 3]) {
    let a = vec![vec![1.0, 0.5], vec![2.0, -1.0], vec![0.5, 1.5]];
    let b = vec![
        vec![1.0, 2.0],
        vec![0.5, -1.0],
        vec![-1.0, 0.5],
        vec![2.0, 1.0],
    ];
    let c = vec![
        vec![1.0, -0.5],
        vec![0.0, 1.0],
        vec![2.0, 1.0],
        vec![-1.0, 0.5],
        vec![0.5, 2.0],
    ];
    let tensor = cp_tensor(&a, &b, &c, &[3.0, 1.5]);
    (tensor, [a, b, c])
}

/// An exactly rank-`rank` tensor with random factors and weights in `[1, 3)`.
fn random_low_rank(shape: [usize; 3], rank: usize, seed: u64) -> DenseTensor {
    let mut rng = Lcg(seed);
    let a = random_factor(&mut rng, shape[0], rank);
    let b = random_factor(&mut rng, shape[1], rank);
    let c = random_factor(&mut rng, shape[2], rank);
    let lambdas: Vec<f64> = (0..rank).map(|_| 2.0 + rng.next()).collect();
    cp_tensor(&a, &b, &c, &lambdas)
}

/// A factor whose columns share a common component: column `r` is
/// `√ρ · common + √(1 − ρ) · own_r`, so two columns have correlation about `ρ`.
/// Strongly correlated columns make ALS slow and give it swamps.
fn collinear_factor(rng: &mut Lcg, rows: usize, rank: usize, correlation: f64) -> Factor {
    let common: Vec<f64> = (0..rows).map(|_| rng.next()).collect();
    let mut factor = vec![vec![0.0; rank]; rows];
    for r in 0..rank {
        let own: Vec<f64> = (0..rows).map(|_| rng.next()).collect();
        for (row, (c, o)) in factor.iter_mut().zip(common.iter().zip(&own)) {
            row[r] = correlation.sqrt() * c + (1.0 - correlation).sqrt() * o;
        }
    }
    factor
}

/// An exactly rank-`rank` tensor whose factors have correlated columns.
fn collinear_low_rank(shape: [usize; 3], rank: usize, seed: u64, correlation: f64) -> DenseTensor {
    let mut rng = Lcg(seed);
    let a = collinear_factor(&mut rng, shape[0], rank, correlation);
    let b = collinear_factor(&mut rng, shape[1], rank, correlation);
    let c = collinear_factor(&mut rng, shape[2], rank, correlation);
    let lambdas: Vec<f64> = (0..rank).map(|_| 2.0 + rng.next()).collect();
    cp_tensor(&a, &b, &c, &lambdas)
}

fn column(factor: &Factor, r: usize) -> Vec<f64> {
    factor.iter().map(|row| row[r]).collect()
}

fn inner(u: &[f64], v: &[f64]) -> f64 {
    u.iter().zip(v).map(|(x, y)| x * y).sum()
}

/// Asserts that `target` lies in the span of the orthonormal `basis` (relative
/// residual below 1e-9).
fn assert_in_span(basis: &[Vec<f64>], target: &[f64], label: &str) {
    let mut residual = target.to_vec();
    for u in basis {
        let coefficient = inner(u, target);
        for (res, ui) in residual.iter_mut().zip(u) {
            *res -= coefficient * ui;
        }
    }
    let relative = inner(&residual, &residual).sqrt() / inner(target, target).sqrt();
    assert!(
        relative < 1e-9,
        "{label}: not in the span of the start ({relative})"
    );
}

fn shape_of(tensor: &DenseTensor) -> [usize; 3] {
    [tensor.shape[0], tensor.shape[1], tensor.shape[2]]
}

fn start(tensor: &DenseTensor, rank: usize, init: CpInit) -> [Factor; 3] {
    initial_cp_factors(&tensor.data, shape_of(tensor), rank, init)
}

fn assert_all_finite(cp: &CpDecomposition) {
    for factor in [&cp.a, &cp.b, &cp.c] {
        assert!(factor.iter().flatten().all(|v| v.is_finite()));
    }
    assert!(cp.lambdas.iter().all(|v| v.is_finite()));
}

#[test]
fn hosvd_is_the_default_start() {
    assert_eq!(CpInit::default(), CpInit::Hosvd);
    let tensor = random_low_rank([4, 5, 6], 2, 11);
    let by_default = cp_als(&tensor, 2, 40, 1e-12);
    let explicit = cp_als_with_init(&tensor, 2, 40, 1e-12, CpInit::Hosvd);
    assert_eq!(by_default.iterations, explicit.iterations);
    assert_eq!(by_default.converged, explicit.converged);
    assert_eq!(by_default.lambdas, explicit.lambdas);
    assert_eq!(by_default.a, explicit.a);
    assert_eq!(by_default.b, explicit.b);
    assert_eq!(by_default.c, explicit.c);
}

#[test]
fn hosvd_start_is_deterministic() {
    let tensor = random_low_rank([6, 7, 8], 3, 5);
    let first = start(&tensor, 3, CpInit::Hosvd);
    let second = start(&tensor, 3, CpInit::Hosvd);
    assert_eq!(first, second);
}

#[test]
fn hosvd_start_columns_are_orthonormal() {
    let tensor = random_low_rank([6, 7, 8], 3, 5);
    for (mode, factor) in start(&tensor, 3, CpInit::Hosvd).iter().enumerate() {
        assert_eq!(factor.len(), tensor.shape[mode]);
        for p in 0..3 {
            for q in 0..3 {
                let expected = if p == q { 1.0 } else { 0.0 };
                let got = inner(&column(factor, p), &column(factor, q));
                assert!(
                    (got - expected).abs() < 1e-10,
                    "mode {mode}: <u{p}, u{q}> = {got}"
                );
            }
        }
    }
}

#[test]
fn hosvd_start_spans_the_column_spaces_of_an_exactly_low_rank_tensor() {
    let (tensor, truth) = rank_two_fixture();
    let factors = start(&tensor, 2, CpInit::Hosvd);
    for (mode, (factor, true_factor)) in factors.iter().zip(&truth).enumerate() {
        let basis = [column(factor, 0), column(factor, 1)];
        for r in 0..2 {
            assert_in_span(
                &basis,
                &column(true_factor, r),
                &format!("mode {mode}, true column {r}"),
            );
        }
    }
}

#[test]
fn hosvd_start_orders_the_columns_by_singular_value_and_fixes_their_signs() {
    // X = 5 u0 ∘ v0 ∘ w0 − 1 u1 ∘ v1 ∘ w1 with orthonormal pairs in every mode, so
    // the singular values of every unfolding are 5 and 1 and the leading left
    // singular vectors are ±u0, ±u1 (resp. v, w).  The start must list the
    // dominant one first, with the component of largest magnitude positive.
    let u = [
        vec![1.0 / 3.0, 2.0 / 3.0, 2.0 / 3.0],
        vec![2.0 / 3.0, 1.0 / 3.0, -2.0 / 3.0],
    ];
    let v = [vec![0.5, 0.5, 0.5, 0.5], vec![0.5, -0.5, 0.5, -0.5]];
    let w = [vec![1.0, 0.0, 0.0, 0.0, 0.0], vec![0.0, 1.0, 0.0, 0.0, 0.0]];
    let as_factor = |pair: &[Vec<f64>; 2]| -> Factor {
        (0..pair[0].len())
            .map(|row| vec![pair[0][row], pair[1][row]])
            .collect()
    };
    let tensor = cp_tensor(&as_factor(&u), &as_factor(&v), &as_factor(&w), &[5.0, -1.0]);
    let factors = start(&tensor, 2, CpInit::Hosvd);
    for (mode, (factor, expected)) in factors.iter().zip([&u, &v, &w]).enumerate() {
        for (r, want) in expected.iter().enumerate() {
            let got = column(factor, r);
            for (g, e) in got.iter().zip(want) {
                assert!(
                    (g - e).abs() < 1e-12,
                    "mode {mode}, column {r}: {got:?} != {want:?}"
                );
            }
        }
    }
}

#[test]
fn hosvd_start_pads_missing_singular_vectors_with_distinct_unit_columns() {
    // 2x3x4 with rank 4: the modes have 2, 3 and 4 rows, so the first two modes
    // run out of singular vectors.
    let tensor = random_low_rank([2, 3, 4], 2, 3);
    let factors = start(&tensor, 4, CpInit::Hosvd);
    assert_eq!(factors, start(&tensor, 4, CpInit::Hosvd), "deterministic");
    for (mode, factor) in factors.iter().enumerate() {
        let rows = tensor.shape[mode];
        assert_eq!(factor.len(), rows);
        assert!(factor.iter().all(|row| row.len() == 4));
        assert!(factor.iter().flatten().all(|v| v.is_finite()));
        let genuine = rows.min(4);
        for p in 0..4 {
            let cp = column(factor, p);
            let norm = inner(&cp, &cp).sqrt();
            assert!((norm - 1.0).abs() < 1e-10, "mode {mode} column {p}: {norm}");
            for q in (p + 1)..4 {
                let cos = inner(&cp, &column(factor, q)).abs();
                assert!(
                    cos < 1.0 - 1e-6,
                    "mode {mode}: columns {p} and {q} parallel"
                );
                if q < genuine {
                    assert!(cos < 1e-10, "mode {mode}: singular vectors {p}, {q}");
                }
            }
        }
    }
    // Different modes draw different padding.
    assert_ne!(column(&factors[0], 2), column(&factors[0], 3));
}

#[test]
fn random_start_is_reproducible_and_depends_on_the_seed() {
    let tensor = random_low_rank([3, 4, 5], 2, 9);
    let first = start(&tensor, 2, CpInit::Random { seed: 7 });
    assert_eq!(first, start(&tensor, 2, CpInit::Random { seed: 7 }));
    assert_ne!(first, start(&tensor, 2, CpInit::Random { seed: 8 }));
    for (mode, factor) in first.iter().enumerate() {
        assert_eq!(factor.len(), tensor.shape[mode]);
        assert!(
            factor.iter().flatten().all(|v| (-0.5..0.5).contains(v)),
            "mode {mode}: entries in [-0.5, 0.5)"
        );
    }
}

#[test]
fn random_start_with_seed_42_is_the_previous_flat_start() {
    // The generator `TensorDecomposition::cp_als` used before the HOSVD start.
    let mut state = 42u64;
    let mut legacy = move || -> f64 {
        state = state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        (state >> 11) as f64 / (1u64 << 53) as f64 - 0.5
    };
    let (shape, rank) = ([3usize, 4, 5], 2usize);
    let tensor = DenseTensor::zeros(&shape);
    let factors = start(&tensor, rank, CpInit::Random { seed: 42 });
    for (factor, rows) in factors.iter().zip(shape) {
        let flat: Vec<f64> = factor.iter().flatten().copied().collect();
        let expected: Vec<f64> = (0..rows * rank).map(|_| legacy()).collect();
        assert_eq!(flat, expected);
    }
}

#[test]
fn ramp_start_is_the_previous_dense_start() {
    let tensor = DenseTensor::zeros(&[3, 4, 5]);
    let rank = 3;
    let factors = start(&tensor, rank, CpInit::Ramp);
    let init_val = 1.0 / rank as f64;
    // The expressions `cp_als` used before the HOSVD start existed.
    let legacy_a = |i: usize, r: usize| ((i + r) as f64 + 1.0) * init_val;
    let legacy_b = |j: usize, r: usize| ((j + r + 1) as f64) * init_val;
    let legacy_c = |k: usize, r: usize| ((k + r + 2) as f64) * init_val;
    for (i, row) in factors[0].iter().enumerate() {
        for (r, &value) in row.iter().enumerate() {
            assert_eq!(value, legacy_a(i, r), "A[{i}][{r}]");
        }
    }
    for (j, row) in factors[1].iter().enumerate() {
        for (r, &value) in row.iter().enumerate() {
            assert_eq!(value, legacy_b(j, r), "B[{j}][{r}]");
        }
    }
    for (k, row) in factors[2].iter().enumerate() {
        for (r, &value) in row.iter().enumerate() {
            assert_eq!(value, legacy_c(k, r), "C[{k}][{r}]");
        }
        assert_eq!(row.len(), rank);
    }
}

#[test]
fn hosvd_start_needs_far_fewer_sweeps_on_the_rank_two_fixture() {
    let (tensor, _) = rank_two_fixture();
    let hosvd = cp_als_with_init(&tensor, 2, 5000, 1e-10, CpInit::Hosvd);
    let ramp = cp_als_with_init(&tensor, 2, 5000, 1e-10, CpInit::Ramp);
    assert!(hosvd.converged, "HOSVD start: {} sweeps", hosvd.iterations);
    assert!(cp_relative_error(&tensor, &hosvd) < 1e-8);
    assert!(
        hosvd.iterations * 4 < ramp.iterations,
        "HOSVD start {} sweeps vs ramp start {}",
        hosvd.iterations,
        ramp.iterations
    );
}

#[test]
fn hosvd_start_converges_on_rank_three_tensors_where_the_ramp_start_stalls() {
    // The ramp start has rank at most two in every mode, so it starts a rank-3
    // problem rank-deficient.
    for (shape, seed) in [([6, 7, 8], 21), ([8, 6, 4], 22), ([5, 6, 7], 23)] {
        let tensor = random_low_rank(shape, 3, seed);
        let hosvd = cp_als_with_init(&tensor, 3, 3000, 1e-10, CpInit::Hosvd);
        let ramp = cp_als_with_init(&tensor, 3, 3000, 1e-10, CpInit::Ramp);
        assert!(
            hosvd.converged && cp_relative_error(&tensor, &hosvd) < 1e-8,
            "{shape:?}: HOSVD start did not reach the exact fit in {} sweeps",
            hosvd.iterations
        );
        let ramp_error = cp_relative_error(&tensor, &ramp);
        assert!(
            ramp.iterations > 4 * hosvd.iterations || ramp_error > 1e-6,
            "{shape:?}: ramp start {} sweeps (error {ramp_error}) vs HOSVD {}",
            ramp.iterations,
            hosvd.iterations
        );
    }
}

#[test]
fn hosvd_start_stays_finite_when_the_rank_exceeds_every_mode_size() {
    let tensor = random_low_rank([2, 2, 2], 3, 31);
    let cp = cp_als(&tensor, 3, 200, 1e-12);
    assert_all_finite(&cp);
    assert_eq!((cp.a.len(), cp.b.len(), cp.c.len()), (2, 2, 2));
    assert_eq!(cp.rank(), 3);
    assert!(cp_relative_error(&tensor, &cp) < 1e-6);
}

#[test]
fn hosvd_start_of_a_zero_tensor_is_finite() {
    let tensor = DenseTensor::zeros(&[3, 3, 3]);
    let factors = start(&tensor, 2, CpInit::Hosvd);
    for factor in &factors {
        assert!(factor.iter().flatten().all(|v| v.is_finite()));
    }
    let cp = cp_als(&tensor, 2, 20, 1e-8);
    assert_all_finite(&cp);
    assert!(cp.lambdas.iter().all(|&l| l == 0.0));
}

#[test]
fn flat_cp_als_defaults_to_the_hosvd_start() {
    let tensor = random_low_rank([4, 5, 6], 2, 41);
    let shape = [4, 5, 6];
    let by_default = TensorDecomposition::cp_als(&tensor.data, &shape, 2, 60, 1e-10);
    let explicit =
        TensorDecomposition::cp_als_with_init(&tensor.data, &shape, 2, 60, 1e-10, CpInit::Hosvd);
    assert_eq!(by_default.iterations, explicit.iterations);
    assert_eq!(by_default.core, explicit.core);
    assert_eq!(by_default.factors, explicit.factors);
}

#[test]
fn flat_hosvd_start_follows_the_row_major_layout_of_the_flat_tensor() {
    // The flat API reads `x[i * J * K + j * K + k]` (the last index fastest).  The
    // tensor is filled in that order straight from the factors, without
    // `DenseTensor`, and `max_iter = 0` returns the start factors unchanged.  A
    // start computed from a permuted layout would miss the true column spaces of the
    // factors of modes 1 and 2 (mode 0 only needs the rows of the unfolding).
    let (dense, truth) = rank_two_fixture();
    let [a, b, c] = &truth;
    let lambdas = [3.0, 1.5];
    let shape = [a.len(), b.len(), c.len()];
    let mut x = Vec::with_capacity(shape.iter().product());
    for ai in a {
        for bj in b {
            for ck in c {
                x.push(
                    (0..lambdas.len())
                        .map(|r| lambdas[r] * ai[r] * bj[r] * ck[r])
                        .sum::<f64>(),
                );
            }
        }
    }
    assert_eq!(x.len(), dense.data.len());
    for (flat, dense_value) in x.iter().zip(&dense.data) {
        assert!((flat - dense_value).abs() < 1e-14, "layouts differ");
    }

    let flat_start = TensorDecomposition::cp_als_with_init(&x, &shape, 2, 0, 1e-10, CpInit::Hosvd);
    assert_eq!(flat_start.iterations, 0);
    for (mode, (rows, true_factor)) in flat_start.factors.iter().zip(&truth).enumerate() {
        assert_eq!(rows.len(), shape[mode] * 2, "mode {mode}: row-major factor");
        let basis: Vec<Vec<f64>> = (0..2)
            .map(|r| rows.chunks(2).map(|row| row[r]).collect())
            .collect();
        for r in 0..2 {
            assert_in_span(
                &basis,
                &column(true_factor, r),
                &format!("flat start, mode {mode}, true column {r}"),
            );
        }
    }

    // The dense entry point starts from the very same factors.
    let dense_start = cp_als_with_init(&dense, 2, 0, 1e-10, CpInit::Hosvd);
    assert_eq!(dense_start.iterations, 0);
    for (flat, dense_factor) in
        flat_start
            .factors
            .iter()
            .zip([&dense_start.a, &dense_start.b, &dense_start.c])
    {
        let rows: Vec<f64> = dense_factor.iter().flatten().copied().collect();
        assert_eq!(flat.len(), rows.len());
        for (f, d) in flat.iter().zip(&rows) {
            assert!((f - d).abs() < 1e-12, "flat start {f} != dense start {d}");
        }
    }
}

#[test]
fn flat_hosvd_start_needs_fewer_sweeps_than_random_restarts_on_average() {
    // ALS from a random start is fast on easy tensors and falls into swamps on
    // hard ones; the HOSVD start is deterministic, about as fast as a typical
    // random start and never one of the bad draws.  Compare it with the average
    // over sixteen restarts (`CpInit::Random`, seeds 1..=16) on one tensor with
    // uncorrelated factor columns (where some restarts get stuck for thousands of
    // sweeps) and on tensors whose factor columns are correlated.
    let cases = [
        ([6, 7, 8], 3, 61, 0.0),
        ([6, 7, 8], 3, 61, 0.5),
        ([6, 7, 8], 3, 62, 0.5),
        ([8, 8, 8], 4, 63, 0.5),
        ([5, 6, 7], 3, 64, 0.5),
    ];
    for (shape, rank, seed, correlation) in cases {
        let tensor = collinear_low_rank(shape, rank, seed, correlation);
        let hosvd = TensorDecomposition::cp_als(&tensor.data, &shape, rank, 4000, 1e-10);
        let restarts: Vec<usize> = (1..=16)
            .map(|restart| {
                TensorDecomposition::cp_als_with_init(
                    &tensor.data,
                    &shape,
                    rank,
                    4000,
                    1e-10,
                    CpInit::Random { seed: restart },
                )
                .iterations
            })
            .collect();
        let mean = restarts.iter().sum::<usize>() as f64 / restarts.len() as f64;
        assert!(
            hosvd.iterations < 4000,
            "{shape:?} rank {rank} seed {seed} corr {correlation}: HOSVD start did not converge"
        );
        assert!(
            (hosvd.iterations as f64) < mean,
            "{shape:?} rank {rank} seed {seed} corr {correlation}: HOSVD start {} sweeps vs {mean} on average over restarts {restarts:?}",
            hosvd.iterations
        );
    }
}
