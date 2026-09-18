// Copyright 2026 COOLJAPAN OU (Team KitaSan)
// SPDX-License-Identifier: Apache-2.0

//! Regression tests for the CP-ALS decomposition: Khatri–Rao operand order
//! against the `mode_n_unfold` layout, and the weights of an unconverged run.

use super::*;

/// Build `Σ_r λ_r a_r ∘ b_r ∘ c_r` element by element, independently of
/// `cp_reconstruct`.
fn cp_tensor(a: &[Vec<f64>], b: &[Vec<f64>], c: &[Vec<f64>], lambdas: &[f64]) -> DenseTensor {
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

fn column_norm(factor: &[Vec<f64>], r: usize) -> f64 {
    factor.iter().map(|row| row[r] * row[r]).sum::<f64>().sqrt()
}

fn rank_two_fixture() -> DenseTensor {
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
    cp_tensor(&a, &b, &c, &[3.0, 1.5])
}

fn rank_one_fixture() -> (DenseTensor, f64) {
    let a = vec![vec![1.0], vec![2.0], vec![-1.0]];
    let b = vec![vec![0.5], vec![1.0], vec![1.5], vec![-2.0]];
    let c = vec![vec![1.0], vec![-1.0], vec![2.0], vec![0.5], vec![3.0]];
    let norm_product = column_norm(&a, 0) * column_norm(&b, 0) * column_norm(&c, 0);
    (cp_tensor(&a, &b, &c, &[10.0]), 10.0 * norm_product)
}

#[test]
fn mode_n_unfold_varies_the_last_remaining_index_fastest() {
    // cp_als pairs each unfolding with a Khatri-Rao product in exactly this
    // column order; if the layout ever changes, cp_als must change with it.
    let (n0, n1, n2) = (2usize, 3usize, 4usize);
    let data: Vec<f64> = (0..n0 * n1 * n2).map(|v| v as f64).collect();
    let tensor = DenseTensor::from_data(&[n0, n1, n2], data);
    let x0 = tensor.mode_n_unfold(0);
    let x1 = tensor.mode_n_unfold(1);
    let x2 = tensor.mode_n_unfold(2);
    for i in 0..n0 {
        for j in 0..n1 {
            for k in 0..n2 {
                let value = tensor.get(&[i, j, k]);
                assert_eq!(x0[i][j * n2 + k], value, "mode 0 at ({i},{j},{k})");
                assert_eq!(x1[j][i * n2 + k], value, "mode 1 at ({i},{j},{k})");
                assert_eq!(x2[k][i * n1 + j], value, "mode 2 at ({i},{j},{k})");
            }
        }
    }
}

#[test]
fn cp_als_recovers_a_non_symmetric_rank_two_tensor() {
    // Shape 3x4x5 with different factor matrices in every mode: the swapped
    // Khatri-Rao operands of the old code gave a relative error of about 1.
    let tensor = rank_two_fixture();
    let cp = cp_als(&tensor, 2, 500, 1e-14);
    let rel = cp_relative_error(&tensor, &cp);
    assert!(rel < 1e-8, "relative error {rel}");
    assert_eq!((cp.a.len(), cp.b.len(), cp.c.len()), (3, 4, 5));
    assert_eq!(cp.lambdas.len(), 2);
    for r in 0..2 {
        for factor in [&cp.a, &cp.b, &cp.c] {
            assert!((column_norm(factor, r) - 1.0).abs() < 1e-12);
        }
    }
}

#[test]
fn cp_als_reports_convergence_for_an_exact_fit() {
    let tensor = rank_two_fixture();
    let cp = cp_als(&tensor, 2, 2000, 1e-10);
    assert!(
        cp.converged,
        "an exactly representable tensor must converge"
    );
    assert!(cp.iterations < 2000, "iterations {}", cp.iterations);
    assert!(cp_relative_error(&tensor, &cp) < 1e-9);
}

#[test]
fn cp_als_keeps_the_weights_when_max_iter_is_reached() {
    let (tensor, expected_weight) = rank_one_fixture();
    // tol = 0.0 can never be met, so the run ends by exhausting `max_iter`.
    let cp = cp_als(&tensor, 1, 2, 0.0);
    assert!(!cp.converged);
    assert_eq!(cp.iterations, 2);
    assert!(
        (cp.lambdas[0] - expected_weight).abs() < 1e-9 * expected_weight,
        "weight {} vs {expected_weight}",
        cp.lambdas[0]
    );
    let rel = cp_relative_error(&tensor, &cp);
    assert!(rel < 1e-6, "relative error {rel}");
}

#[test]
fn cp_als_single_sweep_already_returns_scaled_factors() {
    let (tensor, expected_weight) = rank_one_fixture();
    let cp = cp_als(&tensor, 1, 1, 0.0);
    assert!(!cp.converged);
    assert_eq!(cp.iterations, 1);
    assert!((cp.lambdas[0] - expected_weight).abs() < 1e-9 * expected_weight);
    assert!(cp_relative_error(&tensor, &cp) < 1e-6);
}

#[test]
fn cp_als_flags_an_unreachable_tolerance_as_not_converged() {
    let tensor = rank_two_fixture();
    // Three sweeps cannot reach a relative error of 1e-300.
    let cp = cp_als(&tensor, 2, 3, 1e-300);
    assert!(!cp.converged);
    assert_eq!(cp.iterations, 3);
    assert_eq!(cp.rank(), 2);
}

#[test]
fn cp_als_zero_sweeps_returns_the_initial_guess_unconverged() {
    let (tensor, _) = rank_one_fixture();
    let cp = cp_als(&tensor, 1, 0, 1e-8);
    assert!(!cp.converged);
    assert_eq!(cp.iterations, 0);
    assert_eq!(cp.lambdas, vec![1.0]);
}
