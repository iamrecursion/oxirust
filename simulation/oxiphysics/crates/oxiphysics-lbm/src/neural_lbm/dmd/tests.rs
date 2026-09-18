// Copyright 2026 COOLJAPAN OU (Team KitaSan)
// SPDX-License-Identifier: Apache-2.0

//! Tests of the exact dynamic mode decomposition against linear systems with
//! known spectra.

use super::*;
use std::f64::consts::PI;

fn mat_vec(a: &[Vec<f64>], x: &[f64]) -> Vec<f64> {
    a.iter()
        .map(|row| row.iter().zip(x.iter()).map(|(&p, &q)| p * q).sum())
        .collect()
}

/// Snapshots `y_k = A^k y_0`, `k = 0..count`.
fn propagate(a: &[Vec<f64>], y0: &[f64], count: usize) -> Vec<Vec<f64>> {
    let mut states = Vec::with_capacity(count);
    let mut y = y0.to_vec();
    for _ in 0..count {
        states.push(y.clone());
        y = mat_vec(a, &y);
    }
    states
}

/// Fixed deterministic `size x rank` embedding of a low-dimensional state into a
/// field-sized vector.
fn embedding(size: usize, rank: usize) -> Vec<Vec<f64>> {
    (0..size)
        .map(|i| {
            (0..rank)
                .map(|r| {
                    (0.37 * (i + 1) as f64 * (r + 1) as f64).sin()
                        + 0.2 * (1.3 * i as f64 + r as f64).cos()
                })
                .collect()
        })
        .collect()
}

fn embed(e: &[Vec<f64>], y: &[f64]) -> Vec<f64> {
    mat_vec(e, y)
}

fn damped_rotation(rho: f64, theta: f64) -> Vec<Vec<f64>> {
    let (c, s) = (rho * theta.cos(), rho * theta.sin());
    vec![vec![c, -s], vec![s, c]]
}

/// Block-diagonal `diag(rho R(theta), 0.8, -0.6)`.
fn mixed_system(rho: f64, theta: f64) -> Vec<Vec<f64>> {
    let rot = damped_rotation(rho, theta);
    vec![
        vec![rot[0][0], rot[0][1], 0.0, 0.0],
        vec![rot[1][0], rot[1][1], 0.0, 0.0],
        vec![0.0, 0.0, 0.8, 0.0],
        vec![0.0, 0.0, 0.0, -0.6],
    ]
}

fn fit(snapshots: Vec<Vec<f64>>) -> DataDrivenLbm {
    let mut ddl = DataDrivenLbm::new();
    for s in snapshots {
        ddl.add_snapshot(s);
    }
    ddl.dmd_modes().unwrap();
    ddl
}

fn assert_has_eigenvalue(ddl: &DataDrivenLbm, want: Complex, tol: f64) {
    assert!(
        ddl.eigenvalues.iter().any(|&z| (z - want).norm() < tol),
        "eigenvalue {want:?} missing from {:?}",
        ddl.eigenvalues
    );
}

fn norm(v: &[f64]) -> f64 {
    v.iter().map(|x| x * x).sum::<f64>().sqrt()
}

fn distance(a: &[f64], b: &[f64]) -> f64 {
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| (x - y).powi(2))
        .sum::<f64>()
        .sqrt()
}

#[test]
fn damped_rotation_eigenvalues_are_recovered() {
    let (rho, theta) = (0.93, 0.35);
    let a = damped_rotation(rho, theta);
    let ddl = fit(propagate(&a, &[1.0, 0.4], 12));
    assert_eq!(ddl.eigenvalues.len(), 2);
    assert_eq!(ddl.modes.len(), 2);
    assert_eq!(ddl.amplitudes.len(), 2);
    assert_eq!(ddl.singular_values.len(), 2);
    // The eigenvalues are complex: rho exp(+- i theta).
    assert_has_eigenvalue(
        &ddl,
        Complex::new(rho * theta.cos(), rho * theta.sin()),
        1e-10,
    );
    assert_has_eigenvalue(
        &ddl,
        Complex::new(rho * theta.cos(), -rho * theta.sin()),
        1e-10,
    );
    for z in &ddl.eigenvalues {
        assert!((z.norm() - rho).abs() < 1e-10, "|lambda| = {}", z.norm());
        assert!(
            z.im.abs() > 0.3,
            "the eigenvalues must be genuinely complex"
        );
    }
}

#[test]
fn rank_two_data_in_a_nine_dimensional_field_retains_exactly_two_modes() {
    let e = embedding(9, 2);
    let a = damped_rotation(0.97, 0.5);
    let snapshots: Vec<Vec<f64>> = propagate(&a, &[1.0, -0.3], 9)
        .iter()
        .map(|y| embed(&e, y))
        .collect();
    let ddl = fit(snapshots);
    assert_eq!(ddl.modes.len(), 2, "the numerical rank of the data is 2");
    assert_eq!(ddl.singular_values.len(), 2);
    assert!(ddl.singular_values[0] >= ddl.singular_values[1]);
    assert!(ddl.modes.iter().all(|m| m.len() == 9));
    assert_has_eigenvalue(
        &ddl,
        Complex::new(0.97 * 0.5_f64.cos(), 0.97 * 0.5_f64.sin()),
        1e-9,
    );
    assert_has_eigenvalue(
        &ddl,
        Complex::new(0.97 * 0.5_f64.cos(), -0.97 * 0.5_f64.sin()),
        1e-9,
    );
}

#[test]
fn modes_are_unit_vectors_and_eigenvalues_are_ordered_by_modulus() {
    let e = embedding(15, 4);
    let a = mixed_system(0.95, 0.6);
    let snapshots: Vec<Vec<f64>> = propagate(&a, &[1.0, -0.5, 0.7, 0.4], 20)
        .iter()
        .map(|y| embed(&e, y))
        .collect();
    let ddl = fit(snapshots);
    assert_eq!(ddl.eigenvalues.len(), 4);
    for mode in &ddl.modes {
        let length = mode.iter().map(|z| z.norm_sq()).sum::<f64>().sqrt();
        assert!((length - 1.0).abs() < 1e-12, "mode length {length}");
    }
    let moduli: Vec<f64> = ddl.eigenvalues.iter().map(|z| z.norm()).collect();
    assert!(
        moduli.windows(2).all(|w| w[0] >= w[1] - 1e-12),
        "{moduli:?}"
    );
    // Real data: the complex eigenvalues come in conjugate pairs.
    for &z in &ddl.eigenvalues {
        if z.im.abs() > 1e-9 {
            assert_has_eigenvalue(&ddl, z.conj(), 1e-9);
        }
    }
    assert_has_eigenvalue(&ddl, Complex::new(0.8, 0.0), 1e-9);
    assert_has_eigenvalue(&ddl, Complex::new(-0.6, 0.0), 1e-9);
    assert_has_eigenvalue(
        &ddl,
        Complex::new(0.95 * 0.6_f64.cos(), 0.95 * 0.6_f64.sin()),
        1e-9,
    );
}

#[test]
fn the_rom_reproduces_the_training_snapshots() {
    let e = embedding(15, 4);
    let a = mixed_system(0.95, 0.6);
    let snapshots: Vec<Vec<f64>> = propagate(&a, &[1.0, -0.5, 0.7, 0.4], 20)
        .iter()
        .map(|y| embed(&e, y))
        .collect();
    let ddl = fit(snapshots.clone());
    for (k, snapshot) in snapshots.iter().enumerate() {
        let predicted = ddl.rom_predict(k);
        assert_eq!(predicted.len(), 15);
        let error = distance(&predicted, snapshot);
        assert!(
            error < 1e-9 * norm(&snapshots[0]),
            "snapshot {k}: error {error}"
        );
        assert!(ddl.reconstruction_error(k) < 1e-9 * norm(&snapshots[0]));
    }
}

#[test]
fn the_rom_extrapolates_the_linear_dynamics() {
    let e = embedding(15, 4);
    let a = mixed_system(0.95, 0.6);
    let y0 = [1.0, -0.5, 0.7, 0.4];
    let all = propagate(&a, &y0, 61);
    let ddl = fit(all[..20].iter().map(|y| embed(&e, y)).collect());
    for k in [20_usize, 25, 40, 60] {
        let truth = embed(&e, &all[k]);
        let error = distance(&ddl.rom_predict(k), &truth);
        assert!(error < 1e-9, "step {k}: extrapolation error {error}");
    }
}

#[test]
fn a_two_dimensional_rotation_extrapolates_to_a_k_fold_rotation() {
    // x_k = A^k x_0 for the damped rotation: x_{k} - A^{k} x_0 must vanish far
    // beyond the training window as well.
    let (rho, theta) = (0.99, 0.2);
    let a = damped_rotation(rho, theta);
    let x0 = [0.7, -1.3];
    let all = propagate(&a, &x0, 101);
    let ddl = fit(all[..10].to_vec());
    let predicted = ddl.rom_predict(100);
    assert!(
        distance(&predicted, &all[100]) < 1e-9,
        "{predicted:?} vs {:?}",
        all[100]
    );
}

#[test]
fn rank_cap_truncates_the_decomposition() {
    let e = embedding(15, 4);
    let a = mixed_system(0.95, 0.6);
    let mut ddl = DataDrivenLbm::new();
    for y in propagate(&a, &[1.0, -0.5, 0.7, 0.4], 20) {
        ddl.add_snapshot(embed(&e, &y));
    }
    ddl.dmd_modes_with_rank(2).unwrap();
    assert_eq!(ddl.modes.len(), 2);
    assert_eq!(ddl.singular_values.len(), 2);
    assert_eq!(ddl.eigenvalues.len(), 2);
    let predicted = ddl.rom_predict(3);
    assert_eq!(predicted.len(), 15);
    assert!(predicted.iter().all(|v| v.is_finite()));
    // A cap above the numerical rank changes nothing.
    ddl.dmd_modes_with_rank(50).unwrap();
    assert_eq!(ddl.modes.len(), 4);
    assert!(matches!(
        ddl.dmd_modes_with_rank(0),
        Err(Error::ParameterOutOfBounds { .. })
    ));
}

#[test]
fn periodic_translation_has_unit_circle_eigenvalues() {
    // A pattern advected by one cell per step on a periodic line: the propagator
    // is a cyclic shift, whose DMD eigenvalues are the roots of unity exp(-+ 2 pi i k / 16).
    let size = 16_usize;
    let pattern = |j: f64| {
        (2.0 * PI * j / 16.0).cos()
            + 0.5 * (2.0 * PI * 2.0 * j / 16.0 + 0.3).sin()
            + 0.2 * (2.0 * PI * 3.0 * j / 16.0).sin()
    };
    let snapshot =
        |k: usize| -> Vec<f64> { (0..size).map(|j| pattern(j as f64 - k as f64)).collect() };
    let ddl = fit((0..14).map(snapshot).collect());
    assert_eq!(ddl.eigenvalues.len(), 6, "three harmonics -> six modes");
    for k in 1..=3 {
        let phase = 2.0 * PI * k as f64 / 16.0;
        assert_has_eigenvalue(&ddl, Complex::new(phase.cos(), phase.sin()), 1e-8);
        assert_has_eigenvalue(&ddl, Complex::new(phase.cos(), -phase.sin()), 1e-8);
    }
    for z in &ddl.eigenvalues {
        assert!((z.norm() - 1.0).abs() < 1e-8);
    }
    let error = distance(&ddl.rom_predict(40), &snapshot(40));
    assert!(
        error < 1e-8,
        "advected pattern after 40 steps: error {error}"
    );
}

#[test]
fn stationary_data_has_a_unit_eigenvalue() {
    let ddl = fit(vec![vec![1.0, 2.0, 3.0]; 5]);
    assert_eq!(ddl.eigenvalues.len(), 1);
    assert!((ddl.eigenvalues[0] - Complex::one()).norm() < 1e-12);
    let far = ddl.rom_predict(100);
    assert!(distance(&far, &[1.0, 2.0, 3.0]) < 1e-10);
}

#[test]
fn two_snapshots_give_a_single_real_mode() {
    let ddl = fit(vec![vec![1.0, 0.0, 0.0], vec![0.9, 0.1, 0.0]]);
    assert_eq!(ddl.modes.len(), 1);
    // lambda = (x0 . x1) / (x0 . x0)
    assert!((ddl.eigenvalues[0] - Complex::new(0.9, 0.0)).norm() < 1e-14);
    // The exact mode is parallel to the second snapshot.
    let mode = &ddl.modes[0];
    let reference = norm(&[0.9, 0.1, 0.0]);
    let expected = [0.9 / reference, 0.1 / reference, 0.0];
    let alignment: f64 = mode
        .iter()
        .zip(expected.iter())
        .map(|(z, &e)| z.re * e)
        .sum::<f64>()
        .abs();
    assert!((alignment - 1.0).abs() < 1e-12, "alignment {alignment}");
}

#[test]
fn data_without_signal_yields_an_empty_decomposition_and_a_zero_forecast() {
    let ddl = fit(vec![vec![0.0; 4]; 3]);
    assert!(ddl.modes.is_empty() && ddl.eigenvalues.is_empty() && ddl.amplitudes.is_empty());
    assert_eq!(ddl.rom_predict(2), vec![0.0; 4]);
    assert_eq!(ddl.reconstruction_error(1), 0.0);
}

#[test]
fn affine_growth_with_a_defective_propagator_stays_finite() {
    // x_k = a + k b: the propagator on span{a, b} is a Jordan block with the
    // double eigenvalue 1.  The decomposition must neither fail nor blow up.
    let mut ddl = DataDrivenLbm::new();
    for k in 0..5 {
        ddl.add_snapshot((0..9).map(|i| (i + k) as f64 * 0.1).collect());
    }
    ddl.dmd_modes().unwrap();
    assert_eq!(ddl.modes.len(), 2);
    for z in &ddl.eigenvalues {
        assert!((z.norm() - 1.0).abs() < 1e-6, "|lambda| = {}", z.norm());
    }
    let predicted = ddl.rom_predict(2);
    assert_eq!(predicted.len(), 9);
    assert!(predicted.iter().all(|v| v.is_finite()));
    let truth: Vec<f64> = (0..9).map(|i| (i + 2) as f64 * 0.1).collect();
    assert!(distance(&predicted, &truth) < 1e-5);
}

#[test]
fn invalid_snapshot_sets_are_rejected() {
    let mut ddl = DataDrivenLbm::new();
    assert!(matches!(
        ddl.dmd_modes(),
        Err(Error::IncompatibleParameters { .. })
    ));
    ddl.add_snapshot(vec![1.0, 2.0]);
    assert!(matches!(
        ddl.dmd_modes(),
        Err(Error::IncompatibleParameters { .. })
    ));
    ddl.add_snapshot(vec![1.0, 2.0, 3.0]);
    assert!(matches!(
        ddl.dmd_modes(),
        Err(Error::ComponentCountMismatch {
            expected: 2,
            found: 3,
            ..
        })
    ));
    let mut nan = DataDrivenLbm::new();
    nan.add_snapshot(vec![1.0, 2.0]);
    nan.add_snapshot(vec![1.0, f64::NAN]);
    assert!(matches!(
        nan.dmd_modes(),
        Err(Error::NonFiniteField { index: 1, .. })
    ));
    let mut empty = DataDrivenLbm::new();
    empty.add_snapshot(Vec::new());
    empty.add_snapshot(Vec::new());
    assert!(empty.dmd_modes().is_err());
}

#[test]
fn snapshots_whose_gram_matrix_overflows_are_reported() {
    let mut ddl = DataDrivenLbm::new();
    ddl.add_snapshot(vec![1.0e200, 1.0e200]);
    ddl.add_snapshot(vec![1.0e200, -1.0e200]);
    assert!(matches!(ddl.dmd_modes(), Err(Error::NonFiniteField { .. })));
    assert!(ddl.modes.is_empty());
}

#[test]
fn a_failed_decomposition_leaves_the_previous_one_untouched() {
    let a = damped_rotation(0.9, 0.4);
    let mut ddl = fit(propagate(&a, &[1.0, 0.0], 8));
    let eigenvalues = ddl.eigenvalues.clone();
    let before = ddl.rom_predict(3);
    ddl.add_snapshot(vec![1.0, 2.0, 3.0]);
    assert!(ddl.dmd_modes().is_err());
    assert_eq!(ddl.eigenvalues, eigenvalues);
    assert_eq!(ddl.rom_predict(3), before);
}

#[test]
fn snapshots_added_after_the_fit_do_not_change_the_rom() {
    let a = damped_rotation(0.9, 0.4);
    let all = propagate(&a, &[1.0, 0.5], 9);
    let mut ddl = fit(all[..8].to_vec());
    let before = ddl.rom_predict(8);
    ddl.add_snapshot(all[8].clone());
    assert_eq!(ddl.snapshot_count(), 9);
    assert_eq!(ddl.rom_predict(8), before);
    // The ninth snapshot was not used for the fit but the ROM predicts it.
    assert!(ddl.reconstruction_error(8) < 1e-9);
}

#[test]
fn reconstruction_error_is_nan_without_a_decomposition() {
    let mut ddl = DataDrivenLbm::new();
    assert!(ddl.reconstruction_error(0).is_nan());
    assert!(ddl.rom_predict(3).is_empty());
    ddl.add_snapshot(vec![1.0, 2.0]);
    assert!(ddl.reconstruction_error(0).is_nan());
    ddl.add_snapshot(vec![0.9, 1.9]);
    ddl.dmd_modes().unwrap();
    assert!(ddl.reconstruction_error(5).is_nan());
    assert!(ddl.reconstruction_error(0).is_finite());
}
