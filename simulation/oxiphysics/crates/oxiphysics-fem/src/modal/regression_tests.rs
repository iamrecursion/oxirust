// Copyright 2026 COOLJAPAN OU (Team KitaSan)
// SPDX-License-Identifier: Apache-2.0

//! Regression tests of the modal-analysis audit findings: degenerate input of the
//! eigen solvers (F1244), the over-/critically damped modal superposition (F1249)
//! and the modal truncation against the total mass (F1251).

use super::sdof::advance_sdof;
use super::*;
use crate::sparse::CsrMatrix;

fn identity_csr(n: usize) -> CsrMatrix {
    diagonal_csr(&vec![1.0; n])
}

fn diagonal_csr(diag: &[f64]) -> CsrMatrix {
    let triplets: Vec<(usize, usize, f64)> =
        diag.iter().enumerate().map(|(i, &v)| (i, i, v)).collect();
    CsrMatrix::from_triplets(diag.len(), diag.len(), &triplets)
}

/// Fixed-fixed chain: `2k` on the diagonal, `-k` next to it.
fn spring_chain_csr(n: usize, k: f64) -> CsrMatrix {
    let mut triplets = Vec::new();
    for i in 0..n {
        triplets.push((i, i, 2.0 * k));
        if i + 1 < n {
            triplets.push((i, i + 1, -k));
            triplets.push((i + 1, i, -k));
        }
    }
    CsrMatrix::from_triplets(n, n, &triplets)
}

/// Analytic mass-normalised eigenpairs `(ω_j, φ_j)` of the fixed-fixed chain with
/// `M = I`: `ω_j² = 2k (1 − cos(jπ/(n+1)))`, `φ_j(i) = √(2/(n+1)) sin(i j π/(n+1))`.
fn spring_chain_modes(n: usize, k: f64) -> (Vec<f64>, Vec<Vec<f64>>) {
    let frequencies = (1..=n)
        .map(|j| {
            let angle = j as f64 * PI / (n + 1) as f64;
            (2.0 * k * (1.0 - angle.cos())).sqrt()
        })
        .collect();
    let shapes = (1..=n)
        .map(|j| {
            (1..=n)
                .map(|i| {
                    (2.0 / (n + 1) as f64).sqrt() * ((i * j) as f64 * PI / (n + 1) as f64).sin()
                })
                .collect()
        })
        .collect();
    (frequencies, shapes)
}

fn max_abs_diff(a: &[f64], b: &[f64]) -> f64 {
    a.iter()
        .zip(b.iter())
        .fold(0.0_f64, |m, (x, y)| m.max((x - y).abs()))
}

// ---------------------------------------------------------------------------
// F1244: degenerate input of the eigen solvers and of the normalisation
// ---------------------------------------------------------------------------

#[test]
fn eigen_solvers_reject_an_empty_system() {
    // `mode_idx % n` used to panic with n == 0, `&beta[1..dim]` with dim == 0.
    let empty = CsrMatrix::new(0, 0);
    for n_modes in [0, 1] {
        assert_eq!(
            inverse_iteration(&empty, &empty, n_modes, 10, 1e-8).err(),
            Some(ModalError::EmptySystem)
        );
        assert_eq!(
            lanczos_iteration(&empty, &empty, n_modes, 10, 1e-8).err(),
            Some(ModalError::EmptySystem)
        );
    }
}

#[test]
fn eigen_solvers_report_dimension_mismatches_instead_of_panicking() {
    let k = diagonal_csr(&[1.0, 4.0, 9.0]);
    let mismatch = |got| Some(ModalError::DimensionMismatch { expected: 3, got });

    // mass matrix of the wrong size (the `assert_eq!` of the old code)
    let m2 = identity_csr(2);
    assert_eq!(inverse_iteration(&k, &m2, 1, 10, 1e-8).err(), mismatch(2));
    assert_eq!(lanczos_iteration(&k, &m2, 1, 10, 1e-8).err(), mismatch(2));

    // non-square stiffness matrix (3 x 2)
    let rectangular = CsrMatrix::from_triplets(3, 2, &[(0, 0, 1.0), (1, 1, 1.0)]);
    let m3 = identity_csr(3);
    assert_eq!(
        inverse_iteration(&rectangular, &m3, 1, 10, 1e-8).err(),
        mismatch(2)
    );
    assert_eq!(
        lanczos_iteration(&rectangular, &m3, 1, 10, 1e-8).err(),
        mismatch(2)
    );

    // non-square mass matrix (3 x 4)
    let wide_mass = CsrMatrix::from_triplets(3, 4, &[(0, 0, 1.0)]);
    assert_eq!(
        inverse_iteration(&k, &wide_mass, 1, 10, 1e-8).err(),
        mismatch(4)
    );
}

#[test]
fn eigen_solvers_reject_more_modes_than_degrees_of_freedom() {
    // With n_modes > n the deflation removed the whole space, the loop broke out and
    // modes with ω = 0 were pushed anyway.
    let k = diagonal_csr(&[1.0, 4.0]);
    let m = identity_csr(2);
    let expected = Some(ModalError::TooManyModes { requested: 3, n: 2 });
    assert_eq!(inverse_iteration(&k, &m, 3, 200, 1e-8).err(), expected);
    assert_eq!(lanczos_iteration(&k, &m, 3, 10, 1e-8).err(), expected);

    // asking for exactly n modes still works and yields no ω = 0 mode
    let result = inverse_iteration(&k, &m, 2, 500, 1e-10).expect("two modes of a 2-DOF system");
    assert_eq!(result.frequencies.len(), 2);
    assert!(
        (result.frequencies[0] - 1.0).abs() < 1e-6,
        "{:?}",
        result.frequencies
    );
    assert!(
        (result.frequencies[1] - 2.0).abs() < 1e-6,
        "{:?}",
        result.frequencies
    );
}

#[test]
fn inverse_iteration_that_cannot_certify_convergence_is_an_error() {
    // Convergence needs two iterations (`rel_change` of the second one), so a limit of
    // zero or one iteration can never deliver a mode: it used to push a garbage mode
    // (ω = 0 for max_iter = 0).
    let k = diagonal_csr(&[1.0, 4.0, 9.0]);
    let m = identity_csr(3);
    for max_iter in [0, 1] {
        assert_eq!(
            inverse_iteration(&k, &m, 1, max_iter, 1e-8).err(),
            Some(ModalError::NotConverged { mode: 0 }),
            "max_iter = {max_iter}"
        );
    }
    // an unreachable tolerance is reported as well
    assert_eq!(
        inverse_iteration(&k, &m, 1, 40, 0.0).err(),
        Some(ModalError::NotConverged { mode: 0 })
    );
}

#[test]
fn inverse_iteration_with_a_collapsing_iterate_pushes_no_mode() {
    // A zero mass matrix: the M-norm of every iterate vanishes.
    let k = diagonal_csr(&[1.0, 4.0, 9.0]);
    let zero_mass = CsrMatrix::new(3, 3);
    assert_eq!(
        inverse_iteration(&k, &zero_mass, 1, 100, 1e-8).err(),
        Some(ModalError::NotConverged { mode: 0 })
    );
}

#[test]
fn lanczos_never_returns_fewer_pairs_than_requested() {
    // K = 2 I has a single distinct eigenvalue: the Krylov space of one start vector
    // is one-dimensional, so one Ritz pair exists and the 2nd mode is missing.
    let k = diagonal_csr(&[2.0, 2.0, 2.0]);
    let m = identity_csr(3);
    assert_eq!(
        lanczos_iteration(&k, &m, 3, 10, 1e-8).err(),
        Some(ModalError::NotConverged { mode: 1 })
    );
    // one mode is available
    let one = lanczos_iteration(&k, &m, 1, 10, 1e-8).expect("one Ritz pair exists");
    assert_eq!(one.eigenvalues.len(), 1);
    assert!((one.eigenvalues[0] - 2.0_f64.sqrt()).abs() < 1e-8);
}

#[test]
fn zero_modes_requested_gives_an_empty_result() {
    let k = diagonal_csr(&[1.0, 4.0]);
    let m = identity_csr(2);
    let inverse = inverse_iteration(&k, &m, 0, 10, 1e-8).expect("nothing to compute");
    assert!(inverse.frequencies.is_empty() && inverse.mode_shapes.is_empty());
    let lanczos = lanczos_iteration(&k, &m, 0, 10, 1e-8).expect("nothing to compute");
    assert!(lanczos.eigenvalues.is_empty() && lanczos.eigenvectors.is_empty());
    assert_eq!(lanczos.n_steps, 0);
}

#[test]
fn solvers_still_find_the_spring_chain_spectrum() {
    // Positive control for the checks above: a regular problem is unaffected.
    let n = 6;
    let k = spring_chain_csr(n, 3.0);
    let m = identity_csr(n);
    let (omega, _) = spring_chain_modes(n, 3.0);
    let inverse = inverse_iteration(&k, &m, 3, 2000, 1e-10).expect("converges");
    let lanczos = lanczos_iteration(&k, &m, 3, 2 * n, 1e-12).expect("full Krylov space");
    for (j, &exact) in omega.iter().enumerate().take(3) {
        assert!(
            (inverse.frequencies[j] - exact).abs() < 1e-6 * exact,
            "inverse iteration mode {j}: {} vs {exact}",
            inverse.frequencies[j]
        );
        assert!(
            (lanczos.eigenvalues[j] - exact).abs() < 1e-6 * exact,
            "lanczos mode {j}: {} vs {exact}",
            lanczos.eigenvalues[j]
        );
    }
}

#[test]
fn normalize_mode_shape_requires_a_mass_matrix() {
    // `mass.expect(..)` used to panic.
    let mut phi = vec![1.0, 2.0];
    assert_eq!(
        normalize_mode_shape(&mut phi, NormalizationMethod::Mass, None),
        Err(ModalError::MassMatrixRequired)
    );
    assert_eq!(phi, vec![1.0, 2.0], "the mode is left untouched on error");
}

#[test]
fn normalize_mode_shape_checks_sizes_and_the_generalised_mass() {
    let mut phi = vec![1.0, 2.0, 3.0];
    let m2 = identity_csr(2);
    assert_eq!(
        normalize_mode_shape(&mut phi, NormalizationMethod::Mass, Some(&m2)),
        Err(ModalError::DimensionMismatch {
            expected: 3,
            got: 2
        })
    );
    let mut empty: Vec<f64> = Vec::new();
    assert_eq!(
        normalize_mode_shape(
            &mut empty,
            NormalizationMethod::Mass,
            Some(&CsrMatrix::new(0, 0))
        ),
        Err(ModalError::EmptySystem)
    );

    // A mass matrix that is not positive definite, and a zero mode: φᵀMφ cannot be 1.
    let indefinite = diagonal_csr(&[1.0, -1.0]);
    let mut null_mode = vec![1.0, 1.0]; // φᵀMφ = 0
    assert_eq!(
        normalize_mode_shape(&mut null_mode, NormalizationMethod::Mass, Some(&indefinite)),
        Err(ModalError::NonPositiveGeneralizedMass { value: 0.0 })
    );
    assert_eq!(null_mode, vec![1.0, 1.0]);
    let mut negative = vec![1.0, 2.0]; // φᵀMφ = 1 − 4
    assert_eq!(
        normalize_mode_shape(&mut negative, NormalizationMethod::Mass, Some(&indefinite)),
        Err(ModalError::NonPositiveGeneralizedMass { value: -3.0 })
    );
    let mut zero = vec![0.0, 0.0];
    assert_eq!(
        normalize_mode_shape(&mut zero, NormalizationMethod::Mass, Some(&identity_csr(2))),
        Err(ModalError::NonPositiveGeneralizedMass { value: 0.0 })
    );

    // the regular case: M = diag(2, 3), φ = (1, 1) -> φᵀMφ = 5
    let m = diagonal_csr(&[2.0, 3.0]);
    let mut phi = vec![1.0, 1.0];
    normalize_mode_shape(&mut phi, NormalizationMethod::Mass, Some(&m)).expect("regular");
    let generalised_mass: f64 = dot(&phi, &m.mul_vec(&phi));
    assert!((generalised_mass - 1.0).abs() < 1e-14, "{generalised_mass}");
    // the other methods leave a zero vector alone and need no mass matrix
    let mut zero = vec![0.0, 0.0];
    normalize_mode_shape(&mut zero, NormalizationMethod::UnitLength, None).expect("no-op");
    normalize_mode_shape(&mut zero, NormalizationMethod::MaxComponent, None).expect("no-op");
    assert_eq!(zero, vec![0.0, 0.0]);
}

#[test]
fn modal_errors_convert_into_the_crate_error() {
    use crate::Error;
    let invalid: Error = ModalError::EmptySystem.into();
    assert!(matches!(invalid, Error::InvalidInput(_)), "{invalid:?}");
    let mismatch: Error = ModalError::DimensionMismatch {
        expected: 3,
        got: 2,
    }
    .into();
    assert!(
        mismatch.to_string().contains("expected 3, got 2"),
        "{mismatch}"
    );
    let not_converged: Error = ModalError::NotConverged { mode: 2 }.into();
    assert!(
        not_converged.to_string().contains("mode 2"),
        "{not_converged}"
    );
    let static_solve: Error = ModalError::StaticSolveFailed { residual: 0.5 }.into();
    assert!(
        matches!(static_solve, Error::NotConverged { residual, .. } if residual == 0.5),
        "{static_solve:?}"
    );
    let too_many = ModalError::TooManyModes { requested: 5, n: 3 }.to_string();
    assert!(
        too_many.contains("5 modes") && too_many.contains("3 degrees"),
        "{too_many}"
    );
    assert!(
        ModalError::MassMatrixRequired
            .to_string()
            .contains("mass matrix")
    );
}

// ---------------------------------------------------------------------------
// F1249: exact stepping of critically damped and over-damped modes
// ---------------------------------------------------------------------------

/// Free decay of a critically damped oscillator from `(q0, v0)`.
fn critical_free_decay(q0: f64, v0: f64, omega: f64, t: f64) -> (f64, f64) {
    let e = (-omega * t).exp();
    let drive = v0 + omega * q0;
    (e * (q0 + drive * t), e * (v0 - omega * drive * t))
}

/// Free decay of an over-damped oscillator (`ζ > 1`) as `c₁ e^{s₁t} + c₂ e^{s₂t}`,
/// with the slow root written as `−ω/(ζ + √(ζ²−1))` to avoid cancellation.
fn overdamped_free_decay(q0: f64, v0: f64, omega: f64, zeta: f64, t: f64) -> (f64, f64) {
    let root = (zeta * zeta - 1.0).sqrt();
    let s1 = -omega / (zeta + root);
    let s2 = -omega * (zeta + root);
    let c1 = (v0 - s2 * q0) / (s1 - s2);
    let c2 = q0 - c1;
    (
        c1 * (s1 * t).exp() + c2 * (s2 * t).exp(),
        c1 * s1 * (s1 * t).exp() + c2 * s2 * (s2 * t).exp(),
    )
}

#[test]
fn critically_damped_free_decay_matches_the_analytic_solution() {
    // The old branch used q instead of q − q_s and dropped the −ω² q dt term of q̇.
    let (omega, q0, v0) = (7.3, 0.8, -1.4);
    for dt in [1e-3, 0.01, 0.1, 0.5, 2.0] {
        let (q, v) = advance_sdof(q0, v0, omega, 1.0, 0.0, dt);
        let (q_exact, v_exact) = critical_free_decay(q0, v0, omega, dt);
        assert!(
            (q - q_exact).abs() < 1e-12,
            "dt = {dt}: q = {q}, exact {q_exact}"
        );
        assert!(
            (v - v_exact).abs() < 1e-12,
            "dt = {dt}: v = {v}, exact {v_exact}"
        );
    }
}

#[test]
fn overdamped_free_decay_matches_the_analytic_solution() {
    // For ζ > 1 the old code clamped ω_d to 0 and treated the mode as critically damped.
    let (omega, q0, v0) = (7.3, 0.8, -1.4);
    for zeta in [1.0 + 1e-6, 1.5, 2.0, 10.0, 1e3] {
        for dt in [1e-3, 0.01, 0.1, 0.5, 2.0] {
            let (q, v) = advance_sdof(q0, v0, omega, zeta, 0.0, dt);
            let (q_exact, v_exact) = overdamped_free_decay(q0, v0, omega, zeta, dt);
            assert!(
                (q - q_exact).abs() < 1e-10,
                "zeta = {zeta}, dt = {dt}: q = {q}, exact {q_exact}"
            );
            assert!(
                (v - v_exact).abs() < 1e-9 * (1.0 + v_exact.abs()),
                "zeta = {zeta}, dt = {dt}: v = {v}, exact {v_exact}"
            );
        }
    }
    // ζ = 2 differs markedly from the critical solution (the old, wrong result)
    let (q_over, _) = advance_sdof(q0, v0, omega, 2.0, 0.0, 0.2);
    let (q_critical, _) = critical_free_decay(q0, v0, omega, 0.2);
    assert!(
        (q_over - q_critical).abs() > 1e-2,
        "{q_over} vs {q_critical}"
    );
}

#[test]
fn large_arguments_do_not_overflow_or_lose_the_solution() {
    // ζ ω dt = 3000: a cosh/sinh form overflows (0 · ∞ = NaN), the split form is exact.
    let (q, v) = advance_sdof(1.0, 0.0, 100.0, 3.0, 0.0, 10.0);
    assert!(q.is_finite() && v.is_finite(), "q = {q}, v = {v}");
    let (q_exact, v_exact) = overdamped_free_decay(1.0, 0.0, 100.0, 3.0, 10.0);
    assert!(
        (q - q_exact).abs() <= 1e-12 * q_exact.abs().max(1e-300),
        "{q} vs {q_exact}"
    );
    assert!(
        (v - v_exact).abs() <= 1e-12 * v_exact.abs().max(1e-300),
        "{v} vs {v_exact}"
    );
    // a long step of an under-damped mode decays to (nearly) zero as well
    let (q, v) = advance_sdof(1.0, 0.0, 100.0, 0.5, 0.0, 10.0);
    assert!(q.abs() < 1e-200 && v.abs() < 1e-197, "q = {q}, v = {v}");
}

#[test]
fn the_exact_step_has_the_semigroup_property_in_every_regime() {
    // Two half steps equal one full step (the force is constant over both).
    let (omega, force) = (4.2, 3.5);
    for zeta in [0.0, 0.05, 0.7, 1.0, 1.0 + 1e-9, 1.3, 4.0] {
        let full = advance_sdof(0.3, -0.9, omega, zeta, force, 0.8);
        let half = advance_sdof(0.3, -0.9, omega, zeta, force, 0.4);
        let twice = advance_sdof(half.0, half.1, omega, zeta, force, 0.4);
        assert!(
            (full.0 - twice.0).abs() < 1e-12 && (full.1 - twice.1).abs() < 1e-12,
            "zeta = {zeta}: {full:?} vs {twice:?}"
        );
    }
}

#[test]
fn the_static_equilibrium_is_a_fixed_point_in_every_regime() {
    let (omega, force) = (6.0, 2.5);
    let q_static = force / (omega * omega);
    for zeta in [0.0, 0.3, 1.0, 2.0, 10.0] {
        let (q, v) = advance_sdof(q_static, 0.0, omega, zeta, force, 0.37);
        assert!((q - q_static).abs() < 1e-15, "zeta = {zeta}: q = {q}");
        assert!(v.abs() < 1e-15, "zeta = {zeta}: v = {v}");
    }
}

#[test]
fn the_step_is_continuous_across_critical_damping() {
    // Just outside and inside the critical band the exact formulas agree with the
    // critically damped one to first order in the distance δ from ζ = 1.
    let (omega, q0, v0, dt) = (5.0, 1.0, 0.2, 0.4);
    let reference = advance_sdof(q0, v0, omega, 1.0, 0.0, dt);
    assert!(reference.0.is_finite() && reference.1.is_finite());
    for delta in [1e-15, 1e-13, 1e-9, 1e-6, 1e-4] {
        let above = advance_sdof(q0, v0, omega, 1.0 + delta, 0.0, dt);
        let below = advance_sdof(q0, v0, omega, 1.0 - delta, 0.0, dt);
        for (a, b, r) in [
            (above.0, below.0, reference.0),
            (above.1, below.1, reference.1),
        ] {
            assert!(a.is_finite() && b.is_finite(), "delta = {delta}");
            assert!((a - r).abs() < 10.0 * delta, "delta = {delta}: {a} vs {r}");
            assert!((b - r).abs() < 10.0 * delta, "delta = {delta}: {b} vs {r}");
            // smooth in ζ: the symmetric second difference is O(δ²)
            assert!(
                (a + b - 2.0 * r).abs() < 1e-10 + 100.0 * delta * delta,
                "delta = {delta}"
            );
        }
    }
}

/// Response of a single modal oscillator to a unit force switched on at t = 0.
fn step_response(omega: f64, zeta: f64, t: f64) -> f64 {
    let q_static = 1.0 / (omega * omega);
    if (zeta - 1.0).abs() < 1e-14 {
        q_static * (1.0 - (-omega * t).exp() * (1.0 + omega * t))
    } else if zeta < 1.0 {
        let omega_d = omega * (1.0 - zeta * zeta).sqrt();
        q_static
            * (1.0
                - (-zeta * omega * t).exp()
                    * ((omega_d * t).cos() + zeta * omega / omega_d * (omega_d * t).sin()))
    } else {
        let root = (zeta * zeta - 1.0).sqrt();
        let s1 = -omega / (zeta + root);
        let s2 = -omega * (zeta + root);
        q_static * (1.0 - (s2 * (s1 * t).exp() - s1 * (s2 * t).exp()) / (s2 - s1))
    }
}

#[test]
fn modal_superposition_step_load_matches_the_analytic_response_in_every_regime() {
    // From rest, results[k] is the state after k + 1 steps (see the docs of the
    // function), i.e. the response at t = (k + 1) dt of a unit force held constant.
    let omega = 10.0;
    let modal_result = ModalResult {
        frequencies: vec![omega],
        mode_shapes: vec![vec![1.0]],
    };
    let (dt, n_steps) = (0.03, 60);
    let history: Vec<(f64, Vec<f64>)> = (0..n_steps).map(|i| (i as f64 * dt, vec![1.0])).collect();
    for zeta in [0.3, 1.0, 2.0, 5.0] {
        let results = modal_superposition(&modal_result, &[zeta], &history);
        assert_eq!(results.len(), n_steps);
        let q_static = 1.0 / (omega * omega);
        for (k, u) in results.iter().enumerate() {
            let t = (k + 1) as f64 * dt;
            let exact = step_response(omega, zeta, t);
            assert!(
                (u[0] - exact).abs() < 1e-12 * q_static,
                "zeta = {zeta}, step {k}: {} vs {exact}",
                u[0]
            );
        }
        if zeta >= 1.0 {
            // a critically or over-damped step response rises monotonically to q_s
            // without overshoot (the old branch added q_s to an undeflected q)
            let mut previous = 0.0;
            for u in &results {
                assert!(
                    u[0] > previous && u[0] < q_static,
                    "zeta = {zeta}: {}",
                    u[0]
                );
                previous = u[0];
            }
        }
    }
}

#[test]
fn modal_superposition_rigid_body_mode_is_a_free_mass() {
    let modal_result = ModalResult {
        frequencies: vec![0.0],
        mode_shapes: vec![vec![1.0]],
    };
    let history = vec![(0.0, vec![2.0]), (0.5, vec![2.0])];
    let results = modal_superposition(&modal_result, &[0.1], &history);
    // first step dt = 0.5: q = ½ f dt², v = f dt; second step continues from there
    assert!((results[0][0] - 0.25).abs() < 1e-15);
    assert!((results[1][0] - (0.25 + 0.5 + 0.25)).abs() < 1e-15);
}

// ---------------------------------------------------------------------------
// F1251: truncation against the total mass, static correction
// ---------------------------------------------------------------------------

/// M = diag(1, 2, 3) with the mass-normalised modes e_r/√m_r: every mode carries
/// the effective mass m_r in the direction (1, 1, 1), the total mass is 6.
fn diagonal_mass_system() -> (CsrMatrix, Vec<Vec<f64>>, Vec<f64>) {
    let masses = [1.0, 2.0, 3.0];
    let mass = diagonal_csr(&masses);
    let modes = (0..3)
        .map(|r| {
            let mut mode = vec![0.0; 3];
            mode[r] = 1.0 / masses[r].sqrt();
            mode
        })
        .collect();
    (mass, modes, vec![1.0, 1.0, 1.0])
}

#[test]
fn truncation_measures_against_the_total_mass_not_the_extracted_modes() {
    let (mass, modes, direction) = diagonal_mass_system();
    let config = ModalTruncationConfig::default(); // 90 %
    // Two of the three modes: (m1 + m2)/(m1 + m2 + m3) = 3/6.  The old code normalised
    // by the sum of the supplied masses and reported the 90 % target as reached.
    let partial = modal_truncation(&modes[..2], &mass, &direction, &config);
    assert_eq!(partial.n_modes, 2);
    assert!(
        (partial.captured_fraction - 3.0 / 6.0).abs() < 1e-14,
        "{partial:?}"
    );
    assert!(!partial.target_met);

    // all three modes capture everything
    let full = modal_truncation(&modes, &mass, &direction, &config);
    assert_eq!(full.n_modes, 3);
    assert!((full.captured_fraction - 1.0).abs() < 1e-14);
    assert!(full.target_met);
}

#[test]
fn truncation_picks_the_smallest_sufficient_number_of_modes() {
    let (mass, modes, direction) = diagonal_mass_system();
    let pick = |fraction: f64, max_modes: usize| {
        let config = ModalTruncationConfig {
            mass_capture_fraction: fraction,
            max_modes,
            static_correction: false,
        };
        modal_truncation(&modes, &mass, &direction, &config)
    };
    // cumulative fractions: 1/6, 3/6, 6/6
    let one = pick(0.15, 50);
    assert_eq!((one.n_modes, one.target_met), (1, true));
    assert!((one.captured_fraction - 1.0 / 6.0).abs() < 1e-14);
    // exactly reached targets are met despite rounding
    let two = pick(0.5, 50);
    assert_eq!((two.n_modes, two.target_met), (2, true));
    assert!((two.captured_fraction - 0.5).abs() < 1e-14);
    let three = pick(1.0, 50);
    assert_eq!((three.n_modes, three.target_met), (3, true));
    // a target of zero needs no mode
    let none = pick(0.0, 50);
    assert_eq!(
        (none.n_modes, none.captured_fraction, none.target_met),
        (0, 0.0, true)
    );
}

#[test]
fn truncation_reports_what_the_returned_modes_capture_when_max_modes_cuts_off() {
    let (mass, modes, direction) = diagonal_mass_system();
    let config = ModalTruncationConfig {
        mass_capture_fraction: 0.99,
        max_modes: 2,
        static_correction: false,
    };
    let capped = modal_truncation(&modes, &mass, &direction, &config);
    assert_eq!(capped.n_modes, 2);
    // the fraction describes the two returned modes, not the three available ones
    assert!((capped.captured_fraction - 0.5).abs() < 1e-14, "{capped:?}");
    assert!(!capped.target_met);
    // no modes at all
    let nothing = modal_truncation(&[], &mass, &direction, &ModalTruncationConfig::default());
    assert_eq!(
        (
            nothing.n_modes,
            nothing.captured_fraction,
            nothing.target_met
        ),
        (0, 0.0, false)
    );
}

#[test]
fn truncation_of_a_massless_direction_keeps_all_allowed_modes_and_misses_the_target() {
    let (mass, modes, _) = diagonal_mass_system();
    let config = ModalTruncationConfig {
        mass_capture_fraction: 0.9,
        max_modes: 2,
        static_correction: false,
    };
    let result = modal_truncation(&modes, &mass, &[0.0, 0.0, 0.0], &config);
    assert_eq!(
        (result.n_modes, result.captured_fraction, result.target_met),
        (2, 0.0, false)
    );
    assert_eq!(
        cumulative_mass_fraction(&modes, &mass, &[0.0, 0.0, 0.0]),
        vec![0.0; 3]
    );
}

#[test]
fn cumulative_mass_fraction_uses_the_total_mass_as_well() {
    let (mass, modes, direction) = diagonal_mass_system();
    // a partial set ends at 3/6, it is not renormalised to 1
    let partial = cumulative_mass_fraction(&modes[..2], &mass, &direction);
    assert_eq!(partial.len(), 2);
    assert!((partial[0] - 1.0 / 6.0).abs() < 1e-14, "{partial:?}");
    assert!((partial[1] - 3.0 / 6.0).abs() < 1e-14, "{partial:?}");
    let full = cumulative_mass_fraction(&modes, &mass, &direction);
    assert!((full[2] - 1.0).abs() < 1e-14);
    // consistent with the truncation report
    let config = ModalTruncationConfig::default();
    let report = modal_truncation(&modes[..2], &mass, &direction, &config);
    assert!((report.captured_fraction - partial[1]).abs() < 1e-14);
}

#[test]
fn residual_flexibility_of_a_diagonal_system_is_the_missing_static_response() {
    // K = diag(k1, k2, k3), M = I, retained: mode 1 only.  u_res = K⁻¹F − φ1 φ1ᵀF/ω1².
    let stiffness = diagonal_csr(&[4.0, 9.0, 16.0]);
    let modes = vec![vec![1.0, 0.0, 0.0]];
    let frequencies = vec![2.0];
    let force = vec![1.0, 2.0, 3.0];
    let residual =
        residual_flexibility(&stiffness, &modes, &frequencies, &force).expect("restrained system");
    assert!(
        max_abs_diff(&residual, &[0.0, 2.0 / 9.0, 3.0 / 16.0]) < 1e-10,
        "{residual:?}"
    );
    // no retained mode: the residual is the whole static response
    let all = residual_flexibility(&stiffness, &[], &[], &force).expect("restrained system");
    assert!(
        max_abs_diff(&all, &[0.25, 2.0 / 9.0, 3.0 / 16.0]) < 1e-10,
        "{all:?}"
    );
    // a zero load needs no solve
    let zero =
        residual_flexibility(&stiffness, &modes, &frequencies, &[0.0; 3]).expect("zero load");
    assert!(zero.iter().all(|u| *u == 0.0));
}

#[test]
fn residual_flexibility_of_a_complete_mode_set_vanishes() {
    let n = 6;
    let stiffness = spring_chain_csr(n, 3.0);
    let (omega, shapes) = spring_chain_modes(n, 3.0);
    let force = vec![1.0, -0.5, 2.0, 0.0, 0.7, -1.2];
    let residual = residual_flexibility(&stiffness, &shapes, &omega, &force).expect("restrained");
    assert!(residual.iter().all(|u| u.abs() < 1e-9), "{residual:?}");

    // with three of six modes it equals the sum of the three neglected modes
    let residual =
        residual_flexibility(&stiffness, &shapes[..3], &omega[..3], &force).expect("restrained");
    let mut expected = vec![0.0; n];
    for (mode, &w) in shapes.iter().zip(omega.iter()).skip(3) {
        let coefficient = dot(mode, &force) / (w * w);
        for (e, phi) in expected.iter_mut().zip(mode.iter()) {
            *e += coefficient * phi;
        }
    }
    assert!(
        max_abs_diff(&residual, &expected) < 1e-9,
        "{residual:?} vs {expected:?}"
    );
    assert!(
        expected.iter().any(|u| u.abs() > 1e-3),
        "the neglected modes do contribute"
    );
}

#[test]
fn static_correction_flag_selects_whether_dropped_modes_are_restored() {
    let n = 6;
    let stiffness = spring_chain_csr(n, 3.0);
    let (omega, shapes) = spring_chain_modes(n, 3.0);
    let force = vec![1.0, -0.5, 2.0, 0.0, 0.7, -1.2];
    let exact = crate::solvers::conjugate_gradient(&stiffness, &force, &vec![0.0; n], 1000, 1e-15);

    let mut config = ModalTruncationConfig::default();
    let without = truncated_static_response(&stiffness, &shapes, &omega, &force, 3, &config)
        .expect("restrained");
    config.static_correction = true;
    let with = truncated_static_response(&stiffness, &shapes, &omega, &force, 3, &config)
        .expect("restrained");

    // with the correction the truncated model reproduces K⁻¹F, without it the three
    // dropped modes are missing
    assert!(max_abs_diff(&with, &exact) < 1e-9, "{with:?} vs {exact:?}");
    assert!(
        max_abs_diff(&without, &exact) > 1e-3,
        "{without:?} vs {exact:?}"
    );
    // and the difference is the residual flexibility
    let residual =
        residual_flexibility(&stiffness, &shapes[..3], &omega[..3], &force).expect("restrained");
    let difference: Vec<f64> = with
        .iter()
        .zip(without.iter())
        .map(|(a, b)| a - b)
        .collect();
    assert!(max_abs_diff(&difference, &residual) < 1e-12);
    // n_keep is clamped to the supplied modes
    let all = truncated_static_response(&stiffness, &shapes, &omega, &force, 99, &config)
        .expect("restrained");
    assert!(max_abs_diff(&all, &exact) < 1e-9);
}

#[test]
fn static_correction_routines_validate_their_input() {
    let stiffness = diagonal_csr(&[4.0, 9.0]);
    let modes = vec![vec![1.0, 0.0]];
    let force = vec![1.0, 1.0];

    assert_eq!(
        residual_flexibility(&CsrMatrix::new(0, 0), &[], &[], &[]).err(),
        Some(ModalError::EmptySystem)
    );
    let wide = CsrMatrix::from_triplets(2, 3, &[(0, 0, 1.0)]);
    assert_eq!(
        residual_flexibility(&wide, &modes, &[2.0], &force).err(),
        Some(ModalError::DimensionMismatch {
            expected: 2,
            got: 3
        })
    );
    assert_eq!(
        residual_flexibility(&stiffness, &modes, &[2.0], &[1.0, 1.0, 1.0]).err(),
        Some(ModalError::DimensionMismatch {
            expected: 2,
            got: 3
        })
    );
    assert_eq!(
        residual_flexibility(&stiffness, &modes, &[2.0, 3.0], &force).err(),
        Some(ModalError::DimensionMismatch {
            expected: 1,
            got: 2
        })
    );
    assert_eq!(
        residual_flexibility(&stiffness, &[vec![1.0, 0.0, 0.0]], &[2.0], &force).err(),
        Some(ModalError::DimensionMismatch {
            expected: 2,
            got: 3
        })
    );
    let config = ModalTruncationConfig::default();
    assert_eq!(
        truncated_static_response(&stiffness, &modes, &[2.0], &[1.0], 1, &config).err(),
        Some(ModalError::DimensionMismatch {
            expected: 2,
            got: 1
        })
    );
}

#[test]
fn a_singular_stiffness_matrix_fails_the_static_solve() {
    // diag(1, 0): the second DOF is unrestrained, K u = F has no solution.
    let singular = diagonal_csr(&[1.0, 0.0]);
    match residual_flexibility(&singular, &[], &[], &[1.0, 1.0]) {
        Err(ModalError::StaticSolveFailed { residual }) => {
            assert!(
                residual > 0.5,
                "the unreachable component stays: {residual}"
            );
        }
        other => panic!("expected StaticSolveFailed, got {other:?}"),
    }
    let config = ModalTruncationConfig {
        static_correction: true,
        ..ModalTruncationConfig::default()
    };
    assert!(matches!(
        truncated_static_response(&singular, &[], &[], &[1.0, 1.0], 0, &config),
        Err(ModalError::StaticSolveFailed { .. })
    ));
    // without the correction no solve is needed
    let config = ModalTruncationConfig::default();
    let response = truncated_static_response(&singular, &[], &[], &[1.0, 1.0], 0, &config)
        .expect("no correction requested");
    assert_eq!(response, vec![0.0, 0.0]);
}
