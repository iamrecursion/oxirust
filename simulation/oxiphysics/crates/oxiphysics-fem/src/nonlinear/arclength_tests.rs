//! Tests of the Riks predictor and the bordered pseudo-arclength corrector.

use super::*;
use crate::error::Error;
use crate::sparse::CsrMatrix;

fn diagonal(values: &[f64]) -> CsrMatrix {
    let triplets: Vec<(usize, usize, f64)> =
        values.iter().enumerate().map(|(i, &v)| (i, i, v)).collect();
    CsrMatrix::from_triplets(values.len(), values.len(), &triplets)
}

/// Internal force of the classic 1-DOF snap-through, `f(u) = ½ u (u − 1)(u − 2)`; its limit
/// points are at `u = 1 ∓ 1/√3` with `λ = ±1/(3√3)`.
fn snap_force(u: f64) -> f64 {
    u - 1.5 * u * u + 0.5 * u * u * u
}

fn snap_tangent(u: f64) -> f64 {
    1.0 - 3.0 * u + 1.5 * u * u
}

/// `R(u, λ) = f_int(u) − λ f_ref` with `f_ref = 1` for the snap-through system.
fn snap_system(u: &[f64], lambda: f64) -> (CsrMatrix, Vec<f64>) {
    (
        diagonal(&[snap_tangent(u[0])]),
        vec![snap_force(u[0]) - lambda],
    )
}

const SNAP_LIMIT_LOAD: f64 = 0.192_450_089_729_875_26;

/// Advance `state` by one predictor/corrector step and return the corrector result.
fn continuation_step<F>(
    state: &mut RiksState,
    reference_load: &[f64],
    system: F,
    criteria: &ConvergenceCriteria,
) -> ContinuationStepResult
where
    F: Fn(&[f64], f64) -> (CsrMatrix, Vec<f64>) + Copy,
{
    let (u_pred, lambda_pred) =
        riks_predictor(state, reference_load, system, 1.0).expect("regular tangent");
    let du_pred: Vec<f64> = u_pred.iter().zip(&state.u).map(|(n, o)| n - o).collect();
    let step = pseudo_arclength_step(
        state.u.len(),
        &state.u.clone(),
        state.lambda,
        &du_pred,
        lambda_pred - state.lambda,
        state.arc_length,
        reference_load,
        system,
        criteria,
    )
    .expect("valid input");
    assert!(
        step.converged,
        "corrector did not converge at u = {:?}, λ = {}",
        state.u, state.lambda
    );
    state
        .accept_step(&step.displacement, step.lambda)
        .expect("matching sizes");
    step
}

#[test]
fn riks_predictor_solves_the_tangent_system_for_the_supplied_reference_load() {
    let system = |_u: &[f64], _l: f64| (diagonal(&[2.0, 4.0]), vec![0.0; 2]);
    let mut state = RiksState::new(2, 0.3);
    let (u_new, lambda_new) =
        riks_predictor(&mut state, &[1.0, 2.0], system, 1.0).expect("regular tangent");
    // δu_t = K⁻¹ f_ref = [0.5, 0.5]; Δλ = Δs / √(1 + ‖δu_t‖²).
    let dl = 0.3 / 1.5_f64.sqrt();
    assert!((lambda_new - dl).abs() < 1e-14, "λ = {lambda_new}");
    assert!((u_new[0] - 0.5 * dl).abs() < 1e-14 && (u_new[1] - 0.5 * dl).abs() < 1e-14);
    assert!(
        (state.current_arc() - 0.3).abs() < 1e-14,
        "arc {}",
        state.current_arc()
    );

    // Another reference load gives another direction (no invented uniform load).
    let mut other = RiksState::new(2, 0.3);
    let (u2, l2) = riks_predictor(&mut other, &[3.0, 0.0], system, 1.0).expect("regular tangent");
    let dl2 = 0.3 / (1.0_f64 + 2.25).sqrt();
    assert!((l2 - dl2).abs() < 1e-14);
    assert!((u2[0] - 1.5 * dl2).abs() < 1e-14 && u2[1].abs() < 1e-14);
}

#[test]
fn riks_predictor_handles_an_indefinite_tangent() {
    // Conjugate gradients cannot solve this system; the LU-based solve can.
    let system = |_u: &[f64], _l: f64| (diagonal(&[-1.0, 2.0]), vec![0.0; 2]);
    let mut state = RiksState::new(2, 0.5);
    // δu_t = K⁻¹ f_ref = [−1, 0.5]
    state.du = vec![-0.1, 0.05];
    state.dlambda = 0.1;
    let (u_new, lambda_new) =
        riks_predictor(&mut state, &[1.0, 1.0], system, 1.0).expect("regular tangent");
    let dl = 0.5 / (1.0_f64 + 1.25).sqrt();
    assert!((lambda_new - dl).abs() < 1e-14);
    assert!((u_new[0] + dl).abs() < 1e-14 && (u_new[1] - 0.5 * dl).abs() < 1e-14);
}

#[test]
fn riks_predictor_follows_the_previous_increment_through_a_limit_point() {
    // Post-limit-point tangent K_t = −1, so δu_t = −1 for f_ref = 1.  The previous
    // increment moved to larger u while the load decreased (du = +0.1, dλ = −0.1).
    let system = |_u: &[f64], _l: f64| (diagonal(&[-1.0]), vec![0.0]);
    let mut state = RiksState::new(1, 0.2);
    state.du = vec![0.1];
    state.dlambda = -0.1;
    let (u_new, lambda_new) =
        riks_predictor(&mut state, &[1.0], system, 1.0).expect("regular tangent");
    let dl = 0.2 / 2.0_f64.sqrt();
    // δu_t·Δu_prev + Δλ_prev = −0.2 < 0  ⇒  Δλ = −Δs/√2 and Δu = Δλ·δu_t = +Δs/√2.
    assert!((lambda_new + dl).abs() < 1e-14, "λ = {lambda_new}");
    assert!((u_new[0] - dl).abs() < 1e-14, "u = {}", u_new[0]);

    // The explicit sign reverses the direction of travel.
    let mut reversed = RiksState::new(1, 0.2);
    reversed.du = vec![0.1];
    reversed.dlambda = -0.1;
    let (u_rev, l_rev) =
        riks_predictor(&mut reversed, &[1.0], system, -1.0).expect("regular tangent");
    assert!((l_rev - dl).abs() < 1e-14 && (u_rev[0] + dl).abs() < 1e-14);
}

#[test]
fn riks_predictor_scales_to_large_systems_with_gmres() {
    let n = 600;
    let mut triplets = Vec::new();
    for i in 0..n {
        triplets.push((i, i, 2.5));
        if i > 0 {
            triplets.push((i, i - 1, -1.0));
        }
        if i + 1 < n {
            triplets.push((i, i + 1, -1.0));
        }
    }
    let k = CsrMatrix::from_triplets(n, n, &triplets);
    let f_ref = vec![1.0; n];
    let k_for_system = k.clone();
    let system = move |_u: &[f64], _l: f64| (k_for_system.clone(), vec![0.0; n]);
    let mut state = RiksState::new(n, 1.0);
    let (u_new, lambda_new) = riks_predictor(&mut state, &f_ref, system, 1.0).expect("solvable");
    // From the origin Δu = Δλ · δu_t, so K Δu / Δλ must reproduce f_ref.
    let du_tangent: Vec<f64> = u_new.iter().map(|x| x / lambda_new).collect();
    for (i, value) in k.mul_vec(&du_tangent).iter().enumerate() {
        assert!((value - 1.0).abs() < 1e-8, "row {i}: K δu_t = {value}");
    }
}

#[test]
fn riks_predictor_rejects_singular_tangents_and_bad_inputs() {
    let singular = |_u: &[f64], _l: f64| (diagonal(&[1.0, 0.0]), vec![0.0; 2]);
    let mut state = RiksState::new(2, 0.1);
    assert!(matches!(
        riks_predictor(&mut state, &[1.0, 1.0], singular, 1.0),
        Err(Error::SingularSystem)
    ));
    let regular = |_u: &[f64], _l: f64| (diagonal(&[1.0, 1.0]), vec![0.0; 2]);
    assert!(matches!(
        riks_predictor(&mut state, &[1.0], regular, 1.0),
        Err(Error::InvalidInput(_))
    ));
    let mut no_arc = RiksState::new(2, 0.0);
    assert!(matches!(
        riks_predictor(&mut no_arc, &[1.0, 1.0], regular, 1.0),
        Err(Error::InvalidInput(_))
    ));
    assert!(matches!(
        state.accept_step(&[0.0], 0.0),
        Err(Error::InvalidInput(_))
    ));
}

#[test]
fn accept_step_stores_the_converged_increments() {
    let mut state = RiksState::new(2, 0.1);
    state.u = vec![1.0, 2.0];
    state.lambda = 0.5;
    state
        .accept_step(&[1.5, 1.0], 0.25)
        .expect("matching sizes");
    assert_eq!(state.u, vec![1.5, 1.0]);
    assert_eq!(state.lambda, 0.25);
    assert_eq!(state.du, vec![0.5, -1.0]);
    assert!((state.dlambda + 0.25).abs() < 1e-15);
}

#[test]
fn corrector_accepts_an_exact_predictor_of_a_linear_problem_immediately() {
    // R = K u − λ f_ref with constant K: the tangent predictor already solves it.
    let system = |u: &[f64], lambda: f64| {
        let k = diagonal(&[2.0, 4.0]);
        let ku = k.mul_vec(u);
        (k, vec![ku[0] - lambda, ku[1] - 2.0 * lambda])
    };
    let mut state = RiksState::new(2, 0.4);
    let step = continuation_step(
        &mut state,
        &[1.0, 2.0],
        system,
        &ConvergenceCriteria::default(),
    );
    assert_eq!(step.iterations, 0);
    assert!((state.current_arc() - 0.4).abs() < 1e-12);
}

#[test]
fn corrector_follows_the_snap_through_curve_through_both_limit_points() {
    let ds = 0.05;
    let criteria = ConvergenceCriteria::default();
    let mut state = RiksState::new(1, ds);
    let mut path: Vec<(f64, f64)> = Vec::new();
    for _ in 0..120 {
        let step = continuation_step(&mut state, &[1.0], snap_system, &criteria);
        assert!(
            step.iterations <= 6,
            "{} corrector iterations",
            step.iterations
        );
        // The converged increment lies on the arc-length sphere.
        assert!(
            (state.current_arc() - ds).abs() < 1e-8,
            "arc {}",
            state.current_arc()
        );
        path.push((state.lambda, state.u[0]));
        if state.u[0] > 2.1 {
            break;
        }
    }
    assert!(
        state.u[0] > 2.1,
        "the path did not pass the second limit point"
    );

    // Every point is on the analytic curve λ = f_int(u).
    for &(lambda, u) in &path {
        assert!(
            (lambda - snap_force(u)).abs() < 1e-8,
            "(λ, u) = ({lambda}, {u})"
        );
    }
    let (peak, &(peak_lambda, peak_u)) = path
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.0.total_cmp(&b.1.0))
        .expect("path is not empty");
    let (trough, &(trough_lambda, trough_u)) = path
        .iter()
        .enumerate()
        .min_by(|a, b| a.1.0.total_cmp(&b.1.0))
        .expect("path is not empty");
    assert!(
        (peak_lambda - SNAP_LIMIT_LOAD).abs() < 1e-3,
        "λ_max = {peak_lambda}"
    );
    assert!(
        (trough_lambda + SNAP_LIMIT_LOAD).abs() < 1e-3,
        "λ_min = {trough_lambda}"
    );
    assert!((peak_u - (1.0 - 1.0 / 3.0_f64.sqrt())).abs() < 0.06);
    assert!((trough_u - (1.0 + 1.0 / 3.0_f64.sqrt())).abs() < 0.06);
    // The load falls monotonically between the limit points and rises again afterwards.
    assert!(trough > peak);
    for pair in path[peak..=trough].windows(2) {
        assert!(pair[1].0 < pair[0].0);
    }
    assert!(path.last().expect("non-empty").0 > trough_lambda + 0.1);
}

#[test]
fn corrector_follows_a_coupled_two_dof_snap_through() {
    // f_int = [g(u₀) + k(u₀ − u₁), k(u₁ − u₀) + c u₁] with the snap-through spring g.
    // Eliminating u₁ = k/(k + c) · u₀ gives λ(u₀) = g(u₀) + k c/(k + c) · u₀ with
    // limit points at the roots of λ'(u₀) = 0.
    let (k, c) = (0.8, 0.3);
    let system = move |u: &[f64], lambda: f64| {
        let tangent = CsrMatrix::from_triplets(
            2,
            2,
            &[
                (0, 0, snap_tangent(u[0]) + k),
                (0, 1, -k),
                (1, 0, -k),
                (1, 1, k + c),
            ],
        );
        let residual = vec![
            snap_force(u[0]) + k * (u[0] - u[1]) - lambda,
            k * (u[1] - u[0]) + c * u[1],
        ];
        (tangent, residual)
    };
    let reduced = |u0: f64| snap_force(u0) + k * c / (k + c) * u0;
    // λ'(u₀) = (1 + kc/(k+c)) − 3 u₀ + 1.5 u₀² = 0
    let b = 1.0 + k * c / (k + c);
    let disc = (9.0_f64 - 6.0 * b).sqrt();
    let (u_max, u_min) = ((3.0 - disc) / 3.0, (3.0 + disc) / 3.0);

    let criteria = ConvergenceCriteria::default();
    let mut state = RiksState::new(2, 0.1);
    let mut path: Vec<(f64, f64, f64)> = Vec::new();
    for _ in 0..80 {
        continuation_step(&mut state, &[1.0, 0.0], system, &criteria);
        path.push((state.lambda, state.u[0], state.u[1]));
        if state.u[0] > 2.2 {
            break;
        }
    }
    for &(lambda, u0, u1) in &path {
        assert!((u1 - k / (k + c) * u0).abs() < 1e-7, "u₁ = {u1}, u₀ = {u0}");
        assert!(
            (lambda - reduced(u0)).abs() < 1e-7,
            "λ = {lambda}, u₀ = {u0}"
        );
    }
    // The load rises to the first limit point, falls to the second one and rises again.
    let midpoint = 0.5 * (u_max + u_min);
    let peak = path
        .iter()
        .filter(|p| p.1 < midpoint)
        .map(|p| p.0)
        .fold(f64::MIN, f64::max);
    let trough = path
        .iter()
        .filter(|p| p.1 >= midpoint)
        .map(|p| p.0)
        .fold(f64::MAX, f64::min);
    assert!(
        (peak - reduced(u_max)).abs() < 5e-3,
        "λ_max = {peak} vs {}",
        reduced(u_max)
    );
    assert!(
        (trough - reduced(u_min)).abs() < 5e-3,
        "λ_min = {trough} vs {}",
        reduced(u_min)
    );
    assert!(trough < peak - 0.1);
}

#[test]
fn corrector_reports_a_singular_tangent_as_not_converged() {
    // K ≡ 0 with a non-zero residual: no Newton step exists.
    let system = |_u: &[f64], _l: f64| (diagonal(&[0.0]), vec![1.0]);
    let step = pseudo_arclength_step(
        1,
        &[0.0],
        0.0,
        &[0.1],
        0.1,
        0.2,
        &[1.0],
        system,
        &ConvergenceCriteria::default(),
    )
    .expect("valid input");
    assert!(!step.converged);
    assert_eq!(step.iterations, 0);
    assert_eq!(step.displacement, vec![0.1]);
}

#[test]
fn corrector_rejects_inconsistent_inputs() {
    let criteria = ConvergenceCriteria::default();
    let run = |u_prev: &[f64], reference: &[f64], arc: f64| {
        pseudo_arclength_step(
            1,
            u_prev,
            0.0,
            &[0.1],
            0.1,
            arc,
            reference,
            snap_system,
            &criteria,
        )
    };
    assert!(matches!(
        run(&[0.0, 0.0], &[1.0], 0.1),
        Err(Error::InvalidInput(_))
    ));
    assert!(matches!(
        run(&[0.0], &[1.0, 1.0], 0.1),
        Err(Error::InvalidInput(_))
    ));
    assert!(matches!(
        run(&[0.0], &[1.0], 0.0),
        Err(Error::InvalidInput(_))
    ));
    assert!(matches!(
        run(&[0.0], &[1.0], f64::NAN),
        Err(Error::InvalidInput(_))
    ));
    // The callback returns a 2×2 tangent for a 1-DOF problem.
    let wrong_size = |_u: &[f64], _l: f64| (diagonal(&[1.0, 1.0]), vec![0.0]);
    assert!(matches!(
        pseudo_arclength_step(
            1,
            &[0.0],
            0.0,
            &[0.1],
            0.1,
            0.1,
            &[1.0],
            wrong_size,
            &criteria
        ),
        Err(Error::InvalidInput(_))
    ));
}
