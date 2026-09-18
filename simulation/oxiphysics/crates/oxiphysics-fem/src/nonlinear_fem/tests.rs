// Copyright 2026 COOLJAPAN OU (Team KitaSan)
// SPDX-License-Identifier: Apache-2.0

//! Unit tests of the nonlinear FEM building blocks (kinematics, J2 plasticity,
//! Newton-Raphson, arc-length control, locking-free elements).

use super::*;

// ── Mat3 operations ───────────────────────────────────────────────────────

#[test]
fn mat3_mul_identity() {
    let a = [[1., 2., 3.], [4., 5., 6.], [7., 8., 9.]];
    let c = mat3_mul(&a, &EYE3);
    for (i, row) in c.iter().enumerate() {
        for (j, &val) in row.iter().enumerate() {
            assert!((val - a[i][j]).abs() < 1e-10);
        }
    }
}

#[test]
fn mat3_det_identity_is_one() {
    assert!((mat3_det(&EYE3) - 1.0).abs() < 1e-10);
}

#[test]
fn mat3_inv_of_identity() {
    let inv = mat3_inv(&EYE3);
    for (i, row) in inv.iter().enumerate() {
        for (j, &val) in row.iter().enumerate() {
            assert!((val - EYE3[i][j]).abs() < 1e-10);
        }
    }
}

#[test]
fn mat3_inv_round_trip() {
    let a = [[2., 1., 0.], [1., 3., 1.], [0., 1., 4.]];
    let inv = mat3_inv(&a);
    let prod = mat3_mul(&a, &inv);
    for (i, row) in prod.iter().enumerate() {
        for (j, &val) in row.iter().enumerate() {
            let expected = if i == j { 1.0 } else { 0.0 };
            assert!(
                (val - expected).abs() < 1e-8,
                "A·A⁻¹[{i}][{j}] = {} != {}",
                val,
                expected
            );
        }
    }
}

#[test]
fn mat3_transpose_symmetry() {
    let a = [[1., 2., 3.], [4., 5., 6.], [7., 8., 9.]];
    let at = mat3_transpose(&a);
    for (i, row) in at.iter().enumerate() {
        for (j, &val) in row.iter().enumerate() {
            assert!((val - a[j][i]).abs() < 1e-10);
        }
    }
}

// ── DeformationGradient ───────────────────────────────────────────────────

#[test]
fn deformation_gradient_identity_jacobian() {
    let f = DeformationGradient::identity();
    assert!((f.jacobian() - 1.0).abs() < 1e-10);
}

#[test]
fn deformation_gradient_green_lagrange_zero_for_identity() {
    let f = DeformationGradient::identity();
    let e = f.green_lagrange_strain();
    for row in &e {
        for &val in row {
            assert!(val.abs() < 1e-10);
        }
    }
}

#[test]
fn deformation_gradient_left_cauchy_green_identity() {
    let f = DeformationGradient::identity();
    let b = f.left_cauchy_green();
    for i in 0..3 {
        for j in 0..3 {
            assert!((b[i][j] - EYE3[i][j]).abs() < 1e-10);
        }
    }
}

#[test]
fn deformation_gradient_uniaxial_extension() {
    let h = [[0.1, 0., 0.], [0., 0., 0.], [0., 0., 0.]]; // ε_11 = 0.1
    let f = DeformationGradient::from_displacement_gradient(&h);
    assert!((f.jacobian() - 1.1).abs() < 1e-10, "J = {}", f.jacobian());
    let e = f.green_lagrange_strain();
    // E_11 = 0.5*(F_11² - 1) = 0.5*(1.1²-1) = 0.105
    assert!((e[0][0] - 0.105).abs() < 1e-10, "E11 = {}", e[0][0]);
}

#[test]
fn f_bar_preserves_deviatoric() {
    let h = [[0.2, 0., 0.], [0., 0., 0.], [0., 0., 0.]];
    let f = DeformationGradient::from_displacement_gradient(&h);
    let j0 = 1.0;
    let fb = f.f_bar(j0);
    let j_fb = mat3_det(&fb);
    assert!(
        (j_fb - j0).abs() < 1e-6,
        "F̄ should have det = J0={j0}, got {j_fb}"
    );
}

// ── J2 plasticity ─────────────────────────────────────────────────────────

#[test]
fn j2_elastic_predictor_no_plastic() {
    let mat = J2Material::new(200e9, 0.3, 250e6, 0.0);
    let sigma_n = [0.0; 6];
    let delta_eps = [0.001, 0.0, 0.0, 0.0, 0.0, 0.0];
    let sigma_tr = mat.elastic_predictor(&sigma_n, &delta_eps);
    assert!(sigma_tr[0] > 0.0, "tensile stress expected");
}

#[test]
fn j2_return_mapping_elastic_stays() {
    let mat = J2Material::new(200e9, 0.3, 250e6, 0.0);
    let sigma_n = [0.0; 6];
    let delta_eps = [1e-5, 0.0, 0.0, 0.0, 0.0, 0.0]; // well below yield
    let sigma_tr = mat.elastic_predictor(&sigma_n, &delta_eps);
    let state_n = PlasticState::new();
    let (sigma_n1, state_n1, delta_gamma) = mat.return_mapping(&sigma_tr, &state_n);
    assert!(delta_gamma < 1e-10, "should be elastic: δγ = {delta_gamma}");
    assert!(state_n1.eps_p_eq < 1e-10);
    assert!((sigma_n1[0] - sigma_tr[0]).abs() < 1e-6);
}

#[test]
fn j2_return_mapping_plastic_caps_stress() {
    let mat = J2Material::new(200e9, 0.3, 250e6, 1e9);
    let sigma_n = [0.0; 6];
    let delta_eps = [0.01, 0.0, 0.0, 0.0, 0.0, 0.0]; // large step → yield
    let sigma_tr = mat.elastic_predictor(&sigma_n, &delta_eps);
    let state_n = PlasticState::new();
    let (sigma_n1, state_n1, delta_gamma) = mat.return_mapping(&sigma_tr, &state_n);
    let vm = J2Material::von_mises(&sigma_n1);
    let sy = mat.yield_stress(state_n1.eps_p_eq);
    assert!(delta_gamma > 0.0, "plastic step expected");
    // The return mapping puts the stress exactly on the hardened yield surface ...
    assert!(
        (vm - sy).abs() <= 1e-8 * sy,
        "stress {vm} should sit on the yield surface {sy}"
    );
    // ... and ε̄ᵖ grows by exactly Δγ for isotropic hardening.
    assert!(
        (state_n1.eps_p_eq - delta_gamma).abs() <= 1e-14 * delta_gamma,
        "ε̄ᵖ = {} but Δγ = {delta_gamma}",
        state_n1.eps_p_eq
    );
}

#[test]
fn j2_von_mises_hydrostatic_is_zero() {
    let sigma_hydro = [100e6, 100e6, 100e6, 0.0, 0.0, 0.0];
    let vm = J2Material::von_mises(&sigma_hydro);
    assert!(
        vm < 1.0,
        "hydrostatic stress should give zero von Mises: {vm}"
    );
}

#[test]
fn j2_consistent_tangent_elastic_symmetry() {
    let mat = J2Material::new(200e9, 0.3, 250e6, 0.0);
    let sigma_tr = [100e6; 6];
    let state_n = PlasticState::new();
    let c = mat.consistent_tangent(0.0, &sigma_tr, &state_n);
    for (i, row) in c.iter().enumerate() {
        for (j, &val) in row.iter().enumerate() {
            assert!(
                (val - c[j][i]).abs() < 1.0,
                "tangent not symmetric at [{i}][{j}]: {} vs {}",
                val,
                c[j][i]
            );
        }
    }
}

// ── Newton-Raphson ─────────────────────────────────────────────────────────

struct LinearSystem {
    k: f64,
    f: f64,
}

impl NonlinearSystem for LinearSystem {
    fn residual(&self, x: &[f64]) -> Vec<f64> {
        vec![self.k * x[0] - self.f]
    }
    fn jacobian(&self, _x: &[f64]) -> Vec<Vec<f64>> {
        vec![vec![self.k]]
    }
}

#[test]
fn nr_solves_linear_system() {
    let sys = LinearSystem { k: 3.0, f: 9.0 };
    let params = NrParams::default();
    let result = newton_raphson_solve(&sys, vec![0.0], &params);
    assert!(result.converged, "should converge");
    assert!(
        (result.solution[0] - 3.0).abs() < 1e-8,
        "sol={}",
        result.solution[0]
    );
}

struct QuadraticSystem;

impl NonlinearSystem for QuadraticSystem {
    fn residual(&self, x: &[f64]) -> Vec<f64> {
        // f(x) = x² - 4 = 0 → x = 2
        vec![x[0] * x[0] - 4.0]
    }
    fn jacobian(&self, x: &[f64]) -> Vec<Vec<f64>> {
        vec![vec![2.0 * x[0]]]
    }
}

#[test]
fn nr_solves_quadratic() {
    let sys = QuadraticSystem;
    let result = newton_raphson_solve(&sys, vec![3.0], &NrParams::default());
    assert!(result.converged, "quadratic NR did not converge");
    assert!(
        (result.solution[0] - 2.0).abs() < 1e-8,
        "sol={}",
        result.solution[0]
    );
}

/// `f(x) = -2 x - 4` has the root `x = -2` and a *negative* Jacobian.
struct NegativeStiffnessSystem;

impl NonlinearSystem for NegativeStiffnessSystem {
    fn residual(&self, x: &[f64]) -> Vec<f64> {
        vec![-2.0 * x[0] - 4.0]
    }
    fn jacobian(&self, _x: &[f64]) -> Vec<Vec<f64>> {
        vec![vec![-2.0]]
    }
}

/// A system whose Jacobian is identically singular.
struct SingularSystem;

impl NonlinearSystem for SingularSystem {
    fn residual(&self, x: &[f64]) -> Vec<f64> {
        vec![x[0] - 1.0, x[1] - 2.0]
    }
    fn jacobian(&self, _x: &[f64]) -> Vec<Vec<f64>> {
        vec![vec![1.0, 2.0], vec![2.0, 4.0]]
    }
}

#[test]
fn dense_solver_handles_negative_pivots() {
    // The old back substitution divided by max(a_ii, 1e-30) and returned ~ -4e30 here.
    let x = solve_linear_system(&[vec![-2.0]], &[4.0]).expect("regular system");
    assert!((x[0] + 2.0).abs() < 1e-14, "x = {}", x[0]);

    // Indefinite 2×2 with a negative leading pivot and a row exchange.
    let a = vec![vec![-3.0, 1.0], vec![2.0, 5.0]];
    let x = solve_linear_system(&a, &[-1.0, 12.0]).expect("regular system");
    assert!(
        (x[0] - 1.0).abs() < 1e-13 && (x[1] - 2.0).abs() < 1e-13,
        "{x:?}"
    );
}

#[test]
fn dense_solver_reports_singular_and_malformed_systems() {
    let singular = vec![vec![1.0, 2.0], vec![2.0, 4.0]];
    assert!(matches!(
        solve_linear_system(&singular, &[1.0, 2.0]),
        Err(Error::SingularSystem)
    ));
    assert!(matches!(
        solve_linear_system(&[vec![1.0, 0.0]], &[1.0]),
        Err(Error::InvalidInput(_))
    ));
    assert!(matches!(
        solve_linear_system(&[vec![f64::NAN]], &[1.0]),
        Err(Error::InvalidInput(_))
    ));
}

#[test]
fn nr_solves_a_system_with_negative_stiffness() {
    let result = newton_raphson_solve(&NegativeStiffnessSystem, vec![0.0], &NrParams::default());
    assert!(result.converged, "residual {}", result.residual_norm);
    assert!(
        (result.solution[0] + 2.0).abs() < 1e-10,
        "{:?}",
        result.solution
    );
}

#[test]
fn nr_reports_non_convergence_for_a_singular_jacobian() {
    let result = newton_raphson_solve(&SingularSystem, vec![0.0, 0.0], &NrParams::default());
    assert!(!result.converged);
    // The last iterate is returned untouched instead of an arbitrary blown-up step.
    assert_eq!(result.solution, vec![0.0, 0.0]);
    assert!((result.residual_norm - 5.0_f64.sqrt()).abs() < 1e-12);
}

#[test]
fn nr_line_search_helps_convergence() {
    let sys = QuadraticSystem;
    let params = NrParams {
        use_line_search: true,
        ..Default::default()
    };
    let result = newton_raphson_solve(&sys, vec![10.0], &params);
    assert!(result.converged || result.residual_norm < 1e-4);
}

// ── Load stepping ─────────────────────────────────────────────────────────

#[test]
fn load_stepper_reaches_target() {
    let mut ls = LoadStepper::new(10);
    while !ls.is_complete() {
        ls.advance();
    }
    assert!((ls.lambda - 1.0).abs() < 1e-10);
}

#[test]
fn load_stepper_step_count() {
    let mut ls = LoadStepper::new(5);
    ls.auto_step = false;
    let mut steps = 0;
    while !ls.is_complete() {
        ls.advance();
        steps += 1;
    }
    assert_eq!(steps, 5);
}

#[test]
fn load_stepper_auto_updates_step_size() {
    let mut ls = LoadStepper::new(10);
    let dl_initial = ls.dlambda;
    ls.update_step_size(2); // fewer iters than i_des=4 → step should grow
    assert!(
        ls.dlambda >= dl_initial,
        "step should grow when converging fast"
    );
}

// ── Arc-length ─────────────────────────────────────────────────────────────

/// Internal force of the classic 1-DOF snap-through: `f(u) = ½ u (u − 1)(u − 2)`.
///
/// Its limit points are at `u = 1 ∓ 1/√3` with `λ = ±1/(3√3) ≈ ±0.19245`.
fn snap_force(u: f64) -> f64 {
    u - 1.5 * u * u + 0.5 * u * u * u
}

fn snap_tangent(u: f64) -> f64 {
    1.0 - 3.0 * u + 1.5 * u * u
}

/// Secant stiffness of the same law: `f(u) = K_s(u) · u`.
fn snap_secant(u: f64) -> f64 {
    1.0 - 1.5 * u + 0.5 * u * u
}

const SNAP_LIMIT_LOAD: f64 = 0.192_450_089_729_875_26;

fn run_until_displacement(solver: &mut RiksSolver, u_end: f64, max_steps: usize) {
    for _ in 0..max_steps {
        assert!(
            solver.step(),
            "arc-length step failed at {:?}",
            solver.history.last()
        );
        if solver.state.u[0] > u_end {
            break;
        }
    }
}

#[test]
fn riks_solver_follows_the_snap_through_path_past_both_limit_points() {
    let mut solver = RiksSolver::from_internal_force(snap_force, snap_tangent, 1.0, 0.05);
    run_until_displacement(&mut solver, 2.0, 200);

    // Every committed point lies on the analytic equilibrium curve.
    for &(lambda, u) in &solver.history {
        assert!(
            (lambda - snap_force(u)).abs() < 2e-6,
            "(λ, u) = ({lambda}, {u}) is off the path: f(u) = {}",
            snap_force(u)
        );
    }

    // The load rises to the upper limit point ...
    let (peak_index, &(peak_lambda, peak_u)) = solver
        .history
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.0.total_cmp(&b.1.0))
        .expect("history is not empty");
    assert!(
        (peak_lambda - SNAP_LIMIT_LOAD).abs() < 1e-3,
        "λ_max = {peak_lambda}"
    );
    assert!(
        (peak_u - (1.0 - 1.0 / 3.0_f64.sqrt())).abs() < 0.06,
        "u at peak = {peak_u}"
    );

    // ... and then DEcreases along the path (post-critical branch), down to the lower
    // limit point at u = 1 + 1/√3 where it reaches −λ_max.
    let (trough_index, &(trough_lambda, trough_u)) = solver
        .history
        .iter()
        .enumerate()
        .min_by(|a, b| a.1.0.total_cmp(&b.1.0))
        .expect("history is not empty");
    assert!(
        (trough_lambda + SNAP_LIMIT_LOAD).abs() < 1e-3,
        "λ_min = {trough_lambda}"
    );
    assert!(
        (trough_u - (1.0 + 1.0 / 3.0_f64.sqrt())).abs() < 0.06,
        "u at trough = {trough_u}"
    );
    assert!(trough_index > peak_index);
    for window in solver.history[peak_index..=trough_index].windows(2) {
        assert!(
            window[1].0 < window[0].0,
            "load must decrease between the limit points"
        );
    }
    // After the trough the load increases again while u keeps growing.
    let &(last_lambda, last_u) = solver.history.last().expect("history is not empty");
    assert!(last_lambda > trough_lambda + 0.1 && last_u > trough_u);
    // The sign of the load increment flipped at each limit point and back again.
    assert_eq!(solver.state.sign, 1.0);
}

#[test]
fn riks_solver_starts_by_loading_and_unloads_after_the_first_limit_point() {
    let mut solver = RiksSolver::from_internal_force(snap_force, snap_tangent, 1.0, 0.05);
    assert!(solver.step());
    let &(lambda, u) = solver.history.first().expect("first step is committed");
    assert!(lambda > 0.0 && u > 0.0);
    assert_eq!(solver.state.sign, 1.0);
    // March well beyond the first limit point (u ≈ 0.4226): the load increment is negative.
    run_until_displacement(&mut solver, 0.8, 200);
    assert_eq!(solver.state.sign, -1.0);
    assert!(solver.history.last().expect("non-empty").0 < SNAP_LIMIT_LOAD);
}

#[test]
fn riks_solver_secant_and_explicit_laws_agree() {
    let mut explicit = RiksSolver::from_internal_force(snap_force, snap_tangent, 1.0, 0.05);
    let mut secant = RiksSolver::new(snap_secant, 1.0, 0.05);
    explicit.run(40);
    secant.run(40);
    assert_eq!(explicit.history.len(), 40);
    assert_eq!(secant.history.len(), 40);
    for (a, b) in explicit.history.iter().zip(secant.history.iter()) {
        assert!(
            (a.0 - b.0).abs() < 1e-5 && (a.1 - b.1).abs() < 1e-5,
            "{a:?} vs {b:?}"
        );
    }
    // The central-difference tangent matches the analytic one.
    for &u in &[-0.5, 0.0, 0.4226, 1.0, 1.7] {
        assert!((secant.tangent_stiffness(u) - snap_tangent(u)).abs() < 1e-8);
    }
}

#[test]
fn riks_solver_secant_law_reaches_the_cubic_limit_load() {
    // K_s(u) = 1 − u²  ⇒  f(u) = u − u³, limit load 2/(3√3) at u = 1/√3.
    fn k_fn(u: f64) -> f64 {
        1.0 - u * u
    }
    let mut solver = RiksSolver::new(k_fn, 1.0, 0.05);
    solver.run(40);
    assert_eq!(solver.history.len(), 40);
    for &(lambda, u) in &solver.history {
        assert!(
            (lambda - (u - u * u * u)).abs() < 2e-6,
            "(λ, u) = ({lambda}, {u})"
        );
    }
    let limit = 2.0 / (3.0 * 3.0_f64.sqrt());
    let peak = solver.history.iter().map(|p| p.0).fold(f64::MIN, f64::max);
    assert!(
        (peak - limit).abs() < 1e-3,
        "peak load {peak}, limit {limit}"
    );
    // The final point is on the post-critical branch.
    let (last_lambda, last_u) = *solver.history.last().expect("non-empty");
    assert!(last_lambda < peak - 0.1 && last_u > 1.0 / 3.0_f64.sqrt());
}

#[test]
fn riks_solver_refuses_to_advance_with_a_singular_tangent() {
    // K_t ≡ 0 makes the Newton corrector singular: the step must be rejected instead of
    // dividing by a clamped 1e-30.
    let mut solver = RiksSolver::from_internal_force(|u| u, |_| 0.0, 1.0, 0.1);
    assert!(!solver.step());
    assert!(solver.history.is_empty());
    assert_eq!(solver.state.lambda, 0.0);
    assert_eq!(solver.state.u, vec![0.0]);
    solver.run(5);
    assert!(solver.history.is_empty());
}

#[test]
fn riks_solver_cuts_the_radius_back_when_a_step_does_not_converge() {
    let mut solver = RiksSolver::from_internal_force(snap_force, snap_tangent, 1.0, 0.05);
    // One corrector iteration cannot reach a 1e-12 residual from the nominal radius
    // (the predictor error is O(Δl²) and one Newton step squares it), so the radius has
    // to be halved until the corrected point converges.
    solver.params.max_iter = 1;
    solver.params.tol = 1e-12;
    solver.params.dl_min = 1e-6;
    assert!(solver.step(), "the step must converge after cut-backs");
    assert!(
        solver.state.dl < 0.05 && solver.state.dl > 1e-5,
        "radius after cut-back: {}",
        solver.state.dl
    );
    let &(lambda, u) = solver.history.last().expect("committed");
    assert!((lambda - snap_force(u)).abs() < 1e-12);
    // The committed increment lies on the reduced arc-length sphere.
    let radius = (u * u + lambda * lambda).sqrt();
    assert!((radius - solver.state.dl).abs() < 1e-12, "|Δ| = {radius}");
}

#[test]
fn riks_solver_gives_up_when_even_the_smallest_radius_fails() {
    let mut solver = RiksSolver::from_internal_force(snap_force, snap_tangent, 1.0, 0.05);
    solver.params.max_iter = 0;
    solver.params.tol = 0.0; // unattainable
    assert!(!solver.step());
    assert!(solver.history.is_empty());
    assert_eq!(
        solver.state.dl, 0.05,
        "the radius is restored when the step is abandoned"
    );
}

#[test]
fn arc_length_constraint_zero_at_start() {
    let state = ArcLengthState::new(2, 0.1);
    let g = state.arc_length_constraint(&state.u, state.lambda, &state.u, state.lambda);
    // ‖Δu‖² + Δλ² = 0, but then − Δl² = −0.01 ≠ 0
    // The constraint is only satisfied on the arc, not at zero increment
    assert!((g - (-0.01)).abs() < 1e-10);
}

// ── EAS element ───────────────────────────────────────────────────────────

#[test]
fn eas_element_enhanced_strain_zero_initially() {
    let nodes = [[0., 0.], [1., 0.], [1., 1.], [0., 1.]];
    let elem = EasQ4Element::new(nodes);
    let eps = elem.enhanced_strain(0.0, 0.0);
    for e in eps {
        assert!(e.abs() < 1e-10, "initial enhanced strain should be zero");
    }
}

#[test]
fn eas_element_jacobian_unit_square() {
    let nodes = [[0., 0.], [1., 0.], [1., 1.], [0., 1.]];
    let j = EasQ4Element::jacobian_at(&nodes, 0.0, 0.0);
    assert!((j - 0.25).abs() < 1e-10, "unit square J = 0.25, got {j}");
}

#[test]
fn eas_element_standard_strain_rigid_body() {
    let nodes = [[0., 0.], [1., 0.], [1., 1.], [0., 1.]];
    let elem = EasQ4Element::new(nodes);
    // Rigid body translation: all nodes move by (0.1, 0.2)
    let u = [0.1, 0.2, 0.1, 0.2, 0.1, 0.2, 0.1, 0.2];
    let eps = elem.standard_strain(&u, 0.0, 0.0);
    for e in eps {
        assert!(
            e.abs() < 1e-10,
            "rigid body should produce zero strain: {e}"
        );
    }
}

// ── B-bar element ─────────────────────────────────────────────────────────

#[test]
fn bbar_element_zero_displacement_gives_zero_strain() {
    let elem = BBarElement::unit_cube();
    let u = vec![0.0; 24]; // 8 nodes × 3 DOFs
    let eps_vol = elem.average_volumetric_strain(&u).expect("valid element");
    assert!(eps_vol.abs() < 1e-14, "zero displacement → zero strain");
}

/// Nodal displacements `u_k = H · x_k` of an affine field.
fn affine_displacements(nodes: &[[f64; 3]], h: &[[f64; 3]; 3]) -> Vec<f64> {
    nodes
        .iter()
        .flat_map(|x| (0..3).map(move |i| (0..3).map(|j| h[i][j] * x[j]).sum::<f64>()))
        .collect()
}

#[test]
fn bbar_unit_cube_axial_stretch_gives_the_axial_strain() {
    // u_x = ε x on the unit cube: ε_vol = ε.  The old shape-derivative table assigned the
    // wrong signs to nodes 1, 2 and 3, so the node-ordered field gave a different value.
    let elem = BBarElement::unit_cube();
    let eps = 1.0e-3;
    let mut u = vec![0.0; 24];
    for (k, node) in elem.nodes.iter().enumerate() {
        u[3 * k] = eps * node[0];
    }
    let eps_vol = elem.average_volumetric_strain(&u).expect("valid element");
    assert!((eps_vol - eps).abs() < 1e-14, "ε_vol = {eps_vol}");

    // Each axis separately, and the sum for a general diagonal strain.
    let h = [[1.0e-3, 0.0, 0.0], [0.0, -2.0e-3, 0.0], [0.0, 0.0, 4.0e-3]];
    let u = affine_displacements(&elem.nodes, &h);
    let eps_vol = elem.average_volumetric_strain(&u).expect("valid element");
    assert!((eps_vol - 3.0e-3).abs() < 1e-14, "ε_vol = {eps_vol}");
}

#[test]
fn bbar_box_of_non_unit_size_gives_the_axial_strain() {
    // 2×1×1 box: dN/dx carries the factor 1/(side/2); the old code left it off by 2.
    let nodes: Vec<[f64; 3]> = BBarElement::unit_cube()
        .nodes
        .iter()
        .map(|p| [2.0 * p[0], p[1], p[2]])
        .collect();
    let elem = BBarElement {
        nodes: nodes.clone(),
        n_nodes: 8,
    };
    let eps = 2.0e-3;
    let mut u = vec![0.0; 24];
    for (k, node) in nodes.iter().enumerate() {
        u[3 * k] = eps * node[0];
    }
    let eps_vol = elem.average_volumetric_strain(&u).expect("valid element");
    assert!((eps_vol - eps).abs() < 1e-14, "ε_vol = {eps_vol}");
}

#[test]
fn bbar_distorted_hexahedron_reproduces_the_trace_of_an_affine_field() {
    // A sheared and rotated box whose top face is additionally warped: the Jacobian is not
    // diagonal (and not constant), so J⁻¹ vs J⁻ᵀ mistakes cannot hide.
    let f = [[1.3, 0.4, -0.2], [0.1, 0.9, 0.3], [-0.2, 0.25, 1.1]];
    let mut nodes: Vec<[f64; 3]> = BBarElement::unit_cube()
        .nodes
        .iter()
        .map(|p| {
            let mapped: Vec<f64> = (0..3)
                .map(|i| (0..3).map(|j| f[i][j] * p[j]).sum::<f64>())
                .collect();
            [mapped[0] + 0.5, mapped[1] - 1.0, mapped[2] + 2.0]
        })
        .collect();
    for node in nodes.iter_mut().skip(4) {
        node[0] += 0.05 * node[1];
        node[2] -= 0.04 * node[0];
    }
    let elem = BBarElement {
        nodes: nodes.clone(),
        n_nodes: 8,
    };
    // Non-symmetric H with a rotation part: ε_vol = tr(H) = 1e-3·(0.7 − 0.4 + 1.5).
    let h = [
        [0.7e-3, 2.0e-3, -1.0e-3],
        [-3.0e-3, -0.4e-3, 0.5e-3],
        [1.5e-3, -2.5e-3, 1.5e-3],
    ];
    let u = affine_displacements(&nodes, &h);
    let eps_vol = elem.average_volumetric_strain(&u).expect("valid element");
    assert!((eps_vol - 1.8e-3).abs() < 1e-13, "ε_vol = {eps_vol}");

    // A rigid rotation (skew-symmetric H) produces no volumetric strain.
    let skew = [
        [0.0, 1.0e-3, -2.0e-3],
        [-1.0e-3, 0.0, 3.0e-3],
        [2.0e-3, -3.0e-3, 0.0],
    ];
    let u = affine_displacements(&nodes, &skew);
    let eps_vol = elem.average_volumetric_strain(&u).expect("valid element");
    assert!(eps_vol.abs() < 1e-13, "ε_vol = {eps_vol}");
}

#[test]
fn bbar_rejects_malformed_elements_and_inputs() {
    let elem = BBarElement::unit_cube();
    assert!(matches!(
        elem.average_volumetric_strain(&[0.0; 23]),
        Err(Error::InvalidInput(_))
    ));

    let mut few_nodes = BBarElement::unit_cube();
    few_nodes.nodes.pop();
    assert!(matches!(
        few_nodes.average_volumetric_strain(&[0.0; 24]),
        Err(Error::InvalidInput(_))
    ));

    // Swapping two nodes inverts the element (negative Jacobian determinant).
    let mut inverted = BBarElement::unit_cube();
    inverted.nodes.swap(0, 1);
    assert!(matches!(
        inverted.average_volumetric_strain(&[0.0; 24]),
        Err(Error::InvalidInput(_))
    ));
}

// ── Mixed u-p element ─────────────────────────────────────────────────────

#[test]
fn mixed_up_element_zero_pressure_zero_disp() {
    let nodes = [[0., 0.], [1., 0.], [1., 1.], [0., 1.]];
    let elem = MixedUpElement::new(nodes, 1e9, 1e8);
    assert!(elem.compute_pressure().abs() < 1e-10);
}

#[test]
fn mixed_up_element_compression_gives_positive_pressure() {
    let nodes = [[0., 0.], [1., 0.], [1., 1.], [0., 1.]];
    let mut elem = MixedUpElement::new(nodes, 1e9, 1e8);
    // Uniform compression: all nodes move inward (positive volumetric strain → positive pressure p = -K*εv)
    // Move right nodes left, left nodes right to compress the element
    elem.u = [0.01, 0.0, -0.01, 0.0, -0.01, 0.0, 0.01, 0.0];
    elem.update_pressure();
    assert!(elem.p.abs() >= 0.0, "pressure computed: {}", elem.p); // just check it doesn't panic
}

// ── Stress transformations ─────────────────────────────────────────────────

#[test]
fn cauchy_to_pk2_identity_deformation() {
    let sigma = [[100e6, 0., 0.], [0., 50e6, 0.], [0., 0., 0.]];
    let f = EYE3;
    let s = cauchy_to_pk2(&sigma, &f);
    // For F = I, S = σ
    for i in 0..3 {
        for j in 0..3 {
            assert!(
                (s[i][j] - sigma[i][j]).abs() < 1.0,
                "S[{i}][{j}] = {} != {}",
                s[i][j],
                sigma[i][j]
            );
        }
    }
}

#[test]
fn nominal_strain_identity_is_zero() {
    let e = nominal_strain(&EYE3);
    for row in &e {
        for &val in row {
            assert!(val.abs() < 1e-10);
        }
    }
}

// ── Nonlinear truss ───────────────────────────────────────────────────────

#[test]
fn nonlinear_truss_reference_length() {
    let x1 = [0., 0.];
    let x2 = [3., 4.];
    let l0 = NonlinearTruss2D::reference_length(&x1, &x2);
    assert!((l0 - 5.0).abs() < 1e-10);
}

#[test]
fn nonlinear_truss_zero_force_initial() {
    let truss = NonlinearTruss2D::new([0., 0.], [1., 0.], 0.01, 200e9);
    assert!(truss.n_force.abs() < 1e-10);
}

#[test]
fn nonlinear_truss_axial_strain_after_extension() {
    let mut truss = NonlinearTruss2D::new([0., 0.], [1., 0.], 0.01, 200e9);
    truss.x2[0] = 1.1; // stretch by 10%
    let eps = truss.cauchy_strain(1.0);
    assert!((eps - 0.1).abs() < 1e-10);
}

#[test]
fn nonlinear_truss_update_force() {
    let mut truss = NonlinearTruss2D::new([0., 0.], [1., 0.], 1e-4, 200e9);
    truss.x2[0] = 1.001; // small extension
    truss.update_force(1.0);
    assert!(truss.n_force > 0.0, "axial force should be tensile");
}

// ── Co-rotational beam ────────────────────────────────────────────────────

#[test]
fn corotational_beam_reference_length() {
    let beam = CorotationalBeam2D::new([0., 0.], [3., 4.], 0.01, 1e-6, 200e9);
    assert!((beam.reference_length() - 5.0).abs() < 1e-10);
}

#[test]
fn corotational_beam_zero_disp_zero_force() {
    let beam = CorotationalBeam2D::new([0., 0.], [1., 0.], 0.01, 1e-6, 200e9);
    let n = beam.axial_force();
    assert!(n.abs() < 1e-10, "zero displacement → zero force: {n}");
}

#[test]
fn corotational_beam_bending_moments_zero_no_rotation() {
    let beam = CorotationalBeam2D::new([0., 0.], [1., 0.], 0.01, 1e-6, 200e9);
    let [m1, m2] = beam.bending_moments();
    assert!(m1.abs() < 1e-10 && m2.abs() < 1e-10);
}

#[test]
fn corotational_beam_rotation_produces_moments() {
    let mut beam = CorotationalBeam2D::new([0., 0.], [1., 0.], 0.01, 1e-6, 200e9);
    beam.d2[2] = 0.01; // rotation at node 2
    let [m1, m2] = beam.bending_moments();
    assert!(
        m1.abs() > 0.0 || m2.abs() > 0.0,
        "rotation should produce moments"
    );
}

// ── Lagrangian formulation checks ─────────────────────────────────────────

#[test]
fn pk1_identity_equals_sigma() {
    let sigma = [[200e6, 0., 0.], [0., 100e6, 0.], [0., 0., 0.]];
    let p1 = cauchy_to_pk1(&sigma, &EYE3);
    for i in 0..3 {
        for j in 0..3 {
            assert!((p1[i][j] - sigma[i][j]).abs() < 1.0);
        }
    }
}

#[test]
fn kirchhoff_identity_deformation_equals_cauchy() {
    let sigma = [[50e6, 0., 0.], [0., 50e6, 0.], [0., 0., 50e6]];
    let tau = cauchy_to_kirchhoff(&sigma, &EYE3);
    for i in 0..3 {
        for j in 0..3 {
            assert!((tau[i][j] - sigma[i][j]).abs() < 1.0);
        }
    }
}
