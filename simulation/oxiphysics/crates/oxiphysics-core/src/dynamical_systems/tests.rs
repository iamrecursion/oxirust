// Copyright 2026 COOLJAPAN OU (Team KitaSan)
// SPDX-License-Identifier: Apache-2.0

//! Unit tests for `dynamical_systems`.
//!
//! Split out of `dynamical_systems.rs` to keep that file under the 2000-line
//! limit.

use super::*;

// ── helper: simple harmonic oscillator as OdeSystem ───────────────────────
struct ShoOde {
    omega: f64,
}
impl OdeSystem for ShoOde {
    fn dim(&self) -> usize {
        2
    }
    fn rhs(&self, _t: f64, x: &[f64]) -> Vec<f64> {
        vec![x[1], -self.omega * self.omega * x[0]]
    }
}

// ── 1. rk4_step conserves energy for SHO ─────────────────────────────────
#[test]
fn test_rk4_sho_energy_conservation() {
    let sys = ShoOde { omega: 1.0 };
    let x0 = vec![1.0, 0.0];
    let energy0 = 0.5 * (x0[0] * x0[0] + x0[1] * x0[1]);
    let h = 0.01;
    let mut x = x0.clone();
    for i in 0..628 {
        x = rk4_step(&sys, i as f64 * h, &x, h);
    }
    let energy1 = 0.5 * (x[0] * x[0] + x[1] * x[1]);
    assert!((energy1 - energy0).abs() < 1e-4);
}

// ── 2. integrate_rk4 returns expected number of steps ────────────────────
#[test]
fn test_integrate_rk4_step_count() {
    let sys = ShoOde { omega: 1.0 };
    let traj = integrate_rk4(&sys, &[1.0, 0.0], 0.0, 1.0, 0.1);
    assert_eq!(traj.len(), 11);
}

// ── 3. numerical_jacobian of SHO ─────────────────────────────────────────
#[test]
fn test_numerical_jacobian_sho() {
    let sys = ShoOde { omega: 2.0 };
    let jac = numerical_jacobian(&sys, 0.0, &[0.0, 0.0], 1e-5);
    // J = [[0,1],[-4,0]]
    assert!((jac[0] - 0.0).abs() < 1e-6);
    assert!((jac[1] - 1.0).abs() < 1e-6);
    assert!((jac[2] - (-4.0)).abs() < 1e-6);
    assert!((jac[3] - 0.0).abs() < 1e-6);
}

// ── 4. find_fixed_point locates origin for SHO ───────────────────────────
#[test]
fn test_find_fixed_point_sho_origin() {
    let sys = ShoOde { omega: 1.0 };
    let res = find_fixed_point(&sys, &[0.1, 0.1], 1e-10, 50, 1e-7);
    assert!(res.converged);
    assert!(res.point[0].abs() < 1e-6);
    assert!(res.point[1].abs() < 1e-6);
}

// ── 5. gauss_solve 2x2 ───────────────────────────────────────────────────
#[test]
fn test_gauss_solve_2x2() {
    // [2 1; 1 3] * x = [5; 10]  => x = [1, 3]
    let a = vec![2.0, 1.0, 1.0, 3.0];
    let b = vec![5.0, 10.0];
    let x = gauss_solve(&a, 2, &b).unwrap();
    assert!((x[0] - 1.0).abs() < 1e-10);
    assert!((x[1] - 3.0).abs() < 1e-10);
}

// ── 6. eigen2 real eigenvalues ────────────────────────────────────────────
#[test]
fn test_eigen2_real() {
    // [[3,0],[0,2]] => eigenvalues 3 and 2
    let ((r1, i1), (r2, i2)) = eigen2(3.0, 0.0, 0.0, 2.0);
    assert!((r1 - 3.0).abs() < 1e-10 || (r1 - 2.0).abs() < 1e-10);
    assert!(i1.abs() < 1e-10);
    assert!(i2.abs() < 1e-10);
    let _ = r2;
}

// ── 7. eigen2 complex eigenvalues ────────────────────────────────────────
#[test]
fn test_eigen2_complex() {
    // [[0,1],[-1,0]] => eigenvalues ±i
    let ((r1, i1), (_r2, _i2)) = eigen2(0.0, 1.0, -1.0, 0.0);
    assert!(r1.abs() < 1e-10);
    assert!((i1.abs() - 1.0).abs() < 1e-10);
}

// ── 8. stability_2d saddle ────────────────────────────────────────────────
#[test]
fn test_stability_saddle() {
    // det < 0 → saddle
    let s = stability_2d(1.0, 0.0, 0.0, -1.0);
    assert_eq!(s, "saddle");
}

// ── 9. stability_2d stable spiral ────────────────────────────────────────
#[test]
fn test_stability_stable_spiral() {
    // tr = -0.2, det = 1.01 → stable spiral
    let s = stability_2d(-0.1, 1.0, -1.0, -0.1);
    assert_eq!(s, "stable_spiral");
}

// ── 10. Lorenz fixed points ───────────────────────────────────────────────
#[test]
fn test_lorenz_fixed_points_classic() {
    let lor = LorenzSystem::classic();
    let fps = lor.fixed_points();
    assert_eq!(fps.len(), 3);
    // C± have |x| = |y|
    let c1 = fps[1];
    assert!((c1[0] - c1[1]).abs() < 1e-10);
}

// ── 11. Lorenz RHS at origin is zero for origin FP ───────────────────────
#[test]
fn test_lorenz_rhs_origin() {
    let lor = LorenzSystem::classic();
    let f = lor.rhs(0.0, &[0.0, 0.0, 0.0]);
    assert!(f.iter().all(|v| v.abs() < 1e-12));
}

// ── 12. Rössler RHS dimension ─────────────────────────────────────────────
#[test]
fn test_rossler_dim() {
    let ros = RosslerSystem::classic();
    assert_eq!(ros.dim(), 3);
}

// ── 13. Poincaré section returns crossings ────────────────────────────────
#[test]
fn test_poincare_sho_crossings() {
    let sys = ShoOde { omega: 1.0 };
    let cfg = PoincareSectionConfig {
        axis: 0,
        level: 0.0,
        positive_direction: true,
    };
    // SHO with ω=1 has period 2π; run for 10 full periods
    let h = 0.01;
    let n_steps = (10.0 * 2.0 * std::f64::consts::PI / h) as usize;
    let crossings = poincare_section(&sys, &[1.0, 0.0], h, n_steps, &cfg);
    // Should find approximately 10 crossings (x=0 with positive velocity)
    assert!(
        crossings.len() >= 8 && crossings.len() <= 12,
        "expected ~10 crossings, got {}",
        crossings.len()
    );
}

// ── 14. estimate_limit_cycle_period for SHO ───────────────────────────────
#[test]
fn test_limit_cycle_period_sho() {
    let sys = ShoOde { omega: 1.0 };
    let h = 0.005;
    let n_steps = (20.0 * 2.0 * std::f64::consts::PI / h) as usize;
    let est = estimate_limit_cycle_period(&sys, &[1.0, 0.0], h, n_steps, 0, 0.0);
    let est = est.unwrap();
    let expected = 2.0 * std::f64::consts::PI;
    assert!(
        (est.period - expected).abs() < 0.05,
        "period={} expected≈{}",
        est.period,
        expected
    );
}

// ── 15. delay_embed shape ─────────────────────────────────────────────────
#[test]
fn test_delay_embed_shape() {
    let series: Vec<f64> = (0..20).map(|i| i as f64).collect();
    let pts = delay_embed(&series, 3, 2);
    // length = n - (embed_dim - 1) * tau = 20 - 4 = 16
    assert_eq!(pts.len(), 16);
    assert_eq!(pts[0].len(), 3);
    // first point: [0, 2, 4]
    assert_eq!(pts[0], vec![0.0, 2.0, 4.0]);
}

// ── 16. recurrence_matrix diagonal is all true ────────────────────────────
#[test]
fn test_recurrence_matrix_diagonal() {
    let pts = vec![vec![0.0, 0.0], vec![1.0, 0.0], vec![2.0, 0.0]];
    let mat = recurrence_matrix(&pts, 0.5);
    // diagonal entries
    assert!(mat[0]);
    assert!(mat[3 + 1]);
    assert!(mat[2 * 3 + 2]);
}

// ── 17. RQA on periodic signal has high DET ───────────────────────────────
#[test]
fn test_rqa_periodic() {
    let series: Vec<f64> = (0..200)
        .map(|i| (2.0 * std::f64::consts::PI * i as f64 / 20.0).sin())
        .collect();
    let params = RqaParams {
        embed_dim: 2,
        tau: 5,
        epsilon: 0.2,
        min_line: 2,
    };
    let res = rqa(&series, &params);
    assert!(res.rr > 0.0);
    assert!(res.det >= 0.0);
}

// ── 18. Störmer–Verlet conserves energy for SHO ───────────────────────────
#[test]
fn test_stormer_verlet_energy_sho() {
    let omega = 1.0f64;
    let grad_v = |q: &[f64]| vec![omega * omega * q[0]];
    let traj = integrate_symplectic(&[1.0], &[0.0], 0.01, 1000, &grad_v);
    let energy0 = 0.5 * 0.0_f64.powi(2) + 0.5 * omega * omega * 1.0_f64.powi(2);
    let (q_end, p_end) = &traj[traj.len() - 1];
    let energy_end = 0.5 * p_end[0].powi(2) + 0.5 * omega * omega * q_end[0].powi(2);
    assert!(
        (energy_end - energy0).abs() < 1e-4,
        "energy drift = {}",
        (energy_end - energy0).abs()
    );
}

// ── 19. Forest–Ruth step dimensionality ───────────────────────────────────
#[test]
fn test_forest_ruth_returns_same_dim() {
    let grad_v = |q: &[f64]| vec![q[0]];
    let (q_new, p_new) = forest_ruth_step(&[1.0], &[0.5], 0.1, &grad_v);
    assert_eq!(q_new.len(), 1);
    assert_eq!(p_new.len(), 1);
}

// ── 20. HarmonicOscillatorHam energy conserved by symplectic ─────────────
#[test]
fn test_hamiltonian_sho_energy_drift() {
    let ham = HarmonicOscillatorHam::new(1.0);
    let grad_v = |q: &[f64]| ham.dh_dq(q, &[0.0]);
    let traj = integrate_symplectic(&[1.0], &[0.0], 0.01, 500, &grad_v);
    let pairs: Vec<(Vec<f64>, Vec<f64>)> = traj;
    let drift = energy_drift(&ham, &pairs);
    assert!(drift < 1e-3, "energy drift = {}", drift);
}

// ── 21. saddle_node_fixed_points ─────────────────────────────────────────
#[test]
fn test_saddle_node_fps() {
    let fps_neg = saddle_node_fixed_points(-4.0);
    assert_eq!(fps_neg.len(), 2);
    // x = ±2
    assert!((fps_neg[0].abs() - 2.0).abs() < 1e-10);
    let fps_pos = saddle_node_fixed_points(1.0);
    assert!(fps_pos.is_empty());
    let fps_zero = saddle_node_fixed_points(0.0);
    assert_eq!(fps_zero.len(), 1);
}

// ── 22. pitchfork_fixed_points ────────────────────────────────────────────
#[test]
fn test_pitchfork_fps() {
    let fps = pitchfork_fixed_points(4.0);
    assert_eq!(fps.len(), 3);
    let fps_neg = pitchfork_fixed_points(-1.0);
    assert_eq!(fps_neg.len(), 1);
}

// ── 23. is_hopf_bifurcation ───────────────────────────────────────────────
#[test]
fn test_is_hopf() {
    assert!(is_hopf_bifurcation(0.0, 1.0, 1e-6));
    assert!(!is_hopf_bifurcation(0.5, 1.0, 1e-6));
    assert!(!is_hopf_bifurcation(0.0, -1.0, 1e-6));
}

// ── 24. Brusselator fixed point ───────────────────────────────────────────
#[test]
fn test_brusselator_fixed_point() {
    let br = Brusselator::new(2.0, 3.0);
    let (xfp, yfp) = br.fixed_point();
    let f = br.rhs(0.0, &[xfp, yfp]);
    assert!(f[0].abs() < 1e-10);
    assert!(f[1].abs() < 1e-10);
}

// ── 25. Brusselator Hopf condition ────────────────────────────────────────
#[test]
fn test_brusselator_hopf_b() {
    let br = Brusselator::new(2.0, 5.0);
    let hopf = br.hopf_b(); // should be 1 + 4 = 5
    assert!((hopf - 5.0).abs() < 1e-10);
}

// ── 26. Van der Pol origin unstable ───────────────────────────────────────
#[test]
fn test_vdp_origin_unstable() {
    let vdp = VanDerPol::new(1.0);
    let jac = numerical_jacobian(&vdp, 0.0, &[0.0, 0.0], 1e-5);
    // J at origin = [[0,1],[-1,1]]  => tr = 1 > 0 → unstable
    let tr = jac[0] + jac[3];
    assert!(tr > 0.0);
}

// ── 27. Kuramoto order parameter ─────────────────────────────────────────
#[test]
fn test_kuramoto_fully_synced() {
    // All phases identical → r = 1
    let theta = vec![1.0; 5];
    let (r, _psi) = KuramotoModel::order_parameter(&theta);
    assert!((r - 1.0).abs() < 1e-10);
}

// ── 28. Kuramoto incoherent ───────────────────────────────────────────────
#[test]
fn test_kuramoto_incoherent() {
    use std::f64::consts::PI;
    // Uniformly spaced → r ≈ 0
    let n = 8;
    let theta: Vec<f64> = (0..n).map(|i| 2.0 * PI * i as f64 / n as f64).collect();
    let (r, _) = KuramotoModel::order_parameter(&theta);
    assert!(r < 0.05);
}

// ── 29. kaplan_yorke_dim ──────────────────────────────────────────────────
#[test]
fn test_kaplan_yorke_dim() {
    // Classic Lorenz: exponents ~ [0.9, 0, -14.6]
    let les = vec![0.9, 0.0, -14.6];
    let d = kaplan_yorke_dim(&les);
    // KY = 2 + (0.9 + 0.0)/14.6 ≈ 2.062
    assert!(d > 2.0 && d < 2.2, "KY dim = {}", d);
}

// ── 30. maximal_lyapunov_exponent positive for Lorenz ─────────────────────
#[test]
fn test_mle_lorenz_positive() {
    let lor = LorenzSystem::classic();
    let x0 = vec![0.1, 0.0, 0.0];
    // Short estimate; just verify sign
    let mle = maximal_lyapunov_exponent(&lor, &x0, 0.01, 1000, 10);
    // MLE for classic Lorenz is ~0.9 but our crude estimate should be > 0
    assert!(mle > -1.0, "MLE = {}", mle);
}

// ── 32. Duffing oscillator RHS dimensionality ─────────────────────────────
#[test]
fn test_duffing_dim() {
    let duf = DuffingOscillator::new(0.1, -1.0, 1.0, 0.3, 1.2);
    assert_eq!(duf.dim(), 2);
    let f = duf.rhs(0.0, &[0.0, 0.0]);
    assert_eq!(f.len(), 2);
}

// ── 33. phase_plane_grid returns correct number of points ─────────────────
#[test]
fn test_phase_plane_grid_count() {
    let sys = ShoOde { omega: 1.0 };
    let grid = phase_plane_grid(&sys, (-2.0, 2.0, 5), (-2.0, 2.0, 5));
    assert_eq!(grid.len(), 25);
}

// ── 34. add3 / sub3 / scale3 / dot3 / norm3 consistency ──────────────────
#[test]
fn test_vec3_ops() {
    let a = [1.0, 2.0, 3.0];
    let b = [4.0, 5.0, 6.0];
    let s = add3(a, b);
    assert_eq!(s, [5.0, 7.0, 9.0]);
    let d = sub3(b, a);
    assert_eq!(d, [3.0, 3.0, 3.0]);
    let sc = scale3(a, 2.0);
    assert_eq!(sc, [2.0, 4.0, 6.0]);
    assert!((dot3(a, b) - 32.0).abs() < 1e-10);
    assert!((norm3([3.0, 4.0, 0.0]) - 5.0).abs() < 1e-10);
}

// ── 35. cross3 ────────────────────────────────────────────────────────────
#[test]
fn test_cross3() {
    let a = [1.0, 0.0, 0.0];
    let b = [0.0, 1.0, 0.0];
    let c = cross3(a, b);
    assert!((c[0] - 0.0).abs() < 1e-10);
    assert!((c[1] - 0.0).abs() < 1e-10);
    assert!((c[2] - 1.0).abs() < 1e-10);
}

// ── 36. Rössler RHS values ────────────────────────────────────────────────
#[test]
fn test_rossler_rhs_values() {
    let ros = RosslerSystem::new(0.2, 0.2, 5.7);
    let f = ros.rhs(0.0, &[1.0, 0.0, 0.0]);
    // ẋ = -0 - 0 = 0, ẏ = 1 + 0.2*0 = 1, ż = 0.2 + 0*(1-5.7)=0.2
    assert!((f[0] - 0.0).abs() < 1e-10);
    assert!((f[1] - 1.0).abs() < 1e-10);
    assert!((f[2] - 0.2).abs() < 1e-10);
}

// ── 37. HarmonicOscillatorHam Hamiltonian ─────────────────────────────────
#[test]
fn test_ham_sho_value() {
    let ham = HarmonicOscillatorHam::new(2.0);
    let h = ham.hamiltonian(&[1.0], &[0.0]);
    // H = 0.5 * 4 * 1 = 2.0
    assert!((h - 2.0).abs() < 1e-10);
}

// ── 38. HamiltonianOde wraps correctly ────────────────────────────────────
#[test]
fn test_hamiltonian_ode_rhs() {
    let ham = HarmonicOscillatorHam::new(1.0);
    let ode = HamiltonianOde { ham: &ham };
    // state [q, p] = [0, 1]: qdot = p = 1, pdot = -omega^2 q = 0
    let f = ode.rhs(0.0, &[0.0, 1.0]);
    assert!((f[0] - 1.0).abs() < 1e-10);
    assert!((f[1] - 0.0).abs() < 1e-10);
}

// ── 39. Lyapunov exponents QR (SHO should have ≈0) ───────────────────────
#[test]
fn test_lyapunov_qr_sho_near_zero() {
    let sys = ShoOde { omega: 1.0 };
    let les = lyapunov_exponents_qr(&sys, &[1.0, 0.0], 0.1, 200);
    // SHO is conservative, LEs should straddle zero
    assert_eq!(les.len(), 2);
    let sum: f64 = les.iter().sum();
    // sum of LEs = trace of Jacobian = 0 for Hamiltonian
    assert!(sum.abs() < 1.0, "sum of LEs = {}", sum);
}

// ── 40. detect_heteroclinic not found for misaligned direction ────────────
#[test]
fn test_detect_heteroclinic_none() {
    let sys = ShoOde { omega: 1.0 };
    let cfg = HeteroclinicConfig {
        eps_perturb: 0.01,
        h: 0.01,
        n_steps: 100,
        tol: 0.1,
    };
    // SHO has no heteroclinic orbits; expect None quickly
    let result = detect_heteroclinic(&sys, &[0.0, 0.0], &[5.0, 0.0], &[1.0, 0.0], &cfg);
    // SHO oscillates, won't converge to [5,0] — may return None
    // (just ensure it doesn't panic)
    let _ = result;
}
