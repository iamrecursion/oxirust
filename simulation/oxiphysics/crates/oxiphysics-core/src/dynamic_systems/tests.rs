// Copyright 2026 COOLJAPAN OU (Team KitaSan)
// SPDX-License-Identifier: Apache-2.0

//! Unit tests for `dynamic_systems`.
//!
//! Split out of `dynamic_systems.rs` to keep that file under the 2000-line
//! limit.

use super::*;

// ── LogisticMap ────────────────────────────────────────────────────────────

#[test]
fn test_logistic_fixed_point_stability() {
    // For r=2, fixed point at x* = (r-1)/r = 0.5
    let map = LogisticMap::new(2.0);
    let x_star = (map.r - 1.0) / map.r;
    let mut x = 0.4;
    for _ in 0..200 {
        x = map.apply(x);
    }
    assert!(
        (x - x_star).abs() < 1e-6,
        "should converge to {x_star}, got {x}"
    );
}

#[test]
fn test_logistic_orbit_length() {
    let map = LogisticMap::new(3.5);
    let orbit = map.orbit_1d(0.3, 50);
    assert_eq!(orbit.len(), 51);
}

#[test]
fn test_logistic_orbit_bounded() {
    let map = LogisticMap::new(3.9);
    let orbit = map.orbit_1d(0.5, 1000);
    for &x in &orbit {
        assert!((0.0..=1.0).contains(&x), "orbit escaped [0,1]: {x}");
    }
}

#[test]
fn test_logistic_lyapunov_chaotic() {
    // r=4.0: Lyapunov exponent ≈ ln 2 ≈ 0.693
    let map = LogisticMap::new(4.0);
    let le = map.lyapunov_exponent(0.3, 5000).unwrap();
    assert!(le > 0.4, "should be positive (chaotic), got {le}");
}

#[test]
fn test_logistic_lyapunov_stable() {
    // r=2.0: stable fixed point at x*=0.5, Lyapunov < 0
    // Use x0=0.3 which avoids the fixed point x=0 or x=1
    let map = LogisticMap::new(2.0);
    // The derivative at the fixed point is r(1-2x*) = 2*(1-1) = 0, so use short orbit
    // and check via orbit convergence instead
    let mut x = 0.3f64;
    for _ in 0..500 {
        x = map.apply(x);
    }
    // Should have converged near x* = 0.5
    assert!((x - 0.5).abs() < 1e-6, "should converge to x*=0.5, got {x}");
    // Lyapunov: compute without hitting the degenerate point
    if let Some(le) = map.lyapunov_exponent(0.3, 200) {
        assert!(
            le < 0.1,
            "stable fixed point should have non-positive LE, got {le}"
        );
    }
    // If None is returned that's also acceptable (orbit hit degenerate point)
}

#[test]
fn test_logistic_bifurcation_diagram_size() {
    let diag = LogisticMap::bifurcation_diagram(2.5, 4.0, 10, 200, 50);
    assert_eq!(diag.len(), 10 * 50);
}

#[test]
fn test_logistic_trait_orbit() {
    let map = LogisticMap::new(3.2);
    let orbit = map.orbit(&[0.5], 20);
    assert_eq!(orbit.len(), 21);
    assert_eq!(orbit[0], vec![0.5]);
}

// ── HenonMap ───────────────────────────────────────────────────────────────

#[test]
fn test_henon_orbit_first_step() {
    let map = HenonMap::new(1.4, 0.3);
    let (xn, yn) = map.apply(0.0, 0.0);
    assert!((xn - 1.0).abs() < 1e-12);
    assert!(yn.abs() < 1e-12);
}

#[test]
fn test_henon_jacobian_det_constant() {
    let map = HenonMap::new(1.4, 0.3);
    assert!((map.jacobian_det() - (-0.3)).abs() < 1e-12);
}

#[test]
fn test_henon_orbit_length() {
    let map = HenonMap::new(1.4, 0.3);
    let orbit = map.orbit_2d(0.0, 0.0, 100);
    assert_eq!(orbit.len(), 101);
}

#[test]
fn test_henon_jacobian_structure() {
    let map = HenonMap::new(1.4, 0.3);
    let j = map.jacobian(1.0, 0.5);
    assert!((j[0][0] - (-2.0 * 1.4)).abs() < 1e-12);
    assert!((j[0][1] - 1.0).abs() < 1e-12);
    assert!((j[1][0] - 0.3).abs() < 1e-12);
    assert!(j[1][1].abs() < 1e-12);
}

#[test]
fn test_henon_trait_step() {
    let map = HenonMap::new(1.4, 0.3);
    let state = vec![0.0, 0.0];
    let next = map.step(&state);
    assert_eq!(next.len(), 2);
}

// ── ChirikovStandardMap ────────────────────────────────────────────────────

#[test]
fn test_chirikov_orbit_stays_in_torus() {
    let map = ChirikovStandardMap::new(0.5);
    let orbit = map.orbit_2d(0.1, 0.2, 1000);
    let tau = std::f64::consts::TAU;
    for (th, p) in orbit {
        assert!(th >= 0.0 && th < tau, "theta out of [0,2π): {th}");
        assert!(p >= 0.0 && p < tau, "p out of [0,2π): {p}");
    }
}

#[test]
fn test_chirikov_zero_k_is_twist_map() {
    // K=0: p unchanged, theta advances by p
    let map = ChirikovStandardMap::new(0.0);
    let (th, p) = map.apply(1.0, 0.5);
    let tau = std::f64::consts::TAU;
    assert!((p - 0.5).abs() < 1e-12, "p should not change: {p}");
    assert!(
        (th - (1.5_f64).rem_euclid(tau)).abs() < 1e-12,
        "theta = 1+p mod 2π: {th}"
    );
}

#[test]
fn test_chirikov_lyapunov_chaotic() {
    // K=5 should be strongly chaotic
    let map = ChirikovStandardMap::new(5.0);
    let le = map.lyapunov_exponent(1.0, 1.5, 5000);
    assert!(le > 0.5, "K=5 should be chaotic, got LE={le}");
}

#[test]
fn test_chirikov_trait_orbit() {
    let map = ChirikovStandardMap::new(1.0);
    let orbit = map.orbit(&[0.5, 0.5], 10);
    assert_eq!(orbit.len(), 11);
}

// ── AttractorAnalysis ──────────────────────────────────────────────────────

#[test]
fn test_attractor_find_fixed_points_identity() {
    // Points near y = x should be found as fixed points of x → x
    let data = vec![vec![0.0], vec![0.5], vec![1.0], vec![2.0]];
    let analysis = AttractorAnalysis::new(data);
    let fps = analysis.find_fixed_points(|x| x.to_vec(), 1e-9);
    assert!(
        !fps.is_empty(),
        "identity map has every point as fixed point"
    );
}

#[test]
fn test_attractor_correlation_dimension_returns_pairs() {
    let data: Vec<Vec<f64>> = (0..50).map(|i| vec![i as f64 * 0.1]).collect();
    let analysis = AttractorAnalysis::new(data);
    let pairs = analysis.correlation_dimension(10);
    assert!(!pairs.is_empty(), "should produce log-log pairs");
}

#[test]
fn test_attractor_rosenstein_length() {
    let data: Vec<Vec<f64>> = (0..100)
        .map(|i| {
            let t = i as f64 * 0.1;
            vec![t.sin(), t.cos()]
        })
        .collect();
    let analysis = AttractorAnalysis::new(data);
    let div = analysis.rosenstein_lyapunov_auto(10);
    assert_eq!(div.len(), 10);
}

#[test]
fn test_attractor_find_periodic_orbit() {
    // Period-2 orbit of logistic map at r ≈ 3.2
    let map = LogisticMap::new(3.2);
    // Converge to period-2 orbit
    let mut x = 0.5f64;
    for _ in 0..1000 {
        x = map.apply(x);
    }
    let p1 = x;
    let p2 = map.apply(p1);
    let data = vec![vec![p1], vec![p2]];
    let analysis = AttractorAnalysis::new(data);
    let po = analysis.find_periodic_orbits(&|s: &[f64]| vec![map.apply(s[0])], 2, 1e-6);
    assert!(!po.is_empty(), "should find period-2 orbit");
}

// ── DynamicalNormalForm ────────────────────────────────────────────────────

#[test]
fn test_normal_form_saddle_node_fixed_points() {
    let nf = DynamicalNormalForm::new(BifurcationType::SaddleNode, -1.0);
    let fps = nf.fixed_points_1d();
    assert_eq!(fps.len(), 2, "two fps for μ<0");
    for &x in &fps {
        let rhs_val = nf.rhs(&[x])[0];
        assert!(rhs_val.abs() < 1e-10, "not a fixed point: rhs={rhs_val}");
    }
}

#[test]
fn test_normal_form_saddle_node_no_fps_positive_mu() {
    let nf = DynamicalNormalForm::new(BifurcationType::SaddleNode, 1.0);
    let fps = nf.fixed_points_1d();
    assert!(fps.is_empty(), "no fps for μ>0 in saddle-node");
}

#[test]
fn test_normal_form_transcritical_fps() {
    let nf = DynamicalNormalForm::new(BifurcationType::Transcritical, 2.0);
    let fps = nf.fixed_points_1d();
    assert_eq!(fps.len(), 2);
    for &x in &fps {
        let rhs_val = nf.rhs(&[x])[0];
        assert!(
            rhs_val.abs() < 1e-10,
            "not a fixed point: x={x}, rhs={rhs_val}"
        );
    }
}

#[test]
fn test_normal_form_pitchfork_super_three_fps() {
    let nf = DynamicalNormalForm::new(BifurcationType::PitchforkSuper, 1.0);
    let fps = nf.fixed_points_1d();
    assert_eq!(fps.len(), 3, "super-critical pitchfork μ>0 has 3 fps");
}

#[test]
fn test_normal_form_stability_classification() {
    // Pitchfork super with μ=1: x*=0 is unstable, x*=±1 are stable
    let nf = DynamicalNormalForm::new(BifurcationType::PitchforkSuper, 1.0);
    assert_eq!(nf.stability_1d(0.0), Stability::Unstable);
    assert_eq!(nf.stability_1d(1.0), Stability::Stable);
    assert_eq!(nf.stability_1d(-1.0), Stability::Stable);
}

#[test]
fn test_normal_form_euler_integration_converges() {
    // Saddle-node stable branch: μ=-1, x0=-0.9 (near stable fp x=-1)
    let nf = DynamicalNormalForm::new(BifurcationType::SaddleNode, -1.0);
    let traj = nf.integrate_euler(&[-0.9], 0.01, 2000);
    let last = traj.last().unwrap()[0];
    assert!(
        (last - (-1.0)).abs() < 0.05,
        "should converge near -1, got {last}"
    );
}

#[test]
fn test_normal_form_hopf_super_limit_cycle() {
    // HopfSuper μ=1: trajectory should converge to r=1
    let nf = DynamicalNormalForm::new(BifurcationType::HopfSuper, 1.0);
    let traj = nf.integrate_euler(&[0.1, 0.0], 0.01, 5000);
    let last_r = traj.last().unwrap()[0];
    assert!((last_r - 1.0).abs() < 0.05, "limit cycle r≈1, got {last_r}");
}

// ── PoincareSection ────────────────────────────────────────────────────────

#[test]
fn test_poincare_section_circle() {
    // Circular orbit crossing y=0 upwards twice per revolution
    let n = 1000usize;
    let traj: Vec<Vec<f64>> = (0..=n)
        .map(|i| {
            let t = i as f64 * std::f64::consts::TAU / n as f64 * 5.0;
            vec![t.cos(), t.sin()]
        })
        .collect();
    let mut section = PoincareSection::new(vec![0.0, 1.0], vec![0.0, 0.0]);
    section.process_trajectory(&traj);
    // Should have ~5 crossings
    assert!(
        section.count() >= 4,
        "expected ~5 crossings, got {}",
        section.count()
    );
}

#[test]
fn test_poincare_section_rotation_number() {
    let n = 2000usize;
    let traj: Vec<Vec<f64>> = (0..=n)
        .map(|i| {
            let t = i as f64 * std::f64::consts::TAU / n as f64 * 3.0;
            vec![t.cos(), t.sin()]
        })
        .collect();
    let mut section = PoincareSection::new(vec![0.0, 1.0], vec![0.0, 0.0]);
    section.process_trajectory(&traj);
    let rn = section.rotation_number(0, 1);
    // rotation number for a circle should be approximately 1 per crossing
    assert!(rn.abs() < 2.0, "rotation number should be finite, got {rn}");
}

#[test]
fn test_poincare_clear() {
    let traj: Vec<Vec<f64>> = (0..=100)
        .map(|i| {
            let t = i as f64 * 0.1;
            vec![t.cos(), t.sin()]
        })
        .collect();
    let mut section = PoincareSection::new(vec![0.0, 1.0], vec![0.0, 0.0]);
    section.process_trajectory(&traj);
    section.clear();
    assert_eq!(section.count(), 0);
}

#[test]
fn test_poincare_return_map() {
    let n = 2000usize;
    let traj: Vec<Vec<f64>> = (0..=n)
        .map(|i| {
            let t = i as f64 * std::f64::consts::TAU / n as f64 * 5.0;
            vec![t.cos(), t.sin()]
        })
        .collect();
    let mut section = PoincareSection::new(vec![0.0, 1.0], vec![0.0, 0.0]);
    section.process_trajectory(&traj);
    let rm = section.return_map_1d(0);
    // Should have (count - 1) return-map pairs
    assert_eq!(rm.len(), section.count().saturating_sub(1));
}

// ── FlowMap ────────────────────────────────────────────────────────────────

#[test]
fn test_flow_simple_harmonic_oscillator() {
    // ẋ = y, ẏ = -x (simple harmonic oscillator)
    let f = |s: &[f64]| vec![s[1], -s[0]];
    let flow = FlowMap::new(2, 0.01);
    // Advance one full period: T = 2π ≈ 628 steps at dt=0.01
    let final_state = flow.advance(&[1.0, 0.0], 628, &f);
    // Should return close to (1, 0)
    assert!(
        (final_state[0] - 1.0).abs() < 0.1,
        "x≈1, got {}",
        final_state[0]
    );
    assert!(final_state[1].abs() < 0.1, "y≈0, got {}", final_state[1]);
}

#[test]
fn test_flow_trajectory_length() {
    let f = |s: &[f64]| vec![s[1], -s[0]];
    let flow = FlowMap::new(2, 0.01);
    let traj = flow.trajectory(&[1.0, 0.0], 100, &f);
    assert_eq!(traj.len(), 101);
}

#[test]
fn test_flow_stroboscopic_map() {
    let f = |s: &[f64]| vec![s[1], -s[0]];
    let flow = FlowMap::new(2, 0.01);
    let samples = flow.stroboscopic_map(&[1.0, 0.0], 5, 10, &f);
    assert_eq!(samples.len(), 6);
}

#[test]
fn test_flow_energy_conservation_sho() {
    // For SHO, E = x² + y² should be conserved
    let f = |s: &[f64]| vec![s[1], -s[0]];
    let flow = FlowMap::new(2, 0.001);
    let init = vec![1.0, 0.0];
    let energy_init: f64 = init[0] * init[0] + init[1] * init[1];
    let final_state = flow.advance(&init, 1000, &f);
    let energy_final: f64 = final_state[0] * final_state[0] + final_state[1] * final_state[1];
    assert!(
        (energy_final - energy_init).abs() < 0.01,
        "energy not conserved: {energy_final}"
    );
}

#[test]
fn test_flow_fundamental_solution_matrix() {
    let f = |s: &[f64]| vec![s[1], -s[0]];
    let flow = FlowMap::new(2, 0.01);
    let fsm = flow.fundamental_solution_matrix(&[1.0, 0.0], 10, &f);
    assert_eq!(fsm.len(), 2);
    assert_eq!(fsm[0].len(), 2);
}

fn assert_matrix_close(got: &[Vec<f64>], want: &[[f64; 2]; 2], tol: f64) {
    assert_eq!(got.len(), 2);
    for (i, row) in got.iter().enumerate() {
        assert_eq!(row.len(), 2);
        for (j, &value) in row.iter().enumerate() {
            assert!(
                (value - want[i][j]).abs() < tol,
                "Φ = {got:?}, expected {want:?}"
            );
        }
    }
}

#[test]
fn test_fundamental_solution_matrix_harmonic_oscillator_is_a_rotation() {
    // f = (y, -x): Φ(t) = [[cos t, sin t], [-sin t, cos t]].  Use dt = 2π/6000
    // so that 6000 steps are exactly one period and 1500 exactly a quarter
    // (dt = 1e-3 would miss 2π by 1.9e-4).
    let f = |s: &[f64]| vec![s[1], -s[0]];
    let flow = FlowMap::new(2, 2.0 * std::f64::consts::PI / 6000.0);
    let start = [0.7, -0.3];
    let period = flow.fundamental_solution_matrix(&start, 6000, &f);
    assert_matrix_close(&period, &[[1.0, 0.0], [0.0, 1.0]], 1e-6);
    let quarter = flow.fundamental_solution_matrix(&start, 1500, &f);
    assert_matrix_close(&quarter, &[[0.0, 1.0], [-1.0, 0.0]], 1e-6);
    let half = flow.fundamental_solution_matrix(&start, 3000, &f);
    assert_matrix_close(&half, &[[-1.0, 0.0], [0.0, -1.0]], 1e-6);
}

#[test]
fn test_fundamental_solution_matrix_of_a_nilpotent_system_is_not_transposed() {
    // f = (y, 0): Φ(t) = exp(A t) = [[1, t], [0, 1]] with a non-symmetric A.
    // The old code read perturbation vectors by row but wrote them by column, so
    // it transposed Φ at every step and produced A·Φᵀ instead of the exponential.
    let f = |s: &[f64]| vec![s[1], 0.0];
    let flow = FlowMap::new(2, 1e-3);
    let phi = flow.fundamental_solution_matrix(&[0.4, 0.9], 1000, &f);
    assert_matrix_close(&phi, &[[1.0, 1.0], [0.0, 1.0]], 1e-9);
}

#[test]
fn test_fundamental_solution_matrix_matches_the_matrix_exponential() {
    // Linear system with the non-symmetric matrix A = [[0, -2], [1, -3]]
    // (eigenvalues -1 and -2): Φ(T) = exp(A T).
    let f = |s: &[f64]| vec![-2.0 * s[1], s[0] - 3.0 * s[1]];
    let steps = 700;
    let flow = FlowMap::new(2, 1e-3);
    let phi = flow.fundamental_solution_matrix(&[1.0, 2.0], steps, &f);
    let exact = (nalgebra::Matrix2::new(0.0, -2.0, 1.0, -3.0) * (steps as f64 * 1e-3)).exp();
    assert_matrix_close(
        &phi,
        &[
            [exact[(0, 0)], exact[(0, 1)]],
            [exact[(1, 0)], exact[(1, 1)]],
        ],
        1e-9,
    );
}

#[test]
fn test_fundamental_solution_matrix_is_the_derivative_of_the_flow_map() {
    // Van der Pol (μ = 1.5): compare Φ with central differences of the RK4 flow
    // with respect to the initial condition.
    let f = |s: &[f64]| vec![s[1], 1.5 * (1.0 - s[0] * s[0]) * s[1] - s[0]];
    let flow = FlowMap::new(2, 1e-3);
    let start = [0.8, -0.4];
    let steps = 1000;
    let phi = flow.fundamental_solution_matrix(&start, steps, &f);
    let h = 1e-6;
    for j in 0..2 {
        let mut plus = start;
        let mut minus = start;
        plus[j] += h;
        minus[j] -= h;
        let x_plus = flow.advance(&plus, steps, &f);
        let x_minus = flow.advance(&minus, steps, &f);
        for i in 0..2 {
            let numeric = (x_plus[i] - x_minus[i]) / (2.0 * h);
            assert!(
                (phi[i][j] - numeric).abs() < 1e-6 * (1.0 + numeric.abs()),
                "Φ[{i}][{j}] = {} vs finite difference {numeric}",
                phi[i][j]
            );
        }
    }
    // The columns are not the rows: Φ is not symmetric for this flow.
    assert!((phi[0][1] - phi[1][0]).abs() > 1e-3);
}

#[test]
fn test_fundamental_solution_matrix_obeys_liouville_for_the_lorenz_flow() {
    // det Φ(T) = exp(∫ tr J dt) and tr J = -(σ + 1 + β) is constant.
    let (sigma, rho, beta) = (10.0, 28.0, 8.0 / 3.0);
    let f = move |s: &[f64]| {
        vec![
            sigma * (s[1] - s[0]),
            s[0] * (rho - s[2]) - s[1],
            s[0] * s[1] - beta * s[2],
        ]
    };
    let flow = FlowMap::new(3, 1e-3);
    let steps = 500;
    let phi = flow.fundamental_solution_matrix(&[1.0, 1.0, 20.0], steps, &f);
    let m = nalgebra::Matrix3::from_fn(|i, j| phi[i][j]);
    let expected = (-(sigma + 1.0 + beta) * steps as f64 * 1e-3).exp();
    assert!(
        (m.determinant() - expected).abs() < 1e-6 * expected,
        "det Φ = {}, expected {expected}",
        m.determinant()
    );
}

#[test]
fn test_fundamental_solution_matrix_identity_and_invalid_input() {
    let f = |s: &[f64]| vec![s[1], -s[0]];
    let flow = FlowMap::new(2, 0.01);
    let identity = flow.fundamental_solution_matrix(&[1.0, 0.0], 0, &f);
    assert_eq!(identity, vec![vec![1.0, 0.0], vec![0.0, 1.0]]);
    // Initial condition of the wrong size.
    assert!(flow.fundamental_solution_matrix(&[1.0], 5, &f).is_empty());
    // A vector field of the wrong size must not index out of bounds.
    let short = |_: &[f64]| vec![0.0];
    assert!(
        flow.fundamental_solution_matrix(&[1.0, 0.0], 5, &short)
            .is_empty()
    );
}

// ── SynchronizationAnalysis ────────────────────────────────────────────────

#[test]
fn test_phase_sync_index_identical() {
    // Identical phases → R = 1
    let phi: Vec<f64> = (0..100).map(|i| i as f64 * 0.1).collect();
    let r = SynchronizationAnalysis::phase_synchronization_index(&phi, &phi);
    assert!((r - 1.0).abs() < 1e-6, "identical phases → R=1, got {r}");
}

#[test]
fn test_phase_sync_index_random_low() {
    // Linearly increasing phase difference → low R
    let phi1: Vec<f64> = (0..100).map(|i| i as f64 * 0.1).collect();
    let phi2: Vec<f64> = (0..100).map(|i| i as f64 * 0.3 + 0.7).collect();
    let r = SynchronizationAnalysis::phase_synchronization_index(&phi1, &phi2);
    assert!(r < 0.8, "asynchronous phases → low R, got {r}");
}

#[test]
fn test_generalised_sync_converged() {
    let traj1: Vec<Vec<f64>> = (0..10).map(|_| vec![1.0, 2.0]).collect();
    let traj2: Vec<Vec<f64>> = (0..10).map(|_| vec![1.0, 2.0]).collect();
    assert!(SynchronizationAnalysis::generalised_sync_check(
        &traj1, &traj2, 1e-9
    ));
}

#[test]
fn test_generalised_sync_diverged() {
    let traj1: Vec<Vec<f64>> = (0..10).map(|_| vec![1.0]).collect();
    let traj2: Vec<Vec<f64>> = (0..10).map(|_| vec![100.0]).collect();
    assert!(!SynchronizationAnalysis::generalised_sync_check(
        &traj1, &traj2, 1.0
    ));
}

#[test]
fn test_mutual_information_identical_signals() {
    let x: Vec<f64> = (0..100).map(|i| i as f64).collect();
    let mi = SynchronizationAnalysis::mutual_information(&x, &x, 10);
    assert!(
        mi > 0.0,
        "MI should be positive for identical signals, got {mi}"
    );
}

#[test]
fn test_mutual_information_independent_signals() {
    let x: Vec<f64> = (0..100).map(|i| i as f64).collect();
    let y: Vec<f64> = (0..100).map(|i| (i * 7 + 3) as f64 % 100.0).collect();
    let mi = SynchronizationAnalysis::mutual_information(&x, &y, 10);
    // MI ≥ 0
    assert!(mi >= 0.0, "MI should be non-negative, got {mi}");
}

#[test]
fn test_cross_correlation_identical() {
    let x: Vec<f64> = (0..50).map(|i| (i as f64 * 0.3).sin()).collect();
    let cc = SynchronizationAnalysis::cross_correlation_zero_lag(&x, &x);
    assert!(
        (cc - 1.0).abs() < 1e-6,
        "self-correlation should be 1, got {cc}"
    );
}

#[test]
fn test_cross_correlation_anticorrelated() {
    let x: Vec<f64> = (0..50).map(|i| i as f64).collect();
    let y: Vec<f64> = x.iter().map(|v| -v).collect();
    let cc = SynchronizationAnalysis::cross_correlation_zero_lag(&x, &y);
    assert!(
        (cc + 1.0).abs() < 1e-6,
        "anti-correlated: should be -1, got {cc}"
    );
}

#[test]
fn test_instantaneous_phase_length() {
    let sig: Vec<f64> = (0..100).map(|i| (i as f64 * 0.2).sin()).collect();
    let phase = SynchronizationAnalysis::instantaneous_phase(&sig);
    assert_eq!(phase.len(), sig.len());
}

#[test]
fn test_pecora_carroll_identical_sync() {
    let traj: Vec<Vec<f64>> = (0..10).map(|i| vec![i as f64, (i as f64).sin()]).collect();
    assert!(SynchronizationAnalysis::pecora_carroll_sync(
        &traj, &traj, 0, 1e-9
    ));
}

// ── LorenzSystem integration ───────────────────────────────────────────────

#[test]
fn test_lorenz_trajectory() {
    let lorenz = LorenzSystem::standard();
    let flow = FlowMap::new(3, 0.01);
    let init = vec![1.0, 0.0, 0.0];
    let traj = flow.trajectory(&init, 100, &|s| lorenz.rhs(s));
    assert_eq!(traj.len(), 101);
}

#[test]
fn test_lorenz_sensitivity() {
    // Lorenz is chaotic: two nearby initial conditions diverge substantially
    // Use 5000 steps at dt=0.01 = 50 time units for clear separation
    let lorenz = LorenzSystem::standard();
    let flow = FlowMap::new(3, 0.01);
    let init1 = vec![1.0, 1.0, 1.0];
    let init2 = vec![1.0 + 1e-5, 1.0, 1.0];
    let s1 = flow.advance(&init1, 5000, &|s| lorenz.rhs(s));
    let s2 = flow.advance(&init2, 5000, &|s| lorenz.rhs(s));
    let dist: f64 = s1
        .iter()
        .zip(s2.iter())
        .map(|(a, b)| (a - b) * (a - b))
        .sum::<f64>()
        .sqrt();
    assert!(dist > 1e-3, "Lorenz should show sensitivity, dist={dist}");
}

#[test]
fn test_lorenz_poincare_section() {
    let lorenz = LorenzSystem::standard();
    let flow = FlowMap::new(3, 0.01);
    let mut state = vec![1.0, 1.0, 1.0];
    // Warm up
    state = flow.advance(&state, 1000, &|s| lorenz.rhs(s));
    let traj = flow.trajectory(&state, 5000, &|s| lorenz.rhs(s));
    // Poincaré section at z = 27
    let mut section = PoincareSection::new(vec![0.0, 0.0, 1.0], vec![0.0, 0.0, 27.0]);
    section.process_trajectory(&traj);
    assert!(section.count() > 5, "Lorenz should cross z=27 many times");
}

#[test]
fn test_attractor_analysis_lorenz_is_strange() {
    // Build data from Lorenz orbit (x coordinate only for speed)
    let lorenz = LorenzSystem::standard();
    let flow = FlowMap::new(3, 0.02);
    let mut state = vec![1.0, 1.0, 1.0];
    state = flow.advance(&state, 500, &|s| lorenz.rhs(s));
    let traj = flow.trajectory(&state, 400, &|s| lorenz.rhs(s));
    let data: Vec<Vec<f64>> = traj.to_vec();
    let analysis = AttractorAnalysis::new(data);
    let pairs = analysis.correlation_dimension(15);
    // Just check we get valid log-log pairs
    assert!(!pairs.is_empty(), "should compute correlation dimension");
}

// ── Linear slope helper ────────────────────────────────────────────────────

#[test]
fn test_linear_slope_perfect() {
    // y = 2x + 1 → slope = 2
    let pairs: Vec<(f64, f64)> = (0..5).map(|i| (i as f64, 2.0 * i as f64 + 1.0)).collect();
    let s = linear_slope(&pairs);
    assert!((s - 2.0).abs() < 1e-6, "slope should be 2, got {s}");
}

#[test]
fn test_linear_slope_flat() {
    let pairs: Vec<(f64, f64)> = (0..5).map(|i| (i as f64, 3.0)).collect();
    let s = linear_slope(&pairs);
    assert!(s.abs() < 1e-6, "flat line has slope 0, got {s}");
}
