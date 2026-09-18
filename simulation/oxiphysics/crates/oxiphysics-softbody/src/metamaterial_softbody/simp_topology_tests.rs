// Copyright 2026 COOLJAPAN OU (Team KitaSan)
// SPDX-License-Identifier: Apache-2.0

//! Tests for the SIMP topology optimisation of metamaterial unit cells.

use super::*;

#[test]
fn test_topology_opt_volume_fraction_init() {
    let opt = MetamaterialOptimization::new(10, 10, 0.01, 1e9, 0.5, -0.3);
    let vf = opt.current_volume_fraction();
    assert!(
        (vf - 0.5).abs() < 1e-10,
        "initial VF should match target: {vf}"
    );
    assert_eq!(opt.nx(), 10);
    assert_eq!(opt.ny(), 10);
    assert!((opt.element_size() - 0.001).abs() < 1e-15);
}

#[test]
fn test_simp_penalized_stiffness() {
    let elem = DesignElement::new([0.0; 2], 0.001, 1e9);
    let k_full = elem.penalized_stiffness(3.0);
    assert!(
        (k_full - 1e9).abs() < 1.0,
        "full density gives full stiffness"
    );
    let mut elem2 = elem.clone();
    elem2.density = 0.5;
    let k_half = elem2.penalized_stiffness(3.0);
    assert!(k_half < k_full * 0.5, "p=3 penalises intermediate density");
    // Modified SIMP: floor at ρ = 0 and exact derivative.
    elem2.density = 0.0;
    assert!((elem2.simp_modulus(3.0, 1e-9) - 1.0).abs() < 1e-9);
    elem2.density = 0.4;
    let h = 1e-6;
    let mut ep = elem2.clone();
    ep.density += h;
    let mut em = elem2.clone();
    em.density -= h;
    let fd = (ep.simp_modulus(3.0, 1e-9) - em.simp_modulus(3.0, 1e-9)) / (2.0 * h);
    let an = elem2.simp_modulus_derivative(3.0, 1e-9);
    assert!((fd - an).abs() < 1e-6 * an.abs());
}

#[test]
fn q4_stiffness_matches_sigmund_closed_form() {
    let nu = 0.3_f64;
    let ke = q4_plane_stress_stiffness(nu);
    let k = [
        0.5 - nu / 6.0,
        0.125 + nu / 8.0,
        -0.25 - nu / 12.0,
        -0.125 + 3.0 * nu / 8.0,
        -0.25 + nu / 12.0,
        -0.125 - nu / 8.0,
        nu / 6.0,
        0.125 - 3.0 * nu / 8.0,
    ];
    let pattern = [
        [0, 1, 2, 3, 4, 5, 6, 7],
        [1, 0, 7, 6, 5, 4, 3, 2],
        [2, 7, 0, 5, 6, 3, 4, 1],
        [3, 6, 5, 0, 7, 2, 1, 4],
        [4, 5, 6, 7, 0, 1, 2, 3],
        [5, 4, 3, 2, 1, 0, 7, 6],
        [6, 3, 4, 1, 2, 7, 0, 5],
        [7, 2, 1, 4, 3, 6, 5, 0],
    ];
    let scale = 1.0 / (1.0 - nu * nu);
    for r in 0..8 {
        let mut row_sum = 0.0;
        for c in 0..8 {
            let expected = scale * k[pattern[r][c]];
            assert!((ke[r][c] - expected).abs() < 1e-14, "KE[{r}][{c}]");
            assert!((ke[r][c] - ke[c][r]).abs() < 1e-15);
            row_sum += ke[r][c];
        }
        // Rigid translations carry no energy: x-DOF and y-DOF row sums vanish.
        assert!(row_sum.abs() < 1e-14);
    }
}

#[test]
fn compliance_sensitivities_are_negative_and_match_finite_differences() {
    let mut opt = MetamaterialOptimization::new(4, 3, 4.0, 1.0, 0.5, -0.3);
    for (i, e) in opt.elements.iter_mut().enumerate() {
        e.density = 0.3 + 0.05 * i as f64;
    }
    let load = SimpLoadCase::cantilever(4, 3, 1.0);
    let base = opt.solve_compliance(&load).expect("solve");
    assert!(base.compliance > 0.0);
    assert!(base.sensitivities.iter().all(|&s| s < 0.0));
    // fᵀu equals Σ Eₑ uₑᵀ k₀ uₑ.
    let f_dot_u: f64 = load
        .forces
        .iter()
        .zip(base.displacements.iter())
        .map(|(f, u)| f * u)
        .sum();
    assert!((f_dot_u - base.compliance).abs() < 1e-8 * base.compliance);
    let h = 1e-6;
    for e in [0usize, 5, 11] {
        let mut plus = opt.clone();
        plus.elements[e].density += h;
        let mut minus = opt.clone();
        minus.elements[e].density -= h;
        let cp = plus.solve_compliance(&load).expect("solve+").compliance;
        let cm = minus.solve_compliance(&load).expect("solve-").compliance;
        let fd = (cp - cm) / (2.0 * h);
        let an = base.sensitivities[e];
        assert!(
            (fd - an).abs() < 1e-5 * an.abs().max(1e-12),
            "element {e}: fd={fd} analytic={an}"
        );
    }
}

#[test]
fn uniform_sensitivities_are_not_uniform_under_load() {
    // The old implementation produced identical sensitivities for a uniform
    // design; a real FE solve gives spatially varying strain energy.
    let opt = MetamaterialOptimization::new(12, 4, 3.0, 1e9, 0.5, -0.3);
    let load = SimpLoadCase::cantilever(12, 4, 1.0);
    let s = opt.compute_sensitivities(&load).expect("solve");
    let min = s.iter().copied().fold(f64::INFINITY, f64::min);
    let max = s.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    assert!(min < 0.0 && max < 0.0);
    assert!(
        max / min < 0.5,
        "sensitivities should vary: min={min} max={max}"
    );
}

#[test]
fn cantilever_optimisation_produces_non_uniform_design() {
    let (nx, ny) = (24, 8);
    let mut opt = MetamaterialOptimization::new(nx, ny, 3.0, 1.0, 0.5, -0.3);
    opt.filter_radius = 1.5 * opt.element_size();
    let load = SimpLoadCase::cantilever(nx, ny, 1.0);
    let first = opt.step_compliance(&load).expect("iteration 0");
    let mut last = first;
    for _ in 0..30 {
        last = opt.step_compliance(&load).expect("iteration");
        assert!(
            (opt.current_volume_fraction() - 0.5).abs() < 1e-3,
            "volume fraction drifted: {}",
            opt.current_volume_fraction()
        );
    }
    let min = opt.elements.iter().map(|e| e.density).fold(1.0, f64::min);
    let max = opt.elements.iter().map(|e| e.density).fold(0.0, f64::max);
    assert!(
        max - min > 0.5,
        "design should be non-uniform: [{min}, {max}]"
    );
    assert!(
        last.objective < 0.8 * first.objective,
        "compliance should drop: {} -> {}",
        first.objective,
        last.objective
    );
    assert_eq!(opt.iteration, 31);
}

#[test]
fn filter_neighbors_are_local_and_filter_preserves_constant_field() {
    let mut opt = MetamaterialOptimization::new(10, 6, 1.0, 1.0, 0.5, 0.0);
    opt.filter_radius = 1.5 * opt.element_size();
    let nb = opt.filter_neighbors();
    assert_eq!(nb.len(), 60);
    // Interior element: itself + 4 edge neighbours + 4 diagonal neighbours.
    assert_eq!(nb[2 * 10 + 4].len(), 9);
    // Corner element: itself + 2 + 1.
    assert_eq!(nb[0].len(), 4);
    // Uniform density and sensitivity: the filter is the identity.
    let raw = vec![-2.0; 60];
    let filtered = opt.filter_sensitivities(&raw);
    assert!(filtered.iter().all(|&v| (v + 2.0).abs() < 1e-12));
}

#[test]
fn homogenized_solid_recovers_plane_stress_matrix() {
    let mut opt = MetamaterialOptimization::new(4, 4, 1.0, 2.0, 1.0, 0.0);
    for e in opt.elements.iter_mut() {
        e.density = 1.0;
    }
    let h = opt.homogenize().expect("homogenize");
    let nu = 0.3;
    let e_eff = 2.0 * (1e-9 + (1.0 - 1e-9));
    let s = e_eff / (1.0 - nu * nu);
    let d = [
        [s, s * nu, 0.0],
        [s * nu, s, 0.0],
        [0.0, 0.0, s * (1.0 - nu) / 2.0],
    ];
    for (i, (row_h, row_d)) in h.c_h.iter().zip(d.iter()).enumerate() {
        for (j, (ch, dd)) in row_h.iter().zip(row_d.iter()).enumerate() {
            assert!((ch - dd).abs() < 1e-9, "C_H[{i}][{j}] = {ch} vs {dd}");
        }
    }
    assert!((h.poisson_ratio - nu).abs() < 1e-10);
}

#[test]
fn homogenized_layered_laminate_matches_voigt_reuss() {
    // Vertical layers (alternating columns) of solid / soft material:
    // C_22 along the layers follows the Voigt average, C_11 across the
    // layers follows the Reuss average of the plane-strain-like moduli.
    let mut opt = MetamaterialOptimization::new(4, 4, 1.0, 1.0, 0.5, 0.0);
    opt.material_poisson = 0.0;
    for (i, e) in opt.elements.iter_mut().enumerate() {
        e.density = if (i % 4) < 2 { 1.0 } else { 0.5 };
    }
    let e1 = 1.0;
    let e2 = 1e-9 + 0.125 * (1.0 - 1e-9);
    let h = opt.homogenize().expect("homogenize");
    let voigt = 0.5 * (e1 + e2);
    let reuss = 1.0 / (0.5 / e1 + 0.5 / e2);
    assert!((h.c_h[1][1] - voigt).abs() < 1e-9, "C22={}", h.c_h[1][1]);
    assert!((h.c_h[0][0] - reuss).abs() < 1e-9, "C11={}", h.c_h[0][0]);
}

#[test]
fn poisson_sensitivities_match_finite_differences() {
    let mut opt = MetamaterialOptimization::new(4, 3, 1.0, 1.0, 0.5, -0.2);
    for (i, e) in opt.elements.iter_mut().enumerate() {
        e.density = 0.35 + 0.04 * ((i * 7) % 11) as f64;
    }
    let base = opt.poisson_objective().expect("objective");
    let h = 1e-6;
    for e in [0usize, 4, 9] {
        let mut plus = opt.clone();
        plus.elements[e].density += h;
        let mut minus = opt.clone();
        minus.elements[e].density -= h;
        let jp = plus.poisson_objective().expect("+").objective;
        let jm = minus.poisson_objective().expect("-").objective;
        let fd = (jp - jm) / (2.0 * h);
        let an = base.sensitivities[e];
        assert!(
            (fd - an).abs() < 1e-5 * an.abs().max(1e-8),
            "element {e}: fd={fd} analytic={an}"
        );
    }
}

#[test]
fn poisson_optimisation_reduces_objective() {
    let mut opt = MetamaterialOptimization::new(10, 10, 1.0, 1.0, 0.5, 0.05);
    let first = opt.step_poisson().expect("first step");
    let mut last = first;
    for _ in 0..15 {
        last = opt.step_poisson().expect("step");
    }
    assert!(
        last.objective < 0.5 * first.objective,
        "objective {} -> {}",
        first.objective,
        last.objective
    );
    assert!((opt.current_volume_fraction() - 0.5).abs() < 1e-6);
    let nu = opt.poisson_objective().expect("final").poisson_ratio;
    assert!((nu - 0.05).abs() < (0.3_f64 - 0.05).abs(), "nu={nu}");
}

#[test]
fn uniform_design_has_zero_poisson_gradient() {
    let opt = MetamaterialOptimization::new(6, 6, 1.0, 1.0, 0.5, -0.3);
    let obj = opt.poisson_objective().expect("objective");
    assert!((obj.poisson_ratio - 0.3).abs() < 1e-9);
    assert!(obj.sensitivities.iter().all(|s| s.abs() < 1e-9));
}

#[test]
fn invalid_inputs_are_rejected() {
    let opt = MetamaterialOptimization::new(4, 3, 1.0, 1.0, 0.5, 0.0);
    let wrong = SimpLoadCase::cantilever(3, 3, 1.0);
    assert!(opt.solve_compliance(&wrong).is_err());
    let free = SimpLoadCase::new(4, 3);
    assert!(opt.solve_compliance(&free).is_err());
    let mut lc = SimpLoadCase::new(4, 3);
    assert!(lc.fix_node(5, 0).is_err());
    assert!(lc.add_nodal_force(0, 4, [1.0, 0.0]).is_err());
    let mut broken = opt.clone();
    broken.elements.pop();
    assert!(broken.homogenize().is_err());
}
