// Copyright 2026 COOLJAPAN OU (Team KitaSan)
// SPDX-License-Identifier: Apache-2.0

// Tests of the grad-h correction factor Ω (`VariableResolutionKernel::omega_factor`,
// `SmoothingLengthUpdate::omega_correction`). Ω used to be the constant `1 + 1/d`
// for every particle distribution (so `corrected_gradient` scaled every gradient
// by 0.75 in 3-D); the true factor is ≈ 1 for a uniform lattice and depends on
// the neighbourhood.

use super::*;
use std::f64::consts::PI;

/// Deterministic pseudo-random numbers in `[0, 1)` (64-bit LCG).
fn lcg_unit(state: &mut u64) -> f64 {
    *state = state
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    ((*state >> 11) as f64) / ((1u64 << 53) as f64)
}

/// Lattice of `n^dim` particles (z and y collapsed for lower dimensions) with
/// spacing `dx`. Returns the positions, the equal masses `mass` and the
/// position of the central particle.
fn lattice(dim: u32, n: usize, dx: f64, mass: f64) -> (Vec<[f64; 3]>, Vec<f64>, [f64; 3]) {
    let nz = if dim >= 3 { n } else { 1 };
    let ny = if dim >= 2 { n } else { 1 };
    let mut positions = Vec::new();
    for i in 0..n {
        for j in 0..ny {
            for k in 0..nz {
                positions.push([i as f64 * dx, j as f64 * dx, k as f64 * dx]);
            }
        }
    }
    let half = (n / 2) as f64 * dx;
    let centre = [
        half,
        if dim >= 2 { half } else { 0.0 },
        if dim >= 3 { half } else { 0.0 },
    ];
    let masses = vec![mass; positions.len()];
    (positions, masses, centre)
}

/// SPH density sum `ρ = Σ_j m_j W(|r_i − r_j|, h)` with the kernel under test.
fn density(
    kernel: &VariableResolutionKernel,
    pos_i: [f64; 3],
    h: f64,
    positions: &[[f64; 3]],
    masses: &[f64],
) -> f64 {
    positions
        .iter()
        .zip(masses.iter())
        .map(|(p, &m)| {
            let r =
                ((p[0] - pos_i[0]).powi(2) + (p[1] - pos_i[1]).powi(2) + (p[2] - pos_i[2]).powi(2))
                    .sqrt();
            m * kernel.kernel_w(r, h)
        })
        .sum()
}

/// Ω from a central finite difference of the density sum:
/// `Ω = 1 + h/(d ρ) ∂ρ/∂h` (the definition used by the auditors of the closed form).
fn omega_by_finite_difference(
    kernel: &VariableResolutionKernel,
    pos_i: [f64; 3],
    h: f64,
    positions: &[[f64; 3]],
    masses: &[f64],
    dim: u32,
) -> f64 {
    let rho = density(kernel, pos_i, h, positions, masses);
    let delta = 1.0e-5 * h;
    let rho_plus = density(kernel, pos_i, h + delta, positions, masses);
    let rho_minus = density(kernel, pos_i, h - delta, positions, masses);
    let drho_dh = (rho_plus - rho_minus) / (2.0 * delta);
    1.0 + h * drho_dh / (f64::from(dim) * rho)
}

#[test]
fn omega_of_a_uniform_lattice_is_close_to_one_and_not_the_old_constant() {
    // Reference values computed independently (NumPy) for the d-dimensional
    // cubic spline with h = 1.2 dx at the centre of a (2n+1)^d lattice.
    let cases = [
        (1_u32, 15_usize, 1.020_539),
        (2, 9, 0.991_034),
        (3, 7, 0.980_894),
    ];
    for (dim, n, expected) in cases {
        let dx = 0.1;
        let h = 1.2 * dx;
        let (positions, masses, centre) = lattice(dim, n, dx, 1000.0 * dx.powi(dim as i32));
        let kernel = VariableResolutionKernel::new(2.0, dim);
        let rho = density(&kernel, centre, h, &positions, &masses);
        let omega = kernel.omega_factor(centre, h, &positions, &masses, rho);
        assert!(
            (omega - expected).abs() < 1.0e-5,
            "dim {dim}: Ω = {omega}, expected {expected}"
        );
        assert!(
            (omega - 1.0).abs() < 0.05,
            "dim {dim}: |Ω − 1| = {}",
            (omega - 1.0).abs()
        );
        let old_constant = 1.0 + 1.0 / f64::from(dim);
        assert!(
            (omega - old_constant).abs() > 0.3,
            "dim {dim}: Ω = {omega} is still the constant 1 + 1/d"
        );
    }
}

#[test]
fn omega_matches_the_derivative_of_the_density_sum() {
    let mut state = 0x5eed_u64;
    for dim in [1_u32, 2, 3] {
        let kernel = VariableResolutionKernel::new(2.0, dim);
        for trial in 0..5 {
            let h = 0.05 + 0.02 * trial as f64;
            let n = 30 + 5 * trial;
            // Random neighbourhood inside 2.5 h (some points beyond the support)
            // plus the particle itself at the origin.
            let mut positions = vec![[0.0_f64; 3]];
            let mut masses = vec![0.7 + lcg_unit(&mut state)];
            for _ in 0..n {
                let mut p = [0.0_f64; 3];
                for coord in p.iter_mut().take(dim as usize) {
                    *coord = 2.5 * h * (2.0 * lcg_unit(&mut state) - 1.0);
                }
                positions.push(p);
                masses.push(0.7 + lcg_unit(&mut state));
            }
            let rho = density(&kernel, positions[0], h, &positions, &masses);
            let omega = kernel.omega_factor(positions[0], h, &positions, &masses, rho);
            let reference =
                omega_by_finite_difference(&kernel, positions[0], h, &positions, &masses, dim);
            assert!(
                (omega - reference).abs() < 1.0e-6 * (1.0 + reference.abs()),
                "dim {dim}, trial {trial}: Ω = {omega}, finite difference {reference}"
            );
        }
    }
}

#[test]
fn kernel_derivative_with_respect_to_h_matches_finite_differences() {
    for dim in [1_u32, 2, 3] {
        let kernel = VariableResolutionKernel::new(2.0, dim);
        let h = 0.2;
        let delta = 1.0e-6 * h;
        for r in [0.0, 0.05, 0.19, 0.2, 0.31, 0.39] {
            let fd =
                (kernel.kernel_w(r, h + delta) - kernel.kernel_w(r, h - delta)) / (2.0 * delta);
            let analytic = kernel.kernel_dw_dh(r, h);
            assert!(
                (analytic - fd).abs() < 1.0e-5 * fd.abs().max(1.0),
                "dim {dim}, r = {r}: ∂W/∂h = {analytic}, finite difference {fd}"
            );
        }
        // Beyond the support the kernel does not depend on h.
        assert_eq!(kernel.kernel_dw_dh(0.45, h), 0.0);
    }
}

#[test]
fn omega_agrees_with_the_three_dimensional_adaptive_h_implementation() {
    let dx = 0.1;
    let h = 1.2 * dx;
    let (positions, masses, centre) = lattice(3, 7, dx, 1.0);
    let kernel = VariableResolutionKernel::new(2.0, 3);
    let rho = density(&kernel, centre, h, &positions, &masses);
    let ours = kernel.omega_factor(centre, h, &positions, &masses, rho);
    let theirs = crate::adaptive_h::omega_correction(centre, h, &positions, &masses, rho, 3.0);
    assert!((ours - theirs).abs() < 1.0e-12, "{ours} vs {theirs}");
    // ... and with the SPH density the 3-D module itself computes.
    let rho_h = crate::adaptive_h::sph_density(centre, h, &positions, &masses);
    assert!((rho - rho_h).abs() < 1.0e-12 * rho);
}

#[test]
fn omega_depends_on_the_neighbourhood() {
    let h = 0.12;
    let mass = 1000.0 * 0.1_f64.powi(3);
    let kernel = VariableResolutionKernel::new(2.0, 3);
    let omega_for = |positions: &[[f64; 3]], masses: &[f64], centre: [f64; 3]| {
        let rho = density(&kernel, centre, h, positions, masses);
        kernel.omega_factor(centre, h, positions, masses, rho)
    };

    let (positions, masses, centre) = lattice(3, 7, 0.1, mass);
    let regular = omega_for(&positions, &masses, centre);
    assert!((regular - 0.980_894).abs() < 1.0e-5);

    // A denser lattice with the same smoothing length: Ω = 1.008313.
    let (dense_positions, dense_masses, dense_centre) = lattice(3, 7, 0.08, mass);
    let dense = omega_for(&dense_positions, &dense_masses, dense_centre);
    assert!(
        (dense - 1.008_313).abs() < 1.0e-5,
        "dense lattice: Ω = {dense}"
    );

    // Removing the nearest neighbour changes Ω (reference 1.005583).
    let nearest = positions
        .iter()
        .enumerate()
        .filter(|(_, p)| **p != centre)
        .min_by(|(_, a), (_, b)| {
            let da: f64 = (0..3).map(|d| (a[d] - centre[d]).powi(2)).sum();
            let db: f64 = (0..3).map(|d| (b[d] - centre[d]).powi(2)).sum();
            da.total_cmp(&db)
        })
        .map(|(index, _)| index)
        .expect("lattice has neighbours");
    let mut reduced_positions = positions.clone();
    let mut reduced_masses = masses.clone();
    reduced_positions.remove(nearest);
    reduced_masses.remove(nearest);
    let reduced = omega_for(&reduced_positions, &reduced_masses, centre);
    assert!(
        (reduced - 1.005_583).abs() < 1.0e-5,
        "reduced neighbourhood: Ω = {reduced}"
    );

    // An isolated particle: ρ ∝ h^{-d}, hence Ω = 0 exactly.
    let isolated = omega_for(&[centre], &[mass], centre);
    assert!(
        isolated.abs() < 1.0e-12,
        "isolated particle: Ω = {isolated}"
    );
}

#[test]
fn smoothing_length_update_uses_the_same_factor() {
    let dx = 0.1;
    let h = 1.2 * dx;
    for dim in [1_u32, 2, 3] {
        let (positions, masses, centre) = lattice(dim, 7, dx, 1.0);
        let kernel = VariableResolutionKernel::new(2.0, dim);
        let rho = density(&kernel, centre, h, &positions, &masses);
        let expected = kernel.omega_factor(centre, h, &positions, &masses, rho);
        let slu = SmoothingLengthUpdate::new(h, rho, dim);
        let omega = slu.omega_correction(centre, h, &positions, &masses, rho);
        assert!(
            (omega - expected).abs() < 1.0e-14,
            "dim {dim}: {omega} vs {expected}"
        );
        let old_constant = 1.0 + 1.0 / f64::from(dim);
        assert!((omega - old_constant).abs() > 0.1, "dim {dim}: Ω = {omega}");
    }
}

#[test]
fn degenerate_inputs_leave_the_gradient_uncorrected() {
    let kernel = VariableResolutionKernel::new(2.0, 3);
    let (positions, masses, centre) = lattice(3, 5, 0.1, 1.0);
    // No density, no smoothing length, NaN input: Ω = 1 (no correction).
    assert_eq!(
        kernel.omega_factor(centre, 0.12, &positions, &masses, 0.0),
        1.0
    );
    assert_eq!(
        kernel.omega_factor(centre, 0.0, &positions, &masses, 5.0),
        1.0
    );
    assert_eq!(
        kernel.omega_factor(centre, -0.1, &positions, &masses, 5.0),
        1.0
    );
    assert_eq!(
        kernel.omega_factor(centre, f64::NAN, &positions, &masses, 5.0),
        1.0
    );
    assert_eq!(
        kernel.omega_factor(centre, 0.12, &positions, &masses, f64::NAN),
        1.0
    );
    assert_eq!(kernel.omega_factor(centre, 0.12, &[], &[], 5.0), 1.0);

    // corrected_gradient = raw / Ω for a sane Ω ...
    assert!((kernel.corrected_gradient(10.0, 0.8) - 12.5).abs() < 1.0e-12);
    assert!((kernel.corrected_gradient(-3.0, 1.25) + 2.4).abs() < 1.0e-12);
    // ... and leaves the gradient alone for degenerate factors.
    for omega in [0.0, -0.5, 0.5 * OMEGA_MIN, f64::NAN, f64::INFINITY] {
        assert_eq!(kernel.corrected_gradient(10.0, omega), 10.0, "Ω = {omega}");
    }
}

#[test]
fn kernel_normalisation_matches_the_dimension_used_for_omega() {
    // The 1-D, 2-D and 3-D kernels integrate to one over their support; this
    // pins the dimension `d` entering `d W + r ∂W/∂r` in Ω.
    for dim in [1_u32, 2, 3] {
        let kernel = VariableResolutionKernel::new(2.0, dim);
        let h = 0.3;
        let steps = 4000;
        let dr = 2.0 * h / steps as f64;
        let integral: f64 = (0..steps)
            .map(|i| {
                let r = (i as f64 + 0.5) * dr;
                let shell = match dim {
                    1 => 2.0,
                    2 => 2.0 * PI * r,
                    _ => 4.0 * PI * r * r,
                };
                kernel.kernel_w(r, h) * shell * dr
            })
            .sum();
        assert!(
            (integral - 1.0).abs() < 1.0e-4,
            "dim {dim}: ∫W dV = {integral}"
        );
    }
}
