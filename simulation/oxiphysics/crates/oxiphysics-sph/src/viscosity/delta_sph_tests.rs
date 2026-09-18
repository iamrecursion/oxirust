// Copyright 2026 COOLJAPAN OU (Team KitaSan)
// SPDX-License-Identifier: Apache-2.0

//! Regression tests for the δ-SPH density diffusion terms of
//! [`super`]: the term must diffuse density from heavy to light particles
//! (it used to be anti-diffusive, and the tests only checked finiteness).

use super::*;
use crate::kernel::{CubicSplineKernel, grad};

#[test]
fn test_delta_sph_diffusion_lighter_neighbour_lowers_density() {
    // Mirrored case: ρ_i = 1200 > ρ_j = 800 -> density diffuses OUT of i (< 0).
    let diff = delta_sph_diffusion_contribution(
        0.1,
        0.1,
        1480.0,
        1200.0,
        800.0,
        0.001,
        [0.05, 0.0, 0.0],
        [-10.0, 0.0, 0.0],
    );
    // ψ·∇W = -2·(-400)·(-10)/0.05 = -1.6e5, m_j/ρ_j = 0.001/800
    let expected = 0.1 * 0.1 * 1480.0 * (0.001 / 800.0) * (-1.6e5);
    assert!(diff < 0.0, "lighter neighbour must lower ρ_i: {diff}");
    assert!(
        (diff - expected).abs() < 1e-9 * expected.abs(),
        "diff={diff}, expected={expected}"
    );
}

#[test]
fn test_delta_sph_diffusion_is_independent_of_neighbour_direction() {
    // The scalar contribution only depends on |r_ij|: rotate the pair by 90°.
    let along_x = delta_sph_diffusion_contribution(
        0.1,
        0.1,
        1480.0,
        900.0,
        1100.0,
        0.002,
        [0.04, 0.0, 0.0],
        [-8.0, 0.0, 0.0],
    );
    let along_z = delta_sph_diffusion_contribution(
        0.1,
        0.1,
        1480.0,
        900.0,
        1100.0,
        0.002,
        [0.0, 0.0, -0.04],
        [0.0, 0.0, 8.0],
    );
    assert!(along_x > 0.0);
    assert!((along_x - along_z).abs() < 1e-12 * along_x);
}

#[test]
fn test_delta_sph_density_diffusion_smooths_a_density_step() {
    // 1-D line of 8 particles at spacing h/2: four at ρ = 800, four at ρ = 1200.
    let h = 0.1;
    let spacing = 0.5 * h;
    let kernel = CubicSplineKernel;
    let n = 8;
    let positions: Vec<[f64; 3]> = (0..n).map(|i| [i as f64 * spacing, 0.0, 0.0]).collect();
    let densities: Vec<f64> = (0..n).map(|i| if i < 4 { 800.0 } else { 1200.0 }).collect();
    let masses = vec![1000.0 * spacing.powi(3); n];
    let diffusion: Vec<f64> = (0..n)
        .map(|i| {
            let neighbor_indices: Vec<usize> = (0..n).filter(|&j| j != i).collect();
            let grad_ws: Vec<[f64; 3]> = neighbor_indices
                .iter()
                .map(|&j| {
                    let r_ij = [
                        positions[i][0] - positions[j][0],
                        positions[i][1] - positions[j][1],
                        positions[i][2] - positions[j][2],
                    ];
                    grad(&kernel, r_ij, h)
                })
                .collect();
            delta_sph_density_diffusion(
                i,
                &positions,
                &densities,
                &masses,
                &grad_ws,
                &neighbor_indices,
                0.1,
                h,
                1480.0,
            )
        })
        .collect();
    // Interface particles: the light one rises, the heavy one falls.
    assert!(diffusion[3] > 0.0, "ρ_3 must rise: {}", diffusion[3]);
    assert!(diffusion[4] < 0.0, "ρ_4 must fall: {}", diffusion[4]);
    // Mirror symmetry of the set-up (the signs of the two sides swap).
    for i in 0..4 {
        assert!(
            diffusion[i] >= 0.0,
            "light side must not fall: {diffusion:?}"
        );
        assert!(
            diffusion[n - 1 - i] <= 0.0,
            "heavy side must not rise: {diffusion:?}"
        );
    }
    // The pairwise terms are antisymmetric in the volume-weighted sum
    // Σ (m_i/ρ_i) Dρ_i, so diffusion conserves mass to rounding error.
    let scale: f64 = diffusion
        .iter()
        .zip(densities.iter().zip(masses.iter()))
        .map(|(d, (rho, m))| (m / rho * d).abs())
        .sum();
    let net: f64 = diffusion
        .iter()
        .zip(densities.iter().zip(masses.iter()))
        .map(|(d, (rho, m))| m / rho * d)
        .sum();
    assert!(scale > 0.0);
    assert!(net.abs() < 1e-12 * scale, "net={net}, scale={scale}");
}
