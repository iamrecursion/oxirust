// Copyright 2026 COOLJAPAN OU (Team KitaSan)
// SPDX-License-Identifier: Apache-2.0

//! Regression tests for the extended homogenization routines: the periodic cell
//! problem of the 2-D asymptotic homogenization, Hashin's transverse constants of
//! the fibre composite, the coupled differential scheme and the three-mode Puck
//! inter-fibre failure criterion.

use super::extended::*;
use crate::error::Error;

fn rel_diff(a: f64, b: f64) -> f64 {
    (a - b).abs() / b.abs().max(f64::MIN_POSITIVE)
}

// ---------------------------------------------------------------------------
// Asymptotic homogenization of the 2-D conductivity (periodic cell problem)
// ---------------------------------------------------------------------------

/// `rows × cols` map filled by `phase(row, col)`.
fn map_from_fn(rows: usize, cols: usize, phase: impl Fn(usize, usize) -> usize) -> Vec<Vec<usize>> {
    (0..rows)
        .map(|row| (0..cols).map(|col| phase(row, col)).collect())
        .collect()
}

fn transposed(map: &[Vec<usize>]) -> Vec<Vec<usize>> {
    let rows = map.len();
    let cols = map[0].len();
    map_from_fn(cols, rows, |row, col| map[col][row])
}

/// `blocks × blocks` checkerboard of squares of `side × side` pixels.
fn block_checkerboard(blocks: usize, side: usize) -> Vec<Vec<usize>> {
    map_from_fn(blocks * side, blocks * side, |row, col| {
        (row / side + col / side) % 2
    })
}

#[test]
fn layers_along_x_give_voigt_k_xx_and_reuss_k_yy_exactly() {
    // Phase depends on the row (y) only: stripes parallel to x.
    let (k0, k1) = (1.0, 10.0);
    let map = map_from_fn(8, 8, |row, _| usize::from(row >= 3));
    let (f0, f1) = (3.0 / 8.0, 5.0 / 8.0);
    let voigt = f0 * k0 + f1 * k1;
    let reuss = 1.0 / (f0 / k0 + f1 / k1);
    let k = asymptotic_homogenization_2d_conductivity(&map, k0, k1).expect("valid cell");
    assert!(
        rel_diff(k[0], voigt) < 1e-10,
        "K_xx = {} vs Voigt {voigt}",
        k[0]
    );
    assert!(
        rel_diff(k[3], reuss) < 1e-10,
        "K_yy = {} vs Reuss {reuss}",
        k[3]
    );
    assert!(
        k[1].abs() < 1e-10 * voigt && k[2].abs() < 1e-10 * voigt,
        "{k:?}"
    );
}

#[test]
fn layers_along_y_swap_the_two_directions() {
    let (k0, k1) = (2.0, 7.0);
    let map = map_from_fn(6, 12, |_, col| usize::from(col % 4 >= 1));
    let f1 = 0.75;
    let voigt = (1.0 - f1) * k0 + f1 * k1;
    let reuss = 1.0 / ((1.0 - f1) / k0 + f1 / k1);
    let k = asymptotic_homogenization_2d_conductivity(&map, k0, k1).expect("valid cell");
    assert!(
        rel_diff(k[0], reuss) < 1e-10,
        "K_xx = {} vs Reuss {reuss}",
        k[0]
    );
    assert!(
        rel_diff(k[3], voigt) < 1e-10,
        "K_yy = {} vs Voigt {voigt}",
        k[3]
    );
}

#[test]
fn uniform_map_returns_the_phase_conductivity() {
    for (phase, expected) in [(0_usize, 1.5), (1, 6.0)] {
        let map = vec![vec![phase; 5]; 3];
        let k = asymptotic_homogenization_2d_conductivity(&map, 1.5, 6.0).expect("valid cell");
        assert!(rel_diff(k[0], expected) < 1e-12, "{k:?}");
        assert!(rel_diff(k[3], expected) < 1e-12, "{k:?}");
        assert!(k[1].abs() < 1e-12 && k[2].abs() < 1e-12, "{k:?}");
    }
}

#[test]
fn single_pixel_checkerboard_gives_the_harmonic_mean_not_the_geometric_mean() {
    // Every face of a one-pixel checkerboard separates the two phases, so every face
    // conductivity is 2 k0 k1/(k0 + k1) and the cell is a uniform conductor.
    let (k0, k1) = (1.0, 4.0);
    let map = map_from_fn(4, 4, |row, col| (row + col) % 2);
    let k = asymptotic_homogenization_2d_conductivity(&map, k0, k1).expect("valid cell");
    let expected = 2.0 * k0 * k1 / (k0 + k1);
    assert!(
        rel_diff(k[0], expected) < 1e-12 && rel_diff(k[3], expected) < 1e-12,
        "{k:?}"
    );
    assert!(k[1].abs() < 1e-12 && k[2].abs() < 1e-12);
}

#[test]
fn block_checkerboard_converges_to_the_geometric_mean() {
    // Dykhne / Keller: the effective conductivity of the 50/50 two-phase checkerboard
    // is sqrt(k0 k1).  The pixel scheme converges to it as the squares get larger.
    let (k0, k1) = (1.0_f64, 4.0_f64);
    let exact = (k0 * k1).sqrt();
    let mut previous_error = f64::INFINITY;
    for side in [4_usize, 8, 16, 32] {
        let map = block_checkerboard(2, side);
        let k = asymptotic_homogenization_2d_conductivity(&map, k0, k1).expect("valid cell");
        let error = rel_diff(k[0], exact);
        assert!(
            error < previous_error,
            "side {side}: error {error:e} did not decrease from {previous_error:e}"
        );
        // The checkerboard is symmetric under transposition and under reflection.
        assert!(rel_diff(k[0], k[3]) < 1e-10, "side {side}: {k:?}");
        assert!(k[1].abs() < 1e-10 * exact, "side {side}: {k:?}");
        previous_error = error;
    }
    assert!(
        previous_error < 0.02,
        "error at 64x64 pixels: {previous_error:e}"
    );
}

#[test]
fn square_inclusion_respects_wiener_and_hashin_shtrikman_bounds() {
    // Stiff square inclusion (25 % of the cell) in a conducting matrix.
    let (k0, k1) = (1.0, 10.0);
    let n = 32;
    let map = map_from_fn(n, n, |row, col| {
        usize::from((8..24).contains(&row) && (8..24).contains(&col))
    });
    let f1 = 0.25;
    let f0 = 1.0 - f1;
    let k = asymptotic_homogenization_2d_conductivity(&map, k0, k1).expect("valid cell");

    let voigt = f0 * k0 + f1 * k1;
    let reuss = 1.0 / (f0 / k0 + f1 / k1);
    // 2-D Hashin-Shtrikman bounds for isotropic phases.
    let hs_lower = k0 + f1 / (1.0 / (k1 - k0) + f0 / (2.0 * k0));
    let hs_upper = k1 + f0 / (1.0 / (k0 - k1) + f1 / (2.0 * k1));
    for &diag in &[k[0], k[3]] {
        assert!(
            diag <= voigt * (1.0 + 1e-9) && diag >= reuss * (1.0 - 1e-9),
            "{diag}"
        );
        assert!(
            diag >= hs_lower * 0.98 && diag <= hs_upper * 1.02,
            "K = {diag} outside HS [{hs_lower}, {hs_upper}]"
        );
    }
    // The microstructure matters: neither Voigt nor Reuss is returned.
    assert!(k[0] < 0.95 * voigt && k[0] > 1.05 * reuss, "{k:?}");
    // Square symmetry.
    assert!(
        rel_diff(k[0], k[3]) < 1e-10 && k[1].abs() < 1e-10 * k[0],
        "{k:?}"
    );
}

#[test]
fn anisotropic_inclusion_is_anisotropic_symmetric_and_transposes() {
    // A rectangular inclusion elongated along x conducts better along x.
    let (k0, k1) = (1.0, 20.0);
    let map = map_from_fn(16, 16, |row, col| {
        usize::from((6..10).contains(&row) && (1..15).contains(&col))
    });
    let k = asymptotic_homogenization_2d_conductivity(&map, k0, k1).expect("valid cell");
    assert!(
        k[0] > 1.2 * k[3],
        "elongated inclusion: K_xx {} vs K_yy {}",
        k[0],
        k[3]
    );
    assert_eq!(k[1], k[2], "the tensor is symmetric");
    let kt =
        asymptotic_homogenization_2d_conductivity(&transposed(&map), k0, k1).expect("valid cell");
    assert!(
        rel_diff(kt[0], k[3]) < 1e-10 && rel_diff(kt[3], k[0]) < 1e-10,
        "{k:?} {kt:?}"
    );
    // The rectangle is symmetric under both reflections, so the tensor is diagonal.
    assert!(k[1].abs() < 1e-9 * k[0], "{k:?}");
}

#[test]
fn phase_exchange_with_swapped_conductivities_gives_the_same_tensor() {
    let (k0, k1) = (1.0, 5.0);
    let map = map_from_fn(10, 10, |row, col| usize::from(row * col % 7 < 3));
    let swapped = map_from_fn(10, 10, |row, col| 1 - map[row][col]);
    let a = asymptotic_homogenization_2d_conductivity(&map, k0, k1).expect("valid cell");
    let b = asymptotic_homogenization_2d_conductivity(&swapped, k1, k0).expect("valid cell");
    for (x, y) in a.iter().zip(b.iter()) {
        assert!((x - y).abs() < 1e-10 * a[0], "{a:?} vs {b:?}");
    }
}

#[test]
fn degenerate_periodic_grids_are_supported() {
    let (k0, k1) = (1.0, 10.0);
    // One row: the phases vary along x only, so conduction along x is in series.
    let row = vec![vec![0, 1, 0, 1]];
    let k = asymptotic_homogenization_2d_conductivity(&row, k0, k1).expect("valid cell");
    let series = 1.0 / (0.5 / k0 + 0.5 / k1);
    assert!(rel_diff(k[0], series) < 1e-10, "{k:?}");
    assert!(rel_diff(k[3], 0.5 * (k0 + k1)) < 1e-10, "{k:?}");
    // One column.
    let column = transposed(&row);
    let kc = asymptotic_homogenization_2d_conductivity(&column, k0, k1).expect("valid cell");
    assert!(rel_diff(kc[3], series) < 1e-10 && rel_diff(kc[0], 0.5 * (k0 + k1)) < 1e-10);
    // A single pixel.
    let single = asymptotic_homogenization_2d_conductivity(&[vec![1]], k0, k1).expect("valid");
    assert_eq!(single, [k1, 0.0, 0.0, k1]);
}

#[test]
fn invalid_cells_are_rejected() {
    let ok = vec![vec![0, 1], vec![1, 0]];
    let cases: Vec<(Vec<Vec<usize>>, f64, f64)> = vec![
        (vec![], 1.0, 2.0),
        (vec![vec![]], 1.0, 2.0),
        (vec![vec![0, 1], vec![0]], 1.0, 2.0),
        (vec![vec![0, 2], vec![1, 0]], 1.0, 2.0),
        (ok.clone(), 0.0, 2.0),
        (ok.clone(), 1.0, -2.0),
        (ok.clone(), f64::NAN, 2.0),
        (ok, 1.0, f64::INFINITY),
    ];
    for (map, k0, k1) in cases {
        let result = asymptotic_homogenization_2d_conductivity(&map, k0, k1);
        assert!(
            matches!(result, Err(Error::InvalidInput(_))),
            "{map:?} {k0} {k1}: {result:?}"
        );
    }
}

// ---------------------------------------------------------------------------
// Fibre composite: Hashin k23 with transversely isotropic relations
// ---------------------------------------------------------------------------

fn isotropic_shear(e: f64, nu: f64) -> f64 {
    e / (2.0 * (1.0 + nu))
}

#[test]
fn identical_fibre_and_matrix_return_the_phase_exactly() {
    // The old Christensen-style estimate gave nu23 = -0.688, clamped to -0.5.
    let (e, nu) = (70.0e9, 0.3);
    let g = isotropic_shear(e, nu);
    for vf in [0.0, 0.25, 0.6, 1.0] {
        let c = fiber_composite_rom(e, e, nu, nu, g, g, vf);
        assert!(rel_diff(c.e1, e) < 1e-12, "vf {vf}: E1 = {}", c.e1);
        assert!(rel_diff(c.e2, e) < 1e-12, "vf {vf}: E2 = {}", c.e2);
        assert!((c.nu12 - nu).abs() < 1e-12, "vf {vf}: nu12 = {}", c.nu12);
        assert!((c.nu23 - nu).abs() < 1e-12, "vf {vf}: nu23 = {}", c.nu23);
        assert!(
            rel_diff(c.g12, g) < 1e-12 && rel_diff(c.g23, g) < 1e-12,
            "vf {vf}"
        );
    }
}

#[test]
fn limiting_fractions_return_the_constituents() {
    let (e_f, nu_f) = (230.0e9, 0.2);
    let (e_m, nu_m) = (3.5e9, 0.35);
    let (g_f, g_m) = (isotropic_shear(e_f, nu_f), isotropic_shear(e_m, nu_m));
    let matrix = fiber_composite_rom(e_f, e_m, nu_f, nu_m, g_f, g_m, 0.0);
    assert!(rel_diff(matrix.e1, e_m) < 1e-12 && rel_diff(matrix.e2, e_m) < 1e-12);
    assert!((matrix.nu23 - nu_m).abs() < 1e-12, "nu23 = {}", matrix.nu23);
    let fibre = fiber_composite_rom(e_f, e_m, nu_f, nu_m, g_f, g_m, 1.0);
    assert!(rel_diff(fibre.e1, e_f) < 1e-12 && rel_diff(fibre.e2, e_f) < 1e-12);
    assert!((fibre.nu23 - nu_f).abs() < 1e-12, "nu23 = {}", fibre.nu23);
}

#[test]
fn transversely_isotropic_relations_hold_and_k23_lies_between_wiener_bounds() {
    let (e_f, nu_f) = (230.0e9, 0.2);
    let (e_m, nu_m) = (3.5e9, 0.35);
    let (g_f, g_m) = (isotropic_shear(e_f, nu_f), isotropic_shear(e_m, nu_m));
    let vf = 0.6;
    let c = fiber_composite_rom(e_f, e_m, nu_f, nu_m, g_f, g_m, vf);

    // G23 = E2 / (2 (1 + nu23)) for a transversely isotropic material.
    assert!(
        rel_diff(c.g23, c.e2 / (2.0 * (1.0 + c.nu23))) < 1e-12,
        "G23 = {}, E2/(2(1+nu23)) = {}",
        c.g23,
        c.e2 / (2.0 * (1.0 + c.nu23))
    );
    assert!(c.nu23 > 0.0 && c.nu23 < 0.5, "nu23 = {}", c.nu23);

    // Plane-strain bulk modulus recovered from the returned constants:
    // k23 = E2 / (2 (1 - nu23 - 2 nu12² E2/E1)); compare with Hashin's formula.
    let k_from_constants = c.e2 / (2.0 * (1.0 - c.nu23 - 2.0 * c.nu12 * c.nu12 * c.e2 / c.e1));
    let k_f = e_f / (3.0 * (1.0 - 2.0 * nu_f)) + g_f / 3.0;
    let k_m = e_m / (3.0 * (1.0 - 2.0 * nu_m)) + g_m / 3.0;
    let hashin = k_m + vf / (1.0 / (k_f - k_m) + (1.0 - vf) / (k_m + g_m));
    assert!(
        rel_diff(k_from_constants, hashin) < 1e-10,
        "k23 from constants {k_from_constants:e} vs Hashin {hashin:e}"
    );
    let voigt = vf * k_f + (1.0 - vf) * k_m;
    let reuss = 1.0 / (vf / k_f + (1.0 - vf) / k_m);
    assert!(
        reuss < hashin && hashin < voigt,
        "{reuss:e} < {hashin:e} < {voigt:e}"
    );
}

#[test]
fn nu23_is_not_clamped_and_the_transversely_isotropic_material_stays_stable() {
    // Glass / epoxy with a nearly incompressible matrix: the stiff fibres restrain the
    // axial strain, so the transverse Poisson's ratio of the composite exceeds 1/2
    // (plane-strain ratio nu/(1 - nu) = 0.61 of the matrix).  The old formula clamped
    // nu23 to [-0.5, 0.5].
    let (e_f, nu_f) = (72.0e9, 0.22);
    let (e_m, nu_m) = (3.0e9, 0.38);
    let (g_f, g_m) = (isotropic_shear(e_f, nu_f), isotropic_shear(e_m, nu_m));
    let at_fifth = fiber_composite_rom(e_f, e_m, nu_f, nu_m, g_f, g_m, 0.2);
    assert!(at_fifth.nu23 > 0.5, "nu23 = {}", at_fifth.nu23);
    for vf in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7] {
        let c = fiber_composite_rom(e_f, e_m, nu_f, nu_m, g_f, g_m, vf);
        assert!(
            c.nu23 > nu_m - 0.05 && c.nu23 < 1.0,
            "vf {vf}: nu23 = {}",
            c.nu23
        );
        assert!(c.e2 > 0.0 && c.g23 > 0.0 && c.g12 > 0.0);
        // Positive-definite compliance: k23 = E2 / (2 (1 - nu23 - 2 nu12² E2/E1)) > 0.
        let stability = 1.0 - c.nu23 - 2.0 * c.nu12 * c.nu12 * c.e2 / c.e1;
        assert!(stability > 0.0, "vf {vf}: stability margin {stability}");
    }
}

// ---------------------------------------------------------------------------
// Differential scheme
// ---------------------------------------------------------------------------

#[test]
fn differential_scheme_identical_phases_leave_the_moduli_unchanged() {
    let (k, g) = (60.0e9, 30.0e9);
    let moduli = differential_scheme(k, g, k, g, 0.4, 7).expect("valid input");
    assert_eq!(moduli, (k, g));
}

#[test]
fn differential_scheme_limit_fractions_return_the_constituents() {
    let (k_m, g_m, k_i, g_i) = (50.0e9, 30.0e9, 300.0e9, 150.0e9);
    let none = differential_scheme(k_m, g_m, k_i, g_i, 0.0, 0).expect("f = 0");
    assert_eq!(none, (k_m, g_m));
    let all = differential_scheme(k_m, g_m, k_i, g_i, 1.0, 0).expect("f = 1");
    assert_eq!(all, (k_i, g_i));
}

#[test]
fn voids_in_a_matrix_with_poisson_ratio_one_fifth_scale_with_one_minus_c_squared() {
    // For K_i = G_i = 0 the ratio a = K/G obeys da/dt = a (3a + 4)(12 - 9a)/(4 (9a + 8))
    // (t = -ln(1 - c)), which vanishes at a = 4/3 (nu = 0.2), where dG/dt = -2 G:
    // K = K_m (1 - c)² and G = G_m (1 - c)² exactly.
    let g_m = 30.0e9;
    let k_m = 4.0 / 3.0 * g_m;
    for f in [0.1, 0.3, 0.5] {
        let (k, g) = differential_scheme(k_m, g_m, 0.0, 0.0, f, 200).expect("valid input");
        let scale = (1.0 - f).powi(2);
        assert!(
            rel_diff(k, k_m * scale) < 1e-10,
            "f = {f}: K = {k:e}, expected {:e}",
            k_m * scale
        );
        assert!(
            rel_diff(g, g_m * scale) < 1e-10,
            "f = {f}: G = {g:e}, expected {:e}",
            g_m * scale
        );
    }
}

#[test]
fn rigid_spheres_in_a_nearly_incompressible_matrix_follow_roscoe_brinkman() {
    // Incompressible matrix, rigid spheres: G*/G_m = (1 - c)^(-5/2) (Roscoe 1952,
    // Brinkman 1952), the viscosity analogue of the differential scheme.
    let g_m = 1.0e9;
    let k_m = 1.0e6 * g_m;
    let rigid = 1.0e9 * g_m;
    for f in [0.1, 0.3, 0.5] {
        let (_, g) = differential_scheme(k_m, g_m, rigid, rigid, f, 400).expect("valid input");
        let expected = g_m * (1.0 - f).powf(-2.5);
        assert!(
            rel_diff(g, expected) < 1e-5,
            "f = {f}: G = {g:e}, expected {expected:e}"
        );
    }
}

#[test]
fn differential_scheme_small_fraction_slopes_match_the_dilute_eshelby_increments() {
    let (k_m, g_m, k_i, g_i) = (50.0e9, 30.0e9, 300.0e9, 150.0e9);
    let f = 1.0e-6;
    let (k, g) = differential_scheme(k_m, g_m, k_i, g_i, f, 4).expect("valid input");
    // Dilute Eshelby: K slope (K_i - K_m)(K_m + 4G_m/3)/(K_i + 4G_m/3); G slope
    // (G_i - G_m)(G_m + ζ)/(G_i + ζ) with ζ = G_m (9K_m + 8G_m)/(6 (K_m + 2G_m)).
    let k_slope = (k_i - k_m) * (k_m + 4.0 * g_m / 3.0) / (k_i + 4.0 * g_m / 3.0);
    let zeta = g_m * (9.0 * k_m + 8.0 * g_m) / (6.0 * (k_m + 2.0 * g_m));
    let g_slope = (g_i - g_m) * (g_m + zeta) / (g_i + zeta);
    assert!(
        rel_diff((k - k_m) / f, k_slope) < 1e-5,
        "{} vs {k_slope}",
        (k - k_m) / f
    );
    assert!(
        rel_diff((g - g_m) / f, g_slope) < 1e-5,
        "{} vs {g_slope}",
        (g - g_m) / f
    );
}

#[test]
fn differential_scheme_lies_between_hashin_shtrikman_and_wiener_bounds() {
    let (k_m, g_m, k_i, g_i) = (50.0e9, 30.0e9, 300.0e9, 150.0e9);
    for f_i in [0.1, 0.3, 0.6] {
        let f_m = 1.0 - f_i;
        let (k, g) = differential_scheme(k_m, g_m, k_i, g_i, f_i, 200).expect("valid input");
        let k_lo = hs_bulk_lower_bound(k_m, g_m, k_i, f_m, f_i);
        let k_hi = hs_bulk_upper_bound(k_i, g_i, k_m, f_m, f_i);
        let g_lo = hs_shear_lower_bound(k_m, g_m, g_i, f_m, f_i);
        let g_hi = hs_shear_upper_bound(k_i, g_i, g_m, f_m, f_i);
        assert!(
            k_lo * (1.0 - 1e-9) <= k && k <= k_hi * (1.0 + 1e-9),
            "f {f_i}: K {k:e} not in [{k_lo:e}, {k_hi:e}]"
        );
        assert!(
            g_lo * (1.0 - 1e-9) <= g && g <= g_hi * (1.0 + 1e-9),
            "f {f_i}: G {g:e} not in [{g_lo:e}, {g_hi:e}]"
        );
        let k_voigt = f_m * k_m + f_i * k_i;
        let k_reuss = 1.0 / (f_m / k_m + f_i / k_i);
        assert!(k_reuss < k && k < k_voigt, "f {f_i}");
    }
}

#[test]
fn rigid_inclusions_stay_below_the_voigt_bound_and_above_the_dilute_estimate() {
    // The pre-fix integrator followed a linear (Voigt-like) path in c.  For rigid
    // spheres dK/dc = (K + 4G/3)/(1 - c) increases with c (K and G grow), so K(f) lies
    // above the tangent K_m + f (K_m + 4G_m/3) at c = 0 and far below Voigt.
    let (k_m, g_m) = (50.0e9, 30.0e9);
    let rigid = 1.0e3 * k_m;
    let f = 0.3;
    let k = differential_scheme_bulk(k_m, g_m, rigid, rigid, f, 400).expect("valid input");
    let voigt = (1.0 - f) * k_m + f * rigid;
    assert!(k < 0.1 * voigt, "K = {k:e}, Voigt = {voigt:e}");
    let tangent = k_m + f * (k_m + 4.0 * g_m / 3.0);
    assert!(
        k > tangent * 1.05,
        "K = {k:e}, dilute tangent = {tangent:e}"
    );
}

#[test]
fn differential_scheme_rk4_has_fourth_order_accuracy() {
    let (k_m, g_m, k_i, g_i, f) = (50.0e9, 30.0e9, 300.0e9, 150.0e9, 0.5);
    let (k_ref, g_ref) = differential_scheme(k_m, g_m, k_i, g_i, f, 4000).expect("reference");
    let error = |n: usize| {
        let (k, g) = differential_scheme(k_m, g_m, k_i, g_i, f, n).expect("valid input");
        rel_diff(k, k_ref).max(rel_diff(g, g_ref))
    };
    let (e2, e4, e8) = (error(2), error(4), error(8));
    assert!(e2 > e4 && e4 > e8, "errors {e2:e}, {e4:e}, {e8:e}");
    assert!(
        e2 / e4 > 8.0 && e4 / e8 > 8.0,
        "orders: {:.1}, {:.1}",
        e2 / e4,
        e4 / e8
    );
}

#[test]
fn differential_scheme_bulk_is_the_first_component() {
    let (k_m, g_m, k_i, g_i) = (50.0e9, 30.0e9, 300.0e9, 150.0e9);
    let (k, _) = differential_scheme(k_m, g_m, k_i, g_i, 0.3, 50).expect("valid input");
    let bulk = differential_scheme_bulk(k_m, g_m, k_i, g_i, 0.3, 50).expect("valid input");
    assert_eq!(k, bulk);
}

#[test]
fn differential_scheme_rejects_invalid_input() {
    let cases = [
        differential_scheme(0.0, 30e9, 1.0, 1.0, 0.2, 10),
        differential_scheme(50e9, -1.0, 1.0, 1.0, 0.2, 10),
        differential_scheme(f64::NAN, 30e9, 1.0, 1.0, 0.2, 10),
        differential_scheme(50e9, 30e9, -1.0, 1.0, 0.2, 10),
        differential_scheme(50e9, 30e9, 1.0, f64::NAN, 0.2, 10),
        differential_scheme(50e9, 30e9, 1.0, 1.0, 1.2, 10),
        differential_scheme(50e9, 30e9, 1.0, 1.0, -0.1, 10),
        differential_scheme(50e9, 30e9, 1.0, 1.0, f64::NAN, 10),
        differential_scheme(50e9, 30e9, 1.0, 1.0, 0.2, 0),
    ];
    for (idx, case) in cases.iter().enumerate() {
        assert!(
            matches!(case, Err(Error::InvalidInput(_))),
            "case {idx}: {case:?}"
        );
    }
}

// ---------------------------------------------------------------------------
// Puck inter-fibre failure (modes A, B, C)
// ---------------------------------------------------------------------------

/// Carbon/epoxy-like ply strengths and inclination parameters.
const R_T: f64 = 62.0e6;
const R_C: f64 = 200.0e6;
const S21: f64 = 92.0e6;
const P_T: f64 = 0.30;
const P_C: f64 = 0.25;

fn puck(sigma_2: f64, tau_21: f64) -> f64 {
    puck_inter_fiber_failure(sigma_2, tau_21, R_T, R_C, S21, P_T, P_C).expect("valid Puck input")
}

#[test]
fn puck_mode_a_fracture_curve_passes_through_the_transverse_tensile_strength() {
    // The pre-fix expression gave 1 + p_t R_T / S21 = 1.20 here.
    assert!((puck(R_T, 0.0) - 1.0).abs() < 1e-12, "{}", puck(R_T, 0.0));
}

#[test]
fn puck_pure_shear_at_the_shear_strength_gives_one_from_both_sides() {
    assert!((puck(0.0, S21) - 1.0).abs() < 1e-12);
    assert!((puck(0.0, -S21) - 1.0).abs() < 1e-12);
    // Mode B approaches the same value for vanishing compression.
    assert!(
        (puck(-1.0e-3, S21) - 1.0).abs() < 1e-9,
        "{}",
        puck(-1.0e-3, S21)
    );
}

#[test]
fn puck_pure_transverse_compression_follows_mode_c() {
    // Mode B alone would report 0 for any pure transverse compression.
    assert!((puck(-R_C, 0.0) - 1.0).abs() < 1e-12, "{}", puck(-R_C, 0.0));
    assert!((puck(-0.5 * R_C, 0.0) - 0.5).abs() < 1e-12);
    assert!(puck(-30.0e6, 0.0) > 0.0);
}

#[test]
fn puck_modes_join_continuously_and_the_transition_point_lies_on_the_fracture_curve() {
    let s = (1.0 + 2.0 * P_C * R_C / S21).sqrt();
    let transverse_shear_strength = R_C / (1.0 + s);
    let transition_shear = S21 * s.sqrt();
    // The corner of the fracture curve between modes B and C.
    let corner = puck(-transverse_shear_strength, transition_shear);
    assert!(
        (corner - 1.0).abs() < 1e-12,
        "f_E at the B/C corner: {corner}"
    );
    // Straddle the B/C boundary sigma_2 = -(R_A / tau_c) |tau| at several shear levels.
    for tau in [20.0e6, 55.0e6, 90.0e6] {
        let boundary = -transverse_shear_strength / transition_shear * tau;
        let below = puck(boundary * (1.0 - 1e-9), tau);
        let above = puck(boundary * (1.0 + 1e-9), tau);
        assert!(
            (below - above).abs() < 1e-7,
            "tau {tau}: {below} vs {above}"
        );
    }
}

#[test]
fn puck_effort_is_homogeneous_of_degree_one_in_every_mode() {
    let states = [
        (40.0e6, 30.0e6),   // mode A
        (-20.0e6, 60.0e6),  // mode B
        (-120.0e6, 20.0e6), // mode C
        (-60.0e6, 40.0e6),  // near the B/C boundary
        (-150.0e6, 0.0),    // pure compression
    ];
    for (sigma_2, tau_21) in states {
        let base = puck(sigma_2, tau_21);
        assert!(base.is_finite() && base > 0.0);
        for alpha in [0.37, 1.0, 2.9] {
            let scaled = puck(alpha * sigma_2, alpha * tau_21);
            assert!(
                (scaled - alpha * base).abs() < 1e-12 * alpha * base,
                "({sigma_2}, {tau_21}) x {alpha}: {scaled} vs {}",
                alpha * base
            );
        }
    }
}

#[test]
fn puck_effort_is_non_negative_and_grows_with_the_load() {
    for sigma_2 in [-250.0e6, -100.0e6, -20.0e6, 0.0, 20.0e6, 60.0e6] {
        for tau_21 in [0.0, 25.0e6, 80.0e6] {
            let f = puck(sigma_2, tau_21);
            assert!(f.is_finite() && f >= 0.0, "({sigma_2}, {tau_21}) -> {f}");
            assert!(puck(1.5 * sigma_2, 1.5 * tau_21) >= f);
        }
    }
    // Transverse tension adds to the shear effort; compression relieves it (mode B).
    assert!(puck(20.0e6, 50.0e6) > puck(0.0, 50.0e6));
    assert!(puck(-20.0e6, 50.0e6) < puck(0.0, 50.0e6));
}

#[test]
fn puck_rejects_invalid_input() {
    let ok = (10.0e6, 5.0e6, R_T, R_C, S21, P_T, P_C);
    let cases = [
        puck_inter_fiber_failure(f64::NAN, ok.1, ok.2, ok.3, ok.4, ok.5, ok.6),
        puck_inter_fiber_failure(ok.0, f64::INFINITY, ok.2, ok.3, ok.4, ok.5, ok.6),
        puck_inter_fiber_failure(ok.0, ok.1, 0.0, ok.3, ok.4, ok.5, ok.6),
        puck_inter_fiber_failure(ok.0, ok.1, ok.2, -1.0, ok.4, ok.5, ok.6),
        puck_inter_fiber_failure(ok.0, ok.1, ok.2, ok.3, 0.0, ok.5, ok.6),
        puck_inter_fiber_failure(ok.0, ok.1, ok.2, ok.3, f64::NAN, ok.5, ok.6),
        puck_inter_fiber_failure(ok.0, ok.1, ok.2, ok.3, ok.4, -0.1, ok.6),
        puck_inter_fiber_failure(ok.0, ok.1, ok.2, ok.3, ok.4, ok.5, -0.1),
        // p_t R_T = 99.2e6 > S21 = 92e6: the mode-A curve would miss R_T.
        puck_inter_fiber_failure(ok.0, ok.1, ok.2, ok.3, ok.4, 1.6, ok.6),
    ];
    for (idx, case) in cases.iter().enumerate() {
        assert!(
            matches!(case, Err(Error::InvalidInput(_))),
            "case {idx}: {case:?}"
        );
    }
}
