// Copyright 2026 COOLJAPAN OU (Team KitaSan)
// SPDX-License-Identifier: Apache-2.0

//! Regression tests for the two-layer planetary wave model.

use super::super::shallow_water::ShallowWaterLbm;
use super::super::{CX, CY, macroscopic};
use super::*;

/// Surface gravity of the lattice-unit test model: `√(g (h1 + h2)) = 0.28`.
const GRAVITY: f64 = 0.08;
/// Reduced gravity `g/25`.
const G_PRIME: f64 = 0.0032;
const H1: f64 = 0.5;
const H2: f64 = 0.5;

fn lattice_model(nx: usize, ny: usize, omega: f64) -> PlanetaryWave {
    PlanetaryWave::new(nx, ny, omega, omega, 0.0, G_PRIME, H1, H2).with_gravity(GRAVITY)
}

/// Amplitude of the `cos(2π cycles ix / nx)` component of a field.
fn cosine_amplitude(field: &[Vec<f64>], cycles: f64) -> f64 {
    let nx = field.len();
    let ny = field[0].len();
    let mut acc = 0.0;
    for (ix, column) in field.iter().enumerate() {
        acc += column.iter().sum::<f64>() * (2.0 * PI * cycles * ix as f64 / nx as f64).cos();
    }
    2.0 * acc / (nx * ny) as f64
}

/// Fractional sample positions where a series changes sign.
fn zero_crossings(series: &[f64]) -> Vec<f64> {
    series
        .windows(2)
        .enumerate()
        .filter(|(_, pair)| pair[0] * pair[1] < 0.0)
        .map(|(i, pair)| i as f64 + pair[0] / (pair[0] - pair[1]))
        .collect()
}

/// Sample position of the smallest value (parabolic refinement).
fn time_of_minimum(series: &[f64]) -> f64 {
    let (index, _) =
        series.iter().enumerate().fold(
            (0, f64::INFINITY),
            |best, (i, &v)| {
                if v < best.1 { (i, v) } else { best }
            },
        );
    if index == 0 || index + 1 >= series.len() {
        return index as f64;
    }
    let (left, mid, right) = (series[index - 1], series[index], series[index + 1]);
    let curvature = left - 2.0 * mid + right;
    if curvature.abs() < 1e-300 {
        return index as f64;
    }
    index as f64 + 0.5 * (left - right) / curvature
}

/// Run `steps` steps and record the `cycles` cosine amplitude of `field(pw)`
/// before the first and after every step.
fn record(
    pw: &mut PlanetaryWave,
    steps: usize,
    cycles: f64,
    field: impl Fn(&PlanetaryWave) -> Vec<Vec<f64>>,
) -> Vec<f64> {
    let mut series = vec![cosine_amplitude(&field(pw), cycles)];
    for _ in 0..steps {
        pw.step();
        series.push(cosine_amplitude(&field(pw), cycles));
    }
    series
}

// ─────────────────────────────────────────────────────────────────────────────
// Construction and parameters
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn test_planetary_wave_construction() {
    let pw = PlanetaryWave::new(16, 16, 1.0, 1.0, 1.6e-11, 0.02, 300.0, 4700.0);
    assert_eq!(pw.nx, 16);
    assert_eq!(pw.f0, 0.0, "the default model is equatorial");
    assert!((pw.gravity - PlanetaryWave::DEFAULT_GRAVITY_RATIO * 0.02).abs() < 1e-15);
}

#[test]
fn rest_state_populations_carry_the_layer_depths() {
    let pw = lattice_model(6, 5, 1.2);
    for ix in 0..pw.nx {
        for iy in 0..pw.ny {
            let (h1, ux1, uy1) = macroscopic(&pw.f1[ix][iy]);
            let (h2, ux2, uy2) = macroscopic(&pw.f2[ix][iy]);
            assert!((h1 - H1).abs() < 1e-15 && (h2 - H2).abs() < 1e-15);
            assert!(ux1.abs() + uy1.abs() + ux2.abs() + uy2.abs() < 1e-15);
        }
    }
    // The Zhou equilibrium depends on g: a different gravity re-initialises.
    let stronger = lattice_model(6, 5, 1.2).with_gravity(2.0 * GRAVITY);
    assert!(
        (stronger.f1[0][0][0] - pw.f1[0][0][0]).abs() > 1e-3,
        "the rest populations must follow the surface gravity"
    );
    let (h1, _, _) = macroscopic(&stronger.f1[0][0]);
    assert!((h1 - H1).abs() < 1e-15);
}

#[test]
fn check_parameters_rejects_inconsistent_models() {
    assert!(lattice_model(8, 8, 1.2).check_parameters().is_ok());
    let mut pw = lattice_model(8, 8, 1.2);
    pw.g_prime = 2.0 * pw.gravity;
    assert!(pw.check_parameters().is_err(), "g' must not exceed g");
    let mut pw = lattice_model(8, 8, 1.2);
    pw.h2 = -1.0;
    assert!(
        pw.check_parameters().is_err(),
        "thicknesses must be positive"
    );
    let mut pw = lattice_model(8, 8, 1.2);
    pw.omega1 = 2.0;
    assert!(pw.check_parameters().is_err(), "omega must stay below 2");
    let mut pw = lattice_model(8, 8, 1.2);
    pw.f0 = f64::NAN;
    assert!(pw.check_parameters().is_err(), "f0 must be finite");
    assert!(
        PlanetaryWave::new(0, 4, 1.0, 1.0, 0.0, 0.1, 1.0, 1.0)
            .check_parameters()
            .is_err()
    );
}

#[test]
fn coriolis_parameter_is_f0_plus_beta_times_the_distance_from_the_centre_row() {
    let pw = PlanetaryWave::new(4, 10, 1.0, 1.0, 3.0e-4, G_PRIME, H1, H2).with_coriolis(2.0e-3);
    assert_eq!(pw.coriolis_at(5), 2.0e-3);
    assert!((pw.coriolis_at(8) - (2.0e-3 + 3.0 * 3.0e-4)).abs() < 1e-18);
    assert!((pw.coriolis_at(0) - (2.0e-3 - 5.0 * 3.0e-4)).abs() < 1e-18);
    let equatorial = PlanetaryWave::new(4, 10, 1.0, 1.0, 3.0e-4, G_PRIME, H1, H2);
    assert_eq!(equatorial.coriolis_at(5), 0.0, "f0 defaults to the equator");
}

// ─────────────────────────────────────────────────────────────────────────────
// Wave speeds and deformation radius
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn gravity_wave_speeds_solve_the_two_layer_dispersion_relation() {
    let pw = lattice_model(8, 8, 1.2);
    let (fast, slow) = pw.gravity_wave_speeds();
    let total = H1 + H2;
    assert!((fast * fast + slow * slow - GRAVITY * total).abs() < 1e-15);
    assert!((fast * fast * slow * slow - GRAVITY * G_PRIME * H1 * H2).abs() < 1e-18);
    assert!(fast > 5.0 * slow, "the surface mode is much faster");
    // Rigid-lid limits.
    assert!((fast - (GRAVITY * total).sqrt()).abs() < 0.01 * fast);
    assert!((slow - (G_PRIME * H1 * H2 / total).sqrt()).abs() < 0.02 * slow);
    // Massless upper layer (g' = g): two independent shallow-water layers.
    let decoupled =
        PlanetaryWave::new(8, 8, 1.2, 1.2, 0.0, GRAVITY, H1, 2.0 * H2).with_gravity(GRAVITY);
    let (fast, slow) = decoupled.gravity_wave_speeds();
    assert!((fast - (GRAVITY * 2.0 * H2).sqrt()).abs() < 1e-14, "{fast}");
    assert!((slow - (GRAVITY * H1).sqrt()).abs() < 1e-14, "{slow}");
}

#[test]
fn test_barotropic_phase_speed() {
    let pw = PlanetaryWave::new(16, 16, 1.0, 1.0, 2e-11, 0.02, 300.0, 4700.0);
    let cp = pw.barotropic_phase_speed(1e-6);
    assert!(cp < 0.0, "barotropic Rossby waves propagate westward");
    assert!((cp + 2e-11 / 1e-12).abs() < 1e-9);
}

#[test]
fn test_baroclinic_phase_speed() {
    let pw = PlanetaryWave::new(16, 16, 1.0, 1.0, 2e-11, 0.02, 300.0, 4700.0);
    let cp_bt = pw.barotropic_phase_speed(1e-5);
    let cp_bc = pw.baroclinic_phase_speed(1e-5);
    assert!(
        cp_bc > cp_bt,
        "baroclinic speed slower (less negative) than barotropic at short wave"
    );
    assert!(
        cp_bc < 0.0,
        "baroclinic Rossby waves also propagate westward"
    );
}

#[test]
fn baroclinic_phase_speed_uses_the_internal_deformation_radius() {
    // f-plane: L_R^2 = c^2 / f0^2 with the baroclinic gravity-wave speed c.
    let f0 = 1.0e-4;
    let pw = PlanetaryWave::new(8, 8, 1.0, 1.0, 2.0e-11, 0.02, 300.0, 4700.0).with_coriolis(f0);
    let (_, c) = pw.gravity_wave_speeds();
    let rigid_lid = (0.02f64 * 300.0 * 4700.0 / 5000.0).sqrt();
    assert!(
        (c - rigid_lid).abs() < 2e-3 * rigid_lid,
        "c = {c}, rigid-lid value {rigid_lid}"
    );
    let radius = pw.baroclinic_deformation_radius();
    assert!((radius - c / f0).abs() < 1e-9 * radius);
    let k = 3.0e-5;
    let expected = -2.0e-11 / (k * k + f0 * f0 / (c * c));
    assert!((pw.baroclinic_phase_speed(k) - expected).abs() < 1e-12 * expected.abs());
    // Long waves are slowed down to -beta L_R^2.
    let long = pw.baroclinic_phase_speed(1.0e-9);
    assert!(
        (long + 2.0e-11 * radius * radius).abs() < 1e-6 * long.abs(),
        "{long}"
    );

    // Equatorial beta plane (f0 = 0): L^2 = c / beta.
    let equatorial = PlanetaryWave::new(8, 8, 1.0, 1.0, 2.0e-11, 0.02, 300.0, 4700.0);
    let radius = equatorial.baroclinic_deformation_radius();
    assert!((radius * radius - c / 2.0e-11).abs() < 1e-9 * radius * radius);
    let expected = -2.0e-11 / (k * k + 2.0e-11 / c);
    assert!((equatorial.baroclinic_phase_speed(k) - expected).abs() < 1e-12 * expected.abs());
}

#[test]
fn baroclinic_phase_speed_is_dimensionally_consistent() {
    // Re-expressing every length in a unit lambda times smaller multiplies the
    // thicknesses, g' and g by lambda, divides beta and k by lambda and leaves
    // f0 alone, so the phase speed (a length per time) must be multiplied by
    // lambda.  The old L_R^2 = g' h1 / beta had the wrong dimensions and
    // failed this.
    let lambda = 1.0e3;
    let (k, beta, g_prime, h1, h2) = (4.0e-6, 2.0e-11, 0.02, 300.0, 4700.0);
    for f0 in [0.0, 1.0e-4] {
        let metres = PlanetaryWave::new(8, 8, 1.0, 1.0, beta, g_prime, h1, h2).with_coriolis(f0);
        let scaled = PlanetaryWave::new(
            8,
            8,
            1.0,
            1.0,
            beta / lambda,
            g_prime * lambda,
            h1 * lambda,
            h2 * lambda,
        )
        .with_coriolis(f0);
        // The surface gravity is a length per time squared as well.
        let scaled = scaled.with_gravity(metres.gravity * lambda);
        let phase = metres.baroclinic_phase_speed(k);
        let phase_scaled = scaled.baroclinic_phase_speed(k / lambda);
        assert!(
            (phase_scaled - phase * lambda).abs() < 1e-9 * (phase * lambda).abs(),
            "f0 = {f0}: {phase_scaled} vs {}",
            phase * lambda
        );
    }
}

#[test]
fn baroclinic_phase_speed_handles_degenerate_cases() {
    let pw = PlanetaryWave::new(8, 8, 1.0, 1.0, 0.0, 0.02, 300.0, 4700.0);
    assert_eq!(
        pw.baroclinic_phase_speed(1e-5),
        0.0,
        "no beta, no Rossby wave"
    );
    assert_eq!(pw.baroclinic_phase_speed(0.0), 0.0);
    assert!(pw.baroclinic_deformation_radius().is_infinite());
    let no_stratification = PlanetaryWave::new(8, 8, 1.0, 1.0, 1e-11, 0.0, 300.0, 4700.0);
    assert_eq!(no_stratification.baroclinic_phase_speed(1e-5), 0.0);
}

// ─────────────────────────────────────────────────────────────────────────────
// Initial conditions
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn test_planetary_wave_kelvin_ic() {
    let mut pw = PlanetaryWave::new(16, 16, 1.0, 1.0, 1e-5, 0.02, 300.0, 4700.0);
    pw.kelvin_wave_ic(10.0, 1.0);
    let h_max = (0..16)
        .flat_map(|ix| (0..16).map(move |iy| (ix, iy)))
        .map(|(ix, iy)| macroscopic(&pw.f1[ix][iy]).0)
        .fold(f64::NEG_INFINITY, f64::max);
    assert!(h_max > 300.0, "Kelvin wave should raise interface height");
    // Only the upper layer is perturbed.
    let h2_spread = (0..16)
        .map(|ix| macroscopic(&pw.f2[ix][3]).0)
        .fold(0.0f64, |acc, h| acc.max((h - 4700.0).abs()));
    assert!(h2_spread < 1e-9);
}

#[test]
fn mode_initial_conditions_are_eigenvectors_of_the_two_layer_system() {
    let nx = 16;
    for baroclinic in [true, false] {
        let mut pw = lattice_model(nx, 4, 1.2);
        let (fast, slow) = pw.gravity_wave_speeds();
        let (c, c2) = if baroclinic {
            pw.baroclinic_wave_ic(0.01, 1.0);
            (slow, slow * slow)
        } else {
            pw.barotropic_wave_ic(0.01, 1.0);
            (fast, fast * fast)
        };
        assert!(c > 0.0);
        let interface = pw.interface_displacement();
        let surface = pw.surface_elevation();
        for ix in 0..nx {
            let h1p = interface[ix][0];
            let eta = surface[ix][0];
            let h2p = eta - h1p;
            // c^2 h1' = g h1 eta' and c^2 h2' = h2 (g eta' - g' h1').
            assert!(
                (c2 * h1p - GRAVITY * H1 * eta).abs() < 1e-14,
                "layer 1, ix = {ix}"
            );
            assert!(
                (c2 * h2p - H2 * (GRAVITY * eta - G_PRIME * h1p)).abs() < 1e-14,
                "layer 2, ix = {ix}"
            );
        }
        let amplitude = cosine_amplitude(&interface, 1.0);
        assert!((amplitude - 0.01).abs() < 1e-14, "{amplitude}");
    }
}

#[test]
fn baroclinic_mode_keeps_the_free_surface_nearly_flat() {
    let mut pw = lattice_model(16, 4, 1.2);
    pw.baroclinic_wave_ic(0.01, 1.0);
    let surface = cosine_amplitude(&pw.surface_elevation(), 1.0);
    let interface = cosine_amplitude(&pw.interface_displacement(), 1.0);
    assert!(
        surface.abs() < 0.03 * interface.abs(),
        "surface {surface}, interface {interface}"
    );
    let mut pw = lattice_model(16, 4, 1.2);
    pw.barotropic_wave_ic(0.01, 1.0);
    let surface = cosine_amplitude(&pw.surface_elevation(), 1.0);
    let interface = cosine_amplitude(&pw.interface_displacement(), 1.0);
    assert!(
        surface > 1.5 * interface,
        "surface {surface}, interface {interface}"
    );
}

// ─────────────────────────────────────────────────────────────────────────────
// Layer coupling
// ─────────────────────────────────────────────────────────────────────────────

/// Largest `|h u_x|` of a layer's populations.
fn max_momentum_x(layer: &[Vec<[f64; NQ]>]) -> f64 {
    layer
        .iter()
        .flatten()
        .map(|cell| (0..NQ).map(|q| CX[q] * cell[q]).sum::<f64>().abs())
        .fold(0.0, f64::max)
}

/// Response of layer 2 after one step to a thickness anomaly of layer 1 only.
fn layer_two_response(g_prime: f64) -> f64 {
    let mut pw = PlanetaryWave::new(16, 4, 1.4, 1.4, 0.0, g_prime, H1, H2).with_gravity(GRAVITY);
    pw.kelvin_wave_ic(0.05, 1.0);
    pw.step();
    max_momentum_x(&pw.f2)
}

/// Response of layer 1 after one step to a thickness anomaly of layer 2 only.
fn layer_one_response(g_prime: f64) -> f64 {
    let mut pw = PlanetaryWave::new(16, 4, 1.4, 1.4, 0.0, g_prime, H1, H2).with_gravity(GRAVITY);
    let (h1, h2) = (pw.h1, pw.h2);
    pw.fill_rest(h1, h2, |ix, _| {
        (0.0, 0.05 * (2.0 * PI * ix as f64 / 16.0).cos())
    });
    pw.step();
    max_momentum_x(&pw.f1)
}

#[test]
fn upper_layer_anomaly_forces_the_lower_layer_in_proportion_to_g_minus_g_prime() {
    let uncoupled_limit = layer_two_response(GRAVITY);
    assert!(
        uncoupled_limit < 1e-14,
        "g' = g (massless upper layer) must not move the lower layer: {uncoupled_limit}"
    );
    let full = layer_two_response(0.0);
    let half = layer_two_response(0.5 * GRAVITY);
    assert!(full > 1e-5, "the lower layer must respond: {full}");
    assert!(
        (half / full - 0.5).abs() < 1e-4,
        "response ratio {} instead of 1/2",
        half / full
    );
}

#[test]
fn lower_layer_anomaly_forces_the_upper_layer_independently_of_g_prime() {
    let a = layer_one_response(0.0);
    let b = layer_one_response(0.5 * GRAVITY);
    let c = layer_one_response(GRAVITY);
    assert!(a > 1e-5, "the upper layer must respond: {a}");
    assert!((a - b).abs() < 1e-9 * a, "{a} vs {b}");
    assert!((a - c).abs() < 1e-9 * a, "{a} vs {c}");
}

#[test]
fn lower_layer_is_pushed_away_from_a_thick_upper_layer() {
    let mut pw = lattice_model(16, 4, 1.4);
    pw.kelvin_wave_ic(0.05, 1.0); // crest of layer 1 at ix = 0
    for _ in 0..3 {
        pw.step();
    }
    let jx = |ix: usize| -> f64 { (0..NQ).map(|q| CX[q] * pw.f2[ix][1][q]).sum() };
    assert!(
        jx(2) > 1e-6,
        "east of the crest layer 2 flows east: {}",
        jx(2)
    );
    assert!(
        jx(14) < -1e-6,
        "west of the crest layer 2 flows west: {}",
        jx(14)
    );
    // Without rotation or beta nothing moves in y.
    let jy: f64 = (0..NQ).map(|q| CY[q] * pw.f2[2][1][q]).sum();
    assert!(jy.abs() < 1e-14, "{jy}");
}

#[test]
fn each_layer_conserves_its_mass() {
    let mut pw = PlanetaryWave::new(16, 8, 1.6, 1.3, 1.0e-4, G_PRIME, H1, H2)
        .with_gravity(GRAVITY)
        .with_coriolis(2.0e-3);
    pw.kelvin_wave_ic(0.03, 1.0);
    let (m1, m2) = pw.total_mass();
    for _ in 0..300 {
        pw.step();
    }
    let (n1, n2) = pw.total_mass();
    assert!((n1 - m1).abs() < 1e-11 * m1, "layer 1: {m1} -> {n1}");
    assert!((n2 - m2).abs() < 1e-11 * m2, "layer 2: {m2} -> {n2}");
}

// ─────────────────────────────────────────────────────────────────────────────
// Wave propagation
// ─────────────────────────────────────────────────────────────────────────────

/// Angular frequency from the zero crossings of a standing-wave amplitude
/// series `a cos(ω t)`: the first and third crossing are one period apart.
fn frequency_from_crossings(series: &[f64]) -> f64 {
    let zeros = zero_crossings(series);
    assert!(zeros.len() >= 3, "too few zero crossings: {zeros:?}");
    2.0 * PI / (zeros[2] - zeros[0])
}

#[test]
fn baroclinic_mode_oscillates_at_the_internal_gravity_wave_speed() {
    let nx = 32;
    let mut pw = lattice_model(nx, 4, 1.9);
    pw.baroclinic_wave_ic(0.01, 1.0);
    let (_, c_bc) = pw.gravity_wave_speeds();
    let k = 2.0 * PI / nx as f64;
    let expected = k * c_bc;
    let steps = (1.45 * 2.0 * PI / expected) as usize;
    let mut max_surface = 0.0f64;
    let mut series = vec![cosine_amplitude(&pw.interface_displacement(), 1.0)];
    for _ in 0..steps {
        pw.step();
        series.push(cosine_amplitude(&pw.interface_displacement(), 1.0));
        max_surface = max_surface.max(cosine_amplitude(&pw.surface_elevation(), 1.0).abs());
    }
    let omega = frequency_from_crossings(&series);
    assert!(
        (omega - expected).abs() < 0.03 * expected,
        "baroclinic frequency {omega}, expected k c = {expected} (c = {c_bc})"
    );
    // The mode is baroclinic: the free surface stays flat.
    assert!(max_surface < 0.1 * 0.01, "surface elevation {max_surface}");
    // Uncoupled layers would oscillate with sqrt(g h1) ~ 0.2, seven times faster.
    assert!(omega < 0.25 * k * (GRAVITY * H1).sqrt());
}

#[test]
fn barotropic_mode_oscillates_at_the_surface_gravity_wave_speed() {
    let nx = 32;
    let mut pw = lattice_model(nx, 4, 1.9);
    pw.barotropic_wave_ic(0.01, 1.0);
    let (c_bt, _) = pw.gravity_wave_speeds();
    let k = 2.0 * PI / nx as f64;
    let expected = k * c_bt;
    let steps = (1.45 * 2.0 * PI / expected) as usize;
    let series = record(&mut pw, steps, 1.0, PlanetaryWave::interface_displacement);
    let omega = frequency_from_crossings(&series);
    assert!(
        (omega - expected).abs() < 0.04 * expected,
        "barotropic frequency {omega}, expected k c = {expected}"
    );
}

#[test]
fn rotation_turns_the_interface_wave_into_an_inertia_gravity_wave() {
    // f-plane, standing baroclinic mode started at rest: the interface anomaly
    // is f0^2/w^2 + (c k)^2/w^2 cos(w t) with w^2 = f0^2 + (c k)^2, which has
    // its minimum at t = pi / w.
    let nx = 32;
    let f0 = 0.004;
    let mut pw = lattice_model(nx, 4, 1.9).with_coriolis(f0);
    pw.baroclinic_wave_ic(0.01, 1.0);
    let (_, c_bc) = pw.gravity_wave_speeds();
    let k = 2.0 * PI / nx as f64;
    let omega = (f0 * f0 + k * k * c_bc * c_bc).sqrt();
    let non_rotating = k * c_bc;
    let steps = (1.4 * PI / omega) as usize;
    let mut max_uy = 0.0f64;
    let mut series = vec![cosine_amplitude(&pw.interface_displacement(), 1.0)];
    for _ in 0..steps {
        pw.step();
        series.push(cosine_amplitude(&pw.interface_displacement(), 1.0));
        let (_, _, uy) = macroscopic(&pw.f1[nx / 4][1]);
        max_uy = max_uy.max(uy.abs());
    }
    let t_min = time_of_minimum(&series);
    let expected = PI / omega;
    assert!(
        (t_min - expected).abs() < 0.04 * expected,
        "minimum at t = {t_min}, expected pi/w = {expected} (non-rotating pi/(k c) = {})",
        PI / non_rotating
    );
    assert!(
        (PI / non_rotating - expected).abs() > 0.15 * expected,
        "the rotating period must differ clearly from the non-rotating one"
    );
    assert!(
        max_uy > 1e-6,
        "geostrophic adjustment drives a cross-wave flow: {max_uy}"
    );
}

#[test]
fn inertial_oscillation_rotates_the_flow_without_changing_its_speed() {
    // Uniform flow on an f-plane: Crank-Nicolson rotation by 2 atan(f/2) per
    // step, the speed is conserved to rounding.
    let f0 = 0.05;
    let u0 = 0.02;
    let mut pw = lattice_model(6, 5, 1.3).with_coriolis(f0);
    for ix in 0..pw.nx {
        for iy in 0..pw.ny {
            pw.f1[ix][iy] = ShallowWaterLbm::feq_sw(H1, u0, 0.0, GRAVITY);
            pw.f2[ix][iy] = ShallowWaterLbm::feq_sw(H2, u0, 0.0, GRAVITY);
        }
    }
    let steps = 120;
    for _ in 0..steps {
        pw.step();
    }
    let angle = steps as f64 * 2.0 * (0.5 * f0).atan();
    for (layer, thickness) in [(&pw.f1, H1), (&pw.f2, H2)] {
        let (h, ux, uy) = macroscopic(&layer[2][3]);
        assert!((h - thickness).abs() < 1e-13, "depth {h}");
        assert!(
            ((ux * ux + uy * uy).sqrt() - u0).abs() < 1e-11,
            "speed {}",
            (ux * ux + uy * uy).sqrt()
        );
        // The Coriolis force -f k x u turns the flow clockwise.
        assert!((ux - u0 * angle.cos()).abs() < 1e-9, "ux {ux}");
        assert!((uy + u0 * angle.sin()).abs() < 1e-9, "uy {uy}");
    }
}

#[test]
fn test_planetary_wave_step_finite() {
    let mut pw = PlanetaryWave::new(8, 8, 1.2, 1.4, 1e-4, G_PRIME, H1, H2)
        .with_gravity(GRAVITY)
        .with_coriolis(1e-3);
    pw.kelvin_wave_ic(0.02, 1.0);
    for _ in 0..200 {
        pw.step();
    }
    let disp = pw.interface_displacement();
    assert!(disp.iter().flatten().all(|&v| v.is_finite()));
    assert!(
        pw.surface_elevation()
            .iter()
            .flatten()
            .all(|&v| v.is_finite())
    );
    assert!(disp.iter().flatten().any(|&v| v.abs() > 1e-6));
}
