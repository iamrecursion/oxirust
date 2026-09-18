//! Auto-generated module
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use scirs2_core::Complex64;
use scirs2_special::*;
use std::f64::consts::PI;

#[allow(dead_code)]
pub(super) fn ising_magnetization(t: f64, tc: f64) -> f64 {
    if t >= tc { 0.0 } else { (1.0 - t / tc).powf(0.125) }
}
#[allow(dead_code)]
pub(super) fn ising_correlation_length(t: f64, tc: f64) -> f64 {
    (t - tc).abs().powf(-1.0)
}
#[allow(dead_code)]
pub(super) fn ising_heat_capacity(t: f64, tc: f64) -> f64 {
    if (t - tc).abs() < 1e-6 { 1000.0 } else { (t - tc).abs().powf(-0.1) }
}
#[allow(dead_code)]
pub(super) fn bcs_gap_ratio(t: f64, tc: f64) -> f64 {
    if t >= tc { 0.0 } else { (1.0 - t / tc).sqrt() }
}
#[allow(dead_code)]
pub(super) fn bcs_heat_capacity(t: f64, tc: f64) -> f64 {
    if t >= tc {
        t / tc
    } else {
        let gap_ratio = bcs_gap_ratio(t, tc);
        gap_ratio * gap_ratio * (-1.76 * tc / t).exp()
    }
}
#[allow(dead_code)]
pub(super) fn bcs_critical_field(t: f64, tc: f64) -> f64 {
    if t >= tc { 0.0 } else { (1.0 - (t / tc).powi(2)).sqrt() }
}
#[allow(dead_code)]
pub(super) fn quantum_susceptibility(g: f64) -> f64 {
    g.abs().powf(-0.75)
}
#[allow(dead_code)]
pub(super) fn quantum_correlation_length(g: f64) -> f64 {
    g.abs().powf(-1.0)
}
#[allow(dead_code)]
pub(super) fn quantum_scaling_function(g: f64) -> f64 {
    (-g.abs()).exp()
}
#[allow(dead_code)]
pub(super) fn chirp_time(_mass_solar: f64, f0hz: f64) -> f64 {
    let total_mass_kg = _mass_solar * 1.989e30;
    let g: f64 = 6.674e-11;
    let c: f64 = 2.998e8;
    5.0 * c.powi(5) / (256.0 * PI) * (total_mass_kg * g / c.powi(3)).powf(-5.0 / 3.0)
        / f0hz.powf(8.0 / 3.0)
}
#[allow(dead_code)]
pub(super) fn inspiral_frequency(t: f64, f0: f64, tau: f64) -> f64 {
    f0 * (1.0 - t / tau).powf(-3.0 / 8.0)
}
#[allow(dead_code)]
pub(super) fn inspiral_strain(t: f64, mass: f64, tau: f64) -> f64 {
    let distance = 410e6 * 3.086e16;
    1e-21 * (mass / 30.0) * (100e6 * 3.086e16 / distance)
        * (1.0 - t / tau).powf(-1.0 / 4.0)
}
#[allow(dead_code)]
pub(super) fn detection_probability(snr: f64, threshold: f64) -> f64 {
    if snr >= threshold { 0.999 } else { (snr / threshold).powi(2) }
}
#[allow(dead_code)]
pub(super) fn false_alarm_rate_from_snr(snr: f64) -> f64 {
    (-snr.powi(2) / 2.0).exp() / 1e6
}
#[allow(dead_code)]
pub(super) fn gaussian_significance(snr: f64) -> f64 {
    snr / (2.0_f64).sqrt()
}
#[allow(dead_code)]
pub(super) fn parameter_uncertainty_mass(m1: f64, m2: f64, distance: f64) -> f64 {
    0.1 / (distance / 100.0).sqrt()
}
#[allow(dead_code)]
pub(super) fn parameter_uncertainty_distance(_m1: f64, m2: f64, distance: f64) -> f64 {
    0.5 * (distance / 100.0).sqrt()
}
#[allow(dead_code)]
pub(super) fn average_thermal_velocity(_temp_k: f64, masskg: f64) -> f64 {
    let k_b = 1.381e-23;
    (8.0 * k_b * _temp_k / (PI * masskg)).sqrt()
}
#[allow(dead_code)]
pub(super) fn thermal_velocity(_temp_k: f64, masskg: f64) -> f64 {
    let k_b = 1.381e-23;
    (k_b * _temp_k / masskg).sqrt()
}
#[allow(dead_code)]
pub(super) fn most_probable_velocity(_temp_k: f64, masskg: f64) -> f64 {
    let k_b = 1.381e-23;
    (2.0 * k_b * _temp_k / masskg).sqrt()
}
#[allow(dead_code)]
pub(super) fn fusion_reaction_rate(_tempkev: f64, z1: f64, z2: f64) -> f64 {
    let gamow_energy = gamow_peak_energy(_tempkev, z1, z2);
    1e-16 * (gamow_energy / _tempkev).exp()
}
#[allow(dead_code)]
pub(super) fn gamow_peak_energy(_tempkev: f64, z1: f64, z2: f64) -> f64 {
    1.22 * (z1 * z2).powf(2.0 / 3.0) * _tempkev.powf(1.0 / 3.0)
}
#[allow(dead_code)]
pub(super) fn fusion_power_density(_temp_kev: f64, rate: f64, qvalue: f64) -> f64 {
    let density = 1e20;
    rate * 1e-6 * density * density * qvalue * 1.602e-13 * 1e-6
}
#[allow(dead_code)]
pub(super) fn plasma_dispersion_function(zeta: Complex64) -> Complex64 {
    Complex64::new(1.0 / zeta.re, -PI.sqrt() * (-zeta.re.powi(2)).exp())
}
#[allow(dead_code)]
pub(super) fn larmor_radius_cm(_energy_kev: f64, b_fieldt: f64) -> f64 {
    let mass_kg = 3.34e-27;
    let charge = 1.602e-19;
    let velocity = (2.0 * _energy_kev * 1000.0 * 1.602e-19 / mass_kg).sqrt();
    mass_kg * velocity / (charge * b_fieldt) * 100.0
}
#[allow(dead_code)]
pub(super) fn cyclotron_frequency_mhz(_b_fieldt: f64) -> f64 {
    let charge = 1.602e-19;
    let mass_kg = 3.34e-27;
    charge * _b_fieldt / (2.0 * PI * mass_kg) / 1e6
}
#[allow(dead_code)]
pub(super) fn banana_orbit_width(_energy_kev: f64, b_fieldt: f64) -> f64 {
    larmor_radius_cm(_energy_kev, b_fieldt) * 2.0
}
#[allow(dead_code)]
pub(super) fn hydrogen_wavelength(n1: i32, n2: i32) -> f64 {
    let rydberg = 1.097e7;
    let wavelength_m = 1.0
        / (rydberg * (1.0 / (n1 * n1) as f64 - 1.0 / (n2 * n2) as f64));
    wavelength_m * 1e9
}
#[allow(dead_code)]
pub(super) fn sodium_d_line_wavelength(line: &str) -> f64 {
    match line {
        "D₁" => 589.6,
        "D₂" => 589.3,
        _ => 589.0,
    }
}
#[allow(dead_code)]
pub(super) fn fine_structure_splitting(n: i32, l: i32) -> f64 {
    let alpha = 1.0 / 137.0;
    let rydberg_ev = 13.6;
    alpha * alpha * rydberg_ev * 1000.0 / (n as f64).powi(3)
}
#[allow(dead_code)]
pub(super) fn vibrational_rotational_energy(
    v: i32,
    j: i32,
    constants: (f64, f64, f64),
) -> f64 {
    let (omega_e, b_e, alpha_e) = constants;
    omega_e * (v as f64 + 0.5) + b_e * j as f64 * (j + 1) as f64
        - alpha_e * (v as f64 + 0.5) * j as f64 * (j + 1) as f64
}
#[allow(dead_code)]
pub(super) fn co_fundamental_p_branch(j: i32) -> f64 {
    2170.2 - 2.0 * 1.931 * j as f64
}
#[allow(dead_code)]
pub(super) fn co_fundamental_r_branch(j: i32) -> f64 {
    2170.2 + 2.0 * 1.931 * (j + 1) as f64
}
#[allow(dead_code)]
pub(super) fn charging_energy_mev(capacitance: f64) -> f64 {
    let e = 1.602e-19;
    e * e / (2.0 * capacitance) / 1.602e-16
}
#[allow(dead_code)]
pub(super) fn max_electrons_thermal(_charging_energy_mev: f64, tempk: f64) -> f64 {
    let k_b_mev = 8.617e-5 * 1000.0;
    _charging_energy_mev / (k_b_mev * tempk)
}
#[allow(dead_code)]
pub(super) fn wigner_surmise_goe(s: f64) -> f64 {
    (PI / 2.0) * s * (-PI * s * s / 4.0).exp()
}
#[allow(dead_code)]
pub(super) fn wigner_surmise_gue(s: f64) -> f64 {
    (32.0 / (PI * PI)) * s * s * (-4.0 * s * s / PI).exp()
}
#[allow(dead_code)]
pub(super) fn shot_noise_tunnel_junction(current: f64) -> f64 {
    2.0 * 1.602e-19 * current
}
#[allow(dead_code)]
pub(super) fn diffusion_coefficient(radius: f64, temp: f64, viscosity: f64) -> f64 {
    let k_b = 1.381e-23;
    k_b * temp / (6.0 * PI * viscosity * radius)
}
#[allow(dead_code)]
pub(super) fn momentum_correlation_time(radius: f64, viscosity: f64) -> f64 {
    let mass = 4.0 / 3.0 * PI * radius.powi(3) * 1000.0;
    mass / (6.0 * PI * viscosity * radius)
}
#[allow(dead_code)]
pub(super) fn finitesize_correlation(size: f64, nu: f64) -> f64 {
    1.0 / size.powf(1.0 / nu)
}
#[allow(dead_code)]
pub(super) fn finitesize_susceptibility(size: f64, exponents: (f64, f64, f64)) -> f64 {
    let (nu, gamma, beta) = exponents;
    size.powf(-gamma / nu)
}
#[allow(dead_code)]
pub(super) fn finitesize_order_parameter(size: f64, beta: f64) -> f64 {
    size.powf(-beta)
}
#[allow(dead_code)]
pub(super) fn jacobi_cn(u: f64, m: f64) -> f64 {
    if m < 0.1 {
        u.cos()
    } else if m > 0.9 {
        1.0 / u.cosh()
    } else {
        (1.0 - m * u.sin().powi(2)).sqrt()
    }
}
#[allow(dead_code)]
pub(super) fn complete_elliptic_k(m: f64) -> f64 {
    if m < 0.1 { PI / 2.0 } else { PI / 2.0 * (1.0 + m / 4.0) }
}
#[allow(dead_code)]
pub(super) fn peregrine_amplitude_squared(z: f64, t: f64) -> f64 {
    let denominator = 1.0 + 4.0 * z * z + 4.0 * t * t;
    let numerator = 4.0 * (1.0 + 2.0 * z);
    (1.0 - numerator / denominator).powi(2)
}
#[allow(dead_code)]
pub(super) fn peregrine_phase(z: f64, t: f64) -> f64 {
    z + 2.0 * (2.0 * z / (1.0 + 4.0 * z * z + 4.0 * t * t)).atan()
}
#[allow(dead_code)]
pub(super) fn kerr_comb_threshold(detuning: f64) -> f64 {
    1.0 + detuning.abs()
}
#[allow(dead_code)]
pub(super) fn estimate_comb_lines(detuning: f64, power: f64) -> f64 {
    if detuning < 0.0 && power > kerr_comb_threshold(detuning) {
        10.0 * (-detuning).sqrt()
    } else {
        1.0
    }
}
#[allow(dead_code)]
pub(super) fn cosmic_ray_flux(energy: f64, gamma: f64) -> f64 {
    1e4 * energy.powf(-gamma)
}
#[allow(dead_code)]
pub(super) fn werner_state_entropy(p: f64) -> f64 {
    if p == 0.0 || p == 1.0 {
        0.0
    } else {
        let lambda1 = (1.0 + 3.0 * p) / 4.0;
        let lambda2 = (1.0 - p) / 4.0;
        -lambda1 * lambda1.log2() - 3.0 * lambda2 * lambda2.log2()
    }
}
#[allow(dead_code)]
pub(super) fn werner_state_concurrence(p: f64) -> f64 {
    if p > 1.0 / 3.0 { 3.0 * p - 1.0 } else { 0.0 }
}
#[allow(dead_code)]
pub(super) fn shor_classical_complexity(n: f64) -> f64 {
    (n.powf(1.0 / 3.0)).exp()
}
#[allow(dead_code)]
pub(super) fn shor_quantum_complexity(n: f64) -> f64 {
    n.powi(3)
}
