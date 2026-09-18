//! Demonstration of physics and engineering special functions
//!
//! This example shows various physics and engineering functions available
//! in the scirs2-special module.

use scirs2_core::Complex64;
use scirs2_special::physics_engineering::*;

#[allow(dead_code)]
fn main() {
    println!("=== Physics and Engineering Functions Demo ===\n");

    // Blackbody radiation examples
    println!("1. Blackbody Radiation");
    println!("----------------------");
    let sun_temp = 5778.0; // K
    let wavelengthmax = blackbody::wien_displacement(sun_temp).expect("Test: operation failed");
    println!(
        "Sun's peak emission wavelength: {:.1} nm",
        wavelengthmax * 1e9
    );

    let power = blackbody::stefan_boltzmann_law(sun_temp).expect("Test: operation failed");
    println!("Sun's surface power density: {:.2e} W/m²", power);

    let frequency = 6e14; // Hz (green light)
    let radiance = blackbody::planck_law(frequency, sun_temp).expect("Test: operation failed");
    println!(
        "Spectral radiance at 500nm: {:.2e} W⋅sr⁻¹⋅m⁻²⋅Hz⁻¹",
        radiance
    );

    // Antenna and RF examples
    println!("\n2. Antenna and RF Engineering");
    println!("-----------------------------");
    let dipole = antenna::dipole_pattern(std::f64::consts::PI / 4.0, 0.5);
    println!("Half-wave dipole pattern at 45°: {:.3}", dipole);

    let array_factor = antenna::array_factor(8, 0.5, std::f64::consts::PI / 6.0, 0.0);
    println!(
        "8-element array factor: {:.3} + {:.3}i",
        array_factor.re, array_factor.im
    );

    let pr_dbm =
        antenna::friis_equation(10.0, 6.0, 6.0, 2.4e9, 1000.0).expect("Test: operation failed");
    println!("Received power at 1km (2.4GHz): {:.1} dBm", pr_dbm);

    // Acoustics examples
    println!("\n3. Acoustics");
    println!("-------------");
    let c_sound = acoustics::speed_of_sound(20.0, 0.5, 101325.0).expect("Test: operation failed");
    println!("Speed of sound at 20°C, 50% humidity: {:.1} m/s", c_sound);

    let spl = acoustics::sound_pressure_level(2.0, 20e-6).expect("Test: operation failed");
    println!("SPL for 2 Pa: {:.1} dB", spl);

    let a_weight_1k = acoustics::a_weighting(1000.0).expect("Test: operation failed");
    let a_weight_100 = acoustics::a_weighting(100.0).expect("Test: operation failed");
    println!("A-weighting at 1kHz: {:.1} dB", a_weight_1k);
    println!("A-weighting at 100Hz: {:.1} dB", a_weight_100);

    let helmholtz_freq =
        acoustics::helmholtz_frequency(0.001, 0.1, 1e-4, 20.0).expect("Test: operation failed");
    println!("Helmholtz resonator frequency: {:.1} Hz", helmholtz_freq);

    // Optics examples
    println!("\n4. Optics and Photonics");
    println!("------------------------");
    let brewster = optics::brewster_angle(1.0, 1.5).expect("Test: operation failed");
    println!(
        "Brewster's angle for glass: {:.1}°",
        brewster * 180.0 / std::f64::consts::PI
    );

    let na = optics::numerical_aperture(1.48, 1.46).expect("Test: operation failed");
    println!("Fiber numerical aperture: {:.3}", na);

    let beam_radius =
        optics::gaussian_beam_radius(0.1, 1e-3, 633e-9).expect("Test: operation failed");
    println!(
        "Gaussian beam radius at 10cm: {:.2} mm",
        beam_radius * 1000.0
    );

    let (r_s, _t_s) =
        optics::fresnel_coefficients(1.0, 1.5, 0.5, 's').expect("Test: operation failed");
    println!("Fresnel reflection (s-pol, 30°): {:.3}", r_s.norm());

    // Thermal examples
    println!("\n5. Heat Transfer");
    println!("-----------------");
    let thermal_diff =
        thermal::thermal_diffusion_length(1e-6, 1000.0).expect("Test: operation failed");
    println!(
        "Thermal diffusion length at 1kHz: {:.2} μm",
        thermal_diff * 1e6
    );

    let biot = thermal::biot_number(100.0, 0.01, 200.0).expect("Test: operation failed");
    println!("Biot number: {:.3}", biot);

    let nusselt = thermal::nusselt_vertical_plate(1e8).expect("Test: operation failed");
    println!("Nusselt number (Ra=1e8): {:.1}", nusselt);

    let view_factor =
        thermal::view_factor_parallel_plates(1.0, 1.0, 0.5).expect("Test: operation failed");
    println!("View factor between parallel plates: {:.3}", view_factor);

    // Semiconductor examples
    println!("\n6. Semiconductor Physics");
    println!("------------------------");
    let ni =
        semiconductor::intrinsic_carrier_concentration_si(300.0).expect("Test: operation failed");
    println!(
        "Silicon intrinsic carrier concentration at 300K: {:.2e} cm⁻³",
        ni
    );

    let fermi = semiconductor::fermi_dirac(1.0, 0.5, 300.0).expect("Test: operation failed");
    println!("Fermi-Dirac at E-Ef=0.5eV: {:.6}", fermi);

    let debye = semiconductor::debye_length(300.0, 1e16, 11.8).expect("Test: operation failed");
    println!("Debye length in Si (n=1e16): {:.2} nm", debye * 1e7);

    // Plasma physics examples
    println!("\n7. Plasma Physics");
    println!("-----------------");
    let plasma_freq = plasma::plasma_frequency(1e18).expect("Test: operation failed");
    println!("Plasma frequency (n=1e18 m⁻³): {:.2e} Hz", plasma_freq);

    let debye_plasma = plasma::debye_length_plasma(1e4, 1e18).expect("Test: operation failed");
    println!("Debye length in plasma: {:.2} μm", debye_plasma * 1e6);

    let cyclotron =
        plasma::cyclotron_frequency(1.0, 9.109e-31, -1.602e-19).expect("Test: operation failed");
    println!("Electron cyclotron frequency at 1T: {:.2e} Hz", cyclotron);

    let alfven = plasma::alfven_velocity(0.1, 1e-3).expect("Test: operation failed");
    println!("Alfvén velocity: {:.2e} m/s", alfven);

    // Quantum mechanics examples
    println!("\n8. Quantum Mechanics");
    println!("--------------------");
    let de_broglie = quantum::de_broglie_wavelength(6.626e-27).expect("Test: operation failed");
    println!("de Broglie wavelength: {:.2} nm", de_broglie * 1e9);

    let compton = quantum::compton_wavelength(9.109e-31).expect("Test: operation failed");
    println!("Electron Compton wavelength: {:.3} pm", compton * 1e12);

    let bohr = quantum::bohr_radius(1).expect("Test: operation failed");
    println!("Bohr radius: {:.2} Å", bohr * 1e10);

    let rydberg = quantum::rydberg_wavelength(3, 2, 1).expect("Test: operation failed");
    println!("H-alpha wavelength: {:.1} nm", rydberg * 1e9);

    // Transmission line examples
    println!("\n9. Transmission Lines");
    println!("---------------------");
    let z0 = transmission_lines::coax_impedance(10e-3, 3e-3, 2.1).expect("Test: operation failed");
    println!("Coax impedance: {:.1} Ω", z0);

    let skin = transmission_lines::skin_depth(1e9, 5.8e7, 1.0).expect("Test: operation failed");
    println!("Skin depth in copper at 1GHz: {:.2} μm", skin * 1e6);

    let z_load = Complex64::new(75.0, 25.0);
    let gamma =
        transmission_lines::reflection_coefficient(z_load, 50.0).expect("Test: operation failed");
    println!(
        "Reflection coefficient: {:.3} ∠{:.1}°",
        gamma.norm(),
        gamma.arg() * 180.0 / std::f64::consts::PI
    );

    let vswr = transmission_lines::vswr(gamma.norm()).expect("Test: operation failed");
    println!("VSWR: {:.2}", vswr);

    println!("\n=== Demo Complete ===");
}
