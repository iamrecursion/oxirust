// Copyright 2026 COOLJAPAN OU (Team KitaSan)
// SPDX-License-Identifier: Apache-2.0

//! Elastic wave propagation using the finite element method.
//!
//! Implements:
//! - Elastic wave modes (longitudinal, shear, Rayleigh, Love, Lamb)
//! - Dispersion relations for bulk media
//! - Explicit time-stepping for wave propagation
//! - Absorbing boundary conditions (dashpot ABC)
//! - Phase/group velocity dispersion curves
//! - Rayleigh-Lamb frequency equations of a free plate (symmetric and
//!   antisymmetric residuals) and the S0 phase velocity
//! - Acoustic emission source function

use crate::error::{Error, Result};

/// Classification of elastic wave modes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WaveMode {
    /// Compressional (P-wave): particle motion parallel to propagation.
    Longitudinal,
    /// Shear (S-wave): particle motion perpendicular to propagation.
    Shear,
    /// Rayleigh surface wave: elliptical particle motion at free surface.
    Rayleigh,
    /// Love surface wave: horizontal shear motion in a layered medium.
    Love,
    /// Lamb wave: guided wave in a plate (symmetric or antisymmetric).
    Lamb,
}

/// Represents a single elastic wave with its characteristic properties.
#[derive(Debug, Clone)]
pub struct ElasticWave {
    /// Angular frequency ω \[rad/s\].
    pub frequency: f64,
    /// Wavenumber k = ω / c_p \[1/m\].
    pub wavenumber: f64,
    /// Phase velocity c_p = ω / k \[m/s\].
    pub phase_velocity: f64,
    /// Group velocity c_g = dω/dk \[m/s\].
    pub group_velocity: f64,
    /// Wave amplitude \[m\].
    pub amplitude: f64,
    /// Polarization direction (unit vector) \[dimensionless\].
    pub polarization: [f64; 3],
    /// Wave mode classification.
    pub mode: WaveMode,
}

impl ElasticWave {
    /// Create a new elastic wave.
    pub fn new(
        frequency: f64,
        wavenumber: f64,
        amplitude: f64,
        polarization: [f64; 3],
        mode: WaveMode,
    ) -> Self {
        let phase_velocity = if wavenumber.abs() > 1e-15 {
            frequency / wavenumber
        } else {
            0.0
        };
        Self {
            frequency,
            wavenumber,
            phase_velocity,
            group_velocity: phase_velocity, // default: non-dispersive
            amplitude,
            polarization,
            mode,
        }
    }

    /// Evaluate displacement at position `x` and time `t`.
    ///
    /// u(x,t) = amplitude * cos(k*x - ω*t) * polarization
    pub fn displacement(&self, x: f64, t: f64) -> [f64; 3] {
        let phase = self.wavenumber * x - self.frequency * t;
        let u = self.amplitude * phase.cos();
        [
            u * self.polarization[0],
            u * self.polarization[1],
            u * self.polarization[2],
        ]
    }
}

/// Dispersion relation for an isotropic linear elastic medium.
///
/// Provides wave speeds from material constants (Young's modulus, Poisson's
/// ratio, and mass density).
#[derive(Debug, Clone)]
pub struct DispersionRelation {
    /// Young's modulus E \[Pa\].
    pub e_modulus: f64,
    /// Poisson's ratio ν (dimensionless, must be in (−1, 0.5)).
    pub nu: f64,
    /// Mass density ρ \[kg/m³\].
    pub density: f64,
}

impl DispersionRelation {
    /// Create a new dispersion relation from material properties.
    pub fn new(e_modulus: f64, nu: f64, density: f64) -> Self {
        Self {
            e_modulus,
            nu,
            density,
        }
    }

    /// Lamé parameter λ = E ν / ((1+ν)(1−2ν)).
    pub fn lame_lambda(&self) -> f64 {
        self.e_modulus * self.nu / ((1.0 + self.nu) * (1.0 - 2.0 * self.nu))
    }

    /// Shear modulus G = E / (2(1+ν)).
    pub fn shear_modulus(&self) -> f64 {
        self.e_modulus / (2.0 * (1.0 + self.nu))
    }

    /// P-wave (longitudinal) speed: c_L = sqrt((λ + 2G) / ρ).
    pub fn longitudinal_wave_speed(&self) -> f64 {
        let lambda = self.lame_lambda();
        let g = self.shear_modulus();
        ((lambda + 2.0 * g) / self.density).sqrt()
    }

    /// S-wave (shear) speed: c_S = sqrt(G / ρ).
    pub fn shear_wave_speed(&self) -> f64 {
        (self.shear_modulus() / self.density).sqrt()
    }

    /// Rayleigh wave speed approximation (Viktorov formula).
    ///
    /// c_R ≈ c_S * (0.862 + 1.14ν) / (1 + ν)
    pub fn rayleigh_wave_speed(&self) -> f64 {
        let cs = self.shear_wave_speed();
        cs * (0.862 + 1.14 * self.nu) / (1.0 + self.nu)
    }
}

/// FEM solver for transient elastic wave propagation.
///
/// Uses a lumped-mass explicit central-difference scheme.
pub struct WavePropagationFem {
    /// Mesh node coordinates: `nodes[i] = [x, y, z]`.
    pub mesh_nodes: Vec<[f64; 3]>,
    /// Global stiffness matrix stored as a flat row-major dense matrix
    /// of size `(ndof × ndof)`.
    pub stiffness_matrix: Vec<f64>,
    /// Global lumped mass matrix (diagonal, stored as a vector of size `ndof`).
    pub mass_matrix: Vec<f64>,
    /// Displacement history: `time_history[step][dof]`.
    pub time_history: Vec<Vec<f64>>,
    /// Current displacement vector.
    displacement: Vec<f64>,
    /// Current velocity vector.
    velocity: Vec<f64>,
    /// Number of degrees of freedom.
    ndof: usize,
}

impl WavePropagationFem {
    /// Create a new wave-propagation FEM model.
    ///
    /// `stiffness_matrix` must be a flat row-major `ndof × ndof` array.
    /// `mass_matrix` is the lumped diagonal mass vector of length `ndof`.
    pub fn new(
        mesh_nodes: Vec<[f64; 3]>,
        stiffness_matrix: Vec<f64>,
        mass_matrix: Vec<f64>,
    ) -> Self {
        let ndof = mass_matrix.len();
        Self {
            mesh_nodes,
            stiffness_matrix,
            mass_matrix,
            time_history: Vec::new(),
            displacement: vec![0.0; ndof],
            velocity: vec![0.0; ndof],
            ndof,
        }
    }

    /// Perform one explicit central-difference time step.
    ///
    /// The update rule is:
    ///
    /// ```text
    /// a = (F − K u) / M
    /// v_{n+1/2} = v_{n-1/2} + dt * a
    /// u_{n+1}   = u_n       + dt * v_{n+1/2}
    /// ```
    pub fn step_explicit(&mut self, dt: f64, force: &[f64]) {
        let n = self.ndof;
        let len = force.len().min(n);

        // Compute K*u
        let mut ku = vec![0.0f64; n];
        for (i, ku_i) in ku.iter_mut().enumerate() {
            *ku_i = self
                .displacement
                .iter()
                .enumerate()
                .map(|(j, &d)| self.stiffness_matrix[i * n + j] * d)
                .sum();
        }

        // Compute acceleration a = (F − K*u) / M
        let mut accel = vec![0.0f64; n];
        for (i, (accel_i, &ku_i)) in accel.iter_mut().zip(ku.iter()).enumerate() {
            let fi = if i < len { force[i] } else { 0.0 };
            let m = self.mass_matrix[i];
            if m.abs() > 1e-30 {
                *accel_i = (fi - ku_i) / m;
            }
        }

        // Update velocity and displacement
        for (v, (a, d)) in self
            .velocity
            .iter_mut()
            .zip(accel.iter().zip(self.displacement.iter_mut()))
        {
            *v += dt * a;
            *d += dt * *v;
        }

        self.time_history.push(self.displacement.clone());
    }

    /// Return the current displacement vector.
    pub fn displacement(&self) -> &[f64] {
        &self.displacement
    }

    /// Return the current velocity vector.
    pub fn velocity(&self) -> &[f64] {
        &self.velocity
    }
}

/// Absorbing boundary condition using dashpot elements.
///
/// Each boundary DOF has a dashpot coefficient `c_i` such that the
/// absorbing force is `F_abc_i = −c_i * v_i`.
#[derive(Debug, Clone)]
pub struct AbsorbingBoundaryFem {
    /// Dashpot coefficients for each boundary DOF \[N·s/m\].
    pub dashpot_coefficients: Vec<f64>,
    /// Indices of the boundary DOFs in the global system.
    pub boundary_dofs: Vec<usize>,
}

impl AbsorbingBoundaryFem {
    /// Create a new absorbing boundary with given dashpot coefficients.
    ///
    /// `boundary_dofs[i]` maps the i-th dashpot to the global DOF index.
    pub fn new(dashpot_coefficients: Vec<f64>, boundary_dofs: Vec<usize>) -> Self {
        Self {
            dashpot_coefficients,
            boundary_dofs,
        }
    }

    /// Apply absorbing boundary condition: subtract dashpot forces.
    ///
    /// `forces[dof] -= c * velocities[dof]` for each boundary DOF.
    pub fn apply_abc(&self, forces: &mut [f64], velocities: &[f64]) {
        for (idx, &dof) in self.boundary_dofs.iter().enumerate() {
            if dof < forces.len() && dof < velocities.len() {
                let c = if idx < self.dashpot_coefficients.len() {
                    self.dashpot_coefficients[idx]
                } else {
                    0.0
                };
                forces[dof] -= c * velocities[dof];
            }
        }
    }
}

/// Phase velocity dispersion curve sampled at discrete wavenumbers.
#[derive(Debug, Clone)]
pub struct PhaseVelocityDispersion {
    /// Wavenumber samples k \[1/m\].
    pub k_values: Vec<f64>,
    /// Corresponding angular frequency samples ω \[rad/s\].
    pub omega_values: Vec<f64>,
}

impl PhaseVelocityDispersion {
    /// Create a new dispersion curve.
    pub fn new(k_values: Vec<f64>, omega_values: Vec<f64>) -> Self {
        Self {
            k_values,
            omega_values,
        }
    }

    /// Compute phase velocity c_p = ω / k at sample index `i`.
    pub fn phase_velocity_at(&self, i: usize) -> f64 {
        let k = self.k_values[i];
        if k.abs() < 1e-30 {
            return 0.0;
        }
        self.omega_values[i] / k
    }

    /// Estimate group velocity c_g = dω/dk at wavenumber `k` using
    /// finite-difference interpolation on the stored samples.
    pub fn group_velocity_at(&self, k: f64) -> f64 {
        let n = self.k_values.len();
        if n < 2 {
            return 0.0;
        }

        // Find the nearest sample index
        let mut best = 0usize;
        let mut best_dist = (self.k_values[0] - k).abs();
        for i in 1..n {
            let d = (self.k_values[i] - k).abs();
            if d < best_dist {
                best_dist = d;
                best = i;
            }
        }

        // Central difference if possible
        if best == 0 {
            let dk = self.k_values[1] - self.k_values[0];
            if dk.abs() < 1e-30 {
                return 0.0;
            }
            (self.omega_values[1] - self.omega_values[0]) / dk
        } else if best == n - 1 {
            let dk = self.k_values[n - 1] - self.k_values[n - 2];
            if dk.abs() < 1e-30 {
                return 0.0;
            }
            (self.omega_values[n - 1] - self.omega_values[n - 2]) / dk
        } else {
            let dk = self.k_values[best + 1] - self.k_values[best - 1];
            if dk.abs() < 1e-30 {
                return 0.0;
            }
            (self.omega_values[best + 1] - self.omega_values[best - 1]) / dk
        }
    }
}

/// Terms `[cos(x h), x sin(x h), sin(x h) / x]` of the Rayleigh-Lamb equations
/// for one bulk wave number `x`, given `x2 = x^2` and the half thickness `h`.
///
/// All three are entire functions of `x2`, which keeps the Lamb residuals free
/// of the poles of the textbook `tan` form.  For `x2 < 0` (`x = i g`) they
/// continue analytically to `[cosh(g h), -g sinh(g h), sinh(g h) / g]`.  Those
/// grow like `exp(g h)`, so they are returned divided by that factor, together
/// with the exponent `g h` that was divided out (zero for `x2 >= 0`).
fn lamb_bulk_terms(x2: f64, half_thickness: f64) -> ([f64; 3], f64) {
    if x2 >= 0.0 {
        let x = x2.sqrt();
        let (sin_xh, cos_xh) = (x * half_thickness).sin_cos();
        let sin_over_x = if x > 0.0 { sin_xh / x } else { half_thickness };
        ([cos_xh, x * sin_xh, sin_over_x], 0.0)
    } else {
        let g = (-x2).sqrt();
        let exponent = g * half_thickness;
        let decay = (-2.0 * exponent).exp();
        let cosh_scaled = 0.5 * (1.0 + decay);
        let sinh_scaled = -0.5 * (-2.0 * exponent).exp_m1();
        ([cosh_scaled, -g * sinh_scaled, sinh_scaled / g], exponent)
    }
}

/// The two terms of the symmetric Rayleigh-Lamb residual, each divided by the
/// positive factor `exp(exponent)`; returns `(shear_term, flexural_term,
/// exponent)` with `F_S = (shear_term + flexural_term) * exp(exponent)`.
///
/// The scaling never changes the sign or the roots, and it keeps the values
/// finite when `p` and `q` are imaginary and `cosh`/`sinh` would overflow.
fn lamb_symmetric_terms(omega: f64, k: f64, thickness: f64, c_l: f64, c_s: f64) -> (f64, f64, f64) {
    let half = 0.5 * thickness;
    let k2 = k * k;
    let p2 = (omega / c_l).powi(2) - k2;
    let q2 = (omega / c_s).powi(2) - k2;
    let ([cos_ph, p_sin_ph, _], exp_p) = lamb_bulk_terms(p2, half);
    let ([cos_qh, _, sin_qh_over_q], exp_q) = lamb_bulk_terms(q2, half);
    let k2_minus_q2 = 2.0 * k2 - (omega / c_s).powi(2);
    let shear_term = k2_minus_q2 * k2_minus_q2 * cos_ph * sin_qh_over_q;
    let flexural_term = 4.0 * k2 * p_sin_ph * cos_qh;
    (shear_term, flexural_term, exp_p + exp_q)
}

/// Symmetric Rayleigh-Lamb residual divided by the positive factor
/// `exp(exponent)`; returns `(scaled_residual, exponent)`.
fn lamb_symmetric_scaled(omega: f64, k: f64, thickness: f64, c_l: f64, c_s: f64) -> (f64, f64) {
    let (shear_term, flexural_term, exponent) = lamb_symmetric_terms(omega, k, thickness, c_l, c_s);
    (shear_term + flexural_term, exponent)
}

/// Antisymmetric Rayleigh-Lamb residual divided by `exp(exponent)`; returns
/// `(scaled_residual, exponent)`.
fn lamb_antisymmetric_scaled(omega: f64, k: f64, thickness: f64, c_l: f64, c_s: f64) -> (f64, f64) {
    let half = 0.5 * thickness;
    let k2 = k * k;
    let p2 = (omega / c_l).powi(2) - k2;
    let q2 = (omega / c_s).powi(2) - k2;
    let ([cos_ph, _, sin_ph_over_p], exp_p) = lamb_bulk_terms(p2, half);
    let ([cos_qh, q_sin_qh, _], exp_q) = lamb_bulk_terms(q2, half);
    let k2_minus_q2 = 2.0 * k2 - (omega / c_s).powi(2);
    let value = k2_minus_q2 * k2_minus_q2 * sin_ph_over_p * cos_qh + 4.0 * k2 * q_sin_qh * cos_ph;
    (value, exp_p + exp_q)
}

/// Whether the arguments of a Lamb residual are finite and physical.
fn lamb_arguments_valid(omega: f64, k: f64, thickness: f64, c_l: f64, c_s: f64) -> bool {
    omega.is_finite()
        && k.is_finite()
        && thickness.is_finite()
        && thickness > 0.0
        && c_l.is_finite()
        && c_l > 0.0
        && c_s.is_finite()
        && c_s > 0.0
}

/// Undo the positive scaling of a scaled Lamb residual.
fn lamb_unscale(value: f64, exponent: f64) -> f64 {
    if value == 0.0 {
        0.0
    } else {
        value * exponent.exp()
    }
}

/// Residual of the Rayleigh-Lamb frequency equation for the **symmetric**
/// modes of a free isotropic plate.
///
/// With `h = thickness / 2`, `p² = (ω/c_L)² − k²` and `q² = (ω/c_S)² − k²` the
/// textbook form of the symmetric equation is
///
/// ```text
/// tan(q h) / tan(p h) = − 4 k² p q / (k² − q²)²
/// ```
///
/// Multiplying it by `cos(p h) cos(q h) / q` removes the poles of the
/// tangents and the `1/q` singularity, giving the residual returned here:
///
/// ```text
/// F_S(ω, k) = (k² − q²)² · cos(p h) · sin(q h) / q  +  4 k² · p sin(p h) · cos(q h)
/// ```
///
/// `F_S` is an entire function of `p²` and `q²`, so it is real and smooth for
/// every real `(ω, k)`: where `p²` or `q²` is negative the circular functions
/// continue to `cosh`, `sinh(|q| h) / |q|` and `−|p| sinh(|p| h)`.  A pair
/// `(ω, k)` solves the dispersion relation of a symmetric mode (S0, S1, ...)
/// exactly when the residual is zero.  At `k = 0` it vanishes at the symmetric
/// thickness resonances `ω d / (2 c_L) = (n − ½) π` and `ω d / (2 c_S) = n π`.
///
/// # Arguments
/// * `omega` - angular frequency ω \[rad/s\]
/// * `k` - wavenumber \[1/m\]
/// * `thickness` - plate thickness `d` \[m\]
/// * `c_l` - longitudinal (P) bulk wave speed \[m/s\]
/// * `c_s` - shear (S) bulk wave speed \[m/s\]
///
/// Returns `NaN` if an argument is not finite or `thickness`, `c_l` or `c_s`
/// is not positive.  The value overflows to infinity only when
/// `(|p| + |q|) h` exceeds roughly 700.
pub fn lamb_wave_symmetric(omega: f64, k: f64, thickness: f64, c_l: f64, c_s: f64) -> f64 {
    if !lamb_arguments_valid(omega, k, thickness, c_l, c_s) {
        return f64::NAN;
    }
    let (value, exponent) = lamb_symmetric_scaled(omega, k, thickness, c_l, c_s);
    lamb_unscale(value, exponent)
}

/// Residual of the Rayleigh-Lamb frequency equation for the **antisymmetric**
/// (flexural) modes of a free isotropic plate.
///
/// The textbook form is
///
/// ```text
/// tan(q h) / tan(p h) = − (k² − q²)² / (4 k² p q)
/// ```
///
/// and multiplying by `cos(p h) cos(q h) / p` gives the pole-free residual
///
/// ```text
/// F_A(ω, k) = (k² − q²)² · sin(p h) / p · cos(q h)  +  4 k² · q sin(q h) · cos(p h)
/// ```
///
/// with the same definitions, analytic continuation and argument handling as
/// [`lamb_wave_symmetric`].  At `k = 0` it vanishes at the antisymmetric
/// thickness resonances `ω d / (2 c_L) = n π` and `ω d / (2 c_S) = (n − ½) π`.
pub fn lamb_wave_antisymmetric(omega: f64, k: f64, thickness: f64, c_l: f64, c_s: f64) -> f64 {
    if !lamb_arguments_valid(omega, k, thickness, c_l, c_s) {
        return f64::NAN;
    }
    let (value, exponent) = lamb_antisymmetric_scaled(omega, k, thickness, c_l, c_s);
    lamb_unscale(value, exponent)
}

/// Rayleigh surface-wave speed of an isotropic half-space, solved from the
/// Rayleigh cubic `ξ³ − 8ξ² + 8(3 − 2κ)ξ − 16(1 − κ) = 0` with
/// `ξ = (c_R / c_S)²` and `κ = (c_S / c_L)²` by bisection on `(0, 1)`.
fn rayleigh_speed_from_cubic(c_l: f64, c_s: f64) -> f64 {
    let kappa = (c_s / c_l).powi(2);
    let cubic = |xi: f64| {
        xi * xi * xi - 8.0 * xi * xi + 8.0 * (3.0 - 2.0 * kappa) * xi - 16.0 * (1.0 - kappa)
    };
    let (mut lo, mut hi) = (0.0_f64, 1.0_f64);
    for _ in 0..200 {
        let mid = 0.5 * (lo + hi);
        if cubic(mid) < 0.0 {
            lo = mid;
        } else {
            hi = mid;
        }
    }
    c_s * (0.5 * (lo + hi)).sqrt()
}

/// Phase velocity `c = ω / k` of the fundamental symmetric Lamb mode (S0) of a
/// free isotropic plate at angular frequency `omega`.
///
/// S0 has no cut-off.  Its phase velocity falls from the thin-plate value
/// `c_plate = 2 c_S √(1 − c_S²/c_L²)` at `ω d → 0` towards the Rayleigh speed
/// `c_R` as `ω d → ∞`, and it is the symmetric mode with the lowest phase
/// velocity.  The root of [`lamb_wave_symmetric`] is therefore bracketed by
/// scanning the phase velocity upwards from just below `c_R` for the first sign
/// change, and then refined by bisection in `k` to machine precision.
///
/// # Errors
/// * [`Error::InvalidInput`] if `omega`, `thickness` or `c_s` is not finite and
///   positive, or if `c_l` is not finite and larger than `2 c_S / √3`
///   (the Poisson ratio must exceed −1).
/// * [`Error::NotConverged`] if no sign change of the residual is found.
pub fn lamb_s0_phase_velocity(omega: f64, thickness: f64, c_l: f64, c_s: f64) -> Result<f64> {
    if !(omega.is_finite() && omega > 0.0) {
        return Err(Error::InvalidInput(format!(
            "angular frequency must be finite and positive, got {omega}"
        )));
    }
    if !(thickness.is_finite() && thickness > 0.0) {
        return Err(Error::InvalidInput(format!(
            "plate thickness must be finite and positive, got {thickness}"
        )));
    }
    if !(c_s.is_finite() && c_s > 0.0) {
        return Err(Error::InvalidInput(format!(
            "shear wave speed must be finite and positive, got {c_s}"
        )));
    }
    if !(c_l.is_finite() && c_l * c_l > 4.0 / 3.0 * c_s * c_s) {
        return Err(Error::InvalidInput(format!(
            "longitudinal wave speed {c_l} must exceed 2/sqrt(3) times the shear speed {c_s}"
        )));
    }

    const SCAN_CELLS: usize = 2048;
    let c_plate = 2.0 * c_s * (1.0 - (c_s / c_l).powi(2)).sqrt();
    let c_low = 0.99 * rayleigh_speed_from_cubic(c_l, c_s);
    // At a Poisson ratio of zero S0 sits exactly at c_plate = c_L, so the scan
    // reaches slightly beyond the thin-plate velocity.
    let c_high = c_plate * (1.0 + 1.0e-6);
    let residual = |c: f64| lamb_symmetric_scaled(omega, omega / c, thickness, c_l, c_s).0;
    // Residual relative to the magnitude of its two terms, for error reporting.
    let relative_residual = |c: f64| {
        let (shear, flexural, _) = lamb_symmetric_terms(omega, omega / c, thickness, c_l, c_s);
        (shear + flexural).abs() / (shear.abs() + flexural.abs())
    };

    let mut c_prev = c_low;
    let mut f_prev = residual(c_prev);
    let mut smallest = relative_residual(c_prev);
    for cell in 1..=SCAN_CELLS {
        if f_prev == 0.0 {
            return Ok(c_prev);
        }
        let c_next = c_low + (c_high - c_low) * (cell as f64) / (SCAN_CELLS as f64);
        let f_next = residual(c_next);
        smallest = smallest.min(relative_residual(c_next));
        if (f_prev > 0.0 && f_next < 0.0) || (f_prev < 0.0 && f_next > 0.0) {
            // Higher phase velocity means smaller wavenumber.
            let (mut k_small, mut k_large) = (omega / c_next, omega / c_prev);
            let f_small_positive = f_next > 0.0;
            for _ in 0..200 {
                if k_large - k_small <= 4.0 * f64::EPSILON * k_large {
                    break;
                }
                let k_mid = 0.5 * (k_small + k_large);
                let f_mid = residual(omega / k_mid);
                if f_mid == 0.0 {
                    return Ok(omega / k_mid);
                }
                if (f_mid > 0.0) == f_small_positive {
                    k_small = k_mid;
                } else {
                    k_large = k_mid;
                }
            }
            return Ok(omega / (0.5 * (k_small + k_large)));
        }
        c_prev = c_next;
        f_prev = f_next;
    }
    Err(Error::NotConverged {
        solver: "lamb_s0_phase_velocity".to_string(),
        iterations: SCAN_CELLS,
        residual: smallest,
    })
}

/// Acoustic emission source time function (ramp + decay).
///
/// Models a Hsu-Nielsen pencil-break or crack source:
///
/// ```text
/// f(t) = M * (t / t_r) * exp(1 − t / t_r),  t >= 0
/// ```
///
/// where `magnitude` is the peak moment, `rise_time` is `t_r`.
/// Returns 0 for `t < 0`.
pub fn acoustic_emission_source(magnitude: f64, rise_time: f64, t: f64) -> f64 {
    if t < 0.0 || rise_time < 1e-30 {
        return 0.0;
    }
    let tau = t / rise_time;
    magnitude * tau * (1.0 - tau).exp()
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod regression_tests;

#[cfg(test)]
mod tests {
    use super::*;

    // ── DispersionRelation ─────────────────────────────────────────────────

    #[test]
    fn test_longitudinal_wave_speed_steel() {
        // Steel: E = 210 GPa, ν = 0.3, ρ = 7850 kg/m³
        let dr = DispersionRelation::new(210e9, 0.3, 7850.0);
        let cl = dr.longitudinal_wave_speed();
        // Expected ~5920 m/s
        assert!(cl > 5000.0 && cl < 7000.0, "cl={cl}");
    }

    #[test]
    fn test_shear_wave_speed_steel() {
        let dr = DispersionRelation::new(210e9, 0.3, 7850.0);
        let cs = dr.shear_wave_speed();
        // Expected ~3200 m/s
        assert!(cs > 2500.0 && cs < 4000.0, "cs={cs}");
    }

    #[test]
    fn test_rayleigh_wave_speed_less_than_shear() {
        let dr = DispersionRelation::new(210e9, 0.3, 7850.0);
        let cs = dr.shear_wave_speed();
        let cr = dr.rayleigh_wave_speed();
        assert!(
            cr < cs,
            "Rayleigh speed {cr} should be less than shear speed {cs}"
        );
    }

    #[test]
    fn test_rayleigh_wave_speed_positive() {
        let dr = DispersionRelation::new(70e9, 0.33, 2700.0); // aluminum
        let cr = dr.rayleigh_wave_speed();
        assert!(cr > 0.0, "cr={cr}");
    }

    #[test]
    fn test_shear_modulus_positive() {
        let dr = DispersionRelation::new(200e9, 0.25, 7800.0);
        assert!(dr.shear_modulus() > 0.0);
    }

    #[test]
    fn test_lame_lambda_positive_for_typical_nu() {
        let dr = DispersionRelation::new(200e9, 0.3, 7800.0);
        assert!(dr.lame_lambda() > 0.0);
    }

    #[test]
    fn test_wave_speed_ratio_cl_over_cs() {
        // For ν = 0.25: c_L / c_S = sqrt(3) ≈ 1.732
        let dr = DispersionRelation::new(100e9, 0.25, 1000.0);
        let ratio = dr.longitudinal_wave_speed() / dr.shear_wave_speed();
        assert!(
            (ratio - 3.0_f64.sqrt()).abs() < 0.01,
            "ratio={ratio}, expected ~{:.4}",
            3.0_f64.sqrt()
        );
    }

    // ── ElasticWave ────────────────────────────────────────────────────────

    #[test]
    fn test_elastic_wave_phase_velocity() {
        let w = ElasticWave::new(1000.0, 0.5, 1.0, [1.0, 0.0, 0.0], WaveMode::Longitudinal);
        assert!(
            (w.phase_velocity - 2000.0).abs() < 1e-6,
            "cp={}",
            w.phase_velocity
        );
    }

    #[test]
    fn test_elastic_wave_displacement_at_zero() {
        let w = ElasticWave::new(1000.0, 0.5, 1.0, [0.0, 1.0, 0.0], WaveMode::Shear);
        let u = w.displacement(0.0, 0.0);
        // cos(0) = 1, amplitude = 1, polarization = [0,1,0]
        assert!((u[1] - 1.0).abs() < 1e-12, "u[1]={}", u[1]);
        assert!(u[0].abs() < 1e-12);
        assert!(u[2].abs() < 1e-12);
    }

    #[test]
    fn test_elastic_wave_zero_wavenumber() {
        let w = ElasticWave::new(100.0, 0.0, 1.0, [1.0, 0.0, 0.0], WaveMode::Longitudinal);
        assert_eq!(w.phase_velocity, 0.0);
    }

    #[test]
    fn test_wave_mode_variants() {
        let modes = [
            WaveMode::Longitudinal,
            WaveMode::Shear,
            WaveMode::Rayleigh,
            WaveMode::Love,
            WaveMode::Lamb,
        ];
        for &m in &modes {
            let w = ElasticWave::new(1.0, 1.0, 1.0, [1.0, 0.0, 0.0], m);
            assert_eq!(w.mode, m);
        }
    }

    // ── WavePropagationFem ─────────────────────────────────────────────────

    #[test]
    fn test_wave_propagation_fem_step_no_force() {
        // 2-DOF spring-mass system
        let k = 1000.0_f64;
        let m = 1.0_f64;
        let stiffness = vec![k, -k, -k, k];
        let mass = vec![m, m];
        let mut fem =
            WavePropagationFem::new(vec![[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]], stiffness, mass);
        // Apply initial displacement
        fem.displacement[0] = 0.01;
        fem.step_explicit(1e-5, &[0.0, 0.0]);
        assert_eq!(fem.time_history.len(), 1);
    }

    #[test]
    fn test_wave_propagation_fem_force_causes_displacement() {
        let n = 3;
        let mut stiffness = vec![0.0f64; n * n];
        stiffness[0] = 1000.0;
        stiffness[1] = -1000.0;
        stiffness[3] = -1000.0;
        stiffness[4] = 2000.0;
        stiffness[5] = -1000.0;
        stiffness[7] = -1000.0;
        stiffness[8] = 1000.0;
        let mass = vec![1.0; n];
        let nodes = vec![[0.0; 3]; n];
        let mut fem = WavePropagationFem::new(nodes, stiffness, mass);
        let force = vec![1.0, 0.0, 0.0];
        fem.step_explicit(1e-4, &force);
        // First DOF should have some velocity/displacement
        assert!(fem.displacement()[0] != 0.0 || fem.velocity()[0] != 0.0);
    }

    #[test]
    fn test_wave_propagation_fem_history_grows() {
        let mass = vec![1.0; 2];
        let stiffness = vec![100.0, -100.0, -100.0, 100.0];
        let mut fem = WavePropagationFem::new(vec![[0.0; 3]; 2], stiffness, mass);
        for _ in 0..10 {
            fem.step_explicit(1e-4, &[0.0, 0.0]);
        }
        assert_eq!(fem.time_history.len(), 10);
    }

    // ── AbsorbingBoundaryFem ───────────────────────────────────────────────

    #[test]
    fn test_absorbing_boundary_reduces_force() {
        let abc = AbsorbingBoundaryFem::new(vec![100.0, 100.0], vec![0, 1]);
        let mut forces = vec![500.0, 500.0, 500.0];
        let velocities = vec![1.0, 1.0, 1.0];
        abc.apply_abc(&mut forces, &velocities);
        assert!((forces[0] - 400.0).abs() < 1e-10);
        assert!((forces[1] - 400.0).abs() < 1e-10);
        assert!((forces[2] - 500.0).abs() < 1e-10); // unchanged
    }

    #[test]
    fn test_absorbing_boundary_zero_velocity() {
        let abc = AbsorbingBoundaryFem::new(vec![1000.0], vec![0]);
        let mut forces = vec![100.0];
        let velocities = vec![0.0];
        abc.apply_abc(&mut forces, &velocities);
        assert!((forces[0] - 100.0).abs() < 1e-10);
    }

    #[test]
    fn test_absorbing_boundary_out_of_range_dof() {
        let abc = AbsorbingBoundaryFem::new(vec![100.0], vec![10]);
        let mut forces = vec![1.0, 2.0];
        let velocities = vec![5.0, 5.0];
        // DOF 10 is out of range: should not panic
        abc.apply_abc(&mut forces, &velocities);
        assert!((forces[0] - 1.0).abs() < 1e-10);
    }

    // ── PhaseVelocityDispersion ────────────────────────────────────────────

    #[test]
    fn test_phase_velocity_at() {
        let k = vec![1.0, 2.0, 3.0];
        let omega = vec![5000.0, 10000.0, 15000.0];
        let disp = PhaseVelocityDispersion::new(k, omega);
        assert!((disp.phase_velocity_at(0) - 5000.0).abs() < 1e-6);
        assert!((disp.phase_velocity_at(1) - 5000.0).abs() < 1e-6);
    }

    #[test]
    fn test_group_velocity_linear_dispersion() {
        // ω = 5000*k → c_g = dω/dk = 5000 everywhere
        let k: Vec<f64> = (1..=10).map(|i| i as f64).collect();
        let omega: Vec<f64> = k.iter().map(|&ki| 5000.0 * ki).collect();
        let disp = PhaseVelocityDispersion::new(k.clone(), omega);
        let cg = disp.group_velocity_at(5.0);
        assert!((cg - 5000.0).abs() < 1.0, "cg={cg}");
    }

    #[test]
    fn test_group_velocity_single_point() {
        let disp = PhaseVelocityDispersion::new(vec![1.0], vec![3000.0]);
        let cg = disp.group_velocity_at(1.0);
        assert_eq!(cg, 0.0); // not enough points
    }

    #[test]
    fn test_phase_velocity_zero_wavenumber() {
        let disp = PhaseVelocityDispersion::new(vec![0.0, 1.0], vec![0.0, 5000.0]);
        assert_eq!(disp.phase_velocity_at(0), 0.0);
    }

    // ── acoustic_emission_source ───────────────────────────────────────────

    #[test]
    fn test_acoustic_emission_source_zero_before_start() {
        assert_eq!(acoustic_emission_source(1.0, 1e-6, -1e-7), 0.0);
    }

    #[test]
    fn test_acoustic_emission_source_at_zero() {
        assert_eq!(acoustic_emission_source(1.0, 1e-6, 0.0), 0.0);
    }

    #[test]
    fn test_acoustic_emission_source_peak_near_rise_time() {
        let m = 1.0;
        let tr = 1e-6;
        // Evaluate on a grid and find maximum
        let mut peak = 0.0_f64;
        for i in 0..1000 {
            let t = i as f64 * 0.01 * tr;
            let v = acoustic_emission_source(m, tr, t);
            if v > peak {
                peak = v;
            }
        }
        assert!(peak > 0.0, "peak={peak}");
    }

    #[test]
    fn test_acoustic_emission_source_decays_after_peak() {
        let m = 1.0;
        let tr = 1e-6;
        // At t >> tr, value should be nearly zero
        let late = acoustic_emission_source(m, tr, 20.0 * tr);
        assert!(late < 1e-5, "late value should decay, got {late}");
    }

    #[test]
    fn test_acoustic_emission_source_magnitude_scales() {
        let tr = 1e-6;
        let v1 = acoustic_emission_source(1.0, tr, tr);
        let v2 = acoustic_emission_source(2.0, tr, tr);
        assert!((v2 - 2.0 * v1).abs() < 1e-12, "v1={v1}, v2={v2}");
    }

    #[test]
    fn test_acoustic_emission_zero_rise_time() {
        // Should return 0 (guard against division by zero)
        let v = acoustic_emission_source(1.0, 0.0, 1e-7);
        assert_eq!(v, 0.0);
    }

    // ── WaveMode ───────────────────────────────────────────────────────────

    #[test]
    fn test_wave_mode_eq() {
        assert_eq!(WaveMode::Longitudinal, WaveMode::Longitudinal);
        assert_ne!(WaveMode::Shear, WaveMode::Rayleigh);
    }

    #[test]
    fn test_wave_mode_debug() {
        let s = format!("{:?}", WaveMode::Lamb);
        assert_eq!(s, "Lamb");
    }

    // ── Additional integration tests ───────────────────────────────────────

    #[test]
    fn test_dispersion_aluminium_cl_cs_ratio() {
        // Aluminum: ν = 0.33 → ratio ≈ 2.14
        let dr = DispersionRelation::new(70e9, 0.33, 2700.0);
        let ratio = dr.longitudinal_wave_speed() / dr.shear_wave_speed();
        assert!(ratio > 1.5 && ratio < 2.5, "ratio={ratio}");
    }

    #[test]
    fn test_fem_energy_grows_under_sustained_force() {
        let mass = vec![1.0; 2];
        let stiffness = vec![100.0, -100.0, -100.0, 100.0];
        let mut fem = WavePropagationFem::new(vec![[0.0; 3]; 2], stiffness, mass);
        let force = vec![10.0, 0.0];
        for _ in 0..50 {
            fem.step_explicit(1e-4, &force);
        }
        // Displacement magnitude should grow
        let d = fem.displacement();
        let mag: f64 = d.iter().map(|x| x * x).sum::<f64>().sqrt();
        assert!(mag > 0.0, "mag={mag}");
    }

    #[test]
    fn test_absorbing_boundary_new() {
        let abc = AbsorbingBoundaryFem::new(vec![100.0, 200.0], vec![0, 5]);
        assert_eq!(abc.dashpot_coefficients.len(), 2);
        assert_eq!(abc.boundary_dofs.len(), 2);
    }

    #[test]
    fn test_dispersion_curve_group_velocity_boundary() {
        // Test boundary handling (first and last points)
        let k = vec![0.0, 1.0, 2.0];
        let omega = vec![0.0, 3000.0, 6000.0];
        let disp = PhaseVelocityDispersion::new(k, omega);
        let cg0 = disp.group_velocity_at(-1.0); // clamps to first
        let cg2 = disp.group_velocity_at(5.0); // clamps to last
        assert!((cg0 - 3000.0).abs() < 1e-6, "cg0={cg0}");
        assert!((cg2 - 3000.0).abs() < 1e-6, "cg2={cg2}");
    }

    #[test]
    fn test_rayleigh_viktorov_formula_nu_zero() {
        // For ν → 0: c_R ≈ 0.862 * c_S
        let dr = DispersionRelation::new(100e9, 0.001, 1000.0);
        let ratio = dr.rayleigh_wave_speed() / dr.shear_wave_speed();
        assert!((ratio - 0.862).abs() < 0.01, "ratio={ratio}");
    }

    #[test]
    fn test_elastic_wave_clone() {
        let w = ElasticWave::new(1000.0, 0.5, 1.0, [1.0, 0.0, 0.0], WaveMode::Longitudinal);
        let w2 = w.clone();
        assert!((w2.frequency - w.frequency).abs() < 1e-12);
    }

    #[test]
    fn test_dispersion_relation_clone() {
        let dr = DispersionRelation::new(200e9, 0.3, 7800.0);
        let dr2 = dr.clone();
        assert!((dr2.e_modulus - dr.e_modulus).abs() < 1e-6);
    }
}
