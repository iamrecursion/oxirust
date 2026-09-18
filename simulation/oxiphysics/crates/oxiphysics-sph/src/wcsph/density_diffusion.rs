// Copyright 2026 COOLJAPAN OU (Team KitaSan)
// SPDX-License-Identifier: Apache-2.0

//! δ-SPH density diffusion for weakly compressible SPH.
//!
//! The continuity equation of WCSPH is augmented by a numerical diffusion term
//! that damps the spurious high-frequency density (and hence pressure) noise
//! of the weakly compressible formulation (Molteni & Colagrossi 2009):
//!
//! ```text
//! Dρ_i/Dt|_diff = δ h c_s Σ_j (m_j/ρ_j) ψ_ij · ∇_i W_ij
//! ψ_ij          = 2 (ρ_j − ρ_i) (x_j − x_i) / |x_j − x_i|²
//! ```
//!
//! Because `∇_i W_ij = (dW/dr) (x_i − x_j)/|x_i − x_j|` points *against*
//! `x_j − x_i` (the kernel decreases outwards, `dW/dr < 0`), the scalar product is
//! `ψ_ij · ∇_i W_ij = −2 (ρ_j − ρ_i) (dW/dr) / r`: **positive for a denser
//! neighbour**, so density diffuses from high to low values.
//!
//! In a hydrostatic column the density genuinely varies with depth, and the
//! plain term above would diffuse that stratification away.  Fourtakas et al.
//! (2019) remove the hydrostatic part of the density difference,
//!
//! ```text
//! ψ_ij  = 2 [(ρ_j − ρ_i) − ρ^H_ij] (x_j − x_i) / |x_j − x_i|²
//! ρ^H_ij = ρ₀ [ (1 + ρ₀ g·(x_j − x_i) / C_B)^(1/γ) − 1 ]
//! ```
//!
//! with the Tait coefficient `C_B = ρ₀ c₀² / γ` and the gravity vector `g`, so
//! that an exactly hydrostatic fluid at rest is left untouched (see
//! [`HydrostaticReference`]).

use crate::kernel::SphKernel;
use crate::parallel::{Execution, collect_per_particle};
use crate::particle::ParticleSet;
use oxiphysics_core::math::Vec3;

use super::types::WcsphParams;

/// Hydrostatic reference state for the Fourtakas et al. (2019) correction of the
/// δ-SPH density diffusion.
#[derive(Debug, Clone, Copy)]
pub struct HydrostaticReference {
    /// Rest density `ρ₀` of the fluid (kg/m³).
    pub rest_density: f64,
    /// Tait coefficient `C_B = ρ₀ c₀² / γ` (Pa), i.e. `p = C_B ((ρ/ρ₀)^γ − 1)`.
    pub tait_coefficient: f64,
    /// Tait exponent `γ`.
    pub gamma: f64,
    /// Gravitational acceleration `g` (m/s²); it points *down*.
    pub gravity: Vec3,
}

impl HydrostaticReference {
    /// Create a hydrostatic reference from its raw ingredients.
    pub fn new(rest_density: f64, tait_coefficient: f64, gamma: f64, gravity: Vec3) -> Self {
        Self {
            rest_density,
            tait_coefficient,
            gamma,
            gravity,
        }
    }

    /// Create the reference matching the equation of state of `params`.
    ///
    /// `params.stiffness` is `K = ρ₀ c₀²` and the pressure is
    /// `(K/γ) ((ρ/ρ₀)^γ − 1)` (see [`WcsphParams`]), so `C_B = K / γ`.
    pub fn from_params(params: &WcsphParams, gravity: Vec3) -> Self {
        Self::new(
            params.rest_density,
            params.stiffness / params.gamma,
            params.gamma,
            gravity,
        )
    }

    /// Hydrostatic density difference `ρ^H_ij` of a pair whose separation is
    /// `x_ji = x_j − x_i`.
    ///
    /// It is positive when `j` lies below `i` (along `g`) and negative when `j`
    /// lies above, and vanishes for a horizontal separation.  The expression is
    /// meant for separations well below `C_B / (ρ₀ |g|)` (the pressure scale
    /// height); the base of the power is floored at the smallest positive
    /// normal number so that the result stays finite beyond that.
    pub fn density_difference(&self, x_ji: Vec3) -> f64 {
        let base = 1.0 + self.rest_density * self.gravity.dot(&x_ji) / self.tait_coefficient;
        self.rest_density * (base.max(f64::MIN_POSITIVE).powf(1.0 / self.gamma) - 1.0)
    }
}

/// Rate `Dρ_i/Dt|_diff` of the δ-SPH density diffusion for particle `i`,
/// optionally with the hydrostatic correction.
pub(super) fn delta_sph_diffusion_rate(
    i: usize,
    positions: &[Vec3],
    masses: &[f64],
    densities: &[f64],
    neighbors: &[Vec<usize>],
    kernel: &impl SphKernel,
    h: f64,
    delta: f64,
    c_s: f64,
    hydrostatic: Option<&HydrostaticReference>,
) -> f64 {
    let xi = positions[i];
    let rho_i = densities[i];
    let mut diffusion = 0.0_f64;
    for &j in &neighbors[i] {
        // x_ij = x_i − x_j, so x_j − x_i = −x_ij.
        let xij = xi - positions[j];
        let r = xij.norm();
        if r < f64::EPSILON {
            continue;
        }
        let rho_j = densities[j];
        let dw_dr = kernel.grad_w(r, h);
        let mut density_jump = rho_j - rho_i;
        if let Some(reference) = hydrostatic {
            density_jump -= reference.density_difference(-xij);
        }
        // ψ_ij · ∇_i W_ij = −2 [(ρ_j − ρ_i) − ρ^H_ij] (dW/dr) / r
        let psi_dot_grad_w = -2.0 * density_jump * dw_dr / r;
        let vj = masses[j] / rho_j.max(f64::EPSILON);
        diffusion += vj * psi_dot_grad_w;
    }
    delta * h * c_s * diffusion
}

/// Advance every density by `dt` times its δ-SPH diffusion rate (never letting a
/// density drop below `f64::EPSILON`).
///
/// The rate of a particle is gathered from its own neighbours, so with the
/// `parallel` feature the rates are computed on the rayon pool (bitwise the same
/// as serially) before the densities are advanced.
pub(super) fn apply_delta_sph_diffusion(
    execution: Execution,
    ps: &mut ParticleSet,
    neighbors: &[Vec<usize>],
    kernel: &impl SphKernel,
    h: f64,
    delta: f64,
    c_s: f64,
    dt: f64,
    hydrostatic: Option<&HydrostaticReference>,
) {
    let rates: Vec<f64> = collect_per_particle(execution, ps.positions.len(), |i| {
        delta_sph_diffusion_rate(
            i,
            &ps.positions,
            &ps.masses,
            &ps.densities,
            neighbors,
            kernel,
            h,
            delta,
            c_s,
            hydrostatic,
        )
    });
    for (rho, &rate) in ps.densities.iter_mut().zip(rates.iter()) {
        *rho += rate * dt;
        if *rho < f64::EPSILON {
            *rho = f64::EPSILON;
        }
    }
}

/// δ-SPH density diffusion with the hydrostatic correction of Fourtakas et al.
/// (2019) for particle `i`.
///
/// ```text
/// Dρ_i/Dt|_diff = δ h c_s Σ_j (m_j/ρ_j) ψ_ij · ∇_i W_ij
/// ψ_ij          = 2 [(ρ_j − ρ_i) − ρ^H_ij] (x_j − x_i) / |x_j − x_i|²
/// ```
///
/// where `ρ^H_ij` is [`HydrostaticReference::density_difference`] of the pair.
/// For an exactly hydrostatic fluid the bracket vanishes and the term leaves
/// the stratification alone, while it still damps non-hydrostatic density noise.
/// Without gravity it reduces to
/// [`fourtakas_density_diffusion`](super::functions::fourtakas_density_diffusion).
///
/// # Arguments
/// * `delta` – diffusion coefficient δ (typically 0.1)
/// * `h` – smoothing length
/// * `c_s` – speed of sound
/// * `reference` – hydrostatic reference state
pub fn fourtakas_hydrostatic_density_diffusion(
    i: usize,
    positions: &[Vec3],
    masses: &[f64],
    densities: &[f64],
    neighbors: &[Vec<usize>],
    kernel: &impl SphKernel,
    h: f64,
    delta: f64,
    c_s: f64,
    reference: &HydrostaticReference,
) -> f64 {
    delta_sph_diffusion_rate(
        i,
        positions,
        masses,
        densities,
        neighbors,
        kernel,
        h,
        delta,
        c_s,
        Some(reference),
    )
}

/// Apply the hydrostatic-corrected δ-SPH density diffusion
/// ([`fourtakas_hydrostatic_density_diffusion`]) to all particles over a time
/// step `dt`.
pub fn apply_fourtakas_hydrostatic_diffusion(
    ps: &mut ParticleSet,
    neighbors: &[Vec<usize>],
    kernel: &impl SphKernel,
    h: f64,
    delta: f64,
    c_s: f64,
    dt: f64,
    reference: &HydrostaticReference,
) {
    let execution = Execution::for_count(ps.positions.len());
    apply_delta_sph_diffusion(
        execution,
        ps,
        neighbors,
        kernel,
        h,
        delta,
        c_s,
        dt,
        Some(reference),
    );
}
