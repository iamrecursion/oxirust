// Copyright 2026 COOLJAPAN OU (Team KitaSan)
// SPDX-License-Identifier: Apache-2.0

//! Per-particle gather kernels of the [`ParticleSet`] WCSPH pipeline.
//!
//! Every sum of the pipeline (density summation, pressure, viscosity, XSPH and
//! tensile forces, Shepard/MLS corrections, particle shifting) is a *gather*: the
//! value of particle `i` is computed from the neighbours of `i` alone, in the
//! order of its neighbour list, and nothing is shared between particles.  The
//! kernels here compute one value per particle through
//! [`collect_per_particle`] and write the values back by particle index, so with
//! the `parallel` feature the particles are spread over the rayon pool while each
//! sum is formed exactly as in the serial loop.  The results are bitwise
//! identical to the serial evaluation for every thread count; no reduction order
//! is involved.
//!
//! The public functions of [`functions`](super::functions) call these kernels
//! with [`Execution::for_count`]; the explicit [`Execution`] argument exists so
//! that the tests can run the same kernel serially and in parallel.

use oxiphysics_core::math::Vec3;

use super::functions::{
    density_summation_shepard, mls_kernel_correction_factor, riemann_interface_pressure,
    speed_of_sound_tait,
};
use super::types::WcsphParams;
use crate::kernel::SphKernel;
use crate::parallel::{Execution, collect_per_particle};
use crate::particle::ParticleSet;

/// Density summation `ρ_i = m_i W(0) + Σ_j m_j W(r_ij)` for the first
/// `min(N, neighbors.len())` particles.
pub(super) fn density_summation(
    execution: Execution,
    particles: &mut ParticleSet,
    neighbors: &[Vec<usize>],
    kernel: &dyn SphKernel,
    h: f64,
) {
    let count = particles.len().min(neighbors.len());
    let densities = collect_per_particle(execution, count, |i| {
        let mut rho = particles.masses[i] * kernel.w(0.0, h);
        for &j in &neighbors[i] {
            let r = (particles.positions[i] - particles.positions[j]).norm();
            rho += particles.masses[j] * kernel.w(r, h);
        }
        rho
    });
    for (slot, rho) in particles.densities.iter_mut().zip(densities) {
        *slot = rho;
    }
}

/// Shepard-normalised density from the previous densities (see
/// [`compute_density_shepard`](super::functions::compute_density_shepard)).
pub(super) fn density_shepard(
    execution: Execution,
    particles: &mut ParticleSet,
    neighbors: &[Vec<usize>],
    kernel: &dyn SphKernel,
    h: f64,
) {
    let n = particles.len();
    let old_densities = particles.densities.clone();
    let corrected = collect_per_particle(execution, n, |i| {
        let w_self = kernel.w(0.0, h);
        let mut numerator = particles.masses[i] * w_self;
        let rho_self = old_densities[i].max(1e-14);
        let mut denominator = (particles.masses[i] / rho_self) * w_self;
        for &j in &neighbors[i] {
            let r = (particles.positions[i] - particles.positions[j]).norm();
            let w = kernel.w(r, h);
            numerator += particles.masses[j] * w;
            let rho_j = old_densities[j].max(1e-14);
            denominator += (particles.masses[j] / rho_j) * w;
        }
        if denominator > 1e-14 {
            numerator / denominator
        } else {
            numerator
        }
    });
    for (slot, rho) in particles.densities.iter_mut().zip(corrected) {
        *slot = rho;
    }
}

/// Pressure forces `ρ_i · (−Σ_j m_j (p_i/ρ_i² + p_j/ρ_j²) ∇W_ij)` accumulated
/// into `particles.forces`.
pub(super) fn pressure_force(
    execution: Execution,
    particles: &mut ParticleSet,
    neighbors: &[Vec<usize>],
    kernel: &dyn SphKernel,
    h: f64,
) {
    let n = particles.len();
    let pressure_forces = collect_per_particle(execution, n, |i| {
        let mut force = Vec3::zeros();
        let pi = particles.pressures[i];
        let rhoi = particles.densities[i];
        if rhoi < 1e-14 {
            return force;
        }
        for &j in &neighbors[i] {
            let rhoj = particles.densities[j];
            if rhoj < 1e-14 {
                continue;
            }
            let pj = particles.pressures[j];
            let rij = particles.positions[i] - particles.positions[j];
            let r = rij.norm();
            if r < 1e-14 {
                continue;
            }
            let grad = kernel.grad_w(r, h);
            let rhat = rij / r;
            let factor = -particles.masses[j] * (pi / (rhoi * rhoi) + pj / (rhoj * rhoj)) * grad;
            force += factor * rhat;
        }
        force
    });
    for (i, &pf) in pressure_forces.iter().enumerate().take(n) {
        particles.forces[i] += pf * particles.densities[i];
    }
}

/// Monaghan artificial viscosity forces accumulated into `particles.forces`.
pub(super) fn artificial_viscosity(
    execution: Execution,
    particles: &mut ParticleSet,
    neighbors: &[Vec<usize>],
    kernel: &dyn SphKernel,
    h: f64,
    alpha: f64,
    beta: f64,
    speed_of_sound: f64,
) {
    let n = particles.len();
    let eta2 = 0.01 * h * h;
    let art_visc_forces = collect_per_particle(execution, n, |i| {
        let mut force = Vec3::zeros();
        let rhoi = particles.densities[i];
        if rhoi < 1e-14 {
            return force;
        }
        for &j in &neighbors[i] {
            let rhoj = particles.densities[j];
            if rhoj < 1e-14 {
                continue;
            }
            let rij = particles.positions[i] - particles.positions[j];
            let vij = particles.velocities[i] - particles.velocities[j];
            let r2 = rij.norm_squared();
            let vr = vij.dot(&rij);
            if vr >= 0.0 {
                continue;
            }
            let mu_ij = h * vr / (r2 + eta2);
            let rho_bar = 0.5 * (rhoi + rhoj);
            let pi_ij = (-alpha * speed_of_sound * mu_ij + beta * mu_ij * mu_ij) / rho_bar;
            let r = r2.sqrt();
            if r < 1e-14 {
                continue;
            }
            let grad = kernel.grad_w(r, h);
            let rhat = rij / r;
            force -= particles.masses[j] * pi_ij * grad * rhat;
        }
        force
    });
    for (i, &avf) in art_visc_forces.iter().enumerate().take(n) {
        particles.forces[i] += avf * particles.densities[i];
    }
}

/// Morris et al. (1997) viscous forces accumulated into `particles.forces`;
/// `dynamic_viscosity[i]` is `μ_i` and the pair viscosity is `μ_i + μ_j`.
pub(super) fn morris_viscosity(
    execution: Execution,
    particles: &mut ParticleSet,
    neighbors: &[Vec<usize>],
    kernel: &dyn SphKernel,
    h: f64,
    dynamic_viscosity: &[f64],
) {
    let n = particles.len();
    let eta2 = 0.01 * h * h;
    let visc_forces = collect_per_particle(execution, n, |i| {
        let mut force = Vec3::zeros();
        let rhoi = particles.densities[i];
        if rhoi < 1e-14 {
            return force;
        }
        for &j in &neighbors[i] {
            let rhoj = particles.densities[j];
            if rhoj < 1e-14 {
                continue;
            }
            let rij = particles.positions[i] - particles.positions[j];
            let vij = particles.velocities[i] - particles.velocities[j];
            let r2 = rij.norm_squared();
            let r = r2.sqrt();
            if r < 1e-14 {
                continue;
            }
            let grad = kernel.grad_w(r, h);
            let r_dot_grad_w = r * grad;
            let mu_pair = dynamic_viscosity[i] + dynamic_viscosity[j];
            let factor = particles.masses[j] * mu_pair / (rhoi * rhoj) * r_dot_grad_w / (r2 + eta2);
            force += factor * vij;
        }
        force
    });
    for (i, &vf) in visc_forces.iter().enumerate().take(n) {
        particles.forces[i] += vf * particles.densities[i];
    }
}

/// XSPH-corrected velocities `v_i + ε Σ_j (m_j/ρ_j)(v_j − v_i) W_ij`; particles
/// of vanishing density keep their velocity.
pub(super) fn xsph_velocities(
    execution: Execution,
    particles: &ParticleSet,
    neighbors: &[Vec<usize>],
    kernel: &dyn SphKernel,
    h: f64,
    epsilon: f64,
) -> Vec<Vec3> {
    let n = particles.len();
    let deltas = collect_per_particle(execution, n, |i| {
        let rhoi = particles.densities[i];
        if rhoi < 1e-14 {
            return None;
        }
        let mut delta = Vec3::zeros();
        for &j in &neighbors[i] {
            let rhoj = particles.densities[j];
            if rhoj < 1e-14 {
                continue;
            }
            let rij = particles.positions[i] - particles.positions[j];
            let r = rij.norm();
            let w = kernel.w(r, h);
            let vji = particles.velocities[j] - particles.velocities[i];
            delta += (particles.masses[j] / rhoj) * w * vji;
        }
        Some(delta)
    });
    let mut corrected = particles.velocities.clone();
    for (velocity, delta) in corrected.iter_mut().zip(deltas) {
        if let Some(delta) = delta {
            *velocity += epsilon * delta;
        }
    }
    corrected
}

/// Per-particle tensile-instability correction forces (Monaghan 2000), see
/// [`compute_tensile_instability_correction`](super::functions::compute_tensile_instability_correction).
pub(super) fn tensile_instability(
    execution: Execution,
    particles: &ParticleSet,
    neighbors: &[Vec<usize>],
    kernel: &dyn SphKernel,
    h: f64,
    delta_p: f64,
    tensile_coeff: f64,
    tensile_exp: i32,
) -> Vec<Vec3> {
    let n = particles.len();
    let w_delta = kernel.w(delta_p, h);
    if w_delta.abs() < 1e-30 {
        return vec![Vec3::zeros(); n];
    }
    collect_per_particle(execution, n, |i| {
        let mut correction = Vec3::zeros();
        let rhoi = particles.densities[i];
        if rhoi < 1e-14 {
            return correction;
        }
        let pi = particles.pressures[i];
        for &j in &neighbors[i] {
            let rhoj = particles.densities[j];
            if rhoj < 1e-14 {
                continue;
            }
            let pj = particles.pressures[j];
            let rij = particles.positions[i] - particles.positions[j];
            let r = rij.norm();
            if r < 1e-14 {
                continue;
            }
            let ri = if pi < 0.0 {
                tensile_coeff * (-pi) / (rhoi * rhoi)
            } else {
                0.0
            };
            let rj = if pj < 0.0 {
                tensile_coeff * (-pj) / (rhoj * rhoj)
            } else {
                0.0
            };
            if ri + rj < 1e-30 {
                continue;
            }
            let w_ratio = kernel.w(r, h) / w_delta;
            let f_n = w_ratio.powi(tensile_exp);
            let grad = kernel.grad_w(r, h);
            let rhat = rij / r;
            correction -= particles.masses[j] * (ri + rj) * f_n * grad * rhat;
        }
        correction
    })
}

/// Shepard-corrected density of every particle (see
/// [`density_summation_shepard`]), computed from the current densities.
pub(super) fn shepard_corrected_densities<K: SphKernel>(
    execution: Execution,
    particles: &ParticleSet,
    neighbors: &[Vec<usize>],
    kernel: &K,
    h: f64,
) -> Vec<f64> {
    collect_per_particle(execution, particles.positions.len(), |i| {
        density_summation_shepard(
            i,
            &particles.positions,
            &particles.masses,
            &particles.densities,
            neighbors,
            kernel,
            h,
        )
    })
}

/// MLS (Shepard) kernel-correction factor of every particle (see
/// [`mls_kernel_correction_factor`]).
pub(super) fn mls_correction_factors<K: SphKernel>(
    execution: Execution,
    particles: &ParticleSet,
    neighbors: &[Vec<usize>],
    kernel: &K,
    h: f64,
) -> Vec<f64> {
    collect_per_particle(execution, particles.positions.len(), |i| {
        mls_kernel_correction_factor(
            i,
            &particles.positions,
            &particles.masses,
            &particles.densities,
            neighbors,
            kernel,
            h,
        )
    })
}

/// Riemann-solver pressure forces (Parshikov & Medin 2002) accumulated into
/// `particles.forces`; see
/// [`compute_riemann_pressure_force`](super::functions::compute_riemann_pressure_force).
pub(super) fn riemann_pressure_force(
    execution: Execution,
    particles: &mut ParticleSet,
    neighbors: &[Vec<usize>],
    kernel: &dyn SphKernel,
    h: f64,
    params: &WcsphParams,
) {
    let n = particles.len();
    // Acoustic impedance Z = ρ c of every particle, with c from the EOS.
    let impedances: Vec<f64> = particles
        .densities
        .iter()
        .map(|&rho| rho * speed_of_sound_tait(rho.max(1e-14), params))
        .collect();
    let p_forces = collect_per_particle(execution, n, |i| {
        let mut force = Vec3::zeros();
        let rho_i = particles.densities[i];
        let p_i = particles.pressures[i];
        if rho_i < 1e-14 {
            return force;
        }
        for &j in &neighbors[i] {
            let rho_j = particles.densities[j];
            let p_j = particles.pressures[j];
            if rho_j < 1e-14 {
                continue;
            }
            let rij = particles.positions[i] - particles.positions[j];
            let r = rij.norm();
            if r < 1e-14 {
                continue;
            }
            // rhat points from j to i, so e_ij = -rhat and
            // (v_i - v_j) . e_ij = -(v_i - v_j) . rhat.
            let rhat = rij / r;
            let vij = particles.velocities[i] - particles.velocities[j];
            let approach_speed = -vij.dot(&rhat);
            let p_star =
                riemann_interface_pressure(p_i, p_j, impedances[i], impedances[j], approach_speed);
            let grad = kernel.grad_w(r, h);
            let factor = -2.0 * particles.masses[j] * p_star / (rho_i * rho_j) * grad;
            force += factor * rhat;
        }
        force
    });
    for (i, &pf) in p_forces.iter().enumerate().take(n) {
        particles.forces[i] += pf * particles.densities[i];
    }
}

/// Fickian particle-shifting displacement `−A h² Σ_j (m_j/ρ_j) ∇W_ij` of every
/// particle.
pub(super) fn particle_shifts(
    execution: Execution,
    particles: &ParticleSet,
    neighbors: &[Vec<usize>],
    kernel: &dyn SphKernel,
    h: f64,
    coefficient: f64,
) -> Vec<Vec3> {
    collect_per_particle(execution, particles.len(), |i| {
        let rhoi = particles.densities[i];
        if rhoi < 1e-14 {
            return Vec3::zeros();
        }
        let mut sum = Vec3::zeros();
        for &j in &neighbors[i] {
            let rhoj = particles.densities[j];
            if rhoj < 1e-14 {
                continue;
            }
            let rij = particles.positions[i] - particles.positions[j];
            let r = rij.norm();
            if r < 1e-14 {
                continue;
            }
            let grad = kernel.grad_w(r, h);
            let rhat = rij / r;
            sum += (particles.masses[j] / rhoj) * grad * rhat;
        }
        -coefficient * h * h * sum
    })
}
