// Copyright 2026 COOLJAPAN OU (Team KitaSan)
// SPDX-License-Identifier: Apache-2.0

//! Electro-osmotic flow (EOF) in a slit microchannel with a D2Q9 lattice
//! Boltzmann solver.
//!
//! The channel is bounded by two walls at `y = 0` and `y = H = ny` (lattice
//! units, nodes at `y_j = j + 1/2`) and is homogeneous along `x`.  A
//! tangential electric field `E_x` acts on the net charge of the electric
//! double layer (EDL), which is described by the linearised Poisson–Boltzmann
//! (Debye–Hückel) potential
//!
//! ```text
//! φ(y) = ζ cosh((H/2 − y)/λ_D) / cosh(H/(2 λ_D)),      ∇²φ = φ / λ_D²,
//! ```
//!
//! so that the charge density is `ρ_e = −ε_r ε₀ ∇²φ` and the body force is
//! `ρ_e E_x`.  The Stokes balance `μ u'' + ρ_e E_x = 0` with no-slip walls
//! (where `φ = ζ`) has the solution
//!
//! ```text
//! u(y) = u_HS (1 − φ(y)/ζ),     u_HS = −ε_r ε₀ ζ E_x / μ,
//! ```
//!
//! which tends to the Helmholtz–Smoluchowski velocity `u_HS` outside the EDL.
//!
//! The lattice solver streams the populations along `y` with half-way
//! bounce-back at both walls and applies the force with the Guo scheme.
//!
//! Because `ε_r ε₀ E_x = −μ u_HS / ζ`, the force density is
//! `ρ_e E_x = μ u_HS ∇²φ̂` with the normalised potential `φ̂ = φ/ζ`.  The
//! physical-to-lattice velocity scale `Δt/Δx` is a free choice of the user and
//! enters through the lattice Helmholtz–Smoluchowski velocity `u_hs_lu`; in
//! lattice units with `ρ₀ = 1` (so `μ = ν`) the force density is
//!
//! ```text
//! F_j = ν_lu · u_hs_lu · (∇²φ̂)_j,        ν_lu = (1/ω − 1/2)/3,
//! ```
//!
//! where `(∇²φ̂)_j` is the discrete Gauss-law charge of cell `j` (the flux
//! difference of `φ̂` across the cell, with the Dirichlet wall value `φ̂ = 1` at
//! the half-way walls).  With this force the profile `u_j = u_hs_lu (1 − φ̂_j)`
//! solves the discrete Stokes balance, including the half-way bounce-back wall
//! closure, exactly; the steady BGK flow therefore reproduces the continuum
//! profile up to the viscosity-dependent bounce-back slip, even when the double
//! layer is only a couple of nodes thick.

use super::{CS2, D2Q9_C, D2Q9_W, EPSILON_0};

/// Opposite-direction index of every D2Q9 velocity (`c_opp = −c`).
const D2Q9_OPP: [usize; 9] = [0, 3, 4, 1, 2, 7, 8, 5, 6];

/// Magnitude of the lattice Helmholtz–Smoluchowski velocity chosen by
/// [`ElectroosmosisLBM::new`] (Mach number about 0.035, safely incompressible).
pub const DEFAULT_HS_VELOCITY_LU: f64 = 0.02;

/// Normalised Debye–Hückel potential `φ̂(y) = φ(y)/ζ` at height `y` in a channel
/// of height `h` (all in lattice units).
///
/// Evaluated as a ratio of hyperbolic cosines in overflow-free form.  A
/// non-positive or non-finite Debye length means "no double layer" and yields 0.
fn normalised_potential(y: f64, h: f64, debye_lu: f64) -> f64 {
    if !debye_lu.is_finite() || debye_lu <= 0.0 {
        return 0.0;
    }
    let a = ((h / 2.0 - y) / debye_lu).abs();
    let b = h / (2.0 * debye_lu);
    (a - b).exp() * (1.0 + (-2.0 * a).exp()) / (1.0 + (-2.0 * b).exp())
}

/// Discrete Gauss-law charge `(∇²φ̂)_j` of every node (cell) of the channel.
///
/// The flux `dφ̂/dy` is evaluated on the `n + 1` cell faces: between two nodes
/// it is the usual difference quotient, and at the two walls it is the
/// one-sided quotient between the wall value `φ̂ = 1` (Dirichlet, `φ = ζ`) and
/// the adjacent node, half a lattice spacing away.  The cell charge is the flux
/// difference across the cell.
fn gauss_charge_profile(phi_hat: &[f64]) -> Vec<f64> {
    let n = phi_hat.len();
    let flux = |k: usize| -> f64 {
        if k == 0 {
            2.0 * (phi_hat[0] - 1.0)
        } else if k == n {
            2.0 * (1.0 - phi_hat[n - 1])
        } else {
            phi_hat[k] - phi_hat[k - 1]
        }
    };
    (0..n).map(|j| flux(j + 1) - flux(j)).collect()
}

/// D2Q9 equilibrium population `i` for density `rho` and velocity `(ux, uy)`.
fn equilibrium(rho: f64, ux: f64, uy: f64, i: usize) -> f64 {
    let cu = D2Q9_C[i][0] * ux + D2Q9_C[i][1] * uy;
    let u2 = ux * ux + uy * uy;
    D2Q9_W[i] * rho * (1.0 + cu / CS2 + cu * cu / (2.0 * CS2 * CS2) - u2 / (2.0 * CS2))
}

/// Density and Guo-corrected velocity `(ρ, u_x, u_y)` of one node for the
/// body force `force_x` along `x`: `ρ u = Σ_i c_i f_i + F/2`.
fn node_moments(f: &[f64; 9], force_x: f64) -> (f64, f64, f64) {
    let rho = f.iter().sum::<f64>().max(1e-10);
    let mut jx = 0.0;
    let mut jy = 0.0;
    for (fi, c) in f.iter().zip(D2Q9_C.iter()) {
        jx += c[0] * fi;
        jy += c[1] * fi;
    }
    (rho, (jx + 0.5 * force_x) / rho, jy / rho)
}

/// Electroosmotic flow simulation state for a slit channel (1-D across `y`).
///
/// The channel has walls at `y = 0` and `y = H = ny` (lattice units, nodes at
/// `y_j = j + 1/2`) and is homogeneous along `x`.  The electric double layer is
/// the Debye–Hückel potential
/// `φ(y) = ζ cosh((H/2 − y)/λ_D) / cosh(H/(2 λ_D))`, and the field `E_x` acts on
/// its charge `ρ_e = −ε_r ε₀ ∇²φ`.  With no-slip walls the Stokes solution is
/// `u(y) = u_HS (1 − φ(y)/ζ)` with the Helmholtz–Smoluchowski velocity
/// `u_HS = −ε_r ε₀ ζ E_x / μ`.
///
/// The lattice solver streams the populations along `y` with half-way
/// bounce-back at both walls and applies the force with the Guo scheme.  Its
/// dynamics are driven by `u_hs_lu`, the lattice Helmholtz–Smoluchowski
/// velocity; the SI parameters `eps_r`, `zeta`, `e_field` and `mu` give its
/// direction and its physical value via [`ElectroosmosisLBM::hs_velocity`].
/// The force density is `F_j = ν_lu · u_hs_lu · (∇²φ̂)_j` with `φ̂ = φ/ζ` (see
/// [`ElectroosmosisLBM::eo_force`]).
#[derive(Debug, Clone)]
pub struct ElectroosmosisLBM {
    /// Number of lattice nodes across the channel (channel height in lattice units).
    pub ny: usize,
    /// Relative permittivity of the electrolyte.
    pub eps_r: f64,
    /// Zeta potential at the walls (V).
    pub zeta: f64,
    /// Applied external electric field E_x (V m⁻¹).
    pub e_field: f64,
    /// Dynamic viscosity (Pa·s).
    pub mu: f64,
    /// Debye length λ_D in lattice units (node spacings).
    pub debye_lu: f64,
    /// Helmholtz–Smoluchowski velocity in lattice units: the steady centre
    /// velocity of a channel much wider than the Debye length.
    pub u_hs_lu: f64,
    /// Debye–Hückel electric potential at each node (V).
    pub phi: Vec<f64>,
    /// Velocity at each node (lattice units), updated by [`ElectroosmosisLBM::step`].
    pub u: Vec<f64>,
    /// Distribution functions f\[node\]\[direction\].
    pub f: Vec<[f64; 9]>,
    /// Discrete Gauss-law charge `(∇²φ̂)_j` per node, derived from the potential.
    charge_profile: Vec<f64>,
}

impl ElectroosmosisLBM {
    /// Create a new `ElectroosmosisLBM` simulation at rest (`ρ = 1`, `u = 0`).
    ///
    /// `ny` — nodes across channel, `eps_r` — relative permittivity,
    /// `zeta` — wall zeta potential (V), `e_field` — applied field (V m⁻¹),
    /// `mu` — viscosity (Pa·s), `debye_lu` — Debye length in lattice units.
    ///
    /// The lattice Helmholtz–Smoluchowski velocity `u_hs_lu` is set to
    /// [`DEFAULT_HS_VELOCITY_LU`] with the sign of the physical
    /// [`hs_velocity`](Self::hs_velocity) (zero if that vanishes); use
    /// [`with_lattice_hs_velocity`](Self::with_lattice_hs_velocity) or
    /// [`with_lattice_scales`](Self::with_lattice_scales) to choose it explicitly.
    pub fn new(ny: usize, eps_r: f64, zeta: f64, e_field: f64, mu: f64, debye_lu: f64) -> Self {
        let mut sim = Self {
            ny,
            eps_r,
            zeta,
            e_field,
            mu,
            debye_lu,
            u_hs_lu: 0.0,
            phi: Vec::new(),
            u: vec![0.0; ny],
            f: vec![[0.0; 9]; ny],
            charge_profile: Vec::new(),
        };
        let u_hs = sim.hs_velocity();
        sim.u_hs_lu = if u_hs > 0.0 {
            DEFAULT_HS_VELOCITY_LU
        } else if u_hs < 0.0 {
            -DEFAULT_HS_VELOCITY_LU
        } else {
            0.0
        };
        sim.rebuild_potential();
        sim.init_equilibrium();
        sim
    }

    /// Set the lattice Helmholtz–Smoluchowski velocity explicitly (builder style).
    pub fn with_lattice_hs_velocity(mut self, u_hs_lu: f64) -> Self {
        self.u_hs_lu = u_hs_lu;
        self
    }

    /// Choose the lattice velocity from the physical one for the lattice spacing
    /// `dx` (m) and time step `dt` (s): `u_hs_lu = u_HS Δt / Δx` (builder style).
    pub fn with_lattice_scales(mut self, dx: f64, dt: f64) -> Self {
        self.u_hs_lu = self.hs_velocity() * dt / dx;
        self
    }

    /// Recompute the Debye–Hückel potential `phi` and the Gauss-law charge
    /// profile from `ny`, `zeta` and `debye_lu`.
    ///
    /// Call this after changing one of those public fields.
    pub fn rebuild_potential(&mut self) {
        let h = self.ny as f64;
        let phi_hat: Vec<f64> = (0..self.ny)
            .map(|j| normalised_potential(j as f64 + 0.5, h, self.debye_lu))
            .collect();
        self.phi = phi_hat.iter().map(|p| self.zeta * p).collect();
        self.charge_profile = gauss_charge_profile(&phi_hat);
    }

    /// Electro-osmotic body-force density at node `j` in lattice units
    /// (`ρ₀ = 1`) for the BGK relaxation frequency `omega`.
    ///
    /// `F_j = ν_lu · u_hs_lu · (∇²φ̂)_j` with `ν_lu = (1/ω − 1/2)/3`: the force
    /// density `ρ_e E_x` of the Debye–Hückel charge, expressed through the
    /// lattice Helmholtz–Smoluchowski velocity.  Returns 0 for `j >= ny`.
    pub fn eo_force(&self, j: usize, omega: f64) -> f64 {
        let nu_lu = CS2 * (1.0 / omega - 0.5);
        self.charge_profile
            .get(j)
            .map_or(0.0, |&charge| nu_lu * self.u_hs_lu * charge)
    }

    /// Run one BGK collision + streaming step with electroosmotic forcing.
    ///
    /// `omega` — relaxation frequency (1/τ), `0 < ω < 2`.  The collision uses
    /// the Guo forcing scheme; streaming moves the populations along `y`
    /// (the flow is homogeneous in `x`) with half-way bounce-back at the walls
    /// `j = 0` and `j = ny − 1`.  Afterwards `u` holds the Guo-corrected
    /// velocity of the new time level.
    pub fn step(&mut self, omega: f64) {
        let ny = self.ny;
        let mut post = vec![[0.0_f64; 9]; ny];
        for (j, post_j) in post.iter_mut().enumerate() {
            let force = self.eo_force(j, omega);
            let (rho, ux, uy) = node_moments(&self.f[j], force);
            for (i, post_i) in post_j.iter_mut().enumerate() {
                let c = D2Q9_C[i];
                let cu = c[0] * ux + c[1] * uy;
                let guo = (1.0 - 0.5 * omega)
                    * D2Q9_W[i]
                    * ((c[0] - ux) / CS2 + cu * c[0] / (CS2 * CS2))
                    * force;
                *post_i = self.f[j][i] - omega * (self.f[j][i] - equilibrium(rho, ux, uy, i)) + guo;
            }
        }
        for (j, node) in self.f.iter_mut().enumerate() {
            for (i, fi) in node.iter_mut().enumerate() {
                // Pull streaming along y; a source outside the channel is a wall,
                // where the population leaving through it returns reversed.
                let source = j as isize - D2Q9_C[i][1] as isize;
                *fi = if (0..ny as isize).contains(&source) {
                    post[source as usize][i]
                } else {
                    post[j][D2Q9_OPP[i]]
                };
            }
        }
        for j in 0..ny {
            let (_, ux, _) = node_moments(&self.f[j], self.eo_force(j, omega));
            self.u[j] = ux;
        }
    }

    /// Analytic Helmholtz-Smoluchowski electro-osmotic velocity (m s⁻¹).
    pub fn hs_velocity(&self) -> f64 {
        -self.eps_r * EPSILON_0 * self.zeta * self.e_field / self.mu
    }

    /// Initialize with equilibrium distributions at rest (`ρ = 1`, `u = 0`).
    pub fn init_equilibrium(&mut self) {
        for node in self.f.iter_mut() {
            for (i, fi) in node.iter_mut().enumerate() {
                *fi = equilibrium(1.0, 0.0, 0.0, i);
            }
        }
        for v in self.u.iter_mut() {
            *v = 0.0;
        }
    }

    /// Continuum Debye–Hückel EOF profile `u_hs_lu (1 − φ̂(y_j))` at node `j`
    /// (lattice units) — the exact Stokes solution to compare the lattice
    /// result against.
    pub fn analytic_velocity(&self, j: usize) -> f64 {
        let y = j as f64 + 0.5;
        self.u_hs_lu * (1.0 - normalised_potential(y, self.ny as f64, self.debye_lu))
    }

    /// Velocity at the channel centre (lattice units): the middle node for odd
    /// `ny`, the mean of the two central nodes for even `ny`.
    pub fn centre_velocity(&self) -> f64 {
        let n = self.ny;
        if n == 0 {
            0.0
        } else if n % 2 == 1 {
            self.u[n / 2]
        } else {
            0.5 * (self.u[n / 2 - 1] + self.u[n / 2])
        }
    }

    /// Total mass `Σ_j Σ_i f_{j,i}` per unit channel length (conserved by the
    /// bounce-back walls).
    pub fn total_mass(&self) -> f64 {
        self.f.iter().map(|node| node.iter().sum::<f64>()).sum()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Channel used by the finding: 64 nodes, Debye length of two lattice units.
    fn channel(debye_lu: f64) -> ElectroosmosisLBM {
        ElectroosmosisLBM::new(64, 80.0, -0.05, 1e4, 1e-3, debye_lu)
    }

    /// Step until the velocity change per 100 steps drops below `tol`.
    fn run_to_steady_state(sim: &mut ElectroosmosisLBM, omega: f64, max_steps: usize, tol: f64) {
        let mut previous = sim.u.clone();
        for step in 1..=max_steps {
            sim.step(omega);
            if step % 100 == 0 {
                let change = sim
                    .u
                    .iter()
                    .zip(previous.iter())
                    .map(|(a, b)| (a - b).abs())
                    .fold(0.0_f64, f64::max);
                if change < tol {
                    return;
                }
                previous.copy_from_slice(&sim.u);
            }
        }
    }

    #[test]
    fn new_starts_at_rest_equilibrium() {
        let sim = channel(2.0);
        assert!((sim.total_mass() - 64.0).abs() < 1e-12);
        for node in &sim.f {
            let rho: f64 = node.iter().sum();
            let jx: f64 = node.iter().zip(D2Q9_C.iter()).map(|(f, c)| f * c[0]).sum();
            let jy: f64 = node.iter().zip(D2Q9_C.iter()).map(|(f, c)| f * c[1]).sum();
            assert!((rho - 1.0).abs() < 1e-14);
            assert!(jx.abs() < 1e-14 && jy.abs() < 1e-14);
        }
        assert!(sim.u.iter().all(|&v| v == 0.0));
    }

    #[test]
    fn default_lattice_velocity_follows_the_physical_direction() {
        // zeta < 0, E > 0  =>  u_HS > 0
        let positive = ElectroosmosisLBM::new(20, 80.0, -0.05, 1e4, 1e-3, 2.0);
        assert!(positive.hs_velocity() > 0.0);
        assert!((positive.u_hs_lu - DEFAULT_HS_VELOCITY_LU).abs() < 1e-15);
        // zeta > 0, E > 0  =>  u_HS < 0
        let negative = ElectroosmosisLBM::new(20, 80.0, 0.05, 1e4, 1e-3, 2.0);
        assert!((negative.u_hs_lu + DEFAULT_HS_VELOCITY_LU).abs() < 1e-15);
        // no zeta potential: no flow, and no NaN from a 0/0 normalisation
        let neutral = ElectroosmosisLBM::new(20, 80.0, 0.0, 1e4, 1e-3, 2.0);
        assert_eq!(neutral.u_hs_lu, 0.0);
        assert!(neutral.phi.iter().all(|&p| p == 0.0));
    }

    #[test]
    fn lattice_scales_convert_the_physical_velocity() {
        let sim = ElectroosmosisLBM::new(20, 80.0, -0.05, 1e4, 1e-3, 2.0);
        let u_hs = sim.hs_velocity();
        let scaled = sim.with_lattice_scales(1e-6, 2e-3);
        assert!((scaled.u_hs_lu - u_hs * 2e-3 / 1e-6).abs() < 1e-9 * u_hs.abs().max(1.0));
        let explicit = scaled.with_lattice_hs_velocity(0.01);
        assert_eq!(explicit.u_hs_lu, 0.01);
    }

    #[test]
    fn potential_is_debye_hueckel_between_the_walls() {
        let sim = channel(2.0);
        // Symmetric, largest next to the walls, nearly zero at the centre.
        for j in 0..32 {
            assert!((sim.phi[j] - sim.phi[63 - j]).abs() < 1e-15);
            assert!(sim.phi[j].abs() >= sim.phi[j + 1].abs());
        }
        let wall_node = -0.05 * (-0.25_f64).exp(); // zeta * exp(-(1/2)/lambda_D)
        assert!(
            (sim.phi[0] - wall_node).abs() < 1e-6,
            "phi[0] = {}",
            sim.phi[0]
        );
        assert!(sim.phi[31].abs() < 1e-6 * 0.05);
    }

    #[test]
    fn force_profile_obeys_the_discrete_gauss_law() {
        let sim = channel(2.0);
        let omega = 1.0;
        let nu_lu = (1.0 / omega - 0.5) / 3.0;
        let phi_hat: Vec<f64> = sim.phi.iter().map(|p| p / sim.zeta).collect();
        // Net force = flux through the two walls: 2 (1 - phi_hat_0) + 2 (1 - phi_hat_last).
        let total: f64 = (0..64).map(|j| sim.eo_force(j, omega)).sum();
        let wall_flux = 2.0 * (1.0 - phi_hat[0]) + 2.0 * (1.0 - phi_hat[63]);
        assert!((total - nu_lu * sim.u_hs_lu * wall_flux).abs() < 1e-14);
        // Interior nodes: three-point Laplacian of the potential.
        for j in 1..63 {
            let laplacian = phi_hat[j + 1] - 2.0 * phi_hat[j] + phi_hat[j - 1];
            assert!((sim.eo_force(j, omega) - nu_lu * sim.u_hs_lu * laplacian).abs() < 1e-15);
        }
        // Same sign everywhere (the diffuse layer carries charge of one sign) and
        // symmetric about the channel centre.
        for j in 0..32 {
            assert!(sim.eo_force(j, omega) > 0.0);
            assert!((sim.eo_force(j, omega) - sim.eo_force(63 - j, omega)).abs() < 1e-15);
        }
        assert_eq!(sim.eo_force(64, omega), 0.0);
    }

    #[test]
    fn steady_centre_velocity_equals_helmholtz_smoluchowski() {
        let mut sim = channel(2.0);
        run_to_steady_state(&mut sim, 1.0, 60_000, 1e-12);
        let ratio = sim.centre_velocity() / sim.u_hs_lu;
        assert!(
            (ratio - 1.0).abs() < 0.02,
            "centre / u_HS = {ratio} (u_c = {}, u_HS = {})",
            sim.centre_velocity(),
            sim.u_hs_lu
        );
    }

    #[test]
    fn steady_profile_follows_the_analytic_double_layer_solution() {
        let mut sim = channel(4.0);
        run_to_steady_state(&mut sim, 1.0, 60_000, 1e-12);
        for j in 0..sim.ny {
            let exact = sim.analytic_velocity(j);
            assert!(
                (sim.u[j] - exact).abs() < 0.01 * sim.u_hs_lu,
                "node {j}: u = {}, analytic = {exact}",
                sim.u[j]
            );
        }
        // Symmetry and monotone growth from the wall to the centre.
        for j in 0..32 {
            assert!((sim.u[j] - sim.u[63 - j]).abs() < 1e-10 * sim.u_hs_lu);
            assert!(sim.u[j] <= sim.u[j + 1] + 1e-12);
        }
        // No-slip: the wall-adjacent node moves slowly compared with the plug flow.
        assert!(sim.u[0] < 0.3 * sim.u_hs_lu);
        assert!(sim.u[0] > 0.0);
    }

    #[test]
    fn steady_velocity_is_independent_of_the_viscosity() {
        // The Helmholtz-Smoluchowski velocity does not depend on the viscosity.
        for &omega in &[0.7, 1.0, 1.4] {
            let mut sim = channel(4.0);
            run_to_steady_state(&mut sim, omega, 120_000, 1e-12);
            let ratio = sim.centre_velocity() / sim.u_hs_lu;
            assert!((ratio - 1.0).abs() < 0.02, "omega = {omega}: ratio {ratio}");
        }
    }

    #[test]
    fn flow_reverses_with_the_sign_of_the_zeta_potential() {
        let mut positive = ElectroosmosisLBM::new(32, 80.0, -0.05, 1e4, 1e-3, 3.0);
        let mut negative = ElectroosmosisLBM::new(32, 80.0, 0.05, 1e4, 1e-3, 3.0);
        run_to_steady_state(&mut positive, 1.2, 30_000, 1e-12);
        run_to_steady_state(&mut negative, 1.2, 30_000, 1e-12);
        assert!(positive.centre_velocity() > 0.0);
        assert!(negative.centre_velocity() < 0.0);
        assert!((positive.centre_velocity() + negative.centre_velocity()).abs() < 1e-12);
    }

    #[test]
    fn bounce_back_walls_conserve_mass_and_density_stays_uniform() {
        let mut sim = channel(2.0);
        let mass0 = sim.total_mass();
        for _ in 0..2000 {
            sim.step(1.0);
        }
        assert!((sim.total_mass() - mass0).abs() < 1e-10);
        for node in &sim.f {
            let rho: f64 = node.iter().sum();
            assert!((rho - 1.0).abs() < 1e-3, "rho = {rho}");
        }
    }

    #[test]
    fn without_a_double_layer_nothing_moves() {
        let mut sim = ElectroosmosisLBM::new(24, 80.0, 0.0, 1e4, 1e-3, 2.0);
        for _ in 0..500 {
            sim.step(1.1);
        }
        assert!(sim.u.iter().all(|&v| v.abs() < 1e-14));
    }

    #[test]
    fn rebuild_potential_tracks_changed_parameters() {
        let mut sim = channel(2.0);
        let before = sim.eo_force(0, 1.0);
        sim.debye_lu = 4.0;
        sim.rebuild_potential();
        assert!((sim.eo_force(0, 1.0) - before).abs() > 1e-6);
        let reference = channel(4.0);
        for j in 0..64 {
            assert!((sim.eo_force(j, 1.0) - reference.eo_force(j, 1.0)).abs() < 1e-15);
            assert!((sim.phi[j] - reference.phi[j]).abs() < 1e-15);
        }
    }
}
