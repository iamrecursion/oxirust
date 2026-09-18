// Copyright 2026 COOLJAPAN OU (Team KitaSan)
// SPDX-License-Identifier: Apache-2.0

//! Cahn-Hilliard lattice Boltzmann solver coupled to an incompressible
//! two-phase flow solver.
//!
//! The solver integrates the diffuse-interface model
//!
//! ```text
//! d(phi)/dt + div(phi u) = div(M grad mu),     mu = beta phi (phi^2 - 1) - kappa lap(phi)
//! div u = 0
//! rho(phi) (du/dt + u.grad u) = -grad p + div[rho(phi) nu(phi) (grad u + grad u^T)]
//!                               + mu grad(phi) + rho g
//! ```
//!
//! on a periodic D2Q9 lattice with two distribution functions:
//!
//! * `g` carries the order parameter.  Its equilibrium has the second moment
//!   `Gamma mu I + phi u u` (instead of the usual `phi c_s^2 I`), so that the
//!   chemical potential, not the concentration, drives the diffusion flux.  The
//!   Chapman-Enskog expansion gives the Cahn-Hilliard equation with the
//!   mobility `M = Gamma (tau_phi - 1/2)`, so the relaxation time and `Gamma`
//!   follow from the requested mobility.
//! * `f` is a velocity-based, weakly compressible distribution for the
//!   normalised pressure `p* = p / (rho c_s^2)` and the velocity `u`, in the
//!   manner of the incompressible phase-field lattice Boltzmann models of Liang
//!   et al. (2014) and Fakhari et al. (2017).  Because the density `rho(phi)`
//!   does not enter the equilibrium, a uniform flow through a moving interface
//!   is an exact solution and the scheme stays stable at a density ratio of ten.
//!   (A conservative pressure-momentum distribution in the manner of He, Chen &
//!   Zhang (1999) carries `rho u` and breaks down as soon as the lattice
//!   advection of the order parameter lags the advection of the momentum by a
//!   few percent.)  The variable density and viscosity are recovered with the
//!   acceleration `a = [F - c_s^2 p* grad(rho) + nu S.grad(rho)] / rho`, where
//!   `F` is the capillary and gravity force density and `S = grad u + grad u^T`,
//!   applied with the second-order accurate Guo forcing.  The per-node
//!   relaxation time `tau_f = 1/2 + nu(phi)/c_s^2` gives the kinematic
//!   viscosity `nu(phi)`; the `grad(rho)` terms turn the pressure gradient
//!   `c_s^2 grad(p*)` and the viscous stress `div(nu S)` of the lattice scheme
//!   into `grad(p)/rho` and `div(rho nu S)/rho`.
//!
//! The surface-tension force `mu grad(phi)` is the capillary force of the
//! free-energy functional; the gradient and the Laplacian use the isotropic
//! D2Q9 stencils.
//!
//! # Conservation
//!
//! The order parameter is conserved to round-off.  The mean of the external
//! force density is removed, so the continuum equations conserve the momentum
//! `sum rho u` of the periodic box; the velocity-based scheme conserves
//! `sum u` instead and `sum rho u` only to the discretisation error of the
//! density-gradient corrections.  That error would accelerate the box slowly
//! but without bound, so each step ends with a uniform velocity shift that
//! restores `sum rho u` of the step's start.
//!
//! # Accuracy
//!
//! The scheme is second order in space for smooth fields.  The lattice
//! advection of features as sharp as the interface (`k xi ~ 1`) lags the flow by
//! a few percent (third-order dispersion), and the capillary-wave frequency of a
//! wave of wave number `k` is lower than the sharp-interface value
//! `sqrt(sigma k^3 / (rho_A + rho_B))` by about `0.4 k xi` (plus the viscous
//! damping correction), the usual diffuse-interface error.

use super::{CS2, EX, EY, PhaseFieldLbmParams, W9};
use crate::{Error, Result};

/// `c_s^4` of the D2Q9 lattice.
const CS4: f64 = CS2 * CS2;

/// Index of the opposite lattice direction.
const OPPOSITE: [usize; 9] = [0, 3, 4, 1, 2, 7, 8, 5, 6];

/// Table of the periodic neighbours of every node: entry `alpha` of node `i`
/// is the node at `x_i + e_alpha`.
fn build_neighbours(nx: usize, ny: usize) -> Vec<[usize; 9]> {
    let mut table = Vec::with_capacity(nx * ny);
    for y in 0..ny {
        for x in 0..nx {
            let mut entry = [0_usize; 9];
            for (alpha, slot) in entry.iter_mut().enumerate() {
                let xn = (x as isize + EX[alpha] as isize).rem_euclid(nx as isize) as usize;
                let yn = (y as isize + EY[alpha] as isize).rem_euclid(ny as isize) as usize;
                *slot = yn * nx + xn;
            }
            table.push(entry);
        }
    }
    table
}

/// Isotropic gradient `(1/c_s^2) sum_alpha w_alpha e_alpha f(x + e_alpha)` of a
/// scalar field at the node whose neighbours are `nb`.
#[inline]
fn gradient_at(nb: &[usize; 9], field: &[f64]) -> (f64, f64) {
    let mut gx = 0.0;
    let mut gy = 0.0;
    for (alpha, &node) in nb.iter().enumerate().skip(1) {
        let weighted = W9[alpha] * field[node];
        gx += EX[alpha] * weighted;
        gy += EY[alpha] * weighted;
    }
    (gx / CS2, gy / CS2)
}

/// Isotropic Laplacian `(2/c_s^2) sum_alpha w_alpha (f(x + e_alpha) - f(x))` of
/// a scalar field at the node whose neighbours are `nb`.
#[inline]
fn laplacian_at(nb: &[usize; 9], field: &[f64]) -> f64 {
    let centre = field[nb[0]];
    let mut sum = 0.0;
    for (alpha, &node) in nb.iter().enumerate().skip(1) {
        sum += W9[alpha] * (field[node] - centre);
    }
    2.0 * sum / CS2
}

/// Equilibrium of the order-parameter distribution.
///
/// `g_alpha = w_alpha [Gamma mu / c_s^2 + phi (e.u/c_s^2 + (e.u)^2/(2 c_s^4) - u^2/(2 c_s^2))]`
/// for the moving directions and `g_0 = phi - sum_{alpha > 0} g_alpha`, so that
///
/// * `sum g = phi`,
/// * `sum e g = phi u`,
/// * `sum e e g = Gamma mu I + phi u u`.
pub(super) fn phase_equilibrium(phi: f64, mu: f64, gamma: f64, u: [f64; 2]) -> [f64; 9] {
    let u2 = u[0] * u[0] + u[1] * u[1];
    let chemical = gamma * mu / CS2;
    let mut equilibrium = [0.0_f64; 9];
    let mut moving = 0.0;
    for alpha in 1..9 {
        let eu = EX[alpha] * u[0] + EY[alpha] * u[1];
        let value =
            W9[alpha] * (chemical + phi * (eu / CS2 + eu * eu / (2.0 * CS4) - u2 / (2.0 * CS2)));
        equilibrium[alpha] = value;
        moving += value;
    }
    equilibrium[0] = phi - moving;
    equilibrium
}

/// Equilibrium of the velocity-based flow distribution for the normalised
/// pressure `p_star = p / (rho c_s^2)`.
///
/// `f_alpha = w_alpha [p* + e.u/c_s^2 + (e.u)^2/(2 c_s^4) - u^2/(2 c_s^2)]`, so that
///
/// * `sum f = p*`,
/// * `sum e f = u`,
/// * `sum e e f = c_s^2 p* I + u u`.
pub(super) fn flow_equilibrium(p_star: f64, u: [f64; 2]) -> [f64; 9] {
    let u2 = u[0] * u[0] + u[1] * u[1];
    let mut equilibrium = [0.0_f64; 9];
    for (alpha, slot) in equilibrium.iter_mut().enumerate() {
        let eu = EX[alpha] * u[0] + EY[alpha] * u[1];
        *slot = W9[alpha] * (p_star + eu / CS2 + eu * eu / (2.0 * CS4) - u2 / (2.0 * CS2));
    }
    equilibrium
}

/// Guo forcing term `w_alpha [(e_alpha - u)/c_s^2 + (e_alpha.u) e_alpha/c_s^4] . a`
/// of an acceleration `a`.  Its moments are `sum F = 0` and `sum e F = a`.
pub(super) fn flow_forcing(u: [f64; 2], acceleration: [f64; 2]) -> [f64; 9] {
    let mut forcing = [0.0_f64; 9];
    for (alpha, slot) in forcing.iter_mut().enumerate() {
        let eu = EX[alpha] * u[0] + EY[alpha] * u[1];
        let rel_x = (EX[alpha] - u[0]) / CS2 + eu * EX[alpha] / CS4;
        let rel_y = (EY[alpha] - u[1]) / CS2 + eu * EY[alpha] / CS4;
        *slot = W9[alpha] * (rel_x * acceleration[0] + rel_y * acceleration[1]);
    }
    forcing
}

/// Two-distribution-function LBM for the Cahn-Hilliard + Navier-Stokes system.
///
/// On a fully periodic `nx x ny` D2Q9 grid in lattice units (`dx = dt = 1`,
/// `c_s^2 = 1/3`, row-major storage `y * nx + x`) the solver integrates
///
/// ```text
/// dφ/dt + ∇·(φ u) = ∇·(M ∇μ),      μ = β φ (φ² − 1) − κ ∇²φ
/// ∇·u = 0
/// ρ(φ) (∂u/∂t + u·∇u) = −∇p + ∇·[ρ(φ) ν(φ) (∇u + ∇uᵀ)] + μ ∇φ + ρ g
/// ```
///
/// with the coefficients `κ`, `β` of [`PhaseFieldLbmParams::kappa`] and
/// [`PhaseFieldLbmParams::beta`] (surface tension σ, interface width ε), the
/// densities and viscosities interpolated linearly from φ, and gravity
/// `g = (gx, gy)` of the parameters.
///
/// * The order-parameter distribution `g` relaxes towards an equilibrium whose
///   second moment is `Γ μ I + φ u u`, so the chemical potential drives the
///   diffusion flux and the lattice scheme reproduces the mobility
///   `M = Γ (τ_φ − ½)`.
/// * The flow distribution `f` is velocity based: `Σ f = p / (ρ c_s²)` and
///   `Σ e f = u − a/2`, relaxed with the local `τ_f = ½ + ν(φ)/c_s²` and forced
///   with the Guo scheme by the acceleration
///   `a = [F − c_s² p* ∇ρ + ν (∇u + ∇uᵀ)·∇ρ] / ρ`, `F = μ ∇φ + ρ g` (the
///   `∇ρ` terms turn the pressure gradient and viscous stress of the lattice
///   scheme into the variable-density `∇p/ρ` and `∇·(ρ ν S)/ρ`).  A uniform
///   flow through a moving interface is therefore an exact solution.
/// * The domain mean of `F` is removed (the periodic box is a freely falling
///   frame, only density contrasts are accelerated) and each step restores the
///   total momentum `Σ ρ u` of its start, so the momentum is conserved.
///
/// The macroscopic fields (`phi`, `rho`, `mu`, `pressure`, `ux`, `uy`,
/// `force_x`, `force_y`, `acc_x`, `acc_y`) are always the moments of the
/// distributions at the current time level.  After editing `phi`, `ux`, `uy`
/// or `pressure` by hand, call
/// [`CahnHilliardLbm::init_equilibrium_distributions`] to rebuild the
/// distributions from them.
#[derive(Debug, Clone)]
pub struct CahnHilliardLbm {
    /// Grid width in x-direction.
    pub nx: usize,
    /// Grid height in y-direction.
    pub ny: usize,
    /// Simulation parameters.
    pub params: PhaseFieldLbmParams,
    /// Order-parameter distribution functions `g[i][alpha]`; `sum_alpha g = phi`.
    pub g: Vec<[f64; 9]>,
    /// Flow distribution functions `f[i][alpha]`: `sum_alpha f = p / (rho c_s^2)`
    /// and `sum_alpha e_alpha f = u - a/2` with the acceleration `a` of
    /// [`CahnHilliardLbm::acc_x`].
    pub f: Vec<[f64; 9]>,
    /// Order parameter φ at each node (ranges −1 to +1).
    pub phi: Vec<f64>,
    /// Fluid density `rho(phi)` at each node.
    pub rho: Vec<f64>,
    /// Velocity ux at each node.
    pub ux: Vec<f64>,
    /// Velocity uy at each node.
    pub uy: Vec<f64>,
    /// Chemical potential μ at each node.
    pub mu: Vec<f64>,
    /// Hydrodynamic pressure `p = rho c_s^2 sum f` at each node (defined up to
    /// the conserved mean of `sum f`; it includes the capillary pressure, so the
    /// pressure difference between the bulk of a droplet and its surroundings
    /// is `sigma / R` in 2-D).
    pub pressure: Vec<f64>,
    /// x-component of the external force density `mu d(phi)/dx + rho g_x` with
    /// its domain mean removed.
    pub force_x: Vec<f64>,
    /// y-component of the external force density `mu d(phi)/dy + rho g_y` with
    /// its domain mean removed.
    pub force_y: Vec<f64>,
    /// x-component of the acceleration applied by the next collision.
    pub acc_x: Vec<f64>,
    /// y-component of the acceleration applied by the next collision.
    pub acc_y: Vec<f64>,
    neighbours: Vec<[usize; 9]>,
    scratch_g: Vec<[f64; 9]>,
    scratch_f: Vec<[f64; 9]>,
}

impl CahnHilliardLbm {
    /// Construct a new solver on an `nx x ny` periodic grid, initialised to a
    /// quiescent homogeneous mixture `phi = 0`.
    pub fn new(nx: usize, ny: usize, params: PhaseFieldLbmParams) -> Self {
        let n = nx * ny;
        let rho0 = params.rho_from_phi(0.0);
        let mut solver = CahnHilliardLbm {
            nx,
            ny,
            params,
            g: vec![[0.0; 9]; n],
            f: vec![[0.0; 9]; n],
            phi: vec![0.0; n],
            rho: vec![rho0; n],
            ux: vec![0.0; n],
            uy: vec![0.0; n],
            mu: vec![0.0; n],
            pressure: vec![0.0; n],
            force_x: vec![0.0; n],
            force_y: vec![0.0; n],
            acc_x: vec![0.0; n],
            acc_y: vec![0.0; n],
            neighbours: build_neighbours(nx, ny),
            scratch_g: vec![[0.0; 9]; n],
            scratch_f: vec![[0.0; 9]; n],
        };
        solver.init_equilibrium_distributions();
        solver
    }

    /// Flat node index from (x, y).
    #[inline]
    pub fn idx(&self, x: usize, y: usize) -> usize {
        y * self.nx + x
    }

    /// Initialize with flat interfaces: phase A (φ = +1) fills the upper half
    /// `y > ny/2`, phase B (φ = −1) the lower half.
    ///
    /// The grid is periodic, so the profile
    /// `φ = ± tanh(d / (ε/√2))`, `d = min(|y − ny/2|, y, ny − y)`,
    /// contains a second interface at `y = 0` (≡ `ny`) in addition to the one
    /// at `y = ny/2`; both have the equilibrium width of the free-energy
    /// functional.  Velocity and pressure are taken from the `ux`, `uy` and
    /// `pressure` fields (zero for a fresh solver).
    pub fn init_flat_interface(&mut self) {
        let xi = self.params.interface_half_width();
        let ny_f = self.ny as f64;
        let middle = ny_f / 2.0;
        for y in 0..self.ny {
            let yf = y as f64;
            let distance = (yf - middle).abs().min(yf).min(ny_f - yf);
            let signed = if yf >= middle { distance } else { -distance };
            let phi_val = (signed / xi).tanh();
            for x in 0..self.nx {
                let i = self.idx(x, y);
                self.phi[i] = phi_val;
            }
        }
        self.init_equilibrium_distributions();
    }

    /// Initialize a circular droplet of phase A (φ = +1) with radius `radius`
    /// centred at `(cx, cy)` in phase B (φ = −1).
    ///
    /// The profile is `φ = −tanh((r − R)/(ε/√2))` with `r` the distance to the
    /// centre on the periodic grid (minimum image).  Velocity and pressure are
    /// taken from the `ux`, `uy` and `pressure` fields (zero for a fresh
    /// solver).
    pub fn init_droplet(&mut self, cx: f64, cy: f64, radius: f64) {
        let xi = self.params.interface_half_width();
        let nx_f = self.nx as f64;
        let ny_f = self.ny as f64;
        for y in 0..self.ny {
            for x in 0..self.nx {
                let mut dx = (x as f64 - cx).abs();
                dx = dx.min((nx_f - dx).abs());
                let mut dy = (y as f64 - cy).abs();
                dy = dy.min((ny_f - dy).abs());
                let r = (dx * dx + dy * dy).sqrt();
                let i = self.idx(x, y);
                self.phi[i] = -((r - radius) / xi).tanh();
            }
        }
        self.init_equilibrium_distributions();
    }

    /// Compute chemical potential μ = β·φ·(φ²−1) − κ·∇²φ with the isotropic
    /// D2Q9 Laplacian.
    pub fn compute_chemical_potential(&mut self) {
        let beta = self.params.beta();
        let kappa = self.params.kappa();
        for (i, nb) in self.neighbours.iter().enumerate() {
            let phi_c = self.phi[i];
            let lap_phi = laplacian_at(nb, &self.phi);
            self.mu[i] = beta * phi_c * (phi_c * phi_c - 1.0) - kappa * lap_phi;
        }
    }

    /// Recompute the density and the chemical potential from the current order
    /// parameter.
    fn refresh_density_and_potential(&mut self) {
        for (rho, &phi) in self.rho.iter_mut().zip(self.phi.iter()) {
            *rho = self.params.rho_from_phi(phi);
        }
        self.compute_chemical_potential();
    }

    /// Compute the external force density and the acceleration of the next
    /// collision from the current `phi`, `rho`, `mu`, the pressure moment
    /// `sum f` and the velocity (which enters the viscous correction).
    ///
    /// The external force density is `mu grad(phi) + rho g` minus its domain
    /// mean: the grid is periodic, so a uniform force would accelerate the whole
    /// box (a freely falling frame has neither a pressure gradient nor relative
    /// motion), and the discrete capillary force has a tiny non-zero mean.
    ///
    /// The acceleration is `a = [F - c_s^2 p* grad(rho) + nu S.grad(rho)] / rho`
    /// with `p* = sum f` and `S = grad u + grad u^T`.
    fn compute_accelerations(&mut self) {
        let n = self.neighbours.len();
        let gx = self.params.gx;
        let gy = self.params.gy;
        let mut sum_x = 0.0;
        let mut sum_y = 0.0;
        for (i, nb) in self.neighbours.iter().enumerate() {
            let (dphi_x, dphi_y) = gradient_at(nb, &self.phi);
            let fx = self.mu[i] * dphi_x + self.rho[i] * gx;
            let fy = self.mu[i] * dphi_y + self.rho[i] * gy;
            self.force_x[i] = fx;
            self.force_y[i] = fy;
            sum_x += fx;
            sum_y += fy;
        }
        if n == 0 {
            return;
        }
        let mean_x = sum_x / n as f64;
        let mean_y = sum_y / n as f64;
        for (i, nb) in self.neighbours.iter().enumerate() {
            self.force_x[i] -= mean_x;
            self.force_y[i] -= mean_y;
            let rho = self.rho[i];
            let (drho_x, drho_y) = gradient_at(nb, &self.rho);
            let (dux_dx, dux_dy) = gradient_at(nb, &self.ux);
            let (duy_dx, duy_dy) = gradient_at(nb, &self.uy);
            let nu = self.params.nu_from_phi(self.phi[i]);
            let p_star: f64 = self.f[i].iter().sum();
            let strain_xx = 2.0 * dux_dx;
            let strain_yy = 2.0 * duy_dy;
            let strain_xy = dux_dy + duy_dx;
            self.acc_x[i] = (self.force_x[i] - CS2 * p_star * drho_x
                + nu * (strain_xx * drho_x + strain_xy * drho_y))
                / rho;
            self.acc_y[i] = (self.force_y[i] - CS2 * p_star * drho_y
                + nu * (strain_xy * drho_x + strain_yy * drho_y))
                / rho;
        }
    }

    /// Initialize both distributions with their equilibrium for the current
    /// macroscopic fields `phi`, `ux`, `uy` and `pressure`.
    ///
    /// The density, chemical potential, force and acceleration are recomputed
    /// from them.  The momentum of the flow distribution is shifted by half the
    /// acceleration so that the velocity moment reproduces `ux`, `uy` exactly.
    pub fn init_equilibrium_distributions(&mut self) {
        self.refresh_density_and_potential();
        let gamma = self.params.gamma();
        let n = self.neighbours.len();
        // First pass: the pressure moment and the velocity define the
        // acceleration (the viscous correction needs the velocity gradient).
        for i in 0..n {
            let p_star = self.pressure[i] / (self.rho[i] * CS2);
            self.f[i] = flow_equilibrium(p_star, [self.ux[i], self.uy[i]]);
        }
        self.compute_accelerations();
        for i in 0..n {
            let u = [self.ux[i], self.uy[i]];
            let p_star = self.pressure[i] / (self.rho[i] * CS2);
            self.g[i] = phase_equilibrium(self.phi[i], self.mu[i], gamma, u);
            let u_shifted = [u[0] - 0.5 * self.acc_x[i], u[1] - 0.5 * self.acc_y[i]];
            self.f[i] = flow_equilibrium(p_star, u_shifted);
        }
    }

    /// Check grid, parameters and array sizes.
    fn check_consistency(&self) -> Result<()> {
        Error::check_grid(self.nx, self.ny)?;
        self.params.validate()?;
        let n = self.nx * self.ny;
        let sizes = [
            ("phi", self.phi.len()),
            ("rho", self.rho.len()),
            ("ux", self.ux.len()),
            ("uy", self.uy.len()),
            ("mu", self.mu.len()),
            ("pressure", self.pressure.len()),
            ("force_x", self.force_x.len()),
            ("force_y", self.force_y.len()),
            ("acc_x", self.acc_x.len()),
            ("acc_y", self.acc_y.len()),
            ("g", self.g.len()),
            ("f", self.f.len()),
            ("neighbours", self.neighbours.len()),
        ];
        for (name, found) in sizes {
            if found != n {
                return Err(Error::ComponentCountMismatch {
                    expected: n,
                    found,
                    field: name.to_string(),
                });
            }
        }
        Ok(())
    }

    /// Perform one full LBM timestep: collision + streaming, then update of the
    /// macroscopic fields.
    ///
    /// Invalid parameters or inconsistent field sizes leave the state untouched;
    /// use [`CahnHilliardLbm::try_step`] to learn why.
    pub fn step(&mut self) {
        let _ = self.try_step();
    }

    /// Perform one full LBM timestep, reporting invalid input.
    ///
    /// # Errors
    /// Returns an error — and leaves the state untouched — for an empty grid,
    /// invalid parameters (see [`PhaseFieldLbmParams::validate`]) or field
    /// arrays that do not match the grid.  If the step produces a non-finite
    /// order parameter, velocity or pressure, the offending field is reported
    /// and the solver must be re-initialised.
    pub fn try_step(&mut self) -> Result<()> {
        self.check_consistency()?;
        let momentum = self.total_momentum();
        self.collide();
        self.stream();
        self.update_macroscopic();
        self.restore_momentum(momentum);
        for (name, values) in [
            ("phi", &self.phi),
            ("ux", &self.ux),
            ("uy", &self.uy),
            ("pressure", &self.pressure),
        ] {
            if let Some(index) = values.iter().position(|v| !v.is_finite()) {
                Error::check_finite(values[index], name, index)?;
            }
        }
        Ok(())
    }

    /// BGK collision for both distributions.
    ///
    /// The order-parameter distribution relaxes with the constant `tau_phi`; the
    /// flow distribution relaxes with the local `tau_f = 1/2 + nu(phi)/c_s^2` and
    /// receives the Guo forcing of the acceleration with the factor
    /// `1 - 1/(2 tau_f)`.
    fn collide(&mut self) {
        let gamma = self.params.gamma();
        let inv_tau_phi = 1.0 / self.params.tau_phi;
        for i in 0..self.neighbours.len() {
            let phi = self.phi[i];
            let u = [self.ux[i], self.uy[i]];
            let geq = phase_equilibrium(phi, self.mu[i], gamma, u);
            for (g, eq) in self.g[i].iter_mut().zip(geq.iter()) {
                *g += inv_tau_phi * (eq - *g);
            }

            let tau_f = self.params.tau_f_from_phi(phi);
            let p_star: f64 = self.f[i].iter().sum();
            let feq = flow_equilibrium(p_star, u);
            let forcing = flow_forcing(u, [self.acc_x[i], self.acc_y[i]]);
            let forcing_weight = 1.0 - 0.5 / tau_f;
            for alpha in 0..9 {
                self.f[i][alpha] +=
                    (feq[alpha] - self.f[i][alpha]) / tau_f + forcing_weight * forcing[alpha];
            }
        }
    }

    /// Streaming step with periodic boundary conditions.
    fn stream(&mut self) {
        for (i, nb) in self.neighbours.iter().enumerate() {
            for alpha in 0..9 {
                let source = nb[OPPOSITE[alpha]];
                self.scratch_g[i][alpha] = self.g[source][alpha];
                self.scratch_f[i][alpha] = self.f[source][alpha];
            }
        }
        std::mem::swap(&mut self.g, &mut self.scratch_g);
        std::mem::swap(&mut self.f, &mut self.scratch_f);
    }

    /// Update the macroscopic quantities from the distributions:
    /// `phi = sum g`; then `rho(phi)`, `mu`, the force and the acceleration
    /// `a`; the velocity `u = sum e f + a/2` and the pressure
    /// `p = rho c_s^2 sum f`.
    fn update_macroscopic(&mut self) {
        for (phi, g) in self.phi.iter_mut().zip(self.g.iter()) {
            *phi = g.iter().sum();
        }
        self.refresh_density_and_potential();
        // The viscous correction of the acceleration reads the velocity of the
        // previous time level, which is still stored in `ux`, `uy`.
        self.compute_accelerations();
        for i in 0..self.neighbours.len() {
            let mut zeroth = 0.0;
            let mut jx = 0.0;
            let mut jy = 0.0;
            for alpha in 0..9 {
                let value = self.f[i][alpha];
                zeroth += value;
                jx += EX[alpha] * value;
                jy += EY[alpha] * value;
            }
            self.ux[i] = jx + 0.5 * self.acc_x[i];
            self.uy[i] = jy + 0.5 * self.acc_y[i];
            self.pressure[i] = self.rho[i] * CS2 * zeroth;
        }
    }

    /// Restore the total momentum `sum rho u` to `target`.
    ///
    /// The external force has a vanishing mean and the continuum equations
    /// conserve the momentum of the periodic box, but the velocity-based scheme
    /// conserves `sum u` and honours `sum rho u` only up to the discretisation
    /// error of the density-gradient corrections, which would otherwise
    /// accelerate the whole box slowly but without bound.  A uniform velocity
    /// shift `du` of both the velocity and the first moment of the flow
    /// distribution (`f_alpha += w_alpha e_alpha.du / c_s^2`) removes the
    /// drift and leaves the pressure moment and all gradients untouched.
    fn restore_momentum(&mut self, target: (f64, f64)) {
        let (px, py) = self.total_momentum();
        let mass: f64 = self.rho.iter().sum();
        if mass <= 0.0 || !mass.is_finite() {
            return;
        }
        let dux = (target.0 - px) / mass;
        let duy = (target.1 - py) / mass;
        for i in 0..self.neighbours.len() {
            self.ux[i] += dux;
            self.uy[i] += duy;
            for alpha in 0..9 {
                self.f[i][alpha] += W9[alpha] * (EX[alpha] * dux + EY[alpha] * duy) / CS2;
            }
        }
    }

    /// Compute the total free energy
    /// `F = sum [(β/4)(φ²−1)² + κ/2 |∇φ|²]` (lattice units, `dx = 1`).
    ///
    /// The gradient energy `κ/(2 c_s^2) sum_alpha w_alpha (φ(x+e_α) − φ(x))²` is
    /// the discrete energy whose variational derivative is exactly the chemical
    /// potential of [`CahnHilliardLbm::compute_chemical_potential`], so the
    /// energy of a flat interface per unit length is the surface tension
    /// `σ`.
    pub fn total_free_energy(&self) -> f64 {
        let beta = self.params.beta();
        let kappa = self.params.kappa();
        let mut total = 0.0;
        for (i, nb) in self.neighbours.iter().enumerate() {
            let phi = self.phi[i];
            let f_bulk = 0.25 * beta * (phi * phi - 1.0).powi(2);
            let mut grad2 = 0.0;
            for (alpha, &node) in nb.iter().enumerate().skip(1) {
                let d = self.phi[node] - phi;
                grad2 += W9[alpha] * d * d;
            }
            total += f_bulk + 0.5 * kappa * grad2 / CS2;
        }
        total
    }

    /// Compute the total mass (integral of φ + 1) / 2.
    pub fn total_mass(&self) -> f64 {
        self.phi.iter().map(|&p| 0.5 * (p + 1.0)).sum()
    }

    /// Total momentum `(sum rho u_x, sum rho u_y)` of the fluid.
    pub fn total_momentum(&self) -> (f64, f64) {
        let mut px = 0.0;
        let mut py = 0.0;
        for ((&rho, &ux), &uy) in self.rho.iter().zip(self.ux.iter()).zip(self.uy.iter()) {
            px += rho * ux;
            py += rho * uy;
        }
        (px, py)
    }

    /// Total kinetic energy `1/2 sum rho |u|^2`.
    pub fn kinetic_energy(&self) -> f64 {
        let mut energy = 0.0;
        for ((&rho, &ux), &uy) in self.rho.iter().zip(self.ux.iter()).zip(self.uy.iter()) {
            energy += 0.5 * rho * (ux * ux + uy * uy);
        }
        energy
    }

    /// Largest speed `|u|` on the grid.
    pub fn max_speed(&self) -> f64 {
        self.ux
            .iter()
            .zip(self.uy.iter())
            .map(|(&ux, &uy)| (ux * ux + uy * uy).sqrt())
            .fold(0.0, f64::max)
    }
}
