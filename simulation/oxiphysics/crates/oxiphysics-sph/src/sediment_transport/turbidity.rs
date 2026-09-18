// Copyright 2026 COOLJAPAN OU (Team KitaSan)
// SPDX-License-Identifier: Apache-2.0

//! Particle-resolved SPH solver for turbidity (sediment-laden gravity) currents.
//!
//! The solver is a weakly-compressible SPH (WCSPH) scheme in the Boussinesq
//! approximation:
//!
//! * **Density** – kernel summation `ρ_i = Σ_j m_j W_ij` over fluid and wall
//!   particles (Wendland C2 kernel, support `2h`, 2-D or 3-D normalisation).
//! * **Equation of state** – Tait, `p = B[(ρ/ρ₀)^γ − 1]` with
//!   `B = ρ₀ c₀² / γ` about the ambient fluid density `ρ₀`.
//! * **Momentum** – symmetric pressure gradient
//!   `−Σ_j m_j (p_i/ρ_i² + p_j/ρ_j²) ∇_i W_ij`, Monaghan artificial viscosity
//!   between fluid pairs, and the Boussinesq buoyancy
//!   `−g′(c) ẑ = −g c (ρ_s − ρ_f)/ρ_f ẑ` that drives the current.
//! * **Walls** – fixed wall particles whose pressure is extrapolated from the
//!   fluid following Adami, Hu & Adams (2012), including the body-force term,
//!   so that walls are impermeable and support the hydrostatic load.  Walls
//!   are free-slip (no viscous coupling).
//! * **Concentration** – Lagrangian advection with the particles, turbulent
//!   diffusion via the conservative pair form of [`super::sph_diffusion_rate`],
//!   and gravitational settling at `w_s` discretised as a conservative upwind
//!   pairwise flux; settling flux into walls whose normal faces upwards is
//!   recorded as deposited sediment.
//!
//! Gravity acts along `−z`.  For 2-D runs all particles must share the same
//! `y` coordinate (the simulation lives in the x–z plane).
//!
//! The ambient (clear) fluid must also be represented by particles (with
//! `c = 0`): the front speed of a gravity current is set by the exchange flow
//! between heavy and light fluid.

use std::f64::consts::PI;

use super::GRAVITY;
use crate::error::{Error, Result};
use crate::neighbor::LinkedCellList;

// ---------------------------------------------------------------------------
// Turbidity particle
// ---------------------------------------------------------------------------

/// A single SPH particle of a turbidity current (sediment-laden fluid) or of
/// the clear ambient fluid surrounding it (`concentration = 0`).
#[derive(Debug, Clone)]
pub struct TurbidityParticle {
    /// Particle position \[x, y, z\] (m).
    pub position: [f64; 3],
    /// Particle velocity \[vx, vy, vz\] (m/s).
    pub velocity: [f64; 3],
    /// Volumetric sediment concentration (0–1).
    pub concentration: f64,
    /// Particle mass (kg).
    pub mass: f64,
    /// Smoothing length h (m).
    pub h: f64,
}

impl TurbidityParticle {
    /// Create a new [`TurbidityParticle`].
    pub fn new(
        position: [f64; 3],
        velocity: [f64; 3],
        concentration: f64,
        mass: f64,
        h: f64,
    ) -> Self {
        Self {
            position,
            velocity,
            concentration: concentration.clamp(0.0, 1.0),
            mass,
            h,
        }
    }

    /// Compute the bulk density of the particle (fluid + sediment mixture).
    ///
    /// # Arguments
    /// * `rho_f` – fluid density (kg/m³)
    /// * `rho_s` – sediment density (kg/m³)
    pub fn bulk_density(&self, rho_f: f64, rho_s: f64) -> f64 {
        (1.0 - self.concentration) * rho_f + self.concentration * rho_s
    }

    /// Buoyancy-reduced gravity `g′ = g (ρ_bulk − ρ_f)/ρ_f` acting on the
    /// particle (m/s²).
    ///
    /// # Arguments
    /// * `rho_f` – ambient fluid density (kg/m³)
    /// * `rho_s` – sediment grain density (kg/m³)
    pub fn reduced_gravity(&self, rho_f: f64, rho_s: f64) -> f64 {
        let rho_bulk = self.bulk_density(rho_f, rho_s);
        GRAVITY * (rho_bulk - rho_f) / rho_f
    }

    /// Advance an *isolated* particle (no SPH neighbours) by one explicit
    /// symplectic-Euler step under reduced gravity and quadratic drag.
    ///
    /// The drag acceleration is `−c_d |v| v / h`: the dimensionless drag
    /// coefficient is made dimensional with the particle size `h`, so the
    /// term has units of m/s².  Collective dynamics are handled by
    /// [`TurbidityCurrentSph::step`].
    ///
    /// # Arguments
    /// * `dt`     – time step (s)
    /// * `rho_f`  – ambient fluid density (kg/m³)
    /// * `rho_s`  – sediment grain density (kg/m³)
    /// * `cd`     – drag coefficient (dimensionless)
    pub fn step(&mut self, dt: f64, rho_f: f64, rho_s: f64, cd: f64) {
        let g_r = self.reduced_gravity(rho_f, rho_s);
        let drag = drag_acceleration(self.velocity, cd, self.h);
        self.velocity[0] += dt * drag[0];
        self.velocity[1] += dt * drag[1];
        self.velocity[2] += dt * (drag[2] - g_r);
        for k in 0..3 {
            self.position[k] += dt * self.velocity[k];
        }
    }

    /// Reduce the particle concentration by settling out of a particle of
    /// size `h` at settling velocity `w_s` over `dt`.
    ///
    /// Solves `dc/dt = −w_s c / h` exactly, i.e. `c ← c·exp(−w_s dt / h)`,
    /// which is unconditionally positive.  This is the single-particle
    /// (no-neighbour) model; inside [`TurbidityCurrentSph::step`] settling is
    /// a conservative flux between neighbouring particles.
    ///
    /// # Arguments
    /// * `w_s` – settling velocity (m/s)
    /// * `dt`  – time step (s)
    pub fn settle(&mut self, w_s: f64, dt: f64) {
        if self.h <= 0.0 || w_s <= 0.0 || dt <= 0.0 {
            return;
        }
        self.concentration = (self.concentration * (-w_s * dt / self.h).exp()).max(0.0);
    }
}

/// Quadratic drag acceleration `−c_d |v| v / h` (m/s²).
fn drag_acceleration(v: [f64; 3], cd: f64, h: f64) -> [f64; 3] {
    let v_mag = (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt();
    if v_mag <= 0.0 || h <= 0.0 || cd == 0.0 {
        return [0.0; 3];
    }
    let fac = -cd * v_mag / h;
    [fac * v[0], fac * v[1], fac * v[2]]
}

// ---------------------------------------------------------------------------
// Wall particle
// ---------------------------------------------------------------------------

/// A fixed wall (boundary) particle of a [`TurbidityCurrentSph`].
#[derive(Debug, Clone)]
pub struct TurbidityWallParticle {
    /// Position \[x, y, z\] (m).
    pub position: [f64; 3],
    /// Unit wall normal pointing *into* the fluid.  A floor has `[0, 0, 1]`.
    /// Settling flux is deposited only into walls with `normal[2] > 0`.
    pub normal: [f64; 3],
    /// Particle mass (kg), normally `ρ₀ Δx^d` like the fluid particles.
    pub mass: f64,
    /// Smoothing length h (m).
    pub h: f64,
}

impl TurbidityWallParticle {
    /// Create a wall particle; the normal is normalised (a zero normal is
    /// kept as zero, which makes the particle inert for settling).
    pub fn new(position: [f64; 3], normal: [f64; 3], mass: f64, h: f64) -> Self {
        let n = (normal[0] * normal[0] + normal[1] * normal[1] + normal[2] * normal[2]).sqrt();
        let normal = if n > 0.0 {
            [normal[0] / n, normal[1] / n, normal[2] / n]
        } else {
            [0.0; 3]
        };
        Self {
            position,
            normal,
            mass,
            h,
        }
    }
}

// ---------------------------------------------------------------------------
// Kernel
// ---------------------------------------------------------------------------

/// Spatial dimension of a [`TurbidityCurrentSph`] simulation (selects the
/// kernel normalisation).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TurbidityDimension {
    /// Planar x–z simulation (all particles share one `y`).
    Two,
    /// Full 3-D simulation.
    Three,
}

impl TurbidityDimension {
    /// Wendland C2 normalisation constant for smoothing length `h`.
    fn wendland_sigma(self, h: f64) -> f64 {
        match self {
            Self::Two => 7.0 / (4.0 * PI * h * h),
            Self::Three => 21.0 / (16.0 * PI * h * h * h),
        }
    }

    /// Wendland C2 kernel `W(r, h)` with support `2h`.
    fn w(self, r: f64, h: f64) -> f64 {
        let q = r / h;
        if q >= 2.0 {
            return 0.0;
        }
        let t = 1.0 - 0.5 * q;
        self.wendland_sigma(h) * t * t * t * t * (2.0 * q + 1.0)
    }

    /// Signed radial derivative `dW/dr` (≤ 0 inside the support).
    fn dw_dr(self, r: f64, h: f64) -> f64 {
        let q = r / h;
        if q >= 2.0 {
            return 0.0;
        }
        let t = 1.0 - 0.5 * q;
        -5.0 * q * t * t * t * self.wendland_sigma(h) / h
    }
}

// ---------------------------------------------------------------------------
// Turbidity current ensemble
// ---------------------------------------------------------------------------

/// Particle-resolved WCSPH turbidity-current solver (Boussinesq).
///
/// * Density by kernel summation over fluid and wall particles (Wendland C2,
///   support `2h`, 2-D or 3-D normalisation via [`TurbidityDimension`]);
///   Tait EOS `p = B[(ρ/ρ₀)^γ − 1]`, `B = ρ₀ c₀²/γ`, about `rho_fluid`.
/// * Symmetric pressure gradient, Monaghan artificial viscosity between fluid
///   pairs, and buoyancy `−g′(c) ẑ` with `g′ = g c (ρ_s − ρ_f)/ρ_f`.
/// * Fixed free-slip wall particles ([`TurbidityWallParticle`]) whose
///   pressure is extrapolated from the fluid (Adami, Hu & Adams 2012),
///   including the buoyancy term.
/// * Concentration advected with the particles, diffused with
///   [`super::sph_diffusion_rate`] (signed `dW/dr`), and settled at `w_s` by
///   a conservative upwind pair flux; flux into upward-facing walls is
///   accumulated as deposited sediment.
///
/// Gravity acts along `−z`; 2-D runs live in the x–z plane (all particles
/// share one `y`).  The ambient clear fluid must be present as particles
/// with `c = 0`.
#[derive(Debug, Clone)]
pub struct TurbidityCurrentSph {
    /// Fluid particles (turbid and clear).
    pub particles: Vec<TurbidityParticle>,
    /// Fixed wall particles.
    pub walls: Vec<TurbidityWallParticle>,
    /// Ambient fluid density ρ₀ (kg/m³); reference density of the EOS.
    pub rho_fluid: f64,
    /// Sediment grain density (kg/m³).
    pub rho_sediment: f64,
    /// Artificial speed of sound c₀ (m/s); should be ≥ 10× the largest flow
    /// speed so that density fluctuations stay below ~1 %.
    pub sound_speed: f64,
    /// Tait exponent γ (7 for water).
    pub gamma: f64,
    /// Monaghan artificial-viscosity coefficient α (dimensionless).
    pub alpha_viscosity: f64,
    /// Turbulent sediment diffusivity ε_s (m²/s).
    pub sediment_diffusivity: f64,
    /// Spatial dimension (kernel normalisation).
    pub dimension: TurbidityDimension,
    densities: Vec<f64>,
    pressures: Vec<f64>,
    wall_pressures: Vec<f64>,
    deposited_volume: f64,
    time: f64,
}

impl TurbidityCurrentSph {
    /// Create a new empty 3-D [`TurbidityCurrentSph`] with default numerical
    /// parameters (c₀ = 20 m/s, γ = 7, α = 0.02, ε_s = 0).
    pub fn new(rho_fluid: f64, rho_sediment: f64) -> Self {
        Self {
            particles: Vec::new(),
            walls: Vec::new(),
            rho_fluid,
            rho_sediment,
            sound_speed: 20.0,
            gamma: 7.0,
            alpha_viscosity: 0.02,
            sediment_diffusivity: 0.0,
            dimension: TurbidityDimension::Three,
            densities: Vec::new(),
            pressures: Vec::new(),
            wall_pressures: Vec::new(),
            deposited_volume: 0.0,
            time: 0.0,
        }
    }

    /// Set the artificial speed of sound c₀ (m/s).
    pub fn with_sound_speed(mut self, c0: f64) -> Self {
        self.sound_speed = c0;
        self
    }

    /// Set the Monaghan artificial-viscosity coefficient α.
    pub fn with_artificial_viscosity(mut self, alpha: f64) -> Self {
        self.alpha_viscosity = alpha;
        self
    }

    /// Set the turbulent sediment diffusivity ε_s (m²/s).
    pub fn with_sediment_diffusivity(mut self, eps_s: f64) -> Self {
        self.sediment_diffusivity = eps_s;
        self
    }

    /// Set the spatial dimension.
    pub fn with_dimension(mut self, dimension: TurbidityDimension) -> Self {
        self.dimension = dimension;
        self
    }

    /// Add a fluid particle to the current.
    pub fn add_particle(&mut self, p: TurbidityParticle) {
        self.particles.push(p);
    }

    /// Add a fixed wall particle.
    pub fn add_wall_particle(&mut self, w: TurbidityWallParticle) {
        self.walls.push(w);
    }

    /// Remove fluid particles whose concentration has fallen below a
    /// threshold.
    ///
    /// Only meaningful when the ensemble tracks the turbid fluid alone; in a
    /// resolved simulation (ambient fluid present as particles) removing
    /// particles opens voids in the continuum.
    ///
    /// # Arguments
    /// * `c_min` – minimum concentration threshold
    pub fn prune(&mut self, c_min: f64) {
        self.particles.retain(|p| p.concentration >= c_min);
        self.densities.clear();
        self.pressures.clear();
    }

    /// Sediment volume held in suspension, `Σ_i (m_i/ρ₀) c_i` (m³).
    pub fn suspended_sediment_volume(&self) -> f64 {
        if self.rho_fluid <= 0.0 {
            return 0.0;
        }
        self.particles
            .iter()
            .map(|p| p.mass / self.rho_fluid * p.concentration)
            .sum()
    }

    /// Total suspended sediment mass `ρ_s Σ_i (m_i/ρ₀) c_i` (kg).
    pub fn total_sediment_mass(&self) -> f64 {
        self.suspended_sediment_volume() * self.rho_sediment
    }

    /// Sediment volume deposited onto upward-facing walls so far (m³).
    pub fn deposited_sediment_volume(&self) -> f64 {
        self.deposited_volume
    }

    /// Sediment mass deposited onto upward-facing walls so far (kg).
    pub fn deposited_sediment_mass(&self) -> f64 {
        self.deposited_volume * self.rho_sediment
    }

    /// Return the number of fluid particles.
    pub fn num_particles(&self) -> usize {
        self.particles.len()
    }

    /// Simulated time (s).
    pub fn time(&self) -> f64 {
        self.time
    }

    /// Fluid particle densities from the last step (empty before the first).
    pub fn densities(&self) -> &[f64] {
        &self.densities
    }

    /// Fluid particle pressures from the last step (empty before the first).
    pub fn pressures(&self) -> &[f64] {
        &self.pressures
    }

    /// Wall pressures (Adami extrapolation) from the last step.
    pub fn wall_pressures(&self) -> &[f64] {
        &self.wall_pressures
    }

    /// Largest `x` among particles with `concentration ≥ c_threshold` and
    /// `z ≤ z_max` (position of a gravity-current nose travelling in `+x`).
    pub fn front_position(&self, c_threshold: f64, z_max: f64) -> Option<f64> {
        self.particles
            .iter()
            .filter(|p| p.concentration >= c_threshold && p.position[2] <= z_max)
            .map(|p| p.position[0])
            .reduce(f64::max)
    }

    /// Total fluid momentum `Σ m_i v_i` (kg·m/s).
    pub fn total_momentum(&self) -> [f64; 3] {
        let mut m = [0.0; 3];
        for p in &self.particles {
            for (k, mk) in m.iter_mut().enumerate() {
                *mk += p.mass * p.velocity[k];
            }
        }
        m
    }

    /// Maximum stable time step from the acoustic CFL condition
    /// (`0.25 h/(c₀ + |v|_max)`), the body-force condition
    /// (`0.25 √(h/|g′|_max)`) and the explicit diffusion limit
    /// (`0.125 h²/ε_s`).  Returns `f64::INFINITY` for an empty ensemble.
    pub fn max_stable_dt(&self) -> f64 {
        let mut dt = f64::INFINITY;
        for p in &self.particles {
            if p.h <= 0.0 {
                continue;
            }
            let v = p.velocity;
            let v_mag = (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt();
            dt = dt.min(0.25 * p.h / (self.sound_speed + v_mag));
            let g_r = p.reduced_gravity(self.rho_fluid, self.rho_sediment).abs();
            if g_r > 0.0 {
                dt = dt.min(0.25 * (p.h / g_r).sqrt());
            }
            if self.sediment_diffusivity > 0.0 {
                dt = dt.min(0.125 * p.h * p.h / self.sediment_diffusivity);
            }
        }
        dt
    }

    fn validate(&self, dt: f64, cd: f64, w_s: f64) -> Result<()> {
        let check = |name: &str, value: f64, min: f64, strict: bool| -> Result<()> {
            let ok = value.is_finite() && if strict { value > min } else { value >= min };
            if ok {
                Ok(())
            } else {
                Err(Error::InvalidParameter {
                    name: name.to_string(),
                    value,
                    min,
                    max: f64::INFINITY,
                })
            }
        };
        check("dt", dt, 0.0, true)?;
        check("cd", cd, 0.0, false)?;
        check("w_s", w_s, 0.0, false)?;
        check("rho_fluid", self.rho_fluid, 0.0, true)?;
        check("rho_sediment", self.rho_sediment, 0.0, true)?;
        check("sound_speed", self.sound_speed, 0.0, true)?;
        check("gamma", self.gamma, 0.0, true)?;
        check("alpha_viscosity", self.alpha_viscosity, 0.0, false)?;
        check(
            "sediment_diffusivity",
            self.sediment_diffusivity,
            0.0,
            false,
        )?;
        for (i, p) in self.particles.iter().enumerate() {
            if !(p.h.is_finite() && p.h > 0.0) {
                return Err(Error::NonPositiveSmoothingLength {
                    particle_index: i,
                    h: p.h,
                });
            }
            check("particle mass", p.mass, 0.0, true)?;
        }
        for (k, w) in self.walls.iter().enumerate() {
            if !(w.h.is_finite() && w.h > 0.0) {
                return Err(Error::NonPositiveSmoothingLength {
                    particle_index: self.particles.len() + k,
                    h: w.h,
                });
            }
            check("wall mass", w.mass, 0.0, true)?;
        }
        Ok(())
    }

    /// Advance the turbidity current by one WCSPH time step.
    ///
    /// # Arguments
    /// * `dt`   – time step (s); see [`Self::max_stable_dt`]
    /// * `cd`   – quadratic drag coefficient for an additional
    ///   `−c_d |v| v / h` damping of every fluid particle (use 0 for a fully
    ///   resolved simulation)
    /// * `w_s`  – sediment settling velocity (m/s)
    ///
    /// # Errors
    /// [`Error::InvalidParameter`] / [`Error::NonPositiveSmoothingLength`] for
    /// invalid inputs, [`Error::NonFiniteQuantity`] if the state blows up
    /// (time step too large).  On error the particle state is unchanged.
    pub fn step(&mut self, dt: f64, cd: f64, w_s: f64) -> Result<()> {
        self.validate(dt, cd, w_s)?;
        let n_f = self.particles.len();
        if n_f == 0 {
            self.time += dt;
            return Ok(());
        }
        let n_w = self.walls.len();
        let dim = self.dimension;
        let rho0 = self.rho_fluid;
        let gamma = self.gamma;
        let b_eos = rho0 * self.sound_speed * self.sound_speed / gamma;

        // ---- neighbour search over fluid + wall particles -------------------
        let mut positions: Vec<[f64; 3]> = Vec::with_capacity(n_f + n_w);
        positions.extend(self.particles.iter().map(|p| p.position));
        positions.extend(self.walls.iter().map(|w| w.position));
        let h_max = self
            .particles
            .iter()
            .map(|p| p.h)
            .chain(self.walls.iter().map(|w| w.h))
            .fold(0.0_f64, f64::max);
        let neighbours = LinkedCellList::all_neighbors(&positions, 2.0 * h_max);
        let mass_of = |j: usize| -> f64 {
            if j < n_f {
                self.particles[j].mass
            } else {
                self.walls[j - n_f].mass
            }
        };
        let h_of = |j: usize| -> f64 {
            if j < n_f {
                self.particles[j].h
            } else {
                self.walls[j - n_f].h
            }
        };

        // ---- density summation and Tait EOS (fluid) --------------------------
        let mut rho = vec![0.0; n_f];
        let mut pressure = vec![0.0; n_f];
        for i in 0..n_f {
            let hi = self.particles[i].h;
            let mut sum = self.particles[i].mass * dim.w(0.0, hi);
            for &j in &neighbours[i] {
                let r = dist(positions[i], positions[j]);
                sum += mass_of(j) * dim.w(r, 0.5 * (hi + h_of(j)));
            }
            rho[i] = sum;
            pressure[i] = b_eos * ((sum / rho0).powf(gamma) - 1.0);
        }

        // Boussinesq body force (z component) per fluid particle.
        let body_z: Vec<f64> = self
            .particles
            .iter()
            .map(|p| -p.reduced_gravity(rho0, self.rho_sediment))
            .collect();

        // ---- wall pressure / density (Adami et al. 2012) ---------------------
        let mut wall_p = vec![0.0; n_w];
        let mut wall_rho = vec![rho0; n_w];
        for k in 0..n_w {
            let iw = n_f + k;
            let hw = self.walls[k].h;
            let mut num = 0.0;
            let mut den = 0.0;
            for &f in neighbours[iw].iter().filter(|&&f| f < n_f) {
                let r = dist(positions[iw], positions[f]);
                let w = dim.w(r, 0.5 * (hw + self.particles[f].h));
                let dz = positions[iw][2] - positions[f][2];
                num += (pressure[f] + rho[f] * body_z[f] * dz) * w;
                den += w;
            }
            if den > 0.0 {
                let p_w = num / den;
                wall_p[k] = p_w;
                wall_rho[k] = rho0 * (p_w / b_eos + 1.0).max(1e-6).powf(1.0 / gamma);
            }
        }

        // ---- accelerations and concentration rates ---------------------------
        let alpha = self.alpha_viscosity;
        let c0 = self.sound_speed;
        let eps_s = self.sediment_diffusivity;
        let mut acc = vec![[0.0; 3]; n_f];
        let mut dc = vec![0.0; n_f];
        let mut deposition_rate = 0.0;
        for i in 0..n_f {
            let pi = &self.particles[i];
            let pr_i = pressure[i] / (rho[i] * rho[i]);
            let vol_i = pi.mass / rho0;
            let mut a = [0.0; 3];
            for &j in &neighbours[i] {
                let rij = sub(positions[i], positions[j]);
                let r = norm(rij);
                let hij = 0.5 * (pi.h + h_of(j));
                if r <= 1e-12 * hij {
                    continue;
                }
                let dw = dim.dw_dr(r, hij);
                if dw == 0.0 {
                    continue;
                }
                let grad = [dw * rij[0] / r, dw * rij[1] / r, dw * rij[2] / r];
                let mj = mass_of(j);
                let vol_j = mj / rho0;
                if j < n_f {
                    let pj = &self.particles[j];
                    // Pressure.
                    let fac_p = pr_i + pressure[j] / (rho[j] * rho[j]);
                    // Monaghan artificial viscosity.
                    let vij = sub(pi.velocity, pj.velocity);
                    let vr = dot(vij, rij);
                    let pi_visc = if vr < 0.0 && alpha > 0.0 {
                        let mu = hij * vr / (r * r + 0.01 * hij * hij);
                        -alpha * c0 * mu / (0.5 * (rho[i] + rho[j]))
                    } else {
                        0.0
                    };
                    let fac = -mj * (fac_p + pi_visc);
                    for k in 0..3 {
                        a[k] += fac * grad[k];
                    }
                    // Turbulent diffusion (signed dW/dr; conservative pair form).
                    dc[i] += super::sph_diffusion_rate(
                        pi.concentration,
                        pj.concentration,
                        mj,
                        rho0,
                        eps_s,
                        r,
                        dw,
                    );
                    // Settling: conservative upwind flux, sediment falls from
                    // the higher particle into the lower one.
                    if w_s > 0.0 {
                        let c_up = if positions[j][2] > positions[i][2] {
                            pj.concentration
                        } else {
                            pi.concentration
                        };
                        dc[i] += 2.0 * w_s * vol_j * c_up * grad[2];
                    }
                } else {
                    let k_w = j - n_f;
                    let fac = -mj * (pr_i + wall_p[k_w] / (wall_rho[k_w] * wall_rho[k_w]));
                    for k in 0..3 {
                        a[k] += fac * grad[k];
                    }
                    // Settling into upward-facing walls (deposition).
                    let n = self.walls[k_w].normal;
                    let u_n = -w_s * n[2];
                    if u_n < 0.0 {
                        let flux_dot_grad = pi.concentration * u_n * dot(n, grad);
                        let rate = -2.0 * vol_j * flux_dot_grad;
                        dc[i] += rate;
                        deposition_rate -= vol_i * rate;
                    }
                }
            }
            let drag = drag_acceleration(pi.velocity, cd, pi.h);
            a[0] += drag[0];
            a[1] += drag[1];
            a[2] += drag[2] + body_z[i];
            acc[i] = a;
        }

        // ---- finiteness check before committing ------------------------------
        for i in 0..n_f {
            let bad = [
                ("density", rho[i]),
                ("acceleration", norm(acc[i])),
                ("concentration rate", dc[i]),
            ];
            for (name, value) in bad {
                if !value.is_finite() {
                    return Err(Error::NonFiniteQuantity {
                        quantity: name.to_string(),
                        particle_index: i,
                        value,
                    });
                }
            }
        }

        // ---- symplectic Euler update -----------------------------------------
        for ((p, a), dci) in self.particles.iter_mut().zip(&acc).zip(&dc) {
            for ((v, x), ak) in p.velocity.iter_mut().zip(p.position.iter_mut()).zip(a) {
                *v += dt * ak;
                *x += dt * *v;
            }
            p.concentration = (p.concentration + dt * dci).max(0.0);
        }
        self.deposited_volume += dt * deposition_rate;
        self.densities = rho;
        self.pressures = pressure;
        self.wall_pressures = wall_p;
        self.time += dt;
        Ok(())
    }
}

fn sub(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}

fn dot(a: [f64; 3], b: [f64; 3]) -> f64 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

fn norm(a: [f64; 3]) -> f64 {
    dot(a, a).sqrt()
}

fn dist(a: [f64; 3], b: [f64; 3]) -> f64 {
    norm(sub(a, b))
}

#[cfg(test)]
mod tests;
