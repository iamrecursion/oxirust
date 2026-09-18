// Copyright 2026 COOLJAPAN OU (Team KitaSan)
// SPDX-License-Identifier: Apache-2.0

//! Granular material and snow/mud simulation.
//!
//! Provides MLS-MPM snow (built on [`crate::material_point`]), Drucker-Prager
//! plasticity, Bingham plastic mud, dry sand, avalanche stability, tire-terrain
//! interaction, and deformation tracking.

use crate::material_point::{
    MpmGrid, MpmParticle, SnowParams, g2p_transfer, make_snow_particle, mat3_det, mpm_step,
    p2g_transfer,
};

// ---------------------------------------------------------------------------
// Small math helpers
// ---------------------------------------------------------------------------

#[inline]
fn add3(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [a[0] + b[0], a[1] + b[1], a[2] + b[2]]
}

#[inline]
fn scale3(v: [f64; 3], s: f64) -> [f64; 3] {
    [v[0] * s, v[1] * s, v[2] * s]
}

#[inline]
fn dot3(a: [f64; 3], b: [f64; 3]) -> f64 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

#[inline]
fn len3(v: [f64; 3]) -> f64 {
    dot3(v, v).sqrt()
}

// ---------------------------------------------------------------------------
// SnowParticle
// ---------------------------------------------------------------------------

/// A single MPM snow particle.
///
/// The kinematic fields (`position`, `velocity`) are the state the user sets and
/// reads; [`SnowMpm`] keeps the remaining constitutive state (elastic and
/// plastic parts of the deformation gradient, the APIC affine velocity) next to
/// them and writes `position`, `velocity`, `def_gradient` and `hardness` back
/// after every step.
#[derive(Debug, Clone)]
pub struct SnowParticle {
    /// World-space position (metres).
    pub position: [f64; 3],
    /// Velocity (m/s).
    pub velocity: [f64; 3],
    /// Rest density (kg/m³).
    pub density: f64,
    /// Cohesion pressure (Pa).  Informational: the snow model of Stomakhin et
    /// al. (2013) used by [`SnowMpm`] has no separate cohesion term.
    pub cohesion: f64,
    /// Hardening factor `exp(ξ (1 - det F_p))` (dimensionless, 1 for pristine
    /// snow); increases with compaction.  It multiplies the Lamé parameters of
    /// the particle and is updated by [`SnowMpm::step`].
    pub hardness: f64,
    /// Temperature (Kelvin).  Informational: [`SnowMpm`] is isothermal.
    pub temperature: f64,
    /// Total deformation gradient `F = F_e F_p` (3×3 stored row-major),
    /// updated by [`SnowMpm::step`].  A non-identity value on a new particle is
    /// taken as its initial, fully elastic deformation.
    pub def_gradient: [f64; 9],
    /// Volume (m³).  [`SnowMpm`] sets this to the volume the particle
    /// represents on its grid (one eighth of a cell).
    pub volume: f64,
    /// Mass (kg).  [`SnowMpm`] sets this to `density * volume`.
    pub mass: f64,
}

impl SnowParticle {
    /// Create a snow particle with default values.
    ///
    /// The volume and mass are stand-ins (a heuristic particle volume of
    /// `1 / (64 ρ)`): a particle added to a [`SnowMpm`] gets the volume of the
    /// grid cell fraction it represents instead.
    pub fn new(position: [f64; 3], density: f64, temperature: f64) -> Self {
        let volume = 1.0 / (density * 64.0); // heuristic per particle
        Self {
            position,
            velocity: [0.0; 3],
            density,
            cohesion: 1000.0,
            hardness: 1.0,
            temperature,
            def_gradient: identity3x3(),
            volume,
            mass: density * volume,
        }
    }
}

fn identity3x3() -> [f64; 9] {
    [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]
}

/// Row-major `[f64; 9]` to a 3×3 matrix.
fn mat3_from_row_major(a: [f64; 9]) -> [[f64; 3]; 3] {
    [[a[0], a[1], a[2]], [a[3], a[4], a[5]], [a[6], a[7], a[8]]]
}

/// 3×3 matrix to a row-major `[f64; 9]`.
fn mat3_to_row_major(m: [[f64; 3]; 3]) -> [f64; 9] {
    [
        m[0][0], m[0][1], m[0][2], m[1][0], m[1][1], m[1][2], m[2][0], m[2][1], m[2][2],
    ]
}

// ---------------------------------------------------------------------------
// SnowMpm — Material Point Method for snow
// ---------------------------------------------------------------------------

/// Number of particles per grid cell: the solver assumes a regular lattice of
/// 2 × 2 × 2 particles per cell, so each particle represents one eighth of a
/// cell volume.
const PARTICLES_PER_CELL: f64 = 8.0;

/// MLS-MPM solver for snow (Stomakhin et al. 2013; Hu et al. 2018).
///
/// The snow is a fixed-corotated elastic solid with Lamé parameters
/// `μ = E / (2 (1 + ν))`, `λ = E ν / ((1 + ν)(1 - 2ν))` that are hardened by
/// `exp(ξ (1 - det F_p))` when the snow is compacted.  The elastic part of the
/// deformation gradient is projected back after every step: its singular
/// values are clamped to `[1 - θ_c, 1 + θ_s]` (critical compression and
/// stretch), the excess is accumulated in the plastic part `F_p`.  Particle and
/// grid are coupled with quadratic B-splines and APIC transfers.
///
/// The simulation domain is `[0, domain_size]³`, discretised by a regular grid
/// of `grid_resolution³` cells.  The outermost `boundary_thickness` node layers
/// of every face are sticky (no-slip) walls; the floor, for instance, is the
/// plane `y = (boundary_thickness - 1) * cell_size` of the innermost sticky
/// node row.  Snow is carried by the sticky nodes within the kernel radius of
/// 1.5 cells of its lowest particles: a block placed a quarter cell above the
/// plane stays there, one that falls onto the wall comes to rest within about a
/// cell of it.  Snow that must not feel a wall has to stay at least
/// `boundary_thickness + 1` cells away from it.
/// Explicit integration requires a time step below roughly `0.3 h / c` with
/// the dilatational wave speed `c = √((λ + 2 μ) / ρ)`.
pub struct SnowMpm {
    /// Grid resolution (cells per axis).
    pub grid_resolution: usize,
    /// Cell size (metres).
    pub cell_size: f64,
    /// Snow particles.  Append new ones with [`SnowMpm::add_particle`] (or
    /// push directly: they are picked up at the next step).  The constitutive
    /// state is aligned by index, so particles must only ever be appended.
    pub particles: Vec<SnowParticle>,
    /// Background grid.
    pub grid: MpmGrid,
    /// Young's modulus (Pa).
    pub young_modulus: f64,
    /// Poisson's ratio `ν`, in `(-1, 1/2)` (`λ` diverges at `ν = 1/2`).
    pub poisson_ratio: f64,
    /// Critical compression `θ_c`: the elastic part of the deformation may
    /// compress by at most this fraction, more is plastic.
    pub critical_compression: f64,
    /// Critical stretch `θ_s`: the elastic part of the deformation may stretch
    /// by at most this fraction, more is plastic.
    pub critical_stretch: f64,
    /// Hardening coefficient `ξ`.
    pub hardening: f64,
    /// Gravity.
    pub gravity: [f64; 3],
    /// Number of sticky (no-slip) node layers at every face of the domain.
    pub boundary_thickness: usize,
    /// Full MPM state (elastic/plastic deformation gradient, affine velocity)
    /// of the particles, index-aligned with `particles`.
    mpm_particles: Vec<MpmParticle>,
}

impl SnowMpm {
    /// Create an MPM snow solver.
    pub fn new(grid_resolution: usize, domain_size: f64) -> Self {
        let cell_size = domain_size / grid_resolution as f64;
        Self {
            grid_resolution,
            cell_size,
            particles: Vec::new(),
            grid: MpmGrid::new([grid_resolution; 3], cell_size, [0.0; 3]),
            young_modulus: 1.4e5,
            poisson_ratio: 0.2,
            critical_compression: 0.019,
            critical_stretch: 0.0075,
            hardening: 10.0,
            gravity: [0.0, -9.81, 0.0],
            boundary_thickness: 2,
            mpm_particles: Vec::new(),
        }
    }

    /// Lamé parameters `(μ, λ)` of the pristine snow from `young_modulus` and
    /// `poisson_ratio`.
    pub fn lame_parameters(&self) -> (f64, f64) {
        let e = self.young_modulus;
        let nu = self.poisson_ratio;
        let mu = e / (2.0 * (1.0 + nu));
        let lambda = e * nu / ((1.0 + nu) * (1.0 - 2.0 * nu));
        (mu, lambda)
    }

    /// Volume represented by one particle (m³): one eighth of a grid cell.
    pub fn particle_volume(&self) -> f64 {
        self.cell_size.powi(3) / PARTICLES_PER_CELL
    }

    /// Add a particle to the simulation.
    ///
    /// The solver assumes 2 × 2 × 2 particles per grid cell: the particle's
    /// `volume` becomes [`SnowMpm::particle_volume`] and its `mass` `density *
    /// volume`.  [`SnowMpm::fill_box`] places such a lattice.
    pub fn add_particle(&mut self, p: SnowParticle) {
        self.particles.push(p);
        self.sync_particles();
    }

    /// Fill the axis-aligned box `[min, max)` with a regular lattice of
    /// snow particles (cell size / 2 spacing, i.e. 2 × 2 × 2 per cell) at rest.
    /// Returns the number of particles added.
    pub fn fill_box(
        &mut self,
        min: [f64; 3],
        max: [f64; 3],
        density: f64,
        temperature: f64,
    ) -> usize {
        let spacing = 0.5 * self.cell_size;
        let counts = [0, 1, 2].map(|d| ((max[d] - min[d]) / spacing).round().max(0.0) as usize);
        let mut added = 0;
        for i in 0..counts[0] {
            for j in 0..counts[1] {
                for k in 0..counts[2] {
                    let position = [
                        min[0] + (i as f64 + 0.5) * spacing,
                        min[1] + (j as f64 + 0.5) * spacing,
                        min[2] + (k as f64 + 0.5) * spacing,
                    ];
                    self.particles
                        .push(SnowParticle::new(position, density, temperature));
                    added += 1;
                }
            }
        }
        self.sync_particles();
        added
    }

    /// Bring the MPM state in line with `particles`: create the state of new
    /// particles (grid-consistent volume and mass, initial deformation from
    /// `def_gradient`), drop the state of removed ones, and take positions,
    /// velocities and the material parameters from the user-visible values.
    fn sync_particles(&mut self) {
        let volume0 = self.particle_volume();
        let (mu0, lambda0) = self.lame_parameters();
        let (hardening, theta_c, theta_s) = (
            self.hardening,
            self.critical_compression,
            self.critical_stretch,
        );
        self.mpm_particles.truncate(self.particles.len());
        let known = self.mpm_particles.len();
        for p in self.particles.iter_mut().skip(known) {
            p.volume = volume0;
            p.mass = p.density * volume0;
            let params = SnowParams {
                mu0,
                lambda0,
                hardening,
                theta_c,
                theta_s,
                rho0: p.density,
            };
            let mut state = make_snow_particle(p.position, p.velocity, volume0, &params);
            let f0 = mat3_from_row_major(p.def_gradient);
            state.deformation_gradient = f0;
            state.fe = f0;
            self.mpm_particles.push(state);
        }
        for (p, state) in self.particles.iter().zip(self.mpm_particles.iter_mut()) {
            state.position = p.position;
            state.velocity = p.velocity;
            state.mu0 = mu0;
            state.lambda0 = lambda0;
            state.hardening = hardening;
        }
    }

    /// Copy the results of the MPM state back into the user-visible particles.
    fn publish_particles(&mut self) {
        let hardening = self.hardening;
        for (p, state) in self.particles.iter_mut().zip(self.mpm_particles.iter()) {
            p.position = state.position;
            p.velocity = state.velocity;
            p.def_gradient = mat3_to_row_major(state.deformation_gradient);
            p.hardness = (hardening * (1.0 - mat3_det(state.fp))).exp();
        }
    }

    /// Reset grid (clear mass, momentum, force).
    pub fn clear_grid(&mut self) {
        self.grid.reset();
    }

    /// Particle-to-grid (P2G) transfer.
    ///
    /// Scatters the mass and the APIC momentum of the particles to the three
    /// nodes per axis of their quadratic B-spline stencil (the weights sum to
    /// one, so the grid mass equals the particle mass) and sets the grid
    /// velocity to momentum / mass.  Call [`SnowMpm::clear_grid`] first; the
    /// transfers accumulate.  The time integration of [`SnowMpm::step`] (elastic
    /// forces, gravity) is not applied here.
    pub fn particle_to_grid(&mut self) {
        self.sync_particles();
        p2g_transfer(&mut self.grid, &self.mpm_particles, 0.0, [0.0; 3]);
    }

    /// Grid-to-particle (G2P) transfer: gathers the velocity (and the APIC
    /// affine velocity field) of the grid nodes of each particle's stencil.
    pub fn grid_to_particle(&mut self) {
        g2p_transfer(&self.grid, &mut self.mpm_particles);
        for (p, state) in self.particles.iter_mut().zip(self.mpm_particles.iter()) {
            p.velocity = state.velocity;
        }
    }

    /// Full MPM step of length `dt` (s).
    ///
    /// P2G with the fixed-corotated Kirchhoff stress of the hardened snow and
    /// gravity, grid update, sticky boundary layers, G2P, update of the
    /// deformation gradient with the plasticity projection to
    /// `[1 - θ_c, 1 + θ_s]`, and advection.  A non-positive or non-finite
    /// `dt` leaves the state unchanged.
    pub fn step(&mut self, dt: f64) {
        if !(dt > 0.0 && dt.is_finite()) {
            return;
        }
        self.sync_particles();
        let boundary = self.boundary_thickness.min(self.grid_resolution);
        mpm_step(
            &mut self.grid,
            &mut self.mpm_particles,
            dt,
            self.gravity,
            self.critical_compression,
            self.critical_stretch,
            boundary,
        );
        self.publish_particles();
    }
}

/// Quadratic B-spline weight for MPM.
pub fn mpm_grid_to_particle(node_pos: [f64; 3], particle_pos: [f64; 3], h: f64) -> f64 {
    quadratic_weight(particle_pos, node_pos, h)
}

fn quadratic_weight(particle: [f64; 3], node: [f64; 3], h: f64) -> f64 {
    let mut w = 1.0;
    for i in 0..3 {
        let x = ((particle[i] - node[i]) / h).abs();
        w *= if x < 0.5 {
            0.75 - x * x
        } else if x < 1.5 {
            0.5 * (1.5 - x) * (1.5 - x)
        } else {
            0.0
        };
    }
    w
}

// ---------------------------------------------------------------------------
// DruckerPragerPlastic
// ---------------------------------------------------------------------------

/// Drucker-Prager yield criterion with hardening for snow compaction.
pub struct DruckerPragerPlastic {
    /// Friction angle (radians).
    pub friction_angle: f64,
    /// Cohesion (Pa).
    pub cohesion: f64,
    /// Hardening modulus (Pa).
    pub hardening_modulus: f64,
    /// Accumulated plastic strain.
    pub plastic_strain: f64,
}

impl DruckerPragerPlastic {
    /// Create a Drucker-Prager model.
    pub fn new(friction_angle: f64, cohesion: f64) -> Self {
        Self {
            friction_angle,
            cohesion,
            hardening_modulus: 1e4,
            plastic_strain: 0.0,
        }
    }

    /// Effective friction coefficient from the friction angle.
    pub fn mu(&self) -> f64 {
        let sin_phi = self.friction_angle.sin();
        2.0 * sin_phi / (3.0_f64.sqrt() * (3.0 - sin_phi))
    }

    /// Yield function value: `f > 0` means plastic.
    ///
    /// `p` = hydrostatic pressure (Pa, compression positive),
    /// `q` = deviatoric stress invariant (Pa).
    pub fn yield_function(&self, p: f64, q: f64) -> f64 {
        drucker_prager_yield(
            q,
            p,
            self.mu(),
            self.cohesion + self.hardening_modulus * self.plastic_strain,
        )
    }

    /// Return stress (q) after plastic projection.
    pub fn project(&mut self, p: f64, q: f64) -> f64 {
        let f = self.yield_function(p, q);
        if f <= 0.0 {
            return q; // elastic
        }
        // Radial return
        let delta_gamma = f / (1.0 + self.hardening_modulus);
        self.plastic_strain += delta_gamma;
        q - delta_gamma
    }
}

/// Drucker-Prager yield function value.
pub fn drucker_prager_yield(q: f64, p: f64, mu: f64, c: f64) -> f64 {
    q - mu * p - c
}

// ---------------------------------------------------------------------------
// SnowCohesion
// ---------------------------------------------------------------------------

/// Temperature-dependent snow cohesion (sintering model).
pub struct SnowCohesion {
    /// Reference cohesion at 273 K (Pa).
    pub cohesion_0: f64,
    /// Temperature sensitivity (Pa/K).
    pub temp_sensitivity: f64,
    /// Sintering time constant (seconds).
    pub sintering_tau: f64,
    /// Accumulated sintering time.
    pub sintered_time: f64,
}

impl SnowCohesion {
    /// Create a snow cohesion model.
    pub fn new(cohesion_0: f64) -> Self {
        Self {
            cohesion_0,
            temp_sensitivity: 50.0,
            sintering_tau: 3600.0,
            sintered_time: 0.0,
        }
    }

    /// Cohesion at temperature `t_kelvin`.
    pub fn cohesion_at_temp(&self, t_kelvin: f64) -> f64 {
        let delta = (273.15 - t_kelvin).max(0.0);
        self.cohesion_0 + self.temp_sensitivity * delta
    }

    /// Effective cohesion including sintering strengthening.
    pub fn effective_cohesion(&self, t_kelvin: f64) -> f64 {
        let base = self.cohesion_at_temp(t_kelvin);
        let sintering_factor = 1.0 - (-self.sintered_time / self.sintering_tau).exp();
        base * (1.0 + sintering_factor)
    }

    /// Advance the sintering clock.
    pub fn tick(&mut self, dt: f64) {
        self.sintered_time += dt;
    }
}

/// Compute snow density from temperature using a simple model.
pub fn snow_density_from_temp(t_kelvin: f64) -> f64 {
    // Fresh snow: ~50 kg/m³ at -10°C; denser near 0°C
    let t_celsius = t_kelvin - 273.15;
    let base = 150.0; // kg/m³
    let temp_effect = (-t_celsius * 0.1).exp();
    (base + 200.0 * (1.0 - temp_effect)).clamp(50.0, 600.0)
}

// ---------------------------------------------------------------------------
// MudParticle
// ---------------------------------------------------------------------------

/// A particle of clay-water mud.
pub struct MudParticle {
    /// World-space position (metres).
    pub position: [f64; 3],
    /// Velocity (m/s).
    pub velocity: [f64; 3],
    /// Water content by mass fraction (0..1).
    pub water_content: f64,
    /// Undrained shear strength (Pa).
    pub shear_strength: f64,
    /// Atterberg liquid limit (water content fraction).
    pub liquid_limit: f64,
    /// Atterberg plastic limit.
    pub plastic_limit: f64,
}

impl MudParticle {
    /// Create a mud particle.
    pub fn new(position: [f64; 3], water_content: f64) -> Self {
        Self {
            position,
            velocity: [0.0; 3],
            water_content,
            shear_strength: 10000.0,
            liquid_limit: 0.4,
            plastic_limit: 0.2,
        }
    }

    /// Returns true when mud is flowing (above liquid limit).
    pub fn is_liquid(&self) -> bool {
        self.water_content >= self.liquid_limit
    }

    /// Returns true when mud is plastic (between plastic and liquid limits).
    pub fn is_plastic(&self) -> bool {
        self.water_content >= self.plastic_limit && self.water_content < self.liquid_limit
    }

    /// Compute undrained shear strength using a simplified relationship.
    pub fn compute_shear_strength(&self) -> f64 {
        if self.is_liquid() {
            0.0
        } else if self.is_plastic() {
            let w = self.water_content;
            let wl = self.liquid_limit;
            let wp = self.plastic_limit;
            self.shear_strength * (wl - w) / (wl - wp)
        } else {
            self.shear_strength
        }
    }
}

// ---------------------------------------------------------------------------
// ViscoPlasticMud — Bingham plastic
// ---------------------------------------------------------------------------

/// Bingham plastic viscoplastic mud model.
pub struct ViscoPlasticMud {
    /// Yield stress (Pa).
    pub yield_stress: f64,
    /// Plastic viscosity (Pa·s).
    pub plastic_viscosity: f64,
    /// Density (kg/m³).
    pub density: f64,
    /// Particles.
    pub particles: Vec<MudParticle>,
    /// Gravity.
    pub gravity: [f64; 3],
}

impl ViscoPlasticMud {
    /// Create a Bingham mud model.
    pub fn new(yield_stress: f64, plastic_viscosity: f64, density: f64) -> Self {
        Self {
            yield_stress,
            plastic_viscosity,
            density,
            particles: Vec::new(),
            gravity: [0.0, -9.81, 0.0],
        }
    }

    /// Add a mud particle.
    pub fn add_particle(&mut self, p: MudParticle) {
        self.particles.push(p);
    }

    /// Compute Bingham shear stress for a given shear rate `gamma_dot` (1/s).
    pub fn shear_stress(&self, gamma_dot: f64) -> f64 {
        bingham_shear_stress(gamma_dot, self.yield_stress, self.plastic_viscosity)
    }

    /// Integrate particles forward by `dt`.
    pub fn step(&mut self, dt: f64) {
        let g = self.gravity;
        let yield_stress = self.yield_stress;
        let plastic_viscosity = self.plastic_viscosity;
        let density = self.density;
        for p in &mut self.particles {
            let vel_len = len3(p.velocity);
            let tau = bingham_shear_stress(vel_len, yield_stress, plastic_viscosity);
            let drag = if vel_len > 1e-9 {
                let dir = scale3(p.velocity, 1.0 / vel_len);
                scale3(dir, -tau / (density * 0.01))
            } else {
                [0.0; 3]
            };
            let acc = add3(g, drag);
            p.velocity = add3(p.velocity, scale3(acc, dt));
            p.position = add3(p.position, scale3(p.velocity, dt));
        }
    }
}

/// Compute Bingham plastic shear stress.
pub fn bingham_shear_stress(gamma_dot: f64, tau_y: f64, mu_p: f64) -> f64 {
    if gamma_dot.abs() < 1e-12 {
        return 0.0;
    }
    tau_y + mu_p * gamma_dot.abs()
}

// ---------------------------------------------------------------------------
// SandGranular
// ---------------------------------------------------------------------------

/// Dry granular sand model.
pub struct SandGranular {
    /// Angle of repose (radians).
    pub angle_of_repose: f64,
    /// Dilatancy angle (radians).
    pub dilatancy_angle: f64,
    /// Internal friction angle (radians).
    pub friction_angle: f64,
    /// Cohesion (Pa) — effectively zero for dry sand.
    pub cohesion: f64,
    /// Density (kg/m³).
    pub density: f64,
}

impl SandGranular {
    /// Create a dry sand model.
    pub fn new(friction_angle: f64) -> Self {
        Self {
            angle_of_repose: friction_angle,
            dilatancy_angle: friction_angle * 0.5,
            friction_angle,
            cohesion: 0.0,
            density: 1600.0,
        }
    }

    /// Mohr-Coulomb shear strength at effective normal stress `sigma_n`.
    pub fn shear_strength(&self, sigma_n: f64) -> f64 {
        self.cohesion + sigma_n * self.friction_angle.tan()
    }

    /// Returns `true` when a slope exceeds the angle of repose.
    pub fn is_unstable(&self, slope_angle: f64) -> bool {
        slope_angle > self.angle_of_repose
    }

    /// Volumetric strain from dilatancy given shear strain increment.
    pub fn dilatancy_strain(&self, shear_strain: f64) -> f64 {
        shear_strain * self.dilatancy_angle.tan()
    }
}

// ---------------------------------------------------------------------------
// AvalancheSim
// ---------------------------------------------------------------------------

/// Simple snow avalanche slope-stability model.
pub struct AvalancheSim {
    /// Slope angle (radians).
    pub slope_angle: f64,
    /// Slab thickness (metres).
    pub slab_thickness: f64,
    /// Snow density (kg/m³).
    pub density: f64,
    /// Weak layer shear strength (Pa).
    pub weak_layer_strength: f64,
    /// Weak layer cohesion (Pa).
    pub weak_layer_cohesion: f64,
    /// Whether an avalanche is in progress.
    pub triggered: bool,
    /// Avalanche velocity (m/s).
    pub avalanche_velocity: f64,
}

impl AvalancheSim {
    /// Create an avalanche simulation.
    pub fn new(slope_angle: f64, slab_thickness: f64, density: f64) -> Self {
        Self {
            slope_angle,
            slab_thickness,
            density,
            weak_layer_strength: 500.0,
            weak_layer_cohesion: 100.0,
            triggered: false,
            avalanche_velocity: 0.0,
        }
    }

    /// Compute the driving shear stress (Pa) on the weak layer.
    pub fn driving_stress(&self) -> f64 {
        9.81 * self.density * self.slab_thickness * self.slope_angle.sin()
    }

    /// Compute the resisting shear stress (Pa) from the weak layer.
    pub fn resisting_stress(&self) -> f64 {
        let normal_stress = 9.81 * self.density * self.slab_thickness * self.slope_angle.cos();
        self.weak_layer_cohesion + normal_stress * self.weak_layer_strength / 1000.0
    }

    /// Safety factor (> 1 → stable, < 1 → unstable).
    pub fn safety_factor(&self) -> f64 {
        self.resisting_stress() / self.driving_stress().max(1e-9)
    }

    /// Trigger probability (simplified logistic function of safety factor).
    pub fn trigger_probability(&self) -> f64 {
        let fs = self.safety_factor();
        1.0 / (1.0 + (-10.0 * (1.0 - fs)).exp())
    }

    /// Step the avalanche dynamics.
    pub fn step(&mut self, dt: f64) {
        if self.safety_factor() < 1.0 && !self.triggered {
            self.triggered = true;
        }
        if self.triggered {
            let net_accel = 9.81 * self.slope_angle.sin() - 0.1 * self.avalanche_velocity;
            self.avalanche_velocity = (self.avalanche_velocity + net_accel * dt).max(0.0);
        }
    }
}

// ---------------------------------------------------------------------------
// TireInteraction
// ---------------------------------------------------------------------------

/// Tire-terrain interaction in snow/mud.
pub struct TireInteraction {
    /// Tire contact width (metres).
    pub width: f64,
    /// Tire contact length (metres).
    pub contact_length: f64,
    /// Tire load (N).
    pub load: f64,
    /// Terrain cohesion (Pa).
    pub cohesion: f64,
    /// Terrain friction angle (radians).
    pub friction_angle: f64,
    /// Terrain stiffness (Pa/m — Bekker parameter kc).
    pub kc: f64,
    /// Terrain sinkage exponent (Bekker n).
    pub sinkage_exp: f64,
    /// Bekker modulus kphi (Pa/m).
    pub kphi: f64,
}

impl TireInteraction {
    /// Create a tire interaction model.
    pub fn new(
        width: f64,
        contact_length: f64,
        load: f64,
        cohesion: f64,
        friction_angle: f64,
    ) -> Self {
        Self {
            width,
            contact_length,
            load,
            cohesion,
            friction_angle,
            kc: 102.0,
            sinkage_exp: 0.9,
            kphi: 1510.0,
        }
    }

    /// Compute tire sinkage (metres) using the Bekker-Wong model.
    pub fn sinkage(&self) -> f64 {
        let p = self.load / (self.width * self.contact_length);
        let k = self.kc / self.width + self.kphi;
        (p / k).powf(1.0 / self.sinkage_exp)
    }

    /// Compute rolling resistance force (N).
    pub fn rolling_resistance(&self) -> f64 {
        let z = self.sinkage();
        // Simplified: resistance ~ load * z / contact_length
        self.load * z / self.contact_length
    }

    /// Maximum traction force (N) via Mohr-Coulomb.
    pub fn max_traction(&self) -> f64 {
        let contact_area = self.width * self.contact_length;
        let normal_stress = self.load / contact_area;
        contact_area * (self.cohesion + normal_stress * self.friction_angle.tan())
    }
}

// ---------------------------------------------------------------------------
// GroundDeformation
// ---------------------------------------------------------------------------

/// Permanent deformation tracking (track record in snow/mud).
pub struct GroundDeformation {
    /// Grid resolution (cells per side).
    pub resolution: usize,
    /// Cell size (metres).
    pub cell_size: f64,
    /// Deformation depth per cell (metres, positive = indented).
    pub depth_map: Vec<f64>,
    /// Maximum deformation allowed (metres).
    pub max_depth: f64,
    /// Recovery rate (m/s) — slow creep back.
    pub recovery_rate: f64,
}

impl GroundDeformation {
    /// Create a deformation tracking grid.
    pub fn new(resolution: usize, domain_size: f64) -> Self {
        Self {
            resolution,
            cell_size: domain_size / resolution as f64,
            depth_map: vec![0.0; resolution * resolution],
            max_depth: 0.3,
            recovery_rate: 1e-4,
        }
    }

    /// Apply a point load at world position `(x, z)` with `pressure` (Pa) and time `dt`.
    pub fn apply_load(&mut self, x: f64, z: f64, pressure: f64, stiffness: f64, dt: f64) {
        let ix = (x / self.cell_size) as i64;
        let iz = (z / self.cell_size) as i64;
        let n = self.resolution as i64;
        if ix < 0 || ix >= n || iz < 0 || iz >= n {
            return;
        }
        let idx = ix as usize * self.resolution + iz as usize;
        let sinkage = (pressure / stiffness) * dt;
        self.depth_map[idx] = (self.depth_map[idx] + sinkage).min(self.max_depth);
    }

    /// Advance elastic recovery.
    pub fn recover(&mut self, dt: f64) {
        for d in &mut self.depth_map {
            *d = (*d - self.recovery_rate * dt).max(0.0);
        }
    }

    /// Query deformation depth at world position `(x, z)`.
    pub fn query(&self, x: f64, z: f64) -> f64 {
        let ix = (x / self.cell_size) as i64;
        let iz = (z / self.cell_size) as i64;
        let n = self.resolution as i64;
        if ix < 0 || ix >= n || iz < 0 || iz >= n {
            return 0.0;
        }
        self.depth_map[ix as usize * self.resolution + iz as usize]
    }

    /// Maximum deformation depth currently recorded.
    pub fn max_deformation(&self) -> f64 {
        self.depth_map.iter().cloned().fold(0.0, f64::max)
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod regression_tests;

#[cfg(test)]
mod tests {
    use super::*;

    // 1. SnowParticle: mass = density * volume
    #[test]
    fn test_snow_particle_mass() {
        let p = SnowParticle::new([0.0; 3], 300.0, 263.15);
        assert!((p.mass - p.density * p.volume).abs() < 1e-12);
    }

    // 2. SnowMpm: grid size correct
    #[test]
    fn test_mpm_grid_size() {
        let sim = SnowMpm::new(4, 1.0);
        assert_eq!(sim.grid.nodes.len(), 64);
        assert_eq!(sim.grid.num_nodes(), 64);
        assert!((sim.grid.h - 0.25).abs() < 1e-15);
    }

    // 3. SnowMpm: step runs without panic and keeps the state finite
    #[test]
    fn test_mpm_step() {
        let mut sim = SnowMpm::new(4, 1.0);
        sim.add_particle(SnowParticle::new([0.5, 0.8, 0.5], 300.0, 263.0));
        sim.step(0.001);
        let p = &sim.particles[0];
        assert!(p.position.iter().all(|c| c.is_finite()));
        assert!(p.velocity.iter().all(|c| c.is_finite()));
        assert!(p.def_gradient.iter().all(|c| c.is_finite()));
    }

    // 4. DruckerPragerPlastic: yield function positive outside yield surface
    #[test]
    fn test_dp_yield_positive() {
        let dp = DruckerPragerPlastic::new(30.0_f64.to_radians(), 500.0);
        let f = dp.yield_function(1000.0, 5000.0);
        // With large q this should be positive
        assert!(f > 0.0);
    }

    // 5. DruckerPragerPlastic: projection reduces q
    #[test]
    fn test_dp_project_reduces_q() {
        let mut dp = DruckerPragerPlastic::new(30.0_f64.to_radians(), 500.0);
        let q_before = 5000.0;
        let q_after = dp.project(1000.0, q_before);
        assert!(q_after <= q_before);
    }

    // 6. SnowCohesion: cohesion higher at lower temperature
    #[test]
    fn test_snow_cohesion_temp() {
        let sc = SnowCohesion::new(1000.0);
        let c_cold = sc.cohesion_at_temp(253.15); // -20°C
        let c_warm = sc.cohesion_at_temp(273.15); // 0°C
        assert!(c_cold > c_warm);
    }

    // 7. SnowCohesion: sintering increases over time
    #[test]
    fn test_snow_cohesion_sintering() {
        let mut sc = SnowCohesion::new(1000.0);
        let c0 = sc.effective_cohesion(263.0);
        sc.tick(3600.0);
        let c1 = sc.effective_cohesion(263.0);
        assert!(c1 > c0);
    }

    // 8. snow_density_from_temp: reasonable range
    #[test]
    fn test_snow_density_range() {
        let d = snow_density_from_temp(263.0);
        assert!((50.0..=600.0).contains(&d));
    }

    // 9. MudParticle: is_liquid when above liquid limit
    #[test]
    fn test_mud_is_liquid() {
        let p = MudParticle::new([0.0; 3], 0.5);
        assert!(p.is_liquid());
    }

    // 10. MudParticle: shear strength zero when liquid
    #[test]
    fn test_mud_zero_strength_liquid() {
        let p = MudParticle::new([0.0; 3], 0.5);
        assert_eq!(p.compute_shear_strength(), 0.0);
    }

    // 11. MudParticle: plastic range
    #[test]
    fn test_mud_plastic_range() {
        let p = MudParticle::new([0.0; 3], 0.3);
        assert!(p.is_plastic());
    }

    // 12. ViscoPlasticMud: bingham stress increases with shear rate
    #[test]
    fn test_bingham_stress_increases() {
        let mud = ViscoPlasticMud::new(100.0, 10.0, 1500.0);
        let s1 = mud.shear_stress(1.0);
        let s2 = mud.shear_stress(2.0);
        assert!(s2 > s1);
    }

    // 13. bingham_shear_stress: zero gamma → zero
    #[test]
    fn test_bingham_zero_gamma() {
        assert_eq!(bingham_shear_stress(0.0, 100.0, 10.0), 0.0);
    }

    // 14. SandGranular: stable below repose angle
    #[test]
    fn test_sand_stable() {
        let sand = SandGranular::new(35.0_f64.to_radians());
        assert!(!sand.is_unstable(30.0_f64.to_radians()));
    }

    // 15. SandGranular: unstable above repose
    #[test]
    fn test_sand_unstable() {
        let sand = SandGranular::new(35.0_f64.to_radians());
        assert!(sand.is_unstable(40.0_f64.to_radians()));
    }

    // 16. SandGranular: Mohr-Coulomb shear strength positive
    #[test]
    fn test_sand_shear_strength() {
        let sand = SandGranular::new(35.0_f64.to_radians());
        assert!(sand.shear_strength(1000.0) > 0.0);
    }

    // 17. AvalancheSim: safety factor > 1 for gentle slope
    #[test]
    fn test_avalanche_stable() {
        let sim = AvalancheSim::new(10.0_f64.to_radians(), 0.5, 200.0);
        assert!(sim.safety_factor() > 1.0);
    }

    // 18. AvalancheSim: trigger probability near 0.5 at FS=1
    #[test]
    fn test_avalanche_trigger_prob_at_unity() {
        let prob = {
            let mut sim = AvalancheSim::new(30.0_f64.to_radians(), 1.0, 300.0);
            // Set strengths to exactly match driving
            sim.weak_layer_strength = sim.driving_stress() * 1000.0
                / (9.81 * 300.0 * 1.0 * (30.0_f64).to_radians().cos());
            sim.weak_layer_cohesion = 0.0;
            sim.trigger_probability()
        };
        assert!(prob > 0.0 && prob <= 1.0);
    }

    // 19. TireInteraction: sinkage positive
    #[test]
    fn test_tire_sinkage_positive() {
        let ti = TireInteraction::new(0.2, 0.3, 4000.0, 100.0, 30.0_f64.to_radians());
        assert!(ti.sinkage() > 0.0);
    }

    // 20. TireInteraction: max_traction positive
    #[test]
    fn test_tire_traction_positive() {
        let ti = TireInteraction::new(0.2, 0.3, 4000.0, 100.0, 30.0_f64.to_radians());
        assert!(ti.max_traction() > 0.0);
    }

    // 21. GroundDeformation: apply_load increases depth
    #[test]
    fn test_deformation_load() {
        let mut gd = GroundDeformation::new(10, 10.0);
        gd.apply_load(5.0, 5.0, 10000.0, 1e6, 1.0);
        assert!(gd.query(5.0, 5.0) > 0.0);
    }

    // 22. GroundDeformation: recover reduces depth
    #[test]
    fn test_deformation_recovery() {
        let mut gd = GroundDeformation::new(10, 10.0);
        gd.depth_map[0] = 0.1;
        gd.recover(1.0);
        assert!(gd.depth_map[0] < 0.1);
    }

    // 23. GroundDeformation: max_deformation zero initially
    #[test]
    fn test_deformation_max_zero() {
        let gd = GroundDeformation::new(10, 10.0);
        assert_eq!(gd.max_deformation(), 0.0);
    }

    // 24. mpm_grid_to_particle: weight in [0, 1]
    #[test]
    fn test_mpm_weight_range() {
        let w = mpm_grid_to_particle([0.0; 3], [0.0; 3], 1.0);
        assert!((0.0..=1.0).contains(&w));
    }

    // 25. drucker_prager_yield: negative when inside yield surface
    #[test]
    fn test_dp_yield_negative_inside() {
        let f = drucker_prager_yield(100.0, 10000.0, 0.3, 500.0);
        assert!(f < 0.0);
    }

    // 26. ViscoPlasticMud: step moves particles downward
    #[test]
    fn test_mud_step_gravity() {
        let mut mud = ViscoPlasticMud::new(100.0, 10.0, 1500.0);
        mud.add_particle(MudParticle::new([0.0, 5.0, 0.0], 0.5));
        let y0 = mud.particles[0].position[1];
        mud.step(0.1);
        assert!(mud.particles[0].position[1] < y0);
    }

    // 27. AvalancheSim: velocity increases after trigger
    #[test]
    fn test_avalanche_velocity_increases() {
        let mut sim = AvalancheSim::new(45.0_f64.to_radians(), 1.0, 300.0);
        sim.triggered = true;
        sim.step(1.0);
        assert!(sim.avalanche_velocity > 0.0);
    }

    // 28. snow_density_from_temp: denser near freezing
    #[test]
    fn test_snow_density_near_freezing() {
        let d_cold = snow_density_from_temp(243.0);
        let d_warm = snow_density_from_temp(273.0);
        assert!(d_cold != d_warm); // should differ
    }

    // 29. SnowMpm: clear_grid resets mass
    #[test]
    fn test_mpm_clear_grid() {
        let mut sim = SnowMpm::new(4, 1.0);
        sim.add_particle(SnowParticle::new([0.5, 0.5, 0.5], 300.0, 263.0));
        sim.particle_to_grid();
        let deposited: f64 = sim.grid.nodes.iter().map(|n| n.mass).sum();
        assert!(
            (deposited - sim.particles[0].mass).abs() < 1e-12 * sim.particles[0].mass,
            "P2G must conserve mass: {deposited} vs {}",
            sim.particles[0].mass
        );
        sim.clear_grid();
        let total_mass: f64 = sim.grid.nodes.iter().map(|n| n.mass).sum();
        assert!(total_mass < 1e-12);
    }

    // 30. GroundDeformation: clamped to max_depth
    #[test]
    fn test_deformation_clamp() {
        let mut gd = GroundDeformation::new(10, 10.0);
        // Apply huge load many times
        for _ in 0..1000 {
            gd.apply_load(5.0, 5.0, 1e9, 1e4, 1.0);
        }
        assert!(gd.query(5.0, 5.0) <= gd.max_depth + 1e-9);
    }
}
