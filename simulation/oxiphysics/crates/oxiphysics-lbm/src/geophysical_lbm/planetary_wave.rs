// Copyright 2026 COOLJAPAN OU (Team KitaSan)
// SPDX-License-Identifier: Apache-2.0

//! Two-layer rotating shallow-water model: barotropic and baroclinic
//! planetary waves.

use std::f64::consts::PI;

use crate::error::{Error, Result};

use super::shallow_water::ShallowWaterLbm;
use super::{NQ, guo_force, macroscopic, stream_periodic};

#[cfg(test)]
mod tests;

/// Smallest layer thickness for which a cell is still evolved.
const MIN_THICKNESS: f64 = 1e-12;

/// Two-layer rotating shallow-water lattice Boltzmann model.
///
/// A light upper layer (thickness `h1`) floats on a denser lower layer
/// (`h2`) over a flat bottom.  Each layer is a Zhou (2004) shallow-water
/// lattice (see [`ShallowWaterLbm`]) with the surface gravity `g`, whose
/// equilibrium already carries the *self* pressure force `−g hᵢ ∇hᵢ`.  The
/// layers interact through the hydrostatic pressure, which gives the
/// momentum equations
///
/// ```text
/// ∂t(h1 u1) + ∇·(h1 u1 ⊗ u1) = −g h1 ∇(h1 + h2)              − f k × h1 u1
/// ∂t(h2 u2) + ∇·(h2 u2 ⊗ u2) = −h2 (g ∇(h1 + h2) − g′ ∇h1)   − f k × h2 u2
/// ```
///
/// with the reduced gravity `g′ = g (ρ₂ − ρ₁)/ρ₂` (`0 ≤ g′ ≤ g`).  After
/// subtracting the self terms built into the equilibria, the Guo forces added
/// in [`step`](Self::step) are the cross terms `F1 = −g h1 ∇h2` and
/// `F2 = −(g − g′) h2 ∇h1` (central differences, periodic) plus the Coriolis
/// force with `f(y) = f0 + β (iy − ny/2)`.  The Coriolis term is evaluated with
/// the force-corrected velocity (a Crank–Nicolson solve of the 2 × 2 rotation),
/// so inertial oscillations keep their energy.
///
/// The linear long-wave modes without rotation are the two roots of
/// `c⁴ − g (h1 + h2) c² + g g′ h1 h2 = 0`: a fast *barotropic* mode with
/// `c ≈ √(g (h1 + h2))` (surface waves) and a slow *baroclinic* mode with
/// `c ≈ √(g′ h1 h2/(h1 + h2))` (interface waves); see
/// [`gravity_wave_speeds`](Self::gravity_wave_speeds).  The baroclinic
/// Rossby-wave phase speed is
/// [`baroclinic_phase_speed`](Self::baroclinic_phase_speed).
///
/// # Units and limits
///
/// Everything is in lattice units (`Δx = Δt = 1`).  The model is stable when
/// the surface wave speed `√(g (h1 + h2))` stays well below the lattice speed
/// (about `0.3` or less) and the layer thicknesses are of order one, see
/// [`check_parameters`](Self::check_parameters).  The domain is periodic in
/// both directions: with `β ≠ 0` the Coriolis parameter jumps across the
/// periodic seam `iy = 0 | ny − 1`, so β-plane disturbances should stay away
/// from it.
pub struct PlanetaryWave {
    /// Grid width.
    pub nx: usize,
    /// Grid height.
    pub ny: usize,
    /// Relaxation frequency for layer 1 (upper).
    pub omega1: f64,
    /// Relaxation frequency for layer 2 (lower).
    pub omega2: f64,
    /// Beta-plane parameter `β = df/dy` (per lattice spacing).
    pub beta: f64,
    /// Coriolis parameter `f0` at the centre row `iy = ny/2`; `0` (the
    /// default) places the equator there.
    pub f0: f64,
    /// Surface gravity `g` (lattice units); it sets the barotropic wave speed.
    pub gravity: f64,
    /// Reduced gravity `g′ = g (ρ₂ − ρ₁)/ρ₂` of the interface (baroclinic mode).
    pub g_prime: f64,
    /// Layer-1 (upper) rest thickness.
    pub h1: f64,
    /// Layer-2 (lower) rest thickness.
    pub h2: f64,
    /// Distribution functions layer 1.
    pub f1: Vec<Vec<[f64; NQ]>>,
    /// Distribution functions layer 2.
    pub f2: Vec<Vec<[f64; NQ]>>,
}

impl PlanetaryWave {
    /// Ratio `g / g′` of the surface gravity to the reduced gravity that
    /// [`new`](Self::new) assumes: a density contrast
    /// `(ρ₂ − ρ₁)/ρ₂ = g′/g` of 4 %.  Use [`with_gravity`](Self::with_gravity)
    /// to choose another value.
    pub const DEFAULT_GRAVITY_RATIO: f64 = 25.0;

    /// Construct a two-layer planetary wave model at rest.
    ///
    /// The surface gravity is `DEFAULT_GRAVITY_RATIO · g_prime` and `f0 = 0`
    /// (equatorial β-plane); see [`with_gravity`](Self::with_gravity) and
    /// [`with_coriolis`](Self::with_coriolis).
    pub fn new(
        nx: usize,
        ny: usize,
        omega1: f64,
        omega2: f64,
        beta: f64,
        g_prime: f64,
        h1: f64,
        h2: f64,
    ) -> Self {
        let gravity = Self::DEFAULT_GRAVITY_RATIO * g_prime;
        let f1 = vec![vec![ShallowWaterLbm::feq_sw(h1, 0.0, 0.0, gravity); ny]; nx];
        let f2 = vec![vec![ShallowWaterLbm::feq_sw(h2, 0.0, 0.0, gravity); ny]; nx];
        Self {
            nx,
            ny,
            omega1,
            omega2,
            beta,
            f0: 0.0,
            gravity,
            g_prime,
            h1,
            h2,
            f1,
            f2,
        }
    }

    /// Set the surface gravity and put both layers back to the rest state
    /// of that gravity (the equilibrium populations depend on `g`).
    pub fn with_gravity(mut self, gravity: f64) -> Self {
        self.gravity = gravity;
        self.reset_to_rest();
        self
    }

    /// Set the Coriolis parameter `f0` at the centre row.
    pub fn with_coriolis(mut self, f0: f64) -> Self {
        self.f0 = f0;
        self
    }

    /// Put both layers at rest with their rest thickness `h1`, `h2`.
    pub fn reset_to_rest(&mut self) {
        self.fill_rest(self.h1, self.h2, |_, _| (0.0, 0.0));
    }

    /// Fill both layers with the equilibria of the thicknesses
    /// `h1 + thickness(ix, iy).0` and `h2 + thickness(ix, iy).1`, at rest.
    fn fill_rest(&mut self, h1: f64, h2: f64, thickness: impl Fn(usize, usize) -> (f64, f64)) {
        let g = self.gravity;
        for ix in 0..self.nx {
            for iy in 0..self.ny {
                let (d1, d2) = thickness(ix, iy);
                self.f1[ix][iy] = ShallowWaterLbm::feq_sw(h1 + d1, 0.0, 0.0, g);
                self.f2[ix][iy] = ShallowWaterLbm::feq_sw(h2 + d2, 0.0, 0.0, g);
            }
        }
    }

    /// Validate the physical parameters: positive finite thicknesses, relaxation
    /// frequencies in `(0, 2)`, `0 ≤ g′ ≤ g` and finite rotation parameters.
    ///
    /// # Errors
    /// Returns the [`Error`] of the first parameter that is out of bounds.
    pub fn check_parameters(&self) -> Result<()> {
        Error::check_grid(self.nx, self.ny)?;
        for (omega, name) in [(self.omega1, "omega1"), (self.omega2, "omega2")] {
            Error::check_range(omega, f64::MIN_POSITIVE, 2.0, name)?;
            Error::check_tau(1.0 / omega)?;
        }
        Error::check_range(self.h1, f64::MIN_POSITIVE, f64::MAX, "h1")?;
        Error::check_range(self.h2, f64::MIN_POSITIVE, f64::MAX, "h2")?;
        Error::check_range(self.gravity, 0.0, f64::MAX, "gravity")?;
        Error::check_range(self.g_prime, 0.0, self.gravity, "g_prime")?;
        Error::check_finite(self.beta, "beta", 0)?;
        Error::check_finite(self.f0, "f0", 0)
    }

    /// Local Coriolis parameter `f = f0 + β (iy − ny/2)` at lattice row `iy`.
    #[inline]
    pub fn coriolis_at(&self, iy: usize) -> f64 {
        self.f0 + self.beta * (iy as f64 - 0.5 * self.ny as f64)
    }

    /// Lift layer 1 by `amplitude · cos(2π kx x / nx)` (`kx` cycles per domain,
    /// uniform in `y`), all at rest.
    ///
    /// The anomaly changes the free surface as well as the interface, so it
    /// excites both the barotropic and the baroclinic gravity-wave modes; see
    /// [`baroclinic_wave_ic`](Self::baroclinic_wave_ic) and
    /// [`barotropic_wave_ic`](Self::barotropic_wave_ic) for single-mode starts.
    pub fn kelvin_wave_ic(&mut self, amplitude: f64, kx: f64) {
        let nx = self.nx as f64;
        let (h1, h2) = (self.h1, self.h2);
        self.fill_rest(h1, h2, |ix, _| {
            let phase = (2.0 * PI * kx * ix as f64 / nx).cos();
            (amplitude * phase, 0.0)
        });
    }

    /// Start at rest from the baroclinic (interface) gravity-wave mode of the
    /// layer-1 thickness amplitude `amplitude` and `kx` cycles per domain.
    ///
    /// The thickness anomalies are the eigenvector of the slow root of
    /// [`gravity_wave_speeds`](Self::gravity_wave_speeds): the interface
    /// moves while the free surface stays (almost) flat.
    pub fn baroclinic_wave_ic(&mut self, amplitude: f64, kx: f64) {
        let c2 = self.gravity_wave_speeds().1.powi(2);
        self.mode_ic(c2, amplitude, kx);
    }

    /// Start at rest from the barotropic (surface) gravity-wave mode: both
    /// layers are stretched in proportion to their thickness.
    pub fn barotropic_wave_ic(&mut self, amplitude: f64, kx: f64) {
        let c2 = self.gravity_wave_speeds().0.powi(2);
        self.mode_ic(c2, amplitude, kx);
    }

    /// Thickness anomalies `(h1′, h2′)` of the long gravity-wave eigenmode
    /// with squared phase speed `c2` and `h1′ = amplitude`:
    /// `c² h1′ = g h1 η′`, `c² h2′ = h2 (g η′ − g′ h1′)` with
    /// `η′ = h1′ + h2′`.
    pub fn mode_thickness_anomalies(&self, c2: f64, amplitude: f64) -> (f64, f64) {
        let weight = self.h1 * self.gravity;
        let eta = if weight > 0.0 {
            c2 * amplitude / weight
        } else {
            amplitude
        };
        (amplitude, eta - amplitude)
    }

    fn mode_ic(&mut self, c2: f64, amplitude: f64, kx: f64) {
        let nx = self.nx as f64;
        let (h1, h2) = (self.h1, self.h2);
        let (_, a2) = self.mode_thickness_anomalies(c2, amplitude);
        let ratio = if amplitude != 0.0 {
            a2 / amplitude
        } else {
            0.0
        };
        self.fill_rest(h1, h2, |ix, _| {
            let phase = (2.0 * PI * kx * ix as f64 / nx).cos();
            (amplitude * phase, amplitude * ratio * phase)
        });
    }

    /// Phase speeds `(c_barotropic, c_baroclinic)` of the long gravity waves
    /// of the non-rotating two-layer system: the roots `c²` of
    /// `c⁴ − g (h1 + h2) c² + g g′ h1 h2 = 0`.
    ///
    /// For `g ≫ g′` they tend to `√(g (h1 + h2))` and
    /// `√(g′ h1 h2 / (h1 + h2))`.
    pub fn gravity_wave_speeds(&self) -> (f64, f64) {
        let g = self.gravity;
        let total = self.h1 + self.h2;
        let sum = g * total;
        let product = g * self.g_prime * self.h1 * self.h2;
        let discriminant = (sum * sum - 4.0 * product).max(0.0);
        let c2_fast = 0.5 * (sum + discriminant.sqrt());
        if c2_fast <= 0.0 {
            return (0.0, 0.0);
        }
        // The slow root from the product of the roots: no cancellation.
        let c2_slow = product / c2_fast;
        (c2_fast.sqrt(), c2_slow.max(0.0).sqrt())
    }

    /// `1/L²` of a mode with the squared wave speed `c2`: `f0²/c²` off the
    /// equator and `|β|/c` (the equatorial deformation scale `L² = c/|β|`)
    /// when `f0 = 0`.
    fn inverse_radius_squared(&self, c2: f64) -> f64 {
        if c2 <= 0.0 {
            return f64::INFINITY;
        }
        if self.f0.abs() > 1e-14 {
            self.f0 * self.f0 / c2
        } else {
            self.beta.abs() / c2.sqrt()
        }
    }

    /// Internal (baroclinic) deformation radius: `L_R = c₂/|f0|` for
    /// `|f0| > 1e-14`, else the equatorial scale `√(c₂/|β|)` with the
    /// baroclinic gravity-wave speed `c₂`.  Infinite when neither rotation nor
    /// β is present.
    pub fn baroclinic_deformation_radius(&self) -> f64 {
        let c2 = self.gravity_wave_speeds().1.powi(2);
        let inverse = self.inverse_radius_squared(c2);
        if inverse > 0.0 {
            (1.0 / inverse).sqrt()
        } else {
            f64::INFINITY
        }
    }

    /// Advance one timestep for both layers.
    ///
    /// Both depth fields are computed first; each layer then gets its
    /// interface-pressure cross force and the Coriolis force through the Guo
    /// scheme (with the force-corrected velocity in the Zhou equilibrium) and
    /// is streamed periodically.
    pub fn step(&mut self) {
        let (nx, ny) = (self.nx, self.ny);
        if nx == 0 || ny == 0 {
            return;
        }
        let depth1 = layer_depths(&self.f1);
        let depth2 = layer_depths(&self.f2);
        let (g, g_prime) = (self.gravity, self.g_prime);
        for ix in 0..nx {
            let ixp = (ix + 1) % nx;
            let ixm = (ix + nx - 1) % nx;
            for iy in 0..ny {
                let iyp = (iy + 1) % ny;
                let iym = (iy + ny - 1) % ny;
                let gradient = |depth: &[f64]| {
                    (
                        0.5 * (depth[ixp * ny + iy] - depth[ixm * ny + iy]),
                        0.5 * (depth[ix * ny + iyp] - depth[ix * ny + iym]),
                    )
                };
                let (dh1_dx, dh1_dy) = gradient(&depth1);
                let (dh2_dx, dh2_dy) = gradient(&depth2);
                let fc = self.coriolis_at(iy);
                // Upper layer: −g ∇h2 (its own −g ∇h1 is in the equilibrium).
                relax_layer(
                    &mut self.f1[ix][iy],
                    self.omega1,
                    g,
                    fc,
                    [-g * dh2_dx, -g * dh2_dy],
                );
                // Lower layer: −(g − g′) ∇h1 (its own −g ∇h2 is in the equilibrium).
                let coupling = g - g_prime;
                relax_layer(
                    &mut self.f2[ix][iy],
                    self.omega2,
                    g,
                    fc,
                    [-coupling * dh1_dx, -coupling * dh1_dy],
                );
            }
        }
        self.f1 = stream_periodic(&self.f1, nx, ny);
        self.f2 = stream_periodic(&self.f2, nx, ny);
    }

    /// Barotropic Rossby wave phase speed for wavenumber kx (non-divergent
    /// limit `−β/k²`).
    pub fn barotropic_phase_speed(&self, kx: f64) -> f64 {
        let k2 = kx * kx;
        if k2 < 1e-14 {
            return 0.0;
        }
        -self.beta / k2
    }

    /// Baroclinic Rossby wave phase speed `−β / (k² + 1/L_R²)` including the
    /// internal deformation radius `L_R`.
    ///
    /// With `c² = g′ h_eff` and `h_eff = h1 h2/(h1 + h2)` (exactly: the slow
    /// root of [`gravity_wave_speeds`](Self::gravity_wave_speeds)),
    /// `L_R² = c²/f0²` when `|f0| > 1e-14` and the equatorial
    /// `L² = c/|β|` otherwise.
    pub fn baroclinic_phase_speed(&self, kx: f64) -> f64 {
        let k2 = kx * kx;
        if k2 < 1e-14 {
            return 0.0;
        }
        let c2 = self.gravity_wave_speeds().1.powi(2);
        -self.beta / (k2 + self.inverse_radius_squared(c2))
    }

    /// Compute interface displacement (h1 - h1_mean) as proxy for baroclinicity.
    ///
    /// `h1` is the depth of the interface below the free surface, so for a
    /// flat surface this is the interface depth anomaly.
    pub fn interface_displacement(&self) -> Vec<Vec<f64>> {
        let h1 = self.h1;
        (0..self.nx)
            .map(|ix| {
                (0..self.ny)
                    .map(|iy| macroscopic(&self.f1[ix][iy]).0 - h1)
                    .collect()
            })
            .collect()
    }

    /// Free-surface elevation anomaly `h1 + h2 − (h1_mean + h2_mean)`.
    pub fn surface_elevation(&self) -> Vec<Vec<f64>> {
        let rest = self.h1 + self.h2;
        (0..self.nx)
            .map(|ix| {
                (0..self.ny)
                    .map(|iy| {
                        macroscopic(&self.f1[ix][iy]).0 + macroscopic(&self.f2[ix][iy]).0 - rest
                    })
                    .collect()
            })
            .collect()
    }

    /// Total mass `(Σ h1, Σ h2)` of the two layers (conserved to rounding).
    pub fn total_mass(&self) -> (f64, f64) {
        let sum = |f: &[Vec<[f64; NQ]>]| {
            f.iter()
                .flatten()
                .map(|cell| cell.iter().sum::<f64>())
                .sum()
        };
        (sum(&self.f1), sum(&self.f2))
    }
}

/// Layer depth `h = Σ f_q` of every cell, flattened as `ix * ny + iy`.
fn layer_depths(f: &[Vec<[f64; NQ]>]) -> Vec<f64> {
    f.iter()
        .flat_map(|column| column.iter().map(|cell| cell.iter().sum::<f64>()))
        .collect()
}

/// Force-corrected velocity `u = u0 + ½ a(u)` under the acceleration
/// `a = (ax, ay) + f (u_y, −u_x)` (Coriolis term solved implicitly).
fn corrected_velocity(u0x: f64, u0y: f64, f: f64, ax: f64, ay: f64) -> (f64, f64) {
    let bx = u0x + 0.5 * ax;
    let by = u0y + 0.5 * ay;
    let half = 0.5 * f;
    let det = 1.0 + half * half;
    ((bx + half * by) / det, (by - half * bx) / det)
}

/// BGK collision of one layer cell with the Zhou equilibrium and the Guo
/// forcing of the acceleration `accel` plus the Coriolis acceleration at
/// Coriolis parameter `f`.
fn relax_layer(cell: &mut [f64; NQ], omega: f64, gravity: f64, f: f64, accel: [f64; 2]) {
    let (h, u0x, u0y) = macroscopic(cell);
    if h <= MIN_THICKNESS {
        return;
    }
    let (ux, uy) = corrected_velocity(u0x, u0y, f, accel[0], accel[1]);
    // Force density = thickness × acceleration.
    let fx = h * (accel[0] + f * uy);
    let fy = h * (accel[1] - f * ux);
    let feq_arr = ShallowWaterLbm::feq_sw(h, ux, uy, gravity);
    for (f_q, feq_q) in cell.iter_mut().zip(feq_arr.iter()) {
        *f_q += omega * (*feq_q - *f_q);
    }
    guo_force(cell, fx, fy, ux, uy, omega);
}
