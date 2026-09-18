// Copyright 2026 COOLJAPAN OU (Team KitaSan)
// SPDX-License-Identifier: Apache-2.0

//! Neutral atmospheric boundary layer with a log-law wall model.

use super::{CS2, MIRROR_Y, NQ, clamp, feq, guo_force, macroscopic, stream_walled_y};

/// von Kármán constant.
const VON_KARMAN: f64 = 0.41;

/// Height of the first cell centre above the wall plane (the wall plane lies
/// half-way between the bottom row and the reflected ghost row).
const FIRST_CELL_HEIGHT: f64 = 0.5;

/// Neutral atmospheric boundary layer with a rough-wall log-law stress model.
///
/// A Smagorinsky large-eddy closure sets the local viscosity.  There is no
/// thermal stratification, so this is a *neutral* boundary layer (no
/// Monin–Obukhov stability correction).  The vertical coordinate is `y`; the
/// domain is periodic in x with free-slip (specular) walls at the bottom and
/// top, located half a lattice spacing beyond the first and last row.
///
/// * **Wall model** — the bottom row feels the rough-wall stress of the
///   logarithmic law, `τ_w = ρ u*²` with `u* = κ |u| / ln(1 + z₁/z₀)` (`z₁ = ½`
///   the height of the first cell centre, `z₀` the roughness length), as the
///   body force density `F_x = −τ_w sign(u_x) / Δz` (`Δz = 1`).  A smaller
///   roughness length gives a smaller drag.
/// * **Geostrophic forcing** — the top row is relaxed towards the geostrophic
///   wind by the body force density `F_x = r ρ (u_geo − u_x)` with the
///   relaxation rate `r =` [`relax_rate`](Self::relax_rate).
///
/// Both forces are added with the Guo scheme, so they change the momentum and
/// leave the mass untouched.
pub struct AtmosphericBoundaryLayer {
    /// Grid width.
    pub nx: usize,
    /// Grid height.
    pub ny: usize,
    /// BGK relaxation frequency (background viscosity).
    pub omega_base: f64,
    /// Smagorinsky constant Cs.
    pub cs_smag: f64,
    /// Roughness length z0 in lattice units.
    pub z0: f64,
    /// Geostrophic wind speed (driving).
    pub u_geo: f64,
    /// Relaxation rate `r` of the top row towards the geostrophic wind
    /// (per time step); `0.01` unless changed.
    pub relax_rate: f64,
    /// Distribution functions.
    pub f: Vec<Vec<[f64; NQ]>>,
    /// Turbulent viscosity field.
    pub nu_t: Vec<Vec<f64>>,
}

impl AtmosphericBoundaryLayer {
    /// Construct a new ABL simulation domain with the flow initialised to the
    /// geostrophic wind.
    pub fn new(nx: usize, ny: usize, omega_base: f64, cs_smag: f64, z0: f64, u_geo: f64) -> Self {
        let f = vec![vec![feq(1.0, u_geo, 0.0); ny]; nx];
        let nu_t = vec![vec![0.0f64; ny]; nx];
        Self {
            nx,
            ny,
            omega_base,
            cs_smag,
            z0,
            u_geo,
            relax_rate: 0.01,
            f,
            nu_t,
        }
    }

    /// Velocity `(u_x, u_y)` at row `iy`; rows beyond a free-slip wall mirror
    /// the adjacent fluid row (`u_x` kept, `u_y` reversed).
    fn velocity_with_mirror(&self, ix: usize, iy: isize) -> (f64, f64) {
        if iy < 0 {
            let (_, ux, uy) = macroscopic(&self.f[ix][0]);
            (ux, -uy)
        } else if iy as usize >= self.ny {
            let (_, ux, uy) = macroscopic(&self.f[ix][self.ny - 1]);
            (ux, -uy)
        } else {
            let (_, ux, uy) = macroscopic(&self.f[ix][iy as usize]);
            (ux, uy)
        }
    }

    /// Compute local strain-rate magnitude |S| (central differences, periodic
    /// in x, mirrored at the free-slip walls).
    fn strain_rate(&self, ix: usize, iy: usize) -> f64 {
        let nx = self.nx;
        let ixp = (ix + 1) % nx;
        let ixm = (ix + nx - 1) % nx;
        let row = iy as isize;
        let (uxpx, uypx) = self.velocity_with_mirror(ixp, row);
        let (uxmx, uymx) = self.velocity_with_mirror(ixm, row);
        let (uxpy, uypy) = self.velocity_with_mirror(ix, row + 1);
        let (uxmy, uymy) = self.velocity_with_mirror(ix, row - 1);
        let s11 = 0.5 * (uxpx - uxmx);
        let s22 = 0.5 * (uypy - uymy);
        let s12 = 0.25 * ((uxpy - uxmy) + (uypx - uymx));
        (2.0 * (s11 * s11 + s22 * s22 + 2.0 * s12 * s12)).sqrt()
    }

    /// Update turbulent viscosity using Smagorinsky model.
    pub fn update_turbulent_viscosity(&mut self) {
        let cs = self.cs_smag;
        for ix in 0..self.nx {
            for iy in 0..self.ny {
                let smag = self.strain_rate(ix, iy);
                self.nu_t[ix][iy] = cs * cs * smag;
            }
        }
    }

    /// Friction velocity `u* = κ |u| / ln(1 + z₁/z₀)` of the logarithmic law
    /// for the first-cell velocity `u`.
    pub fn friction_velocity_for(&self, u: f64) -> f64 {
        let z0 = self.z0.max(1e-12);
        VON_KARMAN * u.abs() / (1.0 + FIRST_CELL_HEIGHT / z0).ln().max(1e-6)
    }

    /// Mean friction velocity over the bottom row of the current state.
    pub fn friction_velocity(&self) -> f64 {
        if self.nx == 0 || self.ny == 0 {
            return 0.0;
        }
        let sum: f64 = (0..self.nx)
            .map(|ix| self.friction_velocity_for(macroscopic(&self.f[ix][0]).1))
            .sum();
        sum / self.nx as f64
    }

    /// Total mass `Σ ρ` over the grid (conserved to rounding).
    pub fn total_mass(&self) -> f64 {
        self.f
            .iter()
            .flatten()
            .map(|cell| cell.iter().sum::<f64>())
            .sum()
    }

    /// Advance one timestep with Smagorinsky LES, the log-law wall stress and
    /// the geostrophic relaxation, then stream with free-slip walls.
    pub fn step(&mut self) {
        self.update_turbulent_viscosity();
        let (nx, ny) = (self.nx, self.ny);
        let omega_base = self.omega_base;
        let u_geo = self.u_geo;
        let relax_rate = self.relax_rate;
        for ix in 0..nx {
            for iy in 0..ny {
                let nu_t_local = self.nu_t[ix][iy];
                let nu_eff = (1.0 / omega_base - 0.5) * CS2 + nu_t_local;
                let omega_eff = clamp(1.0 / (nu_eff / CS2 + 0.5), 0.01, 1.99);
                let (rho, ux0, uy0) = macroscopic(&self.f[ix][iy]);
                // Body force density along x: geostrophic relaxation on the top
                // row, log-law wall stress on the bottom row.
                let mut fx = 0.0;
                if iy + 1 == ny {
                    fx += relax_rate * rho * (u_geo - ux0);
                }
                if iy == 0 {
                    let u_star = self.friction_velocity_for(ux0);
                    fx -= rho * u_star * u_star * ux0.signum();
                }
                let ux = if rho > 1e-12 {
                    ux0 + 0.5 * fx / rho
                } else {
                    0.0
                };
                let feq_arr = feq(rho, ux, uy0);
                let cell = &mut self.f[ix][iy];
                for (f_q, feq_q) in cell.iter_mut().zip(feq_arr.iter()) {
                    *f_q += omega_eff * (*feq_q - *f_q);
                }
                guo_force(cell, fx, 0.0, ux, uy0, omega_eff);
            }
        }
        self.f = stream_walled_y(&self.f, nx, ny, &MIRROR_Y);
    }

    /// Return vertically-averaged horizontal wind profile.
    pub fn wind_profile(&self) -> Vec<f64> {
        (0..self.ny)
            .map(|iy| {
                let ux_sum: f64 = (0..self.nx).map(|ix| macroscopic(&self.f[ix][iy]).1).sum();
                ux_sum / self.nx as f64
            })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::super::CX;
    use super::*;

    const NX: usize = 8;
    const NY: usize = 12;

    /// A layer at rest (`u = 0`) with the given roughness length and geostrophic
    /// wind, so that the forcing has something to accelerate.
    fn resting_layer(z0: f64, u_geo: f64) -> AtmosphericBoundaryLayer {
        let mut abl = AtmosphericBoundaryLayer::new(NX, NY, 1.0, 0.1, z0, u_geo);
        for column in abl.f.iter_mut() {
            for cell in column.iter_mut() {
                *cell = feq(1.0, 0.0, 0.0);
            }
        }
        abl
    }

    /// Total x-momentum `Σ c_x f`.
    fn x_momentum(abl: &AtmosphericBoundaryLayer) -> f64 {
        abl.f
            .iter()
            .flatten()
            .map(|cell| (0..NQ).map(|q| CX[q] * cell[q]).sum::<f64>())
            .sum()
    }

    #[test]
    fn geostrophic_forcing_adds_momentum_and_no_mass() {
        let u_geo = 0.05;
        let mut abl = resting_layer(0.01, u_geo);
        let mass_before = abl.total_mass();
        abl.step();
        // The top row is relaxed towards u_geo with F_x = r rho (u_geo - u):
        // the Guo scheme adds exactly F_x to the x-momentum of each top node.
        let expected = NX as f64 * abl.relax_rate * u_geo;
        let momentum = x_momentum(&abl);
        assert!(
            (momentum - expected).abs() < 1e-12,
            "x-momentum {momentum}, expected the forcing impulse {expected}"
        );
        assert!(
            (abl.total_mass() - mass_before).abs() < 1e-12 * mass_before,
            "the forcing must not create mass"
        );
        let (_, ux_top, _) = macroscopic(&abl.f[0][NY - 1]);
        assert!(
            ux_top > 0.0 && ux_top < u_geo,
            "the top row accelerates towards the wind: {ux_top}"
        );
    }

    #[test]
    fn log_law_wall_stress_decelerates_the_bottom_row() {
        let (u0, z0) = (0.05, 0.01);
        let mut abl = AtmosphericBoundaryLayer::new(NX, NY, 1.0, 0.1, z0, u0);
        let u_star = abl.friction_velocity_for(u0);
        let expected_u_star = 0.41 * u0 / (1.0 + 0.5 / z0).ln();
        assert!((u_star - expected_u_star).abs() < 1e-15, "u* = {u_star}");
        let momentum_before = x_momentum(&abl);
        abl.step();
        // Uniform wind u0 = u_geo: the top relaxation vanishes, the wall
        // removes tau_w = rho u*^2 per bottom node (Delta z = 1).
        let removed = momentum_before - x_momentum(&abl);
        let expected = NX as f64 * u_star * u_star;
        assert!(
            expected > 1e-6,
            "the log-law stress must be non-trivial: {expected}"
        );
        assert!(
            (removed - expected).abs() < 1e-12,
            "wall stress removed {removed} of x-momentum, expected {expected}"
        );
        let profile = abl.wind_profile();
        assert!(
            profile[0] < profile[NY / 2],
            "the bottom row must be slower than the interior: {profile:?}"
        );
    }

    #[test]
    fn friction_velocity_decreases_with_the_roughness_length() {
        let rough = AtmosphericBoundaryLayer::new(NX, NY, 1.0, 0.1, 0.1, 0.05);
        let smooth = AtmosphericBoundaryLayer::new(NX, NY, 1.0, 0.1, 0.001, 0.05);
        let u = 0.04;
        assert!(rough.friction_velocity_for(u) > smooth.friction_velocity_for(u));
        // The layer is initialised to the geostrophic wind.
        let expected = 0.41 * 0.05 / (1.0 + 0.5 / 0.1_f64).ln();
        assert!((rough.friction_velocity() - expected).abs() < 1e-13);
    }

    #[test]
    fn mass_is_conserved_while_the_wind_spins_up() {
        let mut abl = resting_layer(0.01, 0.05);
        let mass_before = abl.total_mass();
        for _ in 0..500 {
            abl.step();
        }
        let drift = (abl.total_mass() - mass_before).abs() / mass_before;
        assert!(drift < 1e-10, "relative mass drift {drift}");
        // The momentum comes from the forcing impulse r rho u_geo per step on
        // the top row (at most NX r u_geo per step), reduced by the wall stress.
        let max_impulse = NX as f64 * abl.relax_rate * 0.05 * 500.0;
        let momentum = x_momentum(&abl);
        assert!(
            momentum > 0.5 * max_impulse && momentum <= max_impulse,
            "x-momentum {momentum} must lie in (0.5, 1] of the forcing impulse {max_impulse}"
        );
    }

    #[test]
    fn top_row_approaches_the_geostrophic_wind() {
        let u_geo = 0.05;
        let mut abl = resting_layer(0.01, u_geo);
        abl.relax_rate = 0.05;
        let mut previous = 0.0;
        for block in 0..4 {
            for _ in 0..500 {
                abl.step();
            }
            let top = abl.wind_profile()[NY - 1];
            assert!(
                top > previous && top <= u_geo,
                "block {block}: top-row wind {top} must rise monotonically to {u_geo}"
            );
            previous = top;
        }
        assert!(
            previous > 0.9 * u_geo,
            "after 2000 steps the top row has reached {previous} of {u_geo}"
        );
        let profile = abl.wind_profile();
        assert!(
            profile.windows(2).all(|pair| pair[1] > pair[0]),
            "the wind increases with height: {profile:?}"
        );
    }

    #[test]
    fn smaller_roughness_length_gives_larger_near_wall_wind() {
        let near_wall_wind = |z0: f64| {
            let mut abl = resting_layer(z0, 0.05);
            abl.relax_rate = 0.05;
            for _ in 0..1500 {
                abl.step();
            }
            abl.wind_profile()[0]
        };
        let rough = near_wall_wind(0.1);
        let smooth = near_wall_wind(0.001);
        assert!(
            smooth > 1.05 * rough,
            "near-wall wind: z0 = 0.001 gives {smooth}, z0 = 0.1 gives {rough}"
        );
    }
}
