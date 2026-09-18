// Copyright 2026 COOLJAPAN OU (Team KitaSan)
// SPDX-License-Identifier: Apache-2.0

//! Free-surface LBM using Volume-of-Fluid (VOF) mass tracking.
//!
//! Implements the Körner 2005 / Thürey VOF scheme for 2D free-surface LBM with
//! D2Q9. Provides a parallel side state (`FreeSurfaceState`) to `CellGrid2D`
//! that tracks per-cell fill levels, masses, and cell state flags
//! (Gas / Interface / Fluid).
//!
//! ## Time step
//! 1. [`apply_body_force_2d`] — half-step Guo shift of the stored velocities;
//! 2. [`free_surface_bgk_stream`] — BGK collision with the Guo force on liquid
//!    cells, periodic pull streaming with half-way bounce-back at obstacle
//!    cells, moments;
//! 3. [`free_surface_step`] — gas-side reconstruction of the Interface cells,
//!    mass exchange (on the *post-stream* populations), fill update, flag
//!    conversion and the interface closure invariant.
//!
//! The lattice is periodic; obstacle cells of the `CellGrid2D` are solid walls
//! (half-way bounce-back).  They are neither liquid nor gas for the VOF logic:
//! they never exchange mass, never convert, and never count as neighbours of
//! any phase.  A closed box of obstacle cells conserves the liquid mass
//! `Σ ρ (Fluid) + Σ m (Interface)` to rounding without any global correction.
//!
//! ## References
//! - Körner, C. et al. (2005). *Lattice Boltzmann model for free surface flow
//!   for modeling foaming*. J. Stat. Phys., 121(1–2), 179–196.
//! - Thürey, N. (2007). *Physically based Animation of Free Surface Flows
//!   with the Lattice Boltzmann Method*. PhD thesis, FAU Erlangen-Nürnberg.

use crate::grid::types::CellGrid2D;

// ─────────────────────────────────────────────────────────────────────────────
// D2Q9 constants (local copies — keeps this module self-contained)
// ─────────────────────────────────────────────────────────────────────────────

/// D2Q9 weights.
const W9: [f64; 9] = [
    4.0 / 9.0,
    1.0 / 9.0,
    1.0 / 9.0,
    1.0 / 9.0,
    1.0 / 9.0,
    1.0 / 36.0,
    1.0 / 36.0,
    1.0 / 36.0,
    1.0 / 36.0,
];

/// D2Q9 velocity vectors `[cx, cy]`.
const C9: [[i32; 2]; 9] = [
    [0, 0],   // 0 rest
    [1, 0],   // 1 E
    [0, 1],   // 2 N
    [-1, 0],  // 3 W
    [0, -1],  // 4 S
    [1, 1],   // 5 NE
    [-1, 1],  // 6 NW
    [-1, -1], // 7 SW
    [1, -1],  // 8 SE
];

/// D2Q9 opposite direction indices.
const OPP9: [usize; 9] = [0, 3, 4, 1, 2, 7, 8, 5, 6];

/// Speed of sound squared in LBM lattice units.
const CS2: f64 = 1.0 / 3.0;

/// Atmospheric density used for gas-side reconstruction.
const RHO_ATM: f64 = 1.0;

/// Fill-level tolerance for Interface → Fluid / Gas conversions.
const FILL_EPS: f64 = 1.0e-3;

// ─────────────────────────────────────────────────────────────────────────────
// Cell state flag
// ─────────────────────────────────────────────────────────────────────────────

/// Cell phase state for VOF free-surface tracking.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CellState {
    /// Cell is entirely gaseous (no fluid).
    Gas,
    /// Cell is at the fluid–gas interface.
    Interface,
    /// Cell is entirely filled with fluid.
    Fluid,
}

// ─────────────────────────────────────────────────────────────────────────────
// FreeSurfaceState
// ─────────────────────────────────────────────────────────────────────────────

/// VOF free-surface state parallel to a `CellGrid2D`.
///
/// `state`, `fill`, and `mass` are in row-major order: `idx(ix, iy) = iy * nx + ix`.
/// This matches `CellGrid2D::idx` exactly.
pub struct FreeSurfaceState {
    /// Per-cell phase state.
    pub state: Vec<CellState>,
    /// Per-cell fill level φ ∈ \[0, 1\].
    pub fill: Vec<f64>,
    /// Per-cell fluid mass m = φ · ρ.
    pub mass: Vec<f64>,
    /// Grid width.
    nx: usize,
    /// Grid height.
    ny: usize,
}

impl FreeSurfaceState {
    /// Create a free-surface state for a grid of size `nx × ny` with all
    /// cells initialised to `Gas` (fill = 0, mass = 0).
    pub fn new(nx: usize, ny: usize) -> Self {
        let n = nx * ny;
        Self {
            state: vec![CellState::Gas; n],
            fill: vec![0.0; n],
            mass: vec![0.0; n],
            nx,
            ny,
        }
    }

    /// Flat index: `idx(ix, iy) = iy * nx + ix` — matches `CellGrid2D::idx`.
    #[inline]
    pub fn idx(&self, ix: usize, iy: usize) -> usize {
        iy * self.nx + ix
    }

    /// Grid width.
    #[inline]
    pub fn nx(&self) -> usize {
        self.nx
    }

    /// Grid height.
    #[inline]
    pub fn ny(&self) -> usize {
        self.ny
    }

    /// Initialise a 2D dam-break scenario.
    ///
    /// The left half of the domain, the columns `ix < nx / 2` and the rows
    /// `iy < fluid_height`, is the fluid column (fill = 1, mass = ρ₀ = 1).  The
    /// outer shell of the column (its top row and its rightmost column) becomes
    /// Interface; everything else is Gas.  `fluid_height` is clamped to `ny`.
    ///
    /// The block does not know about walls: cells of the grid that are
    /// obstacles (for example the bottom row and the left column) keep the
    /// flags assigned here and are ignored by the solver; use
    /// [`clear_obstacles`](Self::clear_obstacles) to reset them.
    ///
    /// The corresponding `CellGrid2D` should be initialised at equilibrium
    /// with `rho0 = 1.0` before calling this.
    pub fn new_dam_break(nx: usize, ny: usize, fluid_height: usize) -> Self {
        let n = nx * ny;
        let mut s = Self {
            state: vec![CellState::Gas; n],
            fill: vec![0.0; n],
            mass: vec![0.0; n],
            nx,
            ny,
        };

        let fluid_height = fluid_height.min(ny);
        // Classic column-collapse dam break: left half of domain is the water column.
        let fluid_width = (nx / 2).max(1);

        for iy in 0..ny {
            for ix in 0..nx {
                let k = s.idx(ix, iy);
                let in_fluid_block = ix < fluid_width && iy < fluid_height;
                if !in_fluid_block {
                    continue; // Gas: already set
                }
                // Classify as Interface if on the exposed border (top row or
                // rightmost column of the fluid block), otherwise Fluid.
                let is_interface = iy + 1 == fluid_height || ix + 1 == fluid_width;
                if is_interface {
                    s.state[k] = CellState::Interface;
                } else {
                    s.state[k] = CellState::Fluid;
                }
                s.fill[k] = 1.0;
                s.mass[k] = 1.0;
            }
        }
        s
    }

    /// Total fluid mass: sum of `mass` over all Fluid and Interface cells.
    ///
    /// Obstacle cells of the grid are ignored by the solver, but an entry that
    /// was flagged Fluid or Interface (for example by
    /// [`new_dam_break`](Self::new_dam_break) for a wall inside the fluid
    /// block) still counts here; call [`clear_obstacles`](Self::clear_obstacles)
    /// first to measure the liquid only.
    pub fn total_mass(&self) -> f64 {
        self.state
            .iter()
            .zip(self.mass.iter())
            .filter_map(|(st, &m)| {
                if matches!(st, CellState::Fluid | CellState::Interface) {
                    Some(m)
                } else {
                    None
                }
            })
            .sum()
    }

    /// Mark every obstacle cell of `grid` as `Gas` with zero fill and mass.
    ///
    /// Obstacle cells are solid: they hold no liquid and the VOF update never
    /// looks at their entries.  This makes [`total_mass`](Self::total_mass)
    /// count the liquid only.
    ///
    /// # Panics
    /// Panics if the grid size differs from the size of this state.
    pub fn clear_obstacles(&mut self, grid: &CellGrid2D) {
        assert_eq!(
            (self.nx, self.ny),
            (grid.nx, grid.ny),
            "FreeSurfaceState and CellGrid2D must have the same size"
        );
        for (k, cell) in grid.cells.iter().enumerate() {
            if cell.obstacle {
                self.state[k] = CellState::Gas;
                self.fill[k] = 0.0;
                self.mass[k] = 0.0;
            }
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Internal helpers
// ─────────────────────────────────────────────────────────────────────────────

/// Number of fixed-point sweeps that solve the implicit gas-side
/// reconstruction of an interface cell for its own velocity.
const RECONSTRUCTION_SWEEPS: usize = 8;

/// D2Q9 equilibrium distribution component for direction `a`.
///
/// `f_eq_a = w_a · ρ · (1 + (e_a·u)/cs² + (e_a·u)²/(2cs⁴) − |u|²/(2cs²))`
#[inline]
fn feq(rho: f64, ux: f64, uy: f64, a: usize) -> f64 {
    let cx = C9[a][0] as f64;
    let cy = C9[a][1] as f64;
    let eu = cx * ux + cy * uy;
    let u_sq = ux * ux + uy * uy;
    W9[a] * rho * (1.0 + eu / CS2 + eu * eu / (2.0 * CS2 * CS2) - u_sq / (2.0 * CS2))
}

/// All nine equilibrium components for density `rho` and velocity `(ux, uy)`.
#[inline]
fn equilibrium(rho: f64, ux: f64, uy: f64) -> [f64; 9] {
    std::array::from_fn(|a| feq(rho, ux, uy, a))
}

/// Density `ρ = Σ f` and velocity `u = Σ e f / ρ` of a population vector.
#[inline]
fn moments(f: &[f64; 9]) -> (f64, f64, f64) {
    let mut rho = 0.0_f64;
    let mut mx = 0.0_f64;
    let mut my = 0.0_f64;
    for (fa, ca) in f.iter().zip(C9.iter()) {
        rho += fa;
        mx += fa * ca[0] as f64;
        my += fa * ca[1] as f64;
    }
    if rho.abs() > 1e-15 {
        (rho, mx / rho, my / rho)
    } else {
        (rho, 0.0, 0.0)
    }
}

/// Flat index of the periodic neighbour of `(ix, iy)` in lattice direction `a`.
///
/// Streaming wraps around the domain, so every neighbour query of the VOF
/// logic wraps in the same way.
#[inline]
fn neighbour_index(nx: usize, ny: usize, ix: usize, iy: usize, a: usize) -> usize {
    let jx = (ix as isize + C9[a][0] as isize).rem_euclid(nx as isize) as usize;
    let jy = (iy as isize + C9[a][1] as isize).rem_euclid(ny as isize) as usize;
    jy * nx + jx
}

/// Perform a BGK collision + periodic streaming step restricted to Fluid and
/// Interface cells, with an optional Guo body force.
///
/// **Collision.**  Gas cells do not collide; Fluid and Interface cells relax
/// towards equilibrium with the density and (force-shifted) velocity stored in
/// the cell.  The body force `(gx, gy)` is an acceleration; it enters through
/// the Guo 2002 scheme with the force density `F = ρ g`: the stored velocity
/// (pre-shifted by `0.5 g` in [`apply_body_force_2d`]) is used in the
/// equilibrium and the term `(1 − ω/2) F_i` is added to the populations.
///
/// **Streaming.**  Every non-obstacle cell, Gas cells included, pulls its
/// populations from the upwind node `x − e_a` of the periodic lattice.  An
/// obstacle cell upwind reflects the population: the cell takes its own
/// post-collision opposite population (half-way bounce-back, no-slip wall half
/// a cell inside the obstacle).  Obstacle cells therefore neither take part nor
/// exchange mass, a closed box of obstacles conserves the liquid mass exactly,
/// and an additional [`CellGrid2D::apply_bounce_back`] is not needed.  A Gas
/// cell that is upwind of an Interface cell holds the post-collision population
/// that the Interface cell sent towards it, which the reconstruction in
/// [`free_surface_step`] needs.
///
/// **Moments.**  Density and velocity of Fluid and Interface cells are
/// recomputed from the streamed populations.  The values of Interface cells
/// are provisional: the populations that would arrive from Gas neighbours are
/// missing and are reconstructed in [`free_surface_step`], which also refreshes
/// those moments.
///
/// Call [`free_surface_step`] **after** this function to advance the VOF
/// state.
pub fn free_surface_bgk_stream(
    grid: &mut CellGrid2D,
    fs: &FreeSurfaceState,
    omega: f64,
    gx: f64,
    gy: f64,
) {
    let nx = grid.nx;
    let ny = grid.ny;
    let guo_coeff = 1.0 - 0.5 * omega;

    // ── BGK collision + Guo force on Fluid and Interface cells only ──
    for iy in 0..ny {
        for ix in 0..nx {
            let ki = fs.idx(ix, iy);
            let cell = &mut grid.cells[ki];
            if cell.obstacle || fs.state[ki] == CellState::Gas {
                continue;
            }
            let rho = cell.rho;
            // cell.ux/uy already carry the half-step shift 0.5·g (Guo scheme).
            let ux = cell.ux;
            let uy = cell.uy;
            for a in 0..9usize {
                let f_eq = feq(rho, ux, uy, a);
                cell.f[a] -= omega * (cell.f[a] - f_eq);
                // Guo force term: F_i = w_i [(e_i − u)/cs² + (e_i·u) e_i/cs⁴] · (ρ g)
                let cx = C9[a][0] as f64;
                let cy = C9[a][1] as f64;
                let cu = cx * ux + cy * uy;
                let force_i = W9[a]
                    * rho
                    * (((cx - ux) * gx + (cy - uy) * gy) / CS2
                        + cu * (cx * gx + cy * gy) / (CS2 * CS2));
                cell.f[a] += guo_coeff * force_i;
            }
        }
    }

    // ── Pull streaming: periodic, half-way bounce-back at obstacles ──
    let f_old: Vec<[f64; 9]> = grid.cells.iter().map(|c| c.f).collect();
    let solid: Vec<bool> = grid.cells.iter().map(|c| c.obstacle).collect();
    for iy in 0..ny {
        for ix in 0..nx {
            let ki = fs.idx(ix, iy);
            if solid[ki] {
                continue;
            }
            for a in 0..9usize {
                // Upwind node x − e_a = x + e_ā.
                let ks = neighbour_index(nx, ny, ix, iy, OPP9[a]);
                grid.cells[ki].f[a] = if solid[ks] {
                    f_old[ki][OPP9[a]]
                } else {
                    f_old[ks][a]
                };
            }
        }
    }

    // ── Update macroscopic for Fluid and Interface cells only ──
    for iy in 0..ny {
        for ix in 0..nx {
            let ki = fs.idx(ix, iy);
            let cell = &mut grid.cells[ki];
            if cell.obstacle || fs.state[ki] == CellState::Gas {
                continue;
            }
            let (rho, ux, uy) = moments(&cell.f);
            cell.rho = rho;
            cell.ux = ux;
            cell.uy = uy;
        }
    }
}

/// Apply the half-step force shift of the Guo 2002 scheme to the stored
/// velocity of all Fluid and Interface cells of `grid`.
///
/// The stored macroscopic velocity `(ux, uy)` is the plain first moment
/// `Σ e f / ρ`.  This function adds `0.5 · (gx, gy)` so that the BGK collision
/// in [`free_surface_bgk_stream`] uses the force-corrected velocity
/// `u = Σ e f / ρ + F/(2ρ)` with `F = ρ g`; the force term itself is added to
/// the populations there.  Gas cells and obstacle cells are skipped; `fs`
/// identifies the Gas cells.  Pass `gx = 0.0` and `gy = -g_lattice` for
/// downward gravity (negative y-direction).
///
/// Call this **before** [`free_surface_bgk_stream`], with the same `(gx, gy)`.
/// A fluid in hydrostatic balance has the stored velocity `−0.5 g`, so its
/// physical velocity is the stored one plus `0.5 g`.
pub fn apply_body_force_2d(
    grid: &mut CellGrid2D,
    fs: &FreeSurfaceState,
    gx: f64,
    gy: f64,
    _omega: f64,
) {
    let nx = grid.nx;
    let ny = grid.ny;
    for iy in 0..ny {
        for ix in 0..nx {
            let ki = fs.idx(ix, iy);
            let cell = &mut grid.cells[ki];
            if cell.obstacle || fs.state[ki] == CellState::Gas {
                continue;
            }
            // Guo scheme: u_star = u + F/(2ρ) = u + g/2.
            cell.ux += 0.5 * gx;
            cell.uy += 0.5 * gy;
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Main per-step update
// ─────────────────────────────────────────────────────────────────────────────

/// Advance the VOF free-surface state by one step.
///
/// This must be called **after** [`free_surface_bgk_stream`] (or an equivalent
/// BGK / MRT + periodic streaming cycle) on the same grid, in which every Gas
/// cell pulled the post-collision populations that its Interface neighbours sent
/// towards it.
///
/// The function implements the Körner 2005 / Thürey algorithm on the periodic
/// lattice; obstacle cells are ignored throughout (they are neither liquid nor
/// gas, never exchange mass, and are never converted):
/// 1. Reconstruct the gas-side populations that arrive at Interface cells.
/// 2. Perform mass exchange between neighbouring cells.
/// 3. Update fill levels.
/// 4. Convert filled / emptied Interface cells with excess / deficit
///    redistribution, initialising new Interface cells from the equilibrium of
///    the surrounding liquid.
/// 5. Enforce the Interface closure invariant (no Fluid cell touches a Gas
///    cell).
///
/// # Panics
/// Panics if `fs.nx() != grid.nx || fs.ny() != grid.ny`.
pub fn free_surface_step(grid: &mut CellGrid2D, fs: &mut FreeSurfaceState) {
    assert_eq!(
        fs.nx(),
        grid.nx,
        "FreeSurfaceState nx must equal CellGrid2D nx"
    );
    assert_eq!(
        fs.ny(),
        grid.ny,
        "FreeSurfaceState ny must equal CellGrid2D ny"
    );

    reconstruct_gas_side(grid, fs);

    let delta_mass = exchange_mass(grid, fs);
    for (mass, dm) in fs.mass.iter_mut().zip(&delta_mass) {
        *mass += dm;
    }

    update_fill_levels(grid, fs);
    convert_cells(grid, fs);
    close_interface_layer(grid, fs);
    refresh_interface_fill(grid, fs);
}

/// Step 1: gas-side distribution reconstruction for Interface cells.
///
/// Take an Interface cell `x` and a direction `a` whose neighbour `g = x + e_a`
/// (periodic) is a Gas cell.  Streaming made `g` pull the post-collision
/// population `f_a` that `x` sent towards it, so `f_a^*(x) = f_g[a]`.  The
/// population that would arrive from `g` is `f_ā(x)` (it moves along `−e_a`);
/// it is not known and is replaced by the Körner 2005 closure that imposes the
/// atmospheric density `ρ_atm` and the velocity of the cell,
///
/// ```text
/// f_ā(x) = f_eq,a(ρ_atm, u) + f_eq,ā(ρ_atm, u) − f_a^*(x).
/// ```
///
/// The velocity `u` is the velocity of the reconstructed cell itself.  Because
/// the streamed populations that came from Gas cells are stale, it is obtained
/// from the populations that did arrive from liquid or wall neighbours plus the
/// reconstructed ones, by a fixed-point iteration (the equilibrium sum
/// `f_eq,a + f_eq,ā` depends on `u` only quadratically).  Afterwards the
/// density and velocity stored in the cell are recomputed from the completed
/// populations.
fn reconstruct_gas_side(grid: &mut CellGrid2D, fs: &FreeSurfaceState) {
    let nx = fs.nx();
    let ny = fs.ny();
    for iy in 0..ny {
        for ix in 0..nx {
            let ki = fs.idx(ix, iy);
            if grid.cells[ki].obstacle || fs.state[ki] != CellState::Interface {
                continue;
            }

            // Gas links: (index of the missing incoming population ā,
            // post-collision population a that was sent towards the gas cell).
            let mut links = [(0_usize, 0.0_f64); 8];
            let mut n_links = 0_usize;
            for (a, &opposite) in OPP9.iter().enumerate().skip(1) {
                let kg = neighbour_index(nx, ny, ix, iy, a);
                if fs.state[kg] == CellState::Gas && !grid.cells[kg].obstacle {
                    links[n_links] = (opposite, grid.cells[kg].f[a]);
                    n_links += 1;
                }
            }
            if n_links == 0 {
                continue;
            }
            let links = &links[..n_links];

            // Moments of the populations that did arrive from liquid / walls.
            let mut missing = [false; 9];
            for &(q, _) in links {
                missing[q] = true;
            }
            let cell = &grid.cells[ki];
            let mut rho_known = 0.0_f64;
            let mut mx_known = 0.0_f64;
            let mut my_known = 0.0_f64;
            for (q, (&fq, cq)) in cell.f.iter().zip(C9.iter()).enumerate() {
                if !missing[q] {
                    rho_known += fq;
                    mx_known += fq * cq[0] as f64;
                    my_known += fq * cq[1] as f64;
                }
            }

            // Fixed point for the velocity of the completed cell.
            let mut ux = cell.ux;
            let mut uy = cell.uy;
            for _ in 0..RECONSTRUCTION_SWEEPS {
                let mut rho = rho_known;
                let mut mx = mx_known;
                let mut my = my_known;
                for &(q, sent) in links {
                    let fq = feq(RHO_ATM, ux, uy, OPP9[q]) + feq(RHO_ATM, ux, uy, q) - sent;
                    rho += fq;
                    mx += fq * C9[q][0] as f64;
                    my += fq * C9[q][1] as f64;
                }
                if rho.abs() > 1e-15 {
                    ux = mx / rho;
                    uy = my / rho;
                } else {
                    ux = 0.0;
                    uy = 0.0;
                }
            }

            let cell = &mut grid.cells[ki];
            for &(q, sent) in links {
                cell.f[q] = feq(RHO_ATM, ux, uy, OPP9[q]) + feq(RHO_ATM, ux, uy, q) - sent;
            }
            let (rho, ux, uy) = moments(&cell.f);
            cell.rho = rho;
            cell.ux = ux;
            cell.uy = uy;
        }
    }
}

/// Step 2: mass exchange of the Interface cells with their neighbours.
///
/// For an Interface cell `i` and a neighbour `j = i + e_a` (periodic) the
/// streamed populations give the net inflow
/// `Δm = f_ā(i) − f_a(j)`: `f_ā(i)` arrived from `j` along `−e_a`, and
/// `f_a(j)` is the population `j` pulled from `i`, i.e. what left `i` towards
/// `j`.  It is weighted with the average fill `½(φ_i + φ_j)` for an Interface
/// neighbour, `1` for a Fluid neighbour and `0` for Gas and obstacle
/// neighbours (nothing is exchanged with a half-way bounce-back wall).
///
/// Returns the mass change of every cell (non-zero for Interface cells only).
fn exchange_mass(grid: &CellGrid2D, fs: &FreeSurfaceState) -> Vec<f64> {
    let nx = fs.nx();
    let ny = fs.ny();
    let mut delta_mass = vec![0.0_f64; nx * ny];
    for iy in 0..ny {
        for ix in 0..nx {
            let ki = fs.idx(ix, iy);
            if grid.cells[ki].obstacle || fs.state[ki] != CellState::Interface {
                continue;
            }
            let fill_i = fs.fill[ki];
            for (a, &opposite) in OPP9.iter().enumerate().skip(1) {
                let kj = neighbour_index(nx, ny, ix, iy, a);
                if grid.cells[kj].obstacle {
                    continue;
                }
                let scale = match fs.state[kj] {
                    CellState::Gas => continue,
                    CellState::Fluid => 1.0,
                    CellState::Interface => 0.5 * (fill_i + fs.fill[kj]),
                };
                let inflow = grid.cells[ki].f[opposite];
                let outflow = grid.cells[kj].f[a];
                delta_mass[ki] += (inflow - outflow) * scale;
            }
        }
    }
    delta_mass
}

/// Step 3: update the fill levels from the masses and densities.
///
/// A Fluid cell is full and its mass is its density, a Gas cell is empty, an
/// Interface cell has `φ = m/ρ`.
fn update_fill_levels(grid: &CellGrid2D, fs: &mut FreeSurfaceState) {
    for (k, cell) in grid.cells.iter().enumerate() {
        if cell.obstacle {
            continue;
        }
        match fs.state[k] {
            CellState::Fluid => {
                fs.fill[k] = 1.0;
                fs.mass[k] = cell.rho;
            }
            CellState::Gas => {
                fs.fill[k] = 0.0;
                fs.mass[k] = 0.0;
            }
            CellState::Interface => {
                fs.fill[k] = fs.mass[k] / cell.rho.max(1e-8);
            }
        }
    }
}

/// Average density and velocity of the liquid neighbours of cell `k`.
///
/// `states` is the phase flag field that decides which neighbours count as
/// liquid (not Gas); obstacle neighbours never count.  Without a liquid
/// neighbour the atmosphere at rest is returned.
fn neighbour_liquid_average(
    grid: &CellGrid2D,
    states: &[CellState],
    nx: usize,
    ny: usize,
    k: usize,
) -> (f64, f64, f64) {
    let (ix, iy) = (k % nx, k / nx);
    let mut rho_sum = 0.0_f64;
    let mut ux_sum = 0.0_f64;
    let mut uy_sum = 0.0_f64;
    let mut count = 0_usize;
    for a in 1..9usize {
        let kn = neighbour_index(nx, ny, ix, iy, a);
        let neighbour = &grid.cells[kn];
        if neighbour.obstacle || states[kn] == CellState::Gas {
            continue;
        }
        rho_sum += neighbour.rho;
        ux_sum += neighbour.ux;
        uy_sum += neighbour.uy;
        count += 1;
    }
    if count == 0 {
        return (RHO_ATM, 0.0, 0.0);
    }
    let n = count as f64;
    (rho_sum / n, ux_sum / n, uy_sum / n)
}

/// Turn the Gas cell `k` into an empty Interface cell whose populations are the
/// equilibrium of `(rho, ux, uy)` (the average liquid state around it).
///
/// Without this the cell would keep the stale populations of the gas region,
/// which flow into the liquid as spurious mass once the cell streams.
fn make_interface_cell(
    grid: &mut CellGrid2D,
    fs: &mut FreeSurfaceState,
    k: usize,
    (rho, ux, uy): (f64, f64, f64),
) {
    let cell = &mut grid.cells[k];
    cell.f = equilibrium(rho, ux, uy);
    cell.rho = rho;
    cell.ux = ux;
    cell.uy = uy;
    fs.state[k] = CellState::Interface;
    fs.fill[k] = 0.0;
    fs.mass[k] = 0.0;
}

/// Turn the Fluid cell `k` into a full Interface cell (`φ = 1`, `m = ρ`).
fn demote_to_interface(grid: &CellGrid2D, fs: &mut FreeSurfaceState, k: usize) {
    fs.state[k] = CellState::Interface;
    fs.fill[k] = 1.0;
    fs.mass[k] = grid.cells[k].rho.max(1e-8);
}

/// Step 4: convert filled and emptied Interface cells.
///
/// An Interface cell with `φ ≥ 1 + ε` becomes Fluid, one with `φ ≤ −ε` becomes
/// Gas.  The interface layer stays closed: Gas neighbours of a filled cell
/// become (empty) Interface cells, Fluid neighbours of an emptied cell become
/// (full) Interface cells.  The excess mass `m − ρ` of a filled cell and the
/// (negative) mass `m` of an emptied cell are handed in equal parts to the
/// Interface neighbours that remain; if a cell has none, the mass is shared by
/// all Interface cells, so that the total liquid mass is conserved.
fn convert_cells(grid: &mut CellGrid2D, fs: &mut FreeSurfaceState) {
    let nx = fs.nx();
    let ny = fs.ny();
    let n = nx * ny;

    let mut filled: Vec<usize> = Vec::new();
    let mut emptied: Vec<usize> = Vec::new();
    for k in 0..n {
        if grid.cells[k].obstacle || fs.state[k] != CellState::Interface {
            continue;
        }
        if fs.fill[k] >= 1.0 + FILL_EPS {
            filled.push(k);
        } else if fs.fill[k] <= -FILL_EPS {
            emptied.push(k);
        }
    }
    if filled.is_empty() && emptied.is_empty() {
        return;
    }

    // Keep the layer closed around the converting cells.  The new Interface
    // cells are initialised from the liquid state before any conversion.
    let reference = fs.state.clone();
    let mut marked = vec![false; n];
    let mut to_promote: Vec<usize> = Vec::new();
    let mut to_demote: Vec<usize> = Vec::new();
    for &k in &filled {
        for a in 1..9usize {
            let kn = neighbour_index(nx, ny, k % nx, k / nx, a);
            if !grid.cells[kn].obstacle && reference[kn] == CellState::Gas && !marked[kn] {
                marked[kn] = true;
                to_promote.push(kn);
            }
        }
    }
    for &k in &emptied {
        for a in 1..9usize {
            let kn = neighbour_index(nx, ny, k % nx, k / nx, a);
            if !grid.cells[kn].obstacle && reference[kn] == CellState::Fluid && !marked[kn] {
                marked[kn] = true;
                to_demote.push(kn);
            }
        }
    }
    for &k in &to_promote {
        let liquid = neighbour_liquid_average(grid, &reference, nx, ny, k);
        make_interface_cell(grid, fs, k, liquid);
    }
    for &k in &to_demote {
        demote_to_interface(grid, fs, k);
    }

    // Convert, remembering the mass that has to be handed on.
    let mut handover: Vec<(usize, f64)> = Vec::with_capacity(filled.len() + emptied.len());
    for &k in &filled {
        let rho = grid.cells[k].rho.max(1e-8);
        handover.push((k, fs.mass[k] - rho));
        fs.state[k] = CellState::Fluid;
        fs.fill[k] = 1.0;
        fs.mass[k] = rho;
    }
    for &k in &emptied {
        handover.push((k, fs.mass[k]));
        fs.state[k] = CellState::Gas;
        fs.fill[k] = 0.0;
        fs.mass[k] = 0.0;
    }

    let mut orphaned = 0.0_f64;
    for (k, amount) in handover {
        let mut receivers = [0_usize; 8];
        let mut n_receivers = 0_usize;
        for a in 1..9usize {
            let kn = neighbour_index(nx, ny, k % nx, k / nx, a);
            if !grid.cells[kn].obstacle && fs.state[kn] == CellState::Interface {
                receivers[n_receivers] = kn;
                n_receivers += 1;
            }
        }
        if n_receivers == 0 {
            orphaned += amount;
            continue;
        }
        let share = amount / n_receivers as f64;
        for &kn in &receivers[..n_receivers] {
            fs.mass[kn] += share;
        }
    }
    if orphaned != 0.0 {
        let interface: Vec<usize> = (0..n)
            .filter(|&k| !grid.cells[k].obstacle && fs.state[k] == CellState::Interface)
            .collect();
        if !interface.is_empty() {
            let share = orphaned / interface.len() as f64;
            for &k in &interface {
                fs.mass[k] += share;
            }
        }
    }
}

/// Step 5: interface layer closure invariant.
///
/// After the conversions no Fluid cell may touch a Gas cell (8-neighbourhood,
/// periodic): a Fluid cell with a Gas neighbour becomes a full Interface cell
/// (`φ = 1`, `m = ρ`), a Gas cell with a Fluid neighbour becomes an empty
/// Interface cell initialised from the surrounding liquid.  Obstacle cells are
/// neither.  Every pass strictly reduces the number of Fluid and Gas cells, so
/// the iteration terminates.
fn close_interface_layer(grid: &mut CellGrid2D, fs: &mut FreeSurfaceState) {
    let nx = fs.nx();
    let ny = fs.ny();
    let mut changed = true;
    while changed {
        changed = false;
        for k in 0..nx * ny {
            if grid.cells[k].obstacle {
                continue;
            }
            let (ix, iy) = (k % nx, k / nx);
            let touches = |phase: CellState| {
                (1..9usize).any(|a| {
                    let kn = neighbour_index(nx, ny, ix, iy, a);
                    !grid.cells[kn].obstacle && fs.state[kn] == phase
                })
            };
            match fs.state[k] {
                CellState::Fluid if touches(CellState::Gas) => {
                    demote_to_interface(grid, fs, k);
                    changed = true;
                }
                CellState::Gas if touches(CellState::Fluid) => {
                    let liquid = neighbour_liquid_average(grid, &fs.state, nx, ny, k);
                    make_interface_cell(grid, fs, k, liquid);
                    changed = true;
                }
                _ => {}
            }
        }
    }
}

/// Recompute the fill of every Interface cell from its mass.
///
/// The masses of Interface cells change during the redistribution of the
/// conversion step, and the densities of new Interface cells are only known
/// afterwards.
fn refresh_interface_fill(grid: &CellGrid2D, fs: &mut FreeSurfaceState) {
    for (k, cell) in grid.cells.iter().enumerate() {
        if !cell.obstacle && fs.state[k] == CellState::Interface {
            fs.fill[k] = fs.mass[k] / cell.rho.max(1e-8);
        }
    }
}

/// Apply a global mass correction to keep `total_mass` exactly equal to
/// `target_mass`.
///
/// Any discrepancy (positive or negative) is redistributed uniformly to all
/// Interface cells (obstacle cells excluded).  A correct simulation conserves
/// the liquid mass without this correction; it is a safety net for open
/// configurations in which liquid legitimately leaves through the gas side.
///
/// Call once per step after `free_surface_step`.
pub fn apply_mass_correction(grid: &CellGrid2D, fs: &mut FreeSurfaceState, target_mass: f64) {
    let current_mass = fs.total_mass();
    let error = target_mass - current_mass;
    if error.abs() < 1e-15 {
        return;
    }

    // Count Interface cells available to absorb the correction.
    let n_interface = grid
        .cells
        .iter()
        .zip(fs.state.iter())
        .filter(|(cell, state)| !cell.obstacle && **state == CellState::Interface)
        .count();

    if n_interface == 0 {
        return;
    }

    let correction_per_cell = error / n_interface as f64;

    for (k, cell) in grid.cells.iter().enumerate() {
        if !cell.obstacle && fs.state[k] == CellState::Interface {
            fs.mass[k] += correction_per_cell;
            fs.fill[k] = fs.mass[k] / cell.rho.max(1e-8);
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod regression_tests;

#[cfg(test)]
mod tests {
    use super::*;
    use crate::grid::types::CellGrid2D;

    // ── helpers ─────────────────────────────────────────────────────────────

    /// A grid whose rim is made of obstacle (wall) cells.
    fn walled_grid(nx: usize, ny: usize) -> CellGrid2D {
        let mut grid = CellGrid2D::new(nx, ny, 1.0);
        for x in 0..nx {
            let kb = grid.idx(x, 0);
            let kt = grid.idx(x, ny - 1);
            grid.cells[kb].obstacle = true;
            grid.cells[kt].obstacle = true;
        }
        for y in 0..ny {
            let kl = grid.idx(0, y);
            let kr = grid.idx(nx - 1, y);
            grid.cells[kl].obstacle = true;
            grid.cells[kr].obstacle = true;
        }
        grid
    }

    /// Run `n_steps` of (body force + VOF-restricted BGK step with half-way
    /// bounce-back at the walls + VOF step) with the given gravity
    /// acceleration `g_lattice` (−y direction), without any global mass
    /// correction.
    fn run_steps_gravity(
        grid: &mut CellGrid2D,
        fs: &mut FreeSurfaceState,
        n_steps: usize,
        omega: f64,
        g_lattice: f64,
    ) {
        for _ in 0..n_steps {
            // Shift macroscopic velocities by 0.5*g (Guo pre-step).
            apply_body_force_2d(grid, fs, 0.0, -g_lattice, omega);
            // BGK with Guo force term + stream (Fluid/Interface only).
            free_surface_bgk_stream(grid, fs, omega, 0.0, -g_lattice);
            // VOF mass tracking and flag reinit.
            free_surface_step(grid, fs);
        }
    }

    // ── 1. Mass conservation — dam break ────────────────────────────────────
    /// Run a 60×40 dam-break for 200 steps in a closed box and assert that the
    /// liquid mass is conserved without any global correction.
    #[test]
    fn test_mass_conservation_dam_break() {
        let nx = 60;
        let ny = 40;
        let fluid_height = 20;
        let omega = 1.0;

        let mut grid = walled_grid(nx, ny);
        let mut fs = FreeSurfaceState::new_dam_break(nx, ny, fluid_height);
        fs.clear_obstacles(&grid);
        let mass_initial = fs.total_mass();

        // g=5e-4 keeps Mach low (u ~ sqrt(2gH) ≈ sqrt(2*5e-4*20) ≈ 0.14).
        let g_run = 5e-4_f64;
        run_steps_gravity(&mut grid, &mut fs, 200, omega, g_run);

        let mass_final = fs.total_mass();
        let rel_err = (mass_final - mass_initial).abs() / mass_initial.max(1e-15);
        assert!(
            rel_err < 1e-3,
            "Mass conservation violated: initial={mass_initial:.6}, \
             final={mass_final:.6}, rel_err={rel_err:.2e}"
        );
        // The exchange is antisymmetric, the conversions hand their excess on:
        // the liquid mass is conserved to rounding.
        assert!(rel_err < 1e-9, "mass drift {rel_err:.2e} beyond rounding");
    }

    // ── 2. Dam-break front position advances ────────────────────────────────
    /// After 200 steps the rightmost wet (Interface or Fluid) column must have
    /// advanced at least 5 cells from its initial position, confirming that
    /// the free surface actually moves under gravity / momentum.
    #[test]
    fn test_dam_break_front_position() {
        // The classic column-collapse setup: left half filled with fluid up to
        // fluid_height, right half and top are Gas.  `new_dam_break` puts the
        // fluid column in the left `nx/2` columns.
        let nx = 60;
        let ny = 40;
        let fluid_height = 20;
        let omega = 1.0;

        let mut grid = walled_grid(nx, ny);
        let mut fs = FreeSurfaceState::new_dam_break(nx, ny, fluid_height);
        fs.clear_obstacles(&grid);

        // The initial right-most wet column is the rightmost column of the
        // fluid block, i.e. nx/2 - 1 = 29.
        let initial_front = rightmost_wet_column(&fs);

        // Gravity g = 5e-4 gives u ~ sqrt(2 g H) ~ 0.14 (Ma ~ 0.25): the
        // collapse is well resolved and stable.  (g = 5e-3 would exceed the
        // lattice Mach limit and blow up.)
        let g_lattice = 5e-4;
        run_steps_gravity(&mut grid, &mut fs, 200, omega, g_lattice);

        let final_front = rightmost_wet_column(&fs);
        let advance = final_front.saturating_sub(initial_front);
        assert!(
            advance >= 5,
            "Dam-break front should advance ≥5 cells in 200 steps, \
             but initial={initial_front}, final={final_front}"
        );
    }

    /// Rightmost column index that contains at least one Interface or Fluid cell.
    fn rightmost_wet_column(fs: &FreeSurfaceState) -> usize {
        let nx = fs.nx();
        let ny = fs.ny();
        let mut rightmost = 0_usize;
        for iy in 0..ny {
            for ix in 0..nx {
                let k = fs.idx(ix, iy);
                if matches!(fs.state[k], CellState::Interface | CellState::Fluid) && ix > rightmost
                {
                    rightmost = ix;
                }
            }
        }
        rightmost
    }

    // ── 3. No Fluid cell adjacent to Gas cell ───────────────────────────────
    /// After each step, no Fluid cell should be 8-connected to a Gas cell
    /// (walls are neither).
    #[test]
    fn test_no_fluid_gas_adjacency() {
        let nx = 30;
        let ny = 20;
        let fluid_height = 10;
        let omega = 1.0;

        let mut grid = walled_grid(nx, ny);
        let mut fs = FreeSurfaceState::new_dam_break(nx, ny, fluid_height);
        fs.clear_obstacles(&grid);
        let g_lattice = 1e-4;

        for step in 0..50 {
            apply_body_force_2d(&mut grid, &fs, 0.0, -g_lattice, omega);
            free_surface_bgk_stream(&mut grid, &fs, omega, 0.0, -g_lattice);
            free_surface_step(&mut grid, &mut fs);

            for iy in 0..ny {
                for ix in 0..nx {
                    let ki = fs.idx(ix, iy);
                    if grid.cells[ki].obstacle || fs.state[ki] != CellState::Fluid {
                        continue;
                    }
                    for (a, ca) in C9.iter().enumerate().skip(1) {
                        let jx = ix as isize + ca[0] as isize;
                        let jy = iy as isize + ca[1] as isize;
                        if jx < 0 || jy < 0 || jx >= nx as isize || jy >= ny as isize {
                            continue;
                        }
                        let kj = fs.idx(jx as usize, jy as usize);
                        if grid.cells[kj].obstacle {
                            continue;
                        }
                        assert_ne!(
                            fs.state[kj],
                            CellState::Gas,
                            "Step {step}: Fluid cell ({ix},{iy}) is adjacent to Gas at \
                             ({},{}) via direction {a}",
                            jx,
                            jy
                        );
                    }
                }
            }
        }
    }

    // ── 4. Static enclosed pool stays stable ────────────────────────────────
    /// An all-Fluid box (no Interface / Gas) with obstacle walls: after 20
    /// steps all interior densities should still be ~1.0 (no spurious flow).
    #[test]
    fn test_static_pool_stable() {
        let nx = 10;
        let ny = 10;
        let omega = 1.0;

        let mut grid = walled_grid(nx, ny);

        // All interior cells are Fluid.
        let mut fs = FreeSurfaceState::new(nx, ny);
        for iy in 1..(ny - 1) {
            for ix in 1..(nx - 1) {
                let k = fs.idx(ix, iy);
                fs.state[k] = CellState::Fluid;
                fs.fill[k] = 1.0;
                fs.mass[k] = 1.0;
            }
        }

        for _ in 0..20 {
            free_surface_bgk_stream(&mut grid, &fs, omega, 0.0, 0.0);
            free_surface_step(&mut grid, &mut fs);
        }

        // All non-obstacle cells should still have rho ≈ 1.0.
        for iy in 1..(ny - 1) {
            for ix in 1..(nx - 1) {
                let k = grid.idx(ix, iy);
                let rho = grid.cells[k].rho;
                assert!(
                    (rho - 1.0).abs() < 0.1,
                    "Spurious density at ({ix},{iy}): rho={rho}"
                );
            }
        }
    }
}
