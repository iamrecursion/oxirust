//! Regression tests for the VOF free-surface update: reconstruction of the
//! population that arrives from a gas neighbour, mass exchange on the
//! post-stream populations, half-way bounce-back at walls, periodic
//! neighbours, and the hydrostatic / closed-box conservation properties.

use super::*;
use crate::grid::types::CellGrid2D;

// ── helpers ─────────────────────────────────────────────────────────────────

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

/// Liquid mass: Fluid and Interface cells that are not obstacles.
fn liquid_mass(grid: &CellGrid2D, fs: &FreeSurfaceState) -> f64 {
    grid.cells
        .iter()
        .zip(fs.state.iter().zip(fs.mass.iter()))
        .filter(|(cell, (state, _))| {
            !cell.obstacle && matches!(state, CellState::Fluid | CellState::Interface)
        })
        .map(|(_, (_, &mass))| mass)
        .sum()
}

/// Mark every cell of `fs` as fluid (full, mass 1).
fn fill_with_fluid(fs: &mut FreeSurfaceState) {
    fs.state.fill(CellState::Fluid);
    fs.fill.fill(1.0);
    fs.mass.fill(1.0);
}

/// Deterministic pseudo-random numbers in `[-1, 1)` (64-bit LCG).
fn next_unit(state: &mut u64) -> f64 {
    *state = state
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    ((*state >> 11) as f64 / (1u64 << 53) as f64) * 2.0 - 1.0
}

/// Quiescent pool in a closed box: fluid in the rows `1..surface_row`, one row
/// of Interface cells with the given fill at `surface_row`, gas above.
fn hydrostatic_pool(
    nx: usize,
    ny: usize,
    surface_row: usize,
    surface_fill: f64,
) -> (CellGrid2D, FreeSurfaceState) {
    let grid = walled_grid(nx, ny);
    let mut fs = FreeSurfaceState::new(nx, ny);
    for iy in 1..=surface_row {
        for ix in 1..nx - 1 {
            let k = fs.idx(ix, iy);
            if iy < surface_row {
                fs.state[k] = CellState::Fluid;
                fs.fill[k] = 1.0;
                fs.mass[k] = 1.0;
            } else {
                fs.state[k] = CellState::Interface;
                fs.fill[k] = surface_fill;
                fs.mass[k] = surface_fill;
            }
        }
    }
    (grid, fs)
}

/// One time step of the documented pipeline with gravity `g` along `−y`.
fn gravity_step(grid: &mut CellGrid2D, fs: &mut FreeSurfaceState, omega: f64, g: f64) {
    apply_body_force_2d(grid, fs, 0.0, -g, omega);
    free_surface_bgk_stream(grid, fs, omega, 0.0, -g);
    free_surface_step(grid, fs);
}

/// Largest physical speed over the liquid cells.  The stored velocity is the
/// plain first moment, the physical one adds the half-step force shift
/// `0.5 (gx, gy)` of the Guo scheme.
fn max_physical_speed(grid: &CellGrid2D, fs: &FreeSurfaceState, gx: f64, gy: f64) -> f64 {
    grid.cells
        .iter()
        .zip(fs.state.iter())
        .filter(|(cell, state)| !cell.obstacle && **state != CellState::Gas)
        .map(|(cell, _)| {
            let ux = cell.ux + 0.5 * gx;
            let uy = cell.uy + 0.5 * gy;
            (ux * ux + uy * uy).sqrt()
        })
        .fold(0.0_f64, f64::max)
}

// ── F2540: gas-side reconstruction ──────────────────────────────────────────

#[test]
fn reconstruction_replaces_the_population_arriving_from_the_gas() {
    let n = 5;
    let mut grid = CellGrid2D::new(n, n, 1.0);
    let mut fs = FreeSurfaceState::new(n, n);
    fill_with_fluid(&mut fs);
    let interface = fs.idx(2, 2);
    let gas = fs.idx(2, 3); // north neighbour: a = 2, e_a = (0, 1)
    fs.state[interface] = CellState::Interface;
    fs.fill[interface] = 0.5;
    fs.mass[interface] = 0.5;
    fs.state[gas] = CellState::Gas;
    fs.fill[gas] = 0.0;
    fs.mass[gas] = 0.0;

    // The gas cell pulled the population the interface cell sent north.
    let sent = 0.07;
    grid.cells[gas].f = [9.0; 9];
    grid.cells[gas].f[2] = sent;
    // The population arriving from the gas moves south (index 4).  Choose the
    // known populations so that the completed cell is at rest: f_2 = f_4 and
    // f_4 = 2 w_4 ρ_atm − sent.
    let balanced = 2.0 * W9[4] * RHO_ATM - sent;
    grid.cells[interface].f = W9;
    grid.cells[interface].f[2] = balanced;
    grid.cells[interface].f[4] = 5.0; // stale value streamed out of the gas cell

    reconstruct_gas_side(&mut grid, &fs);

    let cell = &grid.cells[interface];
    assert!(
        (cell.f[4] - balanced).abs() < 1e-14,
        "the incoming population f_4 must be reconstructed: {}",
        cell.f[4]
    );
    assert_eq!(
        cell.f[2], balanced,
        "the outgoing slot f_2 must stay untouched"
    );
    for q in [0, 1, 3, 5, 6, 7, 8] {
        assert_eq!(cell.f[q], W9[q], "population {q} must stay untouched");
    }
    let rho: f64 = cell.f.iter().sum();
    assert!((cell.rho - rho).abs() < 1e-14, "density refreshed");
    assert!(
        cell.ux.abs() < 1e-13 && cell.uy.abs() < 1e-13,
        "cell at rest"
    );
}

#[test]
fn reconstruction_imposes_atmospheric_density_at_the_cell_velocity() {
    let n = 6;
    let mut grid = CellGrid2D::new(n, n, 1.0);
    let mut fs = FreeSurfaceState::new(n, n);
    fill_with_fluid(&mut fs);
    let interface = fs.idx(2, 2);
    fs.state[interface] = CellState::Interface;
    fs.fill[interface] = 0.5;
    fs.mass[interface] = 0.5;
    // Gas to the north (a = 2) and north-east (a = 5).
    let gas_links = [(2_usize, 2_usize, 3_usize), (5, 3, 3)];
    let post_collision = equilibrium(1.03, 0.04, 0.01);
    grid.cells[interface].f = post_collision;
    grid.cells[interface].rho = 1.03;
    grid.cells[interface].ux = 0.04;
    grid.cells[interface].uy = 0.01;
    for &(a, gx, gy) in &gas_links {
        let kg = fs.idx(gx, gy);
        fs.state[kg] = CellState::Gas;
        fs.fill[kg] = 0.0;
        fs.mass[kg] = 0.0;
        grid.cells[kg].f = [9.0; 9];
        grid.cells[kg].f[a] = post_collision[a];
    }

    reconstruct_gas_side(&mut grid, &fs);

    let cell = &grid.cells[interface];
    for &(a, _, _) in &gas_links {
        let q = OPP9[a];
        let closure = feq(RHO_ATM, cell.ux, cell.uy, a) + feq(RHO_ATM, cell.ux, cell.uy, q);
        assert!(
            (cell.f[q] + post_collision[a] - closure).abs() < 1e-12,
            "link {a}: f_{q} + f_{a}* = {} vs closure {closure}",
            cell.f[q] + post_collision[a]
        );
        // The outgoing population is not touched.
        assert_eq!(cell.f[a], post_collision[a]);
    }
    let (rho, ux, uy) = moments(&cell.f);
    assert!((cell.rho - rho).abs() < 1e-14);
    assert!((cell.ux - ux).abs() < 1e-14 && (cell.uy - uy).abs() < 1e-14);
}

#[test]
fn reconstruction_follows_the_periodic_lattice() {
    let (nx, ny) = (6, 5);
    let mut grid = CellGrid2D::new(nx, ny, 1.0);
    let mut fs = FreeSurfaceState::new(nx, ny);
    fill_with_fluid(&mut fs);

    // Cell A on the left edge: its periodic west neighbour (nx-1, 2) is fluid,
    // so nothing may be reconstructed (the old code treated the outside of the
    // array as gas).
    let a_cell = fs.idx(0, 2);
    fs.state[a_cell] = CellState::Interface;
    fs.fill[a_cell] = 0.5;
    fs.mass[a_cell] = 0.5;
    let stale: [f64; 9] = std::array::from_fn(|q| 3.0 + q as f64);
    grid.cells[a_cell].f = stale;

    // Cell B on the right edge of the bottom row: its periodic east neighbour
    // (0, 0) is gas, so f_3 (arriving from the east, moving west) is rebuilt.
    let b_cell = fs.idx(nx - 1, 0);
    let gas = fs.idx(0, 0);
    fs.state[b_cell] = CellState::Interface;
    fs.fill[b_cell] = 0.5;
    fs.mass[b_cell] = 0.5;
    fs.state[gas] = CellState::Gas;
    fs.fill[gas] = 0.0;
    fs.mass[gas] = 0.0;
    let sent = 0.05; // population b sent east, pulled by the gas cell
    grid.cells[gas].f = [9.0; 9];
    grid.cells[gas].f[1] = sent;
    let balanced = 2.0 * W9[3] * RHO_ATM - sent;
    grid.cells[b_cell].f = W9;
    grid.cells[b_cell].f[1] = balanced;
    grid.cells[b_cell].f[3] = 7.0;

    reconstruct_gas_side(&mut grid, &fs);

    assert_eq!(
        grid.cells[a_cell].f, stale,
        "a periodic fluid neighbour is not a gas neighbour"
    );
    let b = &grid.cells[b_cell];
    assert!(
        (b.f[3] - balanced).abs() < 1e-14,
        "periodic gas neighbour: f_3 = {}",
        b.f[3]
    );
    assert_eq!(b.f[1], balanced);
}

// ── F2541: mass exchange on post-stream populations ─────────────────────────

/// A 5×3 box whose only non-wall cells are `(1, 1)` and `(2, 1)`.
fn two_cell_box() -> (CellGrid2D, FreeSurfaceState) {
    let mut grid = CellGrid2D::new(5, 3, 1.0);
    for cell in grid.cells.iter_mut() {
        cell.obstacle = true;
    }
    let (i, j) = (grid.idx(1, 1), grid.idx(2, 1));
    grid.cells[i].obstacle = false;
    grid.cells[j].obstacle = false;
    let mut fs = FreeSurfaceState::new(5, 3);
    fs.state[i] = CellState::Interface;
    fs.fill[i] = 0.5;
    fs.mass[i] = 0.5;
    fs.state[j] = CellState::Fluid;
    fs.fill[j] = 1.0;
    fs.mass[j] = 1.0;
    (grid, fs)
}

#[test]
fn mass_exchange_uses_the_post_stream_populations() {
    let (mut grid, mut fs) = two_cell_box();
    let (i, j) = (grid.idx(1, 1), grid.idx(2, 1));
    // j lies east of i (a = 1).  After streaming, f_3(i) arrived from j and
    // f_1(j) is what i sent to j.
    grid.cells[i].f[3] = 0.3;
    grid.cells[j].f[1] = 0.1;
    // The pre-stream pairing (f_3(j) − f_1(i)) is different and must not be used.
    grid.cells[j].f[3] = 0.9;
    grid.cells[i].f[1] = 0.6;

    let delta = exchange_mass(&grid, &fs);
    assert!(
        (delta[i] - (0.3 - 0.1)).abs() < 1e-14,
        "fluid neighbour: Δm = {}, expected inflow − outflow = 0.2",
        delta[i]
    );
    assert_eq!(delta[j], 0.0, "fluid cells keep their mass equal to ρ");

    // Interface neighbour: weighted with the mean fill 0.4, antisymmetric.
    fs.state[j] = CellState::Interface;
    fs.fill[j] = 0.3;
    let delta = exchange_mass(&grid, &fs);
    assert!((delta[i] - 0.2 * 0.4).abs() < 1e-14, "Δm_i = {}", delta[i]);
    assert!((delta[j] + 0.2 * 0.4).abs() < 1e-14, "Δm_j = {}", delta[j]);

    // Gas neighbours exchange nothing.
    fs.state[j] = CellState::Gas;
    let delta = exchange_mass(&grid, &fs);
    assert_eq!(delta[i], 0.0);
}

#[test]
fn mass_exchange_wraps_periodically_and_ignores_walls() {
    // Only (0, 1) and (3, 1) are liquid: they are periodic neighbours.
    let mut grid = CellGrid2D::new(4, 3, 1.0);
    for cell in grid.cells.iter_mut() {
        cell.obstacle = true;
    }
    let (i, j) = (grid.idx(0, 1), grid.idx(3, 1));
    grid.cells[i].obstacle = false;
    grid.cells[j].obstacle = false;
    let mut fs = FreeSurfaceState::new(4, 3);
    fs.state[i] = CellState::Interface;
    fs.fill[i] = 0.5;
    fs.mass[i] = 0.5;
    fs.state[j] = CellState::Fluid;
    fs.fill[j] = 1.0;
    fs.mass[j] = 1.0;
    // j is west of i across the periodic boundary (a = 3): the inflow is f_1(i),
    // the outflow is f_3(j).  Every wall link carries stale junk.
    grid.cells[i].f = [4.0; 9];
    grid.cells[i].f[1] = 0.4;
    grid.cells[j].f[3] = 0.15;
    let delta = exchange_mass(&grid, &fs);
    assert!(
        (delta[i] - 0.25).abs() < 1e-14,
        "periodic fluid neighbour: Δm = {}",
        delta[i]
    );
}

// ── hydrostatic pool and closed-box conservation ────────────────────────────

#[test]
fn hydrostatic_pool_stays_at_rest_without_mass_correction() {
    let (nx, ny, surface_row) = (16, 14, 8);
    let (omega, g) = (1.0, 1.0e-4);
    let (mut grid, mut fs) = hydrostatic_pool(nx, ny, surface_row, 0.5);
    let mass_before = liquid_mass(&grid, &fs);
    for _ in 0..500 {
        gravity_step(&mut grid, &mut fs, omega, g);
    }
    let speed = max_physical_speed(&grid, &fs, 0.0, -g);
    assert!(
        speed < 1e-4,
        "spurious current of the flat pool: {speed:.3e}"
    );
    let drift = (liquid_mass(&grid, &fs) - mass_before).abs() / mass_before;
    assert!(drift < 1e-10, "liquid mass drift {drift:.3e}");

    // The flat surface neither moved nor was converted.
    for iy in 1..ny - 1 {
        for ix in 1..nx - 1 {
            let expected = match iy.cmp(&surface_row) {
                std::cmp::Ordering::Less => CellState::Fluid,
                std::cmp::Ordering::Equal => CellState::Interface,
                std::cmp::Ordering::Greater => CellState::Gas,
            };
            assert_eq!(fs.state[fs.idx(ix, iy)], expected, "cell ({ix},{iy})");
        }
    }
    let fills: Vec<f64> = (1..nx - 1)
        .map(|ix| fs.fill[fs.idx(ix, surface_row)])
        .collect();
    let (lowest, highest) = fills
        .iter()
        .fold((f64::MAX, f64::MIN), |(lo, hi), &f| (lo.min(f), hi.max(f)));
    assert!(
        lowest > 0.4 && highest < 0.6,
        "the surface layer stays half full: [{lowest}, {highest}]"
    );

    // The pressure rises linearly with depth: Δρ = ρ g Δy / cs² = 3 g Δy.
    let mid = nx / 2;
    let rho_top = grid.cells[grid.idx(mid, surface_row)].rho;
    let rho_bottom = grid.cells[grid.idx(mid, 1)].rho;
    let expected = 3.0 * g * (surface_row - 1) as f64;
    assert!(
        ((rho_bottom - rho_top) - expected).abs() < 0.03 * expected,
        "hydrostatic density rise {} vs {expected}",
        rho_bottom - rho_top
    );
}

#[test]
fn closed_box_dam_break_conserves_mass_without_correction() {
    for &(nx, ny, height, g, steps) in &[
        (60, 40, 20, 5.0e-4, 200),
        (40, 30, 12, 1.0e-3, 300),
        (50, 24, 16, 5.0e-4, 400),
    ] {
        let mut grid = walled_grid(nx, ny);
        let mut fs = FreeSurfaceState::new_dam_break(nx, ny, height);
        fs.clear_obstacles(&grid);
        let before = liquid_mass(&grid, &fs);
        for _ in 0..steps {
            gravity_step(&mut grid, &mut fs, 1.0, g);
        }
        let drift = (liquid_mass(&grid, &fs) - before).abs() / before;
        assert!(
            drift < 1e-3,
            "{nx}x{ny}, h = {height}: relative mass change {drift:.3e} after {steps} steps"
        );
        assert!(drift < 1e-9, "{nx}x{ny}: drift {drift:.3e} beyond rounding");
        assert!(
            grid.cells
                .iter()
                .all(|c| c.obstacle || (c.rho.is_finite() && c.ux.is_finite() && c.uy.is_finite())),
            "{nx}x{ny}: the flow must stay finite"
        );
    }
}

#[test]
fn walls_are_half_way_bounce_back_and_do_not_leak() {
    // A box full of liquid with random populations: the walls may neither
    // create nor destroy mass.
    let (nx, ny) = (12, 10);
    let mut grid = walled_grid(nx, ny);
    let mut fs = FreeSurfaceState::new(nx, ny);
    let mut state = 11_u64;
    for iy in 1..ny - 1 {
        for ix in 1..nx - 1 {
            let k = fs.idx(ix, iy);
            fs.state[k] = CellState::Fluid;
            fs.fill[k] = 1.0;
            fs.mass[k] = 1.0;
            let rho = 1.0 + 0.02 * next_unit(&mut state);
            let u = (0.03 * next_unit(&mut state), 0.03 * next_unit(&mut state));
            let cell = &mut grid.cells[k];
            cell.f = equilibrium(rho, u.0, u.1);
            cell.rho = rho;
            cell.ux = u.0;
            cell.uy = u.1;
        }
    }
    let population_mass = |grid: &CellGrid2D| -> f64 {
        grid.cells
            .iter()
            .filter(|c| !c.obstacle)
            .map(|c| c.f.iter().sum::<f64>())
            .sum()
    };
    let before = population_mass(&grid);
    for _ in 0..300 {
        free_surface_bgk_stream(&mut grid, &fs, 1.3, 0.0, 0.0);
        free_surface_step(&mut grid, &mut fs);
    }
    let after = population_mass(&grid);
    assert!(
        (after - before).abs() < 1e-12 * before,
        "population mass {before} -> {after}"
    );
    // Reflection also holds in a single step: a lone liquid cell surrounded
    // by walls gets back its own post-collision populations, reversed.
    let mut grid = walled_grid(5, 5);
    for cell in grid.cells.iter_mut() {
        cell.obstacle = true;
    }
    let k = grid.idx(2, 2);
    grid.cells[k].obstacle = false;
    let mut fs = FreeSurfaceState::new(5, 5);
    fs.state[k] = CellState::Fluid;
    fs.fill[k] = 1.0;
    fs.mass[k] = 1.0;
    let pocket: [f64; 9] = std::array::from_fn(|q| 0.05 * (q as f64 + 1.0));
    grid.cells[k].f = pocket;
    grid.cells[k].rho = pocket.iter().sum();
    // ω = 1 relaxes to equilibrium; use ω = 0 (pure streaming) to keep the
    // populations.
    free_surface_bgk_stream(&mut grid, &fs, 0.0, 0.0, 0.0);
    for q in 0..9 {
        assert!(
            (grid.cells[k].f[q] - pocket[OPP9[q]]).abs() < 1e-15,
            "population {q} must be the reflected population {}",
            OPP9[q]
        );
    }
}

#[test]
fn obstacle_entries_of_the_state_do_not_influence_the_solver() {
    let (nx, ny, surface_row) = (14, 12, 6);
    let (omega, g) = (1.0, 1.0e-4);
    let (mut grid_clean, mut fs_clean) = hydrostatic_pool(nx, ny, surface_row, 0.5);
    let (mut grid_dirty, mut fs_dirty) = hydrostatic_pool(nx, ny, surface_row, 0.5);
    // Flag the wall entries as fluid / interface (as `new_dam_break` does for
    // the walls inside its fluid block).
    for (k, cell) in grid_dirty.cells.iter().enumerate() {
        if cell.obstacle {
            fs_dirty.state[k] = if k % 2 == 0 {
                CellState::Fluid
            } else {
                CellState::Interface
            };
            fs_dirty.fill[k] = 1.0;
            fs_dirty.mass[k] = 1.0;
        }
    }
    for _ in 0..60 {
        gravity_step(&mut grid_clean, &mut fs_clean, omega, g);
        gravity_step(&mut grid_dirty, &mut fs_dirty, omega, g);
    }
    for k in 0..nx * ny {
        if grid_clean.cells[k].obstacle {
            continue;
        }
        assert_eq!(fs_clean.state[k], fs_dirty.state[k], "state of cell {k}");
        assert_eq!(fs_clean.fill[k], fs_dirty.fill[k], "fill of cell {k}");
        assert_eq!(fs_clean.mass[k], fs_dirty.mass[k], "mass of cell {k}");
        assert_eq!(
            grid_clean.cells[k].f, grid_dirty.cells[k].f,
            "f of cell {k}"
        );
    }
}

// ── conversions ─────────────────────────────────────────────────────────────

#[test]
fn filled_cell_promotes_its_gas_neighbours_from_the_liquid_equilibrium() {
    let n = 7;
    let mut grid = CellGrid2D::new(n, n, 1.0);
    let mut fs = FreeSurfaceState::new(n, n);
    let centre = fs.idx(3, 3);
    let (rho, ux, uy) = (1.05, 0.03, -0.02);
    fs.state[centre] = CellState::Interface;
    fs.fill[centre] = 1.5;
    fs.mass[centre] = 1.5 * rho;
    grid.cells[centre].f = equilibrium(rho, ux, uy);
    grid.cells[centre].rho = rho;
    grid.cells[centre].ux = ux;
    grid.cells[centre].uy = uy;
    // Stale populations in the gas that must not be inherited.
    for (k, cell) in grid.cells.iter_mut().enumerate() {
        if k != centre {
            cell.f = [8.0; 9];
        }
    }
    let mass_before = liquid_mass(&grid, &fs);

    convert_cells(&mut grid, &mut fs);

    assert_eq!(fs.state[centre], CellState::Fluid);
    assert!((fs.mass[centre] - rho).abs() < 1e-14);
    let share = 0.5 * rho / 8.0;
    let mut promoted = 0;
    for a in 1..9 {
        let kn = neighbour_index(n, n, 3, 3, a);
        assert_eq!(fs.state[kn], CellState::Interface, "neighbour {a}");
        assert!((fs.mass[kn] - share).abs() < 1e-14, "excess share {a}");
        assert_eq!(fs.fill[kn], 0.0, "fill is refreshed later");
        let expected = equilibrium(rho, ux, uy);
        for (q, (&got, &want)) in grid.cells[kn].f.iter().zip(expected.iter()).enumerate() {
            assert!((got - want).abs() < 1e-14, "neighbour {a}, population {q}");
        }
        assert!((grid.cells[kn].rho - rho).abs() < 1e-14);
        promoted += 1;
    }
    assert_eq!(promoted, 8);
    assert!(
        (liquid_mass(&grid, &fs) - mass_before).abs() < 1e-13,
        "the conversion conserves the liquid mass"
    );
}

#[test]
fn emptied_cell_demotes_its_fluid_neighbours_and_hands_back_its_mass() {
    let n = 7;
    let mut grid = CellGrid2D::new(n, n, 1.0);
    let mut fs = FreeSurfaceState::new(n, n);
    fill_with_fluid(&mut fs);
    let centre = fs.idx(3, 3);
    fs.state[centre] = CellState::Interface;
    fs.fill[centre] = -0.01;
    fs.mass[centre] = -0.01;
    let mass_before = liquid_mass(&grid, &fs);

    convert_cells(&mut grid, &mut fs);

    assert_eq!(fs.state[centre], CellState::Gas);
    assert_eq!(fs.mass[centre], 0.0);
    for a in 1..9 {
        let kn = neighbour_index(n, n, 3, 3, a);
        assert_eq!(fs.state[kn], CellState::Interface, "neighbour {a}");
        // ρ = 1 minus an eighth of the negative mass of the emptied cell.
        let expected = 1.0 - 0.01 / 8.0;
        assert!(
            (fs.mass[kn] - expected).abs() < 1e-14,
            "neighbour {a}: mass {} vs {expected}",
            fs.mass[kn]
        );
    }
    assert!(
        (liquid_mass(&grid, &fs) - mass_before).abs() < 1e-13,
        "the conversion conserves the liquid mass"
    );
}

#[test]
fn full_step_keeps_the_interface_layer_closed_around_an_overfilled_cell() {
    // A droplet sitting in the middle of a periodic domain without walls: an
    // overfilled interface cell turns to fluid and the layer grows around it.
    let n = 9;
    let mut grid = CellGrid2D::new(n, n, 1.0);
    let mut fs = FreeSurfaceState::new(n, n);
    let centre = fs.idx(4, 4);
    fs.state[centre] = CellState::Interface;
    fs.fill[centre] = 1.2;
    fs.mass[centre] = 1.2;
    let before = liquid_mass(&grid, &fs);
    free_surface_bgk_stream(&mut grid, &fs, 1.0, 0.0, 0.0);
    free_surface_step(&mut grid, &mut fs);
    // The lone cell keeps its mass (nothing to exchange with gas), turns to
    // fluid, and the eight cells around it become interface cells; the liquid
    // mass is conserved.
    assert!((liquid_mass(&grid, &fs) - before).abs() < 1e-12);
    assert_eq!(fs.state[centre], CellState::Fluid);
    for a in 1..9 {
        let kn = neighbour_index(n, n, 4, 4, a);
        assert_eq!(fs.state[kn], CellState::Interface, "neighbour {a}");
    }
    for k in 0..n * n {
        if fs.state[k] == CellState::Fluid {
            let (ix, iy) = (k % n, k / n);
            for a in 1..9 {
                let kn = neighbour_index(n, n, ix, iy, a);
                assert_ne!(fs.state[kn], CellState::Gas, "fluid cell {k} touches gas");
            }
        }
    }
}

// ── mass correction ─────────────────────────────────────────────────────────

#[test]
fn mass_correction_spreads_over_the_interface_cells_but_not_the_walls() {
    let grid = walled_grid(6, 6);
    let mut fs = FreeSurfaceState::new(6, 6);
    let real = [fs.idx(2, 2), fs.idx(3, 2)];
    let wall = fs.idx(0, 0); // obstacle cell wrongly flagged as interface
    for &k in &real {
        fs.state[k] = CellState::Interface;
        fs.fill[k] = 0.5;
        fs.mass[k] = 0.5;
    }
    fs.state[wall] = CellState::Interface;
    fs.fill[wall] = 0.5;
    fs.mass[wall] = 0.5;
    let target = fs.total_mass() + 0.3;
    apply_mass_correction(&grid, &mut fs, target);
    assert!((fs.total_mass() - target).abs() < 1e-14);
    for &k in &real {
        assert!((fs.mass[k] - 0.65).abs() < 1e-14, "interface cell mass");
    }
    assert_eq!(fs.mass[wall], 0.5, "the wall entry must not absorb mass");
}

#[test]
fn clear_obstacles_resets_wall_entries_only() {
    let (nx, ny) = (8, 6);
    let grid = walled_grid(nx, ny);
    let mut fs = FreeSurfaceState::new_dam_break(nx, ny, 4);
    let wall_mass: f64 = grid
        .cells
        .iter()
        .zip(fs.mass.iter())
        .filter(|(c, _)| c.obstacle)
        .map(|(_, &m)| m)
        .sum();
    assert!(wall_mass > 0.0, "test setup: the block covers walls");
    let total_before = fs.total_mass();
    fs.clear_obstacles(&grid);
    assert!((fs.total_mass() - (total_before - wall_mass)).abs() < 1e-12);
    for (k, cell) in grid.cells.iter().enumerate() {
        if cell.obstacle {
            assert_eq!(
                (fs.state[k], fs.fill[k], fs.mass[k]),
                (CellState::Gas, 0.0, 0.0)
            );
        }
    }
    // Interior entries of the fluid block are untouched.
    assert_eq!(fs.state[fs.idx(2, 2)], CellState::Fluid);
}
