// Copyright 2026 COOLJAPAN OU (Team KitaSan)
// SPDX-License-Identifier: Apache-2.0

//! Tests of the walls, the moving lid and the body force of `PyLbmGrid`.
//!
//! The grid used to be a periodic box in which `lid_driven_cavity()` could
//! never move (no walls, no lid) and in which a body acceleration injected
//! only half of its momentum. The tests below pin
//!
//! * the half-way bounce-back geometry and the momentum a moving lid injects,
//! * exact mass conservation of a closed box with a moving lid,
//! * the steady lid-driven cavity at `Re = 100` against Ghia, Ghia and Shin
//!   (J. Comput. Phys. 48, 1982),
//! * the Guo forcing: `rho a` of momentum per step for every relaxation rate
//!   and the plane Poiseuille profile of a solid-walled channel,
//! * the periodic scheme against an independent plain implementation.

use super::*;

/// Total mass of the fluid (solid cells read zero density).
fn total_mass(grid: &PyLbmGrid) -> f64 {
    grid.density_field().iter().sum()
}

/// Largest speed `|u|` over all cells.
fn max_speed(grid: &PyLbmGrid) -> f64 {
    grid.velocity_field()
        .chunks(2)
        .map(|u| (u[0] * u[0] + u[1] * u[1]).sqrt())
        .fold(0.0, f64::max)
}

/// A closed box with a lid.
fn closed_box(width: usize, height: usize, viscosity: f64, lid: [f64; 2]) -> PyLbmGrid {
    let mut config = PyLbmConfig::new(width, height, viscosity);
    config.walls = true;
    config.lid_velocity = lid;
    PyLbmGrid::new(&config)
}

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

#[test]
fn test_new_config_is_a_periodic_domain_without_lid_and_force() {
    let config = PyLbmConfig::new(10, 7, 0.1);
    assert!(!config.walls);
    assert_eq!(config.lid_velocity, [0.0, 0.0]);
    assert_eq!(config.body_force, [0.0, 0.0]);
}

#[test]
fn test_lid_driven_cavity_preset_is_a_closed_box_with_a_moving_lid_at_re_100() {
    let config = PyLbmConfig::lid_driven_cavity();
    assert_eq!((config.width, config.height), (64, 64));
    assert!(config.walls, "a cavity needs walls");
    assert_eq!(config.body_force, [0.0, 0.0]);
    assert_eq!(config.lid_velocity[1], 0.0);
    let lid_speed = config.lid_velocity[0];
    assert!(lid_speed > 0.0, "the lid must move");
    // Re = U H / nu with the cavity side H = 64 cells.
    let reynolds = lid_speed * config.height as f64 / config.viscosity;
    assert!((reynolds - 100.0).abs() < 1e-9, "Re = {reynolds}");
    // Mach number U / c_s = U sqrt(3) stays small and omega is stable.
    assert!(lid_speed * 3.0f64.sqrt() < 0.2);
    assert!(
        (0.0..2.0).contains(&config.omega()),
        "omega = {}",
        config.omega()
    );
}

#[test]
fn test_grid_copies_walls_and_lid_from_the_config() {
    let grid = PyLbmGrid::new(&PyLbmConfig::lid_driven_cavity());
    assert!(grid.walls);
    assert_eq!(grid.lid_velocity, [0.1, 0.0]);
    let grid = PyLbmGrid::new(&PyLbmConfig::new(6, 6, 0.1));
    assert!(!grid.walls);
    assert_eq!(grid.lid_velocity, [0.0, 0.0]);
}

// ---------------------------------------------------------------------------
// Half-way bounce-back and the moving lid
// ---------------------------------------------------------------------------

#[test]
fn test_lid_gives_the_top_row_a_third_of_its_speed_in_the_first_step() {
    // A diagonal link cut by the lid (e = (+-1, -1)) gains the population
    // 6 w rho (e . u_lid) = +-rho U / 6, and the vertical link none: the
    // momentum of the cell below the lid is rho U / 3 along x and 0 along y.
    // The links of the side walls and of the corners carry no lid term.
    let mut grid = closed_box(8, 6, 0.1, [0.06, 0.0]);
    grid.step();
    for x in 0..8 {
        let top = grid.velocity_at(x, 5);
        assert!(
            (top[0] - 0.02).abs() < 1e-14 && top[1].abs() < 1e-14,
            "top row, x = {x}: {top:?}"
        );
        assert!(
            (grid.density_at(x, 5) - 1.0).abs() < 1e-14,
            "the lid terms cancel, rho = {}",
            grid.density_at(x, 5)
        );
        for y in 0..5 {
            let v = grid.velocity_at(x, y);
            assert!(
                v[0].abs() < 1e-14 && v[1].abs() < 1e-14,
                "x = {x}, y = {y}: {v:?}"
            );
        }
    }
}

#[test]
fn test_reversing_the_lid_mirrors_the_flow() {
    // The D2Q9 scheme is mirror-covariant: a lid moving towards -x drives the
    // mirror image (x -> width - 1 - x, ux -> -ux) of the flow of a lid moving
    // towards +x. This pins the treatment of the two top corners and of the
    // side walls to be the same for both directions of the lid.
    let (w, h) = (10, 8);
    let mut right = closed_box(w, h, 0.1, [0.05, 0.0]);
    let mut left = closed_box(w, h, 0.1, [-0.05, 0.0]);
    for _ in 0..80 {
        right.step();
        left.step();
    }
    for x in 0..w {
        let r = right.velocity_at(x, h - 1)[0];
        let l = left.velocity_at(x, h - 1)[0];
        assert!(r > 0.0 && l < 0.0, "x = {x}: right {r}, left {l}");
    }
    for y in 0..h {
        for x in 0..w {
            let a = right.velocity_at(x, y);
            let b = left.velocity_at(w - 1 - x, y);
            assert!(
                (a[0] + b[0]).abs() < 1e-12 && (a[1] - b[1]).abs() < 1e-12,
                "({x}, {y}): {a:?} vs mirrored {b:?}"
            );
            let rho_a = right.density_at(x, y);
            let rho_b = left.density_at(w - 1 - x, y);
            assert!(
                (rho_a - rho_b).abs() < 1e-12,
                "({x}, {y}): rho {rho_a} vs {rho_b}"
            );
        }
    }
}

#[test]
fn test_closed_box_without_a_lid_stays_at_rest() {
    let mut grid = closed_box(12, 9, 0.05, [0.0, 0.0]);
    for _ in 0..300 {
        grid.step();
    }
    assert!(max_speed(&grid) < 1e-14, "speed {}", max_speed(&grid));
    for rho in grid.density_field() {
        assert!((rho - 1.0).abs() < 1e-14, "rho = {rho}");
    }
}

#[test]
fn test_lid_velocity_is_ignored_on_a_periodic_domain() {
    // The lid is a wall; without walls there is nothing to move.
    let mut config = PyLbmConfig::new(10, 10, 0.1);
    config.lid_velocity = [0.1, 0.0];
    let mut grid = PyLbmGrid::new(&config);
    for _ in 0..100 {
        grid.step();
    }
    assert!(max_speed(&grid) < 1e-14);
}

#[test]
fn test_closed_box_with_a_moving_lid_conserves_mass() {
    // Every link cut by a wall reflects exactly the population that left the
    // cell, and the lid corrections of the two diagonals cancel per cell, so
    // the total mass is constant up to rounding.
    let mut grid = closed_box(24, 24, 0.05, [0.08, 0.0]);
    let cells = (24 * 24) as f64;
    assert!((total_mass(&grid) - cells).abs() < 1e-12);
    for _ in 0..3000 {
        grid.step();
    }
    let mass = total_mass(&grid);
    assert!(
        (mass - cells).abs() < 1e-9,
        "mass drifted from {cells} to {mass}"
    );
    assert!(max_speed(&grid) > 0.01, "the lid must drive a flow");
}

#[test]
fn test_wall_reflects_the_opposite_population_halfway() {
    // A single cell next to the left wall with only the westward population
    // set: after one step it is the eastward population of that cell.
    let mut grid = closed_box(5, 5, 0.1, [0.0, 0.0]);
    // Collision at omega is skipped by writing the post-collision buffer
    // directly and streaming once.
    let cell = 2 * 5; // (x = 0, y = 2)
    for q in 0..9 {
        grid.f_tmp[cell * 9 + q] = 0.0;
    }
    grid.f_tmp[cell * 9 + 3] = 0.37; // direction (-1, 0): towards the wall
    grid.stream();
    // The population that arrives along direction 1 = (+1, 0) from the wall
    // side is the reflected westward one.
    assert_eq!(grid.f[cell * 9 + 1], 0.37);
}

#[test]
fn test_upstream_classifies_links_of_a_walled_box() {
    let grid = closed_box(4, 3, 0.1, [0.0, 0.0]);
    // Interior cell: every link comes from a fluid cell.
    for q in 0..9 {
        let sx = 1_isize - D2Q9_EX[q] as isize;
        let sy = 1_isize - D2Q9_EY[q] as isize;
        let expected = Upstream::Fluid((sy * 4 + sx) as usize);
        assert_eq!(grid.upstream(1, 1, q), expected, "q = {q}");
    }
    // Corner (0, 0): links from x = -1 or y = -1 are walls.
    assert_eq!(grid.upstream(0, 0, 1), Upstream::Wall); // from (-1, 0)
    assert_eq!(grid.upstream(0, 0, 2), Upstream::Wall); // from (0, -1)
    assert_eq!(grid.upstream(0, 0, 5), Upstream::Wall); // from (-1, -1)
    assert_eq!(grid.upstream(0, 0, 6), Upstream::Wall); // from (1, -1)
    assert_eq!(grid.upstream(0, 0, 0), Upstream::Fluid(0));
    assert_eq!(grid.upstream(0, 0, 7), Upstream::Fluid(4 + 1)); // from (1, 1)
    // Top row: links from y = height belong to the lid, corners included.
    assert_eq!(grid.upstream(0, 2, 4), Upstream::Lid); // from (0, 3)
    assert_eq!(grid.upstream(0, 2, 8), Upstream::Lid); // from (-1, 3)
    assert_eq!(grid.upstream(3, 2, 7), Upstream::Lid); // from (4, 3)
    assert_eq!(grid.upstream(0, 2, 1), Upstream::Wall); // from (-1, 2)
}

#[test]
fn test_without_walls_the_domain_is_periodic() {
    let grid = PyLbmGrid::new(&PyLbmConfig::new(4, 3, 0.1));
    // Cell (0, 0), direction 1 = (+1, 0): the source is (3, 0).
    assert_eq!(grid.upstream(0, 0, 1), Upstream::Fluid(3));
    // Direction 6 = (-1, +1): the source is (1, -1) = (1, 2).
    assert_eq!(grid.upstream(0, 0, 6), Upstream::Fluid(2 * 4 + 1));
}

// ---------------------------------------------------------------------------
// Solid cells
// ---------------------------------------------------------------------------

#[test]
fn test_set_solid_marks_cells_and_rejects_out_of_range_cells() {
    let mut grid = PyLbmGrid::new(&PyLbmConfig::new(6, 4, 0.1));
    assert!(!grid.is_solid(2, 1));
    assert!(grid.set_solid(2, 1, true));
    assert!(grid.is_solid(2, 1));
    assert!(!grid.is_solid(1, 2));
    assert!(!grid.set_solid(6, 0, true), "x out of range");
    assert!(!grid.set_solid(0, 4, true), "y out of range");
    assert!(!grid.is_solid(6, 0) && !grid.is_solid(0, 4));
    assert!(grid.set_solid(2, 1, false));
    assert!(!grid.is_solid(2, 1));
}

#[test]
fn test_solid_cells_report_zero_velocity_and_density() {
    let mut config = PyLbmConfig::new(8, 8, 0.1);
    config.body_force = [1e-4, 0.0];
    let mut grid = PyLbmGrid::new(&config);
    assert!(grid.set_solid(3, 3, true));
    assert!(grid.set_solid(4, 3, true));
    for _ in 0..200 {
        grid.step();
    }
    for x in [3, 4] {
        assert_eq!(grid.velocity_at(x, 3), [0.0, 0.0]);
        assert_eq!(grid.density_at(x, 3), 0.0);
    }
    assert!(grid.density_at(2, 3) > 0.9);
    let field = grid.velocity_field();
    assert_eq!(&field[(3 * 8 + 3) * 2..(3 * 8 + 3) * 2 + 2], &[0.0, 0.0]);
}

// ---------------------------------------------------------------------------
// Lid-driven cavity at Re = 100
// ---------------------------------------------------------------------------

/// `ux / U` on the vertical centre line, per row, for the cavity grid.
fn centre_line_ux(grid: &PyLbmGrid, lid_speed: f64) -> Vec<f64> {
    let (w, h) = (grid.width(), grid.height());
    (0..h)
        .map(|y| {
            // The geometric centre x = w / 2 lies between two columns.
            0.5 * (grid.velocity_at(w / 2 - 1, y)[0] + grid.velocity_at(w / 2, y)[0]) / lid_speed
        })
        .collect()
}

/// `uy / U` on the horizontal centre line, per column.
fn centre_line_uy(grid: &PyLbmGrid, lid_speed: f64) -> Vec<f64> {
    let (w, h) = (grid.width(), grid.height());
    (0..w)
        .map(|x| {
            0.5 * (grid.velocity_at(x, h / 2 - 1)[1] + grid.velocity_at(x, h / 2)[1]) / lid_speed
        })
        .collect()
}

/// Linear interpolation of a per-cell profile at the normalised coordinate
/// `s` in `[0, 1]` (cell `j` has its centre at `(j + 1/2) / n`).
fn sample(profile: &[f64], s: f64) -> f64 {
    let n = profile.len() as f64;
    let t = (s * n - 0.5).clamp(0.0, n - 1.0);
    let j = t.floor() as usize;
    let k = (j + 1).min(profile.len() - 1);
    let frac = t - j as f64;
    profile[j] * (1.0 - frac) + profile[k] * frac
}

/// Centre line `ux / U` of Ghia, Ghia and Shin, J. Comput. Phys. 48 (1982),
/// Table I, `Re = 100`: `(y / H, ux / U)` on the vertical line through the
/// centre of the cavity.
const GHIA_U: [(f64, f64); 15] = [
    (0.9766, 0.84123),
    (0.9688, 0.78871),
    (0.9609, 0.73722),
    (0.9531, 0.68717),
    (0.8516, 0.23151),
    (0.7344, 0.00332),
    (0.6172, -0.13641),
    (0.5000, -0.20581),
    (0.4531, -0.21090),
    (0.2813, -0.15662),
    (0.1719, -0.10150),
    (0.1016, -0.06434),
    (0.0703, -0.04775),
    (0.0625, -0.04192),
    (0.0547, -0.03717),
];

/// Centre line `uy / U` of the same reference, Table II: `(x / L, uy / U)` on
/// the horizontal line through the centre of the cavity.
const GHIA_V: [(f64, f64); 15] = [
    (0.9688, -0.05906),
    (0.9609, -0.07391),
    (0.9531, -0.08864),
    (0.9453, -0.10313),
    (0.9063, -0.16914),
    (0.8594, -0.22445),
    (0.8047, -0.24533),
    (0.5000, 0.05454),
    (0.2344, 0.17527),
    (0.2266, 0.17507),
    (0.1563, 0.16077),
    (0.0938, 0.12317),
    (0.0781, 0.10890),
    (0.0703, 0.10091),
    (0.0625, 0.09233),
];

#[test]
fn test_lid_driven_cavity_preset_reproduces_the_ghia_re_100_centre_lines() {
    // The 64 x 64 preset, 20 000 steps (about 30 lid transit times): the lid
    // drags the fluid to the right below it, the primary vortex turns
    // clockwise, and the flow returns to the left through the lower half.
    let config = PyLbmConfig::lid_driven_cavity();
    let lid_speed = config.lid_velocity[0];
    let mut grid = PyLbmGrid::new(&config);
    let cells = (config.width * config.height) as f64;
    for _ in 0..20_000 {
        grid.step();
    }
    assert!(
        (total_mass(&grid) - cells).abs() < 1e-7,
        "mass {} drifted from {cells}",
        total_mass(&grid)
    );

    // Direction of the flow on the vertical centre line, on the cell columns
    // next to the centre as well as on their average.
    for x in [31, 32] {
        for y in 0..32 {
            let ux = grid.velocity_at(x, y)[0];
            assert!(
                ux < 0.0,
                "ux({x}, {y}) = {ux} must be negative in the lower half"
            );
        }
        for y in 56..64 {
            let ux = grid.velocity_at(x, y)[0];
            assert!(ux > 0.0, "ux({x}, {y}) = {ux} must follow the lid");
        }
    }

    // Quantitative agreement with Ghia et al. (a few per cent of the lid
    // speed: the 64 x 64 half-way bounce-back lattice is second-order, the
    // table values are 129 x 129 finite differences).
    let ux = centre_line_ux(&grid, lid_speed);
    for (y, expected) in GHIA_U {
        let measured = sample(&ux, y);
        assert!(
            (measured - expected).abs() < 0.012,
            "ux/U at y/H = {y}: {measured}, Ghia {expected}"
        );
    }
    let uy = centre_line_uy(&grid, lid_speed);
    for (x, expected) in GHIA_V {
        let measured = sample(&uy, x);
        assert!(
            (measured - expected).abs() < 0.008,
            "uy/U at x/L = {x}: {measured}, Ghia {expected}"
        );
    }

    // Extrema of the primary vortex: min ux/U = -0.2109 near y/H = 0.45,
    // max uy/U = 0.1753 near x/L = 0.23 and min uy/U = -0.2453 near x/L = 0.80.
    let (j_min, u_min) = ux
        .iter()
        .copied()
        .enumerate()
        .fold(
            (0, f64::INFINITY),
            |best, (j, u)| if u < best.1 { (j, u) } else { best },
        );
    assert!((u_min + 0.2109).abs() < 0.012, "min ux/U = {u_min}");
    let y_min = (j_min as f64 + 0.5) / 64.0;
    assert!((0.40..0.52).contains(&y_min), "min ux/U at y/H = {y_min}");
    let v_max = uy.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let v_min = uy.iter().copied().fold(f64::INFINITY, f64::min);
    assert!((v_max - 0.1753).abs() < 0.008, "max uy/U = {v_max}");
    assert!((v_min + 0.2453).abs() < 0.008, "min uy/U = {v_min}");
}

// ---------------------------------------------------------------------------
// Guo body forcing
// ---------------------------------------------------------------------------

#[test]
fn test_fresh_grid_under_a_body_force_reads_zero_velocity() {
    // The populations hold rho (u - a / 2), so the velocity read back (which
    // adds a / 2) is the prescribed one: zero.
    let mut config = PyLbmConfig::new(6, 5, 0.1);
    config.body_force = [3e-4, -2e-4];
    let grid = PyLbmGrid::new(&config);
    for y in 0..5 {
        for x in 0..6 {
            let v = grid.velocity_at(x, y);
            assert!(
                v[0].abs() < 1e-15 && v[1].abs() < 1e-15,
                "({x}, {y}): {v:?}"
            );
            assert!((grid.density_at(x, y) - 1.0).abs() < 1e-15);
        }
    }
}

#[test]
fn test_body_force_adds_exactly_its_acceleration_per_step_for_every_omega() {
    // A uniform periodic fluid gains rho * a of momentum per step whatever the
    // relaxation rate (the old collision gained only half of it), so after N
    // steps the velocity is exactly N a.
    // omega = 1.887, 1.0 and 0.5
    for viscosity in [0.01, 1.0 / 6.0, 0.5] {
        let mut config = PyLbmConfig::new(8, 6, viscosity);
        config.body_force = [1e-5, -4e-6];
        let mut grid = PyLbmGrid::new(&config);
        let steps = 100;
        for _ in 0..steps {
            grid.step();
        }
        let expected = [steps as f64 * 1e-5, steps as f64 * -4e-6];
        for y in 0..6 {
            for x in 0..8 {
                let v = grid.velocity_at(x, y);
                assert!(
                    (v[0] - expected[0]).abs() < 1e-12 && (v[1] - expected[1]).abs() < 1e-12,
                    "nu {viscosity}, ({x}, {y}): {v:?}, expected {expected:?}"
                );
                assert!((grid.density_at(x, y) - 1.0).abs() < 1e-13);
            }
        }
    }
}

/// Fluid layers between the two solid rows of the Poiseuille channels (odd, so
/// that the centre line lies on a lattice row).
const CHANNEL_FLUID_LAYERS: usize = 21;
/// Centre-line speed the channel accelerations are chosen for.
const TARGET_CENTRE_SPEED: f64 = 0.02;
/// Relative tolerance of the discrete profile against the analytic one.
const PROFILE_TOLERANCE: f64 = 0.02;

/// Plane Poiseuille speed at distance `s` from a wall of a channel of width
/// `width`: `a s (H - s) / (2 nu)`.
fn poiseuille_speed(accel: f64, viscosity: f64, width: f64, s: f64) -> f64 {
    accel * s * (width - s) / (2.0 * viscosity)
}

/// A channel of `layers` fluid rows between two solid rows, periodic along x
/// and driven by a body acceleration, run to its steady state.
fn steady_channel(viscosity: f64, layers: usize) -> (PyLbmGrid, f64) {
    let width = layers as f64;
    let accel = 8.0 * viscosity * TARGET_CENTRE_SPEED / (width * width);
    let mut config = PyLbmConfig::new(3, layers + 2, viscosity);
    config.body_force = [accel, 0.0];
    let mut grid = PyLbmGrid::new(&config);
    for x in 0..3 {
        assert!(grid.set_solid(x, 0, true));
        assert!(grid.set_solid(x, layers + 1, true));
    }
    // About ten decay times H^2 / (pi^2 nu) of the slowest diffusive mode.
    let steps = (10.0 * width * width / (std::f64::consts::PI.powi(2) * viscosity)).ceil() as usize;
    for _ in 0..steps {
        grid.step();
    }
    (grid, accel)
}

fn check_poiseuille_channel(viscosity: f64) {
    let layers = CHANNEL_FLUID_LAYERS;
    let width = layers as f64;
    let (grid, accel) = steady_channel(viscosity, layers);
    let centre = layers / 2 + 1;

    // The half-way bounce-back walls sit half a cell outside the fluid rows.
    let mut worst = 0.0f64;
    let mut flux = 0.0;
    for y in 1..=layers {
        let expected = poiseuille_speed(accel, viscosity, width, y as f64 - 0.5);
        for x in 0..3 {
            let v = grid.velocity_at(x, y);
            worst = worst.max((v[0] - expected).abs());
            assert!(
                v[1].abs() < 1e-9,
                "nu {viscosity}: transverse speed {}",
                v[1]
            );
        }
        flux += grid.velocity_at(1, y)[0];
    }
    assert!(
        worst < PROFILE_TOLERANCE * TARGET_CENTRE_SPEED,
        "nu {viscosity}: the profile deviates by {worst} from the analytic one"
    );
    let u_max = grid.velocity_at(1, centre)[0];
    assert!(
        (u_max / TARGET_CENTRE_SPEED - 1.0).abs() < PROFILE_TOLERANCE,
        "nu {viscosity}: centre speed {u_max}, expected a H^2 / (8 nu) = {TARGET_CENTRE_SPEED}"
    );
    // Flow rate: a H^3 / (12 nu).
    let expected_flux = accel * width.powi(3) / (12.0 * viscosity);
    assert!(
        (flux / expected_flux - 1.0).abs() < PROFILE_TOLERANCE,
        "nu {viscosity}: flux {flux}, expected {expected_flux}"
    );
    // The solid rows hold no fluid, and the mass is conserved.
    assert_eq!(grid.velocity_at(1, 0), [0.0, 0.0]);
    assert_eq!(grid.velocity_at(1, layers + 1), [0.0, 0.0]);
    let cells = (3 * layers) as f64;
    assert!(
        (total_mass(&grid) - cells).abs() < 1e-8 * cells,
        "total mass {}",
        total_mass(&grid)
    );
}

#[test]
fn test_solid_walled_channel_reproduces_the_poiseuille_profile_at_omega_one() {
    check_poiseuille_channel(1.0 / 6.0);
}

#[test]
fn test_solid_walled_channel_reproduces_the_poiseuille_profile_at_a_smaller_viscosity() {
    check_poiseuille_channel(0.05);
}

#[test]
fn test_forced_fluid_blocked_by_a_wall_builds_the_hydrostatic_gradient() {
    // The forced fluid cannot pass the solid column that closes the periodic
    // box. At rest the pressure p = rho / 3 balances the force: the density
    // rises along the force with the slope d rho / d x = 3 rho a.
    let accel = 2e-4;
    let mut config = PyLbmConfig::new(12, 4, 0.1);
    config.body_force = [accel, 0.0];
    let mut grid = PyLbmGrid::new(&config);
    for y in 0..4 {
        assert!(grid.set_solid(5, y, true));
    }
    for _ in 0..3000 {
        grid.step();
    }
    // Columns downstream of the solid column: 6, 7, ..., 11, 0, ..., 4.
    let order: Vec<usize> = (6..12).chain(0..5).collect();
    let rho: Vec<f64> = order.iter().map(|&x| grid.density_at(x, 2)).collect();
    // Least-squares slope of rho against the position along the flow.
    let n = rho.len() as f64;
    let mean_s = (n - 1.0) / 2.0;
    let mean_rho = rho.iter().sum::<f64>() / n;
    let (mut cov, mut var) = (0.0, 0.0);
    for (i, r) in rho.iter().enumerate() {
        cov += (i as f64 - mean_s) * (r - mean_rho);
        var += (i as f64 - mean_s).powi(2);
    }
    let slope = cov / var;
    let expected = 3.0 * mean_rho * accel;
    assert!(
        (slope / expected - 1.0).abs() < 0.03,
        "density slope {slope}, expected 3 rho a = {expected}"
    );
    // Nothing flows any more, and nothing is lost.
    assert!(max_speed(&grid) < 1e-6, "speed {}", max_speed(&grid));
    let fluid_cells = (11 * 4) as f64;
    assert!((total_mass(&grid) - fluid_cells).abs() < 1e-9);
}

// ---------------------------------------------------------------------------
// Periodic scheme
// ---------------------------------------------------------------------------

/// Integer D2Q9 velocities, written out independently of the implementation.
const REF_EX: [isize; 9] = [0, 1, 0, -1, 0, 1, -1, -1, 1];
const REF_EY: [isize; 9] = [0, 0, 1, 0, -1, 1, 1, -1, -1];
const REF_W: [f64; 9] = [
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

/// Plain periodic BGK step (no walls, no force) of populations `f` on a
/// `w x h` grid.
fn reference_step(f: &[f64], w: usize, h: usize, omega: f64) -> Vec<f64> {
    let mut post = f.to_vec();
    for cell in 0..w * h {
        let pop = &f[cell * 9..cell * 9 + 9];
        let rho: f64 = pop.iter().sum();
        let ux: f64 = (0..9).map(|q| REF_EX[q] as f64 * pop[q]).sum::<f64>() / rho;
        let uy: f64 = (0..9).map(|q| REF_EY[q] as f64 * pop[q]).sum::<f64>() / rho;
        for q in 0..9 {
            let eu = REF_EX[q] as f64 * ux + REF_EY[q] as f64 * uy;
            let feq = REF_W[q] * rho * (1.0 + 3.0 * eu + 4.5 * eu * eu - 1.5 * (ux * ux + uy * uy));
            post[cell * 9 + q] = pop[q] * (1.0 - omega) + feq * omega;
        }
    }
    let mut out = vec![0.0; f.len()];
    for y in 0..h {
        for x in 0..w {
            for q in 0..9 {
                let sx = (x as isize - REF_EX[q]).rem_euclid(w as isize) as usize;
                let sy = (y as isize - REF_EY[q]).rem_euclid(h as isize) as usize;
                out[(y * w + x) * 9 + q] = post[(sy * w + sx) * 9 + q];
            }
        }
    }
    out
}

/// Fill the grid with the equilibrium of a smooth non-uniform flow.
fn set_smooth_flow(grid: &mut PyLbmGrid) {
    let (w, h) = (grid.width, grid.height);
    for y in 0..h {
        for x in 0..w {
            let sx = 2.0 * std::f64::consts::PI * x as f64 / w as f64;
            let sy = 2.0 * std::f64::consts::PI * y as f64 / h as f64;
            let rho = 1.0 + 0.02 * sx.sin() * sy.cos();
            let ux = 0.03 * sy.sin();
            let uy = 0.02 * sx.cos();
            let feq = equilibrium(rho, ux, uy);
            let cell = y * w + x;
            grid.f[cell * 9..cell * 9 + 9].copy_from_slice(&feq);
        }
    }
}

#[test]
fn test_periodic_grid_matches_a_plain_bgk_reference() {
    let (w, h) = (12, 9);
    let config = PyLbmConfig::new(w, h, 0.07);
    let mut grid = PyLbmGrid::new(&config);
    set_smooth_flow(&mut grid);
    let mut reference = grid.f.clone();
    for step in 0..40 {
        grid.step();
        reference = reference_step(&reference, w, h, config.omega());
        let worst = grid
            .f
            .iter()
            .zip(&reference)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0, f64::max);
        assert!(worst < 1e-13, "step {step}: populations differ by {worst}");
    }
}

#[test]
fn test_uniform_flow_in_a_periodic_box_is_preserved() {
    let mut grid = PyLbmGrid::new(&PyLbmConfig::new(7, 5, 0.1));
    let feq = equilibrium(1.2, 0.04, -0.03);
    for pop in grid.f.chunks_exact_mut(9) {
        pop.copy_from_slice(&feq);
    }
    for _ in 0..100 {
        grid.step();
    }
    for y in 0..5 {
        for x in 0..7 {
            let v = grid.velocity_at(x, y);
            assert!(
                (v[0] - 0.04).abs() < 1e-13 && (v[1] + 0.03).abs() < 1e-13,
                "({x}, {y}): {v:?}"
            );
            assert!((grid.density_at(x, y) - 1.2).abs() < 1e-13);
        }
    }
}

#[test]
fn test_periodic_shear_wave_decays_at_the_viscous_rate() {
    // u_x = U0 sin(k y) decays as exp(-nu k^2 t) with nu = (1 / omega - 1/2) / 3.
    let (w, h) = (4, 64);
    let viscosity = 0.1;
    let mut grid = PyLbmGrid::new(&PyLbmConfig::new(w, h, viscosity));
    let k = 2.0 * std::f64::consts::PI / h as f64;
    let u0 = 0.01;
    for y in 0..h {
        let feq = equilibrium(1.0, u0 * (k * y as f64).sin(), 0.0);
        for x in 0..w {
            let cell = y * w + x;
            grid.f[cell * 9..cell * 9 + 9].copy_from_slice(&feq);
        }
    }
    let steps = 1200;
    for _ in 0..steps {
        grid.step();
    }
    // Amplitude of the sin(k y) component, averaged over x.
    let amplitude = (0..h)
        .map(|y| {
            let mean_ux = (0..w).map(|x| grid.velocity_at(x, y)[0]).sum::<f64>() / w as f64;
            mean_ux * (k * y as f64).sin()
        })
        .sum::<f64>()
        * 2.0
        / h as f64;
    let expected = u0 * (-viscosity * k * k * steps as f64).exp();
    assert!(
        (amplitude / expected - 1.0).abs() < 0.01,
        "amplitude {amplitude}, expected {expected}"
    );
}

#[test]
fn test_empty_and_degenerate_grids_step_without_panicking() {
    for walls in [false, true] {
        let mut config = PyLbmConfig::new(0, 0, 0.1);
        config.walls = walls;
        config.lid_velocity = [0.1, 0.0];
        let mut grid = PyLbmGrid::new(&config);
        grid.step();
        assert_eq!(grid.step_count(), 1);
        assert!(grid.velocity_field().is_empty() && grid.density_field().is_empty());
        assert_eq!(grid.velocity_at(0, 0), [0.0, 0.0]);
        assert_eq!(grid.density_at(0, 0), 0.0);
        assert!(!grid.set_solid(0, 0, true));
        assert!(!grid.is_solid(0, 0));
    }
    // One-cell-wide strips and a single cell.
    for (w, h) in [(5, 1), (1, 5), (1, 1)] {
        let mut grid = closed_box(w, h, 0.1, [0.05, 0.0]);
        let cells = (w * h) as f64;
        for _ in 0..20 {
            grid.step();
        }
        assert!(
            (total_mass(&grid) - cells).abs() < 1e-12,
            "{w}x{h}: mass {}",
            total_mass(&grid)
        );
        assert!(max_speed(&grid).is_finite());
    }
}
