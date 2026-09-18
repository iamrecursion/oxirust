// Copyright 2026 COOLJAPAN OU (Team KitaSan)
// SPDX-License-Identifier: Apache-2.0

//! Regression tests for the zero-gradient outlet of the 3-D flagged grids.
//!
//! The outlet copies all 19 populations of the upstream neighbour
//! `(x − 1, y, z)`.  The initial state varies with `x`, `y` and `z`, so a copy
//! from any other cell (for example `(x, y − 1, z)`, which is what the outlet
//! used to read) produces visibly different populations.

use crate::grid::functions::{C19, W19, equilibrium_3d};
use crate::grid::{FlaggedGrid3D, FullGrid3D, NodeFlag};

const NX: usize = 6;
const NY: usize = 5;
const NZ: usize = 4;

/// Smoothly varying density and velocity of the cell `(x, y, z)`.
fn state(x: usize, y: usize, z: usize) -> (f64, [f64; 3]) {
    let (fx, fy, fz) = (x as f64, y as f64, z as f64);
    let rho = 1.0 + 0.02 * (0.7 * fx + 1.3 * fy + 0.4 * fz).sin();
    let u = [
        0.03 * (0.5 * fx + 0.9 * fy).cos(),
        0.02 * (1.1 * fy + 0.6 * fz).sin(),
        0.015 * (0.8 * fx + 0.3 * fz).cos(),
    ];
    (rho, u)
}

/// Equilibrium populations of `state(x, y, z)`.
fn populations(x: usize, y: usize, z: usize) -> [f64; 19] {
    let (rho, u) = state(x, y, z);
    let mut f = [0.0; 19];
    for (a, value) in f.iter_mut().enumerate() {
        *value = equilibrium_3d(
            W19[a],
            rho,
            u[0],
            u[1],
            u[2],
            C19[a][0] as f64,
            C19[a][1] as f64,
            C19[a][2] as f64,
        );
    }
    f
}

/// Largest absolute difference between the populations of two cells.
fn cell_difference(f: &[f64], idx_a: usize, idx_b: usize) -> f64 {
    (0..19)
        .map(|a| (f[idx_a * 19 + a] - f[idx_b * 19 + a]).abs())
        .fold(0.0, f64::max)
}

/// Fill a flat population array with the varying state.
fn fill(f: &mut [f64]) {
    for z in 0..NZ {
        for y in 0..NY {
            for x in 0..NX {
                let base = (z * NY * NX + y * NX + x) * 19;
                f[base..base + 19].copy_from_slice(&populations(x, y, z));
            }
        }
    }
}

/// Check that every outlet cell of the last column equals its x-neighbour and
/// that the outlet column itself is not uniform (so the test can tell the
/// neighbours apart).
fn assert_outlet_copies_the_x_neighbour(f: &[f64], label: &str) {
    let cell = |x: usize, y: usize, z: usize| z * NY * NX + y * NX + x;
    for z in 0..NZ {
        for y in 0..NY {
            let outlet = cell(NX - 1, y, z);
            let upstream = cell(NX - 2, y, z);
            assert_eq!(
                cell_difference(f, outlet, upstream),
                0.0,
                "{label}: outlet ({}, {y}, {z}) must equal ({}, {y}, {z})",
                NX - 1,
                NX - 2
            );
        }
    }
    // The cell below an outlet in y (wrapping to the previous z-plane at y = 0)
    // is where the old implementation read from; it must carry different data.
    for z in 0..NZ {
        for y in 0..NY {
            let outlet = cell(NX - 1, y, z);
            let (below_y, below_z) = if y > 0 {
                (y - 1, z)
            } else {
                (NY - 1, (z + NZ - 1) % NZ)
            };
            let below = cell(NX - 1, below_y, below_z);
            assert!(
                cell_difference(f, outlet, below) > 1e-6,
                "{label}: the outlet column must vary in y and z (y = {y}, z = {z})"
            );
        }
    }
}

#[test]
fn flagged_grid_3d_outlet_copies_the_upstream_x_neighbour() {
    let mut grid = FlaggedGrid3D::new(NX, NY, NZ, 1.2, 1.0);
    fill(&mut grid.f);
    for z in 0..NZ {
        for y in 0..NY {
            let idx = grid.cell_idx(NX - 1, y, z);
            grid.flags[idx] = NodeFlag::Outlet;
        }
    }
    for step in 0..4 {
        grid.step();
        assert_outlet_copies_the_x_neighbour(&grid.f, &format!("FlaggedGrid3D step {step}"));
        assert!(grid.f.iter().all(|value| value.is_finite()));
    }
}

#[test]
fn full_grid_3d_outlet_copies_the_upstream_x_neighbour() {
    let mut grid = FullGrid3D::new(NX, NY, NZ, 1.2, 1.0);
    fill(&mut grid.f);
    for z in 0..NZ {
        for y in 0..NY {
            let idx = grid.cell_idx(NX - 1, y, z);
            grid.flags[idx] = NodeFlag::Outlet;
        }
    }
    for step in 0..4 {
        grid.step();
        assert_outlet_copies_the_x_neighbour(&grid.f, &format!("FullGrid3D step {step}"));
        assert!(grid.f.iter().all(|value| value.is_finite()));
    }
}

#[test]
fn channel_with_inlet_and_outlet_keeps_the_zero_gradient_in_both_grids() {
    // Inlet column at x = 0, outlet column at x = NX - 1.
    let mut flagged = FlaggedGrid3D::new(NX, NY, NZ, 1.0, 1.0);
    let mut full = FullGrid3D::new(NX, NY, NZ, 1.0, 1.0);
    fill(&mut flagged.f);
    fill(&mut full.f);
    for z in 0..NZ {
        for y in 0..NY {
            flagged.set_inlet(0, y, z, 1.0, 0.04, 0.0);
            full.set_inlet(0, y, z, 1.0, 0.04, 0.0);
            let idx = flagged.cell_idx(NX - 1, y, z);
            flagged.flags[idx] = NodeFlag::Outlet;
            full.flags[idx] = NodeFlag::Outlet;
        }
    }
    for step in 0..3 {
        flagged.step();
        full.step();
        assert_outlet_copies_the_x_neighbour(&flagged.f, &format!("FlaggedGrid3D channel {step}"));
        assert_outlet_copies_the_x_neighbour(&full.f, &format!("FullGrid3D channel {step}"));
    }
}

#[test]
fn outlet_in_the_first_column_leaves_the_populations_alone() {
    // An outlet at x = 0 has no upstream neighbour.  FlaggedGrid3D collides
    // outlet nodes like fluid nodes, so the grid must evolve exactly like an
    // all-fluid grid (the old implementation copied the y - 1 neighbour for
    // every outlet node with idx >= nx).
    let mut with_outlets = FlaggedGrid3D::new(NX, NY, NZ, 1.1, 1.0);
    let mut reference = FlaggedGrid3D::new(NX, NY, NZ, 1.1, 1.0);
    fill(&mut with_outlets.f);
    fill(&mut reference.f);
    for z in 0..NZ {
        for y in 0..NY {
            let idx = with_outlets.cell_idx(0, y, z);
            with_outlets.flags[idx] = NodeFlag::Outlet;
        }
    }
    for step in 0..3 {
        with_outlets.step();
        reference.step();
        assert_eq!(with_outlets.f, reference.f, "step {step}");
    }
}

#[test]
fn outlet_in_the_first_z_plane_is_updated() {
    // The old FullGrid3D guard `idx >= nx * ny` skipped every outlet in the
    // z = 0 plane.
    let mut grid = FullGrid3D::new(NX, NY, NZ, 1.3, 1.0);
    fill(&mut grid.f);
    for y in 0..NY {
        let idx = grid.cell_idx(NX - 1, y, 0);
        grid.flags[idx] = NodeFlag::Outlet;
    }
    grid.step();
    for y in 0..NY {
        let outlet = grid.cell_idx(NX - 1, y, 0);
        let upstream = grid.cell_idx(NX - 2, y, 0);
        assert_eq!(
            cell_difference(&grid.f, outlet, upstream),
            0.0,
            "z = 0 outlet at y = {y}"
        );
    }
}
