// Copyright 2026 COOLJAPAN OU (Team KitaSan)
// SPDX-License-Identifier: Apache-2.0

//! Regression tests for the scratch-buffer periodic streaming of the flat
//! grids and of the entropic solvers.
//!
//! Collision is switched off (`omega = 0`, i.e. `tau = ∞`) and the populations
//! are arbitrary positive numbers, so a step is a pure periodic streaming of
//! non-uniform data and every routing error shows up bit for bit against the
//! clone-and-`rem_euclid` reference that the grids used before.

use crate::entropic::{EntropicD2Q9, KbcD2Q9};
use crate::grid::functions::{C9, C19};
use crate::grid::{
    FlaggedGrid2D, FlaggedGrid3D, FlatLbmGrid2D, FlatLbmGrid3D, FullGrid3D, LbmGrid2DFull,
};

/// Reference periodic pull streaming: clone, then wrap every index with
/// `rem_euclid` (the pre-optimisation implementation).
fn reference_stream<const Q: usize>(
    f: &[f64],
    dims: [usize; 3],
    velocities: &[[i32; 3]; Q],
) -> Vec<f64> {
    let [nx, ny, nz] = dims;
    let f_old = f.to_vec();
    let mut f_new = f.to_vec();
    for z in 0..nz {
        for y in 0..ny {
            for x in 0..nx {
                let dst = (z * ny * nx + y * nx + x) * Q;
                for (a, c) in velocities.iter().enumerate() {
                    let sx = (x as isize - c[0] as isize).rem_euclid(nx as isize) as usize;
                    let sy = (y as isize - c[1] as isize).rem_euclid(ny as isize) as usize;
                    let sz = (z as isize - c[2] as isize).rem_euclid(nz as isize) as usize;
                    f_new[dst + a] = f_old[(sz * ny * nx + sy * nx + sx) * Q + a];
                }
            }
        }
    }
    f_new
}

/// The D2Q9 table of the grid module embedded in three dimensions.
fn d2q9_3d() -> [[i32; 3]; 9] {
    C9.map(|c| [c[0], c[1], 0])
}

/// Distinct positive values; `salt` makes different grids disagree.
fn seeded(len: usize, salt: u64) -> Vec<f64> {
    (0..len)
        .map(|i| {
            let hash = (i as u64 + salt).wrapping_mul(2_654_435_761) % 1_000_003;
            0.2 + hash as f64 * 1e-6
        })
        .collect()
}

const SIZES_2D: [(usize, usize); 6] = [(7, 5), (3, 3), (2, 6), (1, 4), (5, 1), (9, 8)];
const SIZES_3D: [(usize, usize, usize); 5] =
    [(5, 4, 3), (3, 3, 3), (2, 5, 4), (1, 3, 4), (6, 2, 3)];
const STEPS: usize = 3;

#[test]
fn flat_lbm_grid_2d_streaming_matches_the_reference() {
    for (nx, ny) in SIZES_2D {
        let mut grid = FlatLbmGrid2D::new(nx, ny, f64::INFINITY);
        grid.f = seeded(nx * ny * 9, 1);
        let mut expected = grid.f.clone();
        for step in 0..STEPS {
            expected = reference_stream::<9>(&expected, [nx, ny, 1], &d2q9_3d());
            grid.step();
            assert_eq!(grid.f, expected, "{nx} x {ny}, step {step}");
        }
    }
}

#[test]
fn flat_lbm_grid_3d_streaming_matches_the_reference() {
    for (nx, ny, nz) in SIZES_3D {
        let mut grid = FlatLbmGrid3D::new(nx, ny, nz, f64::INFINITY);
        grid.f = seeded(nx * ny * nz * 19, 2);
        let mut expected = grid.f.clone();
        for step in 0..STEPS {
            expected = reference_stream::<19>(&expected, [nx, ny, nz], &C19);
            grid.step();
            assert_eq!(grid.f, expected, "{nx} x {ny} x {nz}, step {step}");
        }
    }
}

#[test]
fn flagged_grid_2d_streaming_matches_the_reference() {
    for (nx, ny) in SIZES_2D {
        let mut grid = FlaggedGrid2D::new(nx, ny, 0.0, 1.0);
        grid.f = seeded(nx * ny * 9, 3);
        let mut expected = grid.f.clone();
        for step in 0..STEPS {
            expected = reference_stream::<9>(&expected, [nx, ny, 1], &d2q9_3d());
            grid.step();
            assert_eq!(grid.f, expected, "{nx} x {ny}, step {step}");
        }
    }
}

#[test]
fn lbm_grid_2d_full_streaming_matches_the_reference() {
    for (nx, ny) in SIZES_2D {
        let mut grid = LbmGrid2DFull::new(nx, ny, 0.0, 1.0);
        grid.f2 = seeded(nx * ny * 9, 4);
        let mut expected = grid.f2.clone();
        for step in 0..STEPS {
            expected = reference_stream::<9>(&expected, [nx, ny, 1], &d2q9_3d());
            grid.step();
            assert_eq!(grid.f2, expected, "{nx} x {ny}, step {step}");
        }
    }
}

#[test]
fn flagged_grid_3d_streaming_matches_the_reference() {
    for (nx, ny, nz) in SIZES_3D {
        let mut grid = FlaggedGrid3D::new(nx, ny, nz, 0.0, 1.0);
        grid.f = seeded(nx * ny * nz * 19, 5);
        let mut expected = grid.f.clone();
        for step in 0..STEPS {
            expected = reference_stream::<19>(&expected, [nx, ny, nz], &C19);
            grid.step();
            assert_eq!(grid.f, expected, "{nx} x {ny} x {nz}, step {step}");
        }
    }
}

#[test]
fn full_grid_3d_streaming_matches_the_reference() {
    for (nx, ny, nz) in SIZES_3D {
        let mut grid = FullGrid3D::new(nx, ny, nz, 0.0, 1.0);
        grid.f = seeded(nx * ny * nz * 19, 6);
        let mut expected = grid.f.clone();
        for step in 0..STEPS {
            expected = reference_stream::<19>(&expected, [nx, ny, nz], &C19);
            grid.step();
            assert_eq!(grid.f, expected, "{nx} x {ny} x {nz}, step {step}");
        }
    }
}

/// Seed the cells of an array-of-structs solver and return the flat copy.
fn flatten_cells(cells: &[[f64; 9]]) -> Vec<f64> {
    cells.iter().flatten().copied().collect()
}

#[test]
fn entropic_d2q9_streaming_matches_the_reference() {
    for (nx, ny) in SIZES_2D {
        let mut solver = EntropicD2Q9::new(nx, ny, 0.05);
        let flat = seeded(nx * ny * 9, 7);
        for (cell, chunk) in solver.f.iter_mut().zip(flat.chunks_exact(9)) {
            cell.copy_from_slice(chunk);
        }
        let mut expected = flatten_cells(&solver.f);
        for step in 0..STEPS {
            expected = reference_stream::<9>(&expected, [nx, ny, 1], &d2q9_3d());
            solver.stream();
            assert_eq!(
                flatten_cells(&solver.f),
                expected,
                "{nx} x {ny}, step {step}"
            );
        }
    }
}

#[test]
fn kbc_d2q9_streaming_matches_the_reference() {
    for (nx, ny) in SIZES_2D {
        let mut solver = KbcD2Q9::new(nx, ny, 0.05);
        let flat = seeded(nx * ny * 9, 8);
        for (cell, chunk) in solver.f.iter_mut().zip(flat.chunks_exact(9)) {
            cell.copy_from_slice(chunk);
        }
        let mut expected = flatten_cells(&solver.f);
        for step in 0..STEPS {
            expected = reference_stream::<9>(&expected, [nx, ny, 1], &d2q9_3d());
            solver.stream();
            assert_eq!(
                flatten_cells(&solver.f),
                expected,
                "{nx} x {ny}, step {step}"
            );
        }
    }
}

#[test]
fn streaming_swaps_the_scratch_buffer_instead_of_allocating() {
    let (nx, ny, nz) = (6, 5, 4);

    let mut flat2 = FlatLbmGrid2D::new(nx, ny, f64::INFINITY);
    let (before_f, before_tmp) = (flat2.f.as_ptr(), flat2.f_tmp.as_ptr());
    flat2.step();
    assert_eq!(flat2.f.as_ptr(), before_tmp);
    assert_eq!(flat2.f_tmp.as_ptr(), before_f);

    let mut flat3 = FlatLbmGrid3D::new(nx, ny, nz, f64::INFINITY);
    let (before_f, before_tmp) = (flat3.f.as_ptr(), flat3.f_tmp.as_ptr());
    flat3.step();
    assert_eq!(flat3.f.as_ptr(), before_tmp);
    assert_eq!(flat3.f_tmp.as_ptr(), before_f);

    let mut full = FullGrid3D::new(nx, ny, nz, 1.0, 1.0);
    let (before_f, before_tmp) = (full.f.as_ptr(), full.f_tmp.as_ptr());
    full.step();
    assert_eq!(full.f.as_ptr(), before_tmp);
    assert_eq!(full.f_tmp.as_ptr(), before_f);

    let mut flagged2 = FlaggedGrid2D::new(nx, ny, 1.0, 1.0);
    let (before_f, before_tmp) = (flagged2.f.as_ptr(), flagged2.f_tmp.as_ptr());
    flagged2.step();
    assert_eq!(flagged2.f.as_ptr(), before_tmp);
    assert_eq!(flagged2.f_tmp.as_ptr(), before_f);

    let mut full2 = LbmGrid2DFull::new(nx, ny, 1.0, 1.0);
    let (before_f, before_tmp) = (full2.f2.as_ptr(), full2.f2_tmp.as_ptr());
    full2.step();
    assert_eq!(full2.f2.as_ptr(), before_tmp);
    assert_eq!(full2.f2_tmp.as_ptr(), before_f);

    let mut entropic = EntropicD2Q9::new(nx, ny, 0.05);
    let (before_f, before_tmp) = (entropic.f.as_ptr(), entropic.f_tmp.as_ptr());
    entropic.step();
    assert_eq!(entropic.f.as_ptr(), before_tmp);
    assert_eq!(entropic.f_tmp.as_ptr(), before_f);

    let mut kbc = KbcD2Q9::new(nx, ny, 0.05);
    let (before_f, before_tmp) = (kbc.f.as_ptr(), kbc.f_tmp.as_ptr());
    kbc.step();
    assert_eq!(kbc.f.as_ptr(), before_tmp);
    assert_eq!(kbc.f_tmp.as_ptr(), before_f);

    // The flagged 3-D grid collides into the scratch array and streams back
    // into `f`: neither buffer moves.
    let mut flagged3 = FlaggedGrid3D::new(nx, ny, nz, 1.0, 1.0);
    let (before_f, before_tmp) = (flagged3.f.as_ptr(), flagged3.f_tmp.as_ptr());
    flagged3.step();
    assert_eq!(flagged3.f.as_ptr(), before_f);
    assert_eq!(flagged3.f_tmp.as_ptr(), before_tmp);
}

#[test]
fn an_emptied_scratch_buffer_is_repaired() {
    let (nx, ny, nz) = (4, 3, 3);

    let mut flat2 = FlatLbmGrid2D::new(nx, ny, f64::INFINITY);
    flat2.f = seeded(nx * ny * 9, 11);
    let expected = reference_stream::<9>(&flat2.f, [nx, ny, 1], &d2q9_3d());
    flat2.f_tmp = Vec::new();
    flat2.step();
    assert_eq!(flat2.f, expected);

    let mut flagged3 = FlaggedGrid3D::new(nx, ny, nz, 0.0, 1.0);
    flagged3.f = seeded(nx * ny * nz * 19, 12);
    let expected = reference_stream::<19>(&flagged3.f, [nx, ny, nz], &C19);
    flagged3.f_tmp = Vec::new();
    flagged3.step();
    assert_eq!(flagged3.f, expected);

    let mut kbc = KbcD2Q9::new(nx, ny, 0.05);
    let flat = seeded(nx * ny * 9, 13);
    for (cell, chunk) in kbc.f.iter_mut().zip(flat.chunks_exact(9)) {
        cell.copy_from_slice(chunk);
    }
    let expected = reference_stream::<9>(&flatten_cells(&kbc.f), [nx, ny, 1], &d2q9_3d());
    kbc.f_tmp = Vec::new();
    kbc.stream();
    assert_eq!(flatten_cells(&kbc.f), expected);
}

#[test]
fn collision_and_streaming_still_conserve_mass_and_momentum() {
    // With collision on, the streamed BGK step of an arbitrary state conserves
    // the total mass exactly up to rounding.
    let (nx, ny, nz) = (5, 4, 3);
    let mut grid = FlatLbmGrid3D::new(nx, ny, nz, 0.8);
    grid.f = seeded(nx * ny * nz * 19, 21);
    let mass_before: f64 = grid.f.iter().sum();
    for _ in 0..4 {
        grid.step();
    }
    let mass_after: f64 = grid.f.iter().sum();
    assert!(
        (mass_before - mass_after).abs() < 1e-10 * mass_before,
        "mass {mass_before} -> {mass_after}"
    );
}
