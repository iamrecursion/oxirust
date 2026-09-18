// Copyright 2026 COOLJAPAN OU (Team KitaSan)
// SPDX-License-Identifier: Apache-2.0

//! Periodic pull streaming for flat population arrays.
//!
//! The flat grids of this module store the `Q` populations of a cell next to
//! each other (`f[cell * Q + a]`, `cell = x + nx * (y + ny * z)`).  A pull
//! step gives every cell the population that its upstream neighbour
//! `x − c_a` carried in direction `a`:
//!
//! ```text
//! dst[cell(x) * Q + a] = src[cell(x − c_a, periodic) * Q + a]
//! ```
//!
//! [`pull_stream_periodic`] performs that step between two separate arrays,
//! so a grid keeps one persistent scratch array and swaps it with the
//! population array after streaming; a step neither allocates nor copies the
//! whole array first.  Cells whose upstream neighbours all lie inside the
//! domain read from a constant offset per direction, and only the cells in
//! the one-cell-wide periodic shell wrap their coordinates.

use std::ops::Range;

/// D2Q9 lattice velocities `(c_x, c_y, 0)` in the ordering of
/// [`crate::lattice::D2Q9_VELOCITIES`], embedded in three dimensions for
/// [`pull_stream_periodic`] with `nz = 1`.
pub const D2Q9_VELOCITIES_3D: [[i32; 3]; 9] = [
    [0, 0, 0],
    [1, 0, 0],
    [0, 1, 0],
    [-1, 0, 0],
    [0, -1, 0],
    [1, 1, 0],
    [-1, 1, 0],
    [-1, -1, 0],
    [1, -1, 0],
];

/// Largest absolute velocity component along each axis.
fn velocity_margins<const Q: usize>(velocities: &[[i32; 3]; Q]) -> [usize; 3] {
    let mut margins = [0usize; 3];
    for velocity in velocities {
        for (margin, &component) in margins.iter_mut().zip(velocity.iter()) {
            *margin = (*margin).max(component.unsigned_abs() as usize);
        }
    }
    margins
}

/// Coordinates in `[margin, n − margin)`: the cells whose upstream neighbours
/// along this axis never leave `[0, n)`.  Empty when the axis is too short.
fn inner_range(n: usize, margin: usize) -> Range<usize> {
    if n > 2 * margin {
        margin..n - margin
    } else {
        0..0
    }
}

/// Wrap `coordinate − velocity` into `[0, n)` (periodic).
#[inline]
fn wrap(coordinate: usize, velocity: i32, n: usize) -> usize {
    (coordinate as isize - velocity as isize).rem_euclid(n as isize) as usize
}

/// Pull one cell whose upstream neighbours may cross the periodic seam.
#[inline]
fn pull_cell_wrapped<const Q: usize>(
    src: &[f64],
    dst: &mut [f64],
    cell: [usize; 3],
    dims: [usize; 3],
    velocities: &[[i32; 3]; Q],
) {
    let [x, y, z] = cell;
    let [nx, ny, nz] = dims;
    let base = ((z * ny + y) * nx + x) * Q;
    for (a, velocity) in velocities.iter().enumerate() {
        let sx = wrap(x, velocity[0], nx);
        let sy = wrap(y, velocity[1], ny);
        let sz = wrap(z, velocity[2], nz);
        dst[base + a] = src[((sz * ny + sy) * nx + sx) * Q + a];
    }
}

/// Periodic pull streaming `dst[cell(x) * Q + a] = src[cell(x − c_a) * Q + a]`
/// of a flat population array.
///
/// `dims = [nx, ny, nz]` is the grid size (`nz = 1` for a two-dimensional
/// grid whose velocities have `c_z = 0`, see [`D2Q9_VELOCITIES_3D`]) and the
/// cell index is `x + nx * (y + ny * z)`.  Every entry of `dst` is written.
/// The result is identical to wrapping every index with `rem_euclid`, but
/// only the cells at the periodic seam do that.
///
/// # Panics
///
/// Panics if `src` or `dst` is shorter than `nx * ny * nz * Q`.
pub fn pull_stream_periodic<const Q: usize>(
    src: &[f64],
    dst: &mut [f64],
    dims: [usize; 3],
    velocities: &[[i32; 3]; Q],
) {
    let [nx, ny, nz] = dims;
    let cells = nx * ny * nz;
    assert!(
        src.len() >= cells * Q && dst.len() >= cells * Q,
        "population arrays must hold nx * ny * nz * Q = {} values (src {}, dst {})",
        cells * Q,
        src.len(),
        dst.len()
    );
    let margins = velocity_margins(velocities);
    let x_inner = inner_range(nx, margins[0]);
    let y_inner = inner_range(ny, margins[1]);
    let z_inner = inner_range(nz, margins[2]);

    // Interior cells read `Q`-strided data at a fixed offset per direction:
    // the upstream cell is `cell − (c_x + nx (c_y + ny c_z))`.  The offset is
    // stored as the two's-complement wrap of its signed value so the source
    // index is a single wrapping subtraction.
    let plane = nx * ny;
    let mut offsets = [0usize; Q];
    for (offset, velocity) in offsets.iter_mut().zip(velocities.iter()) {
        let cell_offset = velocity[2] as isize * plane as isize
            + velocity[1] as isize * nx as isize
            + velocity[0] as isize;
        *offset = (cell_offset * Q as isize) as usize;
    }

    for z in 0..nz {
        for y in 0..ny {
            let row = (z * ny + y) * nx;
            let (wrapped_head, interior, wrapped_tail) =
                if z_inner.contains(&z) && y_inner.contains(&y) {
                    (0..x_inner.start, x_inner.clone(), x_inner.end..nx)
                } else {
                    (0..nx, 0..0, nx..nx)
                };
            for x in wrapped_head {
                pull_cell_wrapped(src, dst, [x, y, z], dims, velocities);
            }
            for x in interior {
                let base = (row + x) * Q;
                for (a, (target, &offset)) in dst[base..base + Q]
                    .iter_mut()
                    .zip(offsets.iter())
                    .enumerate()
                {
                    *target = src[(base + a).wrapping_sub(offset)];
                }
            }
            for x in wrapped_tail {
                pull_cell_wrapped(src, dst, [x, y, z], dims, velocities);
            }
        }
    }
}

/// Stream `f` with [`pull_stream_periodic`] through the scratch array `f_tmp`
/// and swap the two, so that `f` holds the streamed populations and `f_tmp`
/// the previous ones (which the next call overwrites completely).
///
/// `f_tmp` is resized to the length of `f` when needed (a no-op in the steady
/// state), which also repairs a scratch array that was emptied or replaced by
/// code that rebuilt `f` by hand.
pub fn pull_stream_swap<const Q: usize>(
    f: &mut Vec<f64>,
    f_tmp: &mut Vec<f64>,
    dims: [usize; 3],
    velocities: &[[i32; 3]; Q],
) {
    f_tmp.resize(f.len(), 0.0);
    pull_stream_periodic::<Q>(f, f_tmp, dims, velocities);
    std::mem::swap(f, f_tmp);
}

/// [`pull_stream_swap`] for grids that store one `[f64; Q]` array per cell.
pub fn pull_stream_swap_cells<const Q: usize>(
    f: &mut Vec<[f64; Q]>,
    f_tmp: &mut Vec<[f64; Q]>,
    dims: [usize; 3],
    velocities: &[[i32; 3]; Q],
) {
    f_tmp.resize(f.len(), [0.0; Q]);
    pull_stream_periodic::<Q>(f.as_flattened(), f_tmp.as_flattened_mut(), dims, velocities);
    std::mem::swap(f, f_tmp);
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::grid::functions::C19;

    /// The pre-optimisation streaming: clone the array and wrap with
    /// `rem_euclid` for every population.
    fn reference_stream<const Q: usize>(
        f: &[f64],
        dims: [usize; 3],
        velocities: &[[i32; 3]; Q],
    ) -> Vec<f64> {
        let [nx, ny, nz] = dims;
        let f_old = f.to_vec();
        let mut f_new = vec![0.0; f.len()];
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

    /// Distinct, non-uniform values so that every routing error is visible.
    fn seeded(len: usize) -> Vec<f64> {
        (0..len)
            .map(|i| ((i as u64 * 2_654_435_761) % 1_000_003) as f64 * 1e-6 + 0.5)
            .collect()
    }

    #[test]
    fn d2q9_table_matches_the_lattice_module() {
        for (embedded, reference) in D2Q9_VELOCITIES_3D
            .iter()
            .zip(crate::lattice::D2Q9_VELOCITIES.iter())
        {
            assert_eq!(embedded[0], reference[0]);
            assert_eq!(embedded[1], reference[1]);
            assert_eq!(embedded[2], 0);
        }
    }

    #[test]
    fn matches_the_wrapping_reference_on_awkward_2d_sizes() {
        // Dimensions of 1, 2 and 3 have an empty or one-cell interior.
        for (nx, ny) in [
            (1, 1),
            (1, 5),
            (5, 1),
            (2, 2),
            (2, 7),
            (3, 3),
            (4, 3),
            (9, 6),
        ] {
            let f = seeded(nx * ny * 9);
            let expected = reference_stream::<9>(&f, [nx, ny, 1], &D2Q9_VELOCITIES_3D);
            let mut out = vec![-1.0; f.len()];
            pull_stream_periodic::<9>(&f, &mut out, [nx, ny, 1], &D2Q9_VELOCITIES_3D);
            assert_eq!(out, expected, "{nx} x {ny}");
        }
    }

    #[test]
    fn matches_the_wrapping_reference_on_awkward_3d_sizes() {
        for (nx, ny, nz) in [
            (1, 1, 1),
            (2, 3, 4),
            (3, 3, 3),
            (4, 2, 5),
            (5, 4, 3),
            (1, 4, 4),
            (6, 5, 4),
        ] {
            let f = seeded(nx * ny * nz * 19);
            let expected = reference_stream::<19>(&f, [nx, ny, nz], &C19);
            let mut out = vec![-1.0; f.len()];
            pull_stream_periodic::<19>(&f, &mut out, [nx, ny, nz], &C19);
            assert_eq!(out, expected, "{nx} x {ny} x {nz}");
        }
    }

    #[test]
    fn streaming_is_a_permutation_that_conserves_mass() {
        let (nx, ny, nz) = (7, 5, 6);
        let f = seeded(nx * ny * nz * 19);
        let mut out = vec![0.0; f.len()];
        pull_stream_periodic::<19>(&f, &mut out, [nx, ny, nz], &C19);
        let mut sorted_in = f.clone();
        let mut sorted_out = out.clone();
        sorted_in.sort_by(f64::total_cmp);
        sorted_out.sort_by(f64::total_cmp);
        assert_eq!(sorted_in, sorted_out);
    }

    #[test]
    fn rest_direction_is_copied_in_place() {
        let (nx, ny) = (6, 4);
        let f = seeded(nx * ny * 9);
        let mut out = vec![0.0; f.len()];
        pull_stream_periodic::<9>(&f, &mut out, [nx, ny, 1], &D2Q9_VELOCITIES_3D);
        for cell in 0..nx * ny {
            assert_eq!(out[cell * 9], f[cell * 9]);
        }
    }

    #[test]
    fn a_population_moves_one_cell_along_its_velocity() {
        // Mark direction 5 (c = (1, 1)) of the corner cell (nx - 1, ny - 1):
        // after streaming it must sit at (0, 0) through the periodic seam.
        let (nx, ny) = (4, 3);
        let mut f = vec![0.0; nx * ny * 9];
        f[((ny - 1) * nx + (nx - 1)) * 9 + 5] = 1.0;
        let mut out = vec![0.0; f.len()];
        pull_stream_periodic::<9>(&f, &mut out, [nx, ny, 1], &D2Q9_VELOCITIES_3D);
        assert_eq!(out[5], 1.0);
        assert_eq!(out.iter().sum::<f64>(), 1.0);
    }

    #[test]
    #[should_panic(expected = "population arrays must hold")]
    fn undersized_arrays_are_rejected() {
        let f = vec![0.0; 10];
        let mut out = vec![0.0; 10];
        pull_stream_periodic::<9>(&f, &mut out, [3, 3, 1], &D2Q9_VELOCITIES_3D);
    }
}
