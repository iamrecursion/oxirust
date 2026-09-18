// Copyright 2026 COOLJAPAN OU (Team KitaSan)
// SPDX-License-Identifier: Apache-2.0

//! Fast Sweeping Method for the Eikonal equation `|∇φ| = 1` on a voxel grid.
//!
//! The solver follows Zhao, *A fast sweeping method for Eikonal equations*,
//! Math. Comp. 74 (2005): Godunov upwind discretisation, Gauss–Seidel
//! iterations in the eight alternating sweep directions.

use super::{SdfVoxelGrid, VoxelCoord};

/// The eight sweep directions as `(reverse_x, reverse_y, reverse_z)`.
const SWEEP_ORDERS: [(bool, bool, bool); 8] = [
    (false, false, false),
    (true, false, false),
    (false, true, false),
    (true, true, false),
    (false, false, true),
    (true, false, true),
    (false, true, true),
    (true, true, true),
];

/// Fast Sweeping Method for computing the distance field from a set of seed
/// voxels.
///
/// The grid is swept in 8 alternating directions; every voxel is updated with
/// the Godunov upwind solution of the discrete Eikonal equation
/// `Σ max(u - u_axis, 0)² = h²`, where `u_axis` is the smaller of the two
/// neighbour values along each axis.  The sweeps are repeated until no voxel
/// changes by more than `1e-12 · h`.
pub struct FastSweeping;

impl FastSweeping {
    /// Compute the distance field propagated from `seeds` using fast sweeping.
    ///
    /// `seeds` is a list of `(voxel_coord, value)` pairs.  Seed voxels are the
    /// boundary condition of the Eikonal problem: they keep their value, and
    /// every other voxel receives the (first-order accurate) travel time from
    /// the seeds, i.e. `min_s (value_s + |x - s|)` up to the discretisation
    /// error of the scheme.  Seeds outside the grid and seeds with a non-finite
    /// value are ignored; if a voxel is seeded more than once the smallest value
    /// wins.  Voxels that no seed can reach (and every non-seed voxel when the
    /// cell size is not a positive finite number) stay at `f64::INFINITY`.
    ///
    /// Returns a vector of the same length as `grid.data`, indexed
    /// `ix + nx * (iy + ny * iz)`.
    pub fn compute(grid: &SdfVoxelGrid, seeds: &[(VoxelCoord, f64)]) -> Vec<f64> {
        let nx = grid.nx;
        let ny = grid.ny;
        let nz = grid.nz;
        let h = grid.cell_size;
        let n = nx * ny * nz;

        let mut phi = vec![f64::INFINITY; n];
        let mut frozen = vec![false; n];

        for &(coord, value) in seeds {
            if !value.is_finite() {
                continue;
            }
            let Some(idx) = seed_index(coord, nx, ny, nz) else {
                continue;
            };
            if !frozen[idx] || value < phi[idx] {
                phi[idx] = value;
            }
            frozen[idx] = true;
        }
        if n == 0 || !h.is_finite() || h <= 0.0 {
            return phi;
        }

        let tolerance = 1e-12 * h;
        // A pass is eight sweeps.  Convex seed sets converge in two passes (one
        // to propagate, one to confirm); the cap only guards against
        // pathological inputs.
        let max_passes = nx + ny + nz + 16;
        let plane = nx * ny;
        for _ in 0..max_passes {
            let mut max_change = 0.0_f64;
            for &(rev_x, rev_y, rev_z) in &SWEEP_ORDERS {
                for step_z in 0..nz {
                    let iz = if rev_z { nz - 1 - step_z } else { step_z };
                    for step_y in 0..ny {
                        let iy = if rev_y { ny - 1 - step_y } else { step_y };
                        for step_x in 0..nx {
                            let ix = if rev_x { nx - 1 - step_x } else { step_x };
                            let idx = ix + nx * (iy + ny * iz);
                            if frozen[idx] {
                                continue;
                            }
                            let along_x = axis_minimum(&phi, idx, 1, ix, nx);
                            let along_y = axis_minimum(&phi, idx, nx, iy, ny);
                            let along_z = axis_minimum(&phi, idx, plane, iz, nz);
                            let updated = godunov_update([along_x, along_y, along_z], h);
                            if updated < phi[idx] {
                                let change = if phi[idx].is_finite() {
                                    phi[idx] - updated
                                } else {
                                    f64::INFINITY
                                };
                                max_change = max_change.max(change);
                                phi[idx] = updated;
                            }
                        }
                    }
                }
            }
            if max_change < tolerance {
                break;
            }
        }
        phi
    }
}

/// Flat index of a seed coordinate, or `None` if it lies outside the grid.
fn seed_index(coord: VoxelCoord, nx: usize, ny: usize, nz: usize) -> Option<usize> {
    let ix = usize::try_from(coord.ix).ok().filter(|&i| i < nx)?;
    let iy = usize::try_from(coord.iy).ok().filter(|&i| i < ny)?;
    let iz = usize::try_from(coord.iz).ok().filter(|&i| i < nz)?;
    Some(ix + nx * (iy + ny * iz))
}

/// Smaller of the two neighbour values of `idx` along one axis.
///
/// `stride` is the flat-index distance between neighbours along the axis,
/// `pos` the voxel's coordinate on that axis and `len` the grid extent on it.
#[inline]
fn axis_minimum(phi: &[f64], idx: usize, stride: usize, pos: usize, len: usize) -> f64 {
    let lower = if pos > 0 {
        phi[idx - stride]
    } else {
        f64::INFINITY
    };
    let upper = if pos + 1 < len {
        phi[idx + stride]
    } else {
        f64::INFINITY
    };
    lower.min(upper)
}

/// Godunov upwind solution of the discrete Eikonal equation in a voxel.
///
/// `neighbours` holds the smallest neighbour value along each axis.  With the
/// values sorted as `a <= b <= c` the update is the root of
/// `(u-a)² = h²` (one axis), `(u-a)² + (u-b)² = h²` (two axes) or
/// `(u-a)² + (u-b)² + (u-c)² = h²` (three axes) that is consistent with the
/// upwind condition, i.e. the lowest-dimensional one with `u <= next value`.
/// Infinite values (not yet reached) never contribute.
pub(super) fn godunov_update(mut neighbours: [f64; 3], h: f64) -> f64 {
    neighbours.sort_by(|p, q| p.total_cmp(q));
    let [a, b, c] = neighbours;
    if !a.is_finite() {
        return f64::INFINITY;
    }
    // One-dimensional update.
    let mut u = a + h;
    if b.is_finite() && u > b {
        // Two-dimensional update: u = (a + b + sqrt(2 h² - (a - b)²)) / 2.
        // `u > b` implies |a - b| < h, so the radicand exceeds h².
        let radicand = 2.0 * h * h - (a - b) * (a - b);
        u = 0.5 * (a + b + radicand.max(0.0).sqrt());
        if c.is_finite() && u > c {
            // Three-dimensional update:
            // u = (S + sqrt(S² - 3 (Q - h²))) / 3 with S = a + b + c, Q = a² + b² + c².
            let sum = a + b + c;
            let sum_sq = a * a + b * b + c * c;
            let radicand = sum * sum - 3.0 * (sum_sq - h * h);
            u = (sum + radicand.max(0.0).sqrt()) / 3.0;
        }
    }
    u
}
