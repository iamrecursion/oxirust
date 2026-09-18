// Copyright 2026 COOLJAPAN OU (Team KitaSan)
// SPDX-License-Identifier: Apache-2.0

//! Voxel-based collision detection.
//!
//! Provides an SDF (Signed Distance Field) voxel grid, voxelization of triangle
//! meshes, sparse voxel octree, fast-sweeping distance field computation, ray vs
//! voxel grid intersection, sphere vs voxel grid, convex shape vs voxel grid, and
//! voxel-based collision response.
//!
//! # Overview
//!
//! The voxel grid stores per-cell SDF values: negative inside a solid, positive
//! outside, zero on the surface.  The [`SdfVoxelGrid`] struct is the central data
//! structure; helper types provide distance field computation ([`FastSweeping`]),
//! sparse storage ([`SparseVoxelOctree`]), and query primitives
//! ([`VoxelRayQuery`], [`VoxelSphereQuery`], [`VoxelConvexQuery`]).

use std::collections::HashMap;

mod fast_sweeping;
mod mesh_voxelizer;
mod ray_query;
#[cfg(test)]
mod regression_tests;

pub use fast_sweeping::FastSweeping;
pub use mesh_voxelizer::{MeshVoxelizer, TriangleClosestPoint, TriangleFeature};
pub use ray_query::{VoxelRayHit, VoxelRayQuery};

// ─────────────────────────────────────────────────────────────────────────────
// Basic 3-D integer and float coordinate types
// ─────────────────────────────────────────────────────────────────────────────

/// Integer voxel coordinate `(ix, iy, iz)`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct VoxelCoord {
    /// X index.
    pub ix: i32,
    /// Y index.
    pub iy: i32,
    /// Z index.
    pub iz: i32,
}

impl VoxelCoord {
    /// Create a new `VoxelCoord`.
    pub fn new(ix: i32, iy: i32, iz: i32) -> Self {
        Self { ix, iy, iz }
    }

    /// Return the Manhattan distance to another voxel coordinate.
    pub fn manhattan(&self, other: &Self) -> i32 {
        (self.ix - other.ix).abs() + (self.iy - other.iy).abs() + (self.iz - other.iz).abs()
    }

    /// Return the Chebyshev (L∞) distance to another voxel coordinate.
    pub fn chebyshev(&self, other: &Self) -> i32 {
        (self.ix - other.ix)
            .abs()
            .max((self.iy - other.iy).abs())
            .max((self.iz - other.iz).abs())
    }

    /// Return the 6-connected (face) neighbours of this voxel.
    pub fn face_neighbours(&self) -> [Self; 6] {
        [
            Self::new(self.ix - 1, self.iy, self.iz),
            Self::new(self.ix + 1, self.iy, self.iz),
            Self::new(self.ix, self.iy - 1, self.iz),
            Self::new(self.ix, self.iy + 1, self.iz),
            Self::new(self.ix, self.iy, self.iz - 1),
            Self::new(self.ix, self.iy, self.iz + 1),
        ]
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// SdfVoxelGrid
// ─────────────────────────────────────────────────────────────────────────────

/// A dense, axis-aligned SDF voxel grid.
///
/// Each voxel stores a signed distance value; negative values are inside the
/// solid, positive values outside, and zero is on the surface.  The world-space
/// position of voxel `(ix, iy, iz)` is:
/// `origin + (ix + 0.5, iy + 0.5, iz + 0.5) * cell_size`.
pub struct SdfVoxelGrid {
    /// World-space origin of the grid (minimum corner).
    pub origin: [f64; 3],
    /// Side length of one cubic voxel in world units.
    pub cell_size: f64,
    /// Number of voxels in the X direction.
    pub nx: usize,
    /// Number of voxels in the Y direction.
    pub ny: usize,
    /// Number of voxels in the Z direction.
    pub nz: usize,
    /// Flat array of SDF values, indexed `ix + nx*(iy + ny*iz)`.
    pub data: Vec<f64>,
}

impl SdfVoxelGrid {
    /// Create a new grid filled with `f64::INFINITY` (everything outside).
    ///
    /// - `origin` — world-space minimum corner
    /// - `cell_size` — side length of each cubic voxel
    /// - `nx`, `ny`, `nz` — grid resolution per axis
    pub fn new(origin: [f64; 3], cell_size: f64, nx: usize, ny: usize, nz: usize) -> Self {
        let data = vec![f64::INFINITY; nx * ny * nz];
        Self {
            origin,
            cell_size,
            nx,
            ny,
            nz,
            data,
        }
    }

    /// Total number of voxels in the grid.
    pub fn len(&self) -> usize {
        self.nx * self.ny * self.nz
    }

    /// Return `true` if the grid contains no voxels.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Convert a `(ix, iy, iz)` triple to a flat index.
    ///
    /// Returns `None` if the coordinate is out of bounds.
    pub fn to_index(&self, ix: usize, iy: usize, iz: usize) -> Option<usize> {
        if ix < self.nx && iy < self.ny && iz < self.nz {
            Some(ix + self.nx * (iy + self.ny * iz))
        } else {
            None
        }
    }

    /// Return the SDF value at `(ix, iy, iz)`, or `None` if out of bounds.
    pub fn get(&self, ix: usize, iy: usize, iz: usize) -> Option<f64> {
        self.to_index(ix, iy, iz).map(|idx| self.data[idx])
    }

    /// Set the SDF value at `(ix, iy, iz)`.
    ///
    /// Does nothing if the coordinate is out of bounds.
    pub fn set(&mut self, ix: usize, iy: usize, iz: usize, value: f64) {
        if let Some(idx) = self.to_index(ix, iy, iz) {
            self.data[idx] = value;
        }
    }

    /// Convert a world-space point to voxel coordinates.
    ///
    /// Returns `None` if the point is outside the grid.
    pub fn world_to_voxel(&self, p: [f64; 3]) -> Option<[usize; 3]> {
        let ix = ((p[0] - self.origin[0]) / self.cell_size).floor() as i64;
        let iy = ((p[1] - self.origin[1]) / self.cell_size).floor() as i64;
        let iz = ((p[2] - self.origin[2]) / self.cell_size).floor() as i64;
        if ix >= 0
            && iy >= 0
            && iz >= 0
            && (ix as usize) < self.nx
            && (iy as usize) < self.ny
            && (iz as usize) < self.nz
        {
            Some([ix as usize, iy as usize, iz as usize])
        } else {
            None
        }
    }

    /// Return the world-space centre of voxel `(ix, iy, iz)`.
    pub fn voxel_center(&self, ix: usize, iy: usize, iz: usize) -> [f64; 3] {
        [
            self.origin[0] + (ix as f64 + 0.5) * self.cell_size,
            self.origin[1] + (iy as f64 + 0.5) * self.cell_size,
            self.origin[2] + (iz as f64 + 0.5) * self.cell_size,
        ]
    }

    /// Return the voxel containing `p`, with the indices clamped into the grid.
    ///
    /// Unlike [`SdfVoxelGrid::world_to_voxel`] this also accepts points on the
    /// maximum faces of the grid box (which belong to the last voxel) and points
    /// outside the grid (which map to the nearest border voxel).  Returns `None`
    /// only if the grid is empty or `p` has a NaN coordinate.
    pub fn clamped_voxel(&self, p: [f64; 3]) -> Option<[usize; 3]> {
        if self.is_empty() {
            return None;
        }
        let dims = [self.nx, self.ny, self.nz];
        let mut coord = [0usize; 3];
        for axis in 0..3 {
            let cell = ((p[axis] - self.origin[axis]) / self.cell_size).floor();
            if cell.is_nan() {
                return None;
            }
            // `as usize` saturates, so negative values map to 0.
            coord[axis] = (cell.max(0.0) as usize).min(dims[axis] - 1);
        }
        Some(coord)
    }

    /// Trilinearly interpolate the SDF at world-space point `p`.
    ///
    /// The samples sit at the voxel centres.  Inside the grid box
    /// ([`SdfVoxelGrid::aabb`]) the interpolation stencil is clamped to the
    /// grid, so the field is defined right up to the box faces (it is constant
    /// over the outer half cell).  Returns `f64::INFINITY` outside the grid box
    /// and, as in [`SdfVoxelGrid::interpolate_clamped`], when a sample with a
    /// non-zero weight is itself infinite (an empty voxel).
    pub fn interpolate(&self, p: [f64; 3]) -> f64 {
        let (lo, hi) = self.aabb();
        for axis in 0..3 {
            // `contains` is false for a NaN coordinate as well.
            if !(lo[axis]..=hi[axis]).contains(&p[axis]) {
                return f64::INFINITY;
            }
        }
        self.interpolate_clamped(p)
    }

    /// Trilinearly interpolate the SDF at `p` with clamp-to-edge boundary
    /// handling: the stencil indices are clamped to `[0, n - 1]`, so a point
    /// outside the grid sees the value of the nearest border voxel centre.
    ///
    /// A sample whose weight is zero is skipped, so an infinite sample (an empty
    /// voxel) only makes the result infinite if the point actually lies in the
    /// part of the cell it influences.  Returns `f64::INFINITY` for an empty
    /// grid and `f64::NAN` for a NaN coordinate.
    pub fn interpolate_clamped(&self, p: [f64; 3]) -> f64 {
        if self.is_empty() || self.data.len() != self.len() {
            return f64::INFINITY;
        }
        let dims = [self.nx, self.ny, self.nz];
        let mut lower = [0usize; 3];
        let mut upper = [0usize; 3];
        let mut frac = [0.0_f64; 3];
        for axis in 0..3 {
            let top = (dims[axis] - 1) as f64;
            let local = (p[axis] - self.origin[axis]) / self.cell_size - 0.5;
            if local.is_nan() {
                return f64::NAN;
            }
            let local = local.clamp(0.0, top);
            let floor = local.floor();
            lower[axis] = floor as usize;
            upper[axis] = (lower[axis] + 1).min(dims[axis] - 1);
            frac[axis] = local - floor;
        }

        let mut acc = 0.0_f64;
        for di in 0..2_usize {
            let wx = if di == 0 { 1.0 - frac[0] } else { frac[0] };
            if wx == 0.0 {
                continue;
            }
            let ix = if di == 0 { lower[0] } else { upper[0] };
            for dj in 0..2_usize {
                let wy = if dj == 0 { 1.0 - frac[1] } else { frac[1] };
                if wy == 0.0 {
                    continue;
                }
                let iy = if dj == 0 { lower[1] } else { upper[1] };
                for dk in 0..2_usize {
                    let wz = if dk == 0 { 1.0 - frac[2] } else { frac[2] };
                    if wz == 0.0 {
                        continue;
                    }
                    let iz = if dk == 0 { lower[2] } else { upper[2] };
                    let val = self.data[ix + self.nx * (iy + self.ny * iz)];
                    if !val.is_finite() {
                        return val;
                    }
                    acc += val * wx * wy * wz;
                }
            }
        }
        acc
    }

    /// Compute the gradient of the SDF at voxel `(ix, iy, iz)` using central
    /// differences (one-sided differences on the grid boundary).
    ///
    /// Returns a unit-length gradient vector, or `[0, 0, 0]` if the voxel is out
    /// of bounds, the gradient is (numerically) zero or a difference is not
    /// finite.
    pub fn gradient(&self, ix: usize, iy: usize, iz: usize) -> [f64; 3] {
        if ix >= self.nx || iy >= self.ny || iz >= self.nz || self.data.len() != self.len() {
            return [0.0; 3];
        }
        let at = |x: usize, y: usize, z: usize| self.data[x + self.nx * (y + self.ny * z)];
        let (x0, x1) = (ix.saturating_sub(1), (ix + 1).min(self.nx - 1));
        let (y0, y1) = (iy.saturating_sub(1), (iy + 1).min(self.ny - 1));
        let (z0, z1) = (iz.saturating_sub(1), (iz + 1).min(self.nz - 1));
        let difference = |hi_value: f64, lo_value: f64, lo: usize, hi: usize| {
            if hi > lo {
                (hi_value - lo_value) / ((hi - lo) as f64 * self.cell_size)
            } else {
                0.0
            }
        };
        let dx = difference(at(x1, iy, iz), at(x0, iy, iz), x0, x1);
        let dy = difference(at(ix, y1, iz), at(ix, y0, iz), y0, y1);
        let dz = difference(at(ix, iy, z1), at(ix, iy, z0), z0, z1);
        let len = (dx * dx + dy * dy + dz * dz).sqrt();
        if len > 1e-14 {
            [dx / len, dy / len, dz / len]
        } else {
            [0.0; 3]
        }
    }

    /// Return the AABB of the grid as `(min, max)` world-space corners.
    pub fn aabb(&self) -> ([f64; 3], [f64; 3]) {
        let min = self.origin;
        let max = [
            self.origin[0] + self.nx as f64 * self.cell_size,
            self.origin[1] + self.ny as f64 * self.cell_size,
            self.origin[2] + self.nz as f64 * self.cell_size,
        ];
        (min, max)
    }

    /// Fill the entire grid with `value`.
    pub fn fill(&mut self, value: f64) {
        self.data.iter_mut().for_each(|v| *v = value);
    }

    /// Count voxels with SDF value strictly less than `threshold` (inside test).
    pub fn count_inside(&self, threshold: f64) -> usize {
        self.data.iter().filter(|&&v| v < threshold).count()
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// SparseVoxelOctree
// ─────────────────────────────────────────────────────────────────────────────

/// A sparse voxel octree for memory-efficient storage of large voxel grids.
///
/// Only occupied (non-default) voxels are stored.  The octree is represented
/// as a flat `HashMap` mapping `VoxelCoord` to `f64` SDF values plus a level
/// counter.
pub struct SparseVoxelOctree {
    /// Occupied voxels and their SDF values.
    pub cells: HashMap<VoxelCoord, f64>,
    /// World-space origin.
    pub origin: [f64; 3],
    /// Base cell size (leaf level).
    pub cell_size: f64,
    /// Default value returned for absent voxels.
    pub default_value: f64,
}

impl SparseVoxelOctree {
    /// Create an empty sparse octree.
    pub fn new(origin: [f64; 3], cell_size: f64, default_value: f64) -> Self {
        Self {
            cells: HashMap::new(),
            origin,
            cell_size,
            default_value,
        }
    }

    /// Insert a voxel SDF value at `coord`.
    pub fn insert(&mut self, coord: VoxelCoord, value: f64) {
        self.cells.insert(coord, value);
    }

    /// Remove a voxel from the octree.
    pub fn remove(&mut self, coord: &VoxelCoord) {
        self.cells.remove(coord);
    }

    /// Query the SDF value at `coord`.
    ///
    /// Returns `self.default_value` if not present.
    pub fn get(&self, coord: VoxelCoord) -> f64 {
        *self.cells.get(&coord).unwrap_or(&self.default_value)
    }

    /// Number of stored voxels.
    pub fn len(&self) -> usize {
        self.cells.len()
    }

    /// Return `true` if no voxels are stored.
    pub fn is_empty(&self) -> bool {
        self.cells.is_empty()
    }

    /// Convert a world-space point to an integer voxel coordinate.
    pub fn world_to_coord(&self, p: [f64; 3]) -> VoxelCoord {
        VoxelCoord::new(
            ((p[0] - self.origin[0]) / self.cell_size).floor() as i32,
            ((p[1] - self.origin[1]) / self.cell_size).floor() as i32,
            ((p[2] - self.origin[2]) / self.cell_size).floor() as i32,
        )
    }

    /// Convert an integer voxel coordinate to its world-space centre.
    pub fn coord_to_world(&self, coord: VoxelCoord) -> [f64; 3] {
        [
            self.origin[0] + (coord.ix as f64 + 0.5) * self.cell_size,
            self.origin[1] + (coord.iy as f64 + 0.5) * self.cell_size,
            self.origin[2] + (coord.iz as f64 + 0.5) * self.cell_size,
        ]
    }

    /// Return the minimum SDF value stored.
    pub fn min_value(&self) -> f64 {
        self.cells.values().copied().fold(f64::INFINITY, f64::min)
    }

    /// Return all voxel coordinates whose SDF is below `threshold`.
    pub fn inside_voxels(&self, threshold: f64) -> Vec<VoxelCoord> {
        self.cells
            .iter()
            .filter(|&(_, &v)| v < threshold)
            .map(|(&c, _)| c)
            .collect()
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// VoxelSphereQuery — sphere vs voxel grid
// ─────────────────────────────────────────────────────────────────────────────

/// Collision result between a sphere and a voxel SDF grid.
#[derive(Debug, Clone, Copy)]
pub struct VoxelSphereContact {
    /// World-space contact point on the surface.
    pub point: [f64; 3],
    /// Contact normal pointing from grid surface toward sphere centre.
    pub normal: [f64; 3],
    /// Penetration depth (positive = overlapping).
    pub depth: f64,
}

/// Sphere vs voxel grid SDF collision queries.
pub struct VoxelSphereQuery;

impl VoxelSphereQuery {
    /// Test whether a sphere at `center` with `radius` penetrates the voxel grid.
    ///
    /// Returns a contact if the SDF value at the sphere centre is less than `radius`.
    /// A centre outside the grid box, or in a region without data (infinite SDF),
    /// yields no contact.
    pub fn query(grid: &SdfVoxelGrid, center: [f64; 3], radius: f64) -> Option<VoxelSphereContact> {
        let sdf = grid.interpolate(center);
        if !sdf.is_finite() || sdf >= radius {
            return None;
        }
        let depth = radius - sdf;

        // Normal from the gradient at the voxel containing the centre; the centre
        // is inside the grid box here, so the clamped lookup never fails.
        let coord = grid.clamped_voxel(center)?;
        let normal = grid.gradient(coord[0], coord[1], coord[2]);
        let contact_point = [
            center[0] - sdf * normal[0],
            center[1] - sdf * normal[1],
            center[2] - sdf * normal[2],
        ];
        Some(VoxelSphereContact {
            point: contact_point,
            normal,
            depth,
        })
    }

    /// Collect all voxels within distance `radius` from `center`.
    pub fn voxels_in_sphere(grid: &SdfVoxelGrid, center: [f64; 3], radius: f64) -> Vec<[usize; 3]> {
        let cs = grid.cell_size;
        let r_cells = (radius / cs).ceil() as usize + 1;
        let base = grid.world_to_voxel(center);
        let (bx, by, bz) = match base {
            Some(c) => (c[0] as i64, c[1] as i64, c[2] as i64),
            None => return Vec::new(),
        };

        let mut result = Vec::new();
        for dz in -(r_cells as i64)..=(r_cells as i64) {
            for dy in -(r_cells as i64)..=(r_cells as i64) {
                for dx in -(r_cells as i64)..=(r_cells as i64) {
                    let ix = bx + dx;
                    let iy = by + dy;
                    let iz = bz + dz;
                    if ix < 0
                        || iy < 0
                        || iz < 0
                        || ix as usize >= grid.nx
                        || iy as usize >= grid.ny
                        || iz as usize >= grid.nz
                    {
                        continue;
                    }
                    let c = grid.voxel_center(ix as usize, iy as usize, iz as usize);
                    let d_sq = (c[0] - center[0]).powi(2)
                        + (c[1] - center[1]).powi(2)
                        + (c[2] - center[2]).powi(2);
                    if d_sq <= radius * radius {
                        result.push([ix as usize, iy as usize, iz as usize]);
                    }
                }
            }
        }
        result
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// VoxelConvexQuery — convex shape vs voxel grid
// ─────────────────────────────────────────────────────────────────────────────

/// An axis-aligned box (convex shape) for voxel collision queries.
#[derive(Debug, Clone, Copy)]
pub struct ConvexAabb {
    /// World-space minimum corner.
    pub min: [f64; 3],
    /// World-space maximum corner.
    pub max: [f64; 3],
}

impl ConvexAabb {
    /// Create an AABB from min/max corners.
    pub fn new(min: [f64; 3], max: [f64; 3]) -> Self {
        Self { min, max }
    }

    /// Create an AABB centred at `center` with half-extents `half`.
    pub fn from_center(center: [f64; 3], half: [f64; 3]) -> Self {
        Self {
            min: [
                center[0] - half[0],
                center[1] - half[1],
                center[2] - half[2],
            ],
            max: [
                center[0] + half[0],
                center[1] + half[1],
                center[2] + half[2],
            ],
        }
    }

    /// Return the centre of the AABB.
    pub fn center(&self) -> [f64; 3] {
        [
            (self.min[0] + self.max[0]) * 0.5,
            (self.min[1] + self.max[1]) * 0.5,
            (self.min[2] + self.max[2]) * 0.5,
        ]
    }

    /// Test overlap with another AABB.
    pub fn overlaps(&self, other: &ConvexAabb) -> bool {
        self.min[0] <= other.max[0]
            && self.max[0] >= other.min[0]
            && self.min[1] <= other.max[1]
            && self.max[1] >= other.min[1]
            && self.min[2] <= other.max[2]
            && self.max[2] >= other.min[2]
    }
}

/// Convex AABB vs voxel grid collision queries.
pub struct VoxelConvexQuery;

impl VoxelConvexQuery {
    /// Collect all voxels whose centres lie within `aabb`.
    pub fn voxels_in_aabb(grid: &SdfVoxelGrid, aabb: &ConvexAabb) -> Vec<[usize; 3]> {
        let cs = grid.cell_size;
        let ix0 = (((aabb.min[0] - grid.origin[0]) / cs).floor() as i64).max(0) as usize;
        let iy0 = (((aabb.min[1] - grid.origin[1]) / cs).floor() as i64).max(0) as usize;
        let iz0 = (((aabb.min[2] - grid.origin[2]) / cs).floor() as i64).max(0) as usize;
        let ix1 =
            (((aabb.max[0] - grid.origin[0]) / cs).ceil() as usize).min(grid.nx.saturating_sub(1));
        let iy1 =
            (((aabb.max[1] - grid.origin[1]) / cs).ceil() as usize).min(grid.ny.saturating_sub(1));
        let iz1 =
            (((aabb.max[2] - grid.origin[2]) / cs).ceil() as usize).min(grid.nz.saturating_sub(1));

        let mut result = Vec::new();
        for iz in iz0..=iz1 {
            for iy in iy0..=iy1 {
                for ix in ix0..=ix1 {
                    result.push([ix, iy, iz]);
                }
            }
        }
        result
    }

    /// Test whether any voxel inside `aabb` has SDF below `threshold`.
    pub fn any_inside(grid: &SdfVoxelGrid, aabb: &ConvexAabb, threshold: f64) -> bool {
        for [ix, iy, iz] in Self::voxels_in_aabb(grid, aabb) {
            if let Some(sdf) = grid.get(ix, iy, iz)
                && sdf < threshold
            {
                return true;
            }
        }
        false
    }

    /// Compute the minimum SDF value in the volume covered by `aabb`.
    pub fn min_sdf(grid: &SdfVoxelGrid, aabb: &ConvexAabb) -> f64 {
        Self::voxels_in_aabb(grid, aabb)
            .iter()
            .filter_map(|&[ix, iy, iz]| grid.get(ix, iy, iz))
            .fold(f64::INFINITY, f64::min)
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// VoxelCollisionResponse
// ─────────────────────────────────────────────────────────────────────────────

/// Result of a voxel collision response calculation.
#[derive(Debug, Clone, Copy)]
pub struct VoxelCollisionImpulse {
    /// Impulse magnitude applied to the dynamic body.
    pub magnitude: f64,
    /// Unit contact normal (pointing away from grid surface).
    pub normal: [f64; 3],
    /// World-space contact point.
    pub contact_point: [f64; 3],
}

/// Voxel-based collision response for a sphere against an SDF grid.
///
/// Computes a penalty impulse proportional to the penetration depth.
pub struct VoxelCollisionResponse;

impl VoxelCollisionResponse {
    /// Compute a penalty response impulse for a sphere penetrating the grid.
    ///
    /// - `contact` — detected contact from `VoxelSphereQuery::query`
    /// - `velocity` — current velocity of the sphere centre `[vx, vy, vz]`
    /// - `mass` — mass of the sphere
    /// - `stiffness` — penalty spring stiffness (force per unit depth)
    /// - `restitution` — coefficient of restitution (0 = fully inelastic)
    ///
    /// Returns the impulse to be applied to the sphere.
    pub fn penalty_impulse(
        contact: &VoxelSphereContact,
        velocity: [f64; 3],
        mass: f64,
        stiffness: f64,
        restitution: f64,
        _dt: f64,
    ) -> VoxelCollisionImpulse {
        let n = contact.normal;
        let vn = velocity[0] * n[0] + velocity[1] * n[1] + velocity[2] * n[2];
        // Penalty spring force magnitude
        let f_penalty = stiffness * contact.depth;
        // Restitution contribution: impulse to reverse normal velocity
        let j_restitution = if vn < 0.0 {
            -(1.0 + restitution) * vn * mass
        } else {
            0.0
        };
        let j_total = f_penalty * _dt + j_restitution;
        VoxelCollisionImpulse {
            magnitude: j_total,
            normal: n,
            contact_point: contact.point,
        }
    }

    /// Apply a voxel collision impulse to update sphere velocity.
    ///
    /// Returns the updated velocity `[vx, vy, vz]`.
    pub fn apply_impulse(
        velocity: [f64; 3],
        impulse: &VoxelCollisionImpulse,
        mass: f64,
    ) -> [f64; 3] {
        let n = impulse.normal;
        let dv = impulse.magnitude / mass.max(1e-14);
        [
            velocity[0] + dv * n[0],
            velocity[1] + dv * n[1],
            velocity[2] + dv * n[2],
        ]
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// DynamicVoxelUpdate
// ─────────────────────────────────────────────────────────────────────────────

/// Dynamic voxel update utilities for moving objects.
///
/// Stamps or un-stamps a sphere's occupancy into an SDF grid as it moves.
pub struct DynamicVoxelUpdate;

impl DynamicVoxelUpdate {
    /// Stamp a sphere with `radius` at `center` into `grid`.
    ///
    /// Each voxel centre's distance to `center` minus `radius` is written as the
    /// SDF if it improves (decreases) the stored value.
    pub fn stamp_sphere(grid: &mut SdfVoxelGrid, center: [f64; 3], radius: f64) {
        let voxels = VoxelSphereQuery::voxels_in_sphere(grid, center, radius + grid.cell_size);
        for [ix, iy, iz] in voxels {
            let c = grid.voxel_center(ix, iy, iz);
            let dist = ((c[0] - center[0]).powi(2)
                + (c[1] - center[1]).powi(2)
                + (c[2] - center[2]).powi(2))
            .sqrt();
            let sdf = dist - radius;
            if let Some(idx) = grid.to_index(ix, iy, iz)
                && sdf < grid.data[idx]
            {
                grid.data[idx] = sdf;
            }
        }
    }

    /// Erase a sphere from `grid` (restore to +INF in the sphere region).
    pub fn erase_sphere(grid: &mut SdfVoxelGrid, center: [f64; 3], radius: f64) {
        let voxels = VoxelSphereQuery::voxels_in_sphere(grid, center, radius + grid.cell_size);
        for [ix, iy, iz] in voxels {
            if let Some(idx) = grid.to_index(ix, iy, iz) {
                grid.data[idx] = f64::INFINITY;
            }
        }
    }

    /// Move a sphere from `old_center` to `new_center` in `grid`.
    pub fn move_sphere(
        grid: &mut SdfVoxelGrid,
        old_center: [f64; 3],
        new_center: [f64; 3],
        radius: f64,
    ) {
        Self::erase_sphere(grid, old_center, radius);
        Self::stamp_sphere(grid, new_center, radius);
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Internal math helpers
// ─────────────────────────────────────────────────────────────────────────────

#[inline]
fn sub3(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}

#[inline]
fn add3(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [a[0] + b[0], a[1] + b[1], a[2] + b[2]]
}

#[inline]
fn scale3(a: [f64; 3], s: f64) -> [f64; 3] {
    [a[0] * s, a[1] * s, a[2] * s]
}

#[inline]
fn dot3(a: [f64; 3], b: [f64; 3]) -> f64 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

#[inline]
fn cross3(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

#[inline]
fn dist_sq3(a: [f64; 3], b: [f64; 3]) -> f64 {
    let d = sub3(a, b);
    dot3(d, d)
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ---- VoxelCoord ----

    #[test]
    fn test_voxel_coord_new() {
        let c = VoxelCoord::new(1, 2, 3);
        assert_eq!(c.ix, 1);
        assert_eq!(c.iy, 2);
        assert_eq!(c.iz, 3);
    }

    #[test]
    fn test_voxel_coord_manhattan() {
        let a = VoxelCoord::new(0, 0, 0);
        let b = VoxelCoord::new(3, 4, 0);
        assert_eq!(a.manhattan(&b), 7);
    }

    #[test]
    fn test_voxel_coord_chebyshev() {
        let a = VoxelCoord::new(0, 0, 0);
        let b = VoxelCoord::new(3, 1, 2);
        assert_eq!(a.chebyshev(&b), 3);
    }

    #[test]
    fn test_voxel_coord_face_neighbours_count() {
        let c = VoxelCoord::new(5, 5, 5);
        let nb = c.face_neighbours();
        assert_eq!(nb.len(), 6);
    }

    #[test]
    fn test_voxel_coord_face_neighbours_symmetric() {
        let c = VoxelCoord::new(2, 2, 2);
        let nb = c.face_neighbours();
        for n in &nb {
            assert_eq!(c.manhattan(n), 1, "each face neighbour differs by 1");
        }
    }

    // ---- SdfVoxelGrid ----

    #[test]
    fn test_grid_new_dimensions() {
        let g = SdfVoxelGrid::new([0.0; 3], 0.1, 10, 10, 10);
        assert_eq!(g.nx, 10);
        assert_eq!(g.ny, 10);
        assert_eq!(g.nz, 10);
        assert_eq!(g.len(), 1000);
    }

    #[test]
    fn test_grid_is_empty_false() {
        let g = SdfVoxelGrid::new([0.0; 3], 0.1, 2, 2, 2);
        assert!(!g.is_empty());
    }

    #[test]
    fn test_grid_is_empty_true() {
        let g = SdfVoxelGrid::new([0.0; 3], 0.1, 0, 2, 2);
        assert!(g.is_empty());
    }

    #[test]
    fn test_grid_set_get() {
        let mut g = SdfVoxelGrid::new([0.0; 3], 1.0, 4, 4, 4);
        g.set(1, 2, 3, -0.5);
        assert!((g.get(1, 2, 3).unwrap() + 0.5).abs() < 1e-12);
    }

    #[test]
    fn test_grid_get_out_of_bounds_returns_none() {
        let g = SdfVoxelGrid::new([0.0; 3], 1.0, 4, 4, 4);
        assert!(g.get(10, 0, 0).is_none());
    }

    #[test]
    fn test_grid_to_index_round_trip() {
        let g = SdfVoxelGrid::new([0.0; 3], 1.0, 5, 5, 5);
        let idx = g.to_index(2, 3, 4).unwrap();
        // ix + nx*(iy + ny*iz)
        assert_eq!(idx, 2 + 5 * (3 + 5 * 4));
    }

    #[test]
    fn test_grid_world_to_voxel_basic() {
        let g = SdfVoxelGrid::new([0.0; 3], 1.0, 4, 4, 4);
        let coord = g.world_to_voxel([1.5, 0.5, 2.5]).unwrap();
        assert_eq!(coord, [1, 0, 2]);
    }

    #[test]
    fn test_grid_world_to_voxel_outside_returns_none() {
        let g = SdfVoxelGrid::new([0.0; 3], 1.0, 4, 4, 4);
        assert!(g.world_to_voxel([10.0, 0.5, 0.5]).is_none());
    }

    #[test]
    fn test_grid_voxel_center() {
        let g = SdfVoxelGrid::new([0.0; 3], 1.0, 4, 4, 4);
        let c = g.voxel_center(0, 0, 0);
        assert!((c[0] - 0.5).abs() < 1e-12);
        assert!((c[1] - 0.5).abs() < 1e-12);
        assert!((c[2] - 0.5).abs() < 1e-12);
    }

    #[test]
    fn test_grid_fill_and_count_inside() {
        let mut g = SdfVoxelGrid::new([0.0; 3], 1.0, 3, 3, 3);
        g.fill(-1.0);
        assert_eq!(g.count_inside(0.0), 27);
    }

    #[test]
    fn test_grid_aabb() {
        let g = SdfVoxelGrid::new([1.0, 2.0, 3.0], 0.5, 4, 4, 4);
        let (mn, mx) = g.aabb();
        assert!((mn[0] - 1.0).abs() < 1e-12);
        assert!((mx[0] - 3.0).abs() < 1e-12); // 1.0 + 4*0.5
    }

    #[test]
    fn test_grid_interpolate_constant() {
        let mut g = SdfVoxelGrid::new([0.0; 3], 1.0, 4, 4, 4);
        g.fill(2.0);
        let v = g.interpolate([1.5, 1.5, 1.5]);
        assert!(
            (v - 2.0).abs() < 1e-10,
            "constant field interpolates to 2.0, got {v}"
        );
    }

    #[test]
    fn test_grid_gradient_interior() {
        let mut g = SdfVoxelGrid::new([0.0; 3], 1.0, 5, 5, 5);
        // Linear field: sdf(ix,iy,iz) = ix (increases in x)
        for iz in 0..5 {
            for iy in 0..5 {
                for ix in 0..5 {
                    g.set(ix, iy, iz, ix as f64);
                }
            }
        }
        let grad = g.gradient(2, 2, 2);
        assert!(grad[0] > 0.9, "gradient should point in +x, got {:?}", grad);
    }

    // ---- MeshVoxelizer ----

    #[test]
    fn test_point_triangle_dist_sq_at_vertex() {
        let a = [0.0, 0.0, 0.0];
        let b = [1.0, 0.0, 0.0];
        let c = [0.0, 1.0, 0.0];
        let d = MeshVoxelizer::point_triangle_dist_sq(a, a, b, c);
        assert!(d < 1e-14, "distance to own vertex should be 0, got {d}");
    }

    #[test]
    fn test_point_triangle_dist_sq_above_centroid() {
        let a = [0.0, 0.0, 0.0];
        let b = [1.0, 0.0, 0.0];
        let c = [0.0, 1.0, 0.0];
        // Centroid = (1/3, 1/3, 0), point directly above
        let p = [1.0 / 3.0, 1.0 / 3.0, 1.0];
        let d = MeshVoxelizer::point_triangle_dist_sq(p, a, b, c);
        assert!(
            (d - 1.0).abs() < 1e-10,
            "point above centroid dist² should be 1, got {d}"
        );
    }

    #[test]
    fn test_sign_winding_above_positive() {
        let a = [0.0, 0.0, 0.0];
        let b = [1.0, 0.0, 0.0];
        let c = [0.0, 1.0, 0.0];
        let p = [0.0, 0.0, 1.0]; // above (positive normal side)
        let s = MeshVoxelizer::sign_winding(p, a, b, c);
        assert!(s > 0.0, "point above should have positive sign");
    }

    #[test]
    fn test_sign_winding_below_negative() {
        let a = [0.0, 0.0, 0.0];
        let b = [1.0, 0.0, 0.0];
        let c = [0.0, 1.0, 0.0];
        let p = [0.0, 0.0, -1.0]; // below
        let s = MeshVoxelizer::sign_winding(p, a, b, c);
        assert!(s < 0.0, "point below should have negative sign");
    }

    // ---- FastSweeping ----

    #[test]
    fn test_fast_sweeping_single_seed() {
        let grid = SdfVoxelGrid::new([0.0; 3], 1.0, 3, 3, 3);
        let seed = VoxelCoord::new(1, 1, 1);
        let phi = FastSweeping::compute(&grid, &[(seed, 0.0)]);
        // Centre voxel should be 0
        let centre_idx = 1 + 3 * (1 + 3);
        assert!(phi[centre_idx] < 1e-10, "seed voxel should have dist 0");
        // Corner voxels should be farther
        let corner_idx = 0; // (0,0,0)
        assert!(phi[corner_idx] > 0.5, "corner should be farther from seed");
    }

    #[test]
    fn test_fast_sweeping_two_seeds_boundary() {
        let grid = SdfVoxelGrid::new([0.0; 3], 1.0, 5, 1, 1);
        let s0 = VoxelCoord::new(0, 0, 0);
        let s4 = VoxelCoord::new(4, 0, 0);
        let phi = FastSweeping::compute(&grid, &[(s0, 0.0), (s4, 0.0)]);
        // Distance to the nearest seed along the line: 0, 1, 2, 1, 0 cells.
        let expected = [0.0, 1.0, 2.0, 1.0, 0.0];
        for (value, want) in phi.iter().zip(expected) {
            assert!((value - want).abs() < 1e-12, "got {phi:?}");
        }
    }

    // ---- SparseVoxelOctree ----

    #[test]
    fn test_sparse_octree_insert_get() {
        let mut tree = SparseVoxelOctree::new([0.0; 3], 1.0, f64::INFINITY);
        tree.insert(VoxelCoord::new(1, 2, 3), -0.5);
        assert!((tree.get(VoxelCoord::new(1, 2, 3)) + 0.5).abs() < 1e-12);
    }

    #[test]
    fn test_sparse_octree_missing_returns_default() {
        let tree = SparseVoxelOctree::new([0.0; 3], 1.0, 99.0);
        assert!((tree.get(VoxelCoord::new(0, 0, 0)) - 99.0).abs() < 1e-12);
    }

    #[test]
    fn test_sparse_octree_remove() {
        let mut tree = SparseVoxelOctree::new([0.0; 3], 1.0, f64::INFINITY);
        let c = VoxelCoord::new(1, 1, 1);
        tree.insert(c, 1.0);
        tree.remove(&c);
        assert!(tree.is_empty());
    }

    #[test]
    fn test_sparse_octree_len() {
        let mut tree = SparseVoxelOctree::new([0.0; 3], 1.0, f64::INFINITY);
        tree.insert(VoxelCoord::new(0, 0, 0), 0.0);
        tree.insert(VoxelCoord::new(1, 0, 0), 0.0);
        assert_eq!(tree.len(), 2);
    }

    #[test]
    fn test_sparse_octree_world_to_coord() {
        let tree = SparseVoxelOctree::new([0.0; 3], 1.0, f64::INFINITY);
        let c = tree.world_to_coord([1.7, 2.1, 0.9]);
        assert_eq!(c.ix, 1);
        assert_eq!(c.iy, 2);
        assert_eq!(c.iz, 0);
    }

    #[test]
    fn test_sparse_octree_inside_voxels() {
        let mut tree = SparseVoxelOctree::new([0.0; 3], 1.0, f64::INFINITY);
        tree.insert(VoxelCoord::new(0, 0, 0), -1.0);
        tree.insert(VoxelCoord::new(1, 0, 0), 2.0);
        let inside = tree.inside_voxels(0.0);
        assert_eq!(inside.len(), 1);
        assert_eq!(inside[0], VoxelCoord::new(0, 0, 0));
    }

    // ---- ConvexAabb ----

    #[test]
    fn test_aabb_overlap() {
        let a = ConvexAabb::new([0.0; 3], [1.0, 1.0, 1.0]);
        let b = ConvexAabb::new([0.5, 0.5, 0.5], [1.5, 1.5, 1.5]);
        assert!(a.overlaps(&b));
    }

    #[test]
    fn test_aabb_no_overlap() {
        let a = ConvexAabb::new([0.0; 3], [1.0, 1.0, 1.0]);
        let b = ConvexAabb::new([2.0, 0.0, 0.0], [3.0, 1.0, 1.0]);
        assert!(!a.overlaps(&b));
    }

    #[test]
    fn test_aabb_center() {
        let a = ConvexAabb::from_center([1.0, 2.0, 3.0], [0.5, 0.5, 0.5]);
        let c = a.center();
        assert!((c[0] - 1.0).abs() < 1e-12);
        assert!((c[1] - 2.0).abs() < 1e-12);
        assert!((c[2] - 3.0).abs() < 1e-12);
    }

    // ---- VoxelConvexQuery ----

    #[test]
    fn test_convex_query_voxels_in_aabb() {
        let g = SdfVoxelGrid::new([0.0; 3], 1.0, 4, 4, 4);
        let aabb = ConvexAabb::new([0.0; 3], [2.0, 2.0, 2.0]);
        let voxels = VoxelConvexQuery::voxels_in_aabb(&g, &aabb);
        assert!(!voxels.is_empty());
    }

    #[test]
    fn test_convex_query_any_inside_true() {
        let mut g = SdfVoxelGrid::new([0.0; 3], 1.0, 4, 4, 4);
        g.set(1, 1, 1, -0.5);
        let aabb = ConvexAabb::new([0.5, 0.5, 0.5], [2.5, 2.5, 2.5]);
        assert!(VoxelConvexQuery::any_inside(&g, &aabb, 0.0));
    }

    #[test]
    fn test_convex_query_min_sdf() {
        let mut g = SdfVoxelGrid::new([0.0; 3], 1.0, 4, 4, 4);
        g.fill(1.0);
        g.set(2, 2, 2, -3.0);
        let aabb = ConvexAabb::new([0.0; 3], [4.0, 4.0, 4.0]);
        let min = VoxelConvexQuery::min_sdf(&g, &aabb);
        assert!(
            (min + 3.0).abs() < 1e-12,
            "min SDF should be -3.0, got {min}"
        );
    }

    // ---- VoxelSphereQuery ----

    #[test]
    fn test_sphere_query_no_contact() {
        let g = SdfVoxelGrid::new([0.0; 3], 1.0, 4, 4, 4);
        // All values are +INF → no contact
        let result = VoxelSphereQuery::query(&g, [2.0, 2.0, 2.0], 0.3);
        assert!(result.is_none());
    }

    #[test]
    fn test_sphere_query_contact_detected() {
        let mut g = SdfVoxelGrid::new([0.0; 3], 1.0, 4, 4, 4);
        // Set SDF < sphere radius at sphere centre voxel
        g.fill(0.1); // sdf=0.1 everywhere
        let result = VoxelSphereQuery::query(&g, [2.0, 2.0, 2.0], 0.5);
        assert!(
            result.is_some(),
            "sphere with radius > SDF should detect contact"
        );
    }

    #[test]
    fn test_sphere_query_depth_positive() {
        let mut g = SdfVoxelGrid::new([0.0; 3], 1.0, 4, 4, 4);
        g.fill(-0.2); // inside solid
        let c = VoxelSphereQuery::query(&g, [2.0, 2.0, 2.0], 0.5)
            .expect("a sphere centred inside the solid must be in contact");
        assert!(
            (c.depth - 0.7).abs() < 1e-12,
            "depth = radius - sdf = 0.5 + 0.2, got {}",
            c.depth
        );
    }

    #[test]
    fn test_sphere_voxels_in_sphere_non_empty() {
        let g = SdfVoxelGrid::new([0.0; 3], 1.0, 5, 5, 5);
        let voxels = VoxelSphereQuery::voxels_in_sphere(&g, [2.5, 2.5, 2.5], 1.0);
        assert!(!voxels.is_empty(), "should find voxels within radius 1");
    }

    // ---- VoxelRayQuery ----

    #[test]
    fn test_ray_query_no_hit_empty_grid() {
        let g = SdfVoxelGrid::new([0.0; 3], 1.0, 4, 4, 4);
        // All +INF: ray should not hit anything
        let hit = VoxelRayQuery::cast(&g, [-1.0, 2.0, 2.0], [1.0, 0.0, 0.0], 20.0, 0.0);
        assert!(hit.is_none());
    }

    #[test]
    fn test_ray_query_hit_solid_grid() {
        let mut g = SdfVoxelGrid::new([0.0; 3], 1.0, 4, 4, 4);
        g.fill(-1.0); // everything solid
        // A ray that starts inside the solid hits at its origin ...
        let hit = VoxelRayQuery::cast(&g, [1.5, 1.5, 1.5], [1.0, 0.0, 0.0], 20.0, 0.0)
            .expect("ray starting in the solid grid should hit");
        assert!(hit.t.abs() < 1e-12, "t = {}", hit.t);
        // ... and one from outside hits at the grid face.
        let hit = VoxelRayQuery::cast(&g, [-1.0, 2.0, 2.0], [1.0, 0.0, 0.0], 20.0, 0.0)
            .expect("ray from outside should hit the grid face");
        assert!((hit.t - 1.0).abs() < 1e-9, "t = {}", hit.t);
    }

    #[test]
    fn test_ray_intersects_solid() {
        let mut g = SdfVoxelGrid::new([0.0; 3], 1.0, 4, 4, 4);
        g.fill(-1.0);
        assert!(VoxelRayQuery::intersects(
            &g,
            [1.5, 2.0, 2.0],
            [1.0, 0.0, 0.0],
            10.0,
            0.0
        ));
    }

    // ---- VoxelCollisionResponse ----

    #[test]
    fn test_collision_response_zero_depth_no_impulse() {
        let contact = VoxelSphereContact {
            point: [0.0; 3],
            normal: [0.0, 1.0, 0.0],
            depth: 0.0,
        };
        let v = [0.0, -1.0, 0.0];
        let impulse = VoxelCollisionResponse::penalty_impulse(&contact, v, 1.0, 1000.0, 0.5, 0.01);
        // depth=0 → only restitution term (vn < 0)
        assert!(
            impulse.magnitude >= 0.0,
            "impulse magnitude should be non-negative"
        );
    }

    #[test]
    fn test_collision_response_apply_impulse() {
        let impulse = VoxelCollisionImpulse {
            magnitude: 10.0,
            normal: [0.0, 1.0, 0.0],
            contact_point: [0.0; 3],
        };
        let v = [0.0, -5.0, 0.0];
        let v_new = VoxelCollisionResponse::apply_impulse(v, &impulse, 1.0);
        // dv = 10/1.0 in +y direction: vy = -5 + 10 = 5
        assert!(
            (v_new[1] - 5.0).abs() < 1e-12,
            "vy should be -5+10=5, got {}",
            v_new[1]
        );
    }

    // ---- DynamicVoxelUpdate ----

    #[test]
    fn test_stamp_sphere_sets_sdf() {
        let mut g = SdfVoxelGrid::new([0.0; 3], 1.0, 6, 6, 6);
        g.fill(f64::INFINITY);
        DynamicVoxelUpdate::stamp_sphere(&mut g, [3.0, 3.0, 3.0], 1.0);
        let inside = g.count_inside(0.0);
        assert!(inside > 0, "stamping sphere should create inside voxels");
    }

    #[test]
    fn test_erase_sphere_restores_infinity() {
        let mut g = SdfVoxelGrid::new([0.0; 3], 1.0, 6, 6, 6);
        g.fill(f64::INFINITY);
        DynamicVoxelUpdate::stamp_sphere(&mut g, [3.0, 3.0, 3.0], 1.0);
        DynamicVoxelUpdate::erase_sphere(&mut g, [3.0, 3.0, 3.0], 1.0);
        let inside = g.count_inside(0.0);
        assert_eq!(inside, 0, "after erase, no inside voxels should remain");
    }

    #[test]
    fn test_move_sphere() {
        let mut g = SdfVoxelGrid::new([0.0; 3], 1.0, 8, 8, 8);
        g.fill(f64::INFINITY);
        DynamicVoxelUpdate::move_sphere(&mut g, [2.0, 2.0, 2.0], [6.0, 6.0, 6.0], 0.8);
        // Old location should be erased, new location stamped
        // Check a voxel near old centre is back to +INF
        let old_sdf = g.interpolate([2.0, 2.0, 2.0]);
        let new_sdf = g.interpolate([6.0, 6.0, 6.0]);
        assert!(
            old_sdf > new_sdf,
            "new centre should have lower SDF than old centre"
        );
    }

    // ---- Internal math helpers ----

    #[test]
    fn test_sub3() {
        let r = sub3([3.0, 2.0, 1.0], [1.0, 1.0, 1.0]);
        assert_eq!(r, [2.0, 1.0, 0.0]);
    }

    #[test]
    fn test_add3() {
        let r = add3([1.0, 2.0, 3.0], [4.0, 5.0, 6.0]);
        assert_eq!(r, [5.0, 7.0, 9.0]);
    }

    #[test]
    fn test_cross3_unit_vectors() {
        let r = cross3([1.0, 0.0, 0.0], [0.0, 1.0, 0.0]);
        assert!((r[2] - 1.0).abs() < 1e-12, "i×j should be k, got {:?}", r);
    }

    #[test]
    fn test_dist_sq3_known() {
        let d = dist_sq3([0.0; 3], [3.0, 4.0, 0.0]);
        assert!((d - 25.0).abs() < 1e-12);
    }

    #[test]
    fn test_scale3() {
        let r = scale3([1.0, 2.0, 3.0], 2.0);
        assert_eq!(r, [2.0, 4.0, 6.0]);
    }
}
