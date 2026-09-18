// Copyright 2026 COOLJAPAN OU (Team KitaSan)
// SPDX-License-Identifier: Apache-2.0

//! Ray casting against an [`SdfVoxelGrid`].
//!
//! The ray is clipped to the grid box and marched through it by *sphere
//! tracing* the trilinearly interpolated SDF: while the field is positive its
//! value is a lower bound of the distance to the surface, so the ray can safely
//! advance by that amount.  Where the field is unknown (an infinite, i.e. empty,
//! voxel) the ray instead walks from voxel to voxel, so thin features stored in
//! an otherwise empty sparse grid are not skipped.

use super::SdfVoxelGrid;

/// Number of bisection steps used to refine a hit between two samples.
const REFINE_ITERATIONS: usize = 48;

/// Result of a ray-voxel query.
#[derive(Debug, Clone, Copy)]
pub struct VoxelRayHit {
    /// Coordinates of the voxel containing the hit position.
    pub coord: [usize; 3],
    /// Distance `t` along the (normalised) ray at the hit point.
    pub t: f64,
    /// World-space hit position.
    pub position: [f64; 3],
    /// SDF value stored at the centre of the hit voxel.
    pub sdf: f64,
}

/// Ray vs voxel grid intersection queries.
pub struct VoxelRayQuery;

impl VoxelRayQuery {
    /// Cast a ray `origin + t * direction` against `grid` by sphere tracing the
    /// interpolated SDF.
    ///
    /// `ray_dir` need not be normalised; `t` and `max_t` are distances in world
    /// units along the normalised direction.  The ray may start outside the
    /// grid: it is clipped to the grid box first, and a ray that starts inside
    /// the solid reports a hit at `t = 0`.
    ///
    /// Returns the first point along the ray where the trilinearly interpolated
    /// SDF (clamped to the border values, see
    /// [`SdfVoxelGrid::interpolate_clamped`]) falls below `surface_threshold`.
    /// When the hit is found between two samples, `t` and `position` are refined
    /// by bisection onto the threshold crossing.  In regions where the field is
    /// not defined (infinite voxels) the ray walks voxel by voxel and hits the
    /// first voxel whose stored value is below the threshold.
    ///
    /// Returns `None` if the ray misses the grid, no hit occurs within `max_t`,
    /// or an input is not finite (`max_t` may be `f64::INFINITY`).
    pub fn cast(
        grid: &SdfVoxelGrid,
        ray_origin: [f64; 3],
        ray_dir: [f64; 3],
        max_t: f64,
        surface_threshold: f64,
    ) -> Option<VoxelRayHit> {
        let cell = grid.cell_size;
        if grid.is_empty() || grid.data.len() != grid.len() || !cell.is_finite() || cell <= 0.0 {
            return None;
        }
        if max_t.is_nan() || ray_origin.iter().any(|x| !x.is_finite()) {
            return None;
        }
        let len =
            (ray_dir[0] * ray_dir[0] + ray_dir[1] * ray_dir[1] + ray_dir[2] * ray_dir[2]).sqrt();
        if !len.is_finite() || len < 1e-14 {
            return None;
        }
        let d = [ray_dir[0] / len, ray_dir[1] / len, ray_dir[2] / len];

        let (t_enter, t_exit) = Self::ray_aabb_interval(ray_origin, d, grid)?;
        let t_start = t_enter.max(0.0);
        let t_end = t_exit.min(max_t);
        if t_start > t_end {
            return None;
        }

        let point_at = |t: f64| {
            [
                ray_origin[0] + t * d[0],
                ray_origin[1] + t * d[1],
                ray_origin[2] + t * d[2],
            ]
        };
        let hit_at = |t: f64| -> Option<VoxelRayHit> {
            let position = point_at(t);
            let coord = grid.clamped_voxel(position)?;
            let sdf = grid.get(coord[0], coord[1], coord[2])?;
            Some(VoxelRayHit {
                coord,
                t,
                position,
                sdf,
            })
        };

        let step_min = 0.5 * cell;
        // Each iteration either advances by at least `step_min` (sphere tracing)
        // or crosses a voxel boundary, so this bound is never reached for a ray
        // that stays inside the grid box; it only guards against bad input.
        let max_iterations = 8 * (grid.nx + grid.ny + grid.nz) + 64;
        let mut t = t_start;
        // Parameter of the last finite sample that was not a hit; the field is
        // known to be at or above the threshold there, which brackets a refinement.
        let mut previous: Option<f64> = None;
        for _ in 0..max_iterations {
            let p = point_at(t);
            let sdf = grid.interpolate_clamped(p);
            if sdf < surface_threshold {
                let t_hit = match previous {
                    Some(t_prev) => Self::refine(grid, &point_at, t_prev, t, surface_threshold),
                    None => t,
                };
                return hit_at(t_hit);
            }
            if t >= t_end {
                break;
            }
            let advance = if sdf.is_finite() {
                previous = Some(t);
                // Distance to the threshold level set, at least `step_min` so the
                // trace cannot stall in front of the surface.
                (sdf - surface_threshold).max(step_min)
            } else {
                previous = None;
                // Empty region: no distance bound is known.  A stored voxel that
                // is solid is a hit on its own; otherwise walk to the next voxel.
                let coord = grid.clamped_voxel(p)?;
                if grid.get(coord[0], coord[1], coord[2])? < surface_threshold {
                    return hit_at(t);
                }
                Self::distance_to_voxel_exit(grid, p, d, coord) + 1e-6 * cell
            };
            t = (t + advance).min(t_end);
        }
        None
    }

    /// Bisect `[t_outside, t_inside]` onto the crossing of `threshold`.
    ///
    /// `t_outside` must sample at or above the threshold and `t_inside` below
    /// it.  Returns the end of the bracket that is below the threshold, so the
    /// reported position is on the solid side.
    fn refine(
        grid: &SdfVoxelGrid,
        point_at: &impl Fn(f64) -> [f64; 3],
        t_outside: f64,
        t_inside: f64,
        threshold: f64,
    ) -> f64 {
        let mut lo = t_outside;
        let mut hi = t_inside;
        for _ in 0..REFINE_ITERATIONS {
            let mid = 0.5 * (lo + hi);
            if grid.interpolate_clamped(point_at(mid)) < threshold {
                hi = mid;
            } else {
                lo = mid;
            }
        }
        hi
    }

    /// Distance along the unit direction `d` from `p` to the face through which
    /// the ray leaves voxel `coord` (zero if the ray is parallel to every axis).
    fn distance_to_voxel_exit(
        grid: &SdfVoxelGrid,
        p: [f64; 3],
        d: [f64; 3],
        coord: [usize; 3],
    ) -> f64 {
        let mut best = f64::INFINITY;
        for axis in 0..3 {
            if d[axis].abs() < 1e-14 {
                continue;
            }
            let face = if d[axis] > 0.0 {
                coord[axis] + 1
            } else {
                coord[axis]
            };
            let boundary = grid.origin[axis] + face as f64 * grid.cell_size;
            let t = (boundary - p[axis]) / d[axis];
            best = best.min(t.max(0.0));
        }
        if best.is_finite() { best } else { 0.0 }
    }

    /// Clip the ray `origin + t * dir` (unit `dir`) to the grid box.
    ///
    /// Returns the parameters `(t_enter, t_exit)` of the slab-method
    /// intersection, `t_enter` possibly negative when the origin is inside the
    /// box, or `None` if the ray misses the box.
    fn ray_aabb_interval(
        origin: [f64; 3],
        dir: [f64; 3],
        grid: &SdfVoxelGrid,
    ) -> Option<(f64, f64)> {
        let (aabb_min, aabb_max) = grid.aabb();
        let mut t_enter = f64::NEG_INFINITY;
        let mut t_exit = f64::INFINITY;
        for axis in 0..3 {
            if dir[axis].abs() < 1e-14 {
                if origin[axis] < aabb_min[axis] || origin[axis] > aabb_max[axis] {
                    return None;
                }
            } else {
                let t0 = (aabb_min[axis] - origin[axis]) / dir[axis];
                let t1 = (aabb_max[axis] - origin[axis]) / dir[axis];
                let (near, far) = if t0 < t1 { (t0, t1) } else { (t1, t0) };
                t_enter = t_enter.max(near);
                t_exit = t_exit.min(far);
                if t_enter > t_exit {
                    return None;
                }
            }
        }
        Some((t_enter, t_exit))
    }

    /// Check whether a ray intersects any voxel with SDF below `threshold`.
    ///
    /// Equivalent to `cast(..).is_some()`.
    pub fn intersects(
        grid: &SdfVoxelGrid,
        ray_origin: [f64; 3],
        ray_dir: [f64; 3],
        max_t: f64,
        threshold: f64,
    ) -> bool {
        Self::cast(grid, ray_origin, ray_dir, max_t, threshold).is_some()
    }
}
