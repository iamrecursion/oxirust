// Copyright 2026 COOLJAPAN OU (Team KitaSan)
// SPDX-License-Identifier: Apache-2.0

//! Axis-aligned box shape.

use crate::shape::{RayHit, Shape};
use oxiphysics_core::Aabb;
use oxiphysics_core::math::{Mat3, Real, Vec3};

/// An axis-aligned box defined by half-extents.
#[derive(Debug, Clone)]
pub struct BoxShape {
    /// Half-extents along each axis.
    pub half_extents: Vec3,
}

impl BoxShape {
    /// Create a new box with the given half-extents.
    pub fn new(half_extents: Vec3) -> Self {
        Self { half_extents }
    }

    /// Volume: (2hx)(2hy)(2hz).
    pub fn volume_explicit(&self) -> Real {
        8.0 * self.half_extents.x * self.half_extents.y * self.half_extents.z
    }

    /// Surface area: 2*(2hx*2hy + 2hy*2hz + 2hx*2hz).
    pub fn surface_area(&self) -> Real {
        let hx = self.half_extents.x;
        let hy = self.half_extents.y;
        let hz = self.half_extents.z;
        let ax = 2.0 * hx;
        let ay = 2.0 * hy;
        let az = 2.0 * hz;
        2.0 * (ax * ay + ay * az + ax * az)
    }

    /// Inertia tensor as \[\[f64;3\\];3] row-major.
    /// Diagonal: m/12*(b²+c², a²+c², a²+b²) where a,b,c are full side lengths.
    pub fn inertia_tensor_array(&self, mass: f64) -> [[f64; 3]; 3] {
        let hx = self.half_extents.x;
        let hy = self.half_extents.y;
        let hz = self.half_extents.z;
        let ax2 = (2.0 * hx).powi(2);
        let ay2 = (2.0 * hy).powi(2);
        let az2 = (2.0 * hz).powi(2);
        let k = mass / 12.0;
        [
            [k * (ay2 + az2), 0.0, 0.0],
            [0.0, k * (ax2 + az2), 0.0],
            [0.0, 0.0, k * (ax2 + ay2)],
        ]
    }

    /// Ray cast returning (t, normal) as plain arrays (slab method).
    pub fn ray_cast_array(
        &self,
        origin: [f64; 3],
        direction: [f64; 3],
        max_toi: f64,
    ) -> Option<(f64, [f64; 3])> {
        let o = Vec3::new(origin[0], origin[1], origin[2]);
        let d = Vec3::new(direction[0], direction[1], direction[2]);
        let hit = self.ray_cast(&o, &d, max_toi)?;
        Some((hit.toi, [hit.normal.x, hit.normal.y, hit.normal.z]))
    }

    /// GJK support function: farthest point in `direction` (componentwise sign).
    pub fn support(&self, direction: [f64; 3]) -> [f64; 3] {
        [
            self.half_extents.x.copysign(direction[0]),
            self.half_extents.y.copysign(direction[1]),
            self.half_extents.z.copysign(direction[2]),
        ]
    }

    /// All 8 vertices of the box.
    pub fn vertex_list(&self) -> [[f64; 3]; 8] {
        let hx = self.half_extents.x;
        let hy = self.half_extents.y;
        let hz = self.half_extents.z;
        [
            [-hx, -hy, -hz],
            [hx, -hy, -hz],
            [hx, hy, -hz],
            [-hx, hy, -hz],
            [-hx, -hy, hz],
            [hx, -hy, hz],
            [hx, hy, hz],
            [-hx, hy, hz],
        ]
    }

    // ── New methods ──

    /// Returns the 6 face normals of the box (axis-aligned, outward).
    pub fn face_normals() -> [[f64; 3]; 6] {
        [
            [1.0, 0.0, 0.0],
            [-1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, -1.0, 0.0],
            [0.0, 0.0, 1.0],
            [0.0, 0.0, -1.0],
        ]
    }

    /// Returns the 12 edges of the box as pairs of vertex indices into `vertex_list()`.
    pub fn edge_list() -> [(usize, usize); 12] {
        [
            // Bottom face (y = -hy)
            (0, 1),
            (1, 5),
            (5, 4),
            (4, 0),
            // Top face (y = +hy)
            (2, 3),
            (3, 7),
            (7, 6),
            (6, 2),
            // Vertical edges
            (0, 3),
            (1, 2),
            (5, 6),
            (4, 7),
        ]
    }

    /// Returns the 6 faces as groups of 4 vertex indices (into `vertex_list()`).
    /// Each face's vertices are in counter-clockwise order from outside.
    pub fn face_vertex_indices() -> [[usize; 4]; 6] {
        [
            [1, 2, 6, 5], // +X face
            [0, 4, 7, 3], // -X face
            [2, 3, 7, 6], // +Y face
            [0, 1, 5, 4], // -Y face
            [4, 5, 6, 7], // +Z face
            [0, 3, 2, 1], // -Z face
        ]
    }

    /// Area of each face: returns \[+X, -X, +Y, -Y, +Z, -Z\].
    pub fn face_areas(&self) -> [f64; 6] {
        let hx = self.half_extents.x;
        let hy = self.half_extents.y;
        let hz = self.half_extents.z;
        let yz = 4.0 * hy * hz; // +X and -X faces
        let xz = 4.0 * hx * hz; // +Y and -Y faces
        let xy = 4.0 * hx * hy; // +Z and -Z faces
        [yz, yz, xz, xz, xy, xy]
    }

    /// Closest point on (or inside) the box to point `p`.
    pub fn closest_point(&self, p: [f64; 3]) -> [f64; 3] {
        [
            p[0].clamp(-self.half_extents.x, self.half_extents.x),
            p[1].clamp(-self.half_extents.y, self.half_extents.y),
            p[2].clamp(-self.half_extents.z, self.half_extents.z),
        ]
    }

    /// Returns true if `p` is inside (or on the surface of) the box.
    pub fn contains_point(&self, p: [f64; 3]) -> bool {
        p[0].abs() <= self.half_extents.x
            && p[1].abs() <= self.half_extents.y
            && p[2].abs() <= self.half_extents.z
    }

    /// Signed distance from a point to the box surface.
    /// Negative if inside, positive if outside.
    pub fn signed_distance(&self, p: [f64; 3]) -> f64 {
        let dx = p[0].abs() - self.half_extents.x;
        let dy = p[1].abs() - self.half_extents.y;
        let dz = p[2].abs() - self.half_extents.z;

        if dx <= 0.0 && dy <= 0.0 && dz <= 0.0 {
            // Inside: return the largest (least negative) component
            dx.max(dy).max(dz)
        } else {
            // Outside: Euclidean distance from the clamped point
            let cx = dx.max(0.0);
            let cy = dy.max(0.0);
            let cz = dz.max(0.0);
            (cx * cx + cy * cy + cz * cz).sqrt()
        }
    }

    /// Clip a line segment (from `a` to `b`) against this box.
    /// Returns `Some((t_enter, t_exit))` where 0 <= t_enter <= t_exit <= 1,
    /// or `None` if the segment doesn't intersect the box.
    pub fn clip_segment(&self, a: [f64; 3], b: [f64; 3]) -> Option<(f64, f64)> {
        let dir = [b[0] - a[0], b[1] - a[1], b[2] - a[2]];
        let half = [
            self.half_extents.x,
            self.half_extents.y,
            self.half_extents.z,
        ];

        let mut tmin = 0.0_f64;
        let mut tmax = 1.0_f64;

        for i in 0..3 {
            if dir[i].abs() < 1e-12 {
                if a[i] < -half[i] || a[i] > half[i] {
                    return None;
                }
            } else {
                let inv_d = 1.0 / dir[i];
                let mut t1 = (-half[i] - a[i]) * inv_d;
                let mut t2 = (half[i] - a[i]) * inv_d;
                if t1 > t2 {
                    std::mem::swap(&mut t1, &mut t2);
                }
                tmin = tmin.max(t1);
                tmax = tmax.min(t2);
                if tmin > tmax {
                    return None;
                }
            }
        }

        Some((tmin.max(0.0), tmax.min(1.0)))
    }

    /// Determine which face a surface point is on.
    /// Returns the face index (0=+X, 1=-X, 2=+Y, 3=-Y, 4=+Z, 5=-Z).
    pub fn classify_face(&self, p: [f64; 3]) -> usize {
        let dx_pos = (p[0] - self.half_extents.x).abs();
        let dx_neg = (p[0] + self.half_extents.x).abs();
        let dy_pos = (p[1] - self.half_extents.y).abs();
        let dy_neg = (p[1] + self.half_extents.y).abs();
        let dz_pos = (p[2] - self.half_extents.z).abs();
        let dz_neg = (p[2] + self.half_extents.z).abs();

        let dists = [dx_pos, dx_neg, dy_pos, dy_neg, dz_pos, dz_neg];
        let mut min_idx = 0;
        let mut min_val = dists[0];
        for (i, &d) in dists.iter().enumerate().skip(1) {
            if d < min_val {
                min_val = d;
                min_idx = i;
            }
        }
        min_idx
    }

    /// Diagonal length of the box: 2 * sqrt(hx² + hy² + hz²).
    pub fn diagonal_length(&self) -> f64 {
        let hx = self.half_extents.x;
        let hy = self.half_extents.y;
        let hz = self.half_extents.z;
        2.0 * (hx * hx + hy * hy + hz * hz).sqrt()
    }

    /// Edge lengths: \[2*hx, 2*hy, 2*hz\].
    pub fn edge_lengths(&self) -> [f64; 3] {
        [
            2.0 * self.half_extents.x,
            2.0 * self.half_extents.y,
            2.0 * self.half_extents.z,
        ]
    }

    /// Project the box onto an axis and return `(min, max)` interval.
    pub fn project_on_axis(&self, axis: [f64; 3]) -> (f64, f64) {
        // For an AABB, the projection extent is the sum of |axis_i * half_extent_i|
        let extent = self.half_extents.x * axis[0].abs()
            + self.half_extents.y * axis[1].abs()
            + self.half_extents.z * axis[2].abs();
        (-extent, extent)
    }
}

impl Shape for BoxShape {
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    fn bounding_box(&self) -> Aabb {
        Aabb::new(-self.half_extents, self.half_extents)
    }

    fn support_point(&self, direction: &Vec3) -> Vec3 {
        Vec3::new(
            self.half_extents.x.copysign(direction.x),
            self.half_extents.y.copysign(direction.y),
            self.half_extents.z.copysign(direction.z),
        )
    }

    fn volume(&self) -> Real {
        8.0 * self.half_extents.x * self.half_extents.y * self.half_extents.z
    }

    fn center_of_mass(&self) -> Vec3 {
        Vec3::zeros()
    }

    fn inertia_tensor(&self, mass: Real) -> Mat3 {
        let hx = self.half_extents.x;
        let hy = self.half_extents.y;
        let hz = self.half_extents.z;
        // Full extents squared
        let ex2 = (2.0 * hx).powi(2);
        let ey2 = (2.0 * hy).powi(2);
        let ez2 = (2.0 * hz).powi(2);
        let k = mass / 12.0;
        Mat3::new(
            k * (ey2 + ez2),
            0.0,
            0.0,
            0.0,
            k * (ex2 + ez2),
            0.0,
            0.0,
            0.0,
            k * (ex2 + ey2),
        )
    }

    fn ray_cast(&self, ray_origin: &Vec3, ray_direction: &Vec3, max_toi: Real) -> Option<RayHit> {
        // Slab method for AABB ray intersection
        let mut tmin = Real::NEG_INFINITY;
        let mut tmax = Real::INFINITY;
        let mut normal = Vec3::zeros();

        for i in 0..3 {
            let o = ray_origin[i];
            let d = ray_direction[i];
            let half = self.half_extents[i];

            if d.abs() < 1e-12 {
                if o < -half || o > half {
                    return None;
                }
            } else {
                let inv_d = 1.0 / d;
                let mut t1 = (-half - o) * inv_d;
                let mut t2 = (half - o) * inv_d;
                let mut n = Vec3::zeros();
                n[i] = -1.0;

                if t1 > t2 {
                    std::mem::swap(&mut t1, &mut t2);
                    n[i] = 1.0;
                }

                if t1 > tmin {
                    tmin = t1;
                    normal = n;
                }
                if t2 < tmax {
                    tmax = t2;
                }

                if tmin > tmax {
                    return None;
                }
            }
        }

        let t = if tmin >= 0.0 { tmin } else { tmax };
        if t < 0.0 || t > max_toi {
            return None;
        }

        let point = ray_origin + ray_direction * t;
        Some(RayHit {
            point,
            normal,
            toi: t,
        })
    }
}

// ── OBB (Oriented Bounding Box) free functions ──
// These work with rotation matrices (row-major [[f64;3];3]) and plain [f64;3] arrays,
// so they can be used without nalgebra.

/// Dot product of two \[f64;3\] vectors.
fn dot3b(a: [f64; 3], b: [f64; 3]) -> f64 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

/// Matrix-vector multiply: rot (row-major) * v.
fn mat3_mul_vec(rot: [[f64; 3]; 3], v: [f64; 3]) -> [f64; 3] {
    [dot3b(rot[0], v), dot3b(rot[1], v), dot3b(rot[2], v)]
}

/// Transpose of a 3×3 matrix.
fn mat3_transpose(rot: [[f64; 3]; 3]) -> [[f64; 3]; 3] {
    [
        [rot[0][0], rot[1][0], rot[2][0]],
        [rot[0][1], rot[1][1], rot[2][1]],
        [rot[0][2], rot[1][2], rot[2][2]],
    ]
}

/// 3×3 matrix multiply A*B.
fn mat3_mul(a: [[f64; 3]; 3], b: [[f64; 3]; 3]) -> [[f64; 3]; 3] {
    let bt = mat3_transpose(b);
    [
        [dot3b(a[0], bt[0]), dot3b(a[0], bt[1]), dot3b(a[0], bt[2])],
        [dot3b(a[1], bt[0]), dot3b(a[1], bt[1]), dot3b(a[1], bt[2])],
        [dot3b(a[2], bt[0]), dot3b(a[2], bt[1]), dot3b(a[2], bt[2])],
    ]
}

/// Inertia tensor of an OBB in its own local frame (diagonal matrix).
/// `half_extents`: half-widths along each local axis.
/// Returns a 3×3 row-major array.
pub fn obb_inertia_tensor(half_extents: [f64; 3], mass: f64) -> [[f64; 3]; 3] {
    let ax2 = (2.0 * half_extents[0]).powi(2);
    let ay2 = (2.0 * half_extents[1]).powi(2);
    let az2 = (2.0 * half_extents[2]).powi(2);
    let k = mass / 12.0;
    [
        [k * (ay2 + az2), 0.0, 0.0],
        [0.0, k * (ax2 + az2), 0.0],
        [0.0, 0.0, k * (ax2 + ay2)],
    ]
}

/// Inertia tensor of an OBB transformed to world frame via rotation matrix `rot`.
///
/// `rot` follows the row-basis convention shared by [`obb_closest_point`],
/// [`obb_vertices`] and the other OBB helpers: row `i` is the world-space
/// direction of local axis `i`, so a local vector `l` maps to the world vector
/// `rotᵀ · l` (and a world vector `w` to the local vector `rot · w`).
///
/// The world-frame tensor is therefore `I_world = rotᵀ · I_local · rot`.  Row
/// `i` of `rot` is a principal axis of the result with moment
/// `I_local[i][i]`, and the products of inertia carry the sign that follows
/// from the box's orientation (the transposed product `rot · I_local · rotᵀ`
/// is only correct for column-basis matrices and flips their sign).
pub fn obb_inertia_tensor_rotated(
    half_extents: [f64; 3],
    mass: f64,
    rot: [[f64; 3]; 3],
) -> [[f64; 3]; 3] {
    let i_local = obb_inertia_tensor(half_extents, mass);
    // I_world = rot^T * i_local * rot
    let ri = mat3_mul(mat3_transpose(rot), i_local);
    mat3_mul(ri, rot)
}

/// Closest point on the surface of an OBB to a world-space point `p`.
/// `center`: OBB center in world space.
/// `half_extents`: half-widths along each local axis.
/// `rot`: rotation matrix (row i = world direction of local axis i).
/// Returns world-space closest point (may be on interior if `p` is inside).
pub fn obb_closest_point(
    p: [f64; 3],
    center: [f64; 3],
    half_extents: [f64; 3],
    rot: [[f64; 3]; 3],
) -> [f64; 3] {
    // Transform p to local space: local = rot * (p - center)
    let d = [p[0] - center[0], p[1] - center[1], p[2] - center[2]];
    let local = mat3_mul_vec(rot, d);

    // Clamp each component to half-extents
    let clamped = [
        local[0].clamp(-half_extents[0], half_extents[0]),
        local[1].clamp(-half_extents[1], half_extents[1]),
        local[2].clamp(-half_extents[2], half_extents[2]),
    ];

    // Transform back to world space: world = center + rot^T * clamped
    let rt = mat3_transpose(rot);
    let world_offset = mat3_mul_vec(rt, clamped);
    [
        center[0] + world_offset[0],
        center[1] + world_offset[1],
        center[2] + world_offset[2],
    ]
}

/// Compute the projection half-extent of an OBB onto a world-space axis.
/// This is the sum of |half_i * dot(rot_col_i, axis)| over i.
pub fn obb_projection_extent(half_extents: [f64; 3], rot: [[f64; 3]; 3], axis: [f64; 3]) -> f64 {
    // rot row i = world direction of local axis i
    // dot(rot row i, axis) = projection of local axis i onto axis
    let mut ext = 0.0_f64;
    for i in 0..3 {
        ext += half_extents[i] * dot3b(rot[i], axis).abs();
    }
    ext
}

/// Ray-OBB intersection using the slab method in local OBB space.
/// `ray_origin`, `ray_dir`: world-space ray.
/// `center`, `half_extents`, `rot`: OBB parameters.
/// Returns the hit parameter `t` if intersection within `[0, max_toi]`, else `None`.
pub fn obb_ray_intersection(
    ray_origin: [f64; 3],
    ray_dir: [f64; 3],
    center: [f64; 3],
    half_extents: [f64; 3],
    rot: [[f64; 3]; 3],
    max_toi: f64,
) -> Option<f64> {
    // Transform ray to OBB local space
    let d_world = [
        ray_origin[0] - center[0],
        ray_origin[1] - center[1],
        ray_origin[2] - center[2],
    ];
    let o_local = mat3_mul_vec(rot, d_world);
    let dir_local = mat3_mul_vec(rot, ray_dir);

    let mut tmin = f64::NEG_INFINITY;
    let mut tmax = f64::INFINITY;

    for i in 0..3 {
        let o = o_local[i];
        let d = dir_local[i];
        let h = half_extents[i];

        if d.abs() < 1e-12 {
            if o < -h || o > h {
                return None;
            }
        } else {
            let inv_d = 1.0 / d;
            let t1 = (-h - o) * inv_d;
            let t2 = (h - o) * inv_d;
            let (t_near, t_far) = if t1 <= t2 { (t1, t2) } else { (t2, t1) };
            tmin = tmin.max(t_near);
            tmax = tmax.min(t_far);
            if tmin > tmax {
                return None;
            }
        }
    }

    let t = if tmin >= 0.0 { tmin } else { tmax };
    if t < 0.0 || t > max_toi {
        None
    } else {
        Some(t)
    }
}

/// Signed distance from the OBB to a plane (ax + by + cz + d = 0).
/// Returns `(min_signed_dist, max_signed_dist)` for the OBB interval projected on the plane normal.
/// If `min < 0 && max > 0`, the OBB straddles the plane.
pub fn obb_plane_distance(
    center: [f64; 3],
    half_extents: [f64; 3],
    rot: [[f64; 3]; 3],
    plane: [f64; 4],
) -> (f64, f64) {
    let normal = [plane[0], plane[1], plane[2]];
    let plane_d = plane[3];

    // Project OBB center onto plane normal
    let center_dist = dot3b(center, normal) + plane_d;

    // Compute projection half-extent
    let extent = obb_projection_extent(half_extents, rot, normal);

    (center_dist - extent, center_dist + extent)
}

/// Returns all 8 vertices of the OBB in world space.
pub fn obb_vertices(center: [f64; 3], half_extents: [f64; 3], rot: [[f64; 3]; 3]) -> Vec<[f64; 3]> {
    let rt = mat3_transpose(rot);
    let signs = [
        [-1.0_f64, -1.0, -1.0],
        [1.0, -1.0, -1.0],
        [1.0, 1.0, -1.0],
        [-1.0, 1.0, -1.0],
        [-1.0, -1.0, 1.0],
        [1.0, -1.0, 1.0],
        [1.0, 1.0, 1.0],
        [-1.0, 1.0, 1.0],
    ];
    signs
        .iter()
        .map(|s| {
            let local = [
                s[0] * half_extents[0],
                s[1] * half_extents[1],
                s[2] * half_extents[2],
            ];
            let world = mat3_mul_vec(rt, local);
            [
                center[0] + world[0],
                center[1] + world[1],
                center[2] + world[2],
            ]
        })
        .collect()
}

/// Returns all 12 edges of the OBB in world space as (start, end) pairs.
pub fn obb_edges(
    center: [f64; 3],
    half_extents: [f64; 3],
    rot: [[f64; 3]; 3],
) -> Vec<([f64; 3], [f64; 3])> {
    let verts = obb_vertices(center, half_extents, rot);
    // Edge pairs using same winding as BoxShape::edge_list()
    let pairs: [(usize, usize); 12] = [
        (0, 1),
        (1, 5),
        (5, 4),
        (4, 0),
        (2, 3),
        (3, 7),
        (7, 6),
        (6, 2),
        (0, 3),
        (1, 2),
        (5, 6),
        (4, 7),
    ];
    pairs.iter().map(|&(a, b)| (verts[a], verts[b])).collect()
}

/// Returns the 6 face centers of the OBB in world space.
/// Order: +local_x, -local_x, +local_y, -local_y, +local_z, -local_z.
pub fn obb_face_centers(
    center: [f64; 3],
    half_extents: [f64; 3],
    rot: [[f64; 3]; 3],
) -> Vec<[f64; 3]> {
    let rt = mat3_transpose(rot);
    let mut fc = Vec::with_capacity(6);
    for axis in 0..3 {
        for sign in [1.0_f64, -1.0_f64] {
            let mut local = [0.0_f64; 3];
            local[axis] = sign * half_extents[axis];
            let world = mat3_mul_vec(rt, local);
            fc.push([
                center[0] + world[0],
                center[1] + world[1],
                center[2] + world[2],
            ]);
        }
    }
    fc
}

/// GJK support function for an OBB: returns the farthest vertex in `direction`.
pub fn obb_support_fn(
    center: [f64; 3],
    half_extents: [f64; 3],
    rot: [[f64; 3]; 3],
    direction: [f64; 3],
) -> [f64; 3] {
    // Project direction onto local axes to get signs
    let local_dir = mat3_mul_vec(rot, direction);
    let local_support = [
        half_extents[0].copysign(local_dir[0]),
        half_extents[1].copysign(local_dir[1]),
        half_extents[2].copysign(local_dir[2]),
    ];
    let rt = mat3_transpose(rot);
    let world_offset = mat3_mul_vec(rt, local_support);
    [
        center[0] + world_offset[0],
        center[1] + world_offset[1],
        center[2] + world_offset[2],
    ]
}

/// Compute the AABB bounds (lo, hi) of an OBB in world space.
pub fn obb_aabb_bounds(
    center: [f64; 3],
    half_extents: [f64; 3],
    rot: [[f64; 3]; 3],
) -> ([f64; 3], [f64; 3]) {
    let mut lo = center;
    let mut hi = center;
    let rt = mat3_transpose(rot);
    for axis in 0..3 {
        // Each local axis contributes: |rot_col_axis| * half_extents[axis]
        for k in 0..3 {
            let contribution = rt[k][axis].abs() * half_extents[axis];
            lo[k] -= contribution;
            hi[k] += contribution;
        }
    }
    (lo, hi)
}

#[cfg(test)]
mod proptest_tests {
    use super::*;
    use crate::BoxShape;
    use crate::box_shape::Vec3;

    use proptest::prelude::*;

    fn pos_half() -> impl Strategy<Value = f64> {
        0.01_f64..100.0_f64
    }

    proptest! {
        #[test]
        fn prop_box_volume_nonneg(
            hx in pos_half(),
            hy in pos_half(),
            hz in pos_half(),
        ) {
            let b = BoxShape::new(Vec3::new(hx, hy, hz));
            prop_assert!(b.volume() >= 0.0, "volume negative: {}", b.volume());
        }

        #[test]
        fn prop_box_inertia_positive_definite(
            hx in pos_half(),
            hy in pos_half(),
            hz in pos_half(),
            m in 0.01_f64..1000.0_f64,
        ) {
            let b = BoxShape::new(Vec3::new(hx, hy, hz));
            let it = b.inertia_tensor(m);
            prop_assert!(it[(0, 0)] > 0.0, "Ixx not positive: {}", it[(0, 0)]);
            prop_assert!(it[(1, 1)] > 0.0, "Iyy not positive: {}", it[(1, 1)]);
            prop_assert!(it[(2, 2)] > 0.0, "Izz not positive: {}", it[(2, 2)]);
        }

        #[test]
        fn prop_box_volume_scales_cubically(
            hx in pos_half(),
            hy in pos_half(),
            hz in pos_half(),
        ) {
            let b1 = BoxShape::new(Vec3::new(hx, hy, hz));
            let b2 = BoxShape::new(Vec3::new(2.0 * hx, 2.0 * hy, 2.0 * hz));
            let ratio = b2.volume() / b1.volume();
            prop_assert!(
                (ratio - 8.0).abs() < 1e-6,
                "box volume ratio for 2x scale should be 8, got {}",
                ratio
            );
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::BoxShape;
    use crate::box_shape::Vec3;
    use crate::box_shape::obb_aabb_bounds;
    use crate::box_shape::obb_closest_point;
    use crate::box_shape::obb_edges;
    use crate::box_shape::obb_face_centers;
    use crate::box_shape::obb_inertia_tensor;
    use crate::box_shape::obb_inertia_tensor_rotated;
    use crate::box_shape::obb_plane_distance;
    use crate::box_shape::obb_projection_extent;
    use crate::box_shape::obb_ray_intersection;
    use crate::box_shape::obb_support_fn;
    use crate::box_shape::obb_vertices;

    #[test]
    fn test_box_support() {
        let b = BoxShape::new(Vec3::new(1.0, 2.0, 3.0));
        let sp = b.support_point(&Vec3::new(1.0, 1.0, 1.0));
        assert_eq!(sp, Vec3::new(1.0, 2.0, 3.0));
    }

    #[test]
    fn test_box_support_array() {
        let b = BoxShape::new(Vec3::new(1.0, 2.0, 3.0));
        let sp = b.support([1.0, 1.0, 1.0]);
        assert!((sp[0] - 1.0).abs() < 1e-10);
        assert!((sp[1] - 2.0).abs() < 1e-10);
        assert!((sp[2] - 3.0).abs() < 1e-10);
    }

    #[test]
    fn test_box_volume() {
        let b = BoxShape::new(Vec3::new(1.0, 2.0, 3.0));
        assert!((b.volume() - 48.0).abs() < 1e-10);
    }

    #[test]
    fn test_box_surface_area() {
        // Half-extents (1,1,1): full box 2x2x2, SA = 6*4 = 24
        let b = BoxShape::new(Vec3::new(1.0, 1.0, 1.0));
        assert!((b.surface_area() - 24.0).abs() < 1e-10);
    }

    #[test]
    fn test_box_inertia() {
        // Unit cube (half_extents = 0.5), mass = 1
        let b = BoxShape::new(Vec3::new(0.5, 0.5, 0.5));
        let it = b.inertia_tensor(1.0);
        // I = m/12 * (h^2 + d^2) = 1/12 * (1+1) = 1/6
        assert!((it[(0, 0)] - 1.0 / 6.0).abs() < 1e-10);
    }

    #[test]
    fn test_box_inertia_array() {
        let b = BoxShape::new(Vec3::new(0.5, 0.5, 0.5));
        let it = b.inertia_tensor_array(1.0);
        assert!((it[0][0] - 1.0 / 6.0).abs() < 1e-10);
        assert!(it[0][1].abs() < 1e-10);
    }

    #[test]
    fn test_box_raycast() {
        let b = BoxShape::new(Vec3::new(1.0, 1.0, 1.0));
        let origin = Vec3::new(-5.0, 0.0, 0.0);
        let dir = Vec3::new(1.0, 0.0, 0.0);
        let hit = b.ray_cast(&origin, &dir, 100.0).unwrap();
        assert!((hit.toi - 4.0).abs() < 1e-10);
        assert!((hit.point.x + 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_box_raycast_array() {
        let b = BoxShape::new(Vec3::new(1.0, 1.0, 1.0));
        let (t, n) = b
            .ray_cast_array([-5.0, 0.0, 0.0], [1.0, 0.0, 0.0], 100.0)
            .unwrap();
        assert!((t - 4.0).abs() < 1e-10);
        assert!((n[0] + 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_box_vertex_list() {
        let b = BoxShape::new(Vec3::new(1.0, 2.0, 3.0));
        let verts = b.vertex_list();
        assert_eq!(verts.len(), 8);
        // All vertices should have |x|=1, |y|=2, |z|=3
        for v in &verts {
            assert!((v[0].abs() - 1.0).abs() < 1e-10);
            assert!((v[1].abs() - 2.0).abs() < 1e-10);
            assert!((v[2].abs() - 3.0).abs() < 1e-10);
        }
    }

    #[test]
    fn test_box_raycast_miss() {
        let b = BoxShape::new(Vec3::new(1.0, 1.0, 1.0));
        // Ray parallel to face and outside
        let origin = Vec3::new(5.0, 5.0, 0.0);
        let dir = Vec3::new(1.0, 0.0, 0.0);
        assert!(b.ray_cast(&origin, &dir, 100.0).is_none());
    }

    // ── New tests ──

    #[test]
    fn test_box_face_normals() {
        let normals = BoxShape::face_normals();
        assert_eq!(normals.len(), 6);
        // Each normal should have unit length
        for n in &normals {
            let len = (n[0] * n[0] + n[1] * n[1] + n[2] * n[2]).sqrt();
            assert!((len - 1.0).abs() < 1e-10);
        }
    }

    #[test]
    fn test_box_edge_list() {
        let edges = BoxShape::edge_list();
        assert_eq!(edges.len(), 12);
        // All indices should be < 8
        for (a, b) in &edges {
            assert!(*a < 8);
            assert!(*b < 8);
        }
    }

    #[test]
    fn test_box_face_vertex_indices() {
        let faces = BoxShape::face_vertex_indices();
        assert_eq!(faces.len(), 6);
        for face in &faces {
            for &idx in face {
                assert!(idx < 8);
            }
        }
    }

    #[test]
    fn test_box_face_areas() {
        let b = BoxShape::new(Vec3::new(1.0, 2.0, 3.0));
        let areas = b.face_areas();
        // +X/-X faces: 4*2*3 = 24
        assert!((areas[0] - 24.0).abs() < 1e-10);
        assert!((areas[1] - 24.0).abs() < 1e-10);
        // +Y/-Y faces: 4*1*3 = 12
        assert!((areas[2] - 12.0).abs() < 1e-10);
        assert!((areas[3] - 12.0).abs() < 1e-10);
        // +Z/-Z faces: 4*1*2 = 8
        assert!((areas[4] - 8.0).abs() < 1e-10);
        assert!((areas[5] - 8.0).abs() < 1e-10);
    }

    #[test]
    fn test_box_face_areas_sum_equals_surface_area() {
        let b = BoxShape::new(Vec3::new(1.5, 2.5, 3.5));
        let areas = b.face_areas();
        let sum: f64 = areas.iter().sum();
        assert!((sum - b.surface_area()).abs() < 1e-10);
    }

    #[test]
    fn test_box_closest_point_inside() {
        let b = BoxShape::new(Vec3::new(1.0, 1.0, 1.0));
        let cp = b.closest_point([0.5, 0.5, 0.5]);
        assert_eq!(cp, [0.5, 0.5, 0.5]); // inside, returns same point
    }

    #[test]
    fn test_box_closest_point_outside() {
        let b = BoxShape::new(Vec3::new(1.0, 1.0, 1.0));
        let cp = b.closest_point([3.0, 0.0, 0.0]);
        assert!((cp[0] - 1.0).abs() < 1e-10);
        assert!(cp[1].abs() < 1e-10);
        assert!(cp[2].abs() < 1e-10);
    }

    #[test]
    fn test_box_closest_point_corner() {
        let b = BoxShape::new(Vec3::new(1.0, 1.0, 1.0));
        let cp = b.closest_point([3.0, 3.0, 3.0]);
        assert!((cp[0] - 1.0).abs() < 1e-10);
        assert!((cp[1] - 1.0).abs() < 1e-10);
        assert!((cp[2] - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_box_contains_point() {
        let b = BoxShape::new(Vec3::new(1.0, 2.0, 3.0));
        assert!(b.contains_point([0.0, 0.0, 0.0]));
        assert!(b.contains_point([1.0, 2.0, 3.0])); // on surface
        assert!(!b.contains_point([1.1, 0.0, 0.0]));
    }

    #[test]
    fn test_box_signed_distance_inside() {
        let b = BoxShape::new(Vec3::new(2.0, 2.0, 2.0));
        let d = b.signed_distance([0.0, 0.0, 0.0]);
        assert!((d + 2.0).abs() < 1e-10); // center, min dist to face = 2
    }

    #[test]
    fn test_box_signed_distance_outside() {
        let b = BoxShape::new(Vec3::new(1.0, 1.0, 1.0));
        let d = b.signed_distance([3.0, 0.0, 0.0]);
        assert!((d - 2.0).abs() < 1e-10);
    }

    #[test]
    fn test_box_signed_distance_corner() {
        let b = BoxShape::new(Vec3::new(1.0, 1.0, 1.0));
        let d = b.signed_distance([2.0, 2.0, 2.0]);
        // Distance to corner (1,1,1) = sqrt(3)
        assert!((d - 3.0_f64.sqrt()).abs() < 1e-10);
    }

    #[test]
    fn test_box_clip_segment_through() {
        let b = BoxShape::new(Vec3::new(1.0, 1.0, 1.0));
        let result = b.clip_segment([-3.0, 0.0, 0.0], [3.0, 0.0, 0.0]);
        assert!(result.is_some());
        let (t_enter, t_exit) = result.unwrap();
        // Segment goes from x=-3 to x=3, box is [-1,1]
        // t_enter = (1+(-3))/(3-(-3)) = (-3+1)/(6)... let's compute:
        // x = -3 + t*6, hit at x=-1: t=(2)/6=1/3, x=1: t=4/6=2/3
        assert!((t_enter - 1.0 / 3.0).abs() < 1e-10);
        assert!((t_exit - 2.0 / 3.0).abs() < 1e-10);
    }

    #[test]
    fn test_box_clip_segment_miss() {
        let b = BoxShape::new(Vec3::new(1.0, 1.0, 1.0));
        let result = b.clip_segment([5.0, 5.0, 0.0], [6.0, 5.0, 0.0]);
        assert!(result.is_none());
    }

    #[test]
    fn test_box_clip_segment_inside() {
        let b = BoxShape::new(Vec3::new(2.0, 2.0, 2.0));
        let result = b.clip_segment([0.0, 0.0, 0.0], [1.0, 0.0, 0.0]);
        assert!(result.is_some());
        let (t_enter, t_exit) = result.unwrap();
        assert!(t_enter.abs() < 1e-10);
        assert!((t_exit - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_box_classify_face() {
        let b = BoxShape::new(Vec3::new(1.0, 2.0, 3.0));
        assert_eq!(b.classify_face([1.0, 0.0, 0.0]), 0); // +X
        assert_eq!(b.classify_face([-1.0, 0.0, 0.0]), 1); // -X
        assert_eq!(b.classify_face([0.0, 2.0, 0.0]), 2); // +Y
        assert_eq!(b.classify_face([0.0, -2.0, 0.0]), 3); // -Y
        assert_eq!(b.classify_face([0.0, 0.0, 3.0]), 4); // +Z
        assert_eq!(b.classify_face([0.0, 0.0, -3.0]), 5); // -Z
    }

    #[test]
    fn test_box_diagonal_length() {
        let b = BoxShape::new(Vec3::new(1.0, 1.0, 1.0));
        let expected = 2.0 * 3.0_f64.sqrt();
        assert!((b.diagonal_length() - expected).abs() < 1e-10);
    }

    #[test]
    fn test_box_edge_lengths() {
        let b = BoxShape::new(Vec3::new(1.0, 2.0, 3.0));
        let el = b.edge_lengths();
        assert!((el[0] - 2.0).abs() < 1e-10);
        assert!((el[1] - 4.0).abs() < 1e-10);
        assert!((el[2] - 6.0).abs() < 1e-10);
    }

    #[test]
    fn test_box_project_on_axis() {
        let b = BoxShape::new(Vec3::new(1.0, 2.0, 3.0));
        // Project onto X axis
        let (lo, hi) = b.project_on_axis([1.0, 0.0, 0.0]);
        assert!((lo + 1.0).abs() < 1e-10);
        assert!((hi - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_box_project_on_diagonal_axis() {
        let b = BoxShape::new(Vec3::new(1.0, 1.0, 1.0));
        // Project onto (1,1,1)/sqrt(3) axis
        let s = 1.0 / 3.0_f64.sqrt();
        let (lo, hi) = b.project_on_axis([s, s, s]);
        // Extent = hx*|s| + hy*|s| + hz*|s| = 3*s = sqrt(3)
        let expected = 3.0_f64.sqrt();
        assert!((hi - expected).abs() < 1e-10);
        assert!((lo + expected).abs() < 1e-10);
    }

    #[test]
    fn test_box_volume_explicit_matches() {
        let b = BoxShape::new(Vec3::new(1.5, 2.5, 3.5));
        assert!((b.volume_explicit() - b.volume()).abs() < 1e-10);
    }

    // ── OBB free-function tests ──

    #[test]
    fn test_obb_inertia_tensor_unit_cube() {
        // Unit cube: half_extents (0.5,0.5,0.5), mass=1
        // I_xx = m/12*(b²+c²) = 1/12*(1+1) = 1/6
        let it = obb_inertia_tensor([0.5, 0.5, 0.5], 1.0);
        assert!((it[0][0] - 1.0 / 6.0).abs() < 1e-10);
        assert!((it[1][1] - 1.0 / 6.0).abs() < 1e-10);
        assert!((it[2][2] - 1.0 / 6.0).abs() < 1e-10);
        // Off-diagonal must be zero
        assert!(it[0][1].abs() < 1e-10);
        assert!(it[0][2].abs() < 1e-10);
    }

    #[test]
    fn test_obb_inertia_tensor_asymmetric() {
        // half_extents (1,2,3), mass=12
        // a=2,b=4,c=6 → Ixx=12/12*(16+36)=52, Iyy=12/12*(4+36)=40, Izz=12/12*(4+16)=20
        let it = obb_inertia_tensor([1.0, 2.0, 3.0], 12.0);
        assert!((it[0][0] - 52.0).abs() < 1e-10);
        assert!((it[1][1] - 40.0).abs() < 1e-10);
        assert!((it[2][2] - 20.0).abs() < 1e-10);
    }

    #[test]
    fn test_obb_inertia_tensor_rotated_identity() {
        // With identity rotation, should match the axis-aligned formula
        let rot = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];
        let it = obb_inertia_tensor_rotated([1.0, 2.0, 3.0], 12.0, rot);
        let ref_it = obb_inertia_tensor([1.0, 2.0, 3.0], 12.0);
        for i in 0..3 {
            for j in 0..3 {
                assert!(
                    (it[i][j] - ref_it[i][j]).abs() < 1e-10,
                    "mismatch at [{i}][{j}]: {} vs {}",
                    it[i][j],
                    ref_it[i][j]
                );
            }
        }
    }

    /// Standard rotation matrix about `axis` by `angle` (Rodrigues).  Its rows are
    /// an orthonormal basis, so it is a valid row-basis `rot` for the OBB helpers.
    fn rotation_about_axis(axis: [f64; 3], angle: f64) -> [[f64; 3]; 3] {
        let norm = (axis[0] * axis[0] + axis[1] * axis[1] + axis[2] * axis[2]).sqrt();
        let k = [axis[0] / norm, axis[1] / norm, axis[2] / norm];
        let (s, c) = angle.sin_cos();
        let mut r = [[0.0_f64; 3]; 3];
        for (a, row) in r.iter_mut().enumerate() {
            for (b, entry) in row.iter_mut().enumerate() {
                let identity = if a == b { c } else { 0.0 };
                *entry = identity + (1.0 - c) * k[a] * k[b];
            }
        }
        r[0][1] -= s * k[2];
        r[0][2] += s * k[1];
        r[1][0] += s * k[2];
        r[1][2] -= s * k[0];
        r[2][0] -= s * k[1];
        r[2][1] += s * k[0];
        r
    }

    /// Inertia tensor of a rotated OBB from an independent derivation: the second
    /// moments of the uniform solid are integrated with the exact 2-point
    /// Gauss–Legendre product rule (nodes `±h/√3` per axis, weight 1/8 each), with
    /// every local point mapped to the world as `Σ_i l_i · row_i`.
    fn obb_inertia_by_quadrature(
        half_extents: [f64; 3],
        mass: f64,
        rot: [[f64; 3]; 3],
    ) -> [[f64; 3]; 3] {
        let inv_sqrt3 = 1.0 / 3.0_f64.sqrt();
        let mut second_moment = [[0.0_f64; 3]; 3];
        for sx in [-1.0_f64, 1.0] {
            for sy in [-1.0_f64, 1.0] {
                for sz in [-1.0_f64, 1.0] {
                    let local = [
                        sx * half_extents[0] * inv_sqrt3,
                        sy * half_extents[1] * inv_sqrt3,
                        sz * half_extents[2] * inv_sqrt3,
                    ];
                    let mut world = [0.0_f64; 3];
                    for (l, row) in local.iter().zip(rot.iter()) {
                        for (w, r) in world.iter_mut().zip(row.iter()) {
                            *w += l * r;
                        }
                    }
                    for (moment_row, &wa) in second_moment.iter_mut().zip(world.iter()) {
                        for (entry, &wb) in moment_row.iter_mut().zip(world.iter()) {
                            *entry += wa * wb / 8.0;
                        }
                    }
                }
            }
        }
        let trace = second_moment[0][0] + second_moment[1][1] + second_moment[2][2];
        let mut inertia = [[0.0_f64; 3]; 3];
        for (a, (inertia_row, moment_row)) in
            inertia.iter_mut().zip(second_moment.iter()).enumerate()
        {
            for (b, (entry, &moment)) in inertia_row.iter_mut().zip(moment_row.iter()).enumerate() {
                let delta = if a == b { trace } else { 0.0 };
                *entry = mass * (delta - moment);
            }
        }
        inertia
    }

    #[test]
    fn test_obb_inertia_tensor_rotated_30deg_about_z_hand_values() {
        // Row-basis rotation: local x points along world (cos30, sin30, 0), local y
        // along (-sin30, cos30, 0).  Local principal moments are 52, 40, 20 (see
        // `test_obb_inertia_tensor_asymmetric`), so
        //   Ixx = 52 cos² + 40 sin² = 49,  Iyy = 52 sin² + 40 cos² = 43,
        //   Ixy = (52 - 40) cos sin = 12 · cos30 · sin30 = +3√3 (positive).
        let (s, c) = (
            std::f64::consts::FRAC_PI_6.sin(),
            std::f64::consts::FRAC_PI_6.cos(),
        );
        let rot = [[c, s, 0.0], [-s, c, 0.0], [0.0, 0.0, 1.0]];
        let it = obb_inertia_tensor_rotated([1.0, 2.0, 3.0], 12.0, rot);
        let expected_xy = 3.0 * 3.0_f64.sqrt();
        assert!((it[0][0] - 49.0).abs() < 1e-9, "Ixx = {}", it[0][0]);
        assert!((it[1][1] - 43.0).abs() < 1e-9, "Iyy = {}", it[1][1]);
        assert!((it[2][2] - 20.0).abs() < 1e-9, "Izz = {}", it[2][2]);
        assert!(
            it[0][1] > 0.0,
            "product of inertia Ixy must be positive, got {}",
            it[0][1]
        );
        assert!((it[0][1] - expected_xy).abs() < 1e-9, "Ixy = {}", it[0][1]);
        assert!((it[1][0] - expected_xy).abs() < 1e-9, "Iyx = {}", it[1][0]);
        for &(i, j) in &[(0, 2), (2, 0), (1, 2), (2, 1)] {
            assert!(it[i][j].abs() < 1e-9, "I[{i}][{j}] = {}", it[i][j]);
        }
    }

    #[test]
    fn test_obb_inertia_tensor_rotated_rows_are_principal_axes() {
        // Row i of `rot` is the world direction of local axis i, so it must be an
        // eigenvector of the world tensor with eigenvalue I_local[i][i].
        let half = [0.7, 1.9, 1.1];
        let mass = 5.5;
        let local = obb_inertia_tensor(half, mass);
        for &(axis, angle) in &[
            ([0.0, 0.0, 1.0], 0.52),
            ([1.0, 0.0, 0.0], -1.1),
            ([1.0, 2.0, 3.0], 0.7),
            ([-2.0, 0.5, 1.0], 2.4),
        ] {
            let rot = rotation_about_axis(axis, angle);
            let world = obb_inertia_tensor_rotated(half, mass, rot);
            for i in 0..3 {
                let r = rot[i];
                for a in 0..3 {
                    let applied: f64 = (0..3).map(|b| world[a][b] * r[b]).sum();
                    let expected = local[i][i] * r[a];
                    assert!(
                        (applied - expected).abs() < 1e-9,
                        "axis {axis:?} angle {angle}: (I·row{i})[{a}] = {applied}, expected {expected}"
                    );
                }
            }
        }
    }

    #[test]
    fn test_obb_inertia_tensor_rotated_matches_independent_quadrature() {
        let half = [1.0, 2.0, 3.0];
        let mass = 12.0;
        for &(axis, angle) in &[
            ([0.0, 0.0, 1.0], std::f64::consts::FRAC_PI_6),
            ([1.0, 2.0, 3.0], 0.7),
            ([0.3, -1.0, 0.2], 2.0),
        ] {
            let rot = rotation_about_axis(axis, angle);
            let got = obb_inertia_tensor_rotated(half, mass, rot);
            let want = obb_inertia_by_quadrature(half, mass, rot);
            let local_trace: f64 = (0..3).map(|i| obb_inertia_tensor(half, mass)[i][i]).sum();
            let got_trace: f64 = (0..3).map(|i| got[i][i]).sum();
            assert!(
                (got_trace - local_trace).abs() < 1e-9,
                "rotation must preserve the trace: {got_trace} vs {local_trace}"
            );
            for (a, row) in got.iter().enumerate() {
                for (b, &value) in row.iter().enumerate() {
                    assert!(
                        (value - want[a][b]).abs() < 1e-9,
                        "axis {axis:?} angle {angle}: I[{a}][{b}] = {value} vs quadrature {}",
                        want[a][b]
                    );
                    assert!(
                        (value - got[b][a]).abs() < 1e-12,
                        "tensor must be symmetric at [{a}][{b}]"
                    );
                }
            }
        }
    }

    #[test]
    fn test_obb_inertia_tensor_rotated_is_consistent_with_obb_vertices() {
        // The OBB helpers share one convention: the vertex cloud from
        // `obb_vertices` with the same `rot` must have the principal axes that
        // `obb_inertia_tensor_rotated` reports.  The longest local axis (z, half
        // extent 3) has the smallest moment, so the world direction of local z
        // must be the world direction along which the vertices spread the most.
        let half = [1.0, 2.0, 3.0];
        let rot = rotation_about_axis([1.0, 2.0, 3.0], 0.7);
        let it = obb_inertia_tensor_rotated(half, 12.0, rot);
        let verts = obb_vertices([0.0, 0.0, 0.0], half, rot);
        let local_z = rot[2];
        let spread_z = obb_projection_extent(half, rot, local_z);
        let max_proj = verts
            .iter()
            .map(|v| (v[0] * local_z[0] + v[1] * local_z[1] + v[2] * local_z[2]).abs())
            .fold(0.0_f64, f64::max);
        assert!(
            (spread_z - 3.0).abs() < 1e-9,
            "extent along local z = {spread_z}"
        );
        assert!(
            (max_proj - 3.0).abs() < 1e-9,
            "vertex spread along local z = {max_proj}"
        );
        let moment_along_z: f64 = (0..3)
            .map(|a| {
                (0..3)
                    .map(|b| local_z[a] * it[a][b] * local_z[b])
                    .sum::<f64>()
            })
            .sum();
        assert!(
            (moment_along_z - 20.0).abs() < 1e-9,
            "moment about the world direction of local z = {moment_along_z}, expected 20"
        );
    }

    #[test]
    fn test_obb_closest_point_inside() {
        let center = [0.0, 0.0, 0.0];
        let half = [1.0, 1.0, 1.0];
        let rot = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];
        let cp = obb_closest_point([0.5, 0.3, -0.2], center, half, rot);
        // Point is inside; closest = the point itself
        assert!((cp[0] - 0.5).abs() < 1e-10);
        assert!((cp[1] - 0.3).abs() < 1e-10);
        assert!((cp[2] + 0.2).abs() < 1e-10);
    }

    #[test]
    fn test_obb_closest_point_outside_x() {
        let center = [0.0, 0.0, 0.0];
        let half = [1.0, 1.0, 1.0];
        let rot = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];
        let cp = obb_closest_point([5.0, 0.0, 0.0], center, half, rot);
        assert!((cp[0] - 1.0).abs() < 1e-10);
        assert!(cp[1].abs() < 1e-10);
        assert!(cp[2].abs() < 1e-10);
    }

    #[test]
    fn test_obb_closest_point_rotated() {
        // OBB rotated 90° around Z: x-axis becomes y-axis in world
        let center = [0.0, 0.0, 0.0];
        let half = [2.0, 1.0, 1.0];
        // Rotation: x-col=[0,1,0], y-col=[-1,0,0], z-col=[0,0,1]
        let rot = [[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]];
        // A point far in world-y should clamp to OBB local-x extent (2)
        let cp = obb_closest_point([0.0, 5.0, 0.0], center, half, rot);
        // local coords: p_local_x = dot([5y], [0,1,0]) = 5, clamped to 2
        // world closest = center + rot * [2, 0, 0] = [0, 2, 0]
        assert!((cp[0]).abs() < 1e-10);
        assert!((cp[1] - 2.0).abs() < 1e-10);
    }

    #[test]
    fn test_obb_ray_intersection_hit() {
        let center = [0.0, 0.0, 0.0];
        let half = [1.0, 1.0, 1.0];
        let rot = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];
        let result =
            obb_ray_intersection([-5.0, 0.0, 0.0], [1.0, 0.0, 0.0], center, half, rot, 100.0);
        assert!(result.is_some());
        let t = result.unwrap();
        assert!((t - 4.0).abs() < 1e-10);
    }

    #[test]
    fn test_obb_ray_intersection_miss() {
        let center = [0.0, 0.0, 0.0];
        let half = [1.0, 1.0, 1.0];
        let rot = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];
        let result =
            obb_ray_intersection([0.0, 5.0, 0.0], [1.0, 0.0, 0.0], center, half, rot, 100.0);
        assert!(result.is_none());
    }

    #[test]
    fn test_obb_ray_intersection_rotated() {
        // OBB at (3,0,0) with identity rotation, half-extents (1,1,1)
        // Ray from origin along +x should hit at t=2
        let center = [3.0, 0.0, 0.0];
        let half = [1.0, 1.0, 1.0];
        let rot = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];
        let result =
            obb_ray_intersection([0.0, 0.0, 0.0], [1.0, 0.0, 0.0], center, half, rot, 100.0);
        assert!(result.is_some());
        let t = result.unwrap();
        assert!((t - 2.0).abs() < 1e-10);
    }

    #[test]
    fn test_obb_plane_distance_above() {
        // OBB at origin, half (1,1,1), plane normal (0,1,0) at height 3
        // Min signed dist from plane: 3 - half_projection = 3 - 1 = 2
        let center = [0.0, 0.0, 0.0];
        let half = [1.0, 1.0, 1.0];
        let rot = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];
        let plane = [0.0, 1.0, 0.0, -3.0]; // dot(n,p) + d = 0 → y = 3
        let (min_d, max_d) = obb_plane_distance(center, half, rot, plane);
        // center projected: 0*1+0*1+0*0-3 = -3 (below plane, so negative)
        // projection extent: 1
        // min = -3 - 1 = -4, max = -3 + 1 = -2
        assert!((min_d + 4.0).abs() < 1e-10, "min_d={min_d}");
        assert!((max_d + 2.0).abs() < 1e-10, "max_d={max_d}");
    }

    #[test]
    fn test_obb_plane_distance_straddles() {
        // OBB at (0,0,0) half (2,2,2), plane y=0 → min<0, max>0
        let center = [0.0, 0.0, 0.0];
        let half = [2.0, 2.0, 2.0];
        let rot = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];
        let plane = [0.0, 1.0, 0.0, 0.0]; // y plane through origin
        let (min_d, max_d) = obb_plane_distance(center, half, rot, plane);
        assert!(min_d < 0.0);
        assert!(max_d > 0.0);
    }

    #[test]
    fn test_obb_vertices_count() {
        let center = [1.0, 2.0, 3.0];
        let half = [1.0, 1.0, 1.0];
        let rot = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];
        let verts = obb_vertices(center, half, rot);
        assert_eq!(verts.len(), 8);
    }

    #[test]
    fn test_obb_vertices_aabb_bounds() {
        // Identity rotation OBB: vertices should all be within AABB
        let center = [0.0, 0.0, 0.0];
        let half = [1.0, 2.0, 3.0];
        let rot = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];
        let verts = obb_vertices(center, half, rot);
        for v in &verts {
            assert!(v[0].abs() <= 1.0 + 1e-10);
            assert!(v[1].abs() <= 2.0 + 1e-10);
            assert!(v[2].abs() <= 3.0 + 1e-10);
        }
    }

    #[test]
    fn test_obb_vertices_translated() {
        let center = [5.0, 0.0, 0.0];
        let half = [1.0, 1.0, 1.0];
        let rot = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];
        let verts = obb_vertices(center, half, rot);
        // All x-coords should be 4 or 6
        for v in &verts {
            assert!(
                (v[0] - 4.0).abs() < 1e-10 || (v[0] - 6.0).abs() < 1e-10,
                "unexpected x={}",
                v[0]
            );
        }
    }

    #[test]
    fn test_obb_edges_count() {
        let center = [0.0, 0.0, 0.0];
        let half = [1.0, 1.0, 1.0];
        let rot = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];
        let edges = obb_edges(center, half, rot);
        assert_eq!(edges.len(), 12);
    }

    #[test]
    fn test_obb_edges_lengths() {
        let center = [0.0, 0.0, 0.0];
        let half = [1.0, 2.0, 3.0];
        let rot = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];
        let edges = obb_edges(center, half, rot);
        for (a, b) in &edges {
            let dx = b[0] - a[0];
            let dy = b[1] - a[1];
            let dz = b[2] - a[2];
            let len = (dx * dx + dy * dy + dz * dz).sqrt();
            // Each edge should be 2, 4, or 6
            let is_valid =
                (len - 2.0).abs() < 1e-10 || (len - 4.0).abs() < 1e-10 || (len - 6.0).abs() < 1e-10;
            assert!(is_valid, "unexpected edge length {len}");
        }
    }

    #[test]
    fn test_obb_face_centers_count() {
        let center = [0.0, 0.0, 0.0];
        let half = [1.0, 1.0, 1.0];
        let rot = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];
        let fc = obb_face_centers(center, half, rot);
        assert_eq!(fc.len(), 6);
    }

    #[test]
    fn test_obb_face_centers_distance() {
        // For identity OBB with half (1,2,3), face centers should be at ±(1,2,3)
        let center = [0.0, 0.0, 0.0];
        let half = [1.0, 2.0, 3.0];
        let rot = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];
        let fc = obb_face_centers(center, half, rot);
        // Pair 0&1 along x: should be (±1, 0, 0)
        assert!((fc[0][0].abs() - 1.0).abs() < 1e-10);
        assert!((fc[1][0].abs() - 1.0).abs() < 1e-10);
        // Pair 2&3 along y: should be (0, ±2, 0)
        assert!((fc[2][1].abs() - 2.0).abs() < 1e-10);
        assert!((fc[3][1].abs() - 2.0).abs() < 1e-10);
        // Pair 4&5 along z: should be (0, 0, ±3)
        assert!((fc[4][2].abs() - 3.0).abs() < 1e-10);
        assert!((fc[5][2].abs() - 3.0).abs() < 1e-10);
    }

    #[test]
    fn test_obb_support_function_identity() {
        let center = [0.0, 0.0, 0.0];
        let half = [1.0, 2.0, 3.0];
        let rot = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];
        // Support in +x direction should be (1, ±2, ±3), pick most +x = (1, ?, ?)
        let sp = obb_support_fn(center, half, rot, [1.0, 0.0, 0.0]);
        assert!((sp[0] - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_obb_support_function_diagonal() {
        let center = [0.0, 0.0, 0.0];
        let half = [1.0, 1.0, 1.0];
        let rot = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];
        let sp = obb_support_fn(center, half, rot, [1.0, 1.0, 1.0]);
        // Farthest in (1,1,1) direction for unit cube = (1,1,1)
        assert!((sp[0] - 1.0).abs() < 1e-10);
        assert!((sp[1] - 1.0).abs() < 1e-10);
        assert!((sp[2] - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_obb_support_function_translated() {
        let center = [3.0, 0.0, 0.0];
        let half = [1.0, 1.0, 1.0];
        let rot = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];
        let sp = obb_support_fn(center, half, rot, [1.0, 0.0, 0.0]);
        // Center at x=3, half=1, so farthest in +x is at x=4
        assert!((sp[0] - 4.0).abs() < 1e-10);
    }

    #[test]
    fn test_obb_projection_extent_identity() {
        // For identity OBB, projection extent = sum of |half_i * axis_i|
        let half = [1.0, 2.0, 3.0];
        let rot = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];
        let axis = [1.0, 0.0, 0.0];
        let ext = obb_projection_extent(half, rot, axis);
        assert!((ext - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_obb_projection_extent_diagonal() {
        let half = [1.0, 1.0, 1.0];
        let rot = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];
        let s = 1.0 / 3.0_f64.sqrt();
        let ext = obb_projection_extent(half, rot, [s, s, s]);
        assert!((ext - 3.0_f64.sqrt()).abs() < 1e-10);
    }

    #[test]
    fn test_obb_aabb_bounds_identity() {
        let center = [1.0, 2.0, 3.0];
        let half = [1.0, 2.0, 3.0];
        let rot = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];
        let (lo, hi) = obb_aabb_bounds(center, half, rot);
        assert!((lo[0] - 0.0).abs() < 1e-10);
        assert!((hi[0] - 2.0).abs() < 1e-10);
        assert!((lo[1] - 0.0).abs() < 1e-10);
        assert!((hi[1] - 4.0).abs() < 1e-10);
    }

    #[test]
    fn test_obb_aabb_bounds_contains_all_vertices() {
        let center = [0.5, -1.0, 2.0];
        let half = [1.0, 0.5, 1.5];
        let rot = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];
        let (lo, hi) = obb_aabb_bounds(center, half, rot);
        let verts = obb_vertices(center, half, rot);
        for v in &verts {
            assert!(v[0] >= lo[0] - 1e-10 && v[0] <= hi[0] + 1e-10);
            assert!(v[1] >= lo[1] - 1e-10 && v[1] <= hi[1] + 1e-10);
            assert!(v[2] >= lo[2] - 1e-10 && v[2] <= hi[2] + 1e-10);
        }
    }
}
