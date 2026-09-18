// Copyright 2026 COOLJAPAN OU (Team KitaSan)
// SPDX-License-Identifier: Apache-2.0

//! Multibody dynamics: articulated body algorithm, spatial vectors,
//! Featherstone algorithm, closed-loop chains, and hybrid dynamics.
//!
//! This module provides:
//! - Spatial 6-D force and motion screws (Plücker coordinates)
//! - Rigid body inertia in spatial form
//! - Articulated body algorithm (ABA) for forward dynamics
//! - Composite rigid body algorithm (CRBA) for joint-space inertia
//! - Featherstone's recursive Newton-Euler algorithm (RNEA) for inverse dynamics
//! - Tree and loop topologies
//! - Baumgarte / projection constraint stabilization
//! - Hybrid forward/inverse dynamics
//! - Operational space formulation
//! - Redundant DOF handling
//! - Cable-driven systems
//!
//! # Conventions
//!
//! Spatial vectors are stored `[angular; linear]` and follow Featherstone's
//! *Rigid Body Dynamics Algorithms*.  A [`SpatialTransform`] is the placement
//! of a frame in another one; [`SpatialTransform::apply_motion`] maps motion
//! vectors from the outer to the inner frame and
//! [`SpatialTransform::apply_force`] maps force vectors back, so a link whose
//! frame is placed at `X` in its parent's frame sees the parent's motion as
//! `X.apply_motion(..)` and hands its forces to the parent as
//! `X.apply_force(..)`.  Links must be stored parents-first.  External
//! wrenches (`Link::f_ext`) are expressed in the link's own frame, gravity in
//! the world frame.

use crate::dense_linalg::{cholesky_factor, cholesky_solve};
use crate::error::{Error, Result};

/// Joint inertia (`Sᵀ I S`) below which a joint is treated as locked: it
/// cannot accelerate and is skipped by the articulated-body recursion.
const MIN_JOINT_INERTIA: f64 = 1e-15;

/// A pivot smaller than this times `max(1, max|a_ij|)` marks a matrix as
/// singular in [`invert_dense`].
const SINGULAR_PIVOT_TOL: f64 = 1e-12;

/// Damping `μ` of the damped-least-squares inverse `(M + μ² I)⁻¹` used when an
/// exact inverse does not exist.
const DLS_DAMPING: f64 = 1e-6;

/// Maximum number of times the damping is raised (by 10³ each) when `μ²` is
/// too small for the matrix scale.
const DLS_MAX_ESCALATIONS: usize = 8;

// ─────────────────────────────────────────────────────────────────────────────
// 3-D math helpers (no nalgebra; use [f64; 3] arrays)
// ─────────────────────────────────────────────────────────────────────────────

/// Add two 3-vectors.
#[inline]
fn add3(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [a[0] + b[0], a[1] + b[1], a[2] + b[2]]
}

/// Subtract two 3-vectors.
#[inline]
fn sub3(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}

/// Scale a 3-vector.
#[inline]
fn scale3(v: [f64; 3], s: f64) -> [f64; 3] {
    [v[0] * s, v[1] * s, v[2] * s]
}

/// Dot product of two 3-vectors.
#[inline]
fn dot3(a: [f64; 3], b: [f64; 3]) -> f64 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

/// Length of a 3-vector.
#[inline]
fn len3(v: [f64; 3]) -> f64 {
    dot3(v, v).sqrt()
}

/// Normalize a 3-vector.
#[inline]
fn norm3(v: [f64; 3]) -> [f64; 3] {
    let l = len3(v);
    if l < 1e-15 {
        [0.0; 3]
    } else {
        scale3(v, 1.0 / l)
    }
}

/// Cross product of two 3-vectors.
#[inline]
fn cross3(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

// ─── 3×3 matrix helpers ───────────────────────────────────────────────────────

/// Identity 3×3 matrix.
#[inline]
fn mat3_identity() -> [[f64; 3]; 3] {
    [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
}

/// Zero 3×3 matrix.
#[inline]
fn mat3_zero() -> [[f64; 3]; 3] {
    [[0.0; 3]; 3]
}

/// Matrix-vector multiply.
#[inline]
fn mat3_vec(m: [[f64; 3]; 3], v: [f64; 3]) -> [f64; 3] {
    [
        m[0][0] * v[0] + m[0][1] * v[1] + m[0][2] * v[2],
        m[1][0] * v[0] + m[1][1] * v[1] + m[1][2] * v[2],
        m[2][0] * v[0] + m[2][1] * v[1] + m[2][2] * v[2],
    ]
}

/// Matrix-matrix multiply.
#[inline]
fn mat3_mul(a: [[f64; 3]; 3], b: [[f64; 3]; 3]) -> [[f64; 3]; 3] {
    let mut r = [[0.0; 3]; 3];
    for i in 0..3 {
        for j in 0..3 {
            for k in 0..3 {
                r[i][j] += a[i][k] * b[k][j];
            }
        }
    }
    r
}

/// Transpose a 3×3 matrix.
#[inline]
fn mat3_transpose(m: [[f64; 3]; 3]) -> [[f64; 3]; 3] {
    [
        [m[0][0], m[1][0], m[2][0]],
        [m[0][1], m[1][1], m[2][1]],
        [m[0][2], m[1][2], m[2][2]],
    ]
}

/// Add two 3×3 matrices.
#[inline]
fn mat3_add(a: [[f64; 3]; 3], b: [[f64; 3]; 3]) -> [[f64; 3]; 3] {
    let mut r = [[0.0; 3]; 3];
    for i in 0..3 {
        for j in 0..3 {
            r[i][j] = a[i][j] + b[i][j];
        }
    }
    r
}

/// Scale a 3×3 matrix.
#[inline]
fn mat3_scale(m: [[f64; 3]; 3], s: f64) -> [[f64; 3]; 3] {
    let mut r = [[0.0; 3]; 3];
    for i in 0..3 {
        for j in 0..3 {
            r[i][j] = m[i][j] * s;
        }
    }
    r
}

/// Skew-symmetric (cross product) matrix of v.
#[inline]
fn skew(v: [f64; 3]) -> [[f64; 3]; 3] {
    [[0.0, -v[2], v[1]], [v[2], 0.0, -v[0]], [-v[1], v[0], 0.0]]
}

// ─────────────────────────────────────────────────────────────────────────────
// Spatial 6-D Vectors (Plücker coordinates)
// ─────────────────────────────────────────────────────────────────────────────

/// A spatial motion vector (twist) or force vector (wrench) in Plücker coordinates.
///
/// Layout: \[angular; linear\] — consistent with Featherstone's notation.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SpatialVec {
    /// Angular component (ω for twists, τ for wrenches).
    pub angular: [f64; 3],
    /// Linear component (v for twists, f for wrenches).
    pub linear: [f64; 3],
}

impl SpatialVec {
    /// Create a zero spatial vector.
    pub fn zero() -> Self {
        Self {
            angular: [0.0; 3],
            linear: [0.0; 3],
        }
    }

    /// Create from angular and linear parts.
    pub fn new(angular: [f64; 3], linear: [f64; 3]) -> Self {
        Self { angular, linear }
    }

    /// Add two spatial vectors.
    pub fn add(&self, other: &SpatialVec) -> SpatialVec {
        SpatialVec::new(
            add3(self.angular, other.angular),
            add3(self.linear, other.linear),
        )
    }

    /// Subtract two spatial vectors.
    pub fn sub(&self, other: &SpatialVec) -> SpatialVec {
        SpatialVec::new(
            sub3(self.angular, other.angular),
            sub3(self.linear, other.linear),
        )
    }

    /// Scale a spatial vector.
    pub fn scale(&self, s: f64) -> SpatialVec {
        SpatialVec::new(scale3(self.angular, s), scale3(self.linear, s))
    }

    /// Spatial inner product: s^T * f = ω·τ + v·f.
    pub fn dot(&self, other: &SpatialVec) -> f64 {
        dot3(self.angular, other.angular) + dot3(self.linear, other.linear)
    }

    /// Spatial cross product for motion vectors: m × m'.
    pub fn cross_motion(&self, other: &SpatialVec) -> SpatialVec {
        SpatialVec::new(
            cross3(self.angular, other.angular),
            add3(
                cross3(self.angular, other.linear),
                cross3(self.linear, other.angular),
            ),
        )
    }

    /// Spatial cross product for wrench: m × f.
    pub fn cross_force(&self, f: &SpatialVec) -> SpatialVec {
        SpatialVec::new(
            add3(
                cross3(self.angular, f.angular),
                cross3(self.linear, f.linear),
            ),
            cross3(self.angular, f.linear),
        )
    }

    /// Euclidean norm of the spatial vector.
    pub fn norm(&self) -> f64 {
        (dot3(self.angular, self.angular) + dot3(self.linear, self.linear)).sqrt()
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Spatial 6×6 Matrix
// ─────────────────────────────────────────────────────────────────────────────

/// A 6×6 spatial matrix (for inertia, transforms, etc.).
#[derive(Debug, Clone, Copy)]
pub struct SpatialMat {
    /// 4 quadrants, each 3×3: \[\[A, B\\], \[C, D\]].
    pub a: [[f64; 3]; 3],
    /// Upper right block.
    pub b: [[f64; 3]; 3],
    /// Lower left block.
    pub c: [[f64; 3]; 3],
    /// Lower right block.
    pub d: [[f64; 3]; 3],
}

impl SpatialMat {
    /// Zero spatial matrix.
    pub fn zero() -> Self {
        Self {
            a: mat3_zero(),
            b: mat3_zero(),
            c: mat3_zero(),
            d: mat3_zero(),
        }
    }

    /// Identity spatial matrix.
    pub fn identity() -> Self {
        Self {
            a: mat3_identity(),
            b: mat3_zero(),
            c: mat3_zero(),
            d: mat3_identity(),
        }
    }

    /// Multiply a spatial matrix by a spatial vector.
    pub fn mul_vec(&self, v: &SpatialVec) -> SpatialVec {
        SpatialVec::new(
            add3(mat3_vec(self.a, v.angular), mat3_vec(self.b, v.linear)),
            add3(mat3_vec(self.c, v.angular), mat3_vec(self.d, v.linear)),
        )
    }

    /// Multiply two spatial matrices.
    pub fn mul_mat(&self, other: &SpatialMat) -> SpatialMat {
        SpatialMat {
            a: mat3_add(mat3_mul(self.a, other.a), mat3_mul(self.b, other.c)),
            b: mat3_add(mat3_mul(self.a, other.b), mat3_mul(self.b, other.d)),
            c: mat3_add(mat3_mul(self.c, other.a), mat3_mul(self.d, other.c)),
            d: mat3_add(mat3_mul(self.c, other.b), mat3_mul(self.d, other.d)),
        }
    }

    /// Add two spatial matrices.
    pub fn add_mat(&self, other: &SpatialMat) -> SpatialMat {
        SpatialMat {
            a: mat3_add(self.a, other.a),
            b: mat3_add(self.b, other.b),
            c: mat3_add(self.c, other.c),
            d: mat3_add(self.d, other.d),
        }
    }

    /// Scale a spatial matrix.
    pub fn scale_mat(&self, s: f64) -> SpatialMat {
        SpatialMat {
            a: mat3_scale(self.a, s),
            b: mat3_scale(self.b, s),
            c: mat3_scale(self.c, s),
            d: mat3_scale(self.d, s),
        }
    }

    /// Transpose.
    pub fn transpose(&self) -> SpatialMat {
        SpatialMat {
            a: mat3_transpose(self.a),
            b: mat3_transpose(self.c),
            c: mat3_transpose(self.b),
            d: mat3_transpose(self.d),
        }
    }

    /// Outer product `u vᵀ` of two spatial vectors (a rank-one 6×6 matrix).
    pub fn outer(u: &SpatialVec, v: &SpatialVec) -> SpatialMat {
        let block = |x: [f64; 3], y: [f64; 3]| {
            let mut m = [[0.0; 3]; 3];
            for (row, xi) in m.iter_mut().zip(x) {
                for (entry, yj) in row.iter_mut().zip(y) {
                    *entry = xi * yj;
                }
            }
            m
        };
        SpatialMat {
            a: block(u.angular, v.angular),
            b: block(u.angular, v.linear),
            c: block(u.linear, v.angular),
            d: block(u.linear, v.linear),
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Rigid Body Spatial Inertia
// ─────────────────────────────────────────────────────────────────────────────

/// Rigid body spatial inertia matrix I* (6×6 symmetric positive definite).
///
/// In body frame, the spatial inertia is:
/// I* = \[\[Ic + m c× c×^T, m c×\\], \[m c×^T, m I\]]
/// where Ic is the body-frame rotational inertia about CoM,
/// c is the CoM offset from origin, and I is the 3×3 identity.
#[derive(Debug, Clone, Copy)]
pub struct RigidBodyInertia {
    /// Body mass.
    pub mass: f64,
    /// Center of mass offset in body frame.
    pub com: [f64; 3],
    /// Rotational inertia about CoM (symmetric 3×3, row-major).
    pub inertia_com: [[f64; 3]; 3],
}

impl RigidBodyInertia {
    /// Create a new rigid body inertia.
    pub fn new(mass: f64, com: [f64; 3], inertia_com: [[f64; 3]; 3]) -> Self {
        Self {
            mass,
            com,
            inertia_com,
        }
    }

    /// Create a point mass at origin.
    pub fn point_mass(mass: f64) -> Self {
        Self {
            mass,
            com: [0.0; 3],
            inertia_com: mat3_zero(),
        }
    }

    /// Sphere inertia: I = 2/5 m r².
    pub fn sphere(mass: f64, radius: f64) -> Self {
        let i = 2.0 / 5.0 * mass * radius * radius;
        let inertia = [[i, 0.0, 0.0], [0.0, i, 0.0], [0.0, 0.0, i]];
        Self {
            mass,
            com: [0.0; 3],
            inertia_com: inertia,
        }
    }

    /// Box inertia: I_x = m(b²+c²)/12, etc.
    pub fn box_shape(mass: f64, hx: f64, hy: f64, hz: f64) -> Self {
        let ix = mass * (hy * hy + hz * hz) / 3.0;
        let iy = mass * (hx * hx + hz * hz) / 3.0;
        let iz = mass * (hx * hx + hy * hy) / 3.0;
        let inertia = [[ix, 0.0, 0.0], [0.0, iy, 0.0], [0.0, 0.0, iz]];
        Self {
            mass,
            com: [0.0; 3],
            inertia_com: inertia,
        }
    }

    /// Convert to spatial inertia matrix (6×6).
    pub fn to_spatial(&self) -> SpatialMat {
        let m = self.mass;
        let c = self.com;
        let cx = skew(c);
        let cx_t = mat3_transpose(cx);
        // I_rot = Ic + m * c× * c×^T
        let cx_cxt = mat3_mul(cx, cx_t);
        let i_rot = mat3_add(self.inertia_com, mat3_scale(cx_cxt, m));
        // Upper right: m * c×
        let m_cx = mat3_scale(cx, m);
        // Lower left: m * c×^T
        let m_cxt = mat3_scale(cx_t, m);
        // Lower right: m * I
        let m_i = mat3_scale(mat3_identity(), m);
        SpatialMat {
            a: i_rot,
            b: m_cx,
            c: m_cxt,
            d: m_i,
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Spatial Transform (Plücker transform)
// ─────────────────────────────────────────────────────────────────────────────

/// A rigid-body placement `(R, p)`: the pose of a frame B expressed in a
/// frame A.
///
/// A point with coordinates `x_B` in B has the coordinates `x_A = R x_B + p` in
/// A; the columns of `R` are the axes of B written in A coordinates and `p` is
/// the origin of B in A coordinates.  This is the convention of
/// [`SpatialTransform::compose`], [`SpatialTransform::inverse`], the joint
/// transforms and the world transforms computed by forward kinematics.
///
/// The same pair acts on Plücker vectors: [`SpatialTransform::apply_motion`]
/// maps a motion vector from A to B coordinates (Featherstone's `^B X_A`) and
/// [`SpatialTransform::apply_force`] maps a force vector from B to A
/// coordinates (`^A X*_B = (^B X_A)ᵀ`).  For a link, A is the parent frame and
/// B the link frame.
#[derive(Debug, Clone, Copy)]
pub struct SpatialTransform {
    /// Rotation of frame B in frame A (the columns are B's axes in A coordinates).
    pub rot: [[f64; 3]; 3],
    /// Origin of frame B in A coordinates.
    pub pos: [f64; 3],
}

impl SpatialTransform {
    /// Identity transform.
    pub fn identity() -> Self {
        Self {
            rot: mat3_identity(),
            pos: [0.0; 3],
        }
    }

    /// Pure translation.
    pub fn translation(p: [f64; 3]) -> Self {
        Self {
            rot: mat3_identity(),
            pos: p,
        }
    }

    /// Pure rotation (rotation matrix).
    pub fn rotation(r: [[f64; 3]; 3]) -> Self {
        Self {
            rot: r,
            pos: [0.0; 3],
        }
    }

    /// Build from position and rotation.
    pub fn new(rot: [[f64; 3]; 3], pos: [f64; 3]) -> Self {
        Self { rot, pos }
    }

    /// Transform a spatial motion vector (twist) from frame A to frame B
    /// (Featherstone's `^B X_A`): `ω_B = Rᵀ ω_A`, `v_B = Rᵀ (v_A − p × ω_A)`.
    ///
    /// For a link this maps parent-frame motion to link-frame motion.
    pub fn apply_motion(&self, v: &SpatialVec) -> SpatialVec {
        let rt = mat3_transpose(self.rot);
        let v_minus_p_cross_omega = sub3(v.linear, cross3(self.pos, v.angular));
        SpatialVec::new(mat3_vec(rt, v.angular), mat3_vec(rt, v_minus_p_cross_omega))
    }

    /// Transform a spatial force vector (wrench) from frame B to frame A
    /// (Featherstone's `^A X*_B = (^B X_A)ᵀ`): `f_A = R f_B`,
    /// `n_A = R n_B + p × f_A`.
    ///
    /// For a link this carries a link-frame wrench to the parent frame.
    pub fn apply_force(&self, f: &SpatialVec) -> SpatialVec {
        let force = mat3_vec(self.rot, f.linear);
        SpatialVec::new(
            add3(mat3_vec(self.rot, f.angular), cross3(self.pos, force)),
            force,
        )
    }

    /// The 6×6 matrix of [`SpatialTransform::apply_motion`]:
    /// `^B X_A = [[Rᵀ, 0], [−Rᵀ p×, Rᵀ]]`.
    pub fn motion_matrix(&self) -> SpatialMat {
        let rt = mat3_transpose(self.rot);
        SpatialMat {
            a: rt,
            b: mat3_zero(),
            c: mat3_scale(mat3_mul(rt, skew(self.pos)), -1.0),
            d: rt,
        }
    }

    /// The 6×6 matrix of [`SpatialTransform::apply_force`]:
    /// `^A X*_B = (^B X_A)ᵀ`.
    pub fn force_matrix(&self) -> SpatialMat {
        self.motion_matrix().transpose()
    }

    /// Express a spatial inertia given in frame B in frame A:
    /// `^A X*_B I ^B X_A`.
    ///
    /// This is how the inertia of a subtree (composite or articulated) is
    /// accumulated into its parent link.
    pub fn inertia_to_parent(&self, inertia: &SpatialMat) -> SpatialMat {
        let x = self.motion_matrix();
        x.transpose().mul_mat(inertia).mul_mat(&x)
    }

    /// Inverse transform.
    pub fn inverse(&self) -> Self {
        let rt = mat3_transpose(self.rot);
        // p_inv = -R^T * p
        let p_inv = scale3(mat3_vec(rt, self.pos), -1.0);
        Self {
            rot: rt,
            pos: p_inv,
        }
    }

    /// Compose transforms: self * other.
    pub fn compose(&self, other: &SpatialTransform) -> Self {
        let rot = mat3_mul(self.rot, other.rot);
        let pos = add3(self.pos, mat3_vec(self.rot, other.pos));
        Self { rot, pos }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Joint types
// ─────────────────────────────────────────────────────────────────────────────

/// Joint type for articulated body dynamics.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum JointType {
    /// Revolute joint: 1 rotational DOF.
    Revolute,
    /// Prismatic joint: 1 translational DOF.
    Prismatic,
    /// Spherical joint: 3 rotational DOFs.
    Spherical,
    /// Planar joint: 2 translational + 1 rotational DOF.
    Planar,
    /// Free joint: 6 DOFs.
    Free,
    /// Fixed (welded) joint: 0 DOFs.
    Fixed,
}

/// Joint axis definition.
#[derive(Debug, Clone, Copy)]
pub struct JointAxis {
    /// Joint type.
    pub joint_type: JointType,
    /// Joint axis direction (for revolute/prismatic).
    pub axis: [f64; 3],
    /// Joint spatial motion subspace (columns are motion axes).
    pub s_matrix: SpatialVec,
}

impl JointAxis {
    /// Create a revolute joint about the given axis.
    pub fn revolute(axis: [f64; 3]) -> Self {
        let a = norm3(axis);
        Self {
            joint_type: JointType::Revolute,
            axis: a,
            s_matrix: SpatialVec::new(a, [0.0; 3]),
        }
    }

    /// Create a prismatic joint along the given axis.
    pub fn prismatic(axis: [f64; 3]) -> Self {
        let a = norm3(axis);
        Self {
            joint_type: JointType::Prismatic,
            axis: a,
            s_matrix: SpatialVec::new([0.0; 3], a),
        }
    }

    /// DOF count for this joint type.
    pub fn dof(&self) -> usize {
        match self.joint_type {
            JointType::Revolute | JointType::Prismatic => 1,
            JointType::Spherical => 3,
            JointType::Planar => 3,
            JointType::Free => 6,
            JointType::Fixed => 0,
        }
    }

    /// Compute the joint transform for scalar joint position q.
    pub fn joint_transform(&self, q: f64) -> SpatialTransform {
        match self.joint_type {
            JointType::Revolute => {
                let axis = self.axis;
                // Rodrigues rotation
                let c = q.cos();
                let s = q.sin();
                let one_minus_c = 1.0 - c;
                let [ax, ay, az] = axis;
                let rot = [
                    [
                        c + ax * ax * one_minus_c,
                        ax * ay * one_minus_c - az * s,
                        ax * az * one_minus_c + ay * s,
                    ],
                    [
                        ay * ax * one_minus_c + az * s,
                        c + ay * ay * one_minus_c,
                        ay * az * one_minus_c - ax * s,
                    ],
                    [
                        az * ax * one_minus_c - ay * s,
                        az * ay * one_minus_c + ax * s,
                        c + az * az * one_minus_c,
                    ],
                ];
                SpatialTransform::rotation(rot)
            }
            JointType::Prismatic => {
                let p = scale3(self.axis, q);
                SpatialTransform::translation(p)
            }
            JointType::Fixed => SpatialTransform::identity(),
            _ => SpatialTransform::identity(),
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Rigid Body Link
// ─────────────────────────────────────────────────────────────────────────────

/// A single rigid body link in an articulated system.
#[derive(Debug, Clone)]
pub struct Link {
    /// Link name.
    pub name: String,
    /// Parent link index (-1 for root).
    pub parent: i64,
    /// Rigid body inertia.
    pub inertia: RigidBodyInertia,
    /// Joint connecting this link to its parent.
    pub joint: JointAxis,
    /// Fixed transform from parent joint frame to this link's body frame.
    pub x_t: SpatialTransform,
    /// Fixed transform from body frame to joint frame.
    pub x_j_fixed: SpatialTransform,
    /// Current joint position (generalized coordinate).
    pub q: f64,
    /// Current joint velocity (generalized velocity).
    pub qd: f64,
    /// Current joint acceleration (generalized acceleration).
    pub qdd: f64,
    /// Applied joint torque / force.
    pub tau: f64,
    /// External spatial force (wrench about the body-frame origin) on this
    /// body, expressed in the body frame.
    pub f_ext: SpatialVec,
    /// Current body velocity (spatial).
    pub body_vel: SpatialVec,
    /// Current body acceleration (spatial).
    pub body_acc: SpatialVec,
    /// Articulated body inertia (used in ABA).
    pub abi: SpatialMat,
    /// Articulated body bias force (used in ABA).
    pub p_a: SpatialVec,
    /// ABA intermediate: U = I_A * S.
    pub u_aba: SpatialVec,
    /// ABA intermediate: d = S^T * U.
    pub d_aba: f64,
    /// ABA intermediate: u = tau - S^T * p_a.
    pub u_tau: f64,
    /// World-space transform (set during forward kinematics).
    pub x_world: SpatialTransform,
}

impl Link {
    /// Create a new link.
    pub fn new(
        name: impl Into<String>,
        parent: i64,
        inertia: RigidBodyInertia,
        joint: JointAxis,
        x_t: SpatialTransform,
    ) -> Self {
        Self {
            name: name.into(),
            parent,
            inertia,
            joint,
            x_t,
            x_j_fixed: SpatialTransform::identity(),
            q: 0.0,
            qd: 0.0,
            qdd: 0.0,
            tau: 0.0,
            f_ext: SpatialVec::zero(),
            body_vel: SpatialVec::zero(),
            body_acc: SpatialVec::zero(),
            abi: SpatialMat::zero(),
            p_a: SpatialVec::zero(),
            u_aba: SpatialVec::zero(),
            d_aba: 1.0,
            u_tau: 0.0,
            x_world: SpatialTransform::identity(),
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Articulated Body (multibody tree)
// ─────────────────────────────────────────────────────────────────────────────

/// An articulated rigid body system (tree topology).
///
/// Every recursive algorithm walks the links in index order (and in reverse),
/// so a link's parent must have a smaller index than the link itself; roots
/// have `parent == -1` and hang off the (fixed) world.
#[derive(Debug, Clone)]
pub struct ArticulatedBody {
    /// All links (index 0 is typically the base/root).
    pub links: Vec<Link>,
    /// Gravitational acceleration in the world frame (e.g. `[0, -9.81, 0]`).
    pub gravity: [f64; 3],
}

impl ArticulatedBody {
    /// Create an articulated body with given gravity.
    pub fn new(gravity: [f64; 3]) -> Self {
        Self {
            links: Vec::new(),
            gravity,
        }
    }

    /// Add a link and return its index.
    pub fn add_link(&mut self, link: Link) -> usize {
        let idx = self.links.len();
        self.links.push(link);
        idx
    }

    /// Number of links.
    pub fn num_links(&self) -> usize {
        self.links.len()
    }

    /// Total degrees of freedom.
    pub fn num_dof(&self) -> usize {
        self.links.iter().map(|l| l.joint.dof()).sum()
    }

    /// Get the joint position vector.
    pub fn get_q(&self) -> Vec<f64> {
        self.links.iter().map(|l| l.q).collect()
    }

    /// Set the joint position vector.
    pub fn set_q(&mut self, q: &[f64]) {
        for (i, link) in self.links.iter_mut().enumerate() {
            if i < q.len() {
                link.q = q[i];
            }
        }
    }

    /// Get the joint velocity vector.
    pub fn get_qd(&self) -> Vec<f64> {
        self.links.iter().map(|l| l.qd).collect()
    }

    /// Set joint velocities.
    pub fn set_qd(&mut self, qd: &[f64]) {
        for (i, link) in self.links.iter_mut().enumerate() {
            if i < qd.len() {
                link.qd = qd[i];
            }
        }
    }

    /// Placement of link `i`'s frame in its parent's frame (the world frame for
    /// a root): the fixed joint-frame placement followed by the joint motion.
    fn placement_in_parent(&self, i: usize) -> SpatialTransform {
        let link = &self.links[i];
        link.x_t.compose(&link.joint.joint_transform(link.q))
    }

    /// Forward kinematics: compute world transforms and body velocities.
    pub fn forward_kinematics(&mut self) {
        let n = self.links.len();
        for i in 0..n {
            let xj = self.links[i].joint.joint_transform(self.links[i].q);
            let x_lambda_i = self.links[i].x_t.compose(&xj);
            let parent = self.links[i].parent;
            if parent < 0 {
                self.links[i].x_world = x_lambda_i;
                // Body velocity: v = S * qd
                let s = self.links[i].joint.s_matrix;
                self.links[i].body_vel = s.scale(self.links[i].qd);
            } else {
                let parent_world = self.links[parent as usize].x_world;
                self.links[i].x_world = parent_world.compose(&x_lambda_i);
                // Body velocity: v_i = x_T^{-1} * v_parent + S * qd
                let x_parent_to_child = x_lambda_i;
                let v_parent = self.links[parent as usize].body_vel;
                let v_parent_in_child = x_parent_to_child.apply_motion(&v_parent);
                let s = self.links[i].joint.s_matrix;
                self.links[i].body_vel = v_parent_in_child.add(&s.scale(self.links[i].qd));
            }
        }
    }

    // ─── Recursive Newton-Euler Algorithm (RNEA / Inverse Dynamics) ──────────

    /// Recursive Newton-Euler Algorithm: compute joint torques for given
    /// joint accelerations and external forces.
    ///
    /// Featherstone RNEA (1987).  The forward pass propagates velocities and
    /// accelerations outwards; gravity enters as the fictitious base
    /// acceleration `−g`, which is expressed in the world frame and carried
    /// into each root link's frame.  The backward pass accumulates the net link
    /// forces `I a + v ×* I v − f_ext` towards the roots (`f_ext` is a wrench
    /// expressed in the link's own frame) and projects them on the joint
    /// motion subspaces.
    ///
    /// The `body_vel` and `body_acc` fields of every link are updated.
    pub fn rnea(&mut self, qdd: &[f64], gravity: Option<[f64; 3]>) -> Vec<f64> {
        let g = gravity.unwrap_or(self.gravity);
        let n = self.links.len();
        let base_acc = SpatialVec::new([0.0; 3], scale3(g, -1.0));

        // Forward pass: compute body velocities and accelerations
        for i in 0..n {
            let x_lambda_i = self.placement_in_parent(i);
            let parent = self.links[i].parent;
            let s = self.links[i].joint.s_matrix;
            let qd_i = self.links[i].qd;
            let qdd_i = qdd.get(i).copied().unwrap_or(0.0);

            let (v_parent_c, a_parent_c) = if parent < 0 {
                // The world is at rest; only the gravity pseudo-acceleration
                // reaches the root link.
                (SpatialVec::zero(), x_lambda_i.apply_motion(&base_acc))
            } else {
                let p = parent as usize;
                (
                    x_lambda_i.apply_motion(&self.links[p].body_vel),
                    x_lambda_i.apply_motion(&self.links[p].body_acc),
                )
            };
            let v_joint = s.scale(qd_i);
            let v_i = v_parent_c.add(&v_joint);
            // Velocity-product (Coriolis) term: v_i × S q̇
            let coriolis = v_i.cross_motion(&v_joint);
            self.links[i].body_vel = v_i;
            self.links[i].body_acc = a_parent_c.add(&s.scale(qdd_i)).add(&coriolis);
        }

        // Backward pass: compute net forces
        let mut f_vec: Vec<SpatialVec> = vec![SpatialVec::zero(); n];
        for i in (0..n).rev() {
            let i_body = self.links[i].inertia.to_spatial();
            let v_i = self.links[i].body_vel;
            let a_i = self.links[i].body_acc;
            // fi = I*a + v×* I*v - f_ext
            let i_v = i_body.mul_vec(&v_i);
            let v_cross_i_v = v_i.cross_force(&i_v);
            let i_a = i_body.mul_vec(&a_i);
            let f_i = i_a
                .add(&v_cross_i_v)
                .sub(&self.links[i].f_ext)
                .add(&f_vec[i]);
            f_vec[i] = f_i;
            let parent = self.links[i].parent;
            if parent >= 0 {
                let f_parent_contrib = self.placement_in_parent(i).apply_force(&f_i);
                f_vec[parent as usize] = f_vec[parent as usize].add(&f_parent_contrib);
            }
        }

        // Joint torques: tau_i = S_i^T * f_i
        (0..n)
            .map(|i| {
                let s = self.links[i].joint.s_matrix;
                s.dot(&f_vec[i])
            })
            .collect()
    }

    // ─── Articulated Body Algorithm (ABA / Forward Dynamics) ─────────────────

    /// Articulated Body Algorithm: compute joint accelerations for given torques.
    ///
    /// Featherstone ABA (1983), in the form of *Rigid Body Dynamics
    /// Algorithms* (Table 7.1):
    ///
    /// 1. An outward pass computes the link velocities and the
    ///    velocity-product terms `c_i = v_i × S_i q̇_i`, and initialises the
    ///    articulated inertia `I^A_i` with the rigid-body inertia and the bias
    ///    force with `p^A_i = v_i ×* I_i v_i − f_ext,i`.
    /// 2. An inward pass forms `U_i = I^A_i S_i`, `d_i = S_iᵀ U_i`,
    ///    `u_i = τ_i − S_iᵀ p^A_i` and hands the part of the subtree that the
    ///    joint cannot move to the parent: `I^a = I^A_i − U_i U_iᵀ / d_i`,
    ///    `p^a = p^A_i + I^a c_i + U_i u_i / d_i`, `I^A_λ += X* I^a X`,
    ///    `p^A_λ += X* p^a`.
    /// 3. An outward pass resolves the accelerations: `a' = X a_λ + c_i`,
    ///    `q̈_i = (u_i − U_iᵀ a') / d_i`, `a_i = a' + S_i q̈_i`, with gravity
    ///    entering as the base acceleration `−g`.
    ///
    /// External wrenches (`f_ext`, in the link frame), the velocity-product
    /// terms and gravity are all included, so `aba` inverts [`Self::rnea`]:
    /// `aba(rnea(q̈)) == q̈` in every state.  A joint whose articulated inertia
    /// about its motion subspace vanishes (a zero subspace such as a fixed
    /// joint, or a massless subtree) is treated as locked: its acceleration is
    /// zero and it passes its full inertia on to the parent.
    ///
    /// The `body_vel`, `body_acc`, `qdd` and the articulated-body fields of
    /// every link are updated.
    pub fn aba(&mut self, tau: &[f64]) -> Vec<f64> {
        let n = self.links.len();
        let base_acc = SpatialVec::new([0.0; 3], scale3(self.gravity, -1.0));
        let mut velocity_product: Vec<SpatialVec> = Vec::with_capacity(n);

        // Pass 1: velocities, velocity-product terms, rigid-body inertias and
        // bias forces
        for i in 0..n {
            let x_lambda_i = self.placement_in_parent(i);
            let parent = self.links[i].parent;
            let v_joint = self.links[i].joint.s_matrix.scale(self.links[i].qd);
            let v_i = if parent < 0 {
                v_joint
            } else {
                x_lambda_i
                    .apply_motion(&self.links[parent as usize].body_vel)
                    .add(&v_joint)
            };
            velocity_product.push(v_i.cross_motion(&v_joint));
            let i_body = self.links[i].inertia.to_spatial();
            let momentum = i_body.mul_vec(&v_i);
            self.links[i].body_vel = v_i;
            self.links[i].abi = i_body;
            self.links[i].p_a = v_i.cross_force(&momentum).sub(&self.links[i].f_ext);
        }

        // Pass 2: backward articulated inertia and bias force propagation
        for i in (0..n).rev() {
            let s = self.links[i].joint.s_matrix;
            let abi = self.links[i].abi;
            let p_a = self.links[i].p_a;
            let u_vec = abi.mul_vec(&s);
            let d = s.dot(&u_vec);
            let tau_i = tau.get(i).copied().unwrap_or(0.0);
            let u_tau = tau_i - s.dot(&p_a);
            self.links[i].u_aba = u_vec;
            self.links[i].d_aba = d;
            self.links[i].u_tau = u_tau;

            let parent = self.links[i].parent;
            if parent >= 0 {
                // Articulated inertia / bias force of link i as seen by the
                // parent, i.e. with the joint's own freedom projected out.
                let (i_a, p_prop) = if d.abs() < MIN_JOINT_INERTIA {
                    (abi, p_a.add(&abi.mul_vec(&velocity_product[i])))
                } else {
                    let i_a = abi.add_mat(&SpatialMat::outer(&u_vec, &u_vec).scale_mat(-1.0 / d));
                    let p_prop = p_a
                        .add(&i_a.mul_vec(&velocity_product[i]))
                        .add(&u_vec.scale(u_tau / d));
                    (i_a, p_prop)
                };
                let x_lambda_i = self.placement_in_parent(i);
                let parent_link = &mut self.links[parent as usize];
                parent_link.abi = parent_link.abi.add_mat(&x_lambda_i.inertia_to_parent(&i_a));
                parent_link.p_a = parent_link.p_a.add(&x_lambda_i.apply_force(&p_prop));
            }
        }

        // Pass 3: forward propagation of accelerations
        let mut qdd_out = vec![0.0f64; n];
        for (i, qdd_slot) in qdd_out.iter_mut().enumerate() {
            let x_lambda_i = self.placement_in_parent(i);
            let parent = self.links[i].parent;
            let a_parent_c = if parent < 0 {
                x_lambda_i.apply_motion(&base_acc)
            } else {
                x_lambda_i.apply_motion(&self.links[parent as usize].body_acc)
            };
            let a_prime = a_parent_c.add(&velocity_product[i]);
            let d = self.links[i].d_aba;
            let qdd_i = if d.abs() < MIN_JOINT_INERTIA {
                0.0
            } else {
                (self.links[i].u_tau - self.links[i].u_aba.dot(&a_prime)) / d
            };
            let s = self.links[i].joint.s_matrix;
            *qdd_slot = qdd_i;
            self.links[i].qdd = qdd_i;
            self.links[i].body_acc = a_prime.add(&s.scale(qdd_i));
        }

        qdd_out
    }

    // ─── Composite Rigid Body Algorithm (CRBA) ───────────────────────────────

    /// Compute the joint-space inertia matrix H using CRBA.
    ///
    /// Featherstone CRBA (*Rigid Body Dynamics Algorithms*, Table 6.2): the
    /// composite inertia of every subtree is accumulated into its parent in the
    /// parent's frame, `Ic_λ += X* Ic_i X`, and the off-diagonal entries
    /// `H_ij = S_jᵀ X* … X* Ic_i S_i` carry the force `Ic_i S_i` up the tree,
    /// transforming it joint by joint.
    ///
    /// Returns H (n×n, symmetric), which can be used for constrained dynamics.
    pub fn crba(&mut self) -> Vec<Vec<f64>> {
        let n = self.links.len();
        let mut ic: Vec<SpatialMat> = self.links.iter().map(|l| l.inertia.to_spatial()).collect();

        // Backward pass: accumulate composite inertias in the parent frame
        for i in (0..n).rev() {
            let parent = self.links[i].parent;
            if parent >= 0 {
                let transformed = self.placement_in_parent(i).inertia_to_parent(&ic[i]);
                ic[parent as usize] = ic[parent as usize].add_mat(&transformed);
            }
        }

        // Build H matrix: H[i][j] = S_j^T * (force of Ic_i S_i carried to j's frame)
        let mut h = vec![vec![0.0f64; n]; n];
        for i in 0..n {
            let s_i = self.links[i].joint.s_matrix;
            let mut force = ic[i].mul_vec(&s_i);
            // Diagonal entry
            h[i][i] = s_i.dot(&force);
            // Off-diagonal: traverse up, re-expressing the force in each
            // ancestor's frame before projecting on its motion subspace
            let mut j = i;
            while self.links[j].parent >= 0 {
                force = self.placement_in_parent(j).apply_force(&force);
                j = self.links[j].parent as usize;
                let s_j = self.links[j].joint.s_matrix;
                let h_ij = s_j.dot(&force);
                h[i][j] = h_ij;
                h[j][i] = h_ij;
            }
        }

        h
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Baumgarte stabilization
// ─────────────────────────────────────────────────────────────────────────────

/// Baumgarte stabilization parameters.
#[derive(Debug, Clone, Copy)]
pub struct BaumgarteParams {
    /// Position feedback gain α.
    pub alpha: f64,
    /// Velocity feedback gain β.
    pub beta: f64,
}

impl Default for BaumgarteParams {
    fn default() -> Self {
        Self {
            alpha: 0.1,
            beta: 0.1,
        }
    }
}

/// Compute Baumgarte-stabilized constraint acceleration correction.
///
/// δa = -2α * ω_n * ẋ - ω_n² * x
/// where α is the damping ratio, ω_n = β/dt the natural frequency.
pub fn baumgarte_stabilization(
    pos_error: f64,
    vel_error: f64,
    params: &BaumgarteParams,
    dt: f64,
) -> f64 {
    let omega_n = params.beta / dt;
    -2.0 * params.alpha * omega_n * vel_error - omega_n * omega_n * pos_error
}

/// Position-level constraint projection (non-linear stabilization).
///
/// Directly corrects joint positions by projecting onto the constraint manifold.
pub fn project_constraint(q: f64, constraint_val: f64, gain: f64) -> f64 {
    q - gain * constraint_val
}

// ─────────────────────────────────────────────────────────────────────────────
// Operational Space Formulation
// ─────────────────────────────────────────────────────────────────────────────

/// Compute the operational space inertia matrix Λ.
///
/// Λ = (J H⁻¹ J^T)⁻¹
/// where J is the task Jacobian, H is the joint-space inertia.
///
/// This simplified version takes H as a diagonal (for decoupled joints); a
/// joint without an entry in `h_inv_diag` does not contribute.  The inversion
/// is exact for any task dimension; when `J H⁻¹ J^T` is singular (a
/// kinematic singularity of `J`) a damped-least-squares inverse
/// `(M + μ² I)⁻¹` is returned instead, so the result is always finite.
pub fn operational_space_inertia(j: &[Vec<f64>], h_inv_diag: &[f64]) -> Vec<Vec<f64>> {
    let m = j.len(); // task DOFs
    let n = j.iter().map(Vec::len).min().unwrap_or(0); // joint DOFs
    // Compute J * H⁻¹ * J^T
    let mut lambda_inv = vec![vec![0.0f64; m]; m];
    for i in 0..m {
        for k in 0..m {
            let mut sum = 0.0;
            for l in 0..n {
                if l < h_inv_diag.len() {
                    sum += j[i][l] * h_inv_diag[l] * j[k][l];
                }
            }
            lambda_inv[i][k] = sum;
        }
    }
    invert_or_damped(&lambda_inv)
}

/// Compute the dynamically consistent pseudo-inverse J̄ = H⁻¹ J^T Λ.
pub fn dynamically_consistent_jacobian_pinv(
    j: &[Vec<f64>],
    h_inv_diag: &[f64],
    lambda: &[Vec<f64>],
) -> Vec<Vec<f64>> {
    let n = if !j.is_empty() { j[0].len() } else { 0 };
    let m = j.len();
    // J̄ (n×m) = H⁻¹ * J^T * Λ
    let mut j_bar = vec![vec![0.0f64; m]; n];
    for i in 0..n {
        for k in 0..m {
            let mut sum = 0.0;
            for l in 0..m {
                if i < h_inv_diag.len() {
                    sum += h_inv_diag[i] * j[l][i] * lambda[l][k];
                }
            }
            j_bar[i][k] = sum;
        }
    }
    j_bar
}

// ─────────────────────────────────────────────────────────────────────────────
// Closed-loop / loop-closure constraints
// ─────────────────────────────────────────────────────────────────────────────

/// A loop closure constraint: requires the relative transform between two links
/// to remain at a target value.
#[derive(Debug, Clone)]
pub struct LoopConstraint {
    /// Index of the first link.
    pub link_a: usize,
    /// Index of the second link.
    pub link_b: usize,
    /// Target relative position constraint error.
    pub target_pos: [f64; 3],
    /// Constraint gain (Baumgarte).
    pub gain: f64,
    /// Constraint active flag.
    pub active: bool,
}

impl LoopConstraint {
    /// Create a new loop constraint.
    pub fn new(link_a: usize, link_b: usize, target_pos: [f64; 3], gain: f64) -> Self {
        Self {
            link_a,
            link_b,
            target_pos,
            gain,
            active: true,
        }
    }

    /// Compute the position constraint error.
    pub fn position_error(&self, body: &ArticulatedBody) -> [f64; 3] {
        if self.link_a >= body.links.len() || self.link_b >= body.links.len() {
            return [0.0; 3];
        }
        let pa = body.links[self.link_a].x_world.pos;
        let pb = body.links[self.link_b].x_world.pos;
        let rel = sub3(pb, pa);
        sub3(rel, self.target_pos)
    }

    /// Compute the constraint stabilization force.
    pub fn stabilization_force(&self, body: &ArticulatedBody, dt: f64) -> [f64; 3] {
        let err = self.position_error(body);
        scale3(err, -self.gain / dt)
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Cable-driven systems
// ─────────────────────────────────────────────────────────────────────────────

/// A single cable segment in a cable-driven system.
#[derive(Debug, Clone)]
pub struct CableSegment {
    /// Index of the link at the cable attachment point A.
    pub link_a: usize,
    /// Local attachment point on link A (body frame).
    pub local_a: [f64; 3],
    /// Index of the link at the cable attachment point B.
    pub link_b: usize,
    /// Local attachment point on link B (body frame).
    pub local_b: [f64; 3],
    /// Rest length of the cable.
    pub rest_length: f64,
    /// Cable stiffness.
    pub stiffness: f64,
    /// Cable damping.
    pub damping: f64,
    /// Whether this cable can push (false = tension only).
    pub push_pull: bool,
}

impl CableSegment {
    /// Create a new cable segment.
    pub fn new(
        link_a: usize,
        local_a: [f64; 3],
        link_b: usize,
        local_b: [f64; 3],
        rest_length: f64,
        stiffness: f64,
        damping: f64,
    ) -> Self {
        Self {
            link_a,
            local_a,
            link_b,
            local_b,
            rest_length,
            stiffness,
            damping,
            push_pull: false,
        }
    }

    /// Compute the cable force (returns spatial forces on link A and link B).
    pub fn compute_force(&self, body: &ArticulatedBody) -> (SpatialVec, SpatialVec) {
        if self.link_a >= body.links.len() || self.link_b >= body.links.len() {
            return (SpatialVec::zero(), SpatialVec::zero());
        }
        let x_a = body.links[self.link_a].x_world;
        let x_b = body.links[self.link_b].x_world;
        // World positions of attachment points
        let wa = add3(x_a.pos, mat3_vec(x_a.rot, self.local_a));
        let wb = add3(x_b.pos, mat3_vec(x_b.rot, self.local_b));
        let diff = sub3(wb, wa);
        let dist = len3(diff);
        let stretch = dist - self.rest_length;
        if !self.push_pull && stretch <= 0.0 {
            return (SpatialVec::zero(), SpatialVec::zero());
        }
        let dir = if dist > 1e-12 {
            scale3(diff, 1.0 / dist)
        } else {
            [0.0; 3]
        };
        let force_mag = self.stiffness * stretch;
        let f_world = scale3(dir, force_mag);
        // Torque on A: r_A × f
        let r_a = sub3(wa, x_a.pos);
        let torque_a = cross3(r_a, f_world);
        let force_a = SpatialVec::new(torque_a, f_world);
        // Reaction on B: -f
        let r_b = sub3(wb, x_b.pos);
        let torque_b = cross3(r_b, scale3(f_world, -1.0));
        let force_b = SpatialVec::new(torque_b, scale3(f_world, -1.0));
        (force_a, force_b)
    }
}

/// A cable-driven system consisting of multiple cable segments.
#[derive(Debug, Clone)]
pub struct CableDrivenSystem {
    /// The articulated body.
    pub body: ArticulatedBody,
    /// Cable segments.
    pub cables: Vec<CableSegment>,
    /// Cable actuation (desired tension per cable, indexed same as cables).
    pub cable_tensions: Vec<f64>,
}

impl CableDrivenSystem {
    /// Create a new cable-driven system.
    pub fn new(body: ArticulatedBody) -> Self {
        Self {
            body,
            cables: Vec::new(),
            cable_tensions: Vec::new(),
        }
    }

    /// Add a cable segment.
    pub fn add_cable(&mut self, cable: CableSegment) {
        self.cable_tensions.push(0.0);
        self.cables.push(cable);
    }

    /// Apply all cable forces to the articulated body's external forces.
    pub fn apply_cable_forces(&mut self) {
        for cable in &self.cables {
            let (fa, fb) = cable.compute_force(&self.body);
            if cable.link_a < self.body.links.len() {
                let f = self.body.links[cable.link_a].f_ext;
                self.body.links[cable.link_a].f_ext = f.add(&fa);
            }
            if cable.link_b < self.body.links.len() {
                let f = self.body.links[cable.link_b].f_ext;
                self.body.links[cable.link_b].f_ext = f.add(&fb);
            }
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Hybrid dynamics (mix forward and inverse)
// ─────────────────────────────────────────────────────────────────────────────

/// Hybrid dynamics mode per joint.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum HybridMode {
    /// Solve for acceleration (forward dynamics): tau given, qdd computed.
    ForwardDynamics,
    /// Solve for torque (inverse dynamics): qdd given, tau computed.
    InverseDynamics,
}

/// Hybrid dynamics: forward dynamics for some joints, inverse dynamics for
/// the others, solved consistently as one system.
///
/// For every joint `i` the body's `links[i].tau` (forward joints) or
/// `links[i].qdd` (inverse joints) is the prescribed input.  With the
/// joint-space inertia `H` ([`ArticulatedBody::crba`]) and the bias
/// `C = rnea(q̈ = 0)` (gravity, velocity-product terms and external wrenches)
/// the equation of motion `τ = H q̈ + C` is split over the forward set `F` and
/// the inverse set `I`:
///
/// ```text
/// H_FF q̈_F = τ_F − C_F − H_FI q̈_I      (Cholesky solve)
/// τ_I       = H_IF q̈_F + H_II q̈_I + C_I
/// ```
///
/// so prescribed accelerations act on the forward joints exactly as
/// prescribed motion of the other joints does.  All-forward is
/// [`ArticulatedBody::aba`], all-inverse is [`ArticulatedBody::rnea`].  A
/// joint without a mode defaults to forward dynamics; a forward joint without
/// inertia about its motion subspace (e.g. fixed) cannot accelerate and stays
/// at `q̈ = 0`.
///
/// The result holds, per joint, the computed acceleration (forward joints) or
/// the required torque (inverse joints).  On return `links[i].qdd` holds the
/// acceleration of every joint, `links[i].tau` the torque of every inverse
/// joint, and the link velocities / accelerations are consistent with them.
///
/// # Errors
///
/// [`Error::SingularMatrix`] when the forward block `H_FF` is not positive
/// definite (e.g. two forward joints driving the same motion).
pub fn hybrid_dynamics(body: &mut ArticulatedBody, modes: &[HybridMode]) -> Result<Vec<f64>> {
    let n = body.links.len();
    let tau_in: Vec<f64> = body.links.iter().map(|l| l.tau).collect();
    let mut qdd: Vec<f64> = body.links.iter().map(|l| l.qdd).collect();
    let mode_of = |i: usize| modes.get(i).copied().unwrap_or(HybridMode::ForwardDynamics);

    let h = body.crba();
    let bias = body.rnea(&vec![0.0; n], None);

    // Forward joints that can actually move; the other forward joints are locked.
    let mut solve_for = vec![false; n];
    for i in 0..n {
        if mode_of(i) == HybridMode::ForwardDynamics {
            if h[i][i].abs() < MIN_JOINT_INERTIA {
                qdd[i] = 0.0;
            } else {
                solve_for[i] = true;
            }
        }
    }
    let free: Vec<usize> = (0..n).filter(|&i| solve_for[i]).collect();
    let m = free.len();

    if m > 0 {
        let mut h_ff = vec![0.0f64; m * m];
        let mut rhs = vec![0.0f64; m];
        for (a, &i) in free.iter().enumerate() {
            let mut r = tau_in[i] - bias[i];
            for (b, &j) in free.iter().enumerate() {
                h_ff[a * m + b] = h[i][j];
            }
            for (j, &q_dd_j) in qdd.iter().enumerate() {
                if !solve_for[j] {
                    r -= h[i][j] * q_dd_j;
                }
            }
            rhs[a] = r;
        }
        let factor = cholesky_factor(&h_ff, m).ok_or_else(|| Error::singular(f64::INFINITY))?;
        let solution = cholesky_solve(&factor, m, &rhs, 1);
        for (&i, &value) in free.iter().zip(solution.iter()) {
            qdd[i] = value;
        }
    }

    // Torques for the complete acceleration vector (this also leaves the
    // link velocities and accelerations consistent with the solution).
    let tau_out = body.rnea(&qdd, None);

    let mut result = vec![0.0f64; n];
    for i in 0..n {
        body.links[i].qdd = qdd[i];
        if mode_of(i) == HybridMode::ForwardDynamics {
            result[i] = qdd[i];
        } else {
            body.links[i].tau = tau_out[i];
            result[i] = tau_out[i];
        }
    }
    Ok(result)
}

// ─────────────────────────────────────────────────────────────────────────────
// Redundant DOF handling
// ─────────────────────────────────────────────────────────────────────────────

/// Compute the minimum-norm pseudo-inverse of a matrix (simple version for fat matrices).
///
/// For J (m×n) with m < n, returns J⁺ = J^T * (J * J^T)^{-1}, with the m×m
/// matrix `J J^T` inverted exactly for every m.  When `J J^T` is singular (J
/// is rank deficient) the damped-least-squares inverse `(J J^T + μ² I)⁻¹` is
/// used instead, which keeps `J⁺` bounded near kinematic singularities.
pub fn pseudo_inverse_fat(j: &[Vec<f64>]) -> Vec<Vec<f64>> {
    let m = j.len();
    let n = j.iter().map(Vec::len).min().unwrap_or(0);
    if m == 0 || n == 0 {
        return vec![];
    }

    // J * J^T (m×m)
    let mut jjt = vec![vec![0.0f64; m]; m];
    for (i, jjt_row) in jjt.iter_mut().enumerate() {
        for (k, jjt_ik) in jjt_row.iter_mut().enumerate() {
            *jjt_ik = j[i]
                .iter()
                .take(n)
                .zip(j[k].iter().take(n))
                .map(|(a, b)| a * b)
                .sum();
        }
    }

    let jjt_inv = invert_or_damped(&jjt);

    // J⁺ = J^T * (J*J^T)^{-1}  (n×m)
    let mut jt = vec![vec![0.0f64; m]; n];
    for i in 0..n {
        for k in 0..m {
            for l in 0..m {
                jt[i][k] += j[l][i] * jjt_inv[l][k];
            }
        }
    }
    jt
}

/// Invert the square matrix `a` by Gauss-Jordan elimination with partial
/// pivoting.
///
/// Returns `None` when `a` is not square, contains non-finite entries, or is
/// singular: a pivot smaller than [`SINGULAR_PIVOT_TOL`] times
/// `max(1, max|a_ij|)` counts as zero.  The inverse of an empty matrix is
/// empty.
fn invert_dense(a: &[Vec<f64>]) -> Option<Vec<Vec<f64>>> {
    let m = a.len();
    if a.iter().any(|row| row.len() != m) {
        return None;
    }
    if m == 0 {
        return Some(Vec::new());
    }
    let scale = a.iter().flatten().fold(0.0_f64, |acc, v| acc.max(v.abs()));
    if !scale.is_finite() {
        return None;
    }
    let tol = SINGULAR_PIVOT_TOL * scale.max(1.0);

    let mut work: Vec<Vec<f64>> = a.to_vec();
    let mut inv: Vec<Vec<f64>> = (0..m)
        .map(|i| (0..m).map(|k| if i == k { 1.0 } else { 0.0 }).collect())
        .collect();
    for col in 0..m {
        // Partial pivoting: the largest remaining entry of this column.
        let pivot_row =
            (col..m).max_by(|&r1, &r2| work[r1][col].abs().total_cmp(&work[r2][col].abs()))?;
        let pivot = work[pivot_row][col];
        if !pivot.is_finite() || pivot.abs() < tol {
            return None;
        }
        work.swap(col, pivot_row);
        inv.swap(col, pivot_row);
        let inv_pivot = 1.0 / pivot;
        for value in work[col].iter_mut().chain(inv[col].iter_mut()) {
            *value *= inv_pivot;
        }
        let work_pivot_row = work[col].clone();
        let inv_pivot_row = inv[col].clone();
        for row in 0..m {
            if row == col {
                continue;
            }
            let factor = work[row][col];
            if factor == 0.0 {
                continue;
            }
            for (w, p) in work[row].iter_mut().zip(work_pivot_row.iter()) {
                *w -= factor * p;
            }
            for (v, p) in inv[row].iter_mut().zip(inv_pivot_row.iter()) {
                *v -= factor * p;
            }
        }
    }
    Some(inv)
}

/// Damped-least-squares inverse `(M + μ² I)⁻¹` of a (nearly) singular square
/// matrix, with `μ = `[`DLS_DAMPING`].
///
/// For a symmetric positive semi-definite `M` the damped matrix is positive
/// definite, so the inversion succeeds; if `μ²` is still too small next to a
/// very large `M` for the pivot test of [`invert_dense`], the damping is
/// raised by a factor of 10³ per attempt.  A matrix with non-finite entries
/// yields a NaN matrix rather than a plausible-looking wrong answer.
fn damped_inverse(a: &[Vec<f64>]) -> Vec<Vec<f64>> {
    let m = a.len();
    let mut damping = DLS_DAMPING * DLS_DAMPING;
    for _ in 0..DLS_MAX_ESCALATIONS {
        let mut damped = a.to_vec();
        for (i, row) in damped.iter_mut().enumerate() {
            if let Some(diag) = row.get_mut(i) {
                *diag += damping;
            }
        }
        if let Some(inv) = invert_dense(&damped) {
            return inv;
        }
        damping *= 1.0e3;
    }
    vec![vec![f64::NAN; m]; m]
}

/// Exact inverse of `a`, or its damped-least-squares inverse when `a` is
/// singular.
fn invert_or_damped(a: &[Vec<f64>]) -> Vec<Vec<f64>> {
    invert_dense(a).unwrap_or_else(|| damped_inverse(a))
}

/// Compute the null-space projector N = I - J⁺ * J.
pub fn null_space_projector(j_pinv: &[Vec<f64>], j: &[Vec<f64>]) -> Vec<Vec<f64>> {
    let n = j_pinv.len();
    let m = j.len();
    if n == 0 || m == 0 {
        return vec![];
    }
    // N = I - J⁺ J  (n×n)
    let mut n_proj = vec![vec![0.0f64; n]; n];
    for i in 0..n {
        for k in 0..n {
            let delta = if i == k { 1.0 } else { 0.0 };
            let mut sum = 0.0;
            for l in 0..m {
                if l < j_pinv[i].len() && k < j[l].len() {
                    sum += j_pinv[i][l] * j[l][k];
                }
            }
            n_proj[i][k] = delta - sum;
        }
    }
    n_proj
}

// ─────────────────────────────────────────────────────────────────────────────
// Euler integration for articulated bodies
// ─────────────────────────────────────────────────────────────────────────────

/// Semi-implicit Euler integration of an articulated body.
pub fn euler_integrate(body: &mut ArticulatedBody, dt: f64) {
    let qdd = body.aba(&body.links.iter().map(|l| l.tau).collect::<Vec<_>>());
    for (i, link) in body.links.iter_mut().enumerate() {
        let acc = if i < qdd.len() { qdd[i] } else { 0.0 };
        link.qd += acc * dt;
        link.q += link.qd * dt;
    }
}

/// Symplectic Euler (Störmer-Verlet like) integration for stability.
pub fn symplectic_euler_integrate(body: &mut ArticulatedBody, dt: f64) {
    let tau: Vec<f64> = body.links.iter().map(|l| l.tau).collect();
    let qdd = body.aba(&tau);
    for (i, link) in body.links.iter_mut().enumerate() {
        let acc = if i < qdd.len() { qdd[i] } else { 0.0 };
        link.qd += acc * dt; // update velocity first
        link.q += link.qd * dt; // use updated velocity
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Helper: build a simple serial chain
// ─────────────────────────────────────────────────────────────────────────────

/// Build a serial chain of n revolute joints (all about the Z axis).
pub fn build_serial_chain(
    n: usize,
    link_length: f64,
    mass: f64,
    gravity: [f64; 3],
) -> ArticulatedBody {
    let mut body = ArticulatedBody::new(gravity);
    let joint = JointAxis::revolute([0.0, 0.0, 1.0]);
    let x_t = SpatialTransform::translation([link_length, 0.0, 0.0]);
    for i in 0..n {
        let parent = if i == 0 { -1i64 } else { (i as i64) - 1 };
        let inertia = RigidBodyInertia::box_shape(mass, link_length / 2.0, 0.05, 0.05);
        let link = Link::new(format!("link_{i}"), parent, inertia, joint, x_t);
        body.add_link(link);
    }
    body
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod regression_tests;
#[cfg(test)]
mod tests;
