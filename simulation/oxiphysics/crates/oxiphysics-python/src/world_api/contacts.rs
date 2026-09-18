// Copyright 2026 COOLJAPAN OU (Team KitaSan)
// SPDX-License-Identifier: Apache-2.0

//! Contact detection, contact response and ray casting for [`PyPhysicsWorld`].
//!
//! Geometry and collision detection are delegated to the engine. Every
//! [`PyColliderShape`] attached to a body is converted into its
//! `oxiphysics::geometry` counterpart placed at the body pose (position and
//! orientation). Candidate pairs come from the `oxiphysics::collision`
//! sweep-and-prune broadphase over the world-space bounding boxes, and contact
//! manifolds come from its [`NarrowPhaseDispatcher`] (analytic sphere / box /
//! capsule algorithms, face-clipped box manifolds and GJK/EPA for every other
//! convex pair). Infinite planes have no engine shape: they are collided against
//! the features of the partner shape (sphere centre, box corners, capsule
//! segment ends, cylinder / cone rims) here.
//!
//! The response is a sequential-impulse solver run for
//! [`PySimConfig::solver_iterations`] sweeps over every contact point with
//! accumulated (clamped) impulses:
//!
//! * the normal impulse is non-negative and targets `-e * v_n` for pairs that
//!   approach faster than the restitution threshold (zero otherwise, so resting
//!   contacts do not bounce);
//! * Coulomb friction acts in the tangent plane, clamped to `mu * j_n`;
//! * lever arms and world-space inverse inertia tensors (from the engine shape's
//!   `inertia_tensor`) let contacts spin bodies (a sliding sphere starts to roll).
//!
//! Friction and restitution coefficients of the two bodies are combined with
//! [`PySimConfig::friction_mixing`] / [`PySimConfig::restitution_mixing`].
//! Penetration is removed by a mass-weighted positional (Baumgarte) correction
//! applied once per body pair: `c = beta * max(depth - slop, 0) / (1/m_a + 1/m_b)`,
//! body A moves by `-c / m_a` and body B by `+c / m_b` along the A→B normal,
//! where `beta` is [`PySimConfig::baumgarte_factor`] and `slop` is
//! [`PySimConfig::max_penetration`].

use super::{InternalBody, PyPhysicsWorld, approximate_inertia};
use crate::types::{PyColliderShape, PyContactResult, PySimConfig};
use oxiphysics::collision::narrowphase::dispatch::ShapeType;
use oxiphysics::collision::types::CollisionPair;
use oxiphysics::collision::{BroadPhase, NarrowPhaseDispatcher, SweepAndPrune};
use oxiphysics::core::math::{Mat3, Quat, Real, Vec3};
use oxiphysics::core::{Aabb, Transform};
use oxiphysics::geometry::{BoxShape, Capsule, Cone, Cylinder, Shape, Sphere};
use std::collections::BTreeMap;

/// Lowest approach speed (m/s) that ever triggers a restitution bounce.
///
/// The effective threshold is `max(2 |g| dt, MIN_RESTITUTION_SPEED)`: a body
/// resting on another approaches it by about `|g| dt` per step (the velocity
/// gravity adds during integration), which must not be turned into a bounce.
const MIN_RESTITUTION_SPEED: Real = 1e-3;

/// Effective-mass denominators below this are treated as "no response".
const MIN_EFFECTIVE_MASS_DENOM: Real = 1e-12;

// ---------------------------------------------------------------------------
// Coefficient mixing
// ---------------------------------------------------------------------------

/// How the friction / restitution coefficients of two bodies are combined.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum MixRule {
    /// Arithmetic mean `(a + b) / 2`.
    Average,
    /// Smaller coefficient.
    Min,
    /// Larger coefficient.
    Max,
    /// Product `a * b`.
    Multiply,
}

impl MixRule {
    /// Parse a mixing-rule name (case-insensitive).
    ///
    /// Accepts `"average"`/`"mean"`, `"min"`/`"minimum"`, `"max"`/`"maximum"`
    /// and `"multiply"`/`"product"`; returns `None` for anything else.
    pub(super) fn parse(name: &str) -> Option<Self> {
        match name.trim().to_ascii_lowercase().as_str() {
            "average" | "mean" => Some(Self::Average),
            "min" | "minimum" => Some(Self::Min),
            "max" | "maximum" => Some(Self::Max),
            "multiply" | "product" => Some(Self::Multiply),
            _ => None,
        }
    }

    /// Rule configured by `name`; an unrecognised name uses [`MixRule::Average`]
    /// (the documented default of [`PySimConfig`]).
    fn from_config(name: &str) -> Self {
        Self::parse(name).unwrap_or(Self::Average)
    }

    /// Combine two coefficients.
    pub(super) fn combine(self, a: Real, b: Real) -> Real {
        match self {
            Self::Average => 0.5 * (a + b),
            Self::Min => a.min(b),
            Self::Max => a.max(b),
            Self::Multiply => a * b,
        }
    }
}

// ---------------------------------------------------------------------------
// Shape conversion
// ---------------------------------------------------------------------------

/// World pose of a body (normalised orientation; identity if degenerate).
fn body_transform(body: &InternalBody) -> Transform {
    let p = body.position;
    let q = body.orientation;
    let raw = nalgebra::Quaternion::new(q[3], q[0], q[1], q[2]);
    let rotation = if q.iter().all(|v| v.is_finite()) && raw.norm() > 1e-12 {
        Quat::from_quaternion(raw)
    } else {
        Quat::identity()
    };
    Transform::new(Vec3::new(p[0], p[1], p[2]), rotation)
}

/// Engine convex shape for a collider, or `None` for planes and for shapes
/// with non-finite or non-positive dimensions (which cannot collide).
fn convex_shape(shape: &PyColliderShape) -> Option<(Box<dyn Shape>, ShapeType)> {
    let positive = |v: Real| v.is_finite() && v > 0.0;
    match *shape {
        PyColliderShape::Sphere { radius } if positive(radius) => {
            Some((Box::new(Sphere::new(radius)), ShapeType::Sphere))
        }
        PyColliderShape::Box { half_extents } if half_extents.iter().all(|&h| positive(h)) => {
            let half = Vec3::new(half_extents[0], half_extents[1], half_extents[2]);
            Some((Box::new(BoxShape::new(half)), ShapeType::Box))
        }
        PyColliderShape::Capsule {
            radius,
            half_height,
        } if positive(radius) && half_height.is_finite() && half_height >= 0.0 => Some((
            Box::new(Capsule::new(radius, half_height)),
            ShapeType::Capsule,
        )),
        PyColliderShape::Cylinder {
            radius,
            half_height,
        } if positive(radius) && positive(half_height) => Some((
            Box::new(Cylinder::new(radius, half_height)),
            ShapeType::Cylinder,
        )),
        PyColliderShape::Cone {
            radius,
            half_height,
        } if positive(radius) && positive(half_height) => {
            Some((Box::new(Cone::new(radius, half_height)), ShapeType::Cone))
        }
        _ => None,
    }
}

/// World-space plane `n · x = offset` of a plane collider attached to a body
/// with pose `tf`, or `None` for non-plane / degenerate colliders.
///
/// The collider's `normal` and `distance` are expressed in the body frame.
fn world_plane(shape: &PyColliderShape, tf: &Transform) -> Option<(Vec3, Real)> {
    let PyColliderShape::Plane { normal, distance } = *shape else {
        return None;
    };
    let local = Vec3::new(normal[0], normal[1], normal[2]);
    let len = local.norm();
    if !(len.is_finite() && len > 1e-12 && distance.is_finite()) {
        return None;
    }
    let n = tf.rotation * (local / len);
    Some((n, distance + n.dot(&tf.position)))
}

/// World-space AABB of `shape` placed at `tf` (centre transformed, half-extents
/// mapped through `|R|`).
fn world_aabb(shape: &dyn Shape, tf: &Transform) -> Aabb {
    let local = shape.bounding_box();
    let centre = (local.min + local.max) * 0.5;
    let half = (local.max - local.min) * 0.5;
    let abs_rot = tf.rotation.to_rotation_matrix().matrix().abs();
    let world_centre = tf.transform_point(&centre);
    let world_half = abs_rot * half;
    Aabb::new(world_centre - world_half, world_centre + world_half)
}

/// Geometry of one collider leaf.
enum LeafGeometry {
    /// A convex engine shape.
    Convex {
        /// Engine shape (body frame).
        shape: Box<dyn Shape>,
        /// Dispatcher classification.
        kind: ShapeType,
        /// World-space bounding box.
        aabb: Aabb,
        /// Source collider (used for the plane feature points).
        source: PyColliderShape,
    },
    /// An infinite half-space `n · x <= offset` (world frame).
    Plane {
        /// Unit outward normal.
        normal: Vec3,
        /// Plane offset along the normal.
        offset: Real,
    },
}

/// One collider of a body, posed for the current step.
struct Leaf {
    /// Slot index of the owning body.
    body: usize,
    /// Body pose.
    world: Transform,
    /// Collider geometry.
    geometry: LeafGeometry,
}

/// Build the leaf of `shape` attached to body `body` at pose `world`.
fn make_leaf(body: usize, world: &Transform, shape: &PyColliderShape) -> Option<Leaf> {
    let geometry = if let Some((normal, offset)) = world_plane(shape, world) {
        LeafGeometry::Plane { normal, offset }
    } else {
        let (engine, kind) = convex_shape(shape)?;
        let aabb = world_aabb(engine.as_ref(), world);
        LeafGeometry::Convex {
            shape: engine,
            kind,
            aabb,
            source: shape.clone(),
        }
    };
    Some(Leaf {
        body,
        world: world.clone(),
        geometry,
    })
}

/// Collect the leaves of every live body.
fn gather_leaves(world: &PyPhysicsWorld) -> Vec<Leaf> {
    let mut leaves = Vec::new();
    for (index, slot) in world.slots.iter().enumerate() {
        let Some(body) = slot.body.as_ref() else {
            continue;
        };
        let tf = body_transform(body);
        leaves.extend(
            body.shapes
                .iter()
                .filter_map(|shape| make_leaf(index, &tf, shape)),
        );
    }
    leaves
}

// ---------------------------------------------------------------------------
// Narrowphase
// ---------------------------------------------------------------------------

/// One contact point of a body pair, oriented from the lower slot index (A)
/// to the higher one (B).
#[derive(Debug, Clone, Copy)]
struct RawPoint {
    /// World-space contact point (midway between the two surfaces).
    point: Vec3,
    /// Unit normal from body A toward body B.
    normal: Vec3,
    /// Penetration depth (> 0).
    depth: Real,
}

/// Feature points of a convex collider that can touch a plane: the points
/// whose depth below the plane is checked.
fn plane_feature_points(
    source: &PyColliderShape,
    tf: &Transform,
    plane_normal: &Vec3,
) -> Vec<Vec3> {
    let axis = tf.rotation * Vec3::y();
    match *source {
        PyColliderShape::Sphere { radius } => vec![tf.position - plane_normal * radius],
        PyColliderShape::Box { half_extents } => {
            let mut corners = Vec::with_capacity(8);
            for sx in [-1.0, 1.0] {
                for sy in [-1.0, 1.0] {
                    for sz in [-1.0, 1.0] {
                        let local = Vec3::new(
                            sx * half_extents[0],
                            sy * half_extents[1],
                            sz * half_extents[2],
                        );
                        corners.push(tf.transform_point(&local));
                    }
                }
            }
            corners
        }
        PyColliderShape::Capsule {
            radius,
            half_height,
        } => [-1.0, 1.0]
            .iter()
            .map(|s| tf.position + axis * (s * half_height) - plane_normal * radius)
            .collect(),
        PyColliderShape::Cylinder {
            radius,
            half_height,
        } => {
            let mut points = rim_points(
                tf.position - axis * half_height,
                &axis,
                radius,
                plane_normal,
            );
            points.extend(rim_points(
                tf.position + axis * half_height,
                &axis,
                radius,
                plane_normal,
            ));
            points
        }
        PyColliderShape::Cone {
            radius,
            half_height,
        } => {
            let mut points = rim_points(
                tf.position - axis * half_height,
                &axis,
                radius,
                plane_normal,
            );
            points.push(tf.position + axis * half_height);
            points
        }
        PyColliderShape::Plane { .. } => Vec::new(),
    }
}

/// Four points of the circle (centre `c`, unit axis `axis`, radius `r`): the
/// one deepest along `-plane_normal`, its opposite and the two at 90°. A flat
/// disc resting on the plane therefore gets a 4-point patch, a tilted disc its
/// lowest rim point.
fn rim_points(c: Vec3, axis: &Vec3, r: Real, plane_normal: &Vec3) -> Vec<Vec3> {
    let down = -plane_normal;
    let in_plane = down - axis * axis.dot(&down);
    let u = if in_plane.norm() > 1e-9 {
        in_plane.normalize()
    } else {
        any_perpendicular(axis)
    };
    let w = axis.cross(&u);
    vec![c + u * r, c - u * r, c + w * r, c - w * r]
}

/// A unit vector perpendicular to the unit vector `n`.
fn any_perpendicular(n: &Vec3) -> Vec3 {
    let helper = if n.x.abs() < 0.57 {
        Vec3::x()
    } else if n.y.abs() < 0.57 {
        Vec3::y()
    } else {
        Vec3::z()
    };
    n.cross(&helper).normalize()
}

/// Contacts of a convex leaf against the half-space `n · x <= offset`.
///
/// Returned normals point from the plane toward the convex body.
fn plane_convex_points(
    normal: &Vec3,
    offset: Real,
    source: &PyColliderShape,
    tf: &Transform,
) -> Vec<RawPoint> {
    plane_feature_points(source, tf, normal)
        .into_iter()
        .filter_map(|p| {
            let depth = offset - normal.dot(&p);
            (depth.is_finite() && depth > 0.0).then(|| RawPoint {
                point: p + normal * (0.5 * depth),
                normal: *normal,
                depth,
            })
        })
        .collect()
}

/// Contacts of two convex leaves through the engine dispatcher, oriented A→B.
fn convex_convex_points(
    dispatcher: &NarrowPhaseDispatcher,
    a: (&dyn Shape, ShapeType, &Transform),
    b: (&dyn Shape, ShapeType, &Transform),
    pair: CollisionPair,
) -> Vec<RawPoint> {
    let result = dispatcher.dispatch(a.0, a.1, a.2, b.0, b.1, b.2, pair);
    let Some(manifold) = result.manifold else {
        return Vec::new();
    };
    manifold
        .contacts
        .iter()
        .filter_map(|c| {
            let len = c.normal.norm();
            let usable = len.is_finite() && len > 0.5 && c.depth.is_finite() && c.depth > 0.0;
            // The engine normal points from B toward A.
            usable.then(|| RawPoint {
                point: (c.point_a + c.point_b) * 0.5,
                normal: -c.normal / len,
                depth: c.depth,
            })
        })
        .collect()
}

/// Contacts of the two end spheres of a capsule (radius `radius`, segment
/// half-length `half_height` along the local Y axis, pose `tf`), each produced
/// by `collide` with the end sphere as the capsule operand.
fn capsule_end_points<F>(
    radius: Real,
    half_height: Real,
    tf: &Transform,
    mut collide: F,
) -> Vec<RawPoint>
where
    F: FnMut((&dyn Shape, ShapeType, &Transform)) -> Vec<RawPoint>,
{
    if half_height <= 0.0 {
        return Vec::new();
    }
    let axis = tf.rotation * Vec3::y();
    let end_sphere = Sphere::new(radius);
    let mut out = Vec::new();
    for sign in [-1.0, 1.0] {
        let end_tf = Transform::new(tf.position + axis * (sign * half_height), tf.rotation);
        out.extend(collide((&end_sphere, ShapeType::Sphere, &end_tf)));
    }
    out
}

/// Append the points of `extra` that are not within `0.1 * scale` of a point
/// already in `points`.
fn merge_points(points: &mut Vec<RawPoint>, extra: Vec<RawPoint>, scale: Real) {
    let tolerance = 0.1 * scale;
    for candidate in extra {
        let duplicate = points
            .iter()
            .any(|p| (p.point - candidate.point).norm() < tolerance);
        if !duplicate {
            points.push(candidate);
        }
    }
}

/// Whether the pair of bodies needs a contact query at all: at least one of
/// them must be able to move (an awake dynamic body, or a kinematic body that
/// may push a sleeping one), and they must not both be immovable.
fn pair_is_relevant(a: &InternalBody, b: &InternalBody) -> bool {
    let awake_dynamic = |x: &InternalBody| x.inv_mass() > 0.0 && !x.is_sleeping;
    let pusher = |x: &InternalBody| x.is_kinematic;
    let sleeper = |x: &InternalBody| x.inv_mass() > 0.0 && x.is_sleeping;
    if a.inv_mass() <= 0.0 && b.inv_mass() <= 0.0 {
        return false;
    }
    awake_dynamic(a) || awake_dynamic(b) || (pusher(a) && sleeper(b)) || (pusher(b) && sleeper(a))
}

/// Axis (0, 1 or 2) along which the AABB centres spread the most.
fn sweep_axis(aabbs: &[Aabb]) -> usize {
    if aabbs.is_empty() {
        return 0;
    }
    let count = aabbs.len() as Real;
    let mut mean = Vec3::zeros();
    for b in aabbs {
        mean += (b.min + b.max) * 0.5;
    }
    mean /= count;
    let mut var = Vec3::zeros();
    for b in aabbs {
        let d = (b.min + b.max) * 0.5 - mean;
        var += d.component_mul(&d);
    }
    if var.x >= var.y && var.x >= var.z {
        0
    } else if var.y >= var.z {
        1
    } else {
        2
    }
}

/// Whether a box intersects the half-space `n · x <= offset`.
fn aabb_below_plane(aabb: &Aabb, normal: &Vec3, offset: Real) -> bool {
    // Corner of the box that lies furthest along -n.
    let corner = Vec3::new(
        if normal.x >= 0.0 {
            aabb.min.x
        } else {
            aabb.max.x
        },
        if normal.y >= 0.0 {
            aabb.min.y
        } else {
            aabb.max.y
        },
        if normal.z >= 0.0 {
            aabb.min.z
        } else {
            aabb.max.z
        },
    );
    normal.dot(&corner) < offset
}

/// Detect every contact of the world, grouped per body pair `(a, b)` with
/// `a < b` (slot indices), normals pointing from `a` to `b`.
fn detect_manifolds(world: &PyPhysicsWorld) -> BTreeMap<(usize, usize), Vec<RawPoint>> {
    let leaves = gather_leaves(world);
    let body_of = |slot: usize| world.slots.get(slot).and_then(|s| s.body.as_ref());
    let relevant = |a: usize, b: usize| match (body_of(a), body_of(b)) {
        (Some(x), Some(y)) => a != b && pair_is_relevant(x, y),
        _ => false,
    };

    let convex: Vec<usize> = leaves
        .iter()
        .enumerate()
        .filter(|(_, l)| matches!(l.geometry, LeafGeometry::Convex { .. }))
        .map(|(i, _)| i)
        .collect();
    let planes: Vec<usize> = leaves
        .iter()
        .enumerate()
        .filter(|(_, l)| matches!(l.geometry, LeafGeometry::Plane { .. }))
        .map(|(i, _)| i)
        .collect();

    let mut manifolds: BTreeMap<(usize, usize), Vec<RawPoint>> = BTreeMap::new();

    // Convex-convex: engine broadphase + engine narrowphase.
    let aabbs: Vec<Aabb> = convex
        .iter()
        .filter_map(|&i| match &leaves[i].geometry {
            LeafGeometry::Convex { aabb, .. } => Some(aabb.clone()),
            LeafGeometry::Plane { .. } => None,
        })
        .collect();
    let dispatcher = NarrowPhaseDispatcher::default();
    let candidates = SweepAndPrune::new(sweep_axis(&aabbs)).find_pairs(&aabbs);
    for candidate in candidates {
        let (Some(&ia), Some(&ib)) = (convex.get(candidate.a), convex.get(candidate.b)) else {
            continue;
        };
        let (mut la, mut lb) = (&leaves[ia], &leaves[ib]);
        if la.body > lb.body {
            std::mem::swap(&mut la, &mut lb);
        }
        if !relevant(la.body, lb.body) {
            continue;
        }
        let (
            LeafGeometry::Convex {
                shape: sa,
                kind: ka,
                source: source_a,
                ..
            },
            LeafGeometry::Convex {
                shape: sb,
                kind: kb,
                source: source_b,
                ..
            },
        ) = (&la.geometry, &lb.geometry)
        else {
            continue;
        };
        let pair = CollisionPair::new(la.body, lb.body);
        let op_a = (sa.as_ref(), *ka, &la.world);
        let op_b = (sb.as_ref(), *kb, &lb.world);
        let mut points = convex_convex_points(&dispatcher, op_a, op_b, pair);
        // The engine's capsule algorithms report one deepest point; add the
        // contacts of the capsule's end spheres so a lying capsule gets a
        // two-point support instead of see-sawing on a single point.
        if let PyColliderShape::Capsule {
            radius,
            half_height,
        } = *source_a
        {
            let ends = capsule_end_points(radius, half_height, &la.world, |end| {
                convex_convex_points(&dispatcher, end, op_b, pair)
            });
            merge_points(&mut points, ends, radius);
        }
        if let PyColliderShape::Capsule {
            radius,
            half_height,
        } = *source_b
        {
            let ends = capsule_end_points(radius, half_height, &lb.world, |end| {
                convex_convex_points(&dispatcher, op_a, end, pair)
            });
            merge_points(&mut points, ends, radius);
        }
        if !points.is_empty() {
            manifolds
                .entry((la.body, lb.body))
                .or_default()
                .extend(points);
        }
    }

    // Plane-convex: the plane is unbounded, so it is tested against every
    // convex leaf whose box reaches below it.
    for &ip in &planes {
        let plane_leaf = &leaves[ip];
        let LeafGeometry::Plane { normal, offset } = plane_leaf.geometry else {
            continue;
        };
        for &ic in &convex {
            let convex_leaf = &leaves[ic];
            if !relevant(plane_leaf.body, convex_leaf.body) {
                continue;
            }
            let LeafGeometry::Convex { aabb, source, .. } = &convex_leaf.geometry else {
                continue;
            };
            if !aabb_below_plane(aabb, &normal, offset) {
                continue;
            }
            let mut points = plane_convex_points(&normal, offset, source, &convex_leaf.world);
            if points.is_empty() {
                continue;
            }
            let key = if plane_leaf.body < convex_leaf.body {
                (plane_leaf.body, convex_leaf.body)
            } else {
                // The convex body is A: flip the plane→convex normals.
                for p in &mut points {
                    p.normal = -p.normal;
                }
                (convex_leaf.body, plane_leaf.body)
            };
            manifolds.entry(key).or_default().extend(points);
        }
    }

    manifolds
}

// ---------------------------------------------------------------------------
// Solver
// ---------------------------------------------------------------------------

/// Velocity state of one body inside the contact solver.
#[derive(Debug, Clone)]
struct SolverBody {
    /// Inverse mass (0 for static, kinematic, sleeping or massless bodies).
    inv_mass: Real,
    /// World-space inverse inertia tensor (zero when `inv_mass` is 0).
    inv_inertia: Mat3,
    /// Linear velocity.
    velocity: Vec3,
    /// Angular velocity.
    angular_velocity: Vec3,
}

impl SolverBody {
    /// Solver state of `body` (sleeping bodies act as immovable).
    fn from_body(body: &InternalBody) -> Self {
        let movable = body.inv_mass() > 0.0 && !body.is_sleeping;
        let inv_mass = if movable { body.inv_mass() } else { 0.0 };
        let inv_inertia = if movable {
            world_inverse_inertia(body)
        } else {
            Mat3::zeros()
        };
        let v = body.velocity;
        let w = body.angular_velocity;
        Self {
            inv_mass,
            inv_inertia,
            velocity: Vec3::new(v[0], v[1], v[2]),
            angular_velocity: Vec3::new(w[0], w[1], w[2]),
        }
    }

    /// Velocity of the material point at lever arm `r`.
    fn point_velocity(&self, r: &Vec3) -> Vec3 {
        self.velocity + self.angular_velocity.cross(r)
    }
}

/// World-space inverse inertia tensor of a dynamic body.
///
/// Uses the engine's inertia tensor of the body's first convex collider (about
/// the body centre); bodies without a convex collider use the isotropic
/// estimate of [`approximate_inertia`]. A singular tensor gives no rotational
/// response.
fn world_inverse_inertia(body: &InternalBody) -> Mat3 {
    let first_convex = body.shapes.iter().find_map(convex_shape);
    let local_inverse = match first_convex {
        Some((shape, _)) => shape
            .inertia_tensor(body.mass)
            .try_inverse()
            .unwrap_or_else(Mat3::zeros),
        None => {
            let i = approximate_inertia(body);
            if i > 1e-12 {
                Mat3::identity() / i
            } else {
                Mat3::zeros()
            }
        }
    };
    let rot = body_transform(body).rotation.to_rotation_matrix();
    let r = rot.matrix();
    r * local_inverse * r.transpose()
}

/// One contact point prepared for the sequential-impulse sweeps.
#[derive(Debug, Clone)]
struct ContactRow {
    /// Slot index of body A.
    a: usize,
    /// Slot index of body B.
    b: usize,
    /// Contact point (world).
    point: Vec3,
    /// Lever arm from the centre of A.
    r_a: Vec3,
    /// Lever arm from the centre of B.
    r_b: Vec3,
    /// Unit normal A→B.
    normal: Vec3,
    /// Orthonormal tangent basis of the contact plane.
    tangents: [Vec3; 2],
    /// Effective mass along the normal.
    normal_mass: Real,
    /// Effective masses along the tangents.
    tangent_mass: [Real; 2],
    /// Target normal relative velocity (restitution bounce, or 0).
    bounce: Real,
    /// Combined friction coefficient.
    friction: Real,
    /// Combined restitution coefficient.
    restitution: Real,
    /// Penetration depth.
    depth: Real,
    /// Accumulated normal impulse.
    normal_impulse: Real,
    /// Accumulated tangent impulses.
    tangent_impulse: [Real; 2],
}

/// Effective mass `1 / (1/m_a + 1/m_b + ((I_a⁻¹ (r_a×d))×r_a + (I_b⁻¹ (r_b×d))×r_b)·d)`.
fn effective_mass(a: &SolverBody, b: &SolverBody, r_a: &Vec3, r_b: &Vec3, dir: &Vec3) -> Real {
    let ang_a = (a.inv_inertia * r_a.cross(dir)).cross(r_a);
    let ang_b = (b.inv_inertia * r_b.cross(dir)).cross(r_b);
    let k = a.inv_mass + b.inv_mass + dir.dot(&(ang_a + ang_b));
    if k.is_finite() && k > MIN_EFFECTIVE_MASS_DENOM {
        1.0 / k
    } else {
        0.0
    }
}

/// Relative velocity of B with respect to A at the contact point.
fn relative_velocity(bodies: &[SolverBody], row: &ContactRow) -> Vec3 {
    bodies[row.b].point_velocity(&row.r_b) - bodies[row.a].point_velocity(&row.r_a)
}

/// Apply impulse `p` to B and `-p` to A at the row's contact point.
fn apply_impulse(bodies: &mut [SolverBody], row: &ContactRow, p: &Vec3) {
    if let Some(a) = bodies.get_mut(row.a) {
        a.velocity -= p * a.inv_mass;
        a.angular_velocity -= a.inv_inertia * row.r_a.cross(p);
    }
    if let Some(b) = bodies.get_mut(row.b) {
        b.velocity += p * b.inv_mass;
        b.angular_velocity += b.inv_inertia * row.r_b.cross(p);
    }
}

/// One sequential-impulse sweep over a contact row: friction (clamped to the
/// Coulomb disc of radius `mu * j_n`) followed by the non-negative normal
/// impulse.
fn solve_row(bodies: &mut [SolverBody], row: &mut ContactRow) {
    // Friction.
    let v = relative_velocity(bodies, row);
    let mut target = [
        row.tangent_impulse[0] - row.tangent_mass[0] * v.dot(&row.tangents[0]),
        row.tangent_impulse[1] - row.tangent_mass[1] * v.dot(&row.tangents[1]),
    ];
    let limit = row.friction * row.normal_impulse;
    let magnitude = (target[0] * target[0] + target[1] * target[1]).sqrt();
    if magnitude > limit {
        let scale = if magnitude > 0.0 {
            limit / magnitude
        } else {
            0.0
        };
        target = [target[0] * scale, target[1] * scale];
    }
    let delta = [
        target[0] - row.tangent_impulse[0],
        target[1] - row.tangent_impulse[1],
    ];
    row.tangent_impulse = target;
    let p_t = row.tangents[0] * delta[0] + row.tangents[1] * delta[1];
    apply_impulse(bodies, row, &p_t);

    // Normal.
    let vn = relative_velocity(bodies, row).dot(&row.normal);
    let candidate = (row.normal_impulse + row.normal_mass * (row.bounce - vn)).max(0.0);
    let delta_n = candidate - row.normal_impulse;
    row.normal_impulse = candidate;
    apply_impulse(bodies, row, &(row.normal * delta_n));
}

/// Wake sleeping bodies that an awake dynamic or kinematic body runs into
/// faster than the linear sleep threshold.
fn wake_struck_sleepers(
    world: &mut PyPhysicsWorld,
    manifolds: &BTreeMap<(usize, usize), Vec<RawPoint>>,
) {
    let threshold = world.config.linear_sleep_threshold;
    let mut to_wake = Vec::new();
    for (&(a, b), points) in manifolds {
        let (Some(ba), Some(bb)) = (
            world.slots.get(a).and_then(|s| s.body.as_ref()),
            world.slots.get(b).and_then(|s| s.body.as_ref()),
        ) else {
            continue;
        };
        let sleeper_a = ba.inv_mass() > 0.0 && ba.is_sleeping;
        let sleeper_b = bb.inv_mass() > 0.0 && bb.is_sleeping;
        if sleeper_a == sleeper_b {
            continue;
        }
        let (sa, sb) = (SolverBody::from_body(ba), SolverBody::from_body(bb));
        let (pa, pb) = (ba.position, bb.position);
        let approaching = points.iter().any(|p| {
            let r_a = p.point - Vec3::new(pa[0], pa[1], pa[2]);
            let r_b = p.point - Vec3::new(pb[0], pb[1], pb[2]);
            let vn = (sb.point_velocity(&r_b) - sa.point_velocity(&r_a)).dot(&p.normal);
            -vn > threshold
        });
        if approaching {
            to_wake.push(if sleeper_a { a } else { b });
        }
    }
    for index in to_wake {
        if let Some(body) = world.slots.get_mut(index).and_then(|s| s.body.as_mut()) {
            body.is_sleeping = false;
            body.sleep_timer = 0.0;
        }
    }
}

impl PyPhysicsWorld {
    /// Detect all contacts of the current poses and resolve them.
    ///
    /// See the [module documentation](self) for the detection pipeline and the
    /// solver. Contacts of the step are stored in `self.contacts`, one
    /// [`PyContactResult`] per contact point, with the converged normal impulse.
    pub(super) fn detect_and_resolve_contacts(&mut self, dt: f64) {
        let manifolds = detect_manifolds(self);
        if manifolds.is_empty() {
            return;
        }
        wake_struck_sleepers(self, &manifolds);

        let config: &PySimConfig = &self.config;
        let friction_rule = MixRule::from_config(&config.friction_mixing);
        let restitution_rule = MixRule::from_config(&config.restitution_mixing);
        let g = config.gravity;
        let g_norm = (g[0] * g[0] + g[1] * g[1] + g[2] * g[2]).sqrt();
        let bounce_threshold = if g_norm.is_finite() && dt.is_finite() {
            (2.0 * g_norm * dt.abs()).max(MIN_RESTITUTION_SPEED)
        } else {
            MIN_RESTITUTION_SPEED
        };
        let iterations = config.solver_iterations.max(1);
        let beta = config.baumgarte_factor;
        let slop = config.max_penetration.max(0.0);

        // Solver state of every slot (absent slots stay immovable and unused).
        let mut bodies: Vec<SolverBody> = self
            .slots
            .iter()
            .map(|slot| match slot.body.as_ref() {
                Some(body) => SolverBody::from_body(body),
                None => SolverBody {
                    inv_mass: 0.0,
                    inv_inertia: Mat3::zeros(),
                    velocity: Vec3::zeros(),
                    angular_velocity: Vec3::zeros(),
                },
            })
            .collect();

        // Build the rows.
        let mut rows: Vec<ContactRow> = Vec::new();
        let mut solved_pairs: Vec<((usize, usize), RawPoint)> = Vec::new();
        for (&(a, b), points) in &manifolds {
            let (Some(ba), Some(bb)) = (
                self.slots.get(a).and_then(|s| s.body.as_ref()),
                self.slots.get(b).and_then(|s| s.body.as_ref()),
            ) else {
                continue;
            };
            if bodies[a].inv_mass <= 0.0 && bodies[b].inv_mass <= 0.0 {
                continue;
            }
            let friction = friction_rule.combine(ba.friction, bb.friction).max(0.0);
            let restitution = restitution_rule
                .combine(ba.restitution, bb.restitution)
                .clamp(0.0, 1.0);
            let centre_a = Vec3::new(ba.position[0], ba.position[1], ba.position[2]);
            let centre_b = Vec3::new(bb.position[0], bb.position[1], bb.position[2]);
            let mut deepest: Option<RawPoint> = None;
            for p in points {
                let r_a = p.point - centre_a;
                let r_b = p.point - centre_b;
                let t0 = any_perpendicular(&p.normal);
                let t1 = p.normal.cross(&t0);
                let mut row = ContactRow {
                    a,
                    b,
                    point: p.point,
                    r_a,
                    r_b,
                    normal: p.normal,
                    tangents: [t0, t1],
                    normal_mass: effective_mass(&bodies[a], &bodies[b], &r_a, &r_b, &p.normal),
                    tangent_mass: [
                        effective_mass(&bodies[a], &bodies[b], &r_a, &r_b, &t0),
                        effective_mass(&bodies[a], &bodies[b], &r_a, &r_b, &t1),
                    ],
                    bounce: 0.0,
                    friction,
                    restitution,
                    depth: p.depth,
                    normal_impulse: 0.0,
                    tangent_impulse: [0.0; 2],
                };
                let vn = relative_velocity(&bodies, &row).dot(&row.normal);
                if -vn > bounce_threshold {
                    row.bounce = -restitution * vn;
                }
                rows.push(row);
                if deepest.is_none_or(|d| p.depth > d.depth) {
                    deepest = Some(*p);
                }
            }
            if let Some(d) = deepest {
                solved_pairs.push(((a, b), d));
            }
        }

        // Sequential-impulse sweeps.
        for _ in 0..iterations {
            for row in &mut rows {
                solve_row(&mut bodies, row);
            }
        }

        // Write the velocities back to the movable bodies.
        for (slot, state) in self.slots.iter_mut().zip(&bodies) {
            if state.inv_mass <= 0.0 {
                continue;
            }
            if let Some(body) = slot.body.as_mut() {
                body.velocity = [state.velocity.x, state.velocity.y, state.velocity.z];
                body.angular_velocity = [
                    state.angular_velocity.x,
                    state.angular_velocity.y,
                    state.angular_velocity.z,
                ];
            }
        }

        // Mass-weighted positional correction, once per body pair.
        for ((a, b), deepest) in &solved_pairs {
            let (inv_a, inv_b) = (bodies[*a].inv_mass, bodies[*b].inv_mass);
            let w = inv_a + inv_b;
            if w <= 0.0 {
                continue;
            }
            let c = beta * (deepest.depth - slop).max(0.0) / w;
            if c <= 0.0 || !c.is_finite() {
                continue;
            }
            let n = deepest.normal;
            if let Some(body) = self.slots.get_mut(*a).and_then(|s| s.body.as_mut())
                && inv_a > 0.0
            {
                for (k, x) in body.position.iter_mut().enumerate() {
                    *x -= n[k] * c * inv_a;
                }
            }
            if let Some(body) = self.slots.get_mut(*b).and_then(|s| s.body.as_mut())
                && inv_b > 0.0
            {
                for (k, x) in body.position.iter_mut().enumerate() {
                    *x += n[k] * c * inv_b;
                }
            }
        }

        // Report.
        self.contacts.extend(rows.iter().map(|row| PyContactResult {
            body_a: row.a as u32,
            body_b: row.b as u32,
            contact_point: [row.point.x, row.point.y, row.point.z],
            normal: [row.normal.x, row.normal.y, row.normal.z],
            depth: row.depth,
            friction: row.friction,
            restitution: row.restitution,
            impulse: row.normal_impulse,
        }));
    }

    /// Nearest ray hit over every collider of every body.
    ///
    /// `dir` must be unit length; distances are along it. Convex colliders use
    /// the engine shape's `ray_cast` in the body frame; plane colliders are hit
    /// from their front side (a ray starting inside the half-space hits at 0).
    pub(super) fn raycast_colliders(
        &self,
        origin: [f64; 3],
        dir: [f64; 3],
        max_dist: f64,
    ) -> Option<(u32, f64)> {
        let o = Vec3::new(origin[0], origin[1], origin[2]);
        let d = Vec3::new(dir[0], dir[1], dir[2]);
        let mut best: Option<(u32, f64)> = None;
        let mut best_t = max_dist;
        for (index, slot) in self.slots.iter().enumerate() {
            let Some(body) = slot.body.as_ref() else {
                continue;
            };
            let tf = body_transform(body);
            let inv_rot = tf.rotation.inverse();
            let local_origin = inv_rot * (o - tf.position);
            let local_dir = inv_rot * d;
            for shape in &body.shapes {
                let hit = if let Some((n, offset)) = world_plane(shape, &tf) {
                    let height = n.dot(&o) - offset;
                    let along = n.dot(&d);
                    if height <= 0.0 {
                        Some(0.0)
                    } else if along < -1e-12 {
                        Some(height / -along)
                    } else {
                        None
                    }
                } else if let Some((engine, _)) = convex_shape(shape) {
                    engine
                        .ray_cast(&local_origin, &local_dir, best_t)
                        .map(|h| h.toi)
                } else {
                    None
                };
                if let Some(t) = hit
                    && t.is_finite()
                    && t >= 0.0
                    && t < best_t
                {
                    best_t = t;
                    best = Some((index as u32, t));
                }
            }
        }
        best
    }
}

#[cfg(test)]
#[path = "contacts_tests.rs"]
mod tests;
