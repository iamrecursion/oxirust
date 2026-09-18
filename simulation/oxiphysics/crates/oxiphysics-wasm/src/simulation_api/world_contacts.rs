// Copyright 2026 COOLJAPAN OU (Team KitaSan)
// SPDX-License-Identifier: Apache-2.0

//! `WasmWorld` collision detection and contact response.
//!
//! * **Narrow phase.**  Box-plane pairs produce one contact per penetrating
//!   box corner (a stable multi-point manifold for resting boxes).  Every
//!   other pair (sphere-sphere, sphere-box, sphere-plane, box-box, and the
//!   capsule / cylinder combinations) uses the engine narrow phase shared
//!   with [`crate::engine::WasmPhysicsEngine`] (closed-form tests, SAT with
//!   face clipping for box-box, GJK/EPA otherwise).  Plane colliders are
//!   world-space half-spaces `normal . x <= offset`; plane-plane pairs never
//!   collide.  Cones and convex hulls (which `WasmWorld` cannot create) are
//!   not collided.
//! * **Response.**  Sequential impulses over `config.velocity_iters`
//!   iterations: a non-negative normal impulse with restitution (above a
//!   small approach speed) and two Coulomb friction impulses bounded by
//!   `mu * lambda_n`, applied at the contact points so they also change the
//!   angular velocity.  Restitution of a pair is the larger, friction the
//!   geometric mean of the two coefficients.  Penetration beyond a small slop
//!   is then removed by moving the bodies apart in proportion to their
//!   inverse masses.  Static and kinematic bodies have infinite mass.
//! * **Sensors.**  A pair with a sensor collider gets no response; it emits
//!   [`WasmPhysicsEvent::SensorEntered`] / [`WasmPhysicsEvent::SensorExited`]
//!   when the overlap starts / ends.  Solid pairs emit
//!   [`WasmPhysicsEvent::CollisionStarted`] / [`WasmPhysicsEvent::CollisionEnded`].

use std::collections::{BTreeMap, BTreeSet};

use super::types::{WasmColliderShape, WasmContactEvent};
use super::types_3::{WasmPhysicsEvent, WasmWorld};
use super::types_4::{WasmBodyType, WasmCollider, WasmColliderHandle};
use super::vec_math::{
    M3, Q4, V3, m_vec, q_rotate, v_add, v_cross, v_dot, v_madd, v_neg, v_scale, v_sub,
    v_tangent_basis,
};
use super::world_dynamics::{solver_inv_mass, world_inv_inertia};
use crate::engine::crate_bridge::{
    PrimitivePose, PrimitiveShape, collide_primitives, raycast_primitive,
};

/// Penetration allowed before positional correction kicks in (m).
const PENETRATION_SLOP: f64 = 0.005;
/// Fraction of the remaining penetration removed per step.
const POSITION_CORRECTION: f64 = 0.8;
/// Approach speed below which contacts do not bounce (m/s).
const RESTITUTION_THRESHOLD: f64 = 0.5;

/// Engine primitive for a collider shape (`None` for shapes the shared
/// narrow phase does not support).
fn primitive_shape(shape: &WasmColliderShape) -> Option<PrimitiveShape> {
    match shape {
        WasmColliderShape::Sphere { radius } => Some(PrimitiveShape::Sphere { radius: *radius }),
        WasmColliderShape::Box { half_extents } => Some(PrimitiveShape::Box {
            half_extents: *half_extents,
        }),
        WasmColliderShape::Capsule {
            half_height,
            radius,
        } => Some(PrimitiveShape::Capsule {
            radius: *radius,
            half_height: *half_height,
        }),
        WasmColliderShape::Cylinder {
            half_height,
            radius,
        } => Some(PrimitiveShape::Cylinder {
            radius: *radius,
            half_height: *half_height,
        }),
        WasmColliderShape::Plane { normal, offset } => Some(PrimitiveShape::Plane {
            normal: *normal,
            offset: *offset,
        }),
        WasmColliderShape::Cone { .. } | WasmColliderShape::ConvexHull { .. } => None,
    }
}

/// Unit normal and offset of a plane collider (`None` if degenerate).
fn unit_plane(normal: V3, offset: f64) -> Option<(V3, f64)> {
    let len = v_dot(normal, normal).sqrt();
    (len.is_finite() && len > 1e-12 && offset.is_finite())
        .then(|| (v_scale(normal, 1.0 / len), offset / len))
}

/// Ray cast against one collider placed at `position` / `rotation`.
///
/// Returns the hit distance and the outward normal for solids (boxes as
/// oriented boxes, slab test).  Spheres and planes keep their analytic
/// two-sided tests in `WasmWorld::raycast`.
pub(super) fn raycast_collider(
    origin: V3,
    dir: V3,
    shape: &WasmColliderShape,
    position: V3,
    rotation: Q4,
) -> Option<(f64, V3)> {
    let primitive = primitive_shape(shape)?;
    raycast_primitive(
        origin,
        dir,
        &PrimitivePose {
            shape: primitive,
            position,
            rotation,
        },
    )
}

/// One contact point of a manifold.
#[derive(Debug, Clone, Copy)]
struct ContactPoint {
    /// Unit normal pointing from collider B toward collider A.
    normal: V3,
    /// Penetration depth (positive).
    depth: f64,
    /// World-space contact point (between the two surfaces).
    point: V3,
}

/// Contacts of a box (pose `position`, `rotation`) against a plane, one per
/// penetrating corner; `normal` points out of the plane.
fn box_plane_points(
    half: V3,
    position: V3,
    rotation: Q4,
    plane_normal: V3,
    plane_offset: f64,
) -> Vec<ContactPoint> {
    let mut points = Vec::new();
    for corner in 0..8 {
        let local = [
            if corner & 1 == 1 { half[0] } else { -half[0] },
            if corner & 2 == 2 { half[1] } else { -half[1] },
            if corner & 4 == 4 { half[2] } else { -half[2] },
        ];
        let world = v_add(position, q_rotate(rotation, local));
        let depth = plane_offset - v_dot(plane_normal, world);
        if depth > 0.0 {
            points.push(ContactPoint {
                normal: plane_normal,
                depth,
                // Midway between the corner and its projection on the plane.
                point: v_madd(world, plane_normal, 0.5 * depth),
            });
        }
    }
    points
}

/// Narrow phase between two placed colliders (normals from B toward A).
fn collide_pair(
    shape_a: &WasmColliderShape,
    pose_a: (V3, Q4),
    shape_b: &WasmColliderShape,
    pose_b: (V3, Q4),
) -> Vec<ContactPoint> {
    match (shape_a, shape_b) {
        (WasmColliderShape::Box { half_extents }, WasmColliderShape::Plane { normal, offset }) => {
            unit_plane(*normal, *offset)
                .map(|(n, d)| box_plane_points(*half_extents, pose_a.0, pose_a.1, n, d))
                .unwrap_or_default()
        }
        (WasmColliderShape::Plane { normal, offset }, WasmColliderShape::Box { half_extents }) => {
            unit_plane(*normal, *offset)
                .map(|(n, d)| {
                    box_plane_points(*half_extents, pose_b.0, pose_b.1, n, d)
                        .into_iter()
                        .map(|c| ContactPoint {
                            normal: v_neg(c.normal),
                            ..c
                        })
                        .collect()
                })
                .unwrap_or_default()
        }
        _ => {
            let (Some(pa), Some(pb)) = (primitive_shape(shape_a), primitive_shape(shape_b)) else {
                return Vec::new();
            };
            let a = PrimitivePose {
                shape: pa,
                position: pose_a.0,
                rotation: pose_a.1,
            };
            let b = PrimitivePose {
                shape: pb,
                position: pose_b.0,
                rotation: pose_b.1,
            };
            collide_primitives(&a, &b)
                .map(|c| {
                    vec![ContactPoint {
                        normal: c.normal,
                        depth: c.depth,
                        point: v_scale(v_add(c.point_a, c.point_b), 0.5),
                    }]
                })
                .unwrap_or_default()
        }
    }
}

/// Velocity state of a body during the contact solve.
#[derive(Debug, Clone)]
struct ContactBody {
    position: V3,
    velocity: V3,
    angular_velocity: V3,
    inv_mass: f64,
    inv_inertia: M3,
}

/// One contact point prepared for the sequential impulse solver.
#[derive(Debug, Clone)]
struct ContactRow {
    manifold: usize,
    point: ContactPoint,
    r_a: V3,
    r_b: V3,
    tangents: [V3; 2],
    normal_mass: f64,
    tangent_mass: [f64; 2],
    /// Restitution coefficient of the pair.
    restitution: f64,
    /// Normal separation speed the solver aims for (restitution bounce).
    target_normal_velocity: f64,
    lambda_n: f64,
    lambda_t: [f64; 2],
}

/// Contacts between two colliders.
#[derive(Debug, Clone)]
struct Manifold {
    collider_a: WasmColliderHandle,
    collider_b: WasmColliderHandle,
    body_a: u32,
    body_b: u32,
    friction: f64,
    max_depth: f64,
    normal: V3,
}

fn effective_mass(body: &ContactBody, r: V3, dir: V3) -> f64 {
    let rn = v_cross(r, dir);
    body.inv_mass + v_dot(rn, m_vec(&body.inv_inertia, rn))
}

/// Velocity of the material point at lever `r`.
fn point_velocity(body: &ContactBody, r: V3) -> V3 {
    v_add(body.velocity, v_cross(body.angular_velocity, r))
}

/// Apply the impulse `p` at lever `r`.
fn apply_impulse(body: &mut ContactBody, r: V3, p: V3) {
    body.velocity = v_madd(body.velocity, p, body.inv_mass);
    body.angular_velocity = v_add(
        body.angular_velocity,
        m_vec(&body.inv_inertia, v_cross(r, p)),
    );
}

/// Apply `p` to body A and `-p` to body B (no-op on the same slot).
fn apply_pair_impulse(
    bodies: &mut BTreeMap<u32, ContactBody>,
    body_a: u32,
    body_b: u32,
    r_a: V3,
    r_b: V3,
    p: V3,
) {
    if let Some(a) = bodies.get_mut(&body_a) {
        apply_impulse(a, r_a, p);
    }
    if let Some(b) = bodies.get_mut(&body_b) {
        apply_impulse(b, r_b, v_neg(p));
    }
}

/// Relative velocity of A with respect to B at the contact.
fn relative_velocity(bodies: &BTreeMap<u32, ContactBody>, m: &Manifold, r_a: V3, r_b: V3) -> V3 {
    match (bodies.get(&m.body_a), bodies.get(&m.body_b)) {
        (Some(a), Some(b)) => v_sub(point_velocity(a, r_a), point_velocity(b, r_b)),
        _ => [0.0; 3],
    }
}

/// Collider pair key with the lower handle first.
fn pair_key(a: WasmColliderHandle, b: WasmColliderHandle) -> (u32, u32) {
    (a.0.min(b.0), a.0.max(b.0))
}

impl WasmWorld {
    /// Detect all collider contacts, resolve them and emit contact, collision
    /// and sensor events (see the module docs).
    pub(super) fn detect_and_resolve_contacts(&mut self) {
        let colliders: Vec<&WasmCollider> = self.colliders.values().collect();
        let mut manifolds: Vec<Manifold> = Vec::new();
        let mut rows: Vec<ContactRow> = Vec::new();
        let mut touching: BTreeSet<(u32, u32)> = BTreeSet::new();
        let mut sensing: BTreeMap<(u32, u32), (WasmColliderHandle, WasmColliderHandle)> =
            BTreeMap::new();

        for (i, ca) in colliders.iter().enumerate() {
            let Some(body_a) = ca.body.and_then(|h| self.bodies.get(&h.0)) else {
                continue;
            };
            for cb in colliders.iter().skip(i + 1) {
                let Some(body_b) = cb.body.and_then(|h| self.bodies.get(&h.0)) else {
                    continue;
                };
                if body_a.handle == body_b.handle {
                    continue;
                }
                let a_moves = body_a.body_type == WasmBodyType::Dynamic;
                let b_moves = body_b.body_type == WasmBodyType::Dynamic;
                if !a_moves && !b_moves {
                    continue;
                }
                let reach = ca.shape.bounding_radius() + cb.shape.bounding_radius();
                if reach.is_finite() {
                    let d = v_sub(body_a.position, body_b.position);
                    if v_dot(d, d) > reach * reach {
                        continue;
                    }
                }
                let points = collide_pair(
                    &ca.shape,
                    (body_a.position, body_a.rotation),
                    &cb.shape,
                    (body_b.position, body_b.rotation),
                );
                if points.is_empty() {
                    continue;
                }
                if ca.is_sensor || cb.is_sensor {
                    let (sensor, other) = if ca.is_sensor {
                        (ca.handle, cb.handle)
                    } else {
                        (cb.handle, ca.handle)
                    };
                    sensing.insert(pair_key(ca.handle, cb.handle), (sensor, other));
                    continue;
                }
                touching.insert(pair_key(ca.handle, cb.handle));
                let restitution = ca.restitution.max(cb.restitution).clamp(0.0, 1.0);
                let manifold_index = manifolds.len();
                let mut max_depth = 0.0_f64;
                let mut deepest_normal = points[0].normal;
                for point in &points {
                    if point.depth > max_depth {
                        max_depth = point.depth;
                        deepest_normal = point.normal;
                    }
                    rows.push(ContactRow {
                        manifold: manifold_index,
                        point: *point,
                        r_a: v_sub(point.point, body_a.position),
                        r_b: v_sub(point.point, body_b.position),
                        tangents: [[0.0; 3]; 2],
                        normal_mass: 0.0,
                        tangent_mass: [0.0; 2],
                        restitution,
                        target_normal_velocity: 0.0,
                        lambda_n: 0.0,
                        lambda_t: [0.0; 2],
                    });
                }
                manifolds.push(Manifold {
                    collider_a: ca.handle,
                    collider_b: cb.handle,
                    body_a: body_a.handle.0,
                    body_b: body_b.handle.0,
                    friction: (ca.friction.max(0.0) * cb.friction.max(0.0)).sqrt(),
                    max_depth,
                    normal: deepest_normal,
                });
            }
        }

        self.resolve_manifolds(&manifolds, &mut rows);
        self.emit_pair_events(&touching, &sensing);
        self.contact_pairs = touching;
        self.sensor_pairs = sensing.keys().copied().collect();

        for row in &rows {
            let m = &manifolds[row.manifold];
            let friction_impulse = v_add(
                v_scale(row.tangents[0], row.lambda_t[0]),
                v_scale(row.tangents[1], row.lambda_t[1]),
            );
            self.contact_events.push(WasmContactEvent {
                collider_a: m.collider_a,
                collider_b: m.collider_b,
                normal: row.point.normal,
                contact_point: row.point.point,
                depth: row.point.depth,
                impulse: row.lambda_n,
                friction_impulse,
            });
        }
    }

    /// Sequential-impulse velocity solve plus positional correction.
    fn resolve_manifolds(&mut self, manifolds: &[Manifold], rows: &mut [ContactRow]) {
        if rows.is_empty() {
            return;
        }
        let mut bodies: BTreeMap<u32, ContactBody> = BTreeMap::new();
        for m in manifolds {
            for id in [m.body_a, m.body_b] {
                if let Some(body) = self.bodies.get(&id) {
                    bodies.entry(id).or_insert_with(|| ContactBody {
                        position: body.position,
                        velocity: body.linear_vel,
                        angular_velocity: body.angular_vel,
                        inv_mass: solver_inv_mass(body),
                        inv_inertia: world_inv_inertia(body),
                    });
                }
            }
        }

        // Prepare rows.
        for row in rows.iter_mut() {
            let m = &manifolds[row.manifold];
            let (Some(a), Some(b)) = (bodies.get(&m.body_a), bodies.get(&m.body_b)) else {
                continue;
            };
            let n = row.point.normal;
            let (t1, t2) = v_tangent_basis(n);
            row.tangents = [t1, t2];
            let k_n = effective_mass(a, row.r_a, n) + effective_mass(b, row.r_b, n);
            row.normal_mass = if k_n > 1e-15 { 1.0 / k_n } else { 0.0 };
            for (mass, t) in row.tangent_mass.iter_mut().zip([t1, t2]) {
                let k_t = effective_mass(a, row.r_a, t) + effective_mass(b, row.r_b, t);
                *mass = if k_t > 1e-15 { 1.0 / k_t } else { 0.0 };
            }
            let restitution = row.restitution;
            let vn = v_dot(relative_velocity(&bodies, m, row.r_a, row.r_b), n);
            row.target_normal_velocity = if vn < -RESTITUTION_THRESHOLD {
                -restitution * vn
            } else {
                0.0
            };
        }

        // Sequential impulses.
        let iterations = self.config.velocity_iters.max(1);
        for _ in 0..iterations {
            for row in rows.iter_mut() {
                let m = &manifolds[row.manifold];
                let n = row.point.normal;
                // Friction first (bounded by the current normal impulse).
                for k in 0..2 {
                    let t = row.tangents[k];
                    let vt = v_dot(relative_velocity(&bodies, m, row.r_a, row.r_b), t);
                    let bound = m.friction * row.lambda_n;
                    let new_lambda =
                        (row.lambda_t[k] - row.tangent_mass[k] * vt).clamp(-bound, bound);
                    let applied = new_lambda - row.lambda_t[k];
                    row.lambda_t[k] = new_lambda;
                    apply_pair_impulse(
                        &mut bodies,
                        m.body_a,
                        m.body_b,
                        row.r_a,
                        row.r_b,
                        v_scale(t, applied),
                    );
                }
                let vn = v_dot(relative_velocity(&bodies, m, row.r_a, row.r_b), n);
                let new_lambda =
                    (row.lambda_n + row.normal_mass * (row.target_normal_velocity - vn)).max(0.0);
                let applied = new_lambda - row.lambda_n;
                row.lambda_n = new_lambda;
                apply_pair_impulse(
                    &mut bodies,
                    m.body_a,
                    m.body_b,
                    row.r_a,
                    row.r_b,
                    v_scale(n, applied),
                );
            }
        }

        // Positional correction, split by inverse mass, once per manifold.
        for m in manifolds {
            let excess = m.max_depth - PENETRATION_SLOP;
            if excess <= 0.0 {
                continue;
            }
            let wa = bodies.get(&m.body_a).map_or(0.0, |b| b.inv_mass);
            let wb = bodies.get(&m.body_b).map_or(0.0, |b| b.inv_mass);
            let w = wa + wb;
            if w <= 0.0 {
                continue;
            }
            let correction = v_scale(m.normal, POSITION_CORRECTION * excess / w);
            if let Some(a) = bodies.get_mut(&m.body_a) {
                a.position = v_madd(a.position, correction, wa);
            }
            if let Some(b) = bodies.get_mut(&m.body_b) {
                b.position = v_madd(b.position, correction, -wb);
            }
        }

        for (id, state) in bodies {
            if state.inv_mass <= 0.0 {
                continue;
            }
            if let Some(body) = self.bodies.get_mut(&id) {
                body.position = state.position;
                body.linear_vel = state.velocity;
                body.angular_vel = state.angular_velocity;
            }
        }
    }

    /// Emit start / end events by comparing with the previous step's pairs.
    fn emit_pair_events(
        &mut self,
        touching: &BTreeSet<(u32, u32)>,
        sensing: &BTreeMap<(u32, u32), (WasmColliderHandle, WasmColliderHandle)>,
    ) {
        for key in touching.difference(&self.contact_pairs) {
            self.events.push(WasmPhysicsEvent::CollisionStarted {
                collider_a: WasmColliderHandle(key.0),
                collider_b: WasmColliderHandle(key.1),
            });
        }
        for key in self.contact_pairs.difference(touching) {
            self.events.push(WasmPhysicsEvent::CollisionEnded {
                collider_a: WasmColliderHandle(key.0),
                collider_b: WasmColliderHandle(key.1),
            });
        }
        for (key, (sensor, other)) in sensing {
            if !self.sensor_pairs.contains(key) {
                self.events.push(WasmPhysicsEvent::SensorEntered {
                    sensor: *sensor,
                    other: *other,
                });
            }
        }
        let previous: Vec<(u32, u32)> = self.sensor_pairs.iter().copied().collect();
        for key in previous {
            if sensing.contains_key(&key) {
                continue;
            }
            // Report the sensor side when it still exists.
            let (sensor, other) = match self.colliders.get(&key.1) {
                Some(c) if c.is_sensor => (key.1, key.0),
                _ => (key.0, key.1),
            };
            self.events.push(WasmPhysicsEvent::SensorExited {
                sensor: WasmColliderHandle(sensor),
                other: WasmColliderHandle(other),
            });
        }
    }
}
