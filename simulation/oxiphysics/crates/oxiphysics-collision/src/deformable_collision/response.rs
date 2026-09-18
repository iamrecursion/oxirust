// Copyright 2026 COOLJAPAN OU (Team KitaSan)
// SPDX-License-Identifier: Apache-2.0

//! Response of deformable meshes to the contacts found by the detectors of
//! [`super::functions`]: penalty forces and position constraints.
//!
//! A [`DeformContact`] relates a primitive of mesh A (`feature_a`) to a
//! primitive of mesh B (`feature_b`), and the two indices refer to different
//! meshes. A correction therefore always acts on one mesh at a time, and the
//! functions come in pairs: the plain functions act on the A side of the
//! contacts and the `_b` functions on their B side.
//!
//! | contact type                         | A side moves             | B side moves            |
//! |--------------------------------------|--------------------------|-------------------------|
//! | [`DeformContactType::VertexRigid`]   | vertex `feature_a`       | (rigid body)            |
//! | [`DeformContactType::VertexFace`]    | vertex `feature_a`       | none (the face is hit)  |
//! | [`DeformContactType::FaceVertex`]    | none (the face is hit)   | vertex `feature_b`      |
//! | [`DeformContactType::EdgeEdge`]      | edge `feature_a`         | edge `feature_b`        |
//! | [`DeformContactType::FaceRigid`]     | none                     | none                    |
//!
//! Contacts that do not move anything on the requested side are skipped, so one
//! contact list can be passed to both calls. For two meshes call the plain
//! function on mesh A and the `_b` function on mesh B. For self-collision both
//! sides are the same mesh and both calls take the same mesh.
//!
//! Contact normals point from B to A, so the A side is pushed along the normal
//! and the B side against it.

use super::functions::{compute_penalty_force, v3_add, v3_neg, v3_scale};
use super::types::{
    DeformConstraint, DeformContact, DeformContactType, DeformableMesh, PenaltyParams,
};

/// The mesh of a contact that a correction acts on.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Side {
    /// The mesh that `feature_a` indexes.
    A,
    /// The mesh that `feature_b` indexes.
    B,
}

/// The vertices of one mesh that carry one side of a contact, together with the
/// direction in which the contact has to push them.
struct Carrier {
    /// Vertex indices; only the first `count` are used.
    vertices: [usize; 2],
    /// Weight of the contact point in each vertex: 1 for a single vertex,
    /// `(1 - s, s)` for a point at parameter `s` along an edge.
    weights: [f64; 2],
    count: usize,
    /// Unit direction in which the carrier leaves the contact.
    push: [f64; 3],
}

impl Carrier {
    /// A single vertex of `mesh`, or `None` if the mesh has no such vertex.
    fn vertex(mesh: &DeformableMesh, vertex: usize, push: [f64; 3]) -> Option<Self> {
        if vertex >= mesh.positions.len() {
            return None;
        }
        Some(Self {
            vertices: [vertex, 0],
            weights: [1.0, 0.0],
            count: 1,
            push,
        })
    }

    /// The point at parameter `param` of edge `edge_id` (`face * 3 + k`, see
    /// [`DeformContactType::EdgeEdge`]) of `mesh`, or `None` if the mesh has no
    /// such edge.
    fn edge(mesh: &DeformableMesh, edge_id: usize, param: f64, push: [f64; 3]) -> Option<Self> {
        let face = mesh.faces.get(edge_id / 3)?;
        let (v0, v1) = face.edges()[edge_id % 3];
        if v0 >= mesh.positions.len() || v1 >= mesh.positions.len() {
            return None;
        }
        let s = if param.is_nan() {
            0.0
        } else {
            param.clamp(0.0, 1.0)
        };
        Some(Self {
            vertices: [v0, v1],
            weights: [1.0 - s, s],
            count: 2,
            push,
        })
    }

    /// The carrying vertices with the weight of the contact point in each.
    fn iter(&self) -> impl Iterator<Item = (usize, f64)> + '_ {
        self.vertices[..self.count]
            .iter()
            .copied()
            .zip(self.weights[..self.count].iter().copied())
    }
}

/// What carries `side` of `contact` in `mesh`, or `None` if the contact moves
/// nothing on that side (or refers to primitives that `mesh` does not have).
fn side_carrier(mesh: &DeformableMesh, side: Side, contact: &DeformContact) -> Option<Carrier> {
    let normal = contact.normal;
    match (side, contact.contact_type) {
        (Side::A, DeformContactType::VertexRigid | DeformContactType::VertexFace) => {
            Carrier::vertex(mesh, contact.feature_a, normal)
        }
        (Side::B, DeformContactType::FaceVertex) => {
            Carrier::vertex(mesh, contact.feature_b, v3_neg(normal))
        }
        (Side::A, DeformContactType::EdgeEdge) => {
            Carrier::edge(mesh, contact.feature_a, contact.bary[0], normal)
        }
        (Side::B, DeformContactType::EdgeEdge) => {
            Carrier::edge(mesh, contact.feature_b, contact.bary[1], v3_neg(normal))
        }
        _ => None,
    }
}

/// Velocity of the contact point on `carrier`.
fn carrier_velocity(mesh: &DeformableMesh, carrier: &Carrier) -> Option<[f64; 3]> {
    let mut velocity = [0.0; 3];
    for (vertex, weight) in carrier.iter() {
        let v = mesh.velocities.get(vertex)?;
        velocity = v3_add(velocity, v3_scale(*v, weight));
    }
    Some(velocity)
}

fn apply_penalty_forces_on(
    mesh: &mut DeformableMesh,
    side: Side,
    contacts: &[DeformContact],
    params: &PenaltyParams,
    dt: f64,
) {
    for contact in contacts {
        let Some(carrier) = side_carrier(mesh, side, contact) else {
            continue;
        };
        let Some(velocity) = carrier_velocity(mesh, &carrier) else {
            continue;
        };
        // The force pushes the carrier out of the contact: along its push
        // direction, which is the contact normal on the A side and its negation
        // on the B side.
        let pushed = DeformContact::new(
            contact.position,
            carrier.push,
            contact.depth,
            contact.bary,
            contact.contact_type,
            contact.feature_a,
            contact.feature_b,
        );
        let force = compute_penalty_force(&pushed, velocity, params);
        // A point force is shared among the vertices by the weights of the
        // point in them (the weights add up to one).
        for (vertex, weight) in carrier.iter() {
            let inv_m = mesh.inv_mass.get(vertex).copied().unwrap_or(0.0);
            if inv_m <= 0.0 {
                continue;
            }
            if let Some(v) = mesh.velocities.get_mut(vertex) {
                *v = v3_add(*v, v3_scale(force, weight * inv_m * dt));
            }
        }
    }
}

/// Apply the penalty forces of the A side of `contacts` to `mesh`.
///
/// Modifies the velocities in place: every vertex that carries the A side of a
/// contact (see the module documentation) gets the penalty force of the contact
/// along the contact normal, `stiffness * depth - damping * v_n` plus friction,
/// scaled by its inverse mass and `dt`. An edge-edge contact shares its force
/// among the two edge vertices by the position of the contact along the edge.
/// The other body is treated as immovable: only the velocity of the vertices of
/// `mesh` enters the damping and friction terms.
///
/// `mesh` is mesh A of the contacts: the mesh that `feature_a` indexes. Contacts
/// of a rigid body are all on the A side. Use [`apply_penalty_forces_b`] for the
/// vertices of mesh B.
pub fn apply_penalty_forces(
    mesh: &mut DeformableMesh,
    contacts: &[DeformContact],
    params: &PenaltyParams,
    dt: f64,
) {
    apply_penalty_forces_on(mesh, Side::A, contacts, params, dt);
}

/// Apply the penalty forces of the B side of `contacts` to `mesh_b`.
///
/// Like [`apply_penalty_forces`], for the mesh that `feature_b` indexes: the
/// vertex of a [`DeformContactType::FaceVertex`] contact and the B edge of an
/// edge-edge contact are pushed against the contact normal (away from mesh A).
/// Contacts that only move vertices of mesh A are skipped, so mesh A is never
/// touched by indices of mesh B.
pub fn apply_penalty_forces_b(
    mesh_b: &mut DeformableMesh,
    contacts: &[DeformContact],
    params: &PenaltyParams,
    dt: f64,
) {
    apply_penalty_forces_on(mesh_b, Side::B, contacts, params, dt);
}

fn inverse_mass(mesh: &DeformableMesh, vertex: usize) -> f64 {
    mesh.inv_mass.get(vertex).copied().unwrap_or(0.0)
}

fn constraints_on(
    mesh: &DeformableMesh,
    side: Side,
    contacts: &[DeformContact],
) -> Vec<DeformConstraint> {
    let mut constraints = Vec::new();
    for contact in contacts {
        let Some(carrier) = side_carrier(mesh, side, contact) else {
            continue;
        };
        if carrier.count == 1 {
            let vertex = carrier.vertices[0];
            let Some(&position) = mesh.positions.get(vertex) else {
                continue;
            };
            // Every detector defines `depth` as the contact thickness minus the
            // signed distance of the vertex from the surface, so moving the
            // vertex by `depth` along the push direction puts it exactly one
            // thickness outside the surface.
            constraints.push(DeformConstraint {
                vertex,
                target: v3_add(position, v3_scale(carrier.push, contact.depth)),
                normal: carrier.push,
                depth: contact.depth,
            });
        } else {
            edge_constraints(mesh, &carrier, contact.depth, &mut constraints);
        }
    }
    constraints
}

/// Constraints that move an edge so its contact point travels half of `depth`.
///
/// The edges of an edge-edge contact are pushed apart by `depth` in total and
/// each edge covers half. Within an edge the displacement of the vertices is
/// weighted by the weight of the contact point in the vertex and by the inverse
/// mass (position-based dynamics), which moves the contact point by exactly the
/// share even if one vertex is pinned. A fully pinned edge adds no constraint.
fn edge_constraints(
    mesh: &DeformableMesh,
    carrier: &Carrier,
    depth: f64,
    constraints: &mut Vec<DeformConstraint>,
) {
    let share = 0.5 * depth;
    let denominator: f64 = carrier
        .iter()
        .map(|(vertex, weight)| weight * weight * inverse_mass(mesh, vertex))
        .sum();
    if denominator <= 0.0 || !denominator.is_finite() {
        return;
    }
    for (vertex, weight) in carrier.iter() {
        let distance = weight * inverse_mass(mesh, vertex) / denominator * share;
        if distance <= 0.0 {
            continue;
        }
        let Some(&position) = mesh.positions.get(vertex) else {
            continue;
        };
        constraints.push(DeformConstraint {
            vertex,
            target: v3_add(position, v3_scale(carrier.push, distance)),
            normal: carrier.push,
            depth: distance,
        });
    }
}

/// Generate position constraints for the A side of `contacts` on `mesh`.
///
/// Each vertex contact ([`DeformContactType::VertexRigid`],
/// [`DeformContactType::VertexFace`]) gives a constraint that moves the vertex
/// `feature_a` along the contact normal to `position + normal * depth`, which is
/// one contact thickness outside the surface for every detector. An edge-edge
/// contact gives constraints on the two vertices of its A edge, which together
/// cover half of the depth (the B edge covers the other half, see
/// [`generate_constraints_b`]).
///
/// `mesh` is mesh A of the contacts: the mesh that `feature_a` indexes, and the
/// mesh whose current positions the contacts were detected on. Contacts that do
/// not move a vertex of the A side are skipped.
pub fn generate_constraints(
    mesh: &DeformableMesh,
    contacts: &[DeformContact],
) -> Vec<DeformConstraint> {
    constraints_on(mesh, Side::A, contacts)
}

/// Generate position constraints for the B side of `contacts` on `mesh_b`.
///
/// Like [`generate_constraints`], for the mesh that `feature_b` indexes: the
/// vertex of a [`DeformContactType::FaceVertex`] contact is moved against the
/// contact normal by the depth, and the vertices of the B edge of an edge-edge
/// contact cover half of its depth. The constraints carry the push direction
/// (the negated contact normal) as their `normal`, so
/// [`solve_constraints`](super::solve_constraints) applies them to `mesh_b`
/// unchanged.
pub fn generate_constraints_b(
    mesh_b: &DeformableMesh,
    contacts: &[DeformContact],
) -> Vec<DeformConstraint> {
    constraints_on(mesh_b, Side::B, contacts)
}
