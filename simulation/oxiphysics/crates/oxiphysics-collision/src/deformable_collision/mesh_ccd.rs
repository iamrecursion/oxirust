// Copyright 2026 COOLJAPAN OU (Team KitaSan)
// SPDX-License-Identifier: Apache-2.0

//! Continuous collision detection between two deformable meshes over one time
//! step.
//!
//! Every vertex of both meshes moves linearly from its position at the start of
//! the step (`prev`) to its position at the end (`curr`). The earliest time at
//! which the two surfaces come within the contact thickness is found with
//! the primitive tests of [`super::functions`]:
//!
//! - every vertex of mesh A against every candidate face of mesh B
//!   ([`DeformContactType::VertexFace`]),
//! - every vertex of mesh B against every candidate face of mesh A
//!   ([`DeformContactType::FaceVertex`]),
//! - every edge of a candidate face of A against every edge of a candidate face
//!   of B ([`DeformContactType::EdgeEdge`]).
//!
//! Two triangles can only meet if the boxes they sweep during the step overlap,
//! so the candidate face pairs come from bounding volume hierarchies built over
//! the swept boxes ([`DeformBvh::build_swept`]), not over a single
//! configuration: a sheet that moves completely through another one within the
//! step is on the far side of it at the end, and only its swept box overlaps the
//! other sheet.

use super::functions::{
    EPSILON, ccd_edge_edge_analytic, ccd_vertex_face_analytic, closest_point_on_triangle,
    closest_points_edge_edge, v3_add, v3_cross, v3_dot, v3_len, v3_lerp, v3_neg, v3_norm, v3_scale,
    v3_sub,
};
use super::types::{DeformBvh, DeformCcdResult, DeformContactType, DeformableMesh};

type Vec3 = [f64; 3];

/// Position of a vertex at the start and at the end of the time step.
#[derive(Clone, Copy)]
struct Motion {
    start: Vec3,
    end: Vec3,
}

impl Motion {
    /// Position at time `t` in \[0, 1\].
    fn at(&self, t: f64) -> Vec3 {
        v3_lerp(self.start, self.end, t)
    }
}

/// A deformable mesh over the time step: its start and end configuration.
struct MovingMesh<'a> {
    prev: &'a DeformableMesh,
    curr: &'a DeformableMesh,
}

impl MovingMesh<'_> {
    /// Motion of `vertex`, or `None` if either configuration lacks it.
    fn motion(&self, vertex: usize) -> Option<Motion> {
        Some(Motion {
            start: *self.prev.positions.get(vertex)?,
            end: *self.curr.positions.get(vertex)?,
        })
    }

    /// Face `index` with the motion of its vertices and edges, or `None` if the
    /// face or one of its vertices does not exist.
    fn face(&self, index: usize) -> Option<MovingFace> {
        let f = self.curr.faces.get(index)?;
        let edges = f.edges();
        let edge_motion = |k: usize| -> Option<(Motion, Motion)> {
            let (v0, v1) = edges[k];
            Some((self.motion(v0)?, self.motion(v1)?))
        };
        Some(MovingFace {
            index,
            vertices: [f.v0, f.v1, f.v2],
            motions: [self.motion(f.v0)?, self.motion(f.v1)?, self.motion(f.v2)?],
            edge_motions: [edge_motion(0)?, edge_motion(1)?, edge_motion(2)?],
        })
    }
}

/// A triangle of a [`MovingMesh`].
struct MovingFace {
    /// Index of the face in the mesh.
    index: usize,
    /// Vertex indices of the corners.
    vertices: [usize; 3],
    /// Motion of the corners.
    motions: [Motion; 3],
    /// Motion of the end points of the edges, in the order of
    /// [`TriFace::edges`](super::types::TriFace::edges) (the edge id of edge `k`
    /// of face `f` is `f * 3 + k`).
    edge_motions: [(Motion, Motion); 3],
}

/// Which primitives of a face pair meet.
#[derive(Clone, Copy)]
enum HitKind {
    /// Corner `corner` of the face of mesh A against the face of mesh B.
    VertexOfA { corner: usize },
    /// Corner `corner` of the face of mesh B against the face of mesh A.
    VertexOfB { corner: usize },
    /// Edge `edge_a` of the face of A against edge `edge_b` of the face of B.
    Edges { edge_a: usize, edge_b: usize },
}

/// A hit of one face pair.
#[derive(Clone, Copy)]
struct Hit {
    toi: f64,
    kind: HitKind,
}

/// Time of impact of a moving vertex with a moving triangle.
fn vertex_face_toi(vertex: &Motion, tri: &[Motion; 3], thickness: f64) -> Option<f64> {
    ccd_vertex_face_analytic(
        vertex.start,
        vertex.end,
        tri[0].start,
        tri[0].end,
        tri[1].start,
        tri[1].end,
        tri[2].start,
        tri[2].end,
        thickness,
    )
}

/// The earliest hit of two faces that is earlier than `bound` (any hit if there
/// is no bound), testing the 6 vertex-face and the 9 edge-edge pairs.
fn earliest_face_hit(
    a: &MovingFace,
    b: &MovingFace,
    thickness: f64,
    bound: Option<f64>,
) -> Option<Hit> {
    let mut best: Option<Hit> = None;
    let mut consider = |toi: Option<f64>, kind: HitKind| {
        let Some(toi) = toi else {
            return;
        };
        let limit = best.map(|hit| hit.toi).or(bound);
        if limit.is_none_or(|limit| toi < limit) {
            best = Some(Hit { toi, kind });
        }
    };
    for (corner, vertex) in a.motions.iter().enumerate() {
        consider(
            vertex_face_toi(vertex, &b.motions, thickness),
            HitKind::VertexOfA { corner },
        );
    }
    for (corner, vertex) in b.motions.iter().enumerate() {
        consider(
            vertex_face_toi(vertex, &a.motions, thickness),
            HitKind::VertexOfB { corner },
        );
    }
    for (edge_a, (p0, p1)) in a.edge_motions.iter().enumerate() {
        for (edge_b, (q0, q1)) in b.edge_motions.iter().enumerate() {
            let toi = ccd_edge_edge_analytic(
                p0.start, p0.end, p1.start, p1.end, q0.start, q0.end, q1.start, q1.end, thickness,
            );
            consider(toi, HitKind::Edges { edge_a, edge_b });
        }
    }
    best
}

/// `v` scaled to unit length, or `fallback` if `v` has no direction.
fn unit_or(v: Vec3, fallback: Vec3) -> Vec3 {
    let length = v3_len(v);
    if length > 0.0 && length.is_finite() {
        v3_scale(v, 1.0 / length)
    } else {
        fallback
    }
}

/// Normal (from the triangle toward the vertex) and vertex position at time
/// `toi` of a vertex hitting a triangle.
fn vertex_face_geometry(vertex: &Motion, tri: &[Motion; 3], toi: f64) -> (Vec3, Vec3) {
    let p = vertex.at(toi);
    let [a, b, c] = [tri[0].at(toi), tri[1].at(toi), tri[2].at(toi)];
    let (closest, _, dist_sq) = closest_point_on_triangle(p, a, b, c);
    if dist_sq > EPSILON * EPSILON {
        return (v3_norm(v3_sub(p, closest)), p);
    }
    // The vertex lies on the triangle at the time of impact, so the face normal
    // gives the direction. Keep the orientation the triangle had at the start of
    // the step, then point it to the side the vertex came from; a vertex that
    // starts on the triangle has no such side and the normal opposes its travel.
    let travel = v3_sub(vertex.end, vertex.start);
    let start_normal = unit_or(
        v3_cross(
            v3_sub(tri[1].start, tri[0].start),
            v3_sub(tri[2].start, tri[0].start),
        ),
        [0.0; 3],
    );
    let mut normal = unit_or(
        v3_cross(v3_sub(b, a), v3_sub(c, a)),
        unit_or(start_normal, unit_or(v3_neg(travel), [0.0, 1.0, 0.0])),
    );
    if v3_dot(normal, start_normal) < 0.0 {
        normal = v3_neg(normal);
    }
    let side = v3_dot(start_normal, v3_sub(vertex.start, tri[0].start));
    let orientation = if side.abs() > EPSILON {
        side
    } else {
        -v3_dot(normal, travel)
    };
    if orientation < 0.0 {
        (v3_neg(normal), p)
    } else {
        (normal, p)
    }
}

/// Normal (from edge B toward edge A) and contact position at time `toi` of two
/// edges that meet.
fn edge_edge_geometry(
    edge_a: &(Motion, Motion),
    edge_b: &(Motion, Motion),
    toi: f64,
) -> (Vec3, Vec3) {
    let (p0, p1) = (edge_a.0.at(toi), edge_a.1.at(toi));
    let (q0, q1) = (edge_b.0.at(toi), edge_b.1.at(toi));
    let (_, _, cp, cq, dist_sq) = closest_points_edge_edge(p0, p1, q0, q1);
    let position = v3_scale(v3_add(cp, cq), 0.5);
    if dist_sq > EPSILON * EPSILON {
        return (v3_norm(v3_sub(cp, cq)), position);
    }
    // The edges touch at the time of impact: use their common perpendicular,
    // oriented to the side from which edge A approached. Edges that already
    // touch at the start have no such side; the normal then opposes the motion
    // of edge A relative to edge B.
    let (_, _, cp0, cq0, _) = closest_points_edge_edge(
        edge_a.0.start,
        edge_a.1.start,
        edge_b.0.start,
        edge_b.1.start,
    );
    let came_from = v3_sub(cp0, cq0);
    let common = v3_cross(v3_sub(p1, p0), v3_sub(q1, q0));
    if v3_len(common) <= 0.0 {
        return (unit_or(came_from, [0.0, 1.0, 0.0]), position);
    }
    let normal = v3_norm(common);
    let side = v3_dot(normal, came_from);
    let orientation = if side.abs() > EPSILON {
        side
    } else {
        let travel = |edge: &(Motion, Motion)| {
            let moved = v3_add(
                v3_sub(edge.0.end, edge.0.start),
                v3_sub(edge.1.end, edge.1.start),
            );
            v3_scale(moved, 0.5)
        };
        -v3_dot(normal, v3_sub(travel(edge_a), travel(edge_b)))
    };
    if orientation < 0.0 {
        (v3_neg(normal), position)
    } else {
        (normal, position)
    }
}

/// The contact record of the earliest `hit` of faces `a` and `b`.
fn contact_of(a: &MovingFace, b: &MovingFace, hit: Hit) -> DeformCcdResult {
    match hit.kind {
        HitKind::VertexOfA { corner } => {
            let (normal, position) = vertex_face_geometry(&a.motions[corner], &b.motions, hit.toi);
            DeformCcdResult {
                toi: hit.toi,
                normal,
                position,
                feature_a: a.vertices[corner],
                feature_b: b.index,
                contact_type: DeformContactType::VertexFace,
            }
        }
        HitKind::VertexOfB { corner } => {
            let (toward_vertex, position) =
                vertex_face_geometry(&b.motions[corner], &a.motions, hit.toi);
            DeformCcdResult {
                toi: hit.toi,
                // From the face of A toward the vertex of B; contacts point from
                // B to A.
                normal: v3_neg(toward_vertex),
                position,
                feature_a: a.index,
                feature_b: b.vertices[corner],
                contact_type: DeformContactType::FaceVertex,
            }
        }
        HitKind::Edges { edge_a, edge_b } => {
            let (normal, position) =
                edge_edge_geometry(&a.edge_motions[edge_a], &b.edge_motions[edge_b], hit.toi);
            DeformCcdResult {
                toi: hit.toi,
                normal,
                position,
                feature_a: a.index * 3 + edge_a,
                feature_b: b.index * 3 + edge_b,
                contact_type: DeformContactType::EdgeEdge,
            }
        }
    }
}

/// Run CCD over an entire deformable mesh pair. Returns the earliest contact.
///
/// Mesh A moves from `mesh_a_prev` to `mesh_a_curr` and mesh B from
/// `mesh_b_prev` to `mesh_b_curr` over the step (each pair has the same
/// connectivity; the faces are those of the `curr` meshes), every vertex on a
/// straight line. The result is the contact with the smallest time of impact
/// `toi` in \[0, 1\]: the first time at which a vertex of either mesh comes within
/// `thickness` of a face of the other mesh, or an edge of A within `thickness`
/// of an edge of B. A pair that is already closer than `thickness` at the start
/// has `toi == 0`.
///
/// The result says which primitives met in [`DeformCcdResult::contact_type`]:
/// `feature_a` and `feature_b` index mesh A and mesh B respectively, and the
/// normal points from B to A. Vertices of B hitting faces of A are reported as
/// [`DeformContactType::FaceVertex`], so they are not mistaken for vertices of A.
///
/// The candidate face pairs come from bounding volume hierarchies of the boxes
/// the faces sweep during the step. Use [`ccd_deformable_deformable_swept`] to
/// reuse hierarchies built with [`DeformBvh::build_swept`].
pub fn ccd_deformable_deformable(
    mesh_a_prev: &DeformableMesh,
    mesh_a_curr: &DeformableMesh,
    mesh_b_prev: &DeformableMesh,
    mesh_b_curr: &DeformableMesh,
    thickness: f64,
) -> Option<DeformCcdResult> {
    let bvh_a = DeformBvh::build_swept(mesh_a_prev, mesh_a_curr, thickness);
    let bvh_b = DeformBvh::build_swept(mesh_b_prev, mesh_b_curr, thickness);
    ccd_deformable_deformable_swept(
        mesh_a_prev,
        mesh_a_curr,
        mesh_b_prev,
        mesh_b_curr,
        &bvh_a,
        &bvh_b,
        thickness,
    )
}

/// [`ccd_deformable_deformable`] with prebuilt broad-phase hierarchies.
///
/// `swept_bvh_a` and `swept_bvh_b` must be hierarchies of the swept boxes of
/// mesh A and mesh B for this step and for (at least) the same `thickness`:
/// built with [`DeformBvh::build_swept`] or brought up to date with
/// [`DeformBvh::refit_swept`]. A hierarchy of a single configuration
/// ([`DeformBvh::build`]) does not enclose the motion, and a pair that moves
/// through each other within the step would be missed.
pub fn ccd_deformable_deformable_swept(
    mesh_a_prev: &DeformableMesh,
    mesh_a_curr: &DeformableMesh,
    mesh_b_prev: &DeformableMesh,
    mesh_b_curr: &DeformableMesh,
    swept_bvh_a: &DeformBvh,
    swept_bvh_b: &DeformBvh,
    thickness: f64,
) -> Option<DeformCcdResult> {
    let mesh_a = MovingMesh {
        prev: mesh_a_prev,
        curr: mesh_a_curr,
    };
    let mesh_b = MovingMesh {
        prev: mesh_b_prev,
        curr: mesh_b_curr,
    };
    // Earliest hit so far with the faces that produced it.
    let mut earliest: Option<(MovingFace, MovingFace, Hit)> = None;
    for (face_a, face_b) in swept_bvh_a.query_overlap_pairs(swept_bvh_b) {
        let (Some(a), Some(b)) = (mesh_a.face(face_a), mesh_b.face(face_b)) else {
            continue;
        };
        let bound = earliest.as_ref().map(|(_, _, hit)| hit.toi);
        if let Some(hit) = earliest_face_hit(&a, &b, thickness, bound) {
            earliest = Some((a, b, hit));
        }
    }
    let (a, b, hit) = earliest?;
    Some(contact_of(&a, &b, hit))
}
