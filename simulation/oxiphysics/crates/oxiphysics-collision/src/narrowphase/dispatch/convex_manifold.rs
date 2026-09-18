// Copyright 2026 COOLJAPAN OU (Team KitaSan)
// SPDX-License-Identifier: Apache-2.0

//! One-shot full contact manifolds for general convex polyhedra.
//!
//! Generalises the box-box face-clip manifold ([`super::box_manifold`]) to
//! arbitrary convex hulls. Given a separating axis (typically the EPA penetration
//! normal) and two convex polyhedra, this module
//!
//! 1. picks the **reference face** on one body (most anti-parallel to the axis),
//! 2. picks the **incident face** on the other (most parallel to the axis),
//! 3. Sutherland-Hodgman clips the incident polygon against the reference face's
//!    side planes,
//! 4. keeps the clip points at or below the reference plane,
//! 5. reduces to at most four representative contacts.
//!
//! When neither body presents a clear face to the separating axis (an edge-edge
//! contact), it falls back to a one/two-point edge-closest-point witness.
//!
//! The polygon faces and edges of a [`ConvexHull`] do not depend on its pose, so
//! they are built once per hull and cached on the shape
//! ([`ConvexHull::topology`]); [`convex_shape_manifold`] poses the cached
//! topology with the transform of the shape (an `O(n)` pass over the vertices)
//! instead of extracting the faces again for every contact query.
//!
//! All returned contacts carry a normal pointing FROM body B TOWARD body A (the
//! crate convention). `point_a` lies on body A's surface and `point_b` on body
//! B's surface, so swapping the dispatch order (A<->B) negates the normal and
//! swaps the witness points, leaving the manifold geometrically identical.

use std::borrow::Cow;
use std::collections::HashSet;

use oxiphysics_core::Transform;
use oxiphysics_core::math::Vec3;
use oxiphysics_geometry::{BoxShape, ConvexHull, HullTopology, Shape};

use super::functions::world_hull_vertices;
use super::types::NarrowPhaseResult;
use crate::contact_generation::{
    add3, clip_polygon_by_plane, cross3, dot3, norm3, normalize3, scale3, sub3,
};
use crate::types::{CollisionPair, Contact, ContactManifold};

/// Contacts shallower than this distance below the reference plane are still
/// kept so that resting (near-zero penetration) faces yield a full manifold.
const CONTACT_SLOP: f64 = 1e-3;

/// Reciprocal of the spatial grid used to quantise contact sort keys. The grid
/// (1e-4 m) is far above floating-point noise yet far below the separation of
/// distinct features, so the contact ordering is stable under tiny nudges.
const SORT_QUANTUM_INV: f64 = 1.0e4;

/// A convex polyhedron in world space with polygon faces.
///
/// Faces are stored as vertex-index loops (counter-clockwise when viewed from
/// outside) together with their outward unit normals. This is the input form the
/// face-clip manifold consumes; a triangulated hull is merged into coplanar
/// polygon loops first (see [`Polyhedron::from_hull_vertices`]) so the reference
/// face's side planes bound the full face, not a single triangle.
pub struct Polyhedron {
    /// World-space vertices.
    pub vertices: Vec<[f64; 3]>,
    /// Faces as vertex-index loops into `vertices` (CCW from outside).
    pub faces: Vec<Vec<usize>>,
    /// Outward unit normal per face (parallel to `faces`).
    pub normals: Vec<[f64; 3]>,
}

impl Polyhedron {
    /// Build a world-space polyhedron from an oriented box.
    pub fn from_box(box_shape: &BoxShape, transform: &Transform) -> Polyhedron {
        let local = box_shape.vertex_list();
        let vertices: Vec<[f64; 3]> = local
            .iter()
            .map(|&v| {
                let p = transform.transform_point(&Vec3::from(v));
                [p.x, p.y, p.z]
            })
            .collect();
        let faces: Vec<Vec<usize>> = BoxShape::face_vertex_indices()
            .iter()
            .map(|f| f.to_vec())
            .collect();
        let normals: Vec<[f64; 3]> = BoxShape::face_normals()
            .iter()
            .map(|&ln| {
                let wn = transform.transform_vector(&Vec3::from(ln));
                normalize3([wn.x, wn.y, wn.z])
            })
            .collect();
        Polyhedron {
            vertices,
            faces,
            normals,
        }
    }

    /// Build a world-space polyhedron from a convex hull's world-space vertices.
    ///
    /// The faces are those of [`HullTopology::build`]: near-duplicate points are
    /// dropped, the facet planes come from a quickhull of the point set, and the
    /// points on each plane are gathered and ordered into a CCW polygon loop, so a
    /// facet keeps all of its coplanar points whatever triangulation the hull
    /// builder produced. Returns `None` when the points are degenerate (fewer
    /// than four non-coplanar vertices) so callers can fall back to a point
    /// witness.
    ///
    /// The topology is rebuilt on every call. A [`ConvexHull`] shape keeps it in a
    /// cache instead ([`ConvexHull::topology`]); [`convex_shape_manifold`] poses
    /// that cached topology rather than extracting the faces again.
    pub fn from_hull_vertices(world_vertices: &[[f64; 3]]) -> Option<Polyhedron> {
        let (vertices, faces, normals, _edges) = HullTopology::build(world_vertices)?.into_parts();
        Some(Polyhedron {
            vertices,
            faces,
            normals,
        })
    }

    /// Build a world-space polyhedron from a [`ConvexHull`] shape posed with
    /// `transform`, using the topology cached on the shape
    /// ([`ConvexHull::topology`]): only the vertices and the face normals are
    /// carried over to the world frame, the faces are not extracted again.
    ///
    /// Returns `None` when the hull spans no volume. For a hull that has not been
    /// edited since it was created, the result is the polyhedron of
    /// [`Polyhedron::from_hull_vertices`] on the world-space vertices.
    pub fn from_hull(hull: &ConvexHull, transform: &Transform) -> Option<Polyhedron> {
        let posed = PosedHull::new(hull, transform)?;
        Some(Polyhedron {
            faces: posed.topology.faces().to_vec(),
            vertices: posed.vertices,
            normals: posed.normals,
        })
    }

    /// Borrow the face data the manifold builders read.
    fn view(&self) -> PolyView<'_> {
        PolyView {
            vertices: &self.vertices,
            faces: &self.faces,
            normals: &self.normals,
            edges: None,
        }
    }
}

/// Borrowed face data of a convex polyhedron: everything the manifold builders
/// read, whether it is owned by a [`Polyhedron`] or posed from the topology cached
/// on a hull shape ([`PosedHull`]).
#[derive(Clone, Copy)]
struct PolyView<'a> {
    /// World-space vertices.
    vertices: &'a [[f64; 3]],
    /// Faces as vertex-index loops into `vertices` (CCW from outside).
    faces: &'a [Vec<usize>],
    /// Outward unit normal per face (parallel to `faces`).
    normals: &'a [[f64; 3]],
    /// The undirected edges as index pairs into `vertices`, each listed once, when
    /// they are known (the cached topology of a hull); `None` derives them from
    /// `faces`.
    edges: Option<&'a [[usize; 2]]>,
}

impl PolyView<'_> {
    /// World-space polygon (vertex positions) of face `idx`.
    fn face_polygon(&self, idx: usize) -> Vec<[f64; 3]> {
        self.faces[idx].iter().map(|&k| self.vertices[k]).collect()
    }

    /// Index of the face whose outward normal best matches `dir` (max dot).
    fn most_parallel_face(&self, dir: [f64; 3]) -> usize {
        let mut best = 0;
        let mut best_dot = f64::NEG_INFINITY;
        for (i, n) in self.normals.iter().enumerate() {
            let d = dot3(*n, dir);
            if d > best_dot {
                best_dot = d;
                best = i;
            }
        }
        best
    }

    /// All undirected edges as world-space endpoint pairs, each edge once and in
    /// the direction in which it is first met when walking the faces in order.
    fn edge_segments(&self) -> Vec<([f64; 3], [f64; 3])> {
        if let Some(edges) = self.edges {
            return edges
                .iter()
                .map(|&[a, b]| (self.vertices[a], self.vertices[b]))
                .collect();
        }
        let mut seen: HashSet<(usize, usize)> = HashSet::new();
        let mut edges = Vec::new();
        for face in self.faces {
            let n = face.len();
            for i in 0..n {
                let a = face[i];
                let b = face[(i + 1) % n];
                let key = if a < b { (a, b) } else { (b, a) };
                if seen.insert(key) {
                    edges.push((self.vertices[a], self.vertices[b]));
                }
            }
        }
        edges
    }
}

/// A [`ConvexHull`] posed in world space: the face and edge topology comes from
/// the cache of the shape (built once per hull, see [`ConvexHull::topology`]) and
/// only the vertices and the face normals are carried over to the world frame, in
/// `O(n + f)` per query instead of rebuilding the polyhedron.
struct PosedHull<'a> {
    /// Local-frame faces and edges, borrowed from the hull's cache.
    topology: Cow<'a, HullTopology>,
    /// World-space vertices (parallel to the topology's vertices).
    vertices: Vec<[f64; 3]>,
    /// World-space outward unit normals (parallel to the topology's faces).
    normals: Vec<[f64; 3]>,
}

impl<'a> PosedHull<'a> {
    /// Pose `hull` with `transform`; `None` when the hull spans no volume.
    fn new(hull: &'a ConvexHull, transform: &Transform) -> Option<Self> {
        let topology = hull.topology()?;
        let vertices = topology
            .vertices()
            .iter()
            .map(|&v| {
                let p = transform.transform_point(&Vec3::from(v));
                [p.x, p.y, p.z]
            })
            .collect();
        let normals = topology
            .normals()
            .iter()
            .map(|&n| {
                let w = transform.transform_vector(&Vec3::from(n));
                [w.x, w.y, w.z]
            })
            .collect();
        Some(Self {
            topology,
            vertices,
            normals,
        })
    }

    /// Borrow the posed face data.
    fn view(&self) -> PolyView<'_> {
        PolyView {
            vertices: &self.vertices,
            faces: self.topology.faces(),
            normals: &self.normals,
            edges: Some(self.topology.edges()),
        }
    }
}

/// One body of a polyhedral narrow-phase pair, ready for the manifold builders.
enum ConvexBody<'a> {
    /// Faces built from the world-space vertices of the shape (box corners);
    /// `None` when they are degenerate.
    Built(Option<Polyhedron>),
    /// A hull posed with the topology cached on the shape; `None` when the hull
    /// spans no volume.
    Posed(Option<PosedHull<'a>>),
}

impl<'a> ConvexBody<'a> {
    /// The polyhedral body of `shape` in world space, or `None` when the shape is
    /// not polyhedral (sphere, capsule, mesh, ...).
    fn of(shape: &'a dyn Shape, transform: &Transform) -> Option<Self> {
        if let Some(hull) = shape.as_any().downcast_ref::<ConvexHull>() {
            return Some(Self::Posed(PosedHull::new(hull, transform)));
        }
        let vertices = world_hull_vertices(shape, transform)?;
        Some(Self::Built(Polyhedron::from_hull_vertices(&vertices)))
    }

    /// The face data, or `None` when the body is degenerate.
    fn view(&self) -> Option<PolyView<'_>> {
        match self {
            Self::Built(polyhedron) => polyhedron.as_ref().map(Polyhedron::view),
            Self::Posed(hull) => hull.as_ref().map(PosedHull::view),
        }
    }
}

/// One-shot face-clip manifold for two convex polyhedra given a separating axis.
///
/// `sep_normal` points FROM body B TOWARD body A (the crate convention) and
/// `depth > 0` is the penetration along it. The body owning the face most
/// anti-parallel to the axis becomes the reference (clipping) body; the other
/// contributes the incident face. The incident polygon is Sutherland-Hodgman
/// clipped against the reference face's side planes and the surviving points at
/// or below the reference plane become contacts (reduced to four).
///
/// Returns a [`ContactManifold`] whose normal is `sep_normal` for every contact.
/// An empty `faces_*`/`vertices_*` set yields an empty manifold.
pub fn polyhedron_manifold_from_normal(
    poly_a: &Polyhedron,
    poly_b: &Polyhedron,
    sep_normal: [f64; 3],
    depth: f64,
    pair: CollisionPair,
) -> ContactManifold {
    face_clip_manifold(poly_a.view(), poly_b.view(), sep_normal, depth, pair)
}

/// [`polyhedron_manifold_from_normal`] on borrowed face data.
fn face_clip_manifold(
    poly_a: PolyView<'_>,
    poly_b: PolyView<'_>,
    sep_normal: [f64; 3],
    depth: f64,
    pair: CollisionPair,
) -> ContactManifold {
    let n_ba = normalize3(sep_normal);

    // Reference is the body whose best face is most anti-parallel to the axis.
    // A's outward toward the interface is +n_ba (B->A faces A's surface that
    // looks toward B, i.e. faces along -n_ba); B's outward is -n_ba. Pick the
    // body presenting the flatter face to the separation.
    let a_face = poly_a.most_parallel_face([-n_ba[0], -n_ba[1], -n_ba[2]]);
    let b_face = poly_b.most_parallel_face(n_ba);
    let a_align = dot3(poly_a.normals[a_face], [-n_ba[0], -n_ba[1], -n_ba[2]]);
    let b_align = dot3(poly_b.normals[b_face], n_ba);

    let ref_is_a = a_align >= b_align;
    let (poly_ref, ref_face_idx, poly_inc, ref_outward) = if ref_is_a {
        (poly_a, a_face, poly_b, [-n_ba[0], -n_ba[1], -n_ba[2]])
    } else {
        (poly_b, b_face, poly_a, n_ba)
    };

    // Incident face: most anti-parallel to the reference outward normal.
    let inc_face_idx =
        poly_inc.most_parallel_face([-ref_outward[0], -ref_outward[1], -ref_outward[2]]);

    let ref_poly = poly_ref.face_polygon(ref_face_idx);
    let inc_poly = poly_inc.face_polygon(inc_face_idx);

    let mut contacts: Vec<Contact> =
        clip_face_contacts(&ref_poly, &inc_poly, ref_outward, n_ba, ref_is_a);

    // Fallback: degenerate orientation produced nothing — synthesise one contact
    // at the reference-face centroid straddling the interface.
    if contacts.is_empty() {
        let m = polygon_centroid(&ref_poly);
        let d = depth.max(0.0);
        let pa = add3(m, scale3(n_ba, d * 0.5));
        let pb = sub3(m, scale3(n_ba, d * 0.5));
        contacts.push(Contact::new(
            Vec3::from(pa),
            Vec3::from(pb),
            Vec3::from(n_ba),
            d,
        ));
    }

    let contacts = finalize_contacts(contacts);
    let mut manifold = ContactManifold::new(pair);
    for c in contacts {
        manifold.add_contact(c);
    }
    manifold
}

/// Clip the incident polygon against the reference face and emit contacts.
///
/// `ref_outward` is the reference face's outward normal (reference -> incident);
/// `n_ba` is the contact normal stored on every emitted contact (B->A).
/// `ref_is_a` decides which witness point is on A vs B so both dispatch orders
/// agree (see the box-box manifold for the full argument).
fn clip_face_contacts(
    ref_poly: &[[f64; 3]],
    inc_poly: &[[f64; 3]],
    ref_outward: [f64; 3],
    n_ba: [f64; 3],
    ref_is_a: bool,
) -> Vec<Contact> {
    let mut contacts = Vec::new();
    if ref_poly.is_empty() || inc_poly.is_empty() {
        return contacts;
    }

    let mut clipped = inc_poly.to_vec();
    let nref = ref_poly.len();
    for i in 0..nref {
        if clipped.is_empty() {
            break;
        }
        let es = ref_poly[i];
        let ee = ref_poly[(i + 1) % nref];
        let edge_dir = sub3(ee, es);
        let side_normal = normalize3(cross3(ref_outward, edge_dir));
        clipped = clip_polygon_by_plane(&clipped, es, side_normal);
    }

    for p in &clipped {
        let pen = dot3(sub3(ref_poly[0], *p), ref_outward);
        if pen > -CONTACT_SLOP {
            let depth_point = pen.max(0.0);
            // `p` lies on the incident face; project it onto the reference plane
            // along the outward normal to obtain the reference-surface point.
            let p_ref = add3(*p, scale3(ref_outward, pen));
            let (pa_arr, pb_arr) = if ref_is_a { (p_ref, *p) } else { (*p, p_ref) };
            contacts.push(Contact::new(
                Vec3::from(pa_arr),
                Vec3::from(pb_arr),
                Vec3::from(n_ba),
                depth_point,
            ));
        }
    }

    contacts
}

/// Centroid of a polygon (mean of its vertices). Zero for an empty polygon.
fn polygon_centroid(poly: &[[f64; 3]]) -> [f64; 3] {
    if poly.is_empty() {
        return [0.0, 0.0, 0.0];
    }
    let mut acc = [0.0_f64; 3];
    for p in poly {
        acc = add3(acc, *p);
    }
    scale3(acc, 1.0 / poly.len() as f64)
}

/// Reduce to at most four contacts then sort by a quantised spatial key.
///
/// The reduction keeps the deepest point, the farthest from it, the
/// maximum-area third, and the area-maximising fourth (lowest-index ties). The
/// stable quantised ordering keeps the i-th contact identity steady frame to
/// frame, which is what the manifold cache matches on for warm-start continuity.
fn finalize_contacts(contacts: Vec<Contact>) -> Vec<Contact> {
    let mut contacts = reduce_contacts_to_4(contacts);
    let q = |v: f64| (v * SORT_QUANTUM_INV).round();
    contacts.sort_by(|c1, c2| {
        q(c1.point_b.x)
            .partial_cmp(&q(c2.point_b.x))
            .unwrap_or(std::cmp::Ordering::Equal)
            .then(
                q(c1.point_b.z)
                    .partial_cmp(&q(c2.point_b.z))
                    .unwrap_or(std::cmp::Ordering::Equal),
            )
            .then(
                q(c1.point_b.y)
                    .partial_cmp(&q(c2.point_b.y))
                    .unwrap_or(std::cmp::Ordering::Equal),
            )
    });
    contacts
}

/// Reduce a contact set to at most four representative points.
///
/// Deterministic with lowest-index tie-breaking: (1) deepest, (2) farthest from
/// #1, (3) max-area triangle with {#1,#2}, (4) area-maximising against the
/// triangle {#1,#2,#3}. Four or fewer inputs are returned unchanged.
fn reduce_contacts_to_4(contacts: Vec<Contact>) -> Vec<Contact> {
    if contacts.len() <= 4 {
        return contacts;
    }
    let pts: Vec<Vec3> = contacts.iter().map(|c| c.point()).collect();
    let n = contacts.len();

    let mut i0 = 0;
    let mut best_depth = f64::NEG_INFINITY;
    for (i, c) in contacts.iter().enumerate() {
        if c.depth > best_depth {
            best_depth = c.depth;
            i0 = i;
        }
    }

    let mut i1 = i0;
    let mut best_d = f64::NEG_INFINITY;
    for i in 0..n {
        if i == i0 {
            continue;
        }
        let d = (pts[i] - pts[i0]).norm();
        if d > best_d {
            best_d = d;
            i1 = i;
        }
    }

    let edge = pts[i1] - pts[i0];
    let mut i2 = i0;
    let mut best_area = f64::NEG_INFINITY;
    for i in 0..n {
        if i == i0 || i == i1 {
            continue;
        }
        let area = (edge.cross(&(pts[i] - pts[i0]))).norm();
        if area > best_area {
            best_area = area;
            i2 = i;
        }
    }

    let mut i3 = i0;
    let mut best_area3 = f64::NEG_INFINITY;
    for i in 0..n {
        if i == i0 || i == i1 || i == i2 {
            continue;
        }
        let p = pts[i];
        let a012 = (pts[i1] - pts[i0]).cross(&(p - pts[i0])).norm();
        let a123 = (pts[i2] - pts[i1]).cross(&(p - pts[i1])).norm();
        let a203 = (pts[i0] - pts[i2]).cross(&(p - pts[i2])).norm();
        let total = a012 + a123 + a203;
        if total > best_area3 {
            best_area3 = total;
            i3 = i;
        }
    }

    vec![
        contacts[i0].clone(),
        contacts[i1].clone(),
        contacts[i2].clone(),
        contacts[i3].clone(),
    ]
}

/// Closest points between two segments `[p1,q1]` and `[p2,q2]` (Ericson).
///
/// Returns `(closest_on_seg1, closest_on_seg2, distance)`; degenerate
/// (point-like) segments clamp their parametric coordinates to `[0,1]`.
fn closest_segment_segment(
    p1: [f64; 3],
    q1: [f64; 3],
    p2: [f64; 3],
    q2: [f64; 3],
) -> ([f64; 3], [f64; 3], f64) {
    let d1 = sub3(q1, p1);
    let d2 = sub3(q2, p2);
    let r = sub3(p1, p2);
    let a = dot3(d1, d1);
    let e = dot3(d2, d2);
    let f = dot3(d2, r);
    let eps = 1e-12;

    let s;
    let t;
    if a <= eps && e <= eps {
        s = 0.0;
        t = 0.0;
    } else if a <= eps {
        s = 0.0;
        t = (f / e).clamp(0.0, 1.0);
    } else {
        let c = dot3(d1, r);
        if e <= eps {
            t = 0.0;
            s = (-c / a).clamp(0.0, 1.0);
        } else {
            let b = dot3(d1, d2);
            let denom = a * e - b * b;
            let s0 = if denom != 0.0 {
                ((b * f - c * e) / denom).clamp(0.0, 1.0)
            } else {
                0.0
            };
            let t0 = (b * s0 + f) / e;
            if t0 < 0.0 {
                t = 0.0;
                s = (-c / a).clamp(0.0, 1.0);
            } else if t0 > 1.0 {
                t = 1.0;
                s = ((b - c) / a).clamp(0.0, 1.0);
            } else {
                s = s0;
                t = t0;
            }
        }
    }

    let c1 = add3(p1, scale3(d1, s));
    let c2 = add3(p2, scale3(d2, t));
    let dist = norm3(sub3(c1, c2));
    (c1, c2, dist)
}

/// Edge-edge witness manifold (1 contact) for two convex polyhedra.
///
/// Searches every edge pair for the closest approach whose direction aligns with
/// `sep_normal`, and emits a single contact at that witness. `c1` (on A's edge)
/// becomes `point_a`, `c2` (on B's edge) becomes `point_b`, normal is `sep_normal`
/// (B->A). Returns an empty manifold only if both bodies have no edges.
pub fn edge_edge_manifold(
    poly_a: &Polyhedron,
    poly_b: &Polyhedron,
    sep_normal: [f64; 3],
    depth: f64,
    pair: CollisionPair,
) -> ContactManifold {
    edge_pair_manifold(poly_a.view(), poly_b.view(), sep_normal, depth, pair)
}

/// [`edge_edge_manifold`] on borrowed face data.
fn edge_pair_manifold(
    poly_a: PolyView<'_>,
    poly_b: PolyView<'_>,
    sep_normal: [f64; 3],
    depth: f64,
    pair: CollisionPair,
) -> ContactManifold {
    let n_ba = normalize3(sep_normal);
    let edges_a = poly_a.edge_segments();
    let edges_b = poly_b.edge_segments();

    let mut best_c1 = [0.0_f64, 0.0, 0.0];
    let mut best_c2 = [0.0_f64, 0.0, 0.0];
    let mut best_dist = f64::INFINITY;
    let mut best_aligned_dist = f64::INFINITY;
    let mut have_aligned = false;
    let mut have_any = false;

    for (a0, a1) in &edges_a {
        for (b0, b1) in &edges_b {
            let (c1, c2, dist) = closest_segment_segment(*a0, *a1, *b0, *b1);
            let dir = sub3(c1, c2);
            let alignment = if norm3(dir) > 1e-9 {
                dot3(normalize3(dir), n_ba).abs()
            } else {
                0.0
            };
            if dist < best_dist {
                best_dist = dist;
                if !have_aligned {
                    best_c1 = c1;
                    best_c2 = c2;
                }
                have_any = true;
            }
            if alignment > 0.5 && dist < best_aligned_dist {
                best_aligned_dist = dist;
                best_c1 = c1;
                best_c2 = c2;
                have_aligned = true;
            }
        }
    }

    let mut manifold = ContactManifold::new(pair);
    if have_aligned || have_any {
        manifold.add_contact(Contact::new(
            Vec3::from(best_c1),
            Vec3::from(best_c2),
            Vec3::from(n_ba),
            depth.max(0.0),
        ));
    }
    manifold
}

/// Minimum face/axis alignment for the face-clip path. A face-face or
/// face-vertex contact has at least one body whose best face is *nearly parallel*
/// to the separation axis (dot close to 1). A crossed edge-edge contact presents
/// only oblique faces to the axis on BOTH bodies (e.g. ~0.707 for 45°-tilted
/// boxes), so neither reaches this bar and the edge-edge witness path is used.
const FACE_ALIGN_MIN: f64 = 0.95;

/// Decide whether `sep_normal` is a face contact for at least one body.
///
/// Returns `true` when either body presents a face whose outward normal is nearly
/// parallel to the separation axis (alignment `>= FACE_ALIGN_MIN`) — a genuine
/// face-face or face-vertex contact the clip path handles exactly. When BOTH
/// bodies present only oblique faces to the axis the contact is edge-edge and the
/// caller should use [`edge_edge_manifold`].
fn is_face_contact(poly_a: PolyView<'_>, poly_b: PolyView<'_>, sep_normal: [f64; 3]) -> bool {
    let n_ba = normalize3(sep_normal);
    let a = poly_a.most_parallel_face([-n_ba[0], -n_ba[1], -n_ba[2]]);
    let b = poly_b.most_parallel_face(n_ba);
    let a_align = dot3(poly_a.normals[a], [-n_ba[0], -n_ba[1], -n_ba[2]]);
    let b_align = dot3(poly_b.normals[b], n_ba);
    a_align >= FACE_ALIGN_MIN || b_align >= FACE_ALIGN_MIN
}

/// Separating-axis data seeding a convex manifold, from a GJK/EPA query.
///
/// `normal` points FROM body B TOWARD body A (B->A) and `depth > 0` is the
/// penetration along it. `point_a`/`point_b` are the EPA witness on each body,
/// used as the single-point fallback when a hull is too degenerate to build a
/// polygon polyhedron.
#[derive(Debug, Clone, Copy)]
pub struct ConvexSeparation {
    /// Separating/penetration normal, B->A.
    pub normal: [f64; 3],
    /// Penetration depth along `normal` (>= 0).
    pub depth: f64,
    /// EPA witness point on body A (fallback).
    pub point_a: [f64; 3],
    /// EPA witness point on body B (fallback).
    pub point_b: [f64; 3],
}

/// One-shot full manifold for two convex hulls given world-space vertices and a
/// separating axis (typically the EPA penetration normal, B->A).
///
/// Builds polygon polyhedra for both hulls, then routes to the face-clip path
/// ([`polyhedron_manifold_from_normal`]) or, when neither body presents a face to
/// the axis, the edge-edge witness ([`edge_edge_manifold`]). Falls back to the
/// EPA witness point ([`ConvexSeparation::point_a`]/`point_b`) when a hull is
/// degenerate (planar / too few points).
///
/// The faces are extracted from the vertices on every call
/// ([`Polyhedron::from_hull_vertices`]); for [`ConvexHull`] shapes
/// [`convex_shape_manifold`] reuses the topology cached on the shape instead.
pub fn convex_hull_manifold_from_vertices(
    verts_a: &[[f64; 3]],
    verts_b: &[[f64; 3]],
    sep: ConvexSeparation,
    pair: CollisionPair,
) -> ContactManifold {
    let poly_a = Polyhedron::from_hull_vertices(verts_a);
    let poly_b = Polyhedron::from_hull_vertices(verts_b);
    polyhedra_manifold(
        poly_a.as_ref().map(Polyhedron::view),
        poly_b.as_ref().map(Polyhedron::view),
        sep,
        pair,
    )
}

/// One-shot full manifold for a pair of polyhedral shapes (a [`BoxShape`] or a
/// [`ConvexHull`] on either side) given a separating axis, typically the EPA
/// penetration normal (B->A).
///
/// The result is that of [`convex_hull_manifold_from_vertices`] on the
/// world-space vertices of the two shapes, but a [`ConvexHull`] does not have its
/// faces extracted again: the face and edge topology is built once per hull and
/// cached on the shape ([`ConvexHull::topology`]), so a query only poses the
/// cached vertices and normals with the shape's transform. A box has no
/// topology to cache beyond its eight corners and is extracted per query as
/// before.
///
/// Returns `None` when either shape is not polyhedral (sphere, capsule, mesh,
/// ...); the caller then keeps the single-point GJK/EPA contact. A polyhedral
/// shape that spans no volume (a flat hull) yields the EPA witness point, as in
/// [`convex_hull_manifold_from_vertices`].
pub fn convex_shape_manifold(
    shape_a: &dyn Shape,
    transform_a: &Transform,
    shape_b: &dyn Shape,
    transform_b: &Transform,
    sep: ConvexSeparation,
    pair: CollisionPair,
) -> Option<ContactManifold> {
    let body_a = ConvexBody::of(shape_a, transform_a)?;
    let body_b = ConvexBody::of(shape_b, transform_b)?;
    Some(polyhedra_manifold(body_a.view(), body_b.view(), sep, pair))
}

/// Route a pair of polyhedra to the face-clip manifold or, when neither presents
/// a face to the separating axis, to the edge-edge witness; a missing polyhedron
/// (degenerate body) keeps the single EPA witness point.
fn polyhedra_manifold(
    poly_a: Option<PolyView<'_>>,
    poly_b: Option<PolyView<'_>>,
    sep: ConvexSeparation,
    pair: CollisionPair,
) -> ContactManifold {
    let n_ba = normalize3(sep.normal);
    match (poly_a, poly_b) {
        (Some(poly_a), Some(poly_b)) => {
            if is_face_contact(poly_a, poly_b, n_ba) {
                face_clip_manifold(poly_a, poly_b, n_ba, sep.depth, pair)
            } else {
                edge_pair_manifold(poly_a, poly_b, n_ba, sep.depth, pair)
            }
        }
        _ => {
            // Degenerate hull (planar / too few points): keep the EPA point.
            let mut manifold = ContactManifold::new(pair);
            manifold.add_contact(Contact::new(
                Vec3::from(sep.point_a),
                Vec3::from(sep.point_b),
                Vec3::from(n_ba),
                sep.depth.max(0.0),
            ));
            manifold
        }
    }
}

/// Build a full convex manifold for a convex shape pair from a GJK/EPA result.
///
/// `verts_*` are the world-space hull vertices of each body (box corners or
/// convex-hull vertices); `sep` carries the EPA normal/depth and fallback
/// witness. Builds polyhedra and runs the face-clip / edge-edge path, returning a
/// `NarrowPhaseResult` with the full manifold (normal B->A) or `separated` if
/// empty. The narrow-phase dispatcher itself uses [`convex_shape_manifold`],
/// which reuses the topology cached on hull shapes.
pub fn convex_pair_manifold(
    verts_a: &[[f64; 3]],
    verts_b: &[[f64; 3]],
    sep: ConvexSeparation,
    pair: CollisionPair,
) -> NarrowPhaseResult {
    let manifold = convex_hull_manifold_from_vertices(verts_a, verts_b, sep, pair);
    if manifold.is_empty() {
        NarrowPhaseResult::separated()
    } else {
        NarrowPhaseResult::contact(manifold)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use oxiphysics_core::Transform;

    fn cube_vertices(center: [f64; 3], half: f64) -> Vec<[f64; 3]> {
        let h = half;
        let mut v = Vec::new();
        for sx in [-1.0, 1.0] {
            for sy in [-1.0, 1.0] {
                for sz in [-1.0, 1.0] {
                    v.push([center[0] + sx * h, center[1] + sy * h, center[2] + sz * h]);
                }
            }
        }
        v
    }

    #[test]
    fn test_box_polyhedron_has_6_quads() {
        let b = BoxShape::new(Vec3::new(0.5, 0.5, 0.5));
        let t = Transform::from_position(Vec3::new(0.0, 0.0, 0.0));
        let poly = Polyhedron::from_box(&b, &t);
        assert_eq!(poly.faces.len(), 6);
        assert_eq!(poly.normals.len(), 6);
        for f in &poly.faces {
            assert_eq!(f.len(), 4);
        }
    }

    #[test]
    fn test_hull_cube_merges_to_6_quads() {
        // A cube point cloud triangulates into 12 triangles; coplanar merge must
        // recover the 6 square faces (4 vertices each).
        let verts = cube_vertices([0.0, 0.0, 0.0], 0.5);
        let poly = Polyhedron::from_hull_vertices(&verts).expect("cube hull");
        assert_eq!(poly.faces.len(), 6, "cube should merge to 6 quad faces");
        for f in &poly.faces {
            assert_eq!(f.len(), 4, "each merged face is a quad");
        }
    }

    #[test]
    fn test_hull_face_normals_unit_and_axis_aligned() {
        let verts = cube_vertices([0.0, 0.0, 0.0], 0.5);
        let poly = Polyhedron::from_hull_vertices(&verts).expect("cube hull");
        for n in &poly.normals {
            assert!((norm3(*n) - 1.0).abs() < 1e-9, "unit normal");
            // Each axis-aligned cube normal has exactly one near-unit component.
            let comps = [n[0].abs(), n[1].abs(), n[2].abs()];
            let big = comps.iter().filter(|&&c| c > 0.9).count();
            assert_eq!(big, 1, "axis-aligned face normal");
        }
    }

    #[test]
    fn test_two_cubes_face_to_face_full_manifold() {
        // Cube A centred at y=0.99, cube B at y=0 (half=0.5): overlap 0.01 along y.
        // Separation normal B->A points +y; expect a 4-point manifold.
        let verts_a = cube_vertices([0.0, 0.99, 0.0], 0.5);
        let verts_b = cube_vertices([0.0, 0.0, 0.0], 0.5);
        let manifold = convex_hull_manifold_from_vertices(
            &verts_a,
            &verts_b,
            ConvexSeparation {
                normal: [0.0, 1.0, 0.0],
                depth: 0.01,
                point_a: [0.0, 0.49, 0.0],
                point_b: [0.0, 0.5, 0.0],
            },
            CollisionPair::new(0, 1),
        );
        assert_eq!(manifold.contacts.len(), 4, "face-face -> 4 contacts");
        for c in &manifold.contacts {
            assert!(c.normal.y > 0.999, "normal points +y");
            assert!(c.depth >= 0.0);
        }
    }

    #[test]
    fn test_face_to_face_both_orders_consistent() {
        let verts_a = cube_vertices([0.0, 0.99, 0.0], 0.5);
        let verts_b = cube_vertices([0.0, 0.0, 0.0], 0.5);
        let m1 = convex_hull_manifold_from_vertices(
            &verts_a,
            &verts_b,
            ConvexSeparation {
                normal: [0.0, 1.0, 0.0],
                depth: 0.01,
                point_a: [0.0, 0.49, 0.0],
                point_b: [0.0, 0.5, 0.0],
            },
            CollisionPair::new(0, 1),
        );
        // Swapped order: A and B exchange, normal negates (B->A becomes -y).
        let m2 = convex_hull_manifold_from_vertices(
            &verts_b,
            &verts_a,
            ConvexSeparation {
                normal: [0.0, -1.0, 0.0],
                depth: 0.01,
                point_a: [0.0, 0.5, 0.0],
                point_b: [0.0, 0.49, 0.0],
            },
            CollisionPair::new(1, 0),
        );
        assert_eq!(m1.contacts.len(), m2.contacts.len());
        for c1 in &m1.contacts {
            let p1 = c1.point();
            let best = m2
                .contacts
                .iter()
                .min_by(|a, b| {
                    (p1 - a.point())
                        .norm()
                        .partial_cmp(&(p1 - b.point()).norm())
                        .unwrap_or(std::cmp::Ordering::Equal)
                })
                .expect("nonempty");
            assert!((best.normal + c1.normal).norm() < 1e-6, "normals opposite");
            assert!((c1.point_a - best.point_b).norm() < 1e-6, "A<->B witness");
            assert!((c1.point_b - best.point_a).norm() < 1e-6, "A<->B witness");
        }
    }

    #[test]
    fn test_polyhedron_edges_cube_has_12() {
        let verts = cube_vertices([0.0, 0.0, 0.0], 0.5);
        let poly = Polyhedron::from_hull_vertices(&verts).expect("cube hull");
        let edges = poly.view().edge_segments();
        assert_eq!(edges.len(), 12, "cube has 12 unique edges");
    }

    fn hex_prism_verts(c: [f64; 3], r: f64, hh: f64) -> Vec<[f64; 3]> {
        let mut v = Vec::new();
        for k in 0..6 {
            let ang = std::f64::consts::TAU * (k as f64) / 6.0;
            let x = c[0] + r * ang.cos();
            let z = c[2] + r * ang.sin();
            v.push([x, c[1] + hh, z]);
            v.push([x, c[1] - hh, z]);
        }
        v
    }

    #[test]
    fn test_hex_prism_extracts_8_faces_with_caps() {
        // A hexagonal prism has 8 faces: a +y and a -y hexagon cap (6 verts each)
        // plus 6 rectangular side faces (4 verts each). The robust extractor must
        // recover all of them — this is the case the triangle-soup hull mangled.
        let verts = hex_prism_verts([0.0, 0.0, 0.0], 1.0, 0.5);
        let poly = Polyhedron::from_hull_vertices(&verts).expect("hex hull");
        assert_eq!(poly.faces.len(), 8, "hex prism has 8 faces");

        let top = poly
            .normals
            .iter()
            .position(|n| n[1] > 0.99)
            .expect("a +y cap face exists");
        assert_eq!(poly.faces[top].len(), 6, "top cap is a hexagon");

        let bottom = poly
            .normals
            .iter()
            .position(|n| n[1] < -0.99)
            .expect("a -y cap face exists");
        assert_eq!(poly.faces[bottom].len(), 6, "bottom cap is a hexagon");

        let sides = poly.normals.iter().filter(|n| n[1].abs() < 0.01).count();
        assert_eq!(sides, 6, "six vertical side faces");
    }

    #[test]
    fn test_hex_hex_face_to_face_manifold() {
        let lower = hex_prism_verts([0.0, 0.0, 0.0], 1.0, 0.5);
        let upper = hex_prism_verts([0.0, 0.99, 0.0], 1.0, 0.5);
        let m = convex_hull_manifold_from_vertices(
            &upper,
            &lower,
            ConvexSeparation {
                normal: [0.0, 1.0, 0.0],
                depth: 0.01,
                point_a: [0.0, 0.49, 0.0],
                point_b: [0.0, 0.5, 0.0],
            },
            CollisionPair::new(0, 1),
        );
        // Hexagon-on-hexagon clip yields a 6-point footprint, reduced to 4.
        assert_eq!(m.contacts.len(), 4, "hex face-to-face -> 4-point manifold");
        for c in &m.contacts {
            assert!(c.normal.y > 0.999, "normal +y");
        }
    }

    // ── face extraction (the algorithm and its oracle test live in
    //    `oxiphysics_geometry::HullTopology`) ─────────────────────────────────

    /// `count` points spread evenly over a sphere (Fibonacci lattice): all of them
    /// are hull vertices and no four are coplanar.
    fn sphere_vertices(count: usize, radius: f64) -> Vec<[f64; 3]> {
        let golden_angle = std::f64::consts::PI * (3.0 - 5.0_f64.sqrt());
        (0..count)
            .map(|i| {
                let y = 1.0 - 2.0 * (i as f64 + 0.5) / count as f64;
                let ring = (1.0 - y * y).sqrt();
                let phi = golden_angle * i as f64;
                [
                    radius * ring * phi.cos(),
                    radius * y,
                    radius * ring * phi.sin(),
                ]
            })
            .collect()
    }

    /// The 27 points of a 3x3x3 lattice: corners, edge midpoints, face centres
    /// and the centre (many coplanar and collinear points, one interior point).
    fn lattice_vertices() -> Vec<[f64; 3]> {
        let mut v = Vec::new();
        for x in [-0.5, 0.0, 0.5] {
            for y in [-0.5, 0.0, 0.5] {
                for z in [-0.5, 0.0, 0.5] {
                    v.push([x, y, z]);
                }
            }
        }
        v
    }

    #[test]
    fn test_lattice_faces_gather_every_coplanar_point() {
        let points = lattice_vertices();
        let poly = Polyhedron::from_hull_vertices(&points).expect("lattice hull");
        assert_eq!(poly.faces.len(), 6);
        for face in &poly.faces {
            // Corners, edge midpoints and the face centre: nine points per face,
            // and the interior centre point belongs to none.
            assert_eq!(face.len(), 9);
            assert!(face.iter().all(|&vi| points[vi] != [0.0, 0.0, 0.0]));
        }
    }

    #[test]
    fn test_round_hull_extraction_is_a_valid_triangulated_polyhedron() {
        // 300 points: the old cubic triple enumeration needed minutes here.
        let points = sphere_vertices(300, 1.0);
        let poly = Polyhedron::from_hull_vertices(&points).expect("sphere hull");
        // All points are extreme and in general position: every face is a
        // triangle and Euler's formula gives F = 2 V - 4.
        assert_eq!(poly.faces.len(), 2 * 300 - 4);
        assert!(poly.faces.iter().all(|f| f.len() == 3));
        for (face, normal) in poly.faces.iter().zip(&poly.normals) {
            assert!((norm3(*normal) - 1.0).abs() < 1e-9);
            let offset = dot3(poly.vertices[face[0]], *normal);
            // Supporting plane, outward: every vertex is below it, the origin
            // (inside) is strictly below.
            assert!(
                poly.vertices
                    .iter()
                    .all(|v| dot3(*v, *normal) <= offset + 1e-7)
            );
            assert!(offset > 0.0);
        }
    }
}

#[cfg(test)]
mod cache_tests;
