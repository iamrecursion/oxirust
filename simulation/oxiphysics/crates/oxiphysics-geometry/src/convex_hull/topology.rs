// Copyright 2026 COOLJAPAN OU (Team KitaSan)
// SPDX-License-Identifier: Apache-2.0

//! Polygonal face and edge structure of a convex hull.
//!
//! A [`ConvexHull`](super::types::ConvexHull) stores only a vertex cloud, but
//! contact generation needs more: the polygon faces with their outward normals
//! (the face-clip manifold) and the edges (the edge-edge witness).
//! [`HullTopology`] derives them from the cloud, and
//! [`ConvexHull::topology`](super::types::ConvexHull::topology) caches the
//! result on the hull, so a query only has to carry the rigid pose of the hull
//! over to the cached vertices and normals instead of rebuilding the polyhedron.

use std::collections::{HashMap, HashSet};

use super::types::ConvexHull3D;

/// Two face normals whose dot product exceeds this are treated as coplanar when
/// merging triangulated hull faces into polygon loops.
const COPLANAR_DOT: f64 = 1.0 - 1e-6;

/// Two points closer than this (Euclidean) collapse to one vertex.
const DUPLICATE_DISTANCE: f64 = 1e-9;

/// Cross products shorter than this mark a collinear triple of points.
const COLLINEAR_CROSS: f64 = 1e-12;

/// Thickness of a facet plane relative to the extent of the cloud: points closer
/// to the plane than this lie on the facet.
const PLANE_THICKNESS: f64 = 1e-7;

/// Extracted convex-hull face data: parallel `(face vertex-loops, outward unit
/// normals)`.
type FaceSet = (Vec<Vec<usize>>, Vec<[f64; 3]>);

/// Polygon faces and edges of the convex polyhedron spanned by a point set.
///
/// The polyhedron is described in the frame of the point set it was built from
/// (for a [`ConvexHull`](super::types::ConvexHull): its local frame), so the
/// description is unaffected by the pose of the body and can be computed once
/// per shape.
///
/// * [`vertices`](Self::vertices) are the input points with near-duplicates
///   removed (the first occurrence is kept, in input order).  Interior points
///   and points on the boundary that are not corners are kept as well: a facet
///   keeps every point that lies on its plane.
/// * [`faces`](Self::faces) are vertex-index loops into `vertices`, listed in a
///   canonical order and wound counter-clockwise seen from outside.  A face is a
///   full polygon: triangles of the hull that lie in one plane are merged.
/// * [`normals`](Self::normals) holds the outward unit normal of every face.
/// * [`edges`](Self::edges) lists every undirected edge of the faces once, in
///   the direction in which it is first met when walking the faces in order.
#[derive(Debug, Clone, PartialEq)]
pub struct HullTopology {
    vertices: Vec<[f64; 3]>,
    faces: Vec<Vec<usize>>,
    normals: Vec<[f64; 3]>,
    edges: Vec<[usize; 2]>,
}

impl HullTopology {
    /// Build the topology of the convex hull of `points`.
    ///
    /// The facet planes come from a quickhull of the points ([`ConvexHull3D`],
    /// expected `O(n log n)`); hull triangles whose planes agree are one facet.
    /// Every input point lying on a facet plane, hull vertex or not, is then
    /// gathered onto that facet (one `O(n)` pass per facet) and ordered into a
    /// counter-clockwise loop about the outward normal, so a facet carries all of
    /// its coplanar points (the cap of a prism, the extra points on a box face)
    /// however the hull triangulated it.  Overall the build is quadratic in the
    /// number of points (duplicate removal, one pass per facet), which is why
    /// [`ConvexHull::topology`](super::types::ConvexHull::topology) does it once
    /// per hull.
    ///
    /// Each facet plane is defined by the first (lowest-index) non-collinear
    /// triple of its points and the faces are listed in the order of those
    /// triples, so the result does not depend on how the hull builder happened
    /// to triangulate.
    ///
    /// Returns `None` if the points are degenerate: a non-finite coordinate,
    /// fewer than four distinct points, or no four points that span a volume.
    /// It also returns `None` if the hull builder's answer is inconsistent (its
    /// triangles do not form a closed surface, or a facet plane does not support
    /// every point): a topology that may miss a facet is worse than none, and
    /// the narrow phase then keeps the single EPA contact.
    pub fn build(points: &[[f64; 3]]) -> Option<Self> {
        if points.iter().flatten().any(|c| !c.is_finite()) {
            return None;
        }
        let vertices = dedup_vertices(points);
        if vertices.len() < 4 {
            return None;
        }
        let (faces, normals) = extract_convex_faces(&vertices)?;
        if faces.len() < 4 {
            return None;
        }
        let edges = unique_edges(&faces);
        Some(Self {
            vertices,
            faces,
            normals,
            edges,
        })
    }

    /// The distinct points the faces index into.
    pub fn vertices(&self) -> &[[f64; 3]] {
        &self.vertices
    }

    /// Faces as vertex-index loops into [`vertices`](Self::vertices),
    /// counter-clockwise seen from outside.
    pub fn faces(&self) -> &[Vec<usize>] {
        &self.faces
    }

    /// Outward unit normal of every face (parallel to [`faces`](Self::faces)).
    pub fn normals(&self) -> &[[f64; 3]] {
        &self.normals
    }

    /// Every undirected edge once, as a pair of indices into
    /// [`vertices`](Self::vertices).
    pub fn edges(&self) -> &[[usize; 2]] {
        &self.edges
    }

    /// Split into `(vertices, faces, normals, edges)` without copying.
    pub fn into_parts(self) -> TopologyParts {
        (self.vertices, self.faces, self.normals, self.edges)
    }
}

/// The owned pieces of a [`HullTopology`]: `(vertices, faces, normals, edges)`.
pub type TopologyParts = (
    Vec<[f64; 3]>,
    Vec<Vec<usize>>,
    Vec<[f64; 3]>,
    Vec<[usize; 2]>,
);

fn dot(a: [f64; 3], b: [f64; 3]) -> f64 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

fn cross(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

fn norm(a: [f64; 3]) -> f64 {
    dot(a, a).sqrt()
}

fn normalize(a: [f64; 3]) -> [f64; 3] {
    let n = norm(a);
    if n < 1e-15 {
        return [0.0, 1.0, 0.0];
    }
    [a[0] / n, a[1] / n, a[2] / n]
}

fn add(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [a[0] + b[0], a[1] + b[1], a[2] + b[2]]
}

fn sub(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}

fn scale(a: [f64; 3], s: f64) -> [f64; 3] {
    [a[0] * s, a[1] * s, a[2] * s]
}

/// Remove near-duplicate points so degenerate triples don't spawn bad planes.
///
/// Two points within [`DUPLICATE_DISTANCE`] (Euclidean) collapse to one; the
/// first occurrence is kept.  Convex-hull face extraction needs distinct
/// vertices, and physics hulls occasionally carry repeated points.
fn dedup_vertices(input: &[[f64; 3]]) -> Vec<[f64; 3]> {
    let mut out: Vec<[f64; 3]> = Vec::with_capacity(input.len());
    for &p in input {
        if !out.iter().any(|&q| norm(sub(p, q)) < DUPLICATE_DISTANCE) {
            out.push(p);
        }
    }
    out
}

/// Extract convex polygon faces from a vertex set.
///
/// Returns `(faces, outward_normals)`, or `None` if the point set is degenerate
/// (e.g. all coplanar) and no valid face plane is found.  See
/// [`HullTopology::build`] for how the faces are derived.
fn extract_convex_faces(vertices: &[[f64; 3]]) -> Option<FaceSet> {
    let hull = ConvexHull3D::build(vertices)?;
    facets_from_hull(vertices, &hull)
}

/// The facets of `vertices` given the triangulated `hull` of the same points.
///
/// The hull only proposes the facet planes; every plane is checked against all
/// of `vertices` and the facet takes every point on it.  Returns `None` if `hull`
/// cannot be trusted: its triangles are not a closed surface, or one of its
/// planes has a point of `vertices` in front of it (a vertex it left out).
fn facets_from_hull(vertices: &[[f64; 3]], hull: &ConvexHull3D) -> Option<FaceSet> {
    let n = vertices.len();
    // Interior reference point: the centroid lies strictly inside a non-degenerate
    // hull, used to orient each face normal outward.
    let mut centroid = [0.0_f64; 3];
    for v in vertices {
        centroid = add(centroid, *v);
    }
    centroid = scale(centroid, 1.0 / n as f64);

    let extent = vertices
        .iter()
        .map(|v| norm(sub(*v, centroid)))
        .fold(0.0_f64, f64::max)
        .max(1.0);
    let plane_eps = PLANE_THICKNESS * extent;

    // Outward plane `(unit normal, offset)` through three points.
    let outward_plane = |a: [f64; 3], b: [f64; 3], c: [f64; 3]| -> Option<([f64; 3], f64)> {
        let raw = cross(sub(b, a), sub(c, a));
        if norm(raw) < COLLINEAR_CROSS {
            return None; // collinear triple
        }
        let mut normal = normalize(raw);
        if dot(sub(centroid, a), normal) > 0.0 {
            normal = scale(normal, -1.0);
        }
        Some((normal, dot(a, normal)))
    };
    // Indices of the points on a plane, in ascending order.
    let on_plane = |normal: [f64; 3], offset: f64| -> Vec<usize> {
        (0..n)
            .filter(|&vi| (dot(vertices[vi], normal) - offset).abs() < plane_eps)
            .collect()
    };

    // One plane per group of coplanar hull triangles.  They are only complete if
    // the triangles cover the whole boundary, i.e. form a closed surface (as
    // `ConvexHull3D::build` promises).
    if !is_closed_surface(&hull.faces) {
        return None;
    }
    let mut planes: Vec<([f64; 3], f64)> = Vec::new();
    for tri in &hull.faces {
        let (Some(&a), Some(&b), Some(&c)) = (
            hull.vertices.get(tri[0]),
            hull.vertices.get(tri[1]),
            hull.vertices.get(tri[2]),
        ) else {
            continue;
        };
        let Some((normal, offset)) = outward_plane(a, b, c) else {
            continue;
        };
        let known = planes
            .iter()
            .any(|(m, d)| dot(*m, normal) > COPLANAR_DOT && (d - offset).abs() < plane_eps);
        if !known {
            planes.push((normal, offset));
        }
    }

    // `(first triple, outward normal, CCW loop)` per facet.
    let mut found: Vec<([usize; 3], [f64; 3], Vec<usize>)> = Vec::new();
    for (hull_normal, hull_offset) in planes {
        let approx = on_plane(hull_normal, hull_offset);
        if approx.len() < 3 {
            continue;
        }
        // Canonical plane: through the first non-collinear triple of the facet's
        // points. It is kept if it still supports every point; otherwise (a nearly
        // collinear triple tilting the plane) the hull's own plane is used.
        let canonical = first_noncollinear_triple(vertices, &approx).and_then(|triple| {
            let [i, j, k] = triple;
            let (normal, offset) = outward_plane(vertices[i], vertices[j], vertices[k])?;
            let supporting = vertices
                .iter()
                .all(|v| dot(*v, normal) <= offset + plane_eps);
            let members = on_plane(normal, offset);
            (supporting && members.len() >= 3).then_some((triple, normal, members))
        });
        let (key, normal, members) = match canonical {
            Some(facet) => facet,
            None => {
                // The hull's own plane takes over, so it has to be a real facet
                // plane: no point may lie in front of it.
                let supporting = vertices
                    .iter()
                    .all(|v| dot(*v, hull_normal) <= hull_offset + plane_eps);
                if !supporting {
                    return None;
                }
                (
                    [approx[0], approx[1], approx[2]],
                    hull_normal,
                    approx.clone(),
                )
            }
        };
        let loop_idx = order_face_ccw(vertices, &members, normal);
        found.push((key, normal, loop_idx));
    }
    found.sort_by_key(|entry| entry.0);

    if found.is_empty() {
        None
    } else {
        let mut faces = Vec::with_capacity(found.len());
        let mut normals = Vec::with_capacity(found.len());
        for (_, normal, loop_idx) in found {
            faces.push(loop_idx);
            normals.push(normal);
        }
        Some((faces, normals))
    }
}

/// Whether `triangles` form a closed surface: every undirected edge is shared by
/// exactly two triangles (so no triangle is missing, doubled or dangling).
fn is_closed_surface(triangles: &[[usize; 3]]) -> bool {
    let mut uses: HashMap<(usize, usize), usize> = HashMap::new();
    for &[a, b, c] in triangles {
        for (p, q) in [(a, b), (b, c), (c, a)] {
            let key = if p < q { (p, q) } else { (q, p) };
            *uses.entry(key).or_insert(0) += 1;
        }
    }
    !uses.is_empty() && uses.values().all(|&count| count == 2)
}

/// The lexicographically first triple `(i, j, k)` of `indices` (ascending, so
/// `i < j < k`) whose points are not collinear, or `None` if they all are.
fn first_noncollinear_triple(vertices: &[[f64; 3]], indices: &[usize]) -> Option<[usize; 3]> {
    for (pi, &i) in indices.iter().enumerate() {
        for (pj, &j) in indices.iter().enumerate().skip(pi + 1) {
            for &k in indices.iter().skip(pj + 1) {
                let raw = cross(sub(vertices[j], vertices[i]), sub(vertices[k], vertices[i]));
                if norm(raw) >= COLLINEAR_CROSS {
                    return Some([i, j, k]);
                }
            }
        }
    }
    None
}

/// Order coplanar face vertices counter-clockwise about `outward_normal`.
///
/// Projects each point onto the face plane's tangent basis and sorts by polar
/// angle around the face centroid, giving a simple (non-self-intersecting) convex
/// loop wound CCW when viewed from outside (along `+outward_normal`).
fn order_face_ccw(
    vertices: &[[f64; 3]],
    on_plane: &[usize],
    outward_normal: [f64; 3],
) -> Vec<usize> {
    let mut center = [0.0_f64; 3];
    for &vi in on_plane {
        center = add(center, vertices[vi]);
    }
    center = scale(center, 1.0 / on_plane.len() as f64);

    // Tangent basis (u, v) in the face plane with u x v = outward_normal.
    let seed = sub(vertices[on_plane[0]], center);
    let u = if norm(seed) > 1e-12 {
        normalize(seed)
    } else {
        // Degenerate seed; pick any in-plane direction.
        let trial = if outward_normal[0].abs() < 0.9 {
            [1.0, 0.0, 0.0]
        } else {
            [0.0, 1.0, 0.0]
        };
        normalize(sub(
            trial,
            scale(outward_normal, dot(trial, outward_normal)),
        ))
    };
    let v = cross(outward_normal, u);

    let mut ordered: Vec<usize> = on_plane.to_vec();
    ordered.sort_by(|&p, &q| {
        let dp = sub(vertices[p], center);
        let dq = sub(vertices[q], center);
        let ap = dot(dp, v).atan2(dot(dp, u));
        let aq = dot(dq, v).atan2(dot(dq, u));
        ap.partial_cmp(&aq).unwrap_or(std::cmp::Ordering::Equal)
    });
    ordered
}

/// Every undirected edge of the face loops once, in the direction in which it is
/// first met when walking the faces in order.
fn unique_edges(faces: &[Vec<usize>]) -> Vec<[usize; 2]> {
    let mut seen: HashSet<(usize, usize)> = HashSet::new();
    let mut edges = Vec::new();
    for face in faces {
        let len = face.len();
        for (i, &a) in face.iter().enumerate() {
            let b = face[(i + 1) % len];
            let key = if a < b { (a, b) } else { (b, a) };
            if seen.insert(key) {
                edges.push([a, b]);
            }
        }
    }
    edges
}

#[cfg(test)]
mod tests;
