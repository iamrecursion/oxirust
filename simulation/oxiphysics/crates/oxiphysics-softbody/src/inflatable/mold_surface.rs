// Copyright 2026 COOLJAPAN OU (Team KitaSan)
// SPDX-License-Identifier: Apache-2.0

//! Closest-point queries and inside/outside classification against a
//! triangulated mold surface.
//!
//! `MoldSurface` is what [`BlowMolding::mold_collision`] runs on. For a query
//! point `p` it finds the closest point `q` of the mold surface and decides on
//! which side of the surface `p` lies:
//!
//! * `q` comes from a bounding-volume hierarchy over the triangles and from
//!   Ericson's closest-point-on-triangle routine, which also reports the
//!   closest *feature* of the triangle (a corner, an edge or the interior of
//!   the face).
//! * The side is the sign of `(p - q) . n`, where `n` is the angle-weighted
//!   pseudo-normal of that feature (Baerentzen and Aanaes, "Signed distance
//!   computation using the angle weighted pseudonormal", IEEE TVCG 11(3),
//!   2005). Unlike the face normal of the nearest triangle, the pseudo-normal
//!   gives the right answer next to convex and concave edges and corners, so
//!   the classification is exact for every closed, consistently oriented
//!   triangle mesh, convex or not.
//!
//! The pseudo-normals point away from the region the mold encloses, so a
//! point is *outside* the mold when `(p - q) . n > 0`. A closed mesh is turned
//! around automatically when its signed volume is negative (its triangles wind
//! clockwise seen from outside); an open mesh is taken as it is, as a one-sided
//! barrier whose forbidden side is the side its triangle normals point to.
//!
//! [`BlowMolding::mold_collision`]: super::BlowMolding::mold_collision

use super::functions::{vec_add, vec_cross, vec_dot, vec_len, vec_scale, vec_sub};
use std::collections::HashMap;

/// Largest number of triangles in a leaf of the bounding-volume hierarchy.
const LEAF_TRIANGLES: usize = 4;

/// A triangle whose doubled area is below this fraction of its squared longest
/// edge is a sliver without a usable normal and is dropped.
const DEGENERATE_RELATIVE_AREA: f64 = 1e-12;

/// A point closer than this fraction of the mold's bounding-box diagonal to the
/// outer side of the surface counts as touching it rather than as outside.
/// Rounding puts a point that was just projected onto the surface this far off
/// it, and such a point must not be projected (and its velocity changed) again.
const SURFACE_TOLERANCE: f64 = 1e-12;

/// The closest feature of a triangle to a query point.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum TriangleFeature {
    /// Corner `k` of the triangle.
    Vertex(usize),
    /// Edge `k`, joining corners `k` and `(k + 1) % 3`.
    Edge(usize),
    /// The interior of the face.
    Face,
}

/// Closest point of the triangle `(a, b, c)` to `p`, with the feature of the
/// triangle that point lies on (Ericson, *Real-Time Collision Detection*,
/// section 5.1.5).
///
/// The triangle must not be degenerate.
pub(super) fn closest_point_on_triangle(
    p: [f64; 3],
    a: [f64; 3],
    b: [f64; 3],
    c: [f64; 3],
) -> ([f64; 3], TriangleFeature) {
    let ab = vec_sub(b, a);
    let ac = vec_sub(c, a);
    let ap = vec_sub(p, a);
    let d1 = vec_dot(ab, ap);
    let d2 = vec_dot(ac, ap);
    if d1 <= 0.0 && d2 <= 0.0 {
        return (a, TriangleFeature::Vertex(0));
    }
    let bp = vec_sub(p, b);
    let d3 = vec_dot(ab, bp);
    let d4 = vec_dot(ac, bp);
    if d3 >= 0.0 && d4 <= d3 {
        return (b, TriangleFeature::Vertex(1));
    }
    let vc = d1 * d4 - d3 * d2;
    if vc <= 0.0 && d1 >= 0.0 && d3 <= 0.0 {
        let t = d1 / (d1 - d3);
        return (vec_add(a, vec_scale(ab, t)), TriangleFeature::Edge(0));
    }
    let cp = vec_sub(p, c);
    let d5 = vec_dot(ab, cp);
    let d6 = vec_dot(ac, cp);
    if d6 >= 0.0 && d5 <= d6 {
        return (c, TriangleFeature::Vertex(2));
    }
    let vb = d5 * d2 - d1 * d6;
    if vb <= 0.0 && d2 >= 0.0 && d6 <= 0.0 {
        let t = d2 / (d2 - d6);
        return (vec_add(a, vec_scale(ac, t)), TriangleFeature::Edge(2));
    }
    let va = d3 * d6 - d5 * d4;
    if va <= 0.0 && d4 - d3 >= 0.0 && d5 - d6 >= 0.0 {
        let t = (d4 - d3) / ((d4 - d3) + (d5 - d6));
        let bc = vec_sub(c, b);
        return (vec_add(b, vec_scale(bc, t)), TriangleFeature::Edge(1));
    }
    let inverse_sum = 1.0 / (va + vb + vc);
    let v = vb * inverse_sum;
    let w = vc * inverse_sum;
    let offset = vec_add(vec_scale(ab, v), vec_scale(ac, w));
    (vec_add(a, offset), TriangleFeature::Face)
}

/// A mold triangle with the normals needed to classify points against it.
#[derive(Debug, Clone, Copy)]
struct MoldTriangle {
    /// Welded vertex ids of the three corners.
    corners: [usize; 3],
    /// Positions of the corners.
    points: [[f64; 3]; 3],
    /// Unit face normal, pointing away from the enclosed region.
    face_normal: [f64; 3],
    /// Edge pseudo-normals, pointing away from the enclosed region. Entry `k`
    /// belongs to the edge joining corners `k` and `(k + 1) % 3`.
    edge_normals: [[f64; 3]; 3],
}

/// What a node of the bounding-volume hierarchy holds.
#[derive(Debug, Clone, Copy)]
enum NodeKind {
    /// The triangles `order[first..first + count]`.
    Leaf { first: usize, count: usize },
    /// Two child nodes.
    Inner { left: usize, right: usize },
}

/// A node of the bounding-volume hierarchy.
#[derive(Debug, Clone, Copy)]
struct BvhNode {
    min: [f64; 3],
    max: [f64; 3],
    kind: NodeKind,
}

/// The best triangle found so far by a closest-point search.
#[derive(Debug, Clone, Copy)]
struct Candidate {
    distance_sq: f64,
    triangle: usize,
    point: [f64; 3],
    feature: TriangleFeature,
}

/// Result of a closest-point query against a [`MoldSurface`].
#[derive(Debug, Clone, Copy)]
pub(super) struct SurfaceHit {
    /// Closest point of the mold surface.
    pub(super) point: [f64; 3],
    /// Unit pseudo-normal of the closest feature, pointing away from the region
    /// the mold encloses.
    pub(super) normal: [f64; 3],
}

/// A triangulated mold with its pseudo-normals and a bounding-volume hierarchy.
#[derive(Debug, Clone)]
pub(super) struct MoldSurface {
    triangles: Vec<MoldTriangle>,
    /// Angle-weighted pseudo-normal of every welded vertex id (zero for ids no
    /// triangle uses).
    vertex_normals: Vec<[f64; 3]>,
    nodes: Vec<BvhNode>,
    /// Triangle indices in the leaf order of the hierarchy.
    order: Vec<usize>,
    /// Distance from the outer side of the surface within which a point counts
    /// as touching it ([`SURFACE_TOLERANCE`] times the bounding-box diagonal).
    tolerance: f64,
}

impl MoldSurface {
    /// Build the surface of the mold `(vertices, tris)`.
    ///
    /// Triangles that reference a missing vertex, have a non-finite corner or
    /// are degenerate are ignored, and vertices at bit-identical positions are
    /// welded, so a triangle soup with duplicated vertices behaves like an
    /// indexed mesh. Returns `None` when no usable triangle is left.
    pub(super) fn new(vertices: &[[f64; 3]], tris: &[[usize; 3]]) -> Option<Self> {
        let welded = weld_vertices(vertices);
        let usable: Vec<([usize; 3], [[f64; 3]; 3])> = tris
            .iter()
            .filter_map(|&corners| usable_triangle(vertices, &welded, corners))
            .collect();
        if usable.is_empty() {
            return None;
        }

        let (lower, upper) = bounding_box(usable.iter().flat_map(|(_, points)| points.iter()));
        let centre = vec_scale(vec_add(lower, upper), 0.5);
        let tolerance = SURFACE_TOLERANCE * vec_len(vec_sub(upper, lower));

        // A closed, consistently oriented mold encloses a volume whose sign
        // tells which way its triangles wind; an open one has no such sign.
        let corner_lists: Vec<[usize; 3]> = usable.iter().map(|(corners, _)| *corners).collect();
        let six_volume: f64 = usable
            .iter()
            .map(|(_, [a, b, c])| {
                let (a, b, c) = (
                    vec_sub(*a, centre),
                    vec_sub(*b, centre),
                    vec_sub(*c, centre),
                );
                vec_dot(a, vec_cross(b, c))
            })
            .sum();
        let flip = if is_closed_and_consistent(&corner_lists) && six_volume < 0.0 {
            -1.0
        } else {
            1.0
        };

        let mut vertex_normals = vec![[0.0_f64; 3]; vertices.len()];
        let mut edge_sums: HashMap<(usize, usize), [f64; 3]> = HashMap::new();
        let mut triangles: Vec<MoldTriangle> = Vec::with_capacity(usable.len());
        for (corners, points) in &usable {
            let [a, b, c] = *points;
            let face_normal = vec_scale(unit(vec_cross(vec_sub(b, a), vec_sub(c, a))), flip);
            for (k, &corner) in corners.iter().enumerate() {
                let next = (k + 1) % 3;
                let previous = (k + 2) % 3;
                let angle = corner_angle(points[k], points[next], points[previous]);
                vertex_normals[corner] =
                    vec_add(vertex_normals[corner], vec_scale(face_normal, angle));
                let sum = edge_sums
                    .entry(edge_key(corner, corners[next]))
                    .or_insert([0.0; 3]);
                *sum = vec_add(*sum, face_normal);
            }
            triangles.push(MoldTriangle {
                corners: *corners,
                points: *points,
                face_normal,
                edge_normals: [face_normal; 3],
            });
        }
        for normal in &mut vertex_normals {
            *normal = unit(*normal);
        }
        for triangle in &mut triangles {
            let corners = triangle.corners;
            for (k, edge_normal) in triangle.edge_normals.iter_mut().enumerate() {
                let key = edge_key(corners[k], corners[(k + 1) % 3]);
                if let Some(sum) = edge_sums.get(&key) {
                    let normal = unit(*sum);
                    // Opposite faces on one edge (a zero-thickness fold) cancel.
                    if vec_dot(normal, normal) > 0.5 {
                        *edge_normal = normal;
                    }
                }
            }
        }

        let (nodes, order) = build_hierarchy(&triangles);
        Some(Self {
            triangles,
            vertex_normals,
            nodes,
            order,
            tolerance,
        })
    }

    /// Closest point of the surface to `p`, or `None` when `p` is not finite.
    ///
    /// Ties between equally close triangles go to the one that comes first in
    /// the mold, so the result does not depend on the layout of the hierarchy.
    pub(super) fn closest(&self, p: [f64; 3]) -> Option<SurfaceHit> {
        if !p.iter().all(|c| c.is_finite()) {
            return None;
        }
        let mut best = Candidate {
            distance_sq: f64::INFINITY,
            triangle: usize::MAX,
            point: p,
            feature: TriangleFeature::Face,
        };
        self.search(0, p, &mut best);
        let triangle = self.triangles.get(best.triangle)?;
        Some(SurfaceHit {
            point: best.point,
            normal: self.feature_normal(triangle, best.feature),
        })
    }

    /// Move `position` onto the mold surface if it lies outside the mold, and
    /// remove the outward part of `velocity` there.
    ///
    /// The contact normal is the pseudo-normal of the closest feature, which is
    /// also what decides that `position` is outside, so the velocity left behind
    /// has no outward part. A point on or inside the mold (within rounding of
    /// its surface), and a point that is not finite, is left untouched, so a
    /// second call changes nothing. Returns whether `position` was treated as
    /// outside.
    pub(super) fn constrain(
        &self,
        position: &mut [f64; 3],
        velocity: Option<&mut [f64; 3]>,
    ) -> bool {
        let Some(hit) = self.closest(*position) else {
            return false;
        };
        if vec_dot(vec_sub(*position, hit.point), hit.normal) <= self.tolerance {
            return false;
        }
        *position = hit.point;
        if let Some(velocity) = velocity {
            let outward_speed = vec_dot(*velocity, hit.normal);
            if outward_speed > 0.0 {
                *velocity = vec_sub(*velocity, vec_scale(hit.normal, outward_speed));
            }
        }
        true
    }

    /// Pseudo-normal of `feature` of `triangle`.
    fn feature_normal(&self, triangle: &MoldTriangle, feature: TriangleFeature) -> [f64; 3] {
        let candidate = match feature {
            TriangleFeature::Face => triangle.face_normal,
            TriangleFeature::Edge(k) => triangle.edge_normals[k % 3],
            TriangleFeature::Vertex(k) => self.vertex_normals[triangle.corners[k % 3]],
        };
        // Normals are unit or zero; a cancelled corner falls back on the face.
        if vec_dot(candidate, candidate) > 0.5 {
            candidate
        } else {
            triangle.face_normal
        }
    }

    /// Depth-first closest-point search below `node_index`, nearest child
    /// first, pruning subtrees that cannot beat `best`.
    fn search(&self, node_index: usize, p: [f64; 3], best: &mut Candidate) {
        let Some(node) = self.nodes.get(node_index) else {
            return;
        };
        if box_distance_sq(node.min, node.max, p) > best.distance_sq {
            return;
        }
        match node.kind {
            NodeKind::Leaf { first, count } => {
                let Some(leaf) = self.order.get(first..first + count) else {
                    return;
                };
                for &triangle_index in leaf {
                    let [a, b, c] = self.triangles[triangle_index].points;
                    let (point, feature) = closest_point_on_triangle(p, a, b, c);
                    let offset = vec_sub(p, point);
                    let distance_sq = vec_dot(offset, offset);
                    let closer = distance_sq < best.distance_sq
                        || (distance_sq == best.distance_sq && triangle_index < best.triangle);
                    if closer {
                        *best = Candidate {
                            distance_sq,
                            triangle: triangle_index,
                            point,
                            feature,
                        };
                    }
                }
            }
            NodeKind::Inner { left, right } => {
                let distance_to = |index: usize| {
                    self.nodes.get(index).map_or(f64::INFINITY, |child| {
                        box_distance_sq(child.min, child.max, p)
                    })
                };
                let (near, far) = if distance_to(left) <= distance_to(right) {
                    (left, right)
                } else {
                    (right, left)
                };
                self.search(near, p, best);
                self.search(far, p, best);
            }
        }
    }
}

/// Map every vertex to the first vertex at a bit-identical position.
fn weld_vertices(vertices: &[[f64; 3]]) -> Vec<usize> {
    // `x + 0.0` turns -0.0 into +0.0 so both zeros weld.
    let key = |v: &[f64; 3]| {
        [
            (v[0] + 0.0).to_bits(),
            (v[1] + 0.0).to_bits(),
            (v[2] + 0.0).to_bits(),
        ]
    };
    let mut first_seen: HashMap<[u64; 3], usize> = HashMap::with_capacity(vertices.len());
    vertices
        .iter()
        .enumerate()
        .map(|(index, v)| *first_seen.entry(key(v)).or_insert(index))
        .collect()
}

/// The welded corner ids and positions of a triangle, or `None` when it
/// references a missing vertex, has a non-finite corner or is degenerate.
fn usable_triangle(
    vertices: &[[f64; 3]],
    welded: &[usize],
    corners: [usize; 3],
) -> Option<([usize; 3], [[f64; 3]; 3])> {
    let a = *vertices.get(corners[0])?;
    let b = *vertices.get(corners[1])?;
    let c = *vertices.get(corners[2])?;
    if ![a, b, c].iter().all(|v| v.iter().all(|x| x.is_finite())) {
        return None;
    }
    let ab = vec_sub(b, a);
    let ac = vec_sub(c, a);
    let bc = vec_sub(c, b);
    let longest_sq = vec_dot(ab, ab).max(vec_dot(ac, ac)).max(vec_dot(bc, bc));
    let double_area = vec_len(vec_cross(ab, ac));
    let has_normal = double_area.is_finite() && double_area > DEGENERATE_RELATIVE_AREA * longest_sq;
    if !has_normal {
        return None;
    }
    let ids = [
        *welded.get(corners[0])?,
        *welded.get(corners[1])?,
        *welded.get(corners[2])?,
    ];
    Some((ids, [a, b, c]))
}

/// Whether every edge is used by exactly one triangle in each direction, i.e.
/// the triangles form a closed, consistently oriented surface.
fn is_closed_and_consistent(triangles: &[[usize; 3]]) -> bool {
    let mut directed: HashMap<(usize, usize), u32> = HashMap::new();
    for corners in triangles {
        for (k, &from) in corners.iter().enumerate() {
            *directed.entry((from, corners[(k + 1) % 3])).or_insert(0) += 1;
        }
    }
    directed
        .iter()
        .all(|(&(from, to), &count)| count == 1 && directed.get(&(to, from)) == Some(&1))
}

/// Key of the undirected edge `{a, b}`.
fn edge_key(a: usize, b: usize) -> (usize, usize) {
    if a < b { (a, b) } else { (b, a) }
}

/// Angle at `corner` between the edges to `next` and `previous`.
fn corner_angle(corner: [f64; 3], next: [f64; 3], previous: [f64; 3]) -> f64 {
    let e1 = vec_sub(next, corner);
    let e2 = vec_sub(previous, corner);
    vec_len(vec_cross(e1, e2)).atan2(vec_dot(e1, e2))
}

/// `v` scaled to unit length, or zero when it has no length.
fn unit(v: [f64; 3]) -> [f64; 3] {
    let length = vec_len(v);
    if length > 0.0 && length.is_finite() {
        vec_scale(v, 1.0 / length)
    } else {
        [0.0; 3]
    }
}

/// Smallest axis-aligned box around `points`.
fn bounding_box<'a>(points: impl Iterator<Item = &'a [f64; 3]>) -> ([f64; 3], [f64; 3]) {
    let mut lower = [f64::INFINITY; 3];
    let mut upper = [f64::NEG_INFINITY; 3];
    for point in points {
        for (axis, &x) in point.iter().enumerate() {
            lower[axis] = lower[axis].min(x);
            upper[axis] = upper[axis].max(x);
        }
    }
    (lower, upper)
}

/// Squared distance from `p` to the box `[min, max]` (zero inside).
fn box_distance_sq(min: [f64; 3], max: [f64; 3], p: [f64; 3]) -> f64 {
    let mut sum = 0.0;
    for (axis, &x) in p.iter().enumerate() {
        let excess = (min[axis] - x).max(x - max[axis]).max(0.0);
        sum += excess * excess;
    }
    sum
}

/// Build a median-split bounding-volume hierarchy over `triangles`.
///
/// Returns the nodes (the root is node 0) and the triangle order the leaves
/// refer to.
fn build_hierarchy(triangles: &[MoldTriangle]) -> (Vec<BvhNode>, Vec<usize>) {
    let centroids: Vec<[f64; 3]> = triangles
        .iter()
        .map(|triangle| {
            let [a, b, c] = triangle.points;
            vec_scale(vec_add(vec_add(a, b), c), 1.0 / 3.0)
        })
        .collect();
    let mut order: Vec<usize> = (0..triangles.len()).collect();
    let mut nodes: Vec<BvhNode> = Vec::with_capacity(2 * triangles.len() / LEAF_TRIANGLES + 1);
    build_node(triangles, &centroids, &mut order, 0, &mut nodes);
    (nodes, order)
}

/// Append the subtree over `order` (which starts at position `first` of the
/// full order) to `nodes` and return its index.
fn build_node(
    triangles: &[MoldTriangle],
    centroids: &[[f64; 3]],
    order: &mut [usize],
    first: usize,
    nodes: &mut Vec<BvhNode>,
) -> usize {
    let (min, max) = bounding_box(
        order
            .iter()
            .flat_map(|&index| triangles[index].points.iter()),
    );
    let node_index = nodes.len();
    let leaf = NodeKind::Leaf {
        first,
        count: order.len(),
    };
    nodes.push(BvhNode {
        min,
        max,
        kind: leaf,
    });
    if order.len() <= LEAF_TRIANGLES {
        return node_index;
    }

    let (centroid_min, centroid_max) = bounding_box(order.iter().map(|&index| &centroids[index]));
    let spread = |axis: usize| centroid_max[axis] - centroid_min[axis];
    let axis = (1..3).fold(0, |widest, axis| {
        if spread(axis) > spread(widest) {
            axis
        } else {
            widest
        }
    });
    let middle = order.len() / 2;
    order.select_nth_unstable_by(middle, |&i, &j| {
        centroids[i][axis]
            .total_cmp(&centroids[j][axis])
            .then(i.cmp(&j))
    });
    let (lower, upper) = order.split_at_mut(middle);
    let left = build_node(triangles, centroids, lower, first, nodes);
    let right = build_node(triangles, centroids, upper, first + middle, nodes);
    nodes[node_index].kind = NodeKind::Inner { left, right };
    node_index
}
