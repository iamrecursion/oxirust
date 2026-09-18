// Copyright 2026 COOLJAPAN OU (Team KitaSan)
// SPDX-License-Identifier: Apache-2.0

//! Quickhull core shared by [`ConvexHull3D`](super::types::ConvexHull3D) and
//! [`ConvexHull3DVec`](crate::quickhull::ConvexHull3DVec).
//!
//! This is the Quickhull algorithm of Barber, Dobkin and Huhdanpaa (1996) on a
//! triangulated surface with explicit face adjacency:
//!
//! * A seed tetrahedron is built from extreme points: the two extremes along the
//!   axis of largest extent, the point farthest from the line through them, and
//!   the point farthest from the plane of the three.  Exactly tied candidates
//!   (the points of a flat face, a collinear run) are resolved by the
//!   lexicographic order of the coordinates, so the seed does not depend on the
//!   order of the input.
//! * Every other point is assigned to the *outside set* of the face it lies
//!   farthest in front of; points that are in front of no face are inside the
//!   hull and are never looked at again.
//! * The hull grows by repeatedly taking a face with a non-empty outside set and
//!   adding the *farthest* point `p` of that set.  The faces that see `p` form a
//!   connected region found by a search over the face adjacency, starting at the
//!   face that owns `p`.  Their boundary, the *horizon*, is a closed loop of
//!   directed edges; the visible faces are replaced by a fan of new faces from
//!   each horizon edge to `p`.  A new face keeps the direction of the horizon edge
//!   it stands on, so the surface stays consistently wound (counter-clockwise
//!   seen from outside) without any geometric orientation test, and the new faces
//!   are wired to each other and to the faces across the horizon through the
//!   vertices at the ends of the horizon edges.
//! * The outside points of the removed faces are re-assigned to the new faces.
//!
//! The result is a triangulated surface, not a polytope with merged coplanar
//! facets: for input with many coplanar or collinear points (a lattice, a
//! cylinder) a point inside a flat face or on an edge can remain a vertex of the
//! triangulation, and where a flat face has more than three corners its
//! triangulation depends on the order of the construction.  Neither makes the
//! surface invalid; exactly tied farthest points are resolved lexicographically,
//! so the construction is deterministic.
//!
//! Compared with inserting the points in input order and testing each against
//! every face, the work per point is proportional to the size of its visible
//! region instead of the whole hull, and expensive far-away points are handled
//! first, which keeps the hull small while it grows: for `n` points the expected
//! cost is close to `O(n log n)` instead of `O(n h)` for `h` hull faces.
//!
//! # Tolerances and robustness
//!
//! Two distances are involved.  A point is *added* to the hull only if it lies
//! more than [`HullTolerances::visible_distance`] (the tolerance) in front of some
//! face plane, so that points within the noise of the input of a flat face or of
//! the hull are never added; this keeps the surface free of faces between nearly
//! collinear or coplanar points.  A face is *removed* for an added point when the
//! point is in front of it by more than [`REGION_FRACTION`] of the tolerance.  A
//! face that the point is only a little in front of is kept, which keeps the new
//! faces fat, but the edge between it and the new faces is then slightly reflex.
//! A reflex edge could make a later point inside the kept face look like an
//! outside point and fold the surface over itself, so a face across the horizon
//! is removed as well whenever its far corner would lie more than the tolerance in
//! front of the new face: for faces that are not thin, the far corner of a face is
//! never more than the tolerance beyond the plane of a neighbouring face.
//!
//! The rounded plane of a thin face (see [`THIN_FACE_QUALITY`]) is unreliable, so
//! for such a face, and only for it, the exact orientation predicate of
//! [`oxiphysics_core::exact_predicates`] decides whether a point is in front of
//! it.  The orientation of the seed tetrahedron is exact as well.
//!
//! If the region of a point is nevertheless not a disc (its boundary touches
//! itself, which a convex surface rules out but a tolerance can allow), the removal
//! distance is lowered step by step; a point for which that does not succeed is
//! within rounding error of the hull and is dropped.
//!
//! All arithmetic is done relative to the centre of the bounding box of the
//! input, so the accuracy does not depend on where the cloud sits in space.

use super::functions::{cross, dot, sub};
use oxiphysics_core::exact_predicates::{Orientation, orient3d};

/// Distance tolerances of the hull construction, as absolute lengths.
#[derive(Debug, Clone, Copy)]
pub(crate) struct HullTolerances {
    /// A point is added to the hull only if it lies more than this far in front
    /// of the plane of some face (as computed in floating point); points closer
    /// to every face plane count as part of the hull and are left out.
    pub visible_distance: f64,
    /// Smallest distance of the third seed point from the line through the first
    /// two and of the fourth seed point from the plane through the first three.
    /// Point sets that do not reach it are degenerate (coincident, collinear or
    /// coplanar) and have no hull.
    pub seed_distance: f64,
}

/// The closed, consistently wound triangle surface of a hull.
#[derive(Debug, Clone)]
pub(crate) struct HullSurface {
    /// The input points that are corners of a face, in the order of the input.
    pub vertices: Vec<[f64; 3]>,
    /// Triangles as indices into [`HullSurface::vertices`], counter-clockwise
    /// seen from outside.
    pub faces: Vec<[usize; 3]>,
}

/// Size of a point cloud.
#[derive(Debug, Clone, Copy)]
pub(crate) struct CloudExtent {
    /// Largest side of the axis-aligned bounding box.
    pub longest_side: f64,
    /// Diagonal of the axis-aligned bounding box.
    pub diagonal: f64,
}

/// Extent of the bounding box of `points`, or `None` if there are no points, a
/// coordinate is not finite or all points coincide.
pub(crate) fn cloud_extent(points: &[[f64; 3]]) -> Option<CloudExtent> {
    let (low, high) = bounding_box(points)?;
    let side = sub(high, low);
    let longest_side = side.iter().copied().fold(0.0_f64, f64::max);
    let diagonal = side[0].hypot(side[1]).hypot(side[2]);
    (longest_side > 0.0 && longest_side.is_finite() && diagonal.is_finite()).then_some(
        CloudExtent {
            longest_side,
            diagonal,
        },
    )
}

/// Component-wise bounds of a non-empty cloud of finite points.
fn bounding_box(points: &[[f64; 3]]) -> Option<([f64; 3], [f64; 3])> {
    let first = *points.first()?;
    let mut low = first;
    let mut high = first;
    for point in points {
        for ((lo, hi), &c) in low.iter_mut().zip(high.iter_mut()).zip(point.iter()) {
            if !c.is_finite() {
                return None;
            }
            *lo = lo.min(c);
            *hi = hi.max(c);
        }
    }
    Some((low, high))
}

/// Convex hull of `points` as a triangle surface, or `None` if there are fewer
/// than four points, a coordinate is not finite, or the points are coincident,
/// collinear or coplanar within [`HullTolerances::seed_distance`].
///
/// The vertices of the surface are the input points that are corners of a face,
/// in input order, every corner of the hull among them.  Input points within
/// [`HullTolerances::visible_distance`] of the surface may be left out, and a
/// point inside a flat face or on an edge may be kept as a vertex of the
/// triangulation.
pub(crate) fn build_hull(points: &[[f64; 3]], tolerances: HullTolerances) -> Option<HullSurface> {
    if points.len() < 4 {
        return None;
    }
    let (low, high) = bounding_box(points)?;
    // Work relative to the centre of the box: this removes the position of the
    // cloud from every distance computation.
    let centre = [
        0.5 * (low[0] + high[0]),
        0.5 * (low[1] + high[1]),
        0.5 * (low[2] + high[2]),
    ];
    let shifted: Vec<[f64; 3]> = points.iter().map(|&p| sub(p, centre)).collect();
    let side = sub(high, low);
    let axis = if side[0] >= side[1] && side[0] >= side[2] {
        0
    } else if side[1] >= side[2] {
        1
    } else {
        2
    };
    let seed = seed_tetrahedron(&shifted, axis, tolerances.seed_distance)?;
    let mut hull = Quickhull::new(&shifted, points, tolerances.visible_distance, seed)?;
    hull.run();
    let triangles = hull.triangles();
    if triangles.is_empty() {
        return None;
    }

    let mut used: Vec<usize> = triangles.iter().flatten().copied().collect();
    used.sort_unstable();
    used.dedup();
    let faces: Vec<[usize; 3]> = triangles
        .iter()
        .map(|triangle| {
            let remap = |old: usize| used.binary_search(&old).ok();
            Some([
                remap(triangle[0])?,
                remap(triangle[1])?,
                remap(triangle[2])?,
            ])
        })
        .collect::<Option<_>>()?;
    let vertices = used.iter().map(|&i| points[i]).collect();
    Some(HullSurface { vertices, faces })
}

/// Index of the point that wins the comparison `better(candidate, incumbent)`,
/// scanning in input order.
fn select(
    points: &[[f64; 3]],
    exclude: &[usize],
    better: impl Fn(&[f64; 3], &[f64; 3]) -> bool,
) -> Option<usize> {
    let mut best: Option<usize> = None;
    for (i, point) in points.iter().enumerate() {
        if exclude.contains(&i) {
            continue;
        }
        if best.is_none_or(|incumbent| better(point, &points[incumbent])) {
            best = Some(i);
        }
    }
    best
}

/// Four affinely independent points to start the hull from, or `None` if the
/// points are coincident, collinear or coplanar within `seed_distance`.
fn seed_tetrahedron(points: &[[f64; 3]], axis: usize, seed_distance: f64) -> Option<[usize; 4]> {
    // Extremes along the axis of largest extent; ties go to the lexicographically
    // smallest / largest point.
    let lowest = select(points, &[], |p, q| {
        p[axis] < q[axis] || (p[axis] == q[axis] && p < q)
    })?;
    let highest = select(points, &[], |p, q| {
        p[axis] > q[axis] || (p[axis] == q[axis] && p > q)
    })?;
    if lowest == highest {
        return None;
    }
    let origin = points[lowest];
    let along = sub(points[highest], origin);
    let along_length = dot(along, along).sqrt();
    if along_length <= seed_distance {
        return None;
    }
    // The point farthest from the line through the two extremes.
    let line_distance = |p: &[f64; 3]| {
        let c = cross(sub(*p, origin), along);
        dot(c, c).sqrt() / along_length
    };
    let third = select(points, &[lowest, highest], |p, q| {
        let (dp, dq) = (line_distance(p), line_distance(q));
        dp > dq || (dp == dq && p > q)
    })?;
    if line_distance(&points[third]) <= seed_distance {
        return None;
    }
    // The point farthest from the plane through the three, on either side.
    let normal = cross(along, sub(points[third], origin));
    let normal_length = dot(normal, normal).sqrt();
    if normal_length == 0.0 || !normal_length.is_finite() {
        return None;
    }
    let plane_distance = |p: &[f64; 3]| dot(normal, sub(*p, origin)).abs() / normal_length;
    let fourth = select(points, &[lowest, highest, third], |p, q| {
        let (dp, dq) = (plane_distance(p), plane_distance(q));
        dp > dq || (dp == dq && p > q)
    })?;
    if plane_distance(&points[fourth]) <= seed_distance {
        return None;
    }
    Some([lowest, highest, third, fourth])
}

/// Fraction of [`HullTolerances::visible_distance`] by which an added point must
/// lie in front of a face for the face to be removed.
const REGION_FRACTION: f64 = 1e-2;

/// A face whose area is below this fraction of the square of its longest edge is
/// *thin*: the rounded direction of its normal is off by about the rounding error
/// divided by this number, so its rounded plane cannot be trusted at the scale
/// of [`REGION_FRACTION`] of the tolerance.
const THIN_FACE_QUALITY: f64 = 1e-3;

/// Number of times the removal distance is lowered (by a factor of 4) when the
/// region of a point is not a disc, before the point is given up.
const MAX_TOLERANCE_REDUCTIONS: u32 = 6;

/// A triangular face of the hull under construction.
#[derive(Debug, Clone)]
struct Face {
    /// Corners, counter-clockwise seen from outside.
    vertices: [usize; 3],
    /// `neighbors[k]` is the face across the edge `vertices[k] -> vertices[k + 1]`.
    neighbors: [usize; 3],
    /// Unit outward normal (zero for a degenerate face, which sees no point).
    normal: [f64; 3],
    /// Offset of the face plane: it is `normal . x = offset`.
    offset: f64,
    /// `true` for a thin face, whose rounded plane is unreliable.
    thin: bool,
    /// Points in front of the face that are assigned to it.
    outside: Vec<usize>,
    /// The point of `outside` that is farthest in front of the face.
    furthest: usize,
    /// Its distance from the face plane.
    furthest_distance: f64,
    /// `false` for a face that has been replaced.
    alive: bool,
}

/// A boundary edge of the visible region.
#[derive(Debug, Clone, Copy)]
struct HorizonEdge {
    /// Start and end of the edge, in the direction of the visible face.
    from: usize,
    to: usize,
    /// The face on the far side of the edge and the index of the edge there.
    outer_face: usize,
    outer_edge: usize,
}

/// Plane through the counter-clockwise triangle `(a, b, c)`: the unit normal, the
/// offset and the quality of the triangle (twice its area over the square of its
/// longest edge, between 0 and 1).  The normal is zero for a triangle without
/// area.
///
/// The cross product is formed at the corner opposite the longest edge, the best
/// conditioned one, and the offset is taken at the centroid.
fn plane_through(a: [f64; 3], b: [f64; 3], c: [f64; 3]) -> ([f64; 3], f64, f64) {
    let ab = sub(b, a);
    let bc = sub(c, b);
    let ca = sub(a, c);
    let (len_ab, len_bc, len_ca) = (dot(ab, ab), dot(bc, bc), dot(ca, ca));
    let longest = len_ab.max(len_bc).max(len_ca);
    let raw = if len_bc >= len_ab && len_bc >= len_ca {
        cross(ab, sub(c, a))
    } else if len_ca >= len_ab {
        cross(bc, sub(a, b))
    } else {
        cross(ca, sub(b, c))
    };
    let length = dot(raw, raw).sqrt();
    if length > 0.0 && length.is_finite() {
        let normal = [raw[0] / length, raw[1] / length, raw[2] / length];
        let centroid = [
            (a[0] + b[0] + c[0]) / 3.0,
            (a[1] + b[1] + c[1]) / 3.0,
            (a[2] + b[2] + c[2]) / 3.0,
        ];
        (normal, dot(normal, centroid), length / longest)
    } else {
        ([0.0; 3], 0.0, 0.0)
    }
}

/// The hull under construction.
struct Quickhull<'a> {
    /// The points relative to the centre of their bounding box, for rounded
    /// distances.
    points: &'a [[f64; 3]],
    /// The same points as given, for the exact predicates.
    exact: &'a [[f64; 3]],
    /// A point is added only if it is further in front of a face than this.
    visible_distance: f64,
    faces: Vec<Face>,
    /// Indices of replaced faces that can be reused.
    free_faces: Vec<usize>,
    /// Per face: `2 * epoch` if it was tested in the visibility search of that
    /// epoch and is not visible, `2 * epoch + 1` if it is visible.
    mark: Vec<u64>,
    epoch: u64,
    // Scratch buffers, kept to avoid allocating for every point.
    stack: Vec<usize>,
    visible: Vec<usize>,
    horizon: Vec<HorizonEdge>,
    new_faces: Vec<usize>,
    /// Per point: `(epoch, index into horizon)` of the horizon edge that starts
    /// at the point in that epoch.
    starts: Vec<(u64, usize)>,
}

impl<'a> Quickhull<'a> {
    /// Hull of the seed tetrahedron with the outside sets filled in, or `None`
    /// if the seed faces cannot be wound into a closed surface.
    fn new(
        points: &'a [[f64; 3]],
        exact: &'a [[f64; 3]],
        visible_distance: f64,
        seed: [usize; 4],
    ) -> Option<Self> {
        let mut hull = Self {
            points,
            exact,
            visible_distance,
            faces: Vec::new(),
            free_faces: Vec::new(),
            mark: Vec::new(),
            epoch: 0,
            stack: Vec::new(),
            visible: Vec::new(),
            horizon: Vec::new(),
            new_faces: Vec::new(),
            starts: vec![(0, 0); points.len()],
        };
        let [s0, s1, s2, s3] = seed;
        let faces = [
            ([s0, s1, s2], s3),
            ([s0, s3, s1], s2),
            ([s0, s2, s3], s1),
            ([s1, s3, s2], s0),
        ];
        for (corners, opposite) in faces {
            let [a, b, c] = corners;
            // The seed corner opposite to a face lies behind it when the face is
            // wound counter-clockwise seen from outside.
            let face = match orient3d(exact[a], exact[b], exact[c], exact[opposite]) {
                Orientation::Positive => hull.make_face(a, b, c),
                Orientation::Negative => hull.make_face(a, c, b),
                Orientation::Degenerate => return None,
            };
            hull.push_face(face);
        }
        if !hull.link_seed_faces() {
            return None;
        }
        for (i, point) in points.iter().enumerate() {
            if seed.contains(&i) {
                continue;
            }
            let best = (0..hull.faces.len())
                .map(|f| (f, hull.distance(f, *point)))
                .filter(|&(f, d)| hull.in_front(f, i, d))
                .max_by(|a, b| a.1.total_cmp(&b.1));
            if let Some((face, distance)) = best {
                hull.assign(face, i, distance);
            }
        }
        Some(hull)
    }

    /// A face through the three corners with its plane; not yet stored.
    fn make_face(&self, a: usize, b: usize, c: usize) -> Face {
        let (normal, offset, quality) =
            plane_through(self.points[a], self.points[b], self.points[c]);
        Face {
            vertices: [a, b, c],
            neighbors: [usize::MAX; 3],
            normal,
            offset,
            thin: quality < THIN_FACE_QUALITY,
            outside: Vec::new(),
            furthest: usize::MAX,
            furthest_distance: f64::NEG_INFINITY,
            alive: true,
        }
    }

    /// Store `face`, reusing the slot of a replaced face if there is one, and
    /// return its index.
    fn push_face(&mut self, face: Face) -> usize {
        if let Some(slot) = self.free_faces.pop() {
            self.faces[slot] = face;
            slot
        } else {
            self.faces.push(face);
            self.mark.push(0);
            self.faces.len() - 1
        }
    }

    /// Connect the four faces of the seed tetrahedron through their shared edges.
    /// Returns `false` if some edge has no partner, i.e. the faces are not a
    /// consistently wound closed surface.
    fn link_seed_faces(&mut self) -> bool {
        for f in 0..self.faces.len() {
            for k in 0..3 {
                let (a, b) = (
                    self.faces[f].vertices[k],
                    self.faces[f].vertices[(k + 1) % 3],
                );
                let twin = (0..self.faces.len()).filter(|&g| g != f).find(|&g| {
                    (0..3).any(|j| {
                        self.faces[g].vertices[j] == b && self.faces[g].vertices[(j + 1) % 3] == a
                    })
                });
                let Some(g) = twin else {
                    return false;
                };
                self.faces[f].neighbors[k] = g;
            }
        }
        true
    }

    /// `true` if point `point` lies strictly in front of face `f`.  The answer is
    /// exact: it does not use the rounded plane of the face.
    fn sees(&self, f: usize, point: usize) -> bool {
        let [a, b, c] = self.faces[f].vertices;
        orient3d(
            self.exact[a],
            self.exact[b],
            self.exact[c],
            self.exact[point],
        ) == Orientation::Negative
    }

    /// `true` if point `point`, at the rounded distance `distance` from the plane of
    /// face `f`, is far enough in front of the face to be added to the hull.  For a
    /// thin face the rounded distance is not trusted: the point must also be in
    /// front of it by the exact predicate.
    fn in_front(&self, f: usize, point: usize, distance: f64) -> bool {
        distance > self.visible_distance && (!self.faces[f].thin || self.sees(f, point))
    }

    /// `true` if face `f` is to be removed because point `apex` is in front of it
    /// by more than `threshold` (decided exactly for a thin face).
    fn removed_by(&self, f: usize, apex: usize, threshold: f64) -> bool {
        if self.faces[f].thin {
            self.sees(f, apex)
        } else {
            self.distance(f, self.points[apex]) > threshold
        }
    }

    /// Signed distance of `point` from the plane of face `f`; positive in front.
    fn distance(&self, f: usize, point: [f64; 3]) -> f64 {
        let face = &self.faces[f];
        dot(face.normal, point) - face.offset
    }

    /// Add `point` to the outside set of face `f`; `distance` is its distance
    /// from the face plane.
    fn assign(&mut self, f: usize, point: usize, distance: f64) {
        let candidate = self.points[point];
        let face = &mut self.faces[f];
        // Exactly tied points are resolved by the lexicographic order, so the
        // choice does not depend on the order of the input.
        let better = distance > face.furthest_distance
            || (distance == face.furthest_distance
                && face.furthest != usize::MAX
                && candidate > self.points[face.furthest]);
        face.outside.push(point);
        if better {
            face.furthest = point;
            face.furthest_distance = distance;
        }
    }

    /// Grow the hull until no face has a point in front of it.
    fn run(&mut self) {
        let mut pending: Vec<usize> = (0..self.faces.len())
            .filter(|&f| !self.faces[f].outside.is_empty())
            .collect();
        while let Some(f) = pending.pop() {
            if !self.faces[f].alive || self.faces[f].outside.is_empty() {
                continue;
            }
            self.expand(f, &mut pending);
        }
    }

    /// Add the farthest point of the outside set of face `f` to the hull.
    fn expand(&mut self, f: usize, pending: &mut Vec<usize>) {
        let apex = self.faces[f].furthest;
        let mut threshold = REGION_FRACTION * self.visible_distance;
        let mut found = self.find_visible_region(f, apex, threshold);
        for _ in 0..MAX_TOLERANCE_REDUCTIONS {
            if found {
                break;
            }
            threshold *= 0.25;
            found = self.find_visible_region(f, apex, threshold);
        }
        if found {
            self.replace_visible_region(apex, pending);
        } else {
            // The point is within rounding error of the hull: drop it.
            self.drop_furthest(f);
            if !self.faces[f].outside.is_empty() {
                pending.push(f);
            }
        }
    }

    /// Fill `visible` with the faces that are removed when point `apex` is added
    /// (those it is in front of by more than `threshold`, and the faces that must
    /// follow them, see [`Quickhull::add_reflex_neighbours`]), found by a search
    /// over the adjacency from face `start`, and `horizon` with the edges that
    /// bound them.  Returns `false` if that boundary is not a single simple loop.
    fn find_visible_region(&mut self, start: usize, apex: usize, threshold: f64) -> bool {
        self.epoch += 1;
        let tested = 2 * self.epoch;
        let visible_mark = tested + 1;
        self.visible.clear();
        self.stack.clear();
        self.mark[start] = visible_mark;
        self.visible.push(start);
        self.stack.push(start);
        loop {
            while let Some(f) = self.stack.pop() {
                for k in 0..3 {
                    let g = self.faces[f].neighbors[k];
                    if self.mark[g] >= tested {
                        continue;
                    }
                    if self.removed_by(g, apex, threshold) {
                        self.mark[g] = visible_mark;
                        self.visible.push(g);
                        self.stack.push(g);
                    } else {
                        self.mark[g] = tested;
                    }
                }
            }
            if !self.collect_horizon(visible_mark) {
                return false;
            }
            if !self.add_reflex_neighbours(apex, visible_mark) {
                break;
            }
        }
        self.horizon_is_single_loop()
    }

    /// Fill `horizon` with the edges of the faces marked `visible_mark` whose
    /// partner face is not marked.  Returns `false` if some partner does not have
    /// the reverse edge, which a consistent surface never has.
    fn collect_horizon(&mut self, visible_mark: u64) -> bool {
        self.horizon.clear();
        for i in 0..self.visible.len() {
            let f = self.visible[i];
            for k in 0..3 {
                let g = self.faces[f].neighbors[k];
                if self.mark[g] == visible_mark {
                    continue;
                }
                let from = self.faces[f].vertices[k];
                let to = self.faces[f].vertices[(k + 1) % 3];
                // The face across the edge runs it the other way round.
                let outer_edge = (0..3).find(|&j| {
                    self.faces[g].vertices[j] == to && self.faces[g].vertices[(j + 1) % 3] == from
                });
                let Some(outer_edge) = outer_edge else {
                    return false;
                };
                self.horizon.push(HorizonEdge {
                    from,
                    to,
                    outer_face: g,
                    outer_edge,
                });
            }
        }
        true
    }

    /// Add to the region the faces across the horizon that the new face on the
    /// horizon edge would make reflex by more than the tolerance: those whose far
    /// corner would lie more than [`Quickhull::visible_distance`] in front of the
    /// new face (for a thin new face: at all, decided exactly).  Such a face is
    /// not removed by the point itself, because the point is only a little in
    /// front of it, but it is large compared with the new face.  Returns whether
    /// a face was added.
    fn add_reflex_neighbours(&mut self, apex: usize, visible_mark: u64) -> bool {
        let mut added = false;
        for i in 0..self.horizon.len() {
            let edge = self.horizon[i];
            let outer = edge.outer_face;
            if self.mark[outer] == visible_mark {
                continue;
            }
            let far = self.faces[outer].vertices[(edge.outer_edge + 2) % 3];
            let (normal, offset, quality) = plane_through(
                self.points[edge.from],
                self.points[edge.to],
                self.points[apex],
            );
            let reflex = if quality < THIN_FACE_QUALITY {
                orient3d(
                    self.exact[edge.from],
                    self.exact[edge.to],
                    self.exact[apex],
                    self.exact[far],
                ) == Orientation::Negative
            } else {
                dot(normal, self.points[far]) - offset > self.visible_distance
            };
            if reflex {
                self.mark[outer] = visible_mark;
                self.visible.push(outer);
                self.stack.push(outer);
                added = true;
            }
        }
        added
    }

    /// `true` if the horizon edges form one closed loop that passes through every
    /// vertex at most once, so that a fan of new faces over it is a manifold.
    fn horizon_is_single_loop(&mut self) -> bool {
        let count = self.horizon.len();
        if count < 3 {
            return false;
        }
        for (i, edge) in self.horizon.iter().enumerate() {
            if self.starts[edge.from].0 == self.epoch {
                return false;
            }
            self.starts[edge.from] = (self.epoch, i);
        }
        let mut current = 0;
        let mut steps = 0;
        loop {
            let (epoch, next) = self.starts[self.horizon[current].to];
            if epoch != self.epoch {
                return false;
            }
            current = next;
            steps += 1;
            if current == 0 {
                return steps == count;
            }
            if steps >= count {
                return false;
            }
        }
    }

    /// Replace the faces in `visible` by a fan of faces from `horizon` to
    /// `apex`, and re-assign their outside points to the new faces.
    fn replace_visible_region(&mut self, apex: usize, pending: &mut Vec<usize>) {
        self.new_faces.clear();
        for i in 0..self.horizon.len() {
            let edge = self.horizon[i];
            let face = self.make_face(edge.from, edge.to, apex);
            let id = self.push_face(face);
            self.new_faces.push(id);
            self.faces[id].neighbors[0] = edge.outer_face;
            self.faces[edge.outer_face].neighbors[edge.outer_edge] = id;
        }
        // The face that starts where this one ends shares the edge `to -> apex`.
        for i in 0..self.horizon.len() {
            let id = self.new_faces[i];
            let next = self.new_faces[self.starts[self.horizon[i].to].1];
            self.faces[id].neighbors[1] = next;
            self.faces[next].neighbors[2] = id;
        }

        for i in 0..self.visible.len() {
            let old = self.visible[i];
            let outside = std::mem::take(&mut self.faces[old].outside);
            for point in outside {
                if point != apex {
                    self.reassign(point);
                }
            }
            self.faces[old].alive = false;
        }
        self.free_faces.extend(self.visible.iter().copied());
        for i in 0..self.new_faces.len() {
            let id = self.new_faces[i];
            if !self.faces[id].outside.is_empty() {
                pending.push(id);
            }
        }
    }

    /// Assign `point` to the new face it is farthest in front of, if any.
    fn reassign(&mut self, point: usize) {
        let position = self.points[point];
        let mut best: Option<(usize, f64)> = None;
        for i in 0..self.new_faces.len() {
            let id = self.new_faces[i];
            let distance = self.distance(id, position);
            if self.in_front(id, point, distance) && best.is_none_or(|(_, d)| distance > d) {
                best = Some((id, distance));
            }
        }
        if let Some((id, distance)) = best {
            self.assign(id, point, distance);
        }
    }

    /// Remove the farthest point from the outside set of face `f` without adding
    /// it to the hull.
    fn drop_furthest(&mut self, f: usize) {
        let dropped = self.faces[f].furthest;
        let mut outside = std::mem::take(&mut self.faces[f].outside);
        outside.retain(|&p| p != dropped);
        self.faces[f].furthest = usize::MAX;
        self.faces[f].furthest_distance = f64::NEG_INFINITY;
        for point in outside {
            let distance = self.distance(f, self.points[point]);
            self.assign(f, point, distance);
        }
    }

    /// The surviving faces as triangles of point indices.
    fn triangles(&self) -> Vec<[usize; 3]> {
        self.faces
            .iter()
            .filter(|face| face.alive)
            .map(|face| face.vertices)
            .collect()
    }
}

#[cfg(test)]
mod tests;
