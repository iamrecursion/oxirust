// Copyright 2026 COOLJAPAN OU (Team KitaSan)
// SPDX-License-Identifier: Apache-2.0

//! Exact 2-D Delaunay triangulation: incremental Bowyer-Watson insertion with
//! ghost triangles, adjacency-based point location and a biased randomized
//! insertion order.
//!
//! The triangulation never relies on a floating-point tolerance: every
//! geometric decision is made with the exact predicates
//! [`orient2d`](oxiphysics_core::exact_predicates::orient2d) and
//! [`incircle`](oxiphysics_core::exact_predicates::incircle), so the result is
//! the same for any scale of the input (a grid scaled by `1e-6` triangulates
//! like the unit grid), and degenerate inputs (collinear points, cocircular
//! points, grids, duplicates) neither fail nor lose triangles.
//!
//! ## Algorithm
//!
//! * **Ghost triangles.** Instead of a finite super-triangle (which cuts off
//!   hull triangles whose circumcircle reaches the super-triangle's corners),
//!   every hull edge is closed by a *ghost* triangle `(u, v, G)` that has the
//!   vertex at infinity `G` as third corner.  The mesh is thereby a closed
//!   triangulated sphere in which every triangle has three neighbours.  A ghost
//!   triangle is "in conflict" with a point `p` when `p` lies strictly outside
//!   the hull across its edge, or on the open hull edge itself; a real triangle
//!   is in conflict when `p` lies strictly inside its circumcircle.
//! * **Insertion.** A point is located by *walking* across the edges that
//!   separate the current triangle from it, starting at the triangle created
//!   last.  The cavity is the connected set of conflicting triangles around the
//!   located one, found by flooding through the adjacency with a per-triangle
//!   visit stamp (no per-insertion allocation proportional to the mesh size);
//!   it is replaced by the fan of triangles joining the point to the cavity
//!   boundary, and the adjacency is restitched along the boundary.
//! * **Order.** Points are inserted in a *biased randomized insertion order*
//!   (BRIO): each point is assigned to one of `log2(n)` rounds of geometrically
//!   growing size, and every round is sorted along a Hilbert curve.  The
//!   randomization bounds the expected size of the cavities, the curve keeps
//!   consecutive points close so the walks are short: expected `O(n log n)`.
//!
//! References: Bowyer 1981; Watson 1981; Amenta, Choi & Rote, "Incremental
//! constructions con BRIO", SoCG 2003; Shewchuk 1997 (predicates).

use super::functions::strictly_between_collinear;
use super::types::{DelaunayTri, Point2};
use oxiphysics_core::exact_predicates::{Orientation, incircle, orient2d};

/// Vertex index of the point at infinity shared by all ghost triangles.
const GHOST: usize = usize::MAX;

/// "No triangle" marker for unset adjacency and scratch entries.
const NONE: usize = usize::MAX;

/// The coordinates of a point as the array the exact predicates take.
#[inline]
fn xy(p: Point2) -> [f64; 2] {
    [p.x, p.y]
}

/// Compute the Delaunay triangulation of a 2-D point set.
///
/// Returns the triangles as indices into `points`, each in counter-clockwise
/// order; every triangle's circumcircle is free of other input points
/// (cocircular points on the circle are allowed, in which case the choice among
/// the equally valid triangulations is deterministic).  The triangles tile the
/// convex hull of the input, collinear hull points included, so their number is
/// `2 n - h - 2` for `n` distinct points of which `h` lie on the hull boundary.
///
/// Duplicate points (identical coordinates) are triangulated once: only one of
/// the copies (the first in the internal insertion order) appears in the
/// triangles and the others are skipped; points with a non-finite coordinate are
/// ignored.
/// Fewer than three distinct points, or points that are all collinear, give an
/// empty result.
///
/// All decisions use exact arithmetic (Shewchuk's adaptive `orient2d` and
/// `incircle`), so the result does not depend on the scale of the coordinates.
/// The running time is expected `O(n log n)` thanks to a biased randomized
/// insertion order along a space-filling curve with walk-based point location.
pub fn delaunay_2d(points: &[Point2]) -> Vec<DelaunayTri> {
    match triangulate(points) {
        Some(mesh) => mesh.into_triangles(),
        None => Vec::new(),
    }
}

/// Build the triangulation, or `None` when no triangle exists.
fn triangulate(points: &[Point2]) -> Option<Mesh<'_>> {
    let candidates: Vec<usize> = (0..points.len())
        .filter(|&i| points[i].x.is_finite() && points[i].y.is_finite())
        .collect();
    if candidates.len() < 3 {
        return None;
    }
    let order = insertion_order(points, &candidates);
    let (a, b, c) = initial_triangle(points, &order)?;
    let mut mesh = Mesh::new(points, a, b, c);
    let mut hint = 0;
    for &i in &order {
        if i != a && i != b && i != c {
            hint = mesh.insert(i, hint);
        }
    }
    Some(mesh)
}

/// The first three points of `order` that span a triangle: the first point, the
/// first point different from it, and the first point not collinear with those
/// two.  The triple is returned counter-clockwise.  `None` when all points are
/// equal or collinear.
fn initial_triangle(points: &[Point2], order: &[usize]) -> Option<(usize, usize, usize)> {
    let a = *order.first()?;
    let b = *order.iter().find(|&&i| points[i] != points[a])?;
    let c = *order.iter().find(|&&i| {
        orient2d(xy(points[a]), xy(points[b]), xy(points[i])) != Orientation::Degenerate
    })?;
    if orient2d(xy(points[a]), xy(points[b]), xy(points[c])) == Orientation::Positive {
        Some((a, b, c))
    } else {
        Some((a, c, b))
    }
}

/// Deterministic SplitMix64 generator for the insertion rounds.
struct SplitMix64(u64);

impl SplitMix64 {
    fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }
}

/// Bits per axis of the Hilbert-curve grid used to order the points.
const HILBERT_BITS: u32 = 20;

/// Position of the cell `(x, y)` along the Hilbert curve of the
/// `2^HILBERT_BITS x 2^HILBERT_BITS` grid.
fn hilbert_index(mut x: u32, mut y: u32) -> u64 {
    let side: u32 = 1 << HILBERT_BITS;
    let mut index: u64 = 0;
    let mut s = side >> 1;
    while s > 0 {
        let rx = u32::from(x & s != 0);
        let ry = u32::from(y & s != 0);
        index += u64::from(s) * u64::from(s) * u64::from((3 * rx) ^ ry);
        if ry == 0 {
            if rx == 1 {
                x = side - 1 - x;
                y = side - 1 - y;
            }
            std::mem::swap(&mut x, &mut y);
        }
        s >>= 1;
    }
    index
}

/// Biased randomized insertion order of the `candidates`: round `r` holds about
/// `n / 2^(rounds - r)` points chosen by coin flips, rounds are inserted from
/// the smallest to the largest, and every round is sorted along a Hilbert curve
/// over the bounding box (so the order does not depend on the scale).
fn insertion_order(points: &[Point2], candidates: &[usize]) -> Vec<usize> {
    let (mut min_x, mut min_y) = (f64::INFINITY, f64::INFINITY);
    let (mut max_x, mut max_y) = (f64::NEG_INFINITY, f64::NEG_INFINITY);
    for &i in candidates {
        min_x = min_x.min(points[i].x);
        max_x = max_x.max(points[i].x);
        min_y = min_y.min(points[i].y);
        max_y = max_y.max(points[i].y);
    }
    let extent = (max_x - min_x).max(max_y - min_y);
    let cells = f64::from((1u32 << HILBERT_BITS) - 1);
    let quantize = |value: f64, low: f64| -> u32 {
        if extent > 0.0 && extent.is_finite() {
            (((value - low) / extent).clamp(0.0, 1.0) * cells) as u32
        } else {
            0
        }
    };
    let rounds = (usize::BITS - candidates.len().leading_zeros()).max(1);
    let mut rng = SplitMix64(0x0123_4567_89AB_CDEF);
    let mut keyed: Vec<(u32, u64, usize)> = candidates
        .iter()
        .map(|&i| {
            let flips = rng.next_u64().trailing_zeros().min(rounds - 1);
            let round = rounds - 1 - flips;
            let key = hilbert_index(quantize(points[i].x, min_x), quantize(points[i].y, min_y));
            (round, key, i)
        })
        .collect();
    keyed.sort_unstable();
    keyed.into_iter().map(|(_, _, i)| i).collect()
}

/// The triangle of the vertices `x -> y` (a counter-clockwise boundary edge of
/// a cavity) and the new vertex `p`, stored with the ghost vertex, if any, last.
fn new_triangle(x: usize, y: usize, p: usize) -> [usize; 3] {
    if x == GHOST {
        [y, p, GHOST]
    } else if y == GHOST {
        [p, x, GHOST]
    } else {
        [x, y, p]
    }
}

/// A boundary edge of the insertion cavity.
struct BoundaryEdge {
    /// Start of the edge in the counter-clockwise order of the cavity triangle.
    x: usize,
    /// End of the edge.
    y: usize,
    /// Triangle across the edge, outside the cavity.
    outer: usize,
    /// Neighbour slot of `outer` that points back into the cavity.
    outer_slot: usize,
}

/// The triangulation under construction.
///
/// Real triangles are stored counter-clockwise.  A ghost triangle `[u, v,
/// GHOST]` stands for the half-plane to the left of the directed edge `u -> v`
/// (the outside of the hull across the hull edge `v -> u`).  `nbrs[t][k]` is
/// the triangle across the edge opposite vertex `k` of triangle `t`, so every
/// triangle always has exactly three neighbours.
struct Mesh<'a> {
    points: &'a [Point2],
    verts: Vec<[usize; 3]>,
    nbrs: Vec<[usize; 3]>,
    alive: Vec<bool>,
    /// Dead triangle slots available for reuse.
    free: Vec<usize>,
    /// Visit stamps of the cavity flood: `2 * epoch` marks a triangle of the
    /// cavity under construction, `2 * epoch + 1` one known to be outside it.
    stamp: Vec<u32>,
    epoch: u32,
    /// Per vertex (the ghost has slot `points.len()`): the new triangle whose
    /// cavity-boundary edge starts / ends there; scratch for restitching.
    start_of: Vec<usize>,
    end_of: Vec<usize>,
    /// Scratch buffers reused across insertions.
    cavity: Vec<usize>,
    stack: Vec<usize>,
    boundary: Vec<BoundaryEdge>,
    created: Vec<usize>,
}

impl<'a> Mesh<'a> {
    /// The mesh of the single triangle `(a, b, c)` (counter-clockwise) closed by
    /// its three ghost triangles.
    fn new(points: &'a [Point2], a: usize, b: usize, c: usize) -> Self {
        let capacity = 2 * points.len() + 8;
        let mut verts = Vec::with_capacity(capacity);
        let mut nbrs = Vec::with_capacity(capacity);
        // Triangle 0 is the real one; triangles 1, 2, 3 are the ghosts across
        // its edges a->b, b->c and c->a.
        verts.extend([[a, b, c], [b, a, GHOST], [c, b, GHOST], [a, c, GHOST]]);
        nbrs.extend([[2, 3, 1], [3, 2, 0], [1, 3, 0], [2, 1, 0]]);
        Self {
            points,
            verts,
            nbrs,
            alive: vec![true; 4],
            free: Vec::new(),
            stamp: Vec::new(),
            epoch: 0,
            start_of: vec![NONE; points.len() + 1],
            end_of: vec![NONE; points.len() + 1],
            cavity: Vec::new(),
            stack: Vec::new(),
            boundary: Vec::new(),
            created: Vec::new(),
        }
    }

    /// Scratch slot of vertex `v` (the ghost is stored after the real points).
    #[inline]
    fn vertex_slot(&self, v: usize) -> usize {
        if v == GHOST { self.points.len() } else { v }
    }

    /// `true` if triangle `t` is in conflict with the point `p`: `p` lies
    /// strictly inside the circumcircle of a real triangle, or strictly outside
    /// the hull across (or on the open hull edge of) a ghost triangle.
    fn in_conflict(&self, t: usize, p: Point2) -> bool {
        let [a, b, c] = self.verts[t];
        if c == GHOST {
            let (pa, pb) = (self.points[a], self.points[b]);
            return match orient2d(xy(pa), xy(pb), xy(p)) {
                Orientation::Positive => true,
                Orientation::Negative => false,
                Orientation::Degenerate => strictly_between_collinear(pa, p, pb),
            };
        }
        incircle(
            xy(self.points[a]),
            xy(self.points[b]),
            xy(self.points[c]),
            xy(p),
        ) == Orientation::Positive
    }

    /// Locate a triangle in conflict with `p` by walking from `hint`: a real
    /// triangle containing `p` in its closed region, or a ghost triangle `p` is
    /// outside of.  `None` when `p` coincides with a vertex of the mesh.
    ///
    /// The walk crosses an edge only when `p` is strictly on its far side; the
    /// edges are examined from a rotating starting edge, and a step budget
    /// backs it with an exhaustive scan.
    fn locate(&self, p: Point2, hint: usize) -> Option<usize> {
        let mut t = if self.alive[hint] {
            hint
        } else {
            self.alive.iter().position(|&alive| alive)?
        };
        let budget = 4 * self.verts.len() + 16;
        'walk: for step in 0..budget {
            let [a, b, c] = self.verts[t];
            if c == GHOST {
                if self.in_conflict(t, p) {
                    return Some(t);
                }
                // Not outside across this hull edge: continue in the real
                // triangle on the other side of it.
                t = self.nbrs[t][2];
                continue;
            }
            let corners = [a, b, c];
            for offset in 0..3 {
                let k = (step + offset) % 3;
                let from = self.points[corners[(k + 1) % 3]];
                let to = self.points[corners[(k + 2) % 3]];
                if orient2d(xy(from), xy(to), xy(p)) == Orientation::Negative {
                    t = self.nbrs[t][k];
                    continue 'walk;
                }
            }
            return if corners.iter().any(|&v| self.points[v] == p) {
                None
            } else {
                Some(t)
            };
        }
        self.scan_for(p)
    }

    /// Exhaustive fallback of [`Mesh::locate`].
    fn scan_for(&self, p: Point2) -> Option<usize> {
        for t in (0..self.verts.len()).filter(|&t| self.alive[t]) {
            let [a, b, c] = self.verts[t];
            if c == GHOST {
                if self.in_conflict(t, p) {
                    return Some(t);
                }
                continue;
            }
            let corners = [a, b, c];
            let inside = (0..3).all(|k| {
                let from = self.points[corners[(k + 1) % 3]];
                let to = self.points[corners[(k + 2) % 3]];
                orient2d(xy(from), xy(to), xy(p)) != Orientation::Negative
            });
            if inside {
                return if corners.iter().any(|&v| self.points[v] == p) {
                    None
                } else {
                    Some(t)
                };
            }
        }
        None
    }

    /// Start a new cavity flood: cover every triangle slot with a stamp and move
    /// to a fresh epoch, which un-marks all triangles at once.
    fn begin_epoch(&mut self) {
        if self.stamp.len() < self.verts.len() {
            self.stamp.resize(self.verts.len(), 0);
        }
        if self.epoch >= u32::MAX / 2 - 1 {
            self.stamp.fill(0);
            self.epoch = 0;
        }
        self.epoch += 1;
    }

    /// Store `triangle` in a recycled or a new slot and return the slot.
    fn allocate(&mut self, triangle: [usize; 3]) -> usize {
        if let Some(slot) = self.free.pop() {
            self.verts[slot] = triangle;
            self.nbrs[slot] = [NONE; 3];
            self.alive[slot] = true;
            slot
        } else {
            self.verts.push(triangle);
            self.nbrs.push([NONE; 3]);
            self.alive.push(true);
            self.verts.len() - 1
        }
    }

    /// Insert the point `p_idx`; returns a triangle near it to start the next
    /// walk from.  A point that coincides with an existing vertex is skipped.
    fn insert(&mut self, p_idx: usize, hint: usize) -> usize {
        let p = self.points[p_idx];
        let Some(seed) = self.locate(p, hint) else {
            return hint;
        };
        self.begin_epoch();
        let cavity_mark = 2 * self.epoch;
        let outside_mark = cavity_mark + 1;
        let mut cavity = std::mem::take(&mut self.cavity);
        let mut stack = std::mem::take(&mut self.stack);
        let mut boundary = std::mem::take(&mut self.boundary);
        let mut created = std::mem::take(&mut self.created);
        cavity.clear();
        stack.clear();
        boundary.clear();
        created.clear();

        // Flood the conflict region through the adjacency.
        self.stamp[seed] = cavity_mark;
        stack.push(seed);
        while let Some(t) = stack.pop() {
            cavity.push(t);
            for k in 0..3 {
                let nb = self.nbrs[t][k];
                let mark = self.stamp[nb];
                if mark == cavity_mark || mark == outside_mark {
                    continue;
                }
                if self.in_conflict(nb, p) {
                    self.stamp[nb] = cavity_mark;
                    stack.push(nb);
                } else {
                    self.stamp[nb] = outside_mark;
                }
            }
        }

        // Boundary of the cavity, with the back-pointer slot of every outer
        // triangle (read before anything is rewritten).
        let mut consistent = true;
        for &t in &cavity {
            let corners = self.verts[t];
            for k in 0..3 {
                let outer = self.nbrs[t][k];
                if self.stamp[outer] == cavity_mark {
                    continue;
                }
                let Some(outer_slot) = self.nbrs[outer].iter().position(|&n| n == t) else {
                    consistent = false;
                    continue;
                };
                boundary.push(BoundaryEdge {
                    x: corners[(k + 1) % 3],
                    y: corners[(k + 2) % 3],
                    outer,
                    outer_slot,
                });
            }
        }

        let mut next_hint = hint;
        if consistent {
            // Replace the cavity by the fan of triangles from `p` to its boundary.
            for &t in &cavity {
                self.alive[t] = false;
                self.free.push(t);
            }
            for edge in &boundary {
                let triangle = new_triangle(edge.x, edge.y, p_idx);
                let idx = self.allocate(triangle);
                if let Some(slot_p) = triangle.iter().position(|&v| v == p_idx) {
                    self.nbrs[idx][slot_p] = edge.outer;
                }
                self.nbrs[edge.outer][edge.outer_slot] = idx;
                let (start, end) = (self.vertex_slot(edge.x), self.vertex_slot(edge.y));
                self.start_of[start] = idx;
                self.end_of[end] = idx;
                created.push(idx);
            }
            // The new triangles share the edges that contain `p`: the edge
            // `y -> p` with the triangle whose boundary edge starts at `y`, the
            // edge `p -> x` with the one whose boundary edge ends at `x`.
            for (edge, &idx) in boundary.iter().zip(created.iter()) {
                let triangle = self.verts[idx];
                let across_y = self.start_of[self.vertex_slot(edge.y)];
                let across_x = self.end_of[self.vertex_slot(edge.x)];
                if let Some(slot_x) = triangle.iter().position(|&v| v == edge.x) {
                    self.nbrs[idx][slot_x] = across_y;
                }
                if let Some(slot_y) = triangle.iter().position(|&v| v == edge.y) {
                    self.nbrs[idx][slot_y] = across_x;
                }
            }
            if let Some(&last) = created.last() {
                next_hint = last;
            }
        }

        self.cavity = cavity;
        self.stack = stack;
        self.boundary = boundary;
        self.created = created;
        next_hint
    }

    /// The real triangles of the mesh, counter-clockwise.
    fn into_triangles(self) -> Vec<DelaunayTri> {
        self.verts
            .iter()
            .zip(self.alive.iter())
            .filter(|(corners, alive)| **alive && corners[2] != GHOST)
            .map(|(corners, _)| DelaunayTri::new(corners[0], corners[1], corners[2]))
            .collect()
    }
}

#[cfg(test)]
impl Mesh<'_> {
    /// Check the structural invariants of the mesh: every live triangle has
    /// three live neighbours that point back across the same edge (reversed),
    /// ghosts keep the ghost vertex last, real triangles are strictly
    /// counter-clockwise and the Euler relation of the closed surface holds.
    fn check_invariants(&self) -> Result<(), String> {
        let live: Vec<usize> = (0..self.verts.len()).filter(|&t| self.alive[t]).collect();
        let mut ghosts = 0usize;
        let mut vertices = std::collections::BTreeSet::new();
        for &t in &live {
            let corners = self.verts[t];
            let ghost_count = corners.iter().filter(|&&v| v == GHOST).count();
            if ghost_count > 1 || (ghost_count == 1 && corners[2] != GHOST) {
                return Err(format!(
                    "triangle {t} {corners:?} misplaces the ghost vertex"
                ));
            }
            if ghost_count == 1 {
                ghosts += 1;
            } else if orient2d(
                xy(self.points[corners[0]]),
                xy(self.points[corners[1]]),
                xy(self.points[corners[2]]),
            ) != Orientation::Positive
            {
                return Err(format!(
                    "real triangle {t} {corners:?} is not counter-clockwise"
                ));
            }
            vertices.extend(corners.iter().copied());
            for k in 0..3 {
                let nb = self.nbrs[t][k];
                if nb == NONE || nb >= self.verts.len() || !self.alive[nb] {
                    return Err(format!("triangle {t} has a dead or missing neighbour {nb}"));
                }
                // The edge opposite vertex k, reversed, must be an edge of nb
                // whose own neighbour across it is t.
                let (u, v) = (corners[(k + 1) % 3], corners[(k + 2) % 3]);
                let other = self.verts[nb];
                let slot = (0..3).find(|&j| other[(j + 1) % 3] == v && other[(j + 2) % 3] == u);
                match slot {
                    Some(j) if self.nbrs[nb][j] == t => {}
                    _ => {
                        return Err(format!(
                            "triangles {t} and {nb} disagree about their shared edge {u}->{v}"
                        ));
                    }
                }
            }
        }
        // Closed triangulated sphere: V - E + F = 2 with E = 3F / 2.
        let faces = live.len();
        if faces % 2 != 0 {
            return Err(format!("odd triangle count {faces}"));
        }
        let edges = 3 * faces / 2;
        if vertices.len() + faces != edges + 2 {
            return Err(format!(
                "Euler relation violated: {} vertices, {edges} edges, {faces} faces",
                vertices.len()
            ));
        }
        if ghosts < 3 {
            return Err(format!(
                "a hull has at least 3 edges, found {ghosts} ghosts"
            ));
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;

    struct Xorshift(u64);

    impl Xorshift {
        fn next_f64(&mut self) -> f64 {
            self.0 ^= self.0 << 13;
            self.0 ^= self.0 >> 7;
            self.0 ^= self.0 << 17;
            (self.0 >> 11) as f64 / (1u64 << 53) as f64
        }
    }

    fn pt(x: f64, y: f64) -> Point2 {
        Point2::new(x, y)
    }

    fn random_points(seed: u64, n: usize) -> Vec<Point2> {
        let mut rng = Xorshift(seed);
        (0..n).map(|_| pt(rng.next_f64(), rng.next_f64())).collect()
    }

    fn grid(m: usize, spacing: f64) -> Vec<Point2> {
        (0..m * m)
            .map(|k| pt((k % m) as f64 * spacing, (k / m) as f64 * spacing))
            .collect()
    }

    /// Distinct points on the boundary of the convex hull (collinear ones
    /// included) by an exact monotone-chain scan.
    fn hull_boundary_count(points: &[Point2]) -> usize {
        let mut sorted: Vec<Point2> = points.to_vec();
        sorted.sort_by(|a, b| a.x.total_cmp(&b.x).then(a.y.total_cmp(&b.y)));
        sorted.dedup();
        if sorted.len() < 3 {
            return sorted.len();
        }
        let turn = |a: Point2, b: Point2, c: Point2| orient2d(xy(a), xy(b), xy(c));
        let chain = |pts: &mut dyn Iterator<Item = Point2>| -> Vec<Point2> {
            let mut out: Vec<Point2> = Vec::new();
            for p in pts {
                while out.len() >= 2
                    && turn(out[out.len() - 2], out[out.len() - 1], p) == Orientation::Negative
                {
                    out.pop();
                }
                out.push(p);
            }
            out
        };
        let lower = chain(&mut sorted.iter().copied());
        let upper = chain(&mut sorted.iter().rev().copied());
        // Both chains keep collinear points; the shared end points count once.
        let mut boundary: Vec<Point2> = lower;
        boundary.extend(upper);
        boundary.sort_by(|a, b| a.x.total_cmp(&b.x).then(a.y.total_cmp(&b.y)));
        boundary.dedup();
        boundary.len()
    }

    fn distinct_count(points: &[Point2]) -> usize {
        let mut sorted: Vec<Point2> = points.to_vec();
        sorted.sort_by(|a, b| a.x.total_cmp(&b.x).then(a.y.total_cmp(&b.y)));
        sorted.dedup();
        sorted.len()
    }

    fn twice_area(points: &[Point2], tri: &DelaunayTri) -> f64 {
        let (a, b, c) = (points[tri.a], points[tri.b], points[tri.c]);
        (b.x - a.x) * (c.y - a.y) - (b.y - a.y) * (c.x - a.x)
    }

    /// Every triangle is strictly counter-clockwise.
    fn assert_counter_clockwise(points: &[Point2], tris: &[DelaunayTri]) {
        for t in tris {
            assert_eq!(
                orient2d(xy(points[t.a]), xy(points[t.b]), xy(points[t.c])),
                Orientation::Positive,
                "triangle {t:?} is not counter-clockwise"
            );
        }
    }

    /// Exact empty-circumcircle audit against every point.
    fn assert_empty_circumcircles(points: &[Point2], tris: &[DelaunayTri]) {
        for t in tris {
            for (i, &p) in points.iter().enumerate() {
                if i == t.a || i == t.b || i == t.c {
                    continue;
                }
                assert_ne!(
                    incircle(xy(points[t.a]), xy(points[t.b]), xy(points[t.c]), xy(p)),
                    Orientation::Positive,
                    "point {i} lies inside the circumcircle of {t:?}"
                );
            }
        }
    }

    /// The Delaunay lemma: every interior edge is locally Delaunay.  Also
    /// verifies that each directed edge occurs at most once (a valid
    /// triangulation) and returns the number of boundary edges.
    fn assert_locally_delaunay(points: &[Point2], tris: &[DelaunayTri]) -> usize {
        let mut owner: HashMap<(usize, usize), usize> = HashMap::new();
        for (ti, t) in tris.iter().enumerate() {
            for (u, v) in [(t.a, t.b), (t.b, t.c), (t.c, t.a)] {
                assert!(
                    owner.insert((u, v), ti).is_none(),
                    "directed edge {u}->{v} used twice"
                );
            }
        }
        let mut boundary = 0;
        for t in tris {
            for (u, v, apex) in [(t.a, t.b, t.c), (t.b, t.c, t.a), (t.c, t.a, t.b)] {
                match owner.get(&(v, u)) {
                    None => boundary += 1,
                    Some(&other) => {
                        let o = &tris[other];
                        let opposite = [o.a, o.b, o.c]
                            .into_iter()
                            .find(|&w| w != u && w != v)
                            .expect("a triangle has a vertex off a given edge");
                        assert_ne!(
                            incircle(
                                xy(points[u]),
                                xy(points[v]),
                                xy(points[apex]),
                                xy(points[opposite])
                            ),
                            Orientation::Positive,
                            "edge {u}-{v} is not locally Delaunay"
                        );
                    }
                }
            }
        }
        boundary
    }

    /// The structural invariants of the internal mesh and the number of
    /// triangles `2 n - h - 2`.
    fn assert_valid_triangulation(points: &[Point2]) -> Vec<DelaunayTri> {
        let mesh = triangulate(points).expect("the points span a triangle");
        mesh.check_invariants().expect("mesh invariants");
        let tris = mesh.into_triangles();
        assert_counter_clockwise(points, &tris);
        let n = distinct_count(points);
        let h = hull_boundary_count(points);
        assert_eq!(tris.len(), 2 * n - h - 2, "triangle count for n={n}, h={h}");
        let boundary_edges = assert_locally_delaunay(points, &tris);
        assert_eq!(boundary_edges, h, "boundary edges are the hull edges");
        tris
    }

    #[test]
    fn fewer_than_three_distinct_points_or_collinear_points_give_nothing() {
        assert!(delaunay_2d(&[]).is_empty());
        assert!(delaunay_2d(&[pt(0.0, 0.0), pt(1.0, 0.0)]).is_empty());
        assert!(delaunay_2d(&[pt(1.0, 1.0), pt(1.0, 1.0), pt(1.0, 1.0)]).is_empty());
        let line: Vec<Point2> = (0..20)
            .map(|i| pt(f64::from(i), 2.0 * f64::from(i)))
            .collect();
        assert!(delaunay_2d(&line).is_empty());
        let horizontal: Vec<Point2> = (0..9).map(|i| pt(f64::from(i) * 0.1, 3.0)).collect();
        assert!(delaunay_2d(&horizontal).is_empty());
        // Non-finite coordinates are ignored.
        let with_nan = [
            pt(0.0, 0.0),
            pt(1.0, 0.0),
            pt(f64::NAN, 1.0),
            pt(0.0, f64::INFINITY),
        ];
        assert!(delaunay_2d(&with_nan).is_empty());
    }

    #[test]
    fn single_triangle_and_square() {
        let tris = delaunay_2d(&[pt(0.0, 0.0), pt(4.0, 0.0), pt(1.0, 3.0)]);
        assert_eq!(tris.len(), 1);
        // Clockwise input is reported counter-clockwise.
        let tris = delaunay_2d(&[pt(0.0, 0.0), pt(1.0, 3.0), pt(4.0, 0.0)]);
        assert_eq!(tris.len(), 1);
        assert_counter_clockwise(&[pt(0.0, 0.0), pt(1.0, 3.0), pt(4.0, 0.0)], &tris);
        let square = [pt(0.0, 0.0), pt(1.0, 0.0), pt(1.0, 1.0), pt(0.0, 1.0)];
        let tris = assert_valid_triangulation(&square);
        assert_eq!(tris.len(), 2);
        let area: f64 = tris.iter().map(|t| twice_area(&square, t)).sum::<f64>() / 2.0;
        assert!((area - 1.0).abs() < 1e-15);
    }

    #[test]
    fn random_points_are_delaunay_with_the_exact_triangle_count() {
        for (seed, n) in [(0x1111_u64, 5), (0x2222, 17), (0x3333, 64), (0x4444, 300)] {
            let points = random_points(seed, n);
            let tris = assert_valid_triangulation(&points);
            assert_empty_circumcircles(&points, &tris);
            // The triangles tile the hull: total area equals the hull area.
            let area: f64 = tris.iter().map(|t| twice_area(&points, t)).sum::<f64>() / 2.0;
            assert!(area > 0.0 && area <= 1.0 + 1e-12, "area {area}");
        }
    }

    #[test]
    fn hull_triangles_with_large_circumcircles_are_kept() {
        // A flat arc: the hull triangles near the ends have circumcircles that
        // reach far beyond the point set.  A finite super-triangle drops them.
        let mut points: Vec<Point2> = (0..40)
            .map(|i| {
                let x = f64::from(i) * 0.25;
                pt(x, 1e-3 * (x - 5.0) * (x - 5.0))
            })
            .collect();
        points.push(pt(5.0, 50.0));
        let tris = assert_valid_triangulation(&points);
        assert_empty_circumcircles(&points, &tris);
    }

    #[test]
    fn grids_with_cocircular_points_triangulate_completely() {
        for m in [2usize, 3, 4, 7, 12] {
            let points = grid(m, 1.0);
            let tris = assert_valid_triangulation(&points);
            assert_eq!(tris.len(), 2 * (m - 1) * (m - 1), "{m} x {m} grid");
            assert_empty_circumcircles(&points, &tris);
        }
    }

    #[test]
    fn triangulation_is_scale_invariant() {
        // The same grid at the unit scale and at 1e-6 (and 1e6): identical
        // triangle counts, no point silently lost to an absolute tolerance.
        for m in [5usize, 9, 16] {
            let unit = delaunay_2d(&grid(m, 1.0));
            assert_eq!(unit.len(), 2 * (m - 1) * (m - 1));
            for spacing in [1e-6, 1e-12, 1e6, 3e-3] {
                let scaled_points = grid(m, spacing);
                let scaled = delaunay_2d(&scaled_points);
                assert_eq!(scaled.len(), unit.len(), "{m} x {m} grid at {spacing}");
                assert_counter_clockwise(&scaled_points, &scaled);
                assert_locally_delaunay(&scaled_points, &scaled);
            }
        }
        // Random points: the combinatorics are identical after a power-of-two
        // scaling (exactly representable, so every predicate keeps its sign).
        let points = random_points(0x5CA1E, 200);
        let base = delaunay_2d(&points);
        let scaled_points: Vec<Point2> = points
            .iter()
            .map(|p| pt(p.x * 2f64.powi(-40), p.y * 2f64.powi(-40)))
            .collect();
        let scaled = delaunay_2d(&scaled_points);
        assert_eq!(base, scaled);
    }

    #[test]
    fn cocircular_points_and_degenerate_hulls() {
        // 24 points on a circle plus its centre: any triangulation of the polygon
        // is Delaunay, the centre is joined to all of them.
        let mut points: Vec<Point2> = (0..24)
            .map(|i| {
                let a = f64::from(i) * std::f64::consts::TAU / 24.0;
                pt(a.cos(), a.sin())
            })
            .collect();
        points.push(pt(0.0, 0.0));
        assert_valid_triangulation(&points);
        // Points on the hull edges (collinear boundary points).
        let mut boundary_points = vec![pt(0.0, 0.0), pt(4.0, 0.0), pt(4.0, 4.0), pt(0.0, 4.0)];
        boundary_points.extend([
            pt(1.0, 0.0),
            pt(2.0, 0.0),
            pt(3.0, 0.0),
            pt(4.0, 2.0),
            pt(0.0, 3.0),
        ]);
        boundary_points.push(pt(2.0, 2.0));
        let tris = assert_valid_triangulation(&boundary_points);
        assert_empty_circumcircles(&boundary_points, &tris);
        // A fan: many collinear points plus one apex.
        let mut fan: Vec<Point2> = (0..30).map(|i| pt(f64::from(i), 0.0)).collect();
        fan.push(pt(13.5, 4.0));
        let tris = assert_valid_triangulation(&fan);
        assert_eq!(tris.len(), 29);
    }

    #[test]
    fn duplicates_are_triangulated_once() {
        let base = random_points(0xD0D0, 60);
        let plain = assert_valid_triangulation(&base);
        let mut with_duplicates = base.clone();
        for k in 0..40 {
            with_duplicates.push(base[(k * 7) % base.len()]);
        }
        let tris = assert_valid_triangulation(&with_duplicates);
        assert_eq!(tris.len(), plain.len());
        assert_empty_circumcircles(&with_duplicates, &tris);
        // Of several copies of a location only one is used by the triangles,
        // and every distinct location is used.
        let mut used: Vec<(u64, u64)> = Vec::new();
        for t in &tris {
            for v in [t.a, t.b, t.c] {
                used.push((
                    with_duplicates[v].x.to_bits(),
                    with_duplicates[v].y.to_bits(),
                ));
            }
        }
        used.sort_unstable();
        used.dedup();
        assert_eq!(used.len(), distinct_count(&with_duplicates));
    }

    #[test]
    fn insertion_order_does_not_change_the_triangle_set_in_general_position() {
        let points = random_points(0x0DDB_A11, 120);
        let tris = delaunay_2d(&points);
        let canonical = |tris: &[DelaunayTri], ids: &[usize]| -> Vec<[usize; 3]> {
            let mut out: Vec<[usize; 3]> = tris
                .iter()
                .map(|t| {
                    let mut tri = [ids[t.a], ids[t.b], ids[t.c]];
                    tri.sort_unstable();
                    tri
                })
                .collect();
            out.sort_unstable();
            out
        };
        let identity: Vec<usize> = (0..points.len()).collect();
        let want = canonical(&tris, &identity);
        // Same points in reverse order: the Delaunay triangulation of a set in
        // general position is unique.
        let reversed: Vec<Point2> = points.iter().rev().copied().collect();
        let reversed_ids: Vec<usize> = (0..points.len()).rev().collect();
        let got = canonical(&delaunay_2d(&reversed), &reversed_ids);
        assert_eq!(got, want);
    }

    #[test]
    fn hilbert_curve_visits_neighbouring_cells_consecutively() {
        // The 4 x 4 block of cells at the origin is the start of the curve: its
        // indices are exactly 0..16 and consecutive cells are edge neighbours.
        let mut cells: Vec<(u64, u32, u32)> = Vec::new();
        for x in 0..4u32 {
            for y in 0..4u32 {
                cells.push((hilbert_index(x, y), x, y));
            }
        }
        cells.sort_unstable();
        let indices: Vec<u64> = cells.iter().map(|c| c.0).collect();
        assert_eq!(indices, (0..16).collect::<Vec<u64>>());
        for pair in cells.windows(2) {
            let step = pair[0].1.abs_diff(pair[1].1) + pair[0].2.abs_diff(pair[1].2);
            assert_eq!(step, 1, "{:?} -> {:?}", pair[0], pair[1]);
        }
        // Cells of the same 2 x 2 quadrant are visited contiguously.
        let quadrant = |c: &(u64, u32, u32)| (c.1 / 2, c.2 / 2);
        let mut changes = 0;
        for pair in cells.windows(2) {
            if quadrant(&pair[0]) != quadrant(&pair[1]) {
                changes += 1;
            }
        }
        assert!(changes <= 3, "{changes} quadrant changes");
    }

    #[test]
    fn large_random_cloud_is_fast_and_delaunay() {
        // 1e5 points: the former Bowyer-Watson rebuilt the list of all triangles
        // for every insertion (quadratic).
        let n = 100_000;
        let points = random_points(0xBEEF_F00D, n);
        let start = std::time::Instant::now();
        let tris = delaunay_2d(&points);
        let elapsed = start.elapsed();
        assert!(elapsed.as_secs_f64() < 30.0, "{n} points took {elapsed:?}");
        let h = hull_boundary_count(&points);
        assert_eq!(tris.len(), 2 * n - h - 2);
        assert_counter_clockwise(&points, &tris);
        let boundary = assert_locally_delaunay(&points, &tris);
        assert_eq!(boundary, h);
        // Global audit of a sample of triangles against all points.
        for t in tris.iter().step_by(tris.len() / 40 + 1) {
            for (i, &p) in points.iter().enumerate() {
                if i == t.a || i == t.b || i == t.c {
                    continue;
                }
                assert_ne!(
                    incircle(xy(points[t.a]), xy(points[t.b]), xy(points[t.c]), xy(p)),
                    Orientation::Positive,
                    "point {i} lies inside the circumcircle of {t:?}"
                );
            }
        }
    }
}
