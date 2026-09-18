// Copyright 2026 COOLJAPAN OU (Team KitaSan)
// SPDX-License-Identifier: Apache-2.0

//! Straight skeleton and mitred offset curves of a simple 2-D polygon.
//!
//! The straight skeleton (Aichholzer, Aurenhammer, Alberts & Gärtner, 1995) is
//! the trace of the vertices of the polygon's *wavefront*: every edge moves
//! inwards at unit speed along its normal and the corners follow the
//! bisectors of the moving edge lines, so that the wavefront at time `t` is
//! the mitred offset curve of the polygon at distance `t`. Two kinds of events
//! change the wavefront's topology (Felkel & Obdržálek, 1998):
//!
//! * an **edge event**: an edge shrinks to a point and its two end vertices
//!   merge into one;
//! * a **split event**: a reflex vertex hits an opposite edge and splits the
//!   wavefront polygon in two.
//!
//! The arcs of the skeleton are the paths of the wavefront vertices from their
//! birth (a polygon vertex or an event) to their death. The same simulation,
//! stopped at time `t`, yields the offset rings at distance `t`; run on the
//! reversed polygon it yields the outward offsets.
//!
//! # Implementation
//!
//! A wavefront vertex is the intersection of two moving edge lines
//! `n · (x - o) = t`; its velocity is `(n1 + n2) / (1 + n1 · n2)`, so it moves
//! `1 / cos(θ/2)` times faster than the edges for edge normals at angle `θ`.
//! Events are kept in a priority queue ordered by time; vertex records never
//! change after creation, so a queued event is valid exactly while the vertices
//! it names are alive (and, for a split, while its hit point still lies on a
//! piece of the hit edge's wavefront at the event time, checked when the event
//! is popped).
//!
//! Real polygons are full of simultaneous events (rectangles, regular
//! polygons, symmetric shapes), which are resolved at the moment they occur
//! by local clean-up rules rather than being avoided by perturbation:
//!
//! * adjacent vertices at one point whose edge is not growing are merged;
//! * a vertex between two *antiparallel* edges (a zero-width strip, an
//!   infinitely fast "needle") is resolved against its neighbours: the strip
//!   collapses along its length and emits its ridge as an arc;
//! * a wavefront polygon with zero area (all vertices on one line, including
//!   two-vertex and single-point polygons) is retired and its ridge emitted.
//!
//! All comparisons use tolerances relative to the extent of the polygon.

use crate::error::{Error, Result};
use std::cmp::Ordering;
use std::collections::{BinaryHeap, HashSet};

type P2 = [f64; 2];

/// Position tolerance relative to the extent of the polygon.
const POSITION_TOLERANCE: f64 = 1e-9;
/// Speed differences below this are treated as zero: two vertices meeting
/// with a smaller relative speed stay within [`POSITION_TOLERANCE`] of each
/// other for the whole life of the wavefront (a time of the order of the
/// polygon's extent).
const RATE_EPS: f64 = POSITION_TOLERANCE;
/// The edge directions `d_in`, `d_out` of a vertex with `d_in · d_out < 0` and
/// `|d_in × d_out|` below this are antiparallel (a needle): the sine of the
/// angle is known to `1e-16`, so below `1e-8` the vertex velocity
/// `(n1 + n2) / (1 + n1 · n2)` would be dominated by rounding.
const NEEDLE_SINE: f64 = 1e-8;
/// `cross(d_in, d_out)` below `-REFLEX_EPS` marks a reflex vertex.
const REFLEX_EPS: f64 = 1e-12;

fn sub(a: P2, b: P2) -> P2 {
    [a[0] - b[0], a[1] - b[1]]
}

fn dot(a: P2, b: P2) -> f64 {
    a[0] * b[0] + a[1] * b[1]
}

fn cross(a: P2, b: P2) -> f64 {
    a[0] * b[1] - a[1] * b[0]
}

fn dist(a: P2, b: P2) -> f64 {
    let d = sub(a, b);
    (d[0] * d[0] + d[1] * d[1]).sqrt()
}

fn midpoint(a: P2, b: P2) -> P2 {
    [0.5 * (a[0] + b[0]), 0.5 * (a[1] + b[1])]
}

/// A node of a [`StraightSkeletonGraph`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct StraightSkeletonNode {
    /// Position in the plane.
    pub position: [f64; 2],
    /// Offset distance at which the wavefront reaches the node: `0` for the
    /// polygon's own vertices, the height of the node on the roof model (and
    /// the radius of the inscribed circle of a convex polygon) otherwise.
    pub time: f64,
}

/// The straight skeleton of a polygon as a graph.
///
/// Every polygon vertex is a node of degree one (its bisector arc); the other
/// nodes are the events where wavefront vertices meet.
#[derive(Debug, Clone, PartialEq)]
pub struct StraightSkeletonGraph {
    /// The nodes of the skeleton.
    pub nodes: Vec<StraightSkeletonNode>,
    /// The arcs as pairs of indices into [`nodes`](Self::nodes).
    pub arcs: Vec<[usize; 2]>,
}

impl StraightSkeletonGraph {
    /// Number of arcs incident to each node.
    pub fn degrees(&self) -> Vec<usize> {
        let mut degrees = vec![0; self.nodes.len()];
        for arc in &self.arcs {
            degrees[arc[0]] += 1;
            degrees[arc[1]] += 1;
        }
        degrees
    }

    /// The arcs as pairs of end points.
    pub fn edges(&self) -> Vec<([f64; 2], [f64; 2])> {
        self.arcs
            .iter()
            .map(|a| (self.nodes[a[0]].position, self.nodes[a[1]].position))
            .collect()
    }
}

/// Straight skeleton for a simple 2-D polygon.
///
/// The polygon is given by its vertices in order (either orientation, the
/// closing edge implied); it must be simple (no self-intersections or
/// touching edges) and free of zero-width spikes. Consecutive duplicate
/// vertices are ignored, collinear vertices are kept: each contributes its own
/// bisector arc and its own vertex of the offset rings.
///
/// A polygon that does not satisfy this has no skeleton: [`skeleton_edges`]
/// and [`compute_offsets`] return an empty list for it, and
/// [`skeleton_graph`] and [`try_compute_offsets`] return the reason.
///
/// The wavefront simulation needs `O(n²)` time and memory in the worst case
/// for `n` vertices: about a tenth of a second for 1000 vertices, seconds for
/// several thousand.
///
/// [`skeleton_edges`]: StraightSkeleton::skeleton_edges
/// [`compute_offsets`]: StraightSkeleton::compute_offsets
/// [`skeleton_graph`]: StraightSkeleton::skeleton_graph
/// [`try_compute_offsets`]: StraightSkeleton::try_compute_offsets
#[derive(Debug, Clone)]
pub struct StraightSkeleton {
    polygon: Vec<[f64; 2]>,
}

impl StraightSkeleton {
    /// Construct a `StraightSkeleton` from a polygon given as an ordered vertex list.
    pub fn new(polygon: Vec<[f64; 2]>) -> Self {
        Self { polygon }
    }

    /// Compute the skeleton as a graph with the offset time of every node.
    ///
    /// # Errors
    ///
    /// * [`Error::TooFewPoints`] for fewer than three distinct vertices;
    /// * [`Error::DegenerateGeometry`] for non-finite coordinates, zero area,
    ///   a polygon that is not simple or has a zero-width spike;
    /// * [`Error::ConvergenceFailure`] if the wavefront simulation does not
    ///   terminate (only for polygons that defeat the tolerances).
    pub fn skeleton_graph(&self) -> Result<StraightSkeletonGraph> {
        skeleton_graph(&self.polygon)
    }

    /// Return the skeleton arcs as pairs of 2-D points.
    ///
    /// The arcs are the paths of the wavefront vertices (the corners of the
    /// polygon as its edges move inwards at unit speed); an arc starts at a
    /// polygon vertex or at the event where its wavefront vertex was born and
    /// ends at the event where it vanished. A 2×1 rectangle gives four corner
    /// arcs and the ridge between `(0.5, 0.5)` and `(1.5, 0.5)`. Empty for
    /// polygons without skeleton, see [`skeleton_graph`](Self::skeleton_graph).
    pub fn skeleton_edges(&self) -> Vec<([f64; 2], [f64; 2])> {
        self.skeleton_graph()
            .map(|graph| graph.edges())
            .unwrap_or_default()
    }

    /// Compute the mitred offset ring(s) of the polygon at signed distance
    /// `dist` (positive = inward, negative = outward), or report why that is
    /// impossible.
    ///
    /// The rings are the wavefront at time `|dist|`: edges move by `|dist|`,
    /// corners stay on the bisectors, edges that shrink to nothing disappear and
    /// the wavefront splits where a reflex vertex runs into an opposite edge.
    /// Inward rings keep the orientation of the polygon; an outward offset has
    /// one outer boundary (same orientation) and, where the wavefront has
    /// closed a pocket, holes (opposite orientation), in no particular order.
    /// When no event has happened the ring has one vertex per polygon vertex in
    /// the same order. `dist == 0` returns the polygon itself. The result is
    /// empty when the polygon has collapsed.
    ///
    /// # Errors
    ///
    /// As [`skeleton_graph`](Self::skeleton_graph), and
    /// [`Error::OutOfRange`] for a non-finite `dist`.
    pub fn try_compute_offsets(&self, dist: f64) -> Result<Vec<Vec<[f64; 2]>>> {
        offset_rings(&self.polygon, dist)
    }

    /// Compute offset polygon(s) at signed distance `dist` (positive = inward).
    ///
    /// Returns a list of offset rings; empty if the polygon collapses. See
    /// [`try_compute_offsets`](Self::try_compute_offsets) for the details and
    /// the error cases (reported here as an empty list).
    pub fn compute_offsets(&self, dist: f64) -> Vec<Vec<[f64; 2]>> {
        self.try_compute_offsets(dist).unwrap_or_default()
    }
}

// ─── Input validation ───────────────────────────────────────────────────────

/// A cleaned polygon.
struct Prepared {
    /// Vertices without consecutive duplicates, in the original orientation.
    ring: Vec<P2>,
    /// Whether `ring` is counter-clockwise.
    ccw: bool,
    /// Largest side of the bounding box.
    scale: f64,
    /// Diagonal of the bounding box.
    diagonal: f64,
}

fn signed_area(ring: &[P2]) -> f64 {
    let origin = ring[0];
    let mut twice = 0.0;
    for i in 0..ring.len() {
        let a = sub(ring[i], origin);
        let b = sub(ring[(i + 1) % ring.len()], origin);
        twice += cross(a, b);
    }
    0.5 * twice
}

fn point_segment_distance(p: P2, a: P2, b: P2) -> f64 {
    let ab = sub(b, a);
    let len_sq = dot(ab, ab);
    if len_sq == 0.0 {
        return dist(p, a);
    }
    let t = (dot(sub(p, a), ab) / len_sq).clamp(0.0, 1.0);
    dist(p, [a[0] + t * ab[0], a[1] + t * ab[1]])
}

/// Distance between the segments `a-b` and `c-d` (zero when they cross).
fn segment_distance(a: P2, b: P2, c: P2, d: P2) -> f64 {
    let d1 = cross(sub(b, a), sub(c, a));
    let d2 = cross(sub(b, a), sub(d, a));
    let d3 = cross(sub(d, c), sub(a, c));
    let d4 = cross(sub(d, c), sub(b, c));
    if d1 * d2 < 0.0 && d3 * d4 < 0.0 {
        return 0.0;
    }
    point_segment_distance(a, c, d)
        .min(point_segment_distance(b, c, d))
        .min(point_segment_distance(c, a, b))
        .min(point_segment_distance(d, a, b))
}

/// Reject polygons with crossing or touching non-adjacent edges.
fn check_simple(ring: &[P2], eps: f64) -> Result<()> {
    let n = ring.len();
    for i in 0..n {
        let (a, b) = (ring[i], ring[(i + 1) % n]);
        let (lo_x, hi_x) = (a[0].min(b[0]) - eps, a[0].max(b[0]) + eps);
        let (lo_y, hi_y) = (a[1].min(b[1]) - eps, a[1].max(b[1]) + eps);
        for j in (i + 1)..n {
            if j == i + 1 || (i == 0 && j == n - 1) {
                continue;
            }
            let (c, d) = (ring[j], ring[(j + 1) % n]);
            if c[0].min(d[0]) > hi_x
                || c[0].max(d[0]) < lo_x
                || c[1].min(d[1]) > hi_y
                || c[1].max(d[1]) < lo_y
            {
                continue;
            }
            if segment_distance(a, b, c, d) <= eps {
                return Err(Error::degenerate_geometry(format!(
                    "polygon is not simple: edges {i} and {j} touch or cross"
                )));
            }
        }
    }
    Ok(())
}

fn prepare(polygon: &[P2]) -> Result<Prepared> {
    if polygon.len() < 3 {
        return Err(Error::too_few_points(3, polygon.len()));
    }
    if polygon
        .iter()
        .any(|p| !p[0].is_finite() || !p[1].is_finite())
    {
        return Err(Error::degenerate_geometry(
            "polygon has a non-finite coordinate",
        ));
    }
    let mut lo = [f64::INFINITY; 2];
    let mut hi = [f64::NEG_INFINITY; 2];
    for p in polygon {
        for axis in 0..2 {
            lo[axis] = lo[axis].min(p[axis]);
            hi[axis] = hi[axis].max(p[axis]);
        }
    }
    let scale = (hi[0] - lo[0]).max(hi[1] - lo[1]);
    if scale <= 0.0 {
        return Err(Error::degenerate_geometry("polygon has no extent"));
    }
    let eps = POSITION_TOLERANCE * scale;

    let mut ring: Vec<P2> = Vec::with_capacity(polygon.len());
    for &p in polygon {
        if ring.last().is_none_or(|&q| dist(p, q) > eps) {
            ring.push(p);
        }
    }
    while ring.len() > 1 && dist(ring[0], ring[ring.len() - 1]) <= eps {
        ring.pop();
    }
    if ring.len() < 3 {
        return Err(Error::too_few_points(3, ring.len()));
    }

    let area = signed_area(&ring);
    if area.abs() <= 1e-12 * scale * scale {
        return Err(Error::degenerate_geometry("polygon has zero area"));
    }
    check_simple(&ring, eps)?;
    Ok(Prepared {
        ring,
        ccw: area > 0.0,
        scale,
        diagonal: (hi[0] - lo[0]).hypot(hi[1] - lo[1]),
    })
}

// ─── Wavefront simulation ───────────────────────────────────────────────────

/// The supporting line of a polygon edge. The wavefront of the edge at time
/// `t` is `{x : normal · (x - origin) = t}`.
#[derive(Debug, Clone, Copy)]
struct Line {
    /// Unit direction of the directed edge (the region lies on its left).
    dir: P2,
    /// Unit normal towards the region.
    normal: P2,
    /// A point of the edge.
    origin: P2,
}

/// A vertex of the wavefront: the intersection of two moving edge lines.
///
/// The record is immutable apart from the links and the `alive` flag; the
/// position at time `t` is `pos + (t - time) * vel`.
#[derive(Debug, Clone, Copy)]
struct Vertex {
    pos: P2,
    time: f64,
    vel: P2,
    line_in: usize,
    line_out: usize,
    prev: usize,
    next: usize,
    alive: bool,
    /// The two lines are antiparallel: the vertex would move infinitely fast
    /// and is resolved immediately after its creation.
    needle: bool,
    reflex: bool,
    /// Its split events have been queued.
    split_queued: bool,
    /// The skeleton node at which the vertex was born.
    node: usize,
}

#[derive(Debug, Clone, Copy)]
struct Seed {
    pos: P2,
    line_in: usize,
    line_out: usize,
    prev: usize,
    next: usize,
    node: usize,
}

#[derive(Debug, Clone, Copy)]
enum EventKind {
    /// The edge between `a` and `b` (`a.next == b`) shrinks to a point.
    Edge { a: usize, b: usize },
    /// The reflex vertex hits the wavefront of edge line `line`.
    Split { vertex: usize, line: usize },
}

#[derive(Debug, Clone, Copy)]
struct Event {
    time: f64,
    kind: EventKind,
}

impl Event {
    fn rank(&self) -> (u8, usize, usize) {
        match self.kind {
            EventKind::Edge { a, b } => (0, a, b),
            EventKind::Split { vertex, line } => (1, vertex, line),
        }
    }
}

impl PartialEq for Event {
    fn eq(&self, other: &Self) -> bool {
        self.cmp(other) == Ordering::Equal
    }
}

impl Eq for Event {}

impl PartialOrd for Event {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for Event {
    /// Reversed, so that the earliest event (edge events before split events
    /// at equal times, then by index for determinism) is on top of the
    /// max-heap.
    fn cmp(&self, other: &Self) -> Ordering {
        other
            .time
            .total_cmp(&self.time)
            .then_with(|| other.rank().cmp(&self.rank()))
    }
}

struct Wavefront {
    lines: Vec<Line>,
    verts: Vec<Vertex>,
    nodes: Vec<StraightSkeletonNode>,
    arcs: Vec<[usize; 2]>,
    heap: BinaryHeap<Event>,
    /// For every line the vertices whose outgoing edge lies on it: the start
    /// vertices of the line's wavefront pieces (stale entries are dead).
    pieces: Vec<Vec<usize>>,
    eps: f64,
    now: f64,
    /// Events later than this are not needed and never queued.
    horizon: f64,
    /// Bounding box of the polygon when the wavefront moves inwards: it never
    /// leaves it, so events outside are impossible.
    bounds: Option<[P2; 2]>,
    /// Nodes created at (almost) the current time, for snapping.
    recent: Vec<usize>,
    steps: usize,
    max_steps: usize,
}

impl Wavefront {
    /// Start the simulation for `ring`, a simple polygon whose region lies on
    /// the left of its directed edges.
    fn new(ring: &[P2], scale: f64, horizon: f64, inward: bool) -> Result<Self> {
        let n = ring.len();
        let eps = POSITION_TOLERANCE * scale;
        let bounds = inward.then(|| {
            let mut lo = [f64::INFINITY; 2];
            let mut hi = [f64::NEG_INFINITY; 2];
            for p in ring {
                for axis in 0..2 {
                    lo[axis] = lo[axis].min(p[axis] - eps);
                    hi[axis] = hi[axis].max(p[axis] + eps);
                }
            }
            [lo, hi]
        });
        let mut lines = Vec::with_capacity(n);
        for i in 0..n {
            let a = ring[i];
            let e = sub(ring[(i + 1) % n], a);
            let len = (e[0] * e[0] + e[1] * e[1]).sqrt();
            let dir = [e[0] / len, e[1] / len];
            lines.push(Line {
                dir,
                normal: [-dir[1], dir[0]],
                origin: a,
            });
        }
        let mut wf = Self {
            lines,
            verts: Vec::with_capacity(4 * n),
            nodes: Vec::with_capacity(2 * n),
            arcs: Vec::with_capacity(3 * n),
            heap: BinaryHeap::new(),
            pieces: vec![Vec::new(); n],
            eps,
            now: 0.0,
            horizon,
            bounds,
            recent: Vec::new(),
            steps: 0,
            max_steps: 256 * (n + 8),
        };
        for (i, &pos) in ring.iter().enumerate() {
            let node = wf.push_node(pos);
            let id = wf.make_vertex(Seed {
                pos,
                line_in: (i + n - 1) % n,
                line_out: i,
                prev: (i + n - 1) % n,
                next: (i + 1) % n,
                node,
            });
            if wf.verts[id].needle {
                return Err(Error::degenerate_geometry(format!(
                    "polygon has a zero-width spike at vertex {i}"
                )));
            }
        }
        wf.normalize((0..n).collect())?;
        Ok(wf)
    }

    fn pos_at(&self, v: usize, t: f64) -> P2 {
        let w = &self.verts[v];
        let dt = t - w.time;
        [w.pos[0] + dt * w.vel[0], w.pos[1] + dt * w.vel[1]]
    }

    fn push_node(&mut self, position: P2) -> usize {
        let id = self.nodes.len();
        self.nodes.push(StraightSkeletonNode {
            position,
            time: self.now,
        });
        id
    }

    /// The skeleton node at `pos` at the current time, shared between all
    /// events at (nearly) the same place.
    fn node_at(&mut self, pos: P2) -> usize {
        let (eps, now) = (self.eps, self.now);
        let nodes = &self.nodes;
        self.recent.retain(|&i| nodes[i].time >= now - eps);
        for &i in &self.recent {
            if dist(self.nodes[i].position, pos) <= eps {
                return i;
            }
        }
        let id = self.push_node(pos);
        self.recent.push(id);
        id
    }

    fn add_arc(&mut self, a: usize, b: usize) {
        if a != b && dist(self.nodes[a].position, self.nodes[b].position) > self.eps {
            self.arcs.push([a, b]);
        }
    }

    /// Kill `v` at skeleton node `death`, emitting its arc.
    fn retire(&mut self, v: usize, death: usize) {
        self.verts[v].alive = false;
        let born = self.verts[v].node;
        self.add_arc(born, death);
    }

    fn make_vertex(&mut self, seed: Seed) -> usize {
        let (d1, d2) = (self.lines[seed.line_in].dir, self.lines[seed.line_out].dir);
        let turn = cross(d1, d2);
        let needle = turn.abs() <= NEEDLE_SINE && dot(d1, d2) < 0.0;
        let vel = if needle {
            [0.0, 0.0]
        } else {
            // (n1 + n2) / (1 + n1 . n2) with 1 + n1 . n2 = |n1 + n2|^2 / 2:
            // the bisector scaled by 1 / cos(angle / 2), without cancelling
            // in the denominator for nearly antiparallel edges.
            let (n1, n2) = (
                self.lines[seed.line_in].normal,
                self.lines[seed.line_out].normal,
            );
            let sum = [n1[0] + n2[0], n1[1] + n2[1]];
            let sum_sq = dot(sum, sum);
            [2.0 * sum[0] / sum_sq, 2.0 * sum[1] / sum_sq]
        };
        let reflex = turn < -REFLEX_EPS;
        let id = self.verts.len();
        self.verts.push(Vertex {
            pos: seed.pos,
            time: self.now,
            vel,
            line_in: seed.line_in,
            line_out: seed.line_out,
            prev: seed.prev,
            next: seed.next,
            alive: true,
            needle,
            reflex,
            split_queued: false,
            node: seed.node,
        });
        self.pieces[seed.line_out].push(id);
        id
    }

    /// Make the neighbours of `id` point to it.
    fn splice(&mut self, id: usize) {
        let (prev, next) = (self.verts[id].prev, self.verts[id].next);
        self.verts[prev].next = id;
        self.verts[next].prev = id;
    }

    fn collect_cycle(&self, start: usize) -> Vec<usize> {
        let mut ring = vec![start];
        let mut c = self.verts[start].next;
        while c != start && ring.len() <= self.verts.len() {
            ring.push(c);
            c = self.verts[c].next;
        }
        ring
    }

    fn same_cycle(&self, a: usize, b: usize) -> bool {
        let mut c = self.verts[a].next;
        let mut guard = 0;
        while c != a && guard <= self.verts.len() {
            if c == b {
                return true;
            }
            c = self.verts[c].next;
            guard += 1;
        }
        false
    }

    fn bump(&mut self) -> Result<()> {
        self.steps += 1;
        if self.steps > self.max_steps {
            return Err(Error::convergence_failure(
                "straight skeleton",
                self.steps,
                0.0,
            ));
        }
        Ok(())
    }

    // ── Event scheduling ────────────────────────────────────────────────────

    /// Time at which the edge between `a` and `b` (`a.next == b`) collapses.
    fn edge_event_time(&self, a: usize, b: usize) -> Option<f64> {
        let (va, vb) = (&self.verts[a], &self.verts[b]);
        if va.needle || vb.needle {
            return None;
        }
        let dir = self.lines[va.line_out].dir;
        let t_ref = va.time.max(vb.time);
        let gap = dot(sub(self.pos_at(b, t_ref), self.pos_at(a, t_ref)), dir);
        let rate = dot(sub(vb.vel, va.vel), dir);
        if gap <= self.eps {
            return (rate <= RATE_EPS).then_some(t_ref);
        }
        (rate < -RATE_EPS).then(|| t_ref + gap / -rate)
    }

    fn push_edge_event(&mut self, a: usize, b: usize) {
        if let Some(time) = self.edge_event_time(a, b)
            && time <= self.horizon
        {
            self.heap.push(Event {
                time,
                kind: EventKind::Edge { a, b },
            });
        }
    }

    /// Queue the times at which reflex vertex `v` reaches the wavefront of each
    /// edge line; whether the hit lies on an actual piece of that wavefront is
    /// decided when the event is popped.
    fn push_split_events(&mut self, v: usize) {
        let w = self.verts[v];
        for (e, line) in self.lines.iter().enumerate() {
            if e == w.line_in || e == w.line_out {
                continue;
            }
            let closing = 1.0 - dot(line.normal, w.vel);
            if closing <= RATE_EPS {
                continue;
            }
            let ahead = dot(line.normal, sub(w.pos, line.origin)) - w.time;
            if ahead < -self.eps {
                continue;
            }
            let time = w.time + ahead.max(0.0) / closing;
            if time > self.horizon {
                continue;
            }
            if let Some([lo, hi]) = self.bounds {
                let dt = time - w.time;
                let hit = [w.pos[0] + dt * w.vel[0], w.pos[1] + dt * w.vel[1]];
                if (0..2).any(|axis| hit[axis] < lo[axis] || hit[axis] > hi[axis]) {
                    continue;
                }
            }
            self.heap.push(Event {
                time,
                kind: EventKind::Split { vertex: v, line: e },
            });
        }
    }

    fn schedule(&mut self, x: usize) {
        let (prev, next) = (self.verts[x].prev, self.verts[x].next);
        self.push_edge_event(prev, x);
        self.push_edge_event(x, next);
        if self.verts[x].reflex && !self.verts[x].split_queued {
            self.verts[x].split_queued = true;
            self.push_split_events(x);
        }
    }

    // ── Local clean-up ──────────────────────────────────────────────────────

    /// Bring the wavefront around the vertices in `seeds` (new, at the current
    /// time) into a canonical state and queue their events.
    fn normalize(&mut self, seeds: Vec<usize>) -> Result<()> {
        let mut work = seeds;
        while let Some(x) = work.pop() {
            if !self.verts[x].alive {
                continue;
            }
            self.bump()?;
            let (prev, next) = (self.verts[x].prev, self.verts[x].next);
            // A one- or two-vertex wavefront has no area left.
            if next == x || self.verts[next].next == x {
                self.collapse_cycle(x);
                continue;
            }
            if self.verts[x].needle {
                self.resolve_needle(x, &mut work);
                continue;
            }
            // The velocity of an unresolved needle neighbour is unknown.
            if self.verts[prev].needle {
                work.push(x);
                work.push(prev);
                continue;
            }
            if self.verts[next].needle {
                work.push(x);
                work.push(next);
                continue;
            }
            if let Some(merged) = self.contract_coincident(x) {
                work.push(merged);
                continue;
            }
            if self.cycle_is_flat(x) {
                self.collapse_cycle(x);
                continue;
            }
            self.schedule(x);
        }
        Ok(())
    }

    /// Replace the adjacent vertices `a` and `b` (`a.next == b`), which meet at
    /// the current time, by one vertex between `a`'s incoming and `b`'s
    /// outgoing edge.
    fn merge_pair(&mut self, a: usize, b: usize) -> usize {
        let at = midpoint(self.pos_at(a, self.now), self.pos_at(b, self.now));
        let node = self.node_at(at);
        let (prev, next) = (self.verts[a].prev, self.verts[b].next);
        let (line_in, line_out) = (self.verts[a].line_in, self.verts[b].line_out);
        self.retire(a, node);
        self.retire(b, node);
        let merged = self.make_vertex(Seed {
            pos: at,
            line_in,
            line_out,
            prev,
            next,
            node,
        });
        self.splice(merged);
        merged
    }

    /// Merge `x` with a neighbour at the same place unless the edge between
    /// them is growing (as it does after a vertex event).
    fn contract_coincident(&mut self, x: usize) -> Option<usize> {
        let (prev, next) = (self.verts[x].prev, self.verts[x].next);
        for (a, b) in [(prev, x), (x, next)] {
            let (pa, pb) = (self.pos_at(a, self.now), self.pos_at(b, self.now));
            if dist(pa, pb) > self.eps {
                continue;
            }
            let dir = self.lines[self.verts[a].line_out].dir;
            let rate = dot(sub(self.verts[b].vel, self.verts[a].vel), dir);
            if rate <= RATE_EPS {
                return Some(self.merge_pair(a, b));
            }
        }
        None
    }

    /// Resolve the vertex `x` between two antiparallel edges.
    ///
    /// At the current time the two edges lie on one line: `x` is the tip of a
    /// zero-width strip, its neighbours `p` and `n` lie on that line on the
    /// same side of `x`. The strip collapses from the tip down to the nearer
    /// neighbour (emitting the ridge as an arc); if both are equally far all
    /// three meet.
    fn resolve_needle(&mut self, x: usize, work: &mut Vec<usize>) {
        let (p, n) = (self.verts[x].prev, self.verts[x].next);
        let (at_x, at_p, at_n) = (
            self.pos_at(x, self.now),
            self.pos_at(p, self.now),
            self.pos_at(n, self.now),
        );
        let (to_p, to_n) = (dist(at_x, at_p), dist(at_x, at_n));
        if (to_p - to_n).abs() <= self.eps {
            let at = midpoint(at_p, at_n);
            let node = self.node_at(at);
            let (before, after) = (self.verts[p].prev, self.verts[n].next);
            let (line_in, line_out) = (self.verts[p].line_in, self.verts[n].line_out);
            self.retire(p, node);
            self.retire(x, node);
            self.retire(n, node);
            if before != n {
                let merged = self.make_vertex(Seed {
                    pos: at,
                    line_in,
                    line_out,
                    prev: before,
                    next: after,
                    node,
                });
                self.splice(merged);
                work.push(merged);
            }
        } else if to_p < to_n {
            let node = self.node_at(at_p);
            let before = self.verts[p].prev;
            let (line_in, line_out) = (self.verts[p].line_in, self.verts[x].line_out);
            self.retire(x, node);
            self.retire(p, node);
            let kept = self.make_vertex(Seed {
                pos: at_p,
                line_in,
                line_out,
                prev: before,
                next: n,
                node,
            });
            self.splice(kept);
            work.push(kept);
        } else {
            let node = self.node_at(at_n);
            let after = self.verts[n].next;
            let (line_in, line_out) = (self.verts[x].line_in, self.verts[n].line_out);
            self.retire(x, node);
            self.retire(n, node);
            let kept = self.make_vertex(Seed {
                pos: at_n,
                line_in,
                line_out,
                prev: p,
                next: after,
                node,
            });
            self.splice(kept);
            work.push(kept);
        }
    }

    /// Whether all vertices of the wavefront polygon through `x` lie on one
    /// line (so that it has zero area).
    fn cycle_is_flat(&self, x: usize) -> bool {
        let points: Vec<P2> = self
            .collect_cycle(x)
            .iter()
            .map(|&v| self.pos_at(v, self.now))
            .collect();
        let first = farthest_from(&points, points[0]);
        let second = farthest_from(&points, first);
        let span = dist(first, second);
        if span <= self.eps {
            return true;
        }
        let u = [(second[0] - first[0]) / span, (second[1] - first[1]) / span];
        points
            .iter()
            .all(|&p| cross(u, sub(p, first)).abs() <= self.eps)
    }

    /// Retire the whole wavefront polygon through `x`, which has no area left:
    /// its vertices die where they are and the segment they span becomes a
    /// chain of ridge arcs.
    fn collapse_cycle(&mut self, x: usize) {
        let ring = self.collect_cycle(x);
        let points: Vec<P2> = ring.iter().map(|&v| self.pos_at(v, self.now)).collect();
        let nodes: Vec<usize> = points.iter().map(|&p| self.node_at(p)).collect();
        for (&v, &node) in ring.iter().zip(nodes.iter()) {
            self.retire(v, node);
        }
        let first = farthest_from(&points, points[0]);
        let second = farthest_from(&points, first);
        let span = dist(first, second);
        if span > self.eps {
            let u = [(second[0] - first[0]) / span, (second[1] - first[1]) / span];
            let mut order: Vec<usize> = (0..points.len()).collect();
            order.sort_by(|&i, &j| dot(points[i], u).total_cmp(&dot(points[j], u)));
            for pair in order.windows(2) {
                self.add_arc(nodes[pair[0]], nodes[pair[1]]);
            }
        }
    }

    // ── Event processing ────────────────────────────────────────────────────

    fn handle_edge(&mut self, a: usize, b: usize, time: f64) -> Result<()> {
        self.now = self.now.max(time);
        let merged = self.merge_pair(a, b);
        self.normalize(vec![merged])
    }

    /// Find the piece `(p, q)` of the wavefront of `line` that contains the
    /// point where `v` reaches the line at `time`.
    fn locate_split(&self, v: usize, line: usize, time: f64) -> Option<(usize, usize, P2)> {
        let hit = self.pos_at(v, time);
        let dir = self.lines[line].dir;
        for &p in &self.pieces[line] {
            if p == v || !self.verts[p].alive {
                continue;
            }
            let q = self.verts[p].next;
            if q == v {
                continue;
            }
            let from_start = dot(sub(hit, self.pos_at(p, time)), dir);
            let to_end = dot(sub(self.pos_at(q, time), hit), dir);
            if from_start < -self.eps || to_end < -self.eps {
                continue;
            }
            if self.same_cycle(v, p) {
                return Some((p, q, hit));
            }
        }
        None
    }

    /// Split the wavefront at reflex vertex `v`, which reaches the piece
    /// `(p, q)` of the wavefront of `line` at `hit`.
    fn handle_split(
        &mut self,
        v: usize,
        line: usize,
        piece: (usize, usize),
        hit: P2,
        time: f64,
    ) -> Result<()> {
        let (p, q) = piece;
        self.now = self.now.max(time);
        let node = self.node_at(hit);
        let w = self.verts[v];
        self.retire(v, node);
        let first = self.make_vertex(Seed {
            pos: hit,
            line_in: w.line_in,
            line_out: line,
            prev: w.prev,
            next: q,
            node,
        });
        let second = self.make_vertex(Seed {
            pos: hit,
            line_in: line,
            line_out: w.line_out,
            prev: p,
            next: w.next,
            node,
        });
        self.verts[w.prev].next = first;
        self.verts[q].prev = first;
        self.verts[p].next = second;
        self.verts[w.next].prev = second;
        self.normalize(vec![first, second])
    }

    /// Process events until the queue is empty.
    fn run(&mut self) -> Result<()> {
        while let Some(event) = self.heap.pop() {
            match event.kind {
                EventKind::Edge { a, b } => {
                    if !(self.verts[a].alive && self.verts[b].alive && self.verts[a].next == b) {
                        continue;
                    }
                    self.bump()?;
                    self.handle_edge(a, b, event.time)?;
                }
                EventKind::Split { vertex, line } => {
                    if !self.verts[vertex].alive {
                        continue;
                    }
                    let time = event.time.max(self.now);
                    let Some((p, q, hit)) = self.locate_split(vertex, line, time) else {
                        continue;
                    };
                    self.bump()?;
                    self.handle_split(vertex, line, (p, q), hit, time)?;
                }
            }
        }
        Ok(())
    }

    // ── Results ─────────────────────────────────────────────────────────────

    /// The wavefront polygons at time `t`.
    fn rings_at(&self, t: f64) -> Vec<Vec<P2>> {
        let mut seen = vec![false; self.verts.len()];
        let mut rings = Vec::new();
        for start in 0..self.verts.len() {
            if !self.verts[start].alive || seen[start] {
                continue;
            }
            let mut ring: Vec<P2> = Vec::new();
            let mut c = start;
            loop {
                seen[c] = true;
                let p = self.pos_at(c, t);
                if ring.last().is_none_or(|&q| dist(p, q) > self.eps) {
                    ring.push(p);
                }
                c = self.verts[c].next;
                if c == start || seen[c] {
                    break;
                }
            }
            while ring.len() > 1 && dist(ring[0], ring[ring.len() - 1]) <= self.eps {
                ring.pop();
            }
            if ring.len() >= 3 && signed_area(&ring).abs() > self.eps * self.eps {
                rings.push(ring);
            }
        }
        rings
    }

    fn into_graph(self) -> Result<StraightSkeletonGraph> {
        if self.verts.iter().any(|v| v.alive) {
            return Err(Error::convergence_failure(
                "straight skeleton",
                self.steps,
                0.0,
            ));
        }
        let mut seen: HashSet<(usize, usize)> = HashSet::new();
        let mut arcs: Vec<[usize; 2]> = Vec::with_capacity(self.arcs.len());
        let mut used = vec![false; self.nodes.len()];
        for &[a, b] in &self.arcs {
            if seen.insert((a.min(b), a.max(b))) {
                arcs.push([a, b]);
                used[a] = true;
                used[b] = true;
            }
        }
        let mut remap = vec![usize::MAX; self.nodes.len()];
        let mut nodes = Vec::new();
        for (i, node) in self.nodes.iter().enumerate() {
            if used[i] {
                remap[i] = nodes.len();
                nodes.push(*node);
            }
        }
        for arc in &mut arcs {
            *arc = [remap[arc[0]], remap[arc[1]]];
        }
        Ok(StraightSkeletonGraph { nodes, arcs })
    }
}

/// The point of `points` farthest from `from`.
fn farthest_from(points: &[P2], from: P2) -> P2 {
    let mut best = from;
    let mut best_dist = 0.0;
    for &p in points {
        let d = dist(p, from);
        if d > best_dist {
            best = p;
            best_dist = d;
        }
    }
    best
}

fn skeleton_graph(polygon: &[P2]) -> Result<StraightSkeletonGraph> {
    let prepared = prepare(polygon)?;
    let ring: Vec<P2> = if prepared.ccw {
        prepared.ring
    } else {
        prepared.ring.into_iter().rev().collect()
    };
    // The wavefront has vanished long before it has travelled the diagonal.
    let mut wavefront = Wavefront::new(&ring, prepared.scale, prepared.diagonal, true)?;
    wavefront.run()?;
    wavefront.into_graph()
}

fn offset_rings(polygon: &[P2], distance: f64) -> Result<Vec<Vec<P2>>> {
    if !distance.is_finite() {
        return Err(Error::out_of_range("dist", distance, -f64::MAX, f64::MAX));
    }
    let prepared = prepare(polygon)?;
    if distance == 0.0 {
        return Ok(vec![prepared.ring]);
    }
    // The wavefront must move to the left of the directed edges: the region
    // is inside a counter-clockwise polygon, outside a clockwise one.
    let interior = distance > 0.0;
    let reversed = interior != prepared.ccw;
    let ring: Vec<P2> = if reversed {
        prepared.ring.into_iter().rev().collect()
    } else {
        prepared.ring
    };
    let travel = distance.abs();
    let eps = POSITION_TOLERANCE * prepared.scale;
    let mut wavefront = Wavefront::new(&ring, prepared.scale, travel + eps, interior)?;
    wavefront.run()?;
    let mut rings = wavefront.rings_at(travel);
    if reversed {
        for ring in &mut rings {
            ring.reverse();
        }
    }
    Ok(rings)
}
