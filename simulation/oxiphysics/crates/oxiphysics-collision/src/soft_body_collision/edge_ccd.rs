// Copyright 2026 COOLJAPAN OU (Team KitaSan)
// SPDX-License-Identifier: Apache-2.0

//! Exact edge-edge continuous collision detection for cloth and soft bodies.
//!
//! Every end point of an edge moves linearly over the step, so the four end
//! points of an edge pair are coplanar at the roots of a cubic in time (Provot
//! 1997, Bridson et al. 2002). A pair is examined in this order:
//!
//! 1. If the edges are already within the contact radius at the start of the
//!    step, the time of impact is zero.
//! 2. Otherwise the roots of the coplanarity cubic are located exactly (the
//!    cubic is split at the zeros of its derivative into monotone pieces and
//!    each piece is bisected), and the earliest root at which the segments are
//!    within the contact radius is the time of impact. For edges that cross this
//!    is the time at which they cross.
//! 3. If no root qualifies, the edges can still come within the contact radius
//!    without ever being coplanar (an end point grazing past an end point, or
//!    planar motion, for which the cubic vanishes identically). Then the first
//!    time the distance between the segments reaches the contact radius is
//!    found by interval subdivision with a Lipschitz bound: the distance between
//!    the segments changes no faster than the largest relative displacement of
//!    two end points, so an interval whose end distances exceed the radius by
//!    more than that bound allows for cannot hold a contact and is discarded.
//!    Unlike uniform time sampling this cannot step over a brief dip of the
//!    distance. The work is bounded: a pair that hovers just above the contact
//!    radius for the whole step, so that no gap can be certified within the
//!    budget, is reported as touching rather than risk a missed contact.
//!
//! Everything is evaluated in a local frame whose origin is a vertex of the pair
//! and whose unit of length is the longest edge vector, so the result does not
//! depend on the units or the position of the mesh.

use super::{EdgeCcdResult, add3, cross3, dot3, len3, normalize3, scale3, sub3};

/// Relative floor of the contact radius.
///
/// Two edges that cross exactly are at distance zero only in exact arithmetic:
/// evaluated at the rounded root time of the coplanarity cubic the distance
/// comes out around `1e-16` times the extent of the configuration, so a contact
/// radius of exactly zero would reject the root of every generic crossing. The
/// radius of a pair is therefore never taken smaller than this fraction of the
/// larger of its longest edge vector and the distance of its farthest end point
/// from the first vertex of edge A (the extent of the pair, which does not
/// depend on where the pair is in space). The floor is orders of magnitude below
/// any physical contact radius, which it leaves unchanged.
const RADIUS_FLOOR: f64 = 1e-10;

/// Size, relative to the product of the lengths that enter the coplanarity
/// cubic, below which its coefficients are rounding noise. A cubic that small
/// vanishes identically: the four end points stay coplanar for the whole step.
const COPLANAR_NOISE: f64 = 1e-12;

/// Distance, in units of the longest edge vector, to which the first entry of
/// the distance into the contact shell is located (see [`shell_entry`]).
const SHELL_TOLERANCE: f64 = 1e-12;

/// Largest number of distance evaluations of one [`shell_entry`] search.
///
/// A pair that stays within `L / (2 * MAX_SHELL_EVALUATIONS)` of the contact
/// radius for the whole step (`L` being the Lipschitz bound of the distance)
/// cannot be certified as separated by interval subdivision within this budget;
/// the search then reports contact rather than risk letting the edges tunnel.
const MAX_SHELL_EVALUATIONS: usize = 4096;

/// An edge whose two end points move linearly over one step.
#[derive(Clone, Copy, Debug)]
pub(super) struct MovingEdge {
    /// End point positions at the start of the step.
    pub start: [[f64; 3]; 2],
    /// End point positions at the end of the step.
    pub end: [[f64; 3]; 2],
}

impl MovingEdge {
    /// Build the edge `(i0, i1)` of a mesh; `None` if an index is outside
    /// `verts0` or `verts1`.
    pub(super) fn from_mesh(
        i0: usize,
        i1: usize,
        verts0: &[[f64; 3]],
        verts1: &[[f64; 3]],
    ) -> Option<Self> {
        Some(MovingEdge {
            start: [*verts0.get(i0)?, *verts0.get(i1)?],
            end: [*verts1.get(i0)?, *verts1.get(i1)?],
        })
    }

    /// End points at normalised step time `t`.
    fn at(&self, t: f64) -> [[f64; 3]; 2] {
        let lerp = |a: [f64; 3], b: [f64; 3]| add3(scale3(a, 1.0 - t), scale3(b, t));
        [
            lerp(self.start[0], self.end[0]),
            lerp(self.start[1], self.end[1]),
        ]
    }

    /// Displacement of each end point over the step.
    fn displacement(&self) -> [[f64; 3]; 2] {
        [
            sub3(self.end[0], self.start[0]),
            sub3(self.end[1], self.start[1]),
        ]
    }

    /// Axis-aligned box enclosing the whole trajectory of the edge.
    ///
    /// Each end point moves on a straight segment, so the box of the four
    /// positions at `t = 0` and `t = 1` contains the edge at every time.
    fn swept_bounds(&self) -> ([f64; 3], [f64; 3]) {
        let mut lo = self.start[0];
        let mut hi = self.start[0];
        for p in self.start.iter().chain(self.end.iter()) {
            for ((l, h), &c) in lo.iter_mut().zip(hi.iter_mut()).zip(p.iter()) {
                *l = l.min(c);
                *h = h.max(c);
            }
        }
        (lo, hi)
    }

    /// Largest distance of an end point position from `origin`.
    fn extent_from(&self, origin: [f64; 3]) -> f64 {
        self.start
            .iter()
            .chain(self.end.iter())
            .fold(0.0_f64, |largest, &p| largest.max(len3(sub3(p, origin))))
    }

    /// Whether every coordinate is finite.
    fn is_finite(&self) -> bool {
        self.start
            .iter()
            .chain(self.end.iter())
            .flatten()
            .all(|c| c.is_finite())
    }

    /// Length of the longer of the two edge vectors (at the start and at the
    /// end of the step).
    fn longest_vector(&self) -> f64 {
        len3(sub3(self.start[1], self.start[0])).max(len3(sub3(self.end[1], self.end[0])))
    }

    /// The edge with every position mapped to `(position - origin) * inv_unit`.
    fn rescaled(&self, origin: [f64; 3], inv_unit: f64) -> MovingEdge {
        let map = |p: [f64; 3]| scale3(sub3(p, origin), inv_unit);
        MovingEdge {
            start: [map(self.start[0]), map(self.start[1])],
            end: [map(self.end[0]), map(self.end[1])],
        }
    }
}

/// Closest pair of points of two segments.
#[derive(Clone, Copy, Debug)]
pub(super) struct SegmentPair {
    /// Distance between the segments.
    pub distance: f64,
    /// Parameter (0 to 1) of the closest point on the first segment.
    pub s: f64,
    /// Parameter (0 to 1) of the closest point on the second segment.
    pub u: f64,
    /// Closest point on the first segment.
    pub closest_a: [f64; 3],
    /// Closest point on the second segment.
    pub closest_b: [f64; 3],
}

/// Closest points of the segments `a0..a1` and `b0..b1` (Ericson, *Real-Time
/// Collision Detection*, 5.1.9).
///
/// Degenerate segments (a single point) and parallel segments are handled
/// without any absolute tolerance, so the result does not depend on the scale of
/// the input: when the parameter on the second segment has to be clamped to one
/// of its end points, the parameter on the first segment is recomputed against
/// that end point, which makes the pair a true minimum-distance pair.
pub(super) fn segment_pair(a0: [f64; 3], a1: [f64; 3], b0: [f64; 3], b1: [f64; 3]) -> SegmentPair {
    let d1 = sub3(a1, a0);
    let d2 = sub3(b1, b0);
    let r = sub3(a0, b0);
    let a = dot3(d1, d1);
    let e = dot3(d2, d2);
    let f = dot3(d2, r);
    let (s, u) = if a == 0.0 && e == 0.0 {
        (0.0, 0.0)
    } else if a == 0.0 {
        (0.0, (f / e).clamp(0.0, 1.0))
    } else {
        let c = dot3(d1, r);
        if e == 0.0 {
            ((-c / a).clamp(0.0, 1.0), 0.0)
        } else {
            let b = dot3(d1, d2);
            let denom = a * e - b * b;
            // Point of the first segment nearest to the line through the second
            // one; arbitrary (the start) if the segments are parallel, where
            // every choice is equally near.
            let s_line = if denom > 1e-14 * a * e {
                ((b * f - c * e) / denom).clamp(0.0, 1.0)
            } else {
                0.0
            };
            let u_line = (b * s_line + f) / e;
            if u_line < 0.0 {
                ((-c / a).clamp(0.0, 1.0), 0.0)
            } else if u_line > 1.0 {
                (((b - c) / a).clamp(0.0, 1.0), 1.0)
            } else {
                (s_line, u_line)
            }
        }
    };
    let closest_a = add3(a0, scale3(d1, s));
    let closest_b = add3(b0, scale3(d2, u));
    SegmentPair {
        distance: len3(sub3(closest_a, closest_b)),
        s,
        u,
        closest_a,
        closest_b,
    }
}

/// The segments of the edges `a` and `b` at normalised time `t`.
fn pair_at(a: &MovingEdge, b: &MovingEdge, t: f64) -> SegmentPair {
    let [a0, a1] = a.at(t);
    let [b0, b1] = b.at(t);
    segment_pair(a0, a1, b0, b1)
}

/// Coefficients `(c3, c2, c1, c0)` of the coplanarity cubic of an edge pair.
///
/// The four end points are coplanar when the scalar triple product of the two
/// edge vectors and the vector between their start points vanishes; each of the
/// three vectors is affine in time, so the triple product is a cubic polynomial.
/// The second value is the scale (a bound on the magnitude of the coefficients)
/// used to decide whether the cubic vanishes identically.
fn coplanarity_cubic(a: &MovingEdge, b: &MovingEdge) -> ([f64; 4], f64) {
    let edge_a = sub3(a.start[1], a.start[0]);
    let edge_b = sub3(b.start[1], b.start[0]);
    let link = sub3(b.start[0], a.start[0]);
    let [da0, da1] = a.displacement();
    let [db0, db1] = b.displacement();
    let d_edge_a = sub3(da1, da0);
    let d_edge_b = sub3(db1, db0);
    let d_link = sub3(db0, da0);

    let (r1, r2, r3) = (edge_a, edge_b, link);
    let (w1, w2, w3) = (d_edge_a, d_edge_b, d_link);
    let c3 = dot3(w1, cross3(w2, w3));
    let c2 = dot3(r1, cross3(w2, w3)) + dot3(w1, cross3(r2, w3)) + dot3(w1, cross3(w2, r3));
    let c1 = dot3(r1, cross3(r2, w3)) + dot3(r1, cross3(w2, r3)) + dot3(w1, cross3(r2, r3));
    let c0 = dot3(r1, cross3(r2, r3));
    let scale = (len3(r1) + len3(w1)) * (len3(r2) + len3(w2)) * (len3(r3) + len3(w3));
    ([c3, c2, c1, c0], scale)
}

/// Real roots in `[0, 1]` of the cubic `c3 t^3 + c2 t^2 + c1 t + c0`, ascending.
///
/// The zeros of the derivative split the interval into at most three pieces on
/// which the cubic is monotone; a piece holds a root exactly when the cubic
/// changes sign across it, and the root is then bisected to full precision. A
/// double root, where the cubic touches zero at a zero of the derivative
/// without changing sign, is reported when the cubic is below rounding noise
/// there. Spurious candidates are harmless: the caller validates every root
/// geometrically.
pub(super) fn unit_interval_roots(coefficients: [f64; 4]) -> Vec<f64> {
    let [c3, c2, c1, c0] = coefficients;
    let cubic_at = |t: f64| ((c3 * t + c2) * t + c1) * t + c0;
    let magnitude = c3.abs().max(c2.abs()).max(c1.abs()).max(c0.abs());
    let noise = 1e-12 * magnitude;

    // Zeros of the derivative 3 c3 t^2 + 2 c2 t + c1 inside (0, 1).
    let (qa, qb, qc) = (3.0 * c3, 2.0 * c2, c1);
    let mut critical: Vec<f64> = Vec::with_capacity(2);
    if qa == 0.0 {
        if qb != 0.0 {
            critical.push(-qc / qb);
        }
    } else {
        let discriminant = qb * qb - 4.0 * qa * qc;
        if discriminant >= 0.0 {
            let root = discriminant.sqrt();
            let q = -0.5 * (qb + if qb < 0.0 { -root } else { root });
            critical.push(q / qa);
            if q != 0.0 {
                critical.push(qc / q);
            }
        }
    }
    critical.retain(|t| *t > 0.0 && *t < 1.0);
    critical.sort_by(f64::total_cmp);

    let mut breaks = Vec::with_capacity(4);
    breaks.push(0.0);
    breaks.extend(critical.iter().copied());
    breaks.push(1.0);

    let mut roots: Vec<f64> = Vec::with_capacity(3);
    for piece in breaks.windows(2) {
        let (lo, hi) = (piece[0], piece[1]);
        let (f_lo, f_hi) = (cubic_at(lo), cubic_at(hi));
        if f_lo == 0.0 {
            roots.push(lo);
        } else if f_hi != 0.0 && (f_lo < 0.0) != (f_hi < 0.0) {
            roots.push(bisect_sign_change(&cubic_at, lo, hi));
        }
    }
    if cubic_at(1.0) == 0.0 {
        roots.push(1.0);
    }
    for &t in &critical {
        if cubic_at(t).abs() <= noise {
            roots.push(t);
        }
    }
    roots.sort_by(f64::total_cmp);
    roots.dedup();
    roots
}

/// The root of `f` in `[lo, hi]`, given that `f(lo)` and `f(hi)` have opposite
/// signs, by bisection down to adjacent floating-point numbers.
fn bisect_sign_change(f: &impl Fn(f64) -> f64, mut lo: f64, mut hi: f64) -> f64 {
    let lo_negative = f(lo) < 0.0;
    for _ in 0..200 {
        let mid = 0.5 * (lo + hi);
        if mid <= lo || mid >= hi {
            break;
        }
        let value = f(mid);
        if value == 0.0 {
            return mid;
        }
        if (value < 0.0) == lo_negative {
            lo = mid;
        } else {
            hi = mid;
        }
    }
    0.5 * (lo + hi)
}

/// The earliest time at which the edges are coplanar and within `radius` of
/// each other, or `None` if there is none or the motion is planar throughout.
fn coplanarity_contact(a: &MovingEdge, b: &MovingEdge, radius: f64) -> Option<f64> {
    let (cubic, scale) = coplanarity_cubic(a, b);
    let largest = cubic.iter().fold(0.0_f64, |m, c| m.max(c.abs()));
    if largest <= COPLANAR_NOISE * scale {
        // Coplanar for the whole step: every time is a root, so the first
        // contact is the first time within `radius`, found by the shell search.
        return None;
    }
    unit_interval_roots(cubic)
        .into_iter()
        .find(|&t| pair_at(a, b, t).distance <= radius)
}

/// The first time at which the distance between the segments of `a` and `b`
/// reaches `radius`, or `None` if it stays above it for the whole step.
///
/// `start_distance` is the distance at `t = 0`, which must exceed `radius`. The
/// distance of two segments whose end points move linearly is a minimum over
/// point pairs, each of which moves with a velocity that is a convex
/// combination of the differences of the end point displacements; the distance
/// is therefore Lipschitz continuous in time with constant `L`, the largest
/// such difference. On an interval `[lo, hi]` of width `w` it cannot fall below
/// `(d(lo) + d(hi) - L w) / 2`, so an interval whose bound exceeds `radius` is
/// discarded. The remaining intervals are halved, earliest first, until they are
/// narrower than [`SHELL_TOLERANCE`] allows; the answer is then within that
/// tolerance of the true entry time. Because every discarded interval is
/// certified, no dip of the distance below `radius` can be skipped.
fn shell_entry(a: &MovingEdge, b: &MovingEdge, radius: f64, start_distance: f64) -> Option<f64> {
    let [da0, da1] = a.displacement();
    let [db0, db1] = b.displacement();
    let lipschitz = [(da0, db0), (da0, db1), (da1, db0), (da1, db1)]
        .iter()
        .fold(0.0_f64, |largest, &(pa, pb)| {
            largest.max(len3(sub3(pa, pb)))
        });

    let distance_at = |t: f64| pair_at(a, b, t).distance;
    let leaf_width = if lipschitz > 0.0 {
        (2.0 * SHELL_TOLERANCE / lipschitz).max(4.0 * f64::EPSILON)
    } else {
        1.0
    };

    let mut evaluations = 1usize;
    // Intervals still to examine as `(lo, hi, distance at lo, distance at hi)`;
    // the earliest one is on top.
    let mut pending = vec![(0.0, 1.0, start_distance, distance_at(1.0))];
    while let Some((lo, hi, d_lo, d_hi)) = pending.pop() {
        let lower_bound = 0.5 * (d_lo + d_hi - lipschitz * (hi - lo));
        if lower_bound > radius || lower_bound.is_nan() {
            continue;
        }
        if hi - lo <= leaf_width {
            return Some(if d_lo <= radius { lo } else { hi });
        }
        if evaluations >= MAX_SHELL_EVALUATIONS {
            // Not certified apart and out of budget: report the earliest time at
            // which contact is still possible.
            return Some(lo);
        }
        let mid = 0.5 * (lo + hi);
        let d_mid = distance_at(mid);
        evaluations += 1;
        pending.push((mid, hi, d_mid, d_hi));
        pending.push((lo, mid, d_lo, d_mid));
    }
    None
}

/// The time of the first contact of the edges `a` and `b` within `radius`, in
/// the local frame (see [`edge_pair_contact`]).
fn first_contact_time(a: &MovingEdge, b: &MovingEdge, radius: f64) -> Option<f64> {
    let start = pair_at(a, b, 0.0);
    if start.distance <= radius {
        return Some(0.0);
    }
    coplanarity_contact(a, b, radius).or_else(|| shell_entry(a, b, radius, start.distance))
}

/// Contact normal (pointing from edge B towards edge A) of an edge pair at the
/// time of impact `toi`.
///
/// When the edges are still apart at `toi` (a graze of the contact shell, or a
/// contact radius larger than the gap) the normal is the unit separation of the
/// closest points. At an exact coplanarity root the closest points coincide and
/// the separation carries no direction, so the common perpendicular of the two
/// edges is used instead, oriented towards the side on which edge A started the
/// step. For parallel touching edges the separation at the start of the step,
/// then the approach direction, provide the normal.
fn contact_normal(a: &MovingEdge, b: &MovingEdge, toi: f64) -> [f64; 3] {
    let at_toi = pair_at(a, b, toi);
    let [pa0, pa1] = a.at(toi);
    let [pb0, pb1] = b.at(toi);
    let dir_a = sub3(pa1, pa0);
    let dir_b = sub3(pb1, pb0);
    let degenerate = 1e-9 * (len3(dir_a) + len3(dir_b));

    let separation = sub3(at_toi.closest_a, at_toi.closest_b);
    let gap = len3(separation);
    if gap > degenerate {
        return scale3(separation, 1.0 / gap);
    }

    let start = pair_at(a, b, 0.0);
    let reference = sub3(start.closest_a, start.closest_b);
    let reference_len = len3(reference);
    let [da0, da1] = a.displacement();
    let [db0, db1] = b.displacement();
    let velocity_a = add3(scale3(da0, 1.0 - at_toi.s), scale3(da1, at_toi.s));
    let velocity_b = add3(scale3(db0, 1.0 - at_toi.u), scale3(db1, at_toi.u));
    let relative_velocity = sub3(velocity_a, velocity_b);

    let cross = cross3(dir_a, dir_b);
    let cross_len = len3(cross);
    if cross_len > 1e-12 * len3(dir_a) * len3(dir_b) {
        let normal = scale3(cross, 1.0 / cross_len);
        let orientation = if reference_len > degenerate {
            dot3(normal, reference)
        } else {
            -dot3(normal, relative_velocity)
        };
        return if orientation < 0.0 {
            scale3(normal, -1.0)
        } else {
            normal
        };
    }
    if reference_len > degenerate {
        return scale3(reference, 1.0 / reference_len);
    }
    normalize3(scale3(relative_velocity, -1.0))
}

/// Time of impact and contact normal of one pair of edges.
#[derive(Clone, Copy, Debug)]
pub(super) struct EdgeContact {
    /// Time of impact in `[0, 1]`.
    pub toi: f64,
    /// Unit contact normal at the time of impact, pointing from edge B towards
    /// edge A.
    pub normal: [f64; 3],
}

/// The local frame in which an edge pair is examined.
struct PairFrame {
    /// Origin of the frame: the first vertex of edge A at the start of the step.
    origin: [f64; 3],
    /// Unit of length: the longest edge vector of the pair.
    unit: f64,
    /// Contact radius in the units of the input, raised to the rounding floor
    /// (see [`RADIUS_FLOOR`]).
    radius: f64,
}

impl PairFrame {
    /// The frame of the pair `a`, `b` for the contact radius `radius`, or `None`
    /// if the pair has no finite extent.
    fn new(a: &MovingEdge, b: &MovingEdge, radius: f64) -> Option<Self> {
        let origin = a.start[0];
        let length = a.longest_vector().max(b.longest_vector());
        let unit = if length.is_finite() && length > 0.0 {
            length
        } else {
            1.0
        };
        let extent = a.extent_from(origin).max(b.extent_from(origin));
        if !extent.is_finite() {
            return None;
        }
        Some(PairFrame {
            origin,
            unit,
            radius: radius.max(RADIUS_FLOOR * unit.max(extent)),
        })
    }
}

/// Continuous collision detection of two moving edges within `radius`.
///
/// `None` for a negative or NaN `radius` (a contact distance that can never be
/// reached), non-finite coordinates, or edges that stay further apart than
/// `radius` during the whole step.
pub(super) fn edge_pair_contact(
    a: &MovingEdge,
    b: &MovingEdge,
    radius: f64,
) -> Option<EdgeContact> {
    if radius.is_nan() || radius < 0.0 || !a.is_finite() || !b.is_finite() {
        return None;
    }
    let frame = PairFrame::new(a, b, radius)?;
    let inv_unit = 1.0 / frame.unit;
    let local_a = a.rescaled(frame.origin, inv_unit);
    let local_b = b.rescaled(frame.origin, inv_unit);
    if !local_a.is_finite() || !local_b.is_finite() {
        return None;
    }
    let toi = first_contact_time(&local_a, &local_b, frame.radius * inv_unit)?;
    Some(EdgeContact {
        toi,
        normal: contact_normal(&local_a, &local_b, toi),
    })
}

/// A moving edge together with the data the broad phase needs.
struct PreparedEdge {
    edge: MovingEdge,
    /// Lower corner of the box enclosing the whole trajectory.
    lo: [f64; 3],
    /// Upper corner of the box enclosing the whole trajectory.
    hi: [f64; 3],
}

impl PreparedEdge {
    /// Build the edge `(i0, i1)`; `None` if an index is outside `verts0` or `verts1`.
    fn new(i0: usize, i1: usize, verts0: &[[f64; 3]], verts1: &[[f64; 3]]) -> Option<Self> {
        let edge = MovingEdge::from_mesh(i0, i1, verts0, verts1)?;
        let (lo, hi) = edge.swept_bounds();
        Some(PreparedEdge { edge, lo, hi })
    }
}

/// Exact edge-edge CCD of two edge sets over one step; see
/// [`super::batch_edge_edge_ccd`].
pub(super) fn edge_edge_batch(
    radius: f64,
    same_mesh: bool,
    edges_a: &[(usize, usize)],
    verts0_a: &[[f64; 3]],
    verts1_a: &[[f64; 3]],
    edges_b: &[(usize, usize)],
    verts0_b: &[[f64; 3]],
    verts1_b: &[[f64; 3]],
) -> Vec<EdgeCcdResult> {
    if radius.is_nan() || radius < 0.0 {
        return Vec::new();
    }
    let prepared_b: Vec<Option<PreparedEdge>> = edges_b
        .iter()
        .map(|&(b0, b1)| PreparedEdge::new(b0, b1, verts0_b, verts1_b))
        .collect();
    let mut results = Vec::new();
    for (index_a, &(a0, a1)) in edges_a.iter().enumerate() {
        let Some(a) = PreparedEdge::new(a0, a1, verts0_a, verts1_a) else {
            continue;
        };
        for (index_b, (&(b0, b1), prepared)) in edges_b.iter().zip(prepared_b.iter()).enumerate() {
            let Some(b) = prepared else {
                continue;
            };
            // Edges of one mesh that share a vertex always touch there.
            if same_mesh && (a0 == b0 || a0 == b1 || a1 == b0 || a1 == b1) {
                continue;
            }
            // Broad phase: two edges can only come within the contact radius of
            // each other if their swept boxes, grown by that radius, overlap. The
            // radius used by the narrow phase is the given one raised to the
            // rounding floor, which is at most `RADIUS_FLOOR` times the diagonal
            // of the union of the two boxes (every length the floor is taken
            // from is a distance between points of that union).
            let diagonal = len3(sub3(
                [
                    a.hi[0].max(b.hi[0]),
                    a.hi[1].max(b.hi[1]),
                    a.hi[2].max(b.hi[2]),
                ],
                [
                    a.lo[0].min(b.lo[0]),
                    a.lo[1].min(b.lo[1]),
                    a.lo[2].min(b.lo[2]),
                ],
            ));
            let reach = radius + RADIUS_FLOOR * diagonal;
            if (0..3).any(|k| a.lo[k] > b.hi[k] + reach || b.lo[k] > a.hi[k] + reach) {
                continue;
            }
            if let Some(contact) = edge_pair_contact(&a.edge, &b.edge, radius) {
                results.push(EdgeCcdResult {
                    edge_a: index_a,
                    edge_b: index_b,
                    toi: contact.toi,
                    normal: contact.normal,
                });
            }
        }
    }
    results
}
