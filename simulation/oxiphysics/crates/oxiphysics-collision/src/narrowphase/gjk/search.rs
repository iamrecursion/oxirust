// Copyright 2026 COOLJAPAN OU (Team KitaSan)
// SPDX-License-Identifier: Apache-2.0

//! Robust overlap-versus-separation search shared by the GJK entry points.
//!
//! The search runs GJK in distance form on the Minkowski difference `A - B`:
//! every step takes the point `v` of the current simplex that is closest to the
//! origin, queries the support point `w` of `A - B` in the unit direction
//! `d = -v / |v|` and keeps the vertices that carry the new closest point. Each
//! step yields two bounds on the distance between the shapes:
//!
//! * `|v|` is an upper bound (`v` is a point of `A - B`);
//! * `lower = -w.d` is a lower bound (the plane through `w` with normal `d`
//!   supports `A - B`, so every point of `A - B` lies at least `lower` beyond
//!   the origin along `-d`).
//!
//! The decision rules follow from these bounds and from the rounding floor of
//! the computation (`floor`, a small multiple of the machine epsilon times the
//! magnitude of the world-space support points, never less than a `1e-13`
//! fraction of the Minkowski extent):
//!
//! * `lower > floor` certifies a separating plane: **separated**;
//! * four vertices carrying the closest point mean that the origin is enclosed
//!   by a tetrahedron of `A - B`: **overlap** (that tetrahedron seeds EPA);
//! * `|v| <= floor` means that the shapes touch within rounding: **overlap**;
//! * when the iteration cannot make progress any more (the support point repeats
//!   a vertex, `|v|` stalls for several steps or the iteration budget runs out)
//!   the best distance estimate decides: the pair overlaps only if `|v|` lies
//!   within a relative `1e-10` of the Minkowski extent (the resolution of the
//!   sub-volume signs near contact), otherwise it is reported as separated. A
//!   budget exhaustion is therefore never turned into an overlap of shapes that
//!   are measurably apart.
//!
//! The closest point of each simplex comes from the signed-volume
//! sub-algorithm ([`closest_on_simplex`]), which resolves sliver simplices that
//! a tolerance-based Johnson sub-algorithm treats as flat. When an overlap is
//! decided on a simplex with fewer than four vertices (the origin lies on a
//! segment or triangle, as happens for symmetric configurations), the simplex
//! is grown into an enclosing tetrahedron for EPA.
//!
//! The direction is normalised, so the separation test is scale invariant; the
//! classical boolean formulation compared an unnormalised `w.(-v)` against an
//! absolute tolerance and could oscillate between simplices of nearly touching
//! shapes until its budget ran out, which it then reported as an overlap.

use oxiphysics_core::math::{Real, Vec3};

use super::distance::{DUPLICATE_TOLERANCE, STALL_STEPS, STALL_TOLERANCE};
use super::functions::{MAX_ITERATIONS, do_simplex};
use super::signed_volume::closest_on_simplex;
use super::types::{GjkIterationRecord, GjkSearch, Simplex, SupportPoint};

/// Upper bound on the number of support queries of one search.
pub(crate) const SEARCH_MAX_ITERATIONS: usize = 128;
/// Distance, relative to the extent of `A - B`, below which the shapes count as
/// touching.
const TOUCH_TOLERANCE: Real = 1e-13;
/// Distance, relative to the extent of `A - B`, within which a search that
/// stalls without a certificate counts the shapes as touching.
const UNRESOLVED_TOLERANCE: Real = 1e-10;
/// Safety factor on the rounding error `epsilon * coordinate magnitude` of a
/// support point and of a dot product with it.
const NOISE_FACTOR: Real = 64.0;

/// Magnitudes seen by the search, from which its tolerances derive.
#[derive(Debug, Clone, Copy)]
struct Magnitudes {
    /// Largest Minkowski vertex `|w|`.
    extent: Real,
    /// Largest world-space coordinate magnitude of a support point of A or B.
    coordinates: Real,
}

impl Magnitudes {
    fn new(first: &SupportPoint) -> Self {
        let mut magnitudes = Self {
            extent: 0.0,
            coordinates: 0.0,
        };
        magnitudes.include(first);
        magnitudes
    }

    fn include(&mut self, point: &SupportPoint) {
        let extent = point.point.norm();
        let coordinates = point.support_a.norm().max(point.support_b.norm());
        if extent.is_finite() {
            self.extent = self.extent.max(extent);
        }
        if coordinates.is_finite() {
            self.coordinates = self.coordinates.max(coordinates).max(self.extent);
        }
    }

    /// Distances at or below this value cannot be told apart from contact.
    fn floor(&self) -> Real {
        (TOUCH_TOLERANCE * self.extent)
            .max(NOISE_FACTOR * Real::EPSILON * self.coordinates.max(self.extent))
            .max(Real::MIN_POSITIVE)
    }

    /// Width of the band around contact within which a search that can no
    /// longer make progress counts the shapes as touching.
    ///
    /// Such a search ends on a simplex whose closest point `v` is resolved only
    /// to the precision with which the signs of its (often sliver-shaped)
    /// sub-volumes are determined. Near contact of curved shapes that is a
    /// relative `1e-11` or so of the Minkowski extent, so the band is a relative
    /// [`UNRESOLVED_TOLERANCE`] of the extent, and never less than twice the
    /// rounding floor.
    fn precision_band(&self) -> Real {
        (UNRESOLVED_TOLERANCE * self.extent).max(2.0 * self.floor())
    }

    /// Change of `|v|` below which a step counts as making no progress: a
    /// relative rounding error of `|v|` itself or the absolute rounding error of
    /// the vertex coordinates, whichever is larger.
    fn stall_threshold(&self, previous_length: Real) -> Real {
        (STALL_TOLERANCE * previous_length).max(Real::EPSILON * self.coordinates)
    }
}

/// The first `count` vertices as a [`Simplex`].
fn to_simplex(vertices: &[SupportPoint; 4], count: usize) -> Simplex {
    let mut simplex = Simplex::new();
    for vertex in vertices.iter().take(count) {
        simplex.push(*vertex);
    }
    simplex
}

/// Unit seed direction, falling back to `+x` for a zero or non-finite input.
fn seed_direction(direction: Vec3) -> Vec3 {
    let length = direction.norm();
    if length.is_finite() && length > Real::MIN_POSITIVE {
        direction / length
    } else {
        Vec3::x()
    }
}

/// Decide overlap versus separation of the Minkowski difference whose support
/// mapping is `support_fn`.
///
/// `seed` is the first search direction (typically the offset from A to B or a
/// warm-start direction); it only affects the iteration count, never the
/// answer. When `trace` is given, one record per support query is appended to
/// it. Returns the outcome with the terminating simplex, and the number of
/// support queries made after the seed.
pub(crate) fn boundary_search<F>(
    mut support_fn: F,
    seed: Vec3,
    trace: Option<&mut Vec<GjkIterationRecord>>,
) -> (GjkSearch, usize)
where
    F: FnMut(&Vec3) -> SupportPoint,
{
    let (outcome, iterations) = decide(&mut support_fn, seed, trace);
    match outcome {
        GjkSearch::Overlap(carrier) if carrier.len() < 4 => {
            let (seed_simplex, extra) = enclosing_seed(&mut support_fn, carrier);
            (GjkSearch::Overlap(seed_simplex), iterations + extra)
        }
        other => (other, iterations),
    }
}

/// The decision part of [`boundary_search`]: overlap versus separation, with
/// the terminating simplex as found.
fn decide<F>(
    support_fn: &mut F,
    seed: Vec3,
    mut trace: Option<&mut Vec<GjkIterationRecord>>,
) -> (GjkSearch, usize)
where
    F: FnMut(&Vec3) -> SupportPoint,
{
    let seed = seed_direction(seed);
    let first = support_fn(&seed);
    let mut magnitudes = Magnitudes::new(&first);
    // The seed query is a support query like any other: `first` is extreme
    // along `seed`, so the plane through it bounds `A - B` and may already
    // certify the gap. Its direction is exact (it does not derive from a nearly
    // vanishing closest point), which matters for nearly touching shapes.
    let seed_lower = -first.point.dot(&seed);
    let seed_separated = seed_lower.is_finite() && seed_lower > magnitudes.floor();
    if let Some(records) = trace.as_deref_mut() {
        records.push(GjkIterationRecord {
            simplex_size: 0,
            direction: seed,
            new_support: first.point,
            dot_positive: !seed_separated,
        });
    }
    if seed_separated {
        let mut simplex = Simplex::new();
        simplex.push(first);
        return (GjkSearch::Separated(simplex), 0);
    }
    let mut vertices = [first; 4];
    let mut count = 1;
    let mut iterations = 0;
    let mut previous_length = Real::INFINITY;
    let mut stalled_steps = 0;
    loop {
        // Closest point of the current simplex and the vertices carrying it.
        let mut points = [Vec3::zeros(); 4];
        for (slot, vertex) in points.iter_mut().zip(&vertices[..count]) {
            *slot = vertex.point;
        }
        let nearest = closest_on_simplex(&points[..count]);
        let mut carried = [first; 4];
        for (slot, &index) in carried.iter_mut().zip(&nearest.keep[..nearest.len]) {
            *slot = vertices[index];
        }
        vertices = carried;
        count = nearest.len;
        if count == 4 {
            return (GjkSearch::Overlap(to_simplex(&vertices, count)), iterations);
        }
        let closest = nearest.point;
        let length = closest.norm();
        if !length.is_finite() {
            // Non-finite shape data: there is no evidence of contact.
            return (
                GjkSearch::Separated(to_simplex(&vertices, count)),
                iterations,
            );
        }
        if length <= magnitudes.floor() {
            return (GjkSearch::Overlap(to_simplex(&vertices, count)), iterations);
        }
        if iterations > 0 && previous_length - length <= magnitudes.stall_threshold(previous_length)
        {
            stalled_steps += 1;
        } else {
            stalled_steps = 0;
        }
        if stalled_steps >= STALL_STEPS || iterations >= SEARCH_MAX_ITERATIONS {
            return (undecided(&vertices, count, length, &magnitudes), iterations);
        }
        previous_length = length;
        let direction = search_direction(&vertices, count, &closest, length);
        let extreme = support_fn(&direction);
        iterations += 1;
        magnitudes.include(&extreme);
        let lower = -extreme.point.dot(&direction);
        let floor = magnitudes.floor();
        let separated = lower > floor;
        if let Some(records) = trace.as_deref_mut() {
            records.push(GjkIterationRecord {
                simplex_size: count,
                direction,
                new_support: extreme.point,
                dot_positive: !separated,
            });
        }
        if !lower.is_finite() {
            return (
                GjkSearch::Separated(to_simplex(&vertices, count)),
                iterations,
            );
        }
        if separated {
            return (
                GjkSearch::Separated(to_simplex(&vertices, count)),
                iterations,
            );
        }
        // A support point that repeats a vertex of the simplex cannot improve
        // the closest point. (While `lower <= floor`, the bounds `lower` and
        // `|v|` themselves only agree once `|v|` is within the floor, which the
        // touching test above already caught; a converged-bounds test with a
        // rounding allowance would wrongly stop shallow overlaps of curved
        // shapes, whose closest simplex point approaches the origin slowly.)
        let tolerance = (DUPLICATE_TOLERANCE * magnitudes.extent).max(floor);
        let repeated = vertices[..count]
            .iter()
            .any(|vertex| (vertex.point - extreme.point).norm() <= tolerance);
        if repeated {
            return (undecided(&vertices, count, length, &magnitudes), iterations);
        }
        vertices[count] = extreme;
        count += 1;
    }
}

/// Unit direction from the closest simplex point `closest` (of length
/// `length > 0`) towards the origin.
///
/// When the closest point lies inside a triangle the direction is the
/// triangle's normal: a cross product of edges is accurate to a relative
/// rounding error regardless of how close the origin is to the triangle, while
/// `-closest / length` inherits an angular error of `epsilon * size / length`
/// that would keep a face separated by a tiny gap from being certified.
fn search_direction(
    vertices: &[SupportPoint; 4],
    count: usize,
    closest: &Vec3,
    length: Real,
) -> Vec3 {
    let fallback = -closest / length;
    if count != 3 {
        return fallback;
    }
    let normal =
        (vertices[1].point - vertices[0].point).cross(&(vertices[2].point - vertices[0].point));
    let normal_length = normal.norm();
    if !normal_length.is_finite() || normal_length <= Real::MIN_POSITIVE {
        return fallback;
    }
    let unit = normal / normal_length;
    // Orient towards the origin; reject a normal that disagrees with the
    // closest-point direction (a sliver triangle whose normal is unreliable).
    let oriented = if unit.dot(&fallback) < 0.0 {
        -unit
    } else {
        unit
    };
    if oriented.dot(&fallback) < 0.5 {
        fallback
    } else {
        oriented
    }
}

/// Grow the carrier of an overlap decision with fewer than four vertices into
/// a tetrahedron that encloses the origin, the seed EPA needs.
///
/// The overlap itself is already decided (the origin lies within the rounding
/// floor of the carrier), so this only spends support queries on the seed: it
/// continues with the classical simplex-refinement steps ([`do_simplex`]),
/// which expand a feature that passes through the origin in a perpendicular
/// direction. If no enclosing tetrahedron is found within the budget (shapes
/// that merely touch have no interior around the origin) the carrier is
/// returned unchanged. Returns the simplex and the number of support queries.
fn enclosing_seed<F>(support_fn: &mut F, carrier: Simplex) -> (Simplex, usize)
where
    F: FnMut(&Vec3) -> SupportPoint,
{
    let mut simplex = carrier.clone();
    let mut direction = match simplex.len() {
        0 => return (carrier, 0),
        1 => -simplex.points[0].point,
        _ => match do_simplex(&mut simplex) {
            Some(direction) => direction,
            None => return (simplex, 0),
        },
    };
    let mut queries = 0;
    while queries < MAX_ITERATIONS {
        let length = direction.norm();
        let unit = if length.is_finite() && length > Real::MIN_POSITIVE {
            direction / length
        } else {
            perpendicular_hint(&simplex)
        };
        let extreme = support_fn(&unit);
        queries += 1;
        let reach = extreme.point.dot(&unit);
        if reach.is_nan() || reach < 0.0 {
            break;
        }
        simplex.push(extreme);
        match do_simplex(&mut simplex) {
            Some(next) => direction = next,
            None => {
                if simplex.len() == 4 {
                    return (simplex, queries);
                }
                break;
            }
        }
    }
    (carrier, queries)
}

/// A direction off the affine hull of a degenerate simplex, used when the
/// refinement step yields a vanishing direction.
fn perpendicular_hint(simplex: &Simplex) -> Vec3 {
    let edge = match simplex.points.as_slice() {
        [first, second, ..] => second.point - first.point,
        _ => return Vec3::x(),
    };
    let axis = if edge.x.abs() <= edge.y.abs() && edge.x.abs() <= edge.z.abs() {
        Vec3::x()
    } else if edge.y.abs() <= edge.z.abs() {
        Vec3::y()
    } else {
        Vec3::z()
    };
    let perpendicular = edge.cross(&axis);
    let length = perpendicular.norm();
    if length.is_finite() && length > Real::MIN_POSITIVE {
        perpendicular / length
    } else {
        Vec3::x()
    }
}

/// Outcome of a search that stopped without a certificate either way.
///
/// No separating plane above the rounding floor was found, and the origin is
/// not enclosed. The best distance estimate `length = |v|` (an upper bound)
/// decides: within the precision band of contact the shapes touch, beyond it
/// they are reported as separated so that an exhausted or stalled search never
/// produces an overlap of shapes that are measurably apart.
fn undecided(
    vertices: &[SupportPoint; 4],
    count: usize,
    length: Real,
    magnitudes: &Magnitudes,
) -> GjkSearch {
    let simplex = to_simplex(vertices, count);
    if length <= magnitudes.precision_band() {
        GjkSearch::Overlap(simplex)
    } else {
        GjkSearch::Separated(simplex)
    }
}
