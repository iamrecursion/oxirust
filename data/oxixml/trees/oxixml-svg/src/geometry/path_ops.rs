//! Whole-path geometry: normalization, bounding boxes, lengths, flattening,
//! and affine transformation.
//!
//! Everything in this module works on [`PathData`] — the parsed `d`
//! attribute — and everything in it goes through one internal step first:
//! the path is walked once ([`PathData::walk`]) and reduced to a sequence of
//! absolute moves, lines, cubic Bézier curves, elliptical arcs in center
//! form, and closepaths. That reduction is where the two pieces of path
//! semantics the parser deliberately leaves alone are applied:
//!
//! * **Smooth-curve reflection.** `S` takes its first control point from
//!   the reflection of the previous curve's second control point about the
//!   current point — but only if the previous command was `C` or `S`. After
//!   anything else (a lineto, a moveto, a closepath, a quadratic, an arc)
//!   the control point **is** the current point. `T` follows the same rule
//!   against `Q` and `T`. Getting this wrong is the single most common
//!   path-geometry bug, and [`normalize`] pins it.
//! * **Arc correction.** An `A` command's radii and rotation are stored as
//!   authored, and SVG 1.1 Appendix F.6.2's out-of-range handling —
//!   coincident endpoints mean no segment at all, a zero radius means a
//!   straight line, radii too small are scaled up — is applied here, by
//!   [`crate::geometry::arc`].
//!
//! # Exactness
//!
//! [`bounding_box`] is exact: cubics contribute their analytic extrema and
//! arcs their closed-form axis extrema, with no flattening anywhere.
//!
//! [`total_length`], [`point_at_length`] and [`flatten`] are adaptive and
//! take a tolerance; see [`crate::geometry::bezier`] for the error model.
//! For arcs they carry one further approximation, since an elliptical arc's
//! length is an elliptic integral with no closed form: the arc is converted
//! to cubic Béziers first, which costs about `2.7e-4` of a radius.
//!
//! [`normalize`] and [`transform_path`] convert arcs to cubic Béziers, so
//! they are approximate for arcs in the same way — and unavoidably so for
//! [`transform_path`], since a general affine transform of an SVG arc is not
//! an SVG arc with any parameters at all unless the transform is a
//! similarity.

use crate::geometry::arc::{ArcOutcome, ArcParams, CenterParams};
use crate::geometry::bezier::{CubicBezier, QuadBezier};
use crate::geometry::matrix::Matrix;
use crate::geometry::primitives::{Point, Rect};
use crate::values::path::{PathData, PathSegment};

/// One segment of a path, reduced to absolute coordinates and one of five
/// shapes.
///
/// Arcs keep both parameterizations: the endpoint form reproduces the
/// segment's exact endpoints when it is converted to Bézier curves, and the
/// center form answers the geometric questions (where does the ellipse turn
/// around) that the Bézier approximation could only estimate.
#[derive(Debug, Clone, Copy, PartialEq)]
enum Piece {
    /// A moveto: draws nothing, but starts a subpath at this point.
    Move { to: Point },
    /// A straight line, from a lineto, a horizontal or vertical lineto, or a
    /// degenerate arc.
    Line { from: Point, to: Point },
    /// A cubic Bézier curve, from a `C`, `S`, `Q` or `T` command.
    Cubic(CubicBezier),
    /// An elliptical arc, in both parameterizations.
    Arc {
        endpoint: ArcParams,
        center: CenterParams,
    },
    /// A closepath: the straight line back to the subpath's first point.
    Close { from: Point, to: Point },
}

impl Piece {
    /// The point the piece starts at. A moveto reports the point it
    /// establishes, since it has no incoming edge of its own.
    fn start(&self) -> Point {
        match self {
            Self::Move { to } => *to,
            Self::Line { from, .. } | Self::Close { from, .. } => *from,
            Self::Cubic(curve) => curve.p0,
            Self::Arc { endpoint, .. } => endpoint.from,
        }
    }

    /// The piece's length, measured to `tolerance`; zero for a moveto.
    fn length(&self, tolerance: f64) -> f64 {
        match self {
            Self::Move { .. } => 0.0,
            Self::Line { from, to } | Self::Close { from, to } => distance(*from, *to),
            Self::Cubic(curve) => curve.arc_length(tolerance),
            Self::Arc { endpoint, .. } => endpoint
                .to_cubics()
                .iter()
                .map(|curve| curve.arc_length(tolerance))
                .sum(),
        }
    }
}

/// The Euclidean distance between two points.
fn distance(a: Point, b: Point) -> f64 {
    (b.x - a.x).hypot(b.y - a.y)
}

/// Linear interpolation between two points.
fn lerp(a: Point, b: Point, t: f64) -> Point {
    Point::new(a.x + (b.x - a.x) * t, a.y + (b.y - a.y) * t)
}

/// The reflection of `control` about `current`, or `current` itself when
/// there is no control point to reflect.
///
/// This is the smooth-command rule of SVG 1.1 §8.3.6: "If there is no
/// previous command or if the previous command was not a `C`, `c`, `S` or
/// `s`, assume the first control point is coincident with the current
/// point."
fn reflect(control: Option<Point>, current: Point) -> Point {
    match control {
        Some(control) => Point::new(2.0 * current.x - control.x, 2.0 * current.y - control.y),
        None => current,
    }
}

/// Grows an optional bounding box to include `rect`.
fn include_rect(bounds: &mut Option<Rect>, rect: Rect) {
    *bounds = Some(match *bounds {
        None => rect,
        Some(existing) => existing.union(&rect),
    });
}

/// Reduces a path to absolute pieces, resolving smooth-curve reflection and
/// arc out-of-range handling.
///
/// The two reflection states are tracked separately and independently, which
/// is what the specification's wording requires: `S` reflects only after
/// `C`/`S`, and `T` only after `Q`/`T`, so a `Q` between two `C`s does not
/// give the second `C`'s `S` successor anything to reflect.
fn pieces(path: &PathData) -> Vec<Piece> {
    let mut out = Vec::with_capacity(path.len());
    let mut last_cubic_control: Option<Point> = None;
    let mut last_quad_control: Option<Point> = None;

    for step in path.walk() {
        let from = Point::new(step.start.0, step.start.1);
        let to = Point::new(step.end.0, step.end.1);
        let mut cubic_control = None;
        let mut quad_control = None;

        match step.absolute {
            PathSegment::MoveTo { .. } => out.push(Piece::Move { to }),
            PathSegment::LineTo { .. }
            | PathSegment::HorizontalTo { .. }
            | PathSegment::VerticalTo { .. } => out.push(Piece::Line { from, to }),
            PathSegment::CurveTo { x1, y1, x2, y2, .. } => {
                let control2 = Point::new(x2, y2);
                out.push(Piece::Cubic(CubicBezier::new(
                    from,
                    Point::new(x1, y1),
                    control2,
                    to,
                )));
                cubic_control = Some(control2);
            }
            PathSegment::SmoothCurveTo { x2, y2, .. } => {
                let control2 = Point::new(x2, y2);
                out.push(Piece::Cubic(CubicBezier::new(
                    from,
                    reflect(last_cubic_control, from),
                    control2,
                    to,
                )));
                cubic_control = Some(control2);
            }
            PathSegment::QuadTo { x1, y1, .. } => {
                let control = Point::new(x1, y1);
                out.push(Piece::Cubic(QuadBezier::new(from, control, to).to_cubic()));
                quad_control = Some(control);
            }
            PathSegment::SmoothQuadTo { .. } => {
                let control = reflect(last_quad_control, from);
                out.push(Piece::Cubic(QuadBezier::new(from, control, to).to_cubic()));
                quad_control = Some(control);
            }
            PathSegment::ArcTo {
                rx,
                ry,
                x_axis_rotation,
                large_arc,
                sweep,
                ..
            } => {
                let endpoint = ArcParams::new(from, rx, ry, x_axis_rotation, large_arc, sweep, to);
                match endpoint.to_center() {
                    // F.6.2: coincident endpoints omit the segment entirely.
                    ArcOutcome::Omitted => {}
                    // F.6.2: a zero radius draws a straight line.
                    ArcOutcome::Line => out.push(Piece::Line { from, to }),
                    ArcOutcome::Arc(center) => out.push(Piece::Arc { endpoint, center }),
                }
            }
            PathSegment::Close => out.push(Piece::Close { from, to }),
        }

        last_cubic_control = cubic_control;
        last_quad_control = quad_control;
    }
    out
}

/// Rebuilds path data from pieces, optionally mapping every point through a
/// matrix on the way.
///
/// Arcs are expanded to cubic Béziers here, which is what makes the output
/// grammar `M`/`L`/`C`/`Z` and what makes an affine transform expressible at
/// all.
fn emit(pieces: &[Piece], matrix: Option<&Matrix>) -> PathData {
    let map = |point: Point| match matrix {
        Some(matrix) => {
            let (x, y) = matrix.apply(point.x, point.y);
            Point::new(x, y)
        }
        None => point,
    };
    let mut segments = Vec::with_capacity(pieces.len());
    let push_cubic = |segments: &mut Vec<PathSegment>, curve: &CubicBezier| {
        let control1 = map(curve.p1);
        let control2 = map(curve.p2);
        let end = map(curve.p3);
        segments.push(PathSegment::CurveTo {
            abs: true,
            x1: control1.x,
            y1: control1.y,
            x2: control2.x,
            y2: control2.y,
            x: end.x,
            y: end.y,
        });
    };
    // Path data must begin with a moveto. The parser enforces that, but a
    // `PathData` assembled by hand need not, and the implicit start point is
    // then the origin — which a transform would otherwise leave behind,
    // since there would be no segment carrying it.
    if let Some(first) = pieces.first()
        && !matches!(first, Piece::Move { .. })
    {
        let start = map(first.start());
        segments.push(PathSegment::MoveTo {
            abs: true,
            x: start.x,
            y: start.y,
        });
    }

    for piece in pieces {
        match piece {
            Piece::Move { to } => {
                let to = map(*to);
                segments.push(PathSegment::MoveTo {
                    abs: true,
                    x: to.x,
                    y: to.y,
                });
            }
            Piece::Line { to, .. } => {
                let to = map(*to);
                segments.push(PathSegment::LineTo {
                    abs: true,
                    x: to.x,
                    y: to.y,
                });
            }
            Piece::Cubic(curve) => push_cubic(&mut segments, curve),
            Piece::Arc { endpoint, .. } => {
                for curve in endpoint.to_cubics() {
                    push_cubic(&mut segments, &curve);
                }
            }
            Piece::Close { .. } => segments.push(PathSegment::Close),
        }
    }
    PathData::from_segments(segments)
}

/// Rewrites a path using only absolute movetos, linetos, cubic Béziers and
/// closepaths.
///
/// Every simplification the SVG path grammar allows is undone:
///
/// * relative commands become absolute (including the leading `m`, which
///   the specification already reads as an `M`);
/// * `H` and `V` become `L`, taking the missing coordinate from the current
///   point;
/// * `S` and `T` become `C` and `Q` with their implied control point made
///   explicit, per the reflection rule described in the
///   [module documentation](self);
/// * `Q` and `T` become `C` by exact degree elevation, so the output has one
///   curve family rather than two;
/// * `A` becomes a chain of `C` commands, or a single `L`, or nothing at
///   all, per SVG 1.1 Appendix F.6.2.
///
/// The result therefore uses exactly four commands — `M`, `L`, `C`, `Z` —
/// which is the grammar every downstream consumer (a renderer, a length
/// measurement, a transform) wants. Normalizing an already-normalized path
/// changes nothing further.
///
/// A path that does not begin with a moveto gains one, at the origin the
/// specification's current point starts from. [`PathData::parse`] rejects
/// such a path, but one assembled segment by segment can contain it, and the
/// output is always valid path data.
///
/// ```
/// use oxixml_svg::geometry::path_ops::normalize;
/// use oxixml_svg::values::path::PathData;
///
/// let path = PathData::parse("m 10 20 l 5 5 h 5 z").expect("valid path data");
/// assert_eq!(normalize(&path).to_string(), "M 10 20 L 15 25 L 20 25 Z");
///
/// // `S` after a lineto has nothing to reflect, so its first control point
/// // is the current point itself.
/// let smooth = PathData::parse("M0 0 L10 0 S20 10 30 0").expect("valid path data");
/// assert_eq!(normalize(&smooth).to_string(), "M 0 0 L 10 0 C 10 0 20 10 30 0");
/// ```
#[must_use]
pub fn normalize(path: &PathData) -> PathData {
    emit(&pieces(path), None)
}

/// Applies an affine transform to a path.
///
/// The path is [`normalize`]d first and the resulting control points are
/// mapped through `matrix`, which is exact for lines and Bézier curves: an
/// affine image of a Bézier curve is the Bézier curve through the affine
/// images of its control points.
///
/// Arcs are the exception, and the reason normalization is not optional
/// here. A general affine transform of an SVG elliptical arc cannot be
/// written as an SVG elliptical arc at all — the transformed ellipse's axes
/// are no longer the original's, and recovering the new `rx`, `ry` and
/// rotation needs an eigen-decomposition that a non-similarity transform can
/// make degenerate. Converting to cubics first is the standard resolution,
/// at the documented approximation cost.
///
/// ```
/// use oxixml_svg::geometry::matrix::Matrix;
/// use oxixml_svg::geometry::path_ops::transform_path;
/// use oxixml_svg::values::path::PathData;
///
/// let path = PathData::parse("M0 0 H10 V10").expect("valid path data");
/// let moved = transform_path(&path, &Matrix::translate(5.0, 5.0));
/// assert_eq!(moved.to_string(), "M 5 5 L 15 5 L 15 15");
/// ```
#[must_use]
pub fn transform_path(path: &PathData, matrix: &Matrix) -> PathData {
    emit(&pieces(path), Some(matrix))
}

/// The exact bounding box of a path's geometry, or `None` for a path with no
/// segments at all.
///
/// Exact in the strong sense: cubic Béziers contribute the analytic extrema
/// of their derivative rather than their control-point hull, and elliptical
/// arcs contribute the closed-form extrema of the ellipse rather than those
/// of a Bézier approximation. Nothing here is flattened, and there is no
/// tolerance to pass.
///
/// This is the *geometry* (fill) bounding box, the one SVG's `getBBox`
/// returns: it ignores stroke width, markers, and clipping. A lone moveto
/// contributes its point, so a path that draws nothing still has a
/// (zero-sized) box; a closepath contributes nothing, since both of its
/// endpoints are already in the box.
///
/// ```
/// use oxixml_svg::geometry::path_ops::bounding_box;
/// use oxixml_svg::values::path::PathData;
///
/// // Both control points reach y = 3, but the curve itself only reaches
/// // 2.25 — a hull-based box would be a quarter too tall.
/// let path = PathData::parse("M0 0 C0 3 4 3 4 0").expect("valid path data");
/// let bounds = bounding_box(&path).expect("a non-empty path");
/// assert_eq!((bounds.x, bounds.y, bounds.width, bounds.height), (0.0, 0.0, 4.0, 2.25));
/// ```
#[must_use]
pub fn bounding_box(path: &PathData) -> Option<Rect> {
    let mut bounds: Option<Rect> = None;
    for piece in pieces(path) {
        match piece {
            Piece::Move { to } => include_rect(&mut bounds, Rect::from_point(to)),
            Piece::Line { from, to } => include_rect(&mut bounds, Rect::from_points(from, to)),
            Piece::Cubic(curve) => include_rect(&mut bounds, curve.tight_bounds()),
            Piece::Arc { endpoint, center } => {
                // The endpoints come from the path rather than from
                // `point_at`, so the box always contains them exactly.
                include_rect(&mut bounds, Rect::from_points(endpoint.from, endpoint.to));
                include_rect(&mut bounds, center.tight_bounds());
            }
            // A closepath runs between two points the box already has.
            Piece::Close { .. } => {}
        }
    }
    bounds
}

/// The total length of everything a path draws, measured to `tolerance`.
///
/// Movetos contribute nothing, closepaths contribute the length of the
/// straight line they draw back to the start of their subpath, and curves
/// are measured by adaptive subdivision — see [`crate::geometry::bezier`]
/// for the error model and for how a `tolerance` that is not finite and
/// positive is replaced.
///
/// This is the geometric length; the `pathLength` attribute, which rescales
/// it for `stroke-dasharray` and `textPath` purposes, is an element-level
/// concern applied on top of this number rather than a replacement for it.
///
/// ```
/// use oxixml_svg::geometry::path_ops::total_length;
/// use oxixml_svg::values::path::PathData;
///
/// // A closed unit square: three drawn sides plus the closepath's fourth.
/// let path = PathData::parse("M0 0 H1 V1 H0 Z").expect("valid path data");
/// assert!((total_length(&path, 1e-9) - 4.0).abs() < 1e-12);
/// ```
#[must_use]
pub fn total_length(path: &PathData, tolerance: f64) -> f64 {
    pieces(path)
        .iter()
        .map(|piece| piece.length(tolerance))
        .sum()
}

/// The point at a given distance along a path, together with the direction
/// the path is heading there.
///
/// This is SVG's `getPointAtLength`, plus the tangent angle its
/// `animateMotion` and `textPath` consumers need. The angle is in **radians**,
/// measured as `atan2(dy, dx)` in user space — so, with SVG's y axis
/// pointing down, a positive angle turns clockwise on screen.
///
/// `distance` is clamped rather than rejected: a distance at or below zero
/// gives the path's start, and one at or beyond the total length gives the
/// end of the last segment that draws anything. When two segments meet
/// exactly at `distance`, the earlier one answers — with its *end* tangent,
/// not the next segment's start tangent.
///
/// `None` means there is nothing to point at: an empty path, one that draws
/// nothing (a lone moveto, or segments of zero length), or one whose
/// measured length is not finite because a coordinate overflowed while
/// parsing. Note the
/// deliberate asymmetry with [`bounding_box`], which *does* report a box for
/// a lone moveto: a point with no direction has a position but no tangent,
/// and returning an arbitrary angle would be worse than returning nothing.
///
/// ```
/// use oxixml_svg::geometry::path_ops::point_at_length;
/// use oxixml_svg::values::path::PathData;
///
/// let path = PathData::parse("M0 0 L10 0").expect("valid path data");
/// let (point, angle) = point_at_length(&path, 5.0, 1e-9).expect("the path draws something");
/// assert_eq!((point.x, point.y), (5.0, 0.0));
/// assert_eq!(angle, 0.0);
///
/// // A path that draws nothing has no point to report.
/// let dot = PathData::parse("M10 10").expect("valid path data");
/// assert!(point_at_length(&dot, 0.0, 1e-9).is_none());
/// ```
#[must_use]
pub fn point_at_length(path: &PathData, distance: f64, tolerance: f64) -> Option<(Point, f64)> {
    let pieces = pieces(path);
    let total: f64 = pieces.iter().map(|piece| piece.length(tolerance)).sum();
    // A total that is zero, negative, infinite or a NaN leaves nothing to
    // measure along: the path draws nothing, or its coordinates overflowed
    // while parsing.
    if !(total.is_finite() && total > 0.0) {
        return None;
    }
    let target = if distance.is_nan() {
        0.0
    } else {
        distance.clamp(0.0, total)
    };

    let mut traveled = 0.0;
    let mut last_drawing: Option<&Piece> = None;
    for piece in &pieces {
        let length = piece.length(tolerance);
        if length <= 0.0 {
            continue;
        }
        if traveled + length >= target {
            return Some(locate(piece, target - traveled, tolerance));
        }
        traveled += length;
        last_drawing = Some(piece);
    }
    // Only reachable when rounding leaves the accumulated length a hair
    // short of the total measured above; the end of the path is the answer.
    last_drawing.map(|piece| locate(piece, piece.length(tolerance), tolerance))
}

/// Approximates a path by one polyline per subpath.
///
/// Each returned polyline starts at its subpath's first point and follows it
/// through every curve, flattened to within `tolerance`. A closepath appends
/// the subpath's first point again, so a closed subpath comes back as a ring
/// whose last point equals its first; a subpath that is only a moveto comes
/// back as a single-point polyline, matching [`bounding_box`]'s treatment of
/// the same path.
///
/// Per SVG 1.1 §8.3.3, a drawing command after a closepath starts a new
/// subpath at the closed subpath's start point, so it opens a new polyline
/// rather than extending the closed one.
///
/// ```
/// use oxixml_svg::geometry::path_ops::flatten;
/// use oxixml_svg::values::path::PathData;
///
/// let path = PathData::parse("M0 0 L1 0 L1 1 Z M5 5").expect("valid path data");
/// let polylines = flatten(&path, 1e-6);
/// assert_eq!(polylines.len(), 2);
/// // The triangle closes back onto its own first point.
/// assert_eq!(polylines[0].len(), 4);
/// assert_eq!(polylines[0][0], polylines[0][3]);
/// // The lone moveto is a single point.
/// assert_eq!(polylines[1].len(), 1);
/// ```
#[must_use]
pub fn flatten(path: &PathData, tolerance: f64) -> Vec<Vec<Point>> {
    let mut polylines: Vec<Vec<Point>> = Vec::new();
    let mut current: Option<Vec<Point>> = None;

    for piece in pieces(path) {
        match piece {
            Piece::Move { to } => {
                if let Some(finished) = current.take() {
                    polylines.push(finished);
                }
                current = Some(vec![to]);
            }
            Piece::Line { from, to } => {
                open_at(&mut current, from).push(to);
            }
            Piece::Cubic(curve) => {
                let points = open_at(&mut current, curve.p0);
                curve.flatten(tolerance, &mut |point| points.push(point));
            }
            Piece::Arc { endpoint, .. } => {
                let points = open_at(&mut current, endpoint.from);
                for curve in endpoint.to_cubics() {
                    curve.flatten(tolerance, &mut |point| points.push(point));
                }
            }
            Piece::Close { from, to } => {
                open_at(&mut current, from).push(to);
                if let Some(finished) = current.take() {
                    polylines.push(finished);
                }
            }
        }
    }
    if let Some(finished) = current.take() {
        polylines.push(finished);
    }
    polylines
}

/// The polyline being built, started at `from` if none is open.
fn open_at(current: &mut Option<Vec<Point>>, from: Point) -> &mut Vec<Point> {
    current.get_or_insert_with(|| vec![from])
}

/// The point at `local` units along a single piece, and the tangent angle
/// there.
fn locate(piece: &Piece, local: f64, tolerance: f64) -> (Point, f64) {
    match piece {
        // A moveto never reaches this: `point_at_length` skips zero-length
        // pieces, and a moveto is always zero-length.
        Piece::Move { to } => (*to, 0.0),
        Piece::Line { from, to } | Piece::Close { from, to } => {
            let length = distance(*from, *to);
            let fraction = if length > 0.0 { local / length } else { 0.0 };
            (
                lerp(*from, *to, fraction.clamp(0.0, 1.0)),
                (to.y - from.y).atan2(to.x - from.x),
            )
        }
        Piece::Cubic(curve) => locate_in_cubic(curve, local, tolerance),
        Piece::Arc { endpoint, .. } => {
            let curves = endpoint.to_cubics();
            let mut remaining = local;
            for (index, curve) in curves.iter().enumerate() {
                let length = curve.arc_length(tolerance);
                if remaining <= length || index + 1 == curves.len() {
                    return locate_in_cubic(curve, remaining, tolerance);
                }
                remaining -= length;
            }
            // An arc always yields at least one curve, so the loop above
            // always returns; this is the shape of the fallback rather than
            // a reachable path.
            (endpoint.to, 0.0)
        }
    }
}

/// The point at `local` units along one cubic, and the tangent angle there.
///
/// The curve is flattened once, which yields a monotone table of
/// `(parameter, point)` samples; the target length brackets one chord of
/// that table, and interpolating the parameter across the bracket turns a
/// length back into a curve parameter. The point is then evaluated on the
/// curve itself rather than on the chord, so it is exactly on the curve
/// rather than within `tolerance` of it.
fn locate_in_cubic(curve: &CubicBezier, local: f64, tolerance: f64) -> (Point, f64) {
    let mut previous_point = curve.p0;
    let mut previous_t = 0.0;
    let mut traveled = 0.0;
    let mut found: Option<f64> = None;

    curve.flatten_with_params(tolerance, &mut |t, point| {
        if found.is_some() {
            return;
        }
        let chord = distance(previous_point, point);
        if chord > 0.0 && traveled + chord >= local {
            let fraction = ((local - traveled) / chord).clamp(0.0, 1.0);
            found = Some(previous_t + (t - previous_t) * fraction);
        } else {
            traveled += chord;
            previous_point = point;
            previous_t = t;
        }
    });

    let t = found.unwrap_or(1.0);
    (curve.eval(t), cubic_tangent(curve, t))
}

/// The tangent angle of a cubic at parameter `t`, in radians.
///
/// The derivative answers directly unless it vanishes — at a cusp, or
/// wherever a control point coincides with the endpoint beside it, which
/// real path data does contain. A secant across the parameter then stands in
/// for it, and the chord for that.
fn cubic_tangent(curve: &CubicBezier, t: f64) -> f64 {
    let derivative = curve.derivative_at(t);
    if derivative.x != 0.0 || derivative.y != 0.0 {
        return derivative.y.atan2(derivative.x);
    }
    let epsilon = 1e-6;
    let before = curve.eval((t - epsilon).max(0.0));
    let after = curve.eval((t + epsilon).min(1.0));
    if before != after {
        return (after.y - before.y).atan2(after.x - before.x);
    }
    (curve.p3.y - curve.p0.y).atan2(curve.p3.x - curve.p0.x)
}

#[cfg(test)]
mod tests {
    use std::f64::consts::{FRAC_PI_2, PI, TAU};

    use super::*;

    fn parse(input: &str) -> PathData {
        PathData::parse(input).unwrap_or_else(|error| panic!("{input:?}: {error}"))
    }

    /// The command letters of a path, as a string: a compact way to assert
    /// that normalization left only `M`, `L`, `C` and `Z` behind.
    fn commands(path: &PathData) -> String {
        path.segments().iter().map(PathSegment::command).collect()
    }

    fn assert_close(actual: f64, expected: f64, epsilon: f64) {
        assert!(
            (actual - expected).abs() <= epsilon,
            "expected {expected}, got {actual} (tolerance {epsilon})"
        );
    }

    fn assert_point_close(actual: Point, expected: Point, epsilon: f64) {
        assert!(
            (actual.x - expected.x).abs() <= epsilon && (actual.y - expected.y).abs() <= epsilon,
            "expected {expected:?}, got {actual:?} (tolerance {epsilon})"
        );
    }

    fn assert_rect_close(actual: Rect, expected: Rect, epsilon: f64) {
        for (actual, expected) in [
            (actual.x, expected.x),
            (actual.y, expected.y),
            (actual.width, expected.width),
            (actual.height, expected.height),
        ] {
            assert!(
                (actual - expected).abs() <= epsilon,
                "expected {expected:?}, got {actual:?} (tolerance {epsilon})"
            );
        }
    }

    /// The unit circle drawn as four quarter arcs, clockwise on screen.
    const CIRCLE: &str = "M1 0 A1 1 0 0 1 0 1 A1 1 0 0 1 -1 0 A1 1 0 0 1 0 -1 A1 1 0 0 1 1 0 Z";

    #[test]
    fn normalization_makes_every_command_absolute_and_explicit() {
        let path = parse("m 10 20 l 5 5 h 5 v -5 z");
        assert_eq!(
            normalize(&path).to_string(),
            "M 10 20 L 15 25 L 20 25 L 20 20 Z"
        );
    }

    #[test]
    fn normalization_leaves_only_four_commands() {
        let path = parse(
            "m 10 20 l 5 5 h 5 v 5 c 1 1 2 2 3 3 s 1 1 2 2 q 1 1 2 2 t 3 3 a 5 5 0 1 1 4 4 z",
        );
        let normalized = normalize(&path);
        let letters = commands(&normalized);
        assert!(
            letters.chars().all(|command| "MLCZ".contains(command)),
            "normalization left {letters:?}"
        );
        // The four drawing commands before the arc keep their one-to-one
        // shape; only the arc expands into several curves.
        assert!(
            letters.starts_with("MLLLCCCC"),
            "normalization left {letters:?}"
        );
        assert!(letters.ends_with('Z'));
        assert!(normalized.segments().iter().all(PathSegment::is_absolute));
    }

    #[test]
    fn normalization_is_idempotent() {
        let path = parse("M0 0 H10 V10 C1 1 2 2 3 3 S4 4 5 5 Q6 6 7 7 T8 8 A9 9 0 1 0 10 10 Z");
        let once = normalize(&path);
        assert_eq!(normalize(&once), once);
    }

    #[test]
    fn a_smooth_curve_reflects_only_after_another_cubic() {
        // Reflecting (10,5) about the current point (10,0) gives (10,-5).
        assert_eq!(
            normalize(&parse("M0 0 C0 5 10 5 10 0 S20 10 30 0")),
            normalize(&parse("M0 0 C0 5 10 5 10 0 C10 -5 20 10 30 0"))
        );
        // After a lineto there is nothing to reflect, so the first control
        // point is the current point itself.
        assert_eq!(
            normalize(&parse("M0 0 L10 0 S20 10 30 0")),
            normalize(&parse("M0 0 L10 0 C10 0 20 10 30 0"))
        );
        // A quadratic is not a cubic: `S` after `Q` does not reflect either.
        assert_eq!(
            normalize(&parse("M0 0 Q5 5 10 0 S20 10 30 0")),
            normalize(&parse("M0 0 Q5 5 10 0 C10 0 20 10 30 0"))
        );
        // Neither does `S` at the very start of a path, nor after a
        // closepath or a moveto.
        assert_eq!(
            normalize(&parse("M0 0 S20 10 30 0")),
            normalize(&parse("M0 0 C0 0 20 10 30 0"))
        );
        assert_eq!(
            normalize(&parse("M0 0 C0 5 10 5 10 0 Z S20 10 30 0")),
            normalize(&parse("M0 0 C0 5 10 5 10 0 Z C0 0 20 10 30 0"))
        );
        assert_eq!(
            normalize(&parse("M0 0 C0 5 10 5 10 0 M0 0 S20 10 30 0")),
            normalize(&parse("M0 0 C0 5 10 5 10 0 M0 0 C0 0 20 10 30 0"))
        );
    }

    #[test]
    fn a_smooth_quadratic_reflects_only_after_another_quadratic() {
        // Reflecting (10,10) about the current point (20,0) gives (30,-10).
        assert_eq!(
            normalize(&parse("M0 0 Q10 10 20 0 T40 0")),
            normalize(&parse("M0 0 Q10 10 20 0 Q30 -10 40 0"))
        );
        // A cubic is not a quadratic: `T` after `C` uses the current point.
        assert_eq!(
            normalize(&parse("M0 0 C0 5 10 5 10 0 T20 0")),
            normalize(&parse("M0 0 C0 5 10 5 10 0 Q10 0 20 0"))
        );
        assert_eq!(
            normalize(&parse("M0 0 L10 0 T20 0")),
            normalize(&parse("M0 0 L10 0 Q10 0 20 0"))
        );
        // A chain of `T`s keeps reflecting, alternating the control point.
        assert_eq!(
            normalize(&parse("M0 0 Q10 10 20 0 T40 0 T60 0")),
            normalize(&parse("M0 0 Q10 10 20 0 Q30 -10 40 0 Q50 10 60 0"))
        );
    }

    #[test]
    fn a_quadratic_is_elevated_rather_than_approximated() {
        let quadratic = parse("M0 0 Q1 3 2 0");
        let normalized = normalize(&quadratic);
        assert_eq!(commands(&normalized), "MC");
        let PathSegment::CurveTo {
            x1,
            y1,
            x2,
            y2,
            x,
            y,
            ..
        } = normalized.segments()[1]
        else {
            panic!("a quadratic normalizes to a cubic");
        };
        assert_close(x1, 2.0 / 3.0, 1e-12);
        assert_close(y1, 2.0, 1e-12);
        assert_close(x2, 4.0 / 3.0, 1e-12);
        assert_close(y2, 2.0, 1e-12);
        assert_eq!((x, y), (2.0, 0.0));
    }

    #[test]
    fn a_zero_radius_arc_normalizes_to_a_line() {
        assert_eq!(
            normalize(&parse("M0 0 A0 0 0 0 1 10 10")).to_string(),
            "M 0 0 L 10 10"
        );
        assert_eq!(
            normalize(&parse("M0 0 A5 0 0 1 1 10 10")).to_string(),
            "M 0 0 L 10 10"
        );
    }

    #[test]
    fn an_arc_between_coincident_endpoints_disappears() {
        let path = parse("M5 5 A10 10 0 1 1 5 5");
        assert_eq!(normalize(&path).to_string(), "M 5 5");
        assert_eq!(total_length(&path, 1e-9), 0.0);
        assert_eq!(
            bounding_box(&path),
            Some(Rect::from_point(Point::new(5.0, 5.0)))
        );
    }

    #[test]
    fn the_bounding_box_of_a_cubic_uses_its_extrema_not_its_hull() {
        let path = parse("M0 0 C0 3 4 3 4 0");
        let bounds = bounding_box(&path).expect("a non-empty path");
        assert_eq!(
            bounds,
            Rect::new(0.0, 0.0, 4.0, 2.25).expect("a valid rect")
        );
    }

    #[test]
    fn a_lone_moveto_bounds_its_own_point() {
        let bounds = bounding_box(&parse("M10 10")).expect("a moveto contributes its point");
        assert_eq!(bounds, Rect::from_point(Point::new(10.0, 10.0)));
        assert!(bounds.is_empty());
    }

    #[test]
    fn an_empty_path_has_no_geometry_at_all() {
        let empty = PathData::new();
        assert!(normalize(&empty).is_empty());
        assert_eq!(bounding_box(&empty), None);
        assert_eq!(total_length(&empty, 1e-9), 0.0);
        assert_eq!(point_at_length(&empty, 0.0, 1e-9), None);
        assert!(flatten(&empty, 1e-9).is_empty());
        assert!(transform_path(&empty, &Matrix::translate(1.0, 1.0)).is_empty());
    }

    #[test]
    fn the_bounding_box_of_an_arc_is_exact_rather_than_approximated() {
        // The lower half of the unit circle: the extreme point (0,-1) is not
        // an endpoint, and a Bézier approximation would miss it by 2.7e-4.
        let bounds = bounding_box(&parse("M-1 0 A1 1 0 0 1 1 0")).expect("a non-empty path");
        assert_rect_close(
            bounds,
            Rect::new(-1.0, -1.0, 2.0, 1.0).expect("a valid rect"),
            1e-12,
        );

        // A whole circle drawn as four arcs is bounded exactly too.
        let circle = bounding_box(&parse(CIRCLE)).expect("a non-empty path");
        assert_rect_close(
            circle,
            Rect::new(-1.0, -1.0, 2.0, 2.0).expect("a valid rect"),
            1e-12,
        );
    }

    #[test]
    fn a_tilted_elliptical_arc_carries_its_radii_and_rotation_through() {
        // Every other arc in these tests is circular and untilted, which
        // cannot tell correct `rx`/`ry`/rotation wiring from transposed
        // wiring: with rx == ry and phi == 0 the three are interchangeable.
        let tilted = parse("M0 0 A100 40 30 1 1 80 60");
        let bounds = bounding_box(&tilted).expect("a non-empty path");

        // The rotation really reaches the ellipse mathematics.
        let upright = bounding_box(&parse("M0 0 A100 40 0 1 1 80 60")).expect("a non-empty path");
        assert_ne!(bounds, upright);
        // The radii are not swapped on the way.
        let swapped = bounding_box(&parse("M0 0 A40 100 30 1 1 80 60")).expect("a non-empty path");
        assert_ne!(bounds, swapped);

        // And the box is the arc's own closed form rather than the box of
        // its Bézier approximation, which would be a fraction too small.
        let arc = ArcParams::new(
            Point::new(0.0, 0.0),
            100.0,
            40.0,
            30.0,
            true,
            true,
            Point::new(80.0, 60.0),
        );
        let ArcOutcome::Arc(center) = arc.to_center() else {
            panic!("a tilted ellipse through two distinct points is a real arc");
        };
        let expected = center
            .tight_bounds()
            .union(&Rect::from_points(arc.from, arc.to));
        assert_rect_close(bounds, expected, 1e-12);

        // Both flags reach the conversion too: the other three arcs between
        // the same endpoints are genuinely different curves.
        let small = parse("M0 0 A100 40 30 0 1 80 60");
        assert!(
            (total_length(&tilted, 1e-6) - total_length(&small, 1e-6)).abs() > 1.0,
            "the large arc and the small arc must not measure the same"
        );
        let other_sweep = bounding_box(&parse("M0 0 A100 40 30 1 0 80 60"));
        assert_ne!(other_sweep, Some(bounds));
    }

    #[test]
    fn a_closepath_contributes_no_new_bounds() {
        let open = bounding_box(&parse("M0 0 L10 0 L10 10")).expect("a non-empty path");
        let closed = bounding_box(&parse("M0 0 L10 0 L10 10 Z")).expect("a non-empty path");
        assert_eq!(open, closed);
    }

    #[test]
    fn the_length_of_a_circle_drawn_as_four_arcs_is_its_circumference() {
        let unit = total_length(&parse(CIRCLE), 1e-9);
        assert_close(unit, TAU, 1e-3);

        // The same shape at a hundred times the scale: the error is
        // proportional, so it is the *relative* error that is bounded.
        let big = parse(
            "M100 0 A100 100 0 0 1 0 100 A100 100 0 0 1 -100 0 \
             A100 100 0 0 1 0 -100 A100 100 0 0 1 100 0 Z",
        );
        let measured = total_length(&big, 1e-6);
        assert!(
            (measured - 100.0 * TAU).abs() / (100.0 * TAU) < 1e-3,
            "measured {measured} against {}",
            100.0 * TAU
        );
    }

    #[test]
    fn a_closepath_contributes_the_line_it_draws() {
        assert_close(total_length(&parse("M0 0 H1 V1 H0 Z"), 1e-9), 4.0, 1e-12);
        // Without the closepath the fourth side is not drawn.
        assert_close(total_length(&parse("M0 0 H1 V1 H0"), 1e-9), 3.0, 1e-12);
        // A moveto contributes nothing, however far it jumps.
        assert_close(
            total_length(&parse("M0 0 L3 4 M100 100 L103 104"), 1e-9),
            10.0,
            1e-12,
        );
    }

    #[test]
    fn a_point_halfway_along_a_straight_segment() {
        let path = parse("M0 0 L10 0");
        let (point, angle) = point_at_length(&path, 5.0, 1e-9).expect("the path draws something");
        assert_eq!(point, Point::new(5.0, 0.0));
        assert_eq!(angle, 0.0);
    }

    #[test]
    fn distances_outside_the_path_clamp_to_its_ends() {
        let path = parse("M0 0 L10 0 L10 10");
        let start = point_at_length(&path, -5.0, 1e-9).expect("the path draws something");
        assert_eq!(start.0, Point::new(0.0, 0.0));
        assert_eq!(start.1, 0.0);

        let end = point_at_length(&path, 1000.0, 1e-9).expect("the path draws something");
        assert_point_close(end.0, Point::new(10.0, 10.0), 1e-12);
        assert_close(end.1, FRAC_PI_2, 1e-12);

        // A distance that is not a number is treated as the start rather
        // than propagating a NaN into the answer.
        let nan = point_at_length(&path, f64::NAN, 1e-9).expect("the path draws something");
        assert_eq!(nan.0, Point::new(0.0, 0.0));
    }

    #[test]
    fn a_distance_on_a_segment_boundary_belongs_to_the_earlier_segment() {
        let path = parse("M0 0 L10 0 L10 10");
        let (point, angle) = point_at_length(&path, 10.0, 1e-9).expect("the path draws something");
        assert_point_close(point, Point::new(10.0, 0.0), 1e-12);
        // The first segment's end tangent, not the second's start tangent.
        assert_eq!(angle, 0.0);

        // One step further along, the second segment answers.
        let (_, angle) = point_at_length(&path, 10.5, 1e-9).expect("the path draws something");
        assert_close(angle, FRAC_PI_2, 1e-12);
    }

    #[test]
    fn a_trailing_moveto_never_answers_a_point_query() {
        let path = parse("M0 0 L10 0 M50 50");
        let (point, angle) =
            point_at_length(&path, 1000.0, 1e-9).expect("the path draws something");
        assert_point_close(point, Point::new(10.0, 0.0), 1e-12);
        assert_eq!(angle, 0.0);
    }

    #[test]
    fn a_path_that_draws_nothing_has_no_point_to_report() {
        // A lone moveto has a position — `bounding_box` reports it — but no
        // direction, so there is no point-at-length answer.
        let dot = parse("M10 10");
        assert!(bounding_box(&dot).is_some());
        assert_eq!(point_at_length(&dot, 0.0, 1e-9), None);
        // A zero-length lineto is the same case.
        assert_eq!(point_at_length(&parse("M0 0 L0 0"), 0.0, 1e-9), None);
        assert_eq!(point_at_length(&parse("M0 0 M5 5 M9 9"), 0.0, 1e-9), None);
    }

    #[test]
    fn points_along_a_circle_track_its_angle() {
        let path = parse(CIRCLE);
        let total = total_length(&path, 1e-9);
        for step in 0..8 {
            let fraction = f64::from(step) / 8.0;
            let (point, angle) =
                point_at_length(&path, total * fraction, 1e-9).expect("the circle draws something");
            let expected = fraction * TAU;
            // On the unit circle the point is at angle `expected` and the
            // tangent is a quarter turn ahead of it.
            assert_point_close(point, Point::new(expected.cos(), expected.sin()), 1e-3);
            let turn = (angle - (expected + FRAC_PI_2)).rem_euclid(TAU);
            assert!(
                turn < 1e-2 || (TAU - turn) < 1e-2,
                "tangent {angle} at fraction {fraction} should lead the radius by a quarter turn"
            );
        }
    }

    #[test]
    fn a_point_query_on_a_curve_lands_on_the_curve() {
        let path = parse("M0 0 C0 3 4 3 4 0");
        let total = total_length(&path, 1e-9);
        let curve = CubicBezier::new(
            Point::new(0.0, 0.0),
            Point::new(0.0, 3.0),
            Point::new(4.0, 3.0),
            Point::new(4.0, 0.0),
        );
        for step in 0..=10 {
            let fraction = f64::from(step) / 10.0;
            let (point, _) =
                point_at_length(&path, total * fraction, 1e-9).expect("the curve draws something");
            // The reported point is on the curve: some parameter reproduces
            // it closely. The curve is about 5.3 units long, so sampling it
            // 20000 times leaves gaps of 2.7e-4 and a point exactly on the
            // curve can still be half a gap from the nearest sample.
            let closest = (0..=20_000)
                .map(|index| {
                    let t = f64::from(index) / 20_000.0;
                    distance(curve.eval(t), point)
                })
                .fold(f64::INFINITY, f64::min);
            assert!(closest < 5e-4, "point {point:?} is {closest} off the curve");
        }
        // Distance along the curve grows monotonically with the query.
        let midpoint = point_at_length(&path, total / 2.0, 1e-9)
            .expect("half way")
            .0;
        assert_close(midpoint.x, 2.0, 1e-6);
        assert_close(midpoint.y, 2.25, 1e-6);
    }

    #[test]
    fn flattening_produces_one_polyline_per_subpath() {
        let polylines = flatten(&parse("M0 0 L1 0 L1 1 Z M5 5 L6 5"), 1e-6);
        assert_eq!(polylines.len(), 2);
        assert_eq!(
            polylines[0],
            vec![
                Point::new(0.0, 0.0),
                Point::new(1.0, 0.0),
                Point::new(1.0, 1.0),
                Point::new(0.0, 0.0),
            ]
        );
        assert_eq!(
            polylines[1],
            vec![Point::new(5.0, 5.0), Point::new(6.0, 5.0)]
        );
    }

    #[test]
    fn a_subpath_that_is_only_a_moveto_flattens_to_one_point() {
        let polylines = flatten(&parse("M0 0 L1 0 M5 5"), 1e-6);
        assert_eq!(polylines.len(), 2);
        assert_eq!(polylines[1], vec![Point::new(5.0, 5.0)]);
    }

    #[test]
    fn a_drawing_command_after_a_closepath_starts_a_new_polyline() {
        let polylines = flatten(&parse("M0 0 L1 0 Z L2 2"), 1e-6);
        assert_eq!(polylines.len(), 2);
        assert_eq!(
            polylines[0],
            vec![
                Point::new(0.0, 0.0),
                Point::new(1.0, 0.0),
                Point::new(0.0, 0.0)
            ]
        );
        // The new subpath starts at the closed subpath's own start point.
        assert_eq!(
            polylines[1],
            vec![Point::new(0.0, 0.0), Point::new(2.0, 2.0)]
        );
    }

    #[test]
    fn flattening_a_curve_stays_within_the_tolerance() {
        let path = parse("M0 0 C0 3 4 3 4 0");
        let coarse = flatten(&path, 1.0);
        let fine = flatten(&path, 1e-4);
        assert_eq!(coarse.len(), 1);
        assert!(fine[0].len() > coarse[0].len());
        assert_eq!(fine[0][0], Point::new(0.0, 0.0));
        assert_eq!(fine[0][fine[0].len() - 1], Point::new(4.0, 0.0));
        // The polyline's length converges on the measured arc length.
        let polyline_length: f64 = fine[0]
            .windows(2)
            .map(|pair| distance(pair[0], pair[1]))
            .sum();
        assert_close(polyline_length, total_length(&path, 1e-4), 1e-9);
    }

    #[test]
    fn flattening_a_circle_reproduces_its_circumference() {
        let polylines = flatten(&parse(CIRCLE), 1e-6);
        assert_eq!(polylines.len(), 1);
        let length: f64 = polylines[0]
            .windows(2)
            .map(|pair| distance(pair[0], pair[1]))
            .sum();
        assert_close(length, TAU, 1e-3);
        // The ring closes onto its own first point.
        let ring = &polylines[0];
        assert_point_close(ring[ring.len() - 1], ring[0], 1e-12);
    }

    #[test]
    fn a_rotation_carries_the_bounding_box_with_it() {
        let path = parse("M0 0 L10 0 L10 5 L0 5 Z");
        assert_eq!(
            bounding_box(&path),
            Rect::new(0.0, 0.0, 10.0, 5.0).ok_or(()).ok()
        );
        // rotate(90) maps (x, y) to (-y, x), so the box turns onto its side.
        let turned = transform_path(&path, &Matrix::rotate(90.0));
        let bounds = bounding_box(&turned).expect("a non-empty path");
        assert_rect_close(
            bounds,
            Rect::new(-5.0, 0.0, 5.0, 10.0).expect("a valid rect"),
            1e-12,
        );
    }

    #[test]
    fn transforming_preserves_lengths_under_a_rigid_motion() {
        let path = parse("M0 0 C0 3 4 3 4 0 L10 10");
        let before = total_length(&path, 1e-9);
        let matrix = Matrix::rotate_about(37.0, 2.0, -1.0).multiply(&Matrix::translate(5.0, 7.0));
        let after = total_length(&transform_path(&path, &matrix), 1e-9);
        assert_close(after, before, 1e-9);
    }

    #[test]
    fn transforming_scales_lengths_by_a_uniform_scale_factor() {
        let path = parse("M0 0 L3 4");
        let scaled = transform_path(&path, &Matrix::scale(2.0, 2.0));
        assert_close(total_length(&scaled, 1e-9), 10.0, 1e-12);
        assert_eq!(scaled.to_string(), "M 0 0 L 6 8");
    }

    #[test]
    fn transforming_an_arc_converts_it_to_curves_first() {
        let path = parse("M-1 0 A1 1 0 0 1 1 0");
        let turned = transform_path(&path, &Matrix::rotate(90.0));
        assert_eq!(commands(&turned), "MCC");
        // The lower half circle rotated a quarter turn is the right half
        // circle: x from 0 to 1, y from -1 to 1.
        let bounds = bounding_box(&turned).expect("a non-empty path");
        assert_rect_close(
            bounds,
            Rect::new(0.0, -1.0, 1.0, 2.0).expect("a valid rect"),
            1e-3,
        );
    }

    #[test]
    fn normalization_preserves_the_geometry_it_rewrites() {
        let cases = [
            "M0 0 H10 V10 Z",
            "M0 0 C0 5 10 5 10 0 S20 -5 20 0",
            "M0 0 Q10 10 20 0 T40 0",
            "m 10 20 l 5 5 h 5 v 5 z",
            CIRCLE,
        ];
        for input in cases {
            let path = parse(input);
            let normalized = normalize(&path);
            assert_close(
                total_length(&normalized, 1e-9),
                total_length(&path, 1e-9),
                1e-6,
            );
            let before = bounding_box(&path).expect("a non-empty path");
            let after = bounding_box(&normalized).expect("a non-empty path");
            // Arcs become curves, which costs the documented 2.7e-4 of a
            // radius; everything else is exact.
            assert_rect_close(after, before, 1e-3);
        }
    }

    #[test]
    fn a_path_that_starts_without_a_moveto_still_has_geometry() {
        // `PathData::parse` rejects such a path, but one can be built
        // directly, and the operations must not panic on it.
        let path = PathData::from_segments(vec![PathSegment::LineTo {
            abs: true,
            x: 10.0,
            y: 0.0,
        }]);
        // Normalization makes the implicit start point explicit, so the
        // output is valid path data even though the input was not.
        assert_eq!(normalize(&path).to_string(), "M 0 0 L 10 0");
        assert_eq!(
            bounding_box(&path),
            Rect::new(0.0, 0.0, 10.0, 0.0).ok_or(()).ok()
        );
        assert_close(total_length(&path, 1e-9), 10.0, 1e-12);
        assert_eq!(flatten(&path, 1e-9).len(), 1);
        // And the implicit start point is transformed along with everything
        // else, rather than staying behind at the origin.
        assert_eq!(
            transform_path(&path, &Matrix::translate(5.0, 5.0)).to_string(),
            "M 5 5 L 15 5"
        );
    }

    #[test]
    fn a_degenerate_curve_still_reports_a_tangent() {
        // The first control point coincides with the start point, so the
        // derivative vanishes there and the secant fallback takes over.
        let curve = CubicBezier::new(
            Point::new(0.0, 0.0),
            Point::new(0.0, 0.0),
            Point::new(10.0, 0.0),
            Point::new(10.0, 0.0),
        );
        let angle = cubic_tangent(&curve, 0.0);
        assert!(angle.is_finite());
        assert_close(angle, 0.0, 1e-9);

        // A curve collapsed to a single point has no direction anywhere; the
        // chord fallback then reports the angle of a zero vector, which is
        // zero rather than a NaN.
        let point = Point::new(4.0, 4.0);
        let collapsed = CubicBezier::new(point, point, point, point);
        assert_eq!(cubic_tangent(&collapsed, 0.5), 0.0);
    }

    #[test]
    fn tangents_follow_the_direction_of_travel() {
        let cases = [
            ("M0 0 L10 0", 0.0),
            ("M0 0 L0 10", FRAC_PI_2),
            ("M0 0 L-10 0", PI),
            ("M0 0 L0 -10", -FRAC_PI_2),
        ];
        for (input, expected) in cases {
            let (_, angle) =
                point_at_length(&parse(input), 5.0, 1e-9).expect("the path draws something");
            assert_close(angle, expected, 1e-12);
        }
    }
}
