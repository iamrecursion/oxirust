//! Quadratic and cubic Bézier curves.
//!
//! [`QuadBezier`] and [`CubicBezier`] are the two curve families SVG path
//! data names directly (`Q`/`T` and `C`/`S`) and — once elliptical arcs have
//! been converted by [`crate::geometry::arc`] — the only curve families the
//! rest of the geometry toolkit has to reason about.
//!
//! Every operation here is analytic where an analytic answer exists
//! ([`CubicBezier::eval`], [`CubicBezier::derivative_at`],
//! [`CubicBezier::split_at`], [`CubicBezier::extrema`],
//! [`CubicBezier::tight_bounds`]) and adaptive where none does
//! ([`CubicBezier::arc_length`], [`CubicBezier::flatten`]): a Bézier curve's
//! arc length has no closed form in general, so it is measured by
//! subdividing until each piece is flat to a caller-supplied tolerance and
//! summing chord lengths.
//!
//! # Flatness and the error model
//!
//! Subdivision stops when a piece is *flat*: when the greatest distance
//! between the curve and the straight chord joining its endpoints is at most
//! `tolerance`. That distance has a closed-form bound. Writing
//! `L(t) = (1-t)*p0 + t*p3` for the chord and `u = 3*p1 - 2*p0 - p3`,
//! `v = 3*p2 - p0 - 2*p3`, a cubic satisfies
//!
//! ```text
//! B(t) - L(t) = t*(1-t) * [ (1-t)*u + t*v ]
//! ```
//!
//! so, because `(1-t)*u + t*v` is a convex combination of `u` and `v` and
//! `t*(1-t) <= 1/4`, the deviation is bounded by `max(|u|, |v|) / 4`. The
//! bound is applied per axis, which is why the flatness predicate reads
//! `max(u.x², v.x²) + max(u.y², v.y²) <= 16 * tolerance²`. The quadratic case
//! is the same derivation one degree down: `B(t) - L(t) = t*(1-t) * w` with
//! `w = 2*p1 - p0 - p2`, whose maximum is `|w| / 4`, attained at `t = 1/2`.
//! Degree elevation carries `w` into *both* `u` and `v`, so flattening a
//! quadratic through [`QuadBezier::to_cubic`] applies exactly the quadratic
//! criterion — the two tests coincide rather than merely approximate one
//! another.
//!
//! Two consequences are worth stating plainly, since callers depend on them:
//!
//! * [`arc_length`](CubicBezier::arc_length) sums chords, and a chord is
//!   never longer than the arc it subtends, so the result is always a slight
//!   **under**estimate. For a piece of chord length `c` that is flat to
//!   within `h`, the deficit is of order `h² / c` — far below `tolerance`
//!   itself, and shrinking quadratically as `tolerance` does.
//! * Subdivision is capped at [`MAX_SUBDIVISION_DEPTH`] halvings per curve,
//!   so a pathological curve (an enormous coordinate range, a tolerance of
//!   zero) terminates with a merely coarse answer instead of recursing
//!   forever. A `tolerance` that is not finite and positive is replaced by
//!   [`DEFAULT_TOLERANCE`], and a piece with a non-finite control point —
//!   path data may name `1e999`, which saturates to an infinity — is
//!   accepted as-is rather than subdivided, since no amount of subdivision
//!   would ever satisfy the tolerance.

use crate::geometry::primitives::{Point, Rect};

/// The flattening tolerance used when a caller passes one that is not finite
/// and positive.
///
/// A tenth of a user-space unit is the conventional default for screen-scale
/// SVG content: well below one device pixel at typical scales, while keeping
/// the number of emitted line segments small.
pub const DEFAULT_TOLERANCE: f64 = 0.1;

/// The greatest number of times [`arc_length`](CubicBezier::arc_length) and
/// [`flatten`](CubicBezier::flatten) will halve a curve before accepting a
/// piece as flat.
///
/// A depth of 24 allows up to 2^24 pieces, which no realistic tolerance and
/// coordinate range together require; the cap exists only so that a
/// tolerance too small to ever be met — `1e-300` is finite and positive, and
/// so is honoured rather than replaced — cannot cause unbounded recursion.
/// Non-finite control points are handled before this, by stopping outright.
pub const MAX_SUBDIVISION_DEPTH: u32 = 24;

/// Replaces a tolerance that is not finite and positive with
/// [`DEFAULT_TOLERANCE`].
fn effective_tolerance(tolerance: f64) -> f64 {
    if tolerance.is_finite() && tolerance > 0.0 {
        tolerance
    } else {
        DEFAULT_TOLERANCE
    }
}

/// Linear interpolation between two points, at parameter `t`.
fn lerp(a: Point, b: Point, t: f64) -> Point {
    Point::new(a.x + (b.x - a.x) * t, a.y + (b.y - a.y) * t)
}

/// The Euclidean distance between two points.
fn distance(a: Point, b: Point) -> f64 {
    (b.x - a.x).hypot(b.y - a.y)
}

/// The real roots of `a*t² + b*t + c`, with `None` for a root that does not
/// exist.
///
/// Uses the numerically stable form (recovering the cancellation-prone root
/// from the product of the roots) rather than the textbook quadratic
/// formula, which loses precision when `4*a*c` is small beside `b²`. A
/// leading coefficient of exactly zero degenerates to the linear case.
fn solve_quadratic(a: f64, b: f64, c: f64) -> (Option<f64>, Option<f64>) {
    if a == 0.0 {
        if b == 0.0 {
            return (None, None);
        }
        return (Some(-c / b), None);
    }
    let discriminant = b * b - 4.0 * a * c;
    if discriminant < 0.0 {
        return (None, None);
    }
    let root = discriminant.sqrt();
    // `q` takes the larger-magnitude numerator, so `q / a` never suffers
    // cancellation; the other root follows from the product of the roots,
    // `c / a`, as `c / q`.
    let q = -0.5 * (b + if b >= 0.0 { root } else { -root });
    let first = q / a;
    let second = if q == 0.0 { None } else { Some(c / q) };
    (Some(first), second)
}

/// Keeps `t` only when it is a usable interior parameter: finite and
/// strictly inside `(0, 1)`.
fn interior_root(t: Option<f64>) -> Option<f64> {
    t.filter(|t| t.is_finite() && *t > 0.0 && *t < 1.0)
}

/// Sorts `values` ascending and drops duplicates, treating values within a
/// hair of one another as equal.
fn sort_dedup(values: &mut Vec<f64>) {
    values.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    values.dedup_by(|a, b| (*a - *b).abs() < 1e-12);
}

/// A quadratic Bézier curve: a start point, one control point, and an end
/// point.
///
/// This is the curve SVG path data's `Q` and `T` commands describe. Its
/// parametric form is `B(t) = (1-t)²*p0 + 2*(1-t)*t*p1 + t²*p2` over
/// `t` in `[0, 1]`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct QuadBezier {
    /// The start point, `B(0)`.
    pub p0: Point,
    /// The control point. The curve does not pass through it unless the
    /// curve is degenerate; it only pulls the curve towards itself.
    pub p1: Point,
    /// The end point, `B(1)`.
    pub p2: Point,
}

impl QuadBezier {
    /// Builds a quadratic curve from its start point, control point, and end
    /// point.
    #[must_use]
    pub const fn new(p0: Point, p1: Point, p2: Point) -> Self {
        Self { p0, p1, p2 }
    }

    /// The point on the curve at parameter `t`.
    ///
    /// `t` is not clamped: a value outside `[0, 1]` evaluates the curve's
    /// analytic extension beyond its endpoints, which is occasionally useful
    /// and never an error.
    #[must_use]
    pub fn eval(&self, t: f64) -> Point {
        // De Casteljau's algorithm rather than the expanded polynomial: the
        // same cost, and markedly better conditioned near t = 1.
        let a = lerp(self.p0, self.p1, t);
        let b = lerp(self.p1, self.p2, t);
        lerp(a, b, t)
    }

    /// The first derivative `B'(t)`: a vector pointing along the curve's
    /// tangent, whose magnitude is the parametric speed.
    ///
    /// The result is a direction, not a point on the curve; it is returned
    /// as a [`Point`] only because this module has no separate vector type.
    /// It is the zero vector at a cusp, where no tangent direction exists.
    #[must_use]
    pub fn derivative_at(&self, t: f64) -> Point {
        // B'(t) = 2*[(1-t)*(p1-p0) + t*(p2-p1)]
        let ax = self.p1.x - self.p0.x;
        let ay = self.p1.y - self.p0.y;
        let bx = self.p2.x - self.p1.x;
        let by = self.p2.y - self.p1.y;
        Point::new(
            2.0 * ((1.0 - t) * ax + t * bx),
            2.0 * ((1.0 - t) * ay + t * by),
        )
    }

    /// Splits the curve at parameter `t` into the piece before `t` and the
    /// piece after it.
    ///
    /// Both pieces are reparameterized over `[0, 1]`; together they trace
    /// exactly the geometry of the original curve.
    #[must_use]
    pub fn split_at(&self, t: f64) -> (Self, Self) {
        let a = lerp(self.p0, self.p1, t);
        let b = lerp(self.p1, self.p2, t);
        let mid = lerp(a, b, t);
        (Self::new(self.p0, a, mid), Self::new(mid, b, self.p2))
    }

    /// The piece of the curve between parameters `t0` and `t1`,
    /// reparameterized over `[0, 1]`.
    ///
    /// Intended for `0 <= t0 <= t1 <= 1`. Any other ordering still returns a
    /// curve — the arithmetic is the same de Casteljau construction applied
    /// to the analytic extension — but the geometry it traces is then no
    /// longer a sub-arc of this curve.
    #[must_use]
    pub fn subsegment(&self, t0: f64, t1: f64) -> Self {
        let after = self.split_at(t0).1;
        let remaining = 1.0 - t0;
        let rescaled = if remaining == 0.0 {
            0.0
        } else {
            (t1 - t0) / remaining
        };
        after.split_at(rescaled).0
    }

    /// The identical curve expressed as a cubic, by exact degree elevation.
    ///
    /// The elevated control points are `p0 + (2/3)*(p1 - p0)` and
    /// `p2 + (2/3)*(p1 - p2)`. The conversion is exact rather than an
    /// approximation: the two curves agree at every parameter value, as the
    /// example checks.
    ///
    /// ```
    /// use oxixml_svg::geometry::bezier::QuadBezier;
    /// use oxixml_svg::geometry::primitives::Point;
    ///
    /// let quad = QuadBezier::new(
    ///     Point::new(0.0, 0.0),
    ///     Point::new(1.0, 3.0),
    ///     Point::new(2.0, 0.0),
    /// );
    /// let cubic = quad.to_cubic();
    /// for step in 0..=10 {
    ///     let t = f64::from(step) / 10.0;
    ///     assert!((quad.eval(t).x - cubic.eval(t).x).abs() < 1e-12);
    ///     assert!((quad.eval(t).y - cubic.eval(t).y).abs() < 1e-12);
    /// }
    /// ```
    #[must_use]
    pub fn to_cubic(&self) -> CubicBezier {
        const TWO_THIRDS: f64 = 2.0 / 3.0;
        CubicBezier::new(
            self.p0,
            Point::new(
                self.p0.x + TWO_THIRDS * (self.p1.x - self.p0.x),
                self.p0.y + TWO_THIRDS * (self.p1.y - self.p0.y),
            ),
            Point::new(
                self.p2.x + TWO_THIRDS * (self.p1.x - self.p2.x),
                self.p2.y + TWO_THIRDS * (self.p1.y - self.p2.y),
            ),
            self.p2,
        )
    }

    /// The interior parameter values at which the curve's x or y coordinate
    /// reaches a local extremum, sorted ascending and deduplicated.
    ///
    /// A quadratic's derivative is linear per axis, so there is at most one
    /// root per axis: `t = (p0 - p1) / (p0 - 2*p1 + p2)`. An axis whose
    /// denominator vanishes (a straight, uniformly parameterized one)
    /// contributes nothing rather than an infinity or a NaN. Roots at
    /// exactly `0` or `1` are omitted, since the endpoints are always
    /// considered separately by the callers that need them.
    #[must_use]
    pub fn extrema(&self) -> Vec<f64> {
        let mut roots = Vec::new();
        for (a, b, c) in [
            (self.p0.x, self.p1.x, self.p2.x),
            (self.p0.y, self.p1.y, self.p2.y),
        ] {
            let denominator = a - 2.0 * b + c;
            let root = if denominator == 0.0 {
                None
            } else {
                Some((a - b) / denominator)
            };
            if let Some(t) = interior_root(root) {
                roots.push(t);
            }
        }
        sort_dedup(&mut roots);
        roots
    }

    /// The smallest axis-aligned rectangle containing the curve.
    ///
    /// "Tight" as opposed to the control-point hull, which encloses the
    /// curve but is usually far larger: this box is found from the endpoints
    /// together with the extrema reported by [`QuadBezier::extrema`], so it
    /// touches the curve on every side it can.
    #[must_use]
    pub fn tight_bounds(&self) -> Rect {
        let mut bounds = Rect::from_points(self.p0, self.p2);
        for t in self.extrema() {
            bounds = bounds.expand_to_include(self.eval(t));
        }
        bounds
    }

    /// The curve's arc length, measured to the given `tolerance`.
    ///
    /// See the [module documentation](self) for the error model: the answer
    /// is a slight underestimate that converges quadratically as `tolerance`
    /// shrinks.
    #[must_use]
    pub fn arc_length(&self, tolerance: f64) -> f64 {
        self.to_cubic().arc_length(tolerance)
    }

    /// Approximates the curve by a polyline, calling `emit` with the end
    /// point of each line segment.
    ///
    /// The start point `p0` is **not** emitted — the caller already has it —
    /// and the last emitted point is exactly `p2`, so a polyline is built by
    /// pushing `p0` first and then everything `emit` receives. Every emitted
    /// segment lies within `tolerance` of the curve.
    pub fn flatten(&self, tolerance: f64, emit: &mut impl FnMut(Point)) {
        self.to_cubic().flatten(tolerance, emit);
    }

    /// [`QuadBezier::flatten`], additionally reporting the curve parameter
    /// each emitted point was taken at.
    ///
    /// Degree elevation preserves the parameterization, so the parameters
    /// reported here are this quadratic's own: `self.eval(t)` reproduces the
    /// emitted point. They increase monotonically and the last is exactly
    /// `1.0`.
    pub fn flatten_with_params(&self, tolerance: f64, emit: &mut impl FnMut(f64, Point)) {
        self.to_cubic().flatten_with_params(tolerance, emit);
    }

    /// True when every coordinate of every control point is finite.
    #[must_use]
    pub fn is_finite(&self) -> bool {
        [self.p0, self.p1, self.p2]
            .iter()
            .all(|p| p.x.is_finite() && p.y.is_finite())
    }
}

/// A cubic Bézier curve: a start point, two control points, and an end
/// point.
///
/// This is the curve SVG path data's `C` and `S` commands describe, and the
/// curve every other curved primitive in this crate is ultimately reduced
/// to — quadratics by exact degree elevation ([`QuadBezier::to_cubic`]) and
/// elliptical arcs by the standard approximation
/// ([`crate::geometry::arc`]). Its parametric form is
/// `B(t) = (1-t)³*p0 + 3*(1-t)²*t*p1 + 3*(1-t)*t²*p2 + t³*p3` over
/// `t` in `[0, 1]`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CubicBezier {
    /// The start point, `B(0)`.
    pub p0: Point,
    /// The first control point, which the tangent at `p0` points towards.
    pub p1: Point,
    /// The second control point, which the tangent at `p3` points away from.
    pub p2: Point,
    /// The end point, `B(1)`.
    pub p3: Point,
}

impl CubicBezier {
    /// Builds a cubic curve from its start point, two control points, and
    /// end point.
    #[must_use]
    pub const fn new(p0: Point, p1: Point, p2: Point, p3: Point) -> Self {
        Self { p0, p1, p2, p3 }
    }

    /// The cubic that traces the straight line from `from` to `to`.
    ///
    /// The control points are placed one third and two thirds of the way
    /// along, which is the only placement that keeps the parameterization
    /// uniform as well as the geometry straight — so `eval(t)` matches the
    /// line's own `lerp(from, to, t)` at every `t`. Used when a caller needs
    /// a single curve type for a run of mixed segments, for example after an
    /// elliptical arc has degenerated into a line.
    #[must_use]
    pub fn from_line(from: Point, to: Point) -> Self {
        Self::new(
            from,
            lerp(from, to, 1.0 / 3.0),
            lerp(from, to, 2.0 / 3.0),
            to,
        )
    }

    /// The identical curve, as a cubic, of a quadratic; see
    /// [`QuadBezier::to_cubic`].
    #[must_use]
    pub fn from_quad(quad: &QuadBezier) -> Self {
        quad.to_cubic()
    }

    /// The point on the curve at parameter `t`.
    ///
    /// `t` is not clamped: a value outside `[0, 1]` evaluates the curve's
    /// analytic extension beyond its endpoints, which is occasionally useful
    /// and never an error.
    #[must_use]
    pub fn eval(&self, t: f64) -> Point {
        let a = lerp(self.p0, self.p1, t);
        let b = lerp(self.p1, self.p2, t);
        let c = lerp(self.p2, self.p3, t);
        let d = lerp(a, b, t);
        let e = lerp(b, c, t);
        lerp(d, e, t)
    }

    /// The first derivative `B'(t)`: a vector pointing along the curve's
    /// tangent, whose magnitude is the parametric speed.
    ///
    /// The result is a direction, not a point on the curve. It is the zero
    /// vector at a cusp and wherever a control point coincides with the
    /// endpoint next to it at `t = 0` or `t = 1`, so callers that need a
    /// tangent *direction* must check for it rather than normalizing blindly.
    #[must_use]
    pub fn derivative_at(&self, t: f64) -> Point {
        // B'(t) = 3*[(1-t)²*(p1-p0) + 2*(1-t)*t*(p2-p1) + t²*(p3-p2)]
        let one_minus = 1.0 - t;
        let w0 = 3.0 * one_minus * one_minus;
        let w1 = 6.0 * one_minus * t;
        let w2 = 3.0 * t * t;
        Point::new(
            w0 * (self.p1.x - self.p0.x)
                + w1 * (self.p2.x - self.p1.x)
                + w2 * (self.p3.x - self.p2.x),
            w0 * (self.p1.y - self.p0.y)
                + w1 * (self.p2.y - self.p1.y)
                + w2 * (self.p3.y - self.p2.y),
        )
    }

    /// Splits the curve at parameter `t` into the piece before `t` and the
    /// piece after it.
    ///
    /// Both pieces are reparameterized over `[0, 1]`; together they trace
    /// exactly the geometry of the original curve. This is de Casteljau's
    /// construction, and it is what the adaptive routines below subdivide
    /// with.
    #[must_use]
    pub fn split_at(&self, t: f64) -> (Self, Self) {
        let a = lerp(self.p0, self.p1, t);
        let b = lerp(self.p1, self.p2, t);
        let c = lerp(self.p2, self.p3, t);
        let d = lerp(a, b, t);
        let e = lerp(b, c, t);
        let mid = lerp(d, e, t);
        (Self::new(self.p0, a, d, mid), Self::new(mid, e, c, self.p3))
    }

    /// The piece of the curve between parameters `t0` and `t1`,
    /// reparameterized over `[0, 1]`.
    ///
    /// Intended for `0 <= t0 <= t1 <= 1`; see
    /// [`QuadBezier::subsegment`] for what other orderings mean.
    #[must_use]
    pub fn subsegment(&self, t0: f64, t1: f64) -> Self {
        let after = self.split_at(t0).1;
        let remaining = 1.0 - t0;
        let rescaled = if remaining == 0.0 {
            0.0
        } else {
            (t1 - t0) / remaining
        };
        after.split_at(rescaled).0
    }

    /// The interior parameter values at which the curve's x or y coordinate
    /// reaches a local extremum, sorted ascending and deduplicated.
    ///
    /// A cubic's derivative is quadratic per axis, so each axis contributes
    /// up to two roots of `a*t² + b*t + c` with
    /// `a = 3*(-p0 + 3*p1 - 3*p2 + p3)`, `b = 6*(p0 - 2*p1 + p2)` and
    /// `c = 3*(p1 - p0)`. Degenerate axes are handled without producing a
    /// NaN: a vanishing `a` falls back to the linear root, a vanishing `a`
    /// and `b` together contribute nothing, and a negative discriminant
    /// (no real extremum, the common case for a gently curving axis) is
    /// simply empty. Roots outside the open interval `(0, 1)` are dropped,
    /// since the endpoints are always considered separately.
    #[must_use]
    pub fn extrema(&self) -> Vec<f64> {
        let mut roots = Vec::new();
        for (p0, p1, p2, p3) in [
            (self.p0.x, self.p1.x, self.p2.x, self.p3.x),
            (self.p0.y, self.p1.y, self.p2.y, self.p3.y),
        ] {
            let a = 3.0 * (-p0 + 3.0 * p1 - 3.0 * p2 + p3);
            let b = 6.0 * (p0 - 2.0 * p1 + p2);
            let c = 3.0 * (p1 - p0);
            let (first, second) = solve_quadratic(a, b, c);
            for root in [interior_root(first), interior_root(second)]
                .into_iter()
                .flatten()
            {
                roots.push(root);
            }
        }
        sort_dedup(&mut roots);
        roots
    }

    /// The smallest axis-aligned rectangle containing the curve.
    ///
    /// Computed from the endpoints together with the extrema reported by
    /// [`CubicBezier::extrema`], so it is exact — it touches the curve on
    /// every side it can, unlike the control-point hull.
    ///
    /// ```
    /// use oxixml_svg::geometry::bezier::CubicBezier;
    /// use oxixml_svg::geometry::primitives::Point;
    ///
    /// // A symmetric arch: both control points are pulled 3 units up, but
    /// // the curve itself only reaches 3/4 of that.
    /// let curve = CubicBezier::new(
    ///     Point::new(0.0, 0.0),
    ///     Point::new(0.0, 3.0),
    ///     Point::new(4.0, 3.0),
    ///     Point::new(4.0, 0.0),
    /// );
    /// let bounds = curve.tight_bounds();
    /// assert!((bounds.height - 2.25).abs() < 1e-12);
    /// assert_eq!(bounds.width, 4.0);
    /// ```
    #[must_use]
    pub fn tight_bounds(&self) -> Rect {
        let mut bounds = Rect::from_points(self.p0, self.p3);
        for t in self.extrema() {
            bounds = bounds.expand_to_include(self.eval(t));
        }
        bounds
    }

    /// True when the curve is flat to within `tolerance`: when no point of
    /// it lies further than `tolerance` from the straight chord `p0`–`p3`.
    ///
    /// This is the predicate the adaptive routines subdivide on; see the
    /// [module documentation](self) for its derivation. The answer is not
    /// meaningful for a curve with a non-finite control point — a NaN
    /// propagates into a comparison that is false whichever way it is
    /// written — which is why the adaptive routines test
    /// [`CubicBezier::is_finite`] separately rather than relying on this.
    #[must_use]
    pub fn is_flat(&self, tolerance: f64) -> bool {
        let ux = 3.0 * self.p1.x - 2.0 * self.p0.x - self.p3.x;
        let uy = 3.0 * self.p1.y - 2.0 * self.p0.y - self.p3.y;
        let vx = 3.0 * self.p2.x - self.p0.x - 2.0 * self.p3.x;
        let vy = 3.0 * self.p2.y - self.p0.y - 2.0 * self.p3.y;
        let x = (ux * ux).max(vx * vx);
        let y = (uy * uy).max(vy * vy);
        x + y <= 16.0 * tolerance * tolerance
    }

    /// The curve's arc length, measured to the given `tolerance`.
    ///
    /// See the [module documentation](self) for the error model: the answer
    /// is a slight underestimate that converges quadratically as `tolerance`
    /// shrinks. A straight cubic is measured exactly, since its chord is the
    /// curve.
    ///
    /// ```
    /// use oxixml_svg::geometry::bezier::CubicBezier;
    /// use oxixml_svg::geometry::primitives::Point;
    ///
    /// let straight = CubicBezier::from_line(Point::new(0.0, 0.0), Point::new(3.0, 4.0));
    /// assert!((straight.arc_length(1e-6) - 5.0).abs() < 1e-12);
    /// ```
    #[must_use]
    pub fn arc_length(&self, tolerance: f64) -> f64 {
        let mut total = 0.0;
        let mut previous = self.p0;
        self.flatten_with_params(tolerance, &mut |_, point| {
            total += distance(previous, point);
            previous = point;
        });
        total
    }

    /// Approximates the curve by a polyline, calling `emit` with the end
    /// point of each line segment.
    ///
    /// The start point `p0` is **not** emitted — the caller already has it —
    /// and the last emitted point is exactly `p3`, so a polyline is built by
    /// pushing `p0` first and then everything `emit` receives. Every emitted
    /// segment lies within `tolerance` of the curve.
    pub fn flatten(&self, tolerance: f64, emit: &mut impl FnMut(Point)) {
        self.flatten_with_params(tolerance, &mut |_, point| emit(point));
    }

    /// [`CubicBezier::flatten`], additionally reporting the curve parameter
    /// each emitted point was taken at.
    ///
    /// The parameters increase monotonically and the last is exactly `1.0`.
    /// This is what makes a cumulative-length table invertible: pair each
    /// emitted point with the running chord length, and a target length can
    /// be turned back into a curve parameter — which is precisely how
    /// [`crate::geometry::path_ops::point_at_length`] works.
    pub fn flatten_with_params(&self, tolerance: f64, emit: &mut impl FnMut(f64, Point)) {
        let tolerance = effective_tolerance(tolerance);
        flatten_recursive(self, 0.0, 1.0, tolerance, 0, emit);
        // The recursion emits the end point of every accepted piece, so the
        // final point is already exactly `p3` when the last piece is
        // accepted; emitting it here as well would duplicate it.
    }

    /// True when every coordinate of every control point is finite.
    #[must_use]
    pub fn is_finite(&self) -> bool {
        [self.p0, self.p1, self.p2, self.p3]
            .iter()
            .all(|p| p.x.is_finite() && p.y.is_finite())
    }
}

/// Emits the end point of every flat piece of `curve`, whose parameter range
/// within the original curve is `[t0, t1]`.
///
/// Three conditions stop the recursion: the piece is flat, the depth cap is
/// reached, or the piece has a non-finite control point. The last is what
/// keeps an overflowed coordinate (path data may legitimately name `1e999`,
/// which saturates to an infinity) from costing 2^[`MAX_SUBDIVISION_DEPTH`]
/// emitted points on a curve that can never satisfy any tolerance.
fn flatten_recursive(
    curve: &CubicBezier,
    t0: f64,
    t1: f64,
    tolerance: f64,
    depth: u32,
    emit: &mut impl FnMut(f64, Point),
) {
    if depth >= MAX_SUBDIVISION_DEPTH || !curve.is_finite() || curve.is_flat(tolerance) {
        emit(t1, curve.p3);
        return;
    }
    let (left, right) = curve.split_at(0.5);
    let middle = (t0 + t1) / 2.0;
    flatten_recursive(&left, t0, middle, tolerance, depth + 1, emit);
    flatten_recursive(&right, middle, t1, tolerance, depth + 1, emit);
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The control-point distance that turns a cubic into a quarter circle:
    /// `(4/3) * tan(90° / 4)`. Re-derived rather than copied so the test
    /// checks arithmetic rather than a transcribed constant.
    fn quarter_circle_k() -> f64 {
        4.0 / 3.0 * (std::f64::consts::FRAC_PI_8).tan()
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

    fn sample_cubic() -> CubicBezier {
        CubicBezier::new(
            Point::new(0.0, 0.0),
            Point::new(3.0, 5.0),
            Point::new(-2.0, 4.0),
            Point::new(1.0, 1.0),
        )
    }

    #[test]
    fn cubic_evaluates_to_its_endpoints_at_zero_and_one() {
        let curve = sample_cubic();
        assert_eq!(curve.eval(0.0), curve.p0);
        assert_eq!(curve.eval(1.0), curve.p3);
    }

    #[test]
    fn cubic_derivative_at_the_endpoints_is_three_times_the_end_leg() {
        let curve = sample_cubic();
        let start = curve.derivative_at(0.0);
        assert_point_close(
            start,
            Point::new(
                3.0 * (curve.p1.x - curve.p0.x),
                3.0 * (curve.p1.y - curve.p0.y),
            ),
            1e-12,
        );
        let end = curve.derivative_at(1.0);
        assert_point_close(
            end,
            Point::new(
                3.0 * (curve.p3.x - curve.p2.x),
                3.0 * (curve.p3.y - curve.p2.y),
            ),
            1e-12,
        );
    }

    #[test]
    fn cubic_derivative_matches_a_finite_difference() {
        let curve = sample_cubic();
        let h = 1e-6;
        for step in 1..10 {
            let t = f64::from(step) / 10.0;
            let before = curve.eval(t - h);
            let after = curve.eval(t + h);
            let numeric = Point::new(
                (after.x - before.x) / (2.0 * h),
                (after.y - before.y) / (2.0 * h),
            );
            assert_point_close(curve.derivative_at(t), numeric, 1e-6);
        }
    }

    #[test]
    fn splitting_a_cubic_preserves_its_geometry() {
        let curve = sample_cubic();
        let (left, right) = curve.split_at(0.3);
        assert_eq!(left.p0, curve.p0);
        assert_eq!(right.p3, curve.p3);
        assert_eq!(left.p3, right.p0);
        for step in 0..=10 {
            let s = f64::from(step) / 10.0;
            assert_point_close(left.eval(s), curve.eval(0.3 * s), 1e-12);
            assert_point_close(right.eval(s), curve.eval(0.3 + 0.7 * s), 1e-12);
        }
    }

    #[test]
    fn a_cubic_subsegment_traces_the_matching_stretch_of_the_curve() {
        let curve = sample_cubic();
        let piece = curve.subsegment(0.25, 0.75);
        for step in 0..=10 {
            let s = f64::from(step) / 10.0;
            assert_point_close(piece.eval(s), curve.eval(0.25 + 0.5 * s), 1e-12);
        }
    }

    #[test]
    fn a_quad_subsegment_traces_the_matching_stretch_of_the_curve() {
        let quad = QuadBezier::new(
            Point::new(0.0, 0.0),
            Point::new(1.0, 3.0),
            Point::new(4.0, 0.0),
        );
        let piece = quad.subsegment(0.2, 0.9);
        for step in 0..=10 {
            let s = f64::from(step) / 10.0;
            assert_point_close(piece.eval(s), quad.eval(0.2 + 0.7 * s), 1e-12);
        }
    }

    #[test]
    fn degree_elevation_reproduces_the_quadratic_exactly() {
        let quad = QuadBezier::new(
            Point::new(-1.0, 2.0),
            Point::new(4.5, -3.25),
            Point::new(7.0, 8.0),
        );
        let cubic = quad.to_cubic();
        assert_eq!(cubic.p0, quad.p0);
        assert_eq!(cubic.p3, quad.p2);
        for step in 0..=20 {
            let t = f64::from(step) / 20.0;
            assert_point_close(cubic.eval(t), quad.eval(t), 1e-12);
        }
        // The derivatives agree too, which is what lets the quadratic's
        // flattening delegate to the cubic without changing the tangents a
        // caller reads back.
        for step in 0..=20 {
            let t = f64::from(step) / 20.0;
            assert_point_close(cubic.derivative_at(t), quad.derivative_at(t), 1e-12);
        }
        assert_eq!(CubicBezier::from_quad(&quad), cubic);
    }

    #[test]
    fn cubic_extrema_are_the_analytic_roots_of_the_derivative() {
        // x controls 0, 3, -2, 1 give 48t² - 48t + 9 = 0, i.e. t = 1/4, 3/4.
        let curve = CubicBezier::new(
            Point::new(0.0, 0.0),
            Point::new(3.0, 0.0),
            Point::new(-2.0, 0.0),
            Point::new(1.0, 0.0),
        );
        let extrema = curve.extrema();
        assert_eq!(extrema.len(), 2);
        assert_close(extrema[0], 0.25, 1e-12);
        assert_close(extrema[1], 0.75, 1e-12);
    }

    #[test]
    fn a_double_root_reports_one_extremum_not_two() {
        // x controls 0, 1, 0, 1 give 12t² - 12t + 3 = 0, i.e. (2t-1)² = 0:
        // the cusp-like case where both roots coincide at t = 1/2.
        let curve = CubicBezier::new(
            Point::new(0.0, 0.0),
            Point::new(1.0, 0.0),
            Point::new(0.0, 0.0),
            Point::new(1.0, 0.0),
        );
        let extrema = curve.extrema();
        assert_eq!(extrema.len(), 1);
        assert_close(extrema[0], 0.5, 1e-9);
    }

    #[test]
    fn a_cusped_cubic_reports_the_cusp_parameter_on_both_axes() {
        // Both axes share the double root at t = 1/2, and the two identical
        // roots must collapse into a single reported parameter.
        let curve = CubicBezier::new(
            Point::new(0.0, 0.0),
            Point::new(1.0, 1.0),
            Point::new(0.0, 0.0),
            Point::new(1.0, 1.0),
        );
        assert_eq!(curve.extrema().len(), 1);
        assert_close(curve.extrema()[0], 0.5, 1e-9);
    }

    #[test]
    fn a_straight_cubic_has_no_interior_extrema() {
        let straight = CubicBezier::from_line(Point::new(0.0, 0.0), Point::new(3.0, 4.0));
        assert!(straight.extrema().is_empty());
        // A cubic with collinear, evenly spaced controls is the same curve
        // written by hand.
        let by_hand = CubicBezier::new(
            Point::new(0.0, 0.0),
            Point::new(1.0, 0.0),
            Point::new(2.0, 0.0),
            Point::new(3.0, 0.0),
        );
        assert!(by_hand.extrema().is_empty());
    }

    #[test]
    fn a_degenerate_point_curve_produces_no_roots_and_a_zero_box() {
        let point = Point::new(2.0, -3.0);
        let curve = CubicBezier::new(point, point, point, point);
        assert!(curve.extrema().is_empty());
        assert_eq!(curve.tight_bounds(), Rect::from_point(point));
        assert_eq!(curve.arc_length(1e-9), 0.0);
    }

    #[test]
    fn the_bounds_of_a_symmetric_arch_are_exact() {
        // Both control points are pulled three units up, yet the curve
        // itself reaches only 3/4 of that: the analytic maximum of
        // 3t(1-t) at t = 1/2 is 3/4, scaled by the control height.
        let curve = CubicBezier::new(
            Point::new(0.0, 0.0),
            Point::new(0.0, 3.0),
            Point::new(4.0, 3.0),
            Point::new(4.0, 0.0),
        );
        let bounds = curve.tight_bounds();
        assert_eq!(bounds.x, 0.0);
        assert_eq!(bounds.y, 0.0);
        assert_eq!(bounds.width, 4.0);
        assert_eq!(bounds.height, 2.25);
        // The control-point hull would have claimed a height of 3.
        assert!(bounds.height < 3.0);
    }

    #[test]
    fn quad_extrema_and_bounds_match_the_analytic_answer() {
        let quad = QuadBezier::new(
            Point::new(0.0, 0.0),
            Point::new(1.0, 2.0),
            Point::new(2.0, 0.0),
        );
        let extrema = quad.extrema();
        assert_eq!(extrema.len(), 1);
        assert_close(extrema[0], 0.5, 1e-12);
        let bounds = quad.tight_bounds();
        assert_eq!(bounds.width, 2.0);
        assert_eq!(bounds.height, 1.0);
        // The elevated cubic must agree, since it is the same curve.
        assert_eq!(quad.to_cubic().tight_bounds(), bounds);
    }

    #[test]
    fn a_straight_quad_has_no_interior_extrema() {
        let quad = QuadBezier::new(
            Point::new(0.0, 0.0),
            Point::new(1.0, 1.0),
            Point::new(2.0, 2.0),
        );
        assert!(quad.extrema().is_empty());
        assert_eq!(quad.tight_bounds(), Rect::from_points(quad.p0, quad.p2));
    }

    #[test]
    fn the_arc_length_of_a_straight_cubic_is_its_chord_length() {
        let straight = CubicBezier::from_line(Point::new(0.0, 0.0), Point::new(3.0, 4.0));
        assert_close(straight.arc_length(1e-9), 5.0, 1e-12);

        let collinear = CubicBezier::new(
            Point::new(1.0, 1.0),
            Point::new(2.0, 1.0),
            Point::new(3.0, 1.0),
            Point::new(4.0, 1.0),
        );
        assert_close(collinear.arc_length(1e-9), 3.0, 1e-12);

        // A quadratic straight line measures the same way.
        let quad = QuadBezier::new(
            Point::new(0.0, 0.0),
            Point::new(1.5, 2.0),
            Point::new(3.0, 4.0),
        );
        assert_close(quad.arc_length(1e-9), 5.0, 1e-12);
    }

    #[test]
    fn the_arc_length_of_a_quarter_circle_cubic_is_a_quarter_of_the_circumference() {
        let k = quarter_circle_k();
        let curve = CubicBezier::new(
            Point::new(1.0, 0.0),
            Point::new(1.0, k),
            Point::new(k, 1.0),
            Point::new(0.0, 1.0),
        );
        // The residual is the *approximation* error of representing a
        // circular arc by a cubic, not error in the length measurement: the
        // classic k = 4/3*tan(pi/8) curve bulges about 2.7e-4 of a radius
        // outside the true circle around its middle.
        let expected = std::f64::consts::FRAC_PI_2;
        assert_close(curve.arc_length(1e-12), expected, 1e-3);
        assert!((curve.arc_length(1e-12) - expected).abs() / expected < 1e-3);
    }

    #[test]
    fn shrinking_the_tolerance_never_shortens_the_measured_length() {
        let curve = sample_cubic();
        let coarse = curve.arc_length(1.0);
        let fine = curve.arc_length(1e-3);
        let finer = curve.arc_length(1e-9);
        assert!(coarse <= fine + 1e-12, "{coarse} should not exceed {fine}");
        assert!(fine <= finer + 1e-12, "{fine} should not exceed {finer}");
        // Chords never exceed the arc, so every measurement is a lower bound.
        assert!(finer >= fine);
    }

    #[test]
    fn a_tolerance_that_is_not_finite_and_positive_falls_back_to_the_default() {
        let curve = sample_cubic();
        let default = curve.arc_length(DEFAULT_TOLERANCE);
        for tolerance in [0.0, -1.0, f64::NAN, f64::INFINITY] {
            assert_eq!(curve.arc_length(tolerance), default);
        }
    }

    #[test]
    fn flattening_emits_points_on_the_curve_ending_at_the_end_point() {
        let curve = sample_cubic();
        let mut emitted: Vec<(f64, Point)> = Vec::new();
        curve.flatten_with_params(1e-3, &mut |t, point| emitted.push((t, point)));

        assert!(emitted.len() > 4, "a curved cubic needs several pieces");
        let mut previous_t = 0.0;
        for &(t, point) in &emitted {
            assert!(
                t > previous_t,
                "parameters must increase: {t} after {previous_t}"
            );
            assert_point_close(point, curve.eval(t), 1e-9);
            previous_t = t;
        }
        let last = emitted[emitted.len() - 1];
        assert_eq!(last.0, 1.0);
        assert_eq!(last.1, curve.p3);
        // The start point is the caller's to supply.
        assert_ne!(emitted[0].1, curve.p0);
    }

    #[test]
    fn flattening_a_straight_cubic_emits_a_single_segment() {
        let straight = CubicBezier::from_line(Point::new(0.0, 0.0), Point::new(10.0, 0.0));
        let mut points = Vec::new();
        straight.flatten(1e-9, &mut |point| points.push(point));
        assert_eq!(points, vec![Point::new(10.0, 0.0)]);
    }

    #[test]
    fn a_tighter_tolerance_emits_more_segments() {
        let curve = sample_cubic();
        let count = |tolerance| {
            let mut n = 0_usize;
            curve.flatten(tolerance, &mut |_| n += 1);
            n
        };
        assert!(count(1e-4) > count(1e-1));
    }

    #[test]
    fn every_flattened_chord_stays_within_the_tolerance_of_the_curve() {
        let curve = sample_cubic();
        let tolerance = 1e-2;
        let mut points = vec![(0.0, curve.p0)];
        curve.flatten_with_params(tolerance, &mut |t, point| points.push((t, point)));
        for window in points.windows(2) {
            let (t0, a) = window[0];
            let (t1, b) = window[1];
            // Sample the curve across the piece and measure how far it
            // strays from the chord that replaced it.
            for step in 0..=8 {
                let s = f64::from(step) / 8.0;
                let on_curve = curve.eval(t0 + (t1 - t0) * s);
                let on_chord = lerp(a, b, s);
                assert!(
                    distance(on_curve, on_chord) <= tolerance * 1.000_001,
                    "chord deviates by {} at t = {}",
                    distance(on_curve, on_chord),
                    t0 + (t1 - t0) * s
                );
            }
        }
    }

    #[test]
    fn flatness_recognizes_lines_and_rejects_sharp_curves() {
        let straight = CubicBezier::from_line(Point::new(0.0, 0.0), Point::new(1.0, 0.0));
        assert!(straight.is_flat(1e-12));

        let arch = CubicBezier::new(
            Point::new(0.0, 0.0),
            Point::new(0.0, 3.0),
            Point::new(4.0, 3.0),
            Point::new(4.0, 0.0),
        );
        assert!(!arch.is_flat(0.5));
        assert!(arch.is_flat(100.0));
    }

    #[test]
    fn subdivision_stops_at_the_depth_cap() {
        // No curve is flat to within 1e-300, so only the cap can end this
        // recursion. It is driven directly rather than through `flatten`,
        // which would otherwise emit 2^MAX_SUBDIVISION_DEPTH points to make
        // the same point.
        let curve = sample_cubic();
        let mut count = 0_usize;
        flatten_recursive(
            &curve,
            0.0,
            1.0,
            1e-300,
            MAX_SUBDIVISION_DEPTH,
            &mut |_, _| {
                count += 1;
            },
        );
        assert_eq!(count, 1);

        let mut count = 0_usize;
        flatten_recursive(
            &curve,
            0.0,
            1.0,
            1e-300,
            MAX_SUBDIVISION_DEPTH - 3,
            &mut |_, _| count += 1,
        );
        assert_eq!(count, 8);
    }

    #[test]
    fn a_non_finite_curve_flattens_to_one_chord_instead_of_subdividing() {
        // Path data may name a magnitude that overflows an f64 (`1e999`),
        // which saturates to an infinity. Such a curve can never satisfy any
        // tolerance, so it must stop immediately rather than run to the
        // depth cap.
        let overflowed = CubicBezier::new(
            Point::new(0.0, 0.0),
            Point::new(f64::INFINITY, 0.0),
            Point::new(1.0, 0.0),
            Point::new(1.0, 0.0),
        );
        assert!(!overflowed.is_finite());
        let mut points = Vec::new();
        overflowed.flatten(1e-9, &mut |point| points.push(point));
        assert_eq!(points, vec![Point::new(1.0, 0.0)]);
        assert_eq!(overflowed.arc_length(1e-9), 1.0);

        let with_nan = CubicBezier::new(
            Point::new(0.0, 0.0),
            Point::new(f64::NAN, 0.0),
            Point::new(1.0, 0.0),
            Point::new(1.0, 0.0),
        );
        let mut count = 0_usize;
        with_nan.flatten(1e-9, &mut |_| count += 1);
        assert_eq!(count, 1);
    }

    #[test]
    fn finiteness_is_reported_for_both_curve_kinds() {
        assert!(sample_cubic().is_finite());
        let quad = QuadBezier::new(
            Point::new(0.0, 0.0),
            Point::new(1.0, 1.0),
            Point::new(2.0, 0.0),
        );
        assert!(quad.is_finite());
        assert!(!QuadBezier::new(quad.p0, Point::new(f64::INFINITY, 0.0), quad.p2).is_finite());
    }

    #[test]
    fn the_stable_quadratic_solver_finds_both_roots_of_a_lopsided_equation() {
        // Roots 1e8 and 1e-8: the textbook formula loses the small one to
        // cancellation entirely.
        let (first, second) = solve_quadratic(1.0, -(1e8 + 1e-8), 1.0);
        let first = first.expect("a positive discriminant has a first root");
        let second = second.expect("a nonzero q yields a second root");
        let (small, large) = if first < second {
            (first, second)
        } else {
            (second, first)
        };
        assert!((large - 1e8).abs() / 1e8 < 1e-12);
        assert!((small - 1e-8).abs() / 1e-8 < 1e-6);
    }

    #[test]
    fn the_quadratic_solver_handles_degenerate_coefficients() {
        assert_eq!(solve_quadratic(0.0, 0.0, 5.0), (None, None));
        assert_eq!(solve_quadratic(0.0, 2.0, -4.0), (Some(2.0), None));
        assert_eq!(solve_quadratic(1.0, 0.0, 1.0), (None, None));
    }
}
