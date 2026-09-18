//! Elliptical arcs: the two parameterizations of SVG's `A` command, and the
//! conversion between them.
//!
//! Path data writes an elliptical arc in **endpoint** form ([`ArcParams`]) —
//! "sweep from where I am to `(x, y)` along an ellipse with these radii and
//! this tilt, picking one of the four candidate arcs with two flags" — while
//! every geometric question about the arc (where is its middle, what is its
//! bounding box, what does it look like as Bézier curves) is asked of the
//! **center** form ([`CenterParams`]): a center, radii, a tilt, a start
//! angle, and a signed sweep.
//!
//! # What is normative here and what is not
//!
//! [`ArcParams::to_center`] implements SVG 1.1 Appendix F.6 exactly: the
//! out-of-range corrections of F.6.2, the endpoint-to-center equations of
//! F.6.5, and the radius correction of F.6.6.
//! [`CenterParams::to_endpoint`] implements the reverse direction, F.6.4.
//! Those are the specification's own mathematics.
//!
//! [`CenterParams::to_cubics`] is **not**: neither SVG 1.1 nor SVG 2 defines
//! an arc-to-Bézier algorithm, because neither needs one — an arc is a
//! primitive a renderer draws directly. The conversion here is the de-facto
//! standard technique every geometry library uses, and it is exact only in
//! the limit: each piece of at most a quarter turn is replaced by the cubic
//! that shares the arc's endpoints, endpoint tangents, and *midpoint*, using
//! the control-point distance `alpha = (4/3) * tan(sweep / 4)`. The residual
//! is a radial error of about `2.7e-4` of a radius for a 90-degree piece,
//! falling roughly as the sixth power of the piece angle. Callers who need
//! an exact answer for a bounded quantity should prefer the closed forms
//! that exist: [`CenterParams::point_at`] for a point and
//! [`CenterParams::tight_bounds`] for a bounding box.
//!
//! # Angle conventions
//!
//! [`ArcParams::x_axis_rotation`] is in **degrees**, matching the path-data
//! grammar. Everything in [`CenterParams`] is in **radians**, matching
//! [`f64`]'s trigonometry. Angles increase from the positive x axis towards
//! the positive y axis — which, since SVG's y axis points down the screen,
//! looks clockwise on screen. That is also the direction the `sweep` flag
//! selects: `sweep = true` gives a positive [`CenterParams::delta_theta`].

use std::f64::consts::{FRAC_PI_2, PI, TAU};

use crate::geometry::bezier::CubicBezier;
use crate::geometry::primitives::{Point, Rect};

/// The largest sweep, in degrees, that [`CenterParams::to_cubics`] will
/// approximate with a single cubic Bézier curve.
///
/// A quarter turn is the conventional choice: it keeps the radial error near
/// `2.7e-4` of a radius, which is invisible at any realistic scale, while
/// producing at most four curves for a whole ellipse. Use
/// [`CenterParams::to_cubics_with_max_angle`] to trade curve count for
/// accuracy.
pub const MAX_SEGMENT_ANGLE_DEG: f64 = 90.0;

/// The smallest per-piece sweep angle, in radians, that
/// [`CenterParams::to_cubics_with_max_angle`] honours.
///
/// A caller asking for pieces smaller than a thousandth of a radian is
/// asking for more than five thousand curves per turn, which no accuracy
/// requirement justifies (the approximation error is already at the level of
/// `f64` rounding well before that); the request is clamped rather than
/// obeyed so that a mistaken tiny value cannot allocate without bound.
pub const MIN_SEGMENT_ANGLE_RAD: f64 = 1e-3;

/// An elliptical arc in endpoint parameterization: exactly what an `A`
/// command in path data says.
///
/// The radii and rotation are stored **as authored**, which is how
/// [`crate::values::path::PathSegment::ArcTo`] keeps them: `rx`/`ry` may be
/// negative and `x_axis_rotation` need not be reduced. Correcting all of
/// that is [`ArcParams::to_center`]'s job, per SVG 1.1 Appendix F.6.2.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ArcParams {
    /// The point the arc starts at: the current point when the `A` command
    /// is reached.
    pub from: Point,
    /// The x radius of the ellipse, as authored. A negative value has its
    /// sign dropped during conversion.
    pub rx: f64,
    /// The y radius of the ellipse, as authored.
    pub ry: f64,
    /// The angle, in degrees, from the x axis of the current coordinate
    /// system to the x axis of the ellipse, as authored (not reduced modulo
    /// 360).
    pub x_axis_rotation: f64,
    /// The large-arc flag: `true` selects the arc sweeping more than 180
    /// degrees, `false` the one sweeping less.
    pub large_arc: bool,
    /// The sweep flag: `true` selects the arc drawn in the direction of
    /// increasing angle.
    pub sweep: bool,
    /// The point the arc ends at.
    pub to: Point,
}

/// What SVG 1.1 Appendix F.6.2's out-of-range handling makes of an arc.
///
/// Two of the three cases are not arcs at all, which is why
/// [`ArcParams::to_center`] cannot simply return a [`CenterParams`]. Both
/// degenerate cases are normative and both occur in real content — a
/// zero-radius arc is a common way to write "a straight line, but keep the
/// `A` command around" in generated path data.
///
/// This enum is deliberately not `non_exhaustive`: the specification
/// enumerates exactly these three outcomes, so callers are meant to match
/// them exhaustively.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ArcOutcome {
    /// The endpoints coincide. Per F.6.2 the segment is "equivalent to
    /// omitting the elliptical arc segment entirely": nothing is drawn, and
    /// the current point does not move.
    Omitted,
    /// A radius is zero (or the parameters are not finite). Per F.6.2 the
    /// segment is drawn as a straight line from
    /// [`ArcParams::from`] to [`ArcParams::to`].
    Line,
    /// A genuine elliptical arc, in center parameterization.
    Arc(CenterParams),
}

impl ArcOutcome {
    /// The center parameterization, when this outcome is a real arc.
    #[must_use]
    pub const fn center_params(&self) -> Option<&CenterParams> {
        match self {
            Self::Arc(params) => Some(params),
            Self::Omitted | Self::Line => None,
        }
    }

    /// True for [`ArcOutcome::Omitted`].
    #[must_use]
    pub const fn is_omitted(&self) -> bool {
        matches!(self, Self::Omitted)
    }

    /// True for [`ArcOutcome::Line`].
    #[must_use]
    pub const fn is_line(&self) -> bool {
        matches!(self, Self::Line)
    }
}

/// An elliptical arc in center parameterization: a center, corrected radii, a
/// tilt, a start angle, and a signed sweep.
///
/// Every field is already corrected — the radii are positive and large
/// enough to reach both endpoints, and `phi` is reduced — because the only
/// way to obtain one of these is [`ArcParams::to_center`], which applies
/// SVG 1.1 Appendix F.6.2 and F.6.6 first. Angles are in radians.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CenterParams {
    /// The x coordinate of the ellipse's center.
    pub cx: f64,
    /// The y coordinate of the ellipse's center.
    pub cy: f64,
    /// The x radius of the ellipse: positive, and scaled up from the
    /// authored value if that value was too small to join the endpoints
    /// (F.6.6).
    pub rx: f64,
    /// The y radius of the ellipse, corrected the same way.
    pub ry: f64,
    /// The ellipse's tilt in radians, reduced to `[0, 2*pi)`.
    pub phi: f64,
    /// The angle, in radians, of the arc's start point on the ellipse.
    pub theta1: f64,
    /// The signed sweep in radians: positive when the `sweep` flag was set,
    /// negative otherwise, and never larger than a full turn in magnitude.
    pub delta_theta: f64,
}

impl ArcParams {
    /// Builds an endpoint-form arc from the seven values an `A` command
    /// carries, plus the current point it starts from.
    #[must_use]
    pub const fn new(
        from: Point,
        rx: f64,
        ry: f64,
        x_axis_rotation: f64,
        large_arc: bool,
        sweep: bool,
        to: Point,
    ) -> Self {
        Self {
            from,
            rx,
            ry,
            x_axis_rotation,
            large_arc,
            sweep,
            to,
        }
    }

    /// Converts to center parameterization, applying SVG 1.1 Appendix F.6.
    ///
    /// The order of operations is the specification's own, and it matters:
    ///
    /// 1. **F.6.2** — coincident endpoints mean the segment is omitted
    ///    entirely, which is checked before anything is divided by anything.
    /// 2. **F.6.2** — the radii are made positive, and a zero radius means a
    ///    straight line. Non-finite parameters (path data may name `1e999`,
    ///    which saturates to an infinity) are treated as a straight line
    ///    too: no ellipse can be recovered from them, and a line to the
    ///    stated endpoint is the least surprising fallback.
    /// 3. **F.6.2** — the rotation is reduced modulo 360 degrees.
    /// 4. **F.6.5 step 1** — the start point is expressed in the frame the
    ///    ellipse is axis-aligned in.
    /// 5. **F.6.6** — if the ellipse is too small to reach both endpoints
    ///    (`lambda > 1`), both radii are scaled up by `sqrt(lambda)`, the
    ///    least scaling that makes the endpoints reachable.
    /// 6. **F.6.5 steps 2 to 4** — the center, start angle, and sweep,
    ///    with the sweep corrected by the `sweep` flag.
    ///
    /// ```
    /// use oxixml_svg::geometry::arc::{ArcOutcome, ArcParams};
    /// use oxixml_svg::geometry::primitives::Point;
    ///
    /// // A quarter of the unit circle, from (1,0) to (0,1).
    /// let arc = ArcParams::new(Point::new(1.0, 0.0), 1.0, 1.0, 0.0, false, true, Point::new(0.0, 1.0));
    /// let ArcOutcome::Arc(center) = arc.to_center() else {
    ///     panic!("a unit-circle quarter is a real arc");
    /// };
    /// assert!(center.cx.abs() < 1e-12 && center.cy.abs() < 1e-12);
    /// assert!((center.delta_theta - std::f64::consts::FRAC_PI_2).abs() < 1e-12);
    /// ```
    #[must_use]
    pub fn to_center(&self) -> ArcOutcome {
        // F.6.2: "If the endpoints (x1, y1) and (x2, y2) are identical, then
        // this is equivalent to omitting the elliptical arc segment
        // entirely."
        if self.from == self.to {
            return ArcOutcome::Omitted;
        }
        if !(self.from.x.is_finite()
            && self.from.y.is_finite()
            && self.to.x.is_finite()
            && self.to.y.is_finite()
            && self.rx.is_finite()
            && self.ry.is_finite()
            && self.x_axis_rotation.is_finite())
        {
            return ArcOutcome::Line;
        }

        // F.6.2: "If rx or ry have negative signs, these are dropped; the
        // absolute value is used instead." "If rx = 0 or ry = 0, then treat
        // this as a straight line from (x1, y1) to (x2, y2)."
        let mut rx = self.rx.abs();
        let mut ry = self.ry.abs();
        if rx == 0.0 || ry == 0.0 {
            return ArcOutcome::Line;
        }

        // F.6.2: "phi is taken mod 360 degrees."
        let phi = self.x_axis_rotation.rem_euclid(360.0).to_radians();
        let (sin_phi, cos_phi) = phi.sin_cos();

        // F.6.5 step 1: the start point in the ellipse's own frame, with the
        // chord's midpoint moved to the origin.
        let dx2 = (self.from.x - self.to.x) / 2.0;
        let dy2 = (self.from.y - self.to.y) / 2.0;
        let x1p = cos_phi * dx2 + sin_phi * dy2;
        let y1p = -sin_phi * dx2 + cos_phi * dy2;

        // F.6.6: scale the radii up if they are too small to span the chord.
        let lambda = (x1p * x1p) / (rx * rx) + (y1p * y1p) / (ry * ry);
        if lambda > 1.0 {
            let scale = lambda.sqrt();
            rx *= scale;
            ry *= scale;
        }

        // F.6.5 step 2: the center, in the same frame.
        let rx2 = rx * rx;
        let ry2 = ry * ry;
        let x1p2 = x1p * x1p;
        let y1p2 = y1p * y1p;
        let denominator = rx2 * y1p2 + ry2 * x1p2;
        if denominator == 0.0 {
            // Only reachable when both radii overflow to infinity while
            // squaring, or when the corrected radii are so large that the
            // scaled chord underflows; a line is the safe reading.
            return ArcOutcome::Line;
        }
        // Clamped at zero: floating-point error drives this fractionally
        // negative exactly when the ellipse is the minimum size for the
        // chord, which is the common case of radii chosen equal to half the
        // chord length.
        let radicand = ((rx2 * ry2 - rx2 * y1p2 - ry2 * x1p2) / denominator).max(0.0);
        let sign = if self.large_arc == self.sweep {
            -1.0
        } else {
            1.0
        };
        let coefficient = sign * radicand.sqrt();
        let cxp = coefficient * (rx * y1p / ry);
        let cyp = coefficient * (-(ry * x1p) / rx);

        // F.6.5 step 3: back to the original frame.
        let cx = cos_phi * cxp - sin_phi * cyp + (self.from.x + self.to.x) / 2.0;
        let cy = sin_phi * cxp + cos_phi * cyp + (self.from.y + self.to.y) / 2.0;

        // F.6.5 steps 4 and 5: the start angle and the sweep.
        let ux = (x1p - cxp) / rx;
        let uy = (y1p - cyp) / ry;
        let vx = (-x1p - cxp) / rx;
        let vy = (-y1p - cyp) / ry;
        let theta1 = angle_between(1.0, 0.0, ux, uy);
        // `angle_between` already returns a value in [-pi, pi]; reducing it
        // modulo a full turn first — as a literal reading of the
        // specification's "mod 360" invites — would flip the sign of every
        // negative sweep and break the flag correction below.
        let mut delta_theta = angle_between(ux, uy, vx, vy);
        if !self.sweep && delta_theta > 0.0 {
            delta_theta -= TAU;
        } else if self.sweep && delta_theta < 0.0 {
            delta_theta += TAU;
        }

        ArcOutcome::Arc(CenterParams {
            cx,
            cy,
            rx,
            ry,
            phi,
            theta1,
            delta_theta,
        })
    }

    /// The arc as a chain of cubic Bézier curves.
    ///
    /// A convenience over [`ArcParams::to_center`] plus
    /// [`CenterParams::to_cubics`] that also gives the two degenerate cases
    /// of F.6.2 a curve-shaped answer: an omitted arc becomes no curves at
    /// all, and a zero-radius arc becomes the single straight cubic from
    /// [`CubicBezier::from_line`]. Callers that must distinguish "draw
    /// nothing" from "draw a line" — a path normalizer, for instance, which
    /// wants to emit an `L` command — should match on
    /// [`ArcParams::to_center`] instead.
    ///
    /// The first curve starts exactly at [`ArcParams::from`] and the last
    /// ends exactly at [`ArcParams::to`]: the endpoints are taken from this
    /// value rather than recomputed from the center form, so a chain of
    /// segments never develops a hairline gap at an arc.
    #[must_use]
    pub fn to_cubics(&self) -> Vec<CubicBezier> {
        match self.to_center() {
            ArcOutcome::Omitted => Vec::new(),
            ArcOutcome::Line => vec![CubicBezier::from_line(self.from, self.to)],
            ArcOutcome::Arc(center) => {
                let mut curves = center.to_cubics();
                if let Some(first) = curves.first_mut() {
                    first.p0 = self.from;
                }
                if let Some(last) = curves.last_mut() {
                    last.p3 = self.to;
                }
                curves
            }
        }
    }
}

impl CenterParams {
    /// Builds a center-form arc directly from its seven components, with
    /// `phi`, `theta1` and `delta_theta` in radians.
    ///
    /// No correction is applied: this is the constructor for callers that
    /// already hold a center form (a circle or ellipse element, say, which
    /// has one by construction). An arc that came from path data should be
    /// converted with [`ArcParams::to_center`] instead, which applies the
    /// specification's out-of-range handling.
    #[must_use]
    pub const fn new(
        cx: f64,
        cy: f64,
        rx: f64,
        ry: f64,
        phi: f64,
        theta1: f64,
        delta_theta: f64,
    ) -> Self {
        Self {
            cx,
            cy,
            rx,
            ry,
            phi,
            theta1,
            delta_theta,
        }
    }

    /// The ellipse's center as a [`Point`].
    #[must_use]
    pub const fn center(&self) -> Point {
        Point::new(self.cx, self.cy)
    }

    /// The angle the arc ends at: `theta1 + delta_theta`.
    #[must_use]
    pub fn theta2(&self) -> f64 {
        self.theta1 + self.delta_theta
    }

    /// The point on the ellipse at angle `theta`, in radians.
    ///
    /// This is the forward evaluation SVG 1.1 Appendix F.6.4 gives:
    ///
    /// ```text
    /// x = cx + rx*cos(phi)*cos(theta) - ry*sin(phi)*sin(theta)
    /// y = cy + rx*sin(phi)*cos(theta) + ry*cos(phi)*sin(theta)
    /// ```
    ///
    /// `theta` is not restricted to the swept range; the whole ellipse is
    /// addressable, which is what makes the bounding-box computation
    /// possible.
    #[must_use]
    pub fn point_at(&self, theta: f64) -> Point {
        let (sin_phi, cos_phi) = self.phi.sin_cos();
        let (sin_theta, cos_theta) = theta.sin_cos();
        Point::new(
            self.cx + self.rx * cos_phi * cos_theta - self.ry * sin_phi * sin_theta,
            self.cy + self.rx * sin_phi * cos_theta + self.ry * cos_phi * sin_theta,
        )
    }

    /// The derivative of [`CenterParams::point_at`] with respect to `theta`:
    /// a vector along the ellipse's tangent at that angle.
    ///
    /// It is **not** normalized, and must not be: its length carries the
    /// `rx`/`ry` scaling that makes the Bézier construction in
    /// [`CenterParams::to_cubics`] correct for a non-circular, tilted
    /// ellipse. Normalizing it — or, equivalently, building the cubic on a
    /// unit circle and transforming afterwards without rescaling the
    /// controls — is the classic way to get an arc conversion that looks
    /// right for circles and is subtly wrong for everything else.
    #[must_use]
    pub fn derivative_at(&self, theta: f64) -> Point {
        let (sin_phi, cos_phi) = self.phi.sin_cos();
        let (sin_theta, cos_theta) = theta.sin_cos();
        Point::new(
            -self.rx * cos_phi * sin_theta - self.ry * sin_phi * cos_theta,
            -self.rx * sin_phi * sin_theta + self.ry * cos_phi * cos_theta,
        )
    }

    /// The arc's start point, `point_at(theta1)`.
    #[must_use]
    pub fn start_point(&self) -> Point {
        self.point_at(self.theta1)
    }

    /// The arc's end point, `point_at(theta1 + delta_theta)`.
    #[must_use]
    pub fn end_point(&self) -> Point {
        self.point_at(self.theta2())
    }

    /// True when `theta` lies within the swept range, counting full turns:
    /// an angle is contained if any of `theta + 2*pi*k` falls between
    /// `theta1` and `theta1 + delta_theta`, inclusive of both ends.
    ///
    /// This is what makes the closed-form bounding box work: the ellipse's
    /// four axis extrema are computed on the whole ellipse and then filtered
    /// down to the ones the arc actually reaches.
    #[must_use]
    pub fn contains_angle(&self, theta: f64) -> bool {
        if self.delta_theta >= 0.0 {
            (theta - self.theta1).rem_euclid(TAU) <= self.delta_theta
        } else {
            (self.theta1 - theta).rem_euclid(TAU) <= -self.delta_theta
        }
    }

    /// Converts back to endpoint parameterization, per SVG 1.1 Appendix
    /// F.6.4.
    ///
    /// The endpoints come from [`CenterParams::point_at`], the rotation is
    /// converted back to degrees, and the two flags are recovered from the
    /// sweep: `large_arc` is set when more than half a turn is swept, and
    /// `sweep` when the sweep is positive. Round-tripping an
    /// [`ArcParams`] through [`ArcParams::to_center`] and back reproduces
    /// the original endpoints and flags, and reproduces the radii unless
    /// F.6.2/F.6.6 corrected them.
    #[must_use]
    pub fn to_endpoint(&self) -> ArcParams {
        ArcParams {
            from: self.start_point(),
            rx: self.rx,
            ry: self.ry,
            x_axis_rotation: self.phi.to_degrees(),
            large_arc: self.delta_theta.abs() > PI,
            sweep: self.delta_theta > 0.0,
            to: self.end_point(),
        }
    }

    /// The arc as a chain of cubic Bézier curves, each spanning at most
    /// [`MAX_SEGMENT_ANGLE_DEG`].
    ///
    /// See the [module documentation](self) on why this is an approximation
    /// and how large the error is. The chain is contiguous — each curve
    /// starts exactly where the previous one ended — and never empty: an
    /// arc with a zero sweep still yields one (degenerate) curve.
    #[must_use]
    pub fn to_cubics(&self) -> Vec<CubicBezier> {
        self.to_cubics_with_max_angle(MAX_SEGMENT_ANGLE_DEG.to_radians())
    }

    /// [`CenterParams::to_cubics`] with a caller-chosen limit, in radians,
    /// on how much of the arc a single cubic may cover.
    ///
    /// Smaller pieces mean more curves and less error: the radial error of
    /// the construction falls roughly as the sixth power of the piece angle,
    /// so halving the limit divides the error by about sixty-four. The limit
    /// is clamped to `[MIN_SEGMENT_ANGLE_RAD, pi]`, and a limit that is not
    /// finite and positive is replaced by a quarter turn.
    #[must_use]
    pub fn to_cubics_with_max_angle(&self, max_angle: f64) -> Vec<CubicBezier> {
        let max_angle = if max_angle.is_finite() && max_angle > 0.0 {
            max_angle.clamp(MIN_SEGMENT_ANGLE_RAD, PI)
        } else {
            FRAC_PI_2
        };
        let sweep = if self.delta_theta.is_finite() {
            self.delta_theta
        } else {
            0.0
        };
        let count = ((sweep.abs() / max_angle).ceil() as usize).max(1);
        let step = sweep / count as f64;
        // The control-point distance that makes the cubic share the arc's
        // endpoints, endpoint tangents, and midpoint.
        let alpha = 4.0 / 3.0 * (step / 4.0).tan();

        let mut curves = Vec::with_capacity(count);
        let mut theta = self.theta1;
        let mut point = self.point_at(theta);
        let mut tangent = self.derivative_at(theta);
        for _ in 0..count {
            let next_theta = theta + step;
            let next_point = self.point_at(next_theta);
            let next_tangent = self.derivative_at(next_theta);
            curves.push(CubicBezier::new(
                point,
                Point::new(point.x + alpha * tangent.x, point.y + alpha * tangent.y),
                Point::new(
                    next_point.x - alpha * next_tangent.x,
                    next_point.y - alpha * next_tangent.y,
                ),
                next_point,
            ));
            theta = next_theta;
            point = next_point;
            tangent = next_tangent;
        }
        curves
    }

    /// The exact bounding box of the arc.
    ///
    /// Exact, not the box of the Bézier approximation: the angles at which
    /// the *rotated* ellipse's tangent runs horizontal or vertical have a
    /// closed form,
    ///
    /// ```text
    /// theta_x = atan2(-ry*sin(phi), rx*cos(phi))   (and that angle plus pi)
    /// theta_y = atan2( ry*cos(phi), rx*sin(phi))   (and that angle plus pi)
    /// ```
    ///
    /// so the box is the two endpoints expanded by whichever of those four
    /// extrema the arc actually sweeps through.
    ///
    /// ```
    /// use oxixml_svg::geometry::arc::{ArcOutcome, ArcParams};
    /// use oxixml_svg::geometry::primitives::Point;
    ///
    /// // The lower half of the unit circle, swept from (-1,0) to (1,0).
    /// let arc = ArcParams::new(Point::new(-1.0, 0.0), 1.0, 1.0, 0.0, false, true, Point::new(1.0, 0.0));
    /// let ArcOutcome::Arc(center) = arc.to_center() else {
    ///     panic!("a unit-circle half is a real arc");
    /// };
    /// let bounds = center.tight_bounds();
    /// assert!((bounds.x + 1.0).abs() < 1e-12);
    /// assert!((bounds.width - 2.0).abs() < 1e-12);
    /// assert!((bounds.height - 1.0).abs() < 1e-12);
    /// ```
    #[must_use]
    pub fn tight_bounds(&self) -> Rect {
        let (sin_phi, cos_phi) = self.phi.sin_cos();
        let theta_x = (-self.ry * sin_phi).atan2(self.rx * cos_phi);
        let theta_y = (self.ry * cos_phi).atan2(self.rx * sin_phi);
        let mut bounds = Rect::from_points(self.start_point(), self.end_point());
        for base in [theta_x, theta_y] {
            for half_turn in [0.0, PI] {
                let theta = base + half_turn;
                if self.contains_angle(theta) {
                    bounds = bounds.expand_to_include(self.point_at(theta));
                }
            }
        }
        bounds
    }

    /// True when every component is finite.
    #[must_use]
    pub fn is_finite(&self) -> bool {
        self.cx.is_finite()
            && self.cy.is_finite()
            && self.rx.is_finite()
            && self.ry.is_finite()
            && self.phi.is_finite()
            && self.theta1.is_finite()
            && self.delta_theta.is_finite()
    }
}

/// The angle from the vector `u` to the vector `v`, in radians, signed by
/// the direction of rotation and lying in `[-pi, pi]`.
///
/// This is SVG 1.1 Appendix F.6.5's angle function,
/// `sign(ux*vy - uy*vx) * acos((u . v) / (|u| * |v|))`, with two guards the
/// specification leaves implicit: the cosine is clamped to `[-1, 1]`, since
/// floating-point error pushes it fractionally outside exactly in the
/// near-degenerate cases that matter (an `acos` of `1.0000000000000002` is a
/// NaN that then contaminates the whole conversion), and a zero-length
/// vector yields zero rather than a NaN.
fn angle_between(ux: f64, uy: f64, vx: f64, vy: f64) -> f64 {
    let lengths = (ux * ux + uy * uy).sqrt() * (vx * vx + vy * vy).sqrt();
    if lengths == 0.0 {
        return 0.0;
    }
    let cosine = ((ux * vx + uy * vy) / lengths).clamp(-1.0, 1.0);
    let cross = ux * vy - uy * vx;
    if cross < 0.0 {
        -cosine.acos()
    } else {
        cosine.acos()
    }
}

/// Converts an arc from endpoint to center parameterization; the
/// free-function spelling of [`ArcParams::to_center`], for callers that
/// dispatch through a table of conversions.
#[must_use]
pub fn endpoint_to_center(arc: &ArcParams) -> ArcOutcome {
    arc.to_center()
}

/// Converts an arc from center to endpoint parameterization; the
/// free-function spelling of [`CenterParams::to_endpoint`].
#[must_use]
pub fn center_to_endpoint(arc: &CenterParams) -> ArcParams {
    arc.to_endpoint()
}

/// Approximates an endpoint-form arc by cubic Bézier curves; the
/// free-function spelling of [`ArcParams::to_cubics`].
#[must_use]
pub fn arc_to_cubics(arc: &ArcParams) -> Vec<CubicBezier> {
    arc.to_cubics()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The control-point factor for a quarter turn, `(4/3) * tan(pi/8)`.
    const QUARTER_K: f64 = 0.552_284_749_830_793_4;

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

    fn center_of(arc: &ArcParams) -> CenterParams {
        match arc.to_center() {
            ArcOutcome::Arc(center) => center,
            other => panic!("expected a real arc, got {other:?}"),
        }
    }

    /// How far the point `(x, y)` is from the ellipse, measured as the
    /// radial coordinate in the ellipse's own untilted frame: exactly `1` on
    /// the ellipse.
    fn ellipse_radius(center: &CenterParams, point: Point) -> f64 {
        let (sin_phi, cos_phi) = center.phi.sin_cos();
        let dx = point.x - center.cx;
        let dy = point.y - center.cy;
        let x = dx * cos_phi + dy * sin_phi;
        let y = -dx * sin_phi + dy * cos_phi;
        ((x / center.rx).powi(2) + (y / center.ry).powi(2)).sqrt()
    }

    #[test]
    fn a_quarter_of_the_unit_circle_has_the_expected_center_form() {
        let arc = ArcParams::new(
            Point::new(1.0, 0.0),
            1.0,
            1.0,
            0.0,
            false,
            true,
            Point::new(0.0, 1.0),
        );
        let center = center_of(&arc);
        assert_close(center.cx, 0.0, 1e-12);
        assert_close(center.cy, 0.0, 1e-12);
        assert_close(center.rx, 1.0, 1e-12);
        assert_close(center.ry, 1.0, 1e-12);
        assert_close(center.phi, 0.0, 1e-12);
        assert_close(center.theta1, 0.0, 1e-12);
        assert_close(center.delta_theta, FRAC_PI_2, 1e-12);
    }

    #[test]
    fn a_quarter_circle_becomes_one_cubic_with_the_classic_control_distance() {
        let arc = ArcParams::new(
            Point::new(1.0, 0.0),
            1.0,
            1.0,
            0.0,
            false,
            true,
            Point::new(0.0, 1.0),
        );
        let curves = arc.to_cubics();
        assert_eq!(curves.len(), 1);
        let curve = curves[0];
        assert_close(QUARTER_K, 4.0 / 3.0 * (PI / 8.0).tan(), 1e-15);
        assert_point_close(curve.p0, Point::new(1.0, 0.0), 1e-12);
        assert_point_close(curve.p1, Point::new(1.0, QUARTER_K), 1e-12);
        assert_point_close(curve.p2, Point::new(QUARTER_K, 1.0), 1e-12);
        assert_point_close(curve.p3, Point::new(0.0, 1.0), 1e-12);
        // The construction's defining property: the cubic's midpoint is on
        // the arc, at 45 degrees.
        assert_point_close(
            curve.eval(0.5),
            Point::new(
                std::f64::consts::FRAC_1_SQRT_2,
                std::f64::consts::FRAC_1_SQRT_2,
            ),
            1e-12,
        );
    }

    #[test]
    fn a_tilted_elliptical_arc_keeps_its_cubic_midpoint_on_the_ellipse() {
        // The circular anchor above cannot tell a correct conversion from
        // one that normalizes the tangent or builds on a unit circle and
        // transforms afterwards: with rx == ry and no tilt, all three agree.
        // This one has rx != ry *and* a tilt, so it does distinguish them.
        let center = CenterParams::new(3.0, -2.0, 2.0, 1.0, 30.0_f64.to_radians(), 0.4, FRAC_PI_2);
        let curves = center.to_cubics();
        assert_eq!(curves.len(), 1);
        let curve = curves[0];

        assert_point_close(curve.p0, center.point_at(0.4), 1e-12);
        assert_point_close(curve.p3, center.point_at(0.4 + FRAC_PI_2), 1e-12);
        assert_close(ellipse_radius(&center, curve.p0), 1.0, 1e-12);
        assert_close(ellipse_radius(&center, curve.p3), 1.0, 1e-12);

        let middle = curve.eval(0.5);
        assert_close(ellipse_radius(&center, middle), 1.0, 1e-9);
        assert_point_close(middle, center.point_at(0.4 + FRAC_PI_2 / 2.0), 1e-9);

        // Everywhere else the cubic is close to the ellipse but not on it,
        // which is the documented 2.7e-4 approximation error.
        for step in 1..20 {
            let radius = ellipse_radius(&center, curve.eval(f64::from(step) / 20.0));
            assert_close(radius, 1.0, 3e-4);
        }
    }

    #[test]
    fn the_four_flag_combinations_choose_the_four_candidate_arcs() {
        // Two unit circles pass through both (0,0) and (1,1): one centered
        // at (1,0) and one at (0,1). Each carries a short and a long arc
        // between the two points, and the flags pick among the four.
        let build = |large_arc, sweep| {
            center_of(&ArcParams::new(
                Point::new(0.0, 0.0),
                1.0,
                1.0,
                0.0,
                large_arc,
                sweep,
                Point::new(1.0, 1.0),
            ))
        };
        let quarter = FRAC_PI_2;
        let cases = [
            // (large_arc, sweep, center, delta, the arc's own midpoint)
            (
                false,
                false,
                Point::new(1.0, 0.0),
                -quarter,
                Point::new(1.0 - 0.5_f64.sqrt(), 0.5_f64.sqrt()),
            ),
            (
                false,
                true,
                Point::new(0.0, 1.0),
                quarter,
                Point::new(0.5_f64.sqrt(), 1.0 - 0.5_f64.sqrt()),
            ),
            (
                true,
                false,
                Point::new(0.0, 1.0),
                -3.0 * quarter,
                Point::new(-(0.5_f64.sqrt()), 1.0 + 0.5_f64.sqrt()),
            ),
            (
                true,
                true,
                Point::new(1.0, 0.0),
                3.0 * quarter,
                Point::new(1.0 + 0.5_f64.sqrt(), -(0.5_f64.sqrt())),
            ),
        ];
        for (large_arc, sweep, expected_center, expected_delta, expected_middle) in cases {
            let center = build(large_arc, sweep);
            let label = format!("large_arc={large_arc}, sweep={sweep}");
            assert_point_close(center.center(), expected_center, 1e-12);
            assert!(
                (center.delta_theta - expected_delta).abs() < 1e-12,
                "{label}: expected sweep {expected_delta}, got {}",
                center.delta_theta
            );
            // The sweep's sign is the sweep flag, and its magnitude crosses
            // half a turn exactly when the large-arc flag is set.
            assert_eq!(center.delta_theta > 0.0, sweep, "{label}");
            assert_eq!(center.delta_theta.abs() > PI, large_arc, "{label}");
            // Both endpoints lie on the ellipse, whichever arc was chosen.
            assert_point_close(center.start_point(), Point::new(0.0, 0.0), 1e-12);
            assert_point_close(center.end_point(), Point::new(1.0, 1.0), 1e-12);
            // And the arc bulges to the side the flags asked for.
            assert_point_close(
                center.point_at(center.theta1 + center.delta_theta / 2.0),
                expected_middle,
                1e-12,
            );
        }
    }

    #[test]
    fn coincident_endpoints_omit_the_segment_entirely() {
        let arc = ArcParams::new(
            Point::new(5.0, 5.0),
            10.0,
            10.0,
            0.0,
            true,
            true,
            Point::new(5.0, 5.0),
        );
        assert_eq!(arc.to_center(), ArcOutcome::Omitted);
        assert!(arc.to_center().is_omitted());
        assert!(arc.to_cubics().is_empty());
        assert_eq!(arc.to_center().center_params(), None);
    }

    #[test]
    fn a_zero_radius_becomes_a_straight_line() {
        for (rx, ry) in [(0.0, 5.0), (5.0, 0.0), (0.0, 0.0)] {
            let arc = ArcParams::new(
                Point::new(0.0, 0.0),
                rx,
                ry,
                0.0,
                true,
                true,
                Point::new(3.0, 4.0),
            );
            assert_eq!(arc.to_center(), ArcOutcome::Line, "rx={rx}, ry={ry}");
            assert!(arc.to_center().is_line());
            let curves = arc.to_cubics();
            assert_eq!(curves.len(), 1);
            assert_eq!(curves[0].p0, Point::new(0.0, 0.0));
            assert_eq!(curves[0].p3, Point::new(3.0, 4.0));
            assert_close(curves[0].arc_length(1e-9), 5.0, 1e-9);
        }
    }

    #[test]
    fn negative_radii_have_their_signs_dropped() {
        let positive = ArcParams::new(
            Point::new(0.0, 0.0),
            2.0,
            3.0,
            0.0,
            false,
            true,
            Point::new(1.0, 1.0),
        );
        let negative = ArcParams::new(
            Point::new(0.0, 0.0),
            -2.0,
            -3.0,
            0.0,
            false,
            true,
            Point::new(1.0, 1.0),
        );
        assert_eq!(positive.to_center(), negative.to_center());
    }

    #[test]
    fn the_rotation_is_reduced_modulo_a_full_turn() {
        let at = |rotation| {
            center_of(&ArcParams::new(
                Point::new(0.0, 0.0),
                3.0,
                1.0,
                rotation,
                false,
                true,
                Point::new(1.0, 1.0),
            ))
        };
        let base = at(30.0);
        assert_close(base.phi, 30.0_f64.to_radians(), 1e-12);
        assert_eq!(at(390.0), base);
        assert_eq!(at(-330.0), base);
        assert_eq!(at(750.0), base);
    }

    #[test]
    fn radii_too_small_are_scaled_up_and_still_reach_both_endpoints() {
        // The chord is 2 units long but the radii are only 0.5: F.6.6 scales
        // both by sqrt(lambda) = 2, giving the unique circle of radius 1
        // through both points.
        let arc = ArcParams::new(
            Point::new(0.0, 0.0),
            0.5,
            0.5,
            0.0,
            false,
            true,
            Point::new(2.0, 0.0),
        );
        let center = center_of(&arc);
        assert_close(center.rx, 1.0, 1e-12);
        assert_close(center.ry, 1.0, 1e-12);
        assert_point_close(center.center(), Point::new(1.0, 0.0), 1e-12);
        assert_point_close(center.start_point(), arc.from, 1e-12);
        assert_point_close(center.end_point(), arc.to, 1e-12);
        // The scaling is the least that works: both endpoints sit exactly on
        // the corrected ellipse.
        assert_close(ellipse_radius(&center, arc.from), 1.0, 1e-12);
        assert_close(ellipse_radius(&center, arc.to), 1.0, 1e-12);
    }

    #[test]
    fn radii_that_exactly_span_the_chord_do_not_produce_a_nan() {
        // lambda is exactly 1 here, so the radicand of F.6.5 step 2 is zero
        // up to rounding — and rounding takes it fractionally negative,
        // which is a NaN without the clamp.
        let arc = ArcParams::new(
            Point::new(0.0, 0.0),
            1.0,
            1.0,
            0.0,
            false,
            true,
            Point::new(2.0, 0.0),
        );
        let center = center_of(&arc);
        assert!(center.is_finite());
        assert_point_close(center.center(), Point::new(1.0, 0.0), 1e-12);
        assert_close(center.delta_theta.abs(), PI, 1e-12);

        // The same, tilted and elliptical, so the clamp is exercised with
        // less symmetric arithmetic.
        let tilted = ArcParams::new(
            Point::new(0.0, 0.0),
            1.0,
            2.0,
            37.0,
            true,
            false,
            Point::new(1.629_0, 2.744_0),
        );
        let center = center_of(&tilted);
        assert!(center.is_finite());
    }

    #[test]
    fn non_finite_parameters_degrade_to_a_straight_line() {
        let cases = [
            ArcParams::new(
                Point::new(f64::INFINITY, 0.0),
                1.0,
                1.0,
                0.0,
                false,
                true,
                Point::new(1.0, 1.0),
            ),
            ArcParams::new(
                Point::new(0.0, 0.0),
                f64::INFINITY,
                1.0,
                0.0,
                false,
                true,
                Point::new(1.0, 1.0),
            ),
            ArcParams::new(
                Point::new(0.0, 0.0),
                1.0,
                1.0,
                f64::NAN,
                false,
                true,
                Point::new(1.0, 1.0),
            ),
        ];
        for arc in cases {
            assert_eq!(arc.to_center(), ArcOutcome::Line, "{arc:?}");
        }
    }

    #[test]
    fn endpoint_and_center_forms_round_trip() {
        let cases = [
            ArcParams::new(
                Point::new(0.0, 0.0),
                1.0,
                1.0,
                0.0,
                false,
                true,
                Point::new(1.0, 1.0),
            ),
            ArcParams::new(
                Point::new(0.0, 0.0),
                1.0,
                1.0,
                0.0,
                true,
                false,
                Point::new(1.0, 1.0),
            ),
            ArcParams::new(
                Point::new(-3.0, 7.5),
                8.0,
                6.0,
                37.0,
                true,
                true,
                Point::new(2.0, -1.25),
            ),
            ArcParams::new(
                Point::new(10.0, 10.0),
                6.0,
                6.0,
                0.0,
                false,
                false,
                Point::new(14.0, 13.0),
            ),
            ArcParams::new(
                Point::new(0.0, 0.0),
                0.5,
                0.5,
                90.0,
                false,
                true,
                Point::new(2.0, 0.0),
            ),
        ];
        for arc in cases {
            let center = center_of(&arc);
            let back = center.to_endpoint();
            assert_point_close(back.from, arc.from, 1e-9);
            assert_point_close(back.to, arc.to, 1e-9);
            assert_eq!(back.sweep, arc.sweep, "{arc:?}");
            // An arc whose radii F.6.6 had to scale up spans exactly half a
            // turn, and then both settings of the large-arc flag describe
            // the same arc: the recovered flag is `false` whichever way it
            // was authored, and that is not a round-trip failure.
            if (center.delta_theta.abs() - PI).abs() > 1e-9 {
                assert_eq!(back.large_arc, arc.large_arc, "{arc:?}");
            } else {
                assert!(!back.large_arc, "{arc:?}");
            }
            // Converting the recovered endpoint form again must reproduce
            // the very same center form.
            let again = center_of(&back);
            assert_point_close(again.center(), center.center(), 1e-9);
            assert_close(again.rx, center.rx, 1e-9);
            assert_close(again.ry, center.ry, 1e-9);
            assert_close(again.delta_theta, center.delta_theta, 1e-9);
            // The free-function spellings agree with the methods.
            assert_eq!(endpoint_to_center(&arc), arc.to_center());
            assert_eq!(center_to_endpoint(&center), back);
            assert_eq!(arc_to_cubics(&arc), arc.to_cubics());
        }
    }

    #[test]
    fn a_chain_of_cubics_is_contiguous_and_spans_the_whole_sweep() {
        let arc = ArcParams::new(
            Point::new(-3.0, 7.5),
            8.0,
            6.0,
            37.0,
            true,
            true,
            Point::new(2.0, -1.25),
        );
        let center = center_of(&arc);
        assert!(center.delta_theta.abs() > PI);
        let curves = arc.to_cubics();
        assert!(
            curves.len() >= 3,
            "a sweep over 180 degrees needs 3+ pieces"
        );
        assert_eq!(
            curves[0].p0, arc.from,
            "the chain starts at the arc's start"
        );
        assert_eq!(
            curves[curves.len() - 1].p3,
            arc.to,
            "the chain ends at the arc's end"
        );
        for pair in curves.windows(2) {
            assert_eq!(pair[0].p3, pair[1].p0, "the chain must not have gaps");
        }
        // Every joint sits on the ellipse.
        for curve in &curves {
            assert_close(ellipse_radius(&center, curve.p0), 1.0, 1e-9);
            assert_close(ellipse_radius(&center, curve.p3), 1.0, 1e-9);
        }
    }

    #[test]
    fn the_piece_count_follows_the_maximum_segment_angle() {
        let center = CenterParams::new(0.0, 0.0, 1.0, 1.0, 0.0, 0.0, TAU);
        assert_eq!(center.to_cubics().len(), 4);
        assert_eq!(center.to_cubics_with_max_angle(PI).len(), 2);
        assert_eq!(center.to_cubics_with_max_angle(FRAC_PI_2 / 2.0).len(), 8);
        // Out-of-range limits are clamped rather than obeyed.
        assert_eq!(center.to_cubics_with_max_angle(f64::NAN).len(), 4);
        assert_eq!(center.to_cubics_with_max_angle(0.0).len(), 4);
        assert_eq!(center.to_cubics_with_max_angle(TAU).len(), 2);
        assert_eq!(
            center.to_cubics_with_max_angle(1e-12).len(),
            (TAU / MIN_SEGMENT_ANGLE_RAD).ceil() as usize
        );
        // A zero sweep still produces one (degenerate) curve.
        let still = CenterParams::new(0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0);
        assert_eq!(still.to_cubics().len(), 1);
    }

    #[test]
    fn finer_pieces_track_the_ellipse_more_closely() {
        let center = CenterParams::new(0.0, 0.0, 3.0, 1.0, 20.0_f64.to_radians(), 0.0, PI);
        let worst = |max_angle: f64| {
            center
                .to_cubics_with_max_angle(max_angle)
                .iter()
                .flat_map(|curve| {
                    (0..=20).map(move |step| {
                        (ellipse_radius(&center, curve.eval(f64::from(step) / 20.0)) - 1.0).abs()
                    })
                })
                .fold(0.0_f64, f64::max)
        };
        let coarse = worst(FRAC_PI_2);
        let fine = worst(FRAC_PI_2 / 4.0);
        assert!(coarse < 1e-3, "a quarter-turn piece stays under 1e-3");
        assert!(fine < coarse / 100.0, "{fine} should be far below {coarse}");
    }

    #[test]
    fn the_bounding_box_of_a_lower_semicircle_is_exact() {
        let arc = ArcParams::new(
            Point::new(-1.0, 0.0),
            1.0,
            1.0,
            0.0,
            false,
            true,
            Point::new(1.0, 0.0),
        );
        let center = center_of(&arc);
        let bounds = center.tight_bounds();
        assert_close(bounds.x, -1.0, 1e-12);
        assert_close(bounds.y, -1.0, 1e-12);
        assert_close(bounds.width, 2.0, 1e-12);
        assert_close(bounds.height, 1.0, 1e-12);
    }

    #[test]
    fn the_bounding_box_of_an_upper_semicircle_is_the_mirror_image() {
        let arc = ArcParams::new(
            Point::new(-1.0, 0.0),
            1.0,
            1.0,
            0.0,
            false,
            false,
            Point::new(1.0, 0.0),
        );
        let bounds = center_of(&arc).tight_bounds();
        assert_close(bounds.x, -1.0, 1e-12);
        assert_close(bounds.y, 0.0, 1e-12);
        assert_close(bounds.width, 2.0, 1e-12);
        assert_close(bounds.height, 1.0, 1e-12);
    }

    #[test]
    fn the_bounding_box_of_a_whole_tilted_ellipse_matches_the_closed_form() {
        // A full turn of a 2x1 ellipse tilted 45 degrees: the box's half
        // width is sqrt(rx²cos²phi + ry²sin²phi) and its half height is
        // sqrt(rx²sin²phi + ry²cos²phi), both sqrt(2.5) here.
        let center = CenterParams::new(0.0, 0.0, 2.0, 1.0, 45.0_f64.to_radians(), 0.0, TAU);
        let bounds = center.tight_bounds();
        let half = 2.5_f64.sqrt();
        assert_close(bounds.x, -half, 1e-12);
        assert_close(bounds.y, -half, 1e-12);
        assert_close(bounds.width, 2.0 * half, 1e-12);
        assert_close(bounds.height, 2.0 * half, 1e-12);
    }

    #[test]
    fn the_exact_bounding_box_agrees_with_the_bezier_approximation() {
        let cases = [
            ArcParams::new(
                Point::new(-3.0, 7.5),
                4.0,
                2.0,
                37.0,
                true,
                true,
                Point::new(2.0, -1.25),
            ),
            ArcParams::new(
                Point::new(0.0, 0.0),
                5.0,
                5.0,
                0.0,
                false,
                false,
                Point::new(3.0, 4.0),
            ),
            ArcParams::new(
                Point::new(10.0, 0.0),
                10.0,
                4.0,
                110.0,
                true,
                false,
                Point::new(-4.0, 2.0),
            ),
        ];
        for arc in cases {
            let center = center_of(&arc);
            let exact = center.tight_bounds();
            let approximate = arc
                .to_cubics()
                .iter()
                .map(CubicBezier::tight_bounds)
                .reduce(|a, b| a.union(&b))
                .expect("a real arc yields at least one curve");
            for (exact_value, approximate_value) in [
                (exact.x, approximate.x),
                (exact.y, approximate.y),
                (exact.max_x(), approximate.max_x()),
                (exact.max_y(), approximate.max_y()),
            ] {
                assert_close(approximate_value, exact_value, 1e-2);
            }
            // The approximation can only ever fall inside the true box by a
            // hair, never wildly outside it.
            assert!(exact.width >= approximate.width - 1e-2);
            assert!(exact.height >= approximate.height - 1e-2);
        }
    }

    #[test]
    fn contains_angle_follows_the_sweep_direction() {
        let clockwise = CenterParams::new(0.0, 0.0, 1.0, 1.0, 0.0, 0.0, FRAC_PI_2);
        assert!(clockwise.contains_angle(0.0));
        assert!(clockwise.contains_angle(FRAC_PI_2));
        assert!(clockwise.contains_angle(0.3));
        assert!(clockwise.contains_angle(TAU));
        assert!(!clockwise.contains_angle(-0.3));
        assert!(!clockwise.contains_angle(PI));

        let anticlockwise = CenterParams::new(0.0, 0.0, 1.0, 1.0, 0.0, 0.0, -FRAC_PI_2);
        assert!(anticlockwise.contains_angle(0.0));
        assert!(anticlockwise.contains_angle(-0.3));
        assert!(anticlockwise.contains_angle(-FRAC_PI_2));
        assert!(!anticlockwise.contains_angle(0.3));
    }

    #[test]
    fn the_angle_function_is_guarded_against_rounding_and_zero_vectors() {
        // Parallel vectors: the cosine rounds to either side of 1, and the
        // side above it would be a NaN without the clamp. The answer is
        // never exactly zero — `acos` loses half its precision near 1, so a
        // ratio one ulp below 1 already reports 1.5e-8 — but it is always a
        // number, which is what the conversion needs.
        for scale in [1e-8, 1e-3, 1.0, 1e3, 1e8] {
            for (x, y) in [(1.0, 0.0), (0.0, 1.0), (1.0, 1.0), (0.3, -0.7)] {
                let angle = angle_between(x * scale, y * scale, x * 3.0, y * 3.0);
                assert!(angle.is_finite(), "({x}, {y}) at {scale} gave {angle}");
                assert!(angle.abs() < 1e-7, "({x}, {y}) at {scale} gave {angle}");
            }
        }
        assert_eq!(angle_between(0.0, 0.0, 1.0, 0.0), 0.0);
        assert_eq!(angle_between(1.0, 0.0, 0.0, 0.0), 0.0);
        assert_close(angle_between(1.0, 0.0, 0.0, 1.0), FRAC_PI_2, 1e-12);
        assert_close(angle_between(1.0, 0.0, 0.0, -1.0), -FRAC_PI_2, 1e-12);
        assert_close(angle_between(1.0, 0.0, -1.0, 0.0), PI, 1e-12);
        // Antiparallel vectors sit at the other end of `acos`'s range, where
        // the clamp matters just as much and the same half-precision loss
        // applies: the answer is a hair under pi rather than exactly pi.
        assert_close(angle_between(2.0, 4.0, -1.0, -2.0), PI, 1e-7);
    }

    #[test]
    fn the_derivative_carries_the_radii_rather_than_being_normalized() {
        // On a 3x1 ellipse the tangent at theta = 0 points along +y with
        // length ry, and at theta = pi/2 along -x with length rx. A
        // normalized tangent would report length 1 for both, and would make
        // the cubic conversion wrong for every non-circular ellipse.
        let center = CenterParams::new(0.0, 0.0, 3.0, 1.0, 0.0, 0.0, FRAC_PI_2);
        assert_point_close(center.derivative_at(0.0), Point::new(0.0, 1.0), 1e-12);
        assert_point_close(
            center.derivative_at(FRAC_PI_2),
            Point::new(-3.0, 0.0),
            1e-12,
        );
    }

    #[test]
    fn theta2_and_the_endpoints_agree_with_point_at() {
        let center = CenterParams::new(1.0, 2.0, 3.0, 4.0, 0.5, 0.25, 1.0);
        assert_close(center.theta2(), 1.25, 1e-15);
        assert_eq!(center.start_point(), center.point_at(0.25));
        assert_eq!(center.end_point(), center.point_at(1.25));
        assert_eq!(center.center(), Point::new(1.0, 2.0));
        assert!(center.is_finite());
        assert!(!CenterParams::new(f64::NAN, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0).is_finite());
    }
}
