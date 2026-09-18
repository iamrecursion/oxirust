//! Points and axis-aligned rectangles.
//!
//! [`Point`] and [`Rect`] are the plain coordinate types the rest of the
//! geometry toolkit (Bézier/arc evaluation, path bounding boxes, the
//! `viewBox`-to-viewport algorithm) is built from. They are deliberately
//! kept independent of [`crate::geometry::matrix::Matrix`]: this module has
//! no notion of transforming a point or a rectangle by a matrix.
//! Composing a [`Matrix`](crate::geometry::matrix::Matrix) with a [`Point`]
//! or [`Rect`] is the job of the module that actually needs that
//! composition (for example the `viewBox`-to-viewport transform), which
//! should call [`Matrix::apply`](crate::geometry::matrix::Matrix::apply) on
//! the raw `(x, y)` pair and wrap the result back into a [`Point`] itself.

use std::fmt;

/// A single point in a 2D coordinate system.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Point {
    /// The x coordinate.
    pub x: f64,
    /// The y coordinate.
    pub y: f64,
}

impl Point {
    /// Creates a point at `(x, y)`.
    #[must_use]
    pub const fn new(x: f64, y: f64) -> Self {
        Self { x, y }
    }
}

impl Default for Point {
    /// The origin, `(0, 0)`.
    fn default() -> Self {
        Self::new(0.0, 0.0)
    }
}

impl fmt::Display for Point {
    /// Formats the point as `x,y`, the coordinate-pair form used throughout
    /// the SVG microsyntaxes (`points`, path data, …).
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{},{}", self.x, self.y)
    }
}

/// An axis-aligned rectangle: an origin `(x, y)` plus a non-negative `width`
/// and `height`.
///
/// `Rect` is also used throughout this crate as a **bounding-box
/// accumulator**: since a rectangle has no natural "empty/absent" value
/// that is distinct from a valid zero-sized rectangle at the origin, the
/// accumulation pattern is built on `Option<Rect>` rather than on a sentinel
/// `Rect` value. Starting from `None` and folding in points one at a time
/// looks like:
///
/// ```
/// use oxixml_svg::geometry::primitives::{Point, Rect};
///
/// let points = [Point::new(3.0, 1.0), Point::new(-2.0, 5.0), Point::new(0.0, 0.0)];
/// let mut bbox: Option<Rect> = None;
/// for &p in &points {
///     bbox = Some(match bbox {
///         None => Rect::from_point(p),
///         Some(r) => r.expand_to_include(p),
///     });
/// }
/// let bbox = bbox.expect("at least one point was folded in");
/// assert_eq!(bbox, Rect::from_points(Point::new(-2.0, 0.0), Point::new(3.0, 5.0)));
/// ```
///
/// Two already-computed rectangles (for example, the bounding boxes of two
/// sibling path segments) are combined with [`Rect::union`] instead.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Rect {
    /// The x coordinate of the rectangle's origin (its minimum x edge).
    pub x: f64,
    /// The y coordinate of the rectangle's origin (its minimum y edge).
    pub y: f64,
    /// The rectangle's width. Never negative.
    pub width: f64,
    /// The rectangle's height. Never negative.
    pub height: f64,
}

impl Rect {
    /// Builds a rectangle from an origin and a size, or `None` if `width`
    /// or `height` is negative.
    ///
    /// A `width`/`height` of exactly `0` is accepted: a zero-sized
    /// rectangle is a valid (if degenerate) value, matching the SVG
    /// specification's own treatment of a zero-width or zero-height
    /// `viewBox` as grammar-valid.
    #[must_use]
    pub fn new(x: f64, y: f64, width: f64, height: f64) -> Option<Self> {
        if width < 0.0 || height < 0.0 {
            None
        } else {
            Some(Self {
                x,
                y,
                width,
                height,
            })
        }
    }

    /// An alias of [`Rect::new`], for callers that prefer the explicit
    /// "x, y, width, height" name over the generic "new".
    #[must_use]
    pub fn from_xywh(x: f64, y: f64, width: f64, height: f64) -> Option<Self> {
        Self::new(x, y, width, height)
    }

    /// Builds the smallest rectangle that contains both `a` and `b`,
    /// treating them as opposite corners in either order.
    ///
    /// Unlike [`Rect::new`], this always succeeds: the width and height are
    /// derived as absolute differences, so they are never negative.
    #[must_use]
    pub fn from_points(a: Point, b: Point) -> Self {
        let x = a.x.min(b.x);
        let y = a.y.min(b.y);
        Self {
            x,
            y,
            width: (a.x - b.x).abs(),
            height: (a.y - b.y).abs(),
        }
    }

    /// Builds a zero-sized rectangle located exactly at `point`.
    ///
    /// This is the natural seed for the bounding-box accumulator pattern
    /// documented on [`Rect`] itself: it is equivalent to
    /// `Rect::from_points(point, point)`.
    #[must_use]
    pub fn from_point(point: Point) -> Self {
        Self::from_points(point, point)
    }

    /// The rectangle's maximum x edge, `x + width`.
    #[must_use]
    pub fn max_x(&self) -> f64 {
        self.x + self.width
    }

    /// The rectangle's maximum y edge, `y + height`.
    #[must_use]
    pub fn max_y(&self) -> f64 {
        self.y + self.height
    }

    /// True when the rectangle has zero area (a zero `width` or `height`).
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.width <= 0.0 || self.height <= 0.0
    }

    /// True when `point` lies within the rectangle, inclusive of its edges.
    #[must_use]
    pub fn contains(&self, point: Point) -> bool {
        point.x >= self.x && point.x <= self.max_x() && point.y >= self.y && point.y <= self.max_y()
    }

    /// True when `self` and `other` share at least one point, inclusive of
    /// touching edges (two rectangles that only share a boundary line or
    /// corner are considered intersecting).
    #[must_use]
    pub fn intersects(&self, other: &Rect) -> bool {
        self.x <= other.max_x()
            && other.x <= self.max_x()
            && self.y <= other.max_y()
            && other.y <= self.max_y()
    }

    /// The smallest rectangle that contains both `self` and `other`.
    #[must_use]
    pub fn union(&self, other: &Rect) -> Rect {
        let x = self.x.min(other.x);
        let y = self.y.min(other.y);
        let max_x = self.max_x().max(other.max_x());
        let max_y = self.max_y().max(other.max_y());
        Rect {
            x,
            y,
            width: max_x - x,
            height: max_y - y,
        }
    }

    /// The smallest rectangle that contains both `self` and `point`.
    ///
    /// This is the per-point step of the bounding-box accumulator pattern
    /// documented on [`Rect`] itself; it is equivalent to
    /// `self.union(&Rect::from_point(point))`.
    #[must_use]
    pub fn expand_to_include(&self, point: Point) -> Rect {
        self.union(&Rect::from_point(point))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn point_new_and_default() {
        let p = Point::new(1.5, -2.5);
        assert_eq!(p.x, 1.5);
        assert_eq!(p.y, -2.5);
        assert_eq!(Point::default(), Point::new(0.0, 0.0));
    }

    #[test]
    fn point_display_is_a_comma_separated_pair() {
        assert_eq!(Point::new(1.0, 2.5).to_string(), "1,2.5");
        assert_eq!(Point::new(-3.0, 0.0).to_string(), "-3,0");
    }

    #[test]
    fn rect_new_rejects_negative_width_or_height() {
        assert_eq!(Rect::new(0.0, 0.0, -1.0, 5.0), None);
        assert_eq!(Rect::new(0.0, 0.0, 5.0, -1.0), None);
        assert!(Rect::new(0.0, 0.0, 0.0, 0.0).is_some());
    }

    #[test]
    fn rect_new_accepts_zero_size() {
        let r = Rect::new(1.0, 2.0, 0.0, 0.0).expect("zero size is grammar-valid");
        assert!(r.is_empty());
    }

    #[test]
    fn from_xywh_is_an_alias_of_new() {
        assert_eq!(
            Rect::from_xywh(1.0, 2.0, 3.0, 4.0),
            Rect::new(1.0, 2.0, 3.0, 4.0)
        );
        assert_eq!(Rect::from_xywh(0.0, 0.0, -1.0, 1.0), None);
    }

    #[test]
    fn from_points_normalizes_either_corner_order() {
        let a = Point::new(3.0, 5.0);
        let b = Point::new(-1.0, 2.0);
        let expected = Rect {
            x: -1.0,
            y: 2.0,
            width: 4.0,
            height: 3.0,
        };
        assert_eq!(Rect::from_points(a, b), expected);
        assert_eq!(Rect::from_points(b, a), expected);
    }

    #[test]
    fn from_point_is_a_zero_sized_rect_at_that_point() {
        let r = Rect::from_point(Point::new(4.0, -2.0));
        assert_eq!(r.x, 4.0);
        assert_eq!(r.y, -2.0);
        assert_eq!(r.width, 0.0);
        assert_eq!(r.height, 0.0);
        assert!(r.is_empty());
    }

    #[test]
    fn max_x_and_max_y_are_the_far_edges() {
        let r = Rect::new(1.0, 2.0, 3.0, 4.0).expect("valid rect");
        assert_eq!(r.max_x(), 4.0);
        assert_eq!(r.max_y(), 6.0);
    }

    #[test]
    fn contains_is_inclusive_of_edges() {
        let r = Rect::new(0.0, 0.0, 10.0, 10.0).expect("valid rect");
        assert!(r.contains(Point::new(0.0, 0.0)));
        assert!(r.contains(Point::new(10.0, 10.0)));
        assert!(r.contains(Point::new(5.0, 5.0)));
        assert!(!r.contains(Point::new(10.001, 5.0)));
        assert!(!r.contains(Point::new(-0.001, 5.0)));
    }

    #[test]
    fn intersects_is_inclusive_of_touching_edges() {
        let a = Rect::new(0.0, 0.0, 10.0, 10.0).expect("valid rect");
        let touching = Rect::new(10.0, 0.0, 5.0, 5.0).expect("valid rect");
        let overlapping = Rect::new(5.0, 5.0, 10.0, 10.0).expect("valid rect");
        let disjoint = Rect::new(20.0, 20.0, 1.0, 1.0).expect("valid rect");

        assert!(a.intersects(&touching));
        assert!(a.intersects(&overlapping));
        assert!(!a.intersects(&disjoint));
    }

    #[test]
    fn union_covers_both_rectangles() {
        let a = Rect::new(0.0, 0.0, 2.0, 2.0).expect("valid rect");
        let b = Rect::new(5.0, -3.0, 1.0, 1.0).expect("valid rect");
        let u = a.union(&b);
        assert_eq!(u.x, 0.0);
        assert_eq!(u.y, -3.0);
        assert_eq!(u.max_x(), 6.0);
        assert_eq!(u.max_y(), 2.0);
        assert!(u.contains(Point::new(0.0, 0.0)));
        assert!(u.contains(Point::new(5.5, -2.5)));
    }

    #[test]
    fn union_is_commutative() {
        let a = Rect::new(0.0, 0.0, 2.0, 3.0).expect("valid rect");
        let b = Rect::new(-4.0, -4.0, 1.0, 1.0).expect("valid rect");
        assert_eq!(a.union(&b), b.union(&a));
    }

    #[test]
    fn expand_to_include_grows_to_cover_a_new_point() {
        let r = Rect::new(0.0, 0.0, 2.0, 2.0).expect("valid rect");
        let grown = r.expand_to_include(Point::new(5.0, -1.0));
        assert_eq!(grown.x, 0.0);
        assert_eq!(grown.y, -1.0);
        assert_eq!(grown.max_x(), 5.0);
        assert_eq!(grown.max_y(), 2.0);
        assert!(grown.contains(Point::new(0.0, 0.0)));
    }

    #[test]
    fn expand_to_include_is_a_no_op_for_an_interior_point() {
        let r = Rect::new(0.0, 0.0, 10.0, 10.0).expect("valid rect");
        assert_eq!(r.expand_to_include(Point::new(5.0, 5.0)), r);
    }

    /// The bounding-box accumulator pattern documented on [`Rect`]: start
    /// from `None`, and fold points in one at a time.
    #[test]
    fn bounding_box_accumulator_pattern_from_none() {
        let points = [
            Point::new(3.0, 1.0),
            Point::new(-2.0, 5.0),
            Point::new(0.0, 0.0),
        ];
        let mut bbox: Option<Rect> = None;
        for &p in &points {
            bbox = Some(match bbox {
                None => Rect::from_point(p),
                Some(r) => r.expand_to_include(p),
            });
        }
        let bbox = bbox.expect("at least one point was folded in");
        assert_eq!(
            bbox,
            Rect::from_points(Point::new(-2.0, 0.0), Point::new(3.0, 5.0))
        );
    }

    /// Pins the accumulator path (`from_point` + `expand_to_include`, used
    /// when folding many points one at a time) against the direct
    /// two-corner path (`from_points`, used when only two points are ever
    /// known at once): they must agree for every corner ordering, since
    /// path bounding-box code (Bézier/arc extrema) mixes both styles.
    #[test]
    fn expand_to_include_agrees_with_from_points() {
        for (p, q) in [
            (Point::new(0.0, 0.0), Point::new(5.0, 3.0)),
            (Point::new(-4.0, 7.0), Point::new(2.0, -1.0)),
            (Point::new(5.0, 3.0), Point::new(0.0, 0.0)),
            (Point::new(3.0, 3.0), Point::new(3.0, 3.0)),
        ] {
            let via_accumulator = Rect::from_point(p).expand_to_include(q);
            let via_from_points = Rect::from_points(p, q);
            assert_eq!(
                via_accumulator, via_from_points,
                "mismatch for p={p:?}, q={q:?}"
            );
        }
    }
}
