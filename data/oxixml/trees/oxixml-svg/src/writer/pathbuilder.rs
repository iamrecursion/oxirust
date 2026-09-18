//! A fluent, consuming builder for [`PathData`].
//!
//! [`PathBuilder`] mirrors the `PathBuilder` of the pandrs SVG engine (the
//! ergonomic benchmark this crate's writer module is built to meet and
//! beat): each command method takes `self` by value and returns `Self`, so a
//! path is written as one unbroken chain ending in [`PathBuilder::finish`].
//! Every command of the `d` microsyntax (SVG 1.1 §8.3, SVG 2 §9.3.9) has an
//! absolute method and a `_rel` relative counterpart, matching the `abs`
//! field [`PathSegment`] itself carries — there is no `rel: bool` parameter
//! to get backwards, just a differently named method for each case.
//!
//! ```
//! use oxixml_svg::writer::pathbuilder::PathBuilder;
//!
//! let path = PathBuilder::new()
//!     .move_to(0.0, 0.0)
//!     .line_to(10.0, 0.0)
//!     .line_to_rel(0.0, 10.0)
//!     .close()
//!     .finish();
//! assert_eq!(path.to_string(), "M 0 0 L 10 0 l 0 10 Z");
//! ```

use crate::values::path::{PathData, PathSegment};

/// A fluent, consuming builder that accumulates [`PathSegment`]s and
/// produces a [`PathData`].
///
/// See the [module documentation](self) for the absolute/relative method
/// naming convention and the overall chaining style.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct PathBuilder {
    segments: Vec<PathSegment>,
}

impl PathBuilder {
    /// Creates an empty path builder.
    #[must_use]
    pub const fn new() -> Self {
        Self {
            segments: Vec::new(),
        }
    }

    /// Creates an empty path builder with room for `capacity` segments.
    #[must_use]
    pub fn with_capacity(capacity: usize) -> Self {
        Self {
            segments: Vec::with_capacity(capacity),
        }
    }

    /// `M x y` — starts a new subpath at the absolute point `(x, y)`.
    #[must_use]
    pub fn move_to(mut self, x: f64, y: f64) -> Self {
        self.segments.push(PathSegment::MoveTo { abs: true, x, y });
        self
    }

    /// `m dx dy` — starts a new subpath at `(dx, dy)` relative to the
    /// current point.
    #[must_use]
    pub fn move_to_rel(mut self, dx: f64, dy: f64) -> Self {
        self.segments.push(PathSegment::MoveTo {
            abs: false,
            x: dx,
            y: dy,
        });
        self
    }

    /// `L x y` — draws a straight line to the absolute point `(x, y)`.
    #[must_use]
    pub fn line_to(mut self, x: f64, y: f64) -> Self {
        self.segments.push(PathSegment::LineTo { abs: true, x, y });
        self
    }

    /// `l dx dy` — draws a straight line to `(dx, dy)` relative to the
    /// current point.
    #[must_use]
    pub fn line_to_rel(mut self, dx: f64, dy: f64) -> Self {
        self.segments.push(PathSegment::LineTo {
            abs: false,
            x: dx,
            y: dy,
        });
        self
    }

    /// `H x` — draws a horizontal line to the absolute x coordinate `x`,
    /// keeping the current y coordinate.
    #[must_use]
    pub fn hline_to(mut self, x: f64) -> Self {
        self.segments
            .push(PathSegment::HorizontalTo { abs: true, x });
        self
    }

    /// `h dx` — draws a horizontal line `dx` relative to the current x
    /// coordinate, keeping the current y coordinate.
    #[must_use]
    pub fn hline_to_rel(mut self, dx: f64) -> Self {
        self.segments
            .push(PathSegment::HorizontalTo { abs: false, x: dx });
        self
    }

    /// `V y` — draws a vertical line to the absolute y coordinate `y`,
    /// keeping the current x coordinate.
    #[must_use]
    pub fn vline_to(mut self, y: f64) -> Self {
        self.segments.push(PathSegment::VerticalTo { abs: true, y });
        self
    }

    /// `v dy` — draws a vertical line `dy` relative to the current y
    /// coordinate, keeping the current x coordinate.
    #[must_use]
    pub fn vline_to_rel(mut self, dy: f64) -> Self {
        self.segments
            .push(PathSegment::VerticalTo { abs: false, y: dy });
        self
    }

    /// `C x1 y1 x2 y2 x y` — draws a cubic Bézier curve to the absolute end
    /// point `(x, y)`, using `(x1, y1)` and `(x2, y2)` as its control
    /// points.
    #[must_use]
    pub fn curve_to(mut self, x1: f64, y1: f64, x2: f64, y2: f64, x: f64, y: f64) -> Self {
        self.segments.push(PathSegment::CurveTo {
            abs: true,
            x1,
            y1,
            x2,
            y2,
            x,
            y,
        });
        self
    }

    /// `c dx1 dy1 dx2 dy2 dx dy` — draws a cubic Bézier curve to
    /// `(dx, dy)` relative to the current point, using control points also
    /// given relative to the current point.
    #[must_use]
    pub fn curve_to_rel(
        mut self,
        dx1: f64,
        dy1: f64,
        dx2: f64,
        dy2: f64,
        dx: f64,
        dy: f64,
    ) -> Self {
        self.segments.push(PathSegment::CurveTo {
            abs: false,
            x1: dx1,
            y1: dy1,
            x2: dx2,
            y2: dy2,
            x: dx,
            y: dy,
        });
        self
    }

    /// `S x2 y2 x y` — draws a cubic Bézier curve to the absolute end point
    /// `(x, y)`, reflecting the previous curve's second control point to
    /// synthesize the first one and using `(x2, y2)` as the second.
    #[must_use]
    pub fn smooth_curve_to(mut self, x2: f64, y2: f64, x: f64, y: f64) -> Self {
        self.segments.push(PathSegment::SmoothCurveTo {
            abs: true,
            x2,
            y2,
            x,
            y,
        });
        self
    }

    /// `s dx2 dy2 dx dy` — the relative form of [`PathBuilder::smooth_curve_to`].
    #[must_use]
    pub fn smooth_curve_to_rel(mut self, dx2: f64, dy2: f64, dx: f64, dy: f64) -> Self {
        self.segments.push(PathSegment::SmoothCurveTo {
            abs: false,
            x2: dx2,
            y2: dy2,
            x: dx,
            y: dy,
        });
        self
    }

    /// `Q x1 y1 x y` — draws a quadratic Bézier curve to the absolute end
    /// point `(x, y)`, using `(x1, y1)` as its control point.
    #[must_use]
    pub fn quad_to(mut self, x1: f64, y1: f64, x: f64, y: f64) -> Self {
        self.segments.push(PathSegment::QuadTo {
            abs: true,
            x1,
            y1,
            x,
            y,
        });
        self
    }

    /// `q dx1 dy1 dx dy` — the relative form of [`PathBuilder::quad_to`].
    #[must_use]
    pub fn quad_to_rel(mut self, dx1: f64, dy1: f64, dx: f64, dy: f64) -> Self {
        self.segments.push(PathSegment::QuadTo {
            abs: false,
            x1: dx1,
            y1: dy1,
            x: dx,
            y: dy,
        });
        self
    }

    /// `T x y` — draws a quadratic Bézier curve to the absolute end point
    /// `(x, y)`, reflecting the previous quadratic curve's control point to
    /// synthesize this one's.
    #[must_use]
    pub fn smooth_quad_to(mut self, x: f64, y: f64) -> Self {
        self.segments
            .push(PathSegment::SmoothQuadTo { abs: true, x, y });
        self
    }

    /// `t dx dy` — the relative form of [`PathBuilder::smooth_quad_to`].
    #[must_use]
    pub fn smooth_quad_to_rel(mut self, dx: f64, dy: f64) -> Self {
        self.segments.push(PathSegment::SmoothQuadTo {
            abs: false,
            x: dx,
            y: dy,
        });
        self
    }

    /// `A rx ry x-axis-rotation large-arc sweep x y` — draws an elliptical
    /// arc to the absolute end point `(x, y)`.
    ///
    /// `rx`/`ry` and `x_axis_rotation` are stored exactly as given, with no
    /// normalization: a negative radius, a zero radius, or an
    /// `x_axis_rotation` outside `0..360` are all accepted here and left for
    /// the geometry layer to normalize per SVG 1.1 Appendix F.6.2, exactly as
    /// [`PathSegment::ArcTo`] itself documents.
    // The elliptical-arc command has exactly seven arguments in the SVG
    // grammar itself (SVG 1.1 §8.3.8); grouping them into an ad hoc params
    // struct would only add ceremony around a fixed, spec-defined shape that
    // every other language's SVG arc API also spells out positionally.
    #[must_use]
    #[allow(clippy::too_many_arguments)]
    pub fn arc_to(
        mut self,
        rx: f64,
        ry: f64,
        x_axis_rotation: f64,
        large_arc: bool,
        sweep: bool,
        x: f64,
        y: f64,
    ) -> Self {
        self.segments.push(PathSegment::ArcTo {
            abs: true,
            rx,
            ry,
            x_axis_rotation,
            large_arc,
            sweep,
            x,
            y,
        });
        self
    }

    /// `a rx ry x-axis-rotation large-arc sweep dx dy` — the relative form
    /// of [`PathBuilder::arc_to`].
    #[must_use]
    #[allow(clippy::too_many_arguments)]
    pub fn arc_to_rel(
        mut self,
        rx: f64,
        ry: f64,
        x_axis_rotation: f64,
        large_arc: bool,
        sweep: bool,
        dx: f64,
        dy: f64,
    ) -> Self {
        self.segments.push(PathSegment::ArcTo {
            abs: false,
            rx,
            ry,
            x_axis_rotation,
            large_arc,
            sweep,
            x: dx,
            y: dy,
        });
        self
    }

    /// `Z` — closes the current subpath with a straight line back to its
    /// start point. `Z` and `z` are equivalent, so there is no `_rel`
    /// counterpart.
    #[must_use]
    pub fn close(mut self) -> Self {
        self.segments.push(PathSegment::Close);
        self
    }

    /// The number of commands accumulated so far.
    #[must_use]
    pub fn len(&self) -> usize {
        self.segments.len()
    }

    /// True when no command has been accumulated yet.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.segments.is_empty()
    }

    /// Consumes the builder, returning the accumulated commands as a
    /// [`PathData`].
    #[must_use]
    pub fn finish(self) -> PathData {
        PathData::from_segments(self.segments)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn empty_builder_is_empty_and_finishes_to_an_empty_path() {
        let builder = PathBuilder::new();
        assert!(builder.is_empty());
        assert_eq!(builder.len(), 0);
        assert_eq!(builder.finish(), PathData::new());
    }

    #[test]
    fn with_capacity_starts_empty_too() {
        assert!(PathBuilder::with_capacity(8).is_empty());
    }

    #[test]
    fn a_simple_triangle_renders_the_expected_canonical_path() {
        let path = PathBuilder::new()
            .move_to(0.0, 0.0)
            .line_to(100.0, 0.0)
            .line_to(100.0, 100.0)
            .close()
            .finish();
        assert_eq!(path.to_string(), "M 0 0 L 100 0 L 100 100 Z");
        assert_eq!(path.len(), 4);
    }

    #[test]
    fn relative_variants_carry_abs_false_and_display_lowercase() {
        let path = PathBuilder::new()
            .move_to(10.0, 20.0)
            .line_to_rel(5.0, 5.0)
            .hline_to_rel(5.0)
            .vline_to_rel(-5.0)
            .close()
            .finish();
        assert_eq!(path.to_string(), "M 10 20 l 5 5 h 5 v -5 Z");
    }

    #[test]
    fn absolute_variants_carry_abs_true_and_display_uppercase() {
        let path = PathBuilder::new()
            .move_to(0.0, 0.0)
            .hline_to(10.0)
            .vline_to(10.0)
            .finish();
        assert_eq!(path.to_string(), "M 0 0 H 10 V 10");
    }

    #[test]
    fn cubic_and_smooth_cubic_curves_round_trip_their_fields() {
        let path = PathBuilder::new()
            .move_to(0.0, 0.0)
            .curve_to(1.0, 2.0, 3.0, 4.0, 5.0, 6.0)
            .smooth_curve_to(7.0, 8.0, 9.0, 10.0)
            .curve_to_rel(1.0, 1.0, 2.0, 2.0, 3.0, 3.0)
            .smooth_curve_to_rel(1.0, 1.0, 2.0, 2.0)
            .finish();
        assert_eq!(
            path.segments()[1],
            PathSegment::CurveTo {
                abs: true,
                x1: 1.0,
                y1: 2.0,
                x2: 3.0,
                y2: 4.0,
                x: 5.0,
                y: 6.0,
            }
        );
        assert_eq!(
            path.segments()[2],
            PathSegment::SmoothCurveTo {
                abs: true,
                x2: 7.0,
                y2: 8.0,
                x: 9.0,
                y: 10.0,
            }
        );
        assert!(matches!(
            path.segments()[3],
            PathSegment::CurveTo { abs: false, .. }
        ));
        assert!(matches!(
            path.segments()[4],
            PathSegment::SmoothCurveTo { abs: false, .. }
        ));
    }

    #[test]
    fn quadratic_and_smooth_quadratic_curves_round_trip_their_fields() {
        let path = PathBuilder::new()
            .move_to(0.0, 0.0)
            .quad_to(1.0, 2.0, 3.0, 4.0)
            .smooth_quad_to(5.0, 6.0)
            .quad_to_rel(1.0, 1.0, 2.0, 2.0)
            .smooth_quad_to_rel(1.0, 1.0)
            .finish();
        assert_eq!(
            path.segments()[1],
            PathSegment::QuadTo {
                abs: true,
                x1: 1.0,
                y1: 2.0,
                x: 3.0,
                y: 4.0,
            }
        );
        assert_eq!(
            path.segments()[2],
            PathSegment::SmoothQuadTo {
                abs: true,
                x: 5.0,
                y: 6.0,
            }
        );
        assert!(matches!(
            path.segments()[3],
            PathSegment::QuadTo { abs: false, .. }
        ));
        assert!(matches!(
            path.segments()[4],
            PathSegment::SmoothQuadTo { abs: false, .. }
        ));
    }

    #[test]
    fn arc_to_keeps_radii_rotation_and_flags_exactly_as_given() {
        let path = PathBuilder::new()
            .move_to(0.0, 0.0)
            .arc_to(-3.0, 4.0, 450.0, true, false, 10.0, 10.0)
            .arc_to_rel(1.0, 1.0, 0.0, false, true, 1.0, 1.0)
            .finish();
        assert_eq!(
            path.segments()[1],
            PathSegment::ArcTo {
                abs: true,
                rx: -3.0,
                ry: 4.0,
                x_axis_rotation: 450.0,
                large_arc: true,
                sweep: false,
                x: 10.0,
                y: 10.0,
            }
        );
        assert_eq!(
            path.segments()[2],
            PathSegment::ArcTo {
                abs: false,
                rx: 1.0,
                ry: 1.0,
                x_axis_rotation: 0.0,
                large_arc: false,
                sweep: true,
                x: 1.0,
                y: 1.0,
            }
        );
    }

    #[test]
    fn close_appends_a_single_close_segment_with_no_relative_form() {
        let path = PathBuilder::new().move_to(0.0, 0.0).close().finish();
        assert_eq!(path.segments()[1], PathSegment::Close);
    }

    #[test]
    fn finish_output_reparses_to_the_same_segments() {
        let path = PathBuilder::new()
            .move_to(0.0, 0.0)
            .line_to(10.0, 0.0)
            .curve_to(10.0, 10.0, 0.0, 10.0, 0.0, 0.0)
            .close()
            .finish();
        let reparsed =
            PathData::parse(&path.to_string()).expect("builder output is valid path data");
        assert_eq!(reparsed, path);
    }

    #[test]
    fn the_documented_example_holds() {
        let path = PathBuilder::new()
            .move_to(0.0, 0.0)
            .line_to(10.0, 0.0)
            .line_to_rel(0.0, 10.0)
            .close()
            .finish();
        assert_eq!(path.to_string(), "M 0 0 L 10 0 l 0 10 Z");
    }
}
