//! The `viewBox`-to-viewport transform.
//!
//! One algorithm does all the work here — SVG 1.1 §7.7 and SVG 2 §8.9,
//! implemented by [`view_box_transform`] — and everything else in this
//! module is a convenience over it. Given a `viewBox` rectangle in user
//! space, the viewport rectangle it must be mapped onto, and a
//! `preserveAspectRatio` value saying how to reconcile the two aspect
//! ratios, it produces the [`Matrix`] that maps the former onto the latter:
//!
//! ```text
//! scale-x = e-width  / vb-width
//! scale-y = e-height / vb-height
//! if align != none:
//!     meet:  scale-x = scale-y = min(scale-x, scale-y)
//!     slice: scale-x = scale-y = max(scale-x, scale-y)
//! translate-x = e-x - vb-x * scale-x   (+ centering per align)
//! translate-y = e-y - vb-y * scale-y   (+ centering per align)
//! transform   = translate(translate-x, translate-y) * scale(scale-x, scale-y)
//! ```
//!
//! # When there is no transform
//!
//! Every entry point returns an [`Option`], and `None` is not an error
//! condition so much as the specification's own "this element is not
//! rendered":
//!
//! * A `viewBox` with a zero width or height "disables rendering of the
//!   element" outright (SVG 1.1 §7.7). There is no sensible matrix — the
//!   scale factor would be infinite — so there is no matrix.
//! * A viewport with a zero width or height is the same story from the
//!   other side: SVG 2 §8.7 says a zero `width` or `height` on an `svg`
//!   element "disables rendering". **Callers computing a viewport from
//!   attributes should expect `None` for a zero-sized element**, rather
//!   than a degenerate zero-scale matrix.
//! * Non-finite input (a coordinate that overflowed `f64` while parsing)
//!   yields `None` rather than a matrix full of infinities.
//!
//! A *missing* `viewBox` is a different case entirely, and not a failure:
//! see [`nested_viewport_transform`], which then maps the element's own
//! origin and nothing else.
//!
//! # What this module deliberately does not do
//!
//! [`PreserveAspectRatio::defer`] is ignored. It is an SVG 1.1-only keyword
//! that means "if the referenced content has its own `preserveAspectRatio`,
//! let that one win" — a question about which element's attribute to read,
//! answered before this algorithm is ever reached, not about how to compute
//! a transform once the attribute has been chosen.
//!
//! Establishing viewports for `symbol`, `use`, `marker` and `pattern`
//! involves working out the viewport rectangle in the first place, which
//! needs the element tree, its computed lengths, and the referencing
//! element's bounding box. That belongs to the tree layer; this module
//! provides the primitive it composes with, through
//! [`nested_viewport_transform`].

use crate::geometry::matrix::Matrix;
use crate::geometry::primitives::Rect;
use crate::values::aspect::{Align, MeetOrSlice, PreserveAspectRatio, ViewBox};

/// Where the scaled `viewBox` sits inside the viewport when it does not fill
/// it: `0.0` for a minimum edge, `0.5` for a center line, `1.0` for a
/// maximum edge — per axis.
///
/// `None` is [`Align::None`], which scales the two axes independently so
/// that no leftover space exists to distribute in the first place.
///
/// The match is exhaustive over all ten [`Align`] variants on purpose: a
/// wildcard arm would silently treat any future alignment keyword as the
/// wrong one.
const fn alignment_factors(align: Align) -> Option<(f64, f64)> {
    match align {
        Align::None => None,
        Align::XMinYMin => Some((0.0, 0.0)),
        Align::XMidYMin => Some((0.5, 0.0)),
        Align::XMaxYMin => Some((1.0, 0.0)),
        Align::XMinYMid => Some((0.0, 0.5)),
        Align::XMidYMid => Some((0.5, 0.5)),
        Align::XMaxYMid => Some((1.0, 0.5)),
        Align::XMinYMax => Some((0.0, 1.0)),
        Align::XMidYMax => Some((0.5, 1.0)),
        Align::XMaxYMax => Some((1.0, 1.0)),
    }
}

/// The transform that maps `view_box` onto `viewport` under `par`.
///
/// This is the specification's algorithm verbatim; see the
/// [module documentation](self) for the formula and for the cases that
/// return `None` (a zero-sized `viewBox` or viewport, or non-finite input).
///
/// ```
/// use oxixml_svg::geometry::primitives::Rect;
/// use oxixml_svg::geometry::viewport::view_box_transform;
/// use oxixml_svg::values::aspect::{PreserveAspectRatio, ViewBox};
///
/// let view_box = ViewBox::new(0.0, 0.0, 30.0, 40.0);
/// let viewport = Rect::new(0.0, 0.0, 300.0, 400.0).expect("a positive size");
/// let par = PreserveAspectRatio::default();
///
/// let matrix = view_box_transform(&view_box, &viewport, &par).expect("both are renderable");
/// // Both axes scale by exactly 10, so the aspect ratios never disagree.
/// assert_eq!(matrix.to_string(), "matrix(10, 0, 0, 10, 0, 0)");
/// assert_eq!(matrix.apply(30.0, 40.0), (300.0, 400.0));
/// ```
#[must_use]
pub fn view_box_transform(
    view_box: &ViewBox,
    viewport: &Rect,
    par: &PreserveAspectRatio,
) -> Option<Matrix> {
    if !view_box.is_renderable() || viewport.is_empty() {
        return None;
    }
    if !(view_box.min_x.is_finite()
        && view_box.min_y.is_finite()
        && view_box.width.is_finite()
        && view_box.height.is_finite()
        && viewport.x.is_finite()
        && viewport.y.is_finite()
        && viewport.width.is_finite()
        && viewport.height.is_finite())
    {
        return None;
    }

    let mut scale_x = viewport.width / view_box.width;
    let mut scale_y = viewport.height / view_box.height;

    let factors = alignment_factors(par.align);
    if factors.is_some() {
        // Uniform scaling: `meet` fits the whole viewBox inside the
        // viewport and leaves slack on one axis, `slice` covers the whole
        // viewport and overflows on one axis.
        let uniform = match par.meet_or_slice {
            MeetOrSlice::Meet => scale_x.min(scale_y),
            MeetOrSlice::Slice => scale_x.max(scale_y),
        };
        scale_x = uniform;
        scale_y = uniform;
    }

    let mut translate_x = viewport.x - view_box.min_x * scale_x;
    let mut translate_y = viewport.y - view_box.min_y * scale_y;
    if let Some((factor_x, factor_y)) = factors {
        translate_x += (viewport.width - view_box.width * scale_x) * factor_x;
        translate_y += (viewport.height - view_box.height * scale_y) * factor_y;
    }

    Some(Matrix::translate(translate_x, translate_y).multiply(&Matrix::scale(scale_x, scale_y)))
}

/// The transform a viewport-establishing element contributes to its
/// children's coordinate system.
///
/// `viewport` is the element's viewport rectangle, already resolved in the
/// parent's user space: `(x, y, width, height)` for a nested `svg`, the
/// instantiating `use` element's rectangle for a `symbol`, the tile
/// rectangle for a `pattern`, and so on. Working that rectangle out is the
/// tree layer's job; mapping content into it is this one's.
///
/// With a `viewBox`, this is [`view_box_transform`]. Without one, a new
/// viewport still moves the origin — "a unit in the new user space equals a
/// unit in the current user space", but measured from the element's own
/// corner — so the answer is a plain translation, and `par` is irrelevant.
///
/// ```
/// use oxixml_svg::geometry::primitives::Rect;
/// use oxixml_svg::geometry::viewport::nested_viewport_transform;
/// use oxixml_svg::values::aspect::PreserveAspectRatio;
///
/// let viewport = Rect::new(20.0, 10.0, 100.0, 50.0).expect("a positive size");
/// let matrix = nested_viewport_transform(&viewport, None, &PreserveAspectRatio::default())
///     .expect("a renderable viewport");
/// // No viewBox: content is placed as-is, at the element's own origin.
/// assert_eq!(matrix.apply(0.0, 0.0), (20.0, 10.0));
/// ```
#[must_use]
pub fn nested_viewport_transform(
    viewport: &Rect,
    view_box: Option<&ViewBox>,
    par: &PreserveAspectRatio,
) -> Option<Matrix> {
    if viewport.is_empty() || !(viewport.x.is_finite() && viewport.y.is_finite()) {
        return None;
    }
    match view_box {
        Some(view_box) => view_box_transform(view_box, viewport, par),
        None => Some(Matrix::translate(viewport.x, viewport.y)),
    }
}

/// The transform an outermost `svg` element of the given size contributes.
///
/// A shorthand for [`nested_viewport_transform`] over the viewport
/// `(0, 0, width, height)`: the root viewport starts at the origin, so
/// without a `viewBox` the answer is the identity matrix, and with one it is
/// the full algorithm.
///
/// ```
/// use oxixml_svg::geometry::viewport::svg_viewport_transform;
/// use oxixml_svg::values::aspect::{PreserveAspectRatio, ViewBox};
///
/// let par = PreserveAspectRatio::default();
/// assert!(
///     svg_viewport_transform(100.0, 100.0, None, &par)
///         .expect("a renderable viewport")
///         .is_identity()
/// );
///
/// // A 100x100 viewBox in a 200x100 viewport, centered by the default
/// // xMidYMid meet: scaled by 1 and pushed 50 units right.
/// let view_box = ViewBox::new(0.0, 0.0, 100.0, 100.0);
/// let matrix = svg_viewport_transform(200.0, 100.0, Some(&view_box), &par)
///     .expect("a renderable viewport");
/// assert_eq!(matrix.to_string(), "matrix(1, 0, 0, 1, 50, 0)");
/// ```
#[must_use]
pub fn svg_viewport_transform(
    width: f64,
    height: f64,
    view_box: Option<&ViewBox>,
    par: &PreserveAspectRatio,
) -> Option<Matrix> {
    let viewport = Rect::new(0.0, 0.0, width, height)?;
    nested_viewport_transform(&viewport, view_box, par)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn rect(x: f64, y: f64, width: f64, height: f64) -> Rect {
        Rect::new(x, y, width, height).expect("the test rectangles are all valid")
    }

    fn par(align: Align, meet_or_slice: MeetOrSlice) -> PreserveAspectRatio {
        PreserveAspectRatio::new(false, align, meet_or_slice)
    }

    fn transform(view_box: ViewBox, viewport: Rect, par: PreserveAspectRatio) -> Matrix {
        view_box_transform(&view_box, &viewport, &par).expect("a renderable viewBox and viewport")
    }

    /// The specification's own worked shape: a 30x40 `viewBox` in a 300x400
    /// viewport scales by exactly 10 on both axes, so no alignment or
    /// meet/slice choice can make any difference.
    #[test]
    fn matching_aspect_ratios_scale_uniformly_whatever_the_alignment() {
        let view_box = ViewBox::new(0.0, 0.0, 30.0, 40.0);
        let viewport = rect(0.0, 0.0, 300.0, 400.0);
        let expected = Matrix::new(10.0, 0.0, 0.0, 10.0, 0.0, 0.0);
        for align in [
            Align::None,
            Align::XMinYMin,
            Align::XMidYMid,
            Align::XMaxYMax,
            Align::XMinYMax,
            Align::XMaxYMin,
        ] {
            for meet_or_slice in [MeetOrSlice::Meet, MeetOrSlice::Slice] {
                assert_eq!(
                    transform(view_box, viewport, par(align, meet_or_slice)),
                    expected,
                    "{align} {meet_or_slice}"
                );
            }
        }
    }

    #[test]
    fn meet_letterboxes_a_square_view_box_in_a_wide_viewport() {
        let view_box = ViewBox::new(0.0, 0.0, 100.0, 100.0);
        let viewport = rect(0.0, 0.0, 200.0, 100.0);

        // Uniform scale 1 (the smaller of 2 and 1), 100 units of slack on x.
        let centered = transform(view_box, viewport, par(Align::XMidYMid, MeetOrSlice::Meet));
        assert_eq!(centered, Matrix::new(1.0, 0.0, 0.0, 1.0, 50.0, 0.0));

        let left = transform(view_box, viewport, par(Align::XMinYMin, MeetOrSlice::Meet));
        assert_eq!(left, Matrix::new(1.0, 0.0, 0.0, 1.0, 0.0, 0.0));

        let right = transform(view_box, viewport, par(Align::XMaxYMax, MeetOrSlice::Meet));
        assert_eq!(right, Matrix::new(1.0, 0.0, 0.0, 1.0, 100.0, 0.0));

        // With `meet` the whole viewBox lands inside the viewport.
        for align in [Align::XMinYMin, Align::XMidYMid, Align::XMaxYMax] {
            let matrix = transform(view_box, viewport, par(align, MeetOrSlice::Meet));
            let (x0, y0) = matrix.apply(0.0, 0.0);
            let (x1, y1) = matrix.apply(100.0, 100.0);
            assert!(
                x0 >= viewport.x - 1e-12 && x1 <= viewport.max_x() + 1e-12,
                "{align}"
            );
            assert!(
                y0 >= viewport.y - 1e-12 && y1 <= viewport.max_y() + 1e-12,
                "{align}"
            );
        }
    }

    #[test]
    fn slice_covers_a_wide_viewport_and_overflows_the_other_axis() {
        let view_box = ViewBox::new(0.0, 0.0, 100.0, 100.0);
        let viewport = rect(0.0, 0.0, 200.0, 100.0);

        // Uniform scale 2 (the larger of 2 and 1), 100 units of overflow on y.
        let centered = transform(view_box, viewport, par(Align::XMidYMid, MeetOrSlice::Slice));
        assert_eq!(centered, Matrix::new(2.0, 0.0, 0.0, 2.0, 0.0, -50.0));

        let top = transform(view_box, viewport, par(Align::XMinYMin, MeetOrSlice::Slice));
        assert_eq!(top, Matrix::new(2.0, 0.0, 0.0, 2.0, 0.0, 0.0));

        let bottom = transform(view_box, viewport, par(Align::XMaxYMax, MeetOrSlice::Slice));
        assert_eq!(bottom, Matrix::new(2.0, 0.0, 0.0, 2.0, 0.0, -100.0));

        // With `slice` the viewport is entirely covered.
        for align in [Align::XMinYMin, Align::XMidYMid, Align::XMaxYMax] {
            let matrix = transform(view_box, viewport, par(align, MeetOrSlice::Slice));
            let (x0, y0) = matrix.apply(0.0, 0.0);
            let (x1, y1) = matrix.apply(100.0, 100.0);
            assert!(
                x0 <= viewport.x + 1e-12 && x1 >= viewport.max_x() - 1e-12,
                "{align}"
            );
            assert!(
                y0 <= viewport.y + 1e-12 && y1 >= viewport.max_y() - 1e-12,
                "{align}"
            );
        }
    }

    #[test]
    fn align_none_scales_the_two_axes_independently() {
        let view_box = ViewBox::new(0.0, 0.0, 100.0, 100.0);
        let viewport = rect(0.0, 0.0, 200.0, 100.0);
        let expected = Matrix::new(2.0, 0.0, 0.0, 1.0, 0.0, 0.0);
        // Neither meet nor slice has any effect once alignment is `none`.
        assert_eq!(
            transform(view_box, viewport, par(Align::None, MeetOrSlice::Meet)),
            expected
        );
        assert_eq!(
            transform(view_box, viewport, par(Align::None, MeetOrSlice::Slice)),
            expected
        );
        // The viewBox exactly fills the viewport, corner to corner.
        assert_eq!(expected.apply(0.0, 0.0), (0.0, 0.0));
        assert_eq!(expected.apply(100.0, 100.0), (200.0, 100.0));
    }

    #[test]
    fn every_alignment_places_the_view_box_where_its_name_says() {
        let view_box = ViewBox::new(0.0, 0.0, 100.0, 100.0);
        // Twice as wide and twice as tall as the scaled viewBox, so both
        // axes have exactly 100 units of slack to distribute.
        let viewport = rect(0.0, 0.0, 200.0, 200.0);
        let cases = [
            (Align::XMinYMin, 0.0, 0.0),
            (Align::XMidYMin, 50.0, 0.0),
            (Align::XMaxYMin, 100.0, 0.0),
            (Align::XMinYMid, 0.0, 50.0),
            (Align::XMidYMid, 50.0, 50.0),
            (Align::XMaxYMid, 100.0, 50.0),
            (Align::XMinYMax, 0.0, 100.0),
            (Align::XMidYMax, 50.0, 100.0),
            (Align::XMaxYMax, 100.0, 100.0),
        ];
        // A square viewBox in a square viewport scales by 2 with no slack
        // at all, so every alignment agrees.
        for (align, _, _) in cases {
            assert_eq!(
                transform(view_box, viewport, par(align, MeetOrSlice::Meet)),
                Matrix::new(2.0, 0.0, 0.0, 2.0, 0.0, 0.0),
                "{align} has no slack to distribute"
            );
        }

        // A 100x50 viewBox scales by 2 (the smaller of 2 and 4) and leaves
        // 100 units of slack on y alone.
        let wide = ViewBox::new(0.0, 0.0, 100.0, 50.0);
        for (align, _, expected_y) in cases {
            assert_eq!(
                transform(wide, viewport, par(align, MeetOrSlice::Meet)),
                Matrix::new(2.0, 0.0, 0.0, 2.0, 0.0, expected_y),
                "{align}"
            );
        }

        // A 50x100 viewBox is the mirror image: the slack is all on x.
        let tall = ViewBox::new(0.0, 0.0, 50.0, 100.0);
        for (align, expected_x, _) in cases {
            assert_eq!(
                transform(tall, viewport, par(align, MeetOrSlice::Meet)),
                Matrix::new(2.0, 0.0, 0.0, 2.0, expected_x, 0.0),
                "{align}"
            );
        }
    }

    #[test]
    fn a_view_box_origin_is_subtracted_before_scaling() {
        let view_box = ViewBox::new(10.0, 20.0, 30.0, 40.0);
        let viewport = rect(0.0, 0.0, 30.0, 40.0);
        let matrix = transform(view_box, viewport, par(Align::XMidYMid, MeetOrSlice::Meet));
        assert_eq!(matrix, Matrix::new(1.0, 0.0, 0.0, 1.0, -10.0, -20.0));
        // The viewBox's own corner lands on the viewport's corner.
        assert_eq!(matrix.apply(10.0, 20.0), (0.0, 0.0));
        assert_eq!(matrix.apply(40.0, 60.0), (30.0, 40.0));
    }

    #[test]
    fn a_viewport_origin_offsets_the_result() {
        let view_box = ViewBox::new(0.0, 0.0, 30.0, 40.0);
        let viewport = rect(5.0, 7.0, 30.0, 40.0);
        let matrix = transform(view_box, viewport, par(Align::XMidYMid, MeetOrSlice::Meet));
        assert_eq!(matrix, Matrix::new(1.0, 0.0, 0.0, 1.0, 5.0, 7.0));
        assert_eq!(matrix.apply(0.0, 0.0), (5.0, 7.0));
    }

    #[test]
    fn both_origins_and_a_scale_compose_in_the_right_order() {
        // Scale first, then translate: a viewBox origin is scaled away, a
        // viewport origin is not.
        let view_box = ViewBox::new(10.0, 10.0, 10.0, 10.0);
        let viewport = rect(100.0, 200.0, 50.0, 50.0);
        let matrix = transform(view_box, viewport, par(Align::XMidYMid, MeetOrSlice::Meet));
        assert_eq!(matrix, Matrix::new(5.0, 0.0, 0.0, 5.0, 50.0, 150.0));
        assert_eq!(matrix.apply(10.0, 10.0), (100.0, 200.0));
        assert_eq!(matrix.apply(20.0, 20.0), (150.0, 250.0));
    }

    #[test]
    fn the_defer_keyword_makes_no_difference_here() {
        let view_box = ViewBox::new(0.0, 0.0, 100.0, 100.0);
        let viewport = rect(0.0, 0.0, 200.0, 100.0);
        let plain = PreserveAspectRatio::new(false, Align::XMidYMid, MeetOrSlice::Meet);
        let deferred = PreserveAspectRatio::new(true, Align::XMidYMid, MeetOrSlice::Meet);
        assert_eq!(
            view_box_transform(&view_box, &viewport, &plain),
            view_box_transform(&view_box, &viewport, &deferred)
        );
    }

    #[test]
    fn a_zero_sized_view_box_disables_rendering() {
        let viewport = rect(0.0, 0.0, 100.0, 100.0);
        let par = PreserveAspectRatio::default();
        for view_box in [
            ViewBox::new(0.0, 0.0, 0.0, 100.0),
            ViewBox::new(0.0, 0.0, 100.0, 0.0),
            ViewBox::new(0.0, 0.0, 0.0, 0.0),
        ] {
            assert!(!view_box.is_renderable());
            assert_eq!(view_box_transform(&view_box, &viewport, &par), None);
        }
        // A negative size is only reachable through `ViewBox::new`, since
        // the parser rejects it; it is not renderable either.
        let negative = ViewBox::new(0.0, 0.0, -100.0, 100.0);
        assert_eq!(view_box_transform(&negative, &viewport, &par), None);
    }

    #[test]
    fn a_zero_sized_viewport_disables_rendering() {
        let view_box = ViewBox::new(0.0, 0.0, 100.0, 100.0);
        let par = PreserveAspectRatio::default();
        assert_eq!(
            view_box_transform(&view_box, &rect(0.0, 0.0, 0.0, 100.0), &par),
            None
        );
        assert_eq!(
            view_box_transform(&view_box, &rect(0.0, 0.0, 100.0, 0.0), &par),
            None
        );
        assert_eq!(
            svg_viewport_transform(0.0, 100.0, Some(&view_box), &par),
            None
        );
        assert_eq!(svg_viewport_transform(0.0, 100.0, None, &par), None);
        // A negative size never even builds a viewport rectangle.
        assert_eq!(svg_viewport_transform(-1.0, 100.0, None, &par), None);
    }

    #[test]
    fn non_finite_input_yields_no_transform() {
        let par = PreserveAspectRatio::default();
        let viewport = rect(0.0, 0.0, 100.0, 100.0);
        assert_eq!(
            view_box_transform(&ViewBox::new(f64::NAN, 0.0, 10.0, 10.0), &viewport, &par),
            None
        );
        assert_eq!(
            view_box_transform(
                &ViewBox::new(0.0, 0.0, f64::INFINITY, 10.0),
                &viewport,
                &par
            ),
            None
        );
        let view_box = ViewBox::new(0.0, 0.0, 10.0, 10.0);
        assert_eq!(
            view_box_transform(
                &view_box,
                &Rect::new(f64::NAN, 0.0, 100.0, 100.0).expect("NaN is not negative"),
                &par
            ),
            None
        );
        assert_eq!(
            svg_viewport_transform(f64::INFINITY, 100.0, Some(&view_box), &par),
            None
        );
    }

    #[test]
    fn a_missing_view_box_only_moves_the_origin() {
        let par = PreserveAspectRatio::default();
        let viewport = rect(20.0, 10.0, 100.0, 50.0);
        let matrix =
            nested_viewport_transform(&viewport, None, &par).expect("a renderable viewport");
        assert_eq!(matrix, Matrix::translate(20.0, 10.0));
        assert_eq!(matrix.apply(0.0, 0.0), (20.0, 10.0));
        assert_eq!(matrix.apply(5.0, 5.0), (25.0, 15.0));
        // At the root, where the viewport starts at the origin, that
        // translation is the identity.
        let root = svg_viewport_transform(100.0, 50.0, None, &par).expect("a renderable viewport");
        assert!(root.is_identity());
    }

    #[test]
    fn a_nested_viewport_with_a_view_box_is_the_full_algorithm() {
        let par = PreserveAspectRatio::default();
        let viewport = rect(20.0, 10.0, 100.0, 50.0);
        let view_box = ViewBox::new(0.0, 0.0, 50.0, 50.0);
        assert_eq!(
            nested_viewport_transform(&viewport, Some(&view_box), &par),
            view_box_transform(&view_box, &viewport, &par)
        );
        // Uniform scale 1 (the smaller of 2 and 1), centered in the 100-unit
        // width and offset by the viewport's own origin.
        assert_eq!(
            nested_viewport_transform(&viewport, Some(&view_box), &par),
            Some(Matrix::new(1.0, 0.0, 0.0, 1.0, 45.0, 10.0))
        );
    }

    #[test]
    fn the_root_helper_matches_the_general_one() {
        let par = par(Align::XMaxYMin, MeetOrSlice::Slice);
        let view_box = ViewBox::new(-5.0, -5.0, 20.0, 10.0);
        assert_eq!(
            svg_viewport_transform(300.0, 400.0, Some(&view_box), &par),
            view_box_transform(&view_box, &rect(0.0, 0.0, 300.0, 400.0), &par)
        );
    }

    #[test]
    fn every_alignment_keyword_has_its_own_factors() {
        assert_eq!(alignment_factors(Align::None), None);
        let factors = [
            Align::XMinYMin,
            Align::XMidYMin,
            Align::XMaxYMin,
            Align::XMinYMid,
            Align::XMidYMid,
            Align::XMaxYMid,
            Align::XMinYMax,
            Align::XMidYMax,
            Align::XMaxYMax,
        ]
        .map(|align| alignment_factors(align).expect("only `none` has no factors"));
        // All nine are distinct: no two alignment keywords collapse onto the
        // same placement.
        for (index, first) in factors.iter().enumerate() {
            for second in &factors[index + 1..] {
                assert_ne!(first, second);
            }
        }
    }
}
