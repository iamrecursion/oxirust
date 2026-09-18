//! Equivalent paths and geometric bounding boxes for the basic shapes
//! (`rect`, `circle`, `ellipse`, `line`, `polyline`, `polygon`, `path`).
//!
//! # Scope: geometry only, not decoration
//!
//! [`bounding_box`] computes the *geometric* bounding box of the shape's
//! own outline — it does **not** account for `transform`, `stroke-width`,
//! markers, or any other decoration that would grow or move the painted
//! area. A "decorated" bounding box (what a renderer would need to know
//! how much screen space an element actually occupies) is a strictly
//! larger computation layered on top of this one — future work, not this
//! function's job.
//!
//! # Equivalent-path recipes
//!
//! [`equivalent_path`] follows the SVG 2 specification's own "equivalent
//! path" pseudocode for each basic shape, always using the *general* form
//! (rounded-rectangle corners as elliptical arcs, a full ellipse as four
//! quarter-arcs) rather than special-casing the sharp-corner/circular
//! degenerate cases into straight lines: an elliptical arc whose radius is
//! zero is, by the path grammar's own out-of-range rule (SVG 1.1 Appendix
//! F.6.2), already equivalent to a straight line, so the general form is
//! exact even when `rx`/`ry` end up `0`. Every arc below is the *minor* arc
//! (`large_arc = false`) swept in the increasing-angle direction
//! (`sweep = true`); [`bounding_box`]'s independent, analytic rect/circle
//! computation cross-checks this in the test module below rather than
//! trusting it by inspection.

use crate::error::ValueError;
use crate::geometry::path_ops;
use crate::geometry::primitives::{Point, Rect};
use crate::names::attrs::AttrId;
use crate::names::elements::ElementId;
use crate::tree::typed::SvgElement;
use crate::values::length::{Length, LengthContext, PercentBasis};
use crate::values::path::{PathData, PathSegment};

/// Resolves `attr` as a `<length>`, or `default` if the attribute is
/// absent.
fn resolve_length(
    element: SvgElement<'_>,
    attr: AttrId,
    ctx: &LengthContext,
    basis: PercentBasis,
    default: f64,
) -> Result<f64, ValueError> {
    match element.attr(attr) {
        None => Ok(default),
        Some(raw) => Length::parse(raw).map(|length| length.resolve(ctx, basis)),
    }
}

/// Resolves `attr` as a `<length>` that may also be the literal keyword
/// `auto`. Returns `None` for `auto`, for an absent attribute, and — per
/// SVG 2's "a negative value is treated as if the property were
/// unspecified" rule for `rx`/`ry` — for a value that resolves to a
/// negative number; `Some(value)` otherwise.
fn resolve_auto_length(
    element: SvgElement<'_>,
    attr: AttrId,
    ctx: &LengthContext,
    basis: PercentBasis,
) -> Result<Option<f64>, ValueError> {
    match element.attr(attr) {
        None => Ok(None),
        Some(raw) if raw.trim() == "auto" => Ok(None),
        Some(raw) => {
            let value = Length::parse(raw)?.resolve(ctx, basis);
            Ok(if value < 0.0 { None } else { Some(value) })
        }
    }
}

/// Resolves `rect`'s `rx`/`ry` per SVG 2's auto/mutual-derivation/clamp
/// algorithm: missing (or `auto`, or negative) both means sharp corners
/// (`0`, `0`); missing one copies the other; either is then clamped to
/// half of `width`/`height` respectively. `width`/`height` must already be
/// known positive (the caller has already handled the zero/negative,
/// disables-rendering case).
fn rect_radii(
    element: SvgElement<'_>,
    ctx: &LengthContext,
    width: f64,
    height: f64,
) -> Result<(f64, f64), ValueError> {
    let rx = resolve_auto_length(element, AttrId::Rx, ctx, PercentBasis::Width)?;
    let ry = resolve_auto_length(element, AttrId::Ry, ctx, PercentBasis::Height)?;
    let (rx, ry) = mutually_derive(rx, ry);
    Ok((rx.min(width / 2.0), ry.min(height / 2.0)))
}

/// Resolves `ellipse`'s `rx`/`ry` per SVG 2's mutual-derivation algorithm
/// (the same as `rect`'s, minus the width/height clamp `rect` alone has).
fn ellipse_radii(element: SvgElement<'_>, ctx: &LengthContext) -> Result<(f64, f64), ValueError> {
    let rx = resolve_auto_length(element, AttrId::Rx, ctx, PercentBasis::Width)?;
    let ry = resolve_auto_length(element, AttrId::Ry, ctx, PercentBasis::Height)?;
    Ok(mutually_derive(rx, ry))
}

/// The shared "if one is auto/negative/absent, copy the other; if both
/// are, both become `0`" resolution `rect`'s and `ellipse`'s `rx`/`ry`
/// share.
const fn mutually_derive(rx: Option<f64>, ry: Option<f64>) -> (f64, f64) {
    match (rx, ry) {
        (None, None) => (0.0, 0.0),
        (Some(rx), None) => (rx, rx),
        (None, Some(ry)) => (ry, ry),
        (Some(rx), Some(ry)) => (rx, ry),
    }
}

/// Builds a rounded (or, when `rx`/`ry` are both `0`, sharp) rectangle's
/// equivalent path — the SVG 2 specification's own nine-command pseudocode
/// for the `rect` element, unconditionally, per the [module
/// documentation](self)'s note on why the degenerate case needs no special
/// casing.
fn rect_path(x: f64, y: f64, width: f64, height: f64, rx: f64, ry: f64) -> PathData {
    use PathSegment::{ArcTo, Close, LineTo, MoveTo};
    let arc = |x: f64, y: f64| ArcTo {
        abs: true,
        rx,
        ry,
        x_axis_rotation: 0.0,
        large_arc: false,
        sweep: true,
        x,
        y,
    };
    PathData::from_segments(vec![
        MoveTo {
            abs: true,
            x: x + rx,
            y,
        },
        LineTo {
            abs: true,
            x: x + width - rx,
            y,
        },
        arc(x + width, y + ry),
        LineTo {
            abs: true,
            x: x + width,
            y: y + height - ry,
        },
        arc(x + width - rx, y + height),
        LineTo {
            abs: true,
            x: x + rx,
            y: y + height,
        },
        arc(x, y + height - ry),
        LineTo {
            abs: true,
            x,
            y: y + ry,
        },
        arc(x + rx, y),
        Close,
    ])
}

/// Builds a circle/ellipse's equivalent path: four quarter-turn arcs
/// starting at the rightmost point, tracing the outline once clockwise
/// (increasing angle, in SVG's y-down coordinate system) back to the
/// start.
fn ellipse_path(cx: f64, cy: f64, rx: f64, ry: f64) -> PathData {
    use PathSegment::{ArcTo, Close, MoveTo};
    let arc = |x: f64, y: f64| ArcTo {
        abs: true,
        rx,
        ry,
        x_axis_rotation: 0.0,
        large_arc: false,
        sweep: true,
        x,
        y,
    };
    PathData::from_segments(vec![
        MoveTo {
            abs: true,
            x: cx + rx,
            y: cy,
        },
        arc(cx, cy + ry),
        arc(cx - rx, cy),
        arc(cx, cy - ry),
        arc(cx + rx, cy),
        Close,
    ])
}

/// Builds a `line` element's equivalent path: one open, unclosed segment.
fn line_path(x1: f64, y1: f64, x2: f64, y2: f64) -> PathData {
    use PathSegment::{LineTo, MoveTo};
    PathData::from_segments(vec![
        MoveTo {
            abs: true,
            x: x1,
            y: y1,
        },
        LineTo {
            abs: true,
            x: x2,
            y: y2,
        },
    ])
}

/// Builds a `polyline`/`polygon` element's equivalent path: a `moveto` to
/// the first point, a `lineto` per remaining point, and — only for
/// `polygon`, via `close` — a trailing `closepath`. An empty `points` list
/// produces an empty path.
fn polyline_path(points: &[[f64; 2]], close: bool) -> PathData {
    use PathSegment::{Close, LineTo, MoveTo};
    let mut iter = points.iter();
    let Some(&[x, y]) = iter.next() else {
        return PathData::new();
    };
    let mut segments = Vec::with_capacity(points.len() + usize::from(close));
    segments.push(MoveTo { abs: true, x, y });
    for &[x, y] in iter {
        segments.push(LineTo { abs: true, x, y });
    }
    if close {
        segments.push(Close);
    }
    PathData::from_segments(segments)
}

/// `rect`'s equivalent path. A zero or negative resolved `width`/`height`
/// disables rendering (SVG 2 §7 geometry-property semantics): an empty
/// path, not an error.
fn rect_equivalent_path(
    element: SvgElement<'_>,
    ctx: &LengthContext,
) -> Result<PathData, ValueError> {
    let x = resolve_length(element, AttrId::X, ctx, PercentBasis::Width, 0.0)?;
    let y = resolve_length(element, AttrId::Y, ctx, PercentBasis::Height, 0.0)?;
    let width = resolve_length(element, AttrId::Width, ctx, PercentBasis::Width, 0.0)?;
    let height = resolve_length(element, AttrId::Height, ctx, PercentBasis::Height, 0.0)?;
    if width <= 0.0 || height <= 0.0 {
        return Ok(PathData::new());
    }
    let (rx, ry) = rect_radii(element, ctx, width, height)?;
    Ok(rect_path(x, y, width, height, rx, ry))
}

/// `circle`'s equivalent path. A zero or negative resolved `r` disables
/// rendering: an empty path, not an error.
fn circle_equivalent_path(
    element: SvgElement<'_>,
    ctx: &LengthContext,
) -> Result<PathData, ValueError> {
    let cx = resolve_length(element, AttrId::Cx, ctx, PercentBasis::Width, 0.0)?;
    let cy = resolve_length(element, AttrId::Cy, ctx, PercentBasis::Height, 0.0)?;
    let r = resolve_length(element, AttrId::R, ctx, PercentBasis::Diagonal, 0.0)?;
    if r <= 0.0 {
        return Ok(PathData::new());
    }
    Ok(ellipse_path(cx, cy, r, r))
}

/// `ellipse`'s equivalent path. A zero or negative resolved `rx`/`ry`
/// (after mutual auto-derivation) disables rendering: an empty path, not
/// an error.
fn ellipse_equivalent_path(
    element: SvgElement<'_>,
    ctx: &LengthContext,
) -> Result<PathData, ValueError> {
    let cx = resolve_length(element, AttrId::Cx, ctx, PercentBasis::Width, 0.0)?;
    let cy = resolve_length(element, AttrId::Cy, ctx, PercentBasis::Height, 0.0)?;
    let (rx, ry) = ellipse_radii(element, ctx)?;
    if rx <= 0.0 || ry <= 0.0 {
        return Ok(PathData::new());
    }
    Ok(ellipse_path(cx, cy, rx, ry))
}

/// `line`'s equivalent path. Unlike the other shapes, coincident endpoints
/// are not a documented "disables rendering" case — the endpoints are
/// still well-defined geometry — so this never substitutes an empty path.
fn line_equivalent_path(
    element: SvgElement<'_>,
    ctx: &LengthContext,
) -> Result<PathData, ValueError> {
    let x1 = resolve_length(element, AttrId::X1, ctx, PercentBasis::Width, 0.0)?;
    let y1 = resolve_length(element, AttrId::Y1, ctx, PercentBasis::Height, 0.0)?;
    let x2 = resolve_length(element, AttrId::X2, ctx, PercentBasis::Width, 0.0)?;
    let y2 = resolve_length(element, AttrId::Y2, ctx, PercentBasis::Height, 0.0)?;
    Ok(line_path(x1, y1, x2, y2))
}

/// `polyline`/`polygon`'s equivalent path. `points` has no unit/percentage
/// grammar of its own (plain numbers only), so no [`LengthContext`] is
/// needed here. A missing or entirely-empty `points` attribute disables
/// rendering: an empty path.
fn polyline_equivalent_path(element: SvgElement<'_>, close: bool) -> Result<PathData, ValueError> {
    match element.points() {
        None => Ok(PathData::new()),
        Some(Ok(points)) => Ok(polyline_path(&points, close)),
        Some(Err(error)) => Err(error),
    }
}

/// `path`'s "equivalent path" is itself: its own parsed `d` attribute. A
/// missing `d` is `none`, which disables rendering: an empty path.
fn path_equivalent_path(element: SvgElement<'_>) -> Result<PathData, ValueError> {
    match element.path_data() {
        None => Ok(PathData::new()),
        Some(result) => result,
    }
}

/// The equivalent path data for a basic shape element (`rect`, `circle`,
/// `ellipse`, `line`, `polyline`, `polygon`, `path`), resolving geometry
/// attribute percentages against `ctx` with the correct
/// [`PercentBasis`] per attribute (`x`/`width`/`cx`/`x1`/`x2` against
/// [`PercentBasis::Width`]; `y`/`height`/`cy`/`y1`/`y2` against
/// [`PercentBasis::Height`]; `r` against [`PercentBasis::Diagonal`], per
/// the specification's own rule for a radius that is not inherently
/// horizontal or vertical).
///
/// `None` when `element` is not one of the seven shape kinds above (a
/// container, a paint server, …) — distinct from `Some(Ok(PathData::new()))`,
/// which means "a shape, but its resolved geometry is degenerate/absent and
/// therefore disables rendering" (see each per-shape helper's own
/// documentation for exactly which attribute values trigger that).
#[must_use]
pub fn equivalent_path(
    element: SvgElement<'_>,
    ctx: &LengthContext,
) -> Option<Result<PathData, ValueError>> {
    Some(match element.kind()? {
        ElementId::Rect => rect_equivalent_path(element, ctx),
        ElementId::Circle => circle_equivalent_path(element, ctx),
        ElementId::Ellipse => ellipse_equivalent_path(element, ctx),
        ElementId::Line => line_equivalent_path(element, ctx),
        ElementId::Polyline => polyline_equivalent_path(element, false),
        ElementId::Polygon => polyline_equivalent_path(element, true),
        ElementId::Path => path_equivalent_path(element),
        _ => return None,
    })
}

/// The smallest axis-aligned rectangle containing at least one point of
/// each of `points`, or `None` if `points` is empty.
fn points_bounding_box(points: &[[f64; 2]]) -> Option<Rect> {
    let mut iter = points.iter();
    let &[x, y] = iter.next()?;
    let mut bbox = Rect::from_point(Point::new(x, y));
    for &[x, y] in iter {
        bbox = bbox.expand_to_include(Point::new(x, y));
    }
    Some(bbox)
}

/// The geometric bounding box of a basic shape element, resolved the same
/// way [`equivalent_path`] resolves geometry attributes.
///
/// This is **analytic**, not path-derived, for `rect`/`circle`/`ellipse`
/// (whose bounding box is exact and cheap to compute directly, without
/// building or walking a path at all — rounded corners never change a
/// `rect`'s bounding box, since the corner arcs only ever cut *into* it)
/// and `line`/`polyline`/`polygon` (a direct fold over their coordinates);
/// only `path` delegates to [`path_ops::bounding_box`], the exact,
/// non-flattening computation over arbitrary path data.
///
/// Returns `Ok(None)` both when `element` is not one of the seven shape
/// kinds [`equivalent_path`] recognizes, and when it is one but its
/// resolved geometry is empty/degenerate (matching that function's
/// "disables rendering" cases) — this function does not distinguish the
/// two, unlike [`equivalent_path`]'s outer `Option` vs. inner
/// `Ok(PathData::new())`, since a caller asking for a bounding box has no
/// use for an empty one either way.
///
/// This is the shape's own outline only: no `transform`, `stroke-width`,
/// or marker decoration is applied (see the [module documentation](self)).
///
/// # Errors
///
/// Returns [`ValueError`] when a geometry attribute is present but fails
/// to parse.
pub fn bounding_box(
    element: SvgElement<'_>,
    ctx: &LengthContext,
) -> Result<Option<Rect>, ValueError> {
    let Some(kind) = element.kind() else {
        return Ok(None);
    };
    match kind {
        ElementId::Rect => {
            let x = resolve_length(element, AttrId::X, ctx, PercentBasis::Width, 0.0)?;
            let y = resolve_length(element, AttrId::Y, ctx, PercentBasis::Height, 0.0)?;
            let width = resolve_length(element, AttrId::Width, ctx, PercentBasis::Width, 0.0)?;
            let height = resolve_length(element, AttrId::Height, ctx, PercentBasis::Height, 0.0)?;
            Ok(Rect::new(x, y, width, height).filter(|rect| !rect.is_empty()))
        }
        ElementId::Circle => {
            let cx = resolve_length(element, AttrId::Cx, ctx, PercentBasis::Width, 0.0)?;
            let cy = resolve_length(element, AttrId::Cy, ctx, PercentBasis::Height, 0.0)?;
            let r = resolve_length(element, AttrId::R, ctx, PercentBasis::Diagonal, 0.0)?;
            if r <= 0.0 {
                return Ok(None);
            }
            Ok(Rect::new(cx - r, cy - r, 2.0 * r, 2.0 * r))
        }
        ElementId::Ellipse => {
            let cx = resolve_length(element, AttrId::Cx, ctx, PercentBasis::Width, 0.0)?;
            let cy = resolve_length(element, AttrId::Cy, ctx, PercentBasis::Height, 0.0)?;
            let (rx, ry) = ellipse_radii(element, ctx)?;
            if rx <= 0.0 || ry <= 0.0 {
                return Ok(None);
            }
            Ok(Rect::new(cx - rx, cy - ry, 2.0 * rx, 2.0 * ry))
        }
        ElementId::Line => {
            let x1 = resolve_length(element, AttrId::X1, ctx, PercentBasis::Width, 0.0)?;
            let y1 = resolve_length(element, AttrId::Y1, ctx, PercentBasis::Height, 0.0)?;
            let x2 = resolve_length(element, AttrId::X2, ctx, PercentBasis::Width, 0.0)?;
            let y2 = resolve_length(element, AttrId::Y2, ctx, PercentBasis::Height, 0.0)?;
            Ok(Some(Rect::from_points(
                Point::new(x1, y1),
                Point::new(x2, y2),
            )))
        }
        ElementId::Polyline | ElementId::Polygon => match element.points() {
            None => Ok(None),
            Some(Ok(points)) => Ok(points_bounding_box(&points)),
            Some(Err(error)) => Err(error),
        },
        ElementId::Path => match element.path_data() {
            None => Ok(None),
            Some(Ok(path)) => Ok(path_ops::bounding_box(&path)),
            Some(Err(error)) => Err(error),
        },
        _ => Ok(None),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tree::document::SvgDocument;

    const NS: &str = r##"xmlns="http://www.w3.org/2000/svg""##;

    fn element_from(shape_tag: &str) -> SvgDocument {
        SvgDocument::parse_str(&format!("<svg {NS}>{shape_tag}</svg>"))
            .expect("fixture is a valid SVG document")
    }

    fn approx_eq(a: f64, b: f64, epsilon: f64) -> bool {
        (a - b).abs() <= epsilon
    }

    #[test]
    fn rect_equivalent_path_and_bounding_box_agree_with_rounded_corners() {
        let doc = element_from(r##"<rect x="10" y="20" width="100" height="50" rx="15"/>"##);
        let ctx = LengthContext::default();
        let element = doc.svg_element().children().next().expect("rect child");

        let path = equivalent_path(element, &ctx)
            .expect("rect is a shape")
            .expect("valid geometry");
        assert_eq!(path.len(), 10, "9 drawing commands plus Z");

        // ry auto-derives from rx (15), clamped to min(width/2, height/2) = 25.
        let from_path = path_ops::bounding_box(&path).expect("non-empty path has a bbox");
        let analytic = Rect::new(10.0, 20.0, 100.0, 50.0).expect("valid rect");
        assert!(approx_eq(from_path.x, analytic.x, 1e-9));
        assert!(approx_eq(from_path.y, analytic.y, 1e-9));
        assert!(approx_eq(from_path.width, analytic.width, 1e-9));
        assert!(approx_eq(from_path.height, analytic.height, 1e-9));

        let direct = bounding_box(element, &ctx)
            .expect("valid geometry")
            .expect("non-empty");
        assert_eq!(
            direct, analytic,
            "rounded corners never change a rect's bbox"
        );
    }

    #[test]
    fn rect_with_sharp_corners_uses_the_same_general_path_form() {
        let doc = element_from(r##"<rect x="0" y="0" width="10" height="10"/>"##);
        let ctx = LengthContext::default();
        let element = doc.svg_element().children().next().expect("rect child");

        let path = equivalent_path(element, &ctx)
            .expect("rect is a shape")
            .expect("valid geometry");
        let bbox = path_ops::bounding_box(&path).expect("non-empty path has a bbox");
        assert_eq!(bbox, Rect::new(0.0, 0.0, 10.0, 10.0).expect("valid rect"));
    }

    #[test]
    fn rect_rx_auto_derives_from_ry_and_clamps_to_half_the_smaller_dimension() {
        let doc = element_from(r##"<rect x="0" y="0" width="10" height="40" ry="100"/>"##);
        let ctx = LengthContext::default();
        let element = doc.svg_element().children().next().expect("rect child");

        let path = equivalent_path(element, &ctx)
            .expect("rect is a shape")
            .expect("valid geometry");
        // rx auto-derives from ry (100), then both clamp: rx to width/2 = 5,
        // ry to height/2 = 20.
        match path.segments()[2] {
            PathSegment::ArcTo { rx, ry, .. } => {
                assert!(approx_eq(rx, 5.0, 1e-9), "rx clamped to width/2, got {rx}");
                assert!(
                    approx_eq(ry, 20.0, 1e-9),
                    "ry clamped to height/2, got {ry}"
                );
            }
            other => panic!("expected an ArcTo segment, got {other:?}"),
        }
    }

    #[test]
    fn rect_negative_rx_is_treated_as_auto() {
        let doc = element_from(r##"<rect x="0" y="0" width="10" height="10" rx="-5" ry="3"/>"##);
        let ctx = LengthContext::default();
        let element = doc.svg_element().children().next().expect("rect child");
        let path = equivalent_path(element, &ctx)
            .expect("rect is a shape")
            .expect("valid geometry");
        match path.segments()[2] {
            PathSegment::ArcTo { rx, ry, .. } => {
                assert!(
                    approx_eq(rx, 3.0, 1e-9),
                    "negative rx falls back to ry, got {rx}"
                );
                assert!(approx_eq(ry, 3.0, 1e-9));
            }
            other => panic!("expected an ArcTo segment, got {other:?}"),
        }
    }

    #[test]
    fn zero_or_negative_rect_dimensions_disable_rendering() {
        let ctx = LengthContext::default();
        for tag in [
            r##"<rect x="0" y="0" width="0" height="10"/>"##,
            r##"<rect x="0" y="0" width="10" height="0"/>"##,
            r##"<rect x="0" y="0"/>"##,
        ] {
            let doc = element_from(tag);
            let element = doc.svg_element().children().next().expect("rect child");
            let path = equivalent_path(element, &ctx)
                .expect("rect is a shape")
                .expect("not an error, just empty");
            assert!(path.is_empty(), "tag {tag:?} should produce an empty path");
            assert_eq!(bounding_box(element, &ctx), Ok(None), "tag {tag:?}");
        }
    }

    #[test]
    fn circle_equivalent_path_arc_endpoints_lie_on_the_circle_and_bounding_box_is_exact() {
        let doc = element_from(r##"<circle cx="5" cy="7" r="3"/>"##);
        let ctx = LengthContext::default();
        let element = doc.svg_element().children().next().expect("circle child");

        let path = equivalent_path(element, &ctx)
            .expect("circle is a shape")
            .expect("valid geometry");
        assert_eq!(path.len(), 6, "MoveTo + 4 ArcTo + Close");

        for segment in path.segments() {
            let (x, y) = match *segment {
                PathSegment::MoveTo { x, y, .. } | PathSegment::ArcTo { x, y, .. } => (x, y),
                PathSegment::Close => continue,
                other => panic!("unexpected segment {other:?}"),
            };
            let distance = ((x - 5.0).powi(2) + (y - 7.0).powi(2)).sqrt();
            assert!(
                approx_eq(distance, 3.0, 1e-9),
                "point ({x}, {y}) is not on the circle: distance {distance}"
            );
        }

        let from_path = path_ops::bounding_box(&path).expect("non-empty path has a bbox");
        let analytic = Rect::new(2.0, 4.0, 6.0, 6.0).expect("valid rect");
        assert!(approx_eq(from_path.x, analytic.x, 1e-9));
        assert!(approx_eq(from_path.y, analytic.y, 1e-9));
        assert!(approx_eq(from_path.width, analytic.width, 1e-9));
        assert!(approx_eq(from_path.height, analytic.height, 1e-9));

        assert_eq!(
            bounding_box(element, &ctx).expect("valid geometry"),
            Some(analytic)
        );
    }

    #[test]
    fn circle_zero_or_negative_radius_disables_rendering() {
        let ctx = LengthContext::default();
        for tag in [
            r##"<circle cx="0" cy="0" r="0"/>"##,
            r##"<circle cx="0" cy="0"/>"##,
        ] {
            let doc = element_from(tag);
            let element = doc.svg_element().children().next().expect("circle child");
            let path = equivalent_path(element, &ctx)
                .expect("circle is a shape")
                .expect("not an error");
            assert!(path.is_empty());
            assert_eq!(bounding_box(element, &ctx), Ok(None));
        }
    }

    #[test]
    fn ellipse_mutual_auto_derivation_and_bounding_box() {
        let doc = element_from(r##"<ellipse cx="0" cy="0" rx="4"/>"##);
        let ctx = LengthContext::default();
        let element = doc.svg_element().children().next().expect("ellipse child");

        // ry is absent (auto), so it copies rx (4) -- a circle in effect.
        let path = equivalent_path(element, &ctx)
            .expect("ellipse is a shape")
            .expect("valid geometry");
        assert_eq!(
            bounding_box(element, &ctx).expect("valid geometry"),
            Some(Rect::new(-4.0, -4.0, 8.0, 8.0).expect("valid rect"))
        );
        assert_eq!(
            path_ops::bounding_box(&path),
            Some(Rect::new(-4.0, -4.0, 8.0, 8.0).expect("valid rect"))
        );

        let both_auto = element_from(r##"<ellipse cx="0" cy="0"/>"##);
        let e2 = both_auto
            .svg_element()
            .children()
            .next()
            .expect("ellipse child");
        assert_eq!(
            bounding_box(e2, &ctx),
            Ok(None),
            "both auto disables rendering"
        );
    }

    #[test]
    fn line_bounding_box_is_always_some_even_when_degenerate() {
        let ctx = LengthContext::default();
        let doc = element_from(r##"<line x1="1" y1="2" x2="4" y2="6"/>"##);
        let element = doc.svg_element().children().next().expect("line child");
        let path = equivalent_path(element, &ctx)
            .expect("line is a shape")
            .expect("valid geometry");
        assert_eq!(path.len(), 2, "MoveTo + LineTo, never closed");
        assert_eq!(
            bounding_box(element, &ctx).expect("valid geometry"),
            Some(Rect::new(1.0, 2.0, 3.0, 4.0).expect("valid rect"))
        );

        let zero_length = element_from(r##"<line x1="5" y1="5" x2="5" y2="5"/>"##);
        let degenerate = zero_length
            .svg_element()
            .children()
            .next()
            .expect("line child");
        assert_eq!(
            bounding_box(degenerate, &ctx).expect("valid geometry"),
            Some(Rect::new(5.0, 5.0, 0.0, 0.0).expect("valid rect")),
            "a zero-length line is still a well-defined (degenerate) bbox, not None"
        );
    }

    #[test]
    fn polyline_and_polygon_equivalent_paths_and_bounding_boxes() {
        let ctx = LengthContext::default();

        let polyline_doc = element_from(r##"<polyline points="0,0 10,0 10,10"/>"##);
        let polyline = polyline_doc
            .svg_element()
            .children()
            .next()
            .expect("polyline child");
        let path = equivalent_path(polyline, &ctx)
            .expect("polyline is a shape")
            .expect("valid points");
        assert_eq!(path.len(), 3, "MoveTo + 2 LineTo, never closed");
        assert_eq!(
            bounding_box(polyline, &ctx).expect("valid points"),
            Some(Rect::new(0.0, 0.0, 10.0, 10.0).expect("valid rect"))
        );

        let polygon_doc = element_from(r##"<polygon points="0,0 10,0 10,10"/>"##);
        let polygon = polygon_doc
            .svg_element()
            .children()
            .next()
            .expect("polygon child");
        let closed = equivalent_path(polygon, &ctx)
            .expect("polygon is a shape")
            .expect("valid points");
        assert_eq!(closed.len(), 4, "MoveTo + 2 LineTo + Close");

        let empty_points = element_from(r##"<polyline/>"##);
        let empty = empty_points
            .svg_element()
            .children()
            .next()
            .expect("polyline child");
        assert!(
            equivalent_path(empty, &ctx)
                .expect("polyline is a shape")
                .expect("absent points is not an error")
                .is_empty()
        );
        assert_eq!(bounding_box(empty, &ctx), Ok(None));

        let malformed = element_from(r##"<polyline points="0,0 1,1 2"/>"##);
        let malformed_element = malformed
            .svg_element()
            .children()
            .next()
            .expect("polyline child");
        assert!(
            equivalent_path(malformed_element, &ctx)
                .expect("polyline is a shape")
                .is_err()
        );
        assert!(bounding_box(malformed_element, &ctx).is_err());
    }

    #[test]
    fn path_equivalent_path_delegates_to_its_own_d_attribute() {
        let ctx = LengthContext::default();

        let doc = element_from(r##"<path d="M0 0 L10 0 L10 10 Z"/>"##);
        let element = doc.svg_element().children().next().expect("path child");
        let path = equivalent_path(element, &ctx)
            .expect("path is a shape")
            .expect("valid d");
        assert_eq!(path.len(), 4);
        assert_eq!(
            bounding_box(element, &ctx).expect("valid d"),
            Some(Rect::new(0.0, 0.0, 10.0, 10.0).expect("valid rect"))
        );

        let no_d = element_from("<path/>");
        let no_d_element = no_d.svg_element().children().next().expect("path child");
        assert!(
            equivalent_path(no_d_element, &ctx)
                .expect("path is a shape")
                .expect("absent d is not an error")
                .is_empty()
        );
        assert_eq!(bounding_box(no_d_element, &ctx), Ok(None));

        let bad_d = element_from(r##"<path d="M0 0 X1 1"/>"##);
        let bad_element = bad_d.svg_element().children().next().expect("path child");
        assert!(
            equivalent_path(bad_element, &ctx)
                .expect("path is a shape")
                .is_err()
        );
        assert!(bounding_box(bad_element, &ctx).is_err());
    }

    #[test]
    fn non_shape_elements_are_not_shapes() {
        let ctx = LengthContext::default();
        let doc = element_from("<g/>");
        let g = doc.svg_element().children().next().expect("g child");
        assert_eq!(equivalent_path(g, &ctx), None);
        assert_eq!(bounding_box(g, &ctx), Ok(None));
    }

    #[test]
    fn percentages_resolve_against_the_length_context_viewport() {
        let ctx = LengthContext::default().with_viewport(200.0, 100.0);
        let doc = element_from(r##"<rect x="50%" y="50%" width="50%" height="50%"/>"##);
        let element = doc.svg_element().children().next().expect("rect child");
        assert_eq!(
            bounding_box(element, &ctx).expect("valid geometry"),
            Some(Rect::new(100.0, 50.0, 100.0, 50.0).expect("valid rect"))
        );
    }

    #[test]
    fn malformed_geometry_attribute_surfaces_a_value_error() {
        let ctx = LengthContext::default();
        let doc = element_from(r##"<rect x="0" y="0" width="not-a-length" height="10"/>"##);
        let element = doc.svg_element().children().next().expect("rect child");
        assert!(
            equivalent_path(element, &ctx)
                .expect("rect is a shape")
                .is_err()
        );
        assert!(bounding_box(element, &ctx).is_err());
    }
}
