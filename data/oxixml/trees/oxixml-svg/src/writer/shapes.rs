//! Shape and style convenience methods for [`SvgBuilder`].
//!
//! Every one-shot shape method here opens an element, sets its geometry
//! attributes, and closes it again, leaving the attribute target on the
//! element that was just closed — see the [two-cursor
//! model](crate::writer::builder#the-two-cursors) — so
//! `svg.rect(0.0, 0.0, 10.0, 10.0).fill(paint)` styles the rectangle it just
//! drew, not whatever contains it.

use crate::values::paint::Paint;
use crate::values::path::PathData;
use crate::values::transform::TransformList;
use crate::writer::builder::SvgBuilder;

impl SvgBuilder {
    /// Draws a `<rect>` at `(x, y)` with the given `width` and `height`.
    pub fn rect(&mut self, x: f64, y: f64, width: f64, height: f64) -> &mut Self {
        self.open("rect");
        self.attr_number("x", x);
        self.attr_number("y", y);
        self.attr_number("width", width);
        self.attr_number("height", height);
        self.close()
    }

    /// Draws a `<rect>` with rounded corners of radius `(rx, ry)`.
    pub fn rounded_rect(
        &mut self,
        x: f64,
        y: f64,
        width: f64,
        height: f64,
        rx: f64,
        ry: f64,
    ) -> &mut Self {
        self.open("rect");
        self.attr_number("x", x);
        self.attr_number("y", y);
        self.attr_number("width", width);
        self.attr_number("height", height);
        self.attr_number("rx", rx);
        self.attr_number("ry", ry);
        self.close()
    }

    /// Draws a `<circle>` centered at `(cx, cy)` with radius `r`.
    pub fn circle(&mut self, cx: f64, cy: f64, r: f64) -> &mut Self {
        self.open("circle");
        self.attr_number("cx", cx);
        self.attr_number("cy", cy);
        self.attr_number("r", r);
        self.close()
    }

    /// Draws an `<ellipse>` centered at `(cx, cy)` with radii `(rx, ry)`.
    pub fn ellipse(&mut self, cx: f64, cy: f64, rx: f64, ry: f64) -> &mut Self {
        self.open("ellipse");
        self.attr_number("cx", cx);
        self.attr_number("cy", cy);
        self.attr_number("rx", rx);
        self.attr_number("ry", ry);
        self.close()
    }

    /// Draws a `<line>` from `(x1, y1)` to `(x2, y2)`.
    pub fn line(&mut self, x1: f64, y1: f64, x2: f64, y2: f64) -> &mut Self {
        self.open("line");
        self.attr_number("x1", x1);
        self.attr_number("y1", y1);
        self.attr_number("x2", x2);
        self.attr_number("y2", y2);
        self.close()
    }

    /// Draws a `<polyline>` (an open, unfilled-by-default polygonal chain)
    /// through `points`.
    pub fn polyline(&mut self, points: &[[f64; 2]]) -> &mut Self {
        self.open("polyline");
        self.attr_points(points);
        self.close()
    }

    /// Draws a `<polygon>` (a closed polygonal chain) through `points`.
    pub fn polygon(&mut self, points: &[[f64; 2]]) -> &mut Self {
        self.open("polygon");
        self.attr_points(points);
        self.close()
    }

    /// Draws a `<path>` from already-built [`PathData`].
    pub fn path(&mut self, data: &PathData) -> &mut Self {
        self.open("path");
        self.attr_path_d(data);
        self.close()
    }

    /// Draws a `<text>` element at `(x, y)` containing `content`.
    pub fn text_at(&mut self, x: f64, y: f64, content: &str) -> &mut Self {
        self.open("text");
        self.attr_number("x", x);
        self.attr_number("y", y);
        self.text(content);
        self.close()
    }

    /// Draws an `<image>` at `(x, y)` with the given `width`/`height`,
    /// referencing `href`.
    pub fn image(&mut self, x: f64, y: f64, width: f64, height: f64, href: &str) -> &mut Self {
        self.open("image");
        self.attr_number("x", x);
        self.attr_number("y", y);
        self.attr_number("width", width);
        self.attr_number("height", height);
        self.attr_href(href);
        self.close()
    }

    /// Instantiates `href` at `(x, y)` with a `<use>` element.
    pub fn use_ref(&mut self, href: &str, x: f64, y: f64) -> &mut Self {
        self.open("use");
        self.attr_href(href);
        self.attr_number("x", x);
        self.attr_number("y", y);
        self.close()
    }

    /// Opens a `<g>` grouping container and descends into it.
    ///
    /// Unlike the shape methods above, this does **not** close the element
    /// it opens: build the group's contents, then call
    /// [`SvgBuilder::close`] to leave it.
    pub fn group(&mut self) -> &mut Self {
        self.open("g")
    }

    // ------------------------------------------------------ style shorthand

    /// Sets `transform` on the current attribute target.
    pub fn with_transform(&mut self, transform: &TransformList) -> &mut Self {
        self.attr_transform("transform", transform)
    }

    /// Sets `fill` on the current attribute target. An alias of
    /// [`SvgBuilder::fill`].
    pub fn with_fill(&mut self, paint: Paint) -> &mut Self {
        self.fill(paint)
    }

    /// Sets `stroke` on the current attribute target. An alias of
    /// [`SvgBuilder::stroke`].
    pub fn with_stroke(&mut self, paint: Paint) -> &mut Self {
        self.stroke(paint)
    }

    /// Sets `stroke-width` on the current attribute target. An alias of
    /// [`SvgBuilder::stroke_width`].
    pub fn with_stroke_width(&mut self, width: f64) -> &mut Self {
        self.stroke_width(width)
    }

    /// Sets `opacity` on the current attribute target.
    pub fn with_opacity(&mut self, opacity: f64) -> &mut Self {
        self.attr_number("opacity", opacity)
    }

    /// Sets `class` on the current attribute target. An alias of
    /// [`SvgBuilder::attr_class`].
    pub fn with_class(&mut self, class: &str) -> &mut Self {
        self.attr_class(class)
    }

    /// Sets `fill` on the current attribute target.
    pub fn fill(&mut self, paint: Paint) -> &mut Self {
        self.attr_paint("fill", paint)
    }

    /// Sets `stroke` on the current attribute target.
    pub fn stroke(&mut self, paint: Paint) -> &mut Self {
        self.attr_paint("stroke", paint)
    }

    /// Sets `stroke-width` on the current attribute target.
    pub fn stroke_width(&mut self, width: f64) -> &mut Self {
        self.attr_number("stroke-width", width)
    }
}
