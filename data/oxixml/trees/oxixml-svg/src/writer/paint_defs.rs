//! Gradient, pattern, clip-path, mask, and marker builders for
//! [`SvgBuilder`].
//!
//! Every resource this module builds — a `linearGradient`, a
//! `radialGradient`, a `pattern`, a `clipPath`, a `mask`, a `marker` — is
//! appended as a child of a single, lazily created `<defs>` element directly
//! under the document's root (see [`SvgBuilder::open_defs`]), and identified
//! by an `id` that is immediately resolvable through
//! [`oxixml_dom::Document::get_element_by_id`] (every element's `id` is
//! declared `ID`-typed document-wide at [`SvgBuilder::plain`]). Each method
//! returns a [`FuncIri`] naming that `id` as `url(#id)`, ready to hand to
//! [`SvgBuilder::attr_paint`]/[`crate::values::paint::Paint::Iri`] or to any
//! plain attribute (`clip-path`, `mask`, `marker-start`, ...) that takes a
//! bare `<funciri>`.
//!
//! Two shapes of API coexist, matching how open-ended each resource's
//! content model is:
//!
//! * **Closed**: [`SvgBuilder::linear_gradient`] and
//!   [`SvgBuilder::radial_gradient`] take their `<stop>` children as a
//!   `stops` parameter and build the whole resource in one call, without
//!   touching the builder's insertion point or attribute target at all — a
//!   gradient's content model (a flat list of stops) is fixed and small
//!   enough that there is no reason to make the caller open/close each stop
//!   by hand.
//! * **Open for content**: [`SvgBuilder::pattern`], [`SvgBuilder::clip_path`],
//!   [`SvgBuilder::mask`], and [`SvgBuilder::marker`] push the new resource
//!   element onto the insertion-point stack and return immediately; the
//!   caller adds arbitrary shape/group content with the ordinary builder
//!   methods and then calls [`SvgBuilder::close`] to leave it, exactly as
//!   [`SvgBuilder::group`] works.

use oxixml_dom::NodeId;

use crate::ns::SVG_NAMESPACE;
use crate::values::color::Color;
use crate::values::iri::FuncIri;
use crate::writer::builder::SvgBuilder;

/// The coordinate space a gradient's or pattern's own geometry attributes
/// (`x1`/`y1`/`x2`/`y2`, `cx`/`cy`/`r`, `x`/`y`/`width`/`height`, ...) are
/// resolved in — the `gradientUnits`/`patternUnits` attribute.
///
/// `gradientUnits`, `patternUnits`, `maskContentUnits`, and `clipPathUnits`
/// all share this exact two-keyword vocabulary (SVG 1.1 §7.11/§13.3), so one
/// enum serves every caller in this module that needs it.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum CoordinateUnits {
    /// `objectBoundingBox` (the default in the SVG grammar): `0` maps to the
    /// referencing element's bounding-box minimum edge, `1` to its maximum
    /// edge, per axis independently.
    #[default]
    ObjectBoundingBox,
    /// `userSpaceOnUse`: plain coordinates in the user coordinate system in
    /// force where the resource is referenced.
    UserSpaceOnUse,
}

impl CoordinateUnits {
    /// The attribute-value keyword this variant serializes as.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::ObjectBoundingBox => "objectBoundingBox",
            Self::UserSpaceOnUse => "userSpaceOnUse",
        }
    }
}

/// One `<stop>` of a linear or radial gradient.
///
/// [`SvgBuilder::linear_gradient`] and [`SvgBuilder::radial_gradient`]
/// accept a slice of anything that converts into this type, so a caller can
/// pass either `&[GradientStop]` directly or the shorter `&[(f64, Color)]`
/// (via the [`From`] implementation below), matching the "or" the two forms
/// this crate's mission brief allows.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct GradientStop {
    /// The stop's position along the gradient, `0.0..=1.0`.
    pub offset: f64,
    /// The stop's color.
    pub color: Color,
    /// The stop's opacity, `0.0..=1.0`. `1.0` (fully opaque) unless set
    /// otherwise with [`GradientStop::with_opacity`].
    pub opacity: f64,
}

impl GradientStop {
    /// Creates a fully opaque stop at `offset` with `color`.
    #[must_use]
    pub const fn new(offset: f64, color: Color) -> Self {
        Self {
            offset,
            color,
            opacity: 1.0,
        }
    }

    /// Creates a stop with an explicit `opacity`.
    #[must_use]
    pub const fn with_opacity(offset: f64, color: Color, opacity: f64) -> Self {
        Self {
            offset,
            color,
            opacity,
        }
    }
}

impl From<(f64, Color)> for GradientStop {
    /// Builds a fully opaque stop from an `(offset, color)` pair, the
    /// shorthand form [`crate::writer::paint_defs`]'s gradient methods also
    /// accept directly.
    fn from((offset, color): (f64, Color)) -> Self {
        Self::new(offset, color)
    }
}

impl SvgBuilder {
    /// Finds the `<defs>` child of the document's root, creating one and
    /// appending it if none exists yet. Does not touch the insertion point
    /// or attribute target; see [`SvgBuilder::open_defs`] for the
    /// stack-pushing counterpart used to add arbitrary content directly
    /// under `<defs>` itself.
    ///
    /// Returns `None` when [`SvgBuilder::finish`] has already taken this
    /// builder's document — [`SvgBuilder::guard_not_finished`] has already
    /// recorded a [`crate::writer::builder::BuildFlaw::UsedAfterFinish`] in
    /// that case — so every caller in this module checks this and does
    /// nothing further rather than risk appending to the fresh document's
    /// bare root node.
    fn ensure_defs(&mut self) -> Option<NodeId> {
        if !self.guard_not_finished() {
            return None;
        }
        let root = self.root_id();
        let existing = self
            .doc()
            .get_elements_by_tag_name_ns(root, Some(SVG_NAMESPACE), "defs")
            .find(|&id| self.doc().parent(id) == Some(root));
        if let Some(defs) = existing {
            return Some(defs);
        }
        let defs = self
            .doc_mut()
            .create_element_ns(Some(SVG_NAMESPACE), "defs")
            .expect("\"defs\" is always a well-formed element name");
        self.doc_mut().append_child(root, defs).expect(
            "guard_not_finished() guarantees the root is an attached element, which \
             always accepts a freshly created element child",
        );
        Some(defs)
    }

    /// Opens the document's single `<defs>` container (finding it, or
    /// creating and appending it to the document root if none exists yet)
    /// and descends into it, for adding resource elements this module has
    /// no dedicated method for (a `<symbol>`, a `<filter>`, ...). Call
    /// [`SvgBuilder::close`] to leave it.
    ///
    /// A call made after [`SvgBuilder::finish`] records a
    /// [`crate::writer::builder::BuildFlaw::UsedAfterFinish`] and has no
    /// effect.
    pub fn open_defs(&mut self) -> &mut Self {
        let Some(defs) = self.ensure_defs() else {
            return self;
        };
        self.push_open(defs);
        self
    }

    /// Sets the `id` attribute of `element` directly, independent of the
    /// current attribute target.
    fn set_id(&mut self, element: NodeId, id: &str) {
        self.set_attr_on(element, "id", id);
    }

    /// Appends one `<stop>` child per element of `stops` to `gradient`,
    /// as `offset`/`stop-color`/`stop-opacity` presentation attributes.
    fn append_stops<S: Copy + Into<GradientStop>>(&mut self, gradient: NodeId, stops: &[S]) {
        for &stop in stops {
            let stop: GradientStop = stop.into();
            let element = self
                .doc_mut()
                .create_element_ns(Some(SVG_NAMESPACE), "stop")
                .expect("\"stop\" is always a well-formed element name");
            self.doc_mut()
                .append_child(gradient, element)
                .expect("appending a freshly created stop to its gradient cannot fail");
            self.set_number_on(element, "offset", stop.offset);
            let color_text = stop.color.to_string();
            self.set_attr_on(element, "stop-color", &color_text);
            self.set_number_on(element, "stop-opacity", stop.opacity);
        }
    }

    /// Builds a `<linearGradient>` from `(x1, y1)` to `(x2, y2)` with the
    /// given `stops`, appended to the document's `<defs>`.
    ///
    /// `stops` accepts either `&[GradientStop]` or the shorthand
    /// `&[(f64, Color)]`. Does not disturb the current insertion point or
    /// attribute target.
    ///
    /// A call made after [`SvgBuilder::finish`] records a
    /// [`crate::writer::builder::BuildFlaw::UsedAfterFinish`] and builds
    /// nothing, but still returns the `url(#id)` reference the gradient would
    /// have had, for a caller that does not check [`SvgBuilder::flaws`].
    // `linearGradient`'s own attribute set (SVG 1.1 §13.2.2) fixes this
    // shape; a params struct would only add ceremony around it.
    #[allow(clippy::too_many_arguments)]
    pub fn linear_gradient<S: Copy + Into<GradientStop>>(
        &mut self,
        id: &str,
        units: CoordinateUnits,
        x1: f64,
        y1: f64,
        x2: f64,
        y2: f64,
        stops: &[S],
    ) -> FuncIri {
        let Some(defs) = self.ensure_defs() else {
            return FuncIri::new(format!("#{id}"));
        };
        let gradient = self
            .doc_mut()
            .create_element_ns(Some(SVG_NAMESPACE), "linearGradient")
            .expect("\"linearGradient\" is always a well-formed element name");
        self.doc_mut().append_child(defs, gradient).expect(
            "defs, once obtained from ensure_defs(), is always an attached element, \
             which always accepts a freshly created element child",
        );
        self.set_id(gradient, id);
        self.set_attr_on(gradient, "gradientUnits", units.as_str());
        self.set_number_on(gradient, "x1", x1);
        self.set_number_on(gradient, "y1", y1);
        self.set_number_on(gradient, "x2", x2);
        self.set_number_on(gradient, "y2", y2);
        self.append_stops(gradient, stops);
        FuncIri::new(format!("#{id}"))
    }

    /// Builds a `<radialGradient>` centered at `(cx, cy)` with radius `r`
    /// with the given `stops`, appended to the document's `<defs>`.
    ///
    /// `focal` sets an explicit `(fx, fy)` focal point distinct from the
    /// center; `None` leaves it at its spec default (equal to `(cx, cy)`).
    /// `stops` accepts either `&[GradientStop]` or the shorthand
    /// `&[(f64, Color)]`. Does not disturb the current insertion point or
    /// attribute target.
    ///
    /// A call made after [`SvgBuilder::finish`] records a
    /// [`crate::writer::builder::BuildFlaw::UsedAfterFinish`] and builds
    /// nothing, but still returns the `url(#id)` reference the gradient would
    /// have had, for a caller that does not check [`SvgBuilder::flaws`].
    // `radialGradient`'s own attribute set (SVG 1.1 §13.2.3) fixes this
    // shape; a params struct would only add ceremony around it.
    #[allow(clippy::too_many_arguments)]
    pub fn radial_gradient<S: Copy + Into<GradientStop>>(
        &mut self,
        id: &str,
        units: CoordinateUnits,
        cx: f64,
        cy: f64,
        r: f64,
        focal: Option<(f64, f64)>,
        stops: &[S],
    ) -> FuncIri {
        let Some(defs) = self.ensure_defs() else {
            return FuncIri::new(format!("#{id}"));
        };
        let gradient = self
            .doc_mut()
            .create_element_ns(Some(SVG_NAMESPACE), "radialGradient")
            .expect("\"radialGradient\" is always a well-formed element name");
        self.doc_mut().append_child(defs, gradient).expect(
            "defs, once obtained from ensure_defs(), is always an attached element, \
             which always accepts a freshly created element child",
        );
        self.set_id(gradient, id);
        self.set_attr_on(gradient, "gradientUnits", units.as_str());
        self.set_number_on(gradient, "cx", cx);
        self.set_number_on(gradient, "cy", cy);
        self.set_number_on(gradient, "r", r);
        if let Some((fx, fy)) = focal {
            self.set_number_on(gradient, "fx", fx);
            self.set_number_on(gradient, "fy", fy);
        }
        self.append_stops(gradient, stops);
        FuncIri::new(format!("#{id}"))
    }

    /// Opens a `<pattern>` tile of size `width` × `height` at `(x, y)`,
    /// resolved per `units`, appended to the document's `<defs>`. Descends
    /// into it — add the tile's content, then call [`SvgBuilder::close`] to
    /// leave it.
    ///
    /// A call made after [`SvgBuilder::finish`] records a
    /// [`crate::writer::builder::BuildFlaw::UsedAfterFinish`] and opens
    /// nothing: the insertion point is left unchanged, so a matching
    /// [`SvgBuilder::close`] is not needed and would itself record an
    /// [`crate::writer::builder::BuildFlaw::UnbalancedClose`] if attempted.
    pub fn pattern(
        &mut self,
        id: &str,
        units: CoordinateUnits,
        x: f64,
        y: f64,
        width: f64,
        height: f64,
    ) -> FuncIri {
        let Some(defs) = self.ensure_defs() else {
            return FuncIri::new(format!("#{id}"));
        };
        let element = self
            .doc_mut()
            .create_element_ns(Some(SVG_NAMESPACE), "pattern")
            .expect("\"pattern\" is always a well-formed element name");
        self.doc_mut().append_child(defs, element).expect(
            "defs, once obtained from ensure_defs(), is always an attached element, \
             which always accepts a freshly created element child",
        );
        self.set_id(element, id);
        self.set_attr_on(element, "patternUnits", units.as_str());
        self.set_number_on(element, "x", x);
        self.set_number_on(element, "y", y);
        self.set_number_on(element, "width", width);
        self.set_number_on(element, "height", height);
        self.push_open(element);
        FuncIri::new(format!("#{id}"))
    }

    /// Opens a `<clipPath>`, appended to the document's `<defs>`. Descends
    /// into it — add the clipping shape(s), then call [`SvgBuilder::close`]
    /// to leave it.
    ///
    /// A call made after [`SvgBuilder::finish`] records a
    /// [`crate::writer::builder::BuildFlaw::UsedAfterFinish`] and opens
    /// nothing; see [`SvgBuilder::pattern`]'s documentation for the same note
    /// about a matching `close`.
    pub fn clip_path(&mut self, id: &str) -> FuncIri {
        let Some(defs) = self.ensure_defs() else {
            return FuncIri::new(format!("#{id}"));
        };
        let element = self
            .doc_mut()
            .create_element_ns(Some(SVG_NAMESPACE), "clipPath")
            .expect("\"clipPath\" is always a well-formed element name");
        self.doc_mut().append_child(defs, element).expect(
            "defs, once obtained from ensure_defs(), is always an attached element, \
             which always accepts a freshly created element child",
        );
        self.set_id(element, id);
        self.push_open(element);
        FuncIri::new(format!("#{id}"))
    }

    /// Opens a `<mask>`, appended to the document's `<defs>`. Descends into
    /// it — add the mask's content, then call [`SvgBuilder::close`] to leave
    /// it.
    ///
    /// A call made after [`SvgBuilder::finish`] records a
    /// [`crate::writer::builder::BuildFlaw::UsedAfterFinish`] and opens
    /// nothing; see [`SvgBuilder::pattern`]'s documentation for the same note
    /// about a matching `close`.
    pub fn mask(&mut self, id: &str) -> FuncIri {
        let Some(defs) = self.ensure_defs() else {
            return FuncIri::new(format!("#{id}"));
        };
        let element = self
            .doc_mut()
            .create_element_ns(Some(SVG_NAMESPACE), "mask")
            .expect("\"mask\" is always a well-formed element name");
        self.doc_mut().append_child(defs, element).expect(
            "defs, once obtained from ensure_defs(), is always an attached element, \
             which always accepts a freshly created element child",
        );
        self.set_id(element, id);
        self.push_open(element);
        FuncIri::new(format!("#{id}"))
    }

    /// Opens a `<marker>` of size `width` × `height` with reference point
    /// `(ref_x, ref_y)`, appended to the document's `<defs>`. Descends into
    /// it — add the marker's graphical content, then call
    /// [`SvgBuilder::close`] to leave it.
    ///
    /// `markerUnits` and `orient` are left at their spec defaults
    /// (`strokeWidth` and `0`); set them with a raw [`SvgBuilder::attr`]
    /// call after opening if a different value is needed.
    ///
    /// A call made after [`SvgBuilder::finish`] records a
    /// [`crate::writer::builder::BuildFlaw::UsedAfterFinish`] and opens
    /// nothing; see [`SvgBuilder::pattern`]'s documentation for the same note
    /// about a matching `close`.
    pub fn marker(&mut self, id: &str, width: f64, height: f64, ref_x: f64, ref_y: f64) -> FuncIri {
        let Some(defs) = self.ensure_defs() else {
            return FuncIri::new(format!("#{id}"));
        };
        let element = self
            .doc_mut()
            .create_element_ns(Some(SVG_NAMESPACE), "marker")
            .expect("\"marker\" is always a well-formed element name");
        self.doc_mut().append_child(defs, element).expect(
            "defs, once obtained from ensure_defs(), is always an attached element, \
             which always accepts a freshly created element child",
        );
        self.set_id(element, id);
        self.set_number_on(element, "markerWidth", width);
        self.set_number_on(element, "markerHeight", height);
        self.set_number_on(element, "refX", ref_x);
        self.set_number_on(element, "refY", ref_y);
        self.push_open(element);
        FuncIri::new(format!("#{id}"))
    }
}
