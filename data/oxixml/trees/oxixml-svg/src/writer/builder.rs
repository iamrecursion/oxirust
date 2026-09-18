//! The programmatic SVG document builder.
//!
//! [`SvgBuilder`] is the ergonomic replacement for the hand-rolled
//! `format!("<svg ...")` emitters scattered across the COOLJAPAN ecosystem
//! (see the crate documentation): a caller opens elements, sets typed
//! attributes on whichever element is currently in scope, and closes back
//! out, then serializes the whole thing. Every character of text and every
//! attribute value is escaped correctly because it all passes through
//! [`oxixml_dom::Document`]'s own serializer — nothing in this module ever
//! writes an angle bracket or an ampersand by hand.
//!
//! # The two cursors
//!
//! A builder tracks two positions, not one:
//!
//! * the **insertion point** (an internal stack of [`NodeId`]s) is *where
//!   new children go*: [`SvgBuilder::open`] appends a fresh element there and
//!   descends into it, [`SvgBuilder::close`] climbs back out, and
//!   [`SvgBuilder::text`]/[`SvgBuilder::comment`] append to whatever is
//!   currently on top;
//! * the **attribute target** is *which element a bare setter
//!   (`attr`/`attr_number`/`fill`/...) applies to*. `open` moves it to the
//!   element just opened; `close` moves it to the element just closed.
//!
//! The second rule is what makes the shape convenience methods in
//! [`crate::writer::shapes`] compose the way their names suggest:
//! `open`+attrs+`close` internally, so after `svg.rect(...)` the insertion
//! point is back at the rect's parent but the attribute target is still the
//! rect itself, and `svg.rect(...).fill(paint)` styles the rectangle, not its
//! parent.
//!
//! # Guard rails, not panics
//!
//! Nothing in this module panics. An operation that cannot succeed — an
//! element or attribute name that is not well-formed, a non-finite number, an
//! unbalanced [`SvgBuilder::close`], any call made after [`SvgBuilder::finish`]
//! — is recorded as a [`BuildFlaw`] instead, and the builder is left in a
//! well-defined state so a long chain of calls is never interrupted.
//! [`SvgBuilder::flaws`] returns everything recorded so far.

use oxixml_dom::{Document, NodeId, SerializeOptions};

use crate::ns::SVG_NAMESPACE;
use crate::values::aspect::{PreserveAspectRatio, ViewBox};
use crate::values::color::Color;
use crate::values::length::Length;
use crate::values::paint::Paint;
use crate::values::path::PathData;
use crate::values::transform::TransformList;
use crate::writer::fmtnum;

/// A defect recorded while building a document, instead of a panic or a
/// silently discarded operation.
///
/// See the [module documentation](self) for why [`SvgBuilder`] never panics.
/// Every variant names the attribute or element name involved, when there is
/// one, so a caller can find the offending call site.
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub enum BuildFlaw {
    /// [`SvgBuilder::open`] (or a shape convenience method) was given a name
    /// that is not a well-formed XML element name. A plain `<g>` was opened
    /// in its place, so the surrounding open/close nesting stays balanced.
    InvalidElementName {
        /// The name that was rejected.
        name: String,
    },
    /// An attribute setter was given a name that is not a well-formed XML
    /// attribute name, or a value containing a character XML text cannot
    /// represent. The attribute was not set.
    InvalidAttribute {
        /// The attribute name involved.
        name: String,
    },
    /// A numeric setter was given a `NaN` or infinite value, which has no
    /// valid SVG `<number>` representation. The attribute was not set.
    NonFiniteNumber {
        /// The attribute name involved.
        name: String,
    },
    /// [`SvgBuilder::text`] or [`SvgBuilder::comment`] was given content
    /// containing a character XML cannot represent (or, for a comment, the
    /// literal sequence `--`, or a trailing `-`). The node was not appended.
    InvalidContent,
    /// [`SvgBuilder::close`] was called with no element open beyond the
    /// document's own root, which an ordinary `close` never pops.
    UnbalancedClose,
    /// A structural or content-appending call — [`SvgBuilder::open`],
    /// [`SvgBuilder::text`], [`SvgBuilder::comment`], [`SvgBuilder::open_defs`],
    /// or one of the resource builders in [`crate::writer::paint_defs`] — was
    /// made after [`SvgBuilder::finish`] had already taken this builder's
    /// document. The call had no effect: nothing was created or appended, and
    /// the insertion point and attribute target are unchanged.
    UsedAfterFinish,
}

impl std::fmt::Display for BuildFlaw {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::InvalidElementName { name } => {
                write!(f, "{name:?} is not a well-formed XML element name")
            }
            Self::InvalidAttribute { name } => write!(
                f,
                "the attribute {name:?} could not be set (invalid name or value)"
            ),
            Self::NonFiniteNumber { name } => write!(
                f,
                "attribute {name:?} was given a non-finite number and was left unset"
            ),
            Self::InvalidContent => f.write_str(
                "text or comment content could not be represented in XML and was dropped",
            ),
            Self::UnbalancedClose => {
                f.write_str("close() was called with no element open beyond the document root")
            }
            Self::UsedAfterFinish => f.write_str(
                "the builder was used after finish() had already taken its document; the call had no effect",
            ),
        }
    }
}

impl std::error::Error for BuildFlaw {}

/// A programmatic, chainable builder for an SVG document.
///
/// See the [module documentation](self) for the two-cursor model and the
/// no-panic guard-rail policy. Construct one with [`SvgBuilder::new`],
/// [`SvgBuilder::plain`], or [`SvgBuilder::with_size_and_view_box`]; shape
/// convenience methods live on this type too, added by
/// [`crate::writer::shapes`] and [`crate::writer::paint_defs`] in their own
/// `impl` blocks.
#[derive(Debug)]
pub struct SvgBuilder {
    doc: Document,
    /// The insertion-point stack; `stack[0]` is always the document's root
    /// `<svg>` element and is never popped by an ordinary [`Self::close`].
    stack: Vec<NodeId>,
    /// The attribute target; see the [module documentation](self).
    target: NodeId,
    precision: Option<usize>,
    flaws: Vec<BuildFlaw>,
    /// Set once [`Self::finish`] has taken this builder's document. Every
    /// structural or content-appending call checks
    /// [`Self::guard_not_finished`] first and, once this is set, records a
    /// [`BuildFlaw::UsedAfterFinish`] instead of touching the (fresh, mostly
    /// empty) document underneath — see [`Self::finish`]'s own documentation
    /// for why a builder must stay usable, but inert, past that point.
    finished: bool,
}

impl SvgBuilder {
    /// Creates a builder for an `<svg>` document with the given pixel
    /// `width` and `height`, written as plain (unitless) numbers.
    ///
    /// A `NaN` or infinite `width`/`height` is recorded as a
    /// [`BuildFlaw::NonFiniteNumber`] and left unset, exactly as
    /// [`Self::attr_number`] would.
    #[must_use]
    pub fn new(width: f64, height: f64) -> Self {
        let mut builder = Self::plain();
        builder.attr_number("width", width);
        builder.attr_number("height", height);
        builder
    }

    /// Creates a builder for a bare `<svg>` root with no `width`/`height`
    /// attributes.
    #[must_use]
    pub fn plain() -> Self {
        let mut doc = Document::new();
        // Declared once, crate-wide: every element's `id` attribute (set via
        // `attr_id` or the raw `attr("id", ...)`) becomes eligible for
        // `Document::get_element_by_id`, which the gradient/pattern/clip/
        // mask/marker machinery in `crate::writer::paint_defs` relies on.
        doc.declare_id_attribute(None, "id");
        let root = doc
            .create_element_ns(Some(SVG_NAMESPACE), "svg")
            .expect("\"svg\" is a well-formed name in the SVG namespace");
        let document_node = doc.root();
        doc.append_child(document_node, root)
            .expect("a fresh document accepts its first and only document element");
        // Declared explicitly on the DOM tree itself, rather than left to the
        // serializer's namespace-fixup default, so the document is
        // self-describing regardless of the `SerializeOptions` a caller
        // later chooses.
        doc.declare_namespace(root, None, SVG_NAMESPACE)
            .expect("declaring `xmlns` on a freshly created root cannot fail");
        Self {
            doc,
            stack: vec![root],
            target: root,
            precision: None,
            flaws: Vec::new(),
            finished: false,
        }
    }

    /// Sets this builder's `viewBox` attribute, returning `self` for further
    /// chaining. Equivalent to `self.attr_view_box(view_box)` spelled as a
    /// constructor-style method for use right after [`Self::new`].
    #[must_use]
    pub fn with_view_box(mut self, view_box: ViewBox) -> Self {
        self.attr_view_box(view_box);
        self
    }

    /// Creates a builder for an `<svg>` document with the given size and
    /// `viewBox` in one call. Equivalent to
    /// `Self::new(width, height).with_view_box(view_box)`.
    #[must_use]
    pub fn with_size_and_view_box(width: f64, height: f64, view_box: ViewBox) -> Self {
        Self::new(width, height).with_view_box(view_box)
    }

    // ---------------------------------------------------------- precision

    /// Sets the number of decimal places `attr_number`, path data, and point
    /// lists round to when serializing, or `None` (the default) for the
    /// shortest round-tripping form.
    ///
    /// Typed setters that delegate to another value's own
    /// [`Display`](std::fmt::Display) — `attr_length`, `attr_color`,
    /// `attr_paint`, `attr_transform`, `attr_view_box`,
    /// `attr_preserve_aspect_ratio` — are not affected: only bare numbers
    /// and the two microsyntaxes built entirely out of them (`d`, `points`)
    /// respect this setting.
    pub fn set_precision(&mut self, precision: Option<usize>) -> &mut Self {
        self.precision = precision;
        self
    }

    /// The precision most recently set with [`Self::set_precision`].
    #[must_use]
    pub const fn precision(&self) -> Option<usize> {
        self.precision
    }

    // ------------------------------------------------------------ cursors

    /// The element new children (via [`Self::open`], [`Self::text`],
    /// [`Self::comment`]) are appended to.
    fn insertion_point(&self) -> NodeId {
        self.stack
            .last()
            .copied()
            .unwrap_or_else(|| self.doc.root())
    }

    /// The document's root `<svg>` element, regardless of how deep the
    /// insertion point currently is nested.
    pub(crate) fn root_id(&self) -> NodeId {
        self.stack
            .first()
            .copied()
            .unwrap_or_else(|| self.doc.root())
    }

    /// Shared, crate-internal read access to the underlying document, for
    /// [`crate::writer::paint_defs`]'s lower-level element construction
    /// (which targets `<defs>` rather than the current insertion point).
    pub(crate) fn doc(&self) -> &Document {
        &self.doc
    }

    /// The `&mut` counterpart of [`Self::doc`].
    pub(crate) fn doc_mut(&mut self) -> &mut Document {
        &mut self.doc
    }

    /// Pushes `element` as the new insertion point and attribute target.
    /// Used by [`Self::open`] and by the "open for content" resource
    /// builders in [`crate::writer::paint_defs`] (`pattern`, `clip_path`,
    /// `mask`, `marker`).
    pub(crate) fn push_open(&mut self, element: NodeId) {
        self.stack.push(element);
        self.target = element;
    }

    /// Records a defect without panicking; see the [module
    /// documentation](self).
    pub(crate) fn record_flaw(&mut self, flaw: BuildFlaw) {
        self.flaws.push(flaw);
    }

    /// Records a [`BuildFlaw::UsedAfterFinish`] and returns `false` when
    /// [`Self::finish`] has already taken this builder's document; otherwise
    /// does nothing and returns `true`.
    ///
    /// Every method that appends a node to the tree — [`Self::open`],
    /// [`Self::text`], [`Self::comment`], and, via [`Self::root_id`],
    /// [`crate::writer::paint_defs`]'s `ensure_defs` — checks this first.
    /// While it keeps returning `true`, [`Self::insertion_point`] and
    /// [`Self::root_id`] are guaranteed to name an [`oxixml_dom::NodeKind::Element`]
    /// that is already attached to the tree: the stack starts at [`Self::plain`]'s
    /// freshly appended root element, `close` only pops, and every push comes
    /// from a node [`Self::open`] or a `paint_defs` resource builder just
    /// created and appended successfully. An element unconditionally accepts
    /// further element/text/comment children, so every `append_child` call
    /// gated behind this guard is infallible — unlike a bare
    /// [`oxixml_dom::Document`]'s own root, which accepts at most one element
    /// child, and which [`Self::finish`] leaves the insertion point pointing
    /// at.
    pub(crate) fn guard_not_finished(&mut self) -> bool {
        if self.finished {
            self.record_flaw(BuildFlaw::UsedAfterFinish);
            false
        } else {
            true
        }
    }

    /// Sets a raw string attribute on `element` specifically (as opposed to
    /// [`Self::attr`], which always targets [`Self::target`]), recording a
    /// [`BuildFlaw::InvalidAttribute`] instead of failing.
    pub(crate) fn set_attr_on(&mut self, element: NodeId, name: &str, value: &str) {
        if self.doc.set_attribute(element, name, value).is_err() {
            self.record_flaw(BuildFlaw::InvalidAttribute {
                name: name.to_owned(),
            });
        }
    }

    /// The `element`-targeted counterpart of [`Self::attr_number`], used by
    /// [`crate::writer::paint_defs`] to set geometry attributes on a
    /// gradient/pattern/marker element without disturbing the current
    /// attribute target.
    pub(crate) fn set_number_on(&mut self, element: NodeId, name: &str, value: f64) {
        if !fmtnum::is_attr_safe(value) {
            self.record_flaw(BuildFlaw::NonFiniteNumber {
                name: name.to_owned(),
            });
            return;
        }
        let text = fmtnum::fmt_number_with(value, self.precision);
        self.set_attr_on(element, name, &text);
    }

    // ---------------------------------------------------------- elements

    /// Opens a new child element named `name` under the current insertion
    /// point, descends into it, and moves the attribute target to it.
    ///
    /// `name` is created in [`SVG_NAMESPACE`]. When it is not a well-formed
    /// XML element name, a plain `<g>` is opened in its place and a
    /// [`BuildFlaw::InvalidElementName`] is recorded, so the open/close
    /// nesting stays balanced and every later call in the chain still has a
    /// real element to act on.
    ///
    /// A call made after [`Self::finish`] records a
    /// [`BuildFlaw::UsedAfterFinish`] and has no effect; see that method's
    /// documentation.
    pub fn open(&mut self, name: &str) -> &mut Self {
        if !self.guard_not_finished() {
            return self;
        }
        let parent = self.insertion_point();
        let element = match self.doc.create_element_ns(Some(SVG_NAMESPACE), name) {
            Ok(element) => element,
            Err(_) => {
                self.flaws.push(BuildFlaw::InvalidElementName {
                    name: name.to_owned(),
                });
                self.doc
                    .create_element_ns(Some(SVG_NAMESPACE), "g")
                    .expect("\"g\" is always a well-formed element name")
            }
        };
        self.doc.append_child(parent, element).expect(
            "guard_not_finished() guarantees the insertion point is an attached element, \
             which always accepts a freshly created element child",
        );
        self.push_open(element);
        self
    }

    /// Closes the most recently opened element: the insertion point climbs
    /// back to its parent, and the attribute target moves to the element
    /// that was just closed (so `svg.open("rect")...close()` leaves
    /// attribute setters still targeting the rect, not its parent).
    ///
    /// Calling this with nothing open beyond the document's own root records
    /// a [`BuildFlaw::UnbalancedClose`] and otherwise does nothing: the root
    /// itself is never popped.
    pub fn close(&mut self) -> &mut Self {
        if self.stack.len() > 1 {
            if let Some(popped) = self.stack.pop() {
                self.target = popped;
            }
        } else {
            self.flaws.push(BuildFlaw::UnbalancedClose);
        }
        self
    }

    // ----------------------------------------------------------- content

    /// Appends a text node to the current insertion point. XML-escapes on
    /// serialization; never writes an unescaped `<`, `>`, or `&` itself.
    ///
    /// Content containing a character XML text cannot represent is not
    /// appended; a [`BuildFlaw::InvalidContent`] is recorded instead. A call
    /// made after [`Self::finish`] records a [`BuildFlaw::UsedAfterFinish`]
    /// and has no effect; see that method's documentation.
    pub fn text(&mut self, content: &str) -> &mut Self {
        if !self.guard_not_finished() {
            return self;
        }
        let parent = self.insertion_point();
        match self.doc.create_text_node(content) {
            Ok(node) => {
                self.doc.append_child(parent, node).expect(
                    "guard_not_finished() guarantees the insertion point is an attached \
                     element, which always accepts a freshly created text child",
                );
            }
            Err(_) => self.flaws.push(BuildFlaw::InvalidContent),
        }
        self
    }

    /// Appends a comment node to the current insertion point.
    ///
    /// Content that cannot be represented as an XML comment (a character
    /// XML forbids, the literal sequence `--`, or a trailing `-`) is not
    /// appended; a [`BuildFlaw::InvalidContent`] is recorded instead. A call
    /// made after [`Self::finish`] records a [`BuildFlaw::UsedAfterFinish`]
    /// and has no effect; see that method's documentation.
    pub fn comment(&mut self, content: &str) -> &mut Self {
        if !self.guard_not_finished() {
            return self;
        }
        let parent = self.insertion_point();
        match self.doc.create_comment(content) {
            Ok(node) => {
                self.doc.append_child(parent, node).expect(
                    "guard_not_finished() guarantees the insertion point is an attached \
                     element, which always accepts a freshly created comment child",
                );
            }
            Err(_) => self.flaws.push(BuildFlaw::InvalidContent),
        }
        self
    }

    // ------------------------------------------------------- attributes

    /// Sets a raw string attribute on the current attribute target (see the
    /// [module documentation](self)).
    ///
    /// When `name` is not a well-formed attribute name, or `value` contains
    /// a character XML text cannot represent, a
    /// [`BuildFlaw::InvalidAttribute`] is recorded and the attribute is not
    /// set.
    pub fn attr(&mut self, name: &str, value: &str) -> &mut Self {
        let target = self.target;
        self.set_attr_on(target, name, value);
        self
    }

    /// Sets a bare-number attribute, formatted per [`Self::precision`] (see
    /// [`crate::writer::fmtnum`]).
    ///
    /// A `NaN` or infinite `value` is not a valid SVG `<number>`; it is
    /// recorded as a [`BuildFlaw::NonFiniteNumber`] and the attribute is
    /// left unset, rather than writing `NaN`/`inf` into the document.
    pub fn attr_number(&mut self, name: &str, value: f64) -> &mut Self {
        let target = self.target;
        self.set_number_on(target, name, value);
        self
    }

    /// Sets an attribute from a [`Length`], via its own
    /// [`Display`](std::fmt::Display) (not affected by [`Self::precision`]).
    pub fn attr_length(&mut self, name: &str, value: Length) -> &mut Self {
        let text = value.to_string();
        self.attr(name, &text)
    }

    /// Sets an attribute from a [`Color`], via its own
    /// [`Display`](std::fmt::Display) (the `#rrggbb`/`#rrggbbaa` hex form).
    pub fn attr_color(&mut self, name: &str, value: Color) -> &mut Self {
        let text = value.to_string();
        self.attr(name, &text)
    }

    /// Sets an attribute from a [`Paint`], via its own
    /// [`Display`](std::fmt::Display).
    pub fn attr_paint(&mut self, name: &str, value: Paint) -> &mut Self {
        let text = value.to_string();
        self.attr(name, &text)
    }

    /// Sets an attribute from a [`TransformList`], via its own
    /// [`Display`](std::fmt::Display).
    pub fn attr_transform(&mut self, name: &str, value: &TransformList) -> &mut Self {
        let text = value.to_string();
        self.attr(name, &text)
    }

    /// Sets the `d` attribute from a [`PathData`], rounded to
    /// [`Self::precision`] decimal places when one is set, or written in
    /// [`PathData`]'s canonical (fully explicit, one-space-separated) form
    /// otherwise — minification is a serializer *mode*
    /// [`PathData::to_string_compact`] offers explicitly, not something this
    /// method applies implicitly.
    pub fn attr_path_d(&mut self, value: &PathData) -> &mut Self {
        let text = match self.precision {
            Some(precision) => value.fmt_with_precision(precision),
            None => value.to_string(),
        };
        self.attr("d", &text)
    }

    /// Sets the `points` attribute from a list of `(x, y)` pairs, formatted
    /// per [`Self::precision`].
    pub fn attr_points(&mut self, points: &[[f64; 2]]) -> &mut Self {
        let text = fmtnum::format_points(points, self.precision);
        self.attr("points", &text)
    }

    /// Sets the `viewBox` attribute from a [`ViewBox`], via its own
    /// [`Display`](std::fmt::Display).
    pub fn attr_view_box(&mut self, value: ViewBox) -> &mut Self {
        let text = value.to_string();
        self.attr("viewBox", &text)
    }

    /// Sets the `preserveAspectRatio` attribute from a
    /// [`PreserveAspectRatio`], via its own [`Display`](std::fmt::Display).
    pub fn attr_preserve_aspect_ratio(&mut self, value: PreserveAspectRatio) -> &mut Self {
        let text = value.to_string();
        self.attr("preserveAspectRatio", &text)
    }

    /// Sets the SVG 2 plain `href` attribute (not the legacy
    /// `xlink:href`).
    pub fn attr_href(&mut self, href: &str) -> &mut Self {
        self.attr("href", href)
    }

    /// Sets the `class` attribute.
    pub fn attr_class(&mut self, class: &str) -> &mut Self {
        self.attr("class", class)
    }

    /// Sets the `id` attribute.
    ///
    /// Every element's `id` is declared as an `ID`-typed attribute
    /// document-wide at construction (see [`Self::plain`]), so a value set
    /// here is immediately resolvable through
    /// [`oxixml_dom::Document::get_element_by_id`].
    pub fn attr_id(&mut self, id: &str) -> &mut Self {
        self.attr("id", id)
    }

    /// Sets the `style` attribute (a raw CSS declaration list).
    pub fn attr_style(&mut self, style: &str) -> &mut Self {
        self.attr("style", style)
    }

    // --------------------------------------------------------- finishing

    /// Every defect recorded so far; see the [module documentation](self).
    /// Empty when nothing went wrong.
    #[must_use]
    pub fn flaws(&self) -> &[BuildFlaw] {
        &self.flaws
    }

    /// Takes the built [`Document`] out of this builder.
    ///
    /// This takes `&mut self` rather than consuming the builder outright, so
    /// that a chain of `&mut Self`-returning calls (`open`, `attr_number`,
    /// the shape convenience methods, ...) can end in `.finish()` directly —
    /// `SvgBuilder::new(w, h).rect(x, y, w, h).finish()` reads naturally and
    /// compiles, which it would not if `finish` moved `self`. Internally the
    /// document is swapped out for a fresh, empty one
    /// ([`std::mem::replace`]); the builder is left usable but inert
    /// afterward — every further structural or content call
    /// ([`Self::open`], [`Self::text`], [`Self::comment`], [`Self::open_defs`],
    /// or any of [`crate::writer::paint_defs`]'s resource builders) records a
    /// [`BuildFlaw::UsedAfterFinish`] and does nothing else, and every
    /// attribute setter records its ordinary failure flaw the same way it
    /// would for any other invalid target, so nothing ever panics and no
    /// second, unrelated document is ever built by accident. Calling `finish`
    /// is meant to be the last thing a caller does with a builder; use
    /// `to_string()` (via this type's [`Display`](std::fmt::Display)
    /// implementation)/[`Self::to_string_pretty`]/[`Self::to_string_with`]
    /// instead to serialize without disturbing an in-progress builder.
    pub fn finish(&mut self) -> Document {
        let finished = std::mem::replace(&mut self.doc, Document::new());
        let document_node = self.doc.root();
        self.stack = vec![document_node];
        self.target = document_node;
        self.finished = true;
        finished
    }

    /// Serializes this builder's document with the given `options`.
    ///
    /// This is the escape hatch for anything the plain `to_string()`/
    /// [`Self::to_string_pretty`] do not expose directly: including the XML
    /// declaration ([`oxixml_dom::XmlDeclaration::Include`]), a custom
    /// indent or newline string, or disabling empty-element folding or
    /// namespace fixup.
    #[must_use]
    pub fn to_string_with(&self, options: &SerializeOptions) -> String {
        self.doc.to_xml_with(options)
    }

    /// Serializes this builder's document with two-space pretty-printed
    /// indentation and no XML declaration.
    #[must_use]
    pub fn to_string_pretty(&self) -> String {
        self.to_string_with(&SerializeOptions::pretty())
    }
}

impl std::fmt::Display for SvgBuilder {
    /// Serializes this builder's document compactly (no indentation, no XML
    /// declaration) — the `Display` implementation is what
    /// `to_string()`/`format!("{svg}")` on an [`SvgBuilder`] use, so that a
    /// chain such as `SvgBuilder::new(w, h).rect(x, y, w, h).to_string()`
    /// reads naturally without an inherent `to_string` method shadowing it.
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.to_string_with(&SerializeOptions::new()))
    }
}

#[cfg(test)]
mod tests {
    use oxixml_dom::{NodeKind, XmlDeclaration, build};

    use super::*;
    use crate::values::color::Color;
    use crate::values::length::LengthUnit;
    use crate::values::transform::Transform;
    use crate::writer::paint_defs::{CoordinateUnits, GradientStop};
    use crate::writer::pathbuilder::PathBuilder;

    /// Finds the first descendant element of `root` named `local_name`, in
    /// document order.
    fn find_element(doc: &oxixml_dom::Document, root: NodeId, local_name: &str) -> NodeId {
        doc.descendants(root)
            .find(|&id| {
                doc.element(id).map(oxixml_dom::ElementData::local_name) == Some(local_name)
            })
            .unwrap_or_else(|| panic!("no <{local_name}> element found"))
    }

    #[test]
    fn new_sets_width_height_and_declares_the_svg_namespace() {
        let svg = SvgBuilder::new(200.0, 100.0);
        let text = svg.to_string();
        assert!(text.starts_with("<svg"));
        assert!(text.contains("width=\"200\""));
        assert!(text.contains("height=\"100\""));
        assert!(text.contains("xmlns=\"http://www.w3.org/2000/svg\""));
        assert!(svg.flaws().is_empty());
    }

    #[test]
    fn plain_declares_the_namespace_but_sets_no_size() {
        let svg = SvgBuilder::plain();
        let text = svg.to_string();
        assert!(!text.contains("width="));
        assert!(!text.contains("height="));
        assert!(text.contains("xmlns=\"http://www.w3.org/2000/svg\""));
    }

    #[test]
    fn with_view_box_and_with_size_and_view_box_agree() {
        let vb = ViewBox::new(0.0, 0.0, 100.0, 50.0);
        let a = SvgBuilder::new(200.0, 100.0).with_view_box(vb);
        let b = SvgBuilder::with_size_and_view_box(200.0, 100.0, vb);
        assert_eq!(a.to_string(), b.to_string());
        assert!(a.to_string().contains("viewBox=\"0 0 100 50\""));
    }

    /// The discriminating test for the two-cursor model: after a shape
    /// convenience method returns, a style shorthand must style *that
    /// shape*, not whatever contains it.
    #[test]
    fn style_shorthand_targets_the_shape_just_closed_not_its_parent() {
        let mut svg = SvgBuilder::new(100.0, 100.0);
        svg.rect(0.0, 0.0, 10.0, 10.0);
        svg.fill(Paint::Color(Color::rgb(0xFF, 0x00, 0x00)));

        let reparsed = build::parse_str(&svg.to_string()).expect("well-formed XML");
        let root = reparsed.document_element().expect("a document element");
        assert_eq!(reparsed.get_attribute(root, "fill"), None);
        let rect = find_element(&reparsed, root, "rect");
        assert_eq!(reparsed.get_attribute(rect, "fill"), Some("#ff0000"));
    }

    #[test]
    fn one_liner_chain_ending_in_finish_compiles_and_works() {
        let doc = SvgBuilder::new(20.0, 20.0)
            .rect(0.0, 0.0, 5.0, 5.0)
            .finish();
        assert!(doc.document_element().is_some());
    }

    #[test]
    fn one_liner_chain_ending_in_to_string_compiles_and_works() {
        let xml = SvgBuilder::new(20.0, 20.0)
            .rect(0.0, 0.0, 5.0, 5.0)
            .to_string();
        assert!(xml.contains("<rect"));
    }

    #[test]
    fn finish_leaves_the_builder_inert_but_still_panic_free() {
        let mut svg = SvgBuilder::new(20.0, 20.0);
        svg.rect(0.0, 0.0, 5.0, 5.0);
        let doc = svg.finish();
        assert!(doc.document_element().is_some());
        // Further use after `finish` is out of contract but must not panic:
        // it records a flaw instead of touching anything.
        svg.attr_number("x", 1.0);
        assert!(!svg.flaws().is_empty());
    }

    /// The scenario `finish`'s own documentation promises: every structural
    /// or content call made afterward — including the very first one, which
    /// a bare, freshly swapped-in [`oxixml_dom::Document`] would otherwise
    /// happily accept as its first element or comment child — records
    /// [`BuildFlaw::UsedAfterFinish`] and does nothing, across every method
    /// that appends to the tree: [`SvgBuilder::open`], [`SvgBuilder::text`],
    /// [`SvgBuilder::comment`], [`SvgBuilder::open_defs`], and
    /// [`SvgBuilder::linear_gradient`] (representative of every
    /// `paint_defs` resource builder, which all route through the same
    /// `ensure_defs`). A second `finish()` afterward proves it conclusively:
    /// the fresh document received nothing at all.
    #[test]
    fn every_structural_call_after_finish_records_used_after_finish_and_never_panics() {
        let mut svg = SvgBuilder::new(20.0, 20.0);
        svg.rect(0.0, 0.0, 5.0, 5.0);
        let _ = svg.finish();

        svg.open("rect");
        svg.text("stray text");
        svg.comment("stray comment");
        svg.open_defs();
        let iri = svg.linear_gradient(
            "ignored",
            CoordinateUnits::UserSpaceOnUse,
            0.0,
            0.0,
            1.0,
            0.0,
            &[(0.0, Color::WHITE), (1.0, Color::BLACK)],
        );
        assert_eq!(
            iri.reference(),
            "#ignored",
            "the reference is still returned"
        );

        let used_after_finish_count = svg
            .flaws()
            .iter()
            .filter(|flaw| matches!(flaw, BuildFlaw::UsedAfterFinish))
            .count();
        assert_eq!(
            used_after_finish_count,
            5,
            "open, text, comment, open_defs, and linear_gradient must each record \
             UsedAfterFinish exactly once: {:?}",
            svg.flaws()
        );

        let second = svg.finish();
        assert!(
            second.document_element().is_none(),
            "nothing any of the calls above did reached the fresh document"
        );
    }

    /// The exact shape that first exposed this bug: a shape convenience
    /// method (not a raw `open`/`text`/`comment` call) used twice after
    /// `finish`. `rect` opens, sets four attributes, and closes internally,
    /// so it produces a mix of flaw kinds — [`BuildFlaw::UsedAfterFinish`]
    /// from `open`, an attribute flaw from each `attr_number` (the target is
    /// the fresh document's own document node, not an element), and
    /// [`BuildFlaw::UnbalancedClose`] from `close` (the guarded-off `open`
    /// never pushed anything to unwind) — this test only asserts that
    /// `UsedAfterFinish` is among them and, above all, that neither call
    /// panics.
    #[test]
    fn shape_convenience_methods_after_finish_never_panic_either() {
        let mut svg = SvgBuilder::new(10.0, 10.0);
        svg.rect(0.0, 0.0, 1.0, 1.0);
        let first = svg.finish();
        assert!(first.document_element().is_some());

        svg.rect(0.0, 0.0, 1.0, 1.0);
        svg.rect(0.0, 0.0, 1.0, 1.0);
        assert!(
            svg.flaws()
                .iter()
                .any(|flaw| matches!(flaw, BuildFlaw::UsedAfterFinish))
        );

        let second = svg.finish();
        assert!(
            second.document_element().is_none(),
            "neither rect() call after finish() reached the fresh document"
        );
    }

    #[test]
    fn full_document_round_trips_structure_namespace_and_text_escaping() {
        let mut svg = SvgBuilder::new(200.0, 100.0);
        svg.rect(10.0, 10.0, 50.0, 30.0);
        svg.fill(Paint::Color(Color::rgb(0x10, 0x20, 0x30)));

        let path = PathBuilder::new()
            .move_to(0.0, 0.0)
            .line_to(10.0, 10.0)
            .close()
            .finish();
        svg.path(&path);

        let content = "<tag> & \"quote\" 'apos' \u{65E5}\u{672C}\u{8A9E}";
        svg.text_at(5.0, 90.0, content);
        svg.comment(" a comment ");

        let xml = svg.to_string();
        assert!(xml.contains("&lt;tag&gt;"));
        assert!(xml.contains("&amp;"));
        assert!(
            xml.contains('\u{65E5}'),
            "non-ASCII text is kept literal, not escaped"
        );
        assert!(xml.contains("<!-- a comment -->"));

        let reparsed = build::parse_str(&xml).expect("builder output is well-formed XML");
        let root = reparsed.document_element().expect("a document element");
        assert_eq!(
            reparsed
                .element(root)
                .and_then(oxixml_dom::ElementData::namespace_uri),
            Some(SVG_NAMESPACE)
        );
        assert_eq!(
            reparsed.lookup_namespace_uri(root, None),
            Some(SVG_NAMESPACE)
        );
        for id in reparsed.descendants(root) {
            if reparsed.kind(id) == Some(NodeKind::Element) {
                assert_eq!(
                    reparsed
                        .element(id)
                        .and_then(oxixml_dom::ElementData::namespace_uri),
                    Some(SVG_NAMESPACE),
                    "every descendant element must resolve to the SVG namespace"
                );
            }
        }

        let text_el = find_element(&reparsed, root, "text");
        assert_eq!(reparsed.text_content(text_el).as_deref(), Some(content));
    }

    #[test]
    fn attribute_values_round_trip_through_their_own_value_parsers() {
        let mut svg = SvgBuilder::new(100.0, 100.0);
        svg.open("rect");
        svg.attr_length("x", Length::new(1.5, LengthUnit::Px));
        svg.attr_color("fill", Color::rgb(0x11, 0x22, 0x33));
        svg.attr_transform(
            "transform",
            &TransformList::new(vec![Transform::Translate {
                tx: 1.0,
                ty: Some(2.0),
            }]),
        );
        svg.close();
        let path = PathBuilder::new()
            .move_to(0.0, 0.0)
            .line_to(10.0, 10.0)
            .finish();
        svg.path(&path);

        let reparsed = build::parse_str(&svg.to_string()).expect("well-formed XML");
        let root = reparsed.document_element().expect("a document element");
        let rect = find_element(&reparsed, root, "rect");

        let x = reparsed
            .get_attribute(rect, "x")
            .expect("x attribute present");
        assert_eq!(Length::parse(x), Ok(Length::new(1.5, LengthUnit::Px)));

        let fill = reparsed
            .get_attribute(rect, "fill")
            .expect("fill attribute present");
        assert_eq!(Color::parse(fill), Ok(Color::rgb(0x11, 0x22, 0x33)));

        let transform = reparsed
            .get_attribute(rect, "transform")
            .expect("transform attribute present");
        let matrix = TransformList::parse(transform)
            .expect("valid transform list")
            .to_matrix();
        assert_eq!(matrix.apply(0.0, 0.0), (1.0, 2.0));

        let path_el = find_element(&reparsed, root, "path");
        let d = reparsed
            .get_attribute(path_el, "d")
            .expect("d attribute present");
        assert_eq!(PathData::parse(d).expect("valid path data"), path);
    }

    #[test]
    fn gradient_reference_resolves_by_id() {
        let mut svg = SvgBuilder::new(100.0, 100.0);
        let stops = [
            GradientStop::new(0.0, Color::WHITE),
            GradientStop::new(1.0, Color::BLACK),
        ];
        let iri = svg.linear_gradient(
            "g1",
            CoordinateUnits::ObjectBoundingBox,
            0.0,
            0.0,
            1.0,
            0.0,
            &stops,
        );
        svg.rect(0.0, 0.0, 100.0, 100.0);
        svg.fill(Paint::Iri {
            href: iri.reference().into(),
            fallback: None,
        });

        let doc = svg.finish();
        let found = doc
            .get_element_by_id("g1")
            .expect("id declared document-wide resolves");
        assert_eq!(
            doc.element(found).map(oxixml_dom::ElementData::local_name),
            Some("linearGradient")
        );
    }

    /// The end-to-end scenario the writer module exists for: a gradient, a
    /// filled rect referencing it, a built path, and escaped/non-ASCII text
    /// all in one document. Checks every facet at once — structure, every
    /// element's namespace (including the gradient and its `<stop>`s inside
    /// `<defs>`), attribute values round-tripping through their own value
    /// parsers, the `url(#id)` reference resolving, and text escaping — so a
    /// regression in how these pieces compose together is caught even if no
    /// single-facet test above happens to.
    #[test]
    fn a_full_mixed_document_round_trips_every_facet_at_once() {
        let mut svg =
            SvgBuilder::with_size_and_view_box(200.0, 100.0, ViewBox::new(0.0, 0.0, 200.0, 100.0));

        let stops = [
            GradientStop::new(0.0, Color::rgb(0xFF, 0x00, 0x00)),
            GradientStop::new(1.0, Color::rgb(0x00, 0x00, 0xFF)),
        ];
        let gradient = svg.linear_gradient(
            "mix-grad",
            CoordinateUnits::ObjectBoundingBox,
            0.0,
            0.0,
            1.0,
            0.0,
            &stops,
        );

        svg.rect(10.0, 10.0, 50.0, 30.0);
        svg.fill(Paint::Iri {
            href: gradient.reference().into(),
            fallback: None,
        });

        let path = PathBuilder::new()
            .move_to(0.0, 0.0)
            .line_to(20.0, 20.0)
            .close()
            .finish();
        svg.path(&path);

        let content = "<r&d> \"quotes\" 'apostrophes' \u{65E5}\u{672C}\u{8A9E}";
        svg.text_at(5.0, 90.0, content);

        let xml = svg.to_string();
        assert!(xml.contains("&lt;r&amp;d&gt;"));
        assert!(xml.contains('\u{65E5}'), "non-ASCII text stays literal");

        let mut reparsed = build::parse_str(&xml).expect("builder output is well-formed XML");
        let root = reparsed.document_element().expect("a document element");

        for id in reparsed.descendants(root) {
            if reparsed.kind(id) == Some(NodeKind::Element) {
                assert_eq!(
                    reparsed
                        .element(id)
                        .and_then(oxixml_dom::ElementData::namespace_uri),
                    Some(SVG_NAMESPACE),
                    "every element, including the gradient and its stops in <defs>, is in the SVG namespace"
                );
            }
        }

        // A fresh reparse has no DTD declaring `id` an ID-type attribute, so
        // `get_element_by_id` honors nothing yet; a consumer of arbitrary
        // parsed SVG declares it itself, exactly as `SvgBuilder::plain` does
        // for the document it builds.
        reparsed.declare_id_attribute(None, "id");
        let resolved = reparsed
            .get_element_by_id("mix-grad")
            .expect("the gradient's id resolves once `id` is declared as an ID attribute");
        assert_eq!(
            reparsed
                .element(resolved)
                .map(oxixml_dom::ElementData::local_name),
            Some("linearGradient")
        );

        let rect = find_element(&reparsed, root, "rect");
        let fill = reparsed
            .get_attribute(rect, "fill")
            .expect("fill attribute present");
        assert_eq!(
            Paint::parse(fill),
            Ok(Paint::Iri {
                href: "#mix-grad".into(),
                fallback: None,
            })
        );

        let path_el = find_element(&reparsed, root, "path");
        let d = reparsed
            .get_attribute(path_el, "d")
            .expect("d attribute present");
        assert_eq!(PathData::parse(d).expect("valid path data"), path);

        let text_el = find_element(&reparsed, root, "text");
        assert_eq!(reparsed.text_content(text_el).as_deref(), Some(content));
    }

    #[test]
    fn open_defs_and_a_gradient_share_a_single_defs_element() {
        let mut svg = SvgBuilder::new(10.0, 10.0);
        let stops = [(0.0, Color::WHITE), (1.0, Color::BLACK)];
        svg.linear_gradient(
            "g1",
            CoordinateUnits::UserSpaceOnUse,
            0.0,
            0.0,
            1.0,
            0.0,
            &stops,
        );
        svg.open_defs();
        svg.open("symbol");
        svg.attr_id("sym1");
        svg.close();
        svg.close();

        let doc = svg.finish();
        let root = doc.document_element().expect("a document element");
        let defs_count = doc
            .descendants(root)
            .filter(|&id| doc.element(id).map(oxixml_dom::ElementData::local_name) == Some("defs"))
            .count();
        assert_eq!(
            defs_count, 1,
            "the gradient and open_defs must share one <defs>"
        );
        let defs = find_element(&doc, root, "defs");
        assert_eq!(doc.children(defs).count(), 2, "linearGradient + symbol");
    }

    #[test]
    fn pretty_and_compact_output_reparse_to_the_same_structure() {
        let mut svg = SvgBuilder::new(50.0, 50.0);
        svg.rect(0.0, 0.0, 10.0, 10.0);
        svg.circle(5.0, 5.0, 2.0);

        let compact_text = svg.to_string();
        let compact = build::parse_str(&compact_text).expect("compact reparse");
        assert_eq!(
            compact.to_xml(),
            compact_text,
            "reparsing the compact output and re-serializing it must be idempotent"
        );

        let mut pretty = build::parse_str(&svg.to_string_pretty()).expect("pretty reparse");
        // Pretty-printing inserts whitespace-only text nodes between
        // elements that the compact form never had; strip them so the
        // comparison is about element/attribute structure, not layout.
        let whitespace_only: Vec<NodeId> = pretty
            .descendants(pretty.root())
            .filter(|&id| {
                pretty.kind(id) == Some(NodeKind::Text)
                    && pretty
                        .character_data(id)
                        .is_some_and(|text| text.trim().is_empty())
            })
            .collect();
        for id in whitespace_only {
            let parent = pretty.parent(id).expect("a text node always has a parent");
            pretty
                .remove_child(parent, id)
                .expect("removing a known child cannot fail");
        }

        assert!(
            compact.structural_eq(&pretty),
            "{:?}",
            compact.structural_diff(&pretty)
        );
    }

    #[test]
    fn precision_control_rounds_attr_number_path_and_points() {
        let mut svg = SvgBuilder::plain();
        svg.set_precision(Some(2));
        assert_eq!(svg.precision(), Some(2));

        svg.open("rect");
        svg.attr_number("x", 1.0 / 3.0);
        svg.close();
        assert!(svg.to_string().contains("x=\"0.33\""));

        let path = PathBuilder::new().move_to(1.0 / 3.0, 0.0).finish();
        svg.path(&path);
        assert!(svg.to_string().contains("d=\"M 0.33 0\""));

        svg.polyline(&[[1.0 / 3.0, 2.0 / 3.0]]);
        assert!(svg.to_string().contains("points=\"0.33,0.67\""));
    }

    #[test]
    fn unbalanced_close_is_recorded_as_a_flaw_not_a_panic() {
        let mut svg = SvgBuilder::plain();
        svg.close();
        assert_eq!(svg.flaws().len(), 1);
        assert_eq!(svg.flaws()[0], BuildFlaw::UnbalancedClose);
        // The root itself must still be there and usable afterward.
        svg.attr_class("still-alive");
        assert_eq!(svg.flaws().len(), 1);
    }

    #[test]
    fn invalid_element_name_falls_back_to_a_group_and_is_recorded() {
        let mut svg = SvgBuilder::plain();
        svg.open("not a name");
        svg.attr_number("cx", 1.0);
        svg.close();
        assert_eq!(svg.flaws().len(), 1);
        assert!(matches!(
            &svg.flaws()[0],
            BuildFlaw::InvalidElementName { name } if name == "not a name"
        ));
        assert!(svg.to_string().contains("<g cx=\"1\""));
    }

    #[test]
    fn non_finite_attr_number_is_recorded_and_left_unset() {
        let mut svg = SvgBuilder::plain();
        svg.attr_number("opacity", f64::NAN);
        assert_eq!(svg.flaws().len(), 1);
        assert!(matches!(
            &svg.flaws()[0],
            BuildFlaw::NonFiniteNumber { name } if name == "opacity"
        ));
        assert!(!svg.to_string().contains("opacity"));
    }

    #[test]
    fn to_string_with_can_include_the_xml_declaration() {
        let svg = SvgBuilder::new(10.0, 10.0);
        let options = SerializeOptions::new().with_xml_declaration(XmlDeclaration::Include);
        let xml = svg.to_string_with(&options);
        assert!(xml.starts_with("<?xml version=\"1.0\"?>"));
        assert!(!svg.to_string().starts_with("<?xml"));
    }

    /// A properly balanced `open_defs`/`close` pair must fully return the
    /// insertion point to the document root, even the second time: reopening
    /// `<defs>` finds the same element rather than nesting a second one, and
    /// closing it pops back out cleanly.
    #[test]
    fn reopening_defs_after_closing_returns_later_content_to_the_root() {
        let mut svg = SvgBuilder::plain();
        svg.open_defs();
        svg.close();
        svg.open_defs();
        svg.close();
        svg.rect(0.0, 0.0, 1.0, 1.0);

        let reparsed = build::parse_str(&svg.to_string()).expect("builder output is well-formed");
        let root = reparsed.document_element().expect("a document element");
        let rect = find_element(&reparsed, root, "rect");
        let parent = reparsed.parent(rect).expect("rect has a parent");
        assert_eq!(
            reparsed
                .element(parent)
                .map(oxixml_dom::ElementData::local_name),
            Some("svg"),
            "rect must be a child of the root <svg>, not <defs>"
        );
    }

    /// [`SvgBuilder::open_defs`] pushes a new insertion-point stack frame on
    /// every call, exactly like [`SvgBuilder::open`]/[`SvgBuilder::group`]:
    /// calling it twice without an intervening [`SvgBuilder::close`] resolves
    /// to the *same* `<defs>` element both times (so no nested `<defs>` is
    /// ever created), but still requires two `close` calls to fully unwind.
    /// One `close` therefore leaves later content nested inside `<defs>`
    /// rather than back at the root — this is the ordinary stack-discipline
    /// contract every open/close pair in this module shares, not a defect
    /// specific to `open_defs`.
    #[test]
    fn open_defs_without_a_matching_close_still_requires_two_closes_to_unwind() {
        let mut svg = SvgBuilder::plain();
        svg.open_defs();
        svg.open_defs(); // no close() in between: one extra stack frame.
        svg.close(); // only unwinds the second open_defs().
        svg.rect(0.0, 0.0, 1.0, 1.0);

        let reparsed = build::parse_str(&svg.to_string()).expect("builder output is well-formed");
        let root = reparsed.document_element().expect("a document element");
        let rect = find_element(&reparsed, root, "rect");
        let parent = reparsed.parent(rect).expect("rect has a parent");
        assert_eq!(
            reparsed
                .element(parent)
                .map(oxixml_dom::ElementData::local_name),
            Some("defs"),
            "an unclosed open_defs() leaves later content nested inside <defs>"
        );
        let defs_count = reparsed
            .descendants(root)
            .filter(|&id| {
                reparsed
                    .element(id)
                    .map(oxixml_dom::ElementData::local_name)
                    == Some("defs")
            })
            .count();
        assert_eq!(defs_count, 1, "the two open_defs() calls share one <defs>");
    }
}
