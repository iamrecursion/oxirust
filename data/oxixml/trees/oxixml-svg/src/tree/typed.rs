//! [`SvgElement`]: a typed, read-only accessor over one SVG-namespace
//! element node in an [`oxixml_dom::Document`].
//!
//! This is the "typed accessor" pattern the crate builds its whole document
//! model on: `SvgElement` borrows the document rather than copying anything
//! out of it, so every accessor re-reads the underlying DOM node on each
//! call and always reflects the tree as it currently is. It is `Copy` (a
//! document reference plus a [`NodeId`]), so it can be passed around and
//! stored in short-lived collections freely.

use oxixml_dom::{Document, NodeId};

use crate::error::ValueError;
use crate::names::attrs::AttrId;
use crate::names::elements::ElementId;
use crate::ns::{SVG_NAMESPACE, XLINK_NAMESPACE, XML_NAMESPACE};
use crate::style::attr::{Declaration, parse_style_attribute};
use crate::style::css::split_classes;
use crate::style::property::PropertyId;
use crate::values::aspect::{PreserveAspectRatio, ViewBox};
use crate::values::clock::ClockDuration;
use crate::values::color::Color;
use crate::values::iri;
use crate::values::length::{self, Angle, Length};
use crate::values::lex;
use crate::values::paint::{self, Paint};
use crate::values::path::PathData;
use crate::values::points;
use crate::values::transform::TransformList;

/// A typed, read-only view of one SVG-namespace element node.
///
/// Every accessor reads straight through to the underlying
/// [`Document`]; nothing is cached or copied. Construct one through
/// [`crate::tree::document::SvgDocument::element`] or
/// [`crate::tree::document::SvgDocument::svg_element`], or with
/// [`SvgElement::from_node`] directly against a [`Document`] you already
/// hold.
#[derive(Debug, Clone, Copy)]
pub struct SvgElement<'a> {
    doc: &'a Document,
    node: NodeId,
}

impl<'a> SvgElement<'a> {
    /// Creates a typed view of `node` without checking that it is actually
    /// an SVG-namespace element.
    ///
    /// Used internally where that check was already performed (for
    /// example [`crate::tree::document::SvgDocument`]'s cached root, whose
    /// namespace was already verified while the document was being
    /// built). Prefer [`SvgElement::from_node`] when `node` has not
    /// already been checked.
    pub(crate) const fn new(doc: &'a Document, node: NodeId) -> Self {
        Self { doc, node }
    }

    /// Creates a typed view of `node`, or `None` if `node` does not exist,
    /// is not an element, or is not in the SVG namespace.
    #[must_use]
    pub fn from_node(doc: &'a Document, node: NodeId) -> Option<Self> {
        let data = doc.element(node)?;
        if data.namespace_uri() == Some(SVG_NAMESPACE) {
            Some(Self::new(doc, node))
        } else {
            None
        }
    }

    /// The node identifier this view wraps.
    #[must_use]
    pub const fn node(self) -> NodeId {
        self.node
    }

    /// The document this view borrows from.
    #[must_use]
    pub const fn document(self) -> &'a Document {
        self.doc
    }

    /// The element's recognized [`ElementId`], or `None` when the local
    /// name is not one this crate's SVG vocabulary catalog names (an
    /// unrecognized SVG-namespace element, e.g. a future or vendor tag —
    /// distinct from a foreign-namespace element, which can never produce
    /// an `SvgElement` at all).
    #[must_use]
    pub fn kind(self) -> Option<ElementId> {
        ElementId::from_name(self.local_name())
    }

    /// The element's local (unprefixed) name, e.g. `"rect"`.
    #[must_use]
    pub fn local_name(self) -> &'a str {
        self.doc
            .element(self.node)
            .expect("SvgElement always wraps an element node, checked at construction")
            .local_name()
    }

    /// The element's `id` attribute value, if it has one.
    #[must_use]
    pub fn id(self) -> Option<&'a str> {
        self.attr(AttrId::Id)
    }

    /// The element's `class` attribute, split into individual white-space
    /// separated tokens. Empty (yielding nothing) if the attribute is
    /// absent.
    pub fn class_list(self) -> impl Iterator<Item = &'a str> + 'a {
        split_classes(self.attr(AttrId::Class).unwrap_or(""))
    }

    /// The raw string value of `attr` on this element, resolving the
    /// correct namespace for the `xml:`/`xlink:` attribute families.
    ///
    /// [`AttrId::is_xlink`] variants (the legacy `xlink:href` family) are
    /// looked up in the [`XLINK_NAMESPACE`]; [`AttrId::XmlLang`],
    /// [`AttrId::XmlSpace`], and [`AttrId::XmlBase`] are looked up in the
    /// [`XML_NAMESPACE`]; every other attribute — including the SVG 2
    /// non-namespaced [`AttrId::Href`] and the SVG 2 HTML-aligned
    /// [`AttrId::Lang`] — is looked up unprefixed, in no namespace.
    #[must_use]
    pub fn attr(self, attr: AttrId) -> Option<&'a str> {
        let name = attr.as_str();
        if attr.is_xlink() {
            let local = name.strip_prefix("xlink:").unwrap_or(name);
            self.doc
                .get_attribute_ns(self.node, Some(XLINK_NAMESPACE), local)
        } else if matches!(attr, AttrId::XmlLang | AttrId::XmlSpace | AttrId::XmlBase) {
            let local = name.strip_prefix("xml:").unwrap_or(name);
            self.doc
                .get_attribute_ns(self.node, Some(XML_NAMESPACE), local)
        } else {
            self.doc.get_attribute(self.node, name)
        }
    }

    /// The raw string value of the attribute literally named `name`, with
    /// no namespace resolution — a plain passthrough to
    /// [`Document::get_attribute`]. Use [`SvgElement::attr`] for
    /// [`AttrId`]-based, namespace-aware lookups.
    #[must_use]
    pub fn attr_by_name(self, name: &str) -> Option<&'a str> {
        self.doc.get_attribute(self.node, name)
    }

    /// Parses `attr`'s raw value with `parse`, or `None` if the attribute
    /// is absent. Shared by every `*_attr`/typed-getter method below.
    fn parsed<T>(
        self,
        attr: AttrId,
        parse: fn(&str) -> Result<T, ValueError>,
    ) -> Option<Result<T, ValueError>> {
        self.attr(attr).map(parse)
    }

    /// The value of `attr`, parsed as a [`Length`]. `None` when the
    /// attribute is absent; `Some(Err(_))` when it is present but
    /// malformed.
    #[must_use]
    pub fn length_attr(self, attr: AttrId) -> Option<Result<Length, ValueError>> {
        self.parsed(attr, Length::parse)
    }

    /// The value of `attr`, parsed as a plain SVG `<number>`.
    #[must_use]
    pub fn number_attr(self, attr: AttrId) -> Option<Result<f64, ValueError>> {
        self.parsed(attr, lex::parse_number)
    }

    /// The value of `attr`, parsed as an [`Angle`].
    #[must_use]
    pub fn angle_attr(self, attr: AttrId) -> Option<Result<Angle, ValueError>> {
        self.parsed(attr, Angle::parse)
    }

    /// The value of `attr`, parsed as a [`Color`].
    #[must_use]
    pub fn color_attr(self, attr: AttrId) -> Option<Result<Color, ValueError>> {
        self.parsed(attr, Color::parse)
    }

    /// The value of `attr`, parsed as a [`Paint`].
    #[must_use]
    pub fn paint_attr(self, attr: AttrId) -> Option<Result<Paint, ValueError>> {
        self.parsed(attr, Paint::parse)
    }

    /// The value of `attr`, parsed as a [`TransformList`]. Works for any of
    /// [`AttrId::Transform`], [`AttrId::GradientTransform`], and
    /// [`AttrId::PatternTransform`], which all share the same grammar.
    #[must_use]
    pub fn transform_attr(self, attr: AttrId) -> Option<Result<TransformList, ValueError>> {
        self.parsed(attr, TransformList::parse)
    }

    /// The `d` attribute, parsed as [`PathData`].
    #[must_use]
    pub fn path_data(self) -> Option<Result<PathData, ValueError>> {
        self.parsed(AttrId::D, PathData::parse)
    }

    /// The `points` attribute (`polyline`/`polygon`), parsed strictly: the
    /// first grammar violation is an error, unlike the lenient
    /// partial-recovery [`points::parse_points`] performs for rendering
    /// purposes.
    #[must_use]
    pub fn points(self) -> Option<Result<Vec<[f64; 2]>, ValueError>> {
        self.parsed(AttrId::Points, points::parse_points_strict)
    }

    /// The `viewBox` attribute, parsed as a [`ViewBox`].
    #[must_use]
    pub fn view_box(self) -> Option<Result<ViewBox, ValueError>> {
        self.parsed(AttrId::ViewBox, ViewBox::parse)
    }

    /// The `preserveAspectRatio` attribute.
    #[must_use]
    pub fn preserve_aspect_ratio(self) -> Option<Result<PreserveAspectRatio, ValueError>> {
        self.parsed(AttrId::PreserveAspectRatio, PreserveAspectRatio::parse)
    }

    /// The `stroke-dasharray` attribute, parsed as a dash list.
    ///
    /// The `none` keyword (matched case-insensitively, like every other
    /// CSS-wide/property keyword this crate recognizes) resolves to an
    /// empty list rather than a parse error, matching its documented "no
    /// dashing" meaning; every other value goes through
    /// [`length::parse_dash_array`].
    #[must_use]
    pub fn dash_array(self) -> Option<Result<Vec<Length>, ValueError>> {
        self.attr(AttrId::StrokeDasharray).map(|raw| {
            if raw.trim_matches(lex::is_wsp).eq_ignore_ascii_case("none") {
                Ok(Vec::new())
            } else {
                length::parse_dash_array(raw)
            }
        })
    }

    /// The value of `attr`, parsed as an opacity (a `<number>` or
    /// `<percentage>`, clamped to `0.0..=1.0`).
    #[must_use]
    pub fn opacity_attr(self, attr: AttrId) -> Option<Result<f64, ValueError>> {
        self.parsed(attr, paint::parse_opacity)
    }

    /// The `dur` attribute: an animation element's active duration,
    /// `"indefinite"`, `"media"`, or a clock value.
    #[must_use]
    pub fn dur(self) -> Option<Result<ClockDuration, ValueError>> {
        self.parsed(AttrId::Dur, ClockDuration::parse)
    }

    /// The effective reference target: SVG 2's bare `href` when present,
    /// otherwise the legacy `xlink:href`, per [`iri::effective_href`].
    #[must_use]
    pub fn href(self) -> Option<&'a str> {
        iri::effective_href(self.attr(AttrId::Href), self.attr(AttrId::XlinkHref))
    }

    /// The nearest ancestor that is itself an SVG-namespace element.
    /// `None` at the document's `svg` root, or if this element has since
    /// been detached from the tree.
    #[must_use]
    pub fn parent_element(self) -> Option<Self> {
        let parent = self.doc.parent(self.node)?;
        Self::from_node(self.doc, parent)
    }

    /// This element's SVG-namespace element children, in document order.
    /// Non-element children (text, comments, processing instructions, …)
    /// and foreign-namespace element children are skipped, but do not
    /// interrupt the walk past them.
    pub fn children(self) -> impl Iterator<Item = Self> + 'a {
        let doc = self.doc;
        doc.children(self.node)
            .filter_map(move |child| Self::from_node(doc, child))
    }

    /// This element's SVG-namespace element descendants, in document order
    /// (an iterative walk, not a recursive one). Foreign-namespace elements
    /// are skipped, the same way [`SvgElement::children`] skips them.
    pub fn descendants(self) -> impl Iterator<Item = Self> + 'a {
        let doc = self.doc;
        doc.descendants(self.node)
            .filter_map(move |node| Self::from_node(doc, node))
    }

    /// The concatenated character data of every text/CDATA descendant, in
    /// document order — a passthrough to [`Document::text_content`].
    #[must_use]
    pub fn text_content(self) -> Option<String> {
        self.doc.text_content(self.node)
    }

    /// This element's `style` attribute, parsed into declarations. Empty
    /// if the attribute is absent, empty, or contains only malformed
    /// declarations.
    #[must_use]
    pub fn style_declarations(self) -> Vec<Declaration> {
        self.attr(AttrId::Style)
            .map(parse_style_attribute)
            .unwrap_or_default()
    }

    /// The raw string value of the presentation attribute matching
    /// `property` (that is, the XML attribute with the same name as the
    /// CSS property), or `None` when `property` does not allow a
    /// presentation attribute at all, or the attribute is absent.
    #[must_use]
    pub fn presentation_attr(self, property: PropertyId) -> Option<&'a str> {
        if property.presentation_attribute_allowed() {
            self.attr_by_name(property.as_str())
        } else {
            None
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use oxixml_dom::build;

    const FIXTURE: &str = r##"<svg xmlns="http://www.w3.org/2000/svg"
          xmlns:xlink="http://www.w3.org/1999/xlink" id="root" xml:lang="en">
  <g id="g1" class="a b c">
    <rect id="r1" class="shape big" x="1" y="2" width="10" height="20"
          fill="red" style="fill:blue; stroke:green !important"
          stroke-dasharray="1 2 3" transform="translate(1 2)"
          fill-opacity="150%"/>
    <rect id="r2" width="bad"/>
    <rect id="dashed-none" stroke-dasharray="NONE"/>
    <use id="u1" href="#r1" xlink:href="#legacy"/>
    <use id="u2" xlink:href="#legacy-only"/>
    <polyline id="poly-ok" points="0,0 1,1 2,2"/>
    <polyline id="poly-bad" points="0,0 1,1 2"/>
    <path id="path-ok" d="M0 0 L1 1"/>
    <path id="path-bad" d="M0 0 X1 1"/>
    <animate id="anim1" dur="4s"/>
    <text id="t1">Hello<tspan id="tsp"> world</tspan></text>
    <foreignObject id="fo1"><div xmlns="http://www.w3.org/1999/xhtml" id="foreign-div"/></foreignObject>
  </g>
  <style id="sheet">rect { fill: black; }</style>
</svg>"##;

    fn parse() -> Document {
        build::parse_str(FIXTURE).expect("fixture is well-formed XML")
    }

    fn root(doc: &Document) -> NodeId {
        doc.document_element()
            .expect("fixture has a document element")
    }

    /// Finds an element by its `id` attribute with a direct document-order
    /// scan, the same way `crate::tree::refs::IdIndex` does — the fixture
    /// declares no DTD/`xml:id`, so `Document::get_element_by_id` (which
    /// requires one of those) would not find anything here.
    fn by_id<'d>(doc: &'d Document, id: &str) -> SvgElement<'d> {
        let node = doc
            .traverse(root(doc))
            .find(|&n| doc.get_attribute(n, "id") == Some(id))
            .unwrap_or_else(|| panic!("no element with id {id:?}"));
        SvgElement::from_node(doc, node)
            .unwrap_or_else(|| panic!("element {id:?} is not an SVG-namespace element"))
    }

    #[test]
    fn from_node_accepts_svg_and_rejects_foreign_namespace_elements() {
        let doc = parse();
        let svg = root(&doc);
        assert!(SvgElement::from_node(&doc, svg).is_some());
        assert!(SvgElement::from_node(&doc, by_id(&doc, "g1").node()).is_some());

        let div = doc
            .get_elements_by_tag_name_ns(svg, Some("http://www.w3.org/1999/xhtml"), "div")
            .next()
            .expect("foreign div is present");
        assert!(SvgElement::from_node(&doc, div).is_none());
    }

    #[test]
    fn kind_local_name_and_id() {
        let doc = parse();
        let svg = SvgElement::from_node(&doc, root(&doc)).expect("svg root");
        assert_eq!(svg.kind(), Some(ElementId::Svg));
        assert_eq!(svg.local_name(), "svg");
        assert_eq!(svg.id(), Some("root"));

        let r1 = by_id(&doc, "r1");
        assert_eq!(r1.kind(), Some(ElementId::Rect));
        assert_eq!(r1.id(), Some("r1"));
    }

    #[test]
    fn class_list_splits_on_white_space() {
        let doc = parse();
        let g1 = by_id(&doc, "g1");
        let classes: Vec<&str> = g1.class_list().collect();
        assert_eq!(classes, vec!["a", "b", "c"]);

        let svg = SvgElement::from_node(&doc, root(&doc)).expect("svg root");
        assert_eq!(svg.class_list().count(), 0, "no class attribute at all");
    }

    #[test]
    fn attr_resolves_xml_and_xlink_namespaces() {
        let doc = parse();
        let svg = SvgElement::from_node(&doc, root(&doc)).expect("svg root");
        assert_eq!(svg.attr(AttrId::XmlLang), Some("en"));

        let u1 = by_id(&doc, "u1");
        assert_eq!(u1.attr(AttrId::Href), Some("#r1"));
        assert_eq!(u1.attr(AttrId::XlinkHref), Some("#legacy"));
    }

    #[test]
    fn attr_by_name_is_a_plain_passthrough() {
        let doc = parse();
        let r1 = by_id(&doc, "r1");
        assert_eq!(r1.attr_by_name("fill"), Some("red"));
        assert_eq!(r1.attr_by_name("nope"), None);
    }

    #[test]
    fn length_attr_covers_absent_present_and_malformed() {
        let doc = parse();
        let r1 = by_id(&doc, "r1");
        assert_eq!(
            r1.length_attr(AttrId::X),
            Some(Ok(Length::new(
                1.0,
                crate::values::length::LengthUnit::None
            )))
        );
        assert_eq!(r1.length_attr(AttrId::Rx), None, "rx is absent");

        let r2 = by_id(&doc, "r2");
        let error = r2
            .length_attr(AttrId::Width)
            .expect("width is present")
            .expect_err("\"bad\" is not a valid length");
        assert!(matches!(error, ValueError::InvalidNumber { .. }));
    }

    #[test]
    fn number_angle_color_and_paint_attr() {
        let doc = parse();
        let r1 = by_id(&doc, "r1");
        assert_eq!(
            r1.number_attr(AttrId::X),
            Some(Ok(1.0)),
            "x is a plain number too"
        );
        assert_eq!(r1.number_attr(AttrId::Rx), None);

        // A bare number is a grammar-valid <angle> (interpreted as degrees),
        // so the same "y" attribute doubles as an angle-attribute fixture.
        assert_eq!(
            r1.angle_attr(AttrId::Y),
            Some(Ok(Angle::new(2.0, crate::values::length::AngleUnit::None)))
        );

        let red = Color::rgb(0xFF, 0x00, 0x00);
        assert_eq!(r1.color_attr(AttrId::Fill), Some(Ok(red)));
        assert_eq!(r1.paint_attr(AttrId::Fill), Some(Ok(Paint::Color(red))));
        assert_eq!(
            r1.paint_attr(AttrId::Stroke),
            None,
            "stroke is only in style=, not an attribute"
        );
    }

    #[test]
    fn transform_attr_parses_the_shared_transform_grammar() {
        let doc = parse();
        let r1 = by_id(&doc, "r1");
        let parsed = r1
            .transform_attr(AttrId::Transform)
            .expect("transform is present")
            .expect("valid transform list");
        assert_eq!(parsed.0.len(), 1);
        assert_eq!(r1.transform_attr(AttrId::GradientTransform), None);
    }

    #[test]
    fn path_data_absent_present_and_malformed() {
        let doc = parse();
        let ok = by_id(&doc, "path-ok");
        let path = ok
            .path_data()
            .expect("d is present")
            .expect("valid path data");
        assert_eq!(path.len(), 2);

        let bad = by_id(&doc, "path-bad");
        let error = bad
            .path_data()
            .expect("d is present")
            .expect_err("X is not a path command");
        assert!(error.offset().is_some());

        let r1 = by_id(&doc, "r1");
        assert_eq!(r1.path_data(), None, "rect has no d attribute");
    }

    #[test]
    fn points_is_strict_unlike_the_lenient_recovery_parser() {
        let doc = parse();
        let ok = by_id(&doc, "poly-ok");
        assert_eq!(
            ok.points(),
            Some(Ok(vec![[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]]))
        );

        let bad = by_id(&doc, "poly-bad");
        assert!(bad.points().expect("points is present").is_err());
    }

    #[test]
    fn dash_array_handles_absent_none_and_a_real_list() {
        let doc = parse();
        let r1 = by_id(&doc, "r1");
        let dashes = r1
            .dash_array()
            .expect("stroke-dasharray is present")
            .expect("valid dash array");
        assert_eq!(dashes.len(), 3);

        let none = by_id(&doc, "dashed-none");
        assert_eq!(
            none.dash_array(),
            Some(Ok(Vec::new())),
            "the none keyword (any case) means an empty dash list"
        );

        let r2 = by_id(&doc, "r2");
        assert_eq!(
            r2.dash_array(),
            None,
            "no stroke-dasharray attribute at all"
        );
    }

    #[test]
    fn opacity_attr_and_dur() {
        let doc = parse();
        let r1 = by_id(&doc, "r1");
        assert_eq!(
            r1.opacity_attr(AttrId::FillOpacity),
            Some(Ok(1.0)),
            "150% clamps to 1.0"
        );
        assert_eq!(r1.opacity_attr(AttrId::StrokeOpacity), None);

        let anim = by_id(&doc, "anim1");
        assert_eq!(
            anim.dur(),
            Some(Ok(ClockDuration::parse("4s").expect("4s is valid")))
        );
        assert_eq!(r1.dur(), None, "rect has no dur attribute");
    }

    #[test]
    fn href_prefers_the_bare_svg2_attribute_over_xlink() {
        let doc = parse();
        let u1 = by_id(&doc, "u1");
        assert_eq!(u1.href(), Some("#r1"));

        let u2 = by_id(&doc, "u2");
        assert_eq!(u2.href(), Some("#legacy-only"), "falls back to xlink:href");
    }

    #[test]
    fn tree_navigation_skips_foreign_namespace_content() {
        let doc = parse();
        let svg = SvgElement::from_node(&doc, root(&doc)).expect("svg root");
        let top_children: Vec<&str> = svg.children().map(SvgElement::local_name).collect();
        assert_eq!(top_children, vec!["g", "style"]);

        let g1 = by_id(&doc, "g1");
        assert_eq!(g1.parent_element().map(SvgElement::local_name), Some("svg"));
        assert!(
            svg.parent_element().is_none(),
            "svg is the document element"
        );

        let fo1 = by_id(&doc, "fo1");
        assert_eq!(
            fo1.children().count(),
            0,
            "the xhtml div child is a foreign element"
        );

        let descendant_names: Vec<&str> = svg.descendants().map(SvgElement::local_name).collect();
        assert!(descendant_names.contains(&"rect"));
        assert!(
            !descendant_names.contains(&"div"),
            "foreign-namespace descendants are skipped"
        );
    }

    #[test]
    fn text_content_concatenates_descendant_text() {
        let doc = parse();
        let t1 = by_id(&doc, "t1");
        assert_eq!(t1.text_content().as_deref(), Some("Hello world"));
    }

    #[test]
    fn style_declarations_and_presentation_attr() {
        let doc = parse();
        let r1 = by_id(&doc, "r1");
        let declarations = r1.style_declarations();
        assert_eq!(declarations.len(), 2);
        assert_eq!(&*declarations[0].name, "fill");
        assert!(declarations[1].important);

        assert_eq!(r1.presentation_attr(PropertyId::Fill), Some("red"));
        assert_eq!(
            r1.presentation_attr(PropertyId::Stroke),
            None,
            "stroke is only set via style=, not as a presentation attribute"
        );

        let r2 = by_id(&doc, "r2");
        assert_eq!(r2.style_declarations(), Vec::new());
    }
}
