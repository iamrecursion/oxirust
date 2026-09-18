//! [`SvgDocument`]: parsing an input into a typed SVG document.

use std::io::BufRead;

use oxixml_dom::build::{self, BuilderOptions};
use oxixml_dom::{Document, NodeId, SerializeOptions};

use crate::error::{SvgError, ValueError};
use crate::names::attrs::AttrId;
use crate::ns::SVG_NAMESPACE;
use crate::tree::typed::SvgElement;
use crate::values::aspect::{PreserveAspectRatio, ViewBox};
use crate::values::length::Length;

/// Builder-style options controlling how an input is parsed and validated
/// into an [`SvgDocument`].
///
/// ```
/// use oxixml_svg::tree::document::SvgOptions;
///
/// let options = SvgOptions::new().with_max_element_depth(64);
/// assert_eq!(options.builder_options().max_element_depth(), 64);
/// ```
#[derive(Debug, Clone, Default, PartialEq)]
pub struct SvgOptions {
    allow_any_root: bool,
    builder: BuilderOptions,
}

impl SvgOptions {
    /// Returns the default options: the document element must be an
    /// `{http://www.w3.org/2000/svg}svg` element, and the underlying XML
    /// reader/tree builder uses [`BuilderOptions::default`].
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// When `true`, skip the "document element must be `svg`" check
    /// entirely and adopt whatever the document element is as the
    /// [`SvgDocument`]'s root. Default `false`.
    ///
    /// A document with no document element at all is still rejected either
    /// way — this only widens *which* document element is accepted, not
    /// whether one is required.
    #[must_use]
    pub const fn with_allow_any_root(mut self, allow: bool) -> Self {
        self.allow_any_root = allow;
        self
    }

    /// Sets the maximum element nesting depth the underlying XML reader
    /// accepts. See [`BuilderOptions::with_max_element_depth`].
    #[must_use]
    pub fn with_max_element_depth(mut self, depth: usize) -> Self {
        self.builder = self.builder.with_max_element_depth(depth);
        self
    }

    /// Replaces the DOM builder options wholesale, for callers that need to
    /// control more of the underlying parse (entity policy, comment/PI
    /// retention, …) than [`Self::with_max_element_depth`] exposes on its
    /// own.
    #[must_use]
    pub fn with_builder_options(mut self, options: BuilderOptions) -> Self {
        self.builder = options;
        self
    }

    /// Reports whether the "document element must be `svg`" check is
    /// skipped.
    #[must_use]
    pub const fn allow_any_root(&self) -> bool {
        self.allow_any_root
    }

    /// The DOM builder options an input will be parsed with.
    #[must_use]
    pub const fn builder_options(&self) -> &BuilderOptions {
        &self.builder
    }
}

/// A parsed SVG document: an [`oxixml_dom::Document`] together with a
/// cached [`NodeId`] for its `{http://www.w3.org/2000/svg}svg` document
/// element.
///
/// ```
/// use oxixml_svg::tree::document::SvgDocument;
///
/// let doc = SvgDocument::parse_str(
///     r#"<svg xmlns="http://www.w3.org/2000/svg" width="10" height="20"/>"#,
/// )
/// .expect("valid SVG document");
/// assert_eq!(doc.svg_element().local_name(), "svg");
/// assert_eq!(doc.width().expect("present").expect("valid").value, 10.0);
/// ```
#[derive(Debug, Clone)]
pub struct SvgDocument {
    document: Document,
    root: NodeId,
}

impl SvgDocument {
    /// Parses `text` as XML and adopts it as an SVG document, using
    /// [`SvgOptions::default`].
    ///
    /// # Errors
    ///
    /// Returns [`SvgError::Xml`] if `text` is not well-formed XML, and
    /// [`SvgError::NotSvg`] if the document element is not an `svg`
    /// element in the SVG namespace.
    pub fn parse_str(text: &str) -> Result<Self, SvgError> {
        Self::parse_str_with(text, &SvgOptions::default())
    }

    /// Parses `text` as XML using explicit `options`.
    ///
    /// # Errors
    ///
    /// See [`SvgDocument::parse_str`].
    pub fn parse_str_with(text: &str, options: &SvgOptions) -> Result<Self, SvgError> {
        let document = build::parse_str_with(text, &options.builder)?;
        Self::from_document_with(document, options)
    }

    /// Parses a document from a buffered UTF-8 reader, using
    /// [`SvgOptions::default`].
    ///
    /// # Errors
    ///
    /// See [`SvgDocument::parse_str`]; I/O failures surface as
    /// [`SvgError::Xml`].
    pub fn parse_reader<R: BufRead>(reader: R) -> Result<Self, SvgError> {
        Self::parse_reader_with(reader, &SvgOptions::default())
    }

    /// Parses a document from a buffered UTF-8 reader using explicit
    /// `options`.
    ///
    /// # Errors
    ///
    /// See [`SvgDocument::parse_reader`].
    pub fn parse_reader_with<R: BufRead>(
        reader: R,
        options: &SvgOptions,
    ) -> Result<Self, SvgError> {
        let document = build::parse_reader_with(reader, &options.builder)?;
        Self::from_document_with(document, options)
    }

    /// Adopts an already-built [`Document`] as an SVG document, applying
    /// the same document-element check [`SvgDocument::parse_str`] does,
    /// using [`SvgOptions::default`].
    ///
    /// # Errors
    ///
    /// Returns [`SvgError::NotSvg`] if the document element is not an
    /// `svg` element in the SVG namespace.
    pub fn from_document(document: Document) -> Result<Self, SvgError> {
        Self::from_document_with(document, &SvgOptions::default())
    }

    /// Adopts an already-built [`Document`] as an SVG document using
    /// explicit `options`.
    ///
    /// # Errors
    ///
    /// See [`SvgDocument::from_document`].
    pub fn from_document_with(document: Document, options: &SvgOptions) -> Result<Self, SvgError> {
        let root = match document.document_element() {
            Some(node) if options.allow_any_root || is_svg_root(&document, node) => node,
            Some(node) => {
                return Err(SvgError::NotSvg {
                    found: Some(qualified_name_of(&document, node).into()),
                });
            }
            None => return Err(SvgError::NotSvg { found: None }),
        };
        Ok(Self { document, root })
    }
}

/// Reports whether `node` is `{http://www.w3.org/2000/svg}svg`.
fn is_svg_root(document: &Document, node: NodeId) -> bool {
    document.element(node).is_some_and(|data| {
        data.local_name() == "svg" && data.namespace_uri() == Some(SVG_NAMESPACE)
    })
}

/// The qualified name of an element node already known to exist, for the
/// [`SvgError::NotSvg`] diagnostic.
fn qualified_name_of(document: &Document, node: NodeId) -> String {
    document
        .element(node)
        .expect("document_element() always returns an element node")
        .qualified_name()
}

impl SvgDocument {
    /// The underlying DOM document.
    #[must_use]
    pub const fn document(&self) -> &Document {
        &self.document
    }

    /// A mutable reference to the underlying DOM document.
    ///
    /// Mutating the tree through this reference does not update the cached
    /// [`SvgDocument::root`] — removing or renaming the document element
    /// this way can leave the cached root dangling or no longer
    /// SVG-namespaced; re-validate with [`SvgDocument::from_document`] (via
    /// [`SvgDocument::into_document`]) if that matters for your use case.
    #[must_use]
    pub fn document_mut(&mut self) -> &mut Document {
        &mut self.document
    }

    /// Consumes this value, returning the underlying DOM document.
    #[must_use]
    pub fn into_document(self) -> Document {
        self.document
    }

    /// The node identifier of the `svg` document element.
    #[must_use]
    pub const fn root(&self) -> NodeId {
        self.root
    }

    /// A typed view of the `svg` document element.
    #[must_use]
    pub fn svg_element(&self) -> SvgElement<'_> {
        SvgElement::new(&self.document, self.root)
    }

    /// A typed view of `node`, or `None` if `node` does not exist, is not
    /// an element, or is not in the SVG namespace.
    #[must_use]
    pub fn element(&self, node: NodeId) -> Option<SvgElement<'_>> {
        SvgElement::from_node(&self.document, node)
    }

    /// The `svg` root's `width` attribute, parsed as a [`Length`]. `None`
    /// if the attribute is absent; `Some(Err(_))` if present but malformed.
    #[must_use]
    pub fn width(&self) -> Option<Result<Length, ValueError>> {
        self.svg_element().length_attr(AttrId::Width)
    }

    /// The `svg` root's `height` attribute, parsed as a [`Length`].
    #[must_use]
    pub fn height(&self) -> Option<Result<Length, ValueError>> {
        self.svg_element().length_attr(AttrId::Height)
    }

    /// The `svg` root's `viewBox` attribute, parsed as a [`ViewBox`].
    #[must_use]
    pub fn view_box(&self) -> Option<Result<ViewBox, ValueError>> {
        self.svg_element().view_box()
    }

    /// The `svg` root's `preserveAspectRatio` attribute. `None` if absent
    /// — callers apply the `xMidYMid meet` default themselves, since that
    /// default is contextual (see the specification catalog's per-element
    /// defaults table) rather than universal.
    #[must_use]
    pub fn preserve_aspect_ratio(&self) -> Option<Result<PreserveAspectRatio, ValueError>> {
        self.svg_element().preserve_aspect_ratio()
    }

    /// Serializes the document back to XML text, using default
    /// [`SerializeOptions`].
    #[must_use]
    pub fn to_xml(&self) -> String {
        self.document.to_xml()
    }

    /// Serializes the document back to XML text using explicit `options`.
    #[must_use]
    pub fn to_xml_with(&self, options: &SerializeOptions) -> String {
        self.document.to_xml_with(options)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_str_accepts_a_plain_svg_root() {
        let doc = SvgDocument::parse_str(
            r##"<svg xmlns="http://www.w3.org/2000/svg" id="root" width="10" height="20"/>"##,
        )
        .expect("valid SVG document");
        assert_eq!(doc.svg_element().local_name(), "svg");
        assert_eq!(doc.svg_element().id(), Some("root"));
        assert_eq!(doc.root(), doc.document().document_element().expect("root"));
    }

    #[test]
    fn parse_str_rejects_malformed_xml() {
        let error =
            SvgDocument::parse_str("<svg>").expect_err("unclosed element is not well-formed");
        assert!(matches!(error, SvgError::Xml(_)));
    }

    #[test]
    fn parse_str_rejects_a_non_svg_root_by_default() {
        let error = SvgDocument::parse_str("<catalog/>").expect_err("catalog is not an svg root");
        match error {
            SvgError::NotSvg { found: Some(name) } => assert_eq!(&*name, "catalog"),
            other => panic!("expected NotSvg {{ found: Some(\"catalog\") }}, got {other:?}"),
        }
    }

    #[test]
    fn parse_str_rejects_an_svg_namespaced_non_svg_element() {
        // The document element exists and is well-formed, but it is not
        // named "svg" — still an error even though *some* content in the
        // SVG namespace is present.
        let error = SvgDocument::parse_str(r##"<rect xmlns="http://www.w3.org/2000/svg"/>"##)
            .expect_err("rect is not svg");
        assert!(matches!(error, SvgError::NotSvg { found: Some(_) }));
    }

    #[test]
    fn from_document_reports_no_document_element_when_there_is_none() {
        // `oxixml_dom::build::*` itself never produces a `Document` with no
        // document element (it fails outright with
        // `BuildError::NoDocumentElement` first), so the only way to reach
        // this branch is to adopt a `Document` built some other way — here,
        // a freshly created, still-empty one.
        let error = SvgDocument::from_document(Document::new())
            .expect_err("an empty document has no document element");
        assert!(matches!(error, SvgError::NotSvg { found: None }));
    }

    #[test]
    fn allow_any_root_skips_the_svg_namespace_check() {
        let options = SvgOptions::new().with_allow_any_root(true);
        let doc = SvgDocument::parse_str_with("<catalog/>", &options)
            .expect("allow_any_root accepts a non-svg root");
        assert_eq!(doc.svg_element().local_name(), "catalog");
    }

    #[test]
    fn allow_any_root_still_requires_a_document_element() {
        let options = SvgOptions::new().with_allow_any_root(true);
        let error = SvgDocument::from_document_with(Document::new(), &options)
            .expect_err("no document element at all, even with allow_any_root");
        assert!(matches!(error, SvgError::NotSvg { found: None }));
    }

    #[test]
    fn from_document_adopts_an_existing_document() {
        let raw =
            oxixml_dom::build::parse_str(r##"<svg xmlns="http://www.w3.org/2000/svg" id="a"/>"##)
                .expect("valid xml");
        let doc = SvgDocument::from_document(raw).expect("svg root");
        assert_eq!(doc.svg_element().id(), Some("a"));
    }

    #[test]
    fn with_max_element_depth_is_enforced_by_the_underlying_reader() {
        let options = SvgOptions::new().with_max_element_depth(1);
        let deeply_nested = r##"<svg xmlns="http://www.w3.org/2000/svg"><g><g/></g></svg>"##;
        let error = SvgDocument::parse_str_with(deeply_nested, &options)
            .expect_err("nesting exceeds the configured depth of 1");
        assert!(matches!(error, SvgError::Xml(_)));
        assert_eq!(options.builder_options().max_element_depth(), 1);
    }

    #[test]
    fn width_height_view_box_and_preserve_aspect_ratio() {
        let doc = SvgDocument::parse_str(
            r##"<svg xmlns="http://www.w3.org/2000/svg" width="10" height="bad"
                      viewBox="0 0 100 200" preserveAspectRatio="xMinYMax slice"/>"##,
        )
        .expect("valid xml");
        assert_eq!(doc.width().expect("present").expect("valid").value, 10.0);
        assert!(
            doc.height().expect("present").is_err(),
            "\"bad\" is malformed"
        );

        let view_box = doc.view_box().expect("present").expect("valid");
        assert_eq!((view_box.min_x, view_box.min_y), (0.0, 0.0));
        assert_eq!((view_box.width, view_box.height), (100.0, 200.0));

        assert!(doc.preserve_aspect_ratio().expect("present").is_ok());

        let bare = SvgDocument::parse_str(r##"<svg xmlns="http://www.w3.org/2000/svg"/>"##)
            .expect("valid xml");
        assert_eq!(bare.width(), None);
        assert_eq!(bare.view_box(), None);
        assert_eq!(bare.preserve_aspect_ratio(), None);
    }

    #[test]
    fn to_xml_round_trips_through_document() {
        let doc =
            SvgDocument::parse_str(r##"<svg xmlns="http://www.w3.org/2000/svg" id="root"/>"##)
                .expect("valid xml");
        let text = doc.to_xml();
        let reparsed = SvgDocument::parse_str(&text).expect("re-parses cleanly");
        assert_eq!(reparsed.svg_element().id(), Some("root"));
    }

    #[test]
    fn element_rejects_foreign_namespace_nodes() {
        let doc = SvgDocument::parse_str(
            r##"<svg xmlns="http://www.w3.org/2000/svg"><foreignObject>
                    <div xmlns="http://www.w3.org/1999/xhtml" id="d"/>
                 </foreignObject></svg>"##,
        )
        .expect("valid xml");
        assert!(doc.element(doc.root()).is_some());

        let div = doc
            .document()
            .get_elements_by_tag_name_ns(doc.root(), Some("http://www.w3.org/1999/xhtml"), "div")
            .next()
            .expect("foreign div present");
        assert!(doc.element(div).is_none());
    }

    #[test]
    fn document_mut_and_into_document_expose_the_underlying_dom() {
        let mut doc =
            SvgDocument::parse_str(r##"<svg xmlns="http://www.w3.org/2000/svg" id="root"/>"##)
                .expect("valid xml");
        let root = doc.root();
        doc.document_mut()
            .set_attribute(root, "data-touched", "yes")
            .expect("set_attribute on a valid element succeeds");
        assert_eq!(
            doc.document().get_attribute(root, "data-touched"),
            Some("yes")
        );
        let plain = doc.into_document();
        assert_eq!(plain.get_attribute(root, "data-touched"), Some("yes"));
    }
}
