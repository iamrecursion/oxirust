//! `oxixml-svg` — a typed SVG 1.1/2.0 document model, microsyntax parser,
//! geometry toolkit, programmatic builder, and linter over an `oxixml-dom`
//! tree.
//!
//! Part of the [OxiXML](https://github.com/cool-japan/oxixml) workspace — the
//! COOLJAPAN Pure-Rust foundation for all XML and RDF processing. SVG is the
//! one graphics format the rest of the ecosystem cannot avoid: dataframe
//! plotting (pandrs), CAD export (oxicad), and document/media pipelines
//! (fop, oxitext, oximedia) all need to emit or consume it, and each has so
//! far done so with its own hand-rolled `format!("<svg ...")` emitter —
//! unescaped, untyped, and unable to answer even a simple "what is this
//! path's bounding box" question. This crate is the shared foundation: parse
//! real-world SVG into an ordinary [`oxixml_dom::Document`], read it back
//! out through typed accessors, compute exact geometry, build new documents
//! through an escaping-safe API, and lint the result — all without ever
//! rasterizing a single pixel.
//!
//! # What is here
//!
//! * **[`values`]** — one parser/serializer pair per SVG attribute
//!   microsyntax: [`Length`] and [`crate::values::length::Angle`], [`Color`],
//!   [`Paint`], [`TransformList`], [`ViewBox`]/[`PreserveAspectRatio`],
//!   [`PathData`]/[`PathSegment`], [`crate::values::points`] (polyline/polygon
//!   coordinates), [`crate::values::clock`] (SMIL timing values), and
//!   [`crate::values::iri::FuncIri`] (`url(...)`).
//! * **[`names`]** — the closed vocabulary catalog: every element as
//!   [`crate::names::elements::ElementId`] (88 names) and every attribute as
//!   [`crate::names::attrs::AttrId`] (216 names), plus the
//!   [`crate::names::content`] tables that say which children and which
//!   attributes each element admits.
//! * **[`style`]** — the presentation-property catalog
//!   ([`crate::style::property::PropertyId`]), the `style=""` attribute
//!   parser ([`crate::style::attr::Declaration`]), and a miniature `<style>`
//!   stylesheet engine ([`crate::style::css::Stylesheet`]) covering
//!   type/class/id selectors, grouping, and the descendant combinator.
//! * **[`geometry`]** — the pure-math layer everything else builds on:
//!   [`Matrix`], [`Point`]/[`Rect`], cubic/quadratic Bézier evaluation
//!   ([`crate::geometry::bezier`]), elliptical-arc conversion
//!   ([`crate::geometry::arc`]), whole-path operations
//!   ([`crate::geometry::path_ops`]: bounding box, arc length, flattening),
//!   and the `viewBox`-to-viewport algorithm
//!   ([`crate::geometry::viewport::view_box_transform`]).
//! * **[`tree`]** — the typed document model over an [`oxixml_dom::Document`]:
//!   [`SvgDocument`] and [`SvgElement`] for reading,
//!   [`crate::tree::refs::IdIndex`] and cycle-checked reference resolution,
//!   [`crate::tree::cascade::StyleResolver`] for computed style, and
//!   [`crate::tree::shapes::equivalent_path`] for turning a basic shape into
//!   the path it is equivalent to.
//! * **[`writer`]** — [`SvgBuilder`], a chainable, escaping-safe builder that
//!   replaces hand-rolled string formatting, with shape convenience methods,
//!   [`PathBuilder`], and gradient/pattern/clip-path/mask/marker
//!   construction ([`crate::writer::paint_defs`]).
//! * **[`lint`]** — [`Linter`], which validates a plain
//!   [`oxixml_dom::Document`] against unknown names, content-model
//!   violations, dangling/circular references, deprecated constructs, and
//!   every microsyntax this crate implements, producing [`Diagnostic`]s at
//!   three [`Severity`] levels.
//! * **[`error`]** / **[`ns`]** — the two error families ([`ValueError`] for
//!   one microsyntax value, [`SvgError`] for a whole document) and the
//!   namespace constants ([`SVG_NAMESPACE`], [`XLINK_NAMESPACE`]) every
//!   module above builds on.
//!
//! # Example
//!
//! ```
//! use oxixml_svg::{Length, LengthUnit, Linter, Rect, SvgBuilder, SvgDocument};
//!
//! // Parse a document and read a typed presentation attribute.
//! let doc = SvgDocument::parse_str(
//!     r#"<svg xmlns="http://www.w3.org/2000/svg" width="200" height="100">
//!          <rect x="10" y="20" width="50" height="30"/>
//!        </svg>"#,
//! )
//! .expect("well-formed SVG");
//! assert_eq!(doc.width(), Some(Ok(Length::new(200.0, LengthUnit::None))));
//!
//! // Compute the rect's bounding box through the geometry engine: turn it
//! // into its equivalent path, then measure that path.
//! let rect = doc.svg_element().children().next().expect("the rect element");
//! let ctx = oxixml_svg::values::length::LengthContext::default();
//! let path = oxixml_svg::tree::shapes::equivalent_path(rect, &ctx)
//!     .expect("rect is a recognized shape")
//!     .expect("its geometry attributes are all valid");
//! let bbox =
//!     oxixml_svg::geometry::path_ops::bounding_box(&path).expect("a non-empty rect has a bbox");
//! assert_eq!(bbox, Rect::new(10.0, 20.0, 50.0, 30.0).expect("a valid rect"));
//!
//! // Lint the document: nothing here is unknown, deprecated, or dangling.
//! let diagnostics = Linter::new().lint(doc.document());
//! assert!(diagnostics.is_empty(), "{diagnostics:?}");
//!
//! // Build a small document and check it serializes and re-parses cleanly.
//! let mut built = SvgBuilder::new(20.0, 20.0);
//! built.rect(0.0, 0.0, 10.0, 10.0);
//! let xml = built.to_string();
//! let reparsed = SvgDocument::parse_str(&xml).expect("builder output is well-formed SVG");
//! assert_eq!(reparsed.to_xml(), xml, "serialization is a fixed point");
//! ```
//!
//! # Design
//!
//! This crate is built entirely on [`oxixml_dom::Document`]: parsing
//! produces an ordinary DOM tree with a cached SVG root, not a bespoke tree
//! type, so every DOM operation (namespace-aware traversal, mutation,
//! serialization) keeps working on the document an [`SvgDocument`] wraps.
//! There is deliberately **no rasterization** anywhere in this crate —
//! turning geometry and paint into pixels is out of scope for a foundation
//! crate and belongs to a downstream renderer.
//!
//! Coverage targets the union of SVG 1.1 (Second Edition) and SVG 2, with a
//! small number of documented cuts: modern CSS color spaces (`hwb()`,
//! `lab()`, `lch()`, `oklab()`, `oklch()`, `color()`) are rejected rather
//! than parsed (see [`crate::values::color`]); the `<style>` selector engine
//! supports type/class/id/universal selectors, grouping, and the descendant
//! combinator but not pseudo-classes, attribute selectors, or the
//! child/sibling combinators (see [`crate::style::css`]); and an animation
//! element's `keyTimes`/`keySplines`/`values` are not cross-checked for
//! matching item counts, since that needs the whole set of an animation
//! element's attributes together rather than a per-attribute grammar (see
//! [`crate::lint::rules`]). Every cut is documented at its point of use
//! rather than silently guessed past. Measured conformance against the W3C
//! SVG 1.1 test suite is in this crate's README and `tests/w3c_svg.rs`.

#![deny(missing_docs)]
#![forbid(unsafe_code)]

pub mod error;
pub mod geometry;
pub mod lint;
pub mod names;
pub mod ns;
pub mod style;
pub mod tree;
pub mod values;
pub mod writer;

// Curated re-exports of the flagship types, so common use needs only
// `use oxixml_svg::Whatever;`. Deliberately not a glob re-export — see the
// module documentation above for the full surface, and each module's own
// `pub use` lines (e.g. `oxixml_svg::values::ClockDuration`) for ergonomic
// module-level paths to everything not re-exported this far.
pub use error::{SvgError, ValueError};
pub use geometry::matrix::Matrix;
pub use geometry::primitives::{Point, Rect};
pub use lint::diagnostic::{Diagnostic, Severity};
pub use lint::rules::Linter;
pub use ns::{SVG_NAMESPACE, XLINK_NAMESPACE};
pub use tree::document::{SvgDocument, SvgOptions};
pub use tree::typed::SvgElement;
pub use values::aspect::{PreserveAspectRatio, ViewBox};
pub use values::color::Color;
pub use values::length::{Length, LengthUnit};
pub use values::paint::Paint;
pub use values::path::{PathData, PathSegment};
pub use values::transform::TransformList;
pub use writer::builder::SvgBuilder;
pub use writer::pathbuilder::PathBuilder;
