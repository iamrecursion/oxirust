//! `oxixml-fo` — the XSL-FO 1.1 formatting-object tree and property model.
//!
//! Part of the [OxiXML](https://github.com/cool-japan/oxixml) workspace — the
//! COOLJAPAN Pure-Rust foundation for all XML and RDF processing.
//!
//! This crate is the **front half of an XSL-FO formatter**, the part Apache FOP
//! calls the *FO tree*: it reads a stylesheet result in the
//! [XSL-FO namespace](FO_NAMESPACE), builds a tree of typed formatting objects,
//! validates it against the content models of
//! [XSL 1.1 §6](https://www.w3.org/TR/xsl11/#fo-section), and *refines* every
//! property of every object according to
//! [§5](https://www.w3.org/TR/xsl11/#prrefine) — shorthand expansion,
//! inheritance, initial values, expression evaluation, and the
//! writing-mode-dependent correspondence between the relative and absolute
//! spellings of a property.
//!
//! # The rendering boundary
//!
//! **Pagination, area-tree construction, space resolution, float and footnote
//! placement, font metrics, and PDF (or any other) output are out of scope.**
//! They belong to the downstream renderer. What this crate hands over is a
//! fully refined [`FoTree`] plus the shared vocabulary in [`area`]: the area
//! data types of XSL 1.1 §4 and the line-breaking primitives of
//! [`area::line`]. Nothing in [`area`] reads an [`FoTree`]; a renderer drives
//! those types itself.
//!
//! Refinement also stops short where the specification says it must: a
//! percentage whose reference value is the geometry of an area survives into
//! the computed value as [`property::Value::Percent`] unless the caller
//! supplies the reference through [`property::RefineOptions`].
//!
//! # Example
//!
//! ```
//! use oxixml_fo::fo::FoElement;
//! use oxixml_fo::property::{Length, PropertyId, Value};
//! use oxixml_fo::FoDocument;
//!
//! let tree = FoDocument::parse(
//!     r#"<fo:root xmlns:fo="http://www.w3.org/1999/XSL/Format" font-size="10pt">
//!          <fo:layout-master-set>
//!            <fo:simple-page-master master-name="A4"
//!                                   page-height="297mm" page-width="210mm">
//!              <fo:region-body margin="20mm"/>
//!            </fo:simple-page-master>
//!          </fo:layout-master-set>
//!          <fo:page-sequence master-reference="A4">
//!            <fo:flow flow-name="xsl-region-body">
//!              <fo:block font-size="150%" space-before="6pt" margin-left="1em">
//!                Hello, formatting objects.
//!              </fo:block>
//!            </fo:flow>
//!          </fo:page-sequence>
//!        </fo:root>"#,
//! )
//! .expect("a well-formed XSL-FO document");
//!
//! // Find the fo:block and inspect its refined properties.
//! let block = tree
//!     .elements()
//!     .iter()
//!     .copied()
//!     .find(|id| tree.element(*id) == Some(FoElement::Block))
//!     .expect("the flow contains a block");
//! let properties = tree.properties(block);
//!
//! // `font-size` is inherited, so XSL 1.1 §5.1.4 lets it sit on `fo:root`;
//! // `150%` is 150% of the 10pt that reaches the block through `fo:flow`.
//! assert_eq!(
//!     properties.get(PropertyId::FontSize),
//!     Some(&Value::Length(Length::from_points(15.0)))
//! );
//!
//! // `margin-left="1em"` uses the *refined* font size of this object, and
//! // under the initial `lr-tb` writing mode it also sets `start-indent`.
//! assert_eq!(
//!     properties.get(PropertyId::MarginLeft),
//!     Some(&Value::Length(Length::from_points(15.0)))
//! );
//! assert_eq!(
//!     properties.get(PropertyId::StartIndent),
//!     Some(&Value::Length(Length::from_points(15.0)))
//! );
//! ```
//!
//! # What is here
//!
//! * [`FoDocument`] and [`ParseOptions`] — reading a document into a tree.
//! * [`FoTree`], [`FoNode`], [`NodeId`] and [`ResolvedProperties`] — the tree
//!   itself and the refined values hanging off it.
//! * [`fo`] — the [`FoElement`] vocabulary, the category of each
//!   object (`%block;`, `%inline;`, neutral, out-of-line …), the content models,
//!   and the per-object property applicability tables.
//! * [`property`] — the whole property subsystem: all 272 properties, their
//!   datatypes, the expression language and its core function library,
//!   shorthand expansion, writing modes and the corresponding-property
//!   relation, and the refinement engine.
//! * [`area`] — the area-model and line-breaking primitives described above.
//! * [`FoError`] — every way a document can be rejected.

#![deny(missing_docs)]

pub mod area;
pub mod error;
pub mod fo;
mod parse;
pub mod property;
mod tree;

pub use crate::error::{FoError, Result, SourcePosition};
pub use crate::fo::FoElement;
pub use crate::parse::{FO_NAMESPACE, FoDocument, ParseOptions, XML_NAMESPACE};
pub use crate::property::{PropertyId, PropertyKey, RefineOptions, Value};
pub use crate::tree::{FoNode, FoTree, ForeignElement, NodeId, NodeKind, ResolvedProperties};
