//! `oxixml-sax` — SAX2-style push parsing driven by [`oxixml_xml`].
//!
//! Part of the [OxiXML](https://github.com/cool-japan/oxixml) workspace — the
//! COOLJAPAN Pure-Rust foundation for all XML and RDF processing.
//!
//! This crate is a Rust translation of the classic
//! [SAX2](https://www.saxproject.org/) event model on top of
//! [`oxixml_xml::reader::NsReader`]: instead of pulling events one at a time,
//! [`XMLReader::parse`] pumps the whole document and pushes each event to
//! whichever handler traits you install:
//!
//! * [`ContentHandler`] — element structure, character data, namespace prefix
//!   scope, processing instructions and skipped entities. This is the interface
//!   most consumers need.
//! * [`LexicalHandler`] — comments, the `CDATA`/text boundary, and the document
//!   type declaration's boundaries. Mirrors `org.xml.sax.ext.LexicalHandler`.
//! * [`DtdHandler`] — `NOTATION` declarations and unparsed (`NDATA`) entities.
//!   Mirrors `org.xml.sax.DTDHandler`.
//! * [`DeclHandler`] — `ELEMENT`, `ATTLIST` and general `ENTITY` declarations
//!   from the internal DTD subset. Mirrors `org.xml.sax.ext.DeclHandler`.
//! * [`ErrorHandler`] — parse errors, each paired with a [`Locator`] snapshot of
//!   where in the document it happened. Mirrors `org.xml.sax.ErrorHandler`.
//! * [`EntityResolver`] — a hook to observe (though not, currently, redirect I/O
//!   for) external identifiers. Mirrors `org.xml.sax.EntityResolver`.
//!
//! Every trait method has a no-op default implementation, so a handler only
//! needs to override what it actually uses. Install the handlers you care about
//! into a [`Handlers`] bundle and hand it to [`XMLReader::parse`] (or one of its
//! `parse_str`/`parse_reader`/`parse_file` siblings).
//!
//! # Example
//!
//! ```
//! use oxixml_sax::{Attributes, ContentHandler, Handlers, SaxError, XMLReader};
//!
//! #[derive(Default)]
//! struct CollectTitles {
//!     in_title: bool,
//!     titles: Vec<String>,
//! }
//!
//! impl ContentHandler for CollectTitles {
//!     fn start_element(
//!         &mut self,
//!         _uri: Option<&str>,
//!         local_name: &str,
//!         _qname: &str,
//!         _attributes: &Attributes,
//!     ) -> Result<(), SaxError> {
//!         self.in_title = local_name == "title";
//!         if self.in_title {
//!             self.titles.push(String::new());
//!         }
//!         Ok(())
//!     }
//!
//!     fn characters(&mut self, text: &str) -> Result<(), SaxError> {
//!         if self.in_title {
//!             self.titles.last_mut().expect("pushed on start_element").push_str(text);
//!         }
//!         Ok(())
//!     }
//!
//!     fn end_element(&mut self, _uri: Option<&str>, local_name: &str, _qname: &str) -> Result<(), SaxError> {
//!         if local_name == "title" {
//!             self.in_title = false;
//!         }
//!         Ok(())
//!     }
//! }
//!
//! let xml = "<library><book><title>Cat's Cradle</title></book></library>";
//!
//! let mut handler = CollectTitles::default();
//! let mut handlers = Handlers::new().with_content(&mut handler);
//! XMLReader::new().parse_str(xml, &mut handlers)?;
//!
//! assert_eq!(handler.titles, vec!["Cat's Cradle".to_string()]);
//! # Ok::<(), SaxError>(())
//! ```

#![deny(missing_docs)]

mod attributes;
mod content_handler;
mod decl_handler;
mod dtd_handler;
mod dtd_subset;
mod entity;
mod entity_resolver;
mod error;
mod error_handler;
mod handlers;
mod lexical_handler;
mod locator;
mod reader;

pub use attributes::{Attribute, Attributes};
pub use content_handler::{ContentHandler, NullContentHandler};
pub use decl_handler::{DeclHandler, NullDeclHandler};
pub use dtd_handler::{DtdHandler, NullDtdHandler};
pub use entity_resolver::{EntityResolver, NullEntityResolver};
pub use error::{Result, SaxError};
pub use error_handler::{ErrorHandler, NullErrorHandler};
pub use handlers::{FEATURE_NAMESPACE_PREFIXES, FEATURE_NAMESPACES, Feature, Handlers};
pub use lexical_handler::{LexicalHandler, NullLexicalHandler};
pub use locator::{Locator, SaxParseException};
pub use reader::XMLReader;
