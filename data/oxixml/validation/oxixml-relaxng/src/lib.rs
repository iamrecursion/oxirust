//! `oxixml-relaxng` — a complete [RELAX NG][spec] validator in Pure Rust, for both
//! the XML syntax and the [compact syntax][rnc].
//!
//! Part of the [OxiXML](https://github.com/cool-japan/oxixml) workspace — the
//! COOLJAPAN Pure-Rust foundation for all XML and RDF processing.
//!
//! # What it does
//!
//! Compiling a schema runs the whole pipeline the specification prescribes, in
//! order and with every constraint checked:
//!
//! 1. the schema is parsed into the **data model** of
//!    [§2](https://relaxng.org/spec-20011203.html#data-model) — from XML, or from
//!    the compact syntax through the translation of the RNC specification;
//! 2. it is checked against the **full syntax** of
//!    [§3](https://relaxng.org/spec-20011203.html#syntax);
//! 3. the twenty-one **simplification rules** of
//!    [§4](https://relaxng.org/spec-20011203.html#simplification) are applied,
//!    including `include`/`externalRef` expansion, `combine`, grammar flattening
//!    and the `notAllowed`/`empty` propagation, producing the **simple syntax** of
//!    [§5](https://relaxng.org/spec-20011203.html#simple-syntax);
//! 4. the **restrictions** of
//!    [§7](https://relaxng.org/spec-20011203.html#restrictions) are verified —
//!    prohibited paths, string sequences, duplicate attributes and the interleave
//!    rules;
//! 5. the result is compiled into a hash-consed pattern arena.
//!
//! Validation is [James Clark's derivative algorithm][deriv]: each event turns
//! the current pattern into the pattern the rest of the document must match, and
//! the document is valid when the final pattern is nullable. Errors are recovered
//! from, so one run reports every problem it can rather than only the first.
//!
//! Two datatype libraries are built in: the one identified by the empty URI
//! (`string` and `token`), and the **W3C XML Schema** library — the full XSD 1.0
//! built-in hierarchy, with its constraining facets available as `param`
//! elements, backed by [`oxixml_xsd`] and [`oxixml_regex`].
//!
//! [spec]: https://relaxng.org/spec-20011203.html
//! [rnc]: https://relaxng.org/compact-20021121.html
//! [deriv]: https://relaxng.org/jclark/derivative.html
//!
//! # Example
//!
//! ```
//! use oxixml_relaxng::RelaxNg;
//!
//! // The compact syntax and the XML syntax compile to the same thing.
//! let schema = RelaxNg::from_compact(
//!     r#"
//!     element addressBook {
//!       element card {
//!         element name { text },
//!         element email { text }
//!       }*
//!     }
//!     "#,
//! )?;
//!
//! assert!(schema.validate(
//!     r#"<addressBook>
//!          <card><name>John Smith</name><email>js@example.com</email></card>
//!        </addressBook>"#
//! ).is_ok());
//!
//! // A missing `email` element is reported, with the name that was expected.
//! let errors = schema
//!     .validate("<addressBook><card><name>John Smith</name></card></addressBook>")
//!     .expect_err("the card is incomplete");
//! assert!(errors[0].to_string().contains("email"));
//! # Ok::<(), Box<dyn std::error::Error>>(())
//! ```

#![deny(missing_docs)]
#![forbid(unsafe_code)]

mod compact;
mod compile;
mod datatype;
mod deriv;
mod error;
mod flatten;
mod grammar_check;
mod name;
mod pattern;
mod resolver;
mod restrict;
mod simplify;
mod tree;
mod uri;
mod validate;
mod xml_syntax;

use std::io::Read;
use std::path::Path;

pub use crate::error::{Position, RelaxNgError, SchemaError, SchemaErrorKind, ValidationErrorKind};
pub use crate::resolver::{FileResolver, NoResolver, ResolveError, SchemaResolver};

use crate::compile::Schema;
use crate::datatype::Libraries;
use crate::error::syntax;
use crate::tree::Element;

/// A compiled RELAX NG schema.
///
/// Compiling is the expensive step; a `RelaxNg` can then validate any number of
/// documents. Validation needs only a shared reference and never mutates the
/// compiled schema.
#[derive(Debug)]
pub struct RelaxNg {
    schema: Schema,
}

impl RelaxNg {
    /// Compiles a schema written in the XML syntax.
    ///
    /// The schema has no base URI, so `include` and `externalRef` can only be
    /// used with absolute `file:` URIs. Use [`RelaxNg::from_file`] or
    /// [`RelaxNgBuilder`] when relative references have to resolve.
    ///
    /// # Errors
    ///
    /// Returns a [`SchemaError`] when the document is not a correct RELAX NG
    /// schema.
    ///
    /// # Examples
    ///
    /// ```
    /// use oxixml_relaxng::RelaxNg;
    ///
    /// let schema = RelaxNg::from_xml(
    ///     r#"<element name="greeting" xmlns="http://relaxng.org/ns/structure/1.0">
    ///          <text/>
    ///        </element>"#,
    /// )?;
    /// assert!(schema.validate("<greeting>hello</greeting>").is_ok());
    /// assert!(schema.validate("<greeting><x/></greeting>").is_err());
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    pub fn from_xml(schema: &str) -> Result<Self, SchemaError> {
        RelaxNgBuilder::new().compile_xml(schema)
    }

    /// Compiles a schema written in the compact syntax.
    ///
    /// # Errors
    ///
    /// Returns a [`SchemaError`] when the text is not a correct RELAX NG compact
    /// syntax schema.
    ///
    /// # Examples
    ///
    /// ```
    /// use oxixml_relaxng::RelaxNg;
    ///
    /// let schema = RelaxNg::from_compact("element greeting { text }")?;
    /// assert!(schema.validate("<greeting>hello</greeting>").is_ok());
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    pub fn from_compact(schema: &str) -> Result<Self, SchemaError> {
        RelaxNgBuilder::new().compile_compact(schema)
    }

    /// Compiles a schema from a file, using the file as the base URI so that
    /// relative `include` and `externalRef` references resolve.
    ///
    /// A `.rnc` extension selects the compact syntax; anything else is read as
    /// XML.
    ///
    /// # Errors
    ///
    /// Returns a [`SchemaError`] when the file cannot be read or is not a correct
    /// RELAX NG schema.
    pub fn from_file(path: impl AsRef<Path>) -> Result<Self, SchemaError> {
        let path = path.as_ref();
        let text = std::fs::read_to_string(path)
            .map_err(|error| syntax(format!("cannot read `{}`: {error}", path.display())))?;
        let absolute = path.canonicalize().unwrap_or_else(|_| path.to_path_buf());
        let builder = RelaxNgBuilder::new().base_uri(resolver::path_to_file_uri(&absolute));
        if path.extension().is_some_and(|extension| extension == "rnc") {
            builder.compile_compact(&text)
        } else {
            builder.compile_xml(&text)
        }
    }

    /// Validates an instance document held in a string.
    ///
    /// # Errors
    ///
    /// Returns every [`RelaxNgError`] found. The list is never empty when the
    /// document is invalid.
    pub fn validate(&self, document: &str) -> Result<(), Vec<RelaxNgError>> {
        let options = oxixml_dom::BuilderOptions::new()
            .with_comments(false)
            .with_processing_instructions(false)
            .with_cdata_as_text(true)
            .with_well_formedness_checks(true);
        let document = oxixml_dom::build::parse_str_with(document, &options).map_err(|error| {
            vec![RelaxNgError::new(
                ValidationErrorKind::NotWellFormed,
                format!("the instance is not well-formed XML: {error}"),
            )]
        })?;
        self.validate_document(&document)
    }

    /// Validates an instance document read from a reader.
    ///
    /// # Errors
    ///
    /// Returns every [`RelaxNgError`] found, or a single
    /// [`ValidationErrorKind::NotWellFormed`] error when the input cannot be read
    /// or parsed.
    pub fn validate_reader<R: Read>(&self, mut reader: R) -> Result<(), Vec<RelaxNgError>> {
        let mut text = String::new();
        reader.read_to_string(&mut text).map_err(|error| {
            vec![RelaxNgError::new(
                ValidationErrorKind::NotWellFormed,
                format!("cannot read the instance: {error}"),
            )]
        })?;
        self.validate(&text)
    }

    /// Validates an already-parsed document tree.
    ///
    /// # Errors
    ///
    /// Returns every [`RelaxNgError`] found.
    ///
    /// # Examples
    ///
    /// ```
    /// use oxixml_relaxng::RelaxNg;
    ///
    /// let schema = RelaxNg::from_compact("element a { attribute b { xsd:int } }")?;
    /// let document = oxixml_dom::build::parse_str(r#"<a b="42"/>"#)?;
    /// assert!(schema.validate_document(&document).is_ok());
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    pub fn validate_document(
        &self,
        document: &oxixml_dom::Document,
    ) -> Result<(), Vec<RelaxNgError>> {
        let errors = validate::validate_document(&self.schema, document);
        if errors.is_empty() {
            Ok(())
        } else {
            Err(errors)
        }
    }
}

/// Compiles a schema with a base URI and a resource resolver.
///
/// # Examples
///
/// ```
/// use oxixml_relaxng::{RelaxNgBuilder, ResolveError, SchemaResolver};
///
/// struct InMemory;
///
/// impl SchemaResolver for InMemory {
///     fn resolve(&self, uri: &str) -> Result<String, ResolveError> {
///         if uri == "http://example.com/card.rng" {
///             Ok(r#"<element name="card" xmlns="http://relaxng.org/ns/structure/1.0">
///                     <text/>
///                   </element>"#
///                 .to_owned())
///         } else {
///             Err(ResolveError::new(format!("no such resource: {uri}")))
///         }
///     }
/// }
///
/// let schema = RelaxNgBuilder::new()
///     .base_uri("http://example.com/book.rng")
///     .resolver(Box::new(InMemory))
///     .compile_xml(
///         r#"<element name="deck" xmlns="http://relaxng.org/ns/structure/1.0">
///              <zeroOrMore><externalRef href="card.rng"/></zeroOrMore>
///            </element>"#,
///     )?;
///
/// assert!(schema.validate("<deck><card>ace</card></deck>").is_ok());
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
pub struct RelaxNgBuilder {
    base_uri: Option<String>,
    resolver: Box<dyn SchemaResolver>,
}

impl std::fmt::Debug for RelaxNgBuilder {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("RelaxNgBuilder")
            .field("base_uri", &self.base_uri)
            .finish_non_exhaustive()
    }
}

impl Default for RelaxNgBuilder {
    fn default() -> Self {
        Self::new()
    }
}

impl RelaxNgBuilder {
    /// Creates a builder with no base URI that reads `include` and `externalRef`
    /// targets from the filesystem.
    #[must_use]
    pub fn new() -> Self {
        Self {
            base_uri: None,
            resolver: Box::new(FileResolver),
        }
    }

    /// Sets the base URI that relative `href` values resolve against.
    #[must_use]
    pub fn base_uri(mut self, uri: impl Into<String>) -> Self {
        self.base_uri = Some(uri.into());
        self
    }

    /// Replaces the resolver used to retrieve `include` and `externalRef`
    /// targets.
    #[must_use]
    pub fn resolver(mut self, resolver: Box<dyn SchemaResolver>) -> Self {
        self.resolver = resolver;
        self
    }

    /// Compiles a schema written in the XML syntax.
    ///
    /// # Errors
    ///
    /// Returns a [`SchemaError`] when the document is not a correct RELAX NG
    /// schema.
    pub fn compile_xml(&self, text: &str) -> Result<RelaxNg, SchemaError> {
        let root = xml_syntax::parse_str(text, self.base_uri.as_deref())?;
        self.finish(root)
    }

    /// Compiles a schema written in the compact syntax.
    ///
    /// # Errors
    ///
    /// Returns a [`SchemaError`] when the text is not a correct RELAX NG compact
    /// syntax schema.
    pub fn compile_compact(&self, text: &str) -> Result<RelaxNg, SchemaError> {
        let root = compact::parse(text, self.base_uri.as_deref())?;
        self.finish(root)
    }

    fn finish(&self, mut root: Element) -> Result<RelaxNg, SchemaError> {
        grammar_check::check_pattern(&root)?;
        simplify::run(&mut root, self.resolver.as_ref(), &Libraries)?;
        let grammar = flatten::run(root)?;
        restrict::check(&grammar)?;
        let schema = compile::compile(&grammar)?;
        Ok(RelaxNg { schema })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_simple_schema_round_trips() -> Result<(), SchemaError> {
        let schema = RelaxNg::from_xml(
            r#"<element name="foo" xmlns="http://relaxng.org/ns/structure/1.0">
                 <attribute name="bar"><text/></attribute>
                 <zeroOrMore><element name="baz"><empty/></element></zeroOrMore>
               </element>"#,
        )?;
        assert!(schema.validate(r#"<foo bar="x"/>"#).is_ok());
        assert!(
            schema
                .validate(r#"<foo bar="x"><baz/><baz/></foo>"#)
                .is_ok()
        );
        assert!(schema.validate("<foo/>").is_err());
        assert!(schema.validate(r#"<foo bar="x"><qux/></foo>"#).is_err());
        Ok(())
    }

    #[test]
    fn interleave_and_datatypes() -> Result<(), SchemaError> {
        let schema = RelaxNg::from_compact(
            r#"
            element root {
              element a { xsd:int } & element b { xsd:string }
            }
            "#,
        )?;
        assert!(schema.validate("<root><a>1</a><b>x</b></root>").is_ok());
        assert!(schema.validate("<root><b>x</b><a>1</a></root>").is_ok());
        assert!(schema.validate("<root><a>x</a><b>x</b></root>").is_err());
        Ok(())
    }

    #[test]
    fn errors_are_reported_for_several_problems() -> Result<(), SchemaError> {
        let schema =
            RelaxNg::from_compact("element r { element a { empty }, element b { empty } }")?;
        let errors = schema
            .validate("<r><x/><y/></r>")
            .expect_err("neither child is allowed");
        assert!(errors.len() >= 2, "{errors:?}");
        Ok(())
    }
}
