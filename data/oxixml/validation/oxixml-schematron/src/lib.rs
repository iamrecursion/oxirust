//! `oxixml-schematron` — [ISO/IEC 19757-3](https://www.iso.org/standard/55982.html) Schematron
//! rule-based validation, on the [`oxixml_xpath`] engine.
//!
//! Part of the [OxiXML](https://github.com/cool-japan/oxixml) workspace — the COOLJAPAN
//! Pure-Rust foundation for all XML and RDF processing.
//!
//! Schematron is a different kind of schema language from DTD, XSD or RELAX NG: instead of
//! describing a grammar, a schema is a set of *patterns*, each a set of *rules* that fire on the
//! nodes matching a *context*, each asserting or reporting the truth of an XPath *test*. This
//! crate [`Schematron::compile`]s a schema once — resolving `sch:phase` activation, `sch:let`
//! variables, `sch:pattern is-a`/`sch:param` instantiation of abstract patterns, and
//! `sch:rule`/`sch:extends` rule inheritance — into a [`Compiled`] schema that
//! [`Compiled::validate`]s any number of documents, producing a structured
//! [`SchematronReport`] with an [SVRL](https://schematron.com/document/321.html) serialization.
//!
//! # Example
//!
//! ```
//! use oxixml_schematron::Schematron;
//!
//! let schema = r#"
//!     <schema xmlns="http://purl.oclc.org/dsdl/schematron">
//!       <ns prefix="c" uri="urn:example:catalog"/>
//!       <pattern id="prices">
//!         <rule context="c:book">
//!           <assert test="c:price" id="has-price">every book needs a price</assert>
//!           <report test="c:price &lt; 0" role="warning">a negative price looks wrong</report>
//!         </rule>
//!       </pattern>
//!     </schema>
//! "#;
//!
//! let compiled = Schematron::compile(schema, None)?;
//!
//! let good = r#"<c:catalog xmlns:c="urn:example:catalog">
//!                 <c:book><c:price>9.99</c:price></c:book>
//!               </c:catalog>"#;
//! assert!(compiled.validate(good)?.is_valid());
//!
//! let bad = r#"<c:catalog xmlns:c="urn:example:catalog"><c:book/></c:catalog>"#;
//! let report = compiled.validate(bad)?;
//! assert!(!report.is_valid());
//! let failure = report.failed_asserts().next().expect("one failure");
//! assert_eq!(failure.id.as_deref(), Some("has-price"));
//! assert_eq!(failure.message, "every book needs a price");
//! # Ok::<(), oxixml_schematron::SchematronError>(())
//! ```
//!
//! # What is here
//!
//! | Concern | Type |
//! |---|---|
//! | Compiling | [`Schematron::compile`], [`Schematron::compile_with`] |
//! | A compiled schema | [`Compiled`] |
//! | `sch:include`/`sch:extends href` resolution | [`parse::IncludeResolver`] |
//! | Validating | [`Compiled::validate`], [`Compiled::validate_tree`] |
//! | Results | [`SchematronReport`], [`ActivePatternReport`], [`FiredRule`], [`CheckResult`], [`CheckOutcome`] |
//! | Errors | [`SchematronError`], [`SchematronResult`] |
//!
//! # Query binding
//!
//! Every XPath expression in a schema — `context`, `test`, `let`, `select`, `path`, `subject` —
//! compiles under [`oxixml_xpath`], the language level chosen by `queryBinding`: the default and
//! `"xslt"`/`"xslt1"` select XPath 1.0 semantics (what XSLT 1.0's Schematron skeleton evaluates
//! against), `"xslt2"` selects XPath 2.0, and `"xslt3"` selects XPath 3.0. See
//! [`compile::query_binding_version`] for the exact mapping.
//!
//! **The `xslt` query binding's *executable* half — running an embedded XSLT sequence
//! constructor, as opposed to compiling its `select`/`test`/`context` attributes as XPath — is
//! out of scope for this crate and left for a future XSLT-hosted validator.** Every schema this
//! crate accepts is valid ISO Schematron; the boundary is that `sch:let` values, messages and
//! attributes must be XPath expressions rather than arbitrary `xsl:` sequence constructors,
//! which is how the overwhelming majority of real-world Schematron schemas are written.
//!
//! `context` is a restricted subset of XPath — an [XSLT match pattern] — that this crate
//! evaluates by rewriting it into a selecting path expression from the document root (see
//! [`compile::context_selecting_expression`]) rather than by implementing XSLT's full pattern
//! recognition algorithm, which needs an XSLT processor's node-test machinery this crate does
//! not have. This is the same translation essentially every non-XSLT Schematron engine performs,
//! and it is exact for the pattern grammar real schemas use in practice: axis-free, `/`- and
//! `|`-composed steps with predicates.
//!
//! [XSLT match pattern]: https://www.w3.org/TR/xslt-30/#patterns

#![deny(missing_docs)]
#![forbid(unsafe_code)]

pub mod compile;
mod error;
mod location;
mod message;
mod model;
pub mod parse;
pub mod report;
mod validate;

pub use compile::{Compiled, Schematron};
pub use error::{SchematronError, SchematronResult};
pub use model::{SCHEMATRON_1_5_NS, SCHEMATRON_NS, SVRL_NS};
pub use report::{ActivePatternReport, CheckOutcome, CheckResult, FiredRule, SchematronReport};
