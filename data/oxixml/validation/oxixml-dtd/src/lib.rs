//! `oxixml-dtd` — DTD parsing and validation for XML 1.0.
//!
//! Part of the [OxiXML](https://github.com/cool-japan/oxixml) workspace — the
//! COOLJAPAN Pure-Rust foundation for all XML and RDF processing. This crate
//! implements the half of the XML 1.0 Recommendation that a *validating*
//! processor needs and a well-formedness parser does not: the grammar of the
//! internal and external DTD subsets, the compiled model they describe, and the
//! validity constraints a document must satisfy against it.
//!
//! # What is here
//!
//! * [`DtdParser`] — reads the `markupdecl` grammar of XML 1.0 §2.8: element
//!   type declarations with their content models (`EMPTY`, `ANY`, mixed, and
//!   `children` models with sequence, choice and the `?`/`*`/`+` operators),
//!   attribute-list declarations (every `AttType` and every `DefaultDecl`),
//!   internal and external general and parameter entities, notations,
//!   conditional sections, comments and processing instructions.
//! * [`Dtd`] — the compiled result. Content models arrive already turned into
//!   finite automata (see [`automaton`]), attribute lists already merged, and
//!   attribute defaults already normalized.
//! * [`Validator`] — checks a stream of element and character events against a
//!   [`Dtd`]: content models, attribute types and defaults, `ID` uniqueness,
//!   `IDREF` resolution, notations, and the standalone-document declaration.
//! * [`DocumentValidator`] / [`validate_str`] — the same, driven end to end
//!   over a document with the [`oxixml_xml`] pull parser, including recursive
//!   expansion of general entities as content.
//!
//! # Safety against entity expansion
//!
//! Two hardening measures are on by default and cannot be switched off, only
//! adjusted:
//!
//! * **Expansion budget.** Every entity expansion — parameter entities while
//!   reading the DTD, general entities in attribute values and in content — is
//!   charged against an [`ExpansionLimits`] budget bounding nesting depth,
//!   expansion count and total expanded size. A "billion laughs" bomb hits it
//!   in milliseconds and produces
//!   [`DtdErrorKind::EntityExpansionLimit`] rather than exhausting memory.
//! * **No implicit external access.** The external subset and external entities
//!   are only read when the caller installs an [`EntityResolver`]. The default,
//!   [`DenyExternalEntities`], resolves nothing, so a document cannot make the
//!   parser open a file or a socket.
//!
//! ```
//! use oxixml_dtd::{DtdParser, ExpansionLimits};
//!
//! // The classic exponential-expansion bomb, in miniature.
//! let bomb = r#"
//!     <!ENTITY lol "lol">
//!     <!ENTITY lol1 "&lol;&lol;&lol;&lol;&lol;&lol;&lol;&lol;&lol;&lol;">
//!     <!ENTITY lol2 "&lol1;&lol1;&lol1;&lol1;&lol1;&lol1;&lol1;&lol1;&lol1;&lol1;">
//!     <!ENTITY lol3 "&lol2;&lol2;&lol2;&lol2;&lol2;&lol2;&lol2;&lol2;&lol2;&lol2;">
//! "#;
//! let dtd = DtdParser::new().with_limits(ExpansionLimits::STRICT).parse_internal_subset(bomb)?;
//!
//! // Declaring it is cheap; expanding it is not — and the guard says so.
//! let mut budget = ExpansionLimits::STRICT.budget();
//! let error = dtd.expand_general_entity("lol3", &mut budget).expect_err("over budget");
//! assert!(error.is_expansion_limit());
//! # Ok::<(), oxixml_dtd::DtdError>(())
//! ```
//!
//! # Example
//!
//! ```
//! use oxixml_dtd::{DtdParser, ValidationOptions, Validator, validate_str};
//!
//! // Parse a DTD and look at the compiled model.
//! let dtd = DtdParser::new().parse_internal_subset(
//!     r#"<!ELEMENT book (title, chapter+)>
//!        <!ELEMENT title (#PCDATA)>
//!        <!ELEMENT chapter (#PCDATA)>
//!        <!ATTLIST chapter id ID #REQUIRED lang NMTOKEN "en">"#,
//! )?;
//! assert_eq!(dtd.element("book").map(|d| d.content().to_string()).as_deref(), Some("(title,chapter+)"));
//!
//! // Drive the validator by hand…
//! let mut validator = Validator::new(&dtd, ValidationOptions::default());
//! validator.start_element("book", &[]);
//! validator.start_element("title", &[]);
//! validator.characters("Moby-Dick");
//! validator.end_element("title");
//! let defaults = validator.start_element("chapter", &[("id", "c1")]);
//! assert_eq!(defaults, vec![("lang".to_owned(), "en".to_owned())]);
//! validator.end_element("chapter");
//! validator.end_element("book");
//! assert!(validator.finish().is_empty());
//!
//! // …or validate a whole document at once.
//! let report = validate_str(
//!     r#"<!DOCTYPE book [
//!          <!ELEMENT book (title)>
//!          <!ELEMENT title (#PCDATA)>
//!        ]>
//!        <book><title>Moby-Dick</title></book>"#,
//! )?;
//! assert!(report.is_valid());
//! # Ok::<(), Box<dyn std::error::Error>>(())
//! ```

#![deny(missing_docs)]
#![forbid(unsafe_code)]

pub mod automaton;
pub mod document;
pub mod entity;
pub mod error;
pub mod limits;
pub mod model;
mod names;
pub mod parser;
pub mod resolver;
pub mod validate;

pub use automaton::{CompiledContent, ContentAutomaton, ContentMatcher, Dfa, PositionAutomaton};
pub use document::{DocumentValidator, ValidationReport, validate_str};
pub use error::{DtdError, DtdErrorKind, Error, ExpansionLimit, ValidityError, ValidityErrorKind};
pub use limits::{Budget, ExpansionLimits};
pub use model::{
    AttributeDecl, AttributeList, AttributeType, ContentParticle, ContentSpec, DefaultDecl, Dtd,
    ElementDecl, EntityDecl, EntityDefinition, EntityKind, ExternalId, MixedContent, NotationDecl,
    Occurrence,
};
pub use parser::{DtdParser, MAX_CONTENT_MODEL_DEPTH, MAX_CONTENT_MODEL_POSITIONS};
pub use resolver::{
    DenyExternalEntities, EntityResolver, ExternalKind, FilesystemResolver, MemoryResolver,
    ResolveRequest, ResolvedEntity, ResolverError,
};
pub use validate::{ValidationOptions, Validator};
