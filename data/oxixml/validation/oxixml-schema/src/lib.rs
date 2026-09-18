//! `oxixml-schema` — [W3C XML Schema](https://www.w3.org/TR/xmlschema11-1/)
//! (XSD) schema assembly and validation, in Pure Rust.
//!
//! Part of the [OxiXML](https://github.com/cool-japan/oxixml) workspace — the
//! COOLJAPAN Pure-Rust foundation for all XML and RDF processing. This crate is
//! the equivalent of the Apache Xerces schema engine: it reads *schema
//! documents*, turns them into the **schema component model** of
//! [Part 1 §3](https://www.w3.org/TR/xmlschema11-1/#components), and then
//! assesses instance documents against it.
//!
//! # The two halves
//!
//! **Assembly.** [`SchemaSet`] collects schema documents — added by hand with
//! [`SchemaSet::add_document`], read from disk with [`SchemaSet::add_file`], or
//! pulled in transitively through `xs:include`, `xs:import`, `xs:redefine` and
//! (in 1.1) `xs:override` by a [`SchemaResolver`]. [`SchemaSet::compile`] then
//! resolves every `QName` reference, merges the constraining facets along each
//! restriction chain, computes the *effective content model* of every complex
//! type as a finite automaton, and checks the schema component constraints —
//! including [Unique Particle Attribution][upa] and Element Declarations
//! Consistent. The result is an immutable [`Schema`].
//!
//! **Assessment.** [`Schema::validate_document`] (or [`Schema::validate_str`])
//! walks an instance, matching each element against a declaration, honouring
//! `xsi:type`, `xsi:nil`, substitution groups, wildcards and their
//! `processContents`, `default`/`fixed` value constraints, the `ID`/`IDREF`
//! rules and the identity constraints `xs:key`, `xs:keyref` and `xs:unique`. It
//! returns every rule that failed — each with the `cvc-…` identifier the
//! Recommendation gives it and an XPath to the offending node — together with a
//! light [PSVI](validate::Psvi).
//!
//! An instance may also point at the schema documents it expects to be assessed
//! against, through the `xsi:schemaLocation` and `xsi:noNamespaceSchemaLocation`
//! of [§4.3.2](https://www.w3.org/TR/xmlschema-1/#xsi_schemaLocation). Those
//! hints are followed only by [`Schema::validate_document_at`] and
//! [`Schema::validate_file`], which know where the instance came from and can
//! therefore resolve them; the components already assembled take precedence
//! over anything a hint brings in, and nothing is ever retrieved over the
//! network. [`Schema::validate_document`] and [`Schema::validate_str`] ignore
//! the hints altogether and touch no files.
//!
//! # Datatypes
//!
//! The whole built-in datatype hierarchy of
//! [Part 2](https://www.w3.org/TR/xmlschema11-2/) is constructed as ordinary
//! components (see [`components`]), so nothing is special-cased: `xs:byte`
//! really is `xs:short` with two bounds facets, and `xs:NCName` really is
//! `xs:Name` with the pattern `[\i-[:]][\c-[:]]*` — compiled by
//! [`oxixml_regex`]. Values live in [`value::Value`] and are compared in the
//! *value space*, so `1.0` and `1.00` are the same `xs:decimal`.
//!
//! # Versions
//!
//! [`XsdVersion::V1_0`] is the default. [`XsdVersion::V1_1`] adds the 1.1
//! datatypes (`xs:dateTimeStamp`, `xs:dayTimeDuration`,
//! `xs:yearMonthDuration`, `xs:anyAtomicType`, `xs:error`), the
//! `explicitTimezone` facet, `notNamespace` and `notQName` on wildcards,
//! `xs:override`, `xs:openContent`, extensible `xs:all` groups,
//! `xs:schema/@defaultAttributes`, identity constraints by `ref`, the
//! conditional-inclusion (`vc:`) attributes — and the three constructs that
//! carry an XPath expression: `xs:assert` ([§3.4.4.2][assert]), the
//! `assertion` facet (Part 2 §4.3.12) and conditional type assignment
//! (`xs:alternative`, §3.3.2), all of which are **evaluated** during
//! assessment.
//!
//! ## Layering: why this crate depends on `oxixml-xpath`
//!
//! XSD 1.1 §3.13 defines those three constructs in terms of XPath 2.0 over the
//! post-schema-validation infoset, so there is no implementing them without an
//! XPath evaluator; this crate therefore depends on the sibling
//! [`oxixml_xpath`]. The layering is deliberate and cycle-free:
//! `oxixml-xpath` is schema-unaware — it evaluates over the XDM of
//! `oxixml-infoset` and never refers to a schema component — so nothing points
//! back this way. [`crate::xpath`] remains this crate's own, much smaller
//! matcher for the *identity-constraint* path language of §3.11.6, which is
//! not XPath 2.0 and needs no evaluator.
//!
//! # Conformance
//!
//! `tests/xsts.rs` runs the [W3C XML Schema Test Suite][xsts] whenever the
//! environment variable `OXIXML_XML_SUITES_DIR` points at a checkout of it, and
//! skips silently otherwise. It makes **two** passes over the suite: one under
//! [`XsdVersion::V1_0`] over every `testGroup` that applies to 1.0, and one
//! under [`XsdVersion::V1_1`] over every group that applies to 1.1.
//!
//! * **XSD 1.0** — 39 007 of 39 131 schema and instance assertions answered
//!   correctly, **99.7 %**.
//! * **XSD 1.1** — 41 275 of 41 427, **99.6 %** over the whole suite, and
//!   **99.1 %** (2 247 of 2 267) over the three 1.1 contributions (Saxonica,
//!   IBM, Oracle), where the 1.1-only machinery actually lives. No group is
//!   skipped for using a 1.1 feature any more.
//!
//! The harness asserts a floor just under each of those rates, so a regression
//! fails the build rather than eroding quietly.
//!
//! ## Documented gaps
//!
//! What is left is, in order of size:
//!
//! * **Particle restriction under 1.1** (§3.9.6.2). XSD 1.1 replaced the
//!   syntactic *rcase-* table of 1.0 with a statement about the languages the
//!   two content models accept, which admits restrictions the table refuses.
//!   That statement is decided exactly wherever the base particle is an
//!   `xs:all` group, and a `choice` restricting a `choice` is no longer held
//!   to the table's order-preservation clause — but every other shape still
//!   goes through the 1.0 table, so a `choice` or `sequence` base whose
//!   restriction only the language test admits is rejected.
//! * A residue of schema representation constraints — chiefly around
//!   `xs:openContent` and wildcards — whose only effect is to *accept* a
//!   schema document the Recommendation rejects. The crate is occasionally
//!   more permissive than the Recommendation about a schema, but not about an
//!   instance.
//! * Two regular-expression corners that belong to the shared
//!   [`oxixml_regex`] engine rather than to this crate: a `-` that opens a
//!   second range inside one character class (`[a-d-b-c]`, which XSD forbids)
//!   is accepted rather than rejected, and a few `\p{…}` category memberships
//!   differ from the Unicode database the suite assumes.
//! * `xs:ENTITY` and `xs:ENTITIES` values are not checked against the entities
//!   a DTD declares.
//!
//! [assert]: https://www.w3.org/TR/xmlschema11-1/#cvc-assertion
//!
//! [upa]: https://www.w3.org/TR/xmlschema-1/#cos-nonambig
//! [xsts]: https://www.w3.org/XML/2004/xml-schema-test-suite/
//!
//! # Example
//!
//! ```
//! use oxixml_schema::SchemaSet;
//!
//! let mut set = SchemaSet::new();
//! set.add_document(
//!     None,
//!     r#"<xs:schema xmlns:xs="http://www.w3.org/2001/XMLSchema">
//!          <xs:element name="order">
//!            <xs:complexType>
//!              <xs:sequence>
//!                <xs:element name="item" maxOccurs="unbounded">
//!                  <xs:complexType>
//!                    <xs:attribute name="sku" use="required">
//!                      <xs:simpleType>
//!                        <xs:restriction base="xs:string">
//!                          <xs:pattern value="[A-Z]{2}-[0-9]{4}"/>
//!                        </xs:restriction>
//!                      </xs:simpleType>
//!                    </xs:attribute>
//!                    <xs:attribute name="qty" type="xs:positiveInteger" default="1"/>
//!                  </xs:complexType>
//!                </xs:element>
//!              </xs:sequence>
//!            </xs:complexType>
//!            <xs:key name="sku-key">
//!              <xs:selector xpath="item"/>
//!              <xs:field xpath="@sku"/>
//!            </xs:key>
//!          </xs:element>
//!        </xs:schema>"#,
//! )?;
//! let schema = set.compile()?;
//!
//! assert!(schema.validate_str(r#"<order><item sku="AB-1234" qty="3"/></order>"#).valid);
//!
//! // A bad `sku` breaks the pattern facet…
//! let bad = schema.validate_str(r#"<order><item sku="oops"/></order>"#);
//! assert_eq!(bad.first_error().map(|error| error.code()), Some("cvc-pattern-valid"));
//!
//! // …and a repeated one breaks the key.
//! let duplicate = schema.validate_str(
//!     r#"<order><item sku="AB-1234"/><item sku="AB-1234"/></order>"#,
//! );
//! assert_eq!(
//!     duplicate.first_error().map(|error| error.code()),
//!     Some("cvc-identity-constraint.4.2.2")
//! );
//!
//! // An empty order does not satisfy `minOccurs="1"` on `item`.
//! assert!(!schema.validate_str("<order/>").valid);
//! # Ok::<(), oxixml_schema::SchemaError>(())
//! ```

#![deny(missing_docs)]

mod assertion;
pub mod automaton;
mod bigdecimal;
mod builtins;
mod compile;
pub mod components;
mod error;
mod names;
mod parse;
mod represent;
mod resolver;
mod schema;
pub mod simple;
mod source;
pub mod validate;
pub mod value;
mod versioning;
mod xdm;
pub mod xpath;

pub use crate::bigdecimal::BigDecimal;
pub use crate::compile::SchemaSet;
pub use crate::error::{SchemaError, SchemaResult, ValidationError};
pub use crate::names::{VC_NS, XML_NS, XSD_NS, XSI_NS, XsdVersion, is_xsd_namespace};
pub use crate::resolver::{
    FileSystemResolver, NullResolver, ResolvedSchema, SchemaResolver, join_location,
};
pub use crate::schema::Schema;
pub use crate::validate::{AttributePsvi, ElementPsvi, Psvi, ValidationOutcome, Validity};
