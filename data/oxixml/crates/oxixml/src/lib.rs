//! # OxiXML
//!
//! The COOLJAPAN Pure-Rust foundation for all XML and RDF processing.
//!
//! OxiXML is a from-scratch, dependency-free XML and RDF stack for Rust. At the parsing layer
//! it is a drop-in replacement for [`quick-xml`](https://docs.rs/quick-xml) (see the `xml`
//! feature, and the `quickxml-compat` shim that lets existing `quick_xml`-based code switch
//! over with a one-line `Cargo.toml` change). At the RDF layer it replaces the
//! [Oxigraph](https://oxigraph.org) toolchain — `oxrdf`, `oxttl`, `oxrdfxml`, `oxjsonld`,
//! `oxrdfio`, `spargebra`, `sparesults`, `oxsdatatypes`, `oxiri`, and `rdf-canon` all have a
//! feature here with an equivalent or compatible API. And with the tree, validation, query,
//! transform, and security tiers this crate gathers up, it assembles an XML-processing surface
//! comparable to what a JVM stack builds from Apache Xerces (DOM/SAX/XSD/RELAX NG),
//! Xalan/Saxon (XSLT/XPath/XQuery), and Apache Santuario (XML-DSig/XML-ENC) — every one of
//! those roles implemented **in Pure Rust**: no C, C++, or Fortran anywhere in the default
//! build, for any feature.
//!
//! The workspace behind this facade is around forty independently publishable crates. This
//! crate (`oxixml`) is the umbrella over all of them, each hidden behind its own Cargo feature
//! so that a dependent pays only for what it asks for.
//!
//! # Choosing features
//!
//! There is no default feature — enable exactly what you need:
//!
//! ```toml
//! [dependencies]
//! oxixml = { version = "0.1.2", features = ["turtle", "xslt"] }
//! ```
//!
//! or reach for everything at once with `features = ["full"]`. Enabling a feature also enables
//! whatever it is built on — `xslt` pulls in `xpath` and `dom` (which pulls in `xml`); `rdfa`
//! pulls in `html` (which pulls in `dom`) and `rdf`; and so on — so the module it names, and
//! every module that one depends on, are all reachable the moment it is turned on.
//!
//! # The feature families
//!
//! ## Foundation
//!
//! The atoms every other tier is built from.
//!
//! | Feature | Module | Crate | What it gives you |
//! |---|---|---|---|
//! | `support` | `support` | `oxixml-support` | SWAR byte scanning, line/column tracking, SHA/HMAC, and a minimal bigint. |
//! | `unicode` | `unicode` | `oxixml-unicode` | UCD character classes, blocks, scripts, case folding, NFC/NFD/NFKC/NFKD. |
//! | `encoding` | `encoding` | `oxixml-encoding` | UTF-8/16/32, Latin-1/windows-1252, and legacy Japanese (Shift_JIS/EUC-JP/ISO-2022-JP). |
//! | `text` | `text` | `oxixml-text` | base64, hex, percent-encoding, ECMAScript/XSD number formatting. |
//! | `regex` | `regex` | `oxixml-regex` | The XSD `pattern` and XPath `fn:matches` regular-expression engine. |
//! | `iri` | `iri` | `oxixml-iri` | RFC 3987 IRIs: parse, resolve, relativize. |
//! | `langtag` | `langtag` | `oxixml-langtag` | RFC 5646 language tags: parse, normalize, compare. |
//! | `xsd` | `xsd` | `oxixml-xsd` | The XML Schema Part 2 built-in datatypes (`decimal`, `dateTime`, `duration`, …). |
//! | `qname` | `qname` | `oxixml-qname` | `NCName`/`QName`, `ExpandedName`, and the namespace-binding stack. |
//!
//! ## XML core
//!
//! | Feature | Module | Crate | What it gives you |
//! |---|---|---|---|
//! | `xml` | `xml` | `oxixml-xml` | The XML 1.0/1.1 pull parser and writer — quick-xml 0.41-compatible. |
//! | `quickxml-compat` | `quickxml_compat` | `oxixml-quickxml-compat` | A drop-in `quick-xml` shim: alias it as `package = "oxixml-quickxml-compat"`. |
//!
//! ## Trees
//!
//! | Feature | Module | Crate | What it gives you |
//! |---|---|---|---|
//! | `dom` | `dom` | `oxixml-dom` | An arena-based DOM Level 3 Core document tree, built on `xml`. |
//! | `sax` | `sax` | `oxixml-sax` | A SAX2-style push (`ContentHandler`) API over `xml`. |
//! | `infoset` | `infoset` | `oxixml-infoset` | The XDM tree that `xpath`/`xquery`/`xslt`/`schema` evaluate over. |
//! | `html` | `html` | `oxixml-html` | An HTML5 tokenizer/tree-constructor producing an ordinary `dom` document. |
//! | `svg` | `svg` | `oxixml-svg` | A typed SVG document model, microsyntax toolkit, geometry engine, builder, and linter, built on `dom`. |
//!
//! ## Validation
//!
//! | Feature | Module | Crate | What it gives you |
//! |---|---|---|---|
//! | `dtd` | `dtd` | `oxixml-dtd` | DTD parsing, content-model compilation, and instance validation. |
//! | `schema` | `schema` | `oxixml-schema` | W3C XML Schema (XSD) 1.0/1.1 assembly and validation — an Apache Xerces analogue. |
//! | `relaxng` | `relaxng` | `oxixml-relaxng` | RELAX NG, XML and compact syntax. |
//! | `schematron` | `schematron` | `oxixml-schematron` | ISO Schematron rules, compiled and run over `xpath`. |
//! | `catalog` | `catalog` | `oxixml-catalog` | OASIS XML Catalogs: `public`/`system`/URI resolution and rewriting. |
//! | `xinclude` | `xinclude` | `oxixml-xinclude` | XInclude processing over a `dom` tree. |
//!
//! ## Query & transform
//!
//! | Feature | Module | Crate | What it gives you |
//! |---|---|---|---|
//! | `functions` | `functions` | `oxixml-fn` | The XPath/XQuery/XSLT Functions & Operators (F&O) library. |
//! | `xpath` | `xpath` | `oxixml-xpath` | XPath 1.0-3.1, evaluated over `infoset`. |
//! | `xquery` | `xquery` | `oxixml-xquery` | XQuery 3.1, built on `xpath`. |
//! | `xslt` | `xslt` | `oxixml-xslt` | XSLT 1.0, plus the essentials of 2.0/3.0 — a Xalan/Saxon analogue. |
//!
//! ## Security & format
//!
//! | Feature | Module | Crate | What it gives you |
//! |---|---|---|---|
//! | `c14n` | `c14n` | `oxixml-c14n` | Canonical XML 1.0/1.1/2.0 and Exclusive XML Canonicalization, over `dom`. |
//! | `dsig` | `dsig` | `oxixml-dsig` | XML Signature (XML-DSig) generation and verification. |
//! | `enc` | `enc` | `oxixml-enc` | XML Encryption (XML-ENC) element/content encryption and decryption. |
//! | `fo` | `fo` | `oxixml-fo` | An XSL-FO 1.1 formatting-object tree, property refinement, and area-model primitives. |
//!
//! ## RDF
//!
//! | Feature | Module | Crate | What it gives you |
//! |---|---|---|---|
//! | `rdf` | `model` | `oxixml-model` | The core RDF/RDF-star data model: `NamedNode`, `BlankNode`, `Literal`, `Triple`, `Quad`, `Dataset`. |
//! | `json` | `json` | `oxixml-json` | A streaming JSON event parser/serializer, shared by `jsonld` and `sparql-results`. |
//! | `turtle` | `turtle` | `oxixml-turtle` | Turtle, TriG, N-Triples, N-Quads, and N3. |
//! | `rdfxml` | `rdfxml` | `oxixml-rdfxml` | RDF/XML. |
//! | `jsonld` | `jsonld` | `oxixml-jsonld` | JSON-LD 1.1: expand, compact, flatten, and the `toRdf`/`fromRdf` algorithms. |
//! | `rdfa` | `rdfa` | `oxixml-rdfa` | RDFa 1.1 Core, extracted from an `html` document. |
//! | `trix` | `trix` | `oxixml-trix` | TriX, the XML serialization of an RDF dataset. |
//! | `hdt` | `hdt` | `oxixml-hdt` | HDT (Header-Dictionary-Triples), a compact binary RDF representation. |
//! | `canon` | `canon` | `oxixml-canon` | RDF Dataset Canonicalization (RDFC-1.0 / URDNA2015). |
//! | `rdfio` | `io` | `oxixml-io` | Unified format detection, parsing, and serialization across `turtle`/`rdfxml`/`jsonld`/`trix`. |
//! | `vocab` | `vocab` | `oxixml-vocab` | Constants for RDF, RDFS, OWL, SKOS, Dublin Core, and a dozen other vocabularies. |
//!
//! ## SPARQL
//!
//! | Feature | Module | Crate | What it gives you |
//! |---|---|---|---|
//! | `sparql-results` | `sparql_results` | `oxixml-sparql-results` | The SPARQL 1.1 Query Results XML/JSON/CSV/TSV formats. |
//! | `sparql-syntax` | `sparql_syntax` | `oxixml-sparql-syntax` | SPARQL 1.1/1.2 Query and Update grammars. |
//!
//! ## Cross-cutting modifiers
//!
//! These pull in no crate of their own: each forwards a wrapped crate's own Cargo feature, and so
//! only takes effect on whichever of the crates above are already enabled. Combine them freely,
//! e.g. `features = ["turtle", "rdf-12"]` or `features = ["xml", "serialize"]`.
//!
//! | Feature | Applies to | What it gives you |
//! |---|---|---|
//! | `rdf-12` | `rdf`, `turtle`, `canon`, `rdfio`, `sparql-results` | RDF 1.2 / RDF-star: triple terms and base-direction language strings. |
//! | `rdfc-10` | `rdf` | RDFC-1.0 inside the data model: `model::dataset::CanonicalizationAlgorithm::Rdfc10`. |
//! | `sparql-12` | `sparql-syntax` | SPARQL 1.2: the `VERSION` declaration and triple-term/reifier syntax. |
//! | `serialize` | `xml`, `quickxml-compat` | serde (de)serialization of XML through `xml::de` and `xml::se`. |
//! | `serde-types` | `xml`, `quickxml-compat` | The ready-made serde adapters in `xml::serde_helpers`. |
//! | `async-tokio` | `xml`, `quickxml-compat` | `tokio`-based asynchronous reading and writing. |
//! | `escape-html` | `xml`, `quickxml-compat` | Recognize the full set of HTML5 character references when unescaping. |
//! | `overlapped-lists` | `xml`, `quickxml-compat` | Buffer elements that interrupt a sequence being deserialized. |
//! | `encoding` | `xml`, `quickxml-compat` | Non-UTF-8 documents (on top of exposing the `encoding` module itself). |
//!
//! `full` enables every feature above at once, modifiers included.
//!
//! # Examples
//!
//! Each example below only runs when its feature is enabled (`cargo test --all-features` runs
//! all of them); the rest of this section is unaffected by which features happen to be on.
//!
//! ## Parse XML with the reader (`xml`)
//!
//! ```
//! # #[cfg(feature = "xml")] fn xml_reader_example() -> Result<(), Box<dyn std::error::Error>> {
//! use oxixml::xml::events::Event;
//! use oxixml::xml::reader::Reader;
//!
//! let mut reader = Reader::from_str("<catalog><book id=\"b1\"/></catalog>");
//! reader.config_mut().trim_text(true);
//!
//! let mut saw_book = false;
//! loop {
//!     match reader.read_event()? {
//!         Event::Empty(start) => saw_book = start.name().as_ref() == b"book",
//!         Event::Eof => break,
//!         _ => {}
//!     }
//! }
//! assert!(saw_book);
//! # Ok(())
//! # }
//! # #[cfg(feature = "xml")]
//! # xml_reader_example().expect("xml reader example");
//! ```
//!
//! ## Build and serialize a DOM (`dom`)
//!
//! ```
//! # #[cfg(feature = "dom")] fn dom_example() -> Result<(), Box<dyn std::error::Error>> {
//! use oxixml::dom::{SerializeOptions, build};
//!
//! let mut doc = build::parse_str(
//!     r#"<library><book id="b1"><title>Kokoro</title></book></library>"#,
//! )?;
//! let library = doc.document_element().expect("a document element");
//!
//! let book = doc.create_element("book")?;
//! doc.set_attribute(book, "id", "b2")?;
//! doc.append_child(library, book)?;
//!
//! let xml = doc.to_xml_with(&SerializeOptions::pretty());
//! assert!(xml.contains("<book id=\"b2\"/>"));
//!
//! // A round-trip through the serializer preserves the tree.
//! let reparsed = build::parse_str(&doc.to_xml())?;
//! assert!(doc.structural_eq(&reparsed), "{:?}", doc.structural_diff(&reparsed));
//! # Ok(())
//! # }
//! # #[cfg(feature = "dom")]
//! # dom_example().expect("dom example");
//! ```
//!
//! ## Lint an SVG document (`svg`)
//!
//! ```
//! # #[cfg(feature = "svg")] fn svg_example() -> Result<(), Box<dyn std::error::Error>> {
//! use oxixml::svg::lint::diagnostic::Severity;
//! use oxixml::svg::lint::rules::Linter;
//! use oxixml::svg::tree::document::SvgDocument;
//!
//! let doc = SvgDocument::parse_str(
//!     r#"<svg xmlns="http://www.w3.org/2000/svg" width="100" height="100">
//!          <rect width="-1" height="10"/>
//!        </svg>"#,
//! )?;
//! assert_eq!(doc.svg_element().local_name(), "svg");
//!
//! let diagnostics = Linter::new().lint(doc.document());
//! assert!(diagnostics.iter().any(|d| d.severity() == Severity::Error));
//! # Ok(())
//! # }
//! # #[cfg(feature = "svg")]
//! # svg_example().expect("svg example");
//! ```
//!
//! ## Evaluate an XPath (`xpath`)
//!
//! ```
//! # #[cfg(feature = "xpath")] fn xpath_example() -> Result<(), Box<dyn std::error::Error>> {
//! use oxixml::infoset::Tree;
//! use oxixml::xpath::{StaticContext, XPath, XPathVersion};
//!
//! let tree = Tree::parse_str(
//!     r#"<catalog>
//!          <book year="1965"><title>Dune</title><price>9.99</price></book>
//!          <book year="1815"><title>Emma</title><price>5.50</price></book>
//!        </catalog>"#,
//! )?;
//! let root = tree.root();
//!
//! let old_titles = XPath::compile(
//!     "//book[@year < 1900]/title/string()",
//!     XPathVersion::XPath31,
//!     StaticContext::new(),
//! )?;
//! assert_eq!(old_titles.evaluate_to_string(root.clone())?, "Emma");
//!
//! let total = XPath::compile(
//!     "sum(for $b in //book return xs:decimal($b/price))",
//!     XPathVersion::XPath31,
//!     StaticContext::new(),
//! )?;
//! assert_eq!(total.evaluate_to_string(root)?, "15.49");
//! # Ok(())
//! # }
//! # #[cfg(feature = "xpath")]
//! # xpath_example().expect("xpath example");
//! ```
//!
//! ## Validate against XSD (`schema`)
//!
//! ```
//! # #[cfg(feature = "schema")] fn schema_example() -> Result<(), Box<dyn std::error::Error>> {
//! use oxixml::schema::SchemaSet;
//!
//! let mut set = SchemaSet::new();
//! set.add_document(
//!     None,
//!     r#"<xs:schema xmlns:xs="http://www.w3.org/2001/XMLSchema">
//!          <xs:element name="item">
//!            <xs:complexType>
//!              <xs:attribute name="sku" use="required">
//!                <xs:simpleType>
//!                  <xs:restriction base="xs:string">
//!                    <xs:pattern value="[A-Z]{2}-[0-9]{4}"/>
//!                  </xs:restriction>
//!                </xs:simpleType>
//!              </xs:attribute>
//!            </xs:complexType>
//!          </xs:element>
//!        </xs:schema>"#,
//! )?;
//! let schema = set.compile()?;
//!
//! assert!(schema.validate_str(r#"<item sku="AB-1234"/>"#).valid);
//!
//! let bad = schema.validate_str(r#"<item sku="oops"/>"#);
//! assert_eq!(bad.first_error().map(|error| error.code()), Some("cvc-pattern-valid"));
//! # Ok(())
//! # }
//! # #[cfg(feature = "schema")]
//! # schema_example().expect("schema example");
//! ```
//!
//! ## Transform with XSLT (`xslt`)
//!
//! ```
//! # #[cfg(feature = "xslt")] fn xslt_example() -> Result<(), Box<dyn std::error::Error>> {
//! use oxixml::infoset::Tree;
//! use oxixml::xslt::Stylesheet;
//!
//! let stylesheet = Stylesheet::compile_str(
//!     r#"<xsl:stylesheet version="1.0" xmlns:xsl="http://www.w3.org/1999/XSL/Transform">
//!          <xsl:output method="xml" omit-xml-declaration="yes"/>
//!          <xsl:template match="/catalog"><titles count="{count(book)}"><xsl:apply-templates
//!              select="book"><xsl:sort select="@year" data-type="number"/>
//!            </xsl:apply-templates></titles></xsl:template>
//!          <xsl:template match="book"><title year="{@year}"><xsl:value-of
//!            select="title"/></title></xsl:template>
//!        </xsl:stylesheet>"#,
//! )?;
//!
//! let source = Tree::parse_str(
//!     r#"<catalog>
//!          <book year="1965"><title>Dune</title></book>
//!          <book year="1815"><title>Emma</title></book>
//!        </catalog>"#,
//! )?;
//!
//! let result = stylesheet.transform(&source.root())?;
//! assert_eq!(
//!     result.as_str(),
//!     r#"<titles count="2"><title year="1815">Emma</title><title year="1965">Dune</title></titles>"#
//! );
//! # Ok(())
//! # }
//! # #[cfg(feature = "xslt")]
//! # xslt_example().expect("xslt example");
//! ```
//!
//! ## Parse Turtle (`turtle`)
//!
//! ```
//! # #[cfg(feature = "turtle")] fn turtle_example() -> Result<(), Box<dyn std::error::Error>> {
//! use oxixml::model::NamedNode;
//! use oxixml::model::vocab::rdf;
//! use oxixml::turtle::TurtleParser;
//!
//! let data = br#"@base <http://example.com/> .
//! @prefix schema: <http://schema.org/> .
//! <foo> a schema:Person ;
//!     schema:name "Foo" ."#;
//!
//! let schema_person = NamedNode::new("http://schema.org/Person")?;
//! let mut count = 0;
//! for triple in TurtleParser::new().for_slice(data) {
//!     let triple = triple?;
//!     if triple.predicate == rdf::TYPE {
//!         assert!(triple.object == schema_person);
//!     }
//!     count += 1;
//! }
//! assert_eq!(count, 2);
//! # Ok(())
//! # }
//! # #[cfg(feature = "turtle")]
//! # turtle_example().expect("turtle example");
//! ```
//!
//! ## Canonicalize a dataset (`canon`)
//!
//! ```
//! # #[cfg(feature = "canon")] fn canon_example() -> Result<(), Box<dyn std::error::Error>> {
//! use oxixml::canon::canonicalize;
//! use oxixml::model::{BlankNode, Dataset, GraphNameRef, NamedNode, QuadRef};
//!
//! let p = NamedNode::new("http://example.com/p")?;
//! let q = NamedNode::new("http://example.com/q")?;
//!
//! // Two blank nodes, distinguishable only by structure.
//! let mut dataset = Dataset::new();
//! let a = BlankNode::default();
//! let b = BlankNode::default();
//! dataset.insert(QuadRef::new(&p, &q, &a, GraphNameRef::DefaultGraph));
//! dataset.insert(QuadRef::new(&a, &p, &b, GraphNameRef::DefaultGraph));
//!
//! let result = canonicalize(&dataset)?;
//!
//! // The blank nodes are relabeled to stable `c14n` identifiers.
//! assert_eq!(result.issued_map().len(), 2);
//! assert!(result.canonical_n_quads().contains("_:c14n0"));
//! # Ok(())
//! # }
//! # #[cfg(feature = "canon")]
//! # canon_example().expect("canon example");
//! ```
//!
//! ## Convert RDF formats (`rdfio`)
//!
//! ```
//! # #[cfg(feature = "rdfio")] fn rdf_io_example() -> Result<(), Box<dyn std::error::Error>> {
//! use oxixml::io::{RdfFormat, RdfParser, RdfSerializer};
//!
//! // Detect the format from a file extension, then parse.
//! let format = RdfFormat::from_extension("ttl").expect("known extension");
//! let turtle = r#"@prefix schema: <http://schema.org/> .
//! _:alice a schema:Person ;
//!     schema:name "Alice" ;
//!     schema:knows _:bob .
//! _:bob a schema:Person ;
//!     schema:name "Bob" ."#;
//!
//! let quads = RdfParser::from_format(format)
//!     .rename_blank_nodes()
//!     .for_reader(turtle.as_bytes())
//!     .collect::<Result<Vec<_>, _>>()?;
//! assert_eq!(quads.len(), 5);
//!
//! // Re-serialize the same quads as N-Quads (a dataset format, so every quad round-trips).
//! let mut serializer = RdfSerializer::from_format(RdfFormat::NQuads).for_writer(Vec::new());
//! for quad in &quads {
//!     serializer.serialize_quad(quad)?;
//! }
//! let nquads = String::from_utf8(serializer.finish()?)?;
//! assert_eq!(nquads.lines().count(), 5);
//! # Ok(())
//! # }
//! # #[cfg(feature = "rdfio")]
//! # rdf_io_example().expect("rdf io example");
//! ```
//!
//! # `docs.rs`
//!
//! The published documentation is built with every feature enabled; each re-export below is
//! badged with the feature (or features) that must be enabled to reach it.

#![cfg_attr(docsrs, feature(doc_cfg))]
#![deny(missing_docs)]
#![forbid(unsafe_code)]

/// Canonical XML canonicalization (`oxixml-c14n`): Canonical XML 1.0/1.1/2.0 and Exclusive
/// XML Canonicalization, built on [`dom`].
#[cfg(feature = "c14n")]
#[cfg_attr(docsrs, doc(cfg(feature = "c14n")))]
pub use oxixml_c14n as c14n;

/// RDF Dataset Canonicalization (`oxixml-canon`): RDFC-1.0 / URDNA2015 over [`model`].
#[cfg(feature = "canon")]
#[cfg_attr(docsrs, doc(cfg(feature = "canon")))]
pub use oxixml_canon as canon;

/// OASIS XML Catalogs (`oxixml-catalog`): `public`/`system`/URI resolution and rewriting.
#[cfg(feature = "catalog")]
#[cfg_attr(docsrs, doc(cfg(feature = "catalog")))]
pub use oxixml_catalog as catalog;

/// The in-memory document tree (`oxixml-dom`): a DOM Level 3 Core document over an arena,
/// built on [`xml`].
#[cfg(feature = "dom")]
#[cfg_attr(docsrs, doc(cfg(feature = "dom")))]
pub use oxixml_dom as dom;

/// XML Signature (`oxixml-dsig`): XML-DSig generation and verification, built on [`c14n`].
#[cfg(feature = "dsig")]
#[cfg_attr(docsrs, doc(cfg(feature = "dsig")))]
pub use oxixml_dsig as dsig;

/// DTD processing (`oxixml-dtd`): parsing, content-model compilation, and instance
/// validation, built on [`xml`].
#[cfg(feature = "dtd")]
#[cfg_attr(docsrs, doc(cfg(feature = "dtd")))]
pub use oxixml_dtd as dtd;

/// XML Encryption (`oxixml-enc`): XML-ENC element/content encryption and decryption, built
/// on [`dom`].
#[cfg(feature = "enc")]
#[cfg_attr(docsrs, doc(cfg(feature = "enc")))]
pub use oxixml_enc as enc;

/// Character-encoding decoders/encoders (`oxixml-encoding`): UTF-8/16/32, Latin-1,
/// windows-1252, and the legacy Japanese encodings.
#[cfg(feature = "encoding")]
#[cfg_attr(docsrs, doc(cfg(feature = "encoding")))]
pub use oxixml_encoding as encoding;

/// The XPath/XQuery/XSLT Functions & Operators library (`oxixml-fn`), built on [`infoset`].
#[cfg(feature = "functions")]
#[cfg_attr(docsrs, doc(cfg(feature = "functions")))]
pub use oxixml_fn as functions;

/// XSL-FO 1.1 (`oxixml-fo`): a formatting-object tree, property refinement, and area-model
/// primitives, built on [`xml`].
#[cfg(feature = "fo")]
#[cfg_attr(docsrs, doc(cfg(feature = "fo")))]
pub use oxixml_fo as fo;

/// HDT (`oxixml-hdt`): Header-Dictionary-Triples, a compact binary RDF representation over
/// [`model`].
#[cfg(feature = "hdt")]
#[cfg_attr(docsrs, doc(cfg(feature = "hdt")))]
pub use oxixml_hdt as hdt;

/// HTML5 parsing (`oxixml-html`): a tokenizer and tree constructor producing an ordinary
/// [`dom`] document.
#[cfg(feature = "html")]
#[cfg_attr(docsrs, doc(cfg(feature = "html")))]
pub use oxixml_html as html;

/// The XDM tree (`oxixml-infoset`): the XPath/XQuery/XSLT Data Model that `xpath`,
/// `xquery`, `xslt`, and `schema` evaluate over, built on [`xml`].
#[cfg(feature = "infoset")]
#[cfg_attr(docsrs, doc(cfg(feature = "infoset")))]
pub use oxixml_infoset as infoset;

/// Unified RDF format detection, parsing, and serialization (`oxixml-io`) across [`turtle`],
/// [`rdfxml`], [`jsonld`], and [`trix`]. Exposed as the `rdfio` feature (`io` collides with
/// [`std::io`] as a bare glob-import target, but is fine as a module path).
#[cfg(feature = "rdfio")]
#[cfg_attr(docsrs, doc(cfg(feature = "rdfio")))]
pub use oxixml_io as io;

/// RFC 3987 Internationalized Resource Identifiers (`oxixml-iri`): parse, resolve,
/// relativize.
#[cfg(feature = "iri")]
#[cfg_attr(docsrs, doc(cfg(feature = "iri")))]
pub use oxixml_iri as iri;

/// A streaming JSON event parser/serializer (`oxixml-json`), shared by `jsonld` and
/// `sparql_results`.
#[cfg(feature = "json")]
#[cfg_attr(docsrs, doc(cfg(feature = "json")))]
pub use oxixml_json as json;

/// JSON-LD 1.1 (`oxixml-jsonld`): expand, compact, flatten, and the `toRdf`/`fromRdf`
/// algorithms, built on [`model`] and [`json`].
#[cfg(feature = "jsonld")]
#[cfg_attr(docsrs, doc(cfg(feature = "jsonld")))]
pub use oxixml_jsonld as jsonld;

/// RFC 5646 language tags (`oxixml-langtag`): parse, normalize, compare.
#[cfg(feature = "langtag")]
#[cfg_attr(docsrs, doc(cfg(feature = "langtag")))]
pub use oxixml_langtag as langtag;

/// The core RDF/RDF-star data model (`oxixml-model`): `NamedNode`, `BlankNode`, `Literal`,
/// `Triple`, `Quad`, and `Dataset`. Exposed as the `rdf` feature.
#[cfg(feature = "rdf")]
#[cfg_attr(docsrs, doc(cfg(feature = "rdf")))]
pub use oxixml_model as model;

/// XML names (`oxixml-qname`): `NCName`/`QName`, `ExpandedName`, and the namespace-binding
/// stack.
#[cfg(feature = "qname")]
#[cfg_attr(docsrs, doc(cfg(feature = "qname")))]
pub use oxixml_qname as qname;

/// A drop-in `quick-xml` compatibility shim (`oxixml-quickxml-compat`) over [`xml`]. Alias
/// it directly as `quick_xml` in a downstream `Cargo.toml` (`package = "oxixml-quickxml-compat"`)
/// to migrate with a one-line dependency change.
#[cfg(feature = "quickxml-compat")]
#[cfg_attr(docsrs, doc(cfg(feature = "quickxml-compat")))]
pub use oxixml_quickxml_compat as quickxml_compat;

/// RDFa 1.1 Core (`oxixml-rdfa`): extracted from an [`html`] document into [`model`] triples.
#[cfg(feature = "rdfa")]
#[cfg_attr(docsrs, doc(cfg(feature = "rdfa")))]
pub use oxixml_rdfa as rdfa;

/// RDF/XML (`oxixml-rdfxml`), built on [`model`] and [`xml`].
#[cfg(feature = "rdfxml")]
#[cfg_attr(docsrs, doc(cfg(feature = "rdfxml")))]
pub use oxixml_rdfxml as rdfxml;

/// The XML-Schema and XPath regular-expression engine (`oxixml-regex`).
#[cfg(feature = "regex")]
#[cfg_attr(docsrs, doc(cfg(feature = "regex")))]
pub use oxixml_regex as regex;

/// RELAX NG (`oxixml-relaxng`), both the XML and the compact syntax, built on [`dom`].
#[cfg(feature = "relaxng")]
#[cfg_attr(docsrs, doc(cfg(feature = "relaxng")))]
pub use oxixml_relaxng as relaxng;

/// A SAX2-style push parser API (`oxixml-sax`), built on [`xml`].
#[cfg(feature = "sax")]
#[cfg_attr(docsrs, doc(cfg(feature = "sax")))]
pub use oxixml_sax as sax;

/// W3C XML Schema / XSD 1.0 and 1.1 (`oxixml-schema`): assembly and validation, built on
/// [`dom`] and [`xpath`].
#[cfg(feature = "schema")]
#[cfg_attr(docsrs, doc(cfg(feature = "schema")))]
pub use oxixml_schema as schema;

/// ISO Schematron (`oxixml-schematron`): rules compiled and run over [`xpath`].
#[cfg(feature = "schematron")]
#[cfg_attr(docsrs, doc(cfg(feature = "schematron")))]
pub use oxixml_schematron as schematron;

/// The SPARQL 1.1 Query Results XML/JSON/CSV/TSV formats (`oxixml-sparql-results`), built on
/// [`model`], [`xml`], and [`json`].
#[cfg(feature = "sparql-results")]
#[cfg_attr(docsrs, doc(cfg(feature = "sparql-results")))]
pub use oxixml_sparql_results as sparql_results;

/// SPARQL 1.1/1.2 Query and Update grammars (`oxixml-sparql-syntax`), built on [`model`].
#[cfg(feature = "sparql-syntax")]
#[cfg_attr(docsrs, doc(cfg(feature = "sparql-syntax")))]
pub use oxixml_sparql_syntax as sparql_syntax;

/// Zero-dependency substrate (`oxixml-support`): SWAR byte scanning, line/column tracking,
/// FIPS 180-4 SHA/HMAC, and a minimal unsigned bigint.
#[cfg(feature = "support")]
#[cfg_attr(docsrs, doc(cfg(feature = "support")))]
pub use oxixml_support as support;

/// A typed SVG document model, microsyntax toolkit, geometry engine, builder, and linter
/// (`oxixml-svg`), built on [`dom`].
#[cfg(feature = "svg")]
#[cfg_attr(docsrs, doc(cfg(feature = "svg")))]
pub use oxixml_svg as svg;

/// Text codecs shared across the stack (`oxixml-text`): base64, hex, percent-encoding, and
/// ECMAScript/XSD number formatting.
#[cfg(feature = "text")]
#[cfg_attr(docsrs, doc(cfg(feature = "text")))]
pub use oxixml_text as text;

/// TriX (`oxixml-trix`), the XML serialization of an RDF dataset, built on [`model`] and
/// [`xml`].
#[cfg(feature = "trix")]
#[cfg_attr(docsrs, doc(cfg(feature = "trix")))]
pub use oxixml_trix as trix;

/// Turtle, TriG, N-Triples, N-Quads, and N3 (`oxixml-turtle`), built on [`model`].
#[cfg(feature = "turtle")]
#[cfg_attr(docsrs, doc(cfg(feature = "turtle")))]
pub use oxixml_turtle as turtle;

/// Unicode property data, character classes, and normalization (`oxixml-unicode`): blocks,
/// scripts, case folding, NFC/NFD/NFKC/NFKD.
#[cfg(feature = "unicode")]
#[cfg_attr(docsrs, doc(cfg(feature = "unicode")))]
pub use oxixml_unicode as unicode;

/// Constants for RDF, RDFS, OWL, SKOS, Dublin Core, and a dozen other well-known
/// vocabularies (`oxixml-vocab`), built on [`model`].
#[cfg(feature = "vocab")]
#[cfg_attr(docsrs, doc(cfg(feature = "vocab")))]
pub use oxixml_vocab as vocab;

/// XInclude processing (`oxixml-xinclude`) over a [`dom`] tree.
#[cfg(feature = "xinclude")]
#[cfg_attr(docsrs, doc(cfg(feature = "xinclude")))]
pub use oxixml_xinclude as xinclude;

/// The XML 1.0/1.1 pull parser and writer (`oxixml-xml`), quick-xml 0.41-compatible.
#[cfg(feature = "xml")]
#[cfg_attr(docsrs, doc(cfg(feature = "xml")))]
pub use oxixml_xml as xml;

/// XPath 1.0-3.1 (`oxixml-xpath`), evaluated over [`infoset`].
#[cfg(feature = "xpath")]
#[cfg_attr(docsrs, doc(cfg(feature = "xpath")))]
pub use oxixml_xpath as xpath;

/// XQuery 3.1 (`oxixml-xquery`), built on [`xpath`].
#[cfg(feature = "xquery")]
#[cfg_attr(docsrs, doc(cfg(feature = "xquery")))]
pub use oxixml_xquery as xquery;

/// The XML Schema Part 2 built-in datatypes (`oxixml-xsd`): `decimal`, `dateTime`,
/// `duration`, and the rest of the built-in type hierarchy.
#[cfg(feature = "xsd")]
#[cfg_attr(docsrs, doc(cfg(feature = "xsd")))]
pub use oxixml_xsd as xsd;

/// XSLT 1.0, plus the essentials of 2.0/3.0 (`oxixml-xslt`), built on [`xpath`] and [`dom`].
#[cfg(feature = "xslt")]
#[cfg_attr(docsrs, doc(cfg(feature = "xslt")))]
pub use oxixml_xslt as xslt;
