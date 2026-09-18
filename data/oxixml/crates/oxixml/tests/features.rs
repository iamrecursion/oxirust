//! Asserts that every enabled facade feature's re-export resolves to real, working items from
//! the sibling crate it wraps.
//!
//! Each test is gated on exactly the feature it checks, so `cargo test -p oxixml --features X`
//! runs only the subset relevant to `X` (plus whatever `X` pulls in), and
//! `cargo test -p oxixml --all-features` runs the lot. A path that stopped resolving — a stale
//! re-export name, a renamed type, a feature edge that no longer matches the real dependency
//! graph — fails to compile here rather than silently going stale.

// ---------------------------------------------------------------------------------------------
// Foundation
// ---------------------------------------------------------------------------------------------

#[cfg(feature = "support")]
#[test]
fn support_reexport_resolves() {
    assert_eq!(oxixml::support::bytes::memchr(b'x', b"abxcd"), Some(2));
}

#[cfg(feature = "unicode")]
#[test]
fn unicode_reexport_resolves() {
    assert_eq!(oxixml::unicode::case::simple_lowercase('A'), 'a');
    assert!(oxixml::unicode::norm::is_nfc("abc"));
}

#[cfg(feature = "encoding")]
#[test]
fn encoding_reexport_resolves() {
    assert!(oxixml::encoding::Encoding::for_label("utf-8").is_some());
}

#[cfg(feature = "text")]
#[test]
fn text_reexport_resolves() {
    assert_eq!(oxixml::text::hex::encode_lower(&[0xab]), "ab");
}

#[cfg(feature = "regex")]
#[test]
fn regex_reexport_resolves() {
    let re = oxixml::regex::Regex::xsd("[a-z]+").expect("valid XSD pattern");
    assert!(re.is_match("abc"));
    assert!(!re.is_match("ABC"));
}

#[cfg(feature = "iri")]
#[test]
fn iri_reexport_resolves() {
    let iri = oxixml::iri::Iri::parse("http://example.com/path").expect("valid IRI");
    assert_eq!(iri.scheme(), "http");
}

#[cfg(feature = "langtag")]
#[test]
fn langtag_reexport_resolves() {
    let tag = oxixml::langtag::LanguageTag::parse("en-US").expect("valid language tag");
    assert_eq!(tag.primary_language(), "en");
}

#[cfg(feature = "xsd")]
#[test]
fn xsd_reexport_resolves() {
    use std::str::FromStr;
    let decimal = oxixml::xsd::Decimal::from_str("1.50").expect("valid xsd:decimal");
    assert_eq!(decimal.to_string(), "1.5");
}

#[cfg(feature = "qname")]
#[test]
fn qname_reexport_resolves() {
    assert!(oxixml::qname::is_ncname("book"));
    assert!(!oxixml::qname::is_ncname("book:title"));
    let name = oxixml::qname::QName::parse("book:title").expect("valid QName");
    assert_eq!(name.local_part(), "title");
}

// ---------------------------------------------------------------------------------------------
// XML core
// ---------------------------------------------------------------------------------------------

#[cfg(feature = "xml")]
#[test]
fn xml_reexport_resolves() {
    let mut reader = oxixml::xml::reader::Reader::from_str("<a/>");
    let event = reader.read_event().expect("a single empty element");
    assert!(matches!(event, oxixml::xml::events::Event::Empty(_)));
}

#[cfg(feature = "quickxml-compat")]
#[test]
fn quickxml_compat_reexport_resolves() {
    let mut reader = oxixml::quickxml_compat::reader::Reader::from_str("<a/>");
    let event = reader.read_event().expect("a single empty element");
    assert!(matches!(
        event,
        oxixml::quickxml_compat::events::Event::Empty(_)
    ));
}

// ---------------------------------------------------------------------------------------------
// Trees
// ---------------------------------------------------------------------------------------------

#[cfg(feature = "dom")]
#[test]
fn dom_reexport_resolves() {
    let doc = oxixml::dom::build::parse_str("<a/>").expect("well-formed document");
    assert!(doc.document_element().is_some());
}

#[cfg(feature = "sax")]
#[test]
fn sax_reexport_resolves() {
    let mut handler = oxixml::sax::NullContentHandler;
    let mut handlers = oxixml::sax::Handlers::new().with_content(&mut handler);
    oxixml::sax::XMLReader::new()
        .parse_str("<a/>", &mut handlers)
        .expect("well-formed document");
}

#[cfg(feature = "infoset")]
#[test]
fn infoset_reexport_resolves() {
    use oxixml::infoset::XdmNode;

    let tree = oxixml::infoset::Tree::parse_str("<a/>").expect("well-formed document");
    assert_eq!(tree.root().node_kind(), oxixml::infoset::NodeKind::Document);
}

#[cfg(feature = "html")]
#[test]
fn html_reexport_resolves() {
    let parsed = oxixml::html::parse_document("<!DOCTYPE html><p>hi");
    assert_eq!(parsed.quirks_mode(), oxixml::html::QuirksMode::NoQuirks);
}

// ---------------------------------------------------------------------------------------------
// Validation
// ---------------------------------------------------------------------------------------------

#[cfg(feature = "dtd")]
#[test]
fn dtd_reexport_resolves() {
    let report = oxixml::dtd::validate_str("<!DOCTYPE a [<!ELEMENT a (#PCDATA)>]><a/>")
        .expect("a valid document");
    assert!(report.is_valid());
}

#[cfg(feature = "schema")]
#[test]
fn schema_reexport_resolves() {
    let set = oxixml::schema::SchemaSet::new();
    let _ = set;
}

#[cfg(feature = "relaxng")]
#[test]
fn relaxng_reexport_resolves() {
    let schema = oxixml::relaxng::RelaxNg::from_compact("element a { text }")
        .expect("valid compact-syntax schema");
    assert!(schema.validate("<a>hi</a>").is_ok());
}

#[cfg(feature = "schematron")]
#[test]
fn schematron_reexport_resolves() {
    let schema = r#"<schema xmlns="http://purl.oclc.org/dsdl/schematron">
        <pattern><rule context="a"><assert test="true()">ok</assert></rule></pattern>
    </schema>"#;
    let compiled = oxixml::schematron::Schematron::compile(schema, None).expect("valid schema");
    let _ = compiled;
}

#[cfg(feature = "catalog")]
#[test]
fn catalog_reexport_resolves() {
    let xml = r#"<catalog xmlns="urn:oasis:names:tc:entity:xmlns:xml:catalog">
        <public publicId="-//Example//DTD Example//EN" uri="example.dtd"/>
    </catalog>"#;
    let catalog =
        oxixml::catalog::Catalog::parse(xml, "file:///etc/xml/catalog.xml").expect("valid catalog");
    assert_eq!(
        catalog
            .resolve_public("-//Example//DTD Example//EN", None)
            .as_deref(),
        Some("file:///etc/xml/example.dtd")
    );
}

#[cfg(feature = "xinclude")]
#[test]
fn xinclude_reexport_resolves() {
    let mut doc = oxixml::dom::build::parse_str(
        r#"<doc xmlns:xi="http://www.w3.org/2001/XInclude"><xi:include href="https://example.com/inc.xml"/></doc>"#,
    )
    .expect("well-formed document");
    let loader =
        oxixml::xinclude::MapLoader::new().with_text("https://example.com/inc.xml", "<p>hi</p>");
    oxixml::xinclude::process_document(&mut doc, "https://example.com/base.xml", &loader)
        .expect("resolvable include");
    let root = doc.document_element().expect("a document element");
    assert!(doc.get_elements_by_tag_name(root, "p").next().is_some());
}

// ---------------------------------------------------------------------------------------------
// Query & transform
// ---------------------------------------------------------------------------------------------

#[cfg(feature = "functions")]
#[test]
fn functions_reexport_resolves() {
    assert_eq!(oxixml::functions::strings::upper_case("abCd0"), "ABCD0");
}

#[cfg(feature = "xpath")]
#[test]
fn xpath_reexport_resolves() {
    let tree = oxixml::infoset::Tree::parse_str("<a><b/></a>").expect("well-formed document");
    let compiled = oxixml::xpath::XPath::compile(
        "count(//b)",
        oxixml::xpath::XPathVersion::XPath31,
        oxixml::xpath::StaticContext::new(),
    )
    .expect("valid XPath expression");
    assert_eq!(
        compiled.evaluate_to_number(tree.root()).expect("a number"),
        1.0
    );
}

#[cfg(feature = "xquery")]
#[test]
fn xquery_reexport_resolves() {
    let compiled = oxixml::xquery::XQuery::compile_default("1 + 1").expect("valid XQuery");
    let _ = compiled;
}

#[cfg(feature = "xslt")]
#[test]
fn xslt_reexport_resolves() {
    let stylesheet = oxixml::xslt::Stylesheet::compile_str(
        r#"<xsl:stylesheet version="1.0" xmlns:xsl="http://www.w3.org/1999/XSL/Transform">
             <xsl:output omit-xml-declaration="yes"/>
             <xsl:template match="/"><out/></xsl:template>
           </xsl:stylesheet>"#,
    )
    .expect("valid stylesheet");
    let source = oxixml::infoset::Tree::parse_str("<in/>").expect("well-formed document");
    assert_eq!(
        stylesheet
            .transform_to_string(&source.root())
            .expect("a transform result"),
        "<out/>"
    );
}

// ---------------------------------------------------------------------------------------------
// Security & format
// ---------------------------------------------------------------------------------------------

#[cfg(feature = "c14n")]
#[test]
fn c14n_reexport_resolves() {
    let doc = oxixml::dom::build::parse_str("<a b=\"1\" a=\"2\"/>").expect("well-formed document");
    let canonical = oxixml::c14n::c14n_1_0(&doc, &oxixml::c14n::NodeSet::document(&doc), false)
        .expect("canonicalizable document");
    assert!(canonical.starts_with(b"<a a=\"2\" b=\"1\">"));
}

#[cfg(feature = "dsig")]
#[test]
fn dsig_reexport_resolves() {
    assert!(oxixml::dsig::DSIG_NAMESPACE.starts_with("http://www.w3.org/2000/09/xmldsig"));
}

#[cfg(feature = "enc")]
#[test]
fn enc_reexport_resolves() {
    let options = oxixml::enc::EncryptOptions::new();
    let _ = options;
}

#[cfg(feature = "fo")]
#[test]
fn fo_reexport_resolves() {
    assert!(oxixml::fo::FoDocument::parse("not xml").is_err());
}

// ---------------------------------------------------------------------------------------------
// RDF
// ---------------------------------------------------------------------------------------------

#[cfg(feature = "rdf")]
#[test]
fn rdf_reexport_resolves() {
    let node = oxixml::model::NamedNode::new("http://example.com/").expect("valid IRI");
    assert_eq!(node.as_str(), "http://example.com/");
}

#[cfg(feature = "json")]
#[test]
fn json_reexport_resolves() {
    let mut parser = oxixml::json::SliceJsonParser::new(b"42");
    assert_eq!(
        parser.parse_next().expect("a JSON number"),
        oxixml::json::JsonEvent::Number("42".into())
    );
}

#[cfg(feature = "turtle")]
#[test]
fn turtle_reexport_resolves() {
    let triples: Vec<_> = oxixml::turtle::TurtleParser::new()
        .for_slice(b"<http://example.com/s> <http://example.com/p> <http://example.com/o> .")
        .collect::<Result<_, _>>()
        .expect("valid turtle");
    let triples: Vec<oxixml::model::Triple> = triples;
    assert_eq!(triples.len(), 1);
}

#[cfg(feature = "rdfxml")]
#[test]
fn rdfxml_reexport_resolves() {
    let file = r#"<?xml version="1.0"?>
        <rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#" xmlns:s="http://schema.org/">
          <s:Person rdf:about="http://example.com/foo" s:name="Foo" />
        </rdf:RDF>"#;
    let triples: Vec<_> = oxixml::rdfxml::RdfXmlParser::new()
        .for_slice(file)
        .collect::<Result<Vec<_>, _>>()
        .expect("valid RDF/XML");
    assert_eq!(triples.len(), 2);
}

#[cfg(feature = "jsonld")]
#[test]
fn jsonld_reexport_resolves() {
    fn _type_resolves(_options: oxixml::jsonld::JsonLdOptions<'_>) {}
    let options = oxixml::jsonld::JsonLdOptions::default();
    _type_resolves(options);
}

#[cfg(feature = "rdfa")]
#[test]
fn rdfa_reexport_resolves() {
    let html = r##"<!DOCTYPE html>
        <html prefix="schema: http://schema.org/">
          <body><div about="#me" typeof="schema:Person">
            <span property="schema:name">Alice</span>
          </div></body>
        </html>"##;
    let triples: Vec<_> = oxixml::rdfa::RdfaParser::new()
        .with_base_iri("https://example.com/")
        .expect("valid base IRI")
        .for_slice(html)
        .collect::<Result<Vec<_>, _>>()
        .expect("valid RDFa");
    assert_eq!(triples.len(), 2);
}

#[cfg(feature = "trix")]
#[test]
fn trix_reexport_resolves() {
    let quad = oxixml::model::Quad::new(
        oxixml::model::NamedNode::new("http://example.com/s").expect("valid IRI"),
        oxixml::model::NamedNode::new("http://example.com/p").expect("valid IRI"),
        oxixml::model::Literal::new_simple_literal("o"),
        oxixml::model::NamedNode::new("http://example.com/g").expect("valid IRI"),
    );
    let mut serializer = oxixml::trix::TriXSerializer::new().for_writer(Vec::new());
    serializer.serialize_quad(&quad).expect("serializable quad");
    let bytes = serializer.finish().expect("finished output");

    let parsed: Vec<_> = oxixml::trix::TriXParser::new()
        .for_slice(&bytes)
        .collect::<Result<Vec<_>, _>>()
        .expect("valid TriX");
    assert_eq!(parsed, vec![quad]);
}

#[cfg(feature = "hdt")]
#[test]
fn hdt_reexport_resolves() {
    let ex = |s: &str| {
        oxixml::model::NamedNode::new(format!("http://example.com/{s}")).expect("valid IRI")
    };
    let triples = vec![oxixml::model::Triple::new(ex("a"), ex("knows"), ex("b"))];

    let mut bytes = Vec::new();
    oxixml::hdt::HdtWriter::new()
        .write(&mut bytes, triples)
        .expect("writable HDT");

    let hdt = oxixml::hdt::Hdt::from_bytes(&bytes).expect("readable HDT");
    assert_eq!(hdt.len(), 1);
}

#[cfg(feature = "canon")]
#[test]
fn canon_reexport_resolves() {
    let mut dataset = oxixml::model::Dataset::new();
    dataset.insert(oxixml::model::QuadRef::new(
        &oxixml::model::NamedNode::new("http://example.com/s").expect("valid IRI"),
        &oxixml::model::NamedNode::new("http://example.com/p").expect("valid IRI"),
        &oxixml::model::NamedNode::new("http://example.com/o").expect("valid IRI"),
        oxixml::model::GraphNameRef::DefaultGraph,
    ));
    let result = oxixml::canon::canonicalize(&dataset).expect("canonicalizable dataset");
    assert!(result.canonical_n_quads().ends_with(" .\n"));
}

#[cfg(feature = "rdfio")]
#[test]
fn rdfio_reexport_resolves() {
    let format = oxixml::io::RdfFormat::from_extension("ttl").expect("known extension");
    assert_eq!(format, oxixml::io::RdfFormat::Turtle);
}

#[cfg(feature = "vocab")]
#[test]
fn vocab_reexport_resolves() {
    assert!(oxixml::vocab::rdf::NAMESPACE.ends_with('#'));
    assert_eq!(
        oxixml::vocab::rdf::TYPE.as_str(),
        "http://www.w3.org/1999/02/22-rdf-syntax-ns#type"
    );
}

// ---------------------------------------------------------------------------------------------
// SPARQL
// ---------------------------------------------------------------------------------------------

#[cfg(feature = "sparql-results")]
#[test]
fn sparql_results_reexport_resolves() {
    assert_eq!(
        oxixml::sparql_results::QueryResultsFormat::from_extension("srj"),
        Some(oxixml::sparql_results::QueryResultsFormat::Json)
    );
}

#[cfg(feature = "sparql-syntax")]
#[test]
fn sparql_syntax_reexport_resolves() {
    let query = oxixml::sparql_syntax::SparqlParser::new()
        .parse_query("SELECT * WHERE { ?s ?p ?o }")
        .expect("valid SPARQL query");
    let _ = query;
}

// ---------------------------------------------------------------------------------------------
// Cross-cutting modifiers
//
// Regression coverage for the facade forwarding a wrapped crate's *own* Cargo feature. Before
// these existed, `--features "turtle,rdf-12"` and `--features "xml,serialize"` failed outright
// with "the package 'oxixml' does not contain this feature", and no combination — `full`
// included — could reach RDF-star, RDFC-1.0, SPARQL 1.2, or any of `oxixml-xml`'s own features.
// ---------------------------------------------------------------------------------------------

#[cfg(all(feature = "rdf-12", feature = "turtle"))]
#[test]
fn rdf_12_forwards_to_turtle() {
    // A triple term (`<<( … )>>`) parses only when `oxixml-turtle/rdf-12` is on.
    let triples: Vec<oxixml::model::Triple> = oxixml::turtle::TurtleParser::new()
        .for_slice(
            b"<http://example.com/s> <http://example.com/p> \
              <<( <http://example.com/a> <http://example.com/b> <http://example.com/c> )>> .",
        )
        .collect::<Result<_, _>>()
        .expect("valid RDF 1.2 turtle");
    assert_eq!(triples.len(), 1);
}

#[cfg(all(feature = "rdf-12", feature = "rdf"))]
#[test]
fn rdf_12_forwards_to_model() {
    assert_eq!(
        oxixml::model::vocab::rdf::REIFIES.as_str(),
        "http://www.w3.org/1999/02/22-rdf-syntax-ns#reifies"
    );
}

#[cfg(all(feature = "rdfc-10", feature = "rdf"))]
#[test]
fn rdfc_10_forwards_to_model() {
    use oxixml::model::dataset::{CanonicalizationAlgorithm, CanonicalizationHashAlgorithm};

    let mut dataset = oxixml::model::Dataset::new();
    let subject = oxixml::model::BlankNode::new("b0").expect("valid blank node id");
    dataset.insert(oxixml::model::QuadRef::new(
        &subject,
        &oxixml::model::NamedNode::new("http://example.com/p").expect("valid IRI"),
        &oxixml::model::NamedNode::new("http://example.com/o").expect("valid IRI"),
        oxixml::model::GraphNameRef::DefaultGraph,
    ));
    dataset.canonicalize(CanonicalizationAlgorithm::Rdfc10 {
        hash_algorithm: CanonicalizationHashAlgorithm::Sha256,
    });
    assert_eq!(dataset.len(), 1);
}

#[cfg(all(feature = "sparql-12", feature = "sparql-syntax"))]
#[test]
fn sparql_12_forwards_to_sparql_syntax() {
    // The `VERSION` prologue declaration is SPARQL 1.2-only.
    let query = oxixml::sparql_syntax::SparqlParser::new()
        .parse_query("VERSION \"1.2\" SELECT * WHERE { ?s ?p ?o }")
        .expect("valid SPARQL 1.2 query");
    let _ = query;
}

#[cfg(feature = "serialize")]
#[test]
fn serialize_forwards_to_xml() {
    #[derive(serde::Serialize, serde::Deserialize, Debug, PartialEq)]
    struct Root {
        name: String,
    }

    let parsed: Root =
        oxixml::xml::de::from_str("<Root><name>hi</name></Root>").expect("deserializable XML");
    assert_eq!(
        parsed,
        Root {
            name: "hi".to_owned()
        }
    );
    assert_eq!(
        oxixml::xml::se::to_string(&parsed).expect("serializable value"),
        "<Root><name>hi</name></Root>"
    );
}

#[cfg(all(feature = "serde-types", feature = "serialize"))]
#[test]
fn serde_types_forwards_to_xml() {
    // `xml::serde_helpers` exists only under `oxixml-xml/serde-types`.
    #[derive(serde::Deserialize, Debug, PartialEq)]
    struct Root {
        #[serde(deserialize_with = "oxixml::xml::serde_helpers::text_content::deserialize")]
        child: String,
    }

    let parsed: Root =
        oxixml::xml::de::from_str("<Root><child>hi</child></Root>").expect("deserializable XML");
    assert_eq!(
        parsed,
        Root {
            child: "hi".to_owned()
        }
    );
}

#[cfg(all(feature = "overlapped-lists", feature = "serialize"))]
#[test]
fn overlapped_lists_forwards_to_xml() {
    // Without `oxixml-xml/overlapped-lists`, a sequence interrupted by another field is cut
    // short at the interruption; with it, the intervening elements are buffered and the
    // sequence is completed.
    #[derive(serde::Deserialize, Debug, PartialEq)]
    struct Root {
        item: Vec<u32>,
        other: String,
    }

    let parsed: Root =
        oxixml::xml::de::from_str("<Root><item>1</item><other>x</other><item>2</item></Root>")
            .expect("deserializable XML");
    assert_eq!(
        parsed,
        Root {
            item: vec![1, 2],
            other: "x".to_owned(),
        }
    );
}

#[cfg(feature = "async-tokio")]
#[test]
fn async_tokio_forwards_to_xml() {
    // `Reader::read_event_into_async` exists only under `oxixml-xml/async-tokio`.
    let runtime = tokio::runtime::Builder::new_current_thread()
        .build()
        .expect("a tokio runtime");
    runtime.block_on(async {
        let mut reader = oxixml::xml::Reader::from_reader(b"<a/>".as_slice());
        let mut buf = Vec::new();
        let event = reader
            .read_event_into_async(&mut buf)
            .await
            .expect("a single empty element");
        assert!(matches!(event, oxixml::xml::events::Event::Empty(_)));
    });
}

#[cfg(all(feature = "async-tokio", feature = "turtle"))]
#[test]
fn async_tokio_forwards_to_turtle() {
    // `for_tokio_async_reader` / `for_tokio_async_writer` exist only under
    // `oxixml-turtle/async-tokio`.
    let runtime = tokio::runtime::Builder::new_current_thread()
        .build()
        .expect("a tokio runtime");
    runtime.block_on(async {
        let file = "<http://example.com/s> <http://example.com/p> <http://example.com/o> .";
        let mut parser =
            oxixml::turtle::NTriplesParser::new().for_tokio_async_reader(file.as_bytes());
        let triple = parser
            .next()
            .await
            .expect("a triple")
            .expect("a valid triple");

        let mut serializer =
            oxixml::turtle::NTriplesSerializer::new().for_tokio_async_writer(Vec::new());
        serializer
            .serialize_triple(&triple)
            .await
            .expect("a successful write");
        assert_eq!(serializer.finish(), format!("{file}\n").into_bytes());
    });
}

#[cfg(all(feature = "async-tokio", feature = "rdfxml"))]
#[test]
fn async_tokio_forwards_to_rdfxml() {
    let runtime = tokio::runtime::Builder::new_current_thread()
        .build()
        .expect("a tokio runtime");
    runtime.block_on(async {
        let file = r#"<?xml version="1.0"?>
<rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">
 <rdf:Description rdf:about="http://example.com/s">
   <rdf:type rdf:resource="http://schema.org/Person" />
 </rdf:Description>
</rdf:RDF>"#;
        let mut parser =
            oxixml::rdfxml::RdfXmlParser::new().for_tokio_async_reader(file.as_bytes());
        assert!(parser.next().await.expect("a triple").is_ok());
    });
}

#[cfg(all(feature = "async-tokio", feature = "jsonld"))]
#[test]
fn async_tokio_forwards_to_jsonld() {
    let runtime = tokio::runtime::Builder::new_current_thread()
        .build()
        .expect("a tokio runtime");
    runtime.block_on(async {
        let file = r#"{"@id": "http://example.com/s", "@type": "http://schema.org/Person"}"#;
        let mut parser =
            oxixml::jsonld::JsonLdParser::new().for_tokio_async_reader(file.as_bytes());
        assert!(parser.next().await.expect("a quad").is_ok());
    });
}

#[cfg(all(feature = "async-tokio", feature = "trix"))]
#[test]
fn async_tokio_forwards_to_trix() {
    let runtime = tokio::runtime::Builder::new_current_thread()
        .build()
        .expect("a tokio runtime");
    runtime.block_on(async {
        let mut serializer = oxixml::trix::TriXSerializer::new().for_tokio_async_writer(Vec::new());
        serializer
            .serialize_quad(&oxixml::model::Quad::new(
                oxixml::model::NamedNode::new_unchecked("http://example.com/s"),
                oxixml::model::NamedNode::new_unchecked("http://example.com/p"),
                oxixml::model::NamedNode::new_unchecked("http://example.com/o"),
                oxixml::model::GraphName::DefaultGraph,
            ))
            .await
            .expect("a successful write");
        let bytes = serializer.finish().await.expect("a finished document");

        let mut parser = oxixml::trix::TriXParser::new().for_tokio_async_reader(bytes.as_slice());
        assert!(parser.next().await.expect("a quad").is_ok());
    });
}

#[cfg(all(feature = "async-tokio", feature = "rdfio"))]
#[test]
fn async_tokio_forwards_to_rdfio() {
    // The umbrella's async API exists only when every format backend forwards `async-tokio`.
    let runtime = tokio::runtime::Builder::new_current_thread()
        .build()
        .expect("a tokio runtime");
    runtime.block_on(async {
        let quad = oxixml::model::Quad::new(
            oxixml::model::NamedNode::new_unchecked("http://example.com/s"),
            oxixml::model::NamedNode::new_unchecked("http://example.com/p"),
            oxixml::model::NamedNode::new_unchecked("http://example.com/o"),
            oxixml::model::GraphName::DefaultGraph,
        );
        let mut serializer = oxixml::io::RdfSerializer::from_format(oxixml::io::RdfFormat::NQuads)
            .for_tokio_async_writer(Vec::new());
        serializer
            .serialize_quad(&quad)
            .await
            .expect("a successful write");
        let bytes = serializer.finish().await.expect("a finished document");

        let mut parser = oxixml::io::RdfParser::from_format(oxixml::io::RdfFormat::NQuads)
            .for_tokio_async_reader(bytes.as_slice());
        assert_eq!(
            parser.next().await.expect("a quad").expect("a valid quad"),
            quad
        );
    });
}

#[cfg(all(feature = "async-tokio", feature = "sparql-results"))]
#[test]
fn async_tokio_forwards_to_sparql_results() {
    let runtime = tokio::runtime::Builder::new_current_thread()
        .build()
        .expect("a tokio runtime");
    runtime.block_on(async {
        let bytes = oxixml::sparql_results::QueryResultsSerializer::from_format(
            oxixml::sparql_results::QueryResultsFormat::Json,
        )
        .serialize_boolean_to_tokio_async_write(Vec::new(), true)
        .await
        .expect("a successful write");
        assert_eq!(bytes, br#"{"head":{},"boolean":true}"#);

        let output = oxixml::sparql_results::QueryResultsParser::from_format(
            oxixml::sparql_results::QueryResultsFormat::Json,
        )
        .for_tokio_async_reader(bytes.as_slice())
        .await
        .expect("a parsed result");
        assert!(matches!(
            output,
            oxixml::sparql_results::TokioAsyncReaderQueryResultsParserOutput::Boolean(true)
        ));
    });
}

#[cfg(feature = "escape-html")]
#[test]
fn escape_html_forwards_to_xml() {
    // `&nbsp;` is an HTML5-only character reference: without `escape-html` this is an error.
    assert_eq!(
        oxixml::xml::escape::unescape("&nbsp;").expect("an HTML5 character reference"),
        "\u{a0}"
    );
}

#[cfg(all(feature = "encoding", feature = "xml"))]
#[test]
fn encoding_forwards_to_xml() {
    // `xml::encoding::decode` exists only under `oxixml-xml/encoding`.
    let windows_1252 =
        oxixml::encoding::Encoding::for_label("windows-1252").expect("a known encoding label");
    assert_eq!(
        oxixml::xml::encoding::decode(b"\x80", windows_1252).expect("decodable bytes"),
        "\u{20ac}"
    );
}

// `full` is documented as "every feature in one shot"; the modifiers below used to be
// unreachable through it -- and, before they were forwarded, through the facade at all. These
// guards fail the build rather than silently letting `full` go stale again.
#[cfg(all(feature = "full", not(feature = "rdf-12")))]
compile_error!("the `full` feature must enable `rdf-12`");
#[cfg(all(feature = "full", not(feature = "rdfc-10")))]
compile_error!("the `full` feature must enable `rdfc-10`");
#[cfg(all(feature = "full", not(feature = "sparql-12")))]
compile_error!("the `full` feature must enable `sparql-12`");
#[cfg(all(feature = "full", not(feature = "serialize")))]
compile_error!("the `full` feature must enable `serialize`");
#[cfg(all(feature = "full", not(feature = "serde-types")))]
compile_error!("the `full` feature must enable `serde-types`");
#[cfg(all(feature = "full", not(feature = "async-tokio")))]
compile_error!("the `full` feature must enable `async-tokio`");
#[cfg(all(feature = "full", not(feature = "escape-html")))]
compile_error!("the `full` feature must enable `escape-html`");
#[cfg(all(feature = "full", not(feature = "overlapped-lists")))]
compile_error!("the `full` feature must enable `overlapped-lists`");
#[cfg(all(feature = "full", not(feature = "encoding")))]
compile_error!("the `full` feature must enable `encoding`");
