//! Feature-by-feature coverage of the RDF/XML grammar, independent of the W3C test suite.

use oxixml_model::{NamedOrBlankNode, Term, Triple};
use oxixml_rdfxml::{RdfXmlParser, RdfXmlSerializer};
use std::error::Error;

/// Parses `input` with the given base IRI and returns the triples as sorted N-Triples lines.
fn parse_sorted(input: &str, base_iri: &str) -> Result<Vec<String>, Box<dyn Error>> {
    let mut lines = RdfXmlParser::new()
        .with_base_iri(base_iri)?
        .for_slice(input)
        .map(|triple| triple.map(|triple| format!("{triple} .")))
        .collect::<Result<Vec<_>, _>>()?;
    lines.sort();
    Ok(lines)
}

/// Parses `input` and returns the resulting error message.
fn parse_error(input: &str) -> String {
    RdfXmlParser::new()
        .with_base_iri("http://example.com/")
        .expect("a valid base IRI")
        .for_slice(input)
        .collect::<Result<Vec<Triple>, _>>()
        .expect_err("the document should be rejected")
        .to_string()
}

#[test]
fn node_element_without_rdf_rdf_wrapper() -> Result<(), Box<dyn Error>> {
    let triples = parse_sorted(
        r#"<schema:Person xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#"
                          xmlns:schema="http://schema.org/" rdf:about="s" />"#,
        "http://example.com/",
    )?;
    assert_eq!(
        triples,
        [
            "<http://example.com/s> <http://www.w3.org/1999/02/22-rdf-syntax-ns#type> <http://schema.org/Person> ."
        ]
    );
    Ok(())
}

#[test]
fn rdf_id_resolves_against_base_with_a_fragment() -> Result<(), Box<dyn Error>> {
    let triples = parse_sorted(
        r#"<rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">
             <rdf:Description rdf:ID="foo"><rdf:value>abc</rdf:value></rdf:Description>
           </rdf:RDF>"#,
        "http://example.com/dir/file.rdf",
    )?;
    assert_eq!(
        triples,
        [
            "<http://example.com/dir/file.rdf#foo> <http://www.w3.org/1999/02/22-rdf-syntax-ns#value> \"abc\" ."
        ]
    );
    Ok(())
}

#[test]
fn rdf_id_may_not_be_reused() {
    let error = parse_error(
        r#"<rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">
             <rdf:Description rdf:ID="foo"><rdf:value>a</rdf:value></rdf:Description>
             <rdf:Description rdf:ID="foo"><rdf:value>b</rdf:value></rdf:Description>
           </rdf:RDF>"#,
    );
    assert!(error.contains("already been used as rdf:ID"), "{error}");
}

#[test]
fn rdf_id_must_be_an_ncname() {
    let error = parse_error(
        r#"<rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">
             <rdf:Description rdf:ID="333"><rdf:value>a</rdf:value></rdf:Description>
           </rdf:RDF>"#,
    );
    assert!(error.contains("not a valid rdf:ID value"), "{error}");
}

#[test]
fn property_attributes_and_type_attribute() -> Result<(), Box<dyn Error>> {
    let triples = parse_sorted(
        r#"<rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#"
                    xmlns:ex="http://example.org/">
             <rdf:Description rdf:about="s" ex:name="Foo" rdf:type="http://example.org/T" />
           </rdf:RDF>"#,
        "http://example.com/",
    )?;
    assert_eq!(
        triples,
        [
            "<http://example.com/s> <http://example.org/name> \"Foo\" .",
            "<http://example.com/s> <http://www.w3.org/1999/02/22-rdf-syntax-ns#type> <http://example.org/T> ."
        ]
    );
    Ok(())
}

#[test]
fn parse_type_resource_creates_an_intermediate_node() -> Result<(), Box<dyn Error>> {
    let triples = RdfXmlParser::new()
        .with_base_iri("http://example.com/")?
        .for_slice(
            r#"<rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#"
                        xmlns:ex="http://example.org/">
                 <rdf:Description rdf:about="s">
                   <ex:p rdf:parseType="Resource"><ex:q>v</ex:q></ex:p>
                 </rdf:Description>
               </rdf:RDF>"#,
        )
        .collect::<Result<Vec<_>, _>>()?;
    assert_eq!(triples.len(), 2);
    let intermediate = triples
        .iter()
        .find(|t| t.predicate.as_str() == "http://example.org/p")
        .ok_or("the ex:p triple should be present")?;
    assert!(matches!(intermediate.object, Term::BlankNode(_)));
    let leaf = triples
        .iter()
        .find(|t| t.predicate.as_str() == "http://example.org/q")
        .ok_or("the ex:q triple should be present")?;
    assert!(matches!(leaf.subject, NamedOrBlankNode::BlankNode(_)));
    Ok(())
}

#[test]
fn parse_type_collection_builds_an_rdf_list() -> Result<(), Box<dyn Error>> {
    let triples = RdfXmlParser::new()
        .with_base_iri("http://example.com/")?
        .for_slice(
            r#"<rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#"
                        xmlns:ex="http://example.org/">
                 <rdf:Description rdf:about="s">
                   <ex:p rdf:parseType="Collection">
                     <rdf:Description rdf:about="a" />
                     <rdf:Description rdf:about="b" />
                   </ex:p>
                 </rdf:Description>
               </rdf:RDF>"#,
        )
        .collect::<Result<Vec<_>, _>>()?;
    // 2 rdf:first + 2 rdf:rest + the ex:p triple
    assert_eq!(triples.len(), 5);
    assert_eq!(
        triples
            .iter()
            .filter(|t| t.predicate.as_str() == "http://www.w3.org/1999/02/22-rdf-syntax-ns#first")
            .count(),
        2
    );
    let nil = Term::from(oxixml_model::NamedNode::new(
        "http://www.w3.org/1999/02/22-rdf-syntax-ns#nil",
    )?);
    assert!(triples.iter().any(|t| t.object == nil));
    Ok(())
}

#[test]
fn parse_type_literal_preserves_markup_and_namespaces() -> Result<(), Box<dyn Error>> {
    let triples = parse_sorted(
        r#"<rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#"
                    xmlns:ex="http://example.org/">
             <rdf:Description rdf:about="s">
               <ex:p rdf:parseType="Literal"><b>bold</b> &amp; more</ex:p>
             </rdf:Description>
           </rdf:RDF>"#,
        "http://example.com/",
    )?;
    assert_eq!(triples.len(), 1);
    let literal = &triples[0];
    assert!(
        literal.contains("<http://www.w3.org/1999/02/22-rdf-syntax-ns#XMLLiteral>"),
        "{literal}"
    );
    assert!(literal.contains("bold"), "{literal}");
    assert!(
        literal.contains("xmlns:rdf=\\\"http://www.w3.org/1999/02/22-rdf-syntax-ns#\\\""),
        "{literal}"
    );
    Ok(())
}

/// Regression test for RDF/XML 1.1 sec. 7.2.20 (`parseTypeOtherPropertyElt`): "All
/// rdf:parseType attribute values other than the strings "Resource", "Literal" or "Collection"
/// are treated as if the value was "Literal"". The statement used to be silently dropped.
#[test]
fn unknown_parse_type_is_treated_as_literal() -> Result<(), Box<dyn Error>> {
    for parse_type in ["Whatever", "literal", "LITERAL", "", "Resourcex"] {
        let triples = parse_sorted(
            &format!(
                r#"<rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#"
                    xmlns:ex="http://example.org/">
             <rdf:Description rdf:about="s">
               <ex:p rdf:parseType="{parse_type}">kept</ex:p>
             </rdf:Description>
           </rdf:RDF>"#
            ),
            "http://example.com/",
        )?;
        assert_eq!(
            triples,
            ["<http://example.com/s> <http://example.org/p> \"kept\"^^<http://www.w3.org/1999/02/22-rdf-syntax-ns#XMLLiteral> .".to_owned()],
            "parseType={parse_type:?}"
        );
    }
    Ok(())
}

/// `rdf:parseType` leaves no room for a second object: W3C `rdfms-empty-property-elements`
/// error001/error002 combine it with `rdf:resource`.
#[test]
fn parse_type_combined_with_resource_is_an_error() {
    for property in [
        r#"<random:someProperty rdf:parseType="Literal" rdf:resource="http://random.ioctl.org/#foo" />"#,
        r#"<random:someProperty rdf:parseType="Literal" rdf:resource="http://random.ioctl.org/#foo"></random:someProperty>"#,
        r#"<random:someProperty rdf:parseType="Resource" rdf:nodeID="x"/>"#,
        r#"<random:someProperty rdf:parseType="Collection" rdf:resource="http://random.ioctl.org/#foo"/>"#,
    ] {
        let document = format!(
            r#"<rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#"
                        xmlns:random="http://random.ioctl.org/#">
             <rdf:Description rdf:about="http://random.ioctl.org/#bar">{property}</rdf:Description>
           </rdf:RDF>"#
        );
        assert!(
            parse_sorted(&document, "http://example.com/").is_err(),
            "{property}"
        );
    }
}

/// Regression test: the W3C `rdfms-empty-property-elements` test003 expects an *empty*
/// `rdf:XMLLiteral`, not a syntax error.
#[test]
fn empty_parse_type_literal_yields_an_empty_xml_literal() -> Result<(), Box<dyn Error>> {
    for property in [
        r#"<ex:p rdf:parseType="Literal"/>"#,
        r#"<ex:p rdf:parseType="Literal"></ex:p>"#,
        r#"<ex:p rdf:parseType="Whatever"/>"#,
    ] {
        let triples = parse_sorted(
            &format!(
                r#"<rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#"
                    xmlns:ex="http://example.org/">
             <rdf:Description rdf:about="s">{property}</rdf:Description>
           </rdf:RDF>"#
            ),
            "http://example.com/",
        )?;
        assert_eq!(
            triples,
            ["<http://example.com/s> <http://example.org/p> \"\"^^<http://www.w3.org/1999/02/22-rdf-syntax-ns#XMLLiteral> .".to_owned()],
            "{property}"
        );
    }
    Ok(())
}

#[test]
fn rdf_li_becomes_numbered_container_membership_properties() -> Result<(), Box<dyn Error>> {
    let triples = parse_sorted(
        r#"<rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">
             <rdf:Bag rdf:about="bag">
               <rdf:li>one</rdf:li>
               <rdf:li>two</rdf:li>
             </rdf:Bag>
           </rdf:RDF>"#,
        "http://example.com/",
    )?;
    assert_eq!(
        triples,
        [
            "<http://example.com/bag> <http://www.w3.org/1999/02/22-rdf-syntax-ns#_1> \"one\" .",
            "<http://example.com/bag> <http://www.w3.org/1999/02/22-rdf-syntax-ns#_2> \"two\" .",
            "<http://example.com/bag> <http://www.w3.org/1999/02/22-rdf-syntax-ns#type> <http://www.w3.org/1999/02/22-rdf-syntax-ns#Bag> ."
        ]
    );
    Ok(())
}

#[test]
fn rdf_id_on_a_property_element_reifies_the_statement() -> Result<(), Box<dyn Error>> {
    let triples = parse_sorted(
        r#"<rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#"
                    xmlns:ex="http://example.org/">
             <rdf:Description rdf:about="s"><ex:p rdf:ID="stmt">v</ex:p></rdf:Description>
           </rdf:RDF>"#,
        "http://example.com/",
    )?;
    assert_eq!(
        triples,
        [
            "<http://example.com/#stmt> <http://www.w3.org/1999/02/22-rdf-syntax-ns#object> \"v\" .",
            "<http://example.com/#stmt> <http://www.w3.org/1999/02/22-rdf-syntax-ns#predicate> <http://example.org/p> .",
            "<http://example.com/#stmt> <http://www.w3.org/1999/02/22-rdf-syntax-ns#subject> <http://example.com/s> .",
            "<http://example.com/#stmt> <http://www.w3.org/1999/02/22-rdf-syntax-ns#type> <http://www.w3.org/1999/02/22-rdf-syntax-ns#Statement> .",
            "<http://example.com/s> <http://example.org/p> \"v\" ."
        ]
    );
    Ok(())
}

#[test]
fn xml_lang_and_xml_base_are_scoped() -> Result<(), Box<dyn Error>> {
    let triples = parse_sorted(
        r#"<rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#"
                    xmlns:ex="http://example.org/" xml:lang="en" xml:base="http://outer.example/">
             <rdf:Description rdf:about="s">
               <ex:p>outer</ex:p>
               <ex:p xml:lang="fr">inner</ex:p>
               <ex:p xml:lang="">plain</ex:p>
             </rdf:Description>
             <rdf:Description xml:base="http://inner.example/" rdf:about="t">
               <ex:p rdf:resource="rel" />
             </rdf:Description>
           </rdf:RDF>"#,
        "http://example.com/",
    )?;
    assert_eq!(
        triples,
        [
            "<http://inner.example/t> <http://example.org/p> <http://inner.example/rel> .",
            "<http://outer.example/s> <http://example.org/p> \"inner\"@fr .",
            "<http://outer.example/s> <http://example.org/p> \"outer\"@en .",
            "<http://outer.example/s> <http://example.org/p> \"plain\" ."
        ]
    );
    Ok(())
}

#[test]
fn rdf_datatype_produces_a_typed_literal() -> Result<(), Box<dyn Error>> {
    let triples = parse_sorted(
        r#"<rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#"
                    xmlns:ex="http://example.org/">
             <rdf:Description rdf:about="s">
               <ex:p rdf:datatype="http://www.w3.org/2001/XMLSchema#integer">1</ex:p>
             </rdf:Description>
           </rdf:RDF>"#,
        "http://example.com/",
    )?;
    assert_eq!(
        triples,
        [
            "<http://example.com/s> <http://example.org/p> \"1\"^^<http://www.w3.org/2001/XMLSchema#integer> ."
        ]
    );
    Ok(())
}

#[test]
fn internal_entities_are_expanded() -> Result<(), Box<dyn Error>> {
    let triples = parse_sorted(
        r#"<?xml version="1.0"?>
           <!DOCTYPE rdf:RDF [<!ENTITY ex "http://example.org/">]>
           <rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">
             <rdf:Description rdf:about="&ex;s"><rdf:value>&ex;v</rdf:value></rdf:Description>
           </rdf:RDF>"#,
        "http://example.com/",
    )?;
    assert_eq!(
        triples,
        [
            "<http://example.org/s> <http://www.w3.org/1999/02/22-rdf-syntax-ns#value> \"http://example.org/v\" ."
        ]
    );
    Ok(())
}

#[test]
fn banned_terms_are_rejected() {
    for banned in ["rdf:aboutEach", "rdf:aboutEachPrefix", "rdf:li", "rdf:RDF"] {
        let document = format!(
            r#"<rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">
                 <rdf:Description rdf:about="s" {banned}="x" />
               </rdf:RDF>"#
        );
        let error = parse_error(&document);
        assert!(
            error.contains("is not a valid attribute"),
            "{banned}: {error}"
        );
    }
}

#[test]
fn reserved_terms_are_rejected_as_element_names() {
    let error = parse_error(
        r#"<rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">
             <rdf:ID rdf:about="s" />
           </rdf:RDF>"#,
    );
    assert!(error.contains("Invalid node element tag name"), "{error}");
}

#[test]
fn both_resource_and_node_id_is_rejected() {
    let error = parse_error(
        r#"<rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#"
                    xmlns:ex="http://example.org/">
             <rdf:Description rdf:about="s">
               <ex:p rdf:resource="o" rdf:nodeID="b" />
             </rdf:Description>
           </rdf:RDF>"#,
    );
    assert!(error.contains("rdf:resource and rdf:nodeID"), "{error}");
}

#[test]
fn relative_iri_without_a_base_is_rejected() {
    let error = RdfXmlParser::new()
        .for_slice(
            r#"<rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">
                 <rdf:Description rdf:about="s"><rdf:value>v</rdf:value></rdf:Description>
               </rdf:RDF>"#,
        )
        .collect::<Result<Vec<Triple>, _>>()
        .expect_err("no base IRI is set")
        .to_string();
    assert!(error.contains("no xml:base is set"), "{error}");
}

#[test]
fn lenient_mode_accepts_relative_iris() -> Result<(), Box<dyn Error>> {
    let triples = RdfXmlParser::new()
        .lenient()
        .for_slice(
            r#"<rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">
                 <rdf:Description rdf:about="s"><rdf:value>v</rdf:value></rdf:Description>
               </rdf:RDF>"#,
        )
        .collect::<Result<Vec<_>, _>>()?;
    assert_eq!(triples.len(), 1);
    assert_eq!(triples[0].subject.to_string(), "<s>");
    Ok(())
}

#[test]
fn non_utf8_encoding_declaration_is_rejected() {
    let error = parse_error(
        r#"<?xml version="1.0" encoding="ISO-8859-1"?>
           <rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#" />"#,
    );
    assert!(error.contains("Only UTF-8 is supported"), "{error}");
}

#[test]
fn reader_and_slice_parsers_agree() -> Result<(), Box<dyn Error>> {
    let document = r#"<rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#"
                               xmlns:ex="http://example.org/">
                        <ex:T rdf:about="http://example.com/s"><ex:p>v</ex:p></ex:T>
                      </rdf:RDF>"#;
    let mut from_slice = RdfXmlParser::new()
        .for_slice(document)
        .map(|t| t.map(|t| t.to_string()))
        .collect::<Result<Vec<_>, _>>()?;
    let mut from_reader = RdfXmlParser::new()
        .for_reader(document.as_bytes())
        .map(|t| t.map(|t| t.to_string()))
        .collect::<Result<Vec<_>, _>>()?;
    from_slice.sort();
    from_reader.sort();
    assert_eq!(from_slice, from_reader);
    Ok(())
}

#[test]
fn prefixes_and_base_iri_are_exposed() -> Result<(), Box<dyn Error>> {
    let document = r#"<rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#"
                               xmlns:ex="http://example.org/" xml:base="http://example.com/">
                        <rdf:Description rdf:about="s"><ex:p>v</ex:p></rdf:Description>
                      </rdf:RDF>"#;
    let mut parser = RdfXmlParser::new().for_slice(document);
    assert!(parser.base_iri().is_none());
    assert_eq!(parser.prefixes().collect::<Vec<_>>(), []);
    parser.next().ok_or("one triple")??;
    assert_eq!(parser.base_iri(), Some("http://example.com/"));
    assert_eq!(
        parser.prefixes().collect::<Vec<_>>(),
        [
            ("rdf", "http://www.w3.org/1999/02/22-rdf-syntax-ns#"),
            ("ex", "http://example.org/")
        ]
    );
    Ok(())
}

#[test]
fn serializer_output_round_trips_through_the_parser() -> Result<(), Box<dyn Error>> {
    let document = r#"<rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#"
                               xmlns:ex="http://example.org/">
                        <ex:T rdf:about="http://example.com/s">
                          <ex:p>text</ex:p>
                          <ex:q rdf:resource="http://example.com/o" />
                          <ex:r rdf:datatype="http://www.w3.org/2001/XMLSchema#integer">3</ex:r>
                          <ex:lang xml:lang="en">hello</ex:lang>
                        </ex:T>
                      </rdf:RDF>"#;
    let mut original = RdfXmlParser::new()
        .for_slice(document)
        .collect::<Result<Vec<_>, _>>()?;
    original.sort_by_key(ToString::to_string);

    let mut serializer = RdfXmlSerializer::new()
        .with_prefix("ex", "http://example.org/")?
        .for_writer(Vec::new());
    for triple in &original {
        serializer.serialize_triple(triple)?;
    }
    let serialized = serializer.finish()?;

    let mut round_tripped = RdfXmlParser::new()
        .for_slice(&serialized)
        .collect::<Result<Vec<_>, _>>()?;
    round_tripped.sort_by_key(ToString::to_string);
    assert_eq!(
        original,
        round_tripped,
        "{}",
        String::from_utf8_lossy(&serialized)
    );
    Ok(())
}
