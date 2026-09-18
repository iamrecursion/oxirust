//! Parsing correctness tests: term kinds, graphs, unicode and error handling.

use oxixml_model::{BlankNode, GraphName, Literal, NamedNode, Quad};
use oxixml_trix::TriXParser;

fn parse_ok(xml: &str) -> Vec<Quad> {
    TriXParser::new()
        .for_slice(xml)
        .collect::<Result<Vec<_>, _>>()
        .unwrap_or_else(|e| panic!("expected {xml:?} to parse, got error: {e}"))
}

fn parse_err(xml: &str) -> String {
    let err = TriXParser::new()
        .for_slice(xml)
        .collect::<Result<Vec<_>, _>>()
        .expect_err("expected a parse error");
    err.to_string()
}

const NS_DECL: &str = r#"xmlns="http://www.w3.org/2004/03/trix/trix-1/""#;

#[test]
fn empty_document_has_no_quads() -> Result<(), Box<dyn std::error::Error>> {
    assert_eq!(parse_ok(&format!(r#"<TriX {NS_DECL}/>"#)), vec![]);
    assert_eq!(parse_ok(&format!(r#"<TriX {NS_DECL}></TriX>"#)), vec![]);
    assert_eq!(
        parse_ok(&format!(r#"<TriX {NS_DECL}><graph/></TriX>"#)),
        vec![]
    );
    assert_eq!(
        parse_ok(&format!(r#"<TriX {NS_DECL}><graph></graph></TriX>"#)),
        vec![]
    );
    Ok(())
}

#[test]
fn default_graph_triple_all_uris() -> Result<(), Box<dyn std::error::Error>> {
    let xml = format!(
        r#"<TriX {NS_DECL}>
             <graph>
               <triple>
                 <uri>http://example.com/s</uri>
                 <uri>http://example.com/p</uri>
                 <uri>http://example.com/o</uri>
               </triple>
             </graph>
           </TriX>"#
    );
    let quads = parse_ok(&xml);
    assert_eq!(
        quads,
        vec![Quad::new(
            NamedNode::new("http://example.com/s")?,
            NamedNode::new("http://example.com/p")?,
            NamedNode::new("http://example.com/o")?,
            GraphName::DefaultGraph,
        )]
    );
    Ok(())
}

#[test]
fn named_graph_via_uri() -> Result<(), Box<dyn std::error::Error>> {
    let xml = format!(
        r#"<TriX {NS_DECL}>
             <graph>
               <uri>http://example.com/g</uri>
               <triple>
                 <uri>http://example.com/s</uri>
                 <uri>http://example.com/p</uri>
                 <uri>http://example.com/o</uri>
               </triple>
             </graph>
           </TriX>"#
    );
    let quads = parse_ok(&xml);
    assert_eq!(quads[0].graph_name, NamedNode::new("http://example.com/g")?);
    Ok(())
}

#[test]
fn named_graph_via_blank_node_id() -> Result<(), Box<dyn std::error::Error>> {
    let xml = format!(
        r#"<TriX {NS_DECL}>
             <graph>
               <id>g1</id>
               <triple>
                 <uri>http://example.com/s</uri>
                 <uri>http://example.com/p</uri>
                 <uri>http://example.com/o</uri>
               </triple>
             </graph>
           </TriX>"#
    );
    let quads = parse_ok(&xml);
    assert_eq!(quads[0].graph_name, BlankNode::new_unchecked("g1"));
    Ok(())
}

#[test]
fn all_term_kinds() -> Result<(), Box<dyn std::error::Error>> {
    let xml = format!(
        r#"<TriX {NS_DECL}>
             <graph>
               <triple>
                 <id>b1</id>
                 <uri>http://example.com/p1</uri>
                 <plainLiteral>hello</plainLiteral>
               </triple>
               <triple>
                 <uri>http://example.com/s</uri>
                 <uri>http://example.com/p2</uri>
                 <plainLiteral xml:lang="en">hi</plainLiteral>
               </triple>
               <triple>
                 <uri>http://example.com/s</uri>
                 <uri>http://example.com/p3</uri>
                 <typedLiteral datatype="http://www.w3.org/2001/XMLSchema#integer">42</typedLiteral>
               </triple>
               <triple>
                 <uri>http://example.com/s</uri>
                 <uri>http://example.com/p4</uri>
                 <id>b2</id>
               </triple>
             </graph>
           </TriX>"#
    );
    let quads = parse_ok(&xml);
    assert_eq!(quads.len(), 4);
    assert_eq!(quads[0].subject, BlankNode::new_unchecked("b1"));
    assert_eq!(quads[0].object, Literal::new_simple_literal("hello"));
    assert_eq!(
        quads[1].object,
        Literal::new_language_tagged_literal_unchecked("hi", "en")
    );
    assert_eq!(
        quads[2].object,
        Literal::new_typed_literal(
            "42",
            NamedNode::new("http://www.w3.org/2001/XMLSchema#integer")?
        )
    );
    assert_eq!(quads[3].object, BlankNode::new_unchecked("b2"));
    Ok(())
}

#[test]
fn language_tag_is_lowercased() -> Result<(), Box<dyn std::error::Error>> {
    let xml = format!(
        r#"<TriX {NS_DECL}>
             <graph>
               <triple>
                 <uri>http://example.com/s</uri>
                 <uri>http://example.com/p</uri>
                 <plainLiteral xml:lang="EN-us">hi</plainLiteral>
               </triple>
             </graph>
           </TriX>"#
    );
    let quads = parse_ok(&xml);
    assert_eq!(quads[0].object.to_string(), "\"hi\"@en-us");
    Ok(())
}

#[test]
fn blank_node_ids_are_scoped_to_the_document() -> Result<(), Box<dyn std::error::Error>> {
    // The same `<id>b1</id>` used as both a subject and an object must map to the
    // *same* blank node.
    let xml = format!(
        r#"<TriX {NS_DECL}>
             <graph>
               <triple>
                 <id>b1</id>
                 <uri>http://example.com/p</uri>
                 <uri>http://example.com/o</uri>
               </triple>
               <triple>
                 <uri>http://example.com/s</uri>
                 <uri>http://example.com/p</uri>
                 <id>b1</id>
               </triple>
             </graph>
           </TriX>"#
    );
    let quads = parse_ok(&xml);
    assert_eq!(quads[0].subject, BlankNode::new_unchecked("b1"));
    assert_eq!(quads[1].object, BlankNode::new_unchecked("b1"));
    Ok(())
}

#[test]
fn unicode_content_round_trips_through_text_and_entities() -> Result<(), Box<dyn std::error::Error>>
{
    let xml = format!(
        r#"<TriX {NS_DECL}>
             <graph>
               <triple>
                 <uri>http://example.com/s</uri>
                 <uri>http://example.com/p</uri>
                 <plainLiteral>日本語 caf&#233; a &amp; b &lt;tag&gt;</plainLiteral>
               </triple>
             </graph>
           </TriX>"#
    );
    let quads = parse_ok(&xml);
    assert_eq!(quads[0].object.to_string(), "\"日本語 café a & b <tag>\"");
    Ok(())
}

#[test]
fn multiple_graphs_are_kept_separate() -> Result<(), Box<dyn std::error::Error>> {
    let xml = format!(
        r#"<TriX {NS_DECL}>
             <graph>
               <uri>http://example.com/g1</uri>
               <triple>
                 <uri>http://example.com/s</uri>
                 <uri>http://example.com/p</uri>
                 <uri>http://example.com/o1</uri>
               </triple>
             </graph>
             <graph>
               <uri>http://example.com/g2</uri>
               <triple>
                 <uri>http://example.com/s</uri>
                 <uri>http://example.com/p</uri>
                 <uri>http://example.com/o2</uri>
               </triple>
             </graph>
             <graph>
               <triple>
                 <uri>http://example.com/s</uri>
                 <uri>http://example.com/p</uri>
                 <uri>http://example.com/o3</uri>
               </triple>
             </graph>
           </TriX>"#
    );
    let quads = parse_ok(&xml);
    assert_eq!(quads.len(), 3);
    assert_eq!(
        quads[0].graph_name,
        NamedNode::new("http://example.com/g1")?
    );
    assert_eq!(
        quads[1].graph_name,
        NamedNode::new("http://example.com/g2")?
    );
    assert_eq!(quads[2].graph_name, GraphName::DefaultGraph);
    Ok(())
}

// --- Error cases -----------------------------------------------------------

#[test]
fn error_wrong_root_element() {
    let message = parse_err(&format!(r#"<NotTriX {NS_DECL}/>"#));
    assert!(message.contains("root <TriX>"), "{message}");
}

#[test]
fn error_element_outside_trix_namespace() {
    let message = parse_err(r#"<TriX xmlns="urn:not-trix"/>"#);
    assert!(message.contains("TriX namespace"), "{message}");
}

#[test]
fn error_triple_with_too_few_terms() {
    let xml = format!(
        r#"<TriX {NS_DECL}><graph><triple><uri>http://example.com/s</uri></triple></graph></TriX>"#
    );
    let message = parse_err(&xml);
    assert!(message.contains("exactly 3 child terms"), "{message}");
}

#[test]
fn error_triple_with_too_many_terms() {
    let xml = format!(
        r#"<TriX {NS_DECL}><graph><triple>
             <uri>http://example.com/s</uri>
             <uri>http://example.com/p</uri>
             <uri>http://example.com/o</uri>
             <uri>http://example.com/extra</uri>
           </triple></graph></TriX>"#
    );
    let message = parse_err(&xml);
    assert!(message.contains("more than 3"), "{message}");
}

#[test]
fn error_empty_triple_element() {
    let xml = format!(r#"<TriX {NS_DECL}><graph><triple/></graph></TriX>"#);
    let message = parse_err(&xml);
    assert!(message.contains("found 0"), "{message}");
}

#[test]
fn error_typed_literal_missing_datatype() {
    let xml = format!(
        r#"<TriX {NS_DECL}><graph><triple>
             <uri>http://example.com/s</uri>
             <uri>http://example.com/p</uri>
             <typedLiteral>42</typedLiteral>
           </triple></graph></TriX>"#
    );
    let message = parse_err(&xml);
    assert!(message.contains("datatype"), "{message}");
}

#[test]
fn error_invalid_datatype_iri() {
    let xml = format!(
        r#"<TriX {NS_DECL}><graph><triple>
             <uri>http://example.com/s</uri>
             <uri>http://example.com/p</uri>
             <typedLiteral datatype="not an iri">42</typedLiteral>
           </triple></graph></TriX>"#
    );
    let message = parse_err(&xml);
    assert!(message.contains("invalid IRI"), "{message}");
}

#[test]
fn error_invalid_language_tag() {
    let xml = format!(
        r#"<TriX {NS_DECL}><graph><triple>
             <uri>http://example.com/s</uri>
             <uri>http://example.com/p</uri>
             <plainLiteral xml:lang="not a tag">hi</plainLiteral>
           </triple></graph></TriX>"#
    );
    let message = parse_err(&xml);
    assert!(message.contains("language tag"), "{message}");
}

#[test]
fn error_invalid_uri() {
    let xml = format!(
        r#"<TriX {NS_DECL}><graph><triple>
             <uri>not a uri</uri>
             <uri>http://example.com/p</uri>
             <uri>http://example.com/o</uri>
           </triple></graph></TriX>"#
    );
    let message = parse_err(&xml);
    assert!(message.contains("invalid IRI"), "{message}");
}

#[test]
fn error_invalid_blank_node_id() {
    let xml = format!(
        r#"<TriX {NS_DECL}><graph><triple>
             <id>not a valid ncname</id>
             <uri>http://example.com/p</uri>
             <uri>http://example.com/o</uri>
           </triple></graph></TriX>"#
    );
    let message = parse_err(&xml);
    assert!(message.contains("blank node id"), "{message}");
}

#[test]
fn error_literal_predicate() {
    let xml = format!(
        r#"<TriX {NS_DECL}><graph><triple>
             <uri>http://example.com/s</uri>
             <plainLiteral>not-a-predicate</plainLiteral>
             <uri>http://example.com/o</uri>
           </triple></graph></TriX>"#
    );
    let message = parse_err(&xml);
    assert!(message.contains("predicate must be a <uri>"), "{message}");
}

#[test]
fn error_literal_subject() {
    let xml = format!(
        r#"<TriX {NS_DECL}><graph><triple>
             <plainLiteral>not-a-subject</plainLiteral>
             <uri>http://example.com/p</uri>
             <uri>http://example.com/o</uri>
           </triple></graph></TriX>"#
    );
    let message = parse_err(&xml);
    assert!(message.contains("subject must be"), "{message}");
}

#[test]
fn error_literal_graph_name() {
    let xml = format!(r#"<TriX {NS_DECL}><graph><plainLiteral>nope</plainLiteral></graph></TriX>"#);
    let message = parse_err(&xml);
    assert!(message.contains("graph name element"), "{message}");
}

#[test]
fn error_graph_name_after_first_triple() {
    let xml = format!(
        r#"<TriX {NS_DECL}><graph>
             <triple>
               <uri>http://example.com/s</uri>
               <uri>http://example.com/p</uri>
               <uri>http://example.com/o</uri>
             </triple>
             <uri>http://example.com/too-late</uri>
           </graph></TriX>"#
    );
    let message = parse_err(&xml);
    assert!(message.contains("first child"), "{message}");
}

#[test]
fn error_doctype_forbidden() {
    let xml = format!(r#"<!DOCTYPE TriX><TriX {NS_DECL}/>"#);
    let message = parse_err(&xml);
    assert!(message.contains("DOCTYPE"), "{message}");
}

#[test]
fn error_nested_element_inside_term() {
    let xml = format!(
        r#"<TriX {NS_DECL}><graph><triple>
             <uri><nested/>http://example.com/s</uri>
             <uri>http://example.com/p</uri>
             <uri>http://example.com/o</uri>
           </triple></graph></TriX>"#
    );
    let message = parse_err(&xml);
    assert!(message.contains("nested elements"), "{message}");
}

#[test]
fn error_text_outside_term() {
    let xml = format!(r#"<TriX {NS_DECL}><graph>stray text</graph></TriX>"#);
    let message = parse_err(&xml);
    assert!(message.contains("unexpected text"), "{message}");
}

#[test]
fn error_unexpected_element_inside_graph() {
    let xml = format!(r#"<TriX {NS_DECL}><graph><notATerm/></graph></TriX>"#);
    let message = parse_err(&xml);
    assert!(message.contains("graph name element"), "{message}");
}

#[test]
fn error_unexpected_element_inside_triple() {
    let xml = format!(r#"<TriX {NS_DECL}><graph><triple><notATerm/></triple></graph></TriX>"#);
    let message = parse_err(&xml);
    assert!(message.contains("term element"), "{message}");
}

#[test]
fn error_unclosed_tag_is_reported_as_ill_formed_xml() {
    let xml = format!(r#"<TriX {NS_DECL}><graph>"#);
    let message = parse_err(&xml);
    assert!(!message.is_empty());
}

#[test]
fn error_missing_root_element_on_empty_input() {
    let message = parse_err("");
    assert!(message.contains("missing document element"), "{message}");
}

/// Regression test: the iterator must fuse (keep returning `None`) once it has reported the
/// "missing document element" error, not repeat the same `Err` forever. `.take(5)` bounds how
/// many times `next()` is called so a regression fails fast with a length mismatch instead of
/// hanging the test process.
#[test]
fn iterator_terminates_after_error_on_empty_input() {
    let results: Vec<_> = TriXParser::new().for_slice("").take(5).collect();
    assert_eq!(results.len(), 1, "{results:?}");
    assert!(results[0].is_err(), "{results:?}");
}

/// Same regression as above, but for the "unexpected end of file: an element was not closed"
/// path (truncated input rather than empty input) -- the two error branches of `on_eof` used to
/// both skip setting `finished`.
#[test]
fn iterator_terminates_after_error_on_truncated_input() {
    let xml = format!(r#"<TriX {NS_DECL}><graph>"#);
    let results: Vec<_> = TriXParser::new().for_slice(&xml).take(5).collect();
    assert_eq!(results.len(), 1, "{results:?}");
    assert!(results[0].is_err(), "{results:?}");
}

/// Regression test: a self-closing term element is spec-legal XML for an element with empty
/// content. It used to desynchronize the parser, which consumed the *following* sibling's
/// events looking for an `End` tag that a self-closing element never produces.
#[test]
fn self_closing_term_elements_are_empty_literals() -> Result<(), Box<dyn std::error::Error>> {
    let s = NamedNode::new("http://example.com/s")?;
    let p = NamedNode::new("http://example.com/p")?;
    let triple = |object: &str| {
        format!(
            r#"<triple><uri>http://example.com/s</uri><uri>http://example.com/p</uri>{object}</triple>"#
        )
    };

    for object in [
        "<plainLiteral/>",
        "<plainLiteral></plainLiteral>",
        r#"<typedLiteral datatype="http://example.com/d"/>"#,
        r#"<typedLiteral datatype="http://example.com/d"></typedLiteral>"#,
    ] {
        // Two triples in a row: the self-closing element must not swallow the second one.
        let xml = format!(
            r#"<TriX {NS_DECL}><graph>{}{}</graph></TriX>"#,
            triple(object),
            triple(object)
        );
        let quads = parse_ok(&xml);
        assert_eq!(quads.len(), 2, "{xml}");
        let expected_object = if object.contains("typedLiteral") {
            Literal::new_typed_literal("", NamedNode::new("http://example.com/d")?)
        } else {
            Literal::new_simple_literal("")
        };
        for quad in quads {
            assert_eq!(
                quad,
                Quad::new(
                    s.clone(),
                    p.clone(),
                    expected_object.clone(),
                    GraphName::DefaultGraph
                ),
                "{xml}"
            );
        }
    }
    Ok(())
}

/// Regression test: a self-closing `<uri/>`/`<id/>` graph-name element must be reported as an
/// invalid (empty) name rather than eating the rest of the graph.
#[test]
fn self_closing_graph_name_element_is_rejected_cleanly() {
    for name in ["<uri/>", "<id/>"] {
        let xml = format!(
            r#"<TriX {NS_DECL}><graph>{name}<triple><uri>http://example.com/s</uri><uri>http://example.com/p</uri><plainLiteral>o</plainLiteral></triple></graph></TriX>"#
        );
        let message = parse_err(&xml);
        assert!(!message.is_empty(), "{xml}");
        // The failure must be about the empty name itself, not about a desynchronized parse.
        assert!(
            !message.contains("nested elements"),
            "unexpected desynchronization error: {message}"
        );
    }
}

/// Regression test: a self-closing `<uri/>`/`<id/>` in *term* position, followed by more terms.
#[test]
fn self_closing_uri_term_does_not_swallow_siblings() {
    let xml = format!(
        r#"<TriX {NS_DECL}><graph><triple><uri/><uri>http://example.com/p</uri><plainLiteral>o</plainLiteral></triple></graph></TriX>"#
    );
    let message = parse_err(&xml);
    assert!(!message.is_empty());
    assert!(
        !message.contains("found <triple>"),
        "unexpected desynchronization error: {message}"
    );
}

/// Regression test: a blank-node graph name written with an explicit empty element pair keeps
/// working, and a self-closing `<id/>` for an *object* term yields a clean error.
#[test]
fn empty_id_object_is_rejected() {
    let xml = format!(
        r#"<TriX {NS_DECL}><graph><triple><uri>http://example.com/s</uri><uri>http://example.com/p</uri><id/></triple></graph></TriX>"#
    );
    let message = parse_err(&xml);
    assert!(!message.is_empty(), "{message}");
}
