//! Self-contained RDFa cases.
//!
//! These cover the same ground as `rdfa_suite.rs` — one case per feature of the processing model —
//! but need no external test suite, so the crate has meaningful coverage on a bare checkout. Each
//! case states its expectation as a set of N-Triples lines, compared against the extracted graph
//! with blank-node labels normalized away.

use std::collections::BTreeSet;

use oxixml_model::{Term, Triple};
use oxixml_rdfa::{HostLanguage, RdfaParser};

/// The base every case is extracted against.
const BASE: &str = "http://example.com/doc";

/// Extracts `markup` as `host` and returns the graph as normalized N-Triples lines.
fn extract(markup: &str, host: HostLanguage) -> BTreeSet<String> {
    let triples: Vec<Triple> = RdfaParser::new()
        .with_host_language(host)
        .with_base_iri(BASE)
        .expect("a valid base IRI")
        .for_slice(markup)
        .collect::<Result<Vec<_>, _>>()
        .expect("extraction succeeds");
    triples.iter().map(normalize).collect()
}

/// Renders a triple with every blank node replaced by `_:_`, so that a case can assert the shape
/// of a graph without depending on the labels a processor happened to pick.
fn normalize(triple: &Triple) -> String {
    fn subject(value: &oxixml_model::NamedOrBlankNode) -> String {
        match value {
            oxixml_model::NamedOrBlankNode::BlankNode(_) => "_:_".to_owned(),
            other => other.to_string(),
        }
    }
    let object = match &triple.object {
        Term::BlankNode(_) => "_:_".to_owned(),
        other => other.to_string(),
    };
    format!("{} {} {object}", subject(&triple.subject), triple.predicate)
}

/// Asserts that extracting `markup` yields exactly `expected`.
#[track_caller]
fn assert_graph(host: HostLanguage, markup: &str, expected: &[&str]) {
    let actual = extract(markup, host);
    let expected: BTreeSet<String> = expected.iter().map(|line| (*line).to_owned()).collect();
    assert_eq!(
        actual,
        expected,
        "\n  extracted:\n{}\n  expected:\n{}",
        actual
            .iter()
            .map(|line| format!("    {line}\n"))
            .collect::<String>(),
        expected
            .iter()
            .map(|line| format!("    {line}\n"))
            .collect::<String>(),
    );
}

/// Wraps a body fragment into a minimal HTML5 document.
fn html(body: &str) -> String {
    format!("<!DOCTYPE html>\n<html>\n<head><title>t</title></head>\n<body>{body}</body>\n</html>")
}

/// Wraps a body fragment into a minimal HTML5 document with attributes on `<html>`.
fn html_with(root_attributes: &str, body: &str) -> String {
    format!(
        "<!DOCTYPE html>\n<html {root_attributes}>\n<head><title>t</title></head>\n\
         <body>{body}</body>\n</html>"
    )
}

// ------------------------------------------------------------------- literals

#[test]
fn property_with_text_content_makes_a_plain_literal() {
    assert_graph(
        HostLanguage::Html5,
        &html(r#"<p about="http://example.com/a" property="http://example.com/p">hello</p>"#),
        &["<http://example.com/a> <http://example.com/p> \"hello\""],
    );
}

#[test]
fn content_overrides_the_text() {
    assert_graph(
        HostLanguage::Html5,
        &html(
            r#"<p about="http://example.com/a" property="http://example.com/p"
                  content="real">ignored</p>"#,
        ),
        &["<http://example.com/a> <http://example.com/p> \"real\""],
    );
}

#[test]
fn datatype_makes_a_typed_literal() {
    assert_graph(
        HostLanguage::Html5,
        &html_with(
            r#"prefix="xsd: http://www.w3.org/2001/XMLSchema#""#,
            r#"<p about="http://example.com/a" property="http://example.com/p"
                  datatype="xsd:integer">42</p>"#,
        ),
        &["<http://example.com/a> <http://example.com/p> \
             \"42\"^^<http://www.w3.org/2001/XMLSchema#integer>"],
    );
}

#[test]
fn an_empty_datatype_forces_a_plain_literal_even_with_a_language() {
    assert_graph(
        HostLanguage::Html5,
        &html_with(
            r#"lang="en""#,
            r#"<p about="http://example.com/a" property="http://example.com/p"
                  datatype="">text</p>"#,
        ),
        &["<http://example.com/a> <http://example.com/p> \"text\"@en"],
    );
}

#[test]
fn an_unresolvable_datatype_falls_back_to_a_plain_literal() {
    // RDFa Core §7.5 step 11: a `@datatype` that resolves to nothing is "empty according to the
    // section on CURIE and IRI Processing".
    assert_graph(
        HostLanguage::Html5,
        &html(
            r#"<p about="http://example.com/a" property="http://example.com/p"
                  datatype="not/a/term">text</p>"#,
        ),
        &["<http://example.com/a> <http://example.com/p> \"text\""],
    );
}

#[test]
fn xml_lang_tags_a_literal_and_an_empty_value_clears_it() {
    assert_graph(
        HostLanguage::Xml,
        "<root xml:lang=\"ja\" about=\"http://example.com/a\">\
         <p property=\"http://example.com/p\">こころ</p>\
         <p property=\"http://example.com/q\" xml:lang=\"\">Kokoro</p></root>",
        &[
            "<http://example.com/a> <http://example.com/p> \"こころ\"@ja",
            "<http://example.com/a> <http://example.com/q> \"Kokoro\"",
        ],
    );
}

#[test]
fn xml_lang_beats_lang_in_html() {
    assert_graph(
        HostLanguage::Html5,
        &html(
            r#"<p about="http://example.com/a" property="http://example.com/p"
                  lang="en" xml:lang="fr">bonjour</p>"#,
        ),
        &["<http://example.com/a> <http://example.com/p> \"bonjour\"@fr"],
    );
}

#[test]
fn an_xml_literal_serializes_the_element_content() {
    let graph = extract(
        &html_with(
            r#"prefix="rdf: http://www.w3.org/1999/02/22-rdf-syntax-ns#""#,
            r#"<p about="http://example.com/a" property="http://example.com/p"
                  datatype="rdf:XMLLiteral">E = mc<sup>2</sup></p>"#,
        ),
        HostLanguage::Html5,
    );
    let line = graph.iter().next().expect("one triple");
    assert!(line.contains("<sup"), "{line}");
    assert!(line.contains("2</sup>"), "{line}");
    assert!(
        line.ends_with("^^<http://www.w3.org/1999/02/22-rdf-syntax-ns#XMLLiteral>"),
        "{line}"
    );
}

#[test]
fn an_html_literal_serializes_the_element_content() {
    let graph = extract(
        &html_with(
            r#"prefix="rdf: http://www.w3.org/1999/02/22-rdf-syntax-ns#""#,
            r#"<p about="http://example.com/a" property="http://example.com/p"
                  datatype="rdf:HTML">a<b>c</b></p>"#,
        ),
        HostLanguage::Html5,
    );
    let line = graph.iter().next().expect("one triple");
    assert!(line.contains("a<b>c</b>"), "{line}");
    assert!(
        line.ends_with("^^<http://www.w3.org/1999/02/22-rdf-syntax-ns#HTML>"),
        "{line}"
    );
}

// ------------------------------------------------------------------ datetime

#[test]
fn datetime_infers_an_xsd_datatype() {
    assert_graph(
        HostLanguage::Html5,
        &html(
            r#"<p about="http://example.com/a" property="http://example.com/p"
                  datetime="2013-01-02">Jan 2</p>"#,
        ),
        &["<http://example.com/a> <http://example.com/p> \
             \"2013-01-02\"^^<http://www.w3.org/2001/XMLSchema#date>"],
    );
}

#[test]
fn a_time_element_uses_its_own_text_as_the_datetime() {
    assert_graph(
        HostLanguage::Html5,
        &html(
            r#"<div about="http://example.com/a">
                 <time property="http://example.com/p">2013-01-02T10:00:00</time></div>"#,
        ),
        &["<http://example.com/a> <http://example.com/p> \
             \"2013-01-02T10:00:00\"^^<http://www.w3.org/2001/XMLSchema#dateTime>"],
    );
}

#[test]
fn a_datetime_that_matches_no_xsd_type_is_a_plain_literal() {
    assert_graph(
        HostLanguage::Html5,
        &html(
            r#"<p about="http://example.com/a" property="http://example.com/p"
                  datetime="next tuesday">soon</p>"#,
        ),
        &["<http://example.com/a> <http://example.com/p> \"next tuesday\""],
    );
}

#[test]
fn datetime_is_not_an_rdfa_attribute_outside_html() {
    assert_graph(
        HostLanguage::Xml,
        r#"<root about="http://example.com/a"><p property="http://example.com/p"
             datetime="2013-01-02">Jan 2</p></root>"#,
        &["<http://example.com/a> <http://example.com/p> \"Jan 2\""],
    );
}

// -------------------------------------------------------- subjects and typing

#[test]
fn an_empty_about_is_the_base() {
    assert_graph(
        HostLanguage::Html5,
        &html(r#"<p about="" property="http://example.com/p">x</p>"#),
        &["<http://example.com/doc> <http://example.com/p> \"x\""],
    );
}

#[test]
fn typeof_without_a_subject_makes_a_blank_node() {
    assert_graph(
        HostLanguage::Html5,
        &html_with(
            r#"prefix="s: http://schema.org/""#,
            r#"<div typeof="s:Person"><span property="s:name">Sōseki</span></div>"#,
        ),
        &[
            "_:_ <http://www.w3.org/1999/02/22-rdf-syntax-ns#type> <http://schema.org/Person>",
            "_:_ <http://schema.org/name> \"Sōseki\"",
        ],
    );
}

#[test]
fn a_named_blank_node_is_shared_across_the_document() {
    assert_graph(
        HostLanguage::Html5,
        &html(
            r#"<p about="[_:a]" property="http://example.com/p">one</p>
               <p about="[_:a]" property="http://example.com/q">two</p>"#,
        ),
        &[
            "_:_ <http://example.com/p> \"one\"",
            "_:_ <http://example.com/q> \"two\"",
        ],
    );
}

#[test]
fn an_empty_safe_curie_names_nothing() {
    // `@about="[]"` is ignored, so the subject falls back to the enclosing one.
    assert_graph(
        HostLanguage::Html5,
        &html(r#"<p about="[]" property="http://example.com/p">x</p>"#),
        &["<http://example.com/doc> <http://example.com/p> \"x\""],
    );
}

// ----------------------------------------------------------- rel, rev, chaining

#[test]
fn rel_and_href_make_a_resource_triple() {
    assert_graph(
        HostLanguage::Html5,
        &html(
            r#"<a about="http://example.com/a" rel="http://example.com/p"
                  href="http://example.com/b">b</a>"#,
        ),
        &["<http://example.com/a> <http://example.com/p> <http://example.com/b>"],
    );
}

#[test]
fn rev_reverses_the_triple() {
    assert_graph(
        HostLanguage::Html5,
        &html(
            r#"<a about="http://example.com/a" rev="http://example.com/p"
                  href="http://example.com/b">b</a>"#,
        ),
        &["<http://example.com/b> <http://example.com/p> <http://example.com/a>"],
    );
}

#[test]
fn a_hanging_rel_is_completed_by_every_nested_subject() {
    assert_graph(
        HostLanguage::Html5,
        &html(
            r#"<div about="http://example.com/a" rel="http://example.com/p">
                 <span about="http://example.com/b"></span>
                 <span about="http://example.com/c"></span>
               </div>"#,
        ),
        &[
            "<http://example.com/a> <http://example.com/p> <http://example.com/b>",
            "<http://example.com/a> <http://example.com/p> <http://example.com/c>",
        ],
    );
}

#[test]
fn an_object_resource_becomes_the_subject_of_nested_statements() {
    assert_graph(
        HostLanguage::Html5,
        &html(
            r#"<div about="http://example.com/a" rel="http://example.com/p"
                    resource="http://example.com/b">
                 <span property="http://example.com/q">nested</span>
               </div>"#,
        ),
        &[
            "<http://example.com/a> <http://example.com/p> <http://example.com/b>",
            "<http://example.com/b> <http://example.com/q> \"nested\"",
        ],
    );
}

#[test]
fn a_bare_term_in_rel_is_dropped_when_property_is_on_the_same_element() {
    // HTML+RDFa 1.1 §3.1. `license` is a term, so it disappears and `@rel` counts as absent.
    assert_graph(
        HostLanguage::Html5,
        &html(
            r#"<a about="http://example.com/a" property="http://example.com/p" rel="license"
                  href="http://example.com/b">text</a>"#,
        ),
        &["<http://example.com/a> <http://example.com/p> <http://example.com/b>"],
    );
}

// ------------------------------------------------------------ prefixes, terms

#[test]
fn the_initial_context_supplies_common_prefixes() {
    assert_graph(
        HostLanguage::Html5,
        &html(r#"<p about="http://example.com/a" property="foaf:name">Sōseki</p>"#),
        &["<http://example.com/a> <http://xmlns.com/foaf/0.1/name> \"Sōseki\""],
    );
}

#[test]
fn legacy_xmlns_prefixes_still_work() {
    assert_graph(
        HostLanguage::Xml,
        r#"<root xmlns:ex="http://example.org/" about="http://example.com/a">
             <p property="ex:p">x</p></root>"#,
        &["<http://example.com/a> <http://example.org/p> \"x\""],
    );
}

#[test]
fn prefixes_are_case_insensitive_and_innermost_wins() {
    assert_graph(
        HostLanguage::Html5,
        &html_with(
            r#"prefix="EX: http://outer.example/""#,
            r#"<p about="http://example.com/a" property="ex:p">outer</p>
               <div prefix="ex: http://inner.example/">
                 <p about="http://example.com/a" property="ex:p">inner</p></div>"#,
        ),
        &[
            "<http://example.com/a> <http://outer.example/p> \"outer\"",
            "<http://example.com/a> <http://inner.example/p> \"inner\"",
        ],
    );
}

#[test]
fn vocab_expands_terms_and_announces_itself() {
    assert_graph(
        HostLanguage::Html5,
        &html(
            r#"<div vocab="http://schema.org/" about="http://example.com/a">
                 <span property="name">Sōseki</span></div>"#,
        ),
        &[
            "<http://example.com/doc> <http://www.w3.org/ns/rdfa#usesVocabulary> \
             <http://schema.org/>",
            "<http://example.com/a> <http://schema.org/name> \"Sōseki\"",
        ],
    );
}

#[test]
fn the_xhtml_host_language_adds_the_xhtml_terms() {
    let markup = "<?xml version=\"1.0\"?>\n\
                  <html xmlns=\"http://www.w3.org/1999/xhtml\"><head><title>t</title>\
                  <link rel=\"next\" href=\"http://example.com/b\"/></head><body/></html>";
    assert_graph(
        HostLanguage::Xhtml1,
        markup,
        &[
            "<http://example.com/doc> <http://www.w3.org/1999/xhtml/vocab#next> \
           <http://example.com/b>",
        ],
    );
    // The same document in the HTML host language has no `next` term, so nothing is generated.
    assert_graph(HostLanguage::Xhtml5, markup, &[]);
}

// ------------------------------------------------------------------- lists

#[test]
fn inlist_builds_an_rdf_collection() {
    assert_graph(
        HostLanguage::Html5,
        &html(
            r#"<div about="http://example.com/a">
                 <p property="http://example.com/p" inlist="">one</p>
                 <p property="http://example.com/p" inlist="">two</p>
               </div>"#,
        ),
        &[
            "<http://example.com/a> <http://example.com/p> _:_",
            "_:_ <http://www.w3.org/1999/02/22-rdf-syntax-ns#first> \"one\"",
            "_:_ <http://www.w3.org/1999/02/22-rdf-syntax-ns#first> \"two\"",
            "_:_ <http://www.w3.org/1999/02/22-rdf-syntax-ns#rest> _:_",
            "_:_ <http://www.w3.org/1999/02/22-rdf-syntax-ns#rest> \
             <http://www.w3.org/1999/02/22-rdf-syntax-ns#nil>",
        ],
    );
}

#[test]
fn an_empty_inlist_is_rdf_nil() {
    assert_graph(
        HostLanguage::Html5,
        &html(r#"<div about="http://example.com/a" rel="http://example.com/p" inlist=""></div>"#),
        &["<http://example.com/a> <http://example.com/p> \
             <http://www.w3.org/1999/02/22-rdf-syntax-ns#nil>"],
    );
}

#[test]
fn collections_with_the_same_predicate_but_different_subjects_stay_apart() {
    assert_graph(
        HostLanguage::Html5,
        &html(
            r#"<div about="http://example.com/a">
                 <p property="http://example.com/p" inlist="">outer</p>
                 <span rel="http://example.com/link" resource="http://example.com/b">
                   <p property="http://example.com/p" inlist="">inner</p>
                 </span>
               </div>"#,
        ),
        &[
            "<http://example.com/a> <http://example.com/p> _:_",
            "<http://example.com/a> <http://example.com/link> <http://example.com/b>",
            "<http://example.com/b> <http://example.com/p> _:_",
            "_:_ <http://www.w3.org/1999/02/22-rdf-syntax-ns#first> \"outer\"",
            "_:_ <http://www.w3.org/1999/02/22-rdf-syntax-ns#first> \"inner\"",
            "_:_ <http://www.w3.org/1999/02/22-rdf-syntax-ns#rest> \
             <http://www.w3.org/1999/02/22-rdf-syntax-ns#nil>",
        ],
    );
}

// ------------------------------------------------------------- base handling

#[test]
fn the_base_element_wins_over_the_supplied_base() {
    assert_graph(
        HostLanguage::Html5,
        "<!DOCTYPE html>\n<html><head><base href=\"http://other.example/page\"/></head>\
         <body><p about=\"\" property=\"http://example.com/p\">x</p></body></html>",
        &["<http://other.example/page> <http://example.com/p> \"x\""],
    );
}

#[test]
fn xml_base_is_honoured_in_xml_and_ignored_in_html() {
    assert_graph(
        HostLanguage::Xml,
        r#"<root xml:base="http://other.example/page">
             <p about="" property="http://example.com/p">x</p></root>"#,
        &["<http://other.example/page> <http://example.com/p> \"x\""],
    );
    assert_graph(
        HostLanguage::Html5,
        &html(
            r#"<div xml:base="http://other.example/page">
                 <p about="" property="http://example.com/p">x</p></div>"#,
        ),
        &["<http://example.com/doc> <http://example.com/p> \"x\""],
    );
}

// ------------------------------------------------------------------- @role

#[test]
fn role_names_its_subject_with_the_element_id() {
    assert_graph(
        HostLanguage::Html5,
        &html(r#"<div id="heading1" role="heading">h</div>"#),
        &[
            "<http://example.com/doc#heading1> <http://www.w3.org/1999/xhtml/vocab#role> \
             <http://www.w3.org/1999/xhtml/vocab#heading>",
        ],
    );
}

#[test]
fn role_takes_terms_curies_and_iris_and_ignores_vocab() {
    assert_graph(
        HostLanguage::Html5,
        &html_with(
            r#"vocab="http://schema.org/""#,
            r#"<div prefix="ex: http://example.org/roles/" id="r"
                    role="ex:one two http://example.org/three">x</div>"#,
        ),
        &[
            "<http://example.com/doc> <http://www.w3.org/ns/rdfa#usesVocabulary> \
             <http://schema.org/>",
            "<http://example.com/doc#r> <http://www.w3.org/1999/xhtml/vocab#role> \
             <http://example.org/roles/one>",
            "<http://example.com/doc#r> <http://www.w3.org/1999/xhtml/vocab#role> \
             <http://www.w3.org/1999/xhtml/vocab#two>",
            "<http://example.com/doc#r> <http://www.w3.org/1999/xhtml/vocab#role> \
             <http://example.org/three>",
        ],
    );
}

// -------------------------------------------------------------- property copying

#[test]
fn rdfa_copy_pulls_a_pattern_into_the_referring_resource() {
    assert_graph(
        HostLanguage::Html5,
        &html_with(
            r#"vocab="http://schema.org/""#,
            r#"<div about="http://example.com/a" typeof="Person">
                 <link property="rdfa:copy" href="http://example.com/p1"/>
               </div>
               <div resource="http://example.com/p1" typeof="rdfa:Pattern">
                 <span property="name">Sōseki</span>
               </div>"#,
        ),
        &[
            "<http://example.com/doc> <http://www.w3.org/ns/rdfa#usesVocabulary> \
             <http://schema.org/>",
            "<http://example.com/a> <http://www.w3.org/1999/02/22-rdf-syntax-ns#type> \
             <http://schema.org/Person>",
            "<http://example.com/a> <http://schema.org/name> \"Sōseki\"",
        ],
    );
}

#[test]
fn property_copying_does_not_happen_outside_html() {
    let graph = extract(
        r#"<root vocab="http://schema.org/">
             <div about="http://example.com/a" typeof="Person">
               <link property="rdfa:copy" href="http://example.com/p1"/>
             </div>
             <div resource="http://example.com/p1" typeof="rdfa:Pattern">
               <span property="name">Sōseki</span>
             </div>
           </root>"#,
        HostLanguage::Xml,
    );
    assert!(
        graph.contains(
            "<http://example.com/a> <http://www.w3.org/ns/rdfa#copy> <http://example.com/p1>"
        ),
        "{graph:?}"
    );
    assert!(
        !graph.contains("<http://example.com/a> <http://schema.org/name> \"Sōseki\""),
        "{graph:?}"
    );
}

// -------------------------------------------------------------------- errors

#[test]
fn a_bad_base_iri_is_rejected() {
    assert!(RdfaParser::new().with_base_iri("not an iri").is_err());
}

#[test]
fn malformed_xml_surfaces_as_one_error() {
    let mut triples = RdfaParser::new()
        .with_host_language(HostLanguage::Xml)
        .for_slice("<root><unclosed></root>");
    assert!(triples.next().expect("an item").is_err());
    assert!(triples.next().is_none());
}

#[test]
fn a_document_with_no_rdfa_yields_nothing() {
    assert_graph(HostLanguage::Html5, &html("<p>just prose</p>"), &[]);
}
