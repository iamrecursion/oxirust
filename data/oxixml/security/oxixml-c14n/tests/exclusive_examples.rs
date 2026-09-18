//! The worked examples of the Exclusive XML Canonicalization Recommendation.
//!
//! [Exclusive C14N §2](https://www.w3.org/TR/xml-exc-c14n/#sec-Motivation) makes
//! its case by showing the same element canonicalized inclusively in two
//! different envelopes — producing two different octet sequences, and therefore
//! two different digests — and then exclusively, producing one. Both halves are
//! reproduced here: the inclusive expectations pin down [`c14n_1_0`] on document
//! subsets, and the exclusive ones pin down [`exclusive_c14n`].
//!
//! The Recommendation wraps its listings for readability; the inputs below use
//! explicit, unambiguous white space instead, and the expected outputs follow
//! from it.

mod common;

use common::{element_named, parse, text};
use oxixml_c14n::{InclusiveNamespaces, NodeSet, c14n_1_0, exclusive_c14n};
use oxixml_dom::Document;

/// The node-set `(//. | //@* | //namespace::*)[ancestor-or-self::E]`, i.e. one
/// element subtree with its attributes and namespace nodes.
fn subtree(document: &Document, name: &str) -> NodeSet {
    NodeSet::subtree(document, element_named(document, name))
}

/// §2.1 — a document that is already in canonical form.
#[test]
fn simple_example_unenveloped() {
    let doc = parse("<n1:elem1 xmlns:n1=\"http://b.example\">content</n1:elem1>");
    let canonical = c14n_1_0(&doc, &NodeSet::document(&doc), false).expect("canonicalization");
    assert_eq!(
        text(&canonical),
        "<n1:elem1 xmlns:n1=\"http://b.example\">content</n1:elem1>"
    );
}

/// §2.1 — once enveloped, inclusive canonicalization drags `n0` in from the
/// context and the signature over the unenveloped form breaks.
#[test]
fn simple_example_inclusive_pulls_in_the_envelope() {
    let doc = parse(concat!(
        "<n0:pdu xmlns:n0=\"http://a.example\">",
        "<n1:elem1 xmlns:n1=\"http://b.example\">content</n1:elem1>",
        "</n0:pdu>",
    ));
    let canonical = c14n_1_0(&doc, &subtree(&doc, "n1:elem1"), false).expect("canonicalization");
    assert_eq!(
        text(&canonical),
        concat!(
            "<n1:elem1 xmlns:n0=\"http://a.example\" xmlns:n1=\"http://b.example\">",
            "content</n1:elem1>",
        )
    );
}

/// §2.1 — exclusive canonicalization ignores the envelope entirely.
#[test]
fn simple_example_exclusive_ignores_the_envelope() {
    let doc = parse(concat!(
        "<n0:pdu xmlns:n0=\"http://a.example\">",
        "<n1:elem1 xmlns:n1=\"http://b.example\">content</n1:elem1>",
        "</n0:pdu>",
    ));
    let canonical = exclusive_c14n(
        &doc,
        &subtree(&doc, "n1:elem1"),
        &InclusiveNamespaces::none(),
        false,
    )
    .expect("canonicalization");
    assert_eq!(
        text(&canonical),
        "<n1:elem1 xmlns:n1=\"http://b.example\">content</n1:elem1>"
    );
}

/// The first envelope of §2.2.
const ENVELOPE_A: &str = concat!(
    "<n0:local xmlns:n0=\"foo:bar\" xmlns:n3=\"ftp://example.org\">",
    "<n1:elem2 xmlns:n1=\"http://example.net\" xml:lang=\"en\">",
    "<n3:stuff xmlns:n3=\"ftp://example.org\"/>",
    "</n1:elem2>",
    "</n0:local>",
);

/// The second envelope of §2.2: same `elem2`, completely different context.
const ENVELOPE_B: &str = concat!(
    "<n2:pdu xmlns:n1=\"http://example.com\" xmlns:n2=\"http://foo.example\" ",
    "xml:lang=\"fr\" xml:space=\"retain\">",
    "<n1:elem2 xmlns:n1=\"http://example.net\" xml:lang=\"en\">",
    "<n3:stuff xmlns:n3=\"ftp://example.org\"/>",
    "</n1:elem2>",
    "</n2:pdu>",
);

/// §2.2 — inclusive canonicalization of the first envelope. `n0` is imported
/// from the context and the identical `n3` declaration on `stuff` is elevated to
/// the apex, where it becomes superfluous on the child.
#[test]
fn re_enveloping_inclusive_first_context() {
    let doc = parse(ENVELOPE_A);
    let canonical = c14n_1_0(&doc, &subtree(&doc, "n1:elem2"), false).expect("canonicalization");
    assert_eq!(
        text(&canonical),
        concat!(
            "<n1:elem2 xmlns:n0=\"foo:bar\" xmlns:n1=\"http://example.net\" ",
            "xmlns:n3=\"ftp://example.org\" xml:lang=\"en\">",
            "<n3:stuff></n3:stuff></n1:elem2>",
        )
    );
}

/// §2.2 — inclusive canonicalization of the second envelope: `n0` is gone, `n2`
/// has appeared, `n3` is no longer elevated and `xml:space` has been imported.
#[test]
fn re_enveloping_inclusive_second_context() {
    let doc = parse(ENVELOPE_B);
    let canonical = c14n_1_0(&doc, &subtree(&doc, "n1:elem2"), false).expect("canonicalization");
    assert_eq!(
        text(&canonical),
        concat!(
            "<n1:elem2 xmlns:n1=\"http://example.net\" xmlns:n2=\"http://foo.example\" ",
            "xml:lang=\"en\" xml:space=\"retain\">",
            "<n3:stuff xmlns:n3=\"ftp://example.org\"></n3:stuff></n1:elem2>",
        )
    );
}

/// §2.2 — exclusive canonicalization produces the same octets in both
/// envelopes, which is the whole point of the algorithm.
#[test]
fn re_enveloping_exclusive_is_context_independent() {
    let expected = concat!(
        "<n1:elem2 xmlns:n1=\"http://example.net\" xml:lang=\"en\">",
        "<n3:stuff xmlns:n3=\"ftp://example.org\"></n3:stuff></n1:elem2>",
    );
    for input in [ENVELOPE_A, ENVELOPE_B] {
        let doc = parse(input);
        let canonical = exclusive_c14n(
            &doc,
            &subtree(&doc, "n1:elem2"),
            &InclusiveNamespaces::none(),
            false,
        )
        .expect("canonicalization");
        assert_eq!(text(&canonical), expected, "input: {input}");
    }
}

/// §1.3 — a prefix that only appears inside content is invisible to exclusive
/// canonicalization; the `InclusiveNamespaces` `PrefixList` is the escape hatch.
#[test]
fn inclusive_namespaces_prefix_list_rescues_qnames_in_content() {
    let doc = parse(concat!(
        "<env xmlns:xsi=\"http://www.w3.org/2001/XMLSchema-instance\" ",
        "xmlns:xsd=\"http://www.w3.org/2001/XMLSchema\">",
        "<number xsi:type=\"xsd:decimal\">10.09</number>",
        "</env>",
    ));
    let subset = subtree(&doc, "number");

    // Without the list, `xsd` is lost: nothing in a *name* uses it.
    let lost = exclusive_c14n(&doc, &subset, &InclusiveNamespaces::none(), false)
        .expect("canonicalization");
    assert_eq!(
        text(&lost),
        concat!(
            "<number xmlns:xsi=\"http://www.w3.org/2001/XMLSchema-instance\" ",
            "xsi:type=\"xsd:decimal\">10.09</number>",
        )
    );

    // With it, the declaration is rendered by the inclusive rules.
    let kept = exclusive_c14n(&doc, &subset, &InclusiveNamespaces::parse("xsd"), false)
        .expect("canonicalization");
    assert_eq!(
        text(&kept),
        concat!(
            "<number xmlns:xsd=\"http://www.w3.org/2001/XMLSchema\" ",
            "xmlns:xsi=\"http://www.w3.org/2001/XMLSchema-instance\" ",
            "xsi:type=\"xsd:decimal\">10.09</number>",
        )
    );
}

/// §3.1 — `xmlns=""` is rendered only when the element visibly utilizes the
/// default namespace *and* an output ancestor put a non-empty one in scope.
#[test]
fn exclusive_renders_xmlns_empty_only_to_cancel_an_ancestor() {
    // The apex declares a default namespace, the child undeclares it.
    let doc = parse("<outer xmlns=\"urn:d\"><inner xmlns=\"\"/></outer>");
    let canonical = exclusive_c14n(
        &doc,
        &NodeSet::document(&doc),
        &InclusiveNamespaces::none(),
        false,
    )
    .expect("canonicalization");
    assert_eq!(
        text(&canonical),
        "<outer xmlns=\"urn:d\"><inner xmlns=\"\"></inner></outer>"
    );

    // Starting the subset at the child leaves nothing to cancel.
    let canonical = exclusive_c14n(
        &doc,
        &subtree(&doc, "inner"),
        &InclusiveNamespaces::none(),
        false,
    )
    .expect("canonicalization");
    assert_eq!(text(&canonical), "<inner></inner>");
}

/// §2.4.1 of Canonical XML 2.0 describes the same "push down" behaviour that
/// Exclusive C14N has: a declaration made high up is emitted once in each
/// element that actually uses it.
#[test]
fn unused_declarations_are_pushed_down_to_each_user() {
    let doc = parse(concat!(
        "<wsse:Security xmlns:wsse=\"urn:secext\" xmlns:wsu=\"urn:utility\">",
        "<wsse:UserName wsu:Id=\"i1\"/>",
        "<wsse:Timestamp wsu:Id=\"i2\"/>",
        "</wsse:Security>",
    ));
    let canonical = exclusive_c14n(
        &doc,
        &NodeSet::document(&doc),
        &InclusiveNamespaces::none(),
        false,
    )
    .expect("canonicalization");
    assert_eq!(
        text(&canonical),
        concat!(
            "<wsse:Security xmlns:wsse=\"urn:secext\">",
            "<wsse:UserName xmlns:wsu=\"urn:utility\" wsu:Id=\"i1\"></wsse:UserName>",
            "<wsse:Timestamp xmlns:wsu=\"urn:utility\" wsu:Id=\"i2\"></wsse:Timestamp>",
            "</wsse:Security>",
        )
    );
}

/// Exclusive canonicalization never imports `xml:*` attributes, no matter how
/// deep the omitted context is.
#[test]
fn exclusive_never_imports_xml_attributes() {
    let doc = parse(concat!(
        "<outer xml:lang=\"fr\" xml:space=\"preserve\" xml:base=\"http://example.com/\">",
        "<inner/>",
        "</outer>",
    ));
    let canonical = exclusive_c14n(
        &doc,
        &subtree(&doc, "inner"),
        &InclusiveNamespaces::none(),
        false,
    )
    .expect("canonicalization");
    assert_eq!(text(&canonical), "<inner></inner>");

    // Canonical XML 1.0 does import them.
    let inclusive = c14n_1_0(&doc, &subtree(&doc, "inner"), false).expect("canonicalization");
    assert_eq!(
        text(&inclusive),
        concat!(
            "<inner xml:base=\"http://example.com/\" xml:lang=\"fr\" ",
            "xml:space=\"preserve\"></inner>",
        )
    );
}

/// The `#default` token puts the default namespace declaration back under the
/// inclusive rules.
#[test]
fn the_default_token_makes_the_default_namespace_inclusive() {
    let doc = parse("<a:outer xmlns=\"urn:d\" xmlns:a=\"urn:a\"><a:inner/></a:outer>");
    let subset = subtree(&doc, "a:inner");

    let without = exclusive_c14n(&doc, &subset, &InclusiveNamespaces::none(), false)
        .expect("canonicalization");
    assert_eq!(text(&without), "<a:inner xmlns:a=\"urn:a\"></a:inner>");

    let with = exclusive_c14n(
        &doc,
        &subset,
        &InclusiveNamespaces::parse("#default"),
        false,
    )
    .expect("canonicalization");
    assert_eq!(
        text(&with),
        "<a:inner xmlns=\"urn:d\" xmlns:a=\"urn:a\"></a:inner>"
    );
}
