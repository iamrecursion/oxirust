//! Canonical XML 2.0 behaviour, without needing the W3C suite on disk.
//!
//! The vectors reproduced here are taken from the
//! [Canonical XML 2.0 test cases](https://www.w3.org/TR/xml-c14n2-testcases/)
//! and from the examples in [§2.5](https://www.w3.org/TR/xml-c14n2/#sec-Namespace-Processing)
//! of the specification itself, so that the 2.0 implementation keeps a
//! self-contained conformance floor. `tests/w3c_c14n2_testcases.rs` runs the
//! full published suite when it is available.

mod common;

use common::{element_named, parse, text};
use oxixml_c14n::{C14n2Parameters, C14n2Subset, PrefixRewrite, QNameAware, c14n_2_0};

/// The parameter set the suite calls `c14nDefault`.
fn defaults() -> C14n2Parameters {
    C14n2Parameters::new().with_trim_text_nodes(false)
}

/// The parameter set the suite calls `c14nPrefix`.
fn sequential() -> C14n2Parameters {
    defaults().with_prefix_rewrite(PrefixRewrite::Sequential)
}

fn run(input: &str, parameters: &C14n2Parameters) -> String {
    let document = parse(input);
    let canonical =
        c14n_2_0(&document, &C14n2Subset::document(), parameters).expect("canonicalization");
    text(&canonical)
}

/// `inNsPushdown` — a declaration made on the apex but used only by the
/// children is emitted once in each child, not once at the top.
#[test]
fn namespace_declarations_are_pushed_down() {
    let input = concat!(
        "<a:foo xmlns:a=\"http://a\" xmlns:b=\"http://b\" xmlns:c=\"http://c\">\n",
        " <b:bar/>\n <b:bar/>\n <b:bar/>\n <a:bar b:att1=\"val\"/>\n</a:foo>\n",
    );
    assert_eq!(
        run(input, &defaults()),
        concat!(
            "<a:foo xmlns:a=\"http://a\">\n",
            " <b:bar xmlns:b=\"http://b\"></b:bar>\n",
            " <b:bar xmlns:b=\"http://b\"></b:bar>\n",
            " <b:bar xmlns:b=\"http://b\"></b:bar>\n",
            " <a:bar xmlns:b=\"http://b\" b:att1=\"val\"></a:bar>\n",
            "</a:foo>",
        )
    );
    assert_eq!(
        run(input, &sequential()),
        concat!(
            "<n0:foo xmlns:n0=\"http://a\">\n",
            " <n1:bar xmlns:n1=\"http://b\"></n1:bar>\n",
            " <n1:bar xmlns:n1=\"http://b\"></n1:bar>\n",
            " <n1:bar xmlns:n1=\"http://b\"></n1:bar>\n",
            " <n0:bar xmlns:n1=\"http://b\" n1:att1=\"val\"></n0:bar>\n",
            "</n0:foo>",
        )
    );
}

/// `inNsSort` — namespace declarations sort by prefix, attributes by namespace
/// name then local name, and rewritten prefixes are handed out in ascending
/// order of namespace name.
#[test]
fn declarations_and_attributes_sort_independently() {
    let input = concat!(
        "<a:foo xmlns:a=\"http://z3\" xmlns:b=\"http://z2\" b:att1=\"val1\" c:att3=\"val3\" ",
        "b:att2=\"val2\" xmlns:c=\"http://z1\" xmlns:d=\"http://z0\">\n",
        " <c:bar/>\n <c:bar d:att3=\"val3\"/>\n</a:foo>\n",
    );
    assert_eq!(
        run(input, &defaults()),
        concat!(
            "<a:foo xmlns:a=\"http://z3\" xmlns:b=\"http://z2\" xmlns:c=\"http://z1\" ",
            "c:att3=\"val3\" b:att1=\"val1\" b:att2=\"val2\">\n",
            " <c:bar></c:bar>\n",
            " <c:bar xmlns:d=\"http://z0\" d:att3=\"val3\"></c:bar>\n",
            "</a:foo>",
        )
    );
    assert_eq!(
        run(input, &sequential()),
        concat!(
            "<n2:foo xmlns:n0=\"http://z1\" xmlns:n1=\"http://z2\" xmlns:n2=\"http://z3\" ",
            "n0:att3=\"val3\" n1:att1=\"val1\" n1:att2=\"val2\">\n",
            " <n0:bar></n0:bar>\n",
            " <n0:bar xmlns:n3=\"http://z0\" n3:att3=\"val3\"></n0:bar>\n",
            "</n2:foo>",
        )
    );
}

/// `inNsSuperfluous` — several prefixes bound to one namespace name collapse
/// onto a single rewritten prefix.
#[test]
fn several_prefixes_for_one_namespace_collapse_when_rewritten() {
    let input = concat!(
        "<foo xmlns:a=\"http://z0\" xmlns:b=\"http://z0\" a:att1=\"val1\" b:att2=\"val2\" ",
        "xmlns=\"http://z0\"> \n",
        " <c:bar xmlns:a=\"http://z0\" xmlns:c=\"http://z0\" c:att3=\"val3\"/>\n",
        " <d:bar xmlns:d=\"http://z0\"/>\n</foo>\n",
    );
    assert_eq!(
        run(input, &sequential()),
        concat!(
            "<n0:foo xmlns:n0=\"http://z0\" n0:att1=\"val1\" n0:att2=\"val2\"> \n",
            " <n0:bar n0:att3=\"val3\"></n0:bar>\n",
            " <n0:bar></n0:bar>\n",
            "</n0:foo>",
        )
    );
}

/// `inNsXml` — the `xml` prefix is never declared and never rewritten, and a
/// `QName` hiding in an attribute value is only seen when `QNameAware` says so.
#[test]
fn qname_aware_attributes() {
    let input = concat!(
        "<foo xmlns=\"http://z0\" xml:id=\"23\">\n",
        "  <bar xsi:type=\"xsd:string\" ",
        "xmlns:xsi=\"http://www.w3.org/2001/XMLSchema-instance\" ",
        "xmlns:xsd=\"http://www.w3.org/2001/XMLSchema\">data</bar>\n</foo>\n",
    );

    // Without `QNameAware`, `xsd` is not visibly utilized and is dropped.
    assert_eq!(
        run(input, &defaults()),
        concat!(
            "<foo xmlns=\"http://z0\" xml:id=\"23\">\n",
            "  <bar xmlns:xsi=\"http://www.w3.org/2001/XMLSchema-instance\" ",
            "xsi:type=\"xsd:string\">data</bar>\n</foo>",
        )
    );

    let mut aware = QNameAware::new();
    aware.add_qualified_attribute("http://www.w3.org/2001/XMLSchema-instance", "type");

    // With it, `xsd` survives — but the value is only rewritten when prefixes
    // are being rewritten at all.
    assert_eq!(
        run(input, &defaults().with_qname_aware(aware.clone())),
        concat!(
            "<foo xmlns=\"http://z0\" xml:id=\"23\">\n",
            "  <bar xmlns:xsd=\"http://www.w3.org/2001/XMLSchema\" ",
            "xmlns:xsi=\"http://www.w3.org/2001/XMLSchema-instance\" ",
            "xsi:type=\"xsd:string\">data</bar>\n</foo>",
        )
    );
    assert_eq!(
        run(input, &sequential().with_qname_aware(aware)),
        concat!(
            "<n0:foo xmlns:n0=\"http://z0\" xml:id=\"23\">\n",
            "  <n0:bar xmlns:n1=\"http://www.w3.org/2001/XMLSchema\" ",
            "xmlns:n2=\"http://www.w3.org/2001/XMLSchema-instance\" ",
            "n2:type=\"n1:string\">data</n0:bar>\n</n0:foo>",
        )
    );
}

/// `inNsContent` — an element whose content is a `QName`, and an element whose
/// content is an XPath expression whose prefixes must survive and be rewritten.
#[test]
fn qname_aware_and_xpath_aware_elements() {
    let input = concat!(
        "<a:foo xmlns:a=\"http://a\" xmlns:b=\"http://b\" xmlns:child=\"http://c\" ",
        "xmlns:soap-env=\"http://schemas.xmlsoap.org/wsdl/soap/\" ",
        "xmlns:xsd=\"http://www.w3.org/2001/XMLSchema\">\n",
        " <a:bar>xsd:string</a:bar>\n",
        " <dsig2:IncludedXPath xmlns:dsig2=\"http://www.w3.org/2010/xmldsig2#\">",
        "/soap-env:body/child::b:foo[@att1 != \"c:val\" and @att2 != 'xsd:string']",
        "</dsig2:IncludedXPath>\n</a:foo>\n",
    );

    let mut aware = QNameAware::new();
    aware.add_element("http://a", "bar");
    aware.add_xpath_element("http://www.w3.org/2010/xmldsig2#", "IncludedXPath");

    assert_eq!(
        run(input, &defaults().with_qname_aware(aware.clone())),
        concat!(
            "<a:foo xmlns:a=\"http://a\">\n",
            " <a:bar xmlns:xsd=\"http://www.w3.org/2001/XMLSchema\">xsd:string</a:bar>\n",
            " <dsig2:IncludedXPath xmlns:b=\"http://b\" ",
            "xmlns:dsig2=\"http://www.w3.org/2010/xmldsig2#\" ",
            "xmlns:soap-env=\"http://schemas.xmlsoap.org/wsdl/soap/\">",
            "/soap-env:body/child::b:foo[@att1 != \"c:val\" and @att2 != 'xsd:string']",
            "</dsig2:IncludedXPath>\n</a:foo>",
        )
    );
    assert_eq!(
        run(input, &sequential().with_qname_aware(aware)),
        concat!(
            "<n0:foo xmlns:n0=\"http://a\">\n",
            " <n0:bar xmlns:n1=\"http://www.w3.org/2001/XMLSchema\">n1:string</n0:bar>\n",
            " <n4:IncludedXPath xmlns:n2=\"http://b\" ",
            "xmlns:n3=\"http://schemas.xmlsoap.org/wsdl/soap/\" ",
            "xmlns:n4=\"http://www.w3.org/2010/xmldsig2#\">",
            "/n3:body/child::n2:foo[@att1 != \"c:val\" and @att2 != 'xsd:string']",
            "</n4:IncludedXPath>\n</n0:foo>",
        )
    );
}

/// `TrimTextNodes` removes the leading and trailing white space of each
/// coalesced text run — unless `xml:space="preserve"` is in context.
#[test]
fn text_trimming_respects_xml_space() {
    let input = concat!(
        "<doc>\n   <clean>   </clean>\n   <dirty>   A   B   </dirty>\n",
        "   <mixed>\n      A\n      <clean>   </clean>\n      B\n   </mixed>\n</doc>\n",
    );
    assert_eq!(
        run(input, &defaults().with_trim_text_nodes(true)),
        "<doc><clean></clean><dirty>A   B</dirty><mixed>A<clean></clean>B</mixed></doc>"
    );

    let preserved = "<doc xml:space=\"preserve\">   <a>  keep  </a>   </doc>";
    assert_eq!(
        run(preserved, &defaults().with_trim_text_nodes(true)),
        "<doc xml:space=\"preserve\">   <a>  keep  </a>   </doc>"
    );
}

/// `IgnoreComments` is what its name says, and the line feeds around top-level
/// nodes follow Canonical XML 1.0.
#[test]
fn comments_and_top_level_line_feeds() {
    let input = concat!(
        "<?xml version=\"1.0\"?>\n\n<?xml-stylesheet   href=\"doc.xsl\"?>\n\n",
        "<doc>Hello, world!<!-- Comment 1 --></doc>\n\n<!-- Comment 2 -->\n",
    );
    assert_eq!(
        run(input, &defaults()),
        "<?xml-stylesheet href=\"doc.xsl\"?>\n<doc>Hello, world!</doc>"
    );
    assert_eq!(
        run(input, &defaults().with_ignore_comments(false)),
        concat!(
            "<?xml-stylesheet href=\"doc.xsl\"?>\n",
            "<doc>Hello, world!<!-- Comment 1 --></doc>\n",
            "<!-- Comment 2 -->",
        )
    );
}

/// The inclusion and exclusion lists of §2.1.
#[test]
fn inclusion_and_exclusion_lists() {
    let document =
        parse("<r xmlns:p=\"urn:p\"><a p:keep=\"1\" drop=\"2\">A</a><b>B</b><c>C</c></r>");
    let a = element_named(&document, "a");
    let b = element_named(&document, "b");
    let c = element_named(&document, "c");

    // Two apex nodes, concatenated in document order however they were listed.
    let subset = C14n2Subset::elements([c, a]);
    let canonical = c14n_2_0(&document, &subset, &defaults()).expect("canonicalization");
    assert_eq!(
        text(&canonical),
        "<a xmlns:p=\"urn:p\" drop=\"2\" p:keep=\"1\">A</a><c>C</c>"
    );

    // Excluding a subtree prunes it; excluding an attribute takes its prefix's
    // declaration with it.
    let mut subset = C14n2Subset::document();
    subset.exclude_element(b);
    let drop_attribute = document
        .get_attribute_node(a, "p:keep")
        .expect("the attribute exists");
    subset.exclude_attribute(drop_attribute);
    let canonical = c14n_2_0(&document, &subset, &defaults()).expect("canonicalization");
    assert_eq!(text(&canonical), "<r><a drop=\"2\">A</a><c>C</c></r>");
}

/// An XPath node-set can be projected onto the restricted 2.0 input model,
/// which is how [`CanonicalizationAlgorithm::Canonical2_0`] is dispatched.
#[test]
fn node_sets_project_onto_the_2_0_input_model() {
    use oxixml_c14n::{CanonicalizationAlgorithm, NodeSet};

    let document = parse("<r xmlns:p=\"urn:p\"><a p:k=\"1\">A</a><b>B</b></r>");
    let a = element_named(&document, "a");
    let b = element_named(&document, "b");

    // The whole document, minus one element and one attribute.
    let mut node_set = NodeSet::document(&document);
    node_set.remove_subtree(&document, b);
    let attribute = document
        .get_attribute_node(a, "p:k")
        .expect("the attribute exists");
    node_set.remove(attribute);

    let projected = C14n2Subset::from_node_set(&document, &node_set);
    assert!(projected.is_whole_document());
    assert!(projected.is_excluded_element(b));
    assert!(projected.is_excluded_attribute(attribute));
    assert_eq!(
        text(&c14n_2_0(&document, &projected, &defaults()).expect("canonicalization")),
        "<r><a>A</a></r>"
    );

    // A subtree becomes an inclusion list of one apex.
    let subtree = NodeSet::subtree(&document, a);
    let projected = C14n2Subset::from_node_set(&document, &subtree);
    assert!(!projected.is_whole_document());
    assert_eq!(projected.apex_nodes(&document), vec![a]);

    // And that is exactly what dispatching on the algorithm URI does.
    assert_eq!(
        CanonicalizationAlgorithm::Canonical2_0
            .apply(&document, &subtree)
            .expect("canonicalization"),
        c14n_2_0(
            &document,
            &projected,
            &C14n2Parameters::new().with_trim_text_nodes(true),
        )
        .expect("canonicalization")
    );
}

/// Canonical XML 2.0 never imports `xml:*` attributes from the context, which is
/// what makes an apex node's output independent of its envelope.
#[test]
fn xml_attributes_are_not_imported() {
    let document = parse(concat!(
        "<outer xml:lang=\"fr\" xml:base=\"http://example.com/\" xmlns:p=\"urn:p\">",
        "<p:inner/></outer>",
    ));
    let inner = element_named(&document, "p:inner");
    let canonical = c14n_2_0(&document, &C14n2Subset::elements([inner]), &defaults())
        .expect("canonicalization");
    assert_eq!(text(&canonical), "<p:inner xmlns:p=\"urn:p\"></p:inner>");
}
