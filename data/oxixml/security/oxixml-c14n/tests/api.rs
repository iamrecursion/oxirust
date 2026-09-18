//! The public surface: algorithm dispatch, the convenience wrappers, node-set
//! manipulation and the error paths.

mod common;

use common::{element_named, parse, parse_nested, text};
use oxixml_c14n::{
    C14n2Parameters, C14n2Subset, C14nError, CanonicalizationAlgorithm, InclusiveNamespaces,
    NodeSet, c14n_1_0, c14n_1_0_document, c14n_1_0_subtree, c14n_2_0, exclusive_c14n,
    exclusive_c14n_subtree, join_uri_references,
};
use oxixml_dom::NodeKind;

const SAMPLE: &str = concat!(
    "<e:env xmlns:e=\"urn:e\" xmlns:x=\"urn:x\">",
    "<e:body id=\"b\">text<!-- note --></e:body>",
    "</e:env>",
);

#[test]
fn every_algorithm_uri_dispatches_to_a_working_implementation() {
    let document = parse(SAMPLE);
    let subset = NodeSet::document(&document);
    for algorithm in CanonicalizationAlgorithm::ALL {
        let recovered = CanonicalizationAlgorithm::from_uri(algorithm.uri())
            .expect("every advertised URI is recognised");
        assert_eq!(recovered, algorithm);
        let canonical = algorithm
            .apply(&document, &subset)
            .expect("canonicalization succeeds");
        let rendered = text(&canonical);
        assert!(rendered.starts_with("<e:env"), "{algorithm}: {rendered}");
        assert_eq!(
            rendered.contains("<!-- note -->"),
            algorithm.includes_comments(),
            "{algorithm}: comment handling"
        );
    }
}

#[test]
fn the_prefix_list_only_reaches_the_exclusive_variants() {
    let document = parse(SAMPLE);
    let body = element_named(&document, "e:body");
    let subset = NodeSet::subtree(&document, body);
    let list = InclusiveNamespaces::parse("x");

    // Exclusive without the list drops the unused `x` declaration …
    assert_eq!(
        text(
            &CanonicalizationAlgorithm::Exclusive1_0
                .apply(&document, &subset)
                .expect("canonicalization")
        ),
        "<e:body xmlns:e=\"urn:e\" id=\"b\">text</e:body>"
    );
    // … and with it, keeps it.
    assert_eq!(
        text(
            &CanonicalizationAlgorithm::Exclusive1_0
                .apply_with(&document, &subset, &list)
                .expect("canonicalization")
        ),
        "<e:body xmlns:e=\"urn:e\" xmlns:x=\"urn:x\" id=\"b\">text</e:body>"
    );
    // Inclusive canonicalization ignores the list; it renders the context anyway.
    assert_eq!(
        text(
            &CanonicalizationAlgorithm::Canonical1_0
                .apply_with(&document, &subset, &list)
                .expect("canonicalization")
        ),
        "<e:body xmlns:e=\"urn:e\" xmlns:x=\"urn:x\" id=\"b\">text</e:body>"
    );
}

#[test]
fn the_convenience_wrappers_agree_with_the_general_forms() {
    let document = parse(SAMPLE);
    let body = element_named(&document, "e:body");

    assert_eq!(
        c14n_1_0_document(&document, true).expect("canonicalization"),
        c14n_1_0(&document, &NodeSet::document(&document), true).expect("canonicalization")
    );
    assert_eq!(
        c14n_1_0_subtree(&document, body, false).expect("canonicalization"),
        c14n_1_0(&document, &NodeSet::subtree(&document, body), false).expect("canonicalization")
    );
    assert_eq!(
        exclusive_c14n_subtree(&document, body, &InclusiveNamespaces::none(), false)
            .expect("canonicalization"),
        exclusive_c14n(
            &document,
            &NodeSet::subtree(&document, body),
            &InclusiveNamespaces::none(),
            false
        )
        .expect("canonicalization")
    );
}

/// An XPath selection may keep an element while dropping one of its namespace
/// nodes; the node-set has to be able to say so, and the "superfluous
/// declaration" rule keys on node-set membership rather than on scope.
#[test]
fn namespace_nodes_can_be_excluded_individually() {
    let document = parse(SAMPLE);
    let env = element_named(&document, "e:env");
    let body = element_named(&document, "e:body");

    // Dropping the namespace node from the apex alone does not remove the
    // declaration: `body` still has one, and its nearest visible ancestor no
    // longer carries a matching node, so the declaration is pushed down.
    let mut subset = NodeSet::document(&document);
    assert!(subset.contains_namespace_node(env, "x"));
    subset.set_namespace_node(&document, env, "x", false);
    assert!(!subset.contains_namespace_node(env, "x"));
    assert_eq!(
        text(&c14n_1_0(&document, &subset, false).expect("canonicalization")),
        "<e:env xmlns:e=\"urn:e\"><e:body xmlns:x=\"urn:x\" id=\"b\">text</e:body></e:env>"
    );

    // Dropping it everywhere removes it from the canonical form.
    subset.set_namespace_node(&document, body, "x", false);
    assert_eq!(
        text(&c14n_1_0(&document, &subset, false).expect("canonicalization")),
        "<e:env xmlns:e=\"urn:e\"><e:body id=\"b\">text</e:body></e:env>"
    );
}

/// A CDATA section is "replaced with its character content", and the content is
/// then escaped like any other text.
#[test]
fn cdata_sections_become_escaped_text() {
    let document = parse("<a><![CDATA[<raw> & \"more\"]]></a>");
    assert_eq!(
        text(&c14n_1_0_document(&document, false).expect("canonicalization")),
        "<a>&lt;raw&gt; &amp; \"more\"</a>"
    );
}

/// A `#xD` that reached the tree through a character reference must come back
/// out as `&#xD;` in text and as `&#xD;` in an attribute value, while `#x9` and
/// `#xA` are escaped in attribute values only.
///
/// Line feed normalization is *not* done here: Canonical XML §2.1 places it
/// "on input, before parsing", because once a `#xD` is in the tree there is no
/// way to tell a literal one from `&#xD;`, and the two must canonicalize
/// differently. See the crate documentation.
#[test]
fn carriage_returns_survive_as_character_references() {
    let document = parse("<a t=\"x&#xD;&#xA;&#x9;y\">one&#xD;two</a>");
    assert_eq!(
        text(&c14n_1_0_document(&document, false).expect("canonicalization")),
        "<a t=\"x&#xD;&#xA;&#x9;y\">one&#xD;two</a>"
    );
}

/// The data model of Canonical XML has no entity reference nodes, so one that
/// could not be expanded is an error rather than silently dropped output.
#[test]
fn unexpanded_entity_references_are_an_error() {
    let document = parse("<!DOCTYPE a [<!ENTITY ext SYSTEM \"x.txt\">]><a>&ext;</a>");
    let error = c14n_1_0_document(&document, false)
        .expect_err("an unexpanded entity reference cannot be canonicalized");
    assert_eq!(
        error,
        C14nError::UnexpandedEntityReference {
            name: "ext".to_owned()
        }
    );
    assert!(error.to_string().contains("&ext;"));
}

/// Every error variant renders a message that names what went wrong.
#[test]
fn errors_display_usefully() {
    let cases = [
        C14nError::UnknownNode {
            node: oxixml_dom::NodeId::from_index(99),
        },
        C14nError::NotAnElement {
            node: oxixml_dom::NodeId::from_index(1),
            kind: NodeKind::Text,
        },
        C14nError::RelativeNamespaceUri {
            prefix: String::new(),
            namespace: "rel".to_owned(),
        },
        C14nError::RelativeNamespaceUri {
            prefix: "p".to_owned(),
            namespace: "rel".to_owned(),
        },
        C14nError::UnboundPrefix {
            prefix: "p".to_owned(),
        },
        C14nError::UnknownAlgorithm {
            uri: "urn:x".to_owned(),
        },
        C14nError::InvalidParameter {
            parameter: "PrefixRewrite",
            value: "x".to_owned(),
        },
        C14nError::InvalidSubset {
            detail: "no apex node",
        },
    ];
    for error in cases {
        assert!(!error.to_string().is_empty(), "{error:?}");
    }
}

/// `join_uri_references` is exported because `oxixml-dsig` needs the same
/// `xml:base` folding when it re-assembles a reference.
#[test]
fn xml_base_folding_is_public() {
    assert_eq!(join_uri_references("a/b/", "../c"), "a/c");
}

/// An empty node-set produces empty output rather than an error.
#[test]
fn an_empty_subset_produces_nothing() {
    let document = parse(SAMPLE);
    let subset = NodeSet::empty(&document);
    assert!(subset.is_empty());
    assert_eq!(subset.len(), 0);
    assert_eq!(
        c14n_1_0(&document, &subset, true).expect("canonicalization"),
        Vec::<u8>::new()
    );
}

/// A deeply nested document must canonicalize, not overflow the stack.
///
/// Every XML Signature and XML Encryption path funnels attacker-supplied
/// documents through canonicalization, and it happens *before* any signature is
/// checked. A recursive traversal turns a ~30 kB document into a stack
/// overflow, which aborts the process rather than returning a [`C14nError`], so
/// the walk keeps its pending work on the heap instead.
#[test]
fn a_deeply_nested_document_does_not_overflow_the_stack() {
    // 10 000 is as deep as the parser that feeds this crate is known to go; the
    // canonicalizer itself is now limited only by memory.
    for depth in [4_000_usize, 10_000] {
        let xml = format!("{}{}", "<a>".repeat(depth), "</a>".repeat(depth));
        let document = parse_nested(&xml, depth);
        let subset = NodeSet::document(&document);
        for algorithm in CanonicalizationAlgorithm::ALL {
            let canonical = algorithm
                .apply(&document, &subset)
                .expect("canonicalization succeeds at any depth");
            assert_eq!(canonical.len(), depth * 7, "{algorithm} at depth {depth}");
        }

        let canonical = c14n_2_0(&document, &C14n2Subset::document(), &C14n2Parameters::new())
            .expect("Canonical XML 2.0 succeeds at any depth");
        assert_eq!(canonical.len(), depth * 7);
    }
}
