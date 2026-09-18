//! The worked examples of the Canonical XML 1.0 and 1.1 Recommendations.
//!
//! [Canonical XML 1.0 §3](https://www.w3.org/TR/xml-c14n#Examples) and
//! [Canonical XML 1.1 §3](https://www.w3.org/TR/xml-c14n11/#Examples) print an
//! input document and the exact octets its canonical form must have. Those are
//! the closest thing the Recommendations have to a conformance suite, so every
//! one of them is ported here as a byte-exact golden test.
//!
//! The inputs use a document type declaration to declare default attributes,
//! attribute types and entities; see [`common`] for how the validating-processor
//! behaviour they assume is supplied.

mod common;

use common::{
    apply_attribute_default, apply_tokenized_type, element_named, expand_entity_reference, parse,
    text,
};
use oxixml_c14n::{NodeSet, c14n_1_0, c14n_1_1};
use oxixml_dom::{Document, NodeId};

/// §3.1 — PIs, comments, and outside of the document element.
const EXAMPLE_3_1: &str = concat!(
    "<?xml version=\"1.0\"?>\n",
    "\n",
    "<?xml-stylesheet   href=\"doc.xsl\"\n",
    "   type=\"text/xsl\"   ?>\n",
    "\n",
    "<!DOCTYPE doc SYSTEM \"doc.dtd\">\n",
    "\n",
    "<doc>Hello, world!<!-- Comment 1 --></doc>\n",
    "\n",
    "<?pi-without-data     ?>\n",
    "\n",
    "<!-- Comment 2 -->\n",
    "\n",
    "<!-- Comment 3 -->\n",
);

#[test]
fn example_3_1_without_comments() {
    let doc = parse(EXAMPLE_3_1);
    let canonical = c14n_1_0(&doc, &NodeSet::document(&doc), false).expect("canonicalization");
    assert_eq!(
        text(&canonical),
        concat!(
            "<?xml-stylesheet href=\"doc.xsl\"\n",
            "   type=\"text/xsl\"   ?>\n",
            "<doc>Hello, world!</doc>\n",
            "<?pi-without-data?>",
        )
    );
}

#[test]
fn example_3_1_with_comments() {
    let doc = parse(EXAMPLE_3_1);
    let canonical = c14n_1_0(&doc, &NodeSet::document(&doc), true).expect("canonicalization");
    assert_eq!(
        text(&canonical),
        concat!(
            "<?xml-stylesheet href=\"doc.xsl\"\n",
            "   type=\"text/xsl\"   ?>\n",
            "<doc>Hello, world!<!-- Comment 1 --></doc>\n",
            "<?pi-without-data?>\n",
            "<!-- Comment 2 -->\n",
            "<!-- Comment 3 -->",
        )
    );
}

/// §3.2 — white space in document content is preserved verbatim.
#[test]
fn example_3_2_whitespace_in_document_content() {
    let input = concat!(
        "<doc>\n",
        "   <clean>   </clean>\n",
        "   <dirty>   A   B   </dirty>\n",
        "   <mixed>\n",
        "      A\n",
        "      <clean>   </clean>\n",
        "      B\n",
        "      <dirty>   A   B   </dirty>\n",
        "      C\n",
        "   </mixed>\n",
        "</doc>\n",
    );
    let doc = parse(input);
    let canonical = c14n_1_0(&doc, &NodeSet::document(&doc), false).expect("canonicalization");
    // The canonical form is the input document element, byte for byte.
    assert_eq!(text(&canonical), input.trim_end_matches('\n'));
}

/// §3.3 — start and end tags: empty elements expand, white space inside tags is
/// normalized away, superfluous namespace declarations disappear, default
/// attributes appear, and everything sorts.
#[test]
fn example_3_3_start_and_end_tags() {
    let input = concat!(
        "<!DOCTYPE doc [<!ATTLIST e9 attr CDATA \"default\">]>\n",
        "<doc>\n",
        "   <e1   />\n",
        "   <e2   ></e2>\n",
        "   <e3   name = \"elem3\"   id=\"elem3\"   />\n",
        "   <e4   name=\"elem4\"   id=\"elem4\"   ></e4>\n",
        "   <e5 a:attr=\"out\" b:attr=\"sorted\" attr2=\"all\" attr=\"I'm\"\n",
        "      xmlns:b=\"http://www.ietf.org\"\n",
        "      xmlns:a=\"http://www.w3.org\"\n",
        "      xmlns=\"http://example.org\"/>\n",
        "   <e6 xmlns=\"\" xmlns:a=\"http://www.w3.org\">\n",
        "      <e7 xmlns=\"http://www.ietf.org\">\n",
        "         <e8 xmlns=\"\" xmlns:a=\"http://www.w3.org\">\n",
        "            <e9 xmlns=\"\" xmlns:a=\"http://www.ietf.org\"/>\n",
        "         </e8>\n",
        "      </e7>\n",
        "   </e6>\n",
        "</doc> \n",
    );
    let mut doc = parse(input);
    apply_attribute_default(&mut doc, "e9", "attr", "default");

    let canonical = c14n_1_0(&doc, &NodeSet::document(&doc), false).expect("canonicalization");
    assert_eq!(
        text(&canonical),
        concat!(
            "<doc>\n",
            "   <e1></e1>\n",
            "   <e2></e2>\n",
            "   <e3 id=\"elem3\" name=\"elem3\"></e3>\n",
            "   <e4 id=\"elem4\" name=\"elem4\"></e4>\n",
            "   <e5 xmlns=\"http://example.org\" xmlns:a=\"http://www.w3.org\" ",
            "xmlns:b=\"http://www.ietf.org\" attr=\"I'm\" attr2=\"all\" ",
            "b:attr=\"sorted\" a:attr=\"out\"></e5>\n",
            "   <e6 xmlns:a=\"http://www.w3.org\">\n",
            "      <e7 xmlns=\"http://www.ietf.org\">\n",
            "         <e8 xmlns=\"\">\n",
            "            <e9 xmlns:a=\"http://www.ietf.org\" attr=\"default\"></e9>\n",
            "         </e8>\n",
            "      </e7>\n",
            "   </e6>\n",
            "</doc>",
        )
    );
}

/// §3.4 — character modifications and character references.
#[test]
fn example_3_4_character_modifications() {
    let input = concat!(
        "<!DOCTYPE doc [\n",
        "<!ATTLIST normId id ID #IMPLIED>\n",
        "<!ATTLIST normNames attr NMTOKENS #IMPLIED>\n",
        "]>\n",
        "<doc>\n",
        "   <text>First line&#x0d;&#10;Second line</text>\n",
        "   <value>&#x32;</value>\n",
        "   <compute><![CDATA[value>\"0\" && value<\"10\" ?\"valid\":\"error\"]]></compute>\n",
        "   <compute expr='value>\"0\" &amp;&amp; value&lt;\"10\" ?\"valid\":\"error\"'>valid</compute>\n",
        "   <norm attr=' &apos;   &#x20;&#13;&#xa;&#9;   &apos; '/>\n",
        "   <normNames attr='   A   &#x20;&#13;&#xa;&#9;   B   '/>\n",
        "   <normId id=' &apos;   &#x20;&#13;&#xa;&#9;   &apos; '/>\n",
        "</doc>\n",
    );
    let mut doc = parse(input);
    // `attr` on <normNames> is NMTOKENS and `id` on <normId> is ID; both are
    // tokenized types, so their values get the extra normalization. `attr` on
    // <norm> is undeclared and therefore CDATA, and keeps its spaces.
    apply_tokenized_type(&mut doc, "normNames", "attr");
    apply_tokenized_type(&mut doc, "normId", "id");

    let canonical = c14n_1_0(&doc, &NodeSet::document(&doc), false).expect("canonicalization");
    assert_eq!(
        text(&canonical),
        concat!(
            "<doc>\n",
            "   <text>First line&#xD;\nSecond line</text>\n",
            "   <value>2</value>\n",
            "   <compute>value&gt;\"0\" &amp;&amp; value&lt;\"10\" ?\"valid\":\"error\"</compute>\n",
            "   <compute expr=\"value>&quot;0&quot; &amp;&amp; value&lt;&quot;10&quot; ",
            "?&quot;valid&quot;:&quot;error&quot;\">valid</compute>\n",
            "   <norm attr=\" '    &#xD;&#xA;&#x9;   ' \"></norm>\n",
            "   <normNames attr=\"A &#xD;&#xA;&#x9; B\"></normNames>\n",
            "   <normId id=\"' &#xD;&#xA;&#x9; '\"></normId>\n",
            "</doc>",
        )
    );
}

/// §3.5 — entity references, including an external one.
#[test]
fn example_3_5_entity_references() {
    let input = concat!(
        "<!DOCTYPE doc [\n",
        "<!ATTLIST doc attrExtEnt ENTITY #IMPLIED>\n",
        "<!ENTITY ent1 \"Hello\">\n",
        "<!ENTITY ent2 SYSTEM \"world.txt\">\n",
        "<!ENTITY entExt SYSTEM \"earth.gif\" NDATA gif>\n",
        "<!NOTATION gif SYSTEM \"viewgif.exe\">\n",
        "]>\n",
        "<doc attrExtEnt=\"entExt\">\n",
        "   &ent1;, &ent2;!\n",
        "</doc>\n",
        "\n",
        "<!-- Let world.txt contain \"world\" (excluding the quotes) -->\n",
    );
    let mut doc = parse(input);
    // "Let world.txt contain \"world\"".
    expand_entity_reference(&mut doc, "ent2", "world");

    let canonical = c14n_1_0(&doc, &NodeSet::document(&doc), false).expect("canonicalization");
    assert_eq!(
        text(&canonical),
        "<doc attrExtEnt=\"entExt\">\n   Hello, world!\n</doc>"
    );
}

/// §3.6 — UTF-8 encoding: the canonical form is always UTF-8, whatever the
/// document declared.
#[test]
fn example_3_6_utf8_encoding() {
    let doc = parse("<?xml version=\"1.0\" encoding=\"ISO-8859-1\"?>\n<doc>&#169;</doc>\n");
    let canonical = c14n_1_0(&doc, &NodeSet::document(&doc), false).expect("canonicalization");
    assert_eq!(canonical, b"<doc>\xc2\xa9</doc>");
}

/// The input shared by §3.7 of Canonical XML 1.0.
const EXAMPLE_3_7: &str = concat!(
    "<!DOCTYPE doc [\n",
    "<!ATTLIST e2 xml:space (default|preserve) 'preserve'>\n",
    "<!ATTLIST e3 id ID #IMPLIED>\n",
    "]>\n",
    "<doc xmlns=\"http://www.ietf.org\" xmlns:w3c=\"http://www.w3.org\">\n",
    "   <e1>\n",
    "      <e2 xmlns=\"\">\n",
    "         <e3 id=\"E3\"/>\n",
    "      </e2>\n",
    "   </e1>\n",
    "</doc>\n",
);

/// The input of §3.8 of Canonical XML 1.1: §3.7 plus `xml:base` and `xml:id`.
const EXAMPLE_3_8: &str = concat!(
    "<!DOCTYPE doc [\n",
    "<!ATTLIST e2 xml:space (default|preserve) 'preserve'>\n",
    "<!ATTLIST e3 id ID #IMPLIED>\n",
    "]>\n",
    "<doc xmlns=\"http://www.ietf.org\" xmlns:w3c=\"http://www.w3.org\" ",
    "xml:base=\"something/else\">\n",
    "   <e1>\n",
    "      <e2 xmlns=\"\" xml:id=\"abc\" xml:base=\"bar/\">\n",
    "         <e3 id=\"E3\" xml:base=\"foo\"/>\n",
    "      </e2>\n",
    "   </e1>\n",
    "</doc>\n",
);

/// Builds the node-set both §3.7 and §3.8 select with
/// ```text
/// (//. | //@* | //namespace::*)
/// [
///    self::ietf:e1 or (parent::ietf:e1 and not(self::text() or self::e2))
///    or
///    count(id("E3")|ancestor-or-self::node()) = count(ancestor-or-self::node())
/// ]
/// ```
/// which keeps `e1` with its namespace nodes, and `e3` with its attribute and
/// namespace nodes — but neither `e2`, nor `doc`, nor any text node.
fn document_subset(document: &Document) -> NodeSet {
    let e1: NodeId = element_named(document, "e1");
    let e3: NodeId = element_named(document, "e3");
    let mut subset = NodeSet::empty(document);
    subset.insert(e1);
    subset.insert(e3);
    for attribute in document.attributes(e3) {
        subset.insert(attribute);
    }
    subset
}

/// Canonical XML 1.0 §3.7 — document subsets: an empty default namespace
/// propagates out of the omitted parent, `xml:space` is inherited into the
/// orphan, and the omitted `xmlns:w3c` survives on the apex.
#[test]
fn example_3_7_document_subsets() {
    let mut doc = parse(EXAMPLE_3_7);
    apply_attribute_default(&mut doc, "e2", "xml:space", "preserve");
    let subset = document_subset(&doc);

    let canonical = c14n_1_0(&doc, &subset, false).expect("canonicalization");
    assert_eq!(
        text(&canonical),
        concat!(
            "<e1 xmlns=\"http://www.ietf.org\" xmlns:w3c=\"http://www.w3.org\">",
            "<e3 xmlns=\"\" id=\"E3\" xml:space=\"preserve\"></e3>",
            "</e1>",
        )
    );
}

/// Canonical XML 1.1 §3.8 — document subsets and XML attributes: `xml:id` is
/// *not* inherited, `xml:space` is, and `xml:base` is fixed up.
///
/// The expected value of `e3`'s `xml:base` is the one from
/// [erratum E11-01](https://www.w3.org/2008/05/xml-c14n11-errata), not the one
/// printed in the Recommendation. Only the *contiguously* omitted ancestors
/// fold in — [§2.4](https://www.w3.org/TR/xml-c14n11/#DocSubsets) spells that
/// out, "It is important to note that En … E1 is for contiguously omitted
/// elements, for example only e2 in the example in Section 3.8" — so `e3` gets
/// `bar/` joined with `foo` and nothing else. `doc`'s `something/else` is
/// already carried by the fixed-up `e1` that `e3` is rendered inside; folding
/// it in a second time would make `e3`'s effective base
/// `something/something/bar/foo`.
#[test]
fn example_3_8_document_subsets_and_xml_attributes() {
    let mut doc = parse(EXAMPLE_3_8);
    apply_attribute_default(&mut doc, "e2", "xml:space", "preserve");
    let subset = document_subset(&doc);

    let canonical = c14n_1_1(&doc, &subset, false).expect("canonicalization");
    assert_eq!(
        text(&canonical),
        concat!(
            "<e1 xmlns=\"http://www.ietf.org\" xmlns:w3c=\"http://www.w3.org\" ",
            "xml:base=\"something/else\">",
            "<e3 xmlns=\"\" id=\"E3\" xml:base=\"bar/foo\" xml:space=\"preserve\">",
            "</e3></e1>",
        )
    );
}

/// An included ancestor *between* two omitted ones ends the run.
///
/// With `r` and `b` omitted from `{a, e}`, only `b` is contiguous with `e`;
/// `r`'s base already reached the output through `a`. Folding it again would
/// emit `p/s/t/` on an element whose parent already renders `p/q/`, making the
/// effective base `p/q/p/s/t/` instead of `p/q/s/t/`.
#[test]
fn only_the_contiguous_run_of_omitted_ancestors_folds_in() {
    let doc = parse(concat!(
        "<r xml:base=\"p/\"><a xml:base=\"q/\"><b xml:base=\"s/\">",
        "<e xml:base=\"t/\"/></b></a></r>",
    ));
    let a = element_named(&doc, "a");
    let e = element_named(&doc, "e");
    let mut subset = NodeSet::empty(&doc);
    subset.insert(a);
    subset.insert(e);
    for element in [a, e] {
        for attribute in doc.attributes(element) {
            subset.insert(attribute);
        }
    }

    let canonical = c14n_1_1(&doc, &subset, false).expect("canonicalization");
    assert_eq!(
        text(&canonical),
        "<a xml:base=\"p/q/\"><e xml:base=\"s/t/\"></e></a>"
    );
}

/// The same subset under Canonical XML 1.0, which inherits every `xml:*`
/// attribute including `xml:id` and does no `xml:base` fixup — the difference
/// the 1.1 revision exists to fix.
#[test]
fn version_1_0_inherits_xml_id_and_does_not_fix_up_xml_base() {
    let mut doc = parse(EXAMPLE_3_8);
    apply_attribute_default(&mut doc, "e2", "xml:space", "preserve");
    let subset = document_subset(&doc);

    let canonical = c14n_1_0(&doc, &subset, false).expect("canonicalization");
    assert_eq!(
        text(&canonical),
        concat!(
            "<e1 xmlns=\"http://www.ietf.org\" xmlns:w3c=\"http://www.w3.org\" ",
            "xml:base=\"something/else\">",
            "<e3 xmlns=\"\" id=\"E3\" xml:base=\"foo\" xml:id=\"abc\" xml:space=\"preserve\">",
            "</e3></e1>",
        )
    );
}

/// Canonical XML 1.0 §4.7 — the XPath data model cannot tell an element that
/// undeclares the default namespace from one whose parent did, so both spellings
/// must canonicalize identically once the middle element is dropped.
#[test]
fn section_4_7_default_namespace_propagation() {
    for input in [
        "<e1 xmlns=\"a:b\"><e2 xmlns=\"\"><e3/></e2></e1>",
        "<e1 xmlns=\"a:b\"><e2><e3 xmlns=\"\"/></e2></e1>",
    ] {
        let doc = parse(input);
        let subset = {
            let mut subset = NodeSet::empty(&doc);
            subset.insert(element_named(&doc, "e1"));
            subset.insert(element_named(&doc, "e3"));
            subset
        };
        let canonical = c14n_1_0(&doc, &subset, false).expect("canonicalization");
        assert_eq!(
            text(&canonical),
            "<e1 xmlns=\"a:b\"><e3 xmlns=\"\"></e3></e1>",
            "input: {input}"
        );
    }
}

/// A relative namespace URI is an operation failure, as §2.1 requires.
#[test]
fn relative_namespace_uris_are_rejected() {
    let doc = parse("<a xmlns:p=\"relative/uri\"><p:b/></a>");
    let error = c14n_1_0(&doc, &NodeSet::document(&doc), false)
        .expect_err("a relative namespace URI must fail");
    assert_eq!(
        error,
        oxixml_c14n::C14nError::RelativeNamespaceUri {
            prefix: "p".to_owned(),
            namespace: "relative/uri".to_owned(),
        }
    );
}

/// Canonicalizing a canonical form must be a no-op — the property every
/// signature relies on.
#[test]
fn canonicalization_is_idempotent() {
    let doc = parse(EXAMPLE_3_7);
    let once = c14n_1_0(&doc, &NodeSet::document(&doc), true).expect("canonicalization");
    let reparsed = parse(&text(&once));
    let twice = c14n_1_0(&reparsed, &NodeSet::document(&reparsed), true).expect("canonicalization");
    assert_eq!(text(&once), text(&twice));
}
