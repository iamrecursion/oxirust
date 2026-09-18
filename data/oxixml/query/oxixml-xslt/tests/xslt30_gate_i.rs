//! Regression tests for result-tree namespace fidelity, simple-content separators, stylesheet
//! whitespace stripping, serialization defaults and the static checks around them.
//!
//! Every test reproduces one W3C `xslt30-test` case with a minimal inline stylesheet, so each
//! fix is provable red-then-green without the separate `oxixml-testsuite` harness. The case
//! identifiers in the comments are the ones that harness reports.

use oxixml_infoset::Tree;
use oxixml_xslt::{Stylesheet, XsltError};

fn stylesheet(body: &str) -> String {
    format!(
        r#"<xsl:stylesheet version="3.0" xmlns:xsl="http://www.w3.org/1999/XSL/Transform"
             xmlns:xs="http://www.w3.org/2001/XMLSchema" exclude-result-prefixes="xs">
             <xsl:output omit-xml-declaration="yes"/>
             {body}
           </xsl:stylesheet>"#
    )
}

fn run_source(xsl: &str, source: &str) -> Result<String, XsltError> {
    let compiled = Stylesheet::compile_str(xsl)?;
    let tree = Tree::parse_str(source)
        .map_err(|error| XsltError::dynamic("XTDE0000", error.message().to_owned()))?;
    compiled.transform_to_string(&tree.root())
}

/// The static error code a stylesheet is rejected with, or `None` when it compiles.
fn compile_error(xsl: &str) -> Option<String> {
    Stylesheet::compile_str(xsl)
        .err()
        .and_then(|error| error.code().map(str::to_owned))
}

// === Namespace fixup rewrites the name (XSLT 3.0 §5.7.3) ======================================

/// bug-1501: two attributes copied from two namespaces onto one element were both written
/// `x:aaa` in the source; fixup has to give one of them another prefix *and report it*, or
/// `distinct-values(…/name())` counts one name where the suite requires two.
#[test]
fn fixup_reports_the_prefix_it_settled_on() {
    let xsl = stylesheet(
        r#"<xsl:template match="in"><out><xsl:variable name="atts" as="element()">
             <e><xsl:copy-of select="foo/@* | bar/@*"/></e>
           </xsl:variable>
           <ok><xsl:value-of select="count(distinct-values($atts//@*/name())) = 2"/></ok>
           </out></xsl:template>"#,
    );
    let source = r#"<in><foo x:aaa="apple" xmlns:x="http://test1.uri/"/>
                        <bar x:aaa="orange" xmlns:x="http://test2.uri/"/></in>"#;
    assert_eq!(
        run_source(&xsl, source).expect("the transformation runs"),
        "<out><ok>true</ok></out>"
    );
}

/// bug-5101: a copied attribute keeps the prefix its name was written with instead of taking a
/// generated one, so long as that prefix is free in the constructed element's scope.
#[test]
fn a_copied_attribute_keeps_the_prefix_it_was_written_with() {
    let xsl = stylesheet(
        r#"<xsl:template match="doc"><out><xsl:copy-of select="e/@*"/></out></xsl:template>"#,
    );
    let source = r#"<doc><e zz:a="1" xmlns:zz="urn:z"/></doc>"#;
    assert_eq!(
        run_source(&xsl, source).expect("the transformation runs"),
        r#"<out xmlns:zz="urn:z" zz:a="1"/>"#
    );
}

// === The default namespace is undeclared when it goes out of scope ============================

/// bug-1802: an element in no namespace written inside one that declared a default namespace
/// needs `xmlns=""`, or the start tag reads back as being in the outer namespace.
#[test]
fn a_child_in_no_namespace_undeclares_the_default() {
    let xsl = stylesheet(
        r#"<xsl:template match="pre:foo" xmlns:pre="http://pre.uri/">
             <xsl:copy><xsl:element name="bar"/></xsl:copy>
           </xsl:template>"#,
    );
    assert_eq!(
        run_source(&xsl, r#"<foo xmlns="http://pre.uri/"/>"#).expect("the transformation runs"),
        r#"<foo xmlns="http://pre.uri/"><bar xmlns=""/></foo>"#
    );
}

/// The negative control: a child that keeps the default namespace declares nothing extra.
#[test]
fn a_child_in_the_default_namespace_undeclares_nothing() {
    let xsl = stylesheet(
        r#"<xsl:template match="pre:foo" xmlns:pre="http://pre.uri/">
             <xsl:copy><xsl:element name="bar" namespace="http://pre.uri/"/></xsl:copy>
           </xsl:template>"#,
    );
    assert_eq!(
        run_source(&xsl, r#"<foo xmlns="http://pre.uri/"/>"#).expect("the transformation runs"),
        r#"<foo xmlns="http://pre.uri/"><bar/></foo>"#
    );
}

// === Simple content: a zero-length text node parts adjacent atomics (§5.7.1) ==================

/// seqtor-023: the two `-` values are parted by the zero-length text nodes between them, so the
/// single-space separator of §5.7.1 never applies and the result is `--`.
#[test]
fn a_zero_length_text_node_parts_adjacent_atomic_values() {
    let xsl = stylesheet(
        r#"<xsl:output method="text"/>
           <xsl:template match="/"><xsl:sequence select="'-'"/>
             <xsl:text/><xsl:sequence select="''"/><xsl:text/>
             <xsl:sequence select="'-'"/></xsl:template>"#,
    );
    assert_eq!(
        run_source(&xsl, "<doc/>").expect("the transformation runs"),
        "--"
    );
}

/// The negative control: with nothing between them the two values *are* adjacent and take the
/// separator.
#[test]
fn adjacent_atomic_values_still_take_the_separator() {
    let xsl = stylesheet(
        r#"<xsl:output method="text"/>
           <xsl:template match="/"><xsl:sequence select="'-'"/>
             <xsl:sequence select="'-'"/></xsl:template>"#,
    );
    assert_eq!(
        run_source(&xsl, "<doc/>").expect("the transformation runs"),
        "- -"
    );
}

/// seqtor-041: three zero-length `xsl:text` results are three items of an `as="text()*"`
/// variable, whatever produced the empty string.
#[test]
fn three_zero_length_text_results_are_three_items() {
    let xsl = stylesheet(
        r#"<xsl:variable name="foo" as="text()*">
             <xsl:text expand-text="yes">{ 'empty'[false()]}</xsl:text>
             <xsl:text/>
             <xsl:text expand-text="yes">{ 'empty'[false()]}</xsl:text>
           </xsl:variable>
           <xsl:template match="/"><result><count><xsl:value-of select="count($foo)"/></count></result></xsl:template>"#,
    );
    assert_eq!(
        run_source(&xsl, "<doc/>").expect("the transformation runs"),
        "<result><count>3</count></result>"
    );
}

// === Stylesheet whitespace stripping (XSLT 3.0 §4.2, §4.3) ====================================

/// whitespace-012: a comment is removed from the stylesheet *before* whitespace is stripped, so
/// the two text nodes it separated are one not-whitespace-only text node by then.
#[test]
fn a_comment_does_not_split_a_text_node_for_stripping() {
    let xsl =
        stylesheet(r#"<xsl:template match="/"><out><e>   h<!--c-->   </e></out></xsl:template>"#);
    assert_eq!(
        run_source(&xsl, "<doc/>").expect("the transformation runs"),
        "<out><e>   h   </e></out>"
    );
}

/// whitespace-013: a processing instruction behaves the same way.
#[test]
fn a_processing_instruction_does_not_split_a_text_node_for_stripping() {
    let xsl = stylesheet(r#"<xsl:template match="/"><out><e>   <?pi?>h</e></out></xsl:template>"#);
    assert_eq!(
        run_source(&xsl, "<doc/>").expect("the transformation runs"),
        "<out><e>   h</e></out>"
    );
}

/// The negative control: an element child really does separate the two runs, so the trailing
/// whitespace-only text node is stripped as §4.3 requires.
#[test]
fn an_element_child_still_splits_a_text_node_for_stripping() {
    let xsl = stylesheet(r#"<xsl:template match="/"><out><e>   h<b/>   </e></out></xsl:template>"#);
    assert_eq!(
        run_source(&xsl, "<doc/>").expect("the transformation runs"),
        "<out><e>   h<b/></e></out>"
    );
}

/// whitespace-015: `xml:space="preserve"` keeps the whitespace of the sequence constructor, but
/// the layout around the leading `xsl:sort` declarations is not part of it.
#[test]
fn whitespace_around_a_leading_declaration_is_not_content() {
    let xsl = stylesheet(
        r#"<xsl:template match="/"><out><xsl:for-each select="doc" xml:space="preserve">
   <xsl:sort/>
   <xsl:sort/>ab</xsl:for-each></out></xsl:template>"#,
    );
    assert_eq!(
        run_source(&xsl, "<doc/>").expect("the transformation runs"),
        "<out>ab</out>"
    );
}

// === Serialization =============================================================================

/// output-0130: with no `xsl:output` at all, a result whose document element is `html` in the
/// XHTML namespace serializes under the `xhtml` method, which injects a `meta` naming
/// `text/html` — not the `application/xml` the `xml` method would default to.
#[test]
fn an_xhtml_document_element_selects_the_xhtml_method() {
    let xsl = r#"<xsl:transform version="3.0" xmlns:xsl="http://www.w3.org/1999/XSL/Transform"
                   xmlns="http://www.w3.org/1999/xhtml">
                   <xsl:template match="/"><html><head><title>t</title></head></html></xsl:template>
                 </xsl:transform>"#;
    let out = run_source(xsl, "<doc/>").expect("the transformation runs");
    assert!(
        out.contains(r#"content="text/html; charset=UTF-8""#),
        "{out}"
    );
    assert!(
        out.starts_with(r#"<?xml version="1.0" encoding="UTF-8"?>"#),
        "{out}"
    );
}

/// output-0195a: the HTML output method has no literal spelling for the C1 controls, so `#x9F`
/// is written as a character reference.
#[test]
fn the_html_method_writes_c1_controls_as_references() {
    let xsl = stylesheet(
        r#"<xsl:output method="html"/>
           <xsl:template match="/"><doc>&#x9f;</doc></xsl:template>"#,
    );
    assert_eq!(
        run_source(&xsl, "<doc/>").expect("the transformation runs"),
        "<doc>&#159;</doc>"
    );
}

/// output-0312: erratum E31 makes a zero-length `doctype-system` an override that removes the
/// document type declaration rather than a request for an empty system identifier.
#[test]
fn an_empty_doctype_system_writes_no_declaration() {
    let xsl = stylesheet(
        r#"<xsl:output method="xml" doctype-system="" doctype-public=""/>
           <xsl:template match="/"><a><b/></a></xsl:template>"#,
    );
    assert_eq!(
        run_source(&xsl, "<doc/>").expect("the transformation runs"),
        "<a><b/></a>"
    );
}

/// output-0232: `suppress-indentation` covers the whole subtree below the element it names.
#[test]
fn suppress_indentation_reaches_descendants() {
    let xsl = stylesheet(
        r#"<xsl:output method="xml" indent="yes" suppress-indentation="p"/>
           <xsl:template match="/"><book><p>a<i>b</i>c</p></book></xsl:template>"#,
    );
    let out = run_source(&xsl, "<doc/>").expect("the transformation runs");
    assert!(out.contains("<p>a<i>b</i>c</p>"), "{out}");
}

/// character-map-005: two `xsl:output` declarations naming two character maps contribute to one
/// output definition; the later declaration wins where they disagree.
#[test]
fn two_output_declarations_union_their_character_maps() {
    let xsl = stylesheet(
        r#"<xsl:output method="xml" use-character-maps="m1"/>
           <xsl:output method="xml" use-character-maps="m2"/>
           <xsl:character-map name="m1">
             <xsl:output-character character="a" string="[A]"/>
             <xsl:output-character character="c" string="[C]"/>
           </xsl:character-map>
           <xsl:character-map name="m2">
             <xsl:output-character character="c" string="{C}"/>
           </xsl:character-map>
           <xsl:template match="/"><out>abc</out></xsl:template>"#,
    );
    assert_eq!(
        run_source(&xsl, "<doc/>").expect("the transformation runs"),
        "<out>[A]b{C}</out>"
    );
}

/// character-map-009: a character map is not applied where URI escaping has been.
#[test]
fn a_character_map_leaves_a_uri_attribute_alone() {
    let xsl = stylesheet(
        r#"<xsl:output method="html" indent="no" use-character-maps="m"/>
           <xsl:character-map name="m">
             <xsl:output-character character="z" string="[Z]"/>
           </xsl:character-map>
           <xsl:template match="/"><a href="z.html">z</a></xsl:template>"#,
    );
    let out = run_source(&xsl, "<doc/>").expect("the transformation runs");
    assert!(out.contains(r#"href="z.html""#), "{out}");
    assert!(out.contains(">[Z]<"), "{out}");
}

// === Static errors =============================================================================

/// error-0010n: `xsl:choose` has the content model `(xsl:when+, xsl:otherwise?)`, so an
/// `xsl:otherwise` on its own is as much an error as an empty `xsl:choose`.
#[test]
fn an_xsl_choose_without_a_when_is_rejected() {
    let xsl = stylesheet(
        r#"<xsl:template match="/"><out><xsl:choose><xsl:otherwise/></xsl:choose></out></xsl:template>"#,
    );
    assert_eq!(compile_error(&xsl).as_deref(), Some("XTSE0010"));
}

/// error-0110a: the version attribute holds an `xs:decimal`, which has no exponent.
#[test]
fn a_version_with_an_exponent_is_rejected() {
    let xsl = r#"<xsl:stylesheet version="2.0e3" xmlns:xsl="http://www.w3.org/1999/XSL/Transform">
                   <xsl:template name="main"><out/></xsl:template>
                 </xsl:stylesheet>"#;
    assert_eq!(compile_error(xsl).as_deref(), Some("XTSE0110"));
}

/// The negative control: the versions the suite actually writes stay acceptable.
#[test]
fn ordinary_versions_are_accepted() {
    for version in ["1.0", "2.0", "3.0", "4.0"] {
        let xsl = format!(
            r#"<xsl:stylesheet version="{version}" xmlns:xsl="http://www.w3.org/1999/XSL/Transform">
                 <xsl:template name="main"><out/></xsl:template>
               </xsl:stylesheet>"#
        );
        assert_eq!(compile_error(&xsl), None, "version {version}");
    }
}

/// error-0260d: an XSLT element required to be empty stays empty even under
/// `xml:space="preserve"`.
#[test]
fn a_strip_space_with_preserved_whitespace_is_rejected() {
    let xsl = stylesheet(
        "<xsl:strip-space elements=\"*\" xml:space=\"preserve\">\n  \n</xsl:strip-space>\
         <xsl:template name=\"main\"><out/></xsl:template>",
    );
    assert_eq!(compile_error(&xsl).as_deref(), Some("XTSE0260"));
}

/// error-0270a, strip-space-028: one name test may not appear in both `xsl:strip-space` and
/// `xsl:preserve-space` at the same import precedence, however it is spelled.
#[test]
fn a_name_test_claimed_by_both_space_controls_is_rejected() {
    let xsl = stylesheet(
        r#"<xsl:strip-space elements="a b Q{}c"/><xsl:preserve-space elements="Q{}b"/>
           <xsl:template name="main"><out/></xsl:template>"#,
    );
    assert_eq!(compile_error(&xsl).as_deref(), Some("XTSE0270"));
}

/// The negative control: the two declarations may name disjoint tests.
#[test]
fn disjoint_space_controls_are_accepted() {
    let xsl = stylesheet(
        r#"<xsl:strip-space elements="a"/><xsl:preserve-space elements="b"/>
           <xsl:template name="main"><out/></xsl:template>"#,
    );
    assert_eq!(compile_error(&xsl), None);
}

/// error-0670b: two sibling `xsl:with-param` elements may not name the same expanded QName,
/// however their prefixes are spelled.
#[test]
fn duplicate_sibling_with_params_are_rejected() {
    let xsl = stylesheet(
        r#"<xsl:template name="x"><xsl:param xmlns:x="http://a.uri/" name="x:y"/></xsl:template>
           <xsl:template match="/"><out><xsl:call-template name="x">
             <xsl:with-param xmlns:a="http://a.uri/" name="a:y" select="3"/>
             <xsl:with-param xmlns:b="http://a.uri/" name="b:y" select="4"/>
           </xsl:call-template></out></xsl:template>"#,
    );
    assert_eq!(compile_error(&xsl).as_deref(), Some("XTSE0670"));
}

/// error-3150a: `xsl:catch` may have a `select` attribute or content, not both.
#[test]
fn an_xsl_catch_with_select_and_content_is_rejected() {
    let xsl = stylesheet(
        r#"<xsl:template name="main"><out><xsl:try select="1 to 10">
             <xsl:catch select="22"><xsl:value-of select="23"/></xsl:catch>
           </xsl:try></out></xsl:template>"#,
    );
    assert_eq!(compile_error(&xsl).as_deref(), Some("XTSE3150"));
}

/// output-0284: `doctype-public` supplies a `PubidLiteral`, whose characters XML 1.0 §2.3 fixes.
#[test]
fn an_invalid_public_identifier_is_rejected() {
    let xsl = stylesheet(
        r#"<xsl:output method="xml" doctype-public="&#xa3;[~~~]"/>
           <xsl:template match="/"><doc>hello</doc></xsl:template>"#,
    );
    assert_eq!(compile_error(&xsl).as_deref(), Some("XTSE0020"));
}

/// The negative control: an ordinary public identifier still compiles.
#[test]
fn an_ordinary_public_identifier_is_accepted() {
    let xsl = stylesheet(
        r#"<xsl:output method="xml" doctype-public="-//W3C//DTD XHTML 1.0 Strict//EN"
             doctype-system="x.dtd"/>
           <xsl:template match="/"><doc>hello</doc></xsl:template>"#,
    );
    assert_eq!(compile_error(&xsl), None);
}
