//! Regression tests for simple-content construction, disabled output escaping, declared modes,
//! namespace aliasing and the result-tree bookkeeping around them.
//!
//! Every test reproduces one W3C `xslt30-test` case with a minimal inline stylesheet, so each
//! fix is provable red-then-green without the separate `oxixml-testsuite` harness. The case
//! identifiers in the comments are the ones that harness reports.

use oxixml_infoset::Tree;
use oxixml_xslt::{Stylesheet, XsltError};

fn stylesheet(body: &str) -> String {
    format!(
        r#"<xsl:stylesheet version="3.0" xmlns:xsl="http://www.w3.org/1999/XSL/Transform"
             xmlns:xs="http://www.w3.org/2001/XMLSchema" xmlns:f="http://f/"
             exclude-result-prefixes="xs f">
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

fn run(xsl: &str) -> Result<String, XsltError> {
    run_source(xsl, "<in/>")
}

/// The dynamic error code a transformation fails with, or `None` when it succeeds.
fn run_error(xsl: &str, source: &str) -> Option<String> {
    run_source(xsl, source)
        .err()
        .and_then(|error| error.code().map(str::to_owned))
}

// === Constructing simple content merges adjacent text nodes (XSLT 3.0 §5.7.2) =================

/// whitespace-025: a function declared `as="text()*"` returns the three text nodes `'`, `test`
/// and `'`. Simple-content construction merges adjacent text nodes *before* the separator is
/// applied, so `xsl:value-of` over them is `'test'` and not `' test '`.
#[test]
fn adjacent_text_nodes_merge_before_the_separator() {
    let xsl = stylesheet(
        r#"<xsl:function name="f:quote" as="text()*">
             <xsl:param name="x" as="xs:string"/>
             <xsl:text>'</xsl:text><xsl:value-of select="$x"/><xsl:text>'</xsl:text>
           </xsl:function>
           <xsl:template match="/">
             <out><xsl:value-of select="f:quote('test')"/></out>
           </xsl:template>"#,
    );
    assert_eq!(
        run(&xsl).expect("the transformation runs"),
        "<out>'test'</out>"
    );
}

/// The negative control: two *atomic* values are still separated, since only text nodes merge.
#[test]
fn adjacent_atomic_values_still_take_the_separator() {
    let xsl = stylesheet(
        r#"<xsl:template match="/"><out><xsl:value-of select="('a','b')"/></out></xsl:template>"#,
    );
    assert_eq!(
        run(&xsl).expect("the transformation runs"),
        "<out>a b</out>"
    );
}

/// avt-1205: an attribute value template constructs simple content too, and a zero-length text
/// node is discarded *before* the merge, so it does not keep the text nodes on either side of it
/// apart — `<xsl:text/>[<xsl:value-of/>]<xsl:text/>` is `[0]`, not `[ 0 ]`.
#[test]
fn a_zero_length_text_node_does_not_part_the_merge() {
    let xsl = stylesheet(
        r#"<xsl:function name="f:t" as="text()*">
             <xsl:param name="p"/>
             <xsl:text/>[<xsl:value-of select="$p"/>]<xsl:text/>
           </xsl:function>
           <xsl:template match="/">
             <out test="{f:t(0)}{f:t(1)}"/>
           </xsl:template>"#,
    );
    assert_eq!(
        run(&xsl).expect("the transformation runs"),
        r#"<out test="[0][1]"/>"#
    );
}

/// seqtor-029: a text value template joins the items of its expression with a single space, so a
/// recursive function that builds nothing but text nodes must come out with no space anywhere.
#[test]
fn a_text_value_template_merges_the_nodes_a_function_returns() {
    let xsl = stylesheet(
        r#"<xsl:function name="f:create" as="item()+">
             <xsl:param name="i" as="xs:integer"/>
             <xsl:sequence expand-text="yes">{if ($i = 0) then 'end' else f:create($i - 1)}</xsl:sequence>
             <xsl:sequence expand-text="yes">{codepoints-to-string($i + 65)}</xsl:sequence>
           </xsl:function>
           <xsl:template match="/">
             <out><xsl:sequence select="'start'"/><xsl:sequence select="f:create(3)"/></out>
           </xsl:template>"#,
    );
    assert_eq!(
        run(&xsl).expect("the transformation runs"),
        "<out>startendABCD</out>"
    );
}

// === disable-output-escaping survives a sequence but not a temporary tree =====================

/// doe-0180: a template declared `as="text()"` hands its text node back as a sequence item, and
/// writing that item into the final result still disables escaping — the five characters `&lt;`
/// reach the output unchanged instead of coming out as `&amp;lt;`.
#[test]
fn a_text_valued_template_carries_disabled_escaping_into_the_result() {
    let xsl = stylesheet(
        r#"<xsl:template match="/"><out><xsl:apply-templates select="in" mode="doe"/></out></xsl:template>
           <xsl:template match="in" mode="doe" as="text()">
             <xsl:value-of select="'&amp;lt;'" disable-output-escaping="yes"/>
           </xsl:template>"#,
    );
    assert_eq!(
        run(&xsl).expect("the transformation runs"),
        "<out>&lt;</out>"
    );
}

/// doe-0183: the very same text node stored in a variable that builds a tree loses the request,
/// so the `<` it holds comes back out escaped.
#[test]
fn a_temporary_tree_drops_disabled_escaping() {
    let xsl = stylesheet(
        r#"<xsl:template match="/"><out>
             <xsl:variable name="x">
               <xsl:value-of select="'&lt;'" disable-output-escaping="yes"/>
             </xsl:variable>
             <xsl:copy-of select="$x"/>
           </out></xsl:template>"#,
    );
    assert_eq!(
        run(&xsl).expect("the transformation runs"),
        "<out>&lt;</out>"
    );
}

/// doe-0185: the same, one element deeper — a temporary tree drops the request wherever in it
/// the text node sits.
#[test]
fn a_temporary_tree_drops_disabled_escaping_below_an_element() {
    let xsl = stylesheet(
        r#"<xsl:template match="/"><out>
             <xsl:variable name="x">
               <elem><xsl:value-of select="'&lt;'" disable-output-escaping="yes"/></elem>
             </xsl:variable>
             <xsl:copy-of select="$x"/>
           </out></xsl:template>"#,
    );
    assert_eq!(
        run(&xsl).expect("the transformation runs"),
        "<out><elem>&lt;</elem></out>"
    );
}

/// doe-0190 and doe-0191: an `xsl:try` body is a sequence, so the request survives it both for a
/// bare text node and for one inside a constructed element.
#[test]
fn an_xsl_try_body_carries_disabled_escaping() {
    let xsl = stylesheet(
        r#"<xsl:template match="/"><out>
             <xsl:try>
               <z><xsl:value-of select="'&amp;lt;'" disable-output-escaping="yes"/></z>
               <xsl:value-of select="'&amp;gt;'" disable-output-escaping="yes"/>
               <xsl:catch errors="*"><error/></xsl:catch>
             </xsl:try>
           </out></xsl:template>"#,
    );
    assert_eq!(
        run(&xsl).expect("the transformation runs"),
        "<out><z>&lt;</z>&gt;</out>"
    );
}

/// The negative control for the whole cluster: a text node the source document supplies never
/// carries the request, however it is copied.
#[test]
fn source_text_never_carries_disabled_escaping() {
    let xsl = stylesheet(
        r#"<xsl:template match="/"><out><xsl:copy-of select="in/node()"/></out></xsl:template>"#,
    );
    assert_eq!(
        run_source(&xsl, "<in>&amp;lt;</in>").expect("the transformation runs"),
        "<out>&amp;lt;</out>"
    );
}

// === Declared modes: on-multiple-match and typed (XSLT 3.0 §6.6.2) ============================

/// error-0540a: `xsl:mode/@on-multiple-match="fail"` turns an ambiguous template-rule match into
/// `XTDE0540` instead of the default recovery, which keeps the last of the tied rules. `a[b]`
/// and `a[*]` both carry the default priority of a predicated name test, so nothing separates
/// them.
#[test]
fn an_ambiguous_match_is_xtde0540_when_the_mode_says_fail() {
    let xsl = stylesheet(
        r#"<xsl:mode on-multiple-match="fail"/>
           <xsl:template match="/"><out><xsl:apply-templates select="a"/></out></xsl:template>
           <xsl:template match="a[b]">first</xsl:template>
           <xsl:template match="a[*]">second</xsl:template>"#,
    );
    assert_eq!(run_error(&xsl, "<a><b/></a>").as_deref(), Some("XTDE0540"));
}

/// The negative control: without the declaration the same stylesheet recovers, and the rule
/// declared last wins.
#[test]
fn an_ambiguous_match_recovers_by_default() {
    let xsl = stylesheet(
        r#"<xsl:template match="/"><out><xsl:apply-templates select="a"/></out></xsl:template>
           <xsl:template match="a[b]">first</xsl:template>
           <xsl:template match="a[*]">second</xsl:template>"#,
    );
    assert_eq!(
        run_source(&xsl, "<a><b/></a>").expect("the transformation runs"),
        "<out>second</out>"
    );
}

/// The other negative control: `on-multiple-match="fail"` says nothing about rules of *different*
/// priority, which the conflict-resolution algorithm separates on its own.
#[test]
fn distinct_priorities_are_not_an_ambiguous_match() {
    let xsl = stylesheet(
        r#"<xsl:mode on-multiple-match="fail"/>
           <xsl:template match="/"><out><xsl:apply-templates select="a"/></out></xsl:template>
           <xsl:template match="a[b]" priority="2">first</xsl:template>
           <xsl:template match="a[*]" priority="1">second</xsl:template>"#,
    );
    assert_eq!(
        run_source(&xsl, "<a><b/></a>").expect("the transformation runs"),
        "<out>first</out>"
    );
}

/// mode-1439: a mode declared `typed="yes"` expects the nodes processed in it to have been
/// validated against a schema; an untyped source node is `XTTE3100`.
#[test]
fn a_typed_mode_rejects_an_untyped_node() {
    let xsl = stylesheet(
        r#"<xsl:mode typed="yes"/>
           <xsl:template match="/"><out><xsl:apply-templates select="in"/></out></xsl:template>
           <xsl:template match="in">A</xsl:template>"#,
    );
    assert_eq!(run_error(&xsl, "<in/>").as_deref(), Some("XTTE3100"));
}

/// The negative control: an undeclared mode makes no such demand.
#[test]
fn an_untyped_mode_accepts_an_untyped_node() {
    let xsl = stylesheet(
        r#"<xsl:template match="/"><out><xsl:apply-templates select="in"/></out></xsl:template>
           <xsl:template match="in">A</xsl:template>"#,
    );
    assert_eq!(run(&xsl).expect("the transformation runs"), "<out>A</out>");
}

/// doe-0184: a *stylesheet function* body is temporary output state, so the request is ignored
/// there however the text node is written — `xsl:text` no less than `xsl:value-of` — and the
/// copy into the final result must not resurrect it.
#[test]
fn a_stylesheet_function_body_ignores_disabled_escaping() {
    let xsl = stylesheet(
        r#"<xsl:function name="f:raw" as="text()">
             <xsl:text disable-output-escaping="yes">&lt;</xsl:text>
           </xsl:function>
           <xsl:function name="f:raw2" as="text()">
             <xsl:param name="in" as="xs:string"/>
             <xsl:value-of select="$in" disable-output-escaping="yes"/>
           </xsl:function>
           <xsl:template match="/"><out>
             <xsl:copy-of select="f:raw()"/><xsl:copy-of select="f:raw2('&lt;')"/>
           </out></xsl:template>"#,
    );
    assert_eq!(
        run(&xsl).expect("the transformation runs"),
        "<out>&lt;&lt;</out>"
    );
}

// === The result-tree bookkeeping errors of XSLT 3.0 §5.7.1 ====================================

/// error-0430a: two namespace nodes mapping one prefix to two namespaces cannot both be in scope
/// on the element being constructed.
#[test]
fn two_bindings_of_one_prefix_are_xtde0430() {
    let xsl = stylesheet(
        r#"<xsl:template match="/"><out>
             <xsl:namespace name="n">tweedledee</xsl:namespace>
             <xsl:namespace name="n">tweedledum</xsl:namespace>
           </out></xsl:template>"#,
    );
    assert_eq!(run_error(&xsl, "<in/>").as_deref(), Some("XTDE0430"));
}

/// The negative control: declaring the very same binding twice is not a conflict.
#[test]
fn one_binding_declared_twice_is_absorbed() {
    let xsl = stylesheet(
        r#"<xsl:template match="/"><x:out xmlns:x="urn:x">
             <xsl:namespace name="n">urn:n</xsl:namespace>
             <xsl:namespace name="n">urn:n</xsl:namespace>
           </x:out></xsl:template>"#,
    );
    assert!(run(&xsl).is_ok());
}

/// error-0440a and error-0440b: a default namespace may not be declared on an element that is
/// itself in no namespace, since the element's own name would then mean something else.
#[test]
fn a_default_namespace_on_a_no_namespace_element_is_xtde0440() {
    let xsl = stylesheet(
        r#"<xsl:template match="/"><out>
             <xsl:namespace name="">urn:a</xsl:namespace>
           </out></xsl:template>"#,
    );
    assert_eq!(run_error(&xsl, "<in/>").as_deref(), Some("XTDE0440"));
}

/// The negative control: the same declaration on an element that *is* in a namespace is fine.
#[test]
fn a_default_namespace_on_a_namespaced_element_is_allowed() {
    let xsl = stylesheet(
        r#"<xsl:template match="/"><x:out xmlns:x="urn:x">
             <xsl:namespace name="">urn:a</xsl:namespace>
           </x:out></xsl:template>"#,
    );
    assert!(run(&xsl).is_ok());
}

/// error-0420a: `xsl:copy` applied to a document node builds a document node, and an attribute
/// among its content is `XTDE0420` — not an attribute of whatever element encloses the
/// instruction.
#[test]
fn an_attribute_in_the_content_of_a_copied_document_node_is_xtde0420() {
    let xsl = stylesheet(
        r#"<xsl:template match="/"><a><xsl:copy>
             <xsl:attribute name="x">5</xsl:attribute>
           </xsl:copy></a></xsl:template>"#,
    );
    assert_eq!(run_error(&xsl, "<in/>").as_deref(), Some("XTDE0420"));
}

/// copy-1210: a namespace node is refused there for the same reason.
#[test]
fn a_namespace_in_the_content_of_a_copied_document_node_is_xtde0420() {
    let xsl = stylesheet(
        r#"<xsl:template match="/"><out><xsl:copy>
             <xsl:namespace name="n">urn:n</xsl:namespace>
           </xsl:copy></out></xsl:template>"#,
    );
    assert_eq!(run_error(&xsl, "<in/>").as_deref(), Some("XTDE0420"));
}

/// The negative control: ordinary content still copies through, so the identity template keeps
/// working.
#[test]
fn ordinary_content_still_flows_through_a_copied_document_node() {
    let xsl = stylesheet(
        r#"<xsl:template match="/"><out><xsl:copy>
             <kept/>text
           </xsl:copy></out></xsl:template>"#,
    );
    assert_eq!(
        run(&xsl).expect("the transformation runs"),
        "<out><kept/>text\n           </out>"
    );
}
