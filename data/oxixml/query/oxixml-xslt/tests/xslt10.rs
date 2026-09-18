//! XSLT 1.0 behaviour, end to end.
//!
//! Each case is a whole stylesheet and a whole source document, transformed and compared
//! against the text a conforming processor writes. Together they cover the instruction set,
//! the template-rule conflict rules, the built-in rules, patterns, `xsl:number`, `xsl:sort`,
//! keys, whitespace stripping and the serializer.

use oxixml_infoset::Tree;
use oxixml_xslt::{Stylesheet, XsltError};

/// Wraps a body in a stylesheet that omits the XML declaration.
fn stylesheet(body: &str) -> String {
    format!(
        r#"<xsl:stylesheet version="1.0" xmlns:xsl="http://www.w3.org/1999/XSL/Transform">
             <xsl:output omit-xml-declaration="yes"/>
             {body}
           </xsl:stylesheet>"#
    )
}

/// Transforms `source` with a stylesheet built from `body`.
fn run(body: &str, source: &str) -> Result<String, XsltError> {
    let compiled = Stylesheet::compile_str(&stylesheet(body))?;
    let tree = Tree::parse_str(source)
        .map_err(|error| XsltError::dynamic("XTDE0000", error.message().to_owned()))?;
    compiled.transform_to_string(&tree.root())
}

/// Asserts the transformation writes exactly `expected`.
fn check(body: &str, source: &str, expected: &str) {
    match run(body, source) {
        Ok(actual) => assert_eq!(actual, expected, "stylesheet body: {body}"),
        Err(error) => panic!("{error} (stylesheet body: {body})"),
    }
}

#[test]
fn a_literal_result_element_is_copied_to_the_output() {
    check(
        r#"<xsl:template match="/"><out/></xsl:template>"#,
        "<in/>",
        "<out/>",
    );
}

#[test]
fn value_of_takes_the_string_value_of_the_first_node() {
    check(
        r#"<xsl:template match="/"><o><xsl:value-of select="//b"/></o></xsl:template>"#,
        "<a><b>one</b><b>two</b></a>",
        "<o>one</o>",
    );
}

#[test]
fn attribute_value_templates_are_evaluated() {
    check(
        r#"<xsl:template match="/a"><o n="{@n}" k="x{1+1}y"/></xsl:template>"#,
        r#"<a n="7"/>"#,
        r#"<o n="7" k="x2y"/>"#,
    );
}

#[test]
fn the_built_in_rules_copy_text_and_recurse() {
    check(
        r#"<xsl:template match="b"><B><xsl:apply-templates/></B></xsl:template>"#,
        "<a>x<b>y</b>z</a>",
        "x<B>y</B>z",
    );
}

#[test]
fn template_priority_resolves_a_conflict() {
    check(
        concat!(
            r#"<xsl:template match="/"><xsl:apply-templates select="//b"/></xsl:template>"#,
            r#"<xsl:template match="b">low</xsl:template>"#,
            r#"<xsl:template match="a/b" priority="2">high</xsl:template>"#,
        ),
        "<a><b/></a>",
        "high",
    );
}

#[test]
fn a_more_specific_pattern_wins_by_default_priority() {
    check(
        concat!(
            r#"<xsl:template match="/"><xsl:apply-templates select="//b"/></xsl:template>"#,
            r#"<xsl:template match="*">star</xsl:template>"#,
            r#"<xsl:template match="b">named</xsl:template>"#,
        ),
        "<a><b/></a>",
        "named",
    );
}

#[test]
fn union_patterns_get_a_priority_per_alternative() {
    check(
        concat!(
            r#"<xsl:template match="/"><xsl:apply-templates select="//b|//c"/></xsl:template>"#,
            r#"<xsl:template match="b|c">[<xsl:value-of select="name()"/>]</xsl:template>"#,
        ),
        "<a><b/><c/></a>",
        "[b][c]",
    );
}

#[test]
fn predicates_in_patterns_are_positional() {
    check(
        concat!(
            r#"<xsl:template match="/"><xsl:apply-templates select="//b"/></xsl:template>"#,
            r#"<xsl:template match="b[1]">first</xsl:template>"#,
            r#"<xsl:template match="b">other</xsl:template>"#,
        ),
        "<a><b/><b/><b/></a>",
        "firstotherother",
    );
}

#[test]
fn descendant_patterns_match_at_any_depth() {
    check(
        concat!(
            r#"<xsl:template match="/"><xsl:apply-templates select="//c"/></xsl:template>"#,
            r#"<xsl:template match="a//c">deep</xsl:template>"#,
        ),
        "<a><b><c/></b></a>",
        "deep",
    );
}

#[test]
fn attribute_and_kind_patterns_match() {
    check(
        concat!(
            r#"<xsl:template match="/"><xsl:apply-templates select="a/@n|a/text()|a/comment()"/>"#,
            r#"</xsl:template>"#,
            r#"<xsl:template match="@n">A</xsl:template>"#,
            r#"<xsl:template match="text()">T</xsl:template>"#,
            r#"<xsl:template match="comment()">C</xsl:template>"#,
        ),
        r#"<a n="1">t<!--c--></a>"#,
        "ATC",
    );
}

#[test]
fn choose_selects_the_first_true_branch() {
    check(
        concat!(
            r#"<xsl:template match="/a"><xsl:choose>"#,
            r#"<xsl:when test="@n = 1">one</xsl:when>"#,
            r#"<xsl:when test="@n = 2">two</xsl:when>"#,
            r#"<xsl:otherwise>many</xsl:otherwise>"#,
            r#"</xsl:choose></xsl:template>"#,
        ),
        r#"<a n="2"/>"#,
        "two",
    );
}

#[test]
fn for_each_changes_the_focus() {
    check(
        r#"<xsl:template match="/a"><xsl:for-each select="b">[<xsl:value-of
             select="position()"/>/<xsl:value-of select="last()"/>]</xsl:for-each></xsl:template>"#,
        "<a><b/><b/></a>",
        "[1/2][2/2]",
    );
}

#[test]
fn variables_are_visible_to_the_following_siblings() {
    check(
        concat!(
            r#"<xsl:template match="/"><xsl:variable name="v" select="1 + 1"/>"#,
            r#"<o><xsl:value-of select="$v"/></o></xsl:template>"#,
        ),
        "<in/>",
        "<o>2</o>",
    );
}

#[test]
fn a_variable_with_content_is_a_result_tree_fragment() {
    check(
        concat!(
            r#"<xsl:template match="/"><xsl:variable name="v"><i>x</i></xsl:variable>"#,
            r#"<o><xsl:value-of select="$v"/><xsl:copy-of select="$v"/></o></xsl:template>"#,
        ),
        "<in/>",
        "<o>x<i>x</i></o>",
    );
}

#[test]
fn named_templates_take_parameters() {
    check(
        concat!(
            r#"<xsl:template match="/"><xsl:call-template name="t">"#,
            r#"<xsl:with-param name="n" select="3"/></xsl:call-template></xsl:template>"#,
            r#"<xsl:template name="t"><xsl:param name="n" select="0"/>"#,
            r#"<o><xsl:value-of select="$n"/></o></xsl:template>"#,
        ),
        "<in/>",
        "<o>3</o>",
    );
}

#[test]
fn a_recursive_named_template_terminates() {
    check(
        concat!(
            r#"<xsl:template match="/"><xsl:call-template name="down">"#,
            r#"<xsl:with-param name="n" select="4"/></xsl:call-template></xsl:template>"#,
            r#"<xsl:template name="down"><xsl:param name="n"/>"#,
            r#"<xsl:if test="$n > 0"><xsl:value-of select="$n"/>"#,
            r#"<xsl:call-template name="down"><xsl:with-param name="n" select="$n - 1"/>"#,
            r#"</xsl:call-template></xsl:if></xsl:template>"#,
        ),
        "<in/>",
        "4321",
    );
}

#[test]
fn computed_elements_and_attributes_are_constructed() {
    check(
        concat!(
            r#"<xsl:template match="/a"><xsl:element name="{@e}">"#,
            r#"<xsl:attribute name="{@k}">v</xsl:attribute>t</xsl:element></xsl:template>"#,
        ),
        r#"<a e="out" k="n"/>"#,
        r#"<out n="v">t</out>"#,
    );
}

#[test]
fn comments_and_processing_instructions_are_constructed() {
    check(
        concat!(
            r#"<xsl:template match="/"><o><xsl:comment>c</xsl:comment>"#,
            r#"<xsl:processing-instruction name="p">d</xsl:processing-instruction>"#,
            r#"</o></xsl:template>"#,
        ),
        "<in/>",
        "<o><!--c--><?p d?></o>",
    );
}

#[test]
fn copy_is_shallow_and_copy_of_is_deep() {
    check(
        concat!(
            r#"<xsl:template match="/a"><xsl:copy><xsl:copy-of select="b"/></xsl:copy>"#,
            r#"</xsl:template>"#,
        ),
        r#"<a n="1"><b m="2"><c/></b></a>"#,
        r#"<a><b m="2"><c/></b></a>"#,
    );
}

#[test]
fn the_identity_transform_round_trips() {
    check(
        concat!(
            r#"<xsl:template match="@*|node()"><xsl:copy>"#,
            r#"<xsl:apply-templates select="@*|node()"/></xsl:copy></xsl:template>"#,
        ),
        r#"<a n="1"><b>t</b><!--c--></a>"#,
        r#"<a n="1"><b>t</b><!--c--></a>"#,
    );
}

#[test]
fn sort_orders_by_one_key() {
    check(
        concat!(
            r#"<xsl:template match="/a"><xsl:for-each select="b">"#,
            r#"<xsl:sort select="."/><xsl:value-of select="."/></xsl:for-each></xsl:template>"#,
        ),
        "<a><b>c</b><b>a</b><b>b</b></a>",
        "abc",
    );
}

#[test]
fn sort_honours_data_type_and_order() {
    check(
        concat!(
            r#"<xsl:template match="/a"><xsl:for-each select="b">"#,
            r#"<xsl:sort select="." data-type="number" order="descending"/>"#,
            r#"<xsl:value-of select="."/>,</xsl:for-each></xsl:template>"#,
        ),
        "<a><b>9</b><b>10</b><b>2</b></a>",
        "10,9,2,",
    );
}

#[test]
fn two_sort_keys_break_ties() {
    check(
        concat!(
            r#"<xsl:template match="/a"><xsl:for-each select="b">"#,
            r#"<xsl:sort select="@g"/><xsl:sort select="." data-type="number"/>"#,
            r#"<xsl:value-of select="concat(@g, .)"/><xsl:text> </xsl:text>"#,
            r#"</xsl:for-each></xsl:template>"#,
        ),
        r#"<a><b g="y">2</b><b g="x">3</b><b g="x">1</b></a>"#,
        "x1 x3 y2 ",
    );
}

#[test]
fn keys_index_the_document() {
    check(
        concat!(
            r#"<xsl:key name="k" match="b" use="@id"/>"#,
            r#"<xsl:template match="/"><o><xsl:value-of select="key('k', 'two')"/></o>"#,
            r#"</xsl:template>"#,
        ),
        r#"<a><b id="one">1</b><b id="two">2</b></a>"#,
        "<o>2</o>",
    );
}

#[test]
fn number_counts_siblings() {
    check(
        concat!(
            r#"<xsl:template match="/a"><xsl:for-each select="b">"#,
            r#"<xsl:number format="1. "/></xsl:for-each></xsl:template>"#,
        ),
        "<a><b/><b/><b/></a>",
        "1. 2. 3. ",
    );
}

#[test]
fn number_at_level_multiple_walks_the_ancestors() {
    check(
        concat!(
            r#"<xsl:template match="/"><xsl:for-each select="//c">"#,
            r#"<xsl:number level="multiple" count="b|c" format="1.1"/> </xsl:for-each>"#,
            r#"</xsl:template>"#,
        ),
        "<a><b><c/><c/></b><b><c/></b></a>",
        "1.11.22.1",
    );
}

#[test]
fn number_at_level_any_counts_the_whole_document() {
    check(
        concat!(
            r#"<xsl:template match="/"><xsl:for-each select="//c">"#,
            r#"<xsl:number level="any" count="c"/> </xsl:for-each></xsl:template>"#,
        ),
        "<a><b><c/></b><b><c/><c/></b></a>",
        "123",
    );
}

#[test]
fn number_takes_an_explicit_value_and_a_format() {
    check(
        r#"<xsl:template match="/"><xsl:number value="4" format="I"/></xsl:template>"#,
        "<in/>",
        "IV",
    );
}

#[test]
fn modes_keep_rule_sets_apart() {
    check(
        concat!(
            r#"<xsl:template match="/"><xsl:apply-templates select="//b"/>"#,
            r#"<xsl:apply-templates select="//b" mode="m"/></xsl:template>"#,
            r#"<xsl:template match="b">plain</xsl:template>"#,
            r#"<xsl:template match="b" mode="m">moded</xsl:template>"#,
        ),
        "<a><b/></a>",
        "plainmoded",
    );
}

#[test]
fn attribute_sets_are_applied_and_chained() {
    check(
        concat!(
            r#"<xsl:attribute-set name="base"><xsl:attribute name="a">1</xsl:attribute>"#,
            r#"</xsl:attribute-set>"#,
            r#"<xsl:attribute-set name="more" use-attribute-sets="base">"#,
            r#"<xsl:attribute name="b">2</xsl:attribute></xsl:attribute-set>"#,
            r#"<xsl:template match="/"><o xsl:use-attribute-sets="more"/></xsl:template>"#,
        ),
        "<in/>",
        r#"<o a="1" b="2"/>"#,
    );
}

#[test]
fn strip_space_removes_whitespace_only_text() {
    check(
        concat!(
            r#"<xsl:strip-space elements="a"/>"#,
            r#"<xsl:template match="/a"><o><xsl:value-of select="count(node())"/></o>"#,
            r#"</xsl:template>"#,
        ),
        "<a>\n  <b/>\n</a>",
        "<o>1</o>",
    );
}

#[test]
fn preserve_space_overrides_strip_space() {
    check(
        concat!(
            r#"<xsl:strip-space elements="*"/><xsl:preserve-space elements="a"/>"#,
            r#"<xsl:template match="/a"><o><xsl:value-of select="count(node())"/></o>"#,
            r#"</xsl:template>"#,
        ),
        "<a>\n  <b/>\n</a>",
        "<o>3</o>",
    );
}

#[test]
fn text_output_writes_only_character_data() {
    check(
        concat!(
            r#"<xsl:output method="text"/>"#,
            r#"<xsl:template match="/"><o>a<i>b</i></o></xsl:template>"#,
        ),
        "<in/>",
        "ab",
    );
}

#[test]
fn html_output_uses_html_conventions() {
    check(
        concat!(
            r#"<xsl:output method="html"/>"#,
            r#"<xsl:template match="/"><html><head><br/></head><p>a &amp; b</p></html>"#,
            r#"</xsl:template>"#,
        ),
        "<in/>",
        // `include-content-type` defaults to `yes`, so the serializer states the encoding.
        concat!(
            r#"<html><head><meta http-equiv="Content-Type" content="text/html; charset=UTF-8">"#,
            "<br></head><p>a &amp; b</p></html>",
        ),
    );
}

#[test]
fn a_doctype_declaration_is_written() {
    check(
        concat!(
            r#"<xsl:output doctype-system="out.dtd" doctype-public="-//X//EN"/>"#,
            r#"<xsl:template match="/"><out/></xsl:template>"#,
        ),
        "<in/>",
        "<!DOCTYPE out PUBLIC \"-//X//EN\" \"out.dtd\">\n<out/>",
    );
}

#[test]
fn cdata_section_elements_wrap_their_text() {
    check(
        concat!(
            r#"<xsl:output cdata-section-elements="o"/>"#,
            r#"<xsl:template match="/"><o>a &lt; b</o></xsl:template>"#,
        ),
        "<in/>",
        "<o><![CDATA[a < b]]></o>",
    );
}

#[test]
fn disable_output_escaping_writes_markup_verbatim() {
    check(
        concat!(
            r#"<xsl:template match="/"><o><xsl:text disable-output-escaping="yes">"#,
            r#"&lt;raw/&gt;</xsl:text></o></xsl:template>"#,
        ),
        "<in/>",
        "<o><raw/></o>",
    );
}

#[test]
fn decimal_format_drives_format_number() {
    check(
        concat!(
            r#"<xsl:decimal-format name="euro" decimal-separator="," grouping-separator="."/>"#,
            r#"<xsl:template match="/"><o><xsl:value-of "#,
            r#"select="format-number(1234.5, '#.##0,00', 'euro')"/></o></xsl:template>"#,
        ),
        "<in/>",
        "<o>1.234,50</o>",
    );
}

#[test]
fn apply_imports_is_not_available_without_an_import() {
    check(
        concat!(
            r#"<xsl:template match="/"><xsl:apply-templates select="a"/></xsl:template>"#,
            r#"<xsl:template match="a">x<xsl:apply-imports/></xsl:template>"#,
        ),
        "<a>t</a>",
        "xt",
    );
}

#[test]
fn namespaces_are_carried_onto_literal_result_elements() {
    check(
        r#"<xsl:template match="/"><o xmlns="urn:x"><i/></o></xsl:template>"#,
        "<in/>",
        r#"<o xmlns="urn:x"><i/></o>"#,
    );
}

#[test]
fn namespace_alias_rewrites_a_namespace() {
    check(
        concat!(
            r#"<xsl:namespace-alias xmlns:a="urn:a" xmlns:b="urn:b""#,
            r#" stylesheet-prefix="a" result-prefix="b"/>"#,
            r#"<xsl:template match="/" xmlns:a="urn:a">"#,
            r#"<a:o/></xsl:template>"#,
        ),
        "<in/>",
        r#"<a:o xmlns:a="urn:b"/>"#,
    );
}

#[test]
fn message_with_terminate_stops_the_transformation() {
    let failure = run(
        r#"<xsl:template match="/"><xsl:message terminate="yes">stop</xsl:message></xsl:template>"#,
        "<in/>",
    )
    .expect_err("terminate=\"yes\" is an error");
    assert_eq!(failure.code(), Some("XTMM9000"));
    assert_eq!(failure.message(), "stop");
}

#[test]
fn a_simplified_stylesheet_is_one_template() -> Result<(), XsltError> {
    let compiled = Stylesheet::compile_str(
        r#"<out xmlns:xsl="http://www.w3.org/1999/XSL/Transform" xsl:version="1.0">
             <xsl:value-of select="/a"/>
           </out>"#,
    )?;
    let source = Tree::parse_str("<a>x</a>")
        .map_err(|error| XsltError::dynamic("XTDE0000", error.message().to_owned()))?;
    let text = compiled.transform_to_string(&source.root())?;
    assert!(text.contains("<out>"), "{text}");
    assert!(text.contains('x'), "{text}");
    Ok(())
}

#[test]
fn a_malformed_pattern_is_a_static_error() {
    let failure =
        Stylesheet::compile_str(&stylesheet(r#"<xsl:template match="a[">x</xsl:template>"#))
            .expect_err("the pattern does not parse");
    assert_eq!(failure.phase(), oxixml_xslt::ErrorPhase::Static);
}

#[test]
fn an_unknown_xslt_element_is_a_static_error() {
    let failure = Stylesheet::compile_str(&stylesheet(
        r#"<xsl:template match="/"><xsl:nonesuch/></xsl:template>"#,
    ))
    .expect_err("xsl:nonesuch is not an instruction");
    assert_eq!(failure.code(), Some("XTSE0010"));
}

#[test]
fn an_unknown_instruction_falls_back() {
    check(
        concat!(
            r#"<xsl:template match="/"><xsl:evaluate xpath="1">"#,
            r#"<xsl:fallback>fell back</xsl:fallback></xsl:evaluate></xsl:template>"#,
        ),
        "<in/>",
        "fell back",
    );
}
