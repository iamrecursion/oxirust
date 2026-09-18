//! The XSLT 2.0 and 3.0 constructs this processor implements.

use oxixml_infoset::{AtomicValue, Tree};
use oxixml_xslt::{MemoryResolver, Parameters, StaticOptions, Stylesheet, XsltError};
use std::sync::Arc;

fn stylesheet(body: &str) -> String {
    format!(
        r#"<xsl:stylesheet version="2.0" xmlns:xsl="http://www.w3.org/1999/XSL/Transform"
             xmlns:xs="http://www.w3.org/2001/XMLSchema"
             xmlns:f="urn:f" exclude-result-prefixes="xs f">
             <xsl:output omit-xml-declaration="yes"/>
             {body}
           </xsl:stylesheet>"#
    )
}

fn run(body: &str, source: &str) -> Result<String, XsltError> {
    let compiled = Stylesheet::compile_str(&stylesheet(body))?;
    let tree = Tree::parse_str(source)
        .map_err(|error| XsltError::dynamic("XTDE0000", error.message().to_owned()))?;
    compiled.transform_to_string(&tree.root())
}

fn check(body: &str, source: &str, expected: &str) {
    match run(body, source) {
        Ok(actual) => assert_eq!(actual, expected, "stylesheet body: {body}"),
        Err(error) => panic!("{error} (stylesheet body: {body})"),
    }
}

#[test]
fn value_of_joins_a_sequence_with_a_separator() {
    check(
        r#"<xsl:template match="/a"><o><xsl:value-of select="b" separator=","/></o></xsl:template>"#,
        "<a><b>1</b><b>2</b></a>",
        "<o>1,2</o>",
    );
}

#[test]
fn sequence_returns_atomic_values() {
    check(
        concat!(
            r#"<xsl:template match="/"><xsl:variable name="v" as="xs:integer*">"#,
            r#"<xsl:sequence select="1, 2, 3"/></xsl:variable>"#,
            r#"<o><xsl:value-of select="count($v)"/>:<xsl:value-of select="sum($v)"/></o>"#,
            r#"</xsl:template>"#,
        ),
        "<in/>",
        "<o>3:6</o>",
    );
}

#[test]
fn a_stylesheet_function_is_callable_from_an_expression() {
    check(
        concat!(
            r#"<xsl:function name="f:double" as="xs:integer">"#,
            r#"<xsl:param name="n" as="xs:integer"/>"#,
            r#"<xsl:sequence select="$n * 2"/></xsl:function>"#,
            r#"<xsl:template match="/"><o><xsl:value-of select="f:double(21)"/></o>"#,
            r#"</xsl:template>"#,
        ),
        "<in/>",
        "<o>42</o>",
    );
}

#[test]
fn a_recursive_stylesheet_function_terminates() {
    check(
        concat!(
            r#"<xsl:function name="f:sum" as="xs:integer">"#,
            r#"<xsl:param name="n" as="xs:integer"/>"#,
            r#"<xsl:sequence select="if ($n le 0) then 0 else $n + f:sum($n - 1)"/>"#,
            r#"</xsl:function>"#,
            r#"<xsl:template match="/"><o><xsl:value-of select="f:sum(10)"/></o>"#,
            r#"</xsl:template>"#,
        ),
        "<in/>",
        "<o>55</o>",
    );
}

#[test]
fn for_each_group_groups_by_a_key() {
    check(
        concat!(
            r#"<xsl:template match="/a"><xsl:for-each-group select="b" group-by="@g">"#,
            r#"<g k="{current-grouping-key()}" n="{count(current-group())}"/>"#,
            r#"</xsl:for-each-group></xsl:template>"#,
        ),
        r#"<a><b g="x"/><b g="y"/><b g="x"/></a>"#,
        r#"<g k="x" n="2"/><g k="y" n="1"/>"#,
    );
}

#[test]
fn for_each_group_groups_adjacent_items() {
    check(
        concat!(
            r#"<xsl:template match="/a"><xsl:for-each-group select="b" group-adjacent="@g">"#,
            r#"<g k="{current-grouping-key()}" n="{count(current-group())}"/>"#,
            r#"</xsl:for-each-group></xsl:template>"#,
        ),
        r#"<a><b g="x"/><b g="x"/><b g="y"/><b g="x"/></a>"#,
        r#"<g k="x" n="2"/><g k="y" n="1"/><g k="x" n="1"/>"#,
    );
}

#[test]
fn for_each_group_starting_and_ending_with_a_pattern() {
    check(
        concat!(
            r#"<xsl:template match="/a"><xsl:for-each-group select="*" group-starting-with="h">"#,
            r#"<g n="{count(current-group())}"/></xsl:for-each-group></xsl:template>"#,
        ),
        "<a><h/><p/><p/><h/><p/></a>",
        r#"<g n="3"/><g n="2"/>"#,
    );
    check(
        concat!(
            r#"<xsl:template match="/a"><xsl:for-each-group select="*" group-ending-with="e">"#,
            r#"<g n="{count(current-group())}"/></xsl:for-each-group></xsl:template>"#,
        ),
        "<a><p/><e/><p/><p/><e/></a>",
        r#"<g n="2"/><g n="3"/>"#,
    );
}

#[test]
fn analyze_string_splits_on_a_regular_expression() {
    check(
        concat!(
            r#"<xsl:template match="/a"><xsl:analyze-string select="." regex="([0-9]+)">"#,
            r#"<xsl:matching-substring>[<xsl:value-of select="regex-group(1)"/>]"#,
            r#"</xsl:matching-substring>"#,
            r#"<xsl:non-matching-substring>(<xsl:value-of select="."/>)"#,
            r#"</xsl:non-matching-substring>"#,
            r#"</xsl:analyze-string></xsl:template>"#,
        ),
        "<a>a12b3</a>",
        "(a)[12](b)[3]",
    );
}

#[test]
fn next_match_falls_through_to_the_next_rule() {
    check(
        concat!(
            r#"<xsl:template match="/"><xsl:apply-templates select="//b"/></xsl:template>"#,
            r#"<xsl:template match="b" priority="2">A<xsl:next-match/></xsl:template>"#,
            r#"<xsl:template match="b" priority="1">B</xsl:template>"#,
        ),
        "<a><b/></a>",
        "AB",
    );
}

#[test]
fn tunnel_parameters_pass_through_intermediate_templates() {
    check(
        concat!(
            r#"<xsl:template match="/"><xsl:apply-templates select="//b">"#,
            r#"<xsl:with-param name="t" select="'deep'" tunnel="yes"/></xsl:apply-templates>"#,
            r#"</xsl:template>"#,
            r#"<xsl:template match="b"><xsl:apply-templates select="c"/></xsl:template>"#,
            r#"<xsl:template match="c"><xsl:param name="t" tunnel="yes"/>"#,
            r#"<o><xsl:value-of select="$t"/></o></xsl:template>"#,
        ),
        "<a><b><c/></b></a>",
        "<o>deep</o>",
    );
}

#[test]
fn a_mode_declaration_selects_the_no_match_policy() {
    check(
        concat!(
            r#"<xsl:mode name="m" on-no-match="shallow-copy"/>"#,
            r#"<xsl:template match="/"><xsl:apply-templates select="a" mode="m"/></xsl:template>"#,
        ),
        r#"<a n="1"><b>t</b></a>"#,
        r#"<a n="1"><b>t</b></a>"#,
    );
}

#[test]
fn iterate_carries_parameters_and_breaks() {
    check(
        concat!(
            r#"<xsl:template match="/a"><xsl:iterate select="b">"#,
            r#"<xsl:param name="total" select="0"/>"#,
            r#"<xsl:on-completion><o><xsl:value-of select="$total"/></o></xsl:on-completion>"#,
            r#"<xsl:next-iteration><xsl:with-param name="total" select="$total + ."/>"#,
            r#"</xsl:next-iteration></xsl:iterate></xsl:template>"#,
        ),
        "<a><b>1</b><b>2</b><b>3</b></a>",
        "<o>6</o>",
    );
}

#[test]
fn perform_sort_returns_a_sorted_sequence() {
    check(
        concat!(
            r#"<xsl:template match="/a"><o><xsl:perform-sort select="b">"#,
            r#"<xsl:sort select="." data-type="number"/></xsl:perform-sort></o>"#,
            r#"</xsl:template>"#,
        ),
        "<a><b>3</b><b>1</b><b>2</b></a>",
        "<o><b>1</b><b>2</b><b>3</b></o>",
    );
}

#[test]
fn where_populated_suppresses_an_empty_construction() {
    // An element with no children counts as unpopulated, so the wrapper disappears entirely.
    check(
        concat!(
            r#"<xsl:template match="/a"><xsl:where-populated><o>"#,
            r#"<xsl:value-of select="missing"/></o></xsl:where-populated>done</xsl:template>"#,
        ),
        "<a/>",
        "done",
    );
    // With content, it survives.
    check(
        concat!(
            r#"<xsl:template match="/a"><xsl:where-populated><o>"#,
            r#"<xsl:value-of select="b"/></o></xsl:where-populated>done</xsl:template>"#,
        ),
        "<a><b>x</b></a>",
        "<o>x</o>done",
    );
}

#[test]
fn a_character_map_rewrites_output() {
    check(
        concat!(
            r#"<xsl:output use-character-maps="m"/>"#,
            r#"<xsl:character-map name="m">"#,
            r#"<xsl:output-character character="§" string="[sec]"/></xsl:character-map>"#,
            r#"<xsl:template match="/"><o>a§b</o></xsl:template>"#,
        ),
        "<in/>",
        "<o>a[sec]b</o>",
    );
}

#[test]
fn result_documents_are_collected_separately() -> Result<(), XsltError> {
    let compiled = Stylesheet::compile_str(&stylesheet(concat!(
        r#"<xsl:template match="/"><main/>"#,
        r#"<xsl:result-document href="side.xml"><side/></xsl:result-document>"#,
        r#"</xsl:template>"#,
    )))?;
    let source = Tree::parse_str("<in/>")
        .map_err(|error| XsltError::dynamic("XTDE0000", error.message().to_owned()))?;
    let result = compiled.transform(&source.root())?;
    assert_eq!(result.as_str(), "<main/>");
    assert_eq!(result.secondary_results().len(), 1);
    assert_eq!(result.secondary_results()[0].0, "side.xml");
    assert!(result.secondary_results()[0].1.ends_with("<side/>"));
    Ok(())
}

#[test]
fn an_initial_template_is_the_entry_point() -> Result<(), XsltError> {
    let compiled = Stylesheet::compile_str(&stylesheet(
        r#"<xsl:template name="main"><o>started</o></xsl:template>"#,
    ))?;
    let source = Tree::parse_str("<in/>")
        .map_err(|error| XsltError::dynamic("XTDE0000", error.message().to_owned()))?;
    let result = compiled.transform_with(
        &source.root(),
        &Parameters::new().with_initial_template("main"),
    )?;
    assert_eq!(result.as_str(), "<o>started</o>");
    Ok(())
}

#[test]
fn the_initial_mode_can_be_chosen() -> Result<(), XsltError> {
    let compiled = Stylesheet::compile_str(&stylesheet(concat!(
        r#"<xsl:template match="/">default</xsl:template>"#,
        r#"<xsl:template match="/" mode="m">moded</xsl:template>"#,
    )))?;
    let source = Tree::parse_str("<in/>")
        .map_err(|error| XsltError::dynamic("XTDE0000", error.message().to_owned()))?;
    let result =
        compiled.transform_with(&source.root(), &Parameters::new().with_initial_mode("m"))?;
    assert_eq!(result.as_str(), "moded");
    Ok(())
}

#[test]
fn a_stylesheet_parameter_can_be_supplied() -> Result<(), XsltError> {
    let compiled = Stylesheet::compile_str(&stylesheet(concat!(
        r#"<xsl:param name="who" select="'nobody'"/>"#,
        r#"<xsl:template match="/"><o><xsl:value-of select="$who"/></o></xsl:template>"#,
    )))?;
    let source = Tree::parse_str("<in/>")
        .map_err(|error| XsltError::dynamic("XTDE0000", error.message().to_owned()))?;

    assert_eq!(
        compiled.transform_to_string(&source.root())?,
        "<o>nobody</o>"
    );

    let result = compiled.transform_with(
        &source.root(),
        &Parameters::new().with_parameter("who", AtomicValue::string("world")),
    )?;
    assert_eq!(result.as_str(), "<o>world</o>");
    Ok(())
}

#[test]
fn messages_are_collected() -> Result<(), XsltError> {
    let compiled = Stylesheet::compile_str(&stylesheet(
        r#"<xsl:template match="/"><xsl:message>hello</xsl:message><o/></xsl:template>"#,
    ))?;
    let source = Tree::parse_str("<in/>")
        .map_err(|error| XsltError::dynamic("XTDE0000", error.message().to_owned()))?;
    let result = compiled.transform(&source.root())?;
    assert_eq!(result.messages(), ["hello"]);
    Ok(())
}

#[test]
fn include_and_import_bring_in_another_module() -> Result<(), XsltError> {
    let resolver = Arc::new(MemoryResolver::new().with_document(
        "urn:base",
        r#"<xsl:stylesheet version="2.0" xmlns:xsl="http://www.w3.org/1999/XSL/Transform">
             <xsl:template match="b">imported</xsl:template>
             <xsl:template name="shared">from the import</xsl:template>
           </xsl:stylesheet>"#,
    ));
    let compiled = Stylesheet::compile_str_with(
        r#"<xsl:stylesheet version="2.0" xmlns:xsl="http://www.w3.org/1999/XSL/Transform">
             <xsl:import href="urn:base"/>
             <xsl:output omit-xml-declaration="yes"/>
             <xsl:template match="/">
               <xsl:apply-templates select="//b"/>|<xsl:call-template name="shared"/>
             </xsl:template>
           </xsl:stylesheet>"#,
        StaticOptions::new().with_resolver(resolver),
    )?;
    let source = Tree::parse_str("<a><b/></a>")
        .map_err(|error| XsltError::dynamic("XTDE0000", error.message().to_owned()))?;
    assert_eq!(
        compiled.transform_to_string(&source.root())?,
        "imported|from the import"
    );
    Ok(())
}

#[test]
fn an_importing_module_overrides_what_it_imports() -> Result<(), XsltError> {
    let resolver = Arc::new(MemoryResolver::new().with_document(
        "urn:base",
        r#"<xsl:stylesheet version="2.0" xmlns:xsl="http://www.w3.org/1999/XSL/Transform">
             <xsl:template match="b">base</xsl:template>
           </xsl:stylesheet>"#,
    ));
    let compiled = Stylesheet::compile_str_with(
        r#"<xsl:stylesheet version="2.0" xmlns:xsl="http://www.w3.org/1999/XSL/Transform">
             <xsl:import href="urn:base"/>
             <xsl:output omit-xml-declaration="yes"/>
             <xsl:template match="/"><xsl:apply-templates select="//b"/></xsl:template>
             <xsl:template match="b">derived(<xsl:apply-imports/>)</xsl:template>
           </xsl:stylesheet>"#,
        StaticOptions::new().with_resolver(resolver),
    )?;
    let source = Tree::parse_str("<a><b/></a>")
        .map_err(|error| XsltError::dynamic("XTDE0000", error.message().to_owned()))?;
    assert_eq!(
        compiled.transform_to_string(&source.root())?,
        "derived(base)"
    );
    Ok(())
}

#[test]
fn the_result_tree_is_navigable_as_a_document() -> Result<(), XsltError> {
    use oxixml_infoset::XdmNode;
    let compiled = Stylesheet::compile_str(&stylesheet(
        r#"<xsl:template match="/"><o><i>x</i></o></xsl:template>"#,
    ))?;
    let source = Tree::parse_str("<in/>")
        .map_err(|error| XsltError::dynamic("XTDE0000", error.message().to_owned()))?;
    let result = compiled.transform(&source.root())?;
    let root = result.root();
    assert_eq!(root.children().len(), 1);
    assert_eq!(root.string_value(), "x");
    assert_eq!(result.as_sequence().len(), 1);
    Ok(())
}

#[test]
fn indentation_lays_out_element_only_content() {
    check(
        concat!(
            r#"<xsl:output indent="yes"/>"#,
            r#"<xsl:template match="/"><o><i/><i/></o></xsl:template>"#,
        ),
        "<in/>",
        "<o>\n   <i/>\n   <i/>\n</o>\n",
    );
}

#[test]
fn xhtml_output_writes_xml_syntax_with_html_empty_elements() {
    check(
        concat!(
            r#"<xsl:output method="xhtml"/>"#,
            r#"<xsl:template match="/"><html><br/><p>t</p></html></xsl:template>"#,
        ),
        "<in/>",
        "<html><br /><p>t</p></html>",
    );
}

#[test]
fn system_property_reports_the_processor() {
    check(
        concat!(
            r#"<xsl:template match="/"><o v="{system-property('xsl:version')}""#,
            r#" a="{element-available('xsl:for-each')}""#,
            r#" f="{function-available('substring')}"/></xsl:template>"#,
        ),
        "<in/>",
        r#"<o v="2.0" a="true" f="true"/>"#,
    );
}

#[test]
fn current_reports_the_node_being_processed() {
    check(
        concat!(
            r#"<xsl:template match="/a"><xsl:for-each select="b">"#,
            r#"<xsl:value-of select="../b[. = current()/@n]/@n"/></xsl:for-each>"#,
            r#"</xsl:template>"#,
        ),
        r#"<a><b n="1">1</b><b n="2">2</b></a>"#,
        "12",
    );
}

#[test]
fn try_catches_a_dynamic_error_and_rolls_the_output_back() {
    check(
        concat!(
            r#"<xsl:template match="/"><o><xsl:try>"#,
            r#"<discarded/><xsl:value-of select="1 idiv 0"/>"#,
            r#"<xsl:catch errors="err:FOAR0001"><caught/></xsl:catch>"#,
            r#"</xsl:try></o></xsl:template>"#,
        ),
        "<in/>",
        "<o><caught/></o>",
    );
}

#[test]
fn a_catch_sees_the_error_variables() {
    check(
        concat!(
            r#"<xsl:template match="/"><xsl:try select="1 idiv 0">"#,
            r#"<xsl:catch select="local-name-from-QName($err:code)"/>"#,
            r#"</xsl:try></xsl:template>"#,
        ),
        "<in/>",
        "FOAR0001",
    );
}

#[test]
fn a_catch_that_does_not_admit_the_error_lets_it_through() {
    let error = run(
        concat!(
            r#"<xsl:template match="/"><xsl:try select="1 idiv 0">"#,
            r#"<xsl:catch errors="err:XPTY0004"><caught/></xsl:catch>"#,
            r#"</xsl:try></xsl:template>"#,
        ),
        "<in/>",
    )
    .expect_err("the error is not caught");
    assert_eq!(error.code(), Some("FOAR0001"));
}

#[test]
fn a_failed_assertion_raises_xtmm9001() {
    let error = run(
        r#"<xsl:template match="/"><xsl:assert test="false()"/></xsl:template>"#,
        "<in/>",
    )
    .expect_err("the assertion fails");
    assert_eq!(error.code(), Some("XTMM9001"));
}

#[test]
fn an_assertion_that_holds_produces_nothing() {
    check(
        r#"<xsl:template match="/"><o><xsl:assert test="true()"/></o></xsl:template>"#,
        "<in/>",
        "<o/>",
    );
}

#[test]
fn a_pattern_may_use_except_and_a_parenthesized_alternative() {
    check(
        concat!(
            r#"<xsl:template match="/"><o><xsl:apply-templates select="x/*"/></o></xsl:template>"#,
            r#"<xsl:template match="x/(a|b)"><hit><xsl:value-of select="name()"/></hit>"#,
            r#"</xsl:template>"#,
        ),
        "<x><a/><b/><c/></x>",
        "<o><hit>a</hit><hit>b</hit></o>",
    );
}

#[test]
fn xsl_number_keeps_the_prefix_and_suffix_of_an_empty_place_marker() {
    check(
        concat!(
            r#"<xsl:template match="/a"><o><xsl:number level="multiple" count="z" format="A."/>"#,
            r#"</o></xsl:template>"#,
        ),
        "<a/>",
        "<o>.</o>",
    );
}
