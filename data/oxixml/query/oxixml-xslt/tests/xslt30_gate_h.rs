//! Regression tests for result-document routing, grouping and key comparison, conditional
//! content, template-rule chaining and the availability functions.
//!
//! Every test reproduces one W3C `xslt30-test` case with a minimal inline stylesheet, so each
//! fix is provable red-then-green without the separate `oxixml-testsuite` harness. The case
//! identifiers in the comments are the ones that harness reports.

use oxixml_infoset::Tree;
use oxixml_xslt::{Parameters, Stylesheet, XsltError};

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

/// Runs with no source document, which is how the suite invokes a case whose `<test>` names no
/// environment.
fn run_without_source(xsl: &str) -> Result<String, XsltError> {
    let compiled = Stylesheet::compile_str(xsl)?;
    Ok(compiled
        .transform_optional(None, &Parameters::new())?
        .as_str()
        .to_owned())
}

/// The secondary results a transformation with no source document wrote.
fn secondary_of(xsl: &str) -> Result<Vec<(String, String)>, XsltError> {
    let compiled = Stylesheet::compile_str(xsl)?;
    Ok(compiled
        .transform_optional(None, &Parameters::new())?
        .secondary_results()
        .to_vec())
}

fn error_code(outcome: Result<String, XsltError>) -> String {
    match outcome {
        Ok(text) => format!("no error, produced `{text}`"),
        Err(error) => error.code().unwrap_or("no code").to_owned(),
    }
}

// ---------------------------------------------------------------------------------------------
// xsl:result-document and the principal result destination
// ---------------------------------------------------------------------------------------------

/// result-document-0205: a nested `xsl:result-document` with no `href` writes the *principal*
/// result, not more content of the secondary tree it is written inside.
#[test]
fn a_result_document_without_an_href_builds_the_principal_result() -> Result<(), XsltError> {
    let xsl = stylesheet(
        r#"<xsl:template name="xsl:initial-template">
             <xsl:result-document href="secondary.xml">
               <doc-secondary>
                 <xsl:result-document><doc-primary>primary</doc-primary></xsl:result-document>
               </doc-secondary>
             </xsl:result-document>
           </xsl:template>"#,
    );
    assert_eq!(
        run_without_source(&xsl)?,
        "<doc-primary>primary</doc-primary>"
    );
    let secondary = secondary_of(&xsl)?;
    assert_eq!(secondary.len(), 1);
    assert_eq!(secondary[0].0, "secondary.xml");
    assert!(secondary[0].1.contains("<doc-secondary/>"));
    Ok(())
}

/// error-1490c, result-document-1003/1005: content written to the principal result directly and
/// again through an `xsl:result-document` is two final result trees at one URI.
#[test]
fn writing_the_principal_result_twice_is_xtde1490() {
    let xsl = stylesheet(
        r#"<xsl:template name="xsl:initial-template">
             <out>direct</out>
             <xsl:result-document><out>redirected</out></xsl:result-document>
           </xsl:template>"#,
    );
    assert_eq!(error_code(run_without_source(&xsl)), "XTDE1490");
}

/// result-document-1004: two `xsl:result-document` instructions naming the principal
/// destination, with nothing written to it directly, are still two result trees at one URI.
#[test]
fn two_result_documents_on_the_principal_destination_are_xtde1490() {
    let xsl = stylesheet(
        r#"<xsl:template name="xsl:initial-template">
             <xsl:result-document><one/></xsl:result-document>
             <xsl:result-document href=""><two/></xsl:result-document>
           </xsl:template>"#,
    );
    assert_eq!(error_code(run_without_source(&xsl)), "XTDE1490");
}

/// Two secondary result documents at one URI remain `XTDE1490` as well.
#[test]
fn two_secondary_result_documents_at_one_uri_are_xtde1490() {
    let xsl = stylesheet(
        r#"<xsl:template name="xsl:initial-template">
             <xsl:result-document href="a.xml"><one/></xsl:result-document>
             <xsl:result-document href="a.xml"><two/></xsl:result-document>
           </xsl:template>"#,
    );
    assert_eq!(error_code(run_without_source(&xsl)), "XTDE1490");
}

// ---------------------------------------------------------------------------------------------
// Dynamic errors the engine did not raise
// ---------------------------------------------------------------------------------------------

/// error-0030a: `terminate` written as an attribute value template must still evaluate to one
/// of the values the attribute permits.
#[test]
fn a_terminate_attribute_value_template_must_yield_yes_or_no() {
    let xsl = stylesheet(
        r#"<xsl:param name="x" select="'bananas'"/>
           <xsl:template name="xsl:initial-template">
             <out><xsl:message terminate="{$x}"/></out>
           </xsl:template>"#,
    );
    assert_eq!(error_code(run_without_source(&xsl)), "XTDE0030");
}

/// error-0044a/0044aa/0044ac: naming an initial mode with nothing to apply the rules to.
#[test]
fn an_initial_mode_without_an_initial_match_selection_is_xtde0044() -> Result<(), XsltError> {
    let xsl = stylesheet(r#"<xsl:template match="/" mode="m"><out/></xsl:template>"#);
    let compiled = Stylesheet::compile_str(&xsl)?;
    let outcome = compiled.transform_optional(None, &Parameters::new().with_initial_mode("m"));
    assert_eq!(
        outcome
            .err()
            .and_then(|error| error.code().map(str::to_owned)),
        Some("XTDE0044".to_owned())
    );
    Ok(())
}

/// The same invocation with a source document supplies the initial match selection, so it runs.
#[test]
fn an_initial_mode_with_a_source_document_is_no_error() -> Result<(), XsltError> {
    let xsl = stylesheet(r#"<xsl:template match="/" mode="m"><out/></xsl:template>"#);
    let compiled = Stylesheet::compile_str(&xsl)?;
    let tree = Tree::parse_str("<doc/>")
        .map_err(|error| XsltError::dynamic("XTDE0000", error.message().to_owned()))?;
    let result =
        compiled.transform_with(&tree.root(), &Parameters::new().with_initial_mode("m"))?;
    assert_eq!(result.as_str(), "<out/>");
    Ok(())
}

/// error-0890b: `xml` is an `NCName` but not a `PITarget`.
#[test]
fn a_processing_instruction_named_xml_is_xtde0890() {
    let xsl = stylesheet(
        r#"<xsl:template name="xsl:initial-template">
             <out><xsl:processing-instruction name="xml">version='1.0'</xsl:processing-instruction></out>
           </xsl:template>"#,
    );
    assert_eq!(error_code(run_without_source(&xsl)), "XTDE0890");
}

/// error-1071b, for-each-group-015b: positional grouping leaves no current grouping key.
#[test]
fn current_grouping_key_under_positional_grouping_is_xtde1071() {
    let xsl = stylesheet(
        r#"<xsl:template name="xsl:initial-template">
             <out>
               <xsl:for-each-group select="1 to 50" group-starting-with=".[. mod 10 = 0]">
                 <xsl:value-of select="current-grouping-key()"/>
               </xsl:for-each-group>
             </out>
           </xsl:template>"#,
    );
    assert_eq!(error_code(run_without_source(&xsl)), "XTDE1071");
}

/// The value-based groupings still report their key.
#[test]
fn current_grouping_key_under_group_by_is_the_key() -> Result<(), XsltError> {
    let xsl = stylesheet(
        r#"<xsl:template name="xsl:initial-template">
             <out>
               <xsl:for-each-group select="1 to 4" group-by=". mod 2">
                 <g><xsl:value-of select="current-grouping-key()"/></g>
               </xsl:for-each-group>
             </out>
           </xsl:template>"#,
    );
    assert_eq!(run_without_source(&xsl)?, "<out><g>1</g><g>0</g></out>");
    Ok(())
}

/// error-1428a: `type-available` needs an `EQName`.
#[test]
fn type_available_of_a_malformed_name_is_xtde1428() {
    let xsl = stylesheet(
        r#"<xsl:param name="t" select="'foo::bar'"/>
           <xsl:template name="xsl:initial-template">
             <out><xsl:value-of select="type-available($t)"/></out>
           </xsl:template>"#,
    );
    assert_eq!(error_code(run_without_source(&xsl)), "XTDE1428");
}

// ---------------------------------------------------------------------------------------------
// Grouping and key comparison
// ---------------------------------------------------------------------------------------------

/// for-each-group-061/062/064/065: grouping keys are compared by value, so two `xs:dateTime`
/// values written in different time zones are one group.
#[test]
fn grouping_keys_compare_by_value_not_by_lexical_form() -> Result<(), XsltError> {
    let xsl = stylesheet(
        r#"<xsl:template match="/">
             <out>
               <xsl:for-each-group select="//c" group-by="xs:dateTime(@at)">
                 <g n="{count(current-group())}"/>
               </xsl:for-each-group>
             </out>
           </xsl:template>"#,
    );
    let source = r#"<doc><c at="2002-10-05T22:00:00-05:00"/><c at="2002-10-06T03:00:00Z"/></doc>"#;
    assert_eq!(run_source(&xsl, source)?, r#"<out><g n="2"/></out>"#);
    Ok(())
}

/// for-each-group-082: `node-name()` keys compare by expanded name, not by prefix.
#[test]
fn grouping_by_node_name_ignores_the_prefix() -> Result<(), XsltError> {
    let xsl = stylesheet(
        r#"<xsl:template match="/">
             <out>
               <xsl:for-each-group select="/doc/*" group-by="node-name(.)">
                 <g n="{count(current-group())}"/>
               </xsl:for-each-group>
             </out>
           </xsl:template>"#,
    );
    let source = r#"<doc xmlns:a="urn:x" xmlns:b="urn:x"><a:e/><b:e/><e/></doc>"#;
    assert_eq!(
        run_source(&xsl, source)?,
        r#"<out><g n="2"/><g n="1"/></out>"#
    );
    Ok(())
}

/// key-069: a key value is matched by value, so an `xs:dateTime` written in another time zone
/// still finds the node.
#[test]
fn key_values_match_by_value() -> Result<(), XsltError> {
    let xsl = stylesheet(
        r#"<xsl:key name="i" match="item" use="xs:dateTime(.)"/>
           <xsl:template match="/">
             <out><xsl:value-of select="key('i', xs:dateTime('2002-10-05T22:00:00-05:00'))/@id"/></out>
           </xsl:template>"#,
    );
    let source = r#"<doc><item id="a">2002-10-06T03:00:00Z</item></doc>"#;
    assert_eq!(run_source(&xsl, source)?, "<out>a</out>");
    Ok(())
}

/// key-081: an integer key and a string key are different keys, since `4 eq "4"` is a type
/// error rather than a match.
#[test]
fn an_integer_key_is_not_a_string_key() -> Result<(), XsltError> {
    let xsl = stylesheet(
        r#"<xsl:key name="k" match="b" use="string-length(.)"/>
           <xsl:template match="/">
             <out><one><xsl:copy-of select="key('k', 4)"/></one><two><xsl:copy-of select="key('k', '4')"/></two></out>
           </xsl:template>"#,
    );
    let source = "<a><b>John</b><b>Ann</b></a>";
    assert_eq!(
        run_source(&xsl, source)?,
        "<out><one><b>John</b></one><two/></out>"
    );
    Ok(())
}

/// key-070: `NaN` is equal to no key value, itself included.
#[test]
fn a_not_a_number_key_matches_nothing() -> Result<(), XsltError> {
    let xsl = stylesheet(
        r#"<xsl:key name="i" match="item" use="xs:double(.)"/>
           <xsl:template match="/">
             <out><xsl:value-of select="key('i', xs:double('NaN'))/@id"/></out>
           </xsl:template>"#,
    );
    let source = r#"<doc><item id="d">NaN</item></doc>"#;
    assert_eq!(run_source(&xsl, source)?, "<out/>");
    Ok(())
}

/// key-073/074: an `xsl:key` with a sequence constructor files the node under *each* value the
/// constructor produces, not under one string holding all of them.
#[test]
fn a_key_with_a_sequence_constructor_files_every_value() -> Result<(), XsltError> {
    let xsl = stylesheet(
        r#"<xsl:key name="k" match="item"><xsl:sequence select="string-to-codepoints(.)"/></xsl:key>
           <xsl:template match="/">
             <out><xsl:value-of select="key('k', 98)/@id"/></out>
           </xsl:template>"#,
    );
    let source = r#"<doc><item id="x">ab</item></doc>"#;
    assert_eq!(run_source(&xsl, source)?, "<out>x</out>");
    Ok(())
}

// ---------------------------------------------------------------------------------------------
// Conditional content, attribute sets and rule chaining
// ---------------------------------------------------------------------------------------------

/// on-empty-012, on-non-empty-012/014: a document node with no children writes nothing, so the
/// containing sequence constructor is empty.
#[test]
fn an_empty_temporary_tree_leaves_the_constructor_empty() -> Result<(), XsltError> {
    let xsl = stylesheet(
        r#"<xsl:param name="z"><xsl:copy-of select="/unicorn"/></xsl:param>
           <xsl:template match="/">
             <out>
               <xsl:copy-of select="$z"/>
               <xsl:on-non-empty>NON-EMPTY</xsl:on-non-empty>
               <xsl:on-empty>EMPTY</xsl:on-empty>
             </out>
           </xsl:template>"#,
    );
    assert_eq!(run_source(&xsl, "<doc/>")?, "<out>EMPTY</out>");
    Ok(())
}

/// The negative control: real content still counts as non-empty.
#[test]
fn written_content_leaves_the_constructor_non_empty() -> Result<(), XsltError> {
    let xsl = stylesheet(
        r#"<xsl:template match="/">
             <out>
               <xsl:text>x</xsl:text>
               <xsl:on-non-empty>NON-EMPTY</xsl:on-non-empty>
               <xsl:on-empty>EMPTY</xsl:on-empty>
             </out>
           </xsl:template>"#,
    );
    assert_eq!(run_source(&xsl, "<doc/>")?, "<out>xNON-EMPTY</out>");
    Ok(())
}

/// attribute-set-1802: only global variables are in scope inside an `xsl:attribute-set`.
#[test]
fn an_attribute_set_sees_only_global_variables() -> Result<(), XsltError> {
    let xsl = stylesheet(
        r#"<xsl:variable name="foo" select="'correct'"/>
           <xsl:template match="/">
             <xsl:variable name="foo" select="'incorrect'"/>
             <out xsl:use-attribute-sets="attrs"/>
           </xsl:template>
           <xsl:attribute-set name="attrs">
             <xsl:attribute name="test"><xsl:value-of select="$foo"/></xsl:attribute>
           </xsl:attribute-set>"#,
    );
    assert_eq!(run_source(&xsl, "<doc/>")?, r#"<out test="correct"/>"#);
    Ok(())
}

/// next-match-024: with a `priority` the union pattern stays one template rule, so
/// `xsl:next-match` moves on to another template.
#[test]
fn next_match_leaves_a_template_that_declares_a_priority() -> Result<(), XsltError> {
    let xsl = stylesheet(
        r#"<xsl:template match="doc"><out><xsl:apply-templates select="x"/></out></xsl:template>
           <xsl:template match="*:x|element(x)" priority="1"><foo/><xsl:next-match/></xsl:template>
           <xsl:template match="node()"><bar/></xsl:template>"#,
    );
    assert_eq!(
        run_source(&xsl, "<doc><x/></doc>")?,
        "<out><foo/><bar/></out>"
    );
    Ok(())
}

/// next-match-023: without a `priority` the union pattern is several rules, each with its own
/// default priority, so `xsl:next-match` reaches the second alternative.
#[test]
fn next_match_visits_every_alternative_of_a_default_priority_union() -> Result<(), XsltError> {
    let xsl = stylesheet(
        r#"<xsl:template match="doc"><out><xsl:apply-templates select="x"/></out></xsl:template>
           <xsl:template match="*:x|element(x)"><foo/><xsl:next-match/></xsl:template>
           <xsl:template match="node()"><bar/></xsl:template>"#,
    );
    assert_eq!(
        run_source(&xsl, "<doc><x/></doc>")?,
        "<out><foo/><foo/><bar/></out>"
    );
    Ok(())
}

// ---------------------------------------------------------------------------------------------
// fn:regex-group scoping and the availability functions
// ---------------------------------------------------------------------------------------------

/// analyze-string-034/077: a stylesheet function body sees no captured substrings, whatever
/// `xsl:analyze-string` is in force around the call.
#[test]
fn a_stylesheet_function_sees_no_captured_substrings() -> Result<(), XsltError> {
    let xsl = format!(
        r#"<xsl:stylesheet version="3.0" xmlns:xsl="http://www.w3.org/1999/XSL/Transform"
             xmlns:f="urn:f" exclude-result-prefixes="f">
             <xsl:output omit-xml-declaration="yes"/>
             <xsl:function name="f:group"><xsl:param name="n"/><xsl:sequence select="regex-group($n)"/></xsl:function>
             {body}
           </xsl:stylesheet>"#,
        body = r#"<xsl:template match="/">
             <out>
               <xsl:analyze-string select="'ab'" regex="(a)">
                 <xsl:matching-substring>[<xsl:value-of select="regex-group(1)"/>|<xsl:value-of select="f:group(1)"/>]</xsl:matching-substring>
                 <xsl:non-matching-substring><xsl:value-of select="."/></xsl:non-matching-substring>
               </xsl:analyze-string>
             </out>
           </xsl:template>"#
    );
    assert_eq!(run_source(&xsl, "<doc/>")?, "<out>[a|]b</out>");
    Ok(())
}

/// analyze-string-076: a pattern sees no captured substrings either.
#[test]
fn a_pattern_sees_no_captured_substrings() -> Result<(), XsltError> {
    let xsl = stylesheet(
        r#"<xsl:template match="/">
             <xsl:variable name="items" select="/doc/item"/>
             <out>
               <xsl:analyze-string select="'a'" regex="(a)">
                 <xsl:matching-substring>
                   <xsl:for-each-group select="$items" group-starting-with="item[@attr=string(regex-group(1))]">
                     <g n="{count(current-group())}"/>
                   </xsl:for-each-group>
                 </xsl:matching-substring>
               </xsl:analyze-string>
             </out>
           </xsl:template>"#,
    );
    // With the captured substrings empty the pattern matches only `@attr=""`, which starts one
    // group at the second item; had it seen `a` it would have started one at the first.
    let source = r#"<doc><item attr="a"/><item attr=""/></doc>"#;
    assert_eq!(
        run_source(&xsl, source)?,
        r#"<out><g n="1"/><g n="1"/></out>"#
    );
    Ok(())
}

/// function-available-1011: an `EQName` argument is resolved against the namespace it writes
/// out, and a schema type names its constructor function.
#[test]
fn function_available_understands_eqnames() -> Result<(), XsltError> {
    let xsl = stylesheet(
        r#"<xsl:template match="/">
             <out>
               <a><xsl:value-of select="function-available('Q{http://www.w3.org/2001/XMLSchema}integer', 1)"/></a>
               <b><xsl:value-of select="function-available('Q{http://www.w3.org/2005/xpath-functions}abs', 1)"/></b>
               <c><xsl:value-of select="function-available('Q{}abs', 1)"/></c>
             </out>
           </xsl:template>"#,
    );
    assert_eq!(
        run_source(&xsl, "<doc/>")?,
        "<out><a>true</a><b>true</b><c>false</c></out>"
    );
    Ok(())
}

/// function-available-1008: `fn:concat` is variadic, so it is available at every arity of two
/// upwards and at none below.
#[test]
fn function_available_knows_concat_is_variadic() -> Result<(), XsltError> {
    let xsl = stylesheet(
        r#"<xsl:template match="/">
             <out g="{function-available('concat', 17)}" m="{function-available('concat', 0)}"/>
           </xsl:template>"#,
    );
    assert_eq!(run_source(&xsl, "<doc/>")?, r#"<out g="true" m="false"/>"#);
    Ok(())
}

/// type-available-0150: the same `EQName` handling, against the schema namespace.
#[test]
fn type_available_understands_eqnames() -> Result<(), XsltError> {
    let xsl = stylesheet(
        r#"<xsl:param name="p" select="'Q{http://www.w3.org/2001/XMLSchema}'"/>
           <xsl:template match="/">
             <out>
               <a><xsl:value-of select="type-available($p||'anyType')"/></a>
               <b><xsl:value-of select="type-available($p||'integer')"/></b>
               <c><xsl:value-of select="type-available($p||'heh')"/></c>
               <d><xsl:value-of select="type-available('Q{}int')"/></d>
             </out>
           </xsl:template>"#,
    );
    assert_eq!(
        run_source(&xsl, "<doc/>")?,
        "<out><a>true</a><b>true</b><c>false</c><d>false</d></out>"
    );
    Ok(())
}
