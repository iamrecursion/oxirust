//! Regression tests for the default entry point, text value templates and collation selection —
//! three areas the W3C `xslt30-test` suite covers heavily and this processor handled wrongly.
//!
//! Every test reproduces one suite case with a minimal inline stylesheet, so each fix is
//! provable red-then-green without the separate `oxixml-testsuite` harness. The case
//! identifiers in the comments are the ones the harness reports.

use oxixml_infoset::Tree;
use oxixml_xslt::{MemoryResolver, Parameters, StaticOptions, Stylesheet, XsltError};
use std::sync::Arc;

/// The case-blind collation the suite's `collations` set selects everywhere.
const HTML_ASCII_CASE_INSENSITIVE: &str =
    "http://www.w3.org/2005/xpath-functions/collation/html-ascii-case-insensitive";

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

/// Runs with no source document at all, which is how the suite invokes every case whose `<test>`
/// names no environment.
fn run_without_source(xsl: &str) -> Result<String, XsltError> {
    let compiled = Stylesheet::compile_str(xsl)?;
    Ok(compiled
        .transform_optional(None, &Parameters::new())?
        .as_str()
        .to_owned())
}

// === The default entry point (XSLT 3.0 §2.3.4) ================================================

/// analyze-string-099, iterate-040, on-empty-115a: `xsl:initial-template` is the default entry
/// point, and a case that supplies no source document still expects it to run even though the
/// same template also carries a `match` pattern — with nothing to match against, the
/// apply-templates reading of §2.3.3 has no work to do.
#[test]
fn an_initial_template_that_is_also_a_rule_runs_without_a_source() {
    let xsl = stylesheet(
        r#"<xsl:template match="/" name="xsl:initial-template"><out>entered</out></xsl:template>"#,
    );
    assert_eq!(
        run_without_source(&xsl).expect("the default entry point runs"),
        "<out>entered</out>"
    );
}

/// The same stylesheet given a source document keeps the apply-templates reading, so the rule
/// becomes the *current* template rule and an `xsl:next-match` inside it resolves.
#[test]
fn a_source_document_still_reaches_the_initial_template_by_matching() {
    let xsl = stylesheet(
        r#"<xsl:template match="/" name="xsl:initial-template"><out><xsl:next-match/></out></xsl:template>
           <xsl:template match="/" priority="-1">fallback</xsl:template>"#,
    );
    assert_eq!(
        run_source(&xsl, "<doc/>").expect("the rule is current"),
        "<out>fallback</out>"
    );
}

// === Text value templates (XSLT 3.0 §5.5.1) ===================================================

fn expanding_stylesheet(body: &str) -> String {
    format!(
        r#"<xsl:stylesheet version="3.0" xmlns:xsl="http://www.w3.org/1999/XSL/Transform"
             expand-text="yes">
             <xsl:output omit-xml-declaration="yes"/>
             {body}
           </xsl:stylesheet>"#
    )
}

/// match-241: a `{...}` text node written as a *direct* child of `xsl:template` is a text value
/// template like any other. It reaches the compiler by a different path from the one inside a
/// literal result element, and that path used to emit the braces verbatim.
#[test]
fn a_text_value_template_directly_under_a_template_is_expanded() {
    let xsl = expanding_stylesheet(r#"<xsl:template match="/">[{1 + 1}]</xsl:template>"#);
    assert_eq!(run_source(&xsl, "<doc/>").expect("the text expands"), "[2]");
}

/// The same for an `xsl:function` body, which shares the compiler path with `xsl:template`.
#[test]
fn a_text_value_template_directly_under_a_function_is_expanded() {
    let xsl = r#"<xsl:stylesheet version="3.0" xmlns:xsl="http://www.w3.org/1999/XSL/Transform"
             xmlns:f="urn:f" exclude-result-prefixes="f" expand-text="yes">
             <xsl:output omit-xml-declaration="yes"/>
             <xsl:function name="f:label"><xsl:param name="n"/>n={$n}</xsl:function>
             <xsl:template match="/"><out><xsl:value-of select="f:label(7)"/></out></xsl:template>
           </xsl:stylesheet>"#;
    assert_eq!(
        run_source(xsl, "<doc/>").expect("the function body expands"),
        "<out>n=7</out>"
    );
}

/// `xsl:try` collects its content on its own path too, and honours `expand-text` the same way.
#[test]
fn a_text_value_template_directly_under_try_is_expanded() {
    let xsl = expanding_stylesheet(
        r#"<xsl:template match="/"><out><xsl:try>{2 * 3}<xsl:catch>caught</xsl:catch></xsl:try></out></xsl:template>"#,
    );
    assert_eq!(
        run_source(&xsl, "<doc/>").expect("the try content expands"),
        "<out>6</out>"
    );
}

// === Predicate patterns on parentless nodes (XSLT 3.0 §5.5.3) =================================

/// match-249: a variable declared `as="element(a)"` holds a *parentless* element, and a template
/// rule whose pattern carries a predicate has to match it. The candidate sequence a predicate is
/// evaluated over is the node itself when there is no parent to draw siblings from.
#[test]
fn a_predicate_pattern_matches_a_parentless_constructed_element() {
    let xsl = stylesheet(
        r#"<xsl:template match="a[@code='x']">hit</xsl:template>
           <xsl:template match="a">miss</xsl:template>
           <xsl:template match="/"><out>
             <xsl:variable name="e" as="element(a)"><a code="x"/></xsl:variable>
             <xsl:apply-templates select="$e"/>
           </out></xsl:template>"#,
    );
    assert_eq!(
        run_source(&xsl, "<doc/>").expect("the rule matches"),
        "<out>hit</out>"
    );
}

/// The predicate still decides: a value it rejects falls through to the next rule.
#[test]
fn a_predicate_pattern_still_rejects_a_parentless_node_that_fails_it() {
    let xsl = stylesheet(
        r#"<xsl:template match="a[@code='y']">hit</xsl:template>
           <xsl:template match="a">miss</xsl:template>
           <xsl:template match="/"><out>
             <xsl:variable name="e" as="element(a)"><a code="x"/></xsl:variable>
             <xsl:apply-templates select="$e"/>
           </out></xsl:template>"#,
    );
    assert_eq!(
        run_source(&xsl, "<doc/>").expect("the rule does not match"),
        "<out>miss</out>"
    );
}

/// A positional predicate sees a one-item sequence, so `[1]` holds and `[2]` does not.
#[test]
fn a_positional_predicate_on_a_parentless_node_sees_a_single_item() {
    let xsl = stylesheet(
        r#"<xsl:template match="a[1]">first</xsl:template>
           <xsl:template match="a">other</xsl:template>
           <xsl:template match="/"><out>
             <xsl:variable name="e" as="element(a)"><a/></xsl:variable>
             <xsl:apply-templates select="$e"/>
           </out></xsl:template>"#,
    );
    assert_eq!(
        run_source(&xsl, "<doc/>").expect("the rule matches"),
        "<out>first</out>"
    );
}

/// match-114, match-116: an abbreviated `attribute()` step takes the *attribute* axis, not the
/// child axis, so a template rule written `match="attribute()"` matches attribute nodes. On the
/// child axis it could never match anything at all.
#[test]
fn an_abbreviated_attribute_kind_test_takes_the_attribute_axis() {
    let xsl = stylesheet(
        r#"<xsl:template match="attribute()" priority="1">seen</xsl:template>
           <xsl:template match="/"><out><xsl:apply-templates select="doc/@*"/></out></xsl:template>"#,
    );
    assert_eq!(
        run_source(&xsl, "<doc a='v'/>").expect("the rule matches"),
        "<out>seen</out>"
    );
}

/// The same rule reaches a parentless attribute, which is what `as="attribute()"` binds.
#[test]
fn an_attribute_kind_test_matches_a_parentless_attribute() {
    let xsl = stylesheet(
        r#"<xsl:template match="attribute()" priority="1">seen</xsl:template>
           <xsl:template match="doc"><out>
             <xsl:variable name="a" as="attribute()"><xsl:copy-of select="attribute()"/></xsl:variable>
             <xsl:apply-templates select="$a"/>
           </out></xsl:template>"#,
    );
    assert_eq!(
        run_source(&xsl, "<doc a='v'/>").expect("the rule matches"),
        "<out>seen</out>"
    );
}

// === Collation selection (XSLT 3.0 §5.4.4) ====================================================

/// collations-0109: `default-collation` on `xsl:template` settles the collation `compare()` uses
/// inside it. The default collation is a *static* property, so the one in force where the
/// expression was written wins over the stylesheet-wide one.
#[test]
fn a_default_collation_on_a_template_reaches_compare() {
    let xsl = stylesheet(&format!(
        r#"<xsl:template match="/" default-collation="{HTML_ASCII_CASE_INSENSITIVE}">
             <out><xsl:value-of select="compare('Thou', 'THOU')"/></out>
           </xsl:template>"#
    ));
    assert_eq!(
        run_source(&xsl, "<doc/>").expect("the collation applies"),
        "<out>0</out>"
    );
}

/// collations-0111: the attribute may sit on the instruction itself.
#[test]
fn a_default_collation_on_an_instruction_reaches_its_expression() {
    let xsl = stylesheet(&format!(
        r#"<xsl:template match="/">
             <out>
               <a><xsl:value-of select="starts-with('THOUGH', 'th')"
                                default-collation="{HTML_ASCII_CASE_INSENSITIVE}"/></a>
               <b><xsl:value-of select="starts-with('THOUGH', 'th')"/></b>
             </out>
           </xsl:template>"#
    ));
    assert_eq!(
        run_source(&xsl, "<doc/>").expect("the collation applies"),
        "<out><a>true</a><b>false</b></out>"
    );
}

/// collations-0121: on a literal result element the attribute is written `xsl:default-collation`,
/// and it governs the expressions of the element's descendants.
#[test]
fn an_xsl_default_collation_on_a_literal_result_element_reaches_its_descendants() {
    let xsl = stylesheet(&format!(
        r#"<xsl:template match="/">
             <out xsl:default-collation="{HTML_ASCII_CASE_INSENSITIVE}">
               <xsl:value-of select="'Adele' eq 'ADELE'"/>
             </out>
           </xsl:template>"#
    ));
    assert_eq!(
        run_source(&xsl, "<doc/>").expect("the collation applies"),
        "<out>true</out>"
    );
}

/// collations-0101: `xsl:sort` with neither `collation` nor `lang` sorts under the default
/// collation in force where it stands, so a case-blind default keeps `THOROUGH` and
/// `tHoRoUgH` adjacent in their original order instead of separating them by case.
#[test]
fn a_sort_without_a_collation_uses_the_default_collation_in_scope() {
    let xsl = stylesheet(&format!(
        r#"<xsl:template match="/" default-collation="{HTML_ASCII_CASE_INSENSITIVE}">
             <out><xsl:for-each select="w/x"><xsl:sort select="."/>
               <w><xsl:value-of select="."/></w>
             </xsl:for-each></out>
           </xsl:template>"#
    ));
    assert_eq!(
        run_source(&xsl, "<w><x>tHoRoUgH</x><x>THOROUGH</x><x>THE</x></w>")
            .expect("the collation applies"),
        "<out><w>THE</w><w>tHoRoUgH</w><w>THOROUGH</w></out>"
    );
}

/// collations-0127: an imported module's own `xsl:stylesheet` attributes govern its
/// declarations. The module starts from a fresh environment, so `default-collation` written
/// there has to be read off the module element itself.
#[test]
fn an_imported_modules_default_collation_governs_its_own_templates() {
    let imported = format!(
        r#"<xsl:stylesheet version="3.0" xmlns:xsl="http://www.w3.org/1999/XSL/Transform"
             default-collation="{HTML_ASCII_CASE_INSENSITIVE}">
             <xsl:template name="one"><one><xsl:value-of select="'Adele' = 'ADELE'"/></one></xsl:template>
           </xsl:stylesheet>"#
    );
    let resolver = MemoryResolver::new().with_document("urn:imported", &imported);
    let principal = stylesheet(
        r#"<xsl:import href="urn:imported"/>
           <xsl:template match="/"><out><xsl:call-template name="one"/></out></xsl:template>"#,
    );
    let options = StaticOptions::new()
        .with_resolver(Arc::new(resolver))
        .with_base_uri("urn:principal");
    let compiled = Stylesheet::compile_str_with(&principal, options).expect("the modules compile");
    let tree = Tree::parse_str("<doc/>").expect("the source parses");
    assert_eq!(
        compiled
            .transform_to_string(&tree.root())
            .expect("the collation applies"),
        "<out><one>true</one></out>"
    );
}

/// collations-0102: with no `case-order` written, nothing may break a tie the collation calls
/// equal — the sort is stable, so two spellings of one word keep the order the population had.
#[test]
fn without_case_order_a_collation_tie_keeps_the_population_order() {
    let xsl = stylesheet(&format!(
        r#"<xsl:template match="/">
             <out><xsl:for-each select="w/x">
               <xsl:sort select="." collation="{HTML_ASCII_CASE_INSENSITIVE}"/>
               <w><xsl:value-of select="."/></w>
             </xsl:for-each></out>
           </xsl:template>"#
    ));
    assert_eq!(
        run_source(&xsl, "<w><x>tHoRoUgH</x><x>THOROUGH</x></w>").expect("the sort is stable"),
        "<out><w>tHoRoUgH</w><w>THOROUGH</w></out>"
    );
}

/// sort-067: writing `case-order` as well changes nothing, because a named `collation` settles
/// the ordering on its own and the attribute is ignored beside it. The suite case says so in as
/// many words — "Test that case-order is ignored when collation is present" — and pins it with
/// the code point collation, whose `With-collation` result is plain code point order however
/// `case-order` reads.
#[test]
fn an_explicit_collation_makes_case_order_inert() {
    let sorted = |case_order: &str| {
        let xsl = stylesheet(&format!(
            r#"<xsl:template match="/">
                 <out><xsl:for-each select="w/x">
                   <xsl:sort select="." collation="{HTML_ASCII_CASE_INSENSITIVE}"{case_order}/>
                   <w><xsl:value-of select="."/></w>
                 </xsl:for-each></out>
               </xsl:template>"#
        ));
        run_source(&xsl, "<w><x>tHoRoUgH</x><x>THOROUGH</x></w>").expect("the sort runs")
    };
    let stable = "<out><w>tHoRoUgH</w><w>THOROUGH</w></out>";
    assert_eq!(sorted(r#" case-order="upper-first""#), stable);
    assert_eq!(sorted(r#" case-order="lower-first""#), stable);
}

/// collations-0104: `xsl:for-each-group` with no `collation` groups under the default collation
/// in force, so words differing only in case fall into one group.
#[test]
fn a_grouping_without_a_collation_uses_the_default_collation_in_scope() {
    let xsl = stylesheet(&format!(
        r#"<xsl:template match="/" default-collation="{HTML_ASCII_CASE_INSENSITIVE}">
             <out><xsl:for-each-group select="w/x" group-by="string(.)">
               <g><xsl:value-of select="count(current-group())"/></g>
             </xsl:for-each-group></out>
           </xsl:template>"#
    ));
    assert_eq!(
        run_source(&xsl, "<w><x>thou</x><x>THOU</x><x>the</x></w>").expect("the collation applies"),
        "<out><g>2</g><g>1</g></out>"
    );
}
