//! Regression tests for static variables and parameters, shadow attributes, `xsl:on-completion`,
//! stylesheet-function memoization, XML whitespace in stylesheet text and the pattern grammar.
//!
//! Every test reproduces one W3C `xslt30-test` case with a minimal inline stylesheet, so each fix
//! is provable red-then-green without the separate `oxixml-testsuite` harness. The case
//! identifiers in the comments are the ones that harness reports.

use oxixml_infoset::{AtomicValue, Tree};
use oxixml_xslt::{MemoryResolver, Parameters, StaticOptions, Stylesheet, XsltError};
use std::sync::Arc;

fn stylesheet(body: &str) -> String {
    format!(
        r#"<xsl:stylesheet version="3.0" xmlns:xsl="http://www.w3.org/1999/XSL/Transform"
             xmlns:xs="http://www.w3.org/2001/XMLSchema" xmlns:x="http://xxx.com/"
             exclude-result-prefixes="xs x" expand-text="yes">
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

/// Compiles under `options` and runs with no source document.
fn run_with_options(xsl: &str, options: StaticOptions) -> Result<String, XsltError> {
    let compiled = Stylesheet::compile_str_with(xsl, options)?;
    Ok(compiled
        .transform_optional(None, &Parameters::new())?
        .as_str()
        .to_owned())
}

/// The transformation's text, or a readable stand-in for the error it raised — `XsltError` is
/// not comparable, and a failed run should say what went wrong rather than only that it failed.
fn text(outcome: Result<String, XsltError>) -> String {
    match outcome {
        Ok(text) => text,
        Err(error) => format!("error {}", error.code().unwrap_or("no code")),
    }
}

fn error_code(outcome: Result<String, XsltError>) -> String {
    match outcome {
        Ok(text) => format!("no error, produced `{text}`"),
        Err(error) => error.code().unwrap_or("no code").to_owned(),
    }
}

// ---------------------------------------------------------------------------------------------
// Static parameters and shadow attributes (XSLT 3.0 §3.14.2, §9.6, §9.7)
// ---------------------------------------------------------------------------------------------

/// date-094g and its family: a `required="yes"` static parameter supplies the *expression* an
/// `xsl:value-of` evaluates, through the shadow attribute `_select`.
#[test]
fn a_static_parameter_supplies_a_shadow_attribute() {
    let xsl = stylesheet(
        r#"<xsl:param name="test-case" required="yes" static="yes"/>
           <xsl:template name="xsl:initial-template">
             <out><xsl:value-of _select="{$test-case}"/></out>
           </xsl:template>"#,
    );
    let options = StaticOptions::new().with_static_parameter(
        "test-case",
        AtomicValue::string("string-join(('a', 'b'), '-')"),
    );
    assert_eq!(text(run_with_options(&xsl, options)), "<out>a-b</out>");
}

/// The same shadow attribute, given an expression the constructor rejects, must let the
/// constructor's own error out — which is the whole point of date-094e..q.
#[test]
fn a_shadow_attribute_carries_the_expressions_own_error() {
    let xsl = stylesheet(
        r#"<xsl:param name="test-case" required="yes" static="yes"/>
           <xsl:template name="xsl:initial-template">
             <out><xsl:value-of _select="{$test-case}"/></out>
           </xsl:template>"#,
    );
    let options = StaticOptions::new()
        .with_static_parameter("test-case", AtomicValue::string("xs:date('1999-01-32')"));
    assert_eq!(error_code(run_with_options(&xsl, options)), "FORG0001");
}

/// A static parameter with no supplied value falls back on its own `select`.
#[test]
fn a_static_parameter_defaults_to_its_own_select() {
    let xsl = stylesheet(
        r#"<xsl:param name="pick" static="yes" select="'2 + 3'"/>
           <xsl:template name="xsl:initial-template">
             <out><xsl:value-of _select="{$pick}"/></out>
           </xsl:template>"#,
    );
    assert_eq!(text(run_without_source(&xsl)), "<out>5</out>");
}

/// A `required="yes"` static parameter nothing supplies is `XTDE0050`.
#[test]
fn an_unsupplied_required_static_parameter_is_xtde0050() {
    let xsl = stylesheet(
        r#"<xsl:param name="pick" required="yes" static="yes"/>
           <xsl:template name="xsl:initial-template"><out/></xsl:template>"#,
    );
    assert_eq!(error_code(run_without_source(&xsl)), "XTDE0050");
}

/// error-0640o-2: §9.7 puts only the static variables declared *earlier* into a static
/// expression's scope, so `select="$other"` on the first of two static parameters is `XPST0008`
/// — and stays so even when the parameter's value is supplied from outside, because the
/// constraint is on the expression rather than on the value.
#[test]
fn a_forward_reference_between_static_parameters_is_xpst0008() {
    let xsl = stylesheet(
        r#"<xsl:param name="circular" static="yes" select="$other"/>
           <xsl:param name="other" static="yes" select="$circular"/>
           <xsl:template name="xsl:initial-template" use-when="$other"><out/></xsl:template>"#,
    );
    assert_eq!(error_code(run_without_source(&xsl)), "XPST0008");
    let options =
        StaticOptions::new().with_static_parameter("circular", AtomicValue::boolean(true));
    assert_eq!(error_code(run_with_options(&xsl, options)), "XPST0008");
}

/// A backward reference between static parameters is exactly what the mechanism is for.
#[test]
fn a_backward_reference_between_static_parameters_is_accepted() {
    let xsl = stylesheet(
        r#"<xsl:param name="first" static="yes" select="'3'"/>
           <xsl:param name="second" static="yes" select="concat($first, ' + 4')"/>
           <xsl:template name="xsl:initial-template">
             <out><xsl:value-of _select="{$second}"/></out>
           </xsl:template>"#,
    );
    assert_eq!(text(run_without_source(&xsl)), "<out>7</out>");
}

/// error-3450a: an imported module declares a static variable one way and the importing module,
/// later in stylesheet tree order and of higher import precedence, declares it another.
#[test]
fn inconsistent_static_variables_across_import_precedence_are_xtse3450() {
    let imported = r#"<xsl:stylesheet version="3.0"
          xmlns:xsl="http://www.w3.org/1999/XSL/Transform">
          <xsl:variable name="v" static="yes" select="3"/>
        </xsl:stylesheet>"#;
    let xsl = stylesheet(
        r#"<xsl:import href="urn:imported"/>
           <xsl:variable name="v" static="yes" select="2"/>
           <xsl:template name="xsl:initial-template"><out><x v="{$v}"/></out></xsl:template>"#,
    );
    let options = StaticOptions::new().with_resolver(Arc::new(
        MemoryResolver::new().with_document("urn:imported", imported),
    ));
    assert_eq!(error_code(run_with_options(&xsl, options)), "XTSE3450");
}

/// The negative control: two declarations that agree are legal, and the static variable is also
/// an ordinary global variable, readable from the attribute value template `{$v}`.
#[test]
fn consistent_static_variables_across_import_precedence_are_accepted() {
    let imported = r#"<xsl:stylesheet version="3.0"
          xmlns:xsl="http://www.w3.org/1999/XSL/Transform">
          <xsl:variable name="v" static="yes" select="2"/>
        </xsl:stylesheet>"#;
    let xsl = stylesheet(
        r#"<xsl:import href="urn:imported"/>
           <xsl:variable name="v" static="yes" select="2"/>
           <xsl:template name="xsl:initial-template"><out><x v="{$v}"/></out></xsl:template>"#,
    );
    let options = StaticOptions::new().with_resolver(Arc::new(
        MemoryResolver::new().with_document("urn:imported", imported),
    ));
    assert_eq!(
        text(run_with_options(&xsl, options)),
        r#"<out><x v="2"/></out>"#
    );
}

/// §3.14.2: the mechanism "does not apply to attributes of literal result elements". An
/// underscore-prefixed attribute there is an ordinary attribute of that name.
#[test]
fn a_shadow_attribute_on_a_literal_result_element_is_an_ordinary_attribute() {
    let xsl = stylesheet(
        r#"<xsl:param name="pick" static="yes" select="'x'"/>
           <xsl:template name="xsl:initial-template"><out _select="{$pick}"/></xsl:template>"#,
    );
    assert_eq!(text(run_without_source(&xsl)), r#"<out _select="x"/>"#);
}

/// A shadow attribute also supplies the module an `xsl:include` names.
#[test]
fn a_shadow_attribute_supplies_the_href_of_an_include() {
    let included = r#"<xsl:stylesheet version="3.0"
          xmlns:xsl="http://www.w3.org/1999/XSL/Transform">
          <xsl:template name="xsl:initial-template"><out>included</out></xsl:template>
        </xsl:stylesheet>"#;
    let xsl = stylesheet(
        r#"<xsl:param name="which" static="yes" select="'two'"/>
           <xsl:include _href="urn:module-{$which}"/>"#,
    );
    let options = StaticOptions::new().with_resolver(Arc::new(
        MemoryResolver::new().with_document("urn:module-two", included),
    ));
    assert_eq!(text(run_with_options(&xsl, options)), "<out>included</out>");
}

/// A static parameter is also a global variable, and the value it holds at run time is the one
/// the compilation settled: the shadow attributes that read it were resolved against that value,
/// so an invocation naming the same parameter again cannot make the two disagree.
#[test]
fn the_compiled_value_of_a_static_parameter_wins_at_run_time() {
    let xsl = stylesheet(
        r#"<xsl:param name="pick" static="yes" select="'compiled'"/>
           <xsl:template name="xsl:initial-template"><out>{$pick}</out></xsl:template>"#,
    );
    let compiled = Stylesheet::compile_str(&xsl).expect("the stylesheet compiles");
    let parameters = Parameters::new().with_parameter("pick", AtomicValue::string("invocation"));
    let produced = compiled
        .transform_optional(None, &parameters)
        .expect("the transformation runs");
    assert_eq!(produced.as_str(), "<out>compiled</out>");
}

// ---------------------------------------------------------------------------------------------
// xsl:on-completion (XSLT 3.0 §8.8)
// ---------------------------------------------------------------------------------------------

/// iterate-030: `xsl:on-completion` takes its value from `select` when it has one.
#[test]
fn on_completion_takes_its_value_from_select() {
    let xsl = stylesheet(
        r#"<xsl:template name="xsl:initial-template">
             <out><xsl:iterate select="1 to 3">
               <xsl:param name="count" select="0"/>
               <xsl:on-completion select="'count ' || $count"/>
               <xsl:next-iteration><xsl:with-param name="count" select="$count + 1"/>
               </xsl:next-iteration>
             </xsl:iterate></out>
           </xsl:template>"#,
    );
    assert_eq!(text(run_without_source(&xsl)), "<out>count 3</out>");
}

/// iterate-032: a variable declared in the body of the `xsl:iterate` is out of scope in
/// `xsl:on-completion`, which is evaluated once the iteration is over.
#[test]
fn on_completion_cannot_see_a_variable_declared_in_the_body() {
    let xsl = stylesheet(
        r#"<xsl:param name="p" as="element()*" select="()"/>
           <xsl:template name="xsl:initial-template">
             <out><xsl:iterate select="$p">
               <xsl:param name="count" select="0"/>
               <xsl:on-completion select="$v"/>
               <xsl:variable name="v" select="$count + number(.)"/>
               <xsl:next-iteration><xsl:with-param name="count" select="$count + 1"/>
               </xsl:next-iteration>
             </xsl:iterate></out>
           </xsl:template>"#,
    );
    assert_eq!(error_code(run_without_source(&xsl)), "XPST0008");
}

/// iterate-043: a body variable that *shadows* an `xsl:param` of the same `xsl:iterate` does not
/// take the name out of scope — `xsl:on-completion` still reads the parameter, whose final value
/// the iteration carried.
#[test]
fn on_completion_reads_the_parameter_a_body_variable_shadows() {
    let xsl = stylesheet(
        r#"<xsl:template name="xsl:initial-template">
             <section><xsl:iterate select="1 to 5">
               <xsl:param name="p1" as="xs:integer" select="4"/>
               <xsl:on-completion select="$p1"/>
               <xsl:variable name="p1" select="$p1 + 2"/>
               <xsl:next-iteration><xsl:with-param name="p1" select="$p1 + 3"/>
               </xsl:next-iteration>
             </xsl:iterate></section>
           </xsl:template>"#,
    );
    assert_eq!(text(run_without_source(&xsl)), "<section>29</section>");
}

// ---------------------------------------------------------------------------------------------
// Stylesheet-function memoization (XSLT 3.0 §10.3)
// ---------------------------------------------------------------------------------------------

/// function-1026: `new-each-time="no"` is the XSLT 3.0 spelling of the draft-era `cache="yes"`,
/// so ten calls with seven distinct arguments produce seven distinct results.
#[test]
fn new_each_time_no_memoizes_a_stylesheet_function() {
    let xsl = stylesheet(
        r#"<xsl:function name="x:test" new-each-time="no" as="xs:string">
             <xsl:param name="n" as="xs:integer"/>
             <xsl:variable name="new-node" as="element()"><e><xsl:value-of select="$n"/></e>
             </xsl:variable>
             <xsl:sequence select="generate-id($new-node)"/>
           </xsl:function>
           <xsl:template name="xsl:initial-template">
             <xsl:variable name="ids" as="xs:string*">
               <xsl:for-each select="(1,4,6,8,3,5,6,2,1,3)">
                 <xsl:sequence select="x:test(.)"/>
               </xsl:for-each>
             </xsl:variable>
             <out><xsl:value-of select="count(distinct-values($ids))"/></out>
           </xsl:template>"#,
    );
    assert_eq!(text(run_without_source(&xsl)), "<out>7</out>");
}

/// The negative control: without the attribute, every call is evaluated afresh.
#[test]
fn a_stylesheet_function_is_not_memoized_by_default() {
    let xsl = stylesheet(
        r#"<xsl:function name="x:test" as="xs:string">
             <xsl:param name="n" as="xs:integer"/>
             <xsl:variable name="new-node" as="element()"><e><xsl:value-of select="$n"/></e>
             </xsl:variable>
             <xsl:sequence select="generate-id($new-node)"/>
           </xsl:function>
           <xsl:template name="xsl:initial-template">
             <xsl:variable name="ids" as="xs:string*">
               <xsl:for-each select="(1,4,6,8,3,5,6,2,1,3)">
                 <xsl:sequence select="x:test(.)"/>
               </xsl:for-each>
             </xsl:variable>
             <out><xsl:value-of select="count(distinct-values($ids))"/></out>
           </xsl:template>"#,
    );
    assert_eq!(text(run_without_source(&xsl)), "<out>10</out>");
}

// ---------------------------------------------------------------------------------------------
// XML whitespace in stylesheet text (XML 1.0 §2.3, XSLT 3.0 §4.3)
// ---------------------------------------------------------------------------------------------

/// for-each-group-009: `&#xa0;` is not XML whitespace, so a text node made of one is a text node
/// the stylesheet meant to write. Rust's `str::trim` would discard it.
#[test]
fn a_non_breaking_space_is_not_stylesheet_whitespace() {
    let xsl =
        stylesheet(r#"<xsl:template name="xsl:initial-template"><out>&#xa0;</out></xsl:template>"#);
    assert_eq!(text(run_without_source(&xsl)), "<out>\u{a0}</out>");
}

/// The control: ordinary layout whitespace is still stripped.
#[test]
fn layout_whitespace_is_still_stripped() {
    let xsl =
        stylesheet("<xsl:template name=\"xsl:initial-template\"><out>\n   </out></xsl:template>");
    assert_eq!(text(run_without_source(&xsl)), "<out/>");
}

// ---------------------------------------------------------------------------------------------
// The pattern grammar (XSLT 3.0 §5.5.2) and the element syntax summary (§3.7)
// ---------------------------------------------------------------------------------------------

/// match-079: `KeyValue ::= Literal | VarRef`, so an arithmetic expression as the second
/// argument of `key()` at the head of a pattern is `XTSE0340`.
#[test]
fn key_in_a_pattern_rejects_a_computed_second_argument() {
    let xsl = stylesheet(
        r#"<xsl:key name="k" match="*" use="42"/>
           <xsl:template match="key('k', 40+2)//a" priority="5"><out>WRONG</out></xsl:template>
           <xsl:template match="/"><out>OK</out></xsl:template>"#,
    );
    assert_eq!(error_code(run_source(&xsl, "<a/>")), "XTSE0340");
}

/// The values the grammar does admit still compile: a literal, a variable reference, and the
/// third argument that names the document to search.
#[test]
fn key_and_id_patterns_accept_the_values_the_grammar_admits() {
    let xsl = stylesheet(
        r#"<xsl:param name="x" select="'c'"/>
           <xsl:key name="k" match="*" use="@id"/>
           <xsl:template match="key('k', 3)"><a/></xsl:template>
           <xsl:template match="key('k', $x)"><b/></xsl:template>
           <xsl:template match="id('c', $x)"><c/></xsl:template>
           <xsl:template match="/"><out>OK</out></xsl:template>"#,
    );
    assert_eq!(text(run_source(&xsl, "<a/>")), "<out>OK</out>");
}

/// next-match-024: §6.5 splits a union pattern into one template rule per alternative "only if
/// there is no explicit `priority` attribute". With one, the rule is single, so an
/// `xsl:next-match` written inside it moves on rather than matching the template's own second
/// alternative.
#[test]
fn a_declared_priority_keeps_a_union_pattern_one_template_rule() {
    let xsl = stylesheet(
        r#"<xsl:template match="a | b" priority="5"><first/><xsl:next-match/></xsl:template>
           <xsl:template match="a"><second/></xsl:template>
           <xsl:template match="/"><out><xsl:apply-templates select="a"/></out></xsl:template>"#,
    );
    assert_eq!(
        text(run_source(&xsl, "<a/>")),
        "<out><first/><second/></out>"
    );
}

/// next-match-021 and friends: without a declared priority the split still happens, so
/// `xsl:next-match` does reach the template's other alternative.
#[test]
fn a_default_priority_union_pattern_is_several_template_rules() {
    let xsl = stylesheet(
        r#"<xsl:template match="a | *"><first/><xsl:next-match/></xsl:template>
           <xsl:template match="/"><out><xsl:apply-templates select="a"/></out></xsl:template>"#,
    );
    assert_eq!(
        text(run_source(&xsl, "<a/>")),
        "<out><first/><first/></out>"
    );
}

/// The single-rule form has to work for the item patterns too: a `.[…]` pattern is matched
/// against an atomic value through a different entry point from a node's, and both have to agree
/// that a declared priority makes the alternatives one rule rather than recurring between
/// themselves.
#[test]
fn a_declared_priority_pattern_matches_an_atomic_value() {
    let xsl = stylesheet(
        r#"<xsl:template match=".[. = 1]" priority="5" mode="m"><hit>{.}</hit></xsl:template>
           <xsl:template name="xsl:initial-template">
             <out><xsl:apply-templates select="(1, 2)" mode="m"/></out>
           </xsl:template>"#,
    );
    assert_eq!(text(run_without_source(&xsl)), "<out><hit>1</hit>2</out>");
}

// ---------------------------------------------------------------------------------------------
// The initial match selection (XSLT 3.0 §2.3.3)
// ---------------------------------------------------------------------------------------------

/// mode-1801: an apply-templates invocation may name a sequence that is not the source document
/// — here five integers — while the global variables still see the global context item.
#[test]
fn an_initial_match_selection_drives_the_apply_templates_invocation() {
    let xsl = stylesheet(
        r#"<xsl:variable name="root-name" select="name(/*)"/>
           <xsl:template match=".[. instance of xs:integer]" mode="m">
             <v n="{.}" p="{position()}" of="{last()}" root="{$root-name}"/>
           </xsl:template>"#,
    );
    let compiled = Stylesheet::compile_str(&xsl).expect("the stylesheet compiles");
    let source = Tree::parse_str("<doc/>").expect("the source parses");
    let parameters = Parameters::new()
        .with_initial_mode("m")
        .with_initial_match_selection(oxixml_infoset::Sequence::from_items([
            AtomicValue::integer(1).into(),
            AtomicValue::integer(2).into(),
        ]));
    let produced = compiled
        .transform_with(&source.root(), &parameters)
        .expect("the transformation runs");
    assert_eq!(
        produced.as_str(),
        r#"<v n="1" p="1" of="2" root="doc"/><v n="2" p="2" of="2" root="doc"/>"#
    );
}

/// The XTDE0044 guard stays: naming an initial mode with neither a source document nor an
/// initial match selection is still an error.
#[test]
fn an_initial_mode_without_any_selection_is_still_xtde0044() {
    let xsl = stylesheet(r#"<xsl:template match="a" mode="m"><v/></xsl:template>"#);
    let compiled = Stylesheet::compile_str(&xsl).expect("the stylesheet compiles");
    let outcome = compiled
        .transform_optional(None, &Parameters::new().with_initial_mode("m"))
        .map(|produced| produced.as_str().to_owned());
    assert_eq!(error_code(outcome), "XTDE0044");
}

/// function-1018: XSLT 3.0 withdrew the `as` attribute of `xsl:sequence`, so writing one is
/// `XTSE0090`.
#[test]
fn xsl_sequence_rejects_an_as_attribute() {
    let xsl = stylesheet(
        r#"<xsl:function name="x:test">
             <xsl:param name="n"/>
             <xsl:sequence as="xs:integer" select="$n"/>
           </xsl:function>
           <xsl:template match="/">
             <out one="{x:test('banana')}"/>
           </xsl:template>"#,
    );
    assert_eq!(error_code(run_source(&xsl, "<a/>")), "XTSE0090");
}
