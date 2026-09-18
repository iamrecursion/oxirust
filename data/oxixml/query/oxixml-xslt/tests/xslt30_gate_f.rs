//! Regression tests for the XSLT 3.0 conformance-gate mission: every test reproduces one W3C
//! `xslt30-test` failure with a minimal inline stylesheet (or, where the shape matters, against
//! the real fixture), so each fix is provable red-then-green without the separate
//! `oxixml-testsuite` harness.
//!
//! The case identifiers in the comments are the ones the harness reports.

use oxixml_infoset::Tree;
use oxixml_xslt::{Parameters, StaticOptions, Stylesheet, XsltError};

fn stylesheet(body: &str) -> String {
    format!(
        r#"<xsl:stylesheet version="3.0" xmlns:xsl="http://www.w3.org/1999/XSL/Transform"
             xmlns:xs="http://www.w3.org/2001/XMLSchema" xmlns:f="http://example.com/f"
             xmlns:my="http://my.com/" exclude-result-prefixes="xs f my">
             <xsl:output omit-xml-declaration="yes"/>
             {body}
           </xsl:stylesheet>"#
    )
}

fn compile_only(xsl: &str) -> Result<(), XsltError> {
    Stylesheet::compile_str(xsl).map(|_| ())
}

fn run_source(xsl: &str, source: &str) -> Result<String, XsltError> {
    run_source_with(xsl, source, &Parameters::new())
}

/// Runs `xsl` in the *strict* reading of XSLT 3.0 §27.4, where a processor that does not claim
/// the XSLT 1.0 compatibility feature must reject an element whose effective version is 1.0.
/// The library's default is the permissive reading, so the backwards-compatibility tests below
/// say which one they mean; the conformance harness sets the same option, because its own
/// dependency filter declares `backwards_compatibility` unsupported.
fn run_strict(xsl: &str, template: &str) -> Result<String, XsltError> {
    let options = StaticOptions::new().with_backwards_compatibility(false);
    let compiled = Stylesheet::compile_str_with(xsl, options)?;
    let tree = Tree::parse_str("<empty/>")
        .map_err(|error| XsltError::dynamic("XTDE0000", error.message().to_owned()))?;
    let parameters = Parameters::new().with_initial_template(template);
    Ok(compiled
        .transform_with(&tree.root(), &parameters)?
        .as_str()
        .to_owned())
}

fn run_source_with(xsl: &str, source: &str, parameters: &Parameters) -> Result<String, XsltError> {
    let compiled = Stylesheet::compile_str(xsl)?;
    let tree = Tree::parse_str(source)
        .map_err(|error| XsltError::dynamic("XTDE0000", error.message().to_owned()))?;
    Ok(compiled
        .transform_with(&tree.root(), parameters)?
        .as_str()
        .to_owned())
}

fn run_named(body: &str, template: &str) -> Result<String, XsltError> {
    run_source_with(
        &stylesheet(body),
        "<empty/>",
        &Parameters::new().with_initial_template(template),
    )
}

/// Like [`run_named`], but with a genuinely **absent** global context item — no source document
/// at all, per [XSLT 3.0 §2.3.4](https://www.w3.org/TR/xslt-30/#initiating-transformation) —
/// rather than a fabricated one.
fn run_named_without_source(body: &str, template: &str) -> Result<String, XsltError> {
    let compiled = Stylesheet::compile_str(&stylesheet(body))?;
    Ok(compiled
        .transform_optional(None, &Parameters::new().with_initial_template(template))?
        .as_str()
        .to_owned())
}

/// The error code a stylesheet raises, whether while compiling or while running.
fn code_of(body: &str, template: &str) -> String {
    let error = run_named(body, template).expect_err("the stylesheet must fail");
    error.code().unwrap_or_default().to_owned()
}

// === xsl:next-match and the current template rule (XSLT 3.0 §6.7) =============================

/// next-match-031: `xsl:for-each` is on the specification's list of instructions that clear the
/// current template rule, so an `xsl:next-match` inside one is `XTDE0560`. Before the fix the
/// rule survived the focus change and the instruction re-matched the new focus, recursing until
/// the depth guard fired with `XTDE0640`.
#[test]
fn next_match_inside_for_each_has_no_current_template_rule() {
    let xsl = stylesheet(
        r#"<xsl:template match="/"><out><xsl:apply-templates/></out></xsl:template>
           <xsl:template match="b" priority="1"><x><xsl:apply-templates/></x></xsl:template>
           <xsl:template match="b[@test='c']" priority="2">
             <y><xsl:for-each select=".."><xsl:next-match/></xsl:for-each></y>
           </xsl:template>"#,
    );
    let error = run_source(&xsl, r#"<doc><a><b test="c">1</b></a></doc>"#)
        .expect_err("xsl:next-match inside xsl:for-each must fail");
    assert_eq!(error.code(), Some("XTDE0560"), "{error}");
}

/// next-match-029: the same list ends with "`xsl:template` if and only if the called template
/// specifies `<xsl:context-item use="absent"/>`".
#[test]
fn next_match_inside_a_context_item_absent_template_has_no_current_template_rule() {
    let xsl = stylesheet(
        r#"<xsl:template match="/"><out><xsl:apply-templates/></out></xsl:template>
           <xsl:template match="b" priority="1"><x><xsl:apply-templates/></x></xsl:template>
           <xsl:template match="b[@test='c']" priority="2">
             <y><xsl:call-template name="test"/></y>
           </xsl:template>
           <xsl:template name="test">
             <xsl:context-item use="absent"/>
             <z><xsl:next-match/></z>
           </xsl:template>"#,
    );
    let error = run_source(&xsl, r#"<doc><a><b test="c">1</b></a></doc>"#)
        .expect_err("xsl:next-match with an absent focus must fail");
    assert_eq!(error.code(), Some("XTDE0560"), "{error}");
}

/// next-match-030, which already passed: `xsl:copy` clears the rule only when it has a `select`.
/// The guard makes sure the broader `with_focus` change did not simply clear it everywhere.
#[test]
fn a_plain_xsl_copy_keeps_the_current_template_rule() {
    let xsl = stylesheet(
        r#"<xsl:template match="/"><out><xsl:apply-templates/></out></xsl:template>
           <xsl:template match="b" priority="1">low</xsl:template>
           <xsl:template match="b" priority="2"><xsl:copy><xsl:next-match/></xsl:copy></xsl:template>"#,
    );
    assert_eq!(
        run_source(&xsl, "<doc><b/></doc>").expect("xsl:copy keeps the rule"),
        "<out><b>low</b></out>"
    );
}

// === xsl:context-item (XSLT 3.0 §10.1.1) ======================================================

/// error-3090a: `use="required"` and no context item at the `xsl:call-template`.
#[test]
fn a_required_context_item_that_is_absent_is_xtte3090() {
    assert_eq!(
        code_of(
            r#"<xsl:template name="main"><out><xsl:value-of select="f:f()"/></out></xsl:template>
               <xsl:function name="f:f">
                 <xsl:call-template name="t"/>
               </xsl:function>
               <xsl:template name="t">
                 <xsl:context-item use="required"/>
                 <xsl:value-of select="."/>
               </xsl:template>"#,
            "main"
        ),
        "XTTE3090"
    );
}

/// context-item-002: a supplied context item that does not match `as` is `XTTE0590` — "the error
/// code is the same as for `xsl:param`" — with no attempt to convert it.
#[test]
fn a_context_item_of_the_wrong_type_is_xtte0590() {
    assert_eq!(
        code_of(
            r#"<xsl:template name="t">
                 <xsl:context-item as="xs:string"/>
                 <xsl:sequence select="."/>
               </xsl:template>
               <xsl:template name="main">
                 <out><xsl:for-each select="1 to 5"><xsl:call-template name="t"/></xsl:for-each></out>
               </xsl:template>"#,
            "main"
        ),
        "XTTE0590"
    );
}

/// context-item-001: the matching case still runs, and `use="absent"` (context-item-011) drops
/// the focus without complaining that one was supplied.
#[test]
fn a_matching_context_item_is_accepted_and_use_absent_drops_the_focus() {
    assert_eq!(
        run_named(
            r#"<xsl:template name="t">
                 <xsl:context-item as="xs:integer"/>
                 <xsl:sequence select="."/>
               </xsl:template>
               <xsl:template name="main">
                 <out><xsl:for-each select="1 to 5"><xsl:call-template name="t"/></xsl:for-each></out>
               </xsl:template>"#,
            "main"
        )
        .expect("an integer matches xs:integer"),
        "<out>1 2 3 4 5</out>"
    );
    assert_eq!(
        run_named(
            r#"<xsl:template name="t">
                 <xsl:context-item use="absent"/>
                 <xsl:param name="p" select="3"/>
                 <xsl:sequence select="if ($p = 3) then 'no focus' else $p"/>
               </xsl:template>
               <xsl:template name="main">
                 <out><xsl:for-each select="1 to 2"><xsl:call-template name="t"/></xsl:for-each></out>
               </xsl:template>"#,
            "main"
        )
        .expect("use=absent is not an error even with a focus"),
        "<out>no focus no focus</out>"
    );
}

/// context-item-010: "[ERR XTSE3088] It is a static error if the `as` attribute is present when
/// `use="absent"` is specified."
#[test]
fn as_together_with_use_absent_is_xtse3088() {
    let xsl = stylesheet(
        r#"<xsl:template name="t">
             <xsl:context-item as="xs:integer" use="absent"/>
             <xsl:sequence select="1"/>
           </xsl:template>"#,
    );
    let error = compile_only(&xsl).expect_err("as with use=absent is a static error");
    assert_eq!(error.code(), Some("XTSE3088"), "{error}");
}

/// context-item-016: "If the containing `xsl:template` element has no `name` attribute then the
/// only permitted value is `required`" — anything else is outside the attribute's value space.
#[test]
fn use_other_than_required_on_an_unnamed_template_rule_is_xtse0020() {
    for usage in ["absent", "optional"] {
        let xsl = stylesheet(&format!(
            r#"<xsl:template match="doc"><xsl:context-item use="{usage}"/><xsl:copy-of select="."/></xsl:template>"#
        ));
        let error = compile_only(&xsl).expect_err("use={usage} on a rule is a static error");
        assert_eq!(error.code(), Some("XTSE0020"), "use={usage}: {error}");
    }
    // `use="required"` is the permitted one, so it must still compile.
    let xsl = stylesheet(
        r#"<xsl:template match="doc"><xsl:context-item use="required"/><xsl:copy-of select="."/></xsl:template>"#,
    );
    compile_only(&xsl).expect("use=required on a template rule is legal");
}

/// context-item-901: the attribute holds an `ItemTypeXP30`, not a `SequenceType`, so an
/// occurrence indicator is not one of its permitted values.
#[test]
fn an_occurrence_indicator_in_context_item_as_is_xtse0020() {
    let xsl = stylesheet(
        r#"<xsl:template name="t"><xsl:context-item as="xs:integer?"/><xsl:sequence select="."/></xsl:template>"#,
    );
    let error = compile_only(&xsl).expect_err("an occurrence indicator is a static error");
    assert_eq!(error.code(), Some("XTSE0020"), "{error}");
}

/// context-item-008: `<initial-template name="main">` with **no** `<environment>` at all — the
/// catalogue asks for a genuinely absent global context item, not a fabricated document. `main`
/// declares no `xsl:context-item` of its own, so it just inherits the absent focus; each
/// `xsl:for-each` iteration gives `t` a definite `xs:integer` focus (`use="optional"` accepts
/// it), and the final call outside the loop reaches `t` with the absent focus again — legal
/// because `use="optional"` does not require one, and `.` is never evaluated once `$p` is
/// overridden.
#[test]
fn a_named_initial_template_runs_with_no_global_context_item_at_all() {
    let xsl = r#"<xsl:template name="t">
                   <xsl:context-item use="optional" as="xs:integer"/>
                   <xsl:param name="p" select="3"/>
                   <xsl:sequence select="if ($p = 3) then . else $p"/>
                 </xsl:template>
                 <xsl:template name="main">
                   <out>
                     <xsl:for-each select="1 to 5"><xsl:call-template name="t"/></xsl:for-each>
                     <xsl:call-template name="t"><xsl:with-param name="p" select="6"/></xsl:call-template>
                   </out>
                 </xsl:template>"#;
    assert_eq!(
        run_named_without_source(xsl, "main").expect("an absent, unused focus is not an error"),
        "<out>1 2 3 4 5 6</out>"
    );
}

/// context-item-009: the same shape as context-item-008, but `t`'s `<xsl:context-item as="xs:integer"/>`
/// leaves `use` at its default — which context-item-009's own catalogue description calls
/// "optional defaulted" — so an absent, untouched focus must still not be an error.
#[test]
fn a_defaulted_optional_context_item_tolerates_an_absent_unused_focus() {
    let xsl = r#"<xsl:template name="t">
                   <xsl:context-item as="xs:integer"/>
                   <xsl:param name="p" select="3"/>
                   <xsl:sequence select="if ($p = 3) then . else $p"/>
                 </xsl:template>
                 <xsl:template name="main">
                   <out>
                     <xsl:for-each select="1 to 5"><xsl:call-template name="t"/></xsl:for-each>
                     <xsl:call-template name="t"><xsl:with-param name="p" select="6"/></xsl:call-template>
                   </out>
                 </xsl:template>"#;
    assert_eq!(
        run_named_without_source(xsl, "main").expect("an absent, unused focus is not an error"),
        "<out>1 2 3 4 5 6</out>"
    );
}

/// context-item-014: `main` itself declares `<xsl:context-item as="document-node(element(doc))"
/// use="required"/>` and is reached as the initial template with no context item supplied at
/// all, so instantiating `main` must fail with `XTTE3090` before its body (an `apply-templates`
/// that would otherwise need the missing document) ever runs.
#[test]
fn a_required_context_item_on_the_initial_template_itself_is_xtte3090() {
    let xsl = r#"<xsl:template match="doc">
                   <xsl:context-item use="required"/>
                   <xsl:copy-of select="."/>
                 </xsl:template>
                 <xsl:template name="main">
                   <xsl:context-item as="document-node(element(doc))" use="required"/>
                   <out><xsl:apply-templates select="doc"/></out>
                 </xsl:template>"#;
    let error = run_named_without_source(xsl, "main")
        .expect_err("a required context item that was never supplied must fail");
    assert_eq!(error.code(), Some("XTTE3090"), "{error}");
}

// === Implicitly mandatory parameters (XSLT 3.0 §9.3) ==========================================

/// error-0610b: "If a parameter has an implicit default value which cannot be converted to the
/// required type … then the parameter is implicitly mandatory", and invoking a template without
/// supplying such a parameter is `XTDE0700` — not `XTTE0600`, which is about a default that
/// exists and is wrong.
#[test]
fn an_implicitly_mandatory_template_parameter_is_xtde0700() {
    let xsl = stylesheet(
        r#"<xsl:template match="/"><out><xsl:apply-templates/></out></xsl:template>
           <xsl:template match="doc">
             <xsl:param name="p" as="xs:integer"/>
             <xsl:value-of select="$p"/>
           </xsl:template>"#,
    );
    let error = run_source(&xsl, "<doc/>").expect_err("the parameter is implicitly mandatory");
    assert_eq!(error.code(), Some("XTDE0700"), "{error}");
}

/// error-0600a, the neighbouring case: a parameter that *does* have a default which cannot
/// satisfy its own type keeps `XTTE0600`.
#[test]
fn a_bad_explicit_default_is_still_xtte0600() {
    let xsl = stylesheet(
        r#"<xsl:template match="/"><out><xsl:apply-templates/></out></xsl:template>
           <xsl:template match="doc">
             <xsl:param name="p" as="xs:integer" select="3.5"/>
             <xsl:value-of select="$p"/>
           </xsl:template>"#,
    );
    let error = run_source(&xsl, "<doc/>").expect_err("3.5 is not an xs:integer");
    assert_eq!(error.code(), Some("XTTE0600"), "{error}");
}

/// error-0610d: the same rule for a stylesheet parameter.
#[test]
fn an_implicitly_mandatory_stylesheet_parameter_is_xtde0700() {
    let xsl = stylesheet(
        r#"<xsl:param name="records" as="xs:integer"/>
           <xsl:template name="initial"><out><xsl:value-of select="$records"/></out></xsl:template>"#,
    );
    let error = run_source_with(
        &xsl,
        "<empty/>",
        &Parameters::new().with_initial_template("initial"),
    )
    .expect_err("no value was supplied");
    assert_eq!(error.code(), Some("XTDE0700"), "{error}");
    // Supplying one makes the same stylesheet run.
    assert_eq!(
        run_source_with(
            &xsl,
            "<empty/>",
            &Parameters::new()
                .with_initial_template("initial")
                .with_parameter("records", oxixml_infoset::AtomicValue::integer(4_i64)),
        )
        .expect("a supplied value satisfies the type"),
        "<out>4</out>"
    );
}

/// error-0050a: `required="yes"` on a stylesheet parameter keeps its own code, `XTDE0050`.
#[test]
fn an_explicitly_required_stylesheet_parameter_is_still_xtde0050() {
    let xsl = stylesheet(
        r#"<xsl:param name="x" required="yes"/>
           <xsl:template name="initial"><out><xsl:value-of select="$x"/></out></xsl:template>"#,
    );
    let error = run_source_with(
        &xsl,
        "<empty/>",
        &Parameters::new().with_initial_template("initial"),
    )
    .expect_err("no value was supplied");
    assert_eq!(error.code(), Some("XTDE0050"), "{error}");
}

// === Stylesheet function arguments (XSLT 3.0 §10.3) ===========================================

/// error-0790a3: XSLT 3.0 dropped `XTTE0790`; a call on a stylesheet function is an ordinary
/// XPath function call, so an argument that cannot be converted is `XPTY0004`.
#[test]
fn a_stylesheet_function_argument_of_the_wrong_type_is_xpty0004() {
    assert_eq!(
        code_of(
            r#"<xsl:template name="main"><out><xsl:sequence select="my:f(2)"/></out></xsl:template>
               <xsl:function name="my:f">
                 <xsl:param name="x" as="xs:date"/>
                 <xsl:sequence select="$x"/>
               </xsl:function>"#,
            "main"
        ),
        "XPTY0004"
    );
}

// === The scope of a global variable (XSLT 3.0 §9.9, §9.11) ====================================

/// variable-0120 (and error-0640b, whose catalogue entry calls this code "arguably preferable"):
/// a global variable is not visible within its own declaration, so naming itself there is the
/// static error `XPST0008`, not the dynamic circularity `XTDE0640`.
#[test]
fn a_global_variable_that_names_itself_is_xpst0008() {
    let xsl = stylesheet(
        r#"<xsl:variable name="v" select="$v+1"/>
           <xsl:template match="/"><out><xsl:value-of select="$v"/></out></xsl:template>"#,
    );
    let error = compile_only(&xsl).expect_err("a self-reference is a static error");
    assert_eq!(error.code(), Some("XPST0008"), "{error}");
}

/// error-0640c: a cycle that runs through *another* declaration is still `XTDE0640`.
#[test]
fn an_indirect_cycle_between_global_variables_is_still_xtde0640() {
    let xsl = stylesheet(
        r#"<xsl:variable name="x" select="$y+1"/>
           <xsl:variable name="y" select="$z+1"/>
           <xsl:variable name="z" select="$x+1"/>
           <xsl:template match="/"><out><xsl:value-of select="$x"/></out></xsl:template>"#,
    );
    let error = run_source(&xsl, "<doc/>").expect_err("the declarations are circular");
    assert_eq!(error.code(), Some("XTDE0640"), "{error}");
}

/// A range variable that rebinds the name is not a self-reference: the check asks the XPath
/// parser, which knows about shadowing, rather than scanning for the name.
#[test]
fn a_range_variable_shadowing_the_global_name_is_not_a_self_reference() {
    let xsl = stylesheet(
        r#"<xsl:variable name="v" select="for $v in 1 to 3 return $v * 2"/>
           <xsl:template match="/"><out><xsl:value-of select="$v"/></out></xsl:template>"#,
    );
    assert_eq!(
        run_source(&xsl, "<doc/>").expect("the range variable shadows the global"),
        "<out>2 4 6</out>"
    );
}

// === Backwards compatible processing (XSLT 3.0 §3.9) ==========================================

/// error-0160a and initial-template-081: an element whose effective version is 1.0 is
/// `XTDE0160` on a processor that does not implement XSLT 1.0 behaviour.
#[test]
fn evaluating_an_element_with_effective_version_1_0_is_xtde0160() {
    let xsl =
        stylesheet(r#"<xsl:template name="main"><out><x xsl:version="1.0"/></out></xsl:template>"#);
    let error = run_strict(&xsl, "main").expect_err("1.0 code may not be evaluated");
    assert_eq!(error.code(), Some("XTDE0160"), "{error}");
    // The default reading accepts it instead of refusing to run the stylesheet at all.
    assert_eq!(
        run_named(
            r#"<xsl:template name="main"><out><x xsl:version="1.0"/></out></xsl:template>"#,
            "main"
        )
        .expect("the permissive default evaluates 1.0 code"),
        "<out><x/></out>"
    );
}

/// initial-template-080: "A processor that does not support backwards compatibility mode raises
/// no error if no 1.0 code is executed" — the error is dynamic precisely so an unvisited 1.0
/// branch costs nothing.
#[test]
fn an_unevaluated_version_1_0_branch_raises_nothing() {
    let xsl = r#"<xsl:stylesheet xmlns:xsl="http://www.w3.org/1999/XSL/Transform" version="1.0">
                   <xsl:output omit-xml-declaration="yes"/>
                   <xsl:template name="main" version="2.0"><out><xsl:value-of select="23"/></out></xsl:template>
                   <xsl:template name="other" version="1.0"><out><xsl:value-of select="25"/></out></xsl:template>
                 </xsl:stylesheet>"#;
    assert_eq!(
        run_strict(xsl, "main").expect("the 2.0 template is fine"),
        "<out>23</out>"
    );
    let error = run_strict(xsl, "other").expect_err("the 1.0 template is not");
    assert_eq!(error.code(), Some("XTDE0160"), "{error}");
}

/// The `version` attribute of `xsl:output` names a version of the *output method*; §3.9
/// excludes it from the effective version, so it must not turn the stylesheet into 1.0 code.
#[test]
fn the_version_attribute_of_xsl_output_is_not_an_xslt_version() {
    let xsl = r#"<xsl:stylesheet xmlns:xsl="http://www.w3.org/1999/XSL/Transform" version="3.0">
                   <xsl:output method="xml" version="1.0" omit-xml-declaration="yes"/>
                   <xsl:template match="/"><out>ok</out></xsl:template>
                 </xsl:stylesheet>"#;
    assert_eq!(
        run_source(xsl, "<doc/>").expect("xsl:output/@version is not an XSLT version"),
        "<out>ok</out>"
    );
}

// === Serializing a map (output-0710/0711/0712) ================================================

/// A map that reaches the *serializer* is the serialization error `SENR0001`; one that reaches
/// *complex content construction* stays `XTDE0450`.
#[test]
fn a_map_at_the_top_of_the_principal_result_is_senr0001() {
    assert_eq!(
        code_of(
            r#"<xsl:template name="main">
                 <xsl:variable name="maps" as="map(*)*">
                   <xsl:for-each select="1 to 5">
                     <xsl:map><xsl:map-entry key="'index'"><xsl:sequence select="."/></xsl:map-entry></xsl:map>
                   </xsl:for-each>
                 </xsl:variable>
                 <xsl:sequence select="$maps"/>
               </xsl:template>"#,
            "main"
        ),
        "SENR0001"
    );
}

/// The same map inside a constructed element is a tree-construction fault, `XTDE0450`.
#[test]
fn a_map_inside_constructed_content_is_still_xtde0450() {
    assert_eq!(
        code_of(
            r#"<xsl:template name="main">
                 <out><xsl:sequence select="map{'a': 1}"/></out>
               </xsl:template>"#,
            "main"
        ),
        "XTDE0450"
    );
}

// === xsl:analyze-string's focus (XSLT 3.0 §15) ================================================

/// result-document-1502 turns on this: "the position of the substring within the sequence of
/// matching and non-matching substrings will be the context position; and the number of matching
/// and non-matching substrings will be the context size". Both were hard-coded to 1.
#[test]
fn analyze_string_numbers_the_substrings_it_processes() {
    assert_eq!(
        run_named(
            r#"<xsl:template name="main">
                 <out><xsl:analyze-string select="'aXbXc'" regex="X">
                   <xsl:matching-substring><m p="{position()}" l="{last()}"/></xsl:matching-substring>
                   <xsl:non-matching-substring><n p="{position()}" l="{last()}"/></xsl:non-matching-substring>
                 </xsl:analyze-string></out>
               </xsl:template>"#,
            "main"
        )
        .expect("xsl:analyze-string runs"),
        r#"<out><n p="1" l="5"/><m p="2" l="5"/><n p="3" l="5"/><m p="4" l="5"/><n p="5" l="5"/></out>"#
    );
}

// === The default entry point (XSLT 3.0 §2.3.3, §2.3.4) ========================================

/// coco-102: a template named `xsl:initial-template` that is *also* a template rule is reached
/// by matching when the caller supplies a source, so it becomes the current template rule and an
/// `xsl:next-match` inside it works.
#[test]
fn an_initial_template_that_is_also_a_rule_becomes_the_current_template_rule() {
    let xsl = stylesheet(
        r#"<xsl:template match="/" name="xsl:initial-template"><out><xsl:next-match/></out></xsl:template>
           <xsl:template match="/" priority="-1">fallback</xsl:template>"#,
    );
    assert_eq!(
        run_source(&xsl, "<doc/>").expect("the rule is current"),
        "<out>fallback</out>"
    );
}

/// A template named `xsl:initial-template` with no `match` is still the call-template entry
/// point, which is how most of the suite's named-entry-point cases run.
#[test]
fn a_named_only_initial_template_is_still_the_default_entry_point() {
    let xsl = stylesheet(
        r#"<xsl:template name="xsl:initial-template"><out>entered</out></xsl:template>
           <xsl:template match="/">not this one</xsl:template>"#,
    );
    assert_eq!(
        run_source(&xsl, "<doc/>").expect("the named entry point runs"),
        "<out>entered</out>"
    );
}

// === The instantiation-depth guard ============================================================

/// call-template-1003 recurses a thousand levels deep on purpose. The ceiling is a resource
/// limit, not a language rule, and it is sized for an ordinary 8 MiB stack — measured, one level
/// of template instantiation costs about 7 KiB of native stack in a release build and well over
/// ten times that in a debug one — so a caller running on a bigger stack raises it rather than
/// the library guessing. The depths here are deliberately tiny: what is under test is that the
/// parameter moves the ceiling, and recursing far in a *debug* build — where the measured cost
/// is around 100 KiB per level — would overflow a test thread's own stack long before the
/// ceiling was reached. The thousand-level case itself is covered by the conformance harness,
/// which runs on a 256 MiB thread.
///
/// Six levels, against a declared ceiling of three: the *native-stack* guard — two thirds of the
/// 2 MiB a thread is assumed to have — is not what this test is about, and at a debug build's
/// hundred-odd kilobytes per level it is the guard that would fire first if the depth were much
/// larger. Six leaves it roughly half the budget of headroom.
#[test]
fn the_recursion_ceiling_is_a_parameter() {
    let xsl = stylesheet(
        r#"<xsl:template name="main">
             <out><xsl:call-template name="rec"><xsl:with-param name="n" select="6"/></xsl:call-template></out>
           </xsl:template>
           <xsl:template name="rec">
             <xsl:param name="n" as="xs:integer"/>
             <xsl:if test="$n gt 0">
               <xsl:call-template name="rec"><xsl:with-param name="n" select="$n - 1"/></xsl:call-template>
             </xsl:if>
           </xsl:template>"#,
    );
    let run = |limit: Option<usize>| {
        let mut parameters = Parameters::new().with_initial_template("main");
        if let Some(limit) = limit {
            parameters = parameters.with_recursion_limit(limit);
        }
        run_source_with(&xsl, "<empty/>", &parameters)
    };
    let error = run(Some(3)).expect_err("6 levels is past a ceiling of 3");
    assert_eq!(error.code(), Some("XTDE0640"), "{error}");
    assert_eq!(
        run(Some(200)).expect("a ceiling of 200 admits 6 levels"),
        "<out/>"
    );
    assert_eq!(
        run(None).expect("the default ceiling admits 6 levels"),
        "<out/>"
    );
    assert_eq!(Parameters::new().recursion_limit(), None);
    assert_eq!(
        Parameters::new().with_recursion_limit(7).recursion_limit(),
        Some(7)
    );
}

// === fn:key and the calling focus (XSLT 3.0 §18.3) ============================================

/// key-072: the two-argument `fn:key` identifies the document to search from "the context node",
/// and a path step rebinds it — `$e/key('key1','s')` searches `$e`'s tree even though the
/// expression is written inside an `xsl:function`, which has no focus of its own.
#[test]
fn key_honours_a_context_node_rebound_by_a_path_step() {
    let xsl = stylesheet(
        r#"<xsl:key name="key1" match="*" use="@name"/>
           <xsl:template match="/">
             <testing><xsl:if test="f:test(.)"><xsl:text>true</xsl:text></xsl:if></testing>
           </xsl:template>
           <xsl:function name="f:test" as="xs:boolean">
             <xsl:param as="document-node()" name="e"/>
             <xsl:sequence select="exists($e/key('key1','s'))"/>
           </xsl:function>"#,
    );
    assert_eq!(
        run_source(&xsl, r#"<doc><chap name="s"/></doc>"#).expect("key-072 runs"),
        "<testing>true</testing>"
    );
}

/// error-1270a, the neighbouring case: with no context node anywhere — neither at the call site
/// nor in the transformation — the two-argument form is `XTDE1270`, not a silent empty result.
#[test]
fn key_with_no_context_node_at_all_is_still_xtde1270() {
    assert_eq!(
        code_of(
            r#"<xsl:key name="key1" match="*" use="@name"/>
               <xsl:template name="main"><out><xsl:value-of select="f:test()"/></out></xsl:template>
               <xsl:function name="f:test">
                 <xsl:sequence select="exists(key('key1','s'))"/>
               </xsl:function>"#,
            "main"
        ),
        "XTDE1270"
    );
}
