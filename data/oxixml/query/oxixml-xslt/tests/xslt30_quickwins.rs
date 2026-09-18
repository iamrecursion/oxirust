//! Regression tests for the XSLT 3.0 "quick wins" mission: each test reproduces one W3C
//! `xslt30-test` failure (see the recon dump for the exact case IDs) with a minimal inline
//! stylesheet, so every fix is provable red-then-green without depending on the separate
//! `oxixml-testsuite` harness.

use oxixml_infoset::Tree;
use oxixml_xslt::{Stylesheet, XsltError};

fn stylesheet_v20(body: &str) -> String {
    format!(
        r#"<xsl:stylesheet version="2.0" xmlns:xsl="http://www.w3.org/1999/XSL/Transform">
             <xsl:output omit-xml-declaration="yes"/>
             {body}
           </xsl:stylesheet>"#
    )
}

fn stylesheet_v10(body: &str) -> String {
    format!(
        r#"<xsl:stylesheet version="1.0" xmlns:xsl="http://www.w3.org/1999/XSL/Transform">
             <xsl:output omit-xml-declaration="yes"/>
             {body}
           </xsl:stylesheet>"#
    )
}

fn run_with(xsl: &str, source: &str) -> Result<String, XsltError> {
    let compiled = Stylesheet::compile_str(xsl)?;
    let tree = Tree::parse_str(source)
        .map_err(|error| XsltError::dynamic("XTDE0000", error.message().to_owned()))?;
    compiled.transform_to_string(&tree.root())
}

fn run_v20(body: &str, source: &str) -> Result<String, XsltError> {
    run_with(&stylesheet_v20(body), source)
}

fn check_v20(body: &str, source: &str, expected: &str) {
    match run_v20(body, source) {
        Ok(actual) => assert_eq!(actual, expected, "stylesheet body: {body}"),
        Err(error) => panic!("{error} (stylesheet body: {body})"),
    }
}

// === Fix 1: forwards-compatible XPath grammar for version="2.0" modules ========================
//
// `function-0302b`, `function-0303`, `copy-3803`, `analyze-string-100` all declare
// `version="2.0"` but use XPath 3.0/3.1 syntax (`||`, `!`). XSLT 3.0 §3.9 reserves
// grammar-restricting backwards-compatible behavior for `version="1.0"` alone; 2.0, 3.0 and 3.1
// "differ only in which grammar productions are accepted" (`XPathVersion` docs), so a
// `version="2.0"` module on this (3.0) processor must compile at the implemented level.

#[test]
fn a_version_2_module_accepts_the_xpath_3_1_string_concatenation_operator() {
    // function-0302b / function-0303: `element-available($xsl || 'value-of')`.
    check_v20(
        r#"<xsl:template match="/"><xsl:param name="xsl">xsl:</xsl:param>
             <out><xsl:value-of select="element-available($xsl || 'value-of')"/></out>
           </xsl:template>"#,
        "<in/>",
        "<out>true</out>",
    );
}

#[test]
fn a_version_2_module_accepts_the_xpath_3_0_simple_map_operator() {
    // analyze-string-100: `1!regex-group(.)` inside xsl:matching-substring.
    check_v20(
        r#"<xsl:template match="/">
             <out><xsl:analyze-string select="'a1'" regex="([a-z])([0-9])">
               <xsl:matching-substring><one><xsl:value-of select="1!regex-group(.)"/></one></xsl:matching-substring>
             </xsl:analyze-string></out>
           </xsl:template>"#,
        "<in/>",
        "<out><one>a</one></out>",
    );
}

#[test]
fn a_version_2_module_accepts_a_string_concatenation_after_a_following_sibling_axis() {
    // copy-3803: `$temp//z3/following-sibling::node()/('['||local-name(.)||']')`.
    check_v20(
        r#"<xsl:template match="/">
             <xsl:variable name="temp"><z1/><z2/><z3/><z4/></xsl:variable>
             <out><xsl:value-of select="$temp/z1/following-sibling::node()/('['||local-name(.)||']')" separator=""/></out>
           </xsl:template>"#,
        "<in/>",
        "<out>[z2][z3][z4]</out>",
    );
}

#[test]
fn a_version_1_module_still_rejects_xpath_3_syntax() {
    // Regression guard: true 1.0 backwards-compatible processing is unaffected by the fix above
    // — `version="1.0"` still gets the XPath 1.0 grammar (and value model), so a 3.0-only
    // operator is a static error, not silently accepted.
    let xsl = stylesheet_v10(
        r#"<xsl:template match="/"><out><xsl:value-of select="'a'||'b'"/></out></xsl:template>"#,
    );
    let error = run_with(&xsl, "<in/>").expect_err("version=\"1.0\" must reject `||`");
    assert_eq!(error.code(), Some("XPST0003"));
}

// === Fix 2: fn:current-output-uri() =============================================================

use oxixml_xslt::{StaticOptions, Transformation};

fn run_v20_with_base(body: &str, source: &str, base: &str) -> Result<Transformation, XsltError> {
    let compiled = Stylesheet::compile_str_with(
        &stylesheet_v20(body),
        StaticOptions::new().with_base_uri(base),
    )?;
    let tree = Tree::parse_str(source)
        .map_err(|error| XsltError::dynamic("XTDE0000", error.message().to_owned()))?;
    compiled.transform(&tree.root())
}

#[test]
fn current_output_uri_reports_the_open_result_document_and_is_absent_for_the_principal_output() {
    // current-output-uri-002/-003 (shape): the empty sequence for the principal output — this
    // processor has no destination URI to report for it — and the resolved `href` while
    // xsl:result-document's own content is being evaluated.
    let transformation = run_v20_with_base(
        r#"<xsl:template match="/">
             <out>principal:<xsl:value-of select="string-join(current-output-uri(), '|')"/></out>
             <xsl:result-document href="second/out.xml">
               <out><xsl:value-of select="current-output-uri()"/></out>
             </xsl:result-document>
           </xsl:template>"#,
        "<in/>",
        "http://example.com/base/style.xsl",
    )
    .unwrap_or_else(|error| panic!("{error}"));
    assert_eq!(transformation.as_str(), "<out>principal:</out>");
    let secondary = transformation.secondary_results();
    assert_eq!(secondary.len(), 1);
    let (href, text) = &secondary[0];
    assert_eq!(href, "http://example.com/base/second/out.xml");
    assert_eq!(text, &format!("<out>{href}</out>"));
}

#[test]
fn current_output_uri_is_absent_while_constructing_a_nested_variables_content() {
    // current-output-uri-001/-005 (shape): *constructing the content* of an xsl:variable (or
    // an xsl:function's result) is temporary output state, so current-output-uri() is the
    // empty sequence there — even with an xsl:result-document open around it — because it is
    // no longer building the final result tree the open href names. (A `select`-form
    // reference, by contrast, is evaluated in whatever *ambient* state already held — see
    // current-output-uri-004 in the suite — which is exactly what the previous test's
    // `<out><xsl:value-of select="current-output-uri()"/></out>` exercises: no enclosing
    // variable/function, so it sees the open href.)
    let transformation = run_v20_with_base(
        r#"<xsl:template match="/">
             <xsl:result-document href="second/out.xml">
               <xsl:variable name="v"><xsl:value-of select="string-join(current-output-uri(), '|')"/></xsl:variable>
               <out><xsl:value-of select="$v"/></out>
             </xsl:result-document>
           </xsl:template>"#,
        "<in/>",
        "http://example.com/base/style.xsl",
    )
    .unwrap_or_else(|error| panic!("{error}"));
    let secondary = transformation.secondary_results();
    assert_eq!(secondary.len(), 1);
    assert_eq!(secondary[0].1, "<out/>");
}

// === Fix 3: fn:copy-of() / fn:snapshot() ========================================================

fn stylesheet_v30(body: &str) -> String {
    format!(
        r#"<xsl:stylesheet version="3.0" xmlns:xsl="http://www.w3.org/1999/XSL/Transform">
             <xsl:output omit-xml-declaration="yes"/>
             {body}
           </xsl:stylesheet>"#
    )
}

#[test]
fn copy_of_explicit_and_defaulted_context_item_agree() {
    // unparsed-entity-uri-07 (shape), corrected: `copy-of(.)` and the zero-argument
    // `copy-of()` both build a *fresh* document-node copy of the whole source tree.
    check_v20(
        r#"<xsl:template match="/">
             <xsl:variable name="p1" as="document-node()" select="copy-of(.)"/>
             <xsl:variable name="p2" as="document-node()" select="copy-of()"/>
             <out>
               <a><xsl:value-of select="boolean($p1 is .)"/></a>
               <b><xsl:value-of select="$p1/*/local-name()"/></b>
               <c><xsl:value-of select="$p1/*/@id"/></c>
               <d><xsl:value-of select="boolean($p2 is $p1)"/></d>
             </out>
           </xsl:template>"#,
        "<r id='7'/>",
        "<out><a>false</a><b>r</b><c>7</c><d>false</d></out>",
    );
}

#[test]
fn snapshot_of_a_descendant_keeps_its_ancestors_navigable() {
    // unparsed-entity-uri-08 (shape): `snapshot(.)` at the top level of a `match="/"` template.
    // Applied to a non-root node, `snapshot` additionally keeps the ancestor path — `root()`
    // reaches a copied document, not the original.
    check_v20(
        r#"<xsl:template match="/r/child">
             <xsl:variable name="s" select="snapshot(.)"/>
             <out>
               <a><xsl:value-of select="local-name($s)"/></a>
               <b><xsl:value-of select="boolean($s/root() is /)"/></b>
               <c><xsl:value-of select="$s/root()/r/@id"/></c>
             </out>
           </xsl:template>"#,
        "<r id='7'><child/></r>",
        "<out><a>child</a><b>false</b><c>7</c></out>",
    );
}

#[test]
fn copy_of_and_snapshot_are_registered_at_zero_and_one_arity() {
    // Direct RED-then-GREEN evidence for the registration itself, independent of the richer
    // assertions above: before this fix, every one of these four calls raised XPST0017 ("no
    // function copy-of/snapshot of arity 0/1 is in scope") — the exact failure recon recorded
    // for unparsed-entity-uri-07/08.
    for call in ["copy-of(.)", "copy-of()", "snapshot(.)", "snapshot()"] {
        let xsl = stylesheet_v30(&format!(
            r#"<xsl:template match="/"><out><xsl:value-of
                select="exists({call})"/></out></xsl:template>"#
        ));
        let result = run_with(&xsl, "<r/>");
        assert!(
            result.is_ok(),
            "`{call}` should compile and run, got {:?}",
            result.err()
        );
    }
}

#[test]
fn unparsed_entity_uri_07_and_08_against_the_real_suite_files() {
    // Honesty check, not a pass/fail assertion: runs the *actual* W3C `unparsed-entity-07` and
    // `-08` stylesheets and source document (env-guarded — skips silently without the suite)
    // to record what this fix does and does not change about their outcome.
    //
    // Both cases carry `<dependencies><feature value="dtd"/></dependencies>`: this processor's
    // data model does not retain a source document's DTD (`unparsed-entity-uri()` and
    // `unparsed-entity-public-id()` always report "no such entity", by design — see
    // `run/functions.rs`), so `/out/a` and `/out/d` cannot come out as the suite expects
    // regardless of `copy-of`/`snapshot`. What *this* fix repairs is the `XPST0017` that used
    // to abort the transformation outright; the DTD gap is separate, pre-existing and out of
    // this mission's scope.
    let Ok(suites_dir) = std::env::var("OXIXML_XML_SUITES_DIR") else {
        return;
    };
    let base = std::path::Path::new(&suites_dir).join("xslt30-test/tests/fn/unparsed-entity-uri");
    let source_path = base.join("unparsed-entity-A.xml");
    let Ok(source_xml) = std::fs::read_to_string(&source_path) else {
        return;
    };
    let source = Tree::parse_str(&source_xml).expect("unparsed-entity-A.xml parses");

    for (case, function) in [
        ("unparsed-entity-07", "copy-of"),
        ("unparsed-entity-08", "snapshot"),
    ] {
        let xsl_path = base.join(format!("{case}.xsl"));
        let Ok(xsl) = std::fs::read_to_string(&xsl_path) else {
            return;
        };
        let compiled = Stylesheet::compile_str(&xsl)
            .unwrap_or_else(|error| panic!("{case} ({function}) failed to compile: {error}"));
        let result = compiled
            .transform_to_string(&source.root())
            .unwrap_or_else(|error| {
                panic!(
                    "{case} ({function}) no longer raises XPST0017, but now fails to run: {error}"
                )
            });
        // The transformation now COMPLETES — this is the fix: before it, both cases raised
        // `XPST0017` at compile time and never reached this point at all. Whether the assertion
        // the suite makes (`/out/a = 'true'`, `/out/d = '-//Textuality...'`) is met depends on
        // the unimplemented DTD/unparsed-entity gap, which is why this test only records the
        // result rather than asserting the suite's own expectation.
        // Measured (2026): `/out` comes out `<a>false</a><b/><c>false</c><d/>` for both cases —
        // `/out/b = ''` and `/out/c = 'false'` now match the suite, but `/out/a` and `/out/d`
        // still do not, exactly because of the DTD gap described above. The case is therefore
        // still reported as failed by a harness that checks the suite's own `<all-of>`
        // assertion — this fix changes *why* (a real assertion mismatch, not a crash), not the
        // case's headline pass/fail — which is why this test checks structural completion only.
        assert!(
            result.contains("<out>"),
            "{case} ({function}) produced no <out> element at all: {result}"
        );
    }
}

// === Fix 4: xsl:message must not abort the transformation ======================================

#[test]
fn a_dynamic_error_in_messages_content_does_not_abort_the_transformation() {
    // message-0404: `<xsl:message><msg><xsl:value-of select="100 div $zero"/></msg></xsl:message>`
    // — division by zero (FOAR0001) while building the message's content, terminate="no"
    // (the default). Bug 30283: the spec is explicit that this must not fail the transformation.
    check_v20(
        r#"<xsl:param name="zero" select="0"/>
           <xsl:template match="/">
             <out>
               <xsl:message><msg><xsl:value-of select="100 div $zero"/></msg></xsl:message>
               <ran>yes</ran>
             </out>
           </xsl:template>"#,
        "<in/>",
        "<out><ran>yes</ran></out>",
    );
}

#[test]
fn a_non_qname_error_code_falls_back_to_xtmm9000() {
    // message-0406: `error-code="{23 + $zero}CODE"` evaluates to `23CODE`, which is not a
    // lexical QName (a QName never starts with a digit) — this must fall back to the default
    // `XTMM9000` rather than being raised verbatim as a malformed error name.
    let xsl = stylesheet_v30(
        r#"<xsl:param name="zero" select="0"/>
           <xsl:template match="/">
             <out><xsl:message error-code="{23 + $zero}CODE" terminate="yes">The message</xsl:message></out>
           </xsl:template>"#,
    );
    let error = run_with(&xsl, "<in/>").expect_err("terminate=\"yes\" must still terminate");
    assert_eq!(error.code(), Some("XTMM9000"));
}

#[test]
fn messages_that_cannot_be_stringified_still_let_the_transformation_finish() {
    // message-0407/0408/0409 (shape): xsl:message's `select` yields an array, a map, or a
    // function item — none of which has a string value — with terminate="no" (the default).
    // "The transformation succeeds, but the message content is undefined."
    for select in [
        "['a', 'b', 'c']",
        "map{1: true(), 0: false()}",
        "function($x){$x*2}",
    ] {
        let xsl = stylesheet_v30(&format!(
            r#"<xsl:template match="/">
                 <out><xsl:message select="{select}"/><ran>yes</ran></out>
               </xsl:template>"#
        ));
        match run_with(&xsl, "<in/>") {
            Ok(text) => assert_eq!(text, "<out><ran>yes</ran></out>", "select=\"{select}\""),
            Err(error) => panic!("select=\"{select}\" should not abort: {error}"),
        }
    }
}

// === Fix 5: xsl:map / xsl:map-entry ==============================================================

#[test]
fn xsl_map_builds_a_map_from_its_map_entry_children() {
    check_v20(
        r#"<xsl:template match="/">
             <xsl:variable name="m" as="map(*)">
               <xsl:map>
                 <xsl:map-entry key="'a'" select="1"/>
                 <xsl:map-entry key="'b'"><xsl:sequence select="2"/></xsl:map-entry>
               </xsl:map>
             </xsl:variable>
             <out><size><xsl:value-of
               select="map:size($m)" xmlns:map="http://www.w3.org/2005/xpath-functions/map"/></size><a><xsl:value-of
               select="map:get($m, 'a')" xmlns:map="http://www.w3.org/2005/xpath-functions/map"/></a><b><xsl:value-of
               select="map:get($m, 'b')" xmlns:map="http://www.w3.org/2005/xpath-functions/map"/></b></out>
           </xsl:template>"#,
        "<in/>",
        "<out><size>2</size><a>1</a><b>2</b></out>",
    );
}

#[test]
fn xsl_map_merges_maps_produced_by_nested_content() {
    // error-3365a's shape (without triggering the duplicate): xsl:map-entry nested inside
    // xsl:for-each, one level removed from xsl:map itself.
    check_v20(
        r#"<xsl:template match="/">
             <xsl:variable name="v" as="map(*)">
               <xsl:map>
                 <xsl:for-each select="1 to 5">
                   <xsl:map-entry key="." select="true()"/>
                 </xsl:for-each>
               </xsl:map>
             </xsl:variable>
             <out><xsl:value-of select="map:size($v)"
               xmlns:map="http://www.w3.org/2005/xpath-functions/map"/></out>
           </xsl:template>"#,
        "<in/>",
        "<out>5</out>",
    );
}

#[test]
fn xsl_map_rejects_a_duplicate_key_with_xtde3365() {
    // error-3365a: `1 to 20` modulo 10 revisits every key twice.
    let xsl = stylesheet_v30(
        r#"<xsl:template match="/">
             <out>
               <xsl:variable name="v" as="map(*)">
                 <xsl:map>
                   <xsl:for-each select="1 to 20">
                     <xsl:map-entry key=". mod 10" select="true()"/>
                   </xsl:for-each>
                 </xsl:map>
               </xsl:variable>
               <xsl:value-of select="count($v)"/>
             </out>
           </xsl:template>"#,
    );
    let error = run_with(&xsl, "<in/>").expect_err("a duplicate key must be a dynamic error");
    assert_eq!(error.code(), Some("XTDE3365"));
}

#[test]
fn xsl_map_rejects_a_non_map_item_with_xtte3375() {
    // error-3375a: mixing xsl:map-entry (a map) with xsl:sequence (a plain integer) inside
    // xsl:map's content.
    let xsl = stylesheet_v30(
        r#"<xsl:template match="/">
             <out>
               <xsl:variable name="v" as="map(*)">
                 <xsl:map>
                   <xsl:for-each select="1 to 10">
                     <xsl:if test=". lt 5"><xsl:map-entry key="." select="."/></xsl:if>
                     <xsl:if test=". ge 5"><xsl:sequence select="."/></xsl:if>
                   </xsl:for-each>
                 </xsl:map>
               </xsl:variable>
               <xsl:value-of select="count($v)"/>
             </out>
           </xsl:template>"#,
    );
    let error = run_with(&xsl, "<in/>").expect_err("a non-map item must be a type error");
    assert_eq!(error.code(), Some("XTTE3375"));
}

#[test]
fn xsl_map_entry_without_timezone_special_casing_keeps_both_dates() {
    // error-3368a: date keys with and without a timezone are simply distinct keys — the
    // special "incomparable timezone" dynamic error this test once checked for was withdrawn
    // from the spec, so both entries should just coexist.
    check_v20(
        r#"<xsl:template match="/">
             <xsl:variable name="v" as="map(*)" xmlns:xs="http://www.w3.org/2001/XMLSchema">
               <xsl:map>
                 <xsl:map-entry key="xs:date('2014-08-25')" select="0"/>
                 <xsl:map-entry key="xs:date('2014-08-26Z')" select="1"/>
               </xsl:map>
             </xsl:variable>
             <out><xsl:value-of select="map:size($v)"
               xmlns:map="http://www.w3.org/2005/xpath-functions/map"/></out>
           </xsl:template>"#,
        "<in/>",
        "<out>2</out>",
    );
}

#[test]
fn element_available_reports_xsl_map() {
    check_v20(
        r#"<xsl:template match="/"><out><xsl:value-of
             select="element-available('xsl:map') and element-available('xsl:map-entry')"/></out>
           </xsl:template>"#,
        "<in/>",
        "<out>true</out>",
    );
}

// === Fix 6: static detection of param/with-param consistency violations ========================

#[test]
fn a_required_param_with_content_is_a_static_error() {
    // param-0113: `required="yes"` on an `xsl:param` with a non-empty sequence constructor.
    let xsl = stylesheet_v20(
        r#"<xsl:template match="doc">
             <xsl:param name="par1" required="yes">test param</xsl:param>
             <out><xsl:value-of select="$par1"/></out>
           </xsl:template>"#,
    );
    let error = compile_only(&xsl).expect_err("required=\"yes\" with content is a static error");
    assert_eq!(error.code(), Some("XTSE0010"));
    assert_eq!(error.phase(), oxixml_xslt::ErrorPhase::Static);
}

#[test]
fn a_required_param_with_select_is_a_static_error() {
    // param-0111: the same rule for a `select` default instead of content.
    let xsl = stylesheet_v20(
        r#"<xsl:template name="boo">
             <xsl:param name="par1" select="'oops'" required="yes"/>
             <xsl:value-of select="$par1"/>
           </xsl:template>"#,
    );
    let error = compile_only(&xsl).expect_err("required=\"yes\" with select is a static error");
    assert_eq!(error.code(), Some("XTSE0010"));
}

#[test]
fn a_required_param_with_neither_select_nor_content_still_compiles() {
    // The unremarkable, correct form — required="yes" alone — must keep working.
    let xsl = stylesheet_v20(
        r#"<xsl:template name="main">
             <xsl:param name="par1" required="yes"/>
             <xsl:value-of select="$par1"/>
           </xsl:template>"#,
    );
    compile_only(&xsl).unwrap_or_else(|error| panic!("{error}"));
}

#[test]
fn call_template_missing_a_required_parameter_is_a_static_error() {
    // call-template-2101: `ntmp1` declares `pvar3` as required="yes"; this call site supplies
    // only `pvar1`.
    let xsl = stylesheet_v20(
        r#"<xsl:template match="doc">
             <out>
               <xsl:call-template name="ntmp1">
                 <xsl:with-param name="pvar1" select="'x'"/>
               </xsl:call-template>
             </out>
           </xsl:template>
           <xsl:template name="ntmp1">
             <xsl:param name="pvar1" required="no">default1</xsl:param>
             <xsl:param name="pvar2" required="no">default2</xsl:param>
             <xsl:param name="pvar3" required="yes"/>
             <xsl:value-of select="$pvar1"/>
           </xsl:template>"#,
    );
    let error = compile_only(&xsl).expect_err("a missing required parameter is a static error");
    assert_eq!(error.code(), Some("XTSE0690"));
    assert_eq!(error.phase(), oxixml_xslt::ErrorPhase::Static);
}

#[test]
fn call_template_supplying_every_required_parameter_still_compiles() {
    let xsl = stylesheet_v20(
        r#"<xsl:template match="doc">
             <out>
               <xsl:call-template name="ntmp1">
                 <xsl:with-param name="pvar3" select="'x'"/>
               </xsl:call-template>
             </out>
           </xsl:template>
           <xsl:template name="ntmp1">
             <xsl:param name="pvar3" required="yes"/>
             <xsl:value-of select="$pvar3"/>
           </xsl:template>"#,
    );
    compile_only(&xsl).unwrap_or_else(|error| panic!("{error}"));
}

#[test]
fn a_different_call_site_that_does_supply_it_does_not_mask_the_one_that_does_not() {
    // Regression guard for the per-call-site grouping fix: one call site supplying `pvar3`
    // must not hide that a *different* call site to the same template does not.
    let xsl = stylesheet_v20(
        r#"<xsl:template match="doc">
             <out>
               <xsl:call-template name="ntmp1">
                 <xsl:with-param name="pvar3" select="'ok'"/>
               </xsl:call-template>
               <xsl:call-template name="ntmp1"/>
             </out>
           </xsl:template>
           <xsl:template name="ntmp1">
             <xsl:param name="pvar3" required="yes"/>
             <xsl:value-of select="$pvar3"/>
           </xsl:template>"#,
    );
    let error = compile_only(&xsl).expect_err("the second, param-less call site must still fail");
    assert_eq!(error.code(), Some("XTSE0690"));
}

#[test]
fn a_default_value_that_violates_its_own_type_is_xtte0600_not_xtte0590() {
    // error-0600a: `select="3.5"` cannot satisfy `as="xs:integer"` — the DEFAULT itself is
    // ill-typed, not a value some caller supplied, so this is XTTE0600 rather than the generic
    // XTTE0590 a bad *supplied* value would raise.
    let xsl = stylesheet_v20(
        r#"<xsl:template match="/"><xsl:apply-templates/></xsl:template>
           <xsl:template match="doc">
             <xsl:param name="p" as="xs:integer" select="3.5"/>
             <out><xsl:value-of select="$p"/></out>
           </xsl:template>"#,
    );
    let error = run_with(&xsl, "<doc/>").expect_err("3.5 does not satisfy xs:integer");
    assert_eq!(error.code(), Some("XTTE0600"));
}

#[test]
fn a_caller_supplied_value_that_violates_the_type_stays_xtte0590() {
    // Regression guard: the *other* branch — an actually supplied value that mismatches — must
    // keep its original code.
    let xsl = stylesheet_v20(
        r#"<xsl:template match="/">
             <xsl:call-template name="callee">
               <xsl:with-param name="p" select="'not a number'"/>
             </xsl:call-template>
           </xsl:template>
           <xsl:template name="callee">
             <xsl:param name="p" as="xs:integer" select="1"/>
             <out><xsl:value-of select="$p"/></out>
           </xsl:template>"#,
    );
    let error = run_with(&xsl, "<in/>").expect_err("a non-numeric string is not xs:integer");
    assert_eq!(error.code(), Some("XTTE0590"));
}

/// Compiles `xsl` and discards the result, for a test that only cares whether compilation
/// itself succeeds or fails — most of this section's cases are *static* errors.
fn compile_only(xsl: &str) -> Result<(), XsltError> {
    Stylesheet::compile_str(xsl).map(|_| ())
}

#[test]
fn error_0690a_against_the_real_suite_file() {
    // error-0690a: not one of this item's three named cases, but the exact same root cause
    // (a `<xsl:call-template>` that does not supply a required, non-tunnel parameter) —
    // verified against the actual W3C fixture rather than only a paraphrase.
    let Ok(suites_dir) = std::env::var("OXIXML_XML_SUITES_DIR") else {
        return;
    };
    let path =
        std::path::Path::new(&suites_dir).join("xslt30-test/tests/misc/error/error-0690a.xsl");
    let Ok(xsl) = std::fs::read_to_string(&path) else {
        return;
    };
    let error = compile_only(&xsl).expect_err("error-0690a must be a static error");
    assert_eq!(error.code(), Some("XTSE0690"));
}

// === Fix 7: instruction-specific computed-name error codes =====================================

#[test]
fn xsl_attribute_with_a_malformed_computed_name_raises_xtde0850() {
    // error-0850a: `xsl:attribute`'s computed name is not a lexical QName — `xsl:element`'s
    // XTDE0820 must not leak into `xsl:attribute`'s own XTDE0850.
    let xsl = stylesheet_v20(
        r#"<xsl:param name="p" select="'#'"/>
           <xsl:template match="/">
             <out><xsl:attribute name="{concat('a', $p)}"/></out>
           </xsl:template>"#,
    );
    let error = run_with(&xsl, "<in/>").expect_err("`a#` is not a lexical QName");
    assert_eq!(error.code(), Some("XTDE0850"));
}

#[test]
fn xsl_attribute_with_an_unbound_computed_prefix_raises_xtde0860() {
    // error-0860a: same shape, for the unbound-prefix condition.
    let xsl = stylesheet_v20(
        r#"<xsl:param name="p" select="'ns'"/>
           <xsl:template match="/">
             <out><xsl:attribute name="{concat('your:', $p)}">value</xsl:attribute></out>
           </xsl:template>"#,
    );
    let error = run_with(&xsl, "<in/>").expect_err("the `your` prefix is not bound");
    assert_eq!(error.code(), Some("XTDE0860"));
}

#[test]
fn xsl_element_computed_name_errors_are_unaffected() {
    // Regression guard: `xsl:element` keeps XTDE0820/XTDE0830, unchanged by the fix above.
    let malformed = stylesheet_v20(
        r#"<xsl:template match="/"><out><xsl:element name="a#"/></out></xsl:template>"#,
    );
    assert_eq!(
        run_with(&malformed, "<in/>")
            .expect_err("`a#` is not a lexical QName")
            .code(),
        Some("XTDE0820")
    );
    let unbound = stylesheet_v20(
        r#"<xsl:template match="/"><out><xsl:element name="your:ns"/></out></xsl:template>"#,
    );
    assert_eq!(
        run_with(&unbound, "<in/>")
            .expect_err("the `your` prefix is not bound")
            .code(),
        Some("XTDE0830")
    );
}

// === Fix 9: fn:key()'s two-argument form — the genuinely-no-context-node case ===================
//
// (Fix 8 needed no code change here or in oxixml-fn — see the report. The other half of Fix 9,
// key-072's *path-step-rebound* context node, is a cross-crate limitation this mission's crate
// boundary does not reach; see the report for the traced mechanism.)

#[test]
fn key_with_no_context_node_at_all_is_xtde1270_not_xpty0004() {
    // error-1270a: `key('k', 'abc')` inside an `xsl:function` body, which has no context item
    // at all — this must be XTDE1270 (the same code a node whose tree is not document-rooted
    // already gets), not the generic XPTY0004 an ordinary argument-type mismatch would be.
    let xsl = stylesheet_v20(
        r#"<xsl:key name="k" match="*" use="'pqr'"/>
           <xsl:template match="/">
             <out><xsl:sequence select="my:f()" xmlns:my="http://my.com/"/></out>
           </xsl:template>
           <xsl:function name="my:f" xmlns:my="http://my.com/">
             <xsl:sequence select="key('k', 'abc')"/>
           </xsl:function>"#,
    );
    let error = run_with(&xsl, "<in/>").expect_err("key() with no context node is an error");
    assert_eq!(error.code(), Some("XTDE1270"));
}

// === Fix 10: xsl:analyze-string and zero-length-capable regexes ================================

#[test]
fn analyze_string_accepts_a_regex_that_can_match_zero_length() {
    // analyze-string-090b (shape): a `^[\t ]*$` line-anchored pattern with the `m` flag matches
    // blank/whitespace-only lines with a zero-length-capable pattern (an empty line matches
    // `[\t ]*` with zero repetitions); XSLT 3.0 must not reject it the way XSLT 2.0's
    // `XTDE1150` did — the transformation must simply run to completion.
    // (`$` in multiline mode matches *before* a line terminator, so the newlines themselves
    // stay attached to the surrounding non-matching text rather than the zero-length match.)
    check_v20(
        r#"<xsl:template match="/">
             <xsl:variable name="v" select="'x&#10;&#10;y'"/>
             <out><xsl:analyze-string select="$v" regex="^[\t ]*$" flags="m">
               <xsl:matching-substring><blank/></xsl:matching-substring>
               <xsl:non-matching-substring><line><xsl:value-of select="."/></line></xsl:non-matching-substring>
             </xsl:analyze-string></out>
           </xsl:template>"#,
        "<in/>",
        "<out><line>x\n</line><blank/><line>\ny</line></out>",
    );
}

#[test]
fn analyze_string_accepts_the_empty_pattern() {
    // analyze-string-091b: `regex=""` — the most extreme zero-length-only pattern — splits the
    // input into individual characters via `xsl:non-matching-substring`; per XSLT 3.0 §5.7.1,
    // separate `xsl:sequence`-selected strings written to constructed content are
    // space-separated, matching the suite's own `/out='a b c d e f'` (here with "abcdef").
    check_v20(
        r#"<xsl:template match="/">
             <out><xsl:analyze-string select="'abcdef'" regex="" flags="m">
               <xsl:non-matching-substring><xsl:sequence select="."/></xsl:non-matching-substring>
             </xsl:analyze-string></out>
           </xsl:template>"#,
        "<in/>",
        "<out>a b c d e f</out>",
    );
}

#[test]
fn analyze_string_accepts_a_csv_style_alternation_with_an_empty_branch() {
    // analyze-string-092 (shape): `(?:^|,)(?:"quoted"|(unquoted))` — the unquoted branch can
    // match zero characters for an empty CSV field, while the quoted branch matches real
    // content; both must work in the same scan.
    check_v20(
        r#"<xsl:template match="/">
             <out><xsl:analyze-string select="'a,,b'" regex="(?:^|,)([^,]*)">
               <xsl:matching-substring><col><xsl:value-of select="regex-group(1)"/></col></xsl:matching-substring>
             </xsl:analyze-string></out>
           </xsl:template>"#,
        "<in/>",
        "<out><col>a</col><col/><col>b</col></out>",
    );
}

#[test]
fn analyze_string_092_against_the_real_suite_file() {
    // The actual suite stylesheet and its more elaborate CSV-with-quoting regex, env-guarded.
    let Ok(suites_dir) = std::env::var("OXIXML_XML_SUITES_DIR") else {
        return;
    };
    let path = std::path::Path::new(&suites_dir)
        .join("xslt30-test/tests/insn/analyze-string/analyze-string-092.xsl");
    let Ok(xsl) = std::fs::read_to_string(&path) else {
        return;
    };
    let compiled = Stylesheet::compile_str(&xsl).unwrap_or_else(|error| panic!("compile: {error}"));
    let source = Tree::parse_str("<in/>").expect("trivial source parses");
    let parameters = oxixml_xslt::Parameters::new().with_initial_template("main");
    let result = compiled
        .transform_with(&source.root(), &parameters)
        .unwrap_or_else(|error| panic!("transform: {error}"));
    let text = result.into_string();
    assert!(text.contains("<col>Ten Thousand</col>"), "{text}");
    assert!(text.contains("<col>10000</col>"), "{text}");
    assert!(
        text.contains("<col/>") || text.contains("<col></col>"),
        "{text}"
    );
    assert!(text.contains("10,000"), "{text}");
    assert!(text.contains("10K"), "{text}");
}

#[test]
fn output_0710_against_the_real_suite_file() {
    // This started life as an honesty check: `xsl:map`/`xsl:map-entry` (this file's Fix 5) let
    // output-0710 build its five maps, but writing them to the principal result was still
    // `XTDE0450` — the *complex content construction* error — where the suite asks for
    // `SENR0001`, the *serialization* one. The conformance-gate mission closed that gap
    // (`Emitter::final_result`), so the case now reaches the code the catalogue names, and the
    // check has become a real assertion.
    let Ok(suites_dir) = std::env::var("OXIXML_XML_SUITES_DIR") else {
        return;
    };
    let path =
        std::path::Path::new(&suites_dir).join("xslt30-test/tests/decl/output/output-0710.xsl");
    let Ok(xsl) = std::fs::read_to_string(&path) else {
        return;
    };
    let compiled = Stylesheet::compile_str(&xsl).unwrap_or_else(|error| panic!("compile: {error}"));
    let source = Tree::parse_str("<in/>").expect("trivial source parses");
    let error = compiled
        .transform(&source.root())
        .expect_err("a map cannot be serialized");
    assert_eq!(error.code(), Some("SENR0001"));
}

#[test]
fn result_document_1006_against_the_real_suite_file() {
    // `<t:variable name="base-out" select="current-output-uri()"/>` at the top level — before
    // any `xsl:result-document` is open — is `()`, since this processor has no principal-output
    // URI to report. `href="{$base-out}"` on the *nested* `xsl:result-document` therefore reads
    // as `href=""`, which names the principal result destination. The enclosing template has
    // already written `<out>` there directly, so the two final result trees share the principal
    // URI: XSLT 3.0 §23.1's `XTDE1490`, which is one of the two codes the catalogue accepts.
    let Ok(suites_dir) = std::env::var("OXIXML_XML_SUITES_DIR") else {
        return;
    };
    let path = std::path::Path::new(&suites_dir)
        .join("xslt30-test/tests/insn/result-document/result-document-1006.xsl");
    let Ok(xsl) = std::fs::read_to_string(&path) else {
        return;
    };
    let compiled = Stylesheet::compile_str(&xsl).unwrap_or_else(|error| panic!("compile: {error}"));
    let source =
        Tree::parse_str("<!-- A required but useless input XML doc for a transformation --><doc/>")
            .expect("suite source parses");
    let error = compiled
        .transform(&source.root())
        .expect_err("two final result trees share the principal result URI");
    assert_eq!(error.code(), Some("XTDE1490"));
}
