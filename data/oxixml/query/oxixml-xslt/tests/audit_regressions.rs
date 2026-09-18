//! Regression tests for the audit findings against `oxixml-xslt`.
//!
//! * `xsl:key`'s `collation` and `composite` attributes were validated and stored but never
//!   read at run time, so the index was keyed on raw strings;
//! * `xsl:sort case-order` was inert under the processor's default (code point) collation,
//!   because it was only consulted for strings the collation already called equal;
//! * `xsl:number/@start-at` shifted the place marker with unchecked arithmetic;
//! * template instantiation and `xsl:include` chains both recursed on the native stack without
//!   measuring it, so ordinary inputs could abort the process instead of raising an error;
//! * `xsl:catch/@errors` resolved an unprefixed name into the `err:` namespace and did not
//!   recognize the `Q{uri}local` spelling, and `$err:code` lost its prefix;
//! * `element-available()` denied `xsl:try`, `xsl:catch` and `xsl:assert`;
//! * `xsl:global-context-item` was parsed and then discarded;
//! * the import-precedence consistency checks were quadratic in the declaration count;
//! * an `as` attribute that was not a sequence type at all was silently left unchecked.

use oxixml_infoset::Tree;
use oxixml_xsd::TimezoneOffset;
use oxixml_xslt::{MemoryResolver, Parameters, StaticOptions, Stylesheet, XsltError};
use std::sync::Arc;

const UCA_PRIMARY: &str = "http://www.w3.org/2013/collation/UCA?lang=en;strength=primary";

fn stylesheet(body: &str) -> String {
    format!(
        r#"<xsl:stylesheet version="3.0" xmlns:xsl="http://www.w3.org/1999/XSL/Transform"
                          xmlns:xs="http://www.w3.org/2001/XMLSchema"
                          exclude-result-prefixes="xs">
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

fn run_with_parameters(
    body: &str,
    source: &str,
    parameters: &Parameters,
) -> Result<String, XsltError> {
    let compiled = Stylesheet::compile_str(&stylesheet(body))?;
    let tree = Tree::parse_str(source)
        .map_err(|error| XsltError::dynamic("XTDE0000", error.message().to_owned()))?;
    Ok(compiled
        .transform_with(&tree.root(), parameters)?
        .into_string())
}

fn check(body: &str, source: &str, expected: &str) {
    match run(body, source) {
        Ok(actual) => assert_eq!(actual, expected, "stylesheet body: {body}"),
        Err(error) => panic!("{error} (stylesheet body: {body})"),
    }
}

// --- xsl:key collation -------------------------------------------------------------------------

#[test]
fn a_key_collation_decides_which_values_match() {
    check(
        &format!(
            r#"<xsl:key name="k" match="a" use="@id" collation="{UCA_PRIMARY}"/>
               <xsl:template match="/"><xsl:value-of select="count(key('k','a'))"/></xsl:template>"#
        ),
        "<r><a id='A'/><a id='a'/></r>",
        "2",
    );
}

#[test]
fn a_key_without_a_collation_still_matches_by_code_point() {
    check(
        r#"<xsl:key name="k" match="a" use="@id"/>
           <xsl:template match="/"><xsl:value-of select="count(key('k','a'))"/></xsl:template>"#,
        "<r><a id='A'/><a id='a'/></r>",
        "1",
    );
}

// --- xsl:key composite -------------------------------------------------------------------------

#[test]
fn a_composite_key_is_not_matched_by_one_of_its_components() {
    check(
        r#"<xsl:key name="k" match="a" use="(@p,@q)" composite="yes"/>
           <xsl:template match="/"><xsl:value-of select="count(key('k','a'))"/></xsl:template>"#,
        "<r><a p='a' q='b'/></r>",
        "0",
    );
}

#[test]
fn a_composite_key_matches_the_whole_sequence() {
    check(
        r#"<xsl:key name="k" match="a" use="(@p,@q)" composite="yes"/>
           <xsl:template match="/"><xsl:value-of select="count(key('k',('a','b')))"/></xsl:template>"#,
        "<r><a p='a' q='b'/></r>",
        "1",
    );
}

#[test]
fn a_composite_key_distinguishes_the_component_order() {
    check(
        r#"<xsl:key name="k" match="a" use="(@p,@q)" composite="yes"/>
           <xsl:template match="/"><xsl:value-of select="count(key('k',('b','a')))"/></xsl:template>"#,
        "<r><a p='a' q='b'/></r>",
        "0",
    );
}

#[test]
fn a_non_composite_key_still_files_each_item_separately() {
    check(
        r#"<xsl:key name="k" match="a" use="(@p,@q)"/>
           <xsl:template match="/"><xsl:value-of
              select="count(key('k','a')), count(key('k','b')), count(key('k',('a','b')))"/></xsl:template>"#,
        "<r><a p='a' q='b'/></r>",
        "1 1 1",
    );
}

// --- xsl:sort case-order -----------------------------------------------------------------------

#[test]
fn case_order_lower_first_puts_lower_case_first_under_the_default_collation() {
    check(
        r#"<xsl:template match="/r"><xsl:for-each select="a">
             <xsl:sort select="@k" case-order="lower-first"/>
             <xsl:value-of select="@k"/></xsl:for-each></xsl:template>"#,
        "<r><a k='A'/><a k='a'/></r>",
        "aA",
    );
}

#[test]
fn case_order_upper_first_puts_upper_case_first_under_the_default_collation() {
    check(
        r#"<xsl:template match="/r"><xsl:for-each select="a">
             <xsl:sort select="@k" case-order="upper-first"/>
             <xsl:value-of select="@k"/></xsl:for-each></xsl:template>"#,
        "<r><a k='a'/><a k='A'/></r>",
        "Aa",
    );
}

#[test]
fn case_order_keeps_distinct_letters_in_order() {
    check(
        r#"<xsl:template match="/r"><xsl:for-each select="a">
             <xsl:sort select="@k" case-order="lower-first"/>
             <xsl:value-of select="@k"/></xsl:for-each></xsl:template>"#,
        "<r><a k='B'/><a k='a'/><a k='b'/><a k='A'/></r>",
        "aAbB",
    );
}

#[test]
fn without_case_order_the_code_point_collation_is_unchanged() {
    check(
        r#"<xsl:template match="/r"><xsl:for-each select="a">
             <xsl:sort select="@k"/>
             <xsl:value-of select="@k"/></xsl:for-each></xsl:template>"#,
        "<r><a k='a'/><a k='B'/></r>",
        "Ba",
    );
}

// --- dynamic errors inside a pattern ------------------------------------------------------------

/// XSLT 3.0 §5.5.3 lets a processor treat a dynamic error raised while matching a pattern as a
/// non-match. Now that the XPath layer raises `FORG0001` for a non-numeric `xs:untypedAtomic`
/// operand of a numeric comparison — instead of silently substituting `NaN` — the pattern
/// `*[.=117]` would otherwise abort a transformation over any document holding non-numeric text.
#[test]
fn a_value_error_in_a_pattern_is_a_non_match() {
    check(
        r#"<xsl:template match="/"><out><xsl:apply-templates select="//*"/></out></xsl:template>
           <xsl:template match="*[.=117]">[<xsl:value-of select="."/>]</xsl:template>
           <xsl:template match="*"/>"#,
        "<r><a>qqe</a><b>117</b></r>",
        "<out>[117]</out>",
    );
}

/// An `XT…` error keeps its meaning: a circular `xsl:key` is still `XTDE0640`, not a non-match.
#[test]
fn a_stylesheet_error_in_a_pattern_is_still_reported() {
    let body = r#"<xsl:key name="k" match="a[key('k','x')]" use="@id"/>
                  <xsl:template match="/"><xsl:value-of select="count(key('k','x'))"/></xsl:template>"#;
    let error = run(body, "<r><a id='x'/></r>").expect_err("a circular key is XTDE0640");
    assert_eq!(error.code(), Some("XTDE0640"));
}

// --- implicit timezone in xsl:sort --------------------------------------------------------------
//
// F&O 3.1 §10.4.4 reads a date/time sort key with no timezone in the *dynamic context's*
// implicit timezone before comparing, exactly as the `lt` operator would — XSLT 3.0 §13.1
// defines an untyped sort key's ordering in terms of that same operator. `Parameters` has no way
// to set a non-UTC implicit timezone until this fix, so the test also exercises that new knob.

const SORT_BY_TYPED_DATE_TIME: &str = r#"
    <xsl:template match="/r"><xsl:for-each select="a">
      <xsl:sort select="xs:dateTime(@d)"/>
      <xsl:value-of select="@id"/>
    </xsl:for-each></xsl:template>"#;

const TWO_DATE_TIMES: &str =
    "<r><a id='X' d='2000-01-01T00:00:00'/><a id='Y' d='2000-01-01T01:00:00Z'/></r>";

#[test]
fn sort_by_a_date_time_key_compares_values_not_lexical_text() {
    // A denotes 1999-12-31T23:00:00Z and B denotes 1999-12-31T23:30:00Z, so A is the *earlier*
    // instant and must sort first. Lexically, though, "1999-12-31T23:30:00Z" < "2000-01-01T…" —
    // comparing the raw text would put B first instead, so this pair only sorts correctly when
    // the key is the typed `xs:dateTime` value, not its spelling.
    check(
        SORT_BY_TYPED_DATE_TIME,
        "<r><a id='A' d='2000-01-01T00:00:00+01:00'/><a id='B' d='1999-12-31T23:30:00Z'/></r>",
        "AB",
    );
}

#[test]
fn sort_by_a_date_time_key_reads_a_floating_value_in_the_context_implicit_timezone() {
    // Under -05:00 the floating value X reads as 2000-01-01T05:00:00Z, four hours *after* Y's
    // 2000-01-01T01:00:00Z, so the order reverses relative to the UTC-default case above.
    let parameters =
        Parameters::new().with_implicit_timezone(TimezoneOffset::new(-300).expect("valid offset"));
    let actual = run_with_parameters(SORT_BY_TYPED_DATE_TIME, TWO_DATE_TIMES, &parameters)
        .expect("the transformation succeeds");
    assert_eq!(
        actual, "YX",
        "xsl:sort must use the context's implicit timezone, not hardcode UTC"
    );
}

// --- xsl:number/@start-at arithmetic -----------------------------------------------------------

/// `start-at` shifts the place marker by `value + start - 1`. Both ends of the integer range
/// reach that expression straight from the source document, so it is computed with checked
/// arithmetic: the largest representable start reports the condition instead of wrapping, and
/// the smallest one — which is its own negation — must still print a single minus sign.
#[test]
fn a_start_at_that_leaves_the_integer_range_is_a_dynamic_error() {
    let body = r#"<xsl:template match="/"><xsl:for-each select="r/a">
                    <xsl:number level="single" count="a" start-at="9223372036854775807"/>
                  </xsl:for-each></xsl:template>"#;
    let error = run(body, "<r><a/><a/></r>").expect_err("the shift leaves the integer range");
    assert_eq!(error.code(), Some("XTDE0980"), "{error}");
}

#[test]
fn the_smallest_start_at_prints_one_minus_sign() {
    let body = r#"<xsl:template match="/"><xsl:for-each select="r/a">
                    <xsl:number level="single" count="a" start-at="-9223372036854775808"/>
                  </xsl:for-each></xsl:template>"#;
    let numbered = run(body, "<r><a/></r>").expect("the shift stays in range");
    assert_eq!(numbered, "-9223372036854775808");
}

#[test]
fn the_smallest_start_at_spells_out_without_panicking() {
    let body = r#"<xsl:template match="/"><xsl:for-each select="r/a">
                    <xsl:number level="single" count="a" start-at="-9223372036854775808" format="w"/>
                  </xsl:for-each></xsl:template>"#;
    let spelled = run(body, "<r><a/></r>").expect("the shift stays in range");
    assert!(spelled.starts_with("minus nine billion"), "{spelled}");
}

// --- the native stack under recursive template instantiation -----------------------------------

/// Runs `body` on a thread with the stack `std::thread::spawn` gives by default, which is what a
/// thread pool or an async executor's worker hands a library.
fn on_an_ordinary_thread<T: Send + 'static>(body: impl FnOnce() -> T + Send + 'static) -> T {
    std::thread::Builder::new()
        .stack_size(oxixml_xslt::DEFAULT_STACK_SIZE)
        .spawn(body)
        .expect("a worker thread")
        .join()
        .expect("the worker thread did not panic")
}

/// A recursive named template must report `XTDE0640` rather than run the thread out of stack:
/// exhausting it aborts the process, which no caller can catch. The stylesheet below is the
/// shape that costs real stack — the recursive call's result is captured in a variable, so it
/// cannot be a tail call — and it recurses far past what any thread could carry.
#[test]
fn recursive_template_instantiation_stops_before_the_stack_does() {
    let error = on_an_ordinary_thread(|| {
        let xsl = stylesheet(
            r#"<xsl:template name="main">
                 <xsl:call-template name="rec">
                   <xsl:with-param name="n" select="100000"/>
                 </xsl:call-template>
               </xsl:template>
               <xsl:template name="rec">
                 <xsl:param name="n"/>
                 <xsl:if test="$n gt 0">
                   <xsl:variable name="inner">
                     <xsl:call-template name="rec">
                       <xsl:with-param name="n" select="$n - 1"/>
                     </xsl:call-template>
                   </xsl:variable>
                   <xsl:value-of select="$inner"/>
                 </xsl:if>
               </xsl:template>"#,
        );
        let compiled = Stylesheet::compile_str(&xsl).expect("the stylesheet compiles");
        compiled
            .transform_optional(None, &Parameters::new().with_initial_template("main"))
            .err()
            .map(|error| error.code().unwrap_or_default().to_owned())
    });
    assert_eq!(error.as_deref(), Some("XTDE0640"));
}

// --- xsl:include chains ------------------------------------------------------------------------

/// A chain of stylesheet modules, each including the next, is an ordinary way to layer a large
/// stylesheet and nothing bounds its length — so expanding one must not cost a native stack
/// frame per module. Two thousand of them are folded on an ordinary thread's stack.
#[test]
fn a_long_include_chain_expands_without_recursing() {
    let compiled = on_an_ordinary_thread(|| {
        const LENGTH: usize = 2_000;
        let mut resolver = MemoryResolver::new();
        for index in 0..LENGTH {
            let body = if index + 1 == LENGTH {
                r#"<xsl:template name="deep"><out/></xsl:template>"#.to_owned()
            } else {
                format!(r#"<xsl:include href="urn:module:{}"/>"#, index + 1)
            };
            resolver = resolver.with_document(
                format!("urn:module:{index}"),
                &format!(
                    r#"<xsl:stylesheet version="3.0"
                          xmlns:xsl="http://www.w3.org/1999/XSL/Transform">{body}</xsl:stylesheet>"#
                ),
            );
        }
        let options = StaticOptions::new()
            .with_resolver(Arc::new(resolver))
            .with_base_uri("urn:principal");
        Stylesheet::compile_str_with(
            r#"<xsl:stylesheet version="3.0" xmlns:xsl="http://www.w3.org/1999/XSL/Transform">
                 <xsl:output omit-xml-declaration="yes"/>
                 <xsl:include href="urn:module:0"/>
               </xsl:stylesheet>"#,
            options,
        )
        .map(|compiled| {
            compiled
                .transform_optional(None, &Parameters::new().with_initial_template("deep"))
                .map(|transformation| transformation.into_string())
        })
    });
    let text = compiled
        .expect("a two-thousand-module chain compiles")
        .expect("and runs");
    assert_eq!(text, "<out/>");
}

/// The chain is still checked for cycles: a module that includes itself is `XTSE0180`.
#[test]
fn a_module_that_includes_itself_is_still_rejected() {
    let resolver = MemoryResolver::new().with_document(
        "urn:loop",
        r#"<xsl:stylesheet version="3.0" xmlns:xsl="http://www.w3.org/1999/XSL/Transform">
             <xsl:include href="urn:loop"/>
           </xsl:stylesheet>"#,
    );
    let options = StaticOptions::new()
        .with_resolver(Arc::new(resolver))
        .with_base_uri("urn:principal");
    let error = Stylesheet::compile_str_with(
        r#"<xsl:stylesheet version="3.0" xmlns:xsl="http://www.w3.org/1999/XSL/Transform">
             <xsl:include href="urn:loop"/>
           </xsl:stylesheet>"#,
        options,
    )
    .expect_err("a module may not include itself");
    assert_eq!(error.code(), Some("XTSE0180"), "{error}");
}

/// mode-1505, mode-1905: a mode may be declared more than once at the import precedence of a
/// module, and stay legal when a single declaration at a higher precedence resolves the conflict
/// (XSLT 3.0 §6.6.1). The duplicate at the lower precedence is not `XTSE0545` on its own — which
/// means the count cannot be judged until the whole pass is done, since the resolving
/// declaration follows the duplicates in declaration order.
#[test]
fn a_mode_redeclared_below_its_winning_precedence_is_not_a_conflict() {
    let resolver = MemoryResolver::new().with_document(
        "urn:imported",
        r#"<xsl:stylesheet version="3.0" xmlns:xsl="http://www.w3.org/1999/XSL/Transform">
             <xsl:mode name="m" on-no-match="shallow-skip"/>
             <xsl:mode name="m" on-no-match="deep-skip"/>
           </xsl:stylesheet>"#,
    );
    let options = StaticOptions::new()
        .with_resolver(Arc::new(resolver))
        .with_base_uri("urn:principal");
    Stylesheet::compile_str_with(
        r#"<xsl:stylesheet version="3.0" xmlns:xsl="http://www.w3.org/1999/XSL/Transform">
             <xsl:import href="urn:imported"/>
             <xsl:mode name="m" on-no-match="text-only-copy"/>
           </xsl:stylesheet>"#,
        options,
    )
    .expect("a lower-precedence redeclaration resolved by a higher one compiles");
}

/// The guard it relaxes still fires: two `xsl:mode` declarations for one mode at its *winning*
/// import precedence remain `XTSE0545`.
#[test]
fn two_mode_declarations_at_the_winning_precedence_are_rejected() {
    let error = Stylesheet::compile_str(
        r#"<xsl:stylesheet version="3.0" xmlns:xsl="http://www.w3.org/1999/XSL/Transform">
             <xsl:mode name="m" on-no-match="shallow-skip"/>
             <xsl:mode name="m" on-no-match="deep-skip"/>
           </xsl:stylesheet>"#,
    )
    .expect_err("a mode declared twice at its highest precedence is a conflict");
    assert_eq!(error.code(), Some("XTSE0545"), "{error}");
}

// --- xsl:catch/@errors name tests --------------------------------------------------------------

/// try-036: `xpath-default-namespace` does not reach the error names of `xsl:catch`, and an
/// unprefixed one is in no namespace — so it does not catch `err:FOAR0001`.
#[test]
fn an_unprefixed_catch_name_is_in_no_namespace() {
    check(
        r#"<xsl:template match="/" xpath-default-namespace="http://www.w3.org/2005/xqt-errors">
             <out><xsl:try select="1 div 0">
               <xsl:catch errors="FOAR0001" select="'Should not be caught'"/>
               <xsl:catch errors="*" select="'OK'"/>
             </xsl:try></out>
           </xsl:template>"#,
        "<r/>",
        "<out>OK</out>",
    );
}

/// try-037: and the same unprefixed name does catch an error raised in no namespace.
#[test]
fn an_unprefixed_catch_name_catches_a_no_namespace_error() {
    check(
        r#"<xsl:template match="/">
             <out><xsl:try select="error(QName('', 'too-late'))">
               <xsl:catch errors="too-late" select="'Caught'"/>
               <xsl:catch errors="*" select="'Fail'"/>
             </xsl:try></out>
           </xsl:template>"#,
        "<r/>",
        "<out>Caught</out>",
    );
}

#[test]
fn a_catch_name_may_be_a_braced_uri_literal() {
    check(
        r#"<xsl:template match="/">
             <out><xsl:try select="1 div 0">
               <xsl:catch errors="Q{http://www.w3.org/2005/xqt-errors}FOAR0001" select="'Caught'"/>
               <xsl:catch errors="*" select="'Fail'"/>
             </xsl:try></out>
           </xsl:template>"#,
        "<r/>",
        "<out>Caught</out>",
    );
}

// --- the err: variables ------------------------------------------------------------------------

/// try-002: `{$err:code}` writes the prefixed lexical form of the error name.
#[test]
fn the_error_code_variable_keeps_its_prefix() {
    check(
        r#"<xsl:template match="/">
             <xsl:try select="1 div 0">
               <xsl:catch><out code="{$err:code}"/></xsl:catch>
             </xsl:try>
           </xsl:template>"#,
        "<r/>",
        r#"<out code="err:FOAR0001"/>"#,
    );
}

// --- element-available -------------------------------------------------------------------------

/// try-012: `xsl:try`, `xsl:catch` and `xsl:assert` are implemented, so the availability table
/// has to say so.
#[test]
fn the_implemented_instructions_are_reported_as_available() {
    check(
        r#"<xsl:template match="/">
             <out try="{element-available('xsl:try')}"
                  catch="{element-available('xsl:catch')}"
                  assert="{element-available('xsl:assert')}"/>
           </xsl:template>"#,
        "<r/>",
        r#"<out try="true" catch="true" assert="true"/>"#,
    );
}

// --- xsl:global-context-item -------------------------------------------------------------------

#[test]
fn a_required_global_context_item_that_is_not_supplied_is_an_error() {
    let xsl = stylesheet(
        r#"<xsl:global-context-item use="required"/>
           <xsl:template name="main"><out/></xsl:template>"#,
    );
    let compiled = Stylesheet::compile_str(&xsl).expect("the stylesheet compiles");
    let error = compiled
        .transform_optional(None, &Parameters::new().with_initial_template("main"))
        .expect_err("no global context item was supplied");
    assert_eq!(error.code(), Some("XTDE3086"), "{error}");
}

#[test]
fn a_global_context_item_of_the_wrong_type_is_an_error() {
    let body = r#"<xsl:global-context-item use="required" as="element(nonesuch)"/>
                  <xsl:template match="/"><out/></xsl:template>"#;
    let error = run(body, "<actual-root/>").expect_err("the document does not match the type");
    assert_eq!(error.code(), Some("XTTE0590"), "{error}");
}

#[test]
fn a_global_context_item_of_the_right_type_is_accepted() {
    check(
        r#"<xsl:global-context-item use="required" as="document-node()"/>
           <xsl:template match="/"><out><xsl:value-of select="name(/*)"/></out></xsl:template>"#,
        "<actual-root/>",
        "<out>actual-root</out>",
    );
}

#[test]
fn an_absent_global_context_item_is_not_visible_to_a_global_variable() {
    let body = r#"<xsl:global-context-item use="absent"/>
                  <xsl:variable name="g" select="/doc"/>
                  <xsl:template match="/"><out><xsl:value-of select="count($g)"/></out></xsl:template>"#;
    let error = run(body, "<doc/>").expect_err("the global variable has no context item");
    assert_eq!(error.code(), Some("XPDY0002"), "{error}");
}

#[test]
fn an_as_attribute_with_use_absent_is_a_static_error() {
    let xsl = stylesheet(r#"<xsl:global-context-item use="absent" as="document-node()"/>"#);
    let error = Stylesheet::compile_str(&xsl).expect_err("`as` may not accompany use=\"absent\"");
    assert_eq!(error.code(), Some("XTSE3089"), "{error}");
}

#[test]
fn two_global_context_item_declarations_in_one_module_are_a_static_error() {
    let xsl = stylesheet(
        r#"<xsl:global-context-item use="required"/>
           <xsl:global-context-item use="required"/>"#,
    );
    let error = Stylesheet::compile_str(&xsl).expect_err("only one declaration is allowed");
    assert_eq!(error.code(), Some("XTSE3087"), "{error}");
}

// --- the import-precedence consistency checks --------------------------------------------------

/// The checks that ask whether the declarations winning on import precedence agree used to
/// rescan the whole list once per declaration, which made compiling a machine-generated
/// stylesheet quadratic in the number of declarations: four thousand agreeing `xsl:output`
/// attributes took half a minute. They now cost one pass.
#[test]
fn many_agreeing_declarations_compile_in_linear_time() {
    const COUNT: usize = 4_000;
    let mut body = String::new();
    for _ in 0..COUNT {
        body.push_str(r#"<xsl:output method="xml" indent="yes"/>"#);
        body.push_str(r#"<xsl:decimal-format name="d" decimal-separator="."/>"#);
    }
    body.push_str(r#"<xsl:template match="/"><out/></xsl:template>"#);
    let started = std::time::Instant::now();
    let compiled = Stylesheet::compile_str(&stylesheet(&body)).expect("the stylesheet compiles");
    let elapsed = started.elapsed();
    let tree = oxixml_infoset::Tree::parse_str("<r/>").expect("the source parses");
    assert_eq!(
        compiled
            .transform_to_string(&tree.root())
            .expect("the transformation runs")
            .trim_end(),
        "<out/>"
    );
    assert!(
        elapsed < std::time::Duration::from_secs(30),
        "compiling {COUNT} declarations took {elapsed:?}"
    );
}

// --- the `as` attribute ------------------------------------------------------------------------

#[test]
fn an_as_attribute_that_is_not_a_sequence_type_is_a_static_error() {
    for text in ["xs:integer(((", "as as as", "not a type at all"] {
        let xsl = stylesheet(&format!(
            r#"<xsl:template match="/">
                 <xsl:variable name="v" as="{text}" select="'hello'"/>
                 <out><xsl:value-of select="$v"/></out>
               </xsl:template>"#
        ));
        let error =
            Stylesheet::compile_str(&xsl).expect_err("`{text}` does not hold a sequence type");
        assert_eq!(error.phase(), oxixml_xslt::ErrorPhase::Static, "{error}");
    }
}

/// The same shape recurs in the checks that ask whether a *name* is declared twice at the
/// winning import precedence, and in the one for `xsl:mode`: both used to rescan everything
/// declared so far once per declaration.
#[test]
fn many_distinct_declarations_compile_in_linear_time() {
    const COUNT: usize = 8_000;
    let mut body = String::new();
    for index in 0..COUNT {
        body.push_str(&format!(
            r#"<xsl:template name="t{index}"><out/></xsl:template>"#
        ));
    }
    body.push_str(r#"<xsl:template match="/"><out/></xsl:template>"#);
    let started = std::time::Instant::now();
    Stylesheet::compile_str(&stylesheet(&body)).expect("the stylesheet compiles");
    let elapsed = started.elapsed();
    assert!(
        elapsed < std::time::Duration::from_secs(30),
        "compiling {COUNT} declarations took {elapsed:?}"
    );
}

/// The `xsl:mode` check has the same shape.
#[test]
fn many_modes_compile_in_linear_time() {
    const COUNT: usize = 4_000;
    let mut body = String::new();
    for index in 0..COUNT {
        body.push_str(&format!(r#"<xsl:mode name="m{index}"/>"#));
    }
    body.push_str(r##"<xsl:template match="/" mode="#all"><out/></xsl:template>"##);
    let started = std::time::Instant::now();
    Stylesheet::compile_str(&stylesheet(&body)).expect("the stylesheet compiles");
    let elapsed = started.elapsed();
    assert!(
        elapsed < std::time::Duration::from_secs(30),
        "compiling {COUNT} modes took {elapsed:?}"
    );
}

/// `use-character-maps` builds a graph of names, and checking it for cycles used to recurse once
/// per link with no memory of what it had already proved acyclic. A chain of two thousand maps
/// is checked on an ordinary thread's stack.
#[test]
fn a_long_character_map_chain_is_checked_without_recursing() {
    const LENGTH: usize = 2_000;
    let compiled = on_an_ordinary_thread(|| {
        let mut body = String::new();
        for index in 0..LENGTH {
            let uses = if index + 1 == LENGTH {
                String::new()
            } else {
                format!(r#" use-character-maps="c{}""#, index + 1)
            };
            body.push_str(&format!(
                r#"<xsl:character-map name="c{index}"{uses}>
                     <xsl:output-character character="{}" string="[{index}]"/>
                   </xsl:character-map>"#,
                char::from(b'a' + u8::try_from(index % 26).unwrap_or(0))
            ));
        }
        body.push_str(
            r#"<xsl:output use-character-maps="c0"/>
               <xsl:template match="/"><out>a</out></xsl:template>"#,
        );
        Stylesheet::compile_str(&stylesheet(&body)).map(|compiled| {
            let tree = oxixml_infoset::Tree::parse_str("<r/>").expect("the source parses");
            compiled
                .transform_to_string(&tree.root())
                .map(|text| text.trim_end().to_owned())
        })
    });
    let text = compiled
        .expect("a two-thousand-map chain compiles")
        .expect("and runs");
    assert_eq!(text, "<out>[0]</out>");
}

#[test]
fn a_character_map_that_uses_itself_is_still_rejected() {
    let body = r#"<xsl:character-map name="a" use-character-maps="b">
                    <xsl:output-character character="x" string="X"/>
                  </xsl:character-map>
                  <xsl:character-map name="b" use-character-maps="a">
                    <xsl:output-character character="y" string="Y"/>
                  </xsl:character-map>
                  <xsl:template match="/"><out/></xsl:template>"#;
    let error =
        Stylesheet::compile_str(&stylesheet(body)).expect_err("the two maps reference each other");
    assert_eq!(error.code(), Some("XTSE1600"), "{error}");
}
