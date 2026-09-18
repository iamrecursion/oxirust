//! Regression tests for the bare-`.` predicate pattern, the `document('')` self-reference,
//! entry-point template parameters and the evaluation stack the engine declares.
//!
//! Every test reproduces one W3C `xslt30-test` case with a minimal inline stylesheet, so each fix
//! is provable red-then-green without the separate `oxixml-testsuite` harness. The case
//! identifiers in the comments are the ones that harness reports.

use oxixml_infoset::{AtomicValue, ExpandedName, Tree};
use oxixml_xslt::{MemoryResolver, Parameters, StaticOptions, Stylesheet, XsltError};
use std::sync::Arc;

/// The stylesheet URI every self-reference test compiles under, so that `document('')` has a base
/// URI to resolve the empty reference against.
const PRINCIPAL_URI: &str = "http://example.com/style/principal.xsl";

fn run_source(xsl: &str, source: &str) -> Result<String, XsltError> {
    let compiled = Stylesheet::compile_str(xsl)?;
    let tree = Tree::parse_str(source)
        .map_err(|error| XsltError::dynamic("XTDE0000", error.message().to_owned()))?;
    compiled.transform_to_string(&tree.root())
}

/// Compiles `xsl` as the module named by `PRINCIPAL_URI`, with `modules` registered as the other
/// documents (and stylesheet modules) the resolver serves.
fn run_module_source(
    xsl: &str,
    modules: &[(&str, &str)],
    source: &str,
) -> Result<String, XsltError> {
    let mut resolver = MemoryResolver::new().with_document(PRINCIPAL_URI, xsl);
    for (uri, text) in modules {
        resolver = resolver.with_document(*uri, text);
    }
    let options = StaticOptions::new()
        .with_base_uri(PRINCIPAL_URI)
        .with_resolver(Arc::new(resolver));
    let compiled = Stylesheet::compile_str_with(xsl, options)?;
    let tree = Tree::parse_str(source)
        .map_err(|error| XsltError::dynamic("XTDE0000", error.message().to_owned()))?;
    compiled.transform_to_string(&tree.root())
}

/// The transformation's text, or a readable stand-in for the error it raised — `XsltError` is
/// not comparable, and a failed run should say what went wrong rather than only that it failed.
fn text(outcome: Result<String, XsltError>) -> String {
    match outcome {
        Ok(text) => text,
        Err(error) => format!(
            "error {}: {}",
            error.code().unwrap_or("no code"),
            error.message()
        ),
    }
}

fn error_code(outcome: Result<String, XsltError>) -> String {
    match outcome {
        Ok(text) => format!("no error, produced `{text}`"),
        Err(error) => error.code().unwrap_or("no code").to_owned(),
    }
}

// --- the bare `.` predicate pattern ------------------------------------------------------

/// `doe-0180`: a template rule written `match="."` is a predicate pattern with no predicates, and
/// XSLT 3.0 §5.5.2 spells `PredicatePattern ::= "." PredicateList` with `PredicateList` empty —
/// so it matches an atomic value, which is what `<xsl:apply-templates select="'&lt;'"/>` supplies.
#[test]
fn a_bare_dot_pattern_matches_an_atomic_value() {
    let xsl = r#"<xsl:stylesheet version="3.0" xmlns:xsl="http://www.w3.org/1999/XSL/Transform">
          <xsl:output omit-xml-declaration="yes"/>
          <xsl:template match="/">
            <out><xsl:apply-templates select="'x'" mode="m"/></out>
          </xsl:template>
          <xsl:template match="." mode="m" as="text()">
            <xsl:value-of select="concat(., '!')"/>
          </xsl:template>
        </xsl:stylesheet>"#;
    assert_eq!(text(run_source(xsl, "<doc/>")), "<out>x!</out>");
}

/// `apply-templates-002`: with the bare `.` rule now matching, its body runs and
/// `<xsl:apply-templates/>` with no `select` asks for the children of an atomic value.
#[test]
fn applying_templates_to_the_children_of_an_atomic_value_is_a_type_error() {
    let xsl = r#"<xsl:stylesheet version="3.0" xmlns:xsl="http://www.w3.org/1999/XSL/Transform">
          <xsl:template name="main">
            <xsl:for-each select="1 to 5"><xsl:apply-templates select="."/></xsl:for-each>
          </xsl:template>
          <xsl:template match=".">
            <out><xsl:apply-templates/></out>
          </xsl:template>
        </xsl:stylesheet>"#;
    let compiled = Stylesheet::compile_str(xsl).expect("the stylesheet compiles");
    let outcome = compiled
        .transform_optional(None, &Parameters::new().with_initial_template("main"))
        .map(|transformation| transformation.as_str().to_owned());
    assert_eq!(error_code(outcome), "XTTE0510");
}

/// A bare `.` matches *every* item, so it is the least selective pattern the language has and
/// must not outrank a kind test: the W3C suite's own `snapshot-equivalent.xsl` dispatches nodes
/// with `match="node()"` while keeping `match="."` for "atomic values and function items".
#[test]
fn a_bare_dot_pattern_does_not_outrank_a_kind_test() {
    let xsl = r#"<xsl:stylesheet version="3.0" xmlns:xsl="http://www.w3.org/1999/XSL/Transform">
          <xsl:output omit-xml-declaration="yes"/>
          <xsl:template match="/">
            <out><xsl:apply-templates select="doc/e" mode="m"/></out>
          </xsl:template>
          <xsl:template match="." mode="m">any</xsl:template>
          <xsl:template match="node()" mode="m">node</xsl:template>
        </xsl:stylesheet>"#;
    assert_eq!(text(run_source(xsl, "<doc><e/></doc>")), "<out>node</out>");
}

/// A predicate pattern with predicates keeps the priority XSLT 3.0 §6.5 gives it, half a point
/// each, so it still wins over the bare `.` for the items it selects.
#[test]
fn a_predicate_pattern_outranks_the_bare_dot() {
    let xsl = r#"<xsl:stylesheet version="3.0" xmlns:xsl="http://www.w3.org/1999/XSL/Transform"
              xmlns:xs="http://www.w3.org/2001/XMLSchema" exclude-result-prefixes="xs">
          <xsl:output omit-xml-declaration="yes"/>
          <xsl:template name="main">
            <out><xsl:apply-templates select="'s', 3" mode="m"/></out>
          </xsl:template>
          <xsl:template match=".[. instance of xs:integer]" mode="m">int</xsl:template>
          <xsl:template match="." mode="m">other</xsl:template>
        </xsl:stylesheet>"#;
    let compiled = Stylesheet::compile_str(xsl).expect("the stylesheet compiles");
    let produced = compiled
        .transform_optional(None, &Parameters::new().with_initial_template("main"))
        .map(|transformation| transformation.as_str().to_owned());
    assert_eq!(text(produced), "<out>otherint</out>");
}

/// A pattern that only *starts* with a dot is an ordinary path pattern and keeps matching nodes
/// only: `..` selects the parent, which no atomic value has.
#[test]
fn a_dot_that_starts_a_path_is_not_a_predicate_pattern() {
    let xsl = r#"<xsl:stylesheet version="3.0" xmlns:xsl="http://www.w3.org/1999/XSL/Transform">
          <xsl:output omit-xml-declaration="yes"/>
          <xsl:template name="main">
            <out><xsl:apply-templates select="'s'" mode="m"/></out>
          </xsl:template>
          <xsl:template match="e/.." mode="m">path</xsl:template>
        </xsl:stylesheet>"#;
    let compiled = Stylesheet::compile_str(xsl).expect("the stylesheet compiles");
    let produced = compiled
        .transform_optional(None, &Parameters::new().with_initial_template("main"))
        .map(|transformation| transformation.as_str().to_owned());
    // No rule matches the string, so the built-in rule for an atomic value copies it.
    assert_eq!(text(produced), "<out>s</out>");
}

/// `document-1001`'s stylesheet carries two `ped:test` elements at the top level. XSLT 3.0 §3.6.2
/// makes them *user-defined data elements*: data the stylesheet holds for itself, ignored by the
/// processor and reachable only through `document('')`. Compiling one as a template body would
/// give the module a rule matching the root that suppresses every other rule.
#[test]
fn a_user_defined_data_element_declares_nothing() {
    let xsl = r#"<xsl:stylesheet version="3.0" xmlns:xsl="http://www.w3.org/1999/XSL/Transform"
              xmlns:ped="http://ped.com/" exclude-result-prefixes="ped">
          <xsl:output omit-xml-declaration="yes"/>
          <xsl:template match="test"><out>reached</out></xsl:template>
          <ped:data>never emitted</ped:data>
        </xsl:stylesheet>"#;
    assert_eq!(
        text(run_source(xsl, "<doc><test/></doc>")),
        "<out>reached</out>"
    );
}

/// §3.6.2: "It is a static error if the name of the element has a null namespace URI."
#[test]
fn a_top_level_data_element_must_be_in_a_namespace() {
    let xsl = r#"<xsl:stylesheet version="3.0" xmlns:xsl="http://www.w3.org/1999/XSL/Transform">
          <xsl:template match="test"><out>reached</out></xsl:template>
          <data>no namespace</data>
        </xsl:stylesheet>"#;
    assert_eq!(
        error_code(run_source(xsl, "<doc><test/></doc>")),
        "XTSE0130"
    );
}

/// A genuine simplified stylesheet — a module whose *document element* is a literal result
/// element — still compiles to one template matching the root.
#[test]
fn a_simplified_stylesheet_still_compiles() {
    let xsl = r#"<out xmlns:xsl="http://www.w3.org/1999/XSL/Transform" xsl:version="3.0"
              ><xsl:value-of select="/doc/test"/></out>"#;
    assert_eq!(
        text(run_source(xsl, "<doc><test>hi</test></doc>")),
        "<?xml version=\"1.0\" encoding=\"UTF-8\"?><out>hi</out>"
    );
}

// --- `document('')`, the stylesheet module itself ----------------------------------------

/// `document-1001` / `select-5601`: `doc('')` resolves the empty reference against the base URI
/// of the stylesheet module the call is written in, and that module has to be retrievable as a
/// document.
#[test]
fn doc_of_the_empty_uri_is_the_stylesheet_module() {
    let xsl = r#"<?xml version="1.0"?>
        <xsl:stylesheet version="3.0" xmlns:xsl="http://www.w3.org/1999/XSL/Transform"
              xmlns:ped="http://ped.com/" exclude-result-prefixes="ped">
          <xsl:output omit-xml-declaration="yes"/>
          <xsl:template match="test">
            <out><xsl:copy-of select="doc('')/xsl:stylesheet/ped:test[@attrib='yeha']"/></out>
          </xsl:template>
          <ped:test attrib="yeha">YEE-HA</ped:test>
          <ped:test attrib="haye">Wrong item</ped:test>
        </xsl:stylesheet>"#;
    assert_eq!(
        text(run_module_source(xsl, &[], "<doc><test>ERROR</test></doc>")),
        "<out><ped:test xmlns:ped=\"http://ped.com/\" \
         xmlns:xsl=\"http://www.w3.org/1999/XSL/Transform\" attrib=\"yeha\">YEE-HA</ped:test></out>"
    );
}

/// `document-1003` / `document-1004`: the empty reference is local to the module it is written
/// in, so a call inside an included module names *that* module, not the principal one.
#[test]
fn doc_of_the_empty_uri_in_an_included_module_is_that_module() {
    let included_uri = "http://example.com/style/included.xsl";
    let included = r#"<?xml version="1.0"?>
        <xsl:stylesheet version="3.0" xmlns:xsl="http://www.w3.org/1999/XSL/Transform"
              xmlns:ped="http://ped.com/" exclude-result-prefixes="ped">
          <xsl:template match="test">
            <out><xsl:copy-of select="doc('')/xsl:stylesheet/ped:test[@attrib='yeha']"/></out>
          </xsl:template>
          <ped:test attrib="yeha">YEE-HA</ped:test>
        </xsl:stylesheet>"#;
    let principal = r#"<?xml version="1.0"?>
        <xsl:stylesheet version="3.0" xmlns:xsl="http://www.w3.org/1999/XSL/Transform"
              xmlns:ped="http://ped.com/" exclude-result-prefixes="ped">
          <xsl:output omit-xml-declaration="yes"/>
          <xsl:include href="included.xsl"/>
          <xsl:template match="doc"><o><xsl:apply-templates/></o></xsl:template>
          <ped:test attrib="yeha">Oops</ped:test>
        </xsl:stylesheet>"#;
    assert_eq!(
        text(run_module_source(
            principal,
            &[(included_uri, included)],
            "<doc><test>ERROR</test></doc>"
        )),
        "<o><out><ped:test xmlns:ped=\"http://ped.com/\" \
         xmlns:xsl=\"http://www.w3.org/1999/XSL/Transform\" attrib=\"yeha\">YEE-HA</ped:test>\
         </out></o>"
    );
}

// --- entry-point template parameters ------------------------------------------------------

/// `initial-template-002`: the invocation supplies the initial template's own parameters, tunnel
/// ones included, and they are a different channel from a stylesheet parameter of the same name.
#[test]
fn the_initial_template_takes_the_parameters_the_invocation_supplies() {
    let xsl = r#"<xsl:stylesheet version="3.0" xmlns:xsl="http://www.w3.org/1999/XSL/Transform"
              xmlns:my="http://my.net/" exclude-result-prefixes="my">
          <xsl:output omit-xml-declaration="yes"/>
          <xsl:param name="a" select="12"/>
          <xsl:variable name="b" select="$a * 2"/>
          <xsl:template name="param1">
            <xsl:param name="a" required="yes"/>
            <xsl:param name="my:b" tunnel="yes"/>
            <a><xsl:sequence select="$a, $my:b, $b"/></a>
          </xsl:template>
        </xsl:stylesheet>"#;
    let compiled = Stylesheet::compile_str(xsl).expect("the stylesheet compiles");
    let parameters = Parameters::new()
        .with_initial_template("param1")
        .with_initial_template_parameters([
            (
                ExpandedName::new(None::<&str>, "a"),
                AtomicValue::integer(1234).into(),
                false,
            ),
            (
                ExpandedName::new(Some("http://my.net/"), "b"),
                AtomicValue::integer(999).into(),
                true,
            ),
        ]);
    let produced = compiled
        .transform_optional(None, &parameters)
        .map(|transformation| transformation.as_str().to_owned());
    assert_eq!(text(produced), "<a>1234 999 24</a>");
}

/// `initial-template-003`: a *stylesheet* parameter of the same name governs the global variable
/// that reads it and never reaches the template's own `xsl:param`.
#[test]
fn a_stylesheet_parameter_does_not_bind_an_initial_template_parameter() {
    let xsl = r#"<xsl:stylesheet version="3.0" xmlns:xsl="http://www.w3.org/1999/XSL/Transform"
              xmlns:my="http://my.net/" exclude-result-prefixes="my">
          <xsl:output omit-xml-declaration="yes"/>
          <xsl:param name="a" select="12"/>
          <xsl:variable name="b" select="$a * 2"/>
          <xsl:template name="param1">
            <xsl:param name="a" required="yes"/>
            <xsl:param name="my:b" tunnel="yes"/>
            <a><xsl:sequence select="$a, $my:b, $b"/></a>
          </xsl:template>
        </xsl:stylesheet>"#;
    let compiled = Stylesheet::compile_str(xsl).expect("the stylesheet compiles");
    let parameters = Parameters::new()
        .with_parameter("a", AtomicValue::integer(77))
        .with_initial_template("param1")
        .with_initial_template_parameters([
            (
                ExpandedName::new(None::<&str>, "a"),
                AtomicValue::integer(1234).into(),
                false,
            ),
            (
                ExpandedName::new(Some("http://my.net/"), "b"),
                AtomicValue::integer(999).into(),
                true,
            ),
        ]);
    let produced = compiled
        .transform_optional(None, &parameters)
        .map(|transformation| transformation.as_str().to_owned());
    assert_eq!(text(produced), "<a>1234 999 154</a>");
}

/// `initial-template-902`: a required parameter the invocation leaves out is `XTDE0700`, even
/// when a stylesheet parameter of that name has a value.
#[test]
fn a_required_initial_template_parameter_must_be_supplied() {
    let xsl = r#"<xsl:stylesheet version="3.0" xmlns:xsl="http://www.w3.org/1999/XSL/Transform"
              xmlns:my="http://my.net/" exclude-result-prefixes="my">
          <xsl:param name="a" select="12"/>
          <xsl:template name="param1">
            <xsl:param name="a" required="yes"/>
            <a><xsl:sequence select="$a"/></a>
          </xsl:template>
        </xsl:stylesheet>"#;
    let compiled = Stylesheet::compile_str(xsl).expect("the stylesheet compiles");
    let parameters = Parameters::new()
        .with_parameter("a", AtomicValue::integer(77))
        .with_initial_template("param1")
        .with_initial_template_parameters([(
            ExpandedName::new(Some("http://my.net/"), "b"),
            AtomicValue::integer(999).into(),
            true,
        )]);
    let produced = compiled
        .transform_optional(None, &parameters)
        .map(|transformation| transformation.as_str().to_owned());
    assert_eq!(error_code(produced), "XTDE0700");
}

// --- static rules the module validator enforces -------------------------------------------

/// `on-empty-003` / `on-empty-004`: `xsl:on-empty` decides what to write once the whole
/// containing sequence constructor has been evaluated, so an instruction may not follow it.
#[test]
fn an_instruction_after_on_empty_is_rejected() {
    let xsl = r#"<xsl:stylesheet version="3.0" xmlns:xsl="http://www.w3.org/1999/XSL/Transform">
          <xsl:template match="/">
            <out>
              <xsl:on-empty select="23"/>
              <xsl:copy-of select="/in/non-existent"/>
            </out>
          </xsl:template>
        </xsl:stylesheet>"#;
    assert_eq!(error_code(run_source(xsl, "<doc/>")), "XTSE0010");
}

/// `on-empty-007`: text that is not whitespace counts as a sibling just as an instruction does.
#[test]
fn significant_text_after_on_empty_is_rejected() {
    let xsl = r#"<xsl:stylesheet version="3.0" xmlns:xsl="http://www.w3.org/1999/XSL/Transform">
          <xsl:template match="/">
            <out>
              <a/>
              <xsl:on-empty select="23"/>
              49
            </out>
          </xsl:template>
        </xsl:stylesheet>"#;
    assert_eq!(error_code(run_source(xsl, "<doc/>")), "XTSE0010");
}

/// `on-non-empty-013`: "on-empty must come last" — even `xsl:on-non-empty` may not follow it.
#[test]
fn on_non_empty_after_on_empty_is_rejected() {
    let xsl = r#"<xsl:stylesheet version="3.0" xmlns:xsl="http://www.w3.org/1999/XSL/Transform">
          <xsl:template match="/">
            <out>
              <xsl:on-empty>EMPTY</xsl:on-empty>
              <xsl:on-non-empty>NON-EMPTY</xsl:on-non-empty>
            </out>
          </xsl:template>
        </xsl:stylesheet>"#;
    assert_eq!(error_code(run_source(xsl, "<doc/>")), "XTSE0010");
}

/// `on-non-empty-014`: the other order is what the language is for, and stays legal.
#[test]
fn on_non_empty_before_on_empty_is_accepted() {
    let xsl = r#"<xsl:stylesheet version="3.0" xmlns:xsl="http://www.w3.org/1999/XSL/Transform">
          <xsl:output omit-xml-declaration="yes"/>
          <xsl:template match="/">
            <out>
              <xsl:on-non-empty>NON-EMPTY</xsl:on-non-empty>
              <xsl:on-empty>EMPTY</xsl:on-empty>
            </out>
          </xsl:template>
        </xsl:stylesheet>"#;
    assert_eq!(text(run_source(xsl, "<doc/>")), "<out>EMPTY</out>");
}

/// `function-0117`: `override` is a synonym for `override-extension-function`, so the two may
/// not disagree.
#[test]
fn conflicting_override_attributes_are_rejected() {
    let xsl = r#"<xsl:stylesheet version="3.0" xmlns:xsl="http://www.w3.org/1999/XSL/Transform"
              xmlns:app="http://app.com/" exclude-result-prefixes="app">
          <xsl:function name="app:f" override-extension-function="1" override="no">
            <xsl:param name="doc"/>
            <xsl:sequence select="count($doc//*)"/>
          </xsl:function>
          <xsl:template match="/"><out><xsl:value-of select="app:f(/)"/></out></xsl:template>
        </xsl:stylesheet>"#;
    assert_eq!(error_code(run_source(xsl, "<doc/>")), "XTSE0020");
}

/// `function-0116`: the same two attributes saying the same thing stay legal.
#[test]
fn agreeing_override_attributes_are_accepted() {
    let xsl = r#"<xsl:stylesheet version="3.0" xmlns:xsl="http://www.w3.org/1999/XSL/Transform"
              xmlns:app="http://app.com/" exclude-result-prefixes="app">
          <xsl:output omit-xml-declaration="yes"/>
          <xsl:function name="app:f" override-extension-function="yes" override="true">
            <xsl:param name="doc"/>
            <xsl:sequence select="count($doc//*)"/>
          </xsl:function>
          <xsl:template match="/"><out><xsl:value-of select="app:f(/)"/></out></xsl:template>
        </xsl:stylesheet>"#;
    assert_eq!(text(run_source(xsl, "<doc><a/></doc>")), "<out>2</out>");
}

/// `function-0120b` and `error-0020j`: a stylesheet function's arguments are always supplied by
/// the call and never tunnelled, so `required="no"` and `tunnel="yes"` contradict the language.
/// `function-0119b` pins the other side: `required="yes"` only restates what already holds and
/// stays legal in XSLT 3.0.
#[test]
fn a_function_parameter_may_not_be_optional_or_tunnelled() {
    let function_with = |attribute: &str| {
        format!(
            r#"<xsl:stylesheet version="3.0" xmlns:xsl="http://www.w3.org/1999/XSL/Transform"
                  xmlns:app="http://app.com/" exclude-result-prefixes="app">
              <xsl:output omit-xml-declaration="yes"/>
              <xsl:function name="app:f">
                <xsl:param name="doc" {attribute}/>
                <xsl:sequence select="count($doc//*)"/>
              </xsl:function>
              <xsl:template match="/"><out><xsl:value-of select="app:f(/)"/></out></xsl:template>
            </xsl:stylesheet>"#
        )
    };
    for attribute in ["required=\"no\"", "tunnel=\"yes\""] {
        assert_eq!(
            error_code(run_source(&function_with(attribute), "<doc/>")),
            "XTSE0020",
            "`{attribute}` should be rejected"
        );
    }
    for attribute in ["required=\"yes\"", "tunnel=\"no\"", ""] {
        assert_eq!(
            text(run_source(&function_with(attribute), "<doc><a/></doc>")),
            "<out>2</out>",
            "`{attribute}` should be accepted"
        );
    }
}

// --- the evaluation stack the engine declares ---------------------------------------------

/// A stylesheet function recursing through the XPath evaluator is bounded by that engine's own
/// per-thread call ceiling, which defaults to 512. A caller that declared a large stack must get
/// a ceiling scaled to it rather than being stopped at the default.
#[test]
fn a_declared_stack_raises_the_xpath_call_ceiling_too() {
    let xsl = r#"<xsl:stylesheet version="3.0" xmlns:xsl="http://www.w3.org/1999/XSL/Transform"
              xmlns:xs="http://www.w3.org/2001/XMLSchema" xmlns:f="http://example.com/f"
              exclude-result-prefixes="xs f">
          <xsl:output omit-xml-declaration="yes"/>
          <xsl:function name="f:count" as="xs:integer">
            <xsl:param name="n" as="xs:integer"/>
            <xsl:sequence select="if ($n eq 0) then 0 else 1 + f:count($n - 1)"/>
          </xsl:function>
          <xsl:template name="main">
            <out><xsl:value-of select="f:count(700)"/></out>
          </xsl:template>
        </xsl:stylesheet>"#;
    let compiled = Stylesheet::compile_str(xsl).expect("the stylesheet compiles");
    // 700 nested calls is past `oxixml_xpath::DEFAULT_MAX_CALL_DEPTH`, the 512 a thread that
    // declared nothing is held to, and well inside what this thread really has.
    let parameters = Parameters::new()
        .with_initial_template("main")
        .with_recursion_limit(10_000)
        .with_stack_size(256 * 1024 * 1024);
    // The recursion runs on a thread with the stack the parameters declare, which is what makes
    // declaring it honest.
    let produced = std::thread::Builder::new()
        .stack_size(256 * 1024 * 1024)
        .spawn(move || {
            compiled
                .transform_optional(None, &parameters)
                .map(|transformation| transformation.as_str().to_owned())
        })
        .expect("the worker thread starts")
        .join()
        .expect("the worker thread finishes");
    assert_eq!(text(produced), "<out>700</out>");
}
