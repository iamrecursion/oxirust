//! Regression tests for the clusters the full W3C `xslt30-test` catalogue still reported once
//! every test set in it was being measured: ID typing surviving whitespace stripping, XPath
//! comments inside a text value template, the signatures a stylesheet function publishes to the
//! static context, extension-function fallback, and the base output URI a transformation writes
//! its principal result to.
//!
//! Every test reproduces one catalogue case with a minimal inline stylesheet, so each fix is
//! provable red-then-green without the separate `oxixml-testsuite` harness. The case identifiers
//! in the comments are the ones that harness reports.

use oxixml_infoset::Tree;
use oxixml_xslt::{MemoryResolver, Parameters, StaticOptions, Stylesheet, XsltError};
use std::sync::Arc;

/// Compiles `xsl` and runs it over `source`, returning the serialized result.
fn run(xsl: &str, source: &str) -> Result<String, XsltError> {
    let compiled = Stylesheet::compile_str(xsl)?;
    let tree = Tree::parse_str(source)
        .map_err(|error| XsltError::dynamic("XTDE0000", error.message().to_owned()))?;
    compiled.transform_to_string(&tree.root())
}

/// Compiles `xsl` and runs it from the named initial template.
fn run_named(xsl: &str, template: &str) -> Result<String, XsltError> {
    let compiled = Stylesheet::compile_str(xsl)?;
    let tree = Tree::parse_str("<doc/>")
        .map_err(|error| XsltError::dynamic("XTDE0000", error.message().to_owned()))?;
    compiled
        .transform_with(
            &tree.root(),
            &Parameters::new().with_initial_template(template),
        )
        .map(oxixml_xslt::Transformation::into_string)
}

/// The error code a compilation or a transformation raised, or `None` when it succeeded.
fn error_code(outcome: &Result<String, XsltError>) -> Option<String> {
    outcome
        .as_ref()
        .err()
        .and_then(|error| error.code().map(str::to_owned))
}

// --- ID typing across a whitespace-stripped tree (XSLT 3.0 §4.4) ----------------------------

/// A source document whose internal subset types `id` as an `ID` attribute, and whose element
/// content carries whitespace `xsl:strip-space` removes.
const ID_TYPED_SOURCE: &str = r#"<?xml version="1.0"?>
<!DOCTYPE doc [
  <!ELEMENT doc (a*)>
  <!ELEMENT a (b*)>
  <!ELEMENT b (#PCDATA)>
  <!ATTLIST a id ID #IMPLIED>
  <!ATTLIST b id ID #IMPLIED>
]>
<doc>
  <a id="id1"><b id="id2">two</b></a>
</doc>"#;

/// `id-028` … `id-031`: an `id(…)` pattern must still select after `xsl:strip-space` has run.
///
/// Stripping re-materializes the source document through this crate's own builder, and the
/// builder used to publish `[is-id]` only for `xml:id` — so every DTD-declared identifier was
/// lost on the way in and `id('id2')` selected nothing.
#[test]
fn an_id_pattern_matches_in_a_whitespace_stripped_tree() {
    let xsl = r#"<xsl:stylesheet xmlns:xsl="http://www.w3.org/1999/XSL/Transform" version="3.0">
      <xsl:strip-space elements="doc a"/>
      <xsl:template match="/"><out><xsl:apply-templates select="doc/a//text()"/></out></xsl:template>
      <xsl:template match="id('id2')/text()"><hit><xsl:value-of select="."/></hit></xsl:template>
      <xsl:template match="text()"><miss><xsl:value-of select="."/></miss></xsl:template>
    </xsl:stylesheet>"#;
    assert_eq!(
        run(xsl, ID_TYPED_SOURCE).expect("transform"),
        "<?xml version=\"1.0\" encoding=\"UTF-8\"?><out><hit>two</hit></out>"
    );
}

/// The same tree answers `fn:id` itself, not only the pattern form.
#[test]
fn fn_id_answers_in_a_whitespace_stripped_tree() {
    let xsl = r#"<xsl:stylesheet xmlns:xsl="http://www.w3.org/1999/XSL/Transform" version="3.0">
      <xsl:strip-space elements="doc a"/>
      <xsl:template match="/"><out><xsl:value-of select="id('id2')/name()"/></out></xsl:template>
    </xsl:stylesheet>"#;
    assert_eq!(
        run(xsl, ID_TYPED_SOURCE).expect("transform"),
        "<?xml version=\"1.0\" encoding=\"UTF-8\"?><out>b</out>"
    );
}

/// A tree the stylesheet *constructs* stays untyped: only `xml:id` makes an attribute an ID
/// there, which is what `key-077` relies on and what keeps `fn:id` from answering out of an
/// ordinary `id` attribute in a temporary tree.
#[test]
fn a_constructed_attribute_named_id_is_not_typed() {
    let xsl = r#"<xsl:stylesheet xmlns:xsl="http://www.w3.org/1999/XSL/Transform" version="3.0">
      <xsl:template match="/">
        <xsl:variable name="t"><e id="a" xml:id="b"/></xsl:variable>
        <out><xsl:value-of select="count(id('a', $t)), count(id('b', $t))"/></out>
      </xsl:template>
    </xsl:stylesheet>"#;
    assert_eq!(
        run(xsl, "<doc/>").expect("transform"),
        "<?xml version=\"1.0\" encoding=\"UTF-8\"?><out>0 1</out>"
    );
}

// --- XPath comments inside a value template (XSLT 3.0 §5.5) ---------------------------------

/// `cvt-023`: a `}` inside an XPath comment does not end the embedded expression.
#[test]
fn a_closing_brace_inside_an_xpath_comment_is_not_a_terminator() {
    let xsl = r#"<xsl:stylesheet xmlns:xsl="http://www.w3.org/1999/XSL/Transform" version="3.0">
      <xsl:template name="main">
        <out xsl:expand-text="yes">The Lord of the {(:implausible } :) 'Rings'}</out>
      </xsl:template>
    </xsl:stylesheet>"#;
    assert_eq!(
        run_named(xsl, "main").expect("transform"),
        "<?xml version=\"1.0\" encoding=\"UTF-8\"?><out>The Lord of the Rings</out>"
    );
}

/// The stray-`}` diagnostic is still raised where nothing hides it.
#[test]
fn a_stray_closing_brace_is_still_a_static_error() {
    let xsl = r#"<xsl:stylesheet xmlns:xsl="http://www.w3.org/1999/XSL/Transform" version="3.0">
      <xsl:template name="main"><out a="}"/></xsl:template>
    </xsl:stylesheet>"#;
    assert_eq!(
        error_code(&run_named(xsl, "main")).as_deref(),
        Some("XTSE0370")
    );
}

// --- extension instructions and xsl:fallback (XSLT 3.0 §3.7, §3.9) ---------------------------

/// `extension-functions-0106`: an `xsl:fallback` with no content is still a fallback — the
/// instruction it stands in for produces nothing rather than raising `XTDE1450`.
#[test]
fn an_empty_xsl_fallback_stands_in_for_an_extension_instruction() {
    let xsl = r#"<xsl:transform xmlns:xsl="http://www.w3.org/1999/XSL/Transform" version="3.0"
        extension-element-prefixes="ext" xmlns:ext="http://example.com/extension-unknown">
      <xsl:template match="doc"><out><ext:special><xsl:fallback/></ext:special></out></xsl:template>
    </xsl:transform>"#;
    assert_eq!(
        run(xsl, "<doc/>").expect("transform"),
        "<?xml version=\"1.0\" encoding=\"UTF-8\"?><out/>"
    );
}

/// An extension instruction with no `xsl:fallback` at all still raises `XTDE1450`.
#[test]
fn an_extension_instruction_without_a_fallback_still_raises() {
    let xsl = r#"<xsl:transform xmlns:xsl="http://www.w3.org/1999/XSL/Transform" version="3.0"
        extension-element-prefixes="ext" xmlns:ext="http://example.com/extension-unknown">
      <xsl:template match="doc"><out><ext:special/></out></xsl:template>
    </xsl:transform>"#;
    assert_eq!(error_code(&run(xsl, "<doc/>")).as_deref(), Some("XTDE1450"));
}

/// `extension-functions-0105`: a reserved namespace may not name an extension instruction.
#[test]
fn an_extension_instruction_in_a_reserved_namespace_is_rejected() {
    let xsl = r#"<xsl:transform xmlns:xsl="http://www.w3.org/1999/XSL/Transform" version="3.0"
        extension-element-prefixes="xs" xmlns:xs="http://www.w3.org/2001/XMLSchema">
      <xsl:template match="doc"><out><xs:special><xsl:fallback/></xs:special></out></xsl:template>
    </xsl:transform>"#;
    assert_eq!(error_code(&run(xsl, "<doc/>")).as_deref(), Some("XTSE0800"));
}

/// `extension-functions-0104`: the availability functions expand their argument against the
/// namespaces in scope for the expression, even in a `use-when` the compiler evaluates before
/// there is a transformation — so an undeclared prefix is `XTDE1400` there too.
#[test]
fn function_available_reports_an_undeclared_prefix_inside_use_when() {
    let xsl = r#"<t:transform xmlns:t="http://www.w3.org/1999/XSL/Transform" version="2.0">
      <t:template match="doc">
        <out><t:value-of select="'error'" use-when="function-available('test:foo')"/></out>
      </t:template>
    </t:transform>"#;
    assert_eq!(error_code(&run(xsl, "<doc/>")).as_deref(), Some("XTDE1400"));
}

/// A prefix that *is* in scope where the `use-when` is written resolves normally.
#[test]
fn function_available_resolves_a_declared_prefix_inside_use_when() {
    let xsl = r#"<t:transform xmlns:t="http://www.w3.org/1999/XSL/Transform"
        xmlns:test="http://example.com/test" version="2.0">
      <t:template match="doc">
        <out><t:value-of select="'error'" use-when="function-available('test:foo')"/></out>
      </t:template>
    </t:transform>"#;
    assert_eq!(
        run(xsl, "<doc/>").expect("transform"),
        "<?xml version=\"1.0\" encoding=\"UTF-8\"?><out xmlns:test=\"http://example.com/test\"/>"
    );
}

// --- the signature a stylesheet function publishes (XSLT 3.0 §10.3) --------------------------

/// `higher-order-functions-032` … `-034`: a reference to an `xsl:function` carries the parameter
/// and result types the declaration gives, so `instance of function(…) as …` judges more than
/// its arity.
#[test]
fn a_stylesheet_function_reference_carries_its_declared_types() {
    let xsl = r#"<xsl:stylesheet xmlns:xsl="http://www.w3.org/1999/XSL/Transform"
        xmlns:xs="http://www.w3.org/2001/XMLSchema" xmlns:local="http://local/" version="3.0">
      <xsl:function name="local:f" as="element(e)?">
        <xsl:param name="x" as="xs:long"/>
        <e x="{$x}"/>
      </xsl:function>
      <xsl:template name="main"><out><xsl:value-of select="
          local:f#1 instance of function(xs:long) as element(e),
          local:f#1 instance of function(xs:long) as element(e)?,
          local:f#1 instance of function(xs:long) as element(e)*,
          local:f#1 instance of function(xs:string) as element(e)?,
          local:f#1 instance of function(*)"/></out></xsl:template>
    </xsl:stylesheet>"#;
    assert_eq!(
        run_named(xsl, "main").expect("transform"),
        "<?xml version=\"1.0\" encoding=\"UTF-8\"?>\
         <out xmlns:local=\"http://local/\" xmlns:xs=\"http://www.w3.org/2001/XMLSchema\">\
         false true true false true</out>"
    );
}

/// A function that declares nothing publishes `item()*` on both sides rather than no signature
/// at all.
#[test]
fn an_undeclared_stylesheet_function_publishes_the_default_signature() {
    let xsl = r#"<xsl:stylesheet xmlns:xsl="http://www.w3.org/1999/XSL/Transform"
        xmlns:local="http://local/" version="3.0">
      <xsl:function name="local:g"><xsl:param name="x"/><xsl:sequence select="$x"/></xsl:function>
      <xsl:template name="main"><out><xsl:value-of select="
          local:g#1 instance of function(item()*) as item()*,
          local:g#1 instance of function(item()*) as xs:integer"/></out></xsl:template>
    </xsl:stylesheet>"#;
    assert_eq!(
        run_named(xsl, "main").expect("transform"),
        "<?xml version=\"1.0\" encoding=\"UTF-8\"?>\
         <out xmlns:local=\"http://local/\">true false</out>"
    );
}

/// `use-when-0428`, `use-when-0429`: `use-when` is part of the stylesheet's lexical processing,
/// so an error in one written inside an `xsl:fallback` that guards an instruction this processor
/// *does* implement is still reported, even though the content is never instantiated.
#[test]
fn use_when_inside_an_ignored_fallback_is_still_evaluated() {
    let unparseable = r#"<xsl:transform xmlns:xsl="http://www.w3.org/1999/XSL/Transform" version="2.0">
      <xsl:template match="/"><result><xsl:value-of>
        <xsl:text>text</xsl:text>
        <xsl:fallback><row xsl:use-when="invalid///xpath">never</row></xsl:fallback>
      </xsl:value-of></result></xsl:template>
    </xsl:transform>"#;
    assert_eq!(
        error_code(&run(unparseable, "<doc/>")).as_deref(),
        Some("XPST0003")
    );
    let contextless = r#"<xsl:transform xmlns:xsl="http://www.w3.org/1999/XSL/Transform" version="2.0">
      <xsl:template match="/"><result><xsl:value-of>
        <xsl:text>text</xsl:text>
        <xsl:fallback><out xsl:use-when="error">never</out></xsl:fallback>
      </xsl:value-of></result></xsl:template>
    </xsl:transform>"#;
    assert_eq!(
        error_code(&run(contextless, "<doc/>")).as_deref(),
        Some("XPDY0002")
    );
}

/// `static-008`: a static `xsl:param` with neither a `select` attribute nor content binds the
/// zero-length string, not the empty sequence.
#[test]
fn a_static_parameter_with_no_value_binds_the_empty_string() {
    let xsl = r#"<xsl:transform xmlns:xsl="http://www.w3.org/1999/XSL/Transform" version="3.0">
      <xsl:param name="p" static="yes"/>
      <xsl:template name="main" expand-text="yes"><out>{empty($p)}-{$p = ''}</out></xsl:template>
    </xsl:transform>"#;
    assert_eq!(
        run_named(xsl, "main").expect("transform"),
        "<?xml version=\"1.0\" encoding=\"UTF-8\"?><out>false-true</out>"
    );
}

/// `static-027`: a `static="yes"` variable keeps the value the compilation settled on, even
/// where an ordinary global variable of the same name is in scope at run time — the static
/// `$p` an imported module declares is 1, the run-time `$p` is 0, and the static `$q` is 11.
#[test]
fn a_static_variable_keeps_its_compiled_value() {
    let imported = r#"<xsl:transform xmlns:xsl="http://www.w3.org/1999/XSL/Transform" version="3.0">
       <xsl:variable name="p" static="yes" select="1"/>
    </xsl:transform>"#;
    let principal = r#"<xsl:transform xmlns:xsl="http://www.w3.org/1999/XSL/Transform" version="3.0">
       <xsl:import href="a.xsl"/>
       <xsl:variable name="p" static="no" select="0"/>
       <xsl:variable name="q" static="yes" select="$p+10"/>
       <xsl:template name="main"><result p="{$p}" q="{$q}"/></xsl:template>
    </xsl:transform>"#;
    let resolver = MemoryResolver::new().with_document("http://example.com/a.xsl", imported);
    let options = StaticOptions::new()
        .with_base_uri("http://example.com/principal.xsl")
        .with_resolver(Arc::new(resolver));
    let compiled = Stylesheet::compile_str_with(principal, options).expect("compile");
    let tree = Tree::parse_str("<doc/>").expect("a source document");
    let out = compiled
        .transform_with(
            &tree.root(),
            &Parameters::new().with_initial_template("main"),
        )
        .map(oxixml_xslt::Transformation::into_string)
        .expect("transform");
    assert_eq!(
        out,
        "<?xml version=\"1.0\" encoding=\"UTF-8\"?><result p=\"0\" q=\"11\"/>"
    );
}

/// `static-009` … `static-011`: the other half of §9.3's rule — a declaration that *does* carry
/// an `as` attribute defaults to the empty sequence, so `empty($p)` is true and arithmetic over
/// it raises nothing.
#[test]
fn a_static_parameter_with_a_declared_type_defaults_to_the_empty_sequence() {
    let xsl = r#"<xsl:transform xmlns:xsl="http://www.w3.org/1999/XSL/Transform"
        xmlns:xs="http://www.w3.org/2001/XMLSchema" version="3.0">
      <xsl:param name="p" static="yes" as="xs:integer?"/>
      <xsl:template name="main" expand-text="yes"><out>{empty($p)}-{10 * $p}-{$p = 'x'}</out></xsl:template>
    </xsl:transform>"#;
    assert_eq!(
        run_named(xsl, "main").expect("transform"),
        "<?xml version=\"1.0\" encoding=\"UTF-8\"?>\
         <out xmlns:xs=\"http://www.w3.org/2001/XMLSchema\">true--false</out>"
    );
}

// --- namespaces (XSLT 3.0 §5.1, §11.5.2) ----------------------------------------------------

/// `namespace-3109`, `namespace-3113`: an unprefixed `name` on `xsl:element` expands against the
/// default namespace in scope *where the instruction is written*, which an `xmlns=""` on the
/// instruction itself undeclares.
#[test]
fn an_undeclared_default_namespace_reaches_a_computed_element_name() {
    let xsl = r#"<xsl:stylesheet xmlns:xsl="http://www.w3.org/1999/XSL/Transform" version="3.0"
        xmlns="http://testguys.com/">
      <xsl:template match="/"><out><xsl:element name="foo" xmlns=""><yyy/></xsl:element></out></xsl:template>
    </xsl:stylesheet>"#;
    assert_eq!(
        run(xsl, "<doc/>").expect("transform"),
        "<?xml version=\"1.0\" encoding=\"UTF-8\"?>\
         <out xmlns=\"http://testguys.com/\"><foo xmlns=\"\"><yyy/></foo></out>"
    );
}

/// `namespace-2623`: an `xsl:attribute` whose `namespace` is the `xmlns` namespace is
/// `XTDE0865`; the same fault on `xsl:element` keeps `XTDE0835`.
#[test]
fn an_attribute_in_the_xmlns_namespace_is_reported_under_its_own_code() {
    let attribute = r#"<xsl:stylesheet xmlns:xsl="http://www.w3.org/1999/XSL/Transform" version="3.0">
      <xsl:template match="/"><out>
        <xsl:attribute name="a:foo" namespace="{'http://www.w3.org'}/2000/xmlns/">v</xsl:attribute>
      </out></xsl:template>
    </xsl:stylesheet>"#;
    assert_eq!(
        error_code(&run(attribute, "<doc/>")).as_deref(),
        Some("XTDE0865")
    );
    let element = r#"<xsl:stylesheet xmlns:xsl="http://www.w3.org/1999/XSL/Transform" version="3.0">
      <xsl:template match="/"><out>
        <xsl:element name="a:foo" namespace="{'http://www.w3.org'}/2000/xmlns/"/>
      </out></xsl:template>
    </xsl:stylesheet>"#;
    assert_eq!(
        error_code(&run(element, "<doc/>")).as_deref(),
        Some("XTDE0835")
    );
}

// --- forwards-compatible processing (XSLT 3.0 §3.9) ------------------------------------------

/// `forwards-005`: a top-level element whose effective version is a later one is ignored, even
/// when this processor recognizes the element and knows it is not a declaration.
#[test]
fn a_later_version_top_level_element_is_ignored() {
    let xsl = r#"<xsl:transform xmlns:xsl="http://www.w3.org/1999/XSL/Transform" version="3.0">
      <xsl:template name="main"><out>ok</out></xsl:template>
      <xsl:value-of select="$x" version="4.0"/>
    </xsl:transform>"#;
    assert_eq!(
        run_named(xsl, "main").expect("transform"),
        "<?xml version=\"1.0\" encoding=\"UTF-8\"?><out>ok</out>"
    );
}

/// `forwards-008`: such an element is ignored *whole* — its `use-when` belongs to the later
/// version too and is not evaluated, so an unknown function there is not an error.
#[test]
fn the_use_when_of_an_ignored_later_version_declaration_is_not_evaluated() {
    let xsl = r#"<xsl:stylesheet xmlns:xsl="http://www.w3.org/1999/XSL/Transform" version="4.0">
      <xsl:template name="main"><out>ok</out></xsl:template>
      <xsl:new-declaration use-when="fn:new-function()"/>
    </xsl:stylesheet>"#;
    assert_eq!(
        run_named(xsl, "main").expect("transform"),
        "<?xml version=\"1.0\" encoding=\"UTF-8\"?><out>ok</out>"
    );
}

// --- fn:snapshot over attribute and namespace nodes (XSLT 3.0 §21.2) -------------------------

/// `snapshot-0101e`: the snapshot of an attribute keeps the ancestor path, so the copy still has
/// the parent element it was taken from.
#[test]
fn the_snapshot_of_an_attribute_keeps_its_ancestors() {
    let xsl = r#"<xsl:stylesheet xmlns:xsl="http://www.w3.org/1999/XSL/Transform" version="3.0"
        expand-text="yes">
      <xsl:template match="/">
        <xsl:variable name="result" select="snapshot(//@a)"/>
        <out>
          <a>{count($result) eq 1}</a>
          <b>{$result[1] instance of attribute(a)}</b>
          <c>{exists($result[1]/parent::e)}</c>
          <d>{empty($result[1]/child::node())}</d>
        </out>
      </xsl:template>
    </xsl:stylesheet>"#;
    assert_eq!(
        run(xsl, "<doc><e a='1'/></doc>").expect("transform"),
        "<?xml version=\"1.0\" encoding=\"UTF-8\"?>\
         <out><a>true</a><b>true</b><c>true</c><d>true</d></out>"
    );
}

/// `snapshot-0103`, `snapshot-0104`: a parentless attribute or namespace node is an item of a
/// sequence like any other, both when it is constructed and when it is copied.
#[test]
fn a_parentless_attribute_or_namespace_node_can_be_copied() {
    let xsl = r#"<xsl:stylesheet xmlns:xsl="http://www.w3.org/1999/XSL/Transform" version="3.0"
        expand-text="yes">
      <xsl:variable name="orphans" as="node()*">
        <xsl:attribute name="a" select="1"/>
        <xsl:namespace name="n">http://www.example.com/"</xsl:namespace>
      </xsl:variable>
      <xsl:template match="/">
        <xsl:variable name="copies" select="$orphans!copy-of(.)"/>
        <out>
          <a>{count($copies)}</a>
          <b>{$copies[1] instance of attribute(a)}</b>
          <c>{$copies[2] instance of namespace-node()}</c>
          <d>{string($copies[2])}</d>
        </out>
      </xsl:template>
    </xsl:stylesheet>"#;
    assert_eq!(
        run(xsl, "<doc/>").expect("transform"),
        "<?xml version=\"1.0\" encoding=\"UTF-8\"?>\
         <out><a>2</a><b>true</b><c>true</c><d>http://www.example.com/\"</d></out>"
    );
}

// --- fn:transform (F&O 3.1 §16.4) -----------------------------------------------------------

/// `transform-002`: a stylesheet named by `stylesheet-location`, applied to an
/// `initial-match-selection` that is an atomic value, whose result is the `output` entry of the
/// map the function returns.
#[test]
fn fn_transform_runs_a_stylesheet_named_by_location() {
    let inner = r#"<xsl:stylesheet xmlns:xsl="http://www.w3.org/1999/XSL/Transform"
        xmlns:xs="http://www.w3.org/2001/XMLSchema" version="3.0" expand-text="yes">
      <xsl:template match=".[. instance of xs:integer]"><in>{17 + .}</in></xsl:template>
    </xsl:stylesheet>"#;
    let principal = r#"<xsl:stylesheet xmlns:xsl="http://www.w3.org/1999/XSL/Transform" version="3.0">
      <xsl:template name="main"><out><xsl:sequence select="
        transform(map{'stylesheet-location': 'inner.xsl', 'initial-match-selection': 42})?output"/></out>
      </xsl:template>
    </xsl:stylesheet>"#;
    let resolver = MemoryResolver::new().with_document("http://example.com/inner.xsl", inner);
    let options = StaticOptions::new()
        .with_base_uri("http://example.com/principal.xsl")
        .with_resolver(Arc::new(resolver));
    let compiled = Stylesheet::compile_str_with(principal, options).expect("compile");
    let tree = Tree::parse_str("<doc/>").expect("a source document");
    let out = compiled
        .transform_with(
            &tree.root(),
            &Parameters::new().with_initial_template("main"),
        )
        .map(oxixml_xslt::Transformation::into_string)
        .expect("transform");
    assert_eq!(
        out,
        "<?xml version=\"1.0\" encoding=\"UTF-8\"?>\
         <out><in xmlns:xs=\"http://www.w3.org/2001/XMLSchema\">59</in></out>"
    );
}

/// `transform-001`: a stylesheet the options name but that cannot be retrieved is `FOXT0002`.
#[test]
fn fn_transform_reports_a_stylesheet_it_cannot_retrieve() {
    let principal = r#"<xsl:stylesheet xmlns:xsl="http://www.w3.org/1999/XSL/Transform" version="3.0">
      <xsl:template name="main"><out><xsl:sequence select="
        transform(map{'stylesheet-location': 'nowhere.xsl', 'initial-match-selection': 42})?*"/></out>
      </xsl:template>
    </xsl:stylesheet>"#;
    let options = StaticOptions::new()
        .with_base_uri("http://example.com/principal.xsl")
        .with_resolver(Arc::new(MemoryResolver::new()));
    let compiled = Stylesheet::compile_str_with(principal, options).expect("compile");
    let tree = Tree::parse_str("<doc/>").expect("a source document");
    let error = compiled
        .transform_with(
            &tree.root(),
            &Parameters::new().with_initial_template("main"),
        )
        .expect_err("the stylesheet cannot be retrieved");
    assert_eq!(error.code(), Some("FOXT0002"));
}

/// `transform-008`: an `xsl:result-document` in the invoked transformation contributes an entry
/// keyed by the URI it wrote to, beside `output`.
#[test]
fn fn_transform_reports_secondary_results_by_uri() {
    let inner = r#"<xsl:stylesheet xmlns:xsl="http://www.w3.org/1999/XSL/Transform" version="3.0">
      <xsl:template as="element(a)" name="xsl:initial-template">
        <xsl:result-document href="http://www.example.com/out.xml"><out>479</out></xsl:result-document>
        <a>892</a>
      </xsl:template>
    </xsl:stylesheet>"#;
    let principal = r#"<xsl:stylesheet xmlns:xsl="http://www.w3.org/1999/XSL/Transform" version="3.0">
      <xsl:template name="main">
        <xsl:variable name="result" select="transform(map{
          'delivery-format': 'raw',
          'stylesheet-location': 'inner.xsl',
          'initial-template': QName('http://www.w3.org/1999/XSL/Transform', 'xsl:initial-template')})"/>
        <result>
          <primary><xsl:sequence select="$result?output"/></primary>
          <secondary><xsl:sequence select="$result('http://www.example.com/out.xml')"/></secondary>
        </result>
      </xsl:template>
    </xsl:stylesheet>"#;
    let resolver = MemoryResolver::new().with_document("http://example.com/inner.xsl", inner);
    let options = StaticOptions::new()
        .with_base_uri("http://example.com/principal.xsl")
        .with_resolver(Arc::new(resolver));
    let compiled = Stylesheet::compile_str_with(principal, options).expect("compile");
    let tree = Tree::parse_str("<doc/>").expect("a source document");
    let out = compiled
        .transform_with(
            &tree.root(),
            &Parameters::new().with_initial_template("main"),
        )
        .map(oxixml_xslt::Transformation::into_string)
        .expect("transform");
    assert_eq!(
        out,
        "<?xml version=\"1.0\" encoding=\"UTF-8\"?>\
         <result><primary><a>892</a></primary><secondary><out>479</out></secondary></result>"
    );
}

// --- the base output URI (XSLT 3.0 §23.1) ----------------------------------------------------

/// `current-output-uri-002`, `-003`: the base output URI is what `fn:current-output-uri()`
/// reports for the principal destination, and what `xsl:result-document/@href` resolves against.
#[test]
fn the_base_output_uri_is_reported_and_resolved_against() {
    let xsl = r#"<xsl:stylesheet xmlns:xsl="http://www.w3.org/1999/XSL/Transform" version="3.0">
      <xsl:template name="main">
        <out><xsl:value-of select="current-output-uri()"/></out>
        <xsl:result-document href="second/other.xml"><o/></xsl:result-document>
      </xsl:template>
    </xsl:stylesheet>"#;
    let compiled = Stylesheet::compile_str(xsl).expect("compile");
    let tree = Tree::parse_str("<doc/>").expect("a source document");
    let outcome = compiled
        .transform_with(
            &tree.root(),
            &Parameters::new()
                .with_initial_template("main")
                .with_base_output_uri("file:///results/main.xml"),
        )
        .expect("transform");
    assert_eq!(
        outcome.as_str(),
        "<?xml version=\"1.0\" encoding=\"UTF-8\"?><out>file:///results/main.xml</out>"
    );
    assert_eq!(
        outcome
            .secondary_results()
            .first()
            .map(|(uri, _)| uri.as_str()),
        Some("file:///results/second/other.xml")
    );
}

/// With no base output URI declared, the principal destination has no URI to report and an
/// `href` still resolves against the stylesheet's own base URI.
#[test]
fn without_a_base_output_uri_the_principal_destination_has_none() {
    let xsl = r#"<xsl:stylesheet xmlns:xsl="http://www.w3.org/1999/XSL/Transform" version="3.0">
      <xsl:template name="main"><out><xsl:value-of select="count(current-output-uri())"/></out></xsl:template>
    </xsl:stylesheet>"#;
    assert_eq!(
        run_named(xsl, "main").expect("transform"),
        "<?xml version=\"1.0\" encoding=\"UTF-8\"?><out>0</out>"
    );
}

// --- a caller-supplied deadline --------------------------------------------------------------

/// A transformation that would not finish stops at the deadline with the implementation-defined
/// limit error rather than holding the thread.
#[test]
fn a_deadline_stops_a_runaway_transformation() {
    let xsl = r#"<xsl:stylesheet xmlns:xsl="http://www.w3.org/1999/XSL/Transform"
        xmlns:xs="http://www.w3.org/2001/XMLSchema" version="3.0">
      <xsl:template name="main"><out>
        <xsl:for-each select="1 to 1000000000"><xsl:text>.</xsl:text></xsl:for-each>
      </out></xsl:template>
    </xsl:stylesheet>"#;
    let compiled = Stylesheet::compile_str(xsl).expect("compile");
    let tree = Tree::parse_str("<doc/>").expect("a source document");
    let outcome = compiled.transform_with(
        &tree.root(),
        &Parameters::new()
            .with_initial_template("main")
            .with_deadline(std::time::Instant::now() + std::time::Duration::from_millis(50)),
    );
    let error = outcome.expect_err("the deadline stops it");
    assert!(
        matches!(error.code(), Some("XPDY0130")),
        "unexpected code {:?}",
        error.code()
    );
}

/// A transformation that finishes inside the deadline is untouched by it.
#[test]
fn a_deadline_does_not_disturb_a_transformation_that_finishes() {
    let xsl = r#"<xsl:stylesheet xmlns:xsl="http://www.w3.org/1999/XSL/Transform" version="3.0">
      <xsl:template name="main"><out><xsl:value-of select="count(1 to 100)"/></out></xsl:template>
    </xsl:stylesheet>"#;
    let compiled = Stylesheet::compile_str(xsl).expect("compile");
    let tree = Tree::parse_str("<doc/>").expect("a source document");
    let out = compiled
        .transform_with(
            &tree.root(),
            &Parameters::new()
                .with_initial_template("main")
                .with_deadline(std::time::Instant::now() + std::time::Duration::from_secs(60)),
        )
        .map(oxixml_xslt::Transformation::into_string)
        .expect("transform");
    assert_eq!(
        out,
        "<?xml version=\"1.0\" encoding=\"UTF-8\"?><out>100</out>"
    );
}
