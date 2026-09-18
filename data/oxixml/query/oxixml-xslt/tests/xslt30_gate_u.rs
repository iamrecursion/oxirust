//! Regression tests for the clusters the full W3C `xslt30-test` catalogue reported once the base
//! output URI was being supplied and the base URI of copied nodes was being asked for: the
//! constructs that clear the *current output URI*, the base URI a copy keeps, the result a
//! destination that builds no final result tree leaves behind, and the cost of evaluating an
//! expression while a large global variable is in scope.
//!
//! Every test reproduces one catalogue case with a minimal inline stylesheet, so each fix is
//! provable red-then-green without the separate `oxixml-testsuite` harness. The case identifiers
//! in the comments are the ones that harness reports.

use oxixml_infoset::{Tree, XdmNode};
use oxixml_xslt::{
    MemoryResolver, Parameters, StaticOptions, Stylesheet, Transformation, XsltError,
};
use std::sync::Arc;
use std::time::{Duration, Instant};

/// Parses `source` as the principal source document.
fn source(xml: &str) -> Tree {
    Tree::parse_str(xml).expect("the source document parses")
}

/// Compiles `xsl` and runs it over `xml`, returning the serialized result.
fn run(xsl: &str, xml: &str) -> Result<String, XsltError> {
    Stylesheet::compile_str(xsl)?.transform_to_string(&source(xml).root())
}

/// Compiles `xsl` and runs it from its default entry point with `uri` as the base output URI.
fn run_with_output_uri(xsl: &str, uri: &str) -> Result<String, XsltError> {
    Stylesheet::compile_str(xsl)?
        .transform_with(
            &source("<doc/>").root(),
            &Parameters::new().with_base_output_uri(uri),
        )
        .map(Transformation::into_string)
}

// --- the constructs that clear the current output URI (XSLT 3.0 §24.3) -----------------------

/// The base output URI a case supplying one declares; the assertions read the tail of it.
const OUTPUT_URI: &str = "file:///results/current-output-uri-008.xml";

/// `current-output-uri-008`: a **pattern** is evaluated with the current output URI absent, so a
/// rule guarded by `empty(current-output-uri())` is the one that fires.
///
/// §24.3 clears the URI "while evaluating stylesheet functions, dynamic function calls,
/// evaluation of global variables, stylesheet parameters, and patterns". The first three were
/// already covered by *temporary output state*; a pattern is not an output state at all, and was
/// therefore answering the base output URI and selecting the wrong rule.
#[test]
fn a_pattern_is_matched_with_the_current_output_uri_absent() {
    let xsl = r#"<xsl:stylesheet xmlns:xsl="http://www.w3.org/1999/XSL/Transform" version="3.0">
      <xsl:variable name="base" as="element()"><base><value/></base></xsl:variable>
      <xsl:template match="$base/value[current-output-uri()]"><wrong/></xsl:template>
      <xsl:template match="$base/value[empty(current-output-uri())]"><correct/></xsl:template>
      <xsl:template name="xsl:initial-template">
        <result><xsl:apply-templates select="$base/value"/></result>
      </xsl:template>
    </xsl:stylesheet>"#;
    assert_eq!(
        run_with_output_uri(xsl, OUTPUT_URI).expect("transform"),
        "<?xml version=\"1.0\" encoding=\"UTF-8\"?><result><correct/></result>"
    );
}

/// The same rule for the pattern an `xsl:for-each-group` matches: `-008` groups `-1 to 1` with
/// `group-starting-with=".[.=count(current-output-uri())]"`, so the group that starts is the one
/// at `0` — which holds only while the count is zero.
#[test]
fn a_grouping_pattern_is_matched_with_the_current_output_uri_absent() {
    let xsl = r#"<xsl:stylesheet xmlns:xsl="http://www.w3.org/1999/XSL/Transform" version="3.0">
      <xsl:template name="xsl:initial-template">
        <result>
          <xsl:for-each-group select="-1 to +1" group-starting-with=".[.=count(current-output-uri())]">
            <group at="{.}"/>
          </xsl:for-each-group>
        </result>
      </xsl:template>
    </xsl:stylesheet>"#;
    assert_eq!(
        run_with_output_uri(xsl, OUTPUT_URI).expect("transform"),
        "<?xml version=\"1.0\" encoding=\"UTF-8\"?><result><group at=\"-1\"/><group at=\"0\"/></result>"
    );
}

/// Outside a pattern the same call still answers the base output URI, which is what keeps
/// `current-output-uri-002`, `-003`, `-004` and `-014` passing.
#[test]
fn a_template_body_still_sees_the_base_output_uri() {
    let xsl = r#"<xsl:stylesheet xmlns:xsl="http://www.w3.org/1999/XSL/Transform" version="3.0">
      <xsl:template name="xsl:initial-template"><out><xsl:value-of select="current-output-uri()"/></out></xsl:template>
    </xsl:stylesheet>"#;
    assert_eq!(
        run_with_output_uri(xsl, OUTPUT_URI).expect("transform"),
        format!("<?xml version=\"1.0\" encoding=\"UTF-8\"?><out>{OUTPUT_URI}</out>")
    );
}

/// `current-output-uri-016`: reached through a **function item**, the call answers the empty
/// sequence — §24.3's "dynamic function calls", which the case cites as W3C bug #30411.
#[test]
fn a_dynamic_call_on_current_output_uri_is_the_empty_sequence() {
    let xsl = r#"<xsl:stylesheet xmlns:xsl="http://www.w3.org/1999/XSL/Transform" version="3.0">
      <xsl:variable name="out-uri" select="current-output-uri#0"/>
      <xsl:template name="xsl:initial-template">
        <out>current-output-uri-is-empty=<xsl:value-of select="empty($out-uri())"/></out>
      </xsl:template>
    </xsl:stylesheet>"#;
    assert_eq!(
        run_with_output_uri(xsl, OUTPUT_URI).expect("transform"),
        "<?xml version=\"1.0\" encoding=\"UTF-8\"?><out>current-output-uri-is-empty=true</out>"
    );
}

// --- the base URI a copy keeps (XSLT 3.0 §11.9.1, §11.9.2) -----------------------------------

/// The source `base-uri-024`, `-035` and `-046` are all read from: an element under two nested
/// `xml:base` declarations.
const BASED_SOURCE: &str = r#"<doc>
  <str1 xml:base="http://www.example.com/str1/">string1<substring1 xml:base="http://www.example.com/xml/">substring</substring1></str1>
</doc>"#;

/// `base-uri-024`: a copy that is still **parentless** keeps the base URI of the node it was
/// copied from, not the one in force where the copy was written.
///
/// The variable declares `as="element()"`, so its value is the copied element itself rather than
/// a document node wrapping it — §11.9.1's "when a node is copied, its base URI is copied", with
/// nothing to inherit from.
#[test]
fn a_parentless_copy_keeps_the_base_uri_it_was_copied_with() {
    let xsl = r#"<xsl:transform xmlns:xsl="http://www.w3.org/1999/XSL/Transform"
       xml:base="http://www.example.org/" version="2.0">
      <xsl:template match="/doc"><out><xsl:apply-templates select="str1/*"/></out></xsl:template>
      <xsl:template match="substring1">
        <xsl:variable name="copy" as="element()" xml:base="/main/"><xsl:copy>new</xsl:copy></xsl:variable>
        <xsl:value-of select="base-uri($copy)"/>
      </xsl:template>
    </xsl:transform>"#;
    assert_eq!(
        run(xsl, BASED_SOURCE).expect("transform"),
        "<?xml version=\"1.0\" encoding=\"UTF-8\"?><out>http://www.example.com/xml/</out>"
    );
}

/// `base-uri-035`, the other half of the same rule: once the copy is **attached** — here as the
/// child of the document node a variable without `as` builds — it inherits its parent's base URI,
/// which is the variable's own `xml:base`.
#[test]
fn an_attached_copy_inherits_the_base_uri_of_its_new_parent() {
    let xsl = r#"<xsl:transform xmlns:xsl="http://www.w3.org/1999/XSL/Transform"
       xml:base="http://www.example.org/" version="2.0">
      <xsl:template match="/doc"><out><xsl:apply-templates select="str1/*"/></out></xsl:template>
      <xsl:template match="substring1">
        <xsl:variable name="copy" xml:base="/main/"><xsl:copy>new</xsl:copy></xsl:variable>
        <xsl:value-of select="base-uri($copy/substring1)"/>
      </xsl:template>
    </xsl:transform>"#;
    assert_eq!(
        run(xsl, BASED_SOURCE).expect("transform"),
        "<?xml version=\"1.0\" encoding=\"UTF-8\"?><out>http://www.example.org/main/</out>"
    );
}

/// `base-uri-046`, the exception §11.9.2 states in as many words: a copied element that carries
/// an `xml:base` attribute of its own takes its base URI from that attribute, "resolved if it is
/// relative against the base URI of the `xsl:copy-of` instruction" — against where the copy was
/// written, and not against where the original stood.
#[test]
fn a_copied_xml_base_resolves_against_the_copying_instruction() {
    let xsl = r#"<xsl:transform xmlns:xsl="http://www.w3.org/1999/XSL/Transform"
       xml:base="http://www.example.org/" version="2.0">
      <xsl:variable name="copy" as="element()"><xsl:copy-of select="/doc/str1"/></xsl:variable>
      <xsl:template match="/doc"><out><xsl:value-of select="base-uri($copy)"/></out></xsl:template>
    </xsl:transform>"#;
    let relative =
        r#"<doc xml:base="http://www.xmlexample.eu/"><str1 xml:base="/xml/">string1</str1></doc>"#;
    assert_eq!(
        run(xsl, relative).expect("transform"),
        "<?xml version=\"1.0\" encoding=\"UTF-8\"?><out>http://www.example.org/xml/</out>"
    );
}

/// `base-uri-053`: every way of copying a retrieved document — `xsl:copy`, `xsl:copy-of`, and the
/// built-in `shallow-copy` and `deep-copy` rules — leaves the copy with the document's base URI,
/// and passes it on to the nodes built inside the copy.
///
/// The built-in `shallow-copy` rule is `<xsl:copy>` over *every* item
/// ([§6.7.3](https://www.w3.org/TR/xslt-30/#built-in-rule-shallow-copy)), a document node
/// included, so it has to build a document node of its own rather than write out the children it
/// walks.
#[test]
fn copies_of_a_retrieved_document_keep_its_base_uri() {
    let xsl = r#"<xsl:transform xmlns:xsl="http://www.w3.org/1999/XSL/Transform" version="3.0">
      <xsl:variable name="doc" select="doc('urn:retrieved')"/>
      <xsl:template name="xsl:initial-template">
        <out>
          <original a="{base-uri($doc)}" b="{base-uri($doc/*)}"/>
          <xsl:variable name="shallow" as="item()*">
            <xsl:copy select="$doc"><z/></xsl:copy>
            <xsl:copy select="$doc/*"><z/></xsl:copy>
          </xsl:variable>
          <shallow a="{base-uri($shallow[1])}" b="{base-uri($shallow[2])}"/>
          <deeper a="{base-uri($shallow[1]/z)}" b="{base-uri($shallow[2]/z)}"/>
          <xsl:variable name="deep" as="item()*"><xsl:copy-of select="$doc"/></xsl:variable>
          <deep a="{base-uri($deep[1])}" b="{base-uri($deep[1]/*)}"/>
          <xsl:variable name="built-in" as="item()*">
            <xsl:apply-templates select="$doc" mode="shallow"/>
            <xsl:apply-templates select="$doc" mode="deep"/>
          </xsl:variable>
          <built-in a="{base-uri($built-in[1])}" b="{base-uri($built-in[2])}"/>
        </out>
      </xsl:template>
      <xsl:mode name="shallow" on-no-match="shallow-copy"/>
      <xsl:template match="/*" mode="shallow"/>
      <xsl:mode name="deep" on-no-match="deep-copy"/>
    </xsl:transform>"#;
    let resolver = Arc::new(MemoryResolver::new().with_document("urn:retrieved", "<r><c/></r>"));
    let compiled = Stylesheet::compile_str_with(xsl, StaticOptions::new().with_resolver(resolver))
        .expect("compile");
    let produced = compiled
        .transform_to_string(&source("<doc/>").root())
        .expect("transform");
    for element in ["original", "shallow", "deeper", "deep", "built-in"] {
        for attribute in ['a', 'b'] {
            assert!(
                produced.contains(&format!("<{element} {attribute}=\"urn:retrieved\""))
                    || produced.contains(&format!(" {attribute}=\"urn:retrieved\"/>")),
                "{element}/@{attribute} lost the retrieved document's base URI: {produced}"
            );
        }
    }
    assert!(
        !produced.contains("=\"\""),
        "some base URI came back empty: {produced}"
    );
}

// --- the result of a destination that builds no final result tree ----------------------------

/// The stylesheet `maps-017` is built from: a map serialized to a string with `fn:serialize`,
/// written inside a literal result element, under `xsl:output method="json"`.
const JSON_METHOD_STYLESHEET: &str = r#"<xsl:stylesheet version="3.0"
    xmlns:xsl="http://www.w3.org/1999/XSL/Transform" xpath-default-namespace="http://www.example.com">
    <xsl:output method="json"/>
    <xsl:template match="/">
        <xsl:variable name="v" as="map(*)">
            <xsl:map-entry key="string(//bar)" select="'greeting'"/>
        </xsl:variable>
        <out><xsl:value-of select="serialize($v, map{'method':'json'})"/></out>
    </xsl:template>
</xsl:stylesheet>"#;

/// The source `maps-017` supplies: the key of the map is read from it through the default
/// namespace the stylesheet declares for XPath.
const NAMESPACED_SOURCE: &str =
    r#"<foo:root xmlns:foo="http://www.example.com"><foo:bar>Hello world</foo:bar></foo:root>"#;

/// `maps-017`: the `json` output method builds no final result tree, but the nodes it was handed
/// are still the transformation's result and a caller must be able to navigate them.
///
/// [XSLT 3.0 §26](https://www.w3.org/TR/xslt-30/#serialization) defaults `build-tree` to `no` for
/// that method because its result may be a map or an array, which no tree can hold — the items
/// were therefore being dropped, leaving an empty document node behind, even where every one of
/// them was an ordinary node.
#[test]
fn a_destination_that_builds_no_tree_still_publishes_its_nodes() {
    let compiled = Stylesheet::compile_str(JSON_METHOD_STYLESHEET).expect("compile");
    let produced = compiled
        .transform_with(&source(NAMESPACED_SOURCE).root(), &Parameters::new())
        .expect("transform");
    let element = produced
        .root()
        .children()
        .into_iter()
        .find(|child| {
            child
                .node_name()
                .is_some_and(|name| name.local_name() == "out")
        })
        .expect("the result tree holds the `out` element the stylesheet built");
    assert_eq!(element.string_value(), r#"{"Hello world":"greeting"}"#);
    // The same items are the raw result, which is what a caller asking for a sequence gets.
    assert_eq!(produced.as_sequence().len(), 1);
    // The serialization is the method's own and is untouched by the tree being navigable.
    assert_eq!(
        produced.as_str(),
        r#""<out>{\"Hello world\":\"greeting\"}<\/out>""#
    );
}

// --- the cost of an expression evaluated while a global variable is in scope ------------------

/// A stylesheet whose function is called `calls` times with a global variable of `items` entries
/// in scope. The body evaluates three expressions per call, so the per-call cost is what the
/// measurement below compares.
fn function_call_stylesheet(items: usize, calls: usize) -> String {
    format!(
        r#"<xsl:stylesheet version="3.0" xmlns:xsl="http://www.w3.org/1999/XSL/Transform"
             xmlns:xs="http://www.w3.org/2001/XMLSchema" xmlns:f="urn:f">
          <xsl:variable name="big" select="for $i in 1 to {items} return string($i)"/>
          <xsl:function name="f:step" as="xs:integer">
            <xsl:param name="n" as="xs:integer"/>
            <xsl:variable name="a" select="$n + 1"/>
            <xsl:variable name="b" select="$a * 2"/>
            <xsl:sequence select="$b - $a"/>
          </xsl:function>
          <xsl:template name="xsl:initial-template">
            <out><xsl:value-of select="sum(for $i in 1 to {calls} return f:step($i))"/></out>
          </xsl:template>
        </xsl:stylesheet>"#
    )
}

/// How long one such transformation takes.
fn time_of(items: usize, calls: usize) -> Duration {
    let xsl = function_call_stylesheet(items, calls);
    let compiled = Stylesheet::compile_str(&xsl).expect("compile");
    let root = source("<doc/>").root();
    let started = Instant::now();
    compiled
        .transform_with(&root, &Parameters::new())
        .expect("transform");
    started.elapsed()
}

/// The size of a global variable must not be a cost paid per expression.
///
/// Every expression used to be evaluated in a dynamic context built by binding each global
/// afresh, and a binding takes the value by value — so a stylesheet that reads a file into a
/// global copied every item of it once per expression, and a stylesheet function called a hundred
/// thousand times paid for it a hundred thousand times over. `normalize-unicode-008` is that
/// stylesheet: it tokenizes a 2.1 MB file into a global and calls a function per line.
///
/// The claim is a *ratio* rather than a duration because the ratio is what the defect governs and
/// what survives a loaded machine: with the bindings shared the two runs below do the same work,
/// and without it the large one did a hundred times more.
#[test]
fn a_large_global_costs_nothing_per_expression() {
    const CALLS: usize = 300;
    let small = time_of(1, CALLS).max(Duration::from_millis(1));
    let large = time_of(20_000, CALLS);
    assert!(
        large < small * 10 + Duration::from_millis(500),
        "a 20 000-item global made {CALLS} function calls take {large:?} against {small:?} with a \
         one-item global, so the bindings are being rebuilt per expression"
    );
}
