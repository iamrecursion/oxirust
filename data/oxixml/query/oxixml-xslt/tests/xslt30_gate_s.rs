//! Regression tests for four groups the full W3C `xslt30-test` catalogue exposed once every test
//! set in it was measured: the base URI a constructed tree carries, the shadow-attribute spelling
//! of an attribute an XSLT element requires, the restricted static context a `use-when` expression
//! is evaluated in, and the two rules the `xml` output method attaches to `version="1.1"`.
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

// --- the base URI of a constructed tree (XSLT 3.0 §9.2, §11.1.4) ---------------------------

/// `base-uri-013`: `base-uri()` of a temporary tree is the base URI of the *variable-binding
/// element*, which an `xml:base` on that element decides — "the base URI of a node in the
/// temporary tree is determined as if all the nodes in the temporary tree came from a single
/// entity whose URI was the base URI of the variable-binding element".
///
/// The builder used to carry only the *document* URI onto the tree it built, which a constructed
/// tree never has, so every one of these answered the empty string.
#[test]
fn a_temporary_tree_takes_the_base_uri_of_its_binding_element() {
    let xsl = r#"<t:transform xmlns:t="http://www.w3.org/1999/XSL/Transform"
             xml:base="http://www.baseuri.exmpl/tests/" version="3.0">
       <t:variable name="temptree" xml:base="d://tests/"><a><b>Text1</b></a></t:variable>
       <t:template match="/"><out><t:value-of select="base-uri($temptree)"/></out></t:template>
    </t:transform>"#;
    assert_eq!(
        run(xsl, "<doc/>").expect("transform"),
        "<?xml version=\"1.0\" encoding=\"UTF-8\"?><out>d://tests/</out>"
    );
}

/// `base-uri-029`: with no `xml:base` anywhere below it, the tree still takes the module's own
/// base URI — the `xml:base` on the `xsl:transform` element here — rather than nothing at all.
#[test]
fn a_temporary_tree_inherits_the_base_uri_of_the_module() {
    let xsl = r#"<t:transform xmlns:t="http://www.w3.org/1999/XSL/Transform"
             xml:base="http://www.example.org/" version="3.0">
       <t:template match="/doc"><out><t:apply-templates select="str1"/></out></t:template>
       <t:template match="str1">
         <t:variable name="copy"><t:copy>new content</t:copy></t:variable>
         <t:value-of select="base-uri($copy/str1)"/>
       </t:template>
    </t:transform>"#;
    assert_eq!(
        run(xsl, "<doc><str1>x</str1></doc>").expect("transform"),
        "<?xml version=\"1.0\" encoding=\"UTF-8\"?><out>http://www.example.org/</out>"
    );
}

/// `base-uri-034`: an `xml:base` written *into* the constructed content changes the base URI of
/// the element that carries it and of everything under it, resolved where it is relative against
/// the base URI in force where the element was written — the same rule XML Base gives a parsed
/// document, which is what §9.2 asks a temporary tree to behave like.
#[test]
fn an_xml_base_in_constructed_content_rebases_the_subtree() {
    let xsl = r#"<t:transform xmlns:t="http://www.w3.org/1999/XSL/Transform"
             xml:base="http://www.example.org/" version="3.0">
       <t:template match="/doc"><out><t:apply-templates select="str1"/></out></t:template>
       <t:template match="str1">
         <t:variable name="e" xml:base="/xxx/">
           <e1 xml:base="/main/"><t:copy>new content</t:copy></e1>
         </t:variable>
         <t:value-of select="base-uri($e/e1/str1)"/>
       </t:template>
    </t:transform>"#;
    assert_eq!(
        run(xsl, "<doc><str1>x</str1></doc>").expect("transform"),
        "<?xml version=\"1.0\" encoding=\"UTF-8\"?><out>http://www.example.org/main/</out>"
    );
}

/// `base-uri-015` and `error-1162a`, the other side of the same rule: [XDM 3.1
/// §5.2](https://www.w3.org/TR/xpath-datamodel-31/#dm-base-uri) gives an attribute, text or
/// comment node the base URI *of its parent*, so one that has no parent has none at all. An
/// element keeps its own, which is why §11.1.4 troubles to say the rule "is only relevant when
/// creating a parentless element".
#[test]
fn a_parentless_text_or_attribute_node_has_no_base_uri() {
    let xsl = r#"<t:transform xmlns:t="http://www.w3.org/1999/XSL/Transform"
             xml:base="http://www.example.org/" version="3.0">
       <t:variable name="attrib" as="attribute()"><t:attribute name="a" select="'v'"/></t:variable>
       <t:variable name="text" as="text()"><t:text>simple text</t:text></t:variable>
       <t:variable name="comment" as="comment()"><t:comment select="'c'"/></t:variable>
       <t:variable name="element" as="element()"><e/></t:variable>
       <t:template match="/"><out>
         <a><t:value-of select="base-uri($attrib)"/></a>
         <t><t:value-of select="base-uri($text)"/></t>
         <c><t:value-of select="base-uri($comment)"/></c>
         <e><t:value-of select="base-uri($element)"/></e>
       </out></t:template>
    </t:transform>"#;
    assert_eq!(
        run(xsl, "<doc/>").expect("transform"),
        "<?xml version=\"1.0\" encoding=\"UTF-8\"?><out><a/><t/><c/>\
         <e>http://www.example.org/</e></out>"
    );
}

// --- shadow attributes (XSLT 3.0 §3.14.2) --------------------------------------------------

/// `initial-function-101a`..`-102k`: §3.14.2 makes the shadow spelling available for *every*
/// no-namespace attribute of an element in the XSLT namespace. `xsl:function` read its `name`
/// off the element directly, so a whole stylesheet failed to compile — `XTSE0010: xsl:function
/// requires the name attribute` — the moment one function in it was declared this way, taking
/// every case that used the stylesheet with it.
#[test]
fn a_shadow_attribute_names_a_stylesheet_function() {
    let xsl = r#"<xsl:stylesheet version="3.0" xmlns:xsl="http://www.w3.org/1999/XSL/Transform"
          xmlns:xs="http://www.w3.org/2001/XMLSchema" xmlns:f="urn:functions"
          exclude-result-prefixes="xs f">
       <xsl:param name="suffix" static="yes" select="'2'"/>
       <xsl:function name="f:square1" visibility="public">
         <xsl:param name="p" as="xs:integer"/>
         <xsl:sequence select="$p * $p"/>
       </xsl:function>
       <xsl:function _name="{'f:square' || $suffix}" visibility="public">
         <xsl:param name="p" as="xs:integer"/>
         <xsl:sequence select="$p * $p * $p"/>
       </xsl:function>
       <xsl:template match="/"><out><xsl:value-of select="f:square1(12), f:square2(2)"/></out></xsl:template>
    </xsl:stylesheet>"#;
    assert_eq!(
        run(xsl, "<doc/>").expect("transform"),
        "<?xml version=\"1.0\" encoding=\"UTF-8\"?><out>144 8</out>"
    );
}

/// The same rule read through the other declarations that require a name: an `xsl:attribute-set`,
/// an `xsl:key` and an `xsl:character-map` all take theirs from a shadow attribute too, and a
/// reference written the ordinary way still finds them.
#[test]
fn a_shadow_attribute_names_the_other_declarations() {
    let xsl = r#"<xsl:stylesheet version="3.0" xmlns:xsl="http://www.w3.org/1999/XSL/Transform">
       <xsl:param name="which" static="yes" select="'big'"/>
       <xsl:attribute-set _name="{$which}"><xsl:attribute name="size">10</xsl:attribute></xsl:attribute-set>
       <xsl:key _name="{$which}" _match="item" use="@id"/>
       <xsl:template match="/">
         <out xsl:use-attribute-sets="big"><xsl:value-of select="key('big', 'a')/@id"/></out>
       </xsl:template>
    </xsl:stylesheet>"#;
    assert_eq!(
        run(xsl, "<doc><item id=\"a\"/></doc>").expect("transform"),
        "<?xml version=\"1.0\" encoding=\"UTF-8\"?><out size=\"10\">a</out>"
    );
}

/// `shadow-003`, `shadow-006`: the settings the *compile-time environment* is built from are
/// no-namespace attributes of XSLT elements too, so they carry the shadow spelling like any
/// other. `_version` on the module element decides the language level the module is read at, and
/// `_xpath-default-namespace` on a template decides what an unprefixed name in its expressions
/// selects — neither of which the compiler could see while it read the attribute literally.
#[test]
fn a_shadow_attribute_supplies_a_compile_time_setting() {
    let version = r#"<xsl:transform xmlns:xsl="http://www.w3.org/1999/XSL/Transform"
          _version="{system-property('xsl:version')}" version="1.0">
       <xsl:template name="main"><out><xsl:value-of select="1 to 5"/></out></xsl:template>
    </xsl:transform>"#;
    assert_eq!(
        run_named(version, "main").expect("transform"),
        "<?xml version=\"1.0\" encoding=\"UTF-8\"?><out>1 2 3 4 5</out>"
    );

    let namespace = r#"<xsl:transform xmlns:xsl="http://www.w3.org/1999/XSL/Transform" version="3.0">
       <xsl:variable name="ONE" static="yes" select="'http://www.example.com/one'"/>
       <xsl:variable name="data">
         <e xmlns="http://www.example.com/one">1</e>
         <e xmlns="http://www.example.com/two">2</e>
       </xsl:variable>
       <xsl:template name="main" _xpath-default-namespace="{$ONE}">
         <out><xsl:value-of select="$data/e"/></out>
       </xsl:template>
    </xsl:transform>"#;
    assert_eq!(
        run_named(namespace, "main").expect("transform"),
        "<?xml version=\"1.0\" encoding=\"UTF-8\"?><out>1</out>"
    );
}

/// `cvt-008`, `cvt-009`: `[xsl:]expand-text` has the value space of `xs:boolean` and is not an
/// attribute value template, so neither a computed value written in braces nor a spelling outside
/// that space is a value it may take — both are `XTSE0020` rather than a silent "no".
#[test]
fn expand_text_takes_a_boolean_and_not_a_value_template() {
    let braces = r#"<xsl:stylesheet xmlns:xsl="http://www.w3.org/1999/XSL/Transform"
          expand-text="no" version="3.0">
       <xsl:param name="yesOrNo" select="'no'"/>
       <xsl:template name="main" expand-text="{$yesOrNo}"><out/></xsl:template>
    </xsl:stylesheet>"#;
    assert_eq!(
        error_code(&run_named(braces, "main")).as_deref(),
        Some("XTSE0020")
    );

    let miscased = r#"<xsl:stylesheet xmlns:xsl="http://www.w3.org/1999/XSL/Transform"
          expand-text="no" version="3.0">
       <xsl:template name="main" expand-text="TRUE"><out/></xsl:template>
    </xsl:stylesheet>"#;
    assert_eq!(
        error_code(&run_named(miscased, "main")).as_deref(),
        Some("XTSE0020")
    );

    // The shadow spelling still supplies a computed value, which is judged once expanded.
    let shadowed = r#"<xsl:stylesheet xmlns:xsl="http://www.w3.org/1999/XSL/Transform"
          expand-text="no" version="3.0">
       <xsl:param name="yesOrNo" static="yes" select="'yes'"/>
       <xsl:template name="main" _expand-text="{$yesOrNo}"><out>{1 + 1}</out></xsl:template>
    </xsl:stylesheet>"#;
    assert_eq!(
        run_named(shadowed, "main").expect("transform"),
        "<?xml version=\"1.0\" encoding=\"UTF-8\"?><out>2</out>"
    );
}

// --- the restricted static context of a static expression (XSLT 3.0 §9.7) -------------------

/// `use-when-0123`: §9.7 gives a static expression no context item, so `.` is `XPDY0002`. Before
/// the fix the error was discarded — the whole evaluation was `unwrap_or(true)` — and the guarded
/// element was kept as though the expression had said so.
#[test]
fn a_use_when_expression_has_no_context_item() {
    let xsl = r#"<xsl:stylesheet xmlns:xsl="http://www.w3.org/1999/XSL/Transform" version="3.0">
       <xsl:template match="para"><p><xsl:value-of select="." use-when=". != 17"/></p></xsl:template>
    </xsl:stylesheet>"#;
    assert_eq!(
        error_code(&run(xsl, "<doc><para>p</para></doc>")).as_deref(),
        Some("XPDY0002")
    );
}

/// `use-when-0124`, `use-when-0131`: the in-scope variables of a static expression are "the
/// static variables … whose declarations occur prior to the element containing the static
/// expression", so a reference to an ordinary stylesheet variable — even one declared right
/// above it — is `XPST0008`.
#[test]
fn a_use_when_expression_sees_only_static_variables() {
    let xsl = r#"<xsl:stylesheet xmlns:xsl="http://www.w3.org/1999/XSL/Transform" version="3.0">
       <xsl:variable name="xx" select="2" static="no"/>
       <xsl:template match="*" use-when="$xx = 2"><p/></xsl:template>
    </xsl:stylesheet>"#;
    assert_eq!(error_code(&run(xsl, "<doc/>")).as_deref(), Some("XPST0008"));

    // The converse, so the rule cannot degrade into "no variable at all": a static variable
    // declared before the element is in scope and decides the inclusion.
    let xsl = r#"<xsl:stylesheet xmlns:xsl="http://www.w3.org/1999/XSL/Transform" version="3.0">
       <xsl:variable name="on" select="false()" static="yes"/>
       <xsl:template match="/"><out><in xsl:use-when="$on"/></out></xsl:template>
    </xsl:stylesheet>"#;
    assert_eq!(
        run(xsl, "<doc/>").expect("transform"),
        "<?xml version=\"1.0\" encoding=\"UTF-8\"?><out/>"
    );
}

/// `use-when-0126`: "stylesheet functions are not included in the context", so a call to one is
/// `XPST0017` however the stylesheet declares it.
#[test]
fn a_use_when_expression_cannot_call_a_stylesheet_function() {
    let xsl = r#"<xsl:stylesheet xmlns:xsl="http://www.w3.org/1999/XSL/Transform"
          xmlns:my="http://my.uri/" version="3.0">
       <xsl:template match="para" use-when="my:f()"><p/></xsl:template>
       <xsl:function name="my:f"><xsl:sequence select="true()"/></xsl:function>
    </xsl:stylesheet>"#;
    assert_eq!(
        error_code(&run(xsl, "<doc><para>p</para></doc>")).as_deref(),
        Some("XPST0017")
    );
}

/// `use-when-0407a`: §9.7's list is F&O plus `element-available`, `function-available`,
/// `type-available` and `system-property` — so `fn:function-available` asked from inside a
/// static expression answers `true` for `generate-id`, which F&O 3.0 defines, and `false` for
/// `current`, `key` and the unparsed-entity accessors, which need a transformation.
#[test]
fn function_available_in_a_use_when_reports_the_restricted_library() {
    let xsl = r#"<t:transform xmlns:t="http://www.w3.org/1999/XSL/Transform" version="3.0">
       <t:template match="/"><out>
          <in t:use-when="function-available('current')" f="current"/>
          <in t:use-when="function-available('key')" f="key"/>
          <in t:use-when="function-available('unparsed-entity-uri')" f="unparsed-entity-uri"/>
          <in t:use-when="function-available('generate-id')" f="generate-id"/>
          <in t:use-when="function-available('system-property')" f="system-property"/>
       </out></t:template>
    </t:transform>"#;
    assert_eq!(
        run(xsl, "<doc/>").expect("transform"),
        "<?xml version=\"1.0\" encoding=\"UTF-8\"?><out>\
         <in f=\"generate-id\"/><in f=\"system-property\"/></out>"
    );
}

/// `use-when-0116`: §3.14.1 — "if the `xsl:stylesheet` or `xsl:transform` element itself is
/// effectively excluded, the effect is to exclude all the children … but not the element or its
/// attributes". The module still contributes the defaults its attributes state, and declares
/// nothing.
#[test]
fn a_module_element_excluded_by_use_when_declares_nothing() {
    let included = r#"<xsl:stylesheet xmlns:xsl="http://www.w3.org/1999/XSL/Transform"
          version="3.0" use-when="false()">
       <xsl:template match="para"><p><xsl:next-match/></p></xsl:template>
    </xsl:stylesheet>"#;
    let principal = r#"<xsl:stylesheet xmlns:xsl="http://www.w3.org/1999/XSL/Transform" version="3.0">
       <xsl:include href="http://example.com/inc.xsl"/>
       <xsl:template match="*"><xsl:copy><xsl:apply-templates/></xsl:copy></xsl:template>
    </xsl:stylesheet>"#;
    let resolver = MemoryResolver::new().with_document("http://example.com/inc.xsl", included);
    let options = StaticOptions::new()
        .with_base_uri("http://example.com/principal.xsl")
        .with_resolver(Arc::new(resolver));
    let compiled = Stylesheet::compile_str_with(principal, options).expect("compile");
    let tree = Tree::parse_str("<doc><para>p1</para></doc>").expect("parse");
    assert_eq!(
        compiled.transform_to_string(&tree.root()).expect("run"),
        "<?xml version=\"1.0\" encoding=\"UTF-8\"?><doc><para>p1</para></doc>"
    );
}

/// `use-when-0220`: §3.14.1 excludes the element "together with all the nodes having that element
/// as an ancestor", and for `xsl:include` that is the whole module it names — whatever the
/// included module's own `use-when` says. The module here is not even retrieved, so the resolver
/// that cannot supply it proves the point.
#[test]
fn an_include_excluded_by_use_when_is_not_loaded() {
    let principal = r#"<xsl:stylesheet xmlns:xsl="http://www.w3.org/1999/XSL/Transform" version="3.0">
       <xsl:include href="http://example.com/missing.xsl" use-when="false()"/>
       <xsl:template match="/"><out/></xsl:template>
    </xsl:stylesheet>"#;
    let options = StaticOptions::new()
        .with_base_uri("http://example.com/principal.xsl")
        .with_resolver(Arc::new(MemoryResolver::new()));
    let compiled = Stylesheet::compile_str_with(principal, options).expect("compile");
    let tree = Tree::parse_str("<doc/>").expect("parse");
    assert_eq!(
        compiled.transform_to_string(&tree.root()).expect("run"),
        "<?xml version=\"1.0\" encoding=\"UTF-8\"?><out/>"
    );
}

/// `use-when-0401`: a parameter is an element like any other, so `use-when="false()"` on an
/// `xsl:with-param` withholds it and the receiving `xsl:param` falls back on its own default.
#[test]
fn use_when_governs_a_parameter() {
    let xsl = r#"<xsl:transform xmlns:xsl="http://www.w3.org/1999/XSL/Transform" version="3.0">
       <xsl:template match="/"><out>
         <xsl:call-template name="temp" use-when="2*2=4">
           <xsl:with-param name="p1" select="'a'" use-when="true()"/>
           <xsl:with-param name="p2" select="'b'" use-when="false()"/>
         </xsl:call-template>
       </out></xsl:template>
       <xsl:template name="temp">
         <xsl:param name="p1" select="'def'"/>
         <xsl:param name="p2" select="'def'"/>
         <xsl:sequence select="concat($p1, $p2)"/>
       </xsl:template>
    </xsl:transform>"#;
    assert_eq!(
        run(xsl, "<doc/>").expect("transform"),
        "<?xml version=\"1.0\" encoding=\"UTF-8\"?><out>adef</out>"
    );
}

/// `static-006`, `static-007`: §9.6 — "when the attribute `static="yes"` is specified, the
/// `xsl:variable` or `xsl:param` element must have empty content. In the case of `xsl:variable`
/// the `select` attribute must be present to define the value of the variable." A sequence
/// constructor is not a value the compiler can have before it compiles the stylesheet.
#[test]
fn a_static_declaration_must_carry_its_value_in_a_select() {
    let with_content = r#"<xsl:transform xmlns:xsl="http://www.w3.org/1999/XSL/Transform" version="3.0">
       <xsl:param name="p" static="yes" required="no">content</xsl:param>
       <xsl:template match="/"><out/></xsl:template>
    </xsl:transform>"#;
    assert_eq!(
        error_code(&run(with_content, "<doc/>")).as_deref(),
        Some("XTSE0010")
    );

    let without_select = r#"<xsl:transform xmlns:xsl="http://www.w3.org/1999/XSL/Transform" version="3.0">
       <xsl:variable name="v" static="yes"/>
       <xsl:template match="/"><out/></xsl:template>
    </xsl:transform>"#;
    assert_eq!(
        error_code(&run(without_select, "<doc/>")).as_deref(),
        Some("XTSE0010")
    );

    // A static parameter needs no `select`: the caller may supply its value instead.
    let parameter = r#"<xsl:transform xmlns:xsl="http://www.w3.org/1999/XSL/Transform" version="3.0">
       <xsl:param name="p" static="yes"/>
       <xsl:template match="/"><out/></xsl:template>
    </xsl:transform>"#;
    assert!(run(parameter, "<doc/>").is_ok());
}

// --- the xml output method under version="1.1" (Serialization 3.1 §7.2) ---------------------

/// `xml-version-007`..`-010`, `-018`: XML 1.1 admits the C0 and C1 control characters *only*
/// written as character references, so a serializer that writes one literally produces a document
/// no parser reads back. Both a text node and an attribute value are the escaper's business.
#[test]
fn xml_1_1_writes_control_characters_as_references() {
    // The stylesheet itself is XML 1.1: no XML 1.0 document may spell these characters, even as
    // references, which is the whole reason the output method has a rule about them.
    let xsl = "<?xml version=\"1.1\" encoding=\"UTF-8\"?>\
       <t:transform xmlns:t=\"http://www.w3.org/1999/XSL/Transform\" version=\"3.0\">\
       <t:output method=\"xml\" version=\"1.1\" encoding=\"UTF-8\"/>\
       <t:template match=\"/\"><out><a status=\"&#x8;&#x1f;\"/><b>&#x82;</b>\
<c>tab\tand newline</c></out></t:template>\
       </t:transform>";
    assert_eq!(
        run(xsl, "<doc/>").expect("transform"),
        "<?xml version=\"1.1\" encoding=\"UTF-8\"?><out><a status=\"&#8;&#31;\"/><b>&#130;</b>\
<c>tab\tand newline</c></out>"
    );
}

/// `xml-version-029`, `-030`: the same characters have no spelling at all under XML 1.0 — a
/// character reference to one is as ill-formed as the character itself — so writing the result is
/// the serialization error `SERE0006`.
#[test]
fn xml_1_0_refuses_a_character_it_cannot_spell() {
    let xsl = "<?xml version=\"1.1\" encoding=\"UTF-8\"?>\
       <t:transform xmlns:t=\"http://www.w3.org/1999/XSL/Transform\" version=\"3.0\">\
       <t:output method=\"xml\" version=\"1.0\" encoding=\"UTF-8\"/>\
       <t:template name=\"main\"><out>&#x7;</out></t:template>\
       </t:transform>";
    assert_eq!(
        error_code(&run_named(xsl, "main")).as_deref(),
        Some("SERE0006")
    );

    // Under 1.1 the same tree serializes, with the character written numerically.
    let permitted = "<?xml version=\"1.1\" encoding=\"UTF-8\"?>\
       <t:transform xmlns:t=\"http://www.w3.org/1999/XSL/Transform\" version=\"3.0\">\
       <t:output method=\"xml\" version=\"1.1\" encoding=\"UTF-8\"/>\
       <t:template name=\"main\"><out>&#x7;</out></t:template>\
       </t:transform>";
    assert_eq!(
        run_named(permitted, "main").expect("transform"),
        "<?xml version=\"1.1\" encoding=\"UTF-8\"?><out>&#7;</out>"
    );
}

// --- system properties ----------------------------------------------------------------------

/// `system-property-016` accepts either answer for `xsl:xsd-version`, but `type-available-0151`
/// and `error-0905a` measure the schema types this processor actually knows, and those are XSD
/// 1.0's: `xs:dateTimeStamp` is not among them. The property has to report the same reading, or a
/// stylesheet that branches on it takes the branch this processor cannot honour.
#[test]
fn the_xsd_version_property_agrees_with_the_types_this_processor_knows() {
    let xsl = r#"<xsl:transform xmlns:xsl="http://www.w3.org/1999/XSL/Transform"
          xmlns:xs="http://www.w3.org/2001/XMLSchema" exclude-result-prefixes="xs" version="3.0">
       <xsl:template match="/"><out>
         <v><xsl:value-of select="system-property('xsl:xsd-version')"/></v>
         <t><xsl:value-of select="type-available('xs:dateTimeStamp')"/></t>
         <d><xsl:value-of select="xs:decimal(system-property('xsl:xsd-version')) = (1.0, 1.1)"/></d>
       </out></xsl:template>
    </xsl:transform>"#;
    assert_eq!(
        run(xsl, "<doc/>").expect("transform"),
        "<?xml version=\"1.0\" encoding=\"UTF-8\"?><out><v>1.0</v><t>false</t><d>true</d></out>"
    );
}
