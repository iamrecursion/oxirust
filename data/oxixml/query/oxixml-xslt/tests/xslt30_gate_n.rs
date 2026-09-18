//! Regression tests for the static base URI an expression is evaluated under, the
//! `document('')` self-reference, `xsl:result-document`'s serialization parameters and the
//! spec-mandated errors for the instructions this processor does not implement.
//!
//! Every test reproduces one W3C `xslt30-test` case with a minimal inline stylesheet, so each fix
//! is provable red-then-green without the separate `oxixml-testsuite` harness. The case
//! identifiers in the comments are the ones that harness reports.

use oxixml_infoset::Tree;
use oxixml_xslt::{MemoryResolver, StaticOptions, Stylesheet, XsltError};
use std::sync::Arc;

/// The stylesheet URI every self-reference test compiles under, so that `document('')` has a base
/// URI to resolve the empty reference against.
const PRINCIPAL_URI: &str = "http://example.com/style/principal.xsl";

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

// --- the static base URI of an expression -------------------------------------------------

/// `resolve-uri-012` / `-013`: XSLT 3.0 §5.4.3 makes the static base URI of an XPath expression
/// the base URI of the *element that contains it*, so an `xml:base` on the template moves it
/// away from the module's own URI.
#[test]
fn xml_base_moves_the_static_base_uri_of_the_expressions_below_it() {
    let xsl = r#"<xsl:stylesheet version="3.0" xmlns:xsl="http://www.w3.org/1999/XSL/Transform">
          <xsl:output omit-xml-declaration="yes"/>
          <xsl:template match="doc" xml:base="http://some.other/">
            <out><xsl:value-of select="static-base-uri()"/></out>
          </xsl:template>
        </xsl:stylesheet>"#;
    assert_eq!(
        text(run_module_source(xsl, &[], "<doc/>")),
        "<out>http://some.other/</out>"
    );
}

/// `resolve-uri-022` / `-023` / `-024`: an `xml:base` that is itself a relative reference is
/// resolved against the base URI in scope where it appears — RFC 3986 §5.1.1 as XML Base §3
/// applies it — and the dot segments are removed on the way.
#[test]
fn a_relative_xml_base_is_resolved_against_the_one_it_is_written_under() {
    let xsl = r#"<xsl:stylesheet version="3.0" xmlns:xsl="http://www.w3.org/1999/XSL/Transform">
          <xsl:output omit-xml-declaration="yes"/>
          <xsl:template match="doc" xml:base="http://example.com/a/b/c/d/../d/e/">
            <out xml:base="Z/../Z">
              <base><xsl:value-of select="static-base-uri()"/></base>
              <up><xsl:value-of select="resolve-uri('..')"/></up>
            </out>
          </xsl:template>
        </xsl:stylesheet>"#;
    assert_eq!(
        text(run_module_source(xsl, &[], "<doc/>")),
        "<out xml:base=\"Z/../Z\"><base>http://example.com/a/b/c/d/e/Z</base>\
         <up>http://example.com/a/b/c/d/</up></out>"
    );
}

/// The static base URI an expression sees follows the module it is written in, which is what
/// makes `fn:resolve-uri` inside an `xsl:include` name documents beside *that* module.
#[test]
fn an_included_modules_expressions_resolve_against_that_module() {
    let included_uri = "http://example.com/other/included.xsl";
    let included = r#"<xsl:stylesheet version="3.0" xmlns:xsl="http://www.w3.org/1999/XSL/Transform">
          <xsl:template match="doc">
            <out><xsl:value-of select="resolve-uri('data.xml')"/></out>
          </xsl:template>
        </xsl:stylesheet>"#;
    let principal = r#"<xsl:stylesheet version="3.0" xmlns:xsl="http://www.w3.org/1999/XSL/Transform">
          <xsl:output omit-xml-declaration="yes"/>
          <xsl:include href="http://example.com/other/included.xsl"/>
        </xsl:stylesheet>"#;
    assert_eq!(
        text(run_module_source(
            principal,
            &[(included_uri, included)],
            "<doc/>"
        )),
        "<out>http://example.com/other/data.xml</out>"
    );
}

// --- `document('')`, the stylesheet's own module ------------------------------------------

/// `key-066`, `variable-0105`, `select-5601`, `select-5701`: XSLT 1.0 §12.1, carried into
/// XSLT 3.0 §18.6, makes an *empty* URI reference in `document()` denote the module containing
/// the call, exactly as `fn:doc('')` does.
#[test]
fn document_of_the_empty_uri_is_the_stylesheet_module() {
    let xsl = r#"<?xml version="1.0"?>
        <xsl:stylesheet version="3.0" xmlns:xsl="http://www.w3.org/1999/XSL/Transform"
              xmlns:ped="http://ped.com/" exclude-result-prefixes="ped">
          <xsl:output omit-xml-declaration="yes"/>
          <xsl:template match="test">
            <out><xsl:value-of select="document('')/xsl:stylesheet/ped:test"/></out>
          </xsl:template>
          <ped:test>YEE-HA</ped:test>
        </xsl:stylesheet>"#;
    assert_eq!(
        text(run_module_source(xsl, &[], "<doc><test>ERROR</test></doc>")),
        "<out>YEE-HA</out>"
    );
}

/// The empty reference is local to the module it is written in: a `document('')` inside an
/// included module names *that* module, which is what taking the base from the call's static
/// context rather than from the transformation buys.
#[test]
fn document_of_the_empty_uri_in_an_included_module_is_that_module() {
    let included_uri = "http://example.com/style/included.xsl";
    let included = r#"<?xml version="1.0"?>
        <xsl:stylesheet version="3.0" xmlns:xsl="http://www.w3.org/1999/XSL/Transform"
              xmlns:ped="http://ped.com/" exclude-result-prefixes="ped">
          <xsl:template match="test">
            <out><xsl:value-of select="document('')/xsl:stylesheet/ped:test"/></out>
          </xsl:template>
          <ped:test>YEE-HA</ped:test>
        </xsl:stylesheet>"#;
    let principal = r#"<?xml version="1.0"?>
        <xsl:stylesheet version="3.0" xmlns:xsl="http://www.w3.org/1999/XSL/Transform"
              xmlns:ped="http://ped.com/" exclude-result-prefixes="ped">
          <xsl:output omit-xml-declaration="yes"/>
          <xsl:include href="included.xsl"/>
          <xsl:template match="doc"><o><xsl:apply-templates/></o></xsl:template>
          <ped:test>Oops</ped:test>
        </xsl:stylesheet>"#;
    assert_eq!(
        text(run_module_source(
            principal,
            &[(included_uri, included)],
            "<doc><test>ERROR</test></doc>"
        )),
        "<o><out>YEE-HA</out></o>"
    );
}

/// A non-empty reference still resolves against the same base, so the empty-reference rule did
/// not swallow the ordinary case.
#[test]
fn document_of_a_relative_uri_resolves_against_the_module() {
    let data_uri = "http://example.com/style/data.xml";
    let xsl = r#"<xsl:stylesheet version="3.0" xmlns:xsl="http://www.w3.org/1999/XSL/Transform">
          <xsl:output omit-xml-declaration="yes"/>
          <xsl:template match="doc">
            <out><xsl:value-of select="document('data.xml')/data"/></out>
          </xsl:template>
        </xsl:stylesheet>"#;
    assert_eq!(
        text(run_module_source(
            xsl,
            &[(data_uri, "<data>loaded</data>")],
            "<doc/>"
        )),
        "<out>loaded</out>"
    );
}

/// An empty sequence of references is still the empty sequence: `document(())` retrieves
/// nothing rather than the stylesheet.
#[test]
fn document_of_an_empty_sequence_retrieves_nothing() {
    let xsl = r#"<xsl:stylesheet version="3.0" xmlns:xsl="http://www.w3.org/1999/XSL/Transform">
          <xsl:output omit-xml-declaration="yes"/>
          <xsl:template match="doc">
            <out><xsl:value-of select="count(document(()))"/></out>
          </xsl:template>
        </xsl:stylesheet>"#;
    assert_eq!(text(run_module_source(xsl, &[], "<doc/>")), "<out>0</out>");
}

// --- keys over documents loaded mid-transform ---------------------------------------------

/// `key-043`: "every time an additional source document is loaded with the XSLT `document()`
/// function, all keys are applied to it as well" — the key index is built per document, so a
/// lookup rooted in a tree loaded mid-transform has to index that tree.
#[test]
fn a_key_covers_a_document_loaded_during_the_transformation() {
    let data_uri = "http://example.com/style/data.xml";
    let xsl = r#"<xsl:stylesheet version="3.0" xmlns:xsl="http://www.w3.org/1999/XSL/Transform">
          <xsl:output omit-xml-declaration="yes"/>
          <xsl:key name="k" match="entry" use="@name"/>
          <xsl:template match="doc">
            <out>
              <xsl:for-each select="document('data.xml')">
                <xsl:value-of select="key('k', 'b')/@value"/>
              </xsl:for-each>
            </out>
          </xsl:template>
        </xsl:stylesheet>"#;
    assert_eq!(
        text(run_module_source(
            xsl,
            &[(
                data_uri,
                r#"<data><entry name="a" value="one"/><entry name="b" value="two"/></data>"#
            )],
            "<doc/>"
        )),
        "<out>two</out>"
    );
}

// --- `xsl:result-document`'s attribute value templates -------------------------------------

/// `result-document-0501`: every serialization parameter on `xsl:result-document` is an
/// attribute value template — XSLT 3.0 §23.1 — so `doctype-system="{doc/foo}"` writes the value
/// the expression produces, not the template as written.
#[test]
fn a_serialization_parameter_of_a_result_document_is_an_attribute_value_template() {
    let xsl = r#"<xsl:stylesheet version="3.0" xmlns:xsl="http://www.w3.org/1999/XSL/Transform">
          <xsl:template match="/">
            <xsl:result-document method="xml" doctype-system="{doc/foo}">
              <out>hello</out>
            </xsl:result-document>
          </xsl:template>
        </xsl:stylesheet>"#;
    assert!(
        text(run_module_source(xsl, &[], "<doc><foo>out.dtd</foo></doc>"))
            .contains("<!DOCTYPE out SYSTEM \"out.dtd\">"),
        "produced {}",
        text(run_module_source(xsl, &[], "<doc><foo>out.dtd</foo></doc>"))
    );
}

/// `result-document-1202` / `-1206`: XSLT 3.0 widened the boolean serialization parameters to the
/// whole `xs:boolean` lexical space, and the effective value of an attribute value template is
/// trimmed before it is read — the cases all produce `" true "`.
#[test]
fn a_boolean_serialization_parameter_reads_the_effective_value() {
    let xsl = r#"<xsl:stylesheet version="3.0" xmlns:xsl="http://www.w3.org/1999/XSL/Transform">
          <xsl:template match="/">
            <xsl:result-document method="xml" omit-xml-declaration="{doc/foo}">
              <out>hello</out>
            </xsl:result-document>
          </xsl:template>
        </xsl:stylesheet>"#;
    assert_eq!(
        text(run_module_source(xsl, &[], "<doc><foo> true </foo></doc>")),
        "<out>hello</out>"
    );
}

/// `result-document-0401`: `cdata-section-elements` is an attribute value template too, and the
/// `QName`s in its effective value are expanded against the namespaces in scope where the
/// instruction is written.
#[test]
fn cdata_section_elements_is_an_attribute_value_template() {
    let xsl = r#"<xsl:stylesheet version="3.0" xmlns:xsl="http://www.w3.org/1999/XSL/Transform"
              xmlns:my="http://www.mytest.example.org" exclude-result-prefixes="my">
          <xsl:output omit-xml-declaration="yes"/>
          <xsl:template match="/doc">
            <xsl:result-document method="xml" omit-xml-declaration="yes"
                  cdata-section-elements="{foo[1]} my:{elem}">
              <out>
                <item1>a &amp; b</item1>
                <item2>a &amp; b</item2>
                <item3 xmlns="http://www.mytest.example.org">a &amp; b</item3>
              </out>
            </xsl:result-document>
          </xsl:template>
        </xsl:stylesheet>"#;
    let produced = text(run_module_source(
        xsl,
        &[],
        "<doc><foo>item1</foo><elem>item3</elem></doc>",
    ));
    assert!(
        produced.contains("<item1><![CDATA[a & b]]></item1>"),
        "produced {produced}"
    );
    assert!(
        produced.contains("<item2>a &amp; b</item2>"),
        "produced {produced}"
    );
    assert!(
        produced.contains("<![CDATA[a & b]]></item3>"),
        "produced {produced}"
    );
}

/// `result-document-0286` / `-0287`: `format` is an attribute value template as well, so the
/// doubled braces of `Q{{http://example.com}}format2` are the escaped form of an `EQName` and
/// must be unescaped before the output definition is looked up.
#[test]
fn an_eqname_format_names_an_output_definition() {
    let xsl = r#"<xsl:stylesheet version="3.0" xmlns:xsl="http://www.w3.org/1999/XSL/Transform"
              xmlns:my="http://example.com">
          <xsl:output method="xml" omit-xml-declaration="yes"/>
          <xsl:output name="Q{http://example.com}format2" method="text"/>
          <xsl:template match="doc">
            <out><xsl:result-document href="second.out"
                    format="Q{{http://example.com}}format2">
              <xsl:text>plain</xsl:text>
            </xsl:result-document></out>
          </xsl:template>
        </xsl:stylesheet>"#;
    assert_eq!(
        text(run_module_source(xsl, &[], "<doc/>")),
        "<out xmlns:my=\"http://example.com\"/>"
    );
    let compiled = Stylesheet::compile_str(xsl).expect("the stylesheet compiles");
    let source = Tree::parse_str("<doc/>").expect("the source parses");
    let transformation = compiled
        .transform(&source.root())
        .expect("the transformation runs");
    assert_eq!(
        transformation.secondary_results(),
        [("second.out".to_owned(), "plain".to_owned())]
    );
}

/// `result-document-0285`: a `format` with an embedded expression names a different output
/// definition on each evaluation, so it can only be resolved when the instruction runs.
#[test]
fn a_format_with_an_embedded_expression_selects_the_definition_at_run_time() {
    let xsl = r#"<xsl:stylesheet version="3.0" xmlns:xsl="http://www.w3.org/1999/XSL/Transform">
          <xsl:output name="format1" method="xml" omit-xml-declaration="yes" indent="no"/>
          <xsl:output name="format2" method="xml" omit-xml-declaration="no" indent="no"/>
          <xsl:template match="/">
            <xsl:for-each select="1 to 2">
              <xsl:result-document href="out{.}.xml" format="format{.}">
                <out/>
              </xsl:result-document>
            </xsl:for-each>
          </xsl:template>
        </xsl:stylesheet>"#;
    let compiled = Stylesheet::compile_str(xsl).expect("the stylesheet compiles");
    let source = Tree::parse_str("<doc/>").expect("the source parses");
    let transformation = compiled
        .transform(&source.root())
        .expect("the transformation runs");
    let written = transformation.secondary_results().to_vec();
    assert_eq!(
        written,
        vec![
            ("out1.xml".to_owned(), "<out/>".to_owned()),
            (
                "out2.xml".to_owned(),
                "<?xml version=\"1.0\" encoding=\"UTF-8\"?><out/>".to_owned()
            ),
        ]
    );
}

/// `error-0290a` / `error-1460d`: the effective value of `format` that names no output
/// definition — or whose prefix is not bound — is `XTDE1460`, raised when the instruction runs.
#[test]
fn an_unknown_format_is_xtde1460_at_run_time() {
    let xsl = r#"<xsl:stylesheet version="3.0" xmlns:xsl="http://www.w3.org/1999/XSL/Transform">
          <xsl:param name="p" select="'your:qname'"/>
          <xsl:output name="xyz" method="xml"/>
          <xsl:template match="/">
            <out><xsl:result-document href="temp.out" format="{$p}"><apple/></xsl:result-document></out>
          </xsl:template>
        </xsl:stylesheet>"#;
    assert_eq!(
        error_code(run_module_source(xsl, &[], "<doc/>")),
        "XTDE1460"
    );
}

// --- `item-separator` ----------------------------------------------------------------------

/// `result-document-1408` / `-1409` / `-1410`: Serialization 3.1 §2 makes `item-separator` part
/// of sequence normalization, so it separates *every* adjacent pair of items — nodes as well as
/// atomic values — and replaces the single space §5.7.1 would otherwise put between adjacent
/// atomic values.
#[test]
fn an_item_separator_parts_every_adjacent_pair_of_items() {
    let xsl = r#"<xsl:stylesheet version="3.0" xmlns:xsl="http://www.w3.org/1999/XSL/Transform">
          <xsl:param name="twiddle" select="'~'"/>
          <xsl:template match="/">
            <xsl:result-document href="out.xml" method="xml" indent="no"
                  omit-xml-declaration="yes" item-separator="{$twiddle}">
              <xsl:comment>start</xsl:comment>
              <xsl:sequence select="11 to 13"/>
              <xsl:comment>end</xsl:comment>
            </xsl:result-document>
          </xsl:template>
        </xsl:stylesheet>"#;
    let compiled = Stylesheet::compile_str(xsl).expect("the stylesheet compiles");
    let source = Tree::parse_str("<doc/>").expect("the source parses");
    let transformation = compiled
        .transform(&source.root())
        .expect("the transformation runs");
    let written: Vec<String> = transformation
        .secondary_results()
        .iter()
        .map(|(_, text)| text.clone())
        .collect();
    assert_eq!(written, vec!["<!--start-->~11~12~13~<!--end-->".to_owned()]);
}

/// The negative control: with no `item-separator` declared, adjacent atomic values keep the
/// single space and adjacent nodes keep nothing at all.
#[test]
fn adjacent_items_are_unseparated_without_the_parameter() {
    let xsl = r#"<xsl:stylesheet version="3.0" xmlns:xsl="http://www.w3.org/1999/XSL/Transform">
          <xsl:template match="/">
            <xsl:result-document href="out.xml" method="xml" indent="no"
                  omit-xml-declaration="yes">
              <xsl:comment>start</xsl:comment>
              <xsl:sequence select="11 to 13"/>
              <xsl:comment>end</xsl:comment>
            </xsl:result-document>
          </xsl:template>
        </xsl:stylesheet>"#;
    let compiled = Stylesheet::compile_str(xsl).expect("the stylesheet compiles");
    let source = Tree::parse_str("<doc/>").expect("the source parses");
    let transformation = compiled
        .transform(&source.root())
        .expect("the transformation runs");
    let written: Vec<String> = transformation
        .secondary_results()
        .iter()
        .map(|(_, text)| text.clone())
        .collect();
    assert_eq!(written, vec!["<!--start-->11 12 13<!--end-->".to_owned()]);
}

// --- the serialization errors of the parameter set ------------------------------------------

/// The serialization errors Serialization 3.1 §3 raises before a character is written, each
/// under the `xslt30-test` case that states it.
#[test]
fn a_parameter_set_the_serializer_cannot_honour_is_reported() {
    let cases: &[(&str, &str, &str)] = &[
        // `output-0182`: `standalone` other than `omit` with several element children.
        (
            "SEPM0004",
            r#"<xsl:output method="xml" standalone="yes"/>"#,
            "<doc>a</doc><doc>b</doc>",
        ),
        // `output-0183`: `doctype-system` with several element children.
        (
            "SEPM0004",
            r#"<xsl:output method="xml" doctype-system="out.dtd"/>"#,
            "<doc>a</doc><doc>b</doc>",
        ),
        // `output-0184` / `-0185`: an encoding that denotes nothing.
        (
            "SESU0007",
            r#"<xsl:output method="html" encoding="XXX-xx"/>"#,
            "<doc>hello</doc>",
        ),
        (
            "SESU0007",
            r#"<xsl:output method="text" encoding="XXX-xx"/>"#,
            "<doc>hello</doc>",
        ),
        // `output-0186`: `standalone` beside an omitted XML declaration.
        (
            "SEPM0009",
            r#"<xsl:output method="xml" omit-xml-declaration="yes" standalone="yes"/>"#,
            "<doc>hello</doc>",
        ),
        // `output-0187`: `doctype-system` beside an omitted declaration under XML 1.1.
        (
            "SEPM0009",
            r#"<xsl:output method="xml" omit-xml-declaration="yes" version="1.1"
                  doctype-system="out.dtd"/>"#,
            "<doc>hello</doc>",
        ),
        // `output-0188`: XML 1.0 cannot undeclare a namespace.
        (
            "SEPM0010",
            r#"<xsl:output method="xml" undeclare-prefixes="yes" version="1.0"/>"#,
            "<doc>hello</doc>",
        ),
        // `output-0189` … `-0192`: an unrecognized normalization form, under every method.
        (
            "SESU0011",
            r#"<xsl:output method="xml" normalization-form="ABC"/>"#,
            "<doc>hello</doc>",
        ),
        (
            "SESU0011",
            r#"<xsl:output method="html" normalization-form="ABC"/>"#,
            "<doc>hello</doc>",
        ),
        (
            "SESU0011",
            r#"<xsl:output method="text" normalization-form="ABC"/>"#,
            "<doc>hello</doc>",
        ),
        // `output-0193`: `fully-normalized` is a documented gap, so it is unsupported too.
        (
            "SESU0011",
            r#"<xsl:output method="xml" normalization-form="fully-normalized"/>"#,
            "<doc>hello</doc>",
        ),
        // `output-0194`: no such HTML version.
        (
            "SESU0013",
            r#"<xsl:output method="html" version="0.0"/>"#,
            "<doc>hello</doc>",
        ),
        // `output-0195`: a C1 control character has no meaning in HTML 4. HTML 5 gives it a
        // numeric character reference, so only a stated version below 5 asks for the error.
        (
            "SERE0014",
            r#"<xsl:output method="html" version="4.0"/>"#,
            "<doc>&#x9f;</doc>",
        ),
        // `output-0196`: a `>` inside a processing instruction ends it in HTML syntax.
        (
            "SERE0015",
            r#"<xsl:output method="html"/>"#,
            r#"<doc><xsl:processing-instruction name="PI">&gt;</xsl:processing-instruction></doc>"#,
        ),
    ];
    for (code, output, body) in cases {
        let xsl = format!(
            r#"<xsl:stylesheet version="3.0" xmlns:xsl="http://www.w3.org/1999/XSL/Transform">
                 {output}
                 <xsl:template match="/">{body}</xsl:template>
               </xsl:stylesheet>"#
        );
        assert_eq!(
            error_code(run_module_source(&xsl, &[], "<in/>")),
            *code,
            "for {output}"
        );
    }
}

/// The negative controls: each condition is narrow, and the parameter sets beside it still
/// serialize.
#[test]
fn a_parameter_set_the_serializer_can_honour_is_left_alone() {
    let cases: &[&str] = &[
        // One element child, so `standalone` and `doctype-system` are both writable.
        r#"<xsl:output method="xml" standalone="yes" doctype-system="out.dtd"/>"#,
        // A legacy encoding is still a known one.
        r#"<xsl:output method="xml" encoding="iso-8859-1"/>"#,
        r#"<xsl:output method="xml" encoding="US-ASCII"/>"#,
        // The four Unicode normalization forms are implemented.
        r#"<xsl:output method="xml" normalization-form="NFC"/>"#,
        r#"<xsl:output method="xml" normalization-form="none"/>"#,
        // `undeclare-prefixes` is meaningful under XML 1.1.
        r#"<xsl:output method="xml" undeclare-prefixes="yes" version="1.1"/>"#,
        // The HTML and XHTML versions this serializer writes.
        r#"<xsl:output method="html" version="4.01"/>"#,
        r#"<xsl:output method="html" version="5.0"/>"#,
        r#"<xsl:output method="xhtml" version="1.0"/>"#,
    ];
    for output in cases {
        let xsl = format!(
            r#"<xsl:stylesheet version="3.0" xmlns:xsl="http://www.w3.org/1999/XSL/Transform">
                 {output}
                 <xsl:template match="/"><doc>hello</doc></xsl:template>
               </xsl:stylesheet>"#
        );
        let produced = text(run_module_source(&xsl, &[], "<in/>"));
        assert!(produced.contains("hello"), "for {output}: {produced}");
    }
}

// --- the static rules of a package manifest -------------------------------------------------

/// `error-3051a` / `error-3051b`: XSLT 3.0 §3.5.2 makes it a static error for a token in
/// `xsl:accept/@names` to name a component that the same `xsl:use-package` also overrides — a
/// rule that holds inside one manifest, so it is decided without the used package.
#[test]
fn accepting_a_component_the_same_manifest_overrides_is_xtse3051() {
    for manifest in [
        r#"<xsl:accept component="template" names="t-public" visibility="final"/>
           <xsl:override>
             <xsl:template name="t-public">overridden</xsl:template>
           </xsl:override>"#,
        r#"<xsl:override>
             <xsl:template name="t-public">overridden</xsl:template>
           </xsl:override>
           <xsl:accept component="template" names="t-final t-public" visibility="final"/>"#,
    ] {
        let xsl = format!(
            r#"<xsl:stylesheet version="3.0" xmlns:xsl="http://www.w3.org/1999/XSL/Transform">
                 <xsl:use-package name="http://example.com/base" package-version="1.0.0">
                   {manifest}
                 </xsl:use-package>
                 <xsl:template name="main"><out/></xsl:template>
               </xsl:stylesheet>"#
        );
        assert_eq!(
            error_code(run_module_source(&xsl, &[], "<doc/>")),
            "XTSE3051"
        );
    }
}

/// The negative controls: a wildcard is exempt, a different component kind does not collide, and
/// a manifest with no `xsl:override` is left to the packaging gap the crate documents.
#[test]
fn a_manifest_without_that_conflict_reports_the_packaging_gap() {
    for manifest in [
        r#"<xsl:accept component="template" names="*" visibility="public"/>
           <xsl:override>
             <xsl:template name="t-public">overridden</xsl:template>
           </xsl:override>"#,
        r#"<xsl:accept component="function" names="t-public" visibility="final"/>
           <xsl:override>
             <xsl:template name="t-public">overridden</xsl:template>
           </xsl:override>"#,
        r#"<xsl:accept component="template" names="t-public" visibility="final"/>"#,
    ] {
        let xsl = format!(
            r#"<xsl:stylesheet version="3.0" xmlns:xsl="http://www.w3.org/1999/XSL/Transform">
                 <xsl:use-package name="http://example.com/base" package-version="1.0.0">
                   {manifest}
                 </xsl:use-package>
                 <xsl:template name="main"><out/></xsl:template>
               </xsl:stylesheet>"#
        );
        assert_eq!(
            error_code(run_module_source(&xsl, &[], "<doc/>")),
            "OXIXML0001"
        );
    }
}

// --- `xsl:number` -----------------------------------------------------------------------

/// `number-0603`: XSLT 3.0 §12.3 lets a processor fall back on decimal notation when the
/// numbering sequence "does not extend far enough", and a hundred thousand `m`s is past that
/// point — so `100000` under `format="i"` comes back as digits.
#[test]
fn roman_numbering_falls_back_to_decimal_when_it_runs_out() {
    let xsl = r#"<xsl:stylesheet version="3.0" xmlns:xsl="http://www.w3.org/1999/XSL/Transform">
          <xsl:output omit-xml-declaration="yes"/>
          <xsl:template match="doc">
            <out><xsl:number value="13" format="i"/>;<xsl:number value="819" format="i"/>;<xsl:number
                value="1999" format="I"/>;<xsl:number value="100000" format="i"/></out>
          </xsl:template>
        </xsl:stylesheet>"#;
    assert_eq!(
        text(run_module_source(xsl, &[], "<doc/>")),
        "<out>xiii;dcccxix;MCMXCIX;100000</out>"
    );
}

/// `item-separator` is part of sequence normalization, so it acts only on the top level: inside
/// constructed element content adjacent atomic values keep the single space XSLT 3.0 §5.7.1 puts
/// between them, and get no separator.
#[test]
fn an_item_separator_does_not_reach_inside_constructed_content() {
    let xsl = r#"<xsl:stylesheet version="3.0" xmlns:xsl="http://www.w3.org/1999/XSL/Transform">
          <xsl:template match="/">
            <xsl:result-document href="out.xml" method="xml" indent="no"
                  omit-xml-declaration="yes" item-separator="~">
              <out><xsl:sequence select="1, 2"/></out>
              <out><xsl:sequence select="3, 4"/></out>
            </xsl:result-document>
          </xsl:template>
        </xsl:stylesheet>"#;
    let compiled = Stylesheet::compile_str(xsl).expect("the stylesheet compiles");
    let source = Tree::parse_str("<doc/>").expect("the source parses");
    let transformation = compiled
        .transform(&source.root())
        .expect("the transformation runs");
    let written: Vec<String> = transformation
        .secondary_results()
        .iter()
        .map(|(_, text)| text.clone())
        .collect();
    assert_eq!(written, vec!["<out>1 2</out>~<out>3 4</out>".to_owned()]);
}

// --- the parameters of the apply-templates entry point ---------------------------------------

/// `initial-mode-004`: XSLT 3.0 §2.3.3 lets the invocation supply the parameters of the rules
/// matched in the initial mode, tunnel ones among them, through a channel of its own.
#[test]
fn the_initial_mode_takes_the_parameters_the_invocation_supplies() {
    let xsl = r#"<xsl:stylesheet version="3.0" xmlns:xsl="http://www.w3.org/1999/XSL/Transform"
              xmlns:my="http://my.net/" exclude-result-prefixes="my">
          <xsl:output omit-xml-declaration="yes"/>
          <xsl:template match="doc">
            <xsl:param name="a"/>
            <xsl:param name="my:b" tunnel="yes"/>
            <out><doc/><xsl:value-of select="$a"/><xsl:text> </xsl:text>
              <xsl:apply-templates select="inner"/></out>
          </xsl:template>
          <xsl:template match="inner">
            <xsl:param name="my:b" tunnel="yes"/>
            <xsl:value-of select="$my:b"/>
          </xsl:template>
        </xsl:stylesheet>"#;
    let compiled = Stylesheet::compile_str(xsl).expect("the stylesheet compiles");
    let source = Tree::parse_str("<doc><inner/></doc>").expect("the source parses");
    let parameters = oxixml_xslt::Parameters::new().with_initial_mode_parameters([
        (
            oxixml_infoset::ExpandedName::new(None::<&str>, "a"),
            oxixml_xpath::XpathValue::from(oxixml_infoset::AtomicValue::integer(1234)),
            false,
        ),
        (
            oxixml_infoset::ExpandedName::new(Some("http://my.net/"), "b"),
            oxixml_xpath::XpathValue::from(oxixml_infoset::AtomicValue::integer(999)),
            true,
        ),
    ]);
    assert_eq!(
        compiled
            .transform_with(&source.root(), &parameters)
            .expect("the transformation runs")
            .into_string(),
        "<out><doc/>1234 999</out>"
    );
}

/// The negative control: with no parameters supplied, the same rules bind nothing, so the
/// channel is not a back door into every transformation.
#[test]
fn the_initial_mode_binds_nothing_when_the_invocation_supplies_nothing() {
    let xsl = r#"<xsl:stylesheet version="3.0" xmlns:xsl="http://www.w3.org/1999/XSL/Transform">
          <xsl:output omit-xml-declaration="yes"/>
          <xsl:template match="doc">
            <xsl:param name="a" select="'default'"/>
            <out><xsl:value-of select="$a"/></out>
          </xsl:template>
        </xsl:stylesheet>"#;
    assert_eq!(
        text(run_module_source(xsl, &[], "<doc/>")),
        "<out>default</out>"
    );
}

// --- the raw result of the initial-template entry point ---------------------------------------

/// `initial-template-004`: XSLT 3.0 §2.3.6 makes the result of a transformation a *sequence*,
/// and a template that declares `as="xs:decimal*"` returns eight typed decimals — which a result
/// tree flattens into the single text node `123.5 -18 12.6 …`. The raw result keeps them apart
/// and keeps their type.
#[test]
fn an_initial_template_that_declares_its_result_answers_a_sequence() {
    let xsl = r#"<xsl:stylesheet version="3.0" xmlns:xsl="http://www.w3.org/1999/XSL/Transform"
              xmlns:xs="http://www.w3.org/2001/XMLSchema">
          <xsl:output omit-xml-declaration="yes"/>
          <xsl:variable name="data" as="element()*">
            <a>123.5</a><a>-18</a><a>12.6</a><a>14.2</a>
            <a>123.60</a><a>-56.0</a><a>123.5</a><a>123.6</a>
          </xsl:variable>
          <xsl:template name="main" as="xs:decimal*">
            <xsl:sequence select="$data/self::a ! xs:decimal(.)"/>
          </xsl:template>
        </xsl:stylesheet>"#;
    let compiled = Stylesheet::compile_str(xsl).expect("the stylesheet compiles");
    let parameters = oxixml_xslt::Parameters::new().with_initial_template("main");
    let result = compiled
        .transform_optional(None, &parameters)
        .expect("the transformation runs");
    let sequence = result.as_sequence();
    assert_eq!(sequence.len(), 8);
    assert_eq!(
        sequence
            .iter()
            .filter_map(oxixml_infoset::Item::as_atomic)
            .map(|value| format!("{:?}:{}", value.atomic_type(), value.string_value()))
            .collect::<Vec<_>>()
            .join(" "),
        "Decimal:123.5 Decimal:-18 Decimal:12.6 Decimal:14.2 \
         Decimal:123.6 Decimal:-56 Decimal:123.5 Decimal:123.6"
    );
    // The serialization is the one the result tree gives, unchanged by the raw result being
    // recorded beside it.
    assert_eq!(result.as_str(), "123.5 -18 12.6 14.2 123.6 -56 123.5 123.6");
}

/// The control: an entry-point template that declares nothing is still a tree-producing entry
/// point, whose result sequence is the one item a caller can keep querying — the document node.
#[test]
fn an_initial_template_without_a_declared_result_answers_the_document_node() {
    let xsl = r#"<xsl:stylesheet version="3.0" xmlns:xsl="http://www.w3.org/1999/XSL/Transform">
          <xsl:output omit-xml-declaration="yes"/>
          <xsl:template name="main"><out>1</out><out>2</out></xsl:template>
        </xsl:stylesheet>"#;
    let compiled = Stylesheet::compile_str(xsl).expect("the stylesheet compiles");
    let parameters = oxixml_xslt::Parameters::new().with_initial_template("main");
    let result = compiled
        .transform_optional(None, &parameters)
        .expect("the transformation runs");
    let sequence = result.as_sequence();
    assert_eq!(sequence.len(), 1);
    assert_eq!(
        sequence
            .first()
            .and_then(oxixml_infoset::Item::as_node)
            .map(|node| format!("{:?}", oxixml_infoset::XdmNode::node_kind(node))),
        Some("Document".to_owned())
    );
    assert_eq!(result.as_str(), "<out>1</out><out>2</out>");
}

/// A declared result that is a node keeps the tree the transformation serializes identical to
/// the one instantiating straight into it would have built — the nodes reach the destination in
/// order, and the elements built around them are still nested.
#[test]
fn a_declared_node_result_builds_the_same_tree() {
    let xsl = r#"<xsl:stylesheet version="3.0" xmlns:xsl="http://www.w3.org/1999/XSL/Transform">
          <xsl:output omit-xml-declaration="yes"/>
          <xsl:template name="main" as="element()+">
            <out><in>a</in></out><out2/>
          </xsl:template>
        </xsl:stylesheet>"#;
    let compiled = Stylesheet::compile_str(xsl).expect("the stylesheet compiles");
    let parameters = oxixml_xslt::Parameters::new().with_initial_template("main");
    let result = compiled
        .transform_optional(None, &parameters)
        .expect("the transformation runs");
    assert_eq!(result.as_sequence().len(), 2);
    assert_eq!(result.as_str(), "<out><in>a</in></out><out2/>");
}

// --- the set of excluded namespaces -----------------------------------------------------------

/// `copy-1220`: `exclude-result-prefixes="#all"` on `xsl:stylesheet` excludes the namespaces in
/// scope *there* — XSLT 3.0 §11.1.3 — and not the ones an element further down declares for
/// itself, which the stylesheet element had never heard of.
#[test]
fn all_excludes_the_namespaces_in_scope_where_it_is_written() {
    let xsl = r##"<xsl:stylesheet version="3.0" xmlns:xsl="http://www.w3.org/1999/XSL/Transform"
              xmlns:gone="urn:gone" exclude-result-prefixes="#all">
          <xsl:output omit-xml-declaration="yes"/>
          <xsl:template name="main">
            <out xmlns:kept="urn:kept" xmlns:gone2="urn:gone"/>
          </xsl:template>
        </xsl:stylesheet>"##;
    let compiled = Stylesheet::compile_str(xsl).expect("the stylesheet compiles");
    let parameters = oxixml_xslt::Parameters::new().with_initial_template("main");
    assert_eq!(
        compiled
            .transform_optional(None, &parameters)
            .expect("the transformation runs")
            .into_string(),
        // `urn:kept` survives, and `urn:gone` does not — the second prefix bound to it is
        // excluded with the first, since what is excluded is the URI.
        r#"<out xmlns:kept="urn:kept"/>"#
    );
}

/// `lre-028`: the attribute lists prefixes but excludes *URIs*, and each prefix is resolved
/// where the attribute is written. A prefix rebound below the exclusion names a different URI
/// and survives; another prefix bound to an excluded URI goes with it.
#[test]
fn a_prefix_list_excludes_the_uris_bound_where_it_is_written() {
    let xsl = r#"<xsl:stylesheet version="3.0" xmlns:xsl="http://www.w3.org/1999/XSL/Transform"
              xmlns:one="http://one.com" xmlns:two="http://two.com"
              exclude-result-prefixes="one two">
          <xsl:output omit-xml-declaration="yes"/>
          <xsl:template name="main"
              xmlns:one="http://eins.com" xmlns:un="http://one.com"
              xmlns:four="http://vier.com">
            <out xsl:exclude-result-prefixes="four"/>
          </xsl:template>
        </xsl:stylesheet>"#;
    let compiled = Stylesheet::compile_str(xsl).expect("the stylesheet compiles");
    let parameters = oxixml_xslt::Parameters::new().with_initial_template("main");
    assert_eq!(
        compiled
            .transform_optional(None, &parameters)
            .expect("the transformation runs")
            .into_string(),
        // `one` now names `http://eins.com`, which nothing excluded; `un` names the excluded
        // `http://one.com` and goes; `four` is excluded on the element itself.
        r#"<out xmlns:one="http://eins.com"/>"#
    );
}

// --- stylesheet whitespace, XSLT 3.0 §4.2 -----------------------------------------------------

/// `choose-0604`: `xml:space` is inherited, so the `preserve` written on `xsl:choose` speaks for
/// the `xsl:when` and `xsl:if` nested inside it — and for nothing `xsl:choose` itself contains,
/// whose whitespace §4.2 strips whatever the attribute says.
#[test]
fn xml_space_reaches_the_elements_written_below_it() {
    let xsl = r#"<xsl:stylesheet xmlns:xsl="http://www.w3.org/1999/XSL/Transform" version="2.0">
<xsl:output omit-xml-declaration="yes"/>
<xsl:template match="doc">
  <out>
    <xsl:choose xml:space="preserve">
      <xsl:when test="2 &gt; 1">
        <xsl:if test="9 mod 3 = 0">
          <xsl:text>Test executed successfully.</xsl:text>
        </xsl:if>
      </xsl:when>
    </xsl:choose>
  </out>
</xsl:template>
</xsl:stylesheet>"#;
    assert_eq!(
        text(run_module_source(xsl, &[], "<doc/>")),
        "<out>\n        \n          Test executed successfully.\n        \n      </out>"
    );
}

/// The negative control, and `whitespace-001`/`-004`'s point: an `xml:space="preserve"` written
/// on one of the elements §4.2 lists changes nothing about *its own* whitespace children.
#[test]
fn xml_space_does_not_reach_the_children_of_the_elements_it_is_written_on() {
    let xsl = r#"<xsl:stylesheet xmlns:xsl="http://www.w3.org/1999/XSL/Transform" version="2.0">
<xsl:output omit-xml-declaration="yes"/>
<xsl:template match="doc">
  <out><xsl:apply-templates xml:space="preserve">
     </xsl:apply-templates></out>
</xsl:template>
<xsl:template match="text()"/>
</xsl:stylesheet>"#;
    assert_eq!(text(run_module_source(xsl, &[], "<doc/>")), "<out/>");
}

/// `context-item-019`: a template whose whole body is the whitespace after its
/// `xsl:context-item`. The whitespace *before* the declaration goes — §4.2 strips it "regardless
/// of any `xml:space` attributes" — and the whitespace after it stays.
#[test]
fn whitespace_around_a_declaration_is_stripped_and_the_body_is_kept() {
    let xsl = r#"<xsl:stylesheet version="3.0" xmlns:xsl="http://www.w3.org/1999/XSL/Transform">
    <xsl:output omit-xml-declaration="yes"/>
    <xsl:template match="/"><out><xsl:call-template name="ws"/></out></xsl:template>
    <xsl:template name="ws" xml:space="preserve">  <xsl:context-item as="document-node()"/>  </xsl:template>
  </xsl:stylesheet>"#;
    assert_eq!(text(run_module_source(xsl, &[], "<doc/>")), "<out>  </out>");
}

/// `whitespace-014`'s control: the same rule in front of an `xsl:param`, which also keeps the
/// parameter *reachable* — a preserved space before it would begin the body, and a parameter
/// after the body has begun is no longer one.
#[test]
fn whitespace_in_front_of_a_parameter_is_stripped() {
    let xsl = r#"<xsl:stylesheet version="3.0" xmlns:xsl="http://www.w3.org/1999/XSL/Transform">
    <xsl:output omit-xml-declaration="yes"/>
    <xsl:template match="/"><out><xsl:call-template name="p"><xsl:with-param name="v" select="'x'"/></xsl:call-template></out></xsl:template>
    <xsl:template name="p" xml:space="preserve">
<xsl:param name="v"/>[<xsl:value-of select="$v"/>]</xsl:template>
  </xsl:stylesheet>"#;
    assert_eq!(
        text(run_module_source(xsl, &[], "<doc/>")),
        "<out>[x]</out>"
    );
}

// --- what an attribute unprefixed on a literal result element means ---------------------------

/// `xpath-default-namespace-0104`: on an element that is not in the XSLT namespace the attribute
/// has to be written `xsl:xpath-default-namespace`. An unprefixed one belongs to the result
/// element, is copied to the output, and says nothing about the expressions inside it.
#[test]
fn xpath_default_namespace_on_a_result_element_needs_the_xslt_spelling() {
    let xsl = r#"<t:transform xmlns:t="http://www.w3.org/1999/XSL/Transform" version="2.0">
   <t:output omit-xml-declaration="yes"/>
   <t:template match="/">
      <output>
         <out xpath-default-namespace="http://example.org/hal"><t:value-of select="doc/item[1]/subitem"/><t:text>,</t:text><t:value-of select="doc/item/attribute::attr"/></out>
         <out t:xpath-default-namespace="http://example.org/hal"><t:value-of select="doc/child::item/@attr"/></out>
      </output>
   </t:template>
</t:transform>"#;
    let source = r#"<doc xmlns="http://example.org/hal"><item attr="atr1"><subitem>a</subitem></item><item attr="atr2"/><item attr="atr3"/></doc>"#;
    assert_eq!(
        text(run_module_source(xsl, &[], source)),
        concat!(
            r#"<output><out xpath-default-namespace="http://example.org/hal">,</out>"#,
            "<out>atr1 atr2 atr3</out></output>"
        )
    );
}

// --- pattern syntax and default priority ------------------------------------------------------

/// `match-261`: `Q{uri}*` is the braced spelling of `prefix:*` and carries its default priority
/// of -0.25, which puts it between the rules declaring -0.1 and -0.4.
#[test]
fn a_braced_wildcard_pattern_has_the_wildcard_priority() {
    let xsl = r##"<xsl:stylesheet version="3.0" xmlns:xsl="http://www.w3.org/1999/XSL/Transform"
      xmlns:xhtml="http://www.w3.org/1999/xhtml">
     <xsl:output omit-xml-declaration="yes"/>
     <xsl:template match="Q{http://www.w3.org/1999/xhtml}*" mode="#all">
         <wildcard><xsl:next-match/></wildcard>
     </xsl:template>
     <xsl:template match="xhtml:html" priority="-0.1">
         <minus-0.1><xsl:next-match/></minus-0.1>
     </xsl:template>
     <xsl:template match="xhtml:html" priority="-0.4">
         <minus-0.4><xsl:next-match/></minus-0.4>
     </xsl:template>
     <xsl:template match="xhtml:html" priority="-2.0">
         <minus-2.0/>
     </xsl:template>
     <xsl:template match="xhtml:html" priority="1.5">
         <plus1.5><xsl:next-match/></plus1.5>
     </xsl:template>
     <xsl:template match="/">
         <out><xsl:apply-templates select="//test/*"/></out>
     </xsl:template>
 </xsl:stylesheet>"##;
    let produced = text(run_module_source(
        xsl,
        &[],
        r#"<test><html xmlns="http://www.w3.org/1999/xhtml"/></test>"#,
    ));
    assert!(
        produced.contains(
            "<plus1.5><minus-0.1><wildcard><minus-0.4><minus-2.0/></minus-0.4></wildcard></minus-0.1></plus1.5>"
        ),
        "produced {produced}"
    );
}

/// `match-048`: `child::document-node()` is a legal pattern that can never match, since no
/// document node is the child of anything — while `document-node()` written on its own does
/// match one.
#[test]
fn a_document_node_test_on_the_written_child_axis_never_matches() {
    let explicit = r#"<xsl:stylesheet version="2.0" xmlns:xsl="http://www.w3.org/1999/XSL/Transform">
      <xsl:output omit-xml-declaration="yes"/>
      <xsl:template match="child::document-node()"><d>WRONG</d></xsl:template>
      <xsl:template match="*"><out>OK</out></xsl:template>
    </xsl:stylesheet>"#;
    let abbreviated = r#"<xsl:stylesheet version="2.0" xmlns:xsl="http://www.w3.org/1999/XSL/Transform">
      <xsl:output omit-xml-declaration="yes"/>
      <xsl:template match="document-node()"><d>RIGHT</d></xsl:template>
      <xsl:template match="*"><out>OK</out></xsl:template>
    </xsl:stylesheet>"#;
    assert_eq!(
        text(run_module_source(explicit, &[], "<doc/>")),
        "<out>OK</out>"
    );
    assert_eq!(
        text(run_module_source(abbreviated, &[], "<doc/>")),
        "<d>RIGHT</d>"
    );
}

/// `conflict-resolution-1601`: XSLT 3.0 §6.5 gives the pattern `/` the default priority -0.5, so
/// a rule declaring -0.4 outranks it and one declaring -0.6 does not.
#[test]
fn the_lone_slash_pattern_has_the_least_selective_priority() {
    let xsl = r#"<t:transform xmlns:t="http://www.w3.org/1999/XSL/Transform" version="2.0">
   <t:output method="xml" indent="no" omit-xml-declaration="yes"/>
   <t:template match="/"><out><t:value-of select="doc/a/@x5"/></out></t:template>
   <t:template match="/" priority="-0.4"><out><t:value-of select="doc/a/@x4"/></out></t:template>
   <t:template match="/" priority="-0.6"><out><t:value-of select="doc/a/@x6"/></out></t:template>
</t:transform>"#;
    assert_eq!(
        text(run_module_source(
            xsl,
            &[],
            r#"<doc><a x4="big" x5="medium" x6="small"/></doc>"#
        )),
        "<out>big</out>"
    );
}

/// `match-038`: XPath's leading-lone-slash rule reads `/ union /*` as the path `/union/*`,
/// since a name can begin a relative path and `/` is a complete expression only when the token
/// after it cannot. The rule therefore matches nothing here, and the one for `/` applies.
#[test]
fn a_keyword_after_a_lone_slash_is_a_name_and_not_an_operator() {
    let xsl = r#"<xsl:stylesheet xmlns:xsl="http://www.w3.org/1999/XSL/Transform" version="2.0">
<xsl:output omit-xml-declaration="yes"/>
<xsl:template match="/ union /*" priority="5"><out>WRONG!</out></xsl:template>
<xsl:template match="/"><out>OK!</out></xsl:template>
</xsl:stylesheet>"#;
    assert_eq!(
        text(run_module_source(xsl, &[], "<doc/>")),
        "<out>OK!</out>"
    );
    // The keyword is still an operator where an operator can stand.
    let union = r#"<xsl:stylesheet xmlns:xsl="http://www.w3.org/1999/XSL/Transform" version="2.0">
<xsl:output omit-xml-declaration="yes"/>
<xsl:template match="a union b"><hit><xsl:value-of select="name()"/></hit></xsl:template>
<xsl:template match="/"><out><xsl:apply-templates select="doc/*"/></out></xsl:template>
</xsl:stylesheet>"#;
    assert_eq!(
        text(run_module_source(union, &[], "<doc><a/><b/></doc>")),
        "<out><hit>a</hit><hit>b</hit></out>"
    );
}

/// `match-082a`: XSLT 3.0 §6.5 strips the outer parentheses of a top-level pattern and applies
/// its rules to what is inside, so `(doc|cod)` is two rules of priority 0 — the same priority as
/// the `doc` written beside it, which therefore wins by declaration order.
#[test]
fn parentheses_around_a_pattern_do_not_change_its_priority() {
    let xsl = r#"<xsl:stylesheet xmlns:xsl="http://www.w3.org/1999/XSL/Transform" version="3.0">
    <xsl:output omit-xml-declaration="yes"/>
    <xsl:template match="(doc|cod)"><wrong/></xsl:template>
    <xsl:template match="doc"><ok/></xsl:template>
    <xsl:template match="/"><xsl:apply-templates/></xsl:template>
</xsl:stylesheet>"#;
    assert_eq!(text(run_module_source(xsl, &[], "<doc/>")), "<ok/>");
    // Declared the other way round, the parenthesized rule is the later one and wins — the two
    // are ranked by declaration order, which only equal priorities allow.
    let reversed = r#"<xsl:stylesheet xmlns:xsl="http://www.w3.org/1999/XSL/Transform" version="3.0">
    <xsl:output omit-xml-declaration="yes"/>
    <xsl:template match="doc"><plain/></xsl:template>
    <xsl:template match="(doc|cod)"><parenthesized/></xsl:template>
    <xsl:template match="/"><xsl:apply-templates/></xsl:template>
</xsl:stylesheet>"#;
    assert_eq!(
        text(run_module_source(reversed, &[], "<doc/>")),
        "<parenthesized/>"
    );
}

/// `match-082c`: with the parentheses stripped the two rules tie, and a mode declaring
/// `on-multiple-match="fail"` reports the tie as `XTDE0540` instead of taking the last rule.
#[test]
fn a_tie_between_a_pattern_and_its_parenthesized_twin_is_xtde0540() {
    let xsl = r#"<xsl:stylesheet xmlns:xsl="http://www.w3.org/1999/XSL/Transform" version="3.0">
    <xsl:output omit-xml-declaration="yes"/>
    <xsl:mode on-multiple-match="fail"/>
    <xsl:template match="doc"><wrong/></xsl:template>
    <xsl:template match="(doc|cod)"><wrong/></xsl:template>
    <xsl:template match="/"><xsl:apply-templates/></xsl:template>
</xsl:stylesheet>"#;
    assert_eq!(
        error_code(run_module_source(xsl, &[], "<doc/>")),
        "XTDE0540"
    );
}

/// The result of the transformation is what reached the *principal destination*: an entry point
/// that declares `as` but writes its result through an href-less `xsl:result-document` has not
/// returned it, so the raw sequence is not reported in its place.
#[test]
fn a_redirected_principal_result_reports_no_raw_sequence() {
    let xsl = r#"<xsl:stylesheet version="3.0" xmlns:xsl="http://www.w3.org/1999/XSL/Transform"
              xmlns:xs="http://www.w3.org/2001/XMLSchema" exclude-result-prefixes="xs">
          <xsl:output omit-xml-declaration="yes"/>
          <xsl:template name="main" as="xs:decimal*">
            <xsl:result-document><doc/></xsl:result-document>
          </xsl:template>
        </xsl:stylesheet>"#;
    let compiled = Stylesheet::compile_str(xsl).expect("the stylesheet compiles");
    let parameters = oxixml_xslt::Parameters::new().with_initial_template("main");
    let result = compiled
        .transform_optional(None, &parameters)
        .expect("the transformation runs");
    assert_eq!(result.as_str(), "<doc/>");
    let sequence = result.as_sequence();
    assert_eq!(sequence.len(), 1);
    assert_eq!(
        sequence
            .first()
            .and_then(oxixml_infoset::Item::as_node)
            .map(|node| format!("{:?}", oxixml_infoset::XdmNode::node_kind(node))),
        Some("Document".to_owned())
    );
}
