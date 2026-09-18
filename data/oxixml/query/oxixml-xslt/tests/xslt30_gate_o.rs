//! Regression tests for the run and serialization tail: the properties an `xsl:copy` of a
//! document node carries, the `xhtml`/`html` escaping of URI and non-URI attributes, HTML 5
//! foreign-namespace prefix normalization, `xsl:number`'s language words and arbitrary-precision
//! counting, and the `err:` variables an `xsl:catch` binds.
//!
//! Every test reproduces one W3C `xslt30-test` case with a minimal inline stylesheet, so each fix
//! is provable red-then-green without the separate `oxixml-testsuite` harness. The case
//! identifiers in the comments are the ones that harness reports.

use oxixml_infoset::Tree;
use oxixml_xslt::{Stylesheet, XsltError};

/// Compiles `xsl` and runs it over `source`, returning the serialized result.
fn run(xsl: &str, source: &str) -> Result<String, XsltError> {
    let compiled = Stylesheet::compile_str(xsl)?;
    let tree = Tree::parse_str(source)
        .map_err(|error| XsltError::dynamic("XTDE0000", error.message().to_owned()))?;
    compiled.transform_to_string(&tree.root())
}

/// The transformation's text, or a readable stand-in for the error it raised.
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

// --- attribute escaping under the `xhtml` and `html` methods ------------------------------

/// The stylesheet `output-0102` and `output-0103` share, parameterized on
/// `escape-uri-attributes` and on which template the case starts at.
fn escaping_stylesheet(escape_uri_attributes: &str) -> String {
    format!(
        r#"<xsl:stylesheet version="2.0" xmlns:xsl="http://www.w3.org/1999/XSL/Transform"
             xmlns="http://www.w3.org/1999/xhtml">
           <xsl:output method="xhtml" encoding="UTF-8"
                       escape-uri-attributes="{escape_uri_attributes}" indent="no"/>
           <xsl:template match="/">
             <out>
               <a href='% %C2%96 &#x96; a "  &#xA1; &lt; &gt; &amp; end'>click here</a>
               <a accesskey="&#x96;"/>
               <a href="&#x96;"/>
               <a accesskey="&#xA1;"/>
               <a notxhtmlattr="&#xA1;"/>
             </out>
           </xsl:template>
         </xsl:stylesheet>"#
    )
}

/// `output-0102c`: with `escape-uri-attributes="yes"` a URI attribute is escaped by
/// [Appendix B.2.1 of HTML 4.0](https://www.w3.org/TR/xslt-xquery-serialization-31/#HTML_URI-ESCAPING),
/// which touches only the characters outside printable ASCII. The space, the percent sign and
/// the quotation mark the case writes come back unescaped, `#x96` and `#xA1` come back as UTF-8
/// octets, and the attribute escaper then spells the quotation mark `&#34;`.
#[test]
fn uri_attributes_escape_only_what_is_not_printable_ascii() {
    let out = text(run(&escaping_stylesheet("yes"), "<in/>"));
    assert!(
        out.contains(r#"href="% %C2%96 %C2%96 a &#34;  %C2%A1 &lt; &gt; &amp; end""#),
        "{out}"
    );
    // `output-0102a` / `-0102b`: a lone non-ASCII character is still escaped whole.
    assert!(out.contains(r#"href="%C2%96""#), "{out}");
}

/// `output-0102e`, `output-0103b`, `output-0103e`: a `C1` control in an attribute value is
/// written as a character reference under the `xhtml` method, exactly as it is in a text node.
/// It used to reach the output as a literal `#x96` octet pair, which HTML reads back as a
/// Windows-1252 character.
#[test]
fn c1_controls_in_attribute_values_become_character_references() {
    for escaping in ["yes", "no"] {
        let out = text(run(&escaping_stylesheet(escaping), "<in/>"));
        assert!(out.contains(r#"accesskey="&#150;""#), "{escaping}: {out}");
        assert!(!out.contains('\u{96}'), "{escaping}: {out}");
    }
    let out = text(run(&escaping_stylesheet("no"), "<in/>"));
    assert!(out.contains(r#"href="&#150;""#), "{out}");
}

/// `output-0103c`: with `escape-uri-attributes="no"` nothing is percent-encoded, so the same
/// value keeps its literal `#xA1` — which the encoding can carry — and turns only the `C1`
/// control and the markup delimiters into references.
#[test]
fn unescaped_uri_attributes_keep_every_representable_character() {
    let out = text(run(&escaping_stylesheet("no"), "<in/>"));
    assert!(
        out.contains("href=\"% %C2%96 &#150; a &#34;  \u{a1} &lt; &gt; &amp; end\""),
        "{out}"
    );
    // `output-0103d` / `-0103f`: a non-URI attribute is never escaped, whichever value it has.
    assert!(out.contains("accesskey=\"\u{a1}\""), "{out}");
    assert!(out.contains("notxhtmlattr=\"\u{a1}\""), "{out}");
}

// --- the properties a copy of a document node keeps ---------------------------------------

/// `unparsed-entity-06`: `<xsl:copy/>` applied to a document node builds a document node, and
/// [XDM 3.1 §6.1](https://www.w3.org/TR/xpath-datamodel-31/#DocumentNode) makes the unparsed
/// entities a property of that node — so `fn:unparsed-entity-uri` answers for the copy what it
/// answers for the original. `unparsed-entity-05` asks the same of `xsl:copy-of`, which already
/// carried them.
#[test]
fn a_document_copy_keeps_the_unparsed_entities() {
    let source = concat!(
        r#"<!DOCTYPE doc [<!NOTATION gif SYSTEM "image/gif">"#,
        r#"<!ENTITY pic SYSTEM "pic.gif" NDATA gif>]><doc/>"#
    );
    let body = |instruction: &str| {
        format!(
            r#"<xsl:stylesheet version="2.0" xmlns:xsl="http://www.w3.org/1999/XSL/Transform">
                 <xsl:template match="/">
                   <xsl:variable name="p" as="document-node()">{instruction}</xsl:variable>
                   <out><xsl:for-each select="$p">
                     <a><xsl:value-of select="ends-with(unparsed-entity-uri('pic'), 'pic.gif')"/></a>
                   </xsl:for-each></out>
                 </xsl:template>
               </xsl:stylesheet>"#
        )
    };
    let tree = Tree::parse_str(source).expect("the fixture parses");
    for instruction in ["<xsl:copy/>", r#"<xsl:copy-of select="."/>"#] {
        let compiled =
            Stylesheet::compile_str(&body(instruction)).expect("the stylesheet compiles");
        let out = text(compiled.transform_to_string(&tree.root()));
        assert!(out.contains("<a>true</a>"), "{instruction}: {out}");
    }
}

// --- HTML 5 prefix normalization ----------------------------------------------------------

/// The stylesheet `output-0601`: an HTML 5 serialization of a tree holding SVG, MathML and a
/// vocabulary that is neither.
const HTML5_COPY: &str = r#"<xsl:stylesheet version="3.0"
     xmlns:xsl="http://www.w3.org/1999/XSL/Transform">
   <xsl:output method="html" version="5.0" html-version="5.0" encoding="UTF-8" indent="no"/>
   <xsl:strip-space elements="*"/>
   <xsl:template match="/"><xsl:copy-of select="*"/></xsl:template>
 </xsl:stylesheet>"#;

/// `output-0602a`, `-0602b`, `-0602c`: the three vocabularies HTML 5 embeds without a prefix are
/// written under a default namespace declaration wherever they occur, and their namespace nodes
/// are removed from every other element — while a vocabulary HTML 5 has no unprefixed spelling
/// for keeps both its prefix and its declaration.
#[test]
fn html5_writes_the_embedded_vocabularies_without_a_prefix() {
    let source = r#"<html xmlns:n="NamespaceN">
          <body xmlns:svg="http://www.w3.org/2000/svg">
            <n:zzz>Foreign namespace</n:zzz>
            <svg:svg width="100"><svg:circle r="20"/></svg:svg>
            <mathML:math xmlns:mathML="http://www.w3.org/1998/Math/MathML">
              <mrow xmlns="http://www.w3.org/1998/Math/MathML"><mi>a</mi></mrow>
            </mathML:math>
          </body>
        </html>"#;
    let out = text(run(HTML5_COPY, source));
    assert!(!out.contains("xmlns:svg="), "{out}");
    assert!(!out.contains("xmlns:mathML="), "{out}");
    assert!(
        !out.contains(r#"<body xmlns="http://www.w3.org/2000/svg""#),
        "{out}"
    );
    assert!(
        out.contains(r#"<svg xmlns="http://www.w3.org/2000/svg" width="100">"#),
        "{out}"
    );
    assert!(
        out.contains(r#"<math xmlns="http://www.w3.org/1998/Math/MathML">"#),
        "{out}"
    );
    assert!(out.contains(r#"xmlns:n="NamespaceN""#), "{out}");
    assert!(out.contains("<n:zzz>"), "{out}");
}

/// `output-0603a`, `-0603b`, `-0603c`: an attribute name has no default namespace to fall back
/// on, so the element carrying one keeps — and, since prefix normalization took the binding off
/// its ancestors, re-declares — the prefix that attribute needs.
#[test]
fn a_prefixed_attribute_keeps_its_declaration_under_html5() {
    let source = r#"<html xmlns:n="NamespaceN" xmlns:m="NamespaceM">
          <body xmlns:svg="http://www.w3.org/2000/svg">
            <p m:zzz="value">Foreign attribute</p>
            <mathML:math xmlns:mathML="http://www.w3.org/1998/Math/MathML">
              <mrow xmlns="http://www.w3.org/1998/Math/MathML">
                <mi svg:att="34" z="5" mathML:att="123">a</mi>
              </mrow>
            </mathML:math>
          </body>
        </html>"#;
    let out = text(run(HTML5_COPY, source));
    assert!(
        out.contains(
            r#"<mi xmlns:svg="http://www.w3.org/2000/svg" xmlns:mathML="http://www.w3.org/1998/Math/MathML" svg:att="34""#
        ),
        "{out}"
    );
    assert!(out.contains("svg:att="), "{out}");
    assert!(out.contains("mathML:att="), "{out}");
    assert!(out.contains(r#"xmlns:m="NamespaceM""#), "{out}");
    assert!(out.contains(r#"<p m:zzz="value">"#), "{out}");
}

/// An HTML 4 serialization keeps every prefix the tree holds: before HTML 5 there was no way to
/// spell a foreign vocabulary in an HTML document, so there is nothing to normalize.
#[test]
fn html4_leaves_prefixes_alone() {
    let four = HTML5_COPY.replace("5.0", "4.0");
    let out = text(run(
        &four,
        r#"<html><body><svg:svg xmlns:svg="http://www.w3.org/2000/svg"/></body></html>"#,
    ));
    assert!(out.contains("<svg:svg>"), "{out}");
    assert!(!out.contains("xmlns"), "{out}");
}

// --- the XSLT extensions to the dynamic context are not captured by a function item ---------

/// `for-each-group-078b`, `-079b`, `error-1360b`: XSLT 3.0 §14.2.1 states that "the current
/// group is not retained as part of the closure of a function value … any invocation of this
/// function will fail with a dynamic error", and §18.2 says the same of `fn:current`. So a
/// reference such as `current-group#0` compiles and denotes a function item, and applying it —
/// even inside the very instruction that establishes the group — is the absence error.
#[test]
fn a_function_item_never_sees_the_xslt_dynamic_context() {
    let stylesheet = |reference: &str, call: &str| {
        format!(
            r#"<xsl:stylesheet version="3.0" xmlns:xsl="http://www.w3.org/1999/XSL/Transform">
                 <xsl:variable name="f" select="{reference}"/>
                 <xsl:template match="/">
                   <xsl:for-each-group select="1 to 10" group-by=". mod 2">
                     <a value="{{{call}}}"/>
                   </xsl:for-each-group>
                 </xsl:template>
               </xsl:stylesheet>"#
        )
    };
    for (reference, code) in [
        ("current-group#0", "XTDE1061"),
        ("current-grouping-key#0", "XTDE1071"),
        ("current#0", "XTDE1360"),
    ] {
        // Taken outside the instruction, as the suite cases write it.
        let outside = run(&stylesheet(reference, "$f()"), "<in/>");
        assert_eq!(
            outside
                .err()
                .and_then(|error| error.code().map(str::to_owned)),
            Some(code.to_owned()),
            "{reference} captured outside the group"
        );
        // And taken inside it, which the same sentence forbids just as plainly.
        let inside = run(&stylesheet("()", &format!("{reference}()")), "<in/>");
        assert_eq!(
            inside
                .err()
                .and_then(|error| error.code().map(str::to_owned)),
            Some(code.to_owned()),
            "{reference} captured inside the group"
        );
    }
}

/// The counterpart: a call that names the function is unaffected, however deeply it is nested —
/// `current()` in a predicate is the case XSLT 1.0 was written around.
#[test]
fn a_named_call_still_sees_the_xslt_dynamic_context() {
    let xsl = r#"<xsl:stylesheet version="3.0" xmlns:xsl="http://www.w3.org/1999/XSL/Transform">
         <xsl:template match="/">
           <out>
             <xsl:for-each-group select="1 to 4" group-by=". mod 2">
               <g key="{current-grouping-key()}" n="{count(current-group())}"/>
             </xsl:for-each-group>
             <xsl:for-each select="/in/a">
               <c><xsl:value-of select="/in/b[@id = current()/@ref]/@name"/></c>
             </xsl:for-each>
           </out>
         </xsl:template>
       </xsl:stylesheet>"#;
    let out = text(run(xsl, r#"<in><a ref="x"/><b id="x" name="found"/></in>"#));
    assert!(out.contains(r#"<g key="1" n="2"/>"#), "{out}");
    assert!(out.contains(r#"<g key="0" n="2"/>"#), "{out}");
    assert!(out.contains("<c>found</c>"), "{out}");
}

// --- the `from` boundary of `xsl:number` ---------------------------------------------------

/// `number-3227` and `number-4001`: [XSLT 3.0 §12.3](https://www.w3.org/TR/xslt-30/#number-level-single)
/// keeps the counted node when the `from` node is an **ancestor-or-self** of it, so a node that
/// matches both patterns numbers itself instead of being cut off by its own boundary.
#[test]
fn a_node_matching_both_count_and_from_is_still_numbered() {
    let source = "<doc><a><t/><b><t/><c><t/><d><t/></d></c></b></a><a><t/></a></doc>";
    let single = r#"<xsl:stylesheet version="2.0" xmlns:xsl="http://www.w3.org/1999/XSL/Transform">
         <xsl:template match="doc"><out><xsl:apply-templates select="//t"/></out></xsl:template>
         <xsl:template match="t">
           <n><xsl:number level="single" from="a" count="a|b|c|d" format="1"/></n>
         </xsl:template>
       </xsl:stylesheet>"#;
    // The `t` in each `a` numbers that `a`; the deeper ones number their own nearest counted
    // ancestor, all of which lie inside the same `a`.
    assert!(
        text(run(single, source)).ends_with("<out><n>1</n><n>1</n><n>1</n><n>1</n><n>2</n></out>"),
        "{}",
        text(run(single, source))
    );
    let multiple = single.replace(r#"level="single""#, r#"level="multiple""#);
    assert!(
        text(run(&multiple, source))
            .ends_with("<out><n>1</n><n>1.1</n><n>1.1.1</n><n>1.1.1.1</n><n>2</n></out>"),
        "{}",
        text(run(&multiple, source))
    );
    // With `from` naming a level below the counted one, the boundary still cuts: the `b` above
    // a `from="c"` node is outside the subtree rooted at it.
    let bounded = single
        .replace(r#"level="single""#, r#"level="multiple""#)
        .replace(r#"from="a""#, r#"from="a|c""#);
    assert!(
        text(run(&bounded, source))
            .ends_with("<out><n>1</n><n>1.1</n><n>1</n><n>1.1</n><n>2</n></out>"),
        "{}",
        text(run(&bounded, source))
    );
}
