//! Regression tests for the unparsed-entity accessors, namespace inheritance in a result tree,
//! stylesheet-function memoization over nodes and the `xsl:catch` error information.
//!
//! Every test reproduces one W3C `xslt30-test` case with a minimal inline stylesheet, so each fix
//! is provable red-then-green without the separate `oxixml-testsuite` harness. The case
//! identifiers in the comments are the ones that harness reports.

use oxixml_infoset::{Tree, XdmNode};
use oxixml_xslt::{Stylesheet, XsltError, XsltResult};

/// Compiles `xsl` and transforms `xml`, returning the serialized result.
fn transform(xsl: &str, xml: &str) -> XsltResult<String> {
    let compiled = Stylesheet::compile_str(xsl)?;
    Ok(compiled.transform(&parse(xml)?.root())?.into_string())
}

fn parse(xml: &str) -> XsltResult<Tree> {
    Tree::parse_str(xml).map_err(|error| XsltError::dynamic("XTDE0000", error.message().to_owned()))
}

/// Parses `xml` as the document retrieved from `uri`, which is what gives its DTD's system
/// identifiers something to be resolved against.
fn parse_at(xml: &str, uri: &str) -> XsltResult<Tree> {
    let options = oxixml_infoset::ParseOptions::default().document_uri(uri.to_owned());
    Tree::parse_str_with(xml, options)
        .map_err(|error| XsltError::dynamic("XTDE0000", error.message().to_owned()))
}

/// Compiles `xsl` and transforms the document `xml` retrieved from `uri`.
fn transform_document_at(xsl: &str, xml: &str, uri: &str) -> XsltResult<String> {
    let compiled = Stylesheet::compile_str(xsl)?;
    Ok(compiled
        .transform(&parse_at(xml, uri)?.root())?
        .into_string())
}

/// A stack large enough for a stylesheet whose templates recurse through several nested
/// temporary trees; the conformance harness spawns its threads the same way.
const DEEP_STACK: usize = 64 * 1024 * 1024;

/// [`transform`] on a thread with room for a deep interpreter recursion.
fn transform_deep(xsl: &str, xml: &str) -> XsltResult<String> {
    let xsl = xsl.to_owned();
    let xml = xml.to_owned();
    std::thread::Builder::new()
        .stack_size(DEEP_STACK)
        .spawn(move || {
            let compiled = Stylesheet::compile_str(&xsl)?;
            let parameters = oxixml_xslt::Parameters::new().with_stack_size(DEEP_STACK);
            Ok(compiled
                .transform_with(&parse(&xml)?.root(), &parameters)?
                .into_string())
        })
        .expect("the deep-stack thread starts")
        .join()
        .expect("the deep-stack thread finishes")
}

/// Wraps `body` in a stylesheet element that omits the XML declaration.
fn stylesheet(body: &str) -> String {
    format!(
        r#"<xsl:stylesheet version="3.0" xmlns:xsl="http://www.w3.org/1999/XSL/Transform">
             <xsl:output method="xml" omit-xml-declaration="yes"/>
             {body}
           </xsl:stylesheet>"#
    )
}

/// Wraps `body` in a stylesheet element that also binds the `err:` prefix an `xsl:catch` body
/// reads its variables through, and keeps that prefix out of the result.
fn error_stylesheet(body: &str) -> String {
    format!(
        r#"<xsl:stylesheet version="3.0" xmlns:xsl="http://www.w3.org/1999/XSL/Transform"
             xmlns:err="http://www.w3.org/2005/xqt-errors" exclude-result-prefixes="err">
             <xsl:output method="xml" omit-xml-declaration="yes"/>
             {body}
           </xsl:stylesheet>"#
    )
}

/// The DTD of the suite's `unparsed-entity-A.xml`: two `NDATA` entities, one with a public
/// identifier and one without.
const ENTITY_SOURCE: &str = r#"<?xml version="1.0"?>
<!DOCTYPE doc [
  <!NOTATION gif SYSTEM "../www.foo.com" >
  <!ENTITY hatch-pic SYSTEM "../grafix/OpenHatch.gif" NDATA gif >
  <!ENTITY watch-pic PUBLIC "-//Textuality//TEXT standard boilerplate//EN"
                            "../grafix/OpenWatch.gif" NDATA gif >
  <!ELEMENT doc (#PCDATA)>
]>
<doc/>"#;

/// Reads all four unparsed-entity accessors of the context document in one pass.
const ACCESSOR_BODY: &str = r#"<xsl:template match="doc"><out>
     <a><xsl:value-of select="unparsed-entity-uri('hatch-pic')"/></a>
     <b><xsl:value-of select="unparsed-entity-public-id('hatch-pic')"/></b>
     <c><xsl:value-of select="unparsed-entity-uri('watch-pic')"/></c>
     <d><xsl:value-of select="unparsed-entity-public-id('watch-pic')"/></d>
   </out></xsl:template>"#;

/// `unparsed-entity-01`, `unparsed-entity-02`: the accessors report the DTD's `NDATA`
/// declarations, and the URI they report is the system identifier *resolved against the
/// document's own URI* — the absolute form XDM 3.1 §6.1 asks for, not the identifier as written.
#[test]
fn the_unparsed_entity_accessors_read_the_dtd() -> XsltResult<()> {
    let out = transform_document_at(
        &stylesheet(ACCESSOR_BODY),
        ENTITY_SOURCE,
        "http://example.com/tests/fn/unparsed-entity-uri/unparsed-entity-A.xml",
    )?;
    assert_eq!(
        out,
        "<out><a>http://example.com/tests/fn/grafix/OpenHatch.gif</a><b/>\
         <c>http://example.com/tests/fn/grafix/OpenWatch.gif</c>\
         <d>-//Textuality//TEXT standard boilerplate//EN</d></out>"
    );
    Ok(())
}

/// A document with no URI of its own leaves the system identifier nothing to be resolved
/// against, so it is reported as it was written rather than lost — which is what a caller that
/// knows the base itself can still use.
#[test]
fn a_document_without_a_uri_reports_the_system_identifier_as_written() -> XsltResult<()> {
    let out = transform(&stylesheet(ACCESSOR_BODY), ENTITY_SOURCE)?;
    assert_eq!(
        out,
        "<out><a>../grafix/OpenHatch.gif</a><b/><c>../grafix/OpenWatch.gif</c>\
         <d>-//Textuality//TEXT standard boilerplate//EN</d></out>"
    );
    Ok(())
}

/// The second argument of the two-argument form is declared `node()`, not `node()?`: writing one
/// that yields nothing must be reported, never read as "use the context node" — that would
/// answer silently from a different document.
#[test]
fn an_empty_second_argument_is_reported_rather_than_defaulted() -> XsltResult<()> {
    let failed = transform(
        &stylesheet(
            r#"<xsl:template match="doc"><out><xsl:value-of
                 select="unparsed-entity-uri('hatch-pic', ())"/></out></xsl:template>"#,
        ),
        ENTITY_SOURCE,
    );
    assert_eq!(
        failed
            .expect_err("an empty second argument is not a node")
            .code(),
        Some("XPTY0004")
    );
    Ok(())
}

/// `unparsed-entity-03`: the two-argument forms take the document from their second argument, so
/// they answer inside an `xsl:function`, where there is no context node at all.
#[test]
fn the_two_argument_accessors_take_the_document_from_the_argument() -> XsltResult<()> {
    let xsl = r#"<xsl:stylesheet version="3.0" xmlns:xsl="http://www.w3.org/1999/XSL/Transform"
                   xmlns:f="urn:f" exclude-result-prefixes="f">
                   <xsl:output method="xml" omit-xml-declaration="yes"/>
                   <xsl:template match="doc"><out><xsl:sequence select="f:f(.)"/></out></xsl:template>
                   <xsl:function name="f:f">
                     <xsl:param name="doc"/>
                     <b><xsl:value-of select="unparsed-entity-public-id('watch-pic', $doc)"/></b>
                   </xsl:function>
                 </xsl:stylesheet>"#;
    assert_eq!(
        transform(xsl, ENTITY_SOURCE)?,
        "<out><b>-//Textuality//TEXT standard boilerplate//EN</b></out>"
    );
    Ok(())
}

/// A name the document does not declare is the zero-length string, not an error — the negative
/// control that keeps the `XTDE1370` condition from swallowing an ordinary miss.
#[test]
fn an_undeclared_entity_name_is_the_empty_string() -> XsltResult<()> {
    let out = transform(
        &stylesheet(
            r#"<xsl:template match="doc"><out><xsl:value-of
                 select="concat('[', unparsed-entity-uri('nonesuch'), ']')"/></out></xsl:template>"#,
        ),
        ENTITY_SOURCE,
    )?;
    assert_eq!(out, "<out>[]</out>");
    Ok(())
}

/// `XTDE1370`/`XTDE1380`: a node whose tree is not rooted at a document node has no
/// `[unparsed entities]` property to read, however the node was reached.
#[test]
fn a_tree_without_a_document_node_is_xtde1370() -> XsltResult<()> {
    let uri = transform(
        &stylesheet(
            r#"<xsl:template match="doc">
                 <xsl:variable name="e" as="element()"><e/></xsl:variable>
                 <out><xsl:value-of select="unparsed-entity-uri('hatch-pic', $e)"/></out>
               </xsl:template>"#,
        ),
        ENTITY_SOURCE,
    );
    assert_eq!(
        uri.expect_err("a parentless element is not rooted at a document node")
            .code(),
        Some("XTDE1370")
    );

    let public = transform(
        &stylesheet(
            r#"<xsl:template match="doc">
                 <xsl:variable name="e" as="element()"><e/></xsl:variable>
                 <out><xsl:value-of select="unparsed-entity-public-id('hatch-pic', $e)"/></out>
               </xsl:template>"#,
        ),
        ENTITY_SOURCE,
    );
    assert_eq!(
        public
            .expect_err("the public-id accessor raises its own code")
            .code(),
        Some("XTDE1380")
    );
    Ok(())
}

/// The suite's `copy-0612` shape: a nest of `xsl:copy` instructions whose `inherit-namespaces`
/// is `inherit`, grafting a copy of a second temporary tree — whose `copy-namespaces` is
/// `copy` — under the innermost one. The result lists every element with its in-scope
/// namespaces, exactly as the case's `f:in-scope-namespaces` does.
fn namespace_inheritance(inherit: &str, copy: &str) -> XsltResult<String> {
    let xsl = format!(
        r#"<xsl:stylesheet version="3.0" xmlns:xsl="http://www.w3.org/1999/XSL/Transform"
             xmlns:xs="http://www.w3.org/2001/XMLSchema" exclude-result-prefixes="xs">
             <xsl:output method="xml" omit-xml-declaration="yes"/>
             <xsl:variable name="outer">
               <a xmlns="uri:a"><b:b xmlns:b="uri:b"><c xmlns="uri:c" xmlns:d="uri:d">
                 <graft xmlns=""/>
               </c></b:b></a>
             </xsl:variable>
             <xsl:variable name="inner">
               <p xmlns="uri:p"><q xmlns="uri:q"><r xmlns="uri:r" xmlns:s="uri:s"/></q></p>
             </xsl:variable>
             <xsl:mode on-no-match="shallow-copy"/>
             <xsl:template match="*">
               <xsl:copy inherit-namespaces="{inherit}"><xsl:apply-templates/></xsl:copy>
             </xsl:template>
             <xsl:template match="graft">
               <xsl:copy-of select="$inner" copy-namespaces="{copy}"/>
             </xsl:template>
             <xsl:template match="in">
               <xsl:variable name="capture"><xsl:apply-templates select="$outer"/></xsl:variable>
               <out>
                 <xsl:for-each select="$capture//*">
                   <element name="{{local-name()}}" in-scope="{{string-join(
                     for $prefix in sort(in-scope-prefixes(.)[. != 'xml'])
                       return concat('|', $prefix, '=', namespace-uri-for-prefix($prefix, .)))}}"/>
                 </xsl:for-each>
               </out>
             </xsl:template>
           </xsl:stylesheet>"#
    );
    transform_deep(&xsl, "<in/>")
}

/// `copy-0613`, `copy-0617`, `copy-0621`, `copy-0625`: with `inherit-namespaces="no"` the
/// elements written inside a constructed element's content start from an empty in-scope set, so
/// the grafted subtree carries only what it declares itself.
#[test]
fn inherit_namespaces_no_blocks_the_constructed_elements_namespaces() -> XsltResult<()> {
    let out = namespace_inheritance("no", "yes")?;
    assert!(
        out.contains(r#"<element name="p" in-scope="|=uri:p"/>"#),
        "{out}"
    );
    assert!(
        out.contains(r#"<element name="q" in-scope="|=uri:q"/>"#),
        "{out}"
    );
    assert!(
        out.contains(r#"<element name="r" in-scope="|=uri:r|s=uri:s"/>"#),
        "{out}"
    );
    // The elements the copies themselves carry are unaffected: `xsl:copy` copies the source
    // element's own namespace nodes whatever inheritance says.
    assert!(
        out.contains(r#"<element name="c" in-scope="|=uri:c|b=uri:b|d=uri:d"/>"#),
        "{out}"
    );
    Ok(())
}

/// `copy-0615`, `copy-0623`, `copy-0627`: the same, with the graft's own namespaces dropped by
/// `copy-namespaces="no"` — every element is then left with exactly what namespace fixup needs.
#[test]
fn blocked_inheritance_leaves_only_what_fixup_needs() -> XsltResult<()> {
    let out = namespace_inheritance("no", "no")?;
    for expected in [
        r#"<element name="p" in-scope="|=uri:p"/>"#,
        r#"<element name="q" in-scope="|=uri:q"/>"#,
        r#"<element name="r" in-scope="|=uri:r"/>"#,
    ] {
        assert!(out.contains(expected), "{expected} missing from {out}");
    }
    Ok(())
}

/// `copy-0614`: the negative control, and the reason the default must be `yes` — with
/// inheritance in force the grafted elements pick up the `b` and `d` bindings of the element
/// they are written into.
#[test]
fn inheritance_is_on_by_default() -> XsltResult<()> {
    let inherited = namespace_inheritance("yes", "no")?;
    for expected in [
        r#"<element name="p" in-scope="|=uri:p|b=uri:b|d=uri:d"/>"#,
        r#"<element name="q" in-scope="|=uri:q|b=uri:b|d=uri:d"/>"#,
        r#"<element name="r" in-scope="|=uri:r|b=uri:b|d=uri:d"/>"#,
    ] {
        assert!(
            inherited.contains(expected),
            "{expected} missing from {inherited}"
        );
    }
    Ok(())
}

/// `copy-4901`: a copy is spelled the way the original was. `<Part>` in the default namespace
/// stays `Part` even though a prefix is also bound to that namespace, and `<ns11:f>` keeps its
/// prefix — the prefix is part of the name in the data model, not a spelling fixup is free to
/// choose.
#[test]
fn a_copy_keeps_the_prefix_the_source_was_written_with() -> XsltResult<()> {
    let source = r#"<ns9:SomeRS xmlns="some1" xmlns:ns1="some1" xmlns:ns9="some9"
                      xmlns:ns11="some11">
                      <Part type="a"><s/></Part>
                      <ns11:f id="sm1"/>
                    </ns9:SomeRS>"#;
    let xsl = stylesheet(
        r#"<xsl:variable name="payload" select="/*[1]"/>
           <xsl:template match="/">
             <xsl:element name="{local-name($payload)}" namespace="{namespace-uri($payload)}">
               <xsl:copy-of select="$payload/*" copy-namespaces="no"/>
             </xsl:element>
           </xsl:template>"#,
    );
    let compiled = Stylesheet::compile_str(&xsl)?;
    let tree = parse(source)?;
    let result = compiled.transform(&tree.root())?;
    let out = result.into_string();
    // `name()` on the copies, read back through a second transformation, is what the case asserts.
    let names = transform(
        &stylesheet(
            r#"<xsl:template match="/"><names><xsl:for-each select="//*">
                 <n><xsl:value-of select="concat(name(), '|', namespace-uri())"/></n>
               </xsl:for-each></names></xsl:template>"#,
        ),
        &out,
    )?;
    assert!(names.contains("<n>Part|some1</n>"), "{names}");
    assert!(names.contains("<n>s|some1</n>"), "{names}");
    assert!(names.contains("<n>ns11:f|some11</n>"), "{names}");
    Ok(())
}

/// `copy-1221`: an element of a deep copy does not acquire a namespace merely because its
/// copied parent needed one. Under `copy-namespaces="no"` the copy of `<a a:att="A"><aa/></a>`
/// gives `a` the binding its attribute needs and `aa` only the binding its own name needs.
#[test]
fn a_copied_element_does_not_inherit_its_copied_parents_fixup() -> XsltResult<()> {
    let xsl = r#"<xsl:stylesheet version="3.0" xmlns:xsl="http://www.w3.org/1999/XSL/Transform"
                   xmlns:xs="http://www.w3.org/2001/XMLSchema" exclude-result-prefixes="xs">
                   <xsl:output method="xml" omit-xml-declaration="yes"/>
                   <xsl:variable name="fragment">
                     <wrapper>
                       <a xmlns:a="http://www.a.com/" a:att="A"><aa xmlns="http://www.aa.com/"/></a>
                       <a xmlns="http://www.a.com/"><aa xmlns:aa="http://www.aa.com/"/></a>
                     </wrapper>
                   </xsl:variable>
                   <xsl:template match="/">
                     <xsl:variable name="result">
                       <doc xmlns="http://www.out.com/">
                         <xsl:copy-of select="$fragment/wrapper/*" copy-namespaces="no"/>
                       </doc>
                     </xsl:variable>
                     <out>
                       <m><xsl:value-of select="string-join(sort(
                         $result/*/*[1]/namespace::*[name() ne 'xml']/(name()||'='||.||'; ')))"/></m>
                       <n><xsl:value-of select="string-join(sort(
                         $result/*/*[1]/*[1]/namespace::*[name() ne 'xml']/(name()||'='||.||'; ')))"/></n>
                       <o><xsl:value-of select="string-join(sort(
                         $result/*/*[2]/namespace::*[name() ne 'xml']/(name()||'='||.||'; ')))"/></o>
                       <p><xsl:value-of select="string-join(sort(
                         $result/*/*[2]/*[1]/namespace::*[name() ne 'xml']/(name()||'='||.||'; ')))"/></p>
                     </out>
                   </xsl:template>
                 </xsl:stylesheet>"#;
    let out = transform(xsl, "<in/>")?;
    assert_eq!(
        out,
        "<out><m>a=http://www.a.com/; </m><n>=http://www.aa.com/; </n>\
         <o>=http://www.a.com/; </o><p>=http://www.a.com/; </p></out>"
    );
    Ok(())
}

/// The negative control for the copy: under `copy-namespaces="yes"` every copied element keeps
/// the whole in-scope set it had in the source, so nothing is lost by not inheriting from the
/// copied parent.
#[test]
fn a_copy_keeps_the_sources_own_namespaces() -> XsltResult<()> {
    let xsl = r#"<xsl:stylesheet version="3.0" xmlns:xsl="http://www.w3.org/1999/XSL/Transform"
                   xmlns:xs="http://www.w3.org/2001/XMLSchema" exclude-result-prefixes="xs">
                   <xsl:output method="xml" omit-xml-declaration="yes"/>
                   <xsl:variable name="fragment">
                     <wrapper xmlns:w="http://www.w.com/">
                       <a xmlns:a="http://www.a.com/" a:att="A"><aa xmlns="http://www.aa.com/"/></a>
                     </wrapper>
                   </xsl:variable>
                   <xsl:template match="/">
                     <xsl:variable name="result">
                       <doc><xsl:copy-of select="$fragment/wrapper/*"/></doc>
                     </xsl:variable>
                     <out><xsl:value-of select="string-join(sort(
                       $result/*/*[1]/*[1]/namespace::*[name() ne 'xml']/(name()||'='||.||'; ')))"/></out>
                   </xsl:template>
                 </xsl:stylesheet>"#;
    assert_eq!(
        transform(xsl, "<in/>")?,
        "<out>=http://www.aa.com/; a=http://www.a.com/; w=http://www.w.com/; </out>"
    );
    Ok(())
}

/// `try-002`: `$err:line-number` and `$err:column-number` are `xs:integer?`, and this processor
/// records no position for a stylesheet node — so the value is the empty sequence, which an
/// attribute value template writes as nothing at all, and not a `-1` a reader would take for a
/// real line.
#[test]
fn an_unavailable_line_number_is_the_empty_sequence() -> XsltResult<()> {
    let out = transform(
        &stylesheet(
            r#"<xsl:template match="in">
                 <xsl:try select="10 div 0">
                   <xsl:catch errors="*">
                     <e line="{$err:line-number}" column="{$err:column-number}"
                        lines="{count($err:line-number)}"/>
                   </xsl:catch>
                 </xsl:try>
               </xsl:template>"#,
        ),
        "<in/>",
    )?;
    assert_eq!(out, r#"<e line="" column="" lines="0"/>"#);
    Ok(())
}

/// `message-0501`: the content of a terminating `xsl:message` reaches the catching `xsl:catch`
/// as `$err:value` — as *nodes*, so a comment inside the message survives.
#[test]
fn a_terminating_message_passes_its_content_as_the_error_value() -> XsltResult<()> {
    let out = transform(
        &error_stylesheet(
            r#"<xsl:template match="in">
                 <xsl:try>
                   <xsl:message terminate="yes">
                     <test>Take me to your leader<xsl:comment>REALLY</xsl:comment></test>
                   </xsl:message>
                   <xsl:catch>
                     <err>
                       <code><xsl:value-of select="$err:code"/></code>
                       <value><xsl:sequence select="$err:value"/></value>
                     </err>
                   </xsl:catch>
                 </xsl:try>
               </xsl:template>"#,
        ),
        "<in/>",
    )?;
    assert_eq!(
        out,
        "<err><code>err:XTMM9000</code><value><test>Take me to your leader<!--REALLY--></test>\
         </value></err>"
    );
    Ok(())
}

/// A condition the specification names on its own carries no value, so `$err:value` stays the
/// empty sequence — the negative control for the payload.
#[test]
fn a_specification_error_carries_no_value() -> XsltResult<()> {
    let out = transform(
        &error_stylesheet(
            r#"<xsl:template match="in">
                 <xsl:try select="10 div 0">
                   <xsl:catch><e values="{count($err:value)}"/></xsl:catch>
                 </xsl:try>
               </xsl:template>"#,
        ),
        "<in/>",
    )?;
    assert_eq!(out, r#"<e values="0"/>"#);
    Ok(())
}

/// `mode-1802`: the nodes an invocation supplies as its initial match selection are
/// space-stripped like every other document the transformation reads, so a path from one of them
/// back to the root sees the stripped tree.
#[test]
fn the_initial_match_selection_is_space_stripped() -> XsltResult<()> {
    let xsl = stylesheet(
        r#"<xsl:strip-space elements="*"/>
           <xsl:preserve-space elements="v"/>
           <xsl:mode name="m"/>
           <xsl:template match="v" mode="m">
             <chapter book="{/*/node()[2]}"><xsl:copy-of select="current()"/></chapter>
           </xsl:template>"#,
    );
    let compiled = Stylesheet::compile_str(&xsl)?;
    let source = parse(
        "<book>\n <long>The First Book</long>\n <short>Genesis</short>\n <v>one</v>\n</book>",
    )?;
    // The selection is taken from the *unstripped* tree, which is what a host supplying nodes
    // it selected itself always hands over.
    let selected: Vec<_> = source
        .root()
        .children()
        .into_iter()
        .flat_map(|book| book.children())
        .filter(|node| {
            node.node_name()
                .is_some_and(|name| name.local_name() == "v")
        })
        .map(oxixml_infoset::Item::Node)
        .collect();
    assert_eq!(selected.len(), 1);
    let parameters = oxixml_xslt::Parameters::new()
        .with_initial_mode("m")
        .with_initial_match_selection(oxixml_infoset::Sequence::from_items(selected));
    let out = compiled
        .transform_with(&source.root(), &parameters)?
        .into_string();
    assert_eq!(out, "<chapter book=\"Genesis\"><v>one</v></chapter>");
    Ok(())
}

/// The declarations survive `xsl:strip-space`, which replaces the source tree with a stripped
/// copy: the copy is still the same document and still carries its DTD's unparsed entities.
#[test]
fn stripping_whitespace_keeps_the_unparsed_entities() -> XsltResult<()> {
    let source = ENTITY_SOURCE.replace("<doc/>", "<doc>  <e/>  </doc>");
    let out = transform(
        &stylesheet(
            r#"<xsl:strip-space elements="*"/>
               <xsl:template match="doc"><out><xsl:value-of
                 select="unparsed-entity-public-id('watch-pic')"/></out></xsl:template>"#,
        ),
        &source,
    )?;
    assert_eq!(
        out,
        "<out>-//Textuality//TEXT standard boilerplate//EN</out>"
    );
    Ok(())
}

/// The stripped copy answers `fn:unparsed-entity-uri` from the document node alone, so a node
/// deep inside it reaches the declarations through its root.
#[test]
fn a_stripped_tree_answers_from_its_document_node() -> XsltResult<()> {
    let source = ENTITY_SOURCE.replace("<doc/>", "<doc>  <e/>  </doc>");
    let tree = parse(&source)?;
    assert!(tree.root().unparsed_entity("hatch-pic").is_some());
    let out = transform(
        &stylesheet(
            r#"<xsl:strip-space elements="*"/>
               <xsl:template match="e"><out><xsl:value-of
                 select="count(/)"/><xsl:value-of
                 select="unparsed-entity-public-id('watch-pic')"/></out></xsl:template>
               <xsl:template match="doc"><xsl:apply-templates/></xsl:template>"#,
        ),
        &source,
    )?;
    assert_eq!(
        out,
        "<out>1-//Textuality//TEXT standard boilerplate//EN</out>"
    );
    Ok(())
}
