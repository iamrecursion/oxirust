//! Regression tests for the compiler and the public API: the source positions an `xsl:catch`
//! body reports, the `json` and `adaptive` serialization methods `xsl:output` accepts, the
//! content model of `xsl:merge`, the braced wildcard in an `xsl:strip-space` element list, and
//! the visibility a mode declaration gives an entry point.
//!
//! Every test reproduces one W3C `xslt30-test` case with a minimal inline stylesheet, so each fix
//! is provable red-then-green without the separate `oxixml-testsuite` harness. The case
//! identifiers in the comments are the ones that harness reports.

use oxixml_infoset::{Tree, XdmNode};
use oxixml_xslt::{OutputMethod, Parameters, Stylesheet, XsltError};

/// Compiles `xsl` and runs it from the named initial template, returning the serialized result.
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

// --- `$err:line-number` and `$err:column-number` -------------------------------------------

/// The stylesheet `try-018` uses, reduced to the assertion this test makes: an `xsl:sequence`
/// whose `select` raises, and an `xsl:catch` that writes the position back out.
const TRY_018: &str = r#"<?xml version="1.0"?>
<xsl:stylesheet xmlns:xsl="http://www.w3.org/1999/XSL/Transform" version="3.0"
  xmlns:err="http://www.w3.org/2005/xqt-errors" exclude-result-prefixes="err">
  <xsl:template name="main">
    <out>
      <xsl:try>
        <xsl:sequence select="doc('try-018-rubbish.xml')"/>
        <xsl:catch errors="*">
          <line><xsl:value-of select="$err:line-number"/></line>
          <column><xsl:value-of select="$err:column-number"/></column>
        </xsl:catch>
      </xsl:try>
    </out>
  </xsl:template>
</xsl:stylesheet>"#;

/// `try-018`, `try-019`: the `xsl:sequence` whose `select` failed is on line 7 of the module
/// above — the same shape the suite asserts as line 11 of its own file — and its start tag opens
/// at column 9. [XSLT 3.0 §8.3](https://www.w3.org/TR/xslt-30/#element-catch) declares both
/// variables as `xs:integer?`, so before the compiler recorded provenance they were empty.
#[test]
fn a_catch_body_reads_the_position_of_the_instruction_that_failed() -> Result<(), XsltError> {
    let out = run_named(TRY_018, "main")?;
    assert!(
        out.contains("<line>7</line>"),
        "the line of the failing xsl:sequence: {out}"
    );
    assert!(
        out.contains("<column>9</column>"),
        "the column of the failing xsl:sequence: {out}"
    );
    Ok(())
}

/// `try-022`: the position is the *innermost* instruction's, not that of the construct that
/// happens to catch the failure. The `xsl:value-of` that divides by zero is on line 9; the
/// `xsl:result-document` containing it is on line 7, and the `xsl:try` on line 5.
#[test]
fn the_innermost_instruction_supplies_the_position() -> Result<(), XsltError> {
    let xsl = r#"<?xml version="1.0"?>
<xsl:stylesheet xmlns:xsl="http://www.w3.org/1999/XSL/Transform" version="3.0"
  xmlns:err="http://www.w3.org/2005/xqt-errors" exclude-result-prefixes="err">
  <xsl:template name="main">
    <xsl:try>
      <xsl:for-each select="1 to 2">
        <xsl:result-document href="q-inner-{position()}.out">
          <output>
            <xsl:value-of select="position() div 0"/>
          </output>
        </xsl:result-document>
      </xsl:for-each>
      <xsl:catch errors="*">
        <line><xsl:value-of select="$err:line-number"/></line>
      </xsl:catch>
    </xsl:try>
  </xsl:template>
</xsl:stylesheet>"#;
    let out = run_named(xsl, "main")?;
    assert!(
        out.contains("<line>9</line>"),
        "the line of the failing xsl:value-of: {out}"
    );
    Ok(())
}

/// `try-021`: an `xsl:result-document` that opens a second tree under a URI already written is
/// `XTDE1490`, and the position reported is that of the `xsl:result-document` itself — line 7
/// here — because no instruction inside it raised.
#[test]
fn a_duplicate_result_document_reports_its_own_position() -> Result<(), XsltError> {
    let xsl = r#"<?xml version="1.0"?>
<xsl:stylesheet xmlns:xsl="http://www.w3.org/1999/XSL/Transform" version="3.0"
  xmlns:err="http://www.w3.org/2005/xqt-errors" exclude-result-prefixes="err">
  <xsl:template name="main">
    <xsl:try>
      <xsl:for-each select="1 to 2">
        <xsl:result-document href="q-duplicate.out">
          <output>text</output>
        </xsl:result-document>
      </xsl:for-each>
      <xsl:catch errors="*">
        <err code="{$err:code}" line="{$err:line-number}"/>
      </xsl:catch>
    </xsl:try>
  </xsl:template>
</xsl:stylesheet>"#;
    let out = run_named(xsl, "main")?;
    assert!(
        out.contains(r#"code="err:XTDE1490""#),
        "the duplicate-URI error: {out}"
    );
    assert!(out.contains(r#"line="7""#), "its position: {out}");
    Ok(())
}

/// `try-002`: a failure this processor cannot place — here one raised by the `select` of the
/// `xsl:try` itself, which carries no provenance — still binds an *empty* sequence rather than a
/// sentinel, which the case accepts alongside the real line number.
#[test]
fn an_unplaceable_failure_binds_an_empty_position() -> Result<(), XsltError> {
    let xsl = r#"<?xml version="1.0"?>
<xsl:stylesheet xmlns:xsl="http://www.w3.org/1999/XSL/Transform" version="3.0"
  xmlns:err="http://www.w3.org/2005/xqt-errors" exclude-result-prefixes="err">
  <xsl:param name="a" select="0"/>
  <xsl:template name="main">
    <o>
      <xsl:try select="10 div $a">
        <xsl:catch errors="*">
          <err code="{$err:code}" line="{$err:line-number}"/>
        </xsl:catch>
      </xsl:try>
    </o>
  </xsl:template>
</xsl:stylesheet>"#;
    let out = run_named(xsl, "main")?;
    assert!(
        out.contains(r#"code="err:FOAR0001""#),
        "the division by zero: {out}"
    );
    assert!(out.contains(r#"line="""#), "no position at all: {out}");
    Ok(())
}

/// A stylesheet compiled from a tree the caller parsed without positions keeps the empty
/// sequence: the provenance is recorded by [`Stylesheet::compile_str`], not by the engine.
#[test]
fn a_tree_parsed_without_positions_reports_no_line() -> Result<(), XsltError> {
    let stylesheet = Tree::parse_str(TRY_018)
        .map_err(|error| XsltError::stat1c("XTSE0165", error.message().to_owned()))?;
    let compiled = Stylesheet::compile(&stylesheet.root())?;
    let source = Tree::parse_str("<doc/>")
        .map_err(|error| XsltError::dynamic("XTDE0000", error.message().to_owned()))?;
    let out = compiled
        .transform_with(
            &source.root(),
            &Parameters::new().with_initial_template("main"),
        )?
        .into_string();
    assert!(out.contains("<line/>"), "no position recorded: {out}");
    Ok(())
}

// --- the `json` and `adaptive` serialization methods ---------------------------------------

/// The `method` attribute of [XSLT 3.0 §26](https://www.w3.org/TR/xslt-30/#serialization) is an
/// `EQName`, and Serialization 3.1 defines `json` and `adaptive` alongside the four this
/// processor started with. `match-260` is the case that reached the compiler with
/// `method="json"` and was declined.
#[test]
fn the_json_and_adaptive_methods_are_accepted() {
    assert_eq!(OutputMethod::parse("json"), Some(OutputMethod::Json));
    assert_eq!(
        OutputMethod::parse("Q{}adaptive"),
        Some(OutputMethod::Adaptive)
    );
    for method in ["json", "adaptive"] {
        let xsl = format!(
            r#"<xsl:stylesheet xmlns:xsl="http://www.w3.org/1999/XSL/Transform" version="3.0">
                 <xsl:output method="{method}"/>
                 <xsl:template name="main"><xsl:sequence select="1"/></xsl:template>
               </xsl:stylesheet>"#
        );
        assert!(
            Stylesheet::compile_str(&xsl).is_ok(),
            "`method=\"{method}\"` must compile"
        );
    }
}

// --- the content model of `xsl:merge` -------------------------------------------------------

/// `merge-086`, `merge-007`, `merge-030`, `merge-035`: the content of `xsl:merge` is
/// `(xsl:merge-source+, xsl:merge-action, xsl:fallback*)`
/// ([XSLT 3.0 §15.1](https://www.w3.org/TR/xslt-30/#element-merge)), so an `xsl:fallback` before
/// the merge action, a second merge action, a missing merge source and a disallowed child are
/// each `XTSE0010`. Before the model was written down the misplaced `xsl:fallback` simply ran,
/// and the stylesheet compiled.
#[test]
fn a_malformed_merge_is_a_static_error() {
    let cases = [
        // `xsl:fallback` before the merge action — merge-086.
        r#"<xsl:merge>
             <xsl:merge-source select="."><xsl:merge-key select="@k"/></xsl:merge-source>
             <xsl:fallback>22</xsl:fallback>
             <xsl:merge-action><xsl:sequence select="1"/></xsl:merge-action>
           </xsl:merge>"#,
        // Two merge actions — merge-007.
        r#"<xsl:merge>
             <xsl:merge-source select="."><xsl:merge-key select="@k"/></xsl:merge-source>
             <xsl:merge-action><xsl:sequence select="1"/></xsl:merge-action>
             <xsl:merge-action><xsl:sequence select="2"/></xsl:merge-action>
           </xsl:merge>"#,
        // No merge source at all — merge-030.
        r#"<xsl:merge>
             <xsl:merge-action><xsl:sequence select="1"/></xsl:merge-action>
           </xsl:merge>"#,
        // A child the model does not allow — merge-035.
        r#"<xsl:merge>
             <xsl:merge-source select="."><xsl:merge-key select="@k"/></xsl:merge-source>
             <xsl:merge-action><xsl:sequence select="1"/></xsl:merge-action>
             <xsl:comment>no</xsl:comment>
           </xsl:merge>"#,
    ];
    for body in cases {
        let xsl = format!(
            r#"<xsl:stylesheet xmlns:xsl="http://www.w3.org/1999/XSL/Transform" version="3.0">
                 <xsl:template name="main">{body}</xsl:template>
               </xsl:stylesheet>"#
        );
        let outcome = Stylesheet::compile_str(&xsl);
        let code = outcome
            .err()
            .and_then(|error| error.code().map(str::to_owned));
        assert_eq!(code.as_deref(), Some("XTSE0010"), "for {body}");
    }
}

/// A well-formed `xsl:merge` is not rejected by the content model: this processor declines it
/// under its own code, which is what tells "not implemented" from "not valid".
#[test]
fn a_well_formed_merge_is_not_a_content_model_error() {
    let xsl = r#"<xsl:stylesheet xmlns:xsl="http://www.w3.org/1999/XSL/Transform" version="3.0">
                   <xsl:template name="main">
                     <xsl:merge>
                       <xsl:merge-source select="."><xsl:merge-key select="@k"/></xsl:merge-source>
                       <xsl:merge-action><xsl:sequence select="1"/></xsl:merge-action>
                       <xsl:fallback>22</xsl:fallback>
                     </xsl:merge>
                   </xsl:template>
                 </xsl:stylesheet>"#;
    let code = Stylesheet::compile_str(xsl)
        .err()
        .and_then(|error| error.code().map(str::to_owned));
    assert_ne!(code.as_deref(), Some("XTSE0010"));
}

// --- the braced wildcard in an element list --------------------------------------------------

/// `strip-space-027`: `elements="Q{uri}*"` is the `BracedURILiteral` wildcard of XPath 3.1's
/// `Wildcard` production, which [XSLT 3.0 §4.4](https://www.w3.org/TR/xslt-30/#stylesheet-stripping)
/// admits wherever `prefix:*` is allowed. It was read as a name whose local part is `*`, so
/// nothing was stripped.
#[test]
fn a_braced_wildcard_strips_a_whole_namespace() -> Result<(), XsltError> {
    let xsl = r#"<xsl:stylesheet xmlns:xsl="http://www.w3.org/1999/XSL/Transform" version="3.0"
                   xmlns:abc="http://abc.uri/">
                   <xsl:strip-space elements="Q{http://abc.uri/}*"/>
                   <xsl:preserve-space elements="Q{http://abc.uri/}w"/>
                   <xsl:template match="/"><xsl:copy-of select="."/></xsl:template>
                 </xsl:stylesheet>"#;
    let compiled = Stylesheet::compile_str(xsl)?;
    let source = Tree::parse_str(
        "<abc:doc xmlns:abc=\"http://abc.uri/\"><abc:w>   </abc:w><abc:x>   </abc:x></abc:doc>",
    )
    .map_err(|error| XsltError::dynamic("XTDE0000", error.message().to_owned()))?;
    let out = compiled.transform_to_string(&source.root())?;
    assert!(
        out.contains("<abc:w>   </abc:w>"),
        "the preserved element keeps its whitespace: {out}"
    );
    assert!(
        out.contains("<abc:x/>") || out.contains("<abc:x></abc:x>"),
        "the stripped element loses it: {out}"
    );
    Ok(())
}

// --- a private mode is not an entry point -----------------------------------------------------

/// `mode-1902`: an `xsl:mode` declaring `visibility="private"` at a higher import precedence than
/// the module that offered the mode makes it unusable as the initial mode —
/// [XSLT 3.0 §6.6.1](https://www.w3.org/TR/xslt-30/#declaring-modes), `XTDE0045`.
#[test]
fn a_private_mode_cannot_be_the_initial_mode() -> Result<(), XsltError> {
    let xsl = r#"<xsl:stylesheet xmlns:xsl="http://www.w3.org/1999/XSL/Transform" version="3.0">
                   <xsl:mode name="X" visibility="private"/>
                   <xsl:template match="doc" mode="X"><out/></xsl:template>
                 </xsl:stylesheet>"#;
    let compiled = Stylesheet::compile_str(xsl)?;
    let source = Tree::parse_str("<doc/>")
        .map_err(|error| XsltError::dynamic("XTDE0000", error.message().to_owned()))?;
    let outcome =
        compiled.transform_with(&source.root(), &Parameters::new().with_initial_mode("X"));
    assert_eq!(
        outcome
            .err()
            .and_then(|error| error.code().map(str::to_owned)),
        Some("XTDE0045".to_owned())
    );
    Ok(())
}

/// The converse, which the `initial-mode` half of the suite rests on: a mode nothing declares —
/// and one declared without a `visibility` attribute — is still an entry point.
#[test]
fn an_undeclared_mode_is_still_an_entry_point() -> Result<(), XsltError> {
    for declaration in [
        "",
        r#"<xsl:mode name="X"/>"#,
        r#"<xsl:mode name="X" visibility="public"/>"#,
    ] {
        let xsl = format!(
            r#"<xsl:stylesheet xmlns:xsl="http://www.w3.org/1999/XSL/Transform" version="3.0">
                 {declaration}
                 <xsl:template match="doc" mode="X"><out/></xsl:template>
               </xsl:stylesheet>"#
        );
        let compiled = Stylesheet::compile_str(&xsl)?;
        let source = Tree::parse_str("<doc/>")
            .map_err(|error| XsltError::dynamic("XTDE0000", error.message().to_owned()))?;
        let out = compiled
            .transform_with(&source.root(), &Parameters::new().with_initial_mode("X"))?
            .into_string();
        assert!(out.contains("<out/>"), "for `{declaration}`: {out}");
    }
    Ok(())
}

// --- a supplied context item the stripper removed ---------------------------------------------

/// `strip-space-023`: the invocation supplies a whitespace-only text node as the initial context
/// item, and `xsl:strip-space elements="*"` takes it out of the tree the stylesheet sees. Reading
/// `.` in a global variable is then `XPDY0002` — there is no context item — where before the
/// whole document stood in for the node that had been removed.
#[test]
fn a_stripped_context_item_is_absent() -> Result<(), XsltError> {
    let xsl = r#"<xsl:stylesheet xmlns:xsl="http://www.w3.org/1999/XSL/Transform" version="2.0">
                   <xsl:strip-space elements="*"/>
                   <xsl:variable name="initial-source" select="."/>
                   <xsl:template name="main">
                     <xsl:copy-of select="root($initial-source)"/>
                   </xsl:template>
                 </xsl:stylesheet>"#;
    let compiled = Stylesheet::compile_str(xsl)?;
    let source = Tree::parse_str("<a><b> </b><c> </c></a>")
        .map_err(|error| XsltError::dynamic("XTDE0000", error.message().to_owned()))?;
    let b = source
        .document_element()
        .and_then(|a| a.children().first().cloned())
        .ok_or_else(|| XsltError::dynamic("XTDE0000", "no `b` element"))?;
    let text = b
        .children()
        .first()
        .cloned()
        .ok_or_else(|| XsltError::dynamic("XTDE0000", "no text node"))?;
    let outcome = compiled.transform_with(&text, &Parameters::new().with_initial_template("main"));
    assert_eq!(
        outcome
            .err()
            .and_then(|error| error.code().map(str::to_owned)),
        Some("XPDY0002".to_owned())
    );
    Ok(())
}

/// A text node the stripper *keeps* is still the context item it was supplied as.
#[test]
fn a_preserved_context_item_survives_stripping() -> Result<(), XsltError> {
    let xsl = r#"<xsl:stylesheet xmlns:xsl="http://www.w3.org/1999/XSL/Transform" version="2.0">
                   <xsl:strip-space elements="*"/>
                   <xsl:preserve-space elements="b"/>
                   <xsl:variable name="initial-source" select="."/>
                   <xsl:template name="main">
                     <out><xsl:value-of select="name(root($initial-source)/*)"/></out>
                   </xsl:template>
                 </xsl:stylesheet>"#;
    let compiled = Stylesheet::compile_str(xsl)?;
    let source = Tree::parse_str("<a><b> </b><c> </c></a>")
        .map_err(|error| XsltError::dynamic("XTDE0000", error.message().to_owned()))?;
    let text = source
        .document_element()
        .and_then(|a| a.children().first().cloned())
        .and_then(|b| b.children().first().cloned())
        .ok_or_else(|| XsltError::dynamic("XTDE0000", "no text node"))?;
    let out = compiled
        .transform_with(&text, &Parameters::new().with_initial_template("main"))?
        .into_string();
    assert!(
        out.contains("<out>a</out>"),
        "the context item stands: {out}"
    );
    Ok(())
}

// --- `fn:id` in a count pattern ---------------------------------------------------------------

/// `number-4501`: `count="id('b d g')"` is a pattern anchored on a selection
/// ([XSLT 3.0 §5.5.3](https://www.w3.org/TR/xslt-30/#patterns-in-count)), and `xsl:number` counts
/// only the siblings it admits. The pattern machinery reads the call, and the identifiers come
/// from the `ID` attribute the DTD declares.
#[test]
fn a_count_pattern_may_be_an_id_call() -> Result<(), XsltError> {
    let source_text = r#"<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE iddata [<!ELEMENT iddata (el*)><!ELEMENT el EMPTY><!ATTLIST el id ID #REQUIRED>]>
<iddata><el id="a"/><el id="b"/><el id="c"/><el id="d"/><el id="e"/><el id="f"/><el id="g"/><el id="h"/></iddata>"#;
    let xsl = r#"<xsl:stylesheet xmlns:xsl="http://www.w3.org/1999/XSL/Transform" version="2.0">
                   <xsl:template match="iddata"><out><xsl:apply-templates/></out></xsl:template>
                   <xsl:template match="el">
                     <xsl:choose>
                       <xsl:when test="@id='b' or @id='d' or @id='g'">
                         <xsl:number level="single" from="iddata" count="id('b d g')" format="(1) "/>
                       </xsl:when>
                       <xsl:otherwise><xsl:text>(no) </xsl:text></xsl:otherwise>
                     </xsl:choose>
                     <xsl:value-of select="@id"/>
                   </xsl:template>
                 </xsl:stylesheet>"#;
    let source = Tree::parse_str(source_text)
        .map_err(|error| XsltError::dynamic("XTDE0000", error.message().to_owned()))?;
    let out = Stylesheet::compile_str(xsl)?.transform_to_string(&source.root())?;
    assert!(
        out.contains("<out>(no) a(1) b(no) c(2) d(no) e(no) f(3) g(no) h</out>"),
        "only the three identified elements are counted: {out}"
    );
    Ok(())
}

// --- XPath comments inside a match pattern ----------------------------------------------------

/// `match-215`: a pattern is XPath text, so [XPath 3.1 §2.6](https://www.w3.org/TR/xpath-31/#comments)
/// lets a `(: … :)` comment stand wherever whitespace may. Read as a parenthesized expression the
/// pattern still compiled — through the fallback that matches a pattern by *evaluating* it — and
/// that path answers `false` for a parentless element, because there is no parent to evaluate the
/// step from. Scanning the comment away makes it the plain name test it is, which matches the
/// `a` at the top of the temporary tree as well as the one inside it.
#[test]
fn a_pattern_may_carry_xpath_comments() -> Result<(), XsltError> {
    let xsl = r#"<?xml version="1.0" encoding="UTF-8"?>
<xsl:transform xmlns:xsl="http://www.w3.org/1999/XSL/Transform" version="3.0">
   <xsl:variable name="data" as="element()">
     <a nr="1"><a nr="2"/></a>
   </xsl:variable>
   <xsl:template match="/"><out><xsl:apply-templates select="$data"/></out></xsl:template>
   <xsl:template match="(: child-or-top:: :)a"><one nr="{@nr}"><xsl:apply-templates/></one></xsl:template>
   <xsl:template match="(: child-or-top:: :)a"><two nr="{@nr}"><xsl:next-match/></two></xsl:template>
   <xsl:template match="( */(: child:: :) a)"><three nr="{@nr}"><xsl:next-match/></three></xsl:template>
   <xsl:template match="//a"><wrong/></xsl:template>
</xsl:transform>"#;
    let source = Tree::parse_str("<doc/>")
        .map_err(|error| XsltError::dynamic("XTDE0000", error.message().to_owned()))?;
    let out = Stylesheet::compile_str(xsl)?.transform_to_string(&source.root())?;
    let expected = concat!(
        r#"<out><two nr="1"><one nr="1"><three nr="2"><two nr="2">"#,
        r#"<one nr="2"/></two></three></one></two></out>"#
    );
    assert!(
        out.contains(expected),
        "the comment is not part of the name test, and `//a` matches neither: {out}"
    );
    Ok(())
}

/// A comment that is never closed is not a comment: the pattern is malformed, and the scanner
/// leaves the text where it stands so the ordinary diagnosis names it.
#[test]
fn an_unterminated_comment_does_not_swallow_the_pattern() {
    let xsl = r#"<xsl:stylesheet xmlns:xsl="http://www.w3.org/1999/XSL/Transform" version="3.0">
                   <xsl:template match="(: never closed a"><hit/></xsl:template>
                 </xsl:stylesheet>"#;
    assert!(Stylesheet::compile_str(xsl).is_err());
}
