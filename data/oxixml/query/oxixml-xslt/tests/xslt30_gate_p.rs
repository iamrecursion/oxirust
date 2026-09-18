//! Regression tests for the template-dispatch, key, sort and serialization tail: the built-in
//! template rules for items that are not document nodes, `xsl:with-param` with no value at all,
//! the subtree the three-argument `fn:key` searches, sort keys built by a sequence constructor,
//! `xsl:break` inside an `xsl:try`, `xml:id` in a temporary tree, `disable-output-escaping`
//! inside a CDATA section element, and the static context the availability functions expand a
//! name against.
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

// --- the built-in template rules (XSLT 3.0 §6.7) ------------------------------------------

/// `mode-1437a` and `next-match-034`: under `on-no-match="deep-skip"` the built-in rule for a
/// *document* node still applies the templates to its children, so a rule written for the
/// document element is reached.
#[test]
fn deep_skip_still_processes_the_document_node() -> Result<(), XsltError> {
    let xsl = r#"<xsl:stylesheet version="3.0" xmlns:xsl="http://www.w3.org/1999/XSL/Transform">
         <xsl:mode on-no-match="deep-skip"/>
         <xsl:template match="doc"><out><xsl:apply-templates select="*"/></out></xsl:template>
         <xsl:template match="tag"><tag/></xsl:template>
       </xsl:stylesheet>"#;
    assert_eq!(
        run(xsl, "<doc><tag/><bag/></doc>")?,
        "<?xml version=\"1.0\" encoding=\"UTF-8\"?><out><tag/></out>"
    );
    Ok(())
}

/// `match-256`: the same policy does *nothing* for every other item, an atomic value included —
/// the case applies templates to `93` and asserts that the result is empty.
#[test]
fn deep_skip_writes_nothing_for_an_atomic_value() -> Result<(), XsltError> {
    let xsl = r#"<xsl:stylesheet version="3.0" xmlns:xsl="http://www.w3.org/1999/XSL/Transform">
         <xsl:mode on-no-match="deep-skip"/>
         <xsl:template match="/"><out><xsl:apply-templates select="93"/></out></xsl:template>
       </xsl:stylesheet>"#;
    assert_eq!(
        run(xsl, "<in/>")?,
        "<?xml version=\"1.0\" encoding=\"UTF-8\"?><out/>"
    );
    Ok(())
}

/// The negative control: the default rules — and `shallow-skip`, which shares the "empty for
/// everything else" half — still write an atomic value's string as `text-only-copy` prescribes.
#[test]
fn the_default_rules_still_copy_an_atomic_value() -> Result<(), XsltError> {
    let default_rules = r#"<xsl:stylesheet version="3.0" xmlns:xsl="http://www.w3.org/1999/XSL/Transform">
         <xsl:template match="/"><out><xsl:apply-templates select="93"/></out></xsl:template>
       </xsl:stylesheet>"#;
    assert!(run(default_rules, "<in/>")?.ends_with("<out>93</out>"));
    let shallow_skip = r#"<xsl:stylesheet version="3.0" xmlns:xsl="http://www.w3.org/1999/XSL/Transform">
         <xsl:mode on-no-match="shallow-skip"/>
         <xsl:template match="/"><out><xsl:apply-templates select="93"/></out></xsl:template>
       </xsl:stylesheet>"#;
    assert!(run(shallow_skip, "<in/>")?.ends_with("<out/>"));
    Ok(())
}

/// `built-in-templates-0201`: XSLT 3.0 §9.3 makes `<xsl:with-param name="p"/>` — neither a
/// `select` nor content, and no `as` — supply a zero-length *string*, not the empty sequence.
#[test]
fn a_with_param_with_no_value_supplies_a_zero_length_string() -> Result<(), XsltError> {
    let xsl = r#"<xsl:stylesheet version="3.0" xmlns:xsl="http://www.w3.org/1999/XSL/Transform"
             xmlns:xs="http://www.w3.org/2001/XMLSchema">
         <xsl:template match="/">
           <out><xsl:call-template name="t"><xsl:with-param name="p"/>
             <xsl:with-param name="q" as="xs:string?"/></xsl:call-template></out>
         </xsl:template>
         <xsl:template name="t">
           <xsl:param name="p"/>
           <xsl:param name="q"/>
           <p><xsl:value-of select="$p eq ''"/></p>
           <q><xsl:value-of select="count($q)"/></q>
         </xsl:template>
       </xsl:stylesheet>"#;
    assert!(
        run(xsl, "<in/>")?.ends_with("><p>true</p><q>0</q></out>"),
        "the parameter with no `as` is a zero-length string, the one with `as` is empty"
    );
    Ok(())
}

// --- fn:key (XSLT 3.0 §20.2.2) -------------------------------------------------------------

/// `key-002` and `number-1401`: the third argument names the *selected subtree*, so a lookup
/// answers only the nodes below it — not every node of the document.
#[test]
fn the_third_argument_of_fn_key_names_the_subtree_searched() -> Result<(), XsltError> {
    let xsl = r#"<xsl:stylesheet version="3.0" xmlns:xsl="http://www.w3.org/1999/XSL/Transform">
         <xsl:key name="town" match="town" use="@name"/>
         <xsl:template match="/">
           <out>
             <both><xsl:value-of select="count(key('town', 'Enfield'))"/></both>
             <one><xsl:value-of select="key('town', 'Enfield', /doc/state[@code='CT'])/@code"/></one>
           </out>
         </xsl:template>
       </xsl:stylesheet>"#;
    let source = r#"<doc><state code="CT"><town name="Enfield" code="ct"/></state>
                         <state code="NH"><town name="Enfield" code="nh"/></state></doc>"#;
    assert!(
        run(xsl, source)?.ends_with("<out><both>2</both><one>ct</one></out>"),
        "the two-argument form searches the whole document, the three-argument form one state"
    );
    Ok(())
}

/// `key-077`: `xml:id` is an ID wherever it occurs, so `fn:id` finds it in a tree the
/// stylesheet built rather than one a parser did.
#[test]
fn xml_id_is_an_id_in_a_temporary_tree() -> Result<(), XsltError> {
    let xsl = r#"<xsl:stylesheet version="3.0" xmlns:xsl="http://www.w3.org/1999/XSL/Transform">
         <xsl:variable name="lookup">
           <code xml:id="a">Apple</code>
           <code xml:id="d">Damson</code>
         </xsl:variable>
         <xsl:template match="/"><out><xsl:value-of select="id('d', $lookup)"/></out></xsl:template>
       </xsl:stylesheet>"#;
    assert!(run(xsl, "<in/>")?.ends_with("<out>Damson</out>"));
    Ok(())
}

// --- xsl:sort (XSLT 3.0 §13.1) --------------------------------------------------------------

/// `sort-051`: a sort key written as a sequence constructor is *atomized*, so numbers sort as
/// numbers. Compared as strings, `803.23` would come before `99`.
#[test]
fn a_sort_key_from_a_sequence_constructor_is_atomized() -> Result<(), XsltError> {
    let xsl = r#"<xsl:stylesheet version="3.0" xmlns:xsl="http://www.w3.org/1999/XSL/Transform">
         <xsl:template match="/">
           <out><xsl:for-each select="doc/num">
             <xsl:sort order="descending"><xsl:sequence select="number(.)"/></xsl:sort>
             <xsl:value-of select="."/><xsl:text> </xsl:text>
           </xsl:for-each></out>
         </xsl:template>
       </xsl:stylesheet>"#;
    assert!(
        run(
            xsl,
            "<doc><num>99</num><num>803.23</num><num>82</num></doc>"
        )?
        .ends_with("<out>803.23 99 82 </out>"),
        "the key is the number the constructor produced, not its string"
    );
    Ok(())
}

// --- xsl:iterate (XSLT 3.0 §8.8) ------------------------------------------------------------

/// `iterate-035`: `xsl:try` is not a barrier to `xsl:break` — the iteration ends where the
/// break is reached, with what the body produced before it kept.
#[test]
fn an_xsl_break_inside_an_xsl_try_ends_the_iteration() -> Result<(), XsltError> {
    let xsl = r#"<xsl:stylesheet version="3.0" xmlns:xsl="http://www.w3.org/1999/XSL/Transform"
             xmlns:err="http://www.w3.org/2005/xqt-errors">
         <xsl:template match="/">
           <out><xsl:iterate select="1 to 10">
             <xsl:choose>
               <xsl:when test=". = 3">
                 <xsl:try>
                   <pos><xsl:value-of select="."/></pos>
                   <xsl:break/>
                   <xsl:catch errors="err:FOAR0001"><continuing/></xsl:catch>
                 </xsl:try>
               </xsl:when>
               <xsl:otherwise><p/></xsl:otherwise>
             </xsl:choose>
           </xsl:iterate></out>
         </xsl:template>
       </xsl:stylesheet>"#;
    assert!(
        run(xsl, "<in/>")?.ends_with("><p/><p/><pos>3</pos></out>"),
        "the iteration stops at the break rather than running to ten"
    );
    Ok(())
}

// --- serialization -------------------------------------------------------------------------

/// `bug-3801`: `cdata-section-elements` passes over text written with
/// `disable-output-escaping="yes"`, which is already-serialized markup that a CDATA section
/// could only turn back into character data.
#[test]
fn disabled_output_escaping_is_not_wrapped_in_cdata() -> Result<(), XsltError> {
    let xsl = r#"<xsl:stylesheet version="3.0" xmlns:xsl="http://www.w3.org/1999/XSL/Transform">
         <xsl:output method="xml" omit-xml-declaration="yes" cdata-section-elements="t"/>
         <xsl:template match="/">
           <test><t>plain</t><t><xsl:text disable-output-escaping="yes">&lt;a/&gt;</xsl:text></t></test>
         </xsl:template>
       </xsl:stylesheet>"#;
    assert_eq!(
        run(xsl, "<in/>")?,
        "<test><t><![CDATA[plain]]></t><t><a/></t></test>"
    );
    Ok(())
}

// --- the availability functions (XSLT 3.0 §23.1.1, §23.1.4, §23.2.2) ------------------------

/// `function-available-1006`: the lexical name is expanded against the namespace declarations
/// in scope *for the expression*, so a `fn` prefix bound to the obsolete 2003/05 namespace
/// names no function at all.
#[test]
fn function_available_expands_the_name_in_the_static_context_of_the_call() -> Result<(), XsltError>
{
    let xsl = r#"<xsl:stylesheet version="3.0" xmlns:xsl="http://www.w3.org/1999/XSL/Transform"
             xmlns:fn="http://www.w3.org/2003/05/xpath-functions">
         <xsl:template match="/">
           <out>
             <a><xsl:value-of select="function-available('fn:implicit-timezone')"/></a>
             <b><xsl:value-of select="function-available('implicit-timezone')"/></b>
             <c><xsl:value-of select="function-available('Q{http://www.w3.org/2005/xpath-functions}implicit-timezone')"/></c>
           </out>
         </xsl:template>
       </xsl:stylesheet>"#;
    assert!(
        run(xsl, "<in/>")?.ends_with("><a>false</a><b>true</b><c>true</c></out>"),
        "the obsolete namespace holds no function; the unprefixed name takes the standard one"
    );
    Ok(())
}

/// `function-0303`: `element-available` asks about an XSLT *element*, so a declaration answers
/// true as much as an instruction does — `xsl:key` is the name the case singles out. An EQName
/// stating an empty namespace, and an unprefixed name with no default namespace in scope, are
/// both in no namespace and name nothing.
#[test]
fn element_available_covers_the_declarations_too() -> Result<(), XsltError> {
    let xsl = r#"<xsl:stylesheet version="3.0" xmlns:xsl="http://www.w3.org/1999/XSL/Transform">
         <xsl:template match="/">
           <out>
             <a><xsl:value-of select="element-available('xsl:key')"/></a>
             <b><xsl:value-of select="element-available('xsl:value-of')"/></b>
             <c><xsl:value-of select="element-available('xsl:neverHeardOfIt')"/></c>
             <d><xsl:value-of select="element-available('xsl:package')"/></d>
             <e><xsl:value-of select="element-available('value-of')"/></e>
             <f><xsl:value-of select="element-available('Q{}value-of')"/></f>
           </out>
         </xsl:template>
       </xsl:stylesheet>"#;
    assert!(
        run(xsl, "<in/>")?.ends_with(
            "><a>true</a><b>true</b><c>false</c><d>false</d><e>false</e><f>false</f></out>"
        ),
        "a declaration this processor implements is available; one it declines is not"
    );
    Ok(())
}

// --- the `json` and `adaptive` output methods (XSLT 3.0 §26) --------------------------------

/// `result-document-0303`: with `method="json"` the destination builds no final result tree —
/// `build-tree` defaults to `no` — so the body's raw result, here a map, is serialized as JSON.
#[test]
fn a_result_document_serializes_json_from_the_raw_result() -> Result<(), XsltError> {
    let xsl = r#"<xsl:stylesheet version="3.0" xmlns:xsl="http://www.w3.org/1999/XSL/Transform">
         <xsl:template match="/">
           <xsl:result-document href="out.json" method="json">
             <xsl:map>
               <xsl:map-entry key="'a'" select="22"/>
             </xsl:map>
           </xsl:result-document>
         </xsl:template>
       </xsl:stylesheet>"#;
    let compiled = Stylesheet::compile_str(xsl)?;
    let source = Tree::parse_str("<in/>")
        .map_err(|error| XsltError::dynamic("XTDE0000", error.message().to_owned()))?;
    let transformation = compiled.transform(&source.root())?;
    let (_, text) = transformation
        .secondary_results()
        .first()
        .cloned()
        .expect("the instruction wrote one secondary result");
    assert_eq!(text, r#"{"a":22}"#);
    Ok(())
}

/// The same for the `adaptive` method, which writes every item of the sequence in the syntax
/// its own kind is written in — `result-document-0304`'s shape without its `build-tree`.
#[test]
fn a_result_document_serializes_adaptively_from_the_raw_result() -> Result<(), XsltError> {
    let xsl = r#"<xsl:stylesheet version="3.0" xmlns:xsl="http://www.w3.org/1999/XSL/Transform">
         <xsl:template match="/">
           <xsl:result-document href="out.txt" method="adaptive" item-separator="|">
             <xsl:sequence select="1, 'two', true()"/>
           </xsl:result-document>
         </xsl:template>
       </xsl:stylesheet>"#;
    let compiled = Stylesheet::compile_str(xsl)?;
    let source = Tree::parse_str("<in/>")
        .map_err(|error| XsltError::dynamic("XTDE0000", error.message().to_owned()))?;
    let transformation = compiled.transform(&source.root())?;
    let (_, text) = transformation
        .secondary_results()
        .first()
        .cloned()
        .expect("the instruction wrote one secondary result");
    assert_eq!(text, "1|\"two\"|true()");
    Ok(())
}

/// `result-document-0303`/`-1401`: the same holds for an `xsl:result-document` with no `href`,
/// which names the *principal* destination — the transformation's own text is the JSON.
#[test]
fn the_principal_destination_serializes_json_from_the_raw_result() -> Result<(), XsltError> {
    let xsl = r#"<xsl:stylesheet version="3.0" xmlns:xsl="http://www.w3.org/1999/XSL/Transform">
         <xsl:template match="/">
           <xsl:result-document method="json">
             <xsl:sequence select="array{ map{'a': 1}, map{'b': 2} }"/>
           </xsl:result-document>
         </xsl:template>
       </xsl:stylesheet>"#;
    assert_eq!(run(xsl, "<in/>")?, r#"[{"a":1},{"b":2}]"#);
    Ok(())
}

/// `result-document-1401`: an `xsl:map-entry` whose value comes from a sequence constructor
/// files the value "with no conversion" (XSLT 3.0 §11.10) — the integer stays an integer rather
/// than becoming the text node an `xsl:variable` without `as` would have built.
#[test]
fn a_map_entry_keeps_the_type_its_sequence_constructor_produced() -> Result<(), XsltError> {
    let xsl = r#"<xsl:stylesheet version="3.0" xmlns:xsl="http://www.w3.org/1999/XSL/Transform">
         <xsl:template match="/">
           <xsl:result-document method="json">
             <xsl:map>
               <xsl:map-entry key="'value'"><xsl:sequence select="9"/></xsl:map-entry>
             </xsl:map>
           </xsl:result-document>
         </xsl:template>
       </xsl:stylesheet>"#;
    assert_eq!(run(xsl, "<in/>")?, r#"{"value":9}"#);
    Ok(())
}
