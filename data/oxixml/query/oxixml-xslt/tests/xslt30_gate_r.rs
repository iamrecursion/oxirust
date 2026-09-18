//! Regression tests for the template-dispatch and serialization tail: patterns that match an
//! item which is not a node, the import *tree* `xsl:apply-imports` searches, `mode="#all"` rules
//! in a mode only an instruction names, `data-type="text"` on a numeric sort key, the
//! `json-node-output-method` serialization parameter, the default namespace declaration the
//! availability functions expand an unprefixed name with, and `fn:stream-available`.
//!
//! Every test reproduces one W3C `xslt30-test` case with a minimal inline stylesheet, so each fix
//! is provable red-then-green without the separate `oxixml-testsuite` harness. The case
//! identifiers in the comments are the ones that harness reports.

use oxixml_infoset::{Tree, XdmNode};
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

/// Compiles a principal module that imports the named modules, resolving them from memory, and
/// runs it from `xsl:initial-template`.
fn run_modules(modules: &[(&str, &str)], principal: &str) -> Result<String, XsltError> {
    let mut resolver = MemoryResolver::new();
    for (uri, source) in modules {
        resolver = resolver.with_document(*uri, source);
    }
    let options = StaticOptions::new()
        .with_base_uri("http://example.com/principal.xsl")
        .with_resolver(Arc::new(resolver));
    let compiled = Stylesheet::compile_str_with(principal, options)?;
    let tree = Tree::parse_str("<foo><bar/></foo>")
        .map_err(|error| XsltError::dynamic("XTDE0000", error.message().to_owned()))?;
    compiled.transform_to_string(&tree.root())
}

// --- a pattern that matches an item which is not a node (XSLT 3.0 §5.5.2) ------------------

/// `match-260`: `<xsl:apply-templates select="parse-json(…)"/>` applies the template rules to a
/// *map*, and a predicate pattern can match one. With two such rules and a mode declaring
/// `on-multiple-match="fail"`, XSLT 3.0 §6.4 makes the ambiguity `XTDE0540`.
///
/// Before the pattern matcher took an item rather than an atomic value, no rule was tried at all
/// and the built-in rule decided in silence.
#[test]
fn a_predicate_pattern_matches_a_map() {
    let xsl = r#"<xsl:stylesheet version="3.0" xmlns:xsl="http://www.w3.org/1999/XSL/Transform"
          xmlns:xs="http://www.w3.org/2001/XMLSchema">
       <xsl:param name="input" as="xs:string">{ "p" : "v" }</xsl:param>
       <xsl:mode on-multiple-match="fail"/>
       <xsl:template name="xsl:initial-template" match="/">
         <out><xsl:apply-templates select="parse-json($input)"/></out>
       </xsl:template>
       <xsl:template match=".[. instance of map(*)]"><A/></xsl:template>
       <xsl:template match=".[. instance of map(*)]"><B/></xsl:template>
     </xsl:stylesheet>"#;
    let error = run(xsl, "<doc/>").expect_err("two rules match the same map");
    assert_eq!(error.code(), Some("XTDE0540"), "{error}");
}

/// The same pattern without the ambiguity: one rule, and it fires. This pins the positive half —
/// that a map reaches the rule list at all — which the `XTDE0540` above would also satisfy if
/// the matcher simply reported every map as ambiguous.
#[test]
fn a_map_reaches_the_rule_that_matches_it() -> Result<(), XsltError> {
    let xsl = r#"<xsl:stylesheet version="3.0" xmlns:xsl="http://www.w3.org/1999/XSL/Transform"
          xmlns:xs="http://www.w3.org/2001/XMLSchema" exclude-result-prefixes="xs">
       <xsl:param name="input" as="xs:string">{ "p" : "v" }</xsl:param>
       <xsl:template match="/">
         <out><xsl:apply-templates select="parse-json($input)"/></out>
       </xsl:template>
       <xsl:template match=".[. instance of map(*)]"><A/></xsl:template>
     </xsl:stylesheet>"#;
    let out = run(xsl, "<doc/>")?;
    assert!(out.contains("<out><A/></out>"), "{out}");
    Ok(())
}

/// A rule that matches an *array* rather than a map, which travels the same path: the dispatch
/// is over items, not over the two kinds the engine happened to model first.
#[test]
fn a_predicate_pattern_matches_an_array() -> Result<(), XsltError> {
    let xsl = r#"<xsl:stylesheet version="3.0" xmlns:xsl="http://www.w3.org/1999/XSL/Transform">
       <xsl:template match="/">
         <out><xsl:apply-templates select="[1, 2]"/></out>
       </xsl:template>
       <xsl:template match=".[. instance of array(*)]"><A/></xsl:template>
     </xsl:stylesheet>"#;
    assert!(run(xsl, "<doc/>")?.contains("<out><A/></out>"));
    Ok(())
}

// --- the import tree, not import precedence (XSLT 3.0 §6.8) --------------------------------

/// `apply-imports-001`: the principal module imports `a` then `b`, so the import precedences are
/// `a` = 0, `b` = 1, principal = 2. An `xsl:apply-imports` written in `b` must find **nothing** —
/// `b` imports nothing, so its import subtree holds only itself — where a search that took every
/// lower precedence would find `a`'s rule and contribute a stray `Y`.
///
/// XSLT 3.0 §6.8 states the rule and warns against the cheap reading in the same breath: the
/// search considers "template rules contained in stylesheet levels that are descendants in the
/// import tree of the stylesheet level that contains the current template rule", and "this is not
/// the same as saying that the search considers all template rules whose import precedence is
/// lower".
#[test]
fn apply_imports_searches_the_import_subtree() -> Result<(), XsltError> {
    let module = |mark: &str, extra: &str| {
        format!(
            r#"<xsl:stylesheet version="3.0" xmlns:xsl="http://www.w3.org/1999/XSL/Transform">
             <xsl:template match=".[. ge 3]" priority="3">
               <xsl:text>{mark}</xsl:text><xsl:apply-imports/>
             </xsl:template>
             {extra}
           </xsl:stylesheet>"#
        )
    };
    let a = module("Y", "");
    let b = module(
        "Q",
        r#"<xsl:template match=".[. ge 1]" priority="1">
             <xsl:text>R</xsl:text><xsl:apply-imports/>
           </xsl:template>"#,
    );
    let principal = r#"<xsl:stylesheet version="3.0" xmlns:xsl="http://www.w3.org/1999/XSL/Transform">
         <xsl:import href="a.xsl"/>
         <xsl:import href="b.xsl"/>
         <xsl:template match="/">
           <out><xsl:apply-templates select="1 to 3"/></out>
         </xsl:template>
         <xsl:template match=".[. ge 3]" priority="3">
           <xsl:text>B</xsl:text><xsl:apply-imports/>
         </xsl:template>
       </xsl:stylesheet>"#;
    let out = run_modules(
        &[
            ("http://example.com/a.xsl", a.as_str()),
            ("http://example.com/b.xsl", b.as_str()),
        ],
        principal,
    )?;
    assert!(
        out.contains("<out>R1R2BQ3</out>"),
        "the apply-imports inside b must find nothing: {out}"
    );
    Ok(())
}

/// `import-1601`: the structural form of the same defect. The principal imports `b` (which
/// imports `c`) and then `d`; the rule that runs is `d`'s, and `d` imports nothing, so its
/// `xsl:apply-imports` falls straight through to the built-in rule rather than reaching `b` and
/// `c`, which are of lower precedence but are not its descendants.
#[test]
fn apply_imports_does_not_reach_a_sibling_import() -> Result<(), XsltError> {
    let leaf = |mark: &str, imports: &str| {
        format!(
            r#"<xsl:stylesheet version="3.0" xmlns:xsl="http://www.w3.org/1999/XSL/Transform">
             {imports}
             <xsl:template match="foo"><{mark}><xsl:apply-imports/></{mark}></xsl:template>
           </xsl:stylesheet>"#
        )
    };
    let c = leaf("C", "");
    let b = leaf("B", r#"<xsl:import href="c.xsl"/>"#);
    let d = leaf("D", "");
    let principal = r#"<xsl:stylesheet version="3.0" xmlns:xsl="http://www.w3.org/1999/XSL/Transform">
         <xsl:import href="b.xsl"/>
         <xsl:import href="d.xsl"/>
         <xsl:template match="/">
           <out><xsl:apply-templates select="/*"/></out>
         </xsl:template>
         <xsl:template match="foo"><A><xsl:apply-imports/></A></xsl:template>
       </xsl:stylesheet>"#;
    let out = run_modules(
        &[
            ("http://example.com/b.xsl", b.as_str()),
            ("http://example.com/c.xsl", c.as_str()),
            ("http://example.com/d.xsl", d.as_str()),
        ],
        principal,
    )?;
    assert!(
        out.contains("<A><D/></A>"),
        "d imports nothing, so B and C must not be reached: {out}"
    );
    Ok(())
}

// --- `mode="#all"` and a mode nothing declares (XSLT 3.0 §6.4) -----------------------------

/// `match-245`: `m1` is named by an `xsl:apply-templates` and by nothing else — no `xsl:mode`
/// declares it and no `xsl:template` writes it — so it had no rule list at all and the `#all`
/// rules never reached it. A mode an instruction selects is a mode the package uses.
#[test]
fn all_mode_rules_reach_a_mode_only_an_instruction_names() -> Result<(), XsltError> {
    let xsl = r##"<xsl:stylesheet version="3.0" xmlns:xsl="http://www.w3.org/1999/XSL/Transform">
       <xsl:strip-space elements="*"/>
       <xsl:template match="letters">
         <out><xsl:apply-templates mode="m1"/><xsl:apply-templates mode="m2"/></out>
       </xsl:template>
       <xsl:template match="letter[. = 'b']" mode="#all"><in><xsl:value-of select="."/></in></xsl:template>
       <xsl:template match="letter" mode="#all"/>
     </xsl:stylesheet>"##;
    assert!(
        run(
            xsl,
            "<letters><letter>a</letter><letter>b</letter></letters>"
        )?
        .contains("<out><in>b</in><in>b</in></out>")
    );
    Ok(())
}

/// The converse, which is why the rule lists are created only when there is something to put in
/// them: a mode an instruction names and nothing serves is still the `XTDE0045` it always was
/// when an invocation asks for it as the initial mode.
#[test]
fn a_mode_with_no_rules_at_all_is_still_not_an_entry_point() {
    let xsl = r#"<xsl:stylesheet version="3.0" xmlns:xsl="http://www.w3.org/1999/XSL/Transform">
       <xsl:template match="/"><out><xsl:apply-templates mode="m1"/></out></xsl:template>
     </xsl:stylesheet>"#;
    let compiled = Stylesheet::compile_str(xsl).expect("the stylesheet compiles");
    let tree = Tree::parse_str("<doc/>").expect("the source parses");
    let error = compiled
        .transform_with(&tree.root(), &Parameters::new().with_initial_mode("m1"))
        .expect_err("no rule is in mode m1");
    assert_eq!(error.code(), Some("XTDE0045"), "{error}");
}

// --- `xsl:sort/@data-type` (XSLT 3.0 §13.1.2) ----------------------------------------------

/// `sort-026`: `data-type="text"` forces the key to be compared as a string, so a numeric
/// `select` sorts `12, 15, 18, 20, 21, 24, 8`. Defaulting an absent `data-type` to `text` for
/// every version made the explicit attribute indistinguishable from its absence, and the run
/// side could not act on it.
#[test]
fn an_explicit_text_data_type_forces_string_comparison() -> Result<(), XsltError> {
    let xsl = r#"<xsl:stylesheet version="2.0" xmlns:xsl="http://www.w3.org/1999/XSL/Transform">
       <xsl:template match="/">
         <out><xsl:for-each select="doc/a">
           <xsl:sort select="@n" data-type="text"/>
           <xsl:value-of select="@n"/><xsl:text>,</xsl:text>
         </xsl:for-each></out>
       </xsl:template>
     </xsl:stylesheet>"#;
    let source = r#"<doc><a n="8"/><a n="12"/><a n="24"/></doc>"#;
    assert!(run(xsl, source)?.contains("<out>12,24,8,</out>"));
    Ok(())
}

/// `sort-051`, the guard on the fix: with no `data-type` at all, an XSLT 2.0 or 3.0 sort key
/// keeps its own type and is compared the way the `lt` operator would compare it, so the same
/// three numbers come out in numeric order.
#[test]
fn an_absent_data_type_keeps_the_keys_own_type() -> Result<(), XsltError> {
    let xsl = r#"<xsl:stylesheet version="2.0" xmlns:xsl="http://www.w3.org/1999/XSL/Transform">
       <xsl:template match="/">
         <out><xsl:for-each select="doc/a">
           <xsl:sort select="number(@n)"/>
           <xsl:value-of select="@n"/><xsl:text>,</xsl:text>
         </xsl:for-each></out>
       </xsl:template>
     </xsl:stylesheet>"#;
    let source = r#"<doc><a n="8"/><a n="12"/><a n="24"/></doc>"#;
    assert!(run(xsl, source)?.contains("<out>8,12,24,</out>"));
    Ok(())
}

/// XSLT 1.0's schema defaults `data-type` to `text`, and a 1.0 module is compiled as though it
/// had written it: every sort key is a string there, whatever its lexical form looks like.
#[test]
fn an_xslt_1_0_sort_defaults_to_a_string_comparison() -> Result<(), XsltError> {
    let xsl = r#"<xsl:stylesheet version="1.0" xmlns:xsl="http://www.w3.org/1999/XSL/Transform">
       <xsl:template match="/">
         <out><xsl:for-each select="doc/a">
           <xsl:sort select="number(@n)"/>
           <xsl:value-of select="@n"/><xsl:text>,</xsl:text>
         </xsl:for-each></out>
       </xsl:template>
     </xsl:stylesheet>"#;
    let source = r#"<doc><a n="8"/><a n="12"/><a n="24"/></doc>"#;
    assert!(run(xsl, source)?.contains("<out>12,24,8,</out>"));
    Ok(())
}

// --- `json-node-output-method` (Serialization 3.1 §10) -------------------------------------

/// `result-document-1402b`: the nodes inside a JSON value are serialized with the method
/// `json-node-output-method` names before their lexical form is written as a JSON string. With
/// `html` the serializer applies `include-content-type`, which is what puts the
/// `<meta http-equiv="Content-Type" …>` the case matches on into the string.
#[test]
fn json_nodes_are_serialized_with_the_method_the_parameter_names() -> Result<(), XsltError> {
    let xsl = r#"<xsl:stylesheet version="3.0" xmlns:xsl="http://www.w3.org/1999/XSL/Transform">
       <xsl:output method="json" indent="no" json-node-output-method="html"/>
       <xsl:template name="main">
         <xsl:map>
           <xsl:map-entry key="'doc1'">
             <html xmlns="http://www.w3.org/1999/xhtml">
               <head><title>Document 1</title></head>
             </html>
           </xsl:map-entry>
         </xsl:map>
       </xsl:template>
     </xsl:stylesheet>"#;
    let out = run_named(xsl, "main")?;
    assert!(
        out.contains(r#"<meta http-equiv=\"Content-Type\""#),
        "the html method writes the content type: {out}"
    );
    Ok(())
}

/// The default is `xml`, under which no `<meta>` is inserted — so the test above is testing the
/// parameter rather than the serializer's habits.
#[test]
fn json_nodes_default_to_the_xml_method() -> Result<(), XsltError> {
    let xsl = r#"<xsl:stylesheet version="3.0" xmlns:xsl="http://www.w3.org/1999/XSL/Transform">
       <xsl:output method="json" indent="no"/>
       <xsl:template name="main">
         <xsl:map>
           <xsl:map-entry key="'doc1'">
             <html xmlns="http://www.w3.org/1999/xhtml">
               <head><title>Document 1</title></head>
             </html>
           </xsl:map-entry>
         </xsl:map>
       </xsl:template>
     </xsl:stylesheet>"#;
    let out = run_named(xsl, "main")?;
    assert!(
        !out.contains("http-equiv"),
        "no content type is added: {out}"
    );
    Ok(())
}

// --- the availability functions (XSLT 3.0 §5.1, §18.1.1, §18.1.3) --------------------------

/// `function-0302b`, its `a10`: the argument of `element-available` is expanded with "the default
/// namespace of the defining element" — the `xmlns="…"` in scope where the call is written — and
/// not with `[xsl:]xpath-default-namespace`, which is a different property of the static context.
/// Here the defining element is an `xsl:variable` written with the XSLT namespace as its default,
/// so the unprefixed `value-of` names `xsl:value-of`.
#[test]
fn element_available_expands_an_unprefixed_name_with_the_default_namespace() -> Result<(), XsltError>
{
    let xsl = r#"<xsl:stylesheet version="3.0" xmlns:xsl="http://www.w3.org/1999/XSL/Transform">
       <xsl:param name="zls" select="''"/>
       <xsl:template match="/">
         <out>
           <variable name="v10" xmlns="http://www.w3.org/1999/XSL/Transform"
             select="element-available($zls || 'value-of')"/>
           <a10><xsl:value-of select="$v10"/></a10>
           <a8><xsl:value-of select="element-available('get-a-life')"/></a8>
         </out>
       </xsl:template>
     </xsl:stylesheet>"#;
    let out = run(xsl, "<doc/>")?;
    assert!(out.contains("<a10>true</a10>"), "{out}");
    // With no default namespace in scope an unprefixed name is in no namespace, which
    // §18.1.2 makes `false` outright.
    assert!(out.contains("<a8>false</a8>"), "{out}");
    Ok(())
}

/// `function-2001`: `fn:stream-available#1` is part of the XSLT 3.0 function library, so
/// `function-available` must say so — and the function itself answers `false` for every URI,
/// because this processor reports `no` for `xsl:supports-streaming` and therefore no document is
/// ever available for streamed processing (XSLT 3.0 §18.1.3).
#[test]
fn stream_available_is_present_and_answers_false() -> Result<(), XsltError> {
    let xsl = r#"<xsl:stylesheet version="3.0" xmlns:xsl="http://www.w3.org/1999/XSL/Transform">
       <xsl:template match="/">
         <out>
           <declared><xsl:value-of select="function-available('fn:stream-available', 1)"/></declared>
           <answer><xsl:value-of select="stream-available('urn:x')"/></answer>
         </out>
       </xsl:template>
     </xsl:stylesheet>"#;
    let out = run(xsl, "<doc/>")?;
    assert!(out.contains("<declared>true</declared>"), "{out}");
    assert!(out.contains("<answer>false</answer>"), "{out}");
    Ok(())
}

/// `function-available-1017`, `function-available-1018` and the other half of `function-2001`:
/// the higher-order functions are served by the XPath evaluator rather than by an entry of the
/// function library, and `fn:function-available` has to consult it too or it reports every one of
/// them absent.
#[test]
fn function_available_sees_the_higher_order_functions() -> Result<(), XsltError> {
    let xsl = r#"<xsl:stylesheet version="3.0" xmlns:xsl="http://www.w3.org/1999/XSL/Transform"
          xmlns:map="http://www.w3.org/2005/xpath-functions/map"
          xmlns:array="http://www.w3.org/2005/xpath-functions/array">
       <xsl:template match="/">
         <out>
           <a><xsl:value-of select="function-available('map:for-each', 2)"/></a>
           <b><xsl:value-of select="function-available('array:sort', 1)"/></b>
           <c><xsl:value-of select="function-available('fn:for-each', 2)"/></c>
           <d><xsl:value-of select="function-available('map:no-such-thing', 2)"/></d>
         </out>
       </xsl:template>
     </xsl:stylesheet>"#;
    let out = run(xsl, "<doc/>")?;
    assert!(out.contains("<a>true</a>"), "{out}");
    assert!(out.contains("<b>true</b>"), "{out}");
    assert!(out.contains("<c>true</c>"), "{out}");
    assert!(out.contains("<d>false</d>"), "{out}");
    Ok(())
}

// --- the namespace axis (XPath 3.1 §3.3.2.1, XSLT 3.0 §5.5.3) ------------------------------

/// `key-087`, `key-090`: "in XSLT 3.0, namespace nodes can match patterns so they can be used in
/// keys". Two things had to be true for that and neither was: an unprefixed `namespace-node()`
/// step takes the *namespace* axis, the way an unprefixed `attribute()` step takes the attribute
/// axis, and the nodes a key indexes have to include the namespace nodes in the first place.
///
/// The three elements come from one `xmlns:ext` declaration: a namespace node is in scope on the
/// element that declares it and on every descendant.
#[test]
fn a_key_may_index_namespace_nodes() -> Result<(), XsltError> {
    let xsl = r#"<xsl:stylesheet xmlns:xsl="http://www.w3.org/1999/XSL/Transform" version="3.0">
       <xsl:key name="ns" match="namespace-node()" use="name()"/>
       <xsl:template match="/">
         <out><xsl:for-each select="key('ns', 'ext')"><e><xsl:value-of select="name(..)"/></e></xsl:for-each></out>
       </xsl:template>
     </xsl:stylesheet>"#;
    let source = r#"<docs><doc xmlns:ext="urn:ext"><section><inner/></section></doc></docs>"#;
    let out = run(xsl, source)?;
    assert!(
        out.contains("<out><e>doc</e><e>section</e><e>inner</e></out>"),
        "{out}"
    );
    Ok(())
}

/// `element-0306`: `inherit-namespaces="no"` withholds the namespace nodes created *for* the
/// element — the one its own name needed here — and not the ones it inherited from its own
/// parent. XSLT 3.0 §11.4.2 draws that line explicitly, and the case counts namespace nodes on
/// four constructed elements to pin it.
#[test]
fn inherit_namespaces_no_withholds_only_the_elements_own_bindings() -> Result<(), XsltError> {
    let xsl = r#"<t:transform xmlns:fun="urn:fun" xmlns:easy="urn:easy"
          xmlns:t="http://www.w3.org/1999/XSL/Transform" version="2.0">
       <t:template match="doc">
         <out>
           <t:element xmlns="urn:dummy" name="profession" inherit-namespaces="no">
             <t:element name="IT_Architect"/>
             <t:element name="Business_Analyst" namespace=""/>
           </t:element>
           <count><t:value-of select="count(//*)"/></count>
         </out>
       </t:template>
     </t:transform>"#;
    let compiled = Stylesheet::compile_str(xsl)?;
    let tree = Tree::parse_str("<doc/>")
        .map_err(|error| XsltError::dynamic("XTDE0000", error.message().to_owned()))?;
    let result = compiled.transform(&tree.root())?;
    let bindings = |local: &str| -> Vec<String> {
        fn find(node: &oxixml_infoset::NodeRef, local: &str) -> Option<oxixml_infoset::NodeRef> {
            if node
                .node_name()
                .is_some_and(|name| name.local_name() == local)
            {
                return Some(node.clone());
            }
            node.children().iter().find_map(|child| find(child, local))
        }
        find(&result.root(), local)
            .map(|node| {
                node.namespaces()
                    .into_iter()
                    .map(|binding| binding.string_value())
                    .collect()
            })
            .unwrap_or_default()
    };
    // The stylesheet's own two prefixes reach every constructed element; the parent's
    // `urn:dummy` reaches only the child whose name needs it.
    assert_eq!(
        bindings("IT_Architect").len(),
        4,
        "{:?}",
        bindings("IT_Architect")
    );
    assert!(bindings("IT_Architect").contains(&"urn:dummy".to_owned()));
    assert_eq!(
        bindings("Business_Analyst").len(),
        3,
        "{:?}",
        bindings("Business_Analyst")
    );
    assert!(!bindings("Business_Analyst").contains(&"urn:dummy".to_owned()));
    Ok(())
}

// --- the XSLT extensions to the dynamic context (XSLT 3.0 §5.4.4) --------------------------

/// `regex-090`, `regex-091`: a named function reference retains the XPath context of the point it
/// was written at, but not the XSLT extensions to it — "if a dynamic function call is made that
/// depends on the XSLT part of the dynamic context (for example, `regex-group#1(2)`), then the
/// relevant components of the context are cleared". With no captured substrings there is no
/// group, and §18.5 makes that a zero-length string, so nothing is written at all.
#[test]
fn regex_group_through_a_function_item_sees_no_captures() -> Result<(), XsltError> {
    let xsl = r#"<xsl:stylesheet version="3.0" xmlns:xsl="http://www.w3.org/1999/XSL/Transform">
       <xsl:template match="/">
         <out>
           <xsl:analyze-string select="'aaabbb'" regex="(a*)(b*)">
             <xsl:matching-substring>
               <xsl:variable name="g" select="regex-group#1"/>
               <xsl:analyze-string select="'111222'" regex="(1*)(2*)">
                 <xsl:matching-substring><xsl:value-of select="$g(2)"/></xsl:matching-substring>
               </xsl:analyze-string>
             </xsl:matching-substring>
           </xsl:analyze-string>
         </out>
       </xsl:template>
     </xsl:stylesheet>"#;
    let out = run(xsl, "<doc/>")?;
    assert!(out.contains("<out/>"), "{out}");
    Ok(())
}

// --- a place-marker value beyond the machine integers (XSLT 3.0 §12.3) ---------------------

/// `number-0807`: `xsl:number/@value` is converted to an `xs:integer`, and XPath's integers are
/// unbounded. Casting the `xs:double` `1e100` gives "the integer that is exactly equal to the
/// mantissa times 2 to the power of the unbiased exponent", which is a hundred and one digits and
/// not a round power of ten; the place marker used to saturate at `i64::MAX` instead.
#[test]
fn a_place_marker_value_may_exceed_a_machine_integer() -> Result<(), XsltError> {
    let xsl = r#"<xsl:stylesheet version="2.0" xmlns:xsl="http://www.w3.org/1999/XSL/Transform">
       <xsl:template match="/"><out><xsl:number value="1e100" format="[1]"/></out></xsl:template>
     </xsl:stylesheet>"#;
    let out = run(xsl, "<doc/>")?;
    assert!(
        out.contains(
            "[10000000000000000159028911097599180468360808563945281389781327557747838772170381060813469985856815104]"
        ),
        "{out}"
    );
    Ok(())
}

/// The exactness travels through a lexical integer too: a value written with more digits than an
/// `f64` can carry keeps every one of them.
#[test]
fn a_lexical_integer_place_marker_keeps_every_digit() -> Result<(), XsltError> {
    let xsl = r#"<xsl:stylesheet version="2.0" xmlns:xsl="http://www.w3.org/1999/XSL/Transform"
          xmlns:xs="http://www.w3.org/2001/XMLSchema">
       <xsl:template match="/">
         <out><xsl:number value="xs:integer('123456789012345678901234567')" format="1"/></out>
       </xsl:template>
     </xsl:stylesheet>"#;
    let out = run(xsl, "<doc/>")?;
    assert!(out.contains(">123456789012345678901234567<"), "{out}");
    Ok(())
}

/// `number-0111`: the same value under three format tokens at once — plain decimal, a
/// zero-padded token and an Arabic-Indic digit — with a grouping separator every nine digits and
/// a `start-at` list. A value beyond the machine integers has to survive all three.
#[test]
fn a_large_place_marker_keeps_its_digit_family_and_grouping() -> Result<(), XsltError> {
    let xsl = r##"<xsl:stylesheet version="3.0" xmlns:xsl="http://www.w3.org/1999/XSL/Transform"
          exclude-result-prefixes="#all">
       <xsl:template match="/">
         <xsl:variable name="big" select="1234567890"/>
         <xsl:variable name="huge" select="$big * $big * $big"/>
         <out>
           <xsl:number value="(1 to 3)!($huge+.)" grouping-separator=":" grouping-size="9"
             start-at="3 2 1" format="[1(0)(&#x0660;)]"/>
         </out>
       </xsl:template>
     </xsl:stylesheet>"##;
    let out = run(xsl, "<doc/>")?;
    let arabic: String = "1:881676371:789154860:897069003"
        .chars()
        .map(|character| match character.to_digit(10) {
            Some(digit) => char::from_u32(0x0660 + digit).unwrap_or(character),
            None => character,
        })
        .collect();
    assert!(
        out.contains(&format!(
            "<out>[1:881676371:789154860:897069003\
             (1:881676371:789154860:897069003)({arabic})]</out>"
        )),
        "{out}"
    );
    Ok(())
}

/// `function-1901` and `function-2001`: the catalogue's two roll-calls, which walk the whole
/// function list of F&O and of XSLT 3.0 respectively and assert that `fn:function-available`
/// answers `true` for every entry (and `false` for the `op:` namespace, which is notation rather
/// than a library). Run against the suite's own generated inventories when they are available.
#[test]
fn the_function_inventories_report_no_missing_function() {
    let Ok(suites_dir) = std::env::var("OXIXML_XML_SUITES_DIR") else {
        return;
    };
    let base = std::path::Path::new(&suites_dir).join("xslt30-test/tests/decl/function");
    for (stylesheet, inventory) in [
        ("function-1901.xsl", "function-1901.xml"),
        ("function-2001.xsl", "function-2001.xml"),
    ] {
        let (Ok(xsl), Ok(source)) = (
            std::fs::read_to_string(base.join(stylesheet)),
            std::fs::read_to_string(base.join(inventory)),
        ) else {
            panic!("the suite is present but {inventory} is not");
        };
        let compiled =
            Stylesheet::compile_str(&xsl).unwrap_or_else(|error| panic!("{stylesheet}: {error}"));
        let tree = Tree::parse_str(&source).unwrap_or_else(|error| panic!("{inventory}: {error}"));
        let out = compiled
            .transform_to_string(&tree.root())
            .unwrap_or_else(|error| panic!("{stylesheet}: {error}"));
        // `xs:error` — F&O 3.1 §19.1's type with no valid instances, whose constructor raises
        // `FORG0001` for every non-empty argument — is the one entry the evaluator does not
        // serve, so it is the one entry the roll-call is allowed to miss.
        let residue: Vec<&str> = out
            .lines()
            .filter(|line| line.contains("<missing-function"))
            .filter(|line| !line.contains("\"xs:error\""))
            .collect();
        assert!(
            residue.is_empty(),
            "{stylesheet} reports a missing function: {residue:?}"
        );
    }
}
