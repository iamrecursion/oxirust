//! Semantics and error codes the W3C *QT3* suite pins, checked without the suite.
//!
//! Each test names the catalogue cases it was written from, so that a change of behaviour points
//! straight at the wording it would contradict. The queries are the catalogue's, reduced to what
//! the assertion needs; the suite itself runs them against its own environments in `qt3.rs`.

use oxixml_xquery::{DynamicQueryContext, XQuery, XQueryErrorCode, XQueryResult};

fn run(query: &str) -> XQueryResult<String> {
    XQuery::compile_default(query)?.execute_to_string(&DynamicQueryContext::new())
}

fn value(query: &str) -> String {
    run(query).expect("the query evaluates")
}

fn code_of(query: &str) -> Option<XQueryErrorCode> {
    run(query).err().and_then(|error| error.code())
}

/// The local part of the error's name, for the codes this crate's enum does not carry.
fn error_name(query: &str) -> String {
    run(query)
        .err()
        .map(|error| error.name().local_name().to_owned())
        .unwrap_or_default()
}

// --- namespaces in scope inside a constructor -------------------------------------------------

/// A namespace declaration attribute extends the statically known namespaces for the whole of
/// the constructor's content, and the *evaluator* has to see that too: resolving the prefix of a
/// lexical `xs:QName` happens at run time — QT3 `K2-NamespaceURIFromQNameFunc-1`, `-3`, `-4`,
/// `K2-SeqExprCast-479`, `K2-DefaultNamespaceProlog-6`.
#[test]
fn a_constructor_extends_the_namespaces_its_content_sees() {
    assert_eq!(
        value(
            r#"declare namespace ns = "http://example.com/A";
               string(<name xmlns:ns="http://example.com/B">{
                   namespace-uri-from-QName("ns:foo" cast as xs:QName)
               }</name>)"#
        ),
        "http://example.com/B"
    );
    assert_eq!(
        value(
            r#"declare default element namespace "http://example.com/";
               <e xmlns="">{namespace-uri-from-QName(xs:QName("l"))}</e>/string()"#
        ),
        ""
    );
    assert_eq!(
        value(r#"string(<e xmlns:p="http://example.com/">{xs:QName("p:local")}</e>)"#),
        "p:local"
    );
}

/// An element whose own name is in no namespace does not inherit a default namespace binding
/// when it is copied into a constructor's content — QT3 `K2-InScopePrefixesFunc-12`.
#[test]
fn an_unqualified_element_inherits_no_default_namespace() {
    assert_eq!(
        value(
            r#"declare default element namespace "http://www.example.com/";
               let $i := element e {
                   element {QName("", "a")} {},
                   element {QName("http://www.example.com/", "b")} {}
               }
               return (count(in-scope-prefixes($i)),
                       count(in-scope-prefixes(exactly-one($i/*[namespace-uri() eq ""]))),
                       count(in-scope-prefixes(exactly-one($i/b))))"#
        ),
        "2 1 2"
    );
}

/// A braced URI literal admits entity and character references in XQuery, and a reference never
/// ends the literal — QT3 `eqname-029`, `eqname-008`, `eqname-909`.
#[test]
fn a_braced_uri_literal_expands_references() {
    assert_eq!(
        value(
            "Q{http:&#x2F;&#x2F;www.w3.org&#x2F;2005&#x2F;xpath-functions&#x2F;math}pi() \
             eq math:pi()"
        ),
        "true"
    );
    assert_eq!(
        value(
            "string(<out>{ (<a xmlns:ex=\"http://www.example.com/ns\"><ex:b>93.7</ex:b></a>) \
             /Q{http://www&#x2e;example&#x2E;com/ns}b }</out>)"
        ),
        "93.7"
    );
    // `&#x7D;` is content, not the closing brace, so the URI begins with `}` and names nothing.
    assert_eq!(
        error_name("Q{&#x7D;http://www.w3.org/2005/xpath-functions/math}pi()"),
        "XPST0017"
    );
}

/// A string literal expands references too, which is what a node test written inside one needs —
/// QT3 `NodeTest013-2`.
#[test]
fn a_string_literal_expands_references() {
    assert_eq!(
        value("count(<a><?a-pi x?></a>/processing-instruction('a&#x2d;pi'))"),
        "1"
    );
}

// --- base URIs --------------------------------------------------------------------------------

/// A copied element's `xml:base` resolves against the base URI of its *new* parent, and a
/// descendant's against the copy just made — QT3 `K2-BaseURIFunc-33`.
#[test]
fn a_nested_xml_base_resolves_against_its_new_parent() {
    assert_eq!(
        value(
            r#"<e xml:base="http://example.com/ABC/">
                 <a xml:base="../"><b xml:base="DEF/file.test"/></a>
               </e>/a/b/base-uri()"#
        ),
        "http://example.com/DEF/file.test"
    );
}

// --- axes ---------------------------------------------------------------------------------

/// XQuery has no namespace axis: `namespace::*` is not in its grammar, and an abbreviated step
/// whose node test is `namespace-node()` would run on it — QT3 `K2-Axes-54`, `Axes112`,
/// `Axes115`, `Axes117`.
#[test]
fn xquery_has_no_namespace_axis() {
    assert_eq!(
        code_of("<a/>/namespace::*"),
        Some(XQueryErrorCode::Xpst0003)
    );
    assert_eq!(
        code_of("<a/>/namespace-node()"),
        Some(XQueryErrorCode::Xqst0134)
    );
    // An explicit axis makes it an ordinary kind test again.
    assert_eq!(value("count(<a/>/attribute::namespace-node())"), "0");
    assert_eq!(
        value(r#"count(namespace {"p"} {"urn:example"} / self::namespace-node())"#),
        "1"
    );
}

/// A predicate of a reverse-axis step numbers the nodes along the axis — QT3
/// `K2-ancestorAxis-8`, `-11`.
#[test]
fn a_reverse_axis_numbers_its_predicates_along_the_axis() {
    assert_eq!(
        value(r#"name(<a> <b c=""> <c/> </b> <d/> </a>/b/c/ancestor::*[1])"#),
        "b"
    );
    assert_eq!(
        value(r#"name(<a> <b c=""> <c/> </b> <d/> </a>/b/c/ancestor::*[fn:last()])"#),
        "a"
    );
}

// --- error codes -----------------------------------------------------------------------------

/// An operand of `/` that is not a sequence of nodes is `XPTY0019`, wherever the step stands —
/// QT3 `XPTY0019_1`, `XPTY0019_3`, `K2-Axes-51`.
#[test]
fn a_step_applied_to_a_non_node_is_xpty0019() {
    assert_eq!(code_of("<a/>/1/node()"), Some(XQueryErrorCode::Xpty0019));
    assert_eq!(code_of("string(<e/>)/3"), Some(XQueryErrorCode::Xpty0019));
    assert_eq!(
        code_of(
            r#"declare namespace foo = "http://www.example.org";
               declare function foo:something($pos as xs:integer) {
                   if ($pos eq 1) then 1 else <a/>
               };
               let $x := <a><b><c/></b><b><c/></b></a>
               return $x/b/(foo:something(position()))/a"#
        ),
        Some(XQueryErrorCode::Xpty0019)
    );
}

/// From XQuery 3.0 onwards a `cast as` whose target is not a known simple type is `XQST0052` —
/// QT3 `K-SeqExprCast-9a`, `K2-DefaultNamespaceProlog-12a`. An `instance of` keeps `XPST0051`.
#[test]
fn an_unknown_cast_target_is_xqst0052() {
    assert_eq!(
        code_of("3 cast as xs:doesNotExist"),
        Some(XQueryErrorCode::Xqst0052)
    );
    assert_eq!(
        code_of(
            r#"<a xmlns="http://www.w3.org/2001/XMLSchema">{1 cast as byte}
                 <b xmlns="http://www.w3.org/1999/XSL/Transform">{2 cast as byte}</b>
               </a>"#
        ),
        Some(XQueryErrorCode::Xqst0052)
    );
    assert_eq!(
        code_of("3 instance of xs:doesNotExist"),
        Some(XQueryErrorCode::Xpst0051)
    );
}

// --- typed data ------------------------------------------------------------------------------

/// A processor without the Schema Import Feature annotates every element it constructs
/// `xs:untyped`, whatever the construction mode says, because nothing it builds is validated —
/// QT3 `K2-DirectConElemContent-35b`, `K2-ExternalVariablesWith-22a`, `constprolog-10`.
#[test]
fn a_constructed_element_is_untyped() {
    assert_eq!(value("<e/> instance of element(*, xs:untyped)"), "true");
    assert_eq!(value("<e/> instance of element(*, xs:anyType)"), "true");
    assert_eq!(
        value("declare construction preserve; <e/> instance of element(*, xs:anyType)"),
        "true"
    );
    assert_eq!(
        value(r#"declare variable $v as element(*, xs:untyped?)+ := <e/>; exists($v/*)"#),
        "false"
    );
}

// --- function items --------------------------------------------------------------------------

/// A named reference to a declared function carries the signature that function declared, so a
/// `function(…) as …` test judges it by more than its arity — QT3 `hof-037`, `hof-038`.
#[test]
fn a_declared_functions_signature_reaches_its_function_item() {
    assert_eq!(
        value(
            r#"declare function local:f($x as xs:long, $y as xs:NCName) as element(e) {
                   <e x="{$x}" y="{$y}"/>
               };
               (local:f#2 instance of function(*),
                local:f#2 instance of function(xs:long, xs:NCName) as element(e),
                local:f#2 instance of function(xs:anyAtomicType?, xs:anyAtomicType?) as element(e),
                local:f#2 instance of function(item()*, item()*) as element(e))"#
        ),
        "true true false false"
    );
    assert_eq!(
        value(
            r#"declare function local:f($x as xs:long?, $y as xs:NCName?) as element(e) {
                   <e x="{$x}" y="{$y}"/>
               };
               (local:f#2 instance of function(xs:int?, xs:NCName?) as element(e),
                local:f#2 instance of function(xs:long?) as element(e),
                local:f#2 instance of function(xs:long?, xs:NCName?, item()*) as element(e),
                local:f#2 instance of function(xs:long, xs:anyAtomicType?) as element(e),
                local:f#2 instance of function(item()+, item()+) as element(e))"#
        ),
        "true false false false false"
    );
}

/// Function coercion wraps a function item that does not match the expected signature, and the
/// wrapper "has the same name and arity" as what it wraps — QT3 `fn-function-name-024`.
#[test]
fn function_coercion_keeps_the_functions_name() {
    assert_eq!(
        value(
            r#"declare function local:coerce($arg as function(item()*) as item()*)
                   as function(item()*) as item()* { $arg };
               let $coerced := local:coerce(fn:abs#1)
               return if ($coerced instance of function(item()*) as item()*)
                      then string(fn:function-name($coerced))
                      else "error""#
        ),
        "abs"
    );
}

// --- the content of a direct constructor -------------------------------------------------------
//
// A direct element constructor's content is XML, not XQuery: an apostrophe or a quotation mark is
// an ordinary character, `&…;` is a reference, and a nested constructor returns to the enclosing
// element's content rather than to the expression grammar. QT3 `ser/method-html`'s
// `Serialization-html-9` and `-10` write a `<script>` body around quoted markup, which is what
// forces the two lexical states apart.

#[test]
fn content_after_a_nested_constructor_is_not_an_expression() {
    for (query, expected) in [
        ("string(<a><b/>'</a>)", "'"),
        ("string(<a><b/>\"</a>)", "\""),
        ("string(<a><b/>&amp;</a>)", "&"),
        ("string(<a><b/>&apos;&quot;</a>)", "'\""),
        ("string(<a><b/>{{}}</a>)", "{}"),
        ("string(<a><b/>{'x'}'</a>)", "x'"),
        ("string(<a><!-- c -->'&amp;</a>)", "'&"),
        ("string(<a><?p data?>'&amp;</a>)", "'&"),
        ("string(<a><b><c/>'</b>'</a>)", "''"),
        ("string(<a><![CDATA[']]><b/>'</a>)", "''"),
    ] {
        assert_eq!(value(query), expected, "`{query}`");
    }
}

#[test]
fn a_script_constructor_keeps_its_quoted_markup() {
    let query = r#"declare boundary-space strip;
declare namespace output = "http://www.w3.org/2010/xslt-xquery-serialization";
declare option output:method  "html";
declare option output:indent  "no";
declare option output:version  "5.0";

<html xmlns="http://www.w3.org/1999/xhtml"><head><script language="Jack&amp;Jill">go &amp;&amp; write('<p class="Bill&amp;Ben">Jekyll&amp;Hyde</p>');</script></head><body><p>Hi!</p></body></html>
"#;
    assert_eq!(value(query), "go && write('Jekyll&Hyde');Hi!");
    // `Serialization-html-10` is the same query with the element names in mixed case.
    let mixed_case = r#"declare boundary-space strip;
declare namespace output = "http://www.w3.org/2010/xslt-xquery-serialization";
declare option output:method  "html";
declare option output:indent  "no";
declare option output:version  "5.0";

<HTML xmlns="http://www.w3.org/1999/xhtml"><head><SCRIPT language="Jack&amp;Jill">go &amp;&amp; write('<p class="Bill&amp;Ben">Jekyll&amp;Hyde</p>');</SCRIPT></head><body><p>Hi!</p></body></HTML>
"#;
    assert_eq!(value(mixed_case), "go && write('Jekyll&Hyde');Hi!");
}

#[test]
fn a_nested_constructor_keeps_the_enclosing_elements_children() {
    assert_eq!(value("count(<a>x<b/>'<c/>&amp;</a>/node())"), "5");
    assert_eq!(
        value("string-join(<a>x<b/>'<c/>&amp;</a>/text(), '|')"),
        "x|'|&"
    );
}
