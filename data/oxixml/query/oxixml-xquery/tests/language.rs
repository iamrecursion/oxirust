//! End-to-end tests of the XQuery 3.1 language surface: FLWOR, constructors, the prolog,
//! modules, `typeswitch`, `switch`, `try`/`catch` and the delegation boundary with XPath.

use oxixml_infoset::{AtomicValue, Tree};
use oxixml_xquery::{
    BoundarySpace, DynamicQueryContext, InMemoryModules, StaticQueryContext, XQuery, XQueryError,
    XQueryErrorCode, XQueryResult,
};
use std::sync::Arc;

/// Runs a query with no context item and returns `fn:string` of every item, space-joined.
fn run(query: &str) -> XQueryResult<String> {
    XQuery::compile_default(query)?.execute_to_string(&DynamicQueryContext::new())
}

/// Runs a query and serializes the result, so that constructed nodes can be compared as XML.
fn serialize(query: &str) -> XQueryResult<String> {
    run(&format!("fn:serialize(({query}))"))
}

/// Runs a query against a document.
fn run_on(query: &str, document: &str) -> XQueryResult<String> {
    let tree = Tree::parse_str(document)?;
    let context = DynamicQueryContext::new().with_context_item(tree.root());
    XQuery::compile_default(query)?.execute_to_string(&context)
}

fn error_code(query: &str) -> Option<String> {
    match run(query) {
        Ok(_) => None,
        Err(error) => Some(error.name().local_name().to_owned()),
    }
}

// --- the delegation boundary --------------------------------------------------------------------

#[test]
fn pure_xpath_is_delegated_unchanged() -> XQueryResult<()> {
    assert_eq!(run("1 + 2 * 3")?, "7");
    assert_eq!(run("'a' || 'b'")?, "ab");
    assert_eq!(
        run("string-join(for $x in 1 to 4 return string($x * $x), ',')")?,
        "1,4,9,16"
    );
    assert_eq!(run("count((1, 2, 3)[. > 1])")?, "2");
    assert_eq!(run("(1 to 5)[. mod 2 = 1]")?, "1 3 5");
    assert_eq!(run("map { 'a': 1, 'b': 2 }?b")?, "2");
    assert_eq!(run("array { 1, 2, 3 }(2)")?, "2");
    assert_eq!(run("'3' cast as xs:integer + 1")?, "4");
    assert_eq!(run("(1, 'x') instance of item()+")?, "true");
    assert_eq!(run("for-each(1 to 3, function($n) { $n + 1 })")?, "2 3 4");
    assert_eq!(run("1 => (function($n) { $n * 2 })()")?, "2");
    Ok(())
}

#[test]
fn an_xquery_construct_inside_an_xpath_form_is_lifted_out() -> XQueryResult<()> {
    // The `count(...)` call is XPath; its argument is not, so it is lifted into a variable.
    assert_eq!(run("count((<a/>, <b/>))")?, "2");
    assert_eq!(
        run("1 + (for $x in 1 to 3 group by $k := $x return $k)[1]")?,
        "2"
    );
    assert_eq!(run("string-length(string(<a>hello</a>))")?, "5");
    assert_eq!(
        run("(<a>1</a>, <b>2</b>)/string() => string-join('-')")?,
        "1-2"
    );
    assert_eq!(run("<a>x</a> instance of element()")?, "true");
    assert_eq!(run("local-name(<a:b xmlns:a='urn:a'/>)")?, "b");
    Ok(())
}

#[test]
fn an_xquery_construct_inside_a_binder_keeps_its_scope() -> XQueryResult<()> {
    assert_eq!(
        run("string-join(for $x in 1 to 3 return string(<n>{$x}</n>), '')")?,
        "123"
    );
    assert_eq!(run("let $e := <a>1</a> return string($e)")?, "1");
    assert_eq!(run("if (true()) then string(<a>y</a>) else 'n'")?, "y");
    assert_eq!(
        run("some $x in (1, 2) satisfies string(<n>{$x}</n>) = '2'")?,
        "true"
    );
    assert_eq!(
        run("string-join((<r><a>1</a><a>2</a></r>)/a[number(.) > 1]/string(), '')")?,
        "2"
    );
    assert_eq!(run("(function($n) { string(<n>{$n}</n>) })(7)")?, "7");
    Ok(())
}

// --- FLWOR ----------------------------------------------------------------------------------------

#[test]
fn a_single_clause_flwor_matches_xpath() -> XQueryResult<()> {
    assert_eq!(run("for $x in (1, 2, 3) return $x * 2")?, "2 4 6");
    assert_eq!(run("let $x := 5 return $x + 1")?, "6");
    assert_eq!(
        run("for $x in (1, 2), $y in (10, 20) return $x + $y")?,
        "11 21 12 22"
    );
    Ok(())
}

#[test]
fn where_filters_the_tuple_stream() -> XQueryResult<()> {
    assert_eq!(
        run("for $x in 1 to 6 where $x mod 2 = 0 return $x")?,
        "2 4 6"
    );
    assert_eq!(
        run("for $x in 1 to 3 let $y := $x * $x where $y > 2 return $y")?,
        "4 9"
    );
    Ok(())
}

#[test]
fn positional_and_allowing_empty_bindings() -> XQueryResult<()> {
    assert_eq!(run("for $x at $i in ('a', 'b', 'c') return $i")?, "1 2 3");
    assert_eq!(
        run("for $x at $i in ('a', 'b') return concat($i, $x)")?,
        "1a 2b"
    );
    assert_eq!(run("for $x allowing empty in () return 'once'")?, "once");
    assert_eq!(
        run("for $x allowing empty at $i in () return string($i)")?,
        "0"
    );
    Ok(())
}

#[test]
fn count_numbers_the_tuple_stream() -> XQueryResult<()> {
    assert_eq!(
        run("for $x in ('a', 'b', 'c') count $c return $c")?,
        "1 2 3"
    );
    assert_eq!(
        run("for $x in (3, 1, 2) order by $x count $c return concat($c, ':', $x)")?,
        "1:1 2:2 3:3"
    );
    Ok(())
}

#[test]
fn order_by_sorts_stably_and_honours_its_modifiers() -> XQueryResult<()> {
    assert_eq!(run("for $x in (3, 1, 2) order by $x return $x")?, "1 2 3");
    assert_eq!(
        run("for $x in (3, 1, 2) order by $x descending return $x")?,
        "3 2 1"
    );
    assert_eq!(
        run("for $x in ('b', 'a', 'c') stable order by $x return $x")?,
        "a b c"
    );
    // The empty sequence sorts least by default, greatest when asked.
    assert_eq!(
        run("for $x in (2, 1) order by (if ($x = 1) then () else $x) return $x")?,
        "1 2"
    );
    assert_eq!(
        run("for $x in (2, 1) order by (if ($x = 1) then () else $x) empty greatest return $x")?,
        "2 1"
    );
    assert_eq!(
        run("declare default order empty greatest; \
             for $x in (2, 1) order by (if ($x = 1) then () else $x) return $x")?,
        "2 1"
    );
    // Two keys.
    assert_eq!(
        run("for $x in (1, 2, 3, 4) order by $x mod 2, $x descending return $x")?,
        "4 2 3 1"
    );
    Ok(())
}

#[test]
fn group_by_collects_the_non_grouping_variables() -> XQueryResult<()> {
    assert_eq!(
        run("for $x in 1 to 6 group by $k := $x mod 3 order by $k return $k")?,
        "0 1 2"
    );
    assert_eq!(
        run("for $x in 1 to 6 \
             group by $k := $x mod 3 \
             order by $k \
             return concat($k, '=', string-join(for $v in $x return string($v), '+'))")?,
        "0=3+6 1=1+4 2=2+5"
    );
    // A grouping variable that is already in scope needs no `:=`.
    assert_eq!(
        run("for $x in (1, 1, 2) let $k := $x group by $k order by $k return count($x)")?,
        "2 1"
    );
    // The empty sequence is its own group.
    assert_eq!(
        run("for $x in (1, 2) group by $k := (if ($x = 1) then () else 'a') return count($x)")?,
        "1 1"
    );
    Ok(())
}

#[test]
fn tumbling_and_sliding_windows() -> XQueryResult<()> {
    assert_eq!(
        run("for tumbling window $w in (1 to 6) \
             start at $s when $s mod 2 = 1 \
             end at $e when $e mod 2 = 0 \
             return string-join(for $i in $w return string($i), '')")?,
        "12 34 56"
    );
    assert_eq!(
        run("for sliding window $w in (1 to 4) \
             start at $s when true() \
             only end at $e when $e - $s eq 1 \
             return string-join(for $i in $w return string($i), '')")?,
        "12 23 34"
    );
    // A tumbling window with no end condition runs until the next start.
    assert_eq!(
        run("for tumbling window $w in (1, 2, 3, 4, 5) \
             start $first when $first mod 3 = 1 \
             return string-join(for $i in $w return string($i), '')")?,
        "123 45"
    );
    // `previous` and `next` see the neighbours of the boundary item.
    assert_eq!(
        run("for tumbling window $w in (1 to 4) \
             start $c previous $p when empty($p) or $c - $p ge 2 \
             return count($w)")?,
        "4"
    );
    Ok(())
}

#[test]
fn a_typed_binding_matches_rather_than_converts() -> XQueryResult<()> {
    assert_eq!(run("for $x as xs:integer in (1, 2) return $x")?, "1 2");
    assert_eq!(run("let $x as xs:string := 'a' return $x")?, "a");
    assert_eq!(
        run("let $x as xs:double := 1.0e0 return $x instance of xs:double")?,
        "true"
    );
    // §3.12: a variable binding *matches* its declared type — it does not convert to it, so
    // neither an untyped atomic nor a value of a promotable type is accepted.
    for query in [
        "let $x as xs:integer := 'z' return $x",
        "let $x as xs:integer := xs:untypedAtomic('1') return $x",
        "let $x as xs:double := 1 return $x",
        "for $x as xs:string in xs:untypedAtomic('input') return $x",
    ] {
        assert_eq!(error_code(query).as_deref(), Some("XPTY0004"), "`{query}`");
    }
    // A function's parameters and result, by contrast, do get the function conversion rules.
    assert_eq!(
        run(
            "declare function local:f($x as xs:double) as xs:double { $x }; \
             local:f(1) instance of xs:double"
        )?,
        "true"
    );
    Ok(())
}

// --- constructors -----------------------------------------------------------------------------------

#[test]
fn direct_element_constructors() -> XQueryResult<()> {
    assert_eq!(serialize("<a/>")?, "<a/>");
    assert_eq!(serialize("<a>text</a>")?, "<a>text</a>");
    assert_eq!(serialize("<a b='1' c='2'/>")?, r#"<a b="1" c="2"/>"#);
    assert_eq!(serialize("<a>{1 + 1}</a>")?, "<a>2</a>");
    assert_eq!(serialize("<a>{1, 2, 3}</a>")?, "<a>1 2 3</a>");
    assert_eq!(serialize("<a><b/><c/></a>")?, "<a><b/><c/></a>");
    assert_eq!(serialize("<a x='{1 + 1}'/>")?, r#"<a x="2"/>"#);
    assert_eq!(serialize("<a>{{literal}}</a>")?, "<a>{literal}</a>");
    assert_eq!(serialize("<a>&lt;&amp;&gt;</a>")?, "<a>&lt;&amp;&gt;</a>");
    assert_eq!(serialize("<a><![CDATA[1 < 2]]></a>")?, "<a>1 &lt; 2</a>");
    assert_eq!(serialize("<!-- hi -->")?, "<!-- hi -->");
    assert_eq!(serialize("<?target data?>")?, "<?target data?>");
    Ok(())
}

#[test]
fn boundary_whitespace_is_stripped_unless_preserved() -> XQueryResult<()> {
    assert_eq!(serialize("<a>  <b/>  </a>")?, "<a><b/></a>");
    assert_eq!(serialize("<a> x </a>")?, "<a> x </a>");
    let preserving = XQuery::compile(
        "fn:serialize(<a>  <b/>  </a>)",
        StaticQueryContext::new().with_boundary_space(BoundarySpace::Preserve),
    )?;
    assert_eq!(
        preserving.execute_to_string(&DynamicQueryContext::new())?,
        "<a>  <b/>  </a>"
    );
    Ok(())
}

#[test]
fn computed_constructors() -> XQueryResult<()> {
    assert_eq!(serialize("element a { 1 }")?, "<a>1</a>");
    assert_eq!(serialize("element { 'a' } { 'x' }")?, "<a>x</a>");
    assert_eq!(
        serialize("element a { attribute b { 1 }, 'x' }")?,
        r#"<a b="1">x</a>"#
    );
    assert_eq!(serialize("document { <a/> }")?, "<a/>");
    assert_eq!(serialize("text { 'x' }")?, "x");
    assert_eq!(serialize("comment { 'c' }")?, "<!--c-->");
    assert_eq!(serialize("processing-instruction p { 'v' }")?, "<?p v?>");
    assert_eq!(run("count(text { () })")?, "0");
    assert_eq!(run("count(element a { () }/node())")?, "0");
    Ok(())
}

#[test]
fn a_processing_instruction_loses_only_its_separating_whitespace() -> XQueryResult<()> {
    // `DirPIConstructor ::= "<?" PITarget (S DirPIContents)? "?>"`: the whole `S` run separates
    // the target from the content, so `<?p  v?>` and `<?p v?>` hold the same content — and none
    // of the white space *inside* the content goes anywhere.
    assert_eq!(run("string(<?p   v?>)")?, "v");
    assert_eq!(run("string(<?p\t\r\n v?>)")?, "v");
    assert_eq!(run("string(<?p  a  b  ?>)")?, "a  b  ");
    assert_eq!(run("string(<?p?>)")?, "");
    assert_eq!(run("string(<?p   ?>)")?, "");

    // The computed constructor removes the leading white space of its content
    // (XQuery 3.1 §3.9.3.5), and white space there is XML's `S`.
    assert_eq!(run("string(processing-instruction p { '   v ' })")?, "v ");
    assert_eq!(
        run("string(processing-instruction p { '&#x9;&#xA; v' })")?,
        "v"
    );

    // A no-break space is a character, not white space: it is content on both routes. (A
    // processing instruction's content is raw text, so the direct form needs the character
    // itself — XML recognizes no reference inside one.)
    assert_eq!(run("string(<?p \u{a0}v?>)")?, "\u{a0}v");
    assert_eq!(
        run("string(processing-instruction p { '&#xA0;v' })")?,
        "\u{a0}v"
    );
    Ok(())
}

#[test]
fn namespaces_are_resolved_and_fixed_up() -> XQueryResult<()> {
    assert_eq!(
        serialize("<p:a xmlns:p='urn:p'><p:b/></p:a>")?,
        r#"<p:a xmlns:p="urn:p"><p:b/></p:a>"#
    );
    assert_eq!(run("namespace-uri(<p:a xmlns:p='urn:p'/>)")?, "urn:p");
    assert_eq!(
        run("declare namespace q = 'urn:q'; namespace-uri(<q:a/>)")?,
        "urn:q"
    );
    assert_eq!(
        run("declare default element namespace 'urn:d'; namespace-uri(<a/>)")?,
        "urn:d"
    );
    // An unprefixed attribute name is in no namespace.
    assert_eq!(
        run("declare default element namespace 'urn:d'; \
             string-join(for $a in <e x='1'/>/@* return string(namespace-uri($a)), '')")?,
        ""
    );
    assert_eq!(run("count(in-scope-prefixes(<a/>))")?, "1");
    assert_eq!(
        run("string-join(sort(in-scope-prefixes(<a xmlns:p='urn:p'/>)), ',')")?,
        "p,xml"
    );
    Ok(())
}

#[test]
fn constructed_nodes_have_a_new_identity_and_no_parent() -> XQueryResult<()> {
    assert_eq!(run("count(<a/>/..)")?, "0");
    assert_eq!(run("count(<a><b/></a>/b/..)")?, "1");
    assert_eq!(run("let $a := <a/> return $a is $a")?, "true");
    assert_eq!(run("<a/> is <a/>")?, "false");
    // Copying gives the copy a new identity.
    assert_eq!(
        run_on("let $b := <w>{//b}</w> return $b/b is //b", "<a><b/></a>")?,
        "false"
    );
    assert_eq!(run_on("string(<w>{//b}</w>)", "<a><b>t</b></a>")?, "t");
    Ok(())
}

#[test]
fn content_sequence_rules() -> XQueryResult<()> {
    // Attributes must precede other content.
    assert_eq!(
        error_code("element a { 'x', attribute b { 1 } }").as_deref(),
        Some("XQTY0024")
    );
    // Two attributes with the same name.
    assert_eq!(
        error_code("element a { attribute b { 1 }, attribute b { 2 } }").as_deref(),
        Some("XQDY0025")
    );
    assert_eq!(error_code("<a b='1' b='2'/>").as_deref(), Some("XQST0040"));
    // A document node in content is replaced by its children.
    assert_eq!(
        serialize("element a { document { <b/>, <c/> } }")?,
        "<a><b/><c/></a>"
    );
    // Adjacent text merges, empty text vanishes.
    assert_eq!(run("count(<a>{'x'}{'y'}</a>/node())")?, "1");
    assert_eq!(run("count(<a>{''}</a>/node())")?, "0");
    Ok(())
}

#[test]
fn constructor_errors_carry_their_codes() {
    assert_eq!(
        error_code("comment { 'a--b' }").as_deref(),
        Some("XQDY0072")
    );
    assert_eq!(
        error_code("processing-instruction { 'xml' } { 'x' }").as_deref(),
        Some("XQDY0064")
    );
    assert_eq!(
        error_code("processing-instruction p { 'a ?> b' }").as_deref(),
        Some("XQDY0026")
    );
    assert_eq!(
        error_code("element { 'not a name!' } { }").as_deref(),
        Some("XQDY0074")
    );
    assert_eq!(
        error_code("<a xmlns:p='{1}'/>").as_deref(),
        Some("XQST0022")
    );
}

// --- typeswitch, switch, try ---------------------------------------------------------------------

#[test]
fn typeswitch_picks_the_first_matching_case() -> XQueryResult<()> {
    let query = "declare function local:kind($v as item()*) as xs:string { \
                   typeswitch ($v) \
                     case $n as xs:integer return 'integer' \
                     case $s as xs:string | xs:anyURI return concat('text:', $s) \
                     case element() return 'element' \
                     default $d return 'other' \
                 }; \
                 string-join((local:kind(1), local:kind('a'), local:kind(<e/>), local:kind(1.5)), ',')";
    assert_eq!(run(query)?, "integer,text:a,element,other");
    Ok(())
}

#[test]
fn switch_compares_atomized_operands() -> XQueryResult<()> {
    assert_eq!(
        run("switch (2) case 1 return 'one' case 2 case 3 return 'few' default return 'many'")?,
        "few"
    );
    assert_eq!(
        run("switch (()) case 1 return 'one' default return 'none'")?,
        "none"
    );
    assert_eq!(
        run("switch ('a') case 'a' return 'hit' default return 'miss'")?,
        "hit"
    );
    Ok(())
}

#[test]
fn try_catch_binds_the_error_variables() -> XQueryResult<()> {
    assert_eq!(run("try { 1 idiv 0 } catch * { 'caught' }")?, "caught");
    assert_eq!(
        run("try { 1 idiv 0 } catch err:FOAR0001 { 'divide' } catch * { 'other' }")?,
        "divide"
    );
    assert_eq!(
        run("try { 1 idiv 0 } catch *:FOAR0001 { 'divide' } catch * { 'other' }")?,
        "divide"
    );
    assert_eq!(
        run("try { error(xs:QName('err:FORG0001'), 'boom') } catch * { $err:description }")?,
        "boom"
    );
    assert_eq!(
        run("try { 1 idiv 0 } catch * { local-name-from-QName($err:code) }")?,
        "FOAR0001"
    );
    assert_eq!(run("try { 1 } catch * { 2 }")?, "1");
    // An unmatched error propagates.
    assert!(run("try { 1 idiv 0 } catch err:XPTY0004 { 'no' }").is_err());
    Ok(())
}

#[test]
fn ordered_and_unordered_are_transparent() -> XQueryResult<()> {
    assert_eq!(run("ordered { 1 to 3 }")?, "1 2 3");
    assert_eq!(run("unordered { 1 to 3 }")?, "1 2 3");
    assert_eq!(run("declare ordering unordered; count(1 to 3)")?, "3");
    Ok(())
}

// --- the prolog ------------------------------------------------------------------------------------

#[test]
fn prolog_variables_resolve_in_dependency_order() -> XQueryResult<()> {
    assert_eq!(
        run("declare variable $b := $a + 1; declare variable $a := 1; $b")?,
        "2"
    );
    assert_eq!(
        error_code("declare variable $a := $b; declare variable $b := $a; $a").as_deref(),
        Some("XQDY0054")
    );
    assert_eq!(
        error_code("declare variable $a := 1; declare variable $a := 2; $a").as_deref(),
        Some("XQST0049")
    );
    Ok(())
}

#[test]
fn external_variables_come_from_the_dynamic_context() -> XQueryResult<()> {
    let compiled = XQuery::compile_default("declare variable $n external; $n * 2")?;
    let context = DynamicQueryContext::new().with_variable("n", AtomicValue::integer(21_i64));
    assert_eq!(compiled.execute_to_string(&context)?, "42");
    assert!(compiled.execute(&DynamicQueryContext::new()).is_err());

    let with_default = XQuery::compile_default("declare variable $n external := 7; $n")?;
    assert_eq!(
        with_default.execute_to_string(&DynamicQueryContext::new())?,
        "7"
    );
    Ok(())
}

#[test]
fn declared_functions_are_callable_and_may_recurse() -> XQueryResult<()> {
    assert_eq!(
        run("declare function local:double($n) { $n * 2 }; local:double(21)")?,
        "42"
    );
    assert_eq!(
        run(
            "declare function local:fact($n as xs:integer) as xs:integer { \
               if ($n le 1) then 1 else $n * local:fact($n - 1) }; \
             local:fact(6)"
        )?,
        "720"
    );
    // Mutual recursion, and a call from inside a delegated XPath expression.
    assert_eq!(
        run(
            "declare function local:even($n) { if ($n = 0) then true() else local:odd($n - 1) }; \
             declare function local:odd($n) { if ($n = 0) then false() else local:even($n - 1) }; \
             string-join((string(local:even(4)), string(local:odd(4))), ',')"
        )?,
        "true,false"
    );
    // A declared function can build nodes and read the prolog's variables.
    assert_eq!(
        run("declare variable $tag := 'w'; \
             declare function local:wrap($v) { element { $tag } { $v } }; \
             string(local:wrap('x'))")?,
        "x"
    );
    Ok(())
}

#[test]
fn declared_function_errors_carry_their_codes() {
    assert_eq!(
        error_code("declare function local:f() { 1 }; declare function local:f() { 2 }; 1")
            .as_deref(),
        Some("XQST0034")
    );
    assert_eq!(
        error_code("declare function fn:f() { 1 }; 1").as_deref(),
        Some("XQST0045")
    );
    assert_eq!(
        error_code("declare default function namespace ''; declare function f() { 1 }; 1")
            .as_deref(),
        Some("XQST0060")
    );
    assert_eq!(
        error_code("declare function local:f($a, $a) { 1 }; 1").as_deref(),
        Some("XQST0039")
    );
    assert_eq!(
        error_code("declare %public %private function local:f() { 1 }; 1").as_deref(),
        Some("XQST0106")
    );
}

#[test]
fn the_context_item_may_be_declared() -> XQueryResult<()> {
    assert_eq!(run("declare context item := 41; . + 1")?, "42");
    assert_eq!(run("declare context item as xs:integer := 1; . + 1")?, "2");
    let compiled = XQuery::compile_default("declare context item external; . * 2")?;
    let context = DynamicQueryContext::new().with_context_item(AtomicValue::integer(5_i64));
    assert_eq!(compiled.execute_to_string(&context)?, "10");
    // The declared type is *matched*, and the context item is exactly one item.
    assert_eq!(
        error_code("declare context item as xs:double := 1; .").as_deref(),
        Some("XPTY0004")
    );
    assert_eq!(
        error_code("declare context item := (1, 2); .").as_deref(),
        Some("XPTY0004")
    );
    // `ContextItemDecl` takes an `ItemType`, so an occurrence indicator is a syntax error.
    assert_eq!(
        error_code("declare context item as xs:integer+ := 1; .").as_deref(),
        Some("XPST0003")
    );
    Ok(())
}

#[test]
fn decimal_formats_reach_format_number() -> XQueryResult<()> {
    assert_eq!(
        run(
            "declare default decimal-format decimal-separator = ',' grouping-separator = '.'; \
             format-number(1234.5, '#.##0,00')"
        )?,
        "1.234,50"
    );
    assert_eq!(
        run("declare namespace e = 'urn:e'; \
             declare decimal-format e:euro decimal-separator = ',' grouping-separator = '.'; \
             format-number(1.5, '0,00', 'e:euro')")?,
        "1,50"
    );
    Ok(())
}

#[test]
fn the_version_declaration_and_base_uri_are_honoured() -> XQueryResult<()> {
    assert_eq!(run(r#"xquery version "3.1"; 1 + 1"#)?, "2");
    assert_eq!(
        run(r#"declare base-uri "http://example.com/"; string(static-base-uri())"#)?,
        "http://example.com/"
    );
    assert_eq!(
        error_code(r#"xquery version "9.9"; 1"#).as_deref(),
        Some("XQST0031")
    );
    Ok(())
}

#[test]
fn validate_is_a_documented_gap() {
    assert_eq!(error_code("validate { <a/> }").as_deref(), Some("XQST0075"));
    assert_eq!(
        error_code("validate strict { <a/> }").as_deref(),
        Some("XQST0075")
    );
}

// --- modules ----------------------------------------------------------------------------------------

#[test]
fn a_library_module_contributes_functions_and_variables() -> XQueryResult<()> {
    let library = InMemoryModules::new().with_module(
        "urn:example:m",
        r#"module namespace m = "urn:example:m";
           declare variable $m:base := 10;
           declare function m:scale($n as xs:integer) as xs:integer { $n * $m:base };"#,
    );
    let compiled = XQuery::compile(
        r#"import module namespace m = "urn:example:m"; m:scale(4) + $m:base"#,
        StaticQueryContext::new().with_module_resolver(Arc::new(library)),
    )?;
    assert_eq!(
        compiled.execute_to_string(&DynamicQueryContext::new())?,
        "50"
    );
    Ok(())
}

#[test]
fn modules_may_import_modules() -> XQueryResult<()> {
    let library = InMemoryModules::new()
        .with_module(
            "urn:a",
            r#"module namespace a = "urn:a";
               declare function a:one() { 1 };"#,
        )
        .with_module(
            "urn:b",
            r#"module namespace b = "urn:b";
               import module namespace a = "urn:a";
               declare function b:two() { a:one() + 1 };"#,
        );
    let compiled = XQuery::compile(
        r#"import module namespace b = "urn:b"; b:two()"#,
        StaticQueryContext::new().with_module_resolver(Arc::new(library)),
    )?;
    assert_eq!(
        compiled.execute_to_string(&DynamicQueryContext::new())?,
        "2"
    );
    Ok(())
}

#[test]
fn an_unresolvable_import_is_xqst0059() {
    let failure = XQuery::compile(
        r#"import module namespace m = "urn:missing"; 1"#,
        StaticQueryContext::new(),
    )
    .expect_err("no resolver is installed");
    assert!(failure.is(XQueryErrorCode::Xqst0059));

    let library = InMemoryModules::new();
    let failure = XQuery::compile(
        r#"import module namespace m = "urn:missing"; 1"#,
        StaticQueryContext::new().with_module_resolver(Arc::new(library)),
    )
    .expect_err("the module is not registered");
    assert!(failure.is(XQueryErrorCode::Xqst0059));
}

// --- paths and documents --------------------------------------------------------------------------

#[test]
fn queries_run_against_a_document() -> XQueryResult<()> {
    let document = "<library>\
                      <book year='1965'><title>Dune</title><price>9.99</price></book>\
                      <book year='1815'><title>Emma</title><price>5.50</price></book>\
                    </library>";
    assert_eq!(run_on("count(//book)", document)?, "2");
    assert_eq!(
        run_on(
            "for $b in //book order by xs:integer($b/@year) return string($b/title)",
            document
        )?,
        "Emma Dune"
    );
    assert_eq!(
        run_on("sum(for $p in //price return xs:decimal($p))", document)?,
        "15.49"
    );
    assert_eq!(
        run_on(
            "string-join(for $b in //book \
             where xs:integer($b/@year) > 1900 \
             return string($b/title), ',')",
            document
        )?,
        "Dune"
    );
    assert_eq!(
        run_on(
            "fn:serialize(<summary count='{count(//book)}'>{ \
               for $b in //book return <t>{ string($b/title) }</t> \
             }</summary>)",
            document
        )?,
        r#"<summary count="2"><t>Dune</t><t>Emma</t></summary>"#
    );
    Ok(())
}

#[test]
fn a_path_with_a_constructor_in_a_predicate_still_walks_the_tree() -> XQueryResult<()> {
    let document = "<r><a n='1'/><a n='2'/></r>";
    assert_eq!(
        run_on("count(//a[string(<n>{string(@n)}</n>) = '2'])", document)?,
        "1"
    );
    assert_eq!(run_on("string-join(//a/@n/string(), '')", document)?, "12");
    Ok(())
}

// --- string constructors and entity references ------------------------------------------------------

#[test]
fn string_constructors_interpolate() -> XQueryResult<()> {
    assert_eq!(run("``[plain]``")?, "plain");
    assert_eq!(run("``[a`{1 + 1}`b]``")?, "a2b");
    assert_eq!(run("``[`{('x', 'y')}`]``")?, "x y");
    Ok(())
}

#[test]
fn entity_references_resolve_in_string_literals() -> XQueryResult<()> {
    assert_eq!(run("'a&amp;b'")?, "a&b");
    assert_eq!(run("'&#x41;&#66;'")?, "AB");
    assert_eq!(run("concat('&lt;', 'x')")?, "<x");
    Ok(())
}

// --- errors -----------------------------------------------------------------------------------------

#[test]
fn syntax_errors_are_static_and_located() {
    let failure: XQueryError = XQuery::compile_default("1 +").expect_err("truncated");
    assert!(failure.is_static());
    assert_eq!(failure.code(), Some(XQueryErrorCode::Xpst0003));
    assert!(failure.offset().is_some());

    assert!(XQuery::compile_default("<a>{</a>").is_err());
    assert!(XQuery::compile_default("for $x in 1 return").is_err());
    assert!(XQuery::compile_default("nope:x").is_err());
}

// --- the content-sequence rules of §3.9.1.3 ------------------------------------------------------

#[test]
fn adjacent_enclosed_expressions_make_separate_text_nodes() -> XQueryResult<()> {
    // Each enclosed expression contributes its own text node, and adjacent text nodes are then
    // merged with *no* intervening blank — so `{1}{2}{3}` is `123`, while `{1, 2, 3}` is `1 2 3`.
    assert_eq!(serialize("<c>{1}{2}{3}</c>")?, "<c>123</c>");
    assert_eq!(serialize("<c>{1, 2, 3}</c>")?, "<c>1 2 3</c>");
    assert_eq!(serialize("<b>{1, '2', '3'}</b>")?, "<b>1 2 3</b>");
    assert_eq!(serialize("<a>{1}<b/>{2}</a>")?, "<a>1<b/>2</a>");
    // An array in the content is flattened by `array:flatten` before the rest of the rules run.
    assert_eq!(
        serialize(r#"<a>{ [ "one", "little", "fish" ] }</a>"#)?,
        "<a>one little fish</a>"
    );
    // A function item in the content is `XQTY0105`.
    assert_eq!(
        error_code("<a>{ function() { 1 } }</a>").as_deref(),
        Some("XQTY0105")
    );
    Ok(())
}

#[test]
fn a_cdata_section_is_not_boundary_whitespace() -> XQueryResult<()> {
    // §3.9.1.4: characters generated by a CDATA section are not whitespace characters, so the
    // whitespace on either side of one is never boundary whitespace.
    assert_eq!(run(r#"string(<e> <![CDATA[ ]]> </e>)"#)?, "   ");
    assert_eq!(run(r#"string(<e> <![CDATA[]]> </e>)"#)?, "  ");
    assert_eq!(run(r#"string(<a> <![CDATA[ ]]> {"abc"}</a>)"#)?, "   abc");
    // An *empty* CDATA section contributes nothing, so it cannot protect a later run either.
    assert_eq!(
        serialize(r#"<elem><![CDATA[]]>{attribute name {"content"}}<alem/> </elem>"#)?,
        r#"<elem name="content"><alem/></elem>"#
    );
    Ok(())
}

#[test]
fn a_text_constructor_keeps_the_empty_string() -> XQueryResult<()> {
    // §3.9.3.5: only an empty *sequence* yields no node.
    assert_eq!(run("count(text {''})")?, "1");
    assert_eq!(run("count(text {()})")?, "0");
    assert_eq!(
        serialize(r#"<out v="{(text{''}, text{'['}, text{'3'}, text{']'}, text{''})}"/>"#)?,
        r#"<out v=" [ 3 ] "/>"#
    );
    Ok(())
}

#[test]
fn a_constructed_element_takes_its_base_uri_from_xml_base() -> XQueryResult<()> {
    assert_eq!(
        run(r#"declare base-uri "http://example.com/"; base-uri(<e xml:base="fluster"/>)"#)?,
        "http://example.com/fluster"
    );
    // A nested constructor inherits its new parent's base URI.
    assert_eq!(
        run(r#"declare base-uri "http://example.com/";
               base-uri(exactly-one(<e xml:base="fluster"><a/></e>/a))"#)?,
        "http://example.com/fluster"
    );
    // `xml:base` does not affect the *static* base URI.
    assert_eq!(
        run(
            r#"declare base-uri "http://example.com/x"; <e xml:base="../">{static-base-uri()}</e>"#
        )?,
        "http://example.com/x"
    );
    Ok(())
}

#[test]
fn namespace_fixup_uses_the_prefix_as_written() -> XQueryResult<()> {
    // An unprefixed name in the default element namespace binds the *empty* prefix, so the
    // element serializes as `xmlns="…"` rather than with an invented prefix.
    assert_eq!(
        run("declare default element namespace 'urn:d'; fn:serialize(<inner/>)")?,
        r#"<inner xmlns="urn:d"/>"#
    );
    assert_eq!(
        run("declare namespace a = 'urn:a'; fn:serialize(<a:outer/>)")?,
        r#"<a:outer xmlns:a="urn:a"/>"#
    );
    // An element in the default element namespace declares the empty prefix, and nothing else.
    assert_eq!(
        run("declare default element namespace 'urn:d'; \
             string-join(in-scope-prefixes(<inner/>), ' ')")?,
        " xml"
    );
    Ok(())
}

// --- names and reserved namespaces ---------------------------------------------------------------

#[test]
fn computed_constructor_names_are_checked() -> XQueryResult<()> {
    for (query, code) in [
        (
            "element {QName('http://www.w3.org/2000/xmlns/', 'e')} {}",
            "XQDY0096",
        ),
        (
            "attribute {QName('http://example.com/not-xml', 'xml:e')} {}",
            "XQDY0044",
        ),
        ("<e>{namespace x {''}}</e>", "XQDY0101"),
        ("element e { namespace {''} {'urn:u'} }", "XQDY0102"),
        ("processing-instruction {xs:anyURI('pi')} {'t'}", "XPTY0004"),
        ("<e>{namespace {1} {'urn:u'}}</e>", "XPTY0004"),
        ("element { 'Q{{}x' } {}", "XQDY0074"),
    ] {
        assert_eq!(error_code(query).as_deref(), Some(code), "`{query}`");
    }
    // A braced URI literal expands references and is whitespace-normalized.
    assert_eq!(run("namespace-uri(element Q{z&#x20;z}x {})")?, "z z");
    assert_eq!(run("namespace-uri(element Q{ }x {})")?, "");
    Ok(())
}

// --- static errors the grammar and the prolog impose ---------------------------------------------

#[test]
fn the_grammar_rejects_what_the_specification_reserves() -> XQueryResult<()> {
    for query in [
        // A reserved function name may not be declared unprefixed.
        "declare function attribute() { 1 }; 1",
        "declare function switch() { 1 }; 1",
        // A setter may not follow a variable, function, option or context-item declaration.
        "declare variable $v := 1; declare default element namespace 'urn:x'; 1",
        "declare variable $v := 1; declare namespace p = 'urn:p'; 1",
        // A pragma's contents must be separated from its name by whitespace.
        "(#ex:pragma(content)#) {true()}",
        // A processing-instruction target must be a whole NCName.
        "<?invalid|char ?>",
    ] {
        assert_eq!(error_code(query).as_deref(), Some("XPST0003"), "`{query}`");
    }
    // `ordered {}` and `unordered {}` take an optional expression.
    assert_eq!(run("count(ordered{})")?, "0");
    assert_eq!(run("count(unordered{})")?, "0");
    Ok(())
}

#[test]
fn reserved_namespaces_and_prefixes_are_rejected() -> XQueryResult<()> {
    for query in [
        "declare default element namespace 'http://www.w3.org/XML/1998/namespace'; <a/>",
        "declare default function namespace 'http://www.w3.org/2000/xmlns/'; 1",
        "declare namespace xml = 'http://www.w3.org/XML/1998/namespace'; 'a'",
        "declare namespace p = 'http://www.w3.org/2000/xmlns/'; 'a'",
    ] {
        assert_eq!(error_code(query).as_deref(), Some("XQST0070"), "`{query}`");
    }
    Ok(())
}

#[test]
fn an_undeclared_variable_is_a_static_error() -> XQueryResult<()> {
    assert_eq!(
        error_code("try { $x } catch * { 'undefined' }").as_deref(),
        Some("XPST0008")
    );
    assert_eq!(error_code("$nowhere + 1").as_deref(), Some("XPST0008"));
    // A name bound anywhere in the module is accepted, and so is a declared external.
    assert_eq!(run("for $i in 1 to 3 return $i")?, "1 2 3");
    let compiled = XQuery::compile("$given", StaticQueryContext::new().with_variable("given"))?;
    let context = DynamicQueryContext::new().with_variable("given", AtomicValue::integer(7_i64));
    assert_eq!(compiled.execute_to_string(&context)?, "7");
    Ok(())
}

#[test]
fn clause_and_annotation_rules_are_enforced() -> XQueryResult<()> {
    assert_eq!(
        error_code("for $v at $v in (1, 2) return $v").as_deref(),
        Some("XQST0089")
    );
    assert_eq!(
        error_code("let $x := 1 return (for $i in ('a') group by $x return $x)").as_deref(),
        Some("XQST0094")
    );
    assert_eq!(
        error_code("declare option prefixnotdeclared:opt 'v'; 1").as_deref(),
        Some("XPST0081")
    );
    assert_eq!(
        error_code(
            "declare namespace xq = 'http://www.w3.org/2012/xquery'; \
             declare %private %xq:public function local:f() { () }; local:f()"
        )
        .as_deref(),
        Some("XQST0106")
    );
    assert_eq!(
        error_code(
            "declare namespace xq = 'http://www.w3.org/2012/xquery'; \
             declare %private %xq:public variable $v := (); $v"
        )
        .as_deref(),
        Some("XQST0116")
    );
    assert_eq!(
        error_code("declare default collation 'urn:no-such-collation'; 1").as_deref(),
        Some("XQST0038")
    );
    assert_eq!(
        error_code("let $i := (1, 2) order by 1 collation 'urn:nope' return $i").as_deref(),
        Some("XQST0076")
    );
    Ok(())
}

#[test]
fn an_annotation_assertion_in_a_sequence_type_is_accepted() -> XQueryResult<()> {
    // Annotation assertions are implementation-defined; this processor checks the name and then
    // ignores the assertion, so the type behaves as the bare `FunctionTest` does.
    assert_eq!(run("1 instance of %private function(*)")?, "false");
    assert_eq!(
        run("function() { 1 } instance of %local:note function(*)")?,
        "true"
    );
    // A name in a namespace the specification reserves is `XQST0045`.
    assert_eq!(
        error_code("1 instance of %fn:whatever function(*)").as_deref(),
        Some("XQST0045")
    );
    Ok(())
}

#[test]
fn a_private_function_is_visible_only_inside_its_module() -> XQueryResult<()> {
    let mut library = InMemoryModules::new();
    library.insert(
        "urn:pp".to_owned(),
        "module namespace m = 'urn:pp';
         declare %private function m:hidden() { 23 };
         declare %public function m:shown($a as xs:integer) { m:hidden() + $a };"
            .to_owned(),
    );
    let statik = StaticQueryContext::new().with_module_resolver(Arc::new(library));
    let compiled = XQuery::compile(
        "import module namespace p = 'urn:pp'; p:shown(42)",
        statik.clone(),
    )?;
    assert_eq!(
        compiled.execute_to_string(&DynamicQueryContext::new())?,
        "65"
    );
    let failure = XQuery::compile("import module namespace p = 'urn:pp'; p:hidden()", statik)
        .expect_err("a private function is out of scope");
    assert!(failure.is(XQueryErrorCode::Xpst0017));
    Ok(())
}

#[test]
fn switch_compares_with_deep_equal() -> XQueryResult<()> {
    // §3.14 compares with `fn:deep-equal`, under which `NaN` matches `NaN`.
    assert_eq!(
        run("switch (xs:double('NaN')) case xs:float('NaN') return 'yes' default return 'no'")?,
        "yes"
    );
    assert_eq!(
        run("switch (1) case 1.0e0 return 'yes' default return 'no'")?,
        "yes"
    );
    Ok(())
}

#[test]
fn the_error_variables_carry_the_err_prefix() -> XQueryResult<()> {
    assert_eq!(
        run("try { 1 div 0 } catch * { string($err:code) }")?,
        "err:FOAR0001"
    );
    assert_eq!(
        run("try { 1 div 0 } catch * { namespace-uri-from-QName($err:code) }")?,
        "http://www.w3.org/2005/xqt-errors"
    );
    Ok(())
}

#[test]
fn repeated_grouping_variables_occlude_one_another() -> XQueryResult<()> {
    // §3.12.7 processes grouping specs in order, each binding its variable for the ones that
    // follow, so `group by $y := $y, $y := $y mod 2` groups on `$y mod 2` alone.
    assert_eq!(
        run("count(for $y in 1 to 10 group by $y := $y, $y := $y mod 2 return $y)")?,
        "2"
    );
    assert_eq!(
        run("for $y in 1 to 10 group by $y := $y, $y := $y mod 2 order by $y return $y")?,
        "0 1"
    );
    Ok(())
}

#[test]
fn uri_literals_are_whitespace_normalized() -> XQueryResult<()> {
    let mut library = InMemoryModules::new();
    library.insert(
        "urn:spaced ns".to_owned(),
        "module namespace m = 'urn:spaced ns'; declare function m:hello() { 'ok' };".to_owned(),
    );
    let statik = StaticQueryContext::new().with_module_resolver(Arc::new(library));
    let compiled = XQuery::compile(
        "import module namespace p = 'urn:spaced    ns'; p:hello()",
        statik,
    )?;
    assert_eq!(
        compiled.execute_to_string(&DynamicQueryContext::new())?,
        "ok"
    );
    assert_eq!(
        run(r#"declare base-uri " http://example.org/x "; fn:string(fn:static-base-uri())"#)?,
        "http://example.org/x"
    );
    Ok(())
}

// --- the grammar's ambiguities ------------------------------------------------------------------

#[test]
fn a_map_constructor_separator_is_not_part_of_a_name() -> XQueryResult<()> {
    // `Wildcard` carries the grammar note `ws: explicit`, and `*:*` is none of its four forms,
    // so every `:` below separates a map entry's key from its value.
    assert_eq!(run("map:size(<a><b>x</b></a>/map{b:2})")?, "1");
    assert_eq!(run("map:size(<a><b>x</b></a>/map{self::a: b})")?, "1");
    assert_eq!(run("map:size(<a><b>x</b></a>/map{* :b})")?, "1");
    assert_eq!(run("map:size(<a><self>x</self></a>/map{self:2})")?, "1");
    assert_eq!(
        run(r#"declare namespace z = "urn:z"; map:size(<a><z:b>x</z:b></a>/map{z:b:z:b})"#)?,
        "1"
    );
    assert_eq!(
        run(r#"<e><f>foo</f></e> ! map{*:*} => deep-equal(map{"foo":<f>foo</f>})"#)?,
        "true"
    );
    // A genuine wildcard still is one.
    assert_eq!(
        run(r#"declare namespace z = "urn:z"; count(<a><z:b/><c/></a>/z:*)"#)?,
        "1"
    );
    assert_eq!(run("count(<a><b/><c/></a>/*:b)")?, "1");
    Ok(())
}

#[test]
fn a_lone_slash_ends_where_an_operator_begins() -> XQueryResult<()> {
    // `PathExpr ::= ("/" RelativePathExpr?) | …`: a `/` followed by something that cannot begin
    // a relative path is the whole expression, the root of the context node's tree.
    assert_eq!(run_on("count(.[/-5])", "<a>5</a>")?, "0");
    // A numeric predicate is positional, so `6 - 5` keeps the first item.
    assert_eq!(run_on("count(.[/-5])", "<a>6</a>")?, "1");
    assert_eq!(run_on("let $a := . return count(.[/=$a])", "<a/>")?, "1");
    assert_eq!(run_on("count(/ ! *)", "<a><b/></a>")?, "1");
    assert_eq!(run_on("count((/ | //*))", "<a><b/></a>")?, "3");
    // …but an array constructor *can* begin one.
    assert_eq!(run_on("count(/[1, 2])", "<a/>")?, "1");
    assert_eq!(run_on("name(/*)", "<a/>")?, "a");
    Ok(())
}

#[test]
fn a_leading_declare_or_import_may_be_a_name_test() -> XQueryResult<()> {
    // XQuery reserves no words, so a prolog keyword that is not followed by a declaration is
    // the query body — which here needs a context item it does not have.
    assert_eq!(error_code("declare").as_deref(), Some("XPDY0002"));
    assert_eq!(error_code("declare ne gt").as_deref(), Some("XPDY0002"));
    assert_eq!(error_code("import ne import").as_deref(), Some("XPDY0002"));
    assert_eq!(error_code("import gt import").as_deref(), Some("XPDY0002"));
    assert_eq!(run_on("count(//declare)", "<a><declare/></a>")?, "1");
    // A real declaration is still a declaration.
    assert_eq!(run("declare variable $v := 3; $v")?, "3");
    Ok(())
}

// --- the prolog ---------------------------------------------------------------------------------

#[test]
fn a_variable_is_not_in_scope_in_its_own_initializer() {
    // XQuery 3.1 §4.16: the scope of a prolog variable is the whole module *except* the
    // initializing expression of its own declaration.
    assert_eq!(
        error_code("declare variable $var1 := $var1; true()").as_deref(),
        Some("XPST0008")
    );
    // A forward reference to *another* declaration is fine, and a genuine circularity through a
    // function is the dynamic `XQDY0054` that XQuery 3.0 relaxed it to.
    assert_eq!(
        run("declare variable $a := $b; declare variable $b := 2; $a").as_deref(),
        Ok("2")
    );
    assert_eq!(
        error_code("declare variable $v := local:f(); declare function local:f() { $v }; $v")
            .as_deref(),
        Some("XQDY0054")
    );
}

#[test]
fn a_decimal_format_declaration_is_validated() {
    // A name that is not a `DFPropertyName` does not parse; one that parses with a bad value is
    // `XQST0097`; `zero-digit` must be a Unicode digit whose numeric value is zero.
    assert_eq!(
        error_code(r#"declare default decimal-format decimal-sprtr = "."; true()"#).as_deref(),
        Some("XPST0003")
    );
    assert_eq!(
        error_code(r#"declare decimal-format local:df currency-symbol="C"; true()"#).as_deref(),
        Some("XPST0003")
    );
    assert_eq!(
        error_code(r#"declare default decimal-format zero-digit = "A"; true()"#).as_deref(),
        Some("XQST0097")
    );
    assert_eq!(
        error_code(r#"declare default decimal-format zero-digit = "1"; true()"#).as_deref(),
        Some("XQST0097")
    );
    // An Arabic-Indic zero is a digit zero, and is accepted.
    assert_eq!(
        run("declare default decimal-format zero-digit = '\u{660}'; true()").as_deref(),
        Ok("true")
    );
}

#[test]
fn reserved_namespaces_and_annotations_are_rejected() {
    // Every predeclared namespace is reserved for declared functions, `xml` included.
    assert_eq!(
        error_code("declare function xml:foo() { 1 }; 1").as_deref(),
        Some("XQST0045")
    );
    assert_eq!(
        error_code("declare function fn:foo() { 1 }; 1").as_deref(),
        Some("XQST0045")
    );
    // `%public` and `%private` describe a declaration's visibility; an inline function has none.
    assert_eq!(
        error_code("let $f := %public function($a) { $a } return $f(1)").as_deref(),
        Some("XQST0125")
    );
    assert_eq!(
        error_code("let $f := %private function($a) { $a } return $f(1)").as_deref(),
        Some("XQST0125")
    );
    // A sequence type naming an unknown atomic type keeps its own code.
    assert_eq!(
        error_code("declare function local:e() as none { 1 }; local:e()").as_deref(),
        Some("XPST0051")
    );
}

#[test]
fn a_library_module_may_not_give_the_context_item_a_value() {
    let mut library = InMemoryModules::new();
    library.insert(
        "urn:ctx".to_owned(),
        "module namespace m = 'urn:ctx'; declare context item external := 17;".to_owned(),
    );
    let statik = StaticQueryContext::new().with_module_resolver(Arc::new(library));
    let error = XQuery::compile("import module namespace m = 'urn:ctx'; 1", statik)
        .expect_err("a library module's context item may not be initialized");
    assert_eq!(error.name().local_name(), "XQST0113");
}

#[test]
fn a_declared_context_item_gives_the_focus_a_position_and_a_size() -> XQueryResult<()> {
    // The context item declaration may stand *after* the variables it is built from, so a
    // variable initializer that needs the focus is retried once the focus is known.
    assert_eq!(
        run("declare variable $y := (<a>1</a>, <a>2</a>, <a>3</a>);
             declare context item := $y[3];
             declare variable $x external := fn:position();
             $x")?,
        "1"
    );
    assert_eq!(
        run("declare variable $y := <root><a>1</a></root>;
             declare context item := $y;
             declare variable $x external := fn:last();
             $x")?,
        "1"
    );
    Ok(())
}

#[test]
fn the_query_text_has_its_line_endings_normalized() -> XQueryResult<()> {
    // XQuery 3.1 §A.2.2: `#xD #xA` and a lone `#xD` both become `#xA` before parsing. `#x85`
    // and `#x2028` are XML 1.1's line endings and are left alone.
    assert_eq!(
        run("deep-equal(string-to-codepoints('\r\n'), (10))")?,
        "true"
    );
    assert_eq!(
        run("deep-equal(string-to-codepoints('\r \r\n \n\r'), (10, 32, 10, 32, 10, 10))")?,
        "true"
    );
    assert_eq!(
        run("deep-equal(string-to-codepoints('\r\u{85}'), (10, 133))")?,
        "true"
    );
    Ok(())
}

// --- constructors and the function conversion rules ----------------------------------------------

#[test]
fn a_function_argument_is_coerced_to_the_expected_function_type() -> XQueryResult<()> {
    // XQuery 3.1 §3.1.5.2: a function item supplied where a function *type* is expected is
    // wrapped in one that has the expected signature, so an untyped `function($x) { … }` is a
    // perfectly good `function(xs:string) as xs:string`.
    assert_eq!(
        run(
            r#"declare function local:apply($f as function(xs:string) as xs:string,
                                            $s as xs:string) as xs:string { $f($s) };
               local:apply(function($x) { upper-case($x) }, "john")"#
        )?,
        "JOHN"
    );
    assert_eq!(
        run(
            r#"declare function local:split() as (function(xs:string) as xs:string*)
                 { function($x as xs:string) { tokenize($x, '\s') } };
               string-join(local:split()("a nice cup"), '|')"#
        )?,
        "a|nice|cup"
    );
    assert_eq!(
        run(
            r#"declare function local:all($fs as (function(xs:string) as xs:string)*,
                                          $s as xs:string) as xs:string*
                 { for $f in $fs return $f($s) };
               string-join(local:all((upper-case#1, lower-case#1), 'Ab'), '~')"#
        )?,
        "AB~ab"
    );
    // A function of the wrong arity is still a type error.
    assert_eq!(
        error_code(
            r#"declare function local:apply($f as function(xs:string) as xs:string) { $f('a') };
               local:apply(concat#2)"#
        )
        .as_deref(),
        Some("XPTY0004")
    );
    Ok(())
}

#[test]
fn a_string_constructor_atomizes_its_interpolations() -> XQueryResult<()> {
    // XQuery 3.1 §3.15: the interpolated value is atomized, then string-joined with a space.
    assert_eq!(run("``[a `{[10]}` b]``")?, "a 10 b");
    assert_eq!(run("``[a `{array{1, 2}}` b]``")?, "a 1 2 b");
    assert_eq!(run("``[a `{(1, 2)}` b]``")?, "a 1 2 b");
    // A map has no typed value at all.
    assert_eq!(
        error_code("``[a `{map{'k': 1}}` b]``").as_deref(),
        Some("FOTY0013")
    );
    Ok(())
}

#[test]
fn a_constructor_name_may_be_written_with_character_references() -> XQueryResult<()> {
    // A `BracedURILiteral` may not *contain* a brace, but it may spell one with a character
    // reference, and the `}` that closes the URI is then the last one.
    assert_eq!(run("namespace-uri(attribute Q{&#x7b;&#x7d;}x {})")?, "{}");
    assert_eq!(run("local-name(attribute Q{&#x7b;&#x7d;}x {})")?, "x");
    // A name *computed* at run time has had no such scrutiny: a brace there is `XQDY0074`.
    assert_eq!(
        error_code(r#"attribute { "Q{{}x" } {}"#).as_deref(),
        Some("XQDY0074")
    );
    Ok(())
}

#[test]
fn a_computed_processing_instruction_target_is_an_ncname() {
    // `CompPIConstructor ::= "processing-instruction" (NCName | ("{" Expr "}")) EnclosedExpr`.
    assert_eq!(
        error_code(r#"<e xmlns:foo="urn:f">{processing-instruction foo:pi {'t'}}</e>"#).as_deref(),
        Some("XPST0003")
    );
    assert_eq!(
        run("fn:serialize(processing-instruction pi {'t'})").as_deref(),
        Ok("<?pi t?>")
    );
}

#[test]
fn a_nested_direct_constructor_is_not_copied() -> XQueryResult<()> {
    // XQuery 3.1 §3.9.1.3 copies "each node returned by an enclosed expression", and only a copy
    // is subject to the copy-namespaces mode. A nested direct constructor is not an enclosed
    // expression: "Each nested direct constructor is evaluated according to the rules in 3.9.1 …
    // resulting in a new element, comment, or processing instruction node. Then: The parent
    // property of the resulting node is then set to the newly constructed element node."
    //
    // So an `xmlns=""` undeclaration survives being placed in its parent's content, even under
    // the default `preserve, inherit`.
    assert_eq!(
        run(r#"count(in-scope-prefixes(<a xmlns="http://example.com/"> <e xmlns=""/> </a>/e))"#)?,
        "1"
    );
    // And a sibling binding the enclosing constructor needed only for its *own* names is not
    // inherited either.
    assert_eq!(
        run(r#"declare namespace a = "http://example.com/1";
               declare namespace b = "http://example.com/2";
               <e a:n1="content" b:n1="content"> <a:n1/> </e>/
                 (for $i in in-scope-prefixes(.) order by $i return $i, '|',
                  for $i in a:n1/in-scope-prefixes(.) order by $i return $i)"#)?,
        "a b xml | a xml"
    );
    // A node that *is* returned by an enclosed expression is still copied, and still inherits.
    assert_eq!(
        run(r#"declare namespace foo = "http://example.org";
               declare copy-namespaces preserve,inherit;
               let $existing := <existingElement xmlns="urn:existing"/>
               let $new := <foo:newElement xmlns="urn:mine">{$existing}</foo:newElement>
               for $p in in-scope-prefixes($new/*) order by $p return $p"#)?,
        " foo xml"
    );
    Ok(())
}

#[test]
fn an_unprefixed_element_name_binds_the_empty_prefix() -> XQueryResult<()> {
    // §3.9.4's last rule: "If there is an in-scope default namespace, then a binding is created
    // between the empty prefix and that URI" — even when another prefix already denotes it.
    assert_eq!(
        run(r#"declare default element namespace "http://example.com/";
               <e xmlns:p="http://example.com/">{ count(in-scope-prefixes(<e/>)),
                                                  count(in-scope-prefixes(element e {()})) }</e>
               /string()"#)?,
        "3 3"
    );
    assert_eq!(
        run(r#"declare default element namespace "urn:x";
               let $e := <e xmlns:p="urn:x"/>
               return (namespace-uri-for-prefix("", $e),
                       string-join(sort(in-scope-prefixes($e)), " "))"#)?,
        "urn:x  p xml"
    );
    Ok(())
}

#[test]
fn a_constructed_name_keeps_the_prefix_it_was_written_with() -> XQueryResult<()> {
    // §3.9.4: "the prefix used in the node name is changed" only when it would conflict, so a
    // name that was spelt with a prefix reports that prefix from `fn:node-name`.
    assert_eq!(
        run(r#"declare namespace p = "urn:p"; prefix-from-QName(node-name(<p:e/>))"#)?,
        "p"
    );
    assert_eq!(
        run(r#"string(node-name(<a xml:id="xyz"/>/@xml:id))"#)?,
        "xml:id"
    );
    // §3.9.3.2: "The rules that expand attribute names create an implementation-dependent prefix
    // if an attribute name has a namespace URI but no prefix is provided."
    assert_eq!(
        run(
            r#"prefix-from-QName(node-name(attribute {QName("urn:x", "attr")} {})) castable as xs:NCName"#
        )?,
        "true"
    );
    assert_eq!(
        run(r#"prefix-from-QName(node-name(
                 attribute {QName("http://www.w3.org/XML/1998/namespace", "attr")} {}))"#)?,
        "xml"
    );
    // A conflict does rename: the namespace node binds `p` to something else, so the attribute
    // cannot keep it.
    assert_eq!(
        run(
            r#"let $r := <e>{ namespace p {"urn:one"}, attribute {QName("urn:two", "p:att")} {1} }</e>
               return (exists($r/@*:att[prefix-from-QName(node-name(.)) != "p"]),
                       exists(in-scope-prefixes($r)[. = "p"]))"#
        )?,
        "true true"
    );
    Ok(())
}

#[test]
fn a_library_module_has_its_own_static_base_uri() -> XQueryResult<()> {
    // §4.5: `declare base-uri` sets the static base URI of the module it appears in, so a
    // function and a variable declared in a library see the library's, not the importer's.
    let library = InMemoryModules::new().with_module(
        "urn:example:b",
        r#"module namespace b = "urn:example:b";
           declare base-uri "http://example.org/correct/";
           declare variable $b:node := <a><c/></a>;
           declare function b:base() as xs:anyURI? { fn:static-base-uri() };"#,
    );
    let context = StaticQueryContext::new().with_module_resolver(Arc::new(library));
    let compiled = XQuery::compile(
        r#"declare base-uri "http://example.org/wrong/";
           import module namespace b = "urn:example:b";
           (b:base(), fn:static-base-uri(), fn:base-uri($b:node/node()))"#,
        context,
    )?;
    assert_eq!(
        compiled.execute_to_string(&DynamicQueryContext::new())?,
        "http://example.org/correct/ http://example.org/wrong/ http://example.org/correct/"
    );
    Ok(())
}

#[test]
fn a_cast_target_must_be_a_simple_type() {
    // §3.14.2 gives XQuery two errors XPath does not name: `XPST0080` for `xs:NOTATION`,
    // `xs:anySimpleType` and `xs:anyAtomicType`, and `XQST0052` for anything that is not a
    // simple type in the in-scope schema types.
    assert_eq!(
        error_code("'string' cast as xs:anySimpleType").as_deref(),
        Some("XPST0080")
    );
    assert_eq!(
        error_code("'string' castable as xs:anyAtomicType").as_deref(),
        Some("XPST0080")
    );
    assert_eq!(
        error_code("'string' cast as xs:untyped").as_deref(),
        Some("XQST0052")
    );
    assert_eq!(
        error_code("'string' castable as xs:anyType").as_deref(),
        Some("XQST0052")
    );
    assert_eq!(
        error_code("3 cast as xs:doesNotExist").as_deref(),
        Some("XQST0052")
    );
    // An unbound prefix is still XPST0081, and an ordinary cast still compiles.
    assert_eq!(
        error_code("3 cast as nosuch:integer").as_deref(),
        Some("XPST0081")
    );
    assert_eq!(run("'3' cast as xs:integer + 1").as_deref(), Ok("4"));
    assert_eq!(run("'x' cast as xs:untypedAtomic").as_deref(), Ok("x"));
    // A *union* type is a simple type, so `xs:numeric` is a legitimate target.
    assert_eq!(run("'12.5' castable as xs:numeric").as_deref(), Ok("true"));
    assert_eq!(
        run("'12.5.7' castable as xs:numeric").as_deref(),
        Ok("false")
    );
}

#[test]
fn a_group_by_type_declaration_constrains_the_atomized_key() -> XQueryResult<()> {
    // `GroupingSpec ::= GroupingVariable (TypeDeclaration? ":=" ExprSingle)? …` — a type
    // declaration exists only inside the group the `:=` closes.
    assert_eq!(
        error_code(
            "for $x in (<e g='f'/>, <e g='m'/>) let $k := $x/@g \
             group by $k as xs:untypedAtomic+ return string($k)"
        )
        .as_deref(),
        Some("XPST0003")
    );
    // §3.12.7: "If a type declaration is present and the resulting atomized value is not an
    // instance of the specified type, a type error is raised [err:XPTY0004]."
    assert_eq!(
        error_code(
            "for $x in (<e g='f'/>, <e g='m'/>) \
             group by $k as attribute(*) := $x/@g return string($k)"
        )
        .as_deref(),
        Some("XPTY0004")
    );
    assert_eq!(
        run("for $x in 1 to 4 group by $k as xs:integer := $x mod 2 order by $k return $k")?,
        "0 1"
    );
    Ok(())
}

#[test]
fn a_nested_xml_base_resolves_against_its_new_parent() -> XQueryResult<()> {
    // §3.9.1.3: a nested constructor's base-uri "is set to be the same as that of its new parent,
    // unless it … has an xml:base attribute, in which case its base-uri property is set to the
    // value of that attribute, resolved (if it is relative) against the base-uri property of its
    // new parent node".
    assert_eq!(
        run(r#"declare base-uri "http://www.example.com/";
               fn:base-uri(exactly-one((<elem xml:base="fluster/"><a xml:base="now"/></elem>)/a))"#)?,
        "http://www.example.com/fluster/now"
    );
    assert_eq!(
        run(r#"declare base-uri "http://www.example.com/";
               fn:base-uri(exactly-one((<elem xml:base="fluster/"><a/></elem>)/a))"#)?,
        "http://www.example.com/fluster/"
    );
    assert_eq!(
        run(r#"declare base-uri "http://x/"; fn:base-uri(<a/>)"#)?,
        "http://x/"
    );
    Ok(())
}

#[test]
fn an_external_function_declaration_is_held_to_the_name_rules() {
    // §4.18 constrains a declared function's *name*, whether or not it has a body: an unprefixed
    // name lands in the default function namespace, which is one the specification reserves.
    assert_eq!(
        error_code("declare function foo() external; 1").as_deref(),
        Some("XQST0045")
    );
    assert_eq!(
        run("declare function local:f() external; 2").as_deref(),
        Ok("2")
    );
}

/// The three built-in list types are cast targets in XQuery exactly as they are in XPath: the
/// query language scans the `SingleType` and hands the expression to the shared evaluator, so
/// `cast as xs:NMTOKENS`, `castable as` and the constructor function all reach the same rule
/// (XPath 3.1 §19.3.4, and QT3's `CastAs-ListType-*`).
#[test]
fn a_list_type_is_a_cast_target_in_xquery() -> XQueryResult<()> {
    assert_eq!(run("'a b c' cast as xs:NMTOKENS")?, "a b c");
    assert_eq!(run("xs:IDREFS(' aa  bb ')")?, "aa bb");
    assert_eq!(run("'a b' castable as xs:ENTITIES")?, "true");
    assert_eq!(run("'' castable as xs:NMTOKENS")?, "false");
    assert_eq!(run("() castable as xs:IDREFS?")?, "true");
    assert_eq!(
        run("let $x as xs:NMTOKEN+ := ('a b' cast as xs:NMTOKENS) return count($x)")?,
        "2"
    );
    assert_eq!(
        error_code("'' cast as xs:IDREFS").as_deref(),
        Some("FORG0001")
    );
    // XQuery 3.0 reports an unknown cast target as `XQST0052` where XPath says `XPST0051`, and
    // a list type must not be mistaken for one of those.
    assert_eq!(
        error_code("'a' cast as xs:nosuchtype").as_deref(),
        Some("XQST0052")
    );
    Ok(())
}

/// A FLWOR is a loop of the host language's own, so it takes the same deadline checkpoint the
/// XPath evaluator's per-item loops do: a declaration whose instant has already passed stops the
/// query at the next tuple instead of running it to the end.
#[test]
fn a_declared_deadline_stops_a_flwor() -> XQueryResult<()> {
    use oxixml_xpath::EvaluationLimits;
    use std::time::{Duration, Instant};

    const QUERY: &str = "count(for $i in 1 to 1000 return $i * 2)";
    assert_eq!(run(QUERY)?, "1000");
    {
        let _limits = EvaluationLimits::declare_deadline(Instant::now() - Duration::from_secs(1));
        let error = run(QUERY).expect_err("the deadline has already passed");
        assert!(
            error.to_string().contains("deadline"),
            "the message names the limit: {error}"
        );
    }
    // The declaration is gone with the guard.
    assert_eq!(run(QUERY)?, "1000");
    Ok(())
}
