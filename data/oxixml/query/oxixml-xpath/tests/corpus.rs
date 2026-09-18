//! A curated corpus: one assertion per feature of the language, evaluated end to end.
//!
//! The expressions here are deliberately small and their expected results are written out by
//! hand from the specification text, so that a regression names the construct it broke.

use oxixml_infoset::{AtomicValue, Tree};
use oxixml_xpath::{StaticContext, XPath, XPathError, XPathVersion, XpathValue};

const CATALOG: &str = r#"<catalog xmlns:dc="urn:dc">
  <book id="b1" dc:year="1965"><title>Dune</title><price>9.99</price></book>
  <book id="b2" dc:year="1815"><title>Emma</title><price>5.50</price></book>
  <!-- a comment --><?render fast?>
  <note xml:lang="en">first</note>
</catalog>"#;

/// Evaluates `expression` over [`CATALOG`] and renders the result as a space-separated string.
fn on_catalog(expression: &str, version: XPathVersion) -> Result<String, XPathError> {
    let tree = Tree::parse_str(CATALOG)?;
    let compiled = XPath::compile(expression, version, StaticContext::new())?;
    let context = compiled.context_with(tree.root());
    Ok(compiled.evaluate_value_in(&context)?.to_string())
}

fn at(expression: &str) -> Result<String, XPathError> {
    on_catalog(expression, XPathVersion::XPath31)
}

fn at10(expression: &str) -> Result<String, XPathError> {
    on_catalog(expression, XPathVersion::XPath10)
}

/// Evaluates a context-free expression under XPath 3.1.
fn free(expression: &str) -> Result<String, XPathError> {
    let compiled = XPath::compile_31(expression)?;
    Ok(compiled.evaluate_value_in(&compiled.context())?.to_string())
}

fn count(expression: &str) -> Result<usize, XPathError> {
    let tree = Tree::parse_str(CATALOG)?;
    let compiled = XPath::compile_31(expression)?;
    Ok(compiled.evaluate_to_nodes(tree.root())?.len())
}

#[test]
fn paths_and_abbreviations() -> Result<(), XPathError> {
    assert_eq!(count("/catalog")?, 1);
    assert_eq!(count("/catalog/book")?, 2);
    assert_eq!(count("//book")?, 2);
    assert_eq!(count("//book/title")?, 2);
    assert_eq!(count("/catalog/book[1]/..")?, 1);
    assert_eq!(count("//@id")?, 2);
    assert_eq!(count("//book/@*")?, 4);
    assert_eq!(count("/")?, 1);
    assert_eq!(count("//title/text()")?, 2);
    assert_eq!(count("//comment()")?, 1);
    assert_eq!(count("//processing-instruction()")?, 1);
    assert_eq!(count("//processing-instruction('render')")?, 1);
    assert_eq!(count("//processing-instruction('other')")?, 0);
    assert_eq!(count("//node()")?, 20);
    Ok(())
}

#[test]
fn every_axis_selects_what_the_specification_says() -> Result<(), XPathError> {
    assert_eq!(count("/catalog/child::book")?, 2);
    assert_eq!(count("/catalog/descendant::title")?, 2);
    assert_eq!(count("//title/parent::book")?, 2);
    assert_eq!(count("//title/ancestor::catalog")?, 1);
    assert_eq!(count("//title/ancestor-or-self::node()")?, 6);
    assert_eq!(count("//book[1]/following-sibling::book")?, 1);
    assert_eq!(count("//book[2]/preceding-sibling::book")?, 1);
    assert_eq!(count("//book[1]/following::title")?, 1);
    assert_eq!(count("//note/preceding::title")?, 2);
    assert_eq!(count("//book[1]/self::book")?, 1);
    assert_eq!(count("//book[1]/attribute::id")?, 1);
    assert_eq!(count("/catalog/descendant-or-self::catalog")?, 1);
    // The namespace axis sees `dc` and the implicit `xml` binding.
    assert_eq!(count("/catalog/namespace::*")?, 2);
    Ok(())
}

#[test]
fn node_tests_and_wildcards() -> Result<(), XPathError> {
    let tree = Tree::parse_str(CATALOG)?;
    let context = StaticContext::new().with_namespace("d", "urn:dc");
    let compiled = XPath::compile("//@d:year", XPathVersion::XPath31, context)?;
    assert_eq!(compiled.evaluate_to_nodes(tree.root())?.len(), 2);

    assert_eq!(count("//*")?, 8);
    assert_eq!(count("//*:title")?, 2);
    assert_eq!(count("//Q{}title")?, 2);
    assert_eq!(count("//element()")?, 8);
    assert_eq!(count("//element(title)")?, 2);
    assert_eq!(count("//attribute::attribute(id)")?, 2);
    assert_eq!(count("/self::document-node()")?, 1);
    assert_eq!(count("/self::document-node(element(catalog))")?, 1);
    Ok(())
}

#[test]
fn predicates_are_positional_or_boolean() -> Result<(), XPathError> {
    assert_eq!(at("//book[1]/@id/string()")?, "b1");
    assert_eq!(at("//book[last()]/@id/string()")?, "b2");
    assert_eq!(at("//book[position() = 2]/@id/string()")?, "b2");
    assert_eq!(at("//book[@id = 'b2']/title/string()")?, "Emma");
    assert_eq!(at("//book[title = 'Dune']/@id/string()")?, "b1");
    assert_eq!(at("(1 to 5)[. mod 2 = 1]")?, "1 3 5");
    // A reverse axis numbers its predicate in reverse document order.
    assert_eq!(at("//note/preceding::title[1]/string()")?, "Emma");
    Ok(())
}

#[test]
fn xpath_10_coercions() -> Result<(), XPathError> {
    assert_eq!(at10("'3' + 4")?, "7");
    assert_eq!(at10("1 div 0")?, "INF");
    assert_eq!(at10("-1 div 0")?, "-INF");
    assert_eq!(at10("0 div 0")?, "NaN");
    assert_eq!(at10("5 mod 2")?, "1");
    // A node-set compared with a number is an existential numeric test.
    assert_eq!(at10("//price = 9.99")?, "true");
    assert_eq!(at10("//price != 9.99")?, "true"); // the *other* price differs
    assert_eq!(at10("not(//price = 9.99)")?, "false");
    assert_eq!(at10("string(//title)")?, "Dune");
    assert_eq!(at10("number(//price)")?, "9.99");
    assert_eq!(at10("count(//book)")?, "2");
    assert_eq!(at10("//book[1]/@id = 'b1'")?, "true");
    // In 1.0 every literal is a double.
    assert_eq!(at10("1")?, "1");
    Ok(())
}

#[test]
fn xpath_20_typing_replaces_the_coercions() -> Result<(), XPathError> {
    assert!(at("'3' + 4").is_err());
    assert!(at("1 idiv 0").is_err());
    assert_eq!(at("1")?, "1");
    assert_eq!(at("1.5")?, "1.5");
    assert_eq!(at("1.5e0")?, "1.5");
    assert_eq!(at("1 + 1.5")?, "2.5");
    assert_eq!(at("(//price)[1] + 0")?, "9.99"); // untypedAtomic promotes to xs:double
    assert_eq!(at("7 idiv 2")?, "3");
    assert_eq!(at("-7 idiv 2")?, "-3");
    assert_eq!(at("() + 1")?, "");
    Ok(())
}

#[test]
fn comparisons() -> Result<(), XPathError> {
    assert_eq!(free("1 eq 1")?, "true");
    assert_eq!(free("1 ne 2")?, "true");
    assert_eq!(free("'a' lt 'b'")?, "true");
    assert_eq!(free("(1, 2) = 2")?, "true");
    assert_eq!(free("(1, 2) != 1")?, "true");
    assert_eq!(free("() eq 1")?, "");
    assert_eq!(free("() = 1")?, "false");
    assert!(free("(1, 2) eq 1").is_err());
    assert_eq!(free("xs:double('NaN') = xs:double('NaN')")?, "false");
    assert_eq!(at("//book[1] is //book[1]")?, "true");
    assert_eq!(at("//book[1] is //book[2]")?, "false");
    assert_eq!(at("//book[1] << //book[2]")?, "true");
    assert_eq!(at("//book[2] >> //book[1]")?, "true");
    Ok(())
}

#[test]
fn sequences_ranges_and_set_operators() -> Result<(), XPathError> {
    assert_eq!(free("(1, 2, 3)")?, "1 2 3");
    assert_eq!(free("(1, (2, 3), ())")?, "1 2 3");
    assert_eq!(free("1 to 4")?, "1 2 3 4");
    assert_eq!(free("4 to 1")?, "");
    assert_eq!(free("reverse(1 to 3)")?, "3 2 1");
    assert_eq!(count("//book | //title")?, 4);
    assert_eq!(count("//book union //book")?, 2);
    assert_eq!(count("//* intersect //book")?, 2);
    assert_eq!(count("//* except //book")?, 6);
    Ok(())
}

#[test]
fn control_flow() -> Result<(), XPathError> {
    assert_eq!(free("if (1 = 1) then 'yes' else 'no'")?, "yes");
    assert_eq!(free("if (0) then 'yes' else 'no'")?, "no");
    assert_eq!(free("for $x in 1 to 3 return $x * $x")?, "1 4 9");
    assert_eq!(
        free("for $x in (1, 2), $y in (10, 20) return $x * $y")?,
        "10 20 20 40"
    );
    assert_eq!(free("let $x := 2 return $x + $x")?, "4");
    assert_eq!(free("let $x := 1, $y := $x + 1 return $y")?, "2");
    assert_eq!(free("some $x in (1, 2, 3) satisfies $x = 2")?, "true");
    assert_eq!(free("every $x in (1, 2, 3) satisfies $x > 0")?, "true");
    assert_eq!(free("every $x in (1, 2, 3) satisfies $x > 1")?, "false");
    assert_eq!(free("(1 to 3) ! (. * 2)")?, "2 4 6");
    Ok(())
}

#[test]
fn the_type_operators() -> Result<(), XPathError> {
    assert_eq!(free("1 instance of xs:integer")?, "true");
    assert_eq!(free("1 instance of xs:string")?, "false");
    assert_eq!(free("() instance of empty-sequence()")?, "true");
    assert_eq!(free("(1, 2) instance of xs:integer+")?, "true");
    assert_eq!(free("'x' instance of item()")?, "true");
    assert_eq!(at("//book[1] instance of element()")?, "true");
    assert_eq!(free("1 treat as xs:integer")?, "1");
    assert!(free("1 treat as xs:string").is_err());
    assert_eq!(free("'12' cast as xs:integer")?, "12");
    assert_eq!(free("'12' castable as xs:integer")?, "true");
    assert_eq!(free("'x' castable as xs:integer")?, "false");
    assert_eq!(free("xs:integer('12') + 1")?, "13");
    assert_eq!(
        free("xs:date('2020-01-02') lt xs:date('2020-02-01')")?,
        "true"
    );
    assert_eq!(
        free("xs:dateTime('2020-01-02T03:04:05Z') cast as xs:date")?,
        "2020-01-02Z"
    );
    Ok(())
}

#[test]
fn function_items_maps_and_arrays() -> Result<(), XPathError> {
    assert_eq!(free("function($x) { $x + 1 }(41)")?, "42");
    assert_eq!(free("for-each(1 to 3, function($n) { $n * $n })")?, "1 4 9");
    assert_eq!(
        free("filter(1 to 6, function($n) { $n mod 2 = 0 })")?,
        "2 4 6"
    );
    assert_eq!(
        free("fold-left(1 to 4, 0, function($a, $b) { $a + $b })")?,
        "10"
    );
    assert_eq!(
        free("fold-right(1 to 3, 0, function($a, $b) { $a - $b })")?,
        "2"
    );
    assert_eq!(
        free("for-each-pair((1, 2), (10, 20), function($a, $b) { $a * $b })")?,
        "10 40"
    );
    assert_eq!(free("sort((3, 1, 2))")?, "1 2 3");
    assert_eq!(free("fn:abs#1(-3)")?, "3");
    assert_eq!(free("function-arity(fn:abs#1)")?, "1");
    assert_eq!(free("substring(?, 2)('abcd')")?, "bcd");
    assert_eq!(free("'abc' => upper-case()")?, "ABC");
    assert_eq!(free("'a' || 'b' || 'c'")?, "abc");

    assert_eq!(free("map { 'a' : 1, 'b' : 2 }?a")?, "1");
    assert_eq!(free("map:size(map { 'a' : 1 })")?, "1");
    assert_eq!(free("[1, 2, 3]?2")?, "2");
    assert_eq!(free("array:size([1, 2, 3])")?, "3");
    assert_eq!(free("array { 1 to 3 }?*")?, "1 2 3");
    assert_eq!(free("[1, 2](2)")?, "2");
    assert_eq!(free("map { 'k' : 'v' }('k')")?, "v");
    assert_eq!(
        free("array:for-each([1, 2], function($n) { $n * 10 })?*")?,
        "10 20"
    );
    assert_eq!(
        free("map:for-each(map { 'a' : 1 }, function($k, $v) { $v })")?,
        "1"
    );
    Ok(())
}

#[test]
fn the_function_library_is_in_scope() -> Result<(), XPathError> {
    assert_eq!(free("concat('a', 'b', 'c')")?, "abc");
    assert_eq!(free("string-join(('a', 'b'), '-')")?, "a-b");
    assert_eq!(free("substring('metadata', 4, 3)")?, "ada");
    assert_eq!(free("upper-case('abc')")?, "ABC");
    assert_eq!(free("contains('tattoo', 'tat')")?, "true");
    assert_eq!(free("tokenize('1, 2', ',\\s*')")?, "1 2");
    assert_eq!(free("matches('abc', '^a')")?, "true");
    assert_eq!(free("replace('abc', 'b', 'X')")?, "aXc");
    assert_eq!(free("count((1, 2, 3))")?, "3");
    assert_eq!(free("sum((1, 2, 3))")?, "6");
    assert_eq!(free("avg((1, 2, 3))")?, "2");
    assert_eq!(free("max((1, 5, 3))")?, "5");
    assert_eq!(free("distinct-values((1, 1, 2))")?, "1 2");
    assert_eq!(free("index-of((10, 20, 30), 20)")?, "2");
    assert_eq!(free("deep-equal((1, 2), (1, 2))")?, "true");
    assert_eq!(free("round(2.5)")?, "3");
    assert_eq!(free("floor(-1.5)")?, "-2");
    assert_eq!(free("abs(-3)")?, "3");
    assert_eq!(free("math:pi() > 3")?, "true");
    assert_eq!(free("math:sqrt(4)")?, "2");
    assert_eq!(free("format-number(12345.6, '#,##0.00')")?, "12,345.60");
    assert_eq!(at("//note/lang('en')")?, "true");
    assert_eq!(at("name(//book[1])")?, "book");
    assert_eq!(at("//book[1]/local-name()")?, "book");
    assert_eq!(at("//book[1]/string-length(@id)")?, "2");
    assert_eq!(at("normalize-space('  a  b  ')")?, "a b");
    Ok(())
}

#[test]
fn context_dependent_functions_read_the_focus() -> Result<(), XPathError> {
    assert_eq!(at("//book/position()")?, "1 2");
    assert_eq!(at("//book/last()")?, "2 2");
    assert_eq!(at("//title/string()")?, "Dune Emma");
    assert_eq!(at("//title/data()")?, "Dune Emma");
    assert_eq!(at("//book[1]/@id/name()")?, "id");
    assert_eq!(at("//book[1]/root()/self::document-node()/count(.)")?, "1");
    assert_eq!(at("//note/normalize-space()")?, "first");
    assert_eq!(at("count(//book/generate-id())")?, "2");
    Ok(())
}

#[test]
fn static_errors_are_raised_at_compile_time() {
    let context = StaticContext::new;
    for source in [
        "1 +",
        "(1",
        "$undeclared",
        "unbound:name",
        "fn:no-such()",
        "1 cast as xs:nope",
        "1 instance of xs:nope",
        "1 2",
        "//",
    ] {
        assert!(
            XPath::compile(source, XPathVersion::XPath31, context()).is_err(),
            "`{source}` must not compile"
        );
    }
}

#[test]
fn dynamic_errors_carry_their_codes() -> Result<(), XPathError> {
    use oxixml_xpath::XPathErrorCode;

    let compiled = XPath::compile_31(".")?;
    assert_eq!(
        compiled
            .evaluate_without_context()
            .expect_err("no context item")
            .code(),
        Some(XPathErrorCode::Xpdy0002)
    );

    let compiled = XPath::compile_31("1 idiv 0")?;
    assert_eq!(
        compiled
            .evaluate_without_context()
            .expect_err("division by zero")
            .code(),
        Some(XPathErrorCode::Foar0001)
    );

    let compiled = XPath::compile_31("'x' cast as xs:integer")?;
    assert_eq!(
        compiled
            .evaluate_without_context()
            .expect_err("bad cast")
            .code(),
        Some(XPathErrorCode::Forg0001)
    );

    let compiled = XPath::compile_31("(1, 2)/child::a")?;
    assert_eq!(
        compiled
            .evaluate_without_context()
            .expect_err("a number has no children")
            .code(),
        Some(XPathErrorCode::Xpty0019)
    );
    Ok(())
}

#[test]
fn variables_come_from_the_dynamic_context() -> Result<(), XPathError> {
    let tree = Tree::parse_str(CATALOG)?;
    let compiled = XPath::compile(
        "//book[@id = $wanted]/title/string()",
        XPathVersion::XPath31,
        StaticContext::new().with_variable("wanted"),
    )?;
    let context = compiled
        .context_with(tree.root())
        .with_variable("wanted", AtomicValue::string("b2"));
    assert_eq!(compiled.evaluate_to_string_in(&context)?, "Emma");

    // Undeclared variables can be deferred to evaluation when the host language needs it.
    let lenient = XPath::compile(
        "$late",
        XPathVersion::XPath31,
        StaticContext::new().with_undeclared_variables(),
    )?;
    let context = lenient
        .context()
        .with_variable("late", AtomicValue::integer(1_i64));
    assert_eq!(lenient.evaluate_to_string_in(&context)?, "1");
    assert!(lenient.evaluate_without_context().is_err());
    Ok(())
}

#[test]
fn results_are_in_document_order_without_duplicates() -> Result<(), XPathError> {
    let tree = Tree::parse_str(CATALOG)?;
    let compiled = XPath::compile_31("//title | //book/title | //title");
    let compiled = compiled?;
    let nodes = compiled.evaluate_to_nodes(tree.root())?;
    assert_eq!(nodes.len(), 2);

    let reversed = XPath::compile_31("//book/(title, ..)")?;
    let value: XpathValue = reversed.evaluate_value_in(&reversed.context_with(tree.root()))?;
    // Nodes produced by a path step come back sorted and deduplicated.
    assert_eq!(value.len(), 3);
    Ok(())
}

#[test]
fn comments_and_whitespace_are_insignificant() -> Result<(), XPathError> {
    assert_eq!(free("1 (: one :) + (: plus :) 2")?, "3");
    assert_eq!(free("  1  +  2  ")?, "3");
    assert_eq!(free("(: (: nested :) :) 1")?, "1");
    Ok(())
}
