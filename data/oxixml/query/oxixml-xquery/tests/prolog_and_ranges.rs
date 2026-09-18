//! Per-module prolog components, prolog defaults, and the queries that must not build a range.
//!
//! Each test names the W3C *QT3* cases it was written from, so that a change of behaviour points
//! straight at the wording it would contradict. The queries are the catalogue's, reduced to what
//! the assertion needs; the suite itself runs them against its own environments in `qt3.rs`.

use oxixml_infoset::{AtomicType, AtomicValue, ExpandedName};
use oxixml_xpath::XpathValue;
use oxixml_xquery::{
    DynamicQueryContext, InMemoryModules, StaticQueryContext, XQuery, XQueryErrorCode, XQueryResult,
};
use std::sync::Arc;

fn run(query: &str) -> XQueryResult<String> {
    XQuery::compile_default(query)?.execute_to_string(&DynamicQueryContext::new())
}

fn value(query: &str) -> String {
    run(query).expect("the query evaluates")
}

fn code_of(query: &str) -> Option<XQueryErrorCode> {
    run(query).err().and_then(|error| error.code())
}

// --- ranges the query must not build -----------------------------------------------------------

/// `fn:count`, `fn:exists` and `fn:empty` over a comma sequence holding a ten-million-item range
/// answer from its bounds, with the other operands evaluated exactly as the sequence itself would
/// have evaluated them — QT3 `cbcl-count-002`, `cbcl-exists-001`, `cbcl-empty-func-001`.
#[test]
fn counting_a_sequence_holding_a_huge_range_never_builds_it() {
    let generate = "declare function local:generate($arg as xs:integer?) \
                    { if ($arg = 0) then (1, 2, 3) else $arg }; ";
    let sequence = "( (), local:generate( () ), local:generate( 0 ), (1 to 10000000), \
                    local:generate( () ), local:generate(1))";
    assert_eq!(
        value(&format!("{generate} fn:count({sequence})")),
        "10000004"
    );
    assert_eq!(value(&format!("{generate} fn:exists({sequence})")), "true");
    assert_eq!(value(&format!("{generate} fn:empty({sequence})")), "false");
}

/// A general comparison against a range is answered from its two bounds, whether the endpoints
/// fit an `i64` or not — QT3 `RangeExpr-408c`, `-408d`, `-409c`, `-409d`.
#[test]
fn a_general_comparison_against_a_huge_range_is_answered_from_its_bounds() {
    assert_eq!(value("100000000002 = 100000000000 to 100001000003"), "true");
    assert_eq!(value("100001000002 < 100000000000 to 100001000003"), "true");
    assert_eq!(
        value("1000000000000000020001 = 1000000000000000000000 to 1000000000000010000003"),
        "true"
    );
    assert_eq!(
        value("1000000000000000020001 < 1000000000000000000000 to 1000000000000500000003"),
        "true"
    );
}

/// A range that really is built may span the whole Unicode code space, so
/// `fn:codepoints-to-string` reaches the invalid codepoint — QT3
/// `cbcl-codepoints-to-string-005` — and a million-item `for` binding runs — QT3
/// `environment-variable-012`.
#[test]
fn a_range_over_the_unicode_code_space_is_built_rather_than_refused() {
    assert_eq!(
        run("fn:codepoints-to-string( 65536 to 1114112 )")
            .expect_err("1114112 is not an XML character")
            .name()
            .local_name(),
        "FOCH0001"
    );
    assert_eq!(
        value(
            "let $var := string-join(for $i in (1 to (1024 * 1024)) return \"x\") \
             return string-length($var)"
        ),
        "1048576"
    );
}

/// The three ranges `Constr-cont-document-3` builds add up to more than a million codepoints,
/// which the constructor's content has to carry through `fn:codepoints-to-string`.
#[test]
fn a_constructor_carries_a_million_codepoints() {
    let query = "declare variable $codepoints as xs:integer+ := \
                 (9, 10, 13, 32 to 55295, 57344 to 65532, 65536 to 1114111 ); \
                 <allCodepoints><r>{codepoints-to-string($codepoints)}</r></allCodepoints> \
                 /string-length(string(.))";
    assert_eq!(value(query), "1112032");
}

// --- decimal formats are scoped to their module -------------------------------------------------

/// A decimal format declared in a library module is not displaced by a same-named declaration in
/// the module that imported it — XQuery 3.1 §4.17, QT3 `decimal-format-21`.
#[test]
fn decimal_formats_apply_only_within_their_own_module() {
    let module = "module namespace m = \"http://www.w3.org/TestModules/dfd-module-001\";\n\
                  declare decimal-format df001 grouping-separator=\"'\";\n\
                  declare function m:do() as xs:string {\n\
                    fn:format-number(123456.789, \"#'###'###.###\", 'df001')\n\
                  };";
    let mut modules = InMemoryModules::new();
    modules.insert("http://www.w3.org/TestModules/dfd-module-001", module);
    let statik = StaticQueryContext::new().with_module_resolver(Arc::new(modules));
    let query = "import module namespace m=\"http://www.w3.org/TestModules/dfd-module-001\";\n\
                 declare decimal-format df001 grouping-separator=\"!\";\n\
                 format-number(123456.789,'#!###!###.###','df001')||'-'||m:do()";
    let compiled = XQuery::compile(query, statik).expect("the query compiles");
    assert_eq!(
        compiled
            .execute_to_string(&DynamicQueryContext::new())
            .expect("the query evaluates"),
        "123!456.789-123'456.789"
    );
}

/// Declaring the same format twice in one prolog is still `XQST0111`, and the check is per
/// module rather than per program.
#[test]
fn one_prolog_may_not_declare_a_format_twice() {
    assert_eq!(
        code_of(
            "declare decimal-format df grouping-separator=\"!\"; \
             declare decimal-format df grouping-separator=\"'\"; 1"
        ),
        Some(XQueryErrorCode::Xqst0111)
    );
}

/// A prefixed decimal-format name is resolved against the namespaces in scope at the call site,
/// which include those a direct element constructor declared — QT3 `eqname-007`.
#[test]
fn a_constructor_namespace_resolves_a_decimal_format_name() {
    assert_eq!(
        value(
            "declare decimal-format Q{http://www.example.com/ns}format grouping-separator=\"'\"; \
             string(<a xmlns:ex=\"http://www.example.com/ns\">\
             {format-number(1e9, \"#'###'###'##0.00\", 'ex:format')}</a>)"
        ),
        "1'000'000'000.00"
    );
}

// --- fn:filter's function coercion --------------------------------------------------------------

/// `fn:filter`'s `$f` is declared `function(item()) as xs:boolean`, so the function conversion
/// rules apply to its result and an untyped element node is atomized and cast — QT3 `filter-006`.
#[test]
fn a_filter_predicate_returning_an_untyped_element_is_coerced() {
    assert_eq!(
        value("filter((1 to 20), function($i){<a>{1 - ($i mod 2)}</a>})"),
        "2 4 6 8 10 12 14 16 18 20"
    );
}

// --- prolog defaults ------------------------------------------------------------------------------

/// `declare context item ... := E` evaluates `E` when the caller supplies no context item, and a
/// declared type is enforced against a supplied one — the engine half of QT3 `contextDecl-020`,
/// `contextDecl-047` and `contextDecl-058`.
#[test]
fn a_declared_context_item_takes_its_default_and_its_type() {
    assert_eq!(value("declare context item as xs:integer := 42; ."), "42");
    // A supplied value of the wrong type is `XPTY0004`, not "there is no context item".
    let supplied = DynamicQueryContext::new().with_context_item(AtomicValue::string("London"));
    let compiled = XQuery::compile_default("declare context item as xs:integer external; .")
        .expect("the query compiles");
    assert_eq!(
        compiled
            .execute_to_string(&supplied)
            .expect_err("a string does not match xs:integer")
            .code(),
        Some(XQueryErrorCode::Xpty0004)
    );
    // Absence with no default is `XPDY0002`.
    assert_eq!(
        code_of("declare context item as xs:integer external; ."),
        Some(XQueryErrorCode::Xpdy0002)
    );
    // A supplied value that *does* match is used, which is the shape `contextDecl-047` needs.
    let date = DynamicQueryContext::new().with_context_item(
        AtomicValue::parse(AtomicType::Date, "2024-01-01").expect("a date parses"),
    );
    let compiled =
        XQuery::compile_default(". gt xs:date('1900-01-01')").expect("the query compiles");
    assert_eq!(
        compiled.execute_to_string(&date).expect("it evaluates"),
        "true"
    );
}

/// An external variable uses its declared default when the caller supplies none, and takes a
/// caller-supplied value under its *expanded* name however the query spells the prefix — the
/// engine half of QT3 `extvardeclwithouttype-24` and `extvardeclwithtype-24`.
#[test]
fn an_external_variable_takes_its_default_and_its_expanded_name() {
    assert_eq!(
        value("declare variable $x as xs:integer external := 7; $x"),
        "7"
    );
    assert_eq!(
        code_of("declare variable $x as xs:integer external; $x"),
        Some(XQueryErrorCode::Xpdy0002)
    );
    let bound = DynamicQueryContext::new().with_variable_name(
        ExpandedName::new(Some("http://www.w3.org/fots/example-test-namespace"), "x"),
        XpathValue::from(AtomicValue::integer(2_i64)),
    );
    for query in [
        "declare namespace test = \"http://www.w3.org/fots/example-test-namespace\"; \
         declare variable $test:x external; $test:x",
        "declare namespace test = \"http://www.w3.org/fots/example-test-namespace\"; \
         declare variable $test:x as xs:integer external; $test:x",
        "declare namespace other = \"http://www.w3.org/fots/example-test-namespace\"; \
         declare variable $other:x external; $other:x",
    ] {
        let compiled = XQuery::compile_default(query).expect("the query compiles");
        assert_eq!(
            compiled.execute_to_string(&bound).expect("it evaluates"),
            "2",
            "for `{query}`"
        );
    }
}

// --- how deep a query may recurse ---------------------------------------------------------------

/// The call budget counts *user-visible* calls, so a query recurses as deeply as the thread it
/// was given room for — the accounting QT3 `fn-random-number-generator-21` and `-22` need.
#[test]
fn two_hundred_declared_calls_recurse() {
    const STACK: usize = 64 * 1024 * 1024;
    let query = "declare namespace r='http://example.com/random/'; \
        declare %public function r:random-sequence($length as xs:integer) as xs:double* { \
          r:random-sequence($length, fn:random-number-generator()) \
        }; \
        declare %private function r:random-sequence($length as xs:integer, \
                                                    $G as map(xs:string, item())) { \
          if ($length eq 0) then () else ($G?number, r:random-sequence($length - 1, $G?next())) \
        }; \
        count(r:random-sequence(200))";
    let counted = std::thread::Builder::new()
        .stack_size(STACK)
        .spawn(move || {
            XQuery::compile_default(query)?
                .execute_to_string(&DynamicQueryContext::new().with_stack_size(STACK))
        })
        .expect("the helper thread starts")
        .join()
        .expect("the helper thread finishes")
        .expect("two hundred calls are within the budget");
    assert_eq!(counted, "200");
}

/// A runaway recursion is still bounded rather than aborting the process, whichever guard runs
/// out first.
#[test]
fn a_runaway_recursion_is_still_bounded() {
    let code = std::thread::Builder::new()
        .stack_size(oxixml_xquery::DEFAULT_STACK_SIZE)
        .spawn(|| {
            XQuery::compile_default(
                "declare function local:f($n as xs:integer) { local:f($n + 1) }; local:f(0)",
            )
            .expect("the query compiles")
            .execute_to_string(&DynamicQueryContext::new())
            .err()
            .and_then(|error| error.code())
        })
        .expect("the helper thread starts")
        .join()
        .expect("the helper thread finishes");
    assert_eq!(code, Some(XQueryErrorCode::Xpdy0130));
}

// --- output declarations -----------------------------------------------------------------------
//
// XQuery 3.1 §2.2.4 lets a main module set a serialization parameter with `declare option
// output:name "value"`. Nothing in this engine serializes, so the declarations are validated
// while compiling and then handed to whoever does — QT3's whole `ser/` catalogue asserts against
// "the result of the query serialized using the serialization options specified within the query".

const OUTPUT: &str = "http://www.w3.org/2010/xslt-xquery-serialization";

fn compiled(query: &str) -> oxixml_xquery::CompiledQuery {
    XQuery::compile_default(query).expect("the query compiles")
}

#[test]
fn a_prolog_without_an_output_declaration_declares_nothing() -> XQueryResult<()> {
    let compiled = compiled("<a/>");
    assert!(compiled.output_declarations().is_empty());
    assert_eq!(
        compiled.serialization_parameters()?,
        oxixml_fn::xmlio::SerializationParameters::default()
    );
    Ok(())
}

#[test]
fn output_declarations_are_read_in_order_and_by_local_name() {
    let query = format!(
        "declare namespace o = \"{OUTPUT}\";
         declare namespace p = \"urn:p\";
         declare option p:ignored \"x\";
         declare option o:method \"html\";
         declare option Q{{{OUTPUT}}}indent \"yes\";
         declare option unprefixed \"x\";
         <a/>"
    );
    assert_eq!(
        compiled(&query).output_declarations(),
        [
            ("method".to_owned(), "html".to_owned()),
            ("indent".to_owned(), "yes".to_owned()),
        ]
    );
}

#[test]
fn a_qname_valued_parameter_resolves_against_the_prologs_prefixes() -> XQueryResult<()> {
    let query = format!(
        "declare namespace o = \"{OUTPUT}\";
         declare namespace p = \"urn:p\";
         declare option o:cdata-section-elements \"p:a\";
         <a/>"
    );
    let parameters = compiled(&query).serialization_parameters()?;
    assert_eq!(
        parameters.cdata_section_elements(),
        [ExpandedName::new(Some("urn:p"), "a")]
    );
    Ok(())
}

/// An *unprefixed* name in a QName-valued parameter denotes a name in the module's default
/// element namespace, not one in no namespace — QT3 `method-xml/K2-Serialization-28` and `-31`.
/// The declaration cannot travel in the parameter document, so the compiled query carries it.
#[test]
fn an_unprefixed_qname_takes_the_modules_default_element_namespace() -> XQueryResult<()> {
    let query = format!(
        "declare namespace o = \"{OUTPUT}\";
         declare default element namespace \"urn:d\";
         declare namespace p = \"urn:p\";
         declare option o:cdata-section-elements \"b p:a\";
         declare option o:suppress-indentation \"para\";
         <a/>"
    );
    let parameters = compiled(&query).serialization_parameters()?;
    assert_eq!(
        parameters.cdata_section_elements(),
        [
            ExpandedName::new(Some("urn:d"), "b"),
            ExpandedName::new(Some("urn:p"), "a"),
        ]
    );
    assert_eq!(
        parameters.suppress_indentation(),
        [ExpandedName::new(Some("urn:d"), "para")]
    );

    // A module that declares no default element namespace is unchanged: the unprefixed name is
    // in no namespace, which is what the rest of the K2-Serialization set requires.
    let without = format!(
        "declare namespace o = \"{OUTPUT}\";
         declare option o:cdata-section-elements \"b\";
         <a/>"
    );
    assert_eq!(
        compiled(&without)
            .serialization_parameters()?
            .cdata_section_elements(),
        [ExpandedName::new(None::<String>, "b")]
    );
    Ok(())
}

/// Compiling is unchanged by the accessor: an output declaration whose *value* is inadmissible is
/// still a query that compiles, because `XQST0109` and `XQST0110` are what the prolog grammar
/// requires and what values a parameter admits belongs to serialization.
#[test]
fn an_inadmissible_value_is_reported_only_when_it_is_asked_for() {
    let query =
        format!("declare namespace o = \"{OUTPUT}\"; declare option o:indent \"maybe\"; <a/>");
    let compiled = compiled(&query);
    assert_eq!(
        compiled.output_declarations(),
        [("indent".to_owned(), "maybe".to_owned())]
    );
    let failure = compiled
        .serialization_parameters()
        .expect_err("`maybe` is not a yes/no value");
    assert_eq!(failure.name().local_name(), "SEPM0017");
}

/// A declared value keeps every character it was written with, including the whitespace an
/// attribute-value normalization would otherwise fold — QT3 `ser/prod-OptionDecl.serialization`
/// declares a tab as `item-separator`.
#[test]
fn a_declared_value_keeps_its_whitespace_and_markup() -> XQueryResult<()> {
    let query = format!(
        "declare namespace o = \"{OUTPUT}\";
         declare option o:item-separator \"&#9;\";
         declare option o:doctype-system \"a&lt;b&amp;c\";
         <a/>"
    );
    let parameters = compiled(&query).serialization_parameters()?;
    assert_eq!(parameters.item_separator(), Some("\t".to_owned()));
    assert_eq!(parameters.doctype_system(), Some("a<b&c".to_owned()));
    Ok(())
}
