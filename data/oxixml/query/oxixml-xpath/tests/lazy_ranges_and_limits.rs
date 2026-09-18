//! What the engine does with a range it must not build, and how deep an evaluation may go.
//!
//! Each test names the W3C *QT3* cases it was written from, so that a change of behaviour points
//! straight at the wording it would contradict. The queries are the catalogue's, reduced to what
//! the assertion needs; the suite itself runs them against its own environments in `qt3.rs`.

use oxixml_infoset::{AtomicValue, Item, NodeRef, Tree};
use oxixml_xpath::{
    DocumentResolver, DynamicContext, EvaluationLimits, MAX_RANGE, XPath, XPathError,
    XPathErrorCode,
};
use std::sync::Arc;

fn compile(expression: &str) -> Result<oxixml_xpath::CompiledXPath, XPathError> {
    XPath::compile_31(expression)
}

/// The value of `expression`, evaluated with no focus, as a string.
fn value(expression: &str) -> String {
    let compiled = compile(expression).expect("the expression compiles");
    let context = compiled.context();
    compiled
        .evaluate_value_in(&context)
        .expect("the expression evaluates")
        .to_string()
}

/// The code `expression` raises, whether at compile time or at run time.
fn code(expression: &str) -> Option<XPathErrorCode> {
    match compile(expression) {
        Err(error) => error.code(),
        Ok(compiled) => {
            let context = compiled.context();
            compiled.evaluate_value_in(&context).err()?.code()
        }
    }
}

// --- general comparisons against a range -------------------------------------------------------

/// A general comparison with a `to` range on one side is answered from the range's two bounds,
/// so a range far larger than the engine would ever build still answers — QT3 `RangeExpr-408c`,
/// `-408d`.
#[test]
fn a_general_comparison_against_a_huge_range_is_answered_from_its_bounds() {
    assert_eq!(value("100000000002 = 100000000000 to 100001000003"), "true");
    assert_eq!(value("100001000002 < 100000000000 to 100001000003"), "true");
    // Nothing but the bounds is read: a predicate that forces the range to be built is refused.
    assert_eq!(
        code("count((100000000000 to 100101000003)[. mod 2 = 0])"),
        Some(XPathErrorCode::Xpdy0130)
    );
}

/// The same, with endpoints past `i64`, where the membership test has to be decided in the
/// arbitrary-precision integer domain rather than through `xs:double` — QT3 `RangeExpr-409c`,
/// `-409d`.
#[test]
fn a_general_comparison_against_an_arbitrary_precision_range_is_exact() {
    assert_eq!(
        value("1000000000000000020001 = 1000000000000000000000 to 1000000000000010000003"),
        "true"
    );
    assert_eq!(
        value("1000000000000000020001 < 1000000000000000000000 to 1000000000000500000003"),
        "true"
    );
    // A value below the first bound is not a member, and the count is exact.
    assert_eq!(
        value("999999999999999999999 = 1000000000000000000000 to 1000000000000010000003"),
        "false"
    );
    assert_eq!(
        value("count(1000000000000000000000 to 1000000000000000000003)"),
        "4"
    );
    // The positional fast path declines an arbitrary-precision range and lets the (short) run be
    // built, which is what keeps `RangeExpr-409`, `-409a` and `-409b` answering.
    assert_eq!(
        value("(1000000000000000000000 to 1000000000000000000003)[2]"),
        "1000000000000000000001"
    );
}

/// The bound arithmetic agrees with what materializing the range would have produced, in both
/// operand orders and for every operator.
#[test]
fn the_bound_arithmetic_agrees_with_a_materialized_range() {
    for (expression, expected) in [
        ("3 = (1 to 5)", "true"),
        ("7 = (1 to 5)", "false"),
        ("2.5 = (1 to 5)", "false"),
        ("xs:double('NaN') = (1 to 5)", "false"),
        ("2.0 = (1 to 5)", "true"),
        ("1 != (1 to 1)", "false"),
        ("1 != (1 to 2)", "true"),
        ("0 < (1 to 5)", "true"),
        ("5 < (1 to 5)", "false"),
        ("5 <= (1 to 5)", "true"),
        ("6 > (1 to 5)", "true"),
        ("1 > (1 to 5)", "false"),
        ("(1 to 5) < 2", "true"),
        ("(1 to 5) < 1", "false"),
        ("(1 to 5) > 4", "true"),
        ("(1 to 5) >= 5", "true"),
        ("(1 to 5) = 5", "true"),
        ("(1 to 5) != 5", "true"),
        // An empty range satisfies nothing.
        ("1 = (5 to 1)", "false"),
        ("(5 to 1) != 1", "false"),
        // A sequence on the other side is existential over both.
        ("(7, 3) = (1 to 5)", "true"),
        ("(7, 9) = (1 to 5)", "false"),
    ] {
        assert_eq!(value(expression), expected, "for `{expression}`");
    }
}

/// An untyped operand takes the range's numeric type, exactly as it does against a single
/// integer.
#[test]
fn an_untyped_operand_compared_with_a_range_is_cast_to_a_number() {
    let tree = Tree::parse_str("<r><a>3</a><b>x</b></r>").expect("the document parses");
    let compiled = compile("/r/a = (1 to 5)").expect("the expression compiles");
    let context = compiled.context_with(tree.root());
    assert_eq!(
        compiled
            .evaluate_value_in(&context)
            .expect("a numeric lexical form casts")
            .to_string(),
        "true"
    );
    let compiled = compile("/r/b = (1 to 5)").expect("the expression compiles");
    let context = compiled.context_with(tree.root());
    assert_eq!(
        compiled
            .evaluate_value_in(&context)
            .expect_err("`x` is not a number")
            .code(),
        Some(XPathErrorCode::Forg0001)
    );
}

/// A value that cannot be compared with an `xs:integer` at all is still a type error, and it is
/// raised for the same pair the ordinary rule would raise it for.
#[test]
fn an_incomparable_operand_against_a_range_is_still_a_type_error() {
    assert_eq!(code("'a' = (1 to 5)"), Some(XPathErrorCode::Xpty0004));
    assert_eq!(code("(1 to 5) = 'a'"), Some(XPathErrorCode::Xpty0004));
    assert_eq!(
        code("xs:date('2000-01-01') < (1 to 5)"),
        Some(XPathErrorCode::Xpty0004)
    );
    // A satisfying pair reached first still short-circuits, in the order the pairs are
    // enumerated: with the range on the left the first sweep pairs `from` with each value in
    // turn, so `2` satisfies before `'a'` can raise.
    assert_eq!(value("(2 to 5) = (2, 'a')"), "true");
    // Reversing the operands reverses the enumeration, and then the type error comes first.
    assert_eq!(code("(2, 'a') = (2 to 5)"), None);
    assert_eq!(code("('a', 2) = (2 to 5)"), Some(XPathErrorCode::Xpty0004));
}

// --- counting a range without building it ------------------------------------------------------

/// `fn:count`, `fn:exists` and `fn:empty` answer from a range's bounds even when the range is
/// one operand of a comma sequence — QT3 `cbcl-count-002`, `cbcl-exists-001`,
/// `cbcl-empty-func-001` (whose XQuery form runs the same expression through declared
/// functions).
#[test]
fn counting_a_sequence_holding_a_huge_range_never_builds_it() {
    assert_eq!(value("count(((), (1 to 10000000), (1, 2, 3)))"), "10000003");
    assert_eq!(value("exists(((), (1 to 10000000)))"), "true");
    assert_eq!(value("empty(((), (1 to 10000000)))"), "false");
    assert_eq!(value("empty(((), (5 to 1)))"), "true");
    // The count is an `xs:integer`, which is unbounded, so a range longer than an `i64` counts.
    assert_eq!(
        value("count(0 to 9223372036854775807)"),
        "9223372036854775808"
    );
    // Small sequences answer the same way they would have if they had been built.
    assert_eq!(value("count((1, (2 to 4), 5))"), "5");
    assert_eq!(value("count(())"), "0");
}

// --- the materialization ceiling ---------------------------------------------------------------

/// The ceiling on a range that really is built clears the whole Unicode code space, so
/// `fn:codepoints-to-string` reaches the invalid codepoint and reports `FOCH0001` rather than
/// being refused — QT3 `cbcl-codepoints-to-string-005`, and the range sizes
/// `Constr-cont-document-3` and `environment-variable-012` need.
#[test]
fn the_materialization_ceiling_clears_the_unicode_code_space() {
    const { assert!(MAX_RANGE > 1_114_112) };
    assert_eq!(
        compile("fn:codepoints-to-string(65536 to 1114112)")
            .expect("the expression compiles")
            .evaluate_value_in(&compile("1").expect("compiles").context())
            .expect_err("1114112 is not an XML character")
            .name()
            .local_name(),
        "FOCH0001"
    );
    assert_eq!(value("count(65536 to 1114111)"), "1048576");
    // Past the ceiling the range is refused, with the code the specification reserves for an
    // implementation-defined limit.
    assert_eq!(
        code("string-join((1 to 2000000000)[. < 3])"),
        Some(XPathErrorCode::Xpdy0130)
    );
}

// --- fn:filter's function coercion -------------------------------------------------------------

/// `fn:filter`'s `$f` is declared `function(item()) as xs:boolean`, so the function conversion
/// rules apply to its result: an untyped node is atomized and cast — QT3 `filter-006`.
#[test]
fn a_filter_predicate_returning_an_untyped_node_is_coerced() {
    assert_eq!(
        value("filter((1 to 6), function($i){ parse-xml(concat('<a>', 1 - ($i mod 2), '</a>')) })"),
        "2 4 6"
    );
    // A typed boolean still works, and a value with no boolean reading is still a type error.
    assert_eq!(
        value("filter((1 to 6), function($i){ $i mod 2 = 0 })"),
        "2 4 6"
    );
    assert_eq!(
        code("filter((1 to 6), function($i){ $i })"),
        Some(XPathErrorCode::Xpty0004)
    );
    // The same helper serves `array:filter`.
    assert_eq!(
        value("array:size(array:filter([1, 2, 3, 4], function($i){ $i mod 2 = 0 }))"),
        "2"
    );
}

// --- collections of items ----------------------------------------------------------------------

#[derive(Debug)]
struct AtomicCollection;

impl DocumentResolver for AtomicCollection {
    fn document(&self, _uri: &str) -> Option<NodeRef> {
        None
    }

    fn collection_items(&self, uri: Option<&str>) -> Option<Vec<Item>> {
        match uri {
            None => Some(vec![Item::Atomic(AtomicValue::string("goodbye"))]),
            Some("urn:integers") => Some(
                (1..=3)
                    .map(|value| Item::Atomic(AtomicValue::integer(value)))
                    .collect(),
            ),
            Some(_) => None,
        }
    }
}

/// A collection is "a sequence of items", not a sequence of nodes, so a resolver may answer with
/// atomic values — QT3 `cbcl-collection-002`, `-003`, `-004`.
#[test]
fn a_collection_may_hold_atomic_items() {
    let compiled = compile("fn:collection('urn:integers')").expect("the expression compiles");
    let context = DynamicContext::new(Arc::clone(compiled.static_context()))
        .with_document_resolver(Arc::new(AtomicCollection));
    assert_eq!(
        compiled
            .evaluate_value_in(&context)
            .expect("the collection resolves")
            .to_string(),
        "1 2 3"
    );

    let compiled = compile("fn:collection()").expect("the expression compiles");
    let context = DynamicContext::new(Arc::clone(compiled.static_context()))
        .with_document_resolver(Arc::new(AtomicCollection));
    assert_eq!(
        compiled
            .evaluate_value_in(&context)
            .expect("the default collection resolves")
            .to_string(),
        "goodbye"
    );

    // A URI that names nothing is still `FODC0002`, and `fn:uri-collection` of an atomic
    // collection is empty rather than an error.
    let compiled = compile("fn:collection('urn:missing')").expect("the expression compiles");
    let context = DynamicContext::new(Arc::clone(compiled.static_context()))
        .with_document_resolver(Arc::new(AtomicCollection));
    assert_eq!(
        compiled
            .evaluate_value_in(&context)
            .expect_err("nothing is registered under that URI")
            .code(),
        Some(XPathErrorCode::Fodc0002)
    );
    let compiled = compile("fn:uri-collection('urn:integers')").expect("the expression compiles");
    let context = DynamicContext::new(Arc::clone(compiled.static_context()))
        .with_document_resolver(Arc::new(AtomicCollection));
    assert!(
        compiled
            .evaluate_value_in(&context)
            .expect("the collection resolves")
            .is_empty()
    );
}

#[derive(Debug)]
struct NodeCollection(Tree);

impl DocumentResolver for NodeCollection {
    fn document(&self, _uri: &str) -> Option<NodeRef> {
        None
    }

    fn collection(&self, uri: Option<&str>) -> Option<Vec<NodeRef>> {
        uri.is_none().then(|| vec![self.0.root()])
    }
}

/// A resolver written before collections could hold atomic items keeps working: the default
/// [`DocumentResolver::collection_items`] lifts its nodes.
#[test]
fn a_node_only_resolver_still_answers_fn_collection() {
    let tree = Tree::parse_str("<a>text</a>").expect("the document parses");
    let compiled = compile("fn:collection()/string()").expect("the expression compiles");
    let context = DynamicContext::new(Arc::clone(compiled.static_context()))
        .with_document_resolver(Arc::new(NodeCollection(tree)));
    assert_eq!(
        compiled
            .evaluate_value_in(&context)
            .expect("the default collection resolves")
            .to_string(),
        "text"
    );
}

// --- how deep an evaluation may go -------------------------------------------------------------

/// The call budget counts *user-visible* calls, so a recursive function reaches the depth the
/// query is written for when the thread has room for it — the accounting QT3's
/// `fn-random-number-generator-21` (200 calls) and `numberformat122` (5000) need.
#[test]
fn the_call_budget_counts_user_visible_calls() {
    let deep = |stack: usize, calls: usize| -> Option<String> {
        let expression = format!(
            "let $f := function($self, $n) {{ if ($n <= 0) then 0 else $self($self, $n - 1) }} \
             return $f($f, {calls})"
        );
        std::thread::Builder::new()
            .stack_size(stack)
            .spawn(move || {
                let _limits = EvaluationLimits::declare(stack);
                let compiled = XPath::compile_31(&expression).expect("the expression compiles");
                let context = compiled.context();
                compiled.evaluate_to_string_in(&context).ok()
            })
            .expect("the helper thread starts")
            .join()
            .expect("the helper thread finishes")
    };
    // Two hundred calls fit inside the default ceiling on an ordinary thread.
    assert_eq!(deep(64 * 1024 * 1024, 200), Some("0".to_owned()));
    // Five thousand need a thread that was sized for them, and the ceiling follows the size.
    assert_eq!(deep(512 * 1024 * 1024, 5_000), Some("0".to_owned()));
    assert!(EvaluationLimits::max_call_depth() >= oxixml_xpath::DEFAULT_MAX_CALL_DEPTH);
}

/// A declaration nests, and the ceiling never falls below the default however small the declared
/// stack is.
#[test]
fn a_declaration_nests_and_never_lowers_the_default_ceiling() {
    let outer = EvaluationLimits::declare(64 * 1024 * 1024);
    let raised = EvaluationLimits::max_call_depth();
    assert!(raised > oxixml_xpath::DEFAULT_MAX_CALL_DEPTH);
    {
        let _inner = EvaluationLimits::declare(1024);
        assert_eq!(
            EvaluationLimits::max_call_depth(),
            oxixml_xpath::DEFAULT_MAX_CALL_DEPTH
        );
        assert_eq!(EvaluationLimits::stack_size(), 1024);
    }
    assert_eq!(EvaluationLimits::max_call_depth(), raised);
    drop(outer);
    let explicit = EvaluationLimits::declare_with_max_call_depth(8 * 1024 * 1024, 7);
    assert_eq!(EvaluationLimits::max_call_depth(), 7);
    drop(explicit);
}

/// The guards still stop a runaway recursion, whichever of the two runs out first.
#[test]
fn a_runaway_recursion_is_still_bounded() {
    let outcome = std::thread::Builder::new()
        .stack_size(oxixml_xpath::DEFAULT_STACK_SIZE)
        .spawn(|| {
            let compiled = XPath::compile_31(
                "let $f := function($self, $n) { if ($n <= 0) then 0 else $n + $self($self, $n - 1) } \
                 return $f($f, 100000000)",
            )
            .expect("the expression compiles");
            let context = compiled.context();
            compiled
                .evaluate_value_in(&context)
                .err()
                .and_then(|error| error.code())
        })
        .expect("the helper thread starts")
        .join()
        .expect("the helper thread finishes");
    assert_eq!(outcome, Some(XPathErrorCode::Xpdy0130));
}

/// A declaration reaches the *function library's* budget as well as this engine's.
///
/// A call to a function [`oxixml_fn`] holds — every built-in, and every function a host language
/// registers with [`oxixml_fn::FunctionRegistry`], which is how an XQuery `declare function` and
/// an XSLT `xsl:function` are reached — is counted by `oxixml_fn::function::CallLimits` rather
/// than by this crate's own guard. The two are one budget only if a declaration made here is
/// mirrored there: without it a host that declares a 512 MiB thread raises this engine's ceiling
/// to 65 536 and leaves the library's at 512, so a query recursing five thousand calls deep
/// through declared functions still stops — which is what QT3's `numberformat121` and
/// `numberformat122` do.
#[test]
fn a_declaration_reaches_the_function_librarys_budget() {
    use oxixml_fn::function::{CallLimits, DEFAULT_MAX_CALL_DEPTH, DEFAULT_STACK_SIZE};

    assert_eq!(CallLimits::max_call_depth(), DEFAULT_MAX_CALL_DEPTH);
    assert_eq!(CallLimits::stack_size(), DEFAULT_STACK_SIZE);
    {
        let _limits = EvaluationLimits::declare(512 * 1024 * 1024);
        assert_eq!(CallLimits::stack_size(), 512 * 1024 * 1024);
        assert_eq!(
            CallLimits::max_call_depth(),
            EvaluationLimits::max_call_depth()
        );
        assert!(CallLimits::max_call_depth() >= 5_000);
        {
            // A nested declaration nests on both sides, and neither ceiling ever falls below the
            // default however small the declared stack is.
            let _inner = EvaluationLimits::declare(1024);
            assert_eq!(CallLimits::max_call_depth(), DEFAULT_MAX_CALL_DEPTH);
            assert_eq!(CallLimits::stack_size(), 1024);
        }
        assert!(CallLimits::max_call_depth() >= 5_000);
        // An explicit ceiling is mirrored verbatim rather than re-derived.
        let explicit = EvaluationLimits::declare_with_max_call_depth(8 * 1024 * 1024, 7);
        assert_eq!(CallLimits::max_call_depth(), 7);
        drop(explicit);
    }
    assert_eq!(CallLimits::max_call_depth(), DEFAULT_MAX_CALL_DEPTH);
    assert_eq!(CallLimits::stack_size(), DEFAULT_STACK_SIZE);
}
