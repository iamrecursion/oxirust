//! Regression tests for the audit findings against `oxixml-xpath`.
//!
//! * the recursive-descent parser had no nesting-depth guard, so a few hundred bytes of nested
//!   parentheses aborted the *process* with a stack overflow;
//! * a flat but very long path or operator chain, parsed iteratively, overflowed the stack on
//!   the derived recursive `Drop` when the compiled expression was freed;
//! * the same flat chain, once compiled, was *evaluated* recursively — one native frame per
//!   operator or step — and aborted the process on an expression the grammar does not bound;
//! * a recursive inline function had no call-depth budget, so a runaway recursion aborted the
//!   process rather than raising a dynamic error a host could catch;
//! * a general comparison against a number treated a non-numeric `xs:untypedAtomic` operand as
//!   `NaN` instead of raising `FORG0001`;
//! * casting a syntactically valid but too-large decimal from a string reported `FORG0001`
//!   instead of the range error `FOCA0001`;
//! * `fn:apply` reported an arity mismatch as `XPTY0004` instead of `FOAP0001`;
//! * a typed function test compared a built-in's arity and nothing else, so `fn:abs#1` satisfied
//!   `function(xs:string) as xs:date`.

use oxixml_infoset::Tree;
use oxixml_xpath::{
    EvaluationLimits, StaticContext, XPath, XPathError, XPathErrorCode, XPathVersion,
};
use oxixml_xsd::TimezoneOffset;

fn compile(expression: &str) -> Result<oxixml_xpath::CompiledXPath, XPathError> {
    XPath::compile(expression, XPathVersion::XPath31, StaticContext::new())
}

/// Runs `body` on a thread with room for the deepest expression the parser still accepts.
///
/// An unoptimized build spends roughly 80 KiB of stack per nesting level, which is comfortable
/// inside a default 8 MiB main-thread stack but not inside the much smaller one the test
/// harness gives each test, so the cases that deliberately approach the limit get their own.
fn with_room<T: Send + 'static>(body: impl FnOnce() -> T + Send + 'static) -> T {
    std::thread::Builder::new()
        .stack_size(64 * 1024 * 1024)
        .spawn(body)
        .expect("the helper thread starts")
        .join()
        .expect("the helper thread finishes")
}

fn code_of(expression: &str, document: &str) -> Option<XPathErrorCode> {
    let tree = Tree::parse_str(document).expect("the document parses");
    let compiled = compile(expression).expect("the expression compiles");
    let context = compiled.context_with(tree.root());
    compiled
        .evaluate_value_in(&context)
        .err()
        .and_then(|error: XPathError| -> Option<XPathErrorCode> { error.code() })
}

fn value_of(expression: &str, document: &str) -> String {
    let tree = Tree::parse_str(document).expect("the document parses");
    let compiled = compile(expression).expect("the expression compiles");
    let context = compiled.context_with(tree.root());
    compiled
        .evaluate_value_in(&context)
        .expect("the expression evaluates")
        .to_string()
}

// --- parser nesting depth ----------------------------------------------------------------------

#[test]
fn a_deeply_nested_parenthesized_expression_is_a_static_error_not_a_crash() {
    for depth in [200_usize, 1_000, 100_000] {
        let expression = format!("{}1{}", "(".repeat(depth), ")".repeat(depth));
        let code = with_room(move || compile(&expression).err().and_then(|error| error.code()));
        assert_eq!(code, Some(XPathErrorCode::Xpst0003), "at depth {depth}");
    }
}

#[test]
fn deep_nesting_through_predicates_and_arguments_is_bounded_too() {
    let depth = 5_000;
    let predicates = format!("a{}{}", "[a".repeat(depth), "]".repeat(depth));
    let arguments = format!("{}1{}", "count((".repeat(depth), "))".repeat(depth));
    let codes = with_room(move || {
        (
            compile(&predicates).err().and_then(|error| error.code()),
            compile(&arguments).err().and_then(|error| error.code()),
        )
    });
    assert_eq!(codes.0, Some(XPathErrorCode::Xpst0003));
    assert_eq!(codes.1, Some(XPathErrorCode::Xpst0003));
}

#[test]
fn deep_but_legal_nesting_still_compiles() {
    let depth = 60;
    let expression = format!("{}1{}", "(".repeat(depth), ")".repeat(depth));
    assert!(with_room(move || compile(&expression).is_ok()));
}

// --- freeing a long flat chain -----------------------------------------------------------------

#[test]
fn a_hundred_thousand_path_steps_compile_and_are_freed_without_recursing() {
    with_room(|| {
        let expression = std::iter::repeat_n("a", 100_000)
            .collect::<Vec<_>>()
            .join("/");
        let compiled = compile(&expression).expect("a flat path chain is parsed iteratively");
        // Dropping used to overflow the stack: the derived `Drop` recursed once per step.
        drop(compiled);
    });
}

#[test]
fn a_hundred_thousand_operators_compile_and_are_freed_without_recursing() {
    with_room(|| {
        let expression = std::iter::repeat_n("1", 100_000)
            .collect::<Vec<_>>()
            .join("+");
        let compiled = compile(&expression).expect("a flat operator chain is parsed iteratively");
        drop(compiled);
    });
}

// --- untypedAtomic against a number ------------------------------------------------------------

#[test]
fn a_non_numeric_untyped_operand_compared_with_a_number_is_forg0001() {
    let document = "<r><a>notanumber</a></r>";
    assert_eq!(
        code_of("/r/a = 1", document),
        Some(XPathErrorCode::Forg0001)
    );
    assert_eq!(
        code_of("/r/a < 1", document),
        Some(XPathErrorCode::Forg0001)
    );
    assert_eq!(
        code_of("xs:untypedAtomic('x') = 1", document),
        Some(XPathErrorCode::Forg0001)
    );
    // The non-numeric branch already behaved this way, and still does.
    assert_eq!(
        code_of("xs:untypedAtomic('x') = xs:date('2000-01-01')", document),
        Some(XPathErrorCode::Forg0001)
    );
}

#[test]
fn a_numeric_untyped_operand_still_compares_by_value() {
    let document = "<r><a>2</a></r>";
    let tree = Tree::parse_str(document).expect("the document parses");
    let compiled = compile("/r/a = 2").expect("the expression compiles");
    let context = compiled.context_with(tree.root());
    assert_eq!(
        compiled
            .evaluate_value_in(&context)
            .expect("a numeric lexical form casts")
            .to_string(),
        "true"
    );
}

// --- casting a value past the fixed width from a string -----------------------------------------
//
// `xs:integer` and `xs:decimal` are unbounded, so a lexical form too large for the fixed-width
// backing is carried at arbitrary precision rather than rejected. A *bounded* restriction, a
// malformed form, and a facet violation still error.

#[test]
fn a_string_that_overflows_the_fixed_width_decimal_promotes() {
    let document = "<r/>";
    assert_eq!(
        value_of(
            "xs:decimal('99999999999999999999999999999999999999999999.5')",
            document
        ),
        "99999999999999999999999999999999999999999999.5"
    );
    // A genuinely malformed lexical form stays FORG0001.
    assert_eq!(
        code_of("xs:decimal('12x34')", document),
        Some(XPathErrorCode::Forg0001)
    );
    // As does a value a *facet* rejects, rather than the range.
    assert_eq!(
        code_of("xs:nonNegativeInteger('-1')", document),
        Some(XPathErrorCode::Forg0001)
    );
}

#[test]
fn a_string_that_overflows_the_fixed_width_integer_promotes() {
    assert_eq!(
        value_of("xs:integer('999999999999999999999999999')", "<r/>"),
        "999999999999999999999999999"
    );
    // A bounded restriction still rejects a value past its range.
    assert_eq!(
        code_of("xs:long('99999999999999999999')", "<r/>"),
        Some(XPathErrorCode::Forg0001)
    );
}

// --- implicit timezone in value and general comparisons ----------------------------------------
//
// F&O 3.1 §10.4.4 (and the `op:*-equal`/`op:*-less-than` functions it backs) reads a date/time
// operand with no timezone in the *dynamic context's* implicit timezone before comparing — not
// always UTC. A context whose implicit timezone is left at the default UTC cannot tell a
// processor that threads the context through from one that quietly hardcodes UTC, so every case
// here pins a non-UTC implicit timezone, `-05:00`.

fn compile_31(expression: &str) -> oxixml_xpath::CompiledXPath {
    XPath::compile(expression, XPathVersion::XPath31, StaticContext::new())
        .expect("the expression compiles")
}

fn minus_five() -> TimezoneOffset {
    TimezoneOffset::new(-300).expect("-05:00 is a valid offset")
}

#[test]
fn value_comparison_reads_a_floating_date_time_in_the_context_implicit_timezone() {
    let tree = Tree::parse_str("<r/>").expect("the document parses");
    let compiled =
        compile_31("xs:dateTime('2000-01-01T00:00:00') eq xs:dateTime('2000-01-01T05:00:00Z')");
    let context = compiled
        .context_with(tree.root())
        .with_implicit_timezone(minus_five());
    // Read in -05:00, the floating operand denotes the same instant as the zoned one.
    assert_eq!(
        compiled.evaluate_to_boolean_in(&context),
        Ok(true),
        "eq must use the context's implicit timezone, not hardcode UTC"
    );

    // Under the default (UTC) implicit timezone the same pair is *not* the same instant.
    let default_context = compiled.context_with(tree.root());
    assert_eq!(compiled.evaluate_to_boolean_in(&default_context), Ok(false));
}

#[test]
fn value_comparison_orders_a_floating_date_time_by_the_context_implicit_timezone() {
    let tree = Tree::parse_str("<r/>").expect("the document parses");
    // Read in -05:00 the floating operand is 05:00Z, five hours *after* 01:00Z.
    let compiled =
        compile_31("xs:dateTime('2000-01-01T00:00:00') gt xs:dateTime('2000-01-01T01:00:00Z')");
    let context = compiled
        .context_with(tree.root())
        .with_implicit_timezone(minus_five());
    assert_eq!(compiled.evaluate_to_boolean_in(&context), Ok(true));
}

#[test]
fn general_comparison_reads_a_floating_date_time_in_the_context_implicit_timezone() {
    let document = "<r><a>2000-01-01T00:00:00</a></r>";
    let tree = Tree::parse_str(document).expect("the document parses");
    let compiled = compile_31("xs:dateTime(/r/a) = xs:dateTime('2000-01-01T05:00:00Z')");
    let context = compiled
        .context_with(tree.root())
        .with_implicit_timezone(minus_five());
    assert_eq!(
        compiled.evaluate_to_boolean_in(&context),
        Ok(true),
        "general comparisons must thread the implicit timezone too, not only `eq`/`ne`"
    );
}

// --- evaluating a deep or long expression ------------------------------------------------------

/// Runs `body` on a thread with the stack an ordinary main thread has, *declared* to the engine.
///
/// The evaluation guards bound themselves against a declared figure — nothing in `std` reports
/// what is really left — so the helper declares exactly the stack it created. This is
/// deliberately *not* [`with_room`]: an oversized stack would hide the very overflow these cases
/// exist to pin.
fn on_a_default_stack<T: Send + 'static>(body: impl FnOnce() -> T + Send + 'static) -> T {
    const MAIN_THREAD_STACK: usize = 8 * 1024 * 1024;
    std::thread::Builder::new()
        .stack_size(MAIN_THREAD_STACK)
        .spawn(move || {
            let _limits = EvaluationLimits::declare(MAIN_THREAD_STACK);
            body()
        })
        .expect("the helper thread starts")
        .join()
        .expect("the helper thread finishes")
}

/// A recursive inline function used to recurse in Rust until the *process* aborted with a stack
/// overflow, which no host could catch. It now runs out of a bounded budget instead.
#[test]
fn runaway_recursion_raises_xpdy0130_instead_of_aborting_the_process() {
    let code = on_a_default_stack(|| {
        let expression = "let $f := function($self, $n) { \
             if ($n <= 0) then 0 else $n + $self($self, $n - 1) } \
             return $f($f, 1000000)";
        let compiled = compile(expression).expect("the expression compiles");
        let context = compiled.context();
        compiled
            .evaluate_value_in(&context)
            .err()
            .and_then(|error| error.code())
    });
    assert_eq!(code, Some(XPathErrorCode::Xpdy0130));
}

/// The budget is only worth having if it runs out *before* the stack does, so this runs the
/// runaway recursion on the smallest stack a host is likely to hand the engine — the 2 MiB Rust
/// gives a spawned thread, which is also what the engine assumes when nothing declares one — and
/// insists on getting an error back.
///
/// The declaration is deliberately omitted: this is the case for a host that never made one, and
/// [`oxixml_xpath::DEFAULT_STACK_SIZE`] is exactly the 2 MiB the thread really has.
///
/// There is no way to assert this politely: should the guard ever outgrow the stack it assumes,
/// this case does not fail, it aborts the test process — which is precisely the failure the
/// guard exists to prevent, and far better caught here than in a host.
#[test]
fn the_budget_runs_out_before_the_stack_it_is_calibrated_for_does() {
    let code = std::thread::Builder::new()
        .stack_size(oxixml_xpath::DEFAULT_STACK_SIZE)
        .spawn(|| {
            let expression = "let $f := function($self, $n) { \
                 if ($n <= 0) then 0 else $n + $self($self, $n - 1) } \
                 return $f($f, 1000000)";
            let compiled = compile(expression).expect("the expression compiles");
            let context = compiled.context();
            compiled
                .evaluate_value_in(&context)
                .err()
                .and_then(|error| error.code())
        })
        .expect("the helper thread starts")
        .join()
        .expect("the helper thread finishes");
    assert_eq!(code, Some(XPathErrorCode::Xpdy0130));
}

/// Recursion well short of the guards still works, so they are a guard and not a ban.
///
/// Fifty levels is deliberately not a token depth: it pins that a recursive function has room to
/// be useful on the stack an ordinary main thread has. It stops well short of both ceilings on
/// purpose, so that a case meant to demonstrate success cannot itself run the stack down.
#[test]
fn bounded_recursion_still_evaluates() {
    let value = on_a_default_stack(|| {
        let expression = "let $f := function($self, $n) { \
             if ($n <= 0) then 0 else $n + $self($self, $n - 1) } \
             return $f($f, 50)";
        let compiled = compile(expression).expect("the expression compiles");
        let context = compiled.context();
        compiled
            .evaluate_to_string_in(&context)
            .expect("fifty levels are within the budget")
    });
    assert_eq!(value, "1275");
}

/// A flat operator chain is parsed and freed iteratively; it is now *evaluated* iteratively too,
/// so a length the grammar does not bound no longer costs a stack frame per operator.
#[test]
fn a_hundred_thousand_operators_evaluate_without_recursing() {
    let value = on_a_default_stack(|| {
        let expression = std::iter::repeat_n("1", 100_000)
            .collect::<Vec<_>>()
            .join("+");
        let compiled = compile(&expression).expect("a flat operator chain compiles");
        let context = compiled.context();
        compiled
            .evaluate_to_string_in(&context)
            .expect("a flat chain is folded, not recursed")
    });
    assert_eq!(value, "100000");
}

/// The same for the other flat chain the parser builds with a loop: a long path.
#[test]
fn a_hundred_thousand_path_steps_evaluate_without_recursing() {
    let count = on_a_default_stack(|| {
        let expression = std::iter::repeat_n("*", 100_000)
            .collect::<Vec<_>>()
            .join("/");
        let compiled = compile(&expression).expect("a flat path chain compiles");
        let tree = Tree::parse_str("<r><a><b/></a></r>").expect("the document parses");
        let context = compiled.context_with(tree.root());
        compiled
            .evaluate_to_nodes_in(&context)
            .expect("a flat chain is folded, not recursed")
            .len()
    });
    // The document is three elements deep, so a hundred thousand steps select nothing.
    assert_eq!(count, 0);
}

/// The union, intersection and string-concatenation operators fold through the same path.
#[test]
fn other_flat_operator_chains_evaluate_too() {
    let (joined, united) = on_a_default_stack(|| {
        let concatenation = std::iter::repeat_n("'x'", 20_000)
            .collect::<Vec<_>>()
            .join("||");
        let union = std::iter::repeat_n("*", 20_000)
            .collect::<Vec<_>>()
            .join("|");
        let tree = Tree::parse_str("<r><a/><b/></r>").expect("the document parses");
        let concatenated = compile(&concatenation).expect("compiles");
        let context = concatenated.context_with(tree.root());
        let joined = concatenated
            .evaluate_to_string_in(&context)
            .expect("folded")
            .len();
        let compiled = compile(&union).expect("compiles");
        let element = tree.document_element().expect("a document element");
        let context = compiled.context_with(element);
        let united = compiled
            .evaluate_to_nodes_in(&context)
            .expect("folded")
            .len();
        (joined, united)
    });
    assert_eq!(joined, 20_000);
    assert_eq!(united, 2);
}

// --- fn:apply arity ----------------------------------------------------------------------------

/// F&O 3.1 §16.3.3: spreading an array of the wrong length over a function's parameters is
/// `FOAP0001`, which is a different condition — and a different `try`/`catch` target — from the
/// argument type error a mistyped argument raises.
#[test]
fn fn_apply_reports_an_arity_mismatch_as_foap0001() {
    let compiled = compile("apply(false#0, [0])").expect("the expression compiles");
    let context = compiled.context();
    assert_eq!(
        compiled
            .evaluate_value_in(&context)
            .err()
            .and_then(|error| error.code()),
        Some(XPathErrorCode::Foap0001)
    );

    // A matching arity still calls through.
    let matching = compile("apply(concat#2, ['a', 'b'])").expect("the expression compiles");
    let context = matching.context();
    assert_eq!(
        matching
            .evaluate_to_string_in(&context)
            .expect("the arity matches"),
        "ab"
    );
}

// --- typed function tests against a built-in ---------------------------------------------------

/// XPath 3.1 §2.5.5.5 judges a function item against `function(A₁, …, Aₙ) as R` by its *declared*
/// signature. Built-ins used to carry none, so only their arity was ever compared and `fn:abs#1`
/// satisfied `function(xs:string) as xs:date` as readily as its real signature.
#[test]
fn a_typed_function_test_reads_a_builtin_s_declared_signature() {
    let cases = [
        // `fn:name#1` is `function(node()?) as xs:string`.
        ("name#1 instance of function(node()?) as xs:string", "true"),
        ("name#1 instance of function(node()) as xs:string", "true"),
        (
            "name#1 instance of function(element(A)) as xs:anyAtomicType",
            "true",
        ),
        // The result is `xs:string`, which is not the narrower `xs:NCName`.
        (
            "name#1 instance of function(element(A)) as xs:NCName",
            "false",
        ),
        // A parameter must be *contravariant*: `node()*` admits more than `node()?` does.
        ("name#1 instance of function(node()*) as xs:string", "false"),
        ("name#1 instance of function(item()) as xs:string", "false"),
        // `fn:abs#1` is `function(xs:numeric?) as xs:numeric?`.
        (
            "abs#1 instance of function(xs:numeric?) as xs:numeric?",
            "true",
        ),
        (
            "abs#1 instance of function(xs:string) as xs:integer",
            "false",
        ),
        (
            "abs#1 instance of function(xs:double) as xs:double",
            "false",
        ),
        // The arity is still compared, and `function(*)` still matches anything callable.
        (
            "abs#1 instance of function(xs:integer, xs:integer) as xs:integer",
            "false",
        ),
        ("abs#1 instance of function(*)", "true"),
        // Every occurrence indicator has to survive the translation, in the result position as
        // well as the parameter one: `*` on the result of `fn:tokenize`, `?` on the result of
        // `fn:compare`, `?` on a node result, and a plain one-item result.
        (
            "tokenize#1 instance of function(xs:string?) as xs:string*",
            "true",
        ),
        (
            "compare#2 instance of function(xs:string?, xs:string?) as xs:integer?",
            "true",
        ),
        (
            "compare#2 instance of function(xs:string?, xs:string?) as xs:integer",
            "false",
        ),
        ("root#1 instance of function(node()?) as node()?", "true"),
        (
            "empty#1 instance of function(item()*) as xs:boolean",
            "true",
        ),
    ];
    for (expression, expected) in cases {
        let compiled = compile(expression).expect("the expression compiles");
        let context = compiled.context();
        let actual = compiled
            .evaluate_to_string_in(&context)
            .expect("the test evaluates");
        assert_eq!(actual, expected, "{expression}");
    }
}
