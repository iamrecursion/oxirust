//! Regression tests for the audit findings against `oxixml-xquery`.
//!
//! * `order by … collation "…UCA…"` raised `XPTY0004` on ordinary strings, because the sort
//!   keys were compared through the infoset layer, which knows only the two built-in
//!   collations;
//! * `group by … collation` silently swallowed the same `FOCH0002` into "not equal", so keys
//!   the collation calls equal landed in different groups;
//! * a relative `order by … collation` URI must still resolve against the static base URI, and
//!   cyclic module imports must keep compiling (XQuery 3.0 erratum E8);
//! * the recursive-descent parser had no nesting-depth guard, so deeply nested parentheses
//!   aborted the process with a stack overflow;
//! * loading a chain of imported library modules recursed on the native stack and rescanned the
//!   modules already loaded once per import, so a few hundred of them aborted the process;
//! * a runaway recursive function outran the thread's stack well below the nesting limit.

use oxixml_xquery::{
    DynamicQueryContext, InMemoryModules, StaticQueryContext, XQuery, XQueryErrorCode, XQueryResult,
};
use oxixml_xsd::TimezoneOffset;
use std::sync::Arc;

const UCA_PRIMARY: &str = "http://www.w3.org/2013/collation/UCA?lang=en;strength=primary";

fn run(query: &str) -> XQueryResult<String> {
    XQuery::compile_default(query)?.execute_to_string(&DynamicQueryContext::new())
}

fn code_of(query: &str) -> Option<XQueryErrorCode> {
    run(query).err().and_then(|error| error.code())
}

fn run_in(query: &str, context: &DynamicQueryContext) -> XQueryResult<String> {
    XQuery::compile_default(query)?.execute_to_string(context)
}

fn minus_five() -> TimezoneOffset {
    TimezoneOffset::new(-300).expect("-05:00 is a valid offset")
}

/// Runs `body` on a thread with room for the deepest expression the parser still accepts.
fn with_room<T: Send + 'static>(body: impl FnOnce() -> T + Send + 'static) -> T {
    std::thread::Builder::new()
        .stack_size(64 * 1024 * 1024)
        .spawn(body)
        .expect("the helper thread starts")
        .join()
        .expect("the helper thread finishes")
}

// --- order by under a collation ------------------------------------------------------------

#[test]
fn order_by_honours_an_explicit_uca_collation() -> XQueryResult<()> {
    assert_eq!(
        run(&format!(
            "for $x in ('B','a') order by $x collation '{UCA_PRIMARY}' return $x"
        ))?,
        "a B"
    );
    Ok(())
}

#[test]
fn order_by_honours_the_prolog_default_collation() -> XQueryResult<()> {
    assert_eq!(
        run(&format!(
            "declare default collation '{UCA_PRIMARY}'; for $x in ('B','a') order by $x return $x"
        ))?,
        "a B"
    );
    Ok(())
}

#[test]
fn order_by_still_orders_by_code_point_without_a_collation() -> XQueryResult<()> {
    assert_eq!(run("for $x in ('B','a') order by $x return $x")?, "B a");
    Ok(())
}

#[test]
fn order_by_still_reports_incomparable_types() {
    assert_eq!(
        code_of("for $x in (1, 'a') order by $x return $x"),
        Some(XQueryErrorCode::Xpty0004)
    );
}

#[test]
fn an_unknown_order_by_collation_is_reported() {
    assert!(code_of("for $x in ('a') order by $x collation 'urn:nope' return $x").is_some());
}

// --- group by under a collation ------------------------------------------------------------

#[test]
fn group_by_honours_an_explicit_uca_collation() -> XQueryResult<()> {
    assert_eq!(
        run(&format!(
            "count(for $x in ('a','A') group by $k := $x collation '{UCA_PRIMARY}' return $k)"
        ))?,
        "1"
    );
    Ok(())
}

#[test]
fn group_by_honours_the_prolog_default_collation() -> XQueryResult<()> {
    assert_eq!(
        run(&format!(
            "declare default collation '{UCA_PRIMARY}'; \
             count(for $x in ('a','A') group by $k := $x return $k)"
        ))?,
        "1"
    );
    Ok(())
}

#[test]
fn group_by_still_separates_by_code_point_without_a_collation() -> XQueryResult<()> {
    assert_eq!(
        run("count(for $x in ('a','A') group by $k := $x return $k)")?,
        "2"
    );
    Ok(())
}

// --- module import cycles --------------------------------------------------------------------

fn compile_with_modules(query: &str, modules: &[(&str, &str)]) -> Option<XQueryErrorCode> {
    let mut library = InMemoryModules::new();
    for (namespace, source) in modules {
        library = library.with_module(*namespace, *source);
    }
    let context = StaticQueryContext::new().with_module_resolver(Arc::new(library));
    XQuery::compile(query, context).err().and_then(|e| e.code())
}

/// XQuery 3.0 erratum E8 removed the prohibition on cyclic module imports, and the W3C suite
/// (`modules-circular`, `errata8-002a`, `errata8-003`) requires them to compile and run, so
/// `XQST0093` must *not* be raised for a self-import or a cycle.
#[test]
fn a_module_that_imports_itself_still_compiles() {
    let module = "module namespace a = 'urn:example:a'; \
                  import module namespace self = 'urn:example:a'; \
                  declare function a:one() { 1 };";
    assert_eq!(
        compile_with_modules(
            "import module namespace a = 'urn:example:a'; a:one()",
            &[("urn:example:a", module)],
        ),
        None
    );
}

#[test]
fn an_indirect_module_import_cycle_still_compiles() {
    let a = "module namespace a = 'urn:example:a2'; \
             import module namespace b = 'urn:example:b2'; \
             declare function a:one() { 1 };";
    let b = "module namespace b = 'urn:example:b2'; \
             import module namespace a = 'urn:example:a2'; \
             declare function b:two() { 2 };";
    assert_eq!(
        compile_with_modules(
            "import module namespace a = 'urn:example:a2'; a:one()",
            &[("urn:example:a2", a), ("urn:example:b2", b)],
        ),
        None
    );
}

#[test]
fn a_diamond_module_import_is_still_legal() {
    let d = "module namespace d = 'urn:example:d3'; declare function d:four() { 4 };";
    let b = "module namespace b = 'urn:example:b3'; \
             import module namespace d = 'urn:example:d3'; \
             declare function b:two() { 2 };";
    let c = "module namespace c = 'urn:example:c3'; \
             import module namespace d = 'urn:example:d3'; \
             declare function c:three() { 3 };";
    assert_eq!(
        compile_with_modules(
            "import module namespace b = 'urn:example:b3'; \
             import module namespace c = 'urn:example:c3'; \
             (b:two(), c:three())",
            &[
                ("urn:example:b3", b),
                ("urn:example:c3", c),
                ("urn:example:d3", d),
            ],
        ),
        None,
        "a diamond import is not a cycle"
    );
}

// --- parser nesting depth ----------------------------------------------------------------------

#[test]
fn a_deeply_nested_query_is_a_static_error_not_a_crash() {
    for depth in [200_usize, 1_000, 10_000] {
        let query = format!("{}1{}", "(".repeat(depth), ")".repeat(depth));
        let outcome = with_room(move || XQuery::compile_default(&query).is_err());
        assert!(outcome, "the nesting limit is enforced at depth {depth}");
    }
}

#[test]
fn deep_but_legal_nesting_still_compiles() {
    let query = format!("{}1{}", "(".repeat(60), ")".repeat(60));
    assert!(with_room(move || XQuery::compile_default(&query).is_ok()));
}

// --- implicit timezone in `order by` and `group by` ---------------------------------------------
//
// F&O 3.1 §10.4.4 reads a date/time value with no timezone in the *dynamic context's* implicit
// timezone. `order by` and `group by` keys go through the same atomic comparison the `eq`/`lt`
// operators do, so a context whose implicit timezone is not UTC must change both the order and
// the grouping of otherwise-identical-looking dateTime keys — proving the context is actually
// threaded through rather than a hardcoded UTC default silently standing in for it.

#[test]
fn order_by_reads_a_floating_date_time_key_in_the_context_implicit_timezone() -> XQueryResult<()> {
    let query = "for $x in (xs:dateTime('2000-01-01T00:00:00'), xs:dateTime('2000-01-01T01:00:00Z')) \
                 order by $x return xs:string($x)";
    // Under the default (UTC) implicit timezone the floating value reads as 00:00:00Z, before
    // 01:00:00Z.
    assert_eq!(run(query)?, "2000-01-01T00:00:00 2000-01-01T01:00:00Z");
    // Under -05:00 the same floating value reads as 05:00:00Z, five hours *after* 01:00:00Z, so
    // the order reverses.
    let context = DynamicQueryContext::new().with_implicit_timezone(minus_five());
    assert_eq!(
        run_in(query, &context)?,
        "2000-01-01T01:00:00Z 2000-01-01T00:00:00",
        "`order by` must use the context's implicit timezone, not hardcode UTC"
    );
    Ok(())
}

#[test]
fn group_by_reads_a_floating_date_time_key_in_the_context_implicit_timezone() -> XQueryResult<()> {
    let query = "count(for $x in (xs:dateTime('2000-01-01T00:00:00'), xs:dateTime('2000-01-01T05:00:00Z')) \
                 group by $k := $x return $k)";
    // Under the default (UTC) implicit timezone the two keys denote different instants.
    assert_eq!(run(query)?, "2");
    // Under -05:00 the floating key denotes the same instant as the zoned one, so they group
    // together.
    let context = DynamicQueryContext::new().with_implicit_timezone(minus_five());
    assert_eq!(
        run_in(query, &context)?,
        "1",
        "`group by` must use the context's implicit timezone, not hardcode UTC"
    );
    Ok(())
}

// --- long import chains ----------------------------------------------------------------------

/// Runs `body` on a thread with the stack `std::thread::spawn` gives by default, which is what a
/// thread pool or an async executor's worker hands a library.
fn on_an_ordinary_thread<T: Send + 'static>(body: impl FnOnce() -> T + Send + 'static) -> T {
    std::thread::Builder::new()
        .stack_size(oxixml_xquery::DEFAULT_STACK_SIZE)
        .spawn(body)
        .expect("the helper thread starts")
        .join()
        .expect("the helper thread finishes")
}

/// A library built out of modules that each import the next is an ordinary shape, and nothing
/// bounds how long the chain may be — so loading one must cost neither a native stack frame per
/// module nor a rescan of everything loaded so far. Two thousand of them load on an ordinary
/// thread's stack, and quickly enough that the cost is plainly not quadratic.
#[test]
fn a_long_import_chain_loads_without_recursing() {
    let (result, elapsed) = on_an_ordinary_thread(|| {
        const LENGTH: usize = 2_000;
        let mut library = InMemoryModules::new();
        for index in 0..LENGTH {
            let namespace = format!("urn:chain:{index}");
            let import = if index + 1 == LENGTH {
                String::new()
            } else {
                format!("import module namespace next = 'urn:chain:{}';", index + 1)
            };
            let source = format!(
                "module namespace m{index} = '{namespace}'; {import} \
                 declare function m{index}:value() {{ {index} }};"
            );
            library = library.with_module(namespace, source);
        }
        let context = StaticQueryContext::new().with_module_resolver(Arc::new(library));
        let started = std::time::Instant::now();
        let compiled = XQuery::compile(
            "import module namespace m = 'urn:chain:0'; m:value()",
            context,
        )
        .map(|compiled| compiled.execute_to_string(&DynamicQueryContext::new()));
        (compiled, started.elapsed())
    });
    let text = result
        .expect("a two-thousand-module chain compiles")
        .expect("and runs");
    assert_eq!(text, "0");
    assert!(
        elapsed < std::time::Duration::from_secs(30),
        "loading the chain took {elapsed:?}"
    );
}

// --- the native stack under recursive function calls -------------------------------------------

/// A runaway recursive function must report `XPDY0130` rather than run the thread out of stack:
/// exhausting it aborts the process, which neither the caller nor a `try`/`catch` can intercept.
#[test]
fn recursive_function_calls_stop_before_the_stack_does() {
    let code = on_an_ordinary_thread(|| {
        let query = "declare function local:f($n as xs:integer) as xs:integer { \
                       if ($n le 0) then 0 else local:f($n - 1) + 1 \
                     }; local:f(100000)";
        XQuery::compile_default(query)
            .expect("the query compiles")
            .execute_to_string(&DynamicQueryContext::new())
            .err()
            .and_then(|error| error.code())
    });
    assert_eq!(code, Some(XQueryErrorCode::Xpdy0130));
}

/// The same query on a thread with room for it, declared, runs to completion — the ceiling is a
/// property of the thread, and the caller is the only one who knows what it is. The depth stays
/// modest because each call also nests several levels of *expression*, which
/// [`oxixml_xpath`] guards separately and which reports `XPDY0130` of its own.
#[test]
fn a_declared_stack_carries_the_recursion_the_thread_can_hold() {
    let depth = 100;
    let text = std::thread::Builder::new()
        .stack_size(64 * 1024 * 1024)
        .spawn(move || {
            let query = format!(
                "declare function local:f($n as xs:integer) as xs:integer {{ \
                   if ($n le 0) then 0 else local:f($n - 1) + 1 \
                 }}; local:f({depth})"
            );
            let context = DynamicQueryContext::new().with_stack_size(64 * 1024 * 1024);
            XQuery::compile_default(&query)
                .expect("the query compiles")
                .execute_to_string(&context)
        })
        .expect("the helper thread starts")
        .join()
        .expect("the helper thread finishes")
        .expect("the declared stack carries the recursion");
    assert_eq!(text, depth.to_string());
}

/// A declared function is reached through [`oxixml_fn`]'s registry, so the depth it may recurse
/// to is that library's budget rather than this engine's — and a stack declared on the query's
/// dynamic context has to raise both. QT3's `numberformat121` and `numberformat122` recurse five
/// thousand calls deep through `local:timesTenToThe`, which no ceiling admits by default.
#[test]
fn a_declared_stack_carries_a_recursive_declared_function() {
    const PROLOG: &str = "declare function local:times-ten-to-the($n as xs:decimal, \
                          $exp as xs:integer) as xs:decimal { \
                            if ($exp eq 0) then $n \
                            else if ($exp gt 0) \
                              then ($n * 10) => local:times-ten-to-the($exp - 1) \
                              else ($n div 10) => local:times-ten-to-the($exp + 1) \
                          }; ";
    // `FOAR0002` is a numeric-overflow code of the function library rather than of XQuery, so it
    // is matched by name; `XPDY0130` is this crate's own.
    let run = |stack: usize, exponent: i32, picture: &str| -> Result<String, String> {
        let query = format!(
            "{PROLOG} fn:format-number(1 => local:times-ten-to-the({exponent}), '{picture}')"
        );
        std::thread::Builder::new()
            .stack_size(stack)
            .spawn(move || {
                let context = DynamicQueryContext::new().with_stack_size(stack);
                XQuery::compile_default(&query)
                    .expect("the query compiles")
                    .execute_to_string(&context)
                    .map_err(|error| error.name().local_name().to_owned())
            })
            .expect("the helper thread starts")
            .join()
            .expect("the helper thread finishes")
    };
    // `numberformat121`: five thousand calls, then an exponent no double can hold. The catalogue
    // admits either the formatted value or `FOAR0002`; what it never admits is stopping short.
    assert_eq!(
        run(512 * 1024 * 1024, 5_000, "9.9999e9999"),
        Err("FOAR0002".to_owned())
    );
    // `numberformat122`: the same recursion downwards, which underflows to zero.
    assert_eq!(
        run(512 * 1024 * 1024, -5_000, "0.0000e0001"),
        Ok("0.0000e0000".to_owned())
    );
    // Undeclared, the default budget still bounds the same recursion tightly.
    assert_eq!(
        run(oxixml_xquery::DEFAULT_STACK_SIZE, 5_000, "9.9999e9999"),
        Err("XPDY0130".to_owned())
    );
}
