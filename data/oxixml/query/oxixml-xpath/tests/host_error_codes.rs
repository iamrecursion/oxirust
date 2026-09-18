//! The XPath error codes a *host language* depends on, exercised through the constructs the
//! host reaches them by.
//!
//! XSLT and XQuery both delegate expression evaluation to this crate, so an error an XSLT
//! stylesheet is required to report — `XPST0008` for a variable out of scope, `XPTY0020` for an
//! axis step whose focus is not a node, `XPDY0002` for an expression that needs a context item
//! and is given none — has to *originate* here, with the code the host will forward unchanged.
//! Each case below pins one of those codes to the smallest expression that produces it, so that
//! the contract a host compiles against does not move.

use oxixml_infoset::{AtomicValue, Tree, XdmNode};
use oxixml_xpath::{CompiledXPath, StaticContext, XPath, XPathError, XPathErrorCode, XPathVersion};

fn compile(expression: &str) -> Result<CompiledXPath, XPathError> {
    XPath::compile(expression, XPathVersion::XPath31, StaticContext::new())
}

/// Compiles the way a host that discovers its variables at run time does: the static check for
/// undeclared variables is deferred, and the values arrive on the dynamic context.
fn compile_with_late_variables(expression: &str) -> Result<CompiledXPath, XPathError> {
    XPath::compile(
        expression,
        XPathVersion::XPath31,
        StaticContext::new().with_undeclared_variables(),
    )
}

fn static_error(expression: &str) -> XPathError {
    compile(expression).expect_err("the expression is statically in error")
}

/// `XPST0008` — a reference to a variable that is not in the in-scope variables.
///
/// A stylesheet reaches it by naming a variable that is out of scope where the reference stands
/// (`xsl:on-completion` before the `xsl:variable` it names) or that no declaration binds at all.
/// Both readings have to answer with the same code: the check runs at compile time when the
/// host declares its variables up front, and at evaluation time when it binds them late.
#[test]
fn an_out_of_scope_variable_is_xpst0008() {
    let compile_time = static_error("$absent + 1");
    assert_eq!(compile_time.code(), Some(XPathErrorCode::Xpst0008));
    assert!(compile_time.is_static());
    assert!(compile_time.offset().is_some());

    let late = compile_with_late_variables("$absent + 1").expect("the check is deferred");
    let run_time = late
        .evaluate_without_context()
        .expect_err("nothing binds the variable");
    assert_eq!(run_time.code(), Some(XPathErrorCode::Xpst0008));

    // A variable the expression binds itself is in scope after its own `let`, and out of scope
    // before it — which is the shape of a forward reference in a stylesheet.
    assert!(compile("let $v := 1 return $v").is_ok());
    assert_eq!(
        static_error("let $v := $v return $v").code(),
        Some(XPathErrorCode::Xpst0008)
    );
}

/// `XPST0017` — the expanded name and arity of a function call match nothing in scope.
///
/// The arity matters as much as the name: a host that offers `fn:name#0` and `fn:name#1` must
/// still reject `fn:name#2`.
#[test]
fn an_unknown_function_is_xpst0017() {
    for expression in [
        "nonesuch(1)",
        "fn:nonesuch()",
        "name(1, 2)",
        "fn:string-length('a', 'b')",
    ] {
        assert_eq!(
            static_error(expression).code(),
            Some(XPathErrorCode::Xpst0017),
            "`{expression}` names no function"
        );
    }

    // A named function reference is resolved against the same table.
    assert_eq!(
        static_error("nonesuch#1").code(),
        Some(XPathErrorCode::Xpst0017)
    );
}

/// `XPTY0004` — a value that the function conversion rules cannot turn into the required type.
///
/// `fn:analyze-string` is the case a stylesheet meets through `xsl:analyze-string/@select`,
/// whose required type is `xs:string`: neither an `xs:integer` nor a sequence of several strings
/// is converted to one, so both are type errors rather than silently stringified.
#[test]
fn a_value_of_the_wrong_type_is_xpty0004() {
    for expression in [
        "analyze-string(22, 'e')",
        "analyze-string(('red', 'green', 'blue'), 'e')",
        "string-length(('a', 'b'))",
        "abs('x')",
    ] {
        let compiled = compile(expression).expect("the expression compiles");
        let failure = compiled
            .evaluate_without_context()
            .expect_err("the argument does not match the parameter's type");
        assert_eq!(
            failure.code(),
            Some(XPathErrorCode::Xpty0004),
            "`{expression}` is a type error"
        );
    }
}

/// `XPTY0019` — the left operand of `/` holds something that is not a node.
///
/// A template rule whose `xsl:context-item` declares an atomic type reaches it with the very
/// expression the pattern made look harmless: `./child` applied to an `xs:integer`.
#[test]
fn a_path_step_on_a_non_node_is_xpty0019() {
    for expression in ["./child", "(1, 2)/name()", "'text'/x"] {
        let compiled = compile(expression).expect("the expression compiles");
        let failure = compiled
            .evaluate(AtomicValue::integer(1_i64))
            .expect_err("a path step needs nodes on its left");
        assert_eq!(
            failure.code(),
            Some(XPathErrorCode::Xpty0019),
            "`{expression}` steps off a non-node"
        );
    }
}

/// `XPTY0020` — an axis step whose context item is not a node.
///
/// Inside `xsl:matching-substring` the context item is the matched substring, an `xs:string`, so
/// an axis step written there — `following-sibling::x` — has no node to start from. The
/// distinction from [`XPathErrorCode::Xpty0019`] is where the non-node sits: the *focus* of the
/// step rather than the operand of the `/` in front of it.
#[test]
fn an_axis_step_on_a_non_node_focus_is_xpty0020() {
    for expression in [
        "following-sibling::x",
        "child::x",
        "..",
        "attribute::x",
        "/",
    ] {
        let compiled = compile(expression).expect("the expression compiles");
        let failure = compiled
            .evaluate(AtomicValue::string("matched"))
            .expect_err("an axis step needs a node in focus");
        assert_eq!(
            failure.code(),
            Some(XPathErrorCode::Xpty0020),
            "`{expression}` has a non-node in focus"
        );
    }

    // With a node in focus the same expressions are ordinary steps.
    let tree = Tree::parse_str("<r><a/><b/></r>").expect("a well-formed document");
    let first = tree
        .document_element()
        .and_then(|root| root.children().first().cloned())
        .expect("the `a` element");
    let compiled = compile("following-sibling::b").expect("the expression compiles");
    assert_eq!(compiled.evaluate(first).expect("a sequence").len(), 1);
}

/// `XPDY0002` — the expression depends on the context item and none is defined.
///
/// A global `xsl:variable` with `select="."` is evaluated with the initial context item, and a
/// stylesheet invoked by template name — or whose initial selection turned out to be empty — has
/// none. An *absent* context item is not an empty one: the expression must fail rather than
/// answer the empty sequence.
#[test]
fn an_absent_context_item_is_xpdy0002() {
    for expression in [".", "root(.)", "name()", "string()", "position()", "last()"] {
        let compiled = compile(expression).expect("the expression compiles");
        let failure = compiled
            .evaluate_without_context()
            .expect_err("the expression needs a context item");
        assert_eq!(
            failure.code(),
            Some(XPathErrorCode::Xpdy0002),
            "`{expression}` depends on the context item"
        );
    }

    // The empty sequence is a value, not the absence of one: an expression given it in focus
    // answers, and only an expression given *nothing* raises.
    let compiled = compile("count(.)").expect("the expression compiles");
    assert_eq!(
        compiled
            .evaluate(AtomicValue::integer(7_i64))
            .expect("a sequence")
            .len(),
        1
    );
}

/// Every code above is reported with the classification a host language routes on: `XPST…` is
/// static, `XPTY…`/`XPDY…` are dynamic. A host that buckets a failure by phase rather than by
/// code — an `xsl:catch` may only see dynamic errors — needs the two to agree.
#[test]
fn the_codes_carry_the_phase_a_host_routes_on() {
    assert!(XPathErrorCode::Xpst0008.is_static());
    assert!(XPathErrorCode::Xpst0017.is_static());
    assert!(!XPathErrorCode::Xpty0004.is_static());
    assert!(!XPathErrorCode::Xpty0019.is_static());
    assert!(!XPathErrorCode::Xpty0020.is_static());
    assert!(!XPathErrorCode::Xpdy0002.is_static());

    // `XPST0008` is the one code of the set a host can meet in either phase, because the
    // static check is optional: `XPathError::is_static` follows the *code*, so a host that
    // defers the check must route on the code and not on the phase.
    let late = compile_with_late_variables("$absent").expect("the check is deferred");
    let deferred = late
        .evaluate_without_context()
        .expect_err("nothing binds the variable");
    assert_eq!(deferred.code(), Some(XPathErrorCode::Xpst0008));
    assert!(deferred.is_static());
}

// --- what a named function reference captures --------------------------------------------------

/// XPath 3.1 §3.1.6: "if the named function is context-dependent, the function item is a closure
/// over the static and dynamic context in which the named function reference is evaluated".
///
/// The closure has to be over the context at *reference* time rather than at call time, and it
/// has to reach a function the **host** registered as well as a built-in — an XSLT `current#0`
/// or `fn:key#2` bound in a global variable and applied inside a template must see what it was
/// referenced with. A registered function is handed only an `oxixml_fn::FnContext`, which carries
/// no focus, so the focus it may depend on arrives through [`oxixml_xpath::calling_focus`]; this
/// pins that what is published there is the *captured* context and not the call site's.
#[test]
fn a_named_function_reference_captures_the_context_it_is_written_in() {
    use oxixml_fn::{FnContext, builtins};
    use oxixml_infoset::{Item, Sequence};
    use oxixml_xpath::{DynamicContext, Focus, calling_focus};
    use std::sync::Arc;

    const PROBE: &str = "urn:probe";

    // The host's own function joins the library rather than replacing it, which is how a host
    // language builds its function set.
    let mut registry = builtins();
    // A host function whose meaning depends on the focus, exactly as XSLT's own do.
    registry.add(PROBE, "focus-position", 0, |_: &FnContext, _| {
        let position = calling_focus().map_or(0, |focus| focus.position());
        Ok(Sequence::singleton(Item::from(AtomicValue::integer(
            i64::try_from(position).unwrap_or_default(),
        ))))
    });
    let statik = Arc::new(StaticContext::new().with_functions(registry));

    // Both references are taken under the outer focus and applied under three others.
    let expression = XPath::compile(
        "let $builtin := fn:position#0, $host := Q{urn:probe}focus-position#0 \
         return string-join((10, 20, 30) ! ($builtin() || \"/\" || $host()), \"|\")",
        XPathVersion::XPath31,
        StaticContext::clone(&statik),
    )
    .expect("the expression compiles");

    let referenced_at = |position: usize| {
        DynamicContext::new(Arc::clone(&statik))
            .with_version(XPathVersion::XPath31)
            .with_focus(Focus::new(AtomicValue::integer(0_i64), position, 5))
    };
    // Without the capture the three applications would report their own positions, `1/1|2/2|3/3`.
    assert_eq!(
        expression
            .evaluate_to_string_in(&referenced_at(2))
            .expect("the expression evaluates"),
        "2/2|2/2|2/2"
    );
    assert_eq!(
        expression
            .evaluate_to_string_in(&referenced_at(4))
            .expect("the expression evaluates"),
        "4/4|4/4|4/4"
    );
}
