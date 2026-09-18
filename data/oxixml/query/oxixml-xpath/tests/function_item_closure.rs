//! What a *host* function sees when it is reached through a function item.
//!
//! A function registered through [`oxixml_fn::FunctionRegistry`] receives no focus of its own, so
//! a host-language function whose meaning depends on the context item reads
//! [`oxixml_xpath::calling_focus`] instead — XSLT 3.0's `fn:key`, `fn:id` and the unparsed-entity
//! accessors all identify the document to search that way.
//!
//! [XPath 3.1 §3.1.6](https://www.w3.org/TR/xpath-31/#id-named-function-ref) says what such a
//! function must see when a *named function reference* is applied somewhere else: "if the named
//! function is context-dependent, the function item is a closure over the static and dynamic
//! context in which the named function reference is evaluated". XSLT 3.0 §5.4.4 repeats it for
//! the host — named function references "are defined to retain the XPath static and dynamic
//! context at the point of invocation as part of the closure of the resulting function item, and
//! to use this preserved context when a dynamic function call is subsequently made" — and exempts
//! only the *XSLT* extensions to that context, which [`oxixml_xpath::calling_invocation`] reports
//! separately.
//!
//! So the focus a host function reads through a function item is the one that was in force where
//! the item was *made*, not the one in force where it is *applied*. These tests pin both halves,
//! for a named function reference, for `fn:function-lookup`, for a partial application, and
//! across the round trip through the data model that storing the item in a variable makes.

use oxixml_fn::{FN_NAMESPACE, FunctionRegistry, builtins};
use oxixml_infoset::{AtomicValue, Sequence, Tree, XdmNode};
use oxixml_xpath::{
    DynamicContext, InvocationKind, StaticContext, XPath, XPathError, XPathVersion,
    calling_invocation,
};
use std::sync::Arc;

/// The document every test works over: `<doc><a/><b/></doc>`.
fn document() -> Tree {
    Tree::parse_str("<doc><a/><b/></doc>").expect("the document parses")
}

/// The name of the node in focus where the registered function now running was called — what an
/// XSLT function that identifies a document from the context node reads.
fn focus_name() -> String {
    match oxixml_xpath::calling_focus() {
        Some(focus) => match focus.item() {
            oxixml_xpath::XpathItem::Value(oxixml_infoset::Item::Node(node)) => {
                node.node_name().map_or_else(
                    || "(unnamed)".to_owned(),
                    |name| name.local_name().to_owned(),
                )
            }
            other => format!("({})", other.string_value().unwrap_or_default()),
        },
        None => "(no focus)".to_owned(),
    }
}

/// The function library, plus three host functions of the shape XSLT registers: two that read the
/// focus of the call — one of them with an argument, so that a partial application has something
/// to fix — and one that reports how it was reached.
fn registry() -> FunctionRegistry {
    let mut registry: FunctionRegistry = builtins();
    registry.add(FN_NAMESPACE, "focus-name", 0, |_context, _arguments| {
        Ok(Sequence::from(AtomicValue::string(focus_name())))
    });
    registry.add(FN_NAMESPACE, "focus-suffix", 1, |_context, arguments| {
        let mut name = focus_name();
        if let Some(argument) = arguments[0].first() {
            name.push_str(&argument.string_value().unwrap_or_default());
        }
        Ok(Sequence::from(AtomicValue::string(name)))
    });
    registry.add(FN_NAMESPACE, "how-called", 0, |_context, _arguments| {
        let how = match calling_invocation() {
            Some(InvocationKind::Dynamic) => "dynamic",
            Some(InvocationKind::Static) => "static",
            None => "outside a call",
        };
        Ok(Sequence::from(AtomicValue::string(how)))
    });
    registry
}

/// Evaluates `capture` under the focus `capture_focus`, binds its value to `$f`, then evaluates
/// `apply` in the same context and returns its string value.
///
/// `apply` is written so that the focus at the dynamic call is *not* the one `capture` was
/// evaluated under — `'x' ! $f()` rebinds the context item to an atomic value, which is exactly
/// the shape of the XSLT suite's `unparsed-entity-09`: a function item taken while a node is in
/// focus and applied inside `<xsl:for-each select="'hatch-pic'">`.
fn capture_then_apply(
    capture: &str,
    apply: &str,
    capture_focus: &str,
) -> Result<String, XPathError> {
    let static_context = Arc::new(StaticContext::new().with_functions(registry()));
    let compiling = StaticContext::clone(&static_context).with_variable("f");
    let tree = document();
    let focus = XPath::compile(
        capture_focus,
        XPathVersion::XPath31,
        StaticContext::clone(&static_context),
    )?
    .evaluate_value_in(
        &DynamicContext::new(Arc::clone(&static_context)).with_context_item(tree.root()),
    )?
    .first()
    .expect("the focus expression selects a node")
    .clone();
    let capturing = DynamicContext::new(Arc::clone(&static_context)).with_context_item(focus);
    let item = XPath::compile(
        capture,
        XPathVersion::XPath31,
        StaticContext::clone(&static_context),
    )?
    .evaluate_value_in(&capturing)?;
    let applying = capturing.with_variable("f", item);
    XPath::compile(apply, XPathVersion::XPath31, compiling)?.evaluate_to_string_in(&applying)
}

#[test]
fn a_named_function_reference_keeps_the_focus_it_was_made_under() -> Result<(), XPathError> {
    // The item is made while `<a/>` is in focus and applied while the context item is the string
    // `'x'`: the host function must still see `a`.
    assert_eq!(
        capture_then_apply("fn:focus-name#0", "'x' ! $f()", "/doc/a")?,
        "a"
    );
    // The same item applied under a *different node* is not swayed by it either.
    assert_eq!(
        capture_then_apply("fn:focus-name#0", "/doc/b ! $f()", "/doc/a")?,
        "a"
    );
    Ok(())
}

#[test]
fn a_static_call_reads_the_focus_at_the_call_site() -> Result<(), XPathError> {
    // The converse, so the rule above cannot degrade into "a host function never sees a path
    // step": a call that names the function is answered by the focus the step rebound.
    assert_eq!(
        capture_then_apply("fn:focus-name#0", "/doc/b ! fn:focus-name()", "/doc/a")?,
        "b"
    );
    assert_eq!(
        capture_then_apply("fn:focus-name#0", "fn:focus-name()", "/doc/a")?,
        "a"
    );
    Ok(())
}

#[test]
fn function_lookup_keeps_the_focus_of_the_lookup() -> Result<(), XPathError> {
    // XSLT 3.0 §5.4.4 names `fn:function-lookup` beside the named function reference, and it is
    // built through the same closure.
    assert_eq!(
        capture_then_apply(
            "fn:function-lookup(fn:QName('http://www.w3.org/2005/xpath-functions', 'focus-name'), 0)",
            "'x' ! $f()",
            "/doc/a"
        )?,
        "a"
    );
    Ok(())
}

#[test]
fn a_partial_application_keeps_the_focus_it_was_made_under() -> Result<(), XPathError> {
    // A partial application is a function item like any other, and XPath 3.1 §3.1.5.1 gives it
    // the same closure: "the dynamic context of the partial application" is what the underlying
    // function is called with.
    assert_eq!(
        capture_then_apply("fn:focus-suffix(?)", "'x' ! $f('!')", "/doc/a")?,
        "a!"
    );
    Ok(())
}

#[test]
fn the_closure_survives_the_round_trip_through_the_data_model() -> Result<(), XPathError> {
    // Storing the item in a map takes it through [`oxixml_infoset::Item::Function`] and back,
    // which is the route an XSLT variable takes it by.
    assert_eq!(
        capture_then_apply("map{'g': fn:focus-name#0}", "'x' ! $f('g')()", "/doc/a")?,
        "a"
    );
    Ok(())
}

#[test]
fn how_the_function_was_reached_is_reported_beside_the_focus() -> Result<(), XPathError> {
    // The focus is preserved, but the *invocation kind* is not: a host function that must reject
    // a dynamic call outright — XSLT's `fn:current-group`, `XTDE1061` — still recognizes one.
    assert_eq!(
        capture_then_apply("fn:how-called#0", "'x' ! $f()", "/doc/a")?,
        "dynamic"
    );
    assert_eq!(
        capture_then_apply("fn:how-called#0", "fn:how-called()", "/doc/a")?,
        "static"
    );
    Ok(())
}
