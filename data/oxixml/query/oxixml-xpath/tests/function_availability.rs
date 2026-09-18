//! What a host may ask about this engine's function library.
//!
//! [`oxixml_fn`]'s registry holds every function whose implementation is a mapping from sequences
//! to sequences. Three families are not: the context-dependent functions, the higher-order
//! functions, and the external-resource ones. A host that answers questions *about* the library —
//! XSLT's `fn:function-available`, whose answer decides whether a stylesheet takes a fallback
//! branch — has to consult both halves, and [`oxixml_xpath::supports_function`] is the second.
//!
//! The names checked here are the ones the W3C XSLT 3.0 suite probes in
//! `function-available-1017` (the `map:` functions) and `function-available-1018` (the
//! higher-order `fn:` functions); `fn:function-lookup` is checked over the same names, because a
//! name a host reports available must also be one a dynamic call can reach.

use oxixml_fn::builtins;
use oxixml_infoset::{ExpandedName, Tree};
use oxixml_xpath::{
    ARRAY_NAMESPACE, FN_NAMESPACE, MAP_NAMESPACE, StaticContext, XPath, XPathVersion,
    supports_function,
};

fn engine_serves(namespace: &str, local: &str, arity: usize) -> bool {
    supports_function(&ExpandedName::new(Some(namespace), local), arity)
}

/// Whether the *library* — either half — has this function at this arity, which is the question
/// `fn:function-available` asks.
fn available(namespace: &str, local: &str, arity: usize) -> bool {
    let name = ExpandedName::new(Some(namespace), local);
    builtins().contains(&name, arity) || supports_function(&name, arity)
}

/// The string value of `expression`, evaluated at XPath 3.1 with no context item.
fn value(expression: &str) -> String {
    let compiled = XPath::compile(expression, XPathVersion::XPath31, StaticContext::new())
        .expect("the expression compiles");
    let tree = Tree::parse_str("<e/>").expect("the document parses");
    compiled
        .evaluate_value_in(&compiled.context_with(tree.root()))
        .expect("the expression evaluates")
        .to_string()
}

#[test]
fn the_higher_order_functions_are_served_by_the_engine_rather_than_the_registry() {
    // `function-available-1018` asks these ten questions, and a processor that answers
    // `system-property('xsl:supports-higher-order-functions') = 'yes'` must answer all ten `true`.
    for (local, arity) in [
        ("function-name", 1),
        ("function-arity", 1),
        ("filter", 2),
        ("for-each", 2),
        ("function-lookup", 2),
        ("fold-left", 3),
        ("fold-right", 3),
        ("for-each-pair", 3),
        ("sort", 3),
        ("apply", 2),
    ] {
        assert!(
            engine_serves(FN_NAMESPACE, local, arity),
            "fn:{local}#{arity} is served by the evaluator"
        );
        assert!(
            !builtins().contains(&ExpandedName::new(Some(FN_NAMESPACE), local), arity),
            "fn:{local}#{arity} cannot be a registry entry"
        );
    }
}

#[test]
fn every_map_function_the_suite_probes_is_available() {
    // `function-available-1017` asks about these nine names — the eight the registry holds and
    // `map:for-each`, which takes a function item and therefore cannot be one of them.
    for (local, arity) in [
        ("merge", 1),
        ("size", 1),
        ("keys", 1),
        ("contains", 2),
        ("get", 2),
        ("put", 3),
        ("entry", 2),
        ("remove", 2),
        ("for-each", 2),
    ] {
        assert!(
            available(MAP_NAMESPACE, local, arity),
            "map:{local}#{arity} is in the function library"
        );
    }
    assert!(engine_serves(MAP_NAMESPACE, "for-each", 2));

    // And the names the same case expects to be *absent*, at the arity it writes.
    for (local, arity) in [
        ("for-each-entry", 2),
        ("new", 1),
        ("sandwich", 2),
        ("key-value-pairs", 1),
        ("create", 1),
        ("contains", 3),
        ("keys", 2),
        ("for-each", 5),
    ] {
        assert!(
            !available(MAP_NAMESPACE, local, arity),
            "map:{local}#{arity} is not in the function library"
        );
    }
}

#[test]
fn the_array_functions_that_take_a_function_item_are_served_by_the_engine() {
    for (local, arity) in [
        ("for-each", 2),
        ("filter", 2),
        ("fold-left", 3),
        ("fold-right", 3),
        ("for-each-pair", 3),
        ("sort", 1),
        ("sort", 2),
        ("sort", 3),
    ] {
        assert!(
            engine_serves(ARRAY_NAMESPACE, local, arity),
            "array:{local}#{arity} is served by the evaluator"
        );
    }
    assert!(!engine_serves(ARRAY_NAMESPACE, "sort", 4));
    assert!(!engine_serves(ARRAY_NAMESPACE, "flatten", 1));
    assert!(available(ARRAY_NAMESPACE, "flatten", 1), "a registry entry");
}

#[test]
fn an_ordinary_library_entry_is_not_served_by_the_engine() {
    // The predicate answers about the half the registry cannot hold, so a name the registry does
    // hold is `false` here and `true` through the registry — a host asks both.
    assert!(!engine_serves(FN_NAMESPACE, "abs", 1));
    assert!(available(FN_NAMESPACE, "abs", 1));
    assert!(!engine_serves(MAP_NAMESPACE, "size", 1));
    assert!(available(MAP_NAMESPACE, "size", 1));
}

#[test]
fn the_context_dependent_functions_are_served_by_the_engine() {
    for (local, arity) in [
        ("position", 0),
        ("last", 0),
        ("string", 0),
        ("name", 0),
        ("root", 0),
        ("lang", 1),
        ("id", 1),
        ("idref", 1),
    ] {
        assert!(
            engine_serves(FN_NAMESPACE, local, arity),
            "fn:{local}#{arity} reads the focus"
        );
    }
}

#[test]
fn function_lookup_reaches_both_halves_of_the_library() {
    // A higher-order function, looked up dynamically and applied.
    assert_eq!(
        value(
            "function-lookup(QName('http://www.w3.org/2005/xpath-functions/map', 'for-each'), 2)\
             (map { 'a': 1 }, function($k, $v) { $k })"
        ),
        "a"
    );
    // A registry entry reached the same way.
    assert_eq!(
        value(
            "function-lookup(QName('http://www.w3.org/2005/xpath-functions/map', 'size'), 1)\
             (map { 'a': 1, 'b': 2 })"
        ),
        "2"
    );
    // An `array:` function that takes a function item.
    assert_eq!(
        value(
            "array:flatten(function-lookup(\
             QName('http://www.w3.org/2005/xpath-functions/array', 'for-each'), 2)\
             ([1, 2, 3], function($m) { $m * 2 }))"
        ),
        "2 4 6"
    );
    // A name at an arity nothing serves has no item, which is `fn:function-lookup`'s way of
    // saying so (F&O 3.1 §16.3.4) rather than an error.
    assert_eq!(
        value(
            "count(function-lookup(\
             QName('http://www.w3.org/2005/xpath-functions/map', 'for-each'), 5))"
        ),
        "0"
    );
}

#[test]
fn map_for_each_concatenates_in_the_shapes_the_suite_writes() {
    assert_eq!(
        value("map:for-each(map { 'a': 1, 'b': 2 }, function($k, $v) { $v })"),
        "1 2"
    );
    // An action returning a sequence per entry contributes all of it.
    assert_eq!(
        value("map:for-each(map { 'a': 1, 'b': 2 }, function($k, $v) { ($k, $v) })"),
        "a 1 b 2"
    );
    // An action returning nothing contributes nothing, and an empty map has no entries.
    assert_eq!(
        value("count(map:for-each(map { 'a': 1 }, function($k, $v) { () }))"),
        "0"
    );
    assert_eq!(
        value("count(map:for-each(map {}, function($k, $v) { $k }))"),
        "0"
    );
    // The canonical use: a map turned back into a map, entry by entry.
    assert_eq!(
        value(
            "map:size(map:merge(map:for-each(map { 'a': 1, 'b': 2 }, \
             function($k, $v) { map:entry($k, $v + 1) })))"
        ),
        "2"
    );
}
