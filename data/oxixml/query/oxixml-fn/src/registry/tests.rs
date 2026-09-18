//! Unit tests for [`builtins`] — one function per registered name and arity, called
//! through the registry exactly as the evaluator would, not the bare Rust function.

use super::*;
use oxixml_infoset::{Tree, XdmNode};

fn call(local: &str, arguments: &[Sequence]) -> FnResult<Sequence> {
    let registry = builtins();
    let context = FnContext::new();
    let name = ExpandedName::new(Some(FN_NAMESPACE), local);
    let function = registry
        .lookup(&name, arguments.len())
        .ok_or_else(|| type_error(&format!("{local}#{} is not registered", arguments.len())))?;
    function.call(&context, arguments)
}

fn rendered(sequence: &Sequence) -> String {
    sequence
        .iter()
        .map(ToString::to_string)
        .collect::<Vec<_>>()
        .join(",")
}

#[test]
fn the_catalogue_is_populated() {
    let registry = builtins();
    assert!(!registry.is_empty());
    assert!(registry.len() > 100);
    assert!(registry.contains(&ExpandedName::new(Some(FN_NAMESPACE), "abs"), 1));
    assert!(!registry.contains(&ExpandedName::new(Some(FN_NAMESPACE), "abs"), 2));
    assert!(registry.names().count() == registry.len());
}

/// The functions [XSLT 3.0 §18](https://www.w3.org/TR/xslt-30/#add-func) requires that belong to
/// *this* library rather than to a transformation.
///
/// The rest of the list — `fn:current`, `fn:key`, `fn:document`, the grouping and merging
/// accessors, the availability tests — can only be answered by a transformation, so they are
/// registered by the XSLT engine on top of this catalogue.
#[test]
fn the_function_and_operator_half_of_the_xslt_library_is_registered() {
    let registry = builtins();
    let map_namespace = crate::maps::MAP_NAMESPACE;
    let required: &[(&str, &str, usize)] = &[
        (FN_NAMESPACE, "copy-of", 1),
        (FN_NAMESPACE, "snapshot", 1),
        (FN_NAMESPACE, "collation-key", 1),
        (FN_NAMESPACE, "collation-key", 2),
        (FN_NAMESPACE, "deep-equal", 2),
        (FN_NAMESPACE, "deep-equal", 3),
        (FN_NAMESPACE, "json-to-xml", 1),
        (FN_NAMESPACE, "json-to-xml", 2),
        (FN_NAMESPACE, "xml-to-json", 1),
        (FN_NAMESPACE, "xml-to-json", 2),
        (map_namespace, "merge", 1),
        (map_namespace, "size", 1),
        (map_namespace, "keys", 1),
        (map_namespace, "contains", 2),
        (map_namespace, "get", 2),
        (map_namespace, "put", 3),
        (map_namespace, "entry", 2),
        (map_namespace, "remove", 2),
    ];
    let missing: Vec<String> = required
        .iter()
        .filter(|(namespace, local, arity)| {
            !registry.contains(&ExpandedName::new(Some(*namespace), *local), *arity)
        })
        .map(|(namespace, local, arity)| format!("Q{{{namespace}}}{local}#{arity}"))
        .collect();
    assert!(missing.is_empty(), "not registered: {missing:?}");
}

/// No higher-order function is in the catalogue, and none can be.
///
/// A registry entry is a closure over `&[Sequence]`, and a function item inside one of those
/// sequences is an [`oxixml_infoset::XdmFunction`] — a handle whose body only the evaluator that
/// built it can recover. So `fn:for-each`, `map:for-each` and their siblings are implemented here
/// as plain Rust functions over [`crate::FunctionItem`] and dispatched by the evaluator, which is
/// the only party that can turn an argument into something callable.
#[test]
fn the_catalogue_holds_no_higher_order_function() {
    let registry = builtins();
    for (namespace, local, arity) in [
        (FN_NAMESPACE, "for-each", 2),
        (FN_NAMESPACE, "filter", 2),
        (FN_NAMESPACE, "fold-left", 3),
        (FN_NAMESPACE, "apply", 2),
        (crate::maps::MAP_NAMESPACE, "for-each", 2),
        (crate::arrays::ARRAY_NAMESPACE, "for-each", 2),
    ] {
        assert!(
            !registry.contains(&ExpandedName::new(Some(namespace), local), arity),
            "{local}#{arity} cannot be called through the registry"
        );
    }
}

#[test]
fn lookup_returns_a_callable_function_item() -> FnResult<()> {
    let registry = builtins();
    let context = FnContext::new();
    let abs = registry
        .lookup(&ExpandedName::new(Some(FN_NAMESPACE), "abs"), 1)
        .ok_or_else(|| type_error("fn:abs#1"))?;
    assert_eq!(abs.arity(), 1);
    let result = abs.call(&context, &[Sequence::from(AtomicValue::integer(-3_i64))])?;
    assert_eq!(rendered(&result), "3");
    Ok(())
}

#[test]
fn string_functions_through_the_registry() -> FnResult<()> {
    assert_eq!(
        rendered(&call(
            "concat",
            &[
                Sequence::from(AtomicValue::string("a")),
                Sequence::from(AtomicValue::string("b")),
            ]
        )?),
        "ab"
    );
    assert_eq!(
        rendered(&call(
            "upper-case",
            &[Sequence::from(AtomicValue::string("abc"))]
        )?),
        "ABC"
    );
    assert_eq!(
        rendered(&call(
            "substring",
            &[
                Sequence::from(AtomicValue::string("metadata")),
                Sequence::from(AtomicValue::integer(4_i64)),
                Sequence::from(AtomicValue::integer(3_i64)),
            ]
        )?),
        "ada"
    );
    assert_eq!(
        rendered(&call(
            "tokenize",
            &[Sequence::from(AtomicValue::string(" a  b "))]
        )?),
        "a,b"
    );
    Ok(())
}

#[test]
fn sequence_functions_through_the_registry() -> FnResult<()> {
    let items = Sequence::from_iter([1_i64, 2, 3].map(AtomicValue::integer));
    assert_eq!(rendered(&call("count", std::slice::from_ref(&items))?), "3");
    assert_eq!(
        rendered(&call("reverse", std::slice::from_ref(&items))?),
        "3,2,1"
    );
    assert_eq!(rendered(&call("sum", std::slice::from_ref(&items))?), "6");
    assert_eq!(rendered(&call("avg", std::slice::from_ref(&items))?), "2");
    assert_eq!(rendered(&call("max", &[items])?), "3");
    Ok(())
}

#[test]
fn node_functions_through_the_registry() -> Result<(), Box<dyn std::error::Error>> {
    let tree = Tree::parse_str("<a><b/></a>")?;
    let root = Sequence::from(tree.document_element().ok_or("no root")?);
    assert_eq!(
        rendered(&call("local-name", std::slice::from_ref(&root))?),
        "a"
    );
    assert_eq!(
        rendered(&call("has-children", std::slice::from_ref(&root))?),
        "true"
    );
    assert_eq!(rendered(&call("serialize", &[root])?), "<a><b/></a>");
    Ok(())
}

#[test]
fn map_and_array_functions_through_the_registry() -> FnResult<()> {
    let registry = builtins();
    let context = FnContext::new();
    let map = XdmMap::new().put(
        AtomicValue::string("k"),
        Sequence::from(AtomicValue::integer(1_i64)),
    );
    let size = registry
        .lookup(&ExpandedName::new(Some(maps::MAP_NAMESPACE), "size"), 1)
        .ok_or_else(|| type_error("map:size#1"))?;
    assert_eq!(
        rendered(&size.call(&context, &[Sequence::singleton(Item::from(map))])?),
        "1"
    );

    let array = XdmArray::from_iter([AtomicValue::integer(1_i64), AtomicValue::integer(2_i64)]);
    let get = registry
        .lookup(&ExpandedName::new(Some(arrays::ARRAY_NAMESPACE), "get"), 2)
        .ok_or_else(|| type_error("array:get#2"))?;
    assert_eq!(
        rendered(&get.call(
            &context,
            &[
                Sequence::singleton(Item::from(array)),
                Sequence::from(AtomicValue::integer(2_i64)),
            ]
        )?),
        "2"
    );
    Ok(())
}

#[test]
fn fn_error_always_raises() {
    let raised = call("error", &[]).expect_err("fn:error raises");
    assert!(raised.is(FnErrorCode::Foer0000));
}

#[test]
fn parse_xml_and_serialize_round_trip() -> FnResult<()> {
    let parsed = call(
        "parse-xml",
        &[Sequence::from(AtomicValue::string("<a b='1'/>"))],
    )?;
    assert_eq!(rendered(&call("serialize", &[parsed])?), r#"<a b="1"/>"#);
    Ok(())
}

#[test]
fn copy_of_maps_a_mixed_sequence_leaving_atomic_values_untouched()
-> Result<(), Box<dyn std::error::Error>> {
    // copy-of-010 (nodes) / copy-of-011 (an atomic value), through the actual registered
    // one-argument fn:copy-of, not just the bare `nodes::copy_of` helper.
    let tree = Tree::parse_str("<r><a x='1'/><a x='2'/></r>")?;
    let root = tree.document_element().ok_or("no root")?;
    let originals = root.children();
    let mixed = Sequence::from_items(
        originals
            .iter()
            .cloned()
            .map(Item::Node)
            .chain([Item::from(AtomicValue::integer(17_i64))]),
    );

    let copied = call("copy-of", std::slice::from_ref(&mixed))?;
    assert_eq!(copied.len(), 3);
    // The trailing atomic value passes straight through (`rendered` prints each item's
    // string value, empty for a childless, text-less element — the two node items).
    assert_eq!(rendered(&copied), ",,17");

    for (index, original) in originals.iter().enumerate() {
        let Some(Item::Node(copy)) = copied.iter().nth(index).cloned() else {
            return Err(format!("item {index} of the copy is not a node").into());
        };
        assert_ne!(&copy, original, "copy-of gives every node a fresh identity");
        assert_eq!(
            copy.attributes()[0].string_value(),
            original.attributes()[0].string_value()
        );
    }
    Ok(())
}

#[test]
fn copy_of_zero_length_sequence_and_arity_are_registered() {
    let registry = builtins();
    for local in ["copy-of", "snapshot"] {
        assert!(
            registry.contains(&ExpandedName::new(Some(FN_NAMESPACE), local), 1),
            "fn:{local}#1 should be registered"
        );
    }
    // Neither has a zero-argument entry *here*: that form needs the dynamic context, and is
    // registered by whichever host (this crate's only one is `oxixml-xslt`) supplies it.
    for local in ["copy-of", "snapshot"] {
        assert!(
            !registry.contains(&ExpandedName::new(Some(FN_NAMESPACE), local), 0),
            "fn:{local}#0 should not be registered in the context-free library"
        );
    }
}

#[test]
fn snapshot_through_the_registry_keeps_the_ancestor_path() -> Result<(), Box<dyn std::error::Error>>
{
    let tree = Tree::parse_str("<a x='1'><b/><c/></a>")?;
    let root = tree.document_element().ok_or("no root")?;
    let c = root.children()[1].clone();

    let copied = call("snapshot", &[Sequence::from(c.clone())])?;
    let Some(Item::Node(copy)) = copied.first() else {
        return Err("expected a single node".into());
    };
    assert_ne!(copy, &c);
    let parent = copy.parent().ok_or("the ancestor was kept")?;
    assert_eq!(parent.attributes()[0].string_value(), "1");
    assert_eq!(parent.children().len(), 1, "b is excluded");
    Ok(())
}

/// `fn:load-xquery-module` and `fn:transform` must be *found* — `fn:function-lookup` answers a
/// function item for each — and must fail only when they are called, with the error F&O 3.1
/// reserves for a processor that cannot serve the request.
#[test]
fn the_external_processor_functions_are_found_and_fail_only_when_called() {
    let registry = builtins();
    let context = FnContext::new();
    for (local, arity, code) in [
        ("load-xquery-module", 1, FnErrorCode::Foqm0006),
        ("load-xquery-module", 2, FnErrorCode::Foqm0006),
        ("transform", 1, FnErrorCode::Foxt0004),
    ] {
        let name = ExpandedName::new(Some(FN_NAMESPACE), local);
        assert!(
            registry.contains(&name, arity),
            "fn:{local}#{arity} should be registered"
        );
        let Some(function) = registry.lookup(&name, arity) else {
            panic!("fn:{local}#{arity} should be looked up");
        };
        let arguments = vec![Sequence::empty(); arity];
        let error = function
            .call(&context, &arguments)
            .expect_err("an unavailable processor");
        assert!(error.is(code), "fn:{local}#{arity} should raise {code}");
    }
}

/// An argument of the wrong type does not displace the unavailability error: the arguments of
/// these two are not put through the function conversion rules on the way in.
#[test]
fn an_unavailable_processor_reports_itself_whatever_the_argument() {
    let registry = builtins();
    let context = FnContext::new();
    let name = ExpandedName::new(Some(FN_NAMESPACE), "load-xquery-module");
    let Some(function) = registry.lookup(&name, 1) else {
        panic!("fn:load-xquery-module#1 should be registered");
    };
    let error = function
        .call(&context, &[Sequence::from(AtomicValue::integer(0_i64))])
        .expect_err("an unavailable processor");
    assert!(error.is(FnErrorCode::Foqm0006));
}

/// Calls a registered function against a context of the caller's choosing.
fn call_with(context: &FnContext, local: &str, arguments: &[Sequence]) -> FnResult<Sequence> {
    let registry = builtins();
    let name = ExpandedName::new(Some(FN_NAMESPACE), local);
    let function = registry
        .lookup(&name, arguments.len())
        .ok_or_else(|| type_error(&format!("{local}#{} is not registered", arguments.len())))?;
    function.call(context, arguments)
}

/// The option-map form of `cdata-section-elements` reaches the serializer, exactly as the
/// parameter-document form already did (QT3 `fn-serialize/serialize-xml-110`).
#[test]
fn serialize_honours_cdata_section_elements_from_the_option_map()
-> Result<(), Box<dyn std::error::Error>> {
    let tree = Tree::parse_str(
        r#"<a xmlns:p="http://www.example.org/ns/p"><b>bold</b><i>italic</i><p:b>BOLD</p:b><p:i>ITALIC</p:i></a>"#,
    )?;
    let root = Sequence::from(tree.document_element().ok_or("no document element")?);
    let qname = |namespace: Option<&str>, local: &str| {
        AtomicValue::qname(QNameValue::new(None::<&str>, namespace, local))
    };
    let parameters = XdmMap::new()
        .put(
            AtomicValue::string("method"),
            Sequence::from(AtomicValue::string("xml")),
        )
        .put(
            AtomicValue::string("cdata-section-elements"),
            Sequence::from_iter([
                qname(None, "b"),
                qname(Some("http://www.example.org/ns/p"), "b"),
            ]),
        )
        .put(
            AtomicValue::string("suppress-indentation"),
            Sequence::from(qname(None, "para")),
        );
    let serialized = rendered(&call(
        "serialize",
        &[root.clone(), Sequence::singleton(Item::from(parameters))],
    )?);
    assert!(serialized.contains("CDATA[bold]"), "{serialized}");
    assert!(serialized.contains("CDATA[BOLD]"), "{serialized}");
    assert!(!serialized.contains("CDATA[italic]"), "{serialized}");
    assert!(!serialized.contains("CDATA[ITALIC]"), "{serialized}");
    // Without the parameter nothing is wrapped at all.
    assert!(!rendered(&call("serialize", std::slice::from_ref(&root))?).contains("CDATA"));

    // Both name-set parameters are typed `xs:QName*`, so a string is a type error rather than a
    // silently ignored entry (QT3 `serialize-xml-109`).
    for parameter in ["cdata-section-elements", "suppress-indentation"] {
        let ill_typed = XdmMap::new().put(
            AtomicValue::string(parameter),
            Sequence::from(AtomicValue::string("b")),
        );
        let error = call(
            "serialize",
            &[root.clone(), Sequence::singleton(Item::from(ill_typed))],
        )
        .expect_err("a string is not an xs:QName");
        assert!(error.is(FnErrorCode::Xpty0004), "{parameter}");
    }
    Ok(())
}

/// `fn:json-to-xml` defaults to `duplicates=retain`, but an explicit `use-first` must be
/// honoured and keep the *first* entry with a repeated key
/// (QT3 `app-Walmsley/d1e78807d`).
#[test]
fn json_to_xml_honours_an_explicit_duplicates_use_first() -> Result<(), Box<dyn std::error::Error>>
{
    let json = r#"{"name": "first", "name": "second"}"#;
    let options = XdmMap::new().put(
        AtomicValue::string("duplicates"),
        Sequence::from(AtomicValue::string("use-first")),
    );
    let result = call(
        "json-to-xml",
        &[
            Sequence::from(AtomicValue::string(json)),
            Sequence::singleton(Item::from(options)),
        ],
    )?;
    assert_eq!(
        rendered(&call("serialize", &[result])?),
        concat!(
            r#"<map xmlns="http://www.w3.org/2005/xpath-functions">"#,
            r#"<string key="name">first</string></map>"#
        )
    );

    // The default keeps both, which is what `retain` means.
    let retained = call("json-to-xml", &[Sequence::from(AtomicValue::string(json))])?;
    assert_eq!(
        rendered(&call("serialize", &[retained])?),
        concat!(
            r#"<map xmlns="http://www.w3.org/2005/xpath-functions">"#,
            r#"<string key="name">first</string><string key="name">second</string></map>"#
        )
    );
    Ok(())
}

/// F&O 3.1 §14.7.1: the document `fn:parse-xml` returns carries the static base URI of the
/// call as its base URI (QT3 `fn-parse-xml/parse-xml-007`).
#[test]
fn parse_xml_gives_its_result_the_static_base_uri() -> Result<(), Box<dyn std::error::Error>> {
    let context = FnContext::new().with_static_base_uri("http://www.example.com/base/");
    let parsed = call_with(
        &context,
        "parse-xml",
        &[Sequence::from(AtomicValue::string("<a>foo</a>"))],
    )?;
    let [Item::Node(document)] = parsed.as_slice() else {
        return Err("fn:parse-xml returns one document node".into());
    };
    assert_eq!(
        document.base_uri().as_deref(),
        Some("http://www.example.com/base/")
    );

    // With no static base URI there is nothing to report, and nothing is invented.
    let anonymous = call("parse-xml", &[Sequence::from(AtomicValue::string("<a/>"))])?;
    let [Item::Node(document)] = anonymous.as_slice() else {
        return Err("fn:parse-xml returns one document node".into());
    };
    assert_eq!(document.base_uri(), None);
    Ok(())
}

/// F&O 3.1 §14.7.1: whatever base URI the result is given, its `document-uri` is absent —
/// `fn:parse-xml` builds a document, it does not retrieve one (QT3
/// `fn-parse-xml/parse-xml-011`), and the same holds for `fn:parse-xml-fragment`.
#[test]
fn parse_xml_leaves_the_document_uri_of_its_result_absent() -> Result<(), Box<dyn std::error::Error>>
{
    let context = FnContext::new().with_static_base_uri("http://www.example.com/base/");
    for function in ["parse-xml", "parse-xml-fragment"] {
        let parsed = call_with(
            &context,
            function,
            &[Sequence::from(AtomicValue::string("<a>foo</a>"))],
        )?;
        let [Item::Node(document)] = parsed.as_slice() else {
            return Err(format!("fn:{function} returns one document node").into());
        };
        let reported = call_with(&context, "document-uri", std::slice::from_ref(&parsed))?;
        assert!(
            reported.is_empty(),
            "fn:document-uri(fn:{function}(…)) is the empty sequence, not `{}`",
            rendered(&reported)
        );
        assert_eq!(document.document_uri(), None, "for fn:{function}");
        assert_eq!(
            document.base_uri().as_deref(),
            Some("http://www.example.com/base/"),
            "the base URI survives for fn:{function}"
        );
    }
    Ok(())
}
