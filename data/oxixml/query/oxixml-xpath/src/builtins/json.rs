//! The option handling of the JSON functions —
//! [F&O 3.1 §17.5](https://www.w3.org/TR/xpath-functions-31/#json-functions) — for the one
//! option only an evaluator can honour: `fallback`.
//!
//! `fallback` is a *function item*, and [`oxixml_fn`] can see one but not call one: a general
//! function item reaches the data model as [`oxixml_infoset::XdmFunction`], an opaque handle
//! whose body only its own evaluator can downcast. So `fn:parse-json`, `fn:json-to-xml` and
//! `fn:json-doc` read their option map here, build the
//! [`Fallback`](oxixml_fn::json::parser::Fallback) closure over this engine's
//! [`XpathFunction`](crate::XpathFunction), and hand the finished
//! [`JsonOptions`] to the library.

use crate::context::DynamicContext;
use crate::error::{XPathError, XPathResult};
use crate::value::{XpathItem, XpathValue};
use oxixml_fn::json::parser::Duplicates;
use oxixml_fn::json::{JsonFunction, JsonOptions};
use oxixml_fn::{FnError, FnResult};
use oxixml_infoset::{AtomicValue, XdmMap};
use std::cell::RefCell;

/// Whether this module handles the function, in which case it is dispatched here rather than to
/// the library registry.
pub(crate) fn supports(local: &str, arity: usize) -> bool {
    // `fn:json-to-xml#1` has no option map to read, but its result is a document node whose
    // base URI is the static base URI (F&O 3.1 §17.5.5), and only an evaluator knows that — so
    // both arities are served here.
    matches!((local, arity), ("parse-json", 2) | ("json-to-xml", 1 | 2))
}

/// Invokes `fn:parse-json#2` or `fn:json-to-xml#2`.
///
/// # Errors
///
/// The `FOJS` codes the library raises, plus `XPTY0004` for an ill-typed option map or
/// `fallback` function, and whatever the `fallback` function itself raises.
pub(crate) fn call(
    context: &DynamicContext,
    local: &str,
    arguments: &[XpathValue],
) -> XPathResult<XpathValue> {
    let Some(text) = optional_string(&arguments[0])? else {
        // Even with nothing to parse, an ill-typed option map is still an error.
        with_options(context, arguments.get(1), function_of(local), |_| Ok(()))?;
        return Ok(XpathValue::empty());
    };
    match local {
        "parse-json" => with_options(
            context,
            arguments.get(1),
            JsonFunction::ParseJson,
            |options| {
                Ok(XpathValue::from(oxixml_fn::json::parse_json(
                    &text, options,
                )?))
            },
        ),
        _ => with_options(
            context,
            arguments.get(1),
            JsonFunction::JsonToXml,
            |options| {
                Ok(XpathValue::from(oxixml_fn::json::json_to_xml_with_base(
                    &text,
                    options,
                    context.static_context().base_uri(),
                )?))
            },
        ),
    }
}

fn function_of(local: &str) -> JsonFunction {
    match local {
        "json-to-xml" => JsonFunction::JsonToXml,
        _ => JsonFunction::ParseJson,
    }
}

/// Reads the option map of a JSON function and runs `body` with the options it describes.
///
/// The continuation shape is what lets the `fallback` closure be a *borrow*: it lives on this
/// stack frame, and [`JsonOptions`] only ever holds a reference to it.
///
/// # Errors
///
/// `XPTY0004` when the options argument is not a map or the `fallback` option is not a single
/// arity-one function item, `FOJS0005` for an option value the function does not accept, and
/// whatever `body` — or the `fallback` function — raises.
pub(crate) fn with_options<T>(
    context: &DynamicContext,
    options_argument: Option<&XpathValue>,
    function: JsonFunction,
    body: impl FnOnce(&JsonOptions<'_>) -> XPathResult<T>,
) -> XPathResult<T> {
    let defaults = JsonOptions {
        duplicates: match function {
            JsonFunction::JsonToXml => Duplicates::Retain,
            JsonFunction::ParseJson => Duplicates::UseFirst,
        },
        function,
        ..JsonOptions::default()
    };
    let Some(map) = options_map(options_argument)? else {
        return body(&defaults);
    };
    let options = JsonOptions::from_map(&map, defaults)?;
    let Some(fallback) = fallback_function(&map)? else {
        return body(&options);
    };

    // The library's `Fallback` is a plain `Fn(&str) -> FnResult<String>`, so an engine error the
    // function raises has to travel out of band to keep its own code and error object.
    let raised: RefCell<Option<XPathError>> = RefCell::new(None);
    let invoke = |escape: &str| -> FnResult<String> {
        let argument = XpathValue::from(AtomicValue::string(escape));
        let outcome = fallback
            .call(context, &[argument])
            .and_then(|value| Ok(oxixml_fn::json::fallback_result(&value.into_sequence()?)?));
        outcome.map_err(|error| {
            let carried = FnError::with_name(error.name().clone(), error.to_string());
            *raised.borrow_mut() = Some(error);
            carried
        })
    };
    let outcome = body(&JsonOptions {
        fallback: Some(&invoke),
        ..options
    });
    match raised.into_inner() {
        Some(error) => Err(error),
        None => outcome,
    }
}

/// The options argument as a map, or `None` when it was absent or the empty sequence.
fn options_map(argument: Option<&XpathValue>) -> XPathResult<Option<XdmMap>> {
    let Some(argument) = argument else {
        return Ok(None);
    };
    match argument.first() {
        None => Ok(None),
        Some(item) => match item.as_map() {
            Some(map) => Ok(Some(map.clone())),
            None => Err(XPathError::type_error("the options argument must be a map")),
        },
    }
}

/// The `fallback` option's function item.
///
/// The type check itself is the library's — [`oxixml_fn::json::fallback_is_supplied`] — so that
/// a caller which never reaches this module still rejects an ill-typed option.
fn fallback_function(map: &XdmMap) -> XPathResult<Option<crate::function::XpathFunction>> {
    if !oxixml_fn::json::fallback_is_supplied(map)? {
        return Ok(None);
    }
    for (key, value) in map.entries() {
        if key.string_value() != "fallback" {
            continue;
        }
        let callable = XpathValue::from(value.clone())
            .first()
            .and_then(XpathItem::as_callable);
        return callable.map(Some).ok_or_else(|| {
            XPathError::type_error("the `fallback` option must be a single function of arity one")
        });
    }
    Ok(None)
}

/// A `$json-text` argument, which every JSON function declares `xs:string?`.
fn optional_string(value: &XpathValue) -> XPathResult<Option<String>> {
    Ok(value
        .atomize_at_most_one("the JSON text")?
        .map(|atomic| atomic.string_value()))
}
