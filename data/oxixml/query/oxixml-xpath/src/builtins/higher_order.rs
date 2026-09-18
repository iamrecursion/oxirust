//! The higher-order functions of [F&O 3.1 §16](https://www.w3.org/TR/xpath-functions-31/#higher-order-functions)
//! and the `map:`/`array:` functions that take a function item.
//!
//! These cannot live in [`oxixml_fn`]: its
//! [`FunctionItem`](oxixml_fn::FunctionItem) maps sequences to sequences, and a *general*
//! function item — an inline function that closes over variables, or a partial application — is
//! a value only an evaluator can produce. Everything here therefore works on
//! [`XpathFunction`](crate::XpathFunction) and calls back into the evaluator's dynamic context.

use super::{integer, optional, require_function};
use crate::context::{DynamicContext, MAP_NAMESPACE};
use crate::error::{XPathError, XPathErrorCode, XPathResult};
use crate::eval::compare_atomic_values;
use crate::value::{XpathItem, XpathValue};
use oxixml_infoset::{AtomicType, AtomicValue, ExpandedName, Item, XdmArray, XdmMap};
use std::cmp::Ordering;

/// Invokes one of the higher-order `fn:` functions.
///
/// # Errors
///
/// [`XPathErrorCode::Xpty0004`](crate::XPathErrorCode::Xpty0004) for a wrongly typed argument,
/// plus whatever the supplied function raises.
pub(crate) fn call_fn(
    context: &DynamicContext,
    local: &str,
    arguments: &[XpathValue],
) -> XPathResult<XpathValue> {
    match local {
        "for-each" => {
            let action = require_function(&arguments[1], "the $action of fn:for-each")?;
            let mut out = XpathValue::empty();
            for item in arguments[0].iter() {
                out.append(action.call(context, &[XpathValue::singleton(item.clone())])?);
            }
            Ok(out)
        }
        "filter" => {
            let predicate = require_function(&arguments[1], "the $predicate of fn:filter")?;
            let mut out = XpathValue::empty();
            for item in arguments[0].iter() {
                let verdict = predicate.call(context, &[XpathValue::singleton(item.clone())])?;
                if require_boolean(&verdict, "the result of a fn:filter predicate")? {
                    out.push(item.clone());
                }
            }
            Ok(out)
        }
        "fold-left" => {
            let action = require_function(&arguments[2], "the $action of fn:fold-left")?;
            let mut accumulator = arguments[1].clone();
            for item in arguments[0].iter() {
                accumulator =
                    action.call(context, &[accumulator, XpathValue::singleton(item.clone())])?;
            }
            Ok(accumulator)
        }
        "fold-right" => {
            let action = require_function(&arguments[2], "the $action of fn:fold-right")?;
            let mut accumulator = arguments[1].clone();
            for item in arguments[0].iter().rev() {
                accumulator =
                    action.call(context, &[XpathValue::singleton(item.clone()), accumulator])?;
            }
            Ok(accumulator)
        }
        "for-each-pair" => {
            let action = require_function(&arguments[2], "the $action of fn:for-each-pair")?;
            let mut out = XpathValue::empty();
            for (left, right) in arguments[0].iter().zip(arguments[1].iter()) {
                out.append(action.call(
                    context,
                    &[
                        XpathValue::singleton(left.clone()),
                        XpathValue::singleton(right.clone()),
                    ],
                )?);
            }
            Ok(out)
        }
        "sort" => {
            let sorted = sort_items(context, arguments, arguments[0].as_slice())?;
            Ok(XpathValue::from_items(sorted))
        }
        "apply" => {
            let function = require_function(&arguments[0], "the $function of fn:apply")?;
            let array = require_array(&arguments[1], "the $array of fn:apply")?;
            let supplied: Vec<XpathValue> = array
                .members()
                .map(|member| XpathValue::from(member.clone()))
                .collect();
            // F&O 3.1 §16.3.3 gives the arity mismatch its own code: `fn:apply` spreads the
            // array over the parameters, so an array of the wrong length is a fault in the
            // *call* — `FOAP0001` — not the ordinary argument type error `XPTY0004` that
            // [`XpathFunction::call`](crate::XpathFunction::call) would otherwise report.
            if supplied.len() != function.arity() {
                return Err(XPathError::new(
                    XPathErrorCode::Foap0001,
                    format!(
                        "fn:apply was given an array of {} member(s) for {}",
                        supplied.len(),
                        function.describe()
                    ),
                ));
            }
            function.call(context, &supplied)
        }
        "function-lookup" => {
            let name = require_qname(&arguments[0], "the $name of fn:function-lookup")?;
            let arity = require_integer(&arguments[1], "the $arity of fn:function-lookup")?;
            let Ok(arity) = usize::try_from(arity) else {
                return Ok(XpathValue::empty());
            };
            match super::function_item(context, &name, arity) {
                Ok(function) => Ok(XpathValue::singleton(function)),
                Err(_) => Ok(XpathValue::empty()),
            }
        }
        "function-name" => {
            let function = require_function(&arguments[0], "the $function of fn:function-name")?;
            Ok(optional(function.name().map(|name| {
                AtomicValue::qname(oxixml_infoset::QNameValue::new(
                    None::<String>,
                    name.namespace(),
                    name.local_name(),
                ))
            })))
        }
        "function-arity" => {
            let function = require_function(&arguments[0], "the $function of fn:function-arity")?;
            Ok(integer(i64::try_from(function.arity()).unwrap_or(i64::MAX)))
        }
        other => Err(XPathError::type_error(format!(
            "`fn:{other}` is not implemented"
        ))),
    }
}

/// Invokes one of the `map:` or `array:` functions that take a function item.
///
/// # Errors
///
/// [`XPathErrorCode::Xpty0004`](crate::XPathErrorCode::Xpty0004) for a wrongly typed argument,
/// plus whatever the supplied function raises.
pub(crate) fn call_collection(
    context: &DynamicContext,
    name: &ExpandedName,
    arguments: &[XpathValue],
) -> XPathResult<XpathValue> {
    let local = name.local_name();
    if name.namespace() == Some(MAP_NAMESPACE) {
        let map = require_map(&arguments[0], "the $map of map:for-each")?;
        let action = require_function(&arguments[1], "the $action of map:for-each")?;
        let mut out = XpathValue::empty();
        for (key, value) in map.entries() {
            out.append(action.call(
                context,
                &[
                    XpathValue::from(key.clone()),
                    XpathValue::from(value.clone()),
                ],
            )?);
        }
        return Ok(out);
    }
    let array = require_array(&arguments[0], "the $array argument")?;
    match local {
        "for-each" => {
            let action = require_function(&arguments[1], "the $action of array:for-each")?;
            let mut members = Vec::with_capacity(array.len());
            for member in array.members() {
                let result = action.call(context, &[XpathValue::from(member.clone())])?;
                members.push(result.into_sequence()?);
            }
            Ok(XpathValue::from(Item::Array(XdmArray::from_members(
                members,
            ))))
        }
        "filter" => {
            let predicate = require_function(&arguments[1], "the $predicate of array:filter")?;
            let mut members = Vec::new();
            for member in array.members() {
                let verdict = predicate.call(context, &[XpathValue::from(member.clone())])?;
                if require_boolean(&verdict, "the result of an array:filter predicate")? {
                    members.push(member.clone());
                }
            }
            Ok(XpathValue::from(Item::Array(XdmArray::from_members(
                members,
            ))))
        }
        "fold-left" => {
            let action = require_function(&arguments[2], "the $action of array:fold-left")?;
            let mut accumulator = arguments[1].clone();
            for member in array.members() {
                accumulator =
                    action.call(context, &[accumulator, XpathValue::from(member.clone())])?;
            }
            Ok(accumulator)
        }
        "fold-right" => {
            let action = require_function(&arguments[2], "the $action of array:fold-right")?;
            let mut accumulator = arguments[1].clone();
            for member in array.members().rev() {
                accumulator =
                    action.call(context, &[XpathValue::from(member.clone()), accumulator])?;
            }
            Ok(accumulator)
        }
        "for-each-pair" => {
            let other = require_array(&arguments[1], "the second $array of array:for-each-pair")?;
            let action = require_function(&arguments[2], "the $action of array:for-each-pair")?;
            let mut members = Vec::new();
            for (left, right) in array.members().zip(other.members()) {
                let result = action.call(
                    context,
                    &[
                        XpathValue::from(left.clone()),
                        XpathValue::from(right.clone()),
                    ],
                )?;
                members.push(result.into_sequence()?);
            }
            Ok(XpathValue::from(Item::Array(XdmArray::from_members(
                members,
            ))))
        }
        "sort" => {
            let items: Vec<XpathItem> = array
                .members()
                .map(|member| {
                    XpathItem::Value(Item::Array(XdmArray::from_members([member.clone()])))
                })
                .collect();
            // Sort the *members*, each of which is a sequence; wrap each in a one-member array so
            // that the shared sort machinery sees one item per member.
            let sorted = sort_items(context, arguments, &items)?;
            let mut members = Vec::with_capacity(sorted.len());
            for item in sorted {
                let wrapper = item
                    .as_array()
                    .and_then(|array| array.get_at(0).cloned())
                    .unwrap_or_default();
                members.push(wrapper);
            }
            Ok(XpathValue::from(Item::Array(XdmArray::from_members(
                members,
            ))))
        }
        other => Err(XPathError::type_error(format!(
            "`array:{other}` is not implemented"
        ))),
    }
}

/// The shared implementation of `fn:sort` and `array:sort`: a stable sort on the atomized value
/// of each item's sort key, with `xs:untypedAtomic` compared as `xs:string`.
fn sort_items(
    context: &DynamicContext,
    arguments: &[XpathValue],
    items: &[XpathItem],
) -> XPathResult<Vec<XpathItem>> {
    let collation = match arguments.get(1) {
        Some(argument) if !argument.is_empty() => Some(super::optional_string(argument)?),
        _ => None,
    };
    let key = match arguments.get(2) {
        Some(argument) => Some(require_function(&arguments[2], "the $key of sort")?)
            .filter(|_| !argument.is_empty()),
        None => None,
    };

    let mut keyed: Vec<(Vec<AtomicValue>, XpathItem)> = Vec::with_capacity(items.len());
    for item in items {
        let source = match &key {
            Some(key) => key.call(context, &[unwrap_sort_input(item)])?,
            None => XpathValue::singleton(item.clone()),
        };
        keyed.push((sort_key(&source)?, item.clone()));
    }

    let mut failure: Option<XPathError> = None;
    keyed.sort_by(|left, right| {
        if failure.is_some() {
            return Ordering::Equal;
        }
        match compare_keys(context, &left.0, &right.0, collation.as_deref()) {
            Ok(ordering) => ordering,
            Err(error) => {
                failure = Some(error);
                Ordering::Equal
            }
        }
    });
    if let Some(error) = failure {
        return Err(error);
    }
    Ok(keyed.into_iter().map(|(_, item)| item).collect())
}

/// `array:sort` wraps each member in a one-member array so that it can share the machinery of
/// `fn:sort`; the key function must still see the member itself.
fn unwrap_sort_input(item: &XpathItem) -> XpathValue {
    match item {
        XpathItem::Value(Item::Array(array)) if array.len() == 1 => array
            .get_at(0)
            .cloned()
            .map(XpathValue::from)
            .unwrap_or_default(),
        other => XpathValue::singleton(other.clone()),
    }
}

fn sort_key(value: &XpathValue) -> XPathResult<Vec<AtomicValue>> {
    let mut atomized = value.atomize()?;
    for entry in &mut atomized {
        if entry.atomic_type() == AtomicType::UntypedAtomic {
            *entry = entry.clone().cast_label(AtomicType::String)?;
        }
    }
    Ok(atomized)
}

fn compare_keys(
    context: &DynamicContext,
    left: &[AtomicValue],
    right: &[AtomicValue],
    collation: Option<&str>,
) -> XPathResult<Ordering> {
    for (left, right) in left.iter().zip(right.iter()) {
        match compare_atomic_values(context, left, right, collation)? {
            Some(Ordering::Equal) | None => {}
            Some(other) => return Ok(other),
        }
    }
    Ok(left.len().cmp(&right.len()))
}

/// The result of a `function(item()) as xs:boolean` predicate, read through the *function
/// conversion rules*.
///
/// `fn:filter` and `array:filter` declare `$f` as `function(item()) as xs:boolean`, so the
/// function supplied to them is *coerced* to that type
/// ([XPath 3.1 §3.1.5.2](https://www.w3.org/TR/xpath-31/#id-function-coercion)) and the
/// conversion rules apply to what it returns: the value is atomized, and an `xs:untypedAtomic` —
/// which is what atomizing an untyped element node yields — is cast to `xs:boolean`. That is
/// what makes `filter((1 to 20), function($i){<a>{1 - ($i mod 2)}</a>})` select the even
/// numbers (`filter-006`) rather than raise a type error.
///
/// There is no promotion *to* `xs:boolean`, so a returned `xs:integer` stays `XPTY0004`.
fn require_boolean(value: &XpathValue, what: &str) -> XPathResult<bool> {
    let atomized = value.atomize()?;
    let [atomic] = atomized.as_slice() else {
        return Err(XPathError::type_error(format!(
            "{what} must be exactly one xs:boolean"
        )));
    };
    if let Some(verdict) = atomic.as_bool() {
        return Ok(verdict);
    }
    if atomic.atomic_type() == AtomicType::UntypedAtomic {
        // A cast, so a lexical form that is not a boolean is `FORG0001` rather than a type error.
        let cast = AtomicValue::parse(AtomicType::Boolean, &atomic.string_value())?;
        return cast
            .as_bool()
            .ok_or_else(|| XPathError::type_error(format!("{what} must be an xs:boolean")));
    }
    Err(XPathError::type_error(format!(
        "{what} must be an xs:boolean, not an {}",
        atomic.atomic_type()
    )))
}

fn require_map(value: &XpathValue, what: &str) -> XPathResult<XdmMap> {
    value
        .first()
        .and_then(XpathItem::as_map)
        .filter(|_| value.is_singleton())
        .cloned()
        .ok_or_else(|| XPathError::type_error(format!("{what} must be exactly one map")))
}

fn require_array(value: &XpathValue, what: &str) -> XPathResult<XdmArray> {
    value
        .first()
        .and_then(XpathItem::as_array)
        .filter(|_| value.is_singleton())
        .cloned()
        .ok_or_else(|| XPathError::type_error(format!("{what} must be exactly one array")))
}

fn require_qname(value: &XpathValue, what: &str) -> XPathResult<ExpandedName> {
    let atomic = value
        .atomize_at_most_one(what)?
        .ok_or_else(|| XPathError::type_error(format!("{what} must be an xs:QName")))?;
    atomic
        .as_qname()
        .map(oxixml_infoset::QNameValue::expanded_name)
        .ok_or_else(|| XPathError::type_error(format!("{what} must be an xs:QName")))
}

fn require_integer(value: &XpathValue, what: &str) -> XPathResult<i64> {
    let atomic = value
        .atomize_at_most_one(what)?
        .ok_or_else(|| XPathError::type_error(format!("{what} must be an xs:integer")))?;
    let number = oxixml_fn::numeric::cast_to_double(&atomic);
    if number.is_nan() {
        return Err(XPathError::type_error(format!(
            "{what} must be an xs:integer"
        )));
    }
    #[allow(clippy::cast_possible_truncation)]
    Ok(number.trunc().clamp(-9.0e18, 9.0e18) as i64)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::context::StaticContext;
    use crate::function::XpathFunction;
    use std::sync::Arc;

    fn context() -> DynamicContext {
        DynamicContext::new(Arc::new(StaticContext::new()))
    }

    fn doubling() -> XpathValue {
        XpathValue::singleton(XpathFunction::native(None, 1, |_context, arguments| {
            let value = arguments[0]
                .atomize_at_most_one("x")?
                .and_then(|value| value.as_f64())
                .unwrap_or_default();
            Ok(XpathValue::from(AtomicValue::integer(
                #[allow(clippy::cast_possible_truncation)]
                {
                    (value * 2.0) as i64
                },
            )))
        }))
    }

    fn integers(values: &[i64]) -> XpathValue {
        XpathValue::from_iter(values.iter().map(|value| AtomicValue::integer(*value)))
    }

    #[test]
    fn for_each_and_filter() -> XPathResult<()> {
        let context = context();
        let doubled = call_fn(&context, "for-each", &[integers(&[1, 2, 3]), doubling()])?;
        assert_eq!(doubled.to_string(), "2 4 6");

        let odd = XpathValue::singleton(XpathFunction::native(None, 1, |_context, arguments| {
            let value = arguments[0]
                .atomize_at_most_one("x")?
                .and_then(|value| value.as_f64())
                .unwrap_or_default();
            Ok(XpathValue::from(AtomicValue::boolean(value % 2.0 != 0.0)))
        }));
        let filtered = call_fn(&context, "filter", &[integers(&[1, 2, 3, 4]), odd])?;
        assert_eq!(filtered.to_string(), "1 3");
        Ok(())
    }

    #[test]
    fn folds_associate_in_the_right_direction() -> XPathResult<()> {
        let context = context();
        let subtract =
            XpathValue::singleton(XpathFunction::native(None, 2, |_context, arguments| {
                let left = arguments[0]
                    .atomize_at_most_one("l")?
                    .and_then(|value| value.as_f64())
                    .unwrap_or_default();
                let right = arguments[1]
                    .atomize_at_most_one("r")?
                    .and_then(|value| value.as_f64())
                    .unwrap_or_default();
                Ok(XpathValue::from(AtomicValue::integer(
                    #[allow(clippy::cast_possible_truncation)]
                    {
                        (left - right) as i64
                    },
                )))
            }));
        let zero = XpathValue::from(AtomicValue::integer(0_i64));
        assert_eq!(
            call_fn(
                &context,
                "fold-left",
                &[integers(&[1, 2, 3]), zero.clone(), subtract.clone()]
            )?
            .to_string(),
            "-6"
        );
        assert_eq!(
            call_fn(
                &context,
                "fold-right",
                &[integers(&[1, 2, 3]), zero, subtract]
            )?
            .to_string(),
            "2"
        );
        Ok(())
    }

    #[test]
    fn sort_is_stable_and_uses_the_key() -> XPathResult<()> {
        let context = context();
        let sorted = call_fn(&context, "sort", &[integers(&[3, 1, 2])])?;
        assert_eq!(sorted.to_string(), "1 2 3");

        let negate =
            XpathValue::singleton(XpathFunction::native(None, 1, |_context, arguments| {
                let value = arguments[0]
                    .atomize_at_most_one("x")?
                    .and_then(|value| value.as_f64())
                    .unwrap_or_default();
                Ok(XpathValue::from(AtomicValue::integer(
                    #[allow(clippy::cast_possible_truncation)]
                    {
                        -value as i64
                    },
                )))
            }));
        let descending = call_fn(
            &context,
            "sort",
            &[integers(&[3, 1, 2]), XpathValue::empty(), negate],
        )?;
        assert_eq!(descending.to_string(), "3 2 1");
        Ok(())
    }

    /// A map whose keys are the strings `a`, `b` and `c`, each bound to the sequence of the two
    /// integers that follow its position.
    fn sample_map() -> XpathValue {
        let mut map = XdmMap::new();
        for (index, key) in ["a", "b", "c"].into_iter().enumerate() {
            let first = i64::try_from(index).unwrap_or_default() * 2 + 1;
            map.insert(
                AtomicValue::string(key),
                oxixml_infoset::Sequence::from_iter([
                    oxixml_infoset::Item::from(AtomicValue::integer(first)),
                    oxixml_infoset::Item::from(AtomicValue::integer(first + 1)),
                ]),
            );
        }
        XpathValue::from(Item::Map(map))
    }

    fn map_for_each(context: &DynamicContext, action: XpathValue) -> XPathResult<XpathValue> {
        call_collection(
            context,
            &ExpandedName::new(Some(MAP_NAMESPACE), "for-each"),
            &[sample_map(), action],
        )
    }

    #[test]
    fn map_for_each_concatenates_what_the_action_returns() -> XPathResult<()> {
        let context = context();
        // F&O 3.1 §17.1.10: the result is the concatenation of the results of applying `$action`
        // to every entry, so an action returning one item per entry returns one item per entry.
        let key_only =
            XpathValue::singleton(XpathFunction::native(None, 2, |_context, arguments| {
                Ok(arguments[0].clone())
            }));
        assert_eq!(map_for_each(&context, key_only)?.to_string(), "a b c");

        // And an action returning a *sequence* contributes all of it, in order: the value bound
        // to each key is two integers, so nine items come back for three entries.
        let both = XpathValue::singleton(XpathFunction::native(None, 2, |_context, arguments| {
            let mut out = arguments[0].clone();
            out.append(arguments[1].clone());
            Ok(out)
        }));
        assert_eq!(
            map_for_each(&context, both)?.to_string(),
            "a 1 2 b 3 4 c 5 6"
        );
        Ok(())
    }

    #[test]
    fn map_for_each_over_an_empty_map_returns_the_empty_sequence() -> XPathResult<()> {
        let context = context();
        let action =
            XpathValue::singleton(XpathFunction::native(None, 2, |_context, arguments| {
                Ok(arguments[0].clone())
            }));
        let empty = call_collection(
            &context,
            &ExpandedName::new(Some(MAP_NAMESPACE), "for-each"),
            &[XpathValue::from(Item::Map(XdmMap::new())), action],
        )?;
        assert!(empty.is_empty());
        Ok(())
    }

    #[test]
    fn map_for_each_rejects_a_wrongly_typed_argument() {
        /// The error code a call raised, or `None` when it did not raise one.
        fn code(result: XPathResult<XpathValue>) -> Option<XPathErrorCode> {
            result.err().and_then(|error| error.code())
        }

        let context = context();
        let action =
            XpathValue::singleton(XpathFunction::native(None, 2, |_context, arguments| {
                Ok(arguments[0].clone())
            }));
        assert_eq!(
            code(call_collection(
                &context,
                &ExpandedName::new(Some(MAP_NAMESPACE), "for-each"),
                &[XpathValue::from(AtomicValue::integer(1_i64)), action],
            )),
            Some(XPathErrorCode::Xpty0004),
            "the first argument must be a map"
        );
        assert_eq!(
            code(map_for_each(
                &context,
                XpathValue::from(AtomicValue::integer(1_i64))
            )),
            Some(XPathErrorCode::Xpty0004),
            "the second argument must be a function"
        );
    }

    #[test]
    fn function_reflection() -> XPathResult<()> {
        let context = context();
        let function = doubling();
        assert_eq!(
            call_fn(&context, "function-arity", std::slice::from_ref(&function))?.to_string(),
            "1"
        );
        assert!(call_fn(&context, "function-name", &[function])?.is_empty());
        Ok(())
    }
}
