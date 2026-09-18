//! Function calls, function items, map and array constructors, and the lookup operator.
//!
//! An **inline function** closes over the variables in scope where it is written, but *not* over
//! the focus: [XPath 3.1 §3.1.7](https://www.w3.org/TR/xpath-31/#id-inline-func) makes the
//! context item, position and size absent inside the body, so `function() { . }` raises
//! `XPDY0002` however it is called. The closure this module builds therefore captures the
//! dynamic context with [`DynamicContext::without_focus`](crate::DynamicContext::without_focus)
//! applied.
//!
//! **Partial application** — `f(1, ?)` and `$f(?, 2)` — produces a new function item whose arity
//! is the number of placeholders, with the supplied arguments already evaluated. That is what
//! makes `fn:substring(?, 2)` a reusable one-argument function.
//!
//! **Lookup** (`?K`) is defined over both maps and arrays, and over sequences of them: `$x?*`
//! yields every value of every map and every member of every array in `$x`.

use crate::ast::{Expr, KeySpecifier};
use crate::builtins;
use crate::context::DynamicContext;
use crate::error::{XPathError, XPathErrorCode, XPathResult};
use crate::eval::typing::convert;
use crate::function::XpathFunction;
use crate::types::{ItemType, Occurrence, SequenceType};
use crate::value::{XpathItem, XpathValue};
use oxixml_infoset::{AtomicType, AtomicValue, Item, Sequence, XdmArray, XdmMap, same_key};
use std::sync::Arc;

/// Evaluates the call-shaped and constructor-shaped expressions.
///
/// # Errors
///
/// `XPST0017` for an unknown function, `XPTY0004` for a wrongly typed target or argument,
/// `XQDY0137` for duplicate map keys, plus whatever the called function raises.
pub(crate) fn evaluate(expression: &Expr, context: &DynamicContext) -> XPathResult<XpathValue> {
    match expression {
        Expr::FunctionCall { name, arguments } => {
            // `fn:count`/`fn:exists`/`fn:empty` over a `to` range (or `fn:reverse` of one) are
            // answered from the range's bounds, so `count(1 to 2000000000)` costs nothing and
            // never materializes the sequence its argument denotes.
            if arguments.len() == 1 && name.namespace() == Some(oxixml_fn::FN_NAMESPACE) {
                if let Some(result) =
                    super::operators::range_aggregate(name.local_name(), &arguments[0], context)?
                {
                    return Ok(result);
                }
            }
            let mut values = Vec::with_capacity(arguments.len());
            for argument in arguments {
                values.push(super::evaluate(argument, context)?);
            }
            if let Some(constructed) =
                builtins::constructor_from_literal(context, name, arguments, &values)?
            {
                return Ok(constructed);
            }
            builtins::call(context, name, &values)
        }
        Expr::PartialCall { name, arguments } => {
            let function = builtins::function_item(context, name, arguments.len())?;
            let bound = evaluate_slots(arguments, context)?;
            Ok(XpathValue::singleton(function.partially_apply(bound)?))
        }
        Expr::NamedFunctionRef { name, arity } => Ok(XpathValue::singleton(
            builtins::function_item(context, name, *arity)?,
        )),
        Expr::InlineFunction(inline) => Ok(XpathValue::singleton(inline_function(inline, context))),
        Expr::DynamicCall { target, arguments } => {
            let function = require_function(&super::evaluate(target, context)?)?;
            if arguments.iter().any(Option::is_none) {
                let bound = evaluate_slots(arguments, context)?;
                return Ok(XpathValue::singleton(function.partially_apply(bound)?));
            }
            let mut values = Vec::with_capacity(arguments.len());
            for argument in arguments.iter().flatten() {
                values.push(super::evaluate(argument, context)?);
            }
            function.call(context, &values)
        }
        Expr::MapConstructor(entries) => map_constructor(entries, context),
        Expr::ArrayConstructor { curly, members } => array_constructor(*curly, members, context),
        Expr::Lookup { base, key } => lookup(base.as_deref(), key, context),
        other => super::evaluate(other, context),
    }
}

fn evaluate_slots(
    arguments: &[Option<Expr>],
    context: &DynamicContext,
) -> XPathResult<Vec<Option<XpathValue>>> {
    let mut bound = Vec::with_capacity(arguments.len());
    for argument in arguments {
        bound.push(match argument {
            Some(expression) => Some(super::evaluate(expression, context)?),
            None => None,
        });
    }
    Ok(bound)
}

fn inline_function(
    inline: &Arc<crate::ast::InlineFunction>,
    context: &DynamicContext,
) -> XpathFunction {
    let captured = context.clone().without_focus();
    let inline = Arc::clone(inline);
    let arity = inline.parameters.len();
    // XPath 3.1 §3.1.7: a parameter or result the expression leaves untyped is `item()*`. The
    // declared signature is what a `function(…) as …` item test is judged against.
    let default = || SequenceType::new(ItemType::Item, Occurrence::ZeroOrMore);
    let parameters: Vec<SequenceType> = inline
        .parameters
        .iter()
        .map(|(_, declared)| declared.clone().unwrap_or_else(default))
        .collect();
    let result = inline.result.clone().unwrap_or_else(default);
    XpathFunction::native(None, arity, move |_call_context, arguments| {
        let mut scoped = captured.clone();
        for ((name, declared), value) in inline.parameters.iter().zip(arguments.iter()) {
            let value = match declared {
                Some(required) => convert(
                    value.clone(),
                    required,
                    &captured,
                    &format!("the argument `${name}`"),
                )?,
                None => value.clone(),
            };
            scoped = scoped.with_variable_name(name.clone(), value);
        }
        let result = super::evaluate(&inline.body, &scoped)?;
        match &inline.result {
            Some(required) => convert(result, required, &captured, "the function's result"),
            None => Ok(result),
        }
    })
    .with_signature(parameters, result)
}

fn require_function(value: &XpathValue) -> XPathResult<XpathFunction> {
    if value.is_singleton() {
        if let Some(function) = value.first().and_then(XpathItem::as_callable) {
            return Ok(function);
        }
    }
    Err(XPathError::type_error(
        "a dynamic function call needs exactly one function item on its left",
    ))
}

fn map_constructor(entries: &[(Expr, Expr)], context: &DynamicContext) -> XPathResult<XpathValue> {
    let mut map = XdmMap::new();
    for (key, value) in entries {
        let key = super::evaluate(key, context)?
            .atomize_at_most_one("a map key")?
            .ok_or_else(|| XPathError::type_error("a map key must be exactly one atomic value"))?;
        if map.keys().any(|existing| same_key(existing, &key)) {
            return Err(XPathError::with_name(
                oxixml_infoset::ExpandedName::new(Some(oxixml_fn::ERROR_NAMESPACE), "XQDY0137"),
                format!(
                    "the map constructor repeats the key `{}`",
                    key.string_value()
                ),
            ));
        }
        let value = super::evaluate(value, context)?.into_sequence()?;
        map = map.put(key, value);
    }
    Ok(XpathValue::from(Item::Map(map)))
}

fn array_constructor(
    curly: bool,
    members: &[Expr],
    context: &DynamicContext,
) -> XPathResult<XpathValue> {
    let mut collected: Vec<Sequence> = Vec::new();
    if curly {
        // `array { E }` makes one member per *item* of E.
        for member in members {
            for item in super::evaluate(member, context)? {
                collected.push(XpathValue::singleton(item).into_sequence()?);
            }
        }
    } else {
        // `[ E1, E2 ]` makes one member per *expression*, each holding a whole sequence.
        for member in members {
            collected.push(super::evaluate(member, context)?.into_sequence()?);
        }
    }
    Ok(XpathValue::from(Item::Array(XdmArray::from_members(
        collected,
    ))))
}

fn lookup(
    base: Option<&Expr>,
    key: &KeySpecifier,
    context: &DynamicContext,
) -> XPathResult<XpathValue> {
    let input = match base {
        Some(expression) => super::evaluate(expression, context)?,
        None => XpathValue::singleton(context.context_item()?.clone()),
    };
    let keys = match key {
        KeySpecifier::Wildcard => None,
        KeySpecifier::Name(name) => Some(vec![AtomicValue::string(name.clone())]),
        KeySpecifier::Integer(value) => Some(vec![AtomicValue::integer(*value)]),
        KeySpecifier::Computed(expression) => {
            Some(super::evaluate(expression, context)?.atomize()?)
        }
    };
    let mut out = XpathValue::empty();
    for item in input.iter() {
        match item.as_item() {
            Some(Item::Map(map)) => match &keys {
                None => {
                    for (_, value) in map.entries() {
                        out.append(XpathValue::from(value.clone()));
                    }
                }
                Some(keys) => {
                    for key in keys {
                        if let Some(value) = map.get(key) {
                            out.append(XpathValue::from(value.clone()));
                        }
                    }
                }
            },
            Some(Item::Array(array)) => match &keys {
                None => {
                    for member in array.members() {
                        out.append(XpathValue::from(member.clone()));
                    }
                }
                Some(keys) => {
                    for key in keys {
                        let position = array_position(key)?;
                        out.append(XpathValue::from(array.get(position)?.clone()));
                    }
                }
            },
            _ => {
                return Err(XPathError::new(
                    XPathErrorCode::Xpty0004,
                    format!(
                        "the lookup operator needs a map or an array, not {}",
                        item.describe()
                    ),
                ));
            }
        }
    }
    Ok(out)
}

/// The member position an array lookup key names.
///
/// XPath 3.1 §3.11.3.2 applies the function conversion rules to the key, so an
/// `xs:untypedAtomic` — the atomized content of an untyped node — is *cast* to `xs:integer`
/// rather than rejected. A key that is an integer but names no member is out of bounds, which
/// F&O 3.1 §17.3 calls `FOAY0001`, not a type error.
fn array_position(key: &AtomicValue) -> XPathResult<usize> {
    let integer = match key.data() {
        oxixml_infoset::AtomicData::Integer(value) => i64::from(*value),
        _ if key.atomic_type() == AtomicType::UntypedAtomic => {
            let converted = AtomicValue::parse(AtomicType::Integer, &key.string_value())
                .map_err(XPathError::from)?;
            match converted.data() {
                oxixml_infoset::AtomicData::Integer(value) => i64::from(*value),
                _ => {
                    return Err(XPathError::type_error(
                        "an array subscript must be an xs:integer",
                    ));
                }
            }
        }
        _ => {
            return Err(XPathError::type_error(
                "an array subscript must be an xs:integer",
            ));
        }
    };
    usize::try_from(integer).map_err(|_| {
        XPathError::new(
            XPathErrorCode::Foay0001,
            format!("{integer} is not a position in the array"),
        )
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::context::StaticContext;
    use crate::version::XPathVersion;
    use std::sync::Arc;

    fn context() -> DynamicContext {
        DynamicContext::new(Arc::new(StaticContext::new())).with_version(XPathVersion::XPath31)
    }

    fn literal(value: AtomicValue) -> Expr {
        Expr::Literal(value)
    }

    #[test]
    fn array_constructors_differ_in_how_they_group() -> XPathResult<()> {
        let context = context();
        let square = array_constructor(
            false,
            &[Expr::Sequence(vec![
                literal(AtomicValue::integer(1_i64)),
                literal(AtomicValue::integer(2_i64)),
            ])],
            &context,
        )?;
        let array = square.first().and_then(XpathItem::as_array).expect("array");
        assert_eq!(array.len(), 1);

        let curly = array_constructor(
            true,
            &[Expr::Sequence(vec![
                literal(AtomicValue::integer(1_i64)),
                literal(AtomicValue::integer(2_i64)),
            ])],
            &context,
        )?;
        let array = curly.first().and_then(XpathItem::as_array).expect("array");
        assert_eq!(array.len(), 2);
        Ok(())
    }

    #[test]
    fn duplicate_map_keys_are_xqdy0137() {
        let context = context();
        let entries = vec![
            (
                literal(AtomicValue::string("k")),
                literal(AtomicValue::integer(1_i64)),
            ),
            (
                literal(AtomicValue::string("k")),
                literal(AtomicValue::integer(2_i64)),
            ),
        ];
        let failure = map_constructor(&entries, &context).expect_err("duplicate keys");
        assert_eq!(failure.name().local_name(), "XQDY0137");
    }

    #[test]
    fn lookup_reads_maps_and_arrays() -> XPathResult<()> {
        let context = context();
        let map = map_constructor(
            &[(
                literal(AtomicValue::string("a")),
                literal(AtomicValue::integer(7_i64)),
            )],
            &context,
        )?;
        let map_expression = Expr::Literal(AtomicValue::integer(0_i64));
        let _ = map_expression;
        let bound = context.clone().with_variable("m", map);
        let base = Expr::VarRef(oxixml_infoset::ExpandedName::new(None::<String>, "m"));
        assert_eq!(
            lookup(Some(&base), &KeySpecifier::Name("a".to_owned()), &bound)?.to_string(),
            "7"
        );
        assert_eq!(
            lookup(Some(&base), &KeySpecifier::Wildcard, &bound)?.to_string(),
            "7"
        );
        assert!(lookup(Some(&base), &KeySpecifier::Name("z".to_owned()), &bound)?.is_empty());
        Ok(())
    }
}
