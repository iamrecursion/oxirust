//! `cast as`, `castable as`, and the *function conversion rules* an inline function's declared
//! parameter and result types apply.
//!
//! `instance of` and `treat as` need nothing beyond
//! [`SequenceType::matches`](crate::SequenceType::matches) and
//! [`SequenceType::treat`](crate::SequenceType::treat), and are handled in
//! [`super::evaluate`]. Casting is the interesting one, because it atomizes first, allows the
//! empty sequence only when the target carries `?`, and then hands off to [`crate::cast`].

use super::evaluate;
use crate::ast::Expr;
use crate::cast;
use crate::context::DynamicContext;
use crate::error::{XPathError, XPathErrorCode, XPathResult};
use crate::types::{ItemType, ListType, SequenceType, SingleType};
use crate::value::{XpathItem, XpathValue};
use oxixml_infoset::{AtomicType, AtomicValue};

/// `E cast as T`.
///
/// # Errors
///
/// `XPTY0004` when the operand atomizes to more than one value, or to the empty sequence with no
/// `?` on the target; `FORG0001` when the value is not in the target type's value space.
pub(crate) fn cast(
    operand: &Expr,
    target: &SingleType,
    context: &DynamicContext,
) -> XPathResult<XpathValue> {
    let value = evaluate(operand, context)?;
    let Some(atomic) = value.atomize_at_most_one("the operand of `cast as`")? else {
        if target.optional {
            return Ok(XpathValue::empty());
        }
        return Err(XPathError::type_error(format!(
            "the empty sequence cannot be cast to `{target}`"
        )));
    };
    let casted = cast::cast_with(
        &atomic,
        target.atomic_type,
        context.static_context(),
        string_source(operand, context),
    )?;
    Ok(XpathValue::from(casted))
}

/// Whether a cast of `operand` to `xs:QName`/`xs:NOTATION` may resolve a prefix.
///
/// XPath 2.0 §3.12.3 restricted that to a `StringLiteral`; XPath 3.0 removed the restriction
/// (W3C bug 16059), so from 3.0 onwards any string resolves against the statically known
/// namespaces.
fn string_source(operand: &Expr, context: &DynamicContext) -> cast::StringSource {
    if context
        .version()
        .at_least(crate::version::XPathVersion::XPath30)
    {
        return cast::StringSource::Literal;
    }
    match operand {
        Expr::Literal(value) if value.atomic_type() == AtomicType::String => {
            cast::StringSource::Literal
        }
        _ => cast::StringSource::Computed,
    }
}

/// `E castable as T`, which never raises the cast's own error.
///
/// XPath 3.1 §3.14.3 — as revised by erratum
/// [XP.E5](https://www.w3.org/XML/2007/qt-errata/xpath20-errata.html) — is total over the
/// operand's *value*: "the expression `E castable as T` returns `true` if the result of
/// evaluating `E` can be successfully cast into the target type `T` by using a `cast`
/// expression; otherwise it returns `false`. If evaluation of `E` fails with a dynamic error,
/// the `castable` expression as a whole fails." Only the *operand's own* evaluation propagates,
/// so a sequence of more than one item — which `cast as` would reject with `XPTY0004` — makes
/// `castable as` `false` rather than an error. QT3's `CastableAs663` (M. Kay, 2020-11-10),
/// `K-SeqExprCastable-23`/`-34`/`-38` and `sequence-type-10`…`-18` all assert exactly that.
///
/// # Errors
///
/// Whatever evaluating or atomizing the operand raises.
pub(crate) fn castable(
    operand: &Expr,
    target: &SingleType,
    context: &DynamicContext,
) -> XPathResult<XpathValue> {
    let value = evaluate(operand, context)?;
    let atomized = value.atomize()?;
    let atomic = match atomized.as_slice() {
        [] => return Ok(super::boolean(target.optional)),
        [single] => single,
        _ => return Ok(super::boolean(false)),
    };
    Ok(super::boolean(cast::castable(
        atomic,
        target.atomic_type,
        context.static_context(),
        string_source(operand, context),
    )))
}

/// `E cast as T` for one of the three built-in list types.
///
/// [XPath 3.1 §19.3.4](https://www.w3.org/TR/xpath-31/#casting-to-list-types): the input's string
/// value is split on white space and each token is cast to the list's item type, so the result is
/// a *sequence*. All three built-in list types are declared with `minLength = 1`, so a value that
/// yields no token is not in the type at all — `xs:NMTOKENS("")` names no NMTOKEN.
///
/// # Errors
///
/// `XPTY0004` when the operand atomizes to more than one value, or to the empty sequence with no
/// `?` on the target; `FORG0001` when a token is not in the item type's lexical space, or when
/// there is no token.
pub(crate) fn cast_list(
    operand: &Expr,
    target: &ListType,
    context: &DynamicContext,
) -> XPathResult<XpathValue> {
    let value = evaluate(operand, context)?;
    let Some(atomic) = value.atomize_at_most_one("the operand of `cast as`")? else {
        if target.optional {
            return Ok(XpathValue::empty());
        }
        return Err(XPathError::type_error(format!(
            "the empty sequence cannot be cast to `{target}`"
        )));
    };
    Ok(XpathValue::from_iter(cast_tokens(&atomic, target)?))
}

/// `E castable as T` for one of the three built-in list types, which never raises the cast's own
/// error.
///
/// # Errors
///
/// Whatever evaluating or atomizing the operand raises.
pub(crate) fn castable_list(
    operand: &Expr,
    target: &ListType,
    context: &DynamicContext,
) -> XPathResult<XpathValue> {
    let value = evaluate(operand, context)?;
    let atomized = value.atomize()?;
    let atomic = match atomized.as_slice() {
        [] => return Ok(super::boolean(target.optional)),
        [single] => single,
        // A sequence of more than one item is not a single value to cast, so the answer is
        // `false` rather than the `XPTY0004` a `cast as` would raise — the same rule
        // [`castable`] applies, and QT3's `cbcl-castable-ENTITIES-007` pins it for a list type.
        _ => return Ok(super::boolean(false)),
    };
    Ok(super::boolean(cast_tokens(atomic, target).is_ok()))
}

/// Splits `value`'s string form and casts every token to the list's item type.
///
/// This is the whole of [XPath 3.1
/// §19.3.4](https://www.w3.org/TR/xpath-31/#casting-to-list-types), and it is shared by `cast
/// as`, `castable as` and the list type's own constructor function so that the three cannot
/// disagree.
///
/// # Errors
///
/// `XPTY0004` when the input is neither `xs:string`, `xs:untypedAtomic` nor a type derived from
/// `xs:string`; `FORG0001` when a token is not in the item type's lexical space, or when the
/// input holds no token at all.
pub(crate) fn cast_tokens(value: &AtomicValue, target: &ListType) -> XPathResult<Vec<AtomicValue>> {
    // "The supplied value must be an instance of xs:string or xs:untypedAtomic (or a type derived
    // from one of these)": `xs:ENTITY("a")` is a string in the type hierarchy and casts, while
    // `xs:anyURI("abcd")` is not (QT3's `CastAs-ListType-17`).
    if !derived_from_string(value.atomic_type()) {
        return Err(XPathError::type_error(format!(
            "`xs:{}` cannot be cast to `{target}`, which takes a string",
            value.atomic_type().local_name()
        )));
    }
    let lexical = value.string_value();
    let mut items = Vec::new();
    // The list types carry `whiteSpace = collapse`, so the separator is a run of XML white space
    // and leading and trailing runs contribute no token. XML white space is the four characters
    // of the `S` production and nothing else: a value separated by, say, a no-break space is one
    // token holding a character no item type admits, not two tokens.
    for token in lexical
        .split([' ', '\t', '\r', '\n'])
        .filter(|token| !token.is_empty())
    {
        items.push(AtomicValue::parse(target.member, token)?);
    }
    if items.is_empty() {
        return Err(XPathError::new(
            XPathErrorCode::Forg0001,
            format!("`{lexical}` holds no token, and `{target}` requires at least one"),
        ));
    }
    Ok(items)
}

/// Whether `atomic_type` is `xs:untypedAtomic`, `xs:string`, or derived from `xs:string`.
fn derived_from_string(atomic_type: AtomicType) -> bool {
    let mut current = atomic_type;
    loop {
        match current {
            AtomicType::String | AtomicType::UntypedAtomic => return true,
            _ => match current.base_type() {
                Some(base) => current = base,
                None => return false,
            },
        }
    }
}

/// The [function conversion rules](https://www.w3.org/TR/xpath-31/#dt-function-conversion) an
/// inline function applies to an argument or to its result.
///
/// When the required type is atomic, the value is atomized, `xs:untypedAtomic` is cast to the
/// required type, and `xs:decimal`/`xs:float` are promoted to `xs:double` (and `xs:anyURI` to
/// `xs:string`) where the required type asks for it. The converted value must then match the
/// required type, or `XPTY0004` is raised.
///
/// # Errors
///
/// `XPTY0004` when the value cannot be converted to the required type.
pub(crate) fn convert(
    value: XpathValue,
    required: &SequenceType,
    context: &DynamicContext,
    what: &str,
) -> XPathResult<XpathValue> {
    if let SequenceType::Items {
        item_type: item_type @ ItemType::TypedFunction { .. },
        occurrence,
    } = required
    {
        // Coercion applies only to a function item that is *not already* an instance of the
        // required type — otherwise every hand-off through a typed parameter would add another
        // wrapper, and a recursive function that passes one along would build an unbounded
        // chain of them.
        if occurrence.allows(value.len()) && !required.matches(&value) {
            if let Some(coerced) = coerce_functions(&value, item_type) {
                return Ok(coerced);
            }
        }
    }
    let SequenceType::Items {
        item_type: ItemType::Atomic(target),
        occurrence,
    } = required
    else {
        return check(value, required, what);
    };
    let mut converted = Vec::new();
    for atomic in value.atomize()? {
        converted.push(promote(atomic, *target, context)?);
    }
    if !occurrence.allows(converted.len()) {
        return Err(mismatch(what, required, converted.len()));
    }
    let value = XpathValue::from_iter(converted);
    check(value, required, what)
}

/// *Function coercion*, [XPath 3.1
/// §3.1.5.3](https://www.w3.org/TR/xpath-31/#dt-function-coercion): a function item supplied
/// where a `function(…) as …` type is expected, and whose declared signature is not a subtype of
/// it, is **not** rejected — it is wrapped in a new function item that has the expected
/// signature and delegates, so the argument and result types are checked when it is *called*.
///
/// That is what lets `function($e as element(employee)) as xs:integer {…}` be passed where
/// `function(item()) as xs:anyAtomicType` is required (QT3 `fold-left-008`): the parameter types
/// are contravariant, so the declared signature is not a subtype, but every actual call happens
/// to supply an `element(employee)`.
///
/// Answers `None` when the value is not a sequence of function items of the right arity, leaving
/// the ordinary type check to report the mismatch.
fn coerce_functions(value: &XpathValue, required: &ItemType) -> Option<XpathValue> {
    let ItemType::TypedFunction { parameters, result } = required else {
        return None;
    };
    let mut coerced = Vec::with_capacity(value.len());
    for item in value.as_slice() {
        let function = item.as_callable()?;
        if function.arity() != parameters.len() {
            return None;
        }
        let result_type = (**result).clone();
        let delegate = function.clone();
        let wrapper = crate::function::XpathFunction::native(
            function.name().cloned(),
            function.arity(),
            move |context, arguments| {
                let produced = delegate.call(context, arguments)?;
                check(produced, &result_type, "the function's result")
            },
        )
        .with_signature(parameters.clone(), (**result).clone());
        coerced.push(XpathItem::from(wrapper));
    }
    Some(XpathValue::from_iter(coerced))
}

fn promote(
    value: AtomicValue,
    target: AtomicType,
    context: &DynamicContext,
) -> XPathResult<AtomicValue> {
    if value.atomic_type().is_derived_from(target) {
        return Ok(value);
    }
    // XPath 3.1 §3.1.5.2, function conversion rules: "if the expected type is a namespace
    // sensitive type, and the supplied value is of type xs:untypedAtomic, a type error is raised
    // [err:XPTY0117]". The namespace-sensitive types are xs:QName, xs:NOTATION and their
    // derivations — a lexical `p:local` arriving as untyped data has no prefix bindings the
    // callee could honour, so the conversion is forbidden rather than guessed at.
    if value.atomic_type() == AtomicType::UntypedAtomic
        && matches!(target.primitive(), AtomicType::QName | AtomicType::Notation)
    {
        return Err(XPathError::new(
            XPathErrorCode::Xpty0117,
            format!(
                "an xs:untypedAtomic value cannot be converted to the namespace-sensitive \
                 type {target} by the function conversion rules"
            ),
        ));
    }
    let promotable = value.atomic_type() == AtomicType::UntypedAtomic
        || (target == AtomicType::Double && value.is_numeric())
        || (target == AtomicType::Float
            && value.atomic_type().is_derived_from(AtomicType::Decimal))
        || (target == AtomicType::String && value.atomic_type() == AtomicType::AnyUri);
    if !promotable {
        return Ok(value);
    }
    cast::cast(&value, target, context.static_context())
}

fn check(value: XpathValue, required: &SequenceType, what: &str) -> XPathResult<XpathValue> {
    if required.matches(&value) {
        return Ok(value);
    }
    Err(mismatch(what, required, value.len()))
}

fn mismatch(what: &str, required: &SequenceType, count: usize) -> XPathError {
    XPathError::type_error(format!(
        "{what} does not match the required type `{required}` (the value holds {count} item(s))"
    ))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::context::StaticContext;
    use crate::types::Occurrence;
    use crate::version::XPathVersion;
    use std::sync::Arc;

    fn one(item_type: ItemType) -> SequenceType {
        SequenceType::new(item_type, Occurrence::One)
    }

    fn context() -> DynamicContext {
        DynamicContext::new(Arc::new(StaticContext::new())).with_version(XPathVersion::XPath31)
    }

    #[test]
    fn casting_the_empty_sequence_needs_a_question_mark() -> XPathResult<()> {
        let context = context();
        let empty = Expr::Sequence(Vec::new());
        assert!(
            cast(
                &empty,
                &SingleType::new(AtomicType::String, false),
                &context
            )
            .is_err()
        );
        assert!(cast(&empty, &SingleType::new(AtomicType::String, true), &context)?.is_empty());
        Ok(())
    }

    /// A list type's `whiteSpace = collapse` facet separates on XML white space — `#x20`, `#x9`,
    /// `#xD` and `#xA` — and on nothing else, so a no-break space is part of the token it sits
    /// in rather than a separator, and the token it makes is in no item type's lexical space.
    #[test]
    fn a_list_cast_separates_on_xml_white_space_only() -> XPathResult<()> {
        let idrefs = ListType::new(AtomicType::Idref, false);
        let spaced = cast_tokens(&AtomicValue::string(" one \r\n two\t"), &idrefs)?;
        assert_eq!(spaced.len(), 2);
        assert_eq!(spaced[0].string_value(), "one");
        assert_eq!(spaced[1].string_value(), "two");

        assert_eq!(
            cast_tokens(&AtomicValue::string("one\u{a0}two"), &idrefs)
                .expect_err("`one\u{a0}two` is one token and no `xs:IDREF`")
                .code(),
            Some(XPathErrorCode::Forg0001)
        );
        assert_eq!(
            cast_tokens(&AtomicValue::string("  \n "), &idrefs)
                .expect_err("a value of nothing but separators holds no token")
                .code(),
            Some(XPathErrorCode::Forg0001)
        );
        Ok(())
    }

    #[test]
    fn castable_never_raises_the_cast_error() -> XPathResult<()> {
        let context = context();
        let text = Expr::Literal(AtomicValue::string("x"));
        assert_eq!(
            castable(
                &text,
                &SingleType::new(AtomicType::Integer, false),
                &context
            )?
            .to_string(),
            "false"
        );
        assert_eq!(
            castable(&text, &SingleType::new(AtomicType::String, false), &context)?.to_string(),
            "true"
        );
        Ok(())
    }

    #[test]
    fn castable_is_false_rather_than_an_error_for_a_longer_sequence() -> XPathResult<()> {
        let context = context();
        let pair = Expr::Sequence(vec![
            Expr::Literal(AtomicValue::string("one")),
            Expr::Literal(AtomicValue::string("two")),
        ]);
        assert_eq!(
            castable(&pair, &SingleType::new(AtomicType::String, true), &context)?.to_string(),
            "false",
            "XP.E5: only the operand's own dynamic error propagates"
        );
        assert_eq!(
            castable(
                &pair,
                &SingleType::new(AtomicType::Integer, false),
                &context
            )?
            .to_string(),
            "false"
        );
        Ok(())
    }

    /// XPath 2.0 §3.12.3 lets only a `StringLiteral` be cast to `xs:QName`; XPath 3.0 removed
    /// that restriction (W3C bug 16059).
    #[test]
    fn only_xpath_20_restricts_a_qname_cast_to_a_string_literal() -> XPathResult<()> {
        let literal = Expr::Literal(AtomicValue::string("ncname"));
        let computed = Expr::Sequence(vec![literal.clone()]);
        let target = SingleType::new(AtomicType::QName, false);

        let two =
            DynamicContext::new(Arc::new(StaticContext::new())).with_version(XPathVersion::XPath20);
        assert_eq!(castable(&literal, &target, &two)?.to_string(), "true");
        assert_eq!(
            castable(&computed, &target, &two)?.to_string(),
            "false",
            "XPTY0004 inside the cast, and so `false` here"
        );
        assert!(cast(&computed, &target, &two).is_err());

        let three = context();
        assert_eq!(castable(&computed, &target, &three)?.to_string(), "true");
        assert_eq!(
            cast(&computed, &target, &three)?.to_string(),
            "ncname",
            "3.0 resolves against the statically known namespaces whenever the cast runs"
        );
        Ok(())
    }

    #[test]
    fn function_conversion_atomizes_and_promotes() -> XPathResult<()> {
        let context = context();
        let required = one(ItemType::Atomic(AtomicType::Double));
        let converted = convert(
            XpathValue::from(AtomicValue::integer(1_i64)),
            &required,
            &context,
            "the argument",
        )?;
        assert_eq!(converted.to_string(), "1");

        let untyped = convert(
            XpathValue::from(AtomicValue::untyped_atomic("2")),
            &required,
            &context,
            "the argument",
        )?;
        assert_eq!(untyped.to_string(), "2");

        let wrong = convert(
            XpathValue::from(AtomicValue::string("x")),
            &required,
            &context,
            "the argument",
        );
        assert!(wrong.is_err());
        Ok(())
    }

    #[test]
    fn non_atomic_required_types_are_checked_as_they_are() -> XPathResult<()> {
        let context = context();
        let required = SequenceType::new(ItemType::Item, Occurrence::ZeroOrMore);
        let value = XpathValue::from(AtomicValue::string("x"));
        assert_eq!(
            convert(value, &required, &context, "the argument")?.len(),
            1
        );
        Ok(())
    }
}
