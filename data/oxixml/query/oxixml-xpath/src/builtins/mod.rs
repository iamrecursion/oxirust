//! Function dispatch: the [`oxixml_fn`] library, plus the functions only an *evaluator* can
//! supply.
//!
//! [`oxixml_fn`] implements F&O 3.1 as a library of plain Rust functions over XDM values, and
//! [`FunctionRegistry`](oxixml_fn::FunctionRegistry) exposes them by name and arity. Three
//! families cannot live there, and live here instead:
//!
//! * **context-dependent** functions — `fn:position()`, `fn:last()`, and the zero-argument forms
//!   of `fn:string`, `fn:name`, `fn:root` and friends, which read the focus;
//! * **higher-order** functions — `fn:for-each`, `fn:filter`, `fn:fold-left`, `fn:sort`,
//!   `fn:apply`, `fn:function-lookup` and the `map:`/`array:` equivalents, whose arguments are
//!   function items, which the data model has no representation for;
//! * **external-resource** functions — `fn:doc`, `fn:collection` and `fn:unparsed-text`, which
//!   need an I/O policy, supplied by a
//!   [`DocumentResolver`](crate::DocumentResolver).
//!
//! The `math:` namespace and the `xs:*` constructor functions are here too: the first because it
//! is a self-contained numerics module, the second because a constructor is a `cast as` in
//! disguise, and casting needs the static context's namespaces.

mod higher_order;
mod json;
mod math;
mod random;
mod resources;
mod sequences;
pub(crate) mod versions;
mod xpath10;

use crate::cast;
use crate::context::{
    ARRAY_NAMESPACE, DynamicContext, InvocationKind, MAP_NAMESPACE, MATH_NAMESPACE, StaticContext,
};
use crate::error::{XPathError, XPathErrorCode, XPathResult};
use crate::function::XpathFunction;
use crate::types::{ItemType, KindTest, Occurrence, SequenceType};
use crate::value::{XpathItem, XpathValue};
use crate::version::XPathVersion;
use oxixml_fn::{FN_NAMESPACE, nodes};
use oxixml_infoset::{AtomicType, AtomicValue, ExpandedName, XSD_NAMESPACE, XdmNode};

/// Whether a function of this name and arity is in scope, for the parser's `XPST0017` check.
///
/// `version` decides which vintage of the function library is in scope: a function F&O added in
/// 3.0 or 3.1 is simply not there in an expression compiled at an earlier version. See
/// [`versions::available_in`].
pub(crate) fn is_known(
    context: &StaticContext,
    name: &ExpandedName,
    arity: usize,
    version: XPathVersion,
) -> bool {
    if !versions::available_in(name, arity, version, context.has_xslt_functions()) {
        return false;
    }
    engine_supports(name, arity)
        || is_numeric_constructor(name, arity)
        || is_error_constructor(name, arity)
        || constructor_type(name, arity).is_some()
        || list_constructor_type(name, arity).is_some()
        || context.functions().contains(name, arity)
}

/// Builds the function item a named function reference denotes.
pub(crate) fn function_item(
    context: &DynamicContext,
    name: &ExpandedName,
    arity: usize,
) -> XPathResult<XpathFunction> {
    if !is_known(context.static_context(), name, arity, context.version()) {
        return Err(XPathError::new(
            XPathErrorCode::Xpst0017,
            format!("no function `{name}` of arity {arity} is in scope"),
        ));
    }
    let owned = name.clone();
    // XPath 3.1 §3.1.6: "if the named function is context-dependent, the function item is a
    // closure over the *static and dynamic context* in which the named function reference is
    // evaluated". `fn:lang#1` taken while `para[4]` is the context item keeps reading `para[4]`
    // however it is later applied, so the defining context — not the call site's — is what the
    // built-in sees.
    let defining = context.clone();
    // Everything reached through this closure is reached through a *function item*, whatever
    // later applies it, so the call it makes is a dynamic one — see [`InvocationKind`].
    let function =
        XpathFunction::native(Some(name.clone()), arity, move |_call_site, arguments| {
            call_as(&defining, &owned, arguments, InvocationKind::Dynamic)
        });
    // A function the host registered — an XQuery `declare function`, say — is not in F&O's
    // table, and only the host knows the types it declared.
    let signature = context
        .static_context()
        .host_signature(name, arity)
        .or_else(|| declared_signature(name, arity));
    Ok(match signature {
        Some((parameters, result)) => function.with_signature(parameters, result),
        None => function,
    })
}

/// The signature a built-in *declares*, translated into the engine's own [`SequenceType`]s.
///
/// XPath 3.1 §2.5.5.5 judges a function item against `function(A₁, …, Aₙ) as R` by its declared
/// signature, so a built-in that carries none can only be checked for arity — `fn:abs#1` would
/// then satisfy `function(xs:string) as xs:date` as readily as its real signature.
/// [`oxixml_fn::signature`] holds what F&O declares, and both halves have to be known for the
/// judgement to be sound: [`None`] here leaves the item unsignatured and the test arity-only,
/// which is what this engine did for every built-in before.
fn declared_signature(
    name: &ExpandedName,
    arity: usize,
) -> Option<(Vec<SequenceType>, SequenceType)> {
    let namespace = name.namespace()?;
    let local = name.local_name();
    let declared = oxixml_fn::signature::signature(namespace, local, arity)?;
    let result = oxixml_fn::signature::result_type(namespace, local, arity)?;
    // A variadic entry such as `fn:concat` is served a prefix of its parameter table, and only a
    // signature of exactly the reference's arity describes the item being built.
    if declared.len() != arity {
        return None;
    }
    let parameters = declared
        .iter()
        .map(|parameter| declared_type(*parameter))
        .collect::<Option<Vec<_>>>()?;
    Some((parameters, declared_type(result)?))
}

/// Translates one declared type of [`oxixml_fn::signature`] into this engine's own.
///
/// [`None`] for an item type the function library has grown and this engine does not yet map,
/// which leaves the whole signature undeclared rather than declaring a wrong one.
fn declared_type(declared: oxixml_fn::signature::ParamType) -> Option<SequenceType> {
    use oxixml_fn::signature::{Occurrence as Declared, ParamItemType};

    let item_type = match declared.item_type() {
        ParamItemType::Item => ItemType::Item,
        ParamItemType::Node => ItemType::Kind(KindTest::AnyKind),
        ParamItemType::Numeric => ItemType::Numeric,
        ParamItemType::Atomic(atomic) => ItemType::Atomic(atomic),
        ParamItemType::Map => ItemType::AnyMap,
        ParamItemType::Array => ItemType::AnyArray,
        ParamItemType::Function => ItemType::AnyFunction,
        _ => return None,
    };
    let occurrence = match declared.occurrence() {
        Declared::One => Occurrence::One,
        Declared::ZeroOrOne => Occurrence::ZeroOrOne,
        Declared::ZeroOrMore => Occurrence::ZeroOrMore,
        Declared::OneOrMore => Occurrence::OneOrMore,
    };
    Some(SequenceType::new(item_type, occurrence))
}

/// The namespace-sensitive constructor functions — `xs:QName` and any `xs:NOTATION`
/// derivation — applied to a `StringLiteral` argument.
///
/// A constructor function is a `cast as` in disguise, so it inherits XPath 3.1 §3.14.2's rule
/// that only a *literal* string carries enough static information to resolve a prefix. The
/// dispatcher below never sees the argument *expression*, so the literal case is recognized
/// here, before evaluation loses that information; every other shape falls through and is
/// rejected by [`cast::cast`] with `XPTY0004`.
///
/// `Ok(None)` means "not this case, carry on".
///
/// # Errors
///
/// Whatever the cast raises — `FONS0004` for an unbound prefix, `FORG0001` for a lexical form
/// that is not a `QName`.
pub(crate) fn constructor_from_literal(
    context: &DynamicContext,
    name: &ExpandedName,
    arguments: &[crate::ast::Expr],
    values: &[XpathValue],
) -> XPathResult<Option<XpathValue>> {
    let [crate::ast::Expr::Literal(literal)] = arguments else {
        return Ok(None);
    };
    if literal.atomic_type() != AtomicType::String {
        return Ok(None);
    }
    let Some(target) = constructor_type(name, values.len()) else {
        return Ok(None);
    };
    if !matches!(target.primitive(), AtomicType::QName | AtomicType::Notation) {
        return Ok(None);
    }
    let Some(value) = values[0]
        .clone()
        .atomize_at_most_one("a constructor function")?
    else {
        return Ok(Some(XpathValue::empty()));
    };
    let casted = cast::cast_with(
        &value,
        target,
        context.static_context(),
        cast::StringSource::Literal,
    )?;
    Ok(Some(XpathValue::from(casted)))
}

/// Invokes the function `name` with `arguments`.
///
/// # Errors
///
/// [`XPathErrorCode::Xpst0017`] when no such function exists, plus whatever the function raises.
pub(crate) fn call(
    context: &DynamicContext,
    name: &ExpandedName,
    arguments: &[XpathValue],
) -> XPathResult<XpathValue> {
    call_as(context, name, arguments, InvocationKind::Static)
}

/// [`call`], recording how the function was reached for the benefit of a host-language function
/// that has to distinguish a static call from a dynamic one ([`InvocationKind`]).
fn call_as(
    context: &DynamicContext,
    name: &ExpandedName,
    arguments: &[XpathValue],
    kind: InvocationKind,
) -> XPathResult<XpathValue> {
    // XPath 1.0 converts every argument to the type its signature expects, rather than raising
    // a type error the way XPath 2.0 does.
    if context.version().is_xpath_10() && name.namespace() == Some(FN_NAMESPACE) {
        if let Some(result) = xpath10::call(context, name.local_name(), arguments)? {
            return Ok(result);
        }
        if let Some(converted) = xpath10::coerce(name.local_name(), arguments)? {
            return dispatch(context, name, &converted, kind);
        }
    }
    dispatch(context, name, arguments, kind)
}

/// Applies the *function conversion rules* of XPath 3.1 §3.1.5 to the arguments of a built-in
/// whose signature [`oxixml_fn::signature`] declares, before this crate's own implementation of
/// it runs.
///
/// [`oxixml_fn::FunctionRegistry`] already does this for the functions it serves; the families
/// this crate implements itself — the sequence functions, the resource functions, the
/// higher-order functions — bypass the registry, so the same table is consulted here. Only
/// parameters whose declared type is *atomic* are converted: those are the ones where a wrong
/// type would otherwise be silently coerced, and they are the only ones whose value is
/// guaranteed to fit in a [`Sequence`](oxixml_infoset::Sequence) (a general function item does
/// not).
///
/// XPath 1.0 has no type errors — it converts every argument to the type its signature expects —
/// so the rules are not applied in that version.
fn convert_arguments(
    context: &DynamicContext,
    name: &ExpandedName,
    arguments: &[XpathValue],
) -> XPathResult<Option<Vec<XpathValue>>> {
    use oxixml_fn::signature::{ParamItemType, convert_argument, signature};

    if context.version().is_xpath_10() {
        return Ok(None);
    }
    let Some(namespace) = name.namespace() else {
        return Ok(None);
    };
    let Some(parameters) = signature(namespace, name.local_name(), arguments.len()) else {
        return Ok(None);
    };
    let what = format!("{}#{}", name.local_name(), arguments.len());
    let mut converted = Vec::with_capacity(arguments.len());
    for (index, argument) in arguments.iter().enumerate() {
        let required = parameters.get(index);
        let atomic = required.is_some_and(|required| {
            matches!(
                required.item_type(),
                ParamItemType::Atomic(_) | ParamItemType::Numeric
            )
        });
        match (atomic, required) {
            (true, Some(required)) => {
                let sequence = argument.clone().into_sequence()?;
                let outcome = convert_argument(&sequence, required, &what)
                    .map_err(|error| downgrade_xpty0117(error, context.version()));
                converted.push(XpathValue::from(outcome?));
            }
            _ => converted.push(argument.clone()),
        }
    }
    Ok(Some(converted))
}

/// Reports the namespace-sensitive conversion failure with the code the version knows.
///
/// XPath 3.0 gave the condition its own code, `XPTY0117` (W3C bug 16088); before that it was the
/// general `XPTY0004`. QT3 pins both — `LocalNameFromQNameFunc015` under `XP20` against
/// `-015a` under `XP30+`.
fn downgrade_xpty0117(error: oxixml_fn::FnError, version: XPathVersion) -> XPathError {
    let converted = XPathError::from(error);
    if converted.code() == Some(XPathErrorCode::Xpty0117)
        && !version.at_least(XPathVersion::XPath30)
    {
        return XPathError::type_error(converted.to_string());
    }
    converted
}

fn dispatch(
    context: &DynamicContext,
    name: &ExpandedName,
    arguments: &[XpathValue],
    kind: InvocationKind,
) -> XPathResult<XpathValue> {
    if let Some(converted) = convert_arguments(context, name, arguments)? {
        return dispatch_converted(context, name, &converted, kind);
    }
    dispatch_converted(context, name, arguments, kind)
}

fn dispatch_converted(
    context: &DynamicContext,
    name: &ExpandedName,
    arguments: &[XpathValue],
    kind: InvocationKind,
) -> XPathResult<XpathValue> {
    let arity = arguments.len();
    if let Some(result) = call_engine(context, name, arguments)? {
        return Ok(result);
    }
    // XPath 3.1 §2.5.4 gives the union type `xs:numeric` a constructor function like any other
    // built-in type; §19.3's rule for casting to a union leaves a value that already is one of
    // the members alone, and casts everything else to `xs:double`.
    if is_numeric_constructor(name, arity) {
        let Some(value) = arguments[0]
            .clone()
            .atomize_at_most_one("a constructor function")?
        else {
            return Ok(XpathValue::empty());
        };
        if value.is_numeric() {
            return Ok(XpathValue::from(value));
        }
        return construct(context, AtomicType::Double, &arguments[0]);
    }
    if is_error_constructor(name, arity) {
        return construct_error(&arguments[0]);
    }
    if let Some(target) = constructor_type(name, arity) {
        return construct(context, target, &arguments[0]);
    }
    if let Some(member) = list_constructor_type(name, arity) {
        return construct_list(member, &arguments[0]);
    }
    if let Some(function) = context.static_context().functions().lookup(name, arity) {
        let mut converted = Vec::with_capacity(arity);
        for argument in arguments {
            converted.push(argument.clone().into_sequence()?);
        }
        // A registered function receives only an [`oxixml_fn::FnContext`], which by design
        // carries no focus. A host language may none the less define a function whose meaning
        // depends on the context item at the *call site* — XSLT's `fn:key` and `fn:id` identify
        // the document to search that way, and `$e/key('k','v')` rebinds that item per step.
        // Publishing the focus around the call is what lets such a function see the live value
        // rather than whichever one the host happened to record when evaluation began. `kind`
        // travels with it, for the host functions that must reject a dynamic call outright.
        return crate::context::with_call_site(context.focus(), kind, || {
            Ok(XpathValue::from(
                function.call(context.fn_context(), &converted)?,
            ))
        });
    }
    Err(XPathError::new(
        XPathErrorCode::Xpst0017,
        format!("no function `{name}` of arity {arity} is in scope"),
    ))
}

/// Whether `name` is the constructor function of the union type `xs:numeric`, which XPath 3.1
/// §2.5.4 defines alongside the atomic types but which is not itself an [`AtomicType`].
fn is_numeric_constructor(name: &ExpandedName, arity: usize) -> bool {
    arity == 1 && name.namespace() == Some(XSD_NAMESPACE) && name.local_name() == "numeric"
}

/// The atomic type an `xs:*` constructor function builds, if `name` names one.
fn constructor_type(name: &ExpandedName, arity: usize) -> Option<AtomicType> {
    if arity != 1 || name.namespace() != Some(XSD_NAMESPACE) {
        return None;
    }
    let atomic_type = AtomicType::from_local_name(name.local_name())?;
    (!matches!(
        atomic_type,
        AtomicType::AnyAtomicType | AtomicType::Notation
    ))
    .then_some(atomic_type)
}

/// Whether `name` is the constructor function of the built-in *union* type `xs:error`.
///
/// F&O 3.1 §18.4 *Constructor functions for XML Schema built-in union types* gives it the
/// signature `xs:error($arg as xs:anyAtomicType?) as xs:error?` and defines its semantics as
/// "equivalent to casting to the corresponding union type" — with the consequence spelled out in
/// the same section: "Because `xs:error` has no member types, and therefore has an empty value
/// space, casting will always fail with a dynamic error except in the case where the supplied
/// argument is an empty sequence, in which case the result is also an empty sequence."
///
/// It is not an [`AtomicType`], and deliberately not made one: no value can ever have this type,
/// so an [`AtomicValue`] must never be able to claim it. Only the constructor function exists,
/// and it is the one entry of the F&O function list that no cast can serve.
fn is_error_constructor(name: &ExpandedName, arity: usize) -> bool {
    arity == 1 && name.namespace() == Some(XSD_NAMESPACE) && name.local_name() == "error"
}

/// Applies `xs:error`: the empty sequence for an empty argument, `FORG0001` for anything else.
fn construct_error(argument: &XpathValue) -> XPathResult<XpathValue> {
    let Some(value) = argument.clone().atomize_at_most_one("xs:error")? else {
        return Ok(XpathValue::empty());
    };
    Err(XPathError::new(
        XPathErrorCode::Forg0001,
        format!(
            "`{}` cannot be cast to xs:error, which has an empty value space",
            value.string_value()
        ),
    ))
}

/// The *item* type of the three built-in list types XML Schema derives over the `xs:IDREF`,
/// `xs:NMTOKEN` and `xs:ENTITY` hierarchies.
///
/// F&O 3.1 §18.1 gives every built-in type with a lexical form a constructor function, list
/// types included: `xs:IDREFS("a b")` is the sequence `("a", "b")` of `xs:IDREF` values. They
/// are not [`AtomicType`]s — a list is a *sequence* in the data model — so they are constructed
/// here rather than through [`constructor_type`].
fn list_constructor_type(name: &ExpandedName, arity: usize) -> Option<AtomicType> {
    if arity != 1 || name.namespace() != Some(XSD_NAMESPACE) {
        return None;
    }
    match name.local_name() {
        "IDREFS" => Some(AtomicType::Idref),
        "NMTOKENS" => Some(AtomicType::Nmtoken),
        "ENTITIES" => Some(AtomicType::Entity),
        _ => None,
    }
}

fn construct_list(member: AtomicType, argument: &XpathValue) -> XPathResult<XpathValue> {
    let Some(value) = argument.atomize_at_most_one("a constructor function")? else {
        return Ok(XpathValue::empty());
    };
    // F&O 3.1 §18.3: "the semantics of the constructor function are the same as casting to the
    // type", so the constructor and `cast as xs:IDREFS` are one implementation.
    let target = crate::types::ListType::new(member, false);
    Ok(XpathValue::from_iter(crate::eval::typing::cast_tokens(
        &value, &target,
    )?))
}

fn construct(
    context: &DynamicContext,
    target: AtomicType,
    argument: &XpathValue,
) -> XPathResult<XpathValue> {
    let Some(value) = argument.atomize_at_most_one("a constructor function")? else {
        return Ok(XpathValue::empty());
    };
    // XPath 3.0 dropped XPath 2.0's rule that only a string *literal* casts to a
    // namespace-sensitive type, so from 3.0 onwards a computed string resolves its prefix
    // against the statically known namespaces like any other.
    let source_kind = if context
        .version()
        .at_least(crate::version::XPathVersion::XPath30)
    {
        cast::StringSource::Literal
    } else {
        cast::StringSource::Computed
    };
    let casted = cast::cast_with(&value, target, context.static_context(), source_kind)?;
    Ok(XpathValue::from(casted))
}

/// Whether this engine serves the function `name` at `arity` itself, rather than through
/// [`oxixml_fn`]'s [`FunctionRegistry`](oxixml_fn::FunctionRegistry).
///
/// A registry entry is a closure over sequences, so three families of function cannot be one:
/// the context-dependent functions, which read the focus; the higher-order functions, whose
/// arguments are function items only an evaluator can apply; and the external-resource functions,
/// which need a retrieval policy. This engine serves all three, and they are as much part of the
/// function library as `fn:abs` is.
///
/// A host that answers questions *about* the library — XSLT's `fn:function-available`, an editor
/// completing a name — therefore has to ask both: the registry for what it holds, and this for
/// the rest. It is the predicate behind [`XPath`](crate::XPath)'s own `XPST0017` check, so a
/// name it accepts is
/// a name an expression may call, and one it rejects at that arity is not.
///
/// Constructor functions (`xs:date#1` and friends) are not covered: they are a `cast as` in
/// disguise rather than library entries, and a host that wants them asks about the *type*.
///
/// # Examples
///
/// ```
/// use oxixml_infoset::ExpandedName;
/// use oxixml_xpath::{FN_NAMESPACE, MAP_NAMESPACE, supports_function};
///
/// // A higher-order function, which no registry of sequence-to-sequence closures can hold.
/// let for_each = ExpandedName::new(Some(MAP_NAMESPACE), "for-each");
/// assert!(supports_function(&for_each, 2));
/// assert!(!supports_function(&for_each, 1));
///
/// // A context-dependent one.
/// assert!(supports_function(&ExpandedName::new(Some(FN_NAMESPACE), "position"), 0));
///
/// // `fn:abs` is an ordinary library entry, so it is not served here.
/// assert!(!supports_function(&ExpandedName::new(Some(FN_NAMESPACE), "abs"), 1));
/// ```
#[must_use]
pub fn supports_function(name: &ExpandedName, arity: usize) -> bool {
    engine_supports(name, arity)
}

/// Whether this crate implements the function itself, rather than delegating to the library.
fn engine_supports(name: &ExpandedName, arity: usize) -> bool {
    match name.namespace() {
        Some(MATH_NAMESPACE) => math::supports(name.local_name(), arity),
        Some(MAP_NAMESPACE) => matches!((name.local_name(), arity), ("for-each", 2)),
        Some(ARRAY_NAMESPACE) => matches!(
            (name.local_name(), arity),
            ("for-each", 2)
                | ("filter", 2)
                | ("fold-left", 3)
                | ("fold-right", 3)
                | ("for-each-pair", 3)
                | ("sort", 1..=3)
        ),
        Some(FN_NAMESPACE) => fn_supports(name.local_name(), arity),
        _ => false,
    }
}

fn fn_supports(local: &str, arity: usize) -> bool {
    if local == "concat" {
        return arity >= 2;
    }
    if sequences::supports(local, arity) {
        return true;
    }
    if json::supports(local, arity) {
        return true;
    }
    if random::supports(local, arity) {
        return true;
    }
    matches!(
        (local, arity),
        ("position", 0)
            | ("last", 0)
            | ("string", 0)
            | ("data", 0)
            | ("number", 0)
            | ("string-length", 0)
            | ("normalize-space", 0)
            | ("name", 0)
            | ("local-name", 0)
            | ("namespace-uri", 0)
            | ("root", 0)
            | ("base-uri", 0)
            | ("document-uri", 0)
            | ("node-name", 0)
            | ("nilled", 0)
            | ("has-children", 0)
            | ("path", 0)
            | ("generate-id", 0)
            | ("lang", 1)
            | ("id", 1)
            | ("idref", 1)
            | ("element-with-id", 1)
            | ("for-each", 2)
            | ("filter", 2)
            | ("fold-left", 3)
            | ("fold-right", 3)
            | ("for-each-pair", 3)
            | ("sort", 1..=3)
            | ("apply", 2)
            | ("function-lookup", 2)
            | ("function-name", 1)
            | ("function-arity", 1)
            | ("doc", 1)
            | ("doc-available", 1)
            | ("collection", 0 | 1)
            | ("unparsed-text", 1 | 2)
            | ("unparsed-text-lines", 1 | 2)
            | ("unparsed-text-available", 1 | 2)
            | ("json-doc", 1 | 2)
            | ("format-number", 2 | 3)
            | ("uri-collection", 0 | 1)
            | ("environment-variable", 1)
            | ("available-environment-variables", 0)
    )
}

fn call_engine(
    context: &DynamicContext,
    name: &ExpandedName,
    arguments: &[XpathValue],
) -> XPathResult<Option<XpathValue>> {
    let arity = arguments.len();
    match name.namespace() {
        Some(MATH_NAMESPACE) if math::supports(name.local_name(), arity) => {
            math::call(name.local_name(), arguments).map(Some)
        }
        Some(MAP_NAMESPACE | ARRAY_NAMESPACE) if engine_supports(name, arity) => {
            higher_order::call_collection(context, name, arguments).map(Some)
        }
        Some(FN_NAMESPACE) if fn_supports(name.local_name(), arity) => {
            call_fn(context, name.local_name(), arguments).map(Some)
        }
        _ => Ok(None),
    }
}

fn call_fn(
    context: &DynamicContext,
    local: &str,
    arguments: &[XpathValue],
) -> XPathResult<XpathValue> {
    if local == "concat" {
        return concat(arguments);
    }
    if sequences::supports(local, arguments.len()) {
        return sequences::call(local, arguments);
    }
    if random::supports(local, arguments.len()) {
        return random::call(context, arguments);
    }
    if json::supports(local, arguments.len()) {
        return json::call(context, local, arguments);
    }
    if local == "format-number" {
        return format_number(context, arguments);
    }
    match local {
        "position" => Ok(integer(
            i64::try_from(context.position()?).unwrap_or(i64::MAX),
        )),
        "last" => Ok(integer(i64::try_from(context.size()?).unwrap_or(i64::MAX))),
        "string" => Ok(string(context.context_item()?.string_value()?)),
        "data" => Ok(XpathValue::from_iter(context.context_item()?.atomize()?)),
        "number" => {
            let item = context.context_item()?;
            let number = crate::eval::number_of(&XpathValue::from(item.clone()), context)?;
            Ok(XpathValue::from(AtomicValue::double(number)))
        }
        "string-length" => Ok(integer(count_characters(
            &context.context_item()?.string_value()?,
        ))),
        "normalize-space" => Ok(string(oxixml_fn::strings::normalize_space(
            &context.context_item()?.string_value()?,
        ))),
        "name" => Ok(string(nodes::name(context.context_node_argument()?))),
        "local-name" => Ok(string(nodes::local_name(context.context_node_argument()?))),
        "namespace-uri" => Ok(XpathValue::from(AtomicValue::any_uri(
            nodes::namespace_uri(context.context_node_argument()?),
        ))),
        "root" => Ok(XpathValue::from(context.context_node_argument()?.root())),
        "base-uri" => Ok(optional(oxixml_fn::accessors::base_uri(
            context.context_node_argument()?,
        ))),
        "document-uri" => Ok(optional(oxixml_fn::accessors::document_uri(
            context.context_node_argument()?,
        ))),
        "node-name" => Ok(optional(
            oxixml_fn::accessors::node_name(context.context_node_argument()?)
                .map(oxixml_fn::qnames::qname_value),
        )),
        "nilled" => Ok(optional(
            oxixml_fn::accessors::nilled(context.context_node_argument()?)
                .map(AtomicValue::boolean),
        )),
        "has-children" => Ok(XpathValue::from(AtomicValue::boolean(nodes::has_children(
            context.context_node_argument()?,
        )))),
        "path" => Ok(string(nodes::path(context.context_node_argument()?))),
        "generate-id" => Ok(string(nodes::generate_id(context.context_node_argument()?))),
        "lang" => {
            let test = optional_string(&arguments[0])?;
            Ok(XpathValue::from(AtomicValue::boolean(nodes::lang(
                &test,
                context.context_node_argument()?,
            ))))
        }
        "id" => id_family(context, &arguments[0], IdKind::Id),
        "element-with-id" => id_family(context, &arguments[0], IdKind::ElementWithId),
        "idref" => id_family(context, &arguments[0], IdKind::Idref),
        "doc"
        | "doc-available"
        | "collection"
        | "uri-collection"
        | "unparsed-text"
        | "unparsed-text-lines"
        | "unparsed-text-available"
        | "json-doc" => resources::call(context, local, arguments),
        // The set of environment variables is implementation-dependent, and this engine's is
        // empty: a query has no business reading the host process's environment.
        "environment-variable" | "available-environment-variables" => Ok(XpathValue::empty()),
        _ => higher_order::call_fn(context, local, arguments),
    }
}

enum IdKind {
    Id,
    ElementWithId,
    Idref,
}

fn id_family(
    context: &DynamicContext,
    argument: &XpathValue,
    kind: IdKind,
) -> XPathResult<XpathValue> {
    let node = context.context_node_argument()?;
    let values = strings_of(argument)?;
    let found = match kind {
        IdKind::Id => nodes::id(&values, node)?,
        IdKind::ElementWithId => nodes::element_with_id(&values, node)?,
        IdKind::Idref => nodes::idref(&values, node)?,
    };
    Ok(XpathValue::from_iter(found))
}

fn concat(arguments: &[XpathValue]) -> XPathResult<XpathValue> {
    let mut out = String::new();
    for argument in arguments {
        if let Some(value) = argument.atomize_at_most_one("an argument of fn:concat")? {
            out.push_str(&value.string_value());
        }
    }
    Ok(string(out))
}

// --- small conversion helpers ------------------------------------------------------------

pub(crate) fn integer(value: i64) -> XpathValue {
    XpathValue::from(AtomicValue::integer(value))
}

pub(crate) fn string(value: impl Into<String>) -> XpathValue {
    XpathValue::from(AtomicValue::string(value))
}

pub(crate) fn optional(value: Option<AtomicValue>) -> XpathValue {
    value.map(XpathValue::from).unwrap_or_default()
}

pub(crate) fn count_characters(value: &str) -> i64 {
    i64::try_from(value.chars().count()).unwrap_or(i64::MAX)
}

pub(crate) fn optional_string(value: &XpathValue) -> XPathResult<String> {
    Ok(value
        .atomize_at_most_one("a string argument")?
        .map(|value| value.string_value())
        .unwrap_or_default())
}

pub(crate) fn strings_of(value: &XpathValue) -> XPathResult<Vec<String>> {
    Ok(value
        .atomize()?
        .into_iter()
        .map(|value| value.string_value())
        .collect())
}

/// [`fn:format-number`](https://www.w3.org/TR/xpath-functions-31/#func-format-number).
///
/// Two of its rules need the *static context*, which the library form of the function does not
/// have:
///
/// * `$decimal-format-name` is an **EQName**, so a lexical `QName` in it is resolved against the
///   statically known namespaces before the format is looked up (`numberformat20`, `-81`,
///   `-83` — `a:test` and `b:test` name the same format when both prefixes are bound to the
///   same namespace, and the name may be surrounded by whitespace);
/// * the scientific-notation sub-picture was added in F&O **3.1**, so an expression compiled at
///   3.0 or earlier must reject it with `FODF1310` (`numberformat128`).
fn format_number(context: &DynamicContext, arguments: &[XpathValue]) -> XPathResult<XpathValue> {
    let value = arguments[0].clone().atomize_at_most_one("$value")?;
    let picture = match arguments[1].clone().atomize_at_most_one("$picture")? {
        Some(picture) => picture.string_value(),
        None => return Err(XPathError::type_error("the $picture must be an xs:string")),
    };
    let fn_context = context.fn_context();
    let format = match arguments.get(2) {
        None => fn_context.default_decimal_format().clone(),
        Some(argument) => match argument
            .clone()
            .atomize_at_most_one("$decimal-format-name")?
        {
            None => fn_context.default_decimal_format().clone(),
            Some(name) => {
                let resolved = decimal_format_name(context.static_context(), &name.string_value())?;
                fn_context.decimal_format(Some(&resolved))?.clone()
            }
        },
    };
    if !context.version().at_least(XPathVersion::XPath31)
        && oxixml_fn::numeric::picture_has_exponent(&picture, &format)?
    {
        return Err(oxixml_fn::FnError::new(
            oxixml_fn::FnErrorCode::Fodf1310,
            "the exponent part of a picture string needs XPath 3.1",
        )
        .into());
    }
    Ok(string(oxixml_fn::numeric::format_number(
        value.as_ref(),
        &picture,
        &format,
    )?))
}

/// Reads `fn:format-number`'s `$decimal-format-name`, which F&O 3.1 §4.7.4 defines as an EQName
/// with optional surrounding whitespace.
fn decimal_format_name(context: &StaticContext, lexical: &str) -> XPathResult<ExpandedName> {
    let trimmed = lexical.trim();
    if let Some(rest) = trimmed.strip_prefix("Q{") {
        if let Some((namespace, local)) = rest.split_once('}') {
            return Ok(ExpandedName::new(
                (!namespace.is_empty()).then_some(namespace),
                local.trim_start(),
            ));
        }
    }
    let Some((prefix, local)) = trimmed.split_once(':') else {
        return Ok(ExpandedName::new(None::<&str>, trimmed));
    };
    match context.namespace(prefix) {
        Some(namespace) => Ok(ExpandedName::new(Some(namespace), local)),
        None => Err(oxixml_fn::FnError::new(
            oxixml_fn::FnErrorCode::Fodf1280,
            format!("the prefix `{prefix}` of a decimal-format name is not bound"),
        )
        .into()),
    }
}

pub(crate) fn require_function(value: &XpathValue, what: &str) -> XPathResult<XpathFunction> {
    if value.is_singleton() {
        if let Some(function) = value.first().and_then(XpathItem::as_callable) {
            return Ok(function);
        }
    }
    Err(XPathError::type_error(format!(
        "{what} must be exactly one function item"
    )))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::context::StaticContext;
    use oxixml_infoset::Tree;
    use std::sync::Arc;

    fn context() -> DynamicContext {
        DynamicContext::new(Arc::new(StaticContext::new()))
    }

    #[test]
    fn the_catalogue_covers_library_engine_and_constructor_functions() {
        let statik = StaticContext::new();
        let latest = XPathVersion::XPath31;
        let known = |local: &str, namespace: &str, arity: usize| {
            is_known(
                &statik,
                &ExpandedName::new(Some(namespace), local),
                arity,
                latest,
            )
        };
        // From the library.
        assert!(known("abs", FN_NAMESPACE, 1));
        // From this crate.
        assert!(known("position", FN_NAMESPACE, 0));
        assert!(known("pi", MATH_NAMESPACE, 0));
        // A constructor.
        assert!(known("integer", XSD_NAMESPACE, 1));
        // Abstract types have no constructor.
        assert!(!known("anyAtomicType", XSD_NAMESPACE, 1));
        assert!(!known("nope", FN_NAMESPACE, 0));
        // And the catalogue is the version's, not the newest one's.
        assert!(!is_known(
            &statik,
            &ExpandedName::new(Some(FN_NAMESPACE), "sort"),
            1,
            XPathVersion::XPath30
        ));
        assert!(!is_known(
            &statik,
            &ExpandedName::new(Some(FN_NAMESPACE), "format-number"),
            2,
            XPathVersion::XPath20
        ));
    }

    #[test]
    fn concat_accepts_any_arity() -> XPathResult<()> {
        let arguments: Vec<XpathValue> = (0..12).map(|index| string(index.to_string())).collect();
        assert_eq!(concat(&arguments)?.to_string(), "01234567891011");
        Ok(())
    }

    #[test]
    fn context_dependent_functions_read_the_focus() -> XPathResult<()> {
        let tree = Tree::parse_str("<a><b/></a>")?;
        let root = tree.document_element().expect("a document element");
        let context = context().with_context_item(root);
        let name = ExpandedName::new(Some(FN_NAMESPACE), "local-name");
        assert_eq!(call(&context, &name, &[])?.to_string(), "a");
        let position = ExpandedName::new(Some(FN_NAMESPACE), "position");
        assert_eq!(call(&context, &position, &[])?.to_string(), "1");
        Ok(())
    }

    /// `fn:id` reads the focus, `fn:idref` reads it too — and the two-argument form of either
    /// searches the tree of the node it is *given*, which need not be the one the focus is in
    /// (xslt30 `id-043`: `idref(('a', .), doc('id302.xml'))` while the context item is a string
    /// inside another document altogether). Whether a candidate is tokenized is the difference
    /// between the two functions, and both arities answer the same way.
    #[test]
    fn the_id_functions_search_the_tree_they_are_pointed_at() -> XPathResult<()> {
        let referring = Tree::parse_str(
            "<!DOCTYPE t [<!ATTLIST a id ID #REQUIRED><!ATTLIST a ref IDREFS #IMPLIED>]>\
             <t><a id='a'/><a id='b' ref='c d'/><a id='c' ref='d a'/></t>",
        )?;
        let elsewhere = Tree::parse_str("<root/>")?;
        let idref = ExpandedName::new(Some(FN_NAMESPACE), "idref");
        let id = ExpandedName::new(Some(FN_NAMESPACE), "id");

        // One argument: the focus names the tree.
        let here = context().with_context_item(referring.root());
        assert_eq!(call(&here, &idref, &[string("a")])?.len(), 1);
        assert_eq!(call(&here, &id, &[string("a b")])?.len(), 2);
        // `fn:idref` does not tokenize, so the same two identifiers as one string match nothing.
        assert!(call(&here, &idref, &[string("a b")])?.is_empty());

        // Two arguments: the second names it, and the focus is left out of it.
        let outside = context().with_context_item(elsewhere.root());
        let target = XpathValue::from(referring.root());
        assert_eq!(
            call(&outside, &idref, &[string("a"), target.clone()])?.len(),
            1
        );
        assert_eq!(call(&outside, &id, &[string("a b"), target])?.len(), 2);
        assert!(
            call(
                &outside,
                &idref,
                &[string("a"), XpathValue::from(elsewhere.root())]
            )?
            .is_empty()
        );
        Ok(())
    }

    #[test]
    fn constructors_cast() -> XPathResult<()> {
        let context = context();
        let name = ExpandedName::new(Some(XSD_NAMESPACE), "integer");
        assert_eq!(call(&context, &name, &[string("42")])?.to_string(), "42");
        assert!(call(&context, &name, &[string("x")]).is_err());
        assert!(call(&context, &name, &[XpathValue::empty()])?.is_empty());
        Ok(())
    }

    #[test]
    fn the_error_constructor_admits_nothing_but_the_empty_sequence() -> XPathResult<()> {
        let context = context();
        let name = ExpandedName::new(Some(XSD_NAMESPACE), "error");
        // F&O 3.1 §18.4: `xs:error(())` is the empty sequence, and every other argument fails —
        // the type has no member types and therefore an empty value space.
        assert!(call(&context, &name, &[XpathValue::empty()])?.is_empty());
        for argument in [string(""), string("boo"), integer(1)] {
            assert_eq!(
                call(&context, &name, &[argument])
                    .expect_err("xs:error rejects every non-empty argument")
                    .code(),
                Some(XPathErrorCode::Forg0001)
            );
        }
        // The arity is exactly one: there is no `xs:error()` and no `xs:error#0`.
        let statik = StaticContext::new();
        assert!(is_known(&statik, &name, 1, XPathVersion::XPath31));
        for arity in [0, 2] {
            assert!(!is_known(&statik, &name, arity, XPathVersion::XPath31));
        }
        // And it is not an atomic type, so it stays out of the cast machinery entirely.
        assert!(oxixml_infoset::AtomicType::from_local_name("error").is_none());
        assert!(constructor_type(&name, 1).is_none());
        assert!(list_constructor_type(&name, 1).is_none());
        Ok(())
    }
}
