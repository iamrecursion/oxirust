//! Arithmetic, the range operator, the sequence operators and string concatenation.
//!
//! # Arithmetic, in two dialects
//!
//! XPath 1.0 has one numeric type. `+ - * div mod` put *both* operands through `number()` and
//! compute in IEEE doubles, so `"3" + 4` is `7`, `@n * 2` reads the attribute's text as a
//! number, and division by zero yields `INF`, `-INF` or `NaN` rather than an error.
//!
//! XPath 2.0 and later atomize instead: `xs:untypedAtomic` is promoted to `xs:double` (which is
//! what keeps `@n * 2` working on unvalidated input), the numeric tower promotes
//! `xs:integer` → `xs:decimal` → `xs:float` → `xs:double`, `xs:string` is a type error, an empty
//! operand yields the empty sequence, and `xs:decimal`/`xs:integer` division by zero raises
//! `FOAR0001`. Dates, times and durations take part in arithmetic here too, through
//! [`oxixml_fn::numeric`].
//!
//! # Sequence operators
//!
//! `union`, `intersect` and `except` require node sequences and produce node sequences in
//! document order with duplicates removed. `to` builds an ascending run of `xs:integer`s, empty
//! when the first bound exceeds the second.

use super::{atomize_for_arithmetic, evaluate, require_nodes};
use crate::ast::{BinaryOp, Expr};
use crate::context::DynamicContext;
use crate::error::{XPathError, XPathErrorCode, XPathResult};
use crate::value::XpathValue;
use oxixml_fn::FN_NAMESPACE;
use oxixml_fn::numeric::{ArithOp, arithmetic, unary_minus};
use oxixml_infoset::{AtomicValue, BigInteger, NodeRef};
use std::collections::BTreeSet;

/// The largest sequence the range operator will materialize.
///
/// XPath does not bound `to`, but a `1 to 1000000000` in a hostile or careless expression must
/// not take the process down with it. Exceeding the limit raises `XPDY0130`, the code the
/// specification reserves for an implementation-defined limit.
///
/// The figure is an item count, and a materialized item costs a little under two hundred bytes,
/// so the ceiling caps a single range at a few hundred megabytes. It has to clear the whole
/// Unicode code space with room to spare — `codepoints-to-string(65536 to 1114112)` is 1 048 577
/// items and has to reach `FOCH0001` on the last one rather than be refused — which is what fixes
/// it at two mebi-items rather than lower. Ranges that are only *counted*, only tested for
/// membership, or only indexed never come here at all: `count(1 to 2000000000)`,
/// `2 = (1 to 2000000000)` and `(1 to 2000000000)[3]` are each answered from the two bounds.
pub const MAX_RANGE: i64 = 2_097_152;

/// Evaluates a binary operator.
///
/// Every binary operator is left-associative, so `a + b + c` is `Binary(Binary(a, b), c)` and a
/// chain of them leans left however long it grows. The parser builds such a chain with a loop
/// and [`Expr`]'s own `Drop` frees it with one, and this evaluates it with one too: the left
/// spine is flattened onto the heap first and the results folded forward, so `1+1+…+1` costs the
/// same stack at a hundred thousand terms as it does at two. Recursing once per operator instead
/// would exhaust the thread's stack — a process abort — on an expression the grammar does not
/// bound.
///
/// # Errors
///
/// `XPTY0004` for a wrongly typed operand, `FOAR0001` for an exact division by zero.
pub(crate) fn binary(
    op: BinaryOp,
    left: &Expr,
    right: &Expr,
    context: &DynamicContext,
) -> XPathResult<XpathValue> {
    // The chain is collected outermost first, so folding it runs from the innermost operand
    // outwards — which is the order left-associativity prescribes.
    let mut chain: Vec<(BinaryOp, &Expr)> = vec![(op, right)];
    let mut base = left;
    while let Expr::Binary { op, left, right } = base {
        chain.push((*op, right));
        base = left;
    }
    let mut value = evaluate(base, context)?;
    for (op, right) in chain.into_iter().rev() {
        value = apply(op, value, right, context)?;
    }
    Ok(value)
}

/// Applies one operator to an already evaluated left operand.
fn apply(
    op: BinaryOp,
    left: XpathValue,
    right: &Expr,
    context: &DynamicContext,
) -> XPathResult<XpathValue> {
    match op {
        BinaryOp::Union | BinaryOp::Intersect | BinaryOp::Except => {
            set_operation(op, left, right, context)
        }
        BinaryOp::Range => range(left, right, context),
        BinaryOp::Concat => {
            let left = concat_operand(&left)?;
            let right = concat_operand(&evaluate(right, context)?)?;
            Ok(XpathValue::from(AtomicValue::string(format!(
                "{left}{right}"
            ))))
        }
        _ => arithmetic_operation(op, left, right, context),
    }
}

fn arithmetic_operation(
    op: BinaryOp,
    left: XpathValue,
    right: &Expr,
    context: &DynamicContext,
) -> XPathResult<XpathValue> {
    let right = evaluate(right, context)?;
    if context.version().is_xpath_10() {
        let a = super::compare::number_10(&left);
        let b = super::compare::number_10(&right);
        return Ok(XpathValue::from(AtomicValue::double(arithmetic_10(
            op, a, b,
        ))));
    }
    let Some(a) = single_operand(&left, "the left operand of an arithmetic operator")? else {
        return Ok(XpathValue::empty());
    };
    let Some(b) = single_operand(&right, "the right operand of an arithmetic operator")? else {
        return Ok(XpathValue::empty());
    };
    let operation = match op {
        BinaryOp::Add => ArithOp::Add,
        BinaryOp::Subtract => ArithOp::Subtract,
        BinaryOp::Multiply => ArithOp::Multiply,
        BinaryOp::Divide => ArithOp::Divide,
        BinaryOp::IntegerDivide => ArithOp::IntegerDivide,
        BinaryOp::Modulus => ArithOp::Modulus,
        other => {
            return Err(XPathError::type_error(format!(
                "`{}` is not an arithmetic operator",
                other.as_str()
            )));
        }
    };
    Ok(XpathValue::from(arithmetic(operation, &a, &b)?))
}

/// XPath 1.0 arithmetic: IEEE doubles throughout, with no error cases at all.
fn arithmetic_10(op: BinaryOp, left: f64, right: f64) -> f64 {
    match op {
        BinaryOp::Add => left + right,
        BinaryOp::Subtract => left - right,
        BinaryOp::Multiply => left * right,
        BinaryOp::Divide => left / right,
        BinaryOp::Modulus => left % right,
        _ => f64::NAN,
    }
}

fn single_operand(value: &XpathValue, what: &str) -> XPathResult<Option<AtomicValue>> {
    let atomized = atomize_for_arithmetic(value)?;
    match atomized.len() {
        0 => Ok(None),
        1 => Ok(atomized.into_iter().next()),
        other => Err(XPathError::type_error(format!(
            "{what} must be at most one value but atomizes to {other}"
        ))),
    }
}

/// Unary `-` and `+`.
///
/// # Errors
///
/// `XPTY0004` when the operand atomizes to more than one value or to a non-numeric value.
pub(crate) fn unary(
    negate: bool,
    operand: &XpathValue,
    context: &DynamicContext,
) -> XPathResult<XpathValue> {
    if context.version().is_xpath_10() {
        let number = super::compare::number_10(operand);
        return Ok(XpathValue::from(AtomicValue::double(if negate {
            -number
        } else {
            number
        })));
    }
    let Some(value) = single_operand(operand, "the operand of a unary sign")? else {
        return Ok(XpathValue::empty());
    };
    if !negate {
        return Ok(XpathValue::from(oxixml_fn::numeric::unary_plus(&value)?));
    }
    Ok(XpathValue::from(unary_minus(&value)?))
}

fn range(left: XpathValue, right: &Expr, context: &DynamicContext) -> XPathResult<XpathValue> {
    let Some(from) = bound(&left, "the left operand of `to`")? else {
        return Ok(XpathValue::empty());
    };
    let Some(to) = bound(&evaluate(right, context)?, "the right operand of `to`")? else {
        return Ok(XpathValue::empty());
    };
    match (from, to) {
        // Two `i64` endpoints take the fixed-width path unchanged.
        (RangeBound::Fixed(from), RangeBound::Fixed(to)) => range_i64(from, to),
        // An endpoint past the `i64` range makes the whole run arbitrary-precision. The span
        // (hence the item count) is still bounded by `MAX_RANGE`, so this only ever builds a few
        // big integers — `1000000000000000000000 to …003` is four items.
        (from, to) => range_big(from.into_big(), to.into_big()),
    }
}

/// Materializes an ascending run of fixed-width `xs:integer`s.
fn range_i64(from: i64, to: i64) -> XPathResult<XpathValue> {
    if from > to {
        return Ok(XpathValue::empty());
    }
    let count = to
        .checked_sub(from)
        .and_then(|span| span.checked_add(1))
        .ok_or_else(|| XPathError::type_error("the range operator overflows"))?;
    if count > MAX_RANGE {
        return Err(XPathError::new(
            XPathErrorCode::Xpdy0130,
            format!(
                "`{from} to {to}` would materialize {count} items, beyond this implementation's limit of {MAX_RANGE}"
            ),
        ));
    }
    let mut out = Vec::with_capacity(usize::try_from(count.min(4096)).unwrap_or(0));
    let mut current = from;
    loop {
        out.push(AtomicValue::integer(current));
        if current == to {
            break;
        }
        current += 1;
    }
    Ok(XpathValue::from_iter(out))
}

/// Materializes an ascending run of `xs:integer`s at arbitrary precision, demoting each value back
/// to the fixed width when it fits. Used only when an endpoint overflows an `i64`.
fn range_big(from: BigInteger, to: BigInteger) -> XPathResult<XpathValue> {
    if from > to {
        return Ok(XpathValue::empty());
    }
    let one = BigInteger::from_i64(1);
    let span = to
        .checked_sub(&from)
        .and_then(|span| span.checked_add(&one))
        .ok_or_else(|| XPathError::type_error("the range operator overflows"))?;
    let count = span.to_i64().filter(|count| *count <= MAX_RANGE);
    let Some(count) = count else {
        return Err(XPathError::new(
            XPathErrorCode::Xpdy0130,
            format!(
                "`{from} to {to}` would materialize more than {MAX_RANGE} items, beyond this implementation's limit"
            ),
        ));
    };
    let mut out = Vec::with_capacity(usize::try_from(count.min(4096)).unwrap_or(0));
    let mut current = from;
    loop {
        out.push(AtomicValue::from_big_integer(current.clone()));
        if current == to {
            break;
        }
        current = current
            .checked_add(&one)
            .ok_or_else(|| XPathError::type_error("the range operator overflows"))?;
    }
    Ok(XpathValue::from_iter(out))
}

/// A range expression recognized *without* building every integer it contains.
///
/// [XPath 3.1 §3.4](https://www.w3.org/TR/xpath-31/#construct_seq) puts no ceiling on `to`, so a
/// query that only needs the count of `1 to 1000000000`, or its first item, must not pay to
/// materialize a billion `xs:integer`s (and trip [`MAX_RANGE`]). Recognizing the shape lets
/// `fn:count`, `fn:exists`, `fn:empty` and a constant positional filter answer in constant time,
/// while every other use of a range still goes through [`range`] and its bound.
pub(crate) enum RangeForm {
    /// Not a range expression; evaluate it the ordinary way.
    NotARange,
    /// A range expression whose value is the empty sequence (`() to 5`, or `5 to 1`).
    Empty,
    /// `from to to`, ascending and non-empty, optionally reversed by an enclosing `fn:reverse`.
    Bounded {
        /// The first integer of the ascending run.
        from: i64,
        /// The last integer of the ascending run.
        to: i64,
        /// Whether an `fn:reverse` wraps the range, so its items run `to` down to `from`.
        reversed: bool,
    },
    /// The same, ascending and non-empty, with at least one endpoint beyond the `i64` range.
    ///
    /// Only the consumers that never build an item read this — `fn:count`, `fn:exists`,
    /// `fn:empty` and the general comparisons, all of which answer from the two bounds and none
    /// of which cares in which direction the run is read. The positional fast path declines it
    /// and lets the ordinary arbitrary-precision materialization run, which is what keeps
    /// `(1000000000000000000000 to …003)[2]` answering its four-item range.
    BoundedBig {
        /// The first integer of the ascending run.
        from: BigInteger,
        /// The last integer of the ascending run.
        to: BigInteger,
    },
}

impl RangeForm {
    /// The number of items the range yields.
    ///
    /// The count is arbitrary-precision because the range may be: `0 to 9223372036854775807`
    /// holds 2<sup>63</sup> items, one more than an `i64` can express, and `fn:count` returns an
    /// `xs:integer`, which has no such bound.
    fn item_count(&self) -> BigInteger {
        match self.bounds() {
            None => BigInteger::zero(),
            Some((from, to)) => range_length(&from, &to),
        }
    }

    /// The two endpoints of the ascending run, or [`None`] when there is no run.
    fn bounds(&self) -> Option<(BigInteger, BigInteger)> {
        match self {
            Self::NotARange | Self::Empty => None,
            Self::Bounded { from, to, .. } => {
                Some((BigInteger::from_i64(*from), BigInteger::from_i64(*to)))
            }
            Self::BoundedBig { from, to } => Some((from.clone(), to.clone())),
        }
    }

    /// The `position`-th item (1-based) of the range, or [`None`] when `position` is out of range.
    fn item_at(&self, position: i64) -> Option<i64> {
        let Self::Bounded { from, to, reversed } = self else {
            return None;
        };
        if position < 1 {
            return None;
        }
        // `from <= to` holds for `Bounded`, so the length is at least one and never overflows.
        let length = to.checked_sub(*from)?.checked_add(1)?;
        if position > length {
            return None;
        }
        let offset = position - 1;
        Some(if *reversed {
            to - offset
        } else {
            from + offset
        })
    }
}

/// `to - from + 1`, the item count of the ascending range `from` … `to`.
///
/// The arithmetic is exact at arbitrary precision, so it cannot overflow: both operations are
/// defined for every pair of [`BigInteger`]s the caller can hold, and a failure of the
/// underlying magnitude arithmetic would mean the endpoints themselves could not exist.
fn range_length(from: &BigInteger, to: &BigInteger) -> BigInteger {
    let one = BigInteger::from_i64(1);
    to.checked_sub(from)
        .and_then(|span| span.checked_add(&one))
        .unwrap_or_else(BigInteger::zero)
}

/// Recognizes `E1 to E2`, and `fn:reverse(E1 to E2)`, evaluating only the two bound expressions.
///
/// A parenthesized single expression is transparent to the parser, so `(1 to 5)` already arrives
/// as the bare range; a one-item comma-less sequence is peeled here for good measure.
///
/// # Errors
///
/// Whatever evaluating a bound raises — `XPTY0004` for a bound that is not a single `xs:integer`.
pub(crate) fn classify_range(expr: &Expr, context: &DynamicContext) -> XPathResult<RangeForm> {
    match peel_singleton(expr) {
        Expr::Binary {
            op: BinaryOp::Range,
            left,
            right,
        } => {
            let Some(from) = bound(&evaluate(left, context)?, "the left operand of `to`")? else {
                return Ok(RangeForm::Empty);
            };
            let Some(to) = bound(&evaluate(right, context)?, "the right operand of `to`")? else {
                return Ok(RangeForm::Empty);
            };
            // A range with an endpoint past `i64` keeps its bounds at arbitrary precision: the
            // count and the general comparisons answer from them, and the positional fast path
            // declines the shape and lets [`range`] materialize the (necessarily short) run.
            match (from, to) {
                (RangeBound::Fixed(from), RangeBound::Fixed(to)) if from > to => {
                    Ok(RangeForm::Empty)
                }
                (RangeBound::Fixed(from), RangeBound::Fixed(to)) => Ok(RangeForm::Bounded {
                    from,
                    to,
                    reversed: false,
                }),
                (from, to) => {
                    let (from, to) = (from.into_big(), to.into_big());
                    if from > to {
                        return Ok(RangeForm::Empty);
                    }
                    Ok(RangeForm::BoundedBig { from, to })
                }
            }
        }
        Expr::FunctionCall { name, arguments }
            if arguments.len() == 1
                && name.namespace() == Some(FN_NAMESPACE)
                && name.local_name() == "reverse" =>
        {
            match classify_range(&arguments[0], context)? {
                RangeForm::Bounded { from, to, reversed } => Ok(RangeForm::Bounded {
                    from,
                    to,
                    reversed: !reversed,
                }),
                // Reading the run backwards changes neither its size nor its membership, which
                // is all the arbitrary-precision form is ever asked for.
                other => Ok(other),
            }
        }
        _ => Ok(RangeForm::NotARange),
    }
}

/// `fn:count`, `fn:exists` and `fn:empty` answered without building the sequence they are given,
/// or [`None`] when `local` is a different function or the argument's size cannot be known
/// without building it.
///
/// # Errors
///
/// Whatever evaluating a bound, or an operand of a comma sequence, raises.
pub(crate) fn range_aggregate(
    local: &str,
    argument: &Expr,
    context: &DynamicContext,
) -> XPathResult<Option<XpathValue>> {
    if !matches!(local, "count" | "exists" | "empty") {
        return Ok(None);
    }
    let Some(count) = sequence_size(argument, context)? else {
        return Ok(None);
    };
    let value = match local {
        "count" => AtomicValue::from_big_integer(count),
        "exists" => AtomicValue::boolean(!count.is_zero()),
        _ => AtomicValue::boolean(count.is_zero()),
    };
    Ok(Some(XpathValue::from(value)))
}

/// How many items `expr` yields, when that can be established without holding them all.
///
/// A bare range answers from its bounds. So does a comma sequence *containing* one, which is what
/// `count(((), local:f(), (1 to 10000000), local:g()))` needs: its operands are evaluated in the
/// same order and with the same errors as [`super::evaluate`] would evaluate the sequence itself,
/// but only their sizes are kept.
///
/// # Errors
///
/// Whatever evaluating a bound, or an operand of the sequence, raises.
fn sequence_size(expr: &Expr, context: &DynamicContext) -> XPathResult<Option<BigInteger>> {
    match classify_range(expr, context)? {
        RangeForm::NotARange => {}
        form => return Ok(Some(form.item_count())),
    }
    let Expr::Sequence(operands) = expr else {
        return Ok(None);
    };
    let mut total = BigInteger::zero();
    for operand in operands {
        let size = match classify_range(operand, context)? {
            RangeForm::NotARange => {
                let evaluated = evaluate(operand, context)?;
                BigInteger::from_i64(i64::try_from(evaluated.len()).unwrap_or(i64::MAX))
            }
            form => form.item_count(),
        };
        total = total
            .checked_add(&size)
            .ok_or_else(|| XPathError::type_error("the sequence is too long to count"))?;
    }
    Ok(Some(total))
}

/// The two endpoints of the ascending run `expr` denotes, when `expr` is a range expression.
///
/// [`RangeForm::Empty`] answers `Some(None)`: the expression *is* a range, and its value is the
/// empty sequence. [`None`] means it is not a range at all.
///
/// # Errors
///
/// Whatever evaluating a bound raises — `XPTY0004` for a bound that is not a single `xs:integer`.
pub(crate) fn range_bounds(
    expr: &Expr,
    context: &DynamicContext,
) -> XPathResult<Option<Option<(BigInteger, BigInteger)>>> {
    match classify_range(expr, context)? {
        RangeForm::NotARange => Ok(None),
        form => Ok(Some(form.bounds())),
    }
}

/// The single item (1-based `position`) of `base` when `base` is a range and `position` is a
/// constant number, so `(1 to 1000000000)[1]` need not build the range. [`None`] means the fast
/// path does not apply and the ordinary filter must run.
///
/// # Errors
///
/// Whatever [`classify_range`] raises for a malformed bound.
pub(crate) fn range_at_position(
    base: &Expr,
    predicate: &Expr,
    context: &DynamicContext,
) -> XPathResult<Option<XpathValue>> {
    let Some(position) = numeric_literal(predicate) else {
        return Ok(None);
    };
    let form = classify_range(base, context)?;
    // An arbitrary-precision range declines the fast path: its items are `xs:integer`s that no
    // `i64` holds, and the run is short enough to build (`RangeExpr-409b`).
    if matches!(form, RangeForm::NotARange | RangeForm::BoundedBig { .. }) {
        return Ok(None);
    }
    // A positional predicate keeps the item whose 1-based position equals the number; a
    // fractional number matches no position, so the result is empty.
    let selected = whole_number(position).and_then(|position| form.item_at(position));
    Ok(Some(selected.map_or_else(XpathValue::empty, |value| {
        XpathValue::from(AtomicValue::integer(value))
    })))
}

/// A one-item comma-less sequence unwrapped to its element; every other expression unchanged.
fn peel_singleton(expr: &Expr) -> &Expr {
    if let Expr::Sequence(items) = expr {
        if let [only] = items.as_slice() {
            return peel_singleton(only);
        }
    }
    expr
}

/// The value of a numeric literal predicate, or [`None`] for any other predicate.
fn numeric_literal(predicate: &Expr) -> Option<f64> {
    match peel_singleton(predicate) {
        Expr::Literal(atomic) if atomic.is_numeric() => atomic.as_f64(),
        _ => None,
    }
}

/// `value` as an `i64` when it is a whole number, else [`None`].
///
/// A position with a fractional part matches no item and is rejected here; a whole number beyond
/// `i64` saturates on the cast to `i64::MIN`/`i64::MAX`, which [`RangeForm::item_at`] then rejects
/// as out of range — so the answer is the empty sequence either way.
fn whole_number(value: f64) -> Option<i64> {
    if value.is_finite() && value.fract() == 0.0 {
        // Rust saturates a float-to-integer cast, so this is total and never wraps.
        #[allow(clippy::cast_possible_truncation)]
        Some(value as i64)
    } else {
        None
    }
}

/// One endpoint of a `to` range: an `xs:integer` value, carried at fixed width when it fits an
/// `i64` and at arbitrary precision otherwise.
enum RangeBound {
    /// An endpoint within the `i64` range.
    Fixed(i64),
    /// An endpoint that overflowed the `i64` range.
    Big(BigInteger),
}

impl RangeBound {
    /// Widens the endpoint to a [`BigInteger`], for the arbitrary-precision range path.
    fn into_big(self) -> BigInteger {
        match self {
            Self::Fixed(value) => BigInteger::from_i64(value),
            Self::Big(value) => value,
        }
    }
}

fn bound(value: &XpathValue, what: &str) -> XPathResult<Option<RangeBound>> {
    let Some(atomic) = value.atomize_at_most_one(what)? else {
        return Ok(None);
    };
    let promoted = if atomic.atomic_type() == oxixml_infoset::AtomicType::UntypedAtomic {
        AtomicValue::parse(oxixml_infoset::AtomicType::Integer, &atomic.string_value())?
    } else {
        atomic
    };
    match promoted.data() {
        oxixml_infoset::AtomicData::Integer(value) => {
            Ok(Some(RangeBound::Fixed(i64::from(*value))))
        }
        data => match data.as_big_integer() {
            Some(big) => Ok(Some(RangeBound::Big(big.clone()))),
            None => Err(XPathError::type_error(format!(
                "{what} must be an xs:integer"
            ))),
        },
    }
}

fn concat_operand(value: &XpathValue) -> XPathResult<String> {
    Ok(value
        .atomize_at_most_one("an operand of `||`")?
        .map(|value| value.string_value())
        .unwrap_or_default())
}

fn set_operation(
    op: BinaryOp,
    left: XpathValue,
    right: &Expr,
    context: &DynamicContext,
) -> XPathResult<XpathValue> {
    let label = op.as_str();
    let left = require_nodes(&left, label)?;
    let right = require_nodes(&evaluate(right, context)?, label)?;
    let mut result: Vec<NodeRef> = match op {
        BinaryOp::Union => {
            let mut all = left;
            all.extend(right);
            all
        }
        BinaryOp::Intersect => {
            let keys: BTreeSet<_> = right.iter().map(XdmOrder::key).collect();
            left.into_iter()
                .filter(|node| keys.contains(&XdmOrder::key(node)))
                .collect()
        }
        _ => {
            let keys: BTreeSet<_> = right.iter().map(XdmOrder::key).collect();
            left.into_iter()
                .filter(|node| !keys.contains(&XdmOrder::key(node)))
                .collect()
        }
    };
    result.sort();
    result.dedup();
    Ok(XpathValue::from_iter(result))
}

/// A tiny helper so that the set operators can build a lookup set from document-order keys.
trait XdmOrder {
    fn key(&self) -> oxixml_infoset::DocumentOrderKey;
}

impl XdmOrder for NodeRef {
    fn key(&self) -> oxixml_infoset::DocumentOrderKey {
        oxixml_infoset::XdmNode::document_order_key(self)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::context::StaticContext;
    use crate::version::XPathVersion;
    use std::sync::Arc;

    fn context(version: XPathVersion) -> DynamicContext {
        DynamicContext::new(Arc::new(StaticContext::new())).with_version(version)
    }

    fn literal(value: AtomicValue) -> Expr {
        Expr::Literal(value)
    }

    #[test]
    fn xpath_10_arithmetic_coerces_everything_to_doubles() -> XPathResult<()> {
        let context = context(XPathVersion::XPath10);
        let result = binary(
            BinaryOp::Add,
            &literal(AtomicValue::string("3")),
            &literal(AtomicValue::double(4.0)),
            &context,
        )?;
        assert_eq!(result.to_string(), "7");

        let divided = binary(
            BinaryOp::Divide,
            &literal(AtomicValue::double(1.0)),
            &literal(AtomicValue::double(0.0)),
            &context,
        )?;
        assert_eq!(divided.to_string(), "INF");
        Ok(())
    }

    #[test]
    fn xpath_20_arithmetic_is_typed() -> XPathResult<()> {
        let context = context(XPathVersion::XPath31);
        assert!(
            binary(
                BinaryOp::Add,
                &literal(AtomicValue::string("3")),
                &literal(AtomicValue::integer(4_i64)),
                &context,
            )
            .is_err()
        );
        let promoted = binary(
            BinaryOp::Add,
            &literal(AtomicValue::untyped_atomic("3")),
            &literal(AtomicValue::integer(4_i64)),
            &context,
        )?;
        assert_eq!(promoted.to_string(), "7");
        assert!(
            binary(
                BinaryOp::IntegerDivide,
                &literal(AtomicValue::integer(1_i64)),
                &literal(AtomicValue::integer(0_i64)),
                &context,
            )
            .is_err()
        );
        Ok(())
    }

    #[test]
    fn an_empty_operand_makes_the_result_empty() -> XPathResult<()> {
        let context = context(XPathVersion::XPath31);
        let result = binary(
            BinaryOp::Add,
            &Expr::Sequence(Vec::new()),
            &literal(AtomicValue::integer(1_i64)),
            &context,
        )?;
        assert!(result.is_empty());
        Ok(())
    }

    /// Evaluates a whole XPath 3.1 expression that needs no context.
    fn evaluate_31(expression: &str) -> XPathResult<String> {
        let compiled = crate::XPath::compile_31(expression)?;
        Ok(compiled.evaluate_value_in(&compiled.context())?.to_string())
    }

    #[test]
    fn a_huge_range_is_aggregated_without_materializing_it() -> XPathResult<()> {
        // Every one of these ranges is billions of items — far beyond `MAX_RANGE` — so any of
        // them returning at all proves the sequence was never built.
        assert_eq!(evaluate_31("count(1 to 2000000000)")?, "2000000000");
        assert_eq!(evaluate_31("exists(1 to 2000000000)")?, "true");
        assert_eq!(evaluate_31("empty(1 to 2000000000)")?, "false");
        assert_eq!(evaluate_31("empty(5 to 1)")?, "true");
        // `fn:reverse` of a range keeps its count and existence.
        assert_eq!(
            evaluate_31("count(reverse(1 to 2000000000))")?,
            "2000000000"
        );
        Ok(())
    }

    #[test]
    fn a_constant_positional_predicate_over_a_huge_range_is_constant_time() -> XPathResult<()> {
        assert_eq!(evaluate_31("(1 to 1000000000)[1]")?, "1");
        assert_eq!(evaluate_31("(10 to 1000000000)[5]")?, "14");
        // A position past the end selects nothing.
        assert_eq!(evaluate_31("(1 to 1000000000)[0]")?, "");
        // Reversed, `[1]` is the range's last item.
        assert_eq!(evaluate_31("reverse(1 to 1000000000)[1]")?, "1000000000");
        Ok(())
    }

    #[test]
    fn the_fast_paths_agree_with_materialization_on_small_ranges() -> XPathResult<()> {
        assert_eq!(evaluate_31("count(1 to 5)")?, "5");
        assert_eq!(evaluate_31("(1 to 5)[2]")?, "2");
        assert_eq!(evaluate_31("(1 to 5)[6]")?, "");
        assert_eq!(evaluate_31("(1 to 5)[1.5]")?, "");
        // A non-positional predicate is untouched and still materializes the (small) range.
        assert_eq!(evaluate_31("(1 to 5)[. mod 2 = 0]")?, "2 4");
        // A wrongly typed bound is the same type error whether or not the fast path runs.
        assert!(evaluate_31("count('a' to 5)").is_err());
        Ok(())
    }

    #[test]
    fn ranges_ascend_and_collapse() -> XPathResult<()> {
        let context = context(XPathVersion::XPath31);
        let ascending = binary(
            BinaryOp::Range,
            &literal(AtomicValue::integer(1_i64)),
            &literal(AtomicValue::integer(4_i64)),
            &context,
        )?;
        assert_eq!(ascending.to_string(), "1 2 3 4");
        let empty = binary(
            BinaryOp::Range,
            &literal(AtomicValue::integer(4_i64)),
            &literal(AtomicValue::integer(1_i64)),
            &context,
        )?;
        assert!(empty.is_empty());
        Ok(())
    }

    #[test]
    fn concatenation_treats_the_empty_sequence_as_the_empty_string() -> XPathResult<()> {
        let context = context(XPathVersion::XPath31);
        let joined = binary(
            BinaryOp::Concat,
            &literal(AtomicValue::string("a")),
            &Expr::Sequence(Vec::new()),
            &context,
        )?;
        assert_eq!(joined.to_string(), "a");
        Ok(())
    }

    #[test]
    fn set_operators_require_nodes() {
        let context = context(XPathVersion::XPath31);
        assert!(
            binary(
                BinaryOp::Union,
                &literal(AtomicValue::integer(1_i64)),
                &literal(AtomicValue::integer(2_i64)),
                &context,
            )
            .is_err()
        );
    }
}
