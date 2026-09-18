//! General, value and node comparisons — and the one place where XPath 1.0 and XPath 2.0 differ
//! most.
//!
//! # XPath 1.0 ([§3.4](https://www.w3.org/TR/xpath-10/#booleans))
//!
//! There are four types and five rules, chosen by the *kinds* of the operands:
//!
//! | Left | Right | `=` and `!=` | `<`, `<=`, `>`, `>=` |
//! |---|---|---|---|
//! | node-set | node-set | some pair of string values compares | some pair of *numbers* compares |
//! | node-set | number | some node's number compares | same |
//! | node-set | string | some node's string compares | some node's *number* compares |
//! | node-set | boolean | `boolean(node-set)` compares | both converted to numbers |
//! | other | other | boolean if either is boolean, else number if either is a number, else string | both converted to numbers |
//!
//! That is why `@x = 3` is a numeric test in XPath 1.0 even though `@x` is untyped text, and why
//! `not(a = b)` is not the same as `a != b`.
//!
//! # XPath 2.0 and later ([§3.7](https://www.w3.org/TR/xpath-31/#id-comparisons))
//!
//! There is one rule for general comparisons — atomize both operands and look for a pair that
//! satisfies the corresponding *value* comparison, with `xs:untypedAtomic` cast to the other
//! operand's type (to `xs:double` against a number, to `xs:string` against text) — and one for
//! value comparisons, which require at most one item on each side and compare typed values with
//! no cross-type guessing at all.

use super::{boolean, evaluate};
use crate::ast::{Expr, GeneralOp, NodeOp, ValueOp};
use crate::context::DynamicContext;
use crate::error::{XPathError, XPathResult};
use crate::value::{XpathItem, XpathValue};
use oxixml_infoset::{AtomicType, AtomicValue, BigInteger, XdmNode, compare_atomic_in_context};
use oxixml_xsd::Duration;
use std::cmp::Ordering;

/// Orders two atomic values, honouring the dynamic context's collation for text and its
/// [implicit timezone](https://www.w3.org/TR/xpath-functions-31/#dt-implicit-timezone) for
/// date/time values.
///
/// `None` means *incomparable but not an error* — `NaN` against a number, two `xs:duration`
/// values that neither contains. Every date/time comparison substitutes the context's implicit
/// timezone for an operand with none, per F&O 3.1 §10.4.4, so such a pair is never ambiguous
/// here.
///
/// # Errors
///
/// `XPTY0004` when the two types cannot be compared at all, `FOCH0002` for an unknown collation.
pub(crate) fn compare_atomic_values(
    context: &DynamicContext,
    left: &AtomicValue,
    right: &AtomicValue,
    collation: Option<&str>,
) -> XPathResult<Option<Ordering>> {
    if let (Some(left_text), Some(right_text)) = (left.as_str(), right.as_str()) {
        let collation = context.fn_context().collation(collation)?;
        return Ok(Some(collation.compare(left_text, right_text)));
    }
    // `op:duration-equal` is defined on xs:duration, so the two subtypes compare with each other
    // and with their base: `xs:yearMonthDuration('P0Y') eq xs:dayTimeDuration('P0D')` is true.
    if let (Some(left), Some(right)) = (as_duration(left), as_duration(right)) {
        return Ok(left.partial_cmp(&right));
    }
    let implicit_timezone = context.fn_context().implicit_timezone();
    Ok(compare_atomic_in_context(
        left,
        right,
        collation,
        Some(implicit_timezone),
    )?)
}

/// Widens any member of the duration family to `xs:duration`.
fn as_duration(value: &AtomicValue) -> Option<Duration> {
    match value.data() {
        oxixml_infoset::AtomicData::Duration(duration) => Some(*duration),
        oxixml_infoset::AtomicData::YearMonthDuration(duration) => Some(Duration::from(*duration)),
        oxixml_infoset::AtomicData::DayTimeDuration(duration) => Some(Duration::from(*duration)),
        _ => None,
    }
}

/// Which member of the duration family a value belongs to.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum DurationKind {
    /// `xs:duration` itself, which has *no* ordering — only `eq` and `ne` are defined.
    Full,
    /// `xs:yearMonthDuration`, totally ordered among itself.
    YearMonth,
    /// `xs:dayTimeDuration`, totally ordered among itself.
    DayTime,
}

fn duration_kind(value: &AtomicValue) -> Option<DurationKind> {
    match value.data() {
        oxixml_infoset::AtomicData::Duration(_) => Some(DurationKind::Full),
        oxixml_infoset::AtomicData::YearMonthDuration(_) => Some(DurationKind::YearMonth),
        oxixml_infoset::AtomicData::DayTimeDuration(_) => Some(DurationKind::DayTime),
        _ => None,
    }
}

/// Whether the pair supports only `eq` and `ne`.
///
/// [F&O §8.3](https://www.w3.org/TR/xpath-functions-31/#comp.duration) defines `op:duration-equal`
/// for every pair of durations, but `lt`/`le`/`gt`/`ge` only within `xs:yearMonthDuration` and
/// within `xs:dayTimeDuration`. Ordering an `xs:duration`, or two different subtypes, is
/// `XPTY0004`.
///
/// The same holds for every type the [operator mapping
/// table](https://www.w3.org/TR/xpath-31/#mapping) gives an `op:…-equal` function but no
/// `op:…-less-than`: `xs:QName`, `xs:NOTATION`, the two binary types and the five `xs:gXxx`
/// types. `QName("", "a") lt QName("", "b")` is `XPTY0004`, not `false`.
fn equality_only(left: &AtomicValue, right: &AtomicValue) -> bool {
    match (duration_kind(left), duration_kind(right)) {
        (Some(left), Some(right)) => left != right || left == DurationKind::Full,
        _ => !is_ordered(left.atomic_type()) || !is_ordered(right.atomic_type()),
    }
}

/// Whether `lt`, `le`, `gt` and `ge` are defined on values of this type — the same ordering
/// domain [`fn:min`/`fn:max`](oxixml_fn::sequences::is_ordered) use.
fn is_ordered(atomic_type: AtomicType) -> bool {
    oxixml_fn::sequences::is_ordered(atomic_type)
}

/// A general comparison, `= != < <= > >=`.
///
/// # Errors
///
/// `XPTY0004` when no pair of operands is comparable.
pub(crate) fn general(
    op: GeneralOp,
    left: &Expr,
    right: &Expr,
    context: &DynamicContext,
) -> XPathResult<XpathValue> {
    if context.version().is_xpath_10() {
        let left = evaluate(left, context)?;
        let right = evaluate(right, context)?;
        return Ok(boolean(general_10(op, &left, &right, context)?));
    }
    // A `to` range on either side is answered from its two bounds, so
    // `100000000002 = 100000000000 to 100001000003` costs two comparisons rather than a million
    // (`RangeExpr-408c`, `-408d`, `-409c`, `-409d`).
    if let Some(verdict) = general_with_range(op, left, right, context)? {
        return Ok(boolean(verdict));
    }
    let left = evaluate(left, context)?.atomize()?;
    let right = evaluate(right, context)?.atomize()?;
    // XPath 3.1 §3.7.2 evaluates the pairs and raises whatever comparing one of them raises:
    // `xs:anyURI('a') = 1` is `XPTY0004` for the same reason `xs:anyURI('a') eq 1` is. A
    // processor may stop as soon as a pair satisfies the comparison — this one does — but it may
    // not step *over* a pair that fails and look for a satisfying one further on, which is what
    // makes `(xs:anyURI('u'), QName('u','a')) = (1, QName('u','a'))` a type error rather than
    // true (`K-GenCompEq-23`).
    for a in &left {
        for b in &right {
            let (a, b) = promote_pair(a, b, context)?;
            if apply_value(op.value_op(), &a, &b, context)? {
                return Ok(boolean(true));
            }
        }
    }
    Ok(boolean(false))
}

/// A general comparison with a `to` range on one side, answered from the range's two bounds.
///
/// [XPath 3.1 §3.7.2](https://www.w3.org/TR/xpath-31/#id-general-comparisons) quantifies over
/// every pair, but every item of `E1 to E2` is an `xs:integer` and they ascend, so the quantifier
/// collapses: some item exceeds a value exactly when the *last* one does, some item is exceeded
/// by it exactly when the *first* one is, and some item equals it exactly when it is a whole
/// number between the two. That is what lets `100000000002 = 100000000000 to 100001000003`
/// answer without the million items in between — and what lets the arbitrary-precision
/// `1000000000000000020001 = 1000000000000000000000 to 1000000000000010000003` answer at all.
///
/// [`None`] means neither operand is a range and the ordinary rule applies.
///
/// # Errors
///
/// Whatever evaluating an operand raises, and `XPTY0004` for a value that cannot be compared
/// with an `xs:integer` — raised for the same pair, and after the same earlier pairs, as the
/// ordinary rule would raise it for.
fn general_with_range(
    op: GeneralOp,
    left: &Expr,
    right: &Expr,
    context: &DynamicContext,
) -> XPathResult<Option<bool>> {
    if let Some(bounds) = super::operators::range_bounds(right, context)? {
        let others = evaluate(left, context)?.atomize()?;
        return Ok(Some(match bounds {
            None => false,
            Some((from, to)) => against_range(op, &others, &from, &to, false, context)?,
        }));
    }
    if let Some(bounds) = super::operators::range_bounds(left, context)? {
        let others = evaluate(right, context)?.atomize()?;
        return Ok(Some(match bounds {
            None => false,
            Some((from, to)) => against_range(op, &others, &from, &to, true, context)?,
        }));
    }
    Ok(None)
}

/// Whether some item of the ascending range `from` … `to` satisfies `op` with some value of
/// `others`.
///
/// `range_first` says which side of the operator the range is on, which decides both the meaning
/// of an asymmetric operator and the order the pairs are enumerated in.
fn against_range(
    op: GeneralOp,
    others: &[AtomicValue],
    from: &BigInteger,
    to: &BigInteger,
    range_first: bool,
    context: &DynamicContext,
) -> XPathResult<bool> {
    let from_value = AtomicValue::from_big_integer(from.clone());
    let to_value = AtomicValue::from_big_integer(to.clone());
    if range_first {
        // With the range on the left the pairs are enumerated with it in the outer loop, so the
        // first sweep — every `other` against `from` — settles the question of order on its own:
        // whether a value can be compared with an integer does not depend on *which* integer, so
        // an incomparable value raises here or never, and a pair that satisfies the comparison
        // earlier than that raises is found here too.
        for other in others {
            if pair(op, &from_value, other, context)? {
                return Ok(true);
            }
        }
    }
    for other in others {
        if satisfied_by_some(op, other, &from_value, &to_value, range_first, context)? {
            return Ok(true);
        }
    }
    Ok(false)
}

/// Whether some item of the range satisfies `op` with the single value `other`.
fn satisfied_by_some(
    op: GeneralOp,
    other: &AtomicValue,
    from: &AtomicValue,
    to: &AtomicValue,
    range_first: bool,
    context: &DynamicContext,
) -> XPathResult<bool> {
    // Read every operator from the non-range value's side, so that one set of rules serves both
    // operand orders: `(1 to 5) lt $x` asks the same question as `$x gt (1 to 5)`.
    let op = if range_first { converse(op) } else { op };
    // `xs:untypedAtomic` takes the range's type once, up front, so that the wholeness test below
    // sees the same value the comparisons do.
    let (other, _) = promote_pair(other, from, context)?;
    match op {
        GeneralOp::Less => pair(GeneralOp::Less, &other, to, context),
        GeneralOp::LessEqual => pair(GeneralOp::LessEqual, &other, to, context),
        GeneralOp::Greater => pair(GeneralOp::Greater, &other, from, context),
        GeneralOp::GreaterEqual => pair(GeneralOp::GreaterEqual, &other, from, context),
        GeneralOp::Equal => {
            if !pair(GeneralOp::GreaterEqual, &other, from, context)?
                || !pair(GeneralOp::LessEqual, &other, to, context)?
            {
                return Ok(false);
            }
            is_whole(&other, context)
        }
        GeneralOp::NotEqual => {
            // The comparison has to be attempted whatever the answer, so that an incomparable
            // value is still a type error; `from` and `to` differ exactly when the range holds
            // more than one integer, and then some integer necessarily differs from `other`.
            let equals_first = pair(GeneralOp::Equal, &other, from, context)?;
            Ok(from != to || !equals_first)
        }
    }
}

/// The operator with its operands swapped, so `a op b` and `b converse(op) a` ask the same thing.
const fn converse(op: GeneralOp) -> GeneralOp {
    match op {
        GeneralOp::Equal => GeneralOp::Equal,
        GeneralOp::NotEqual => GeneralOp::NotEqual,
        GeneralOp::Less => GeneralOp::Greater,
        GeneralOp::LessEqual => GeneralOp::GreaterEqual,
        GeneralOp::Greater => GeneralOp::Less,
        GeneralOp::GreaterEqual => GeneralOp::LessEqual,
    }
}

/// One pair of a general comparison: promoted and compared exactly as the ordinary rule does it.
fn pair(
    op: GeneralOp,
    left: &AtomicValue,
    right: &AtomicValue,
    context: &DynamicContext,
) -> XPathResult<bool> {
    let (left, right) = promote_pair(left, right, context)?;
    apply_value(op.value_op(), &left, &right, context)
}

/// Whether `value` is equal to some `xs:integer` — the condition membership of an integer range
/// adds to lying between its bounds.
///
/// The test is `value mod 1 eq 0`, computed through the numeric tower so that it is exact for
/// `xs:decimal` at any precision and correct for the two floating-point types, where `NaN` and
/// the infinities all answer `false`.
fn is_whole(value: &AtomicValue, context: &DynamicContext) -> XPathResult<bool> {
    let remainder = oxixml_fn::numeric::arithmetic(
        oxixml_fn::numeric::ArithOp::Modulus,
        value,
        &AtomicValue::integer(1_i64),
    )?;
    apply_value(
        ValueOp::Equal,
        &remainder,
        &AtomicValue::integer(0_i64),
        context,
    )
}

/// XPath 2.0's rule for making a pair of atomic values comparable: `xs:untypedAtomic` takes the
/// other operand's type, becoming `xs:double` against a number and `xs:string` against anything
/// text-like.
fn promote_pair(
    left: &AtomicValue,
    right: &AtomicValue,
    context: &DynamicContext,
) -> XPathResult<(AtomicValue, AtomicValue)> {
    let left_untyped = left.atomic_type() == AtomicType::UntypedAtomic;
    let right_untyped = right.atomic_type() == AtomicType::UntypedAtomic;
    match (left_untyped, right_untyped) {
        (false, false) => Ok((left.clone(), right.clone())),
        (true, true) => Ok((as_string(left)?, as_string(right)?)),
        (true, false) => Ok((coerce_untyped(left, right, context)?, right.clone())),
        (false, true) => Ok((left.clone(), coerce_untyped(right, left, context)?)),
    }
}

fn coerce_untyped(
    untyped: &AtomicValue,
    other: &AtomicValue,
    context: &DynamicContext,
) -> XPathResult<AtomicValue> {
    if other.is_numeric() {
        // XPath 3.1 §3.7.1 *casts* the untyped operand to `xs:double`, so a lexical form that
        // is not a number is `FORG0001` — not a silent `NaN`, which would quietly make every
        // comparison false and hide malformed content.
        return Ok(AtomicValue::parse(
            AtomicType::Double,
            &untyped.string_value(),
        )?);
    }
    if other.atomic_type().is_text() {
        return as_string(untyped);
    }
    // The cast runs through the *expression's* static context, which matters only for the
    // namespace-sensitive types: `xs:untypedAtomic('my:problem') = QName('http://uri.two/',
    // 'problem')` answers differently under each `xmlns:my` in scope, and F&O 3.1 §19.1 says the
    // prefix is resolved against the statically known namespaces of the expression the cast
    // occurs in (XSLT 3.0 `choose-0106` does the same comparison three times to prove it).
    crate::cast::cast_with(
        untyped,
        other.atomic_type(),
        context.static_context(),
        crate::cast::StringSource::Literal,
    )
}

fn as_string(value: &AtomicValue) -> XPathResult<AtomicValue> {
    Ok(AtomicValue::string(value.string_value()))
}

/// A value comparison, `eq ne lt le gt ge`.
///
/// # Errors
///
/// `XPTY0004` when either operand atomizes to more than one value, or when the two values'
/// types are not comparable.
pub(crate) fn value(
    op: ValueOp,
    left: &Expr,
    right: &Expr,
    context: &DynamicContext,
) -> XPathResult<XpathValue> {
    let left = evaluate(left, context)?;
    let right = evaluate(right, context)?;
    let Some(left) = left.atomize_at_most_one("the left operand of a value comparison")? else {
        return Ok(XpathValue::empty());
    };
    let Some(right) = right.atomize_at_most_one("the right operand of a value comparison")? else {
        return Ok(XpathValue::empty());
    };
    // A value comparison never guesses a numeric type: xs:untypedAtomic compares as xs:string.
    let left = if left.atomic_type() == AtomicType::UntypedAtomic {
        as_string(&left)?
    } else {
        left
    };
    let right = if right.atomic_type() == AtomicType::UntypedAtomic {
        as_string(&right)?
    } else {
        right
    };
    Ok(boolean(apply_value(op, &left, &right, context)?))
}

fn apply_value(
    op: ValueOp,
    left: &AtomicValue,
    right: &AtomicValue,
    context: &DynamicContext,
) -> XPathResult<bool> {
    if !matches!(op, ValueOp::Equal | ValueOp::NotEqual) && equality_only(left, right) {
        return Err(XPathError::type_error(format!(
            "values of type {} and {} support only `eq` and `ne`",
            left.atomic_type(),
            right.atomic_type()
        )));
    }
    let ordering = compare_atomic_values(context, left, right, None)?;
    Ok(match (op, ordering) {
        // `NaN` and other incomparable pairs are unequal and unordered.
        (ValueOp::NotEqual, None) => true,
        (_, None) => false,
        (ValueOp::Equal, Some(ordering)) => ordering == Ordering::Equal,
        (ValueOp::NotEqual, Some(ordering)) => ordering != Ordering::Equal,
        (ValueOp::Less, Some(ordering)) => ordering == Ordering::Less,
        (ValueOp::LessEqual, Some(ordering)) => ordering != Ordering::Greater,
        (ValueOp::Greater, Some(ordering)) => ordering == Ordering::Greater,
        (ValueOp::GreaterEqual, Some(ordering)) => ordering != Ordering::Less,
    })
}

/// A node comparison, `is`, `<<` or `>>`.
///
/// # Errors
///
/// `XPTY0004` when either operand is not at most one node.
pub(crate) fn node(
    op: NodeOp,
    left: &Expr,
    right: &Expr,
    context: &DynamicContext,
) -> XPathResult<XpathValue> {
    let Some(left) = single_node(&evaluate(left, context)?)? else {
        return Ok(XpathValue::empty());
    };
    let Some(right) = single_node(&evaluate(right, context)?)? else {
        return Ok(XpathValue::empty());
    };
    let verdict = match op {
        NodeOp::Is => left.is_same_node(&right),
        NodeOp::Precedes => left.document_order_key() < right.document_order_key(),
        NodeOp::Follows => left.document_order_key() > right.document_order_key(),
    };
    Ok(boolean(verdict))
}

fn single_node(value: &XpathValue) -> XPathResult<Option<oxixml_infoset::NodeRef>> {
    match value.as_slice() {
        [] => Ok(None),
        [item] => item.as_node().cloned().map(Some).ok_or_else(|| {
            XPathError::type_error(format!(
                "a node comparison needs a node, but was given {}",
                item.describe()
            ))
        }),
        _ => Err(XPathError::type_error(
            "a node comparison needs at most one node on each side",
        )),
    }
}

// --- XPath 1.0 --------------------------------------------------------------------------

/// The XPath 1.0 `number()` conversion: a node-set becomes the number of its *first* node.
pub(crate) fn number_10(value: &XpathValue) -> f64 {
    match value.as_slice() {
        [] => f64::NAN,
        items if items.iter().any(XpathItem::is_node) => {
            let mut nodes = value.nodes();
            nodes.truncate(1);
            nodes
                .first()
                .map_or(f64::NAN, |node| parse_number(&node.string_value()))
        }
        [XpathItem::Value(oxixml_infoset::Item::Atomic(atomic))] => atomic_number_10(atomic),
        _ => f64::NAN,
    }
}

fn atomic_number_10(value: &AtomicValue) -> f64 {
    if let Some(truthy) = value.as_bool() {
        return f64::from(u8::from(truthy));
    }
    if value.is_numeric() {
        return value.as_f64().unwrap_or(f64::NAN);
    }
    parse_number(&value.string_value())
}

/// XPath 1.0's `Number` production: optional sign, digits, optional fraction — and nothing else,
/// so `1e3`, `INF` and `0x10` are all `NaN`.
fn parse_number(text: &str) -> f64 {
    let trimmed = text.trim_matches(|c: char| matches!(c, ' ' | '\t' | '\r' | '\n'));
    if trimmed.is_empty() {
        return f64::NAN;
    }
    let body = trimmed.strip_prefix('-').unwrap_or(trimmed);
    let mut digits = 0_usize;
    let mut points = 0_usize;
    for character in body.chars() {
        match character {
            '0'..='9' => digits += 1,
            '.' => points += 1,
            _ => return f64::NAN,
        }
    }
    if digits == 0 || points > 1 {
        return f64::NAN;
    }
    trimmed.parse::<f64>().unwrap_or(f64::NAN)
}

/// The XPath 1.0 `string()` conversion: a node-set becomes the string value of its *first* node.
pub(crate) fn string_10(value: &XpathValue) -> XPathResult<String> {
    match value.as_slice() {
        [] => Ok(String::new()),
        items if items.iter().any(XpathItem::is_node) => Ok(value
            .nodes()
            .first()
            .map(XdmNode::string_value)
            .unwrap_or_default()),
        [item] => item.string_value(),
        _ => Ok(String::new()),
    }
}

fn general_10(
    op: GeneralOp,
    left: &XpathValue,
    right: &XpathValue,
    context: &DynamicContext,
) -> XPathResult<bool> {
    let left_nodes = left.iter().any(XpathItem::is_node);
    let right_nodes = right.iter().any(XpathItem::is_node);
    let equality = matches!(op, GeneralOp::Equal | GeneralOp::NotEqual);

    if left_nodes && right_nodes {
        let lefts = strings_or_numbers(left, equality)?;
        let rights = strings_or_numbers(right, equality)?;
        return Ok(exists_pair(op, &lefts, &rights));
    }
    if left_nodes || right_nodes {
        let (nodes, other, flipped) = if left_nodes {
            (left, right, false)
        } else {
            (right, left, true)
        };
        // A boolean on the other side turns the whole node-set into one boolean.
        if equality && is_boolean(other) {
            let node_truth = !nodes.is_empty();
            let other_truth = other.effective_boolean_value()?;
            return Ok(match op {
                GeneralOp::Equal => node_truth == other_truth,
                _ => node_truth != other_truth,
            });
        }
        let numeric = !equality || is_numeric(other) || !is_text(other);
        let lefts = strings_or_numbers(nodes, !numeric)?;
        let rights = if numeric {
            Operands::Numbers(vec![number_10(other)])
        } else {
            Operands::Strings(vec![string_10(other)?])
        };
        let (lefts, rights) = if flipped {
            (rights, lefts)
        } else {
            (lefts, rights)
        };
        return Ok(exists_pair(op, &lefts, &rights));
    }

    if equality {
        if is_boolean(left) || is_boolean(right) {
            let a = left.effective_boolean_value()?;
            let b = right.effective_boolean_value()?;
            return Ok(match op {
                GeneralOp::Equal => a == b,
                _ => a != b,
            });
        }
        if is_numeric(left) || is_numeric(right) {
            return Ok(compare_numbers(op, number_10(left), number_10(right)));
        }
        let a = string_10(left)?;
        let b = string_10(right)?;
        let _ = context;
        return Ok(match op {
            GeneralOp::Equal => a == b,
            _ => a != b,
        });
    }
    Ok(compare_numbers(op, number_10(left), number_10(right)))
}

enum Operands {
    Strings(Vec<String>),
    Numbers(Vec<f64>),
}

fn strings_or_numbers(value: &XpathValue, as_strings: bool) -> XPathResult<Operands> {
    if value.iter().any(XpathItem::is_node) {
        let nodes = value.nodes();
        return Ok(if as_strings {
            Operands::Strings(nodes.iter().map(XdmNode::string_value).collect())
        } else {
            Operands::Numbers(
                nodes
                    .iter()
                    .map(|node| parse_number(&node.string_value()))
                    .collect(),
            )
        });
    }
    Ok(if as_strings {
        Operands::Strings(vec![string_10(value)?])
    } else {
        Operands::Numbers(vec![number_10(value)])
    })
}

fn exists_pair(op: GeneralOp, left: &Operands, right: &Operands) -> bool {
    match (left, right) {
        (Operands::Strings(lefts), Operands::Strings(rights)) => lefts.iter().any(|a| {
            rights.iter().any(|b| match op {
                GeneralOp::Equal => a == b,
                GeneralOp::NotEqual => a != b,
                _ => compare_numbers(op, parse_number(a), parse_number(b)),
            })
        }),
        (Operands::Numbers(lefts), Operands::Numbers(rights)) => lefts
            .iter()
            .any(|a| rights.iter().any(|b| compare_numbers(op, *a, *b))),
        (Operands::Strings(lefts), Operands::Numbers(rights)) => lefts.iter().any(|a| {
            rights
                .iter()
                .any(|b| compare_numbers(op, parse_number(a), *b))
        }),
        (Operands::Numbers(lefts), Operands::Strings(rights)) => lefts.iter().any(|a| {
            rights
                .iter()
                .any(|b| compare_numbers(op, *a, parse_number(b)))
        }),
    }
}

fn compare_numbers(op: GeneralOp, left: f64, right: f64) -> bool {
    match op {
        GeneralOp::Equal => left == right,
        GeneralOp::NotEqual => left != right,
        GeneralOp::Less => left < right,
        GeneralOp::LessEqual => left <= right,
        GeneralOp::Greater => left > right,
        GeneralOp::GreaterEqual => left >= right,
    }
}

fn is_boolean(value: &XpathValue) -> bool {
    matches!(value.as_slice(), [item] if item.as_atomic().is_some_and(|atomic| atomic.as_bool().is_some()))
}

fn is_numeric(value: &XpathValue) -> bool {
    matches!(value.as_slice(), [item] if item.as_atomic().is_some_and(AtomicValue::is_numeric))
}

fn is_text(value: &XpathValue) -> bool {
    matches!(value.as_slice(), [item] if item.as_atomic().is_some_and(|atomic| atomic.atomic_type().is_text()))
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

    #[test]
    fn xpath_10_numbers_reject_scientific_notation() {
        assert!(parse_number("1e3").is_nan());
        assert!(parse_number("INF").is_nan());
        assert_eq!(parse_number(" -12.5 "), -12.5);
        assert!(parse_number("").is_nan());
        assert!(parse_number("1.2.3").is_nan());
    }

    #[test]
    fn value_comparisons_compare_typed_values() -> XPathResult<()> {
        let context = context(XPathVersion::XPath31);
        let one = Expr::Literal(AtomicValue::integer(1_i64));
        let one_double = Expr::Literal(AtomicValue::double(1.0));
        assert_eq!(
            value(ValueOp::Equal, &one, &one_double, &context)?.to_string(),
            "true"
        );
        let text = Expr::Literal(AtomicValue::string("1"));
        assert!(value(ValueOp::Equal, &one, &text, &context).is_err());
        Ok(())
    }

    #[test]
    fn nan_is_unequal_to_itself() -> XPathResult<()> {
        let context = context(XPathVersion::XPath31);
        let nan = Expr::Literal(AtomicValue::double(f64::NAN));
        assert_eq!(
            value(ValueOp::Equal, &nan, &nan, &context)?.to_string(),
            "false"
        );
        assert_eq!(
            value(ValueOp::NotEqual, &nan, &nan, &context)?.to_string(),
            "true"
        );
        assert_eq!(
            value(ValueOp::Less, &nan, &nan, &context)?.to_string(),
            "false"
        );
        Ok(())
    }

    #[test]
    fn general_comparisons_are_existential() -> XPathResult<()> {
        let context = context(XPathVersion::XPath31);
        let sequence = Expr::Sequence(vec![
            Expr::Literal(AtomicValue::integer(1_i64)),
            Expr::Literal(AtomicValue::integer(2_i64)),
        ]);
        let two = Expr::Literal(AtomicValue::integer(2_i64));
        assert_eq!(
            general(GeneralOp::Equal, &sequence, &two, &context)?.to_string(),
            "true"
        );
        assert_eq!(
            general(GeneralOp::Less, &sequence, &two, &context)?.to_string(),
            "true"
        );
        Ok(())
    }

    #[test]
    fn a_value_comparison_on_the_empty_sequence_is_empty() -> XPathResult<()> {
        let context = context(XPathVersion::XPath31);
        let empty = Expr::Sequence(Vec::new());
        let one = Expr::Literal(AtomicValue::integer(1_i64));
        assert!(value(ValueOp::Equal, &empty, &one, &context)?.is_empty());
        assert_eq!(
            general(GeneralOp::Equal, &empty, &one, &context)?.to_string(),
            "false"
        );
        Ok(())
    }
}
