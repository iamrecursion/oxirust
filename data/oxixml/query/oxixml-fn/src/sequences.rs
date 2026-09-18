//! The functions on sequences of
//! [F&O 3.1 §14](https://www.w3.org/TR/xpath-functions-31/#sequence-functions): the general
//! functions (`fn:empty` … `fn:subsequence`), the cardinality tests (`fn:zero-or-one` and
//! friends), `fn:deep-equal`, and the aggregates (`fn:count`, `fn:avg`, `fn:max`, `fn:min`,
//! `fn:sum`).
//!
//! The aggregates all begin by *atomizing and promoting*: `xs:untypedAtomic` values are cast to
//! `xs:double`, and the remaining values must be pairwise comparable or the function raises
//! `FORG0006`.

use crate::collation::Collation;
use crate::context::FnContext;
use crate::error::{FnError, FnErrorCode, FnResult};
use crate::numeric::{ArithOp, arithmetic};
use oxixml_infoset::{
    AtomicData, AtomicType, AtomicValue, Item, Sequence, compare_atomic_in_context,
};
use oxixml_xsd::{Decimal, Double, Integer, TimezoneOffset};
use std::cmp::Ordering;

/// [`fn:empty`](https://www.w3.org/TR/xpath-functions-31/#func-empty).
#[must_use]
pub fn empty(sequence: &Sequence) -> bool {
    sequence.is_empty()
}

/// [`fn:exists`](https://www.w3.org/TR/xpath-functions-31/#func-exists).
#[must_use]
pub fn exists(sequence: &Sequence) -> bool {
    !sequence.is_empty()
}

/// [`fn:head`](https://www.w3.org/TR/xpath-functions-31/#func-head).
#[must_use]
pub fn head(sequence: &Sequence) -> Sequence {
    sequence
        .first()
        .cloned()
        .map(Sequence::singleton)
        .unwrap_or_default()
}

/// [`fn:tail`](https://www.w3.org/TR/xpath-functions-31/#func-tail).
#[must_use]
pub fn tail(sequence: &Sequence) -> Sequence {
    Sequence::from_items(sequence.iter().skip(1).cloned())
}

/// [`fn:insert-before`](https://www.w3.org/TR/xpath-functions-31/#func-insert-before): inserts
/// `inserted` before the one-based `position`, clamping the position into range.
///
/// # Examples
///
/// ```
/// use oxixml_fn::sequences::insert_before;
/// use oxixml_infoset::{AtomicValue, Sequence};
///
/// let source = Sequence::from_iter([
///     AtomicValue::string("a"),
///     AtomicValue::string("b"),
///     AtomicValue::string("c"),
/// ]);
/// let z = Sequence::from(AtomicValue::string("z"));
///
/// let rendered = |sequence: Sequence| {
///     sequence.iter().map(ToString::to_string).collect::<Vec<_>>().join("")
/// };
/// assert_eq!(rendered(insert_before(&source, 0, &z)), "zabc");
/// assert_eq!(rendered(insert_before(&source, 1, &z)), "zabc");
/// assert_eq!(rendered(insert_before(&source, 2, &z)), "azbc");
/// assert_eq!(rendered(insert_before(&source, 4, &z)), "abcz");
/// assert_eq!(rendered(insert_before(&source, 5, &z)), "abcz");
/// ```
#[must_use]
pub fn insert_before(sequence: &Sequence, position: i64, inserted: &Sequence) -> Sequence {
    let length = i64::try_from(sequence.len()).unwrap_or(i64::MAX);
    let index = position.clamp(1, length + 1) - 1;
    let index = usize::try_from(index).unwrap_or(0);
    let mut items: Vec<Item> = sequence.as_slice()[..index].to_vec();
    items.extend(inserted.iter().cloned());
    items.extend(sequence.as_slice()[index..].iter().cloned());
    Sequence::from_items(items)
}

/// [`fn:remove`](https://www.w3.org/TR/xpath-functions-31/#func-remove): drops the item at the
/// one-based `position`; a position outside the sequence changes nothing.
#[must_use]
pub fn remove(sequence: &Sequence, position: i64) -> Sequence {
    Sequence::from_items(
        sequence
            .iter()
            .enumerate()
            .filter(|(index, _)| i64::try_from(*index + 1).unwrap_or(i64::MAX) != position)
            .map(|(_, item)| item.clone()),
    )
}

/// [`fn:reverse`](https://www.w3.org/TR/xpath-functions-31/#func-reverse).
#[must_use]
pub fn reverse(sequence: &Sequence) -> Sequence {
    Sequence::from_items(sequence.iter().rev().cloned())
}

/// [`fn:subsequence`](https://www.w3.org/TR/xpath-functions-31/#func-subsequence): the items
/// whose one-based position `p` satisfies `round(start) <= p < round(start) + round(length)`.
///
/// # Examples
///
/// ```
/// use oxixml_fn::sequences::subsequence;
/// use oxixml_infoset::{AtomicValue, Sequence};
///
/// let items = Sequence::from_iter((1..=5).map(AtomicValue::integer));
/// let rendered = |sequence: Sequence| {
///     sequence.iter().map(ToString::to_string).collect::<Vec<_>>().join("")
/// };
/// assert_eq!(rendered(subsequence(&items, 4.0, None)), "45");
/// assert_eq!(rendered(subsequence(&items, 3.0, Some(2.0))), "34");
/// assert_eq!(rendered(subsequence(&items, 1.5, Some(2.6))), "234");
/// assert_eq!(rendered(subsequence(&items, 0.0, Some(3.0))), "12");
/// assert_eq!(rendered(subsequence(&items, f64::NAN, Some(3.0))), "");
/// ```
#[must_use]
pub fn subsequence(sequence: &Sequence, start: f64, length: Option<f64>) -> Sequence {
    if start.is_nan() {
        return Sequence::empty();
    }
    let first = crate::numeric::round::round_f64_half_up(start);
    let limit = match length {
        None => f64::INFINITY,
        Some(length) if length.is_nan() => return Sequence::empty(),
        Some(length) => first + crate::numeric::round::round_f64_half_up(length),
    };
    if limit.is_nan() {
        return Sequence::empty();
    }
    Sequence::from_items(
        sequence
            .iter()
            .enumerate()
            .filter(|(index, _)| {
                let position = u32::try_from(*index + 1).map_or(f64::INFINITY, f64::from);
                position >= first && position < limit
            })
            .map(|(_, item)| item.clone()),
    )
}

/// [`fn:unordered`](https://www.w3.org/TR/xpath-functions-31/#func-unordered), which this
/// implementation realizes as the identity: the order it is free to choose is the input order.
#[must_use]
pub fn unordered(sequence: &Sequence) -> Sequence {
    sequence.clone()
}

/// [`fn:count`](https://www.w3.org/TR/xpath-functions-31/#func-count).
#[must_use]
pub fn count(sequence: &Sequence) -> i64 {
    i64::try_from(sequence.len()).unwrap_or(i64::MAX)
}

/// [`fn:zero-or-one`](https://www.w3.org/TR/xpath-functions-31/#func-zero-or-one).
///
/// # Errors
///
/// [`FnErrorCode::Forg0003`] when the sequence holds more than one item.
pub fn zero_or_one(sequence: &Sequence) -> FnResult<Sequence> {
    if sequence.len() > 1 {
        return Err(FnError::new(
            FnErrorCode::Forg0003,
            "fn:zero-or-one was given more than one item",
        ));
    }
    Ok(sequence.clone())
}

/// [`fn:one-or-more`](https://www.w3.org/TR/xpath-functions-31/#func-one-or-more).
///
/// # Errors
///
/// [`FnErrorCode::Forg0004`] for the empty sequence.
pub fn one_or_more(sequence: &Sequence) -> FnResult<Sequence> {
    if sequence.is_empty() {
        return Err(FnError::new(
            FnErrorCode::Forg0004,
            "fn:one-or-more was given the empty sequence",
        ));
    }
    Ok(sequence.clone())
}

/// [`fn:exactly-one`](https://www.w3.org/TR/xpath-functions-31/#func-exactly-one).
///
/// # Errors
///
/// [`FnErrorCode::Forg0005`] unless the sequence holds exactly one item.
pub fn exactly_one(sequence: &Sequence) -> FnResult<Sequence> {
    if sequence.len() != 1 {
        return Err(FnError::new(
            FnErrorCode::Forg0005,
            "fn:exactly-one expects exactly one item",
        ));
    }
    Ok(sequence.clone())
}

/// [`fn:deep-equal`](https://www.w3.org/TR/xpath-functions-31/#func-deep-equal).
///
/// # Errors
///
/// [`FnErrorCode::Foch0002`] for an unknown collation, plus whatever atomization raises.
pub fn deep_equal(
    context: &FnContext,
    left: &Sequence,
    right: &Sequence,
    collation: Option<&str>,
) -> FnResult<bool> {
    let collation = context.collation(collation)?;
    crate::deep_equal::deep_equal_with(left, right, &collation, context.implicit_timezone())
}

/// Atomizes a sequence and casts every `xs:untypedAtomic` value to `xs:double`, which is the
/// preamble `fn:sum`, `fn:avg`, `fn:max` and `fn:min` share.
///
/// # Errors
///
/// Whatever atomization raises, or [`FnErrorCode::Forg0001`] when an `xs:untypedAtomic` value is
/// not a number.
pub fn atomize_and_promote(sequence: &Sequence) -> FnResult<Vec<AtomicValue>> {
    atomize_with(sequence, AtomicType::Double)
}

/// Atomizes a sequence and re-labels every `xs:untypedAtomic` value as `xs:string`, which is
/// what `fn:distinct-values` and `fn:index-of` do: they *compare* rather than compute, and the
/// specification says untyped values compare as strings there.
///
/// # Errors
///
/// Whatever atomization raises.
pub fn atomize_as_strings(sequence: &Sequence) -> FnResult<Vec<AtomicValue>> {
    atomize_with(sequence, AtomicType::String)
}

fn atomize_with(sequence: &Sequence, untyped_becomes: AtomicType) -> FnResult<Vec<AtomicValue>> {
    let atomized = sequence.atomize()?;
    let mut out = Vec::with_capacity(atomized.len());
    for item in atomized.iter() {
        let Some(value) = item.as_atomic() else {
            return Err(FnError::new(
                FnErrorCode::Forg0006,
                "a sequence function was given a function item",
            ));
        };
        if value.atomic_type() == AtomicType::UntypedAtomic {
            out.push(AtomicValue::parse(
                untyped_becomes,
                value.as_str().unwrap_or_default(),
            )?);
        } else {
            out.push(value.clone());
        }
    }
    Ok(out)
}

/// [`fn:index-of`](https://www.w3.org/TR/xpath-functions-31/#func-index-of): the one-based
/// positions at which `search` occurs.
///
/// # Errors
///
/// [`FnErrorCode::Foch0002`] for an unknown collation.
///
/// # Examples
///
/// ```
/// use oxixml_fn::{sequences::index_of, FnContext};
/// use oxixml_infoset::{AtomicValue, Sequence};
///
/// let context = FnContext::new();
/// let items = Sequence::from_iter([10, 20, 30, 30, 20, 10].map(AtomicValue::integer));
/// assert_eq!(index_of(&context, &items, &AtomicValue::integer(20_i64), None)?, vec![2, 5]);
/// assert!(index_of(&context, &items, &AtomicValue::integer(99_i64), None)?.is_empty());
/// # Ok::<(), oxixml_fn::FnError>(())
/// ```
pub fn index_of(
    context: &FnContext,
    sequence: &Sequence,
    search: &AtomicValue,
    collation: Option<&str>,
) -> FnResult<Vec<i64>> {
    let collation = context.collation(collation)?;
    let implicit_timezone = context.implicit_timezone();
    let values = atomize_as_strings(sequence)?;
    let search = if search.atomic_type() == AtomicType::UntypedAtomic {
        AtomicValue::parse(AtomicType::String, search.as_str().unwrap_or_default())?
    } else {
        search.clone()
    };
    Ok(values
        .iter()
        .enumerate()
        // F&O 3.1 §14.3.14 selects the positions where `$search eq $item` is true, and `NaN eq
        // NaN` is false — unlike `fn:distinct-values`, which folds the NaNs together.
        .filter(|(_, value)| {
            matches!(
                compare_with(value, &search, collation.as_ref(), implicit_timezone),
                Ok(Some(Ordering::Equal))
            )
        })
        .map(|(index, _)| i64::try_from(index + 1).unwrap_or(i64::MAX))
        .collect())
}

/// [`fn:distinct-values`](https://www.w3.org/TR/xpath-functions-31/#func-distinct-values),
/// preserving the order of first occurrence.
///
/// Two `NaN` values count as duplicates of one another, which is the one place where
/// `fn:distinct-values` departs from the `eq` operator.
///
/// # Errors
///
/// [`FnErrorCode::Foch0002`] for an unknown collation.
pub fn distinct_values(
    context: &FnContext,
    sequence: &Sequence,
    collation: Option<&str>,
) -> FnResult<Sequence> {
    let collation = context.collation(collation)?;
    let implicit_timezone = context.implicit_timezone();
    let values = atomize_as_strings(sequence)?;
    let mut kept: Vec<AtomicValue> = Vec::new();
    for value in values {
        if !kept
            .iter()
            .any(|seen| equal_for_lookup(seen, &value, collation.as_ref(), implicit_timezone))
        {
            kept.push(value);
        }
    }
    Ok(Sequence::from_items(kept.into_iter().map(Item::Atomic)))
}

/// The equality relation `fn:index-of` and `fn:distinct-values` use: the `eq` operator, but with
/// `NaN` equal to itself and with incomparable types simply unequal rather than an error.
fn equal_for_lookup(
    left: &AtomicValue,
    right: &AtomicValue,
    collation: &dyn Collation,
    implicit_timezone: TimezoneOffset,
) -> bool {
    if left.is_nan() && right.is_nan() {
        return true;
    }
    matches!(
        compare_with(left, right, collation, implicit_timezone),
        Ok(Some(Ordering::Equal))
    )
}

/// Orders two atomic values under `collation`, reading a date/time operand with no timezone in
/// `implicit_timezone` — the [dynamic context's
/// implicit timezone](https://www.w3.org/TR/xpath-functions-31/#dt-implicit-timezone), per F&O
/// 3.1 §10.4.4 — exactly as `eq`/`lt` would.
fn compare_with(
    left: &AtomicValue,
    right: &AtomicValue,
    collation: &dyn Collation,
    implicit_timezone: TimezoneOffset,
) -> FnResult<Option<Ordering>> {
    if let (Some(left), Some(right)) = (text_of(left), text_of(right)) {
        return Ok(Some(collation.compare(left, right)));
    }
    // `op:duration-equal` is defined on `xs:duration`, so the two subtypes compare with each
    // other and with their base: `xs:yearMonthDuration('P0Y')` and `xs:dayTimeDuration('P0D')`
    // are one value, and `fn:distinct-values` keeps only the first of them.
    if let (Some(left), Some(right)) = (widen_duration(left), widen_duration(right)) {
        return Ok(left.partial_cmp(&right));
    }
    Ok(compare_atomic_in_context(
        left,
        right,
        None,
        Some(implicit_timezone),
    )?)
}

/// Widens any member of the duration family to `xs:duration`.
fn widen_duration(value: &AtomicValue) -> Option<oxixml_xsd::Duration> {
    match value.data() {
        AtomicData::Duration(duration) => Some(*duration),
        AtomicData::YearMonthDuration(duration) => Some(oxixml_xsd::Duration::from(*duration)),
        AtomicData::DayTimeDuration(duration) => Some(oxixml_xsd::Duration::from(*duration)),
        _ => None,
    }
}

fn text_of(value: &AtomicValue) -> Option<&str> {
    match value.data() {
        AtomicData::Text(text) => Some(text),
        _ => None,
    }
}

/// [`fn:max`](https://www.w3.org/TR/xpath-functions-31/#func-max).
///
/// # Errors
///
/// [`FnErrorCode::Forg0006`] when the values are not pairwise comparable, and
/// [`FnErrorCode::Foch0002`] for an unknown collation.
pub fn max(
    context: &FnContext,
    sequence: &Sequence,
    collation: Option<&str>,
) -> FnResult<Option<AtomicValue>> {
    extremum(context, sequence, collation, Ordering::Greater, "fn:max")
}

/// [`fn:min`](https://www.w3.org/TR/xpath-functions-31/#func-min).
///
/// # Errors
///
/// See [`max`].
pub fn min(
    context: &FnContext,
    sequence: &Sequence,
    collation: Option<&str>,
) -> FnResult<Option<AtomicValue>> {
    extremum(context, sequence, collation, Ordering::Less, "fn:min")
}

/// Whether `lt`, `le`, `gt` and `ge` — and therefore `fn:min`, `fn:max` and `fn:sort` — are
/// defined on values of this type.
///
/// F&O 3.1 §8.2 defines an `op:…-less-than` operator for exactly eleven type families —
/// `numeric`, `string` (which `xs:anyURI` and `xs:untypedAtomic` reach by promotion),
/// `boolean`, `date`, `time`, `dateTime`, `yearMonthDuration`, `dayTimeDuration`, `hexBinary`
/// and `base64Binary`. `xs:QName`, `xs:NOTATION`, bare `xs:duration` and the five `xs:gXxx`
/// types get only `op:…-equal`, so ordering them is an error rather than an arbitrary answer.
///
/// # Examples
///
/// ```
/// use oxixml_fn::sequences::is_ordered;
/// use oxixml_infoset::AtomicType;
///
/// assert!(is_ordered(AtomicType::Integer));
/// assert!(is_ordered(AtomicType::AnyUri));
/// assert!(is_ordered(AtomicType::HexBinary));
/// assert!(!is_ordered(AtomicType::QName));
/// assert!(!is_ordered(AtomicType::GYear));
/// assert!(!is_ordered(AtomicType::Duration));
/// ```
#[must_use]
pub fn is_ordered(atomic_type: AtomicType) -> bool {
    atomic_type.is_numeric()
        || atomic_type.is_text()
        || atomic_type.is_derived_from(AtomicType::YearMonthDuration)
        || atomic_type.is_derived_from(AtomicType::DayTimeDuration)
        || matches!(
            atomic_type.primitive(),
            AtomicType::Boolean
                | AtomicType::Date
                | AtomicType::Time
                | AtomicType::DateTime
                | AtomicType::HexBinary
                | AtomicType::Base64Binary
        )
}

/// Brings the values `fn:min` and `fn:max` were given to the one type the specification says
/// they are compared in, or reports `FORG0006` when there is no such type.
///
/// F&O 3.1 §14.4.6 converts the whole sequence *before* any comparison happens: the numeric
/// values are promoted to the least common numeric type, and a sequence that mixes a numeric
/// with a non-numeric — even when one of the numerics is `NaN` — is `FORG0006`. A type the `lt`
/// operator is not defined on (`xs:QName`, `xs:duration`, `xs:hexBinary`, an `xs:gXxx`, …) is
/// rejected here too.
///
/// F&O 3.1 §14.4.5–6 return an *item of the input sequence*, so the result keeps that item's own
/// type. The one conversion is the specification's own: "if the sequence contains values of type
/// `xs:anyURI` **and** values of type `xs:string`, the `xs:anyURI` values are converted to
/// `xs:string`". `xs:string` there means the type itself, not a restriction of it — QT3 pins all
/// three readings: `fn-max-16` (`xs:anyURI` + `xs:string`) is `xs:string`, `fn-max-17`
/// (`xs:anyURI` alone) is `xs:anyURI`, and `fn-max-18` (`xs:token` + `xs:anyURI`) is `xs:token`.
fn unify_for_extremum(values: &[AtomicValue], what: &str) -> FnResult<Vec<AtomicValue>> {
    let mismatch = || {
        FnError::new(
            FnErrorCode::Forg0006,
            format!("{what} was given values of types that cannot be compared with one another"),
        )
    };
    if let Some(unordered) = values.iter().find(|value| !is_ordered(value.atomic_type())) {
        return Err(FnError::new(
            FnErrorCode::Forg0006,
            format!("{what} cannot order {} values", unordered.atomic_type()),
        ));
    }
    // `op:yearMonthDuration-less-than` and `op:dayTimeDuration-less-than` order each subtype
    // among itself; there is no operator between the two, so a sequence holding both cannot be
    // ordered at all (QT3 `fn-max-8`, `fn-min-8`).
    let mut families = values.iter().filter_map(duration_family);
    if let Some(first) = families.next() {
        if families.any(|family| family != first) {
            return Err(mismatch());
        }
    }
    let numeric = values
        .iter()
        .filter(|value| value.atomic_type().is_numeric())
        .count();
    if numeric > 0 {
        if numeric != values.len() {
            return Err(mismatch());
        }
        let target = if values
            .iter()
            .any(|value| value.atomic_type().is_derived_from(AtomicType::Double))
        {
            Some(AtomicType::Double)
        } else if values
            .iter()
            .any(|value| value.atomic_type().is_derived_from(AtomicType::Float))
        {
            Some(AtomicType::Float)
        } else if values
            .iter()
            .any(|value| !value.atomic_type().is_derived_from(AtomicType::Integer))
        {
            Some(AtomicType::Decimal)
        } else {
            None
        };
        let Some(target) = target else {
            return Ok(values.to_vec());
        };
        let mut promoted = Vec::with_capacity(values.len());
        for value in values {
            promoted.push(if value.atomic_type() == target {
                value.clone()
            } else {
                AtomicValue::parse(target, &value.string_value())?
            });
        }
        return Ok(promoted);
    }
    let has_string = values
        .iter()
        .any(|value| value.atomic_type() == AtomicType::String);
    let has_uri = values
        .iter()
        .any(|value| value.atomic_type() == AtomicType::AnyUri);
    if has_string && has_uri {
        return Ok(values
            .iter()
            .map(|value| {
                if value.atomic_type() == AtomicType::AnyUri {
                    AtomicValue::string(value.string_value())
                } else {
                    value.clone()
                }
            })
            .collect());
    }
    Ok(values.to_vec())
}

fn extremum(
    context: &FnContext,
    sequence: &Sequence,
    collation: Option<&str>,
    wanted: Ordering,
    what: &str,
) -> FnResult<Option<AtomicValue>> {
    let collation = context.collation(collation)?;
    let implicit_timezone = context.implicit_timezone();
    let values = unify_for_extremum(&atomize_and_promote(sequence)?, what)?;
    let Some((first, rest)) = values.split_first() else {
        return Ok(None);
    };
    if values.iter().any(AtomicValue::is_nan) {
        // The specification: if the sequence contains NaN, the result is NaN.
        return Ok(Some(nan_like(&values)));
    }
    let mut best = first.clone();
    for candidate in rest {
        let ordering = compare_with(&best, candidate, collation.as_ref(), implicit_timezone)
            .map_err(|error| incomparable(what, &error))?
            .ok_or_else(|| {
                FnError::new(
                    FnErrorCode::Forg0006,
                    format!("{what} was given values that cannot be ordered"),
                )
            })?;
        if ordering != wanted && ordering != Ordering::Equal {
            best = candidate.clone();
        }
    }
    Ok(Some(best))
}

fn nan_like(values: &[AtomicValue]) -> AtomicValue {
    if values
        .iter()
        .any(|value| matches!(value.data(), AtomicData::Double(_)))
    {
        return AtomicValue::double(f64::NAN);
    }
    AtomicValue::float(f32::NAN)
}

fn incomparable(what: &str, error: &FnError) -> FnError {
    FnError::new(
        FnErrorCode::Forg0006,
        format!("{what}: {}", error.description()),
    )
}

/// [`fn:sum`](https://www.w3.org/TR/xpath-functions-31/#func-sum), the two-argument form.
///
/// `zero` is what the empty sequence sums to; `None` is the specification's "empty sequence"
/// second argument, which makes `fn:sum(())` return the empty sequence.
///
/// # Errors
///
/// [`FnErrorCode::Forg0006`] when the values cannot be added, and whatever the `+` operator
/// raises.
///
/// # Examples
///
/// ```
/// use oxixml_fn::sequences::sum;
/// use oxixml_infoset::{AtomicValue, Sequence};
///
/// let items = Sequence::from_iter([1_i64, 2, 3].map(AtomicValue::integer));
/// let zero = AtomicValue::integer(0_i64);
/// assert_eq!(sum(&items, Some(&zero))?.map(|value| value.string_value()), Some("6".to_owned()));
/// assert_eq!(
///     sum(&Sequence::empty(), Some(&zero))?.map(|value| value.string_value()),
///     Some("0".to_owned())
/// );
/// assert_eq!(sum(&Sequence::empty(), None)?, None);
/// # Ok::<(), oxixml_fn::FnError>(())
/// ```
pub fn sum(sequence: &Sequence, zero: Option<&AtomicValue>) -> FnResult<Option<AtomicValue>> {
    let values = atomize_and_promote(sequence)?;
    check_addable(&values, "fn:sum")?;
    let Some((first, rest)) = values.split_first() else {
        return Ok(zero.cloned());
    };
    let mut total = first.clone();
    for value in rest {
        // The type check above has already ruled out a mismatch, so an arithmetic failure here
        // is an overflow — `FOAR0002` or `FODT0002` — which must reach the caller unchanged.
        total = arithmetic(ArithOp::Add, &total, value)?;
    }
    Ok(Some(total))
}

/// Rejects an input `fn:sum` and `fn:avg` cannot add up.
///
/// F&O 3.1 §14.4.2–3 add the values with the `+` operator, which is defined only over the
/// numeric tower and *within* one of the two ordered duration subtypes; anything else — an
/// `xs:string`, an `xs:anyURI`, a bare `xs:duration`, or a mixture — is
/// [`FnErrorCode::Forg0006`]. Checking up front is what makes `fn:sum("a string")`, a
/// single-item sequence that never reaches an addition, an error rather than an echo of its
/// argument.
fn check_addable(values: &[AtomicValue], what: &str) -> FnResult<()> {
    let mismatch = |value: &AtomicValue| {
        FnError::new(
            FnErrorCode::Forg0006,
            format!("{what} cannot add {} values", value.atomic_type()),
        )
    };
    let Some(first) = values.first() else {
        return Ok(());
    };
    let numeric = first.atomic_type().is_numeric();
    if !numeric && !summable_duration(first) {
        return Err(mismatch(first));
    }
    for value in &values[1..] {
        let compatible = if numeric {
            value.atomic_type().is_numeric()
        } else {
            summable_duration(value)
                && value.atomic_type().primitive() == first.atomic_type().primitive()
                && duration_family(value) == duration_family(first)
        };
        if !compatible {
            return Err(mismatch(value));
        }
    }
    Ok(())
}

/// Whether `value` belongs to one of the two duration subtypes the `+` operator adds.
fn summable_duration(value: &AtomicValue) -> bool {
    duration_family(value).is_some()
}

/// Which addable duration subtype `value` belongs to, if any; `xs:duration` itself has none.
fn duration_family(value: &AtomicValue) -> Option<AtomicType> {
    let atomic_type = value.atomic_type();
    if atomic_type.is_derived_from(AtomicType::YearMonthDuration) {
        return Some(AtomicType::YearMonthDuration);
    }
    if atomic_type.is_derived_from(AtomicType::DayTimeDuration) {
        return Some(AtomicType::DayTimeDuration);
    }
    None
}

/// [`fn:avg`](https://www.w3.org/TR/xpath-functions-31/#func-avg).
///
/// # Errors
///
/// [`FnErrorCode::Forg0006`] when the values cannot be added or divided.
///
/// # Examples
///
/// ```
/// use oxixml_fn::sequences::avg;
/// use oxixml_infoset::{AtomicValue, Sequence};
///
/// let items = Sequence::from_iter([3_i64, 4, 5].map(AtomicValue::integer));
/// assert_eq!(avg(&items)?.map(|value| value.string_value()), Some("4".to_owned()));
/// assert_eq!(avg(&Sequence::empty())?, None);
/// # Ok::<(), oxixml_fn::FnError>(())
/// ```
pub fn avg(sequence: &Sequence) -> FnResult<Option<AtomicValue>> {
    let values = atomize_and_promote(sequence)?;
    check_addable(&values, "fn:avg")?;
    if values.is_empty() {
        return Ok(None);
    }
    let count = i64::try_from(values.len()).unwrap_or(i64::MAX);
    let Some(total) = sum(sequence, None)? else {
        return Ok(None);
    };
    let divisor = AtomicValue::integer(Integer::from(count));
    arithmetic(ArithOp::Divide, &total, &divisor).map(Some)
}

/// Builds the `xs:integer` an aggregate that counts returns.
#[must_use]
pub fn integer(value: i64) -> AtomicValue {
    AtomicValue::integer(Integer::from(value))
}

/// Builds the `xs:decimal` value some aggregates produce.
#[must_use]
pub fn decimal(value: Decimal) -> AtomicValue {
    AtomicValue::decimal(value)
}

/// Builds the `xs:double` value some aggregates produce.
#[must_use]
pub fn double(value: f64) -> AtomicValue {
    AtomicValue::double(Double::from(value))
}

#[cfg(test)]
mod tests {
    use super::*;
    use oxixml_infoset::Tree;

    fn strings(values: &[&str]) -> Sequence {
        Sequence::from_iter(values.iter().map(|value| AtomicValue::string(*value)))
    }

    fn integers(values: &[i64]) -> Sequence {
        Sequence::from_iter(values.iter().copied().map(AtomicValue::integer))
    }

    fn rendered(sequence: &Sequence) -> String {
        sequence
            .iter()
            .map(ToString::to_string)
            .collect::<Vec<_>>()
            .join(",")
    }

    #[test]
    fn general_functions() {
        let items = strings(&["a", "b", "c"]);
        assert!(!empty(&items));
        assert!(exists(&items));
        assert!(empty(&Sequence::empty()));
        assert_eq!(rendered(&head(&items)), "a");
        assert_eq!(rendered(&tail(&items)), "b,c");
        assert_eq!(rendered(&head(&Sequence::empty())), "");
        assert_eq!(rendered(&reverse(&items)), "c,b,a");
        assert_eq!(count(&items), 3);
    }

    #[test]
    fn insert_before_examples_from_the_specification() {
        let source = strings(&["a", "b", "c"]);
        let z = strings(&["z"]);
        assert_eq!(rendered(&insert_before(&source, 0, &z)), "z,a,b,c");
        assert_eq!(rendered(&insert_before(&source, 1, &z)), "z,a,b,c");
        assert_eq!(rendered(&insert_before(&source, 2, &z)), "a,z,b,c");
        assert_eq!(rendered(&insert_before(&source, 3, &z)), "a,b,z,c");
        assert_eq!(rendered(&insert_before(&source, 4, &z)), "a,b,c,z");
        assert_eq!(rendered(&insert_before(&source, 5, &z)), "a,b,c,z");
    }

    #[test]
    fn remove_examples_from_the_specification() {
        let source = strings(&["a", "b", "c"]);
        assert_eq!(rendered(&remove(&source, 0)), "a,b,c");
        assert_eq!(rendered(&remove(&source, 1)), "b,c");
        assert_eq!(rendered(&remove(&source, 6)), "a,b,c");
        assert_eq!(rendered(&remove(&Sequence::empty(), 3)), "");
    }

    #[test]
    fn subsequence_examples_from_the_specification() {
        let items = integers(&[1, 2, 3, 4, 5]);
        assert_eq!(rendered(&subsequence(&items, 4.0, None)), "4,5");
        assert_eq!(rendered(&subsequence(&items, 3.0, Some(2.0))), "3,4");
        assert_eq!(rendered(&subsequence(&items, 1.5, Some(2.6))), "2,3,4");
        assert_eq!(rendered(&subsequence(&items, 0.0, Some(3.0))), "1,2");
        assert_eq!(rendered(&subsequence(&items, f64::NAN, Some(3.0))), "");
        assert_eq!(rendered(&unordered(&items)), "1,2,3,4,5");
    }

    #[test]
    fn cardinality_functions() -> FnResult<()> {
        assert_eq!(zero_or_one(&Sequence::empty())?.len(), 0);
        assert_eq!(zero_or_one(&integers(&[1]))?.len(), 1);
        assert!(
            zero_or_one(&integers(&[1, 2]))
                .expect_err("two items")
                .is(FnErrorCode::Forg0003)
        );
        assert!(
            one_or_more(&Sequence::empty())
                .expect_err("no items")
                .is(FnErrorCode::Forg0004)
        );
        assert_eq!(one_or_more(&integers(&[1, 2]))?.len(), 2);
        assert_eq!(exactly_one(&integers(&[1]))?.len(), 1);
        assert!(
            exactly_one(&Sequence::empty())
                .expect_err("no items")
                .is(FnErrorCode::Forg0005)
        );
        Ok(())
    }

    #[test]
    fn index_of_examples_from_the_specification() -> FnResult<()> {
        let context = FnContext::new();
        let items = integers(&[10, 20, 30, 30, 20, 10]);
        assert_eq!(
            index_of(&context, &items, &AtomicValue::integer(20_i64), None)?,
            vec![2, 5]
        );
        assert!(index_of(&context, &items, &AtomicValue::integer(35_i64), None)?.is_empty());

        let words = strings(&["a", "sport", "and", "a", "pastime"]);
        assert_eq!(
            index_of(&context, &words, &AtomicValue::string("a"), None)?,
            vec![1, 4]
        );
        Ok(())
    }

    /// `fn:index-of` selects on `eq`, and `NaN eq NaN` is false — unlike `fn:distinct-values`,
    /// which folds the NaNs together.
    #[test]
    fn index_of_never_matches_nan() -> FnResult<()> {
        let context = FnContext::new();
        let nan = Sequence::from_iter([AtomicValue::double(f64::NAN)]);
        assert!(
            index_of(&context, &nan, &AtomicValue::double(f64::NAN), None)?.is_empty(),
            "NaN is not equal to itself"
        );
        assert_eq!(
            distinct_values(
                &context,
                &Sequence::from_iter([
                    AtomicValue::double(f64::NAN),
                    AtomicValue::double(f64::NAN),
                ]),
                None
            )?
            .len(),
            1
        );
        Ok(())
    }

    /// F&O 3.1 §14.4.5–6 compare with `gt`/`lt` and return an *item of the input*, so the
    /// result keeps its own type and an unorderable type is `FORG0006`.
    #[test]
    fn min_and_max_keep_the_type_and_reject_what_they_cannot_order() -> FnResult<()> {
        let context = FnContext::new();
        let uris = Sequence::from_iter([
            AtomicValue::parse(AtomicType::AnyUri, "http://a.com")?,
            AtomicValue::parse(AtomicType::AnyUri, "http://b.com")?,
        ]);
        let largest = max(&context, &uris, None)?.expect("two values");
        assert_eq!(largest.atomic_type(), AtomicType::AnyUri);
        assert_eq!(largest.string_value(), "http://b.com");

        let mixed = Sequence::from_iter([
            AtomicValue::parse(AtomicType::Token, "zither")?,
            AtomicValue::parse(AtomicType::AnyUri, "http://b.com")?,
        ]);
        let largest = max(&context, &mixed, None)?.expect("two values");
        assert_eq!(
            largest.atomic_type(),
            AtomicType::Token,
            "a mixture is compared as text but the winner keeps its label"
        );

        let with_string = Sequence::from_iter([
            AtomicValue::parse(AtomicType::AnyUri, "http://c.com")?,
            AtomicValue::string("http://b.com"),
        ]);
        let largest = max(&context, &with_string, None)?.expect("two values");
        assert_eq!(
            largest.atomic_type(),
            AtomicType::String,
            "an xs:string in the sequence converts the xs:anyURI values"
        );
        assert_eq!(largest.string_value(), "http://c.com");

        let across_families = Sequence::from_iter([
            AtomicValue::parse(AtomicType::YearMonthDuration, "P1Y")?,
            AtomicValue::parse(AtomicType::DayTimeDuration, "P1D")?,
        ]);
        assert!(
            max(&context, &across_families, None)
                .expect_err("the two duration subtypes have no ordering between them")
                .is(FnErrorCode::Forg0006)
        );

        for lexical in ["2000", "P1Y"] {
            let atomic_type = if lexical == "2000" {
                AtomicType::GYear
            } else {
                AtomicType::Duration
            };
            let unordered = Sequence::from_iter([AtomicValue::parse(atomic_type, lexical)?]);
            assert!(
                max(&context, &unordered, None)
                    .expect_err("not ordered")
                    .is(FnErrorCode::Forg0006)
            );
        }
        Ok(())
    }

    /// F&O 3.1 §14.4.2–3 add with the `+` operator, so anything it is not defined over —
    /// including a one-item sequence, which never reaches an addition — is `FORG0006`.
    #[test]
    fn sum_and_avg_reject_what_cannot_be_added() -> FnResult<()> {
        let text = Sequence::from_iter([AtomicValue::string("a string")]);
        assert!(
            sum(&text, None)
                .expect_err("strings do not add")
                .is(FnErrorCode::Forg0006)
        );
        assert!(
            avg(&text)
                .expect_err("strings do not average")
                .is(FnErrorCode::Forg0006)
        );
        let durations = Sequence::from_iter([AtomicValue::parse(AtomicType::Duration, "P1Y1M1D")?]);
        assert!(
            sum(&durations, None)
                .expect_err("xs:duration has no addition")
                .is(FnErrorCode::Forg0006)
        );
        let mixed = Sequence::from_iter([
            AtomicValue::integer(1_i64),
            AtomicValue::parse(AtomicType::YearMonthDuration, "P1Y")?,
        ]);
        assert!(sum(&mixed, None).is_err());
        let year_months = Sequence::from_iter([
            AtomicValue::parse(AtomicType::YearMonthDuration, "P1Y")?,
            AtomicValue::parse(AtomicType::YearMonthDuration, "P2M")?,
        ]);
        assert_eq!(
            sum(&year_months, None)?.map(|value| value.string_value()),
            Some("P1Y2M".to_owned())
        );
        Ok(())
    }

    #[test]
    fn distinct_values_examples_from_the_specification() -> FnResult<()> {
        let context = FnContext::new();
        let items = integers(&[1, 2, 3, 1, 2]);
        assert_eq!(rendered(&distinct_values(&context, &items, None)?), "1,2,3");

        let mixed = Sequence::from_iter([
            AtomicValue::integer(1_i64),
            AtomicValue::double(1.0),
            AtomicValue::string("1"),
        ]);
        assert_eq!(distinct_values(&context, &mixed, None)?.len(), 2);

        let nans =
            Sequence::from_iter([AtomicValue::double(f64::NAN), AtomicValue::double(f64::NAN)]);
        assert_eq!(distinct_values(&context, &nans, None)?.len(), 1);
        Ok(())
    }

    #[test]
    fn aggregates_over_numbers() -> FnResult<()> {
        let context = FnContext::new();
        let items = integers(&[3, 4, 5]);
        assert_eq!(
            sum(&items, Some(&AtomicValue::integer(0_i64)))?.map(|value| value.string_value()),
            Some("12".to_owned())
        );
        assert_eq!(
            avg(&items)?.map(|value| value.string_value()),
            Some("4".to_owned())
        );
        assert_eq!(
            max(&context, &items, None)?.map(|value| value.string_value()),
            Some("5".to_owned())
        );
        assert_eq!(
            min(&context, &items, None)?.map(|value| value.string_value()),
            Some("3".to_owned())
        );
        assert_eq!(avg(&Sequence::empty())?, None);
        assert_eq!(max(&context, &Sequence::empty(), None)?, None);
        assert_eq!(sum(&Sequence::empty(), None)?, None);
        Ok(())
    }

    #[test]
    fn aggregates_promote_across_the_numeric_tower() -> FnResult<()> {
        let mixed = Sequence::from_iter([AtomicValue::integer(1_i64), AtomicValue::double(2.5)]);
        let total = sum(&mixed, None)?
            .ok_or_else(|| FnError::new(FnErrorCode::Forg0006, "expected a total"))?;
        assert_eq!(total.atomic_type(), AtomicType::Double);
        assert_eq!(total.string_value(), "3.5");

        let average = avg(&mixed)?
            .ok_or_else(|| FnError::new(FnErrorCode::Forg0006, "expected an average"))?;
        assert_eq!(average.string_value(), "1.75");
        Ok(())
    }

    #[test]
    fn max_and_min_over_strings_use_the_collation() -> FnResult<()> {
        let context = FnContext::new();
        let words = strings(&["a", "b", "c"]);
        assert_eq!(
            max(&context, &words, None)?.map(|value| value.string_value()),
            Some("c".to_owned())
        );
        assert_eq!(
            min(&context, &words, None)?.map(|value| value.string_value()),
            Some("a".to_owned())
        );
        Ok(())
    }

    #[test]
    fn nan_dominates_max_and_min() -> FnResult<()> {
        let context = FnContext::new();
        let items = Sequence::from_iter([
            AtomicValue::double(1.0),
            AtomicValue::double(f64::NAN),
            AtomicValue::double(3.0),
        ]);
        assert!(
            max(&context, &items, None)?
                .ok_or_else(|| FnError::new(FnErrorCode::Forg0006, "expected a value"))?
                .is_nan()
        );
        Ok(())
    }

    #[test]
    fn incomparable_values_raise_forg0006() {
        let context = FnContext::new();
        let mixed = Sequence::from_iter([AtomicValue::integer(1_i64), AtomicValue::string("a")]);
        assert!(
            max(&context, &mixed, None)
                .expect_err("not comparable")
                .is(FnErrorCode::Forg0006)
        );
        assert!(
            sum(&mixed, None)
                .expect_err("not addable")
                .is(FnErrorCode::Forg0006)
        );
    }

    #[test]
    fn untyped_atomic_values_are_promoted_to_double() -> FnResult<()> {
        let items = Sequence::from_iter([
            AtomicValue::untyped_atomic("1"),
            AtomicValue::untyped_atomic("2"),
        ]);
        let total = sum(&items, None)?
            .ok_or_else(|| FnError::new(FnErrorCode::Forg0006, "expected a total"))?;
        assert_eq!(total.atomic_type(), AtomicType::Double);
        assert_eq!(total.string_value(), "3");
        Ok(())
    }

    #[test]
    fn deep_equal_compares_trees() -> Result<(), Box<dyn std::error::Error>> {
        let context = FnContext::new();
        let left = Tree::parse_str("<a><b>x</b></a>")?;
        let right = Tree::parse_str("<a><!-- c --><b>x</b></a>")?;
        assert!(deep_equal(
            &context,
            &Sequence::from(left.document_element().ok_or("no root")?),
            &Sequence::from(right.document_element().ok_or("no root")?),
            None
        )?);
        assert!(!deep_equal(
            &context,
            &integers(&[1, 2]),
            &integers(&[1, 2, 3]),
            None
        )?);
        Ok(())
    }

    #[test]
    fn convenience_constructors() {
        assert_eq!(integer(7).string_value(), "7");
        assert_eq!(double(1.5).string_value(), "1.5");
        assert_eq!(decimal(Decimal::from(3_i64)).string_value(), "3");
    }
}
