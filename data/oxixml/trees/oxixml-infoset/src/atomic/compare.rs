//! Cross-type comparison of atomic values: the XPath `eq`/`lt` operators, the map-key
//! relation `op:same-key`, and the collations they consult.

use super::bignum::BigDecimal;
use super::types::AtomicType;
use super::value::{AtomicData, AtomicValue};
use crate::error::{ErrorCode, XdmError};
use oxixml_support::bigint::BigUint;
use oxixml_xsd::{Decimal, Double, Duration, Float, TimezoneOffset};
use std::cmp::Ordering;
use std::hash::{Hash, Hasher};

/// The Unicode codepoint collation,
/// `http://www.w3.org/2005/xpath-functions/collation/codepoint`, which every conforming
/// processor must support and which is the default here.
pub const CODEPOINT_COLLATION: &str = "http://www.w3.org/2005/xpath-functions/collation/codepoint";

/// The HTML ASCII case-insensitive collation,
/// `http://www.w3.org/2005/xpath-functions/collation/html-ascii-case-insensitive`.
pub const HTML_ASCII_CASE_INSENSITIVE_COLLATION: &str =
    "http://www.w3.org/2005/xpath-functions/collation/html-ascii-case-insensitive";

/// Compares two strings under `collation`.
///
/// `None` selects [`CODEPOINT_COLLATION`]. Only the two collations every processor is required
/// to implement are supported here; a locale-sensitive (UCA) collation belongs to a higher
/// layer, which should compare strings itself rather than calling this function.
///
/// # Errors
///
/// [`ErrorCode::Foch0002`] if `collation` names a collation this crate does not implement.
///
/// # Examples
///
/// ```
/// use std::cmp::Ordering;
/// use oxixml_infoset::{compare_strings, HTML_ASCII_CASE_INSENSITIVE_COLLATION};
///
/// assert_eq!(compare_strings("abc", "abd", None)?, Ordering::Less);
/// assert_eq!(
///     compare_strings("ABC", "abc", Some(HTML_ASCII_CASE_INSENSITIVE_COLLATION))?,
///     Ordering::Equal
/// );
/// assert!(compare_strings("a", "b", Some("urn:unsupported")).is_err());
/// # Ok::<(), oxixml_infoset::XdmError>(())
/// ```
pub fn compare_strings(
    left: &str,
    right: &str,
    collation: Option<&str>,
) -> Result<Ordering, XdmError> {
    match collation {
        None | Some(CODEPOINT_COLLATION) => Ok(left.cmp(right)),
        Some(HTML_ASCII_CASE_INSENSITIVE_COLLATION) => Ok(left
            .chars()
            .map(|c| c.to_ascii_lowercase())
            .cmp(right.chars().map(|c| c.to_ascii_lowercase()))),
        Some(other) => Err(XdmError::new(
            ErrorCode::Foch0002,
            format!("the collation `{other}` is not supported"),
        )),
    }
}

/// XPath's `eq` operator on two atomic values, under the default (codepoint) collation.
///
/// See [`atomic_equal_with_collation`] for the full contract.
///
/// # Errors
///
/// [`ErrorCode::Xpty0004`] if the two values' types are not comparable.
pub fn atomic_equal(left: &AtomicValue, right: &AtomicValue) -> Result<bool, XdmError> {
    atomic_equal_with_collation(left, right, None)
}

/// XPath's [`eq` operator](https://www.w3.org/TR/xpath-31/#id-value-comparisons) on two atomic
/// values.
///
/// The rules that make this different from `==` on [`AtomicValue`]:
///
/// * `xs:untypedAtomic` operands compare as `xs:string` (value comparisons never guess a
///   numeric type), so comparing `xs:untypedAtomic "1"` with `xs:integer 1` is a **type
///   error**, exactly as `xs:string "1" eq 1` is.
/// * Numeric operands are compared after promotion to their common type, so
///   `xs:integer 1 eq xs:double 1.0E0` is true.
/// * `NaN` is not equal to anything, including itself.
/// * `xs:string`, `xs:anyURI` and their restrictions are compared under `collation`; when
///   `collation` is `None`, the collation carried on `left` is used, falling back to the one
///   carried on `right` and then to [`CODEPOINT_COLLATION`].
/// * Values of unrelated primitive types are a **type error**, not `false`.
///
/// # Errors
///
/// [`ErrorCode::Xpty0004`] if the two values' types are not comparable, or
/// [`ErrorCode::Foch0002`] if the effective collation is unsupported.
///
/// # Examples
///
/// ```
/// use oxixml_infoset::{atomic_equal, AtomicValue};
///
/// assert!(atomic_equal(&AtomicValue::integer(1_i64), &AtomicValue::double(1.0))?);
/// assert!(!atomic_equal(&AtomicValue::double(f64::NAN), &AtomicValue::double(f64::NAN))?);
/// assert!(atomic_equal(&AtomicValue::untyped_atomic("1"), &AtomicValue::integer(1_i64)).is_err());
/// assert!(atomic_equal(&AtomicValue::untyped_atomic("1"), &AtomicValue::string("1"))?);
/// assert!(atomic_equal(&AtomicValue::string("x"), &AtomicValue::any_uri("x"))?);
/// assert!(atomic_equal(&AtomicValue::integer(3_i64), &AtomicValue::string("3")).is_err());
/// # Ok::<(), oxixml_infoset::XdmError>(())
/// ```
pub fn atomic_equal_with_collation(
    left: &AtomicValue,
    right: &AtomicValue,
    collation: Option<&str>,
) -> Result<bool, XdmError> {
    match compare_atomic_with_collation(left, right, collation)? {
        Some(ordering) => Ok(ordering == Ordering::Equal),
        None => Ok(false),
    }
}

/// XPath's `eq` on two atomic values, with the dynamic context's collation *and* its
/// [implicit timezone](https://www.w3.org/TR/xpath-functions-31/#dt-implicit-timezone).
///
/// See [`compare_atomic_in_context`] for the full contract.
///
/// # Errors
///
/// [`ErrorCode::Xpty0004`] if the two values' types are not comparable, or
/// [`ErrorCode::Foch0002`] if the collation is unsupported.
pub fn atomic_equal_in_context(
    left: &AtomicValue,
    right: &AtomicValue,
    collation: Option<&str>,
    implicit_timezone: Option<TimezoneOffset>,
) -> Result<bool, XdmError> {
    match compare_atomic_in_context(left, right, collation, implicit_timezone)? {
        Some(ordering) => Ok(ordering == Ordering::Equal),
        None => Ok(false),
    }
}

/// XPath's ordering comparison on two atomic values, under the default (codepoint) collation.
///
/// See [`compare_atomic_with_collation`] for the full contract.
///
/// # Errors
///
/// [`ErrorCode::Xpty0004`] if the two values' types are not comparable.
pub fn compare_atomic(
    left: &AtomicValue,
    right: &AtomicValue,
) -> Result<Option<Ordering>, XdmError> {
    compare_atomic_with_collation(left, right, None)
}

/// Orders two atomic values, returning `None` when the values are *incomparable* even though
/// their types are.
///
/// Incomparability is a real part of the data model, not an error: `NaN` is unordered with
/// respect to every number including itself, and `xs:duration` values are only partially ordered.
///
/// Date/time values are *always* comparable here: F&O 3.1 reads an operand that carries no
/// timezone in the dynamic context's implicit timezone, which this function takes to be UTC. Use
/// [`compare_atomic_in_context`] to supply a different one, or `oxixml_xsd`'s `xsd_partial_cmp`
/// for XML Schema's own (genuinely partial) order relation.
///
/// # Errors
///
/// [`ErrorCode::Xpty0004`] if the two values' types are not comparable at all, or
/// [`ErrorCode::Foch0002`] if the effective collation is unsupported.
///
/// # Examples
///
/// ```
/// use std::cmp::Ordering;
/// use oxixml_infoset::{compare_atomic, AtomicValue};
///
/// assert_eq!(
///     compare_atomic(&AtomicValue::integer(1_i64), &AtomicValue::decimal(2_i64))?,
///     Some(Ordering::Less)
/// );
/// assert_eq!(
///     compare_atomic(&AtomicValue::double(f64::NAN), &AtomicValue::integer(0_i64))?,
///     None
/// );
/// # Ok::<(), oxixml_infoset::XdmError>(())
/// ```
pub fn compare_atomic_with_collation(
    left: &AtomicValue,
    right: &AtomicValue,
    collation: Option<&str>,
) -> Result<Option<Ordering>, XdmError> {
    compare_atomic_in_context(left, right, collation, None)
}

/// XPath's ordering comparison, with the dynamic context's
/// [implicit timezone](https://www.w3.org/TR/xpath-functions-31/#dt-implicit-timezone) supplied
/// explicitly.
///
/// F&O 3.1 defines every `xs:dateTime`/`xs:date`/`xs:time`/`xs:gYear…` comparison in terms of the
/// implicit timezone: an operand that carries no timezone is read *in* that timezone before the
/// two are compared, so — unlike XML Schema's own order relation — the answer is always
/// determinate. Pass `None` to use the default implicit timezone, UTC, which is what
/// [`compare_atomic_with_collation`] and the `PartialOrd`/`PartialEq` impls of the underlying
/// [`oxixml_xsd`] types apply; pass `Some(offset)` from an evaluation context whose implicit
/// timezone differs.
///
/// # Errors
///
/// [`ErrorCode::Xpty0004`] if the two values' types are not comparable at all, or
/// [`ErrorCode::Foch0002`] if the effective collation is unsupported.
///
/// # Examples
///
/// ```
/// use std::cmp::Ordering;
/// use oxixml_infoset::{compare_atomic_in_context, AtomicType, AtomicValue};
/// use oxixml_xsd::TimezoneOffset;
///
/// let floating = AtomicValue::parse(AtomicType::DateTime, "2024-06-01T10:00:00")?;
/// let utc = AtomicValue::parse(AtomicType::DateTime, "2024-06-01T10:00:00Z")?;
///
/// // Read in UTC (the default implicit timezone) the two denote the same instant.
/// assert_eq!(compare_atomic_in_context(&floating, &utc, None, None)?, Some(Ordering::Equal));
///
/// // Read in +09:00 the timezone-less value is nine hours earlier.
/// let jst = TimezoneOffset::new(9 * 60).expect("a valid offset");
/// assert_eq!(
///     compare_atomic_in_context(&floating, &utc, None, Some(jst))?,
///     Some(Ordering::Less)
/// );
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
pub fn compare_atomic_in_context(
    left: &AtomicValue,
    right: &AtomicValue,
    collation: Option<&str>,
    implicit_timezone: Option<TimezoneOffset>,
) -> Result<Option<Ordering>, XdmError> {
    let timezone = implicit_timezone.unwrap_or(TimezoneOffset::UTC);
    let left_data = left.data();
    let right_data = right.data();

    if left_data.is_numeric() && right_data.is_numeric() {
        return Ok(compare_numeric(left_data, right_data));
    }

    match (left_data, right_data) {
        (AtomicData::Text(a), AtomicData::Text(b)) => {
            let effective = collation
                .or_else(|| left.collation())
                .or_else(|| right.collation());
            Ok(Some(compare_strings(a, b, effective)?))
        }
        (AtomicData::Boolean(a), AtomicData::Boolean(b)) => Ok(Some(a.cmp(b))),
        (AtomicData::Duration(a), AtomicData::Duration(b)) => Ok(a.partial_cmp(b)),
        (AtomicData::YearMonthDuration(a), AtomicData::YearMonthDuration(b)) => Ok(Some(a.cmp(b))),
        (AtomicData::DayTimeDuration(a), AtomicData::DayTimeDuration(b)) => Ok(Some(a.cmp(b))),
        // F&O 3.1: every date/time comparison substitutes the implicit timezone for a missing
        // one, so the result is always determinate — never `None`.
        (AtomicData::DateTime(a), AtomicData::DateTime(b)) => {
            Ok(Some(a.compare_with_implicit_timezone(*b, timezone)))
        }
        (AtomicData::Date(a), AtomicData::Date(b)) => {
            Ok(Some(a.compare_with_implicit_timezone(*b, timezone)))
        }
        (AtomicData::Time(a), AtomicData::Time(b)) => {
            Ok(Some(a.compare_with_implicit_timezone(*b, timezone)))
        }
        (AtomicData::GYearMonth(a), AtomicData::GYearMonth(b)) => {
            Ok(Some(a.compare_with_implicit_timezone(*b, timezone)))
        }
        (AtomicData::GYear(a), AtomicData::GYear(b)) => {
            Ok(Some(a.compare_with_implicit_timezone(*b, timezone)))
        }
        (AtomicData::GMonthDay(a), AtomicData::GMonthDay(b)) => {
            Ok(Some(a.compare_with_implicit_timezone(*b, timezone)))
        }
        (AtomicData::GMonth(a), AtomicData::GMonth(b)) => {
            Ok(Some(a.compare_with_implicit_timezone(*b, timezone)))
        }
        (AtomicData::GDay(a), AtomicData::GDay(b)) => {
            Ok(Some(a.compare_with_implicit_timezone(*b, timezone)))
        }
        (AtomicData::Binary(a), AtomicData::Binary(b)) => {
            if binary_family(left.atomic_type()) != binary_family(right.atomic_type()) {
                return Err(incomparable(left, right));
            }
            Ok(Some(a.cmp(b)))
        }
        (AtomicData::QName(a), AtomicData::QName(b)) => {
            // xs:QName supports `eq` but not `lt`/`gt`: report Equal or Greater/Less on the
            // (namespace, local) pair so that callers wanting only equality get the right
            // answer, and callers wanting an order get a stable, documented one.
            Ok(Some(
                (a.namespace(), a.local_name()).cmp(&(b.namespace(), b.local_name())),
            ))
        }
        _ => Err(incomparable(left, right)),
    }
}

fn incomparable(left: &AtomicValue, right: &AtomicValue) -> XdmError {
    XdmError::new(
        ErrorCode::Xpty0004,
        format!(
            "values of type {} and {} are not comparable",
            left.atomic_type(),
            right.atomic_type()
        ),
    )
}

fn binary_family(atomic_type: AtomicType) -> AtomicType {
    if atomic_type.is_derived_from(AtomicType::Base64Binary) {
        AtomicType::Base64Binary
    } else {
        AtomicType::HexBinary
    }
}

/// Compares two numeric payloads after promoting them to their least common type, per
/// [XPath 3.1 §B.1](https://www.w3.org/TR/xpath-31/#promotion).
///
/// This is the `eq`/`lt` relation, so it *promotes*: an `xs:integer` or `xs:decimal` (of either
/// width) compared with an `xs:double` is rounded to that double first. The exact,
/// non-promoting relation the map-key rule needs is [`same_key`], not this.
fn compare_numeric(left: &AtomicData, right: &AtomicData) -> Option<Ordering> {
    use AtomicData::{Decimal as Dec, Double as Dbl, Float as Flt, Integer as Int};
    match (left, right) {
        (Int(a), Int(b)) => Some(a.cmp(b)),
        (Int(a), Dec(b)) => Some(Decimal::from(*a).cmp(b)),
        (Dec(a), Int(b)) => Some(a.cmp(&Decimal::from(*b))),
        (Dec(a), Dec(b)) => Some(a.cmp(b)),
        (Dbl(_), _) | (_, Dbl(_)) => {
            let a = to_double(left)?;
            let b = to_double(right)?;
            a.partial_cmp(&b)
        }
        (Flt(_), _) | (_, Flt(_)) => {
            let a = to_float(left)?;
            let b = to_float(right)?;
            a.partial_cmp(&b)
        }
        // Every remaining pair is drawn from the exact tower (`xs:integer`/`xs:decimal`, either
        // width) with at least one arbitrary-precision operand: compare them exactly, since no
        // float promotion is involved to impose a rounding.
        _ => Some(numeric_to_exact(left)?.cmp(&numeric_to_exact(right)?)),
    }
}

/// The exact value of an `xs:integer`/`xs:decimal` payload — of either width — as a
/// [`BigDecimal`], for the exact comparisons the map-key rule and the arbitrary-precision `eq`/
/// `lt` paths need. Returns [`None`] for a payload that is not an exact-tower member.
fn numeric_to_exact(data: &AtomicData) -> Option<BigDecimal> {
    Some(match data {
        AtomicData::Integer(value) => BigDecimal::from_fixed(Decimal::from(*value)),
        AtomicData::Decimal(value) => BigDecimal::from_fixed(*value),
        AtomicData::BigInteger(value) => BigDecimal::from_big_integer(value.clone()),
        AtomicData::BigDecimal(value) => value.clone(),
        _ => return None,
    })
}

fn to_double(data: &AtomicData) -> Option<Double> {
    Some(match data {
        AtomicData::Integer(value) => Double::from(*value),
        AtomicData::Decimal(value) => Double::from(*value),
        AtomicData::BigInteger(value) => Double::from(value.to_f64()),
        AtomicData::BigDecimal(value) => Double::from(value.to_f64()),
        AtomicData::Float(value) => Double::from(*value),
        AtomicData::Double(value) => *value,
        _ => return None,
    })
}

fn to_float(data: &AtomicData) -> Option<Float> {
    Some(match data {
        AtomicData::Integer(value) => Float::from(*value),
        AtomicData::Decimal(value) => Float::from(*value),
        // Route through the same `f64 → Double → Float` step an `xs:double` takes, so a big
        // value and the double it equals round to a bit-identical `xs:float`.
        AtomicData::BigInteger(value) => Float::from(Double::from(value.to_f64())),
        AtomicData::BigDecimal(value) => Float::from(Double::from(value.to_f64())),
        AtomicData::Float(value) => *value,
        AtomicData::Double(value) => Float::from(*value),
        _ => return None,
    })
}

/// Widens any member of the duration family to `xs:duration`, the type whose value space the
/// three share.
fn widen_duration(data: &AtomicData) -> Option<Duration> {
    match data {
        AtomicData::Duration(duration) => Some(*duration),
        AtomicData::YearMonthDuration(duration) => Some(Duration::from(*duration)),
        AtomicData::DayTimeDuration(duration) => Some(Duration::from(*duration)),
        _ => None,
    }
}

/// XQuery 3.1's [`op:same-key`](https://www.w3.org/TR/xpath-functions-31/#func-map-contains)
/// relation, which decides when two atomic values are the same *map key*.
///
/// It differs from `eq` in three ways that matter:
///
/// * it never raises an error — values of unrelated types are simply different keys;
/// * `NaN` is the same key as `NaN`;
/// * it is always codepoint-based for strings, and it requires two date/time values to agree on
///   whether they carry a timezone.
///
/// `xs:untypedAtomic`, `xs:string` and `xs:anyURI` values with the same characters are the same
/// key, matching the specification's grouping of the three into one comparison rule.
///
/// # Examples
///
/// ```
/// use oxixml_infoset::{same_key, AtomicValue};
///
/// // Numeric keys are compared across types...
/// assert!(same_key(&AtomicValue::integer(1_i64), &AtomicValue::double(1.0)));
/// // ...and NaN is its own key, unlike under `eq`.
/// assert!(same_key(&AtomicValue::double(f64::NAN), &AtomicValue::double(f64::NAN)));
/// // Unrelated types are simply distinct keys rather than an error.
/// assert!(!same_key(&AtomicValue::integer(1_i64), &AtomicValue::string("1")));
/// ```
#[must_use]
pub fn same_key(left: &AtomicValue, right: &AtomicValue) -> bool {
    let (a, b) = (left.data(), right.data());
    if a.is_numeric() && b.is_numeric() {
        return same_key_numeric(a, b);
    }
    // The three duration types share one value space, so `op:duration-equal` — and with it the
    // same-key relation — crosses the family: `xs:duration("P1Y")` and
    // `xs:yearMonthDuration("P12M")` are one key.
    if let (Some(x), Some(y)) = (widen_duration(a), widen_duration(b)) {
        return x == y;
    }
    match (a, b) {
        (AtomicData::Text(x), AtomicData::Text(y)) => x == y,
        (AtomicData::Boolean(x), AtomicData::Boolean(y)) => x == y,
        // `op:same-key` is deliberately independent of the dynamic context, so a date/time value
        // that carries a timezone is never the same key as one that does not — that is XSD 1.1
        // equality, not the F&O `eq` that reads a missing timezone as the implicit one.
        (AtomicData::DateTime(x), AtomicData::DateTime(y)) => x.xsd_eq(*y),
        (AtomicData::Date(x), AtomicData::Date(y)) => x.xsd_eq(*y),
        (AtomicData::Time(x), AtomicData::Time(y)) => x.xsd_eq(*y),
        // The five Gregorian types follow the same rule as the three above: `xsd_eq`, not the
        // `PartialEq` that reads a missing timezone as the implicit one.
        (AtomicData::GYearMonth(x), AtomicData::GYearMonth(y)) => x.xsd_eq(*y),
        (AtomicData::GYear(x), AtomicData::GYear(y)) => x.xsd_eq(*y),
        (AtomicData::GMonthDay(x), AtomicData::GMonthDay(y)) => x.xsd_eq(*y),
        (AtomicData::GMonth(x), AtomicData::GMonth(y)) => x.xsd_eq(*y),
        (AtomicData::GDay(x), AtomicData::GDay(y)) => x.xsd_eq(*y),
        (AtomicData::Binary(x), AtomicData::Binary(y)) => {
            binary_family(left.atomic_type()) == binary_family(right.atomic_type()) && x == y
        }
        (AtomicData::QName(x), AtomicData::QName(y)) => x == y,
        _ => false,
    }
}

/// The map-key relation on two numeric payloads: **exact** value equality, unlike the promoting
/// [`compare_numeric`] that `eq` uses.
///
/// The distinction is observable exactly where it must be: `xs:integer 9223372036854775809` and
/// the `xs:double` it *rounds* to are different keys, even though `eq` — which promotes the
/// integer to that double — would call them equal. `NaN` is its own key; the two zeros are one.
fn same_key_numeric(a: &AtomicData, b: &AtomicData) -> bool {
    if a.is_nan() || b.is_nan() {
        return a.is_nan() && b.is_nan();
    }
    // The common map keys — two fixed-width integers, or decimals — are compared directly, so a
    // map that never leaves those types keeps its former allocation-free path. Anything that
    // brings in a float, double or arbitrary-precision value takes the exact route below.
    use AtomicData::{Decimal as Dec, Integer as Int};
    match (a, b) {
        (Int(x), Int(y)) => return x == y,
        (Dec(x), Dec(y)) => return x == y,
        (Int(x), Dec(y)) => return Decimal::from(*x) == *y,
        (Dec(x), Int(y)) => return *x == Decimal::from(*y),
        _ => {}
    }
    match (exact_key(a), exact_key(b)) {
        (Some(x), Some(y)) => x == y,
        _ => false,
    }
}

/// The exact value of a numeric payload as a map key: a finite value carries its exact decimal;
/// the two infinities are their own keys, distinct from every finite value and from each other.
#[derive(PartialEq)]
enum ExactKey {
    /// A finite value, exactly.
    Finite(BigDecimal),
    /// Positive infinity (only an `xs:double`/`xs:float` can be).
    PosInf,
    /// Negative infinity.
    NegInf,
}

/// The [`ExactKey`] of a numeric payload, or [`None`] for a non-numeric one. `NaN` must have been
/// handled by the caller.
fn exact_key(data: &AtomicData) -> Option<ExactKey> {
    match data {
        AtomicData::Integer(_) | AtomicData::Decimal(_) => {
            Some(ExactKey::Finite(numeric_to_exact(data)?))
        }
        AtomicData::BigInteger(_) | AtomicData::BigDecimal(_) => {
            Some(ExactKey::Finite(numeric_to_exact(data)?))
        }
        AtomicData::Double(value) => f64_exact_key(f64::from(*value)),
        AtomicData::Float(value) => f64_exact_key(f64::from(f32::from(*value))),
        _ => None,
    }
}

/// The [`ExactKey`] of a finite-or-infinite `f64` (an `f32` is widened losslessly first).
fn f64_exact_key(value: f64) -> Option<ExactKey> {
    if value == f64::INFINITY {
        Some(ExactKey::PosInf)
    } else if value == f64::NEG_INFINITY {
        Some(ExactKey::NegInf)
    } else {
        Some(ExactKey::Finite(BigDecimal::from_f64_exact(value)?))
    }
}

/// Hashes a numeric value whose magnitude is past the `f64` range: an arbitrary-precision value
/// on its exact magnitude (so distinct huge keys land in distinct buckets), and a genuine
/// infinity on its sign. `as_f64` is the payload's already-computed (non-finite) double.
fn hash_beyond_double(data: &AtomicData, as_f64: f64, hasher: &mut impl Hasher) {
    match data {
        AtomicData::BigInteger(value) => {
            hash_exact_magnitude(value.is_negative(), value.magnitude(), 0, hasher);
        }
        AtomicData::BigDecimal(value) => {
            hash_exact_magnitude(
                value.is_negative(),
                value.unscaled().magnitude(),
                value.scale(),
                hasher,
            );
        }
        // A genuine `±∞` (an `xs:double`/`xs:float`); its sign keeps the two apart, and the tag
        // keeps both apart from every finite big magnitude above.
        _ => {
            "\u{0}inf".hash(hasher);
            as_f64.is_sign_positive().hash(hasher);
        }
    }
}

/// Hashes an arbitrary-precision value on its canonical `(sign, magnitude, scale)`, so that a
/// [`AtomicData::BigInteger`] and a numerically equal [`AtomicData::BigDecimal`] — one key — hash
/// identically.
fn hash_exact_magnitude(negative: bool, magnitude: &BigUint, scale: u32, hasher: &mut impl Hasher) {
    "\u{0}big".hash(hasher);
    negative.hash(hasher);
    magnitude.to_be_bytes().hash(hasher);
    scale.hash(hasher);
}

/// A hash that is *consistent* with [`same_key`]: whenever `same_key(a, b)` holds,
/// `same_key_hash(a) == same_key_hash(b)`.
///
/// Cross-type numeric keys make this subtler than it looks. Two numeric values are one key when
/// they are the same *value*, so `xs:integer 1`, `xs:decimal 1.0` and `xs:double 1.0E0` hash
/// alike, and so must an `xs:integer` and the `xs:double` it is exactly equal to — including the
/// large ones that only `xs:float` rounding, not the value, would merge. The hash therefore keys
/// on the coarsest float rounding, `xs:float`, which any two equal values share; where that
/// rounding is an infinity or a zero it falls through to the exact `xs:double`. A big value
/// beyond even the double range would all round to `±∞` there, funnelling unboundedly many
/// distinct keys into one bucket, so those are hashed on their exact magnitude instead. Unequal
/// values may of course still collide, which is all a bucket asks: [`crate::XdmMap`] confirms
/// every candidate with [`same_key`].
///
/// Date and time keys are hashed through their position on the timeline, matching the
/// normalization `same_key` compares them by, so `2020-01-01T00:00:00Z` and
/// `2020-01-01T09:00:00+09:00` — one key — also hash alike.
#[must_use]
pub fn same_key_hash(value: &AtomicValue) -> u64 {
    let mut hasher = std::collections::hash_map::DefaultHasher::new();
    let data = value.data();
    if data.is_numeric() {
        if data.is_nan() {
            "\u{0}nan".hash(&mut hasher);
        } else {
            "\u{0}num".hash(&mut hasher);
            let as_f32 = to_float(data).map_or(0.0_f32, f32::from);
            if as_f32.is_finite() && as_f32 != 0.0 {
                as_f32.to_bits().hash(&mut hasher);
            } else {
                // The float rounding is an infinity or a zero. Two values that are one key share
                // it, so they reach here together; hash the exact `xs:double` when it is finite
                // (this keeps large or tiny doubles, and every fixed-width integer and decimal,
                // apart). When even the double is non-finite the value is either a genuine
                // infinity or an arbitrary-precision magnitude past the double range: hash those
                // exactly, so distinct huge keys do not all collapse onto `±∞`'s bucket.
                let as_f64 = data.as_f64().unwrap_or(0.0);
                if as_f64.is_finite() {
                    (as_f64 + 0.0).to_bits().hash(&mut hasher);
                } else {
                    hash_beyond_double(data, as_f64, &mut hasher);
                }
            }
        }
        return hasher.finish();
    }
    // The duration family shares one value space and one same-key relation, so the three types
    // have to hash through the same widened form.
    if let Some(duration) = widen_duration(data) {
        "\u{0}dur".hash(&mut hasher);
        duration.to_be_bytes().hash(&mut hasher);
        return hasher.finish();
    }
    match data {
        AtomicData::Text(text) => {
            "\u{0}text".hash(&mut hasher);
            text.hash(&mut hasher);
        }
        AtomicData::Binary(bytes) => {
            "\u{0}bin".hash(&mut hasher);
            binary_family(value.atomic_type()).hash(&mut hasher);
            bytes.hash(&mut hasher);
        }
        other => {
            std::mem::discriminant(other).hash(&mut hasher);
            other.hash(&mut hasher);
        }
    }
    hasher.finish()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::QNameValue;

    fn parse(ty: AtomicType, lexical: &str) -> AtomicValue {
        AtomicValue::parse(ty, lexical).expect("valid lexical form")
    }

    #[test]
    fn numeric_promotion_crosses_types() -> Result<(), XdmError> {
        assert!(atomic_equal(
            &AtomicValue::integer(1_i64),
            &AtomicValue::decimal(1_i64)
        )?);
        assert!(atomic_equal(
            &AtomicValue::integer(1_i64),
            &AtomicValue::double(1.0)
        )?);
        assert!(atomic_equal(
            &AtomicValue::float(1.5_f32),
            &AtomicValue::double(1.5)
        )?);
        assert_eq!(
            compare_atomic(&AtomicValue::integer(1_i64), &AtomicValue::double(2.0))?,
            Some(Ordering::Less)
        );
        Ok(())
    }

    #[test]
    fn nan_is_unordered_under_eq_but_is_its_own_key() -> Result<(), XdmError> {
        let nan = AtomicValue::double(f64::NAN);
        assert!(!atomic_equal(&nan, &nan)?);
        assert_eq!(compare_atomic(&nan, &AtomicValue::integer(0_i64))?, None);
        assert!(same_key(&nan, &nan));
        assert_eq!(same_key_hash(&nan), same_key_hash(&nan));
        Ok(())
    }

    #[test]
    fn untyped_atomic_behaves_as_a_string_under_eq() -> Result<(), XdmError> {
        let untyped = AtomicValue::untyped_atomic("1");
        assert!(atomic_equal(&untyped, &AtomicValue::string("1"))?);
        assert!(atomic_equal(&untyped, &AtomicValue::any_uri("1"))?);
        // Comparing it with a number is a type error, just as `xs:string "1" eq 1` is.
        assert_eq!(
            atomic_equal(&untyped, &AtomicValue::integer(1_i64))
                .expect_err("not comparable")
                .code(),
            ErrorCode::Xpty0004
        );
        Ok(())
    }

    #[test]
    fn unrelated_types_are_a_type_error_under_eq_but_not_under_same_key() {
        let left = AtomicValue::integer(1_i64);
        let right = AtomicValue::string("1");
        let error = atomic_equal(&left, &right).expect_err("incomparable");
        assert_eq!(error.code(), ErrorCode::Xpty0004);
        assert!(!same_key(&left, &right));
    }

    #[test]
    fn same_key_hash_agrees_across_the_numeric_tower() {
        let values = [
            AtomicValue::integer(1_i64),
            AtomicValue::decimal(1_i64),
            AtomicValue::double(1.0),
            AtomicValue::float(1.0_f32),
        ];
        for pair in values.windows(2) {
            assert!(same_key(&pair[0], &pair[1]), "{:?}", pair);
            assert_eq!(same_key_hash(&pair[0]), same_key_hash(&pair[1]));
        }
        assert!(same_key(
            &AtomicValue::double(0.0),
            &AtomicValue::double(-0.0)
        ));
        assert_eq!(
            same_key_hash(&AtomicValue::double(0.0)),
            same_key_hash(&AtomicValue::double(-0.0))
        );
    }

    #[test]
    fn same_key_groups_the_text_types_and_separates_the_binary_ones() {
        assert!(same_key(
            &AtomicValue::string("x"),
            &AtomicValue::untyped_atomic("x")
        ));
        assert!(same_key(
            &AtomicValue::string("x"),
            &AtomicValue::any_uri("x")
        ));
        assert_eq!(
            same_key_hash(&AtomicValue::string("x")),
            same_key_hash(&AtomicValue::any_uri("x"))
        );
        assert!(!same_key(
            &AtomicValue::hex_binary(vec![1]),
            &AtomicValue::base64_binary(vec![1])
        ));
    }

    /// Whatever `same_key` joins, `same_key_hash` must not put in two buckets — otherwise a
    /// hashed map misses the entry entirely.
    #[test]
    fn same_key_implies_one_hash_across_the_exact_value_boundaries() {
        use crate::BigInteger;

        // A number ten to the four-hundredth, past even the `xs:double` range, as both an
        // integer and a decimal — one key, and the pair that exercises the exact-magnitude hash.
        let ten_pow_400 = {
            let ten = BigInteger::from_i64(10);
            let mut value = BigInteger::from_i64(1);
            for _ in 0..400 {
                value = value.checked_mul(&ten).expect("well within MAX_BITS");
            }
            value
        };
        let pairs = [
            // Exactly equal across the tower is one key, whatever type spells it.
            (AtomicValue::integer(1_i64), AtomicValue::float(1.0_f32)),
            (AtomicValue::double(0.0), AtomicValue::double(-0.0)),
            (AtomicValue::integer(0_i64), AtomicValue::float(-0.0_f32)),
            (
                AtomicValue::float(f32::INFINITY),
                AtomicValue::double(f64::INFINITY),
            ),
            // 2⁶⁴ escapes `i64` but is exact as a double.
            (
                AtomicValue::from_big_integer(BigInteger::from_i128(1_i128 << 64)),
                AtomicValue::double(18_446_744_073_709_551_616.0),
            ),
            (
                AtomicValue::from_big_integer(ten_pow_400.clone()),
                AtomicValue::from_big_decimal(super::BigDecimal::from_big_integer(ten_pow_400)),
            ),
        ];
        for (left, right) in pairs {
            assert!(same_key(&left, &right), "{left:?} vs {right:?}");
            assert_eq!(
                same_key_hash(&left),
                same_key_hash(&right),
                "{left:?} and {right:?} are one key but hash apart"
            );
        }

        // The map-key relation is *exact*, not the promoting `eq`: an integer and the float it
        // merely rounds to are different keys (QT3 `op:same-key/same-key-026`).
        assert!(!same_key(
            &AtomicValue::integer(16_777_217_i64),
            &AtomicValue::float(16_777_216.0_f32)
        ));
    }

    #[test]
    fn two_distinct_values_past_the_double_range_hash_apart() {
        use crate::BigInteger;

        // 10^400 and 10^400 + 1: distinct keys, both beyond the double range where a saturating
        // hash would collapse them together.
        let ten = BigInteger::from_i64(10);
        let mut base = BigInteger::from_i64(1);
        for _ in 0..400 {
            base = base.checked_mul(&ten).expect("in bounds");
        }
        let neighbour = base
            .checked_add(&BigInteger::from_i64(1))
            .expect("in bounds");
        let left = AtomicValue::from_big_integer(base);
        let right = AtomicValue::from_big_integer(neighbour);
        assert!(!same_key(&left, &right));
        assert_ne!(
            same_key_hash(&left),
            same_key_hash(&right),
            "distinct huge keys must not funnel into one bucket"
        );
    }

    /// The map-level consequence of the hash agreeing with the relation.
    #[test]
    fn a_map_finds_a_key_spelled_in_a_different_but_equal_type() {
        let mut map = crate::XdmMap::new();
        map.insert(
            AtomicValue::integer(42_i64),
            crate::Sequence::from(AtomicValue::string("found")),
        );
        // The same value spelled as an `xs:double` or `xs:decimal` finds the entry.
        assert!(map.contains_key(&AtomicValue::double(42.0)));
        assert!(map.contains_key(&AtomicValue::decimal(42_i64)));
        assert!(map.contains_key(&AtomicValue::integer(42_i64)));
        // A different value misses.
        assert!(!map.contains_key(&AtomicValue::integer(43_i64)));
    }

    /// `op:same-key` may not consult the dynamic context, so it uses XML Schema equality, under
    /// which a value carrying a timezone is never equal to one without — for *every* date/time
    /// type, the five Gregorian ones included.
    #[test]
    fn a_timezone_makes_a_different_key_for_every_date_time_type() {
        for (atomic_type, zoned, floating) in [
            (
                AtomicType::DateTime,
                "2015-10-10T00:00:00Z",
                "2015-10-10T00:00:00",
            ),
            (AtomicType::Date, "2015-10-10Z", "2015-10-10"),
            (AtomicType::Time, "00:00:00Z", "00:00:00"),
            (AtomicType::GYear, "2015Z", "2015"),
            (AtomicType::GYearMonth, "2015-10Z", "2015-10"),
            (AtomicType::GMonth, "--10Z", "--10"),
            (AtomicType::GMonthDay, "--10-10Z", "--10-10"),
            (AtomicType::GDay, "---10Z", "---10"),
        ] {
            let a = parse(atomic_type, zoned);
            let b = parse(atomic_type, floating);
            assert!(
                !same_key(&a, &b),
                "{atomic_type}: {zoned} and {floating} must be different keys"
            );
            // They are still `eq` under the implicit timezone, which is a different relation.
            assert_eq!(
                atomic_equal(&a, &b).ok(),
                Some(true),
                "{atomic_type}: {zoned} and {floating} are equal under the implicit timezone"
            );
            // And two spellings of the same zoned value stay one key.
            assert!(same_key(&a, &parse(atomic_type, zoned)));
        }
    }

    /// `op:same-key` compares date/times by their position on the timeline, so two spellings
    /// of one instant are one key — and have to be one bucket.
    #[test]
    fn same_key_hash_agrees_for_one_instant_written_in_two_timezones() {
        let cases = [
            (
                AtomicType::DateTime,
                "2020-01-01T00:00:00Z",
                "2020-01-01T09:00:00+09:00",
            ),
            (AtomicType::Time, "00:00:00Z", "09:00:00+09:00"),
            (AtomicType::Date, "2020-01-01Z", "2020-01-01+00:00"),
            (AtomicType::GYear, "2020Z", "2020+00:00"),
            (AtomicType::GYearMonth, "2020-01Z", "2020-01+00:00"),
        ];
        for (atomic_type, left, right) in cases {
            let a = parse(atomic_type, left);
            let b = parse(atomic_type, right);
            assert!(same_key(&a, &b), "{atomic_type}: {left} vs {right}");
            assert_eq!(
                same_key_hash(&a),
                same_key_hash(&b),
                "{atomic_type}: {left} and {right} are one key but hash apart"
            );
        }
    }

    /// The three duration types share one value space, so they share one key — and therefore
    /// one hash.
    #[test]
    fn the_duration_family_is_one_key_space() -> Result<(), XdmError> {
        let year_month = parse(AtomicType::YearMonthDuration, "P12M");
        let duration = parse(AtomicType::Duration, "P1Y");
        assert!(same_key(&year_month, &duration));
        assert_eq!(same_key_hash(&year_month), same_key_hash(&duration));

        let zero_months = parse(AtomicType::YearMonthDuration, "P0Y");
        let zero_seconds = parse(AtomicType::DayTimeDuration, "P0D");
        assert!(same_key(&zero_months, &zero_seconds));
        assert_eq!(same_key_hash(&zero_months), same_key_hash(&zero_seconds));

        assert!(!same_key(
            &parse(AtomicType::DayTimeDuration, "P1D"),
            &year_month
        ));
        Ok(())
    }

    #[test]
    fn date_time_comparison_applies_the_implicit_timezone() -> Result<(), XdmError> {
        let utc = parse(AtomicType::DateTime, "2020-01-01T00:00:00Z");
        let same_instant = parse(AtomicType::DateTime, "2020-01-01T01:00:00+01:00");
        assert!(atomic_equal(&utc, &same_instant)?);
        assert!(same_key(&utc, &same_instant));

        // F&O 3.1 substitutes the implicit timezone — UTC by default — for a
        // missing one, so a timezone-less value *is* `eq` to the matching UTC
        // one. (XML Schema's own order relation, which leaves the pair
        // incomparable, is available separately as `xsd_partial_cmp`.)
        let floating = parse(AtomicType::DateTime, "2020-01-01T00:00:00");
        assert!(atomic_equal(&utc, &floating)?);
        // `op:same-key` is the exception: it may not depend on the dynamic context, so it uses
        // XSD equality, under which a timezoned value is never equal to a floating one.
        assert!(!same_key(&utc, &floating));

        // A different implicit timezone shifts the timezone-less operand only.
        let jst = TimezoneOffset::new(9 * 60).expect("a valid offset");
        assert!(!atomic_equal_in_context(&utc, &floating, None, Some(jst))?);
        assert_eq!(
            compare_atomic_in_context(&floating, &utc, None, Some(jst))?,
            Some(Ordering::Less),
            "read in +09:00 the floating value is nine hours earlier"
        );
        Ok(())
    }

    #[test]
    fn date_and_time_comparisons_are_never_indeterminate() -> Result<(), XdmError> {
        for (atomic_type, floating, zoned) in [
            (AtomicType::Date, "2024-06-01", "2024-06-01Z"),
            (AtomicType::Time, "10:00:00", "10:00:00Z"),
            (AtomicType::GYearMonth, "2024-06", "2024-06Z"),
            (AtomicType::GYear, "2024", "2024Z"),
            (AtomicType::GMonthDay, "--06-01", "--06-01Z"),
            (AtomicType::GMonth, "--06", "--06Z"),
            (AtomicType::GDay, "---01", "---01Z"),
        ] {
            let a = parse(atomic_type, floating);
            let b = parse(atomic_type, zoned);
            assert_eq!(
                compare_atomic(&a, &b)?,
                Some(Ordering::Equal),
                "{atomic_type}: {floating} vs {zoned}"
            );
        }
        Ok(())
    }

    #[test]
    fn collations_are_honoured_and_validated() -> Result<(), XdmError> {
        let upper = AtomicValue::string("ABC");
        let lower = AtomicValue::string("abc");
        assert!(!atomic_equal(&upper, &lower)?);
        assert!(atomic_equal_with_collation(
            &upper,
            &lower,
            Some(HTML_ASCII_CASE_INSENSITIVE_COLLATION)
        )?);

        // A collation carried on the value is used when none is passed explicitly.
        let carried = upper
            .clone()
            .with_collation(HTML_ASCII_CASE_INSENSITIVE_COLLATION);
        assert!(atomic_equal(&carried, &lower)?);

        let error = atomic_equal_with_collation(&upper, &lower, Some("urn:nope"))
            .expect_err("unsupported collation");
        assert_eq!(error.code(), ErrorCode::Foch0002);
        Ok(())
    }

    #[test]
    fn qnames_compare_on_the_expanded_name() -> Result<(), XdmError> {
        let a = AtomicValue::qname(QNameValue::new(Some("p"), Some("urn:ns"), "x"));
        let b = AtomicValue::qname(QNameValue::new(Some("q"), Some("urn:ns"), "x"));
        assert!(atomic_equal(&a, &b)?);
        assert!(same_key(&a, &b));
        assert_eq!(same_key_hash(&a), same_key_hash(&b));
        Ok(())
    }

    #[test]
    fn durations_are_only_partially_ordered() -> Result<(), XdmError> {
        let one_month = parse(AtomicType::Duration, "P1M");
        let thirty_days = parse(AtomicType::Duration, "P30D");
        assert_eq!(compare_atomic(&one_month, &thirty_days)?, None);
        assert!(!atomic_equal(&one_month, &thirty_days)?);
        Ok(())
    }
}
