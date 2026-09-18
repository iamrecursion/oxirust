//! The arithmetic operators of [F&O 3.1 §4.2](https://www.w3.org/TR/xpath-functions-31/#op.numeric)
//! and [§8.4](https://www.w3.org/TR/xpath-functions-31/#durations-dates-times-arith): the `op:`
//! functions behind XPath's `+`, `-`, `*`, `div`, `idiv` and `mod`.
//!
//! # Type promotion
//!
//! Both operands are first *atomized and promoted*: an `xs:untypedAtomic` operand is cast to
//! `xs:double` (raising `FORG0001` if it is not a number), and two numeric operands are then
//! converted to the least common member of the tower
//! `xs:integer` → `xs:decimal` → `xs:float` → `xs:double`
//! ([XPath 3.1 §B.1](https://www.w3.org/TR/xpath-31/#promotion)). The result carries the
//! promoted type, with one exception the specification calls out: dividing two `xs:integer`
//! values yields an `xs:decimal`.
//!
//! # Durations, dates and times
//!
//! The same entry point implements the two dozen `op:` functions that add a duration to a date,
//! subtract two instants, or scale a duration by a number. Arithmetic on the *unrestricted*
//! `xs:duration` type is deliberately a type error: F&O defines operators only for
//! `xs:yearMonthDuration` and `xs:dayTimeDuration`, because the two components of a general
//! duration are not commensurable.

use crate::error::{FnError, FnErrorCode, FnResult};
use crate::numeric::round::round_f64_half_up;
use oxixml_infoset::{AtomicData, AtomicType, AtomicValue, BigDecimal, BigInteger};
use oxixml_xsd::{
    Date, DateTime, DayTimeDuration, Decimal, Double, Float, Integer, Time, YearMonthDuration,
};
use std::str::FromStr;

/// One of the six arithmetic operators XPath defines over atomic values.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ArithOp {
    /// `+`
    Add,
    /// `-`
    Subtract,
    /// `*`
    Multiply,
    /// `div`
    Divide,
    /// `idiv`
    IntegerDivide,
    /// `mod`
    Modulus,
}

impl ArithOp {
    /// The operator's XPath spelling.
    #[must_use]
    pub const fn symbol(self) -> &'static str {
        match self {
            Self::Add => "+",
            Self::Subtract => "-",
            Self::Multiply => "*",
            Self::Divide => "div",
            Self::IntegerDivide => "idiv",
            Self::Modulus => "mod",
        }
    }
}

/// The rung of the numeric tower a payload sits on.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
enum NumKind {
    Integer,
    Decimal,
    Float,
    Double,
}

fn num_kind(data: &AtomicData) -> Option<NumKind> {
    Some(match data {
        AtomicData::Integer(_) => NumKind::Integer,
        AtomicData::Decimal(_) => NumKind::Decimal,
        AtomicData::Float(_) => NumKind::Float,
        AtomicData::Double(_) => NumKind::Double,
        _ if data.as_big_integer().is_some() => NumKind::Integer,
        _ if data.as_big_decimal().is_some() => NumKind::Decimal,
        _ => return None,
    })
}

fn to_decimal(data: &AtomicData) -> Option<Decimal> {
    match data {
        AtomicData::Integer(value) => Some(Decimal::from(*value)),
        AtomicData::Decimal(value) => Some(*value),
        _ => None,
    }
}

/// The exact value of any integer/decimal payload — of either width — as a [`BigDecimal`], for
/// the arbitrary-precision decimal arithmetic that a fixed-width overflow promotes into.
fn to_big_decimal(data: &AtomicData) -> Option<BigDecimal> {
    match data {
        AtomicData::Integer(value) => Some(BigDecimal::from_fixed(Decimal::from(*value))),
        AtomicData::Decimal(value) => Some(BigDecimal::from_fixed(*value)),
        _ => {
            if let Some(value) = data.as_big_integer() {
                Some(BigDecimal::from_big_integer(value.clone()))
            } else {
                data.as_big_decimal().cloned()
            }
        }
    }
}

/// The value of an integer payload — of either width — as a [`BigInteger`].
fn to_big_integer(data: &AtomicData) -> Option<BigInteger> {
    match data {
        AtomicData::Integer(value) => Some(BigInteger::from(*value)),
        _ => data.as_big_integer().cloned(),
    }
}

fn to_float(data: &AtomicData) -> Option<Float> {
    Some(match data {
        AtomicData::Integer(value) => Float::from(*value),
        AtomicData::Decimal(value) => Float::from(*value),
        AtomicData::Float(value) => *value,
        AtomicData::Double(value) => Float::from(*value),
        // A big value routes through the same `f64 → Double → Float` step an `xs:double` takes.
        _ => Float::from(Double::from(data.as_f64()?)),
    })
}

fn to_double(data: &AtomicData) -> Option<Double> {
    Some(match data {
        AtomicData::Integer(value) => Double::from(*value),
        AtomicData::Decimal(value) => Double::from(*value),
        AtomicData::Float(value) => Double::from(*value),
        AtomicData::Double(value) => *value,
        _ => Double::from(data.as_f64()?),
    })
}

fn overflow() -> FnError {
    FnError::new(FnErrorCode::Foar0002, "numeric overflow or underflow")
}

fn division_by_zero() -> FnError {
    FnError::new(FnErrorCode::Foar0001, "division by zero")
}

fn type_error(op: ArithOp, left: &AtomicValue, right: &AtomicValue) -> FnError {
    FnError::new(
        FnErrorCode::Xpty0004,
        format!(
            "the operator `{}` is not defined for {} and {}",
            op.symbol(),
            left.atomic_type(),
            right.atomic_type()
        ),
    )
}

/// Casts an `xs:untypedAtomic` operand to `xs:double`, as XPath arithmetic requires; every other
/// value is returned unchanged.
///
/// # Errors
///
/// [`FnErrorCode::Forg0001`] when the lexical form is not a number.
pub fn promote_untyped(value: &AtomicValue) -> FnResult<AtomicValue> {
    if value.atomic_type() == AtomicType::UntypedAtomic {
        let lexical = value.as_str().unwrap_or_default();
        return Ok(AtomicValue::parse(AtomicType::Double, lexical)?);
    }
    Ok(value.clone())
}

/// Applies an arithmetic operator to two atomic values.
///
/// # Errors
///
/// * [`FnErrorCode::Xpty0004`] when the operator is not defined for the operand types;
/// * [`FnErrorCode::Foar0001`] on division by zero in the exact (integer and decimal) types;
/// * [`FnErrorCode::Foar0002`] on numeric overflow or underflow;
/// * [`FnErrorCode::Fodt0001`] / [`FnErrorCode::Fodt0002`] on date/time and duration overflow;
/// * [`FnErrorCode::Foca0005`] when a duration is scaled by `NaN`;
/// * [`FnErrorCode::Forg0001`] when an `xs:untypedAtomic` operand is not a number.
///
/// # Examples
///
/// ```
/// use oxixml_fn::numeric::{arithmetic, ArithOp};
/// use oxixml_infoset::{AtomicType, AtomicValue};
///
/// // The result carries the promoted type: integer + decimal is a decimal.
/// let sum = arithmetic(
///     ArithOp::Add,
///     &AtomicValue::integer(1_i64),
///     &AtomicValue::parse(AtomicType::Decimal, "2.5")?,
/// )?;
/// assert_eq!(sum.atomic_type(), AtomicType::Decimal);
/// assert_eq!(sum.string_value(), "3.5");
///
/// // Dividing two integers yields a decimal, not an integer.
/// let quotient = arithmetic(
///     ArithOp::Divide,
///     &AtomicValue::integer(3_i64),
///     &AtomicValue::integer(2_i64),
/// )?;
/// assert_eq!(quotient.atomic_type(), AtomicType::Decimal);
/// assert_eq!(quotient.string_value(), "1.5");
///
/// // ... whereas idiv truncates towards zero.
/// let idiv = arithmetic(
///     ArithOp::IntegerDivide,
///     &AtomicValue::integer(-3_i64),
///     &AtomicValue::integer(2_i64),
/// )?;
/// assert_eq!(idiv.string_value(), "-1");
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
pub fn arithmetic(op: ArithOp, left: &AtomicValue, right: &AtomicValue) -> FnResult<AtomicValue> {
    let left = promote_untyped(left)?;
    let right = promote_untyped(right)?;

    if let (Some(left_kind), Some(right_kind)) = (num_kind(left.data()), num_kind(right.data())) {
        return numeric_arithmetic(op, left.data(), right.data(), left_kind.max(right_kind));
    }
    temporal_arithmetic(op, &left, &right)
}

/// `op:numeric-unary-minus`.
///
/// # Errors
///
/// [`FnErrorCode::Xpty0004`] for a non-numeric argument, [`FnErrorCode::Foar0002`] when the
/// most negative `xs:integer` or `xs:decimal` is negated.
pub fn unary_minus(value: &AtomicValue) -> FnResult<AtomicValue> {
    let value = promote_untyped(value)?;
    let data = match value.data() {
        // Negating `i64::MIN` (or the most negative fixed decimal) overflows the fixed width, so
        // it promotes rather than erroring — `xs:integer` and `xs:decimal` are unbounded.
        AtomicData::Integer(number) => match number.checked_neg() {
            Some(negated) => AtomicData::Integer(negated),
            None => AtomicData::from_big_integer(BigInteger::from(*number).neg()),
        },
        AtomicData::Decimal(number) => match number.checked_neg() {
            Some(negated) => AtomicData::Decimal(negated),
            None => AtomicData::from_big_decimal(BigDecimal::from_fixed(*number).neg()),
        },
        AtomicData::Float(number) => AtomicData::Float(-*number),
        AtomicData::Double(number) => AtomicData::Double(-*number),
        other => {
            if let Some(number) = other.as_big_integer() {
                AtomicData::from_big_integer(number.neg())
            } else if let Some(number) = other.as_big_decimal() {
                AtomicData::from_big_decimal(number.neg())
            } else {
                return Err(FnError::new(
                    FnErrorCode::Xpty0004,
                    format!("unary `-` is not defined for {}", value.atomic_type()),
                ));
            }
        }
    };
    label(data)
}

/// `op:numeric-unary-plus`, which promotes an `xs:untypedAtomic` operand and is otherwise the
/// identity.
///
/// # Errors
///
/// [`FnErrorCode::Xpty0004`] for a non-numeric argument, [`FnErrorCode::Forg0001`] when an
/// `xs:untypedAtomic` operand is not a number.
pub fn unary_plus(value: &AtomicValue) -> FnResult<AtomicValue> {
    let value = promote_untyped(value)?;
    if num_kind(value.data()).is_none() {
        return Err(FnError::new(
            FnErrorCode::Xpty0004,
            format!("unary `+` is not defined for {}", value.atomic_type()),
        ));
    }
    Ok(value)
}

fn label(data: AtomicData) -> FnResult<AtomicValue> {
    let atomic_type = match &data {
        AtomicData::Integer(_) => AtomicType::Integer,
        AtomicData::Decimal(_) => AtomicType::Decimal,
        AtomicData::Float(_) => AtomicType::Float,
        AtomicData::Double(_) => AtomicType::Double,
        AtomicData::YearMonthDuration(_) => AtomicType::YearMonthDuration,
        AtomicData::DayTimeDuration(_) => AtomicType::DayTimeDuration,
        AtomicData::DateTime(_) => AtomicType::DateTime,
        AtomicData::Date(_) => AtomicType::Date,
        AtomicData::Time(_) => AtomicType::Time,
        _ if data.as_big_integer().is_some() => AtomicType::Integer,
        _ if data.as_big_decimal().is_some() => AtomicType::Decimal,
        _ => {
            return Err(FnError::new(
                FnErrorCode::Xpty0004,
                "the arithmetic result has no atomic type",
            ));
        }
    };
    Ok(AtomicValue::new(atomic_type, data)?)
}

fn numeric_arithmetic(
    op: ArithOp,
    left: &AtomicData,
    right: &AtomicData,
    common: NumKind,
) -> FnResult<AtomicValue> {
    match common {
        NumKind::Integer => integer_arithmetic(op, left, right),
        NumKind::Decimal => decimal_arithmetic(op, left, right),
        NumKind::Float => float_arithmetic(op, left, right),
        NumKind::Double => double_arithmetic(op, left, right),
    }
}

fn integer_arithmetic(op: ArithOp, left: &AtomicData, right: &AtomicData) -> FnResult<AtomicValue> {
    // Two fixed-width integers take the `i64` path unchanged; only an overflow, or an operand
    // that is already arbitrary-precision, falls through to the big path.
    if let (AtomicData::Integer(a), AtomicData::Integer(b)) = (left, right) {
        if let Some(result) = fixed_integer_op(op, *a, *b)? {
            return label(result);
        }
    }
    big_integer_arithmetic(op, left, right)
}

/// The fixed-width `xs:integer` operators. `Ok(None)` reports that the exact result overflowed the
/// `i64` and the caller must retry at arbitrary precision; `Err` is a genuine error (division by
/// zero), which no promotion can rescue.
fn fixed_integer_op(op: ArithOp, a: Integer, b: Integer) -> FnResult<Option<AtomicData>> {
    let zero = Integer::from(0_i64);
    Ok(match op {
        ArithOp::Add => a.checked_add(b).map(AtomicData::Integer),
        ArithOp::Subtract => a.checked_sub(b).map(AtomicData::Integer),
        ArithOp::Multiply => a.checked_mul(b).map(AtomicData::Integer),
        ArithOp::Divide => {
            if b == zero {
                return Err(division_by_zero());
            }
            Decimal::from(a)
                .checked_div(Decimal::from(b))
                .map(AtomicData::Decimal)
        }
        ArithOp::IntegerDivide => {
            if b == zero {
                return Err(division_by_zero());
            }
            a.checked_div(b).map(AtomicData::Integer)
        }
        ArithOp::Modulus => {
            if b == zero {
                return Err(division_by_zero());
            }
            a.checked_rem(b).map(AtomicData::Integer)
        }
    })
}

/// The `xs:integer` operators at arbitrary precision, demoting each result back to the fixed width
/// when it fits. A result past [`oxixml_support::bigint::BigUint::MAX_BITS`] is the only overflow
/// that survives here, and it becomes `FOAR0002`.
fn big_integer_arithmetic(
    op: ArithOp,
    left: &AtomicData,
    right: &AtomicData,
) -> FnResult<AtomicValue> {
    let missing = || FnError::new(FnErrorCode::Xpty0004, "an xs:integer operand was expected");
    let a = to_big_integer(left).ok_or_else(missing)?;
    let b = to_big_integer(right).ok_or_else(missing)?;
    let data = match op {
        ArithOp::Add => AtomicData::from_big_integer(a.checked_add(&b).ok_or_else(overflow)?),
        ArithOp::Subtract => AtomicData::from_big_integer(a.checked_sub(&b).ok_or_else(overflow)?),
        ArithOp::Multiply => AtomicData::from_big_integer(a.checked_mul(&b).ok_or_else(overflow)?),
        // Integer division yields a decimal, so it shares the decimal promotion path.
        ArithOp::Divide => return decimal_arithmetic(op, left, right),
        ArithOp::IntegerDivide => {
            if b.is_zero() {
                return Err(division_by_zero());
            }
            AtomicData::from_big_integer(a.div_rem(&b).ok_or_else(overflow)?.0)
        }
        ArithOp::Modulus => {
            if b.is_zero() {
                return Err(division_by_zero());
            }
            AtomicData::from_big_integer(a.div_rem(&b).ok_or_else(overflow)?.1)
        }
    };
    label(data)
}

fn decimal_arithmetic(op: ArithOp, left: &AtomicData, right: &AtomicData) -> FnResult<AtomicValue> {
    if let Some(result) = fixed_decimal_op(op, left, right)? {
        return label(result);
    }
    big_decimal_arithmetic(op, left, right)
}

/// The fixed-width `xs:decimal` operators, like [`fixed_integer_op`]. `Ok(None)` means either an
/// operand is already arbitrary-precision or the exact result overflowed the fixed decimal, both
/// of which the caller retries at arbitrary precision.
fn fixed_decimal_op(
    op: ArithOp,
    left: &AtomicData,
    right: &AtomicData,
) -> FnResult<Option<AtomicData>> {
    let (Some(a), Some(b)) = (to_decimal(left), to_decimal(right)) else {
        return Ok(None);
    };
    let zero = Decimal::from(0_i64);
    Ok(match op {
        ArithOp::Add => a.checked_add(b).map(AtomicData::Decimal),
        ArithOp::Subtract => a.checked_sub(b).map(AtomicData::Decimal),
        ArithOp::Multiply => a.checked_mul(b).map(AtomicData::Decimal),
        ArithOp::Divide => {
            if b == zero {
                return Err(division_by_zero());
            }
            a.checked_div(b).map(AtomicData::Decimal)
        }
        ArithOp::IntegerDivide => {
            if b == zero {
                return Err(division_by_zero());
            }
            fixed_decimal_integer_divide(a, b)
        }
        ArithOp::Modulus => {
            if b == zero {
                return Err(division_by_zero());
            }
            a.checked_rem(b).map(AtomicData::Decimal)
        }
    })
}

/// The fixed-width part of `xs:decimal idiv`: truncate the quotient towards zero into an
/// `xs:integer`, reporting any overflow along the way as `None` so the caller promotes.
///
/// The quotient is taken *exactly*, never by truncating the result of `div`. `Decimal` is a
/// fixed-point type that keeps `value × 10¹⁸` in an `i128` — the encoding
/// [`Decimal::to_be_bytes`] hands out — so the exact quotient of two decimals is the quotient of
/// those two scaled integers, and `i128` division already truncates towards zero, which is what
/// `idiv` asks for. Dividing first and truncating afterwards is *not* the same operation:
/// `op:numeric-divide` rounds half-to-even at the eighteenth fractional digit, so a quotient a
/// hair below an integer — `1999999999999999999.999999999999999999 idiv 1000000000000000000` —
/// rounds up to that integer and then floors one too high.
fn fixed_decimal_integer_divide(a: Decimal, b: Decimal) -> Option<AtomicData> {
    let scaled_dividend = i128::from_be_bytes(a.to_be_bytes());
    let scaled_divisor = i128::from_be_bytes(b.to_be_bytes());
    // `checked_div` also covers `i128::MIN / -1`, whose quotient is one past `i128::MAX`; the
    // caller retries that at arbitrary precision.
    let quotient = scaled_dividend.checked_div(scaled_divisor)?;
    i64::try_from(quotient)
        .ok()
        .map(|quotient| AtomicData::Integer(Integer::from(quotient)))
}

/// The `xs:decimal` operators at arbitrary precision, demoting on fit. A result past
/// [`oxixml_support::bigint::BigUint::MAX_BITS`] becomes `FOAR0002`.
fn big_decimal_arithmetic(
    op: ArithOp,
    left: &AtomicData,
    right: &AtomicData,
) -> FnResult<AtomicValue> {
    let missing = || FnError::new(FnErrorCode::Xpty0004, "an xs:decimal operand was expected");
    let a = to_big_decimal(left).ok_or_else(missing)?;
    let b = to_big_decimal(right).ok_or_else(missing)?;
    let data = match op {
        ArithOp::Add => AtomicData::from_big_decimal(a.checked_add(&b).ok_or_else(overflow)?),
        ArithOp::Subtract => AtomicData::from_big_decimal(a.checked_sub(&b).ok_or_else(overflow)?),
        ArithOp::Multiply => AtomicData::from_big_decimal(a.checked_mul(&b).ok_or_else(overflow)?),
        ArithOp::Divide => {
            if b.is_zero() {
                return Err(division_by_zero());
            }
            AtomicData::from_big_decimal(a.checked_div(&b).ok_or_else(overflow)?)
        }
        ArithOp::IntegerDivide => {
            if b.is_zero() {
                return Err(division_by_zero());
            }
            AtomicData::from_big_integer(a.checked_integer_divide(&b).ok_or_else(overflow)?)
        }
        ArithOp::Modulus => {
            if b.is_zero() {
                return Err(division_by_zero());
            }
            AtomicData::from_big_decimal(a.checked_rem(&b).ok_or_else(overflow)?)
        }
    };
    label(data)
}

fn float_arithmetic(op: ArithOp, left: &AtomicData, right: &AtomicData) -> FnResult<AtomicValue> {
    let missing = || FnError::new(FnErrorCode::Xpty0004, "an xs:float operand was expected");
    let a = to_float(left).ok_or_else(missing)?;
    let b = to_float(right).ok_or_else(missing)?;
    let data = match op {
        ArithOp::Add => AtomicData::Float(a + b),
        ArithOp::Subtract => AtomicData::Float(a - b),
        ArithOp::Multiply => AtomicData::Float(a * b),
        ArithOp::Divide => AtomicData::Float(a / b),
        ArithOp::IntegerDivide => {
            return integer_divide_floating(f64::from(a), f64::from(b));
        }
        ArithOp::Modulus => AtomicData::Float(Float::from(f32::from(a) % f32::from(b))),
    };
    label(data)
}

fn double_arithmetic(op: ArithOp, left: &AtomicData, right: &AtomicData) -> FnResult<AtomicValue> {
    let missing = || FnError::new(FnErrorCode::Xpty0004, "an xs:double operand was expected");
    let a = to_double(left).ok_or_else(missing)?;
    let b = to_double(right).ok_or_else(missing)?;
    let data = match op {
        ArithOp::Add => AtomicData::Double(a + b),
        ArithOp::Subtract => AtomicData::Double(a - b),
        ArithOp::Multiply => AtomicData::Double(a * b),
        ArithOp::Divide => AtomicData::Double(a / b),
        ArithOp::IntegerDivide => {
            return integer_divide_floating(f64::from(a), f64::from(b));
        }
        ArithOp::Modulus => AtomicData::Double(Double::from(f64::from(a) % f64::from(b))),
    };
    label(data)
}

fn integer_divide_floating(a: f64, b: f64) -> FnResult<AtomicValue> {
    if b == 0.0 {
        return Err(division_by_zero());
    }
    if a.is_nan() || b.is_nan() || a.is_infinite() {
        return Err(overflow());
    }
    let quotient = (a / b).trunc();
    let integer = Integer::try_from(Double::from(quotient)).map_err(|_| overflow())?;
    label(AtomicData::Integer(integer))
}

fn duration_overflow() -> FnError {
    FnError::new(
        FnErrorCode::Fodt0002,
        "overflow or underflow in a duration operation",
    )
}

fn date_time_overflow() -> FnError {
    FnError::new(
        FnErrorCode::Fodt0001,
        "overflow or underflow in a date/time operation",
    )
}

fn total_months(duration: YearMonthDuration) -> i64 {
    duration.years().saturating_mul(12) + duration.months()
}

/// The `xs:double` value of a numeric operand used to scale a duration.
fn scaling_factor(data: &AtomicData) -> Option<f64> {
    to_double(data).map(f64::from)
}

fn scaled_by_nan() -> FnError {
    FnError::new(FnErrorCode::Foca0005, "a duration cannot be scaled by NaN")
}

/// `op:multiply-yearMonthDuration` and `op:divide-yearMonthDuration`: scale the length in
/// months and apply `fn:round` to the product.
fn scale_year_month(
    duration: YearMonthDuration,
    factor: f64,
    divide: bool,
) -> FnResult<AtomicData> {
    if factor.is_nan() {
        return Err(scaled_by_nan());
    }
    let months = f64::from(Double::from(Integer::from(total_months(duration))));
    let scaled = if divide {
        if factor == 0.0 {
            // The specification says the result overflows rather than being a division error.
            return Err(duration_overflow());
        }
        if factor.is_infinite() {
            return Ok(AtomicData::YearMonthDuration(YearMonthDuration::new(0_i64)));
        }
        months / factor
    } else {
        if factor.is_infinite() {
            return Err(duration_overflow());
        }
        months * factor
    };
    let rounded = round_f64_half_up(scaled);
    // The rounded product is already a whole number of months, so it is taken straight to `i64`.
    // Routing it through `xs:decimal` would scale it by 10^18 in binary floating point first,
    // and that rescaling is not exact: `Decimal::try_from(Double)` turns 4093 into
    // 4092.999999999999737856, whose truncation is 4092 — which is how
    // `xs:yearMonthDuration('P17Y') * 47` came out one month short (`format-date-006`).
    if !(-9.223_372_036_854_775e18..=9.223_372_036_854_775e18).contains(&rounded) {
        return Err(duration_overflow());
    }
    #[allow(clippy::cast_possible_truncation)]
    let months = rounded as i64;
    Ok(AtomicData::YearMonthDuration(YearMonthDuration::new(
        months,
    )))
}

/// `op:multiply-dayTimeDuration` and `op:divide-dayTimeDuration`: scale the length in seconds,
/// exactly when the factor is an exact numeric type and in `xs:double` arithmetic otherwise.
fn scale_day_time(
    duration: DayTimeDuration,
    factor: &AtomicData,
    divide: bool,
) -> FnResult<AtomicData> {
    let seconds = duration.as_seconds();
    let zero = Decimal::from(0_i64);
    if let Some(exact) = to_decimal(factor) {
        if divide && exact == zero {
            return Err(duration_overflow());
        }
        let scaled = if divide {
            seconds.checked_div(exact)
        } else {
            seconds.checked_mul(exact)
        }
        .ok_or_else(duration_overflow)?;
        return Ok(AtomicData::DayTimeDuration(DayTimeDuration::new(scaled)));
    }
    let factor = scaling_factor(factor).ok_or_else(duration_overflow)?;
    if factor.is_nan() {
        return Err(scaled_by_nan());
    }
    if factor.is_infinite() {
        return if divide {
            Ok(AtomicData::DayTimeDuration(DayTimeDuration::new(zero)))
        } else {
            Err(duration_overflow())
        };
    }
    if divide && factor == 0.0 {
        return Err(duration_overflow());
    }
    let seconds = f64::from(Double::from(seconds));
    let scaled = if divide {
        seconds / factor
    } else {
        seconds * factor
    };
    let scaled = match decimal_from_f64(scaled) {
        Some(value) => value,
        // A magnitude below 10^-18 is not an overflow: `xs:decimal` has no digit that small, and
        // the nearest value it can represent — the one F&O 3.1 §8.4 says the result is rounded
        // to — is zero. Dividing a duration by 1.8e308 gives such a value.
        None if scaled.abs() < 1e-18 => zero,
        None => return Err(duration_overflow()),
    };
    Ok(AtomicData::DayTimeDuration(DayTimeDuration::new(scaled)))
}

/// Converts an [`f64`] to an [`Decimal`] through its *shortest round-tripping* decimal
/// representation.
///
/// Going via the lexical form rather than through a binary rescaling matters: scaling the raw
/// significand by `10^18` is not exact, and would turn `16380.0` into
/// `16380.000000000001048576`.
fn decimal_from_f64(value: f64) -> Option<Decimal> {
    if !value.is_finite() {
        return None;
    }
    decimal_from_lexical(&oxixml_text::num::xsd_canonical_double(value))
}

fn decimal_from_lexical(lexical: &str) -> Option<Decimal> {
    let (mantissa, exponent) = match lexical.split_once(['E', 'e']) {
        Some((mantissa, exponent)) => (mantissa, exponent.parse::<i32>().ok()?),
        None => (lexical, 0),
    };
    let negative = mantissa.starts_with('-');
    let mantissa = mantissa.trim_start_matches(['-', '+']);
    let (integer_part, fraction_part) = mantissa.split_once('.').unwrap_or((mantissa, ""));
    let digits = format!("{integer_part}{fraction_part}");
    if digits.is_empty() || !digits.bytes().all(|byte| byte.is_ascii_digit()) {
        return None;
    }
    let mut unscaled: i128 = digits.parse().ok()?;
    if negative {
        unscaled = unscaled.checked_neg()?;
    }
    let scale = exponent.checked_sub(i32::try_from(fraction_part.len()).ok()?)?;
    if scale >= 0 {
        let factor = 10_i128.checked_pow(u32::try_from(scale).ok()?)?;
        return Decimal::new(unscaled.checked_mul(factor)?, 0).ok();
    }
    let fraction_digits = u32::try_from(-scale).ok()?;
    if fraction_digits > 18 {
        // An xs:decimal carries eighteen fractional digits; the rest are beyond its precision.
        let divisor = 10_i128.checked_pow(fraction_digits - 18)?;
        return Decimal::new(unscaled / divisor, 18).ok();
    }
    Decimal::new(unscaled, fraction_digits).ok()
}

fn temporal_arithmetic(
    op: ArithOp,
    left: &AtomicValue,
    right: &AtomicValue,
) -> FnResult<AtomicValue> {
    use AtomicData::YearMonthDuration as YMD;
    use AtomicData::{Date as D, DateTime as DT, DayTimeDuration as DTD, Time as T};

    let data = match (op, left.data(), right.data()) {
        // op:add-yearMonthDurations / op:subtract-yearMonthDurations
        (ArithOp::Add, YMD(a), YMD(b)) => YMD(a.checked_add(*b).ok_or_else(duration_overflow)?),
        (ArithOp::Subtract, YMD(a), YMD(b)) => {
            YMD(a.checked_sub(*b).ok_or_else(duration_overflow)?)
        }
        // op:add-dayTimeDurations / op:subtract-dayTimeDurations
        (ArithOp::Add, DTD(a), DTD(b)) => DTD(a.checked_add(*b).ok_or_else(duration_overflow)?),
        (ArithOp::Subtract, DTD(a), DTD(b)) => {
            DTD(a.checked_sub(*b).ok_or_else(duration_overflow)?)
        }
        // op:multiply-* and op:divide-*
        (ArithOp::Multiply, YMD(a), other) if num_kind(other).is_some() => scale_year_month(
            *a,
            scaling_factor(other).ok_or_else(duration_overflow)?,
            false,
        )?,
        (ArithOp::Multiply, other, YMD(a)) if num_kind(other).is_some() => scale_year_month(
            *a,
            scaling_factor(other).ok_or_else(duration_overflow)?,
            false,
        )?,
        (ArithOp::Divide, YMD(a), other) if num_kind(other).is_some() => scale_year_month(
            *a,
            scaling_factor(other).ok_or_else(duration_overflow)?,
            true,
        )?,
        (ArithOp::Multiply, DTD(a), other) if num_kind(other).is_some() => {
            scale_day_time(*a, other, false)?
        }
        (ArithOp::Multiply, other, DTD(a)) if num_kind(other).is_some() => {
            scale_day_time(*a, other, false)?
        }
        (ArithOp::Divide, DTD(a), other) if num_kind(other).is_some() => {
            scale_day_time(*a, other, true)?
        }
        // op:divide-yearMonthDuration-by-yearMonthDuration
        (ArithOp::Divide, YMD(a), YMD(b)) => {
            let divisor = total_months(*b);
            if divisor == 0 {
                return Err(division_by_zero());
            }
            AtomicData::Decimal(
                Decimal::from(total_months(*a))
                    .checked_div(Decimal::from(divisor))
                    .ok_or_else(duration_overflow)?,
            )
        }
        // op:divide-dayTimeDuration-by-dayTimeDuration
        (ArithOp::Divide, DTD(a), DTD(b)) => {
            if b.as_seconds() == Decimal::from(0_i64) {
                return Err(division_by_zero());
            }
            AtomicData::Decimal(
                a.as_seconds()
                    .checked_div(b.as_seconds())
                    .ok_or_else(duration_overflow)?,
            )
        }
        // op:subtract-dateTimes / dates / times
        (ArithOp::Subtract, DT(a), DT(b)) => {
            DTD(a.checked_sub(*b).ok_or_else(date_time_overflow)?)
        }
        (ArithOp::Subtract, D(a), D(b)) => DTD(a.checked_sub(*b).ok_or_else(date_time_overflow)?),
        (ArithOp::Subtract, T(a), T(b)) => DTD(a.checked_sub(*b).ok_or_else(date_time_overflow)?),
        // op:add-yearMonthDuration-to-dateTime and friends, both operand orders
        (ArithOp::Add, DT(a), YMD(b)) | (ArithOp::Add, YMD(b), DT(a)) => DT(a
            .checked_add_year_month_duration(*b)
            .ok_or_else(date_time_overflow)?),
        (ArithOp::Add, DT(a), DTD(b)) | (ArithOp::Add, DTD(b), DT(a)) => DT(a
            .checked_add_day_time_duration(*b)
            .ok_or_else(date_time_overflow)?),
        (ArithOp::Add, D(a), YMD(b)) | (ArithOp::Add, YMD(b), D(a)) => D(a
            .checked_add_year_month_duration(*b)
            .ok_or_else(date_time_overflow)?),
        (ArithOp::Add, D(a), DTD(b)) | (ArithOp::Add, DTD(b), D(a)) => D(a
            .checked_add_day_time_duration(*b)
            .ok_or_else(date_time_overflow)?),
        (ArithOp::Add, T(a), DTD(b)) | (ArithOp::Add, DTD(b), T(a)) => T(a
            .checked_add_day_time_duration(*b)
            .ok_or_else(date_time_overflow)?),
        // op:subtract-yearMonthDuration-from-dateTime and friends
        (ArithOp::Subtract, DT(a), YMD(b)) => DT(a
            .checked_sub_year_month_duration(*b)
            .ok_or_else(date_time_overflow)?),
        (ArithOp::Subtract, DT(a), DTD(b)) => DT(a
            .checked_sub_day_time_duration(*b)
            .ok_or_else(date_time_overflow)?),
        (ArithOp::Subtract, D(a), YMD(b)) => D(a
            .checked_sub_year_month_duration(*b)
            .ok_or_else(date_time_overflow)?),
        (ArithOp::Subtract, D(a), DTD(b)) => D(a
            .checked_sub_day_time_duration(*b)
            .ok_or_else(date_time_overflow)?),
        (ArithOp::Subtract, T(a), DTD(b)) => T(a
            .checked_sub_day_time_duration(*b)
            .ok_or_else(date_time_overflow)?),
        _ => return Err(type_error(op, left, right)),
    };
    label(data)
}

/// `op:add-yearMonthDurations` and its eleven siblings, exposed under the names an evaluator
/// binds them to. This is a thin alias for [`arithmetic`] with the corresponding operator.
///
/// # Errors
///
/// Whatever [`arithmetic`] raises.
pub fn add(left: &AtomicValue, right: &AtomicValue) -> FnResult<AtomicValue> {
    arithmetic(ArithOp::Add, left, right)
}

/// Subtraction; see [`add`].
///
/// # Errors
///
/// Whatever [`arithmetic`] raises.
pub fn subtract(left: &AtomicValue, right: &AtomicValue) -> FnResult<AtomicValue> {
    arithmetic(ArithOp::Subtract, left, right)
}

/// Multiplication; see [`add`].
///
/// # Errors
///
/// Whatever [`arithmetic`] raises.
pub fn multiply(left: &AtomicValue, right: &AtomicValue) -> FnResult<AtomicValue> {
    arithmetic(ArithOp::Multiply, left, right)
}

/// Division; see [`add`].
///
/// # Errors
///
/// Whatever [`arithmetic`] raises.
pub fn divide(left: &AtomicValue, right: &AtomicValue) -> FnResult<AtomicValue> {
    arithmetic(ArithOp::Divide, left, right)
}

/// Integer division; see [`add`].
///
/// # Errors
///
/// Whatever [`arithmetic`] raises.
pub fn integer_divide(left: &AtomicValue, right: &AtomicValue) -> FnResult<AtomicValue> {
    arithmetic(ArithOp::IntegerDivide, left, right)
}

/// Modulus; see [`add`].
///
/// # Errors
///
/// Whatever [`arithmetic`] raises.
pub fn modulus(left: &AtomicValue, right: &AtomicValue) -> FnResult<AtomicValue> {
    arithmetic(ArithOp::Modulus, left, right)
}

/// `fn:dateTime`: combines a date and a time into an `xs:dateTime`.
///
/// # Errors
///
/// [`FnErrorCode::Forg0008`] when both arguments carry a timezone and the two disagree,
/// [`FnErrorCode::Fodt0001`] on overflow.
pub fn date_time_from_parts(date: Date, time: Time) -> FnResult<DateTime> {
    let timezone = match (date.timezone_offset(), time.timezone_offset()) {
        (Some(left), Some(right)) if left != right => {
            return Err(FnError::new(
                FnErrorCode::Forg0008,
                "fn:dateTime was given a date and a time in different timezones",
            ));
        }
        (Some(zone), _) | (_, Some(zone)) => Some(zone),
        (None, None) => None,
    };
    let mut lexical = format!(
        "{}-{:02}-{:02}T{:02}:{:02}:{}",
        format_year(date.year()),
        date.month(),
        date.day(),
        time.hour(),
        time.minute(),
        format_seconds(time.second()),
    );
    if let Some(zone) = timezone {
        lexical.push_str(&zone.to_string());
    }
    DateTime::from_str(&lexical).map_err(|error| {
        FnError::new(
            FnErrorCode::Fodt0001,
            format!("fn:dateTime produced the invalid value `{lexical}`: {error}"),
        )
    })
}

/// Formats a year in the `xs:dateTime` lexical space: at least four digits, with a leading `-`
/// for years before the common era.
fn format_year(year: i64) -> String {
    if year < 0 {
        format!("-{:04}", year.unsigned_abs())
    } else {
        format!("{year:04}")
    }
}

/// Formats a seconds component with the two integer digits the lexical space requires.
fn format_seconds(seconds: Decimal) -> String {
    let rendered = seconds.to_string();
    let integer_digits = rendered
        .split('.')
        .next()
        .map(str::len)
        .unwrap_or(rendered.len());
    if integer_digits < 2 {
        format!("0{rendered}")
    } else {
        rendered
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::str::FromStr;

    fn parse(atomic_type: AtomicType, lexical: &str) -> AtomicValue {
        AtomicValue::parse(atomic_type, lexical).expect("a valid lexical form")
    }

    #[test]
    fn integer_operators_follow_the_specification() -> FnResult<()> {
        let ten = AtomicValue::integer(10_i64);
        let three = AtomicValue::integer(3_i64);
        assert_eq!(add(&ten, &three)?.string_value(), "13");
        assert_eq!(subtract(&ten, &three)?.string_value(), "7");
        assert_eq!(multiply(&ten, &three)?.string_value(), "30");
        assert_eq!(integer_divide(&ten, &three)?.string_value(), "3");
        assert_eq!(modulus(&ten, &three)?.string_value(), "1");
        let quotient = divide(&ten, &three)?;
        assert_eq!(quotient.atomic_type(), AtomicType::Decimal);
        Ok(())
    }

    #[test]
    fn idiv_examples_from_the_specification() -> FnResult<()> {
        let cases = [
            (-3_i64, 2_i64, "-1"),
            (3, -2, "-1"),
            (-3, -2, "1"),
            (10, 3, "3"),
        ];
        for (a, b, expected) in cases {
            let result = integer_divide(&AtomicValue::integer(a), &AtomicValue::integer(b))?;
            assert_eq!(result.string_value(), expected, "{a} idiv {b}");
        }
        Ok(())
    }

    /// `idiv` truncates the *exact* quotient, so a dividend a hair below a multiple of the
    /// divisor still answers the smaller integer. Taking `div` first would round the quotient
    /// half-to-even at the eighteenth fractional digit and hand the truncation an integer that
    /// the exact quotient never reaches (QT3 `op/numeric-integer-divide`).
    #[test]
    fn idiv_truncates_the_exact_quotient_not_the_rounded_one() -> FnResult<()> {
        let cases = [
            (
                "1999999999999999999.999999999999999999",
                "1000000000000000000",
                "1",
            ),
            (
                "-1999999999999999999.999999999999999999",
                "1000000000000000000",
                "-1",
            ),
            (
                "1999999999999999999.999999999999999999",
                "-1000000000000000000",
                "-1",
            ),
            (
                "-1999999999999999999.999999999999999999",
                "-1000000000000000000",
                "1",
            ),
            ("0.999999999999999999", "1", "0"),
            ("-0.999999999999999999", "1", "0"),
            ("2.999999999999999999", "1.5", "1"),
            ("-2.999999999999999999", "1.5", "-1"),
        ];
        for (a, b, expected) in cases {
            let result = integer_divide(
                &parse(AtomicType::Decimal, a),
                &parse(AtomicType::Decimal, b),
            )?;
            assert_eq!(result.atomic_type(), AtomicType::Integer, "{a} idiv {b}");
            assert_eq!(result.string_value(), expected, "{a} idiv {b}");
        }
        // `div` still rounds at the eighteenth fractional digit, which is what it is defined to
        // do; the two operators simply do not share a code path any more.
        assert_eq!(
            divide(
                &parse(
                    AtomicType::Decimal,
                    "1999999999999999999.999999999999999999"
                ),
                &parse(AtomicType::Decimal, "1000000000000000000"),
            )?
            .string_value(),
            "2"
        );
        Ok(())
    }

    /// A quotient too wide for the fixed-width `xs:integer` is promoted to arbitrary precision
    /// rather than silently truncated to the fixed width.
    #[test]
    fn idiv_promotes_a_quotient_past_the_fixed_width_integer() -> FnResult<()> {
        let quotient = integer_divide(
            &parse(AtomicType::Decimal, "99999999999999999999.5"),
            &parse(AtomicType::Decimal, "0.000000000000000001"),
        )?;
        assert_eq!(quotient.atomic_type(), AtomicType::Integer);
        assert_eq!(
            quotient.string_value(),
            "99999999999999999999500000000000000000"
        );
        Ok(())
    }

    #[test]
    fn mod_examples_from_the_specification() -> FnResult<()> {
        assert_eq!(
            modulus(&AtomicValue::integer(5_i64), &AtomicValue::integer(3_i64))?.string_value(),
            "2"
        );
        assert_eq!(
            modulus(&AtomicValue::integer(-5_i64), &AtomicValue::integer(3_i64))?.string_value(),
            "-2"
        );
        assert_eq!(
            modulus(&AtomicValue::integer(5_i64), &AtomicValue::integer(-3_i64))?.string_value(),
            "2"
        );
        Ok(())
    }

    #[test]
    fn division_by_zero_is_exact_only_for_exact_types() -> FnResult<()> {
        let error = divide(&AtomicValue::integer(1_i64), &AtomicValue::integer(0_i64))
            .expect_err("integer division by zero");
        assert!(error.is(FnErrorCode::Foar0001));

        let infinite = divide(&AtomicValue::double(1.0), &AtomicValue::double(0.0))?;
        assert_eq!(infinite.string_value(), "INF");
        Ok(())
    }

    #[test]
    fn untyped_atomic_operands_are_cast_to_double() -> FnResult<()> {
        let sum = add(
            &AtomicValue::untyped_atomic("1.5"),
            &AtomicValue::integer(1_i64),
        )?;
        assert_eq!(sum.atomic_type(), AtomicType::Double);
        assert_eq!(sum.string_value(), "2.5");
        assert!(
            add(
                &AtomicValue::untyped_atomic("x"),
                &AtomicValue::integer(1_i64)
            )
            .is_err()
        );
        Ok(())
    }

    #[test]
    fn duration_arithmetic_follows_the_specification() -> FnResult<()> {
        let p2y11m = parse(AtomicType::YearMonthDuration, "P2Y11M");
        let p3y3m = parse(AtomicType::YearMonthDuration, "P3Y3M");
        assert_eq!(add(&p2y11m, &p3y3m)?.string_value(), "P6Y2M");
        assert_eq!(subtract(&p3y3m, &p2y11m)?.string_value(), "P4M");

        let scaled = multiply(&p2y11m, &AtomicValue::double(2.3))?;
        assert_eq!(scaled.string_value(), "P6Y9M");

        let divided = divide(&p2y11m, &AtomicValue::double(1.5))?;
        assert_eq!(divided.string_value(), "P1Y11M");

        let ratio = divide(
            &parse(AtomicType::YearMonthDuration, "P3Y4M"),
            &parse(AtomicType::YearMonthDuration, "-P1Y4M"),
        )?;
        assert_eq!(ratio.atomic_type(), AtomicType::Decimal);
        assert_eq!(ratio.string_value(), "-2.5");
        Ok(())
    }

    #[test]
    fn day_time_duration_arithmetic() -> FnResult<()> {
        let a = parse(AtomicType::DayTimeDuration, "P2DT12H5M");
        let b = parse(AtomicType::DayTimeDuration, "P5DT12H");
        assert_eq!(add(&a, &b)?.string_value(), "P8DT5M");
        assert_eq!(subtract(&a, &b)?.string_value(), "-P2DT23H55M");
        assert_eq!(
            multiply(
                &parse(AtomicType::DayTimeDuration, "PT2H10M"),
                &AtomicValue::double(2.1)
            )?
            .string_value(),
            "PT4H33M"
        );
        let ratio = divide(
            &parse(AtomicType::DayTimeDuration, "P2DT53M11S"),
            &parse(AtomicType::DayTimeDuration, "P1DT10H"),
        )?;
        assert_eq!(ratio.atomic_type(), AtomicType::Decimal);
        Ok(())
    }

    #[test]
    fn date_time_arithmetic() -> FnResult<()> {
        let start = parse(AtomicType::DateTime, "2000-10-30T11:12:00");
        let year_month = parse(AtomicType::YearMonthDuration, "P1Y2M");
        assert_eq!(
            add(&start, &year_month)?.string_value(),
            "2001-12-30T11:12:00"
        );
        assert_eq!(
            subtract(&start, &year_month)?.string_value(),
            "1999-08-30T11:12:00"
        );

        let difference = subtract(
            &parse(AtomicType::Date, "2000-10-30"),
            &parse(AtomicType::Date, "1999-11-28"),
        )?;
        assert_eq!(difference.atomic_type(), AtomicType::DayTimeDuration);
        assert_eq!(difference.string_value(), "P337D");
        Ok(())
    }

    #[test]
    fn unrestricted_durations_have_no_operators() {
        let duration = parse(AtomicType::Duration, "P1Y1D");
        let error = add(&duration, &duration).expect_err("xs:duration has no `+`");
        assert!(error.is(FnErrorCode::Xpty0004));
    }

    #[test]
    fn fn_date_time_combines_the_two_halves() -> FnResult<()> {
        let date = Date::from_str("1999-12-31")
            .map_err(|error| FnError::new(FnErrorCode::Forg0001, error.to_string()))?;
        let time = Time::from_str("12:00:00")
            .map_err(|error| FnError::new(FnErrorCode::Forg0001, error.to_string()))?;
        assert_eq!(
            date_time_from_parts(date, time)?.to_string(),
            "1999-12-31T12:00:00"
        );
        Ok(())
    }
}
