//! The numeric functions of [F&O 3.1 §4.4](https://www.w3.org/TR/xpath-functions-31/#numeric-value-functions):
//! `fn:abs`, `fn:ceiling`, `fn:floor`, `fn:round` and `fn:round-half-to-even`.
//!
//! Every one of them returns a value of the *same primitive numeric type* as its argument:
//! `xs:integer` in, `xs:integer` out (including for the `xs:integer` restrictions such as
//! `xs:short`), `xs:decimal` in, `xs:decimal` out, and likewise for `xs:float` and `xs:double`.

use crate::error::{FnError, FnErrorCode, FnResult};
use oxixml_infoset::{AtomicData, AtomicType, AtomicValue, BigDecimal};
use oxixml_xsd::{Decimal, Double, Float, Integer};

/// The largest scale factor an `xs:decimal` can be shifted by in one step: the type carries
/// eighteen fractional digits.
const DECIMAL_SCALE_LIMIT: i32 = 18;

fn overflow(what: &str) -> FnError {
    FnError::new(FnErrorCode::Foar0002, format!("numeric overflow in {what}"))
}

fn not_numeric(value: &AtomicValue, what: &str) -> FnError {
    FnError::new(
        FnErrorCode::Xpty0004,
        format!(
            "{what} expects a numeric argument, not {}",
            value.atomic_type()
        ),
    )
}

/// The type label the numeric functions give their result, given the argument's payload.
fn result_type(data: &AtomicData) -> Option<AtomicType> {
    Some(match data {
        AtomicData::Integer(_) => AtomicType::Integer,
        AtomicData::Decimal(_) => AtomicType::Decimal,
        AtomicData::Float(_) => AtomicType::Float,
        AtomicData::Double(_) => AtomicType::Double,
        _ if data.as_big_integer().is_some() => AtomicType::Integer,
        _ if data.as_big_decimal().is_some() => AtomicType::Decimal,
        _ => return None,
    })
}

/// [`fn:abs`](https://www.w3.org/TR/xpath-functions-31/#func-abs).
///
/// # Errors
///
/// [`FnErrorCode::Xpty0004`] for a non-numeric argument, [`FnErrorCode::Foar0002`] when the
/// magnitude of the most negative `xs:integer` or `xs:decimal` cannot be represented.
///
/// # Examples
///
/// ```
/// use oxixml_fn::numeric::abs;
/// use oxixml_infoset::AtomicValue;
///
/// assert_eq!(abs(&AtomicValue::integer(10_i64))?.string_value(), "10");
/// assert_eq!(abs(&AtomicValue::integer(-10_i64))?.string_value(), "10");
/// assert_eq!(abs(&AtomicValue::double(-1.0e-2))?.string_value(), "0.01");
/// assert_eq!(abs(&AtomicValue::double(f64::NEG_INFINITY))?.string_value(), "INF");
/// # Ok::<(), oxixml_fn::FnError>(())
/// ```
pub fn abs(value: &AtomicValue) -> FnResult<AtomicValue> {
    let data = match value.data() {
        AtomicData::Integer(number) => {
            AtomicData::Integer(number.checked_abs().ok_or_else(|| overflow("fn:abs"))?)
        }
        AtomicData::Decimal(number) => {
            AtomicData::Decimal(number.checked_abs().ok_or_else(|| overflow("fn:abs"))?)
        }
        AtomicData::Float(number) => AtomicData::Float(number.abs()),
        AtomicData::Double(number) => AtomicData::Double(number.abs()),
        other => {
            if let Some(number) = other.as_big_integer() {
                AtomicData::from_big_integer(number.abs())
            } else if let Some(number) = other.as_big_decimal() {
                AtomicData::from_big_decimal(number.abs())
            } else {
                return Err(not_numeric(value, "fn:abs"));
            }
        }
    };
    relabel(&data)
}

/// [`fn:ceiling`](https://www.w3.org/TR/xpath-functions-31/#func-ceiling).
///
/// # Errors
///
/// [`FnErrorCode::Xpty0004`] for a non-numeric argument, [`FnErrorCode::Foar0002`] on overflow.
///
/// # Examples
///
/// ```
/// use oxixml_fn::numeric::ceiling;
/// use oxixml_infoset::{AtomicType, AtomicValue};
///
/// assert_eq!(ceiling(&AtomicValue::parse(AtomicType::Decimal, "10.5")?)?.string_value(), "11");
/// assert_eq!(ceiling(&AtomicValue::parse(AtomicType::Decimal, "-10.5")?)?.string_value(), "-10");
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
pub fn ceiling(value: &AtomicValue) -> FnResult<AtomicValue> {
    let data = match value.data() {
        AtomicData::Integer(number) => AtomicData::Integer(*number),
        AtomicData::Decimal(number) => AtomicData::Decimal(
            number
                .checked_ceil()
                .ok_or_else(|| overflow("fn:ceiling"))?,
        ),
        AtomicData::Float(number) => AtomicData::Float(number.ceil()),
        AtomicData::Double(number) => AtomicData::Double(number.ceil()),
        other => {
            if let Some(number) = other.as_big_integer() {
                AtomicData::from_big_integer(number.clone())
            } else if let Some(number) = other.as_big_decimal() {
                AtomicData::from_big_decimal(
                    number.ceiling().ok_or_else(|| overflow("fn:ceiling"))?,
                )
            } else {
                return Err(not_numeric(value, "fn:ceiling"));
            }
        }
    };
    relabel(&data)
}

/// [`fn:floor`](https://www.w3.org/TR/xpath-functions-31/#func-floor).
///
/// # Errors
///
/// [`FnErrorCode::Xpty0004`] for a non-numeric argument, [`FnErrorCode::Foar0002`] on overflow.
///
/// # Examples
///
/// ```
/// use oxixml_fn::numeric::floor;
/// use oxixml_infoset::{AtomicType, AtomicValue};
///
/// assert_eq!(floor(&AtomicValue::parse(AtomicType::Decimal, "10.5")?)?.string_value(), "10");
/// assert_eq!(floor(&AtomicValue::parse(AtomicType::Decimal, "-10.5")?)?.string_value(), "-11");
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
pub fn floor(value: &AtomicValue) -> FnResult<AtomicValue> {
    let data = match value.data() {
        AtomicData::Integer(number) => AtomicData::Integer(*number),
        AtomicData::Decimal(number) => {
            AtomicData::Decimal(number.checked_floor().ok_or_else(|| overflow("fn:floor"))?)
        }
        AtomicData::Float(number) => AtomicData::Float(number.floor()),
        AtomicData::Double(number) => AtomicData::Double(number.floor()),
        other => {
            if let Some(number) = other.as_big_integer() {
                AtomicData::from_big_integer(number.clone())
            } else if let Some(number) = other.as_big_decimal() {
                AtomicData::from_big_decimal(number.floor().ok_or_else(|| overflow("fn:floor"))?)
            } else {
                return Err(not_numeric(value, "fn:floor"));
            }
        }
    };
    relabel(&data)
}

/// [`fn:round`](https://www.w3.org/TR/xpath-functions-31/#func-round): rounds to a multiple of
/// `10^-precision`, resolving a tie by choosing the value closer to positive infinity.
///
/// # Errors
///
/// [`FnErrorCode::Xpty0004`] for a non-numeric argument, [`FnErrorCode::Foar0002`] on overflow.
///
/// # Examples
///
/// ```
/// use oxixml_fn::numeric::round;
/// use oxixml_infoset::{AtomicType, AtomicValue};
///
/// assert_eq!(round(&AtomicValue::parse(AtomicType::Decimal, "2.5")?, 0)?.string_value(), "3");
/// assert_eq!(round(&AtomicValue::parse(AtomicType::Decimal, "2.4999")?, 0)?.string_value(), "2");
/// assert_eq!(round(&AtomicValue::parse(AtomicType::Decimal, "-2.5")?, 0)?.string_value(), "-2");
/// assert_eq!(round(&AtomicValue::double(1.125), 2)?.string_value(), "1.13");
/// assert_eq!(round(&AtomicValue::double(8452.0), -2)?.string_value(), "8500");
/// assert_eq!(round(&AtomicValue::double(3.1415e0), 2)?.string_value(), "3.14");
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
pub fn round(value: &AtomicValue, precision: i32) -> FnResult<AtomicValue> {
    rounded(value, precision, Tie::TowardsPositiveInfinity, "fn:round")
}

/// [`fn:round-half-to-even`](https://www.w3.org/TR/xpath-functions-31/#func-round-half-to-even):
/// rounds to a multiple of `10^-precision`, resolving a tie by choosing the even multiple.
///
/// # Errors
///
/// [`FnErrorCode::Xpty0004`] for a non-numeric argument, [`FnErrorCode::Foar0002`] on overflow.
///
/// # Examples
///
/// ```
/// use oxixml_fn::numeric::round_half_to_even;
/// use oxixml_infoset::{AtomicType, AtomicValue};
///
/// let decimal = |lexical| AtomicValue::parse(AtomicType::Decimal, lexical);
/// assert_eq!(round_half_to_even(&decimal("0.5")?, 0)?.string_value(), "0");
/// assert_eq!(round_half_to_even(&decimal("1.5")?, 0)?.string_value(), "2");
/// assert_eq!(round_half_to_even(&decimal("2.5")?, 0)?.string_value(), "2");
/// assert_eq!(round_half_to_even(&AtomicValue::double(35_612.25), -2)?.string_value(), "35600");
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
pub fn round_half_to_even(value: &AtomicValue, precision: i32) -> FnResult<AtomicValue> {
    rounded(value, precision, Tie::TowardsEven, "fn:round-half-to-even")
}

/// How a value exactly half way between two multiples is resolved.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Tie {
    /// `fn:round`: pick the multiple closer to positive infinity.
    TowardsPositiveInfinity,
    /// `fn:round-half-to-even`: pick the even multiple.
    TowardsEven,
}

fn rounded(value: &AtomicValue, precision: i32, tie: Tie, what: &str) -> FnResult<AtomicValue> {
    let data = match value.data() {
        AtomicData::Integer(number) => {
            if precision >= 0 {
                AtomicData::Integer(*number)
            } else {
                let scaled = round_decimal(Decimal::from(*number), precision, tie)
                    .ok_or_else(|| overflow(what))?;
                AtomicData::Integer(Integer::try_from(scaled).map_err(|_| overflow(what))?)
            }
        }
        AtomicData::Decimal(number) => AtomicData::Decimal(
            round_decimal(*number, precision, tie).ok_or_else(|| overflow(what))?,
        ),
        AtomicData::Float(number) => AtomicData::Float(Float::from(Double::from(round_f64(
            f64::from(*number),
            precision,
            tie,
        )))),
        AtomicData::Double(number) => {
            AtomicData::Double(Double::from(round_f64(f64::from(*number), precision, tie)))
        }
        other => {
            let half_to_even = matches!(tie, Tie::TowardsEven);
            if let Some(number) = other.as_big_integer() {
                let rounded = BigDecimal::from_big_integer(number.clone())
                    .round(precision, half_to_even)
                    .and_then(|value| value.truncate())
                    .ok_or_else(|| overflow(what))?;
                AtomicData::from_big_integer(rounded)
            } else if let Some(number) = other.as_big_decimal() {
                AtomicData::from_big_decimal(
                    number
                        .round(precision, half_to_even)
                        .ok_or_else(|| overflow(what))?,
                )
            } else {
                return Err(not_numeric(value, what));
            }
        }
    };
    relabel(&data)
}

fn relabel(data: &AtomicData) -> FnResult<AtomicValue> {
    let atomic_type = result_type(data)
        .ok_or_else(|| FnError::new(FnErrorCode::Xpty0004, "the argument is not numeric"))?;
    Ok(AtomicValue::new(atomic_type, data.clone())?)
}

/// Rounds `value` to a multiple of `10^-precision`.
///
/// Precisions whose magnitude exceeds eighteen — the number of fractional digits an
/// `xs:decimal` carries — are applied by scaling in steps of `10^18`, because a single scale
/// factor would not be representable.
fn round_decimal(value: Decimal, precision: i32, tie: Tie) -> Option<Decimal> {
    if precision >= DECIMAL_SCALE_LIMIT {
        // An xs:decimal has at most eighteen fractional digits, so it is already rounded.
        return Some(value);
    }
    let mut scaled = value;
    let mut applied = 0_i32;
    while applied != precision {
        let step = (precision - applied).clamp(-DECIMAL_SCALE_LIMIT, DECIMAL_SCALE_LIMIT);
        let factor = decimal_power_of_ten(step.unsigned_abs())?;
        scaled = if step > 0 {
            scaled.checked_mul(factor)?
        } else {
            scaled.checked_div(factor)?
        };
        applied += step;
    }

    let rounded = round_decimal_to_integer(scaled, tie)?;

    let mut result = rounded;
    let mut undone = 0_i32;
    while undone != precision {
        let step = (precision - undone).clamp(-DECIMAL_SCALE_LIMIT, DECIMAL_SCALE_LIMIT);
        let factor = decimal_power_of_ten(step.unsigned_abs())?;
        result = if step > 0 {
            result.checked_div(factor)?
        } else {
            result.checked_mul(factor)?
        };
        undone += step;
    }
    Some(result)
}

fn decimal_power_of_ten(exponent: u32) -> Option<Decimal> {
    let power = 10_i64.checked_pow(exponent)?;
    Some(Decimal::from(power))
}

fn round_decimal_to_integer(value: Decimal, tie: Tie) -> Option<Decimal> {
    let floor = value.checked_floor()?;
    let fraction = value.checked_sub(floor)?;
    let half = Decimal::new(5, 1).ok()?;
    let one = Decimal::from(1_i64);
    match fraction.cmp(&half) {
        std::cmp::Ordering::Less => Some(floor),
        std::cmp::Ordering::Greater => floor.checked_add(one),
        std::cmp::Ordering::Equal => match tie {
            Tie::TowardsPositiveInfinity => floor.checked_add(one),
            Tie::TowardsEven => {
                let two = Decimal::from(2_i64);
                let remainder = floor.checked_rem_euclid(two)?;
                if remainder == Decimal::from(0_i64) {
                    Some(floor)
                } else {
                    floor.checked_add(one)
                }
            }
        },
    }
}

/// Rounds an `xs:double` or `xs:float` to a multiple of `10^-precision`.
///
/// [F&O 3.1 §4.4.5](https://www.w3.org/TR/xpath-functions-31/#func-round-half-to-even) says how,
/// in as many words: "the argument is cast to `xs:decimal` using an implementation of
/// `xs:decimal` that imposes no limits on the number of digits that can be represented. The
/// function is applied to this `xs:decimal` value, and the resulting `xs:decimal` is cast back to
/// `xs:double`". So the digits that decide the rounding are the ones the *binary* value really
/// has, and scaling by `10^precision` in floating point cannot be used to find them: `250.025e0`
/// is the double `250.02500000000000568…`, which is strictly above the half-way point and rounds
/// to `250.03`, but multiplying it by `100` gives exactly `25002.5` — a tie the value never had
/// (the XSLT 3.0 suite's `math-3303`, and the same shape with the opposite sign for `fn:round`).
///
/// The expansion is exact and finite: a double is `m × 2^-k`, and `2^-k` is `5^k × 10^-k`, so it
/// has exactly `k` fractional digits once the trailing zeros of `m` are taken out.
fn round_f64(value: f64, precision: i32, tie: Tie) -> f64 {
    if !value.is_finite() || value == 0.0 {
        return value;
    }
    let negative = value.is_sign_negative();
    let magnitude = value.abs();
    let exact = format!("{:.*}", exact_fraction_digits(magnitude), magnitude);
    let Some(rounded) = round_decimal_string(&exact, precision, tie, negative) else {
        return value;
    };
    let result: f64 = rounded.parse().unwrap_or(value);
    let result = if negative { -result } else { result };
    if result == 0.0 && negative {
        // fn:round(-0.2) is -0.0E0, not 0.0E0.
        -0.0
    } else {
        result
    }
}

/// How many fractional digits the exact decimal expansion of the finite, positive `value` has.
fn exact_fraction_digits(value: f64) -> usize {
    let bits = value.to_bits();
    let exponent = ((bits >> 52) & 0x7ff) as i32;
    let mantissa = bits & ((1_u64 << 52) - 1);
    // A subnormal is `mantissa × 2^-1074`; every other value is `(2^52 + mantissa) × 2^(e-1075)`.
    let (significand, mut power) = if exponent == 0 {
        (mantissa, -1074_i32)
    } else {
        ((1_u64 << 52) | mantissa, exponent - 1075)
    };
    if significand == 0 {
        return 0;
    }
    power += significand.trailing_zeros() as i32;
    usize::try_from(-power).unwrap_or(0)
}

/// Rounds the exact non-negative decimal expansion `lexical` to a multiple of `10^-precision`.
///
/// `negative` is the sign of the value the digits came from, which only the tie rule of
/// `fn:round` reads: rounding a tie "towards positive infinity" means away from zero for a
/// positive value and towards zero for a negative one. [`None`] when the precision is so extreme
/// that no scaling is called for.
fn round_decimal_string(lexical: &str, precision: i32, tie: Tie, negative: bool) -> Option<String> {
    let (integer, fraction) = lexical.split_once('.').unwrap_or((lexical, ""));
    // A precision at or beyond the last digit leaves the value alone.
    if precision >= i32::try_from(fraction.len()).unwrap_or(i32::MAX) {
        return None;
    }
    let digits: Vec<u8> = integer
        .bytes()
        .chain(fraction.bytes())
        .map(|digit| digit - b'0')
        .collect();
    let keep = i32::try_from(integer.len()).ok()? + precision;
    if keep < 0 {
        // Every digit is dropped and the value is below half of the unit being rounded to.
        return Some("0".to_owned());
    }
    let keep = usize::try_from(keep).ok()?.min(digits.len());
    let (kept, dropped) = digits.split_at(keep);
    let mut kept = kept.to_vec();
    let round_up = match dropped.first() {
        None => false,
        Some(&first) if first > 5 => true,
        Some(&first) if first < 5 => false,
        // Exactly five: a tie only when nothing follows it.
        _ if dropped[1..].iter().any(|digit| *digit != 0) => true,
        _ => match tie {
            Tie::TowardsPositiveInfinity => !negative,
            Tie::TowardsEven => kept.last().is_some_and(|digit| digit % 2 == 1),
        },
    };
    if round_up {
        let mut index = kept.len();
        loop {
            if index == 0 {
                kept.insert(0, 1);
                break;
            }
            index -= 1;
            if kept[index] == 9 {
                kept[index] = 0;
            } else {
                kept[index] += 1;
                break;
            }
        }
    }
    Some(scaled_lexical(&kept, precision))
}

/// Writes the digit string `digits`, whose last digit has the weight `10^-precision`, as a
/// decimal literal.
fn scaled_lexical(digits: &[u8], precision: i32) -> String {
    let mut lexical: String = digits
        .iter()
        .map(|digit| char::from(b'0' + digit))
        .collect();
    if lexical.is_empty() {
        lexical.push('0');
    }
    match precision.cmp(&0) {
        std::cmp::Ordering::Equal => lexical,
        // The digits are a multiple of `10^-precision`, so the point goes back in.
        std::cmp::Ordering::Greater => {
            let fraction = usize::try_from(precision).unwrap_or(0);
            while lexical.len() <= fraction {
                lexical.insert(0, '0');
            }
            let point = lexical.len() - fraction;
            lexical.insert(point, '.');
            lexical
        }
        // A negative precision rounds to a power of ten, whose zeros are written back.
        std::cmp::Ordering::Less => {
            for _ in 0..precision.unsigned_abs() {
                lexical.push('0');
            }
            lexical
        }
    }
}

fn round_f64_to_integer(value: f64, tie: Tie) -> f64 {
    let floor = value.floor();
    let fraction = value - floor;
    if fraction < 0.5 {
        floor
    } else if fraction > 0.5 {
        floor + 1.0
    } else {
        match tie {
            Tie::TowardsPositiveInfinity => floor + 1.0,
            Tie::TowardsEven => {
                if floor.rem_euclid(2.0) == 0.0 {
                    floor
                } else {
                    floor + 1.0
                }
            }
        }
    }
}

/// Rounds `value` to the nearest integer exactly as `fn:round` does, resolving a tie towards
/// positive infinity. This is the rule the duration scaling operators of F&O §8.4 apply.
#[must_use]
pub(crate) fn round_f64_half_up(value: f64) -> f64 {
    if !value.is_finite() {
        return value;
    }
    round_f64_to_integer(value, Tie::TowardsPositiveInfinity)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn decimal(lexical: &str) -> AtomicValue {
        AtomicValue::parse(AtomicType::Decimal, lexical).expect("a valid xs:decimal")
    }

    #[test]
    fn abs_examples_from_the_specification() -> FnResult<()> {
        assert_eq!(abs(&AtomicValue::integer(10_i64))?.string_value(), "10");
        assert_eq!(abs(&AtomicValue::integer(-10_i64))?.string_value(), "10");
        assert_eq!(abs(&AtomicValue::double(-1.0e-2))?.string_value(), "0.01");
        assert_eq!(
            abs(&AtomicValue::double(f64::NEG_INFINITY))?.string_value(),
            "INF"
        );
        assert!(abs(&AtomicValue::double(f64::NAN))?.is_nan());
        assert!(abs(&AtomicValue::string("x")).is_err());
        Ok(())
    }

    #[test]
    fn ceiling_and_floor_examples_from_the_specification() -> FnResult<()> {
        assert_eq!(ceiling(&decimal("10.5"))?.string_value(), "11");
        assert_eq!(ceiling(&decimal("-10.5"))?.string_value(), "-10");
        assert_eq!(floor(&decimal("10.5"))?.string_value(), "10");
        assert_eq!(floor(&decimal("-10.5"))?.string_value(), "-11");
        assert_eq!(ceiling(&AtomicValue::integer(7_i64))?.string_value(), "7");
        assert_eq!(floor(&AtomicValue::integer(7_i64))?.string_value(), "7");
        Ok(())
    }

    #[test]
    #[expect(
        clippy::approx_constant,
        reason = "the specification's fn:round example is literally 3.1415e0"
    )]
    fn round_examples_from_the_specification() -> FnResult<()> {
        assert_eq!(round(&decimal("2.5"), 0)?.string_value(), "3");
        assert_eq!(round(&decimal("2.4999"), 0)?.string_value(), "2");
        assert_eq!(round(&decimal("-2.5"), 0)?.string_value(), "-2");
        assert_eq!(round(&AtomicValue::integer(1_i64), 0)?.string_value(), "1");
        assert_eq!(
            round(&AtomicValue::double(1.125), 2)?.string_value(),
            "1.13"
        );
        assert_eq!(
            round(&AtomicValue::double(8452.0), -2)?.string_value(),
            "8500"
        );
        assert_eq!(
            round(&AtomicValue::double(3.1415e0), 2)?.string_value(),
            "3.14"
        );
        Ok(())
    }

    #[test]
    fn round_preserves_negative_zero() -> FnResult<()> {
        let rounded = round(&AtomicValue::double(-0.2), 0)?;
        assert_eq!(rounded.string_value(), "-0");
        Ok(())
    }

    #[test]
    fn round_half_to_even_examples_from_the_specification() -> FnResult<()> {
        assert_eq!(round_half_to_even(&decimal("0.5"), 0)?.string_value(), "0");
        assert_eq!(round_half_to_even(&decimal("1.5"), 0)?.string_value(), "2");
        assert_eq!(round_half_to_even(&decimal("2.5"), 0)?.string_value(), "2");
        assert_eq!(
            round_half_to_even(&AtomicValue::double(3.567_812_e3), 2)?.string_value(),
            "3567.81"
        );
        assert_eq!(
            round_half_to_even(&AtomicValue::double(4.7564e-3), 2)?.string_value(),
            "0"
        );
        assert_eq!(
            round_half_to_even(&AtomicValue::double(35_612.25), -2)?.string_value(),
            "35600"
        );
        Ok(())
    }

    /// A double is rounded on the digits it really has, not on the ones a scaling multiplication
    /// would give it — F&O 3.1 §4.4.5 casts the argument to an unlimited `xs:decimal` first.
    ///
    /// The XSLT 3.0 suite's `math-3303` is the case: `250.0250e0` is the double
    /// `250.025000000000005684…`, strictly *above* the half-way point, so it rounds to `250.03`
    /// however the tie would have been resolved. Multiplying by `100` gives exactly `25002.5`,
    /// a tie the value never had, and half-to-even would then answer `250.02`.
    #[test]
    fn a_double_is_rounded_on_its_own_digits() -> FnResult<()> {
        let double = AtomicValue::double;
        for (value, expected) in [
            (180.018_0, "180.02"),
            (150.015_0, "150.01"),
            (250.025_0, "250.03"),
        ] {
            assert_eq!(
                round_half_to_even(&double(value), 2)?.string_value(),
                expected,
                "round-half-to-even({value:?}, 2)"
            );
            assert_eq!(
                round_half_to_even(&double(-value), 2)?.string_value(),
                format!("-{expected}"),
                "round-half-to-even({:?}, 2)",
                -value
            );
        }
        // `fn:round` resolves a *genuine* tie towards positive infinity, and the false tie the
        // multiplication invented used to send `-250.025` the wrong way.
        assert_eq!(round(&double(250.025_0), 2)?.string_value(), "250.03");
        assert_eq!(round(&double(-250.025_0), 2)?.string_value(), "-250.03");
        assert_eq!(round(&double(2.5), 0)?.string_value(), "3");
        assert_eq!(round(&double(-2.5), 0)?.string_value(), "-2");
        // The exact expansion also decides the rounding of a value whose scaled form is far from
        // a tie, which is the ordinary case and must not move.
        assert_eq!(round_half_to_even(&double(1.005), 2)?.string_value(), "1");
        assert_eq!(
            round_half_to_even(&double(0.125), 2)?.string_value(),
            "0.12"
        );
        assert_eq!(
            round_half_to_even(&double(0.375), 2)?.string_value(),
            "0.38"
        );
        assert_eq!(round(&double(-0.5), 0)?.string_value(), "-0");
        assert_eq!(
            round_half_to_even(&double(1e300), 2)?.string_value(),
            "1.0E300"
        );
        assert_eq!(round(&double(1.5e-300), 3)?.string_value(), "0");
        assert_eq!(round(&double(f64::MIN_POSITIVE), 0)?.string_value(), "0");
        Ok(())
    }

    /// The whole of the XSLT 3.0 suite's `math-3701`, which walks `fn:round` over doubles: the
    /// negative counterpart of the false tie is `round(-1.365e1, 1)`, where the double is
    /// `-13.650000000000000355…` and therefore nearer `-13.7`, while scaling by ten produces
    /// exactly `-136.5` and sends the tie rule the other way.
    #[test]
    fn round_over_doubles_follows_the_suites_own_table() -> FnResult<()> {
        let double = AtomicValue::double;
        for (value, precision, expected) in [
            (12.0e0, 1, "12"),
            (1.154e1, 1, "11.5"),
            (1.365e1, 1, "13.7"),
            (-1.365e1, 1, "-13.7"),
            (-2.5e0, 0, "-2"),
            (2.5e0, 0, "3"),
            (2.4999e2, 0, "250"),
            (8492.15e1, -1, "84920"),
            (8495.551e2, -1, "849560"),
            (8496.787e2, -2, "849700"),
            (45e100, -101, "5.0E101"),
            (9.9e-99, 99, "1.0E-98"),
            (0.0e0, 0, "0"),
        ] {
            assert_eq!(
                round(&double(value), precision)?.string_value(),
                expected,
                "round({value:?}, {precision})"
            );
        }
        assert!(round(&double(f64::NAN), 0)?.is_nan());
        assert_eq!(round(&double(f64::INFINITY), 0)?.string_value(), "INF");
        Ok(())
    }

    #[test]
    fn integers_round_at_negative_precision() -> FnResult<()> {
        assert_eq!(
            round(&AtomicValue::integer(8452_i64), -2)?.string_value(),
            "8500"
        );
        assert_eq!(
            round_half_to_even(&AtomicValue::integer(8450_i64), -2)?.string_value(),
            "8400"
        );
        Ok(())
    }

    #[test]
    fn results_keep_the_primitive_type_of_the_argument() -> FnResult<()> {
        let short = AtomicValue::parse(AtomicType::Short, "-5")?;
        assert_eq!(abs(&short)?.atomic_type(), AtomicType::Integer);
        assert_eq!(ceiling(&decimal("1.5"))?.atomic_type(), AtomicType::Decimal);
        assert_eq!(
            floor(&AtomicValue::float(1.5_f32))?.atomic_type(),
            AtomicType::Float
        );
        Ok(())
    }
}
