//! The numeric functions and operators of
//! [F&O 3.1 §4](https://www.w3.org/TR/xpath-functions-31/#numeric-functions).
//!
//! * [`arith`] — the `op:` arithmetic operators, including the date, time and duration
//!   operators of §8.4, which share the same promotion machinery.
//! * [`mod@round`] — `fn:abs`, `fn:ceiling`, `fn:floor`, `fn:round` and `fn:round-half-to-even`.
//! * [`mod@format_number`] — `fn:format-number` and the decimal-format picture-string language.
//!
//! [`number`] itself lives here, because it is the bridge between the value space and the
//! numeric tower.

pub mod arith;
pub mod format_number;
pub mod round;

pub use arith::{
    ArithOp, add, arithmetic, date_time_from_parts, divide, integer_divide, modulus, multiply,
    promote_untyped, subtract, unary_minus, unary_plus,
};
pub use format_number::{DecimalFormat, format_number, picture_has_exponent};
pub use round::{abs, ceiling, floor, round, round_half_to_even};

use crate::error::{FnError, FnErrorCode, FnResult};
use oxixml_infoset::{AtomicType, AtomicValue, Item, Sequence};
use oxixml_xsd::Double;

/// [`fn:number`](https://www.w3.org/TR/xpath-functions-31/#func-number): the `xs:double` value
/// of an item, or `NaN` when it has none.
///
/// The empty sequence and any value that cannot be cast to `xs:double` both give `NaN` — this
/// function never raises a cast error, which is what makes it usable in a sort key.
///
/// # Errors
///
/// [`FnErrorCode::Foty0013`] when the argument is a map, which cannot be atomized, and
/// [`FnErrorCode::Xpty0004`] when it atomizes to more than one value.
///
/// # Examples
///
/// ```
/// use oxixml_fn::numeric::number;
/// use oxixml_infoset::{AtomicValue, Item};
///
/// assert_eq!(number(Some(&Item::from(AtomicValue::untyped_atomic("12.5"))))?, 12.5);
/// assert!(number(Some(&Item::from(AtomicValue::string("abc"))))?.is_nan());
/// assert!(number(None)?.is_nan());
/// # Ok::<(), oxixml_fn::FnError>(())
/// ```
pub fn number(item: Option<&Item>) -> FnResult<f64> {
    let Some(item) = item else {
        return Ok(f64::NAN);
    };
    let atomized = item.atomize()?;
    match atomized.as_slice() {
        [] => Ok(f64::NAN),
        [Item::Atomic(value)] => Ok(cast_to_double(value)),
        _ => Err(FnError::new(
            FnErrorCode::Xpty0004,
            "fn:number expects an item that atomizes to at most one value",
        )),
    }
}

/// [`fn:number`](https://www.w3.org/TR/xpath-functions-31/#func-number) applied to a sequence,
/// which must be empty or a singleton.
///
/// # Errors
///
/// See [`number`].
pub fn number_of(sequence: &Sequence) -> FnResult<f64> {
    match sequence.as_slice() {
        [] => Ok(f64::NAN),
        [item] => number(Some(item)),
        _ => Err(FnError::new(
            FnErrorCode::Xpty0004,
            "fn:number expects at most one item",
        )),
    }
}

/// Casts an atomic value to `xs:double`, answering `NaN` rather than raising when the cast is
/// not possible. This is `fn:number`'s "cast, and on failure return NaN" rule.
#[must_use]
pub fn cast_to_double(value: &AtomicValue) -> f64 {
    if let Some(number) = value.as_f64() {
        return number;
    }
    if let Some(boolean) = value.as_bool() {
        return if boolean { 1.0 } else { 0.0 };
    }
    // The casting table of F&O 3.1 §19.1 has no entry from `xs:anyURI` to `xs:double`, so the
    // `xs:double` constructor `fn:number` applies fails and the answer is `NaN` — not the number
    // the URI happens to spell.
    if value.atomic_type().is_derived_from(AtomicType::AnyUri) {
        return f64::NAN;
    }
    match value.as_str() {
        Some(text) => AtomicValue::parse(AtomicType::Double, text)
            .ok()
            .and_then(|value| value.as_f64())
            .unwrap_or(f64::NAN),
        None => f64::NAN,
    }
}

/// Builds an `xs:double` atomic value from an [`f64`].
#[must_use]
pub fn double_value(number: f64) -> AtomicValue {
    AtomicValue::double(Double::from(number))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn number_examples_from_the_specification() -> FnResult<()> {
        assert_eq!(
            number(Some(&Item::from(AtomicValue::untyped_atomic("12.5"))))?,
            12.5
        );
        assert!(number(Some(&Item::from(AtomicValue::string("abc"))))?.is_nan());
        assert!(number(None)?.is_nan());
        assert_eq!(number(Some(&Item::from(AtomicValue::boolean(true))))?, 1.0);
        Ok(())
    }

    #[test]
    fn number_of_a_sequence_rejects_more_than_one_item() {
        let two = Sequence::from_items([
            Item::from(AtomicValue::integer(1_i64)),
            Item::from(AtomicValue::integer(2_i64)),
        ]);
        assert!(number_of(&two).is_err());
    }
}
