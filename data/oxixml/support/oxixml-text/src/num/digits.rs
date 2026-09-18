//! Shared decimal-digit extraction used by both the [`super::ecmascript`] and [`super::xsd`]
//! formatters.
//!
//! Both formats need the same underlying fact about a finite, non-zero floating-point value:
//! its *shortest round-tripping decimal digit string* and the power of ten that positions it.
//! Rather than re-deriving that (a nontrivial algorithm — see Steele & White / Grisu / Ryū),
//! this module leans on the Rust standard library's own `{:e}` ([`std::fmt::LowerExp`])
//! formatting, which already produces the shortest decimal digits that round-trip back to the
//! original `f32`/`f64` value. We parse that back apart into plain digits + exponent, which
//! both spec-shaped formatters then re-assemble their own way.

/// The decimal digits of a formatted value, and the power-of-ten exponent that positions them.
///
/// `digits` holds the significant decimal digits with no sign, no decimal point, and no
/// leading or trailing zeros (except for the value zero itself, whose single digit is `"0"`).
/// `exponent` is defined so that the original value equals
/// `0.<digits[0]>.<digits[1..]> * 10^(exponent + 1)`; equivalently, the decimal point sits
/// immediately after `digits[0]` when the value is written as `d.ddd * 10^exponent`.
pub(super) struct ExponentialDigits {
    /// Significant decimal digits, most significant first.
    pub(super) digits: String,
    /// The power of ten of the leading digit (`digits[0]` represents that many tens).
    pub(super) exponent: i32,
}

/// Formats `abs_value` with [`std::fmt::LowerExp`] and decomposes the result into
/// [`ExponentialDigits`].
///
/// `abs_value` must be finite and non-negative (callers strip the sign and handle
/// `NaN`/infinite/zero specially before reaching here); `T` is `f32` or `f64`.
pub(super) fn decompose<T: std::fmt::LowerExp>(abs_value: T) -> ExponentialDigits {
    let formatted = format!("{abs_value:e}");
    // `{:e}` always yields `<int-digit>['.'<frac-digits>]['e'<exponent>]`, with `<int-digit>`
    // a single ASCII digit (the mantissa is normalized to exactly one digit before the
    // point) and no sign on the mantissa, since `abs_value` is non-negative.
    let (mantissa, exponent_part) = formatted
        .split_once('e')
        .unwrap_or((formatted.as_str(), "0"));
    let exponent = exponent_part.parse::<i32>().unwrap_or(0);
    let (int_part, frac_part) = mantissa.split_once('.').unwrap_or((mantissa, ""));
    let mut digits = String::with_capacity(int_part.len() + frac_part.len());
    digits.push_str(int_part);
    digits.push_str(frac_part);
    ExponentialDigits { digits, exponent }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn decomposes_simple_values() {
        let d = decompose(0.1_f64);
        assert_eq!(d.digits, "1");
        assert_eq!(d.exponent, -1);

        let d = decompose(123.0_f64);
        assert_eq!(d.digits, "123");
        assert_eq!(d.exponent, 2);

        let d = decompose(0.0_f64);
        assert_eq!(d.digits, "0");
        assert_eq!(d.exponent, 0);
    }

    #[test]
    fn decomposes_subnormal() {
        let d = decompose(5e-324_f64);
        assert_eq!(d.digits, "5");
        assert_eq!(d.exponent, -324);
    }
}
