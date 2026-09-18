//! The ECMAScript `Number::toString(10)` algorithm ([ECMA-262 §6.1.6.1.20], "Number::toString").
//!
//! [ECMA-262 §6.1.6.1.20]: https://tc39.es/ecma262/#sec-numeric-types-number-tostring

use super::digits::decompose;

/// Formats `value` exactly as ECMAScript's `Number::prototype.toString(10)` would (equivalent
/// to `` `${value}` `` in JavaScript for a `Number`).
///
/// * `NaN` formats as `"NaN"`.
/// * `+Infinity`/`-Infinity` format as `"Infinity"`/`"-Infinity"`.
/// * `+0`/`-0` both format as `"0"` (the sign of zero is not shown).
/// * Otherwise the shortest decimal digit string that round-trips back to `value` is chosen,
///   and rendered in plain decimal form for `1e-6 <= |value| < 1e21`, or in exponential form
///   (`d`, or `d.ddd`, followed by `e`, `+`/`-`, and the exponent with no leading zeros)
///   outside that range.
///
/// ```
/// use oxixml_text::num::ecmascript_double;
///
/// assert_eq!(ecmascript_double(0.1), "0.1");
/// assert_eq!(ecmascript_double(100.0), "100");
/// assert_eq!(ecmascript_double(1e21), "1e+21");
/// assert_eq!(ecmascript_double(1e-7), "1e-7");
/// assert_eq!(ecmascript_double(-0.0), "0");
/// assert_eq!(ecmascript_double(f64::NAN), "NaN");
/// ```
#[must_use]
pub fn ecmascript_double(value: f64) -> String {
    if value.is_nan() {
        return "NaN".to_string();
    }
    if value == f64::INFINITY {
        return "Infinity".to_string();
    }
    if value == f64::NEG_INFINITY {
        return "-Infinity".to_string();
    }
    if value == 0.0 {
        return "0".to_string();
    }
    let negative = value.is_sign_negative();
    let parts = decompose(value.abs());
    let body = assemble(&parts.digits, parts.exponent);
    if negative { format!("-{body}") } else { body }
}

/// The `f32` counterpart of [`ecmascript_double`]: emits the **shortest decimal string that
/// round-trips back to the same `f32`**, laid out with ECMAScript's `Number::toString(10)`
/// rules (same `NaN`/`Infinity`/`-0` spellings, same `1e-6 <= |value| < 1e21` plain-decimal
/// window, same exponential form outside it). Useful for round-tripping `xsd:float` values
/// through JSON-like text without widening them first.
///
/// This is deliberately **not** the same as `ecmascript_double(f64::from(value))`. ECMAScript
/// has no 32-bit float type, so a JavaScript `Number` holding the widened value prints the
/// shortest string that round-trips as an `f64`, which for most `f32`s exposes the binary32
/// representation error in full:
///
/// ```
/// use oxixml_text::num::{ecmascript_double, ecmascript_float};
///
/// assert_eq!(ecmascript_float(1.5_f32), "1.5");
/// assert_eq!(ecmascript_float(-0.0_f32), "0");
///
/// // The divergence: shortest-for-`f32` versus shortest-for-`f64`.
/// assert_eq!(ecmascript_float(0.1_f32), "0.1");
/// assert_eq!(ecmascript_double(f64::from(0.1_f32)), "0.10000000149011612");
/// ```
///
/// Use this function when the value's `f32` identity is what must survive the round trip (the
/// `xsd:float` canonical-form case); use `ecmascript_double(f64::from(value))` when you need
/// byte-for-byte agreement with what JavaScript would print for the widened value.
#[must_use]
pub fn ecmascript_float(value: f32) -> String {
    if value.is_nan() {
        return "NaN".to_string();
    }
    if value == f32::INFINITY {
        return "Infinity".to_string();
    }
    if value == f32::NEG_INFINITY {
        return "-Infinity".to_string();
    }
    if value == 0.0 {
        return "0".to_string();
    }
    let negative = value.is_sign_negative();
    let parts = decompose(value.abs());
    let body = assemble(&parts.digits, parts.exponent);
    if negative { format!("-{body}") } else { body }
}

/// Assembles the unsigned ECMAScript numeral from significant `digits` (see
/// [`super::digits::ExponentialDigits`]) and their `exponent`, following steps 5-7 of
/// `Number::toString` (the sign, `NaN`, infinity, and zero cases are handled by the callers
/// above; this only ever sees a finite non-zero magnitude).
fn assemble(digits: &str, exponent: i32) -> String {
    // `k` is the digit count ("s" in the spec has k digits); `n` is the spec's `n`, defined so
    // the value equals `s * 10^(n - k)`. Since `value = 0.digits * 10^(exponent + 1)` here,
    // `n = exponent + 1`.
    let k = i32::try_from(digits.len()).unwrap_or(i32::MAX);
    let n = exponent.saturating_add(1);

    if n >= k && n <= 21 {
        // Step 5: integer with trailing zeros, no fractional part.
        let mut out = String::with_capacity(n.max(0) as usize);
        out.push_str(digits);
        for _ in 0..(n - k) {
            out.push('0');
        }
        out
    } else if n > 0 && n <= 21 {
        // Step 6: split the digits around the decimal point.
        let split_at = n as usize;
        format!("{}.{}", &digits[..split_at], &digits[split_at..])
    } else if n > -6 && n <= 0 {
        // Step 7: "0." followed by leading zeros then all the digits.
        let leading_zeros = usize::try_from(-n).unwrap_or(0);
        let mut out = String::with_capacity(2 + leading_zeros + digits.len());
        out.push_str("0.");
        for _ in 0..leading_zeros {
            out.push('0');
        }
        out.push_str(digits);
        out
    } else {
        // Steps 8-9: exponential notation, explicit '+'/'-' sign on the exponent.
        let display_exponent = n - 1;
        let sign = if display_exponent >= 0 { '+' } else { '-' };
        let magnitude = display_exponent.unsigned_abs();
        if k == 1 {
            format!("{digits}e{sign}{magnitude}")
        } else {
            format!("{}.{}e{sign}{magnitude}", &digits[..1], &digits[1..])
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn zero_and_signs() {
        assert_eq!(ecmascript_double(0.0), "0");
        assert_eq!(ecmascript_double(-0.0), "0");
        assert_eq!(ecmascript_double(1.0), "1");
        assert_eq!(ecmascript_double(-1.0), "-1");
    }

    #[test]
    fn specials() {
        assert_eq!(ecmascript_double(f64::NAN), "NaN");
        assert_eq!(ecmascript_double(f64::INFINITY), "Infinity");
        assert_eq!(ecmascript_double(f64::NEG_INFINITY), "-Infinity");
    }

    #[test]
    fn classic_hard_cases() {
        assert_eq!(ecmascript_double(0.1), "0.1");
        assert_eq!(ecmascript_double(5e-324), "5e-324");
        assert_eq!(ecmascript_double(1e21), "1e+21");
        assert_eq!(
            ecmascript_double(999_999_999_999_999_900_000.0),
            "999999999999999900000"
        );
        assert_eq!(
            ecmascript_double(123_456_789_012_345_680_000.0),
            "123456789012345680000"
        );
        assert_eq!(ecmascript_double(0.000_001), "0.000001");
        assert_eq!(ecmascript_double(0.000_000_1), "1e-7");
    }

    #[test]
    fn decimal_vs_exponential_boundaries() {
        // 1e-6 is the smallest magnitude formatted in decimal form.
        assert_eq!(ecmascript_double(1e-6), "0.000001");
        assert_eq!(ecmascript_double(9.999e-7), "9.999e-7");
        // 1e21 is the smallest magnitude formatted in exponential form.
        assert_eq!(ecmascript_double(9.99e20), "999000000000000000000");
        assert_eq!(ecmascript_double(1e21), "1e+21");
    }

    #[test]
    fn negative_values_keep_leading_minus() {
        assert_eq!(ecmascript_double(-0.1), "-0.1");
        assert_eq!(ecmascript_double(-5e-324), "-5e-324");
        assert_eq!(ecmascript_double(-1e21), "-1e+21");
    }

    #[test]
    fn integers_render_without_decimal_point() {
        assert_eq!(ecmascript_double(100.0), "100");
        assert_eq!(ecmascript_double(1234.0), "1234");
    }

    #[test]
    fn f32_variant_matches_shortest_round_trip() {
        assert_eq!(ecmascript_float(1.5_f32), "1.5");
        assert_eq!(ecmascript_float(0.0_f32), "0");
        assert_eq!(ecmascript_float(-0.0_f32), "0");
        assert_eq!(ecmascript_float(f32::NAN), "NaN");
        assert_eq!(ecmascript_float(f32::INFINITY), "Infinity");
        assert_eq!(ecmascript_float(f32::NEG_INFINITY), "-Infinity");
        assert_eq!(ecmascript_float(100.0_f32), "100");
    }

    /// Pins the contract documented on [`ecmascript_float`]: its output is the shortest string
    /// that round-trips back to the *same `f32`*, which is deliberately different from what
    /// ECMAScript prints for the widened `f64`. The doc comment once claimed the latter; this
    /// test is what keeps the two from drifting apart again.
    #[test]
    fn f32_variant_is_shortest_for_f32_not_for_the_widened_f64() {
        for (value, shortest_f32) in [
            (0.1_f32, "0.1"),
            (1.1_f32, "1.1"),
            (1e-40_f32, "1e-40"),
            (f32::MAX, "3.4028235e+38"),
        ] {
            assert_eq!(ecmascript_float(value), shortest_f32, "{value:e}");
            // The output round-trips as an `f32` ...
            let parsed: f32 = ecmascript_float(value)
                .parse()
                .expect("ecmascript_float output must parse as f32");
            assert_eq!(parsed, value, "{value:e}");
            // ... and is genuinely shorter/different from the widened-`f64` rendering.
            assert_ne!(
                ecmascript_float(value),
                ecmascript_double(f64::from(value)),
                "{value:e}"
            );
        }
        assert_eq!(ecmascript_double(f64::from(0.1_f32)), "0.10000000149011612");
    }

    #[test]
    fn round_trips_via_str_parse() {
        let samples: &[f64] = &[
            0.0,
            1.0,
            -1.0,
            0.1,
            std::f64::consts::PI,
            1e-300,
            1e300,
            5e-324,
            f64::MAX,
            f64::MIN_POSITIVE,
            12345.6789,
            1e21,
            1e20,
            9_007_199_254_740_993.0,
        ];
        for &value in samples {
            let text = ecmascript_double(value);
            let parsed: f64 = text
                .parse()
                .expect("ecmascript_double output must parse as f64");
            assert_eq!(parsed, value, "round trip failed for {value:e} -> {text:?}");
        }
    }
}
