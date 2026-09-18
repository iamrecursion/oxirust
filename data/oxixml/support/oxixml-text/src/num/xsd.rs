//! The XSD 1.1 canonical mapping for `double` and `float`
//! ([XSD 1.1 Part 2 §3.3.5]/[§3.3.4]).
//!
//! Canonical form always uses exponential notation: a mantissa with exactly one digit before
//! the decimal point and at least one digit after it (`"1.0E5"`, never `"1E5"`), an upper-case
//! `E`, and an exponent with no leading zeros and no `+` sign on positive exponents. `INF`,
//! `-INF`, and `NaN` spell out the non-finite values; unlike the ECMAScript formatting in
//! [`super::ecmascript`], the sign of zero is preserved (`"-0.0E0"` for negative zero).
//!
//! [XSD 1.1 Part 2 §3.3.5]: https://www.w3.org/TR/xmlschema11-2/#double
//! [§3.3.4]: https://www.w3.org/TR/xmlschema11-2/#float

use super::digits::decompose;

/// Formats `value` in the XSD 1.1 canonical form for `xsd:double`.
///
/// ```
/// use oxixml_text::num::xsd_canonical_double;
///
/// assert_eq!(xsd_canonical_double(1.0), "1.0E0");
/// assert_eq!(xsd_canonical_double(-0.0017), "-1.7E-3");
/// assert_eq!(xsd_canonical_double(0.0), "0.0E0");
/// assert_eq!(xsd_canonical_double(-0.0), "-0.0E0");
/// assert_eq!(xsd_canonical_double(f64::INFINITY), "INF");
/// assert_eq!(xsd_canonical_double(f64::NAN), "NaN");
/// ```
#[must_use]
pub fn xsd_canonical_double(value: f64) -> String {
    if value.is_nan() {
        return "NaN".to_string();
    }
    if value == f64::INFINITY {
        return "INF".to_string();
    }
    if value == f64::NEG_INFINITY {
        return "-INF".to_string();
    }
    let negative = value.is_sign_negative();
    let parts = decompose(value.abs());
    let body = assemble(&parts.digits, parts.exponent);
    if negative { format!("-{body}") } else { body }
}

/// The `f32` counterpart of [`xsd_canonical_double`], for `xsd:float`.
///
/// ```
/// use oxixml_text::num::xsd_canonical_float;
///
/// assert_eq!(xsd_canonical_float(100_000.0_f32), "1.0E5");
/// assert_eq!(xsd_canonical_float(f32::NEG_INFINITY), "-INF");
/// ```
#[must_use]
pub fn xsd_canonical_float(value: f32) -> String {
    if value.is_nan() {
        return "NaN".to_string();
    }
    if value == f32::INFINITY {
        return "INF".to_string();
    }
    if value == f32::NEG_INFINITY {
        return "-INF".to_string();
    }
    let negative = value.is_sign_negative();
    let parts = decompose(value.abs());
    let body = assemble(&parts.digits, parts.exponent);
    if negative { format!("-{body}") } else { body }
}

/// Assembles the unsigned canonical numeral `M.NNNEexp` from significant `digits` and their
/// `exponent` (see [`super::digits::ExponentialDigits`]).
fn assemble(digits: &str, exponent: i32) -> String {
    let mantissa = if digits.len() == 1 {
        format!("{digits}.0")
    } else {
        format!("{}.{}", &digits[..1], &digits[1..])
    };
    format!("{mantissa}E{exponent}")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn simple_values() {
        assert_eq!(xsd_canonical_double(1.0), "1.0E0");
        assert_eq!(xsd_canonical_double(-1.7e-3), "-1.7E-3");
        assert_eq!(xsd_canonical_double(0.0), "0.0E0");
        assert_eq!(xsd_canonical_double(-0.0), "-0.0E0");
        assert_eq!(xsd_canonical_double(100_000.0), "1.0E5");
        assert_eq!(xsd_canonical_double(-100_000.0), "-1.0E5");
    }

    #[test]
    fn specials() {
        assert_eq!(xsd_canonical_double(f64::NAN), "NaN");
        assert_eq!(xsd_canonical_double(f64::INFINITY), "INF");
        assert_eq!(xsd_canonical_double(f64::NEG_INFINITY), "-INF");
    }

    #[test]
    fn subnormals() {
        // The smallest positive subnormal f64.
        let smallest = f64::from_bits(1);
        assert_eq!(xsd_canonical_double(smallest), "5.0E-324");
        assert_eq!(xsd_canonical_double(-smallest), "-5.0E-324");

        let smallest_f32 = f32::from_bits(1);
        assert_eq!(xsd_canonical_float(smallest_f32), "1.0E-45");
    }

    #[test]
    fn extremes() {
        assert_eq!(xsd_canonical_double(f64::MAX), "1.7976931348623157E308");
        assert_eq!(
            xsd_canonical_double(f64::MIN_POSITIVE),
            "2.2250738585072014E-308"
        );
    }

    #[test]
    fn float_variant() {
        assert_eq!(xsd_canonical_float(1.0_f32), "1.0E0");
        assert_eq!(xsd_canonical_float(0.0_f32), "0.0E0");
        assert_eq!(xsd_canonical_float(-0.0_f32), "-0.0E0");
        assert_eq!(xsd_canonical_float(f32::MAX), "3.4028235E38");
    }

    #[test]
    fn round_trips_via_str_parse() {
        let samples: &[f64] = &[
            0.0,
            -0.0,
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
        ];
        for &value in samples {
            let text = xsd_canonical_double(value);
            // XSD canonical form uses "E" where Rust's parser expects "e"; substitute for the
            // round-trip check.
            let rust_parseable = text.replace('E', "e");
            let parsed: f64 = rust_parseable
                .parse()
                .expect("xsd_canonical_double output must parse as f64 (after E -> e)");
            assert_eq!(
                parsed.to_bits(),
                value.to_bits(),
                "round trip failed for {value:e} -> {text:?}"
            );
        }
    }
}
