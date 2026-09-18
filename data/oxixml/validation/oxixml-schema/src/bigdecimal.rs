//! An exact, arbitrary-precision decimal.
//!
//! The XSD value space of `decimal` — and therefore of `integer` and everything
//! below it — is *unbounded*: `unsignedLong` alone already exceeds [`i64`], and
//! `xs:integer` has no bounds at all. `totalDigits`, `fractionDigits` and the
//! four bounds facets are therefore evaluated on the decimal digits themselves,
//! and nothing in this module can overflow.

use std::cmp::Ordering;
use std::fmt;

/// A decimal number held as a sign plus its integer and fraction digits.
///
/// The representation is canonical: `integer` never carries leading zeros (it is
/// `"0"` for magnitudes below one), `fraction` never carries trailing zeros, and
/// zero is never negative. Consequently [`PartialEq`] *is* value-space equality:
/// `1.0` and `1.00` are the same `BigDecimal`.
///
/// # Examples
///
/// ```
/// use oxixml_schema::BigDecimal;
///
/// let a = BigDecimal::parse("+1.2500").expect("a decimal");
/// assert_eq!(a.to_string(), "1.25");
/// assert_eq!(a, BigDecimal::parse("1.25").expect("a decimal"));
/// assert!(BigDecimal::parse("18446744073709551615").expect("a decimal")
///     > BigDecimal::parse("9223372036854775807").expect("a decimal"));
/// ```
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct BigDecimal {
    negative: bool,
    integer: String,
    fraction: String,
}

impl BigDecimal {
    /// Parses the XSD `decimal` lexical form: an optional sign, then digits with
    /// an optional fractional part. At least one digit is required overall.
    ///
    /// Returns [`None`] when `lexical` is not in the lexical space.
    #[must_use]
    pub fn parse(lexical: &str) -> Option<Self> {
        let (negative, digits) = match lexical.as_bytes().first() {
            Some(b'-') => (true, &lexical[1..]),
            Some(b'+') => (false, &lexical[1..]),
            _ => (false, lexical),
        };
        let (integer, fraction) = match digits.split_once('.') {
            Some((integer, fraction)) => (integer, fraction),
            None => (digits, ""),
        };
        if integer.is_empty() && fraction.is_empty() {
            return None;
        }
        if !integer.bytes().all(|b| b.is_ascii_digit())
            || !fraction.bytes().all(|b| b.is_ascii_digit())
        {
            return None;
        }
        Some(Self::normalized(negative, integer, fraction))
    }

    /// Parses the XSD `integer` lexical form: an optional sign and digits only.
    #[must_use]
    pub fn parse_integer(lexical: &str) -> Option<Self> {
        if lexical.contains('.') {
            return None;
        }
        Self::parse(lexical)
    }

    /// The value zero.
    #[must_use]
    pub fn zero() -> Self {
        Self {
            negative: false,
            integer: "0".to_owned(),
            fraction: String::new(),
        }
    }

    fn normalized(negative: bool, integer: &str, fraction: &str) -> Self {
        let integer = integer.trim_start_matches('0');
        let fraction = fraction.trim_end_matches('0');
        let integer = if integer.is_empty() {
            "0".to_owned()
        } else {
            integer.to_owned()
        };
        let is_zero = integer == "0" && fraction.is_empty();
        Self {
            negative: negative && !is_zero,
            integer,
            fraction: fraction.to_owned(),
        }
    }

    /// The number of significant digits, as the `totalDigits` facet counts them:
    /// the integer digits without leading zeros plus the fraction digits without
    /// trailing zeros, and at least one.
    #[must_use]
    pub fn total_digits(&self) -> usize {
        let integer = if self.integer == "0" {
            0
        } else {
            self.integer.len()
        };
        (integer + self.fraction.len()).max(1)
    }

    /// The number of fraction digits, ignoring trailing zeros.
    #[must_use]
    pub fn fraction_digits(&self) -> usize {
        self.fraction.len()
    }

    /// Returns `true` when the value is strictly negative.
    #[must_use]
    pub fn is_negative(&self) -> bool {
        self.negative
    }

    /// Returns `true` when the value is zero.
    #[must_use]
    pub fn is_zero(&self) -> bool {
        self.integer == "0" && self.fraction.is_empty()
    }

    /// Returns `true` when the value has no fractional part.
    #[must_use]
    pub fn is_integral(&self) -> bool {
        self.fraction.is_empty()
    }

    /// Converts to a [`usize`], if the value is a non-negative integer that fits.
    #[must_use]
    pub fn to_usize(&self) -> Option<usize> {
        if self.negative || !self.fraction.is_empty() {
            return None;
        }
        self.integer.parse().ok()
    }

    /// Converts to an [`f64`], saturating to infinity on overflow.
    #[must_use]
    pub fn to_f64(&self) -> f64 {
        self.to_string().parse().unwrap_or(f64::NAN)
    }

    fn magnitude_cmp(&self, other: &Self) -> Ordering {
        self.integer
            .len()
            .cmp(&other.integer.len())
            .then_with(|| self.integer.cmp(&other.integer))
            .then_with(|| {
                let width = self.fraction.len().max(other.fraction.len());
                pad(&self.fraction, width).cmp(&pad(&other.fraction, width))
            })
    }
}

fn pad(digits: &str, width: usize) -> String {
    let mut padded = String::with_capacity(width);
    padded.push_str(digits);
    while padded.len() < width {
        padded.push('0');
    }
    padded
}

impl PartialOrd for BigDecimal {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for BigDecimal {
    fn cmp(&self, other: &Self) -> Ordering {
        match (self.negative, other.negative) {
            (false, true) => Ordering::Greater,
            (true, false) => Ordering::Less,
            (false, false) => self.magnitude_cmp(other),
            (true, true) => other.magnitude_cmp(self),
        }
    }
}

impl fmt::Display for BigDecimal {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.negative {
            f.write_str("-")?;
        }
        f.write_str(&self.integer)?;
        if !self.fraction.is_empty() {
            write!(f, ".{}", self.fraction)?;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn parsed(lexical: &str) -> BigDecimal {
        BigDecimal::parse(lexical).expect("a valid decimal")
    }

    #[test]
    fn lexical_variants_normalize() {
        assert_eq!(parsed("+1.2500").to_string(), "1.25");
        assert_eq!(parsed("0001").to_string(), "1");
        assert_eq!(parsed("-0.0").to_string(), "0");
        assert_eq!(parsed(".5").to_string(), "0.5");
        assert_eq!(parsed("5.").to_string(), "5");
    }

    #[test]
    fn equality_is_on_the_value() {
        assert_eq!(parsed("1.0"), parsed("1.00"));
        assert_ne!(parsed("1"), parsed("1.001"));
    }

    #[test]
    fn ordering_handles_signs_and_widths() {
        assert!(parsed("-2") < parsed("-1"));
        assert!(parsed("9") < parsed("10"));
        assert!(parsed("1.09") < parsed("1.1"));
    }

    #[test]
    fn digit_counts() {
        assert_eq!(parsed("123.45").total_digits(), 5);
        assert_eq!(parsed("123.45").fraction_digits(), 2);
        assert_eq!(parsed("0").total_digits(), 1);
        assert_eq!(parsed("0.001").total_digits(), 3);
    }

    #[test]
    fn malformed_input_is_rejected() {
        assert!(BigDecimal::parse("").is_none());
        assert!(BigDecimal::parse("+").is_none());
        assert!(BigDecimal::parse(".").is_none());
        assert!(BigDecimal::parse("1e5").is_none());
        assert!(BigDecimal::parse_integer("1.0").is_none());
        assert!(BigDecimal::parse_integer("12").is_some());
    }
}
