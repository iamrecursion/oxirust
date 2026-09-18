//! [`Count`] — one value of an `xsl:number` place marker.
//!
//! The counting forms of the instruction (`level`, `count`, `from`) produce positions within a
//! tree, which fit a machine integer with room to spare. The `value` attribute does not:
//! [XSLT 3.0 §12.3](https://www.w3.org/TR/xslt-30/#number) converts its result to an
//! `xs:integer`, and XPath's integers are unbounded — `number-0807` writes `value="1e100"` and
//! wants the hundred-and-one digits back rather than a saturated `i64`.
//!
//! So a place-marker value is either a machine integer or the decimal digits of one that is not.
//! Every numbering sequence that has a range of its own — Roman numerals, words, the alphabetic
//! sequences — keeps working on the first and falls back to decimal notation on the second, which
//! is what §12.3 licenses in as many words: a processor may use decimal notation "if the
//! numbering sequence does not extend far enough".

/// One value of an `xsl:number` place marker.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum Count {
    /// A value that fits a machine integer.
    Small(i64),
    /// A value that does not, held as `(negative, decimal digits)` with no sign, no separators
    /// and no leading zero.
    Large(bool, String),
}

impl From<i64> for Count {
    fn from(value: i64) -> Self {
        Self::Small(value)
    }
}

impl std::fmt::Display for Count {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Small(value) => write!(f, "{value}"),
            Self::Large(true, digits) => write!(f, "-{digits}"),
            Self::Large(false, digits) => write!(f, "{digits}"),
        }
    }
}

impl Count {
    /// The value a lexical form denotes, when it is a plain decimal integer.
    ///
    /// "Plain" excludes everything a `xs:double` or `xs:decimal` may write — a decimal point, an
    /// exponent, `INF`, `NaN` — because those are values [`Count::from_f64`] settles; what is
    /// kept here is the exactness of an integer that no `f64` could carry.
    pub(crate) fn parse_integer(lexical: &str) -> Option<Self> {
        let text = lexical.trim();
        let (negative, digits) = match text.strip_prefix('-') {
            Some(rest) => (true, rest),
            None => (false, text.strip_prefix('+').unwrap_or(text)),
        };
        if digits.is_empty() || !digits.bytes().all(|byte| byte.is_ascii_digit()) {
            return None;
        }
        if let Ok(value) = text.parse::<i64>() {
            return Some(Self::Small(value));
        }
        let trimmed = digits.trim_start_matches('0');
        Some(match trimmed.is_empty() {
            true => Self::Small(0),
            false => Self::Large(negative, trimmed.to_owned()),
        })
    }

    /// The integer a double rounds to, exactly.
    ///
    /// Casting an `xs:double` to `xs:integer` truncates towards zero and keeps every digit of the
    /// result — "the integer that is exactly equal to the mantissa times 2 to the power of the
    /// unbiased exponent", which for `1e100` is not a power of ten. The formatter writes exactly
    /// that, so the conversion has to be exact here too.
    pub(crate) fn from_f64(value: f64) -> Self {
        if !value.is_finite() {
            return Self::Small(0);
        }
        let rounded = value.round();
        // `as` saturates rather than wrapping, so the guard is what tells a representable value
        // from one that would silently become `i64::MAX`.
        if rounded.abs() < 9.223_372_036_854_775_e18 {
            return Self::Small(rounded as i64);
        }
        // `{:.0}` writes the exact decimal expansion of a binary double, which is what the cast
        // to `xs:integer` is defined to produce.
        let text = format!("{rounded:.0}");
        Self::parse_integer(&text).unwrap_or(Self::Small(0))
    }

    /// The value as a machine integer, or `None` when it is too large for one.
    pub(crate) const fn as_i64(&self) -> Option<i64> {
        match self {
            Self::Small(value) => Some(*value),
            Self::Large(_, _) => None,
        }
    }

    /// Whether the value is below zero.
    pub(crate) const fn is_negative(&self) -> bool {
        match self {
            Self::Small(value) => *value < 0,
            Self::Large(negative, _) => *negative,
        }
    }

    /// The decimal digits of the value's magnitude, with no sign.
    pub(crate) fn magnitude(&self) -> String {
        match self {
            // The magnitude is taken unsigned so that the smallest representable integer, whose
            // negation does not fit, still prints its digits rather than overflowing.
            Self::Small(value) => value.unsigned_abs().to_string(),
            Self::Large(_, digits) => digits.clone(),
        }
    }

    /// The value `start` renumbers this one to, or `None` when a machine integer would overflow.
    ///
    /// `xsl:number/@start-at` moves the whole place marker, so `start-at="0"` numbers from zero:
    /// the value is `self + start - 1`. Both ends of the integer range reach that expression
    /// straight from the source document, so a machine integer is added and decremented in two
    /// checked steps rather than one — `start-at="-9223372036854775808"` on the first item is in
    /// range and must not report an overflow on the way there.
    ///
    /// The offset is itself an integer of ordinary size, which is why a decimal-string addition
    /// of one machine integer is all the arithmetic a value beyond that range needs.
    pub(crate) fn shifted_to_start(&self, start: i64) -> Option<Self> {
        self.shifted(start)?.shifted(-1)
    }

    /// The value plus `offset`, or `None` when a machine integer would overflow.
    fn shifted(&self, offset: i64) -> Option<Self> {
        match self {
            Self::Small(value) => value.checked_add(offset).map(Self::Small),
            Self::Large(negative, digits) => {
                // A magnitude beyond `i64` shifted by a signed offset: the offset acts on the
                // magnitude in the direction the sign gives it. Negating it cannot overflow
                // because a magnitude this large is never reached from a counting form, so the
                // sign is carried by the offset alone.
                let towards = match negative {
                    true => 0_i128 - i128::from(offset),
                    false => i128::from(offset),
                };
                let magnitude = match towards >= 0 {
                    true => Some(add_magnitude(digits, towards.unsigned_abs() as u64)),
                    false => subtract_magnitude(digits, towards.unsigned_abs() as u64),
                }?;
                Some(match magnitude.trim_start_matches('0') {
                    "" => Self::Small(0),
                    trimmed => Self::Large(*negative, trimmed.to_owned()),
                })
            }
        }
    }
}

/// `digits + delta`, both non-negative, as decimal digits.
fn add_magnitude(digits: &str, delta: u64) -> String {
    let mut carry = u128::from(delta);
    let mut out: Vec<u8> = Vec::with_capacity(digits.len() + 20);
    for byte in digits.bytes().rev() {
        let digit = u128::from(byte - b'0') + carry % 10;
        carry /= 10;
        out.push(b'0' + (digit % 10) as u8);
        carry += digit / 10;
    }
    while carry > 0 {
        out.push(b'0' + (carry % 10) as u8);
        carry /= 10;
    }
    out.reverse();
    String::from_utf8(out).unwrap_or_default()
}

/// `digits - delta`, both non-negative, as decimal digits, or `None` if the result would be
/// negative.
fn subtract_magnitude(digits: &str, delta: u64) -> Option<String> {
    let mut borrow = u128::from(delta);
    let mut out: Vec<u8> = Vec::with_capacity(digits.len());
    for byte in digits.bytes().rev() {
        let mut digit = i32::from(byte - b'0') - (borrow % 10) as i32;
        borrow /= 10;
        if digit < 0 {
            digit += 10;
            borrow += 1;
        }
        out.push(b'0' + digit as u8);
    }
    if borrow > 0 {
        return None;
    }
    out.reverse();
    String::from_utf8(out).ok()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_double_beyond_the_integer_range_keeps_every_digit() {
        let count = Count::from_f64(1e100);
        assert_eq!(count.as_i64(), None);
        assert_eq!(count.magnitude().len(), 101);
        assert!(
            count
                .magnitude()
                .starts_with("1000000000000000015902891109759918")
        );
        assert!(!count.is_negative());
    }

    #[test]
    fn an_ordinary_double_is_a_machine_integer() {
        assert_eq!(Count::from_f64(7.4), Count::Small(7));
        assert_eq!(Count::from_f64(-7.6), Count::Small(-8));
        assert_eq!(Count::from_f64(f64::NAN), Count::Small(0));
    }

    #[test]
    fn a_lexical_integer_keeps_its_exact_value() {
        assert_eq!(Count::parse_integer(" 42 "), Some(Count::Small(42)));
        assert_eq!(Count::parse_integer("-42"), Some(Count::Small(-42)));
        assert_eq!(Count::parse_integer("000"), Some(Count::Small(0)));
        assert_eq!(Count::parse_integer("1.5"), None);
        assert_eq!(Count::parse_integer("1e100"), None);
        assert_eq!(Count::parse_integer(""), None);
        let big = "123456789012345678901234567890";
        assert_eq!(
            Count::parse_integer(big),
            Some(Count::Large(false, big.to_owned()))
        );
    }

    #[test]
    fn a_shift_carries_across_the_whole_number() {
        let big = Count::Large(false, "999999999999999999999".to_owned());
        assert_eq!(
            big.shifted_to_start(2),
            Some(Count::Large(false, "1000000000000000000000".to_owned()))
        );
        assert_eq!(
            big.shifted_to_start(0),
            Some(Count::Large(false, "999999999999999999998".to_owned()))
        );
        assert_eq!(Count::Small(5).shifted_to_start(0), Some(Count::Small(4)));
        assert_eq!(Count::Small(i64::MAX).shifted_to_start(2), None);
        // `value + start - 1` is computed in two checked steps, so the smallest representable
        // start is in range for the first item rather than overflowing on the way there.
        assert_eq!(
            Count::Small(1).shifted_to_start(i64::MIN),
            Some(Count::Small(i64::MIN))
        );
    }

    #[test]
    fn a_negative_large_value_shifts_away_from_zero() {
        let big = Count::Large(true, "1000000000000000000000".to_owned());
        assert_eq!(
            big.shifted_to_start(2),
            Some(Count::Large(true, "999999999999999999999".to_owned()))
        );
    }
}
