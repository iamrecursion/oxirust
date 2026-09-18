//! [`fn:format-integer`](https://www.w3.org/TR/xpath-functions-31/#func-format-integer) and the
//! *format token* language of
//! [F&O 3.1 §4.6.1](https://www.w3.org/TR/xpath-functions-31/#formatting-integers).
//!
//! The date/time formatting functions of §9.8 are specified by reference to this one, so the
//! numbering sequences live here rather than beside `fn:format-dateTime`.
//!
//! The numbering sequences implemented are the ones the specification names explicitly:
//!
//! * *decimal digit patterns* such as `1`, `001` and `#'##0`, in any Unicode decimal digit
//!   family, with regular and irregular grouping separators;
//! * `a` and `A` — `a b c … z aa ab …`;
//! * `i` and `I` — Roman numerals (which have no representation above 3999, so larger values
//!   fall back on `1`, as the specification requires);
//! * `w`, `W` and `Ww` — English number words, with the `o` modifier producing ordinals.
//!
//! Any other format token is a numbering sequence this implementation does not support, which
//! is *not* an error: the value is formatted with the token `1` instead.

use crate::error::{FnError, FnErrorCode, FnResult};
use oxixml_unicode::gc::{GeneralCategory, general_category};

/// The parsed form of `fn:format-integer`'s second argument.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct IntegerPicture {
    primary: String,
    ordinal: bool,
    /// The parenthesized variation selector of the `c`/`o` modifier, which this implementation
    /// records but does not interpret.
    variation: Option<String>,
}

impl IntegerPicture {
    /// Parses a picture of the form `primary-token(";" modifier)?`.
    ///
    /// # Errors
    ///
    /// [`FnErrorCode::Fodf1310`] when the primary token is an invalid decimal digit pattern or
    /// the modifier does not match `^([co](\(.+\))?)?[at]?$`.
    ///
    /// # Examples
    ///
    /// ```
    /// use oxixml_fn::datetime::IntegerPicture;
    ///
    /// assert!(IntegerPicture::parse("0000").is_ok());
    /// assert!(IntegerPicture::parse("1;o").is_ok());
    /// assert!(IntegerPicture::parse("#;##0;").is_ok());
    /// assert!(IntegerPicture::parse("1;xyz").is_err());
    /// assert!(IntegerPicture::parse("0,,0").is_err());
    /// ```
    pub fn parse(picture: &str) -> FnResult<Self> {
        let (primary, modifier) = match picture.rsplit_once(';') {
            Some((primary, modifier)) => (primary, modifier),
            None => (picture, ""),
        };
        if primary.is_empty() {
            return Err(invalid("the primary format token is empty"));
        }
        let (ordinal, variation) = parse_modifier(modifier)?;
        if is_decimal_digit_pattern(primary) {
            validate_decimal_digit_pattern(primary)?;
        }
        Ok(Self {
            primary: primary.to_owned(),
            ordinal,
            variation,
        })
    }

    /// Builds a picture from a primary format token that has already been split off.
    ///
    /// [`IntegerPicture::parse`] reads `;` as the separator before `fn:format-integer`'s ordinal
    /// modifier. A *date/time* component's presentation modifier has no such syntax — F&O 3.1
    /// §9.8.4.2 lets `;` be an ordinary grouping separator there, so `[Y9;999]` is `2;012` —
    /// and that caller uses this constructor instead.
    ///
    /// # Errors
    ///
    /// [`FnErrorCode::Fodf1310`] when `primary` is not a usable format token.
    pub fn from_primary(primary: &str, ordinal: bool) -> FnResult<Self> {
        if primary.is_empty() {
            return Err(invalid("the primary format token is empty"));
        }
        if is_decimal_digit_pattern(primary) {
            validate_decimal_digit_pattern(primary)?;
        }
        Ok(Self {
            primary: primary.to_owned(),
            ordinal,
            variation: None,
        })
    }

    /// The primary format token.
    #[must_use]
    pub fn primary(&self) -> &str {
        &self.primary
    }

    /// Whether the `o` (ordinal) modifier was requested.
    #[must_use]
    pub const fn is_ordinal(&self) -> bool {
        self.ordinal
    }

    /// The parenthesized variation selector, if one was supplied.
    #[must_use]
    pub fn variation(&self) -> Option<&str> {
        self.variation.as_deref()
    }

    /// Formats `value` according to this picture.
    #[must_use]
    pub fn format(&self, value: i64) -> String {
        format_with_token(value, &self.primary, self.ordinal)
    }
}

fn invalid(reason: &str) -> FnError {
    FnError::new(
        FnErrorCode::Fodf1310,
        format!("invalid format token: {reason}"),
    )
}

fn parse_modifier(modifier: &str) -> FnResult<(bool, Option<String>)> {
    let mut rest = modifier;
    let mut ordinal = false;
    let mut variation = None;
    if let Some(first) = rest.chars().next() {
        if first == 'c' || first == 'o' {
            ordinal = first == 'o';
            rest = &rest[first.len_utf8()..];
            if rest.starts_with('(') {
                let close = rest
                    .find(')')
                    .ok_or_else(|| invalid("an unterminated variation selector"))?;
                if close < 2 {
                    return Err(invalid("an empty variation selector"));
                }
                variation = Some(rest[1..close].to_owned());
                rest = &rest[close + 1..];
            }
        }
    }
    match rest {
        "" | "a" | "t" => Ok((ordinal, variation)),
        other => Err(invalid(&format!(
            "`{other}` is not a valid format modifier"
        ))),
    }
}

/// The digit family zero of `character`, when it is a Unicode decimal digit.
#[must_use]
pub fn digit_family_zero(character: char) -> Option<char> {
    if general_category(character) != GeneralCategory::DecimalNumber {
        return None;
    }
    let code = u32::from(character);
    (0..10).find_map(|offset| {
        let candidate = char::from_u32(code.checked_sub(offset)?)?;
        let is_digit = general_category(candidate) == GeneralCategory::DecimalNumber;
        let previous_is_digit = char::from_u32(u32::from(candidate).wrapping_sub(1))
            .is_some_and(|previous| general_category(previous) == GeneralCategory::DecimalNumber);
        (is_digit && !previous_is_digit).then_some(candidate)
    })
}

fn is_digit_family_member(character: char) -> bool {
    general_category(character) == GeneralCategory::DecimalNumber
}

fn is_decimal_digit_pattern(token: &str) -> bool {
    token.chars().any(is_digit_family_member)
}

fn validate_decimal_digit_pattern(token: &str) -> FnResult<()> {
    let characters: Vec<char> = token.chars().collect();
    let is_digit_sign = |character: char| is_digit_family_member(character) || character == '#';
    let Some(first) = characters.first().copied() else {
        return Err(invalid("the primary format token is empty"));
    };
    let Some(last) = characters.last().copied() else {
        return Err(invalid("the primary format token is empty"));
    };
    if !is_digit_sign(first) {
        return Err(invalid(
            "a decimal digit pattern must start with a digit sign",
        ));
    }
    if !is_digit_family_member(last) {
        return Err(invalid(
            "a decimal digit pattern must end with a mandatory digit sign",
        ));
    }
    for window in characters.windows(2) {
        if !is_digit_sign(window[0]) && !is_digit_sign(window[1]) {
            return Err(invalid("two adjacent grouping separators"));
        }
    }
    // Every mandatory digit sign must come from the *same* decimal digit family (F&O 3.1
    // §4.6.1), so `123١` — Latin digits followed by an Arabic-Indic one — is FODF1310.
    let mut family = None;
    for character in characters.iter().copied() {
        let Some(zero) = digit_family_zero(character) else {
            continue;
        };
        match family {
            None => family = Some(zero),
            Some(seen) if seen == zero => {}
            Some(_) => {
                return Err(invalid(
                    "a decimal digit pattern mixes digits of more than one digit family",
                ));
            }
        }
    }
    // Optional digit signs may not follow mandatory ones.
    let first_mandatory = characters.iter().position(|c| is_digit_family_member(*c));
    let last_optional = characters.iter().rposition(|c| *c == '#');
    if let (Some(mandatory), Some(optional)) = (first_mandatory, last_optional) {
        if mandatory < optional {
            return Err(invalid(
                "an optional digit sign follows a mandatory digit sign",
            ));
        }
    }
    Ok(())
}

/// [`fn:format-integer`](https://www.w3.org/TR/xpath-functions-31/#func-format-integer).
///
/// # Errors
///
/// [`FnErrorCode::Fodf1310`] when the picture is invalid.
///
/// # Examples
///
/// ```
/// use oxixml_fn::datetime::format_integer;
///
/// assert_eq!(format_integer(123, "0000")?, "0123");
/// assert_eq!(format_integer(7, "a")?, "g");
/// assert_eq!(format_integer(57, "I")?, "LVII");
/// assert_eq!(format_integer(1234, "#;##0;")?, "1;234");
/// assert_eq!(format_integer(21, "1;o")?, "21st");
/// assert_eq!(format_integer(123, "w")?, "one hundred and twenty-three");
/// assert_eq!(format_integer(3, "Ww;o")?, "Third");
/// # Ok::<(), oxixml_fn::FnError>(())
/// ```
pub fn format_integer(value: i64, picture: &str) -> FnResult<String> {
    Ok(IntegerPicture::parse(picture)?.format(value))
}

fn format_with_token(value: i64, token: &str, ordinal: bool) -> String {
    if is_decimal_digit_pattern(token) {
        return decimal_pattern(value, token, ordinal);
    }
    let magnitude = value.unsigned_abs();
    let sign = if value < 0 { "-" } else { "" };
    let body = match token {
        "a" | "A" if magnitude > 0 => {
            let letters = alphabetic(magnitude);
            if token == "A" {
                letters.to_uppercase()
            } else {
                letters
            }
        }
        "i" | "I" if (1..4000).contains(&magnitude) => {
            let numerals = roman(magnitude);
            if token == "i" {
                numerals.to_lowercase()
            } else {
                numerals
            }
        }
        "w" => words(magnitude, ordinal),
        "W" => words(magnitude, ordinal).to_uppercase(),
        "Ww" => title_case(&words(magnitude, ordinal)),
        // Any unsupported numbering sequence falls back on the token `1`.
        _ => return decimal_pattern(value, "1", ordinal),
    };
    format!("{sign}{body}")
}

/// Formats using a decimal digit pattern: zero padding, digit family substitution and grouping.
fn decimal_pattern(value: i64, token: &str, ordinal: bool) -> String {
    let characters: Vec<char> = token.chars().collect();
    let zero = characters
        .iter()
        .copied()
        .find_map(digit_family_zero)
        .unwrap_or('0');
    let is_digit_sign = |character: char| is_digit_family_member(character) || character == '#';

    let mandatory = characters
        .iter()
        .copied()
        .filter(|c| is_digit_family_member(*c))
        .count();
    let digit_positions = characters
        .iter()
        .copied()
        .filter(|c| is_digit_sign(*c))
        .count();

    let mut separators: Vec<(usize, char)> = Vec::new();
    for (index, character) in characters.iter().copied().enumerate() {
        if !is_digit_sign(character) {
            let position = characters[index + 1..]
                .iter()
                .copied()
                .filter(|c| is_digit_sign(*c))
                .count();
            separators.push((position, character));
        }
    }
    let regular = regular_grouping(&separators, digit_positions);

    let magnitude = value.unsigned_abs().to_string();
    let mut digits: Vec<char> = magnitude.chars().collect();
    while digits.len() < mandatory {
        digits.insert(0, '0');
    }

    let mut rendered: Vec<char> = Vec::with_capacity(digits.len() * 2);
    for (index, digit) in digits.iter().copied().enumerate() {
        let from_the_right = digits.len() - index - 1;
        let value = digit.to_digit(10).unwrap_or(0);
        rendered.push(char::from_u32(u32::from(zero) + value).unwrap_or(digit));
        if from_the_right > 0 {
            if let Some((_, separator)) = separator_at(from_the_right, &separators, regular) {
                rendered.push(separator);
            }
        }
    }
    let body: String = rendered.into_iter().collect();
    let sign = if value < 0 { "-" } else { "" };
    if ordinal {
        format!("{sign}{body}{}", ordinal_suffix(value.unsigned_abs()))
    } else {
        format!("{sign}{body}")
    }
}

fn regular_grouping(separators: &[(usize, char)], digit_positions: usize) -> Option<(usize, char)> {
    let (smallest, character) = separators
        .iter()
        .copied()
        .min_by_key(|(position, _)| *position)?;
    if smallest == 0 {
        return None;
    }
    if !separators
        .iter()
        .all(|(position, other)| position % smallest == 0 && *other == character)
    {
        return None;
    }
    let mut multiple = smallest;
    while multiple < digit_positions {
        if !separators.iter().any(|(position, _)| *position == multiple) {
            return None;
        }
        multiple += smallest;
    }
    Some((smallest, character))
}

fn separator_at(
    position: usize,
    separators: &[(usize, char)],
    regular: Option<(usize, char)>,
) -> Option<(usize, char)> {
    match regular {
        Some((size, character)) if size > 0 && position % size == 0 => Some((size, character)),
        Some(_) => None,
        None => separators
            .iter()
            .copied()
            .find(|(candidate, _)| *candidate == position),
    }
}

/// The `a`/`A` sequence: `a b c … z aa ab …`.
fn alphabetic(mut value: u64) -> String {
    let mut letters = Vec::new();
    while value > 0 {
        let remainder = (value - 1) % 26;
        letters.push(char::from(b'a' + u8::try_from(remainder).unwrap_or(0)));
        value = (value - 1) / 26;
    }
    letters.iter().rev().collect()
}

/// The `i`/`I` sequence, defined for 1 to 3999.
fn roman(value: u64) -> String {
    const TABLE: [(u64, &str); 13] = [
        (1000, "M"),
        (900, "CM"),
        (500, "D"),
        (400, "CD"),
        (100, "C"),
        (90, "XC"),
        (50, "L"),
        (40, "XL"),
        (10, "X"),
        (9, "IX"),
        (5, "V"),
        (4, "IV"),
        (1, "I"),
    ];
    let mut remaining = value;
    let mut out = String::new();
    for (magnitude, numeral) in TABLE {
        while remaining >= magnitude {
            out.push_str(numeral);
            remaining -= magnitude;
        }
    }
    out
}

const UNITS: [&str; 20] = [
    "zero",
    "one",
    "two",
    "three",
    "four",
    "five",
    "six",
    "seven",
    "eight",
    "nine",
    "ten",
    "eleven",
    "twelve",
    "thirteen",
    "fourteen",
    "fifteen",
    "sixteen",
    "seventeen",
    "eighteen",
    "nineteen",
];

const TENS: [&str; 10] = [
    "", "", "twenty", "thirty", "forty", "fifty", "sixty", "seventy", "eighty", "ninety",
];

const SCALES: [(u64, &str); 5] = [
    (1_000_000_000_000_000, "quadrillion"),
    (1_000_000_000_000, "trillion"),
    (1_000_000_000, "billion"),
    (1_000_000, "million"),
    (1_000, "thousand"),
];

/// English number words, in the `w` sequence; `ordinal` selects the ordinal form.
fn words(value: u64, ordinal: bool) -> String {
    let cardinal = cardinal_words(value);
    if ordinal {
        ordinal_words(&cardinal)
    } else {
        cardinal
    }
}

fn cardinal_words(value: u64) -> String {
    if value < 20 {
        return UNITS[usize::try_from(value).unwrap_or(0)].to_owned();
    }
    if value < 100 {
        let tens = TENS[usize::try_from(value / 10).unwrap_or(0)];
        let unit = value % 10;
        return if unit == 0 {
            tens.to_owned()
        } else {
            format!("{tens}-{}", UNITS[usize::try_from(unit).unwrap_or(0)])
        };
    }
    if value < 1_000 {
        let hundreds = format!(
            "{} hundred",
            UNITS[usize::try_from(value / 100).unwrap_or(0)]
        );
        let remainder = value % 100;
        return if remainder == 0 {
            hundreds
        } else {
            format!("{hundreds} and {}", cardinal_words(remainder))
        };
    }
    for (magnitude, name) in SCALES {
        if value >= magnitude {
            let leading = format!("{} {name}", cardinal_words(value / magnitude));
            let remainder = value % magnitude;
            return if remainder == 0 {
                leading
            } else if remainder < 100 {
                format!("{leading} and {}", cardinal_words(remainder))
            } else {
                format!("{leading} {}", cardinal_words(remainder))
            };
        }
    }
    UNITS[0].to_owned()
}

fn ordinal_words(cardinal: &str) -> String {
    let split = cardinal.rfind(['-', ' ']).map_or(0, |index| {
        index + cardinal[index..].chars().next().map_or(1, char::len_utf8)
    });
    let (head, tail) = cardinal.split_at(split);
    format!("{head}{}", ordinal_word(tail))
}

fn ordinal_word(word: &str) -> String {
    match word {
        "one" => "first".to_owned(),
        "two" => "second".to_owned(),
        "three" => "third".to_owned(),
        "five" => "fifth".to_owned(),
        "eight" => "eighth".to_owned(),
        "nine" => "ninth".to_owned(),
        "twelve" => "twelfth".to_owned(),
        "zero" => "zeroth".to_owned(),
        other if other.ends_with('y') => format!("{}ieth", &other[..other.len() - 1]),
        other => format!("{other}th"),
    }
}

fn title_case(value: &str) -> String {
    let mut out = String::with_capacity(value.len());
    let mut at_word_start = true;
    for character in value.chars() {
        if at_word_start {
            out.extend(character.to_uppercase());
        } else {
            out.push(character);
        }
        at_word_start = character == ' ' || character == '-';
    }
    out
}

/// The English ordinal suffix for a decimal number: `st`, `nd`, `rd` or `th`.
#[must_use]
pub fn ordinal_suffix(value: u64) -> &'static str {
    match (value % 100, value % 10) {
        (11..=13, _) => "th",
        (_, 1) => "st",
        (_, 2) => "nd",
        (_, 3) => "rd",
        _ => "th",
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn examples_from_the_specification() -> FnResult<()> {
        assert_eq!(format_integer(123, "0000")?, "0123");
        assert_eq!(format_integer(7, "a")?, "g");
        assert_eq!(format_integer(57, "I")?, "LVII");
        assert_eq!(format_integer(1234, "#;##0;")?, "1;234");
        assert_eq!(format_integer(21, "1;o")?, "21st");
        assert_eq!(format_integer(123, "w")?, "one hundred and twenty-three");
        Ok(())
    }

    #[test]
    fn decimal_digit_patterns_pad_and_group() -> FnResult<()> {
        assert_eq!(format_integer(15, "0'000")?, "0'015");
        assert_eq!(format_integer(1_000_000, "0'000")?, "1'000'000");
        assert_eq!(format_integer(1_000_000, "#'##0")?, "1'000'000");
        assert_eq!(format_integer(15, "#'##0")?, "15");
        assert_eq!(format_integer(300, "01")?, "300");
        assert_eq!(format_integer(-42, "000")?, "-042");
        Ok(())
    }

    #[test]
    fn alphabetic_and_roman_sequences() -> FnResult<()> {
        assert_eq!(format_integer(1, "a")?, "a");
        assert_eq!(format_integer(26, "a")?, "z");
        assert_eq!(format_integer(27, "a")?, "aa");
        assert_eq!(format_integer(28, "A")?, "AB");
        assert_eq!(format_integer(4, "i")?, "iv");
        assert_eq!(format_integer(1990, "I")?, "MCMXC");
        // Out of range for Roman numerals: falls back on the token `1`.
        assert_eq!(format_integer(4000, "I")?, "4000");
        assert_eq!(format_integer(0, "i")?, "0");
        Ok(())
    }

    #[test]
    fn english_words_and_ordinals() -> FnResult<()> {
        assert_eq!(format_integer(1, "w")?, "one");
        assert_eq!(format_integer(20, "w")?, "twenty");
        assert_eq!(format_integer(21, "w")?, "twenty-one");
        assert_eq!(format_integer(100, "w")?, "one hundred");
        assert_eq!(format_integer(1_000_000, "w")?, "one million");
        assert_eq!(
            format_integer(1_234, "w")?,
            "one thousand two hundred and thirty-four"
        );
        assert_eq!(format_integer(3, "Ww;o")?, "Third");
        assert_eq!(format_integer(21, "w;o")?, "twenty-first");
        assert_eq!(format_integer(40, "w;o")?, "fortieth");
        assert_eq!(format_integer(5, "W")?, "FIVE");
        Ok(())
    }

    #[test]
    fn ordinal_suffixes_handle_the_teens() {
        assert_eq!(ordinal_suffix(1), "st");
        assert_eq!(ordinal_suffix(2), "nd");
        assert_eq!(ordinal_suffix(3), "rd");
        assert_eq!(ordinal_suffix(4), "th");
        assert_eq!(ordinal_suffix(11), "th");
        assert_eq!(ordinal_suffix(12), "th");
        assert_eq!(ordinal_suffix(13), "th");
        assert_eq!(ordinal_suffix(21), "st");
        assert_eq!(ordinal_suffix(111), "th");
    }

    #[test]
    fn invalid_pictures_raise_fodf1310() {
        for picture in ["0,,0", ",000", "000,", "0#0", "1;xyz", ""] {
            assert!(
                format_integer(1, picture).is_err(),
                "`{picture}` should be rejected"
            );
        }
    }

    #[test]
    fn unsupported_numbering_sequences_fall_back_on_decimal() -> FnResult<()> {
        assert_eq!(format_integer(42, "\u{2460}")?, "42");
        Ok(())
    }

    #[test]
    fn digit_families_other_than_ascii_are_supported() -> FnResult<()> {
        // Arabic-Indic digit one selects the Arabic-Indic family.
        assert_eq!(format_integer(123, "\u{661}")?, "\u{661}\u{662}\u{663}");
        assert_eq!(digit_family_zero('\u{663}'), Some('\u{660}'));
        assert_eq!(digit_family_zero('x'), None);
        Ok(())
    }
}
