//! [`fn:format-number`](https://www.w3.org/TR/xpath-functions-31/#func-format-number) and the
//! decimal-format picture-string language of
//! [F&O 3.1 §4.7](https://www.w3.org/TR/xpath-functions-31/#formatting-numbers).
//!
//! The specification splits the work into two phases, and so does this module:
//!
//! * an **analysis** phase ([§4.7.4](https://www.w3.org/TR/xpath-functions-31/#analyzing-picture-string))
//!   that validates the picture string against eleven syntactic rules and reduces each
//!   sub-picture to the dozen variables the second phase consumes;
//! * a **formatting** phase ([§4.7.5](https://www.w3.org/TR/xpath-functions-31/#formatting-the-number))
//!   that scales the number, rounds it half-to-even, lays out the digits and inserts the
//!   grouping separators.
//!
//! Rounding is done on an *unbounded* decimal representation, as the specification requires:
//! `xs:decimal`'s eighteen fractional digits are a property of the datatype, not of the
//! formatter, so `fn:format-number(1 div 3, '0.00000000000000000000000')` keeps producing digits
//! rather than truncating.

use crate::error::{FnError, FnErrorCode, FnResult};
use oxixml_infoset::{AtomicType, AtomicValue};
use std::fmt;

/// The set of properties a decimal format supplies, as listed in
/// [F&O 3.1 §4.7.1](https://www.w3.org/TR/xpath-functions-31/#defining-decimal-format).
///
/// # Examples
///
/// ```
/// use oxixml_fn::numeric::DecimalFormat;
///
/// // The defaults are the ones XSLT and XQuery declare for the unnamed format.
/// let format = DecimalFormat::default();
/// assert_eq!(format.decimal_separator(), '.');
/// assert_eq!(format.infinity(), "Infinity");
///
/// // A European format.
/// let european = DecimalFormat::default()
///     .with_decimal_separator(',')
///     .with_grouping_separator('.');
/// assert_eq!(european.grouping_separator(), '.');
/// ```
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DecimalFormat {
    decimal_separator: char,
    grouping_separator: char,
    exponent_separator: char,
    infinity: String,
    minus_sign: char,
    nan: String,
    percent: char,
    per_mille: char,
    zero_digit: char,
    digit: char,
    pattern_separator: char,
}

impl Default for DecimalFormat {
    /// The property values every processor starts from: `.` `,` `e` `Infinity` `-` `NaN` `%`
    /// `‰` `0` `#` `;`.
    fn default() -> Self {
        Self {
            decimal_separator: '.',
            grouping_separator: ',',
            exponent_separator: 'e',
            infinity: "Infinity".to_owned(),
            minus_sign: '-',
            nan: "NaN".to_owned(),
            percent: '%',
            per_mille: '\u{2030}',
            zero_digit: '0',
            digit: '#',
            pattern_separator: ';',
        }
    }
}

macro_rules! decimal_format_properties {
    ($( $field:ident : $ty:ty, $with:ident, $doc:literal ),+ $(,)?) => {
        impl DecimalFormat {
            $(
                #[doc = concat!("The `", $doc, "` property.")]
                #[must_use]
                pub fn $field(&self) -> $ty {
                    self.$field.clone()
                }

                #[doc = concat!("Replaces the `", $doc, "` property.")]
                #[must_use]
                pub fn $with(mut self, value: impl Into<$ty>) -> Self {
                    self.$field = value.into();
                    self
                }
            )+
        }
    };
}

decimal_format_properties! {
    decimal_separator: char, with_decimal_separator, "decimal-separator",
    grouping_separator: char, with_grouping_separator, "grouping-separator",
    exponent_separator: char, with_exponent_separator, "exponent-separator",
    infinity: String, with_infinity, "infinity",
    minus_sign: char, with_minus_sign, "minus-sign",
    nan: String, with_nan, "NaN",
    percent: char, with_percent, "percent",
    per_mille: char, with_per_mille, "per-mille",
    zero_digit: char, with_zero_digit, "zero-digit",
    digit: char, with_digit, "digit",
    pattern_separator: char, with_pattern_separator, "pattern-separator",
}

impl DecimalFormat {
    /// Whether `character` is one of the ten digits of this format's *decimal digit family* —
    /// the ten consecutive code points starting at `zero-digit`.
    #[must_use]
    pub fn is_family_digit(&self, character: char) -> bool {
        self.family_value(character).is_some()
    }

    /// The numeric value of `character` when it belongs to the decimal digit family.
    #[must_use]
    pub fn family_value(&self, character: char) -> Option<u8> {
        let offset = u32::from(character).checked_sub(u32::from(self.zero_digit))?;
        u8::try_from(offset).ok().filter(|value| *value < 10)
    }

    /// The digit-family character for the value `digit`, which must be less than ten.
    #[must_use]
    pub fn family_digit(&self, digit: u8) -> char {
        char::from_u32(u32::from(self.zero_digit) + u32::from(digit.min(9)))
            .unwrap_or(self.zero_digit)
    }

    /// Checks the constraint of §4.7.1 that the characters used in a picture string must be
    /// distinct from one another and from every member of the decimal digit family.
    ///
    /// # Errors
    ///
    /// [`FnErrorCode::Fodf1280`] when two properties collide.
    pub fn validate(&self) -> FnResult<()> {
        let picture_characters = [
            ("decimal-separator", self.decimal_separator),
            ("grouping-separator", self.grouping_separator),
            ("exponent-separator", self.exponent_separator),
            ("percent", self.percent),
            ("per-mille", self.per_mille),
            ("digit", self.digit),
            ("pattern-separator", self.pattern_separator),
        ];
        for (index, (name, character)) in picture_characters.iter().enumerate() {
            if self.is_family_digit(*character) {
                return Err(FnError::new(
                    FnErrorCode::Fodf1280,
                    format!(
                        "the decimal format property {name} is `{character}`, which is a member of the decimal digit family"
                    ),
                ));
            }
            for (other_name, other) in &picture_characters[index + 1..] {
                if character == other {
                    return Err(FnError::new(
                        FnErrorCode::Fodf1280,
                        format!(
                            "the decimal format properties {name} and {other_name} are both `{character}`"
                        ),
                    ));
                }
            }
        }
        Ok(())
    }
}

/// An arbitrary-precision decimal number: `sign × digits × 10^scale`.
///
/// The specification requires the formatter to work with "an implementation of `xs:decimal`
/// that imposes no limits on the `totalDigits` or `fractionDigits` facets", which is exactly
/// what this is.
#[derive(Debug, Clone, PartialEq, Eq)]
struct BigDecimal {
    negative: bool,
    /// Significant digits, most significant first, with no leading or trailing zero.
    digits: Vec<u8>,
    scale: i32,
}

impl BigDecimal {
    fn zero(negative: bool) -> Self {
        Self {
            negative,
            digits: Vec::new(),
            scale: 0,
        }
    }

    fn is_zero(&self) -> bool {
        self.digits.is_empty()
    }

    fn normalize(&mut self) {
        while self.digits.first() == Some(&0) {
            self.digits.remove(0);
        }
        while self.digits.last() == Some(&0) {
            self.digits.pop();
            self.scale += 1;
        }
        if self.digits.is_empty() {
            self.scale = 0;
        }
    }

    /// Parses a lexical form of the shape `[-+]?digits[.digits][(e|E)[-+]?digits]`.
    fn parse(lexical: &str) -> Option<Self> {
        let mut characters = lexical.chars().peekable();
        let mut negative = false;
        match characters.peek() {
            Some('-') => {
                negative = true;
                characters.next();
            }
            Some('+') => {
                characters.next();
            }
            _ => {}
        }
        let mut digits = Vec::new();
        let mut fraction_digits = 0_i32;
        let mut seen_point = false;
        let mut exponent = 0_i32;
        let mut seen_digit = false;
        while let Some(character) = characters.peek().copied() {
            match character {
                '0'..='9' => {
                    seen_digit = true;
                    digits.push(u8::try_from(u32::from(character) - u32::from('0')).ok()?);
                    if seen_point {
                        fraction_digits += 1;
                    }
                    characters.next();
                }
                '.' if !seen_point => {
                    seen_point = true;
                    characters.next();
                }
                'e' | 'E' => {
                    characters.next();
                    let rest: String = characters.by_ref().collect();
                    exponent = rest.parse::<i32>().ok()?;
                    break;
                }
                _ => return None,
            }
        }
        if !seen_digit {
            return None;
        }
        let mut value = Self {
            negative,
            digits,
            scale: exponent - fraction_digits,
        };
        value.normalize();
        Some(value)
    }

    /// `self × 10^places`.
    fn scale_by(&mut self, places: i32) {
        if !self.is_zero() {
            self.scale += places;
        }
    }

    /// The base-ten exponent of the most significant digit: the `n` in `d.ddd × 10^n`.
    fn magnitude(&self) -> i32 {
        i32::try_from(self.digits.len()).unwrap_or(i32::MAX) + self.scale - 1
    }

    /// Rounds to at most `fraction_digits` digits after the decimal separator, resolving ties
    /// towards the even digit.
    fn round_half_to_even(&mut self, fraction_digits: usize) {
        let target_scale = -i32::try_from(fraction_digits).unwrap_or(i32::MAX);
        if self.scale >= target_scale || self.is_zero() {
            return;
        }
        let drop = usize::try_from(target_scale - self.scale).unwrap_or(usize::MAX);
        if drop > self.digits.len() {
            // Everything is dropped and the value is strictly less than half a unit.
            *self = Self::zero(self.negative);
            self.scale = target_scale;
            return;
        }
        while self.digits.len() < drop + 1 {
            self.digits.insert(0, 0);
        }
        let keep = self.digits.len() - drop;
        let dropped: Vec<u8> = self.digits.split_off(keep);
        let round_up = match dropped.first().copied().unwrap_or(0) {
            digit if digit > 5 => true,
            5 => {
                dropped[1..].iter().any(|digit| *digit != 0)
                    || self.digits.last().copied().unwrap_or(0) % 2 == 1
            }
            _ => false,
        };
        self.scale = target_scale;
        if round_up {
            self.increment();
        }
        self.normalize();
    }

    /// Adds one unit in the last place.
    fn increment(&mut self) {
        for digit in self.digits.iter_mut().rev() {
            if *digit == 9 {
                *digit = 0;
            } else {
                *digit += 1;
                return;
            }
        }
        self.digits.insert(0, 1);
    }

    /// Splits into (integer digits, fractional digits), both with the redundant zeros removed.
    fn split(&self) -> (Vec<u8>, Vec<u8>) {
        if self.is_zero() {
            return (Vec::new(), Vec::new());
        }
        let point = i32::try_from(self.digits.len()).unwrap_or(i32::MAX) + self.scale;
        if point <= 0 {
            let mut fraction = vec![0_u8; usize::try_from(-point).unwrap_or(0)];
            fraction.extend_from_slice(&self.digits);
            (Vec::new(), fraction)
        } else if let Ok(point) = usize::try_from(point) {
            if point >= self.digits.len() {
                let mut integer = self.digits.clone();
                integer.resize(point, 0);
                (integer, Vec::new())
            } else {
                (self.digits[..point].to_vec(), self.digits[point..].to_vec())
            }
        } else {
            (self.digits.clone(), Vec::new())
        }
    }
}

/// The variables §4.7.4 derives from one sub-picture.
#[derive(Debug, Clone, Default)]
struct SubPicture {
    prefix: String,
    suffix: String,
    integer_grouping: Vec<usize>,
    regular_grouping: Option<usize>,
    fractional_grouping: Vec<usize>,
    minimum_integer_part_size: usize,
    scaling_factor: usize,
    minimum_fractional_part_size: usize,
    maximum_fractional_part_size: usize,
    minimum_exponent_size: usize,
    percent: bool,
    per_mille: bool,
}

fn picture_error(reason: impl fmt::Display) -> FnError {
    FnError::new(
        FnErrorCode::Fodf1310,
        format!("invalid picture string: {reason}"),
    )
}

impl SubPicture {
    fn analyze(picture: &[char], format: &DecimalFormat) -> FnResult<Self> {
        // F&O 3.1 §4.7.4 classifies a character as *active* when it is a member of the decimal
        // digit family, or is the decimal-separator-sign, grouping-sign, optional-digit-sign,
        // pattern-separator-sign — or the **exponent-separator-sign**, which is one *particular*
        // occurrence of the exponent-separator character rather than every occurrence of it. An
        // `e` that is not that occurrence is an ordinary passive character, which is what makes
        // `0.0000eDog` a picture with the literal suffix `eDog` rather than an error.
        let is_core_active = |character: char| {
            character == format.decimal_separator
                || character == format.grouping_separator
                || character == format.digit
                || character == format.pattern_separator
                || format.is_family_digit(character)
        };

        // Which occurrence of the exponent separator, if any, is the exponent-separator-sign?
        let mut exponent_sign = None;
        for (index, character) in picture.iter().copied().enumerate() {
            if character != format.exponent_separator {
                continue;
            }
            let preceded = picture[..index].iter().copied().any(is_core_active);
            let followed = picture[index + 1..].iter().copied().any(is_core_active);
            if preceded && followed {
                if exponent_sign.is_some() {
                    return Err(picture_error(
                        "a sub-picture has more than one exponent separator",
                    ));
                }
                exponent_sign = Some(index);
            }
        }

        let is_active_at = |index: usize, character: char| {
            is_core_active(character) || exponent_sign == Some(index)
        };
        let is_active_anywhere = |slice: &[char], offset: usize| {
            slice
                .iter()
                .copied()
                .enumerate()
                .any(|(index, character)| is_active_at(offset + index, character))
        };

        let (mantissa, exponent) = match exponent_sign {
            Some(index) => {
                // The exponent part runs from the exponent-separator-sign to the start of the
                // sub-picture's suffix, which is its trailing run of passive characters.
                let tail = &picture[index + 1..];
                let active_end = tail
                    .iter()
                    .copied()
                    .rposition(is_core_active)
                    .map_or(0, |last| last + 1);
                (&picture[..index], &tail[..active_end])
            }
            None => (picture, &picture[picture.len()..]),
        };

        let decimal_positions: Vec<usize> = mantissa
            .iter()
            .enumerate()
            .filter(|(_, character)| **character == format.decimal_separator)
            .map(|(index, _)| index)
            .collect();
        if decimal_positions.len() > 1 {
            return Err(picture_error(
                "a sub-picture has more than one decimal separator",
            ));
        }
        let (integer, fraction) = match decimal_positions.first() {
            Some(index) => (&mantissa[..*index], &mantissa[index + 1..]),
            None => (mantissa, &mantissa[mantissa.len()..]),
        };

        let percent = picture.contains(&format.percent);
        let per_mille = picture.contains(&format.per_mille);
        if percent && per_mille {
            return Err(picture_error(
                "a sub-picture contains both a percent and a per-mille sign",
            ));
        }
        if picture.iter().filter(|c| **c == format.percent).count() > 1
            || picture.iter().filter(|c| **c == format.per_mille).count() > 1
        {
            return Err(picture_error(
                "a sub-picture contains more than one percent or per-mille sign",
            ));
        }
        if (percent || per_mille) && exponent_sign.is_some() {
            return Err(picture_error(
                "a sub-picture with a percent or per-mille sign must not use scientific notation",
            ));
        }

        let is_digit_position =
            |character: char| character == format.digit || format.is_family_digit(character);
        if !mantissa.iter().copied().any(is_digit_position) {
            return Err(picture_error(
                "the mantissa part of a sub-picture must contain at least one digit position",
            ));
        }

        // A passive character may not be sandwiched between two active characters.
        for (index, character) in picture.iter().copied().enumerate() {
            if is_active_at(index, character) {
                continue;
            }
            let preceded = is_active_anywhere(&picture[..index], 0);
            let followed = is_active_anywhere(&picture[index + 1..], index + 1);
            if preceded && followed {
                return Err(picture_error(format!(
                    "the character `{character}` appears between two active characters"
                )));
            }
        }

        // Grouping separators may not be adjacent to one another or to the decimal separator,
        // and may not end the integer part.
        for window in picture.windows(2) {
            if window[0] == format.grouping_separator && window[1] == format.grouping_separator {
                return Err(picture_error("two adjacent grouping separators"));
            }
            if (window[0] == format.grouping_separator && window[1] == format.decimal_separator)
                || (window[0] == format.decimal_separator && window[1] == format.grouping_separator)
            {
                return Err(picture_error(
                    "a grouping separator is adjacent to the decimal separator",
                ));
            }
        }
        if decimal_positions.is_empty() && integer.last() == Some(&format.grouping_separator) {
            return Err(picture_error(
                "a grouping separator ends the integer part of a sub-picture",
            ));
        }

        // Mandatory digits must precede optional ones in the integer part, and follow them in
        // the fractional part.
        let integer_last_optional = integer.iter().rposition(|c| *c == format.digit);
        let integer_first_mandatory = integer.iter().position(|c| format.is_family_digit(*c));
        if let (Some(optional), Some(mandatory)) = (integer_last_optional, integer_first_mandatory)
        {
            if mandatory < optional {
                return Err(picture_error(
                    "an optional digit follows a mandatory digit in the integer part",
                ));
            }
        }
        let fraction_first_optional = fraction.iter().position(|c| *c == format.digit);
        let fraction_last_mandatory = fraction.iter().rposition(|c| format.is_family_digit(*c));
        if let (Some(optional), Some(mandatory)) =
            (fraction_first_optional, fraction_last_mandatory)
        {
            if optional < mandatory {
                return Err(picture_error(
                    "a mandatory digit follows an optional digit in the fractional part",
                ));
            }
        }

        // The exponent part must be nothing but mandatory digits.
        if exponent_sign.is_some()
            && (exponent.is_empty() || !exponent.iter().all(|c| format.is_family_digit(*c)))
        {
            return Err(picture_error(
                "the exponent part must consist of one or more mandatory digits",
            ));
        }

        let prefix_end = picture
            .iter()
            .copied()
            .enumerate()
            .position(|(index, character)| is_active_at(index, character))
            .unwrap_or(picture.len());
        let suffix_start = picture
            .iter()
            .copied()
            .enumerate()
            .rposition(|(index, character)| is_active_at(index, character))
            .map_or(0, |last| last + 1);
        let prefix: String = picture[..prefix_end].iter().collect();
        let suffix: String = picture[suffix_start..].iter().collect();

        let integer_digit_count = integer
            .iter()
            .copied()
            .filter(|c| is_digit_position(*c))
            .count();
        let mut integer_grouping = Vec::new();
        for (index, character) in integer.iter().copied().enumerate() {
            if character == format.grouping_separator {
                let to_the_right = integer[index + 1..]
                    .iter()
                    .copied()
                    .filter(|c| is_digit_position(*c))
                    .count();
                integer_grouping.push(to_the_right);
            }
        }
        let regular_grouping = regular_grouping_size(&integer_grouping, integer_digit_count);

        let mut fractional_grouping = Vec::new();
        for (index, character) in fraction.iter().copied().enumerate() {
            if character == format.grouping_separator {
                let to_the_left = fraction[..index]
                    .iter()
                    .copied()
                    .filter(|c| is_digit_position(*c))
                    .count();
                fractional_grouping.push(to_the_left);
            }
        }

        let mandatory_integer = integer
            .iter()
            .filter(|c| format.is_family_digit(**c))
            .count();
        let mandatory_fraction = fraction
            .iter()
            .filter(|c| format.is_family_digit(**c))
            .count();
        let optional_fraction = fraction.iter().filter(|c| **c == format.digit).count();
        let minimum_exponent_size = exponent
            .iter()
            .filter(|c| format.is_family_digit(**c))
            .count();

        let mut analyzed = Self {
            prefix,
            suffix,
            integer_grouping,
            regular_grouping,
            fractional_grouping,
            minimum_integer_part_size: mandatory_integer,
            scaling_factor: mandatory_integer,
            minimum_fractional_part_size: mandatory_fraction,
            maximum_fractional_part_size: mandatory_fraction + optional_fraction,
            minimum_exponent_size,
            percent,
            per_mille,
        };

        if analyzed.minimum_integer_part_size == 0 && analyzed.maximum_fractional_part_size == 0 {
            if minimum_exponent_size > 0 {
                analyzed.minimum_fractional_part_size = 1;
                analyzed.maximum_fractional_part_size = 1;
            } else {
                analyzed.minimum_integer_part_size = 1;
            }
        }
        if minimum_exponent_size > 0
            && analyzed.minimum_integer_part_size == 0
            && integer.contains(&format.digit)
        {
            analyzed.minimum_integer_part_size = 1;
        }
        if analyzed.minimum_integer_part_size == 0 && analyzed.minimum_fractional_part_size == 0 {
            analyzed.minimum_fractional_part_size = 1;
        }
        Ok(analyzed)
    }
}

/// The grouping size `G` when the picture's grouping separators are *regular*, per §4.7.4.
fn regular_grouping_size(positions: &[usize], digit_count: usize) -> Option<usize> {
    let smallest = positions.iter().copied().min()?;
    if smallest == 0 {
        return None;
    }
    if !positions.iter().all(|position| position % smallest == 0) {
        return None;
    }
    let mut multiple = smallest;
    while multiple < digit_count {
        if !positions.contains(&multiple) {
            return None;
        }
        multiple += smallest;
    }
    Some(smallest)
}

/// [`fn:format-number`](https://www.w3.org/TR/xpath-functions-31/#func-format-number).
///
/// An absent `$value` is formatted as `NaN`, as the specification requires.
///
/// # Errors
///
/// [`FnErrorCode::Fodf1310`] when the picture string is invalid, [`FnErrorCode::Fodf1280`] when
/// the decimal format itself is inconsistent, and [`FnErrorCode::Xpty0004`] when `$value` is not
/// numeric.
///
/// # Examples
///
/// ```
/// use oxixml_fn::numeric::{format_number, DecimalFormat};
/// use oxixml_infoset::{AtomicType, AtomicValue};
///
/// let format = DecimalFormat::default();
/// let decimal = |lexical| AtomicValue::parse(AtomicType::Decimal, lexical);
///
/// assert_eq!(format_number(Some(&decimal("12345.6")?), "#,###.00", &format)?, "12,345.60");
/// assert_eq!(format_number(Some(&decimal("12345.6")?), "#,##0.00", &format)?, "12,345.60");
/// assert_eq!(format_number(Some(&AtomicValue::integer(2500_i64)), "#,###.00", &format)?, "2,500.00");
/// assert_eq!(format_number(Some(&decimal("0.234")?), "0.0000", &format)?, "0.2340");
/// assert_eq!(format_number(Some(&decimal("0.234")?), "#.00%", &format)?, "23.40%");
/// assert_eq!(format_number(Some(&AtomicValue::double(0.234)), "0.0e0", &format)?, "2.3e-1");
/// assert_eq!(format_number(None, "#.00", &format)?, "NaN");
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
pub fn format_number(
    value: Option<&AtomicValue>,
    picture: &str,
    format: &DecimalFormat,
) -> FnResult<String> {
    format.validate()?;
    let characters: Vec<char> = picture.chars().collect();
    let separators: Vec<usize> = characters
        .iter()
        .enumerate()
        .filter(|(_, character)| **character == format.pattern_separator)
        .map(|(index, _)| index)
        .collect();
    if separators.len() > 1 {
        return Err(picture_error(
            "a picture string may contain at most one pattern separator",
        ));
    }
    let (positive_picture, negative_picture) = match separators.first() {
        Some(index) => (&characters[..*index], Some(&characters[index + 1..])),
        None => (&characters[..], None),
    };

    let positive = SubPicture::analyze(positive_picture, format)?;
    let negative = match negative_picture {
        Some(picture) => SubPicture::analyze(picture, format)?,
        None => {
            let mut negative = positive.clone();
            negative.prefix = format!("{}{}", format.minus_sign, positive.prefix);
            negative
        }
    };

    let Some(value) = value else {
        return Ok(format.nan.clone());
    };
    let number = numeric_lexical(value)?;
    if number.is_nan {
        return Ok(format.nan.clone());
    }
    let sub_picture = if number.negative {
        &negative
    } else {
        &positive
    };
    if number.is_infinite {
        return Ok(format!(
            "{}{}{}",
            sub_picture.prefix, format.infinity, sub_picture.suffix
        ));
    }

    let mut decimal = number.value;
    if sub_picture.percent || sub_picture.per_mille {
        let places = if sub_picture.percent { 2 } else { 3 };
        match scale_in_value_space(value, places) {
            // A floating-point multiplication that overflows gives an infinity, and §4.7.5
            // then writes the format's `infinity` string. W3C bug 28845 settled this, and
            // `cbcl-fn-format-number-034`/`-036` pin it: `format-number(1e308, '0%')`.
            Scaled::Infinite => {
                return Ok(format!(
                    "{}{}{}",
                    sub_picture.prefix, format.infinity, sub_picture.suffix
                ));
            }
            Scaled::Floating(scaled) => decimal = scaled,
            Scaled::Exact => decimal.scale_by(places),
        }
    }

    let exponent = if sub_picture.minimum_exponent_size > 0 && !decimal.is_zero() {
        let scaling = i32::try_from(sub_picture.scaling_factor).unwrap_or(0);
        let exponent = decimal.magnitude() - scaling + 1;
        decimal.scale_by(-exponent);
        Some(exponent)
    } else if sub_picture.minimum_exponent_size > 0 {
        Some(0)
    } else {
        None
    };

    decimal.round_half_to_even(sub_picture.maximum_fractional_part_size);
    let (mut integer_digits, mut fraction_digits) = decimal.split();
    while integer_digits.len() < sub_picture.minimum_integer_part_size {
        integer_digits.insert(0, 0);
    }
    while fraction_digits.len() < sub_picture.minimum_fractional_part_size {
        fraction_digits.push(0);
    }
    while fraction_digits.len() > sub_picture.maximum_fractional_part_size {
        fraction_digits.pop();
    }

    let mut rendered = String::new();
    rendered.push_str(&sub_picture.prefix);
    rendered.push_str(&group_integer(&integer_digits, sub_picture, format));
    if !fraction_digits.is_empty() {
        rendered.push(format.decimal_separator);
        rendered.push_str(&group_fraction(&fraction_digits, sub_picture, format));
    }
    if let Some(exponent) = exponent {
        rendered.push(format.exponent_separator);
        if exponent < 0 {
            rendered.push(format.minus_sign);
        }
        let magnitude = exponent.unsigned_abs().to_string();
        let digits: Vec<u8> = magnitude
            .bytes()
            .map(|byte| byte.saturating_sub(b'0'))
            .collect();
        for _ in digits.len()..sub_picture.minimum_exponent_size {
            rendered.push(format.family_digit(0));
        }
        for digit in digits {
            rendered.push(format.family_digit(digit));
        }
    }
    rendered.push_str(&sub_picture.suffix);
    Ok(rendered)
}

fn group_integer(digits: &[u8], sub_picture: &SubPicture, format: &DecimalFormat) -> String {
    let mut rendered: Vec<char> = Vec::with_capacity(digits.len() * 2);
    for (index, digit) in digits.iter().copied().enumerate() {
        // The separator goes immediately after the digit that has `from_the_right` digits
        // between it and the decimal separator.
        let from_the_right = digits.len() - index - 1;
        rendered.push(format.family_digit(digit));
        if from_the_right > 0 && grouped_at(from_the_right, sub_picture) {
            rendered.push(format.grouping_separator);
        }
    }
    rendered.into_iter().collect()
}

fn grouped_at(position: usize, sub_picture: &SubPicture) -> bool {
    match sub_picture.regular_grouping {
        Some(size) => size > 0 && position % size == 0,
        None => sub_picture.integer_grouping.contains(&position),
    }
}

fn group_fraction(digits: &[u8], sub_picture: &SubPicture, format: &DecimalFormat) -> String {
    let mut rendered: Vec<char> = Vec::with_capacity(digits.len() * 2);
    for (index, digit) in digits.iter().copied().enumerate() {
        if index > 0 && sub_picture.fractional_grouping.contains(&index) {
            rendered.push(format.grouping_separator);
        }
        rendered.push(format.family_digit(digit));
    }
    rendered.into_iter().collect()
}

/// The decomposition of a numeric value the formatter works from.
struct NumericLexical {
    negative: bool,
    is_nan: bool,
    is_infinite: bool,
    value: BigDecimal,
}

/// What the percent / per-mille multiplication of §4.7.5 did to the value.
enum Scaled {
    /// The type has exact arithmetic, so a decimal shift is the multiplication.
    Exact,
    /// The multiplication overflowed the floating-point type to an infinity.
    Infinite,
    /// The floating-point product, as its own canonical decimal expansion.
    Floating(BigDecimal),
}

/// Multiplies by `10^places` in the *value space of `value`'s own type*.
///
/// [F&O 3.1 §4.7.5](https://www.w3.org/TR/xpath-functions-31/#formatting-the-number) rule 3
/// multiplies by 100 or 1000 before anything else, and the multiplication is the one the input's
/// type defines — not an exact decimal shift. For `xs:double` and `xs:float` that means one IEEE
/// rounding (`cbcl-fn-format-number-035` compares `format-number($d, '0.0…‰')` against
/// `format-number($d * 1000, '0.0…')` digit for digit) and, at the top of the range, an
/// overflow to infinity (`cbcl-fn-format-number-034`, W3C bug 28845).
fn scale_in_value_space(value: &AtomicValue, places: i32) -> Scaled {
    let factor = if places == 2 { 100.0_f64 } else { 1000.0_f64 };
    let Some(magnitude) = value.as_f64() else {
        return Scaled::Exact;
    };
    let scaled = match value.atomic_type().primitive() {
        AtomicType::Double => AtomicValue::double(magnitude * factor),
        #[allow(clippy::cast_possible_truncation)]
        AtomicType::Float => AtomicValue::float((magnitude as f32) * (factor as f32)),
        _ => return Scaled::Exact,
    };
    let lexical = scaled.string_value();
    if lexical.ends_with("INF") {
        return Scaled::Infinite;
    }
    match BigDecimal::parse(&lexical) {
        Some(mut decimal) => {
            decimal.negative = lexical.starts_with('-');
            Scaled::Floating(decimal)
        }
        None => Scaled::Exact,
    }
}

/// Whether `picture` uses the scientific-notation sub-picture.
///
/// The exponent part of a picture string was added in
/// [F&O 3.1 §4.7.4](https://www.w3.org/TR/xpath-functions-31/#analyzing-picture-string); a 3.0
/// processor has to reject it with `FODF1310` (`numberformat128`), and only the caller knows
/// which version it is compiling.
///
/// # Errors
///
/// [`FnErrorCode::Fodf1310`] when the picture is not a valid one at all.
pub fn picture_has_exponent(picture: &str, format: &DecimalFormat) -> FnResult<bool> {
    for sub_picture in picture.split(format.pattern_separator) {
        let characters: Vec<char> = sub_picture.chars().collect();
        if SubPicture::analyze(&characters, format)?.minimum_exponent_size > 0 {
            return Ok(true);
        }
    }
    Ok(false)
}

fn numeric_lexical(value: &AtomicValue) -> FnResult<NumericLexical> {
    let value = if value.atomic_type() == AtomicType::UntypedAtomic {
        AtomicValue::parse(AtomicType::Double, value.as_str().unwrap_or_default())?
    } else {
        value.clone()
    };
    if !value.is_numeric() {
        return Err(FnError::new(
            FnErrorCode::Xpty0004,
            format!(
                "fn:format-number expects a numeric value, not {}",
                value.atomic_type()
            ),
        ));
    }
    if value.is_nan() {
        return Ok(NumericLexical {
            negative: false,
            is_nan: true,
            is_infinite: false,
            value: BigDecimal::zero(false),
        });
    }
    let lexical = value.string_value();
    let negative = lexical.starts_with('-');
    if lexical.ends_with("INF") {
        return Ok(NumericLexical {
            negative,
            is_nan: false,
            is_infinite: true,
            value: BigDecimal::zero(negative),
        });
    }
    let mut decimal = BigDecimal::parse(&lexical).ok_or_else(|| {
        FnError::new(
            FnErrorCode::Xpty0004,
            format!("`{lexical}` is not a numeric lexical form"),
        )
    })?;
    // xs:double and xs:float treat negative zero as negative; xs:decimal and xs:integer do not.
    decimal.negative = negative;
    Ok(NumericLexical {
        negative,
        is_nan: false,
        is_infinite: false,
        value: decimal,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn decimal(lexical: &str) -> AtomicValue {
        AtomicValue::parse(AtomicType::Decimal, lexical).expect("a valid xs:decimal")
    }

    fn formatted(value: &AtomicValue, picture: &str) -> String {
        format_number(Some(value), picture, &DecimalFormat::default())
            .expect("a valid picture string")
    }

    /// §4.7.5 multiplies by 100 or 1000 in the *value space of the input's type*, so a double
    /// may overflow to infinity (W3C bug 28845) while an exact type stays exact.
    #[test]
    fn percent_and_per_mille_scale_in_the_input_type() {
        assert_eq!(formatted(&AtomicValue::double(1e308), "0%"), "Infinity%");
        assert_eq!(formatted(&AtomicValue::double(-1e308), "0%"), "-Infinity%");
        assert_eq!(formatted(&decimal("0.5"), "0%"), "50%");
        assert_eq!(formatted(&decimal("0.5"), "0\u{2030}"), "500\u{2030}");
        // A double that does not overflow carries exactly one IEEE rounding, which is what
        // `format-number($d * 1000, …)` would also produce.
        assert_eq!(
            formatted(
                &AtomicValue::double(1.102_030_405_060_708),
                "0.00000000000000000\u{2030}"
            ),
            formatted(
                &AtomicValue::double(1.102_030_405_060_708 * 1000.0),
                "0.00000000000000000",
            ) + "\u{2030}",
        );
    }

    /// The exponent part of a picture string arrived with F&O 3.1, so a 3.0 caller has to be
    /// able to detect and reject it.
    #[test]
    fn an_exponent_in_the_picture_is_detectable() -> FnResult<()> {
        let format = DecimalFormat::default();
        assert!(picture_has_exponent("9.9999e999", &format)?);
        assert!(picture_has_exponent("#,##0.00;9.9e9", &format)?);
        assert!(!picture_has_exponent("#,##0.00", &format)?);
        Ok(())
    }

    #[test]
    fn examples_from_the_specification() {
        assert_eq!(formatted(&decimal("12345.6"), "#,###.00"), "12,345.60");
        assert_eq!(formatted(&decimal("12345.6"), "#,##0.00"), "12,345.60");
        assert_eq!(
            formatted(&AtomicValue::integer(2500_i64), "#,###.00"),
            "2,500.00"
        );
        assert_eq!(formatted(&decimal("0.234"), "0.0000"), "0.2340");
        assert_eq!(formatted(&decimal("0.234"), "#.00%"), "23.40%");
        assert_eq!(formatted(&decimal("0.234"), "#.000%"), "23.400%");
    }

    #[test]
    fn negative_sub_pictures_are_selected_by_sign() {
        assert_eq!(formatted(&AtomicValue::integer(-3_i64), "0"), "-3");
        assert_eq!(formatted(&AtomicValue::integer(-3_i64), "0;(0)"), "(3)");
        assert_eq!(formatted(&AtomicValue::integer(3_i64), "0;(0)"), "3");
    }

    #[test]
    fn minimum_sizes_pad_with_zero_digits() {
        assert_eq!(formatted(&AtomicValue::integer(3_i64), "0000"), "0003");
        assert_eq!(formatted(&decimal("1.5"), "0.000"), "1.500");
        assert_eq!(formatted(&decimal("0.23"), "#"), "0");
        assert_eq!(formatted(&decimal("1.999"), "#.##"), "2.0");
    }

    #[test]
    fn rounding_is_half_to_even() {
        assert_eq!(formatted(&decimal("0.125"), "0.00"), "0.12");
        assert_eq!(formatted(&decimal("0.135"), "0.00"), "0.14");
        assert_eq!(formatted(&decimal("2.5"), "0"), "2");
        assert_eq!(formatted(&decimal("3.5"), "0"), "4");
    }

    #[test]
    fn scientific_notation() {
        assert_eq!(formatted(&AtomicValue::double(0.234), "0.0e0"), "2.3e-1");
        assert_eq!(
            formatted(&AtomicValue::double(1234.5678), "00.000e0"),
            "12.346e2"
        );
        assert_eq!(formatted(&AtomicValue::double(0.1), "#.9e9"), "0.1e0");
        assert_eq!(formatted(&AtomicValue::double(0.123), "#.e9"), "0.1e0");
    }

    #[test]
    fn special_values() {
        assert_eq!(formatted(&AtomicValue::double(f64::NAN), "#.00"), "NaN");
        assert_eq!(
            formatted(&AtomicValue::double(f64::INFINITY), "#.00"),
            "Infinity"
        );
        assert_eq!(
            formatted(&AtomicValue::double(f64::NEG_INFINITY), "#.00"),
            "-Infinity"
        );
        assert_eq!(
            format_number(None, "#.00", &DecimalFormat::default()).expect("valid"),
            "NaN"
        );
    }

    #[test]
    fn custom_decimal_formats_change_the_characters() {
        let european = DecimalFormat::default()
            .with_decimal_separator(',')
            .with_grouping_separator('.');
        assert_eq!(
            format_number(Some(&decimal("1234567.89")), "#.##0,00", &european).expect("valid"),
            "1.234.567,89"
        );
    }

    #[test]
    fn irregular_grouping_positions_are_honoured() {
        assert_eq!(
            formatted(&AtomicValue::integer(1_234_567_i64), "#,##,##0"),
            "12,34,567"
        );
    }

    #[test]
    fn invalid_pictures_raise_fodf1310() {
        let format = DecimalFormat::default();
        for picture in [
            "0;0;0",
            "0.0.0",
            "#%\u{2030}",
            "abc",
            "0,,0",
            "0,.0",
            "#0#",
            "0.#0",
            "0e0e0",
        ] {
            let error = format_number(Some(&AtomicValue::integer(1_i64)), picture, &format)
                .expect_err(picture);
            assert!(error.is(FnErrorCode::Fodf1310), "{picture}: {error}");
        }
    }

    #[test]
    fn unbounded_precision_is_available() {
        let third = AtomicValue::parse(AtomicType::Decimal, "0.333333333333333333")
            .expect("a valid xs:decimal");
        assert_eq!(
            formatted(&third, "0.000000000000000000"),
            "0.333333333333333333"
        );
    }

    #[test]
    fn colliding_decimal_format_properties_raise_fodf1280() {
        let broken = DecimalFormat::default().with_digit('.');
        let error = format_number(Some(&AtomicValue::integer(1_i64)), "0", &broken)
            .expect_err("digit collides with decimal-separator");
        assert!(error.is(FnErrorCode::Fodf1280));
    }

    /// §4.7.3 classifies a character as *active* by the decimal format in force, not by what it
    /// looks like: with `grouping-separator="'"` an apostrophe groups digits, and the picture
    /// that uses it is valid however unusual the character is. (A picture that pairs the same
    /// apostrophe with the *default* format is the passive character between two active ones
    /// that `FODF1310` forbids, which is what makes the format the formatter is handed the
    /// thing that decides.)
    #[test]
    fn an_unusual_grouping_separator_is_an_active_character() {
        let swiss = DecimalFormat::default().with_grouping_separator('\'');
        assert_eq!(
            format_number(Some(&decimal("123456.789")), "#'###'###.###", &swiss)
                .expect("a valid picture"),
            "123'456.789"
        );
        assert_eq!(
            format_number(Some(&AtomicValue::double(1e9)), "#'###'###'##0.00", &swiss)
                .expect("a valid picture"),
            "1'000'000'000.00"
        );
        let error = format_number(
            Some(&decimal("123456.789")),
            "#'###'###.###",
            &DecimalFormat::default(),
        )
        .expect_err("an apostrophe between two active characters");
        assert!(error.is(FnErrorCode::Fodf1310), "{error}");
    }
}
