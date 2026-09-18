//! The string construction and inspection functions of
//! [F&O 3.1 §5.2–§5.4](https://www.w3.org/TR/xpath-functions-31/#string-functions): assembly
//! (`fn:concat`, `fn:string-join`), decomposition (`fn:substring`, `fn:string-length`),
//! normalization (`fn:normalize-space`, `fn:normalize-unicode`), case conversion and
//! `fn:translate`, plus the code point conversions of §5.2.

use crate::error::{FnError, FnErrorCode, FnResult};
use crate::numeric::round::round_f64_half_up;
use oxixml_infoset::{Item, Sequence};
use oxixml_unicode::case::{full_lowercase, full_uppercase};
use oxixml_unicode::norm::{is_nfc, nfc, nfd, nfkc, nfkd};
use oxixml_unicode::xml::XmlVersion;

/// [`fn:codepoints-to-string`](https://www.w3.org/TR/xpath-functions-31/#func-codepoints-to-string).
///
/// # Errors
///
/// [`FnErrorCode::Foch0001`] when a code point is not a valid XML character.
///
/// # Examples
///
/// ```
/// use oxixml_fn::strings::codepoints_to_string;
///
/// assert_eq!(codepoints_to_string(&[84, 104, 232, 114, 232, 115, 101])?, "Thèrèse");
/// assert_eq!(codepoints_to_string(&[])?, "");
/// assert!(codepoints_to_string(&[0]).is_err());
/// # Ok::<(), oxixml_fn::FnError>(())
/// ```
pub fn codepoints_to_string(codepoints: &[i64]) -> FnResult<String> {
    codepoints_to_string_in(codepoints, XmlVersion::V10)
}

/// [`codepoints_to_string`] against the `Char` production of a given version of XML.
///
/// [XDM 3.1 §2](https://www.w3.org/TR/xpath-datamodel-31/#terminology) defines a *character* as
/// an instance of the `Char` production of whichever version of XML the implementation supports,
/// so which code points this admits is a property of the processor rather than of the call:
/// `8` is `FOCH0001` under XML 1.0 and a backspace under XML 1.1, and `0` is `FOCH0001` under
/// both. A host tells the library which version it implements with
/// [`FnContext::with_xml_version`](crate::FnContext::with_xml_version), which is what the
/// registered `fn:codepoints-to-string` reads.
///
/// # Errors
///
/// [`FnErrorCode::Foch0001`] when a code point is not a valid XML character under `version`.
///
/// # Examples
///
/// ```
/// use oxixml_fn::strings::codepoints_to_string_in;
/// use oxixml_unicode::xml::XmlVersion;
///
/// assert!(codepoints_to_string_in(&[8], XmlVersion::V10).is_err());
/// assert_eq!(codepoints_to_string_in(&[8], XmlVersion::V11)?, "\u{8}");
/// // NUL is no character in either version.
/// assert!(codepoints_to_string_in(&[0], XmlVersion::V11).is_err());
/// # Ok::<(), oxixml_fn::FnError>(())
/// ```
pub fn codepoints_to_string_in(codepoints: &[i64], version: XmlVersion) -> FnResult<String> {
    let mut out = String::with_capacity(codepoints.len());
    for codepoint in codepoints.iter().copied() {
        let character = u32::try_from(codepoint)
            .ok()
            .and_then(char::from_u32)
            .filter(|character| oxixml_unicode::xml::is_char(*character, version))
            .ok_or_else(|| {
                FnError::new(
                    FnErrorCode::Foch0001,
                    format!("{codepoint} is not a valid XML character"),
                )
            })?;
        out.push(character);
    }
    Ok(out)
}

/// [`fn:string-to-codepoints`](https://www.w3.org/TR/xpath-functions-31/#func-string-to-codepoints).
///
/// # Examples
///
/// ```
/// use oxixml_fn::strings::string_to_codepoints;
///
/// assert_eq!(string_to_codepoints("Thérèse"), vec![84, 104, 233, 114, 232, 115, 101]);
/// assert!(string_to_codepoints("").is_empty());
/// ```
#[must_use]
pub fn string_to_codepoints(value: &str) -> Vec<i64> {
    value.chars().map(|c| i64::from(u32::from(c))).collect()
}

/// [`fn:concat`](https://www.w3.org/TR/xpath-functions-31/#func-concat): concatenates the string
/// values of its arguments, each of which must atomize to at most one value.
///
/// # Errors
///
/// [`FnErrorCode::Xpty0004`] when an argument atomizes to more than one value.
///
/// # Examples
///
/// ```
/// use oxixml_fn::strings::concat;
/// use oxixml_infoset::{AtomicValue, Sequence};
///
/// let arguments = [
///     Sequence::from(AtomicValue::string("un")),
///     Sequence::from(AtomicValue::string("deux")),
///     Sequence::empty(),
///     Sequence::from(AtomicValue::integer(3_i64)),
/// ];
/// assert_eq!(concat(&arguments)?, "undeux3");
/// # Ok::<(), oxixml_fn::FnError>(())
/// ```
pub fn concat(arguments: &[Sequence]) -> FnResult<String> {
    let mut out = String::new();
    for argument in arguments {
        let atomized = argument.atomize()?;
        match atomized.as_slice() {
            [] => {}
            [item] => out.push_str(&item.string_value()?),
            _ => {
                return Err(FnError::new(
                    FnErrorCode::Xpty0004,
                    "each argument of fn:concat must be at most one atomic value",
                ));
            }
        }
    }
    Ok(out)
}

/// [`fn:string-join`](https://www.w3.org/TR/xpath-functions-31/#func-string-join).
///
/// # Errors
///
/// Whatever atomization raises — [`FnErrorCode::Foty0013`] for a map, for instance.
///
/// # Examples
///
/// ```
/// use oxixml_fn::strings::string_join;
/// use oxixml_infoset::{AtomicValue, Sequence};
///
/// let values = Sequence::from_iter([
///     AtomicValue::string("Now"),
///     AtomicValue::string("is"),
///     AtomicValue::string("the"),
///     AtomicValue::string("time"),
/// ]);
/// assert_eq!(string_join(&values, " ")?, "Now is the time");
/// assert_eq!(string_join(&Sequence::empty(), "sep")?, "");
/// # Ok::<(), oxixml_fn::FnError>(())
/// ```
pub fn string_join(sequence: &Sequence, separator: &str) -> FnResult<String> {
    let atomized = sequence.atomize()?;
    let mut out = String::new();
    for (index, item) in atomized.iter().enumerate() {
        if index > 0 {
            out.push_str(separator);
        }
        out.push_str(&item.string_value()?);
    }
    Ok(out)
}

/// [`fn:substring`](https://www.w3.org/TR/xpath-functions-31/#func-substring), the three-argument
/// form; pass `None` for `length` to get the two-argument form.
///
/// Positions are one-based, and both numeric arguments are rounded with `fn:round` before the
/// range `[start, start + length)` is applied, so fractional and out-of-range positions behave
/// exactly as the specification's examples require.
///
/// # Examples
///
/// ```
/// use oxixml_fn::strings::substring;
///
/// assert_eq!(substring("motor car", 6.0, None), " car");
/// assert_eq!(substring("metadata", 4.0, Some(3.0)), "ada");
/// assert_eq!(substring("12345", 1.5, Some(2.6)), "234");
/// assert_eq!(substring("12345", 0.0, Some(3.0)), "12");
/// assert_eq!(substring("12345", 5.0, Some(-3.0)), "");
/// assert_eq!(substring("12345", -3.0, Some(5.0)), "1");
/// assert_eq!(substring("12345", f64::NAN, Some(3.0)), "");
/// assert_eq!(substring("12345", 1.0, Some(f64::INFINITY)), "12345");
/// assert_eq!(substring("12345", f64::NEG_INFINITY, Some(f64::INFINITY)), "");
/// ```
#[must_use]
pub fn substring(value: &str, start: f64, length: Option<f64>) -> String {
    if start.is_nan() {
        return String::new();
    }
    let first = round_f64_half_up(start);
    let limit = match length {
        None => f64::INFINITY,
        Some(length) if length.is_nan() => return String::new(),
        Some(length) => first + round_f64_half_up(length),
    };
    if limit.is_nan() {
        return String::new();
    }
    value
        .chars()
        .enumerate()
        .filter_map(|(index, character)| {
            let position = index_as_f64(index + 1);
            (position >= first && position < limit).then_some(character)
        })
        .collect()
}

fn index_as_f64(index: usize) -> f64 {
    u32::try_from(index).map_or(f64::INFINITY, f64::from)
}

/// [`fn:string-length`](https://www.w3.org/TR/xpath-functions-31/#func-string-length), counted in
/// characters (code points), not in UTF-8 bytes.
///
/// # Examples
///
/// ```
/// use oxixml_fn::strings::string_length;
///
/// assert_eq!(string_length("Harp not on that string, madam; that is past."), 45);
/// assert_eq!(string_length(""), 0);
/// assert_eq!(string_length("é"), 1);
/// ```
#[must_use]
pub fn string_length(value: &str) -> usize {
    value.chars().count()
}

/// [`fn:normalize-space`](https://www.w3.org/TR/xpath-functions-31/#func-normalize-space): trims
/// and collapses runs of the four XML whitespace characters.
///
/// # Examples
///
/// ```
/// use oxixml_fn::strings::normalize_space;
///
/// assert_eq!(normalize_space(" The    wealthy curled darlings\n of our nation. "),
///            "The wealthy curled darlings of our nation.");
/// assert_eq!(normalize_space("\t\n"), "");
/// ```
#[must_use]
pub fn normalize_space(value: &str) -> String {
    oxixml_text::ws::collapse(value).into_owned()
}

/// The Unicode normalization forms [`normalize_unicode`] accepts.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum NormalizationForm {
    /// `NFC`, canonical decomposition followed by canonical composition.
    Nfc,
    /// `NFD`, canonical decomposition.
    Nfd,
    /// `NFKC`, compatibility decomposition followed by canonical composition.
    Nfkc,
    /// `NFKD`, compatibility decomposition.
    Nfkd,
    /// The zero-length form name, which requests no normalization at all.
    None,
}

impl NormalizationForm {
    /// Parses a form name, which is matched after stripping whitespace and upper-casing, as
    /// F&O §5.4.7 requires.
    ///
    /// # Errors
    ///
    /// [`FnErrorCode::Foch0003`] for a form this implementation does not support, including
    /// `FULLY-NORMALIZED`.
    pub fn parse(name: &str) -> FnResult<Self> {
        let normalized: String = name.trim().to_uppercase();
        match normalized.as_str() {
            "" => Ok(Self::None),
            "NFC" => Ok(Self::Nfc),
            "NFD" => Ok(Self::Nfd),
            "NFKC" => Ok(Self::Nfkc),
            "NFKD" => Ok(Self::Nfkd),
            other => Err(FnError::new(
                FnErrorCode::Foch0003,
                format!("the normalization form `{other}` is not supported"),
            )),
        }
    }
}

/// [`fn:normalize-unicode`](https://www.w3.org/TR/xpath-functions-31/#func-normalize-unicode).
///
/// # Examples
///
/// ```
/// use oxixml_fn::strings::{normalize_unicode, NormalizationForm};
///
/// // U+0065 U+0301 (e + combining acute) composes to U+00E9.
/// assert_eq!(normalize_unicode("e\u{301}", NormalizationForm::Nfc), "\u{e9}");
/// assert_eq!(normalize_unicode("\u{e9}", NormalizationForm::Nfd), "e\u{301}");
/// // The zero-length form name leaves the string alone.
/// assert_eq!(normalize_unicode("e\u{301}", NormalizationForm::None), "e\u{301}");
/// ```
#[must_use]
pub fn normalize_unicode(value: &str, form: NormalizationForm) -> String {
    match form {
        NormalizationForm::None => value.to_owned(),
        // UAX #15's NFC quick check never allocates, and a `true` answer is *exact*: the string
        // already is its own NFC. Almost every string a query normalizes is already composed,
        // and taking it through decomposition, canonical ordering and recomposition — two
        // `Vec<char>` allocations and two table walks per character — only to get the same
        // characters back is the one avoidable cost in this function.
        NormalizationForm::Nfc if is_nfc(value) => value.to_owned(),
        NormalizationForm::Nfc => nfc(value),
        NormalizationForm::Nfd => nfd(value),
        NormalizationForm::Nfkc => nfkc(value),
        NormalizationForm::Nfkd => nfkd(value),
    }
}

/// [`fn:upper-case`](https://www.w3.org/TR/xpath-functions-31/#func-upper-case).
///
/// The specification defines the result as the Unicode default (language-agnostic, Root locale)
/// *full* uppercase mapping: the unconditional multi-character mappings of `SpecialCasing.txt`
/// apply, so `ß` becomes `SS` and the `ﬁ` ligature expands to `FI`. Because it is the
/// language-agnostic mapping, no locale tailoring (Turkish dotted/dotless `I`, and so on) is
/// applied.
///
/// # Examples
///
/// ```
/// use oxixml_fn::strings::upper_case;
///
/// assert_eq!(upper_case("abCd0"), "ABCD0");
/// assert_eq!(upper_case("ABCd0"), "ABCD0");
/// assert_eq!(upper_case("straße"), "STRASSE");
/// assert_eq!(upper_case("\u{FB01}le"), "FILE");
/// ```
#[must_use]
pub fn upper_case(value: &str) -> String {
    full_uppercase(value)
}

/// [`fn:lower-case`](https://www.w3.org/TR/xpath-functions-31/#func-lower-case).
///
/// As with [`upper_case`], the Unicode default (language-agnostic, Root locale) *full* lowercase
/// mapping is used, including the `Final_Sigma` context test that maps a word-final Greek capital
/// sigma to `ς` rather than `σ`.
///
/// # Examples
///
/// ```
/// use oxixml_fn::strings::lower_case;
///
/// assert_eq!(lower_case("ABc!D"), "abc!d");
/// assert_eq!(lower_case("ΟΔΥΣΣΕΥΣ"), "οδυσσευς"); // final sigma -> ς
/// ```
#[must_use]
pub fn lower_case(value: &str) -> String {
    full_lowercase(value)
}

/// [`fn:translate`](https://www.w3.org/TR/xpath-functions-31/#func-translate): replaces each
/// character of `value` that occurs in `map` with the character at the same position in
/// `translation`, deleting it when `translation` is shorter.
///
/// # Examples
///
/// ```
/// use oxixml_fn::strings::translate;
///
/// assert_eq!(translate("bar", "abc", "ABC"), "BAr");
/// assert_eq!(translate("--aaa--", "abc-", "ABC"), "AAA");
/// assert_eq!(translate("abcdabc", "abc", "AB"), "ABdAB");
/// ```
#[must_use]
pub fn translate(value: &str, map: &str, translation: &str) -> String {
    let map: Vec<char> = map.chars().collect();
    let translation: Vec<char> = translation.chars().collect();
    let mut out = String::with_capacity(value.len());
    for character in value.chars() {
        match map.iter().position(|candidate| *candidate == character) {
            None => out.push(character),
            Some(position) => {
                if let Some(replacement) = translation.get(position) {
                    out.push(*replacement);
                }
            }
        }
    }
    out
}

/// The string value of a sequence that must be empty or a singleton, which is how every
/// `xs:string?` argument of the string functions is supplied.
///
/// # Errors
///
/// [`FnErrorCode::Xpty0004`] for a longer sequence, [`FnErrorCode::Foty0014`] for a function
/// item.
pub fn optional_string(sequence: &Sequence) -> FnResult<String> {
    match sequence.as_slice() {
        [] => Ok(String::new()),
        [Item::Atomic(value)] => Ok(value.string_value()),
        [item] => Ok(item.string_value()?),
        _ => Err(FnError::new(
            FnErrorCode::Xpty0004,
            "an xs:string? argument must be at most one item",
        )),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use oxixml_infoset::AtomicValue;

    #[test]
    fn codepoint_round_trip() -> FnResult<()> {
        assert_eq!(
            codepoints_to_string(&[84, 104, 232, 114, 232, 115, 101])?,
            "Thèrèse"
        );
        assert_eq!(
            string_to_codepoints("Thérèse"),
            vec![84, 104, 233, 114, 232, 115, 101]
        );
        assert_eq!(codepoints_to_string(&[])?, "");
        assert!(string_to_codepoints("").is_empty());
        let error = codepoints_to_string(&[0]).expect_err("NUL is not an XML character");
        assert!(error.is(FnErrorCode::Foch0001));
        assert!(codepoints_to_string(&[-1]).is_err());
        Ok(())
    }

    #[test]
    fn concat_examples_from_the_specification() -> FnResult<()> {
        let arguments = [
            Sequence::from(AtomicValue::string("un")),
            Sequence::from(AtomicValue::string("deux")),
            Sequence::from(AtomicValue::string("trois")),
        ];
        assert_eq!(concat(&arguments)?, "undeuxtrois");
        assert_eq!(
            concat(&[
                Sequence::from(AtomicValue::string("Thy ")),
                Sequence::empty(),
                Sequence::from(AtomicValue::string("old ")),
                Sequence::from(AtomicValue::string("groans")),
            ])?,
            "Thy old groans"
        );
        Ok(())
    }

    #[test]
    fn string_join_examples_from_the_specification() -> FnResult<()> {
        let values = Sequence::from_iter([
            AtomicValue::string("Now"),
            AtomicValue::string("is"),
            AtomicValue::string("the"),
            AtomicValue::string("time"),
        ]);
        assert_eq!(string_join(&values, " ")?, "Now is the time");
        assert_eq!(string_join(&values, "")?, "Nowisthetime");
        assert_eq!(string_join(&Sequence::empty(), "separator")?, "");
        Ok(())
    }

    #[test]
    fn substring_examples_from_the_specification() {
        assert_eq!(substring("motor car", 6.0, None), " car");
        assert_eq!(substring("metadata", 4.0, Some(3.0)), "ada");
        assert_eq!(substring("12345", 1.5, Some(2.6)), "234");
        assert_eq!(substring("12345", 0.0, Some(3.0)), "12");
        assert_eq!(substring("12345", 5.0, Some(-3.0)), "");
        assert_eq!(substring("12345", -3.0, Some(5.0)), "1");
        assert_eq!(substring("12345", f64::NAN, Some(3.0)), "");
        assert_eq!(substring("12345", 1.0, Some(f64::NAN)), "");
        assert_eq!(substring("12345", -42.0, Some(f64::INFINITY)), "12345");
        assert_eq!(
            substring("12345", f64::NEG_INFINITY, Some(f64::INFINITY)),
            ""
        );
    }

    #[test]
    fn string_length_and_normalize_space() {
        assert_eq!(
            string_length("Harp not on that string, madam; that is past."),
            45
        );
        assert_eq!(
            normalize_space(" The    wealthy curled darlings\n of our nation. "),
            "The wealthy curled darlings of our nation."
        );
        assert_eq!(normalize_space(""), "");
    }

    #[test]
    fn normalization_forms_are_parsed_case_insensitively() -> FnResult<()> {
        assert_eq!(NormalizationForm::parse("nfc")?, NormalizationForm::Nfc);
        assert_eq!(
            NormalizationForm::parse("  NFKD ")?,
            NormalizationForm::Nfkd
        );
        assert_eq!(NormalizationForm::parse("")?, NormalizationForm::None);
        let error = NormalizationForm::parse("FULLY-NORMALIZED").expect_err("unsupported");
        assert!(error.is(FnErrorCode::Foch0003));
        Ok(())
    }

    /// Which code points are characters is the processor's XML version. QT3 `fn/matches`-era
    /// `K2-Serialization-7`/`-7a` write the two answers as a pair over the same query, so a
    /// processor answers for exactly one version and the default is XML 1.0.
    #[test]
    fn codepoints_to_string_reads_the_declared_xml_version() -> FnResult<()> {
        for control in [1_i64, 7, 8, 11, 12, 14, 31] {
            assert!(
                codepoints_to_string(&[control]).is_err(),
                "{control} is no XML 1.0 character"
            );
            assert_eq!(
                codepoints_to_string_in(&[control], XmlVersion::V11)?
                    .chars()
                    .count(),
                1,
                "{control} is an XML 1.1 character"
            );
        }
        // Neither version has a character for NUL, a surrogate, or the two non-characters.
        for never in [0_i64, 0xd800, 0xfffe, 0xffff, -1, 0x11_0000] {
            assert!(codepoints_to_string_in(&[never], XmlVersion::V10).is_err());
            assert!(codepoints_to_string_in(&[never], XmlVersion::V11).is_err());
        }
        // Tab, line feed and carriage return are characters in both.
        assert_eq!(codepoints_to_string(&[9, 10, 13])?, "\t\n\r");
        Ok(())
    }

    #[test]
    fn normalize_unicode_composes_and_decomposes() {
        assert_eq!(
            normalize_unicode("e\u{301}", NormalizationForm::Nfc),
            "\u{e9}"
        );
        assert_eq!(
            normalize_unicode("\u{e9}", NormalizationForm::Nfd),
            "e\u{301}"
        );
        assert_eq!(normalize_unicode("\u{fb01}", NormalizationForm::Nfkc), "fi");
        assert_eq!(normalize_unicode("x", NormalizationForm::None), "x");
    }

    #[test]
    fn case_conversion_examples_from_the_specification() {
        assert_eq!(upper_case("abCd0"), "ABCD0");
        assert_eq!(lower_case("ABc!D"), "abc!d");
        assert_eq!(upper_case("\u{e9}"), "\u{c9}");
    }

    #[test]
    fn case_conversion_applies_full_unicode_mappings() {
        // ß uppercases to the two-character SS sequence (SpecialCasing.txt).
        assert_eq!(upper_case("straße"), "STRASSE");
        assert_eq!(upper_case("\u{df}"), "SS");
        // Ligatures expand: ﬁ -> FI, ﬄ -> FFL.
        assert_eq!(upper_case("\u{fb01}le"), "FILE");
        assert_eq!(upper_case("a\u{fb04}e"), "AFFLE");
        // Final-sigma: a word-final capital sigma lowercases to ς, an interior one to σ.
        assert_eq!(lower_case("ΟΔΥΣΣΕΥΣ"), "οδυσσευς");
        assert_eq!(lower_case("ΣΟΦΙΑ"), "σοφια");
    }

    #[test]
    fn translate_examples_from_the_specification() {
        assert_eq!(translate("bar", "abc", "ABC"), "BAr");
        assert_eq!(translate("--aaa--", "abc-", "ABC"), "AAA");
        assert_eq!(translate("abcdabc", "abc", "AB"), "ABdAB");
    }

    #[test]
    fn optional_string_handles_the_empty_sequence() -> FnResult<()> {
        assert_eq!(optional_string(&Sequence::empty())?, "");
        assert_eq!(
            optional_string(&Sequence::from(AtomicValue::integer(5_i64)))?,
            "5"
        );
        Ok(())
    }
}
