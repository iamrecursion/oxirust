//! The root weight table a UCA collation reads, and the implicit weights it computes for
//! everything the table does not name.
//!
//! [UTS #10](https://www.unicode.org/reports/tr10/) maps text to *collation elements*, each a
//! triple of weights — a primary for the base character, a secondary for the accents, a tertiary
//! for the case — and compares two strings one whole level at a time. The table that supplies
//! those weights is the DUCET, some thirty thousand entries of it. This module carries the part an
//! XML processor needs as Rust constants and derives the rest, so that the crate stays
//! self-contained.
//!
//! # The repertoire that has explicit weights
//!
//! * The **variable** characters — whitespace, then punctuation, then the general symbols, then
//!   the currency symbols. That is both the root order and the order the four groups of the
//!   `maxVariable` parameter nest in, and it puts every one of them below the digits and the
//!   letters, which code point order does not.
//! * The **decimal digits** of every script, ordered by value rather than by code point, so that
//!   `٣` weighs as `3` does.
//! * The **basic Latin letters** `a`–`z`. Every accented Latin letter reduces to one of them once
//!   the canonical decomposition has moved its accents into combining marks of their own, so the
//!   table needs no entry for `é`, `ü` or `ñ`.
//! * The few Latin characters that weigh as *several* letters and that no canonical decomposition
//!   produces — `ß` as `ss`, `ĳ` as `ij`, the `ﬁ` ligatures as their parts. Each half carries a
//!   tertiary *variant* weight, so the expansion is an equality at the first two levels and a
//!   difference at the third: `Straße` and `Strasse` are the same word at primary strength and two
//!   spellings of it at tertiary strength.
//!
//! # Everything else
//!
//! Every other code point takes an **implicit** weight, which is [UTS
//! #10 §10](https://www.unicode.org/reports/tr10/#Implicit_Weights)'s answer for a character the
//! table does not name: a weight derived from the code point itself, placed above every explicit
//! weight. The consequence is that the scripts this table does not enumerate — Greek, Cyrillic,
//! Hebrew, Arabic, the Indic scripts, the kana and the Han ideographs — sort after the Latin
//! letters, and among themselves in Unicode's own block order, which is the order the root table
//! puts them in. Within a script the ordering is by code point, which is what UTS #10 §10.1
//! prescribes inside each of its implicit ranges.
//!
//! What that costs, compared with the whole DUCET, is the *relative* placement of two unenumerated
//! scripts whose blocks and root weights disagree, and the accent orderings a script gives its own
//! marks. Both are documented gaps rather than silent approximations, and neither affects the
//! Latin, digit and punctuation text an XML processor compares.

use oxixml_unicode::case::{simple_fold, simple_lowercase};
use oxixml_unicode::gc::{GeneralCategory, general_category};

/// How far a primary weight shifts its group, leaving the low forty bits for the position within
/// it — room for every Unicode code point and then some.
const GROUP_SHIFT: u32 = 40;

/// Whitespace: the first thing the root table weighs.
pub(super) const GROUP_WHITESPACE: u64 = 1;
/// Punctuation, which the root table places above whitespace.
pub(super) const GROUP_PUNCTUATION: u64 = 2;
/// The general symbols — mathematical, modifier and the rest.
pub(super) const GROUP_SYMBOL: u64 = 3;
/// The currency symbols, which the root table places above the general ones.
pub(super) const GROUP_CURRENCY: u64 = 4;
/// The length of a digit run under `numeric=yes`, which orders numbers by magnitude before their
/// digits are looked at.
const GROUP_NUMBER: u64 = 5;
/// The decimal digits.
const GROUP_DIGIT: u64 = 6;
/// The letters.
const GROUP_LETTER: u64 = 7;
/// Everything the table does not name.
const GROUP_IMPLICIT: u64 = 8;
/// One above every group, so that [`QUATERNARY_MAX`] outweighs any primary shifted down to the
/// fourth level.
const GROUP_ABOVE_ALL: u64 = 9;

/// The weight a non-variable element contributes at the fourth level under `alternate=shifted`.
///
/// [UTS #10 §3.4](https://www.unicode.org/reports/tr10/#Variable_Weighting) gives it the maximum,
/// so that a variable character placed against an ordinary one always loses.
pub(super) const QUATERNARY_MAX: u64 = primary(GROUP_ABOVE_ALL, 0);

/// The secondary weight of a base character that carries no accent of its own.
pub(super) const SECONDARY_COMMON: u64 = 0x20;

/// The secondary weight of a letter a tailoring equates with another — Swedish `ø` with `ö`,
/// say — which is a secondary difference rather than no difference at all.
pub(super) const SECONDARY_TAILORED: u64 = 0x21;

/// Where the combining marks' secondary weights start, above every base character's.
const SECONDARY_MARK: u64 = 0x1000;

/// The tertiary variant rank of an ordinary character.
pub(super) const VARIANT_NONE: u8 = 0;

/// The tertiary variant rank of a character that weighs as an expansion — either `s` of `ß`.
const VARIANT_EXPANSION: u8 = 1;

// The groups are the root table's own order, and `MaxVariable` compares the four variable ones
// with `<=`, so their relative values are load-bearing rather than arbitrary.
const _: () = assert!(GROUP_WHITESPACE < GROUP_PUNCTUATION);
const _: () = assert!(GROUP_PUNCTUATION < GROUP_SYMBOL);
const _: () = assert!(GROUP_SYMBOL < GROUP_CURRENCY);
const _: () = assert!(GROUP_CURRENCY < GROUP_NUMBER);
const _: () = assert!(GROUP_NUMBER < GROUP_DIGIT);
const _: () = assert!(GROUP_DIGIT < GROUP_LETTER);
const _: () = assert!(GROUP_LETTER < GROUP_IMPLICIT);
const _: () = assert!(GROUP_IMPLICIT < GROUP_ABOVE_ALL);

/// Assembles a primary weight from its group and its position within the group.
const fn primary(group: u64, position: u64) -> u64 {
    (group << GROUP_SHIFT) | position
}

/// The primary weight of the `rank`th letter of the alphabet, counting `a` as 1.
///
/// Consecutive letters are 0x100 apart, so that a tailoring can slot a letter of its own between
/// two of them without disturbing the rest.
pub(super) const fn letter_primary(rank: u64) -> u64 {
    primary(GROUP_LETTER, rank * 0x100)
}

/// The primary weight the length of a `numeric=yes` digit run contributes.
pub(super) const fn number_length_primary(length: u64) -> u64 {
    primary(GROUP_NUMBER, length)
}

/// The primary weight of a decimal digit of the given value.
pub(super) const fn digit_primary(value: u64) -> u64 {
    primary(GROUP_DIGIT, value)
}

/// The tertiary weight of an element whose case and variant ranks are given.
///
/// Case is the more significant of the two, so a compatibility variant — either `s` of `ß` — is a
/// smaller difference than a change of case. The weight is never zero, which would take the
/// element out of the level altogether.
pub(super) const fn tertiary_weight(case: u8, variant: u8) -> u16 {
    (case as u16) * 0x100 + variant as u16 + 1
}

/// How many base-255 digits a sort key writes a primary weight as.
pub(super) const PRIMARY_WIDTH: usize = 6;
/// How many base-255 digits a sort key writes a secondary weight as.
pub(super) const SECONDARY_WIDTH: usize = 4;
/// How many base-255 digits a sort key writes a tertiary weight as.
pub(super) const TERTIARY_WIDTH: usize = 2;
/// How many base-255 digits a sort key writes a case-level weight as.
pub(super) const CASE_WIDTH: usize = 1;
/// The widest any level writes a single weight, which sizes the scratch buffer.
pub(super) const MAX_WEIGHT_WIDTH: usize = PRIMARY_WIDTH;

/// The largest value `width` base-255 digits can hold.
const fn capacity(width: usize) -> u64 {
    let mut value = 1_u64;
    let mut remaining = width;
    while remaining > 0 {
        value *= 255;
        remaining -= 1;
    }
    value - 1
}

// A sort key is only faithful to the comparison while every weight fits the width the key writes
// it in, so the widths are checked against the table rather than trusted.
const _: () = assert!(QUATERNARY_MAX <= capacity(PRIMARY_WIDTH));
const _: () = assert!(SECONDARY_MARK + char::MAX as u64 <= capacity(SECONDARY_WIDTH));
const _: () = assert!(tertiary_weight(1, u8::MAX) as u64 <= capacity(TERTIARY_WIDTH));
const _: () = assert!(1 <= capacity(CASE_WIDTH));

/// Whether `character` is a combining mark, which weighs at the secondary level only.
pub(super) fn is_combining(character: char) -> bool {
    matches!(
        general_category(character),
        GeneralCategory::NonspacingMark
            | GeneralCategory::SpacingMark
            | GeneralCategory::EnclosingMark
    )
}

/// The group the `alternate` parameter may set `character` aside with, or `None` when it is not a
/// variable character at all.
pub(super) fn variable_group(character: char) -> Option<u64> {
    match general_category(character) {
        GeneralCategory::SpaceSeparator
        | GeneralCategory::LineSeparator
        | GeneralCategory::ParagraphSeparator => Some(GROUP_WHITESPACE),
        // The root table weighs the layout controls as whitespace rather than discarding them;
        // every other control character is completely ignorable.
        GeneralCategory::Control => {
            matches!(character, '\t' | '\n' | '\u{b}' | '\u{c}' | '\r' | '\u{85}')
                .then_some(GROUP_WHITESPACE)
        }
        GeneralCategory::ConnectorPunctuation
        | GeneralCategory::DashPunctuation
        | GeneralCategory::OpenPunctuation
        | GeneralCategory::ClosePunctuation
        | GeneralCategory::InitialPunctuation
        | GeneralCategory::FinalPunctuation
        | GeneralCategory::OtherPunctuation => Some(GROUP_PUNCTUATION),
        GeneralCategory::MathSymbol
        | GeneralCategory::ModifierSymbol
        | GeneralCategory::OtherSymbol => Some(GROUP_SYMBOL),
        GeneralCategory::CurrencySymbol => Some(GROUP_CURRENCY),
        _ => None,
    }
}

/// `0` for a lower-case or caseless character, `1` for an upper-case one.
///
/// Upper case is exactly "has a lower-case form of its own", which is what the root table's
/// tertiary weights record.
pub(super) fn case_of(character: char) -> u8 {
    u8::from(simple_lowercase(character) != character)
}

/// The characters that weigh as a *sequence* of other characters.
///
/// `ß` is the entry the conformance suite turns on: `Straße` and `Strasse` are one word at primary
/// and secondary strength. The rest are the ligatures whose decomposition is a compatibility one,
/// which the canonical decomposition does not apply and the table therefore has to spell out.
const EXPANSIONS: &[(char, &str)] = &[
    ('\u{00df}', "ss"),  // LATIN SMALL LETTER SHARP S
    ('\u{1e9e}', "SS"),  // LATIN CAPITAL LETTER SHARP S
    ('\u{0132}', "IJ"),  // LATIN CAPITAL LIGATURE IJ
    ('\u{0133}', "ij"),  // LATIN SMALL LIGATURE IJ
    ('\u{fb00}', "ff"),  // LATIN SMALL LIGATURE FF
    ('\u{fb01}', "fi"),  // LATIN SMALL LIGATURE FI
    ('\u{fb02}', "fl"),  // LATIN SMALL LIGATURE FL
    ('\u{fb03}', "ffi"), // LATIN SMALL LIGATURE FFI
    ('\u{fb04}', "ffl"), // LATIN SMALL LIGATURE FFL
    ('\u{fb05}', "st"),  // LATIN SMALL LIGATURE LONG S T
    ('\u{fb06}', "st"),  // LATIN SMALL LIGATURE ST
];

/// The letters `character` weighs as, and the tertiary variant rank each of them carries, when it
/// weighs as more than one letter.
pub(super) fn expansion(character: char) -> Option<(&'static str, u8)> {
    EXPANSIONS
        .iter()
        .find(|(from, _)| *from == character)
        .map(|(_, to)| (*to, VARIANT_EXPANSION))
}

/// The primary and secondary weights of a character that weighs as exactly one element, or `None`
/// when it is completely ignorable and weighs at no level at all.
pub(super) fn plain_weight(character: char) -> Option<(u64, u64)> {
    if let Some(group) = variable_group(character) {
        return Some((primary(group, character as u64), SECONDARY_COMMON));
    }
    match general_category(character) {
        // A format character — a zero-width joiner, a bidirectional override, a soft hyphen — is
        // invisible to the reader and so weighs at no level, as are the control characters the
        // root table does not count as whitespace.
        GeneralCategory::Format | GeneralCategory::Control => None,
        GeneralCategory::DecimalNumber => {
            Some((digit_primary(decimal_value(character)), SECONDARY_COMMON))
        }
        _ => {
            let folded = simple_fold(character);
            let weight = match latin_rank(folded) {
                Some(rank) => letter_primary(rank),
                None => implicit_primary(folded),
            };
            Some((weight, SECONDARY_COMMON))
        }
    }
}

/// The value of `character` when it is a decimal digit of any script, which is what a
/// `numeric=yes` run is read one digit at a time with.
pub(super) fn decimal_digit(character: char) -> Option<u64> {
    (general_category(character) == GeneralCategory::DecimalNumber)
        .then(|| decimal_value(character))
}

/// The secondary weight of a combining mark.
///
/// Marks are ordered among themselves by code point, which within a script is the order the root
/// table gives them, and every one of them outweighs the [`SECONDARY_COMMON`] a bare base
/// character contributes — so `data` sorts before `dâta`, as a secondary difference must.
pub(super) fn mark_secondary(character: char) -> u64 {
    SECONDARY_MARK + character as u64
}

/// The position of a basic Latin letter in the alphabet, counting `a` as 1.
fn latin_rank(character: char) -> Option<u64> {
    character
        .is_ascii_lowercase()
        .then(|| character as u64 - 'a' as u64 + 1)
}

/// The implicit weight of a code point the table does not name, above every explicit weight.
fn implicit_primary(character: char) -> u64 {
    primary(GROUP_IMPLICIT, character as u64)
}

/// The value of a decimal digit of any script.
///
/// Unicode lays the decimal digits out in contiguous runs of ten, so the value of one of them is
/// its distance from the start of its run — found by walking back at most nine places, which needs
/// no table of every digit block.
fn decimal_value(character: char) -> u64 {
    let code = character as u32;
    let mut value = 0;
    while value < 9 {
        let Some(previous) = code.checked_sub(value + 1).and_then(char::from_u32) else {
            break;
        };
        if general_category(previous) != GeneralCategory::DecimalNumber {
            break;
        }
        value += 1;
    }
    u64::from(value)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_root_groups_are_in_the_order_the_table_documents() {
        assert_eq!(variable_group(' '), Some(GROUP_WHITESPACE));
        assert_eq!(variable_group('\t'), Some(GROUP_WHITESPACE));
        assert_eq!(variable_group('-'), Some(GROUP_PUNCTUATION));
        assert_eq!(variable_group('_'), Some(GROUP_PUNCTUATION));
        assert_eq!(variable_group('='), Some(GROUP_SYMBOL));
        assert_eq!(variable_group('$'), Some(GROUP_CURRENCY));
        assert_eq!(variable_group('a'), None);
        assert_eq!(variable_group('7'), None);
    }

    /// The whole point of the group layer: punctuation weighs below the digits and the letters
    /// however high its code point is, which code point order does not manage.
    #[test]
    fn punctuation_weighs_below_the_digits_and_the_letters() {
        let weight = |character: char| plain_weight(character).map(|(primary, _)| primary);
        assert!(weight('_') < weight('0'));
        assert!(weight('~') < weight('0'));
        assert!(weight('0') < weight('a'));
        assert!(weight('a') < weight('z'));
        assert!(
            weight('z') < weight('\u{3042}'),
            "a kana takes an implicit weight"
        );
        assert_eq!(weight('a'), weight('A'), "case is not a primary difference");
    }

    #[test]
    fn a_format_character_weighs_at_no_level() {
        assert_eq!(plain_weight('\u{200d}'), None, "zero width joiner");
        assert_eq!(plain_weight('\u{00ad}'), None, "soft hyphen");
        assert_eq!(
            plain_weight('\u{0000}'),
            None,
            "a control that is not whitespace"
        );
        assert!(
            plain_weight('\t').is_some(),
            "a layout control is whitespace"
        );
    }

    #[test]
    fn the_digits_of_every_script_weigh_by_value() {
        for (character, value) in [('0', 0), ('7', 7), ('9', 9)] {
            assert_eq!(decimal_value(character), value, "for `{character}`");
        }
        // ARABIC-INDIC DIGIT THREE and DEVANAGARI DIGIT THREE.
        assert_eq!(decimal_value('\u{0663}'), 3);
        assert_eq!(decimal_value('\u{0969}'), 3);
        assert_eq!(plain_weight('\u{0663}'), plain_weight('3'));
    }

    #[test]
    fn an_implicit_weight_keeps_code_point_order() {
        let weight = |character: char| plain_weight(character).map(|(primary, _)| primary);
        assert!(
            weight('\u{03b1}') < weight('\u{0430}'),
            "Greek before Cyrillic"
        );
        assert!(
            weight('\u{0430}') < weight('\u{4e00}'),
            "Cyrillic before Han"
        );
        assert!(weight('\u{4e00}') < weight('\u{4e8c}'), "Han by code point");
    }

    #[test]
    fn the_expansions_are_the_letters_they_spell() {
        assert_eq!(expansion('\u{00df}'), Some(("ss", VARIANT_EXPANSION)));
        assert_eq!(expansion('\u{1e9e}'), Some(("SS", VARIANT_EXPANSION)));
        assert_eq!(expansion('s'), None);
        for (from, to) in EXPANSIONS {
            assert!(
                to.chars().all(|part| expansion(part).is_none()),
                "the expansion of `{from}` is spelled with letters that expand no further"
            );
        }
    }

    #[test]
    fn case_is_read_from_the_lower_case_mapping() {
        assert_eq!(case_of('a'), 0);
        assert_eq!(case_of('A'), 1);
        assert_eq!(case_of('7'), 0);
        assert_eq!(case_of('\u{0301}'), 0, "a combining mark is caseless");
    }

    /// Case outranks the variant, and neither can produce the zero that would drop an element out
    /// of the tertiary level.
    #[test]
    fn a_tertiary_weight_ranks_case_above_the_variant() {
        assert!(tertiary_weight(0, VARIANT_NONE) > 0);
        assert!(tertiary_weight(0, VARIANT_NONE) < tertiary_weight(0, VARIANT_EXPANSION));
        assert!(tertiary_weight(0, u8::MAX) < tertiary_weight(1, VARIANT_NONE));
    }
}
