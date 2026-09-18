//! The language tailorings a UCA collation URI's `lang` parameter can ask for.
//!
//! A tailoring is a small set of departures from the root order of [UTS
//! #10](https://www.unicode.org/reports/tr10/), the ones a language's own dictionaries make. Most
//! European languages need none: they read the root order as it stands, and their accented letters
//! sort as the base letter with a secondary difference, which is exactly what the canonical
//! decomposition already produces. The languages that *do* need one mostly need the same kind of
//! rule — a letter that the root order treats as an accented `a` is a letter in its own right,
//! placed somewhere the root order does not put it.
//!
//! This module carries the tailorings as a layer over the root table rather than as a second copy
//! of it: [`Tailoring::letter`] is offered every base character together with the combining marks
//! that follow it, and answers with a weight of its own only for the handful of letters its
//! language moves. Everything else falls through to [`super::weights`].
//!
//! A `lang` this module has no tailoring for is *not* silently approximated: [`Tailoring::for_tag`]
//! answers `None`, and the URI's `fallback` parameter then decides between the root order and
//! `FOCH0002`.

use super::weights::{SECONDARY_COMMON, SECONDARY_TAILORED, letter_primary};

/// COMBINING DIAERESIS, which is what `ä` and `ö` decompose to.
const COMBINING_DIAERESIS: char = '\u{0308}';
/// COMBINING RING ABOVE, which is what `å` decomposes to.
const COMBINING_RING_ABOVE: char = '\u{030a}';

/// The rank of `y`, which Swedish equates `ü` with.
const RANK_Y: u64 = 25;
/// The rank Swedish gives `å`, the first of the three letters it places after `z`.
const RANK_A_RING: u64 = 27;
/// The rank Swedish gives `ä`.
const RANK_A_DIAERESIS: u64 = 28;
/// The rank Swedish gives `ö`.
const RANK_O_DIAERESIS: u64 = 29;

/// A language tailoring layered over the root order.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub(super) enum Tailoring {
    /// The root order itself, unchanged.
    ///
    /// English needs no tailoring at all. German needs none either for the ordering DIN 5007-1
    /// defines: `ä`, `ö` and `ü` sort as `a`, `o` and `u` with a secondary difference, which the
    /// canonical decomposition gives them, and `ß` sorts as `ss`, which the root table's own
    /// expansion gives it. (The *telephone-book* ordering of DIN 5007-2, which sorts `ä` as `ae`,
    /// is a different collation and is not what a bare `lang=de` asks for.)
    #[default]
    Root,
    /// Swedish, which places three letters after `z`.
    ///
    /// `å`, `ä` and `ö` are letters of the alphabet rather than accented vowels, and follow `z` in
    /// that order. `æ` and `ø` — the Danish and Norwegian spellings of the last two — are sorted
    /// with `ä` and `ö`, and `ü` with `y`, each with a secondary difference, as SS 613621
    /// prescribes.
    Swedish,
}

/// The weights a tailoring gives a letter of its own.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) struct TailoredLetter {
    /// The primary weight, which is what places the letter in the alphabet.
    pub(super) primary: u64,
    /// The secondary weight, which separates a letter from the one the tailoring equates it with.
    pub(super) secondary: u64,
    /// How many of the combining marks that followed the base character the letter absorbed.
    pub(super) marks_used: usize,
}

impl TailoredLetter {
    /// A letter of the alphabet in its own right, spelled as a base character and one mark.
    const fn composed(rank: u64) -> Self {
        Self {
            primary: letter_primary(rank),
            secondary: SECONDARY_COMMON,
            marks_used: 1,
        }
    }

    /// A letter the tailoring sorts *with* another, a secondary difference apart.
    const fn equated(rank: u64, marks_used: usize) -> Self {
        Self {
            primary: letter_primary(rank),
            secondary: SECONDARY_TAILORED,
            marks_used,
        }
    }
}

impl Tailoring {
    /// The tailoring a `lang` parameter asks for, or `None` when this collation has none for it.
    ///
    /// Only the primary subtag is read, so `de-AT` and `sv-FI` reach the same tailoring their
    /// languages do. The empty tag, `und` and `root` all name the root order, which is also what a
    /// URI with no `lang` at all uses.
    pub(super) fn for_tag(tag: &str) -> Option<Self> {
        let primary = tag.split(['-', '_']).next().unwrap_or(tag);
        match primary.to_ascii_lowercase().as_str() {
            "" | "und" | "root" | "en" | "de" => Some(Self::Root),
            "sv" => Some(Self::Swedish),
            _ => None,
        }
    }

    /// The weights this tailoring gives `base` followed by `marks`, when it gives it any of its
    /// own.
    ///
    /// `marks` is every combining mark between the base character and the next one, so that a
    /// tailoring can recognise a letter its language spells with an accent the canonical
    /// decomposition has just taken apart.
    pub(super) fn letter(self, base: char, marks: &[char]) -> Option<TailoredLetter> {
        match self {
            Self::Root => None,
            Self::Swedish => swedish_letter(base, marks),
        }
    }
}

/// The Swedish alphabet's departures from the root order.
fn swedish_letter(base: char, marks: &[char]) -> Option<TailoredLetter> {
    match (base, marks.first()) {
        ('a' | 'A', Some(&COMBINING_RING_ABOVE)) => Some(TailoredLetter::composed(RANK_A_RING)),
        ('a' | 'A', Some(&COMBINING_DIAERESIS)) => Some(TailoredLetter::composed(RANK_A_DIAERESIS)),
        ('o' | 'O', Some(&COMBINING_DIAERESIS)) => Some(TailoredLetter::composed(RANK_O_DIAERESIS)),
        ('u' | 'U', Some(&COMBINING_DIAERESIS)) => Some(TailoredLetter::equated(RANK_Y, 1)),
        // `æ` and `ø` decompose to nothing, so they are matched as themselves.
        ('\u{00e6}' | '\u{00c6}', _) => Some(TailoredLetter::equated(RANK_A_DIAERESIS, 0)),
        ('\u{00f8}' | '\u{00d8}', _) => Some(TailoredLetter::equated(RANK_O_DIAERESIS, 0)),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_language_tag_reaches_the_tailoring_of_its_primary_subtag() {
        assert_eq!(Tailoring::for_tag("en"), Some(Tailoring::Root));
        assert_eq!(Tailoring::for_tag("EN-GB"), Some(Tailoring::Root));
        assert_eq!(Tailoring::for_tag("de"), Some(Tailoring::Root));
        assert_eq!(Tailoring::for_tag("de-AT"), Some(Tailoring::Root));
        assert_eq!(Tailoring::for_tag("und"), Some(Tailoring::Root));
        assert_eq!(Tailoring::for_tag("sv"), Some(Tailoring::Swedish));
        assert_eq!(Tailoring::for_tag("sv-FI"), Some(Tailoring::Swedish));
        assert_eq!(Tailoring::for_tag("ja"), None);
        assert_eq!(
            Tailoring::for_tag("se"),
            None,
            "Northern Sami is not Swedish"
        );
    }

    #[test]
    fn the_root_tailoring_moves_no_letter() {
        assert_eq!(Tailoring::Root.letter('a', &[COMBINING_RING_ABOVE]), None);
        assert_eq!(Tailoring::Root.letter('\u{00f8}', &[]), None);
    }

    #[test]
    fn swedish_places_three_letters_after_z() {
        let ring = Tailoring::Swedish
            .letter('a', &[COMBINING_RING_ABOVE])
            .expect("å is a Swedish letter");
        let a_diaeresis = Tailoring::Swedish
            .letter('A', &[COMBINING_DIAERESIS])
            .expect("ä is a Swedish letter");
        let o_diaeresis = Tailoring::Swedish
            .letter('o', &[COMBINING_DIAERESIS])
            .expect("ö is a Swedish letter");
        assert_eq!(ring.marks_used, 1);
        assert!(letter_primary(26) < ring.primary, "å follows z");
        assert!(ring.primary < a_diaeresis.primary);
        assert!(a_diaeresis.primary < o_diaeresis.primary);
    }

    #[test]
    fn swedish_equates_the_danish_spellings_a_secondary_difference_apart() {
        let ash = Tailoring::Swedish
            .letter('\u{00e6}', &[])
            .expect("æ is sorted with ä");
        let a_diaeresis = Tailoring::Swedish
            .letter('a', &[COMBINING_DIAERESIS])
            .expect("ä is a Swedish letter");
        assert_eq!(ash.primary, a_diaeresis.primary);
        assert!(ash.secondary > a_diaeresis.secondary);
        assert_eq!(ash.marks_used, 0, "æ has no mark to absorb");
    }
}
