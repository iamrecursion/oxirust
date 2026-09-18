//! Character classes shared by both dialects.
//!
//! A character class is represented as a *predicate tree* rather than an
//! enumerated set of code points: enumerating category and block escapes such
//! as `\p{L}` would require materialising hundreds of thousands of ranges. The
//! tree mirrors the XSD grammar productions directly — a class is a possibly
//! negated list of members, optionally with a nested subtraction — so
//! membership, negation, and range subtraction are all decided in one linear
//! walk (see [`ClassExpr::contains`]).
//!
//! Case-insensitive membership follows the definition in
//! [*XPath and XQuery Functions and Operators 3.1*][fo] §5.6.2: a character
//! `c2` is a case-variant of `c1` when their simple lower-case forms are equal
//! or their simple upper-case forms are equal. [`any_variant`] tests a small
//! closed set of case-variants of the input character, which additionally
//! captures compatibility characters (for example `U+212A KELVIN SIGN` folding
//! to `k`).
//!
//! [fo]: https://www.w3.org/TR/xpath-functions-31/#flags

use oxixml_unicode::gc::GeneralCategory;
use oxixml_unicode::xml;
use oxixml_unicode::{blocks, case, gc};

/// Applies `pred` to `c` and to every simple case-variant of `c`, returning
/// `true` as soon as one of them satisfies the predicate.
///
/// The variants tested are the character itself, its simple lower/upper/title
/// mappings, its simple case fold, and the upper/lower mappings of that fold.
/// This is the set used for case-insensitive matching of literal characters and
/// character ranges.
pub(crate) fn any_variant(c: char, mut pred: impl FnMut(char) -> bool) -> bool {
    let f = case::simple_fold(c);
    pred(c)
        || pred(case::simple_lowercase(c))
        || pred(case::simple_uppercase(c))
        || pred(case::simple_titlecase(c))
        || pred(f)
        || pred(case::simple_uppercase(f))
        || pred(case::simple_lowercase(f))
}

/// A multi-character escape (`\d`, `\w`, `\s`, `\i`, `\c` and their negations).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Shorthand {
    /// `\d` — a character in Unicode general category `Nd`.
    Digit,
    /// `\D` — the complement of `\d`.
    NotDigit,
    /// `\w` — a "word" character: anything not in categories `P`, `Z`, or `C`.
    Word,
    /// `\W` — the complement of `\w`.
    NotWord,
    /// `\s` — one of space, tab, line feed, or carriage return.
    Space,
    /// `\S` — the complement of `\s`.
    NotSpace,
    /// `\i` — an XML `NameStartChar`.
    NameStart,
    /// `\I` — the complement of `\i`.
    NotNameStart,
    /// `\c` — an XML `NameChar`.
    NameChar,
    /// `\C` — the complement of `\c`.
    NotNameChar,
}

impl Shorthand {
    /// Returns `true` if `c` is a member of the shorthand class.
    fn contains(self, c: char) -> bool {
        match self {
            Shorthand::Digit => is_digit(c),
            Shorthand::NotDigit => !is_digit(c),
            Shorthand::Word => is_word(c),
            Shorthand::NotWord => !is_word(c),
            Shorthand::Space => is_space(c),
            Shorthand::NotSpace => !is_space(c),
            Shorthand::NameStart => xml::is_name_start_char(c),
            Shorthand::NotNameStart => !xml::is_name_start_char(c),
            Shorthand::NameChar => xml::is_name_char(c),
            Shorthand::NotNameChar => !xml::is_name_char(c),
        }
    }
}

// `is_digit` (`\d`/`\D`) and `is_word` (`\w`/`\W`) below, and `PropName`'s
// `Group`/`Category` matching further down, all resolve straight through to
// `oxixml_unicode::gc` -- i.e. to the crate's single, current Unicode 17
// snapshot (`oxixml_unicode::UNICODE_VERSION`), with no per-escape
// historical overrides. See the doc comment on `impl PropName` for why that
// is a deliberate choice, not a gap: every XSTS `msData/regex` instance
// test whose value touches a code point outside the XSD-1.0-era Unicode
// snapshot (Ethiopic digits U+1369/U+1371, and Mathematical Alphanumeric
// Symbols / other post-2001 SMP allocations) has been checked directly
// against `Regex::xsd`, and the *test metadata*, not this mapping, is the
// outlier; see `tests/w3c_regex_w3c.rs` for the full sweep.

fn is_digit(c: char) -> bool {
    gc::general_category(c) == GeneralCategory::DecimalNumber
}

fn is_space(c: char) -> bool {
    matches!(c, ' ' | '\t' | '\n' | '\r')
}

fn is_word(c: char) -> bool {
    !matches!(gc::general_category(c).group(), 'P' | 'Z' | 'C')
}

/// The name behind a `\p{...}` / `\P{...}` category or block escape.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum PropName {
    /// A one-letter category group, such as `\p{L}`.
    Group(char),
    /// A two-letter general category, such as `\p{Lu}`.
    Category(GeneralCategory),
    /// A Unicode block, written `\p{IsBasicLatin}`; the stored name has the
    /// leading `Is` removed and is already resolved (see
    /// [`resolve_block_name`]) to a spelling `oxixml_unicode::blocks` can
    /// look up directly.
    Block(String),
    /// An `\p{IsXxx}` block escape whose `Xxx` is syntactically a valid XSD
    /// `IsBlock` (a non-empty run of ASCII letters, digits, and hyphens) but
    /// names no block this engine recognises. Valid only in the *XSD*
    /// dialect (see [`resolve_block_name`] for why) and matches **every**
    /// character, mirroring how `\P{IsXxx}` on the same name then matches
    /// none.
    UnknownBlock,
}

// Category-membership audit against every `Regex_w3c` instance test in the
// XSTS `msData/regex` cluster (2590 test groups; see
// `tests/w3c_regex_w3c.rs` for the full data-driven sweep this crate is
// graded against). This mapping is intentionally *unchanged* for the 18
// disputed cases below -- confirmed mechanically
// (`Regex::xsd(...).is_match(...)` against every code point of every
// affected test's value, cross-checked against
// `oxixml_unicode::gc::general_category`, the archived Unicode 3.1.0 and
// 4.0.0 `UnicodeData.txt` snapshots, and the archived text of the W3C bug
// cited below) rather than assumed, and the conclusion is that the crate is
// already correct:
//
// * 16 cases -- `reJ11` (`\p{Lu}*`), `reJ13` (`\p{Ll}*`), `reJ19`
//   (`\p{Lo}*`), `reJ21` (`\p{M}*`), `reJ23` (`\p{Mn}*`), `reJ25`
//   (`\p{Mc}*`), `reJ29` (`\p{N}*`), `reJ31` (`\p{Nd}*`), `reJ33`
//   (`\p{Nl}*`), `reJ35` (`\p{No}*`), `reJ61` (`\p{S}*`), `reJ69`
//   (`\p{So}*`), `reJ75` (`\p{Cf}*`), `reJ77` (`(\p{Co})*`), `reS21` and
//   `reS42` (`\d`) -- test a code point outside the Basic Multilingual
//   Plane (for example `reJ13`'s U+1D7C9, "MATHEMATICAL SANS-SERIF BOLD
//   ITALIC PI SYMBOL") against exactly the category the pattern names, and
//   the code point really does carry that category in Unicode 17
//   ([`oxixml_unicode::UNICODE_VERSION`]). msMeta's `<instanceTest>`
//   `<expected validity="invalid"/>` for all 16 *contradicts* the original
//   test author's own comment in the corresponding `.xsd` file
//   (`type='valid'`) and is marked `status="queried"` in msMeta, not
//   `"accepted"`, with `bugzilla="...show_bug.cgi?id=4113"`. That bug
//   (filed by Michael Kay -- Saxon's author and the editor of the QT3
//   `fn-matches.re` catalogue this crate is separately graded against, and
//   readable via the Wayback Machine now that the old W3C Bugzilla itself
//   is gone) identifies the common thread across exactly this set of cases
//   as "Unicode character classes assigned to characters outside the BMP" --
//   *not* an assignment-timing problem: every one of these code points
//   already carried its present-day category in Unicode 3.1.0 (March
//   2001), before or concurrent with the suite's own "circa 2001-2004"
//   authoring window, per that archived `UnicodeData-3.1.0.txt` snapshot.
//   The bug sat open, unresolved, from December 2006 until the old W3C
//   Bugzilla was archived in 2019: msMeta's "invalid" is the disputed value,
//   never a settled correction, most plausibly reflecting a non-BMP
//   handling limitation of whatever reference implementation the suite was
//   checked against at the time (outside what this crate can verify), not
//   a live ambiguity in the category assignment itself.
// * 2 cases -- `reT63` and `reT84` (both `\D`, "not `\p{Nd}`") -- test
//   U+1D7CD and U+1D800 respectively. Both remain genuinely unassigned
//   (`Cn`) in Unicode 17, exactly as in the Unicode 3.1.0/4.0.0 snapshots
//   (there is a real, permanent two-code-point gap between the `Ll` run
//   ending at U+1D7CB and the `Nd` run starting at U+1D7CE in
//   `oxixml_unicode`'s category table); an unassigned code point is by
//   definition not `Nd`, so `\D` correctly matches both under every
//   Unicode version checked. msMeta's "invalid" here has no textual basis
//   this crate could find in any of those snapshots.
// * XSD's own text only ever commits an implementation to "an
//   implementation-defined, but fixed, version of ... Unicode"; it does not
//   pin one. Every actively-maintained processor (Xerces, Saxon itself,
//   .NET) validates `\p{...}` against its *current* bundled Unicode data
//   and would agree with this crate rather than with the disputed
//   metadata. See `tests/w3c_regex_w3c.rs` for the executable regression
//   test that locks this policy in across the whole cluster, including the
//   paired reJ14/reJ22 negative control (`\P{Ll}`/`\P{M}` on `U+01C5`, an
//   unrelated `Lt` character, which is *not* disputed and which this crate
//   has always rejected correctly) and `tests/xsd_corner_cases.rs`'s
//   `category_escapes_use_current_unicode_not_stale_xsts_metadata`.
impl PropName {
    /// Returns `true` if `c` has the named property (before any `\P` negation).
    fn contains(&self, c: char) -> bool {
        match self {
            PropName::Group(g) => gc::matches_group(c, *g),
            PropName::Category(cat) => gc::general_category(c) == *cat,
            PropName::Block(name) => blocks::in_block(c, name),
            PropName::UnknownBlock => true,
        }
    }
}

/// Legacy XSD/Unicode-3.1 block-escape names whose underlying Unicode block
/// has since been renamed. `\p{IsGreek}`, `\p{IsPrivateUse}`,
/// `\p{IsCombiningMarksforSymbols}`, and `\p{IsCyrillicSupplementary}` are
/// exact spellings that XSD 1.0 Appendix F / XSD 1.1 Appendix G.4.2.3's
/// block-name table is built from (based on Unicode 3.1's `Blocks.txt`, with
/// the fourth added in Unicode 3.2/4.0.0); Unicode has since renamed the
/// corresponding blocks to "Greek and Coptic", "Private Use Area",
/// "Combining Diacritical Marks for Symbols", and "Cyrillic Supplement"
/// (renamed in 4.0.1) respectively — confirmed against the W3C's own
/// [block-name changelog][blocknames], which lists exactly these four
/// name-changing rows (as opposed to the range-only adjustments elsewhere in
/// that table, which do not need an alias here since the *name* did not
/// move). This drift is a known, W3C-acknowledged spec/Unicode-version
/// mismatch ("some of the names of code blocks ... have changed in
/// different versions of Unicode, for example Greek is now 'Greek and
/// Coptic' ... This change has not been reflected in the list of allowed
/// block names" — `www-xml-schema-comments` bug 5486), so a conformant
/// processor keeps accepting the legacy spelling. Exercised by the XML
/// Schema Test Suite: `msData/regex/reN8` (`\p{IsGreek}`),
/// `reL78`/`reL98`/`reL99`/`reM78`/`reM98`/`reM99`/`reN98`/`reN99`
/// (`\p{IsPrivateUse}`), and `reL43`/`reM43`/`reN43`
/// (`\p{IsCombiningMarksforSymbols}`); `\p{IsCyrillicSupplementary}` is not
/// exercised by that suite but is the same class of drift, so it is handled
/// the same way rather than silently falling through to
/// [`PropName::UnknownBlock`] (which would make it match every character
/// instead of just the Cyrillic Supplement block).
///
/// [blocknames]: https://www.w3.org/TR/xsd-unicode-blocknames/
const LEGACY_BLOCK_ALIASES: &[(&str, &str)] = &[
    ("Greek", "Greek and Coptic"),
    ("PrivateUse", "Private Use Area"),
    (
        "CombiningMarksforSymbols",
        "Combining Diacritical Marks for Symbols",
    ),
    ("CyrillicSupplementary", "Cyrillic Supplement"),
];

/// The permanent Unicode blocks that lie entirely inside the UTF-16 surrogate
/// range `U+D800..=U+DFFF`. A Rust [`char`] is a Unicode *scalar value*,
/// which excludes surrogate code points by definition, so no `char` can ever
/// belong to one of these blocks — but `\p{IsHighSurrogates}` and
/// `\p{IsLowSurrogates}` are legitimate, permanently-fixed Unicode block
/// names (the surrogate range has been partitioned this way since Unicode
/// 2.0) that must still compile; they simply match no character, exactly
/// like `\p{Cs}` (general category "Surrogate"). Exercised by
/// `msData/regex/reN75` (`\p{IsHighSurrogates}`), `reM77`
/// (`\p{IsLowSurrogates}`), and `HighSurrogates`.
const SURROGATE_BLOCK_NAMES: &[&str] = &[
    "High Surrogates",
    "High Private Use Surrogates",
    "Low Surrogates",
];

/// Case-insensitive, space/hyphen/underscore-insensitive name comparison,
/// mirroring the matching rule `oxixml_unicode::blocks` itself uses (loose
/// [UAX #44-LM3][lm3]). Used only to test a candidate name against the small
/// compatibility tables above.
///
/// [lm3]: https://www.unicode.org/reports/tr44/#UAX44-LM3
fn loose_name_eq(a: &str, b: &str) -> bool {
    fn normalize(s: &str) -> String {
        s.chars()
            .filter(|c| !matches!(c, ' ' | '-' | '_'))
            .flat_map(char::to_lowercase)
            .collect()
    }
    normalize(a) == normalize(b)
}

/// Resolves the text after `Is` in a `\p{Is...}` / `\P{Is...}` escape (for
/// example `"BasicLatin"` or `"Greek"`) to the block name that should be
/// stored in a [`PropName::Block`] and matched against later.
///
/// Tries, in order: (1) `name` as given, via `oxixml_unicode::blocks`; (2)
/// the [`LEGACY_BLOCK_ALIASES`]; (3) `name` unchanged, if it names one of the
/// permanently-unrepresentable [`SURROGATE_BLOCK_NAMES`]. Returns `None` if
/// none of these recognise `name` — the caller decides what an unrecognised
/// name means for the active dialect.
pub(crate) fn resolve_block_name(name: &str) -> Option<String> {
    if blocks::block_range(name).is_some() {
        return Some(name.to_owned());
    }
    for &(legacy, modern) in LEGACY_BLOCK_ALIASES {
        if loose_name_eq(name, legacy) {
            return Some(modern.to_owned());
        }
    }
    if SURROGATE_BLOCK_NAMES
        .iter()
        .any(|surrogate| loose_name_eq(name, surrogate))
    {
        return Some(name.to_owned());
    }
    None
}

/// Resolves a two-letter general-category abbreviation such as `"Lu"`.
pub(crate) fn category_from_abbreviation(name: &str) -> Option<GeneralCategory> {
    const ALL: [GeneralCategory; 30] = [
        GeneralCategory::UppercaseLetter,
        GeneralCategory::LowercaseLetter,
        GeneralCategory::TitlecaseLetter,
        GeneralCategory::ModifierLetter,
        GeneralCategory::OtherLetter,
        GeneralCategory::NonspacingMark,
        GeneralCategory::SpacingMark,
        GeneralCategory::EnclosingMark,
        GeneralCategory::DecimalNumber,
        GeneralCategory::LetterNumber,
        GeneralCategory::OtherNumber,
        GeneralCategory::ConnectorPunctuation,
        GeneralCategory::DashPunctuation,
        GeneralCategory::OpenPunctuation,
        GeneralCategory::ClosePunctuation,
        GeneralCategory::InitialPunctuation,
        GeneralCategory::FinalPunctuation,
        GeneralCategory::OtherPunctuation,
        GeneralCategory::MathSymbol,
        GeneralCategory::CurrencySymbol,
        GeneralCategory::ModifierSymbol,
        GeneralCategory::OtherSymbol,
        GeneralCategory::SpaceSeparator,
        GeneralCategory::LineSeparator,
        GeneralCategory::ParagraphSeparator,
        GeneralCategory::Control,
        GeneralCategory::Format,
        GeneralCategory::Surrogate,
        GeneralCategory::PrivateUse,
        GeneralCategory::Unassigned,
    ];
    ALL.into_iter().find(|cat| cat.abbreviation() == name)
}

/// A single member of a character group.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum ClassItem {
    /// A single literal character.
    Single(char),
    /// An inclusive code-point range `lo..=hi` (`lo <= hi` is guaranteed by the
    /// parser).
    Range(char, char),
    /// A multi-character escape.
    Shorthand(Shorthand),
    /// A `\p{...}` or `\P{...}` property escape; `negated` is set for `\P`.
    Prop {
        /// The property matched.
        name: PropName,
        /// Whether the sense of the property is inverted (`\P`).
        negated: bool,
    },
}

impl ClassItem {
    /// Returns `true` if `c` matches this member, honouring case-insensitivity.
    fn contains(&self, c: char, ci: bool) -> bool {
        match self {
            ClassItem::Single(x) => {
                if ci {
                    any_variant(c, |v| v == *x)
                } else {
                    c == *x
                }
            }
            ClassItem::Range(lo, hi) => {
                if ci {
                    any_variant(c, |v| *lo <= v && v <= *hi)
                } else {
                    *lo <= c && c <= *hi
                }
            }
            // Shorthand and property escapes are category driven; the
            // case-variant rule applies only to literal characters and ranges.
            ClassItem::Shorthand(s) => s.contains(c),
            ClassItem::Prop { name, negated } => name.contains(c) ^ negated,
        }
    }
}

/// A compiled character-class expression: a possibly negated group of members,
/// optionally reduced by a nested subtraction.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct ClassExpr {
    /// Whether a leading `^` negated the positive group.
    pub negated: bool,
    /// The positive members of the group.
    pub items: Vec<ClassItem>,
    /// An optional class whose members are removed (`[A-[B]]`).
    pub subtract: Option<Box<ClassExpr>>,
}

impl ClassExpr {
    /// The class with no members at all, which no character matches.
    ///
    /// A compiler uses it to emit an instruction that always fails — the
    /// instruction set has no dedicated `Fail`, and an empty positive group is
    /// exactly that.
    pub(crate) fn never() -> Self {
        Self {
            negated: false,
            items: Vec::new(),
            subtract: None,
        }
    }

    /// Returns `true` if `c` is a member of the class.
    ///
    /// The evaluation order matches the grammar: first the positive group, then
    /// the `^` negation, then the subtraction.
    pub(crate) fn contains(&self, c: char, ci: bool) -> bool {
        let base = self.items.iter().any(|it| it.contains(c, ci));
        let after_negation = self.negated ^ base;
        match &self.subtract {
            Some(sub) => after_negation && !sub.contains(c, ci),
            None => after_negation,
        }
    }

    /// A class that matches every character (a negated empty group).
    pub(crate) fn any() -> ClassExpr {
        ClassExpr {
            negated: true,
            items: Vec::new(),
            subtract: None,
        }
    }

    /// The `.` wildcard outside dot-all mode: any character except line feed or
    /// carriage return.
    pub(crate) fn dot() -> ClassExpr {
        ClassExpr {
            negated: true,
            items: vec![ClassItem::Single('\n'), ClassItem::Single('\r')],
            subtract: None,
        }
    }

    /// A class holding a single positive member.
    pub(crate) fn single_item(item: ClassItem) -> ClassExpr {
        ClassExpr {
            negated: false,
            items: vec![item],
            subtract: None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn range_membership_case_sensitive() {
        let cls = ClassExpr::single_item(ClassItem::Range('a', 'z'));
        assert!(cls.contains('m', false));
        assert!(!cls.contains('M', false));
    }

    #[test]
    fn range_membership_case_insensitive() {
        let cls = ClassExpr::single_item(ClassItem::Range('a', 'z'));
        assert!(cls.contains('M', true));
        // Kelvin sign folds to 'k' which lies in a-z.
        assert!(cls.contains('\u{212a}', true));
    }

    #[test]
    fn negation_and_subtraction() {
        // [a-z-[aeiou]]
        let sub = ClassExpr::single_item(ClassItem::Single('e'));
        let cls = ClassExpr {
            negated: false,
            items: vec![ClassItem::Range('a', 'z')],
            subtract: Some(Box::new(sub)),
        };
        assert!(cls.contains('b', false));
        assert!(!cls.contains('e', false));

        // [^a]
        let neg = ClassExpr {
            negated: true,
            items: vec![ClassItem::Single('a')],
            subtract: None,
        };
        assert!(neg.contains('b', false));
        assert!(!neg.contains('a', false));
    }

    #[test]
    fn word_shorthand_excludes_connector_punctuation() {
        // \w excludes '_' (category Pc) in the XSD/XPath definition.
        assert!(Shorthand::Word.contains('a'));
        assert!(!Shorthand::Word.contains('_'));
        assert!(!Shorthand::Word.contains(' '));
    }

    #[test]
    fn any_variant_covers_case_forms() {
        assert!(any_variant('A', |v| v == 'a'));
        assert!(any_variant('z', |v| v == 'Z'));
        assert!(!any_variant('1', |v| v == 'a'));
    }

    #[test]
    fn resolve_block_name_direct_hit() {
        assert_eq!(
            resolve_block_name("BasicLatin"),
            Some("BasicLatin".to_owned())
        );
    }

    #[test]
    fn resolve_block_name_legacy_aliases() {
        // XSD/Unicode-3.1 spellings must resolve to whatever
        // oxixml_unicode's table currently calls the (renamed) block.
        assert_eq!(
            resolve_block_name("Greek"),
            Some("Greek and Coptic".to_owned())
        );
        assert_eq!(
            resolve_block_name("PrivateUse"),
            Some("Private Use Area".to_owned())
        );
        assert_eq!(
            resolve_block_name("CombiningMarksforSymbols"),
            Some("Combining Diacritical Marks for Symbols".to_owned())
        );
        assert_eq!(
            resolve_block_name("CyrillicSupplementary"),
            Some("Cyrillic Supplement".to_owned())
        );
        // Loosely (case/hyphen/space/underscore) matched too.
        assert_eq!(
            resolve_block_name("private_use"),
            Some("Private Use Area".to_owned())
        );
    }

    #[test]
    fn resolve_block_name_surrogate_blocks() {
        assert!(resolve_block_name("HighSurrogates").is_some());
        assert!(resolve_block_name("LowSurrogates").is_some());
        assert!(resolve_block_name("HighPrivateUseSurrogates").is_some());
    }

    #[test]
    fn resolve_block_name_unknown_returns_none() {
        assert_eq!(resolve_block_name("NoSuchBlockAtAll"), None);
    }

    #[test]
    fn surrogate_block_matches_no_char() {
        let name = resolve_block_name("HighSurrogates").expect("known name");
        let prop = PropName::Block(name);
        // No `char` can ever be a surrogate code point, so this must be
        // empty over the whole scalar-value space, not just a few samples.
        assert!(!prop.contains('a'));
        assert!(!prop.contains('\u{0}'));
        assert!(!prop.contains(char::MAX));
    }

    #[test]
    fn unknown_block_matches_every_char() {
        let prop = PropName::UnknownBlock;
        assert!(prop.contains('a'));
        assert!(prop.contains('\u{0}'));
        assert!(prop.contains(char::MAX));
    }
}
