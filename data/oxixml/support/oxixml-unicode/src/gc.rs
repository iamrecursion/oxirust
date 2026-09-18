//! Unicode general categories.
//!
//! Provides the 30-value [`GeneralCategory`] enumeration, the
//! [`general_category`] lookup (binary search over generated ranges), and
//! [`matches_group`] for the one-letter category groups (`L`, `M`, `N`, `P`,
//! `S`, `Z`, `C`) that regular-expression `\p{L}`-style constructs use.

use crate::tables::category::CATEGORY_RANGES;

/// A Unicode [general category][gc] — the `General_Category` property value of
/// a character.
///
/// The 30 variants are the two-letter category values of the Unicode Standard.
/// Unassigned code points have category [`Unassigned`](GeneralCategory::Unassigned)
/// (`Cn`).
///
/// The discriminant order matches the layout of the generated lookup table and
/// groups the categories by their leading letter (`L`, `M`, `N`, `P`, `S`, `Z`,
/// `C`).
///
/// [gc]: https://www.unicode.org/reports/tr44/#General_Category_Values
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[repr(u8)]
pub enum GeneralCategory {
    /// `Lu` — an uppercase letter.
    UppercaseLetter = 0,
    /// `Ll` — a lowercase letter.
    LowercaseLetter = 1,
    /// `Lt` — a digraph encoded as a single titlecase letter.
    TitlecaseLetter = 2,
    /// `Lm` — a modifier letter.
    ModifierLetter = 3,
    /// `Lo` — other letters, including syllables and ideographs.
    OtherLetter = 4,
    /// `Mn` — a non-spacing combining mark (zero advance width).
    NonspacingMark = 5,
    /// `Mc` — a spacing combining mark (positive advance width).
    SpacingMark = 6,
    /// `Me` — an enclosing combining mark.
    EnclosingMark = 7,
    /// `Nd` — a decimal digit.
    DecimalNumber = 8,
    /// `Nl` — a letter-like numeric character.
    LetterNumber = 9,
    /// `No` — a numeric character of another type.
    OtherNumber = 10,
    /// `Pc` — a connecting punctuation mark, like a tie.
    ConnectorPunctuation = 11,
    /// `Pd` — a dash or hyphen punctuation mark.
    DashPunctuation = 12,
    /// `Ps` — an opening punctuation mark (of a pair).
    OpenPunctuation = 13,
    /// `Pe` — a closing punctuation mark (of a pair).
    ClosePunctuation = 14,
    /// `Pi` — an initial quotation mark.
    InitialPunctuation = 15,
    /// `Pf` — a final quotation mark.
    FinalPunctuation = 16,
    /// `Po` — a punctuation mark of another type.
    OtherPunctuation = 17,
    /// `Sm` — a symbol of mathematical use.
    MathSymbol = 18,
    /// `Sc` — a currency sign.
    CurrencySymbol = 19,
    /// `Sk` — a non-letter-like modifier symbol.
    ModifierSymbol = 20,
    /// `So` — a symbol of another type.
    OtherSymbol = 21,
    /// `Zs` — a space character of non-zero width.
    SpaceSeparator = 22,
    /// `Zl` — the line separator `U+2028`.
    LineSeparator = 23,
    /// `Zp` — the paragraph separator `U+2029`.
    ParagraphSeparator = 24,
    /// `Cc` — a C0 or C1 control code.
    Control = 25,
    /// `Cf` — a format control character.
    Format = 26,
    /// `Cs` — a surrogate code point.
    Surrogate = 27,
    /// `Co` — a private-use character.
    PrivateUse = 28,
    /// `Cn` — a reserved unassigned code point (or a noncharacter).
    Unassigned = 29,
}

/// A one-letter Unicode category group, as used by regular-expression
/// `\p{L}`-style property escapes.
///
/// Every [`GeneralCategory`] belongs to exactly one group, identified by the
/// first letter of its two-letter abbreviation.
const GROUPS: [GeneralCategory; 30] = [
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

impl GeneralCategory {
    /// Reconstructs a category from its table discriminant.
    fn from_index(index: u8) -> GeneralCategory {
        GROUPS[index as usize]
    }

    /// Returns the two-letter Unicode abbreviation of this category, such as
    /// `"Lu"` or `"Nd"`.
    ///
    /// # Examples
    ///
    /// ```
    /// use oxixml_unicode::gc::GeneralCategory;
    ///
    /// assert_eq!(GeneralCategory::UppercaseLetter.abbreviation(), "Lu");
    /// assert_eq!(GeneralCategory::Unassigned.abbreviation(), "Cn");
    /// ```
    #[must_use]
    pub const fn abbreviation(self) -> &'static str {
        match self {
            GeneralCategory::UppercaseLetter => "Lu",
            GeneralCategory::LowercaseLetter => "Ll",
            GeneralCategory::TitlecaseLetter => "Lt",
            GeneralCategory::ModifierLetter => "Lm",
            GeneralCategory::OtherLetter => "Lo",
            GeneralCategory::NonspacingMark => "Mn",
            GeneralCategory::SpacingMark => "Mc",
            GeneralCategory::EnclosingMark => "Me",
            GeneralCategory::DecimalNumber => "Nd",
            GeneralCategory::LetterNumber => "Nl",
            GeneralCategory::OtherNumber => "No",
            GeneralCategory::ConnectorPunctuation => "Pc",
            GeneralCategory::DashPunctuation => "Pd",
            GeneralCategory::OpenPunctuation => "Ps",
            GeneralCategory::ClosePunctuation => "Pe",
            GeneralCategory::InitialPunctuation => "Pi",
            GeneralCategory::FinalPunctuation => "Pf",
            GeneralCategory::OtherPunctuation => "Po",
            GeneralCategory::MathSymbol => "Sm",
            GeneralCategory::CurrencySymbol => "Sc",
            GeneralCategory::ModifierSymbol => "Sk",
            GeneralCategory::OtherSymbol => "So",
            GeneralCategory::SpaceSeparator => "Zs",
            GeneralCategory::LineSeparator => "Zl",
            GeneralCategory::ParagraphSeparator => "Zp",
            GeneralCategory::Control => "Cc",
            GeneralCategory::Format => "Cf",
            GeneralCategory::Surrogate => "Cs",
            GeneralCategory::PrivateUse => "Co",
            GeneralCategory::Unassigned => "Cn",
        }
    }

    /// Returns the one-letter group this category belongs to: `'L'`, `'M'`,
    /// `'N'`, `'P'`, `'S'`, `'Z'`, or `'C'`.
    ///
    /// # Examples
    ///
    /// ```
    /// use oxixml_unicode::gc::GeneralCategory;
    ///
    /// assert_eq!(GeneralCategory::DecimalNumber.group(), 'N');
    /// assert_eq!(GeneralCategory::Control.group(), 'C');
    /// ```
    #[must_use]
    pub const fn group(self) -> char {
        match self {
            GeneralCategory::UppercaseLetter
            | GeneralCategory::LowercaseLetter
            | GeneralCategory::TitlecaseLetter
            | GeneralCategory::ModifierLetter
            | GeneralCategory::OtherLetter => 'L',
            GeneralCategory::NonspacingMark
            | GeneralCategory::SpacingMark
            | GeneralCategory::EnclosingMark => 'M',
            GeneralCategory::DecimalNumber
            | GeneralCategory::LetterNumber
            | GeneralCategory::OtherNumber => 'N',
            GeneralCategory::ConnectorPunctuation
            | GeneralCategory::DashPunctuation
            | GeneralCategory::OpenPunctuation
            | GeneralCategory::ClosePunctuation
            | GeneralCategory::InitialPunctuation
            | GeneralCategory::FinalPunctuation
            | GeneralCategory::OtherPunctuation => 'P',
            GeneralCategory::MathSymbol
            | GeneralCategory::CurrencySymbol
            | GeneralCategory::ModifierSymbol
            | GeneralCategory::OtherSymbol => 'S',
            GeneralCategory::SpaceSeparator
            | GeneralCategory::LineSeparator
            | GeneralCategory::ParagraphSeparator => 'Z',
            GeneralCategory::Control
            | GeneralCategory::Format
            | GeneralCategory::Surrogate
            | GeneralCategory::PrivateUse
            | GeneralCategory::Unassigned => 'C',
        }
    }
}

/// Returns the [`GeneralCategory`] of `c`.
///
/// Unassigned code points, noncharacters, and reserved positions return
/// [`GeneralCategory::Unassigned`] (`Cn`).
///
/// The lookup is a binary search over a sorted range table and runs in
/// `O(log n)` time.
///
/// # Examples
///
/// ```
/// use oxixml_unicode::gc::{general_category, GeneralCategory};
///
/// assert_eq!(general_category('A'), GeneralCategory::UppercaseLetter);
/// assert_eq!(general_category('a'), GeneralCategory::LowercaseLetter);
/// assert_eq!(general_category('5'), GeneralCategory::DecimalNumber);
/// assert_eq!(general_category(' '), GeneralCategory::SpaceSeparator);
/// assert_eq!(general_category('\u{10FFFF}'), GeneralCategory::Unassigned);
/// ```
#[must_use]
pub fn general_category(c: char) -> GeneralCategory {
    let cp = c as u32;
    let mut lo = 0usize;
    let mut hi = CATEGORY_RANGES.len();
    while lo < hi {
        let mid = lo + (hi - lo) / 2;
        let (start, end, idx) = CATEGORY_RANGES[mid];
        if cp < start {
            hi = mid;
        } else if cp > end {
            lo = mid + 1;
        } else {
            return GeneralCategory::from_index(idx);
        }
    }
    GeneralCategory::Unassigned
}

/// Returns `true` if the [`GeneralCategory`] of `c` belongs to the one-letter
/// `group`.
///
/// The accepted groups are `'L'` (letters), `'M'` (marks), `'N'` (numbers),
/// `'P'` (punctuation), `'S'` (symbols), `'Z'` (separators), and `'C'` (other,
/// which includes unassigned code points). Any other `group` value yields
/// `false`. This is the semantics of a regular-expression `\p{L}`-style escape
/// with a single-letter property name.
///
/// # Examples
///
/// ```
/// use oxixml_unicode::gc::matches_group;
///
/// assert!(matches_group('A', 'L'));
/// assert!(matches_group('5', 'N'));
/// assert!(matches_group('\u{0301}', 'M')); // combining acute accent
/// assert!(matches_group('\u{FFFFF}', 'C')); // unassigned -> Cn -> C
/// assert!(!matches_group('A', 'N'));
/// assert!(!matches_group('A', 'X')); // not a group letter
/// ```
#[must_use]
pub fn matches_group(c: char, group: char) -> bool {
    matches!(group, 'L' | 'M' | 'N' | 'P' | 'S' | 'Z' | 'C') && general_category(c).group() == group
}
