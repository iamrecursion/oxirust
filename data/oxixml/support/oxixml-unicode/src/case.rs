//! Case mappings and Unicode case folding.
//!
//! Three related but distinct operations live here:
//!
//! * **Simple case mapping** — the one-to-one
//!   [`simple_uppercase`] / [`simple_lowercase`] / [`simple_titlecase`]
//!   mappings taken from fields 12–14 of `UnicodeData.txt`. Each maps a
//!   character to a single character, independent of context or language.
//! * **Full case mapping** — [`full_uppercase`] / [`full_lowercase`] /
//!   [`full_titlecase`] and their [`CaseLocale`]-tailored counterparts
//!   ([`full_uppercase_locale`], [`full_lowercase_locale`],
//!   [`full_titlecase_locale`]), which apply the complete algorithm of
//!   `SpecialCasing.txt`: the unconditional multi-character mappings (`ß` →
//!   `SS`, the ligatures, the Greek letters with ypogegrammeni), the
//!   language-independent `Final_Sigma` context test (Greek `Σ` → `ς` at the
//!   end of a word), and, given an explicit [`CaseLocale`], the Lithuanian,
//!   Turkish, and Azerbaijani tailorings. See the [case mapping FAQ][casemap]
//!   for background on the underlying algorithm.
//! * **Case folding** — [`simple_fold`] and [`fold_str`], from
//!   `CaseFolding.txt`, used for caseless matching. [`simple_fold`] applies the
//!   `C` and `S` (common and simple) mappings and stays one-to-one, while
//!   [`fold_str`] applies the `C` and `F` (common and full) mappings, which may
//!   expand one character into several (for example `ß` folds to `ss`).
//!
//! [casemap]: https://www.unicode.org/faq/casemap_charprop.html

use crate::norm::ccc;
use crate::tables::case_props::{CASE_IGNORABLE_RANGES, CASED_RANGES, SOFT_DOTTED_RANGES};
use crate::tables::folding::{FULL_FOLD, SIMPLE_FOLD};
use crate::tables::simple_case::{SIMPLE_LOWERCASE, SIMPLE_TITLECASE, SIMPLE_UPPERCASE};
use crate::tables::special_case::SPECIAL_CASE;

/// Binary-searches a sorted `(char, char)` mapping table, returning the mapped
/// value or `key` itself when the table has no entry.
fn map_single(table: &[(char, char)], key: char) -> char {
    match table.binary_search_by(|&(from, _)| from.cmp(&key)) {
        Ok(i) => table[i].1,
        Err(_) => key,
    }
}

/// Returns the simple uppercase mapping of `c`, or `c` itself if it has none.
///
/// This is a one-to-one mapping (`UnicodeData.txt` field 12) and does **not**
/// perform the full uppercasing of `SpecialCasing.txt` (so `ß` maps to itself,
/// not to `SS`); see [`full_uppercase`] for that.
///
/// # Examples
///
/// ```
/// use oxixml_unicode::case::simple_uppercase;
///
/// assert_eq!(simple_uppercase('a'), 'A');
/// assert_eq!(simple_uppercase('é'), 'É');
/// assert_eq!(simple_uppercase('5'), '5');
/// ```
#[must_use]
pub fn simple_uppercase(c: char) -> char {
    map_single(SIMPLE_UPPERCASE, c)
}

/// Returns the simple lowercase mapping of `c`, or `c` itself if it has none.
///
/// This is a one-to-one mapping (`UnicodeData.txt` field 13).
///
/// # Examples
///
/// ```
/// use oxixml_unicode::case::simple_lowercase;
///
/// assert_eq!(simple_lowercase('A'), 'a');
/// assert_eq!(simple_lowercase('É'), 'é');
/// assert_eq!(simple_lowercase('5'), '5');
/// ```
#[must_use]
pub fn simple_lowercase(c: char) -> char {
    map_single(SIMPLE_LOWERCASE, c)
}

/// Returns the simple titlecase mapping of `c`, or `c` itself if it has none.
///
/// This is a one-to-one mapping (`UnicodeData.txt` field 14); it differs from
/// [`simple_uppercase`] only for the handful of digraph characters such as
/// `ǳ` (`U+01F3`).
///
/// # Examples
///
/// ```
/// use oxixml_unicode::case::simple_titlecase;
///
/// assert_eq!(simple_titlecase('a'), 'A');
/// assert_eq!(simple_titlecase('\u{01F3}'), '\u{01F2}'); // ǳ -> ǲ
/// ```
#[must_use]
pub fn simple_titlecase(c: char) -> char {
    map_single(SIMPLE_TITLECASE, c)
}

/// Returns the simple case fold of `c`, or `c` itself if it has none.
///
/// Simple folding (`CaseFolding.txt` statuses `C` and `S`) is one-to-one and is
/// used for caseless comparison when preserving string length matters.
///
/// # Examples
///
/// ```
/// use oxixml_unicode::case::simple_fold;
///
/// assert_eq!(simple_fold('A'), 'a');
/// assert_eq!(simple_fold('Σ'), 'σ');
/// assert_eq!(simple_fold('ẞ'), 'ß'); // capital sharp s -> sharp s (simple)
/// assert_eq!(simple_fold('5'), '5');
/// ```
#[must_use]
pub fn simple_fold(c: char) -> char {
    map_single(SIMPLE_FOLD, c)
}

/// Returns the full case fold of a single character as a slice of characters,
/// or `None` if the character folds to itself.
fn full_fold_char(c: char) -> Option<&'static [char]> {
    match FULL_FOLD.binary_search_by(|&(from, _)| from.cmp(&c)) {
        Ok(i) => Some(FULL_FOLD[i].1),
        Err(_) => None,
    }
}

/// Returns the full case fold of `s`, for locale-independent caseless string
/// comparison.
///
/// Full folding (`CaseFolding.txt` statuses `C` and `F`) may lengthen the
/// string: `ß` folds to `ss`, and `ﬁ` (`U+FB01`) folds to `fi`. Two strings are
/// *caselessly equal* (the default, non-canonical form) when their
/// [`fold_str`] results are equal.
///
/// # Examples
///
/// ```
/// use oxixml_unicode::case::fold_str;
///
/// assert_eq!(fold_str("HELLO"), "hello");
/// assert_eq!(fold_str("Straße"), "strasse");
/// assert_eq!(fold_str("ﬁle"), "file"); // U+FB01 ligature
/// assert_eq!(fold_str("HELLO"), fold_str("hello"));
/// ```
#[must_use]
pub fn fold_str(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    for c in s.chars() {
        match full_fold_char(c) {
            Some(mapping) => out.extend(mapping.iter().copied()),
            None => out.push(c),
        }
    }
    out
}

// ---------------------------------------------------------------------------
// Full case mapping (`SpecialCasing.txt`)
// ---------------------------------------------------------------------------

/// Searches a sorted, non-overlapping `(start, end)` range table for `c`.
fn in_ranges(ranges: &[(u32, u32)], c: char) -> bool {
    let cp = c as u32;
    let mut lo = 0usize;
    let mut hi = ranges.len();
    while lo < hi {
        let mid = lo + (hi - lo) / 2;
        let (start, end) = ranges[mid];
        if cp < start {
            hi = mid;
        } else if cp > end {
            lo = mid + 1;
        } else {
            return true;
        }
    }
    false
}

/// Returns `true` if `c` has the Unicode `Cased` derived core property: it
/// participates in a case pair (general category `Lu`, `Ll`, or `Lt`, or one
/// of a small set of additional letters the UCD marks `Other_Uppercase` or
/// `Other_Lowercase`).
fn is_cased(c: char) -> bool {
    in_ranges(CASED_RANGES, c)
}

/// Returns `true` if `c` has the Unicode `Case_Ignorable` derived core
/// property: combining marks, modifier letters and symbols, and a handful of
/// punctuation and format characters (apostrophe, period, colon, the curly
/// apostrophe, and similar) that a case-context scan skips over without
/// treating them as either cased or a hard stop.
fn is_case_ignorable(c: char) -> bool {
    in_ranges(CASE_IGNORABLE_RANGES, c)
}

/// Returns `true` if `c` has the Unicode `Soft_Dotted` property: its
/// lowercase glyph carries a small dot that disappears when a further accent
/// is placed above it (Latin `i`/`j`, Cyrillic і/ј, and similar letters in
/// other scripts).
fn is_soft_dotted(c: char) -> bool {
    in_ranges(SOFT_DOTTED_RANGES, c)
}

/// Classification of a character for the `SpecialCasing.txt` dot-related
/// context tests (`After_Soft_Dotted`, `After_I`, `More_Above`,
/// `Before_Dot`). A starter (canonical combining class 0) is either
/// [`DotType::SoftDotted`] or [`DotType::NoDot`]; a combining mark of class
/// 230 (`Above`) is [`DotType::Above`]; any other combining mark is
/// [`DotType::OtherAccent`] and does not interrupt a context scan.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum DotType {
    NoDot,
    SoftDotted,
    Above,
    OtherAccent,
}

/// Returns the [`DotType`] of `c`, used by the four dot-related context
/// scans below.
fn dot_type(c: char) -> DotType {
    match ccc(c) {
        230 => DotType::Above,
        0 => {
            if is_soft_dotted(c) {
                DotType::SoftDotted
            } else {
                DotType::NoDot
            }
        }
        _ => DotType::OtherAccent,
    }
}

/// The `Final_Sigma` "Before C"/"After C" cased-letter scan: starting just
/// before (`step = -1`) or just after (`step = 1`) `chars[i]`, skips
/// case-ignorable characters and reports whether a cased letter is reached
/// before a non-cased, non-case-ignorable character or the end of `chars`.
fn scan_for_cased_letter(chars: &[char], i: usize, step: isize) -> bool {
    let mut j = i as isize;
    loop {
        j += step;
        if j < 0 || j as usize >= chars.len() {
            return false;
        }
        let c = chars[j as usize];
        if is_case_ignorable(c) {
            continue;
        }
        return is_cased(c);
    }
}

/// `Final_Sigma`: `chars[i]` (a Greek capital sigma) is preceded by a cased
/// letter (skipping any case-ignorable characters in between) and is not
/// followed by one.
fn final_sigma(chars: &[char], i: usize) -> bool {
    scan_for_cased_letter(chars, i, -1) && !scan_for_cased_letter(chars, i, 1)
}

/// `After_Soft_Dotted`: the nearest character before `chars[i]` that is not
/// [`DotType::OtherAccent`] is [`DotType::SoftDotted`].
fn preceded_by_soft_dotted(chars: &[char], i: usize) -> bool {
    let mut j = i;
    while j > 0 {
        j -= 1;
        match dot_type(chars[j]) {
            DotType::SoftDotted => return true,
            DotType::OtherAccent => continue,
            DotType::NoDot | DotType::Above => return false,
        }
    }
    false
}

/// `After_I`: the nearest character before `chars[i]` that is not
/// [`DotType::OtherAccent`] is a capital `I` (`U+0049`).
fn preceded_by_capital_i(chars: &[char], i: usize) -> bool {
    let mut j = i;
    while j > 0 {
        j -= 1;
        if chars[j] == '\u{49}' {
            return true;
        }
        if dot_type(chars[j]) != DotType::OtherAccent {
            return false;
        }
    }
    false
}

/// `More_Above`: scanning forward from `chars[i]`, a [`DotType::Above`]
/// character (combining class 230) is reached before a new base character
/// or the end of `chars`.
fn followed_by_more_above(chars: &[char], i: usize) -> bool {
    let mut j = i;
    while j + 1 < chars.len() {
        j += 1;
        match dot_type(chars[j]) {
            DotType::Above => return true,
            DotType::OtherAccent => continue,
            DotType::NoDot | DotType::SoftDotted => return false,
        }
    }
    false
}

/// `Before_Dot`: scanning forward from `chars[i]`, a combining dot above
/// (`U+0307`) is reached before a new base character, a combining-class-230
/// mark, or the end of `chars`.
fn followed_by_dot_above(chars: &[char], i: usize) -> bool {
    let mut j = i;
    while j + 1 < chars.len() {
        j += 1;
        if chars[j] == '\u{307}' {
            return true;
        }
        if dot_type(chars[j]) != DotType::OtherAccent {
            return false;
        }
    }
    false
}

/// Looks up the unconditional `SpecialCasing.txt` row for `c`, if any,
/// returning its `(lower, title, upper)` mappings.
fn lookup_special_case(c: char) -> Option<(&'static [char], &'static [char], &'static [char])> {
    match SPECIAL_CASE.binary_search_by(|&(from, ..)| from.cmp(&c)) {
        Ok(i) => {
            let (_, lower, title, upper) = SPECIAL_CASE[i];
            Some((lower, title, upper))
        }
        Err(_) => None,
    }
}

/// Language tailoring for the full case mappings of `SpecialCasing.txt`.
///
/// [`CaseLocale::Root`] applies only the mappings and context tests that
/// hold regardless of language: the unconditional multi-character mappings
/// and the Greek `Final_Sigma` test. The other variants additionally apply
/// the tailoring `SpecialCasing.txt` defines for that language, layered on
/// top of the root behavior.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
#[non_exhaustive]
pub enum CaseLocale {
    /// The locale-independent (root) case mappings.
    #[default]
    Root,
    /// Lithuanian (`lt`): a lowercase `i`/`j`/`į` keeps its dot as an
    /// explicit combining dot above when a further accent is added above
    /// it, and a redundant combining dot above is dropped when
    /// uppercasing or titlecasing a soft-dotted letter.
    Lithuanian,
    /// Turkish (`tr`): `I`/`ı` and `İ`/`i` are the two independent case
    /// pairs (the dotless and dotted forms of `i`), rather than `I`/`i`.
    Turkish,
    /// Azerbaijani (`az`): the same case tailoring as [`CaseLocale::Turkish`].
    Azeri,
}

impl CaseLocale {
    /// Returns `true` for the Turkish and Azerbaijani tailorings, which
    /// share identical casing rules.
    fn is_turkic(self) -> bool {
        matches!(self, CaseLocale::Turkish | CaseLocale::Azeri)
    }
}

/// Writes the full lowercase mapping of `chars[i]` to `out`, tailored for
/// `locale`. Mirrors the reference case-mapping algorithm's dispatch order:
/// language-specific conditional rules are checked first, then the
/// language-independent conditional rules, then the unconditional
/// `SpecialCasing.txt` table, and finally the simple mapping.
fn write_full_lower(out: &mut String, chars: &[char], i: usize, locale: CaseLocale) {
    let c = chars[i];
    if locale == CaseLocale::Lithuanian {
        match c {
            '\u{49}' if followed_by_more_above(chars, i) => {
                out.push('\u{69}');
                out.push('\u{307}');
                return;
            }
            '\u{4A}' if followed_by_more_above(chars, i) => {
                out.push('\u{6A}');
                out.push('\u{307}');
                return;
            }
            '\u{12E}' if followed_by_more_above(chars, i) => {
                out.push('\u{12F}');
                out.push('\u{307}');
                return;
            }
            '\u{CC}' => {
                out.push('\u{69}');
                out.push('\u{307}');
                out.push('\u{300}');
                return;
            }
            '\u{CD}' => {
                out.push('\u{69}');
                out.push('\u{307}');
                out.push('\u{301}');
                return;
            }
            '\u{128}' => {
                out.push('\u{69}');
                out.push('\u{307}');
                out.push('\u{303}');
                return;
            }
            _ => {}
        }
    }
    if locale.is_turkic() {
        match c {
            '\u{130}' => {
                out.push('\u{69}');
                return;
            }
            '\u{307}' if preceded_by_capital_i(chars, i) => return, // dot above is deleted
            '\u{49}' if !followed_by_dot_above(chars, i) => {
                out.push('\u{131}');
                return;
            }
            _ => {}
        }
    }
    if c == '\u{130}' {
        // Preserve canonical equivalence for I with dot (root behavior;
        // the Turkic override above takes precedence when it applies).
        out.push('\u{69}');
        out.push('\u{307}');
        return;
    }
    if c == '\u{3A3}' && final_sigma(chars, i) {
        out.push('\u{3C2}');
        return;
    }
    if let Some((lower, _, _)) = lookup_special_case(c) {
        out.extend(lower.iter().copied());
        return;
    }
    out.push(simple_lowercase(c));
}

/// Writes the full uppercase (`title = false`) or titlecase (`title =
/// true`) mapping of `chars[i]` to `out`, tailored for `locale`.
fn write_full_upper_or_title(
    out: &mut String,
    chars: &[char],
    i: usize,
    locale: CaseLocale,
    title: bool,
) {
    let c = chars[i];
    if locale.is_turkic() && c == '\u{69}' {
        out.push('\u{130}');
        return;
    }
    if locale == CaseLocale::Lithuanian && c == '\u{307}' && preceded_by_soft_dotted(chars, i) {
        return; // dot above is deleted
    }
    if let Some((_, title_map, upper_map)) = lookup_special_case(c) {
        let mapping = if title { title_map } else { upper_map };
        out.extend(mapping.iter().copied());
        return;
    }
    out.push(if title {
        simple_titlecase(c)
    } else {
        simple_uppercase(c)
    });
}

/// Returns the full lowercase mapping of `s` (root locale): the
/// unconditional multi-character mappings of `SpecialCasing.txt` plus the
/// `Final_Sigma` context test, with no language tailoring.
///
/// Unlike [`simple_lowercase`], this may change the length of the string
/// (`İ` expands to `i` followed by a combining dot above), and the mapping
/// of a Greek capital sigma depends on whether it is the last cased
/// character of its word.
///
/// # Examples
///
/// ```
/// use oxixml_unicode::case::full_lowercase;
///
/// assert_eq!(full_lowercase("STRASSE"), "strasse");
/// assert_eq!(full_lowercase("ΟΔΥΣΣΕΥΣ"), "οδυσσευς"); // final sigma -> ς
/// assert_eq!(full_lowercase("ΣΟΦΙΑ"), "σοφια");       // initial sigma -> σ
/// ```
#[must_use]
pub fn full_lowercase(s: &str) -> String {
    full_lowercase_locale(s, CaseLocale::Root)
}

/// Returns the full lowercase mapping of `s`, tailored for `locale`.
///
/// # Examples
///
/// ```
/// use oxixml_unicode::case::{full_lowercase, full_lowercase_locale, CaseLocale};
///
/// // Turkish: I-with-dot-above lowercases to plain i, and a bare capital I
/// // (not before a combining dot above) becomes dotless ı instead of i.
/// assert_eq!(full_lowercase_locale("\u{130}", CaseLocale::Turkish), "i");
/// assert_eq!(full_lowercase_locale("I", CaseLocale::Turkish), "\u{131}");
///
/// // Lithuanian: an accent above a capital I keeps an explicit dot above.
/// assert_eq!(
///     full_lowercase_locale("I\u{301}", CaseLocale::Lithuanian),
///     "i\u{307}\u{301}"
/// );
/// assert_eq!(full_lowercase("I\u{301}"), "i\u{301}"); // root: no explicit dot
/// ```
#[must_use]
pub fn full_lowercase_locale(s: &str, locale: CaseLocale) -> String {
    let chars: Vec<char> = s.chars().collect();
    let mut out = String::with_capacity(s.len());
    for i in 0..chars.len() {
        write_full_lower(&mut out, &chars, i, locale);
    }
    out
}

/// Returns the full uppercase mapping of `s` (root locale): the
/// unconditional multi-character mappings of `SpecialCasing.txt`, with no
/// language tailoring.
///
/// Unlike [`simple_uppercase`], this may change the length of the string:
/// `ß` uppercases to `SS`, and the ligatures and Greek letters with
/// ypogegrammeni expand to their spelled-out uppercase forms.
///
/// Uppercasing a precomposed Greek vowel followed by a combining
/// ypogegrammeni (iota subscript, `U+0345`) and further accents gives the
/// correct result only when the input is already in Normalization Form C;
/// see [`crate::norm::nfc`].
///
/// # Examples
///
/// ```
/// use oxixml_unicode::case::full_uppercase;
///
/// assert_eq!(full_uppercase("straße"), "STRASSE");
/// assert_eq!(full_uppercase("\u{FB01}le"), "FILE"); // ﬁ ligature
/// ```
#[must_use]
pub fn full_uppercase(s: &str) -> String {
    full_uppercase_locale(s, CaseLocale::Root)
}

/// Returns the full uppercase mapping of `s`, tailored for `locale`.
///
/// # Examples
///
/// ```
/// use oxixml_unicode::case::{full_uppercase, full_uppercase_locale, CaseLocale};
///
/// // Turkish: lowercase i uppercases to dotted capital I, not plain I.
/// assert_eq!(full_uppercase_locale("i", CaseLocale::Turkish), "\u{130}");
/// assert_eq!(full_uppercase("i"), "I"); // root: plain I
/// ```
#[must_use]
pub fn full_uppercase_locale(s: &str, locale: CaseLocale) -> String {
    let chars: Vec<char> = s.chars().collect();
    let mut out = String::with_capacity(s.len());
    for i in 0..chars.len() {
        write_full_upper_or_title(&mut out, &chars, i, locale, false);
    }
    out
}

/// Returns the full titlecase mapping of `s` (root locale).
///
/// Titlecasing has no single Unicode-defined notion of "word": this follows
/// the run-based rule used by the reference case operations — the first
/// cased character after a run of uncased characters gets the titlecase
/// mapping, and every subsequent character up to the next uncased one gets
/// the lowercase mapping (so a run's second and later cased characters are
/// lowercased even if they started out uppercase). It is not full UAX #29
/// word segmentation.
///
/// # Examples
///
/// ```
/// use oxixml_unicode::case::full_titlecase;
///
/// assert_eq!(full_titlecase("hello world"), "Hello World");
/// assert_eq!(full_titlecase("STRASSE"), "Strasse");
/// ```
#[must_use]
pub fn full_titlecase(s: &str) -> String {
    full_titlecase_locale(s, CaseLocale::Root)
}

/// Returns the full titlecase mapping of `s`, tailored for `locale`.
///
/// # Examples
///
/// ```
/// use oxixml_unicode::case::{full_titlecase_locale, CaseLocale};
///
/// assert_eq!(full_titlecase_locale("i", CaseLocale::Turkish), "\u{130}");
/// ```
#[must_use]
pub fn full_titlecase_locale(s: &str, locale: CaseLocale) -> String {
    let chars: Vec<char> = s.chars().collect();
    let mut out = String::with_capacity(s.len());
    let mut previous_is_cased = false;
    for (i, &c) in chars.iter().enumerate() {
        if previous_is_cased {
            write_full_lower(&mut out, &chars, i, locale);
        } else {
            write_full_upper_or_title(&mut out, &chars, i, locale, true);
        }
        previous_is_cased = is_cased(c);
    }
    out
}
