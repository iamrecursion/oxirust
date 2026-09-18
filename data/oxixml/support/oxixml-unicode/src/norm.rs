//! Unicode normalization forms (NFC, NFD, NFKC, NFKD).
//!
//! Implements the four normalization forms of [UAX #15][uax15] over `&str`:
//!
//! * [`nfd`] / [`nfkd`] — canonical / compatibility **decomposition**,
//!   followed by canonical ordering of combining marks.
//! * [`nfc`] / [`nfkc`] — the same decomposition followed by canonical
//!   **composition**.
//!
//! Decomposition is fully recursive over the UCD decomposition mappings and the
//! algorithmic Hangul rules; canonical ordering is a stable sort of each
//! maximal run of combining marks by canonical combining class; composition
//! follows the standard recomposition algorithm, using the primary-composite
//! table (canonical two-character decompositions that are not in
//! `Full_Composition_Exclusion`) and the algorithmic Hangul `L·V·T` rules.
//!
//! [`is_nfc`] is a fast [quick check][qc]: it never allocates and returns
//! `true` only when the input is *known* to be in NFC. A `Maybe` result is
//! reported conservatively as `false`, so a `true` answer is always exact while
//! a `false` answer means "run [`nfc`] to be sure".
//!
//! [uax15]: https://www.unicode.org/reports/tr15/
//! [qc]: https://www.unicode.org/reports/tr15/#Detecting_Normalization_Forms

use crate::tables::combining::COMBINING_CLASS;
use crate::tables::composition::COMPOSITION;
use crate::tables::decomposition::DECOMPOSITION;
use crate::tables::quickcheck::{NFC_QC_MAYBE, NFC_QC_NO};

// Hangul algorithmic (de)composition constants (UAX #15, §16 / TUS §3.12).
const S_BASE: u32 = 0xAC00;
const L_BASE: u32 = 0x1100;
const V_BASE: u32 = 0x1161;
const T_BASE: u32 = 0x11A7;
const L_COUNT: u32 = 19;
const V_COUNT: u32 = 21;
const T_COUNT: u32 = 28;
const N_COUNT: u32 = V_COUNT * T_COUNT; // 588
const S_COUNT: u32 = L_COUNT * N_COUNT; // 11172

/// Returns the canonical combining class of `c` (0 for starters).
pub(crate) fn ccc(c: char) -> u8 {
    match COMBINING_CLASS.binary_search_by(|&(from, _)| from.cmp(&c)) {
        Ok(i) => COMBINING_CLASS[i].1,
        Err(_) => 0,
    }
}

/// Looks up the one-step decomposition mapping of `c`, if any.
fn decomposition(c: char) -> Option<(bool, &'static [char])> {
    match DECOMPOSITION.binary_search_by(|&(from, _, _)| from.cmp(&c)) {
        Ok(i) => {
            let (_, is_compat, mapping) = DECOMPOSITION[i];
            Some((is_compat, mapping))
        }
        Err(_) => None,
    }
}

/// Returns `true` if `c` is a precomposed Hangul syllable (`U+AC00..=U+D7A3`).
fn is_hangul_syllable(c: char) -> bool {
    let cp = c as u32;
    (S_BASE..S_BASE + S_COUNT).contains(&cp)
}

/// Algorithmically decomposes a Hangul syllable into its `L`, `V`, and optional
/// `T` jamo, pushing them onto `out`.
fn decompose_hangul(c: char, out: &mut Vec<char>) {
    let s_index = c as u32 - S_BASE;
    let l = L_BASE + s_index / N_COUNT;
    let v = V_BASE + (s_index % N_COUNT) / T_COUNT;
    let t = T_BASE + s_index % T_COUNT;
    // The arithmetic keeps every value inside the BMP jamo ranges.
    if let Some(lc) = char::from_u32(l) {
        out.push(lc);
    }
    if let Some(vc) = char::from_u32(v) {
        out.push(vc);
    }
    if t != T_BASE {
        if let Some(tc) = char::from_u32(t) {
            out.push(tc);
        }
    }
}

/// Algorithmically composes a Hangul jamo pair, if `a` and `b` form one.
fn compose_hangul(a: char, b: char) -> Option<char> {
    let a = a as u32;
    let b = b as u32;
    // L + V -> LV syllable.
    if (L_BASE..L_BASE + L_COUNT).contains(&a) && (V_BASE..V_BASE + V_COUNT).contains(&b) {
        let l_index = a - L_BASE;
        let v_index = b - V_BASE;
        let s = S_BASE + (l_index * V_COUNT + v_index) * T_COUNT;
        return char::from_u32(s);
    }
    // LV + T -> LVT syllable (only when `a` is an LV syllable, i.e. no trailing).
    if (S_BASE..S_BASE + S_COUNT).contains(&a)
        && (a - S_BASE) % T_COUNT == 0
        && (T_BASE < b && b < T_BASE + T_COUNT)
    {
        let t_index = b - T_BASE;
        return char::from_u32(a + t_index);
    }
    None
}

/// Recursively decomposes `c` into `out`. When `compat` is `false` only
/// canonical mappings are followed (NFD/NFC); when `true` compatibility
/// mappings are followed as well (NFKD/NFKC).
fn decompose_into(c: char, compat: bool, out: &mut Vec<char>) {
    if is_hangul_syllable(c) {
        decompose_hangul(c, out);
        return;
    }
    match decomposition(c) {
        Some((is_compat, mapping)) if compat || !is_compat => {
            for &m in mapping {
                decompose_into(m, compat, out);
            }
        }
        _ => out.push(c),
    }
}

/// Sorts each maximal run of combining marks (ccc != 0) into canonical order by
/// combining class, using a stable sort so equal classes keep their relative
/// order.
fn canonical_order(chars: &mut [char]) {
    let n = chars.len();
    let mut i = 0;
    while i < n {
        if ccc(chars[i]) == 0 {
            i += 1;
            continue;
        }
        let mut j = i + 1;
        while j < n && ccc(chars[j]) != 0 {
            j += 1;
        }
        chars[i..j].sort_by_key(|&c| ccc(c));
        i = j;
    }
}

/// Returns the primary composite of `(a, b)`, if one exists.
fn compose_pair(a: char, b: char) -> Option<char> {
    if let Some(s) = compose_hangul(a, b) {
        return Some(s);
    }
    match COMPOSITION.binary_search_by(|&(x, y, _)| (x, y).cmp(&(a, b))) {
        Ok(i) => Some(COMPOSITION[i].2),
        Err(_) => None,
    }
}

/// Applies canonical composition to an already decomposed and canonically
/// ordered character sequence (the standard recomposition algorithm).
fn compose(input: &[char]) -> Vec<char> {
    if input.is_empty() {
        return Vec::new();
    }
    let mut out: Vec<char> = Vec::with_capacity(input.len());
    out.push(input[0]);
    let mut starter_index = 0usize;
    let mut starter_char = input[0];
    // A leading combining mark can never be a composition starter; force its
    // class high so nothing composes onto it.
    let mut last_class = if ccc(input[0]) != 0 { 255 } else { 0 };

    for &ch in &input[1..] {
        let ch_class = ccc(ch);
        if last_class == 0 || last_class < ch_class {
            if let Some(p) = compose_pair(starter_char, ch) {
                out[starter_index] = p;
                starter_char = p;
                continue;
            }
        }
        out.push(ch);
        if ch_class == 0 {
            starter_index = out.len() - 1;
            starter_char = ch;
        }
        last_class = ch_class;
    }
    out
}

/// Decomposes `s` (canonically when `!compat`, compatibly when `compat`) and
/// applies canonical ordering, returning the resulting characters.
fn decompose_str(s: &str, compat: bool) -> Vec<char> {
    let mut chars = Vec::with_capacity(s.len());
    for c in s.chars() {
        decompose_into(c, compat, &mut chars);
    }
    canonical_order(&mut chars);
    chars
}

/// Returns the Canonical Decomposition (NFD) of `s`.
///
/// # Examples
///
/// ```
/// use oxixml_unicode::norm::nfd;
///
/// assert_eq!(nfd("\u{00C5}"), "A\u{030A}");        // Å -> A + ring above
/// assert_eq!(nfd("\u{212B}"), "A\u{030A}");        // Å (angstrom sign)
/// assert_eq!(nfd("\u{AC01}"), "\u{1100}\u{1161}\u{11A8}"); // 각 -> jamo
/// ```
#[must_use]
pub fn nfd(s: &str) -> String {
    decompose_str(s, false).into_iter().collect()
}

/// Returns the Compatibility Decomposition (NFKD) of `s`.
///
/// # Examples
///
/// ```
/// use oxixml_unicode::norm::nfkd;
///
/// assert_eq!(nfkd("\u{FB01}"), "fi");   // ﬁ ligature
/// assert_eq!(nfkd("\u{00BD}"), "1\u{2044}2"); // ½ -> 1 ⁄ 2
/// assert_eq!(nfkd("\u{2460}"), "1");    // circled digit one
/// ```
#[must_use]
pub fn nfkd(s: &str) -> String {
    decompose_str(s, true).into_iter().collect()
}

/// Returns the Canonical Composition (NFC) of `s`.
///
/// # Examples
///
/// ```
/// use oxixml_unicode::norm::nfc;
///
/// assert_eq!(nfc("A\u{030A}"), "\u{00C5}");        // A + ring above -> Å
/// assert_eq!(nfc("\u{212B}"), "\u{00C5}");         // angstrom -> Å
/// assert_eq!(nfc("\u{1100}\u{1161}\u{11A8}"), "\u{AC01}"); // jamo -> 각
/// ```
#[must_use]
pub fn nfc(s: &str) -> String {
    compose(&decompose_str(s, false)).into_iter().collect()
}

/// Returns the Compatibility Composition (NFKC) of `s`.
///
/// # Examples
///
/// ```
/// use oxixml_unicode::norm::nfkc;
///
/// assert_eq!(nfkc("\u{FB01}"), "fi");   // ﬁ ligature
/// assert_eq!(nfkc("\u{2126}"), "\u{03A9}"); // ohm sign -> greek capital omega
/// ```
#[must_use]
pub fn nfkc(s: &str) -> String {
    compose(&decompose_str(s, true)).into_iter().collect()
}

/// Searches the sorted `(start, end)` range table for `c`.
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

/// Returns `true` if `s` is already in Canonical Composition (NFC) form.
///
/// This is the [`NFC_Quick_Check`][qc] fast path: it scans the string once
/// without allocating. It combines the per-character `NFC_QC` property with a
/// check that combining marks are in canonical order. A `true` result is exact;
/// a `false` result means the string is either not NFC or cannot be decided
/// cheaply (the `Maybe` case is treated conservatively as `false`), so callers
/// that need certainty should fall back to comparing against [`nfc`].
///
/// [qc]: https://www.unicode.org/reports/tr15/#Detecting_Normalization_Forms
///
/// # Examples
///
/// ```
/// use oxixml_unicode::norm::{is_nfc, nfc};
///
/// assert!(is_nfc("\u{00C5}"));       // composed Å is NFC
/// assert!(!is_nfc("A\u{030A}"));     // decomposed A + ring is not
/// assert!(is_nfc("hello world"));
/// // Whenever is_nfc is true, the string already equals its NFC.
/// let s = "caf\u{00E9}";
/// assert!(is_nfc(s));
/// assert_eq!(nfc(s), s);
/// ```
#[must_use]
pub fn is_nfc(s: &str) -> bool {
    let mut last_class = 0u8;
    for c in s.chars() {
        let class = ccc(c);
        if last_class > class && class != 0 {
            return false; // combining marks out of canonical order
        }
        if in_ranges(NFC_QC_NO, c) || in_ranges(NFC_QC_MAYBE, c) {
            return false; // NFC_QC = No or Maybe
        }
        last_class = class;
    }
    true
}
