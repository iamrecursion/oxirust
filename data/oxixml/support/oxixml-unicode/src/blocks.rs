//! Unicode blocks.
//!
//! A [block][blk] is a named, contiguous range of code points. This module
//! answers two questions: which block a character belongs to
//! ([`block`]), and which range a block name denotes ([`block_range`],
//! [`in_block`]). Name matching is loose in the spirit of [UAX #44-LM3][lm3]:
//! ASCII case is ignored and spaces, hyphens, and underscores are dropped, so
//! `"Basic Latin"`, `"basic_latin"`, and `"BasicLatin"` all match. This is what
//! the XML Schema regular-expression `\p{IsBasicLatin}` block escapes require.
//!
//! [blk]: https://www.unicode.org/reports/tr44/#Blocks.txt
//! [lm3]: https://www.unicode.org/reports/tr44/#UAX44-LM3

use crate::tables::blocks::BLOCKS;

/// Returns the name of the Unicode block containing `c`, or `None` if `c` lies
/// in an unassigned gap between blocks.
///
/// The lookup is a binary search over a sorted, non-overlapping range table.
///
/// # Examples
///
/// ```
/// use oxixml_unicode::blocks::block;
///
/// assert_eq!(block('A'), Some("Basic Latin"));
/// assert_eq!(block('\u{00E9}'), Some("Latin-1 Supplement"));
/// assert_eq!(block('\u{3042}'), Some("Hiragana"));
/// ```
#[must_use]
pub fn block(c: char) -> Option<&'static str> {
    let cp = c as u32;
    let mut lo = 0usize;
    let mut hi = BLOCKS.len();
    while lo < hi {
        let mid = lo + (hi - lo) / 2;
        let (start, end, name) = BLOCKS[mid];
        if cp < start {
            hi = mid;
        } else if cp > end {
            lo = mid + 1;
        } else {
            return Some(name);
        }
    }
    None
}

/// Loose block-name comparison per [UAX #44-LM3][lm3] (lite): equal when the
/// names agree after dropping ASCII spaces, hyphens, and underscores and
/// lowercasing the remaining ASCII letters.
///
/// [lm3]: https://www.unicode.org/reports/tr44/#UAX44-LM3
fn loose_eq(a: &str, b: &str) -> bool {
    let mut ai = a.chars().filter(|c| !matches!(c, ' ' | '-' | '_'));
    let mut bi = b.chars().filter(|c| !matches!(c, ' ' | '-' | '_'));
    loop {
        match (ai.next(), bi.next()) {
            (None, None) => return true,
            (Some(x), Some(y)) if x.eq_ignore_ascii_case(&y) => {}
            _ => return false,
        }
    }
}

/// Returns the `(start, end)` code-point range of the block whose name matches
/// `name` loosely (case-, space-, hyphen-, and underscore-insensitive), or
/// `None` if no block matches.
///
/// # Examples
///
/// ```
/// use oxixml_unicode::blocks::block_range;
///
/// assert_eq!(block_range("Basic Latin"), Some(('\u{0}', '\u{7F}')));
/// assert_eq!(block_range("basiclatin"), Some(('\u{0}', '\u{7F}')));
/// assert_eq!(block_range("Latin_1_Supplement"), Some(('\u{80}', '\u{FF}')));
/// assert_eq!(block_range("No Such Block"), None);
/// ```
#[must_use]
pub fn block_range(name: &str) -> Option<(char, char)> {
    for &(start, end, block_name) in BLOCKS {
        if loose_eq(block_name, name) {
            let start = char::from_u32(start)?;
            let end = char::from_u32(end)?;
            return Some((start, end));
        }
    }
    None
}

/// Returns `true` if `c` lies in the block named `name` (matched loosely, as by
/// [`block_range`]).
///
/// This is the predicate behind an XML Schema `\p{IsBasicLatin}` block escape.
///
/// # Examples
///
/// ```
/// use oxixml_unicode::blocks::in_block;
///
/// assert!(in_block('A', "BasicLatin"));
/// assert!(in_block('\u{00E9}', "Latin-1 Supplement"));
/// assert!(!in_block('A', "Hiragana"));
/// assert!(!in_block('A', "Nonexistent Block"));
/// ```
#[must_use]
pub fn in_block(c: char, name: &str) -> bool {
    match block_range(name) {
        Some((start, end)) => start <= c && c <= end,
        None => false,
    }
}
