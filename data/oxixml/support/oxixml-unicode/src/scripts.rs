//! Unicode scripts.
//!
//! Provides the [`Script`] enumeration (one variant per script in the UCD's
//! `Scripts.txt`) and the [`script`] lookup. A character with no assigned
//! script — an unassigned code point, or one the UCD leaves as `Unknown` — maps
//! to `None`.

use crate::tables::scripts::SCRIPT_RANGES;
pub use crate::tables::scripts::Script;

/// Returns the [`Script`] property value of `c`, or `None` if `c` has no
/// assigned script (unassigned code points and those the UCD classifies as
/// `Unknown`).
///
/// The lookup is a binary search over a sorted, non-overlapping range table.
///
/// # Examples
///
/// ```
/// use oxixml_unicode::scripts::{script, Script};
///
/// assert_eq!(script('A'), Some(Script::Latin));
/// assert_eq!(script('\u{0391}'), Some(Script::Greek)); // capital alpha
/// assert_eq!(script('\u{3042}'), Some(Script::Hiragana));
/// assert_eq!(script(' '), Some(Script::Common));
/// assert_eq!(script('\u{0301}'), Some(Script::Inherited)); // combining accent
/// ```
#[must_use]
pub fn script(c: char) -> Option<Script> {
    let cp = c as u32;
    let mut lo = 0usize;
    let mut hi = SCRIPT_RANGES.len();
    while lo < hi {
        let mid = lo + (hi - lo) / 2;
        let (start, end, idx) = SCRIPT_RANGES[mid];
        if cp < start {
            hi = mid;
        } else if cp > end {
            lo = mid + 1;
        } else {
            return Script::from_index(idx);
        }
    }
    None
}
