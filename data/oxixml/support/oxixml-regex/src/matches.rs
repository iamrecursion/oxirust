//! The public per-match, per-capturing-group API: [`Regex::find_iter`],
//! [`Matches`], and [`Match`].
//!
//! This is the primitive [`fn:analyze-string`][as] (F&O 3.1 §5.6.5) needs and
//! that [`Regex::is_match`], [`Regex::replace`], and [`Regex::tokenize`] do
//! not expose: the position of the whole match *and* of every capturing
//! group, for every non-overlapping match in the input, not just whether (or
//! how) the input matches.
//!
//! [as]: https://www.w3.org/TR/xpath-functions-31/#func-analyze-string

use std::iter::FusedIterator;
use std::ops::Range;

use crate::{RawMatch, Regex, RegexError};

/// Computes the byte offset of every `char` boundary in `input`, so that a
/// `char`-index position from the engine (an index into
/// `input.chars().collect::<Vec<_>>()`) can be translated to the byte offset
/// [`Match`] reports. `result[i]` is the byte offset of the `i`-th `char`;
/// `result` has `input.chars().count() + 1` entries, the last being
/// `input.len()` (one past the final character, i.e. the position "after the
/// whole string" that a match ending at the end of input reports).
fn char_byte_offsets(input: &str) -> Vec<usize> {
    let mut offsets: Vec<usize> = input.char_indices().map(|(byte, _)| byte).collect();
    offsets.push(input.len());
    offsets
}

/// A single non-overlapping match, produced by [`Regex::find_iter`] or
/// [`Regex::find`].
///
/// All positions ([`Match::range`], [`Match::start`], [`Match::end`], and the
/// group accessors) are **byte offsets** into the original `&str` passed to
/// [`Regex::find_iter`], so `&input[m.range()]` (equivalently
/// [`Match::as_str`]) always slices it correctly. The engine matches by
/// Unicode scalar value (`char`) internally, and every reported offset falls
/// on a `char` boundary, so slicing can never panic on a UTF-8 boundary
/// violation.
///
/// # Zero-length matches
///
/// A pattern that can match the empty string (for example `a*` or `x?`) can
/// produce a zero-length [`Match`] (`m.start() == m.end()`). When
/// [`Regex::find_iter`] yields one, the scan for the *next* match still
/// resumes one `char` further on, never at the same position — otherwise the
/// same zero-length match would repeat forever. This is the rule
/// [`fn:analyze-string`][as] §5.6.5 and [`fn:tokenize`][tok] §5.6.3 both use.
/// Unlike [`Regex::replace`] and [`Regex::tokenize`], `find_iter` does *not*
/// reject a pattern that can match the empty string — `FORX0003` is specific
/// to those two functions.
///
/// [as]: https://www.w3.org/TR/xpath-functions-31/#func-analyze-string
/// [tok]: https://www.w3.org/TR/xpath-functions-31/#func-tokenize
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Match<'t> {
    text: &'t str,
    range: Range<usize>,
    /// Byte range of capturing group `i` (1-based) at index `i - 1`; `None`
    /// when group `i` did not participate in this match.
    groups: Vec<Option<Range<usize>>>,
}

impl<'t> Match<'t> {
    /// Builds a [`Match`] from already-resolved byte positions.
    pub(crate) fn new(
        text: &'t str,
        range: Range<usize>,
        groups: Vec<Option<Range<usize>>>,
    ) -> Match<'t> {
        Match {
            text,
            range,
            groups,
        }
    }

    /// The byte range of the whole match within the original input.
    #[must_use]
    pub fn range(&self) -> Range<usize> {
        self.range.clone()
    }

    /// The byte offset of the start of the match.
    #[must_use]
    pub fn start(&self) -> usize {
        self.range.start
    }

    /// The byte offset of the end of the match (exclusive).
    #[must_use]
    pub fn end(&self) -> usize {
        self.range.end
    }

    /// The text of the whole match.
    ///
    /// # Examples
    ///
    /// ```
    /// use oxixml_regex::Regex;
    ///
    /// let re = Regex::xpath("[0-9]+", "")?;
    /// let m = re.find("abc123def")?.expect("should match");
    /// assert_eq!(m.as_str(), "123");
    /// # Ok::<(), oxixml_regex::RegexError>(())
    /// ```
    #[must_use]
    pub fn as_str(&self) -> &'t str {
        &self.text[self.range.clone()]
    }

    /// Whether the match is zero-length (`start() == end()`). See the
    /// [`Match`] documentation for how iteration advances past one of these.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.range.start == self.range.end
    }

    /// The number of capturing groups in the pattern that produced this
    /// match. Group indices `1..=group_count()` are valid arguments to
    /// [`Match::group`]; this excludes group `0`, the whole match, which is
    /// always present and is what [`Match::range`] already reports.
    #[must_use]
    pub fn group_count(&self) -> usize {
        self.groups.len()
    }

    /// The byte range of capturing group `index`, or `None` if `index` is
    /// `0` or out of range, or if the group did not participate in this
    /// match (for example, a branch of an alternation that was not taken, or
    /// an optional group that matched zero times).
    ///
    /// Note that `index` counts capturing groups only — a `(?:...)`
    /// non-capturing group is never assigned a number and can never be
    /// queried this way.
    ///
    /// # Examples
    ///
    /// ```
    /// use oxixml_regex::Regex;
    ///
    /// let re = Regex::xpath("(a+)(b+)?", "")?;
    /// let m = re.find("aaa")?.expect("should match");
    /// assert_eq!(m.group(1), Some(0..3));
    /// assert_eq!(m.group(2), None); // group 2 did not participate
    /// assert_eq!(m.group(0), None); // 0 is not a capturing-group index
    /// # Ok::<(), oxixml_regex::RegexError>(())
    /// ```
    #[must_use]
    pub fn group(&self, index: usize) -> Option<Range<usize>> {
        if index == 0 {
            return None;
        }
        self.groups.get(index - 1)?.clone()
    }

    /// The text of capturing group `index`; see [`Match::group`].
    #[must_use]
    pub fn group_str(&self, index: usize) -> Option<&'t str> {
        self.group(index).map(|r| &self.text[r])
    }

    /// An iterator over the byte ranges of every capturing group, in order
    /// from `1` to [`Match::group_count`] (`None` for a group that did not
    /// participate). Group `0`, the whole match, is not included — use
    /// [`Match::range`] for that.
    pub fn groups(&self) -> impl ExactSizeIterator<Item = Option<Range<usize>>> + '_ {
        self.groups.iter().cloned()
    }
}

/// An iterator over successive non-overlapping matches, created by
/// [`Regex::find_iter`].
///
/// Yields `Result<Match, RegexError>` rather than `Match` because evaluating
/// a back-reference pattern can exhaust its step budget partway through the
/// input; the only error [`Matches`] can yield is
/// [`RegexError::BudgetExceeded`], after which iteration stops (a later call
/// to `next` returns `None`).
///
/// Each call to [`Matches::next`] gets a fresh budget for the one match it
/// looks for — the same allowance a standalone call to
/// [`Regex::find`](crate::Regex::find) gets, since `find` is defined as
/// `find_iter(input).next()`. That budget is shared across every starting
/// position tried *within* that one search (see `crate::backtrack::find_from`),
/// which is what stops a single catastrophic (or merely expensive-at-many-
/// positions) search for one match from running unbounded; it does not carry
/// over to the *next* match `Matches` looks for. A caller iterating over a
/// huge document expects to keep finding matches for as long as the document
/// has more to give — that is normal, successful operation, not the runaway
/// back-tracking this budget exists to catch, so resetting it per match
/// avoids manufacturing a false [`RegexError::BudgetExceeded`] out of
/// ordinary long iteration. [`Regex::replace`](crate::Regex::replace) and
/// [`Regex::tokenize`](crate::Regex::tokenize) make the opposite choice
/// deliberately: each is one call with a single well-defined result, so they
/// share one budget across every match they find internally.
#[derive(Debug, Clone)]
pub struct Matches<'r, 't> {
    regex: &'r Regex,
    text: &'t str,
    chars: Vec<char>,
    byte_offsets: Vec<usize>,
    /// The next `char` index to resume searching from.
    pos: usize,
    done: bool,
}

impl<'r, 't> Matches<'r, 't> {
    /// Starts a fresh search of `text` with `regex`. Not public: reached only
    /// through [`Regex::find_iter`], which owns the choice of how the
    /// `char`/byte bookkeeping is set up.
    pub(crate) fn new(regex: &'r Regex, text: &'t str) -> Matches<'r, 't> {
        Matches {
            regex,
            text,
            chars: text.chars().collect(),
            byte_offsets: char_byte_offsets(text),
            pos: 0,
            done: false,
        }
    }

    /// Converts a `RawMatch` (engine-native `char` indices) into a public
    /// [`Match`] (byte offsets), and decides where the next search should
    /// resume.
    fn convert_and_advance(&mut self, raw: RawMatch) -> Match<'t> {
        let group_count = self.regex.group_count();
        let mut groups = Vec::with_capacity(group_count);
        for i in 1..=group_count {
            let start = raw.caps.get(2 * i).copied().flatten();
            let end = raw.caps.get(2 * i + 1).copied().flatten();
            groups.push(match (start, end) {
                (Some(s), Some(e)) => Some(self.byte_offsets[s]..self.byte_offsets[e]),
                _ => None,
            });
        }
        let byte_range = self.byte_offsets[raw.start]..self.byte_offsets[raw.end];

        if raw.end == raw.start {
            // Zero-length match: advance by exactly one `char`, or stop if
            // there is no next `char` to advance to (a zero-length match
            // right at the end of the input, e.g. `a*` against "xa"). See
            // the "Zero-length matches" section of [`Match`]'s docs.
            if raw.end >= self.chars.len() {
                self.done = true;
            } else {
                self.pos = raw.end + 1;
            }
        } else {
            self.pos = raw.end;
        }

        Match::new(self.text, byte_range, groups)
    }
}

impl<'r, 't> Iterator for Matches<'r, 't> {
    type Item = Result<Match<'t>, RegexError>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.done {
            return None;
        }
        // A fresh budget per match found, not one shared across the iterator's whole
        // lifetime; see the type-level documentation above for why.
        let mut budget = crate::backtrack::DEFAULT_BUDGET;
        match self.regex.find_from(&self.chars, self.pos, &mut budget) {
            Ok(Some(raw)) => Some(Ok(self.convert_and_advance(raw))),
            Ok(None) => {
                self.done = true;
                None
            }
            Err(err) => {
                self.done = true;
                Some(Err(err))
            }
        }
    }
}

impl<'r, 't> FusedIterator for Matches<'r, 't> {}
