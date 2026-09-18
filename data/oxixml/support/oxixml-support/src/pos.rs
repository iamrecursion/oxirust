//! Line / column / offset bookkeeping for text and byte streams.
//!
//! [`TextPosition`] is a plain triple of `line`, `column` and byte `offset`. [`LineColTracker`]
//! consumes bytes incrementally and keeps a running position, correctly handling the three
//! historical line-break conventions found in XML and RDF documents:
//!
//! * `LF` (`\n`, Unix),
//! * `CRLF` (`\r\n`, Windows), counted as a **single** line break, and
//! * lone `CR` (`\r`, classic Mac).
//!
//! Columns are counted in Unicode scalar values rather than bytes: a UTF-8 continuation byte
//! (`0b10xx_xxxx`) does not advance the column, so a multi-byte character occupies exactly one
//! column.
//!
//! # Numbering base
//!
//! Line, column and byte offset are all **0-based** in storage: the first character of a document
//! is line `0`, column `0`, offset `0`. This matches the convention of the upstream parsers this
//! bookkeeping replaces (`json-event-parser`, `oxttl`, `sparesults`) and of
//! [`oxixml_turtle`'s local `TextPosition`](https://docs.rs/oxixml-turtle). The conventional
//! **1-based** presentation is applied only when rendering: the [`Display`](fmt::Display) impl of
//! [`TextPosition`] adds one to line and column, so the very first character prints as
//! `line 1, column 1`.

use core::fmt;

/// Carriage return.
const CR: u8 = b'\r';
/// Line feed.
const LF: u8 = b'\n';

/// A position within a text: a 0-based `line`, a 0-based `column` (in Unicode scalar values)
/// and a 0-based byte `offset`.
///
/// All three fields are **0-based**, so the very first character of a document is
/// `line 0, column 0, offset 0`. Only the [`Display`](fmt::Display) impl switches to the
/// human-facing 1-based convention, printing that same position as `line 1, column 1`. Code
/// reading [`line`](Self::line) / [`column`](Self::column) programmatically therefore never needs
/// to subtract one, and code formatting them for humans must not add one a second time.
///
/// The default position is the very start of a document: line 0, column 0, offset 0.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct TextPosition {
    /// 0-based line number. Incremented on every `LF`, `CRLF` or lone `CR`.
    pub line: u64,
    /// 0-based column number, counted in Unicode scalar values (not bytes).
    pub column: u64,
    /// 0-based byte offset from the start of the stream.
    pub offset: u64,
}

impl TextPosition {
    /// The position at the very start of a document: line 0, column 0, offset 0.
    #[inline]
    pub const fn start() -> Self {
        Self {
            line: 0,
            column: 0,
            offset: 0,
        }
    }

    /// Construct a position from explicit 0-based `line`, `column` and `offset` values.
    #[inline]
    pub const fn new(line: u64, column: u64, offset: u64) -> Self {
        Self {
            line,
            column,
            offset,
        }
    }
}

impl Default for TextPosition {
    #[inline]
    fn default() -> Self {
        Self::start()
    }
}

impl fmt::Display for TextPosition {
    /// Renders the position using the human-facing **1-based** convention, adding one to the
    /// 0-based [`line`](Self::line) and [`column`](Self::column) fields.
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "line {}, column {}", self.line + 1, self.column + 1)
    }
}

/// Incrementally tracks a [`TextPosition`] as bytes are consumed.
///
/// Like [`TextPosition`], the tracked line and column are **0-based**.
///
/// Feed the stream to [`LineColTracker::advance`] in arbitrary chunks — the tracker remembers
/// enough state (in particular, whether the previous chunk ended on a `CR`) that a `CRLF`
/// sequence split across two calls is still counted as a single line break.
///
/// # Examples
///
/// ```
/// use oxixml_support::pos::LineColTracker;
///
/// let mut tracker = LineColTracker::new();
/// tracker.advance("αβ\nx".as_bytes()); // two 2-byte chars, a newline, then `x`
/// let position = tracker.position();
/// assert_eq!(position.line, 1); // 0-based: the second line
/// assert_eq!(position.column, 1); // 0-based: after consuming `x`
/// assert_eq!(position.offset, 6); // 2 + 2 + 1 + 1 bytes
/// assert_eq!(position.to_string(), "line 2, column 2"); // Display is 1-based
/// ```
#[derive(Debug, Clone)]
pub struct LineColTracker {
    line: u64,
    column: u64,
    offset: u64,
    /// `true` when the last byte consumed was a `CR` that may still be joined with a following
    /// `LF` into a single `CRLF` line break.
    pending_cr: bool,
}

impl LineColTracker {
    /// Create a tracker positioned at the start of a document (line 0, column 0, offset 0).
    #[inline]
    pub const fn new() -> Self {
        Self {
            line: 0,
            column: 0,
            offset: 0,
            pending_cr: false,
        }
    }

    /// Advance the tracker over `bytes`, updating line, column and offset.
    ///
    /// The bytes are interpreted as UTF-8 for the purpose of column counting, but invalid or
    /// partial UTF-8 will not cause a panic — continuation bytes simply do not advance the
    /// column. Line breaks are recognised for `LF`, `CRLF` (as one break) and lone `CR`.
    pub fn advance(&mut self, bytes: &[u8]) {
        for &byte in bytes {
            self.offset += 1;
            if self.pending_cr {
                self.pending_cr = false;
                if byte == LF {
                    // Second half of a `CRLF`: the `CR` already advanced the line.
                    continue;
                }
                // Otherwise the preceding `CR` was a lone line break (already applied) and this
                // byte is the first character of the new line; fall through to handle it.
            }
            match byte {
                CR => {
                    self.line += 1;
                    self.column = 0;
                    self.pending_cr = true;
                }
                LF => {
                    self.line += 1;
                    self.column = 0;
                }
                // Advance the column only for non-continuation bytes so that a multi-byte
                // UTF-8 character counts as a single column.
                _ if (byte & 0xC0) != 0x80 => {
                    self.column += 1;
                }
                _ => {}
            }
        }
    }

    /// The current position.
    #[inline]
    pub const fn position(&self) -> TextPosition {
        TextPosition {
            line: self.line,
            column: self.column,
            offset: self.offset,
        }
    }

    /// The current 0-based line number.
    #[inline]
    pub const fn line(&self) -> u64 {
        self.line
    }

    /// The current 0-based column number (in Unicode scalar values).
    #[inline]
    pub const fn column(&self) -> u64 {
        self.column
    }

    /// The current 0-based byte offset.
    #[inline]
    pub const fn offset(&self) -> u64 {
        self.offset
    }
}

impl Default for LineColTracker {
    #[inline]
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn track(chunks: &[&[u8]]) -> TextPosition {
        let mut tracker = LineColTracker::new();
        for chunk in chunks {
            tracker.advance(chunk);
        }
        tracker.position()
    }

    #[test]
    fn start_position() {
        let tracker = LineColTracker::new();
        assert_eq!(tracker.position(), TextPosition::start());
        assert_eq!(TextPosition::default(), TextPosition::new(0, 0, 0));
    }

    #[test]
    fn plain_line_feeds() {
        let position = track(&[b"abc\ndef"]);
        assert_eq!(position, TextPosition::new(1, 3, 7));
    }

    #[test]
    fn carriage_return_line_feed_is_one_break() {
        let position = track(&[b"abc\r\ndef"]);
        assert_eq!(position, TextPosition::new(1, 3, 8));
    }

    #[test]
    fn lone_carriage_return() {
        let position = track(&[b"abc\rdef"]);
        assert_eq!(position, TextPosition::new(1, 3, 7));
    }

    #[test]
    fn crlf_split_across_chunks_counts_once() {
        let split = track(&[b"abc\r", b"\ndef"]);
        let together = track(&[b"abc\r\ndef"]);
        assert_eq!(split, together);
        assert_eq!(split, TextPosition::new(1, 3, 8));
    }

    #[test]
    fn cr_then_non_lf_across_chunks() {
        // A `CR` at the end of one chunk followed by a non-`LF` byte in the next is a lone
        // break, and the following byte begins line 2.
        let position = track(&[b"a\r", b"b"]);
        assert_eq!(position, TextPosition::new(1, 1, 3));
    }

    #[test]
    fn multibyte_columns() {
        // "é" (U+00E9) is two UTF-8 bytes but a single column.
        let position = track(&["é".as_bytes()]);
        assert_eq!(position, TextPosition::new(0, 1, 2));
        // A 4-byte emoji is still one column.
        let emoji = track(&["😀x".as_bytes()]);
        assert_eq!(emoji, TextPosition::new(0, 2, 5));
    }

    #[test]
    fn double_carriage_return() {
        let position = track(&[b"\r\r"]);
        assert_eq!(position, TextPosition::new(2, 0, 2));
    }

    #[test]
    fn display_format() {
        // Display switches to the 1-based convention.
        let position = TextPosition::new(12, 34, 500);
        assert_eq!(position.to_string(), "line 13, column 35");
    }

    /// Regression test for the 0-based storage / 1-based Display contract: the very first
    /// character of a document must read as line 0, column 0 programmatically while rendering as
    /// the human-facing "1:1" shape. A regression to 1-based storage would silently shift every
    /// programmatic reader of `.line` / `.column` by one.
    #[test]
    fn first_character_is_zero_based_but_displays_one_based() {
        let mut tracker = LineColTracker::new();
        let before = tracker.position();
        assert_eq!(before.line, 0);
        assert_eq!(before.column, 0);
        assert_eq!(before.offset, 0);
        assert_eq!(before.to_string(), "line 1, column 1");
        assert_eq!(TextPosition::start(), before);

        // Consuming the first character leaves the position *after* it: still line 0, column 1.
        tracker.advance(b"a");
        let after = tracker.position();
        assert_eq!(after.line, 0);
        assert_eq!(after.column, 1);
        assert_eq!(after.to_string(), "line 1, column 2");
    }

    /// The first character of every line is column 0, and the first line is line 0.
    #[test]
    fn line_starts_are_column_zero() {
        for break_bytes in [&b"\n"[..], &b"\r\n"[..], &b"\r"[..]] {
            let mut tracker = LineColTracker::new();
            tracker.advance(b"ab");
            tracker.advance(break_bytes);
            let position = tracker.position();
            assert_eq!(position.line, 1, "line for {break_bytes:?}");
            assert_eq!(position.column, 0, "column for {break_bytes:?}");
            assert_eq!(position.to_string(), "line 2, column 1");
        }
    }

    #[test]
    fn ordering_follows_offset() {
        assert!(TextPosition::new(0, 0, 0) < TextPosition::new(0, 1, 1));
        assert!(TextPosition::new(0, 8, 8) < TextPosition::new(1, 0, 9));
    }
}
