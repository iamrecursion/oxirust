//! [`PositionTrackingReader`], a transparent [`BufRead`] wrapper that keeps a
//! running [`TextPosition`] in step with what has actually been consumed from
//! it.

use std::io::{self, BufRead, Read};

use oxixml_support::pos::{LineColTracker, TextPosition};

/// Wraps any [`BufRead`] source, forwarding every [`Read`]/[`BufRead`] call
/// unchanged but feeding the bytes that get *consumed* (via
/// [`BufRead::consume`], the call [`oxixml_xml::reader::NsReader`] actually uses)
/// through a [`LineColTracker`].
///
/// Because [`oxixml_xml`]'s buffered reading path is built entirely on
/// [`fill_buf`](BufRead::fill_buf)/[`consume`](BufRead::consume) — it never calls
/// [`Read::read`] directly — [`position`](Self::position) always reports exactly
/// the line/column/offset of [`Reader::buffer_position`](oxixml_xml::reader::Reader::buffer_position),
/// without needing any cooperation from the reader itself.
pub(crate) struct PositionTrackingReader<R> {
    inner: R,
    tracker: LineColTracker,
}

impl<R> PositionTrackingReader<R> {
    pub(crate) const fn new(inner: R) -> Self {
        Self {
            inner,
            tracker: LineColTracker::new(),
        }
    }

    /// The position corresponding to the number of bytes consumed so far.
    pub(crate) const fn position(&self) -> TextPosition {
        self.tracker.position()
    }
}

impl<R: Read> Read for PositionTrackingReader<R> {
    fn read(&mut self, buf: &mut [u8]) -> io::Result<usize> {
        let n = self.inner.read(buf)?;
        self.tracker.advance(&buf[..n]);
        Ok(n)
    }
}

impl<R: BufRead> BufRead for PositionTrackingReader<R> {
    fn fill_buf(&mut self) -> io::Result<&[u8]> {
        self.inner.fill_buf()
    }

    fn consume(&mut self, amt: usize) {
        // `fill_buf` never discards data that a previous `fill_buf` already
        // returned without an intervening `consume`, so calling it again here
        // reproduces the same bytes (this time as an immutable borrow of
        // `self.inner` alone, disjoint from `self.tracker`) rather than reading
        // more from the underlying source.
        if let Ok(available) = self.inner.fill_buf() {
            let amt = amt.min(available.len());
            self.tracker.advance(&available[..amt]);
        }
        self.inner.consume(amt);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tracks_position_across_reads() {
        let mut reader = PositionTrackingReader::new("abc\ndef".as_bytes());
        // Drive it the same way `BufRead`-based consumers do: fill, consume, repeat.
        loop {
            let available = reader.fill_buf().expect("read");
            if available.is_empty() {
                break;
            }
            let n = available.len();
            reader.consume(n);
        }
        let position = reader.position();
        // `TextPosition` is 0-based: `d` starts line 1, and `abc` -> `def` leaves column 3.
        assert_eq!(position.line, 1);
        assert_eq!(position.column, 3);
        assert_eq!(position.offset, 7);
        assert_eq!(position.to_string(), "line 2, column 4");
    }

    /// Regression test for the 0-based storage of [`TextPosition`]: reading `.line` / `.column`
    /// programmatically on a known input must not be off by one. The SAX-facing
    /// [`Locator`](crate::Locator) is the only place that switches to 1-based numbering.
    #[test]
    fn positions_are_zero_based_when_read_programmatically() {
        let mut reader = PositionTrackingReader::new("<r>\nx".as_bytes());
        assert_eq!(reader.position(), TextPosition::start());
        assert_eq!(reader.position().line, 0);
        assert_eq!(reader.position().column, 0);

        // Consume `<r>`: still line 0, now column 3 (three scalar values consumed).
        let n = reader.fill_buf().expect("read").len().min(3);
        reader.consume(n);
        let position = reader.position();
        assert_eq!(position.line, 0);
        assert_eq!(position.column, 3);
        assert_eq!(position.offset, 3);

        // Consume the newline: line 1, column 0 — the first column of the second line.
        reader.consume(1);
        let position = reader.position();
        assert_eq!(position.line, 1);
        assert_eq!(position.column, 0);
        assert_eq!(position.to_string(), "line 2, column 1");
    }

    #[test]
    fn tracks_position_byte_by_byte() {
        let mut reader = PositionTrackingReader::new("ab\ncd".as_bytes());
        for _ in 0..5 {
            let available = reader.fill_buf().expect("read");
            assert!(!available.is_empty());
            reader.consume(1);
        }
        let position = reader.position();
        // 0-based line/column, as stored.
        assert_eq!(position.line, 1);
        assert_eq!(position.column, 2);
        assert_eq!(position.offset, 5);
        assert_eq!(position.to_string(), "line 2, column 3");
    }
}
