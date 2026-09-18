//! Where a node began in the text a tree was parsed from.
//!
//! Nothing in the [XDM](https://www.w3.org/TR/xpath-datamodel-31/) asks a tree to remember this,
//! and most callers do not want to pay for it, so recording it is opt-in through
//! [`ParseOptions::record_source_positions`](crate::ParseOptions::record_source_positions) and the
//! positions live in a side table on the tree rather than in every node. What needs them is
//! diagnostics: a host language that reports where in a *stylesheet* or *query* module something
//! went wrong — XSLT's `$err:line-number` and `$err:column-number`, for instance — can only do so
//! if the tree it compiled from kept the source coordinates of its elements.

/// A one-based `(line, column)` position in the text a tree was parsed from.
///
/// The position is that of the `<` that opens the node's markup: the start tag for an element,
/// the `<!--` of a comment, the `<?` of a processing instruction, and the first character of the
/// text for a text node. Columns count *characters*, not bytes or display cells.
///
/// # Examples
///
/// ```
/// use oxixml_infoset::{ParseOptions, SourcePosition, Tree, XdmNode};
///
/// let tree = Tree::parse_str_with(
///     "<a>\n  <b/>\n</a>",
///     ParseOptions::default().record_source_positions(true),
/// )?;
/// let b = tree
///     .document_element()
///     .and_then(|a| a.children().get(1).cloned())
///     .expect("the second child");
/// assert_eq!(b.source_position(), Some(SourcePosition { line: 2, column: 3 }));
/// # Ok::<(), oxixml_infoset::XdmError>(())
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct SourcePosition {
    /// The line the node begins on, counting from one.
    pub line: u32,
    /// The character within that line the node begins at, counting from one.
    pub column: u32,
}

/// Turns the byte offsets a reader reports into [`SourcePosition`]s.
///
/// The cursor is advanced monotonically through the document, so counting the line ends of a whole
/// document costs one pass over it however many positions are asked for.
#[derive(Debug, Clone, Copy)]
pub(super) struct PositionCursor {
    offset: usize,
    line: u32,
    column: u32,
}

impl Default for PositionCursor {
    fn default() -> Self {
        Self {
            offset: 0,
            line: 1,
            column: 1,
        }
    }
}

impl PositionCursor {
    /// The position of byte `offset` of `text`.
    ///
    /// `offset` is expected to be at or after the previous call's; an offset before it — which a
    /// forward-only reader never produces — restarts the count from the beginning of `text` rather
    /// than reporting a stale answer.
    pub(super) fn position_at(&mut self, text: &str, offset: usize) -> SourcePosition {
        let offset = offset.min(text.len());
        if offset < self.offset {
            *self = Self::default();
        }
        let bytes = text.as_bytes();
        let mut index = self.offset;
        while index < offset {
            let Some(&byte) = bytes.get(index) else {
                break;
            };
            index += 1;
            match byte {
                // Both line ends of XML 1.0 §2.11, in the source text rather than in the
                // normalized character data: the reader's offsets index the text as written.
                b'\n' => {
                    self.line += 1;
                    self.column = 1;
                }
                b'\r' => {
                    self.line += 1;
                    self.column = 1;
                    if index < offset && bytes.get(index) == Some(&b'\n') {
                        index += 1;
                    }
                }
                // A continuation byte is the tail of a character already counted.
                byte if byte & 0xC0 == 0x80 => {}
                _ => self.column += 1,
            }
        }
        self.offset = index;
        SourcePosition {
            line: self.line,
            column: self.column,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn at(text: &str, offset: usize) -> SourcePosition {
        PositionCursor::default().position_at(text, offset)
    }

    #[test]
    fn the_first_byte_is_line_one_column_one() {
        assert_eq!(at("<a/>", 0), SourcePosition { line: 1, column: 1 });
        assert_eq!(at("", 0), SourcePosition { line: 1, column: 1 });
        // An offset past the end reports the end rather than panicking.
        assert_eq!(at("<a/>", 99), SourcePosition { line: 1, column: 5 });
    }

    #[test]
    fn every_line_end_starts_a_line() {
        let text = "a\nb\r\nc\rd";
        assert_eq!(at(text, 2), SourcePosition { line: 2, column: 1 });
        assert_eq!(at(text, 5), SourcePosition { line: 3, column: 1 });
        assert_eq!(at(text, 7), SourcePosition { line: 4, column: 1 });
    }

    #[test]
    fn columns_count_characters_rather_than_bytes() {
        let text = "<a>\u{e9}\u{1f600}x</a>";
        // `<a>` is three characters, so the accented letter is the fourth.
        assert_eq!(at(text, 3), SourcePosition { line: 1, column: 4 });
        assert_eq!(at(text, 3 + 2), SourcePosition { line: 1, column: 5 });
        assert_eq!(at(text, 3 + 2 + 4), SourcePosition { line: 1, column: 6 });
    }

    #[test]
    fn advancing_in_steps_matches_advancing_at_once() {
        let text = "one\ntwo\nthree\nfour";
        let mut stepwise = PositionCursor::default();
        for offset in 0..text.len() {
            let expected = at(text, offset);
            assert_eq!(stepwise.position_at(text, offset), expected, "at {offset}");
        }
    }

    #[test]
    fn an_offset_before_the_cursor_restarts_the_count() {
        let text = "a\nb\nc";
        let mut cursor = PositionCursor::default();
        assert_eq!(
            cursor.position_at(text, 4),
            SourcePosition { line: 3, column: 1 }
        );
        assert_eq!(
            cursor.position_at(text, 2),
            SourcePosition { line: 2, column: 1 }
        );
    }
}
