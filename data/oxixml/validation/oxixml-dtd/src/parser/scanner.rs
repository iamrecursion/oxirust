//! The character source the DTD parser reads from.
//!
//! A DTD is not one flat string: a parameter-entity reference splices another
//! text in at the point of reference, and that text may itself contain further
//! references. [`Scanner`] therefore keeps a *stack* of input frames — the
//! subset text at the bottom, one frame per entity being expanded above it —
//! and reads through them transparently.
//!
//! Keeping the frames explicit rather than pre-substituting the text buys three
//! things the XML Recommendation needs:
//!
//! * accurate positions: an error is reported at its line and column *within
//!   the entity*, together with the chain of entities that led there;
//! * the "PE Between Declarations" well-formedness constraint of XML 1.0 §2.8,
//!   which is exactly the statement that a markup declaration must begin and
//!   end in the same frame;
//! * the leading and trailing space that XML 1.0 §4.4.8 requires be attached to
//!   a parameter entity's replacement text when it is included in the DTD (and
//!   must *not* be attached when it is included in a literal).

use oxixml_support::pos::{LineColTracker, TextPosition};

/// One input frame: a whole text plus the read cursor into it.
#[derive(Debug)]
struct Frame {
    id: u64,
    text: String,
    cursor: usize,
    tracker: LineColTracker,
    entity: Option<String>,
    base: Option<String>,
}

/// A stack of input frames, read as a single character stream.
#[derive(Debug)]
pub(crate) struct Scanner {
    frames: Vec<Frame>,
    next_id: u64,
    last_id: u64,
}

impl Scanner {
    /// Starts scanning `text`, with `base` as its base URI.
    pub(crate) fn new(text: String, base: Option<String>) -> Self {
        let frame = Frame {
            id: 0,
            text,
            cursor: 0,
            tracker: LineColTracker::new(),
            entity: None,
            base,
        };
        Self {
            frames: vec![frame],
            next_id: 1,
            last_id: 0,
        }
    }

    /// Splices `text` in at the current position as the replacement text of
    /// entity `name`.
    ///
    /// When `padded`, a space is attached before and after the replacement
    /// text, as XML 1.0 §4.4.8 requires for a parameter entity included in the
    /// DTD.
    pub(crate) fn push_entity(
        &mut self,
        name: impl Into<String>,
        text: &str,
        base: Option<String>,
        padded: bool,
    ) {
        let text = if padded {
            format!(" {text} ")
        } else {
            text.to_owned()
        };
        let id = self.next_id;
        self.next_id += 1;
        self.frames.push(Frame {
            id,
            text,
            cursor: 0,
            tracker: LineColTracker::new(),
            entity: Some(name.into()),
            base,
        });
    }

    /// Drops every exhausted frame above the base frame.
    fn settle(&mut self) {
        while self.frames.len() > 1 {
            let Some(top) = self.frames.last() else {
                return;
            };
            if top.cursor < top.text.len() {
                return;
            }
            self.frames.pop();
        }
    }

    /// The remaining text of the innermost non-exhausted frame.
    pub(crate) fn rest(&mut self) -> &str {
        self.settle();
        match self.frames.last() {
            Some(frame) => &frame.text[frame.cursor..],
            None => "",
        }
    }

    /// The next character, without consuming it.
    pub(crate) fn peek(&mut self) -> Option<char> {
        self.rest().chars().next()
    }

    /// The character `offset` characters ahead within the current frame.
    pub(crate) fn peek_ahead(&mut self, offset: usize) -> Option<char> {
        self.rest().chars().nth(offset)
    }

    /// Consumes and returns the next character.
    pub(crate) fn next_char(&mut self) -> Option<char> {
        self.settle();
        let frame = self.frames.last_mut()?;
        let c = frame.text[frame.cursor..].chars().next()?;
        let width = c.len_utf8();
        frame
            .tracker
            .advance(&frame.text.as_bytes()[frame.cursor..frame.cursor + width]);
        frame.cursor += width;
        self.last_id = frame.id;
        Some(c)
    }

    /// Consumes `literal` when it is what comes next in the current frame.
    pub(crate) fn try_consume(&mut self, literal: &str) -> bool {
        if !self.rest().starts_with(literal) {
            return false;
        }
        self.advance_by(literal.len());
        true
    }

    /// Consumes `count` bytes of the current frame. The caller must have
    /// checked that `count` lands on a character boundary.
    fn advance_by(&mut self, count: usize) {
        self.settle();
        let Some(frame) = self.frames.last_mut() else {
            return;
        };
        let end = (frame.cursor + count).min(frame.text.len());
        frame
            .tracker
            .advance(&frame.text.as_bytes()[frame.cursor..end]);
        frame.cursor = end;
        self.last_id = frame.id;
    }

    /// Consumes and returns the longest run of characters satisfying
    /// `predicate`, continuing across frame boundaries.
    ///
    /// Crossing boundaries matters: `<!ATTLIST name %a; %b;>` leaves the cursor
    /// at the end of `%a;`'s replacement text after the first run of white
    /// space, and the white space that separates it from `%b;` lives in the
    /// *outer* frame. Names can never be split this way, because XML 1.0 §4.4.8
    /// pads every parameter entity included in the DTD with a space on each
    /// side.
    pub(crate) fn take_while(&mut self, predicate: impl Fn(char) -> bool) -> String {
        let mut taken = String::new();
        loop {
            let chunk: String = self.rest().chars().take_while(|&c| predicate(c)).collect();
            if chunk.is_empty() {
                return taken;
            }
            self.advance_by(chunk.len());
            taken.push_str(&chunk);
        }
    }

    /// Consumes characters up to and including `terminator`, returning what came
    /// before it, or `None` when the terminator does not occur in the current
    /// frame.
    pub(crate) fn take_until(&mut self, terminator: &str) -> Option<String> {
        let position = self.rest().find(terminator)?;
        let taken = self.rest()[..position].to_owned();
        self.advance_by(position + terminator.len());
        Some(taken)
    }

    /// Whether every frame is exhausted.
    pub(crate) fn is_eof(&mut self) -> bool {
        self.rest().is_empty()
    }

    /// The identifier of the frame the last character came from.
    pub(crate) fn last_frame_id(&self) -> u64 {
        self.last_id
    }

    /// The identifier of the frame the next character will come from.
    pub(crate) fn current_frame_id(&mut self) -> u64 {
        self.settle();
        self.frames.last().map_or(0, |frame| frame.id)
    }

    /// How many entity frames are stacked above the base text.
    pub(crate) fn entity_depth(&self) -> usize {
        self.frames.len() - 1
    }

    /// Whether the parser is currently reading from the base text rather than
    /// from an entity.
    pub(crate) fn in_base_text(&mut self) -> bool {
        self.settle();
        self.frames.len() == 1
    }

    /// The position within the innermost frame.
    pub(crate) fn position(&self) -> TextPosition {
        self.frames
            .last()
            .map_or_else(TextPosition::start, |frame| frame.tracker.position())
    }

    /// The names of the entities being expanded, outermost first.
    pub(crate) fn entity_stack(&self) -> Vec<String> {
        self.frames
            .iter()
            .filter_map(|frame| frame.entity.clone())
            .collect()
    }

    /// The base URI in force, taken from the innermost frame that has one.
    pub(crate) fn base(&self) -> Option<&str> {
        self.frames
            .iter()
            .rev()
            .find_map(|frame| frame.base.as_deref())
    }

    /// The name of the entity currently being expanded, if any.
    pub(crate) fn current_entity(&self) -> Option<&str> {
        self.frames.last().and_then(|frame| frame.entity.as_deref())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn frames_are_read_through_transparently() {
        let mut scanner = Scanner::new("ab".to_owned(), None);
        assert_eq!(scanner.next_char(), Some('a'));
        scanner.push_entity("p", "XY", None, false);
        assert_eq!(scanner.next_char(), Some('X'));
        assert_eq!(scanner.next_char(), Some('Y'));
        assert_eq!(scanner.next_char(), Some('b'));
        assert_eq!(scanner.next_char(), None);
        assert!(scanner.is_eof());
    }

    #[test]
    fn parameter_entities_are_padded_with_spaces_in_the_dtd() {
        let mut scanner = Scanner::new(String::new(), None);
        scanner.push_entity("p", "X", None, true);
        assert_eq!(scanner.next_char(), Some(' '));
        assert_eq!(scanner.next_char(), Some('X'));
        assert_eq!(scanner.next_char(), Some(' '));
        assert_eq!(scanner.next_char(), None);
    }

    #[test]
    fn frame_identity_tracks_where_a_character_came_from() {
        let mut scanner = Scanner::new("<>".to_owned(), None);
        let start = scanner.current_frame_id();
        assert_eq!(scanner.next_char(), Some('<'));
        scanner.push_entity("p", ">", None, false);
        assert_eq!(scanner.next_char(), Some('>'));
        assert_ne!(scanner.last_frame_id(), start);
    }

    #[test]
    fn positions_are_relative_to_the_innermost_frame() {
        let mut scanner = Scanner::new("line1\nline2".to_owned(), None);
        for _ in 0..8 {
            scanner.next_char();
        }
        let position = scanner.position();
        // 0-based storage: the second line is line 1, the third column is column 2.
        assert_eq!(position.line, 1);
        assert_eq!(position.column, 2);
        assert_eq!(position.to_string(), "line 2, column 3");
    }

    #[test]
    fn take_until_stops_at_the_terminator() {
        let mut scanner = Scanner::new("abc-->rest".to_owned(), None);
        assert_eq!(scanner.take_until("-->"), Some("abc".to_owned()));
        assert_eq!(scanner.rest(), "rest");
        assert_eq!(scanner.take_until("-->"), None);
    }

    #[test]
    fn entity_stack_lists_outermost_first() {
        let mut scanner = Scanner::new("x".to_owned(), None);
        scanner.push_entity("outer", "y", None, false);
        scanner.push_entity("inner", "z", None, false);
        assert_eq!(scanner.entity_stack(), ["outer", "inner"]);
        assert_eq!(scanner.entity_depth(), 2);
        assert_eq!(scanner.current_entity(), Some("inner"));
    }
}
