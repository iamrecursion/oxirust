//! The bogus comment, markup declaration open, comment and CDATA section
//! states ([HTML § 13.2.5.41](https://html.spec.whatwg.org/multipage/parsing.html#bogus-comment-state)
//! through
//! [§ 13.2.5.52](https://html.spec.whatwg.org/multipage/parsing.html#comment-end-bang-state)
//! and
//! [§ 13.2.5.69](https://html.spec.whatwg.org/multipage/parsing.html#cdata-section-state)
//! through
//! [§ 13.2.5.71](https://html.spec.whatwg.org/multipage/parsing.html#cdata-section-end-state)).

use super::data::REPLACEMENT;
use super::{State, Tokenizer};
use crate::error::codes;

impl Tokenizer<'_> {
    pub(super) fn step_comment(&mut self) {
        match self.state {
            State::BogusComment => self.bogus_comment_state(),
            State::MarkupDeclarationOpen => self.markup_declaration_open_state(),
            State::CommentStart => self.comment_start_state(),
            State::CommentStartDash => self.comment_start_dash_state(),
            State::Comment => self.comment_state(),
            State::CommentLessThanSign => self.comment_less_than_sign_state(),
            State::CommentLessThanSignBang => self.comment_less_than_sign_bang_state(),
            State::CommentLessThanSignBangDash => self.comment_less_than_sign_bang_dash_state(),
            State::CommentLessThanSignBangDashDash => {
                self.comment_less_than_sign_bang_dash_dash_state();
            }
            State::CommentEndDash => self.comment_end_dash_state(),
            State::CommentEnd => self.comment_end_state(),
            State::CommentEndBang => self.comment_end_bang_state(),
            State::CdataSection => self.cdata_section_state(),
            State::CdataSectionBracket => self.cdata_section_bracket_state(),
            State::CdataSectionEnd => self.cdata_section_end_state(),
            _ => unreachable!("step_comment dispatched a state it does not own"),
        }
    }

    /// <https://html.spec.whatwg.org/multipage/parsing.html#bogus-comment-state>
    fn bogus_comment_state(&mut self) {
        loop {
            match self.consume() {
                Some('>') => {
                    self.state = State::Data;
                    self.emit_comment();
                    return;
                }
                Some('\0') => {
                    self.error(codes::UNEXPECTED_NULL_CHARACTER);
                    self.comment.push(REPLACEMENT);
                }
                Some(ch) => self.comment.push(ch),
                None => {
                    if self.suspended {
                        return;
                    }
                    self.emit_comment();
                    self.emit_eof();
                    return;
                }
            }
        }
    }

    /// <https://html.spec.whatwg.org/multipage/parsing.html#markup-declaration-open-state>
    fn markup_declaration_open_state(&mut self) {
        // The three markup declarations this state recognises are at most
        // seven bytes long (`DOCTYPE`, `[CDATA[`), so seven bytes of input
        // settle which of them — if any — is here.
        if self.needs_lookahead(7) {
            return;
        }
        if self.consume_if_match("--", false) {
            self.comment.clear();
            self.state = State::CommentStart;
        } else if self.consume_if_match("DOCTYPE", true) {
            self.state = State::Doctype;
        } else if self.consume_if_match("[CDATA[", false) {
            if self.in_foreign_content {
                self.state = State::CdataSection;
            } else {
                self.error(codes::CDATA_IN_HTML_CONTENT);
                self.comment.clear();
                self.comment.push_str("[CDATA[");
                self.state = State::BogusComment;
            }
        } else {
            self.error(codes::INCORRECTLY_OPENED_COMMENT);
            self.comment.clear();
            self.state = State::BogusComment;
        }
    }

    /// <https://html.spec.whatwg.org/multipage/parsing.html#comment-start-state>
    fn comment_start_state(&mut self) {
        match self.consume() {
            Some('-') => self.state = State::CommentStartDash,
            Some('>') => {
                self.error(codes::ABRUPT_CLOSING_OF_EMPTY_COMMENT);
                self.state = State::Data;
                self.emit_comment();
            }
            other => {
                if self.suspended {
                    return;
                }
                if let Some(ch) = other {
                    self.unconsume(ch);
                }
                self.state = State::Comment;
            }
        }
    }

    /// <https://html.spec.whatwg.org/multipage/parsing.html#comment-start-dash-state>
    fn comment_start_dash_state(&mut self) {
        match self.consume() {
            Some('-') => self.state = State::CommentEnd,
            Some('>') => {
                self.error(codes::ABRUPT_CLOSING_OF_EMPTY_COMMENT);
                self.state = State::Data;
                self.emit_comment();
            }
            Some(ch) => {
                self.comment.push('-');
                self.unconsume(ch);
                self.state = State::Comment;
            }
            None => {
                if self.suspended {
                    return;
                }
                self.error(codes::EOF_IN_COMMENT);
                self.emit_comment();
                self.emit_eof();
            }
        }
    }

    /// <https://html.spec.whatwg.org/multipage/parsing.html#comment-state>
    fn comment_state(&mut self) {
        loop {
            if let Some(run) = self.take_run(&['<', '-', '\0']) {
                self.comment.push_str(&run);
                continue;
            }
            match self.consume() {
                Some('<') => {
                    self.comment.push('<');
                    self.state = State::CommentLessThanSign;
                    return;
                }
                Some('-') => {
                    self.state = State::CommentEndDash;
                    return;
                }
                Some('\0') => {
                    self.error(codes::UNEXPECTED_NULL_CHARACTER);
                    self.comment.push(REPLACEMENT);
                }
                Some(ch) => self.comment.push(ch),
                None => {
                    if self.suspended {
                        return;
                    }
                    self.error(codes::EOF_IN_COMMENT);
                    self.emit_comment();
                    self.emit_eof();
                    return;
                }
            }
        }
    }

    /// <https://html.spec.whatwg.org/multipage/parsing.html#comment-less-than-sign-state>
    fn comment_less_than_sign_state(&mut self) {
        match self.consume() {
            Some('!') => {
                self.comment.push('!');
                self.state = State::CommentLessThanSignBang;
            }
            Some('<') => self.comment.push('<'),
            other => {
                if self.suspended {
                    return;
                }
                if let Some(ch) = other {
                    self.unconsume(ch);
                }
                self.state = State::Comment;
            }
        }
    }

    /// <https://html.spec.whatwg.org/multipage/parsing.html#comment-less-than-sign-bang-state>
    fn comment_less_than_sign_bang_state(&mut self) {
        match self.consume() {
            Some('-') => self.state = State::CommentLessThanSignBangDash,
            other => {
                if self.suspended {
                    return;
                }
                if let Some(ch) = other {
                    self.unconsume(ch);
                }
                self.state = State::Comment;
            }
        }
    }

    /// <https://html.spec.whatwg.org/multipage/parsing.html#comment-less-than-sign-bang-dash-state>
    fn comment_less_than_sign_bang_dash_state(&mut self) {
        match self.consume() {
            Some('-') => self.state = State::CommentLessThanSignBangDashDash,
            other => {
                if self.suspended {
                    return;
                }
                if let Some(ch) = other {
                    self.unconsume(ch);
                }
                self.state = State::CommentEndDash;
            }
        }
    }

    /// <https://html.spec.whatwg.org/multipage/parsing.html#comment-less-than-sign-bang-dash-dash-state>
    fn comment_less_than_sign_bang_dash_dash_state(&mut self) {
        match self.consume() {
            Some('>') => {
                self.unconsume('>');
                self.state = State::CommentEnd;
            }
            None => {
                if self.suspended {
                    return;
                }
                self.state = State::CommentEnd;
            }
            Some(ch) => {
                self.error(codes::NESTED_COMMENT);
                self.unconsume(ch);
                self.state = State::CommentEnd;
            }
        }
    }

    /// <https://html.spec.whatwg.org/multipage/parsing.html#comment-end-dash-state>
    fn comment_end_dash_state(&mut self) {
        match self.consume() {
            Some('-') => self.state = State::CommentEnd,
            Some(ch) => {
                self.comment.push('-');
                self.unconsume(ch);
                self.state = State::Comment;
            }
            None => {
                if self.suspended {
                    return;
                }
                self.error(codes::EOF_IN_COMMENT);
                self.emit_comment();
                self.emit_eof();
            }
        }
    }

    /// <https://html.spec.whatwg.org/multipage/parsing.html#comment-end-state>
    fn comment_end_state(&mut self) {
        match self.consume() {
            Some('>') => {
                self.state = State::Data;
                self.emit_comment();
            }
            Some('!') => self.state = State::CommentEndBang,
            Some('-') => self.comment.push('-'),
            Some(ch) => {
                self.comment.push_str("--");
                self.unconsume(ch);
                self.state = State::Comment;
            }
            None => {
                if self.suspended {
                    return;
                }
                self.error(codes::EOF_IN_COMMENT);
                self.emit_comment();
                self.emit_eof();
            }
        }
    }

    /// <https://html.spec.whatwg.org/multipage/parsing.html#comment-end-bang-state>
    fn comment_end_bang_state(&mut self) {
        match self.consume() {
            Some('-') => {
                self.comment.push_str("--!");
                self.state = State::CommentEndDash;
            }
            Some('>') => {
                self.error(codes::INCORRECTLY_CLOSED_COMMENT);
                self.state = State::Data;
                self.emit_comment();
            }
            Some(ch) => {
                self.comment.push_str("--!");
                self.unconsume(ch);
                self.state = State::Comment;
            }
            None => {
                if self.suspended {
                    return;
                }
                self.error(codes::EOF_IN_COMMENT);
                self.emit_comment();
                self.emit_eof();
            }
        }
    }

    /// <https://html.spec.whatwg.org/multipage/parsing.html#cdata-section-state>
    fn cdata_section_state(&mut self) {
        loop {
            if let Some(run) = self.take_run(&[']']) {
                self.emit_str(&run);
                continue;
            }
            match self.consume() {
                Some(']') => {
                    self.state = State::CdataSectionBracket;
                    return;
                }
                Some(ch) => self.emit_char(ch),
                None => {
                    if self.suspended {
                        return;
                    }
                    self.error(codes::EOF_IN_CDATA);
                    self.emit_eof();
                    return;
                }
            }
        }
    }

    /// <https://html.spec.whatwg.org/multipage/parsing.html#cdata-section-bracket-state>
    fn cdata_section_bracket_state(&mut self) {
        match self.consume() {
            Some(']') => self.state = State::CdataSectionEnd,
            other => {
                if self.suspended {
                    return;
                }
                self.emit_char(']');
                if let Some(ch) = other {
                    self.unconsume(ch);
                }
                self.state = State::CdataSection;
            }
        }
    }

    /// <https://html.spec.whatwg.org/multipage/parsing.html#cdata-section-end-state>
    fn cdata_section_end_state(&mut self) {
        match self.consume() {
            Some(']') => self.emit_char(']'),
            Some('>') => self.state = State::Data,
            other => {
                if self.suspended {
                    return;
                }
                self.emit_str("]]");
                if let Some(ch) = other {
                    self.unconsume(ch);
                }
                self.state = State::CdataSection;
            }
        }
    }
}
