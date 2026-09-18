//! The data, RCDATA, RAWTEXT, script-data and plain-text states, plus the tag
//! open / tag name states that leave them
//! ([HTML § 13.2.5.1](https://html.spec.whatwg.org/multipage/parsing.html#data-state)
//! through
//! [§ 13.2.5.20](https://html.spec.whatwg.org/multipage/parsing.html#script-data-end-tag-name-state)).

use super::{State, TagKind, Tokenizer};
use crate::error::codes;

/// The character that replaces U+0000 wherever the standard says "emit a
/// U+FFFD REPLACEMENT CHARACTER character token".
pub(super) const REPLACEMENT: char = '\u{fffd}';

impl Tokenizer<'_> {
    pub(super) fn step_data(&mut self) {
        match self.state {
            State::Data => self.data_state(),
            State::Rcdata => self.rcdata_state(),
            State::RawText => self.raw_text_state(),
            State::ScriptData => self.script_data_state(),
            State::PlainText => self.plain_text_state(),
            State::TagOpen => self.tag_open_state(),
            State::EndTagOpen => self.end_tag_open_state(),
            State::TagName => self.tag_name_state(),
            State::RcdataLessThanSign => {
                self.less_than_sign_state(State::Rcdata, State::RcdataEndTagOpen);
            }
            State::RcdataEndTagOpen => {
                self.end_tag_open_in_text(State::Rcdata, State::RcdataEndTagName);
            }
            State::RcdataEndTagName => self.end_tag_name_in_text(State::Rcdata),
            State::RawTextLessThanSign => {
                self.less_than_sign_state(State::RawText, State::RawTextEndTagOpen);
            }
            State::RawTextEndTagOpen => {
                self.end_tag_open_in_text(State::RawText, State::RawTextEndTagName);
            }
            State::RawTextEndTagName => self.end_tag_name_in_text(State::RawText),
            State::ScriptDataLessThanSign => self.script_data_less_than_sign_state(),
            State::ScriptDataEndTagOpen => {
                self.end_tag_open_in_text(State::ScriptData, State::ScriptDataEndTagName);
            }
            State::ScriptDataEndTagName => self.end_tag_name_in_text(State::ScriptData),
            _ => unreachable!("step_data dispatched a state it does not own"),
        }
    }

    /// <https://html.spec.whatwg.org/multipage/parsing.html#data-state>
    fn data_state(&mut self) {
        if let Some(run) = self.take_run(&['&', '<', '\0']) {
            self.emit_str(&run);
            return;
        }
        match self.consume() {
            Some('&') => {
                self.return_state = State::Data;
                self.state = State::CharacterReference;
            }
            Some('<') => self.state = State::TagOpen,
            Some('\0') => {
                self.error(codes::UNEXPECTED_NULL_CHARACTER);
                self.emit_char('\0');
            }
            Some(ch) => self.emit_char(ch),
            None => {
                if self.suspended {
                    return;
                }
                self.emit_eof();
            }
        }
    }

    /// <https://html.spec.whatwg.org/multipage/parsing.html#rcdata-state>
    fn rcdata_state(&mut self) {
        if let Some(run) = self.take_run(&['&', '<', '\0']) {
            self.emit_str(&run);
            return;
        }
        match self.consume() {
            Some('&') => {
                self.return_state = State::Rcdata;
                self.state = State::CharacterReference;
            }
            Some('<') => self.state = State::RcdataLessThanSign,
            Some('\0') => {
                self.error(codes::UNEXPECTED_NULL_CHARACTER);
                self.emit_char(REPLACEMENT);
            }
            Some(ch) => self.emit_char(ch),
            None => {
                if self.suspended {
                    return;
                }
                self.emit_eof();
            }
        }
    }

    /// <https://html.spec.whatwg.org/multipage/parsing.html#rawtext-state>
    fn raw_text_state(&mut self) {
        if let Some(run) = self.take_run(&['<', '\0']) {
            self.emit_str(&run);
            return;
        }
        match self.consume() {
            Some('<') => self.state = State::RawTextLessThanSign,
            Some('\0') => {
                self.error(codes::UNEXPECTED_NULL_CHARACTER);
                self.emit_char(REPLACEMENT);
            }
            Some(ch) => self.emit_char(ch),
            None => {
                if self.suspended {
                    return;
                }
                self.emit_eof();
            }
        }
    }

    /// <https://html.spec.whatwg.org/multipage/parsing.html#script-data-state>
    fn script_data_state(&mut self) {
        if let Some(run) = self.take_run(&['<', '\0']) {
            self.emit_str(&run);
            return;
        }
        match self.consume() {
            Some('<') => self.state = State::ScriptDataLessThanSign,
            Some('\0') => {
                self.error(codes::UNEXPECTED_NULL_CHARACTER);
                self.emit_char(REPLACEMENT);
            }
            Some(ch) => self.emit_char(ch),
            None => {
                if self.suspended {
                    return;
                }
                self.emit_eof();
            }
        }
    }

    /// <https://html.spec.whatwg.org/multipage/parsing.html#plaintext-state>
    fn plain_text_state(&mut self) {
        if let Some(run) = self.take_run(&['\0']) {
            self.emit_str(&run);
            return;
        }
        match self.consume() {
            Some('\0') => {
                self.error(codes::UNEXPECTED_NULL_CHARACTER);
                self.emit_char(REPLACEMENT);
            }
            Some(ch) => self.emit_char(ch),
            None => {
                if self.suspended {
                    return;
                }
                self.emit_eof();
            }
        }
    }

    /// <https://html.spec.whatwg.org/multipage/parsing.html#tag-open-state>
    fn tag_open_state(&mut self) {
        match self.consume() {
            Some('!') => self.state = State::MarkupDeclarationOpen,
            Some('/') => self.state = State::EndTagOpen,
            Some(ch) if ch.is_ascii_alphabetic() => {
                self.start_tag(TagKind::Start, ch.to_ascii_lowercase());
                self.state = State::TagName;
            }
            Some('?') if self.processing_instructions => {
                self.temp_buffer.clear();
                self.state = State::ProcessingInstructionOpen;
            }
            Some('?') => {
                // Tag open state, U+003F: "This is an
                // unexpected-question-mark-instead-of-tag-name parse error.
                // Create a comment token whose data is the empty string.
                // Reconsume in the bogus comment state."
                self.error(codes::UNEXPECTED_QUESTION_MARK_INSTEAD_OF_TAG_NAME);
                self.comment.clear();
                self.unconsume('?');
                self.state = State::BogusComment;
            }
            Some(ch) => {
                self.error(codes::INVALID_FIRST_CHARACTER_OF_TAG_NAME);
                self.emit_char('<');
                self.unconsume(ch);
                self.state = State::Data;
            }
            None => {
                if self.suspended {
                    return;
                }
                self.error(codes::EOF_BEFORE_TAG_NAME);
                self.emit_char('<');
                self.emit_eof();
            }
        }
    }

    /// <https://html.spec.whatwg.org/multipage/parsing.html#end-tag-open-state>
    fn end_tag_open_state(&mut self) {
        match self.consume() {
            Some(ch) if ch.is_ascii_alphabetic() => {
                self.start_tag(TagKind::End, ch.to_ascii_lowercase());
                self.state = State::TagName;
            }
            Some('>') => {
                self.error(codes::MISSING_END_TAG_NAME);
                self.state = State::Data;
            }
            Some(ch) => {
                self.error(codes::INVALID_FIRST_CHARACTER_OF_TAG_NAME);
                self.comment.clear();
                self.unconsume(ch);
                self.state = State::BogusComment;
            }
            None => {
                if self.suspended {
                    return;
                }
                self.error(codes::EOF_BEFORE_TAG_NAME);
                self.emit_str("</");
                self.emit_eof();
            }
        }
    }

    /// <https://html.spec.whatwg.org/multipage/parsing.html#tag-name-state>
    fn tag_name_state(&mut self) {
        loop {
            match self.consume() {
                Some(ch) if super::is_html_whitespace(ch) => {
                    self.state = State::BeforeAttributeName;
                    return;
                }
                Some('/') => {
                    self.state = State::SelfClosingStartTag;
                    return;
                }
                Some('>') => {
                    self.state = State::Data;
                    self.emit_current_tag();
                    return;
                }
                Some('\0') => {
                    self.error(codes::UNEXPECTED_NULL_CHARACTER);
                    self.tag.name.push(REPLACEMENT);
                }
                Some(ch) => self.tag.name.push(ch.to_ascii_lowercase()),
                None => {
                    if self.suspended {
                        return;
                    }
                    self.error(codes::EOF_IN_TAG);
                    self.emit_eof();
                    return;
                }
            }
        }
    }

    /// The RCDATA / RAWTEXT / script-data "less-than sign" states, which are
    /// identical apart from the state they fall back to.
    fn less_than_sign_state(&mut self, text_state: State, end_tag_open: State) {
        match self.consume() {
            Some('/') => {
                self.temp_buffer.clear();
                self.state = end_tag_open;
            }
            other => {
                if self.suspended {
                    return;
                }
                self.emit_char('<');
                if let Some(ch) = other {
                    self.unconsume(ch);
                }
                self.state = text_state;
            }
        }
    }

    /// <https://html.spec.whatwg.org/multipage/parsing.html#script-data-less-than-sign-state>
    fn script_data_less_than_sign_state(&mut self) {
        match self.consume() {
            Some('/') => {
                self.temp_buffer.clear();
                self.state = State::ScriptDataEndTagOpen;
            }
            Some('!') => {
                self.state = State::ScriptDataEscapeStart;
                self.emit_str("<!");
            }
            other => {
                if self.suspended {
                    return;
                }
                self.emit_char('<');
                if let Some(ch) = other {
                    self.unconsume(ch);
                }
                self.state = State::ScriptData;
            }
        }
    }

    /// The RCDATA / RAWTEXT / script-data "end tag open" states.
    fn end_tag_open_in_text(&mut self, text_state: State, end_tag_name: State) {
        match self.consume() {
            Some(ch) if ch.is_ascii_alphabetic() => {
                self.start_tag(TagKind::End, ch.to_ascii_lowercase());
                self.temp_buffer.push(ch);
                self.state = end_tag_name;
            }
            other => {
                if self.suspended {
                    return;
                }
                self.emit_str("</");
                if let Some(ch) = other {
                    self.unconsume(ch);
                }
                self.state = text_state;
            }
        }
    }

    /// The RCDATA / RAWTEXT / script-data "end tag name" states: an end tag
    /// only closes the element when it is the "appropriate end tag token";
    /// anything else is text.
    fn end_tag_name_in_text(&mut self, text_state: State) {
        match self.consume() {
            Some(ch) if super::is_html_whitespace(ch) && self.is_appropriate_end_tag() => {
                self.state = State::BeforeAttributeName;
            }
            Some('/') if self.is_appropriate_end_tag() => {
                self.state = State::SelfClosingStartTag;
            }
            Some('>') if self.is_appropriate_end_tag() => {
                self.state = State::Data;
                self.emit_current_tag();
            }
            Some(ch) if ch.is_ascii_alphanumeric() => {
                self.tag.name.push(ch.to_ascii_lowercase());
                self.temp_buffer.push(ch);
            }
            other => {
                if self.suspended {
                    return;
                }
                self.emit_str("</");
                let buffer = std::mem::take(&mut self.temp_buffer);
                self.emit_str(&buffer);
                if let Some(ch) = other {
                    self.unconsume(ch);
                }
                self.state = text_state;
            }
        }
    }
}

impl Tokenizer<'_> {
    /// Emits the `<` of an unfinished markup construct followed by the
    /// characters that were consumed with it, used by the script escape states.
    pub(super) fn emit_temp_buffer(&mut self) {
        let buffer = std::mem::take(&mut self.temp_buffer);
        self.emit_str(&buffer);
    }
}
