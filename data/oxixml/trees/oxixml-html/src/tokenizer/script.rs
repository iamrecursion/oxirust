//! The script data escaping states
//! ([HTML § 13.2.5.21](https://html.spec.whatwg.org/multipage/parsing.html#script-data-escape-start-state)
//! through
//! [§ 13.2.5.31](https://html.spec.whatwg.org/multipage/parsing.html#script-data-double-escape-end-state)).
//!
//! These states exist so that a `<script>` body may contain `<!--` … `-->` with
//! `<script>` tags inside it — a pattern old pages used to hide script source
//! from pre-HTML browsers — without the inner tags terminating the element.

use super::data::REPLACEMENT;
use super::{State, TagKind, Tokenizer, is_html_whitespace};
use crate::error::codes;
use crate::token::Tag;

impl Tokenizer<'_> {
    pub(super) fn step_script(&mut self) {
        match self.state {
            State::ScriptDataEscapeStart => self.escape_start_state(),
            State::ScriptDataEscapeStartDash => self.escape_start_dash_state(),
            State::ScriptDataEscaped => self.escaped_state(),
            State::ScriptDataEscapedDash => self.escaped_dash_state(),
            State::ScriptDataEscapedDashDash => self.escaped_dash_dash_state(),
            State::ScriptDataEscapedLessThanSign => self.escaped_less_than_sign_state(),
            State::ScriptDataEscapedEndTagOpen => self.escaped_end_tag_open_state(),
            State::ScriptDataEscapedEndTagName => self.escaped_end_tag_name_state(),
            State::ScriptDataDoubleEscapeStart => self.double_escape_start_state(),
            State::ScriptDataDoubleEscaped => self.double_escaped_state(),
            State::ScriptDataDoubleEscapedDash => self.double_escaped_dash_state(),
            State::ScriptDataDoubleEscapedDashDash => self.double_escaped_dash_dash_state(),
            State::ScriptDataDoubleEscapedLessThanSign => {
                self.double_escaped_less_than_sign_state();
            }
            State::ScriptDataDoubleEscapeEnd => self.double_escape_end_state(),
            _ => unreachable!("step_script dispatched a state it does not own"),
        }
    }

    /// <https://html.spec.whatwg.org/multipage/parsing.html#script-data-escape-start-state>
    fn escape_start_state(&mut self) {
        match self.consume() {
            Some('-') => {
                self.emit_char('-');
                self.state = State::ScriptDataEscapeStartDash;
            }
            other => {
                if self.suspended {
                    return;
                }
                if let Some(ch) = other {
                    self.unconsume(ch);
                }
                self.state = State::ScriptData;
            }
        }
    }

    /// <https://html.spec.whatwg.org/multipage/parsing.html#script-data-escape-start-dash-state>
    fn escape_start_dash_state(&mut self) {
        match self.consume() {
            Some('-') => {
                self.emit_char('-');
                self.state = State::ScriptDataEscapedDashDash;
            }
            other => {
                if self.suspended {
                    return;
                }
                if let Some(ch) = other {
                    self.unconsume(ch);
                }
                self.state = State::ScriptData;
            }
        }
    }

    /// <https://html.spec.whatwg.org/multipage/parsing.html#script-data-escaped-state>
    fn escaped_state(&mut self) {
        if let Some(run) = self.take_run(&['-', '<', '\0']) {
            self.emit_str(&run);
            return;
        }
        match self.consume() {
            Some('-') => {
                self.emit_char('-');
                self.state = State::ScriptDataEscapedDash;
            }
            Some('<') => self.state = State::ScriptDataEscapedLessThanSign,
            Some('\0') => {
                self.error(codes::UNEXPECTED_NULL_CHARACTER);
                self.emit_char(REPLACEMENT);
            }
            Some(ch) => self.emit_char(ch),
            None => {
                if self.suspended {
                    return;
                }
                self.error(codes::EOF_IN_SCRIPT_HTML_COMMENT_LIKE_TEXT);
                self.emit_eof();
            }
        }
    }

    /// <https://html.spec.whatwg.org/multipage/parsing.html#script-data-escaped-dash-state>
    fn escaped_dash_state(&mut self) {
        match self.consume() {
            Some('-') => {
                self.emit_char('-');
                self.state = State::ScriptDataEscapedDashDash;
            }
            Some('<') => self.state = State::ScriptDataEscapedLessThanSign,
            Some('\0') => {
                self.error(codes::UNEXPECTED_NULL_CHARACTER);
                self.emit_char(REPLACEMENT);
                self.state = State::ScriptDataEscaped;
            }
            Some(ch) => {
                self.emit_char(ch);
                self.state = State::ScriptDataEscaped;
            }
            None => {
                if self.suspended {
                    return;
                }
                self.error(codes::EOF_IN_SCRIPT_HTML_COMMENT_LIKE_TEXT);
                self.emit_eof();
            }
        }
    }

    /// <https://html.spec.whatwg.org/multipage/parsing.html#script-data-escaped-dash-dash-state>
    fn escaped_dash_dash_state(&mut self) {
        match self.consume() {
            Some('-') => self.emit_char('-'),
            Some('<') => self.state = State::ScriptDataEscapedLessThanSign,
            Some('>') => {
                self.emit_char('>');
                self.state = State::ScriptData;
            }
            Some('\0') => {
                self.error(codes::UNEXPECTED_NULL_CHARACTER);
                self.emit_char(REPLACEMENT);
                self.state = State::ScriptDataEscaped;
            }
            Some(ch) => {
                self.emit_char(ch);
                self.state = State::ScriptDataEscaped;
            }
            None => {
                if self.suspended {
                    return;
                }
                self.error(codes::EOF_IN_SCRIPT_HTML_COMMENT_LIKE_TEXT);
                self.emit_eof();
            }
        }
    }

    /// <https://html.spec.whatwg.org/multipage/parsing.html#script-data-escaped-less-than-sign-state>
    fn escaped_less_than_sign_state(&mut self) {
        match self.consume() {
            Some('/') => {
                self.temp_buffer.clear();
                self.state = State::ScriptDataEscapedEndTagOpen;
            }
            Some(ch) if ch.is_ascii_alphabetic() => {
                self.temp_buffer.clear();
                self.emit_char('<');
                self.unconsume(ch);
                self.state = State::ScriptDataDoubleEscapeStart;
            }
            other => {
                if self.suspended {
                    return;
                }
                self.emit_char('<');
                if let Some(ch) = other {
                    self.unconsume(ch);
                }
                self.state = State::ScriptDataEscaped;
            }
        }
    }

    /// <https://html.spec.whatwg.org/multipage/parsing.html#script-data-escaped-end-tag-open-state>
    fn escaped_end_tag_open_state(&mut self) {
        match self.consume() {
            Some(ch) if ch.is_ascii_alphabetic() => {
                self.tag_kind = TagKind::End;
                self.tag = Tag::default();
                self.attribute = None;
                self.unconsume(ch);
                self.state = State::ScriptDataEscapedEndTagName;
            }
            other => {
                if self.suspended {
                    return;
                }
                self.emit_str("</");
                if let Some(ch) = other {
                    self.unconsume(ch);
                }
                self.state = State::ScriptDataEscaped;
            }
        }
    }

    /// <https://html.spec.whatwg.org/multipage/parsing.html#script-data-escaped-end-tag-name-state>
    fn escaped_end_tag_name_state(&mut self) {
        match self.consume() {
            Some(ch) if is_html_whitespace(ch) && self.is_appropriate_end_tag() => {
                self.state = State::BeforeAttributeName;
            }
            Some('/') if self.is_appropriate_end_tag() => self.state = State::SelfClosingStartTag,
            Some('>') if self.is_appropriate_end_tag() => {
                self.state = State::Data;
                self.emit_current_tag();
            }
            Some(ch) if ch.is_ascii_alphabetic() => {
                self.tag.name.push(ch.to_ascii_lowercase());
                self.temp_buffer.push(ch);
            }
            other => {
                if self.suspended {
                    return;
                }
                self.emit_str("</");
                self.emit_temp_buffer();
                if let Some(ch) = other {
                    self.unconsume(ch);
                }
                self.state = State::ScriptDataEscaped;
            }
        }
    }

    /// <https://html.spec.whatwg.org/multipage/parsing.html#script-data-double-escape-start-state>
    fn double_escape_start_state(&mut self) {
        match self.consume() {
            Some(ch) if is_html_whitespace(ch) || ch == '/' || ch == '>' => {
                self.state = if self.temp_buffer == "script" {
                    State::ScriptDataDoubleEscaped
                } else {
                    State::ScriptDataEscaped
                };
                self.emit_char(ch);
            }
            Some(ch) if ch.is_ascii_alphabetic() => {
                self.temp_buffer.push(ch.to_ascii_lowercase());
                self.emit_char(ch);
            }
            other => {
                if self.suspended {
                    return;
                }
                if let Some(ch) = other {
                    self.unconsume(ch);
                }
                self.state = State::ScriptDataEscaped;
            }
        }
    }

    /// <https://html.spec.whatwg.org/multipage/parsing.html#script-data-double-escaped-state>
    fn double_escaped_state(&mut self) {
        if let Some(run) = self.take_run(&['-', '<', '\0']) {
            self.emit_str(&run);
            return;
        }
        match self.consume() {
            Some('-') => {
                self.emit_char('-');
                self.state = State::ScriptDataDoubleEscapedDash;
            }
            Some('<') => {
                self.emit_char('<');
                self.state = State::ScriptDataDoubleEscapedLessThanSign;
            }
            Some('\0') => {
                self.error(codes::UNEXPECTED_NULL_CHARACTER);
                self.emit_char(REPLACEMENT);
            }
            Some(ch) => self.emit_char(ch),
            None => {
                if self.suspended {
                    return;
                }
                self.error(codes::EOF_IN_SCRIPT_HTML_COMMENT_LIKE_TEXT);
                self.emit_eof();
            }
        }
    }

    /// <https://html.spec.whatwg.org/multipage/parsing.html#script-data-double-escaped-dash-state>
    fn double_escaped_dash_state(&mut self) {
        match self.consume() {
            Some('-') => {
                self.emit_char('-');
                self.state = State::ScriptDataDoubleEscapedDashDash;
            }
            Some('<') => {
                self.emit_char('<');
                self.state = State::ScriptDataDoubleEscapedLessThanSign;
            }
            Some('\0') => {
                self.error(codes::UNEXPECTED_NULL_CHARACTER);
                self.emit_char(REPLACEMENT);
                self.state = State::ScriptDataDoubleEscaped;
            }
            Some(ch) => {
                self.emit_char(ch);
                self.state = State::ScriptDataDoubleEscaped;
            }
            None => {
                if self.suspended {
                    return;
                }
                self.error(codes::EOF_IN_SCRIPT_HTML_COMMENT_LIKE_TEXT);
                self.emit_eof();
            }
        }
    }

    /// <https://html.spec.whatwg.org/multipage/parsing.html#script-data-double-escaped-dash-dash-state>
    fn double_escaped_dash_dash_state(&mut self) {
        match self.consume() {
            Some('-') => self.emit_char('-'),
            Some('<') => {
                self.emit_char('<');
                self.state = State::ScriptDataDoubleEscapedLessThanSign;
            }
            Some('>') => {
                self.emit_char('>');
                self.state = State::ScriptData;
            }
            Some('\0') => {
                self.error(codes::UNEXPECTED_NULL_CHARACTER);
                self.emit_char(REPLACEMENT);
                self.state = State::ScriptDataDoubleEscaped;
            }
            Some(ch) => {
                self.emit_char(ch);
                self.state = State::ScriptDataDoubleEscaped;
            }
            None => {
                if self.suspended {
                    return;
                }
                self.error(codes::EOF_IN_SCRIPT_HTML_COMMENT_LIKE_TEXT);
                self.emit_eof();
            }
        }
    }

    /// <https://html.spec.whatwg.org/multipage/parsing.html#script-data-double-escaped-less-than-sign-state>
    fn double_escaped_less_than_sign_state(&mut self) {
        match self.consume() {
            Some('/') => {
                self.temp_buffer.clear();
                self.emit_char('/');
                self.state = State::ScriptDataDoubleEscapeEnd;
            }
            other => {
                if self.suspended {
                    return;
                }
                if let Some(ch) = other {
                    self.unconsume(ch);
                }
                self.state = State::ScriptDataDoubleEscaped;
            }
        }
    }

    /// <https://html.spec.whatwg.org/multipage/parsing.html#script-data-double-escape-end-state>
    fn double_escape_end_state(&mut self) {
        match self.consume() {
            Some(ch) if is_html_whitespace(ch) || ch == '/' || ch == '>' => {
                self.state = if self.temp_buffer == "script" {
                    State::ScriptDataEscaped
                } else {
                    State::ScriptDataDoubleEscaped
                };
                self.emit_char(ch);
            }
            Some(ch) if ch.is_ascii_alphabetic() => {
                self.temp_buffer.push(ch.to_ascii_lowercase());
                self.emit_char(ch);
            }
            other => {
                if self.suspended {
                    return;
                }
                if let Some(ch) = other {
                    self.unconsume(ch);
                }
                self.state = State::ScriptDataDoubleEscaped;
            }
        }
    }
}
