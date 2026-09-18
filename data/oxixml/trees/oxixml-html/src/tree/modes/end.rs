//! The insertion modes that run once the body is done, and the two frameset
//! modes: "after body", "in frameset", "after frameset", "after after body" and
//! "after after frameset"
//! ([HTML § 13.2.6.4.19](https://html.spec.whatwg.org/multipage/parsing.html#the-after-body-insertion-mode)
//! and following).

use super::split_whitespace_prefix;
use crate::error::codes;
use crate::token::Token;
use crate::tree::{Action, Mode, TreeBuilder};

impl TreeBuilder<'_> {
    /// <https://html.spec.whatwg.org/multipage/parsing.html#the-after-body-insertion-mode>
    pub(crate) fn mode_after_body(&mut self, token: Token) -> Action {
        match token {
            Token::Characters(text) => {
                let (whitespace, rest) = split_whitespace_prefix(&text);
                if !whitespace.is_empty() {
                    let _ = self.dispatch(Mode::InBody, Token::Characters(whitespace.to_owned()));
                }
                if rest.is_empty() {
                    return Action::Done;
                }
                self.error(codes::UNEXPECTED_CHARACTER);
                self.mode = Mode::InBody;
                Action::Reprocess(Token::Characters(rest.to_owned()))
            }
            Token::Comment(_) | Token::ProcessingInstruction(_) => {
                let point = self.html_element_end();
                self.insert_markup(&token, Some(point));
                Action::Done
            }
            Token::Doctype(_) => {
                self.error(codes::UNEXPECTED_DOCTYPE);
                Action::Done
            }
            Token::StartTag(ref tag) if tag.name == "html" => self.dispatch(Mode::InBody, token),
            Token::EndTag(ref tag) if tag.name == "html" => {
                if self.fragment {
                    self.error(codes::UNEXPECTED_END_TAG);
                    return Action::Done;
                }
                self.mode = Mode::AfterAfterBody;
                Action::Done
            }
            Token::Eof => {
                self.stop_parsing();
                Action::Done
            }
            other => {
                self.error(codes::UNEXPECTED_END_TAG);
                self.mode = Mode::InBody;
                Action::Reprocess(other)
            }
        }
    }

    /// <https://html.spec.whatwg.org/multipage/parsing.html#parsing-main-inframeset>
    pub(crate) fn mode_in_frameset(&mut self, token: Token) -> Action {
        match token {
            Token::Characters(text) => {
                // Each character is a token of its own here: the white space
                // ones are inserted and the rest are dropped, so " te st"
                // contributes two spaces and nothing else.
                for ch in text.chars() {
                    if matches!(ch, '\t' | '\n' | '\u{c}' | '\r' | ' ') {
                        let mut buffer = [0_u8; 4];
                        let encoded = ch.encode_utf8(&mut buffer).to_owned();
                        self.insert_characters(&encoded);
                    } else {
                        self.error(codes::UNEXPECTED_CHARACTER);
                    }
                }
                Action::Done
            }
            Token::Comment(_) | Token::ProcessingInstruction(_) => {
                self.insert_markup(&token, None);
                Action::Done
            }
            Token::Doctype(_) => {
                self.error(codes::UNEXPECTED_DOCTYPE);
                Action::Done
            }
            Token::StartTag(ref tag) if tag.name == "html" => self.dispatch(Mode::InBody, token),
            Token::StartTag(tag) if tag.name == "frameset" => {
                self.insert_html_element(&tag);
                Action::Done
            }
            Token::EndTag(ref tag) if tag.name == "frameset" => {
                if self.open.len() == 1 {
                    self.error(codes::UNEXPECTED_END_TAG);
                    return Action::Done;
                }
                self.open.pop();
                if !self.fragment && !self.current_is_html("frameset") {
                    self.mode = Mode::AfterFrameset;
                }
                Action::Done
            }
            Token::StartTag(tag) if tag.name == "frame" => {
                self.insert_html_element(&tag);
                self.open.pop();
                Action::Done
            }
            Token::StartTag(ref tag) if tag.name == "noframes" => {
                self.dispatch(Mode::InHead, token)
            }
            Token::Eof => {
                if self.open.len() != 1 {
                    self.error(codes::UNEXPECTED_EOF);
                }
                self.stop_parsing();
                Action::Done
            }
            _ => {
                self.error(codes::UNEXPECTED_START_TAG);
                Action::Done
            }
        }
    }

    /// <https://html.spec.whatwg.org/multipage/parsing.html#parsing-main-afterframeset>
    pub(crate) fn mode_after_frameset(&mut self, token: Token) -> Action {
        match token {
            Token::Characters(text) => {
                // Each character is a token of its own here: the white space
                // ones are inserted and the rest are dropped, so " te st"
                // contributes two spaces and nothing else.
                for ch in text.chars() {
                    if matches!(ch, '\t' | '\n' | '\u{c}' | '\r' | ' ') {
                        let mut buffer = [0_u8; 4];
                        let encoded = ch.encode_utf8(&mut buffer).to_owned();
                        self.insert_characters(&encoded);
                    } else {
                        self.error(codes::UNEXPECTED_CHARACTER);
                    }
                }
                Action::Done
            }
            Token::Comment(_) | Token::ProcessingInstruction(_) => {
                self.insert_markup(&token, None);
                Action::Done
            }
            Token::Doctype(_) => {
                self.error(codes::UNEXPECTED_DOCTYPE);
                Action::Done
            }
            Token::StartTag(ref tag) if tag.name == "html" => self.dispatch(Mode::InBody, token),
            Token::EndTag(ref tag) if tag.name == "html" => {
                self.mode = Mode::AfterAfterFrameset;
                Action::Done
            }
            Token::StartTag(ref tag) if tag.name == "noframes" => {
                self.dispatch(Mode::InHead, token)
            }
            Token::Eof => {
                self.stop_parsing();
                Action::Done
            }
            _ => {
                self.error(codes::UNEXPECTED_START_TAG);
                Action::Done
            }
        }
    }

    /// <https://html.spec.whatwg.org/multipage/parsing.html#the-after-after-body-insertion-mode>
    pub(crate) fn mode_after_after_body(&mut self, token: Token) -> Action {
        match token {
            Token::Comment(_) | Token::ProcessingInstruction(_) => {
                let point = self.document_end();
                self.insert_markup(&token, Some(point));
                Action::Done
            }
            Token::Doctype(_) => self.dispatch(Mode::InBody, token),
            Token::StartTag(ref tag) if tag.name == "html" => self.dispatch(Mode::InBody, token),
            Token::Characters(text) => {
                let (whitespace, rest) = split_whitespace_prefix(&text);
                if !whitespace.is_empty() {
                    let _ = self.dispatch(Mode::InBody, Token::Characters(whitespace.to_owned()));
                }
                if rest.is_empty() {
                    return Action::Done;
                }
                self.error(codes::UNEXPECTED_CHARACTER);
                self.mode = Mode::InBody;
                Action::Reprocess(Token::Characters(rest.to_owned()))
            }
            Token::Eof => {
                self.stop_parsing();
                Action::Done
            }
            other => {
                self.error(codes::UNEXPECTED_START_TAG);
                self.mode = Mode::InBody;
                Action::Reprocess(other)
            }
        }
    }

    /// <https://html.spec.whatwg.org/multipage/parsing.html#the-after-after-frameset-insertion-mode>
    pub(crate) fn mode_after_after_frameset(&mut self, token: Token) -> Action {
        match token {
            Token::Comment(_) | Token::ProcessingInstruction(_) => {
                let point = self.document_end();
                self.insert_markup(&token, Some(point));
                Action::Done
            }
            Token::Doctype(_) => self.dispatch(Mode::InBody, token),
            Token::StartTag(ref tag) if tag.name == "html" => self.dispatch(Mode::InBody, token),
            Token::Characters(text) => {
                // Each character is a token of its own here: the white space
                // ones go through the "in body" rules and the rest are
                // dropped, so " te st" contributes two spaces and nothing
                // else. Unlike "after after body", this mode does not switch
                // and reprocess, so a run cannot be handled by splitting it in
                // two at the first non-white-space character.
                for ch in text.chars() {
                    if matches!(ch, '\t' | '\n' | '\u{c}' | '\r' | ' ') {
                        let mut buffer = [0_u8; 4];
                        let encoded = ch.encode_utf8(&mut buffer).to_owned();
                        let _ = self.dispatch(Mode::InBody, Token::Characters(encoded));
                    } else {
                        self.error(codes::UNEXPECTED_CHARACTER);
                    }
                }
                Action::Done
            }
            Token::Eof => {
                self.stop_parsing();
                Action::Done
            }
            Token::StartTag(ref tag) if tag.name == "noframes" => {
                self.dispatch(Mode::InHead, token)
            }
            _ => {
                self.error(codes::UNEXPECTED_START_TAG);
                Action::Done
            }
        }
    }
}
