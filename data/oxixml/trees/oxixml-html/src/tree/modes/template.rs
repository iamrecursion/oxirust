//! The "in template" insertion mode
//! ([HTML § 13.2.6.4.16](https://html.spec.whatwg.org/multipage/parsing.html#parsing-main-intemplate)).
//!
//! A `<template>` element's contents are parsed into a separate document
//! fragment, and the insertion mode inside it is chosen by the first
//! table-structure tag that turns up, which is what the stack of template
//! insertion modes records.

use crate::error::codes;
use crate::token::Token;
use crate::tree::{Action, Mode, TreeBuilder};

impl TreeBuilder<'_> {
    /// <https://html.spec.whatwg.org/multipage/parsing.html#parsing-main-intemplate>
    pub(crate) fn mode_in_template(&mut self, token: Token) -> Action {
        match token {
            Token::Characters(_)
            | Token::Comment(_)
            | Token::ProcessingInstruction(_)
            | Token::Doctype(_) => self.dispatch(Mode::InBody, token),
            Token::StartTag(ref tag)
                if matches!(
                    tag.name.as_str(),
                    "base"
                        | "basefont"
                        | "bgsound"
                        | "link"
                        | "meta"
                        | "noframes"
                        | "script"
                        | "style"
                        | "template"
                        | "title"
                ) =>
            {
                self.dispatch(Mode::InHead, token)
            }
            Token::EndTag(ref tag) if tag.name == "template" => self.dispatch(Mode::InHead, token),
            Token::StartTag(ref tag)
                if matches!(
                    tag.name.as_str(),
                    "caption" | "colgroup" | "tbody" | "tfoot" | "thead"
                ) =>
            {
                self.switch_template_mode(Mode::InTable, token)
            }
            Token::StartTag(ref tag) if tag.name == "col" => {
                self.switch_template_mode(Mode::InColumnGroup, token)
            }
            Token::StartTag(ref tag) if tag.name == "tr" => {
                self.switch_template_mode(Mode::InTableBody, token)
            }
            Token::StartTag(ref tag) if matches!(tag.name.as_str(), "td" | "th") => {
                self.switch_template_mode(Mode::InRow, token)
            }
            Token::StartTag(_) => self.switch_template_mode(Mode::InBody, token),
            Token::EndTag(_) => {
                self.error(codes::UNEXPECTED_END_TAG);
                Action::Done
            }
            Token::Eof => {
                if !self.has_open_template() {
                    self.stop_parsing();
                    return Action::Done;
                }
                self.error(codes::UNEXPECTED_EOF);
                self.pop_until_html_named("template");
                self.clear_formatting_to_marker();
                self.template_modes.pop();
                self.reset_insertion_mode();
                Action::Reprocess(Token::Eof)
            }
        }
    }

    fn switch_template_mode(&mut self, mode: Mode, token: Token) -> Action {
        self.template_modes.pop();
        self.template_modes.push(mode);
        self.mode = mode;
        Action::Reprocess(token)
    }
}
