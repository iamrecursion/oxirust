//! The table insertion modes: "in table", "in table text", "in caption", "in
//! column group", "in table body", "in row" and "in cell"
//! ([HTML § 13.2.6.4.9](https://html.spec.whatwg.org/multipage/parsing.html#parsing-main-intable)
//! and following).
//!
//! The distinguishing feature of these modes is *foster parenting*: content
//! that has no business being inside a table is moved out and inserted
//! immediately before the table instead of inside it, which is what browsers do
//! and what pages have come to depend on.

use super::split_whitespace_prefix;
use crate::error::codes;
use crate::token::{Tag, Token};
use crate::tree::{Action, FormatEntry, Mode, Scope, TreeBuilder};

/// The elements a character token may be buffered inside of.
const TABLE_TEXT_PARENTS: &[&str] = &["table", "tbody", "template", "tfoot", "thead", "tr"];

impl TreeBuilder<'_> {
    /// <https://html.spec.whatwg.org/multipage/parsing.html#parsing-main-intable>
    pub(crate) fn mode_in_table(&mut self, token: Token) -> Action {
        match token {
            Token::Characters(_)
                if self
                    .current_node()
                    .is_some_and(|node| node.is_html_any(TABLE_TEXT_PARENTS)) =>
            {
                self.pending_table_text.clear();
                self.pending_table_text_is_whitespace = true;
                self.original_mode = self.mode;
                self.mode = Mode::InTableText;
                Action::Reprocess(token)
            }
            Token::Comment(_) | Token::ProcessingInstruction(_) => {
                self.insert_markup(&token, None);
                Action::Done
            }
            Token::Doctype(_) => {
                self.error(codes::UNEXPECTED_DOCTYPE);
                Action::Done
            }
            Token::StartTag(tag) if tag.name == "caption" => {
                self.clear_stack_to_table_context();
                self.formatting.push(FormatEntry::Marker);
                self.insert_html_element(&tag);
                self.mode = Mode::InCaption;
                Action::Done
            }
            Token::StartTag(tag) if tag.name == "colgroup" => {
                self.clear_stack_to_table_context();
                self.insert_html_element(&tag);
                self.mode = Mode::InColumnGroup;
                Action::Done
            }
            Token::StartTag(ref tag) if tag.name == "col" => {
                self.clear_stack_to_table_context();
                let implied = Tag {
                    name: "colgroup".to_owned(),
                    ..Tag::default()
                };
                self.insert_html_element(&implied);
                self.mode = Mode::InColumnGroup;
                Action::Reprocess(token)
            }
            Token::StartTag(tag) if matches!(tag.name.as_str(), "tbody" | "tfoot" | "thead") => {
                self.clear_stack_to_table_context();
                self.insert_html_element(&tag);
                self.mode = Mode::InTableBody;
                Action::Done
            }
            Token::StartTag(ref tag) if matches!(tag.name.as_str(), "td" | "th" | "tr") => {
                self.clear_stack_to_table_context();
                let implied = Tag {
                    name: "tbody".to_owned(),
                    ..Tag::default()
                };
                self.insert_html_element(&implied);
                self.mode = Mode::InTableBody;
                Action::Reprocess(token)
            }
            Token::StartTag(ref tag) if tag.name == "table" => {
                self.error(codes::UNEXPECTED_START_TAG);
                if !self.has_html_in_scope(Scope::Table, "table") {
                    return Action::Done;
                }
                self.pop_until_html_named("table");
                self.reset_insertion_mode();
                Action::Reprocess(token)
            }
            Token::EndTag(ref tag) if tag.name == "table" => {
                if !self.has_html_in_scope(Scope::Table, "table") {
                    self.error(codes::END_TAG_NOT_IN_SCOPE);
                    return Action::Done;
                }
                self.pop_until_html_named("table");
                self.reset_insertion_mode();
                Action::Done
            }
            Token::EndTag(ref tag)
                if matches!(
                    tag.name.as_str(),
                    "body"
                        | "caption"
                        | "col"
                        | "colgroup"
                        | "html"
                        | "tbody"
                        | "td"
                        | "tfoot"
                        | "th"
                        | "thead"
                        | "tr"
                ) =>
            {
                self.error(codes::UNEXPECTED_END_TAG);
                Action::Done
            }
            Token::StartTag(ref tag)
                if matches!(tag.name.as_str(), "style" | "script" | "template") =>
            {
                self.dispatch(Mode::InHead, token)
            }
            Token::EndTag(ref tag) if tag.name == "template" => self.dispatch(Mode::InHead, token),
            Token::StartTag(ref tag)
                if tag.name == "input"
                    && tag
                        .attribute("type")
                        .is_some_and(|value| value.eq_ignore_ascii_case("hidden")) =>
            {
                self.error(codes::UNEXPECTED_START_TAG);
                let tag = TreeBuilder::take_tag(token);
                self.insert_html_element(&tag);
                self.open.pop();
                Action::Done
            }
            Token::StartTag(ref tag) if tag.name == "form" => {
                self.error(codes::UNEXPECTED_START_TAG);
                if self.has_open_template() || self.form.is_some() {
                    return Action::Done;
                }
                let tag = TreeBuilder::take_tag(token);
                let form = self.insert_html_element(&tag);
                self.form = Some(form);
                self.open.pop();
                Action::Done
            }
            Token::Eof => self.dispatch(Mode::InBody, token),
            other => {
                self.error(codes::FOSTER_PARENTING);
                self.foster_parenting = true;
                let action = self.dispatch(Mode::InBody, other);
                self.foster_parenting = false;
                action
            }
        }
    }

    /// <https://html.spec.whatwg.org/multipage/parsing.html#parsing-main-intabletext>
    pub(crate) fn mode_in_table_text(&mut self, token: Token) -> Action {
        match token {
            Token::Characters(text) => {
                for ch in text.chars() {
                    if ch == '\0' {
                        self.error(codes::UNEXPECTED_NULL_CHARACTER);
                        continue;
                    }
                    if !matches!(ch, '\t' | '\n' | '\u{c}' | '\r' | ' ') {
                        self.pending_table_text_is_whitespace = false;
                    }
                    self.pending_table_text.push(ch);
                }
                Action::Done
            }
            other => {
                let pending = std::mem::take(&mut self.pending_table_text);
                if pending.is_empty() {
                    self.mode = self.original_mode;
                    return Action::Reprocess(other);
                }
                if self.pending_table_text_is_whitespace {
                    self.insert_characters(&pending);
                } else {
                    self.error(codes::FOSTER_PARENTING);
                    self.foster_parenting = true;
                    let _ = self.dispatch(Mode::InBody, Token::Characters(pending));
                    self.foster_parenting = false;
                }
                self.mode = self.original_mode;
                Action::Reprocess(other)
            }
        }
    }

    /// <https://html.spec.whatwg.org/multipage/parsing.html#parsing-main-incaption>
    pub(crate) fn mode_in_caption(&mut self, token: Token) -> Action {
        match token {
            Token::EndTag(ref tag) if tag.name == "caption" => {
                if !self.has_html_in_scope(Scope::Table, "caption") {
                    self.error(codes::END_TAG_NOT_IN_SCOPE);
                    return Action::Done;
                }
                self.generate_implied_end_tags(None);
                if !self.current_is_html("caption") {
                    self.error(codes::UNEXPECTED_END_TAG);
                }
                self.pop_until_html_named("caption");
                self.clear_formatting_to_marker();
                self.mode = Mode::InTable;
                Action::Done
            }
            Token::StartTag(ref tag)
                if matches!(
                    tag.name.as_str(),
                    "caption"
                        | "col"
                        | "colgroup"
                        | "tbody"
                        | "td"
                        | "tfoot"
                        | "th"
                        | "thead"
                        | "tr"
                ) =>
            {
                self.leave_caption(token)
            }
            Token::EndTag(ref tag) if tag.name == "table" => self.leave_caption(token),
            Token::EndTag(ref tag)
                if matches!(
                    tag.name.as_str(),
                    "body"
                        | "col"
                        | "colgroup"
                        | "html"
                        | "tbody"
                        | "td"
                        | "tfoot"
                        | "th"
                        | "thead"
                        | "tr"
                ) =>
            {
                self.error(codes::UNEXPECTED_END_TAG);
                Action::Done
            }
            other => self.dispatch(Mode::InBody, other),
        }
    }

    fn leave_caption(&mut self, token: Token) -> Action {
        if !self.has_html_in_scope(Scope::Table, "caption") {
            self.error(codes::END_TAG_NOT_IN_SCOPE);
            return Action::Done;
        }
        self.generate_implied_end_tags(None);
        if !self.current_is_html("caption") {
            self.error(codes::UNEXPECTED_END_TAG);
        }
        self.pop_until_html_named("caption");
        self.clear_formatting_to_marker();
        self.mode = Mode::InTable;
        Action::Reprocess(token)
    }

    /// <https://html.spec.whatwg.org/multipage/parsing.html#parsing-main-incolgroup>
    pub(crate) fn mode_in_column_group(&mut self, token: Token) -> Action {
        match token {
            Token::Characters(text) => {
                let (whitespace, rest) = split_whitespace_prefix(&text);
                self.insert_characters(whitespace);
                if rest.is_empty() {
                    return Action::Done;
                }
                self.leave_column_group(Token::Characters(rest.to_owned()))
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
            Token::StartTag(tag) if tag.name == "col" => {
                self.insert_html_element(&tag);
                self.open.pop();
                Action::Done
            }
            Token::EndTag(ref tag) if tag.name == "colgroup" => {
                if !self.current_is_html("colgroup") {
                    self.error(codes::UNEXPECTED_END_TAG);
                    return Action::Done;
                }
                self.open.pop();
                self.mode = Mode::InTable;
                Action::Done
            }
            Token::EndTag(ref tag) if tag.name == "col" => {
                self.error(codes::UNEXPECTED_END_TAG);
                Action::Done
            }
            Token::StartTag(ref tag) if tag.name == "template" => {
                self.dispatch(Mode::InHead, token)
            }
            Token::EndTag(ref tag) if tag.name == "template" => self.dispatch(Mode::InHead, token),
            Token::Eof => self.dispatch(Mode::InBody, token),
            other => self.leave_column_group(other),
        }
    }

    fn leave_column_group(&mut self, token: Token) -> Action {
        if !self.current_is_html("colgroup") {
            self.error(codes::UNEXPECTED_END_TAG);
            return Action::Done;
        }
        self.open.pop();
        self.mode = Mode::InTable;
        Action::Reprocess(token)
    }

    /// <https://html.spec.whatwg.org/multipage/parsing.html#parsing-main-intbody>
    pub(crate) fn mode_in_table_body(&mut self, token: Token) -> Action {
        match token {
            Token::StartTag(tag) if tag.name == "tr" => {
                self.clear_stack_to_table_body_context();
                self.insert_html_element(&tag);
                self.mode = Mode::InRow;
                Action::Done
            }
            Token::StartTag(ref tag) if matches!(tag.name.as_str(), "th" | "td") => {
                self.error(codes::UNEXPECTED_START_TAG);
                self.clear_stack_to_table_body_context();
                let implied = Tag {
                    name: "tr".to_owned(),
                    ..Tag::default()
                };
                self.insert_html_element(&implied);
                self.mode = Mode::InRow;
                Action::Reprocess(token)
            }
            Token::EndTag(ref tag) if matches!(tag.name.as_str(), "tbody" | "tfoot" | "thead") => {
                if !self.has_html_in_scope(Scope::Table, &tag.name) {
                    self.error(codes::END_TAG_NOT_IN_SCOPE);
                    return Action::Done;
                }
                self.clear_stack_to_table_body_context();
                self.open.pop();
                self.mode = Mode::InTable;
                Action::Done
            }
            Token::StartTag(ref tag)
                if matches!(
                    tag.name.as_str(),
                    "caption" | "col" | "colgroup" | "tbody" | "tfoot" | "thead"
                ) =>
            {
                self.leave_table_body(token)
            }
            Token::EndTag(ref tag) if tag.name == "table" => self.leave_table_body(token),
            Token::EndTag(ref tag)
                if matches!(
                    tag.name.as_str(),
                    "body" | "caption" | "col" | "colgroup" | "html" | "td" | "th" | "tr"
                ) =>
            {
                self.error(codes::UNEXPECTED_END_TAG);
                Action::Done
            }
            other => self.dispatch(Mode::InTable, other),
        }
    }

    fn leave_table_body(&mut self, token: Token) -> Action {
        if !self.has_any_html_in_scope(Scope::Table, &["tbody", "thead", "tfoot"]) {
            self.error(codes::END_TAG_NOT_IN_SCOPE);
            return Action::Done;
        }
        self.clear_stack_to_table_body_context();
        self.open.pop();
        self.mode = Mode::InTable;
        Action::Reprocess(token)
    }

    /// <https://html.spec.whatwg.org/multipage/parsing.html#parsing-main-intr>
    pub(crate) fn mode_in_row(&mut self, token: Token) -> Action {
        match token {
            Token::StartTag(tag) if matches!(tag.name.as_str(), "th" | "td") => {
                self.clear_stack_to_table_row_context();
                self.insert_html_element(&tag);
                self.mode = Mode::InCell;
                self.formatting.push(FormatEntry::Marker);
                Action::Done
            }
            Token::EndTag(ref tag) if tag.name == "tr" => {
                if !self.has_html_in_scope(Scope::Table, "tr") {
                    self.error(codes::END_TAG_NOT_IN_SCOPE);
                    return Action::Done;
                }
                self.clear_stack_to_table_row_context();
                self.open.pop();
                self.mode = Mode::InTableBody;
                Action::Done
            }
            Token::StartTag(ref tag)
                if matches!(
                    tag.name.as_str(),
                    "caption" | "col" | "colgroup" | "tbody" | "tfoot" | "thead" | "tr"
                ) =>
            {
                self.leave_row(token)
            }
            Token::EndTag(ref tag) if tag.name == "table" => self.leave_row(token),
            Token::EndTag(ref tag) if matches!(tag.name.as_str(), "tbody" | "tfoot" | "thead") => {
                if !self.has_html_in_scope(Scope::Table, &tag.name) {
                    self.error(codes::END_TAG_NOT_IN_SCOPE);
                    return Action::Done;
                }
                if !self.has_html_in_scope(Scope::Table, "tr") {
                    return Action::Done;
                }
                self.clear_stack_to_table_row_context();
                self.open.pop();
                self.mode = Mode::InTableBody;
                Action::Reprocess(token)
            }
            Token::EndTag(ref tag)
                if matches!(
                    tag.name.as_str(),
                    "body" | "caption" | "col" | "colgroup" | "html" | "td" | "th"
                ) =>
            {
                self.error(codes::UNEXPECTED_END_TAG);
                Action::Done
            }
            other => self.dispatch(Mode::InTable, other),
        }
    }

    fn leave_row(&mut self, token: Token) -> Action {
        if !self.has_html_in_scope(Scope::Table, "tr") {
            self.error(codes::END_TAG_NOT_IN_SCOPE);
            return Action::Done;
        }
        self.clear_stack_to_table_row_context();
        self.open.pop();
        self.mode = Mode::InTableBody;
        Action::Reprocess(token)
    }

    /// <https://html.spec.whatwg.org/multipage/parsing.html#parsing-main-intd>
    pub(crate) fn mode_in_cell(&mut self, token: Token) -> Action {
        match token {
            Token::EndTag(ref tag) if matches!(tag.name.as_str(), "td" | "th") => {
                if !self.has_html_in_scope(Scope::Table, &tag.name) {
                    self.error(codes::END_TAG_NOT_IN_SCOPE);
                    return Action::Done;
                }
                self.generate_implied_end_tags(None);
                if !self.current_is_html(&tag.name) {
                    self.error(codes::UNEXPECTED_END_TAG);
                }
                let name = tag.name.clone();
                self.pop_until_html_named(&name);
                self.clear_formatting_to_marker();
                self.mode = Mode::InRow;
                Action::Done
            }
            Token::StartTag(ref tag)
                if matches!(
                    tag.name.as_str(),
                    "caption"
                        | "col"
                        | "colgroup"
                        | "tbody"
                        | "td"
                        | "tfoot"
                        | "th"
                        | "thead"
                        | "tr"
                ) =>
            {
                if !self.has_any_html_in_scope(Scope::Table, &["td", "th"]) {
                    self.error(codes::UNEXPECTED_START_TAG);
                    return Action::Done;
                }
                self.close_cell();
                Action::Reprocess(token)
            }
            Token::EndTag(ref tag)
                if matches!(
                    tag.name.as_str(),
                    "body" | "caption" | "col" | "colgroup" | "html"
                ) =>
            {
                self.error(codes::UNEXPECTED_END_TAG);
                Action::Done
            }
            Token::EndTag(ref tag)
                if matches!(
                    tag.name.as_str(),
                    "table" | "tbody" | "tfoot" | "thead" | "tr"
                ) =>
            {
                if !self.has_html_in_scope(Scope::Table, &tag.name) {
                    self.error(codes::END_TAG_NOT_IN_SCOPE);
                    return Action::Done;
                }
                self.close_cell();
                Action::Reprocess(token)
            }
            other => self.dispatch(Mode::InBody, other),
        }
    }

    /// "Close the cell".
    fn close_cell(&mut self) {
        self.generate_implied_end_tags(None);
        if !(self.current_is_html("td") || self.current_is_html("th")) {
            self.error(codes::UNEXPECTED_END_TAG);
        }
        self.pop_until_any_html_named(&["td", "th"]);
        self.clear_formatting_to_marker();
        self.mode = Mode::InRow;
    }
}
