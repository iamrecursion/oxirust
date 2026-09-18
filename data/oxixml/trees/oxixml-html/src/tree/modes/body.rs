//! The "in body" insertion mode
//! ([HTML § 13.2.6.4.7](https://html.spec.whatwg.org/multipage/parsing.html#parsing-main-inbody)),
//! by far the largest of the modes: it is where ordinary flow content is
//! handled, and therefore where implied end tags, the formatting elements and
//! the adoption agency algorithm all come into play.

use oxixml_dom::NodeId;

use crate::error::codes;
use crate::ns::Namespace;
use crate::token::{Attribute, Tag, Token};
use crate::tokenizer::ContentModel;
use crate::tree::foreign::{adjust_attributes, adjust_svg_tag_name};
use crate::tree::{Action, FormatEntry, Mode, Scope, TreeBuilder};

/// The block-level elements that "in body" opens after closing an open `p`.
const BLOCK_ELEMENTS: &[&str] = &[
    "address",
    "article",
    "aside",
    "blockquote",
    "center",
    "details",
    "dialog",
    "dir",
    "div",
    "dl",
    "fieldset",
    "figcaption",
    "figure",
    "footer",
    "header",
    "hgroup",
    "main",
    "menu",
    "nav",
    "ol",
    "p",
    "search",
    "section",
    "summary",
    "ul",
];

/// The elements whose end tag closes a block, which is the block list plus the
/// few extra names the standard lists separately.
const BLOCK_END_ELEMENTS: &[&str] = &[
    "address",
    "article",
    "aside",
    "blockquote",
    "button",
    "center",
    "details",
    "dialog",
    "dir",
    "div",
    "dl",
    "fieldset",
    "figcaption",
    "figure",
    "footer",
    "header",
    "hgroup",
    "listing",
    "main",
    "menu",
    "nav",
    "ol",
    "pre",
    "search",
    "section",
    "select",
    "summary",
    "ul",
];

/// The headings.
const HEADINGS: &[&str] = &["h1", "h2", "h3", "h4", "h5", "h6"];

/// The elements that may still be open when the document ends without it being
/// a parse error.
const IMPLIED_AT_EOF: &[&str] = &[
    "dd", "dt", "li", "optgroup", "option", "p", "rb", "rp", "rt", "rtc", "tbody", "td", "tfoot",
    "th", "thead", "tr", "body", "html",
];

impl TreeBuilder<'_> {
    /// <https://html.spec.whatwg.org/multipage/parsing.html#parsing-main-inbody>
    #[allow(clippy::too_many_lines)]
    pub(crate) fn mode_in_body(&mut self, token: Token) -> Action {
        match token {
            Token::Characters(text) => {
                for ch in text.chars() {
                    if ch == '\0' {
                        self.error(codes::UNEXPECTED_NULL_CHARACTER);
                        continue;
                    }
                    self.reconstruct_formatting_elements();
                    let mut buffer = [0_u8; 4];
                    let encoded = ch.encode_utf8(&mut buffer).to_owned();
                    self.insert_characters(&encoded);
                    if !matches!(ch, '\t' | '\n' | '\u{c}' | '\r' | ' ') {
                        self.frameset_ok = false;
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
            Token::Eof => {
                if !self.template_modes.is_empty() {
                    return self.dispatch(Mode::InTemplate, Token::Eof);
                }
                if self
                    .open
                    .iter()
                    .any(|node| !node.is_html_any(IMPLIED_AT_EOF))
                {
                    self.error(codes::UNEXPECTED_EOF);
                }
                self.stop_parsing();
                Action::Done
            }
            Token::StartTag(tag) => self.in_body_start_tag(tag),
            Token::EndTag(tag) => self.in_body_end_tag(tag),
        }
    }

    #[allow(clippy::too_many_lines)]
    fn in_body_start_tag(&mut self, tag: Tag) -> Action {
        match tag.name.as_str() {
            "html" => {
                self.error(codes::UNEXPECTED_START_TAG);
                if !self.has_open_template() {
                    if let Some(root) = self.open.first().map(|node| node.id) {
                        self.add_missing_attributes(root, &tag.attributes);
                    }
                }
                Action::Done
            }
            "base" | "basefont" | "bgsound" | "link" | "meta" | "noframes" | "script" | "style"
            | "template" | "title" => self.dispatch(Mode::InHead, Token::StartTag(tag)),
            "body" => {
                self.error(codes::UNEXPECTED_START_TAG);
                let body = self
                    .open
                    .get(1)
                    .filter(|node| node.is_html("body"))
                    .map(|node| node.id);
                match body {
                    Some(body) if !self.has_open_template() => {
                        self.frameset_ok = false;
                        self.add_missing_attributes(body, &tag.attributes);
                    }
                    _ => {}
                }
                Action::Done
            }
            "frameset" => {
                self.error(codes::UNEXPECTED_START_TAG);
                let body = self
                    .open
                    .get(1)
                    .filter(|node| node.is_html("body"))
                    .map(|node| node.id);
                let Some(body) = body else {
                    return Action::Done;
                };
                if !self.frameset_ok {
                    return Action::Done;
                }
                if let Some(parent) = self.document.parent(body) {
                    let result = self.document.remove_child(parent, body);
                    debug_assert!(
                        result.is_ok(),
                        "`parent` was just looked up as `body`'s own parent on the line above, \
                         so `body` is a child of `parent` at this exact point: {result:?}"
                    );
                }
                self.open.truncate(1);
                self.insert_html_element(&tag);
                self.mode = Mode::InFrameset;
                Action::Done
            }
            name if BLOCK_ELEMENTS.contains(&name) => {
                if self.has_html_in_scope(Scope::Button, "p") {
                    self.close_p_element();
                }
                self.insert_html_element(&tag);
                Action::Done
            }
            name if HEADINGS.contains(&name) => {
                if self.has_html_in_scope(Scope::Button, "p") {
                    self.close_p_element();
                }
                if self
                    .current_node()
                    .is_some_and(|node| node.is_html_any(HEADINGS))
                {
                    self.error(codes::UNEXPECTED_START_TAG);
                    self.open.pop();
                }
                self.insert_html_element(&tag);
                Action::Done
            }
            "pre" | "listing" => {
                if self.has_html_in_scope(Scope::Button, "p") {
                    self.close_p_element();
                }
                self.insert_html_element(&tag);
                self.ignore_next_lf = true;
                self.frameset_ok = false;
                Action::Done
            }
            "form" => {
                if self.form.is_some() && !self.has_open_template() {
                    self.error(codes::UNEXPECTED_START_TAG);
                    return Action::Done;
                }
                if self.has_html_in_scope(Scope::Button, "p") {
                    self.close_p_element();
                }
                let form = self.insert_html_element(&tag);
                if !self.has_open_template() {
                    self.form = Some(form);
                }
                Action::Done
            }
            "li" => {
                self.close_list_item(&["li"]);
                if self.has_html_in_scope(Scope::Button, "p") {
                    self.close_p_element();
                }
                self.insert_html_element(&tag);
                Action::Done
            }
            "dd" | "dt" => {
                self.close_list_item(&["dd", "dt"]);
                if self.has_html_in_scope(Scope::Button, "p") {
                    self.close_p_element();
                }
                self.insert_html_element(&tag);
                Action::Done
            }
            "plaintext" => {
                if self.has_html_in_scope(Scope::Button, "p") {
                    self.close_p_element();
                }
                self.insert_html_element(&tag);
                self.tokenizer.set_content_model(ContentModel::PlainText);
                Action::Done
            }
            "button" => {
                if self.has_html_in_scope(Scope::Default, "button") {
                    self.error(codes::UNEXPECTED_START_TAG);
                    self.generate_implied_end_tags(None);
                    self.pop_until_html_named("button");
                }
                self.reconstruct_formatting_elements();
                self.insert_html_element(&tag);
                self.frameset_ok = false;
                Action::Done
            }
            "a" => {
                if let Some(index) = self.formatting_entry_named("a") {
                    self.error(codes::MISNESTED_TAG);
                    let id = self.formatting[index].id();
                    if !self.adoption_agency("a") {
                        self.any_other_end_tag("a");
                    }
                    if let Some(id) = id {
                        if let Some(index) = self.formatting_index_of(id) {
                            self.formatting.remove(index);
                        }
                        self.remove_from_stack(id);
                    }
                }
                self.reconstruct_formatting_elements();
                let element = self.insert_html_element(&tag);
                self.push_formatting_element(element, tag);
                Action::Done
            }
            "b" | "big" | "code" | "em" | "font" | "i" | "s" | "small" | "strike" | "strong"
            | "tt" | "u" => {
                self.reconstruct_formatting_elements();
                let element = self.insert_html_element(&tag);
                self.push_formatting_element(element, tag);
                Action::Done
            }
            "nobr" => {
                self.reconstruct_formatting_elements();
                if self.has_html_in_scope(Scope::Default, "nobr") {
                    self.error(codes::MISNESTED_TAG);
                    if !self.adoption_agency("nobr") {
                        self.any_other_end_tag("nobr");
                    }
                    self.reconstruct_formatting_elements();
                }
                let element = self.insert_html_element(&tag);
                self.push_formatting_element(element, tag);
                Action::Done
            }
            "applet" | "marquee" | "object" => {
                self.reconstruct_formatting_elements();
                self.insert_html_element(&tag);
                self.formatting.push(FormatEntry::Marker);
                self.frameset_ok = false;
                Action::Done
            }
            "table" => {
                if !self.quirks.is_quirks() && self.has_html_in_scope(Scope::Button, "p") {
                    self.close_p_element();
                }
                self.insert_html_element(&tag);
                self.frameset_ok = false;
                self.mode = Mode::InTable;
                Action::Done
            }
            "area" | "br" | "embed" | "img" | "keygen" | "wbr" => {
                self.reconstruct_formatting_elements();
                self.insert_html_element(&tag);
                self.open.pop();
                self.frameset_ok = false;
                Action::Done
            }
            "input" => {
                if self.context_is_html("select") {
                    self.error(codes::UNEXPECTED_START_TAG);
                    return Action::Done;
                }
                if self.has_html_in_scope(Scope::Default, "select") {
                    self.error(codes::UNEXPECTED_START_TAG);
                    self.pop_until_html_named("select");
                }
                self.reconstruct_formatting_elements();
                self.insert_html_element(&tag);
                self.open.pop();
                if !tag
                    .attribute("type")
                    .is_some_and(|value| value.eq_ignore_ascii_case("hidden"))
                {
                    self.frameset_ok = false;
                }
                Action::Done
            }
            "param" | "source" | "track" => {
                self.insert_html_element(&tag);
                self.open.pop();
                Action::Done
            }
            "hr" => {
                if self.has_html_in_scope(Scope::Button, "p") {
                    self.close_p_element();
                }
                if self.has_html_in_scope(Scope::Default, "select") {
                    self.generate_implied_end_tags(None);
                    if self.has_html_in_scope(Scope::Default, "option")
                        || self.has_html_in_scope(Scope::Default, "optgroup")
                    {
                        self.error(codes::UNEXPECTED_START_TAG);
                    }
                }
                self.insert_html_element(&tag);
                self.open.pop();
                self.frameset_ok = false;
                Action::Done
            }
            "image" => {
                self.error(codes::UNEXPECTED_START_TAG);
                let mut tag = tag;
                tag.name = "img".to_owned();
                Action::Reprocess(Token::StartTag(tag))
            }
            "textarea" => {
                self.insert_html_element(&tag);
                self.ignore_next_lf = true;
                self.tokenizer.set_content_model(ContentModel::Rcdata);
                self.original_mode = self.mode;
                self.frameset_ok = false;
                self.mode = Mode::Text;
                Action::Done
            }
            "xmp" => {
                if self.has_html_in_scope(Scope::Button, "p") {
                    self.close_p_element();
                }
                self.reconstruct_formatting_elements();
                self.frameset_ok = false;
                self.parse_text_element(&tag, ContentModel::RawText);
                Action::Done
            }
            "iframe" => {
                self.frameset_ok = false;
                self.parse_text_element(&tag, ContentModel::RawText);
                Action::Done
            }
            "noembed" => {
                self.parse_text_element(&tag, ContentModel::RawText);
                Action::Done
            }
            "noscript" if self.scripting => {
                self.parse_text_element(&tag, ContentModel::RawText);
                Action::Done
            }
            "select" => {
                if self.context_is_html("select") {
                    self.error(codes::UNEXPECTED_START_TAG);
                    return Action::Done;
                }
                if self.has_html_in_scope(Scope::Default, "select") {
                    self.error(codes::UNEXPECTED_START_TAG);
                    self.pop_until_html_named("select");
                    return Action::Done;
                }
                self.reconstruct_formatting_elements();
                self.insert_html_element(&tag);
                self.frameset_ok = false;
                Action::Done
            }
            "option" => {
                if self.has_html_in_scope(Scope::Default, "select") {
                    self.generate_implied_end_tags(Some("optgroup"));
                    if self.has_html_in_scope(Scope::Default, "option") {
                        self.error(codes::UNEXPECTED_START_TAG);
                    }
                } else if self.current_is_html("option") {
                    self.pop_open_element();
                }
                self.reconstruct_formatting_elements();
                let option = self.insert_html_element(&tag);
                // "Update an option's nearest ancestor select" (HTML § the
                // `option` element): this is the reaction the standard's
                // generic node-insertion steps run for a freshly inserted
                // `option`, which is what lets a later `option` (or one
                // marked `selected`) supersede this one as its `select`'s
                // selection.
                self.note_option_inserted(option);
                Action::Done
            }
            "optgroup" => {
                if self.has_html_in_scope(Scope::Default, "select") {
                    self.generate_implied_end_tags(None);
                    if self.has_html_in_scope(Scope::Default, "option")
                        || self.has_html_in_scope(Scope::Default, "optgroup")
                    {
                        self.error(codes::UNEXPECTED_START_TAG);
                    }
                } else if self.current_is_html("option") {
                    self.pop_open_element();
                }
                self.reconstruct_formatting_elements();
                self.insert_html_element(&tag);
                Action::Done
            }
            "selectedcontent" => {
                self.reconstruct_formatting_elements();
                let selectedcontent = self.insert_html_element(&tag);
                // The `selectedcontent` element's "HTML element
                // post-connection steps" (HTML § the `selectedcontent`
                // element): this parser inserts one node at a time, so the
                // reaction can run synchronously right here rather than
                // through a separately queued step.
                self.note_selectedcontent_inserted(selectedcontent);
                Action::Done
            }
            "rb" | "rtc" => {
                if self.has_html_in_scope(Scope::Default, "ruby") {
                    self.generate_implied_end_tags(None);
                    if !self.current_is_html("ruby") {
                        self.error(codes::UNEXPECTED_START_TAG);
                    }
                }
                self.insert_html_element(&tag);
                Action::Done
            }
            "rp" | "rt" => {
                if self.has_html_in_scope(Scope::Default, "ruby") {
                    self.generate_implied_end_tags(Some("rtc"));
                    if !(self.current_is_html("ruby") || self.current_is_html("rtc")) {
                        self.error(codes::UNEXPECTED_START_TAG);
                    }
                }
                self.insert_html_element(&tag);
                Action::Done
            }
            "math" => {
                self.reconstruct_formatting_elements();
                let attributes = adjust_attributes(&tag.attributes, Namespace::MathMl);
                self.insert_element(&tag, Namespace::MathMl, attributes);
                if tag.self_closing {
                    self.open.pop();
                }
                Action::Done
            }
            "svg" => {
                self.reconstruct_formatting_elements();
                let mut tag = tag;
                tag.name = adjust_svg_tag_name(&tag.name).to_owned();
                let attributes = adjust_attributes(&tag.attributes, Namespace::Svg);
                self.insert_element(&tag, Namespace::Svg, attributes);
                if tag.self_closing {
                    self.open.pop();
                }
                Action::Done
            }
            "caption" | "col" | "colgroup" | "frame" | "head" | "tbody" | "td" | "tfoot" | "th"
            | "thead" | "tr" => {
                self.error(codes::UNEXPECTED_START_TAG);
                Action::Done
            }
            _ => {
                self.reconstruct_formatting_elements();
                self.insert_html_element(&tag);
                Action::Done
            }
        }
    }

    #[allow(clippy::too_many_lines)]
    fn in_body_end_tag(&mut self, tag: Tag) -> Action {
        match tag.name.as_str() {
            "template" => self.dispatch(Mode::InHead, Token::EndTag(tag)),
            "body" => {
                if !self.has_html_in_scope(Scope::Default, "body") {
                    self.error(codes::END_TAG_NOT_IN_SCOPE);
                    return Action::Done;
                }
                self.check_open_elements_at_end();
                self.mode = Mode::AfterBody;
                Action::Done
            }
            "html" => {
                if !self.has_html_in_scope(Scope::Default, "body") {
                    self.error(codes::END_TAG_NOT_IN_SCOPE);
                    return Action::Done;
                }
                self.check_open_elements_at_end();
                self.mode = Mode::AfterBody;
                Action::Reprocess(Token::EndTag(tag))
            }
            name if BLOCK_END_ELEMENTS.contains(&name) => {
                if !self.has_html_in_scope(Scope::Default, name) {
                    self.error(codes::END_TAG_NOT_IN_SCOPE);
                    return Action::Done;
                }
                self.generate_implied_end_tags(None);
                if !self.current_is_html(name) {
                    self.error(codes::UNEXPECTED_END_TAG);
                }
                self.pop_until_html_named(name);
                Action::Done
            }
            "form" => {
                if self.has_open_template() {
                    if !self.has_html_in_scope(Scope::Default, "form") {
                        self.error(codes::END_TAG_NOT_IN_SCOPE);
                        return Action::Done;
                    }
                    self.generate_implied_end_tags(None);
                    if !self.current_is_html("form") {
                        self.error(codes::UNEXPECTED_END_TAG);
                    }
                    self.pop_until_html_named("form");
                    return Action::Done;
                }
                let node = self.form.take();
                let Some(node) = node.filter(|id| self.has_node_in_scope(Scope::Default, *id))
                else {
                    self.error(codes::END_TAG_NOT_IN_SCOPE);
                    return Action::Done;
                };
                self.generate_implied_end_tags(None);
                if self.current_node().map(|open| open.id) != Some(node) {
                    self.error(codes::UNEXPECTED_END_TAG);
                }
                self.remove_from_stack(node);
                Action::Done
            }
            "p" => {
                if !self.has_html_in_scope(Scope::Button, "p") {
                    self.error(codes::END_TAG_NOT_IN_SCOPE);
                    let implied = Tag {
                        name: "p".to_owned(),
                        ..Tag::default()
                    };
                    self.insert_html_element(&implied);
                }
                self.close_p_element();
                Action::Done
            }
            "li" => {
                if !self.has_html_in_scope(Scope::ListItem, "li") {
                    self.error(codes::END_TAG_NOT_IN_SCOPE);
                    return Action::Done;
                }
                self.generate_implied_end_tags(Some("li"));
                if !self.current_is_html("li") {
                    self.error(codes::UNEXPECTED_END_TAG);
                }
                self.pop_until_html_named("li");
                Action::Done
            }
            name @ ("dd" | "dt") => {
                if !self.has_html_in_scope(Scope::Default, name) {
                    self.error(codes::END_TAG_NOT_IN_SCOPE);
                    return Action::Done;
                }
                self.generate_implied_end_tags(Some(name));
                if !self.current_is_html(name) {
                    self.error(codes::UNEXPECTED_END_TAG);
                }
                self.pop_until_html_named(name);
                Action::Done
            }
            name if HEADINGS.contains(&name) => {
                if !self.has_any_html_in_scope(Scope::Default, HEADINGS) {
                    self.error(codes::END_TAG_NOT_IN_SCOPE);
                    return Action::Done;
                }
                self.generate_implied_end_tags(None);
                if !self.current_is_html(name) {
                    self.error(codes::UNEXPECTED_END_TAG);
                }
                self.pop_until_any_html_named(HEADINGS);
                Action::Done
            }
            name if crate::ns::FORMATTING_ELEMENTS.contains(&name) => {
                if self.adoption_agency(&tag.name) {
                    Action::Done
                } else {
                    self.any_other_end_tag(&tag.name)
                }
            }
            "applet" | "marquee" | "object" => {
                if !self.has_html_in_scope(Scope::Default, &tag.name) {
                    self.error(codes::END_TAG_NOT_IN_SCOPE);
                    return Action::Done;
                }
                self.generate_implied_end_tags(None);
                if !self.current_is_html(&tag.name) {
                    self.error(codes::UNEXPECTED_END_TAG);
                }
                self.pop_until_html_named(&tag.name);
                self.clear_formatting_to_marker();
                Action::Done
            }
            "br" => {
                self.error(codes::UNEXPECTED_END_TAG);
                let implied = Tag {
                    name: "br".to_owned(),
                    ..Tag::default()
                };
                self.reconstruct_formatting_elements();
                self.insert_html_element(&implied);
                self.open.pop();
                self.frameset_ok = false;
                Action::Done
            }
            name => self.any_other_end_tag(name),
        }
    }

    /// The "any other end tag" steps of the "in body" insertion mode.
    pub(crate) fn any_other_end_tag(&mut self, name: &str) -> Action {
        let mut index = self.open.len();
        while index > 0 {
            index -= 1;
            let node = self.open[index].clone();
            if node.is_html(name) {
                self.generate_implied_end_tags(Some(name));
                if self.current_node().map(|current| current.id) != Some(node.id) {
                    self.error(codes::UNEXPECTED_END_TAG);
                }
                self.pop_until_id(node.id);
                return Action::Done;
            }
            if crate::ns::is_special(node.namespace, &node.name) {
                self.error(codes::UNEXPECTED_END_TAG);
                return Action::Done;
            }
        }
        Action::Done
    }

    /// The shared "if there is a node on the stack that is not one of the
    /// implied ones, this is a parse error" check of the `body` and `html` end
    /// tags.
    fn check_open_elements_at_end(&mut self) {
        if self
            .open
            .iter()
            .any(|node| !node.is_html_any(IMPLIED_AT_EOF))
        {
            self.error(codes::UNEXPECTED_END_TAG);
        }
    }

    /// The shared loop of the `li`, `dd` and `dt` start tags, which closes an
    /// open list item of the same kind.
    fn close_list_item(&mut self, names: &[&str]) {
        self.frameset_ok = false;
        let mut index = self.open.len();
        while index > 0 {
            index -= 1;
            let node = self.open[index].clone();
            if node.is_html_any(names) {
                self.generate_implied_end_tags(Some(&node.name));
                if !self.current_is_html(&node.name) {
                    self.error(codes::UNEXPECTED_START_TAG);
                }
                let name = node.name.clone();
                self.pop_until_html_named(&name);
                return;
            }
            if crate::ns::is_special(node.namespace, &node.name)
                && !node.is_html_any(&["address", "div", "p"])
            {
                return;
            }
        }
    }

    /// Copies the attributes a token carries onto an element, skipping the ones
    /// the element already has — the rule the repeated `<html>` and `<body>`
    /// start tags follow.
    fn add_missing_attributes(&mut self, element: NodeId, attributes: &[Attribute]) {
        for attribute in attributes {
            let name = crate::tree::dom::attribute_name(&attribute.name);
            let rewritten = matches!(name, std::borrow::Cow::Owned(_));
            if self.document.has_attribute(element, name.as_ref()) {
                continue;
            }
            let value = crate::tree::dom::character_data(&attribute.value);
            if let Ok(node) =
                self.document
                    .set_attribute(element, name.as_ref(), value.into_owned())
            {
                if rewritten {
                    self.record_original(node, &attribute.name);
                }
            }
        }
    }
}
