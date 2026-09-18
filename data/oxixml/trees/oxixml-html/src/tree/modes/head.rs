//! The insertion modes that run before the body: "initial", "before html",
//! "before head", "in head", "in head noscript" and "after head", plus the
//! "text" mode that raw text and RCDATA elements switch into.

use super::split_whitespace_prefix;
use crate::error::codes;
use crate::ns::Namespace;
use crate::quirks::quirks_mode_for;
use crate::token::{Tag, Token};
use crate::tokenizer::ContentModel;
use crate::tree::{Action, Mode, OpenElement, TreeBuilder};

impl TreeBuilder<'_> {
    /// <https://html.spec.whatwg.org/multipage/parsing.html#the-initial-insertion-mode>
    pub(crate) fn mode_initial(&mut self, token: Token) -> Action {
        match token {
            Token::Characters(text) => {
                let (_, rest) = split_whitespace_prefix(&text);
                if rest.is_empty() {
                    return Action::Done;
                }
                self.anything_else_initial(Token::Characters(rest.to_owned()))
            }
            Token::Comment(_) | Token::ProcessingInstruction(_) => {
                let point = self.document_end();
                self.insert_markup(&token, Some(point));
                Action::Done
            }
            Token::Doctype(doctype) => {
                if doctype.name.as_deref() != Some("html")
                    || doctype.public_id.is_some()
                    || doctype
                        .system_id
                        .as_deref()
                        .is_some_and(|id| id != "about:legacy-compat")
                {
                    self.error(codes::UNEXPECTED_DOCTYPE);
                }
                self.append_doctype(&doctype);
                self.quirks = quirks_mode_for(&doctype, self.fragment);
                self.mode = Mode::BeforeHtml;
                Action::Done
            }
            other => self.anything_else_initial(other),
        }
    }

    fn anything_else_initial(&mut self, token: Token) -> Action {
        self.error(codes::EXPECTED_DOCTYPE);
        if !self.fragment {
            self.quirks = crate::quirks::QuirksMode::Quirks;
        }
        self.mode = Mode::BeforeHtml;
        Action::Reprocess(token)
    }

    /// Appends the document type node, when its name can be represented in an
    /// XML DOM at all.
    fn append_doctype(&mut self, doctype: &crate::token::Doctype) {
        let name = doctype.name.as_deref().unwrap_or("");
        let xml_name = crate::tree::dom::doctype_name(name);
        let rewritten = xml_name != name;
        if rewritten {
            self.error(codes::UNREPRESENTABLE_NAME);
        }
        let node = self.document.create_document_type(
            &xml_name,
            doctype.public_id.as_deref(),
            doctype.system_id.as_deref(),
        );
        match node {
            Ok(node) => {
                if rewritten {
                    self.record_original(node, name);
                }
                let root = self.document.root();
                let result = self.document.append_child(root, node);
                debug_assert!(
                    result.is_ok(),
                    "append_doctype only ever runs once per parse (the \"initial\" insertion \
                     mode it is called from always leaves for \"before html\" right after, and \
                     never returns), and always before any document element exists, so the \
                     document-type cardinality and ordering constraints cannot be violated: \
                     {result:?}"
                );
            }
            Err(_) => self.error(codes::UNREPRESENTABLE_NAME),
        }
    }

    /// <https://html.spec.whatwg.org/multipage/parsing.html#the-before-html-insertion-mode>
    pub(crate) fn mode_before_html(&mut self, token: Token) -> Action {
        match token {
            Token::Doctype(_) => {
                self.error(codes::UNEXPECTED_DOCTYPE);
                Action::Done
            }
            Token::Comment(_) | Token::ProcessingInstruction(_) => {
                let point = self.document_end();
                self.insert_markup(&token, Some(point));
                Action::Done
            }
            Token::Characters(text) => {
                let (_, rest) = split_whitespace_prefix(&text);
                if rest.is_empty() {
                    return Action::Done;
                }
                self.anything_else_before_html(Token::Characters(rest.to_owned()))
            }
            Token::StartTag(tag) if tag.name == "html" => {
                let element = self.create_element_for_token(
                    &tag,
                    Namespace::Html,
                    crate::tree::insert::plain_attributes(&tag.attributes),
                );
                let root = self.document.root();
                let result = self.document.append_child(root, element);
                debug_assert!(
                    result.is_ok(),
                    "this is the \"before html\" mode's `html` start-tag branch, which always \
                     leaves for \"before head\" right after and never returns, so it runs at \
                     most once per parse -- `element` was just created detached above, and it \
                     is the only code path that ever appends a document element, so there \
                     cannot be one here already: {result:?}"
                );
                self.open
                    .push(OpenElement::new(element, Namespace::Html, "html"));
                self.mode = Mode::BeforeHead;
                Action::Done
            }
            Token::EndTag(ref tag)
                if !matches!(tag.name.as_str(), "head" | "body" | "html" | "br") =>
            {
                self.error(codes::UNEXPECTED_END_TAG);
                Action::Done
            }
            other => self.anything_else_before_html(other),
        }
    }

    fn anything_else_before_html(&mut self, token: Token) -> Action {
        let tag = Tag {
            name: "html".to_owned(),
            ..Tag::default()
        };
        let element = self.create_element_for_token(&tag, Namespace::Html, Vec::new());
        let root = self.document.root();
        let result = self.document.append_child(root, element);
        debug_assert!(
            result.is_ok(),
            "this is \"before html\"'s implied-`html` fallback, which always leaves for \
             \"before head\" right after and never returns, so it runs at most once per parse \
             -- `element` was just created detached above, and the \"html\" start-tag branch \
             above is the only other code path that ever appends a document element, so \
             exactly one of the two runs, never both: {result:?}"
        );
        self.open
            .push(OpenElement::new(element, Namespace::Html, "html"));
        self.mode = Mode::BeforeHead;
        Action::Reprocess(token)
    }

    /// <https://html.spec.whatwg.org/multipage/parsing.html#the-before-head-insertion-mode>
    pub(crate) fn mode_before_head(&mut self, token: Token) -> Action {
        match token {
            Token::Characters(text) => {
                let (_, rest) = split_whitespace_prefix(&text);
                if rest.is_empty() {
                    return Action::Done;
                }
                self.anything_else_before_head(Token::Characters(rest.to_owned()))
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
            Token::StartTag(tag) if tag.name == "head" => {
                let head = self.insert_html_element(&tag);
                self.head = Some(head);
                self.mode = Mode::InHead;
                Action::Done
            }
            Token::EndTag(ref tag)
                if !matches!(tag.name.as_str(), "head" | "body" | "html" | "br") =>
            {
                self.error(codes::UNEXPECTED_END_TAG);
                Action::Done
            }
            other => self.anything_else_before_head(other),
        }
    }

    fn anything_else_before_head(&mut self, token: Token) -> Action {
        let tag = Tag {
            name: "head".to_owned(),
            ..Tag::default()
        };
        let head = self.insert_html_element(&tag);
        self.head = Some(head);
        self.mode = Mode::InHead;
        Action::Reprocess(token)
    }

    /// The generic RCDATA and raw text element parsing algorithms.
    pub(crate) fn parse_text_element(&mut self, tag: &Tag, model: ContentModel) {
        self.insert_html_element(tag);
        self.tokenizer.set_content_model(model);
        self.original_mode = self.mode;
        self.mode = Mode::Text;
    }

    /// <https://html.spec.whatwg.org/multipage/parsing.html#parsing-main-inhead>
    pub(crate) fn mode_in_head(&mut self, token: Token) -> Action {
        match token {
            Token::Characters(text) => {
                let (whitespace, rest) = split_whitespace_prefix(&text);
                self.insert_characters(whitespace);
                if rest.is_empty() {
                    return Action::Done;
                }
                self.anything_else_in_head(Token::Characters(rest.to_owned()))
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
            Token::StartTag(tag)
                if matches!(
                    tag.name.as_str(),
                    "base" | "basefont" | "bgsound" | "link" | "meta"
                ) =>
            {
                self.insert_html_element(&tag);
                self.open.pop();
                Action::Done
            }
            Token::StartTag(tag) if tag.name == "title" => {
                self.parse_text_element(&tag, ContentModel::Rcdata);
                Action::Done
            }
            Token::StartTag(tag)
                if tag.name == "noframes"
                    || tag.name == "style"
                    || (tag.name == "noscript" && self.scripting) =>
            {
                self.parse_text_element(&tag, ContentModel::RawText);
                Action::Done
            }
            Token::StartTag(tag) if tag.name == "noscript" => {
                self.insert_html_element(&tag);
                self.mode = Mode::InHeadNoscript;
                Action::Done
            }
            Token::StartTag(tag) if tag.name == "script" => {
                self.parse_text_element(&tag, ContentModel::ScriptData);
                Action::Done
            }
            Token::StartTag(tag) if tag.name == "template" => {
                self.insert_html_element(&tag);
                self.formatting.push(crate::tree::FormatEntry::Marker);
                self.frameset_ok = false;
                self.mode = Mode::InTemplate;
                self.template_modes.push(Mode::InTemplate);
                Action::Done
            }
            Token::EndTag(tag) if tag.name == "template" => {
                if !self.has_open_template() {
                    self.error(codes::UNEXPECTED_END_TAG);
                    return Action::Done;
                }
                self.generate_implied_end_tags_thoroughly();
                if !self.current_is_html("template") {
                    self.error(codes::UNEXPECTED_END_TAG);
                }
                self.pop_until_html_named("template");
                self.clear_formatting_to_marker();
                self.template_modes.pop();
                self.reset_insertion_mode();
                Action::Done
            }
            Token::EndTag(tag) if tag.name == "head" => {
                self.open.pop();
                self.mode = Mode::AfterHead;
                Action::Done
            }
            Token::EndTag(ref tag) if matches!(tag.name.as_str(), "body" | "html" | "br") => {
                self.anything_else_in_head(token)
            }
            Token::StartTag(ref tag) if tag.name == "head" => {
                self.error(codes::UNEXPECTED_START_TAG);
                Action::Done
            }
            Token::EndTag(_) => {
                self.error(codes::UNEXPECTED_END_TAG);
                Action::Done
            }
            other => self.anything_else_in_head(other),
        }
    }

    fn anything_else_in_head(&mut self, token: Token) -> Action {
        self.open.pop();
        self.mode = Mode::AfterHead;
        Action::Reprocess(token)
    }

    /// <https://html.spec.whatwg.org/multipage/parsing.html#parsing-main-inheadnoscript>
    pub(crate) fn mode_in_head_noscript(&mut self, token: Token) -> Action {
        match token {
            Token::Doctype(_) => {
                self.error(codes::UNEXPECTED_DOCTYPE);
                Action::Done
            }
            Token::StartTag(ref tag) if tag.name == "html" => self.dispatch(Mode::InBody, token),
            Token::EndTag(tag) if tag.name == "noscript" => {
                self.open.pop();
                self.mode = Mode::InHead;
                Action::Done
            }
            Token::Characters(text) => {
                // "A character token that is one of tab, LF, FF, CR, space"
                // and "anything else" are separate entries, and the standard's
                // tokenizer emits one character token per character, so a run
                // that starts with white space and continues with something
                // else takes both branches in turn.
                let (whitespace, rest) = split_whitespace_prefix(&text);
                if !whitespace.is_empty() {
                    let _ = self.dispatch(Mode::InHead, Token::Characters(whitespace.to_owned()));
                }
                if rest.is_empty() {
                    return Action::Done;
                }
                self.error(codes::UNEXPECTED_CHARACTER);
                self.open.pop();
                self.mode = Mode::InHead;
                Action::Reprocess(Token::Characters(rest.to_owned()))
            }
            Token::Comment(_) | Token::ProcessingInstruction(_) => {
                self.dispatch(Mode::InHead, token)
            }
            Token::StartTag(ref tag)
                if matches!(
                    tag.name.as_str(),
                    "basefont" | "bgsound" | "link" | "meta" | "noframes" | "style"
                ) =>
            {
                self.dispatch(Mode::InHead, token)
            }
            Token::StartTag(ref tag) if matches!(tag.name.as_str(), "head" | "noscript") => {
                self.error(codes::UNEXPECTED_START_TAG);
                Action::Done
            }
            Token::EndTag(ref tag) if tag.name != "br" => {
                self.error(codes::UNEXPECTED_END_TAG);
                Action::Done
            }
            other => {
                self.error(codes::UNEXPECTED_CHARACTER);
                self.open.pop();
                self.mode = Mode::InHead;
                Action::Reprocess(other)
            }
        }
    }

    /// <https://html.spec.whatwg.org/multipage/parsing.html#the-after-head-insertion-mode>
    pub(crate) fn mode_after_head(&mut self, token: Token) -> Action {
        match token {
            Token::Characters(text) => {
                let (whitespace, rest) = split_whitespace_prefix(&text);
                self.insert_characters(whitespace);
                if rest.is_empty() {
                    return Action::Done;
                }
                self.anything_else_after_head(Token::Characters(rest.to_owned()))
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
            Token::StartTag(tag) if tag.name == "body" => {
                self.insert_html_element(&tag);
                self.frameset_ok = false;
                self.mode = Mode::InBody;
                Action::Done
            }
            Token::StartTag(tag) if tag.name == "frameset" => {
                self.insert_html_element(&tag);
                self.mode = Mode::InFrameset;
                Action::Done
            }
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
                self.error(codes::UNEXPECTED_START_TAG);
                let head = self.head;
                if let Some(head) = head {
                    self.open
                        .push(OpenElement::new(head, Namespace::Html, "head"));
                }
                let action = self.dispatch(Mode::InHead, token);
                if let Some(head) = head {
                    self.remove_from_stack(head);
                }
                action
            }
            Token::EndTag(ref tag) if tag.name == "template" => self.dispatch(Mode::InHead, token),
            Token::EndTag(ref tag) if matches!(tag.name.as_str(), "body" | "html" | "br") => {
                self.anything_else_after_head(token)
            }
            Token::StartTag(ref tag) if tag.name == "head" => {
                self.error(codes::UNEXPECTED_START_TAG);
                Action::Done
            }
            Token::EndTag(_) => {
                self.error(codes::UNEXPECTED_END_TAG);
                Action::Done
            }
            other => self.anything_else_after_head(other),
        }
    }

    fn anything_else_after_head(&mut self, token: Token) -> Action {
        let tag = Tag {
            name: "body".to_owned(),
            ..Tag::default()
        };
        self.insert_html_element(&tag);
        self.mode = Mode::InBody;
        Action::Reprocess(token)
    }

    /// <https://html.spec.whatwg.org/multipage/parsing.html#parsing-main-incdata>
    pub(crate) fn mode_text(&mut self, token: Token) -> Action {
        match token {
            Token::Characters(text) => {
                self.insert_characters(&text);
                Action::Done
            }
            Token::Eof => {
                self.error(codes::UNEXPECTED_EOF);
                self.open.pop();
                self.mode = self.original_mode;
                Action::Reprocess(Token::Eof)
            }
            Token::EndTag(_) => {
                self.open.pop();
                self.mode = self.original_mode;
                Action::Done
            }
            _ => Action::Done,
        }
    }
}
