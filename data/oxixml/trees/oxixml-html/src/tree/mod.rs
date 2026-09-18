//! Tree construction: the insertion modes of
//! [HTML § 13.2.6](https://html.spec.whatwg.org/multipage/parsing.html#tree-construction).
//!
//! The tree builder drives the [`Tokenizer`](crate::Tokenizer) and turns its
//! tokens into an [`oxixml_dom::Document`]. It keeps the four pieces of state
//! the standard describes — the insertion mode, the stack of open elements, the
//! list of active formatting elements, and the head/form element pointers — and
//! implements the algorithms that operate on them: implied end tags, the six
//! element scopes, foster parenting, the adoption agency algorithm and the
//! foreign-content name adjustments.

mod afe;
mod dom;
mod foreign;
mod insert;
mod modes;
mod select;
mod stack;

use std::collections::HashMap;

use oxixml_dom::{Document, NodeId};

use crate::error::{ParseError, codes};
use crate::ns::Namespace;
use crate::quirks::QuirksMode;
use crate::token::{Tag, Token};
use crate::tokenizer::Tokenizer;

pub(crate) use afe::FormatEntry;
pub(crate) use select::SelectState;
pub(crate) use stack::{OpenElement, OpenElements, Scope};

/// The insertion modes of the tree construction stage.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Mode {
    Initial,
    BeforeHtml,
    BeforeHead,
    InHead,
    InHeadNoscript,
    AfterHead,
    InBody,
    Text,
    InTable,
    InTableText,
    InCaption,
    InColumnGroup,
    InTableBody,
    InRow,
    InCell,
    InTemplate,
    AfterBody,
    InFrameset,
    AfterFrameset,
    AfterAfterBody,
    AfterAfterFrameset,
}

/// What a mode handler wants the dispatcher to do next.
pub(crate) enum Action {
    /// The token has been dealt with.
    Done,
    /// The insertion mode changed; run the token through the new mode.
    Reprocess(Token),
    /// Reprocess the token "according to the rules given in the section
    /// corresponding to the current insertion mode in HTML content" — the
    /// foreign-content breakout, which must bypass the dispatcher's foreign
    /// check even though the adjusted current node is still foreign (which it
    /// always is in the fragment case).
    ReprocessInHtml(Token),
}

/// The tree construction stage.
pub(crate) struct TreeBuilder<'a> {
    pub(crate) tokenizer: Tokenizer<'a>,
    pub(crate) document: Document,
    pub(crate) mode: Mode,
    pub(crate) original_mode: Mode,
    pub(crate) template_modes: Vec<Mode>,
    pub(crate) open: OpenElements,
    pub(crate) formatting: Vec<FormatEntry>,
    pub(crate) head: Option<NodeId>,
    pub(crate) form: Option<NodeId>,
    pub(crate) context: Option<OpenElement>,
    pub(crate) fragment: bool,
    pub(crate) scripting: bool,
    /// The configured ceiling on [`OpenElements`] length; see
    /// [`crate::ParseOptions::with_max_depth`].
    pub(crate) max_depth: usize,
    /// Whether [`codes::MAX_DEPTH_EXCEEDED`] has already been recorded for
    /// the stack of open elements, so a document that lingers at the
    /// ceiling does not get one error per element past it.
    pub(crate) max_depth_exceeded: bool,
    /// The same one-shot latch as [`Self::max_depth_exceeded`], for the list
    /// of active formatting elements; see
    /// [`TreeBuilder::push_formatting_element`].
    pub(crate) formatting_max_depth_exceeded: bool,
    pub(crate) frameset_ok: bool,
    pub(crate) foster_parenting: bool,
    pub(crate) ignore_next_lf: bool,
    pub(crate) pending_table_text: String,
    pub(crate) pending_table_text_is_whitespace: bool,
    pub(crate) quirks: QuirksMode,
    pub(crate) errors: Vec<ParseError>,
    pub(crate) template_contents: Vec<(NodeId, NodeId)>,
    /// The HTML spelling of names and character data that had to be rewritten
    /// to fit an XML DOM, keyed by the node that holds the rewritten value.
    pub(crate) originals: Vec<(NodeId, String)>,
    /// The root the fragment parsing algorithm builds into, if any.
    pub(crate) fragment_root: Option<NodeId>,
    /// Per-`select` bookkeeping the "Customizable Select" `selectedcontent`
    /// reactions need (see the `tree::select` module).
    pub(crate) select_state: HashMap<NodeId, SelectState>,
    /// The number of Noah's-Ark name/attribute comparisons
    /// [`TreeBuilder::push_formatting_element`] has performed, test-only:
    /// this is what a regression test asserts a bound on directly, rather
    /// than on wall-clock time (unreliable on a loaded machine) or on a
    /// document depth chosen to keep wall-clock time reasonable.
    #[cfg(test)]
    pub(crate) noah_ark_comparisons: usize,
}

impl<'a> TreeBuilder<'a> {
    /// Creates a tree builder over an input that is complete already.
    pub(crate) fn new(input: &'a str, scripting: bool, max_depth: usize) -> Self {
        Self::with_tokenizer(Tokenizer::new(input), scripting, max_depth)
    }

    /// Creates a tree builder over an input that arrives in chunks; see
    /// [`Tokenizer::streaming`] and [`Self::pump`].
    pub(crate) fn streaming(scripting: bool, max_depth: usize) -> Self {
        Self::with_tokenizer(Tokenizer::streaming(), scripting, max_depth)
    }

    fn with_tokenizer(tokenizer: Tokenizer<'a>, scripting: bool, max_depth: usize) -> Self {
        Self {
            tokenizer,
            document: Document::new(),
            mode: Mode::Initial,
            original_mode: Mode::Initial,
            template_modes: Vec::new(),
            open: OpenElements::default(),
            formatting: Vec::new(),
            head: None,
            form: None,
            context: None,
            fragment: false,
            scripting,
            max_depth,
            max_depth_exceeded: false,
            formatting_max_depth_exceeded: false,
            frameset_ok: true,
            foster_parenting: false,
            ignore_next_lf: false,
            pending_table_text: String::new(),
            pending_table_text_is_whitespace: true,
            quirks: QuirksMode::NoQuirks,
            errors: Vec::new(),
            template_contents: Vec::new(),
            originals: Vec::new(),
            fragment_root: None,
            select_state: HashMap::new(),
            #[cfg(test)]
            noah_ark_comparisons: 0,
        }
    }

    /// Drives tokenization and tree construction as far as the input
    /// currently allows.
    ///
    /// Returns `true` once the end-of-file token has been processed and the
    /// tree is final, and `false` when the tokenizer stopped because it wants
    /// more input — which only a tokenizer whose input is still open can do,
    /// so on the one-shot path this always runs the parse to completion.
    pub(crate) fn pump(&mut self) -> bool {
        loop {
            let foreign = self.adjusted_current_is_foreign();
            self.tokenizer.set_in_foreign_content(foreign);
            let token = self.tokenizer.try_next_token();
            self.errors.append(&mut self.tokenizer.take_errors());
            let Some(token) = token else {
                return false;
            };
            let is_eof = token == Token::Eof;
            self.process(token);
            if is_eof {
                return true;
            }
        }
    }

    /// Runs the parser to completion over an input that is complete already.
    pub(crate) fn run(&mut self) {
        let finished = self.pump();
        debug_assert!(
            finished,
            "TreeBuilder::run drives a tokenizer built from a complete input, which never asks \
             for more, so the pump can only stop at the end-of-file token"
        );
    }

    /// Records a tree construction parse error at the tokenizer's position.
    pub(crate) fn error(&mut self, code: &'static str) {
        let position = self.tokenizer.position();
        self.errors.push(ParseError::new(position, code));
    }

    // ------------------------------------------------------------ dispatcher

    /// The tree construction dispatcher of
    /// [HTML § 13.2.6](https://html.spec.whatwg.org/multipage/parsing.html#tree-construction).
    fn process(&mut self, token: Token) {
        let mut token = token;
        if std::mem::take(&mut self.ignore_next_lf) {
            if let Token::Characters(text) = &token {
                if let Some(rest) = text.strip_prefix('\n') {
                    if rest.is_empty() {
                        return;
                    }
                    token = Token::Characters(rest.to_owned());
                }
            }
        }
        let mut force_html = false;
        loop {
            let action = if !force_html && self.use_foreign_rules(&token) {
                self.foreign_content(token)
            } else {
                self.dispatch(self.mode, token)
            };
            force_html = false;
            match action {
                Action::Done => return,
                Action::Reprocess(next) => token = next,
                Action::ReprocessInHtml(next) => {
                    token = next;
                    force_html = true;
                }
            }
        }
    }

    /// Decides between the "in foreign content" rules and the current
    /// insertion mode.
    fn use_foreign_rules(&self, token: &Token) -> bool {
        let Some(node) = self.adjusted_current_node() else {
            return false;
        };
        if node.namespace == Namespace::Html || matches!(token, Token::Eof) {
            return false;
        }
        match token {
            Token::StartTag(tag) => {
                if node.namespace == Namespace::MathMl
                    && crate::ns::is_mathml_text_integration_point(&node.name)
                    && tag.name != "mglyph"
                    && tag.name != "malignmark"
                {
                    return false;
                }
                if node.namespace == Namespace::MathMl
                    && node.name == "annotation-xml"
                    && tag.name == "svg"
                {
                    return false;
                }
                !self.is_html_integration_point(node)
            }
            Token::Characters(_) => {
                if node.namespace == Namespace::MathMl
                    && crate::ns::is_mathml_text_integration_point(&node.name)
                {
                    return false;
                }
                !self.is_html_integration_point(node)
            }
            _ => true,
        }
    }

    /// The "HTML integration point" test of
    /// [HTML § 13.2.6](https://html.spec.whatwg.org/multipage/parsing.html#html-integration-point).
    fn is_html_integration_point(&self, node: &OpenElement) -> bool {
        match node.namespace {
            Namespace::MathMl => {
                node.name == "annotation-xml"
                    && self
                        .document
                        .get_attribute(node.id, "encoding")
                        .is_some_and(|value| {
                            value.eq_ignore_ascii_case("text/html")
                                || value.eq_ignore_ascii_case("application/xhtml+xml")
                        })
            }
            Namespace::Svg => crate::ns::is_svg_html_integration_point(&node.name),
            Namespace::Html => false,
        }
    }

    /// Runs one token through one insertion mode.
    pub(crate) fn dispatch(&mut self, mode: Mode, token: Token) -> Action {
        match mode {
            Mode::Initial => self.mode_initial(token),
            Mode::BeforeHtml => self.mode_before_html(token),
            Mode::BeforeHead => self.mode_before_head(token),
            Mode::InHead => self.mode_in_head(token),
            Mode::InHeadNoscript => self.mode_in_head_noscript(token),
            Mode::AfterHead => self.mode_after_head(token),
            Mode::InBody => self.mode_in_body(token),
            Mode::Text => self.mode_text(token),
            Mode::InTable => self.mode_in_table(token),
            Mode::InTableText => self.mode_in_table_text(token),
            Mode::InCaption => self.mode_in_caption(token),
            Mode::InColumnGroup => self.mode_in_column_group(token),
            Mode::InTableBody => self.mode_in_table_body(token),
            Mode::InRow => self.mode_in_row(token),
            Mode::InCell => self.mode_in_cell(token),
            Mode::InTemplate => self.mode_in_template(token),
            Mode::AfterBody => self.mode_after_body(token),
            Mode::InFrameset => self.mode_in_frameset(token),
            Mode::AfterFrameset => self.mode_after_frameset(token),
            Mode::AfterAfterBody => self.mode_after_after_body(token),
            Mode::AfterAfterFrameset => self.mode_after_after_frameset(token),
        }
    }

    // -------------------------------------------------------------- helpers

    /// "Reset the insertion mode appropriately"
    /// ([HTML § 13.2.4.1](https://html.spec.whatwg.org/multipage/parsing.html#reset-the-insertion-mode-appropriately)).
    pub(crate) fn reset_insertion_mode(&mut self) {
        let mut index = self.open.len();
        while index > 0 {
            index -= 1;
            let last = index == 0;
            let node = if last {
                match &self.context {
                    Some(context) => context.clone(),
                    None => self.open[index].clone(),
                }
            } else {
                self.open[index].clone()
            };
            if node.namespace != Namespace::Html {
                if last {
                    self.mode = Mode::InBody;
                    return;
                }
                continue;
            }
            match node.name.as_str() {
                "td" | "th" if !last => {
                    self.mode = Mode::InCell;
                    return;
                }
                "tr" => {
                    self.mode = Mode::InRow;
                    return;
                }
                "tbody" | "thead" | "tfoot" => {
                    self.mode = Mode::InTableBody;
                    return;
                }
                "caption" => {
                    self.mode = Mode::InCaption;
                    return;
                }
                "colgroup" => {
                    self.mode = Mode::InColumnGroup;
                    return;
                }
                "table" => {
                    self.mode = Mode::InTable;
                    return;
                }
                "template" => {
                    if let Some(mode) = self.template_modes.last() {
                        self.mode = *mode;
                        return;
                    }
                }
                "head" if !last => {
                    self.mode = Mode::InHead;
                    return;
                }
                "body" => {
                    self.mode = Mode::InBody;
                    return;
                }
                "frameset" => {
                    self.mode = Mode::InFrameset;
                    return;
                }
                "html" => {
                    self.mode = if self.head.is_none() {
                        Mode::BeforeHead
                    } else {
                        Mode::AfterHead
                    };
                    return;
                }
                _ => {}
            }
            if last {
                self.mode = Mode::InBody;
                return;
            }
        }
        self.mode = Mode::InBody;
    }

    /// "Generate implied end tags", optionally excluding one element name.
    pub(crate) fn generate_implied_end_tags(&mut self, except: Option<&str>) {
        while let Some(node) = self.open.last() {
            if node.namespace != Namespace::Html
                || !crate::ns::IMPLIED_END_TAGS.contains(&node.name.as_str())
                || except == Some(node.name.as_str())
            {
                return;
            }
            self.pop_open_element();
        }
    }

    /// "Generate all implied end tags thoroughly".
    pub(crate) fn generate_implied_end_tags_thoroughly(&mut self) {
        while let Some(node) = self.open.last() {
            if node.namespace != Namespace::Html
                || !(crate::ns::IMPLIED_END_TAGS.contains(&node.name.as_str())
                    || crate::ns::IMPLIED_END_TAGS_THOROUGHLY.contains(&node.name.as_str()))
            {
                return;
            }
            self.pop_open_element();
        }
    }

    /// "Close a p element".
    pub(crate) fn close_p_element(&mut self) {
        self.generate_implied_end_tags(Some("p"));
        if !self.current_is_html("p") {
            self.error(codes::UNEXPECTED_END_TAG);
        }
        self.pop_until_html_named("p");
    }

    /// Returns `true` when this is the fragment case and the context element is
    /// an HTML element with the given name.
    pub(crate) fn context_is_html(&self, name: &str) -> bool {
        self.fragment
            && self
                .context
                .as_ref()
                .is_some_and(|context| context.is_html(name))
    }

    /// Returns `true` when the current node is an HTML element with this name.
    pub(crate) fn current_is_html(&self, name: &str) -> bool {
        self.open
            .last()
            .is_some_and(|node| node.namespace == Namespace::Html && node.name == name)
    }

    /// The "stop parsing" step
    /// ([HTML § the end](https://html.spec.whatwg.org/multipage/parsing.html#the-end)):
    /// this parser has no scripts to run and no load events to fire, but one
    /// part of it is still observable — "pop all the nodes off the stack of
    /// open elements" is what makes an `option` element still open when the
    /// document ends run "maybe clone an option into selectedcontent"
    /// ([HTML § the `option` element](https://html.spec.whatwg.org/multipage/form-elements.html#the-option-element))
    /// the same as one closed by an explicit or implied end tag earlier.
    ///
    /// The stack itself is left alone (nothing reads it again once this
    /// returns, and several callers just finished inspecting it for an
    /// end-of-file parse error): only the reaction that popping an `option`
    /// triggers is reproduced, for every `option` still open at this point.
    pub(crate) fn stop_parsing(&mut self) {
        let open_options: Vec<NodeId> = self
            .open
            .iter()
            .filter(|node| node.is_html("option"))
            .map(|node| node.id)
            .collect();
        for option in open_options {
            self.maybe_clone_option_into_selectedcontent(option);
        }
    }

    /// The tag of a token, cloned, for handlers that need to keep it.
    pub(crate) fn take_tag(token: Token) -> Tag {
        match token {
            Token::StartTag(tag) | Token::EndTag(tag) => tag,
            _ => Tag::default(),
        }
    }
}
