//! The public entry points: parsing a whole document and parsing a fragment.

use oxixml_dom::{Document, NodeId};

use crate::error::ParseError;
use crate::ns::Namespace;
use crate::quirks::QuirksMode;
use crate::token::Tag;
use crate::tokenizer::ContentModel;
use crate::tree::{Mode, OpenElement, TreeBuilder};

/// Knobs on the parser.
///
/// The one that actually changes the tree is `scripting`: with scripting
/// enabled a `<noscript>` element's content is raw text (a browser with
/// JavaScript on never parses it as markup), and with scripting disabled it is
/// ordinary markup. A non-browser consumer — an RDFa extractor, a link
/// checker, a templating engine — wants the default, `false`.
///
/// `processing_instructions` enables this crate's non-standard `<?…?>`
/// extension; see [`Tokenizer::set_processing_instructions`](crate::Tokenizer::set_processing_instructions).
/// It is off by default, so `<?php ?>` is the bogus comment the HTML standard
/// asks for.
///
/// `max_depth` bounds how many elements may be open at once; see
/// [`Self::with_max_depth`].
#[derive(Debug, Clone)]
pub struct ParseOptions {
    scripting: bool,
    processing_instructions: bool,
    max_depth: usize,
}

impl Default for ParseOptions {
    fn default() -> Self {
        Self {
            scripting: false,
            processing_instructions: false,
            max_depth: Self::DEFAULT_MAX_DEPTH,
        }
    }
}

impl ParseOptions {
    /// The default ceiling [`Self::max_depth`] uses when not set explicitly:
    /// the workspace-wide ceiling on structural (element/DOM tree) nesting.
    ///
    /// This is generous enough that no well-formed, humanly-authored document
    /// comes anywhere close to it — it exists to bound the cost of the tree
    /// construction algorithms that walk the stack of open elements (see
    /// [`Self::with_max_depth`]) on a document engineered to be pathologically
    /// deep, not to reject deep-but-ordinary markup.
    pub const DEFAULT_MAX_DEPTH: usize = 256;

    /// The default options: scripting disabled, [`Self::DEFAULT_MAX_DEPTH`].
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Sets the scripting flag.
    #[must_use]
    pub const fn with_scripting(mut self, scripting: bool) -> Self {
        self.scripting = scripting;
        self
    }

    /// Returns the scripting flag.
    #[must_use]
    pub const fn scripting(&self) -> bool {
        self.scripting
    }

    /// Enables this crate's non-standard `<?…?>` processing instruction
    /// extension, which is off by default.
    ///
    /// # Examples
    ///
    /// ```
    /// use oxixml_html::{ParseOptions, dump_tree, parse_document_with};
    ///
    /// // The HTML standard's behaviour: a bogus comment.
    /// let plain = parse_document_with("<?php ?>", &ParseOptions::new());
    /// assert!(dump_tree(&plain).starts_with("| <!-- ?php ? -->"));
    ///
    /// // With the extension, a processing instruction survives.
    /// let options = ParseOptions::new().with_processing_instructions(true);
    /// let extended = parse_document_with("<?php ?>", &options);
    /// assert!(dump_tree(&extended).starts_with("| <?php ?>"));
    /// ```
    #[must_use]
    pub const fn with_processing_instructions(mut self, enabled: bool) -> Self {
        self.processing_instructions = enabled;
        self
    }

    /// Returns whether the `<?…?>` extension is enabled.
    #[must_use]
    pub const fn processing_instructions(&self) -> bool {
        self.processing_instructions
    }

    /// Sets the maximum number of elements that may be simultaneously open
    /// while the tree is built — the length of the stack of open elements —
    /// defaulting to [`Self::DEFAULT_MAX_DEPTH`].
    ///
    /// Several tree construction algorithms (matching an end tag against the
    /// open elements, the six element scopes, the adoption agency algorithm)
    /// walk this stack from the top down, so its length bounds their cost.
    /// Without a ceiling, a document that opens an ordinary (non-"special")
    /// element tens of thousands of levels deep and then closes it with a
    /// handful of end tags that do not match anything currently open makes
    /// every one of those end tags cost time proportional to the nesting
    /// depth rather than to a small constant — a cheap, syntactically
    /// ordinary input that can pin a CPU core for tens of seconds.
    ///
    /// Once the limit is reached, the tree builder keeps parsing rather than
    /// failing the whole document: further elements are still created and
    /// inserted into the tree, they are simply no longer tracked as open, so
    /// they end up as children of whichever element was current when the
    /// limit was first reached rather than of the element that actually
    /// encloses them in the source. [`crate::error::codes::MAX_DEPTH_EXCEEDED`]
    /// is recorded the first time this happens.
    ///
    /// # Examples
    ///
    /// ```
    /// use oxixml_html::{ParseOptions, parse_document_with};
    /// use oxixml_html::error::codes;
    ///
    /// let deep = "<div>".repeat(1_000);
    /// let options = ParseOptions::new().with_max_depth(10);
    /// let parsed = parse_document_with(&deep, &options);
    /// assert!(parsed.errors().iter().any(|e| e.code() == codes::MAX_DEPTH_EXCEEDED));
    /// ```
    #[must_use]
    pub const fn with_max_depth(mut self, max_depth: usize) -> Self {
        self.max_depth = max_depth;
        self
    }

    /// Returns the maximum open-element depth; see [`Self::with_max_depth`].
    #[must_use]
    pub const fn max_depth(&self) -> usize {
        self.max_depth
    }
}

/// The result of a parse: the document, plus everything about it that an XML
/// DOM has nowhere to put.
#[derive(Debug)]
pub struct ParseOutput {
    document: Document,
    root: NodeId,
    fragment: bool,
    quirks_mode: QuirksMode,
    errors: Vec<ParseError>,
    template_contents: Vec<(NodeId, NodeId)>,
    originals: Vec<(NodeId, String)>,
}

impl ParseOutput {
    /// The parsed document.
    #[must_use]
    pub const fn document(&self) -> &Document {
        &self.document
    }

    /// The parsed document, mutably.
    pub const fn document_mut(&mut self) -> &mut Document {
        &mut self.document
    }

    /// Consumes the result and returns just the document.
    #[must_use]
    pub fn into_document(self) -> Document {
        self.document
    }

    /// The node whose children are the parse result: the document node for a
    /// document parse, and a document fragment for a fragment parse.
    #[must_use]
    pub const fn root(&self) -> NodeId {
        self.root
    }

    /// Whether this was a fragment parse.
    #[must_use]
    pub const fn is_fragment(&self) -> bool {
        self.fragment
    }

    /// The quirks mode the DOCTYPE selected.
    #[must_use]
    pub const fn quirks_mode(&self) -> QuirksMode {
        self.quirks_mode
    }

    /// Every parse error, in the order they were noticed.
    #[must_use]
    pub fn errors(&self) -> &[ParseError] {
        &self.errors
    }

    /// The document fragment holding a `<template>` element's contents.
    ///
    /// As [DOM § 4.2.7](https://dom.spec.whatwg.org/#concept-template-contents)
    /// requires, the children of a `<template>` are not children of the element
    /// itself; they live in a separate fragment that this method looks up.
    ///
    /// # Examples
    ///
    /// ```
    /// use oxixml_html::{HTML_NAMESPACE, parse_document};
    ///
    /// let parsed = parse_document("<template><p>hi</p></template>");
    /// let document = parsed.document();
    /// let template = document
    ///     .get_elements_by_tag_name_ns(document.root(), Some(HTML_NAMESPACE), "template")
    ///     .next()
    ///     .expect("a template element");
    /// assert_eq!(document.children(template).count(), 0);
    ///
    /// let content = parsed.template_contents(template).expect("template contents");
    /// assert_eq!(document.text_content(content).as_deref(), Some("hi"));
    /// ```
    #[must_use]
    pub fn template_contents(&self, template: NodeId) -> Option<NodeId> {
        self.template_contents
            .iter()
            .find(|(element, _)| *element == template)
            .map(|(_, content)| *content)
    }

    /// Every `(template element, contents fragment)` pair in the document.
    #[must_use]
    pub fn all_template_contents(&self) -> &[(NodeId, NodeId)] {
        &self.template_contents
    }

    /// The HTML spelling of a name or of character data that had to be
    /// rewritten to fit an XML DOM.
    ///
    /// HTML allows an element named `a<b`, an attribute named `"x`, a comment
    /// containing `--` and a nameless `<!DOCTYPE>`; XML allows none of them, so
    /// the node in the document carries a rewritten value and this method
    /// returns what the source actually said. It returns `None` — the usual
    /// case — when nothing had to be rewritten. [`crate::serialize`] and
    /// [`crate::dump_tree`] consult it, so a round trip is faithful even when
    /// the DOM itself cannot be.
    ///
    /// # Examples
    ///
    /// ```
    /// use oxixml_html::{dump_tree, parse_document};
    ///
    /// let parsed = parse_document("<!--a--b-->");
    /// assert!(dump_tree(&parsed).starts_with("| <!-- a--b -->"));
    /// ```
    #[must_use]
    pub fn original_html(&self, node: NodeId) -> Option<&str> {
        self.originals
            .iter()
            .find(|(candidate, _)| *candidate == node)
            .map(|(_, original)| original.as_str())
    }
}

/// Parses a whole HTML document with the default options.
///
/// # Examples
///
/// ```
/// use oxixml_html::parse_document;
///
/// let parsed = parse_document("<!DOCTYPE html><title>Hi</title><p>Body");
/// assert!(parsed.errors().is_empty());
/// ```
#[must_use]
pub fn parse_document(html: &str) -> ParseOutput {
    parse_document_with(html, &ParseOptions::new())
}

/// Parses a whole HTML document.
#[must_use]
pub fn parse_document_with(html: &str, options: &ParseOptions) -> ParseOutput {
    let mut builder = TreeBuilder::new(html, options.scripting(), options.max_depth());
    configure(&mut builder, options);
    builder.run();
    finish_document(builder)
}

/// Applies the options that are not already constructor arguments. Kept as a
/// function of its own so that the streaming parser cannot drift away from the
/// one-shot entry points.
pub(crate) fn configure(builder: &mut TreeBuilder<'_>, options: &ParseOptions) {
    builder
        .tokenizer
        .set_processing_instructions(options.processing_instructions());
}

/// Assembles the result of a document parse once the tree builder has seen
/// the end-of-file token.
pub(crate) fn finish_document(builder: TreeBuilder<'_>) -> ParseOutput {
    let root = builder.document.root();
    ParseOutput {
        root,
        fragment: false,
        quirks_mode: builder.quirks,
        errors: builder.errors,
        template_contents: builder.template_contents,
        originals: builder.originals,
        document: builder.document,
    }
}

/// Parses an HTML fragment in the context of an HTML element, as `innerHTML`
/// does.
///
/// `context` is the local name of the element the fragment is being parsed
/// into: `"body"` for a page fragment, `"td"` for a table cell, `"select"` for
/// the contents of a dropdown. The context element decides both the tokenizer's
/// starting state and the insertion mode, which is why `<td>x</td>` produces a
/// cell in a `tr` context and only the text `x` in a `div` context.
///
/// # Examples
///
/// ```
/// use oxixml_html::{dump_tree, parse_fragment};
///
/// let parsed = parse_fragment("<td>cell", "tr");
/// assert_eq!(dump_tree(&parsed), "| <td>\n|   \"cell\"\n");
///
/// // The same markup in a div context loses the cell entirely.
/// let parsed = parse_fragment("<td>cell", "div");
/// assert_eq!(dump_tree(&parsed), "| \"cell\"\n");
/// ```
#[must_use]
pub fn parse_fragment(fragment: &str, context: &str) -> ParseOutput {
    parse_fragment_with(fragment, Namespace::Html, context, &ParseOptions::new())
}

/// Parses an HTML fragment in the context of an element in any namespace.
///
/// This is the general form of [`parse_fragment`]: an SVG or MathML context
/// element makes the parser start in foreign content.
#[must_use]
pub fn parse_fragment_with(
    fragment: &str,
    namespace: Namespace,
    context: &str,
    options: &ParseOptions,
) -> ParseOutput {
    let mut builder = TreeBuilder::new(fragment, options.scripting(), options.max_depth());
    configure(&mut builder, options);
    let root = begin_fragment(&mut builder, namespace, context, options);
    builder.run();
    finish_fragment(builder, root)
}

/// The fragment parsing algorithm's steps 3 to 7
/// ([HTML § 13.2.4](https://html.spec.whatwg.org/multipage/parsing.html#parsing-html-fragments)):
/// everything that has to happen before the token loop starts. Returns the
/// `html` element the fragment is built inside.
pub(crate) fn begin_fragment(
    builder: &mut TreeBuilder<'_>,
    namespace: Namespace,
    context: &str,
    options: &ParseOptions,
) -> NodeId {
    builder.fragment = true;
    builder.quirks = QuirksMode::NoQuirks;

    // Step 3: the tokenizer's initial state depends on the context element.
    if namespace == Namespace::Html {
        let model = match context {
            "title" | "textarea" => Some(ContentModel::Rcdata),
            "style" | "xmp" | "iframe" | "noembed" | "noframes" => Some(ContentModel::RawText),
            "script" => Some(ContentModel::ScriptData),
            "noscript" if options.scripting() => Some(ContentModel::RawText),
            "plaintext" => Some(ContentModel::PlainText),
            _ => None,
        };
        if let Some(model) = model {
            builder.tokenizer.set_content_model(model);
        }
    }

    // Step 4: an `html` element is the root the fragment is built inside.
    let root_tag = Tag {
        name: "html".to_owned(),
        ..Tag::default()
    };
    let root = builder.create_element_for_token(&root_tag, Namespace::Html, Vec::new());
    let document_root = builder.document.root();
    let result = builder.document.append_child(document_root, root);
    debug_assert!(
        result.is_ok(),
        "`builder.document` was just created a few lines above by TreeBuilder::new and has no \
         children yet, and `root` was just created detached, so appending it as the very first \
         (and, for a fragment parse, only) document element cannot violate any document-node \
         constraint: {result:?}"
    );
    builder
        .open
        .push(OpenElement::new(root, Namespace::Html, "html"));
    builder.fragment_root = Some(root);

    // The context element is not in the tree, but the algorithms need to be
    // able to look at it, so it gets a detached node of its own.
    let context_tag = Tag {
        name: context.to_owned(),
        ..Tag::default()
    };
    let context_id = builder.create_element_for_token(&context_tag, namespace, Vec::new());
    builder.context = Some(OpenElement::new(context_id, namespace, context));

    // Step 5.
    if namespace == Namespace::Html && context == "template" {
        builder.template_modes.push(Mode::InTemplate);
    }

    // Step 7.
    builder.reset_insertion_mode();

    root
}

/// The fragment parsing algorithm's step 9: the fragment is the children of
/// the `html` element [`begin_fragment`] built into.
pub(crate) fn finish_fragment(mut builder: TreeBuilder<'_>, root: NodeId) -> ParseOutput {
    let document_root = builder.document.root();
    let content = builder.document.create_document_fragment();
    let children: Vec<NodeId> = builder.document.children(root).collect();
    for child in children {
        let result = builder.document.append_child(content, child);
        debug_assert!(
            result.is_ok(),
            "`content` was just created empty above, and `child` is currently a child of \
             `root` (snapshotted right before this loop), so moving it under the \
             still-childless `content` cannot form a cycle; a document fragment accepts \
             exactly the child kinds an element does, so the kind is accepted too: {result:?}"
        );
    }
    let result = builder.document.remove_child(document_root, root);
    debug_assert!(
        result.is_ok(),
        "`root` has been `document_root`'s child since it was appended above, and nothing in \
         between moves or removes the fragment root: {result:?}"
    );

    ParseOutput {
        root: content,
        fragment: true,
        quirks_mode: builder.quirks,
        errors: builder.errors,
        template_contents: builder.template_contents,
        originals: builder.originals,
        document: builder.document,
    }
}
