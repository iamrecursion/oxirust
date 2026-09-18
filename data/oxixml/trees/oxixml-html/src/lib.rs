//! `oxixml-html` — a Pure-Rust HTML5 parser producing an [`oxixml_dom`] tree.
//!
//! Part of the [OxiXML](https://github.com/cool-japan/oxixml) workspace — the
//! COOLJAPAN Pure-Rust foundation for all XML and RDF processing. This crate is
//! what lets the RDF side of the workspace ingest the web: RDFa, microdata and
//! `<link rel="alternate">` discovery all need a parser that reads real-world
//! tag soup exactly the way a browser does, and then hands back the same
//! [`Document`](oxixml_dom::Document) that canonicalization, XPath and XSLT
//! already operate on.
//!
//! # What is here
//!
//! * **[`Tokenizer`]** — all eighty-odd tokenizer states of
//!   [HTML § 13.2.5](https://html.spec.whatwg.org/multipage/parsing.html#tokenization),
//!   including the script-data escaping states, CDATA sections in foreign
//!   content, an opt-in `<?…?>` processing instruction extension, and the
//!   complete named
//!   character reference table (all [`entities::NAMED_REFERENCE_COUNT`] of
//!   them) generated from the standard's own `entities.json`.
//! * **Tree construction** — all twenty-one insertion modes of
//!   [HTML § 13.2.6](https://html.spec.whatwg.org/multipage/parsing.html#tree-construction),
//!   the stack of open elements with its four scopes, the list of active
//!   formatting elements with the Noah's Ark clause, the adoption agency
//!   algorithm, foster parenting, implied end tags, `<template>` contents, and
//!   MathML/SVG foreign content with its name and attribute adjustments.
//! * **[`parse_document`] / [`parse_fragment`]** — the two entry points of the
//!   standard: a whole document, and the `innerHTML` fragment case that needs a
//!   context element.
//! * **[`parse_bytes`]** — the encoding sniffing algorithm of
//!   [HTML § 13.2.3.2](https://html.spec.whatwg.org/multipage/parsing.html#determining-the-character-encoding),
//!   including the `<meta charset>` prescan, on top of [`oxixml_encoding`].
//! * **[`StreamingParser`]** — the same parse, driven a chunk at a time
//!   instead of over one buffer, for a document that arrives from a socket or
//!   a decoder. A chunk boundary anywhere at all — inside a tag, a character
//!   reference, a comment or a multi-byte character — produces the identical
//!   tree; see the [`stream`] module.
//! * **[`serialize`]** — the HTML fragment serialization algorithm of
//!   [HTML § 13.3](https://html.spec.whatwg.org/multipage/syntax.html#serialising-html-fragments),
//!   so a parsed tree can be written back out as HTML rather than as XML.
//! * **[`dump_tree`]** — the html5lib tree dump format, which is what the
//!   standard's own test suite compares against and is by far the most useful
//!   thing to print when a tree is not what you expected.
//!
//! # Example
//!
//! ```
//! use oxixml_html::{QuirksMode, parse_document};
//!
//! let parsed = parse_document("<!DOCTYPE html><p>Hello<p>World");
//! let document = parsed.document();
//!
//! // The parser inserted <html>, <head> and <body>, and closed the first <p>.
//! assert_eq!(parsed.quirks_mode(), QuirksMode::NoQuirks);
//! assert_eq!(
//!     oxixml_html::dump_tree(&parsed),
//!     "\
//! | <!DOCTYPE html>
//! | <html>
//! |   <head>
//! |   <body>
//! |     <p>
//! |       \"Hello\"
//! |     <p>
//! |       \"World\"
//! "
//! );
//!
//! // And it is an ordinary oxixml-dom document, so every DOM operation works.
//! let body = document
//!     .get_elements_by_tag_name_ns(document.root(), Some(oxixml_html::HTML_NAMESPACE), "body")
//!     .next()
//!     .expect("a body element");
//! assert_eq!(document.text_content(body).as_deref(), Some("HelloWorld"));
//! ```
//!
//! # Representing HTML in an XML DOM
//!
//! HTML is more permissive than XML: `<a<b>` is an element named `a<b`,
//! `<!--a--b-->` is a comment containing `--`, and `&#1;` is a text character
//! that no XML document may contain. [`oxixml_dom`] enforces the XML rules, so
//! this crate rewrites the handful of constructs that cannot be represented and
//! records a parse error for each one:
//!
//! * an element name, attribute name or DOCTYPE name that is not an XML `Name`
//!   has every offending character replaced by `_` (and gains a leading `_`
//!   when it would otherwise start with a non-name character), reported as
//!   [`codes::UNREPRESENTABLE_NAME`](error::codes::UNREPRESENTABLE_NAME);
//! * text and attribute values drop characters outside the XML `Char`
//!   production, and comment data has `--` rewritten to `- -` and a trailing
//!   `-` padded with a space, reported as
//!   [`codes::UNREPRESENTABLE_CHARACTER_DATA`](error::codes::UNREPRESENTABLE_CHARACTER_DATA).
//!
//! **The tree shape is never affected**, and the original HTML spelling is kept
//! on the side: [`ParseOutput::original_html`] returns it, and both
//! [`serialize`] and [`dump_tree`] use it, so a document that needed a rewrite
//! still round-trips through this crate unchanged. Well-formed HTML never
//! triggers any of these rewrites at all.
//!
//! # `<template>` contents
//!
//! As the DOM specifies, the children of a `<template>` element do not live in
//! the element itself but in a separate document fragment. Look it up with
//! [`ParseOutput::template_contents`].
//!
//! # Scripting
//!
//! This is a parser, not a browser: it never executes scripts, and the
//! `scripting` flag of [`ParseOptions`] only selects which of the two parses
//! the standard defines for `<noscript>` to perform. It defaults to disabled,
//! which is what a non-browser consumer wants — with it, `<noscript>` content
//! is ordinary markup rather than raw text.
//!
//! # The `<?…?>` extension
//!
//! [HTML § 13.2.5.6](https://html.spec.whatwg.org/multipage/parsing.html#tag-open-state)
//! makes `<?` an `unexpected-question-mark-instead-of-tag-name` parse error
//! reconsumed in the bogus comment state, so `<?php echo 1 ?>` is a comment.
//! That is what this crate does by default. [`ParseOptions::with_processing_instructions`]
//! turns on a non-standard extension that instead tokenizes a restricted
//! XML-style processing instruction, for pipelines that must round-trip such
//! markup; anything the restrictions reject falls back to the standard's bogus
//! comment.
//!
//! # Conformance
//!
//! Measured against the html5lib-tests corpus, discovered through
//! `OXIXML_HTML5LIB_TESTS_DIR` or `OXIXML_XML_SUITES_DIR` (the suites are not
//! vendored, and the harnesses skip silently without them):
//!
//! | suite | result |
//! |---|---|
//! | `tokenizer/` (13 files, all six initial states) | **7032 / 7032** |
//! | `tree-construction/` (57 files) | **1792 / 1792** |
//! | `encoding/` (3 files, runnable blocks) | **82 / 82** |
//!
//! The harnesses assert a threshold at or just under each rate (tree
//! construction's is ratcheted all the way to 100%, since that suite is
//! already a clean pass), so a regression fails loudly. Everything they do
//! not run is an explicit, documented gap, counted as a skip and never as a
//! pass:
//!
//! * **Script execution.** `tree-construction/scripted/` (4 blocks) and
//!   `encoding/scripted/` expect trees that only `document.write` or a
//!   script-driven DOM mutation can produce. This crate has no JavaScript
//!   engine, by design.
//! * **The XML coercion mode.** `tokenizer/xmlViolation.test` covers the
//!   optional coercion of a token stream for an XML serializer, which
//!   HTML § 13.2.5 does not define. This crate coerces during tree
//!   construction instead — see the section on representing HTML in an XML DOM
//!   above — and `tests/api.rs` covers that.
//! * **Encodings `oxixml-encoding` does not implement.** None of the
//!   currently vendored `encoding/` blocks are affected — every `#encoding`
//!   label they use (`euc-jp`, `iso-8859-2`, `utf-8`, `windows-1252`) has a
//!   codec — but `tests/encoding_suite.rs` keeps a named skip for this ready
//!   for whenever upstream adds a block for a label `oxixml-encoding`
//!   deliberately leaves unsupported (`gbk`, `gb18030`, `big5`, `euc-kr`,
//!   `windows-874`, `iso-8859-11`); adding codecs belongs to that crate, not
//!   this one. `encoding/chardet/` additionally asks for the statistical
//!   detection HTML § 13.2.3.2 leaves optional and implementation-defined.
//!
//! Tree construction is a clean 100%, including the `<selectedcontent>` blocks
//! of `webkit02.dat` that used to be a known gap: their expected content is
//! produced by HTML § 4.10.4's "clone option contents" algorithm, which the
//! `select` element runs from its own selectedness and child-mutation steps —
//! DOM behaviour that happens logically *alongside* rather than inside
//! § 13.2.6 tree construction proper, and that this crate now reproduces
//! (`src/tree/select.rs`).

#![deny(missing_docs)]
#![forbid(unsafe_code)]

pub mod encoding;
pub mod entities;
pub mod error;
mod ns;
mod parse;
mod quirks;
mod serialize;
pub mod stream;
mod token;
mod tokenizer;
mod tree;

pub use encoding::{
    EncodingConfidence, SniffedEncoding, parse_bytes, parse_bytes_with, sniff_encoding,
};
pub use error::ParseError;
pub use ns::{
    HTML_NAMESPACE, MATHML_NAMESPACE, Namespace, SVG_NAMESPACE, VOID_ELEMENTS, XLINK_NAMESPACE,
    XML_NAMESPACE, XMLNS_NAMESPACE, is_raw_text_element, is_void_element,
};
pub use parse::{
    ParseOptions, ParseOutput, parse_document, parse_document_with, parse_fragment,
    parse_fragment_with,
};
pub use quirks::QuirksMode;
pub use serialize::{SerializeOptions, dump_tree, serialize, serialize_node};
pub use stream::StreamingParser;
pub use token::{Attribute, Doctype, ProcessingInstruction, Tag, Token};
pub use tokenizer::{ContentModel, Tokenizer};
