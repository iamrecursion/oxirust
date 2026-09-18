//! Host languages, and the processing rules each one changes.
//!
//! RDFa Core 1.1 is deliberately host-language neutral: it defines the attributes and the
//! [processing sequence](https://www.w3.org/TR/rdfa-core/#s_sequence), and leaves a handful of
//! decisions — which attributes carry an IRI, how the base is set, how the natural language of an
//! element is spelled, which initial contexts are loaded — to the language that embeds it.
//! [`HostLanguage`] is that choice.

use crate::context;

/// The host language an RDFa document is written in.
///
/// The variants differ in exactly the ways the host-language specifications say they differ:
///
/// | | [`Html5`](HostLanguage::Html5) | [`Xhtml5`](HostLanguage::Xhtml5) | [`Xhtml1`](HostLanguage::Xhtml1) | [`Xml`](HostLanguage::Xml) |
/// |---|---|---|---|---|
/// | initial contexts | core + html | core + html | core + xhtml | core |
/// | base from `<base href>` | yes | yes | yes | no |
/// | base from `@xml:base` | no | yes | yes | yes |
/// | language from `@lang` | yes | yes | yes | no |
/// | language from `@xml:lang` | yes | yes | yes | yes |
/// | `@datetime` and `<time>` | yes | yes | no | no |
/// | `rdf:HTML` literals | yes | yes | no | no |
/// | property copying (`rdfa:copy`) | yes | yes | no | no |
///
/// `@href` and `@src` are honoured in every host language. RDFa Core leaves them to the host, and
/// every language in the W3C RDFa Test Suite — including the bare-XML one, whose documents are
/// generated from the same sources as the HTML ones — expects them to be processed.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum HostLanguage {
    /// HTML5 (and HTML4) served as `text/html`, per
    /// [HTML+RDFa 1.1](https://www.w3.org/TR/html-rdfa/). This is the default.
    #[default]
    Html5,
    /// XHTML5 served as `application/xhtml+xml`, per
    /// [HTML+RDFa 1.1](https://www.w3.org/TR/html-rdfa/).
    Xhtml5,
    /// XHTML 1.x with the `XHTML+RDFa 1.1` DOCTYPE, per
    /// [XHTML+RDFa 1.1](https://www.w3.org/TR/xhtml-rdfa/). This is the only host language that
    /// loads the XHTML term mappings (`next`, `prev`, `stylesheet`, …).
    Xhtml1,
    /// A generic XML vocabulary, with only the RDFa Core initial context.
    Xml,
}

impl HostLanguage {
    /// Returns `true` when the language is one of the two HTML serializations, which share the
    /// extensions of [HTML+RDFa 1.1](https://www.w3.org/TR/html-rdfa/): `@datetime`, the `<time>`
    /// element, `rdf:HTML` literals and property copying.
    #[must_use]
    pub const fn is_html(self) -> bool {
        matches!(self, Self::Html5 | Self::Xhtml5)
    }

    /// Returns `true` when `<base href="…">` sets the base IRI.
    #[must_use]
    pub const fn uses_base_element(self) -> bool {
        matches!(self, Self::Html5 | Self::Xhtml5 | Self::Xhtml1)
    }

    /// Returns `true` when `@xml:base` sets the base IRI.
    ///
    /// It does not in `text/html`: `@xml:base` is not a conforming HTML attribute, and
    /// [test 0109](https://rdfa.info/test-suite/) of the W3C RDFa Test Suite asserts that an HTML
    /// processor ignores it.
    #[must_use]
    pub const fn uses_xml_base(self) -> bool {
        !matches!(self, Self::Html5)
    }

    /// Returns `true` when `@lang` (as opposed to `@xml:lang`) sets the current language.
    ///
    /// When both are present `@xml:lang` wins, as
    /// [HTML+RDFa 1.1 §3.1](https://www.w3.org/TR/html-rdfa/#additional-rdfa-processing-rules)
    /// requires.
    #[must_use]
    pub const fn uses_plain_lang(self) -> bool {
        !matches!(self, Self::Xml)
    }

    /// The prefix mappings of this language's initial contexts, in application order.
    #[must_use]
    pub const fn initial_prefixes(self) -> &'static [&'static [(&'static str, &'static str)]] {
        match self {
            Self::Html5 | Self::Xhtml5 => &HTML_PREFIX_CONTEXTS,
            Self::Xhtml1 => &XHTML_PREFIX_CONTEXTS,
            Self::Xml => &CORE_PREFIX_CONTEXTS,
        }
    }

    /// The term mappings of this language's initial contexts, in application order.
    #[must_use]
    pub const fn initial_terms(self) -> &'static [&'static [(&'static str, &'static str)]] {
        match self {
            Self::Html5 | Self::Xhtml5 => &HTML_TERM_CONTEXTS,
            Self::Xhtml1 => &XHTML_TERM_CONTEXTS,
            Self::Xml => &CORE_TERM_CONTEXTS,
        }
    }

    /// The default vocabulary of this language's initial contexts.
    ///
    /// No host language defined by the W3C sets one, so this is always `None`; it exists because
    /// `@vocab=""` is specified to reset the local default vocabulary to *this* value rather than
    /// unconditionally to null.
    #[must_use]
    pub const fn default_vocabulary(self) -> Option<&'static str> {
        None
    }
}

/// The prefix tables an HTML5 or XHTML5 document loads.
static HTML_PREFIX_CONTEXTS: [&[(&str, &str)]; 2] =
    [&context::RDFA_CORE_PREFIXES, &context::HTML_PREFIXES];
/// The prefix tables an XHTML 1.x document loads.
static XHTML_PREFIX_CONTEXTS: [&[(&str, &str)]; 2] =
    [&context::RDFA_CORE_PREFIXES, &context::XHTML_PREFIXES];
/// The prefix tables a bare-XML document loads.
static CORE_PREFIX_CONTEXTS: [&[(&str, &str)]; 1] = [&context::RDFA_CORE_PREFIXES];
/// The term tables an HTML5 or XHTML5 document loads.
static HTML_TERM_CONTEXTS: [&[(&str, &str)]; 2] = [&context::RDFA_CORE_TERMS, &context::HTML_TERMS];
/// The term tables an XHTML 1.x document loads.
static XHTML_TERM_CONTEXTS: [&[(&str, &str)]; 2] =
    [&context::RDFA_CORE_TERMS, &context::XHTML_TERMS];
/// The term tables a bare-XML document loads.
static CORE_TERM_CONTEXTS: [&[(&str, &str)]; 1] = [&context::RDFA_CORE_TERMS];
