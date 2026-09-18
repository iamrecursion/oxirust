//! The five namespaces an HTML parser deals with, and the element name
//! classifications the tree construction stage keys off.
//!
//! HTML markup is not namespaced in its syntax, but the DOM it builds is: an
//! HTML parser puts every element it creates into one of exactly three element
//! namespaces ([`Namespace`]), and the "adjust foreign attributes" step of
//! [HTML § 13.2.6.5](https://html.spec.whatwg.org/multipage/parsing.html#adjust-foreign-attributes)
//! additionally places a fixed list of attributes into the XLink, XML and XMLNS
//! namespaces.

/// `http://www.w3.org/1999/xhtml`, the namespace of every element an HTML
/// document contains unless it is inside `<svg>` or `<math>`.
pub const HTML_NAMESPACE: &str = "http://www.w3.org/1999/xhtml";

/// `http://www.w3.org/1998/Math/MathML`.
pub const MATHML_NAMESPACE: &str = "http://www.w3.org/1998/Math/MathML";

/// `http://www.w3.org/2000/svg`.
pub const SVG_NAMESPACE: &str = "http://www.w3.org/2000/svg";

/// `http://www.w3.org/1999/xlink`, used by the `xlink:*` attributes of SVG.
pub const XLINK_NAMESPACE: &str = "http://www.w3.org/1999/xlink";

/// `http://www.w3.org/XML/1998/namespace`, the namespace of `xml:lang` and
/// friends.
pub const XML_NAMESPACE: &str = oxixml_qname::XML_NAMESPACE;

/// `http://www.w3.org/2000/xmlns/`, the namespace of the `xmlns` attributes
/// themselves.
pub const XMLNS_NAMESPACE: &str = oxixml_qname::XMLNS_NAMESPACE;

/// One of the three element namespaces an HTML parser can create elements in.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum Namespace {
    /// `http://www.w3.org/1999/xhtml`.
    Html,
    /// `http://www.w3.org/1998/Math/MathML`.
    MathMl,
    /// `http://www.w3.org/2000/svg`.
    Svg,
}

impl Namespace {
    /// The namespace name (URI) this variant stands for.
    ///
    /// # Examples
    ///
    /// ```
    /// use oxixml_html::Namespace;
    ///
    /// assert_eq!(Namespace::Svg.uri(), "http://www.w3.org/2000/svg");
    /// ```
    #[must_use]
    pub const fn uri(self) -> &'static str {
        match self {
            Self::Html => HTML_NAMESPACE,
            Self::MathMl => MATHML_NAMESPACE,
            Self::Svg => SVG_NAMESPACE,
        }
    }

    /// The short prefix the html5lib tree dump format uses for this namespace:
    /// the empty string for HTML, `math` for MathML and `svg` for SVG.
    #[must_use]
    pub const fn dump_prefix(self) -> &'static str {
        match self {
            Self::Html => "",
            Self::MathMl => "math",
            Self::Svg => "svg",
        }
    }

    /// Resolves a namespace name back to a variant, if it names one of the
    /// three element namespaces.
    #[must_use]
    pub fn from_uri(uri: &str) -> Option<Self> {
        match uri {
            HTML_NAMESPACE => Some(Self::Html),
            MATHML_NAMESPACE => Some(Self::MathMl),
            SVG_NAMESPACE => Some(Self::Svg),
            _ => None,
        }
    }
}

/// The HTML elements that never have children and therefore never have an end
/// tag ("void elements",
/// [HTML § 13.1.2](https://html.spec.whatwg.org/multipage/syntax.html#void-elements)).
pub const VOID_ELEMENTS: &[&str] = &[
    "area", "base", "br", "col", "embed", "hr", "img", "input", "link", "meta", "source", "track",
    "wbr",
];

/// Returns `true` when `name` is an HTML void element.
///
/// # Examples
///
/// ```
/// use oxixml_html::is_void_element;
///
/// assert!(is_void_element("br"));
/// assert!(!is_void_element("div"));
/// ```
#[must_use]
pub fn is_void_element(name: &str) -> bool {
    VOID_ELEMENTS.contains(&name)
}

/// The HTML elements whose content is serialised without escaping, because the
/// tokenizer reads them in the RAWTEXT or script-data states.
#[must_use]
pub fn is_raw_text_element(name: &str) -> bool {
    matches!(
        name,
        "style" | "script" | "xmp" | "iframe" | "noembed" | "noframes" | "plaintext"
    )
}

/// The elements the "have an element in scope" algorithm treats as scope
/// boundaries in the HTML namespace
/// ([HTML § 13.2.4.2](https://html.spec.whatwg.org/multipage/parsing.html#the-stack-of-open-elements)).
pub(crate) const SCOPE_HTML: &[&str] = &[
    "applet", "caption", "html", "table", "td", "th", "marquee", "object", "select", "template",
];

/// The MathML text integration points, which are also scope boundaries.
pub(crate) const SCOPE_MATHML: &[&str] = &["mi", "mo", "mn", "ms", "mtext"];

/// The SVG elements that act as scope boundaries and HTML integration points.
pub(crate) const SCOPE_SVG: &[&str] = &["foreignObject", "desc", "title"];

/// Returns `true` when an element in the MathML namespace is a "MathML text
/// integration point".
pub(crate) fn is_mathml_text_integration_point(name: &str) -> bool {
    SCOPE_MATHML.contains(&name)
}

/// Returns `true` when an element in the SVG namespace is an "HTML integration
/// point" (`foreignObject`, `desc` and `title`).
pub(crate) fn is_svg_html_integration_point(name: &str) -> bool {
    SCOPE_SVG.contains(&name)
}

/// The elements that "generate implied end tags".
pub(crate) const IMPLIED_END_TAGS: &[&str] = &[
    "dd", "dt", "li", "optgroup", "option", "p", "rb", "rp", "rt", "rtc",
];

/// The extra elements considered by "generate all implied end tags
/// thoroughly".
pub(crate) const IMPLIED_END_TAGS_THOROUGHLY: &[&str] =
    &["caption", "colgroup", "tbody", "td", "tfoot", "th"];

/// The "special" elements of
/// [HTML § 13.2.4.2](https://html.spec.whatwg.org/multipage/parsing.html#special),
/// in the HTML namespace.
pub(crate) const SPECIAL_HTML: &[&str] = &[
    "address",
    "applet",
    "area",
    "article",
    "aside",
    "base",
    "basefont",
    "bgsound",
    "blockquote",
    "body",
    "br",
    "button",
    "caption",
    "center",
    "col",
    "colgroup",
    "dd",
    "details",
    "dir",
    "div",
    "dl",
    "dt",
    "embed",
    "fieldset",
    "figcaption",
    "figure",
    "footer",
    "form",
    "frame",
    "frameset",
    "h1",
    "h2",
    "h3",
    "h4",
    "h5",
    "h6",
    "head",
    "header",
    "hgroup",
    "hr",
    "html",
    "iframe",
    "img",
    "input",
    "keygen",
    "li",
    "link",
    "listing",
    "main",
    "marquee",
    "menu",
    "meta",
    "nav",
    "noembed",
    "noframes",
    "noscript",
    "object",
    "ol",
    "p",
    "param",
    "plaintext",
    "pre",
    "script",
    "search",
    "section",
    "select",
    "source",
    "style",
    "summary",
    "table",
    "tbody",
    "td",
    "template",
    "textarea",
    "tfoot",
    "th",
    "thead",
    "title",
    "tr",
    "track",
    "ul",
    "wbr",
    "xmp",
];

/// Returns `true` when the element `(namespace, name)` is a "special" element.
pub(crate) fn is_special(namespace: Namespace, name: &str) -> bool {
    match namespace {
        Namespace::Html => SPECIAL_HTML.contains(&name),
        Namespace::MathMl => matches!(name, "mi" | "mo" | "mn" | "ms" | "mtext" | "annotation-xml"),
        Namespace::Svg => matches!(name, "foreignObject" | "desc" | "title"),
    }
}

/// The formatting elements of
/// [HTML § 13.2.4.3](https://html.spec.whatwg.org/multipage/parsing.html#formatting).
pub(crate) const FORMATTING_ELEMENTS: &[&str] = &[
    "a", "b", "big", "code", "em", "font", "i", "nobr", "s", "small", "strike", "strong", "tt", "u",
];

#[cfg(test)]
mod tests {
    use super::{FORMATTING_ELEMENTS, Namespace, SPECIAL_HTML, is_special, is_void_element};

    #[test]
    fn namespaces_round_trip() {
        for namespace in [Namespace::Html, Namespace::MathMl, Namespace::Svg] {
            assert_eq!(Namespace::from_uri(namespace.uri()), Some(namespace));
        }
        assert_eq!(Namespace::from_uri("urn:x"), None);
    }

    #[test]
    fn special_and_void_tables_agree_with_the_spec() {
        assert!(is_void_element("br"));
        assert!(!is_void_element("p"));
        assert!(is_special(Namespace::Html, "table"));
        assert!(!is_special(Namespace::Html, "b"));
        assert!(is_special(Namespace::Svg, "desc"));
        assert!(!is_special(Namespace::Svg, "g"));
        // The two tables are disjoint: no formatting element is special.
        for name in FORMATTING_ELEMENTS {
            assert!(!SPECIAL_HTML.contains(name), "{name} is both");
        }
    }

    #[test]
    fn special_table_is_sorted() {
        for window in SPECIAL_HTML.windows(2) {
            assert!(window[0] < window[1], "unsorted near {}", window[0]);
        }
    }
}
