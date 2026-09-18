//! The vocabulary the `html` output method of [Serialization 3.1
//! §7](https://www.w3.org/TR/xslt-xquery-serialization-31/#html-output) and the `xhtml` method of
//! [§8](https://www.w3.org/TR/xslt-xquery-serialization-31/#xhtml-output) work from: which
//! elements belong to an HTML vocabulary at all, which of them are empty, which hold unescaped
//! script text, and which attributes carry a URI.

use super::super::params::{OutputMethod, SerializationParameters};
use oxixml_infoset::{NodeKind, NodeRef, XdmNode};

/// The namespace of XHTML, `http://www.w3.org/1999/xhtml`.
///
/// [Serialization 3.1 §7](https://www.w3.org/TR/xslt-xquery-serialization-31/#html-output) has
/// the `html` output method treat an element in this namespace as an HTML element when the
/// effective HTML version is 5 or later, and as an ordinary XML element before that (QT3
/// `ser/method-html.xml` `Serialization-html-3` against `-4`). The `xhtml` method of §8 has it
/// the other way round: this is XHTML's own namespace, so an element in it is always an XHTML
/// element, and an element in *no* namespace only becomes one from HTML 5 onwards
/// (`ser/method-xhtml.xml` `Serialization-xhtml-1a` against `-2`).
pub(super) const XHTML_NAMESPACE: &str = "http://www.w3.org/1999/xhtml";

/// The namespace of SVG, `http://www.w3.org/2000/svg`.
const SVG_NAMESPACE: &str = "http://www.w3.org/2000/svg";

/// The namespace of MathML, `http://www.w3.org/1998/Math/MathML`.
const MATHML_NAMESPACE: &str = "http://www.w3.org/1998/Math/MathML";

/// The vocabularies HTML 5 embeds in a document without a prefix, which the `xhtml` output method
/// therefore writes under a *default* namespace declaration whatever prefix the tree holds (QT3
/// `Serialization-xhtml-50`, `-51`, `-52`, whose titles call it prefix normalization).
const UNPREFIXED_NAMESPACES: &[&str] = &[XHTML_NAMESPACE, SVG_NAMESPACE, MATHML_NAMESPACE];

/// The HTML 4.01 elements whose content model is empty
/// ([HTML 4.01 §3.2.1](https://www.w3.org/TR/html401/intro/sgmltut.html#h-3.2.1)), plus `embed`,
/// which was never in the specification but which every HTML 4 reader treats as empty and which
/// QT3 `Serialization-html-1` and `Serialization-xhtml-1` both require.
const HTML4_EMPTY_ELEMENTS: &[&str] = &[
    "area", "base", "basefont", "br", "col", "embed", "frame", "hr", "img", "input", "isindex",
    "link", "meta", "param",
];

/// The HTML 5 *void elements*, whose content model is empty.
///
/// The two vocabularies disagree in both directions: `frame` and `isindex` left HTML 5, and
/// `keygen`, `source`, `track` and `wbr` joined it (QT3 `Serialization-xhtml-2` and `-4`, which
/// require `<frame></frame>` and `<isindex></isindex>` beside `<keygen />` under `html-version`
/// 5.0).
const HTML5_EMPTY_ELEMENTS: &[&str] = &[
    "area", "base", "br", "col", "embed", "hr", "img", "input", "keygen", "link", "meta", "param",
    "source", "track", "wbr",
];

/// The HTML elements whose text content is CDATA rather than parsed character data, so that the
/// `html` output method must not escape it (Serialization 3.1 §7.5, QT3 `Serialization-html-9`,
/// `-10`).
const HTML_RAW_TEXT_ELEMENTS: &[&str] = &["script", "style"];

/// The `(element, attribute)` pairs HTML 4.01 declares with the type `%URI`, whose values the
/// `html` output method percent-encodes when `escape-uri-attributes` is in force (Serialization
/// 3.1 §7.5).
///
/// The pair matters rather than the attribute name alone: `for` is a URI on `script` and an
/// `IDREF` on `label`, and `data` is a URI on `object` and arbitrary text on anything carrying an
/// HTML 5 `data-*`-style attribute. Both parts are compared without regard to case.
const HTML_URI_ATTRIBUTES: &[(&str, &str)] = &[
    ("a", "href"),
    ("area", "href"),
    ("base", "href"),
    ("blockquote", "cite"),
    ("body", "background"),
    ("del", "cite"),
    ("form", "action"),
    ("frame", "longdesc"),
    ("frame", "src"),
    ("head", "profile"),
    ("iframe", "longdesc"),
    ("iframe", "src"),
    ("img", "longdesc"),
    ("img", "src"),
    ("img", "usemap"),
    ("input", "src"),
    ("input", "usemap"),
    ("ins", "cite"),
    ("link", "href"),
    ("object", "classid"),
    ("object", "codebase"),
    ("object", "data"),
    ("object", "usemap"),
    ("q", "cite"),
    ("script", "for"),
    ("script", "src"),
];

/// The media type the `html` output method declares in the `meta` element that
/// `include-content-type` asks for.
///
/// [Serialization 3.1 §3](https://www.w3.org/TR/xslt-xquery-serialization-31/#serparam) gives the
/// `media-type` parameter this default for the `html` method, and the parameter itself is
/// metadata for the caller rather than something that changes the text — so the declaration is
/// the default whatever the caller passes.
pub(super) const HTML_MEDIA_TYPE: &str = "text/html";

/// Whether the `html` output method treats `element` as an HTML element rather than as an
/// element of some other vocabulary that merely happens to be in the tree.
///
/// [Serialization 3.1 §7](https://www.w3.org/TR/xslt-xquery-serialization-31/#html-output): an
/// element in no namespace always is one; an element in the XHTML namespace is one from HTML 5
/// onwards; an element in any other namespace never is, so an XML island keeps XML's rules
/// (QT3 `Serialization-html-5`, `-6`).
pub(super) fn is_html_element(element: &NodeRef, parameters: &SerializationParameters) -> bool {
    if parameters.method() != OutputMethod::Html {
        return false;
    }
    match element.node_name() {
        None => false,
        Some(name) => match name.namespace() {
            None => true,
            Some(namespace) => namespace == XHTML_NAMESPACE && parameters.is_html5(),
        },
    }
}

/// Whether the `xhtml` output method treats `element` as an element of the XHTML vocabulary.
///
/// [Serialization 3.1 §8](https://www.w3.org/TR/xslt-xquery-serialization-31/#xhtml-output): an
/// element in the XHTML namespace always is one, and an element in no namespace becomes one from
/// HTML 5 onwards, where a document may leave the namespace off. Anything in another namespace —
/// an XML island, or the SVG and MathML HTML 5 embeds — never is, and keeps XML's own rules
/// (QT3 `Serialization-xhtml-1` and `-1a`, `-2`, `-51`).
pub(super) fn is_xhtml_element(element: &NodeRef, parameters: &SerializationParameters) -> bool {
    if parameters.method() != OutputMethod::Xhtml {
        return false;
    }
    match element.node_name() {
        None => false,
        Some(name) => match name.namespace() {
            None => parameters.is_html5(),
            Some(namespace) => namespace == XHTML_NAMESPACE,
        },
    }
}

/// Whether the `xhtml` output method writes an element in `namespace` without a prefix — see
/// [`UNPREFIXED_NAMESPACES`].
pub(super) fn is_unprefixed_namespace(namespace: &str) -> bool {
    UNPREFIXED_NAMESPACES.contains(&namespace)
}

/// Whether `local` names an HTML element with an empty content model, under either vocabulary.
///
/// The `html` output method works from the union of the two, because Serialization 3.1 §7.1
/// leaves the exact set implementation-defined and QT3 pins the union: `Serialization-html-1` and
/// `-45` ask for `embed` under `version` 4.0, `-2` for the HTML 5 names. HTML element names are
/// case-insensitive.
pub(super) fn is_html_empty_element(local: &str) -> bool {
    is_named(HTML4_EMPTY_ELEMENTS, local, true) || is_named(HTML5_EMPTY_ELEMENTS, local, true)
}

/// Whether `local` names an element the `xhtml` output method minimizes to `<br />`.
///
/// Unlike the `html` method, the `xhtml` method has to distinguish the two vocabularies: XHTML is
/// XML, so an element written with a start tag alone would swallow everything after it, and QT3
/// `Serialization-xhtml-4` requires `<frame></frame>` and `<isindex></isindex>` at `html-version`
/// 5.0 where `Serialization-xhtml-1` requires `<frame />` at 4.0. XHTML element names are XML
/// names, so they are matched exactly.
pub(super) fn is_xhtml_empty_element(local: &str, parameters: &SerializationParameters) -> bool {
    let vocabulary = if parameters.is_html5() {
        HTML5_EMPTY_ELEMENTS
    } else {
        HTML4_EMPTY_ELEMENTS
    };
    is_named(vocabulary, local, false)
}

/// Whether `local` is one of `names`, compared without regard to case when `case_blind`.
fn is_named(names: &[&str], local: &str, case_blind: bool) -> bool {
    names.iter().any(|known| {
        if case_blind {
            known.eq_ignore_ascii_case(local)
        } else {
            *known == local
        }
    })
}

/// Whether `local` names an HTML element whose content is script or style text, which the `html`
/// output method writes without escaping anything.
pub(super) fn is_html_raw_text_element(local: &str) -> bool {
    HTML_RAW_TEXT_ELEMENTS
        .iter()
        .any(|known| known.eq_ignore_ascii_case(local))
}

/// Whether `attribute` on `element` is one HTML gives the type `%URI` — see
/// [`HTML_URI_ATTRIBUTES`].
pub(super) fn is_uri_attribute(element: &str, attribute: &str) -> bool {
    HTML_URI_ATTRIBUTES.iter().any(|(owner, name)| {
        owner.eq_ignore_ascii_case(element) && name.eq_ignore_ascii_case(attribute)
    })
}

/// Whether `node` is the `<meta http-equiv="Content-Type" …>` element that the `meta` the
/// `include-content-type` parameter asks for replaces (Serialization 3.1 §7.4).
pub(super) fn is_content_type_meta(node: &NodeRef) -> bool {
    if node.node_kind() != NodeKind::Element {
        return false;
    }
    if !node
        .node_name()
        .is_some_and(|name| name.local_name().eq_ignore_ascii_case("meta"))
    {
        return false;
    }
    node.attributes().iter().any(|attribute| {
        attribute.node_name().is_some_and(|name| {
            name.namespace().is_none() && name.local_name().eq_ignore_ascii_case("http-equiv")
        }) && attribute
            .string_value()
            .trim()
            .eq_ignore_ascii_case("content-type")
    })
}
