//! Reading the RDFa attributes off a DOM element.
//!
//! RDFa attributes live in no namespace, so they are matched on their qualified name. The two
//! attributes RDFa borrows from XML — `xml:lang` and `xml:base` — are matched both as real
//! namespaced attributes (what an XML or XHTML parse produces) and as colon-containing names in no
//! namespace (what an HTML parse produces, because `text/html` has no namespaces for attributes).
//! The same holds for the legacy `xmlns:prefix` mappings.

use oxixml_dom::{Document, NodeId, XML_NAMESPACE, XMLNS_NAMESPACE};

/// Every RDFa-relevant attribute of one element.
#[derive(Debug, Default)]
pub(crate) struct RdfaAttributes<'a> {
    /// `@about` — a `SafeCURIEorCURIEorIRI` naming the subject.
    pub about: Option<&'a str>,
    /// `@resource` — a `SafeCURIEorCURIEorIRI` naming the object.
    pub resource: Option<&'a str>,
    /// `@href` — an IRI naming the object.
    pub href: Option<&'a str>,
    /// `@src` — an IRI naming the object.
    pub src: Option<&'a str>,
    /// `@property` — `TERMorCURIEorAbsIRIs`.
    pub property: Option<&'a str>,
    /// `@rel` — `TERMorCURIEorAbsIRIs`.
    pub rel: Option<&'a str>,
    /// `@rev` — `TERMorCURIEorAbsIRIs`.
    pub rev: Option<&'a str>,
    /// `@typeof` — `TERMorCURIEorAbsIRIs`.
    pub type_of: Option<&'a str>,
    /// `@datatype` — a single `TERMorCURIEorAbsIRI`.
    pub datatype: Option<&'a str>,
    /// `@content` — the literal value, overriding the element's text.
    pub content: Option<&'a str>,
    /// `@datetime` — the HTML5 literal value, overriding the element's text.
    pub datetime: Option<&'a str>,
    /// `@vocab` — an IRI setting the default vocabulary.
    pub vocab: Option<&'a str>,
    /// `@prefix` — whitespace-separated `prefix: IRI` pairs.
    pub prefix: Option<&'a str>,
    /// `@inlist` — present, whatever its value.
    pub inlist: bool,
    /// `@role` — `TERMorCURIEorAbsIRIs`, resolved against the XHTML vocabulary.
    pub role: Option<&'a str>,
    /// `@id`, which the `@role` rules use to name the subject.
    pub id: Option<&'a str>,
    /// `@lang`.
    pub lang: Option<&'a str>,
    /// `@xml:lang`.
    pub xml_lang: Option<&'a str>,
    /// `@xml:base`.
    pub xml_base: Option<&'a str>,
    /// The `xmlns:prefix` declarations made on this element, in document order.
    pub xmlns: Vec<(&'a str, &'a str)>,
}

impl<'a> RdfaAttributes<'a> {
    /// Collects the RDFa attributes of `element`.
    pub(crate) fn collect(document: &'a Document, element: NodeId) -> Self {
        let mut attributes = Self::default();
        for id in document.attributes(element) {
            let Some(data) = document.attr(id) else {
                continue;
            };
            let value = data.value();
            let namespace = data.namespace_uri();
            let prefix = data.prefix();
            let name = data.local_name();
            // An HTML parse leaves `xmlns:foo` and `xml:lang` in no namespace, but `oxixml-dom`
            // still splits their qualified name, so the prefix is what distinguishes them.
            if namespace == Some(XMLNS_NAMESPACE) || prefix == Some("xmlns") || name == "xmlns" {
                // Only `xmlns:prefix` declares a CURIE prefix. RDFa Core §6 is explicit that the
                // XML default namespace is *not* the RDFa default prefix, so a bare `xmlns` is
                // ignored here.
                if prefix == Some("xmlns") {
                    attributes.xmlns.push((name, value));
                }
                continue;
            }
            if namespace == Some(XML_NAMESPACE) || (namespace.is_none() && prefix == Some("xml")) {
                match name {
                    "lang" => attributes.xml_lang = Some(value),
                    "base" => attributes.xml_base = Some(value),
                    _ => {}
                }
                continue;
            }
            if namespace.is_some() || prefix.is_some() {
                continue;
            }
            match name {
                "about" => attributes.about = Some(value),
                "resource" => attributes.resource = Some(value),
                "href" => attributes.href = Some(value),
                "src" => attributes.src = Some(value),
                "property" => attributes.property = Some(value),
                "rel" => attributes.rel = Some(value),
                "rev" => attributes.rev = Some(value),
                "typeof" => attributes.type_of = Some(value),
                "datatype" => attributes.datatype = Some(value),
                "content" => attributes.content = Some(value),
                "datetime" => attributes.datetime = Some(value),
                "vocab" => attributes.vocab = Some(value),
                "prefix" => attributes.prefix = Some(value),
                "inlist" => attributes.inlist = true,
                "role" => attributes.role = Some(value),
                "id" => attributes.id = Some(value),
                "lang" => attributes.lang = Some(value),
                _ => {}
            }
        }
        attributes
    }
}

/// Parses the `prefix: IRI` pairs of a `@prefix` attribute, left to right.
///
/// A pair whose prefix does not end in a colon, or whose IRI is missing, is skipped — RDFa Core
/// requires the processor to keep going rather than to reject the attribute.
pub(crate) fn parse_prefix_attribute(value: &str) -> Vec<(String, String)> {
    let mut out = Vec::new();
    let mut tokens = value.split_ascii_whitespace();
    while let Some(token) = tokens.next() {
        let Some(prefix) = token.strip_suffix(':') else {
            continue;
        };
        let Some(iri) = tokens.next() else {
            break;
        };
        // `@prefix` declares a *named* prefix; it cannot redefine the RDFa default prefix, and the
        // W3C RDFa Test Suite's "@prefix with empty mapping" case asserts that it does not.
        if prefix.is_empty() {
            continue;
        }
        out.push((prefix.to_ascii_lowercase(), iri.to_owned()));
    }
    out
}

#[cfg(test)]
mod tests {
    use super::parse_prefix_attribute;

    #[test]
    fn prefix_pairs_are_read_left_to_right_and_lower_cased() {
        assert_eq!(
            parse_prefix_attribute(
                "  FOAF: http://xmlns.com/foaf/0.1/\n dc: http://purl.org/dc/terms/ "
            ),
            vec![
                ("foaf".to_owned(), "http://xmlns.com/foaf/0.1/".to_owned()),
                ("dc".to_owned(), "http://purl.org/dc/terms/".to_owned()),
            ]
        );
    }

    #[test]
    fn a_prefix_without_a_colon_is_skipped() {
        assert_eq!(
            parse_prefix_attribute("bogus http://example.com/ dc: http://purl.org/dc/terms/"),
            vec![("dc".to_owned(), "http://purl.org/dc/terms/".to_owned())]
        );
    }
}
