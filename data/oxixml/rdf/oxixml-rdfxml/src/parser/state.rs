//! The parser's grammar-production stack and its supporting vocabulary.
//!
//! Each entry of [`RdfXmlState`] corresponds to one open XML element and to the RDF/XML
//! production that element matched. The stack is what makes the parser streaming: no document
//! tree is ever built, only the chain of currently open productions.

use oxixml_iri::Iri;
use oxixml_model::{NamedNode, NamedOrBlankNode, OxString};
use oxixml_xml::Writer;
use oxixml_xml::escape::resolve_xml_entity;
use std::collections::HashMap;

/// The `rdf:about` attribute.
pub(crate) const RDF_ABOUT: &str = "http://www.w3.org/1999/02/22-rdf-syntax-ns#about";
/// The banned `rdf:aboutEach` attribute (removed from RDF in 2002).
pub(crate) const RDF_ABOUT_EACH: &str = "http://www.w3.org/1999/02/22-rdf-syntax-ns#aboutEach";
/// The banned `rdf:aboutEachPrefix` attribute (removed from RDF in 2002).
pub(crate) const RDF_ABOUT_EACH_PREFIX: &str =
    "http://www.w3.org/1999/02/22-rdf-syntax-ns#aboutEachPrefix";
/// The `rdf:bagID` attribute (validated, then ignored: RDF 1.1 dropped bag reification).
pub(crate) const RDF_BAG_ID: &str = "http://www.w3.org/1999/02/22-rdf-syntax-ns#bagID";
/// The `rdf:datatype` attribute.
pub(crate) const RDF_DATATYPE: &str = "http://www.w3.org/1999/02/22-rdf-syntax-ns#datatype";
/// The `rdf:Description` element.
pub(crate) const RDF_DESCRIPTION: &str = "http://www.w3.org/1999/02/22-rdf-syntax-ns#Description";
/// The `rdf:ID` attribute.
pub(crate) const RDF_ID: &str = "http://www.w3.org/1999/02/22-rdf-syntax-ns#ID";
/// The `rdf:li` container-membership element.
pub(crate) const RDF_LI: &str = "http://www.w3.org/1999/02/22-rdf-syntax-ns#li";
/// The `rdf:nodeID` attribute.
pub(crate) const RDF_NODE_ID: &str = "http://www.w3.org/1999/02/22-rdf-syntax-ns#nodeID";
/// The `rdf:parseType` attribute.
pub(crate) const RDF_PARSE_TYPE: &str = "http://www.w3.org/1999/02/22-rdf-syntax-ns#parseType";
/// The `rdf:RDF` document element.
pub(crate) const RDF_RDF: &str = "http://www.w3.org/1999/02/22-rdf-syntax-ns#RDF";
/// The `rdf:resource` attribute.
pub(crate) const RDF_RESOURCE: &str = "http://www.w3.org/1999/02/22-rdf-syntax-ns#resource";

/// The RDF terms that may never be used as an element name.
pub(crate) const RESERVED_RDF_ELEMENTS: [&str; 11] = [
    RDF_ABOUT,
    RDF_ABOUT_EACH,
    RDF_ABOUT_EACH_PREFIX,
    RDF_BAG_ID,
    RDF_DATATYPE,
    RDF_ID,
    RDF_LI,
    RDF_NODE_ID,
    RDF_PARSE_TYPE,
    RDF_RDF,
    RDF_RESOURCE,
];

/// The RDF terms that may never be used as an attribute name.
pub(crate) const RESERVED_RDF_ATTRIBUTES: [&str; 4] =
    [RDF_ABOUT_EACH, RDF_ABOUT_EACH_PREFIX, RDF_LI, RDF_RDF];

/// The object of a property element, which may either be a nested node or character data.
#[derive(Clone, Debug)]
pub(crate) enum NodeOrText {
    /// A node element or an `rdf:resource` / `rdf:nodeID` reference.
    Node(NamedOrBlankNode),
    /// Character data, to be turned into a literal at the closing tag.
    Text(OxString),
}

/// One open RDF/XML grammar production.
pub(crate) enum RdfXmlState {
    /// Outside of any element: only `rdf:RDF` or a bare node element may follow.
    Doc {
        /// The base IRI given to the parser, if any.
        base_iri: Option<Iri<OxString>>,
    },
    /// Inside `rdf:RDF`.
    Rdf {
        /// The `xml:base` declared on this element.
        base_iri: Option<Iri<OxString>>,
        /// The `xml:lang` declared on this element.
        language: Option<OxString>,
    },
    /// Inside a node element, i.e. a subject.
    NodeElt {
        /// The `xml:base` declared on this element.
        base_iri: Option<Iri<OxString>>,
        /// The `xml:lang` declared on this element.
        language: Option<OxString>,
        /// The subject this node element denotes.
        subject: NamedOrBlankNode,
        /// How many `rdf:li` children have been seen so far.
        li_counter: u64,
    },
    /// Inside a resource, literal or empty property element.
    PropertyElt {
        /// The predicate this property element denotes.
        iri: NamedNode,
        /// The `xml:base` declared on this element.
        base_iri: Option<Iri<OxString>>,
        /// The `xml:lang` declared on this element.
        language: Option<OxString>,
        /// The subject this property element attaches to.
        subject: NamedOrBlankNode,
        /// The object seen so far, if any.
        object: Option<NodeOrText>,
        /// The reification statement IRI built from `rdf:ID`.
        id_attr: Option<NamedNode>,
        /// The `rdf:datatype` declared on this element.
        datatype_attr: Option<NamedNode>,
    },
    /// Inside a `rdf:parseType="Collection"` property element.
    ParseTypeCollectionPropertyElt {
        /// The predicate this property element denotes.
        iri: NamedNode,
        /// The `xml:base` declared on this element.
        base_iri: Option<Iri<OxString>>,
        /// The `xml:lang` declared on this element.
        language: Option<OxString>,
        /// The subject this property element attaches to.
        subject: NamedOrBlankNode,
        /// The collection members collected so far.
        objects: Vec<NamedOrBlankNode>,
        /// The reification statement IRI built from `rdf:ID`.
        id_attr: Option<NamedNode>,
    },
    /// Inside a `rdf:parseType="Literal"` property element, or an unknown `rdf:parseType`, whose
    /// content is consumed but discarded — `emit` is then `false`.
    ParseTypeLiteralPropertyElt {
        /// The predicate this property element denotes.
        iri: NamedNode,
        /// The `xml:base` declared on this element.
        base_iri: Option<Iri<OxString>>,
        /// The `xml:lang` declared on this element.
        language: Option<OxString>,
        /// The subject this property element attaches to.
        subject: NamedOrBlankNode,
        /// The verbatim copy of the XML subtree being captured.
        writer: Writer<Vec<u8>>,
        /// The reification statement IRI built from `rdf:ID`.
        id_attr: Option<NamedNode>,
    },
}

/// The `<!ENTITY …>` declarations collected from the document's internal DTD subset, on top of
/// the five predefined XML entities.
#[derive(Default)]
pub(crate) struct EntityRegistry(pub(crate) HashMap<String, String>);

impl EntityRegistry {
    /// Resolves an entity name, preferring the predefined XML entities.
    pub(crate) fn resolve(&self, e: &str) -> Option<&str> {
        resolve_xml_entity(e).or_else(|| self.0.get(e).map(String::as_str))
    }
}

/// Returns `true` if a property element already has a non-whitespace object.
pub(crate) fn is_object_defined(object: &Option<NodeOrText>) -> bool {
    match object {
        Some(NodeOrText::Node(_)) => true,
        Some(NodeOrText::Text(t)) => !t.bytes().all(is_whitespace),
        None => false,
    }
}

/// Returns `true` for the four XML whitespace bytes.
pub(crate) fn is_whitespace(c: u8) -> bool {
    matches!(c, b' ' | b'\t' | b'\n' | b'\r')
}

/// Returns `true` if `encoding` is one of the [WHATWG labels] for UTF-8.
///
/// [WHATWG labels]: https://encoding.spec.whatwg.org/#names-and-labels
pub(crate) fn is_utf8(encoding: &[u8]) -> bool {
    matches!(
        encoding.to_ascii_lowercase().as_slice(),
        b"unicode-1-1-utf-8"
            | b"unicode11utf8"
            | b"unicode20utf8"
            | b"utf-8"
            | b"utf8"
            | b"x-unicode20utf8"
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn utf8_labels() {
        assert!(is_utf8(b"UTF-8"));
        assert!(is_utf8(b"utf8"));
        assert!(!is_utf8(b"ISO-8859-1"));
    }

    #[test]
    fn entity_registry_prefers_predefined() {
        let mut registry = EntityRegistry::default();
        registry.0.insert("amp".into(), "NOPE".into());
        registry.0.insert("custom".into(), "value".into());
        assert_eq!(registry.resolve("amp"), Some("&"));
        assert_eq!(registry.resolve("custom"), Some("value"));
        assert_eq!(registry.resolve("missing"), None);
    }

    #[test]
    fn object_definedness() {
        assert!(!is_object_defined(&None));
        assert!(!is_object_defined(&Some(NodeOrText::Text(OxString::new(
            " \n\t"
        )))));
        assert!(is_object_defined(&Some(NodeOrText::Text(OxString::new(
            "x"
        )))));
    }
}
