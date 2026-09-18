//! [RDF Schema 1.1](https://www.w3.org/TR/rdf-schema/) vocabulary.
//!
//! Source: <https://www.w3.org/2000/01/rdf-schema#>

use oxixml_model::NamedNodeRef;

/// The `rdfs` namespace IRI (`http://www.w3.org/2000/01/rdf-schema#`).
pub const NAMESPACE: &str = "http://www.w3.org/2000/01/rdf-schema#";

macro_rules! term {
    ($name:ident, $local:literal, $doc:expr) => {
        #[doc = $doc]
        pub const $name: NamedNodeRef<'static> =
            NamedNodeRef::new_unchecked(concat!("http://www.w3.org/2000/01/rdf-schema#", $local));
    };
}

term!(RESOURCE, "Resource", "The class resource, everything.");
term!(CLASS, "Class", "The class of classes.");
term!(
    SUB_CLASS_OF,
    "subClassOf",
    "The subject is a subclass of a class."
);
term!(
    SUB_PROPERTY_OF,
    "subPropertyOf",
    "The subject is a subproperty of a property."
);
term!(COMMENT, "comment", "A description of the subject resource.");
term!(LABEL, "label", "A human-readable name for the subject.");
term!(DOMAIN, "domain", "A domain of the subject property.");
term!(RANGE, "range", "A range of the subject property.");
term!(
    SEE_ALSO,
    "seeAlso",
    "Further information about the subject resource."
);
term!(
    IS_DEFINED_BY,
    "isDefinedBy",
    "The definition of the subject resource."
);
term!(
    LITERAL,
    "Literal",
    "The class of literal values, eg. textual strings and integers."
);
term!(CONTAINER, "Container", "The class of RDF containers.");
term!(
    CONTAINER_MEMBERSHIP_PROPERTY,
    "ContainerMembershipProperty",
    "The class of container membership properties, rdf:_1, rdf:_2, ..., all of which are sub-properties of 'member'."
);
term!(MEMBER, "member", "A member of the subject resource.");
term!(DATATYPE, "Datatype", "The class of RDF datatypes.");

/// All 15 terms defined in the [`rdfs`](self) module, in declaration order.
///
/// Exposed for introspection and exhaustiveness testing.
pub const ALL: &[NamedNodeRef<'static>] = &[
    RESOURCE,
    CLASS,
    SUB_CLASS_OF,
    SUB_PROPERTY_OF,
    COMMENT,
    LABEL,
    DOMAIN,
    RANGE,
    SEE_ALSO,
    IS_DEFINED_BY,
    LITERAL,
    CONTAINER,
    CONTAINER_MEMBERSHIP_PROPERTY,
    MEMBER,
    DATATYPE,
];

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashSet;

    #[test]
    fn namespace_ends_with_separator() {
        assert!(NAMESPACE.ends_with('#') || NAMESPACE.ends_with('/'));
    }

    #[test]
    fn terms_are_namespace_prefixed() {
        for term in ALL {
            assert!(
                term.as_str().starts_with(NAMESPACE),
                "{} is not prefixed by {}",
                term.as_str(),
                NAMESPACE
            );
            assert!(term.as_str().len() > NAMESPACE.len());
        }
    }

    #[test]
    fn no_duplicate_terms() {
        let mut seen = HashSet::new();
        for term in ALL {
            assert!(
                seen.insert(term.as_str()),
                "duplicate IRI: {}",
                term.as_str()
            );
        }
    }

    #[test]
    fn expected_term_count() {
        assert_eq!(ALL.len(), 15);
    }
}
