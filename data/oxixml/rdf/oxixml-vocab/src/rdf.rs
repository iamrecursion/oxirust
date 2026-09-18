//! [RDF 1.1/1.2 Concepts](https://www.w3.org/TR/rdf12-concepts/) core syntax vocabulary.
//!
//! Source: <https://www.w3.org/1999/02/22-rdf-syntax-ns#>

use oxixml_model::NamedNodeRef;

/// The `rdf` namespace IRI (`http://www.w3.org/1999/02/22-rdf-syntax-ns#`).
pub const NAMESPACE: &str = "http://www.w3.org/1999/02/22-rdf-syntax-ns#";

macro_rules! term {
    ($name:ident, $local:literal, $doc:expr) => {
        #[doc = $doc]
        pub const $name: NamedNodeRef<'static> = NamedNodeRef::new_unchecked(concat!(
            "http://www.w3.org/1999/02/22-rdf-syntax-ns#",
            $local
        ));
    };
}

term!(
    HTML,
    "HTML",
    "The datatype of RDF literals storing fragments of HTML content"
);
term!(
    LANG_STRING,
    "langString",
    "The datatype of language-tagged string values"
);
term!(
    PLAIN_LITERAL,
    "PlainLiteral",
    "The class of plain (i.e. untyped) literal values, as used in RIF and OWL 2"
);
term!(TYPE, "type", "The subject is an instance of a class.");
term!(PROPERTY, "Property", "The class of RDF properties.");
term!(STATEMENT, "Statement", "The class of RDF statements.");
term!(
    SUBJECT,
    "subject",
    "The subject of the subject RDF statement."
);
term!(
    PREDICATE,
    "predicate",
    "The predicate of the subject RDF statement."
);
term!(OBJECT, "object", "The object of the subject RDF statement.");
term!(BAG, "Bag", "The class of unordered containers.");
term!(SEQ, "Seq", "The class of ordered containers.");
term!(ALT, "Alt", "The class of containers of alternatives.");
term!(
    VALUE,
    "value",
    "Idiomatic property used for structured values."
);
term!(LIST, "List", "The class of RDF Lists.");
term!(
    NIL,
    "nil",
    "The empty list, with no items in it. If the rest of a list is nil then the list has no more items in it."
);
term!(FIRST, "first", "The first item in the subject RDF list.");
term!(
    REST,
    "rest",
    "The rest of the subject RDF list after the first item."
);
term!(
    XML_LITERAL,
    "XMLLiteral",
    "The datatype of XML literal values."
);
term!(
    JSON,
    "JSON",
    "The datatype of RDF literals storing JSON content."
);
term!(
    COMPOUND_LITERAL,
    "CompoundLiteral",
    "A class representing a compound literal."
);
term!(
    COMPOUND_LITERAL_LANGUAGE,
    "language",
    "The language component of a CompoundLiteral."
);
term!(
    COMPOUND_LITERAL_DIRECTION,
    "direction",
    "The base direction component of a CompoundLiteral."
);
term!(
    REIFIES,
    "reifies",
    "Relates a reifier (a resource standing for a triple/statement) to the triple it reifies. Introduced by RDF 1.2 as the preferred way to express reification alongside triple terms."
);
term!(
    DIR_LANG_STRING,
    "dirLangString",
    "The datatype of language-tagged string values that additionally carry a base direction (`ltr` or `rtl`), introduced by RDF 1.2."
);
term!(
    TRIPLE_TERM,
    "TripleTerm",
    "The class of RDF 1.2 triple terms: an asserted-independent representation of a triple that can itself appear as a subject or object (RDF-star)."
);

/// All 25 terms defined in the [`rdf`](self) module, in declaration order.
///
/// Exposed for introspection and exhaustiveness testing.
pub const ALL: &[NamedNodeRef<'static>] = &[
    HTML,
    LANG_STRING,
    PLAIN_LITERAL,
    TYPE,
    PROPERTY,
    STATEMENT,
    SUBJECT,
    PREDICATE,
    OBJECT,
    BAG,
    SEQ,
    ALT,
    VALUE,
    LIST,
    NIL,
    FIRST,
    REST,
    XML_LITERAL,
    JSON,
    COMPOUND_LITERAL,
    COMPOUND_LITERAL_LANGUAGE,
    COMPOUND_LITERAL_DIRECTION,
    REIFIES,
    DIR_LANG_STRING,
    TRIPLE_TERM,
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
        assert_eq!(ALL.len(), 25);
    }
}
