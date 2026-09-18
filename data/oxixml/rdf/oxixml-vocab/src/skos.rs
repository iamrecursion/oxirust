//! [SKOS](https://www.w3.org/TR/skos-reference/) Simple Knowledge Organization System vocabulary.
//!
//! Source: <https://www.w3.org/2004/02/skos/core#>

use oxixml_model::NamedNodeRef;

/// The `skos` namespace IRI (`http://www.w3.org/2004/02/skos/core#`).
pub const NAMESPACE: &str = "http://www.w3.org/2004/02/skos/core#";

macro_rules! term {
    ($name:ident, $local:literal, $doc:expr) => {
        #[doc = $doc]
        pub const $name: NamedNodeRef<'static> =
            NamedNodeRef::new_unchecked(concat!("http://www.w3.org/2004/02/skos/core#", $local));
    };
}

term!(CONCEPT, "Concept", "An idea or notion; a unit of thought.");
term!(
    CONCEPT_SCHEME,
    "ConceptScheme",
    "A set of concepts, optionally including statements about semantic relationships between those concepts."
);
term!(
    COLLECTION,
    "Collection",
    "A meaningful collection of concepts."
);
term!(
    ORDERED_COLLECTION,
    "OrderedCollection",
    "An ordered collection of concepts, where both the grouping and the ordering are meaningful."
);
term!(
    IN_SCHEME,
    "inScheme",
    "Relates a resource (for example a concept) to a concept scheme in which it is included."
);
term!(
    HAS_TOP_CONCEPT,
    "hasTopConcept",
    "Relates, by convention, a concept scheme to a concept which is topmost in the broader/narrower concept hierarchies for that scheme, providing an entry point to these hierarchies."
);
term!(
    TOP_CONCEPT_OF,
    "topConceptOf",
    "Relates a concept to the concept scheme that it is a top level concept of."
);
term!(
    PREF_LABEL,
    "prefLabel",
    "skos:prefLabel, skos:altLabel and skos:hiddenLabel are pairwise disjoint properties."
);
term!(
    ALT_LABEL,
    "altLabel",
    "skos:prefLabel, skos:altLabel and skos:hiddenLabel are pairwise disjoint properties."
);
term!(
    HIDDEN_LABEL,
    "hiddenLabel",
    "skos:prefLabel, skos:altLabel and skos:hiddenLabel are pairwise disjoint properties."
);
term!(
    NOTATION,
    "notation",
    "A notation, also known as classification code, is a string of characters such as \"T58.5\" or \"303.4833\" used to uniquely identify a concept within the scope of a given concept scheme."
);
term!(NOTE, "note", "A general note, for any purpose.");
term!(
    CHANGE_NOTE,
    "changeNote",
    "A note about a modification to a concept."
);
term!(
    DEFINITION,
    "definition",
    "A statement or formal explanation of the meaning of a concept."
);
term!(
    EDITORIAL_NOTE,
    "editorialNote",
    "A note for an editor, translator or maintainer of the vocabulary."
);
term!(EXAMPLE, "example", "An example of the use of a concept.");
term!(
    HISTORY_NOTE,
    "historyNote",
    "A note about the past state/use/meaning of a concept."
);
term!(
    SCOPE_NOTE,
    "scopeNote",
    "A note that helps to clarify the meaning and/or the use of a concept."
);
term!(
    SEMANTIC_RELATION,
    "semanticRelation",
    "Links a concept to a concept related by meaning."
);
term!(
    BROADER,
    "broader",
    "Broader concepts are typically rendered as parents in a concept hierarchy (tree)."
);
term!(
    NARROWER,
    "narrower",
    "Narrower concepts are typically rendered as children in a concept hierarchy (tree)."
);
term!(
    RELATED,
    "related",
    "skos:related is disjoint with skos:broaderTransitive"
);
term!(
    BROADER_TRANSITIVE,
    "broaderTransitive",
    "skos:broaderTransitive is a transitive superproperty of skos:broader."
);
term!(
    NARROWER_TRANSITIVE,
    "narrowerTransitive",
    "skos:narrowerTransitive is a transitive superproperty of skos:narrower."
);
term!(
    MEMBER,
    "member",
    "Relates a collection to one of its members."
);
term!(
    MEMBER_LIST,
    "memberList",
    "For any resource, every item in the list given as the value of the skos:memberList property is also a value of the skos:member property."
);
term!(
    MAPPING_RELATION,
    "mappingRelation",
    "These concept mapping relations mirror semantic relations, and the data model defined below is similar (with the exception of skos:exactMatch) to the data model defined for semantic relations. A distinct vocabulary is provided for concept mapping relations, to provide a convenient way to differentiate links within a concept scheme from links between concept schemes. However, this pattern of usage is not a formal requirement of the SKOS data model, and relies on informal definitions of best practice."
);
term!(
    BROAD_MATCH,
    "broadMatch",
    "skos:broadMatch is used to state a hierarchical mapping link between two conceptual resources in different concept schemes."
);
term!(
    NARROW_MATCH,
    "narrowMatch",
    "skos:narrowMatch is used to state a hierarchical mapping link between two conceptual resources in different concept schemes."
);
term!(
    RELATED_MATCH,
    "relatedMatch",
    "skos:relatedMatch is used to state an associative mapping link between two conceptual resources in different concept schemes."
);
term!(
    EXACT_MATCH,
    "exactMatch",
    "skos:exactMatch is disjoint with each of the properties skos:broadMatch and skos:relatedMatch."
);
term!(
    CLOSE_MATCH,
    "closeMatch",
    "skos:closeMatch is used to link two concepts that are sufficiently similar that they can be used interchangeably in some information retrieval applications. In order to avoid the possibility of \"compound errors\" when combining mappings across more than two concept schemes, skos:closeMatch is not declared to be a transitive property."
);

/// All 32 terms defined in the [`skos`](self) module, in declaration order.
///
/// Exposed for introspection and exhaustiveness testing.
pub const ALL: &[NamedNodeRef<'static>] = &[
    CONCEPT,
    CONCEPT_SCHEME,
    COLLECTION,
    ORDERED_COLLECTION,
    IN_SCHEME,
    HAS_TOP_CONCEPT,
    TOP_CONCEPT_OF,
    PREF_LABEL,
    ALT_LABEL,
    HIDDEN_LABEL,
    NOTATION,
    NOTE,
    CHANGE_NOTE,
    DEFINITION,
    EDITORIAL_NOTE,
    EXAMPLE,
    HISTORY_NOTE,
    SCOPE_NOTE,
    SEMANTIC_RELATION,
    BROADER,
    NARROWER,
    RELATED,
    BROADER_TRANSITIVE,
    NARROWER_TRANSITIVE,
    MEMBER,
    MEMBER_LIST,
    MAPPING_RELATION,
    BROAD_MATCH,
    NARROW_MATCH,
    RELATED_MATCH,
    EXACT_MATCH,
    CLOSE_MATCH,
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
        assert_eq!(ALL.len(), 32);
    }
}
