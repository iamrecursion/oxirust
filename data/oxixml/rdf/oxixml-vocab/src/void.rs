//! [VoID](https://www.w3.org/TR/void/) — the Vocabulary of Interlinked Datasets, for describing RDF datasets and the links between them.
//!
//! Source: <https://www.w3.org/TR/void/>

use oxixml_model::NamedNodeRef;

/// The `void` namespace IRI (`http://rdfs.org/ns/void#`).
pub const NAMESPACE: &str = "http://rdfs.org/ns/void#";

macro_rules! term {
    ($name:ident, $local:literal, $doc:expr) => {
        #[doc = $doc]
        pub const $name: NamedNodeRef<'static> =
            NamedNodeRef::new_unchecked(concat!("http://rdfs.org/ns/void#", $local));
    };
}

term!(
    DATASET,
    "Dataset",
    "A collection of RDF triples that are published, maintained, or aggregated by a single provider."
);
term!(
    LINKSET,
    "Linkset",
    "A collection of RDF links between two `void:Dataset`s."
);
term!(
    TECHNICAL_FEATURE,
    "TechnicalFeature",
    "A technical feature of a `void:Dataset`, such as a supported RDF serialization format."
);
term!(
    DATASET_DESCRIPTION,
    "DatasetDescription",
    "A dataset description document, containing VoID metadata about a dataset."
);
term!(
    CLASS,
    "class",
    "The `rdfs:Class` or `owl:Class` that is the class partition's class of interest: all entities in the partition belong to this class."
);
term!(
    CLASSES,
    "classes",
    "The total number of distinct classes in a `void:Dataset`, i.e. the number of distinct resources occurring as objects of `rdf:type` triples."
);
term!(
    CLASS_PARTITION,
    "classPartition",
    "A subset of a `void:Dataset` that contains only the entities of a certain `rdfs:Class`."
);
term!(
    DATA_DUMP,
    "dataDump",
    "An RDF dump, partial or complete, of a `void:Dataset`."
);
term!(
    DISTINCT_OBJECTS,
    "distinctObjects",
    "The total number of distinct objects in a `void:Dataset`, i.e. the number of distinct resources occurring in the object position of triples."
);
term!(
    DISTINCT_SUBJECTS,
    "distinctSubjects",
    "The total number of distinct subjects in a `void:Dataset`, i.e. the number of distinct resources occurring in the subject position of triples."
);
term!(
    DOCUMENTS,
    "documents",
    "The total number of source documents that constitute a `void:Dataset`, for datasets stored as a set of files."
);
term!(
    ENTITIES,
    "entities",
    "The total number of entities that are described in a `void:Dataset`."
);
term!(
    EXAMPLE_RESOURCE,
    "exampleResource",
    "An example resource that is representative of the resources described in a `void:Dataset`."
);
term!(
    FEATURE,
    "feature",
    "A technical feature of a `void:Dataset`, such as a supported RDF serialization format."
);
term!(
    IN_DATASET,
    "inDataset",
    "Points to the `void:Dataset` that a resource or document is part of."
);
term!(
    LINK_PREDICATE,
    "linkPredicate",
    "The predicate used in the triples that make up a `void:Linkset`."
);
term!(
    OBJECTS_TARGET,
    "objectsTarget",
    "The dataset describing the objects of the triples contained in a `void:Linkset`."
);
term!(
    OPEN_SEARCH_DESCRIPTION,
    "openSearchDescription",
    "An OpenSearch description document for a free-text search service over a `void:Dataset`."
);
term!(
    PROPERTIES,
    "properties",
    "The total number of distinct properties in a `void:Dataset`."
);
term!(
    PROPERTY,
    "property",
    "The `rdf:Property` that is the property partition's property of interest."
);
term!(
    PROPERTY_PARTITION,
    "propertyPartition",
    "A subset of a `void:Dataset` that contains only the triples of a certain `rdf:Property`."
);
term!(
    ROOT_RESOURCE,
    "rootResource",
    "A top-level resource in a `void:Dataset`, part of a small, unrepresentative sample of the dataset's resources."
);
term!(
    SPARQL_ENDPOINT,
    "sparqlEndpoint",
    "A SPARQL protocol endpoint that allows querying a `void:Dataset`."
);
term!(
    SUBJECTS_TARGET,
    "subjectsTarget",
    "The dataset describing the subjects of the triples contained in a `void:Linkset`."
);
term!(
    SUBSET,
    "subset",
    "Points to a `void:Dataset` that is a subset of another `void:Dataset`."
);
term!(
    TARGET,
    "target",
    "One of the two datasets linked by a `void:Linkset`."
);
term!(
    TRIPLES,
    "triples",
    "The total number of triples contained in a `void:Dataset`."
);
term!(
    URI_LOOKUP_ENDPOINT,
    "uriLookupEndpoint",
    "A URI lookup protocol endpoint that allows retrieving a description of a resource in a `void:Dataset` by its URI."
);
term!(
    URI_REGEX_PATTERN,
    "uriRegexPattern",
    "A regular expression that matches the URIs of a `void:Dataset`'s entities."
);
term!(
    URI_SPACE,
    "uriSpace",
    "A URI that is a common string prefix of all the entity URIs in a `void:Dataset`."
);
term!(
    VOCABULARY,
    "vocabulary",
    "An RDFS or OWL vocabulary that is (partially) used in a `void:Dataset`."
);

/// All 31 terms defined in the [`void`](self) module, in declaration order.
///
/// Exposed for introspection and exhaustiveness testing.
pub const ALL: &[NamedNodeRef<'static>] = &[
    DATASET,
    LINKSET,
    TECHNICAL_FEATURE,
    DATASET_DESCRIPTION,
    CLASS,
    CLASSES,
    CLASS_PARTITION,
    DATA_DUMP,
    DISTINCT_OBJECTS,
    DISTINCT_SUBJECTS,
    DOCUMENTS,
    ENTITIES,
    EXAMPLE_RESOURCE,
    FEATURE,
    IN_DATASET,
    LINK_PREDICATE,
    OBJECTS_TARGET,
    OPEN_SEARCH_DESCRIPTION,
    PROPERTIES,
    PROPERTY,
    PROPERTY_PARTITION,
    ROOT_RESOURCE,
    SPARQL_ENDPOINT,
    SUBJECTS_TARGET,
    SUBSET,
    TARGET,
    TRIPLES,
    URI_LOOKUP_ENDPOINT,
    URI_REGEX_PATTERN,
    URI_SPACE,
    VOCABULARY,
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
        assert_eq!(ALL.len(), 31);
    }
}
