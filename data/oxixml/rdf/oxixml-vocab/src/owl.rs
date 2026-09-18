//! [OWL 2](https://www.w3.org/TR/owl2-overview/) Web Ontology Language vocabulary.
//!
//! Source: <https://www.w3.org/2002/07/owl#>

use oxixml_model::NamedNodeRef;

/// The `owl` namespace IRI (`http://www.w3.org/2002/07/owl#`).
pub const NAMESPACE: &str = "http://www.w3.org/2002/07/owl#";

macro_rules! term {
    ($name:ident, $local:literal, $doc:expr) => {
        #[doc = $doc]
        pub const $name: NamedNodeRef<'static> =
            NamedNodeRef::new_unchecked(concat!("http://www.w3.org/2002/07/owl#", $local));
    };
}

term!(
    ALL_DIFFERENT,
    "AllDifferent",
    "The class of collections of pairwise different individuals."
);
term!(
    ALL_DISJOINT_CLASSES,
    "AllDisjointClasses",
    "The class of collections of pairwise disjoint classes."
);
term!(
    ALL_DISJOINT_PROPERTIES,
    "AllDisjointProperties",
    "The class of collections of pairwise disjoint properties."
);
term!(
    ANNOTATION,
    "Annotation",
    "The class of annotated annotations for which the RDF serialization consists of an annotated subject, predicate and object."
);
term!(
    ANNOTATION_PROPERTY,
    "AnnotationProperty",
    "The class of annotation properties."
);
term!(
    ASYMMETRIC_PROPERTY,
    "AsymmetricProperty",
    "The class of asymmetric properties."
);
term!(
    AXIOM,
    "Axiom",
    "The class of annotated axioms for which the RDF serialization consists of an annotated subject, predicate and object."
);
term!(CLASS, "Class", "The class of OWL classes.");
term!(
    DATA_RANGE,
    "DataRange",
    "The class of OWL data ranges, which are special kinds of datatypes. Note: The use of the IRI owl:DataRange has been deprecated as of OWL 2. The IRI rdfs:Datatype SHOULD be used instead."
);
term!(
    DATATYPE_PROPERTY,
    "DatatypeProperty",
    "The class of data properties."
);
term!(
    DEPRECATED_CLASS,
    "DeprecatedClass",
    "The class of deprecated classes."
);
term!(
    DEPRECATED_PROPERTY,
    "DeprecatedProperty",
    "The class of deprecated properties."
);
term!(
    FUNCTIONAL_PROPERTY,
    "FunctionalProperty",
    "The class of functional properties."
);
term!(
    INVERSE_FUNCTIONAL_PROPERTY,
    "InverseFunctionalProperty",
    "The class of inverse-functional properties."
);
term!(
    IRREFLEXIVE_PROPERTY,
    "IrreflexiveProperty",
    "The class of irreflexive properties."
);
term!(
    NAMED_INDIVIDUAL,
    "NamedIndividual",
    "The class of named individuals."
);
term!(
    NEGATIVE_PROPERTY_ASSERTION,
    "NegativePropertyAssertion",
    "The class of negative property assertions."
);
term!(NOTHING, "Nothing", "This is the empty class.");
term!(
    OBJECT_PROPERTY,
    "ObjectProperty",
    "The class of object properties."
);
term!(ONTOLOGY, "Ontology", "The class of ontologies.");
term!(
    ONTOLOGY_PROPERTY,
    "OntologyProperty",
    "The class of ontology properties."
);
term!(
    REFLEXIVE_PROPERTY,
    "ReflexiveProperty",
    "The class of reflexive properties."
);
term!(
    RESTRICTION,
    "Restriction",
    "The class of property restrictions."
);
term!(
    SYMMETRIC_PROPERTY,
    "SymmetricProperty",
    "The class of symmetric properties."
);
term!(
    TRANSITIVE_PROPERTY,
    "TransitiveProperty",
    "The class of transitive properties."
);
term!(THING, "Thing", "The class of OWL individuals.");
term!(
    ALL_VALUES_FROM,
    "allValuesFrom",
    "The property that determines the class that a universal property restriction refers to."
);
term!(
    ANNOTATED_PROPERTY,
    "annotatedProperty",
    "The property that determines the predicate of an annotated axiom or annotated annotation."
);
term!(
    ANNOTATED_SOURCE,
    "annotatedSource",
    "The property that determines the subject of an annotated axiom or annotated annotation."
);
term!(
    ANNOTATED_TARGET,
    "annotatedTarget",
    "The property that determines the object of an annotated axiom or annotated annotation."
);
term!(
    ASSERTION_PROPERTY,
    "assertionProperty",
    "The property that determines the predicate of a negative property assertion."
);
term!(
    BACKWARD_COMPATIBLE_WITH,
    "backwardCompatibleWith",
    "The annotation property that indicates that a given ontology is backward compatible with another ontology."
);
term!(
    BOTTOM_DATA_PROPERTY,
    "bottomDataProperty",
    "The data property that does not relate any individual to any data value."
);
term!(
    BOTTOM_OBJECT_PROPERTY,
    "bottomObjectProperty",
    "The object property that does not relate any two individuals."
);
term!(
    CARDINALITY,
    "cardinality",
    "The property that determines the cardinality of an exact cardinality restriction."
);
term!(
    COMPLEMENT_OF,
    "complementOf",
    "The property that determines that a given class is the complement of another class."
);
term!(
    DATATYPE_COMPLEMENT_OF,
    "datatypeComplementOf",
    "The property that determines that a given data range is the complement of another data range with respect to the data domain."
);
term!(
    DEPRECATED,
    "deprecated",
    "The annotation property that indicates that a given entity has been deprecated."
);
term!(
    DIFFERENT_FROM,
    "differentFrom",
    "The property that determines that two given individuals are different."
);
term!(
    DISJOINT_UNION_OF,
    "disjointUnionOf",
    "The property that determines that a given class is equivalent to the disjoint union of a collection of other classes."
);
term!(
    DISJOINT_WITH,
    "disjointWith",
    "The property that determines that two given classes are disjoint."
);
term!(
    DISTINCT_MEMBERS,
    "distinctMembers",
    "The property that determines the collection of pairwise different individuals in a owl:AllDifferent axiom."
);
term!(
    EQUIVALENT_CLASS,
    "equivalentClass",
    "The property that determines that two given classes are equivalent, and that is used to specify datatype definitions."
);
term!(
    EQUIVALENT_PROPERTY,
    "equivalentProperty",
    "The property that determines that two given properties are equivalent."
);
term!(
    HAS_KEY,
    "hasKey",
    "The property that determines the collection of properties that jointly build a key."
);
term!(
    HAS_SELF,
    "hasSelf",
    "The property that determines the property that a self restriction refers to."
);
term!(
    HAS_VALUE,
    "hasValue",
    "The property that determines the individual that a has-value restriction refers to."
);
term!(
    IMPORTS,
    "imports",
    "The property that is used for importing other ontologies into a given ontology."
);
term!(
    INCOMPATIBLE_WITH,
    "incompatibleWith",
    "The annotation property that indicates that a given ontology is incompatible with another ontology."
);
term!(
    INTERSECTION_OF,
    "intersectionOf",
    "The property that determines the collection of classes or data ranges that build an intersection."
);
term!(
    INVERSE_OF,
    "inverseOf",
    "The property that determines that two given properties are inverse."
);
term!(
    MAX_CARDINALITY,
    "maxCardinality",
    "The property that determines the cardinality of a maximum cardinality restriction."
);
term!(
    MAX_QUALIFIED_CARDINALITY,
    "maxQualifiedCardinality",
    "The property that determines the cardinality of a maximum qualified cardinality restriction."
);
term!(
    MEMBERS,
    "members",
    "The property that determines the collection of members in either a owl:AllDifferent, owl:AllDisjointClasses or owl:AllDisjointProperties axiom."
);
term!(
    MIN_CARDINALITY,
    "minCardinality",
    "The property that determines the cardinality of a minimum cardinality restriction."
);
term!(
    MIN_QUALIFIED_CARDINALITY,
    "minQualifiedCardinality",
    "The property that determines the cardinality of a minimum qualified cardinality restriction."
);
term!(
    ON_CLASS,
    "onClass",
    "The property that determines the class that a qualified object cardinality restriction refers to."
);
term!(
    ON_DATA_RANGE,
    "onDataRange",
    "The property that determines the data range that a qualified data cardinality restriction refers to."
);
term!(
    ON_DATATYPE,
    "onDatatype",
    "The property that determines the datatype that a datatype restriction refers to."
);
term!(
    ONE_OF,
    "oneOf",
    "The property that determines the collection of individuals or data values that build an enumeration."
);
term!(
    ON_PROPERTIES,
    "onProperties",
    "The property that determines the n-tuple of properties that a property restriction on an n-ary data range refers to."
);
term!(
    ON_PROPERTY,
    "onProperty",
    "The property that determines the property that a property restriction refers to."
);
term!(
    PRIOR_VERSION,
    "priorVersion",
    "The annotation property that indicates the predecessor ontology of a given ontology."
);
term!(
    PROPERTY_CHAIN_AXIOM,
    "propertyChainAxiom",
    "The property that determines the n-tuple of properties that build a sub property chain of a given property."
);
term!(
    PROPERTY_DISJOINT_WITH,
    "propertyDisjointWith",
    "The property that determines that two given properties are disjoint."
);
term!(
    QUALIFIED_CARDINALITY,
    "qualifiedCardinality",
    "The property that determines the cardinality of an exact qualified cardinality restriction."
);
term!(
    SAME_AS,
    "sameAs",
    "The property that determines that two given individuals are equal."
);
term!(
    SOME_VALUES_FROM,
    "someValuesFrom",
    "The property that determines the class that an existential property restriction refers to."
);
term!(
    SOURCE_INDIVIDUAL,
    "sourceIndividual",
    "The property that determines the subject of a negative property assertion."
);
term!(
    TARGET_INDIVIDUAL,
    "targetIndividual",
    "The property that determines the object of a negative object property assertion."
);
term!(
    TARGET_VALUE,
    "targetValue",
    "The property that determines the value of a negative data property assertion."
);
term!(
    TOP_DATA_PROPERTY,
    "topDataProperty",
    "The data property that relates every individual to every data value."
);
term!(
    TOP_OBJECT_PROPERTY,
    "topObjectProperty",
    "The object property that relates every two individuals."
);
term!(
    UNION_OF,
    "unionOf",
    "The property that determines the collection of classes or data ranges that build a union."
);
term!(
    VERSION_INFO,
    "versionInfo",
    "The annotation property that provides version information for an ontology or another OWL construct."
);
term!(
    VERSION_IRI,
    "versionIRI",
    "The property that identifies the version IRI of an ontology."
);
term!(
    WITH_RESTRICTIONS,
    "withRestrictions",
    "The property that determines the collection of facet-value pairs that define a datatype restriction."
);
term!(
    REAL,
    "real",
    "The OWL 2 datatype of the set of real numbers (unbounded precision)."
);
term!(
    RATIONAL,
    "rational",
    "The OWL 2 datatype of the set of rational numbers."
);

/// All 79 terms defined in the [`owl`](self) module, in declaration order.
///
/// Exposed for introspection and exhaustiveness testing.
pub const ALL: &[NamedNodeRef<'static>] = &[
    ALL_DIFFERENT,
    ALL_DISJOINT_CLASSES,
    ALL_DISJOINT_PROPERTIES,
    ANNOTATION,
    ANNOTATION_PROPERTY,
    ASYMMETRIC_PROPERTY,
    AXIOM,
    CLASS,
    DATA_RANGE,
    DATATYPE_PROPERTY,
    DEPRECATED_CLASS,
    DEPRECATED_PROPERTY,
    FUNCTIONAL_PROPERTY,
    INVERSE_FUNCTIONAL_PROPERTY,
    IRREFLEXIVE_PROPERTY,
    NAMED_INDIVIDUAL,
    NEGATIVE_PROPERTY_ASSERTION,
    NOTHING,
    OBJECT_PROPERTY,
    ONTOLOGY,
    ONTOLOGY_PROPERTY,
    REFLEXIVE_PROPERTY,
    RESTRICTION,
    SYMMETRIC_PROPERTY,
    TRANSITIVE_PROPERTY,
    THING,
    ALL_VALUES_FROM,
    ANNOTATED_PROPERTY,
    ANNOTATED_SOURCE,
    ANNOTATED_TARGET,
    ASSERTION_PROPERTY,
    BACKWARD_COMPATIBLE_WITH,
    BOTTOM_DATA_PROPERTY,
    BOTTOM_OBJECT_PROPERTY,
    CARDINALITY,
    COMPLEMENT_OF,
    DATATYPE_COMPLEMENT_OF,
    DEPRECATED,
    DIFFERENT_FROM,
    DISJOINT_UNION_OF,
    DISJOINT_WITH,
    DISTINCT_MEMBERS,
    EQUIVALENT_CLASS,
    EQUIVALENT_PROPERTY,
    HAS_KEY,
    HAS_SELF,
    HAS_VALUE,
    IMPORTS,
    INCOMPATIBLE_WITH,
    INTERSECTION_OF,
    INVERSE_OF,
    MAX_CARDINALITY,
    MAX_QUALIFIED_CARDINALITY,
    MEMBERS,
    MIN_CARDINALITY,
    MIN_QUALIFIED_CARDINALITY,
    ON_CLASS,
    ON_DATA_RANGE,
    ON_DATATYPE,
    ONE_OF,
    ON_PROPERTIES,
    ON_PROPERTY,
    PRIOR_VERSION,
    PROPERTY_CHAIN_AXIOM,
    PROPERTY_DISJOINT_WITH,
    QUALIFIED_CARDINALITY,
    SAME_AS,
    SOME_VALUES_FROM,
    SOURCE_INDIVIDUAL,
    TARGET_INDIVIDUAL,
    TARGET_VALUE,
    TOP_DATA_PROPERTY,
    TOP_OBJECT_PROPERTY,
    UNION_OF,
    VERSION_INFO,
    VERSION_IRI,
    WITH_RESTRICTIONS,
    REAL,
    RATIONAL,
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
        assert_eq!(ALL.len(), 79);
    }
}
