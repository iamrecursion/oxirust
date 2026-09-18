//! [DCMI Metadata Terms](https://www.dublincore.org/specifications/dublin-core/dcmi-terms/) vocabulary (the `dcterms` / Dublin Core Terms namespace).
//!
//! Source: <https://www.dublincore.org/specifications/dublin-core/dcmi-terms/>

use oxixml_model::NamedNodeRef;

/// The `dcterms` namespace IRI (`http://purl.org/dc/terms/`).
pub const NAMESPACE: &str = "http://purl.org/dc/terms/";

macro_rules! term {
    ($name:ident, $local:literal, $doc:expr) => {
        #[doc = $doc]
        pub const $name: NamedNodeRef<'static> =
            NamedNodeRef::new_unchecked(concat!("http://purl.org/dc/terms/", $local));
    };
}

term!(
    AGENT,
    "Agent",
    "A resource that acts or has the power to act."
);
term!(AGENT_CLASS, "AgentClass", "A group of agents.");
term!(
    BIBLIOGRAPHIC_RESOURCE,
    "BibliographicResource",
    "A book, article, or other documentary resource."
);
term!(
    BOX,
    "Box",
    "The set of regions in space defined by their geographic coordinates according to the DCMI Box Encoding Scheme."
);
term!(
    DCMI_TYPE,
    "DCMIType",
    "The set of classes specified by the DCMI Type Vocabulary, used to categorize the nature or genre of the resource."
);
term!(
    DDC,
    "DDC",
    "The set of conceptual resources specified by the Dewey Decimal Classification."
);
term!(FILE_FORMAT, "FileFormat", "A digital resource format.");
term!(FREQUENCY, "Frequency", "A rate at which something recurs.");
term!(
    IMT,
    "IMT",
    "The set of media types specified by the Internet Assigned Numbers Authority."
);
term!(
    ISO3166,
    "ISO3166",
    "The set of codes listed in ISO 3166-1 for the representation of names of countries."
);
term!(
    ISO639,
    "ISO639",
    "The three-letter alphabetic codes listed in ISO639-2 for the representation of names of languages."
);
term!(
    JURISDICTION,
    "Jurisdiction",
    "The extent or range of judicial, law enforcement, or other authority."
);
term!(
    LCC,
    "LCC",
    "The set of conceptual resources specified by the Library of Congress Classification."
);
term!(
    LCSH,
    "LCSH",
    "The set of labeled concepts specified by the Library of Congress Subject Headings."
);
term!(
    LICENSE_DOCUMENT,
    "LicenseDocument",
    "A legal document giving official permission to do something with a resource."
);
term!(
    LINGUISTIC_SYSTEM,
    "LinguisticSystem",
    "A system of signs, symbols, sounds, gestures, or rules used in communication."
);
term!(LOCATION, "Location", "A spatial region or named place.");
term!(
    LOCATION_PERIOD_OR_JURISDICTION,
    "LocationPeriodOrJurisdiction",
    "A location, period of time, or jurisdiction."
);
term!(
    MESH,
    "MESH",
    "The set of labeled concepts specified by the Medical Subject Headings."
);
term!(MEDIA_TYPE, "MediaType", "A file format or physical medium.");
term!(
    MEDIA_TYPE_OR_EXTENT,
    "MediaTypeOrExtent",
    "A media type or extent."
);
term!(
    METHOD_OF_ACCRUAL,
    "MethodOfAccrual",
    "A method by which resources are added to a collection."
);
term!(
    METHOD_OF_INSTRUCTION,
    "MethodOfInstruction",
    "A process that is used to engender knowledge, attitudes, and skills."
);
term!(
    NLM,
    "NLM",
    "The set of conceptual resources specified by the National Library of Medicine Classification."
);
term!(
    PERIOD,
    "Period",
    "The set of time intervals defined by their limits according to the DCMI Period Encoding Scheme."
);
term!(
    PERIOD_OF_TIME,
    "PeriodOfTime",
    "An interval of time that is named or defined by its start and end dates."
);
term!(
    PHYSICAL_MEDIUM,
    "PhysicalMedium",
    "A physical material or carrier."
);
term!(PHYSICAL_RESOURCE, "PhysicalResource", "A material thing.");
term!(
    POINT,
    "Point",
    "The set of points in space defined by their geographic coordinates according to the DCMI Point Encoding Scheme."
);
term!(
    POLICY,
    "Policy",
    "A plan or course of action by an authority, intended to influence and determine decisions, actions, and other matters."
);
term!(
    PROVENANCE_STATEMENT,
    "ProvenanceStatement",
    "Any changes in ownership and custody of a resource since its creation that are significant for its authenticity, integrity, and interpretation."
);
term!(
    RFC1766,
    "RFC1766",
    "The set of tags, constructed according to RFC 1766, for the identification of languages."
);
term!(
    RFC3066,
    "RFC3066",
    "The set of tags constructed according to RFC 3066 for the identification of languages."
);
term!(
    RFC4646,
    "RFC4646",
    "The set of tags constructed according to RFC 4646 for the identification of languages."
);
term!(
    RFC5646,
    "RFC5646",
    "The set of tags constructed according to RFC 5646 for the identification of languages."
);
term!(
    RIGHTS_STATEMENT,
    "RightsStatement",
    "A statement about the intellectual property rights (IPR) held in or over a resource, a legal document giving official permission to do something with a resource, or a statement about access rights."
);
term!(
    SIZE_OR_DURATION,
    "SizeOrDuration",
    "A dimension or extent, or a time taken to play or execute."
);
term!(
    STANDARD,
    "Standard",
    "A reference point against which other things can be evaluated or compared."
);
term!(
    TGN,
    "TGN",
    "The set of places specified by the Getty Thesaurus of Geographic Names."
);
term!(
    UDC,
    "UDC",
    "The set of conceptual resources specified by the Universal Decimal Classification."
);
term!(
    URI,
    "URI",
    "The set of identifiers constructed according to the generic syntax for Uniform Resource Identifiers as specified by the Internet Engineering Task Force."
);
term!(
    W3_CDTF,
    "W3CDTF",
    "The set of dates and times constructed according to the W3C Date and Time Formats Specification."
);
term!(ABSTRACT, "abstract", "A summary of the resource.");
term!(
    ACCESS_RIGHTS,
    "accessRights",
    "Information about who access the resource or an indication of its security status."
);
term!(
    ACCRUAL_METHOD,
    "accrualMethod",
    "The method by which items are added to a collection."
);
term!(
    ACCRUAL_PERIODICITY,
    "accrualPeriodicity",
    "The frequency with which items are added to a collection."
);
term!(
    ACCRUAL_POLICY,
    "accrualPolicy",
    "The policy governing the addition of items to a collection."
);
term!(
    ALTERNATIVE,
    "alternative",
    "An alternative name for the resource."
);
term!(
    AUDIENCE,
    "audience",
    "A class of agents for whom the resource is intended or useful."
);
term!(
    AVAILABLE,
    "available",
    "Date that the resource became or will become available."
);
term!(
    BIBLIOGRAPHIC_CITATION,
    "bibliographicCitation",
    "A bibliographic reference for the resource."
);
term!(
    CONFORMS_TO,
    "conformsTo",
    "An established standard to which the described resource conforms."
);
term!(
    CONTRIBUTOR,
    "contributor",
    "An entity responsible for making contributions to the resource."
);
term!(
    COVERAGE,
    "coverage",
    "The spatial or temporal topic of the resource, spatial applicability of the resource, or jurisdiction under which the resource is relevant."
);
term!(CREATED, "created", "Date of creation of the resource.");
term!(
    CREATOR,
    "creator",
    "An entity responsible for making the resource."
);
term!(
    DATE,
    "date",
    "A point or period of time associated with an event in the lifecycle of the resource."
);
term!(
    DATE_ACCEPTED,
    "dateAccepted",
    "Date of acceptance of the resource."
);
term!(
    DATE_COPYRIGHTED,
    "dateCopyrighted",
    "Date of copyright of the resource."
);
term!(
    DATE_SUBMITTED,
    "dateSubmitted",
    "Date of submission of the resource."
);
term!(DESCRIPTION, "description", "An account of the resource.");
term!(
    EDUCATION_LEVEL,
    "educationLevel",
    "A class of agents, defined in terms of progression through an educational or training context, for which the described resource is intended."
);
term!(EXTENT, "extent", "The size or duration of the resource.");
term!(
    FORMAT,
    "format",
    "The file format, physical medium, or dimensions of the resource."
);
term!(
    HAS_FORMAT,
    "hasFormat",
    "A related resource that is substantially the same as the pre-existing described resource, but in another format."
);
term!(
    HAS_PART,
    "hasPart",
    "A related resource that is included either physically or logically in the described resource."
);
term!(
    HAS_VERSION,
    "hasVersion",
    "A related resource that is a version, edition, or adaptation of the described resource."
);
term!(
    IDENTIFIER,
    "identifier",
    "An unambiguous reference to the resource within a given context."
);
term!(
    INSTRUCTIONAL_METHOD,
    "instructionalMethod",
    "A process, used to engender knowledge, attitudes and skills, that the described resource is designed to support."
);
term!(
    IS_FORMAT_OF,
    "isFormatOf",
    "A pre-existing related resource that is substantially the same as the described resource, but in another format."
);
term!(
    IS_PART_OF,
    "isPartOf",
    "A related resource in which the described resource is physically or logically included."
);
term!(
    IS_REFERENCED_BY,
    "isReferencedBy",
    "A related resource that references, cites, or otherwise points to the described resource."
);
term!(
    IS_REPLACED_BY,
    "isReplacedBy",
    "A related resource that supplants, displaces, or supersedes the described resource."
);
term!(
    IS_REQUIRED_BY,
    "isRequiredBy",
    "A related resource that requires the described resource to support its function, delivery, or coherence."
);
term!(
    IS_VERSION_OF,
    "isVersionOf",
    "A related resource of which the described resource is a version, edition, or adaptation."
);
term!(ISSUED, "issued", "Date of formal issuance of the resource.");
term!(LANGUAGE, "language", "A language of the resource.");
term!(
    LICENSE,
    "license",
    "A legal document giving official permission to do something with the resource."
);
term!(
    MEDIATOR,
    "mediator",
    "An entity that mediates access to the resource."
);
term!(
    MEDIUM,
    "medium",
    "The material or physical carrier of the resource."
);
term!(
    MODIFIED,
    "modified",
    "Date on which the resource was changed."
);
term!(
    PROVENANCE,
    "provenance",
    "A statement of any changes in ownership and custody of the resource since its creation that are significant for its authenticity, integrity, and interpretation."
);
term!(
    PUBLISHER,
    "publisher",
    "An entity responsible for making the resource available."
);
term!(
    REFERENCES,
    "references",
    "A related resource that is referenced, cited, or otherwise pointed to by the described resource."
);
term!(RELATION, "relation", "A related resource.");
term!(
    REPLACES,
    "replaces",
    "A related resource that is supplanted, displaced, or superseded by the described resource."
);
term!(
    REQUIRES,
    "requires",
    "A related resource that is required by the described resource to support its function, delivery, or coherence."
);
term!(
    RIGHTS,
    "rights",
    "Information about rights held in and over the resource."
);
term!(
    RIGHTS_HOLDER,
    "rightsHolder",
    "A person or organization owning or managing rights over the resource."
);
term!(
    SOURCE,
    "source",
    "A related resource from which the described resource is derived."
);
term!(
    SPATIAL,
    "spatial",
    "Spatial characteristics of the resource."
);
term!(SUBJECT, "subject", "A topic of the resource.");
term!(
    TABLE_OF_CONTENTS,
    "tableOfContents",
    "A list of subunits of the resource."
);
term!(
    TEMPORAL,
    "temporal",
    "Temporal characteristics of the resource."
);
term!(TITLE, "title", "A name given to the resource.");
term!(TYPE, "type", "The nature or genre of the resource.");
term!(
    VALID,
    "valid",
    "Date (often a range) of validity of a resource."
);

/// All 97 terms defined in the [`dcterms`](self) module, in declaration order.
///
/// Exposed for introspection and exhaustiveness testing.
pub const ALL: &[NamedNodeRef<'static>] = &[
    AGENT,
    AGENT_CLASS,
    BIBLIOGRAPHIC_RESOURCE,
    BOX,
    DCMI_TYPE,
    DDC,
    FILE_FORMAT,
    FREQUENCY,
    IMT,
    ISO3166,
    ISO639,
    JURISDICTION,
    LCC,
    LCSH,
    LICENSE_DOCUMENT,
    LINGUISTIC_SYSTEM,
    LOCATION,
    LOCATION_PERIOD_OR_JURISDICTION,
    MESH,
    MEDIA_TYPE,
    MEDIA_TYPE_OR_EXTENT,
    METHOD_OF_ACCRUAL,
    METHOD_OF_INSTRUCTION,
    NLM,
    PERIOD,
    PERIOD_OF_TIME,
    PHYSICAL_MEDIUM,
    PHYSICAL_RESOURCE,
    POINT,
    POLICY,
    PROVENANCE_STATEMENT,
    RFC1766,
    RFC3066,
    RFC4646,
    RFC5646,
    RIGHTS_STATEMENT,
    SIZE_OR_DURATION,
    STANDARD,
    TGN,
    UDC,
    URI,
    W3_CDTF,
    ABSTRACT,
    ACCESS_RIGHTS,
    ACCRUAL_METHOD,
    ACCRUAL_PERIODICITY,
    ACCRUAL_POLICY,
    ALTERNATIVE,
    AUDIENCE,
    AVAILABLE,
    BIBLIOGRAPHIC_CITATION,
    CONFORMS_TO,
    CONTRIBUTOR,
    COVERAGE,
    CREATED,
    CREATOR,
    DATE,
    DATE_ACCEPTED,
    DATE_COPYRIGHTED,
    DATE_SUBMITTED,
    DESCRIPTION,
    EDUCATION_LEVEL,
    EXTENT,
    FORMAT,
    HAS_FORMAT,
    HAS_PART,
    HAS_VERSION,
    IDENTIFIER,
    INSTRUCTIONAL_METHOD,
    IS_FORMAT_OF,
    IS_PART_OF,
    IS_REFERENCED_BY,
    IS_REPLACED_BY,
    IS_REQUIRED_BY,
    IS_VERSION_OF,
    ISSUED,
    LANGUAGE,
    LICENSE,
    MEDIATOR,
    MEDIUM,
    MODIFIED,
    PROVENANCE,
    PUBLISHER,
    REFERENCES,
    RELATION,
    REPLACES,
    REQUIRES,
    RIGHTS,
    RIGHTS_HOLDER,
    SOURCE,
    SPATIAL,
    SUBJECT,
    TABLE_OF_CONTENTS,
    TEMPORAL,
    TITLE,
    TYPE,
    VALID,
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
        assert_eq!(ALL.len(), 97);
    }
}
