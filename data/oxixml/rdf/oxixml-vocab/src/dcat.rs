//! [DCAT 3](https://www.w3.org/TR/vocab-dcat-3/) — the W3C Data Catalog Vocabulary.
//!
//! Source: <https://www.w3.org/ns/dcat#>

use oxixml_model::NamedNodeRef;

/// The `dcat` namespace IRI (`http://www.w3.org/ns/dcat#`).
pub const NAMESPACE: &str = "http://www.w3.org/ns/dcat#";

macro_rules! term {
    ($name:ident, $local:literal, $doc:expr) => {
        #[doc = $doc]
        pub const $name: NamedNodeRef<'static> =
            NamedNodeRef::new_unchecked(concat!("http://www.w3.org/ns/dcat#", $local));
    };
}

term!(
    CATALOG_CLASS,
    "Catalog",
    "A curated collection of metadata about resources (e.g., datasets and data services in the context of a data catalog)."
);
term!(
    CATALOG_RECORD,
    "CatalogRecord",
    "A record in a data catalog, describing the registration of a single dataset or data service."
);
term!(
    DATA_SERVICE,
    "DataService",
    "A site or end-point providing operations related to the discovery of, access to, or processing functions on, data or related resources."
);
term!(
    DATASET_CLASS,
    "Dataset",
    "A collection of data, published or curated by a single source, and available for access or download in one or more representations."
);
term!(
    DATASET_SERIES,
    "DatasetSeries",
    "A collection of datasets that are published separately, but share some characteristics that group them."
);
term!(
    DISTRIBUTION_CLASS,
    "Distribution",
    "A specific representation of a dataset. A dataset might be available in multiple serializations that may differ in various ways, including natural language, media-type or format, schematic organization, temporal and spatial resolution, level of detail or profiles (which might specify any or all of the above)."
);
term!(
    RELATIONSHIP,
    "Relationship",
    "An association class for attaching additional information to a relationship between DCAT Resources."
);
term!(
    RESOURCE_CLASS,
    "Resource",
    "Resource published or curated by a single agent."
);
term!(
    ROLE,
    "Role",
    "A role is the function of a resource or agent with respect to another resource, in the context of resource attribution or resource relationships."
);
term!(
    ACCESS_SERVICE,
    "accessService",
    "A site or end-point that gives access to the distribution of the dataset."
);
term!(
    ACCESS_URL,
    "accessURL",
    "A URL of a resource that gives access to a distribution of the dataset. E.g. landing page, feed, SPARQL endpoint. Use for all cases except a simple download link, in which case downloadURL is preferred."
);
term!(
    BBOX,
    "bbox",
    "The geographic bounding box of a spatial thing [SDW-BP]."
);
term!(
    BYTE_SIZE,
    "byteSize",
    "The size of a distribution in bytes."
);
term!(
    CATALOG,
    "catalog",
    "A catalog that is listed in the catalog."
);
term!(
    CENTROID,
    "centroid",
    "The geographic center (centroid) of a spatial thing [SDW-BP]."
);
term!(
    COMPRESS_FORMAT,
    "compressFormat",
    "The compression format of the distribution in which the data is contained in a compressed form, e.g. to reduce the size of the downloadable file."
);
term!(
    CONTACT_POINT,
    "contactPoint",
    "Relevant contact information for the catalogued resource. Use of vCard is recommended."
);
term!(
    DATASET,
    "dataset",
    "A dataset that is listed in the catalog."
);
term!(
    DISTRIBUTION,
    "distribution",
    "An available distribution of the dataset."
);
term!(
    DOWNLOAD_URL,
    "downloadURL",
    "The URL of the downloadable file in a given format. E.g. CSV file or RDF file. The format is indicated by the distribution's dcterms:format and/or dcat:mediaType."
);
term!(END_DATE, "endDate", "The end of the period.");
term!(
    ENDPOINT_DESCRIPTION,
    "endpointDescription",
    "A description of the service end-point, including its operations, parameters etc."
);
term!(
    ENDPOINT_URL,
    "endpointURL",
    "The root location or primary endpoint of the service (a web-resolvable IRI)."
);
term!(
    FIRST,
    "first",
    "The first resource in an ordered collection or series of resources, to which the current resource belongs."
);
term!(
    HAS_CURRENT_VERSION,
    "hasCurrentVersion",
    "This resource has a more specific, versioned resource with equivalent content \\[PAV\\]."
);
term!(
    HAS_VERSION,
    "hasVersion",
    "This resource has a more specific, versioned resource \\[PAV\\]."
);
term!(
    HAD_ROLE,
    "hadRole",
    "The function of an entity or agent with respect to another entity or resource."
);
term!(
    IS_DISTRIBUTION_OF,
    "isDistributionOf",
    "Relates a distribution to the dataset it is a distribution of; the inverse of `dcat:distribution`."
);
term!(
    IS_VERSION_OF,
    "isVersionOf",
    "Relates a dataset to another dataset of which it is a new version; the inverse of `dcat:hasVersion`."
);
term!(
    IN_CATALOG,
    "inCatalog",
    "Relates a resource to the catalog it is listed in; the inverse of `dcat:resource`."
);
term!(
    IN_SERIES,
    "inSeries",
    "A dataset series of which the dataset is part."
);
term!(
    KEYWORD,
    "keyword",
    "A keyword or tag describing a resource."
);
term!(
    LANDING_PAGE,
    "landingPage",
    "A Web page that can be navigated to in a Web browser to gain access to the catalog, a dataset, its distributions and/or additional information."
);
term!(
    LAST,
    "last",
    "The last resource in an ordered collection or series of resources, to which the current resource belongs."
);
term!(
    MEDIA_TYPE,
    "mediaType",
    "The media type of the distribution as defined by IANA"
);
term!(
    NEXT,
    "next",
    "Relates a dataset series member to the next member of the series; the inverse of `dcat:prev`."
);
term!(
    NEXT_VERSION,
    "nextVersion",
    "Relates a dataset to a later version of itself; the inverse of `dcat:previousVersion`."
);
term!(
    PACKAGE_FORMAT,
    "packageFormat",
    "The package format of the distribution in which one or more data files are grouped together, e.g. to enable a set of related files to be downloaded together."
);
term!(
    PREV,
    "prev",
    "The previous resource (before the current one) in an ordered collection or series of resources."
);
term!(
    PREVIOUS_VERSION,
    "previousVersion",
    "The previous version of a resource in a lineage \\[PAV\\]."
);
term!(
    QUALIFIED_RELATION,
    "qualifiedRelation",
    "Link to a description of a relationship with another resource."
);
term!(
    RECORD,
    "record",
    "A record describing the registration of a single dataset or data service that is part of the catalog."
);
term!(
    RESOURCE,
    "resource",
    "A resource that is listed in the catalog."
);
term!(
    SERIES_MEMBER,
    "seriesMember",
    "Relates a dataset series to a dataset that is part of the series; the inverse of `dcat:inSeries`."
);
term!(
    SERVES_DATASET,
    "servesDataset",
    "A collection of data that this DataService can distribute."
);
term!(
    SERVICE,
    "service",
    "A service that is listed in the catalog."
);
term!(
    SPATIAL_RESOLUTION_IN_METERS,
    "spatialResolutionInMeters",
    "minimum spatial separation resolvable in a dataset, measured in meters."
);
term!(START_DATE, "startDate", "The start of the period");
term!(
    TEMPORAL_RESOLUTION,
    "temporalResolution",
    "minimum time period resolvable in a dataset."
);
term!(
    THEME,
    "theme",
    "A main category of the resource. A resource can have multiple themes."
);
term!(
    THEME_TAXONOMY,
    "themeTaxonomy",
    "The knowledge organization system (KOS) used to classify catalog's datasets."
);
term!(
    VERSION,
    "version",
    "The version indicator (name or identifier) of a resource."
);

/// All 52 terms defined in the [`dcat`](self) module, in declaration order.
///
/// Exposed for introspection and exhaustiveness testing.
pub const ALL: &[NamedNodeRef<'static>] = &[
    CATALOG_CLASS,
    CATALOG_RECORD,
    DATA_SERVICE,
    DATASET_CLASS,
    DATASET_SERIES,
    DISTRIBUTION_CLASS,
    RELATIONSHIP,
    RESOURCE_CLASS,
    ROLE,
    ACCESS_SERVICE,
    ACCESS_URL,
    BBOX,
    BYTE_SIZE,
    CATALOG,
    CENTROID,
    COMPRESS_FORMAT,
    CONTACT_POINT,
    DATASET,
    DISTRIBUTION,
    DOWNLOAD_URL,
    END_DATE,
    ENDPOINT_DESCRIPTION,
    ENDPOINT_URL,
    FIRST,
    HAS_CURRENT_VERSION,
    HAS_VERSION,
    HAD_ROLE,
    IS_DISTRIBUTION_OF,
    IS_VERSION_OF,
    IN_CATALOG,
    IN_SERIES,
    KEYWORD,
    LANDING_PAGE,
    LAST,
    MEDIA_TYPE,
    NEXT,
    NEXT_VERSION,
    PACKAGE_FORMAT,
    PREV,
    PREVIOUS_VERSION,
    QUALIFIED_RELATION,
    RECORD,
    RESOURCE,
    SERIES_MEMBER,
    SERVES_DATASET,
    SERVICE,
    SPATIAL_RESOLUTION_IN_METERS,
    START_DATE,
    TEMPORAL_RESOLUTION,
    THEME,
    THEME_TAXONOMY,
    VERSION,
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
        assert_eq!(ALL.len(), 52);
    }
}
