//! [OGC GeoSPARQL 1.0](http://www.opengis.net/doc/IS/geosparql/1.0) vocabulary for representing and querying geospatial RDF data.
//!
//! Source: <http://www.opengis.net/ont/geosparql#>

use oxixml_model::NamedNodeRef;

/// The `geo` namespace IRI (`http://www.opengis.net/ont/geosparql#`).
pub const NAMESPACE: &str = "http://www.opengis.net/ont/geosparql#";

macro_rules! term {
    ($name:ident, $local:literal, $doc:expr) => {
        #[doc = $doc]
        pub const $name: NamedNodeRef<'static> =
            NamedNodeRef::new_unchecked(concat!("http://www.opengis.net/ont/geosparql#", $local));
    };
}

term!(
    WKT_LITERAL,
    "wktLiteral",
    "A Well-known Text serialization of a geometry object."
);
term!(
    GML_LITERAL,
    "gmlLiteral",
    "A GML serialization of a geometry object."
);
term!(
    SPATIAL_OBJECT,
    "SpatialObject",
    "The class spatial-object represents everything that can have a spatial representation. It is superclass of feature and geometry."
);
term!(
    GEOMETRY,
    "Geometry",
    "The class represents the top-level geometry type. This class is equivalent to the UML class GM_Object defined in ISO 19107, and it is superclass of all geometry types."
);
term!(
    FEATURE,
    "Feature",
    "This class represents the top-level feature type. This class is equivalent to GFI_Feature defined in ISO 19156:2011, and it is superclass of all feature types."
);
term!(
    DEFAULT_GEOMETRY,
    "defaultGeometry",
    "The default geometry to be used in spatial calculations. It is Usually the most detailed geometry."
);
term!(
    HAS_GEOMETRY,
    "hasGeometry",
    "A spatial representation for a given feature."
);
term!(
    EH_DISJOINT,
    "ehDisjoint",
    "Exists if the subject SpatialObject is spatially disjoint from the object SpatialObject. DE-9IM: FF*FF****"
);
term!(
    RCC8NTPP,
    "rcc8ntpp",
    "Exists if the subject SpatialObject is spatially inside the object SpatialObject. DE-9IM: TFFTFFTTT"
);
term!(
    EH_CONTAINS,
    "ehContains",
    "Exists if the subject SpatialObject spatially contains the object SpatialObject. DE-9IM: T*TFF*FF*"
);
term!(
    RCC8TPPI,
    "rcc8tppi",
    "Exists if the subject SpatialObject spatially covers the object SpatialObject. DE-9IM: TTTFTTFFT"
);
term!(
    RCC8EC,
    "rcc8ec",
    "Exists if the subject SpatialObject spatially meets the object SpatialObject. DE-9IM: FFTFTTTTT"
);
term!(
    SF_EQUALS,
    "sfEquals",
    "Exists if the subject SpatialObject spatially equals the object SpatialObject. DE-9IM: TFFFTFFFT"
);
term!(
    EH_OVERLAP,
    "ehOverlap",
    "Exists if the subject SpatialObject spatially overlaps the object SpatialObject. DE-9IM: T*T***T**"
);
term!(
    RCC8DC,
    "rcc8dc",
    "Exists if the subject SpatialObject is spatially disjoint from the object SpatialObject. DE-9IM: FFTFFTTTT"
);
term!(
    EH_COVERS,
    "ehCovers",
    "Exists if the subject SpatialObject spatially covers the object SpatialObject. DE-9IM: T*TFT*FF*"
);
term!(
    EH_COVERED_BY,
    "ehCoveredBy",
    "Exists if the subject SpatialObject is spatially covered by the object SpatialObject. DE-9IM: TFF*TFT**"
);
term!(
    SF_CONTAINS,
    "sfContains",
    "Exists if the subject SpatialObject spatially contains the object SpatialObject. DE-9IM: T*****FF*"
);
term!(
    RCC8TPP,
    "rcc8tpp",
    "Exists if the subject SpatialObject is spatially covered by the object SpatialObject. DE-9IM: TFFTTFTTT"
);
term!(
    EH_EQUALS,
    "ehEquals",
    "Exists if the subject SpatialObject spatially equals the object SpatialObject. DE-9IM: TFFFTFFFT"
);
term!(
    RCC8PO,
    "rcc8po",
    "Exists if the subject SpatialObject spatially overlaps the object SpatialObject. DE-9IM: TTTTTTTTT"
);
term!(
    SF_OVERLAPS,
    "sfOverlaps",
    "Exists if the subject SpatialObject spatially overlaps the object SpatialObject. DE-9IM: T*T***T**"
);
term!(
    SF_WITHIN,
    "sfWithin",
    "Exists if the subject SpatialObject is spatially within the object SpatialObject. DE-9IM: T*F**F***"
);
term!(
    SF_TOUCHES,
    "sfTouches",
    "Exists if the subject SpatialObject spatially touches the object SpatialObject. DE-9IM: FT******* ^ F**T***** ^ F***T****"
);
term!(
    SF_INTERSECTS,
    "sfIntersects",
    "Exists if the subject SpatialObject is not spatially disjoint from the object SpatialObject. DE-9IM: T******** ^ *T******* ^ ***T***** ^ ****T****"
);
term!(
    SF_CROSSES,
    "sfCrosses",
    "Exists if the subject SpatialObject spatially crosses the object SpatialObject. DE-9IM: T*T******"
);
term!(
    RCC8NTPPI,
    "rcc8ntppi",
    "Exists if the subject SpatialObject spatially contains the object SpatialObject. DE-9IM: TTTFFTFFT"
);
term!(
    RCC8EQ,
    "rcc8eq",
    "Exists if the subject SpatialObject spatially equals the object SpatialObject. DE-9IM: TFFFTFFFT"
);
term!(
    EH_MEET,
    "ehMeet",
    "Exists if the subject SpatialObject spatially meets the object SpatialObject. DE-9IM: FT******* ^ F**T***** ^ F***T****"
);
term!(
    SF_DISJOINT,
    "sfDisjoint",
    "Exists if the subject SpatialObject is spatially disjoint from the object SpatialObject. DE-9IM: FF*FF****"
);
term!(
    EH_INSIDE,
    "ehInside",
    "Exists if the subject SpatialObject is spatially inside the object SpatialObject. DE-9IM: TFF*FFT**"
);
term!(
    SPATIAL_DIMENSION,
    "spatialDimension",
    "The number of measurements or axes needed to describe the spatial position of this geometry in a coordinate system."
);
term!(
    IS_EMPTY,
    "isEmpty",
    "(true) if this geometric object is the empty Geometry. If true, then this geometric object represents the empty point set for the coordinate space."
);
term!(
    COORDINATE_DIMENSION,
    "coordinateDimension",
    "The number of measurements or axes needed to describe the position of this geometry in a coordinate system."
);
term!(AS_WKT, "asWKT", "The WKT serialization of a geometry");
term!(
    HAS_SERIALIZATION,
    "hasSerialization",
    "Connects a geometry object with its text-based serialization."
);
term!(AS_GML, "asGML", "The GML serialization of a geometry");
term!(
    IS_SIMPLE,
    "isSimple",
    "(true) if this geometric object has no anomalous geometric points, such as self intersection or self tangency."
);
term!(
    DIMENSION,
    "dimension",
    "The topological dimension of this geometric object, which must be less than or equal to the coordinate dimension. In non-homogeneous collections, this will return the largest topological dimension of the contained objects."
);

/// All 39 terms defined in the [`geo`](self) module, in declaration order.
///
/// Exposed for introspection and exhaustiveness testing.
pub const ALL: &[NamedNodeRef<'static>] = &[
    WKT_LITERAL,
    GML_LITERAL,
    SPATIAL_OBJECT,
    GEOMETRY,
    FEATURE,
    DEFAULT_GEOMETRY,
    HAS_GEOMETRY,
    EH_DISJOINT,
    RCC8NTPP,
    EH_CONTAINS,
    RCC8TPPI,
    RCC8EC,
    SF_EQUALS,
    EH_OVERLAP,
    RCC8DC,
    EH_COVERS,
    EH_COVERED_BY,
    SF_CONTAINS,
    RCC8TPP,
    EH_EQUALS,
    RCC8PO,
    SF_OVERLAPS,
    SF_WITHIN,
    SF_TOUCHES,
    SF_INTERSECTS,
    SF_CROSSES,
    RCC8NTPPI,
    RCC8EQ,
    EH_MEET,
    SF_DISJOINT,
    EH_INSIDE,
    SPATIAL_DIMENSION,
    IS_EMPTY,
    COORDINATE_DIMENSION,
    AS_WKT,
    HAS_SERIALIZATION,
    AS_GML,
    IS_SIMPLE,
    DIMENSION,
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
        assert_eq!(ALL.len(), 39);
    }
}
