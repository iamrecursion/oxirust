//! [W3C Basic Geo (WGS84 lat/long) Vocabulary](https://www.w3.org/2003/01/geo/) — a simple, lightweight vocabulary for representing WGS84-datum latitude/longitude positions.
//!
//! Source: <https://www.w3.org/2003/01/geo/wgs84_pos#>

use oxixml_model::NamedNodeRef;

/// The `wgs84` namespace IRI (`http://www.w3.org/2003/01/geo/wgs84_pos#`).
pub const NAMESPACE: &str = "http://www.w3.org/2003/01/geo/wgs84_pos#";

macro_rules! term {
    ($name:ident, $local:literal, $doc:expr) => {
        #[doc = $doc]
        pub const $name: NamedNodeRef<'static> = NamedNodeRef::new_unchecked(concat!(
            "http://www.w3.org/2003/01/geo/wgs84_pos#",
            $local
        ));
    };
}

term!(
    SPATIAL_THING,
    "SpatialThing",
    "Anything with spatial extent, i.e. size, shape, or position. e.g. people, places, bowling balls, as well as abstract areas like cubes."
);
term!(
    POINT,
    "Point",
    "A point, typically described using a coordinate system relative to Earth, such as WGS84."
);
term!(
    LAT,
    "lat",
    "The WGS84 latitude of a SpatialThing (decimal degrees)."
);
term!(
    LOCATION,
    "location",
    "The relation between something and the point, or other geometrical thing in space, where it is. For example, the realtionship between a radio tower and a Point with a given lat and long. Or a relationship between a park and its outline as a closed arc of points, or a road and its location as a arc (a sequence of points). Clearly in practice there will be limit to the accuracy of any such statement, but one would expect an accuracy appropriate for the size of the object and uses such as mapping ."
);
term!(
    LONG,
    "long",
    "The WGS84 longitude of a SpatialThing (decimal degrees)."
);
term!(
    ALT,
    "alt",
    "The WGS84 altitude of a SpatialThing (decimal meters above the local reference ellipsoid)."
);
term!(
    LAT_LONG,
    "lat_long",
    "A comma-separated representation of a latitude, longitude coordinate."
);

/// All 7 terms defined in the [`wgs84`](self) module, in declaration order.
///
/// Exposed for introspection and exhaustiveness testing.
pub const ALL: &[NamedNodeRef<'static>] =
    &[SPATIAL_THING, POINT, LAT, LOCATION, LONG, ALT, LAT_LONG];

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
        assert_eq!(ALL.len(), 7);
    }
}
