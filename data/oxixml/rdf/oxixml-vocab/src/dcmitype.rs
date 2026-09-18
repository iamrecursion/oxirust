//! [DCMI Type Vocabulary](https://www.dublincore.org/specifications/dublin-core/dcmi-terms/#section-7) — a general, cross-domain list of resource types (used e.g. via `dcterms:type`).
//!
//! Source: <https://www.dublincore.org/specifications/dublin-core/dcmi-terms/#section-7>

use oxixml_model::NamedNodeRef;

/// The `dcmitype` namespace IRI (`http://purl.org/dc/dcmitype/`).
pub const NAMESPACE: &str = "http://purl.org/dc/dcmitype/";

macro_rules! term {
    ($name:ident, $local:literal, $doc:expr) => {
        #[doc = $doc]
        pub const $name: NamedNodeRef<'static> =
            NamedNodeRef::new_unchecked(concat!("http://purl.org/dc/dcmitype/", $local));
    };
}

term!(COLLECTION, "Collection", "An aggregation of resources.");
term!(DATASET, "Dataset", "Data encoded in a defined structure.");
term!(EVENT, "Event", "A non-persistent, time-based occurrence.");
term!(IMAGE, "Image", "A visual representation other than text.");
term!(
    INTERACTIVE_RESOURCE,
    "InteractiveResource",
    "A resource requiring interaction from the user to be understood, executed, or experienced."
);
term!(
    MOVING_IMAGE,
    "MovingImage",
    "A series of visual representations imparting an impression of motion when shown in succession."
);
term!(
    PHYSICAL_OBJECT,
    "PhysicalObject",
    "An inanimate, three-dimensional object or substance."
);
term!(
    SERVICE,
    "Service",
    "A system that provides one or more functions."
);
term!(
    SOFTWARE,
    "Software",
    "A computer program in source or compiled form."
);
term!(SOUND, "Sound", "A resource primarily intended to be heard.");
term!(STILL_IMAGE, "StillImage", "A static visual representation.");
term!(
    TEXT,
    "Text",
    "A resource consisting primarily of words for reading."
);

/// All 12 terms defined in the [`dcmitype`](self) module, in declaration order.
///
/// Exposed for introspection and exhaustiveness testing.
pub const ALL: &[NamedNodeRef<'static>] = &[
    COLLECTION,
    DATASET,
    EVENT,
    IMAGE,
    INTERACTIVE_RESOURCE,
    MOVING_IMAGE,
    PHYSICAL_OBJECT,
    SERVICE,
    SOFTWARE,
    SOUND,
    STILL_IMAGE,
    TEXT,
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
        assert_eq!(ALL.len(), 12);
    }
}
