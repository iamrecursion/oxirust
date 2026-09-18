//! Ready-to-use [`NamedNodeRef`](super::NamedNodeRef)s for basic RDF vocabularies.
//!
//! Each constant is a `'static` [`NamedNodeRef`](super::NamedNodeRef) built with
//! [`NamedNodeRef::new_unchecked`](super::NamedNodeRef::new_unchecked), so it is [`Copy`],
//! allocates nothing, and can be used directly, e.g.
//! `Literal::new_typed_literal("1", xsd::INTEGER)`.

pub mod rdf {
    //! [RDF](https://www.w3.org/TR/rdf11-concepts/) vocabulary.
    use crate::named_node::NamedNodeRef;

    /// The class of containers of alternatives.
    pub const ALT: NamedNodeRef<'static> =
        NamedNodeRef::new_unchecked("http://www.w3.org/1999/02/22-rdf-syntax-ns#Alt");
    /// The class of unordered containers.
    pub const BAG: NamedNodeRef<'static> =
        NamedNodeRef::new_unchecked("http://www.w3.org/1999/02/22-rdf-syntax-ns#Bag");
    /// The class of language-tagged string literal values with a base direction.
    #[cfg(feature = "rdf-12")]
    #[cfg_attr(docsrs, doc(cfg(feature = "rdf-12")))]
    pub const DIR_LANG_STRING: NamedNodeRef<'static> =
        NamedNodeRef::new_unchecked("http://www.w3.org/1999/02/22-rdf-syntax-ns#dirLangString");
    /// The first item in the subject RDF list.
    pub const FIRST: NamedNodeRef<'static> =
        NamedNodeRef::new_unchecked("http://www.w3.org/1999/02/22-rdf-syntax-ns#first");
    /// The class of HTML literal values.
    pub const HTML: NamedNodeRef<'static> =
        NamedNodeRef::new_unchecked("http://www.w3.org/1999/02/22-rdf-syntax-ns#HTML");
    /// The datatype of RDF literals storing JSON content.
    #[cfg(feature = "rdf-12")]
    #[cfg_attr(docsrs, doc(cfg(feature = "rdf-12")))]
    pub const JSON: NamedNodeRef<'static> =
        NamedNodeRef::new_unchecked("http://www.w3.org/1999/02/22-rdf-syntax-ns#JSON");
    /// The datatype of language-tagged string values.
    pub const LANG_STRING: NamedNodeRef<'static> =
        NamedNodeRef::new_unchecked("http://www.w3.org/1999/02/22-rdf-syntax-ns#langString");
    /// The class of RDF lists.
    pub const LIST: NamedNodeRef<'static> =
        NamedNodeRef::new_unchecked("http://www.w3.org/1999/02/22-rdf-syntax-ns#List");
    /// The empty list.
    pub const NIL: NamedNodeRef<'static> =
        NamedNodeRef::new_unchecked("http://www.w3.org/1999/02/22-rdf-syntax-ns#nil");
    /// The object of the subject RDF statement.
    pub const OBJECT: NamedNodeRef<'static> =
        NamedNodeRef::new_unchecked("http://www.w3.org/1999/02/22-rdf-syntax-ns#object");
    /// The predicate of the subject RDF statement.
    pub const PREDICATE: NamedNodeRef<'static> =
        NamedNodeRef::new_unchecked("http://www.w3.org/1999/02/22-rdf-syntax-ns#predicate");
    /// The class of RDF properties.
    pub const PROPERTY: NamedNodeRef<'static> =
        NamedNodeRef::new_unchecked("http://www.w3.org/1999/02/22-rdf-syntax-ns#Property");
    /// Associates a resource (reifier) with a triple (proposition).
    #[cfg(feature = "rdf-12")]
    #[cfg_attr(docsrs, doc(cfg(feature = "rdf-12")))]
    pub const REIFIES: NamedNodeRef<'static> =
        NamedNodeRef::new_unchecked("http://www.w3.org/1999/02/22-rdf-syntax-ns#reifies");
    /// The rest of the subject RDF list after the first item.
    pub const REST: NamedNodeRef<'static> =
        NamedNodeRef::new_unchecked("http://www.w3.org/1999/02/22-rdf-syntax-ns#rest");
    /// The class of ordered containers.
    pub const SEQ: NamedNodeRef<'static> =
        NamedNodeRef::new_unchecked("http://www.w3.org/1999/02/22-rdf-syntax-ns#Seq");
    /// The class of RDF statements.
    pub const STATEMENT: NamedNodeRef<'static> =
        NamedNodeRef::new_unchecked("http://www.w3.org/1999/02/22-rdf-syntax-ns#Statement");
    /// The subject of the subject RDF statement.
    pub const SUBJECT: NamedNodeRef<'static> =
        NamedNodeRef::new_unchecked("http://www.w3.org/1999/02/22-rdf-syntax-ns#subject");
    /// The subject is an instance of a class.
    pub const TYPE: NamedNodeRef<'static> =
        NamedNodeRef::new_unchecked("http://www.w3.org/1999/02/22-rdf-syntax-ns#type");
    /// Idiomatic property used for structured values.
    pub const VALUE: NamedNodeRef<'static> =
        NamedNodeRef::new_unchecked("http://www.w3.org/1999/02/22-rdf-syntax-ns#value");
    /// The class of XML literal values.
    pub const XML_LITERAL: NamedNodeRef<'static> =
        NamedNodeRef::new_unchecked("http://www.w3.org/1999/02/22-rdf-syntax-ns#XMLLiteral");
}

pub mod rdfs {
    //! [RDFS](https://www.w3.org/TR/rdf-schema/) vocabulary.
    use crate::named_node::NamedNodeRef;

    /// The class of classes.
    pub const CLASS: NamedNodeRef<'static> =
        NamedNodeRef::new_unchecked("http://www.w3.org/2000/01/rdf-schema#Class");
    /// A description of the subject resource.
    pub const COMMENT: NamedNodeRef<'static> =
        NamedNodeRef::new_unchecked("http://www.w3.org/2000/01/rdf-schema#comment");
    /// The class of RDF containers.
    pub const CONTAINER: NamedNodeRef<'static> =
        NamedNodeRef::new_unchecked("http://www.w3.org/2000/01/rdf-schema#Container");
    /// The class of container membership properties, `rdf:_1`, `rdf:_2`, …, all of which are
    /// sub-properties of `member`.
    pub const CONTAINER_MEMBERSHIP_PROPERTY: NamedNodeRef<'static> = NamedNodeRef::new_unchecked(
        "http://www.w3.org/2000/01/rdf-schema#ContainerMembershipProperty",
    );
    /// The class of RDF datatypes.
    pub const DATATYPE: NamedNodeRef<'static> =
        NamedNodeRef::new_unchecked("http://www.w3.org/2000/01/rdf-schema#Datatype");
    /// A domain of the subject property.
    pub const DOMAIN: NamedNodeRef<'static> =
        NamedNodeRef::new_unchecked("http://www.w3.org/2000/01/rdf-schema#domain");
    /// The definition of the subject resource.
    pub const IS_DEFINED_BY: NamedNodeRef<'static> =
        NamedNodeRef::new_unchecked("http://www.w3.org/2000/01/rdf-schema#isDefinedBy");
    /// A human-readable name for the subject.
    pub const LABEL: NamedNodeRef<'static> =
        NamedNodeRef::new_unchecked("http://www.w3.org/2000/01/rdf-schema#label");
    /// The class of literal values, e.g. textual strings and integers.
    pub const LITERAL: NamedNodeRef<'static> =
        NamedNodeRef::new_unchecked("http://www.w3.org/2000/01/rdf-schema#Literal");
    /// A member of the subject resource.
    pub const MEMBER: NamedNodeRef<'static> =
        NamedNodeRef::new_unchecked("http://www.w3.org/2000/01/rdf-schema#member");
    /// A range of the subject property.
    pub const RANGE: NamedNodeRef<'static> =
        NamedNodeRef::new_unchecked("http://www.w3.org/2000/01/rdf-schema#range");
    /// The class resource, everything.
    pub const RESOURCE: NamedNodeRef<'static> =
        NamedNodeRef::new_unchecked("http://www.w3.org/2000/01/rdf-schema#Resource");
    /// Further information about the subject resource.
    pub const SEE_ALSO: NamedNodeRef<'static> =
        NamedNodeRef::new_unchecked("http://www.w3.org/2000/01/rdf-schema#seeAlso");
    /// The subject is a subclass of a class.
    pub const SUB_CLASS_OF: NamedNodeRef<'static> =
        NamedNodeRef::new_unchecked("http://www.w3.org/2000/01/rdf-schema#subClassOf");
    /// The subject is a subproperty of a property.
    pub const SUB_PROPERTY_OF: NamedNodeRef<'static> =
        NamedNodeRef::new_unchecked("http://www.w3.org/2000/01/rdf-schema#subPropertyOf");
}

pub mod xsd {
    //! [RDF compatible XSD datatypes](https://www.w3.org/TR/rdf11-concepts/#dfn-rdf-compatible-xsd-types).
    use crate::named_node::NamedNodeRef;

    /// Absolute or relative URIs and IRIs.
    pub const ANY_URI: NamedNodeRef<'static> =
        NamedNodeRef::new_unchecked("http://www.w3.org/2001/XMLSchema#anyURI");
    /// Base64-encoded binary data.
    pub const BASE_64_BINARY: NamedNodeRef<'static> =
        NamedNodeRef::new_unchecked("http://www.w3.org/2001/XMLSchema#base64Binary");
    /// true, false.
    pub const BOOLEAN: NamedNodeRef<'static> =
        NamedNodeRef::new_unchecked("http://www.w3.org/2001/XMLSchema#boolean");
    /// −128…+127 (8 bit).
    pub const BYTE: NamedNodeRef<'static> =
        NamedNodeRef::new_unchecked("http://www.w3.org/2001/XMLSchema#byte");
    /// Dates (yyyy-mm-dd) with or without timezone.
    pub const DATE: NamedNodeRef<'static> =
        NamedNodeRef::new_unchecked("http://www.w3.org/2001/XMLSchema#date");
    /// Duration of time (days, hours, minutes, seconds only).
    pub const DAY_TIME_DURATION: NamedNodeRef<'static> =
        NamedNodeRef::new_unchecked("http://www.w3.org/2001/XMLSchema#dayTimeDuration");
    /// Date and time with or without timezone.
    pub const DATE_TIME: NamedNodeRef<'static> =
        NamedNodeRef::new_unchecked("http://www.w3.org/2001/XMLSchema#dateTime");
    /// Date and time with required timezone.
    pub const DATE_TIME_STAMP: NamedNodeRef<'static> =
        NamedNodeRef::new_unchecked("http://www.w3.org/2001/XMLSchema#dateTimeStamp");
    /// Arbitrary-precision decimal numbers.
    pub const DECIMAL: NamedNodeRef<'static> =
        NamedNodeRef::new_unchecked("http://www.w3.org/2001/XMLSchema#decimal");
    /// 64-bit floating point numbers incl. ±Inf, ±0, NaN.
    pub const DOUBLE: NamedNodeRef<'static> =
        NamedNodeRef::new_unchecked("http://www.w3.org/2001/XMLSchema#double");
    /// Duration of time.
    pub const DURATION: NamedNodeRef<'static> =
        NamedNodeRef::new_unchecked("http://www.w3.org/2001/XMLSchema#duration");
    /// 32-bit floating point numbers incl. ±Inf, ±0, NaN.
    pub const FLOAT: NamedNodeRef<'static> =
        NamedNodeRef::new_unchecked("http://www.w3.org/2001/XMLSchema#float");
    /// Gregorian calendar day of the month.
    pub const G_DAY: NamedNodeRef<'static> =
        NamedNodeRef::new_unchecked("http://www.w3.org/2001/XMLSchema#gDay");
    /// Gregorian calendar month.
    pub const G_MONTH: NamedNodeRef<'static> =
        NamedNodeRef::new_unchecked("http://www.w3.org/2001/XMLSchema#gMonth");
    /// Gregorian calendar month and day.
    pub const G_MONTH_DAY: NamedNodeRef<'static> =
        NamedNodeRef::new_unchecked("http://www.w3.org/2001/XMLSchema#gMonthDay");
    /// Gregorian calendar year.
    pub const G_YEAR: NamedNodeRef<'static> =
        NamedNodeRef::new_unchecked("http://www.w3.org/2001/XMLSchema#gYear");
    /// Gregorian calendar year and month.
    pub const G_YEAR_MONTH: NamedNodeRef<'static> =
        NamedNodeRef::new_unchecked("http://www.w3.org/2001/XMLSchema#gYearMonth");
    /// Hex-encoded binary data.
    pub const HEX_BINARY: NamedNodeRef<'static> =
        NamedNodeRef::new_unchecked("http://www.w3.org/2001/XMLSchema#hexBinary");
    /// −2147483648…+2147483647 (32 bit).
    pub const INT: NamedNodeRef<'static> =
        NamedNodeRef::new_unchecked("http://www.w3.org/2001/XMLSchema#int");
    /// Arbitrary-size integer numbers.
    pub const INTEGER: NamedNodeRef<'static> =
        NamedNodeRef::new_unchecked("http://www.w3.org/2001/XMLSchema#integer");
    /// Language tags per [BCP47](http://tools.ietf.org/html/bcp47).
    pub const LANGUAGE: NamedNodeRef<'static> =
        NamedNodeRef::new_unchecked("http://www.w3.org/2001/XMLSchema#language");
    /// −9223372036854775808…+9223372036854775807 (64 bit).
    pub const LONG: NamedNodeRef<'static> =
        NamedNodeRef::new_unchecked("http://www.w3.org/2001/XMLSchema#long");
    /// XML Names.
    pub const NAME: NamedNodeRef<'static> =
        NamedNodeRef::new_unchecked("http://www.w3.org/2001/XMLSchema#Name");
    /// XML NCName.
    pub const NC_NAME: NamedNodeRef<'static> =
        NamedNodeRef::new_unchecked("http://www.w3.org/2001/XMLSchema#NCName");
    /// Integer numbers < 0.
    pub const NEGATIVE_INTEGER: NamedNodeRef<'static> =
        NamedNodeRef::new_unchecked("http://www.w3.org/2001/XMLSchema#negativeInteger");
    /// XML NMTOKENs.
    pub const NMTOKEN: NamedNodeRef<'static> =
        NamedNodeRef::new_unchecked("http://www.w3.org/2001/XMLSchema#NMTOKEN");
    /// Integer numbers ≥ 0.
    pub const NON_NEGATIVE_INTEGER: NamedNodeRef<'static> =
        NamedNodeRef::new_unchecked("http://www.w3.org/2001/XMLSchema#nonNegativeInteger");
    /// Integer numbers ≤ 0.
    pub const NON_POSITIVE_INTEGER: NamedNodeRef<'static> =
        NamedNodeRef::new_unchecked("http://www.w3.org/2001/XMLSchema#nonPositiveInteger");
    /// Whitespace-normalized strings.
    pub const NORMALIZED_STRING: NamedNodeRef<'static> =
        NamedNodeRef::new_unchecked("http://www.w3.org/2001/XMLSchema#normalizedString");
    /// Integer numbers > 0.
    pub const POSITIVE_INTEGER: NamedNodeRef<'static> =
        NamedNodeRef::new_unchecked("http://www.w3.org/2001/XMLSchema#positiveInteger");
    /// Times (hh:mm:ss.sss…) with or without timezone.
    pub const TIME: NamedNodeRef<'static> =
        NamedNodeRef::new_unchecked("http://www.w3.org/2001/XMLSchema#time");
    /// −32768…+32767 (16 bit).
    pub const SHORT: NamedNodeRef<'static> =
        NamedNodeRef::new_unchecked("http://www.w3.org/2001/XMLSchema#short");
    /// Character strings (but not all Unicode character strings).
    pub const STRING: NamedNodeRef<'static> =
        NamedNodeRef::new_unchecked("http://www.w3.org/2001/XMLSchema#string");
    /// Tokenized strings.
    pub const TOKEN: NamedNodeRef<'static> =
        NamedNodeRef::new_unchecked("http://www.w3.org/2001/XMLSchema#token");
    /// 0…255 (8 bit).
    pub const UNSIGNED_BYTE: NamedNodeRef<'static> =
        NamedNodeRef::new_unchecked("http://www.w3.org/2001/XMLSchema#unsignedByte");
    /// 0…4294967295 (32 bit).
    pub const UNSIGNED_INT: NamedNodeRef<'static> =
        NamedNodeRef::new_unchecked("http://www.w3.org/2001/XMLSchema#unsignedInt");
    /// 0…18446744073709551615 (64 bit).
    pub const UNSIGNED_LONG: NamedNodeRef<'static> =
        NamedNodeRef::new_unchecked("http://www.w3.org/2001/XMLSchema#unsignedLong");
    /// 0…65535 (16 bit).
    pub const UNSIGNED_SHORT: NamedNodeRef<'static> =
        NamedNodeRef::new_unchecked("http://www.w3.org/2001/XMLSchema#unsignedShort");
    /// Duration of time (months and years only).
    pub const YEAR_MONTH_DURATION: NamedNodeRef<'static> =
        NamedNodeRef::new_unchecked("http://www.w3.org/2001/XMLSchema#yearMonthDuration");
}

pub mod geosparql {
    //! [GeoSpatial](https://opengeospatial.github.io/ogc-geosparql/) vocabulary.
    use crate::named_node::NamedNodeRef;

    /// Geospatial datatype like `"Point({longitude} {latitude})"^^geo:wktLiteral`.
    pub const WKT_LITERAL: NamedNodeRef<'static> =
        NamedNodeRef::new_unchecked("http://www.opengis.net/ont/geosparql#wktLiteral");
}
