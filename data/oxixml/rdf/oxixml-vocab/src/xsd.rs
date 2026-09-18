//! [XML Schema Part 2: Datatypes](https://www.w3.org/TR/xmlschema11-2/) built-in datatypes, as used by RDF-compatible XSD typed literals.
//!
//! Source: <https://www.w3.org/TR/xmlschema11-2/#built-in-datatypes>

use oxixml_model::NamedNodeRef;

/// The `xsd` namespace IRI (`http://www.w3.org/2001/XMLSchema#`).
pub const NAMESPACE: &str = "http://www.w3.org/2001/XMLSchema#";

macro_rules! term {
    ($name:ident, $local:literal, $doc:expr) => {
        #[doc = $doc]
        pub const $name: NamedNodeRef<'static> =
            NamedNodeRef::new_unchecked(concat!("http://www.w3.org/2001/XMLSchema#", $local));
    };
}

term!(
    STRING,
    "string",
    "Character strings (but not all Unicode character strings)."
);
term!(BOOLEAN, "boolean", "true, false.");
term!(DECIMAL, "decimal", "Arbitrary-precision decimal numbers.");
term!(
    FLOAT,
    "float",
    "32-bit floating point numbers incl. +/-Inf, +/-0, NaN."
);
term!(
    DOUBLE,
    "double",
    "64-bit floating point numbers incl. +/-Inf, +/-0, NaN."
);
term!(DURATION, "duration", "Duration of time.");
term!(
    DATE_TIME,
    "dateTime",
    "Date and time with or without timezone."
);
term!(
    TIME,
    "time",
    "Times (hh:mm:ss.sss...) with or without timezone."
);
term!(DATE, "date", "Dates (yyyy-mm-dd) with or without timezone.");
term!(
    G_YEAR_MONTH,
    "gYearMonth",
    "Gregorian calendar year and month."
);
term!(G_YEAR, "gYear", "Gregorian calendar year.");
term!(
    G_MONTH_DAY,
    "gMonthDay",
    "Gregorian calendar month and day."
);
term!(G_DAY, "gDay", "Gregorian calendar day of the month.");
term!(G_MONTH, "gMonth", "Gregorian calendar month.");
term!(HEX_BINARY, "hexBinary", "Hex-encoded binary data.");
term!(BASE64_BINARY, "base64Binary", "Base64-encoded binary data.");
term!(ANY_URI, "anyURI", "Absolute or relative URIs and IRIs.");
term!(
    Q_NAME,
    "QName",
    "Qualified names, as used in XML namespace-aware documents."
);
term!(
    NOTATION,
    "NOTATION",
    "Encoding scheme names declared by an XML `NOTATION` declaration."
);
term!(INTEGER, "integer", "Arbitrary-size integer numbers.");
term!(
    NON_POSITIVE_INTEGER,
    "nonPositiveInteger",
    "Integer numbers <= 0."
);
term!(NEGATIVE_INTEGER, "negativeInteger", "Integer numbers < 0.");
term!(
    LONG,
    "long",
    "-9223372036854775808..+9223372036854775807 (64 bit)."
);
term!(INT, "int", "-2147483648..+2147483647 (32 bit).");
term!(SHORT, "short", "-32768..+32767 (16 bit).");
term!(BYTE, "byte", "-128..+127 (8 bit).");
term!(
    NON_NEGATIVE_INTEGER,
    "nonNegativeInteger",
    "Integer numbers >= 0."
);
term!(
    UNSIGNED_LONG,
    "unsignedLong",
    "0..18446744073709551615 (64 bit)."
);
term!(UNSIGNED_INT, "unsignedInt", "0..4294967295 (32 bit).");
term!(UNSIGNED_SHORT, "unsignedShort", "0..65535 (16 bit).");
term!(UNSIGNED_BYTE, "unsignedByte", "0..255 (8 bit).");
term!(POSITIVE_INTEGER, "positiveInteger", "Integer numbers > 0.");
term!(
    NORMALIZED_STRING,
    "normalizedString",
    "Whitespace-normalized strings."
);
term!(TOKEN, "token", "Tokenized strings.");
term!(
    LANGUAGE,
    "language",
    "Natural language identifiers, per [BCP 47](https://tools.ietf.org/html/bcp47)."
);
term!(
    NMTOKEN,
    "NMTOKEN",
    "XML `NMTOKEN`s, i.e. tokens matching the XML `Nmtoken` production."
);
term!(
    NMTOKENS,
    "NMTOKENS",
    "Whitespace-separated lists of one or more `NMTOKEN`s."
);
term!(NAME, "Name", "XML `Name`s.");
term!(NC_NAME, "NCName", "XML non-colonized names (`NCName`s).");
term!(
    ID,
    "ID",
    "XML `ID`s, unique among all IDs in an XML document."
);
term!(
    IDREF,
    "IDREF",
    "References to XML `ID`-typed attribute values."
);
term!(
    IDREFS,
    "IDREFS",
    "Whitespace-separated lists of one or more `IDREF`s."
);
term!(
    ENTITY,
    "ENTITY",
    "Names of unparsed XML entities declared in a DTD."
);
term!(
    ENTITIES,
    "ENTITIES",
    "Whitespace-separated lists of one or more `ENTITY` names."
);
term!(
    DATE_TIME_STAMP,
    "dateTimeStamp",
    "Date and time with a required timezone."
);
term!(
    DAY_TIME_DURATION,
    "dayTimeDuration",
    "Duration of time restricted to days, hours, minutes and seconds."
);
term!(
    YEAR_MONTH_DURATION,
    "yearMonthDuration",
    "Duration of time restricted to years and months."
);

/// All 47 terms defined in the [`xsd`](self) module, in declaration order.
///
/// Exposed for introspection and exhaustiveness testing.
pub const ALL: &[NamedNodeRef<'static>] = &[
    STRING,
    BOOLEAN,
    DECIMAL,
    FLOAT,
    DOUBLE,
    DURATION,
    DATE_TIME,
    TIME,
    DATE,
    G_YEAR_MONTH,
    G_YEAR,
    G_MONTH_DAY,
    G_DAY,
    G_MONTH,
    HEX_BINARY,
    BASE64_BINARY,
    ANY_URI,
    Q_NAME,
    NOTATION,
    INTEGER,
    NON_POSITIVE_INTEGER,
    NEGATIVE_INTEGER,
    LONG,
    INT,
    SHORT,
    BYTE,
    NON_NEGATIVE_INTEGER,
    UNSIGNED_LONG,
    UNSIGNED_INT,
    UNSIGNED_SHORT,
    UNSIGNED_BYTE,
    POSITIVE_INTEGER,
    NORMALIZED_STRING,
    TOKEN,
    LANGUAGE,
    NMTOKEN,
    NMTOKENS,
    NAME,
    NC_NAME,
    ID,
    IDREF,
    IDREFS,
    ENTITY,
    ENTITIES,
    DATE_TIME_STAMP,
    DAY_TIME_DURATION,
    YEAR_MONTH_DURATION,
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
        assert_eq!(ALL.len(), 47);
    }
}
