//! The built-in datatype hierarchy of XML Schema Part 2, exercised through the
//! public simple-type API.

use oxixml_schema::simple::{SimpleTypeViolation, validate};
use oxixml_schema::value::{NamespaceContext, Value};
use oxixml_schema::{Schema, SchemaSet, XsdVersion};

fn builtins(version: XsdVersion) -> Schema {
    SchemaSet::new()
        .with_version(version)
        .compile()
        .expect("the built-in hierarchy")
}

fn accepts(schema: &Schema, datatype: &str, lexical: &str) -> bool {
    let Some(id) = schema.builtin(datatype) else {
        panic!("no built-in datatype `{datatype}`");
    };
    validate(schema, id, lexical, &NamespaceContext::new()).is_ok()
}

fn value(schema: &Schema, datatype: &str, lexical: &str) -> Option<Value> {
    let id = schema.builtin(datatype)?;
    validate(schema, id, lexical, &NamespaceContext::new())
        .ok()
        .map(|result| result.value)
}

#[test]
fn the_string_family() {
    let schema = builtins(XsdVersion::V1_0);
    assert!(accepts(&schema, "string", " a\tb "));
    assert!(accepts(&schema, "normalizedString", "a b"));
    assert!(accepts(&schema, "token", "a b"));
    assert!(accepts(&schema, "Name", "a:b"));
    assert!(accepts(&schema, "Name", "_x"));
    assert!(!accepts(&schema, "Name", "1x"));
    assert!(accepts(&schema, "NCName", "x"));
    assert!(!accepts(&schema, "NCName", "a:b"));
    assert!(accepts(&schema, "NMTOKEN", "1x"));
    assert!(!accepts(&schema, "NMTOKEN", "a b"));
    assert!(accepts(&schema, "language", "en-GB"));
    assert!(!accepts(&schema, "language", "abcdefghi"));
    assert!(!accepts(&schema, "language", "en-"));
    assert!(!accepts(&schema, "language", "1en"));
    // `whiteSpace` is applied before the pattern: a trailing newline collapses.
    assert!(accepts(&schema, "NCName", "x\n"));
}

#[test]
fn the_numeric_tower() {
    let schema = builtins(XsdVersion::V1_0);
    assert!(accepts(&schema, "decimal", "-1.500"));
    assert!(!accepts(&schema, "decimal", "1e3"));
    assert!(accepts(&schema, "integer", "+0012"));
    assert!(!accepts(&schema, "integer", "12.0"));
    for (datatype, inside, outside) in [
        ("byte", "-128", "-129"),
        ("short", "32767", "32768"),
        ("int", "-2147483648", "-2147483649"),
        ("long", "9223372036854775807", "9223372036854775808"),
        ("unsignedByte", "255", "256"),
        ("unsignedShort", "65535", "65536"),
        ("unsignedInt", "4294967295", "4294967296"),
        (
            "unsignedLong",
            "18446744073709551615",
            "18446744073709551616",
        ),
        ("positiveInteger", "1", "0"),
        ("negativeInteger", "-1", "0"),
        ("nonNegativeInteger", "0", "-1"),
        ("nonPositiveInteger", "0", "1"),
    ] {
        assert!(accepts(&schema, datatype, inside), "{datatype} {inside}");
        assert!(!accepts(&schema, datatype, outside), "{datatype} {outside}");
    }
    // `1.0` and `1.00` denote the same value.
    assert_eq!(
        value(&schema, "decimal", "1.0"),
        value(&schema, "decimal", "1.00")
    );
}

#[test]
fn the_binary_floats() {
    let schema = builtins(XsdVersion::V1_0);
    for lexical in ["0", "-0", "1.0E-3", "INF", "-INF", "NaN", "+1", ".5"] {
        assert!(accepts(&schema, "double", lexical), "double {lexical}");
        assert!(accepts(&schema, "float", lexical), "float {lexical}");
    }
    for lexical in ["infinity", "1e", "nan", "1.0d"] {
        assert!(!accepts(&schema, "double", lexical), "double {lexical}");
    }
    // `float` narrows to 32 bits, so the two are different values.
    assert_ne!(
        value(&schema, "float", "0.1"),
        value(&schema, "double", "0.1")
    );
}

#[test]
fn the_temporal_tower() {
    let schema = builtins(XsdVersion::V1_0);
    for (datatype, lexical) in [
        ("dateTime", "2001-10-26T21:32:52"),
        ("dateTime", "2001-10-26T21:32:52+02:00"),
        ("dateTime", "-0002-10-26T21:32:52Z"),
        ("date", "2001-10-26"),
        ("time", "21:32:52.12679"),
        ("gYearMonth", "2001-10"),
        ("gYear", "2001"),
        ("gMonthDay", "--10-26"),
        ("gMonth", "--10"),
        ("gDay", "---26"),
        ("duration", "P1Y2M3DT10H30M"),
        ("duration", "-P120D"),
    ] {
        assert!(accepts(&schema, datatype, lexical), "{datatype} {lexical}");
    }
    for (datatype, lexical) in [
        ("dateTime", "2001-10-26"),
        ("date", "2001-13-26"),
        ("time", "25:00:00"),
        ("gMonth", "10"),
        ("duration", "P"),
    ] {
        assert!(!accepts(&schema, datatype, lexical), "{datatype} {lexical}");
    }
    // Timezones are normalized: these are the same instant.
    assert_eq!(
        value(&schema, "dateTime", "2000-01-01T00:00:00Z"),
        value(&schema, "dateTime", "1999-12-31T19:00:00-05:00")
    );
}

#[test]
fn the_binary_and_uri_datatypes() {
    let schema = builtins(XsdVersion::V1_0);
    assert!(accepts(&schema, "hexBinary", "0FB7"));
    assert!(!accepts(&schema, "hexBinary", "0FB"));
    assert!(!accepts(&schema, "hexBinary", "0FGZ"));
    assert!(accepts(
        &schema,
        "base64Binary",
        "YW55IGNhcm5hbCBwbGVhc3VyZQ=="
    ));
    assert!(accepts(&schema, "base64Binary", ""));
    assert!(!accepts(&schema, "base64Binary", "YW5="));
    assert!(accepts(&schema, "anyURI", "http://example.com/a?b#c"));
    assert!(accepts(&schema, "anyURI", "relative/path"));
    assert!(!accepts(&schema, "anyURI", "http://example.com/%zz"));
    assert_eq!(
        value(&schema, "hexBinary", "0FB7"),
        Some(Value::Binary(vec![0x0F, 0xB7]))
    );
}

#[test]
fn boolean_has_four_lexical_forms() {
    let schema = builtins(XsdVersion::V1_0);
    assert_eq!(value(&schema, "boolean", "1"), Some(Value::Boolean(true)));
    assert_eq!(
        value(&schema, "boolean", "true"),
        Some(Value::Boolean(true))
    );
    assert_eq!(value(&schema, "boolean", "0"), Some(Value::Boolean(false)));
    assert_eq!(
        value(&schema, "boolean", "false"),
        Some(Value::Boolean(false))
    );
    assert!(!accepts(&schema, "boolean", "TRUE"));
}

#[test]
fn the_list_datatypes_need_at_least_one_item() {
    let schema = builtins(XsdVersion::V1_0);
    assert!(accepts(&schema, "NMTOKENS", "a b 1"));
    assert!(accepts(&schema, "IDREFS", "a b"));
    assert!(!accepts(&schema, "IDREFS", "   "));
    assert!(!accepts(&schema, "ENTITIES", "1a"));
    assert_eq!(
        value(&schema, "NMTOKENS", " a  b "),
        value(&schema, "NMTOKENS", "a b")
    );
}

#[test]
fn qnames_need_their_context() -> Result<(), SimpleTypeViolation> {
    let schema = builtins(XsdVersion::V1_0);
    let qname = schema.builtin("QName").expect("QName");
    let mut namespaces = NamespaceContext::new();
    assert!(validate(&schema, qname, "p:a", &namespaces).is_err());
    namespaces.declare(Some("p"), "urn:x");
    namespaces.declare(Some("q"), "urn:x");
    let left = validate(&schema, qname, "p:a", &namespaces)?.value;
    let right = validate(&schema, qname, "q:a", &namespaces)?.value;
    assert_eq!(left, right, "the prefix is not part of the value");
    Ok(())
}

#[test]
fn version_1_1_adds_its_own_datatypes() {
    let schema = builtins(XsdVersion::V1_1);
    assert!(accepts(&schema, "dateTimeStamp", "2001-10-26T21:32:52Z"));
    assert!(!accepts(&schema, "dateTimeStamp", "2001-10-26T21:32:52"));
    assert!(accepts(&schema, "yearMonthDuration", "P1Y2M"));
    assert!(!accepts(&schema, "yearMonthDuration", "P1D"));
    assert!(accepts(&schema, "dayTimeDuration", "P1DT2H"));
    assert!(!accepts(&schema, "dayTimeDuration", "P1Y"));
}

#[test]
fn the_derivation_hierarchy_is_navigable() {
    let schema = builtins(XsdVersion::V1_0);
    let byte = schema.builtin("byte").expect("byte");
    let integer = schema.builtin("integer").expect("integer");
    let decimal = schema.builtin("decimal").expect("decimal");
    let any_simple = schema.any_simple_type();
    assert!(schema.is_derived_from(byte, integer));
    assert!(schema.is_derived_from(byte, decimal));
    assert!(schema.is_derived_from(byte, any_simple));
    assert!(!schema.is_derived_from(decimal, byte));
    let idrefs = schema.builtin("IDREFS").expect("IDREFS");
    assert_eq!(schema.list_item_type(idrefs), schema.builtin("IDREF"));
}
