//! Atomic value typing: the type hierarchy, the lexical mapping, and the comparison relations.

use oxixml_infoset::{
    AtomicData, AtomicType, AtomicValue, CODEPOINT_COLLATION, ErrorCode,
    HTML_ASCII_CASE_INSENSITIVE_COLLATION, QNameValue, XSD_NAMESPACE, XdmError, atomic_equal,
    atomic_equal_in_context, atomic_equal_with_collation, compare_atomic,
    compare_atomic_in_context, compare_strings, same_key, same_key_hash,
};
use oxixml_xsd::TimezoneOffset;
use std::cmp::Ordering;

fn parse(ty: AtomicType, lexical: &str) -> Result<AtomicValue, XdmError> {
    AtomicValue::parse(ty, lexical)
}

#[test]
fn every_built_in_type_has_a_consistent_place_in_the_hierarchy() {
    for &ty in AtomicType::ALL {
        assert!(ty.is_derived_from(ty), "{ty} is derived from itself");
        assert!(
            ty.is_derived_from(AtomicType::AnyAtomicType),
            "{ty} is derived from xs:anyAtomicType"
        );
        assert_eq!(
            ty.expanded_name().namespace(),
            Some(XSD_NAMESPACE),
            "{ty} is in the XML Schema namespace"
        );
        assert_eq!(AtomicType::from_local_name(ty.local_name()), Some(ty));

        // A type is numeric, textual, temporal or none of those, never two at once.
        let classes =
            u8::from(ty.is_numeric()) + u8::from(ty.is_text()) + u8::from(ty.is_date_time());
        assert!(classes <= 1, "{ty} falls into {classes} classes");
    }
}

#[test]
fn lexical_mapping_produces_canonical_string_values() -> Result<(), XdmError> {
    for (ty, lexical, canonical) in [
        (AtomicType::Integer, "  +042\n", "42"),
        (AtomicType::Decimal, "+1.2500", "1.25"),
        (AtomicType::Boolean, "1", "true"),
        // `fn:string` of an xs:double follows XPath's casting rule, not the XSD canonical form:
        // a magnitude in `[1e-6, 1e6)` is written without an exponent.
        (AtomicType::Double, "1.0e0", "1"),
        (
            AtomicType::DateTime,
            "2020-01-01T00:00:00Z",
            "2020-01-01T00:00:00Z",
        ),
        (AtomicType::Date, "2020-01-01", "2020-01-01"),
        (AtomicType::YearMonthDuration, "P14M", "P1Y2M"),
        (AtomicType::HexBinary, "0fA0", "0FA0"),
        (AtomicType::Base64Binary, "Zm9v", "Zm9v"),
        (
            AtomicType::AnyUri,
            " http://example.com/a ",
            "http://example.com/a",
        ),
        (AtomicType::Token, " a  b ", "a b"),
        // XSD 1.1 §3.3.17 escapes before requiring an IRI, so a space survives the collapse
        // facet and the value is legal (however discouraged).
        (
            AtomicType::AnyUri,
            " http://example.com/ has a space ",
            "http://example.com/ has a space",
        ),
    ] {
        assert_eq!(
            parse(ty, lexical)?.string_value(),
            canonical,
            "canonical form of {ty} from `{lexical}`"
        );
    }
    Ok(())
}

#[test]
fn a_value_keeps_the_type_it_was_labelled_with() -> Result<(), XdmError> {
    let short = parse(AtomicType::Short, "7")?;
    assert_eq!(short.atomic_type(), AtomicType::Short);
    assert!(short.atomic_type().is_derived_from(AtomicType::Integer));
    assert!(matches!(short.data(), AtomicData::Integer(_)));

    // The label participates in equality; the payload alone does not decide it.
    let integer = AtomicValue::integer(7_i64);
    assert_ne!(short, integer);
    assert_eq!(short.data(), integer.data());
    assert!(atomic_equal(&short, &integer)?, "but `eq` compares values");
    Ok(())
}

#[test]
fn invalid_lexical_forms_are_forg0001() {
    for (ty, lexical) in [
        (AtomicType::Integer, "one"),
        (AtomicType::Integer, "1.5"),
        (AtomicType::UnsignedByte, "-1"),
        (AtomicType::Ncname, "has:colon"),
        (AtomicType::Language, "toolongprimary"),
        (AtomicType::DateTimeStamp, "2020-01-01T00:00:00"),
        (AtomicType::HexBinary, "abc"),
        (AtomicType::Base64Binary, "Zg="),
        (AtomicType::AnyAtomicType, "anything"),
    ] {
        let error = parse(ty, lexical).expect_err("{ty} should reject the lexical form");
        assert_eq!(error.code(), ErrorCode::Forg0001, "{ty} `{lexical}`");
    }
}

#[test]
fn language_and_collation_ride_along_without_changing_the_value() {
    let plain = AtomicValue::string("Straße");
    let annotated = plain
        .clone()
        .with_language("de-DE")
        .with_collation(CODEPOINT_COLLATION);

    assert_eq!(plain, annotated, "carried context is not part of the value");
    assert_eq!(annotated.language(), Some("de-DE"));
    assert_eq!(annotated.collation(), Some(CODEPOINT_COLLATION));
    assert_eq!(plain.language(), None);
    assert_eq!(plain.string_value(), annotated.string_value());
}

#[test]
fn numeric_comparison_promotes_across_the_tower() -> Result<(), XdmError> {
    let one_integer = AtomicValue::integer(1_i64);
    let one_decimal = AtomicValue::decimal(1_i64);
    let one_double = AtomicValue::double(1.0);
    let one_float = AtomicValue::float(1.0_f32);

    for other in [&one_decimal, &one_double, &one_float] {
        assert!(atomic_equal(&one_integer, other)?, "1 eq {other}");
        assert!(same_key(&one_integer, other));
        assert_eq!(same_key_hash(&one_integer), same_key_hash(other));
    }
    assert_eq!(
        compare_atomic(&one_integer, &AtomicValue::double(2.0))?,
        Some(Ordering::Less)
    );

    // A decimal too large for an f64 mantissa still compares exactly against an integer.
    let big_integer = parse(AtomicType::Integer, "9007199254740993")?;
    let big_decimal = parse(AtomicType::Decimal, "9007199254740993")?;
    assert!(atomic_equal(&big_integer, &big_decimal)?);
    assert_eq!(
        compare_atomic(
            &big_integer,
            &parse(AtomicType::Integer, "9007199254740992")?
        )?,
        Some(Ordering::Greater)
    );
    Ok(())
}

#[test]
fn nan_and_the_two_zeros() -> Result<(), XdmError> {
    let nan = AtomicValue::double(f64::NAN);
    assert!(nan.is_nan());
    assert!(!atomic_equal(&nan, &nan)?, "NaN is never `eq` to anything");
    assert_eq!(compare_atomic(&nan, &nan)?, None);
    assert!(same_key(&nan, &nan), "but NaN is a usable map key");

    let positive_zero = AtomicValue::double(0.0);
    let negative_zero = AtomicValue::double(-0.0);
    assert_ne!(positive_zero, negative_zero, "identity distinguishes them");
    assert!(
        atomic_equal(&positive_zero, &negative_zero)?,
        "`eq` does not"
    );
    assert!(same_key(&positive_zero, &negative_zero));
    assert_eq!(same_key_hash(&positive_zero), same_key_hash(&negative_zero));
    Ok(())
}

#[test]
fn incomparable_types_raise_xpty0004() {
    let cases = [
        (AtomicValue::integer(1_i64), AtomicValue::string("1")),
        (AtomicValue::boolean(true), AtomicValue::integer(1_i64)),
        (
            AtomicValue::hex_binary(vec![1]),
            AtomicValue::base64_binary(vec![1]),
        ),
    ];
    for (left, right) in cases {
        let error = atomic_equal(&left, &right).expect_err("not comparable");
        assert_eq!(error.code(), ErrorCode::Xpty0004);
        assert!(!same_key(&left, &right), "and they are different map keys");
    }
}

#[test]
fn collations_apply_to_the_string_family_only() -> Result<(), XdmError> {
    assert_eq!(compare_strings("a", "b", None)?, Ordering::Less);
    assert_eq!(
        compare_strings("A", "a", Some(HTML_ASCII_CASE_INSENSITIVE_COLLATION))?,
        Ordering::Equal
    );
    assert_eq!(
        compare_strings("a", "b", Some(CODEPOINT_COLLATION))?,
        Ordering::Less
    );
    assert_eq!(
        compare_strings("a", "b", Some("urn:example:uca"))
            .expect_err("unsupported")
            .code(),
        ErrorCode::Foch0002
    );

    // xs:anyURI and xs:string share a value space, so they compare against each other.
    assert!(atomic_equal(
        &AtomicValue::any_uri("urn:x"),
        &AtomicValue::string("urn:x")
    )?);
    assert!(atomic_equal_with_collation(
        &AtomicValue::string("ABC"),
        &AtomicValue::untyped_atomic("abc"),
        Some(HTML_ASCII_CASE_INSENSITIVE_COLLATION)
    )?);
    Ok(())
}

#[test]
fn qnames_are_compared_on_the_expanded_name_only() -> Result<(), XdmError> {
    let with_p = AtomicValue::qname(QNameValue::new(Some("p"), Some("urn:ns"), "x"));
    let with_q = AtomicValue::qname(QNameValue::new(Some("q"), Some("urn:ns"), "x"));
    let other_namespace = AtomicValue::qname(QNameValue::new(Some("p"), Some("urn:other"), "x"));

    assert!(atomic_equal(&with_p, &with_q)?);
    assert!(!atomic_equal(&with_p, &other_namespace)?);
    assert_eq!(
        with_p.string_value(),
        "p:x",
        "the prefix survives for output"
    );
    assert_eq!(with_q.string_value(), "q:x");

    let resolved = AtomicValue::parse_qname(AtomicType::QName, "p:x", |prefix| {
        (prefix == Some("p")).then(|| "urn:ns".to_owned())
    })?;
    assert!(atomic_equal(&resolved, &with_p)?);
    Ok(())
}

#[test]
fn date_time_values_respect_the_timezone_rules() -> Result<(), XdmError> {
    let utc = parse(AtomicType::DateTime, "2020-06-01T12:00:00Z")?;
    let shifted = parse(AtomicType::DateTime, "2020-06-01T14:00:00+02:00")?;
    let floating = parse(AtomicType::DateTime, "2020-06-01T12:00:00")?;

    assert!(atomic_equal(&utc, &shifted)?, "the same instant");
    assert!(same_key(&utc, &shifted));
    // F&O 3.1 reads a timezone-less operand in the dynamic context's implicit
    // timezone — UTC by default — before comparing, so this pair *is* equal.
    assert!(
        atomic_equal(&utc, &floating)?,
        "a value without a timezone is read in the implicit timezone"
    );
    // `op:same-key` may not depend on the dynamic context, so it uses XML Schema equality
    // instead: a timezoned value is never the same map key as a floating one.
    assert!(!same_key(&utc, &floating));

    // Applying the implicit timezone makes the F&O relation total: unlike XML
    // Schema's own order relation, no pair is left unordered.
    let far = parse(AtomicType::DateTime, "2020-06-01T20:00:00")?;
    assert_eq!(compare_atomic(&utc, &far)?, Some(Ordering::Less));

    // A non-UTC implicit timezone shifts the timezone-less operand only.
    let jst = TimezoneOffset::new(9 * 60).expect("a valid offset");
    assert!(!atomic_equal_in_context(&utc, &floating, None, Some(jst))?);
    assert_eq!(
        compare_atomic_in_context(&floating, &utc, None, Some(jst))?,
        Some(Ordering::Less)
    );
    Ok(())
}

#[test]
fn durations_are_partially_ordered() -> Result<(), XdmError> {
    let month = parse(AtomicType::Duration, "P1M")?;
    let days = parse(AtomicType::Duration, "P30D")?;
    assert_eq!(compare_atomic(&month, &days)?, None);
    assert!(!atomic_equal(&month, &days)?);

    let year_month = parse(AtomicType::YearMonthDuration, "P1Y")?;
    let twelve_months = parse(AtomicType::YearMonthDuration, "P12M")?;
    assert!(atomic_equal(&year_month, &twelve_months)?);
    assert_eq!(
        compare_atomic(&year_month, &parse(AtomicType::YearMonthDuration, "P13M")?)?,
        Some(Ordering::Less)
    );
    Ok(())
}

#[test]
fn binary_values_round_trip_through_both_lexical_mappings() -> Result<(), XdmError> {
    let payload: Vec<u8> = (0_u8..=255).collect();
    let hex = AtomicValue::hex_binary(payload.clone());
    let base64 = AtomicValue::base64_binary(payload.clone());

    assert_eq!(
        parse(AtomicType::HexBinary, &hex.string_value())?.as_bytes(),
        Some(payload.as_slice())
    );
    assert_eq!(
        parse(AtomicType::Base64Binary, &base64.string_value())?.as_bytes(),
        Some(payload.as_slice())
    );
    assert_ne!(hex.string_value(), base64.string_value());
    Ok(())
}
