//! The lexical mapping: [`AtomicValue::parse`] and its helpers.
//!
//! Each XML Schema datatype defines a *lexical mapping* from strings to its value space, plus a
//! `whiteSpace` facet that says how the input is pre-processed. Both are applied here, followed
//! by the facets that constrain a derived type's value space (the numeric bounds of
//! `xs:unsignedShort`, the `NCName` production behind `xs:ID`, and so on).

use super::bignum::{BigInteger, parse_big_decimal, parse_big_integer};
use super::binary::{decode_base64, decode_hex};
use super::qname_value::QNameValue;
use super::types::AtomicType;
use super::value::{AtomicData, AtomicValue};
use crate::error::{ErrorCode, XdmError};
use oxixml_iri::IriRef;
use oxixml_qname::{is_name, is_ncname, is_nmtoken};
use oxixml_xsd::{
    Boolean, Date, DateTime, DayTimeDuration, Decimal, Double, Duration, Float, GDay, GMonth,
    GMonthDay, GYear, GYearMonth, Integer, Time, YearMonthDuration,
};
use std::str::FromStr;

/// How a datatype's `whiteSpace` facet pre-processes a lexical form.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum WhiteSpace {
    Preserve,
    Replace,
    Collapse,
}

impl AtomicType {
    /// The value of this type's `whiteSpace` facet, which decides how a lexical form is
    /// pre-processed before the lexical mapping is applied.
    fn white_space(self) -> WhiteSpace {
        match self {
            Self::String | Self::UntypedAtomic | Self::AnyAtomicType => WhiteSpace::Preserve,
            Self::NormalizedString => WhiteSpace::Replace,
            _ => WhiteSpace::Collapse,
        }
    }
}

/// Applies the `replace` white-space facet: every tab, carriage return and line feed becomes a
/// single space.
fn replace_white_space(input: &str) -> String {
    input
        .chars()
        .map(|c| match c {
            '\t' | '\r' | '\n' => ' ',
            other => other,
        })
        .collect()
}

/// Applies the `collapse` white-space facet: `replace`, then trim, then squeeze runs of spaces.
fn collapse_white_space(input: &str) -> String {
    let mut out = String::with_capacity(input.len());
    let mut pending_space = false;
    for character in input.chars() {
        if matches!(character, ' ' | '\t' | '\r' | '\n') {
            pending_space = !out.is_empty();
        } else {
            if pending_space {
                out.push(' ');
                pending_space = false;
            }
            out.push(character);
        }
    }
    out
}

fn forg0001(atomic_type: AtomicType, lexical: &str, reason: impl AsRef<str>) -> XdmError {
    XdmError::new(
        ErrorCode::Forg0001,
        format!(
            "`{lexical}` is not a valid {atomic_type}: {}",
            reason.as_ref()
        ),
    )
}

impl AtomicValue {
    /// Applies `atomic_type`'s white-space facet and lexical mapping to `lexical`, then checks
    /// the facets that constrain the resulting value.
    ///
    /// This is the operation behind XPath's constructor functions (`xs:integer("42")`) and
    /// behind schema validation of a simple type.
    ///
    /// A few types need context that a bare string cannot supply:
    ///
    /// * `xs:QName` and `xs:NOTATION` accept only an unprefixed `NCName` here, which maps to a
    ///   name in no namespace. Use [`AtomicValue::parse_qname`] to resolve a prefix.
    /// * `xs:anyURI` is validated against the RFC 3987 `IRI-reference` grammar by
    ///   [`oxixml_iri::IriRef`].
    /// * `xs:anyAtomicType` is abstract and is always rejected.
    ///
    /// # Errors
    ///
    /// [`ErrorCode::Forg0001`] if `lexical` is not in the type's lexical space, or if the value
    /// it maps to violates one of the type's facets.
    ///
    /// A lexical form that is *well formed* but names a value outside the range this
    /// implementation represents is reported separately, as F&O 3.1 §10.1.1 requires:
    /// [`ErrorCode::Fodt0001`] for the date, time and Gregorian types, and
    /// [`ErrorCode::Fodt0002`] for the three duration types.
    ///
    /// # Examples
    ///
    /// ```
    /// use oxixml_infoset::{AtomicType, AtomicValue};
    ///
    /// // The white-space facet is applied first: `collapse` for everything but the string
    /// // hierarchy's top two members.
    /// assert_eq!(AtomicValue::parse(AtomicType::Integer, "  42\n")?.string_value(), "42");
    /// assert_eq!(AtomicValue::parse(AtomicType::String, " a b ")?.string_value(), " a b ");
    /// assert_eq!(AtomicValue::parse(AtomicType::Token, " a  b ")?.string_value(), "a b");
    ///
    /// // Derived types re-check their facets.
    /// assert!(AtomicValue::parse(AtomicType::UnsignedByte, "255").is_ok());
    /// assert!(AtomicValue::parse(AtomicType::UnsignedByte, "256").is_err());
    /// assert!(AtomicValue::parse(AtomicType::Ncname, "a:b").is_err());
    ///
    /// // The canonical form is what comes back out.
    /// assert_eq!(AtomicValue::parse(AtomicType::Decimal, "+1.2500")?.string_value(), "1.25");
    /// # Ok::<(), oxixml_infoset::XdmError>(())
    /// ```
    pub fn parse(atomic_type: AtomicType, lexical: &str) -> Result<Self, XdmError> {
        let prepared = match atomic_type.white_space() {
            WhiteSpace::Preserve => lexical.to_owned(),
            WhiteSpace::Replace => replace_white_space(lexical),
            WhiteSpace::Collapse => collapse_white_space(lexical),
        };
        let data = map_lexical(atomic_type, &prepared)?;
        check_facets(atomic_type, &data, &prepared)?;
        Ok(Self::new_unchecked(atomic_type, data))
    }

    /// Parses an `xs:QName` (or `xs:NOTATION`) lexical form, resolving its prefix with
    /// `resolve`.
    ///
    /// `resolve` is called with `Some(prefix)` for a prefixed name and with `None` for an
    /// unprefixed one; returning `None` means "not bound". An unprefixed name whose `resolve`
    /// answers `None` is a name in no namespace, which is always legal; a *prefixed* name whose
    /// prefix is unbound is an error.
    ///
    /// # Errors
    ///
    /// [`ErrorCode::Forg0001`] if the lexical form is not a `QName`, if a prefix is unbound, or
    /// if `atomic_type` is not `xs:QName`/`xs:NOTATION` or a restriction of one.
    ///
    /// # Examples
    ///
    /// ```
    /// use oxixml_infoset::{AtomicType, AtomicValue};
    ///
    /// let value = AtomicValue::parse_qname(AtomicType::QName, " bk:title ", |prefix| {
    ///     (prefix == Some("bk")).then(|| "urn:example:book".to_owned())
    /// })?;
    /// let qname = value.as_qname().expect("an xs:QName");
    /// assert_eq!(qname.prefix(), Some("bk"));
    /// assert_eq!(qname.namespace(), Some("urn:example:book"));
    /// assert_eq!(qname.local_name(), "title");
    /// assert_eq!(value.string_value(), "bk:title");
    ///
    /// // An unbound prefix is an error; an unprefixed name simply has no namespace.
    /// assert!(AtomicValue::parse_qname(AtomicType::QName, "nope:x", |_| None).is_err());
    /// assert!(AtomicValue::parse_qname(AtomicType::QName, "x", |_| None).is_ok());
    /// # Ok::<(), oxixml_infoset::XdmError>(())
    /// ```
    pub fn parse_qname(
        atomic_type: AtomicType,
        lexical: &str,
        resolve: impl FnOnce(Option<&str>) -> Option<String>,
    ) -> Result<Self, XdmError> {
        if !atomic_type.is_derived_from(AtomicType::QName)
            && !atomic_type.is_derived_from(AtomicType::Notation)
        {
            return Err(forg0001(
                atomic_type,
                lexical,
                "only xs:QName and xs:NOTATION can be built from a lexical QName",
            ));
        }
        let collapsed = collapse_white_space(lexical);
        let (prefix, local) = match collapsed.split_once(':') {
            Some((prefix, local)) => (Some(prefix), local),
            None => (None, collapsed.as_str()),
        };
        if !is_ncname(local) || prefix.is_some_and(|prefix| !is_ncname(prefix)) {
            return Err(forg0001(
                atomic_type,
                lexical,
                "does not match the QName production",
            ));
        }
        let namespace = resolve(prefix);
        if prefix.is_some() && namespace.is_none() {
            return Err(forg0001(
                atomic_type,
                lexical,
                "the prefix is not bound to a namespace",
            ));
        }
        Ok(Self::new_unchecked(
            atomic_type,
            AtomicData::QName(QNameValue::new(prefix, namespace, local)),
        ))
    }
}

/// Applies the lexical mapping of `atomic_type`'s *primitive* base to an already white-space
/// processed string.
fn map_lexical(atomic_type: AtomicType, lexical: &str) -> Result<AtomicData, XdmError> {
    let primitive = atomic_type.primitive();
    // The lexical spaces of the date, time and duration types are narrower than what the value
    // parsers in `oxixml-xsd` accept; see [`super::lexical`] for the rules and their sources.
    if let Err(reason) = super::lexical::extra_lexical_restriction(atomic_type, lexical) {
        return Err(forg0001(atomic_type, lexical, reason));
    }
    let mapped = match primitive {
        AtomicType::AnyAtomicType => {
            return Err(forg0001(
                atomic_type,
                lexical,
                "xs:anyAtomicType is abstract and has no lexical mapping",
            ));
        }
        AtomicType::String | AtomicType::UntypedAtomic | AtomicType::AnyUri => {
            AtomicData::Text(lexical.to_owned())
        }
        AtomicType::Boolean => AtomicData::Boolean(parse_with::<Boolean>(atomic_type, lexical)?),
        AtomicType::Decimal => {
            if atomic_type.is_derived_from(AtomicType::Integer) {
                // The fixed-width `i64` path is tried first, so an in-range value is byte-for-byte
                // what it always was. A value that overflows it but is still a valid `xs:integer`
                // is carried at arbitrary precision (demoting straight back if it happens to fit);
                // the type's own range facet is enforced afterwards, in `check_facets`.
                match Integer::from_str(lexical) {
                    Ok(value) => AtomicData::Integer(value),
                    Err(fixed_error) => match parse_big_integer(lexical) {
                        Some(big) => AtomicData::from_big_integer(big),
                        None => {
                            return Err(forg0001(atomic_type, lexical, fixed_error.to_string()));
                        }
                    },
                }
            } else {
                match Decimal::from_str(lexical) {
                    Ok(value) => AtomicData::Decimal(value),
                    Err(fixed_error) => match parse_big_decimal(lexical) {
                        Some(big) => AtomicData::from_big_decimal(big),
                        None => {
                            return Err(forg0001(atomic_type, lexical, fixed_error.to_string()));
                        }
                    },
                }
            }
        }
        AtomicType::Float => {
            require_xsd_float_lexical(atomic_type, lexical)?;
            AtomicData::Float(parse_with::<Float>(atomic_type, lexical)?)
        }
        AtomicType::Double => {
            require_xsd_float_lexical(atomic_type, lexical)?;
            AtomicData::Double(parse_with::<Double>(atomic_type, lexical)?)
        }
        AtomicType::Duration => {
            match atomic_type {
                AtomicType::YearMonthDuration => AtomicData::YearMonthDuration(
                    parse_duration_with::<YearMonthDuration>(atomic_type, lexical)?,
                ),
                AtomicType::DayTimeDuration => AtomicData::DayTimeDuration(parse_duration_with::<
                    DayTimeDuration,
                >(
                    atomic_type, lexical
                )?),
                _ => AtomicData::Duration(parse_duration_with::<Duration>(atomic_type, lexical)?),
            }
        }
        AtomicType::DateTime => {
            AtomicData::DateTime(parse_date_time_with::<DateTime>(atomic_type, lexical)?)
        }
        AtomicType::Date => AtomicData::Date(parse_date_time_with::<Date>(atomic_type, lexical)?),
        AtomicType::Time => AtomicData::Time(parse_date_time_with::<Time>(atomic_type, lexical)?),
        AtomicType::GYearMonth => {
            AtomicData::GYearMonth(parse_date_time_with::<GYearMonth>(atomic_type, lexical)?)
        }
        AtomicType::GYear => {
            AtomicData::GYear(parse_date_time_with::<GYear>(atomic_type, lexical)?)
        }
        AtomicType::GMonthDay => {
            AtomicData::GMonthDay(parse_date_time_with::<GMonthDay>(atomic_type, lexical)?)
        }
        AtomicType::GMonth => {
            AtomicData::GMonth(parse_date_time_with::<GMonth>(atomic_type, lexical)?)
        }
        AtomicType::GDay => AtomicData::GDay(parse_date_time_with::<GDay>(atomic_type, lexical)?),
        AtomicType::HexBinary => AtomicData::Binary(decode_hex(lexical)?),
        AtomicType::Base64Binary => AtomicData::Binary(decode_base64(lexical)?),
        AtomicType::QName | AtomicType::Notation => {
            if !is_ncname(lexical) {
                return Err(forg0001(
                    atomic_type,
                    lexical,
                    "only an unprefixed NCName can be parsed without a namespace context; \
                     use AtomicValue::parse_qname to resolve a prefix",
                ));
            }
            AtomicData::QName(QNameValue::new(None::<&str>, None::<&str>, lexical))
        }
        other => {
            return Err(forg0001(
                atomic_type,
                lexical,
                format!("{other} is not a primitive type with a lexical mapping"),
            ));
        }
    };
    Ok(mapped)
}

/// Checks the `xs:double`/`xs:float` lexical space of
/// [XSD 1.1 §3.3.5](https://www.w3.org/TR/xmlschema11-2/#double) before the value is read.
///
/// Rust's `f64::from_str` accepts far more than XSD does — `Inf`, `+Inf`, `infinity`, `nan` —
/// and every one of those is `FORG0001` to a conforming processor, so the grammar is enforced
/// here rather than left to the host language's parser.
fn require_xsd_float_lexical(atomic_type: AtomicType, lexical: &str) -> Result<(), XdmError> {
    let invalid = || {
        forg0001(
            atomic_type,
            lexical,
            "does not match the xs:double lexical grammar",
        )
    };
    if lexical == "NaN" {
        return Ok(());
    }
    let body = lexical.strip_prefix(['+', '-']).unwrap_or(lexical);
    if body == "INF" {
        return Ok(());
    }
    let (mantissa, exponent) = match body.split_once(['e', 'E']) {
        Some((mantissa, exponent)) => (mantissa, Some(exponent)),
        None => (body, None),
    };
    let (integer, fraction) = match mantissa.split_once('.') {
        Some((integer, fraction)) => (integer, fraction),
        None => (mantissa, ""),
    };
    let digits = |text: &str| text.chars().all(|c| c.is_ascii_digit());
    if !digits(integer) || !digits(fraction) || (integer.is_empty() && fraction.is_empty()) {
        return Err(invalid());
    }
    if let Some(exponent) = exponent {
        let exponent = exponent.strip_prefix(['+', '-']).unwrap_or(exponent);
        if exponent.is_empty() || !digits(exponent) {
            return Err(invalid());
        }
    }
    Ok(())
}

fn parse_with<T: FromStr>(atomic_type: AtomicType, lexical: &str) -> Result<T, XdmError>
where
    T::Err: std::fmt::Display,
{
    T::from_str(lexical).map_err(|error| forg0001(atomic_type, lexical, error.to_string()))
}

/// Parses a date, time or Gregorian value, distinguishing a lexical error (`FORG0001`) from a
/// value that is well-formed but outside the timeline this implementation can represent
/// (`FODT0001`).
///
/// F&O 3.1 §10.1.1 draws exactly this line: `xs:dateTime("not-a-date")` is `FORG0001`, while
/// `xs:dateTime("25252734927766555-07-28T23:59:59-14:00")` — a syntactically perfect lexical
/// form whose year does not fit — is the overflow condition `FODT0001`.
fn parse_date_time_with<T>(atomic_type: AtomicType, lexical: &str) -> Result<T, XdmError>
where
    T: FromStr<Err = oxixml_xsd::ParseDateTimeError>,
{
    T::from_str(lexical).map_err(|error| {
        if error.is_overflow() {
            XdmError::new(
                ErrorCode::Fodt0001,
                format!("`{lexical}` is outside the range of {atomic_type}: {error}"),
            )
        } else {
            forg0001(atomic_type, lexical, error.to_string())
        }
    })
}

/// The duration counterpart of [`parse_date_time_with`]: an out-of-range duration is
/// `FODT0002`, a malformed one `FORG0001`.
fn parse_duration_with<T>(atomic_type: AtomicType, lexical: &str) -> Result<T, XdmError>
where
    T: FromStr<Err = oxixml_xsd::ParseDurationError>,
{
    T::from_str(lexical).map_err(|error| {
        if error.is_overflow() {
            XdmError::new(
                ErrorCode::Fodt0002,
                format!("`{lexical}` is outside the range of {atomic_type}: {error}"),
            )
        } else {
            forg0001(atomic_type, lexical, error.to_string())
        }
    })
}

/// Percent-escapes the characters that an `xs:anyURI` lexical form is allowed to contain but an
/// IRI reference is not.
///
/// XSD 1.1 §3.3.17 maps a lexical form to a value by escaping the characters `RFC 3987` excludes
/// — space and the "unwise"/delimiter set — and only then requiring an IRI reference, which is
/// why the specification notes that "spaces are, in principle, allowed in the ·lexical space· of
/// `anyURI`, however their use is highly discouraged". F&O 3.1 §19.1.1 adds that "for
/// `xs:anyURI`, the extent to which an implementation validates the lexical form of `xs:anyURI`
/// is implementation dependent", so the escaping is applied before validating rather than
/// rejecting outright: `xs:anyURI("a string")` is a legal value whose escaped form is
/// `a%20string`.
///
/// A `%` is deliberately *not* escaped, so a malformed percent-triplet such as `%gg` stays
/// invalid.
fn escape_for_any_uri(lexical: &str) -> std::borrow::Cow<'_, str> {
    fn needs_escape(character: char) -> bool {
        matches!(
            character,
            ' ' | '<' | '>' | '"' | '{' | '}' | '|' | '\\' | '^' | '`'
        ) || character.is_control()
    }
    if !lexical.contains(needs_escape) {
        return std::borrow::Cow::Borrowed(lexical);
    }
    let mut escaped = String::with_capacity(lexical.len());
    let mut buffer = [0_u8; 4];
    for character in lexical.chars() {
        if needs_escape(character) {
            for byte in character.encode_utf8(&mut buffer).as_bytes() {
                escaped.push_str(&format!("%{byte:02X}"));
            }
        } else {
            escaped.push(character);
        }
    }
    std::borrow::Cow::Owned(escaped)
}

/// Checks the facets that distinguish a derived type from its primitive base.
fn check_facets(atomic_type: AtomicType, data: &AtomicData, lexical: &str) -> Result<(), XdmError> {
    match atomic_type {
        AtomicType::AnyUri => {
            if IriRef::parse(&*escape_for_any_uri(lexical)).is_err() {
                return Err(forg0001(
                    atomic_type,
                    lexical,
                    "does not match the RFC 3987 IRI-reference grammar, even after escaping",
                ));
            }
        }
        AtomicType::Language => {
            if !is_language(lexical) {
                return Err(forg0001(
                    atomic_type,
                    lexical,
                    "does not match [a-zA-Z]{1,8}(-[a-zA-Z0-9]{1,8})*",
                ));
            }
        }
        AtomicType::Nmtoken => {
            if !is_nmtoken(lexical) {
                return Err(forg0001(atomic_type, lexical, "is not an Nmtoken"));
            }
        }
        AtomicType::Name => {
            if !is_name(lexical) {
                return Err(forg0001(atomic_type, lexical, "is not a Name"));
            }
        }
        AtomicType::Ncname | AtomicType::Id | AtomicType::Idref | AtomicType::Entity => {
            if !is_ncname(lexical) {
                return Err(forg0001(atomic_type, lexical, "is not an NCName"));
            }
        }
        AtomicType::DateTimeStamp => {
            let AtomicData::DateTime(value) = data else {
                return Err(forg0001(atomic_type, lexical, "is not an xs:dateTime"));
            };
            if value.timezone_offset().is_none() {
                return Err(forg0001(
                    atomic_type,
                    lexical,
                    "xs:dateTimeStamp requires an explicit timezone",
                ));
            }
        }
        _ => match data {
            AtomicData::Integer(value) => {
                check_integer_bounds(atomic_type, i64::from(*value), lexical)?;
            }
            AtomicData::BigInteger(value) => {
                check_big_integer_bounds(atomic_type, value, lexical)?;
            }
            _ => {}
        },
    }
    Ok(())
}

/// The `minInclusive`/`maxInclusive` facets of the built-in `xs:integer` restrictions, checked
/// against an arbitrary-precision value.
///
/// This is the path a value that overflowed the fixed width takes. Unlike the fixed-width
/// [`check_integer_bounds`], it can express `xs:unsignedLong`'s true ceiling of
/// `18446744073709551615`, so `xs:unsignedLong` accepts its whole range and rejects anything
/// past it. A bounded type whose range fits an `i64` rejects every big value, since a big value
/// is by construction outside that range.
fn check_big_integer_bounds(
    atomic_type: AtomicType,
    value: &BigInteger,
    lexical: &str,
) -> Result<(), XdmError> {
    let (min, max): (Option<BigInteger>, Option<BigInteger>) = match atomic_type {
        AtomicType::NonPositiveInteger => (None, Some(BigInteger::from_i64(0))),
        AtomicType::NegativeInteger => (None, Some(BigInteger::from_i64(-1))),
        AtomicType::Long => (
            Some(BigInteger::from_i64(i64::MIN)),
            Some(BigInteger::from_i64(i64::MAX)),
        ),
        AtomicType::Int => (
            Some(BigInteger::from_i64(i64::from(i32::MIN))),
            Some(BigInteger::from_i64(i64::from(i32::MAX))),
        ),
        AtomicType::Short => (
            Some(BigInteger::from_i64(i64::from(i16::MIN))),
            Some(BigInteger::from_i64(i64::from(i16::MAX))),
        ),
        AtomicType::Byte => (
            Some(BigInteger::from_i64(i64::from(i8::MIN))),
            Some(BigInteger::from_i64(i64::from(i8::MAX))),
        ),
        AtomicType::NonNegativeInteger => (Some(BigInteger::from_i64(0)), None),
        AtomicType::UnsignedLong => (
            Some(BigInteger::from_i64(0)),
            Some(BigInteger::from_i128(i128::from(u64::MAX))),
        ),
        AtomicType::UnsignedInt => (
            Some(BigInteger::from_i64(0)),
            Some(BigInteger::from_i64(i64::from(u32::MAX))),
        ),
        AtomicType::UnsignedShort => (
            Some(BigInteger::from_i64(0)),
            Some(BigInteger::from_i64(i64::from(u16::MAX))),
        ),
        AtomicType::UnsignedByte => (
            Some(BigInteger::from_i64(0)),
            Some(BigInteger::from_i64(i64::from(u8::MAX))),
        ),
        AtomicType::PositiveInteger => (Some(BigInteger::from_i64(1)), None),
        // `xs:integer` itself and anything with no numeric facet is unbounded.
        _ => (None, None),
    };
    if min.is_some_and(|min| *value < min) || max.is_some_and(|max| *value > max) {
        return Err(forg0001(
            atomic_type,
            lexical,
            "the value is outside the type's range",
        ));
    }
    Ok(())
}

/// The `minInclusive`/`maxInclusive` facets of the built-in `xs:integer` restrictions.
///
/// `xs:unsignedLong` nominally reaches `18446744073709551615`; a value that large is carried as a
/// [`AtomicData::BigInteger`] and range-checked by [`check_big_integer_bounds`], so on this
/// fixed-width path — which only ever sees values that fit an [`i64`] — the type has no reachable
/// upper bound.
fn check_integer_bounds(
    atomic_type: AtomicType,
    value: i64,
    lexical: &str,
) -> Result<(), XdmError> {
    let (min, max): (Option<i64>, Option<i64>) = match atomic_type {
        AtomicType::NonPositiveInteger => (None, Some(0)),
        AtomicType::NegativeInteger => (None, Some(-1)),
        AtomicType::Long => (Some(i64::MIN), Some(i64::MAX)),
        AtomicType::Int => (Some(i64::from(i32::MIN)), Some(i64::from(i32::MAX))),
        AtomicType::Short => (Some(i64::from(i16::MIN)), Some(i64::from(i16::MAX))),
        AtomicType::Byte => (Some(i64::from(i8::MIN)), Some(i64::from(i8::MAX))),
        AtomicType::NonNegativeInteger | AtomicType::UnsignedLong => (Some(0), None),
        AtomicType::UnsignedInt => (Some(0), Some(i64::from(u32::MAX))),
        AtomicType::UnsignedShort => (Some(0), Some(i64::from(u16::MAX))),
        AtomicType::UnsignedByte => (Some(0), Some(i64::from(u8::MAX))),
        AtomicType::PositiveInteger => (Some(1), None),
        _ => (None, None),
    };
    if min.is_some_and(|min| value < min) || max.is_some_and(|max| value > max) {
        return Err(forg0001(
            atomic_type,
            lexical,
            "the value is outside the type's range",
        ));
    }
    Ok(())
}

/// The `xs:language` pattern facet, `[a-zA-Z]{1,8}(-[a-zA-Z0-9]{1,8})*`.
fn is_language(lexical: &str) -> bool {
    let mut parts = lexical.split('-');
    let Some(primary) = parts.next() else {
        return false;
    };
    if primary.is_empty()
        || primary.len() > 8
        || !primary.bytes().all(|byte| byte.is_ascii_alphabetic())
    {
        return false;
    }
    parts.all(|part| {
        !part.is_empty() && part.len() <= 8 && part.bytes().all(|byte| byte.is_ascii_alphanumeric())
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn white_space_facets_are_applied_before_the_lexical_mapping() -> Result<(), XdmError> {
        assert_eq!(
            AtomicValue::parse(AtomicType::String, " a \t b ")?.string_value(),
            " a \t b "
        );
        assert_eq!(
            AtomicValue::parse(AtomicType::NormalizedString, " a \t b ")?.string_value(),
            " a   b "
        );
        assert_eq!(
            AtomicValue::parse(AtomicType::Token, " a \t b ")?.string_value(),
            "a b"
        );
        assert_eq!(
            AtomicValue::parse(AtomicType::Boolean, "\n true \n")?.as_bool(),
            Some(true)
        );
        Ok(())
    }

    #[test]
    fn integer_ranges_are_enforced() {
        for (ty, ok, bad) in [
            (AtomicType::Byte, "127", "128"),
            (AtomicType::Short, "-32768", "-32769"),
            (AtomicType::Int, "2147483647", "2147483648"),
            (AtomicType::UnsignedByte, "255", "256"),
            (AtomicType::UnsignedShort, "65535", "65536"),
            (AtomicType::UnsignedInt, "4294967295", "4294967296"),
            (AtomicType::PositiveInteger, "1", "0"),
            (AtomicType::NegativeInteger, "-1", "0"),
            (AtomicType::NonNegativeInteger, "0", "-1"),
            (AtomicType::NonPositiveInteger, "0", "1"),
        ] {
            assert!(
                AtomicValue::parse(ty, ok).is_ok(),
                "{ty} should accept {ok}"
            );
            assert!(
                AtomicValue::parse(ty, bad).is_err(),
                "{ty} should reject {bad}"
            );
        }
    }

    #[test]
    fn a_hundred_digit_integer_round_trips_through_its_string() {
        let text = "1".to_owned() + &"0".repeat(99);
        let value = AtomicValue::parse(AtomicType::Integer, &text).expect("parses");
        assert!(matches!(value.data(), AtomicData::BigInteger(_)));
        assert_eq!(value.string_value(), text);
        // string → integer → string is stable.
        let reparsed = AtomicValue::parse(AtomicType::Integer, &value.string_value()).expect("re");
        assert_eq!(reparsed, value);
    }

    #[test]
    fn overflowing_integers_promote_but_bounded_types_still_reject() {
        // `xs:integer` and the unbounded restrictions carry the value at arbitrary precision.
        for ty in [
            AtomicType::Integer,
            AtomicType::NonNegativeInteger,
            AtomicType::PositiveInteger,
        ] {
            let value = AtomicValue::parse(ty, "99999999999999999999999999").expect("promotes");
            assert_eq!(value.string_value(), "99999999999999999999999999");
        }
        // `xs:unsignedLong` reaches its true ceiling now that big values exist...
        assert!(AtomicValue::parse(AtomicType::UnsignedLong, "18446744073709551615").is_ok());
        assert!(AtomicValue::parse(AtomicType::UnsignedLong, "18446744073709551616").is_err());
        // ... but a bounded type whose range fits an i64 rejects every overflowing value.
        assert!(AtomicValue::parse(AtomicType::Long, "9223372036854775808").is_err());
        // A high-precision decimal overflows the fixed width and is carried exactly.
        let precise =
            AtomicValue::parse(AtomicType::Decimal, "9.99999999999999999999999999").expect("ok");
        assert!(matches!(precise.data(), AtomicData::BigDecimal(_)));
        assert_eq!(precise.string_value(), "9.99999999999999999999999999");
    }

    #[test]
    fn string_subtype_patterns_are_enforced() {
        assert!(AtomicValue::parse(AtomicType::Ncname, "abc").is_ok());
        assert!(AtomicValue::parse(AtomicType::Ncname, "a:b").is_err());
        assert!(AtomicValue::parse(AtomicType::Name, "a:b").is_ok());
        assert!(AtomicValue::parse(AtomicType::Nmtoken, "123").is_ok());
        assert!(AtomicValue::parse(AtomicType::Name, "123").is_err());
        assert!(AtomicValue::parse(AtomicType::Language, "en").is_ok());
        assert!(AtomicValue::parse(AtomicType::Language, "en-GB-oed").is_ok());
        assert!(AtomicValue::parse(AtomicType::Language, "abcdefghi").is_err());
        assert!(AtomicValue::parse(AtomicType::Language, "en-").is_err());
        assert!(AtomicValue::parse(AtomicType::Language, "1en").is_err());
    }

    #[test]
    fn date_time_stamp_requires_a_timezone() {
        assert!(AtomicValue::parse(AtomicType::DateTimeStamp, "2020-01-01T00:00:00Z").is_ok());
        assert!(AtomicValue::parse(AtomicType::DateTimeStamp, "2020-01-01T00:00:00").is_err());
        assert!(AtomicValue::parse(AtomicType::DateTime, "2020-01-01T00:00:00").is_ok());
    }

    #[test]
    fn binary_and_uri_lexical_mappings() -> Result<(), XdmError> {
        assert_eq!(
            AtomicValue::parse(AtomicType::HexBinary, " 0fa0 ")?.as_bytes(),
            Some(&[0x0f_u8, 0xa0][..])
        );
        assert_eq!(
            AtomicValue::parse(AtomicType::Base64Binary, "Zm9v")?.as_bytes(),
            Some(&b"foo"[..])
        );
        assert!(AtomicValue::parse(AtomicType::AnyUri, "http://example.com/a").is_ok());
        assert!(AtomicValue::parse(AtomicType::AnyUri, "../relative").is_ok());
        Ok(())
    }

    #[test]
    fn any_uri_escapes_before_validating() -> Result<(), XdmError> {
        // XSD 1.1 §3.3.17: spaces are in principle allowed in the lexical space, because the
        // lexical mapping escapes them before an IRI is required.
        assert_eq!(
            AtomicValue::parse(AtomicType::AnyUri, "a string")?.string_value(),
            "a string"
        );
        assert_eq!(
            AtomicValue::parse(AtomicType::AnyUri, " http://example.com/ foo.xml ")?.string_value(),
            "http://example.com/ foo.xml",
            "the whiteSpace facet is collapse, and the interior space survives escaping"
        );
        assert_eq!(escape_for_any_uri("a string"), "a%20string");
        assert_eq!(escape_for_any_uri("plain"), "plain");
        // `%` is never escaped, so a malformed triplet stays malformed.
        assert!(AtomicValue::parse(AtomicType::AnyUri, "%gg").is_err());
        assert!(AtomicValue::parse(AtomicType::AnyUri, ":/").is_err());
        Ok(())
    }

    #[test]
    fn abstract_and_unprefixed_qname_handling() {
        assert!(AtomicValue::parse(AtomicType::AnyAtomicType, "x").is_err());
        assert!(AtomicValue::parse(AtomicType::QName, "local").is_ok());
        assert!(AtomicValue::parse(AtomicType::QName, "p:local").is_err());
        assert!(
            AtomicValue::parse_qname(AtomicType::String, "x", |_| None).is_err(),
            "parse_qname only accepts the QName family"
        );
    }

    #[test]
    fn durations_map_to_their_specialized_payloads() -> Result<(), XdmError> {
        assert!(matches!(
            AtomicValue::parse(AtomicType::Duration, "P1Y2M")?.data(),
            AtomicData::Duration(_)
        ));
        assert!(matches!(
            AtomicValue::parse(AtomicType::YearMonthDuration, "P1Y2M")?.data(),
            AtomicData::YearMonthDuration(_)
        ));
        assert!(matches!(
            AtomicValue::parse(AtomicType::DayTimeDuration, "P1DT2H")?.data(),
            AtomicData::DayTimeDuration(_)
        ));
        assert!(AtomicValue::parse(AtomicType::DayTimeDuration, "P1Y").is_err());
        Ok(())
    }

    #[test]
    fn collapse_handles_leading_trailing_and_interior_runs() {
        assert_eq!(collapse_white_space("  a \n\t b  "), "a b");
        assert_eq!(collapse_white_space("   "), "");
        assert_eq!(collapse_white_space(""), "");
        assert_eq!(replace_white_space("a\tb\r\nc"), "a b  c");
    }
}
