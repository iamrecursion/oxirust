//! The [W3C XML Schema datatype library][xsd] (`http://www.w3.org/2001/XMLSchema-datatypes`).
//!
//! RELAX NG uses a datatype library through two services only: *does this string
//! denote a value of this datatype, given these parameters* and *do these two
//! strings denote the same value*. This module implements both for the whole of
//! the XSD 1.0 built-in datatype hierarchy, with the constraining facets exposed
//! as RELAX NG `param` elements.
//!
//! Values are compared in the **value space**, not lexically: `1.0` and `1.00`
//! are the same `decimal`, `p:a` and `q:a` are the same `QName` when `p` and `q`
//! are bound to the same namespace, and `2000-01-01T00:00:00Z` and
//! `1999-12-31T19:00:00-05:00` are the same `dateTime`.
//!
//! [xsd]: https://www.w3.org/TR/xmlschema-2/

use std::cmp::Ordering;
use std::str::FromStr;

use oxixml_regex::Regex;
use oxixml_xsd::{Date, DateTime, Duration, GDay, GMonth, GMonthDay, GYear, GYearMonth, Time};

use super::decimal::BigDecimal;
use super::{DatatypeContext, ParamError};
use crate::name::{is_name, is_ncname, is_nmtoken};
use crate::tree::{normalize_whitespace, split_tokens};

/// The URI that identifies this library.
pub(crate) const LIBRARY: &str = "http://www.w3.org/2001/XMLSchema-datatypes";

/// The XSD whiteSpace facet.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub(crate) enum WhiteSpace {
    Preserve,
    Replace,
    Collapse,
}

impl WhiteSpace {
    fn apply(self, value: &str) -> String {
        match self {
            Self::Preserve => value.to_owned(),
            Self::Replace => value
                .chars()
                .map(|c| {
                    if matches!(c, '\t' | '\n' | '\r') {
                        ' '
                    } else {
                        c
                    }
                })
                .collect(),
            Self::Collapse => normalize_whitespace(value),
        }
    }

    fn parse(value: &str) -> Option<Self> {
        match value {
            "preserve" => Some(Self::Preserve),
            "replace" => Some(Self::Replace),
            "collapse" => Some(Self::Collapse),
            _ => None,
        }
    }
}

/// Every built-in datatype of XML Schema Part 2.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[allow(clippy::enum_variant_names)]
pub(crate) enum XsdType {
    String,
    NormalizedString,
    Token,
    Language,
    Name,
    NcName,
    NmToken,
    Id,
    IdRef,
    Entity,
    NmTokens,
    IdRefs,
    Entities,
    Boolean,
    Decimal,
    Integer,
    NonPositiveInteger,
    NegativeInteger,
    Long,
    Int,
    Short,
    Byte,
    NonNegativeInteger,
    UnsignedLong,
    UnsignedInt,
    UnsignedShort,
    UnsignedByte,
    PositiveInteger,
    Float,
    Double,
    Duration,
    DateTime,
    Time,
    Date,
    GYearMonth,
    GYear,
    GMonthDay,
    GMonth,
    GDay,
    HexBinary,
    Base64Binary,
    AnyUri,
    QName,
    Notation,
}

impl XsdType {
    /// Maps a datatype name to its enumerator.
    pub(crate) fn lookup(name: &str) -> Option<Self> {
        Some(match name {
            "string" => Self::String,
            "normalizedString" => Self::NormalizedString,
            "token" => Self::Token,
            "language" => Self::Language,
            "Name" => Self::Name,
            "NCName" => Self::NcName,
            "NMTOKEN" => Self::NmToken,
            "ID" => Self::Id,
            "IDREF" => Self::IdRef,
            "ENTITY" => Self::Entity,
            "NMTOKENS" => Self::NmTokens,
            "IDREFS" => Self::IdRefs,
            "ENTITIES" => Self::Entities,
            "boolean" => Self::Boolean,
            "decimal" => Self::Decimal,
            "integer" => Self::Integer,
            "nonPositiveInteger" => Self::NonPositiveInteger,
            "negativeInteger" => Self::NegativeInteger,
            "long" => Self::Long,
            "int" => Self::Int,
            "short" => Self::Short,
            "byte" => Self::Byte,
            "nonNegativeInteger" => Self::NonNegativeInteger,
            "unsignedLong" => Self::UnsignedLong,
            "unsignedInt" => Self::UnsignedInt,
            "unsignedShort" => Self::UnsignedShort,
            "unsignedByte" => Self::UnsignedByte,
            "positiveInteger" => Self::PositiveInteger,
            "float" => Self::Float,
            "double" => Self::Double,
            "duration" => Self::Duration,
            "dateTime" => Self::DateTime,
            "time" => Self::Time,
            "date" => Self::Date,
            "gYearMonth" => Self::GYearMonth,
            "gYear" => Self::GYear,
            "gMonthDay" => Self::GMonthDay,
            "gMonth" => Self::GMonth,
            "gDay" => Self::GDay,
            "hexBinary" => Self::HexBinary,
            "base64Binary" => Self::Base64Binary,
            "anyURI" => Self::AnyUri,
            "QName" => Self::QName,
            "NOTATION" => Self::Notation,
            _ => return None,
        })
    }

    const fn default_whitespace(self) -> WhiteSpace {
        match self {
            Self::String => WhiteSpace::Preserve,
            Self::NormalizedString => WhiteSpace::Replace,
            _ => WhiteSpace::Collapse,
        }
    }

    /// Which family of constraining facets applies.
    const fn family(self) -> Family {
        match self {
            Self::String
            | Self::NormalizedString
            | Self::Token
            | Self::Language
            | Self::Name
            | Self::NcName
            | Self::NmToken
            | Self::Id
            | Self::IdRef
            | Self::Entity
            | Self::AnyUri
            | Self::QName
            | Self::Notation => Family::Text,
            Self::NmTokens | Self::IdRefs | Self::Entities => Family::List,
            Self::HexBinary | Self::Base64Binary => Family::Binary,
            Self::Boolean => Family::Unordered,
            Self::Decimal
            | Self::Integer
            | Self::NonPositiveInteger
            | Self::NegativeInteger
            | Self::Long
            | Self::Int
            | Self::Short
            | Self::Byte
            | Self::NonNegativeInteger
            | Self::UnsignedLong
            | Self::UnsignedInt
            | Self::UnsignedShort
            | Self::UnsignedByte
            | Self::PositiveInteger => Family::Numeric,
            _ => Family::Ordered,
        }
    }

    /// The inclusive bounds that define the derived integer types.
    fn integer_bounds(self) -> Option<(Option<&'static str>, Option<&'static str>)> {
        Some(match self {
            Self::Integer => (None, None),
            Self::NonPositiveInteger => (None, Some("0")),
            Self::NegativeInteger => (None, Some("-1")),
            Self::Long => (Some("-9223372036854775808"), Some("9223372036854775807")),
            Self::Int => (Some("-2147483648"), Some("2147483647")),
            Self::Short => (Some("-32768"), Some("32767")),
            Self::Byte => (Some("-128"), Some("127")),
            Self::NonNegativeInteger => (Some("0"), None),
            Self::UnsignedLong => (Some("0"), Some("18446744073709551615")),
            Self::UnsignedInt => (Some("0"), Some("4294967295")),
            Self::UnsignedShort => (Some("0"), Some("65535")),
            Self::UnsignedByte => (Some("0"), Some("255")),
            Self::PositiveInteger => (Some("1"), None),
            _ => return None,
        })
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Family {
    /// Character strings: length facets, no order.
    Text,
    /// Whitespace-separated lists: length facets counted in items.
    List,
    /// Binary: length facets counted in octets.
    Binary,
    /// Ordered by a numeric value: bounds plus digit counts.
    Numeric,
    /// Ordered, but with no digit counts.
    Ordered,
    /// Neither ordered nor measurable.
    Unordered,
}

/// A value of one of the XSD datatypes.
#[derive(Debug, Clone)]
pub(crate) enum Value {
    Text(String),
    Boolean(bool),
    Number(BigDecimal),
    Double(f64),
    Duration(Duration),
    DateTime(DateTime),
    Time(Time),
    Date(Date),
    GYearMonth(GYearMonth),
    GYear(GYear),
    GMonthDay(GMonthDay),
    GMonth(GMonth),
    GDay(GDay),
    Binary(Vec<u8>),
    QName(Option<String>, String),
    List(Vec<Value>),
}

impl PartialEq for Value {
    fn eq(&self, other: &Self) -> bool {
        match (self, other) {
            (Self::Text(a), Self::Text(b)) => a == b,
            (Self::Boolean(a), Self::Boolean(b)) => a == b,
            (Self::Number(a), Self::Number(b)) => a == b,
            // `datatypeEqual` must be reflexive, so NaN equals itself here.
            (Self::Double(a), Self::Double(b)) => a == b || (a.is_nan() && b.is_nan()),
            (Self::Duration(a), Self::Duration(b)) => a == b,
            (Self::DateTime(a), Self::DateTime(b)) => a == b,
            (Self::Time(a), Self::Time(b)) => a == b,
            (Self::Date(a), Self::Date(b)) => a == b,
            (Self::GYearMonth(a), Self::GYearMonth(b)) => a == b,
            (Self::GYear(a), Self::GYear(b)) => a == b,
            (Self::GMonthDay(a), Self::GMonthDay(b)) => a == b,
            (Self::GMonth(a), Self::GMonth(b)) => a == b,
            (Self::GDay(a), Self::GDay(b)) => a == b,
            (Self::Binary(a), Self::Binary(b)) => a == b,
            (Self::QName(ns1, l1), Self::QName(ns2, l2)) => ns1 == ns2 && l1 == l2,
            (Self::List(a), Self::List(b)) => a == b,
            _ => false,
        }
    }
}

impl Value {
    fn partial_cmp_value(&self, other: &Self) -> Option<Ordering> {
        match (self, other) {
            (Self::Number(a), Self::Number(b)) => Some(a.cmp(b)),
            (Self::Double(a), Self::Double(b)) => a.partial_cmp(b),
            (Self::Duration(a), Self::Duration(b)) => a.partial_cmp(b),
            (Self::DateTime(a), Self::DateTime(b)) => a.partial_cmp(b),
            (Self::Time(a), Self::Time(b)) => a.partial_cmp(b),
            (Self::Date(a), Self::Date(b)) => a.partial_cmp(b),
            (Self::GYearMonth(a), Self::GYearMonth(b)) => a.partial_cmp(b),
            (Self::GYear(a), Self::GYear(b)) => a.partial_cmp(b),
            (Self::GMonthDay(a), Self::GMonthDay(b)) => a.partial_cmp(b),
            (Self::GMonth(a), Self::GMonth(b)) => a.partial_cmp(b),
            (Self::GDay(a), Self::GDay(b)) => a.partial_cmp(b),
            _ => None,
        }
    }

    /// The measure the `length` facets constrain, when the family defines one.
    fn measure(&self, family: Family) -> Option<usize> {
        match (family, self) {
            (Family::Text, Self::Text(text)) => Some(text.chars().count()),
            (Family::Text, Self::QName(_, local)) => Some(local.chars().count()),
            (Family::List, Self::List(items)) => Some(items.len()),
            (Family::Binary, Self::Binary(bytes)) => Some(bytes.len()),
            _ => None,
        }
    }
}

/// A datatype from this library, together with the facets a `data` pattern
/// applied to it.
#[derive(Debug)]
pub(crate) struct XsdDatatype {
    ty: XsdType,
    whitespace: WhiteSpace,
    patterns: Vec<Regex>,
    length: Option<usize>,
    min_length: Option<usize>,
    max_length: Option<usize>,
    min_inclusive: Option<Value>,
    max_inclusive: Option<Value>,
    min_exclusive: Option<Value>,
    max_exclusive: Option<Value>,
    total_digits: Option<usize>,
    fraction_digits: Option<usize>,
}

impl XsdDatatype {
    /// Builds a datatype with no facets.
    pub(crate) fn plain(ty: XsdType) -> Self {
        Self {
            ty,
            whitespace: ty.default_whitespace(),
            patterns: Vec::new(),
            length: None,
            min_length: None,
            max_length: None,
            min_inclusive: None,
            max_inclusive: None,
            min_exclusive: None,
            max_exclusive: None,
            total_digits: None,
            fraction_digits: None,
        }
    }

    /// Compiles a datatype together with the `param` list of a `data` pattern.
    pub(crate) fn compile(
        name: &str,
        params: &[(String, String)],
        context: &DatatypeContext,
    ) -> Result<Self, ParamError> {
        let ty = XsdType::lookup(name).ok_or_else(|| ParamError::UnknownType(name.to_owned()))?;
        let mut datatype = Self::plain(ty);
        let family = ty.family();
        let mut seen: Vec<&str> = Vec::new();

        for (param, value) in params {
            let name = param.as_str();
            if name != "pattern" {
                if seen.contains(&name) {
                    return Err(ParamError::Duplicate(name.to_owned()));
                }
                seen.push(name);
            }
            match name {
                "pattern" => {
                    let regex = Regex::xsd(value).map_err(|error| {
                        ParamError::BadValue(name.to_owned(), error.to_string())
                    })?;
                    datatype.patterns.push(regex);
                }
                "whiteSpace" => {
                    let requested = WhiteSpace::parse(value)
                        .ok_or_else(|| ParamError::BadValue(name.to_owned(), value.clone()))?;
                    if requested < ty.default_whitespace() {
                        return Err(ParamError::BadValue(
                            name.to_owned(),
                            format!("`{value}` weakens the whiteSpace of `{}`", type_name(ty)),
                        ));
                    }
                    datatype.whitespace = requested;
                }
                "length" | "minLength" | "maxLength" => {
                    if !matches!(family, Family::Text | Family::List | Family::Binary) {
                        return Err(ParamError::NotApplicable(
                            name.to_owned(),
                            type_name(ty).to_owned(),
                        ));
                    }
                    let count = parse_count(name, value)?;
                    match name {
                        "length" => datatype.length = Some(count),
                        "minLength" => datatype.min_length = Some(count),
                        _ => datatype.max_length = Some(count),
                    }
                }
                "minInclusive" | "maxInclusive" | "minExclusive" | "maxExclusive" => {
                    if !matches!(family, Family::Numeric | Family::Ordered) {
                        return Err(ParamError::NotApplicable(
                            name.to_owned(),
                            type_name(ty).to_owned(),
                        ));
                    }
                    let bound = datatype
                        .value(value, context)
                        .ok_or_else(|| ParamError::BadValue(name.to_owned(), value.clone()))?;
                    match name {
                        "minInclusive" => datatype.min_inclusive = Some(bound),
                        "maxInclusive" => datatype.max_inclusive = Some(bound),
                        "minExclusive" => datatype.min_exclusive = Some(bound),
                        _ => datatype.max_exclusive = Some(bound),
                    }
                }
                "totalDigits" | "fractionDigits" => {
                    if family != Family::Numeric {
                        return Err(ParamError::NotApplicable(
                            name.to_owned(),
                            type_name(ty).to_owned(),
                        ));
                    }
                    let count = parse_count(name, value)?;
                    if name == "totalDigits" {
                        if count == 0 {
                            return Err(ParamError::BadValue(name.to_owned(), value.clone()));
                        }
                        datatype.total_digits = Some(count);
                    } else {
                        datatype.fraction_digits = Some(count);
                    }
                }
                other => return Err(ParamError::UnknownParam(other.to_owned())),
            }
        }
        Ok(datatype)
    }

    /// `datatypeAllows`: is `lexical` a legal representation of this datatype?
    pub(crate) fn allows(&self, lexical: &str, context: &DatatypeContext) -> bool {
        let normalized = self.whitespace.apply(lexical);
        if !self.patterns.is_empty()
            && !self
                .patterns
                .iter()
                .all(|pattern| pattern.is_match(&normalized))
        {
            return false;
        }
        let Some(value) = self.parse_normalized(&normalized, context) else {
            return false;
        };
        self.check_facets(&value)
    }

    /// `datatypeEqual`: the value of `lexical`, ignoring the facets.
    pub(crate) fn value(&self, lexical: &str, context: &DatatypeContext) -> Option<Value> {
        let normalized = self.whitespace.apply(lexical);
        self.parse_normalized(&normalized, context)
    }

    fn check_facets(&self, value: &Value) -> bool {
        let family = self.ty.family();
        if self.length.is_some() || self.min_length.is_some() || self.max_length.is_some() {
            let Some(measure) = value.measure(family) else {
                return false;
            };
            if self.length.is_some_and(|length| measure != length)
                || self.min_length.is_some_and(|min| measure < min)
                || self.max_length.is_some_and(|max| measure > max)
            {
                return false;
            }
        }
        if let Some(bound) = &self.min_inclusive
            && !matches!(
                value.partial_cmp_value(bound),
                Some(Ordering::Greater | Ordering::Equal)
            )
        {
            return false;
        }
        if let Some(bound) = &self.max_inclusive
            && !matches!(
                value.partial_cmp_value(bound),
                Some(Ordering::Less | Ordering::Equal)
            )
        {
            return false;
        }
        if let Some(bound) = &self.min_exclusive
            && !matches!(value.partial_cmp_value(bound), Some(Ordering::Greater))
        {
            return false;
        }
        if let Some(bound) = &self.max_exclusive
            && !matches!(value.partial_cmp_value(bound), Some(Ordering::Less))
        {
            return false;
        }
        if let Value::Number(number) = value {
            if self
                .total_digits
                .is_some_and(|limit| number.total_digits() > limit)
            {
                return false;
            }
            if self
                .fraction_digits
                .is_some_and(|limit| number.fraction_digits() > limit)
            {
                return false;
            }
        }
        true
    }

    fn parse_normalized(&self, value: &str, context: &DatatypeContext) -> Option<Value> {
        parse_value(self.ty, value, context)
    }
}

fn parse_count(name: &str, value: &str) -> Result<usize, ParamError> {
    let trimmed = crate::tree::trim_whitespace(value);
    trimmed
        .parse::<usize>()
        .map_err(|_| ParamError::BadValue(name.to_owned(), value.to_owned()))
}

/// The spelling of a datatype, for diagnostics.
fn type_name(ty: XsdType) -> &'static str {
    match ty {
        XsdType::String => "string",
        XsdType::NormalizedString => "normalizedString",
        XsdType::Token => "token",
        XsdType::Language => "language",
        XsdType::Name => "Name",
        XsdType::NcName => "NCName",
        XsdType::NmToken => "NMTOKEN",
        XsdType::Id => "ID",
        XsdType::IdRef => "IDREF",
        XsdType::Entity => "ENTITY",
        XsdType::NmTokens => "NMTOKENS",
        XsdType::IdRefs => "IDREFS",
        XsdType::Entities => "ENTITIES",
        XsdType::Boolean => "boolean",
        XsdType::Decimal => "decimal",
        XsdType::Integer => "integer",
        XsdType::NonPositiveInteger => "nonPositiveInteger",
        XsdType::NegativeInteger => "negativeInteger",
        XsdType::Long => "long",
        XsdType::Int => "int",
        XsdType::Short => "short",
        XsdType::Byte => "byte",
        XsdType::NonNegativeInteger => "nonNegativeInteger",
        XsdType::UnsignedLong => "unsignedLong",
        XsdType::UnsignedInt => "unsignedInt",
        XsdType::UnsignedShort => "unsignedShort",
        XsdType::UnsignedByte => "unsignedByte",
        XsdType::PositiveInteger => "positiveInteger",
        XsdType::Float => "float",
        XsdType::Double => "double",
        XsdType::Duration => "duration",
        XsdType::DateTime => "dateTime",
        XsdType::Time => "time",
        XsdType::Date => "date",
        XsdType::GYearMonth => "gYearMonth",
        XsdType::GYear => "gYear",
        XsdType::GMonthDay => "gMonthDay",
        XsdType::GMonth => "gMonth",
        XsdType::GDay => "gDay",
        XsdType::HexBinary => "hexBinary",
        XsdType::Base64Binary => "base64Binary",
        XsdType::AnyUri => "anyURI",
        XsdType::QName => "QName",
        XsdType::Notation => "NOTATION",
    }
}

fn parse_value(ty: XsdType, value: &str, context: &DatatypeContext) -> Option<Value> {
    match ty {
        XsdType::String | XsdType::NormalizedString | XsdType::Token => {
            Some(Value::Text(value.to_owned()))
        }
        XsdType::Language => is_language(value).then(|| Value::Text(value.to_owned())),
        XsdType::Name => is_name(value).then(|| Value::Text(value.to_owned())),
        XsdType::NcName | XsdType::Id | XsdType::IdRef | XsdType::Entity => {
            is_ncname(value).then(|| Value::Text(value.to_owned()))
        }
        XsdType::NmToken => is_nmtoken(value).then(|| Value::Text(value.to_owned())),
        XsdType::NmTokens | XsdType::IdRefs | XsdType::Entities => {
            let item = match ty {
                XsdType::NmTokens => XsdType::NmToken,
                XsdType::IdRefs => XsdType::IdRef,
                _ => XsdType::Entity,
            };
            let tokens = split_tokens(value);
            if tokens.is_empty() {
                return None;
            }
            let mut items = Vec::with_capacity(tokens.len());
            for token in tokens {
                items.push(parse_value(item, token, context)?);
            }
            Some(Value::List(items))
        }
        XsdType::Boolean => match value {
            "true" | "1" => Some(Value::Boolean(true)),
            "false" | "0" => Some(Value::Boolean(false)),
            _ => None,
        },
        XsdType::Decimal => BigDecimal::parse(value).map(Value::Number),
        XsdType::Integer
        | XsdType::NonPositiveInteger
        | XsdType::NegativeInteger
        | XsdType::Long
        | XsdType::Int
        | XsdType::Short
        | XsdType::Byte
        | XsdType::NonNegativeInteger
        | XsdType::UnsignedLong
        | XsdType::UnsignedInt
        | XsdType::UnsignedShort
        | XsdType::UnsignedByte
        | XsdType::PositiveInteger => {
            let number = BigDecimal::parse_integer(value)?;
            let (low, high) = ty.integer_bounds()?;
            if let Some(low) = low.and_then(BigDecimal::parse_integer)
                && number < low
            {
                return None;
            }
            if let Some(high) = high.and_then(BigDecimal::parse_integer)
                && number > high
            {
                return None;
            }
            Some(Value::Number(number))
        }
        XsdType::Float => parse_binary_float(value).map(|f| Value::Double(f64::from(f as f32))),
        XsdType::Double => parse_binary_float(value).map(Value::Double),
        XsdType::Duration => Duration::from_str(value).ok().map(Value::Duration),
        XsdType::DateTime => DateTime::from_str(value).ok().map(Value::DateTime),
        XsdType::Time => Time::from_str(value).ok().map(Value::Time),
        XsdType::Date => Date::from_str(value).ok().map(Value::Date),
        XsdType::GYearMonth => GYearMonth::from_str(value).ok().map(Value::GYearMonth),
        XsdType::GYear => GYear::from_str(value).ok().map(Value::GYear),
        XsdType::GMonthDay => GMonthDay::from_str(value).ok().map(Value::GMonthDay),
        XsdType::GMonth => GMonth::from_str(value).ok().map(Value::GMonth),
        XsdType::GDay => GDay::from_str(value).ok().map(Value::GDay),
        XsdType::HexBinary => parse_hex_binary(value).map(Value::Binary),
        XsdType::Base64Binary => parse_base64(value).map(Value::Binary),
        XsdType::AnyUri => {
            let escaped = crate::uri::xlink_escape(value);
            crate::uri::parse_uri_reference(&escaped).map(|_| Value::Text(value.to_owned()))
        }
        XsdType::QName | XsdType::Notation => parse_qname(value, context),
    }
}

/// `[a-zA-Z]{1,8}(-[a-zA-Z0-9]{1,8})*`, the XSD `language` pattern.
fn is_language(value: &str) -> bool {
    let mut parts = value.split('-');
    let Some(first) = parts.next() else {
        return false;
    };
    if first.is_empty() || first.len() > 8 || !first.bytes().all(|b| b.is_ascii_alphabetic()) {
        return false;
    }
    parts.all(|part| {
        !part.is_empty() && part.len() <= 8 && part.bytes().all(|b| b.is_ascii_alphanumeric())
    })
}

/// The XSD lexical space of `float` and `double`: a mantissa with an optional
/// exponent, or one of `INF`, `-INF`, `+INF` and `NaN`.
fn parse_binary_float(value: &str) -> Option<f64> {
    match value {
        "INF" | "+INF" => return Some(f64::INFINITY),
        "-INF" => return Some(f64::NEG_INFINITY),
        "NaN" => return Some(f64::NAN),
        _ => {}
    }
    let (mantissa, exponent) = match value.find(['e', 'E']) {
        Some(index) => (&value[..index], Some(&value[index + 1..])),
        None => (value, None),
    };
    BigDecimal::parse(mantissa)?;
    if let Some(exponent) = exponent {
        let digits = exponent.strip_prefix(['+', '-']).unwrap_or(exponent);
        if digits.is_empty() || !digits.bytes().all(|b| b.is_ascii_digit()) {
            return None;
        }
    }
    value.parse::<f64>().ok()
}

fn parse_hex_binary(value: &str) -> Option<Vec<u8>> {
    if value.len() % 2 != 0 || !value.bytes().all(|b| b.is_ascii_hexdigit()) {
        return None;
    }
    let bytes = value.as_bytes();
    let mut out = Vec::with_capacity(bytes.len() / 2);
    for pair in bytes.chunks_exact(2) {
        let high = char::from(pair[0]).to_digit(16)?;
        let low = char::from(pair[1]).to_digit(16)?;
        out.push(u8::try_from(high * 16 + low).ok()?);
    }
    Some(out)
}

fn parse_base64(value: &str) -> Option<Vec<u8>> {
    let cleaned: Vec<u8> = value.bytes().filter(|b| *b != b' ').collect();
    if cleaned.len() % 4 != 0 {
        return None;
    }
    let mut out = Vec::with_capacity(cleaned.len() / 4 * 3);
    for quad in cleaned.chunks_exact(4) {
        let padding = quad.iter().filter(|b| **b == b'=').count();
        if padding > 2 || (padding > 0 && quad[3] != b'=') {
            return None;
        }
        if padding == 2 && quad[2] != b'=' {
            return None;
        }
        // The XSD lexical space forces the unused bits of the last significant
        // character to be zero: `B64 B64 B16 '='` and `B64 B04 '=' '='`.
        if padding == 1 && !b"AEIMQUYcgkosw048".contains(&quad[2]) {
            return None;
        }
        if padding == 2 && !b"AQgw".contains(&quad[1]) {
            return None;
        }
        let mut accumulator = 0_u32;
        for (index, byte) in quad.iter().enumerate() {
            let sextet = if *byte == b'=' {
                if index < 4 - padding {
                    return None;
                }
                0
            } else {
                base64_value(*byte)?
            };
            accumulator = (accumulator << 6) | u32::from(sextet);
        }
        let triple = accumulator.to_be_bytes();
        out.push(triple[1]);
        if padding < 2 {
            out.push(triple[2]);
        }
        if padding < 1 {
            out.push(triple[3]);
        }
    }
    Some(out)
}

fn base64_value(byte: u8) -> Option<u8> {
    Some(match byte {
        b'A'..=b'Z' => byte - b'A',
        b'a'..=b'z' => byte - b'a' + 26,
        b'0'..=b'9' => byte - b'0' + 52,
        b'+' => 62,
        b'/' => 63,
        _ => return None,
    })
}

fn parse_qname(value: &str, context: &DatatypeContext) -> Option<Value> {
    match value.split_once(':') {
        Some((prefix, local)) => {
            if !is_ncname(prefix) || !is_ncname(local) {
                return None;
            }
            let namespace = context.namespaces.lookup(prefix)?;
            Some(Value::QName(Some(namespace.to_owned()), local.to_owned()))
        }
        None => {
            if !is_ncname(value) {
                return None;
            }
            Some(Value::QName(
                context.namespaces.default_namespace().map(str::to_owned),
                value.to_owned(),
            ))
        }
    }
}

#[cfg(test)]
mod tests {
    use std::rc::Rc;

    use super::*;
    use crate::tree::NamespaceMap;

    fn context() -> DatatypeContext {
        DatatypeContext {
            namespaces: Rc::new(NamespaceMap::initial()),
        }
    }

    fn allows(name: &str, value: &str) -> bool {
        let ty = XsdType::lookup(name).expect("a known datatype");
        XsdDatatype::plain(ty).allows(value, &context())
    }

    #[test]
    fn integer_ranges_are_enforced() {
        assert!(allows("byte", "127"));
        assert!(!allows("byte", "128"));
        assert!(allows("unsignedLong", "18446744073709551615"));
        assert!(!allows("unsignedLong", "18446744073709551616"));
        assert!(allows("positiveInteger", "1"));
        assert!(!allows("positiveInteger", "0"));
        assert!(!allows("integer", "1.0"));
    }

    #[test]
    fn whitespace_is_collapsed_for_most_types() {
        assert!(allows("integer", "  12\n"));
        assert!(allows("string", "  12\n"));
        assert!(!allows("NCName", "a b"));
    }

    #[test]
    fn floats_follow_the_xsd_lexical_space() {
        assert!(allows("double", "1.0E-3"));
        assert!(allows("double", "INF"));
        assert!(allows("double", "NaN"));
        assert!(!allows("double", "infinity"));
        assert!(!allows("double", "1e"));
    }

    #[test]
    fn binary_types_round_trip() {
        assert!(allows("hexBinary", "0FB7"));
        assert!(!allows("hexBinary", "0FB"));
        assert!(allows("base64Binary", "YW55IGNhcm5hbCBwbGVhcw=="));
        assert!(!allows("base64Binary", "YW5="));
    }

    #[test]
    fn facets_are_applied() -> Result<(), ParamError> {
        let datatype = XsdDatatype::compile(
            "string",
            &[("minLength".to_owned(), "3".to_owned())],
            &context(),
        )?;
        assert!(datatype.allows("abc", &context()));
        assert!(!datatype.allows("ab", &context()));

        let bounded = XsdDatatype::compile(
            "decimal",
            &[
                ("minInclusive".to_owned(), "1".to_owned()),
                ("maxInclusive".to_owned(), "10".to_owned()),
            ],
            &context(),
        )?;
        assert!(bounded.allows("1.0", &context()));
        assert!(!bounded.allows("10.5", &context()));
        Ok(())
    }

    #[test]
    fn inapplicable_facets_are_rejected() {
        assert!(
            XsdDatatype::compile(
                "string",
                &[("maxInclusive".to_owned(), "1".to_owned())],
                &context()
            )
            .is_err()
        );
        assert!(
            XsdDatatype::compile(
                "decimal",
                &[("length".to_owned(), "1".to_owned())],
                &context()
            )
            .is_err()
        );
        assert!(
            XsdDatatype::compile(
                "string",
                &[("bogus".to_owned(), "1".to_owned())],
                &context()
            )
            .is_err()
        );
    }

    #[test]
    fn values_compare_in_the_value_space() {
        let datatype = XsdDatatype::plain(XsdType::Decimal);
        assert_eq!(
            datatype.value("1.0", &context()),
            datatype.value("1.00", &context())
        );
        let token = XsdDatatype::plain(XsdType::Token);
        assert_eq!(
            token.value(" a  b ", &context()),
            token.value("a b", &context())
        );
    }

    #[test]
    fn language_pattern() {
        assert!(allows("language", "en"));
        assert!(allows("language", "en-GB"));
        assert!(!allows("language", "e1"));
        assert!(!allows("language", "-en"));
    }
}
