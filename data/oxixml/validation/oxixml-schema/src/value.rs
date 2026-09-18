//! The XML Schema *value space*: the primitive datatypes, the `whiteSpace`
//! normalizations, and the lexical mappings that turn a character string into a
//! comparable value.
//!
//! Everything a facet has to decide — is this within `minInclusive`, does it
//! equal an `enumeration` member, how long is it — is decided here, on values,
//! never on lexical forms. `1.0` and `1.00` are the same `decimal`, `p:a` and
//! `q:a` are the same `QName` when `p` and `q` denote the same namespace, and
//! `2000-01-01T00:00:00Z` is the same `dateTime` as `1999-12-31T19:00:00-05:00`.

use std::cmp::Ordering;
use std::str::FromStr;

use oxixml_qname::{is_name, is_ncname, is_nmtoken};
use oxixml_xsd::{
    Date, DateTime, DayTimeDuration, Duration, GDay, GMonth, GMonthDay, GYear, GYearMonth, Time,
    YearMonthDuration,
};

use crate::bigdecimal::BigDecimal;
use crate::names::XsdVersion;

/// The `whiteSpace` facet: how a lexical form is normalized before it is mapped
/// onto a value.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Default)]
pub enum WhiteSpace {
    /// Leave the string exactly as it is (`xs:string`).
    Preserve,
    /// Replace every tab, line feed and carriage return with a space
    /// (`xs:normalizedString`).
    Replace,
    /// Replace, then collapse runs of spaces and trim the ends (`xs:token` and
    /// every non-string datatype).
    #[default]
    Collapse,
}

impl WhiteSpace {
    /// Applies this normalization to `value`.
    #[must_use]
    pub fn apply<'a>(self, value: &'a str) -> std::borrow::Cow<'a, str> {
        match self {
            Self::Preserve => std::borrow::Cow::Borrowed(value),
            Self::Replace => {
                if value.bytes().any(|b| matches!(b, b'\t' | b'\n' | b'\r')) {
                    std::borrow::Cow::Owned(
                        value
                            .chars()
                            .map(|c| {
                                if matches!(c, '\t' | '\n' | '\r') {
                                    ' '
                                } else {
                                    c
                                }
                            })
                            .collect(),
                    )
                } else {
                    std::borrow::Cow::Borrowed(value)
                }
            }
            Self::Collapse => std::borrow::Cow::Owned(collapse(value)),
        }
    }

    /// Parses the lexical form of the facet value.
    #[must_use]
    pub fn parse(value: &str) -> Option<Self> {
        match value {
            "preserve" => Some(Self::Preserve),
            "replace" => Some(Self::Replace),
            "collapse" => Some(Self::Collapse),
            _ => None,
        }
    }
}

impl std::fmt::Display for WhiteSpace {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(match self {
            Self::Preserve => "preserve",
            Self::Replace => "replace",
            Self::Collapse => "collapse",
        })
    }
}

/// Collapses XML whitespace: every run of `#x20 #x9 #xD #xA` becomes a single
/// space, and leading and trailing whitespace is removed.
#[must_use]
pub fn collapse(value: &str) -> String {
    let mut out = String::with_capacity(value.len());
    for token in split_tokens(value) {
        if !out.is_empty() {
            out.push(' ');
        }
        out.push_str(token);
    }
    out
}

/// Splits on runs of XML whitespace, discarding empty tokens.
#[must_use]
pub fn split_tokens(value: &str) -> Vec<&str> {
    value
        .split([' ', '\t', '\n', '\r'])
        .filter(|token| !token.is_empty())
        .collect()
}

/// The 1.1 `explicitTimezone` facet.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ExplicitTimezone {
    /// A timezone offset must be present.
    Required,
    /// A timezone offset must be absent.
    Prohibited,
    /// A timezone offset may be present or absent (the default).
    Optional,
}

impl ExplicitTimezone {
    /// Parses the lexical form of the facet value.
    #[must_use]
    pub fn parse(value: &str) -> Option<Self> {
        match value {
            "required" => Some(Self::Required),
            "prohibited" => Some(Self::Prohibited),
            "optional" => Some(Self::Optional),
            _ => None,
        }
    }
}

/// One of the primitive datatypes of [XML Schema Part 2][part2], plus the two
/// "ur" types that sit above them.
///
/// Every atomic simple type has exactly one primitive ancestor, and that
/// ancestor alone decides the lexical mapping, the ordering and what the length
/// facets measure.
///
/// [part2]: https://www.w3.org/TR/xmlschema11-2/
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Primitive {
    /// `xs:anySimpleType` — no constraints at all; every string is a value.
    AnySimpleType,
    /// `xs:anyAtomicType` — the 1.1 root of the atomic types.
    AnyAtomicType,
    /// `xs:string`
    String,
    /// `xs:boolean`
    Boolean,
    /// `xs:decimal`
    Decimal,
    /// `xs:float`
    Float,
    /// `xs:double`
    Double,
    /// `xs:duration`
    Duration,
    /// `xs:dateTime`
    DateTime,
    /// `xs:time`
    Time,
    /// `xs:date`
    Date,
    /// `xs:gYearMonth`
    GYearMonth,
    /// `xs:gYear`
    GYear,
    /// `xs:gMonthDay`
    GMonthDay,
    /// `xs:gMonth`
    GMonth,
    /// `xs:gDay`
    GDay,
    /// `xs:hexBinary`
    HexBinary,
    /// `xs:base64Binary`
    Base64Binary,
    /// `xs:anyURI`
    AnyUri,
    /// `xs:QName`
    QName,
    /// `xs:NOTATION`
    Notation,
}

impl Primitive {
    /// The `whiteSpace` value fixed for this primitive.
    #[must_use]
    pub const fn white_space(self) -> WhiteSpace {
        match self {
            Self::String | Self::AnySimpleType | Self::AnyAtomicType => WhiteSpace::Preserve,
            _ => WhiteSpace::Collapse,
        }
    }

    /// Whether values of this primitive are ordered (so the bounds facets
    /// apply).
    #[must_use]
    pub const fn is_ordered(self) -> bool {
        matches!(
            self,
            Self::Decimal
                | Self::Float
                | Self::Double
                | Self::Duration
                | Self::DateTime
                | Self::Time
                | Self::Date
                | Self::GYearMonth
                | Self::GYear
                | Self::GMonthDay
                | Self::GMonth
                | Self::GDay
        )
    }

    /// Whether the `length` family of facets applies.
    #[must_use]
    pub const fn is_measurable(self) -> bool {
        matches!(
            self,
            Self::String | Self::HexBinary | Self::Base64Binary | Self::AnyUri
        )
    }

    /// Whether `totalDigits`/`fractionDigits` apply.
    #[must_use]
    pub const fn is_numeric_decimal(self) -> bool {
        matches!(self, Self::Decimal)
    }

    /// Whether `explicitTimezone` applies (the seven date/time primitives).
    #[must_use]
    pub const fn has_timezone(self) -> bool {
        matches!(
            self,
            Self::DateTime
                | Self::Time
                | Self::Date
                | Self::GYearMonth
                | Self::GYear
                | Self::GMonthDay
                | Self::GMonth
                | Self::GDay
        )
    }

    /// The spelling of the primitive in the XML Schema namespace.
    #[must_use]
    pub const fn name(self) -> &'static str {
        match self {
            Self::AnySimpleType => "anySimpleType",
            Self::AnyAtomicType => "anyAtomicType",
            Self::String => "string",
            Self::Boolean => "boolean",
            Self::Decimal => "decimal",
            Self::Float => "float",
            Self::Double => "double",
            Self::Duration => "duration",
            Self::DateTime => "dateTime",
            Self::Time => "time",
            Self::Date => "date",
            Self::GYearMonth => "gYearMonth",
            Self::GYear => "gYear",
            Self::GMonthDay => "gMonthDay",
            Self::GMonth => "gMonth",
            Self::GDay => "gDay",
            Self::HexBinary => "hexBinary",
            Self::Base64Binary => "base64Binary",
            Self::AnyUri => "anyURI",
            Self::QName => "QName",
            Self::Notation => "NOTATION",
        }
    }
}

/// The namespace bindings in force where a lexical form was written.
///
/// `QName` and `NOTATION` values cannot be computed without them, both in a
/// schema document (where a `fixed`/`default`/`enumeration` value may be a
/// `QName`) and in an instance.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct NamespaceContext {
    entries: Vec<(Option<String>, String)>,
}

impl NamespaceContext {
    /// An empty context: only the built-in `xml` prefix resolves.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Adds a binding; later bindings shadow earlier ones.
    pub fn declare(&mut self, prefix: Option<&str>, namespace: &str) {
        self.entries
            .push((prefix.map(str::to_owned), namespace.to_owned()));
    }

    /// Every binding in force, innermost first, with each prefix appearing
    /// once.
    ///
    /// An *undeclaration* — a binding to the empty namespace name — is
    /// reported as it was written, so that a caller rebuilding a static
    /// context can tell "not bound" from "never declared".
    pub fn bindings(&self) -> impl Iterator<Item = (Option<&str>, &str)> {
        let mut seen: Vec<Option<&str>> = Vec::new();
        let mut out = Vec::new();
        for (prefix, namespace) in self.entries.iter().rev() {
            let prefix = prefix.as_deref();
            if seen.contains(&prefix) {
                continue;
            }
            seen.push(prefix);
            out.push((prefix, namespace.as_str()));
        }
        out.into_iter()
    }

    /// Resolves `prefix` (or the default namespace when [`None`]).
    #[must_use]
    pub fn lookup(&self, prefix: Option<&str>) -> Option<&str> {
        for (declared, namespace) in self.entries.iter().rev() {
            if declared.as_deref() == prefix {
                return if namespace.is_empty() {
                    None
                } else {
                    Some(namespace.as_str())
                };
            }
        }
        if prefix == Some("xml") {
            return Some(crate::names::XML_NS);
        }
        None
    }
}

/// A value of one of the XML Schema datatypes.
///
/// Equality is *value-space* equality, and [`Value::partial_cmp_value`] is the
/// spec's partial order (values that the specification leaves incomparable —
/// `P1M` and `P30D`, or a `date` with a timezone and one without whose ranges
/// overlap — compare as [`None`]).
#[derive(Debug, Clone)]
pub enum Value {
    /// `string` and every type derived from it.
    String(String),
    /// `boolean`.
    Boolean(bool),
    /// `decimal` and every type derived from it, including all the integers.
    Decimal(BigDecimal),
    /// `float`.
    Float(f32),
    /// `double`.
    Double(f64),
    /// `duration`.
    Duration(Duration),
    /// `yearMonthDuration` (1.1).
    YearMonthDuration(YearMonthDuration),
    /// `dayTimeDuration` (1.1).
    DayTimeDuration(DayTimeDuration),
    /// `dateTime` and `dateTimeStamp`.
    DateTime(DateTime),
    /// `time`.
    Time(Time),
    /// `date`.
    Date(Date),
    /// `gYearMonth`.
    GYearMonth(GYearMonth),
    /// `gYear`.
    GYear(GYear),
    /// `gMonthDay`.
    GMonthDay(GMonthDay),
    /// `gMonth`.
    GMonth(GMonth),
    /// `gDay`.
    GDay(GDay),
    /// `hexBinary` and `base64Binary`, as the octets they denote.
    Binary(Vec<u8>),
    /// `anyURI`; equality is on the character string, as the specification says.
    Uri(String),
    /// `QName` and `NOTATION`, as a namespace name and a local name.
    QName(Option<String>, String),
    /// A list value: the values of its items, in order.
    List(Vec<Value>),
}

impl PartialEq for Value {
    fn eq(&self, other: &Self) -> bool {
        match (self, other) {
            (Self::String(a), Self::String(b)) | (Self::Uri(a), Self::Uri(b)) => a == b,
            (Self::Boolean(a), Self::Boolean(b)) => a == b,
            (Self::Decimal(a), Self::Decimal(b)) => a == b,
            // Cross-family numeric comparison never happens: a restriction can
            // only narrow within one primitive.
            (Self::Float(a), Self::Float(b)) => a == b || (a.is_nan() && b.is_nan()),
            (Self::Double(a), Self::Double(b)) => a == b || (a.is_nan() && b.is_nan()),
            (Self::Duration(a), Self::Duration(b)) => a == b,
            (Self::YearMonthDuration(a), Self::YearMonthDuration(b)) => a == b,
            (Self::DayTimeDuration(a), Self::DayTimeDuration(b)) => a == b,
            (Self::DateTime(a), Self::DateTime(b)) => a == b,
            (Self::Time(a), Self::Time(b)) => a == b,
            (Self::Date(a), Self::Date(b)) => a == b,
            (Self::GYearMonth(a), Self::GYearMonth(b)) => a == b,
            (Self::GYear(a), Self::GYear(b)) => a == b,
            (Self::GMonthDay(a), Self::GMonthDay(b)) => a == b,
            (Self::GMonth(a), Self::GMonth(b)) => a == b,
            (Self::GDay(a), Self::GDay(b)) => a == b,
            (Self::Binary(a), Self::Binary(b)) => a == b,
            (Self::QName(na, la), Self::QName(nb, lb)) => na == nb && la == lb,
            (Self::List(a), Self::List(b)) => a == b,
            _ => false,
        }
    }
}

impl Value {
    /// The specification's partial order, or [`None`] when the two values are
    /// incomparable (which includes values of unordered datatypes).
    #[must_use]
    pub fn partial_cmp_value(&self, other: &Self) -> Option<Ordering> {
        match (self, other) {
            (Self::Decimal(a), Self::Decimal(b)) => Some(a.cmp(b)),
            (Self::Float(a), Self::Float(b)) => a.partial_cmp(b),
            (Self::Double(a), Self::Double(b)) => a.partial_cmp(b),
            (Self::Duration(a), Self::Duration(b)) => a.partial_cmp(b),
            (Self::YearMonthDuration(a), Self::YearMonthDuration(b)) => a.partial_cmp(b),
            (Self::DayTimeDuration(a), Self::DayTimeDuration(b)) => a.partial_cmp(b),
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

    /// What `length`, `minLength` and `maxLength` measure for this value:
    /// characters for strings, octets for binary, items for lists.
    #[must_use]
    pub fn measure(&self) -> Option<usize> {
        match self {
            Self::String(text) | Self::Uri(text) => Some(text.chars().count()),
            Self::Binary(bytes) => Some(bytes.len()),
            Self::List(items) => Some(items.len()),
            _ => None,
        }
    }

    /// Whether the value carries an explicit timezone offset (date/time values
    /// only).
    #[must_use]
    pub fn has_timezone(&self) -> Option<bool> {
        Some(match self {
            Self::DateTime(v) => v.timezone_offset().is_some(),
            Self::Time(v) => v.timezone_offset().is_some(),
            Self::Date(v) => v.timezone_offset().is_some(),
            Self::GYearMonth(v) => v.timezone_offset().is_some(),
            Self::GYear(v) => v.timezone_offset().is_some(),
            Self::GMonthDay(v) => v.timezone_offset().is_some(),
            Self::GMonth(v) => v.timezone_offset().is_some(),
            Self::GDay(v) => v.timezone_offset().is_some(),
            _ => return None,
        })
    }

    /// The decimal behind the value, when it has one.
    #[must_use]
    pub fn as_decimal(&self) -> Option<&BigDecimal> {
        match self {
            Self::Decimal(decimal) => Some(decimal),
            _ => None,
        }
    }
}

/// Maps `lexical` (already whitespace-normalized) onto a value of `primitive`,
/// under `version`'s reading of the lexical spaces.
///
/// Returns [`None`] when the string is not in the lexical space.
///
/// Three lexical spaces were widened between the two versions, and each is
/// pinned by a test group whose `<expected>` is split by version:
///
/// * `xs:anyURI` — see [`is_any_uri`];
/// * `float`/`double` — see [`parse_binary_float`];
/// * the year `0000` — see [`year_is_in_the_value_space`].
#[must_use]
pub fn parse_primitive(
    primitive: Primitive,
    lexical: &str,
    namespaces: &NamespaceContext,
    version: XsdVersion,
) -> Option<Value> {
    match primitive {
        Primitive::AnySimpleType | Primitive::AnyAtomicType | Primitive::String => {
            Some(Value::String(lexical.to_owned()))
        }
        Primitive::Boolean => match lexical {
            "true" | "1" => Some(Value::Boolean(true)),
            "false" | "0" => Some(Value::Boolean(false)),
            _ => None,
        },
        Primitive::Decimal => BigDecimal::parse(lexical).map(Value::Decimal),
        Primitive::Float => parse_binary_float(lexical, version).map(|f| Value::Float(f as f32)),
        Primitive::Double => parse_binary_float(lexical, version).map(Value::Double),
        Primitive::Duration => Duration::from_str(lexical).ok().map(Value::Duration),
        Primitive::DateTime => DateTime::from_str(lexical)
            .ok()
            .filter(|value| year_is_in_the_value_space(value.year(), version))
            .map(Value::DateTime),
        Primitive::Time => Time::from_str(lexical).ok().map(Value::Time),
        Primitive::Date => Date::from_str(lexical)
            .ok()
            .filter(|value| year_is_in_the_value_space(value.year(), version))
            .map(Value::Date),
        Primitive::GYearMonth => GYearMonth::from_str(lexical)
            .ok()
            .filter(|value| year_is_in_the_value_space(value.year(), version))
            .map(Value::GYearMonth),
        Primitive::GYear => GYear::from_str(lexical)
            .ok()
            .filter(|value| year_is_in_the_value_space(value.year(), version))
            .map(Value::GYear),
        Primitive::GMonthDay => GMonthDay::from_str(lexical).ok().map(Value::GMonthDay),
        Primitive::GMonth => GMonth::from_str(lexical).ok().map(Value::GMonth),
        Primitive::GDay => GDay::from_str(lexical).ok().map(Value::GDay),
        Primitive::HexBinary => parse_hex_binary(lexical).map(Value::Binary),
        Primitive::Base64Binary => parse_base64(lexical).map(Value::Binary),
        Primitive::AnyUri => is_any_uri(lexical, version).then(|| Value::Uri(lexical.to_owned())),
        Primitive::QName | Primitive::Notation => parse_qname(lexical, namespaces),
    }
}

/// Whether a year field may hold `year` under `version`.
///
/// [XML Schema 1.0 Part 2 §3.2.7.1](https://www.w3.org/TR/xmlschema-2/#dateTime-lexical-representation)
/// ends its account of the lexical representation with "the year 0000 is
/// prohibited", and says the same of `date`, `gYearMonth` and `gYear`; the
/// prohibition is a property of the year field, so it holds wherever one
/// appears. [XML Schema 1.1 Part 2 §D.3.2](https://www.w3.org/TR/xmlschema11-2/#dt-yearFrag)
/// dropped it and gave year 0 its proleptic Gregorian meaning (1 BCE), which is
/// the reading `oxixml_xsd` implements — so the older prohibition has to be
/// applied here, where the version is known, rather than in the datatypes.
/// `msData/datatypes/dateTime011` states exactly that split: `0000-01-01T00:00:00`
/// is *invalid* under 1.0 and *valid* under 1.1.
#[must_use]
pub const fn year_is_in_the_value_space(year: i64, version: XsdVersion) -> bool {
    version.is_1_1() || year != 0
}

/// The XSD lexical space of `float` and `double`: a decimal mantissa with an
/// optional exponent, or one of `INF`, `-INF` and `NaN` — plus `+INF` from XSD
/// 1.1 on.
///
/// [XML Schema 1.0 Part 2 §3.2.4.1](https://www.w3.org/TR/xmlschema-2/#float)
/// spells the special values `INF`, `-INF` and `NaN` and gives no `+INF`;
/// [XML Schema 1.1 Part 2 §3.3.5.2](https://www.w3.org/TR/xmlschema11-2/#f-floatLexmap)
/// adds it. `msData/datatypes/float018` and `double018` are the `+INF`
/// instances, *invalid* under 1.0 and *valid* under 1.1.
#[must_use]
pub fn parse_binary_float(value: &str, version: XsdVersion) -> Option<f64> {
    match value {
        "INF" => return Some(f64::INFINITY),
        "+INF" if version.is_1_1() => return Some(f64::INFINITY),
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
    // The lexical space is `((B64S B64S B64S B64S)* ((B64S B64S B64S B64) |
    //  (B64S B64S B16S '=') | (B64S B04S '=' #x20? '=')))?`, so the final quad
    // constrains its third or second character to a subset of the alphabet.
    let cleaned: Vec<u8> = value.bytes().filter(|b| *b != b' ').collect();
    if cleaned.is_empty() {
        return Some(Vec::new());
    }
    if cleaned.len() % 4 != 0 {
        return None;
    }
    let mut out = Vec::with_capacity(cleaned.len() / 4 * 3);
    let quads = cleaned.len() / 4;
    for (index, quad) in cleaned.chunks_exact(4).enumerate() {
        let last_quad = index + 1 == quads;
        let padding = quad.iter().filter(|b| **b == b'=').count();
        if padding > 0 && !last_quad {
            return None;
        }
        match padding {
            0 => {}
            1 => {
                if quad[3] != b'=' || !is_base64_16(quad[2]) {
                    return None;
                }
            }
            2 => {
                if quad[2] != b'=' || quad[3] != b'=' || !is_base64_04(quad[1]) {
                    return None;
                }
            }
            _ => return None,
        }
        let mut accumulator = 0_u32;
        for (position, byte) in quad.iter().enumerate() {
            let sextet = if *byte == b'=' {
                if position < 4 - padding {
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

/// `B16` — `[AEIMQUYcgkosw048]`, the sixteen characters that may precede a
/// single `=`: their two low bits, which the mapping discards, are zero.
fn is_base64_16(byte: u8) -> bool {
    matches!(base64_value(byte), Some(value) if value % 4 == 0)
}

/// `B04` — `[AQgw]`, the four characters that may precede `==`: their four low
/// bits, which the mapping discards, are zero.
fn is_base64_04(byte: u8) -> bool {
    matches!(base64_value(byte), Some(value) if value % 16 == 0)
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

/// Parses a `QName` or `NOTATION` lexical form in `namespaces`.
#[must_use]
pub fn parse_qname(value: &str, namespaces: &NamespaceContext) -> Option<Value> {
    match value.split_once(':') {
        Some((prefix, local)) => {
            if !is_ncname(prefix) || !is_ncname(local) {
                return None;
            }
            let namespace = namespaces.lookup(Some(prefix))?;
            Some(Value::QName(Some(namespace.to_owned()), local.to_owned()))
        }
        None => {
            if !is_ncname(value) {
                return None;
            }
            Some(Value::QName(
                namespaces.lookup(None).map(str::to_owned),
                value.to_owned(),
            ))
        }
    }
}

/// The `anyURI` lexical space of `version`.
///
/// [XML Schema 1.1 Part 2 §3.3.17](https://www.w3.org/TR/xmlschema11-2/#anyURI)
/// makes the value space of `xs:anyURI` "the set of finite-length sequences of
/// zero or more characters that match the `Char` production from `XML`", and
/// the lexical mapping the identity: under 1.1 **every** string is an `anyURI`.
/// The test suite states the change on four schema documents whose `<expected>`
/// is split by version — `anyURI_a001`, `anyURI_a003`, `anyURI_b004` and
/// `anyURI_b006` are all *invalid* under 1.0 and *valid* under 1.1 — and
/// annotates them "In XSD 1.1, any sequence of characters is allowed in
/// xs:anyURI".
///
/// [XML Schema 1.0 Part 2 §3.2.17](https://www.w3.org/TR/xmlschema-2/#anyURI)
/// instead maps the strings that become a legal URI reference once the
/// characters a URI may not carry have been percent-escaped, which is what
/// Xerces does before parsing. So the strings rejected under 1.0 are those that
/// escaping cannot repair: a malformed percent-escape (which escaping leaves
/// alone, `anyURI_b004`'s bare `%`), a second `#`, a relative reference whose
/// first segment contains a colon — which would be read as a scheme, and is not
/// a legal one (`anyURI_a003`'s `:a`) — and the *reverse solidus*, to which
/// [RFC 3986] gives no meaning at all and which no processor introduces by
/// escaping, so `\\` and `//a\b` are not URI references (`anyURI_b006`).
///
/// [RFC 3986]: https://www.rfc-editor.org/rfc/rfc3986#section-2
#[must_use]
pub fn is_any_uri(value: &str, version: XsdVersion) -> bool {
    if version.is_1_1() {
        return true;
    }
    if value.contains('\\') {
        return false;
    }
    let bytes = value.as_bytes();
    let mut fragments = 0_u32;
    let mut index = 0;
    while index < bytes.len() {
        match bytes[index] {
            b'#' => {
                fragments += 1;
                if fragments > 1 {
                    return false;
                }
            }
            b'%' => {
                if index + 2 >= bytes.len()
                    || !bytes[index + 1].is_ascii_hexdigit()
                    || !bytes[index + 2].is_ascii_hexdigit()
                {
                    return false;
                }
                index += 2;
            }
            _ => {}
        }
        index += 1;
    }
    if let Some(colon) = value.find(':') {
        let scheme = &value[..colon];
        let looks_absolute = !scheme.contains(['/', '?', '#']);
        if looks_absolute
            && (scheme.is_empty()
                || !scheme.starts_with(|c: char| c.is_ascii_alphabetic())
                || !scheme
                    .bytes()
                    .all(|b| b.is_ascii_alphanumeric() || matches!(b, b'+' | b'-' | b'.')))
        {
            return false;
        }
    }
    true
}

/// `[a-zA-Z]{1,8}(-[a-zA-Z0-9]{1,8})*`, the `xs:language` pattern.
#[must_use]
pub fn is_language(value: &str) -> bool {
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

/// Re-exported name predicates, so that callers of this module do not have to
/// depend on `oxixml-qname` directly.
#[must_use]
pub fn is_xml_name(value: &str) -> bool {
    is_name(value)
}

/// `NCName`.
#[must_use]
pub fn is_xml_ncname(value: &str) -> bool {
    is_ncname(value)
}

/// `NMTOKEN`.
#[must_use]
pub fn is_xml_nmtoken(value: &str) -> bool {
    is_nmtoken(value)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn value(primitive: Primitive, lexical: &str) -> Option<Value> {
        parse_primitive(
            primitive,
            lexical,
            &NamespaceContext::new(),
            XsdVersion::V1_0,
        )
    }

    fn value_1_1(primitive: Primitive, lexical: &str) -> Option<Value> {
        parse_primitive(
            primitive,
            lexical,
            &NamespaceContext::new(),
            XsdVersion::V1_1,
        )
    }

    #[test]
    fn whitespace_normalizations() {
        assert_eq!(WhiteSpace::Preserve.apply(" a\tb "), " a\tb ");
        assert_eq!(WhiteSpace::Replace.apply(" a\tb "), " a b ");
        assert_eq!(WhiteSpace::Collapse.apply(" a\tb "), "a b");
    }

    #[test]
    fn binary_round_trips() {
        assert_eq!(
            value(Primitive::HexBinary, "0FB7"),
            Some(Value::Binary(vec![0x0F, 0xB7]))
        );
        assert!(value(Primitive::HexBinary, "0FB").is_none());
        assert_eq!(
            value(Primitive::Base64Binary, "YW55"),
            Some(Value::Binary(b"any".to_vec()))
        );
        assert!(value(Primitive::Base64Binary, "YW5=").is_none());
        assert!(value(Primitive::Base64Binary, "YW4=").is_some());
        assert!(value(Primitive::Base64Binary, "YQ==").is_some());
        assert!(value(Primitive::Base64Binary, "YR==").is_none());
    }

    #[test]
    fn floats_follow_the_lexical_space() {
        assert!(value(Primitive::Double, "1.0E-3").is_some());
        assert!(value(Primitive::Double, "INF").is_some());
        assert!(value(Primitive::Double, "infinity").is_none());
        assert!(value(Primitive::Double, "1e").is_none());
    }

    #[test]
    fn any_uri_rejects_only_the_irreparable() {
        assert!(is_any_uri("http://example.com/a b", XsdVersion::V1_0));
        assert!(is_any_uri("../relative#frag", XsdVersion::V1_0));
        assert!(is_any_uri("urn:x:{brace}", XsdVersion::V1_0));
        assert!(!is_any_uri("http://example.com/%zz", XsdVersion::V1_0));
        assert!(!is_any_uri("a#b#c", XsdVersion::V1_0));
        assert!(!is_any_uri("1http://x", XsdVersion::V1_0));
    }

    /// XSD 1.1 gives `xs:anyURI` the whole character space, so every string
    /// the 1.0 lexical space turns away is accepted. The values are the ones
    /// the version-split schema documents of
    /// `msData/datatypes/Facets/anyURI/` enumerate: `anyURI_a001`'s `appinfo`
    /// source, `anyURI_a003`'s `:a`, `anyURI_b004`'s bare `%` and `^`, and
    /// `anyURI_b006`'s reverse-solidus forms.
    #[test]
    fn any_uri_accepts_every_string_under_one_one() {
        const SPLIT: &[&str] = &[
            "9999999999999999999999999999999999999999999999999999999999999anyURI:",
            "1111111111111111111http://foo/bar",
            ":a",
            "%",
            "^",
            "\\",
            "\\\\",
            "//a\\b",
            "..\\..\\a",
            "a#b#c",
        ];
        for lexical in SPLIT {
            assert!(
                is_any_uri(lexical, XsdVersion::V1_1),
                "`{lexical}` is an `xs:anyURI` under XSD 1.1"
            );
            assert_eq!(
                value_1_1(Primitive::AnyUri, lexical),
                Some(Value::Uri((*lexical).to_owned()))
            );
        }
        // All but `^` are outside the 1.0 lexical space. `^` is one of the
        // *unwise* characters of RFC 2396, which percent-escaping repairs, so
        // 1.0 accepts it too: what makes `anyURI_b004`'s schema document
        // invalid under 1.0 is the bare `%` beside it, not the `^`.
        for lexical in SPLIT {
            let expected = *lexical == "^";
            assert_eq!(
                is_any_uri(lexical, XsdVersion::V1_0),
                expected,
                "`{lexical}` under XSD 1.0"
            );
            assert_eq!(value(Primitive::AnyUri, lexical).is_some(), expected);
        }
    }

    /// The one rule XSD 1.0 gains rather than loses: the reverse solidus is in
    /// no URI reference and percent-escaping does not put it in one, so the
    /// `anyURI_b006` enumeration is outside the 1.0 lexical space while the
    /// solidus-only forms beside it stay inside.
    #[test]
    fn any_uri_rejects_the_reverse_solidus_under_one_zero() {
        for lexical in ["/", "//", "///", "/a/", "//a//a"] {
            assert!(
                is_any_uri(lexical, XsdVersion::V1_0),
                "`{lexical}` is an ordinary relative reference"
            );
        }
        for lexical in [
            "\\",
            "\\\\",
            "\\\\\\",
            "\\\\/",
            "\\a\\b",
            "/a\\b",
            "..\\..\\a",
        ] {
            assert!(
                !is_any_uri(lexical, XsdVersion::V1_0),
                "`{lexical}` carries a reverse solidus"
            );
        }
    }

    /// `+INF` joined the `float`/`double` lexical space in XSD 1.1
    /// (`msData/datatypes/float018`, `double018`); `INF`, `-INF` and `NaN`
    /// were always in it.
    #[test]
    fn plus_inf_is_an_xsd_1_1_spelling() {
        assert_eq!(
            value_1_1(Primitive::Double, "+INF"),
            Some(Value::Double(f64::INFINITY))
        );
        assert_eq!(
            value_1_1(Primitive::Float, "+INF"),
            Some(Value::Float(f32::INFINITY))
        );
        assert!(value(Primitive::Double, "+INF").is_none());
        assert!(value(Primitive::Float, "+INF").is_none());
        for version in [XsdVersion::V1_0, XsdVersion::V1_1] {
            assert_eq!(parse_binary_float("INF", version), Some(f64::INFINITY));
            assert_eq!(parse_binary_float("-INF", version), Some(f64::NEG_INFINITY));
            assert!(
                parse_binary_float("NaN", version).is_some_and(f64::is_nan),
                "NaN is spelled the same in both versions"
            );
            assert!(parse_binary_float("+inf", version).is_none());
            assert!(parse_binary_float("Infinity", version).is_none());
        }
    }

    /// XSD 1.0 prohibits the year `0000` outright; XSD 1.1 reads it as 1 BCE.
    /// `msData/datatypes/dateTime011` states the split for `dateTime`, and the
    /// prohibition belongs to the year field, so `date`, `gYearMonth` and
    /// `gYear` follow it.
    #[test]
    fn year_zero_is_an_xsd_1_1_value() {
        const ZERO: &[(Primitive, &str)] = &[
            (Primitive::DateTime, "0000-01-01T00:00:00"),
            (Primitive::Date, "0000-02-29"),
            (Primitive::GYearMonth, "0000-01"),
            (Primitive::GYear, "0000"),
        ];
        for (primitive, lexical) in ZERO {
            assert!(
                value(*primitive, lexical).is_none(),
                "`{lexical}` is prohibited under XSD 1.0"
            );
            assert!(
                value_1_1(*primitive, lexical).is_some(),
                "`{lexical}` is 1 BCE under XSD 1.1"
            );
        }
        // Year 1 and year -1 are unaffected by the prohibition; XSD 1.0's own
        // `-0001` denotes 1 BCE, since it has no year 0 to shift the count.
        for version in [XsdVersion::V1_0, XsdVersion::V1_1] {
            for lexical in ["0001-01-01T00:00:00", "-0001-01-01T00:00:00"] {
                assert!(
                    parse_primitive(
                        Primitive::DateTime,
                        lexical,
                        &NamespaceContext::new(),
                        version
                    )
                    .is_some(),
                    "`{lexical}` is a `dateTime` under XSD {version}"
                );
            }
        }
    }

    #[test]
    fn qname_needs_a_binding() {
        let mut namespaces = NamespaceContext::new();
        assert!(parse_qname("p:a", &namespaces).is_none());
        namespaces.declare(Some("p"), "urn:x");
        assert_eq!(
            parse_qname("p:a", &namespaces),
            Some(Value::QName(Some("urn:x".to_owned()), "a".to_owned()))
        );
    }

    #[test]
    fn decimal_equality_is_in_the_value_space() {
        assert_eq!(
            value(Primitive::Decimal, "1.0"),
            value(Primitive::Decimal, "1.00")
        );
    }
}
