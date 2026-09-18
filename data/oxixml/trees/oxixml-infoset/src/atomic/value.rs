//! [`AtomicValue`] and its payload enum [`AtomicData`].

use super::bignum::{BigDecimal, BigInteger};
use super::binary::{encode_base64, encode_hex};
use super::qname_value::QNameValue;
use super::types::AtomicType;
use oxixml_xsd::{
    Boolean, Date, DateTime, DayTimeDuration, Decimal, Double, Duration, Float, GDay, GMonth,
    GMonthDay, GYear, GYearMonth, Integer, Time, YearMonthDuration,
};
use std::fmt;
use std::hash::{Hash, Hasher};

/// The payload of an [`AtomicValue`]: one value drawn from the value space of one of the XML
/// Schema primitive datatypes.
///
/// There is exactly one variant per *primitive* value space, not per type name — the type name
/// itself lives beside the payload in [`AtomicValue::atomic_type`]. `xs:byte`, `xs:integer` and
/// `xs:unsignedLong` therefore all carry [`AtomicData::Integer`], and `xs:string`, `xs:ID`,
/// `xs:anyURI` and `xs:untypedAtomic` all carry [`AtomicData::Text`].
///
/// # Equality
///
/// [`PartialEq`] compares *within* a primitive value space only — payloads of different
/// variants are never equal, so `xs:double 1.0E0` does not equal `xs:integer 1` here. Within a
/// variant it delegates to the corresponding `oxixml-xsd` type's own equality, except for
/// `xs:float` and `xs:double`, where XML Schema
/// [*identity*](https://www.w3.org/TR/xmlschema11-2/#identity) is used instead so that this
/// type can be [`Eq`] and [`Hash`]: `NaN` equals a bit-identical `NaN`, and `0.0E0` does not
/// equal `-0.0E0`.
///
/// Use [`atomic_equal`](crate::atomic_equal) for XPath's cross-type `eq` semantics and
/// [`same_key`](crate::same_key) for map-key semantics.
#[derive(Debug, Clone)]
#[non_exhaustive]
pub enum AtomicData {
    /// The value space shared by `xs:string` and everything derived from it, plus `xs:anyURI`
    /// and `xs:untypedAtomic`: a sequence of Unicode characters.
    Text(String),
    /// `xs:boolean`.
    Boolean(Boolean),
    /// `xs:decimal`, but *not* its `xs:integer` restriction (which uses
    /// [`AtomicData::Integer`]).
    Decimal(Decimal),
    /// `xs:integer` and every type derived from it.
    Integer(Integer),
    /// An `xs:integer` (or a type derived from it) whose value does not fit an `i64`.
    ///
    /// This variant only ever holds a value the fixed-width [`AtomicData::Integer`] cannot: it is
    /// reachable exclusively through [`AtomicData::from_big_integer`], which demotes on fit, so a
    /// value representable in both widths is always the fixed one. That invariant is what keeps
    /// the discriminant-sensitive [`Hash`] consistent with the cross-width numeric equality the
    /// map-key relation needs. It cannot be constructed from outside this crate.
    #[non_exhaustive]
    BigInteger(BigInteger),
    /// An `xs:decimal` whose value does not fit the fixed-width [`AtomicData::Decimal`] — it needs
    /// more than eighteen fractional digits or a magnitude beyond `i128`.
    ///
    /// Like [`AtomicData::BigInteger`], it is reachable only through the demoting
    /// [`AtomicData::from_big_decimal`] and never holds a fixed-width-representable value.
    #[non_exhaustive]
    BigDecimal(BigDecimal),
    /// `xs:double`.
    Double(Double),
    /// `xs:float`.
    Float(Float),
    /// `xs:duration`.
    Duration(Duration),
    /// `xs:yearMonthDuration`.
    YearMonthDuration(YearMonthDuration),
    /// `xs:dayTimeDuration`.
    DayTimeDuration(DayTimeDuration),
    /// `xs:dateTime` and `xs:dateTimeStamp`.
    DateTime(DateTime),
    /// `xs:date`.
    Date(Date),
    /// `xs:time`.
    Time(Time),
    /// `xs:gYearMonth`.
    GYearMonth(GYearMonth),
    /// `xs:gYear`.
    GYear(GYear),
    /// `xs:gMonthDay`.
    GMonthDay(GMonthDay),
    /// `xs:gMonth`.
    GMonth(GMonth),
    /// `xs:gDay`.
    GDay(GDay),
    /// `xs:hexBinary` and `xs:base64Binary`: a sequence of octets. The two types differ only in
    /// their lexical mapping, never in their value space.
    Binary(Vec<u8>),
    /// `xs:QName` and `xs:NOTATION`.
    QName(QNameValue),
}

impl AtomicData {
    /// Builds an `xs:integer` payload from a [`BigInteger`], demoting to the fixed-width
    /// [`AtomicData::Integer`] whenever the value fits an `i64`.
    ///
    /// This is the *only* way a [`AtomicData::BigInteger`] is ever produced. The demotion keeps
    /// the invariant that a value representable in both widths always takes the fixed form, which
    /// is what lets the discriminant-first [`Hash`] agree with the cross-width numeric equality
    /// the map-key relation compares by.
    #[must_use]
    pub fn from_big_integer(value: BigInteger) -> Self {
        match value.to_i64() {
            Some(small) => Self::Integer(Integer::from(small)),
            None => {
                debug_assert!(
                    value.to_i64().is_none(),
                    "a BigInteger payload must not hold an i64-representable value"
                );
                Self::BigInteger(value)
            }
        }
    }

    /// Builds an `xs:decimal` payload from a [`BigDecimal`], demoting to the fixed-width
    /// [`AtomicData::Decimal`] whenever the value fits it (at most eighteen fractional digits and
    /// an `i128` significand). This is the only way a [`AtomicData::BigDecimal`] is produced; see
    /// [`AtomicData::from_big_integer`] for why the demotion matters.
    #[must_use]
    pub fn from_big_decimal(value: BigDecimal) -> Self {
        match value.to_fixed() {
            Some(small) => Self::Decimal(small),
            None => {
                debug_assert!(
                    value.to_fixed().is_none(),
                    "a BigDecimal payload must not hold a fixed-width-representable value"
                );
                Self::BigDecimal(value)
            }
        }
    }

    /// The primitive type whose value space this payload belongs to.
    ///
    /// [`AtomicData::Binary`] answers `xs:hexBinary` and [`AtomicData::DateTime`] answers
    /// `xs:dateTime`; the distinctions those types' restrictions draw are carried by
    /// [`AtomicValue::atomic_type`], not by the payload.
    #[must_use]
    pub const fn nominal_type(&self) -> AtomicType {
        match self {
            Self::Text(_) => AtomicType::String,
            Self::Boolean(_) => AtomicType::Boolean,
            Self::Decimal(_) => AtomicType::Decimal,
            Self::Integer(_) => AtomicType::Integer,
            Self::BigInteger(_) => AtomicType::Integer,
            Self::BigDecimal(_) => AtomicType::Decimal,
            Self::Double(_) => AtomicType::Double,
            Self::Float(_) => AtomicType::Float,
            Self::Duration(_) => AtomicType::Duration,
            Self::YearMonthDuration(_) => AtomicType::YearMonthDuration,
            Self::DayTimeDuration(_) => AtomicType::DayTimeDuration,
            Self::DateTime(_) => AtomicType::DateTime,
            Self::Date(_) => AtomicType::Date,
            Self::Time(_) => AtomicType::Time,
            Self::GYearMonth(_) => AtomicType::GYearMonth,
            Self::GYear(_) => AtomicType::GYear,
            Self::GMonthDay(_) => AtomicType::GMonthDay,
            Self::GMonth(_) => AtomicType::GMonth,
            Self::GDay(_) => AtomicType::GDay,
            Self::Binary(_) => AtomicType::HexBinary,
            Self::QName(_) => AtomicType::QName,
        }
    }

    /// Whether this payload is a member of the numeric tower.
    #[must_use]
    pub const fn is_numeric(&self) -> bool {
        matches!(
            self,
            Self::Decimal(_)
                | Self::Integer(_)
                | Self::BigInteger(_)
                | Self::BigDecimal(_)
                | Self::Double(_)
                | Self::Float(_)
        )
    }

    /// The value as an [`f64`], for numeric payloads.
    ///
    /// The conversion is the one XPath's numeric promotion performs, and is lossy for
    /// `xs:decimal` and `xs:integer` values that exceed 53 bits of significand.
    #[must_use]
    pub fn as_f64(&self) -> Option<f64> {
        match self {
            Self::Decimal(value) => Some(f64::from(Double::from(*value))),
            Self::Integer(value) => Some(f64::from(Double::from(*value))),
            Self::BigInteger(value) => Some(value.to_f64()),
            Self::BigDecimal(value) => Some(value.to_f64()),
            Self::Double(value) => Some(f64::from(*value)),
            Self::Float(value) => Some(f64::from(*value)),
            _ => None,
        }
    }

    /// The arbitrary-precision integer this payload holds, if it is one that overflowed the fixed
    /// width. A fixed-width [`AtomicData::Integer`] answers [`None`] — read it through
    /// [`AtomicData::Integer`] directly — so a consumer that wants *any* integer must check both.
    ///
    /// This is the read half of the pair whose write half is the demoting
    /// [`AtomicData::from_big_integer`]; together they let a consumer work with big values without
    /// being able to construct a non-canonical one.
    #[must_use]
    pub fn as_big_integer(&self) -> Option<&BigInteger> {
        match self {
            Self::BigInteger(value) => Some(value),
            _ => None,
        }
    }

    /// The arbitrary-precision decimal this payload holds, if it is one that overflowed the fixed
    /// width; see [`AtomicData::as_big_integer`].
    #[must_use]
    pub fn as_big_decimal(&self) -> Option<&BigDecimal> {
        match self {
            Self::BigDecimal(value) => Some(value),
            _ => None,
        }
    }

    /// Whether this payload is a numeric `NaN` (only `xs:float` and `xs:double` can be).
    #[must_use]
    pub fn is_nan(&self) -> bool {
        match self {
            Self::Double(value) => value.is_nan(),
            Self::Float(value) => value.is_nan(),
            _ => false,
        }
    }
}

impl PartialEq for AtomicData {
    /// XML Schema *identity*: see the type-level documentation.
    fn eq(&self, other: &Self) -> bool {
        match (self, other) {
            (Self::Text(a), Self::Text(b)) => a == b,
            (Self::Boolean(a), Self::Boolean(b)) => a == b,
            (Self::Decimal(a), Self::Decimal(b)) => a == b,
            (Self::Integer(a), Self::Integer(b)) => a == b,
            // A `BigInteger`/`BigDecimal` never holds a fixed-width-representable value, so it is
            // only ever equal to another of the same variant — a `BigInteger` and an `Integer`
            // are, by construction, never the same value and correctly fall to the `_` arm.
            (Self::BigInteger(a), Self::BigInteger(b)) => a == b,
            (Self::BigDecimal(a), Self::BigDecimal(b)) => a == b,
            (Self::Double(a), Self::Double(b)) => a.is_identical_with(*b),
            (Self::Float(a), Self::Float(b)) => a.is_identical_with(*b),
            (Self::Duration(a), Self::Duration(b)) => a == b,
            (Self::YearMonthDuration(a), Self::YearMonthDuration(b)) => a == b,
            (Self::DayTimeDuration(a), Self::DayTimeDuration(b)) => a == b,
            (Self::DateTime(a), Self::DateTime(b)) => a == b,
            (Self::Date(a), Self::Date(b)) => a == b,
            (Self::Time(a), Self::Time(b)) => a == b,
            (Self::GYearMonth(a), Self::GYearMonth(b)) => a == b,
            (Self::GYear(a), Self::GYear(b)) => a == b,
            (Self::GMonthDay(a), Self::GMonthDay(b)) => a == b,
            (Self::GMonth(a), Self::GMonth(b)) => a == b,
            (Self::GDay(a), Self::GDay(b)) => a == b,
            (Self::Binary(a), Self::Binary(b)) => a == b,
            (Self::QName(a), Self::QName(b)) => a == b,
            _ => false,
        }
    }
}

impl Eq for AtomicData {}

impl Hash for AtomicData {
    fn hash<H: Hasher>(&self, state: &mut H) {
        std::mem::discriminant(self).hash(state);
        match self {
            Self::Text(value) => value.hash(state),
            Self::Boolean(value) => value.hash(state),
            Self::Decimal(value) => value.hash(state),
            Self::Integer(value) => value.hash(state),
            Self::BigInteger(value) => value.hash(state),
            Self::BigDecimal(value) => value.hash(state),
            // Hash the raw IEEE-754 bits so that `Hash` agrees with the identity-based
            // `PartialEq` above (which distinguishes `0.0` from `-0.0` and treats `NaN` as
            // equal to a bit-identical `NaN`).
            Self::Double(value) => f64::from(*value).to_bits().hash(state),
            Self::Float(value) => f32::from(*value).to_bits().hash(state),
            Self::Duration(value) => value.hash(state),
            Self::YearMonthDuration(value) => value.hash(state),
            Self::DayTimeDuration(value) => value.hash(state),
            Self::DateTime(value) => value.hash(state),
            Self::Date(value) => value.hash(state),
            Self::Time(value) => value.hash(state),
            Self::GYearMonth(value) => value.hash(state),
            Self::GYear(value) => value.hash(state),
            Self::GMonthDay(value) => value.hash(state),
            Self::GMonth(value) => value.hash(state),
            Self::GDay(value) => value.hash(state),
            Self::Binary(value) => value.hash(state),
            Self::QName(value) => value.hash(state),
        }
    }
}

/// An XDM atomic value: a payload from some XML Schema value space, labelled with the atomic
/// type it was validated against, plus the language and collation carried alongside it.
///
/// # The type label
///
/// XDM keeps a value and its type annotation together: `xs:integer 1` and `xs:short 1` have the
/// same payload but are distinguishable by `instance of`, and serialize identically. The label
/// is always consistent with the payload — [`AtomicValue::new`] rejects, for example, an
/// `xs:date` label over an [`AtomicData::Integer`] payload.
///
/// # Language and collation carry
///
/// Two pieces of context travel with a value rather than being part of it:
///
/// * the **language** (`xml:lang`) in force where a string was atomized, which `fn:lang` and
///   language-sensitive collations consult;
/// * the **collation** URI under which the value should be compared, which
///   [`atomic_equal_with_collation`](crate::atomic_equal_with_collation) and the ordering
///   helpers honour.
///
/// Neither participates in [`PartialEq`], [`Eq`] or [`Hash`]: two values with the same type and
/// the same payload are the same value however they were obtained. Use
/// [`AtomicValue::with_language`] and [`AtomicValue::with_collation`] to attach them.
///
/// # Examples
///
/// ```
/// use oxixml_infoset::{AtomicType, AtomicValue};
///
/// let integer = AtomicValue::integer(42);
/// assert_eq!(integer.atomic_type(), AtomicType::Integer);
/// assert_eq!(integer.string_value(), "42");
///
/// // The type label is preserved through derivation.
/// let short = AtomicValue::parse(AtomicType::Short, "42")?;
/// assert_eq!(short.atomic_type(), AtomicType::Short);
/// assert_eq!(short, integer.clone().cast_label(AtomicType::Short)?);
///
/// // Language and collation are carried, not compared.
/// let plain = AtomicValue::string("Straße");
/// let german = plain.clone().with_language("de");
/// assert_eq!(plain, german);
/// assert_eq!(german.language(), Some("de"));
/// # Ok::<(), oxixml_infoset::XdmError>(())
/// ```
#[derive(Debug, Clone)]
pub struct AtomicValue {
    atomic_type: AtomicType,
    data: AtomicData,
    language: Option<String>,
    collation: Option<String>,
}

impl AtomicValue {
    /// Builds a value with an explicit type label.
    ///
    /// # Errors
    ///
    /// Returns [`ErrorCode::Xpty0004`](crate::ErrorCode::Xpty0004) if `atomic_type` is not
    /// compatible with `data`'s value space — for instance an `xs:date` label over an
    /// [`AtomicData::Text`] payload. Range and lexical facets are *not* re-checked here; use
    /// [`AtomicValue::parse`] to validate a lexical form against a type.
    pub fn new(atomic_type: AtomicType, data: AtomicData) -> Result<Self, crate::XdmError> {
        if !label_matches_payload(atomic_type, &data) {
            return Err(crate::XdmError::new(
                crate::ErrorCode::Xpty0004,
                format!(
                    "the type label {atomic_type} is not compatible with a {} payload",
                    data.nominal_type()
                ),
            ));
        }
        Ok(Self {
            atomic_type,
            data,
            language: None,
            collation: None,
        })
    }

    pub(crate) fn new_unchecked(atomic_type: AtomicType, data: AtomicData) -> Self {
        debug_assert!(label_matches_payload(atomic_type, &data));
        Self {
            atomic_type,
            data,
            language: None,
            collation: None,
        }
    }

    /// An `xs:string` value.
    pub fn string(value: impl Into<String>) -> Self {
        Self::new_unchecked(AtomicType::String, AtomicData::Text(value.into()))
    }

    /// An `xs:untypedAtomic` value — the type every unvalidated node atomizes to.
    pub fn untyped_atomic(value: impl Into<String>) -> Self {
        Self::new_unchecked(AtomicType::UntypedAtomic, AtomicData::Text(value.into()))
    }

    /// An `xs:anyURI` value. The lexical form is *not* validated; use
    /// [`AtomicValue::parse`] for that.
    pub fn any_uri(value: impl Into<String>) -> Self {
        Self::new_unchecked(AtomicType::AnyUri, AtomicData::Text(value.into()))
    }

    /// An `xs:boolean` value.
    #[must_use]
    pub fn boolean(value: bool) -> Self {
        Self::new_unchecked(AtomicType::Boolean, AtomicData::Boolean(value.into()))
    }

    /// An `xs:integer` value.
    #[must_use]
    pub fn integer(value: impl Into<Integer>) -> Self {
        Self::new_unchecked(AtomicType::Integer, AtomicData::Integer(value.into()))
    }

    /// An `xs:decimal` value.
    #[must_use]
    pub fn decimal(value: impl Into<Decimal>) -> Self {
        Self::new_unchecked(AtomicType::Decimal, AtomicData::Decimal(value.into()))
    }

    /// An `xs:integer` value from an arbitrary-precision [`BigInteger`], demoting to the
    /// fixed-width payload when it fits an `i64`.
    #[must_use]
    pub fn from_big_integer(value: BigInteger) -> Self {
        Self::new_unchecked(AtomicType::Integer, AtomicData::from_big_integer(value))
    }

    /// An `xs:decimal` value from an arbitrary-precision [`BigDecimal`], demoting to the
    /// fixed-width payload when it fits.
    #[must_use]
    pub fn from_big_decimal(value: BigDecimal) -> Self {
        Self::new_unchecked(AtomicType::Decimal, AtomicData::from_big_decimal(value))
    }

    /// An `xs:double` value.
    #[must_use]
    pub fn double(value: impl Into<Double>) -> Self {
        Self::new_unchecked(AtomicType::Double, AtomicData::Double(value.into()))
    }

    /// An `xs:float` value.
    #[must_use]
    pub fn float(value: impl Into<Float>) -> Self {
        Self::new_unchecked(AtomicType::Float, AtomicData::Float(value.into()))
    }

    /// An `xs:hexBinary` value.
    #[must_use]
    pub fn hex_binary(value: impl Into<Vec<u8>>) -> Self {
        Self::new_unchecked(AtomicType::HexBinary, AtomicData::Binary(value.into()))
    }

    /// An `xs:base64Binary` value.
    #[must_use]
    pub fn base64_binary(value: impl Into<Vec<u8>>) -> Self {
        Self::new_unchecked(AtomicType::Base64Binary, AtomicData::Binary(value.into()))
    }

    /// An `xs:QName` value.
    #[must_use]
    pub fn qname(value: QNameValue) -> Self {
        Self::new_unchecked(AtomicType::QName, AtomicData::QName(value))
    }

    /// An `xs:dateTime` value.
    #[must_use]
    pub fn date_time(value: DateTime) -> Self {
        Self::new_unchecked(AtomicType::DateTime, AtomicData::DateTime(value))
    }

    /// An `xs:date` value.
    #[must_use]
    pub fn date(value: Date) -> Self {
        Self::new_unchecked(AtomicType::Date, AtomicData::Date(value))
    }

    /// An `xs:time` value.
    #[must_use]
    pub fn time(value: Time) -> Self {
        Self::new_unchecked(AtomicType::Time, AtomicData::Time(value))
    }

    /// An `xs:duration` value.
    #[must_use]
    pub fn duration(value: Duration) -> Self {
        Self::new_unchecked(AtomicType::Duration, AtomicData::Duration(value))
    }

    /// An `xs:yearMonthDuration` value.
    #[must_use]
    pub fn year_month_duration(value: YearMonthDuration) -> Self {
        Self::new_unchecked(
            AtomicType::YearMonthDuration,
            AtomicData::YearMonthDuration(value),
        )
    }

    /// An `xs:dayTimeDuration` value.
    #[must_use]
    pub fn day_time_duration(value: DayTimeDuration) -> Self {
        Self::new_unchecked(
            AtomicType::DayTimeDuration,
            AtomicData::DayTimeDuration(value),
        )
    }

    /// The type this value is labelled with.
    #[must_use]
    pub const fn atomic_type(&self) -> AtomicType {
        self.atomic_type
    }

    /// The value's payload.
    #[must_use]
    pub const fn data(&self) -> &AtomicData {
        &self.data
    }

    /// Consumes the value, yielding its payload.
    #[must_use]
    pub fn into_data(self) -> AtomicData {
        self.data
    }

    /// Re-labels this value with a type that shares its payload's value space.
    ///
    /// This models a *downcast within a primitive family* — `xs:integer` to `xs:short`,
    /// `xs:string` to `xs:token` — which the schema layer performs after validating the
    /// corresponding facets. It never converts between value spaces.
    ///
    /// # Errors
    ///
    /// [`ErrorCode::Xpty0004`](crate::ErrorCode::Xpty0004) if the new label does not fit the
    /// payload.
    pub fn cast_label(mut self, atomic_type: AtomicType) -> Result<Self, crate::XdmError> {
        if !label_matches_payload(atomic_type, &self.data) {
            return Err(crate::XdmError::new(
                crate::ErrorCode::Xpty0004,
                format!(
                    "the type label {atomic_type} is not compatible with a {} payload",
                    self.data.nominal_type()
                ),
            ));
        }
        self.atomic_type = atomic_type;
        Ok(self)
    }

    /// Attaches the language (`xml:lang`) in force where this value was produced.
    #[must_use]
    pub fn with_language(mut self, language: impl Into<String>) -> Self {
        self.language = Some(language.into());
        self
    }

    /// Attaches the collation URI under which this value should be compared.
    #[must_use]
    pub fn with_collation(mut self, collation: impl Into<String>) -> Self {
        self.collation = Some(collation.into());
        self
    }

    /// The carried language, if any.
    #[must_use]
    pub fn language(&self) -> Option<&str> {
        self.language.as_deref()
    }

    /// The carried collation URI, if any.
    #[must_use]
    pub fn collation(&self) -> Option<&str> {
        self.collation.as_deref()
    }

    /// Whether this value belongs to the numeric tower.
    #[must_use]
    pub const fn is_numeric(&self) -> bool {
        self.data.is_numeric()
    }

    /// Whether this value is a numeric `NaN`.
    #[must_use]
    pub fn is_nan(&self) -> bool {
        self.data.is_nan()
    }

    /// The value as an [`f64`], for numeric values.
    #[must_use]
    pub fn as_f64(&self) -> Option<f64> {
        self.data.as_f64()
    }

    /// The text payload, for `xs:string`-family, `xs:anyURI` and `xs:untypedAtomic` values.
    #[must_use]
    pub fn as_str(&self) -> Option<&str> {
        match &self.data {
            AtomicData::Text(value) => Some(value),
            _ => None,
        }
    }

    /// The boolean payload, for `xs:boolean` values.
    #[must_use]
    pub fn as_bool(&self) -> Option<bool> {
        match &self.data {
            AtomicData::Boolean(value) => Some(bool::from(*value)),
            _ => None,
        }
    }

    /// The `xs:QName` payload, for `xs:QName` and `xs:NOTATION` values.
    #[must_use]
    pub fn as_qname(&self) -> Option<&QNameValue> {
        match &self.data {
            AtomicData::QName(value) => Some(value),
            _ => None,
        }
    }

    /// The octet payload, for `xs:hexBinary` and `xs:base64Binary` values.
    #[must_use]
    pub fn as_bytes(&self) -> Option<&[u8]> {
        match &self.data {
            AtomicData::Binary(value) => Some(value),
            _ => None,
        }
    }

    /// The value's *string value*: the canonical lexical representation of the payload under
    /// this value's type label.
    ///
    /// This is what `fn:string` returns for an atomic item. Note that `xs:hexBinary` and
    /// `xs:base64Binary` share a payload but produce different string values, and that
    /// `xs:QName`'s string value uses the prefix retained on the value.
    #[must_use]
    pub fn string_value(&self) -> String {
        match &self.data {
            AtomicData::Text(value) => value.clone(),
            AtomicData::Boolean(value) => value.to_string(),
            AtomicData::Decimal(value) => value.to_string(),
            AtomicData::Integer(value) => value.to_string(),
            AtomicData::BigInteger(value) => value.to_string(),
            AtomicData::BigDecimal(value) => value.to_string(),
            AtomicData::Double(value) => xpath_float_string(f64::from(*value)),
            AtomicData::Float(value) => xpath_float32_string(f32::from(*value)),
            AtomicData::Duration(value) => value.to_string(),
            AtomicData::YearMonthDuration(value) => value.to_string(),
            AtomicData::DayTimeDuration(value) => value.to_string(),
            AtomicData::DateTime(value) => value.to_string(),
            AtomicData::Date(value) => value.to_string(),
            AtomicData::Time(value) => value.to_string(),
            AtomicData::GYearMonth(value) => value.to_string(),
            AtomicData::GYear(value) => value.to_string(),
            AtomicData::GMonthDay(value) => value.to_string(),
            AtomicData::GMonth(value) => value.to_string(),
            AtomicData::GDay(value) => value.to_string(),
            AtomicData::Binary(value) => {
                if self.atomic_type.is_derived_from(AtomicType::Base64Binary) {
                    encode_base64(value)
                } else {
                    encode_hex(value)
                }
            }
            AtomicData::QName(value) => value.lexical_form(),
        }
    }
}

impl PartialEq for AtomicValue {
    /// Compares the type label and the payload; the carried language and collation are ignored.
    fn eq(&self, other: &Self) -> bool {
        self.atomic_type == other.atomic_type && self.data == other.data
    }
}

impl Eq for AtomicValue {}

impl Hash for AtomicValue {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.atomic_type.hash(state);
        self.data.hash(state);
    }
}

impl fmt::Display for AtomicValue {
    /// Formats the value's [string value](AtomicValue::string_value).
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.string_value())
    }
}

/// The `xs:double` → `xs:string` mapping of
/// [XPath 3.1 §19.1.3](https://www.w3.org/TR/xpath-functions-31/#casting-to-string).
///
/// This is *not* the XSD canonical representation, which always uses exponential notation:
/// XPath prescribes decimal notation when the absolute value lies in `[1e-6, 1e6)`, with no
/// trailing `.0` on an integral value — so `fn:string(4.0E2)` is `"400"`, not `"4.0E2"` —
/// and exponential notation with a one-digit mantissa everywhere else.
fn xpath_float_string(value: f64) -> String {
    if value.is_nan() {
        return "NaN".to_owned();
    }
    if value == f64::INFINITY {
        return "INF".to_owned();
    }
    if value == f64::NEG_INFINITY {
        return "-INF".to_owned();
    }
    if value == 0.0 {
        return if value.is_sign_negative() { "-0" } else { "0" }.to_owned();
    }
    if (1e-6..1e6).contains(&value.abs()) {
        return format!("{value}");
    }
    exponential_form(&format!("{value:e}"))
}

/// The `xs:float` → `xs:string` mapping, which is [`xpath_float_string`] computed at single
/// precision so that the shortest round-tripping decimal is the `xs:float` one.
fn xpath_float32_string(value: f32) -> String {
    if value.is_nan() {
        return "NaN".to_owned();
    }
    if value == f32::INFINITY {
        return "INF".to_owned();
    }
    if value == f32::NEG_INFINITY {
        return "-INF".to_owned();
    }
    if value == 0.0 {
        return if value.is_sign_negative() { "-0" } else { "0" }.to_owned();
    }
    if (1e-6..1e6).contains(&value.abs()) {
        return format!("{value}");
    }
    exponential_form(&format!("{value:e}"))
}

/// Turns Rust's `1e6`/`1.234e-7` spelling into the `1.0E6`/`1.234E-7` the specification asks
/// for: an uppercase `E`, and a mantissa that always carries a fractional part.
fn exponential_form(rendered: &str) -> String {
    match rendered.split_once('e') {
        Some((mantissa, exponent)) => {
            if mantissa.contains('.') {
                format!("{mantissa}E{exponent}")
            } else {
                format!("{mantissa}.0E{exponent}")
            }
        }
        None => rendered.to_owned(),
    }
}

/// Whether `atomic_type` may label a value whose payload is `data`.
pub(crate) fn label_matches_payload(atomic_type: AtomicType, data: &AtomicData) -> bool {
    match data {
        AtomicData::Text(_) => atomic_type.is_text(),
        AtomicData::Boolean(_) => atomic_type.is_derived_from(AtomicType::Boolean),
        AtomicData::Decimal(_) => {
            atomic_type.is_derived_from(AtomicType::Decimal)
                && !atomic_type.is_derived_from(AtomicType::Integer)
        }
        AtomicData::Integer(_) => atomic_type.is_derived_from(AtomicType::Integer),
        AtomicData::BigInteger(_) => atomic_type.is_derived_from(AtomicType::Integer),
        AtomicData::BigDecimal(_) => {
            atomic_type.is_derived_from(AtomicType::Decimal)
                && !atomic_type.is_derived_from(AtomicType::Integer)
        }
        AtomicData::Double(_) => atomic_type.is_derived_from(AtomicType::Double),
        AtomicData::Float(_) => atomic_type.is_derived_from(AtomicType::Float),
        AtomicData::Duration(_) => atomic_type == AtomicType::Duration,
        AtomicData::YearMonthDuration(_) => atomic_type == AtomicType::YearMonthDuration,
        AtomicData::DayTimeDuration(_) => atomic_type == AtomicType::DayTimeDuration,
        AtomicData::DateTime(_) => atomic_type.is_derived_from(AtomicType::DateTime),
        AtomicData::Date(_) => atomic_type.is_derived_from(AtomicType::Date),
        AtomicData::Time(_) => atomic_type.is_derived_from(AtomicType::Time),
        AtomicData::GYearMonth(_) => atomic_type.is_derived_from(AtomicType::GYearMonth),
        AtomicData::GYear(_) => atomic_type.is_derived_from(AtomicType::GYear),
        AtomicData::GMonthDay(_) => atomic_type.is_derived_from(AtomicType::GMonthDay),
        AtomicData::GMonth(_) => atomic_type.is_derived_from(AtomicType::GMonth),
        AtomicData::GDay(_) => atomic_type.is_derived_from(AtomicType::GDay),
        AtomicData::Binary(_) => {
            atomic_type.is_derived_from(AtomicType::HexBinary)
                || atomic_type.is_derived_from(AtomicType::Base64Binary)
        }
        AtomicData::QName(_) => {
            atomic_type.is_derived_from(AtomicType::QName)
                || atomic_type.is_derived_from(AtomicType::Notation)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ErrorCode;
    use std::collections::hash_map::DefaultHasher;

    fn hash_of(value: &AtomicValue) -> u64 {
        let mut hasher = DefaultHasher::new();
        value.hash(&mut hasher);
        hasher.finish()
    }

    #[test]
    fn constructors_label_their_payloads() {
        assert_eq!(AtomicValue::string("x").atomic_type(), AtomicType::String);
        assert_eq!(
            AtomicValue::untyped_atomic("x").atomic_type(),
            AtomicType::UntypedAtomic
        );
        assert_eq!(AtomicValue::any_uri("x").atomic_type(), AtomicType::AnyUri);
        assert_eq!(
            AtomicValue::boolean(true).atomic_type(),
            AtomicType::Boolean
        );
        assert_eq!(
            AtomicValue::integer(1_i64).atomic_type(),
            AtomicType::Integer
        );
        assert_eq!(
            AtomicValue::hex_binary(vec![1]).atomic_type(),
            AtomicType::HexBinary
        );
    }

    #[test]
    fn labels_must_match_payloads() {
        let error = AtomicValue::new(AtomicType::Date, AtomicData::Text("x".into()))
            .expect_err("xs:date cannot label a text payload");
        assert_eq!(error.code(), ErrorCode::Xpty0004);

        // xs:decimal's own label never fits an integer payload, and vice versa.
        assert!(
            AtomicValue::new(
                AtomicType::Decimal,
                AtomicData::Integer(Integer::from(1_i64))
            )
            .is_err()
        );
        assert!(
            AtomicValue::new(AtomicType::Short, AtomicData::Integer(Integer::from(1_i64))).is_ok()
        );
    }

    #[test]
    fn binary_string_value_depends_on_the_label_not_the_payload() {
        let bytes = vec![0xf0_u8, 0x0d];
        assert_eq!(
            AtomicValue::hex_binary(bytes.clone()).string_value(),
            "F00D"
        );
        assert_eq!(AtomicValue::base64_binary(bytes).string_value(), "8A0=");
    }

    #[test]
    fn equality_ignores_carried_language_and_collation() {
        let plain = AtomicValue::string("x");
        let annotated = plain
            .clone()
            .with_language("en")
            .with_collation("http://www.w3.org/2005/xpath-functions/collation/codepoint");
        assert_eq!(plain, annotated);
        assert_eq!(hash_of(&plain), hash_of(&annotated));
        assert_eq!(annotated.language(), Some("en"));
        assert_eq!(
            annotated.collation(),
            Some("http://www.w3.org/2005/xpath-functions/collation/codepoint")
        );
    }

    #[test]
    fn equality_distinguishes_type_labels() {
        let integer = AtomicValue::integer(1_i64);
        let short = integer.clone().cast_label(AtomicType::Short).expect("fits");
        assert_ne!(integer, short);
        assert_eq!(short.data(), integer.data());
    }

    #[test]
    fn float_identity_semantics() {
        let nan = AtomicValue::double(f64::NAN);
        assert_eq!(nan, nan.clone(), "NaN is identical to itself");
        assert!(nan.is_nan());
        assert_ne!(AtomicValue::double(0.0), AtomicValue::double(-0.0));
        assert_eq!(hash_of(&nan), hash_of(&nan.clone()));
    }

    #[test]
    fn accessors_project_the_payload() {
        assert_eq!(AtomicValue::string("hi").as_str(), Some("hi"));
        assert_eq!(AtomicValue::boolean(false).as_bool(), Some(false));
        assert_eq!(AtomicValue::integer(3_i64).as_f64(), Some(3.0));
        assert_eq!(
            AtomicValue::hex_binary(vec![7]).as_bytes(),
            Some(&[7_u8][..])
        );
        assert_eq!(AtomicValue::integer(3_i64).as_str(), None);
        assert_eq!(AtomicValue::string("hi").as_f64(), None);
        assert_eq!(
            AtomicValue::qname(QNameValue::new(None::<&str>, None::<&str>, "n"))
                .as_qname()
                .map(QNameValue::local_name),
            Some("n")
        );
    }

    #[test]
    fn display_is_the_string_value() {
        assert_eq!(AtomicValue::boolean(true).to_string(), "true");
        assert_eq!(AtomicValue::integer(-5_i64).to_string(), "-5");
    }

    #[test]
    fn big_payloads_demote_to_the_fixed_width_when_they_fit() {
        use crate::{BigDecimal, BigInteger};

        // A fits-in-i64 value pushed through the big path comes back as the fixed variant.
        let small = AtomicData::from_big_integer(BigInteger::from_i64(5));
        assert!(matches!(small, AtomicData::Integer(_)));
        // ... and one that does not fit stays big.
        let big = BigInteger::from_i128(1_i128 << 100);
        assert!(matches!(
            AtomicData::from_big_integer(big),
            AtomicData::BigInteger(_)
        ));

        // A decimal with at most 18 fractional digits and an i128 significand demotes.
        let fits = BigDecimal::from_fixed(Decimal::from(3_i64));
        assert!(matches!(
            AtomicData::from_big_decimal(fits),
            AtomicData::Decimal(_)
        ));
        // The `AtomicValue` constructors demote too, keeping the fixed type label.
        assert_eq!(
            AtomicValue::from_big_integer(BigInteger::from_i64(7)),
            AtomicValue::integer(7_i64)
        );
    }

    #[test]
    fn nominal_types_cover_every_payload_shape() {
        assert_eq!(
            AtomicData::Text(String::new()).nominal_type(),
            AtomicType::String
        );
        assert_eq!(
            AtomicData::Binary(Vec::new()).nominal_type(),
            AtomicType::HexBinary
        );
        assert!(AtomicData::Integer(Integer::from(1_i64)).is_numeric());
        assert!(!AtomicData::Text(String::new()).is_numeric());
    }
}
