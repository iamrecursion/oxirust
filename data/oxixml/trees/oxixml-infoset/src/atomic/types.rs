//! [`AtomicType`] — the XDM atomic type names, their primitive bases, and their substitution
//! (derivation) relationships.

use oxixml_qname::ExpandedName;
use std::fmt;

/// The namespace name of the XML Schema built-in datatypes, `http://www.w3.org/2001/XMLSchema`.
///
/// XDM 3.1 places `xs:untypedAtomic` and `xs:anyAtomicType` in this namespace alongside the
/// datatypes defined by XML Schema itself.
pub const XSD_NAMESPACE: &str = "http://www.w3.org/2001/XMLSchema";

/// One of the atomic types an [`AtomicValue`](crate::AtomicValue) can be labelled with.
///
/// This is the set of *built-in* atomic types of the
/// [XDM 3.1 type hierarchy](https://www.w3.org/TR/xpath-datamodel-31/#types-hierarchy): the
/// nineteen primitive types of [XML Schema 1.1 Part 2](https://www.w3.org/TR/xmlschema11-2/),
/// the built-in atomic types derived from them by restriction, and the two types XDM adds of
/// its own, `xs:untypedAtomic` and `xs:anyAtomicType`.
///
/// User-defined schema types are deliberately *not* modelled here: a schema-aware processor
/// tracks those in its own type registry and reports the nearest built-in ancestor to this
/// layer. [`AtomicType::primitive`] gives that ancestor for every variant.
///
/// # Examples
///
/// ```
/// use oxixml_infoset::AtomicType;
///
/// // Every type knows its primitive base and its place in the numeric tower.
/// assert_eq!(AtomicType::PositiveInteger.primitive(), AtomicType::Decimal);
/// assert!(AtomicType::PositiveInteger.is_numeric());
/// assert!(!AtomicType::Date.is_numeric());
///
/// // Names are rendered in Clark notation, in the XML Schema namespace.
/// assert_eq!(AtomicType::UntypedAtomic.local_name(), "untypedAtomic");
/// assert_eq!(
///     AtomicType::Double.expanded_name().to_string(),
///     "{http://www.w3.org/2001/XMLSchema}double"
/// );
///
/// // Derivation is transitive and reflexive.
/// assert!(AtomicType::Ncname.is_derived_from(AtomicType::String));
/// assert!(AtomicType::String.is_derived_from(AtomicType::AnyAtomicType));
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
#[non_exhaustive]
pub enum AtomicType {
    /// `xs:anyAtomicType`, the common base of every atomic type.
    AnyAtomicType,
    /// `xs:untypedAtomic`, the type of unvalidated content.
    UntypedAtomic,
    /// `xs:string`.
    String,
    /// `xs:normalizedString`.
    NormalizedString,
    /// `xs:token`.
    Token,
    /// `xs:language`.
    Language,
    /// `xs:NMTOKEN`.
    Nmtoken,
    /// `xs:Name`.
    Name,
    /// `xs:NCName`.
    Ncname,
    /// `xs:ID`.
    Id,
    /// `xs:IDREF`.
    Idref,
    /// `xs:ENTITY`.
    Entity,
    /// `xs:anyURI`.
    AnyUri,
    /// `xs:boolean`.
    Boolean,
    /// `xs:decimal`.
    Decimal,
    /// `xs:integer`.
    Integer,
    /// `xs:nonPositiveInteger`.
    NonPositiveInteger,
    /// `xs:negativeInteger`.
    NegativeInteger,
    /// `xs:long`.
    Long,
    /// `xs:int`.
    Int,
    /// `xs:short`.
    Short,
    /// `xs:byte`.
    Byte,
    /// `xs:nonNegativeInteger`.
    NonNegativeInteger,
    /// `xs:unsignedLong`.
    UnsignedLong,
    /// `xs:unsignedInt`.
    UnsignedInt,
    /// `xs:unsignedShort`.
    UnsignedShort,
    /// `xs:unsignedByte`.
    UnsignedByte,
    /// `xs:positiveInteger`.
    PositiveInteger,
    /// `xs:float`.
    Float,
    /// `xs:double`.
    Double,
    /// `xs:duration`.
    Duration,
    /// `xs:yearMonthDuration`.
    YearMonthDuration,
    /// `xs:dayTimeDuration`.
    DayTimeDuration,
    /// `xs:dateTime`.
    DateTime,
    /// `xs:dateTimeStamp`.
    DateTimeStamp,
    /// `xs:date`.
    Date,
    /// `xs:time`.
    Time,
    /// `xs:gYearMonth`.
    GYearMonth,
    /// `xs:gYear`.
    GYear,
    /// `xs:gMonthDay`.
    GMonthDay,
    /// `xs:gMonth`.
    GMonth,
    /// `xs:gDay`.
    GDay,
    /// `xs:hexBinary`.
    HexBinary,
    /// `xs:base64Binary`.
    Base64Binary,
    /// `xs:QName`.
    QName,
    /// `xs:NOTATION`.
    Notation,
}

impl AtomicType {
    /// Every variant, in declaration order. Useful for exhaustive tests and for building name
    /// lookup tables.
    pub const ALL: &'static [Self] = &[
        Self::AnyAtomicType,
        Self::UntypedAtomic,
        Self::String,
        Self::NormalizedString,
        Self::Token,
        Self::Language,
        Self::Nmtoken,
        Self::Name,
        Self::Ncname,
        Self::Id,
        Self::Idref,
        Self::Entity,
        Self::AnyUri,
        Self::Boolean,
        Self::Decimal,
        Self::Integer,
        Self::NonPositiveInteger,
        Self::NegativeInteger,
        Self::Long,
        Self::Int,
        Self::Short,
        Self::Byte,
        Self::NonNegativeInteger,
        Self::UnsignedLong,
        Self::UnsignedInt,
        Self::UnsignedShort,
        Self::UnsignedByte,
        Self::PositiveInteger,
        Self::Float,
        Self::Double,
        Self::Duration,
        Self::YearMonthDuration,
        Self::DayTimeDuration,
        Self::DateTime,
        Self::DateTimeStamp,
        Self::Date,
        Self::Time,
        Self::GYearMonth,
        Self::GYear,
        Self::GMonthDay,
        Self::GMonth,
        Self::GDay,
        Self::HexBinary,
        Self::Base64Binary,
        Self::QName,
        Self::Notation,
    ];

    /// The type's local name, without a prefix (for example `"positiveInteger"`).
    #[must_use]
    pub const fn local_name(self) -> &'static str {
        match self {
            Self::AnyAtomicType => "anyAtomicType",
            Self::UntypedAtomic => "untypedAtomic",
            Self::String => "string",
            Self::NormalizedString => "normalizedString",
            Self::Token => "token",
            Self::Language => "language",
            Self::Nmtoken => "NMTOKEN",
            Self::Name => "Name",
            Self::Ncname => "NCName",
            Self::Id => "ID",
            Self::Idref => "IDREF",
            Self::Entity => "ENTITY",
            Self::AnyUri => "anyURI",
            Self::Boolean => "boolean",
            Self::Decimal => "decimal",
            Self::Integer => "integer",
            Self::NonPositiveInteger => "nonPositiveInteger",
            Self::NegativeInteger => "negativeInteger",
            Self::Long => "long",
            Self::Int => "int",
            Self::Short => "short",
            Self::Byte => "byte",
            Self::NonNegativeInteger => "nonNegativeInteger",
            Self::UnsignedLong => "unsignedLong",
            Self::UnsignedInt => "unsignedInt",
            Self::UnsignedShort => "unsignedShort",
            Self::UnsignedByte => "unsignedByte",
            Self::PositiveInteger => "positiveInteger",
            Self::Float => "float",
            Self::Double => "double",
            Self::Duration => "duration",
            Self::YearMonthDuration => "yearMonthDuration",
            Self::DayTimeDuration => "dayTimeDuration",
            Self::DateTime => "dateTime",
            Self::DateTimeStamp => "dateTimeStamp",
            Self::Date => "date",
            Self::Time => "time",
            Self::GYearMonth => "gYearMonth",
            Self::GYear => "gYear",
            Self::GMonthDay => "gMonthDay",
            Self::GMonth => "gMonth",
            Self::GDay => "gDay",
            Self::HexBinary => "hexBinary",
            Self::Base64Binary => "base64Binary",
            Self::QName => "QName",
            Self::Notation => "NOTATION",
        }
    }

    /// Looks a type up by its local name in the XML Schema namespace.
    ///
    /// # Examples
    ///
    /// ```
    /// use oxixml_infoset::AtomicType;
    ///
    /// assert_eq!(AtomicType::from_local_name("gYearMonth"), Some(AtomicType::GYearMonth));
    /// assert_eq!(AtomicType::from_local_name("NCName"), Some(AtomicType::Ncname));
    /// assert_eq!(AtomicType::from_local_name("ncname"), None); // names are case-sensitive
    /// ```
    #[must_use]
    pub fn from_local_name(local_name: &str) -> Option<Self> {
        Self::ALL
            .iter()
            .copied()
            .find(|candidate| candidate.local_name() == local_name)
    }

    /// The type's [`ExpandedName`], in the XML Schema namespace.
    #[must_use]
    pub fn expanded_name(self) -> ExpandedName {
        ExpandedName::new(Some(XSD_NAMESPACE), self.local_name())
    }

    /// The type this one is *directly* derived from by restriction, or `None` for
    /// [`AtomicType::AnyAtomicType`], which is the root.
    ///
    /// # Examples
    ///
    /// ```
    /// use oxixml_infoset::AtomicType;
    ///
    /// assert_eq!(AtomicType::Ncname.base_type(), Some(AtomicType::Name));
    /// assert_eq!(AtomicType::AnyAtomicType.base_type(), None);
    /// ```
    #[must_use]
    pub const fn base_type(self) -> Option<Self> {
        Some(match self {
            Self::AnyAtomicType => return None,
            Self::UntypedAtomic
            | Self::String
            | Self::AnyUri
            | Self::Boolean
            | Self::Decimal
            | Self::Float
            | Self::Double
            | Self::Duration
            | Self::DateTime
            | Self::Date
            | Self::Time
            | Self::GYearMonth
            | Self::GYear
            | Self::GMonthDay
            | Self::GMonth
            | Self::GDay
            | Self::HexBinary
            | Self::Base64Binary
            | Self::QName
            | Self::Notation => Self::AnyAtomicType,
            Self::NormalizedString => Self::String,
            Self::Token => Self::NormalizedString,
            Self::Language | Self::Nmtoken | Self::Name => Self::Token,
            Self::Ncname => Self::Name,
            Self::Id | Self::Idref | Self::Entity => Self::Ncname,
            Self::Integer => Self::Decimal,
            Self::NonPositiveInteger | Self::Long | Self::NonNegativeInteger => Self::Integer,
            Self::NegativeInteger => Self::NonPositiveInteger,
            Self::Int => Self::Long,
            Self::Short => Self::Int,
            Self::Byte => Self::Short,
            Self::UnsignedLong | Self::PositiveInteger => Self::NonNegativeInteger,
            Self::UnsignedInt => Self::UnsignedLong,
            Self::UnsignedShort => Self::UnsignedInt,
            Self::UnsignedByte => Self::UnsignedShort,
            Self::YearMonthDuration | Self::DayTimeDuration => Self::Duration,
            Self::DateTimeStamp => Self::DateTime,
        })
    }

    /// The primitive type at the root of this type's derivation chain.
    ///
    /// `xs:untypedAtomic` and `xs:anyAtomicType` are their own "primitives" — they head no XML
    /// Schema primitive hierarchy but are never derived from one either.
    ///
    /// # Examples
    ///
    /// ```
    /// use oxixml_infoset::AtomicType;
    ///
    /// assert_eq!(AtomicType::UnsignedByte.primitive(), AtomicType::Decimal);
    /// assert_eq!(AtomicType::Id.primitive(), AtomicType::String);
    /// assert_eq!(AtomicType::DayTimeDuration.primitive(), AtomicType::Duration);
    /// ```
    #[must_use]
    pub const fn primitive(self) -> Self {
        let mut current = self;
        loop {
            match current.base_type() {
                None | Some(Self::AnyAtomicType) => return current,
                Some(base) => current = base,
            }
        }
    }

    /// Whether this type is `other`, or is derived from it by (possibly repeated) restriction.
    ///
    /// # Examples
    ///
    /// ```
    /// use oxixml_infoset::AtomicType;
    ///
    /// assert!(AtomicType::Byte.is_derived_from(AtomicType::Integer));
    /// assert!(AtomicType::Byte.is_derived_from(AtomicType::Byte));
    /// assert!(!AtomicType::Byte.is_derived_from(AtomicType::UnsignedByte));
    /// ```
    #[must_use]
    pub const fn is_derived_from(self, other: Self) -> bool {
        let mut current = self;
        loop {
            if current as u8 == other as u8 {
                return true;
            }
            match current.base_type() {
                None => return false,
                Some(base) => current = base,
            }
        }
    }

    /// Whether values of this type belong to the numeric tower — that is, whether the type is
    /// derived from `xs:decimal` (including `xs:integer` and its restrictions), `xs:float`, or
    /// `xs:double`.
    #[must_use]
    pub const fn is_numeric(self) -> bool {
        matches!(self.primitive(), Self::Decimal | Self::Float | Self::Double)
    }

    /// Whether values of this type are represented lexically as text: the `xs:string`
    /// hierarchy, `xs:anyURI`, and `xs:untypedAtomic`.
    #[must_use]
    pub const fn is_text(self) -> bool {
        matches!(
            self.primitive(),
            Self::String | Self::AnyUri | Self::UntypedAtomic
        )
    }

    /// Whether this type is one of the seven date/time primitives or a restriction of one.
    #[must_use]
    pub const fn is_date_time(self) -> bool {
        matches!(
            self.primitive(),
            Self::DateTime
                | Self::Date
                | Self::Time
                | Self::GYearMonth
                | Self::GYear
                | Self::GMonthDay
                | Self::GMonth
                | Self::GDay
        )
    }
}

impl fmt::Display for AtomicType {
    /// Formats as `xs:localName`, the conventional spelling in specification prose and in error
    /// messages.
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "xs:{}", self.local_name())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashSet;

    #[test]
    fn local_names_are_unique_and_round_trip() {
        let mut seen = HashSet::new();
        for &ty in AtomicType::ALL {
            assert!(seen.insert(ty.local_name()), "duplicate name {ty}");
            assert_eq!(AtomicType::from_local_name(ty.local_name()), Some(ty));
        }
        assert_eq!(seen.len(), AtomicType::ALL.len());
    }

    #[test]
    fn derivation_chains_terminate_at_any_atomic_type() {
        for &ty in AtomicType::ALL {
            assert!(
                ty.is_derived_from(AtomicType::AnyAtomicType),
                "{ty} should be derived from xs:anyAtomicType"
            );
        }
    }

    #[test]
    fn primitives_match_the_xml_schema_hierarchy() {
        assert_eq!(AtomicType::PositiveInteger.primitive(), AtomicType::Decimal);
        assert_eq!(AtomicType::Entity.primitive(), AtomicType::String);
        assert_eq!(AtomicType::DateTimeStamp.primitive(), AtomicType::DateTime);
        assert_eq!(
            AtomicType::YearMonthDuration.primitive(),
            AtomicType::Duration
        );
        assert_eq!(
            AtomicType::UntypedAtomic.primitive(),
            AtomicType::UntypedAtomic
        );
        assert_eq!(
            AtomicType::AnyAtomicType.primitive(),
            AtomicType::AnyAtomicType
        );
    }

    #[test]
    fn classification_predicates() {
        assert!(AtomicType::Byte.is_numeric());
        assert!(AtomicType::Double.is_numeric());
        assert!(!AtomicType::Boolean.is_numeric());
        assert!(AtomicType::Idref.is_text());
        assert!(AtomicType::AnyUri.is_text());
        assert!(AtomicType::UntypedAtomic.is_text());
        assert!(!AtomicType::HexBinary.is_text());
        assert!(AtomicType::GMonthDay.is_date_time());
        assert!(!AtomicType::DayTimeDuration.is_date_time());
    }

    #[test]
    fn display_uses_the_conventional_xs_prefix() {
        assert_eq!(AtomicType::Ncname.to_string(), "xs:NCName");
        assert_eq!(
            AtomicType::Ncname.expanded_name(),
            ExpandedName::new(Some(XSD_NAMESPACE), "NCName")
        );
    }
}
