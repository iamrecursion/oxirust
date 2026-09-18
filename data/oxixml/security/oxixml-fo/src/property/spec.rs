//! Static metadata describing every XSL-FO 1.1 property.

use crate::property::id::{PROPERTY_COUNT, PROPERTY_IDS, PROPERTY_NAMES, PropertyId};
use crate::property::table::SPECS;

/// Bit flags describing which non-keyword forms a property value may take.
///
/// The flags mirror the datatypes named in the `Value:` line of the property
/// definitions in [XSL 1.1 §7](https://www.w3.org/TR/xsl11/#pr-section).
pub mod forms {
    /// The value may be a `<length>`.
    pub const LENGTH: u32 = 1 << 0;
    /// The value may be a `<percentage>`.
    pub const PERCENT: u32 = 1 << 1;
    /// The value may be a `<number>`.
    pub const NUMBER: u32 = 1 << 2;
    /// The value may be an `<integer>`.
    pub const INTEGER: u32 = 1 << 3;
    /// The value may be a `<color>`.
    pub const COLOR: u32 = 1 << 4;
    /// The value may be a `<string>`.
    pub const STRING: u32 = 1 << 5;
    /// The value may be a `<name>`, `<id>` or `<idref>`.
    pub const NAME: u32 = 1 << 6;
    /// The value may be a `<uri-specification>`.
    pub const URI: u32 = 1 << 7;
    /// The value may be a `<character>`.
    pub const CHARACTER: u32 = 1 << 8;
    /// The value may be an `<angle>`.
    pub const ANGLE: u32 = 1 << 9;
    /// The value may be a `<shape>`.
    pub const SHAPE: u32 = 1 << 10;
    /// The value may be a `<time>`.
    pub const TIME: u32 = 1 << 11;
    /// The value may be a `<frequency>`.
    pub const FREQUENCY: u32 = 1 << 12;
}

/// The compound datatype of a property, if it has one.
///
/// A compound property is written in a stylesheet either in *short form*
/// (`space-before="4pt"`) or component by component
/// (`space-before.optimum="4pt"`); see XSL 1.1 §5.11.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Compound {
    /// Not a compound property.
    None,
    /// `<space>`: `.minimum`, `.optimum`, `.maximum`, `.precedence`,
    /// `.conditionality`.
    Space,
    /// `<keep>`: `.within-line`, `.within-column`, `.within-page`.
    Keep,
    /// `<length-range>`: `.minimum`, `.optimum`, `.maximum`.
    LengthRange,
    /// `<length-conditional>`: `.length`, `.conditionality`.
    LengthConditional,
    /// `<length-bp-ip-direction>`: `.block-progression-direction`,
    /// `.inline-progression-direction`.
    LengthBpIp,
}

impl Compound {
    /// The component names of this compound datatype, in specification order.
    #[must_use]
    pub const fn components(self) -> &'static [&'static str] {
        match self {
            Self::None => &[],
            Self::Space => &[
                "minimum",
                "optimum",
                "maximum",
                "precedence",
                "conditionality",
            ],
            Self::Keep => &["within-line", "within-column", "within-page"],
            Self::LengthRange => &["minimum", "optimum", "maximum"],
            Self::LengthConditional => &["length", "conditionality"],
            Self::LengthBpIp => &[
                "block-progression-direction",
                "inline-progression-direction",
            ],
        }
    }

    /// Returns `true` when `name` is a component of this compound datatype.
    #[must_use]
    pub fn has_component(self, name: &str) -> bool {
        self.components().contains(&name)
    }
}

/// What a `<percentage>` written for a property is a percentage *of*.
///
/// Most reference values depend on the geometry of an area and are therefore
/// unknown during refinement; those percentages are preserved in the computed
/// value. The font-relative ones are resolved eagerly.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum PercentBase {
    /// Percentages do not apply to this property.
    None,
    /// The width of the containing block.
    ContainingBlockWidth,
    /// The height of the containing block.
    ContainingBlockHeight,
    /// The size of the padding rectangle.
    PaddingRectangle,
    /// The inline-progression-dimension of the relevant reference or block
    /// area.
    InlineProgressionDimension,
    /// The block-progression-dimension of the relevant reference area.
    BlockProgressionDimension,
    /// The font size in force for the formatting object.
    FontSize,
    /// The `line-height` of the formatting object.
    LineHeight,
    /// The intrinsic width of a graphic.
    IntrinsicWidth,
    /// The intrinsic height of a graphic.
    IntrinsicHeight,
    /// The inherited value of the same property.
    InheritedValue,
}

impl PercentBase {
    /// Whether the reference value is a font metric that refinement already
    /// knows, so the percentage can be turned into a length immediately.
    #[must_use]
    pub const fn is_font_relative(self) -> bool {
        matches!(self, Self::FontSize | Self::LineHeight)
    }
}

/// Static description of one formatting property.
#[derive(Debug, Clone, Copy)]
pub struct PropertySpec {
    /// The property's identifier.
    pub id: PropertyId,
    /// The property name as written in a stylesheet.
    pub name: &'static str,
    /// Whether the property is inherited (XSL 1.1 §5.1.4).
    pub inherited: bool,
    /// Whether the property is one of the twenty-four shorthands of §7.31.
    pub shorthand: bool,
    /// The compound datatype, if any.
    pub compound: Compound,
    /// The enumeration tokens and keywords the property accepts, excluding
    /// `inherit` (which every property accepts).
    pub keywords: &'static [&'static str],
    /// Bit set of the [`forms`] the property's value may take.
    pub forms: u32,
    /// The initial value, in the syntax a stylesheet would use.
    ///
    /// An empty string means the property has no initial value; a value
    /// starting with `#` is an internal marker resolved by the refinement
    /// engine (`#current-color`, `#column-number`, `#fo-precedence`,
    /// `#leader-length`).
    pub initial: &'static str,
    /// What a percentage refers to.
    pub percent_base: PercentBase,
}

impl PropertySpec {
    /// Returns `true` when `keyword` is one of the property's enumeration
    /// tokens.
    #[must_use]
    pub fn accepts_keyword(&self, keyword: &str) -> bool {
        self.keywords.contains(&keyword)
    }

    /// Returns `true` when the property accepts every form in `mask`.
    #[must_use]
    pub const fn accepts(&self, mask: u32) -> bool {
        self.forms & mask != 0
    }
}

impl PropertyId {
    /// The property name as written in a stylesheet.
    ///
    /// ```
    /// use oxixml_fo::property::PropertyId;
    ///
    /// assert_eq!(PropertyId::SpaceBefore.name(), "space-before");
    /// ```
    #[must_use]
    pub fn name(self) -> &'static str {
        PROPERTY_NAMES[self as usize]
    }

    /// Looks a property up by name.
    ///
    /// The name must be the bare property name; strip any `.component` suffix
    /// first, or use [`PropertyKey::parse`](crate::property::PropertyKey::parse).
    ///
    /// ```
    /// use oxixml_fo::property::PropertyId;
    ///
    /// assert_eq!(PropertyId::parse("border-start-width"), Some(PropertyId::BorderStartWidth));
    /// assert_eq!(PropertyId::parse("border-start-width.length"), None);
    /// ```
    #[must_use]
    pub fn parse(name: &str) -> Option<Self> {
        PROPERTY_NAMES
            .binary_search(&name)
            .ok()
            .map(|index| PROPERTY_IDS[index])
    }

    /// The static metadata for this property.
    #[must_use]
    pub fn spec(self) -> &'static PropertySpec {
        &SPECS[self as usize]
    }

    /// Whether the property is inherited (XSL 1.1 §5.1.4).
    #[must_use]
    pub fn is_inherited(self) -> bool {
        self.spec().inherited
    }

    /// Whether the property is one of the twenty-four shorthands of §7.31.
    #[must_use]
    pub fn is_shorthand(self) -> bool {
        self.spec().shorthand
    }

    /// The compound datatype of the property, if it has one.
    #[must_use]
    pub fn compound(self) -> Compound {
        self.spec().compound
    }

    /// Every property, in ascending identifier order.
    #[must_use]
    pub fn all() -> &'static [PropertyId; PROPERTY_COUNT] {
        &PROPERTY_IDS
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_property_round_trips_through_its_name() {
        for id in PropertyId::all() {
            assert_eq!(PropertyId::parse(id.name()), Some(*id), "{}", id.name());
            assert_eq!(id.spec().id, *id);
            assert_eq!(id.spec().name, id.name());
        }
    }

    #[test]
    fn names_are_sorted_for_binary_search() {
        assert!(PROPERTY_NAMES.windows(2).all(|pair| pair[0] < pair[1]));
    }

    #[test]
    fn inheritance_matches_the_recommendation() {
        assert!(PropertyId::WritingMode.is_inherited());
        assert!(PropertyId::FontSize.is_inherited());
        assert!(PropertyId::StartIndent.is_inherited());
        assert!(PropertyId::Color.is_inherited());
        assert!(!PropertyId::TextDecoration.is_inherited());
        assert!(!PropertyId::Id.is_inherited());
        assert!(!PropertyId::DominantBaseline.is_inherited());
        assert!(!PropertyId::SpaceBefore.is_inherited());
        assert!(!PropertyId::BorderTopWidth.is_inherited());
    }

    #[test]
    fn there_are_twenty_four_shorthands() {
        let count = PropertyId::all()
            .iter()
            .filter(|id| id.is_shorthand())
            .count();
        assert_eq!(count, 24);
    }
}
