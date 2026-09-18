//! Computed property values.

use core::fmt;

use crate::property::datatype::{
    Color, Keep, Length, LengthBpIp, LengthConditional, LengthRange, Measure, Shape, Space,
};

/// The computed value of a formatting property.
///
/// A `Value` is the result of [property refinement]: shorthands have been
/// expanded, expressions evaluated, relative units made absolute and
/// inheritance applied. Percentages whose reference value depends on the area
/// geometry are *not* resolved — the spec calls that the *actual* value, and it
/// is the downstream formatter's job — so [`Value::Percent`] and
/// [`Measure::Percent`] survive into the refined tree.
///
/// [property refinement]: https://www.w3.org/TR/xsl11/#prrefine
#[derive(Debug, Clone, PartialEq, Default)]
#[non_exhaustive]
pub enum Value {
    /// The property has no value: it was not specified and has no initial
    /// value (`id`, `master-name`, …).
    #[default]
    Absent,
    /// An enumeration token or keyword, such as `auto` or `lr-tb`.
    Keyword(Box<str>),
    /// An absolute length.
    Length(Length),
    /// An unresolved percentage; `100.0` means 100%.
    Percent(f64),
    /// A `<number>`.
    Number(f64),
    /// An `<integer>`.
    Integer(i64),
    /// A `<color>`.
    Color(Color),
    /// A `<string>`.
    Text(Box<str>),
    /// A `<name>`, `<id>` or `<idref>`.
    Name(Box<str>),
    /// A `<uri-specification>`; the stored text is the IRI, with the `url(…)`
    /// wrapper and any quotes removed.
    Uri(Box<str>),
    /// A `<character>`.
    Character(char),
    /// An `<angle>`, normalized to the range `0.0..360.0` degrees.
    Angle(f64),
    /// A `<shape>`.
    Shape(Shape),
    /// A `<space>`.
    Space(Space),
    /// A `<keep>`.
    Keep(Keep),
    /// A `<length-range>`.
    LengthRange(LengthRange),
    /// A `<length-conditional>`.
    LengthConditional(LengthConditional),
    /// A `<length-bp-ip-direction>`.
    LengthBpIp(LengthBpIp),
    /// The result of `proportional-column-width(n)`; the column takes a share
    /// `n` of the space left over after the fixed column widths.
    ProportionalColumnWidth(f64),
    /// A whitespace- or comma-separated list, as used by `font-family`,
    /// `text-decoration` and friends.
    List(Vec<Value>),
}

impl Value {
    /// Builds a keyword value.
    #[must_use]
    pub fn keyword(text: &str) -> Self {
        Self::Keyword(text.into())
    }

    /// Returns the keyword text, if this value is a keyword.
    #[must_use]
    pub fn as_keyword(&self) -> Option<&str> {
        match self {
            Self::Keyword(text) => Some(text),
            _ => None,
        }
    }

    /// Returns `true` when this value is the given keyword.
    #[must_use]
    pub fn is_keyword(&self, keyword: &str) -> bool {
        self.as_keyword() == Some(keyword)
    }

    /// Returns the absolute length, if this value is one.
    #[must_use]
    pub fn as_length(&self) -> Option<Length> {
        match self {
            Self::Length(length) => Some(*length),
            _ => None,
        }
    }

    /// Returns this value as a [`Measure`], mapping `auto` and percentages.
    #[must_use]
    pub fn as_measure(&self) -> Option<Measure> {
        match self {
            Self::Length(length) => Some(Measure::Length(*length)),
            Self::Percent(pct) => Some(Measure::Percent(*pct)),
            Self::Keyword(keyword) if &**keyword == "auto" => Some(Measure::Auto),
            _ => None,
        }
    }

    /// Returns the number, accepting both `<number>` and `<integer>`.
    #[must_use]
    pub fn as_number(&self) -> Option<f64> {
        match self {
            Self::Number(number) => Some(*number),
            Self::Integer(integer) => Some(*integer as f64),
            _ => None,
        }
    }

    /// Returns the integer, if this value is one.
    #[must_use]
    pub fn as_integer(&self) -> Option<i64> {
        match self {
            Self::Integer(integer) => Some(*integer),
            Self::Number(number) if number.fract() == 0.0 => Some(*number as i64),
            _ => None,
        }
    }

    /// Returns the colour, if this value is one.
    #[must_use]
    pub fn as_color(&self) -> Option<&Color> {
        match self {
            Self::Color(color) => Some(color),
            _ => None,
        }
    }

    /// Returns the textual payload of a string, name, keyword or URI value.
    #[must_use]
    pub fn as_str(&self) -> Option<&str> {
        match self {
            Self::Keyword(text) | Self::Text(text) | Self::Name(text) | Self::Uri(text) => {
                Some(text)
            }
            _ => None,
        }
    }

    /// Returns the `<space>` payload, if this value is one.
    #[must_use]
    pub fn as_space(&self) -> Option<Space> {
        match self {
            Self::Space(space) => Some(*space),
            _ => None,
        }
    }

    /// Returns the `<keep>` payload, if this value is one.
    #[must_use]
    pub fn as_keep(&self) -> Option<Keep> {
        match self {
            Self::Keep(keep) => Some(*keep),
            _ => None,
        }
    }

    /// Returns the `<length-range>` payload, if this value is one.
    #[must_use]
    pub fn as_length_range(&self) -> Option<LengthRange> {
        match self {
            Self::LengthRange(range) => Some(*range),
            _ => None,
        }
    }

    /// Returns the `<length-conditional>` payload, if this value is one.
    #[must_use]
    pub fn as_length_conditional(&self) -> Option<LengthConditional> {
        match self {
            Self::LengthConditional(conditional) => Some(*conditional),
            _ => None,
        }
    }

    /// Returns the `<length-bp-ip-direction>` payload, if this value is one.
    #[must_use]
    pub fn as_length_bp_ip(&self) -> Option<LengthBpIp> {
        match self {
            Self::LengthBpIp(value) => Some(*value),
            _ => None,
        }
    }

    /// Returns the list elements, if this value is a list.
    #[must_use]
    pub fn as_list(&self) -> Option<&[Value]> {
        match self {
            Self::List(items) => Some(items),
            _ => None,
        }
    }

    /// Returns `true` if the property has no value at all.
    #[must_use]
    pub const fn is_absent(&self) -> bool {
        matches!(self, Self::Absent)
    }
}

impl fmt::Display for Value {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Absent => Ok(()),
            Self::Keyword(text) | Self::Name(text) => f.write_str(text),
            Self::Text(text) => write!(f, "\"{text}\""),
            Self::Uri(text) => write!(f, "url(\"{text}\")"),
            Self::Length(length) => write!(f, "{length}"),
            Self::Percent(pct) => write!(f, "{pct}%"),
            Self::Number(number) => write!(f, "{number}"),
            Self::Integer(integer) => write!(f, "{integer}"),
            Self::Color(color) => write!(f, "{color}"),
            Self::Character(character) => write!(f, "{character}"),
            Self::Angle(angle) => write!(f, "{angle}deg"),
            Self::Shape(shape) => write!(
                f,
                "rect({} {} {} {})",
                shape.top, shape.right, shape.bottom, shape.left
            ),
            Self::Space(space) => write!(
                f,
                "space({}, {}, {}, {})",
                space.minimum,
                space.optimum,
                space.maximum,
                space.conditionality.name()
            ),
            Self::Keep(keep) => write!(
                f,
                "keep({:?}, {:?}, {:?})",
                keep.within_line, keep.within_column, keep.within_page
            ),
            Self::LengthRange(range) => {
                write!(
                    f,
                    "({}, {}, {})",
                    range.minimum, range.optimum, range.maximum
                )
            }
            Self::LengthConditional(conditional) => write!(
                f,
                "({}, {})",
                conditional.length,
                conditional.conditionality.name()
            ),
            Self::LengthBpIp(value) => write!(
                f,
                "(bp {}, ip {})",
                value.block_progression_direction, value.inline_progression_direction
            ),
            Self::ProportionalColumnWidth(share) => {
                write!(f, "proportional-column-width({share})")
            }
            Self::List(items) => {
                for (index, item) in items.iter().enumerate() {
                    if index > 0 {
                        f.write_str(" ")?;
                    }
                    write!(f, "{item}")?;
                }
                Ok(())
            }
        }
    }
}
