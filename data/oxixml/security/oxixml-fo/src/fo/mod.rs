//! The formatting-object vocabulary of XSL 1.1.
//!
//! [`FoElement`] enumerates every element the Recommendation defines in the
//! [XSL-FO namespace](crate::FO_NAMESPACE), and [`content`] holds the content
//! models used to validate a tree while it is being built.

mod element;

pub mod content;

pub use crate::fo::content::{ContentModel, Particle};
pub use crate::fo::element::{FO_COUNT, FoElement};

use crate::fo::element::{FO_ELEMENTS, FO_NAMES};

/// How a formatting object may be used inside the content of another one, as
/// defined by [XSL 1.1 §6.2].
///
/// The categories correspond to the `%block;` and `%inline;` parameter entities
/// and to the prose that follows them.
///
/// [XSL 1.1 §6.2]: https://www.w3.org/TR/xsl11/#fo_content
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Category {
    /// A member of `%block;`.
    Block,
    /// A member of `%inline;`.
    Inline,
    /// A *neutral container*, allowed wherever `#PCDATA`, `%block;` or
    /// `%inline;` are.
    Neutral,
    /// `fo:float`: an out-of-line object allowed wherever `#PCDATA`, `%block;`
    /// or `%inline;` are.
    OutOfLineBlock,
    /// `fo:footnote`: an out-of-line object allowed wherever `#PCDATA` or
    /// `%inline;` are.
    OutOfLineInline,
    /// `fo:change-bar-begin` and `fo:change-bar-end`, which define *points* and
    /// may appear anywhere below `fo:flow` or `fo:static-content`.
    Point,
    /// `fo:marker`, allowed only as a leading child of the objects that accept
    /// markers.
    Marker,
    /// `fo:initial-property-set`, allowed only after the leading markers of an
    /// `fo:block`.
    InitialPropertySet,
    /// A pagination, layout, table or list structure object, which may only
    /// appear where its parent's content model names it explicitly.
    Structural,
}

impl FoElement {
    /// The local part of the element name, without the `fo:` prefix.
    ///
    /// ```
    /// use oxixml_fo::fo::FoElement;
    ///
    /// assert_eq!(FoElement::ListItemBody.local_name(), "list-item-body");
    /// ```
    #[must_use]
    pub fn local_name(self) -> &'static str {
        FO_NAMES[self as usize]
    }

    /// The element name as it is conventionally written, with the `fo:` prefix.
    #[must_use]
    pub fn qualified_name(self) -> String {
        format!("fo:{}", self.local_name())
    }

    /// Looks a formatting object up by its local name.
    ///
    /// ```
    /// use oxixml_fo::fo::FoElement;
    ///
    /// assert_eq!(FoElement::parse("simple-page-master"), Some(FoElement::SimplePageMaster));
    /// assert_eq!(FoElement::parse("fo:block"), None, "the prefix is not part of the name");
    /// ```
    #[must_use]
    pub fn parse(local_name: &str) -> Option<Self> {
        FO_NAMES
            .binary_search(&local_name)
            .ok()
            .map(|index| FO_ELEMENTS[index])
    }

    /// Every formatting object, in ascending identifier order.
    #[must_use]
    pub fn all() -> &'static [FoElement; FO_COUNT] {
        &FO_ELEMENTS
    }

    /// How this formatting object may be used inside another one's content.
    #[must_use]
    pub fn category(self) -> Category {
        use FoElement as F;
        match self {
            F::Block | F::BlockContainer | F::TableAndCaption | F::Table | F::ListBlock => {
                Category::Block
            }
            F::BidiOverride
            | F::Character
            | F::ExternalGraphic
            | F::InstreamForeignObject
            | F::Inline
            | F::InlineContainer
            | F::Leader
            | F::PageNumber
            | F::PageNumberCitation
            | F::PageNumberCitationLast
            | F::ScalingValueCitation
            | F::BasicLink
            | F::MultiToggle
            | F::IndexPageCitationList => Category::Inline,
            F::MultiSwitch
            | F::MultiProperties
            | F::IndexRangeBegin
            | F::IndexRangeEnd
            | F::Wrapper
            | F::RetrieveMarker
            | F::RetrieveTableMarker => Category::Neutral,
            F::Float => Category::OutOfLineBlock,
            F::Footnote => Category::OutOfLineInline,
            F::ChangeBarBegin | F::ChangeBarEnd => Category::Point,
            F::Marker => Category::Marker,
            F::InitialPropertySet => Category::InitialPropertySet,
            _ => Category::Structural,
        }
    }

    /// Whether this formatting object generates a *reference area*.
    ///
    /// The distinction matters for the `start-indent` and `end-indent`
    /// formulae of [XSL 1.1 §5.3.2], which drop the inherited indent for
    /// reference-area generators.
    ///
    /// [XSL 1.1 §5.3.2]: https://www.w3.org/TR/xsl11/#d0e4967
    #[must_use]
    pub fn generates_reference_area(self) -> bool {
        matches!(
            self,
            Self::SimplePageMaster
                | Self::RegionBody
                | Self::RegionBefore
                | Self::RegionAfter
                | Self::RegionStart
                | Self::RegionEnd
                | Self::BlockContainer
                | Self::InlineContainer
                | Self::TableCell
                | Self::Table
        )
    }

    /// The properties named in this formatting object's definition in
    /// [XSL 1.1 §6](https://www.w3.org/TR/xsl11/#fo-section), sorted by
    /// [`PropertyId`](crate::property::PropertyId).
    ///
    /// This is the union of the individual properties listed for the object and
    /// of the *common property sets* it references. It does **not** include the
    /// inherited properties that §5.1.4 allows on every object; for the set a
    /// stylesheet may actually write, use [`FoElement::applicable_properties`].
    #[must_use]
    pub fn specific_properties(self) -> &'static [crate::property::PropertyId] {
        crate::property::applicability::APPLICABLE[self as usize]
    }

    /// Every property that may be specified on this formatting object, sorted
    /// by [`PropertyId`](crate::property::PropertyId).
    ///
    /// Three things go into the set:
    ///
    /// * the properties named in the object's definition
    ///   ([`FoElement::specific_properties`]);
    /// * every *inherited* property, because XSL 1.1 §5.1.4 says plainly that
    ///   "the inheritable properties can be placed on any formatting object" —
    ///   this is what makes `font-size` on `fo:root` and `writing-mode` on
    ///   `fo:page-sequence` legal even though §6 does not list them there;
    /// * every shorthand at least one of whose expansion targets is in the set.
    #[must_use]
    pub fn applicable_properties(self) -> &'static [crate::property::PropertyId] {
        &applicable::TABLE[self as usize]
    }

    /// Whether `property` may be specified on this formatting object.
    #[must_use]
    pub fn accepts_property(self, property: crate::property::PropertyId) -> bool {
        self.applicable_properties()
            .binary_search(&property)
            .is_ok()
    }

    /// The content model of this formatting object.
    #[must_use]
    pub fn content_model(self) -> &'static ContentModel {
        content::model(self)
    }
}

/// The full applicability table, built once from the static §6 lists plus the
/// universally acceptable inherited properties of §5.1.4.
mod applicable {
    use std::sync::LazyLock;

    use crate::fo::element::{FO_COUNT, FO_ELEMENTS};
    use crate::property::PropertyId;
    use crate::property::shorthand::expansion_targets;

    pub(super) static TABLE: LazyLock<Vec<Vec<PropertyId>>> = LazyLock::new(|| {
        let inherited: Vec<PropertyId> = PropertyId::all()
            .iter()
            .copied()
            .filter(|property| property.is_inherited() && !property.is_shorthand())
            .collect();
        let mut table = Vec::with_capacity(FO_COUNT);
        for element in FO_ELEMENTS {
            let mut properties: Vec<PropertyId> = element.specific_properties().to_vec();
            properties.extend_from_slice(&inherited);
            properties.sort_unstable();
            properties.dedup();
            let shorthands: Vec<PropertyId> = PropertyId::all()
                .iter()
                .copied()
                .filter(|property| {
                    property.is_shorthand()
                        && expansion_targets(*property)
                            .iter()
                            .any(|target| properties.binary_search(target).is_ok())
                })
                .collect();
            properties.extend(shorthands);
            properties.sort_unstable();
            properties.dedup();
            table.push(properties);
        }
        table
    });
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::property::PropertyId;

    #[test]
    fn names_round_trip_and_are_sorted() {
        assert!(FO_NAMES.windows(2).all(|pair| pair[0] < pair[1]));
        for element in FoElement::all() {
            assert_eq!(FoElement::parse(element.local_name()), Some(*element));
        }
    }

    #[test]
    fn categories_match_section_6_2() {
        assert_eq!(FoElement::Block.category(), Category::Block);
        assert_eq!(FoElement::Table.category(), Category::Block);
        assert_eq!(FoElement::Leader.category(), Category::Inline);
        assert_eq!(FoElement::Wrapper.category(), Category::Neutral);
        assert_eq!(FoElement::Float.category(), Category::OutOfLineBlock);
        assert_eq!(FoElement::Footnote.category(), Category::OutOfLineInline);
        assert_eq!(FoElement::Root.category(), Category::Structural);
    }

    #[test]
    fn applicability_follows_the_property_lists_of_section_6() {
        assert!(FoElement::Block.accepts_property(PropertyId::TextAlign));
        assert!(FoElement::Block.accepts_property(PropertyId::StartIndent));
        assert!(FoElement::Block.accepts_property(PropertyId::Margin));
        assert!(!FoElement::Block.accepts_property(PropertyId::MasterName));
        assert!(!FoElement::Block.accepts_property(PropertyId::ColumnWidth));

        assert!(FoElement::SimplePageMaster.accepts_property(PropertyId::PageHeight));
        assert!(FoElement::SimplePageMaster.accepts_property(PropertyId::MasterName));
        assert!(!FoElement::SimplePageMaster.accepts_property(PropertyId::PaddingTop));

        assert!(FoElement::TableCell.accepts_property(PropertyId::NumberColumnsSpanned));
        assert!(FoElement::TableColumn.accepts_property(PropertyId::ColumnWidth));
        assert!(FoElement::RegionBody.accepts_property(PropertyId::ColumnCount));

        // The specific lists of §6 exclude the inherited properties, which
        // §5.1.4 then admits everywhere.
        assert!(
            !FoElement::Root
                .specific_properties()
                .contains(&PropertyId::FontSize)
        );
        assert!(
            !FoElement::Flow
                .specific_properties()
                .contains(&PropertyId::FontSize)
        );
    }

    #[test]
    fn every_inherited_property_may_be_placed_on_any_formatting_object() {
        // XSL 1.1 §5.1.4: "The inheritable properties can be placed on any
        // formatting object."
        for element in FoElement::all() {
            for property in PropertyId::all() {
                if property.is_inherited() && !property.is_shorthand() {
                    assert!(
                        element.accepts_property(*property),
                        "{} rejects the inherited property {}",
                        element.qualified_name(),
                        property.name()
                    );
                }
            }
        }
    }

    #[test]
    fn applicable_property_lists_are_sorted() {
        for element in FoElement::all() {
            for properties in [
                element.specific_properties(),
                element.applicable_properties(),
            ] {
                assert!(
                    properties.windows(2).all(|pair| pair[0] < pair[1]),
                    "{} is not sorted",
                    element.qualified_name()
                );
            }
        }
    }

    #[test]
    fn reference_area_generators() {
        assert!(FoElement::BlockContainer.generates_reference_area());
        assert!(FoElement::TableCell.generates_reference_area());
        assert!(!FoElement::Block.generates_reference_area());
    }
}
