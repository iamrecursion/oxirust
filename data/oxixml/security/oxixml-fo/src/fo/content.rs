//! Content models, from the `Contents:` clause of every formatting object
//! definition in [XSL 1.1 §6](https://www.w3.org/TR/xsl11/#fo-section).
//!
//! The models here are the XML content models of the Recommendation plus the
//! prose rules of §6.2 that are not expressible in one: the *neutral*
//! containers, the out-of-line objects, the change-bar points, and the leading
//! `fo:marker` children.

use std::sync::LazyLock;

use crate::fo::Category;
use crate::fo::element::{FO_COUNT, FO_ELEMENTS, FoElement};

/// One particle of a content model: a group of interchangeable child elements
/// with an occurrence range.
#[derive(Debug, Clone, Copy)]
pub struct Particle {
    /// The elements this particle accepts.
    pub alternatives: &'static [FoElement],
    /// Minimum number of occurrences.
    pub min: u32,
    /// Maximum number of occurrences; [`u32::MAX`] means unbounded.
    pub max: u32,
}

impl Particle {
    /// Whether `element` matches this particle.
    #[must_use]
    pub fn accepts(&self, element: FoElement) -> bool {
        self.alternatives.contains(&element)
    }
}

/// The content model of a formatting object.
#[derive(Debug, Clone, Copy)]
pub struct ContentModel {
    /// `#PCDATA` is allowed.
    pub text: bool,
    /// `%inline;` is allowed.
    pub inline: bool,
    /// `%block;` is allowed.
    pub block: bool,
    /// At least one `%block;` child is required.
    pub block_required: bool,
    /// A leading run of `fo:marker` children is allowed.
    pub markers: bool,
    /// An `fo:initial-property-set` may follow the leading markers.
    pub initial_property_set: bool,
    /// Elements from a non-XSL namespace are allowed.
    pub foreign: bool,
    /// The named child particles, in order.
    pub particles: &'static [Particle],
    /// Whether `particles` form a choice rather than a sequence.
    pub choice: bool,
}

const EMPTY: ContentModel = ContentModel {
    text: false,
    inline: false,
    block: false,
    block_required: false,
    markers: false,
    initial_property_set: false,
    foreign: false,
    particles: &[],
    choice: false,
};

impl ContentModel {
    /// Whether this object accepts no children at all (`EMPTY`).
    #[must_use]
    pub fn is_empty(&self) -> bool {
        !self.text
            && !self.inline
            && !self.block
            && !self.foreign
            && !self.markers
            && self.particles.is_empty()
    }

    /// Whether a child in category `category` is allowed here, ignoring
    /// ordering and occurrence constraints.
    #[must_use]
    pub fn accepts_category(&self, category: Category) -> bool {
        match category {
            Category::Block => self.block,
            Category::Inline => self.inline,
            // §6.2: neutral containers and `fo:float` may be used anywhere
            // `#PCDATA`, `%block;` or `%inline;` are allowed.
            Category::Neutral | Category::OutOfLineBlock | Category::Point => {
                self.text || self.block || self.inline
            }
            // `fo:footnote` may be used where `#PCDATA` or `%inline;` are.
            Category::OutOfLineInline => self.text || self.inline,
            Category::Marker => self.markers,
            Category::InitialPropertySet => self.initial_property_set,
            Category::Structural => false,
        }
    }

    /// Whether `child` is allowed here, ignoring ordering and occurrence
    /// constraints.
    #[must_use]
    pub fn accepts(&self, child: FoElement) -> bool {
        if self
            .particles
            .iter()
            .any(|particle| particle.accepts(child))
        {
            return true;
        }
        self.accepts_category(child.category())
    }

    /// A human-readable rendering of this content model, for diagnostics.
    #[must_use]
    pub fn describe(&self) -> String {
        if self.is_empty() {
            return "EMPTY".to_owned();
        }
        let mut parts: Vec<String> = Vec::new();
        if self.text {
            parts.push("#PCDATA".to_owned());
        }
        if self.inline {
            parts.push("%inline;".to_owned());
        }
        if self.block {
            parts.push(if self.block_required {
                "%block;+".to_owned()
            } else {
                "%block;".to_owned()
            });
        }
        if self.foreign {
            parts.push("non-XSL-namespace".to_owned());
        }
        for particle in self.particles {
            let names: Vec<&str> = particle
                .alternatives
                .iter()
                .map(|element| element.local_name())
                .collect();
            let occurrence = match (particle.min, particle.max) {
                (0, 1) => "?",
                (0, u32::MAX) => "*",
                (1, u32::MAX) => "+",
                _ => "",
            };
            parts.push(format!("({}){occurrence}", names.join("|")));
        }
        let separator = if self.choice { " | " } else { ", " };
        format!("({})", parts.join(separator))
    }
}

/// Builds a particle list.
macro_rules! particles {
    ($([$($element:ident),+ $(,)?] $min:literal ..= $max:expr),* $(,)?) => {
        &[$(Particle {
            alternatives: &[$(FoElement::$element),+],
            min: $min,
            max: $max,
        }),*]
    };
}

const UNBOUNDED: u32 = u32::MAX;

fn mixed(text: bool, inline: bool, block: bool, markers: bool) -> ContentModel {
    ContentModel {
        text,
        inline,
        block,
        markers,
        ..EMPTY
    }
}

fn blocks(markers: bool) -> ContentModel {
    ContentModel {
        block: true,
        block_required: true,
        markers,
        ..EMPTY
    }
}

fn sequence(particles: &'static [Particle], markers: bool) -> ContentModel {
    ContentModel {
        particles,
        markers,
        ..EMPTY
    }
}

fn choice(particles: &'static [Particle], markers: bool) -> ContentModel {
    ContentModel {
        particles,
        markers,
        choice: true,
        ..EMPTY
    }
}

static MODELS: LazyLock<[ContentModel; FO_COUNT]> = LazyLock::new(build_models);

/// The content model of `element`.
#[must_use]
pub fn model(element: FoElement) -> &'static ContentModel {
    &MODELS[element as usize]
}

fn build_models() -> [ContentModel; FO_COUNT] {
    use FoElement as F;
    let mut models = [EMPTY; FO_COUNT];
    for (slot, element) in models.iter_mut().zip(FO_ELEMENTS) {
        *slot = match element {
            F::BasicLink | F::BidiOverride | F::Inline | F::Wrapper => {
                mixed(true, true, true, true)
            }
            F::Block => ContentModel {
                initial_property_set: true,
                ..mixed(true, true, true, true)
            },
            F::BlockContainer | F::InlineContainer => blocks(true),
            F::Bookmark => sequence(
                particles!([BookmarkTitle] 1..=1, [Bookmark] 0..=UNBOUNDED),
                false,
            ),
            F::BookmarkTitle => mixed(true, false, false, false),
            F::BookmarkTree => sequence(particles!([Bookmark] 1..=UNBOUNDED), false),
            F::Declarations => ContentModel {
                foreign: true,
                ..sequence(particles!([ColorProfile] 0..=UNBOUNDED), false)
            },
            F::Float => blocks(false),
            F::Flow | F::StaticContent => blocks(true),
            F::FlowAssignment => sequence(
                particles!([FlowSourceList] 1..=1, [FlowTargetList] 1..=1),
                false,
            ),
            F::FlowMap => sequence(particles!([FlowAssignment] 1..=UNBOUNDED), false),
            F::FlowSourceList => sequence(particles!([FlowNameSpecifier] 1..=UNBOUNDED), false),
            F::FlowTargetList => sequence(particles!([RegionNameSpecifier] 1..=UNBOUNDED), false),
            F::FolioPrefix
            | F::FolioSuffix
            | F::IndexPageCitationListSeparator
            | F::IndexPageCitationRangeSeparator
            | F::IndexPageNumberPrefix
            | F::IndexPageNumberSuffix
            | F::Leader
            | F::Title => mixed(true, true, false, false),
            F::Footnote => sequence(particles!([Inline] 1..=1, [FootnoteBody] 1..=1), false),
            F::FootnoteBody => blocks(false),
            F::IndexKeyReference => sequence(
                particles!([IndexPageNumberPrefix] 0..=1, [IndexPageNumberSuffix] 0..=1),
                false,
            ),
            F::IndexPageCitationList => sequence(
                particles!(
                    [IndexPageCitationListSeparator] 0..=1,
                    [IndexPageCitationRangeSeparator] 0..=1,
                    [IndexKeyReference] 1..=UNBOUNDED
                ),
                false,
            ),
            F::InstreamForeignObject => ContentModel {
                foreign: true,
                ..EMPTY
            },
            F::LayoutMasterSet => sequence(
                particles!([SimplePageMaster, PageSequenceMaster, FlowMap] 1..=UNBOUNDED),
                false,
            ),
            F::ListBlock => sequence(particles!([ListItem] 1..=UNBOUNDED), true),
            F::ListItem => sequence(
                particles!([ListItemLabel] 1..=1, [ListItemBody] 1..=1),
                true,
            ),
            F::ListItemBody | F::ListItemLabel | F::TableCaption | F::TableCell => blocks(true),
            F::Marker | F::MultiCase | F::MultiToggle => mixed(true, true, true, false),
            F::MultiProperties => sequence(
                particles!([MultiPropertySet] 1..=UNBOUNDED, [Wrapper] 1..=1),
                false,
            ),
            F::MultiSwitch => sequence(particles!([MultiCase] 1..=UNBOUNDED), false),
            F::PageSequence => sequence(
                particles!(
                    [Title] 0..=1,
                    [FolioPrefix] 0..=1,
                    [FolioSuffix] 0..=1,
                    [StaticContent] 0..=UNBOUNDED,
                    [Flow] 1..=UNBOUNDED
                ),
                false,
            ),
            F::PageSequenceMaster => sequence(
                particles!([
                    SinglePageMasterReference,
                    RepeatablePageMasterReference,
                    RepeatablePageMasterAlternatives
                ] 1..=UNBOUNDED),
                false,
            ),
            F::PageSequenceWrapper => sequence(
                particles!([PageSequence, PageSequenceWrapper] 0..=UNBOUNDED),
                false,
            ),
            F::RepeatablePageMasterAlternatives => sequence(
                particles!([ConditionalPageMasterReference] 1..=UNBOUNDED),
                false,
            ),
            F::Root => sequence(
                particles!(
                    [LayoutMasterSet] 1..=1,
                    [Declarations] 0..=1,
                    [BookmarkTree] 0..=1,
                    [PageSequence, PageSequenceWrapper] 1..=UNBOUNDED
                ),
                false,
            ),
            F::SimplePageMaster => sequence(
                particles!(
                    [RegionBody] 1..=UNBOUNDED,
                    [RegionBefore] 0..=1,
                    [RegionAfter] 0..=1,
                    [RegionStart] 0..=1,
                    [RegionEnd] 0..=1
                ),
                false,
            ),
            F::Table => sequence(
                particles!(
                    [TableColumn] 0..=UNBOUNDED,
                    [TableHeader] 0..=1,
                    [TableFooter] 0..=1,
                    [TableBody] 1..=UNBOUNDED
                ),
                true,
            ),
            F::TableAndCaption => sequence(particles!([TableCaption] 0..=1, [Table] 1..=1), true),
            F::TableBody | F::TableFooter | F::TableHeader => choice(
                particles!([TableRow] 1..=UNBOUNDED, [TableCell] 1..=UNBOUNDED),
                true,
            ),
            F::TableRow => sequence(particles!([TableCell] 1..=UNBOUNDED), true),
            // Every remaining formatting object is declared `EMPTY`.
            _ => EMPTY,
        };
    }
    models
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn block_accepts_text_inline_and_block() {
        let model = model(FoElement::Block);
        assert!(model.accepts(FoElement::Inline));
        assert!(model.accepts(FoElement::Block));
        assert!(model.accepts(FoElement::Wrapper));
        assert!(model.accepts(FoElement::Footnote));
        assert!(model.accepts(FoElement::Float));
        assert!(model.accepts(FoElement::Marker));
        assert!(model.accepts(FoElement::InitialPropertySet));
        assert!(!model.accepts(FoElement::TableCell));
        assert!(!model.accepts(FoElement::RegionBody));
    }

    #[test]
    fn leader_forbids_block_level_children() {
        let model = model(FoElement::Leader);
        assert!(model.accepts(FoElement::Inline));
        assert!(!model.accepts(FoElement::Block));
        assert!(!model.accepts(FoElement::Marker));
    }

    #[test]
    fn structural_models_only_accept_their_named_children() {
        let list_item = model(FoElement::ListItem);
        assert!(list_item.accepts(FoElement::ListItemLabel));
        assert!(list_item.accepts(FoElement::ListItemBody));
        assert!(list_item.accepts(FoElement::Marker));
        assert!(!list_item.accepts(FoElement::Block));

        let page_master = model(FoElement::SimplePageMaster);
        assert!(page_master.accepts(FoElement::RegionBody));
        assert!(!page_master.accepts(FoElement::Flow));
    }

    #[test]
    fn empty_models_accept_nothing() {
        assert!(model(FoElement::RegionBody).is_empty());
        assert!(model(FoElement::Character).is_empty());
        assert!(model(FoElement::RetrieveMarker).is_empty());
        assert!(!model(FoElement::RegionBody).accepts(FoElement::Block));
    }

    #[test]
    fn descriptions_are_readable() {
        assert_eq!(model(FoElement::Character).describe(), "EMPTY");
        assert_eq!(
            model(FoElement::ListItem).describe(),
            "((list-item-label), (list-item-body))"
        );
        assert!(model(FoElement::Block).describe().contains("#PCDATA"));
        assert!(
            model(FoElement::TableBody)
                .describe()
                .contains("(table-row)+ | (table-cell)+")
        );
    }
}
