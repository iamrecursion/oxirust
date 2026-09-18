//! Area-model **primitives**.
//!
//! # The rendering boundary
//!
//! `oxixml-fo` is the *front half* of an XSL-FO formatter: it parses a
//! stylesheet result into a refined formatting object tree and hands that tree,
//! with every property computed, to a renderer. **Pagination, page-master
//! selection, space resolution, float and footnote placement, font metrics and
//! PDF (or any other) output are deliberately out of scope** and live in the
//! downstream `fop` project.
//!
//! What this module provides is the vocabulary that the two halves share: the
//! [area tree](https://www.w3.org/TR/xsl11/#area_model) data types of XSL 1.1 §4
//! ([`BlockArea`], [`LineArea`], [`InlineArea`] and the geometry they are built
//! from) and, in [`mod@line`], the line-breaking primitives a renderer needs to turn
//! a stream of inline material into line areas. Nothing here consumes an
//! [`FoTree`](crate::FoTree) or lays anything out; a renderer drives these types
//! itself.

pub mod line;

pub use crate::area::line::{Breakpoint, Item, LineBreaker, LineBreakingStrategy, Penalty};

use crate::property::Length;
use crate::tree::NodeId;

/// The two dimensions of an area, named after the progression directions rather
/// than after the page, as XSL 1.1 §4.2.3 does.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct Dimensions {
    /// Extent along the inline-progression-direction.
    pub inline_progression_dimension: Length,
    /// Extent along the block-progression-direction.
    pub block_progression_dimension: Length,
}

impl Dimensions {
    /// Builds a pair of dimensions.
    #[must_use]
    pub const fn new(
        inline_progression_dimension: Length,
        block_progression_dimension: Length,
    ) -> Self {
        Self {
            inline_progression_dimension,
            block_progression_dimension,
        }
    }
}

/// The offset of an area from the start and before edges of its parent's
/// content rectangle.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct Offset {
    /// Displacement along the inline-progression-direction.
    pub inline: Length,
    /// Displacement along the block-progression-direction.
    pub block: Length,
}

/// The four writing-mode-relative edges of a rectangle, as used for the border,
/// padding and space traits of XSL 1.1 §4.2.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct Edges {
    /// The before edge.
    pub before: Length,
    /// The after edge.
    pub after: Length,
    /// The start edge.
    pub start: Length,
    /// The end edge.
    pub end: Length,
}

impl Edges {
    /// Uniform edges.
    #[must_use]
    pub const fn uniform(length: Length) -> Self {
        Self {
            before: length,
            after: length,
            start: length,
            end: length,
        }
    }

    /// The total extent in the inline-progression-direction.
    #[must_use]
    pub fn inline_extent(self) -> Length {
        self.start + self.end
    }

    /// The total extent in the block-progression-direction.
    #[must_use]
    pub fn block_extent(self) -> Length {
        self.before + self.after
    }
}

/// The traits shared by every area (XSL 1.1 §4.2.2 *Common Traits*), reduced to
/// the geometry a renderer needs.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct AreaTraits {
    /// The content rectangle's dimensions.
    pub dimensions: Dimensions,
    /// The border widths.
    pub border: Edges,
    /// The padding.
    pub padding: Edges,
    /// The resolved space and indents surrounding the area.
    pub space: Edges,
    /// Offset from the parent's content rectangle.
    pub offset: Offset,
}

impl AreaTraits {
    /// The extent of the *allocation rectangle* in the
    /// inline-progression-direction: content plus padding, border and space.
    #[must_use]
    pub fn allocation_inline(self) -> Length {
        self.dimensions.inline_progression_dimension
            + self.border.inline_extent()
            + self.padding.inline_extent()
            + self.space.inline_extent()
    }

    /// The extent of the allocation rectangle in the
    /// block-progression-direction.
    #[must_use]
    pub fn allocation_block(self) -> Length {
        self.dimensions.block_progression_dimension
            + self.border.block_extent()
            + self.padding.block_extent()
            + self.space.block_extent()
    }
}

/// The `area-class` trait of XSL 1.1 §6.1.1.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum AreaClass {
    /// `xsl-normal`: part of the normal flow of text.
    #[default]
    Normal,
    /// `xsl-absolute`: absolutely positioned.
    Absolute,
    /// `xsl-footnote`.
    Footnote,
    /// `xsl-side-float`.
    SideFloat,
    /// `xsl-before-float`.
    BeforeFloat,
}

/// What an [`InlineArea`] carries.
#[derive(Debug, Clone, PartialEq)]
pub enum InlineContent {
    /// A run of glyphs.
    Glyphs(String),
    /// A word space, which line breaking may stretch or shrink.
    Space,
    /// A leader.
    Leader,
    /// A graphic or a foreign object.
    Foreign,
    /// A nested sequence of inline areas, as generated by `fo:inline`.
    Nested(Vec<InlineArea>),
}

/// An inline area: a glyph run, a space, a leader or a nested inline sequence
/// (XSL 1.1 §4.6).
#[derive(Debug, Clone, PartialEq)]
pub struct InlineArea {
    /// The area's traits.
    pub traits: AreaTraits,
    /// What the area contains.
    pub content: InlineContent,
    /// Distance from the before edge of the area to its dominant baseline.
    pub baseline_offset: Length,
    /// The formatting object that generated this area, when known.
    pub generated_by: Option<NodeId>,
}

impl InlineArea {
    /// Builds a glyph area of the given inline extent.
    #[must_use]
    pub fn glyphs(text: impl Into<String>, dimensions: Dimensions) -> Self {
        Self {
            traits: AreaTraits {
                dimensions,
                ..AreaTraits::default()
            },
            content: InlineContent::Glyphs(text.into()),
            baseline_offset: dimensions.block_progression_dimension,
            generated_by: None,
        }
    }
}

/// A line area: the block-level area a line of text occupies (XSL 1.1 §4.5).
#[derive(Debug, Clone, PartialEq, Default)]
pub struct LineArea {
    /// The area's traits.
    pub traits: AreaTraits,
    /// The inline areas stacked along the inline-progression-direction.
    pub children: Vec<InlineArea>,
    /// Distance from the before edge of the line to its dominant baseline.
    pub baseline_offset: Length,
    /// Whether this is the last line of its block, which decides whether
    /// `text-align-last` applies.
    pub is_last: bool,
}

impl LineArea {
    /// The sum of the inline extents of the line's children.
    #[must_use]
    pub fn content_inline_extent(&self) -> Length {
        self.children
            .iter()
            .map(|child| child.traits.allocation_inline())
            .sum()
    }
}

/// What a [`BlockArea`] stacks.
#[derive(Debug, Clone, PartialEq)]
pub enum BlockContent {
    /// Nested block areas.
    Blocks(Vec<BlockArea>),
    /// Line areas, as produced by line building.
    Lines(Vec<LineArea>),
}

impl Default for BlockContent {
    fn default() -> Self {
        Self::Blocks(Vec::new())
    }
}

/// A block area (XSL 1.1 §4.4).
#[derive(Debug, Clone, PartialEq, Default)]
pub struct BlockArea {
    /// The area's traits.
    pub traits: AreaTraits,
    /// The resolved `start-indent`.
    pub start_indent: Length,
    /// The resolved `end-indent`.
    pub end_indent: Length,
    /// Whether this area generates a reference area.
    pub is_reference_area: bool,
    /// The `area-class` trait.
    pub area_class: AreaClass,
    /// What the area stacks.
    pub content: BlockContent,
    /// The formatting object that generated this area, when known.
    pub generated_by: Option<NodeId>,
}

impl BlockArea {
    /// The inline extent available to the area's content, that is its
    /// inline-progression-dimension less the two indents.
    #[must_use]
    pub fn available_inline_extent(&self) -> Length {
        self.traits.dimensions.inline_progression_dimension - self.start_indent - self.end_indent
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn allocation_rectangles_add_the_surrounding_traits() {
        let traits = AreaTraits {
            dimensions: Dimensions::new(Length::from_points(100.0), Length::from_points(20.0)),
            border: Edges::uniform(Length::from_points(1.0)),
            padding: Edges::uniform(Length::from_points(2.0)),
            space: Edges::uniform(Length::from_points(3.0)),
            offset: Offset::default(),
        };
        assert_eq!(traits.allocation_inline(), Length::from_points(112.0));
        assert_eq!(traits.allocation_block(), Length::from_points(32.0));
    }

    #[test]
    fn block_areas_subtract_their_indents() {
        let area = BlockArea {
            traits: AreaTraits {
                dimensions: Dimensions::new(Length::from_points(400.0), Length::ZERO),
                ..AreaTraits::default()
            },
            start_indent: Length::from_points(24.0),
            end_indent: Length::from_points(12.0),
            ..BlockArea::default()
        };
        assert_eq!(area.available_inline_extent(), Length::from_points(364.0));
    }

    #[test]
    fn line_areas_measure_their_children() {
        let line = LineArea {
            children: vec![
                InlineArea::glyphs(
                    "ab",
                    Dimensions::new(Length::from_points(10.0), Length::ZERO),
                ),
                InlineArea::glyphs(
                    "cd",
                    Dimensions::new(Length::from_points(15.0), Length::ZERO),
                ),
            ],
            ..LineArea::default()
        };
        assert_eq!(line.content_inline_extent(), Length::from_points(25.0));
    }
}
