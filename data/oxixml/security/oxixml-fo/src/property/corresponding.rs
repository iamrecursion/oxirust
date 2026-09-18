//! Writing modes and the *corresponding property* relation of
//! [XSL 1.1 §5.3](https://www.w3.org/TR/xsl11/#prcorrespond).
//!
//! Many properties come in two spellings: a *relative* one, named after the
//! writing-mode-relative edges (`padding-start`), and an *absolute* one, named
//! after the geometric edges of the viewport (`padding-left`). Which pairs with
//! which depends on the `writing-mode` in force; the mapping tables below are
//! the ones in §5.3.

use crate::property::id::PropertyId;

/// A direction of progression.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Direction {
    /// Left to right.
    LeftToRight,
    /// Right to left.
    RightToLeft,
    /// Top to bottom.
    TopToBottom,
    /// Bottom to top.
    BottomToTop,
}

/// One of the four geometric edges of an area.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum AbsoluteEdge {
    /// The top edge.
    Top,
    /// The bottom edge.
    Bottom,
    /// The left edge.
    Left,
    /// The right edge.
    Right,
}

/// One of the four writing-mode-relative edges of an area.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum RelativeEdge {
    /// The edge the block-progression-direction starts from.
    Before,
    /// The edge the block-progression-direction ends at.
    After,
    /// The edge the inline-progression-direction starts from.
    Start,
    /// The edge the inline-progression-direction ends at.
    End,
}

/// The value of the `writing-mode` property, decomposed into the two
/// progression directions it fixes.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct WritingMode {
    /// The direction successive lines are stacked in.
    pub block_progression: Direction,
    /// The direction glyphs are laid out in within a line.
    pub inline_progression: Direction,
    /// Whether the inline-progression-direction alternates between lines
    /// (`lr-alternating-rl-tb` and friends).
    pub alternating: bool,
}

impl Default for WritingMode {
    /// The initial value of `writing-mode`, `lr-tb`.
    fn default() -> Self {
        Self {
            block_progression: Direction::TopToBottom,
            inline_progression: Direction::LeftToRight,
            alternating: false,
        }
    }
}

impl WritingMode {
    /// `lr-tb`, the initial value.
    pub const LR_TB: Self = Self {
        block_progression: Direction::TopToBottom,
        inline_progression: Direction::LeftToRight,
        alternating: false,
    };

    /// `rl-tb`.
    pub const RL_TB: Self = Self {
        block_progression: Direction::TopToBottom,
        inline_progression: Direction::RightToLeft,
        alternating: false,
    };

    /// `tb-rl`, the classical vertical Japanese mode.
    pub const TB_RL: Self = Self {
        block_progression: Direction::RightToLeft,
        inline_progression: Direction::TopToBottom,
        alternating: false,
    };

    /// Parses one of the `writing-mode` keywords of XSL 1.1 §7.29.7.
    ///
    /// ```
    /// use oxixml_fo::property::{Direction, WritingMode};
    ///
    /// let mode = WritingMode::parse("tb-rl").expect("a writing mode");
    /// assert_eq!(mode.block_progression, Direction::RightToLeft);
    /// assert_eq!(mode.inline_progression, Direction::TopToBottom);
    /// ```
    #[must_use]
    pub fn parse(keyword: &str) -> Option<Self> {
        use Direction::{BottomToTop, LeftToRight, RightToLeft, TopToBottom};
        let (inline, block, alternating) = match keyword {
            "lr-tb" | "lr" => (LeftToRight, TopToBottom, false),
            "rl-tb" | "rl" => (RightToLeft, TopToBottom, false),
            "tb-rl" | "tb" => (TopToBottom, RightToLeft, false),
            "tb-lr" | "tb-lr-in-lr-pairs" => (TopToBottom, LeftToRight, false),
            "bt-lr" => (BottomToTop, LeftToRight, false),
            "bt-rl" => (BottomToTop, RightToLeft, false),
            "lr-bt" => (LeftToRight, BottomToTop, false),
            "rl-bt" => (RightToLeft, BottomToTop, false),
            "lr-alternating-rl-bt" | "lr-inverting-rl-bt" => (LeftToRight, BottomToTop, true),
            "lr-alternating-rl-tb" | "lr-inverting-rl-tb" => (LeftToRight, TopToBottom, true),
            _ => return None,
        };
        Some(Self {
            block_progression: block,
            inline_progression: inline,
            alternating,
        })
    }

    /// The relative edge that `edge` corresponds to under this writing mode.
    #[must_use]
    pub fn relative_edge(self, edge: AbsoluteEdge) -> RelativeEdge {
        use AbsoluteEdge as A;
        use Direction::{BottomToTop, LeftToRight, RightToLeft, TopToBottom};
        use RelativeEdge as R;
        match (self.block_progression, edge) {
            (TopToBottom, A::Top) | (BottomToTop, A::Bottom) => return R::Before,
            (TopToBottom, A::Bottom) | (BottomToTop, A::Top) => return R::After,
            (LeftToRight, A::Left) | (RightToLeft, A::Right) => return R::Before,
            (LeftToRight, A::Right) | (RightToLeft, A::Left) => return R::After,
            _ => {}
        }
        match (self.inline_progression, edge) {
            (LeftToRight, A::Left) | (RightToLeft, A::Right) => R::Start,
            (LeftToRight, A::Right) | (RightToLeft, A::Left) => R::End,
            (TopToBottom, A::Top) | (BottomToTop, A::Bottom) => R::Start,
            _ => R::End,
        }
    }

    /// The geometric edge that `edge` corresponds to under this writing mode.
    #[must_use]
    pub fn absolute_edge(self, edge: RelativeEdge) -> AbsoluteEdge {
        for candidate in [
            AbsoluteEdge::Top,
            AbsoluteEdge::Bottom,
            AbsoluteEdge::Left,
            AbsoluteEdge::Right,
        ] {
            if self.relative_edge(candidate) == edge {
                return candidate;
            }
        }
        AbsoluteEdge::Top
    }

    /// Whether lines are stacked vertically, which decides whether `height`
    /// maps to the block- or the inline-progression-dimension (§5.3.3).
    #[must_use]
    pub fn block_progression_is_vertical(self) -> bool {
        matches!(
            self.block_progression,
            Direction::TopToBottom | Direction::BottomToTop
        )
    }
}

/// One family of corresponding properties: four relative spellings and four
/// absolute ones, in the order before/after/start/end and top/bottom/left/right.
#[derive(Debug, Clone, Copy)]
pub(crate) struct Family {
    pub(crate) relative: [PropertyId; 4],
    pub(crate) absolute: [PropertyId; 4],
}

impl Family {
    fn relative_for(&self, edge: RelativeEdge) -> PropertyId {
        self.relative[match edge {
            RelativeEdge::Before => 0,
            RelativeEdge::After => 1,
            RelativeEdge::Start => 2,
            RelativeEdge::End => 3,
        }]
    }

    fn absolute_for(&self, edge: AbsoluteEdge) -> PropertyId {
        self.absolute[match edge {
            AbsoluteEdge::Top => 0,
            AbsoluteEdge::Bottom => 1,
            AbsoluteEdge::Left => 2,
            AbsoluteEdge::Right => 3,
        }]
    }

    /// The (relative, absolute) property pairs of this family under `mode`.
    pub(crate) fn pairs(&self, mode: WritingMode) -> [(PropertyId, PropertyId); 4] {
        let edges = [
            AbsoluteEdge::Top,
            AbsoluteEdge::Bottom,
            AbsoluteEdge::Left,
            AbsoluteEdge::Right,
        ];
        let mut pairs = [(self.relative[0], self.absolute[0]); 4];
        for (slot, edge) in pairs.iter_mut().zip(edges) {
            *slot = (
                self.relative_for(mode.relative_edge(edge)),
                self.absolute_for(edge),
            );
        }
        pairs
    }
}

use PropertyId as P;

/// The border, padding and space/margin families of §5.3.1 and §5.3.2.
pub(crate) static FAMILIES: [Family; 5] = [
    Family {
        relative: [
            P::BorderBeforeColor,
            P::BorderAfterColor,
            P::BorderStartColor,
            P::BorderEndColor,
        ],
        absolute: [
            P::BorderTopColor,
            P::BorderBottomColor,
            P::BorderLeftColor,
            P::BorderRightColor,
        ],
    },
    Family {
        relative: [
            P::BorderBeforeStyle,
            P::BorderAfterStyle,
            P::BorderStartStyle,
            P::BorderEndStyle,
        ],
        absolute: [
            P::BorderTopStyle,
            P::BorderBottomStyle,
            P::BorderLeftStyle,
            P::BorderRightStyle,
        ],
    },
    Family {
        relative: [
            P::BorderBeforeWidth,
            P::BorderAfterWidth,
            P::BorderStartWidth,
            P::BorderEndWidth,
        ],
        absolute: [
            P::BorderTopWidth,
            P::BorderBottomWidth,
            P::BorderLeftWidth,
            P::BorderRightWidth,
        ],
    },
    Family {
        relative: [
            P::PaddingBefore,
            P::PaddingAfter,
            P::PaddingStart,
            P::PaddingEnd,
        ],
        absolute: [
            P::PaddingTop,
            P::PaddingBottom,
            P::PaddingLeft,
            P::PaddingRight,
        ],
    },
    // §5.3.2: the space-* properties correspond to the CSS margins.
    Family {
        relative: [P::SpaceBefore, P::SpaceAfter, P::SpaceStart, P::SpaceEnd],
        absolute: [P::MarginTop, P::MarginBottom, P::MarginLeft, P::MarginRight],
    },
];

/// The border-width and padding properties that enter the `start-indent` and
/// `end-indent` formulae of §5.3.2, for the start and end edges.
pub(crate) struct IndentTerms {
    /// `margin-*` for the corresponding geometric edge.
    pub(crate) margin: PropertyId,
    /// `padding-*` for the relative edge.
    pub(crate) padding: PropertyId,
    /// `border-*-width` for the relative edge.
    pub(crate) border_width: PropertyId,
}

/// The terms of the `start-indent` formula under `mode`.
pub(crate) fn start_indent_terms(mode: WritingMode) -> IndentTerms {
    IndentTerms {
        margin: margin_for(mode.absolute_edge(RelativeEdge::Start)),
        padding: P::PaddingStart,
        border_width: P::BorderStartWidth,
    }
}

/// The terms of the `end-indent` formula under `mode`.
pub(crate) fn end_indent_terms(mode: WritingMode) -> IndentTerms {
    IndentTerms {
        margin: margin_for(mode.absolute_edge(RelativeEdge::End)),
        padding: P::PaddingEnd,
        border_width: P::BorderEndWidth,
    }
}

fn margin_for(edge: AbsoluteEdge) -> PropertyId {
    match edge {
        AbsoluteEdge::Top => P::MarginTop,
        AbsoluteEdge::Bottom => P::MarginBottom,
        AbsoluteEdge::Left => P::MarginLeft,
        AbsoluteEdge::Right => P::MarginRight,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn lr_tb_maps_left_to_start() {
        let mode = WritingMode::LR_TB;
        assert_eq!(mode.relative_edge(AbsoluteEdge::Left), RelativeEdge::Start);
        assert_eq!(mode.relative_edge(AbsoluteEdge::Right), RelativeEdge::End);
        assert_eq!(mode.relative_edge(AbsoluteEdge::Top), RelativeEdge::Before);
        assert_eq!(
            mode.relative_edge(AbsoluteEdge::Bottom),
            RelativeEdge::After
        );
    }

    #[test]
    fn rl_tb_swaps_start_and_end() {
        let mode = WritingMode::RL_TB;
        assert_eq!(mode.relative_edge(AbsoluteEdge::Left), RelativeEdge::End);
        assert_eq!(mode.relative_edge(AbsoluteEdge::Right), RelativeEdge::Start);
        assert_eq!(mode.absolute_edge(RelativeEdge::Start), AbsoluteEdge::Right);
    }

    #[test]
    fn tb_rl_rotates_the_whole_mapping() {
        let mode = WritingMode::TB_RL;
        assert_eq!(mode.relative_edge(AbsoluteEdge::Top), RelativeEdge::Start);
        assert_eq!(mode.relative_edge(AbsoluteEdge::Bottom), RelativeEdge::End);
        assert_eq!(
            mode.relative_edge(AbsoluteEdge::Right),
            RelativeEdge::Before
        );
        assert_eq!(mode.relative_edge(AbsoluteEdge::Left), RelativeEdge::After);
        assert!(!mode.block_progression_is_vertical());
    }

    #[test]
    fn the_mapping_is_a_bijection_for_every_writing_mode() {
        for keyword in [
            "lr-tb", "rl-tb", "tb-rl", "tb-lr", "bt-lr", "bt-rl", "lr-bt", "rl-bt",
        ] {
            let mode = WritingMode::parse(keyword).expect("a writing mode");
            let mut seen = Vec::new();
            for edge in [
                AbsoluteEdge::Top,
                AbsoluteEdge::Bottom,
                AbsoluteEdge::Left,
                AbsoluteEdge::Right,
            ] {
                let relative = mode.relative_edge(edge);
                assert!(
                    !seen.contains(&relative),
                    "{keyword} maps twice onto {relative:?}"
                );
                assert_eq!(mode.absolute_edge(relative), edge);
                seen.push(relative);
            }
        }
    }

    #[test]
    fn padding_pairs_follow_the_writing_mode() {
        let family = FAMILIES[3];
        let pairs = family.pairs(WritingMode::LR_TB);
        assert!(pairs.contains(&(P::PaddingStart, P::PaddingLeft)));
        let pairs = family.pairs(WritingMode::RL_TB);
        assert!(pairs.contains(&(P::PaddingStart, P::PaddingRight)));
    }

    #[test]
    fn indent_terms_track_the_writing_mode() {
        assert_eq!(start_indent_terms(WritingMode::LR_TB).margin, P::MarginLeft);
        assert_eq!(
            start_indent_terms(WritingMode::RL_TB).margin,
            P::MarginRight
        );
        assert_eq!(end_indent_terms(WritingMode::LR_TB).margin, P::MarginRight);
    }
}
