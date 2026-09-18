//! Category predicates over [`AttrId`], grouping attributes the way the
//! specification catalog's own sections group them: core, conditional
//! processing, presentation, XLink, the four SMIL animation-attribute
//! groupings, geometry, and SVG-2 deprecation.
//!
//! As with [`crate::names::elements`], every predicate below documents its
//! exact true-set so the transcription is auditable against the catalog
//! without re-deriving it. Several attributes intentionally satisfy more
//! than one predicate at once — [`AttrId::Fill`] is both
//! [`AttrId::is_presentation`] (the paint property) and
//! [`AttrId::is_animation_timing`] (the freeze/remove attribute), because
//! it is genuinely both; see the [module documentation](super) for why
//! that is one variant rather than two.

use super::AttrId;

impl AttrId {
    /// True for the 11 core/XML-global attributes: `id`, `class`, `style`,
    /// `lang`, `xml:lang`, `xml:space`, `xml:base`, `tabindex`,
    /// `autofocus`, `slot`, `role`.
    ///
    /// These apply "everywhere sensible" per the catalog — this predicate
    /// does not narrow that down to a specific element set;
    /// [`crate::names::content::attribute_applies`] treats every core
    /// attribute as universally applicable.
    #[must_use]
    pub const fn is_core(self) -> bool {
        matches!(
            self,
            Self::Id
                | Self::Class
                | Self::Style
                | Self::Lang
                | Self::XmlLang
                | Self::XmlSpace
                | Self::XmlBase
                | Self::Tabindex
                | Self::Autofocus
                | Self::Slot
                | Self::Role
        )
    }

    /// True for the 3 conditional-processing attributes: `requiredFeatures`,
    /// `requiredExtensions`, `systemLanguage`.
    #[must_use]
    pub const fn is_conditional_processing(self) -> bool {
        matches!(
            self,
            Self::RequiredFeatures | Self::RequiredExtensions | Self::SystemLanguage
        )
    }

    /// True for the 73 presentation (property-mirroring) attributes: the
    /// 61-row SVG 1.1 `propidx.html` table, the 11-row SVG 2 new/changed
    /// properties table, and `font-kerning` (SVG 2's replacement for
    /// `kerning`, named in the catalog's inherited-properties list rather
    /// than tabulated as its own row).
    ///
    /// This includes [`Self::Fill`] (the paint property half of its
    /// name collision) and [`Self::D`] (`path`'s `d`, which SVG 2 also
    /// makes a CSS property).
    #[must_use]
    #[allow(clippy::too_many_lines)]
    pub const fn is_presentation(self) -> bool {
        matches!(
            self,
            Self::AlignmentBaseline
                | Self::BaselineShift
                | Self::Clip
                | Self::ClipPath
                | Self::ClipRule
                | Self::Color
                | Self::ColorInterpolation
                | Self::ColorInterpolationFilters
                | Self::ColorProfile
                | Self::ColorRendering
                | Self::Cursor
                | Self::Direction
                | Self::Display
                | Self::DominantBaseline
                | Self::EnableBackground
                | Self::Fill
                | Self::FillOpacity
                | Self::FillRule
                | Self::Filter
                | Self::FloodColor
                | Self::FloodOpacity
                | Self::Font
                | Self::FontFamily
                | Self::FontSize
                | Self::FontSizeAdjust
                | Self::FontStretch
                | Self::FontStyle
                | Self::FontVariant
                | Self::FontWeight
                | Self::FontKerning
                | Self::GlyphOrientationHorizontal
                | Self::GlyphOrientationVertical
                | Self::ImageRendering
                | Self::Kerning
                | Self::LetterSpacing
                | Self::LightingColor
                | Self::Marker
                | Self::MarkerEnd
                | Self::MarkerMid
                | Self::MarkerStart
                | Self::Mask
                | Self::Opacity
                | Self::Overflow
                | Self::PointerEvents
                | Self::ShapeRendering
                | Self::StopColor
                | Self::StopOpacity
                | Self::Stroke
                | Self::StrokeDasharray
                | Self::StrokeDashoffset
                | Self::StrokeLinecap
                | Self::StrokeLinejoin
                | Self::StrokeMiterlimit
                | Self::StrokeOpacity
                | Self::StrokeWidth
                | Self::TextAnchor
                | Self::TextDecoration
                | Self::TextRendering
                | Self::UnicodeBidi
                | Self::Visibility
                | Self::WordSpacing
                | Self::WritingMode
                | Self::PaintOrder
                | Self::VectorEffect
                | Self::Transform
                | Self::TransformOrigin
                | Self::TransformBox
                | Self::TextOverflow
                | Self::WhiteSpace
                | Self::MaskType
                | Self::Isolation
                | Self::MixBlendMode
                | Self::D
        )
    }

    /// True for the 7 legacy XLink-namespaced attributes: `xlink:href`,
    /// `xlink:show`, `xlink:actuate`, `xlink:type`, `xlink:title`,
    /// `xlink:arcrole`, `xlink:role`.
    ///
    /// [`Self::Href`] (the bare, non-namespaced SVG 2 attribute) is
    /// deliberately excluded — it is a different, newer attribute that
    /// happens to serve the same purpose, not an XLink attribute.
    #[must_use]
    pub const fn is_xlink(self) -> bool {
        matches!(
            self,
            Self::XlinkHref
                | Self::XlinkShow
                | Self::XlinkActuate
                | Self::XlinkType
                | Self::XlinkTitle
                | Self::XlinkArcrole
                | Self::XlinkRole
        )
    }

    /// True for the 9 attributes in the catalog's animation "Timing"
    /// grouping: `begin`, `dur`, `end`, `min`, `max`, `restart`,
    /// `repeatCount`, `repeatDur`, `fill`.
    ///
    /// [`Self::Fill`] here is the freeze/remove half of its name
    /// collision — it also satisfies [`Self::is_presentation`].
    #[must_use]
    pub const fn is_animation_timing(self) -> bool {
        matches!(
            self,
            Self::Begin
                | Self::Dur
                | Self::End
                | Self::Min
                | Self::Max
                | Self::Restart
                | Self::RepeatCount
                | Self::RepeatDur
                | Self::Fill
        )
    }

    /// True for the 7 attributes in the catalog's animation "Value"
    /// grouping: `from`, `to`, `by`, `values`, `calcMode`, `keyTimes`,
    /// `keySplines`.
    #[must_use]
    pub const fn is_animation_value(self) -> bool {
        matches!(
            self,
            Self::From
                | Self::To
                | Self::By
                | Self::Values
                | Self::CalcMode
                | Self::KeyTimes
                | Self::KeySplines
        )
    }

    /// True for the 2 attributes in the catalog's animation "Addition"
    /// grouping: `additive`, `accumulate`.
    #[must_use]
    pub const fn is_animation_addition(self) -> bool {
        matches!(self, Self::Additive | Self::Accumulate)
    }

    /// True for the 24 pure-geometry attributes the mission's geometry
    /// grouping names, plus `radialGradient`'s SVG-2 focal-radius trio:
    /// `x`, `y`, `width`, `height`, `r`, `rx`, `ry`, `cx`, `cy`, `x1`,
    /// `y1`, `x2`, `y2`, `points`, `d`, `pathLength`, `dx`, `dy`, `rotate`,
    /// `textLength`, `lengthAdjust`, `fr`, `fx`, `fy`.
    ///
    /// Marker reference-point attributes (`refX`/`refY`/`markerWidth`/
    /// `markerHeight`) and viewport attributes (`viewBox`/
    /// `preserveAspectRatio`) are geometric in nature too, but the mission
    /// brief groups those under "marker" and "viewport" respectively, not
    /// "geometry"; this predicate follows that grouping rather than a
    /// broader "anything positional" reading.
    #[must_use]
    pub const fn is_geometry(self) -> bool {
        matches!(
            self,
            Self::X
                | Self::Y
                | Self::Width
                | Self::Height
                | Self::R
                | Self::Rx
                | Self::Ry
                | Self::Cx
                | Self::Cy
                | Self::X1
                | Self::Y1
                | Self::X2
                | Self::Y2
                | Self::Points
                | Self::D
                | Self::PathLength
                | Self::Dx
                | Self::Dy
                | Self::Rotate
                | Self::TextLength
                | Self::LengthAdjust
                | Self::Fr
                | Self::Fx
                | Self::Fy
        )
    }

    /// True for the 15 attributes this catalog flags as deprecated or
    /// removed in SVG 2: the 6 SVG-1.1-only XLink attributes
    /// (`xlink:show`, `xlink:actuate`, `xlink:type`, `xlink:title`,
    /// `xlink:arcrole`, `xlink:role` — but *not* `xlink:href`, which
    /// "remains valid indefinitely for backward compatibility" per the
    /// catalog and must not be warned on by default), `requiredFeatures`,
    /// `baseProfile`, `version`, `zoomAndPan`, `xml:space` (superseded by
    /// CSS `white-space`), `enable-background`, `kerning` (superseded by
    /// `font-kerning`), `clip` (superseded by `clip-path`), and
    /// `color-profile` (removed along with the `color-profile` element).
    #[must_use]
    pub const fn deprecated_in_svg2(self) -> bool {
        matches!(
            self,
            Self::XlinkShow
                | Self::XlinkActuate
                | Self::XlinkType
                | Self::XlinkTitle
                | Self::XlinkArcrole
                | Self::XlinkRole
                | Self::RequiredFeatures
                | Self::BaseProfile
                | Self::Version
                | Self::ZoomAndPan
                | Self::XmlSpace
                | Self::EnableBackground
                | Self::Kerning
                | Self::Clip
                | Self::ColorProfile
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn is_core_matches_the_eleven_core_attributes() {
        for id in [
            AttrId::Id,
            AttrId::Class,
            AttrId::Style,
            AttrId::Lang,
            AttrId::XmlLang,
            AttrId::XmlSpace,
            AttrId::XmlBase,
            AttrId::Tabindex,
            AttrId::Autofocus,
            AttrId::Slot,
            AttrId::Role,
        ] {
            assert!(id.is_core(), "{id:?} should be core");
        }
        assert_eq!(AttrId::ALL.iter().filter(|id| id.is_core()).count(), 11);
        assert!(!AttrId::Fill.is_core());
    }

    #[test]
    fn is_conditional_processing_matches_the_three_test_attributes() {
        assert!(AttrId::RequiredFeatures.is_conditional_processing());
        assert!(AttrId::RequiredExtensions.is_conditional_processing());
        assert!(AttrId::SystemLanguage.is_conditional_processing());
        assert!(!AttrId::Fill.is_conditional_processing());
    }

    #[test]
    fn is_presentation_has_73_members_and_includes_fill_and_d() {
        assert_eq!(
            AttrId::ALL.iter().filter(|id| id.is_presentation()).count(),
            73
        );
        assert!(AttrId::Fill.is_presentation());
        assert!(AttrId::D.is_presentation());
        assert!(AttrId::Transform.is_presentation());
        assert!(!AttrId::Id.is_presentation());
        assert!(!AttrId::X.is_presentation());
    }

    #[test]
    fn is_xlink_excludes_bare_href() {
        for id in [
            AttrId::XlinkHref,
            AttrId::XlinkShow,
            AttrId::XlinkActuate,
            AttrId::XlinkType,
            AttrId::XlinkTitle,
            AttrId::XlinkArcrole,
            AttrId::XlinkRole,
        ] {
            assert!(id.is_xlink(), "{id:?} should be xlink");
        }
        assert!(!AttrId::Href.is_xlink());
        assert_eq!(AttrId::ALL.iter().filter(|id| id.is_xlink()).count(), 7);
    }

    #[test]
    fn animation_groupings_partition_the_smil_attributes() {
        assert!(AttrId::Begin.is_animation_timing());
        assert!(AttrId::Fill.is_animation_timing());
        assert!(AttrId::From.is_animation_value());
        assert!(AttrId::Values.is_animation_value());
        assert!(AttrId::Additive.is_animation_addition());
        assert!(AttrId::Accumulate.is_animation_addition());
        assert!(!AttrId::Additive.is_animation_timing());
        assert!(!AttrId::Begin.is_animation_value());
        assert_eq!(
            AttrId::ALL
                .iter()
                .filter(|id| id.is_animation_timing())
                .count(),
            9
        );
        assert_eq!(
            AttrId::ALL
                .iter()
                .filter(|id| id.is_animation_value())
                .count(),
            7
        );
        assert_eq!(
            AttrId::ALL
                .iter()
                .filter(|id| id.is_animation_addition())
                .count(),
            2
        );
    }

    #[test]
    fn fill_satisfies_both_presentation_and_animation_timing() {
        // The documented name collision: one variant, two true predicates.
        assert!(AttrId::Fill.is_presentation());
        assert!(AttrId::Fill.is_animation_timing());
    }

    #[test]
    fn is_geometry_has_24_members() {
        assert_eq!(AttrId::ALL.iter().filter(|id| id.is_geometry()).count(), 24);
        assert!(AttrId::Cx.is_geometry());
        assert!(AttrId::Fr.is_geometry());
        assert!(AttrId::D.is_geometry());
        assert!(
            !AttrId::RefX.is_geometry(),
            "refX belongs to the marker grouping"
        );
        assert!(
            !AttrId::ViewBox.is_geometry(),
            "viewBox belongs to the viewport grouping"
        );
    }

    #[test]
    fn deprecated_in_svg2_excludes_xlink_href() {
        assert!(AttrId::XlinkShow.deprecated_in_svg2());
        assert!(AttrId::RequiredFeatures.deprecated_in_svg2());
        assert!(AttrId::BaseProfile.deprecated_in_svg2());
        assert!(AttrId::Version.deprecated_in_svg2());
        assert!(AttrId::ZoomAndPan.deprecated_in_svg2());
        assert!(AttrId::XmlSpace.deprecated_in_svg2());
        assert!(AttrId::EnableBackground.deprecated_in_svg2());
        assert!(AttrId::Kerning.deprecated_in_svg2());
        assert!(AttrId::Clip.deprecated_in_svg2());
        assert!(AttrId::ColorProfile.deprecated_in_svg2());
        assert!(
            !AttrId::XlinkHref.deprecated_in_svg2(),
            "xlink:href remains valid indefinitely and must not warn by default"
        );
        assert!(!AttrId::Href.deprecated_in_svg2());
        assert!(!AttrId::Fill.deprecated_in_svg2());
        assert_eq!(
            AttrId::ALL
                .iter()
                .filter(|id| id.deprecated_in_svg2())
                .count(),
            15
        );
    }
}
