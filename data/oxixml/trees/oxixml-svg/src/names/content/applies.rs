//! [`attribute_applies`]: which elements each attribute is meaningful on,
//! from specification requirements catalog §2's per-attribute "Elements"
//! (core/conditional/geometry/filter/animation groups) and "Applies to"
//! (presentation properties) columns.

use crate::names::attrs::AttrId;
use crate::names::elements::ElementId;

/// Reports whether `attr` applies to `element`, per the specification
/// requirements catalog's §2 attribute tables.
///
/// This is implemented "honestly via groups" rather than as an explicit
/// 216×88 matrix: core and conditional-processing attributes use the
/// catalog's own element lists (§2.1/§2.2); presentation properties use
/// the catalog's per-property "Applies to" column, translated into the
/// [`ElementId`] category predicates that column's own wording names
/// (`"container & graphics elements"` → [`ElementId::is_container`] `||`
/// [`ElementId::is_graphics`], `"shapes & text content elements"` →
/// [`ElementId::is_shape`] `||` [`ElementId::is_text_content`], and so on);
/// geometry and filter/animation attributes use the exact per-element
/// tables in catalog §2.4/§2.8/§2.9.
///
/// # The `fill` name collision
///
/// [`AttrId::Fill`] is one variant covering two unrelated attributes (see
/// [`crate::names::attrs`]'s module documentation): the presentation paint
/// property, and the SMIL animation freeze/remove attribute. Both readings
/// are real SVG, so `attribute_applies` reports `true` for **both**
/// `(Rect, Fill)` (the paint property applies to shapes) and
/// `(Animate, Fill)` (the freeze/remove attribute applies to animation
/// elements) — it does not, and structurally cannot, pick one. A caller
/// that needs to know *which* `fill` a given element/attribute pair means
/// must decide from the host element, not from this function.
///
/// # Permissive choices on catalog-uncertain items
///
/// - `(FeGaussianBlur, EdgeMode)` is `true`. Catalog §8 open item 1 flags
///   `feGaussianBlur`'s `edgeMode` as unconfirmed to even exist; the
///   permissive reading is taken rather than silently assuming it does not.
/// - `(*, RequiredFeatures)` for the catalog's listed elements is `true`
///   even though `requiredFeatures` is removed in SVG 2 — "removed" here
///   means the attribute has no processing effect in SVG 2, not that
///   authoring it is a parse error (catalog §8 open item 4); applicability
///   and deprecation are tracked separately (see
///   [`AttrId::deprecated_in_svg2`]).
#[must_use]
#[allow(clippy::too_many_lines)]
pub fn attribute_applies(element: ElementId, attr: AttrId) -> bool {
    use AttrId as A;
    use ElementId as E;

    match attr {
        // ---- 2.1 Core: universal ----
        A::Id
        | A::Class
        | A::Style
        | A::Lang
        | A::XmlLang
        | A::XmlSpace
        | A::XmlBase
        | A::Tabindex
        | A::Autofocus
        | A::Slot
        | A::Role => true,

        // ---- 2.2 Conditional processing: the catalog's own 30-element list ----
        A::RequiredFeatures | A::RequiredExtensions | A::SystemLanguage => {
            is_conditional_processing_target(element)
        }

        // ---- 2.3 Presentation properties ----
        A::AlignmentBaseline | A::BaselineShift => {
            matches!(element, E::Tspan | E::Tref | E::AltGlyph | E::TextPath)
        }
        A::Clip | A::Overflow => is_viewport_establishing(element),
        A::ClipPath
        | A::Color
        | A::ColorInterpolation
        | A::ColorRendering
        | A::Cursor
        | A::Filter
        | A::Mask
        | A::Opacity
        | A::Isolation
        | A::MixBlendMode
        | A::Transform
        | A::TransformOrigin
        | A::TransformBox
        | A::Display => element.is_container() || element.is_graphics(),
        A::ClipRule => element.is_shape(),
        A::ColorInterpolationFilters => element.is_filter_primitive(),
        A::ColorProfile => matches!(element, E::Image),
        A::Direction
        | A::DominantBaseline
        | A::Font
        | A::FontFamily
        | A::FontSize
        | A::FontSizeAdjust
        | A::FontStretch
        | A::FontStyle
        | A::FontVariant
        | A::FontWeight
        | A::FontKerning
        | A::GlyphOrientationHorizontal
        | A::GlyphOrientationVertical
        | A::Kerning
        | A::LetterSpacing
        | A::TextAnchor
        | A::TextDecoration
        | A::TextRendering
        | A::UnicodeBidi
        | A::WordSpacing
        | A::WritingMode
        | A::TextOverflow
        | A::WhiteSpace => element.is_text_content(),
        A::EnableBackground => element.is_container(),
        // `fill` additionally applies wherever the animation-timing
        // freeze/remove reading is meaningful (see the name-collision note
        // on this function and on `AttrId::Fill`).
        A::Fill => element.is_shape() || element.is_text_content() || element.is_animation(),
        A::FillOpacity | A::FillRule | A::PaintOrder => {
            element.is_shape() || element.is_text_content()
        }
        A::FloodColor | A::FloodOpacity => matches!(element, E::FeFlood | E::FeDropShadow),
        A::ImageRendering => matches!(element, E::Image | E::FeImage),
        A::LightingColor => matches!(element, E::FeDiffuseLighting | E::FeSpecularLighting),
        A::Marker | A::MarkerStart | A::MarkerMid | A::MarkerEnd => {
            matches!(element, E::Path | E::Line | E::Polyline | E::Polygon)
        }
        A::PointerEvents | A::VectorEffect => element.is_graphics(),
        A::ShapeRendering => element.is_shape(),
        A::StopColor | A::StopOpacity => matches!(element, E::Stop),
        A::Stroke
        | A::StrokeDasharray
        | A::StrokeDashoffset
        | A::StrokeLinecap
        | A::StrokeLinejoin
        | A::StrokeMiterlimit
        | A::StrokeOpacity
        | A::StrokeWidth => element.is_shape() || element.is_text_content(),
        A::Visibility => element.is_graphics() || element.is_text_content(),
        A::MaskType => matches!(element, E::Mask),
        A::D => matches!(element, E::Path | E::Glyph | E::MissingGlyph),

        // ---- 2.4 Geometry ----
        A::X | A::Y | A::Width | A::Height => {
            matches!(
                element,
                E::Rect
                    | E::Svg
                    | E::Use
                    | E::Image
                    | E::ForeignObject
                    | E::Mask
                    | E::Filter
                    | E::Symbol
                    | E::Pattern
            ) || element.is_filter_primitive()
        }
        A::R => matches!(element, E::Circle | E::RadialGradient),
        A::Rx | A::Ry => matches!(element, E::Rect | E::Ellipse),
        A::Cx | A::Cy => matches!(element, E::Circle | E::Ellipse | E::RadialGradient),
        A::X1 | A::Y1 | A::X2 | A::Y2 => matches!(element, E::Line | E::LinearGradient),
        A::Points => matches!(element, E::Polyline | E::Polygon),
        A::PathLength => matches!(element, E::Path),
        A::Dx | A::Dy => {
            matches!(
                element,
                E::Text | E::Tspan | E::TextPath | E::AltGlyph | E::FeOffset | E::FeDropShadow
            )
        }
        A::Rotate => matches!(element, E::Text | E::Tspan | E::AltGlyph | E::AnimateMotion),
        A::TextLength | A::LengthAdjust => {
            matches!(element, E::Text | E::Tspan | E::TextPath | E::AltGlyph)
        }
        A::Fr | A::Fx | A::Fy => matches!(element, E::RadialGradient),

        // ---- 2.6 Viewport ----
        A::ViewBox | A::PreserveAspectRatio => {
            matches!(
                element,
                E::Svg | E::Symbol | E::Marker | E::Pattern | E::View | E::Image
            )
        }
        A::ZoomAndPan => matches!(element, E::Svg | E::View),

        // ---- 2.5 href / xlink family: the catalog's own referencing-element list ----
        A::Href
        | A::XlinkHref
        | A::XlinkShow
        | A::XlinkActuate
        | A::XlinkType
        | A::XlinkTitle
        | A::XlinkArcrole
        | A::XlinkRole => is_href_referencing(element),

        // ---- Gradient / pattern ----
        A::GradientUnits | A::GradientTransform | A::SpreadMethod => {
            matches!(element, E::LinearGradient | E::RadialGradient)
        }
        A::Offset => matches!(
            element,
            E::Stop | E::FeFuncR | E::FeFuncG | E::FeFuncB | E::FeFuncA
        ),
        A::PatternUnits | A::PatternContentUnits | A::PatternTransform => {
            matches!(element, E::Pattern)
        }

        // ---- Marker ----
        A::MarkerUnits | A::MarkerWidth | A::MarkerHeight | A::Orient => {
            matches!(element, E::Marker)
        }
        A::RefX | A::RefY => matches!(element, E::Marker | E::Symbol),

        // ---- 2.8 Filter: common ----
        A::In | A::Result => element.is_filter_primitive(),
        A::In2 => matches!(element, E::FeBlend | E::FeComposite | E::FeDisplacementMap),
        A::StdDeviation => matches!(element, E::FeGaussianBlur | E::FeDropShadow),
        A::FilterUnits | A::PrimitiveUnits => matches!(element, E::Filter),

        // ---- 2.8 Filter: per-primitive ----
        A::Mode => matches!(element, E::FeBlend),
        A::Type => matches!(
            element,
            E::Script
                | E::Style
                | E::FeColorMatrix
                | E::FeFuncR
                | E::FeFuncG
                | E::FeFuncB
                | E::FeFuncA
                | E::FeTurbulence
                | E::AnimateTransform
                | E::A
        ),
        A::Values => matches!(
            element,
            E::FeColorMatrix
                | E::Animate
                | E::AnimateColor
                | E::AnimateMotion
                | E::AnimateTransform
        ),
        A::TableValues | A::Slope | A::Intercept | A::Amplitude | A::Exponent => {
            matches!(element, E::FeFuncR | E::FeFuncG | E::FeFuncB | E::FeFuncA)
        }
        A::Order
        | A::KernelMatrix
        | A::Divisor
        | A::Bias
        | A::TargetX
        | A::TargetY
        | A::PreserveAlpha => {
            matches!(element, E::FeConvolveMatrix)
        }
        // Permissive per catalog §8 open item 1: feGaussianBlur's edgeMode
        // is unconfirmed to exist at all; treat it as applying rather than
        // silently assuming it does not.
        A::EdgeMode => matches!(element, E::FeConvolveMatrix | E::FeGaussianBlur),
        A::KernelUnitLength => {
            matches!(
                element,
                E::FeConvolveMatrix | E::FeDiffuseLighting | E::FeSpecularLighting
            )
        }
        A::SurfaceScale => matches!(element, E::FeDiffuseLighting | E::FeSpecularLighting),
        A::DiffuseConstant => matches!(element, E::FeDiffuseLighting),
        A::SpecularConstant => matches!(element, E::FeSpecularLighting),
        A::SpecularExponent => matches!(element, E::FeSpecularLighting | E::FeSpotLight),
        A::Azimuth | A::Elevation => matches!(element, E::FeDistantLight),
        A::PointsAtX | A::PointsAtY | A::PointsAtZ | A::LimitingConeAngle => {
            matches!(element, E::FeSpotLight)
        }
        A::Scale | A::XChannelSelector | A::YChannelSelector => {
            matches!(element, E::FeDisplacementMap)
        }
        A::BaseFrequency | A::NumOctaves | A::Seed | A::StitchTiles => {
            matches!(element, E::FeTurbulence)
        }
        A::Radius => matches!(element, E::FeMorphology),
        A::Operator => matches!(element, E::FeMorphology | E::FeComposite),
        A::K1 | A::K2 | A::K3 | A::K4 => matches!(element, E::FeComposite),
        A::Z => matches!(element, E::FePointLight | E::FeSpotLight),

        // ---- 2.9 Animation ----
        // Target group: attributeName/attributeType are meaningful on
        // animate/set/animateTransform (catalog: "attributeType not
        // applicable to animateMotion" — attributeName is grouped with it
        // under the same three-element Target list).
        A::AttributeName | A::AttributeType => {
            matches!(element, E::Animate | E::Set | E::AnimateTransform)
        }
        // Value group: animate, animateColor, animateMotion,
        // animateTransform — `set` deliberately excluded (it sets one
        // discrete value; it does not interpolate `from`/`to`/`by`/
        // `values`/`calcMode`/`keyTimes`/`keySplines`).
        A::From | A::To | A::By | A::CalcMode | A::KeyTimes | A::KeySplines => {
            is_value_group_animation(element)
        }
        // Timing group minus `begin`: the five timed-duration animation
        // elements; `discard` has `begin` but none of the rest.
        A::Dur | A::End | A::Min | A::Max | A::Restart | A::RepeatCount | A::RepeatDur => {
            is_timed_animation(element)
        }
        A::Begin => is_timed_animation(element) || matches!(element, E::Discard),
        // Addition group: animate, animateMotion, animateTransform (not
        // `set`, not `animateColor`).
        A::Additive | A::Accumulate => {
            matches!(element, E::Animate | E::AnimateMotion | E::AnimateTransform)
        }
        A::KeyPoints | A::Origin => matches!(element, E::AnimateMotion),
        A::Path => matches!(element, E::AnimateMotion | E::TextPath),

        // ---- use / image / script ----
        A::Crossorigin => matches!(element, E::Use | E::Image | E::Script | E::FeImage),

        // ---- `a`-specific ----
        A::Target | A::Download | A::Rel | A::Hreflang => matches!(element, E::A),

        // ---- text / textPath ----
        A::StartOffset | A::Method | A::Spacing | A::Side => matches!(element, E::TextPath),

        // ---- clip / mask units ----
        A::ClipPathUnits => matches!(element, E::ClipPath),
        A::MaskUnits | A::MaskContentUnits => matches!(element, E::Mask),

        // ---- root-element attributes ----
        A::BaseProfile | A::Version => matches!(element, E::Svg),
    }
}

/// The catalog's own §2.2 element list for `requiredFeatures`/
/// `requiredExtensions`/`systemLanguage` (30 elements, transcribed from the
/// catalog's enumerated row rather than its "23 elements" summary count,
/// which does not match its own enumerated list).
fn is_conditional_processing_target(id: ElementId) -> bool {
    use ElementId as E;
    matches!(
        id,
        E::A | E::AltGlyph
            | E::Animate
            | E::AnimateColor
            | E::AnimateMotion
            | E::AnimateTransform
            | E::Circle
            | E::ClipPath
            | E::Cursor
            | E::Defs
            | E::Ellipse
            | E::ForeignObject
            | E::G
            | E::Image
            | E::Line
            | E::Mask
            | E::Path
            | E::Pattern
            | E::Polygon
            | E::Polyline
            | E::Rect
            | E::Set
            | E::Svg
            | E::Switch
            | E::Text
            | E::TextPath
            | E::Tref
            | E::Tspan
            | E::Use
            | E::View
    )
}

/// The catalog's §2.5 element list `xlink:href` (and, by SVG 2 extension,
/// bare `href`, and — permissively — the other legacy XLink attributes
/// that travel alongside it) is meaningful on: `a`, `altGlyph`,
/// `color-profile`, `cursor`, `feImage`, `filter`, `font-face-uri`,
/// `glyphRef`, `image`, `linearGradient`, `mpath`, `pattern`,
/// `radialGradient`, `script`, `textPath`, `use`, `animate`,
/// `animateColor`, `animateMotion`, `animateTransform`, `set`, `tref` (22
/// elements).
fn is_href_referencing(id: ElementId) -> bool {
    use ElementId as E;
    matches!(
        id,
        E::A | E::AltGlyph
            | E::ColorProfile
            | E::Cursor
            | E::FeImage
            | E::Filter
            | E::FontFaceUri
            | E::GlyphRef
            | E::Image
            | E::LinearGradient
            | E::Mpath
            | E::Pattern
            | E::RadialGradient
            | E::Script
            | E::TextPath
            | E::Use
            | E::Animate
            | E::AnimateColor
            | E::AnimateMotion
            | E::AnimateTransform
            | E::Set
            | E::Tref
    )
}

/// True for the 5 timed-duration animation elements: `animate`, `set`,
/// `animateMotion`, `animateTransform`, `animateColor`. Excludes `mpath`
/// (no timing of its own) and `discard` (only `begin`, handled separately).
fn is_timed_animation(id: ElementId) -> bool {
    use ElementId as E;
    matches!(
        id,
        E::Animate | E::Set | E::AnimateMotion | E::AnimateTransform | E::AnimateColor
    )
}

/// True for the 4 elements in the catalog's animation "Value" grouping:
/// `animate`, `animateColor`, `animateMotion`, `animateTransform`. Excludes
/// `set`, which sets one discrete value rather than interpolating.
fn is_value_group_animation(id: ElementId) -> bool {
    use ElementId as E;
    matches!(
        id,
        E::Animate | E::AnimateColor | E::AnimateMotion | E::AnimateTransform
    )
}

/// True for the elements the catalog treats as viewport-establishing for
/// `overflow`/`clip` applicability: `svg`, `symbol`, `image`, `marker`,
/// `pattern`, `foreignObject`.
fn is_viewport_establishing(id: ElementId) -> bool {
    use ElementId as E;
    matches!(
        id,
        E::Svg | E::Symbol | E::Image | E::Marker | E::Pattern | E::ForeignObject
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::names::attrs::AttrId as A;
    use crate::names::elements::ElementId as E;

    #[test]
    fn circle_geometry_attributes() {
        assert!(attribute_applies(E::Circle, A::R));
        assert!(attribute_applies(E::Circle, A::Cx));
        assert!(attribute_applies(E::Circle, A::Cy));
        assert!(!attribute_applies(E::Circle, A::D));
        assert!(!attribute_applies(E::Circle, A::Rx));
    }

    #[test]
    fn rect_geometry_attributes() {
        assert!(attribute_applies(E::Rect, A::Rx));
        assert!(attribute_applies(E::Rect, A::Ry));
        assert!(attribute_applies(E::Rect, A::X));
        assert!(attribute_applies(E::Rect, A::Width));
        assert!(!attribute_applies(E::Rect, A::R));
    }

    #[test]
    fn line_does_not_have_rx() {
        assert!(!attribute_applies(E::Line, A::Rx));
        assert!(attribute_applies(E::Line, A::X1));
        assert!(attribute_applies(E::Line, A::X2));
    }

    #[test]
    fn fill_applies_to_both_shapes_and_animation_elements() {
        // The documented `fill` name collision: both readings are true.
        assert!(attribute_applies(E::Rect, A::Fill));
        assert!(attribute_applies(E::Animate, A::Fill));
        assert!(attribute_applies(E::Text, A::Fill));
        assert!(
            !attribute_applies(E::Svg, A::Fill),
            "svg itself is neither a shape nor text content"
        );
    }

    #[test]
    fn core_attributes_apply_everywhere() {
        for &element in ElementId::ALL {
            assert!(attribute_applies(element, A::Id));
            assert!(attribute_applies(element, A::Class));
            assert!(attribute_applies(element, A::XmlLang));
        }
    }

    #[test]
    fn conditional_processing_matches_the_catalogs_list() {
        assert!(attribute_applies(E::Rect, A::RequiredFeatures));
        assert!(attribute_applies(E::Svg, A::SystemLanguage));
        assert!(attribute_applies(E::Use, A::RequiredExtensions));
        assert!(!attribute_applies(E::Stop, A::RequiredFeatures));
        assert!(!attribute_applies(E::LinearGradient, A::RequiredFeatures));
    }

    #[test]
    fn presentation_fill_does_not_apply_to_animate_element_itself_as_shape() {
        // Sanity check distinguishing the collision test above: `animate`
        // is not a shape or text content element, so it only picks up
        // `fill` through the animation-timing reading, not the paint one.
        assert!(!E::Animate.is_shape());
        assert!(!E::Animate.is_text_content());
        assert!(attribute_applies(E::Animate, A::Fill));
    }

    #[test]
    fn stroke_width_applies_to_shapes_and_text_not_containers() {
        assert!(attribute_applies(E::Rect, A::StrokeWidth));
        assert!(attribute_applies(E::Text, A::StrokeWidth));
        assert!(!attribute_applies(E::G, A::StrokeWidth));
        assert!(!attribute_applies(E::Svg, A::StrokeWidth));
    }

    #[test]
    fn marker_properties_exclude_rect_circle_ellipse() {
        assert!(attribute_applies(E::Path, A::MarkerStart));
        assert!(attribute_applies(E::Line, A::MarkerMid));
        assert!(attribute_applies(E::Polyline, A::MarkerEnd));
        assert!(attribute_applies(E::Polygon, A::Marker));
        assert!(!attribute_applies(E::Rect, A::MarkerStart));
        assert!(!attribute_applies(E::Circle, A::MarkerStart));
        assert!(!attribute_applies(E::Ellipse, A::MarkerStart));
    }

    #[test]
    fn stop_color_and_opacity_apply_only_to_stop() {
        assert!(attribute_applies(E::Stop, A::StopColor));
        assert!(attribute_applies(E::Stop, A::StopOpacity));
        assert!(!attribute_applies(E::LinearGradient, A::StopColor));
    }

    #[test]
    fn flood_attributes_apply_to_fe_flood_and_fe_drop_shadow() {
        assert!(attribute_applies(E::FeFlood, A::FloodColor));
        assert!(attribute_applies(E::FeDropShadow, A::FloodOpacity));
        assert!(!attribute_applies(E::FeGaussianBlur, A::FloodColor));
    }

    #[test]
    fn fe_gaussian_blur_edge_mode_is_permissively_true() {
        // Catalog §8 open item 1: existence unconfirmed, permissive choice.
        assert!(attribute_applies(E::FeGaussianBlur, A::EdgeMode));
        assert!(attribute_applies(E::FeConvolveMatrix, A::EdgeMode));
        assert!(!attribute_applies(E::FeMorphology, A::EdgeMode));
    }

    #[test]
    fn filter_primitive_common_attributes() {
        assert!(attribute_applies(E::FeGaussianBlur, A::In));
        assert!(attribute_applies(E::FeGaussianBlur, A::Result));
        assert!(attribute_applies(E::FeGaussianBlur, A::X));
        assert!(attribute_applies(E::FeBlend, A::In2));
        assert!(attribute_applies(E::FeComposite, A::In2));
        assert!(attribute_applies(E::FeDisplacementMap, A::In2));
        assert!(!attribute_applies(E::FeGaussianBlur, A::In2));
        assert!(!attribute_applies(E::Rect, A::In));
    }

    #[test]
    fn fe_composite_arithmetic_coefficients() {
        assert!(attribute_applies(E::FeComposite, A::K1));
        assert!(attribute_applies(E::FeComposite, A::K4));
        assert!(attribute_applies(E::FeComposite, A::Operator));
        assert!(attribute_applies(E::FeMorphology, A::Operator));
        assert!(attribute_applies(E::FeMorphology, A::Radius));
        assert!(!attribute_applies(E::FeMorphology, A::K1));
    }

    #[test]
    fn lighting_primitive_attributes() {
        assert!(attribute_applies(E::FeDiffuseLighting, A::SurfaceScale));
        assert!(attribute_applies(E::FeDiffuseLighting, A::DiffuseConstant));
        assert!(attribute_applies(
            E::FeSpecularLighting,
            A::SpecularConstant
        ));
        assert!(attribute_applies(
            E::FeSpecularLighting,
            A::SpecularExponent
        ));
        assert!(attribute_applies(E::FeSpotLight, A::SpecularExponent));
        assert!(attribute_applies(E::FeDistantLight, A::Azimuth));
        assert!(attribute_applies(E::FeDistantLight, A::Elevation));
        assert!(attribute_applies(E::FeSpotLight, A::PointsAtX));
        assert!(attribute_applies(E::FeSpotLight, A::LimitingConeAngle));
        assert!(attribute_applies(E::FePointLight, A::Z));
        assert!(!attribute_applies(E::FeDiffuseLighting, A::Azimuth));
    }

    #[test]
    fn gradient_and_pattern_unit_attributes() {
        assert!(attribute_applies(E::LinearGradient, A::GradientUnits));
        assert!(attribute_applies(E::RadialGradient, A::SpreadMethod));
        assert!(attribute_applies(E::RadialGradient, A::Fr));
        assert!(attribute_applies(E::RadialGradient, A::Fx));
        assert!(attribute_applies(E::Pattern, A::PatternUnits));
        assert!(attribute_applies(E::Stop, A::Offset));
        assert!(attribute_applies(E::FeFuncR, A::Offset));
        assert!(!attribute_applies(E::Pattern, A::GradientUnits));
    }

    #[test]
    fn marker_reference_point_attributes() {
        assert!(attribute_applies(E::Marker, A::RefX));
        assert!(attribute_applies(E::Symbol, A::RefX));
        assert!(attribute_applies(E::Marker, A::MarkerWidth));
        assert!(attribute_applies(E::Marker, A::Orient));
        assert!(!attribute_applies(E::Pattern, A::RefX));
    }

    #[test]
    fn animation_groupings_apply_to_the_right_elements() {
        assert!(attribute_applies(E::Animate, A::AttributeName));
        assert!(attribute_applies(E::Set, A::AttributeName));
        assert!(attribute_applies(E::AnimateTransform, A::AttributeName));
        assert!(!attribute_applies(E::AnimateMotion, A::AttributeType));

        assert!(attribute_applies(E::Animate, A::From));
        assert!(attribute_applies(E::AnimateColor, A::To));
        assert!(
            !attribute_applies(E::Set, A::From),
            "set does not interpolate"
        );

        assert!(attribute_applies(E::Animate, A::Dur));
        assert!(attribute_applies(E::Set, A::Dur));
        assert!(!attribute_applies(E::Mpath, A::Dur));
        assert!(!attribute_applies(E::Discard, A::Dur));
        assert!(attribute_applies(E::Discard, A::Begin));

        assert!(attribute_applies(E::Animate, A::Additive));
        assert!(attribute_applies(E::AnimateMotion, A::Additive));
        assert!(!attribute_applies(E::Set, A::Additive));
        assert!(!attribute_applies(E::AnimateColor, A::Additive));

        assert!(attribute_applies(E::AnimateMotion, A::KeyPoints));
        assert!(attribute_applies(E::AnimateMotion, A::Path));
        assert!(attribute_applies(E::TextPath, A::Path));
    }

    #[test]
    fn text_path_specific_attributes() {
        assert!(attribute_applies(E::TextPath, A::StartOffset));
        assert!(attribute_applies(E::TextPath, A::Method));
        assert!(attribute_applies(E::TextPath, A::Spacing));
        assert!(attribute_applies(E::TextPath, A::Side));
        assert!(!attribute_applies(E::Text, A::StartOffset));
    }

    #[test]
    fn a_specific_attributes() {
        assert!(attribute_applies(E::A, A::Target));
        assert!(attribute_applies(E::A, A::Download));
        assert!(attribute_applies(E::A, A::Rel));
        assert!(attribute_applies(E::A, A::Hreflang));
        assert!(!attribute_applies(E::Use, A::Target));
    }

    #[test]
    fn units_attributes_are_scoped_to_their_element() {
        assert!(attribute_applies(E::ClipPath, A::ClipPathUnits));
        assert!(attribute_applies(E::Mask, A::MaskUnits));
        assert!(attribute_applies(E::Mask, A::MaskContentUnits));
        assert!(!attribute_applies(E::Pattern, A::ClipPathUnits));
    }

    #[test]
    fn root_element_attributes_apply_only_to_svg() {
        assert!(attribute_applies(E::Svg, A::BaseProfile));
        assert!(attribute_applies(E::Svg, A::Version));
        assert!(attribute_applies(E::Svg, A::ZoomAndPan));
        assert!(attribute_applies(E::View, A::ZoomAndPan));
        assert!(!attribute_applies(E::G, A::BaseProfile));
    }

    #[test]
    fn href_referencing_matches_the_catalogs_svg1_1_list() {
        for element in [
            E::A,
            E::AltGlyph,
            E::ColorProfile,
            E::Cursor,
            E::FeImage,
            E::Filter,
            E::FontFaceUri,
            E::GlyphRef,
            E::Image,
            E::LinearGradient,
            E::Mpath,
            E::Pattern,
            E::RadialGradient,
            E::Script,
            E::TextPath,
            E::Use,
            E::Animate,
            E::AnimateColor,
            E::AnimateMotion,
            E::AnimateTransform,
            E::Set,
            E::Tref,
        ] {
            assert!(
                attribute_applies(element, A::XlinkHref),
                "{element:?} should accept xlink:href"
            );
            assert!(
                attribute_applies(element, A::Href),
                "{element:?} should accept href"
            );
        }
        assert!(!attribute_applies(E::Rect, A::XlinkHref));
    }

    #[test]
    fn every_element_attribute_pair_has_a_defined_answer() {
        // Sanity: attribute_applies must not panic for any pair in the
        // full 88x216 space.
        for &element in ElementId::ALL {
            for &attr in AttrId::ALL {
                let _ = attribute_applies(element, attr);
            }
        }
    }
}
