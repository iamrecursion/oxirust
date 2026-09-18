//! Presentation properties: SVG's fixed catalog of CSS-styleable properties,
//! their inheritance behavior, and their initial values.
//!
//! [`PropertyId`] enumerates every property transcribed from the SVG 1.1
//! `propidx.html` property index (specification catalog §2.3, first table)
//! together with the SVG 2 additions the same section lists that carry a
//! single specification-wide initial value (`paint-order`, `vector-effect`,
//! `transform`, `transform-origin`, `transform-box`, `text-overflow`,
//! `white-space`, `mask-type`, `isolation`, `mix-blend-mode`, and `d`). For
//! each property this type answers, with no element or attribute value in
//! sight:
//!
//! * [`PropertyId::as_str`] / [`PropertyId::from_name`] — its canonical
//!   lowercase CSS name, and parsing that name back (ASCII case-insensitive,
//!   matching CSS's own case-folding of property names).
//! * [`PropertyId::inherited`] — whether an element that does not specify
//!   the property takes its computed value from the nearest ancestor that
//!   does, per the cascade the catalog describes in §6.4.
//! * [`PropertyId::initial_value`] — the value used when nothing is
//!   inherited either: the bottom of that same cascade.
//! * [`PropertyId::is_animatable`] and
//!   [`PropertyId::presentation_attribute_allowed`] — two further
//!   per-property facts the catalog records.
//!
//! # Values stay untyped here
//!
//! Every value handed back by this module is a plain string: an initial
//! value here, a declaration's value in [`crate::style::attr`], a
//! stylesheet rule's declarations in [`crate::style::css`]. Turning
//! `"black"` into a color, applying the cascade to a real element, and
//! resolving `inherit`/percentages/`currentColor` against a computed parent
//! value are cascade-layer responsibilities built on top of this table, not
//! this module's job.
//!
//! # Deliberately unmodeled rows
//!
//! One row the catalog itself lists has no [`PropertyId`] variant: the SVG 2
//! "geometry properties" row (`x`, `y`, `width`, `height`, `rx`, `ry`, `cx`,
//! `cy`, `r`, `x1`, `y1`, `x2`, `y2`) gives its initial value as "per-element"
//! rather than one value for the whole property, which does not fit
//! [`PropertyId::initial_value`]'s `&'static str` return type. These names
//! remain attribute-level concerns for another module.
//!
//! # A row sourced outside the catalog: `font-kerning`
//!
//! [`PropertyId::FontKerning`] is [`PropertyId::ALL`]'s one property not
//! transcribed from the specification catalog: the catalog names
//! `font-kerning` (SVG 2's replacement for the removed `kerning` property)
//! only in its inherited-property prose, never as its own tabulated row with
//! an initial value. [`names::attrs::AttrId::FontKerning`](crate::names::attrs::AttrId::FontKerning)
//! is nonetheless a real, authorable presentation attribute
//! (`tests/consistency.rs` pins the two catalogs' agreement on this point),
//! so this module fills the row in from CSS Fonts Module Level 3/4 directly
//! — initial value `auto`, inherited, and (by the same general-CSS-convention
//! inference this module already applies to the SVG 2 additions above)
//! discretely animatable — rather than leaving a real, authorable attribute
//! with no way to ask the typed cascade for its value at all.
//!
//! # Known catalog discrepancy: `d`
//!
//! The catalog's property index marks `d` (SVG 2's path data as a
//! presentation property) as inherited. Published SVG 2 drafts describe `d`
//! as not inherited. [`PropertyId::D`] follows the catalog literally, since
//! that is the source of truth this module is built against; see that
//! variant's own documentation.
//!
//! # A gap in the source table: animatability of the SVG 2 additions
//!
//! The catalog's SVG 2 table has no "Animatable" column. The eleven
//! properties in that table report [`is_animatable`](PropertyId::is_animatable)
//! as `true` here, on the general CSS convention that a property is at
//! least discretely animatable unless a specification says otherwise; this
//! is an inference beyond the catalog, not a transcription of it.

use std::fmt;

/// One of the presentation properties this crate's specification catalog
/// tabulates: the CSS-styleable properties SVG defines, reachable through a
/// presentation attribute, the `style` attribute, or a matched stylesheet
/// rule.
///
/// See the [module documentation](self) for what this type does and does
/// not model.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum PropertyId {
    // SVG 1.1 `propidx.html` property index (specification catalog §2.3,
    // first table), in the order the catalog transcribes them.
    /// The `alignment-baseline` property: text alignment relative to the parent element's dominant baseline. Initial value `see baseline algorithm`; not inherited.
    AlignmentBaseline,
    /// The `baseline-shift` property: shifts the dominant baseline, e.g. for subscripts and superscripts. Initial value `baseline`; not inherited.
    BaselineShift,
    /// The `clip` property: a legacy CSS2 rectangular clip region, deprecated in favor of `clip-path`. Initial value `auto`; not inherited.
    Clip,
    /// The `clip-path` property: references or defines the clipping path applied to the element. Initial value `none`; not inherited.
    ClipPath,
    /// The `clip-rule` property: the fill rule used to determine the interior of a `clipPath`'s child shapes. Initial value `nonzero`; inherited.
    ClipRule,
    /// The `color` property: the color that `currentColor` resolves to. Initial value `UA-dependent`; inherited.
    Color,
    /// The `color-interpolation` property: the color space used for color interpolation and compositing. Initial value `sRGB`; inherited.
    ColorInterpolation,
    /// The `color-interpolation-filters` property: the color space used for filter-primitive color operations. Initial value `linearRGB`; inherited.
    ColorInterpolationFilters,
    /// The `color-profile` property (SVG 1.1 only, removed in SVG 2): the color profile applied to a raster image. Initial value `auto`; inherited.
    ColorProfile,
    /// The `color-rendering` property: a rendering-quality hint for color choices. Initial value `auto`; inherited.
    ColorRendering,
    /// The `cursor` property: the cursor displayed while the pointer is over the element. Initial value `auto`; inherited.
    Cursor,
    /// The `direction` property: the base bidirectional-text direction. Initial value `ltr`; inherited.
    Direction,
    /// The `display` property: whether and how the element generates a box. Initial value `inline`; not inherited.
    Display,
    /// The `dominant-baseline` property: the dominant baseline of the current text chunk. Initial value `auto`; not inherited.
    DominantBaseline,
    /// The `enable-background` property (SVG 1.1 only, removed in SVG 2): whether a background image buffer accumulates for later filter primitives. Initial value `accumulate`; not inherited.
    EnableBackground,
    /// The `fill` property: the paint used to fill the interior of a shape or glyph. Initial value `black`; inherited.
    Fill,
    /// The `fill-opacity` property: the opacity of the fill paint. Initial value `1`; inherited.
    FillOpacity,
    /// The `fill-rule` property: the algorithm used to determine the interior of a shape. Initial value `nonzero`; inherited.
    FillRule,
    /// The `filter` property: references the filter effect applied to the element. Initial value `none`; not inherited.
    Filter,
    /// The `flood-color` property: the color a `feFlood` or `feDropShadow` primitive floods with. Initial value `black`; not inherited.
    FloodColor,
    /// The `flood-opacity` property: the opacity a `feFlood` or `feDropShadow` primitive floods with. Initial value `1`; not inherited.
    FloodOpacity,
    /// The `font` shorthand property. Initial value `individual` (each longhand keeps its own initial value); inherited.
    Font,
    /// The `font-family` property: the prioritized list of font family names. Initial value `UA-dependent`; inherited.
    FontFamily,
    /// The `font-size` property: the size of the font. Initial value `medium`; inherited.
    FontSize,
    /// The `font-size-adjust` property: preserves x-height when the preferred font family is unavailable. Initial value `none`; inherited.
    FontSizeAdjust,
    /// The `font-stretch` property: selects a normal, condensed, or expanded font face. Initial value `normal`; inherited.
    FontStretch,
    /// The `font-style` property: selects a normal, italic, or oblique font face. Initial value `normal`; inherited.
    FontStyle,
    /// The `font-variant` property: selects small-caps and other variant glyph forms. Initial value `normal`; inherited.
    FontVariant,
    /// The `font-weight` property: selects a font weight. Initial value `normal`; inherited.
    FontWeight,
    /// The `glyph-orientation-horizontal` property: glyph orientation within horizontal text. Initial value `0deg`; inherited.
    GlyphOrientationHorizontal,
    /// The `glyph-orientation-vertical` property: glyph orientation within vertical text. Initial value `auto`; inherited.
    GlyphOrientationVertical,
    /// The `image-rendering` property: a rendering-quality hint for image scaling. Initial value `auto`; inherited.
    ImageRendering,
    /// The `kerning` property (SVG 1.1 only, replaced by the CSS `font-kerning` property): inter-glyph spacing adjustment from font kerning tables. Initial value `auto`; inherited.
    Kerning,
    /// The `letter-spacing` property: additional spacing between typographic characters. Initial value `normal`; inherited.
    LetterSpacing,
    /// The `lighting-color` property: the light color used by `feDiffuseLighting` and `feSpecularLighting`. Initial value `white`; not inherited.
    LightingColor,
    /// The `marker` shorthand property, setting `marker-start`, `marker-mid`, and `marker-end` together. Initial value `individual`; inherited.
    Marker,
    /// The `marker-end` property: the marker drawn at the final vertex of a path-like shape. Initial value `none`; inherited.
    MarkerEnd,
    /// The `marker-mid` property: the marker drawn at every interior vertex of a path-like shape. Initial value `none`; inherited.
    MarkerMid,
    /// The `marker-start` property: the marker drawn at the first vertex of a path-like shape. Initial value `none`; inherited.
    MarkerStart,
    /// The `mask` property: references the mask applied to the element. Initial value `none`; not inherited.
    Mask,
    /// The `opacity` property: the object (group) opacity applied to the element as a whole. Initial value `1`; not inherited.
    Opacity,
    /// The `overflow` property: whether content outside the element's viewport is clipped. Initial value `see prose (visible for the outermost svg element, hidden otherwise)`; not inherited.
    Overflow,
    /// The `pointer-events` property: which pointer-event interactions the element participates in. Initial value `visiblePainted`; inherited.
    PointerEvents,
    /// The `shape-rendering` property: a rendering-quality hint for geometric shapes. Initial value `auto`; inherited.
    ShapeRendering,
    /// The `stop-color` property: the color of a gradient `stop`. Initial value `black`; not inherited.
    StopColor,
    /// The `stop-opacity` property: the opacity of a gradient `stop`. Initial value `1`; not inherited.
    StopOpacity,
    /// The `stroke` property: the paint used to stroke the outline of a shape or glyph. Initial value `none`; inherited.
    Stroke,
    /// The `stroke-dasharray` property: the dash pattern used when stroking a shape's outline. Initial value `none`; inherited.
    StrokeDasharray,
    /// The `stroke-dashoffset` property: the offset into the dash pattern at which a stroke begins. Initial value `0`; inherited.
    StrokeDashoffset,
    /// The `stroke-linecap` property: the shape drawn at the open ends of a stroked path. Initial value `butt`; inherited.
    StrokeLinecap,
    /// The `stroke-linejoin` property: the shape drawn at the corners of a stroked path. Initial value `miter`; inherited.
    StrokeLinejoin,
    /// The `stroke-miterlimit` property: the limit on the ratio of miter length to stroke width before a join falls back to a bevel. Initial value `4`; inherited.
    StrokeMiterlimit,
    /// The `stroke-opacity` property: the opacity of the stroke paint. Initial value `1`; inherited.
    StrokeOpacity,
    /// The `stroke-width` property: the width of the stroke. Initial value `1`; inherited.
    StrokeWidth,
    /// The `text-anchor` property: how a text chunk is aligned relative to its anchor point. Initial value `start`; inherited.
    TextAnchor,
    /// The `text-decoration` property: underline, overline, and line-through decorations. Initial value `none`; not inherited.
    TextDecoration,
    /// The `text-rendering` property: a rendering-quality hint for text. Initial value `auto`; inherited.
    TextRendering,
    /// The `unicode-bidi` property: how the element participates in the Unicode bidirectional algorithm. Initial value `normal`; not inherited.
    UnicodeBidi,
    /// The `visibility` property: whether the element's rendered output is drawn. Initial value `visible`; inherited.
    Visibility,
    /// The `word-spacing` property: additional spacing between words. Initial value `normal`; inherited.
    WordSpacing,
    /// The `writing-mode` property: the block flow direction used for text layout. Initial value `lr-tb` (the SVG 1.1 value; SVG 2 uses the CSS Writing Modes value `horizontal-tb` for the same default); inherited.
    WritingMode,

    // SVG 2 additions the catalog lists with a single specification-wide
    // initial value (specification catalog §2.3, second table).
    /// The `paint-order` property (SVG 2): the relative painting order among fill, stroke, and markers. Initial value `normal`; inherited.
    PaintOrder,
    /// The `vector-effect` property (SVG 2): rendering effects that resist the element's geometric transform, such as a non-scaling stroke. Initial value `none`; not inherited.
    VectorEffect,
    /// The `transform` property (SVG 2): the element's geometric transform, using SVG's own transform-list grammar rather than the CSS `transform` property's grammar. Initial value `none`; not inherited.
    Transform,
    /// The `transform-origin` property (SVG 2, from CSS Transforms Level 1): the origin `transform` is applied around. Initial value `0 0`; not inherited.
    TransformOrigin,
    /// The `transform-box` property (SVG 2, from CSS Transforms Level 1): the reference box `transform` and `transform-origin` are resolved against. Initial value `view-box`; not inherited.
    TransformBox,
    /// The `text-overflow` property (SVG 2, from CSS Overflow): how clipped inline text signals that it was truncated. Initial value `clip`; not inherited.
    TextOverflow,
    /// The `white-space` property (SVG 2, from CSS Text): how white space inside text content is collapsed and wrapped. Initial value `normal`; inherited.
    WhiteSpace,
    /// The `mask-type` property (SVG 2, from CSS Masking Level 1): whether a `mask` element's content is interpreted as luminance or alpha. Initial value `luminance`; not inherited.
    MaskType,
    /// The `isolation` property (SVG 2, from CSS Compositing and Blending): whether the element creates a new stacking context for compositing purposes. Initial value `auto`; not inherited.
    Isolation,
    /// The `mix-blend-mode` property (SVG 2, from CSS Compositing and Blending): how the element's content blends with the content behind it. Initial value `normal`; not inherited.
    MixBlendMode,
    /// The `d` property (SVG 2): the geometric path data of a `path` element, promoted from an XML-only attribute to a presentation property. Initial value `none`; inherited per the specification catalog, though published SVG 2 drafts describe `d` as not inherited — see the module documentation.
    D,

    // Sourced from CSS Fonts Module Level 3/4 directly rather than the
    // specification catalog — see the module documentation.
    /// The `font-kerning` property (CSS Fonts Module Level 3/4, SVG 2's replacement for the removed `kerning` property): whether the font's built-in kerning pairs are applied. Initial value `auto`; inherited.
    FontKerning,
}

/// The specification-catalog facts about one [`PropertyId`]: its canonical
/// name, initial value, inheritance flag, and animatability. Kept as a
/// private lookup table so every public accessor reads from one place.
struct PropertyRow {
    name: &'static str,
    initial: &'static str,
    inherited: bool,
    animatable: bool,
}

impl PropertyId {
    /// Every property this type models, in the order the specification
    /// catalog transcribes them: the SVG 1.1 property index (§2.3, first
    /// table) followed by the SVG 2 additions the same section lists
    /// (§2.3, second table).
    pub const ALL: &'static [PropertyId] = &[
        Self::AlignmentBaseline,
        Self::BaselineShift,
        Self::Clip,
        Self::ClipPath,
        Self::ClipRule,
        Self::Color,
        Self::ColorInterpolation,
        Self::ColorInterpolationFilters,
        Self::ColorProfile,
        Self::ColorRendering,
        Self::Cursor,
        Self::Direction,
        Self::Display,
        Self::DominantBaseline,
        Self::EnableBackground,
        Self::Fill,
        Self::FillOpacity,
        Self::FillRule,
        Self::Filter,
        Self::FloodColor,
        Self::FloodOpacity,
        Self::Font,
        Self::FontFamily,
        Self::FontSize,
        Self::FontSizeAdjust,
        Self::FontStretch,
        Self::FontStyle,
        Self::FontVariant,
        Self::FontWeight,
        Self::GlyphOrientationHorizontal,
        Self::GlyphOrientationVertical,
        Self::ImageRendering,
        Self::Kerning,
        Self::LetterSpacing,
        Self::LightingColor,
        Self::Marker,
        Self::MarkerEnd,
        Self::MarkerMid,
        Self::MarkerStart,
        Self::Mask,
        Self::Opacity,
        Self::Overflow,
        Self::PointerEvents,
        Self::ShapeRendering,
        Self::StopColor,
        Self::StopOpacity,
        Self::Stroke,
        Self::StrokeDasharray,
        Self::StrokeDashoffset,
        Self::StrokeLinecap,
        Self::StrokeLinejoin,
        Self::StrokeMiterlimit,
        Self::StrokeOpacity,
        Self::StrokeWidth,
        Self::TextAnchor,
        Self::TextDecoration,
        Self::TextRendering,
        Self::UnicodeBidi,
        Self::Visibility,
        Self::WordSpacing,
        Self::WritingMode,
        Self::PaintOrder,
        Self::VectorEffect,
        Self::Transform,
        Self::TransformOrigin,
        Self::TransformBox,
        Self::TextOverflow,
        Self::WhiteSpace,
        Self::MaskType,
        Self::Isolation,
        Self::MixBlendMode,
        Self::D,
        Self::FontKerning,
    ];

    /// Parses a CSS property name into the property it names.
    ///
    /// The match is ASCII case-insensitive, mirroring CSS's own case-folding
    /// of property names (`"Fill"`, `"FILL"`, and `"fill"` all resolve to
    /// [`PropertyId::Fill`]). Returns [`None`] for any name this crate does
    /// not model — see the [module documentation](self) for the properties
    /// that are deliberately excluded.
    #[must_use]
    pub fn from_name(name: &str) -> Option<Self> {
        Self::ALL
            .iter()
            .copied()
            .find(|property| property.as_str().eq_ignore_ascii_case(name))
    }

    /// The canonical lowercase CSS name of the property, e.g. `"stroke-width"`.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        self.row().name
    }

    /// Reports whether an element that does not specify this property takes
    /// its computed value from the nearest ancestor that does, per the
    /// specification catalog's inherited-property list (§6.5).
    #[must_use]
    pub const fn inherited(self) -> bool {
        self.row().inherited
    }

    /// The property's specification-defined default value, exactly as the
    /// catalog's property index (§2.3) states it.
    ///
    /// A handful of rows describe the default in prose rather than as a
    /// single concrete value (`"UA-dependent"`, `"individual"`, `"see
    /// baseline algorithm"`, ...) — see the [module documentation](self).
    /// Callers that need a concrete resolved default for one of those rows
    /// must supply their own UA/element-specific policy; this method
    /// reports the catalog text verbatim.
    #[must_use]
    pub const fn initial_value(self) -> &'static str {
        self.row().initial
    }

    /// Reports whether the property can be the target of a SMIL
    /// (`<animate>`) or CSS animation/transition, per the specification
    /// catalog.
    #[must_use]
    pub const fn is_animatable(self) -> bool {
        self.row().animatable
    }

    /// Reports whether the property may also be written as a presentation
    /// attribute — an XML attribute on the element itself with the same
    /// name as the property — rather than only through `style=` or a
    /// stylesheet rule.
    ///
    /// The specification catalog names no exceptions among the properties
    /// this type models, so this currently always returns `true`.
    #[must_use]
    pub const fn presentation_attribute_allowed(self) -> bool {
        true
    }

    const fn row(self) -> PropertyRow {
        match self {
            Self::AlignmentBaseline => PropertyRow {
                name: "alignment-baseline",
                initial: "see baseline algorithm",
                inherited: false,
                animatable: true,
            },
            Self::BaselineShift => PropertyRow {
                name: "baseline-shift",
                initial: "baseline",
                inherited: false,
                animatable: true,
            },
            Self::Clip => PropertyRow {
                name: "clip",
                initial: "auto",
                inherited: false,
                animatable: true,
            },
            Self::ClipPath => PropertyRow {
                name: "clip-path",
                initial: "none",
                inherited: false,
                animatable: true,
            },
            Self::ClipRule => PropertyRow {
                name: "clip-rule",
                initial: "nonzero",
                inherited: true,
                animatable: true,
            },
            Self::Color => PropertyRow {
                name: "color",
                initial: "UA-dependent",
                inherited: true,
                animatable: true,
            },
            Self::ColorInterpolation => PropertyRow {
                name: "color-interpolation",
                initial: "sRGB",
                inherited: true,
                animatable: true,
            },
            Self::ColorInterpolationFilters => PropertyRow {
                name: "color-interpolation-filters",
                initial: "linearRGB",
                inherited: true,
                animatable: true,
            },
            Self::ColorProfile => PropertyRow {
                name: "color-profile",
                initial: "auto",
                inherited: true,
                animatable: true,
            },
            Self::ColorRendering => PropertyRow {
                name: "color-rendering",
                initial: "auto",
                inherited: true,
                animatable: true,
            },
            Self::Cursor => PropertyRow {
                name: "cursor",
                initial: "auto",
                inherited: true,
                animatable: true,
            },
            Self::Direction => PropertyRow {
                name: "direction",
                initial: "ltr",
                inherited: true,
                animatable: false,
            },
            Self::Display => PropertyRow {
                name: "display",
                initial: "inline",
                inherited: false,
                animatable: true,
            },
            Self::DominantBaseline => PropertyRow {
                name: "dominant-baseline",
                initial: "auto",
                inherited: false,
                animatable: true,
            },
            Self::EnableBackground => PropertyRow {
                name: "enable-background",
                initial: "accumulate",
                inherited: false,
                animatable: false,
            },
            Self::Fill => PropertyRow {
                name: "fill",
                initial: "black",
                inherited: true,
                animatable: true,
            },
            Self::FillOpacity => PropertyRow {
                name: "fill-opacity",
                initial: "1",
                inherited: true,
                animatable: true,
            },
            Self::FillRule => PropertyRow {
                name: "fill-rule",
                initial: "nonzero",
                inherited: true,
                animatable: true,
            },
            Self::Filter => PropertyRow {
                name: "filter",
                initial: "none",
                inherited: false,
                animatable: true,
            },
            Self::FloodColor => PropertyRow {
                name: "flood-color",
                initial: "black",
                inherited: false,
                animatable: true,
            },
            Self::FloodOpacity => PropertyRow {
                name: "flood-opacity",
                initial: "1",
                inherited: false,
                animatable: true,
            },
            Self::Font => PropertyRow {
                name: "font",
                initial: "individual",
                inherited: true,
                animatable: true,
            },
            Self::FontFamily => PropertyRow {
                name: "font-family",
                initial: "UA-dependent",
                inherited: true,
                animatable: true,
            },
            Self::FontSize => PropertyRow {
                name: "font-size",
                initial: "medium",
                inherited: true,
                animatable: true,
            },
            Self::FontSizeAdjust => PropertyRow {
                name: "font-size-adjust",
                initial: "none",
                inherited: true,
                animatable: true,
            },
            Self::FontStretch => PropertyRow {
                name: "font-stretch",
                initial: "normal",
                inherited: true,
                animatable: true,
            },
            Self::FontStyle => PropertyRow {
                name: "font-style",
                initial: "normal",
                inherited: true,
                animatable: true,
            },
            Self::FontVariant => PropertyRow {
                name: "font-variant",
                initial: "normal",
                inherited: true,
                animatable: true,
            },
            Self::FontWeight => PropertyRow {
                name: "font-weight",
                initial: "normal",
                inherited: true,
                animatable: true,
            },
            Self::GlyphOrientationHorizontal => PropertyRow {
                name: "glyph-orientation-horizontal",
                initial: "0deg",
                inherited: true,
                animatable: false,
            },
            Self::GlyphOrientationVertical => PropertyRow {
                name: "glyph-orientation-vertical",
                initial: "auto",
                inherited: true,
                animatable: false,
            },
            Self::ImageRendering => PropertyRow {
                name: "image-rendering",
                initial: "auto",
                inherited: true,
                animatable: true,
            },
            Self::Kerning => PropertyRow {
                name: "kerning",
                initial: "auto",
                inherited: true,
                animatable: true,
            },
            Self::LetterSpacing => PropertyRow {
                name: "letter-spacing",
                initial: "normal",
                inherited: true,
                animatable: true,
            },
            Self::LightingColor => PropertyRow {
                name: "lighting-color",
                initial: "white",
                inherited: false,
                animatable: true,
            },
            Self::Marker => PropertyRow {
                name: "marker",
                initial: "individual",
                inherited: true,
                animatable: true,
            },
            Self::MarkerEnd => PropertyRow {
                name: "marker-end",
                initial: "none",
                inherited: true,
                animatable: true,
            },
            Self::MarkerMid => PropertyRow {
                name: "marker-mid",
                initial: "none",
                inherited: true,
                animatable: true,
            },
            Self::MarkerStart => PropertyRow {
                name: "marker-start",
                initial: "none",
                inherited: true,
                animatable: true,
            },
            Self::Mask => PropertyRow {
                name: "mask",
                initial: "none",
                inherited: false,
                animatable: true,
            },
            Self::Opacity => PropertyRow {
                name: "opacity",
                initial: "1",
                inherited: false,
                animatable: true,
            },
            Self::Overflow => PropertyRow {
                name: "overflow",
                initial: "see prose (visible for the outermost svg element, hidden otherwise)",
                inherited: false,
                animatable: true,
            },
            Self::PointerEvents => PropertyRow {
                name: "pointer-events",
                initial: "visiblePainted",
                inherited: true,
                animatable: true,
            },
            Self::ShapeRendering => PropertyRow {
                name: "shape-rendering",
                initial: "auto",
                inherited: true,
                animatable: true,
            },
            Self::StopColor => PropertyRow {
                name: "stop-color",
                initial: "black",
                inherited: false,
                animatable: true,
            },
            Self::StopOpacity => PropertyRow {
                name: "stop-opacity",
                initial: "1",
                inherited: false,
                animatable: true,
            },
            Self::Stroke => PropertyRow {
                name: "stroke",
                initial: "none",
                inherited: true,
                animatable: true,
            },
            Self::StrokeDasharray => PropertyRow {
                name: "stroke-dasharray",
                initial: "none",
                inherited: true,
                animatable: true,
            },
            Self::StrokeDashoffset => PropertyRow {
                name: "stroke-dashoffset",
                initial: "0",
                inherited: true,
                animatable: true,
            },
            Self::StrokeLinecap => PropertyRow {
                name: "stroke-linecap",
                initial: "butt",
                inherited: true,
                animatable: true,
            },
            Self::StrokeLinejoin => PropertyRow {
                name: "stroke-linejoin",
                initial: "miter",
                inherited: true,
                animatable: true,
            },
            Self::StrokeMiterlimit => PropertyRow {
                name: "stroke-miterlimit",
                initial: "4",
                inherited: true,
                animatable: true,
            },
            Self::StrokeOpacity => PropertyRow {
                name: "stroke-opacity",
                initial: "1",
                inherited: true,
                animatable: true,
            },
            Self::StrokeWidth => PropertyRow {
                name: "stroke-width",
                initial: "1",
                inherited: true,
                animatable: true,
            },
            Self::TextAnchor => PropertyRow {
                name: "text-anchor",
                initial: "start",
                inherited: true,
                animatable: true,
            },
            Self::TextDecoration => PropertyRow {
                name: "text-decoration",
                initial: "none",
                inherited: false,
                animatable: true,
            },
            Self::TextRendering => PropertyRow {
                name: "text-rendering",
                initial: "auto",
                inherited: true,
                animatable: true,
            },
            Self::UnicodeBidi => PropertyRow {
                name: "unicode-bidi",
                initial: "normal",
                inherited: false,
                animatable: false,
            },
            Self::Visibility => PropertyRow {
                name: "visibility",
                initial: "visible",
                inherited: true,
                animatable: true,
            },
            Self::WordSpacing => PropertyRow {
                name: "word-spacing",
                initial: "normal",
                inherited: true,
                animatable: true,
            },
            Self::WritingMode => PropertyRow {
                name: "writing-mode",
                initial: "lr-tb",
                inherited: true,
                animatable: false,
            },
            Self::PaintOrder => PropertyRow {
                name: "paint-order",
                initial: "normal",
                inherited: true,
                animatable: true,
            },
            Self::VectorEffect => PropertyRow {
                name: "vector-effect",
                initial: "none",
                inherited: false,
                animatable: true,
            },
            Self::Transform => PropertyRow {
                name: "transform",
                initial: "none",
                inherited: false,
                animatable: true,
            },
            Self::TransformOrigin => PropertyRow {
                name: "transform-origin",
                initial: "0 0",
                inherited: false,
                animatable: true,
            },
            Self::TransformBox => PropertyRow {
                name: "transform-box",
                initial: "view-box",
                inherited: false,
                animatable: true,
            },
            Self::TextOverflow => PropertyRow {
                name: "text-overflow",
                initial: "clip",
                inherited: false,
                animatable: true,
            },
            Self::WhiteSpace => PropertyRow {
                name: "white-space",
                initial: "normal",
                inherited: true,
                animatable: true,
            },
            Self::MaskType => PropertyRow {
                name: "mask-type",
                initial: "luminance",
                inherited: false,
                animatable: true,
            },
            Self::Isolation => PropertyRow {
                name: "isolation",
                initial: "auto",
                inherited: false,
                animatable: true,
            },
            Self::MixBlendMode => PropertyRow {
                name: "mix-blend-mode",
                initial: "normal",
                inherited: false,
                animatable: true,
            },
            Self::D => PropertyRow {
                name: "d",
                initial: "none",
                inherited: true,
                animatable: true,
            },
            Self::FontKerning => PropertyRow {
                name: "font-kerning",
                initial: "auto",
                inherited: true,
                animatable: true,
            },
        }
    }
}

impl fmt::Display for PropertyId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn from_name_round_trips_every_property_case_insensitively() {
        for property in PropertyId::ALL {
            let name = property.as_str();
            assert_eq!(
                PropertyId::from_name(name),
                Some(*property),
                "lowercase name {name:?} should round-trip"
            );
            assert_eq!(
                PropertyId::from_name(&name.to_ascii_uppercase()),
                Some(*property),
                "uppercase name {name:?} should round-trip"
            );
            let mixed_case: String = name
                .char_indices()
                .map(|(index, c)| {
                    if index % 2 == 0 {
                        c.to_ascii_uppercase()
                    } else {
                        c
                    }
                })
                .collect();
            assert_eq!(
                PropertyId::from_name(&mixed_case),
                Some(*property),
                "mixed-case name {mixed_case:?} should round-trip"
            );
        }
    }

    #[test]
    fn property_names_are_unique_and_lowercase_hyphenated() {
        let mut names: Vec<&str> = PropertyId::ALL
            .iter()
            .map(|property| property.as_str())
            .collect();
        let original_len = names.len();
        names.sort_unstable();
        names.dedup();
        assert_eq!(
            names.len(),
            original_len,
            "PropertyId::ALL must not contain two properties with the same name"
        );
        for name in &names {
            assert!(
                name.chars().all(|c| c.is_ascii_lowercase() || c == '-'),
                "property name {name:?} must be ASCII lowercase with hyphens only"
            );
        }
    }

    #[test]
    fn from_name_rejects_names_this_crate_does_not_model() {
        assert_eq!(PropertyId::from_name(""), None);
        assert_eq!(PropertyId::from_name("not-a-property"), None);
        assert_eq!(
            PropertyId::from_name("width"),
            None,
            "geometry attributes are not presentation properties in this table"
        );
    }

    #[test]
    fn font_kerning_is_sourced_from_css_fonts_rather_than_the_catalog() {
        // Unlike every other PropertyId, font-kerning has no row in the
        // specification catalog at all (see the module documentation) — this
        // pins the CSS Fonts Module Level 3/4 facts this module fills in
        // instead.
        assert_eq!(
            PropertyId::from_name("font-kerning"),
            Some(PropertyId::FontKerning)
        );
        assert_eq!(PropertyId::FontKerning.as_str(), "font-kerning");
        assert_eq!(PropertyId::FontKerning.initial_value(), "auto");
        assert!(PropertyId::FontKerning.inherited());
        assert!(PropertyId::FontKerning.is_animatable());
        assert!(PropertyId::FontKerning.presentation_attribute_allowed());
    }

    #[test]
    fn inheritance_spot_checks_match_the_catalog() {
        assert!(PropertyId::Fill.inherited(), "fill inherits");
        assert!(!PropertyId::Opacity.inherited(), "opacity does not inherit");
        assert!(!PropertyId::Display.inherited(), "display does not inherit");
        assert!(PropertyId::Visibility.inherited(), "visibility inherits");
        assert!(PropertyId::StrokeWidth.inherited());
        assert!(PropertyId::ClipRule.inherited());
        assert!(!PropertyId::ClipPath.inherited());
        assert!(!PropertyId::Mask.inherited());
        assert!(
            PropertyId::PaintOrder.inherited(),
            "paint-order is a SVG2 addition that the catalog marks inherited"
        );
        assert!(!PropertyId::Transform.inherited());
        assert!(!PropertyId::TransformOrigin.inherited());
        assert!(!PropertyId::VectorEffect.inherited());
        assert!(!PropertyId::MaskType.inherited());
        assert!(PropertyId::WhiteSpace.inherited());
    }

    #[test]
    fn initial_value_spot_checks_match_the_catalog() {
        assert_eq!(PropertyId::Fill.initial_value(), "black");
        assert_eq!(PropertyId::Stroke.initial_value(), "none");
        assert_eq!(PropertyId::StrokeWidth.initial_value(), "1");
        assert_eq!(PropertyId::Opacity.initial_value(), "1");
        assert_eq!(PropertyId::Display.initial_value(), "inline");
        assert_eq!(PropertyId::FillRule.initial_value(), "nonzero");
        assert_eq!(PropertyId::StrokeMiterlimit.initial_value(), "4");
        assert_eq!(PropertyId::PaintOrder.initial_value(), "normal");
        assert_eq!(PropertyId::TransformOrigin.initial_value(), "0 0");
        assert_eq!(PropertyId::TransformBox.initial_value(), "view-box");
        assert_eq!(PropertyId::WritingMode.initial_value(), "lr-tb");
    }

    #[test]
    fn animatable_spot_checks_match_the_catalog() {
        assert!(PropertyId::Fill.is_animatable());
        assert!(PropertyId::Opacity.is_animatable());
        assert!(!PropertyId::Direction.is_animatable());
        assert!(!PropertyId::UnicodeBidi.is_animatable());
        assert!(!PropertyId::WritingMode.is_animatable());
        assert!(!PropertyId::GlyphOrientationHorizontal.is_animatable());
        assert!(!PropertyId::GlyphOrientationVertical.is_animatable());
        assert!(!PropertyId::EnableBackground.is_animatable());
    }

    #[test]
    fn presentation_attribute_allowed_is_true_for_every_property() {
        for property in PropertyId::ALL {
            assert!(property.presentation_attribute_allowed());
        }
    }

    #[test]
    fn as_str_uses_canonical_lowercase_hyphenated_names() {
        assert_eq!(PropertyId::StrokeWidth.as_str(), "stroke-width");
        assert_eq!(
            PropertyId::ColorInterpolationFilters.as_str(),
            "color-interpolation-filters"
        );
        assert_eq!(PropertyId::D.as_str(), "d");
        assert_eq!(PropertyId::MixBlendMode.as_str(), "mix-blend-mode");
    }

    #[test]
    fn display_matches_as_str_for_every_property() {
        for property in PropertyId::ALL {
            assert_eq!(property.to_string(), property.as_str());
        }
    }

    #[test]
    fn all_has_exactly_the_documented_seventy_three_properties() {
        assert_eq!(
            PropertyId::ALL.len(),
            73,
            "61 SVG1.1 propidx rows + 11 single-valued SVG2 additions + font-kerning, sourced \
             from CSS Fonts directly rather than the catalog"
        );
    }
}
