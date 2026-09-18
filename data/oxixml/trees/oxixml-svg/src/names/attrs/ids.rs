//! The [`AttrId`] enum: every SVG attribute name, its exact spec spelling,
//! and case-sensitive name lookup.

use std::fmt;

/// One SVG attribute name, spanning the union of the SVG 1.1 Second
/// Edition and SVG 2 attribute vocabularies (216 names total): core,XML
/// global, conditional-processing, presentation (property-mirroring),
/// geometry, viewport, `href`/`xlink:*`, gradient/pattern, marker,
/// filter (common and per-primitive), animation, and a handful of
/// element-specific attributes (`a`'s `target`/`download`/`rel`/
/// `hreflang`, `textPath`'s `startOffset`/`method`/`spacing`/`side`,
/// `use`/`image`/`script`'s `crossorigin`).
///
/// See the [module documentation](crate::names::attrs) for the "one authored name, one
/// variant" invariant this enum holds to, and for the small number of
/// variants sourced from this crate's mission brief rather than the
/// specification catalog directly.
///
/// Marked `#[non_exhaustive]` for the same forward-compatibility reason as
/// [`crate::names::elements::ElementId`]: matches from outside this crate
/// must carry a wildcard arm.
#[non_exhaustive]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum AttrId {
    // ---- 2.1 Core attributes ----
    /// `id` — a document-wide-unique identifier, the target of IRI
    /// fragment references.
    Id,
    /// `class` — a space-separated list of CSS class names.
    Class,
    /// `style` — an inline CSS declaration list; highest normal-priority
    /// cascade tier (see [`crate::style`]).
    Style,
    /// `lang` — SVG 2's HTML-aligned global language attribute, mirroring
    /// `xml:lang`.
    Lang,
    /// `xml:lang` — an RFC 5646 language tag (XML global attribute).
    XmlLang,
    /// `xml:space` — `default | preserve` white space handling. Deprecated
    /// in SVG 2, superseded by the CSS `white-space` property.
    XmlSpace,
    /// `xml:base` — establishes the base URI for relative-reference
    /// resolution within the subtree (XML Base).
    XmlBase,
    /// `tabindex` — SVG 2's HTML-aligned keyboard focus order.
    Tabindex,
    /// `autofocus` — SVG 2's HTML-aligned auto-focus-on-load flag.
    Autofocus,
    /// `slot` — SVG 2's HTML-aligned shadow-DOM slot assignment.
    Slot,
    /// `role` — SVG 2's HTML-aligned ARIA role override.
    Role,

    // ---- 2.2 Conditional processing attributes ----
    /// `requiredFeatures` — a feature-string conditional-processing test.
    /// Removed in SVG 2: feature strings were never interoperable.
    RequiredFeatures,
    /// `requiredExtensions` — an extension-URI conditional-processing
    /// test. Retained in SVG 2.
    RequiredExtensions,
    /// `systemLanguage` — a user-language conditional-processing test.
    /// Retained in SVG 2.
    SystemLanguage,

    // ---- 2.3 Presentation properties (SVG 1.1 propidx table) ----
    /// `alignment-baseline` — aligns a text run to a baseline relative to
    /// its parent text content element.
    AlignmentBaseline,
    /// `baseline-shift` — shifts a text run's dominant baseline (for
    /// example for subscript/superscript).
    BaselineShift,
    /// `clip` — a CSS2 `rect()`-based clipping region. Deprecated in favor
    /// of `clip-path`.
    Clip,
    /// `clip-path` — references a `clipPath` element (or, in SVG 2, a CSS
    /// basic shape) that clips the element's rendering.
    ClipPath,
    /// `clip-rule` — the fill rule (`nonzero`/`evenodd`) used when a shape
    /// is itself the content of a `clipPath`.
    ClipRule,
    /// `color` — the value `currentColor` resolves to.
    Color,
    /// `color-interpolation` — the color space (`sRGB`/`linearRGB`) used
    /// for gradient and animation color interpolation.
    ColorInterpolation,
    /// `color-interpolation-filters` — the color space used internally by
    /// filter primitives.
    ColorInterpolationFilters,
    /// `color-profile` — selects an ICC color profile for raster image
    /// elements. Removed in SVG 2 along with the `color-profile` element.
    ColorProfile,
    /// `color-rendering` — a rendering-quality-vs-speed hint for color
    /// computations.
    ColorRendering,
    /// `cursor` — selects the pointer cursor shown over the element.
    Cursor,
    /// `direction` — the base text direction (`ltr`/`rtl`), interacting
    /// with `unicode-bidi`.
    Direction,
    /// `display` — controls whether the element (and its subtree)
    /// participates in rendering at all.
    Display,
    /// `dominant-baseline` — the dominant baseline used to align text
    /// within a line.
    DominantBaseline,
    /// `enable-background` — declared a background image accumulation
    /// region for `BackgroundImage`/`BackgroundAlpha` filter inputs.
    /// Removed in SVG 2 along with those filter input values.
    EnableBackground,
    /// `fill` — **name-collision variant.** As a presentation property,
    /// the paint used to fill a shape or text run. As a SMIL animation
    /// attribute (on `animate`/`set`/`animateMotion`/`animateTransform`/
    /// `animateColor`), the freeze-vs-remove behavior after the active
    /// duration ends. See the [module documentation](super).
    Fill,
    /// `fill-opacity` — the opacity of the fill paint, independent of
    /// `opacity`.
    FillOpacity,
    /// `fill-rule` — the algorithm (`nonzero`/`evenodd`) used to determine
    /// a shape's interior.
    FillRule,
    /// `filter` — references a `filter` element applied to the element's
    /// rendering.
    Filter,
    /// `flood-color` — the flat color emitted by `feFlood`/`feDropShadow`.
    FloodColor,
    /// `flood-opacity` — the opacity of `flood-color`.
    FloodOpacity,
    /// `font` — the CSS `font` shorthand property.
    Font,
    /// `font-family` — the prioritized list of candidate font families.
    FontFamily,
    /// `font-size` — the font size used for glyph and layout metrics.
    FontSize,
    /// `font-size-adjust` — preserves apparent x-height across font
    /// substitution.
    FontSizeAdjust,
    /// `font-stretch` — the condensed/expanded font width selection.
    FontStretch,
    /// `font-style` — the normal/italic/oblique font style selection.
    FontStyle,
    /// `font-variant` — small-caps and related typographic variant
    /// selection.
    FontVariant,
    /// `font-weight` — the font boldness selection.
    FontWeight,
    /// `font-kerning` — SVG 2's replacement for the removed `kerning`
    /// property.
    FontKerning,
    /// `glyph-orientation-horizontal` — glyph rotation in horizontal
    /// writing modes.
    GlyphOrientationHorizontal,
    /// `glyph-orientation-vertical` — glyph rotation in vertical writing
    /// modes.
    GlyphOrientationVertical,
    /// `image-rendering` — a rendering-quality-vs-speed hint for image
    /// resampling.
    ImageRendering,
    /// `kerning` — enables/disables/overrides font kerning pairs. Removed
    /// in SVG 2 in favor of `font-kerning`.
    Kerning,
    /// `letter-spacing` — additional space inserted between characters.
    LetterSpacing,
    /// `lighting-color` — the light color used by `feDiffuseLighting`/
    /// `feSpecularLighting`.
    LightingColor,
    /// `marker` — the CSS `marker` shorthand property, setting
    /// `marker-start`/`marker-mid`/`marker-end` together.
    Marker,
    /// `marker-end` — the marker drawn at a path/line/polyline/polygon's
    /// final vertex.
    MarkerEnd,
    /// `marker-mid` — the marker drawn at a path/line/polyline/polygon's
    /// interior vertices.
    MarkerMid,
    /// `marker-start` — the marker drawn at a path/line/polyline/polygon's
    /// first vertex.
    MarkerStart,
    /// `mask` — references a `mask` element applied to the element's
    /// rendering.
    Mask,
    /// `opacity` — the group-level opacity applied to the element and its
    /// subtree as a whole.
    Opacity,
    /// `overflow` — whether content outside a viewport/pattern/marker's
    /// bounds is clipped.
    Overflow,
    /// `pointer-events` — which parts of the element (fill/stroke/none/...)
    /// participate in pointer hit-testing.
    PointerEvents,
    /// `shape-rendering` — a rendering-quality-vs-speed hint for shape
    /// geometry (for example crisp edges vs. antialiasing).
    ShapeRendering,
    /// `stop-color` — the color of a gradient `stop`.
    StopColor,
    /// `stop-opacity` — the opacity of a gradient `stop`'s color.
    StopOpacity,
    /// `stroke` — the paint used to stroke a shape or text run's outline.
    Stroke,
    /// `stroke-dasharray` — the on/off dash pattern used when stroking.
    StrokeDasharray,
    /// `stroke-dashoffset` — the starting offset into `stroke-dasharray`'s
    /// pattern.
    StrokeDashoffset,
    /// `stroke-linecap` — the shape drawn at the ends of open subpaths.
    StrokeLinecap,
    /// `stroke-linejoin` — the shape drawn at path vertices.
    StrokeLinejoin,
    /// `stroke-miterlimit` — the maximum miter length ratio before a miter
    /// join falls back to a bevel.
    StrokeMiterlimit,
    /// `stroke-opacity` — the opacity of the stroke paint, independent of
    /// `opacity`.
    StrokeOpacity,
    /// `stroke-width` — the width of the stroked outline.
    StrokeWidth,
    /// `text-anchor` — how a text run is aligned relative to its anchor
    /// point (`start`/`middle`/`end`).
    TextAnchor,
    /// `text-decoration` — underline/overline/line-through decoration.
    TextDecoration,
    /// `text-rendering` — a rendering-quality-vs-speed hint for text
    /// layout and glyph rasterization.
    TextRendering,
    /// `unicode-bidi` — how the element participates in the Unicode
    /// bidirectional algorithm.
    UnicodeBidi,
    /// `visibility` — whether the element is painted, without removing it
    /// from layout (unlike `display`).
    Visibility,
    /// `word-spacing` — additional space inserted between words.
    WordSpacing,
    /// `writing-mode` — the block/inline progression direction for text
    /// layout.
    WritingMode,

    // ---- 2.3 SVG 2 new/changed presentation properties ----
    /// `paint-order` — the relative painting order of fill, stroke, and
    /// markers. New in SVG 2.
    PaintOrder,
    /// `vector-effect` — non-geometric rendering adjustments such as
    /// `non-scaling-stroke`. New in SVG 2.
    VectorEffect,
    /// `transform` — a list of transform functions applied to the
    /// element's user space. SVG 2 promotes this from an XML-only
    /// attribute to a CSS property, with its own grammar distinct from the
    /// CSS `transform` property's (see [`crate::values::transform`]).
    Transform,
    /// `transform-origin` — the reference point transform functions are
    /// applied around. New in SVG 2 (CSS Transforms Level 1).
    TransformOrigin,
    /// `transform-box` — the reference box `transform-origin` is resolved
    /// against. New in SVG 2 (CSS Transforms Level 1).
    TransformBox,
    /// `text-overflow` — how clipped inline-size text content signals
    /// truncation. New in SVG 2.
    TextOverflow,
    /// `white-space` — CSS Text's white-space collapsing model, replacing
    /// `xml:space`'s semantics. New in SVG 2.
    WhiteSpace,
    /// `mask-type` — whether a `mask` element's content contributes
    /// luminance or alpha to the mask. New in SVG 2 (CSS Masking Level 1).
    MaskType,
    /// `isolation` — creates a new stacking context for blending
    /// isolation. New in SVG 2 (CSS Compositing/Blending).
    Isolation,
    /// `mix-blend-mode` — the blend mode used when compositing the element
    /// against its backdrop. New in SVG 2; unifies with `feBlend`'s `mode`
    /// value list.
    MixBlendMode,
    /// `d` — **name-collision variant.** As a `path` (and, historically,
    /// `glyph`/`missing-glyph`) attribute, the path data mini-language
    /// string. SVG 2 additionally makes `d` a CSS property of the same
    /// name, so a stylesheet rule can override a `<path d="...">`.
    D,

    // ---- 2.4 Geometry attributes ----
    /// `x` — a horizontal position or offset; meaning is per-element (top-
    /// left corner, viewport origin, filter-primitive light position, ...).
    X,
    /// `y` — a vertical position or offset; meaning is per-element.
    Y,
    /// `width` — a horizontal extent; meaning is per-element (a
    /// rectangle's/viewport's/pattern's width, ...).
    Width,
    /// `height` — a vertical extent; meaning is per-element.
    Height,
    /// `r` — a radius: `circle`'s radius, or `radialGradient`'s outer
    /// radius.
    R,
    /// `rx` — a horizontal radius: `rect`'s corner radius, or `ellipse`'s
    /// horizontal radius.
    Rx,
    /// `ry` — a vertical radius: `rect`'s corner radius, or `ellipse`'s
    /// vertical radius.
    Ry,
    /// `cx` — a center's horizontal coordinate (`circle`, `ellipse`,
    /// `radialGradient`).
    Cx,
    /// `cy` — a center's vertical coordinate (`circle`, `ellipse`,
    /// `radialGradient`).
    Cy,
    /// `x1` — a line/linear-gradient axis's first horizontal coordinate.
    X1,
    /// `y1` — a line/linear-gradient axis's first vertical coordinate.
    Y1,
    /// `x2` — a line/linear-gradient axis's second horizontal coordinate.
    X2,
    /// `y2` — a line/linear-gradient axis's second vertical coordinate.
    Y2,
    /// `points` — the vertex list of a `polyline`/`polygon`.
    Points,
    /// `pathLength` — an author-supplied total-length hint, used to
    /// rescale `stroke-dasharray`/`textPath` distance calculations.
    PathLength,
    /// `dx` — **name-collision variant.** A horizontal offset: per-glyph
    /// positioning on `text`/`tspan`, or `feOffset`'s/`feDropShadow`'s
    /// translation.
    Dx,
    /// `dy` — **name-collision variant.** A vertical offset, mirroring
    /// [`Self::Dx`].
    Dy,
    /// `rotate` — **name-collision variant.** Per-glyph rotation on
    /// `text`/`tspan`, or `animateMotion`'s `auto`/`auto-reverse`/`<angle>`
    /// path-orientation mode.
    Rotate,
    /// `textLength` — an author-supplied target length a text run is
    /// stretched or compressed to fit.
    TextLength,
    /// `lengthAdjust` — whether `textLength` adjusts glyph spacing only or
    /// glyphs themselves.
    LengthAdjust,
    /// `fr` — `radialGradient`'s focal radius. New in SVG 2.
    Fr,
    /// `fx` — `radialGradient`'s focal point horizontal coordinate.
    Fx,
    /// `fy` — `radialGradient`'s focal point vertical coordinate.
    Fy,

    // ---- 2.6 Viewport attributes ----
    /// `viewBox` — maps a `min-x min-y width height` user-space region onto
    /// the element's viewport.
    ViewBox,
    /// `preserveAspectRatio` — how `viewBox` content is aligned and scaled
    /// (uniformly or not) within the viewport.
    PreserveAspectRatio,
    /// `zoomAndPan` — `disable | magnify`, whether a standalone viewer may
    /// let the user zoom/pan the document. Deprecated in SVG 2 (mission-
    /// sourced: not enumerated by this crate's specification catalog pass,
    /// but a genuine SVG 1.1 attribute on `svg`/`view`).
    ZoomAndPan,

    // ---- 2.5 href / xlink family ----
    /// `href` — SVG 2's non-namespaced IRI reference attribute, replacing
    /// `xlink:href`. Takes precedence over `xlink:href` when both are
    /// present on the same element.
    Href,
    /// `xlink:href` — the SVG 1.1 IRI reference attribute (XLink
    /// namespace). Remains valid indefinitely for backward compatibility;
    /// not deprecated by default (see
    /// [`super::groups`](super)'s `deprecated_in_svg2`).
    XlinkHref,
    /// `xlink:show` — the legacy XLink `show` behavior hint (`new`/
    /// `replace`/...). SVG 1.1-only, dropped from SVG 2's attribute
    /// surface.
    XlinkShow,
    /// `xlink:actuate` — the legacy XLink `actuate` behavior hint
    /// (`onLoad`/`onRequest`). SVG 1.1-only, dropped from SVG 2.
    XlinkActuate,
    /// `xlink:type` — the legacy XLink link-type declaration (always
    /// `simple` in SVG). SVG 1.1-only, dropped from SVG 2.
    XlinkType,
    /// `xlink:title` — a legacy XLink advisory title. SVG 1.1-only,
    /// dropped from SVG 2 (superseded by a child `title` element).
    XlinkTitle,
    /// `xlink:arcrole` — a legacy XLink arc-role URI. SVG 1.1-only,
    /// dropped from SVG 2.
    XlinkArcrole,
    /// `xlink:role` — a legacy XLink role URI. SVG 1.1-only, dropped from
    /// SVG 2.
    XlinkRole,

    // ---- 2.7 Gradient / pattern unit and transform attributes ----
    /// `gradientUnits` — the coordinate system (`userSpaceOnUse`/
    /// `objectBoundingBox`) a gradient's own geometry is expressed in.
    GradientUnits,
    /// `gradientTransform` — an additional transform applied to a
    /// gradient's own coordinate system.
    GradientTransform,
    /// `spreadMethod` — how a gradient extends past its defined 0..1
    /// offset range (`pad`/`reflect`/`repeat`).
    SpreadMethod,
    /// `offset` — **name-collision variant.** A gradient `stop`'s position
    /// along its gradient (0..1 or a percentage), or a `feFuncR`/
    /// `feFuncG`/`feFuncB`/`feFuncA` transfer function's additive constant.
    Offset,
    /// `patternUnits` — the coordinate system a `pattern`'s tile
    /// rectangle (`x`/`y`/`width`/`height`) is expressed in.
    PatternUnits,
    /// `patternContentUnits` — the coordinate system a `pattern`'s content
    /// is expressed in, when `viewBox` is not used.
    PatternContentUnits,
    /// `patternTransform` — an additional transform applied to a
    /// `pattern`'s tile.
    PatternTransform,

    // ---- Marker attributes ----
    /// `markerUnits` — the coordinate system `markerWidth`/`markerHeight`
    /// are expressed in (`strokeWidth`/`userSpaceOnUse`).
    MarkerUnits,
    /// `markerWidth` — a `marker`'s viewport width.
    MarkerWidth,
    /// `markerHeight` — a `marker`'s viewport height.
    MarkerHeight,
    /// `refX` — the horizontal coordinate, within a `marker`/`symbol`'s own
    /// content, aligned to its reference point.
    RefX,
    /// `refY` — the vertical coordinate, within a `marker`/`symbol`'s own
    /// content, aligned to its reference point.
    RefY,
    /// `orient` — a `marker`'s rotation: `auto`, `auto-start-reverse`, or a
    /// fixed angle.
    Orient,

    // ---- 2.8 Filter attributes: common ----
    /// `in` — the first (or only) input of a filter primitive.
    In,
    /// `in2` — the second input of a two-input filter primitive
    /// (`feBlend`, `feComposite`, `feDisplacementMap`).
    In2,
    /// `result` — names a filter primitive's output so a later primitive
    /// can reference it via `in`/`in2`.
    Result,
    /// `stdDeviation` — the Gaussian blur standard deviation used by
    /// `feGaussianBlur` and `feDropShadow`.
    StdDeviation,
    /// `filterUnits` — the coordinate system a `filter`'s region
    /// (`x`/`y`/`width`/`height`) is expressed in.
    FilterUnits,
    /// `primitiveUnits` — the coordinate system filter primitive
    /// attributes such as `stdDeviation`/`radius` are expressed in.
    PrimitiveUnits,

    // ---- 2.8 Filter attributes: per-primitive ----
    /// `mode` — `feBlend`'s blend mode.
    Mode,
    /// `type` — **name-collision variant.** An enumerated keyword whose
    /// valid values differ per host element: `script`/`style`'s MIME/
    /// content type, `feColorMatrix`'s matrix kind, `feFuncR`/`feFuncG`/
    /// `feFuncB`/`feFuncA`'s transfer function kind, `feTurbulence`'s
    /// noise kind, `animateTransform`'s transform component, and (SVG 2)
    /// `a`'s MIME type hint.
    Type,
    /// `values` — **name-collision variant.** `feColorMatrix`'s matrix/
    /// saturate/hueRotate numeric argument(s), or an animation element's
    /// semicolon-separated list of values to animate through.
    Values,
    /// `tableValues` — the lookup table for a `table`/`discrete`
    /// `feFuncR`/`feFuncG`/`feFuncB`/`feFuncA` transfer function.
    TableValues,
    /// `slope` — a `linear`-type `feFuncR`/`feFuncG`/`feFuncB`/`feFuncA`
    /// transfer function's multiplicative factor.
    Slope,
    /// `intercept` — a `linear`-type `feFuncR`/`feFuncG`/`feFuncB`/
    /// `feFuncA` transfer function's additive constant.
    Intercept,
    /// `amplitude` — a `gamma`-type `feFuncR`/`feFuncG`/`feFuncB`/
    /// `feFuncA` transfer function's amplitude factor.
    Amplitude,
    /// `exponent` — a `gamma`-type `feFuncR`/`feFuncG`/`feFuncB`/
    /// `feFuncA` transfer function's exponent.
    Exponent,
    /// `order` — `feConvolveMatrix`'s kernel dimensions.
    Order,
    /// `kernelMatrix` — `feConvolveMatrix`'s convolution kernel values.
    KernelMatrix,
    /// `divisor` — `feConvolveMatrix`'s kernel-sum divisor.
    Divisor,
    /// `bias` — `feConvolveMatrix`'s post-convolution additive bias.
    Bias,
    /// `targetX` — `feConvolveMatrix`'s kernel target column.
    TargetX,
    /// `targetY` — `feConvolveMatrix`'s kernel target row.
    TargetY,
    /// `edgeMode` — `feConvolveMatrix`'s (and, per an open specification
    /// question, possibly `feGaussianBlur`'s) out-of-bounds sampling mode.
    EdgeMode,
    /// `kernelUnitLength` — the intermediate sampling grid resolution for
    /// `feConvolveMatrix`/`feDiffuseLighting`/`feSpecularLighting`.
    KernelUnitLength,
    /// `surfaceScale` — `feDiffuseLighting`'s/`feSpecularLighting`'s
    /// bump-map height scale factor.
    SurfaceScale,
    /// `diffuseConstant` — `feDiffuseLighting`'s reflectance constant.
    DiffuseConstant,
    /// `specularConstant` — `feSpecularLighting`'s reflectance constant.
    SpecularConstant,
    /// `specularExponent` — **name-collision variant.** `feSpecularLighting`'s
    /// shininess exponent, or `feSpotLight`'s focus falloff exponent.
    SpecularExponent,
    /// `azimuth` — `feDistantLight`'s horizontal light angle.
    Azimuth,
    /// `elevation` — `feDistantLight`'s vertical light angle.
    Elevation,
    /// `pointsAtX` — `fePointLight`'s/`feSpotLight`'s target horizontal
    /// coordinate.
    PointsAtX,
    /// `pointsAtY` — `fePointLight`'s/`feSpotLight`'s target vertical
    /// coordinate.
    PointsAtY,
    /// `pointsAtZ` — `fePointLight`'s/`feSpotLight`'s target depth
    /// coordinate.
    PointsAtZ,
    /// `limitingConeAngle` — `feSpotLight`'s cone half-angle beyond which
    /// no light is cast.
    LimitingConeAngle,
    /// `scale` — `feDisplacementMap`'s displacement scale factor.
    Scale,
    /// `xChannelSelector` — `feDisplacementMap`'s channel used for
    /// horizontal displacement.
    XChannelSelector,
    /// `yChannelSelector` — `feDisplacementMap`'s channel used for
    /// vertical displacement.
    YChannelSelector,
    /// `baseFrequency` — `feTurbulence`'s base noise frequency.
    BaseFrequency,
    /// `numOctaves` — `feTurbulence`'s number of summed noise octaves.
    NumOctaves,
    /// `seed` — `feTurbulence`'s pseudo-random seed.
    Seed,
    /// `stitchTiles` — `feTurbulence`'s `stitch`/`noStitch` seamless-tiling
    /// mode.
    StitchTiles,
    /// `radius` — `feMorphology`'s erosion/dilation radius.
    Radius,
    /// `operator` — **name-collision variant.** `feComposite`'s Porter-Duff
    /// (or `arithmetic`) compositing operator, or `feMorphology`'s
    /// `erode`/`dilate` choice.
    Operator,
    /// `k1` — `feComposite`'s `arithmetic` operator coefficient for the
    /// product term.
    K1,
    /// `k2` — `feComposite`'s `arithmetic` operator coefficient for the
    /// first input term.
    K2,
    /// `k3` — `feComposite`'s `arithmetic` operator coefficient for the
    /// second input term.
    K3,
    /// `k4` — `feComposite`'s `arithmetic` operator constant term.
    K4,
    /// `preserveAlpha` — `feConvolveMatrix`'s flag to convolve color
    /// channels only, passing the input alpha through unmodified.
    PreserveAlpha,
    /// `z` — `fePointLight`'s/`feSpotLight`'s light position depth
    /// coordinate.
    Z,

    // ---- 2.9 Animation attributes ----
    /// `attributeName` — the target attribute or property an `animate`/
    /// `set`/`animateTransform` element animates.
    AttributeName,
    /// `attributeType` — disambiguates `attributeName` as a CSS property
    /// vs. an XML attribute. Deprecated in SVG 2 (CSS is always tried
    /// first).
    AttributeType,
    /// `from` — an animation's starting value.
    From,
    /// `to` — an animation's ending value.
    To,
    /// `by` — an animation's relative end value (added to `from`, or to
    /// the underlying value, depending on context).
    By,
    /// `calcMode` — an animation's interpolation mode (`discrete`/
    /// `linear`/`paced`/`spline`).
    CalcMode,
    /// `keyTimes` — the normalized time offsets `values`'s milestones occur
    /// at.
    KeyTimes,
    /// `keySplines` — the cubic-Bézier easing control points between
    /// consecutive `keyTimes` under `calcMode="spline"`.
    KeySplines,
    /// `dur` — an animation's simple (single-iteration) duration.
    Dur,
    /// `begin` — a semicolon-separated list of begin-time specifiers.
    Begin,
    /// `end` — a semicolon-separated list of end-time specifiers.
    End,
    /// `min` — an animation's minimum active duration.
    Min,
    /// `max` — an animation's maximum active duration.
    Max,
    /// `restart` — whether/when an animation may be restarted while
    /// active (`always`/`whenNotActive`/`never`).
    Restart,
    /// `repeatCount` — the number of times an animation repeats, or
    /// `indefinite`.
    RepeatCount,
    /// `repeatDur` — the total duration an animation repeats for, or
    /// `indefinite`.
    RepeatDur,
    /// `additive` — whether an animation's value replaces or adds to the
    /// underlying value (`replace`/`sum`).
    Additive,
    /// `accumulate` — whether repeat iterations build on the previous
    /// iteration's end value (`none`/`sum`).
    Accumulate,
    /// `keyPoints` — `animateMotion`'s distance-along-path milestones,
    /// paired with `keyTimes`.
    KeyPoints,
    /// `origin` — `animateMotion`'s legacy origin attribute; a no-op that
    /// SVG 2 removes.
    Origin,
    /// `path` — **name-collision variant.** `animateMotion`'s path data to
    /// move along, or (SVG 2) `textPath`'s inline path data to lay text
    /// out along.
    Path,

    // ---- use / image / script ----
    /// `crossorigin` — the CORS request mode used when fetching a
    /// cross-origin referenced resource.
    Crossorigin,

    // ---- `a`-specific (SVG 2 new) ----
    /// `target` — the browsing context a hyperlink navigates in. New in
    /// SVG 2.
    Target,
    /// `download` — hints that a hyperlink's target should be downloaded
    /// rather than navigated to. New in SVG 2.
    Download,
    /// `rel` — the relationship between the document and a hyperlink's
    /// target. New in SVG 2.
    Rel,
    /// `hreflang` — the language of a hyperlink's target resource. New in
    /// SVG 2.
    Hreflang,

    // ---- text / textPath ----
    /// `startOffset` — a `textPath`'s starting distance along its
    /// referenced path (mission-sourced: a genuine SVG 1.1 `textPath`
    /// attribute not enumerated by this crate's specification catalog
    /// pass).
    StartOffset,
    /// `method` — a `textPath`'s glyph-rendering method (`align`/
    /// `stretch`) (mission-sourced, see [`Self::StartOffset`]).
    Method,
    /// `spacing` — a `textPath`'s glyph-spacing mode (`auto`/`exact`)
    /// (mission-sourced, see [`Self::StartOffset`]).
    Spacing,
    /// `side` — which side of the path a `textPath`'s text is rendered on.
    /// New in SVG 2.
    Side,

    // ---- clip / mask units ----
    /// `clipPathUnits` — the coordinate system a `clipPath`'s content is
    /// expressed in.
    ClipPathUnits,
    /// `maskUnits` — the coordinate system a `mask`'s region
    /// (`x`/`y`/`width`/`height`) is expressed in.
    MaskUnits,
    /// `maskContentUnits` — the coordinate system a `mask`'s content is
    /// expressed in.
    MaskContentUnits,

    // ---- root-element attributes (mission-sourced) ----
    /// `baseProfile` — an advisory declaration of the minimum SVG profile
    /// a document requires (mission-sourced: a genuine SVG 1.1 attribute on
    /// `svg` not enumerated by this crate's specification catalog pass).
    /// Deprecated in SVG 2 (profiles are an SVG 1.1-only concept).
    BaseProfile,
    /// `version` — an advisory declaration of the SVG language version a
    /// document was authored against (mission-sourced, see
    /// [`Self::BaseProfile`]). Deprecated in SVG 2.
    Version,
}

impl AttrId {
    /// Every [`AttrId`] variant, in the same order as this module's
    /// declaration (grouped by the specification catalog's own section
    /// order: core, conditional processing, presentation, geometry,
    /// viewport, href/xlink, gradient/pattern, marker, filter, animation,
    /// and the small element-specific tails).
    #[rustfmt::skip]
    pub const ALL: &'static [AttrId] = &[
        Self::Id, Self::Class, Self::Style, Self::Lang, Self::XmlLang, Self::XmlSpace,
        Self::XmlBase, Self::Tabindex, Self::Autofocus, Self::Slot, Self::Role,

        Self::RequiredFeatures, Self::RequiredExtensions, Self::SystemLanguage,

        Self::AlignmentBaseline, Self::BaselineShift, Self::Clip, Self::ClipPath,
        Self::ClipRule, Self::Color, Self::ColorInterpolation,
        Self::ColorInterpolationFilters, Self::ColorProfile, Self::ColorRendering,
        Self::Cursor, Self::Direction, Self::Display, Self::DominantBaseline,
        Self::EnableBackground, Self::Fill, Self::FillOpacity, Self::FillRule,
        Self::Filter, Self::FloodColor, Self::FloodOpacity, Self::Font,
        Self::FontFamily, Self::FontSize, Self::FontSizeAdjust, Self::FontStretch,
        Self::FontStyle, Self::FontVariant, Self::FontWeight, Self::FontKerning,
        Self::GlyphOrientationHorizontal, Self::GlyphOrientationVertical,
        Self::ImageRendering, Self::Kerning, Self::LetterSpacing, Self::LightingColor,
        Self::Marker, Self::MarkerEnd, Self::MarkerMid, Self::MarkerStart, Self::Mask,
        Self::Opacity, Self::Overflow, Self::PointerEvents, Self::ShapeRendering,
        Self::StopColor, Self::StopOpacity, Self::Stroke, Self::StrokeDasharray,
        Self::StrokeDashoffset, Self::StrokeLinecap, Self::StrokeLinejoin,
        Self::StrokeMiterlimit, Self::StrokeOpacity, Self::StrokeWidth,
        Self::TextAnchor, Self::TextDecoration, Self::TextRendering,
        Self::UnicodeBidi, Self::Visibility, Self::WordSpacing, Self::WritingMode,

        Self::PaintOrder, Self::VectorEffect, Self::Transform, Self::TransformOrigin,
        Self::TransformBox, Self::TextOverflow, Self::WhiteSpace, Self::MaskType,
        Self::Isolation, Self::MixBlendMode, Self::D,

        Self::X, Self::Y, Self::Width, Self::Height, Self::R, Self::Rx, Self::Ry,
        Self::Cx, Self::Cy, Self::X1, Self::Y1, Self::X2, Self::Y2, Self::Points,
        Self::PathLength, Self::Dx, Self::Dy, Self::Rotate, Self::TextLength,
        Self::LengthAdjust, Self::Fr, Self::Fx, Self::Fy,

        Self::ViewBox, Self::PreserveAspectRatio, Self::ZoomAndPan,

        Self::Href, Self::XlinkHref, Self::XlinkShow, Self::XlinkActuate,
        Self::XlinkType, Self::XlinkTitle, Self::XlinkArcrole, Self::XlinkRole,

        Self::GradientUnits, Self::GradientTransform, Self::SpreadMethod, Self::Offset,
        Self::PatternUnits, Self::PatternContentUnits, Self::PatternTransform,

        Self::MarkerUnits, Self::MarkerWidth, Self::MarkerHeight, Self::RefX,
        Self::RefY, Self::Orient,

        Self::In, Self::In2, Self::Result, Self::StdDeviation, Self::FilterUnits,
        Self::PrimitiveUnits, Self::Mode, Self::Type, Self::Values, Self::TableValues, Self::Slope,
        Self::Intercept, Self::Amplitude, Self::Exponent, Self::Order,
        Self::KernelMatrix, Self::Divisor, Self::Bias, Self::TargetX, Self::TargetY,
        Self::EdgeMode, Self::KernelUnitLength, Self::SurfaceScale,
        Self::DiffuseConstant, Self::SpecularConstant, Self::SpecularExponent,
        Self::Azimuth, Self::Elevation, Self::PointsAtX, Self::PointsAtY,
        Self::PointsAtZ, Self::LimitingConeAngle, Self::Scale, Self::XChannelSelector,
        Self::YChannelSelector, Self::BaseFrequency, Self::NumOctaves, Self::Seed,
        Self::StitchTiles, Self::Radius, Self::Operator, Self::K1, Self::K2, Self::K3,
        Self::K4, Self::PreserveAlpha, Self::Z,

        Self::AttributeName, Self::AttributeType, Self::From, Self::To, Self::By,
        Self::CalcMode, Self::KeyTimes, Self::KeySplines, Self::Dur, Self::Begin,
        Self::End, Self::Min, Self::Max, Self::Restart, Self::RepeatCount,
        Self::RepeatDur, Self::Additive, Self::Accumulate, Self::KeyPoints,
        Self::Origin, Self::Path,

        Self::Crossorigin,

        Self::Target, Self::Download, Self::Rel, Self::Hreflang,

        Self::StartOffset, Self::Method, Self::Spacing, Self::Side,

        Self::ClipPathUnits, Self::MaskUnits, Self::MaskContentUnits,

        Self::BaseProfile, Self::Version,
    ];

    /// Looks up an [`AttrId`] by its exact, case-sensitive authored name
    /// (for example `"stroke-width"` or `"xlink:href"`).
    ///
    /// The authored form is matched exactly, including any `xml:`/`xlink:`
    /// namespace prefix; no case-folding, normalization, or namespace-URI
    /// resolution is performed. Callers holding a namespace-resolved
    /// attribute should pass the qualified/prefixed spelling SVG authors
    /// actually write (`"xlink:href"`, not the XLink namespace URI plus
    /// local name `"href"`) — `"href"` alone maps to [`Self::Href`], the
    /// distinct SVG 2 attribute.
    #[must_use]
    pub fn from_name(name: &str) -> Option<Self> {
        Some(match name {
            "id" => Self::Id,
            "class" => Self::Class,
            "style" => Self::Style,
            "lang" => Self::Lang,
            "xml:lang" => Self::XmlLang,
            "xml:space" => Self::XmlSpace,
            "xml:base" => Self::XmlBase,
            "tabindex" => Self::Tabindex,
            "autofocus" => Self::Autofocus,
            "slot" => Self::Slot,
            "role" => Self::Role,

            "requiredFeatures" => Self::RequiredFeatures,
            "requiredExtensions" => Self::RequiredExtensions,
            "systemLanguage" => Self::SystemLanguage,

            "alignment-baseline" => Self::AlignmentBaseline,
            "baseline-shift" => Self::BaselineShift,
            "clip" => Self::Clip,
            "clip-path" => Self::ClipPath,
            "clip-rule" => Self::ClipRule,
            "color" => Self::Color,
            "color-interpolation" => Self::ColorInterpolation,
            "color-interpolation-filters" => Self::ColorInterpolationFilters,
            "color-profile" => Self::ColorProfile,
            "color-rendering" => Self::ColorRendering,
            "cursor" => Self::Cursor,
            "direction" => Self::Direction,
            "display" => Self::Display,
            "dominant-baseline" => Self::DominantBaseline,
            "enable-background" => Self::EnableBackground,
            "fill" => Self::Fill,
            "fill-opacity" => Self::FillOpacity,
            "fill-rule" => Self::FillRule,
            "filter" => Self::Filter,
            "flood-color" => Self::FloodColor,
            "flood-opacity" => Self::FloodOpacity,
            "font" => Self::Font,
            "font-family" => Self::FontFamily,
            "font-size" => Self::FontSize,
            "font-size-adjust" => Self::FontSizeAdjust,
            "font-stretch" => Self::FontStretch,
            "font-style" => Self::FontStyle,
            "font-variant" => Self::FontVariant,
            "font-weight" => Self::FontWeight,
            "font-kerning" => Self::FontKerning,
            "glyph-orientation-horizontal" => Self::GlyphOrientationHorizontal,
            "glyph-orientation-vertical" => Self::GlyphOrientationVertical,
            "image-rendering" => Self::ImageRendering,
            "kerning" => Self::Kerning,
            "letter-spacing" => Self::LetterSpacing,
            "lighting-color" => Self::LightingColor,
            "marker" => Self::Marker,
            "marker-end" => Self::MarkerEnd,
            "marker-mid" => Self::MarkerMid,
            "marker-start" => Self::MarkerStart,
            "mask" => Self::Mask,
            "opacity" => Self::Opacity,
            "overflow" => Self::Overflow,
            "pointer-events" => Self::PointerEvents,
            "shape-rendering" => Self::ShapeRendering,
            "stop-color" => Self::StopColor,
            "stop-opacity" => Self::StopOpacity,
            "stroke" => Self::Stroke,
            "stroke-dasharray" => Self::StrokeDasharray,
            "stroke-dashoffset" => Self::StrokeDashoffset,
            "stroke-linecap" => Self::StrokeLinecap,
            "stroke-linejoin" => Self::StrokeLinejoin,
            "stroke-miterlimit" => Self::StrokeMiterlimit,
            "stroke-opacity" => Self::StrokeOpacity,
            "stroke-width" => Self::StrokeWidth,
            "text-anchor" => Self::TextAnchor,
            "text-decoration" => Self::TextDecoration,
            "text-rendering" => Self::TextRendering,
            "unicode-bidi" => Self::UnicodeBidi,
            "visibility" => Self::Visibility,
            "word-spacing" => Self::WordSpacing,
            "writing-mode" => Self::WritingMode,

            "paint-order" => Self::PaintOrder,
            "vector-effect" => Self::VectorEffect,
            "transform" => Self::Transform,
            "transform-origin" => Self::TransformOrigin,
            "transform-box" => Self::TransformBox,
            "text-overflow" => Self::TextOverflow,
            "white-space" => Self::WhiteSpace,
            "mask-type" => Self::MaskType,
            "isolation" => Self::Isolation,
            "mix-blend-mode" => Self::MixBlendMode,
            "d" => Self::D,

            "x" => Self::X,
            "y" => Self::Y,
            "width" => Self::Width,
            "height" => Self::Height,
            "r" => Self::R,
            "rx" => Self::Rx,
            "ry" => Self::Ry,
            "cx" => Self::Cx,
            "cy" => Self::Cy,
            "x1" => Self::X1,
            "y1" => Self::Y1,
            "x2" => Self::X2,
            "y2" => Self::Y2,
            "points" => Self::Points,
            "pathLength" => Self::PathLength,
            "dx" => Self::Dx,
            "dy" => Self::Dy,
            "rotate" => Self::Rotate,
            "textLength" => Self::TextLength,
            "lengthAdjust" => Self::LengthAdjust,
            "fr" => Self::Fr,
            "fx" => Self::Fx,
            "fy" => Self::Fy,

            "viewBox" => Self::ViewBox,
            "preserveAspectRatio" => Self::PreserveAspectRatio,
            "zoomAndPan" => Self::ZoomAndPan,

            "href" => Self::Href,
            "xlink:href" => Self::XlinkHref,
            "xlink:show" => Self::XlinkShow,
            "xlink:actuate" => Self::XlinkActuate,
            "xlink:type" => Self::XlinkType,
            "xlink:title" => Self::XlinkTitle,
            "xlink:arcrole" => Self::XlinkArcrole,
            "xlink:role" => Self::XlinkRole,

            "gradientUnits" => Self::GradientUnits,
            "gradientTransform" => Self::GradientTransform,
            "spreadMethod" => Self::SpreadMethod,
            "offset" => Self::Offset,
            "patternUnits" => Self::PatternUnits,
            "patternContentUnits" => Self::PatternContentUnits,
            "patternTransform" => Self::PatternTransform,

            "markerUnits" => Self::MarkerUnits,
            "markerWidth" => Self::MarkerWidth,
            "markerHeight" => Self::MarkerHeight,
            "refX" => Self::RefX,
            "refY" => Self::RefY,
            "orient" => Self::Orient,

            "in" => Self::In,
            "in2" => Self::In2,
            "result" => Self::Result,
            "stdDeviation" => Self::StdDeviation,
            "filterUnits" => Self::FilterUnits,
            "primitiveUnits" => Self::PrimitiveUnits,
            "mode" => Self::Mode,
            "type" => Self::Type,
            "values" => Self::Values,
            "tableValues" => Self::TableValues,
            "slope" => Self::Slope,
            "intercept" => Self::Intercept,
            "amplitude" => Self::Amplitude,
            "exponent" => Self::Exponent,
            "order" => Self::Order,
            "kernelMatrix" => Self::KernelMatrix,
            "divisor" => Self::Divisor,
            "bias" => Self::Bias,
            "targetX" => Self::TargetX,
            "targetY" => Self::TargetY,
            "edgeMode" => Self::EdgeMode,
            "kernelUnitLength" => Self::KernelUnitLength,
            "surfaceScale" => Self::SurfaceScale,
            "diffuseConstant" => Self::DiffuseConstant,
            "specularConstant" => Self::SpecularConstant,
            "specularExponent" => Self::SpecularExponent,
            "azimuth" => Self::Azimuth,
            "elevation" => Self::Elevation,
            "pointsAtX" => Self::PointsAtX,
            "pointsAtY" => Self::PointsAtY,
            "pointsAtZ" => Self::PointsAtZ,
            "limitingConeAngle" => Self::LimitingConeAngle,
            "scale" => Self::Scale,
            "xChannelSelector" => Self::XChannelSelector,
            "yChannelSelector" => Self::YChannelSelector,
            "baseFrequency" => Self::BaseFrequency,
            "numOctaves" => Self::NumOctaves,
            "seed" => Self::Seed,
            "stitchTiles" => Self::StitchTiles,
            "radius" => Self::Radius,
            "operator" => Self::Operator,
            "k1" => Self::K1,
            "k2" => Self::K2,
            "k3" => Self::K3,
            "k4" => Self::K4,
            "preserveAlpha" => Self::PreserveAlpha,
            "z" => Self::Z,

            "attributeName" => Self::AttributeName,
            "attributeType" => Self::AttributeType,
            "from" => Self::From,
            "to" => Self::To,
            "by" => Self::By,
            "calcMode" => Self::CalcMode,
            "keyTimes" => Self::KeyTimes,
            "keySplines" => Self::KeySplines,
            "dur" => Self::Dur,
            "begin" => Self::Begin,
            "end" => Self::End,
            "min" => Self::Min,
            "max" => Self::Max,
            "restart" => Self::Restart,
            "repeatCount" => Self::RepeatCount,
            "repeatDur" => Self::RepeatDur,
            "additive" => Self::Additive,
            "accumulate" => Self::Accumulate,
            "keyPoints" => Self::KeyPoints,
            "origin" => Self::Origin,
            "path" => Self::Path,

            "crossorigin" => Self::Crossorigin,

            "target" => Self::Target,
            "download" => Self::Download,
            "rel" => Self::Rel,
            "hreflang" => Self::Hreflang,

            "startOffset" => Self::StartOffset,
            "method" => Self::Method,
            "spacing" => Self::Spacing,
            "side" => Self::Side,

            "clipPathUnits" => Self::ClipPathUnits,
            "maskUnits" => Self::MaskUnits,
            "maskContentUnits" => Self::MaskContentUnits,

            "baseProfile" => Self::BaseProfile,
            "version" => Self::Version,

            _ => return None,
        })
    }

    /// The exact, case-sensitive authored attribute name, spelled exactly
    /// as the specification authors it (for example `"stroke-width"`,
    /// `"xlink:href"`, `"pathLength"`).
    #[must_use]
    #[allow(clippy::too_many_lines)]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Id => "id",
            Self::Class => "class",
            Self::Style => "style",
            Self::Lang => "lang",
            Self::XmlLang => "xml:lang",
            Self::XmlSpace => "xml:space",
            Self::XmlBase => "xml:base",
            Self::Tabindex => "tabindex",
            Self::Autofocus => "autofocus",
            Self::Slot => "slot",
            Self::Role => "role",

            Self::RequiredFeatures => "requiredFeatures",
            Self::RequiredExtensions => "requiredExtensions",
            Self::SystemLanguage => "systemLanguage",

            Self::AlignmentBaseline => "alignment-baseline",
            Self::BaselineShift => "baseline-shift",
            Self::Clip => "clip",
            Self::ClipPath => "clip-path",
            Self::ClipRule => "clip-rule",
            Self::Color => "color",
            Self::ColorInterpolation => "color-interpolation",
            Self::ColorInterpolationFilters => "color-interpolation-filters",
            Self::ColorProfile => "color-profile",
            Self::ColorRendering => "color-rendering",
            Self::Cursor => "cursor",
            Self::Direction => "direction",
            Self::Display => "display",
            Self::DominantBaseline => "dominant-baseline",
            Self::EnableBackground => "enable-background",
            Self::Fill => "fill",
            Self::FillOpacity => "fill-opacity",
            Self::FillRule => "fill-rule",
            Self::Filter => "filter",
            Self::FloodColor => "flood-color",
            Self::FloodOpacity => "flood-opacity",
            Self::Font => "font",
            Self::FontFamily => "font-family",
            Self::FontSize => "font-size",
            Self::FontSizeAdjust => "font-size-adjust",
            Self::FontStretch => "font-stretch",
            Self::FontStyle => "font-style",
            Self::FontVariant => "font-variant",
            Self::FontWeight => "font-weight",
            Self::FontKerning => "font-kerning",
            Self::GlyphOrientationHorizontal => "glyph-orientation-horizontal",
            Self::GlyphOrientationVertical => "glyph-orientation-vertical",
            Self::ImageRendering => "image-rendering",
            Self::Kerning => "kerning",
            Self::LetterSpacing => "letter-spacing",
            Self::LightingColor => "lighting-color",
            Self::Marker => "marker",
            Self::MarkerEnd => "marker-end",
            Self::MarkerMid => "marker-mid",
            Self::MarkerStart => "marker-start",
            Self::Mask => "mask",
            Self::Opacity => "opacity",
            Self::Overflow => "overflow",
            Self::PointerEvents => "pointer-events",
            Self::ShapeRendering => "shape-rendering",
            Self::StopColor => "stop-color",
            Self::StopOpacity => "stop-opacity",
            Self::Stroke => "stroke",
            Self::StrokeDasharray => "stroke-dasharray",
            Self::StrokeDashoffset => "stroke-dashoffset",
            Self::StrokeLinecap => "stroke-linecap",
            Self::StrokeLinejoin => "stroke-linejoin",
            Self::StrokeMiterlimit => "stroke-miterlimit",
            Self::StrokeOpacity => "stroke-opacity",
            Self::StrokeWidth => "stroke-width",
            Self::TextAnchor => "text-anchor",
            Self::TextDecoration => "text-decoration",
            Self::TextRendering => "text-rendering",
            Self::UnicodeBidi => "unicode-bidi",
            Self::Visibility => "visibility",
            Self::WordSpacing => "word-spacing",
            Self::WritingMode => "writing-mode",

            Self::PaintOrder => "paint-order",
            Self::VectorEffect => "vector-effect",
            Self::Transform => "transform",
            Self::TransformOrigin => "transform-origin",
            Self::TransformBox => "transform-box",
            Self::TextOverflow => "text-overflow",
            Self::WhiteSpace => "white-space",
            Self::MaskType => "mask-type",
            Self::Isolation => "isolation",
            Self::MixBlendMode => "mix-blend-mode",
            Self::D => "d",

            Self::X => "x",
            Self::Y => "y",
            Self::Width => "width",
            Self::Height => "height",
            Self::R => "r",
            Self::Rx => "rx",
            Self::Ry => "ry",
            Self::Cx => "cx",
            Self::Cy => "cy",
            Self::X1 => "x1",
            Self::Y1 => "y1",
            Self::X2 => "x2",
            Self::Y2 => "y2",
            Self::Points => "points",
            Self::PathLength => "pathLength",
            Self::Dx => "dx",
            Self::Dy => "dy",
            Self::Rotate => "rotate",
            Self::TextLength => "textLength",
            Self::LengthAdjust => "lengthAdjust",
            Self::Fr => "fr",
            Self::Fx => "fx",
            Self::Fy => "fy",

            Self::ViewBox => "viewBox",
            Self::PreserveAspectRatio => "preserveAspectRatio",
            Self::ZoomAndPan => "zoomAndPan",

            Self::Href => "href",
            Self::XlinkHref => "xlink:href",
            Self::XlinkShow => "xlink:show",
            Self::XlinkActuate => "xlink:actuate",
            Self::XlinkType => "xlink:type",
            Self::XlinkTitle => "xlink:title",
            Self::XlinkArcrole => "xlink:arcrole",
            Self::XlinkRole => "xlink:role",

            Self::GradientUnits => "gradientUnits",
            Self::GradientTransform => "gradientTransform",
            Self::SpreadMethod => "spreadMethod",
            Self::Offset => "offset",
            Self::PatternUnits => "patternUnits",
            Self::PatternContentUnits => "patternContentUnits",
            Self::PatternTransform => "patternTransform",

            Self::MarkerUnits => "markerUnits",
            Self::MarkerWidth => "markerWidth",
            Self::MarkerHeight => "markerHeight",
            Self::RefX => "refX",
            Self::RefY => "refY",
            Self::Orient => "orient",

            Self::In => "in",
            Self::In2 => "in2",
            Self::Result => "result",
            Self::StdDeviation => "stdDeviation",
            Self::FilterUnits => "filterUnits",
            Self::PrimitiveUnits => "primitiveUnits",
            Self::Mode => "mode",
            Self::Type => "type",
            Self::Values => "values",
            Self::TableValues => "tableValues",
            Self::Slope => "slope",
            Self::Intercept => "intercept",
            Self::Amplitude => "amplitude",
            Self::Exponent => "exponent",
            Self::Order => "order",
            Self::KernelMatrix => "kernelMatrix",
            Self::Divisor => "divisor",
            Self::Bias => "bias",
            Self::TargetX => "targetX",
            Self::TargetY => "targetY",
            Self::EdgeMode => "edgeMode",
            Self::KernelUnitLength => "kernelUnitLength",
            Self::SurfaceScale => "surfaceScale",
            Self::DiffuseConstant => "diffuseConstant",
            Self::SpecularConstant => "specularConstant",
            Self::SpecularExponent => "specularExponent",
            Self::Azimuth => "azimuth",
            Self::Elevation => "elevation",
            Self::PointsAtX => "pointsAtX",
            Self::PointsAtY => "pointsAtY",
            Self::PointsAtZ => "pointsAtZ",
            Self::LimitingConeAngle => "limitingConeAngle",
            Self::Scale => "scale",
            Self::XChannelSelector => "xChannelSelector",
            Self::YChannelSelector => "yChannelSelector",
            Self::BaseFrequency => "baseFrequency",
            Self::NumOctaves => "numOctaves",
            Self::Seed => "seed",
            Self::StitchTiles => "stitchTiles",
            Self::Radius => "radius",
            Self::Operator => "operator",
            Self::K1 => "k1",
            Self::K2 => "k2",
            Self::K3 => "k3",
            Self::K4 => "k4",
            Self::PreserveAlpha => "preserveAlpha",
            Self::Z => "z",

            Self::AttributeName => "attributeName",
            Self::AttributeType => "attributeType",
            Self::From => "from",
            Self::To => "to",
            Self::By => "by",
            Self::CalcMode => "calcMode",
            Self::KeyTimes => "keyTimes",
            Self::KeySplines => "keySplines",
            Self::Dur => "dur",
            Self::Begin => "begin",
            Self::End => "end",
            Self::Min => "min",
            Self::Max => "max",
            Self::Restart => "restart",
            Self::RepeatCount => "repeatCount",
            Self::RepeatDur => "repeatDur",
            Self::Additive => "additive",
            Self::Accumulate => "accumulate",
            Self::KeyPoints => "keyPoints",
            Self::Origin => "origin",
            Self::Path => "path",

            Self::Crossorigin => "crossorigin",

            Self::Target => "target",
            Self::Download => "download",
            Self::Rel => "rel",
            Self::Hreflang => "hreflang",

            Self::StartOffset => "startOffset",
            Self::Method => "method",
            Self::Spacing => "spacing",
            Self::Side => "side",

            Self::ClipPathUnits => "clipPathUnits",
            Self::MaskUnits => "maskUnits",
            Self::MaskContentUnits => "maskContentUnits",

            Self::BaseProfile => "baseProfile",
            Self::Version => "version",
        }
    }
}

impl fmt::Display for AttrId {
    /// Writes the exact, case-sensitive authored attribute name (see
    /// [`Self::as_str`]).
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn all_lists_every_variant_exactly_once() {
        assert_eq!(AttrId::ALL.len(), 216, "216 attribute names total");
        let mut seen = std::collections::HashSet::new();
        for id in AttrId::ALL {
            assert!(seen.insert(*id), "{id:?} listed more than once in ALL");
        }
    }

    #[test]
    fn from_name_as_str_round_trip_every_attribute() {
        for &id in AttrId::ALL {
            let name = id.as_str();
            assert_eq!(
                AttrId::from_name(name),
                Some(id),
                "round trip failed for {id:?} ({name:?})"
            );
        }
    }

    #[test]
    fn as_str_returns_unique_names_for_every_variant() {
        let mut seen = std::collections::HashSet::new();
        for &id in AttrId::ALL {
            assert!(
                seen.insert(id.as_str()),
                "{:?} duplicates the name {:?} already produced by another variant",
                id,
                id.as_str()
            );
        }
    }

    #[test]
    fn as_str_matches_exact_spec_casing() {
        assert_eq!(AttrId::StrokeWidth.as_str(), "stroke-width");
        assert_eq!(AttrId::XlinkHref.as_str(), "xlink:href");
        assert_eq!(AttrId::XmlLang.as_str(), "xml:lang");
        assert_eq!(AttrId::PathLength.as_str(), "pathLength");
        assert_eq!(AttrId::ViewBox.as_str(), "viewBox");
        assert_eq!(AttrId::PreserveAspectRatio.as_str(), "preserveAspectRatio");
        assert_eq!(AttrId::KernelUnitLength.as_str(), "kernelUnitLength");
    }

    #[test]
    fn from_name_is_case_sensitive() {
        assert_eq!(AttrId::from_name("STROKE-WIDTH"), None);
        assert_eq!(AttrId::from_name("Stroke-Width"), None);
        assert_eq!(AttrId::from_name("XLINK:HREF"), None);
        assert_eq!(AttrId::from_name("stroke-width"), Some(AttrId::StrokeWidth));
    }

    #[test]
    fn from_name_distinguishes_href_from_xlink_href() {
        assert_eq!(AttrId::from_name("href"), Some(AttrId::Href));
        assert_eq!(AttrId::from_name("xlink:href"), Some(AttrId::XlinkHref));
        assert_ne!(AttrId::Href, AttrId::XlinkHref);
    }

    #[test]
    fn from_name_rejects_unknown_names() {
        assert_eq!(AttrId::from_name(""), None);
        assert_eq!(AttrId::from_name("data-foo"), None);
        assert_eq!(AttrId::from_name("aria-label"), None);
        assert_eq!(AttrId::from_name("nonexistent-attribute"), None);
    }

    #[test]
    fn display_matches_as_str() {
        assert_eq!(AttrId::StrokeWidth.to_string(), "stroke-width");
        assert_eq!(AttrId::XlinkHref.to_string(), "xlink:href");
    }

    #[test]
    fn collision_variants_are_single_variants_shared_across_domains() {
        // `fill`: presentation paint AND SMIL freeze/remove, one variant.
        assert_eq!(AttrId::from_name("fill"), Some(AttrId::Fill));
        // `type`: shared by script/style/feColorMatrix/feFuncX/feTurbulence/
        // animateTransform/`a`, one variant.
        assert_eq!(AttrId::from_name("type"), Some(AttrId::Type));
        // `offset`: shared by `stop` and feFuncX, one variant.
        assert_eq!(AttrId::from_name("offset"), Some(AttrId::Offset));
        // `rotate`: shared by text positioning and animateMotion, one variant.
        assert_eq!(AttrId::from_name("rotate"), Some(AttrId::Rotate));
        // `operator`: shared by feComposite and feMorphology, one variant.
        assert_eq!(AttrId::from_name("operator"), Some(AttrId::Operator));
        // `d`: path geometry attribute and (SVG 2) CSS property, one variant.
        assert_eq!(AttrId::from_name("d"), Some(AttrId::D));
        // `path`: animateMotion's path data and textPath's inline path data.
        assert_eq!(AttrId::from_name("path"), Some(AttrId::Path));
    }
}
