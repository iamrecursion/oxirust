//! The catalog of SVG element names: the union of the SVG 1.1 Second
//! Edition and SVG 2 element vocabularies.
//!
//! [`ElementId`] enumerates all 88 element names recognized by this crate —
//! 62 stable across both levels, 18 present in SVG 1.1 and removed or
//! deprecated by SVG 2, 1 introduced by SVG 2 (`feDropShadow`, defined in
//! the Filter Effects Level 1 module), 6 drafted for a future SVG level
//! (the mesh-gradient and hatch paint-server families) and parsed only in a
//! preserve-don't-interpret mode, and 1 experimental (`discard`, defined in
//! the separate SVG Animations module with inconsistent implementation
//! support). The count and the per-status breakdown (62+18+1+6+1 = 88) match
//! the closing tally of the SVG specification requirements catalog this
//! module transcribes; that catalog's own §1 header line and its closing
//! enumeration disagree with each other (87 vs. 88), but the closing
//! enumeration's per-status breakdown is internally self-consistent
//! (62+18+1+6+1 = 88) and is what this module implements.
//!
//! # Sourcing methodology
//!
//! Every element and every category predicate below is a direct
//! transcription of the specification requirements catalog compiled for
//! this crate (SVG 1.1 Second Edition `eltindex.html` ∪ SVG 2 editor's
//! draft `eltindex.html` ∪ `SVG2/changes.html`, cross-checked against the
//! Filter Effects Level 1 and SVG Animations modules). Nothing here is
//! reconstructed from general SVG familiarity where the catalog is silent;
//! where the catalog itself flags a value as secondary-sourced or
//! unconfirmed, that is called out on the relevant item.
//!
//! The catalog gives each element a free-text "Category" tag (e.g. `svg` is
//! tagged "Structural, container, (new) viewport-establishing") and,
//! separately, a "Content model class" description (e.g. `font`'s is
//! "Container: `font-face`, `missing-glyph`, `glyph`, `hkern`, `vkern`,
//! animation"). Sections 1.11–1.13 (filter primitives, animation elements,
//! scripting/styling) have no per-element Category column; for those, the
//! section heading and the prose "Purpose" text stand in as the catalog
//! entry (e.g. `feMerge`'s Purpose text "Stacks multiple inputs (container
//! for `feMergeNode`)" is read the same way a literal "container" Category
//! tag would be read elsewhere).
//!
//! Each category predicate below is defined as: **the element's catalog
//! text (Category tag, Content model class, and/or Purpose prose,
//! whichever the relevant table section provides) contains the predicate's
//! keyword.** This is a literal transcription rule applied uniformly, not a
//! re-derivation from an external "formal" SVG/MDN taxonomy — the two
//! occasionally disagree (for example, `missing-glyph`'s Category tag reads
//! "Font, container, graphics" even though its Content model class is "Same
//! as `glyph`", i.e. empty-of-general-children; this module follows the
//! catalog's literal tag, not the seemingly-conflicting content-model
//! prose, and callers needing the fine-grained truth should consult
//! [`crate::names::content`] rather than a single coarse predicate). Every
//! predicate's rustdoc enumerates its exact true-set so the transcription
//! is auditable without re-deriving it.
//!
//! `is_container` is the one predicate that reads *both* the Category tag
//! and the Content model class (a plain "container" keyword anywhere in
//! either), because several elements (`font`, `feMerge`,
//! `feComponentTransfer`) only get the word in their Content model class or
//! Purpose prose, never in a literal Category tag.

use std::fmt;

/// One SVG element name, spanning the union of the SVG 1.1 Second Edition
/// and SVG 2 element vocabularies (88 names total).
///
/// Marked `#[non_exhaustive]`: the SVG vocabulary is not closed (a future
/// SVG level, or a currently-experimental element graduating to stable,
/// would add variants), so external matches must carry a wildcard arm.
/// Code inside this crate may still match exhaustively — `non_exhaustive`
/// only constrains callers outside the defining crate.
#[non_exhaustive]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ElementId {
    // ---- 1.1 Structural elements ----
    /// `<svg>` — the viewport-establishing root/nested container element.
    Svg,
    /// `<g>` — a plain grouping container with no geometry of its own.
    G,
    /// `<defs>` — a container for elements referenced by IRI, not rendered
    /// where it appears.
    Defs,
    /// `<symbol>` — a container defining reusable, `use`-instantiated
    /// content; never rendered directly.
    Symbol,
    /// `<use>` — instantiates a referenced element as if it were a deep
    /// clone at the point of use.
    Use,
    /// `<switch>` — renders the first direct child whose conditional
    /// processing attributes all pass.
    Switch,

    // ---- 1.2 Descriptive elements ----
    /// `<desc>` — a long-form text description of its parent, never
    /// rendered visually.
    Desc,
    /// `<title>` — a short text title of its parent, never rendered
    /// visually (commonly surfaced as a tooltip).
    Title,
    /// `<metadata>` — a container for foreign-namespace metadata (for
    /// example RDF), never rendered.
    Metadata,

    // ---- 1.3 Shape elements ----
    /// `<path>` — an arbitrary vector outline described by the `d`
    /// attribute's path data mini-language.
    Path,
    /// `<rect>` — an axis-aligned rectangle, optionally with rounded
    /// corners via `rx`/`ry`.
    Rect,
    /// `<circle>` — a circle described by a center and radius.
    Circle,
    /// `<ellipse>` — an axis-aligned ellipse described by a center and two
    /// radii.
    Ellipse,
    /// `<line>` — a single straight line segment between two points.
    Line,
    /// `<polyline>` — an open sequence of connected straight line segments.
    Polyline,
    /// `<polygon>` — a closed sequence of connected straight line segments.
    Polygon,

    // ---- 1.4 Text content elements ----
    /// `<text>` — a block of positioned, rendered character data.
    Text,
    /// `<tspan>` — a nested run of character data within `text` with its
    /// own positioning/style adjustments.
    Tspan,
    /// `<textPath>` — a run of character data laid out along a referenced
    /// (SVG 1.1) or inline (SVG 2) path.
    TextPath,
    /// `<tref>` — renders a copy of the character data of a referenced
    /// element. Removed in SVG 2 in favor of `use` or direct duplication.
    Tref,
    /// `<altGlyph>` — character data rendered via explicit glyph
    /// substitution. Removed in SVG 2.
    AltGlyph,
    /// `<altGlyphDef>` — a reusable, out-of-line definition of glyph
    /// substitutions for `altGlyph` to reference. Removed in SVG 2.
    AltGlyphDef,
    /// `<altGlyphItem>` — one alternative candidate glyph set inside an
    /// `altGlyphDef`. Removed in SVG 2.
    AltGlyphItem,
    /// `<glyphRef>` — a single-glyph reference used inside `altGlyphDef`/
    /// `altGlyphItem`. Removed in SVG 2.
    GlyphRef,

    // ---- 1.5 SVG Fonts (SVG 1.1 only) ----
    /// `<font>` — a complete SVG font definition, containing `font-face`,
    /// `glyph`, `missing-glyph`, `hkern`, and `vkern` children. Removed in
    /// SVG 2 in favor of CSS Fonts / WOFF.
    Font,
    /// `<font-face>` — font-wide metrics and metadata for an SVG font.
    /// Removed in SVG 2.
    FontFace,
    /// `<font-face-src>` — a container for the candidate sources
    /// (`font-face-uri`/`font-face-name`) of an SVG font. Removed in SVG 2.
    FontFaceSrc,
    /// `<font-face-uri>` — one URI-referenced candidate source of an SVG
    /// font. Removed in SVG 2.
    FontFaceUri,
    /// `<font-face-format>` — the declared format of a `font-face-uri`'s
    /// referenced font resource. Removed in SVG 2.
    FontFaceFormat,
    /// `<font-face-name>` — a locally-installed-font candidate source of an
    /// SVG font, named rather than URI-referenced. Removed in SVG 2.
    FontFaceName,
    /// `<glyph>` — one glyph outline of an SVG font, with a content model
    /// equivalent to a path-like shape. Removed in SVG 2.
    Glyph,
    /// `<missing-glyph>` — the fallback glyph outline used when no other
    /// glyph in the font matches. Removed in SVG 2.
    MissingGlyph,
    /// `<hkern>` — one horizontal kerning-pair adjustment row of an SVG
    /// font. Removed in SVG 2.
    Hkern,
    /// `<vkern>` — one vertical kerning-pair adjustment row of an SVG font.
    /// Removed in SVG 2.
    Vkern,

    // ---- 1.6 Container/embedding elements ----
    /// `<image>` — embeds a referenced raster or SVG image resource.
    Image,
    /// `<foreignObject>` — embeds foreign-namespace content (typically
    /// HTML) inside the SVG rendering tree.
    ForeignObject,
    /// `<a>` — a hyperlink wrapper, following the same content model as its
    /// SVG parent context.
    A,
    /// `<view>` — a never-rendered, named viewport definition targeted by a
    /// fragment identifier.
    View,

    // ---- 1.7 Gradients / paint servers ----
    /// `<linearGradient>` — a paint server interpolating colors along a
    /// straight axis defined by `stop` children.
    LinearGradient,
    /// `<radialGradient>` — a paint server interpolating colors radially
    /// outward from a focal point, defined by `stop` children.
    RadialGradient,
    /// `<stop>` — one color/offset control point of a gradient.
    Stop,
    /// `<meshgradient>` — a draft mesh-gradient paint server (historically
    /// `mesh`); pushed to a future SVG level. Parse-and-preserve only, no
    /// rendering semantics implemented.
    Meshgradient,
    /// `<meshrow>` — a row of patches within a `meshgradient`. Draft;
    /// parse-and-preserve only.
    Meshrow,
    /// `<meshpatch>` — one patch within a `meshrow`. Draft; parse-and-
    /// preserve only.
    Meshpatch,
    /// `<solidcolor>` — a draft paint server representing one flat color.
    /// Draft; parse-and-preserve only.
    Solidcolor,

    // ---- 1.8 Pattern / hatch ----
    /// `<pattern>` — a paint server that tiles its content across the
    /// painted region.
    Pattern,
    /// `<hatch>` — a draft hatching paint server. Draft; parse-and-preserve
    /// only.
    Hatch,
    /// `<hatchpath>` — one stroked path within a `hatch`. Draft; parse-and-
    /// preserve only.
    Hatchpath,

    // ---- 1.9 Clipping / masking / markers ----
    /// `<clipPath>` — a never-rendered container defining a clipping
    /// region from its shape/text/`use` children.
    ClipPath,
    /// `<mask>` — a never-rendered container defining a luminance or alpha
    /// mask, with the same content model as `g`/`svg`.
    Mask,
    /// `<marker>` — a never-rendered container defining a symbol drawn at
    /// vertices of a referencing shape.
    Marker,

    // ---- 1.10 Color management (SVG 1.1 only) ----
    /// `<color-profile>` — a never-rendered ICC color-management profile
    /// reference. Removed in SVG 2.
    ColorProfile,
    /// `<cursor>` — a never-rendered custom-cursor image reference,
    /// targeted by the `cursor` property. Removed in SVG 2 in favor of CSS
    /// `cursor` with image URLs.
    Cursor,

    // ---- 1.11 Filter primitives ----
    /// `<filter>` — a never-rendered container defining a filter effect
    /// region and its chain of filter primitives.
    Filter,
    /// `<feBlend>` — blends two inputs using a Photoshop-style (SVG 1.1) or
    /// full `mix-blend-mode` (SVG 2) blend mode.
    FeBlend,
    /// `<feColorMatrix>` — applies a 4×5 (or named) color-channel matrix
    /// transform to its input.
    FeColorMatrix,
    /// `<feComponentTransfer>` — applies independent per-channel transfer
    /// functions, one per `feFuncR`/`feFuncG`/`feFuncB`/`feFuncA` child.
    FeComponentTransfer,
    /// `<feFuncR>` — the red-channel transfer function inside a
    /// `feComponentTransfer`.
    FeFuncR,
    /// `<feFuncG>` — the green-channel transfer function inside a
    /// `feComponentTransfer`.
    FeFuncG,
    /// `<feFuncB>` — the blue-channel transfer function inside a
    /// `feComponentTransfer`.
    FeFuncB,
    /// `<feFuncA>` — the alpha-channel transfer function inside a
    /// `feComponentTransfer`.
    FeFuncA,
    /// `<feComposite>` — combines two inputs with a Porter-Duff (or
    /// arithmetic) compositing operator.
    FeComposite,
    /// `<feConvolveMatrix>` — applies a discrete convolution kernel to its
    /// input.
    FeConvolveMatrix,
    /// `<feDiffuseLighting>` — computes diffuse lighting from a single
    /// light-source child against an input's alpha channel as a bump map.
    FeDiffuseLighting,
    /// `<feSpecularLighting>` — computes specular lighting from a single
    /// light-source child against an input's alpha channel as a bump map.
    FeSpecularLighting,
    /// `<feDistantLight>` — a directional, infinite-distance light source;
    /// valid only inside `feDiffuseLighting`/`feSpecularLighting`.
    FeDistantLight,
    /// `<fePointLight>` — a point light source at a fixed 3D position;
    /// valid only inside `feDiffuseLighting`/`feSpecularLighting`.
    FePointLight,
    /// `<feSpotLight>` — a directional spotlight light source; valid only
    /// inside `feDiffuseLighting`/`feSpecularLighting`.
    FeSpotLight,
    /// `<feDisplacementMap>` — displaces an input's pixels using a second
    /// input as a per-pixel displacement map.
    FeDisplacementMap,
    /// `<feDropShadow>` — a convenience shadow filter equivalent to
    /// `feGaussianBlur` → `feOffset` → `feFlood` → `feComposite(over)`.
    /// New in SVG 2 (Filter Effects Level 1).
    FeDropShadow,
    /// `<feFlood>` — fills the filter region (or primitive subregion) with
    /// a flat color.
    FeFlood,
    /// `<feGaussianBlur>` — blurs an input using a Gaussian kernel.
    FeGaussianBlur,
    /// `<feImage>` — uses a referenced raster/SVG image or element as a
    /// filter input.
    FeImage,
    /// `<feMerge>` — stacks multiple inputs in Porter-Duff `over` order,
    /// one per `feMergeNode` child.
    FeMerge,
    /// `<feMergeNode>` — one input layer inside a `feMerge`.
    FeMergeNode,
    /// `<feMorphology>` — erodes or dilates an input's alpha channel.
    FeMorphology,
    /// `<feOffset>` — translates an input by a fixed offset.
    FeOffset,
    /// `<feTile>` — tiles an input's subregion across the filter region.
    FeTile,
    /// `<feTurbulence>` — generates Perlin-noise-based fractal noise or
    /// turbulence.
    FeTurbulence,

    // ---- 1.12 Animation elements ----
    /// `<animate>` — animates a scalar attribute or property between
    /// values over time.
    Animate,
    /// `<set>` — sets an attribute or property to a single value for a
    /// duration, without interpolation.
    Set,
    /// `<animateMotion>` — moves an element along a motion path over time.
    AnimateMotion,
    /// `<animateTransform>` — animates one component of the `transform`
    /// attribute (translate/scale/rotate/skewX/skewY) over time.
    AnimateTransform,
    /// `<animateColor>` — animates a color-valued attribute or property
    /// over time. Removed in SVG 2; folded into `animate`.
    AnimateColor,
    /// `<mpath>` — inside `animateMotion`, references external path/shape
    /// geometry to move along.
    Mpath,
    /// `<discard>` — removes its target element from the document at a
    /// given time. Experimental: defined in the separate SVG Animations
    /// module, with inconsistent browser support.
    Discard,

    // ---- 1.13 Scripting / styling ----
    /// `<script>` — never-rendered embedded or referenced script source.
    Script,
    /// `<style>` — never-rendered embedded CSS style rules.
    Style,
}

impl ElementId {
    /// Every [`ElementId`] variant, in the same order as this module's
    /// declaration (grouped by the specification catalog's own section
    /// order: structural, descriptive, shape, text content, SVG fonts,
    /// container/embedding, gradients, pattern/hatch,
    /// clipping/masking/markers, color management, filter primitives,
    /// animation, scripting/styling).
    pub const ALL: &'static [ElementId] = &[
        Self::Svg,
        Self::G,
        Self::Defs,
        Self::Symbol,
        Self::Use,
        Self::Switch,
        Self::Desc,
        Self::Title,
        Self::Metadata,
        Self::Path,
        Self::Rect,
        Self::Circle,
        Self::Ellipse,
        Self::Line,
        Self::Polyline,
        Self::Polygon,
        Self::Text,
        Self::Tspan,
        Self::TextPath,
        Self::Tref,
        Self::AltGlyph,
        Self::AltGlyphDef,
        Self::AltGlyphItem,
        Self::GlyphRef,
        Self::Font,
        Self::FontFace,
        Self::FontFaceSrc,
        Self::FontFaceUri,
        Self::FontFaceFormat,
        Self::FontFaceName,
        Self::Glyph,
        Self::MissingGlyph,
        Self::Hkern,
        Self::Vkern,
        Self::Image,
        Self::ForeignObject,
        Self::A,
        Self::View,
        Self::LinearGradient,
        Self::RadialGradient,
        Self::Stop,
        Self::Meshgradient,
        Self::Meshrow,
        Self::Meshpatch,
        Self::Solidcolor,
        Self::Pattern,
        Self::Hatch,
        Self::Hatchpath,
        Self::ClipPath,
        Self::Mask,
        Self::Marker,
        Self::ColorProfile,
        Self::Cursor,
        Self::Filter,
        Self::FeBlend,
        Self::FeColorMatrix,
        Self::FeComponentTransfer,
        Self::FeFuncR,
        Self::FeFuncG,
        Self::FeFuncB,
        Self::FeFuncA,
        Self::FeComposite,
        Self::FeConvolveMatrix,
        Self::FeDiffuseLighting,
        Self::FeSpecularLighting,
        Self::FeDistantLight,
        Self::FePointLight,
        Self::FeSpotLight,
        Self::FeDisplacementMap,
        Self::FeDropShadow,
        Self::FeFlood,
        Self::FeGaussianBlur,
        Self::FeImage,
        Self::FeMerge,
        Self::FeMergeNode,
        Self::FeMorphology,
        Self::FeOffset,
        Self::FeTile,
        Self::FeTurbulence,
        Self::Animate,
        Self::Set,
        Self::AnimateMotion,
        Self::AnimateTransform,
        Self::AnimateColor,
        Self::Mpath,
        Self::Discard,
        Self::Script,
        Self::Style,
    ];

    /// Looks up an [`ElementId`] by its exact, case-sensitive SVG element
    /// name (for example `"feGaussianBlur"` or `"font-face-src"`).
    ///
    /// SVG element names are XML local names and are matched exactly as
    /// authored; no case-folding or normalization is performed.
    #[must_use]
    pub fn from_name(name: &str) -> Option<Self> {
        Some(match name {
            "svg" => Self::Svg,
            "g" => Self::G,
            "defs" => Self::Defs,
            "symbol" => Self::Symbol,
            "use" => Self::Use,
            "switch" => Self::Switch,
            "desc" => Self::Desc,
            "title" => Self::Title,
            "metadata" => Self::Metadata,
            "path" => Self::Path,
            "rect" => Self::Rect,
            "circle" => Self::Circle,
            "ellipse" => Self::Ellipse,
            "line" => Self::Line,
            "polyline" => Self::Polyline,
            "polygon" => Self::Polygon,
            "text" => Self::Text,
            "tspan" => Self::Tspan,
            "textPath" => Self::TextPath,
            "tref" => Self::Tref,
            "altGlyph" => Self::AltGlyph,
            "altGlyphDef" => Self::AltGlyphDef,
            "altGlyphItem" => Self::AltGlyphItem,
            "glyphRef" => Self::GlyphRef,
            "font" => Self::Font,
            "font-face" => Self::FontFace,
            "font-face-src" => Self::FontFaceSrc,
            "font-face-uri" => Self::FontFaceUri,
            "font-face-format" => Self::FontFaceFormat,
            "font-face-name" => Self::FontFaceName,
            "glyph" => Self::Glyph,
            "missing-glyph" => Self::MissingGlyph,
            "hkern" => Self::Hkern,
            "vkern" => Self::Vkern,
            "image" => Self::Image,
            "foreignObject" => Self::ForeignObject,
            "a" => Self::A,
            "view" => Self::View,
            "linearGradient" => Self::LinearGradient,
            "radialGradient" => Self::RadialGradient,
            "stop" => Self::Stop,
            "meshgradient" => Self::Meshgradient,
            "meshrow" => Self::Meshrow,
            "meshpatch" => Self::Meshpatch,
            "solidcolor" => Self::Solidcolor,
            "pattern" => Self::Pattern,
            "hatch" => Self::Hatch,
            "hatchpath" => Self::Hatchpath,
            "clipPath" => Self::ClipPath,
            "mask" => Self::Mask,
            "marker" => Self::Marker,
            "color-profile" => Self::ColorProfile,
            "cursor" => Self::Cursor,
            "filter" => Self::Filter,
            "feBlend" => Self::FeBlend,
            "feColorMatrix" => Self::FeColorMatrix,
            "feComponentTransfer" => Self::FeComponentTransfer,
            "feFuncR" => Self::FeFuncR,
            "feFuncG" => Self::FeFuncG,
            "feFuncB" => Self::FeFuncB,
            "feFuncA" => Self::FeFuncA,
            "feComposite" => Self::FeComposite,
            "feConvolveMatrix" => Self::FeConvolveMatrix,
            "feDiffuseLighting" => Self::FeDiffuseLighting,
            "feSpecularLighting" => Self::FeSpecularLighting,
            "feDistantLight" => Self::FeDistantLight,
            "fePointLight" => Self::FePointLight,
            "feSpotLight" => Self::FeSpotLight,
            "feDisplacementMap" => Self::FeDisplacementMap,
            "feDropShadow" => Self::FeDropShadow,
            "feFlood" => Self::FeFlood,
            "feGaussianBlur" => Self::FeGaussianBlur,
            "feImage" => Self::FeImage,
            "feMerge" => Self::FeMerge,
            "feMergeNode" => Self::FeMergeNode,
            "feMorphology" => Self::FeMorphology,
            "feOffset" => Self::FeOffset,
            "feTile" => Self::FeTile,
            "feTurbulence" => Self::FeTurbulence,
            "animate" => Self::Animate,
            "set" => Self::Set,
            "animateMotion" => Self::AnimateMotion,
            "animateTransform" => Self::AnimateTransform,
            "animateColor" => Self::AnimateColor,
            "mpath" => Self::Mpath,
            "discard" => Self::Discard,
            "script" => Self::Script,
            "style" => Self::Style,
            _ => return None,
        })
    }

    /// The exact, case-sensitive SVG element name, spelled exactly as the
    /// specification authors it (for example `"feGaussianBlur"`,
    /// `"font-face-src"`).
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Svg => "svg",
            Self::G => "g",
            Self::Defs => "defs",
            Self::Symbol => "symbol",
            Self::Use => "use",
            Self::Switch => "switch",
            Self::Desc => "desc",
            Self::Title => "title",
            Self::Metadata => "metadata",
            Self::Path => "path",
            Self::Rect => "rect",
            Self::Circle => "circle",
            Self::Ellipse => "ellipse",
            Self::Line => "line",
            Self::Polyline => "polyline",
            Self::Polygon => "polygon",
            Self::Text => "text",
            Self::Tspan => "tspan",
            Self::TextPath => "textPath",
            Self::Tref => "tref",
            Self::AltGlyph => "altGlyph",
            Self::AltGlyphDef => "altGlyphDef",
            Self::AltGlyphItem => "altGlyphItem",
            Self::GlyphRef => "glyphRef",
            Self::Font => "font",
            Self::FontFace => "font-face",
            Self::FontFaceSrc => "font-face-src",
            Self::FontFaceUri => "font-face-uri",
            Self::FontFaceFormat => "font-face-format",
            Self::FontFaceName => "font-face-name",
            Self::Glyph => "glyph",
            Self::MissingGlyph => "missing-glyph",
            Self::Hkern => "hkern",
            Self::Vkern => "vkern",
            Self::Image => "image",
            Self::ForeignObject => "foreignObject",
            Self::A => "a",
            Self::View => "view",
            Self::LinearGradient => "linearGradient",
            Self::RadialGradient => "radialGradient",
            Self::Stop => "stop",
            Self::Meshgradient => "meshgradient",
            Self::Meshrow => "meshrow",
            Self::Meshpatch => "meshpatch",
            Self::Solidcolor => "solidcolor",
            Self::Pattern => "pattern",
            Self::Hatch => "hatch",
            Self::Hatchpath => "hatchpath",
            Self::ClipPath => "clipPath",
            Self::Mask => "mask",
            Self::Marker => "marker",
            Self::ColorProfile => "color-profile",
            Self::Cursor => "cursor",
            Self::Filter => "filter",
            Self::FeBlend => "feBlend",
            Self::FeColorMatrix => "feColorMatrix",
            Self::FeComponentTransfer => "feComponentTransfer",
            Self::FeFuncR => "feFuncR",
            Self::FeFuncG => "feFuncG",
            Self::FeFuncB => "feFuncB",
            Self::FeFuncA => "feFuncA",
            Self::FeComposite => "feComposite",
            Self::FeConvolveMatrix => "feConvolveMatrix",
            Self::FeDiffuseLighting => "feDiffuseLighting",
            Self::FeSpecularLighting => "feSpecularLighting",
            Self::FeDistantLight => "feDistantLight",
            Self::FePointLight => "fePointLight",
            Self::FeSpotLight => "feSpotLight",
            Self::FeDisplacementMap => "feDisplacementMap",
            Self::FeDropShadow => "feDropShadow",
            Self::FeFlood => "feFlood",
            Self::FeGaussianBlur => "feGaussianBlur",
            Self::FeImage => "feImage",
            Self::FeMerge => "feMerge",
            Self::FeMergeNode => "feMergeNode",
            Self::FeMorphology => "feMorphology",
            Self::FeOffset => "feOffset",
            Self::FeTile => "feTile",
            Self::FeTurbulence => "feTurbulence",
            Self::Animate => "animate",
            Self::Set => "set",
            Self::AnimateMotion => "animateMotion",
            Self::AnimateTransform => "animateTransform",
            Self::AnimateColor => "animateColor",
            Self::Mpath => "mpath",
            Self::Discard => "discard",
            Self::Script => "script",
            Self::Style => "style",
        }
    }

    /// The specification status of this element: whether it is stable
    /// across SVG 1.1 and SVG 2, deprecated/removed in SVG 2, new in SVG 2,
    /// a draft-only element, or experimental. See [`ElementStatus`].
    #[must_use]
    pub const fn status(self) -> ElementStatus {
        if matches!(
            self,
            Self::Tref
                | Self::AltGlyph
                | Self::AltGlyphDef
                | Self::AltGlyphItem
                | Self::GlyphRef
                | Self::Font
                | Self::FontFace
                | Self::FontFaceSrc
                | Self::FontFaceUri
                | Self::FontFaceFormat
                | Self::FontFaceName
                | Self::Glyph
                | Self::MissingGlyph
                | Self::Hkern
                | Self::Vkern
                | Self::ColorProfile
                | Self::Cursor
                | Self::AnimateColor
        ) {
            ElementStatus::DeprecatedSvg2
        } else if matches!(self, Self::FeDropShadow) {
            ElementStatus::NewInSvg2
        } else if matches!(
            self,
            Self::Meshgradient
                | Self::Meshrow
                | Self::Meshpatch
                | Self::Solidcolor
                | Self::Hatch
                | Self::Hatchpath
        ) {
            ElementStatus::Draft
        } else if matches!(self, Self::Discard) {
            ElementStatus::Experimental
        } else {
            ElementStatus::Stable
        }
    }

    /// True for the 18 elements SVG 1.1 defined that SVG 2 removes or
    /// deprecates: `tref`, `altGlyph`, `altGlyphDef`, `altGlyphItem`,
    /// `glyphRef`, the 10-element SVG-fonts family (`font`, `font-face`,
    /// `font-face-src`, `font-face-uri`, `font-face-format`,
    /// `font-face-name`, `glyph`, `missing-glyph`, `hkern`, `vkern`),
    /// `color-profile`, `cursor`, and `animateColor`.
    ///
    /// Equivalent to `self.status() == ElementStatus::DeprecatedSvg2`. Both
    /// draft (never in SVG 1.1) and experimental elements are excluded:
    /// there is nothing for them to be "removed" from.
    #[must_use]
    pub const fn removed_in_svg2(self) -> bool {
        matches!(self.status(), ElementStatus::DeprecatedSvg2)
    }

    /// True for the 6 elements drafted for a future SVG level and not yet
    /// interoperably implemented: `meshgradient`, `meshrow`, `meshpatch`,
    /// `solidcolor`, `hatch`, `hatchpath`.
    #[must_use]
    pub const fn is_draft(self) -> bool {
        matches!(self.status(), ElementStatus::Draft)
    }

    /// True only for `discard`, the one element this catalog classifies as
    /// experimental (defined in the separate SVG Animations module, with
    /// inconsistent browser support).
    #[must_use]
    pub const fn is_experimental(self) -> bool {
        matches!(self.status(), ElementStatus::Experimental)
    }

    /// True for the 18 elements whose catalog entry (Category tag, Content
    /// model class, or — for sections without those columns — Purpose
    /// prose) contains the word "container": `svg`, `g`, `defs`, `symbol`,
    /// `switch`, `font`, `missing-glyph`, `foreignObject`, `a`,
    /// `linearGradient`, `radialGradient`, `pattern`, `clipPath`, `mask`,
    /// `marker`, `filter`, `feComponentTransfer`, `feMerge`.
    ///
    /// This is the one predicate that reads both the Category tag *and*
    /// the Content model class/Purpose prose (the union of the two),
    /// because `font`, `feComponentTransfer`, and `feMerge` are called
    /// "container" only in their content-model/purpose description, never
    /// in a Category tag. This predicate is coarse: it does not imply every
    /// member's [`crate::names::content::allows_child`] rule is the same
    /// "general container" shape — `font`'s and `feMerge`'s permitted
    /// children are much narrower than `svg`'s, for instance.
    #[must_use]
    pub const fn is_container(self) -> bool {
        matches!(
            self,
            Self::Svg
                | Self::G
                | Self::Defs
                | Self::Symbol
                | Self::Switch
                | Self::Font
                | Self::MissingGlyph
                | Self::ForeignObject
                | Self::A
                | Self::LinearGradient
                | Self::RadialGradient
                | Self::Pattern
                | Self::ClipPath
                | Self::Mask
                | Self::Marker
                | Self::Filter
                | Self::FeComponentTransfer
                | Self::FeMerge
        )
    }

    /// True for the 7 elements tagged "Structural" in the catalog's
    /// Category column: `svg`, `g`, `defs`, `symbol`, `use`, `switch`, `a`.
    #[must_use]
    pub const fn is_structural(self) -> bool {
        matches!(
            self,
            Self::Svg | Self::G | Self::Defs | Self::Symbol | Self::Use | Self::Switch | Self::A
        )
    }

    /// True for the 7 basic-shape elements tagged "Shape" in the catalog's
    /// Category column: `path`, `rect`, `circle`, `ellipse`, `line`,
    /// `polyline`, `polygon`.
    #[must_use]
    pub const fn is_shape(self) -> bool {
        matches!(
            self,
            Self::Path
                | Self::Rect
                | Self::Circle
                | Self::Ellipse
                | Self::Line
                | Self::Polyline
                | Self::Polygon
        )
    }

    /// True for the 14 elements tagged "graphics" (as a standalone tag,
    /// distinct from the separate "graphics-referencing" tag) in the
    /// catalog's Category column: `g`, `use`, `path`, `rect`, `circle`,
    /// `ellipse`, `line`, `polyline`, `polygon`, `text`, `glyph`,
    /// `missing-glyph`, `image`, `foreignObject`.
    ///
    /// Note that `svg`, `switch`, `symbol`, and `a` are *not* included:
    /// the catalog's Category tag for each of those omits "graphics"
    /// entirely, even though all four are, in practice, rendered.
    #[must_use]
    pub const fn is_graphics(self) -> bool {
        matches!(
            self,
            Self::G
                | Self::Use
                | Self::Path
                | Self::Rect
                | Self::Circle
                | Self::Ellipse
                | Self::Line
                | Self::Polyline
                | Self::Polygon
                | Self::Text
                | Self::Glyph
                | Self::MissingGlyph
                | Self::Image
                | Self::ForeignObject
        )
    }

    /// True for the 6 elements tagged "Text content" in the catalog's
    /// Category column: `text`, `tspan`, `textPath`, `tref`, `altGlyph`,
    /// `glyphRef` (whose tag is the shortened "Text content child (font
    /// glyph reference)").
    #[must_use]
    pub const fn is_text_content(self) -> bool {
        matches!(
            self,
            Self::Text
                | Self::Tspan
                | Self::TextPath
                | Self::Tref
                | Self::AltGlyph
                | Self::GlyphRef
        )
    }

    /// True for the 3 elements tagged "Gradient" in the catalog's Category
    /// column: `linearGradient`, `radialGradient`, `stop`.
    ///
    /// Narrower than [`Self::is_paint_server`]: `stop` is a gradient's
    /// control-point child, not itself a referenceable paint server.
    #[must_use]
    pub const fn is_gradient(self) -> bool {
        matches!(
            self,
            Self::LinearGradient | Self::RadialGradient | Self::Stop
        )
    }

    /// True for the 9 elements tagged "Paint server" in the catalog's
    /// Category column: `linearGradient`, `radialGradient`, `meshgradient`,
    /// `meshrow`, `meshpatch`, `solidcolor`, `pattern`, `hatch`,
    /// `hatchpath`.
    ///
    /// This literal transcription includes `meshrow`, `meshpatch`, and
    /// `hatchpath`, which the catalog tags "Paint server (draft)" even
    /// though they are structural children of a paint server rather than
    /// independently referenceable via `fill="url(#id)"` — that
    /// referenceability distinction is a separate fact for the paint-value
    /// layer ([`crate::values::paint`]) to encode, not this predicate.
    #[must_use]
    pub const fn is_paint_server(self) -> bool {
        matches!(
            self,
            Self::LinearGradient
                | Self::RadialGradient
                | Self::Meshgradient
                | Self::Meshrow
                | Self::Meshpatch
                | Self::Solidcolor
                | Self::Pattern
                | Self::Hatch
                | Self::Hatchpath
        )
    }

    /// True for all 25 `fe*`-prefixed elements, per the catalog's own
    /// summary note: "all `fe*` primitives = filter primitive elements".
    /// This includes the `feFuncR`/`feFuncG`/`feFuncB`/`feFuncA` transfer-
    /// function helpers (valid only inside `feComponentTransfer`),
    /// `feMergeNode` (valid only inside `feMerge`), and the three light
    /// source elements `feDistantLight`/`fePointLight`/`feSpotLight` (valid
    /// only inside `feDiffuseLighting`/`feSpecularLighting`) — the catalog
    /// separately calls the light sources out as "light source elements",
    /// restricted to lighting-primitive children, but still counts them
    /// among the 25 `fe*` filter primitive elements for this predicate.
    /// `filter` itself is excluded: the catalog calls it a "never-rendered
    /// container", not a filter primitive.
    #[must_use]
    pub const fn is_filter_primitive(self) -> bool {
        matches!(
            self,
            Self::FeBlend
                | Self::FeColorMatrix
                | Self::FeComponentTransfer
                | Self::FeFuncR
                | Self::FeFuncG
                | Self::FeFuncB
                | Self::FeFuncA
                | Self::FeComposite
                | Self::FeConvolveMatrix
                | Self::FeDiffuseLighting
                | Self::FeSpecularLighting
                | Self::FeDistantLight
                | Self::FePointLight
                | Self::FeSpotLight
                | Self::FeDisplacementMap
                | Self::FeDropShadow
                | Self::FeFlood
                | Self::FeGaussianBlur
                | Self::FeImage
                | Self::FeMerge
                | Self::FeMergeNode
                | Self::FeMorphology
                | Self::FeOffset
                | Self::FeTile
                | Self::FeTurbulence
        )
    }

    /// True for the 7 elements in the catalog's "Animation elements"
    /// section: `animate`, `set`, `animateMotion`, `animateTransform`,
    /// `animateColor`, `mpath`, `discard`.
    ///
    /// This follows the catalog's own section grouping literally. `mpath`
    /// does not itself animate anything (it is a path reference used only
    /// inside `animateMotion`) but the catalog tables it under "Animation
    /// elements" alongside the timed-animation elements, so it is included
    /// here; [`crate::names::content::allows_child`] still restricts where
    /// `mpath` may actually appear.
    #[must_use]
    pub const fn is_animation(self) -> bool {
        matches!(
            self,
            Self::Animate
                | Self::Set
                | Self::AnimateMotion
                | Self::AnimateTransform
                | Self::AnimateColor
                | Self::Mpath
                | Self::Discard
        )
    }

    /// True for the 3 elements in the catalog's "Descriptive elements"
    /// section: `desc`, `title`, `metadata`.
    #[must_use]
    pub const fn is_descriptive(self) -> bool {
        matches!(self, Self::Desc | Self::Title | Self::Metadata)
    }

    /// True for the 27 elements whose catalog Category tag (or, for
    /// `filter`, the section's summary prose "filter = never-rendered
    /// container") literally contains "never-rendered": `defs`, `symbol`,
    /// `desc`, `title`, `metadata`, `altGlyphDef`, `altGlyphItem`,
    /// `font-face`, `font-face-src`, `font-face-uri`, `font-face-format`,
    /// `font-face-name`, `hkern`, `vkern`, `view`, `linearGradient`,
    /// `radialGradient`, `stop`, `pattern`, `clipPath`, `mask`, `marker`,
    /// `color-profile`, `cursor`, `filter`, `script`, `style`.
    ///
    /// This is a literal transcription, not a "produces no visible pixels"
    /// superset: several elements that are equally invisible in practice
    /// (`font`, `glyph`, `missing-glyph`, the paint-server-draft family,
    /// every filter primitive, every animation element) are **not**
    /// included, simply because the catalog's own text for them never uses
    /// the phrase "never-rendered".
    #[must_use]
    pub const fn is_never_rendered(self) -> bool {
        matches!(
            self,
            Self::Defs
                | Self::Symbol
                | Self::Desc
                | Self::Title
                | Self::Metadata
                | Self::AltGlyphDef
                | Self::AltGlyphItem
                | Self::FontFace
                | Self::FontFaceSrc
                | Self::FontFaceUri
                | Self::FontFaceFormat
                | Self::FontFaceName
                | Self::Hkern
                | Self::Vkern
                | Self::View
                | Self::LinearGradient
                | Self::RadialGradient
                | Self::Stop
                | Self::Pattern
                | Self::ClipPath
                | Self::Mask
                | Self::Marker
                | Self::ColorProfile
                | Self::Cursor
                | Self::Filter
                | Self::Script
                | Self::Style
        )
    }

    /// True for the 14 elements the mission's SVG-fonts family grouping
    /// names: `font`, `font-face`, `font-face-src`, `font-face-uri`,
    /// `font-face-format`, `font-face-name`, `glyph`, `glyphRef`,
    /// `missing-glyph`, `hkern`, `vkern`, `altGlyph`, `altGlyphDef`,
    /// `altGlyphItem`.
    ///
    /// This spans two of the catalog's own sections (1.4 "Text content
    /// elements" for `altGlyph`/`altGlyphDef`/`altGlyphItem`/`glyphRef`,
    /// and 1.5 "SVG Fonts" for the other ten) because all fourteen belong
    /// to the same glyph-substitution/SVG-fonts subsystem that SVG 2
    /// removes wholesale — useful as one group for a "legacy font
    /// machinery" deprecation lint, even though the catalog itself
    /// tables them under two different content-model headings.
    #[must_use]
    pub const fn is_svg_font_related(self) -> bool {
        matches!(
            self,
            Self::Font
                | Self::FontFace
                | Self::FontFaceSrc
                | Self::FontFaceUri
                | Self::FontFaceFormat
                | Self::FontFaceName
                | Self::Glyph
                | Self::GlyphRef
                | Self::MissingGlyph
                | Self::Hkern
                | Self::Vkern
                | Self::AltGlyph
                | Self::AltGlyphDef
                | Self::AltGlyphItem
        )
    }
}

impl fmt::Display for ElementId {
    /// Writes the exact, case-sensitive SVG element name (see
    /// [`Self::as_str`]).
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

/// The specification status of an [`ElementId`], distinguishing elements
/// stable across both SVG levels from ones that are deprecated, new,
/// draft-only, or experimental.
#[non_exhaustive]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ElementStatus {
    /// Present, with unchanged rendering semantics, in both SVG 1.1 Second
    /// Edition and SVG 2 (62 elements).
    Stable,
    /// Defined by SVG 1.1 and removed or deprecated by SVG 2 (18 elements).
    DeprecatedSvg2,
    /// Introduced by SVG 2 (or a module it delegates to) with no SVG 1.1
    /// counterpart (1 element: `feDropShadow`).
    NewInSvg2,
    /// Drafted for a future SVG level, not part of the SVG 2 Candidate
    /// Recommendation core text, and not interoperably implemented (6
    /// elements: the mesh-gradient and hatch paint-server families plus
    /// `solidcolor`).
    Draft,
    /// Defined in a companion specification module with inconsistent or
    /// withdrawn implementation support (1 element: `discard`).
    Experimental,
}

impl fmt::Display for ElementStatus {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Self::Stable => "stable",
            Self::DeprecatedSvg2 => "deprecated in SVG 2",
            Self::NewInSvg2 => "new in SVG 2",
            Self::Draft => "draft",
            Self::Experimental => "experimental",
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn all_lists_every_variant_exactly_once() {
        assert_eq!(ElementId::ALL.len(), 88, "88 element names total");
        let mut seen = std::collections::HashSet::new();
        for id in ElementId::ALL {
            assert!(seen.insert(*id), "{id:?} listed more than once in ALL");
        }
    }

    #[test]
    fn from_name_as_str_round_trip_every_element() {
        for &id in ElementId::ALL {
            let name = id.as_str();
            assert_eq!(
                ElementId::from_name(name),
                Some(id),
                "round trip failed for {id:?} ({name:?})"
            );
        }
    }

    #[test]
    fn as_str_matches_exact_spec_casing() {
        assert_eq!(ElementId::Svg.as_str(), "svg");
        assert_eq!(ElementId::FeGaussianBlur.as_str(), "feGaussianBlur");
        assert_eq!(ElementId::FontFaceSrc.as_str(), "font-face-src");
        assert_eq!(ElementId::ColorProfile.as_str(), "color-profile");
        assert_eq!(ElementId::ForeignObject.as_str(), "foreignObject");
        assert_eq!(ElementId::TextPath.as_str(), "textPath");
        assert_eq!(ElementId::MissingGlyph.as_str(), "missing-glyph");
        assert_eq!(ElementId::FeMergeNode.as_str(), "feMergeNode");
    }

    #[test]
    fn from_name_is_case_sensitive() {
        assert_eq!(ElementId::from_name("SVG"), None);
        assert_eq!(ElementId::from_name("Svg"), None);
        assert_eq!(ElementId::from_name("fegaussianblur"), None);
        assert_eq!(ElementId::from_name("FeGaussianBlur"), None);
        assert_eq!(ElementId::from_name("svg"), Some(ElementId::Svg));
    }

    #[test]
    fn from_name_rejects_unknown_names() {
        assert_eq!(ElementId::from_name(""), None);
        assert_eq!(ElementId::from_name("rectangle"), None);
        assert_eq!(ElementId::from_name("div"), None);
        assert_eq!(ElementId::from_name("video"), None);
    }

    #[test]
    fn display_matches_as_str() {
        assert_eq!(ElementId::Path.to_string(), "path");
        assert_eq!(ElementId::FontFaceSrc.to_string(), "font-face-src");
    }

    #[test]
    fn status_histogram_matches_the_catalogs_closing_tally() {
        let mut stable = 0usize;
        let mut deprecated = 0usize;
        let mut new_in_svg2 = 0usize;
        let mut draft = 0usize;
        let mut experimental = 0usize;
        for &id in ElementId::ALL {
            match id.status() {
                ElementStatus::Stable => stable += 1,
                ElementStatus::DeprecatedSvg2 => deprecated += 1,
                ElementStatus::NewInSvg2 => new_in_svg2 += 1,
                ElementStatus::Draft => draft += 1,
                ElementStatus::Experimental => experimental += 1,
            }
        }
        assert_eq!(stable, 62, "62 stable elements");
        assert_eq!(deprecated, 18, "18 SVG-2-deprecated elements");
        assert_eq!(new_in_svg2, 1, "1 new-in-SVG-2 element (feDropShadow)");
        assert_eq!(draft, 6, "6 draft elements");
        assert_eq!(experimental, 1, "1 experimental element (discard)");
        assert_eq!(stable + deprecated + new_in_svg2 + draft + experimental, 88);
    }

    #[test]
    fn removed_in_svg2_excludes_draft_and_experimental() {
        assert!(ElementId::Tref.removed_in_svg2());
        assert!(ElementId::AnimateColor.removed_in_svg2());
        assert!(!ElementId::Meshgradient.removed_in_svg2());
        assert!(!ElementId::Discard.removed_in_svg2());
        assert!(!ElementId::Svg.removed_in_svg2());
    }

    #[test]
    fn is_draft_and_is_experimental_are_disjoint_from_removed() {
        for &id in ElementId::ALL {
            let flags = [id.removed_in_svg2(), id.is_draft(), id.is_experimental()];
            let true_count = flags.iter().filter(|flag| **flag).count();
            assert!(
                true_count <= 1,
                "{id:?} matched more than one status predicate"
            );
        }
    }

    #[test]
    fn container_predicate_recovers_font_and_filter_helpers() {
        // Category-tag-only reading would miss these three; is_container
        // must also read the Content model class / Purpose prose.
        assert!(ElementId::Font.is_container());
        assert!(ElementId::FeMerge.is_container());
        assert!(ElementId::FeComponentTransfer.is_container());
        assert!(ElementId::Svg.is_container());
        assert!(!ElementId::Use.is_container());
        assert!(!ElementId::Rect.is_container());
        assert!(!ElementId::FeGaussianBlur.is_container());
    }

    #[test]
    fn structural_matches_section_1_1_plus_a() {
        for id in [
            ElementId::Svg,
            ElementId::G,
            ElementId::Defs,
            ElementId::Symbol,
            ElementId::Use,
            ElementId::Switch,
            ElementId::A,
        ] {
            assert!(id.is_structural(), "{id:?} should be structural");
        }
        assert!(!ElementId::Rect.is_structural());
        assert!(!ElementId::View.is_structural());
    }

    #[test]
    fn shape_matches_the_seven_basic_shapes() {
        for id in [
            ElementId::Path,
            ElementId::Rect,
            ElementId::Circle,
            ElementId::Ellipse,
            ElementId::Line,
            ElementId::Polyline,
            ElementId::Polygon,
        ] {
            assert!(id.is_shape());
        }
        assert!(!ElementId::Text.is_shape());
        assert!(!ElementId::Glyph.is_shape());
    }

    #[test]
    fn graphics_excludes_svg_switch_symbol_and_a() {
        assert!(ElementId::G.is_graphics());
        assert!(ElementId::Rect.is_graphics());
        assert!(ElementId::Image.is_graphics());
        assert!(ElementId::Glyph.is_graphics());
        assert!(ElementId::MissingGlyph.is_graphics());
        assert!(!ElementId::Svg.is_graphics());
        assert!(!ElementId::Switch.is_graphics());
        assert!(!ElementId::Symbol.is_graphics());
        assert!(!ElementId::A.is_graphics());
        assert!(
            !ElementId::TextPath.is_graphics(),
            "graphics-referencing is not plain graphics"
        );
    }

    #[test]
    fn text_content_includes_glyph_ref() {
        for id in [
            ElementId::Text,
            ElementId::Tspan,
            ElementId::TextPath,
            ElementId::Tref,
            ElementId::AltGlyph,
            ElementId::GlyphRef,
        ] {
            assert!(id.is_text_content(), "{id:?} should be text content");
        }
        assert!(!ElementId::AltGlyphDef.is_text_content());
        assert!(!ElementId::Font.is_text_content());
    }

    #[test]
    fn gradient_is_narrower_than_paint_server() {
        assert!(ElementId::LinearGradient.is_gradient());
        assert!(ElementId::Stop.is_gradient());
        assert!(!ElementId::Pattern.is_gradient());
        assert!(!ElementId::Meshgradient.is_gradient());

        assert!(ElementId::LinearGradient.is_paint_server());
        assert!(ElementId::Pattern.is_paint_server());
        assert!(ElementId::Meshgradient.is_paint_server());
        assert!(ElementId::Hatch.is_paint_server());
        assert!(!ElementId::Stop.is_paint_server());
    }

    #[test]
    fn filter_primitive_covers_all_25_fe_elements_but_not_filter_itself() {
        let primitives = ElementId::ALL
            .iter()
            .filter(|id| id.is_filter_primitive())
            .count();
        assert_eq!(primitives, 25);
        assert!(!ElementId::Filter.is_filter_primitive());
        assert!(ElementId::FeMergeNode.is_filter_primitive());
        assert!(ElementId::FeDistantLight.is_filter_primitive());
        assert!(ElementId::FeFuncA.is_filter_primitive());
    }

    #[test]
    fn animation_matches_section_1_12_including_mpath_and_discard() {
        for id in [
            ElementId::Animate,
            ElementId::Set,
            ElementId::AnimateMotion,
            ElementId::AnimateTransform,
            ElementId::AnimateColor,
            ElementId::Mpath,
            ElementId::Discard,
        ] {
            assert!(id.is_animation());
        }
        assert!(!ElementId::Filter.is_animation());
    }

    #[test]
    fn descriptive_matches_desc_title_metadata_only() {
        assert!(ElementId::Desc.is_descriptive());
        assert!(ElementId::Title.is_descriptive());
        assert!(ElementId::Metadata.is_descriptive());
        assert_eq!(
            ElementId::ALL
                .iter()
                .filter(|id| id.is_descriptive())
                .count(),
            3
        );
    }

    #[test]
    fn never_rendered_has_exactly_27_members_and_excludes_font_and_glyph() {
        assert_eq!(
            ElementId::ALL
                .iter()
                .filter(|id| id.is_never_rendered())
                .count(),
            27
        );
        assert!(ElementId::Defs.is_never_rendered());
        assert!(ElementId::ClipPath.is_never_rendered());
        assert!(ElementId::Mask.is_never_rendered());
        assert!(ElementId::Pattern.is_never_rendered());
        assert!(ElementId::Marker.is_never_rendered());
        assert!(ElementId::Symbol.is_never_rendered());
        assert!(ElementId::Title.is_never_rendered());
        assert!(ElementId::Desc.is_never_rendered());
        assert!(ElementId::Metadata.is_never_rendered());
        assert!(ElementId::Filter.is_never_rendered());
        assert!(ElementId::Script.is_never_rendered());
        assert!(ElementId::Style.is_never_rendered());
        // Literal-transcription exclusions: invisible in practice, but the
        // catalog's own text for these never says "never-rendered".
        assert!(!ElementId::Font.is_never_rendered());
        assert!(!ElementId::Glyph.is_never_rendered());
        assert!(!ElementId::MissingGlyph.is_never_rendered());
        assert!(!ElementId::FeMerge.is_never_rendered());
        assert!(!ElementId::Animate.is_never_rendered());
    }

    #[test]
    fn svg_font_related_spans_both_source_sections() {
        for id in [
            ElementId::Font,
            ElementId::FontFace,
            ElementId::FontFaceSrc,
            ElementId::FontFaceUri,
            ElementId::FontFaceFormat,
            ElementId::FontFaceName,
            ElementId::Glyph,
            ElementId::GlyphRef,
            ElementId::MissingGlyph,
            ElementId::Hkern,
            ElementId::Vkern,
            ElementId::AltGlyph,
            ElementId::AltGlyphDef,
            ElementId::AltGlyphItem,
        ] {
            assert!(
                id.is_svg_font_related(),
                "{id:?} should be SVG-font related"
            );
        }
        assert_eq!(
            ElementId::ALL
                .iter()
                .filter(|id| id.is_svg_font_related())
                .count(),
            14
        );
        assert!(!ElementId::Text.is_svg_font_related());
    }

    #[test]
    fn element_status_display_is_human_readable() {
        assert_eq!(ElementStatus::Stable.to_string(), "stable");
        assert_eq!(
            ElementStatus::DeprecatedSvg2.to_string(),
            "deprecated in SVG 2"
        );
        assert_eq!(ElementStatus::NewInSvg2.to_string(), "new in SVG 2");
        assert_eq!(ElementStatus::Draft.to_string(), "draft");
        assert_eq!(ElementStatus::Experimental.to_string(), "experimental");
    }
}
