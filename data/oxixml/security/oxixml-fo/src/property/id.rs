// This file is generated from the XSL 1.1 Recommendation property index
// (<https://www.w3.org/TR/xsl11/#property-index>) by `tools/gen.py`. Do not edit by hand.

/// Identifier of one of the 272 XSL-FO 1.1 formatting properties.
///
/// Every property named in [Appendix D, *Property Index*] of the XSL 1.1
/// Recommendation has exactly one variant here, including the twenty-four
/// shorthands. The discriminants are stable within a release but carry no
/// meaning; use [`PropertyId::name`] to obtain the spelling used in a
/// stylesheet and [`PropertyId::parse`] for the inverse direction.
///
/// [Appendix D, *Property Index*]: https://www.w3.org/TR/xsl11/#property-index
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[non_exhaustive]
#[repr(u16)]
pub enum PropertyId {
    /// The `absolute-position` property.
    AbsolutePosition = 0,
    /// The `active-state` property.
    ActiveState = 1,
    /// The `alignment-adjust` property.
    AlignmentAdjust = 2,
    /// The `alignment-baseline` property.
    AlignmentBaseline = 3,
    /// The `allowed-height-scale` property.
    AllowedHeightScale = 4,
    /// The `allowed-width-scale` property.
    AllowedWidthScale = 5,
    /// The `auto-restore` property.
    AutoRestore = 6,
    /// The `azimuth` property.
    Azimuth = 7,
    /// The `background` property.
    Background = 8,
    /// The `background-attachment` property.
    BackgroundAttachment = 9,
    /// The `background-color` property.
    BackgroundColor = 10,
    /// The `background-image` property.
    BackgroundImage = 11,
    /// The `background-position` property.
    BackgroundPosition = 12,
    /// The `background-position-horizontal` property.
    BackgroundPositionHorizontal = 13,
    /// The `background-position-vertical` property.
    BackgroundPositionVertical = 14,
    /// The `background-repeat` property.
    BackgroundRepeat = 15,
    /// The `baseline-shift` property.
    BaselineShift = 16,
    /// The `blank-or-not-blank` property.
    BlankOrNotBlank = 17,
    /// The `block-progression-dimension` property.
    BlockProgressionDimension = 18,
    /// The `border` property.
    Border = 19,
    /// The `border-after-color` property.
    BorderAfterColor = 20,
    /// The `border-after-precedence` property.
    BorderAfterPrecedence = 21,
    /// The `border-after-style` property.
    BorderAfterStyle = 22,
    /// The `border-after-width` property.
    BorderAfterWidth = 23,
    /// The `border-before-color` property.
    BorderBeforeColor = 24,
    /// The `border-before-precedence` property.
    BorderBeforePrecedence = 25,
    /// The `border-before-style` property.
    BorderBeforeStyle = 26,
    /// The `border-before-width` property.
    BorderBeforeWidth = 27,
    /// The `border-bottom` property.
    BorderBottom = 28,
    /// The `border-bottom-color` property.
    BorderBottomColor = 29,
    /// The `border-bottom-style` property.
    BorderBottomStyle = 30,
    /// The `border-bottom-width` property.
    BorderBottomWidth = 31,
    /// The `border-collapse` property.
    BorderCollapse = 32,
    /// The `border-color` property.
    BorderColor = 33,
    /// The `border-end-color` property.
    BorderEndColor = 34,
    /// The `border-end-precedence` property.
    BorderEndPrecedence = 35,
    /// The `border-end-style` property.
    BorderEndStyle = 36,
    /// The `border-end-width` property.
    BorderEndWidth = 37,
    /// The `border-left` property.
    BorderLeft = 38,
    /// The `border-left-color` property.
    BorderLeftColor = 39,
    /// The `border-left-style` property.
    BorderLeftStyle = 40,
    /// The `border-left-width` property.
    BorderLeftWidth = 41,
    /// The `border-right` property.
    BorderRight = 42,
    /// The `border-right-color` property.
    BorderRightColor = 43,
    /// The `border-right-style` property.
    BorderRightStyle = 44,
    /// The `border-right-width` property.
    BorderRightWidth = 45,
    /// The `border-separation` property.
    BorderSeparation = 46,
    /// The `border-spacing` property.
    BorderSpacing = 47,
    /// The `border-start-color` property.
    BorderStartColor = 48,
    /// The `border-start-precedence` property.
    BorderStartPrecedence = 49,
    /// The `border-start-style` property.
    BorderStartStyle = 50,
    /// The `border-start-width` property.
    BorderStartWidth = 51,
    /// The `border-style` property.
    BorderStyle = 52,
    /// The `border-top` property.
    BorderTop = 53,
    /// The `border-top-color` property.
    BorderTopColor = 54,
    /// The `border-top-style` property.
    BorderTopStyle = 55,
    /// The `border-top-width` property.
    BorderTopWidth = 56,
    /// The `border-width` property.
    BorderWidth = 57,
    /// The `bottom` property.
    Bottom = 58,
    /// The `break-after` property.
    BreakAfter = 59,
    /// The `break-before` property.
    BreakBefore = 60,
    /// The `caption-side` property.
    CaptionSide = 61,
    /// The `case-name` property.
    CaseName = 62,
    /// The `case-title` property.
    CaseTitle = 63,
    /// The `change-bar-class` property.
    ChangeBarClass = 64,
    /// The `change-bar-color` property.
    ChangeBarColor = 65,
    /// The `change-bar-offset` property.
    ChangeBarOffset = 66,
    /// The `change-bar-placement` property.
    ChangeBarPlacement = 67,
    /// The `change-bar-style` property.
    ChangeBarStyle = 68,
    /// The `change-bar-width` property.
    ChangeBarWidth = 69,
    /// The `character` property.
    Character = 70,
    /// The `clear` property.
    Clear = 71,
    /// The `clip` property.
    Clip = 72,
    /// The `color` property.
    Color = 73,
    /// The `color-profile-name` property.
    ColorProfileName = 74,
    /// The `column-count` property.
    ColumnCount = 75,
    /// The `column-gap` property.
    ColumnGap = 76,
    /// The `column-number` property.
    ColumnNumber = 77,
    /// The `column-width` property.
    ColumnWidth = 78,
    /// The `content-height` property.
    ContentHeight = 79,
    /// The `content-type` property.
    ContentType = 80,
    /// The `content-width` property.
    ContentWidth = 81,
    /// The `country` property.
    Country = 82,
    /// The `cue` property.
    Cue = 83,
    /// The `cue-after` property.
    CueAfter = 84,
    /// The `cue-before` property.
    CueBefore = 85,
    /// The `destination-placement-offset` property.
    DestinationPlacementOffset = 86,
    /// The `direction` property.
    Direction = 87,
    /// The `display-align` property.
    DisplayAlign = 88,
    /// The `dominant-baseline` property.
    DominantBaseline = 89,
    /// The `elevation` property.
    Elevation = 90,
    /// The `empty-cells` property.
    EmptyCells = 91,
    /// The `end-indent` property.
    EndIndent = 92,
    /// The `ends-row` property.
    EndsRow = 93,
    /// The `extent` property.
    Extent = 94,
    /// The `external-destination` property.
    ExternalDestination = 95,
    /// The `float` property.
    Float = 96,
    /// The `flow-map-name` property.
    FlowMapName = 97,
    /// The `flow-map-reference` property.
    FlowMapReference = 98,
    /// The `flow-name` property.
    FlowName = 99,
    /// The `flow-name-reference` property.
    FlowNameReference = 100,
    /// The `font` property.
    Font = 101,
    /// The `font-family` property.
    FontFamily = 102,
    /// The `font-selection-strategy` property.
    FontSelectionStrategy = 103,
    /// The `font-size` property.
    FontSize = 104,
    /// The `font-size-adjust` property.
    FontSizeAdjust = 105,
    /// The `font-stretch` property.
    FontStretch = 106,
    /// The `font-style` property.
    FontStyle = 107,
    /// The `font-variant` property.
    FontVariant = 108,
    /// The `font-weight` property.
    FontWeight = 109,
    /// The `force-page-count` property.
    ForcePageCount = 110,
    /// The `format` property.
    Format = 111,
    /// The `glyph-orientation-horizontal` property.
    GlyphOrientationHorizontal = 112,
    /// The `glyph-orientation-vertical` property.
    GlyphOrientationVertical = 113,
    /// The `grouping-separator` property.
    GroupingSeparator = 114,
    /// The `grouping-size` property.
    GroupingSize = 115,
    /// The `height` property.
    Height = 116,
    /// The `hyphenate` property.
    Hyphenate = 117,
    /// The `hyphenation-character` property.
    HyphenationCharacter = 118,
    /// The `hyphenation-keep` property.
    HyphenationKeep = 119,
    /// The `hyphenation-ladder-count` property.
    HyphenationLadderCount = 120,
    /// The `hyphenation-push-character-count` property.
    HyphenationPushCharacterCount = 121,
    /// The `hyphenation-remain-character-count` property.
    HyphenationRemainCharacterCount = 122,
    /// The `id` property.
    Id = 123,
    /// The `index-class` property.
    IndexClass = 124,
    /// The `index-key` property.
    IndexKey = 125,
    /// The `indicate-destination` property.
    IndicateDestination = 126,
    /// The `initial-page-number` property.
    InitialPageNumber = 127,
    /// The `inline-progression-dimension` property.
    InlineProgressionDimension = 128,
    /// The `internal-destination` property.
    InternalDestination = 129,
    /// The `intrinsic-scale-value` property.
    IntrinsicScaleValue = 130,
    /// The `intrusion-displace` property.
    IntrusionDisplace = 131,
    /// The `keep-together` property.
    KeepTogether = 132,
    /// The `keep-with-next` property.
    KeepWithNext = 133,
    /// The `keep-with-previous` property.
    KeepWithPrevious = 134,
    /// The `language` property.
    Language = 135,
    /// The `last-line-end-indent` property.
    LastLineEndIndent = 136,
    /// The `leader-alignment` property.
    LeaderAlignment = 137,
    /// The `leader-length` property.
    LeaderLength = 138,
    /// The `leader-pattern` property.
    LeaderPattern = 139,
    /// The `leader-pattern-width` property.
    LeaderPatternWidth = 140,
    /// The `left` property.
    Left = 141,
    /// The `letter-spacing` property.
    LetterSpacing = 142,
    /// The `letter-value` property.
    LetterValue = 143,
    /// The `line-height` property.
    LineHeight = 144,
    /// The `line-height-shift-adjustment` property.
    LineHeightShiftAdjustment = 145,
    /// The `line-stacking-strategy` property.
    LineStackingStrategy = 146,
    /// The `linefeed-treatment` property.
    LinefeedTreatment = 147,
    /// The `margin` property.
    Margin = 148,
    /// The `margin-bottom` property.
    MarginBottom = 149,
    /// The `margin-left` property.
    MarginLeft = 150,
    /// The `margin-right` property.
    MarginRight = 151,
    /// The `margin-top` property.
    MarginTop = 152,
    /// The `marker-class-name` property.
    MarkerClassName = 153,
    /// The `master-name` property.
    MasterName = 154,
    /// The `master-reference` property.
    MasterReference = 155,
    /// The `max-height` property.
    MaxHeight = 156,
    /// The `max-width` property.
    MaxWidth = 157,
    /// The `maximum-repeats` property.
    MaximumRepeats = 158,
    /// The `media-usage` property.
    MediaUsage = 159,
    /// The `merge-pages-across-index-key-references` property.
    MergePagesAcrossIndexKeyReferences = 160,
    /// The `merge-ranges-across-index-key-references` property.
    MergeRangesAcrossIndexKeyReferences = 161,
    /// The `merge-sequential-page-numbers` property.
    MergeSequentialPageNumbers = 162,
    /// The `min-height` property.
    MinHeight = 163,
    /// The `min-width` property.
    MinWidth = 164,
    /// The `number-columns-repeated` property.
    NumberColumnsRepeated = 165,
    /// The `number-columns-spanned` property.
    NumberColumnsSpanned = 166,
    /// The `number-rows-spanned` property.
    NumberRowsSpanned = 167,
    /// The `odd-or-even` property.
    OddOrEven = 168,
    /// The `orphans` property.
    Orphans = 169,
    /// The `overflow` property.
    Overflow = 170,
    /// The `padding` property.
    Padding = 171,
    /// The `padding-after` property.
    PaddingAfter = 172,
    /// The `padding-before` property.
    PaddingBefore = 173,
    /// The `padding-bottom` property.
    PaddingBottom = 174,
    /// The `padding-end` property.
    PaddingEnd = 175,
    /// The `padding-left` property.
    PaddingLeft = 176,
    /// The `padding-right` property.
    PaddingRight = 177,
    /// The `padding-start` property.
    PaddingStart = 178,
    /// The `padding-top` property.
    PaddingTop = 179,
    /// The `page-break-after` property.
    PageBreakAfter = 180,
    /// The `page-break-before` property.
    PageBreakBefore = 181,
    /// The `page-break-inside` property.
    PageBreakInside = 182,
    /// The `page-citation-strategy` property.
    PageCitationStrategy = 183,
    /// The `page-height` property.
    PageHeight = 184,
    /// The `page-number-treatment` property.
    PageNumberTreatment = 185,
    /// The `page-position` property.
    PagePosition = 186,
    /// The `page-width` property.
    PageWidth = 187,
    /// The `pause` property.
    Pause = 188,
    /// The `pause-after` property.
    PauseAfter = 189,
    /// The `pause-before` property.
    PauseBefore = 190,
    /// The `pitch` property.
    Pitch = 191,
    /// The `pitch-range` property.
    PitchRange = 192,
    /// The `play-during` property.
    PlayDuring = 193,
    /// The `position` property.
    Position = 194,
    /// The `precedence` property.
    Precedence = 195,
    /// The `provisional-distance-between-starts` property.
    ProvisionalDistanceBetweenStarts = 196,
    /// The `provisional-label-separation` property.
    ProvisionalLabelSeparation = 197,
    /// The `ref-id` property.
    RefId = 198,
    /// The `ref-index-key` property.
    RefIndexKey = 199,
    /// The `reference-orientation` property.
    ReferenceOrientation = 200,
    /// The `region-name` property.
    RegionName = 201,
    /// The `region-name-reference` property.
    RegionNameReference = 202,
    /// The `relative-align` property.
    RelativeAlign = 203,
    /// The `relative-position` property.
    RelativePosition = 204,
    /// The `rendering-intent` property.
    RenderingIntent = 205,
    /// The `retrieve-boundary` property.
    RetrieveBoundary = 206,
    /// The `retrieve-boundary-within-table` property.
    RetrieveBoundaryWithinTable = 207,
    /// The `retrieve-class-name` property.
    RetrieveClassName = 208,
    /// The `retrieve-position` property.
    RetrievePosition = 209,
    /// The `retrieve-position-within-table` property.
    RetrievePositionWithinTable = 210,
    /// The `richness` property.
    Richness = 211,
    /// The `right` property.
    Right = 212,
    /// The `role` property.
    Role = 213,
    /// The `rule-style` property.
    RuleStyle = 214,
    /// The `rule-thickness` property.
    RuleThickness = 215,
    /// The `scale-option` property.
    ScaleOption = 216,
    /// The `scaling` property.
    Scaling = 217,
    /// The `scaling-method` property.
    ScalingMethod = 218,
    /// The `score-spaces` property.
    ScoreSpaces = 219,
    /// The `script` property.
    Script = 220,
    /// The `show-destination` property.
    ShowDestination = 221,
    /// The `size` property.
    Size = 222,
    /// The `source-document` property.
    SourceDocument = 223,
    /// The `space-after` property.
    SpaceAfter = 224,
    /// The `space-before` property.
    SpaceBefore = 225,
    /// The `space-end` property.
    SpaceEnd = 226,
    /// The `space-start` property.
    SpaceStart = 227,
    /// The `span` property.
    Span = 228,
    /// The `speak` property.
    Speak = 229,
    /// The `speak-header` property.
    SpeakHeader = 230,
    /// The `speak-numeral` property.
    SpeakNumeral = 231,
    /// The `speak-punctuation` property.
    SpeakPunctuation = 232,
    /// The `speech-rate` property.
    SpeechRate = 233,
    /// The `src` property.
    Src = 234,
    /// The `start-indent` property.
    StartIndent = 235,
    /// The `starting-state` property.
    StartingState = 236,
    /// The `starts-row` property.
    StartsRow = 237,
    /// The `stress` property.
    Stress = 238,
    /// The `suppress-at-line-break` property.
    SuppressAtLineBreak = 239,
    /// The `switch-to` property.
    SwitchTo = 240,
    /// The `table-layout` property.
    TableLayout = 241,
    /// The `table-omit-footer-at-break` property.
    TableOmitFooterAtBreak = 242,
    /// The `table-omit-header-at-break` property.
    TableOmitHeaderAtBreak = 243,
    /// The `target-presentation-context` property.
    TargetPresentationContext = 244,
    /// The `target-processing-context` property.
    TargetProcessingContext = 245,
    /// The `target-stylesheet` property.
    TargetStylesheet = 246,
    /// The `text-align` property.
    TextAlign = 247,
    /// The `text-align-last` property.
    TextAlignLast = 248,
    /// The `text-altitude` property.
    TextAltitude = 249,
    /// The `text-decoration` property.
    TextDecoration = 250,
    /// The `text-depth` property.
    TextDepth = 251,
    /// The `text-indent` property.
    TextIndent = 252,
    /// The `text-shadow` property.
    TextShadow = 253,
    /// The `text-transform` property.
    TextTransform = 254,
    /// The `top` property.
    Top = 255,
    /// The `treat-as-word-space` property.
    TreatAsWordSpace = 256,
    /// The `unicode-bidi` property.
    UnicodeBidi = 257,
    /// The `vertical-align` property.
    VerticalAlign = 258,
    /// The `visibility` property.
    Visibility = 259,
    /// The `voice-family` property.
    VoiceFamily = 260,
    /// The `volume` property.
    Volume = 261,
    /// The `white-space` property.
    WhiteSpace = 262,
    /// The `white-space-collapse` property.
    WhiteSpaceCollapse = 263,
    /// The `white-space-treatment` property.
    WhiteSpaceTreatment = 264,
    /// The `widows` property.
    Widows = 265,
    /// The `width` property.
    Width = 266,
    /// The `word-spacing` property.
    WordSpacing = 267,
    /// The `wrap-option` property.
    WrapOption = 268,
    /// The `writing-mode` property.
    WritingMode = 269,
    /// The `xml:lang` property.
    XmlLang = 270,
    /// The `z-index` property.
    ZIndex = 271,
}

/// Number of distinct XSL-FO 1.1 properties.
pub const PROPERTY_COUNT: usize = 272;

pub(crate) static PROPERTY_NAMES: [&str; PROPERTY_COUNT] = [
    "absolute-position",
    "active-state",
    "alignment-adjust",
    "alignment-baseline",
    "allowed-height-scale",
    "allowed-width-scale",
    "auto-restore",
    "azimuth",
    "background",
    "background-attachment",
    "background-color",
    "background-image",
    "background-position",
    "background-position-horizontal",
    "background-position-vertical",
    "background-repeat",
    "baseline-shift",
    "blank-or-not-blank",
    "block-progression-dimension",
    "border",
    "border-after-color",
    "border-after-precedence",
    "border-after-style",
    "border-after-width",
    "border-before-color",
    "border-before-precedence",
    "border-before-style",
    "border-before-width",
    "border-bottom",
    "border-bottom-color",
    "border-bottom-style",
    "border-bottom-width",
    "border-collapse",
    "border-color",
    "border-end-color",
    "border-end-precedence",
    "border-end-style",
    "border-end-width",
    "border-left",
    "border-left-color",
    "border-left-style",
    "border-left-width",
    "border-right",
    "border-right-color",
    "border-right-style",
    "border-right-width",
    "border-separation",
    "border-spacing",
    "border-start-color",
    "border-start-precedence",
    "border-start-style",
    "border-start-width",
    "border-style",
    "border-top",
    "border-top-color",
    "border-top-style",
    "border-top-width",
    "border-width",
    "bottom",
    "break-after",
    "break-before",
    "caption-side",
    "case-name",
    "case-title",
    "change-bar-class",
    "change-bar-color",
    "change-bar-offset",
    "change-bar-placement",
    "change-bar-style",
    "change-bar-width",
    "character",
    "clear",
    "clip",
    "color",
    "color-profile-name",
    "column-count",
    "column-gap",
    "column-number",
    "column-width",
    "content-height",
    "content-type",
    "content-width",
    "country",
    "cue",
    "cue-after",
    "cue-before",
    "destination-placement-offset",
    "direction",
    "display-align",
    "dominant-baseline",
    "elevation",
    "empty-cells",
    "end-indent",
    "ends-row",
    "extent",
    "external-destination",
    "float",
    "flow-map-name",
    "flow-map-reference",
    "flow-name",
    "flow-name-reference",
    "font",
    "font-family",
    "font-selection-strategy",
    "font-size",
    "font-size-adjust",
    "font-stretch",
    "font-style",
    "font-variant",
    "font-weight",
    "force-page-count",
    "format",
    "glyph-orientation-horizontal",
    "glyph-orientation-vertical",
    "grouping-separator",
    "grouping-size",
    "height",
    "hyphenate",
    "hyphenation-character",
    "hyphenation-keep",
    "hyphenation-ladder-count",
    "hyphenation-push-character-count",
    "hyphenation-remain-character-count",
    "id",
    "index-class",
    "index-key",
    "indicate-destination",
    "initial-page-number",
    "inline-progression-dimension",
    "internal-destination",
    "intrinsic-scale-value",
    "intrusion-displace",
    "keep-together",
    "keep-with-next",
    "keep-with-previous",
    "language",
    "last-line-end-indent",
    "leader-alignment",
    "leader-length",
    "leader-pattern",
    "leader-pattern-width",
    "left",
    "letter-spacing",
    "letter-value",
    "line-height",
    "line-height-shift-adjustment",
    "line-stacking-strategy",
    "linefeed-treatment",
    "margin",
    "margin-bottom",
    "margin-left",
    "margin-right",
    "margin-top",
    "marker-class-name",
    "master-name",
    "master-reference",
    "max-height",
    "max-width",
    "maximum-repeats",
    "media-usage",
    "merge-pages-across-index-key-references",
    "merge-ranges-across-index-key-references",
    "merge-sequential-page-numbers",
    "min-height",
    "min-width",
    "number-columns-repeated",
    "number-columns-spanned",
    "number-rows-spanned",
    "odd-or-even",
    "orphans",
    "overflow",
    "padding",
    "padding-after",
    "padding-before",
    "padding-bottom",
    "padding-end",
    "padding-left",
    "padding-right",
    "padding-start",
    "padding-top",
    "page-break-after",
    "page-break-before",
    "page-break-inside",
    "page-citation-strategy",
    "page-height",
    "page-number-treatment",
    "page-position",
    "page-width",
    "pause",
    "pause-after",
    "pause-before",
    "pitch",
    "pitch-range",
    "play-during",
    "position",
    "precedence",
    "provisional-distance-between-starts",
    "provisional-label-separation",
    "ref-id",
    "ref-index-key",
    "reference-orientation",
    "region-name",
    "region-name-reference",
    "relative-align",
    "relative-position",
    "rendering-intent",
    "retrieve-boundary",
    "retrieve-boundary-within-table",
    "retrieve-class-name",
    "retrieve-position",
    "retrieve-position-within-table",
    "richness",
    "right",
    "role",
    "rule-style",
    "rule-thickness",
    "scale-option",
    "scaling",
    "scaling-method",
    "score-spaces",
    "script",
    "show-destination",
    "size",
    "source-document",
    "space-after",
    "space-before",
    "space-end",
    "space-start",
    "span",
    "speak",
    "speak-header",
    "speak-numeral",
    "speak-punctuation",
    "speech-rate",
    "src",
    "start-indent",
    "starting-state",
    "starts-row",
    "stress",
    "suppress-at-line-break",
    "switch-to",
    "table-layout",
    "table-omit-footer-at-break",
    "table-omit-header-at-break",
    "target-presentation-context",
    "target-processing-context",
    "target-stylesheet",
    "text-align",
    "text-align-last",
    "text-altitude",
    "text-decoration",
    "text-depth",
    "text-indent",
    "text-shadow",
    "text-transform",
    "top",
    "treat-as-word-space",
    "unicode-bidi",
    "vertical-align",
    "visibility",
    "voice-family",
    "volume",
    "white-space",
    "white-space-collapse",
    "white-space-treatment",
    "widows",
    "width",
    "word-spacing",
    "wrap-option",
    "writing-mode",
    "xml:lang",
    "z-index",
];

pub(crate) static PROPERTY_IDS: [PropertyId; PROPERTY_COUNT] = [
    PropertyId::AbsolutePosition,
    PropertyId::ActiveState,
    PropertyId::AlignmentAdjust,
    PropertyId::AlignmentBaseline,
    PropertyId::AllowedHeightScale,
    PropertyId::AllowedWidthScale,
    PropertyId::AutoRestore,
    PropertyId::Azimuth,
    PropertyId::Background,
    PropertyId::BackgroundAttachment,
    PropertyId::BackgroundColor,
    PropertyId::BackgroundImage,
    PropertyId::BackgroundPosition,
    PropertyId::BackgroundPositionHorizontal,
    PropertyId::BackgroundPositionVertical,
    PropertyId::BackgroundRepeat,
    PropertyId::BaselineShift,
    PropertyId::BlankOrNotBlank,
    PropertyId::BlockProgressionDimension,
    PropertyId::Border,
    PropertyId::BorderAfterColor,
    PropertyId::BorderAfterPrecedence,
    PropertyId::BorderAfterStyle,
    PropertyId::BorderAfterWidth,
    PropertyId::BorderBeforeColor,
    PropertyId::BorderBeforePrecedence,
    PropertyId::BorderBeforeStyle,
    PropertyId::BorderBeforeWidth,
    PropertyId::BorderBottom,
    PropertyId::BorderBottomColor,
    PropertyId::BorderBottomStyle,
    PropertyId::BorderBottomWidth,
    PropertyId::BorderCollapse,
    PropertyId::BorderColor,
    PropertyId::BorderEndColor,
    PropertyId::BorderEndPrecedence,
    PropertyId::BorderEndStyle,
    PropertyId::BorderEndWidth,
    PropertyId::BorderLeft,
    PropertyId::BorderLeftColor,
    PropertyId::BorderLeftStyle,
    PropertyId::BorderLeftWidth,
    PropertyId::BorderRight,
    PropertyId::BorderRightColor,
    PropertyId::BorderRightStyle,
    PropertyId::BorderRightWidth,
    PropertyId::BorderSeparation,
    PropertyId::BorderSpacing,
    PropertyId::BorderStartColor,
    PropertyId::BorderStartPrecedence,
    PropertyId::BorderStartStyle,
    PropertyId::BorderStartWidth,
    PropertyId::BorderStyle,
    PropertyId::BorderTop,
    PropertyId::BorderTopColor,
    PropertyId::BorderTopStyle,
    PropertyId::BorderTopWidth,
    PropertyId::BorderWidth,
    PropertyId::Bottom,
    PropertyId::BreakAfter,
    PropertyId::BreakBefore,
    PropertyId::CaptionSide,
    PropertyId::CaseName,
    PropertyId::CaseTitle,
    PropertyId::ChangeBarClass,
    PropertyId::ChangeBarColor,
    PropertyId::ChangeBarOffset,
    PropertyId::ChangeBarPlacement,
    PropertyId::ChangeBarStyle,
    PropertyId::ChangeBarWidth,
    PropertyId::Character,
    PropertyId::Clear,
    PropertyId::Clip,
    PropertyId::Color,
    PropertyId::ColorProfileName,
    PropertyId::ColumnCount,
    PropertyId::ColumnGap,
    PropertyId::ColumnNumber,
    PropertyId::ColumnWidth,
    PropertyId::ContentHeight,
    PropertyId::ContentType,
    PropertyId::ContentWidth,
    PropertyId::Country,
    PropertyId::Cue,
    PropertyId::CueAfter,
    PropertyId::CueBefore,
    PropertyId::DestinationPlacementOffset,
    PropertyId::Direction,
    PropertyId::DisplayAlign,
    PropertyId::DominantBaseline,
    PropertyId::Elevation,
    PropertyId::EmptyCells,
    PropertyId::EndIndent,
    PropertyId::EndsRow,
    PropertyId::Extent,
    PropertyId::ExternalDestination,
    PropertyId::Float,
    PropertyId::FlowMapName,
    PropertyId::FlowMapReference,
    PropertyId::FlowName,
    PropertyId::FlowNameReference,
    PropertyId::Font,
    PropertyId::FontFamily,
    PropertyId::FontSelectionStrategy,
    PropertyId::FontSize,
    PropertyId::FontSizeAdjust,
    PropertyId::FontStretch,
    PropertyId::FontStyle,
    PropertyId::FontVariant,
    PropertyId::FontWeight,
    PropertyId::ForcePageCount,
    PropertyId::Format,
    PropertyId::GlyphOrientationHorizontal,
    PropertyId::GlyphOrientationVertical,
    PropertyId::GroupingSeparator,
    PropertyId::GroupingSize,
    PropertyId::Height,
    PropertyId::Hyphenate,
    PropertyId::HyphenationCharacter,
    PropertyId::HyphenationKeep,
    PropertyId::HyphenationLadderCount,
    PropertyId::HyphenationPushCharacterCount,
    PropertyId::HyphenationRemainCharacterCount,
    PropertyId::Id,
    PropertyId::IndexClass,
    PropertyId::IndexKey,
    PropertyId::IndicateDestination,
    PropertyId::InitialPageNumber,
    PropertyId::InlineProgressionDimension,
    PropertyId::InternalDestination,
    PropertyId::IntrinsicScaleValue,
    PropertyId::IntrusionDisplace,
    PropertyId::KeepTogether,
    PropertyId::KeepWithNext,
    PropertyId::KeepWithPrevious,
    PropertyId::Language,
    PropertyId::LastLineEndIndent,
    PropertyId::LeaderAlignment,
    PropertyId::LeaderLength,
    PropertyId::LeaderPattern,
    PropertyId::LeaderPatternWidth,
    PropertyId::Left,
    PropertyId::LetterSpacing,
    PropertyId::LetterValue,
    PropertyId::LineHeight,
    PropertyId::LineHeightShiftAdjustment,
    PropertyId::LineStackingStrategy,
    PropertyId::LinefeedTreatment,
    PropertyId::Margin,
    PropertyId::MarginBottom,
    PropertyId::MarginLeft,
    PropertyId::MarginRight,
    PropertyId::MarginTop,
    PropertyId::MarkerClassName,
    PropertyId::MasterName,
    PropertyId::MasterReference,
    PropertyId::MaxHeight,
    PropertyId::MaxWidth,
    PropertyId::MaximumRepeats,
    PropertyId::MediaUsage,
    PropertyId::MergePagesAcrossIndexKeyReferences,
    PropertyId::MergeRangesAcrossIndexKeyReferences,
    PropertyId::MergeSequentialPageNumbers,
    PropertyId::MinHeight,
    PropertyId::MinWidth,
    PropertyId::NumberColumnsRepeated,
    PropertyId::NumberColumnsSpanned,
    PropertyId::NumberRowsSpanned,
    PropertyId::OddOrEven,
    PropertyId::Orphans,
    PropertyId::Overflow,
    PropertyId::Padding,
    PropertyId::PaddingAfter,
    PropertyId::PaddingBefore,
    PropertyId::PaddingBottom,
    PropertyId::PaddingEnd,
    PropertyId::PaddingLeft,
    PropertyId::PaddingRight,
    PropertyId::PaddingStart,
    PropertyId::PaddingTop,
    PropertyId::PageBreakAfter,
    PropertyId::PageBreakBefore,
    PropertyId::PageBreakInside,
    PropertyId::PageCitationStrategy,
    PropertyId::PageHeight,
    PropertyId::PageNumberTreatment,
    PropertyId::PagePosition,
    PropertyId::PageWidth,
    PropertyId::Pause,
    PropertyId::PauseAfter,
    PropertyId::PauseBefore,
    PropertyId::Pitch,
    PropertyId::PitchRange,
    PropertyId::PlayDuring,
    PropertyId::Position,
    PropertyId::Precedence,
    PropertyId::ProvisionalDistanceBetweenStarts,
    PropertyId::ProvisionalLabelSeparation,
    PropertyId::RefId,
    PropertyId::RefIndexKey,
    PropertyId::ReferenceOrientation,
    PropertyId::RegionName,
    PropertyId::RegionNameReference,
    PropertyId::RelativeAlign,
    PropertyId::RelativePosition,
    PropertyId::RenderingIntent,
    PropertyId::RetrieveBoundary,
    PropertyId::RetrieveBoundaryWithinTable,
    PropertyId::RetrieveClassName,
    PropertyId::RetrievePosition,
    PropertyId::RetrievePositionWithinTable,
    PropertyId::Richness,
    PropertyId::Right,
    PropertyId::Role,
    PropertyId::RuleStyle,
    PropertyId::RuleThickness,
    PropertyId::ScaleOption,
    PropertyId::Scaling,
    PropertyId::ScalingMethod,
    PropertyId::ScoreSpaces,
    PropertyId::Script,
    PropertyId::ShowDestination,
    PropertyId::Size,
    PropertyId::SourceDocument,
    PropertyId::SpaceAfter,
    PropertyId::SpaceBefore,
    PropertyId::SpaceEnd,
    PropertyId::SpaceStart,
    PropertyId::Span,
    PropertyId::Speak,
    PropertyId::SpeakHeader,
    PropertyId::SpeakNumeral,
    PropertyId::SpeakPunctuation,
    PropertyId::SpeechRate,
    PropertyId::Src,
    PropertyId::StartIndent,
    PropertyId::StartingState,
    PropertyId::StartsRow,
    PropertyId::Stress,
    PropertyId::SuppressAtLineBreak,
    PropertyId::SwitchTo,
    PropertyId::TableLayout,
    PropertyId::TableOmitFooterAtBreak,
    PropertyId::TableOmitHeaderAtBreak,
    PropertyId::TargetPresentationContext,
    PropertyId::TargetProcessingContext,
    PropertyId::TargetStylesheet,
    PropertyId::TextAlign,
    PropertyId::TextAlignLast,
    PropertyId::TextAltitude,
    PropertyId::TextDecoration,
    PropertyId::TextDepth,
    PropertyId::TextIndent,
    PropertyId::TextShadow,
    PropertyId::TextTransform,
    PropertyId::Top,
    PropertyId::TreatAsWordSpace,
    PropertyId::UnicodeBidi,
    PropertyId::VerticalAlign,
    PropertyId::Visibility,
    PropertyId::VoiceFamily,
    PropertyId::Volume,
    PropertyId::WhiteSpace,
    PropertyId::WhiteSpaceCollapse,
    PropertyId::WhiteSpaceTreatment,
    PropertyId::Widows,
    PropertyId::Width,
    PropertyId::WordSpacing,
    PropertyId::WrapOption,
    PropertyId::WritingMode,
    PropertyId::XmlLang,
    PropertyId::ZIndex,
];
