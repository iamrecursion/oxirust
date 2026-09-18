// The formatting-object vocabulary of XSL 1.1 (<https://www.w3.org/TR/xsl11/#fo-section>).
// Generated from the Recommendation; edit the generator rather than this table.

/// One of the 81 formatting objects defined by XSL 1.1.
///
/// Variants are named after the local part of the element name in the
/// [XSL-FO namespace](crate::FO_NAMESPACE); [`FoElement::local_name`] and
/// [`FoElement::parse`] convert between the two.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[non_exhaustive]
#[repr(u8)]
pub enum FoElement {
    /// `fo:basic-link`
    BasicLink = 0,
    /// `fo:bidi-override`
    BidiOverride = 1,
    /// `fo:block`
    Block = 2,
    /// `fo:block-container`
    BlockContainer = 3,
    /// `fo:bookmark`
    Bookmark = 4,
    /// `fo:bookmark-title`
    BookmarkTitle = 5,
    /// `fo:bookmark-tree`
    BookmarkTree = 6,
    /// `fo:change-bar-begin`
    ChangeBarBegin = 7,
    /// `fo:change-bar-end`
    ChangeBarEnd = 8,
    /// `fo:character`
    Character = 9,
    /// `fo:color-profile`
    ColorProfile = 10,
    /// `fo:conditional-page-master-reference`
    ConditionalPageMasterReference = 11,
    /// `fo:declarations`
    Declarations = 12,
    /// `fo:external-graphic`
    ExternalGraphic = 13,
    /// `fo:float`
    Float = 14,
    /// `fo:flow`
    Flow = 15,
    /// `fo:flow-assignment`
    FlowAssignment = 16,
    /// `fo:flow-map`
    FlowMap = 17,
    /// `fo:flow-name-specifier`
    FlowNameSpecifier = 18,
    /// `fo:flow-source-list`
    FlowSourceList = 19,
    /// `fo:flow-target-list`
    FlowTargetList = 20,
    /// `fo:folio-prefix`
    FolioPrefix = 21,
    /// `fo:folio-suffix`
    FolioSuffix = 22,
    /// `fo:footnote`
    Footnote = 23,
    /// `fo:footnote-body`
    FootnoteBody = 24,
    /// `fo:index-key-reference`
    IndexKeyReference = 25,
    /// `fo:index-page-citation-list`
    IndexPageCitationList = 26,
    /// `fo:index-page-citation-list-separator`
    IndexPageCitationListSeparator = 27,
    /// `fo:index-page-citation-range-separator`
    IndexPageCitationRangeSeparator = 28,
    /// `fo:index-page-number-prefix`
    IndexPageNumberPrefix = 29,
    /// `fo:index-page-number-suffix`
    IndexPageNumberSuffix = 30,
    /// `fo:index-range-begin`
    IndexRangeBegin = 31,
    /// `fo:index-range-end`
    IndexRangeEnd = 32,
    /// `fo:initial-property-set`
    InitialPropertySet = 33,
    /// `fo:inline`
    Inline = 34,
    /// `fo:inline-container`
    InlineContainer = 35,
    /// `fo:instream-foreign-object`
    InstreamForeignObject = 36,
    /// `fo:layout-master-set`
    LayoutMasterSet = 37,
    /// `fo:leader`
    Leader = 38,
    /// `fo:list-block`
    ListBlock = 39,
    /// `fo:list-item`
    ListItem = 40,
    /// `fo:list-item-body`
    ListItemBody = 41,
    /// `fo:list-item-label`
    ListItemLabel = 42,
    /// `fo:marker`
    Marker = 43,
    /// `fo:multi-case`
    MultiCase = 44,
    /// `fo:multi-properties`
    MultiProperties = 45,
    /// `fo:multi-property-set`
    MultiPropertySet = 46,
    /// `fo:multi-switch`
    MultiSwitch = 47,
    /// `fo:multi-toggle`
    MultiToggle = 48,
    /// `fo:page-number`
    PageNumber = 49,
    /// `fo:page-number-citation`
    PageNumberCitation = 50,
    /// `fo:page-number-citation-last`
    PageNumberCitationLast = 51,
    /// `fo:page-sequence`
    PageSequence = 52,
    /// `fo:page-sequence-master`
    PageSequenceMaster = 53,
    /// `fo:page-sequence-wrapper`
    PageSequenceWrapper = 54,
    /// `fo:region-after`
    RegionAfter = 55,
    /// `fo:region-before`
    RegionBefore = 56,
    /// `fo:region-body`
    RegionBody = 57,
    /// `fo:region-end`
    RegionEnd = 58,
    /// `fo:region-name-specifier`
    RegionNameSpecifier = 59,
    /// `fo:region-start`
    RegionStart = 60,
    /// `fo:repeatable-page-master-alternatives`
    RepeatablePageMasterAlternatives = 61,
    /// `fo:repeatable-page-master-reference`
    RepeatablePageMasterReference = 62,
    /// `fo:retrieve-marker`
    RetrieveMarker = 63,
    /// `fo:retrieve-table-marker`
    RetrieveTableMarker = 64,
    /// `fo:root`
    Root = 65,
    /// `fo:scaling-value-citation`
    ScalingValueCitation = 66,
    /// `fo:simple-page-master`
    SimplePageMaster = 67,
    /// `fo:single-page-master-reference`
    SinglePageMasterReference = 68,
    /// `fo:static-content`
    StaticContent = 69,
    /// `fo:table`
    Table = 70,
    /// `fo:table-and-caption`
    TableAndCaption = 71,
    /// `fo:table-body`
    TableBody = 72,
    /// `fo:table-caption`
    TableCaption = 73,
    /// `fo:table-cell`
    TableCell = 74,
    /// `fo:table-column`
    TableColumn = 75,
    /// `fo:table-footer`
    TableFooter = 76,
    /// `fo:table-header`
    TableHeader = 77,
    /// `fo:table-row`
    TableRow = 78,
    /// `fo:title`
    Title = 79,
    /// `fo:wrapper`
    Wrapper = 80,
}

/// Number of formatting objects defined by XSL 1.1.
pub const FO_COUNT: usize = 81;

pub(crate) static FO_NAMES: [&str; FO_COUNT] = [
    "basic-link",
    "bidi-override",
    "block",
    "block-container",
    "bookmark",
    "bookmark-title",
    "bookmark-tree",
    "change-bar-begin",
    "change-bar-end",
    "character",
    "color-profile",
    "conditional-page-master-reference",
    "declarations",
    "external-graphic",
    "float",
    "flow",
    "flow-assignment",
    "flow-map",
    "flow-name-specifier",
    "flow-source-list",
    "flow-target-list",
    "folio-prefix",
    "folio-suffix",
    "footnote",
    "footnote-body",
    "index-key-reference",
    "index-page-citation-list",
    "index-page-citation-list-separator",
    "index-page-citation-range-separator",
    "index-page-number-prefix",
    "index-page-number-suffix",
    "index-range-begin",
    "index-range-end",
    "initial-property-set",
    "inline",
    "inline-container",
    "instream-foreign-object",
    "layout-master-set",
    "leader",
    "list-block",
    "list-item",
    "list-item-body",
    "list-item-label",
    "marker",
    "multi-case",
    "multi-properties",
    "multi-property-set",
    "multi-switch",
    "multi-toggle",
    "page-number",
    "page-number-citation",
    "page-number-citation-last",
    "page-sequence",
    "page-sequence-master",
    "page-sequence-wrapper",
    "region-after",
    "region-before",
    "region-body",
    "region-end",
    "region-name-specifier",
    "region-start",
    "repeatable-page-master-alternatives",
    "repeatable-page-master-reference",
    "retrieve-marker",
    "retrieve-table-marker",
    "root",
    "scaling-value-citation",
    "simple-page-master",
    "single-page-master-reference",
    "static-content",
    "table",
    "table-and-caption",
    "table-body",
    "table-caption",
    "table-cell",
    "table-column",
    "table-footer",
    "table-header",
    "table-row",
    "title",
    "wrapper",
];

pub(crate) static FO_ELEMENTS: [FoElement; FO_COUNT] = [
    FoElement::BasicLink,
    FoElement::BidiOverride,
    FoElement::Block,
    FoElement::BlockContainer,
    FoElement::Bookmark,
    FoElement::BookmarkTitle,
    FoElement::BookmarkTree,
    FoElement::ChangeBarBegin,
    FoElement::ChangeBarEnd,
    FoElement::Character,
    FoElement::ColorProfile,
    FoElement::ConditionalPageMasterReference,
    FoElement::Declarations,
    FoElement::ExternalGraphic,
    FoElement::Float,
    FoElement::Flow,
    FoElement::FlowAssignment,
    FoElement::FlowMap,
    FoElement::FlowNameSpecifier,
    FoElement::FlowSourceList,
    FoElement::FlowTargetList,
    FoElement::FolioPrefix,
    FoElement::FolioSuffix,
    FoElement::Footnote,
    FoElement::FootnoteBody,
    FoElement::IndexKeyReference,
    FoElement::IndexPageCitationList,
    FoElement::IndexPageCitationListSeparator,
    FoElement::IndexPageCitationRangeSeparator,
    FoElement::IndexPageNumberPrefix,
    FoElement::IndexPageNumberSuffix,
    FoElement::IndexRangeBegin,
    FoElement::IndexRangeEnd,
    FoElement::InitialPropertySet,
    FoElement::Inline,
    FoElement::InlineContainer,
    FoElement::InstreamForeignObject,
    FoElement::LayoutMasterSet,
    FoElement::Leader,
    FoElement::ListBlock,
    FoElement::ListItem,
    FoElement::ListItemBody,
    FoElement::ListItemLabel,
    FoElement::Marker,
    FoElement::MultiCase,
    FoElement::MultiProperties,
    FoElement::MultiPropertySet,
    FoElement::MultiSwitch,
    FoElement::MultiToggle,
    FoElement::PageNumber,
    FoElement::PageNumberCitation,
    FoElement::PageNumberCitationLast,
    FoElement::PageSequence,
    FoElement::PageSequenceMaster,
    FoElement::PageSequenceWrapper,
    FoElement::RegionAfter,
    FoElement::RegionBefore,
    FoElement::RegionBody,
    FoElement::RegionEnd,
    FoElement::RegionNameSpecifier,
    FoElement::RegionStart,
    FoElement::RepeatablePageMasterAlternatives,
    FoElement::RepeatablePageMasterReference,
    FoElement::RetrieveMarker,
    FoElement::RetrieveTableMarker,
    FoElement::Root,
    FoElement::ScalingValueCitation,
    FoElement::SimplePageMaster,
    FoElement::SinglePageMasterReference,
    FoElement::StaticContent,
    FoElement::Table,
    FoElement::TableAndCaption,
    FoElement::TableBody,
    FoElement::TableCaption,
    FoElement::TableCell,
    FoElement::TableColumn,
    FoElement::TableFooter,
    FoElement::TableHeader,
    FoElement::TableRow,
    FoElement::Title,
    FoElement::Wrapper,
];
