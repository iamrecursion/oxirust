//! LUT metadata and tagging utilities.

#![allow(dead_code)]

/// Metadata describing a LUT's provenance and color-space context.
#[derive(Debug, Clone, Default)]
pub struct LutMetadata {
    /// Short human-readable LUT title.
    pub title: String,
    /// Extended description of what the LUT does.
    pub description: String,
    /// Name of the person or organization that created the LUT.
    pub creator: String,
    /// Unix timestamp (milliseconds) when the LUT was created.
    pub created_ms: u64,
    /// Color space of the input signal (e.g. "`ACEScg`", "Rec.709").
    pub input_space: String,
    /// Color space of the output signal (e.g. "Rec.709", "DCI-P3").
    pub output_space: String,
}

impl LutMetadata {
    /// Create a new, empty `LutMetadata`.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Returns `true` when all metadata fields are non-empty.
    #[must_use]
    pub fn is_complete(&self) -> bool {
        !self.title.is_empty()
            && !self.description.is_empty()
            && !self.creator.is_empty()
            && self.created_ms > 0
            && !self.input_space.is_empty()
            && !self.output_space.is_empty()
    }
}

/// Classification tag applied to a LUT.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum LutTag {
    /// A creative or artistic look.
    Creative,
    /// A technical transform (e.g. color-space conversion).
    Technical,
    /// A display-calibration LUT.
    Calibration,
    /// A quick preview or proxy LUT.
    Preview,
}

impl LutTag {
    /// Returns a human-readable label for the tag.
    #[must_use]
    pub fn label(&self) -> &str {
        match self {
            Self::Creative => "Creative",
            Self::Technical => "Technical",
            Self::Calibration => "Calibration",
            Self::Preview => "Preview",
        }
    }
}

/// A LUT bundled with metadata and classification tags.
#[derive(Debug, Clone)]
pub struct TaggedLut {
    /// Descriptive metadata for this LUT.
    pub metadata: LutMetadata,
    /// Zero or more classification tags.
    pub tags: Vec<LutTag>,
    /// Semver-style version string (e.g. "1.0.0").
    pub version: String,
}

impl TaggedLut {
    /// Create a new `TaggedLut` with the given metadata and version.
    #[must_use]
    pub fn new(metadata: LutMetadata, version: impl Into<String>) -> Self {
        Self {
            metadata,
            tags: Vec::new(),
            version: version.into(),
        }
    }

    /// Returns `true` if the given tag is present.
    #[must_use]
    pub fn has_tag(&self, tag: &LutTag) -> bool {
        self.tags.contains(tag)
    }

    /// Add a tag (duplicates are allowed — caller is responsible for dedup).
    pub fn add_tag(&mut self, tag: LutTag) {
        self.tags.push(tag);
    }

    /// Convenience: returns `true` if the LUT carries a [`LutTag::Creative`] tag.
    #[must_use]
    pub fn is_creative(&self) -> bool {
        self.has_tag(&LutTag::Creative)
    }
}

/// An in-memory collection of tagged LUTs.
#[derive(Debug, Clone, Default)]
pub struct LutCollection {
    /// All LUTs in the collection.
    pub luts: Vec<TaggedLut>,
}

impl LutCollection {
    /// Create an empty `LutCollection`.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Add a LUT to the collection.
    pub fn add(&mut self, lut: TaggedLut) {
        self.luts.push(lut);
    }

    /// Return all LUTs that carry the given tag.
    #[must_use]
    pub fn filter_by_tag(&self, tag: &LutTag) -> Vec<&TaggedLut> {
        self.luts.iter().filter(|l| l.has_tag(tag)).collect()
    }

    /// Find a LUT by exact title match (returns the first match).
    #[must_use]
    pub fn find_by_title(&self, title: &str) -> Option<&TaggedLut> {
        self.luts.iter().find(|l| l.metadata.title == title)
    }

    /// Total number of LUTs in the collection.
    #[must_use]
    pub fn count(&self) -> usize {
        self.luts.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn complete_metadata() -> LutMetadata {
        LutMetadata {
            title: "FilmLook".to_string(),
            description: "A cinematic look".to_string(),
            creator: "OxiMedia".to_string(),
            created_ms: 1_700_000_000_000,
            input_space: "ACEScg".to_string(),
            output_space: "Rec.709".to_string(),
        }
    }

    // --- LutMetadata ---

    #[test]
    fn test_metadata_is_complete_when_all_fields_set() {
        let m = complete_metadata();
        assert!(m.is_complete());
    }

    #[test]
    fn test_metadata_incomplete_missing_title() {
        let mut m = complete_metadata();
        m.title = String::new();
        assert!(!m.is_complete());
    }

    #[test]
    fn test_metadata_incomplete_missing_creator() {
        let mut m = complete_metadata();
        m.creator = String::new();
        assert!(!m.is_complete());
    }

    #[test]
    fn test_metadata_incomplete_zero_timestamp() {
        let mut m = complete_metadata();
        m.created_ms = 0;
        assert!(!m.is_complete());
    }

    #[test]
    fn test_metadata_incomplete_missing_input_space() {
        let mut m = complete_metadata();
        m.input_space = String::new();
        assert!(!m.is_complete());
    }

    #[test]
    fn test_metadata_incomplete_missing_output_space() {
        let mut m = complete_metadata();
        m.output_space = String::new();
        assert!(!m.is_complete());
    }

    // --- LutTag ---

    #[test]
    fn test_tag_labels() {
        assert_eq!(LutTag::Creative.label(), "Creative");
        assert_eq!(LutTag::Technical.label(), "Technical");
        assert_eq!(LutTag::Calibration.label(), "Calibration");
        assert_eq!(LutTag::Preview.label(), "Preview");
    }

    // --- TaggedLut ---

    #[test]
    fn test_has_tag_false_when_empty() {
        let lut = TaggedLut::new(complete_metadata(), "1.0.0");
        assert!(!lut.has_tag(&LutTag::Creative));
    }

    #[test]
    fn test_add_tag_and_has_tag() {
        let mut lut = TaggedLut::new(complete_metadata(), "1.0.0");
        lut.add_tag(LutTag::Creative);
        assert!(lut.has_tag(&LutTag::Creative));
        assert!(!lut.has_tag(&LutTag::Technical));
    }

    #[test]
    fn test_is_creative_true() {
        let mut lut = TaggedLut::new(complete_metadata(), "1.0.0");
        lut.add_tag(LutTag::Creative);
        assert!(lut.is_creative());
    }

    #[test]
    fn test_is_creative_false_without_tag() {
        let lut = TaggedLut::new(complete_metadata(), "1.0.0");
        assert!(!lut.is_creative());
    }

    // --- LutCollection ---

    #[test]
    fn test_collection_starts_empty() {
        let col = LutCollection::new();
        assert_eq!(col.count(), 0);
    }

    #[test]
    fn test_collection_add_increases_count() {
        let mut col = LutCollection::new();
        col.add(TaggedLut::new(complete_metadata(), "1.0"));
        col.add(TaggedLut::new(complete_metadata(), "2.0"));
        assert_eq!(col.count(), 2);
    }

    #[test]
    fn test_filter_by_tag_returns_matching() {
        let mut col = LutCollection::new();
        let mut lut_a = TaggedLut::new(complete_metadata(), "1.0");
        lut_a.add_tag(LutTag::Creative);
        let lut_b = TaggedLut::new(complete_metadata(), "2.0"); // no tag
        col.add(lut_a);
        col.add(lut_b);
        let results = col.filter_by_tag(&LutTag::Creative);
        assert_eq!(results.len(), 1);
    }

    #[test]
    fn test_filter_by_tag_returns_empty_when_none_match() {
        let mut col = LutCollection::new();
        col.add(TaggedLut::new(complete_metadata(), "1.0"));
        let results = col.filter_by_tag(&LutTag::Calibration);
        assert!(results.is_empty());
    }

    #[test]
    fn test_find_by_title_found() {
        let mut col = LutCollection::new();
        let mut m = complete_metadata();
        m.title = "SpecialLUT".to_string();
        col.add(TaggedLut::new(m, "1.0"));
        col.add(TaggedLut::new(complete_metadata(), "1.0"));
        let found = col.find_by_title("SpecialLUT");
        assert!(found.is_some());
    }

    #[test]
    fn test_find_by_title_not_found() {
        let col = LutCollection::new();
        assert!(col.find_by_title("NoSuchLUT").is_none());
    }
}
