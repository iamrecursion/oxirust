//! LUT provenance tracking and extended metadata parsing.
//!
//! Complements `lut_metadata` by providing field-level metadata parsing,
//! authoring history, and provenance chain tracking for LUT files.

#![allow(dead_code)]

/// A single key-value metadata field parsed from a LUT file.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MetadataField {
    /// The field name / key (e.g. `"TITLE"`, `"DOMAIN_MIN"`).
    pub key: String,
    /// The raw string value.
    pub value: String,
    /// The line number in the source file (1-based, 0 means unknown).
    pub line: usize,
}

impl MetadataField {
    /// Create a new metadata field.
    #[must_use]
    pub fn new(key: impl Into<String>, value: impl Into<String>, line: usize) -> Self {
        Self {
            key: key.into(),
            value: value.into(),
            line,
        }
    }

    /// Try to parse the value as an `f64`.
    #[must_use]
    pub fn as_f64(&self) -> Option<f64> {
        self.value.trim().parse::<f64>().ok()
    }

    /// Try to parse the value as a `u64`.
    #[must_use]
    pub fn as_u64(&self) -> Option<u64> {
        self.value.trim().parse::<u64>().ok()
    }

    /// Return the value with surrounding double-quotes stripped, if present.
    #[must_use]
    pub fn unquoted_value(&self) -> &str {
        let v = self.value.trim();
        v.strip_prefix('"')
            .and_then(|s| s.strip_suffix('"'))
            .unwrap_or(v)
    }
}

/// Kind of metadata field parsed from a `.cube` header.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LutMetadataFieldKind {
    /// The TITLE keyword.
    Title,
    /// A comment line (prefixed with `#`).
    Comment,
    /// `DOMAIN_MIN` triplet.
    DomainMin,
    /// `DOMAIN_MAX` triplet.
    DomainMax,
    /// `LUT_1D_SIZE` or `LUT_3D_SIZE`.
    Size,
    /// Any unrecognised keyword.
    Unknown,
}

impl LutMetadataFieldKind {
    /// Human-readable label.
    #[must_use]
    pub fn label(self) -> &'static str {
        match self {
            Self::Title => "Title",
            Self::Comment => "Comment",
            Self::DomainMin => "DomainMin",
            Self::DomainMax => "DomainMax",
            Self::Size => "Size",
            Self::Unknown => "Unknown",
        }
    }

    /// Classify a keyword string.
    #[must_use]
    pub fn classify(keyword: &str) -> Self {
        match keyword.trim().to_uppercase().as_str() {
            "TITLE" => Self::Title,
            s if s.starts_with('#') => Self::Comment,
            "DOMAIN_MIN" => Self::DomainMin,
            "DOMAIN_MAX" => Self::DomainMax,
            "LUT_1D_SIZE" | "LUT_3D_SIZE" => Self::Size,
            _ => Self::Unknown,
        }
    }
}

/// A parsed record in a LUT provenance chain.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ProvenanceEntry {
    /// Author or tool that performed this step.
    pub author: String,
    /// Short description of what happened.
    pub action: String,
    /// ISO-8601 timestamp string (or empty if unknown).
    pub timestamp: String,
}

impl ProvenanceEntry {
    /// Create a new provenance entry.
    #[must_use]
    pub fn new(
        author: impl Into<String>,
        action: impl Into<String>,
        timestamp: impl Into<String>,
    ) -> Self {
        Self {
            author: author.into(),
            action: action.into(),
            timestamp: timestamp.into(),
        }
    }

    /// Returns `true` if all fields are populated.
    #[must_use]
    pub fn is_complete(&self) -> bool {
        !self.author.is_empty() && !self.action.is_empty() && !self.timestamp.is_empty()
    }
}

/// Parses metadata fields out of raw `.cube` header text.
#[derive(Debug, Default)]
pub struct MetadataParser {
    /// Parsed fields.
    fields: Vec<MetadataField>,
    /// Provenance entries extracted from comments.
    provenance: Vec<ProvenanceEntry>,
}

impl MetadataParser {
    /// Create an empty parser.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Parse the given text (typically the header of a `.cube` file).
    ///
    /// Lines that look like `KEY value` are stored as fields.
    /// Lines starting with `#` are stored as comment fields and, if they
    /// follow the pattern `# PROVENANCE author | action | timestamp`, they
    /// are also added to the provenance chain.
    pub fn parse(&mut self, text: &str) {
        self.fields.clear();
        self.provenance.clear();

        for (idx, line) in text.lines().enumerate() {
            let trimmed = line.trim();
            if trimmed.is_empty() {
                continue;
            }
            let line_no = idx + 1;

            if let Some(comment_body) = trimmed.strip_prefix('#') {
                let comment_body = comment_body.trim();
                self.fields
                    .push(MetadataField::new("#", comment_body, line_no));

                // Try to extract provenance.
                if let Some(rest) = comment_body.strip_prefix("PROVENANCE") {
                    let parts: Vec<&str> = rest.split('|').map(str::trim).collect();
                    if parts.len() >= 3 {
                        self.provenance
                            .push(ProvenanceEntry::new(parts[0], parts[1], parts[2]));
                    }
                }
                continue;
            }

            // Parse `KEY rest-of-line`.
            if let Some((key, value)) = trimmed.split_once(char::is_whitespace) {
                self.fields
                    .push(MetadataField::new(key, value.trim(), line_no));
            }
        }
    }

    /// Return all parsed fields.
    #[must_use]
    pub fn fields(&self) -> &[MetadataField] {
        &self.fields
    }

    /// Return all provenance entries extracted from comments.
    #[must_use]
    pub fn provenance(&self) -> &[ProvenanceEntry] {
        &self.provenance
    }

    /// Find the first field with the given key (case-sensitive).
    #[must_use]
    pub fn find_field(&self, key: &str) -> Option<&MetadataField> {
        self.fields.iter().find(|f| f.key == key)
    }

    /// Count of all parsed fields.
    #[must_use]
    pub fn field_count(&self) -> usize {
        self.fields.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // --- MetadataField ---

    #[test]
    fn test_metadata_field_as_f64() {
        let f = MetadataField::new("SIZE", "33", 1);
        assert!((f.as_f64().expect("should succeed in test") - 33.0).abs() < 1e-12);
    }

    #[test]
    fn test_metadata_field_as_u64() {
        let f = MetadataField::new("SIZE", " 65 ", 2);
        assert_eq!(f.as_u64(), Some(65));
    }

    #[test]
    fn test_metadata_field_as_f64_invalid() {
        let f = MetadataField::new("TITLE", "hello", 1);
        assert!(f.as_f64().is_none());
    }

    #[test]
    fn test_metadata_field_unquoted() {
        let f = MetadataField::new("TITLE", "\"My LUT\"", 1);
        assert_eq!(f.unquoted_value(), "My LUT");
    }

    #[test]
    fn test_metadata_field_unquoted_no_quotes() {
        let f = MetadataField::new("KEY", "plain", 1);
        assert_eq!(f.unquoted_value(), "plain");
    }

    // --- LutMetadataFieldKind ---

    #[test]
    fn test_classify_title() {
        assert_eq!(
            LutMetadataFieldKind::classify("TITLE"),
            LutMetadataFieldKind::Title
        );
    }

    #[test]
    fn test_classify_domain() {
        assert_eq!(
            LutMetadataFieldKind::classify("DOMAIN_MIN"),
            LutMetadataFieldKind::DomainMin
        );
        assert_eq!(
            LutMetadataFieldKind::classify("DOMAIN_MAX"),
            LutMetadataFieldKind::DomainMax
        );
    }

    #[test]
    fn test_classify_size() {
        assert_eq!(
            LutMetadataFieldKind::classify("LUT_3D_SIZE"),
            LutMetadataFieldKind::Size
        );
        assert_eq!(
            LutMetadataFieldKind::classify("LUT_1D_SIZE"),
            LutMetadataFieldKind::Size
        );
    }

    #[test]
    fn test_classify_unknown() {
        assert_eq!(
            LutMetadataFieldKind::classify("FOOBAR"),
            LutMetadataFieldKind::Unknown
        );
    }

    #[test]
    fn test_kind_labels() {
        assert_eq!(LutMetadataFieldKind::Title.label(), "Title");
        assert_eq!(LutMetadataFieldKind::Comment.label(), "Comment");
        assert_eq!(LutMetadataFieldKind::Unknown.label(), "Unknown");
    }

    // --- ProvenanceEntry ---

    #[test]
    fn test_provenance_complete() {
        let e = ProvenanceEntry::new("Alice", "Created LUT", "2025-01-01T00:00:00Z");
        assert!(e.is_complete());
    }

    #[test]
    fn test_provenance_incomplete() {
        let e = ProvenanceEntry::new("", "Created LUT", "2025-01-01");
        assert!(!e.is_complete());
    }

    // --- MetadataParser ---

    #[test]
    fn test_parser_basic() {
        let header = "TITLE \"FilmLook\"\nLUT_3D_SIZE 33\n";
        let mut parser = MetadataParser::new();
        parser.parse(header);
        assert_eq!(parser.field_count(), 2);
        let title = parser.find_field("TITLE").expect("should succeed in test");
        assert_eq!(title.unquoted_value(), "FilmLook");
    }

    #[test]
    fn test_parser_comments() {
        let header = "# This is a comment\n# Another line\nTITLE test\n";
        let mut parser = MetadataParser::new();
        parser.parse(header);
        // 2 comment fields + 1 TITLE field.
        assert_eq!(parser.field_count(), 3);
    }

    #[test]
    fn test_parser_provenance() {
        let header = "# PROVENANCE Alice | Created LUT | 2025-01-01T00:00:00Z\nTITLE test\n";
        let mut parser = MetadataParser::new();
        parser.parse(header);
        assert_eq!(parser.provenance().len(), 1);
        assert_eq!(parser.provenance()[0].author, "Alice");
    }

    #[test]
    fn test_parser_empty_input() {
        let mut parser = MetadataParser::new();
        parser.parse("");
        assert_eq!(parser.field_count(), 0);
    }

    #[test]
    fn test_parser_skips_blank_lines() {
        let header = "\n\n  \nTITLE foo\n\n";
        let mut parser = MetadataParser::new();
        parser.parse(header);
        assert_eq!(parser.field_count(), 1);
    }
}
