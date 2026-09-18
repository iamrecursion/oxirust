#![allow(dead_code)]
//! Search result export in multiple formats (CSV, JSON, XML).
//!
//! Provides configurable field selection and formatting for exporting
//! search result sets to common interchange formats suitable for
//! downstream processing, reporting, and integration.
//!
//! # Supported formats
//!
//! - **CSV**: RFC 4180 compliant, with configurable delimiter and quoting
//! - **JSON**: Pretty-printed or compact, with optional metadata envelope
//! - **XML**: Well-formed XML with configurable root and item element names

use std::fmt::Write as FmtWrite;
use std::io::Write;

use crate::error::{SearchError, SearchResult};
use crate::SearchResultItem;

// ---------------------------------------------------------------------------
// Export field selection
// ---------------------------------------------------------------------------

/// A field that can be included in the export output.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ExportField {
    /// Asset UUID.
    AssetId,
    /// Relevance score.
    Score,
    /// Title.
    Title,
    /// Description.
    Description,
    /// File path.
    FilePath,
    /// MIME type.
    MimeType,
    /// Duration in milliseconds.
    DurationMs,
    /// Created-at timestamp (Unix seconds).
    CreatedAt,
    /// Modified-at timestamp (Unix seconds).
    ModifiedAt,
    /// File size in bytes.
    FileSize,
    /// Matched fields (comma-separated).
    MatchedFields,
    /// Thumbnail URL.
    ThumbnailUrl,
}

impl ExportField {
    /// Header label for this field.
    #[must_use]
    pub fn label(&self) -> &'static str {
        match self {
            Self::AssetId => "asset_id",
            Self::Score => "score",
            Self::Title => "title",
            Self::Description => "description",
            Self::FilePath => "file_path",
            Self::MimeType => "mime_type",
            Self::DurationMs => "duration_ms",
            Self::CreatedAt => "created_at",
            Self::ModifiedAt => "modified_at",
            Self::FileSize => "file_size",
            Self::MatchedFields => "matched_fields",
            Self::ThumbnailUrl => "thumbnail_url",
        }
    }

    /// Extract the string value for this field from a result item.
    #[must_use]
    pub fn extract(&self, item: &SearchResultItem) -> String {
        match self {
            Self::AssetId => item.asset_id.to_string(),
            Self::Score => format!("{:.4}", item.score),
            Self::Title => item.title.clone().unwrap_or_default(),
            Self::Description => item.description.clone().unwrap_or_default(),
            Self::FilePath => item.file_path.clone(),
            Self::MimeType => item.mime_type.clone().unwrap_or_default(),
            Self::DurationMs => item.duration_ms.map_or_else(String::new, |d| d.to_string()),
            Self::CreatedAt => item.created_at.to_string(),
            Self::ModifiedAt => item.modified_at.map_or_else(String::new, |m| m.to_string()),
            Self::FileSize => item.file_size.map_or_else(String::new, |s| s.to_string()),
            Self::MatchedFields => item.matched_fields.join(","),
            Self::ThumbnailUrl => item.thumbnail_url.clone().unwrap_or_default(),
        }
    }

    /// All available export fields.
    #[must_use]
    pub fn all() -> Vec<ExportField> {
        vec![
            Self::AssetId,
            Self::Score,
            Self::Title,
            Self::Description,
            Self::FilePath,
            Self::MimeType,
            Self::DurationMs,
            Self::CreatedAt,
            Self::ModifiedAt,
            Self::FileSize,
            Self::MatchedFields,
            Self::ThumbnailUrl,
        ]
    }

    /// Default commonly-used fields.
    #[must_use]
    pub fn defaults() -> Vec<ExportField> {
        vec![
            Self::AssetId,
            Self::Score,
            Self::Title,
            Self::FilePath,
            Self::MimeType,
            Self::DurationMs,
            Self::CreatedAt,
        ]
    }
}

// ---------------------------------------------------------------------------
// Export format
// ---------------------------------------------------------------------------

/// Output format for search result export.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ExportFormat {
    /// Comma-separated values (RFC 4180).
    Csv,
    /// JSON array.
    Json,
    /// JSON with metadata envelope (total, query time, etc.).
    JsonEnvelope,
    /// XML document.
    Xml,
}

// ---------------------------------------------------------------------------
// Export configuration
// ---------------------------------------------------------------------------

/// Configuration for search result export.
#[derive(Debug, Clone)]
pub struct ExportConfig {
    /// Output format.
    pub format: ExportFormat,
    /// Fields to include in the export.
    pub fields: Vec<ExportField>,
    /// CSV delimiter character (default: comma).
    pub csv_delimiter: char,
    /// Whether to include headers (CSV) or metadata (JSON/XML).
    pub include_header: bool,
    /// Whether to pretty-print (JSON/XML).
    pub pretty: bool,
    /// XML root element name.
    pub xml_root: String,
    /// XML item element name.
    pub xml_item: String,
}

impl Default for ExportConfig {
    fn default() -> Self {
        Self {
            format: ExportFormat::Csv,
            fields: ExportField::defaults(),
            csv_delimiter: ',',
            include_header: true,
            pretty: true,
            xml_root: "search_results".to_string(),
            xml_item: "result".to_string(),
        }
    }
}

// ---------------------------------------------------------------------------
// Search result exporter
// ---------------------------------------------------------------------------

/// Exports search results to various formats.
#[derive(Debug)]
pub struct SearchExporter {
    config: ExportConfig,
}

impl SearchExporter {
    /// Create a new exporter with the given configuration.
    #[must_use]
    pub fn new(config: ExportConfig) -> Self {
        Self { config }
    }

    /// Create a CSV exporter with default fields.
    #[must_use]
    pub fn csv() -> Self {
        Self::new(ExportConfig {
            format: ExportFormat::Csv,
            ..Default::default()
        })
    }

    /// Create a JSON exporter.
    #[must_use]
    pub fn json() -> Self {
        Self::new(ExportConfig {
            format: ExportFormat::Json,
            ..Default::default()
        })
    }

    /// Create an XML exporter.
    #[must_use]
    pub fn xml() -> Self {
        Self::new(ExportConfig {
            format: ExportFormat::Xml,
            ..Default::default()
        })
    }

    /// Export search results to a string.
    ///
    /// # Errors
    ///
    /// Returns an error if formatting fails.
    pub fn export_to_string(&self, items: &[SearchResultItem]) -> SearchResult<String> {
        match self.config.format {
            ExportFormat::Csv => self.export_csv(items),
            ExportFormat::Json | ExportFormat::JsonEnvelope => self.export_json(items),
            ExportFormat::Xml => self.export_xml(items),
        }
    }

    /// Export search results to a writer (file, buffer, etc.).
    ///
    /// # Errors
    ///
    /// Returns an error if writing fails.
    pub fn export_to_writer<W: Write>(
        &self,
        items: &[SearchResultItem],
        writer: &mut W,
    ) -> SearchResult<()> {
        let output = self.export_to_string(items)?;
        writer
            .write_all(output.as_bytes())
            .map_err(SearchError::Io)?;
        Ok(())
    }

    /// Generate CSV output.
    fn export_csv(&self, items: &[SearchResultItem]) -> SearchResult<String> {
        let delim = self.config.csv_delimiter;
        let mut out = String::new();

        // Header row
        if self.config.include_header {
            let headers: Vec<&str> = self.config.fields.iter().map(ExportField::label).collect();
            out.push_str(&headers.join(&delim.to_string()));
            out.push('\n');
        }

        // Data rows
        for item in items {
            let values: Vec<String> = self
                .config
                .fields
                .iter()
                .map(|f| csv_escape(&f.extract(item), delim))
                .collect();
            out.push_str(&values.join(&delim.to_string()));
            out.push('\n');
        }

        Ok(out)
    }

    /// Generate JSON output.
    fn export_json(&self, items: &[SearchResultItem]) -> SearchResult<String> {
        let mut out = String::new();

        let is_envelope = self.config.format == ExportFormat::JsonEnvelope;
        let indent = if self.config.pretty { "  " } else { "" };
        let nl = if self.config.pretty { "\n" } else { "" };

        if is_envelope {
            write!(out, "{{{nl}").map_err(|e| SearchError::Other(e.to_string()))?;
            write!(out, "{indent}\"total\": {},{nl}", items.len())
                .map_err(|e| SearchError::Other(e.to_string()))?;
            write!(out, "{indent}\"results\": ").map_err(|e| SearchError::Other(e.to_string()))?;
        }

        write!(out, "[{nl}").map_err(|e| SearchError::Other(e.to_string()))?;
        for (i, item) in items.iter().enumerate() {
            let item_indent = if is_envelope && self.config.pretty {
                "    "
            } else {
                indent
            };
            let field_indent = if self.config.pretty {
                format!("{item_indent}  ")
            } else {
                String::new()
            };

            write!(out, "{item_indent}{{{nl}").map_err(|e| SearchError::Other(e.to_string()))?;
            for (j, field) in self.config.fields.iter().enumerate() {
                let value = field.extract(item);
                let json_value = json_encode_value(field, &value);
                write!(out, "{field_indent}\"{}\": {json_value}", field.label())
                    .map_err(|e| SearchError::Other(e.to_string()))?;
                if j + 1 < self.config.fields.len() {
                    write!(out, ",").map_err(|e| SearchError::Other(e.to_string()))?;
                }
                write!(out, "{nl}").map_err(|e| SearchError::Other(e.to_string()))?;
            }
            write!(out, "{item_indent}}}").map_err(|e| SearchError::Other(e.to_string()))?;
            if i + 1 < items.len() {
                write!(out, ",").map_err(|e| SearchError::Other(e.to_string()))?;
            }
            write!(out, "{nl}").map_err(|e| SearchError::Other(e.to_string()))?;
        }

        if is_envelope && self.config.pretty {
            write!(out, "{indent}]{nl}").map_err(|e| SearchError::Other(e.to_string()))?;
            write!(out, "}}{nl}").map_err(|e| SearchError::Other(e.to_string()))?;
        } else {
            write!(out, "]{nl}").map_err(|e| SearchError::Other(e.to_string()))?;
        }

        Ok(out)
    }

    /// Generate XML output.
    fn export_xml(&self, items: &[SearchResultItem]) -> SearchResult<String> {
        let mut out = String::new();
        let root = &self.config.xml_root;
        let item_elem = &self.config.xml_item;
        let indent = if self.config.pretty { "  " } else { "" };
        let nl = if self.config.pretty { "\n" } else { "" };

        write!(out, "<?xml version=\"1.0\" encoding=\"UTF-8\"?>{nl}")
            .map_err(|e| SearchError::Other(e.to_string()))?;
        write!(out, "<{root} total=\"{}\">{nl}", items.len())
            .map_err(|e| SearchError::Other(e.to_string()))?;

        for item in items {
            write!(out, "{indent}<{item_elem}>{nl}")
                .map_err(|e| SearchError::Other(e.to_string()))?;
            for field in &self.config.fields {
                let value = field.extract(item);
                let escaped = xml_escape(&value);
                write!(
                    out,
                    "{indent}{indent}<{label}>{escaped}</{label}>{nl}",
                    label = field.label()
                )
                .map_err(|e| SearchError::Other(e.to_string()))?;
            }
            write!(out, "{indent}</{item_elem}>{nl}")
                .map_err(|e| SearchError::Other(e.to_string()))?;
        }

        write!(out, "</{root}>{nl}").map_err(|e| SearchError::Other(e.to_string()))?;
        Ok(out)
    }
}

// ---------------------------------------------------------------------------
// Encoding helpers
// ---------------------------------------------------------------------------

/// Escape a CSV field value: quote if it contains the delimiter, quotes, or newlines.
fn csv_escape(value: &str, delimiter: char) -> String {
    if value.contains(delimiter)
        || value.contains('"')
        || value.contains('\n')
        || value.contains('\r')
    {
        format!("\"{}\"", value.replace('"', "\"\""))
    } else {
        value.to_string()
    }
}

/// Encode a field value as a JSON value string.
fn json_encode_value(field: &ExportField, value: &str) -> String {
    match field {
        ExportField::Score => {
            // Numeric — output without quotes
            if value.is_empty() {
                "null".to_string()
            } else {
                value.to_string()
            }
        }
        ExportField::DurationMs | ExportField::CreatedAt | ExportField::FileSize => {
            if value.is_empty() {
                "null".to_string()
            } else {
                value.to_string()
            }
        }
        ExportField::ModifiedAt => {
            if value.is_empty() {
                "null".to_string()
            } else {
                value.to_string()
            }
        }
        _ => {
            // String value — JSON-escape and quote
            format!("\"{}\"", json_escape(value))
        }
    }
}

/// Escape special characters for JSON string content.
fn json_escape(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    for ch in s.chars() {
        match ch {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            c if c.is_control() => {
                write!(out, "\\u{:04x}", c as u32).unwrap_or_default();
            }
            c => out.push(c),
        }
    }
    out
}

/// Escape special characters for XML content.
fn xml_escape(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    for ch in s.chars() {
        match ch {
            '&' => out.push_str("&amp;"),
            '<' => out.push_str("&lt;"),
            '>' => out.push_str("&gt;"),
            '"' => out.push_str("&quot;"),
            '\'' => out.push_str("&apos;"),
            c => out.push(c),
        }
    }
    out
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use uuid::Uuid;

    fn make_item(title: &str, score: f32) -> SearchResultItem {
        SearchResultItem {
            asset_id: Uuid::nil(),
            score,
            title: Some(title.to_string()),
            description: Some("A test video".to_string()),
            file_path: "/media/test.mp4".to_string(),
            mime_type: Some("video/mp4".to_string()),
            duration_ms: Some(120_000),
            created_at: 1_700_000_000,
            modified_at: Some(1_700_001_000),
            file_size: Some(5_000_000),
            matched_fields: vec!["title".to_string()],
            thumbnail_url: None,
        }
    }

    fn sample_items() -> Vec<SearchResultItem> {
        vec![
            make_item("Sunset Video", 0.95),
            make_item("Ocean Waves", 0.87),
        ]
    }

    // -- CSV tests --

    #[test]
    fn test_csv_export_header() {
        let exporter = SearchExporter::csv();
        let output = exporter
            .export_to_string(&sample_items())
            .expect("should export");
        let lines: Vec<&str> = output.lines().collect();
        assert!(lines[0].contains("asset_id"));
        assert!(lines[0].contains("score"));
        assert!(lines[0].contains("title"));
    }

    #[test]
    fn test_csv_export_data_rows() {
        let exporter = SearchExporter::csv();
        let output = exporter
            .export_to_string(&sample_items())
            .expect("should export");
        let lines: Vec<&str> = output.lines().collect();
        // header + 2 data rows
        assert_eq!(lines.len(), 3);
        assert!(lines[1].contains("Sunset Video"));
        assert!(lines[2].contains("Ocean Waves"));
    }

    #[test]
    fn test_csv_export_no_header() {
        let config = ExportConfig {
            format: ExportFormat::Csv,
            include_header: false,
            ..Default::default()
        };
        let exporter = SearchExporter::new(config);
        let output = exporter
            .export_to_string(&sample_items())
            .expect("should export");
        let lines: Vec<&str> = output.lines().collect();
        assert_eq!(lines.len(), 2); // no header
    }

    #[test]
    fn test_csv_escape_comma() {
        let escaped = csv_escape("hello, world", ',');
        assert_eq!(escaped, "\"hello, world\"");
    }

    #[test]
    fn test_csv_escape_quotes() {
        let escaped = csv_escape("say \"hello\"", ',');
        assert_eq!(escaped, "\"say \"\"hello\"\"\"");
    }

    #[test]
    fn test_csv_escape_no_special() {
        let escaped = csv_escape("simple", ',');
        assert_eq!(escaped, "simple");
    }

    #[test]
    fn test_csv_custom_delimiter() {
        let config = ExportConfig {
            format: ExportFormat::Csv,
            csv_delimiter: '\t',
            ..Default::default()
        };
        let exporter = SearchExporter::new(config);
        let output = exporter
            .export_to_string(&sample_items())
            .expect("should export");
        assert!(output.contains('\t'));
    }

    // -- JSON tests --

    #[test]
    fn test_json_export_structure() {
        let exporter = SearchExporter::json();
        let output = exporter
            .export_to_string(&sample_items())
            .expect("should export");
        assert!(output.starts_with('['));
        assert!(output.trim_end().ends_with(']'));
        assert!(output.contains("\"title\""));
        assert!(output.contains("\"Sunset Video\""));
    }

    #[test]
    fn test_json_envelope_export() {
        let config = ExportConfig {
            format: ExportFormat::JsonEnvelope,
            ..Default::default()
        };
        let exporter = SearchExporter::new(config);
        let output = exporter
            .export_to_string(&sample_items())
            .expect("should export");
        assert!(output.contains("\"total\": 2"));
        assert!(output.contains("\"results\":"));
    }

    #[test]
    fn test_json_compact() {
        let config = ExportConfig {
            format: ExportFormat::Json,
            pretty: false,
            ..Default::default()
        };
        let exporter = SearchExporter::new(config);
        let output = exporter
            .export_to_string(&sample_items())
            .expect("should export");
        // Should not contain leading spaces
        assert!(!output.contains("  "));
    }

    #[test]
    fn test_json_escape_special_chars() {
        let escaped = json_escape("line1\nline2\ttab\"quote\\back");
        assert_eq!(escaped, "line1\\nline2\\ttab\\\"quote\\\\back");
    }

    #[test]
    fn test_json_numeric_fields() {
        let exporter = SearchExporter::json();
        let output = exporter
            .export_to_string(&sample_items())
            .expect("should export");
        // Score should not be quoted
        assert!(output.contains("\"score\": 0.9500"));
        // Duration should not be quoted
        assert!(output.contains("\"duration_ms\": 120000"));
    }

    // -- XML tests --

    #[test]
    fn test_xml_export_structure() {
        let exporter = SearchExporter::xml();
        let output = exporter
            .export_to_string(&sample_items())
            .expect("should export");
        assert!(output.contains("<?xml version=\"1.0\""));
        assert!(output.contains("<search_results total=\"2\">"));
        assert!(output.contains("<result>"));
        assert!(output.contains("</result>"));
        assert!(output.contains("</search_results>"));
    }

    #[test]
    fn test_xml_field_content() {
        let exporter = SearchExporter::xml();
        let output = exporter
            .export_to_string(&sample_items())
            .expect("should export");
        assert!(output.contains("<title>Sunset Video</title>"));
        assert!(output.contains("<file_path>/media/test.mp4</file_path>"));
    }

    #[test]
    fn test_xml_escape_special() {
        let escaped = xml_escape("<script>alert('xss')&</script>");
        assert_eq!(
            escaped,
            "&lt;script&gt;alert(&apos;xss&apos;)&amp;&lt;/script&gt;"
        );
    }

    #[test]
    fn test_xml_custom_elements() {
        let config = ExportConfig {
            format: ExportFormat::Xml,
            xml_root: "media_results".to_string(),
            xml_item: "media".to_string(),
            ..Default::default()
        };
        let exporter = SearchExporter::new(config);
        let output = exporter
            .export_to_string(&sample_items())
            .expect("should export");
        assert!(output.contains("<media_results"));
        assert!(output.contains("<media>"));
        assert!(output.contains("</media>"));
        assert!(output.contains("</media_results>"));
    }

    // -- Field extraction tests --

    #[test]
    fn test_export_field_label() {
        assert_eq!(ExportField::AssetId.label(), "asset_id");
        assert_eq!(ExportField::Score.label(), "score");
        assert_eq!(ExportField::Title.label(), "title");
        assert_eq!(ExportField::FileSize.label(), "file_size");
    }

    #[test]
    fn test_export_field_extract() {
        let item = make_item("Test", 0.5);
        assert_eq!(ExportField::Title.extract(&item), "Test");
        assert_eq!(ExportField::Score.extract(&item), "0.5000");
        assert_eq!(ExportField::FilePath.extract(&item), "/media/test.mp4");
        assert_eq!(ExportField::DurationMs.extract(&item), "120000");
        assert_eq!(ExportField::FileSize.extract(&item), "5000000");
    }

    #[test]
    fn test_export_field_extract_none_values() {
        let item = SearchResultItem {
            asset_id: Uuid::nil(),
            score: 0.0,
            title: None,
            description: None,
            file_path: String::new(),
            mime_type: None,
            duration_ms: None,
            created_at: 0,
            modified_at: None,
            file_size: None,
            matched_fields: Vec::new(),
            thumbnail_url: None,
        };
        assert_eq!(ExportField::Title.extract(&item), "");
        assert_eq!(ExportField::DurationMs.extract(&item), "");
        assert_eq!(ExportField::ModifiedAt.extract(&item), "");
        assert_eq!(ExportField::FileSize.extract(&item), "");
    }

    #[test]
    fn test_export_field_all() {
        let all = ExportField::all();
        assert_eq!(all.len(), 12);
    }

    #[test]
    fn test_export_field_defaults() {
        let defaults = ExportField::defaults();
        assert!(defaults.len() < ExportField::all().len());
        assert!(defaults.contains(&ExportField::AssetId));
        assert!(defaults.contains(&ExportField::Score));
    }

    // -- Writer export test --

    #[test]
    fn test_export_to_writer() {
        let exporter = SearchExporter::csv();
        let mut buf = Vec::new();
        exporter
            .export_to_writer(&sample_items(), &mut buf)
            .expect("should write");
        let output = String::from_utf8(buf).expect("should be valid utf8");
        assert!(output.contains("Sunset Video"));
    }

    // -- Empty input tests --

    #[test]
    fn test_csv_export_empty() {
        let exporter = SearchExporter::csv();
        let output = exporter.export_to_string(&[]).expect("should export");
        let lines: Vec<&str> = output.lines().collect();
        assert_eq!(lines.len(), 1); // header only
    }

    #[test]
    fn test_json_export_empty() {
        let exporter = SearchExporter::json();
        let output = exporter.export_to_string(&[]).expect("should export");
        assert!(output.contains("[]") || output.trim() == "[\n]");
    }

    #[test]
    fn test_xml_export_empty() {
        let exporter = SearchExporter::xml();
        let output = exporter.export_to_string(&[]).expect("should export");
        assert!(output.contains("total=\"0\""));
    }

    // -- Custom field selection --

    #[test]
    fn test_csv_custom_fields() {
        let config = ExportConfig {
            format: ExportFormat::Csv,
            fields: vec![ExportField::Title, ExportField::Score],
            ..Default::default()
        };
        let exporter = SearchExporter::new(config);
        let output = exporter
            .export_to_string(&sample_items())
            .expect("should export");
        let header = output.lines().next().unwrap_or("");
        assert_eq!(header, "title,score");
    }
}
