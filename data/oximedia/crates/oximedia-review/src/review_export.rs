#![allow(dead_code)]
//! Export review session data to various formats.
//!
//! This module provides functionality to serialize and export review data
//! including comments, annotations, status history, and metrics into
//! JSON, XML, and plain-text summary formats for archival and sharing.

use std::collections::HashMap;
use std::fmt;

/// Supported export formats.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ExportFormat {
    /// JSON format.
    Json,
    /// XML format.
    Xml,
    /// Plain text summary.
    PlainText,
    /// Comma-separated values.
    Csv,
    /// Tab-separated values.
    Tsv,
}

impl fmt::Display for ExportFormat {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Json => write!(f, "JSON"),
            Self::Xml => write!(f, "XML"),
            Self::PlainText => write!(f, "Plain Text"),
            Self::Csv => write!(f, "CSV"),
            Self::Tsv => write!(f, "TSV"),
        }
    }
}

impl ExportFormat {
    /// Get the standard file extension for this format.
    #[must_use]
    pub fn extension(&self) -> &str {
        match self {
            Self::Json => "json",
            Self::Xml => "xml",
            Self::PlainText => "txt",
            Self::Csv => "csv",
            Self::Tsv => "tsv",
        }
    }

    /// Get the MIME type for this format.
    #[must_use]
    pub fn mime_type(&self) -> &str {
        match self {
            Self::Json => "application/json",
            Self::Xml => "application/xml",
            Self::PlainText => "text/plain",
            Self::Csv => "text/csv",
            Self::Tsv => "text/tab-separated-values",
        }
    }
}

/// A single comment entry for export.
#[derive(Debug, Clone)]
pub struct ExportComment {
    /// Comment author name.
    pub author: String,
    /// Comment text content.
    pub text: String,
    /// Frame number the comment refers to, if any.
    pub frame: Option<u64>,
    /// Timestamp of the comment in milliseconds since epoch.
    pub timestamp_ms: u64,
    /// Whether the comment has been resolved.
    pub resolved: bool,
    /// Comment category or tag.
    pub category: String,
}

/// A single annotation entry for export.
#[derive(Debug, Clone)]
pub struct ExportAnnotation {
    /// Annotation author name.
    pub author: String,
    /// Annotation type description.
    pub annotation_type: String,
    /// Frame number.
    pub frame: u64,
    /// X coordinate (normalized 0.0-1.0).
    pub x: f64,
    /// Y coordinate (normalized 0.0-1.0).
    pub y: f64,
    /// Description or label.
    pub label: String,
}

/// Configuration for the export operation.
#[derive(Debug, Clone)]
pub struct ExportConfig {
    /// The output format.
    pub format: ExportFormat,
    /// Whether to include resolved comments.
    pub include_resolved: bool,
    /// Whether to include annotations.
    pub include_annotations: bool,
    /// Whether to include status history.
    pub include_history: bool,
    /// Optional title override for the export.
    pub title: Option<String>,
    /// Custom metadata fields to include.
    pub extra_fields: HashMap<String, String>,
}

impl Default for ExportConfig {
    fn default() -> Self {
        Self {
            format: ExportFormat::Json,
            include_resolved: true,
            include_annotations: true,
            include_history: true,
            title: None,
            extra_fields: HashMap::new(),
        }
    }
}

impl ExportConfig {
    /// Create a new export config with the given format.
    #[must_use]
    pub fn new(format: ExportFormat) -> Self {
        Self {
            format,
            ..Self::default()
        }
    }

    /// Set whether to include resolved comments.
    #[must_use]
    pub fn with_resolved(mut self, include: bool) -> Self {
        self.include_resolved = include;
        self
    }

    /// Set whether to include annotations.
    #[must_use]
    pub fn with_annotations(mut self, include: bool) -> Self {
        self.include_annotations = include;
        self
    }

    /// Set whether to include history.
    #[must_use]
    pub fn with_history(mut self, include: bool) -> Self {
        self.include_history = include;
        self
    }

    /// Set a custom title.
    #[must_use]
    pub fn with_title(mut self, title: impl Into<String>) -> Self {
        self.title = Some(title.into());
        self
    }

    /// Add a custom metadata field.
    #[must_use]
    pub fn with_field(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        self.extra_fields.insert(key.into(), value.into());
        self
    }
}

/// Data bundle to be exported.
#[derive(Debug, Clone)]
pub struct ExportData {
    /// Session title.
    pub title: String,
    /// Session status description.
    pub status: String,
    /// Comments in the session.
    pub comments: Vec<ExportComment>,
    /// Annotations in the session.
    pub annotations: Vec<ExportAnnotation>,
    /// Status history entries (from, to, timestamp descriptions).
    pub history: Vec<String>,
    /// Custom metadata.
    pub metadata: HashMap<String, String>,
}

impl ExportData {
    /// Create an empty export data bundle.
    #[must_use]
    pub fn new(title: &str, status: &str) -> Self {
        Self {
            title: title.to_string(),
            status: status.to_string(),
            comments: Vec::new(),
            annotations: Vec::new(),
            history: Vec::new(),
            metadata: HashMap::new(),
        }
    }

    /// Add a comment.
    pub fn add_comment(&mut self, comment: ExportComment) {
        self.comments.push(comment);
    }

    /// Add an annotation.
    pub fn add_annotation(&mut self, annotation: ExportAnnotation) {
        self.annotations.push(annotation);
    }

    /// Add a history entry.
    pub fn add_history_entry(&mut self, entry: impl Into<String>) {
        self.history.push(entry.into());
    }

    /// Count only unresolved comments.
    #[must_use]
    pub fn unresolved_comment_count(&self) -> usize {
        self.comments.iter().filter(|c| !c.resolved).count()
    }

    /// Count resolved comments.
    #[must_use]
    pub fn resolved_comment_count(&self) -> usize {
        self.comments.iter().filter(|c| c.resolved).count()
    }
}

/// The review exporter converts `ExportData` to the requested format.
#[derive(Debug)]
pub struct ReviewExporter;

impl ReviewExporter {
    /// Export the data to a string in the format specified by the config.
    #[must_use]
    pub fn export(data: &ExportData, config: &ExportConfig) -> String {
        match config.format {
            ExportFormat::Json => Self::export_json(data, config),
            ExportFormat::Xml => Self::export_xml(data, config),
            ExportFormat::PlainText => Self::export_plain_text(data, config),
            ExportFormat::Csv => Self::export_csv(data, config),
            ExportFormat::Tsv => Self::export_tsv(data, config),
        }
    }

    /// Export to JSON format.
    #[must_use]
    fn export_json(data: &ExportData, config: &ExportConfig) -> String {
        let title = config.title.as_deref().unwrap_or(&data.title);
        let mut result = String::new();
        result.push_str("{\n");
        result.push_str(&format!("  \"title\": \"{title}\",\n"));
        result.push_str(&format!("  \"status\": \"{}\",\n", data.status));

        // Comments
        result.push_str("  \"comments\": [\n");
        let comments: Vec<&ExportComment> = data
            .comments
            .iter()
            .filter(|c| config.include_resolved || !c.resolved)
            .collect();
        for (i, c) in comments.iter().enumerate() {
            result.push_str("    {\n");
            result.push_str(&format!("      \"author\": \"{}\",\n", c.author));
            result.push_str(&format!("      \"text\": \"{}\",\n", c.text));
            result.push_str(&format!("      \"resolved\": {},\n", c.resolved));
            result.push_str(&format!("      \"category\": \"{}\"\n", c.category));
            if i + 1 < comments.len() {
                result.push_str("    },\n");
            } else {
                result.push_str("    }\n");
            }
        }
        result.push_str("  ]\n");
        result.push('}');
        result
    }

    /// Export to XML format.
    #[must_use]
    fn export_xml(data: &ExportData, config: &ExportConfig) -> String {
        let title = config.title.as_deref().unwrap_or(&data.title);
        let mut result = String::new();
        result.push_str("<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n");
        result.push_str("<review>\n");
        result.push_str(&format!("  <title>{title}</title>\n"));
        result.push_str(&format!("  <status>{}</status>\n", data.status));

        result.push_str("  <comments>\n");
        for c in &data.comments {
            if !config.include_resolved && c.resolved {
                continue;
            }
            result.push_str("    <comment>\n");
            result.push_str(&format!("      <author>{}</author>\n", c.author));
            result.push_str(&format!("      <text>{}</text>\n", c.text));
            result.push_str(&format!("      <resolved>{}</resolved>\n", c.resolved));
            result.push_str("    </comment>\n");
        }
        result.push_str("  </comments>\n");
        result.push_str("</review>");
        result
    }

    /// Export to plain-text summary.
    #[must_use]
    fn export_plain_text(data: &ExportData, config: &ExportConfig) -> String {
        let title = config.title.as_deref().unwrap_or(&data.title);
        let mut result = String::new();
        result.push_str(&format!("Review: {title}\n"));
        result.push_str(&format!("Status: {}\n", data.status));
        result.push_str(&format!(
            "Comments: {} total, {} unresolved\n",
            data.comments.len(),
            data.unresolved_comment_count()
        ));
        result.push('\n');

        for c in &data.comments {
            if !config.include_resolved && c.resolved {
                continue;
            }
            let resolved_tag = if c.resolved { " [RESOLVED]" } else { "" };
            result.push_str(&format!("- [{}]{} {}\n", c.author, resolved_tag, c.text));
        }

        if config.include_history && !data.history.is_empty() {
            result.push_str("\nHistory:\n");
            for entry in &data.history {
                result.push_str(&format!("  {entry}\n"));
            }
        }

        result
    }

    /// Export comments to CSV format.
    #[must_use]
    fn export_csv(data: &ExportData, config: &ExportConfig) -> String {
        Self::export_delimited(data, config, ',')
    }

    /// Export comments to TSV format.
    #[must_use]
    fn export_tsv(data: &ExportData, config: &ExportConfig) -> String {
        Self::export_delimited(data, config, '\t')
    }

    /// Export comments with a delimiter.
    #[must_use]
    fn export_delimited(data: &ExportData, config: &ExportConfig, delim: char) -> String {
        let mut result = String::new();
        result.push_str(&format!(
            "author{delim}text{delim}frame{delim}resolved{delim}category\n"
        ));
        for c in &data.comments {
            if !config.include_resolved && c.resolved {
                continue;
            }
            let frame_str = c.frame.map_or_else(String::new, |f| f.to_string());
            result.push_str(&format!(
                "{}{delim}{}{delim}{}{delim}{}{delim}{}\n",
                c.author, c.text, frame_str, c.resolved, c.category
            ));
        }
        result
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_data() -> ExportData {
        let mut data = ExportData::new("Test Review", "In Progress");
        data.add_comment(ExportComment {
            author: "Alice".to_string(),
            text: "Fix the color grading".to_string(),
            frame: Some(1000),
            timestamp_ms: 100_000,
            resolved: false,
            category: "color".to_string(),
        });
        data.add_comment(ExportComment {
            author: "Bob".to_string(),
            text: "Audio levels look good".to_string(),
            frame: Some(2000),
            timestamp_ms: 200_000,
            resolved: true,
            category: "audio".to_string(),
        });
        data.add_history_entry("Draft -> Open");
        data.add_history_entry("Open -> In Progress");
        data
    }

    #[test]
    fn test_export_format_display() {
        assert_eq!(ExportFormat::Json.to_string(), "JSON");
        assert_eq!(ExportFormat::Xml.to_string(), "XML");
        assert_eq!(ExportFormat::PlainText.to_string(), "Plain Text");
        assert_eq!(ExportFormat::Csv.to_string(), "CSV");
        assert_eq!(ExportFormat::Tsv.to_string(), "TSV");
    }

    #[test]
    fn test_export_format_extension() {
        assert_eq!(ExportFormat::Json.extension(), "json");
        assert_eq!(ExportFormat::Xml.extension(), "xml");
        assert_eq!(ExportFormat::PlainText.extension(), "txt");
        assert_eq!(ExportFormat::Csv.extension(), "csv");
        assert_eq!(ExportFormat::Tsv.extension(), "tsv");
    }

    #[test]
    fn test_export_format_mime_type() {
        assert_eq!(ExportFormat::Json.mime_type(), "application/json");
        assert_eq!(ExportFormat::Xml.mime_type(), "application/xml");
        assert_eq!(ExportFormat::Csv.mime_type(), "text/csv");
    }

    #[test]
    fn test_export_config_defaults() {
        let config = ExportConfig::default();
        assert_eq!(config.format, ExportFormat::Json);
        assert!(config.include_resolved);
        assert!(config.include_annotations);
        assert!(config.include_history);
        assert!(config.title.is_none());
    }

    #[test]
    fn test_export_config_builder_pattern() {
        let config = ExportConfig::new(ExportFormat::Csv)
            .with_resolved(false)
            .with_annotations(false)
            .with_history(false)
            .with_title("Custom Title")
            .with_field("project", "OxiMedia");
        assert_eq!(config.format, ExportFormat::Csv);
        assert!(!config.include_resolved);
        assert!(!config.include_annotations);
        assert!(!config.include_history);
        assert_eq!(config.title.as_deref(), Some("Custom Title"));
        assert_eq!(
            config.extra_fields.get("project"),
            Some(&"OxiMedia".to_string())
        );
    }

    #[test]
    fn test_export_data_counts() {
        let data = sample_data();
        assert_eq!(data.comments.len(), 2);
        assert_eq!(data.unresolved_comment_count(), 1);
        assert_eq!(data.resolved_comment_count(), 1);
    }

    #[test]
    fn test_export_json_contains_title() {
        let data = sample_data();
        let config = ExportConfig::new(ExportFormat::Json);
        let output = ReviewExporter::export(&data, &config);
        assert!(output.contains("\"title\": \"Test Review\""));
        assert!(output.contains("\"status\": \"In Progress\""));
    }

    #[test]
    fn test_export_json_custom_title() {
        let data = sample_data();
        let config = ExportConfig::new(ExportFormat::Json).with_title("Override");
        let output = ReviewExporter::export(&data, &config);
        assert!(output.contains("\"title\": \"Override\""));
    }

    #[test]
    fn test_export_xml_structure() {
        let data = sample_data();
        let config = ExportConfig::new(ExportFormat::Xml);
        let output = ReviewExporter::export(&data, &config);
        assert!(output.contains("<?xml version"));
        assert!(output.contains("<review>"));
        assert!(output.contains("<title>Test Review</title>"));
        assert!(output.contains("</review>"));
    }

    #[test]
    fn test_export_plain_text_summary() {
        let data = sample_data();
        let config = ExportConfig::new(ExportFormat::PlainText);
        let output = ReviewExporter::export(&data, &config);
        assert!(output.contains("Review: Test Review"));
        assert!(output.contains("Comments: 2 total, 1 unresolved"));
        assert!(output.contains("[Alice]"));
        assert!(output.contains("History:"));
    }

    #[test]
    fn test_export_csv_header() {
        let data = sample_data();
        let config = ExportConfig::new(ExportFormat::Csv);
        let output = ReviewExporter::export(&data, &config);
        assert!(output.starts_with("author,text,frame,resolved,category\n"));
    }

    #[test]
    fn test_export_tsv_header() {
        let data = sample_data();
        let config = ExportConfig::new(ExportFormat::Tsv);
        let output = ReviewExporter::export(&data, &config);
        assert!(output.starts_with("author\ttext\tframe\tresolved\tcategory\n"));
    }

    #[test]
    fn test_export_filter_resolved() {
        let data = sample_data();
        let config = ExportConfig::new(ExportFormat::PlainText).with_resolved(false);
        let output = ReviewExporter::export(&data, &config);
        assert!(output.contains("[Alice]"));
        assert!(!output.contains("[Bob]"));
    }

    #[test]
    fn test_export_annotation() {
        let mut data = ExportData::new("Test", "Open");
        data.add_annotation(ExportAnnotation {
            author: "Alice".to_string(),
            annotation_type: "circle".to_string(),
            frame: 500,
            x: 0.5,
            y: 0.3,
            label: "Check this area".to_string(),
        });
        assert_eq!(data.annotations.len(), 1);
        assert_eq!(data.annotations[0].label, "Check this area");
    }
}
