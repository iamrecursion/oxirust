//! Report formatting and export utilities for profiling sessions.
//!
//! Supports serialising profiling results to JSON, CSV, and a plain-text
//! human-readable format.

#![allow(dead_code)]

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::time::Duration;

/// Output format for a profiling report.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ReportFormat {
    /// Compact machine-readable JSON.
    Json,
    /// Pretty-printed JSON.
    JsonPretty,
    /// Comma-separated values (header row + data rows).
    Csv,
    /// Human-readable plain text.
    PlainText,
}

impl ReportFormat {
    /// Returns the conventional file extension for this format.
    #[must_use]
    pub fn extension(&self) -> &'static str {
        match self {
            Self::Json | Self::JsonPretty => "json",
            Self::Csv => "csv",
            Self::PlainText => "txt",
        }
    }

    /// Returns `true` if the format is a JSON variant.
    #[must_use]
    pub fn is_json(&self) -> bool {
        matches!(self, Self::Json | Self::JsonPretty)
    }
}

/// A named section within a profiling report.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReportSection {
    /// Section heading.
    pub title: String,
    /// Key-value pairs belonging to this section.
    pub entries: Vec<(String, String)>,
}

impl ReportSection {
    /// Creates a new, empty section with the given title.
    #[must_use]
    pub fn new(title: impl Into<String>) -> Self {
        Self {
            title: title.into(),
            entries: Vec::new(),
        }
    }

    /// Adds an entry to this section.
    pub fn add(&mut self, key: impl Into<String>, value: impl Into<String>) {
        self.entries.push((key.into(), value.into()));
    }

    /// Builder-style entry add, consuming and returning `self`.
    #[must_use]
    pub fn with_entry(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        self.add(key, value);
        self
    }
}

/// Compiled profiling report ready for export.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProfileReport {
    /// Human-readable report title.
    pub title: String,
    /// Wall-clock duration of the profiled session.
    pub duration: Duration,
    /// Named sections containing profiling data.
    pub sections: Vec<ReportSection>,
    /// Flat key-value metadata (e.g. version, host).
    pub metadata: HashMap<String, String>,
}

impl ProfileReport {
    /// Creates a new `ProfileReport` with the given title and session duration.
    #[must_use]
    pub fn new(title: impl Into<String>, duration: Duration) -> Self {
        Self {
            title: title.into(),
            duration,
            sections: Vec::new(),
            metadata: HashMap::new(),
        }
    }

    /// Appends a section to the report.
    pub fn add_section(&mut self, section: ReportSection) {
        self.sections.push(section);
    }

    /// Adds a metadata key-value pair.
    pub fn add_metadata(&mut self, key: impl Into<String>, value: impl Into<String>) {
        self.metadata.insert(key.into(), value.into());
    }

    /// Returns the number of sections.
    #[must_use]
    pub fn section_count(&self) -> usize {
        self.sections.len()
    }

    /// Returns the first section with the given title, if any.
    #[must_use]
    pub fn find_section(&self, title: &str) -> Option<&ReportSection> {
        self.sections.iter().find(|s| s.title == title)
    }
}

/// Exports a [`ProfileReport`] to various text-based formats.
///
/// All methods return `String` so the caller can write to any sink.
#[derive(Debug, Default)]
pub struct ReportExporter;

impl ReportExporter {
    /// Creates a new `ReportExporter`.
    #[must_use]
    pub fn new() -> Self {
        Self
    }

    /// Serialises the report to compact JSON.
    ///
    /// # Errors
    ///
    /// Returns a `serde_json` error if serialisation fails.
    pub fn to_json(&self, report: &ProfileReport) -> Result<String, serde_json::Error> {
        serde_json::to_string(report)
    }

    /// Serialises the report to pretty-printed JSON.
    ///
    /// # Errors
    ///
    /// Returns a `serde_json` error if serialisation fails.
    pub fn to_json_pretty(&self, report: &ProfileReport) -> Result<String, serde_json::Error> {
        serde_json::to_string_pretty(report)
    }

    /// Serialises the report to CSV format.
    ///
    /// Each section becomes a group of rows: `section,key,value`.
    #[must_use]
    pub fn to_csv(&self, report: &ProfileReport) -> String {
        let mut out = String::from("section,key,value\n");
        for section in &report.sections {
            for (k, v) in &section.entries {
                let escaped_v = v.replace('"', "\"\"");
                out.push_str(&format!("{},{},\"{}\"\n", section.title, k, escaped_v));
            }
        }
        out
    }

    /// Renders the report as plain text.
    #[must_use]
    pub fn to_plain_text(&self, report: &ProfileReport) -> String {
        let mut out = String::new();
        out.push_str(&format!("=== {} ===\n", report.title));
        out.push_str(&format!("Duration: {:?}\n\n", report.duration));

        if !report.metadata.is_empty() {
            out.push_str("[Metadata]\n");
            let mut meta: Vec<(&String, &String)> = report.metadata.iter().collect();
            meta.sort_by_key(|(k, _)| k.as_str());
            for (k, v) in meta {
                out.push_str(&format!("  {k}: {v}\n"));
            }
            out.push('\n');
        }

        for section in &report.sections {
            out.push_str(&format!("[{}]\n", section.title));
            for (k, v) in &section.entries {
                out.push_str(&format!("  {k}: {v}\n"));
            }
            out.push('\n');
        }
        out
    }

    /// Exports the report using the specified [`ReportFormat`].
    ///
    /// Returns the serialised string on success, or a `String` error message.
    pub fn export(&self, report: &ProfileReport, format: ReportFormat) -> Result<String, String> {
        match format {
            ReportFormat::Json => self.to_json(report).map_err(|e| e.to_string()),
            ReportFormat::JsonPretty => self.to_json_pretty(report).map_err(|e| e.to_string()),
            ReportFormat::Csv => Ok(self.to_csv(report)),
            ReportFormat::PlainText => Ok(self.to_plain_text(report)),
        }
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_report() -> ProfileReport {
        let mut report = ProfileReport::new("Test Report", Duration::from_millis(500));
        report.add_metadata("version", "1.0.0");
        let mut sec = ReportSection::new("CPU");
        sec.add("user_pct", "42.5");
        sec.add("sys_pct", "7.3");
        report.add_section(sec);
        report
    }

    #[test]
    fn test_report_creation() {
        let r = sample_report();
        assert_eq!(r.title, "Test Report");
        assert_eq!(r.duration, Duration::from_millis(500));
        assert_eq!(r.section_count(), 1);
    }

    #[test]
    fn test_find_section_found() {
        let r = sample_report();
        assert!(r.find_section("CPU").is_some());
    }

    #[test]
    fn test_find_section_missing() {
        let r = sample_report();
        assert!(r.find_section("GPU").is_none());
    }

    #[test]
    fn test_to_json_contains_title() {
        let r = sample_report();
        let exporter = ReportExporter::new();
        let json = exporter.to_json(&r).expect("should succeed in test");
        assert!(json.contains("Test Report"));
    }

    #[test]
    fn test_to_json_pretty_is_multiline() {
        let r = sample_report();
        let exporter = ReportExporter::new();
        let json = exporter.to_json_pretty(&r).expect("should succeed in test");
        assert!(json.contains('\n'));
    }

    #[test]
    fn test_to_json_roundtrip() {
        let r = sample_report();
        let exporter = ReportExporter::new();
        let json = exporter.to_json(&r).expect("should succeed in test");
        let decoded: ProfileReport = serde_json::from_str(&json).expect("should succeed in test");
        assert_eq!(decoded.title, r.title);
        assert_eq!(decoded.section_count(), r.section_count());
    }

    #[test]
    fn test_to_csv_has_header() {
        let r = sample_report();
        let exporter = ReportExporter::new();
        let csv = exporter.to_csv(&r);
        assert!(csv.starts_with("section,key,value\n"));
    }

    #[test]
    fn test_to_csv_contains_data() {
        let r = sample_report();
        let exporter = ReportExporter::new();
        let csv = exporter.to_csv(&r);
        assert!(csv.contains("CPU,user_pct"));
    }

    #[test]
    fn test_to_plain_text_contains_title() {
        let r = sample_report();
        let exporter = ReportExporter::new();
        let text = exporter.to_plain_text(&r);
        assert!(text.contains("Test Report"));
        assert!(text.contains("[CPU]"));
    }

    #[test]
    fn test_export_json_format() {
        let r = sample_report();
        let exporter = ReportExporter::new();
        let result = exporter.export(&r, ReportFormat::Json);
        assert!(result.is_ok());
    }

    #[test]
    fn test_export_plain_text_format() {
        let r = sample_report();
        let exporter = ReportExporter::new();
        let result = exporter.export(&r, ReportFormat::PlainText);
        assert!(result.is_ok());
        assert!(result
            .expect("should succeed in test")
            .contains("Duration:"));
    }

    #[test]
    fn test_report_format_extension() {
        assert_eq!(ReportFormat::Json.extension(), "json");
        assert_eq!(ReportFormat::Csv.extension(), "csv");
        assert_eq!(ReportFormat::PlainText.extension(), "txt");
    }

    #[test]
    fn test_report_format_is_json() {
        assert!(ReportFormat::Json.is_json());
        assert!(ReportFormat::JsonPretty.is_json());
        assert!(!ReportFormat::Csv.is_json());
    }

    #[test]
    fn test_section_with_entry_builder() {
        let sec = ReportSection::new("Memory")
            .with_entry("heap_mb", "256")
            .with_entry("rss_mb", "512");
        assert_eq!(sec.entries.len(), 2);
    }

    #[test]
    fn test_metadata_in_plain_text() {
        let r = sample_report();
        let exporter = ReportExporter::new();
        let text = exporter.to_plain_text(&r);
        assert!(text.contains("version: 1.0.0"));
    }
}
