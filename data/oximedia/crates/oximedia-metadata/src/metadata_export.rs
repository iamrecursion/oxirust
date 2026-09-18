#![allow(dead_code)]
//! Metadata export to text-based formats.
//!
//! Supports exporting individual records or batches to JSON, CSV, XML, and TSV.

use std::collections::HashMap;
use std::fmt::Write as FmtWrite;

/// Supported export formats.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ExportFormat {
    /// JavaScript Object Notation.
    Json,
    /// Comma-separated values.
    Csv,
    /// Extensible Markup Language.
    Xml,
    /// Tab-separated values.
    Tsv,
}

impl ExportFormat {
    /// Returns the conventional file extension for this format (without leading dot).
    pub fn extension(self) -> &'static str {
        match self {
            Self::Json => "json",
            Self::Csv => "csv",
            Self::Xml => "xml",
            Self::Tsv => "tsv",
        }
    }

    /// Returns the MIME type for this format.
    pub fn mime_type(self) -> &'static str {
        match self {
            Self::Json => "application/json",
            Self::Csv => "text/csv",
            Self::Xml => "application/xml",
            Self::Tsv => "text/tab-separated-values",
        }
    }
}

impl std::fmt::Display for ExportFormat {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Json => write!(f, "JSON"),
            Self::Csv => write!(f, "CSV"),
            Self::Xml => write!(f, "XML"),
            Self::Tsv => write!(f, "TSV"),
        }
    }
}

/// Exports metadata records to text-based formats.
pub struct MetadataExporter {
    format: ExportFormat,
    records: Vec<HashMap<String, String>>,
}

impl MetadataExporter {
    /// Create a new exporter for the given format.
    pub fn new(format: ExportFormat) -> Self {
        Self {
            format,
            records: Vec::new(),
        }
    }

    /// Add a single record to the export buffer.
    pub fn export_record(&mut self, record: HashMap<String, String>) {
        self.records.push(record);
    }

    /// Add multiple records at once.
    pub fn export_batch(&mut self, records: impl IntoIterator<Item = HashMap<String, String>>) {
        self.records.extend(records);
    }

    /// Returns the number of buffered records.
    pub fn record_count(&self) -> usize {
        self.records.len()
    }

    /// Clear all buffered records.
    pub fn clear(&mut self) {
        self.records.clear();
    }

    /// Render all buffered records to a string in the configured format.
    pub fn render(&self) -> String {
        match self.format {
            ExportFormat::Json => self.render_json(),
            ExportFormat::Csv => self.render_csv(),
            ExportFormat::Xml => self.render_xml(),
            ExportFormat::Tsv => self.render_tsv(),
        }
    }

    // ── private helpers ──────────────────────────────────────────────────────

    fn collect_keys(&self) -> Vec<String> {
        let mut keys: Vec<String> = self
            .records
            .iter()
            .flat_map(|r| r.keys().cloned())
            .collect::<std::collections::HashSet<_>>()
            .into_iter()
            .collect();
        keys.sort();
        keys
    }

    fn render_json(&self) -> String {
        let mut out = String::from("[\n");
        for (i, record) in self.records.iter().enumerate() {
            out.push_str("  {");
            let mut pairs: Vec<_> = record.iter().collect();
            pairs.sort_by_key(|(k, _)| k.as_str());
            for (j, (k, v)) in pairs.iter().enumerate() {
                let comma = if j + 1 < pairs.len() { "," } else { "" };
                let _ = write!(
                    out,
                    "\n    \"{}\": \"{}\"{}",
                    Self::json_escape(k),
                    Self::json_escape(v),
                    comma
                );
            }
            let comma = if i + 1 < self.records.len() { "," } else { "" };
            out.push_str("\n  }");
            out.push_str(comma);
            out.push('\n');
        }
        out.push(']');
        out
    }

    fn render_csv(&self) -> String {
        self.render_delimited(',')
    }

    fn render_tsv(&self) -> String {
        self.render_delimited('\t')
    }

    fn render_delimited(&self, sep: char) -> String {
        let keys = self.collect_keys();
        let mut out = String::new();
        // Header row.
        let header: Vec<_> = keys.iter().map(|k| Self::quote_delimited(k, sep)).collect();
        out.push_str(&header.join(&sep.to_string()));
        out.push('\n');
        // Data rows.
        for record in &self.records {
            let row: Vec<_> = keys
                .iter()
                .map(|k| {
                    let val = record.get(k).map(String::as_str).unwrap_or("");
                    Self::quote_delimited(val, sep)
                })
                .collect();
            out.push_str(&row.join(&sep.to_string()));
            out.push('\n');
        }
        out
    }

    fn render_xml(&self) -> String {
        let mut out = String::from("<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n<records>\n");
        for record in &self.records {
            out.push_str("  <record>\n");
            let mut pairs: Vec<_> = record.iter().collect();
            pairs.sort_by_key(|(k, _)| k.as_str());
            for (k, v) in pairs {
                let _ = writeln!(
                    out,
                    "    <field name=\"{}\">{}</field>",
                    Self::xml_escape(k),
                    Self::xml_escape(v)
                );
            }
            out.push_str("  </record>\n");
        }
        out.push_str("</records>");
        out
    }

    fn json_escape(s: &str) -> String {
        s.replace('\\', "\\\\")
            .replace('"', "\\\"")
            .replace('\n', "\\n")
            .replace('\r', "\\r")
    }

    fn xml_escape(s: &str) -> String {
        s.replace('&', "&amp;")
            .replace('<', "&lt;")
            .replace('>', "&gt;")
            .replace('"', "&quot;")
    }

    fn quote_delimited(s: &str, sep: char) -> String {
        if s.contains(sep) || s.contains('"') || s.contains('\n') {
            format!("\"{}\"", s.replace('"', "\"\""))
        } else {
            s.to_string()
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_record(pairs: &[(&str, &str)]) -> HashMap<String, String> {
        pairs
            .iter()
            .map(|(k, v)| (k.to_string(), v.to_string()))
            .collect()
    }

    #[test]
    fn test_extension_json() {
        assert_eq!(ExportFormat::Json.extension(), "json");
    }

    #[test]
    fn test_extension_csv() {
        assert_eq!(ExportFormat::Csv.extension(), "csv");
    }

    #[test]
    fn test_extension_xml() {
        assert_eq!(ExportFormat::Xml.extension(), "xml");
    }

    #[test]
    fn test_extension_tsv() {
        assert_eq!(ExportFormat::Tsv.extension(), "tsv");
    }

    #[test]
    fn test_mime_type_json() {
        assert_eq!(ExportFormat::Json.mime_type(), "application/json");
    }

    #[test]
    fn test_mime_type_csv() {
        assert_eq!(ExportFormat::Csv.mime_type(), "text/csv");
    }

    #[test]
    fn test_record_count_empty() {
        let exporter = MetadataExporter::new(ExportFormat::Json);
        assert_eq!(exporter.record_count(), 0);
    }

    #[test]
    fn test_export_single_record() {
        let mut exporter = MetadataExporter::new(ExportFormat::Json);
        exporter.export_record(make_record(&[("title", "Test")]));
        assert_eq!(exporter.record_count(), 1);
    }

    #[test]
    fn test_export_batch() {
        let mut exporter = MetadataExporter::new(ExportFormat::Csv);
        let records = vec![
            make_record(&[("a", "1")]),
            make_record(&[("a", "2")]),
            make_record(&[("a", "3")]),
        ];
        exporter.export_batch(records);
        assert_eq!(exporter.record_count(), 3);
    }

    #[test]
    fn test_render_json_contains_key() {
        let mut exporter = MetadataExporter::new(ExportFormat::Json);
        exporter.export_record(make_record(&[("artist", "OxiMedia")]));
        let out = exporter.render();
        assert!(out.contains("\"artist\""));
        assert!(out.contains("\"OxiMedia\""));
    }

    #[test]
    fn test_render_csv_has_header() {
        let mut exporter = MetadataExporter::new(ExportFormat::Csv);
        exporter.export_record(make_record(&[("title", "Song")]));
        let out = exporter.render();
        let lines: Vec<_> = out.lines().collect();
        assert_eq!(lines[0], "title");
        assert_eq!(lines[1], "Song");
    }

    #[test]
    fn test_render_tsv_has_tab_separator() {
        let mut exporter = MetadataExporter::new(ExportFormat::Tsv);
        exporter.export_record(make_record(&[("a", "1"), ("b", "2")]));
        let out = exporter.render();
        assert!(out.contains('\t'));
    }

    #[test]
    fn test_render_xml_structure() {
        let mut exporter = MetadataExporter::new(ExportFormat::Xml);
        exporter.export_record(make_record(&[("track", "1")]));
        let out = exporter.render();
        assert!(out.contains("<records>"));
        assert!(out.contains("<record>"));
        assert!(out.contains("name=\"track\""));
    }

    #[test]
    fn test_clear_records() {
        let mut exporter = MetadataExporter::new(ExportFormat::Json);
        exporter.export_record(make_record(&[("x", "y")]));
        assert_eq!(exporter.record_count(), 1);
        exporter.clear();
        assert_eq!(exporter.record_count(), 0);
    }

    #[test]
    fn test_display_format() {
        assert_eq!(ExportFormat::Json.to_string(), "JSON");
        assert_eq!(ExportFormat::Csv.to_string(), "CSV");
        assert_eq!(ExportFormat::Tsv.to_string(), "TSV");
    }
}
