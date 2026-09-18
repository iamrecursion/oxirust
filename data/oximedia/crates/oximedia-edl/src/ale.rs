//! ALE (Avid Log Exchange) format parsing and generation.
//!
//! ALE is a tab-delimited format used by Avid Media Composer for exchanging
//! metadata about media clips. It consists of three sections:
//! - Heading: key-value pairs (field separator, video format, etc.)
//! - Column: tab-separated column names
//! - Data: tab-separated rows of clip metadata
//!
//! # Format Example
//!
//! ```text
//! Heading
//! FIELD_DELIM\tTABS
//! VIDEO_FORMAT\t1080i
//! AUDIO_FORMAT\t48kHz
//! FPS\t25
//!
//! Column
//! Name\tStart\tEnd\tTracks\tDuration\tTape
//!
//! Data
//! SHOT_001\t01:00:00:00\t01:00:05:00\tV\t125\tA001
//! SHOT_002\t01:00:05:00\t01:00:12:00\tV\t175\tA002
//! ```

#![allow(dead_code)]

use crate::error::{EdlError, EdlResult};
use std::collections::HashMap;

/// An ALE document containing heading, column definitions, and data rows.
#[derive(Debug, Clone)]
pub struct AleDocument {
    /// Heading section key-value pairs.
    pub heading: HashMap<String, String>,
    /// Ordered column names.
    pub columns: Vec<String>,
    /// Data rows, each a map from column name to value.
    pub rows: Vec<AleRow>,
}

/// A single row in an ALE data section.
#[derive(Debug, Clone)]
pub struct AleRow {
    /// Column values indexed by column name.
    pub fields: HashMap<String, String>,
}

impl AleRow {
    /// Get a field value by column name.
    #[must_use]
    pub fn get(&self, column: &str) -> Option<&str> {
        self.fields.get(column).map(|s| s.as_str())
    }

    /// Get a field value or a default.
    #[must_use]
    pub fn get_or<'a>(&'a self, column: &str, default: &'a str) -> &'a str {
        self.fields.get(column).map_or(default, |s| s.as_str())
    }

    /// Check if this row has a non-empty value for the given column.
    #[must_use]
    pub fn has(&self, column: &str) -> bool {
        self.fields
            .get(column)
            .is_some_and(|v| !v.trim().is_empty())
    }
}

impl AleDocument {
    /// Create a new empty ALE document.
    #[must_use]
    pub fn new() -> Self {
        Self {
            heading: HashMap::new(),
            columns: Vec::new(),
            rows: Vec::new(),
        }
    }

    /// Get the field delimiter (defaults to tab).
    #[must_use]
    pub fn field_delimiter(&self) -> &str {
        self.heading
            .get("FIELD_DELIM")
            .map_or("\t", |v| match v.to_uppercase().as_str() {
                "TABS" | "TAB" => "\t",
                _ => "\t",
            })
    }

    /// Get the video format from heading.
    #[must_use]
    pub fn video_format(&self) -> Option<&str> {
        self.heading.get("VIDEO_FORMAT").map(|s| s.as_str())
    }

    /// Get the audio format from heading.
    #[must_use]
    pub fn audio_format(&self) -> Option<&str> {
        self.heading.get("AUDIO_FORMAT").map(|s| s.as_str())
    }

    /// Get the FPS from heading.
    #[must_use]
    pub fn fps(&self) -> Option<f64> {
        self.heading.get("FPS").and_then(|s| s.parse::<f64>().ok())
    }

    /// Get the number of data rows.
    #[must_use]
    pub fn row_count(&self) -> usize {
        self.rows.len()
    }

    /// Get a column index by name.
    #[must_use]
    pub fn column_index(&self, name: &str) -> Option<usize> {
        self.columns.iter().position(|c| c == name)
    }

    /// Check if a column exists.
    #[must_use]
    pub fn has_column(&self, name: &str) -> bool {
        self.columns.iter().any(|c| c == name)
    }

    /// Add a heading entry.
    pub fn set_heading(&mut self, key: impl Into<String>, value: impl Into<String>) {
        self.heading.insert(key.into(), value.into());
    }

    /// Add a column definition.
    pub fn add_column(&mut self, name: impl Into<String>) {
        self.columns.push(name.into());
    }

    /// Add a data row from a map of column name to value.
    pub fn add_row(&mut self, fields: HashMap<String, String>) {
        self.rows.push(AleRow { fields });
    }

    /// Get all values for a specific column across all rows.
    #[must_use]
    pub fn column_values(&self, column: &str) -> Vec<&str> {
        self.rows.iter().filter_map(|r| r.get(column)).collect()
    }

    /// Filter rows where a column matches a value.
    #[must_use]
    pub fn filter_rows(&self, column: &str, value: &str) -> Vec<&AleRow> {
        self.rows
            .iter()
            .filter(|r| r.get(column) == Some(value))
            .collect()
    }

    /// Sort rows by a column (lexicographic).
    pub fn sort_by_column(&mut self, column: &str) {
        let col = column.to_string();
        self.rows.sort_by(|a, b| {
            let va = a.get(&col).unwrap_or("");
            let vb = b.get(&col).unwrap_or("");
            va.cmp(vb)
        });
    }
}

impl Default for AleDocument {
    fn default() -> Self {
        Self::new()
    }
}

// ────────────────────────────────────────────────────────────────────────────
// Parser
// ────────────────────────────────────────────────────────────────────────────

/// Current parsing section.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum AleSection {
    None,
    Heading,
    Column,
    Data,
}

/// Parse an ALE format string into an `AleDocument`.
///
/// # Errors
///
/// Returns an error if the ALE format is invalid.
pub fn parse_ale(input: &str) -> EdlResult<AleDocument> {
    let mut doc = AleDocument::new();
    let mut section = AleSection::None;

    for (line_num, line) in input.lines().enumerate() {
        let trimmed = line.trim();

        if trimmed.is_empty() {
            continue;
        }

        // Section headers
        match trimmed {
            "Heading" | "HEADING" => {
                section = AleSection::Heading;
                continue;
            }
            "Column" | "COLUMN" => {
                section = AleSection::Column;
                continue;
            }
            "Data" | "DATA" => {
                section = AleSection::Data;
                continue;
            }
            _ => {}
        }

        match section {
            AleSection::None => {
                // Lines before any section header are ignored
            }
            AleSection::Heading => {
                // Parse key-value pair (tab-separated)
                if let Some((key, value)) = trimmed.split_once('\t') {
                    doc.heading
                        .insert(key.trim().to_string(), value.trim().to_string());
                } else {
                    // Single-value heading line (the key itself)
                    doc.heading.insert(trimmed.to_string(), String::new());
                }
            }
            AleSection::Column => {
                // Parse column names (tab-separated)
                let cols: Vec<String> = trimmed.split('\t').map(|s| s.trim().to_string()).collect();
                if cols.is_empty() {
                    return Err(EdlError::parse(
                        line_num + 1,
                        "Empty column definition line",
                    ));
                }
                doc.columns = cols;
            }
            AleSection::Data => {
                // Parse data row (tab-separated, matching column count)
                let values: Vec<&str> = trimmed.split('\t').collect();
                if values.len() != doc.columns.len() {
                    return Err(EdlError::parse(
                        line_num + 1,
                        format!(
                            "Data row has {} fields but {} columns defined",
                            values.len(),
                            doc.columns.len()
                        ),
                    ));
                }
                let mut fields = HashMap::new();
                for (col, val) in doc.columns.iter().zip(values.iter()) {
                    fields.insert(col.clone(), val.trim().to_string());
                }
                doc.rows.push(AleRow { fields });
            }
        }
    }

    if doc.columns.is_empty() {
        return Err(EdlError::parse(0, "No column definitions found in ALE"));
    }

    Ok(doc)
}

// ────────────────────────────────────────────────────────────────────────────
// Generator
// ────────────────────────────────────────────────────────────────────────────

/// Generate an ALE format string from an `AleDocument`.
///
/// # Errors
///
/// Returns an error if the document has no columns.
pub fn generate_ale(doc: &AleDocument) -> EdlResult<String> {
    if doc.columns.is_empty() {
        return Err(EdlError::validation("ALE document has no columns defined"));
    }

    let mut output = String::new();

    // Heading section
    output.push_str("Heading\n");
    // Always write FIELD_DELIM first
    output.push_str("FIELD_DELIM\tTABS\n");
    for (key, value) in &doc.heading {
        if key == "FIELD_DELIM" {
            continue; // Already written
        }
        if value.is_empty() {
            output.push_str(&format!("{key}\n"));
        } else {
            output.push_str(&format!("{key}\t{value}\n"));
        }
    }
    output.push('\n');

    // Column section
    output.push_str("Column\n");
    output.push_str(&doc.columns.join("\t"));
    output.push('\n');
    output.push('\n');

    // Data section
    output.push_str("Data\n");
    for row in &doc.rows {
        let values: Vec<&str> = doc
            .columns
            .iter()
            .map(|col| row.get(col).unwrap_or(""))
            .collect();
        output.push_str(&values.join("\t"));
        output.push('\n');
    }

    Ok(output)
}

// ────────────────────────────────────────────────────────────────────────────
// AleRecord — flat convenience type for clip-oriented access
// ────────────────────────────────────────────────────────────────────────────

/// A single ALE clip record with commonly-used fields surfaced directly,
/// plus an open-ended `fields` map for all remaining columns.
///
/// The `clip_name` field is populated from the `Name` column, `tape` from
/// the `Tape` column, `start` from `Start`, and `end` from `End`.  All
/// remaining columns are stored verbatim in `fields`.
#[derive(Debug, Clone)]
pub struct AleRecord {
    /// Clip name (from the `Name` column).
    pub clip_name: String,
    /// Tape/reel identifier (from the `Tape` column).
    pub tape: String,
    /// Source-in timecode string (from the `Start` column).
    pub start: String,
    /// Source-out timecode string (from the `End` column).
    pub end: String,
    /// All other column values keyed by column name.
    pub fields: std::collections::HashMap<String, String>,
}

impl AleRecord {
    /// Build an `AleRecord` from a raw field map and the ordered column list.
    fn from_fields(fields: std::collections::HashMap<String, String>) -> Self {
        let clip_name = fields.get("Name").cloned().unwrap_or_default();
        let tape = fields.get("Tape").cloned().unwrap_or_default();
        let start = fields.get("Start").cloned().unwrap_or_default();
        let end = fields.get("End").cloned().unwrap_or_default();
        Self {
            clip_name,
            tape,
            start,
            end,
            fields,
        }
    }
}

/// Parse an ALE format string and return a `Vec<AleRecord>`.
///
/// This is a high-level convenience wrapper around [`parse_ale`] that maps
/// each data row into an [`AleRecord`].
///
/// # Errors
///
/// Returns an [`crate::error::EdlError`] if the ALE input is malformed.
pub fn parse_ale_records(input: &str) -> EdlResult<Vec<AleRecord>> {
    let doc = parse_ale(input)?;
    let records = doc
        .rows
        .into_iter()
        .map(|row| AleRecord::from_fields(row.fields))
        .collect();
    Ok(records)
}

// ────────────────────────────────────────────────────────────────────────────
// Tests
// ────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    const SAMPLE_ALE: &str = "Heading\n\
        FIELD_DELIM\tTABS\n\
        VIDEO_FORMAT\t1080i\n\
        AUDIO_FORMAT\t48kHz\n\
        FPS\t25\n\
        \n\
        Column\n\
        Name\tStart\tEnd\tTracks\tDuration\tTape\n\
        \n\
        Data\n\
        SHOT_001\t01:00:00:00\t01:00:05:00\tV\t125\tA001\n\
        SHOT_002\t01:00:05:00\t01:00:12:00\tV\t175\tA002\n";

    #[test]
    fn test_parse_ale_heading() {
        let doc = parse_ale(SAMPLE_ALE).expect("parse should succeed");
        assert_eq!(doc.video_format(), Some("1080i"));
        assert_eq!(doc.audio_format(), Some("48kHz"));
        assert_eq!(doc.fps(), Some(25.0));
    }

    #[test]
    fn test_parse_ale_columns() {
        let doc = parse_ale(SAMPLE_ALE).expect("parse should succeed");
        assert_eq!(doc.columns.len(), 6);
        assert_eq!(doc.columns[0], "Name");
        assert_eq!(doc.columns[5], "Tape");
        assert!(doc.has_column("Name"));
        assert!(!doc.has_column("Missing"));
    }

    #[test]
    fn test_parse_ale_data_rows() {
        let doc = parse_ale(SAMPLE_ALE).expect("parse should succeed");
        assert_eq!(doc.row_count(), 2);

        let row0 = &doc.rows[0];
        assert_eq!(row0.get("Name"), Some("SHOT_001"));
        assert_eq!(row0.get("Start"), Some("01:00:00:00"));
        assert_eq!(row0.get("Tape"), Some("A001"));
        assert!(row0.has("Name"));
        assert!(!row0.has("Missing"));
    }

    #[test]
    fn test_parse_ale_second_row() {
        let doc = parse_ale(SAMPLE_ALE).expect("parse should succeed");
        let row1 = &doc.rows[1];
        assert_eq!(row1.get("Name"), Some("SHOT_002"));
        assert_eq!(row1.get("Duration"), Some("175"));
        assert_eq!(row1.get("Tape"), Some("A002"));
    }

    #[test]
    fn test_ale_column_values() {
        let doc = parse_ale(SAMPLE_ALE).expect("parse should succeed");
        let names = doc.column_values("Name");
        assert_eq!(names, vec!["SHOT_001", "SHOT_002"]);
    }

    #[test]
    fn test_ale_filter_rows() {
        let doc = parse_ale(SAMPLE_ALE).expect("parse should succeed");
        let filtered = doc.filter_rows("Tape", "A001");
        assert_eq!(filtered.len(), 1);
        assert_eq!(filtered[0].get("Name"), Some("SHOT_001"));
    }

    #[test]
    fn test_ale_column_index() {
        let doc = parse_ale(SAMPLE_ALE).expect("parse should succeed");
        assert_eq!(doc.column_index("Name"), Some(0));
        assert_eq!(doc.column_index("Tape"), Some(5));
        assert_eq!(doc.column_index("Missing"), None);
    }

    #[test]
    fn test_ale_get_or_default() {
        let doc = parse_ale(SAMPLE_ALE).expect("parse should succeed");
        let row = &doc.rows[0];
        assert_eq!(row.get_or("Name", "unknown"), "SHOT_001");
        assert_eq!(row.get_or("Missing", "unknown"), "unknown");
    }

    #[test]
    fn test_ale_field_delimiter() {
        let doc = parse_ale(SAMPLE_ALE).expect("parse should succeed");
        assert_eq!(doc.field_delimiter(), "\t");
    }

    #[test]
    fn test_parse_ale_no_columns_error() {
        let bad = "Heading\nFIELD_DELIM\tTABS\n\nData\n";
        let result = parse_ale(bad);
        assert!(result.is_err());
    }

    #[test]
    fn test_parse_ale_mismatched_fields_error() {
        let bad = "Heading\nFIELD_DELIM\tTABS\n\n\
            Column\nA\tB\n\n\
            Data\nval1\tval2\tval3\n";
        let result = parse_ale(bad);
        assert!(result.is_err());
    }

    #[test]
    fn test_generate_ale_roundtrip() {
        let doc = parse_ale(SAMPLE_ALE).expect("parse should succeed");
        let generated = generate_ale(&doc).expect("generate should succeed");
        let reparsed = parse_ale(&generated).expect("reparse should succeed");

        assert_eq!(reparsed.columns, doc.columns);
        assert_eq!(reparsed.row_count(), doc.row_count());
        for (orig, re) in doc.rows.iter().zip(reparsed.rows.iter()) {
            for col in &doc.columns {
                assert_eq!(orig.get(col), re.get(col));
            }
        }
    }

    #[test]
    fn test_generate_ale_no_columns_error() {
        let doc = AleDocument::new();
        let result = generate_ale(&doc);
        assert!(result.is_err());
    }

    #[test]
    fn test_ale_sort_by_column() {
        let mut doc = parse_ale(SAMPLE_ALE).expect("parse should succeed");
        // Reverse sort by name
        doc.sort_by_column("Name");
        assert_eq!(doc.rows[0].get("Name"), Some("SHOT_001"));
        assert_eq!(doc.rows[1].get("Name"), Some("SHOT_002"));
    }

    #[test]
    fn test_ale_document_builder() {
        let mut doc = AleDocument::new();
        doc.set_heading("VIDEO_FORMAT", "720p");
        doc.set_heading("FPS", "30");
        doc.add_column("Name");
        doc.add_column("Tape");

        let mut fields = HashMap::new();
        fields.insert("Name".to_string(), "clip1".to_string());
        fields.insert("Tape".to_string(), "R001".to_string());
        doc.add_row(fields);

        assert_eq!(doc.video_format(), Some("720p"));
        assert_eq!(doc.row_count(), 1);
        assert_eq!(doc.rows[0].get("Name"), Some("clip1"));
    }

    #[test]
    fn test_parse_ale_uppercase_sections() {
        let upper = "HEADING\n\
            FIELD_DELIM\tTABS\n\
            FPS\t24\n\
            \n\
            COLUMN\n\
            Name\tTape\n\
            \n\
            DATA\n\
            clip1\tR1\n";
        let doc = parse_ale(upper).expect("parse should succeed");
        assert_eq!(doc.fps(), Some(24.0));
        assert_eq!(doc.row_count(), 1);
    }

    #[test]
    fn test_ale_empty_heading_value() {
        let input = "Heading\nSOME_KEY\n\nColumn\nA\n\nData\nval\n";
        let doc = parse_ale(input).expect("parse should succeed");
        assert_eq!(doc.heading.get("SOME_KEY"), Some(&String::new()));
    }

    /// Requirement: test_ale_parse_basic — minimal ALE string, verifies fields
    /// parse correctly via `parse_ale_records`.
    #[test]
    fn test_ale_parse_basic() {
        let ale_input = "Heading\n\
            FIELD_DELIM\tTABS\n\
            VIDEO_FORMAT\t1080p\n\
            AUDIO_FORMAT\t48kHz\n\
            FPS\t25\n\
            \n\
            Column\n\
            Name\tTape\tStart\tEnd\n\
            \n\
            Data\n\
            CLIP_A\tR001\t01:00:00:00\t01:00:10:00\n\
            CLIP_B\tR002\t02:00:00:00\t02:00:05:00\n";

        let records = parse_ale_records(ale_input).expect("parse_ale_records should succeed");
        assert_eq!(records.len(), 2, "expected 2 ALE records");

        let first = &records[0];
        assert_eq!(first.clip_name, "CLIP_A");
        assert_eq!(first.tape, "R001");
        assert_eq!(first.start, "01:00:00:00");
        assert_eq!(first.end, "01:00:10:00");

        let second = &records[1];
        assert_eq!(second.clip_name, "CLIP_B");
        assert_eq!(second.tape, "R002");
        assert_eq!(second.start, "02:00:00:00");
        assert_eq!(second.end, "02:00:05:00");

        // fields map must also contain the named columns
        assert_eq!(first.fields.get("Name").map(|s| s.as_str()), Some("CLIP_A"));
        assert_eq!(second.fields.get("Tape").map(|s| s.as_str()), Some("R002"));
    }
}
