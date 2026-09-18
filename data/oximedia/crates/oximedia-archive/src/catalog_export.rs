//! Archive catalog export utilities (CSV / JSON manifest generation).

#![allow(dead_code)]

// ---------------------------------------------------------------------------
// ExportFormat
// ---------------------------------------------------------------------------

/// Output format for catalog exports.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ExportFormat {
    /// Comma-separated values.
    Csv,
    /// Newline-delimited JSON (one JSON object per line).
    NdJson,
    /// Tab-separated values.
    Tsv,
}

impl ExportFormat {
    /// File extension (without leading dot) conventionally used for this format.
    #[must_use]
    pub fn file_extension(&self) -> &str {
        match self {
            ExportFormat::Csv => "csv",
            ExportFormat::NdJson => "ndjson",
            ExportFormat::Tsv => "tsv",
        }
    }

    /// MIME type for this format.
    #[must_use]
    pub fn mime_type(&self) -> &str {
        match self {
            ExportFormat::Csv => "text/csv",
            ExportFormat::NdJson => "application/x-ndjson",
            ExportFormat::Tsv => "text/tab-separated-values",
        }
    }
}

// ---------------------------------------------------------------------------
// CatalogRecord
// ---------------------------------------------------------------------------

/// A single row in the catalog export.
#[derive(Debug, Clone, PartialEq)]
pub struct CatalogRecord {
    /// Unique asset identifier.
    pub asset_id: String,
    /// Original filename.
    pub filename: String,
    /// File size in bytes.
    pub size_bytes: u64,
    /// BLAKE3 or SHA-256 content hash (hex).
    pub hash: String,
    /// Format / container (e.g. "MXF", "MOV").
    pub format: String,
    /// Collection or folder label.
    pub collection: String,
}

impl CatalogRecord {
    /// Create a new catalog record.
    #[must_use]
    pub fn new(
        asset_id: impl Into<String>,
        filename: impl Into<String>,
        size_bytes: u64,
        hash: impl Into<String>,
        format: impl Into<String>,
        collection: impl Into<String>,
    ) -> Self {
        Self {
            asset_id: asset_id.into(),
            filename: filename.into(),
            size_bytes,
            hash: hash.into(),
            format: format.into(),
            collection: collection.into(),
        }
    }

    /// Serialise this record to a CSV row string (no trailing newline).
    ///
    /// Fields containing commas are enclosed in double-quotes.
    #[must_use]
    pub fn to_csv_row(&self) -> String {
        let fields = [
            &self.asset_id,
            &self.filename,
            &self.size_bytes.to_string(),
            &self.hash,
            &self.format,
            &self.collection,
        ];
        fields
            .iter()
            .map(|f| {
                if f.contains(',') || f.contains('"') {
                    format!("\"{}\"", f.replace('"', "\"\""))
                } else {
                    (*f).clone()
                }
            })
            .collect::<Vec<_>>()
            .join(",")
    }

    /// Serialise this record to a TSV row string (no trailing newline).
    #[must_use]
    pub fn to_tsv_row(&self) -> String {
        format!(
            "{}\t{}\t{}\t{}\t{}\t{}",
            self.asset_id, self.filename, self.size_bytes, self.hash, self.format, self.collection
        )
    }
}

// ---------------------------------------------------------------------------
// CatalogExporter
// ---------------------------------------------------------------------------

/// Collects catalog records and renders them to the chosen format.
#[derive(Debug, Default)]
pub struct CatalogExporter {
    records: Vec<CatalogRecord>,
}

impl CatalogExporter {
    /// Create a new, empty exporter.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Add a record to the export.
    pub fn add(&mut self, record: CatalogRecord) {
        self.records.push(record);
    }

    /// Number of records currently held.
    #[must_use]
    pub fn record_count(&self) -> usize {
        self.records.len()
    }

    /// Total size in bytes across all records.
    #[must_use]
    pub fn total_size_bytes(&self) -> u64 {
        self.records.iter().map(|r| r.size_bytes).sum()
    }

    /// Render the catalog to a `String` in the given format.
    #[must_use]
    pub fn render(&self, format: ExportFormat) -> String {
        match format {
            ExportFormat::Csv => {
                let mut out = "asset_id,filename,size_bytes,hash,format,collection\n".to_string();
                for r in &self.records {
                    out.push_str(&r.to_csv_row());
                    out.push('\n');
                }
                out
            }
            ExportFormat::Tsv => {
                let mut out =
                    "asset_id\tfilename\tsize_bytes\thash\tformat\tcollection\n".to_string();
                for r in &self.records {
                    out.push_str(&r.to_tsv_row());
                    out.push('\n');
                }
                out
            }
            ExportFormat::NdJson => {
                let mut out = String::new();
                for r in &self.records {
                    out.push_str(&format!(
                        "{{\"asset_id\":\"{}\",\"filename\":\"{}\",\"size_bytes\":{},\"hash\":\"{}\",\"format\":\"{}\",\"collection\":\"{}\"}}\n",
                        r.asset_id, r.filename, r.size_bytes, r.hash, r.format, r.collection
                    ));
                }
                out
            }
        }
    }
}

// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_record() -> CatalogRecord {
        CatalogRecord::new(
            "id-001",
            "clip.mxf",
            1_048_576,
            "deadbeef",
            "MXF",
            "Project-A",
        )
    }

    // --- ExportFormat ---

    #[test]
    fn test_csv_extension() {
        assert_eq!(ExportFormat::Csv.file_extension(), "csv");
    }

    #[test]
    fn test_ndjson_extension() {
        assert_eq!(ExportFormat::NdJson.file_extension(), "ndjson");
    }

    #[test]
    fn test_tsv_extension() {
        assert_eq!(ExportFormat::Tsv.file_extension(), "tsv");
    }

    #[test]
    fn test_csv_mime() {
        assert_eq!(ExportFormat::Csv.mime_type(), "text/csv");
    }

    #[test]
    fn test_ndjson_mime() {
        assert_eq!(ExportFormat::NdJson.mime_type(), "application/x-ndjson");
    }

    // --- CatalogRecord ---

    #[test]
    fn test_csv_row_no_special_chars() {
        let r = sample_record();
        let row = r.to_csv_row();
        assert!(row.contains("id-001"));
        assert!(row.contains("clip.mxf"));
        assert!(row.contains("1048576"));
    }

    #[test]
    fn test_csv_row_with_comma_in_field() {
        let r = CatalogRecord::new("id-002", "clip, take2.mxf", 0, "abc", "MXF", "Proj");
        let row = r.to_csv_row();
        // Filename contains comma so should be quoted
        assert!(row.contains("\"clip, take2.mxf\""));
    }

    #[test]
    fn test_tsv_row_contains_tab() {
        let r = sample_record();
        let row = r.to_tsv_row();
        assert!(row.contains('\t'));
        assert!(row.contains("id-001"));
    }

    #[test]
    fn test_tsv_row_field_count() {
        let r = sample_record();
        let row = r.to_tsv_row();
        assert_eq!(row.split('\t').count(), 6);
    }

    // --- CatalogExporter ---

    #[test]
    fn test_exporter_empty_count() {
        let exp = CatalogExporter::new();
        assert_eq!(exp.record_count(), 0);
    }

    #[test]
    fn test_exporter_add_increments_count() {
        let mut exp = CatalogExporter::new();
        exp.add(sample_record());
        assert_eq!(exp.record_count(), 1);
    }

    #[test]
    fn test_exporter_total_size() {
        let mut exp = CatalogExporter::new();
        exp.add(CatalogRecord::new("a", "f.mxf", 1000, "h", "MXF", "C"));
        exp.add(CatalogRecord::new("b", "g.mov", 2000, "h2", "MOV", "C"));
        assert_eq!(exp.total_size_bytes(), 3000);
    }

    #[test]
    fn test_render_csv_has_header() {
        let exp = CatalogExporter::new();
        let out = exp.render(ExportFormat::Csv);
        assert!(out.starts_with("asset_id,filename"));
    }

    #[test]
    fn test_render_csv_has_data_row() {
        let mut exp = CatalogExporter::new();
        exp.add(sample_record());
        let out = exp.render(ExportFormat::Csv);
        assert!(out.contains("id-001"));
    }

    #[test]
    fn test_render_ndjson_valid_json_lines() {
        let mut exp = CatalogExporter::new();
        exp.add(sample_record());
        let out = exp.render(ExportFormat::NdJson);
        // Each non-empty line should start with '{' and end with '}'
        for line in out.lines().filter(|l| !l.is_empty()) {
            assert!(line.starts_with('{'), "line: {}", line);
            assert!(line.ends_with('}'), "line: {}", line);
        }
    }

    #[test]
    fn test_render_tsv_has_header() {
        let exp = CatalogExporter::new();
        let out = exp.render(ExportFormat::Tsv);
        assert!(out.starts_with("asset_id\t"));
    }
}
