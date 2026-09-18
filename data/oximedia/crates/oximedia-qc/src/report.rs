//! Quality control report generation.
//!
//! This module provides the [`QcReport`] type and utilities for generating
//! detailed validation reports in various formats (JSON, XML, plain text).

use crate::rules::{CheckResult, RuleCategory, Severity};
use std::collections::HashMap;
use std::fmt;

/// Quality control report.
///
/// Contains the results of all QC checks performed on a file,
/// along with summary statistics and metadata.
#[derive(Clone, Debug)]
#[cfg_attr(feature = "json", derive(serde::Serialize, serde::Deserialize))]
pub struct QcReport {
    /// Path to the file that was validated.
    pub file_path: String,

    /// Total number of checks performed.
    pub total_checks: usize,

    /// Number of checks that passed.
    pub passed_checks: usize,

    /// Number of checks that failed.
    pub failed_checks: usize,

    /// Overall validation result.
    pub overall_passed: bool,

    /// All check results.
    pub results: Vec<CheckResult>,

    /// Timestamp when the report was generated.
    pub timestamp: String,

    /// Duration of the validation process in seconds.
    pub validation_duration: Option<f64>,
}

impl QcReport {
    /// Creates a new QC report.
    #[must_use]
    pub fn new(file_path: impl Into<String>) -> Self {
        Self {
            file_path: file_path.into(),
            total_checks: 0,
            passed_checks: 0,
            failed_checks: 0,
            overall_passed: true,
            results: Vec::new(),
            timestamp: chrono::Utc::now().to_rfc3339(),
            validation_duration: None,
        }
    }

    /// Adds a check result to the report.
    pub fn add_result(&mut self, result: CheckResult) {
        self.total_checks += 1;
        if result.passed {
            self.passed_checks += 1;
        } else {
            self.failed_checks += 1;
            // If any check fails with Error or Critical severity, mark overall as failed
            if result.severity >= Severity::Error {
                self.overall_passed = false;
            }
        }
        self.results.push(result);
    }

    /// Adds multiple check results to the report.
    pub fn add_results(&mut self, results: Vec<CheckResult>) {
        for result in results {
            self.add_result(result);
        }
    }

    /// Sets the validation duration.
    pub fn set_validation_duration(&mut self, duration: f64) {
        self.validation_duration = Some(duration);
    }

    /// Returns results filtered by severity.
    #[must_use]
    pub fn results_by_severity(&self, severity: Severity) -> Vec<&CheckResult> {
        self.results
            .iter()
            .filter(|r| !r.passed && r.severity == severity)
            .collect()
    }

    /// Returns results filtered by category.
    #[must_use]
    pub fn results_by_category(&self, _category: RuleCategory) -> Vec<&CheckResult> {
        // Note: We would need to store category in CheckResult to implement this fully
        self.results.iter().filter(|r| !r.passed).collect()
    }

    /// Returns critical errors only.
    #[must_use]
    pub fn critical_errors(&self) -> Vec<&CheckResult> {
        self.results_by_severity(Severity::Critical)
    }

    /// Returns errors only.
    #[must_use]
    pub fn errors(&self) -> Vec<&CheckResult> {
        self.results_by_severity(Severity::Error)
    }

    /// Returns warnings only.
    #[must_use]
    pub fn warnings(&self) -> Vec<&CheckResult> {
        self.results_by_severity(Severity::Warning)
    }

    /// Returns info messages only.
    #[must_use]
    pub fn info_messages(&self) -> Vec<&CheckResult> {
        self.results_by_severity(Severity::Info)
    }

    /// Generates a plain text summary of the report.
    #[must_use]
    pub fn summary(&self) -> String {
        let mut summary = String::new();
        summary.push_str(&format!("QC Report for: {}\n", self.file_path));
        summary.push_str(&format!("Generated: {}\n", self.timestamp));
        summary.push_str(&format!(
            "Overall Status: {}\n",
            if self.overall_passed { "PASS" } else { "FAIL" }
        ));
        summary.push_str(&format!(
            "Checks: {} total, {} passed, {} failed\n",
            self.total_checks, self.passed_checks, self.failed_checks
        ));

        if let Some(duration) = self.validation_duration {
            summary.push_str(&format!("Validation Duration: {duration:.2}s\n"));
        }

        summary.push('\n');

        let critical = self.critical_errors();
        if !critical.is_empty() {
            summary.push_str(&format!("Critical Errors: {}\n", critical.len()));
        }

        let errors = self.errors();
        if !errors.is_empty() {
            summary.push_str(&format!("Errors: {}\n", errors.len()));
        }

        let warnings = self.warnings();
        if !warnings.is_empty() {
            summary.push_str(&format!("Warnings: {}\n", warnings.len()));
        }

        summary
    }

    /// Exports the report as JSON.
    ///
    /// # Errors
    ///
    /// Returns an error if serialization fails.
    #[cfg(feature = "json")]
    pub fn to_json(&self) -> Result<String, serde_json::Error> {
        serde_json::to_string_pretty(self)
    }

    /// Exports the report as compact JSON.
    ///
    /// # Errors
    ///
    /// Returns an error if serialization fails.
    #[cfg(feature = "json")]
    pub fn to_json_compact(&self) -> Result<String, serde_json::Error> {
        serde_json::to_string(self)
    }

    /// Exports the report as XML.
    ///
    /// # Errors
    ///
    /// Returns an error if serialization fails.
    #[cfg(feature = "xml")]
    pub fn to_xml(&self) -> Result<String, quick_xml::Error> {
        let mut buffer = Vec::new();
        let mut writer = quick_xml::Writer::new(&mut buffer);

        // Write XML header
        writer.write_event(quick_xml::events::Event::Decl(
            quick_xml::events::BytesDecl::new("1.0", Some("UTF-8"), None),
        ))?;

        // Write root element
        writer.write_event(quick_xml::events::Event::Start(
            quick_xml::events::BytesStart::new("qc_report"),
        ))?;

        // Write basic info
        self.write_xml_element(&mut writer, "file_path", &self.file_path)?;
        self.write_xml_element(&mut writer, "timestamp", &self.timestamp)?;
        self.write_xml_element(
            &mut writer,
            "overall_passed",
            &self.overall_passed.to_string(),
        )?;
        self.write_xml_element(&mut writer, "total_checks", &self.total_checks.to_string())?;
        self.write_xml_element(
            &mut writer,
            "passed_checks",
            &self.passed_checks.to_string(),
        )?;
        self.write_xml_element(
            &mut writer,
            "failed_checks",
            &self.failed_checks.to_string(),
        )?;

        // Write results
        writer.write_event(quick_xml::events::Event::Start(
            quick_xml::events::BytesStart::new("results"),
        ))?;

        for result in &self.results {
            writer.write_event(quick_xml::events::Event::Start(
                quick_xml::events::BytesStart::new("check"),
            ))?;

            self.write_xml_element(&mut writer, "rule_name", &result.rule_name)?;
            self.write_xml_element(&mut writer, "passed", &result.passed.to_string())?;
            self.write_xml_element(&mut writer, "severity", &result.severity.to_string())?;
            self.write_xml_element(&mut writer, "message", &result.message)?;

            if let Some(rec) = &result.recommendation {
                self.write_xml_element(&mut writer, "recommendation", rec)?;
            }

            writer.write_event(quick_xml::events::Event::End(
                quick_xml::events::BytesEnd::new("check"),
            ))?;
        }

        writer.write_event(quick_xml::events::Event::End(
            quick_xml::events::BytesEnd::new("results"),
        ))?;

        // Close root element
        writer.write_event(quick_xml::events::Event::End(
            quick_xml::events::BytesEnd::new("qc_report"),
        ))?;

        // Convert buffer to string
        // This should not fail since we only write valid UTF-8 to the buffer
        Ok(String::from_utf8(buffer).expect("buffer should contain valid UTF-8"))
    }

    #[cfg(feature = "xml")]
    fn write_xml_element(
        &self,
        writer: &mut quick_xml::Writer<&mut Vec<u8>>,
        name: &str,
        value: &str,
    ) -> Result<(), quick_xml::Error> {
        writer.write_event(quick_xml::events::Event::Start(
            quick_xml::events::BytesStart::new(name),
        ))?;
        writer.write_event(quick_xml::events::Event::Text(
            quick_xml::events::BytesText::new(value),
        ))?;
        writer.write_event(quick_xml::events::Event::End(
            quick_xml::events::BytesEnd::new(name),
        ))?;
        Ok(())
    }

    /// Generates a detailed text report.
    #[must_use]
    pub fn to_text(&self) -> String {
        let mut text = self.summary();
        text.push_str("\nDetailed Results:\n");
        text.push_str("=================\n\n");

        // Group results by severity
        let mut by_severity: HashMap<Severity, Vec<&CheckResult>> = HashMap::new();
        for result in &self.results {
            if !result.passed {
                by_severity.entry(result.severity).or_default().push(result);
            }
        }

        // Display in order of severity
        for severity in &[
            Severity::Critical,
            Severity::Error,
            Severity::Warning,
            Severity::Info,
        ] {
            if let Some(results) = by_severity.get(severity) {
                text.push_str(&format!("{severity} ({}):\n", results.len()));
                for result in results {
                    text.push_str(&format!("  [{}] {}\n", result.rule_name, result.message));
                    if let Some(stream_index) = result.stream_index {
                        text.push_str(&format!("    Stream: {stream_index}\n"));
                    }
                    if let Some(timestamp) = result.timestamp {
                        text.push_str(&format!("    Timestamp: {timestamp:.2}s\n"));
                    }
                    if let Some(rec) = &result.recommendation {
                        text.push_str(&format!("    Recommendation: {rec}\n"));
                    }
                }
                text.push('\n');
            }
        }

        text
    }
}

impl fmt::Display for QcReport {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.to_text())
    }
}

/// Exports the QC report as a self-contained HTML document.
///
/// The generated page includes:
/// - A header with the file path, timestamp, and overall pass/fail badge.
/// - A summary table with check counts broken down by severity.
/// - A detailed findings table showing each failed check with severity,
///   rule name, message, and optional recommendation.
/// - A pass-all notice when no failures were detected.
///
/// All styling is embedded inline; no external resources are required.
///
/// # Errors
///
/// Currently infallible (`Ok` is always returned); the signature uses
/// `Result<String, HtmlExportError>` for forward compatibility.
pub fn report_to_html(report: &QcReport) -> Result<String, HtmlExportError> {
    let status_color = if report.overall_passed {
        "#2e7d32"
    } else {
        "#c62828"
    };
    let status_text = if report.overall_passed {
        "PASS"
    } else {
        "FAIL"
    };

    let critical_count = report.critical_errors().len();
    let error_count = report.errors().len();
    let warning_count = report.warnings().len();
    let info_count = report.info_messages().len();

    // ── Findings rows ────────────────────────────────────────────────────────
    let mut rows = String::new();
    for result in &report.results {
        if result.passed {
            continue;
        }
        let (sev_color, sev_bg) = match result.severity {
            Severity::Critical => ("#b71c1c", "#ffebee"),
            Severity::Error => ("#e65100", "#fff3e0"),
            Severity::Warning => ("#f57f17", "#fffde7"),
            Severity::Info => ("#1565c0", "#e3f2fd"),
        };

        let stream_cell = result
            .stream_index
            .map(|s| s.to_string())
            .unwrap_or_default();
        let ts_cell = result
            .timestamp
            .map(|t| format!("{t:.2}s"))
            .unwrap_or_default();
        let rec_cell = result.recommendation.as_deref().unwrap_or("").to_string();

        rows.push_str(&format!(
            "<tr style=\"background:{sev_bg}\">\
             <td style=\"color:{sev_color};font-weight:bold;padding:6px 8px\">{}</td>\
             <td style=\"padding:6px 8px;font-family:monospace\">{}</td>\
             <td style=\"padding:6px 8px\">{}</td>\
             <td style=\"padding:6px 8px;color:#555\">{stream_cell}</td>\
             <td style=\"padding:6px 8px;color:#555\">{ts_cell}</td>\
             <td style=\"padding:6px 8px;color:#37474f;font-style:italic\">{rec_cell}</td>\
             </tr>",
            html_escape(&result.severity.to_string()),
            html_escape(&result.rule_name),
            html_escape(&result.message),
        ));
    }

    let findings_section = if report.failed_checks == 0 {
        "<p style=\"color:#2e7d32;font-weight:bold\">✓ All checks passed.</p>".to_string()
    } else {
        format!(
            "<table style=\"width:100%;border-collapse:collapse;font-size:0.9em\">\
             <thead><tr style=\"background:#eceff1\">\
             <th style=\"padding:8px;text-align:left\">Severity</th>\
             <th style=\"padding:8px;text-align:left\">Rule</th>\
             <th style=\"padding:8px;text-align:left\">Message</th>\
             <th style=\"padding:8px;text-align:left\">Stream</th>\
             <th style=\"padding:8px;text-align:left\">Timestamp</th>\
             <th style=\"padding:8px;text-align:left\">Recommendation</th>\
             </tr></thead>\
             <tbody>{rows}</tbody>\
             </table>"
        )
    };

    let duration_row = report
        .validation_duration
        .map(|d| {
            format!(
                "<tr><td style=\"padding:4px 8px;color:#555\">Validation Duration</td>\
             <td style=\"padding:4px 8px\">{d:.3}s</td></tr>"
            )
        })
        .unwrap_or_default();

    let html = format!(
        "<!DOCTYPE html>\n\
         <html lang=\"en\">\n\
         <head>\n\
         <meta charset=\"UTF-8\">\n\
         <meta name=\"viewport\" content=\"width=device-width,initial-scale=1\">\n\
         <title>QC Report — {file}</title>\n\
         <style>\
         body{{font-family:system-ui,sans-serif;margin:0;padding:24px;background:#fafafa;color:#212121}}\
         h1{{font-size:1.4em;margin:0 0 4px}}\
         .badge{{display:inline-block;padding:4px 14px;border-radius:4px;\
                 color:#fff;font-weight:bold;font-size:1em;background:{status_color}}}\
         table{{border-collapse:collapse}}td,th{{border-bottom:1px solid #e0e0e0}}\
         </style>\n\
         </head>\n\
         <body>\n\
         <h1>QC Report</h1>\n\
         <p><strong>File:</strong> {file}</p>\n\
         <p><strong>Generated:</strong> {ts}</p>\n\
         <p><strong>Status:</strong> <span class=\"badge\">{status_text}</span></p>\n\
         <h2>Summary</h2>\n\
         <table>\n\
         <tr><td style=\"padding:4px 8px;color:#555\">Total Checks</td>\
             <td style=\"padding:4px 8px\">{total}</td></tr>\n\
         <tr><td style=\"padding:4px 8px;color:#2e7d32\">Passed</td>\
             <td style=\"padding:4px 8px\">{passed}</td></tr>\n\
         <tr><td style=\"padding:4px 8px;color:#b71c1c\">Critical</td>\
             <td style=\"padding:4px 8px\">{critical}</td></tr>\n\
         <tr><td style=\"padding:4px 8px;color:#e65100\">Errors</td>\
             <td style=\"padding:4px 8px\">{errors}</td></tr>\n\
         <tr><td style=\"padding:4px 8px;color:#f57f17\">Warnings</td>\
             <td style=\"padding:4px 8px\">{warnings}</td></tr>\n\
         <tr><td style=\"padding:4px 8px;color:#1565c0\">Info</td>\
             <td style=\"padding:4px 8px\">{info}</td></tr>\n\
         {duration_row}\
         </table>\n\
         <h2>Findings</h2>\n\
         {findings_section}\n\
         </body>\n\
         </html>\n",
        file = html_escape(&report.file_path),
        ts = html_escape(&report.timestamp),
        total = report.total_checks,
        passed = report.passed_checks,
        critical = critical_count,
        errors = error_count,
        warnings = warning_count,
        info = info_count,
    );

    Ok(html)
}

/// Error type for HTML export operations.
#[derive(Debug)]
pub enum HtmlExportError {
    /// IO error during write.
    Io(std::io::Error),
}

impl std::fmt::Display for HtmlExportError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Io(e) => write!(f, "HTML export IO error: {e}"),
        }
    }
}

impl std::error::Error for HtmlExportError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Io(e) => Some(e),
        }
    }
}

/// Escapes special HTML characters to prevent injection.
fn html_escape(s: &str) -> String {
    s.chars()
        .flat_map(|c| match c {
            '&' => "&amp;".chars().collect::<Vec<_>>(),
            '<' => "&lt;".chars().collect(),
            '>' => "&gt;".chars().collect(),
            '"' => "&quot;".chars().collect(),
            '\'' => "&#39;".chars().collect(),
            other => vec![other],
        })
        .collect()
}

impl QcReport {
    /// Exports the report as a self-contained HTML document.
    ///
    /// # Errors
    ///
    /// Returns an [`HtmlExportError`] if HTML generation fails (currently infallible).
    pub fn to_html(&self) -> Result<String, HtmlExportError> {
        report_to_html(self)
    }
}

/// Report format for export.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ReportFormat {
    /// Plain text format.
    Text,
    /// JSON format.
    Json,
    /// Compact JSON format (no pretty printing).
    JsonCompact,
    /// XML format.
    Xml,
    /// Self-contained HTML document.
    Html,
    /// PDF 1.4 document (feature-gated).
    #[cfg(feature = "pdf")]
    Pdf,
}

impl ReportFormat {
    /// Returns the file extension for this format.
    #[must_use]
    pub const fn extension(&self) -> &'static str {
        match self {
            Self::Text => "txt",
            Self::Json | Self::JsonCompact => "json",
            Self::Xml => "xml",
            Self::Html => "html",
            #[cfg(feature = "pdf")]
            Self::Pdf => "pdf",
        }
    }
}

// ── PDF export ───────────────────────────────────────────────────────────────

/// Error type for PDF export operations.
#[cfg(feature = "pdf")]
#[derive(Debug)]
pub enum PdfExportError {
    /// Formatting error during write.
    Io(std::fmt::Error),
}

#[cfg(feature = "pdf")]
impl std::fmt::Display for PdfExportError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Io(e) => write!(f, "PDF export error: {e}"),
        }
    }
}

#[cfg(feature = "pdf")]
impl std::error::Error for PdfExportError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Io(e) => Some(e),
        }
    }
}

#[cfg(feature = "pdf")]
impl From<std::fmt::Error> for PdfExportError {
    fn from(e: std::fmt::Error) -> Self {
        Self::Io(e)
    }
}

/// Escape special PDF string characters: `(`, `)`, and `\`.
#[cfg(feature = "pdf")]
fn pdf_escape(s: &str) -> String {
    let mut out = String::with_capacity(s.len() + 4);
    for ch in s.chars() {
        match ch {
            '(' => out.push_str("\\("),
            ')' => out.push_str("\\)"),
            '\\' => out.push_str("\\\\"),
            // Replace non-ASCII/control chars with '?' for Type1 safety
            c if c.is_ascii() && !c.is_control() => out.push(c),
            _ => out.push('?'),
        }
    }
    out
}

/// A minimal hand-emitted PDF 1.4 builder.
#[cfg(feature = "pdf")]
struct PdfBuilder {
    /// Accumulated bytes of the PDF being built.
    buf: Vec<u8>,
    /// Byte offsets of each object (1-indexed; entry 0 unused).
    offsets: Vec<usize>,
    /// Raw object payloads (stream dicts + streams) indexed by obj number (1-based).
    objects: Vec<String>,
}

#[cfg(feature = "pdf")]
impl PdfBuilder {
    fn new() -> Self {
        Self {
            buf: Vec::with_capacity(65536),
            offsets: Vec::new(),
            objects: Vec::new(),
        }
    }

    /// Reserve a slot for an object and return its 1-based object number.
    fn reserve(&mut self) -> usize {
        self.objects.push(String::new());
        self.objects.len()
    }

    /// Assign content to a previously reserved object slot.
    fn set_object(&mut self, obj_num: usize, content: String) {
        self.objects[obj_num - 1] = content;
    }

    /// Emit all objects into `buf`, recording their byte offsets.
    fn emit_objects(&mut self) {
        // PDF header
        self.buf.extend_from_slice(b"%PDF-1.4\n");
        // Binary comment to mark the file as binary
        self.buf.extend_from_slice(b"%\xc2\xb5\xc2\xb5\n");

        self.offsets = vec![0usize; self.objects.len() + 1]; // index 0 unused

        for (idx, obj_body) in self.objects.iter().enumerate() {
            let obj_num = idx + 1;
            self.offsets[obj_num] = self.buf.len();
            let header = format!("{obj_num} 0 obj\n");
            self.buf.extend_from_slice(header.as_bytes());
            self.buf.extend_from_slice(obj_body.as_bytes());
            self.buf.extend_from_slice(b"\nendobj\n");
        }
    }

    /// Append xref table and trailer, return the final PDF bytes.
    fn finalize(mut self, root_obj: usize) -> Vec<u8> {
        let xref_offset = self.buf.len();
        let n = self.objects.len() + 1; // +1 for object 0

        self.buf.extend_from_slice(b"xref\n");
        self.buf.extend_from_slice(format!("0 {n}\n").as_bytes());
        // Object 0 — free head
        self.buf.extend_from_slice(b"0000000000 65535 f \n");
        // One entry per object
        for obj_num in 1..n {
            let off = self.offsets[obj_num];
            let entry = format!("{off:010} 00000 n \n");
            self.buf.extend_from_slice(entry.as_bytes());
        }

        // Trailer
        let trailer = format!(
            "trailer\n<< /Size {n} /Root {root_obj} 0 R >>\nstartxref\n{xref_offset}\n%%EOF\n"
        );
        self.buf.extend_from_slice(trailer.as_bytes());
        self.buf
    }
}

/// Layout constants for the PDF page.
#[cfg(feature = "pdf")]
mod pdf_layout {
    pub const PAGE_WIDTH: f32 = 595.0;
    pub const PAGE_HEIGHT: f32 = 842.0;
    pub const MARGIN_LEFT: f32 = 50.0;
    pub const MARGIN_BOTTOM: f32 = 50.0;
    pub const TOP_Y: f32 = PAGE_HEIGHT - 50.0; // 792
    pub const LINE_HEIGHT: f32 = 18.0;
    pub const FONT_SIZE: f32 = 11.0;
    pub const TITLE_FONT_SIZE: f32 = 14.0;
}

/// Wraps long text to fit within `max_chars` wide columns, returning lines.
#[cfg(feature = "pdf")]
fn wrap_text(s: &str, max_chars: usize) -> Vec<String> {
    if s.len() <= max_chars {
        return vec![s.to_string()];
    }
    let mut lines = Vec::new();
    let mut remaining = s;
    while remaining.len() > max_chars {
        // Try to break at a space
        let slice = &remaining[..max_chars];
        let break_at = slice.rfind(' ').unwrap_or(max_chars);
        lines.push(remaining[..break_at].to_string());
        remaining = remaining[break_at..].trim_start();
    }
    if !remaining.is_empty() {
        lines.push(remaining.to_string());
    }
    lines
}

/// Generates a PDF 1.4 content stream for one page.
///
/// Returns the raw stream bytes (not yet wrapped in a stream object).
#[cfg(feature = "pdf")]
fn build_page_content(text_lines: &[String]) -> String {
    use pdf_layout::*;
    use std::fmt::Write as FmtWrite;

    let mut s = String::with_capacity(text_lines.len() * 40 + 256);
    // Begin text block
    let _ = write!(
        s,
        "BT\n/F1 {TITLE_FONT_SIZE} Tf\n{MARGIN_LEFT} {TOP_Y} Td\n"
    );

    let mut current_font_size = TITLE_FONT_SIZE;
    let mut first = true;

    for line in text_lines {
        // Switch to body font after the title line
        if first {
            // Title line: already at TITLE_FONT_SIZE, just emit it
            let escaped = pdf_escape(line);
            let _ = writeln!(s, "({escaped}) Tj");
            // Switch font size for subsequent lines
            let _ = writeln!(s, "/F1 {FONT_SIZE} Tf");
            let _ = writeln!(s, "0 -{LINE_HEIGHT} Td");
            current_font_size = FONT_SIZE;
            first = false;
        } else {
            let _ = current_font_size; // suppress unused warning
            let escaped = pdf_escape(line);
            let _ = writeln!(s, "({escaped}) Tj");
            let _ = writeln!(s, "0 -{LINE_HEIGHT} Td");
        }
    }

    s.push_str("ET\n");
    s
}

/// Exports `report` as a minimal PDF 1.4 document.
///
/// The PDF contains one or more A4 pages with:
/// - A title line: "QC Report — PASS/FAIL"
/// - A summary line with check counts
/// - Each `CheckResult` as `[PASS/FAIL] rule_name: message`
///
/// # Errors
///
/// Returns a [`PdfExportError`] on formatting failure (currently infallible).
#[cfg(feature = "pdf")]
pub fn report_to_pdf(report: &QcReport) -> Result<Vec<u8>, PdfExportError> {
    report.to_pdf()
}

#[cfg(feature = "pdf")]
impl QcReport {
    /// Exports the report as a minimal PDF 1.4 document.
    ///
    /// # Errors
    ///
    /// Returns a [`PdfExportError`] on formatting failure.
    pub fn to_pdf(&self) -> Result<Vec<u8>, PdfExportError> {
        use pdf_layout::*;
        use std::fmt::Write as FmtWrite;

        // ── Build the logical lines to lay out ──────────────────────────────
        let status = if self.overall_passed { "PASS" } else { "FAIL" };
        let mut all_lines: Vec<String> = Vec::new();
        all_lines.push(format!("QC Report - {status}"));
        all_lines.push(format!("File: {}", self.file_path));
        all_lines.push(format!("Generated: {}", self.timestamp));
        all_lines.push(format!(
            "Checks: {} total, {} passed, {} failed",
            self.total_checks, self.passed_checks, self.failed_checks
        ));
        all_lines.push(String::new()); // blank separator

        for result in &self.results {
            let pass_str = if result.passed { "PASS" } else { "FAIL" };
            let raw = format!("[{pass_str}] {}: {}", result.rule_name, result.message);
            // Wrap long lines at ~90 chars (roughly fits A4 at 11pt Helvetica)
            for wrapped in wrap_text(&raw, 90) {
                all_lines.push(wrapped);
            }
            if let Some(rec) = &result.recommendation {
                let rec_raw = format!("  -> {rec}");
                for wrapped in wrap_text(&rec_raw, 88) {
                    all_lines.push(wrapped);
                }
            }
        }

        // ── Paginate ────────────────────────────────────────────────────────
        // How many lines fit per page (first page has title so uses same count)
        let lines_per_page = ((TOP_Y - MARGIN_BOTTOM) / LINE_HEIGHT) as usize;
        let lines_per_page = lines_per_page.max(1);

        // Split all_lines into pages
        let mut pages: Vec<Vec<String>> = Vec::new();
        let mut remaining = all_lines.as_slice();
        while !remaining.is_empty() {
            let take = remaining.len().min(lines_per_page);
            pages.push(remaining[..take].to_vec());
            remaining = &remaining[take..];
        }
        if pages.is_empty() {
            pages.push(vec!["QC Report".to_string()]);
        }

        let num_pages = pages.len();

        // ── Allocate object slots ────────────────────────────────────────────
        // obj 1: Catalog
        // obj 2: Pages
        // obj 3..3+num_pages-1: Page objects
        // obj 3+num_pages..end: Content stream objects (one per page)
        let mut builder = PdfBuilder::new();
        let catalog_obj = builder.reserve(); // 1
        let pages_obj = builder.reserve(); // 2
        let first_page_obj = 3usize;

        // Reserve Page objects
        let mut page_obj_nums: Vec<usize> = Vec::with_capacity(num_pages);
        for _ in 0..num_pages {
            page_obj_nums.push(builder.reserve());
        }

        // Reserve Content stream objects
        let mut content_obj_nums: Vec<usize> = Vec::with_capacity(num_pages);
        for _ in 0..num_pages {
            content_obj_nums.push(builder.reserve());
        }

        let _ = first_page_obj; // suppress unused warning

        // ── Catalog ──────────────────────────────────────────────────────────
        builder.set_object(
            catalog_obj,
            format!("<< /Type /Catalog /Pages {pages_obj} 0 R >>"),
        );

        // ── Pages ────────────────────────────────────────────────────────────
        let kids: String = page_obj_nums
            .iter()
            .map(|n| format!("{n} 0 R"))
            .collect::<Vec<_>>()
            .join(" ");
        builder.set_object(
            pages_obj,
            format!(
                "<< /Type /Pages /Kids [{kids}] /Count {num_pages} /MediaBox [0 0 {PAGE_WIDTH} {PAGE_HEIGHT}] >>"
            ),
        );

        // ── Page objects & content streams ───────────────────────────────────
        let font_dict = "/Font << /F1 << /Type /Font /Subtype /Type1 /BaseFont /Helvetica >> >>";
        for (i, page_lines) in pages.iter().enumerate() {
            let page_obj = page_obj_nums[i];
            let content_obj = content_obj_nums[i];

            // Page dictionary
            builder.set_object(
                page_obj,
                format!(
                    "<< /Type /Page /Parent {pages_obj} 0 R \
                     /MediaBox [0 0 {PAGE_WIDTH} {PAGE_HEIGHT}] \
                     /Contents {content_obj} 0 R \
                     /Resources << {font_dict} >> >>",
                ),
            );

            // Content stream
            let stream_content = build_page_content(page_lines);
            let stream_len = stream_content.len();
            let mut stream_obj = String::new();
            let _ = write!(
                stream_obj,
                "<< /Length {stream_len} >>\nstream\n{stream_content}endstream"
            );
            builder.set_object(content_obj, stream_obj);
        }

        // ── Emit ─────────────────────────────────────────────────────────────
        builder.emit_objects();
        Ok(builder.finalize(catalog_obj))
    }
}

#[cfg(test)]
mod report_html_tests {
    use super::*;
    use crate::rules::{CheckResult, Severity};

    fn make_report(pass: bool) -> QcReport {
        let mut r = QcReport::new("test_file.mkv");
        if pass {
            r.add_result(CheckResult::pass("dummy_rule"));
        } else {
            r.add_result(
                CheckResult::fail("codec_check", Severity::Error, "Bad codec")
                    .with_recommendation("Use AV1"),
            );
            r.add_result(CheckResult::fail(
                "loudness",
                Severity::Critical,
                "Too loud",
            ));
            r.add_result(CheckResult::fail(
                "frame_rate",
                Severity::Warning,
                "Non-standard fps",
            ));
        }
        r
    }

    #[test]
    fn test_html_export_pass_contains_pass_badge() {
        let report = make_report(true);
        let html = report.to_html().expect("html export should succeed");
        assert!(html.contains("PASS"));
        assert!(html.contains("All checks passed"));
    }

    #[test]
    fn test_html_export_fail_contains_fail_badge() {
        let report = make_report(false);
        let html = report.to_html().expect("html export should succeed");
        assert!(html.contains("FAIL"));
    }

    #[test]
    fn test_html_export_contains_file_path() {
        let report = make_report(false);
        let html = report.to_html().expect("html export should succeed");
        assert!(html.contains("test_file.mkv"));
    }

    #[test]
    fn test_html_export_contains_rule_names() {
        let report = make_report(false);
        let html = report.to_html().expect("html export should succeed");
        assert!(html.contains("codec_check"));
        assert!(html.contains("loudness"));
        assert!(html.contains("frame_rate"));
    }

    #[test]
    fn test_html_export_contains_recommendation() {
        let report = make_report(false);
        let html = report.to_html().expect("html export should succeed");
        assert!(html.contains("Use AV1"));
    }

    #[test]
    fn test_html_escape_prevents_injection() {
        let mut report = QcReport::new("<script>alert('xss')</script>");
        report.add_result(CheckResult::fail(
            "xss_rule",
            Severity::Warning,
            "msg with <b>html</b>",
        ));
        let html = report.to_html().expect("html export should succeed");
        assert!(!html.contains("<script>"));
        assert!(html.contains("&lt;script&gt;"));
    }

    #[test]
    fn test_html_format_extension() {
        assert_eq!(ReportFormat::Html.extension(), "html");
    }

    #[test]
    fn test_html_report_with_stream_and_timestamp() {
        let mut report = QcReport::new("vid.mkv");
        report.add_result(
            CheckResult::fail("test", Severity::Warning, "warn")
                .with_stream(2)
                .with_timestamp(45.0),
        );
        let html = report.to_html().expect("html export should succeed");
        assert!(html.contains("45.00s"));
    }

    #[test]
    fn test_html_report_duration_displayed() {
        let mut report = QcReport::new("vid.mkv");
        report.set_validation_duration(3.141);
        let html = report.to_html().expect("html export should succeed");
        assert!(html.contains("3.141s"));
    }
}

// Use actual chrono crate

// ── PDF tests ────────────────────────────────────────────────────────────────
#[cfg(all(test, feature = "pdf"))]
mod report_pdf_tests {
    use super::*;
    use crate::rules::{CheckResult, Severity};

    fn make_pdf_report() -> QcReport {
        let mut r = QcReport::new("test_video.mkv");
        r.add_result(CheckResult::pass("audio_loudness"));
        r.add_result(CheckResult::fail(
            "video_bitrate",
            Severity::Error,
            "Bitrate too low",
        ));
        r
    }

    #[test]
    fn test_pdf_header_footer() {
        let report = make_pdf_report();
        let pdf = report.to_pdf().expect("PDF export should succeed");
        assert!(pdf.starts_with(b"%PDF-1.4"), "PDF must start with %PDF-1.4");
        let tail = pdf
            .iter()
            .rev()
            .take(20)
            .cloned()
            .collect::<Vec<u8>>()
            .into_iter()
            .rev()
            .collect::<Vec<u8>>();
        let tail_str = String::from_utf8_lossy(&tail);
        assert!(
            tail_str.contains("%%EOF"),
            "PDF must end with %%EOF, got: {tail_str:?}"
        );
    }

    #[test]
    fn test_pdf_has_catalog_and_page() {
        let report = make_pdf_report();
        let pdf = report.to_pdf().expect("PDF export should succeed");
        assert!(
            pdf.windows(b"/Type /Catalog".len())
                .any(|w| w == b"/Type /Catalog"),
            "PDF must contain /Type /Catalog"
        );
        assert!(
            pdf.windows(b"/Type /Page".len())
                .any(|w| w == b"/Type /Page"),
            "PDF must contain /Type /Page"
        );
    }

    #[test]
    fn test_pdf_has_helvetica() {
        let report = make_pdf_report();
        let pdf = report.to_pdf().expect("PDF export should succeed");
        assert!(
            pdf.windows(b"Helvetica".len()).any(|w| w == b"Helvetica"),
            "PDF must reference Helvetica font"
        );
    }

    #[test]
    fn test_pdf_startxref_offset_valid() {
        let report = make_pdf_report();
        let pdf = report.to_pdf().expect("PDF export should succeed");
        let content = String::from_utf8_lossy(&pdf);

        // Extract the number after `startxref\n`
        let marker = "startxref\n";
        let pos = content.find(marker).expect("startxref not found");
        let after = &content[pos + marker.len()..];
        let offset_str = after.lines().next().expect("no line after startxref");
        let offset: usize = offset_str
            .trim()
            .parse()
            .expect("startxref offset is not a number");

        assert!(
            offset < pdf.len(),
            "startxref offset {offset} must be < pdf length {}",
            pdf.len()
        );
        // The bytes at that offset should be the start of the xref table
        assert!(
            pdf[offset..].starts_with(b"xref"),
            "bytes at startxref offset must start with 'xref'"
        );
    }

    #[test]
    fn test_pdf_contains_report_text() {
        let report = make_pdf_report();
        let pdf = report.to_pdf().expect("PDF export should succeed");

        // File path should appear somewhere in the PDF content streams
        assert!(
            pdf.windows(b"test_video.mkv".len())
                .any(|w| w == b"test_video.mkv"),
            "PDF must contain the file path"
        );
        // Overall status (FAIL because of Error-severity check)
        assert!(
            pdf.windows(b"FAIL".len()).any(|w| w == b"FAIL"),
            "PDF must contain FAIL status"
        );
        // Individual result pass status
        assert!(
            pdf.windows(b"PASS".len()).any(|w| w == b"PASS"),
            "PDF must contain PASS for passing checks"
        );
    }

    #[test]
    fn test_pdf_multipage() {
        // Create a report with 100 results to force multi-page layout
        let mut report = QcReport::new("large_report.mxf");
        for i in 0..100 {
            report.add_result(CheckResult::fail(
                &format!("rule_{i:03}"),
                Severity::Warning,
                &format!("Warning message number {i} with some additional text to fill the line"),
            ));
        }
        let pdf = report.to_pdf().expect("PDF export should succeed");

        // Count occurrences of "/Type /Page" (each Page dict has exactly one)
        let needle = b"/Type /Page";
        let count = pdf.windows(needle.len()).filter(|w| *w == needle).count();
        assert!(
            count >= 2,
            "100 results should produce at least 2 pages, got {count} /Type /Page entries"
        );
    }
}
