//! SHACL validation report serialization.
//!
//! Serializes [`ValidationReport`] instances to W3C SHACL RDF formats:
//! Turtle (text/turtle), N-Triples, and JSON-LD.

use std::fmt::Write as FmtWrite;
use std::io::Write as IoWrite;

use thiserror::Error;

use super::validation::{ValidationReport, ValidationSeverity};

/// Output format for SHACL report serialization.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ShaclReportFormat {
    /// Turtle (text/turtle) — human-readable RDF
    Turtle,
    /// N-Triples — line-based RDF
    NTriples,
    /// JSON-LD — JSON-based RDF
    JsonLd,
}

/// Errors that can occur during SHACL report export.
#[derive(Debug, Error)]
pub enum ReportExportError {
    /// An I/O error occurred.
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),
    /// Serialization failed with a message.
    #[error("Serialization failed: {0}")]
    Serialization(String),
}

/// Exports a SHACL [`ValidationReport`] to various RDF formats.
///
/// # Example
///
/// ```
/// use tensorlogic_oxirs_bridge::shacl::report_export::{ShaclReportExporter, ShaclReportFormat};
/// use tensorlogic_oxirs_bridge::shacl::validation::ValidationReport;
///
/// let exporter = ShaclReportExporter::new("http://example.org/");
/// let report = ValidationReport::new();
/// let ttl = exporter.export_to_string(&report, ShaclReportFormat::Turtle).unwrap();
/// assert!(ttl.contains("sh:ValidationReport"));
/// ```
pub struct ShaclReportExporter {
    base_iri: String,
}

impl ShaclReportExporter {
    /// Create a new exporter with the given base IRI.
    pub fn new(base_iri: impl Into<String>) -> Self {
        ShaclReportExporter {
            base_iri: base_iri.into(),
        }
    }

    /// Return the base IRI used by this exporter.
    pub fn base_iri(&self) -> &str {
        &self.base_iri
    }

    /// Export a [`ValidationReport`] to a string in the specified format.
    pub fn export_to_string(
        &self,
        report: &ValidationReport,
        format: ShaclReportFormat,
    ) -> Result<String, ReportExportError> {
        match format {
            ShaclReportFormat::Turtle => self.to_turtle(report),
            ShaclReportFormat::NTriples => self.to_ntriples(report),
            ShaclReportFormat::JsonLd => self.to_jsonld(report),
        }
    }

    /// Write a [`ValidationReport`] to a file at the given path.
    pub fn write_to_file(
        &self,
        path: &std::path::Path,
        report: &ValidationReport,
        format: ShaclReportFormat,
    ) -> Result<(), ReportExportError> {
        let content = self.export_to_string(report, format)?;
        let mut file = std::fs::File::create(path)?;
        file.write_all(content.as_bytes())?;
        Ok(())
    }

    // ── private serialization helpers ─────────────────────────────────────────

    fn severity_turtle_iri(severity: ValidationSeverity) -> &'static str {
        match severity {
            ValidationSeverity::Violation => "sh:Violation",
            ValidationSeverity::Warning => "sh:Warning",
            ValidationSeverity::Info => "sh:Info",
        }
    }

    fn severity_full_iri(severity: ValidationSeverity) -> &'static str {
        match severity {
            ValidationSeverity::Violation => "http://www.w3.org/ns/shacl#Violation",
            ValidationSeverity::Warning => "http://www.w3.org/ns/shacl#Warning",
            ValidationSeverity::Info => "http://www.w3.org/ns/shacl#Info",
        }
    }

    fn escape_literal(s: &str) -> String {
        s.replace('\\', "\\\\")
            .replace('"', "\\\"")
            .replace('\n', "\\n")
            .replace('\r', "\\r")
    }

    /// Serialize to Turtle format.
    fn to_turtle(&self, report: &ValidationReport) -> Result<String, ReportExportError> {
        let mut out = String::new();

        // Prefix declarations
        writeln!(out, "@prefix sh:  <http://www.w3.org/ns/shacl#> .")
            .map_err(|e| ReportExportError::Serialization(e.to_string()))?;
        writeln!(out, "@prefix xsd: <http://www.w3.org/2001/XMLSchema#> .")
            .map_err(|e| ReportExportError::Serialization(e.to_string()))?;
        writeln!(out, "@prefix ex:  <{}> .", self.base_iri)
            .map_err(|e| ReportExportError::Serialization(e.to_string()))?;
        writeln!(out).map_err(|e| ReportExportError::Serialization(e.to_string()))?;

        // Collect result IRIs to link from the report node.
        let result_iris: Vec<String> = (0..report.results.len())
            .map(|i| format!("ex:result{}", i))
            .collect();

        // sh:ValidationReport node
        writeln!(out, "ex:report0 a sh:ValidationReport ;")
            .map_err(|e| ReportExportError::Serialization(e.to_string()))?;

        let conforms_str = if report.conforms { "true" } else { "false" };
        write!(out, "    sh:conforms {} ", conforms_str)
            .map_err(|e| ReportExportError::Serialization(e.to_string()))?;

        if result_iris.is_empty() {
            writeln!(out, ".").map_err(|e| ReportExportError::Serialization(e.to_string()))?;
        } else {
            writeln!(out, ";").map_err(|e| ReportExportError::Serialization(e.to_string()))?;
            for (idx, iri) in result_iris.iter().enumerate() {
                let is_last = idx == result_iris.len() - 1;
                if is_last {
                    writeln!(out, "    sh:result {} .", iri)
                        .map_err(|e| ReportExportError::Serialization(e.to_string()))?;
                } else {
                    writeln!(out, "    sh:result {} ;", iri)
                        .map_err(|e| ReportExportError::Serialization(e.to_string()))?;
                }
            }
        }

        writeln!(out).map_err(|e| ReportExportError::Serialization(e.to_string()))?;

        // Individual sh:ValidationResult nodes
        for (i, result) in report.results.iter().enumerate() {
            let result_iri = &result_iris[i];
            let focus_iri = format!("ex:node{}", i);

            writeln!(out, "{} a sh:ValidationResult ;", result_iri)
                .map_err(|e| ReportExportError::Serialization(e.to_string()))?;
            writeln!(
                out,
                "    sh:resultSeverity {} ;",
                Self::severity_turtle_iri(result.severity)
            )
            .map_err(|e| ReportExportError::Serialization(e.to_string()))?;

            // Encode focus node: if it looks like an IRI, use <…>, else use ex:…
            if result.focus_node.starts_with("http") {
                writeln!(out, "    sh:focusNode <{}> ;", result.focus_node)
                    .map_err(|e| ReportExportError::Serialization(e.to_string()))?;
            } else {
                writeln!(out, "    sh:focusNode {} ;", focus_iri)
                    .map_err(|e| ReportExportError::Serialization(e.to_string()))?;
            }

            if let Some(ref path) = result.result_path {
                writeln!(out, "    sh:resultPath <{}> ;", path)
                    .map_err(|e| ReportExportError::Serialization(e.to_string()))?;
            }

            if let Some(ref value) = result.value {
                writeln!(out, "    sh:value \"{}\" ;", Self::escape_literal(value))
                    .map_err(|e| ReportExportError::Serialization(e.to_string()))?;
            }

            writeln!(
                out,
                "    sh:resultMessage \"{}\" .",
                Self::escape_literal(&result.message)
            )
            .map_err(|e| ReportExportError::Serialization(e.to_string()))?;

            writeln!(out).map_err(|e| ReportExportError::Serialization(e.to_string()))?;
        }

        Ok(out)
    }

    /// Serialize to N-Triples format.
    fn to_ntriples(&self, report: &ValidationReport) -> Result<String, ReportExportError> {
        let shacl = "http://www.w3.org/ns/shacl#";
        let rdf = "http://www.w3.org/1999/02/22-rdf-syntax-ns#";
        let base = self.base_iri.trim_end_matches('/');

        let report_iri = format!("{}/report0", base);
        let mut out = String::new();

        // rdf:type sh:ValidationReport
        writeln!(
            out,
            "<{}> <{}type> <{}ValidationReport> .",
            report_iri, rdf, shacl
        )
        .map_err(|e| ReportExportError::Serialization(e.to_string()))?;

        // sh:conforms
        let conforms_val = if report.conforms {
            "\"true\"^^<http://www.w3.org/2001/XMLSchema#boolean>"
        } else {
            "\"false\"^^<http://www.w3.org/2001/XMLSchema#boolean>"
        };
        writeln!(
            out,
            "<{}> <{}conforms> {} .",
            report_iri, shacl, conforms_val
        )
        .map_err(|e| ReportExportError::Serialization(e.to_string()))?;

        for (i, result) in report.results.iter().enumerate() {
            let result_iri = format!("{}/result{}", base, i);

            // link report → result
            writeln!(out, "<{}> <{}result> <{}> .", report_iri, shacl, result_iri)
                .map_err(|e| ReportExportError::Serialization(e.to_string()))?;

            // rdf:type sh:ValidationResult
            writeln!(
                out,
                "<{}> <{}type> <{}ValidationResult> .",
                result_iri, rdf, shacl
            )
            .map_err(|e| ReportExportError::Serialization(e.to_string()))?;

            // sh:resultSeverity
            writeln!(
                out,
                "<{}> <{}resultSeverity> <{}> .",
                result_iri,
                shacl,
                Self::severity_full_iri(result.severity)
            )
            .map_err(|e| ReportExportError::Serialization(e.to_string()))?;

            // sh:focusNode
            if result.focus_node.starts_with("http") {
                writeln!(
                    out,
                    "<{}> <{}focusNode> <{}> .",
                    result_iri, shacl, result.focus_node
                )
                .map_err(|e| ReportExportError::Serialization(e.to_string()))?;
            } else {
                let node_iri = format!("{}/node{}", base, i);
                writeln!(
                    out,
                    "<{}> <{}focusNode> <{}> .",
                    result_iri, shacl, node_iri
                )
                .map_err(|e| ReportExportError::Serialization(e.to_string()))?;
            }

            // sh:resultMessage
            writeln!(
                out,
                "<{}> <{}resultMessage> \"{}\" .",
                result_iri,
                shacl,
                Self::escape_literal(&result.message)
            )
            .map_err(|e| ReportExportError::Serialization(e.to_string()))?;
        }

        Ok(out)
    }

    /// Serialize to JSON-LD format.
    fn to_jsonld(&self, report: &ValidationReport) -> Result<String, ReportExportError> {
        let mut json = serde_json::Map::new();

        // @context
        let mut ctx = serde_json::Map::new();
        ctx.insert(
            "sh".to_string(),
            serde_json::Value::String("http://www.w3.org/ns/shacl#".to_string()),
        );
        ctx.insert(
            "xsd".to_string(),
            serde_json::Value::String("http://www.w3.org/2001/XMLSchema#".to_string()),
        );
        ctx.insert(
            "ex".to_string(),
            serde_json::Value::String(self.base_iri.clone()),
        );
        json.insert("@context".to_string(), serde_json::Value::Object(ctx));

        // @id and @type
        let base = self.base_iri.trim_end_matches('/');
        json.insert(
            "@id".to_string(),
            serde_json::Value::String(format!("{}/report0", base)),
        );
        json.insert(
            "@type".to_string(),
            serde_json::Value::String("sh:ValidationReport".to_string()),
        );

        // sh:conforms
        json.insert(
            "sh:conforms".to_string(),
            serde_json::Value::Bool(report.conforms),
        );

        // sh:result array
        if !report.results.is_empty() {
            let results_json: Vec<serde_json::Value> = report
                .results
                .iter()
                .enumerate()
                .map(|(i, r)| {
                    let mut obj = serde_json::Map::new();
                    obj.insert(
                        "@id".to_string(),
                        serde_json::Value::String(format!("{}/result{}", base, i)),
                    );
                    obj.insert(
                        "@type".to_string(),
                        serde_json::Value::String("sh:ValidationResult".to_string()),
                    );
                    let severity = match r.severity {
                        ValidationSeverity::Violation => "sh:Violation",
                        ValidationSeverity::Warning => "sh:Warning",
                        ValidationSeverity::Info => "sh:Info",
                    };
                    obj.insert(
                        "sh:resultSeverity".to_string(),
                        serde_json::Value::String(severity.to_string()),
                    );
                    obj.insert(
                        "sh:focusNode".to_string(),
                        serde_json::Value::String(r.focus_node.clone()),
                    );
                    obj.insert(
                        "sh:resultMessage".to_string(),
                        serde_json::Value::String(r.message.clone()),
                    );
                    if let Some(ref path) = r.result_path {
                        obj.insert(
                            "sh:resultPath".to_string(),
                            serde_json::Value::String(path.clone()),
                        );
                    }
                    if let Some(ref value) = r.value {
                        obj.insert(
                            "sh:value".to_string(),
                            serde_json::Value::String(value.clone()),
                        );
                    }
                    serde_json::Value::Object(obj)
                })
                .collect();
            json.insert(
                "sh:result".to_string(),
                serde_json::Value::Array(results_json),
            );
        }

        serde_json::to_string_pretty(&serde_json::Value::Object(json))
            .map_err(|e| ReportExportError::Serialization(e.to_string()))
    }
}

// ── tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::shacl::validation::{ValidationReport, ValidationResult, ValidationSeverity};

    fn make_exporter() -> ShaclReportExporter {
        ShaclReportExporter::new("http://example.org/")
    }

    fn make_violation(msg: &str) -> ValidationResult {
        ValidationResult::new(
            "http://example.org/node",
            "http://example.org/Shape",
            "http://www.w3.org/ns/shacl#MinCountConstraintComponent",
            msg,
        )
    }

    fn make_warning(msg: &str) -> ValidationResult {
        make_violation(msg).with_severity(ValidationSeverity::Warning)
    }

    #[test]
    fn test_export_turtle_conforms_true() {
        let exporter = make_exporter();
        let report = ValidationReport::new();
        let output = exporter
            .export_to_string(&report, ShaclReportFormat::Turtle)
            .expect("export should succeed");
        assert!(output.contains("sh:ValidationReport"));
        assert!(output.contains("sh:conforms true"));
    }

    #[test]
    fn test_export_turtle_with_violation() {
        let exporter = make_exporter();
        let mut report = ValidationReport::new();
        report.add_result(make_violation("Missing required property"));
        let output = exporter
            .export_to_string(&report, ShaclReportFormat::Turtle)
            .expect("export should succeed");
        assert!(output.contains("sh:ValidationResult"));
        assert!(output.contains("sh:Violation"));
        assert!(output.contains("Missing required property"));
        assert!(output.contains("sh:conforms false"));
    }

    #[test]
    fn test_export_turtle_with_warning() {
        let exporter = make_exporter();
        let mut report = ValidationReport::new();
        report.add_result(make_warning("Non-critical issue"));
        let output = exporter
            .export_to_string(&report, ShaclReportFormat::Turtle)
            .expect("export should succeed");
        assert!(output.contains("sh:ValidationResult"));
        assert!(output.contains("sh:Warning"));
        assert!(output.contains("Non-critical issue"));
    }

    #[test]
    fn test_export_ntriples_conforms() {
        let exporter = make_exporter();
        let report = ValidationReport::new();
        let output = exporter
            .export_to_string(&report, ShaclReportFormat::NTriples)
            .expect("export should succeed");
        assert!(output.contains("ValidationReport"));
        assert!(output.contains("conforms"));
        assert!(output.contains("true"));
    }

    #[test]
    fn test_export_ntriples_with_results() {
        let exporter = make_exporter();
        let mut report = ValidationReport::new();
        report.add_result(make_violation("Error in data"));
        let output = exporter
            .export_to_string(&report, ShaclReportFormat::NTriples)
            .expect("export should succeed");
        assert!(output.contains("ValidationResult"));
        assert!(output.contains("resultSeverity"));
        assert!(output.contains("Violation"));
    }

    #[test]
    fn test_export_jsonld_basic() {
        let exporter = make_exporter();
        let report = ValidationReport::new();
        let output = exporter
            .export_to_string(&report, ShaclReportFormat::JsonLd)
            .expect("export should succeed");
        assert!(output.contains("@type"));
        assert!(output.contains("sh:ValidationReport"));
        assert!(output.contains("@context"));
    }

    #[test]
    fn test_export_jsonld_with_result() {
        let exporter = make_exporter();
        let mut report = ValidationReport::new();
        report.add_result(make_violation("JSON-LD test violation"));
        let output = exporter
            .export_to_string(&report, ShaclReportFormat::JsonLd)
            .expect("export should succeed");
        assert!(output.contains("sh:ValidationResult"));
        assert!(output.contains("sh:result"));
        assert!(output.contains("JSON-LD test violation"));
    }

    #[test]
    fn test_write_to_temp_file() {
        let exporter = make_exporter();
        let report = ValidationReport::new();
        let dir = std::env::temp_dir();
        let path = dir.join("shacl_report_test.ttl");
        exporter
            .write_to_file(&path, &report, ShaclReportFormat::Turtle)
            .expect("write should succeed");
        let content = std::fs::read_to_string(&path).expect("read back should succeed");
        assert!(content.contains("sh:ValidationReport"));
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn test_export_all_formats_same_report() {
        let exporter = make_exporter();
        let mut report = ValidationReport::new();
        report.add_result(make_violation("All formats test"));
        for format in [
            ShaclReportFormat::Turtle,
            ShaclReportFormat::NTriples,
            ShaclReportFormat::JsonLd,
        ] {
            exporter
                .export_to_string(&report, format)
                .unwrap_or_else(|e| panic!("export failed for {:?}: {}", format, e));
        }
    }

    #[test]
    fn test_export_multiple_results() {
        let exporter = make_exporter();
        let mut report = ValidationReport::new();
        report.add_result(make_violation("Error 1"));
        report.add_result(make_violation("Error 2"));
        report.add_result(make_warning("Warning 1"));
        let output = exporter
            .export_to_string(&report, ShaclReportFormat::Turtle)
            .expect("export should succeed");
        assert!(output.contains("ex:result0"));
        assert!(output.contains("ex:result1"));
        assert!(output.contains("ex:result2"));
    }

    #[test]
    fn test_export_base_iri_in_output() {
        let exporter = ShaclReportExporter::new("http://myontology.example.com/ns/");
        let report = ValidationReport::new();
        let output = exporter
            .export_to_string(&report, ShaclReportFormat::Turtle)
            .expect("export should succeed");
        assert!(output.contains("http://myontology.example.com/ns/"));
    }

    #[test]
    fn test_round_trip_file_content() {
        let exporter = make_exporter();
        let mut report = ValidationReport::new();
        report.add_result(make_violation("Round trip check"));
        let dir = std::env::temp_dir();
        let path = dir.join("shacl_roundtrip_test.ttl");
        exporter
            .write_to_file(&path, &report, ShaclReportFormat::Turtle)
            .expect("write should succeed");
        let content = std::fs::read_to_string(&path).expect("read back should succeed");
        assert!(!content.is_empty());
        assert!(content.contains("sh:ValidationReport"));
        let _ = std::fs::remove_file(&path);
    }
}
