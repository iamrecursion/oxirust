//! Comprehensive reporting and statistics generation.

use crate::analysis::SessionAnalysis;
use crate::error::ConformResult;
use crate::exporters::report::MatchReport;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::Path;

/// Report format.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ReportFormat {
    /// JSON format.
    Json,
    /// HTML format.
    Html,
    /// CSV format.
    Csv,
    /// Plain text.
    Text,
    /// Markdown.
    Markdown,
}

/// Comprehensive conform report.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConformReport {
    /// Session name.
    pub session_name: String,
    /// Created at timestamp.
    pub created_at: String,
    /// Match report.
    pub match_report: MatchReport,
    /// Session analysis.
    pub session_analysis: Option<SessionAnalysis>,
    /// Additional metadata.
    pub metadata: HashMap<String, String>,
}

impl ConformReport {
    /// Create a new conform report.
    #[must_use]
    pub fn new(session_name: String, match_report: MatchReport) -> Self {
        Self {
            session_name,
            created_at: chrono::Utc::now().to_rfc3339(),
            match_report,
            session_analysis: None,
            metadata: HashMap::new(),
        }
    }

    /// Set session analysis.
    #[must_use]
    pub fn with_analysis(mut self, analysis: SessionAnalysis) -> Self {
        self.session_analysis = Some(analysis);
        self
    }

    /// Add metadata.
    pub fn add_metadata(&mut self, key: String, value: String) {
        self.metadata.insert(key, value);
    }

    /// Generate report in specified format.
    #[must_use]
    pub fn generate(&self, format: ReportFormat) -> String {
        match format {
            ReportFormat::Json => self.to_json(),
            ReportFormat::Html => self.to_html(),
            ReportFormat::Csv => self.to_csv(),
            ReportFormat::Text => self.to_text(),
            ReportFormat::Markdown => self.to_markdown(),
        }
    }

    /// Convert to JSON.
    fn to_json(&self) -> String {
        serde_json::to_string_pretty(self).unwrap_or_else(|_| "{}".to_string())
    }

    /// Convert to HTML.
    fn to_html(&self) -> String {
        let mut html = String::new();

        html.push_str("<!DOCTYPE html>\n<html>\n<head>\n");
        html.push_str("<meta charset=\"UTF-8\">\n");
        html.push_str("<title>Conform Report</title>\n");
        html.push_str("<style>\n");
        html.push_str(include_str!("reporting_style.css"));
        html.push_str("</style>\n");
        html.push_str("</head>\n<body>\n");

        html.push_str(&format!("<h1>Conform Report: {}</h1>\n", self.session_name));
        html.push_str(&format!("<p>Generated: {}</p>\n", self.created_at));

        // Summary section
        html.push_str("<section class=\"summary\">\n");
        html.push_str("<h2>Summary</h2>\n");
        html.push_str("<table>\n");
        html.push_str(&format!(
            "<tr><td>Total Clips</td><td>{}</td></tr>\n",
            self.match_report.stats.total_clips
        ));
        html.push_str(&format!(
            "<tr><td>Matched</td><td class=\"success\">{}</td></tr>\n",
            self.match_report.stats.matched_count
        ));
        html.push_str(&format!(
            "<tr><td>Missing</td><td class=\"error\">{}</td></tr>\n",
            self.match_report.stats.missing_count
        ));
        html.push_str(&format!(
            "<tr><td>Ambiguous</td><td class=\"warning\">{}</td></tr>\n",
            self.match_report.stats.ambiguous_count
        ));
        html.push_str(&format!(
            "<tr><td>Conform Rate</td><td>{:.1}%</td></tr>\n",
            self.match_report.stats.conform_rate * 100.0
        ));
        html.push_str("</table>\n");
        html.push_str("</section>\n");

        // Matched clips section
        if !self.match_report.matched.is_empty() {
            html.push_str("<section class=\"matched\">\n");
            html.push_str(&format!(
                "<h2>Matched Clips ({})</h2>\n",
                self.match_report.matched.len()
            ));
            html.push_str("<table>\n");
            html.push_str("<thead>\n");
            html.push_str(
                "<tr><th>Clip ID</th><th>Source</th><th>Score</th><th>Method</th></tr>\n",
            );
            html.push_str("</thead>\n<tbody>\n");

            for m in &self.match_report.matched {
                html.push_str(&format!(
                    "<tr><td>{}</td><td>{}</td><td>{:.3}</td><td>{}</td></tr>\n",
                    m.clip.id,
                    m.clip.source_file.as_ref().unwrap_or(&"-".to_string()),
                    m.score,
                    m.method
                ));
            }

            html.push_str("</tbody>\n</table>\n");
            html.push_str("</section>\n");
        }

        // Missing clips section
        if !self.match_report.missing.is_empty() {
            html.push_str("<section class=\"missing\">\n");
            html.push_str(&format!(
                "<h2>Missing Clips ({})</h2>\n",
                self.match_report.missing.len()
            ));
            html.push_str("<ul>\n");

            for clip in &self.match_report.missing {
                html.push_str(&format!(
                    "<li>{} - {}</li>\n",
                    clip.id,
                    clip.source_file.as_ref().unwrap_or(&"-".to_string())
                ));
            }

            html.push_str("</ul>\n");
            html.push_str("</section>\n");
        }

        // Analysis section
        if let Some(ref analysis) = self.session_analysis {
            html.push_str("<section class=\"analysis\">\n");
            html.push_str("<h2>Analysis</h2>\n");

            if let Some(ref timeline_stats) = analysis.timeline_stats {
                html.push_str("<h3>Timeline Statistics</h3>\n");
                html.push_str("<table>\n");
                html.push_str(&format!(
                    "<tr><td>Duration</td><td>{:.2}s</td></tr>\n",
                    timeline_stats.duration_seconds
                ));
                html.push_str(&format!(
                    "<tr><td>Total Clips</td><td>{}</td></tr>\n",
                    timeline_stats.clip_count
                ));
                html.push_str(&format!(
                    "<tr><td>Video Clips</td><td>{}</td></tr>\n",
                    timeline_stats.video_clip_count
                ));
                html.push_str(&format!(
                    "<tr><td>Audio Clips</td><td>{}</td></tr>\n",
                    timeline_stats.audio_clip_count
                ));
                html.push_str("</table>\n");
            }

            html.push_str("</section>\n");
        }

        html.push_str("</body>\n</html>\n");
        html
    }

    /// Convert to CSV.
    fn to_csv(&self) -> String {
        let mut csv = String::new();

        csv.push_str("Clip ID,Source File,Score,Method,Status\n");

        for m in &self.match_report.matched {
            csv.push_str(&format!(
                "{},{},{:.3},{},Matched\n",
                m.clip.id,
                m.clip.source_file.as_ref().unwrap_or(&"-".to_string()),
                m.score,
                m.method
            ));
        }

        for clip in &self.match_report.missing {
            csv.push_str(&format!(
                "{},{},0.0,-,Missing\n",
                clip.id,
                clip.source_file.as_ref().unwrap_or(&"-".to_string())
            ));
        }

        csv
    }

    /// Convert to plain text.
    fn to_text(&self) -> String {
        let mut text = String::new();

        text.push_str(&format!("CONFORM REPORT: {}\n", self.session_name));
        text.push_str(&format!("Generated: {}\n\n", self.created_at));

        text.push_str("SUMMARY\n");
        text.push_str(&format!(
            "  Total Clips:   {}\n",
            self.match_report.stats.total_clips
        ));
        text.push_str(&format!(
            "  Matched:       {}\n",
            self.match_report.stats.matched_count
        ));
        text.push_str(&format!(
            "  Missing:       {}\n",
            self.match_report.stats.missing_count
        ));
        text.push_str(&format!(
            "  Ambiguous:     {}\n",
            self.match_report.stats.ambiguous_count
        ));
        text.push_str(&format!(
            "  Conform Rate:  {:.1}%\n\n",
            self.match_report.stats.conform_rate * 100.0
        ));

        if !self.match_report.matched.is_empty() {
            text.push_str("MATCHED CLIPS\n");
            for m in &self.match_report.matched {
                text.push_str(&format!(
                    "  {} - {} (score: {:.3}, method: {})\n",
                    m.clip.id,
                    m.clip.source_file.as_ref().unwrap_or(&"-".to_string()),
                    m.score,
                    m.method
                ));
            }
            text.push('\n');
        }

        if !self.match_report.missing.is_empty() {
            text.push_str("MISSING CLIPS\n");
            for clip in &self.match_report.missing {
                text.push_str(&format!(
                    "  {} - {}\n",
                    clip.id,
                    clip.source_file.as_ref().unwrap_or(&"-".to_string())
                ));
            }
            text.push('\n');
        }

        text
    }

    /// Convert to Markdown.
    fn to_markdown(&self) -> String {
        let mut md = String::new();

        md.push_str(&format!("# Conform Report: {}\n\n", self.session_name));
        md.push_str(&format!("*Generated: {}*\n\n", self.created_at));

        md.push_str("## Summary\n\n");
        md.push_str("| Metric | Value |\n");
        md.push_str("|--------|-------|\n");
        md.push_str(&format!(
            "| Total Clips | {} |\n",
            self.match_report.stats.total_clips
        ));
        md.push_str(&format!(
            "| Matched | {} |\n",
            self.match_report.stats.matched_count
        ));
        md.push_str(&format!(
            "| Missing | {} |\n",
            self.match_report.stats.missing_count
        ));
        md.push_str(&format!(
            "| Ambiguous | {} |\n",
            self.match_report.stats.ambiguous_count
        ));
        md.push_str(&format!(
            "| Conform Rate | {:.1}% |\n\n",
            self.match_report.stats.conform_rate * 100.0
        ));

        if !self.match_report.matched.is_empty() {
            md.push_str("## Matched Clips\n\n");
            md.push_str("| Clip ID | Source | Score | Method |\n");
            md.push_str("|---------|--------|-------|--------|\n");

            for m in &self.match_report.matched {
                md.push_str(&format!(
                    "| {} | {} | {:.3} | {} |\n",
                    m.clip.id,
                    m.clip.source_file.as_ref().unwrap_or(&"-".to_string()),
                    m.score,
                    m.method
                ));
            }
            md.push('\n');
        }

        if !self.match_report.missing.is_empty() {
            md.push_str("## Missing Clips\n\n");
            for clip in &self.match_report.missing {
                md.push_str(&format!(
                    "- {} - {}\n",
                    clip.id,
                    clip.source_file.as_ref().unwrap_or(&"-".to_string())
                ));
            }
            md.push('\n');
        }

        md
    }

    /// Save report to file.
    ///
    /// # Errors
    ///
    /// Returns an error if the file cannot be written.
    pub fn save<P: AsRef<Path>>(&self, path: P, format: ReportFormat) -> ConformResult<()> {
        let content = self.generate(format);
        std::fs::write(path, content)?;
        Ok(())
    }
}

/// CSS styles for HTML reports (embedded).
#[allow(dead_code)]
const fn reporting_style() -> &'static str {
    r"
body {
    font-family: Arial, sans-serif;
    margin: 20px;
    background-color: #f5f5f5;
}
h1 {
    color: #333;
}
section {
    background: white;
    padding: 20px;
    margin: 20px 0;
    border-radius: 5px;
    box-shadow: 0 2px 4px rgba(0,0,0,0.1);
}
table {
    width: 100%;
    border-collapse: collapse;
}
th, td {
    padding: 12px;
    text-align: left;
    border-bottom: 1px solid #ddd;
}
th {
    background-color: #4CAF50;
    color: white;
}
.success {
    color: #4CAF50;
    font-weight: bold;
}
.error {
    color: #f44336;
    font-weight: bold;
}
.warning {
    color: #ff9800;
    font-weight: bold;
}
"
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::{
        ClipMatch, ClipReference, FrameRate, MatchMethod, MediaFile, Timecode, TrackType,
    };
    use std::collections::HashMap;
    use std::path::PathBuf;

    fn create_test_report() -> ConformReport {
        let clip = ClipReference {
            id: "test1".to_string(),
            source_file: Some("test.mov".to_string()),
            source_in: Timecode::new(1, 0, 0, 0),
            source_out: Timecode::new(1, 0, 10, 0),
            record_in: Timecode::new(1, 0, 0, 0),
            record_out: Timecode::new(1, 0, 10, 0),
            track: TrackType::Video,
            fps: FrameRate::Fps25,
            metadata: HashMap::new(),
        };

        let clip_match = ClipMatch {
            clip,
            media: MediaFile::new(PathBuf::from("/test/file.mov")),
            score: 0.95,
            method: MatchMethod::ExactFilename,
            details: String::new(),
        };

        let match_report = MatchReport::new(vec![clip_match], vec![], vec![]);

        ConformReport::new("Test Session".to_string(), match_report)
    }

    #[test]
    fn test_report_creation() {
        let report = create_test_report();
        assert_eq!(report.session_name, "Test Session");
    }

    #[test]
    fn test_json_generation() {
        let report = create_test_report();
        let json = report.generate(ReportFormat::Json);
        assert!(!json.is_empty());
        assert!(json.contains("Test Session"));
    }

    #[test]
    fn test_text_generation() {
        let report = create_test_report();
        let text = report.generate(ReportFormat::Text);
        assert!(text.contains("CONFORM REPORT"));
        assert!(text.contains("Test Session"));
    }

    #[test]
    fn test_html_generation() {
        let report = create_test_report();
        let html = report.generate(ReportFormat::Html);
        assert!(html.contains("<html>"));
        assert!(html.contains("Test Session"));
    }

    #[test]
    fn test_markdown_generation() {
        let report = create_test_report();
        let md = report.generate(ReportFormat::Markdown);
        assert!(md.contains("# Conform Report"));
        assert!(md.contains("Test Session"));
    }

    #[test]
    fn test_csv_generation() {
        let report = create_test_report();
        let csv = report.generate(ReportFormat::Csv);
        assert!(csv.contains("Clip ID,Source File"));
    }
}
