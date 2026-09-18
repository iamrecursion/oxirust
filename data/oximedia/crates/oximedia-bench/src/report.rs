//! Report generation for benchmark results (JSON, CSV, HTML).

use crate::{BenchResult, BenchmarkResults};
use std::path::Path;

/// HTML report generator.
pub struct HtmlReport<'a> {
    results: &'a BenchmarkResults,
}

impl<'a> HtmlReport<'a> {
    /// Create a new HTML report.
    #[must_use]
    pub fn new(results: &'a BenchmarkResults) -> Self {
        Self { results }
    }

    /// Write the report to a file.
    ///
    /// # Errors
    ///
    /// Returns an error if the file cannot be written.
    pub fn write_to_file(&self, path: impl AsRef<Path>) -> BenchResult<()> {
        let html = self.generate_html();
        std::fs::write(path, html)?;
        Ok(())
    }

    fn generate_html(&self) -> String {
        format!(
            r#"<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>OxiMedia Codec Benchmark Results</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; background: #f5f5f5; }}
        .container {{ max-width: 1200px; margin: 0 auto; background: white; padding: 20px; border-radius: 8px; }}
        h1 {{ color: #333; }}
        table {{ width: 100%; border-collapse: collapse; margin: 20px 0; }}
        th, td {{ padding: 12px; text-align: left; border-bottom: 1px solid #ddd; }}
        th {{ background-color: #4CAF50; color: white; }}
        tr:hover {{ background-color: #f5f5f5; }}
        .metric {{ font-weight: bold; color: #4CAF50; }}
        .summary {{ background: #e8f5e9; padding: 15px; border-radius: 4px; margin: 20px 0; }}
    </style>
</head>
<body>
    <div class="container">
        <h1>OxiMedia Codec Benchmark Results</h1>
        <div class="summary">
            <p><strong>Timestamp:</strong> {}</p>
            <p><strong>Total Duration:</strong> {:.2}s</p>
            <p><strong>Codecs Tested:</strong> {}</p>
        </div>
        {}
    </div>
</body>
</html>"#,
            self.results.timestamp,
            self.results.total_duration.as_secs_f64(),
            self.results.codec_results.len(),
            self.generate_tables()
        )
    }

    fn generate_tables(&self) -> String {
        let mut tables = String::new();

        for codec_result in &self.results.codec_results {
            tables.push_str(&format!(
                "<h2>Codec: {:?}</h2>\n<table>\n<tr>\n\
                <th>Sequence</th><th>Encoding FPS</th><th>Decoding FPS</th>\
                <th>File Size (MB)</th><th>PSNR (dB)</th><th>SSIM</th>\n</tr>\n",
                codec_result.codec_id
            ));

            for seq in &codec_result.sequence_results {
                tables.push_str(&format!(
                    "<tr><td>{}</td><td>{:.2}</td><td>{:.2}</td><td>{:.2}</td><td>{}</td><td>{}</td></tr>\n",
                    seq.sequence_name,
                    seq.encoding_fps,
                    seq.decoding_fps,
                    seq.file_size_bytes as f64 / 1_000_000.0,
                    seq.metrics.psnr.map_or("N/A".to_string(), |p| format!("{p:.2}")),
                    seq.metrics.ssim.map_or("N/A".to_string(), |s| format!("{s:.4}")),
                ));
            }

            tables.push_str("</table>\n");
        }

        tables
    }
}

/// Report exporter trait.
pub trait ReportExporter {
    /// Export to JSON.
    ///
    /// # Errors
    ///
    /// Returns an error if export fails.
    fn export_json(&self, path: impl AsRef<Path>) -> BenchResult<()>;

    /// Export to CSV.
    ///
    /// # Errors
    ///
    /// Returns an error if export fails.
    fn export_csv(&self, path: impl AsRef<Path>) -> BenchResult<()>;

    /// Export to HTML.
    ///
    /// # Errors
    ///
    /// Returns an error if export fails.
    fn export_html(&self, path: impl AsRef<Path>) -> BenchResult<()>;
}

impl ReportExporter for BenchmarkResults {
    fn export_json(&self, path: impl AsRef<Path>) -> BenchResult<()> {
        BenchmarkResults::export_json(self, path)
    }

    fn export_csv(&self, path: impl AsRef<Path>) -> BenchResult<()> {
        BenchmarkResults::export_csv(self, path)
    }

    fn export_html(&self, path: impl AsRef<Path>) -> BenchResult<()> {
        BenchmarkResults::export_html(self, path)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{BenchmarkConfig, BenchmarkResults};
    use std::time::Duration;

    #[test]
    fn test_html_report_generation() {
        let results = BenchmarkResults {
            codec_results: vec![],
            timestamp: "2024-01-01T00:00:00Z".to_string(),
            total_duration: Duration::from_secs(100),
            config: BenchmarkConfig::default(),
        };

        let report = HtmlReport::new(&results);
        let html = report.generate_html();

        assert!(html.contains("<!DOCTYPE html>"));
        assert!(html.contains("OxiMedia Codec Benchmark Results"));
        assert!(html.contains("2024-01-01T00:00:00Z"));
    }
}

/// Advanced HTML report generator with charts and interactivity.
pub struct AdvancedHtmlReport<'a> {
    results: &'a BenchmarkResults,
    include_charts: bool,
    include_detailed_stats: bool,
}

impl<'a> AdvancedHtmlReport<'a> {
    /// Create a new advanced HTML report.
    #[must_use]
    pub fn new(results: &'a BenchmarkResults) -> Self {
        Self {
            results,
            include_charts: true,
            include_detailed_stats: true,
        }
    }

    /// Set whether to include charts.
    #[must_use]
    pub fn with_charts(mut self, include: bool) -> Self {
        self.include_charts = include;
        self
    }

    /// Set whether to include detailed statistics.
    #[must_use]
    pub fn with_detailed_stats(mut self, include: bool) -> Self {
        self.include_detailed_stats = include;
        self
    }

    /// Generate the advanced HTML report.
    #[must_use]
    pub fn generate(&self) -> String {
        let mut html = self.generate_header();
        html.push_str(&self.generate_summary());

        if self.include_charts {
            html.push_str(&self.generate_charts());
        }

        html.push_str(&self.generate_codec_sections());

        if self.include_detailed_stats {
            html.push_str(&self.generate_statistics());
        }

        html.push_str(&self.generate_footer());
        html
    }

    fn generate_header(&self) -> String {
        String::from(
            r#"<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>OxiMedia Codec Benchmark Results - Advanced Report</title>
    <script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.0/dist/chart.umd.min.js"></script>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, Cantarell, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            padding: 20px;
            color: #333;
        }
        .container {
            max-width: 1400px;
            margin: 0 auto;
            background: white;
            border-radius: 12px;
            box-shadow: 0 20px 60px rgba(0,0,0,0.3);
            overflow: hidden;
        }
        header {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 40px;
            text-align: center;
        }
        h1 { font-size: 2.5em; margin-bottom: 10px; }
        .subtitle { opacity: 0.9; font-size: 1.1em; }
        .summary {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 20px;
            padding: 30px;
            background: #f8f9fa;
        }
        .summary-card {
            background: white;
            padding: 20px;
            border-radius: 8px;
            border-left: 4px solid #667eea;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        .summary-card h3 { color: #667eea; margin-bottom: 10px; font-size: 0.9em; text-transform: uppercase; }
        .summary-card .value { font-size: 2em; font-weight: bold; color: #333; }
        .charts { padding: 30px; }
        .chart-container {
            background: white;
            padding: 20px;
            border-radius: 8px;
            margin-bottom: 30px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        }
        table {
            width: 100%;
            border-collapse: separate;
            border-spacing: 0;
            margin: 20px 0;
        }
        th, td {
            padding: 15px;
            text-align: left;
            border-bottom: 1px solid #e0e0e0;
        }
        th {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            font-weight: 600;
            position: sticky;
            top: 0;
        }
        tbody tr:hover { background: #f8f9fa; }
        .metric { font-weight: bold; }
        .metric.good { color: #4CAF50; }
        .metric.average { color: #FF9800; }
        .metric.poor { color: #F44336; }
        .codec-section {
            padding: 30px;
            border-top: 2px solid #e0e0e0;
        }
        .codec-title {
            font-size: 1.8em;
            margin-bottom: 20px;
            color: #764ba2;
        }
        .stats-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 15px;
            margin: 20px 0;
        }
        .stat-box {
            background: #f8f9fa;
            padding: 15px;
            border-radius: 6px;
            border: 1px solid #e0e0e0;
        }
        .stat-label { font-size: 0.85em; color: #666; margin-bottom: 5px; }
        .stat-value { font-size: 1.5em; font-weight: bold; color: #333; }
        footer {
            background: #f8f9fa;
            padding: 20px;
            text-align: center;
            color: #666;
            font-size: 0.9em;
        }
        .badge {
            display: inline-block;
            padding: 4px 12px;
            border-radius: 12px;
            font-size: 0.85em;
            font-weight: 600;
        }
        .badge.speed { background: #E3F2FD; color: #1976D2; }
        .badge.quality { background: #F3E5F5; color: #7B1FA2; }
        .badge.efficiency { background: #E8F5E9; color: #388E3C; }
    </style>
</head>
<body>
    <div class="container">
        <header>
            <h1>OxiMedia Codec Benchmark Results</h1>
            <p class="subtitle">Advanced Performance Analysis Report</p>
        </header>
"#,
        )
    }

    fn generate_summary(&self) -> String {
        format!(
            r#"        <div class="summary">
            <div class="summary-card">
                <h3>Total Benchmarks</h3>
                <div class="value">{}</div>
            </div>
            <div class="summary-card">
                <h3>Codecs Tested</h3>
                <div class="value">{}</div>
            </div>
            <div class="summary-card">
                <h3>Execution Time</h3>
                <div class="value">{:.1}s</div>
            </div>
            <div class="summary-card">
                <h3>Timestamp</h3>
                <div class="value" style="font-size: 1.2em;">{}</div>
            </div>
        </div>
"#,
            self.results
                .codec_results
                .iter()
                .map(|r| r.sequence_results.len())
                .sum::<usize>(),
            self.results.codec_results.len(),
            self.results.total_duration.as_secs_f64(),
            self.results.timestamp
        )
    }

    fn generate_charts(&self) -> String {
        let mut html = String::from(
            r#"        <div class="charts">
            <div class="chart-container">
                <h3>Encoding Speed Comparison</h3>
                <canvas id="encodingSpeedChart"></canvas>
            </div>
            <div class="chart-container">
                <h3>Quality Metrics Comparison</h3>
                <canvas id="qualityChart"></canvas>
            </div>
            <div class="chart-container">
                <h3>File Size Comparison</h3>
                <canvas id="fileSizeChart"></canvas>
            </div>
        </div>
"#,
        );

        // Add Chart.js initialization scripts
        html.push_str(&self.generate_chart_scripts());
        html
    }

    fn generate_chart_scripts(&self) -> String {
        let mut scripts = String::from(
            r#"<script>
document.addEventListener('DOMContentLoaded', function() {
"#,
        );

        // Encoding speed chart
        scripts.push_str("    const encodingCtx = document.getElementById('encodingSpeedChart').getContext('2d');\n");
        scripts.push_str("    new Chart(encodingCtx, {\n");
        scripts.push_str("        type: 'bar',\n");
        scripts.push_str("        data: {\n");
        scripts.push_str(&format!(
            "            labels: {:?},\n",
            self.results
                .codec_results
                .iter()
                .map(|r| format!("{:?}", r.codec_id))
                .collect::<Vec<_>>()
        ));
        scripts.push_str("            datasets: [{\n");
        scripts.push_str("                label: 'Encoding FPS',\n");
        scripts.push_str(&format!(
            "                data: {:?},\n",
            self.results
                .codec_results
                .iter()
                .map(|r| r.statistics.mean_encoding_fps)
                .collect::<Vec<_>>()
        ));
        scripts.push_str("                backgroundColor: 'rgba(102, 126, 234, 0.8)'\n");
        scripts.push_str("            }]\n");
        scripts.push_str("        },\n");
        scripts.push_str("        options: { responsive: true }\n");
        scripts.push_str("    });\n");

        scripts.push_str("});\n</script>\n");
        scripts
    }

    fn generate_codec_sections(&self) -> String {
        let mut html = String::new();

        for codec_result in &self.results.codec_results {
            html.push_str(&format!(
                r#"        <div class="codec-section">
            <h2 class="codec-title">{:?} Results</h2>
"#,
                codec_result.codec_id
            ));

            if let Some(preset) = &codec_result.preset {
                html.push_str(&format!(
                    r#"            <p><span class="badge speed">Preset: {}</span></p>
"#,
                    preset
                ));
            }

            html.push_str("            <table>\n");
            html.push_str("                <thead><tr>\n");
            html.push_str("                    <th>Sequence</th>\n");
            html.push_str("                    <th>Encoding FPS</th>\n");
            html.push_str("                    <th>Decoding FPS</th>\n");
            html.push_str("                    <th>File Size (MB)</th>\n");
            html.push_str("                    <th>PSNR (dB)</th>\n");
            html.push_str("                    <th>SSIM</th>\n");
            html.push_str("                </tr></thead>\n");
            html.push_str("                <tbody>\n");

            for seq in &codec_result.sequence_results {
                html.push_str(&format!(
                    r#"                <tr>
                    <td>{}</td>
                    <td class="metric">{:.2}</td>
                    <td class="metric">{:.2}</td>
                    <td>{:.2}</td>
                    <td>{}</td>
                    <td>{}</td>
                </tr>
"#,
                    seq.sequence_name,
                    seq.encoding_fps,
                    seq.decoding_fps,
                    seq.file_size_bytes as f64 / 1_000_000.0,
                    seq.metrics
                        .psnr
                        .map_or("N/A".to_string(), |p| format!("{p:.2}")),
                    seq.metrics
                        .ssim
                        .map_or("N/A".to_string(), |s| format!("{s:.4}")),
                ));
            }

            html.push_str("                </tbody>\n");
            html.push_str("            </table>\n");
            html.push_str("        </div>\n");
        }

        html
    }

    fn generate_statistics(&self) -> String {
        let mut html = String::from(
            r#"        <div class="codec-section">
            <h2 class="codec-title">Detailed Statistics</h2>
"#,
        );

        for codec_result in &self.results.codec_results {
            html.push_str(&format!(
                r#"            <h3>{:?} Statistics</h3>
            <div class="stats-grid">
                <div class="stat-box">
                    <div class="stat-label">Mean Encoding FPS</div>
                    <div class="stat-value">{:.2}</div>
                </div>
                <div class="stat-box">
                    <div class="stat-label">Median Encoding FPS</div>
                    <div class="stat-value">{:.2}</div>
                </div>
                <div class="stat-box">
                    <div class="stat-label">Std Dev Encoding FPS</div>
                    <div class="stat-value">{:.2}</div>
                </div>
                <div class="stat-box">
                    <div class="stat-label">Mean PSNR</div>
                    <div class="stat-value">{}</div>
                </div>
                <div class="stat-box">
                    <div class="stat-label">Mean SSIM</div>
                    <div class="stat-value">{}</div>
                </div>
                <div class="stat-box">
                    <div class="stat-label">Mean File Size (MB)</div>
                    <div class="stat-value">{:.2}</div>
                </div>
            </div>
"#,
                codec_result.codec_id,
                codec_result.statistics.mean_encoding_fps,
                codec_result.statistics.median_encoding_fps,
                codec_result.statistics.std_dev_encoding_fps,
                codec_result
                    .statistics
                    .mean_psnr
                    .map_or("N/A".to_string(), |p| format!("{p:.2}")),
                codec_result
                    .statistics
                    .mean_ssim
                    .map_or("N/A".to_string(), |s| format!("{s:.4}")),
                codec_result.statistics.mean_file_size as f64 / 1_000_000.0,
            ));
        }

        html.push_str("        </div>\n");
        html
    }

    fn generate_footer(&self) -> String {
        String::from(
            r#"        <footer>
            <p>Generated by OxiMedia Benchmark Suite</p>
            <p>Comprehensive codec performance analysis tool</p>
        </footer>
    </div>
</body>
</html>
"#,
        )
    }

    /// Write the advanced report to a file.
    ///
    /// # Errors
    ///
    /// Returns an error if the file cannot be written.
    pub fn write_to_file(&self, path: impl AsRef<std::path::Path>) -> BenchResult<()> {
        let html = self.generate();
        std::fs::write(path, html)?;
        Ok(())
    }
}

/// JSON report with detailed structure.
pub struct JsonReport<'a> {
    results: &'a BenchmarkResults,
    pretty: bool,
}

impl<'a> JsonReport<'a> {
    /// Create a new JSON report.
    #[must_use]
    pub fn new(results: &'a BenchmarkResults) -> Self {
        Self {
            results,
            pretty: true,
        }
    }

    /// Set whether to pretty-print JSON.
    #[must_use]
    pub fn with_pretty(mut self, pretty: bool) -> Self {
        self.pretty = pretty;
        self
    }

    /// Write the JSON report.
    ///
    /// # Errors
    ///
    /// Returns an error if writing fails.
    pub fn write_to_file(&self, path: impl AsRef<std::path::Path>) -> BenchResult<()> {
        let file = std::fs::File::create(path)?;
        if self.pretty {
            serde_json::to_writer_pretty(file, self.results)?;
        } else {
            serde_json::to_writer(file, self.results)?;
        }
        Ok(())
    }
}

/// Markdown report generator.
pub struct MarkdownReport<'a> {
    results: &'a BenchmarkResults,
}

impl<'a> MarkdownReport<'a> {
    /// Create a new Markdown report.
    #[must_use]
    pub fn new(results: &'a BenchmarkResults) -> Self {
        Self { results }
    }

    /// Generate markdown content.
    #[must_use]
    pub fn generate(&self) -> String {
        let mut md = String::new();

        md.push_str("# OxiMedia Codec Benchmark Results\n\n");
        md.push_str(&format!("**Generated:** {}\n\n", self.results.timestamp));
        md.push_str(&format!(
            "**Duration:** {:.2}s\n\n",
            self.results.total_duration.as_secs_f64()
        ));

        md.push_str("## Summary\n\n");
        md.push_str(&format!(
            "- **Codecs Tested:** {}\n",
            self.results.codec_results.len()
        ));
        md.push_str(&format!(
            "- **Total Benchmarks:** {}\n\n",
            self.results
                .codec_results
                .iter()
                .map(|r| r.sequence_results.len())
                .sum::<usize>()
        ));

        for codec_result in &self.results.codec_results {
            md.push_str(&format!("## {:?} Results\n\n", codec_result.codec_id));

            if let Some(preset) = &codec_result.preset {
                md.push_str(&format!("**Preset:** {}\n\n", preset));
            }

            md.push_str(
                "| Sequence | Encoding FPS | Decoding FPS | File Size (MB) | PSNR (dB) | SSIM |\n",
            );
            md.push_str(
                "|----------|--------------|--------------|----------------|-----------|------|\n",
            );

            for seq in &codec_result.sequence_results {
                md.push_str(&format!(
                    "| {} | {:.2} | {:.2} | {:.2} | {} | {} |\n",
                    seq.sequence_name,
                    seq.encoding_fps,
                    seq.decoding_fps,
                    seq.file_size_bytes as f64 / 1_000_000.0,
                    seq.metrics
                        .psnr
                        .map_or("N/A".to_string(), |p| format!("{p:.2}")),
                    seq.metrics
                        .ssim
                        .map_or("N/A".to_string(), |s| format!("{s:.4}")),
                ));
            }

            md.push('\n');
        }

        md
    }

    /// Write the Markdown report.
    ///
    /// # Errors
    ///
    /// Returns an error if writing fails.
    pub fn write_to_file(&self, path: impl AsRef<std::path::Path>) -> BenchResult<()> {
        let md = self.generate();
        std::fs::write(path, md)?;
        Ok(())
    }
}

#[cfg(test)]
mod extended_tests {
    use super::*;
    use crate::{BenchmarkConfig, BenchmarkResults};
    use std::time::Duration;

    fn create_test_results() -> BenchmarkResults {
        BenchmarkResults {
            codec_results: vec![],
            timestamp: "2024-01-01T00:00:00Z".to_string(),
            total_duration: Duration::from_secs(100),
            config: BenchmarkConfig::default(),
        }
    }

    #[test]
    fn test_advanced_html_report() {
        let results = create_test_results();
        let report = AdvancedHtmlReport::new(&results)
            .with_charts(true)
            .with_detailed_stats(true);

        let html = report.generate();
        assert!(html.contains("<!DOCTYPE html>"));
        assert!(html.contains("Advanced Performance Analysis"));
    }

    #[test]
    fn test_json_report() {
        let results = create_test_results();
        let _report = JsonReport::new(&results).with_pretty(true);
    }

    #[test]
    fn test_markdown_report() {
        let results = create_test_results();
        let report = MarkdownReport::new(&results);
        let md = report.generate();

        assert!(md.contains("# OxiMedia Codec Benchmark Results"));
        assert!(md.contains("## Summary"));
    }
}
