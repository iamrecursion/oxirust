//! HTML report generation.

use super::generate::{Report, ReportSection};

/// HTML reporter.
#[derive(Debug)]
pub struct HtmlReporter {
    style: String,
}

impl HtmlReporter {
    /// Create a new HTML reporter.
    pub fn new() -> Self {
        Self {
            style: Self::default_style(),
        }
    }

    /// Get default CSS style.
    fn default_style() -> String {
        r#"
body {
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
    margin: 0;
    padding: 20px;
    background: #f5f5f5;
}
.container {
    max-width: 1200px;
    margin: 0 auto;
    background: white;
    padding: 30px;
    border-radius: 8px;
    box-shadow: 0 2px 10px rgba(0,0,0,0.1);
}
h1 {
    color: #333;
    border-bottom: 3px solid #4CAF50;
    padding-bottom: 10px;
}
h2 {
    color: #555;
    margin-top: 30px;
}
.meta {
    color: #777;
    font-size: 14px;
    margin-bottom: 20px;
}
.section {
    margin: 20px 0;
    padding: 15px;
    background: #f9f9f9;
    border-left: 4px solid #4CAF50;
}
.subsection {
    margin: 15px 0 15px 20px;
    padding: 10px;
    background: white;
    border-left: 3px solid #2196F3;
}
pre {
    background: #272822;
    color: #f8f8f2;
    padding: 15px;
    border-radius: 4px;
    overflow-x: auto;
}
        "#
        .to_string()
    }

    /// Generate HTML report.
    pub fn generate(&self, report: &Report) -> String {
        let mut html = String::new();

        html.push_str("<!DOCTYPE html>\n<html>\n<head>\n");
        html.push_str("<meta charset=\"UTF-8\">\n");
        html.push_str(&format!("<title>{}</title>\n", report.title));
        html.push_str("<style>\n");
        html.push_str(&self.style);
        html.push_str("</style>\n");
        html.push_str("</head>\n<body>\n");

        html.push_str("<div class=\"container\">\n");
        html.push_str(&format!("<h1>{}</h1>\n", report.title));

        html.push_str("<div class=\"meta\">\n");
        html.push_str(&format!("Generated: {}<br>\n", report.timestamp));
        html.push_str(&format!("Duration: {:?}\n", report.duration));
        html.push_str("</div>\n");

        for section in &report.sections {
            self.render_section(&mut html, section, false);
        }

        html.push_str("</div>\n");
        html.push_str("</body>\n</html>");

        html
    }

    /// Render a section to HTML.
    fn render_section(&self, html: &mut String, section: &ReportSection, is_subsection: bool) {
        let class = if is_subsection {
            "subsection"
        } else {
            "section"
        };

        html.push_str(&format!("<div class=\"{}\">\n", class));
        html.push_str(&format!("<h2>{}</h2>\n", section.title));

        if !section.content.is_empty() {
            html.push_str("<pre>\n");
            html.push_str(&html_escape(&section.content));
            html.push_str("</pre>\n");
        }

        for subsection in &section.subsections {
            self.render_section(html, subsection, true);
        }

        html.push_str("</div>\n");
    }
}

impl Default for HtmlReporter {
    fn default() -> Self {
        Self::new()
    }
}

/// Escape HTML special characters.
fn html_escape(s: &str) -> String {
    s.replace('&', "&amp;")
        .replace('<', "&lt;")
        .replace('>', "&gt;")
        .replace('"', "&quot;")
        .replace('\'', "&#39;")
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Duration;

    #[test]
    fn test_html_reporter() {
        let reporter = HtmlReporter::new();
        let report = Report {
            title: "Test Report".to_string(),
            timestamp: "2024-01-01T00:00:00Z".to_string(),
            duration: Duration::from_secs(1),
            sections: vec![],
        };

        let html = reporter.generate(&report);
        assert!(html.contains("<!DOCTYPE html>"));
        assert!(html.contains("Test Report"));
    }

    #[test]
    fn test_html_escape() {
        let escaped = html_escape("<test & 'quotes'>");
        assert!(escaped.contains("&lt;"));
        assert!(escaped.contains("&amp;"));
        assert!(escaped.contains("&#39;"));
    }

    #[test]
    fn test_html_structure() {
        let reporter = HtmlReporter::default();
        let section = ReportSection::new("Test".to_string(), "Content".to_string());
        let report = Report {
            title: "Test".to_string(),
            timestamp: "2024-01-01T00:00:00Z".to_string(),
            duration: Duration::from_secs(1),
            sections: vec![section],
        };

        let html = reporter.generate(&report);
        assert!(html.contains("<div class=\"section\">"));
    }
}
