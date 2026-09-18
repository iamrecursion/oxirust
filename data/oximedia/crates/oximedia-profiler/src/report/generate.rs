//! Report generation.

use serde::{Deserialize, Serialize};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

/// Profiling report.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Report {
    /// Report title.
    pub title: String,

    /// Timestamp.
    pub timestamp: String,

    /// Total duration.
    pub duration: Duration,

    /// Sections.
    pub sections: Vec<ReportSection>,
}

/// Report section.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReportSection {
    /// Section title.
    pub title: String,

    /// Section content.
    pub content: String,

    /// Subsections.
    pub subsections: Vec<ReportSection>,
}

impl ReportSection {
    /// Create a new report section.
    pub fn new(title: String, content: String) -> Self {
        Self {
            title,
            content,
            subsections: Vec::new(),
        }
    }

    /// Add a subsection.
    pub fn add_subsection(&mut self, section: ReportSection) {
        self.subsections.push(section);
    }
}

/// Report generator.
#[derive(Debug)]
pub struct ReportGenerator {
    title: String,
    start_time: Instant,
    sections: Vec<ReportSection>,
}

impl ReportGenerator {
    /// Create a new report generator.
    pub fn new(title: String) -> Self {
        Self {
            title,
            start_time: Instant::now(),
            sections: Vec::new(),
        }
    }

    /// Add a section.
    pub fn add_section(&mut self, section: ReportSection) {
        self.sections.push(section);
    }

    /// Add a simple section.
    pub fn add_simple_section(&mut self, title: String, content: String) {
        self.sections.push(ReportSection::new(title, content));
    }

    /// Generate the report.
    pub fn generate(&self) -> Report {
        Report {
            title: self.title.clone(),
            timestamp: rfc3339_now(),
            duration: self.start_time.elapsed(),
            sections: self.sections.clone(),
        }
    }

    /// Generate a text report.
    pub fn generate_text(&self) -> String {
        let report = self.generate();
        let mut text = String::new();

        text.push_str(&format!("=== {} ===\n\n", report.title));
        text.push_str(&format!("Generated: {}\n", report.timestamp));
        text.push_str(&format!("Duration: {:?}\n\n", report.duration));

        for section in &report.sections {
            self.render_section(&mut text, section, 0);
        }

        text
    }

    /// Render a section to text.
    fn render_section(&self, text: &mut String, section: &ReportSection, level: usize) {
        let indent = "  ".repeat(level);

        text.push_str(&format!("{}{}:\n", indent, section.title));
        if !section.content.is_empty() {
            for line in section.content.lines() {
                text.push_str(&format!("{}  {}\n", indent, line));
            }
        }

        for subsection in &section.subsections {
            self.render_section(text, subsection, level + 1);
        }

        text.push('\n');
    }
}

impl Default for ReportGenerator {
    fn default() -> Self {
        Self::new("OxiMedia Profiler Report".to_string())
    }
}

/// Return the current UTC time formatted as RFC 3339 / ISO 8601 using only
/// `std::time`. No external dependencies are required.
fn rfc3339_now() -> String {
    let secs = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or(Duration::ZERO)
        .as_secs();
    unix_secs_to_rfc3339(secs)
}

/// Convert a Unix timestamp (seconds since epoch) to an RFC 3339 string.
///
/// The output format is `YYYY-MM-DDTHH:MM:SSZ` (UTC, second precision).
fn unix_secs_to_rfc3339(secs: u64) -> String {
    // Days since Unix epoch (1970-01-01) split into year/month/day.
    let days = secs / 86_400;
    let time = secs % 86_400;

    let hh = time / 3_600;
    let mm = (time % 3_600) / 60;
    let ss = time % 60;

    // Compute Gregorian calendar date from day count using the algorithm from
    // "Calendrical Calculations" (Dershowitz & Reingold, public-domain variant).
    let z = days + 719_468;
    let era = z / 146_097;
    let doe = z % 146_097;
    let yoe = (doe - doe / 1_460 + doe / 36_524 - doe / 146_096) / 365;
    let y = yoe + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = doy - (153 * mp + 2) / 5 + 1;
    let m = if mp < 10 { mp + 3 } else { mp - 9 };
    let y = if m <= 2 { y + 1 } else { y };

    format!("{y:04}-{m:02}-{d:02}T{hh:02}:{mm:02}:{ss:02}Z")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_report_generator() {
        let mut generator = ReportGenerator::new("Test Report".to_string());
        generator.add_simple_section("Section 1".to_string(), "Content 1".to_string());

        let report = generator.generate();
        assert_eq!(report.title, "Test Report");
        assert_eq!(report.sections.len(), 1);
    }

    #[test]
    fn test_report_section() {
        let mut section = ReportSection::new("Main".to_string(), "Content".to_string());
        section.add_subsection(ReportSection::new(
            "Sub".to_string(),
            "Sub content".to_string(),
        ));

        assert_eq!(section.subsections.len(), 1);
    }

    #[test]
    fn test_text_generation() {
        let mut generator = ReportGenerator::new("Test".to_string());
        generator.add_simple_section("Test Section".to_string(), "Test content".to_string());

        let text = generator.generate_text();
        assert!(text.contains("Test"));
        assert!(text.contains("Test Section"));
    }

    #[test]
    fn test_unix_secs_to_rfc3339_epoch() {
        // Unix epoch = 1970-01-01T00:00:00Z
        assert_eq!(unix_secs_to_rfc3339(0), "1970-01-01T00:00:00Z");
    }

    #[test]
    fn test_unix_secs_to_rfc3339_known_date() {
        // 2024-01-01T00:00:00Z = 1_704_067_200 seconds
        assert_eq!(unix_secs_to_rfc3339(1_704_067_200), "2024-01-01T00:00:00Z");
    }

    #[test]
    fn test_rfc3339_now_has_correct_format() {
        let ts = rfc3339_now();
        // Format: YYYY-MM-DDTHH:MM:SSZ — 20 chars
        assert_eq!(ts.len(), 20, "unexpected timestamp length: {ts}");
        assert!(ts.ends_with('Z'), "timestamp should end with Z: {ts}");
        assert_eq!(&ts[10..11], "T", "position 10 should be T: {ts}");
        // Year should be reasonable (after 2020)
        let year: u64 = ts[..4].parse().unwrap_or(0);
        assert!(year >= 2020, "year should be >= 2020: {ts}");
    }
}
