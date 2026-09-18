//! JSON report export.

use super::generate::Report;
use crate::Result;

/// JSON reporter.
#[derive(Debug)]
pub struct JsonReporter {
    pretty: bool,
}

impl JsonReporter {
    /// Create a new JSON reporter.
    pub fn new(pretty: bool) -> Self {
        Self { pretty }
    }

    /// Generate JSON report.
    pub fn generate(&self, report: &Report) -> Result<String> {
        if self.pretty {
            Ok(serde_json::to_string_pretty(report)?)
        } else {
            Ok(serde_json::to_string(report)?)
        }
    }

    /// Save report to file.
    pub fn save_to_file(&self, report: &Report, path: &std::path::Path) -> Result<()> {
        let json = self.generate(report)?;
        std::fs::write(path, json)?;
        Ok(())
    }
}

impl Default for JsonReporter {
    fn default() -> Self {
        Self::new(true)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Duration;

    #[test]
    fn test_json_reporter() {
        let reporter = JsonReporter::new(false);
        let report = Report {
            title: "Test".to_string(),
            timestamp: "2024-01-01T00:00:00Z".to_string(),
            duration: Duration::from_secs(1),
            sections: vec![],
        };

        let json = reporter.generate(&report).expect("should succeed in test");
        assert!(json.contains("Test"));
    }

    #[test]
    fn test_pretty_json() {
        let reporter = JsonReporter::new(true);
        let report = Report {
            title: "Test".to_string(),
            timestamp: "2024-01-01T00:00:00Z".to_string(),
            duration: Duration::from_secs(1),
            sections: vec![],
        };

        let json = reporter.generate(&report).expect("should succeed in test");
        assert!(json.contains('\n')); // Pretty-printed
    }

    #[test]
    fn test_default_reporter() {
        let reporter = JsonReporter::default();
        assert!(reporter.pretty);
    }
}
