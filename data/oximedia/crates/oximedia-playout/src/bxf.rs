//! BXF (Broadcast eXchange Format) integration
//!
//! Provides schedule import/export and traffic system integration
//! using the SMPTE BXF standard.

use crate::Result;
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::path::Path;
use tokio::fs;
use tracing::{debug, info};
use uuid::Uuid;

/// BXF configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BxfConfig {
    /// BXF version
    pub version: String,

    /// Import directory
    pub import_dir: std::path::PathBuf,

    /// Export directory
    pub export_dir: std::path::PathBuf,

    /// Auto-import enabled
    pub auto_import: bool,

    /// Auto-export enabled
    pub auto_export: bool,

    /// Organization ID
    pub organization_id: String,

    /// System ID
    pub system_id: String,
}

impl Default for BxfConfig {
    fn default() -> Self {
        Self {
            version: "5.0".to_string(),
            import_dir: std::path::PathBuf::from("/var/oximedia/bxf/import"),
            export_dir: std::path::PathBuf::from("/var/oximedia/bxf/export"),
            auto_import: true,
            auto_export: true,
            organization_id: "OXIMEDIA".to_string(),
            system_id: "PLAYOUT01".to_string(),
        }
    }
}

/// BXF schedule
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BxfSchedule {
    /// Schedule ID
    pub id: Uuid,

    /// Schedule date
    pub date: DateTime<Utc>,

    /// Channel ID
    pub channel_id: String,

    /// Events in the schedule
    pub events: Vec<BxfEvent>,

    /// Created timestamp
    pub created_at: DateTime<Utc>,
}

/// BXF event
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BxfEvent {
    /// Event ID
    pub id: String,

    /// House number
    pub house_number: String,

    /// Title
    pub title: String,

    /// Start time
    pub start_time: DateTime<Utc>,

    /// Duration in milliseconds
    pub duration_ms: u64,

    /// Event type
    pub event_type: BxfEventType,

    /// ISCI code (for commercials)
    pub isci_code: Option<String>,

    /// Segment number
    pub segment_number: Option<u32>,

    /// Custom metadata
    pub metadata: std::collections::HashMap<String, String>,
}

/// BXF event types
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
pub enum BxfEventType {
    /// Program content
    Program,
    /// Commercial
    Commercial,
    /// Promo
    Promo,
    /// PSA (Public Service Announcement)
    Psa,
    /// Filler
    Filler,
    /// Break
    Break,
}

/// BXF importer
pub struct BxfImporter {
    #[allow(dead_code)]
    config: BxfConfig,
}

impl BxfImporter {
    /// Create new BXF importer
    pub fn new(config: BxfConfig) -> Self {
        Self { config }
    }

    /// Import schedule from BXF file
    pub async fn import_schedule(&self, file_path: &Path) -> Result<BxfSchedule> {
        info!("Importing BXF schedule from: {}", file_path.display());

        let content = fs::read_to_string(file_path).await?;
        let schedule = self.parse_bxf(&content)?;

        debug!("Imported {} events", schedule.events.len());

        Ok(schedule)
    }

    /// Parse BXF XML content
    fn parse_bxf(&self, _content: &str) -> Result<BxfSchedule> {
        // In real implementation, this would parse actual BXF XML
        // For now, return a sample schedule

        Ok(BxfSchedule {
            id: Uuid::new_v4(),
            date: Utc::now(),
            channel_id: "CH01".to_string(),
            events: vec![],
            created_at: Utc::now(),
        })
    }

    /// Validate BXF schedule
    pub fn validate_schedule(&self, schedule: &BxfSchedule) -> Vec<BxfValidationIssue> {
        let mut issues = Vec::new();

        // Check for overlapping events
        for i in 0..schedule.events.len() {
            for j in (i + 1)..schedule.events.len() {
                let event1 = &schedule.events[i];
                let event2 = &schedule.events[j];

                let end1 =
                    event1.start_time + chrono::Duration::milliseconds(event1.duration_ms as i64);
                let end2 =
                    event2.start_time + chrono::Duration::milliseconds(event2.duration_ms as i64);

                if event1.start_time < end2 && event2.start_time < end1 {
                    issues.push(BxfValidationIssue {
                        severity: ValidationSeverity::Error,
                        message: format!("Events {} and {} overlap", event1.id, event2.id),
                        event_id: Some(event1.id.clone()),
                    });
                }
            }
        }

        // Check for missing house numbers
        for event in &schedule.events {
            if event.house_number.is_empty() {
                issues.push(BxfValidationIssue {
                    severity: ValidationSeverity::Warning,
                    message: format!("Event {} missing house number", event.id),
                    event_id: Some(event.id.clone()),
                });
            }
        }

        // Check for missing ISCI codes on commercials
        for event in &schedule.events {
            if event.event_type == BxfEventType::Commercial && event.isci_code.is_none() {
                issues.push(BxfValidationIssue {
                    severity: ValidationSeverity::Warning,
                    message: format!("Commercial {} missing ISCI code", event.id),
                    event_id: Some(event.id.clone()),
                });
            }
        }

        issues
    }
}

/// BXF validation issue
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BxfValidationIssue {
    pub severity: ValidationSeverity,
    pub message: String,
    pub event_id: Option<String>,
}

/// Validation severity
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
pub enum ValidationSeverity {
    Info,
    Warning,
    Error,
}

/// BXF exporter
pub struct BxfExporter {
    config: BxfConfig,
}

impl BxfExporter {
    /// Create new BXF exporter
    pub fn new(config: BxfConfig) -> Self {
        Self { config }
    }

    /// Export schedule to BXF file
    pub async fn export_schedule(&self, schedule: &BxfSchedule, file_path: &Path) -> Result<()> {
        info!("Exporting BXF schedule to: {}", file_path.display());

        let xml = self.generate_bxf_xml(schedule);

        fs::write(file_path, xml).await?;

        debug!("Exported {} events", schedule.events.len());

        Ok(())
    }

    /// Generate BXF XML
    fn generate_bxf_xml(&self, schedule: &BxfSchedule) -> String {
        let mut xml = String::new();

        xml.push_str("<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n");
        xml.push_str(&format!("<BXF version=\"{}\">\n", self.config.version));
        xml.push_str("  <Schedule>\n");
        xml.push_str(&format!("    <ScheduleID>{}</ScheduleID>\n", schedule.id));
        xml.push_str(&format!(
            "    <Date>{}</Date>\n",
            schedule.date.to_rfc3339()
        ));
        xml.push_str(&format!(
            "    <ChannelID>{}</ChannelID>\n",
            schedule.channel_id
        ));
        xml.push_str("    <Events>\n");

        for event in &schedule.events {
            xml.push_str("      <Event>\n");
            xml.push_str(&format!("        <EventID>{}</EventID>\n", event.id));
            xml.push_str(&format!(
                "        <HouseNumber>{}</HouseNumber>\n",
                event.house_number
            ));
            xml.push_str(&format!(
                "        <Title>{}</Title>\n",
                Self::escape_xml(&event.title)
            ));
            xml.push_str(&format!(
                "        <StartTime>{}</StartTime>\n",
                event.start_time.to_rfc3339()
            ));
            xml.push_str(&format!(
                "        <Duration>{}</Duration>\n",
                event.duration_ms
            ));
            xml.push_str(&format!(
                "        <EventType>{:?}</EventType>\n",
                event.event_type
            ));

            if let Some(isci) = &event.isci_code {
                xml.push_str(&format!("        <ISCICode>{isci}</ISCICode>\n"));
            }

            xml.push_str("      </Event>\n");
        }

        xml.push_str("    </Events>\n");
        xml.push_str("  </Schedule>\n");
        xml.push_str("</BXF>\n");

        xml
    }

    /// Escape XML special characters
    fn escape_xml(s: &str) -> String {
        s.replace('&', "&amp;")
            .replace('<', "&lt;")
            .replace('>', "&gt;")
            .replace('"', "&quot;")
            .replace('\'', "&apos;")
    }

    /// Export as-run log to BXF
    pub async fn export_asrun(&self, entries: &[AsRunEntry], file_path: &Path) -> Result<()> {
        info!("Exporting as-run log to BXF: {}", file_path.display());

        let xml = self.generate_asrun_bxf(entries);

        fs::write(file_path, xml).await?;

        debug!("Exported {} as-run entries", entries.len());

        Ok(())
    }

    /// Generate as-run BXF XML
    fn generate_asrun_bxf(&self, entries: &[AsRunEntry]) -> String {
        let mut xml = String::new();

        xml.push_str("<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n");
        xml.push_str(&format!("<BXF version=\"{}\">\n", self.config.version));
        xml.push_str("  <AsRun>\n");
        xml.push_str(&format!(
            "    <OrganizationID>{}</OrganizationID>\n",
            self.config.organization_id
        ));
        xml.push_str(&format!(
            "    <SystemID>{}</SystemID>\n",
            self.config.system_id
        ));
        xml.push_str(&format!(
            "    <GeneratedAt>{}</GeneratedAt>\n",
            Utc::now().to_rfc3339()
        ));
        xml.push_str("    <Entries>\n");

        for entry in entries {
            xml.push_str("      <Entry>\n");
            xml.push_str(&format!("        <EntryID>{}</EntryID>\n", entry.id));
            xml.push_str(&format!(
                "        <ContentID>{}</ContentID>\n",
                entry.content_id
            ));
            xml.push_str(&format!(
                "        <ScheduledStart>{}</ScheduledStart>\n",
                entry.scheduled_start.to_rfc3339()
            ));
            xml.push_str(&format!(
                "        <ActualStart>{}</ActualStart>\n",
                entry.actual_start.to_rfc3339()
            ));
            xml.push_str(&format!(
                "        <ScheduledDuration>{}</ScheduledDuration>\n",
                entry.scheduled_duration_ms
            ));
            xml.push_str(&format!(
                "        <ActualDuration>{}</ActualDuration>\n",
                entry.actual_duration_ms
            ));
            xml.push_str(&format!("        <Status>{:?}</Status>\n", entry.status));
            xml.push_str("      </Entry>\n");
        }

        xml.push_str("    </Entries>\n");
        xml.push_str("  </AsRun>\n");
        xml.push_str("</BXF>\n");

        xml
    }
}

/// Simplified as-run entry for BXF export
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AsRunEntry {
    pub id: Uuid,
    pub content_id: String,
    pub scheduled_start: DateTime<Utc>,
    pub actual_start: DateTime<Utc>,
    pub scheduled_duration_ms: u64,
    pub actual_duration_ms: u64,
    pub status: AsRunStatus,
}

/// As-run status
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
pub enum AsRunStatus {
    Played,
    Skipped,
    Failed,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_bxf_config_default() {
        let config = BxfConfig::default();
        assert_eq!(config.version, "5.0");
        assert!(config.auto_import);
        assert!(config.auto_export);
    }

    #[test]
    fn test_bxf_event_type_equality() {
        assert_eq!(BxfEventType::Program, BxfEventType::Program);
        assert_ne!(BxfEventType::Program, BxfEventType::Commercial);
    }

    #[test]
    fn test_validation_severity_equality() {
        assert_eq!(ValidationSeverity::Error, ValidationSeverity::Error);
        assert_ne!(ValidationSeverity::Error, ValidationSeverity::Warning);
    }

    #[test]
    fn test_bxf_importer_creation() {
        let config = BxfConfig::default();
        let importer = BxfImporter::new(config);
        assert_eq!(importer.config.version, "5.0");
    }

    #[test]
    fn test_bxf_exporter_creation() {
        let config = BxfConfig::default();
        let exporter = BxfExporter::new(config);
        assert_eq!(exporter.config.version, "5.0");
    }

    #[test]
    fn test_xml_escaping() {
        let input = "Title with <special> & \"characters\"";
        let escaped = BxfExporter::escape_xml(input);
        assert!(escaped.contains("&lt;"));
        assert!(escaped.contains("&gt;"));
        assert!(escaped.contains("&amp;"));
    }

    #[test]
    fn test_schedule_validation_empty() {
        let config = BxfConfig::default();
        let importer = BxfImporter::new(config);

        let schedule = BxfSchedule {
            id: Uuid::new_v4(),
            date: Utc::now(),
            channel_id: "CH01".to_string(),
            events: vec![],
            created_at: Utc::now(),
        };

        let issues = importer.validate_schedule(&schedule);
        assert!(issues.is_empty());
    }

    #[test]
    fn test_schedule_validation_missing_house_number() {
        let config = BxfConfig::default();
        let importer = BxfImporter::new(config);

        let event = BxfEvent {
            id: "E001".to_string(),
            house_number: "".to_string(), // Missing
            title: "Test Event".to_string(),
            start_time: Utc::now(),
            duration_ms: 60000,
            event_type: BxfEventType::Program,
            isci_code: None,
            segment_number: None,
            metadata: std::collections::HashMap::new(),
        };

        let schedule = BxfSchedule {
            id: Uuid::new_v4(),
            date: Utc::now(),
            channel_id: "CH01".to_string(),
            events: vec![event],
            created_at: Utc::now(),
        };

        let issues = importer.validate_schedule(&schedule);
        assert!(!issues.is_empty());
        assert_eq!(issues[0].severity, ValidationSeverity::Warning);
    }

    #[test]
    fn test_bxf_xml_generation() {
        let config = BxfConfig::default();
        let exporter = BxfExporter::new(config);

        let schedule = BxfSchedule {
            id: Uuid::new_v4(),
            date: Utc::now(),
            channel_id: "CH01".to_string(),
            events: vec![BxfEvent {
                id: "E001".to_string(),
                house_number: "H12345".to_string(),
                title: "Test Program".to_string(),
                start_time: Utc::now(),
                duration_ms: 1800000,
                event_type: BxfEventType::Program,
                isci_code: None,
                segment_number: Some(1),
                metadata: std::collections::HashMap::new(),
            }],
            created_at: Utc::now(),
        };

        let xml = exporter.generate_bxf_xml(&schedule);
        assert!(xml.contains("<BXF version=\"5.0\">"));
        assert!(xml.contains("<EventID>E001</EventID>"));
        assert!(xml.contains("<HouseNumber>H12345</HouseNumber>"));
    }

    #[test]
    fn test_asrun_bxf_generation() {
        let config = BxfConfig::default();
        let exporter = BxfExporter::new(config);

        let entries = vec![AsRunEntry {
            id: Uuid::new_v4(),
            content_id: "C001".to_string(),
            scheduled_start: Utc::now(),
            actual_start: Utc::now(),
            scheduled_duration_ms: 60000,
            actual_duration_ms: 60050,
            status: AsRunStatus::Played,
        }];

        let xml = exporter.generate_asrun_bxf(&entries);
        assert!(xml.contains("<AsRun>"));
        assert!(xml.contains("<ContentID>C001</ContentID>"));
        assert!(xml.contains("<Status>Played</Status>"));
    }

    #[test]
    fn test_asrun_status_equality() {
        assert_eq!(AsRunStatus::Played, AsRunStatus::Played);
        assert_ne!(AsRunStatus::Played, AsRunStatus::Failed);
    }

    #[test]
    fn test_bxf_event_creation() {
        let event = BxfEvent {
            id: "E001".to_string(),
            house_number: "H001".to_string(),
            title: "Test".to_string(),
            start_time: Utc::now(),
            duration_ms: 30000,
            event_type: BxfEventType::Commercial,
            isci_code: Some("ABC1234H".to_string()),
            segment_number: None,
            metadata: std::collections::HashMap::new(),
        };

        assert_eq!(event.event_type, BxfEventType::Commercial);
        assert_eq!(event.isci_code, Some("ABC1234H".to_string()));
    }
}
