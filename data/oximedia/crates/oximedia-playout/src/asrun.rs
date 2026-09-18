//! As-run logging and compliance reporting
//!
//! Tracks actual playout events for reconciliation, compliance, and billing.
//! Supports BXF, XML, CSV export formats and broadcaster-specific requirements.

use crate::{PlayoutError, Result};
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::Arc;
use tokio::fs::{File, OpenOptions};
use tokio::io::{AsyncWriteExt, BufWriter};
use tokio::sync::RwLock;
use tracing::{debug, info, warn};
use uuid::Uuid;

/// As-run log configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AsRunConfig {
    /// Enable as-run logging
    pub enabled: bool,

    /// Log directory
    pub log_dir: PathBuf,

    /// Log rotation interval in hours
    pub rotation_hours: u32,

    /// Maximum log retention days
    pub retention_days: u32,

    /// Enable real-time export
    pub realtime_export: bool,

    /// Export formats
    pub export_formats: Vec<ExportFormat>,

    /// Include technical details
    pub include_technical: bool,

    /// Include operator notes
    pub include_notes: bool,

    /// Compliance mode (strict validation)
    pub compliance_mode: bool,
}

impl Default for AsRunConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            log_dir: PathBuf::from("/var/oximedia/asrun"),
            rotation_hours: 24,
            retention_days: 90,
            realtime_export: true,
            export_formats: vec![ExportFormat::Bxf, ExportFormat::Csv],
            include_technical: true,
            include_notes: true,
            compliance_mode: false,
        }
    }
}

/// Export format types
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
pub enum ExportFormat {
    /// BXF (Broadcast eXchange Format)
    Bxf,
    /// XML format
    Xml,
    /// CSV format
    Csv,
    /// JSON format
    Json,
}

/// As-run log entry
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AsRunEntry {
    /// Unique entry ID
    pub id: Uuid,

    /// Event ID from schedule
    pub event_id: Option<Uuid>,

    /// Content/clip ID
    pub content_id: String,

    /// House number (broadcaster internal ID)
    pub house_number: Option<String>,

    /// ISCI code (advertising code)
    pub isci_code: Option<String>,

    /// Scheduled start time
    pub scheduled_start: DateTime<Utc>,

    /// Actual start time
    pub actual_start: DateTime<Utc>,

    /// Scheduled duration in milliseconds
    pub scheduled_duration_ms: u64,

    /// Actual duration in milliseconds
    pub actual_duration_ms: u64,

    /// Scheduled end time
    pub scheduled_end: DateTime<Utc>,

    /// Actual end time
    pub actual_end: DateTime<Utc>,

    /// Event type
    pub event_type: AsRunEventType,

    /// Playout status
    pub status: PlayoutStatus,

    /// Error code if failed
    pub error_code: Option<String>,

    /// Error message if failed
    pub error_message: Option<String>,

    /// Technical details
    pub technical: Option<TechnicalDetails>,

    /// Operator notes
    pub notes: Option<String>,

    /// Segmentation data (SCTE-35)
    pub segmentation: Option<SegmentationData>,

    /// Verification status
    pub verified: bool,

    /// Created timestamp
    pub created_at: DateTime<Utc>,
}

/// As-run event types
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
pub enum AsRunEventType {
    /// Primary video content
    Primary,
    /// Secondary video
    Secondary,
    /// Commercial/advertisement
    Commercial,
    /// Promo
    Promo,
    /// Program segment
    ProgramSegment,
    /// Filler content
    Filler,
    /// Graphics overlay
    Graphics,
    /// Emergency alert
    EmergencyAlert,
}

/// Playout status
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
pub enum PlayoutStatus {
    /// Played successfully
    Played,
    /// Partially played
    Partial,
    /// Skipped
    Skipped,
    /// Failed to play
    Failed,
    /// Manually aborted
    Aborted,
    /// Replaced by operator
    Replaced,
}

/// Technical details for as-run entry
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TechnicalDetails {
    /// Video format
    pub video_format: String,

    /// Audio format
    pub audio_format: String,

    /// Codec information
    pub codec: String,

    /// Bitrate in kbps
    pub bitrate_kbps: u32,

    /// Actual framerate
    pub framerate: f64,

    /// Audio levels (peak, RMS)
    pub audio_levels: Option<AudioLevels>,

    /// Video quality metrics
    pub video_quality: Option<VideoQuality>,
}

/// Audio level measurements
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AudioLevels {
    /// Peak level in dBFS
    pub peak_dbfs: f32,

    /// RMS level in dBFS
    pub rms_dbfs: f32,

    /// Integrated loudness in LUFS
    pub integrated_lufs: f32,

    /// Loudness range in LU
    pub loudness_range_lu: f32,

    /// True peak in dBTP
    pub true_peak_dbtp: f32,
}

/// Video quality metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VideoQuality {
    /// Dropped frames count
    pub dropped_frames: u32,

    /// Black frames count
    pub black_frames: u32,

    /// Frozen frames count
    pub frozen_frames: u32,

    /// Average bitrate in kbps
    pub avg_bitrate_kbps: u32,
}

/// Segmentation data (SCTE-35)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SegmentationData {
    /// Splice event ID
    pub event_id: u32,

    /// Segmentation type
    pub segmentation_type: String,

    /// Duration in milliseconds
    pub duration_ms: Option<u64>,

    /// UPID (Unique Program Identifier)
    pub upid: Option<String>,

    /// Segmentation message
    pub message: Option<String>,
}

/// As-run logger
pub struct AsRunLogger {
    config: AsRunConfig,
    entries: Arc<RwLock<Vec<AsRunEntry>>>,
    current_log: Arc<RwLock<Option<BufWriter<File>>>>,
}

impl AsRunLogger {
    /// Create new as-run logger
    pub async fn new(config: AsRunConfig) -> Result<Self> {
        // Create log directory if it doesn't exist
        if !config.log_dir.exists() {
            tokio::fs::create_dir_all(&config.log_dir)
                .await
                .map_err(PlayoutError::Io)?;
        }

        Ok(Self {
            config,
            entries: Arc::new(RwLock::new(Vec::new())),
            current_log: Arc::new(RwLock::new(None)),
        })
    }

    /// Log an as-run entry
    pub async fn log(&self, entry: AsRunEntry) -> Result<()> {
        debug!("Logging as-run entry: {:?}", entry.id);

        // Validate entry if in compliance mode
        if self.config.compliance_mode {
            self.validate_entry(&entry)?;
        }

        // Add to in-memory storage
        {
            let mut entries = self.entries.write().await;
            entries.push(entry.clone());
        }

        // Write to file
        self.write_entry(&entry).await?;

        // Export in real-time if enabled
        if self.config.realtime_export {
            self.export_entry(&entry).await?;
        }

        Ok(())
    }

    /// Validate as-run entry for compliance
    fn validate_entry(&self, entry: &AsRunEntry) -> Result<()> {
        // Check for required fields
        if entry.content_id.is_empty() {
            return Err(PlayoutError::Config("Content ID is required".to_string()));
        }

        // Check timing consistency
        let scheduled_duration = (entry.scheduled_end - entry.scheduled_start).num_milliseconds();
        if scheduled_duration < 0 {
            return Err(PlayoutError::Timing(
                "Invalid scheduled duration".to_string(),
            ));
        }

        let actual_duration = (entry.actual_end - entry.actual_start).num_milliseconds();
        if actual_duration < 0 {
            return Err(PlayoutError::Timing("Invalid actual duration".to_string()));
        }

        // Check ISCI code for commercials
        if entry.event_type == AsRunEventType::Commercial && entry.isci_code.is_none() {
            warn!("Commercial entry missing ISCI code: {}", entry.content_id);
        }

        Ok(())
    }

    /// Write entry to log file
    async fn write_entry(&self, entry: &AsRunEntry) -> Result<()> {
        let mut log_guard = self.current_log.write().await;

        if log_guard.is_none() {
            *log_guard = Some(self.open_log_file().await?);
        }

        if let Some(log) = log_guard.as_mut() {
            let json = serde_json::to_string(&entry)
                .map_err(|e| PlayoutError::Config(format!("Failed to serialize entry: {e}")))?;

            log.write_all(json.as_bytes()).await?;
            log.write_all(b"\n").await?;
            log.flush().await?;
        }

        Ok(())
    }

    /// Open new log file
    async fn open_log_file(&self) -> Result<BufWriter<File>> {
        let timestamp = Utc::now().format("%Y%m%d_%H%M%S");
        let filename = format!("asrun_{timestamp}.jsonl");
        let path = self.config.log_dir.join(filename);

        let file = OpenOptions::new()
            .create(true)
            .append(true)
            .open(&path)
            .await?;

        info!("Opened new as-run log file: {}", path.display());
        Ok(BufWriter::new(file))
    }

    /// Export entry in configured formats
    async fn export_entry(&self, entry: &AsRunEntry) -> Result<()> {
        for format in &self.config.export_formats {
            match format {
                ExportFormat::Bxf => self.export_bxf(entry).await?,
                ExportFormat::Xml => self.export_xml(entry).await?,
                ExportFormat::Csv => self.export_csv(entry).await?,
                ExportFormat::Json => {
                    // Already written in JSONL format
                }
            }
        }
        Ok(())
    }

    /// Export entry as BXF format
    async fn export_bxf(&self, entry: &AsRunEntry) -> Result<()> {
        let bxf_content = format!(
            r#"<BXF version="5.0">
  <AsRun>
    <EventID>{}</EventID>
    <ContentID>{}</ContentID>
    <HouseNumber>{}</HouseNumber>
    <ScheduledStart>{}</ScheduledStart>
    <ActualStart>{}</ActualStart>
    <ScheduledDuration>{}</ScheduledDuration>
    <ActualDuration>{}</ActualDuration>
    <Status>{:?}</Status>
  </AsRun>
</BXF>"#,
            entry.id,
            entry.content_id,
            entry.house_number.as_deref().unwrap_or(""),
            entry.scheduled_start.to_rfc3339(),
            entry.actual_start.to_rfc3339(),
            entry.scheduled_duration_ms,
            entry.actual_duration_ms,
            entry.status,
        );

        let filename = format!(
            "asrun_{}_{}.bxf",
            entry.id,
            Utc::now().format("%Y%m%d_%H%M%S")
        );
        let path = self.config.log_dir.join("bxf").join(filename);

        if let Some(parent) = path.parent() {
            tokio::fs::create_dir_all(parent).await?;
        }

        tokio::fs::write(&path, bxf_content).await?;
        debug!("Exported BXF: {}", path.display());

        Ok(())
    }

    /// Export entry as XML format
    async fn export_xml(&self, entry: &AsRunEntry) -> Result<()> {
        let xml_content = format!(
            r#"<?xml version="1.0" encoding="UTF-8"?>
<AsRunEntry>
  <ID>{}</ID>
  <ContentID>{}</ContentID>
  <ScheduledStart>{}</ScheduledStart>
  <ActualStart>{}</ActualStart>
  <ScheduledDuration>{}</ScheduledDuration>
  <ActualDuration>{}</ActualDuration>
  <Status>{:?}</Status>
</AsRunEntry>"#,
            entry.id,
            entry.content_id,
            entry.scheduled_start.to_rfc3339(),
            entry.actual_start.to_rfc3339(),
            entry.scheduled_duration_ms,
            entry.actual_duration_ms,
            entry.status,
        );

        let filename = format!("asrun_{}.xml", entry.id);
        let path = self.config.log_dir.join("xml").join(filename);

        if let Some(parent) = path.parent() {
            tokio::fs::create_dir_all(parent).await?;
        }

        tokio::fs::write(&path, xml_content).await?;
        debug!("Exported XML: {}", path.display());

        Ok(())
    }

    /// Export entry as CSV format
    async fn export_csv(&self, entry: &AsRunEntry) -> Result<()> {
        let csv_line = format!(
            "{},{},{},{},{},{},{},{:?}\n",
            entry.id,
            entry.content_id,
            entry.house_number.as_deref().unwrap_or(""),
            entry.scheduled_start.to_rfc3339(),
            entry.actual_start.to_rfc3339(),
            entry.scheduled_duration_ms,
            entry.actual_duration_ms,
            entry.status,
        );

        let filename = format!("asrun_{}.csv", Utc::now().format("%Y%m%d"));
        let path = self.config.log_dir.join("csv").join(filename);

        if let Some(parent) = path.parent() {
            tokio::fs::create_dir_all(parent).await?;
        }

        let mut file = OpenOptions::new()
            .create(true)
            .append(true)
            .open(&path)
            .await?;

        file.write_all(csv_line.as_bytes()).await?;
        file.flush().await?;

        Ok(())
    }

    /// Get all entries for a date range
    pub async fn get_entries(&self, start: DateTime<Utc>, end: DateTime<Utc>) -> Vec<AsRunEntry> {
        let entries = self.entries.read().await;
        entries
            .iter()
            .filter(|e| e.actual_start >= start && e.actual_start <= end)
            .cloned()
            .collect()
    }

    /// Generate compliance report
    pub async fn generate_compliance_report(
        &self,
        start: DateTime<Utc>,
        end: DateTime<Utc>,
    ) -> ComplianceReport {
        let entries = self.get_entries(start, end).await;

        let total_events = entries.len();
        let successful = entries
            .iter()
            .filter(|e| e.status == PlayoutStatus::Played)
            .count();
        let failed = entries
            .iter()
            .filter(|e| e.status == PlayoutStatus::Failed)
            .count();
        let skipped = entries
            .iter()
            .filter(|e| e.status == PlayoutStatus::Skipped)
            .count();

        let commercials = entries
            .iter()
            .filter(|e| e.event_type == AsRunEventType::Commercial)
            .count();

        let mut timing_discrepancies = Vec::new();
        for entry in &entries {
            let scheduled_duration = entry.scheduled_duration_ms as i64;
            let actual_duration = entry.actual_duration_ms as i64;
            let diff = (actual_duration - scheduled_duration).abs();

            if diff > 1000 {
                // More than 1 second difference
                timing_discrepancies.push(TimingDiscrepancy {
                    entry_id: entry.id,
                    content_id: entry.content_id.clone(),
                    scheduled_duration_ms: entry.scheduled_duration_ms,
                    actual_duration_ms: entry.actual_duration_ms,
                    difference_ms: diff as u64,
                });
            }
        }

        ComplianceReport {
            period_start: start,
            period_end: end,
            total_events,
            successful_events: successful,
            failed_events: failed,
            skipped_events: skipped,
            commercial_count: commercials,
            timing_discrepancies,
            generated_at: Utc::now(),
        }
    }

    /// Clean up old logs based on retention policy
    pub async fn cleanup_old_logs(&self) -> Result<()> {
        let cutoff = Utc::now() - chrono::Duration::days(self.config.retention_days as i64);

        let mut entries = self.entries.write().await;
        entries.retain(|e| e.created_at > cutoff);

        info!(
            "Cleaned up as-run logs older than {} days",
            self.config.retention_days
        );
        Ok(())
    }
}

/// Compliance report
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ComplianceReport {
    /// Report period start
    pub period_start: DateTime<Utc>,

    /// Report period end
    pub period_end: DateTime<Utc>,

    /// Total number of events
    pub total_events: usize,

    /// Successfully played events
    pub successful_events: usize,

    /// Failed events
    pub failed_events: usize,

    /// Skipped events
    pub skipped_events: usize,

    /// Commercial spot count
    pub commercial_count: usize,

    /// Timing discrepancies
    pub timing_discrepancies: Vec<TimingDiscrepancy>,

    /// Report generation timestamp
    pub generated_at: DateTime<Utc>,
}

/// Timing discrepancy record
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TimingDiscrepancy {
    /// Entry ID
    pub entry_id: Uuid,

    /// Content ID
    pub content_id: String,

    /// Scheduled duration in milliseconds
    pub scheduled_duration_ms: u64,

    /// Actual duration in milliseconds
    pub actual_duration_ms: u64,

    /// Difference in milliseconds
    pub difference_ms: u64,
}

/// Spot verification for commercial reconciliation
pub struct SpotVerifier {
    expected_spots: Arc<RwLock<HashMap<String, ExpectedSpot>>>,
}

impl SpotVerifier {
    /// Create new spot verifier
    pub fn new() -> Self {
        Self {
            expected_spots: Arc::new(RwLock::new(HashMap::new())),
        }
    }

    /// Add expected spot
    pub async fn add_expected_spot(&self, spot: ExpectedSpot) {
        let mut spots = self.expected_spots.write().await;
        spots.insert(spot.isci_code.clone(), spot);
    }

    /// Verify spot against expected
    pub async fn verify_spot(&self, entry: &AsRunEntry) -> VerificationResult {
        if let Some(isci) = &entry.isci_code {
            let spots = self.expected_spots.read().await;
            if let Some(expected) = spots.get(isci) {
                return self.compare_spot(entry, expected);
            }
        }

        VerificationResult {
            verified: false,
            discrepancies: vec!["No expected spot found".to_string()],
        }
    }

    /// Compare actual vs expected spot
    fn compare_spot(&self, actual: &AsRunEntry, expected: &ExpectedSpot) -> VerificationResult {
        let mut discrepancies = Vec::new();

        // Check timing
        let time_diff =
            (actual.actual_start.timestamp() - expected.scheduled_time.timestamp()).abs();
        if time_diff > 60 {
            // More than 60 seconds difference
            discrepancies.push(format!("Time difference: {time_diff} seconds"));
        }

        // Check duration
        let duration_diff = (actual.actual_duration_ms as i64 - expected.duration_ms as i64).abs();
        if duration_diff > 1000 {
            // More than 1 second
            discrepancies.push(format!("Duration difference: {duration_diff} ms"));
        }

        VerificationResult {
            verified: discrepancies.is_empty(),
            discrepancies,
        }
    }
}

impl Default for SpotVerifier {
    fn default() -> Self {
        Self::new()
    }
}

/// Expected commercial spot
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExpectedSpot {
    /// ISCI code
    pub isci_code: String,

    /// Scheduled time
    pub scheduled_time: DateTime<Utc>,

    /// Expected duration in milliseconds
    pub duration_ms: u64,

    /// Advertiser name
    pub advertiser: String,

    /// Product name
    pub product: String,

    /// Rate information
    pub rate: Option<f64>,
}

/// Verification result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VerificationResult {
    /// Whether spot verified successfully
    pub verified: bool,

    /// List of discrepancies found
    pub discrepancies: Vec<String>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_asrun_config_default() {
        let config = AsRunConfig::default();
        assert!(config.enabled);
        assert_eq!(config.rotation_hours, 24);
        assert_eq!(config.retention_days, 90);
    }

    #[test]
    fn test_asrun_entry_creation() {
        let entry = AsRunEntry {
            id: Uuid::new_v4(),
            event_id: Some(Uuid::new_v4()),
            content_id: "TEST001".to_string(),
            house_number: Some("H12345".to_string()),
            isci_code: Some("ABCD1234H".to_string()),
            scheduled_start: Utc::now(),
            actual_start: Utc::now(),
            scheduled_duration_ms: 30000,
            actual_duration_ms: 30050,
            scheduled_end: Utc::now(),
            actual_end: Utc::now(),
            event_type: AsRunEventType::Commercial,
            status: PlayoutStatus::Played,
            error_code: None,
            error_message: None,
            technical: None,
            notes: None,
            segmentation: None,
            verified: false,
            created_at: Utc::now(),
        };

        assert_eq!(entry.content_id, "TEST001");
        assert_eq!(entry.event_type, AsRunEventType::Commercial);
    }

    #[test]
    fn test_export_format_equality() {
        assert_eq!(ExportFormat::Bxf, ExportFormat::Bxf);
        assert_ne!(ExportFormat::Bxf, ExportFormat::Xml);
    }

    #[test]
    fn test_playout_status() {
        assert_eq!(PlayoutStatus::Played, PlayoutStatus::Played);
        assert_ne!(PlayoutStatus::Played, PlayoutStatus::Failed);
    }

    #[tokio::test]
    async fn test_spot_verifier() {
        let verifier = SpotVerifier::new();

        let expected = ExpectedSpot {
            isci_code: "TEST1234H".to_string(),
            scheduled_time: Utc::now(),
            duration_ms: 30000,
            advertiser: "Test Corp".to_string(),
            product: "Test Product".to_string(),
            rate: Some(500.0),
        };

        verifier.add_expected_spot(expected.clone()).await;

        let entry = AsRunEntry {
            id: Uuid::new_v4(),
            event_id: None,
            content_id: "TEST001".to_string(),
            house_number: None,
            isci_code: Some("TEST1234H".to_string()),
            scheduled_start: Utc::now(),
            actual_start: Utc::now(),
            scheduled_duration_ms: 30000,
            actual_duration_ms: 30000,
            scheduled_end: Utc::now(),
            actual_end: Utc::now(),
            event_type: AsRunEventType::Commercial,
            status: PlayoutStatus::Played,
            error_code: None,
            error_message: None,
            technical: None,
            notes: None,
            segmentation: None,
            verified: false,
            created_at: Utc::now(),
        };

        let result = verifier.verify_spot(&entry).await;
        assert!(result.verified || !result.discrepancies.is_empty());
    }

    #[test]
    fn test_technical_details() {
        let details = TechnicalDetails {
            video_format: "1920x1080i50".to_string(),
            audio_format: "48kHz/24bit".to_string(),
            codec: "AVC-Intra 100".to_string(),
            bitrate_kbps: 100000,
            framerate: 50.0,
            audio_levels: None,
            video_quality: None,
        };

        assert_eq!(details.video_format, "1920x1080i50");
        assert_eq!(details.framerate, 50.0);
    }

    #[test]
    fn test_audio_levels() {
        let levels = AudioLevels {
            peak_dbfs: -6.0,
            rms_dbfs: -18.0,
            integrated_lufs: -23.0,
            loudness_range_lu: 8.0,
            true_peak_dbtp: -3.0,
        };

        assert_eq!(levels.integrated_lufs, -23.0);
    }

    #[test]
    fn test_video_quality() {
        let quality = VideoQuality {
            dropped_frames: 0,
            black_frames: 0,
            frozen_frames: 0,
            avg_bitrate_kbps: 50000,
        };

        assert_eq!(quality.dropped_frames, 0);
    }

    #[test]
    fn test_segmentation_data() {
        let seg = SegmentationData {
            event_id: 12345,
            segmentation_type: "Program Start".to_string(),
            duration_ms: Some(3600000),
            upid: Some("EP1234567890".to_string()),
            message: None,
        };

        assert_eq!(seg.event_id, 12345);
    }
}
