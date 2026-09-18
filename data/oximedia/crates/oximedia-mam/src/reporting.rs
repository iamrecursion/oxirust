//! Scheduled report generation for storage usage and asset statistics.
//!
//! This module provides:
//! - Report definitions with configurable parameters and schedules
//! - Multiple built-in report types (storage, assets, ingest, workflow, audit)
//! - Schedule evaluation (cron-like: hourly, daily, weekly, monthly)
//! - Report generation that operates on a pluggable `ReportDataSource`
//! - Output in JSON, CSV, and plain-text formats
//! - A `ReportScheduler` that tracks which reports are due

use chrono::{DateTime, Datelike, Timelike, Utc};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use uuid::Uuid;

// ---------------------------------------------------------------------------
// Report type
// ---------------------------------------------------------------------------

/// Category of report.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ReportType {
    /// Summary of storage consumption by tier, project, and owner.
    StorageUsage,
    /// Asset inventory: count, types, status distribution, growth rate.
    AssetInventory,
    /// Ingest pipeline statistics: throughput, failure rate, average duration.
    IngestStatistics,
    /// Workflow performance: approval times, SLAs, bottlenecks.
    WorkflowPerformance,
    /// Audit summary: event counts, failure rates, top actors.
    AuditSummary,
    /// Top-N most accessed assets and users.
    UsageTop,
    /// Duplicate asset report.
    DuplicateAssets,
    /// Custom report driven by user-defined query parameters.
    Custom(String),
}

impl ReportType {
    /// Human-readable display name.
    #[must_use]
    pub fn display_name(&self) -> &str {
        match self {
            Self::StorageUsage => "Storage Usage",
            Self::AssetInventory => "Asset Inventory",
            Self::IngestStatistics => "Ingest Statistics",
            Self::WorkflowPerformance => "Workflow Performance",
            Self::AuditSummary => "Audit Summary",
            Self::UsageTop => "Usage Top-N",
            Self::DuplicateAssets => "Duplicate Assets",
            Self::Custom(s) => s.as_str(),
        }
    }
}

// ---------------------------------------------------------------------------
// Schedule
// ---------------------------------------------------------------------------

/// Recurrence schedule for a report.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum ReportSchedule {
    /// Run once (ad-hoc).
    Once,
    /// Every N hours.
    Hourly { every_n: u32 },
    /// Every day at a specific UTC hour (0-23).
    Daily { hour: u8 },
    /// Every week on a specific ISO weekday (1=Mon … 7=Sun) at a UTC hour.
    Weekly { weekday: u8, hour: u8 },
    /// Every month on a specific day-of-month (1-28) at a UTC hour.
    Monthly { day: u8, hour: u8 },
}

impl ReportSchedule {
    /// Determine whether the report is due at the given instant.
    ///
    /// `last_run` is `None` if the report has never run before.
    #[must_use]
    pub fn is_due(&self, now: DateTime<Utc>, last_run: Option<DateTime<Utc>>) -> bool {
        match self {
            Self::Once => last_run.is_none(),
            Self::Hourly { every_n } => last_run.map_or(true, |lr| {
                let elapsed_hours = now.signed_duration_since(lr).num_hours();
                elapsed_hours >= i64::from(*every_n)
            }),
            Self::Daily { hour } => last_run.map_or(true, |lr| {
                (now.hour() as u8) >= *hour
                    && (now.date_naive() > lr.date_naive()
                        || (now.date_naive() == lr.date_naive() && (lr.hour() as u8) < *hour))
            }),
            Self::Weekly { weekday, hour } => last_run.map_or(true, |lr| {
                let elapsed_days = now.signed_duration_since(lr).num_days();
                elapsed_days >= 7
                    && now.weekday().number_from_monday() as u8 == *weekday
                    && now.hour() as u8 >= *hour
            }),
            Self::Monthly { day, hour } => last_run.map_or(true, |lr| {
                let elapsed_days = now.signed_duration_since(lr).num_days();
                elapsed_days >= 28 && now.day() as u8 >= *day && now.hour() as u8 >= *hour
            }),
        }
    }
}

// ---------------------------------------------------------------------------
// Report definition
// ---------------------------------------------------------------------------

/// Output format for generated reports.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum OutputFormat {
    Json,
    Csv,
    PlainText,
}

impl OutputFormat {
    /// MIME type for this format.
    #[must_use]
    pub const fn mime_type(&self) -> &'static str {
        match self {
            Self::Json => "application/json",
            Self::Csv => "text/csv",
            Self::PlainText => "text/plain",
        }
    }
}

/// A report definition that can be scheduled.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReportDef {
    pub id: Uuid,
    /// Human-readable name.
    pub name: String,
    /// Description of what this report shows.
    pub description: String,
    /// Type of report to generate.
    pub report_type: ReportType,
    /// Recurrence schedule.
    pub schedule: ReportSchedule,
    /// Output format.
    pub format: OutputFormat,
    /// Extra parameters passed to the generator.
    pub params: HashMap<String, serde_json::Value>,
    /// Email addresses to deliver the report to.
    pub recipients: Vec<String>,
    /// When this definition was created.
    pub created_at: DateTime<Utc>,
    /// Whether the report is currently enabled.
    pub enabled: bool,
    /// When the report was last generated.
    pub last_run_at: Option<DateTime<Utc>>,
}

impl ReportDef {
    /// Create a new enabled report definition.
    #[must_use]
    pub fn new(
        name: impl Into<String>,
        report_type: ReportType,
        schedule: ReportSchedule,
        format: OutputFormat,
    ) -> Self {
        Self {
            id: Uuid::new_v4(),
            name: name.into(),
            description: String::new(),
            report_type,
            schedule,
            format,
            params: HashMap::new(),
            recipients: Vec::new(),
            created_at: Utc::now(),
            enabled: true,
            last_run_at: None,
        }
    }

    /// Builder: set description.
    #[must_use]
    pub fn with_description(mut self, desc: impl Into<String>) -> Self {
        self.description = desc.into();
        self
    }

    /// Builder: add a parameter.
    #[must_use]
    pub fn with_param(mut self, key: impl Into<String>, value: serde_json::Value) -> Self {
        self.params.insert(key.into(), value);
        self
    }

    /// Builder: add a recipient.
    #[must_use]
    pub fn to(mut self, email: impl Into<String>) -> Self {
        self.recipients.push(email.into());
        self
    }

    /// Whether this report is due to run given the current time.
    #[must_use]
    pub fn is_due(&self, now: DateTime<Utc>) -> bool {
        self.enabled && self.schedule.is_due(now, self.last_run_at)
    }
}

// ---------------------------------------------------------------------------
// Report result
// ---------------------------------------------------------------------------

/// The result of a single report run.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReportResult {
    pub id: Uuid,
    /// Id of the definition that produced this result.
    pub definition_id: Uuid,
    /// Name of the report.
    pub name: String,
    /// Report type.
    pub report_type: ReportType,
    /// Generated content.
    pub content: String,
    /// Output format.
    pub format: OutputFormat,
    /// When the report was generated.
    pub generated_at: DateTime<Utc>,
    /// Time taken to generate in milliseconds.
    pub generation_ms: u64,
    /// Whether generation succeeded.
    pub success: bool,
    /// Error message if generation failed.
    pub error: Option<String>,
}

// ---------------------------------------------------------------------------
// Data source
// ---------------------------------------------------------------------------

/// Snapshot of data used to populate a report.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ReportDataSnapshot {
    /// Total number of assets.
    pub total_assets: u64,
    /// Assets by status (status → count).
    pub assets_by_status: HashMap<String, u64>,
    /// Assets by MIME type family (e.g. "video", "audio", "image").
    pub assets_by_type: HashMap<String, u64>,
    /// Total storage used in bytes.
    pub total_storage_bytes: u64,
    /// Storage by tier (tier label → bytes).
    pub storage_by_tier: HashMap<String, u64>,
    /// Storage by project (project name → bytes).
    pub storage_by_project: HashMap<String, u64>,
    /// Number of ingest operations in the period.
    pub ingest_count: u64,
    /// Number of failed ingests.
    pub ingest_failures: u64,
    /// Average ingest duration in seconds.
    pub avg_ingest_duration_secs: f64,
    /// Number of workflow executions.
    pub workflow_count: u64,
    /// Average workflow approval time in hours.
    pub avg_workflow_approval_hours: f64,
    /// Audit events in the period.
    pub audit_event_count: u64,
    /// Audit failures.
    pub audit_failure_count: u64,
    /// Top-N assets: (asset name, access count).
    pub top_assets: Vec<(String, u64)>,
    /// Top-N users: (username, action count).
    pub top_users: Vec<(String, u64)>,
    /// Number of duplicate asset groups detected.
    pub duplicate_groups: u64,
    /// Custom key-value pairs for the Custom report type.
    pub custom_data: HashMap<String, serde_json::Value>,
    /// Period start.
    pub period_start: Option<DateTime<Utc>>,
    /// Period end.
    pub period_end: Option<DateTime<Utc>>,
}

// ---------------------------------------------------------------------------
// Report generator
// ---------------------------------------------------------------------------

/// Generates reports from a data snapshot.
pub struct ReportGenerator;

impl ReportGenerator {
    /// Generate a report result from a definition and data snapshot.
    ///
    /// This method is intentionally synchronous and free of I/O — data
    /// collection is the caller's responsibility.
    #[must_use]
    pub fn generate(def: &ReportDef, data: &ReportDataSnapshot) -> ReportResult {
        let start = std::time::Instant::now();
        let content = match def.format {
            OutputFormat::Json => Self::render_json(def, data),
            OutputFormat::Csv => Self::render_csv(def, data),
            OutputFormat::PlainText => Self::render_plain(def, data),
        };
        let generation_ms = start.elapsed().as_millis() as u64;

        ReportResult {
            id: Uuid::new_v4(),
            definition_id: def.id,
            name: def.name.clone(),
            report_type: def.report_type.clone(),
            content,
            format: def.format,
            generated_at: Utc::now(),
            generation_ms,
            success: true,
            error: None,
        }
    }

    fn render_json(def: &ReportDef, data: &ReportDataSnapshot) -> String {
        let payload = Self::build_payload(def, data);
        serde_json::to_string_pretty(&payload).unwrap_or_else(|e| format!("{{\"error\": \"{e}\"}}"))
    }

    fn render_csv(_def: &ReportDef, data: &ReportDataSnapshot) -> String {
        let mut rows: Vec<String> = vec!["metric,value".to_string()];
        rows.push(format!("total_assets,{}", data.total_assets));
        rows.push(format!("total_storage_bytes,{}", data.total_storage_bytes));
        rows.push(format!("ingest_count,{}", data.ingest_count));
        rows.push(format!("ingest_failures,{}", data.ingest_failures));
        rows.push(format!("workflow_count,{}", data.workflow_count));
        rows.push(format!("audit_event_count,{}", data.audit_event_count));
        rows.push(format!("audit_failure_count,{}", data.audit_failure_count));
        rows.push(format!("duplicate_groups,{}", data.duplicate_groups));

        for (status, count) in &data.assets_by_status {
            rows.push(format!("assets.status.{status},{count}"));
        }
        for (tier, bytes) in &data.storage_by_tier {
            rows.push(format!("storage.tier.{tier},{bytes}"));
        }
        rows.join("\n")
    }

    fn render_plain(def: &ReportDef, data: &ReportDataSnapshot) -> String {
        let mut lines: Vec<String> = vec![
            format!("=== {} ===", def.name),
            format!("Type: {}", def.report_type.display_name()),
            String::new(),
            format!("Total Assets:       {}", data.total_assets),
            format!("Total Storage:      {} bytes", data.total_storage_bytes),
            format!("Ingest Count:       {}", data.ingest_count),
            format!("Ingest Failures:    {}", data.ingest_failures),
            format!("Workflow Count:     {}", data.workflow_count),
            format!(
                "Avg Approval Time:  {:.1}h",
                data.avg_workflow_approval_hours
            ),
            format!("Audit Events:       {}", data.audit_event_count),
            format!("Duplicate Groups:   {}", data.duplicate_groups),
        ];

        if !data.top_assets.is_empty() {
            lines.push(String::new());
            lines.push("Top Assets:".to_string());
            for (name, count) in &data.top_assets {
                lines.push(format!("  {name}: {count} accesses"));
            }
        }

        if !data.top_users.is_empty() {
            lines.push(String::new());
            lines.push("Top Users:".to_string());
            for (user, count) in &data.top_users {
                lines.push(format!("  {user}: {count} actions"));
            }
        }

        lines.join("\n")
    }

    fn build_payload(def: &ReportDef, data: &ReportDataSnapshot) -> serde_json::Value {
        serde_json::json!({
            "report": def.name,
            "type": def.report_type.display_name(),
            "generated_at": Utc::now().to_rfc3339(),
            "period": {
                "start": data.period_start.map(|t| t.to_rfc3339()),
                "end": data.period_end.map(|t| t.to_rfc3339()),
            },
            "summary": {
                "total_assets": data.total_assets,
                "total_storage_bytes": data.total_storage_bytes,
                "ingest_count": data.ingest_count,
                "ingest_failures": data.ingest_failures,
                "ingest_failure_rate": if data.ingest_count > 0 {
                    data.ingest_failures as f64 / data.ingest_count as f64
                } else { 0.0 },
                "workflow_count": data.workflow_count,
                "avg_workflow_approval_hours": data.avg_workflow_approval_hours,
                "audit_event_count": data.audit_event_count,
                "audit_failure_count": data.audit_failure_count,
                "duplicate_groups": data.duplicate_groups,
            },
            "assets_by_status": data.assets_by_status,
            "assets_by_type": data.assets_by_type,
            "storage_by_tier": data.storage_by_tier,
            "storage_by_project": data.storage_by_project,
            "top_assets": data.top_assets.iter().map(|(n, c)| {
                serde_json::json!({"name": n, "access_count": c})
            }).collect::<Vec<_>>(),
            "top_users": data.top_users.iter().map(|(u, c)| {
                serde_json::json!({"user": u, "action_count": c})
            }).collect::<Vec<_>>(),
            "custom": data.custom_data,
        })
    }
}

// ---------------------------------------------------------------------------
// Scheduler
// ---------------------------------------------------------------------------

/// Manages a collection of report definitions and tracks which are due.
#[derive(Debug)]
pub struct ReportScheduler {
    definitions: HashMap<Uuid, ReportDef>,
    results: Vec<ReportResult>,
}

impl ReportScheduler {
    /// Create an empty scheduler.
    #[must_use]
    pub fn new() -> Self {
        Self {
            definitions: HashMap::new(),
            results: Vec::new(),
        }
    }

    /// Register a report definition.
    pub fn register(&mut self, def: ReportDef) {
        self.definitions.insert(def.id, def);
    }

    /// Remove a definition by id.
    pub fn deregister(&mut self, id: Uuid) {
        self.definitions.remove(&id);
    }

    /// Get all definitions that are due at `now`.
    #[must_use]
    pub fn due_reports(&self, now: DateTime<Utc>) -> Vec<&ReportDef> {
        self.definitions
            .values()
            .filter(|d| d.is_due(now))
            .collect()
    }

    /// Generate all due reports, record results, and update `last_run_at`.
    pub fn run_due(&mut self, now: DateTime<Utc>, data: &ReportDataSnapshot) -> Vec<&ReportResult> {
        let due_ids: Vec<Uuid> = self
            .definitions
            .values()
            .filter(|d| d.is_due(now))
            .map(|d| d.id)
            .collect();

        let mut new_result_indices = Vec::new();
        for id in due_ids {
            if let Some(def) = self.definitions.get_mut(&id) {
                let result = ReportGenerator::generate(def, data);
                def.last_run_at = Some(now);
                new_result_indices.push(self.results.len());
                self.results.push(result);
            }
        }

        new_result_indices
            .into_iter()
            .filter_map(|i| self.results.get(i))
            .collect()
    }

    /// Get a definition by id.
    #[must_use]
    pub fn get_def(&self, id: Uuid) -> Option<&ReportDef> {
        self.definitions.get(&id)
    }

    /// All stored results.
    #[must_use]
    pub fn results(&self) -> &[ReportResult] {
        &self.results
    }

    /// Number of registered definitions.
    #[must_use]
    pub fn definition_count(&self) -> usize {
        self.definitions.len()
    }
}

impl Default for ReportScheduler {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_data() -> ReportDataSnapshot {
        let mut data = ReportDataSnapshot {
            total_assets: 1_500,
            total_storage_bytes: 2_000_000_000,
            ingest_count: 200,
            ingest_failures: 10,
            avg_ingest_duration_secs: 45.0,
            workflow_count: 80,
            avg_workflow_approval_hours: 3.5,
            audit_event_count: 5_000,
            audit_failure_count: 42,
            duplicate_groups: 7,
            ..Default::default()
        };
        data.assets_by_status.insert("active".to_string(), 1_200);
        data.assets_by_status.insert("archived".to_string(), 300);
        data.storage_by_tier
            .insert("hot".to_string(), 1_500_000_000);
        data.storage_by_tier.insert("cold".to_string(), 500_000_000);
        data.top_assets.push(("promo_2024.mp4".to_string(), 312));
        data.top_users.push(("alice".to_string(), 540));
        data
    }

    // --- ReportType ---

    #[test]
    fn test_report_type_display_names() {
        assert_eq!(ReportType::StorageUsage.display_name(), "Storage Usage");
        assert_eq!(ReportType::AssetInventory.display_name(), "Asset Inventory");
        assert_eq!(
            ReportType::Custom("MyReport".to_string()).display_name(),
            "MyReport"
        );
    }

    // --- OutputFormat ---

    #[test]
    fn test_output_format_mime_types() {
        assert_eq!(OutputFormat::Json.mime_type(), "application/json");
        assert_eq!(OutputFormat::Csv.mime_type(), "text/csv");
        assert_eq!(OutputFormat::PlainText.mime_type(), "text/plain");
    }

    // --- ReportSchedule ---

    #[test]
    fn test_schedule_once_no_last_run() {
        let now = Utc::now();
        assert!(ReportSchedule::Once.is_due(now, None));
    }

    #[test]
    fn test_schedule_once_already_run() {
        let now = Utc::now();
        assert!(!ReportSchedule::Once.is_due(now, Some(now)));
    }

    #[test]
    fn test_schedule_hourly_not_yet_due() {
        let now = Utc::now();
        let last = now - chrono::Duration::minutes(30);
        assert!(!ReportSchedule::Hourly { every_n: 1 }.is_due(now, Some(last)));
    }

    #[test]
    fn test_schedule_hourly_due() {
        let now = Utc::now();
        let last = now - chrono::Duration::hours(2);
        assert!(ReportSchedule::Hourly { every_n: 1 }.is_due(now, Some(last)));
    }

    // --- ReportDef ---

    #[test]
    fn test_report_def_builder() {
        let def = ReportDef::new(
            "Weekly Storage",
            ReportType::StorageUsage,
            ReportSchedule::Once,
            OutputFormat::Json,
        )
        .with_description("Weekly storage overview")
        .with_param("include_archived", serde_json::json!(true))
        .to("admin@example.com");

        assert!(def.enabled);
        assert_eq!(def.recipients.len(), 1);
        assert!(def.params.contains_key("include_archived"));
    }

    #[test]
    fn test_report_def_is_due_once() {
        let def = ReportDef::new(
            "Ad-hoc",
            ReportType::StorageUsage,
            ReportSchedule::Once,
            OutputFormat::Json,
        );
        assert!(def.is_due(Utc::now()));
    }

    // --- ReportGenerator JSON ---

    #[test]
    fn test_generate_json() {
        let def = ReportDef::new(
            "Test",
            ReportType::StorageUsage,
            ReportSchedule::Once,
            OutputFormat::Json,
        );
        let data = sample_data();
        let result = ReportGenerator::generate(&def, &data);
        assert!(result.success);
        let v: serde_json::Value = serde_json::from_str(&result.content).expect("valid JSON");
        assert_eq!(v["summary"]["total_assets"], 1_500);
        assert!(
            v["summary"]["ingest_failure_rate"]
                .as_f64()
                .expect("should be f64")
                > 0.0
        );
    }

    #[test]
    fn test_generate_csv() {
        let def = ReportDef::new(
            "CSV",
            ReportType::AssetInventory,
            ReportSchedule::Once,
            OutputFormat::Csv,
        );
        let data = sample_data();
        let result = ReportGenerator::generate(&def, &data);
        assert!(result.success);
        assert!(result.content.starts_with("metric,value"));
        assert!(result.content.contains("total_assets,1500"));
        assert!(result.content.contains("assets.status.active,1200"));
    }

    #[test]
    fn test_generate_plain_text() {
        let def = ReportDef::new(
            "Plain Report",
            ReportType::AuditSummary,
            ReportSchedule::Once,
            OutputFormat::PlainText,
        );
        let data = sample_data();
        let result = ReportGenerator::generate(&def, &data);
        assert!(result.success);
        assert!(result.content.contains("Plain Report"));
        assert!(result.content.contains("Total Assets:"));
        assert!(result.content.contains("Top Assets:"));
        assert!(result.content.contains("Top Users:"));
    }

    #[test]
    fn test_report_result_format() {
        let def = ReportDef::new(
            "R",
            ReportType::StorageUsage,
            ReportSchedule::Once,
            OutputFormat::Json,
        );
        let result = ReportGenerator::generate(&def, &sample_data());
        assert_eq!(result.format, OutputFormat::Json);
        assert_eq!(result.definition_id, def.id);
    }

    // --- ReportScheduler ---

    #[test]
    fn test_scheduler_register_and_due() {
        let mut sched = ReportScheduler::new();
        let def = ReportDef::new(
            "R",
            ReportType::StorageUsage,
            ReportSchedule::Once,
            OutputFormat::Json,
        );
        sched.register(def);
        assert_eq!(sched.definition_count(), 1);
        let due = sched.due_reports(Utc::now());
        assert_eq!(due.len(), 1);
    }

    #[test]
    fn test_scheduler_run_due_marks_last_run() {
        let mut sched = ReportScheduler::new();
        let def = ReportDef::new(
            "R",
            ReportType::StorageUsage,
            ReportSchedule::Once,
            OutputFormat::Json,
        );
        let id = def.id;
        sched.register(def);

        let now = Utc::now();
        let results = sched.run_due(now, &sample_data());
        assert_eq!(results.len(), 1);
        assert!(sched
            .get_def(id)
            .expect("def should exist")
            .last_run_at
            .is_some());
    }

    #[test]
    fn test_scheduler_once_not_due_after_run() {
        let mut sched = ReportScheduler::new();
        let def = ReportDef::new(
            "R",
            ReportType::StorageUsage,
            ReportSchedule::Once,
            OutputFormat::Json,
        );
        sched.register(def);
        let now = Utc::now();
        sched.run_due(now, &sample_data());

        // Should no longer be due
        let due = sched.due_reports(now);
        assert!(due.is_empty());
    }

    #[test]
    fn test_scheduler_deregister() {
        let mut sched = ReportScheduler::new();
        let def = ReportDef::new(
            "R",
            ReportType::StorageUsage,
            ReportSchedule::Once,
            OutputFormat::Json,
        );
        let id = def.id;
        sched.register(def);
        sched.deregister(id);
        assert_eq!(sched.definition_count(), 0);
    }

    #[test]
    fn test_scheduler_results_accumulate() {
        let mut sched = ReportScheduler::new();
        for i in 0..3 {
            let def = ReportDef::new(
                format!("R{i}"),
                ReportType::StorageUsage,
                ReportSchedule::Once,
                OutputFormat::Json,
            );
            sched.register(def);
        }
        sched.run_due(Utc::now(), &sample_data());
        assert_eq!(sched.results().len(), 3);
    }

    #[test]
    fn test_ingest_failure_rate_zero_when_no_ingests() {
        let def = ReportDef::new(
            "R",
            ReportType::IngestStatistics,
            ReportSchedule::Once,
            OutputFormat::Json,
        );
        let data = ReportDataSnapshot::default();
        let result = ReportGenerator::generate(&def, &data);
        let v: serde_json::Value = serde_json::from_str(&result.content).expect("valid JSON");
        assert_eq!(v["summary"]["ingest_failure_rate"], 0.0);
    }
}
