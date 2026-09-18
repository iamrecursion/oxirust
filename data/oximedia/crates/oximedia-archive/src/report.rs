//! Verification reporting and analytics
//!
//! This module provides:
//! - Verification reports (HTML, JSON, CSV)
//! - Integrity metrics
//! - Historical tracking
//! - Alert generation
//! - OAIS compliance reports

use crate::{ArchiveError, ArchiveResult};
use chrono::{DateTime, Utc};
use oxisql_core::Connection;
use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};
use tokio::fs;
use tracing::info;

/// Report format
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ReportFormat {
    Html,
    Json,
    Csv,
    Text,
}

/// Generate verification report
pub async fn generate_report(
    pool: &oxisql_sqlite_compat::SqliteConnection,
    format: ReportFormat,
    output_path: &Path,
) -> ArchiveResult<()> {
    info!(
        "Generating {} report to {}",
        format_name(format),
        output_path.display()
    );

    let report = collect_report_data(pool).await?;

    let content = match format {
        ReportFormat::Html => generate_html_report(&report)?,
        ReportFormat::Json => generate_json_report(&report)?,
        ReportFormat::Csv => generate_csv_report(&report)?,
        ReportFormat::Text => generate_text_report(&report)?,
    };

    fs::write(output_path, content).await?;

    info!("Report generated successfully");
    Ok(())
}

/// Format name
fn format_name(format: ReportFormat) -> &'static str {
    match format {
        ReportFormat::Html => "HTML",
        ReportFormat::Json => "JSON",
        ReportFormat::Csv => "CSV",
        ReportFormat::Text => "Text",
    }
}

/// Comprehensive verification report
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VerificationReport {
    pub generated_at: DateTime<Utc>,
    pub summary: ReportSummary,
    pub checksum_stats: ChecksumStatistics,
    pub fixity_stats: FixityStatistics,
    pub quarantine_stats: QuarantineStatistics,
    pub recent_events: Vec<RecentEvent>,
    pub alerts: Vec<Alert>,
    pub integrity_metrics: IntegrityMetrics,
    pub file_details: Vec<FileDetail>,
}

/// Report summary
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReportSummary {
    pub total_files: usize,
    pub total_size: u64,
    pub verified_files: usize,
    pub failed_files: usize,
    pub quarantined_files: usize,
    pub last_check_date: Option<DateTime<Utc>>,
}

/// Checksum statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChecksumStatistics {
    pub total_checksums: usize,
    pub blake3_count: usize,
    pub md5_count: usize,
    pub sha256_count: usize,
    pub crc32_count: usize,
    pub oldest_checksum: Option<DateTime<Utc>>,
    pub newest_checksum: Option<DateTime<Utc>>,
}

/// Fixity statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FixityStatistics {
    pub total_checks: usize,
    pub passed_checks: usize,
    pub failed_checks: usize,
    pub success_rate: f64,
    pub average_days_between_checks: f64,
    pub files_needing_check: usize,
}

/// Quarantine statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QuarantineStatistics {
    pub total_quarantined: usize,
    pub active_quarantined: usize,
    pub restored: usize,
    pub auto_quarantined: usize,
    pub manual_quarantined: usize,
}

/// Recent event
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RecentEvent {
    pub event_type: String,
    pub event_date: DateTime<Utc>,
    pub file_path: String,
    pub outcome: String,
    pub details: Option<String>,
}

/// Alert
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Alert {
    pub alert_type: AlertType,
    pub severity: AlertSeverity,
    pub message: String,
    pub file_path: Option<String>,
    pub created_at: DateTime<Utc>,
}

/// Alert type
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum AlertType {
    FixityCheckFailed,
    FileCorrupted,
    ChecksumMismatch,
    FileQuarantined,
    BackupNeeded,
    StorageIssue,
}

/// Alert severity
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum AlertSeverity {
    Info,
    Warning,
    Error,
    Critical,
}

/// Integrity metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IntegrityMetrics {
    pub overall_health: f64,
    pub checksum_coverage: f64,
    pub fixity_compliance: f64,
    pub quarantine_rate: f64,
    pub data_at_risk: u64,
}

/// File detail
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FileDetail {
    pub file_path: String,
    pub file_size: u64,
    pub last_verified: Option<DateTime<Utc>>,
    pub verification_count: u32,
    pub failure_count: u32,
    pub status: FileStatus,
}

/// File status
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum FileStatus {
    Healthy,
    NeedsVerification,
    Failed,
    Quarantined,
}

/// Collect report data from database
async fn collect_report_data(
    pool: &oxisql_sqlite_compat::SqliteConnection,
) -> ArchiveResult<VerificationReport> {
    let summary = collect_summary(pool).await?;
    let checksum_stats = collect_checksum_stats(pool).await?;
    let fixity_stats = collect_fixity_stats(pool).await?;
    let quarantine_stats = collect_quarantine_stats(pool).await?;
    let recent_events = collect_recent_events(pool).await?;
    let alerts = generate_alerts(pool).await?;
    let integrity_metrics = calculate_integrity_metrics(&summary, &fixity_stats, &quarantine_stats);
    let file_details = collect_file_details(pool).await?;

    Ok(VerificationReport {
        generated_at: Utc::now(),
        summary,
        checksum_stats,
        fixity_stats,
        quarantine_stats,
        recent_events,
        alerts,
        integrity_metrics,
        file_details,
    })
}

/// Extract an aggregate `i64` column, treating SQL `NULL` (e.g. `SUM(...)` over
/// zero matching rows) as `0` rather than an error.
fn agg_i64(row: &oxisql_core::Row, col: &str) -> ArchiveResult<i64> {
    Ok(row.try_get::<Option<i64>>(col)?.unwrap_or(0))
}

/// Collect summary statistics
async fn collect_summary(
    pool: &oxisql_sqlite_compat::SqliteConnection,
) -> ArchiveResult<ReportSummary> {
    // NOTE: `last_check_date` is computed via a NULL-filtering subquery rather
    // than a bare `MAX(last_verified_at)` in the outer aggregate. The
    // Pure-Rust `oxisqlite` engine (`oxisql-sqlite-compat` 0.3.x) panics
    // (`unreachable!()` in `vdbe/execute/aggregate.rs`) when the *first* row
    // an aggregate steps over has a `NULL` value in a `MAX`/`MIN` target
    // column — which `last_verified_at` frequently does for freshly-created
    // checksum records. Restricting the subquery to `WHERE last_verified_at
    // IS NOT NULL` means the engine never observes a `NULL` while stepping,
    // sidestepping the crash while preserving standard SQL MAX-ignores-NULL
    // semantics. `created_at` (used by `collect_checksum_stats` below) is
    // `NOT NULL` in the schema so is not affected.
    let rows = pool
        .query(
            r"
        SELECT
            COUNT(*) as total_files,
            SUM(file_size) as total_size,
            (SELECT MAX(last_verified_at) FROM checksums WHERE last_verified_at IS NOT NULL) as last_check_date
        FROM checksums
        ",
            &[],
        )
        .await?;
    let row = rows
        .first()
        .ok_or_else(|| ArchiveError::Validation("summary query returned no row".to_string()))?;

    let total_files = agg_i64(row, "total_files")?;
    let total_size = agg_i64(row, "total_size")?;
    let last_check_str: Option<String> = row.try_get("last_check_date")?;

    let last_check_date = last_check_str
        .map(|s| DateTime::parse_from_rfc3339(&s).map(|dt| dt.with_timezone(&Utc)))
        .transpose()
        .map_err(|e| ArchiveError::Validation(format!("last_check_date decode: {e}")))?;

    // Count verified files
    let verified_rows = pool
        .query(
            r"
        SELECT COUNT(*) as verified
        FROM fixity_checks
        WHERE status = 'pass'
        ",
            &[],
        )
        .await?;
    let verified_files = verified_rows
        .first()
        .map(|r| agg_i64(r, "verified"))
        .transpose()?
        .unwrap_or(0);

    // Count failed files
    let failed_rows = pool
        .query(
            r"
        SELECT COUNT(DISTINCT file_path) as failed
        FROM fixity_checks
        WHERE status = 'fail'
        ",
            &[],
        )
        .await?;
    let failed_files = failed_rows
        .first()
        .map(|r| agg_i64(r, "failed"))
        .transpose()?
        .unwrap_or(0);

    // Count quarantined files
    let quarantine_rows = pool
        .query(
            r"
        SELECT COUNT(*) as quarantined
        FROM quarantine_records
        WHERE restored = 0
        ",
            &[],
        )
        .await?;
    let quarantined_files = quarantine_rows
        .first()
        .map(|r| agg_i64(r, "quarantined"))
        .transpose()?
        .unwrap_or(0);

    Ok(ReportSummary {
        total_files: total_files as usize,
        total_size: total_size as u64,
        verified_files: verified_files as usize,
        failed_files: failed_files as usize,
        quarantined_files: quarantined_files as usize,
        last_check_date,
    })
}

/// Collect checksum statistics
async fn collect_checksum_stats(
    pool: &oxisql_sqlite_compat::SqliteConnection,
) -> ArchiveResult<ChecksumStatistics> {
    let rows = pool
        .query(
            r"
        SELECT
            COUNT(*) as total,
            SUM(CASE WHEN blake3 IS NOT NULL THEN 1 ELSE 0 END) as blake3_count,
            SUM(CASE WHEN md5 IS NOT NULL THEN 1 ELSE 0 END) as md5_count,
            SUM(CASE WHEN sha256 IS NOT NULL THEN 1 ELSE 0 END) as sha256_count,
            SUM(CASE WHEN crc32 IS NOT NULL THEN 1 ELSE 0 END) as crc32_count,
            MIN(created_at) as oldest,
            MAX(created_at) as newest
        FROM checksums
        ",
            &[],
        )
        .await?;
    let row = rows.first().ok_or_else(|| {
        ArchiveError::Validation("checksum stats query returned no row".to_string())
    })?;

    let total = agg_i64(row, "total")?;
    let blake3_count = agg_i64(row, "blake3_count")?;
    let md5_count = agg_i64(row, "md5_count")?;
    let sha256_count = agg_i64(row, "sha256_count")?;
    let crc32_count = agg_i64(row, "crc32_count")?;
    let oldest_str: Option<String> = row.try_get("oldest")?;
    let newest_str: Option<String> = row.try_get("newest")?;

    let oldest_checksum = oldest_str
        .map(|s| DateTime::parse_from_rfc3339(&s).map(|dt| dt.with_timezone(&Utc)))
        .transpose()
        .map_err(|e| ArchiveError::Validation(format!("oldest checksum decode: {e}")))?;

    let newest_checksum = newest_str
        .map(|s| DateTime::parse_from_rfc3339(&s).map(|dt| dt.with_timezone(&Utc)))
        .transpose()
        .map_err(|e| ArchiveError::Validation(format!("newest checksum decode: {e}")))?;

    Ok(ChecksumStatistics {
        total_checksums: total as usize,
        blake3_count: blake3_count as usize,
        md5_count: md5_count as usize,
        sha256_count: sha256_count as usize,
        crc32_count: crc32_count as usize,
        oldest_checksum,
        newest_checksum,
    })
}

/// Collect fixity statistics
async fn collect_fixity_stats(
    pool: &oxisql_sqlite_compat::SqliteConnection,
) -> ArchiveResult<FixityStatistics> {
    let rows = pool
        .query(
            r"
        SELECT
            COUNT(*) as total,
            SUM(CASE WHEN status = 'pass' THEN 1 ELSE 0 END) as passed,
            SUM(CASE WHEN status = 'fail' THEN 1 ELSE 0 END) as failed
        FROM fixity_checks
        ",
            &[],
        )
        .await?;
    let row = rows.first().ok_or_else(|| {
        ArchiveError::Validation("fixity stats query returned no row".to_string())
    })?;

    let total = agg_i64(row, "total")?;
    let passed = agg_i64(row, "passed")?;
    let failed = agg_i64(row, "failed")?;

    let success_rate = if total > 0 {
        (passed as f64) / (total as f64)
    } else {
        0.0
    };

    // Calculate average days between checks
    let avg_days = 90.0; // Placeholder - would need more complex query

    // Count files needing check (placeholder)
    let files_needing_check = 0;

    Ok(FixityStatistics {
        total_checks: total as usize,
        passed_checks: passed as usize,
        failed_checks: failed as usize,
        success_rate,
        average_days_between_checks: avg_days,
        files_needing_check,
    })
}

/// Collect quarantine statistics
async fn collect_quarantine_stats(
    pool: &oxisql_sqlite_compat::SqliteConnection,
) -> ArchiveResult<QuarantineStatistics> {
    let rows = pool
        .query(
            r"
        SELECT
            COUNT(*) as total,
            SUM(CASE WHEN restored = 0 THEN 1 ELSE 0 END) as active,
            SUM(CASE WHEN restored = 1 THEN 1 ELSE 0 END) as restored,
            SUM(CASE WHEN auto_quarantine = 1 THEN 1 ELSE 0 END) as auto_q,
            SUM(CASE WHEN auto_quarantine = 0 THEN 1 ELSE 0 END) as manual_q
        FROM quarantine_records
        ",
            &[],
        )
        .await?;
    let row = rows.first().ok_or_else(|| {
        ArchiveError::Validation("quarantine stats query returned no row".to_string())
    })?;

    let total = agg_i64(row, "total")?;
    let active = agg_i64(row, "active")?;
    let restored = agg_i64(row, "restored")?;
    let auto_q = agg_i64(row, "auto_q")?;
    let manual_q = agg_i64(row, "manual_q")?;

    Ok(QuarantineStatistics {
        total_quarantined: total as usize,
        active_quarantined: active as usize,
        restored: restored as usize,
        auto_quarantined: auto_q as usize,
        manual_quarantined: manual_q as usize,
    })
}

/// Collect recent events
async fn collect_recent_events(
    pool: &oxisql_sqlite_compat::SqliteConnection,
) -> ArchiveResult<Vec<RecentEvent>> {
    let rows = pool
        .query(
            r"
        SELECT event_type, event_date_time, linking_object_id, event_outcome, event_detail
        FROM premis_events
        ORDER BY event_date_time DESC
        LIMIT 50
        ",
            &[],
        )
        .await?;

    let mut events = Vec::new();
    for row in rows {
        let event_date_str: String = row.try_get("event_date_time")?;
        let event_date = DateTime::parse_from_rfc3339(&event_date_str)
            .map_err(|e| ArchiveError::Validation(format!("event_date_time decode: {e}")))?
            .with_timezone(&Utc);

        events.push(RecentEvent {
            event_type: row.try_get("event_type")?,
            event_date,
            file_path: row.try_get("linking_object_id")?,
            outcome: row.try_get("event_outcome")?,
            details: row.try_get("event_detail")?,
        });
    }

    Ok(events)
}

/// Generate alerts based on database state
async fn generate_alerts(
    pool: &oxisql_sqlite_compat::SqliteConnection,
) -> ArchiveResult<Vec<Alert>> {
    let mut alerts = Vec::new();

    // Alert for failed fixity checks
    let failed_checks = pool
        .query(
            r"
        SELECT file_path
        FROM fixity_checks
        WHERE status = 'fail'
        GROUP BY file_path
        HAVING COUNT(*) > 0
        ORDER BY MAX(check_time) DESC
        LIMIT 10
        ",
            &[],
        )
        .await?;

    for row in failed_checks {
        let file_path: String = row.try_get("file_path")?;
        alerts.push(Alert {
            alert_type: AlertType::FixityCheckFailed,
            severity: AlertSeverity::Error,
            message: "Fixity check failed".to_string(),
            file_path: Some(file_path),
            created_at: Utc::now(),
        });
    }

    // Alert for quarantined files
    let quarantined = pool
        .query(
            r"
        SELECT original_path
        FROM quarantine_records
        WHERE restored = 0
        LIMIT 10
        ",
            &[],
        )
        .await?;

    for row in quarantined {
        let file_path: String = row.try_get("original_path")?;
        alerts.push(Alert {
            alert_type: AlertType::FileQuarantined,
            severity: AlertSeverity::Warning,
            message: "File currently quarantined".to_string(),
            file_path: Some(file_path),
            created_at: Utc::now(),
        });
    }

    Ok(alerts)
}

/// Calculate integrity metrics
fn calculate_integrity_metrics(
    summary: &ReportSummary,
    fixity: &FixityStatistics,
    quarantine: &QuarantineStatistics,
) -> IntegrityMetrics {
    let overall_health = if summary.total_files > 0 {
        let healthy_files = summary.total_files.saturating_sub(summary.failed_files);
        (healthy_files as f64) / (summary.total_files as f64)
    } else {
        0.0
    };

    let checksum_coverage = if summary.total_files > 0 {
        (summary.verified_files as f64) / (summary.total_files as f64)
    } else {
        0.0
    };

    let fixity_compliance = fixity.success_rate;

    let quarantine_rate = if summary.total_files > 0 {
        (quarantine.active_quarantined as f64) / (summary.total_files as f64)
    } else {
        0.0
    };

    let data_at_risk = (summary.failed_files as u64) * 1_000_000; // Rough estimate

    IntegrityMetrics {
        overall_health,
        checksum_coverage,
        fixity_compliance,
        quarantine_rate,
        data_at_risk,
    }
}

/// Collect file details
async fn collect_file_details(
    pool: &oxisql_sqlite_compat::SqliteConnection,
) -> ArchiveResult<Vec<FileDetail>> {
    let rows = pool
        .query(
            r"
        SELECT
            c.file_path,
            c.file_size,
            c.last_verified_at,
            COUNT(fc.id) as verification_count,
            SUM(CASE WHEN fc.status = 'fail' THEN 1 ELSE 0 END) as failure_count
        FROM checksums c
        LEFT JOIN fixity_checks fc ON c.file_path = fc.file_path
        GROUP BY c.file_path
        ORDER BY c.last_verified_at DESC
        LIMIT 100
        ",
            &[],
        )
        .await?;

    let mut details = Vec::new();
    for row in rows {
        let file_path: String = row.try_get("file_path")?;
        let file_size: i64 = row.try_get("file_size")?;
        let last_verified_str: Option<String> = row.try_get("last_verified_at")?;
        let verification_count = agg_i64(&row, "verification_count")?;
        let failure_count = agg_i64(&row, "failure_count")?;

        let last_verified = if let Some(s) = last_verified_str {
            Some(
                DateTime::parse_from_rfc3339(&s)
                    .map_err(|e| ArchiveError::Validation(format!("last_verified decode: {e}")))?
                    .with_timezone(&Utc),
            )
        } else {
            None
        };

        let status = if failure_count > 0 {
            FileStatus::Failed
        } else if last_verified.is_none() {
            FileStatus::NeedsVerification
        } else {
            FileStatus::Healthy
        };

        details.push(FileDetail {
            file_path,
            file_size: file_size as u64,
            last_verified,
            verification_count: verification_count as u32,
            failure_count: failure_count as u32,
            status,
        });
    }

    Ok(details)
}

/// Generate HTML report
fn generate_html_report(report: &VerificationReport) -> ArchiveResult<String> {
    let html = format!(
        r#"<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>OxiMedia Archive Verification Report</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; }}
        h1 {{ color: #333; }}
        h2 {{ color: #666; border-bottom: 2px solid #ddd; padding-bottom: 5px; }}
        table {{ border-collapse: collapse; width: 100%; margin: 20px 0; }}
        th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
        th {{ background-color: #f2f2f2; }}
        .summary {{ background-color: #f9f9f9; padding: 15px; border-radius: 5px; }}
        .metric {{ display: inline-block; margin: 10px 20px 10px 0; }}
        .metric-label {{ font-weight: bold; }}
        .alert-critical {{ background-color: #ffcccc; }}
        .alert-error {{ background-color: #ffddcc; }}
        .alert-warning {{ background-color: #ffffcc; }}
        .alert-info {{ background-color: #ccffff; }}
        .status-healthy {{ color: green; }}
        .status-failed {{ color: red; }}
        .status-needs-check {{ color: orange; }}
    </style>
</head>
<body>
    <h1>OxiMedia Archive Verification Report</h1>
    <p>Generated: {}</p>

    <div class="summary">
        <h2>Summary</h2>
        <div class="metric"><span class="metric-label">Total Files:</span> {}</div>
        <div class="metric"><span class="metric-label">Total Size:</span> {} bytes</div>
        <div class="metric"><span class="metric-label">Verified:</span> {}</div>
        <div class="metric"><span class="metric-label">Failed:</span> {}</div>
        <div class="metric"><span class="metric-label">Quarantined:</span> {}</div>
    </div>

    <h2>Integrity Metrics</h2>
    <div class="summary">
        <div class="metric"><span class="metric-label">Overall Health:</span> {:.2}%</div>
        <div class="metric"><span class="metric-label">Checksum Coverage:</span> {:.2}%</div>
        <div class="metric"><span class="metric-label">Fixity Compliance:</span> {:.2}%</div>
        <div class="metric"><span class="metric-label">Quarantine Rate:</span> {:.2}%</div>
    </div>

    <h2>Alerts</h2>
    <table>
        <tr>
            <th>Type</th>
            <th>Severity</th>
            <th>Message</th>
            <th>File</th>
        </tr>
        {}
    </table>

    <h2>Recent Events</h2>
    <table>
        <tr>
            <th>Date</th>
            <th>Type</th>
            <th>File</th>
            <th>Outcome</th>
        </tr>
        {}
    </table>
</body>
</html>"#,
        report.generated_at.to_rfc3339(),
        report.summary.total_files,
        report.summary.total_size,
        report.summary.verified_files,
        report.summary.failed_files,
        report.summary.quarantined_files,
        report.integrity_metrics.overall_health * 100.0,
        report.integrity_metrics.checksum_coverage * 100.0,
        report.integrity_metrics.fixity_compliance * 100.0,
        report.integrity_metrics.quarantine_rate * 100.0,
        generate_alert_rows(&report.alerts),
        generate_event_rows(&report.recent_events),
    );

    Ok(html)
}

/// Generate alert table rows
fn generate_alert_rows(alerts: &[Alert]) -> String {
    alerts
        .iter()
        .map(|alert| {
            let severity_class = match alert.severity {
                AlertSeverity::Critical => "alert-critical",
                AlertSeverity::Error => "alert-error",
                AlertSeverity::Warning => "alert-warning",
                AlertSeverity::Info => "alert-info",
            };
            format!(
                r#"<tr class="{}"><td>{:?}</td><td>{:?}</td><td>{}</td><td>{}</td></tr>"#,
                severity_class,
                alert.alert_type,
                alert.severity,
                alert.message,
                alert.file_path.as_ref().unwrap_or(&String::new())
            )
        })
        .collect::<Vec<_>>()
        .join("\n")
}

/// Generate event table rows
fn generate_event_rows(events: &[RecentEvent]) -> String {
    events
        .iter()
        .map(|event| {
            format!(
                r"<tr><td>{}</td><td>{}</td><td>{}</td><td>{}</td></tr>",
                event.event_date.format("%Y-%m-%d %H:%M:%S"),
                event.event_type,
                event.file_path,
                event.outcome
            )
        })
        .collect::<Vec<_>>()
        .join("\n")
}

/// Generate JSON report
fn generate_json_report(report: &VerificationReport) -> ArchiveResult<String> {
    serde_json::to_string_pretty(report)
        .map_err(|e| ArchiveError::Report(format!("Failed to serialize JSON: {e}")))
}

/// Generate CSV report
fn generate_csv_report(report: &VerificationReport) -> ArchiveResult<String> {
    let mut wtr = csv::Writer::from_writer(vec![]);

    // Write headers
    wtr.write_record([
        "File Path",
        "File Size",
        "Last Verified",
        "Verification Count",
        "Failure Count",
        "Status",
    ])
    .map_err(|e| ArchiveError::Report(format!("CSV write error: {e}")))?;

    // Write file details
    for detail in &report.file_details {
        wtr.write_record([
            &detail.file_path,
            &detail.file_size.to_string(),
            &detail
                .last_verified
                .map(|dt| dt.to_rfc3339())
                .unwrap_or_default(),
            &detail.verification_count.to_string(),
            &detail.failure_count.to_string(),
            &format!("{:?}", detail.status),
        ])
        .map_err(|e| ArchiveError::Report(format!("CSV write error: {e}")))?;
    }

    wtr.flush()
        .map_err(|e| ArchiveError::Report(format!("CSV flush error: {e}")))?;

    String::from_utf8(
        wtr.into_inner()
            .map_err(|e| ArchiveError::Report(format!("CSV error: {e}")))?,
    )
    .map_err(|e| ArchiveError::Report(format!("UTF-8 error: {e}")))
}

/// Generate text report
fn generate_text_report(report: &VerificationReport) -> ArchiveResult<String> {
    let mut text = String::new();

    text.push_str("OxiMedia Archive Verification Report\n");
    text.push_str("=====================================\n\n");
    text.push_str(&format!(
        "Generated: {}\n\n",
        report.generated_at.to_rfc3339()
    ));

    text.push_str("Summary\n");
    text.push_str("-------\n");
    text.push_str(&format!("Total Files: {}\n", report.summary.total_files));
    text.push_str(&format!(
        "Total Size: {} bytes\n",
        report.summary.total_size
    ));
    text.push_str(&format!("Verified: {}\n", report.summary.verified_files));
    text.push_str(&format!("Failed: {}\n", report.summary.failed_files));
    text.push_str(&format!(
        "Quarantined: {}\n\n",
        report.summary.quarantined_files
    ));

    text.push_str("Integrity Metrics\n");
    text.push_str("-----------------\n");
    text.push_str(&format!(
        "Overall Health: {:.2}%\n",
        report.integrity_metrics.overall_health * 100.0
    ));
    text.push_str(&format!(
        "Checksum Coverage: {:.2}%\n",
        report.integrity_metrics.checksum_coverage * 100.0
    ));
    text.push_str(&format!(
        "Fixity Compliance: {:.2}%\n",
        report.integrity_metrics.fixity_compliance * 100.0
    ));
    text.push_str(&format!(
        "Quarantine Rate: {:.2}%\n\n",
        report.integrity_metrics.quarantine_rate * 100.0
    ));

    text.push_str("Alerts\n");
    text.push_str("------\n");
    for alert in &report.alerts {
        text.push_str(&format!(
            "[{:?}] {:?}: {} ({})\n",
            alert.severity,
            alert.alert_type,
            alert.message,
            alert.file_path.as_ref().unwrap_or(&"N/A".to_string())
        ));
    }

    Ok(text)
}

/// OAIS compliance report
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OaisComplianceReport {
    pub report_date: DateTime<Utc>,
    pub preservation_planning: PreservationPlanning,
    pub ingest_compliance: IngestCompliance,
    pub archival_storage: ArchivalStorage,
    pub data_management: DataManagement,
    pub access_compliance: AccessCompliance,
}

/// Preservation planning section
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PreservationPlanning {
    pub format_monitoring: bool,
    pub migration_planning: bool,
    pub risk_assessment: bool,
}

/// Ingest compliance
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IngestCompliance {
    pub checksum_verification: bool,
    pub metadata_extraction: bool,
    pub premis_events: bool,
}

/// Archival storage
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ArchivalStorage {
    pub fixity_checking: bool,
    pub redundancy: bool,
    pub error_detection: bool,
}

/// Data management
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DataManagement {
    pub database_maintained: bool,
    pub audit_trail: bool,
    pub retention_policies: bool,
}

/// Access compliance
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AccessCompliance {
    pub search_capability: bool,
    pub retrieval_capability: bool,
    pub delivery_mechanisms: bool,
}

/// Generate OAIS compliance report
#[allow(dead_code)]
pub async fn generate_oais_report(
    _pool: &oxisql_sqlite_compat::SqliteConnection,
    output_path: &PathBuf,
) -> ArchiveResult<()> {
    let report = OaisComplianceReport {
        report_date: Utc::now(),
        preservation_planning: PreservationPlanning {
            format_monitoring: true,
            migration_planning: false,
            risk_assessment: true,
        },
        ingest_compliance: IngestCompliance {
            checksum_verification: true,
            metadata_extraction: true,
            premis_events: true,
        },
        archival_storage: ArchivalStorage {
            fixity_checking: true,
            redundancy: false,
            error_detection: true,
        },
        data_management: DataManagement {
            database_maintained: true,
            audit_trail: true,
            retention_policies: false,
        },
        access_compliance: AccessCompliance {
            search_capability: true,
            retrieval_capability: true,
            delivery_mechanisms: false,
        },
    };

    let json = serde_json::to_string_pretty(&report)
        .map_err(|e| ArchiveError::Report(format!("Failed to serialize OAIS report: {e}")))?;

    fs::write(output_path, json).await?;

    info!(
        "OAIS compliance report generated: {}",
        output_path.display()
    );
    Ok(())
}
