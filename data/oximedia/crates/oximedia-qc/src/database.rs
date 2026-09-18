//! Database storage for QC results and history.
//!
//! Provides SQLite-based persistence for QC reports,
//! enabling historical tracking and trend analysis.

#[cfg(feature = "database")]
use oxisql_core::{ToSqlValue, Value};
#[cfg(feature = "database")]
use oxisql_sqlite_compat::SqliteConnectionBlocking;

use crate::report::QcReport;
use oximedia_core::{OxiError, OxiResult};
use std::path::Path;

#[cfg(feature = "database")]
fn map_oxi(e: impl std::fmt::Display) -> OxiError {
    OxiError::Io(std::io::Error::other(format!("SQLite error: {e}")))
}

#[cfg(feature = "database")]
fn col_i64(row: &oxisql_core::Row, idx: usize) -> OxiResult<i64> {
    match row.get_by_index(idx) {
        Some(Value::I64(n)) => Ok(*n),
        Some(Value::Null) | None => Ok(0),
        Some(other) => Err(OxiError::Io(std::io::Error::other(format!(
            "column {idx}: expected integer, got {}",
            other.type_name()
        )))),
    }
}

#[cfg(feature = "database")]
fn col_real_or_zero(row: &oxisql_core::Row, idx: usize) -> f64 {
    match row.get_by_index(idx) {
        Some(Value::F64(f)) => *f,
        Some(Value::I64(n)) => *n as f64,
        _ => 0.0,
    }
}

#[cfg(feature = "database")]
fn col_opt_text(row: &oxisql_core::Row, idx: usize) -> Option<String> {
    match row.get_by_index(idx) {
        Some(Value::Text(s)) => Some(s.clone()),
        _ => None,
    }
}

/// Database for QC results.
#[cfg(feature = "database")]
pub struct QcDatabase {
    conn: SqliteConnectionBlocking,
}

#[cfg(feature = "database")]
impl QcDatabase {
    /// Opens or creates a QC database at the specified path.
    ///
    /// # Errors
    ///
    /// Returns an error if the database cannot be opened or initialized.
    pub fn open<P: AsRef<Path>>(path: P) -> OxiResult<Self> {
        let path_str = path.as_ref().to_string_lossy().into_owned();
        let conn = SqliteConnectionBlocking::open(&path_str).map_err(|e| {
            OxiError::Io(std::io::Error::other(format!(
                "Failed to open database: {e}"
            )))
        })?;
        let db = Self { conn };
        db.initialize_schema()?;
        Ok(db)
    }

    /// Creates an in-memory database for testing.
    ///
    /// # Errors
    ///
    /// Returns an error if the database cannot be created.
    pub fn in_memory() -> OxiResult<Self> {
        let conn = SqliteConnectionBlocking::open_memory().map_err(|e| {
            OxiError::Io(std::io::Error::other(format!(
                "Failed to create in-memory database: {e}"
            )))
        })?;
        let db = Self { conn };
        db.initialize_schema()?;
        Ok(db)
    }

    fn exec(&self, sql: &str, params: &[&dyn ToSqlValue]) -> OxiResult<u64> {
        self.conn.execute(sql, params).map_err(map_oxi)
    }

    fn query_rows(
        &self,
        sql: &str,
        params: &[&dyn ToSqlValue],
    ) -> OxiResult<Vec<oxisql_core::Row>> {
        self.conn.query(sql, params).map_err(map_oxi)
    }

    /// Initializes the database schema.
    fn initialize_schema(&self) -> OxiResult<()> {
        self.conn
            .execute_batch(
                "CREATE TABLE IF NOT EXISTS qc_reports (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                file_path TEXT NOT NULL,
                timestamp TEXT NOT NULL,
                overall_passed INTEGER NOT NULL,
                total_checks INTEGER NOT NULL,
                passed_checks INTEGER NOT NULL,
                failed_checks INTEGER NOT NULL,
                validation_duration REAL,
                report_json TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS qc_issues (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                report_id INTEGER NOT NULL,
                rule_name TEXT NOT NULL,
                severity TEXT NOT NULL,
                message TEXT NOT NULL,
                stream_index INTEGER,
                timestamp_seconds REAL,
                recommendation TEXT,
                FOREIGN KEY (report_id) REFERENCES qc_reports(id)
            );
            CREATE INDEX IF NOT EXISTS idx_qc_reports_file_path ON qc_reports(file_path);
            CREATE INDEX IF NOT EXISTS idx_qc_reports_timestamp ON qc_reports(timestamp);",
            )
            .map(|_| ())
            .map_err(|e| {
                OxiError::Io(std::io::Error::other(format!(
                    "Failed to create schema: {e}"
                )))
            })
    }

    /// Stores a QC report in the database.
    ///
    /// # Errors
    ///
    /// Returns an error if the report cannot be stored.
    #[cfg(feature = "json")]
    pub fn store_report(&mut self, report: &QcReport) -> OxiResult<i64> {
        let report_json = report.to_json().map_err(|e| {
            OxiError::Io(std::io::Error::other(format!(
                "Failed to serialize report: {e}"
            )))
        })?;

        let overall_passed_i = report.overall_passed as i64;
        let total_checks_i = report.total_checks as i64;
        let passed_checks_i = report.passed_checks as i64;
        let failed_checks_i = report.failed_checks as i64;

        self.exec(
            "INSERT INTO qc_reports (
                file_path, timestamp, overall_passed, total_checks,
                passed_checks, failed_checks, validation_duration, report_json
            ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8)",
            &[
                &report.file_path.as_str(),
                &report.timestamp.as_str(),
                &overall_passed_i,
                &total_checks_i,
                &passed_checks_i,
                &failed_checks_i,
                &report.validation_duration,
                &report_json,
            ],
        )
        .map_err(|e| {
            OxiError::Io(std::io::Error::other(format!(
                "Failed to insert report: {e}"
            )))
        })?;

        let rows = self
            .query_rows("SELECT last_insert_rowid()", &[])
            .map_err(|e| {
                OxiError::Io(std::io::Error::other(format!(
                    "Failed to get report id: {e}"
                )))
            })?;
        let report_id = match rows.first().and_then(|r| r.get_by_index(0)) {
            Some(Value::I64(n)) => *n,
            _ => {
                return Err(OxiError::Io(std::io::Error::other(
                    "Failed to get last_insert_rowid",
                )))
            }
        };

        // Store individual issues
        for result in &report.results {
            if !result.passed {
                let stream_index_i = result.stream_index.map(|idx| idx as i64);
                self.exec(
                    "INSERT INTO qc_issues (
                        report_id, rule_name, severity, message,
                        stream_index, timestamp_seconds, recommendation
                    ) VALUES ($1, $2, $3, $4, $5, $6, $7)",
                    &[
                        &report_id,
                        &result.rule_name.as_str(),
                        &result.severity.to_string().as_str(),
                        &result.message.as_str(),
                        &stream_index_i,
                        &result.timestamp,
                        &result.recommendation.as_deref(),
                    ],
                )
                .map_err(|e| {
                    OxiError::Io(std::io::Error::other(format!(
                        "Failed to insert issue: {e}"
                    )))
                })?;
            }
        }

        Ok(report_id)
    }

    /// Retrieves QC reports for a specific file.
    ///
    /// # Errors
    ///
    /// Returns an error if the query fails.
    #[cfg(feature = "json")]
    pub fn get_reports_for_file(&self, file_path: &str) -> OxiResult<Vec<QcReport>> {
        let rows = self
            .query_rows(
                "SELECT report_json FROM qc_reports WHERE file_path = $1 ORDER BY timestamp DESC",
                &[&file_path],
            )
            .map_err(|e| {
                OxiError::Io(std::io::Error::other(format!(
                    "Failed to execute query: {e}"
                )))
            })?;

        let mut result = Vec::new();
        for row in &rows {
            let json = match row.get_by_index(0) {
                Some(Value::Text(s)) => s.clone(),
                _ => {
                    return Err(OxiError::Io(std::io::Error::other(
                        "Failed to read report_json row",
                    )))
                }
            };
            let report: QcReport = serde_json::from_str(&json).map_err(|e| {
                OxiError::Io(std::io::Error::other(format!(
                    "Failed to deserialize report: {e}"
                )))
            })?;
            result.push(report);
        }

        Ok(result)
    }

    /// Gets statistics for a file path.
    ///
    /// # Errors
    ///
    /// Returns an error if the query fails.
    pub fn get_file_statistics(&self, file_path: &str) -> OxiResult<FileStatistics> {
        let rows = self
            .query_rows(
                "SELECT COUNT(*) as total_runs,
                 COALESCE(SUM(overall_passed), 0) as passed_runs,
                 AVG(validation_duration) as avg_duration,
                 MAX(timestamp) as last_check
                 FROM qc_reports WHERE file_path = $1",
                &[&file_path],
            )
            .map_err(|e| {
                OxiError::Io(std::io::Error::other(format!(
                    "Failed to execute query: {e}"
                )))
            })?;

        let row = rows.first().ok_or_else(|| {
            OxiError::Io(std::io::Error::other(
                "No rows returned for file statistics",
            ))
        })?;

        Ok(FileStatistics {
            file_path: file_path.to_string(),
            total_runs: col_i64(row, 0)? as usize,
            passed_runs: col_i64(row, 1)? as usize,
            avg_duration: col_real_or_zero(row, 2),
            last_check: col_opt_text(row, 3),
        })
    }

    /// Gets overall database statistics.
    ///
    /// # Errors
    ///
    /// Returns an error if the query fails.
    pub fn get_overall_statistics(&self) -> OxiResult<OverallStatistics> {
        let rows = self
            .query_rows(
                "SELECT COUNT(DISTINCT file_path) as total_files,
                 COUNT(*) as total_runs,
                 SUM(overall_passed) as passed_runs,
                 AVG(validation_duration) as avg_duration
                 FROM qc_reports",
                &[],
            )
            .map_err(|e| {
                OxiError::Io(std::io::Error::other(format!(
                    "Failed to execute query: {e}"
                )))
            })?;

        let row = rows.first().ok_or_else(|| {
            OxiError::Io(std::io::Error::other(
                "No rows returned for overall statistics",
            ))
        })?;

        Ok(OverallStatistics {
            total_files: col_i64(row, 0)? as usize,
            total_runs: col_i64(row, 1)? as usize,
            passed_runs: col_i64(row, 2)? as usize,
            avg_duration: col_real_or_zero(row, 3),
        })
    }
}

/// Statistics for a specific file.
#[derive(Debug, Clone)]
pub struct FileStatistics {
    /// File path.
    pub file_path: String,
    /// Total number of QC runs.
    pub total_runs: usize,
    /// Number of runs that passed.
    pub passed_runs: usize,
    /// Average validation duration.
    pub avg_duration: f64,
    /// Timestamp of last check.
    pub last_check: Option<String>,
}

/// Overall database statistics.
#[derive(Debug, Clone)]
pub struct OverallStatistics {
    /// Total number of unique files.
    pub total_files: usize,
    /// Total number of QC runs.
    pub total_runs: usize,
    /// Number of runs that passed.
    pub passed_runs: usize,
    /// Average validation duration.
    pub avg_duration: f64,
}

#[cfg(all(test, feature = "database", feature = "json"))]
mod tests {
    use super::*;
    use crate::QcReport;

    #[test]
    fn test_database_creation() {
        let db = QcDatabase::in_memory();
        assert!(db.is_ok());
    }

    #[test]
    fn test_store_and_retrieve_report() {
        let mut db = QcDatabase::in_memory().expect("should succeed in test");
        let report = QcReport::new("test.mkv");

        let id = db.store_report(&report).expect("should succeed in test");
        assert!(id > 0);

        let reports = db
            .get_reports_for_file("test.mkv")
            .expect("should succeed in test");
        assert_eq!(reports.len(), 1);
    }

    #[test]
    fn test_file_statistics() {
        let db = QcDatabase::in_memory().expect("should succeed in test");
        let stats = db
            .get_file_statistics("test.mkv")
            .expect("should succeed in test");
        assert_eq!(stats.total_runs, 0);
    }
}
