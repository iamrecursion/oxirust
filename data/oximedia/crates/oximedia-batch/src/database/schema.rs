//! Database schema definitions

/// Database schema version
pub const SCHEMA_VERSION: u32 = 1;

/// SQL for creating jobs table
pub const CREATE_JOBS_TABLE: &str = "
CREATE TABLE IF NOT EXISTS jobs (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    operation TEXT NOT NULL,
    inputs TEXT,
    outputs TEXT,
    priority INTEGER NOT NULL,
    retry_policy TEXT,
    dependencies TEXT,
    schedule TEXT,
    metadata TEXT,
    created_at TEXT NOT NULL,
    started_at TEXT,
    completed_at TEXT,
    status TEXT NOT NULL,
    error TEXT
)";

/// SQL for creating `job_logs` table
pub const CREATE_JOB_LOGS_TABLE: &str = "
CREATE TABLE IF NOT EXISTS job_logs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    job_id TEXT NOT NULL,
    timestamp TEXT NOT NULL,
    level TEXT NOT NULL,
    message TEXT NOT NULL,
    FOREIGN KEY(job_id) REFERENCES jobs(id)
)";

/// SQL for creating `job_results` table
pub const CREATE_JOB_RESULTS_TABLE: &str = "
CREATE TABLE IF NOT EXISTS job_results (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    job_id TEXT NOT NULL,
    input_file TEXT NOT NULL,
    output_files TEXT,
    success INTEGER NOT NULL,
    error TEXT,
    duration REAL,
    FOREIGN KEY(job_id) REFERENCES jobs(id)
)";

/// SQL for creating indexes
pub const CREATE_INDEXES: &[&str] = &[
    "CREATE INDEX IF NOT EXISTS idx_jobs_status ON jobs(status)",
    "CREATE INDEX IF NOT EXISTS idx_jobs_created ON jobs(created_at)",
    "CREATE INDEX IF NOT EXISTS idx_job_logs_job_id ON job_logs(job_id)",
    "CREATE INDEX IF NOT EXISTS idx_job_results_job_id ON job_results(job_id)",
];

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_schema_version() {
        assert_eq!(SCHEMA_VERSION, 1);
    }

    #[test]
    fn test_create_tables_sql() {
        assert!(CREATE_JOBS_TABLE.contains("CREATE TABLE"));
        assert!(CREATE_JOB_LOGS_TABLE.contains("CREATE TABLE"));
        assert!(CREATE_JOB_RESULTS_TABLE.contains("CREATE TABLE"));
    }

    #[test]
    fn test_create_indexes_sql() {
        assert!(!CREATE_INDEXES.is_empty());
        for sql in CREATE_INDEXES {
            assert!(sql.contains("CREATE INDEX"));
        }
    }
}
