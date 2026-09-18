//! Audit log export

use crate::{audit::AuditEntry, database::RightsDatabase, Result};
use chrono::{DateTime, Utc};

/// Audit exporter
pub struct AuditExporter<'a> {
    db: &'a RightsDatabase,
}

impl<'a> AuditExporter<'a> {
    /// Create a new audit exporter
    pub fn new(db: &'a RightsDatabase) -> Self {
        Self { db }
    }

    /// Export audit entries to JSON
    pub async fn export_json(
        &self,
        start_date: DateTime<Utc>,
        end_date: DateTime<Utc>,
    ) -> Result<String> {
        let entries = self.get_entries_in_range(start_date, end_date).await?;
        serde_json::to_string_pretty(&entries)
            .map_err(|e| crate::RightsError::Serialization(e.to_string()))
    }

    /// Get entries within a date range
    async fn get_entries_in_range(
        &self,
        start_date: DateTime<Utc>,
        end_date: DateTime<Utc>,
    ) -> Result<Vec<AuditEntry>> {
        let rows = self
            .db
            .pool()
            .query(
                r"
            SELECT id, entity_type, entity_id, action, user_id, changes_json, timestamp, ip_address
            FROM audit_trail
            WHERE timestamp >= $1 AND timestamp <= $2
            ORDER BY timestamp DESC
            ",
                &[&start_date.to_rfc3339(), &end_date.to_rfc3339()],
            )
            .await?;

        rows.iter().map(AuditEntry::from_row).collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::audit::AuditTrail;

    #[tokio::test]
    async fn test_audit_export() {
        let temp_dir = tempfile::tempdir().expect("rights test operation should succeed");
        let db_path = format!("sqlite://{}/test.db", temp_dir.path().display());
        let db = RightsDatabase::new(&db_path)
            .await
            .expect("rights test operation should succeed");

        let trail = AuditTrail::new(&db);
        let entry = crate::audit::AuditEntry::new("asset", "asset1", "create");
        trail
            .log(entry)
            .await
            .expect("rights test operation should succeed");

        let exporter = AuditExporter::new(&db);
        let now = Utc::now();
        let json = exporter
            .export_json(
                now - chrono::Duration::hours(1),
                now + chrono::Duration::hours(1),
            )
            .await
            .expect("rights test operation should succeed");

        assert!(json.contains("asset1"));
    }
}
