//! Audit trail logging

#[cfg(not(target_arch = "wasm32"))]
use crate::database::RightsDatabase;
use crate::Result;
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use uuid::Uuid;

/// Audit entry
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AuditEntry {
    /// Unique identifier
    pub id: String,
    /// Entity type (e.g., "grant", "asset")
    pub entity_type: String,
    /// Entity ID
    pub entity_id: String,
    /// Action performed
    pub action: String,
    /// User ID (if applicable)
    pub user_id: Option<String>,
    /// Changes made
    pub changes: HashMap<String, String>,
    /// Timestamp
    pub timestamp: DateTime<Utc>,
    /// IP address
    pub ip_address: Option<String>,
}

impl AuditEntry {
    /// Create a new audit entry
    pub fn new(
        entity_type: impl Into<String>,
        entity_id: impl Into<String>,
        action: impl Into<String>,
    ) -> Self {
        Self {
            id: Uuid::new_v4().to_string(),
            entity_type: entity_type.into(),
            entity_id: entity_id.into(),
            action: action.into(),
            user_id: None,
            changes: HashMap::new(),
            timestamp: Utc::now(),
            ip_address: None,
        }
    }

    /// Set user ID
    pub fn with_user(mut self, user_id: impl Into<String>) -> Self {
        self.user_id = Some(user_id.into());
        self
    }

    /// Add a change
    pub fn add_change(mut self, field: impl Into<String>, value: impl Into<String>) -> Self {
        self.changes.insert(field.into(), value.into());
        self
    }

    /// Set IP address
    pub fn with_ip(mut self, ip: impl Into<String>) -> Self {
        self.ip_address = Some(ip.into());
        self
    }

    #[cfg(not(target_arch = "wasm32"))]
    /// Decode an `audit_trail` row into an [`AuditEntry`].
    pub(crate) fn from_row(r: &oxisql_core::Row) -> Result<Self> {
        let changes_json: Option<String> = r.try_get("changes_json")?;
        let changes = changes_json
            .and_then(|json| serde_json::from_str(&json).ok())
            .unwrap_or_default();

        let timestamp_s: String = r.try_get("timestamp")?;
        let timestamp = DateTime::parse_from_rfc3339(&timestamp_s)
            .unwrap_or_else(|_| Utc::now().fixed_offset())
            .with_timezone(&Utc);

        Ok(AuditEntry {
            id: r.try_get("id")?,
            entity_type: r.try_get("entity_type")?,
            entity_id: r.try_get("entity_id")?,
            action: r.try_get("action")?,
            user_id: r.try_get("user_id")?,
            changes,
            timestamp,
            ip_address: r.try_get("ip_address")?,
        })
    }

    #[cfg(not(target_arch = "wasm32"))]
    /// Save to database
    pub async fn save(&self, db: &RightsDatabase) -> Result<()> {
        let changes_json = serde_json::to_string(&self.changes)
            .map_err(|e| crate::RightsError::Serialization(e.to_string()))?;

        db.pool()
            .execute(
                r"
            INSERT INTO audit_trail
            (id, entity_type, entity_id, action, user_id, changes_json, timestamp, ip_address)
            VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
            ",
                &[
                    &self.id,
                    &self.entity_type,
                    &self.entity_id,
                    &self.action,
                    &self.user_id,
                    &changes_json,
                    &self.timestamp.to_rfc3339(),
                    &self.ip_address,
                ],
            )
            .await?;

        Ok(())
    }
}

/// Audit trail manager
pub struct AuditTrail<'a> {
    db: &'a RightsDatabase,
}

impl<'a> AuditTrail<'a> {
    /// Create a new audit trail manager
    pub fn new(db: &'a RightsDatabase) -> Self {
        Self { db }
    }

    /// Log an audit entry
    pub async fn log(&self, entry: AuditEntry) -> Result<()> {
        entry.save(self.db).await
    }

    /// Get audit entries for an entity
    pub async fn get_for_entity(
        &self,
        entity_type: &str,
        entity_id: &str,
    ) -> Result<Vec<AuditEntry>> {
        let rows = self
            .db
            .pool()
            .query(
                r"
            SELECT id, entity_type, entity_id, action, user_id, changes_json, timestamp, ip_address
            FROM audit_trail
            WHERE entity_type = $1 AND entity_id = $2
            ORDER BY timestamp DESC
            ",
                &[&entity_type, &entity_id],
            )
            .await?;

        rows.iter().map(AuditEntry::from_row).collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_audit_entry_creation() {
        let entry = AuditEntry::new("grant", "grant123", "create")
            .with_user("user1")
            .add_change("status", "active");

        assert_eq!(entry.entity_type, "grant");
        assert_eq!(entry.action, "create");
        assert_eq!(entry.user_id, Some("user1".to_string()));
    }

    #[tokio::test]
    async fn test_audit_logging() {
        let temp_dir = tempfile::tempdir().expect("rights test operation should succeed");
        let db_path = format!("sqlite://{}/test.db", temp_dir.path().display());
        let db = RightsDatabase::new(&db_path)
            .await
            .expect("rights test operation should succeed");

        let trail = AuditTrail::new(&db);
        let entry = AuditEntry::new("asset", "asset1", "update");

        trail
            .log(entry)
            .await
            .expect("rights test operation should succeed");

        let entries = trail
            .get_for_entity("asset", "asset1")
            .await
            .expect("rights test operation should succeed");
        assert_eq!(entries.len(), 1);
    }
}
