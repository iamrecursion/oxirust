//! Usage logging

use crate::{database::RightsDatabase, rights::UsageType, Result};
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use uuid::Uuid;

/// Usage log entry
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UsageLog {
    /// Unique identifier
    pub id: String,
    /// Asset ID
    pub asset_id: String,
    /// Grant ID (if applicable)
    pub grant_id: Option<String>,
    /// Usage type
    pub usage_type: UsageType,
    /// Usage date
    pub usage_date: DateTime<Utc>,
    /// Territory where used
    pub territory: Option<String>,
    /// Platform/medium where used
    pub platform: Option<String>,
    /// Additional metadata
    pub metadata: HashMap<String, String>,
    /// Creation timestamp
    pub created_at: DateTime<Utc>,
}

impl UsageLog {
    /// Create a new usage log entry
    pub fn new(
        asset_id: impl Into<String>,
        usage_type: UsageType,
        usage_date: DateTime<Utc>,
    ) -> Self {
        Self {
            id: Uuid::new_v4().to_string(),
            asset_id: asset_id.into(),
            grant_id: None,
            usage_type,
            usage_date,
            territory: None,
            platform: None,
            metadata: HashMap::new(),
            created_at: Utc::now(),
        }
    }

    /// Set grant ID
    pub fn with_grant(mut self, grant_id: impl Into<String>) -> Self {
        self.grant_id = Some(grant_id.into());
        self
    }

    /// Set territory
    pub fn with_territory(mut self, territory: impl Into<String>) -> Self {
        self.territory = Some(territory.into());
        self
    }

    /// Set platform
    pub fn with_platform(mut self, platform: impl Into<String>) -> Self {
        self.platform = Some(platform.into());
        self
    }

    /// Add metadata
    pub fn add_metadata(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        self.metadata.insert(key.into(), value.into());
        self
    }

    #[cfg(not(target_arch = "wasm32"))]
    /// Decode a `usage_logs` row into a [`UsageLog`].
    fn from_row(r: &oxisql_core::Row) -> Result<Self> {
        let metadata_json: Option<String> = r.try_get("metadata_json")?;
        let metadata = metadata_json
            .and_then(|json| serde_json::from_str(&json).ok())
            .unwrap_or_default();

        let usage_date_s: String = r.try_get("usage_date")?;
        let usage_date = DateTime::parse_from_rfc3339(&usage_date_s)
            .map_err(|e| crate::RightsError::InvalidLicense(format!("Invalid usage_date: {e}")))?
            .with_timezone(&Utc);
        let created_at_s: String = r.try_get("created_at")?;
        let created_at = DateTime::parse_from_rfc3339(&created_at_s)
            .map_err(|e| crate::RightsError::InvalidLicense(format!("Invalid created_at: {e}")))?
            .with_timezone(&Utc);
        let usage_type_s: String = r.try_get("usage_type")?;

        Ok(UsageLog {
            id: r.try_get("id")?,
            asset_id: r.try_get("asset_id")?,
            grant_id: r.try_get("grant_id")?,
            usage_type: UsageType::from_str(&usage_type_s),
            usage_date,
            territory: r.try_get("territory")?,
            platform: r.try_get("platform")?,
            metadata,
            created_at,
        })
    }

    #[cfg(not(target_arch = "wasm32"))]
    /// Save log to database
    pub async fn save(&self, db: &RightsDatabase) -> Result<()> {
        let metadata_json = serde_json::to_string(&self.metadata)
            .map_err(|e| crate::RightsError::Serialization(e.to_string()))?;

        db.pool()
            .execute(
                r"
            INSERT INTO usage_logs
            (id, asset_id, grant_id, usage_type, usage_date, territory, platform, metadata_json, created_at)
            VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9)
            ",
                &[
                    &self.id,
                    &self.asset_id,
                    &self.grant_id,
                    &self.usage_type.as_str(),
                    &self.usage_date.to_rfc3339(),
                    &self.territory,
                    &self.platform,
                    &metadata_json,
                    &self.created_at.to_rfc3339(),
                ],
            )
            .await?;

        Ok(())
    }

    #[cfg(not(target_arch = "wasm32"))]
    /// Load log from database by ID
    pub async fn load(db: &RightsDatabase, id: &str) -> Result<Option<Self>> {
        let row = db
            .pool()
            .query_optional(
                r"
            SELECT id, asset_id, grant_id, usage_type, usage_date, territory, platform, metadata_json, created_at
            FROM usage_logs WHERE id = $1
            ",
                &[&id],
            )
            .await?;

        row.map(|r| Self::from_row(&r)).transpose()
    }

    #[cfg(not(target_arch = "wasm32"))]
    /// List logs for an asset
    pub async fn list_for_asset(db: &RightsDatabase, asset_id: &str) -> Result<Vec<Self>> {
        let rows = db
            .pool()
            .query(
                r"
            SELECT id, asset_id, grant_id, usage_type, usage_date, territory, platform, metadata_json, created_at
            FROM usage_logs WHERE asset_id = $1
            ORDER BY usage_date DESC
            ",
                &[&asset_id],
            )
            .await?;

        rows.iter().map(Self::from_row).collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_usage_log_creation() {
        let log = UsageLog::new("asset1", UsageType::Commercial, Utc::now())
            .with_territory("US")
            .with_platform("Website")
            .add_metadata("campaign", "Summer 2024");

        assert_eq!(log.asset_id, "asset1");
        assert_eq!(log.territory, Some("US".to_string()));
        assert_eq!(log.platform, Some("Website".to_string()));
        assert_eq!(
            log.metadata.get("campaign"),
            Some(&"Summer 2024".to_string())
        );
    }

    #[tokio::test]
    async fn test_usage_log_save() {
        let temp_dir = tempfile::tempdir().expect("rights test operation should succeed");
        let db_path = format!("sqlite://{}/test.db", temp_dir.path().display());
        let db = RightsDatabase::new(&db_path)
            .await
            .expect("rights test operation should succeed");

        // Create asset first to satisfy foreign key constraint
        let asset = crate::rights::Asset::new("Test Asset", crate::rights::AssetType::Video);
        let asset_id = asset.id.clone();
        asset
            .save(&db)
            .await
            .expect("rights test operation should succeed");

        let log = UsageLog::new(&asset_id, UsageType::Web, Utc::now());
        let log_id = log.id.clone();

        log.save(&db)
            .await
            .expect("rights test operation should succeed");

        let loaded = UsageLog::load(&db, &log_id)
            .await
            .expect("rights test operation should succeed");
        assert!(loaded.is_some());
    }
}
