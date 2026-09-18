//! Expiration alerts

#[cfg(not(target_arch = "wasm32"))]
use crate::database::RightsDatabase;
use crate::Result;
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use uuid::Uuid;

/// Type of expiration alert
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum AlertType {
    /// Warning before expiration
    Warning,
    /// Critical - expiring very soon
    Critical,
    /// Expired
    Expired,
}

impl AlertType {
    /// Convert to string representation
    pub fn as_str(&self) -> &str {
        match self {
            AlertType::Warning => "warning",
            AlertType::Critical => "critical",
            AlertType::Expired => "expired",
        }
    }

    /// Parse from string
    #[allow(clippy::should_implement_trait)]
    pub fn from_str(s: &str) -> Self {
        match s {
            "warning" => AlertType::Warning,
            "critical" => AlertType::Critical,
            "expired" => AlertType::Expired,
            _ => AlertType::Warning,
        }
    }
}

/// Expiration alert
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExpirationAlert {
    /// Unique identifier
    pub id: String,
    /// Associated rights grant ID
    pub grant_id: String,
    /// Alert type
    pub alert_type: AlertType,
    /// Alert date (when alert should be shown/sent)
    pub alert_date: DateTime<Utc>,
    /// Whether notification has been sent
    pub notification_sent: bool,
    /// Creation timestamp
    pub created_at: DateTime<Utc>,
}

impl ExpirationAlert {
    /// Create a new expiration alert
    pub fn new(
        grant_id: impl Into<String>,
        alert_type: AlertType,
        alert_date: DateTime<Utc>,
    ) -> Self {
        Self {
            id: Uuid::new_v4().to_string(),
            grant_id: grant_id.into(),
            alert_type,
            alert_date,
            notification_sent: false,
            created_at: Utc::now(),
        }
    }

    /// Mark notification as sent
    pub fn mark_sent(&mut self) {
        self.notification_sent = true;
    }

    #[cfg(not(target_arch = "wasm32"))]
    /// Decode an `expiration_alerts` row into an [`ExpirationAlert`].
    fn from_row(r: &oxisql_core::Row) -> Result<Self> {
        let alert_date_s: String = r.try_get("alert_date")?;
        let alert_date = DateTime::parse_from_rfc3339(&alert_date_s)
            .map(|dt| dt.with_timezone(&Utc))
            .map_err(|e| crate::RightsError::Serialization(e.to_string()))?;
        let created_at_s: String = r.try_get("created_at")?;
        let created_at = DateTime::parse_from_rfc3339(&created_at_s)
            .map(|dt| dt.with_timezone(&Utc))
            .map_err(|e| crate::RightsError::Serialization(e.to_string()))?;
        let alert_type_s: String = r.try_get("alert_type")?;
        Ok(ExpirationAlert {
            id: r.try_get("id")?,
            grant_id: r.try_get("grant_id")?,
            alert_type: AlertType::from_str(&alert_type_s),
            alert_date,
            notification_sent: r.try_get::<i64>("notification_sent")? != 0,
            created_at,
        })
    }

    #[cfg(not(target_arch = "wasm32"))]
    /// Save alert to database
    pub async fn save(&self, db: &RightsDatabase) -> Result<()> {
        db.pool()
            .execute(
                r"
            INSERT INTO expiration_alerts
            (id, grant_id, alert_type, alert_date, notification_sent, created_at)
            VALUES ($1, $2, $3, $4, $5, $6)
            ON CONFLICT(id) DO UPDATE SET
                grant_id = excluded.grant_id,
                alert_type = excluded.alert_type,
                alert_date = excluded.alert_date,
                notification_sent = excluded.notification_sent
            ",
                &[
                    &self.id,
                    &self.grant_id,
                    &self.alert_type.as_str(),
                    &self.alert_date.to_rfc3339(),
                    &i64::from(self.notification_sent),
                    &self.created_at.to_rfc3339(),
                ],
            )
            .await?;

        Ok(())
    }

    #[cfg(not(target_arch = "wasm32"))]
    /// Load alert from database by ID
    pub async fn load(db: &RightsDatabase, id: &str) -> Result<Option<Self>> {
        let row = db
            .pool()
            .query_optional(
                r"
            SELECT id, grant_id, alert_type, alert_date, notification_sent, created_at
            FROM expiration_alerts WHERE id = $1
            ",
                &[&id],
            )
            .await?;

        row.map(|r| Self::from_row(&r)).transpose()
    }

    #[cfg(not(target_arch = "wasm32"))]
    /// Get pending alerts (not yet sent and alert date has passed)
    pub async fn get_pending_alerts(db: &RightsDatabase) -> Result<Vec<Self>> {
        let now = Utc::now();

        let rows = db
            .pool()
            .query(
                r"
            SELECT id, grant_id, alert_type, alert_date, notification_sent, created_at
            FROM expiration_alerts
            WHERE notification_sent = 0 AND alert_date <= $1
            ORDER BY alert_date ASC
            ",
                &[&now.to_rfc3339()],
            )
            .await?;

        rows.iter().map(Self::from_row).collect()
    }

    #[cfg(not(target_arch = "wasm32"))]
    /// Get all alerts for a grant
    pub async fn list_for_grant(db: &RightsDatabase, grant_id: &str) -> Result<Vec<Self>> {
        let rows = db
            .pool()
            .query(
                r"
            SELECT id, grant_id, alert_type, alert_date, notification_sent, created_at
            FROM expiration_alerts WHERE grant_id = $1
            ORDER BY alert_date DESC
            ",
                &[&grant_id],
            )
            .await?;

        rows.iter().map(Self::from_row).collect()
    }

    #[cfg(not(target_arch = "wasm32"))]
    /// Delete alert from database
    pub async fn delete(db: &RightsDatabase, id: &str) -> Result<()> {
        db.pool()
            .execute("DELETE FROM expiration_alerts WHERE id = $1", &[&id])
            .await?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_alert_creation() {
        let now = Utc::now();
        let alert = ExpirationAlert::new("grant123", AlertType::Warning, now);

        assert_eq!(alert.grant_id, "grant123");
        assert_eq!(alert.alert_type, AlertType::Warning);
        assert!(!alert.notification_sent);
    }

    #[test]
    fn test_mark_sent() {
        let now = Utc::now();
        let mut alert = ExpirationAlert::new("grant123", AlertType::Warning, now);

        alert.mark_sent();
        assert!(alert.notification_sent);
    }

    #[tokio::test]
    async fn test_alert_save_and_load() {
        let temp_dir = tempfile::tempdir().expect("rights test operation should succeed");
        let db_path = format!("sqlite://{}/test.db", temp_dir.path().display());
        let db = RightsDatabase::new(&db_path)
            .await
            .expect("rights test operation should succeed");

        // Create asset and owner first
        let asset = crate::rights::Asset::new("Test Asset", crate::rights::AssetType::Video);
        asset
            .save(&db)
            .await
            .expect("rights test operation should succeed");
        let owner = crate::rights::RightsOwner::new("Test Owner");
        owner
            .save(&db)
            .await
            .expect("rights test operation should succeed");

        // Create grant
        let grant = crate::rights::RightsGrant::new(
            &asset.id,
            &owner.id,
            crate::license::LicenseType::Exclusive,
            Utc::now(),
            Some(Utc::now() + chrono::Duration::days(30)),
            true,
        );
        let grant_id = grant.id.clone();
        grant
            .save(&db)
            .await
            .expect("rights test operation should succeed");

        let alert = ExpirationAlert::new(&grant_id, AlertType::Critical, Utc::now());
        let alert_id = alert.id.clone();

        alert
            .save(&db)
            .await
            .expect("rights test operation should succeed");

        let loaded = ExpirationAlert::load(&db, &alert_id)
            .await
            .expect("rights test operation should succeed");
        assert!(loaded.is_some());
        let loaded = loaded.expect("rights test operation should succeed");
        assert_eq!(loaded.alert_type, AlertType::Critical);
    }
}
