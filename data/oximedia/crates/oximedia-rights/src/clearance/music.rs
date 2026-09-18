//! Music clearance tracking

#[cfg(not(target_arch = "wasm32"))]
use crate::database::RightsDatabase;
use crate::{
    clearance::{ClearanceStatus, ClearanceType},
    Result,
};
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use uuid::Uuid;

/// Music clearance
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MusicClearance {
    /// Unique identifier
    pub id: String,
    /// Asset ID
    pub asset_id: String,
    /// Track name
    pub track_name: String,
    /// Artist/composer
    pub artist: String,
    /// Publisher
    pub publisher: Option<String>,
    /// Status
    pub status: ClearanceStatus,
    /// Requested date
    pub requested_date: DateTime<Utc>,
    /// Approved date
    pub approved_date: Option<DateTime<Utc>>,
    /// Expiry date
    pub expiry_date: Option<DateTime<Utc>>,
    /// Notes
    pub notes: Option<String>,
}

impl MusicClearance {
    /// Create a new music clearance
    pub fn new(
        asset_id: impl Into<String>,
        track_name: impl Into<String>,
        artist: impl Into<String>,
    ) -> Self {
        Self {
            id: Uuid::new_v4().to_string(),
            asset_id: asset_id.into(),
            track_name: track_name.into(),
            artist: artist.into(),
            publisher: None,
            status: ClearanceStatus::Requested,
            requested_date: Utc::now(),
            approved_date: None,
            expiry_date: None,
            notes: None,
        }
    }

    /// Approve clearance
    pub fn approve(&mut self) {
        self.status = ClearanceStatus::Cleared;
        self.approved_date = Some(Utc::now());
    }

    #[cfg(not(target_arch = "wasm32"))]
    /// Save to database
    pub async fn save(&self, db: &RightsDatabase) -> Result<()> {
        let metadata = serde_json::json!({
            "track_name": self.track_name,
            "artist": self.artist,
            "publisher": self.publisher,
        });

        db.pool()
            .execute(
                r"
            INSERT INTO clearances
            (id, asset_id, clearance_type, status, requester, approver, requested_date,
             approved_date, expiry_date, notes, metadata_json, created_at, updated_at)
            VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13)
            ON CONFLICT(id) DO UPDATE SET
                status = excluded.status,
                approved_date = excluded.approved_date,
                notes = excluded.notes,
                updated_at = excluded.updated_at
            ",
                &[
                    &self.id,
                    &self.asset_id,
                    &ClearanceType::Music.as_str(),
                    &self.status.as_str(),
                    &None::<String>,
                    &None::<String>,
                    &self.requested_date.to_rfc3339(),
                    &self.approved_date.map(|d| d.to_rfc3339()),
                    &self.expiry_date.map(|d| d.to_rfc3339()),
                    &self.notes,
                    &metadata.to_string(),
                    &Utc::now().to_rfc3339(),
                    &Utc::now().to_rfc3339(),
                ],
            )
            .await?;

        Ok(())
    }
}
