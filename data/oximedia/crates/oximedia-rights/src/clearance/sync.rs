//! Sync rights management

use crate::{clearance::ClearanceStatus, database::RightsDatabase, Result};
use serde::{Deserialize, Serialize};
use uuid::Uuid;

/// Sync rights (music synchronization)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SyncRights {
    /// Unique identifier
    pub id: String,
    /// Asset ID
    pub asset_id: String,
    /// Composition
    pub composition: String,
    /// Status
    pub status: ClearanceStatus,
    /// Rights holder
    pub rights_holder: Option<String>,
}

impl SyncRights {
    /// Create new sync rights
    pub fn new(asset_id: impl Into<String>, composition: impl Into<String>) -> Self {
        Self {
            id: Uuid::new_v4().to_string(),
            asset_id: asset_id.into(),
            composition: composition.into(),
            status: ClearanceStatus::Requested,
            rights_holder: None,
        }
    }

    /// Save to database (simplified)
    pub async fn save(&self, _db: &RightsDatabase) -> Result<()> {
        Ok(())
    }
}
