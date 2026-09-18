//! Stock footage clearance tracking

use crate::{clearance::ClearanceStatus, database::RightsDatabase, Result};
use serde::{Deserialize, Serialize};
use uuid::Uuid;

/// Stock footage clearance
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FootageClearance {
    /// Unique identifier
    pub id: String,
    /// Asset ID
    pub asset_id: String,
    /// Footage source
    pub source: String,
    /// Stock library
    pub library: Option<String>,
    /// Status
    pub status: ClearanceStatus,
    /// License number
    pub license_number: Option<String>,
}

impl FootageClearance {
    /// Create new footage clearance
    pub fn new(asset_id: impl Into<String>, source: impl Into<String>) -> Self {
        Self {
            id: Uuid::new_v4().to_string(),
            asset_id: asset_id.into(),
            source: source.into(),
            library: None,
            status: ClearanceStatus::Requested,
            license_number: None,
        }
    }

    /// Save to database (simplified)
    pub async fn save(&self, _db: &RightsDatabase) -> Result<()> {
        Ok(())
    }
}
