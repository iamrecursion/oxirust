//! Talent release tracking

use crate::{clearance::ClearanceStatus, database::RightsDatabase, Result};
use serde::{Deserialize, Serialize};
use uuid::Uuid;

/// Talent/model release
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TalentRelease {
    /// Unique identifier
    pub id: String,
    /// Asset ID
    pub asset_id: String,
    /// Talent name
    pub talent_name: String,
    /// Status
    pub status: ClearanceStatus,
    /// Release form signed
    pub release_signed: bool,
}

impl TalentRelease {
    /// Create new talent release
    pub fn new(asset_id: impl Into<String>, talent_name: impl Into<String>) -> Self {
        Self {
            id: Uuid::new_v4().to_string(),
            asset_id: asset_id.into(),
            talent_name: talent_name.into(),
            status: ClearanceStatus::Requested,
            release_signed: false,
        }
    }

    /// Save to database (simplified)
    pub async fn save(&self, _db: &RightsDatabase) -> Result<()> {
        Ok(())
    }
}
