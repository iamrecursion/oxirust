//! DRM metadata management

#[cfg(not(target_arch = "wasm32"))]
use crate::database::RightsDatabase;
use crate::Result;
use chrono::Utc;
use serde::{Deserialize, Serialize};
use uuid::Uuid;

/// DRM type
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum DrmType {
    /// PlayReady
    PlayReady,
    /// Widevine
    Widevine,
    /// FairPlay
    FairPlay,
    /// Custom DRM
    Custom(String),
}

impl DrmType {
    /// Convert to string
    pub fn as_str(&self) -> &str {
        match self {
            DrmType::PlayReady => "playready",
            DrmType::Widevine => "widevine",
            DrmType::FairPlay => "fairplay",
            DrmType::Custom(s) => s,
        }
    }
}

/// DRM metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DrmMetadata {
    /// Unique identifier
    pub id: String,
    /// Asset ID
    pub asset_id: String,
    /// DRM type
    pub drm_type: DrmType,
    /// Encryption key ID
    pub encryption_key_id: Option<String>,
    /// Content ID
    pub content_id: Option<String>,
    /// License URL
    pub license_url: Option<String>,
}

impl DrmMetadata {
    /// Create new DRM metadata
    pub fn new(asset_id: impl Into<String>, drm_type: DrmType) -> Self {
        Self {
            id: Uuid::new_v4().to_string(),
            asset_id: asset_id.into(),
            drm_type,
            encryption_key_id: None,
            content_id: None,
            license_url: None,
        }
    }

    /// Set encryption key ID
    pub fn with_key_id(mut self, key_id: impl Into<String>) -> Self {
        self.encryption_key_id = Some(key_id.into());
        self
    }

    /// Set license URL
    pub fn with_license_url(mut self, url: impl Into<String>) -> Self {
        self.license_url = Some(url.into());
        self
    }

    #[cfg(not(target_arch = "wasm32"))]
    /// Save to database
    pub async fn save(&self, db: &RightsDatabase) -> Result<()> {
        let metadata_json = serde_json::json!({
            "drm_type": self.drm_type.as_str(),
        });

        let now = Utc::now();

        db.pool()
            .execute(
                r"
            INSERT INTO drm_metadata
            (id, asset_id, drm_type, encryption_key_id, content_id, license_url, metadata_json, created_at, updated_at)
            VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9)
            ON CONFLICT(id) DO UPDATE SET
                encryption_key_id = excluded.encryption_key_id,
                license_url = excluded.license_url,
                updated_at = excluded.updated_at
            ",
                &[
                    &self.id,
                    &self.asset_id,
                    &self.drm_type.as_str(),
                    &self.encryption_key_id,
                    &self.content_id,
                    &self.license_url,
                    &metadata_json.to_string(),
                    &now.to_rfc3339(),
                    &now.to_rfc3339(),
                ],
            )
            .await?;

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_drm_metadata() {
        let drm = DrmMetadata::new("asset1", DrmType::Widevine)
            .with_key_id("key123")
            .with_license_url("https://license.example.com");

        assert_eq!(drm.encryption_key_id, Some("key123".to_string()));
        assert!(drm.license_url.is_some());
    }
}
