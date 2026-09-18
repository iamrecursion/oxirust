//! Asset rights management

#[cfg(not(target_arch = "wasm32"))]
use crate::database::RightsDatabase;
use crate::Result;
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use uuid::Uuid;

/// Type of media asset
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum AssetType {
    /// Video content
    Video,
    /// Audio content
    Audio,
    /// Image content
    Image,
    /// Document content
    Document,
    /// Music track
    Music,
    /// Stock footage
    StockFootage,
    /// Other type
    Other(String),
}

impl AssetType {
    /// Convert to string representation
    pub fn as_str(&self) -> &str {
        match self {
            AssetType::Video => "video",
            AssetType::Audio => "audio",
            AssetType::Image => "image",
            AssetType::Document => "document",
            AssetType::Music => "music",
            AssetType::StockFootage => "stock_footage",
            AssetType::Other(s) => s,
        }
    }

    /// Parse from string
    #[allow(clippy::should_implement_trait)]
    pub fn from_str(s: &str) -> Self {
        match s {
            "video" => AssetType::Video,
            "audio" => AssetType::Audio,
            "image" => AssetType::Image,
            "document" => AssetType::Document,
            "music" => AssetType::Music,
            "stock_footage" => AssetType::StockFootage,
            other => AssetType::Other(other.to_string()),
        }
    }
}

/// Media asset with rights information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Asset {
    /// Unique identifier
    pub id: String,
    /// Asset name
    pub name: String,
    /// Asset type
    pub asset_type: AssetType,
    /// Description
    pub description: Option<String>,
    /// Creation timestamp
    pub created_at: DateTime<Utc>,
    /// Last update timestamp
    pub updated_at: DateTime<Utc>,
}

impl Asset {
    /// Create a new asset
    pub fn new(name: impl Into<String>, asset_type: AssetType) -> Self {
        let now = Utc::now();
        Self {
            id: Uuid::new_v4().to_string(),
            name: name.into(),
            asset_type,
            description: None,
            created_at: now,
            updated_at: now,
        }
    }

    /// Set description
    pub fn with_description(mut self, description: impl Into<String>) -> Self {
        self.description = Some(description.into());
        self
    }

    #[cfg(not(target_arch = "wasm32"))]
    /// Decode an `assets` row into an [`Asset`].
    fn from_row(r: &oxisql_core::Row) -> Result<Self> {
        let created_at_s: String = r.try_get("created_at")?;
        let created_at = DateTime::parse_from_rfc3339(&created_at_s)
            .map_err(|e| crate::RightsError::InvalidLicense(format!("Invalid created_at: {e}")))?
            .with_timezone(&Utc);
        let updated_at_s: String = r.try_get("updated_at")?;
        let updated_at = DateTime::parse_from_rfc3339(&updated_at_s)
            .map_err(|e| crate::RightsError::InvalidLicense(format!("Invalid updated_at: {e}")))?
            .with_timezone(&Utc);
        let asset_type_s: String = r.try_get("asset_type")?;
        Ok(Asset {
            id: r.try_get("id")?,
            name: r.try_get("name")?,
            asset_type: AssetType::from_str(&asset_type_s),
            description: r.try_get("description")?,
            created_at,
            updated_at,
        })
    }

    #[cfg(not(target_arch = "wasm32"))]
    /// Save asset to database
    pub async fn save(&self, db: &RightsDatabase) -> Result<()> {
        db.pool()
            .execute(
                r"
            INSERT INTO assets (id, name, asset_type, description, created_at, updated_at)
            VALUES ($1, $2, $3, $4, $5, $6)
            ON CONFLICT(id) DO UPDATE SET
                name = excluded.name,
                asset_type = excluded.asset_type,
                description = excluded.description,
                updated_at = excluded.updated_at
            ",
                &[
                    &self.id,
                    &self.name,
                    &self.asset_type.as_str(),
                    &self.description,
                    &self.created_at.to_rfc3339(),
                    &self.updated_at.to_rfc3339(),
                ],
            )
            .await?;

        Ok(())
    }

    #[cfg(not(target_arch = "wasm32"))]
    /// Load asset from database by ID
    pub async fn load(db: &RightsDatabase, id: &str) -> Result<Option<Self>> {
        let row = db
            .pool()
            .query_optional(
                r"
            SELECT id, name, asset_type, description, created_at, updated_at
            FROM assets WHERE id = $1
            ",
                &[&id],
            )
            .await?;

        row.map(|r| Self::from_row(&r)).transpose()
    }

    #[cfg(not(target_arch = "wasm32"))]
    /// List all assets
    pub async fn list(db: &RightsDatabase) -> Result<Vec<Self>> {
        let rows = db
            .pool()
            .query(
                r"
            SELECT id, name, asset_type, description, created_at, updated_at
            FROM assets
            ORDER BY created_at DESC
            ",
                &[],
            )
            .await?;

        rows.iter().map(Self::from_row).collect()
    }

    #[cfg(not(target_arch = "wasm32"))]
    /// Delete asset from database
    pub async fn delete(db: &RightsDatabase, id: &str) -> Result<()> {
        db.pool()
            .execute("DELETE FROM assets WHERE id = $1", &[&id])
            .await?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_asset_creation() {
        let asset =
            Asset::new("Test Video", AssetType::Video).with_description("A test video asset");

        assert_eq!(asset.name, "Test Video");
        assert_eq!(asset.asset_type, AssetType::Video);
        assert_eq!(asset.description, Some("A test video asset".to_string()));
    }

    #[tokio::test]
    async fn test_asset_save_and_load() {
        let temp_dir = tempfile::tempdir().expect("rights test operation should succeed");
        let db_path = format!("sqlite://{}/test.db", temp_dir.path().display());
        let db = RightsDatabase::new(&db_path)
            .await
            .expect("rights test operation should succeed");

        let asset = Asset::new("Test Asset", AssetType::Image);
        let asset_id = asset.id.clone();

        asset
            .save(&db)
            .await
            .expect("rights test operation should succeed");

        let loaded = Asset::load(&db, &asset_id)
            .await
            .expect("rights test operation should succeed");
        assert!(loaded.is_some());
        let loaded = loaded.expect("rights test operation should succeed");
        assert_eq!(loaded.name, "Test Asset");
        assert_eq!(loaded.asset_type, AssetType::Image);
    }

    #[test]
    fn test_asset_type_conversion() {
        assert_eq!(AssetType::Video.as_str(), "video");
        assert_eq!(AssetType::from_str("video"), AssetType::Video);
        assert_eq!(
            AssetType::from_str("custom"),
            AssetType::Other("custom".to_string())
        );
    }
}
