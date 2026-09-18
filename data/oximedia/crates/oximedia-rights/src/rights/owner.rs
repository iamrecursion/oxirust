//! Rights owner management

#[cfg(not(target_arch = "wasm32"))]
use crate::database::RightsDatabase;
use crate::Result;
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use uuid::Uuid;

/// Rights owner (person or organization)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RightsOwner {
    /// Unique identifier
    pub id: String,
    /// Owner name
    pub name: String,
    /// Contact information
    pub contact_info: Option<String>,
    /// Creation timestamp
    pub created_at: DateTime<Utc>,
    /// Last update timestamp
    pub updated_at: DateTime<Utc>,
}

impl RightsOwner {
    /// Create a new rights owner
    pub fn new(name: impl Into<String>) -> Self {
        let now = Utc::now();
        Self {
            id: Uuid::new_v4().to_string(),
            name: name.into(),
            contact_info: None,
            created_at: now,
            updated_at: now,
        }
    }

    /// Set contact information
    pub fn with_contact(mut self, contact: impl Into<String>) -> Self {
        self.contact_info = Some(contact.into());
        self
    }

    #[cfg(not(target_arch = "wasm32"))]
    /// Decode a `rights_owners` row into a [`RightsOwner`].
    fn from_row(r: &oxisql_core::Row) -> Result<Self> {
        let created_at_s: String = r.try_get("created_at")?;
        let created_at = DateTime::parse_from_rfc3339(&created_at_s)
            .map_err(|e| crate::RightsError::InvalidLicense(format!("Invalid created_at: {e}")))?
            .with_timezone(&Utc);
        let updated_at_s: String = r.try_get("updated_at")?;
        let updated_at = DateTime::parse_from_rfc3339(&updated_at_s)
            .map_err(|e| crate::RightsError::InvalidLicense(format!("Invalid updated_at: {e}")))?
            .with_timezone(&Utc);
        Ok(RightsOwner {
            id: r.try_get("id")?,
            name: r.try_get("name")?,
            contact_info: r.try_get("contact_info")?,
            created_at,
            updated_at,
        })
    }

    #[cfg(not(target_arch = "wasm32"))]
    /// Save owner to database
    pub async fn save(&self, db: &RightsDatabase) -> Result<()> {
        db.pool()
            .execute(
                r"
            INSERT INTO rights_owners (id, name, contact_info, created_at, updated_at)
            VALUES ($1, $2, $3, $4, $5)
            ON CONFLICT(id) DO UPDATE SET
                name = excluded.name,
                contact_info = excluded.contact_info,
                updated_at = excluded.updated_at
            ",
                &[
                    &self.id,
                    &self.name,
                    &self.contact_info,
                    &self.created_at.to_rfc3339(),
                    &self.updated_at.to_rfc3339(),
                ],
            )
            .await?;

        Ok(())
    }

    #[cfg(not(target_arch = "wasm32"))]
    /// Load owner from database by ID
    pub async fn load(db: &RightsDatabase, id: &str) -> Result<Option<Self>> {
        let row = db
            .pool()
            .query_optional(
                r"
            SELECT id, name, contact_info, created_at, updated_at
            FROM rights_owners WHERE id = $1
            ",
                &[&id],
            )
            .await?;

        row.map(|r| Self::from_row(&r)).transpose()
    }

    #[cfg(not(target_arch = "wasm32"))]
    /// List all owners
    pub async fn list(db: &RightsDatabase) -> Result<Vec<Self>> {
        let rows = db
            .pool()
            .query(
                r"
            SELECT id, name, contact_info, created_at, updated_at
            FROM rights_owners
            ORDER BY name ASC
            ",
                &[],
            )
            .await?;

        rows.iter().map(Self::from_row).collect()
    }

    #[cfg(not(target_arch = "wasm32"))]
    /// Delete owner from database
    pub async fn delete(db: &RightsDatabase, id: &str) -> Result<()> {
        db.pool()
            .execute("DELETE FROM rights_owners WHERE id = $1", &[&id])
            .await?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_owner_creation() {
        let owner = RightsOwner::new("John Doe").with_contact("john@example.com");

        assert_eq!(owner.name, "John Doe");
        assert_eq!(owner.contact_info, Some("john@example.com".to_string()));
    }

    #[tokio::test]
    async fn test_owner_save_and_load() {
        let temp_dir = tempfile::tempdir().expect("rights test operation should succeed");
        let db_path = format!("sqlite://{}/test.db", temp_dir.path().display());
        let db = RightsDatabase::new(&db_path)
            .await
            .expect("rights test operation should succeed");

        let owner = RightsOwner::new("Test Owner");
        let owner_id = owner.id.clone();

        owner
            .save(&db)
            .await
            .expect("rights test operation should succeed");

        let loaded = RightsOwner::load(&db, &owner_id)
            .await
            .expect("rights test operation should succeed");
        assert!(loaded.is_some());
        let loaded = loaded.expect("rights test operation should succeed");
        assert_eq!(loaded.name, "Test Owner");
    }
}
