//! License agreement tracking

use crate::{database::RightsDatabase, license::LicenseTerms, Result};
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use uuid::Uuid;

/// License agreement status
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum AgreementStatus {
    /// Draft agreement
    Draft,
    /// Pending signature
    Pending,
    /// Active agreement
    Active,
    /// Expired agreement
    Expired,
    /// Terminated agreement
    Terminated,
    /// Cancelled agreement
    Cancelled,
}

impl AgreementStatus {
    /// Convert to string representation
    pub fn as_str(&self) -> &str {
        match self {
            AgreementStatus::Draft => "draft",
            AgreementStatus::Pending => "pending",
            AgreementStatus::Active => "active",
            AgreementStatus::Expired => "expired",
            AgreementStatus::Terminated => "terminated",
            AgreementStatus::Cancelled => "cancelled",
        }
    }

    /// Parse from string
    #[allow(clippy::should_implement_trait)]
    pub fn from_str(s: &str) -> Self {
        match s {
            "draft" => AgreementStatus::Draft,
            "pending" => AgreementStatus::Pending,
            "active" => AgreementStatus::Active,
            "expired" => AgreementStatus::Expired,
            "terminated" => AgreementStatus::Terminated,
            "cancelled" => AgreementStatus::Cancelled,
            _ => AgreementStatus::Draft,
        }
    }
}

/// License agreement
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LicenseAgreement {
    /// Unique identifier
    pub id: String,
    /// Associated rights grant ID
    pub grant_id: String,
    /// Agreement number (human-readable)
    pub agreement_number: String,
    /// License terms
    pub terms: LicenseTerms,
    /// Agreement status
    pub status: AgreementStatus,
    /// Signed date
    pub signed_date: Option<DateTime<Utc>>,
    /// Creation timestamp
    pub created_at: DateTime<Utc>,
    /// Last update timestamp
    pub updated_at: DateTime<Utc>,
}

impl LicenseAgreement {
    /// Create a new license agreement
    pub fn new(grant_id: impl Into<String>, agreement_number: impl Into<String>) -> Self {
        let now = Utc::now();
        Self {
            id: Uuid::new_v4().to_string(),
            grant_id: grant_id.into(),
            agreement_number: agreement_number.into(),
            terms: LicenseTerms::default(),
            status: AgreementStatus::Draft,
            signed_date: None,
            created_at: now,
            updated_at: now,
        }
    }

    /// Set license terms
    pub fn with_terms(mut self, terms: LicenseTerms) -> Self {
        self.terms = terms;
        self
    }

    /// Mark as pending signature
    pub fn mark_pending(&mut self) {
        self.status = AgreementStatus::Pending;
        self.updated_at = Utc::now();
    }

    /// Sign the agreement
    pub fn sign(&mut self) {
        self.status = AgreementStatus::Active;
        self.signed_date = Some(Utc::now());
        self.updated_at = Utc::now();
    }

    /// Terminate the agreement
    pub fn terminate(&mut self) {
        self.status = AgreementStatus::Terminated;
        self.updated_at = Utc::now();
    }

    /// Cancel the agreement
    pub fn cancel(&mut self) {
        self.status = AgreementStatus::Cancelled;
        self.updated_at = Utc::now();
    }

    #[cfg(not(target_arch = "wasm32"))]
    /// Decode a `license_agreements` row into a [`LicenseAgreement`].
    fn from_row(r: &oxisql_core::Row) -> Result<Self> {
        let terms_json: String = r.try_get("terms_json")?;
        let terms = serde_json::from_str(&terms_json).unwrap_or_default();

        let signed_date = r
            .try_get::<Option<String>>("signed_date")?
            .map(|s| {
                DateTime::parse_from_rfc3339(&s)
                    .map(|dt| dt.with_timezone(&Utc))
                    .map_err(|e| crate::RightsError::Serialization(e.to_string()))
            })
            .transpose()?;

        let created_at_s: String = r.try_get("created_at")?;
        let created_at = DateTime::parse_from_rfc3339(&created_at_s)
            .map(|dt| dt.with_timezone(&Utc))
            .map_err(|e| crate::RightsError::Serialization(e.to_string()))?;

        let updated_at_s: String = r.try_get("updated_at")?;
        let updated_at = DateTime::parse_from_rfc3339(&updated_at_s)
            .map(|dt| dt.with_timezone(&Utc))
            .map_err(|e| crate::RightsError::Serialization(e.to_string()))?;

        let status_s: String = r.try_get("status")?;

        Ok(LicenseAgreement {
            id: r.try_get("id")?,
            grant_id: r.try_get("grant_id")?,
            agreement_number: r.try_get("agreement_number")?,
            terms,
            status: AgreementStatus::from_str(&status_s),
            signed_date,
            created_at,
            updated_at,
        })
    }

    #[cfg(not(target_arch = "wasm32"))]
    /// Save agreement to database
    pub async fn save(&self, db: &RightsDatabase) -> Result<()> {
        let terms_json = serde_json::to_string(&self.terms)
            .map_err(|e| crate::RightsError::Serialization(e.to_string()))?;

        db.pool()
            .execute(
                r"
            INSERT INTO license_agreements
            (id, grant_id, agreement_number, terms_json, status, signed_date, created_at, updated_at)
            VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
            ON CONFLICT(id) DO UPDATE SET
                grant_id = excluded.grant_id,
                agreement_number = excluded.agreement_number,
                terms_json = excluded.terms_json,
                status = excluded.status,
                signed_date = excluded.signed_date,
                updated_at = excluded.updated_at
            ",
                &[
                    &self.id,
                    &self.grant_id,
                    &self.agreement_number,
                    &terms_json,
                    &self.status.as_str(),
                    &self.signed_date.map(|d| d.to_rfc3339()),
                    &self.created_at.to_rfc3339(),
                    &self.updated_at.to_rfc3339(),
                ],
            )
            .await?;

        Ok(())
    }

    #[cfg(not(target_arch = "wasm32"))]
    /// Load agreement from database by ID
    pub async fn load(db: &RightsDatabase, id: &str) -> Result<Option<Self>> {
        let row = db
            .pool()
            .query_optional(
                r"
            SELECT id, grant_id, agreement_number, terms_json, status, signed_date, created_at, updated_at
            FROM license_agreements WHERE id = $1
            ",
                &[&id],
            )
            .await?;

        row.map(|r| Self::from_row(&r)).transpose()
    }

    #[cfg(not(target_arch = "wasm32"))]
    /// List agreements for a grant
    pub async fn list_for_grant(db: &RightsDatabase, grant_id: &str) -> Result<Vec<Self>> {
        let rows = db
            .pool()
            .query(
                r"
            SELECT id, grant_id, agreement_number, terms_json, status, signed_date, created_at, updated_at
            FROM license_agreements WHERE grant_id = $1
            ORDER BY created_at DESC
            ",
                &[&grant_id],
            )
            .await?;

        rows.iter().map(Self::from_row).collect()
    }

    #[cfg(not(target_arch = "wasm32"))]
    /// Delete agreement from database
    pub async fn delete(db: &RightsDatabase, id: &str) -> Result<()> {
        db.pool()
            .execute("DELETE FROM license_agreements WHERE id = $1", &[&id])
            .await?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_agreement_creation() {
        let agreement = LicenseAgreement::new("grant123", "AGR-2024-001");
        assert_eq!(agreement.grant_id, "grant123");
        assert_eq!(agreement.agreement_number, "AGR-2024-001");
        assert_eq!(agreement.status, AgreementStatus::Draft);
    }

    #[test]
    fn test_agreement_workflow() {
        let mut agreement = LicenseAgreement::new("grant123", "AGR-2024-001");

        agreement.mark_pending();
        assert_eq!(agreement.status, AgreementStatus::Pending);

        agreement.sign();
        assert_eq!(agreement.status, AgreementStatus::Active);
        assert!(agreement.signed_date.is_some());

        agreement.terminate();
        assert_eq!(agreement.status, AgreementStatus::Terminated);
    }

    #[tokio::test]
    async fn test_agreement_save_and_load() {
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
            chrono::Utc::now(),
            Some(chrono::Utc::now() + chrono::Duration::days(30)),
            true,
        );
        let grant_id = grant.id.clone();
        grant
            .save(&db)
            .await
            .expect("rights test operation should succeed");

        let agreement = LicenseAgreement::new(&grant_id, "AGR-2024-001");
        let agreement_id = agreement.id.clone();

        agreement
            .save(&db)
            .await
            .expect("rights test operation should succeed");

        let loaded = LicenseAgreement::load(&db, &agreement_id)
            .await
            .expect("rights test operation should succeed");
        assert!(loaded.is_some());
        let loaded = loaded.expect("rights test operation should succeed");
        assert_eq!(loaded.agreement_number, "AGR-2024-001");
    }
}
