//! License management

use crate::{
    database::RightsDatabase,
    license::{agreement::AgreementStatus, LicenseAgreement, LicenseType},
    rights::RightsGrant,
    Result, RightsError,
};

/// License manager
pub struct LicenseManager<'a> {
    db: &'a RightsDatabase,
}

impl<'a> LicenseManager<'a> {
    /// Create a new license manager
    pub fn new(db: &'a RightsDatabase) -> Self {
        Self { db }
    }

    /// Validate a license for usage
    pub async fn validate_license(
        &self,
        grant_id: &str,
        usage_type: &crate::rights::UsageType,
        territory: Option<&str>,
    ) -> Result<()> {
        // Load the grant
        let grant = RightsGrant::load(self.db, grant_id)
            .await?
            .ok_or_else(|| RightsError::NotFound(format!("Grant {grant_id} not found")))?;

        // Check if grant is active
        if !grant.is_active() {
            return Err(RightsError::Expired(
                "Rights grant is not currently active".to_string(),
            ));
        }

        // Check usage permissions
        grant.allows_usage(usage_type, territory)?;

        // Check if there's an active agreement
        let agreements = LicenseAgreement::list_for_grant(self.db, grant_id).await?;
        let active_agreement = agreements
            .iter()
            .find(|a| matches!(a.status, AgreementStatus::Active));

        if active_agreement.is_none() {
            return Err(RightsError::InvalidLicense(
                "No active license agreement found".to_string(),
            ));
        }

        Ok(())
    }

    /// Check if a license type allows specific usage
    pub fn check_license_type_allows(
        &self,
        license_type: &LicenseType,
        usage_type: &crate::rights::UsageType,
    ) -> bool {
        match usage_type {
            crate::rights::UsageType::Commercial => license_type.allows_commercial_use(),
            _ => true, // Other usage types are generally allowed
        }
    }

    /// Get all active licenses for an asset
    pub async fn get_active_licenses(&self, asset_id: &str) -> Result<Vec<RightsGrant>> {
        let grants = RightsGrant::list_for_asset(self.db, asset_id).await?;
        Ok(grants
            .into_iter()
            .filter(super::super::rights::grant::RightsGrant::is_active)
            .collect())
    }

    /// Get expiring licenses (expiring within days)
    pub async fn get_expiring_licenses(
        &self,
        asset_id: &str,
        within_days: i64,
    ) -> Result<Vec<RightsGrant>> {
        let grants = RightsGrant::list_for_asset(self.db, asset_id).await?;
        let now = chrono::Utc::now();
        let threshold = now + chrono::Duration::days(within_days);

        Ok(grants
            .into_iter()
            .filter(|g| {
                if let Some(end_date) = g.end_date {
                    end_date <= threshold && end_date > now
                } else {
                    false
                }
            })
            .collect())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_license_type_allows_commercial() {
        // Test license type allows commercial use
        assert!(LicenseType::RoyaltyFree.allows_commercial_use());
        assert!(!LicenseType::CreativeCommonsByNc.allows_commercial_use());
    }
}
