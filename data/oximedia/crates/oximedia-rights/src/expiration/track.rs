//! Expiration tracking

use crate::{database::RightsDatabase, rights::RightsGrant, Result};
use chrono::{Duration, Utc};

/// Expiration tracker
pub struct ExpirationTracker<'a> {
    db: &'a RightsDatabase,
}

impl<'a> ExpirationTracker<'a> {
    /// Create a new expiration tracker
    pub fn new(db: &'a RightsDatabase) -> Self {
        Self { db }
    }

    /// Get all expiring grants within the specified number of days
    pub async fn get_expiring_grants(&self, days: i64) -> Result<Vec<RightsGrant>> {
        let now = Utc::now();
        let threshold = now + Duration::days(days);

        let rows = self
            .db
            .pool()
            .query(
                r"
            SELECT id, asset_id, owner_id, license_type, start_date, end_date,
                   is_exclusive, territory_json, usage_restrictions_json, created_at, updated_at
            FROM rights_grants
            WHERE end_date IS NOT NULL
              AND end_date <= $1
              AND end_date > $2
            ORDER BY end_date ASC
            ",
                &[&threshold.to_rfc3339(), &now.to_rfc3339()],
            )
            .await?;

        rows.iter().map(RightsGrant::from_row).collect()
    }

    /// Get all expired grants
    pub async fn get_expired_grants(&self) -> Result<Vec<RightsGrant>> {
        let now = Utc::now();

        let rows = self
            .db
            .pool()
            .query(
                r"
            SELECT id, asset_id, owner_id, license_type, start_date, end_date,
                   is_exclusive, territory_json, usage_restrictions_json, created_at, updated_at
            FROM rights_grants
            WHERE end_date IS NOT NULL AND end_date <= $1
            ORDER BY end_date DESC
            ",
                &[&now.to_rfc3339()],
            )
            .await?;

        rows.iter().map(RightsGrant::from_row).collect()
    }

    /// Check if a grant is about to expire (within specified days)
    pub fn is_expiring_soon(grant: &RightsGrant, days: i64) -> bool {
        if let Some(end_date) = grant.end_date {
            let now = Utc::now();
            let threshold = now + Duration::days(days);
            end_date <= threshold && end_date > now
        } else {
            false
        }
    }

    /// Get days until expiration (negative if expired)
    pub fn days_until_expiration(grant: &RightsGrant) -> Option<i64> {
        grant.end_date.map(|end_date| {
            let now = Utc::now();
            (end_date - now).num_days()
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::license::LicenseType;

    #[test]
    fn test_is_expiring_soon() {
        let now = Utc::now();

        // Grant expiring in 5 days
        let grant = RightsGrant::new(
            "asset1",
            "owner1",
            LicenseType::Exclusive,
            now - Duration::days(10),
            Some(now + Duration::days(5)),
            true,
        );

        assert!(ExpirationTracker::is_expiring_soon(&grant, 7));
        assert!(!ExpirationTracker::is_expiring_soon(&grant, 3));
    }

    #[test]
    fn test_days_until_expiration() {
        let now = Utc::now();

        let grant = RightsGrant::new(
            "asset1",
            "owner1",
            LicenseType::Exclusive,
            now - Duration::days(10),
            Some(now + Duration::days(5)),
            true,
        );

        let days = ExpirationTracker::days_until_expiration(&grant);
        assert!(days.is_some());
        assert!(
            days.expect("rights test operation should succeed") >= 4
                && days.expect("rights test operation should succeed") <= 5
        );
    }

    #[test]
    fn test_perpetual_grant() {
        let now = Utc::now();

        let grant = RightsGrant::new(
            "asset1",
            "owner1",
            LicenseType::RoyaltyFree,
            now - Duration::days(10),
            None,
            false,
        );

        assert!(!ExpirationTracker::is_expiring_soon(&grant, 30));
        assert!(ExpirationTracker::days_until_expiration(&grant).is_none());
    }
}
