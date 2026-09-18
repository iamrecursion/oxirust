//! Media licensing terms.
//!
//! Provides types for describing the scope, usage, and conditions of media
//! licenses, and for grouping multiple terms into signed license agreements.

#![allow(dead_code)]
#![allow(missing_docs)]

// ── LicenseScope ─────────────────────────────────────────────────────────────

/// The territorial / rights scope of a license.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum LicenseScope {
    /// All rights worldwide, all media, in perpetuity.
    WorldwideAllRights,
    /// Restricted to a specific territory (ISO 3166-1 alpha-2 code).
    Territory,
    /// Restricted to a specific distribution platform.
    Platform,
    /// Time-limited license (see [`LicenseTerm::duration_days`]).
    Duration,
    /// Non-exclusive grant (rights may be licensed to others simultaneously).
    NonExclusive,
}

impl LicenseScope {
    /// Returns `true` for scopes that represent an exclusive grant.
    ///
    /// Only [`LicenseScope::WorldwideAllRights`] is treated as exclusive here;
    /// all other scopes allow concurrent grants to third parties.
    #[must_use]
    pub fn is_exclusive(&self) -> bool {
        matches!(self, Self::WorldwideAllRights)
    }
}

// ── MediaUsage ────────────────────────────────────────────────────────────────

/// The intended usage category covered by a license.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum MediaUsage {
    /// Broadcast television / radio.
    Broadcast,
    /// Online / streaming distribution.
    Online,
    /// Theatrical (cinema) exhibition.
    Cinema,
    /// Educational / non-profit institutional use.
    Education,
    /// Commercial advertising or sponsored content.
    Commercial,
    /// Non-commercial / personal use.
    NonCommercial,
    /// Long-term archival storage without active distribution.
    Archive,
}

impl MediaUsage {
    /// Returns `true` when a synchronisation license is required for this
    /// usage category (i.e., music will be paired with a moving image).
    #[must_use]
    pub fn requires_sync_license(&self) -> bool {
        matches!(
            self,
            Self::Broadcast | Self::Online | Self::Cinema | Self::Commercial
        )
    }
}

// ── LicenseTerm ──────────────────────────────────────────────────────────────

/// A single set of conditions attached to a license.
#[derive(Debug, Clone)]
pub struct LicenseTerm {
    /// Territorial / rights scope.
    pub scope: LicenseScope,
    /// Intended media usage category.
    pub usage: MediaUsage,
    /// Optional ISO 3166-1 alpha-2 territory code (relevant for
    /// [`LicenseScope::Territory`]).
    pub territory: Option<String>,
    /// Optional duration in calendar days (relevant for
    /// [`LicenseScope::Duration`]).
    pub duration_days: Option<u32>,
    /// Fee for this specific term, in US cents.
    pub fee_cents: u64,
}

impl LicenseTerm {
    /// Create a new license term.
    pub fn new(
        scope: LicenseScope,
        usage: MediaUsage,
        territory: Option<String>,
        duration_days: Option<u32>,
        fee_cents: u64,
    ) -> Self {
        Self {
            scope,
            usage,
            territory,
            duration_days,
            fee_cents,
        }
    }

    /// Returns `true` when the term has no expiry (i.e., `duration_days` is
    /// `None`).
    #[must_use]
    pub fn is_perpetual(&self) -> bool {
        self.duration_days.is_none()
    }

    /// Returns `true` when the term has expired at `now_epoch` given that it
    /// was granted at `granted_epoch` (both in Unix seconds).
    ///
    /// Perpetual terms never expire.
    #[must_use]
    pub fn is_expired(&self, now_epoch: u64, granted_epoch: u64) -> bool {
        match self.duration_days {
            None => false,
            Some(days) => {
                let expiry = granted_epoch + u64::from(days) * 86_400;
                now_epoch >= expiry
            }
        }
    }
}

// ── LicenseAgreement ─────────────────────────────────────────────────────────

/// A signed license agreement containing one or more terms.
#[derive(Debug)]
pub struct LicenseAgreement {
    /// Unique numeric identifier.
    pub id: u64,
    /// Name / identifier of the licensor (rights holder granting the license).
    pub licensor: String,
    /// Name / identifier of the licensee (party receiving the license).
    pub licensee: String,
    /// Individual terms bundled in this agreement.
    pub terms: Vec<LicenseTerm>,
    /// Unix timestamp (seconds) when the agreement was signed.
    pub signed_epoch: u64,
}

impl LicenseAgreement {
    /// Create a new license agreement with no terms.
    pub fn new(
        id: u64,
        licensor: impl Into<String>,
        licensee: impl Into<String>,
        signed_epoch: u64,
    ) -> Self {
        Self {
            id,
            licensor: licensor.into(),
            licensee: licensee.into(),
            terms: Vec::new(),
            signed_epoch,
        }
    }

    /// Append a term to this agreement.
    pub fn add_term(&mut self, term: LicenseTerm) {
        self.terms.push(term);
    }

    /// Returns `true` when at least one term is still active (not expired) at
    /// `now_epoch`.
    #[must_use]
    pub fn is_active(&self, now_epoch: u64) -> bool {
        if self.terms.is_empty() {
            return false;
        }
        self.terms
            .iter()
            .any(|t| !t.is_expired(now_epoch, self.signed_epoch))
    }

    /// Number of usage categories covered by this agreement.
    #[must_use]
    pub fn usage_count(&self) -> usize {
        self.terms.len()
    }

    /// Total fee across all terms, in US cents.
    #[must_use]
    pub fn total_fee_cents(&self) -> u64 {
        self.terms.iter().map(|t| t.fee_cents).sum()
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    const EPOCH_2024: u64 = 1_704_067_200; // 2024-01-01 00:00:00 UTC
    const ONE_YEAR_SECS: u64 = 365 * 86_400;

    fn perpetual_term() -> LicenseTerm {
        LicenseTerm::new(
            LicenseScope::WorldwideAllRights,
            MediaUsage::Broadcast,
            None,
            None,
            5_000,
        )
    }

    fn one_year_term() -> LicenseTerm {
        LicenseTerm::new(
            LicenseScope::Duration,
            MediaUsage::Online,
            None,
            Some(365),
            1_000,
        )
    }

    // ── LicenseScope ──

    #[test]
    fn test_scope_worldwide_is_exclusive() {
        assert!(LicenseScope::WorldwideAllRights.is_exclusive());
    }

    #[test]
    fn test_scope_non_exclusive_not_exclusive() {
        assert!(!LicenseScope::NonExclusive.is_exclusive());
    }

    #[test]
    fn test_scope_territory_not_exclusive() {
        assert!(!LicenseScope::Territory.is_exclusive());
    }

    // ── MediaUsage ──

    #[test]
    fn test_usage_broadcast_requires_sync() {
        assert!(MediaUsage::Broadcast.requires_sync_license());
    }

    #[test]
    fn test_usage_education_no_sync() {
        assert!(!MediaUsage::Education.requires_sync_license());
    }

    #[test]
    fn test_usage_commercial_requires_sync() {
        assert!(MediaUsage::Commercial.requires_sync_license());
    }

    #[test]
    fn test_usage_archive_no_sync() {
        assert!(!MediaUsage::Archive.requires_sync_license());
    }

    // ── LicenseTerm ──

    #[test]
    fn test_term_is_perpetual_no_duration() {
        assert!(perpetual_term().is_perpetual());
    }

    #[test]
    fn test_term_is_not_perpetual_with_duration() {
        assert!(!one_year_term().is_perpetual());
    }

    #[test]
    fn test_term_not_expired_perpetual() {
        let t = perpetual_term();
        assert!(!t.is_expired(EPOCH_2024 + ONE_YEAR_SECS * 100, EPOCH_2024));
    }

    #[test]
    fn test_term_expired_after_duration() {
        let t = one_year_term();
        // now = signed + 366 days → expired
        assert!(t.is_expired(EPOCH_2024 + 366 * 86_400, EPOCH_2024));
    }

    #[test]
    fn test_term_not_expired_within_duration() {
        let t = one_year_term();
        // now = signed + 364 days → still active
        assert!(!t.is_expired(EPOCH_2024 + 364 * 86_400, EPOCH_2024));
    }

    // ── LicenseAgreement ──

    #[test]
    fn test_agreement_usage_count() {
        let mut ag = LicenseAgreement::new(1, "Studio", "Broadcaster", EPOCH_2024);
        ag.add_term(perpetual_term());
        ag.add_term(one_year_term());
        assert_eq!(ag.usage_count(), 2);
    }

    #[test]
    fn test_agreement_total_fee() {
        let mut ag = LicenseAgreement::new(2, "Studio", "Streamer", EPOCH_2024);
        ag.add_term(perpetual_term()); // 5_000 cents
        ag.add_term(one_year_term()); // 1_000 cents
        assert_eq!(ag.total_fee_cents(), 6_000);
    }

    #[test]
    fn test_agreement_is_active_with_perpetual_term() {
        let mut ag = LicenseAgreement::new(3, "Studio", "Cinema", EPOCH_2024);
        ag.add_term(perpetual_term());
        assert!(ag.is_active(EPOCH_2024 + ONE_YEAR_SECS * 50));
    }

    #[test]
    fn test_agreement_is_active_all_expired() {
        let mut ag = LicenseAgreement::new(4, "Studio", "Archive", EPOCH_2024);
        ag.add_term(one_year_term()); // expires after 365 days
        assert!(!ag.is_active(EPOCH_2024 + 366 * 86_400));
    }

    #[test]
    fn test_agreement_is_active_empty() {
        let ag = LicenseAgreement::new(5, "Studio", "Partner", EPOCH_2024);
        assert!(!ag.is_active(EPOCH_2024));
    }
}
