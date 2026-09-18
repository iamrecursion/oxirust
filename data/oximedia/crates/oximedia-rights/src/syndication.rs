//! Content syndication rights management.
//!
//! Tracks syndication agreements, windows of availability, and revenue
//! sharing between content owners and distribution partners.

use std::collections::HashMap;

/// A time window during which a syndication agreement is valid.
#[derive(Debug, Clone)]
pub struct SyndicationWindow {
    /// Unix timestamp (ms) when the window opens.
    pub start_ms: u64,
    /// Unix timestamp (ms) when the window closes.
    pub end_ms: u64,
    /// Maximum number of plays allowed during the window; `None` = unlimited.
    pub max_plays: Option<u32>,
    /// Maximum number of concurrent viewers; `None` = unlimited.
    pub max_viewers: Option<u64>,
}

impl SyndicationWindow {
    /// Create a new syndication window.
    pub fn new(
        start_ms: u64,
        end_ms: u64,
        max_plays: Option<u32>,
        max_viewers: Option<u64>,
    ) -> Self {
        Self {
            start_ms,
            end_ms,
            max_plays,
            max_viewers,
        }
    }

    /// Returns `true` if the window is currently open at `current_ms`.
    #[must_use]
    pub fn is_active(&self, current_ms: u64) -> bool {
        current_ms >= self.start_ms && current_ms < self.end_ms
    }
}

/// Tier of a syndication partner, governing limits and priority.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PartnerTier {
    /// Top-tier partner with the highest limits and revenue share.
    Premium,
    /// Standard partner.
    Standard,
    /// Reduced-feature partner.
    Basic,
    /// Evaluation / trial partner with strict limits.
    Trial,
}

impl PartnerTier {
    /// Maximum number of concurrent syndication windows allowed for this tier.
    #[must_use]
    pub fn max_windows(&self) -> u32 {
        match self {
            Self::Premium => 50,
            Self::Standard => 20,
            Self::Basic => 5,
            Self::Trial => 1,
        }
    }
}

/// A syndication partner (distributor or platform).
#[derive(Debug, Clone)]
pub struct SyndicationPartner {
    /// Unique partner identifier.
    pub id: String,
    /// Human-readable partner name.
    pub name: String,
    /// Platform name (e.g. "Amazon Prime", "Hulu").
    pub platform: String,
    /// Partner tier.
    pub tier: PartnerTier,
}

impl SyndicationPartner {
    /// Create a new syndication partner.
    pub fn new(
        id: impl Into<String>,
        name: impl Into<String>,
        platform: impl Into<String>,
        tier: PartnerTier,
    ) -> Self {
        Self {
            id: id.into(),
            name: name.into(),
            platform: platform.into(),
            tier,
        }
    }
}

/// A syndication agreement between a content owner and a partner.
#[derive(Debug, Clone)]
pub struct SyndicationAgreement {
    /// Identifier of the partner.
    pub partner_id: String,
    /// Asset IDs covered by this agreement.
    pub asset_ids: Vec<String>,
    /// Syndication window.
    pub window: SyndicationWindow,
    /// Revenue share percentage paid to the content owner (0–100).
    pub revenue_share_pct: f32,
}

impl SyndicationAgreement {
    /// Create a new syndication agreement.
    pub fn new(
        partner_id: impl Into<String>,
        asset_ids: Vec<String>,
        window: SyndicationWindow,
        revenue_share_pct: f32,
    ) -> Self {
        Self {
            partner_id: partner_id.into(),
            asset_ids,
            window,
            revenue_share_pct,
        }
    }
}

/// Revenue report for a syndication partner.
#[derive(Debug, Clone)]
pub struct RevenueReport {
    /// Partner identifier.
    pub partner_id: String,
    /// Total number of plays counted.
    pub total_plays: u64,
    /// Total revenue attributed to this partner in USD.
    pub revenue_usd: f64,
}

impl RevenueReport {
    /// Create a new revenue report.
    pub fn new(partner_id: impl Into<String>, total_plays: u64, revenue_usd: f64) -> Self {
        Self {
            partner_id: partner_id.into(),
            total_plays,
            revenue_usd,
        }
    }

    /// Split `gross_revenue` according to `pct` (owner's share percentage).
    ///
    /// Returns `(owner_share, partner_share)`.
    #[must_use]
    pub fn split_revenue(gross_revenue: f64, pct: f32) -> (f64, f64) {
        let owner = gross_revenue * (pct as f64 / 100.0);
        let partner = gross_revenue - owner;
        (owner, partner)
    }
}

/// Central manager for syndication agreements.
#[derive(Debug, Default)]
pub struct SyndicationManager {
    agreements: Vec<SyndicationAgreement>,
    /// Play counts keyed by `(partner_id, asset_id)`.
    play_counts: HashMap<(String, String), u64>,
}

impl SyndicationManager {
    /// Create a new empty syndication manager.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Register a new syndication agreement.
    pub fn add_agreement(&mut self, agreement: SyndicationAgreement) {
        self.agreements.push(agreement);
    }

    /// Returns `true` if `partner_id` may play `asset_id` at `current_ms`.
    ///
    /// Checks that an active agreement covers the asset and that play-count
    /// limits have not been exceeded.
    #[must_use]
    pub fn can_play(&self, asset_id: &str, partner_id: &str, current_ms: u64) -> bool {
        self.agreements.iter().any(|a| {
            a.partner_id == partner_id
                && a.asset_ids.iter().any(|id| id == asset_id)
                && a.window.is_active(current_ms)
                && a.window.max_plays.is_none_or(|limit| {
                    let plays = self
                        .play_counts
                        .get(&(partner_id.to_string(), asset_id.to_string()))
                        .copied()
                        .unwrap_or(0);
                    plays < limit as u64
                })
        })
    }

    /// Record a play for accounting purposes.
    pub fn record_play(&mut self, partner_id: impl Into<String>, asset_id: impl Into<String>) {
        *self
            .play_counts
            .entry((partner_id.into(), asset_id.into()))
            .or_insert(0) += 1;
    }

    /// Return all currently active agreements at `current_ms`.
    #[must_use]
    pub fn active_agreements(&self, current_ms: u64) -> Vec<&SyndicationAgreement> {
        self.agreements
            .iter()
            .filter(|a| a.window.is_active(current_ms))
            .collect()
    }

    /// Build a revenue report for `partner_id` given `revenue_per_play`.
    #[must_use]
    pub fn revenue_report(&self, partner_id: &str, revenue_per_play: f64) -> RevenueReport {
        let total_plays: u64 = self
            .play_counts
            .iter()
            .filter(|((pid, _), _)| pid == partner_id)
            .map(|(_, &count)| count)
            .sum();

        let revenue_usd = total_plays as f64 * revenue_per_play;
        RevenueReport::new(partner_id, total_plays, revenue_usd)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const NOW: u64 = 1_700_000_000_000_u64;
    const PAST: u64 = 1_600_000_000_000_u64;
    const FUTURE: u64 = 1_800_000_000_000_u64;

    fn active_window() -> SyndicationWindow {
        SyndicationWindow::new(PAST, FUTURE, None, None)
    }

    fn expired_window() -> SyndicationWindow {
        SyndicationWindow::new(PAST, PAST + 1, None, None)
    }

    fn agreement(partner: &str, assets: &[&str]) -> SyndicationAgreement {
        SyndicationAgreement::new(
            partner,
            assets
                .iter()
                .map(std::string::ToString::to_string)
                .collect(),
            active_window(),
            70.0,
        )
    }

    // --- SyndicationWindow ---

    #[test]
    fn test_window_is_active() {
        let w = active_window();
        assert!(w.is_active(NOW));
    }

    #[test]
    fn test_window_not_active_before_start() {
        let w = SyndicationWindow::new(FUTURE, FUTURE + 1000, None, None);
        assert!(!w.is_active(NOW));
    }

    #[test]
    fn test_window_not_active_after_end() {
        let w = expired_window();
        assert!(!w.is_active(NOW));
    }

    // --- PartnerTier ---

    #[test]
    fn test_partner_tier_max_windows() {
        assert_eq!(PartnerTier::Premium.max_windows(), 50);
        assert_eq!(PartnerTier::Standard.max_windows(), 20);
        assert_eq!(PartnerTier::Basic.max_windows(), 5);
        assert_eq!(PartnerTier::Trial.max_windows(), 1);
    }

    // --- SyndicationPartner ---

    #[test]
    fn test_partner_new() {
        let p = SyndicationPartner::new("p-1", "Netflix", "Netflix", PartnerTier::Premium);
        assert_eq!(p.id, "p-1");
        assert_eq!(p.tier, PartnerTier::Premium);
    }

    // --- RevenueReport ---

    #[test]
    fn test_split_revenue_70_30() {
        let (owner, partner) = RevenueReport::split_revenue(100.0, 70.0);
        assert!((owner - 70.0).abs() < 1e-6);
        assert!((partner - 30.0).abs() < 1e-6);
    }

    #[test]
    fn test_split_revenue_zero_pct() {
        let (owner, partner) = RevenueReport::split_revenue(100.0, 0.0);
        assert!((owner - 0.0).abs() < 1e-6);
        assert!((partner - 100.0).abs() < 1e-6);
    }

    #[test]
    fn test_split_revenue_100_pct() {
        let (owner, partner) = RevenueReport::split_revenue(200.0, 100.0);
        assert!((owner - 200.0).abs() < 1e-6);
        assert!((partner - 0.0).abs() < 1e-6);
    }

    // --- SyndicationManager ---

    #[test]
    fn test_can_play_active_agreement() {
        let mut mgr = SyndicationManager::new();
        mgr.add_agreement(agreement("partner-1", &["asset-A"]));
        assert!(mgr.can_play("asset-A", "partner-1", NOW));
    }

    #[test]
    fn test_can_play_wrong_asset() {
        let mut mgr = SyndicationManager::new();
        mgr.add_agreement(agreement("partner-1", &["asset-A"]));
        assert!(!mgr.can_play("asset-B", "partner-1", NOW));
    }

    #[test]
    fn test_can_play_wrong_partner() {
        let mut mgr = SyndicationManager::new();
        mgr.add_agreement(agreement("partner-1", &["asset-A"]));
        assert!(!mgr.can_play("asset-A", "partner-2", NOW));
    }

    #[test]
    fn test_can_play_expired_window() {
        let mut mgr = SyndicationManager::new();
        let a =
            SyndicationAgreement::new("partner-1", vec!["asset-A".into()], expired_window(), 70.0);
        mgr.add_agreement(a);
        assert!(!mgr.can_play("asset-A", "partner-1", NOW));
    }

    #[test]
    fn test_can_play_play_limit_exceeded() {
        let mut mgr = SyndicationManager::new();
        let a = SyndicationAgreement::new(
            "partner-1",
            vec!["asset-A".into()],
            SyndicationWindow::new(PAST, FUTURE, Some(2), None),
            70.0,
        );
        mgr.add_agreement(a);
        mgr.record_play("partner-1", "asset-A");
        mgr.record_play("partner-1", "asset-A");
        // 2 plays used, limit is 2 → blocked
        assert!(!mgr.can_play("asset-A", "partner-1", NOW));
    }

    #[test]
    fn test_active_agreements() {
        let mut mgr = SyndicationManager::new();
        mgr.add_agreement(agreement("partner-1", &["asset-A"]));
        let a2 =
            SyndicationAgreement::new("partner-2", vec!["asset-B".into()], expired_window(), 50.0);
        mgr.add_agreement(a2);
        let active = mgr.active_agreements(NOW);
        assert_eq!(active.len(), 1);
        assert_eq!(active[0].partner_id, "partner-1");
    }

    #[test]
    fn test_revenue_report() {
        let mut mgr = SyndicationManager::new();
        mgr.add_agreement(agreement("partner-1", &["asset-A"]));
        mgr.record_play("partner-1", "asset-A");
        mgr.record_play("partner-1", "asset-A");
        mgr.record_play("partner-1", "asset-A");

        let report = mgr.revenue_report("partner-1", 1.5);
        assert_eq!(report.total_plays, 3);
        assert!((report.revenue_usd - 4.5).abs() < 1e-6);
    }
}
