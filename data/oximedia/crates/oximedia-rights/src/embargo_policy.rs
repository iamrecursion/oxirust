//! Content embargo policy management.
//!
//! This module manages broadcast/release embargo windows for content items
//! across different regions and distribution channels.

#![allow(dead_code)]
#![allow(missing_docs)]

// ── EmbargoRegion ─────────────────────────────────────────────────────────────

/// Geographical region to which an embargo applies.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum EmbargoRegion {
    /// Embargo applies globally.
    Worldwide,
    /// Canada, United States, and Mexico.
    NorthAmerica,
    /// European Economic Area and associated territories.
    Europe,
    /// Asia-Pacific region.
    AsiaPacific,
    /// Latin America and the Caribbean.
    LatinAmerica,
    /// African continent.
    Africa,
    /// User-defined region identified by a custom code/name.
    Custom(String),
}

impl EmbargoRegion {
    /// Short code identifying the region.
    ///
    /// Standard regions return a fixed code; [`EmbargoRegion::Custom`] returns
    /// the inner string.
    #[must_use]
    pub fn code(&self) -> &str {
        match self {
            Self::Worldwide => "WW",
            Self::NorthAmerica => "NA",
            Self::Europe => "EU",
            Self::AsiaPacific => "AP",
            Self::LatinAmerica => "LA",
            Self::Africa => "AF",
            Self::Custom(c) => c.as_str(),
        }
    }
}

// ── EmbargoType ───────────────────────────────────────────────────────────────

/// The distribution channel covered by an embargo.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum EmbargoType {
    /// Embargo on general / digital release.
    Release,
    /// Embargo on broadcast (linear TV / radio).
    Broadcast,
    /// Embargo on online / streaming distribution.
    Online,
    /// Embargo on theatrical (cinema) exhibition.
    Theatrical,
    /// Embargo on home-video (physical / PVOD) distribution.
    HomeVideo,
}

impl EmbargoType {
    /// Typical release-window length for this distribution channel, in days.
    ///
    /// These are indicative industry windows and may be adjusted per contract.
    #[must_use]
    pub fn window_days(&self) -> u32 {
        match self {
            Self::Release => 0,
            Self::Broadcast => 90,
            Self::Online => 45,
            Self::Theatrical => 0,
            Self::HomeVideo => 120,
        }
    }
}

// ── EmbargoRule ───────────────────────────────────────────────────────────────

/// A single embargo rule restricting content access until a lift date.
#[derive(Debug, Clone)]
pub struct EmbargoRule {
    /// Identifier of the embargoed content item.
    pub content_id: u64,
    /// Region to which this embargo applies.
    pub region: EmbargoRegion,
    /// Distribution channel covered by this embargo.
    pub embargo_type: EmbargoType,
    /// Unix timestamp (seconds) when the embargo is lifted.
    pub lift_epoch: u64,
}

impl EmbargoRule {
    /// Create a new embargo rule.
    pub fn new(
        content_id: u64,
        region: EmbargoRegion,
        embargo_type: EmbargoType,
        lift_epoch: u64,
    ) -> Self {
        Self {
            content_id,
            region,
            embargo_type,
            lift_epoch,
        }
    }

    /// Returns `true` when the embargo has been lifted at `now_epoch`.
    #[must_use]
    pub fn is_lifted(&self, now_epoch: u64) -> bool {
        now_epoch >= self.lift_epoch
    }

    /// Number of days remaining until the embargo is lifted.
    ///
    /// Negative when the lift date is in the past (i.e., already lifted).
    #[must_use]
    pub fn days_remaining(&self, now_epoch: u64) -> i64 {
        let diff = self.lift_epoch as i64 - now_epoch as i64;
        diff / 86_400
    }
}

// ── EmbargoManager ────────────────────────────────────────────────────────────

/// Central manager for content embargo rules.
#[derive(Debug, Default)]
pub struct EmbargoManager {
    /// All embargo rules tracked by this manager.
    pub rules: Vec<EmbargoRule>,
}

impl EmbargoManager {
    /// Create a new, empty embargo manager.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Register a new embargo rule.
    pub fn add(&mut self, rule: EmbargoRule) {
        self.rules.push(rule);
    }

    /// All embargo rules that are still active (not yet lifted) at `now_epoch`.
    #[must_use]
    pub fn active_embargoes(&self, now_epoch: u64) -> Vec<&EmbargoRule> {
        self.rules
            .iter()
            .filter(|r| !r.is_lifted(now_epoch))
            .collect()
    }

    /// Returns `true` if `content_id` has at least one active embargo at
    /// `now_epoch`.
    #[must_use]
    pub fn is_embargoed(&self, content_id: u64, now_epoch: u64) -> bool {
        self.rules
            .iter()
            .any(|r| r.content_id == content_id && !r.is_lifted(now_epoch))
    }

    /// Remove all embargo rules for `content_id` (i.e., lift immediately).
    pub fn lift_embargo(&mut self, content_id: u64) {
        self.rules.retain(|r| r.content_id != content_id);
    }

    /// Total number of rules registered (including already-lifted ones).
    #[must_use]
    pub fn rule_count(&self) -> usize {
        self.rules.len()
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    const NOW: u64 = 1_700_000_000;
    const PAST: u64 = 1_600_000_000;
    const FUTURE: u64 = 1_800_000_000;

    fn broadcast_rule(cid: u64, lift: u64) -> EmbargoRule {
        EmbargoRule::new(cid, EmbargoRegion::Worldwide, EmbargoType::Broadcast, lift)
    }

    // ── EmbargoRegion ──

    #[test]
    fn test_region_code_worldwide() {
        assert_eq!(EmbargoRegion::Worldwide.code(), "WW");
    }

    #[test]
    fn test_region_code_custom() {
        assert_eq!(EmbargoRegion::Custom("MENA".into()).code(), "MENA");
    }

    #[test]
    fn test_region_code_europe() {
        assert_eq!(EmbargoRegion::Europe.code(), "EU");
    }

    // ── EmbargoType ──

    #[test]
    fn test_embargo_type_window_broadcast() {
        assert_eq!(EmbargoType::Broadcast.window_days(), 90);
    }

    #[test]
    fn test_embargo_type_window_home_video() {
        assert_eq!(EmbargoType::HomeVideo.window_days(), 120);
    }

    #[test]
    fn test_embargo_type_window_theatrical() {
        assert_eq!(EmbargoType::Theatrical.window_days(), 0);
    }

    // ── EmbargoRule ──

    #[test]
    fn test_rule_is_lifted_past() {
        let r = broadcast_rule(1, PAST);
        assert!(r.is_lifted(NOW));
    }

    #[test]
    fn test_rule_is_not_lifted_future() {
        let r = broadcast_rule(1, FUTURE);
        assert!(!r.is_lifted(NOW));
    }

    #[test]
    fn test_rule_days_remaining_positive() {
        // FUTURE - NOW = 100_000_000 s ≈ 1157 days
        let r = broadcast_rule(1, FUTURE);
        let days = r.days_remaining(NOW);
        assert!(days > 0);
    }

    #[test]
    fn test_rule_days_remaining_negative() {
        let r = broadcast_rule(1, PAST);
        assert!(r.days_remaining(NOW) < 0);
    }

    // ── EmbargoManager ──

    #[test]
    fn test_manager_add_and_rule_count() {
        let mut mgr = EmbargoManager::new();
        mgr.add(broadcast_rule(10, FUTURE));
        mgr.add(broadcast_rule(11, FUTURE));
        assert_eq!(mgr.rule_count(), 2);
    }

    #[test]
    fn test_manager_active_embargoes() {
        let mut mgr = EmbargoManager::new();
        mgr.add(broadcast_rule(1, FUTURE));
        mgr.add(broadcast_rule(2, PAST));
        assert_eq!(mgr.active_embargoes(NOW).len(), 1);
    }

    #[test]
    fn test_manager_is_embargoed_true() {
        let mut mgr = EmbargoManager::new();
        mgr.add(broadcast_rule(99, FUTURE));
        assert!(mgr.is_embargoed(99, NOW));
    }

    #[test]
    fn test_manager_is_embargoed_false_lifted() {
        let mut mgr = EmbargoManager::new();
        mgr.add(broadcast_rule(99, PAST));
        assert!(!mgr.is_embargoed(99, NOW));
    }

    #[test]
    fn test_manager_lift_embargo() {
        let mut mgr = EmbargoManager::new();
        mgr.add(broadcast_rule(5, FUTURE));
        mgr.add(broadcast_rule(6, FUTURE));
        mgr.lift_embargo(5);
        assert_eq!(mgr.rule_count(), 1);
        assert!(!mgr.is_embargoed(5, NOW));
        assert!(mgr.is_embargoed(6, NOW));
    }
}
