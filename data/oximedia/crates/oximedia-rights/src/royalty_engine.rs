//! Royalty calculation engine with agreement tracking, usage event recording,
//! and statement generation.
//!
//! This module provides a complete royalty management system capable of
//! handling multiple royalty models (per-play, revenue-share, sync fees, etc.)
//! with territory filtering, minimum/maximum payment caps, and per-rights-holder
//! statement generation.
//!
//! ## Calculation Formulas
//!
//! ### Tiered rate formula
//!
//! When using `oximedia_rights::tiered_royalty::TieredRoyaltySchedule`, royalties
//! are calculated by summing the per-tier contributions:
//!
//! ```text
//! royalty = Σ(tier_i: views_in_tier_i × rate_i)
//! ```
//!
//! where each `tier_i` covers a contiguous usage-count range
//! `[threshold_min_i, threshold_max_i)` with a fixed `rate_per_unit`.
//!
//! ### Territory split formula
//!
//! When distributing a total royalty across territories:
//!
//! ```text
//! territory_royalty = total_royalty × territory_share
//! ```
//!
//! `territory_share` is the fraction (0.0–1.0) of the total audience or revenue
//! attributable to that territory.  All shares must sum to 1.0.
//! Per-territory multipliers (from `oximedia_rights::royalty::territory::Territory`)
//! further scale the payout based on market conditions.
//!
//! ### Usage basis options
//!
//! The [`RoyaltyBasis`] enum provides three primary usage-count modes and two
//! flat-rate modes:
//!
//! | Variant | Basis | Formula |
//! |---------|-------|---------|
//! | `PerPlay` | per-stream count | `quantity × 1.0` |
//! | `PerDownload` | per-download count | `download_quantity × 1.0` |
//! | `Custom(rate)` | per-event rate | `quantity × rate` |
//! | `RevenueShare(pct)` | percentage of revenue | `total_revenue × pct` |
//! | `SyncFee(fee)` | flat one-time fee | `fee` (if events > 0, else 0) |
//! | `MechanicalRate` | per physical unit sold | `sale_quantity × 1.0` |

#![allow(missing_docs)]
#![allow(clippy::cast_precision_loss)]

use serde::{Deserialize, Serialize};

// ── RoyaltyBasis ─────────────────────────────────────────────────────────────

/// The basis on which a royalty is calculated.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum RoyaltyBasis {
    /// Fixed amount per play/stream event.
    PerPlay,
    /// Fixed amount per download event.
    PerDownload,
    /// Percentage of revenue (0.0 – 1.0 exclusive).
    RevenueShare(f64),
    /// Fixed mechanical rate per physical unit sold.
    MechanicalRate,
    /// One-time synchronisation licensing fee regardless of event count.
    SyncFee(f64),
    /// Custom per-event rate in the agreement's currency.
    Custom(f64),
}

// ── RoyaltyAgreement ─────────────────────────────────────────────────────────

/// A bilateral royalty agreement between an asset owner and a rights holder.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RoyaltyAgreement {
    /// Unique agreement identifier.
    pub id: String,
    /// The asset this agreement covers.
    pub asset_id: String,
    /// The rights holder entitled to royalties under this agreement.
    pub rights_holder: String,
    /// ISO 3166-1 alpha-2 territory codes covered by this agreement.
    /// An empty vector means the agreement is worldwide.
    pub territory: Vec<String>,
    /// How royalties are calculated.
    pub basis: RoyaltyBasis,
    /// Minimum payment floor per reporting period.
    pub minimum_payment: Option<f64>,
    /// Maximum payment cap per reporting period.
    pub maximum_payment: Option<f64>,
    /// Unix timestamp when this agreement becomes effective.
    pub valid_from: u64,
    /// Unix timestamp when this agreement expires (`None` = no expiry).
    pub valid_until: Option<u64>,
    /// ISO 4217 currency code (e.g. "USD", "EUR").
    pub currency: String,
}

impl RoyaltyAgreement {
    /// Return `true` if this agreement is active at the given Unix timestamp.
    pub fn is_active_at(&self, ts: u64) -> bool {
        if ts < self.valid_from {
            return false;
        }
        match self.valid_until {
            Some(end) => ts <= end,
            None => true,
        }
    }

    /// Return `true` if the given territory is covered by this agreement.
    ///
    /// An empty territory list means worldwide coverage.
    pub fn covers_territory(&self, territory: &str) -> bool {
        self.territory.is_empty() || self.territory.iter().any(|t| t == territory)
    }
}

// ── UsageEventType ───────────────────────────────────────────────────────────

/// The category of a usage event.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub enum UsageEventType {
    /// On-demand streaming.
    Stream,
    /// File download.
    Download,
    /// Broadcast (TV, radio, etc.).
    Broadcast,
    /// Synchronisation licence (use in a film, ad, etc.).
    SyncLicense,
    /// Physical or digital sale.
    Sale,
}

// ── UsageEvent ───────────────────────────────────────────────────────────────

/// A single recorded usage of a media asset.
#[derive(Debug, Clone)]
pub struct UsageEvent {
    /// Unique identifier for this event record.
    pub event_id: String,
    /// The asset that was used.
    pub asset_id: String,
    /// The type of usage.
    pub event_type: UsageEventType,
    /// Unix timestamp of the event.
    pub timestamp: u64,
    /// ISO 3166-1 alpha-2 territory code where the usage occurred.
    pub territory: String,
    /// Revenue generated by this event (if applicable).
    pub revenue: Option<f64>,
    /// Quantity of units (streams, downloads, physical copies …).
    pub quantity: u64,
}

// ── RoyaltyLineItem ──────────────────────────────────────────────────────────

/// One line of a royalty statement, corresponding to a single agreement.
#[derive(Debug)]
pub struct RoyaltyLineItem {
    /// The asset covered by this line.
    pub asset_id: String,
    /// The agreement that generated this line.
    pub agreement_id: String,
    /// Total number of qualifying usage events.
    pub usage_count: u64,
    /// Gross royalty before min/max caps are applied.
    pub gross_royalty: f64,
    /// Whether the minimum-payment floor was applied.
    pub minimum_applied: bool,
    /// Whether the maximum-payment cap was applied.
    pub maximum_applied: bool,
}

impl RoyaltyLineItem {
    /// Effective (net) royalty after caps.
    pub fn net_royalty(&self, agreement: &RoyaltyAgreement) -> f64 {
        let mut amount = self.gross_royalty;
        if let Some(min) = agreement.minimum_payment {
            if amount < min {
                amount = min;
            }
        }
        if let Some(max) = agreement.maximum_payment {
            if amount > max {
                amount = max;
            }
        }
        amount
    }
}

// ── RoyaltyStatement ─────────────────────────────────────────────────────────

/// A complete royalty statement for a single rights holder over a reporting period.
#[derive(Debug)]
pub struct RoyaltyStatement {
    /// The rights holder this statement addresses.
    pub rights_holder: String,
    /// Start of the reporting period (Unix timestamp).
    pub period_start: u64,
    /// End of the reporting period (Unix timestamp).
    pub period_end: u64,
    /// Individual line items, one per agreement.
    pub line_items: Vec<RoyaltyLineItem>,
    /// Sum of gross royalties across all line items.
    pub total_gross: f64,
    /// Sum of net royalties (after min/max) across all line items.
    pub total_net: f64,
    /// ISO 4217 currency code — should be uniform across all agreements.
    pub currency: String,
}

// ── RoyaltyEngine ────────────────────────────────────────────────────────────

/// Central engine that stores agreements and usage events and produces
/// royalty statements.
#[derive(Debug, Default)]
pub struct RoyaltyEngine {
    agreements: Vec<RoyaltyAgreement>,
    events: Vec<UsageEvent>,
}

impl RoyaltyEngine {
    /// Create a new, empty `RoyaltyEngine`.
    pub fn new() -> Self {
        Self::default()
    }

    /// Register a royalty agreement.
    pub fn register_agreement(&mut self, agreement: RoyaltyAgreement) {
        self.agreements.push(agreement);
    }

    /// Record a usage event.
    pub fn record_usage(&mut self, event: UsageEvent) {
        self.events.push(event);
    }

    /// Calculate the gross royalty for a single agreement given a slice of
    /// qualifying usage events.
    ///
    /// Minimum and maximum caps are **not** applied here; they are applied
    /// separately in [`generate_statement`](Self::generate_statement).
    pub fn calculate_royalty(agreement: &RoyaltyAgreement, events: &[UsageEvent]) -> f64 {
        match &agreement.basis {
            RoyaltyBasis::PerPlay => {
                // Count total quantity of play/stream events.
                let count: u64 = events.iter().map(|e| e.quantity).sum();
                count as f64
            }
            RoyaltyBasis::PerDownload => {
                let count: u64 = events
                    .iter()
                    .filter(|e| e.event_type == UsageEventType::Download)
                    .map(|e| e.quantity)
                    .sum();
                count as f64
            }
            RoyaltyBasis::RevenueShare(pct) => {
                let total_rev: f64 = events.iter().filter_map(|e| e.revenue).sum();
                total_rev * pct.clamp(0.0, 1.0)
            }
            RoyaltyBasis::MechanicalRate => {
                // One unit of currency per physical unit sold.
                let units: u64 = events
                    .iter()
                    .filter(|e| e.event_type == UsageEventType::Sale)
                    .map(|e| e.quantity)
                    .sum();
                units as f64
            }
            RoyaltyBasis::SyncFee(fee) => {
                // One-time fee regardless of event count.
                if events.is_empty() {
                    0.0
                } else {
                    *fee
                }
            }
            RoyaltyBasis::Custom(rate) => {
                let count: u64 = events.iter().map(|e| e.quantity).sum();
                count as f64 * rate
            }
        }
    }

    /// Return all agreements that are currently valid at the given timestamp.
    pub fn outstanding_agreements(&self, as_of: u64) -> Vec<&RoyaltyAgreement> {
        self.agreements
            .iter()
            .filter(|a| a.is_active_at(as_of))
            .collect()
    }

    /// Generate a royalty statement for a specific rights holder over the
    /// reporting period `[from_ts, to_ts]` (both inclusive, Unix timestamps).
    ///
    /// The currency of the statement is taken from the first matching
    /// agreement.  If there are no agreements the currency defaults to "USD".
    pub fn generate_statement(
        &self,
        rights_holder: &str,
        from_ts: u64,
        to_ts: u64,
    ) -> RoyaltyStatement {
        let holder_agreements: Vec<&RoyaltyAgreement> = self
            .agreements
            .iter()
            .filter(|a| a.rights_holder == rights_holder)
            .collect();

        let currency = holder_agreements
            .first()
            .map(|a| a.currency.clone())
            .unwrap_or_else(|| "USD".to_string());

        let mut line_items: Vec<RoyaltyLineItem> = Vec::new();
        let mut total_gross = 0.0_f64;
        let mut total_net = 0.0_f64;

        for agreement in &holder_agreements {
            // Collect qualifying events: matching asset_id, territory, and period.
            let qualifying: Vec<&UsageEvent> = self
                .events
                .iter()
                .filter(|e| {
                    e.asset_id == agreement.asset_id
                        && e.timestamp >= from_ts
                        && e.timestamp <= to_ts
                        && agreement.covers_territory(&e.territory)
                })
                .collect();

            let usage_count: u64 = qualifying.iter().map(|e| e.quantity).sum();

            // Collect owned copies for the calculation.
            let owned_events: Vec<UsageEvent> = qualifying
                .iter()
                .map(|&e| UsageEvent {
                    event_id: e.event_id.clone(),
                    asset_id: e.asset_id.clone(),
                    event_type: e.event_type.clone(),
                    timestamp: e.timestamp,
                    territory: e.territory.clone(),
                    revenue: e.revenue,
                    quantity: e.quantity,
                })
                .collect();

            let gross_royalty = Self::calculate_royalty(agreement, &owned_events);

            // Apply minimum and maximum caps.
            let mut net_royalty = gross_royalty;
            let mut minimum_applied = false;
            let mut maximum_applied = false;

            if let Some(min) = agreement.minimum_payment {
                if net_royalty < min {
                    net_royalty = min;
                    minimum_applied = true;
                }
            }
            if let Some(max) = agreement.maximum_payment {
                if net_royalty > max {
                    net_royalty = max;
                    maximum_applied = true;
                }
            }

            total_gross += gross_royalty;
            total_net += net_royalty;

            line_items.push(RoyaltyLineItem {
                asset_id: agreement.asset_id.clone(),
                agreement_id: agreement.id.clone(),
                usage_count,
                gross_royalty,
                minimum_applied,
                maximum_applied,
            });
        }

        RoyaltyStatement {
            rights_holder: rights_holder.to_string(),
            period_start: from_ts,
            period_end: to_ts,
            line_items,
            total_gross,
            total_net,
            currency,
        }
    }

    /// Return the agreements stored in this engine (for inspection).
    pub fn agreements(&self) -> &[RoyaltyAgreement] {
        &self.agreements
    }

    /// Return the events stored in this engine (for inspection).
    pub fn events(&self) -> &[UsageEvent] {
        &self.events
    }
}

// ── Helper to build test fixtures ────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn per_play_agreement(id: &str, asset: &str, holder: &str) -> RoyaltyAgreement {
        RoyaltyAgreement {
            id: id.to_string(),
            asset_id: asset.to_string(),
            rights_holder: holder.to_string(),
            territory: vec![],
            basis: RoyaltyBasis::PerPlay,
            minimum_payment: None,
            maximum_payment: None,
            valid_from: 0,
            valid_until: None,
            currency: "USD".to_string(),
        }
    }

    fn stream_event(id: &str, asset: &str, ts: u64, qty: u64) -> UsageEvent {
        UsageEvent {
            event_id: id.to_string(),
            asset_id: asset.to_string(),
            event_type: UsageEventType::Stream,
            timestamp: ts,
            territory: "US".to_string(),
            revenue: Some(0.01 * qty as f64),
            quantity: qty,
        }
    }

    // ── RoyaltyAgreement ─────────────────────────────────────────────────────

    #[test]
    fn test_agreement_is_active_within_range() {
        let agr = RoyaltyAgreement {
            id: "a1".to_string(),
            asset_id: "asset1".to_string(),
            rights_holder: "Alice".to_string(),
            territory: vec![],
            basis: RoyaltyBasis::PerPlay,
            minimum_payment: None,
            maximum_payment: None,
            valid_from: 1000,
            valid_until: Some(2000),
            currency: "USD".to_string(),
        };
        assert!(agr.is_active_at(1500));
    }

    #[test]
    fn test_agreement_is_not_active_before_start() {
        let agr = RoyaltyAgreement {
            id: "a1".to_string(),
            asset_id: "asset1".to_string(),
            rights_holder: "Alice".to_string(),
            territory: vec![],
            basis: RoyaltyBasis::PerPlay,
            minimum_payment: None,
            maximum_payment: None,
            valid_from: 1000,
            valid_until: Some(2000),
            currency: "USD".to_string(),
        };
        assert!(!agr.is_active_at(500));
    }

    #[test]
    fn test_agreement_covers_territory_empty_means_worldwide() {
        let agr = per_play_agreement("a1", "x", "Alice");
        assert!(agr.covers_territory("JP"));
        assert!(agr.covers_territory("DE"));
    }

    #[test]
    fn test_agreement_covers_territory_specific() {
        let mut agr = per_play_agreement("a1", "x", "Alice");
        agr.territory = vec!["US".to_string(), "CA".to_string()];
        assert!(agr.covers_territory("US"));
        assert!(!agr.covers_territory("JP"));
    }

    // ── calculate_royalty ────────────────────────────────────────────────────

    #[test]
    fn test_per_play_royalty() {
        let agr = per_play_agreement("a1", "asset1", "Alice");
        let events = vec![
            stream_event("e1", "asset1", 100, 10),
            stream_event("e2", "asset1", 200, 5),
        ];
        let royalty = RoyaltyEngine::calculate_royalty(&agr, &events);
        assert!((royalty - 15.0).abs() < 1e-9, "Expected 15, got {royalty}");
    }

    #[test]
    fn test_revenue_share_royalty() {
        let mut agr = per_play_agreement("a1", "asset1", "Alice");
        agr.basis = RoyaltyBasis::RevenueShare(0.1);
        let events = vec![
            UsageEvent {
                event_id: "e1".to_string(),
                asset_id: "asset1".to_string(),
                event_type: UsageEventType::Stream,
                timestamp: 100,
                territory: "US".to_string(),
                revenue: Some(200.0),
                quantity: 1,
            },
            UsageEvent {
                event_id: "e2".to_string(),
                asset_id: "asset1".to_string(),
                event_type: UsageEventType::Stream,
                timestamp: 200,
                territory: "US".to_string(),
                revenue: Some(300.0),
                quantity: 1,
            },
        ];
        // 10% of 500 = 50
        let royalty = RoyaltyEngine::calculate_royalty(&agr, &events);
        assert!((royalty - 50.0).abs() < 1e-9, "Expected 50, got {royalty}");
    }

    #[test]
    fn test_sync_fee_royalty_nonzero_events() {
        let mut agr = per_play_agreement("a1", "asset1", "Alice");
        agr.basis = RoyaltyBasis::SyncFee(5000.0);
        let events = vec![stream_event("e1", "asset1", 100, 1)];
        let royalty = RoyaltyEngine::calculate_royalty(&agr, &events);
        assert!(
            (royalty - 5000.0).abs() < 1e-9,
            "Expected 5000, got {royalty}"
        );
    }

    #[test]
    fn test_sync_fee_royalty_zero_events_yields_zero() {
        let mut agr = per_play_agreement("a1", "asset1", "Alice");
        agr.basis = RoyaltyBasis::SyncFee(5000.0);
        let royalty = RoyaltyEngine::calculate_royalty(&agr, &[]);
        assert!(royalty.abs() < 1e-9, "Expected 0, got {royalty}");
    }

    #[test]
    fn test_custom_rate_royalty() {
        let mut agr = per_play_agreement("a1", "asset1", "Alice");
        agr.basis = RoyaltyBasis::Custom(0.05);
        let events = vec![stream_event("e1", "asset1", 100, 100)];
        let royalty = RoyaltyEngine::calculate_royalty(&agr, &events);
        assert!((royalty - 5.0).abs() < 1e-9, "Expected 5, got {royalty}");
    }

    // ── generate_statement ───────────────────────────────────────────────────

    #[test]
    fn test_generate_statement_basic() {
        let mut engine = RoyaltyEngine::new();
        engine.register_agreement(per_play_agreement("a1", "asset1", "Alice"));
        engine.record_usage(stream_event("e1", "asset1", 500, 10));
        engine.record_usage(stream_event("e2", "asset1", 600, 5));
        let stmt = engine.generate_statement("Alice", 0, 1000);
        assert_eq!(stmt.rights_holder, "Alice");
        assert_eq!(stmt.line_items.len(), 1);
        assert!((stmt.total_gross - 15.0).abs() < 1e-9);
    }

    #[test]
    fn test_generate_statement_time_filter() {
        let mut engine = RoyaltyEngine::new();
        engine.register_agreement(per_play_agreement("a1", "asset1", "Alice"));
        // inside window
        engine.record_usage(stream_event("e1", "asset1", 500, 10));
        // outside window
        engine.record_usage(stream_event("e2", "asset1", 2000, 100));
        let stmt = engine.generate_statement("Alice", 0, 1000);
        assert!((stmt.total_gross - 10.0).abs() < 1e-9);
    }

    #[test]
    fn test_generate_statement_minimum_applied() {
        let mut engine = RoyaltyEngine::new();
        let mut agr = per_play_agreement("a1", "asset1", "Alice");
        agr.minimum_payment = Some(100.0);
        engine.register_agreement(agr);
        engine.record_usage(stream_event("e1", "asset1", 500, 1)); // gross = 1
        let stmt = engine.generate_statement("Alice", 0, 1000);
        assert!(
            (stmt.total_net - 100.0).abs() < 1e-9,
            "min should be applied"
        );
        assert!(stmt.line_items[0].minimum_applied);
    }

    #[test]
    fn test_generate_statement_maximum_applied() {
        let mut engine = RoyaltyEngine::new();
        let mut agr = per_play_agreement("a1", "asset1", "Alice");
        agr.maximum_payment = Some(50.0);
        engine.register_agreement(agr);
        engine.record_usage(stream_event("e1", "asset1", 500, 1000)); // gross = 1000
        let stmt = engine.generate_statement("Alice", 0, 1000);
        assert!(
            (stmt.total_net - 50.0).abs() < 1e-9,
            "max should be applied"
        );
        assert!(stmt.line_items[0].maximum_applied);
    }

    #[test]
    fn test_outstanding_agreements_filters_by_timestamp() {
        let mut engine = RoyaltyEngine::new();
        let mut agr1 = per_play_agreement("a1", "asset1", "Alice");
        agr1.valid_from = 0;
        agr1.valid_until = Some(1000);
        let mut agr2 = per_play_agreement("a2", "asset2", "Bob");
        agr2.valid_from = 2000;
        agr2.valid_until = None;
        engine.register_agreement(agr1);
        engine.register_agreement(agr2);
        // At time 500 only agr1 is active
        let active = engine.outstanding_agreements(500);
        assert_eq!(active.len(), 1);
        assert_eq!(active[0].id, "a1");
    }

    #[test]
    fn test_generate_statement_territory_filter() {
        let mut engine = RoyaltyEngine::new();
        let mut agr = per_play_agreement("a1", "asset1", "Alice");
        agr.territory = vec!["US".to_string()];
        engine.register_agreement(agr);
        // US event
        engine.record_usage(stream_event("e1", "asset1", 100, 10));
        // JP event – should be excluded
        engine.record_usage(UsageEvent {
            event_id: "e2".to_string(),
            asset_id: "asset1".to_string(),
            event_type: UsageEventType::Stream,
            timestamp: 200,
            territory: "JP".to_string(),
            revenue: None,
            quantity: 50,
        });
        let stmt = engine.generate_statement("Alice", 0, 1000);
        assert!(
            (stmt.total_gross - 10.0).abs() < 1e-9,
            "Only US events should count, got {}",
            stmt.total_gross
        );
    }

    #[test]
    fn test_generate_statement_no_matching_holder() {
        let engine = RoyaltyEngine::new();
        let stmt = engine.generate_statement("Unknown", 0, 1000);
        assert!(stmt.line_items.is_empty());
        assert!((stmt.total_gross).abs() < 1e-9);
    }
}
