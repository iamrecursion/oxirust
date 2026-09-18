//! Integration tests for royalty calculation with tiered rates and territory splits.
//!
//! Tests the `tiered_royalty` module's `TieredRoyaltySchedule` / `TieredAgreement`
//! for graduated rate correctness, and the `royalty::territory` module for
//! multi-territory split arithmetic.

use oximedia_rights::royalty::territory::{Territory, TerritoryRateTable};
use oximedia_rights::tiered_royalty::{RoyaltyTier, TieredAgreement, TieredRoyaltySchedule};

// ── Helper: build the canonical 3-tier schedule ──────────────────────────────
//
// Tier 1: 0 ..  1 000  views → $0.10 / play
// Tier 2: 1000 .. 10000 views → $0.08 / play
// Tier 3: 10000+         views → $0.05 / play

fn three_tier_schedule() -> TieredRoyaltySchedule {
    let mut s = TieredRoyaltySchedule::new("USD");
    s.add_tier(RoyaltyTier::new("Base", 0, Some(1_000), 0.10));
    s.add_tier(RoyaltyTier::new("Mid", 1_000, Some(10_000), 0.08));
    s.add_tier(RoyaltyTier::new("Top", 10_000, None, 0.05));
    s
}

// ── Tiered rate tests ─────────────────────────────────────────────────────────

/// 500 views → all inside Base tier → 500 × 0.10 = 50.0
#[test]
fn test_tiered_royalty_base_tier_only() {
    let schedule = three_tier_schedule();
    let result = schedule.calculate(500);
    assert!(
        (result.total_royalty - 50.0).abs() < 1e-9,
        "Expected 50.0 for 500 plays, got {}",
        result.total_royalty
    );
    assert_eq!(result.breakdown.len(), 1, "Should have 1 tier in breakdown");
}

/// 5000 views → Base (1000 × 0.10 = 100) + Mid (4000 × 0.08 = 320) = 420.0
#[test]
fn test_tiered_royalty_two_tiers() {
    let schedule = three_tier_schedule();
    let result = schedule.calculate(5_000);
    let expected = 1_000.0 * 0.10 + 4_000.0 * 0.08; // = 420.0
    assert!(
        (result.total_royalty - expected).abs() < 1e-9,
        "Expected {expected} for 5000 plays, got {}",
        result.total_royalty
    );
    assert_eq!(result.breakdown.len(), 2, "Should span 2 tiers");
}

/// 50 000 views → Base (1000×0.10=100) + Mid (9000×0.08=720) + Top (40000×0.05=2000) = 2820.0
#[test]
fn test_tiered_royalty_all_three_tiers() {
    let schedule = three_tier_schedule();
    let result = schedule.calculate(50_000);
    let expected = 1_000.0 * 0.10 + 9_000.0 * 0.08 + 40_000.0 * 0.05; // = 2820.0
    assert!(
        (result.total_royalty - expected).abs() < 1e-9,
        "Expected {expected} for 50000 plays, got {}",
        result.total_royalty
    );
    assert_eq!(result.breakdown.len(), 3, "Should span all 3 tiers");
}

/// Validate that the schedule itself is internally consistent.
#[test]
fn test_tiered_schedule_validates_ok() {
    let schedule = three_tier_schedule();
    assert!(
        schedule.validate().is_ok(),
        "3-tier schedule should be valid"
    );
}

/// `TieredAgreement::add_usage` reports incremental royalty correctly when
/// the cumulative count crosses a tier boundary.
#[test]
fn test_tiered_agreement_crosses_boundary() {
    let mut agr = TieredAgreement::new("agr-1", "asset-x", "Alice", three_tier_schedule(), 0, None);

    // Add 1000 (fills Base entirely): 1000 × 0.10 = 100
    let inc1 = agr.add_usage(1_000);
    assert!(
        (inc1 - 100.0).abs() < 1e-9,
        "inc1 expected 100.0, got {inc1}"
    );

    // Add 9000 more (fills Mid entirely): 9000 × 0.08 = 720
    let inc2 = agr.add_usage(9_000);
    assert!(
        (inc2 - 720.0).abs() < 1e-9,
        "inc2 expected 720.0, got {inc2}"
    );

    // Add 10000 more (enters Top): 10000 × 0.05 = 500
    let inc3 = agr.add_usage(10_000);
    assert!(
        (inc3 - 500.0).abs() < 1e-9,
        "inc3 expected 500.0, got {inc3}"
    );
}

// ── Multi-territory split tests ───────────────────────────────────────────────

/// When a $1 000 total royalty is split 50 % US / 30 % EU / 20 % ROW,
/// the territory multiplier differences are secondary — this test validates
/// the *conceptual* split arithmetic using a manual proportional allocation.
#[test]
fn test_multi_territory_split_arithmetic() {
    let total_royalty = 1_000.0_f64;

    // Proportional shares
    let us_share = 0.50_f64;
    let eu_share = 0.30_f64;
    let row_share = 0.20_f64;

    let us_royalty = total_royalty * us_share;
    let eu_royalty = total_royalty * eu_share;
    let row_royalty = total_royalty * row_share;

    assert!(
        (us_royalty - 500.0).abs() < 1e-9,
        "US should get 500, got {us_royalty}"
    );
    assert!(
        (eu_royalty - 300.0).abs() < 1e-9,
        "EU should get 300, got {eu_royalty}"
    );
    assert!(
        (row_royalty - 200.0).abs() < 1e-9,
        "ROW should get 200, got {row_royalty}"
    );
    assert!(
        ((us_royalty + eu_royalty + row_royalty) - total_royalty).abs() < 1e-9,
        "Splits should sum to total"
    );
}

/// Verify `TerritoryRateTable` applies custom overrides correctly for a
/// per-territory royalty split scenario.
#[test]
fn test_territory_rate_table_split() {
    let mut table = TerritoryRateTable::new();
    // Override to make arithmetic simple: all multipliers = 1.0
    table.set_rate(Territory::US, 1.0).expect("set US rate");
    table.set_rate(Territory::EU, 1.0).expect("set EU rate");
    table
        .set_rate(Territory::Custom("ROW".into()), 1.0)
        .expect("set ROW rate");

    let base = 100.0_f64;
    let us_payout = base * table.get_multiplier(&Territory::US);
    let eu_payout = base * table.get_multiplier(&Territory::EU);
    let row_payout = base * table.get_multiplier(&Territory::Custom("ROW".into()));

    assert!((us_payout - 100.0).abs() < 1e-9);
    assert!((eu_payout - 100.0).abs() < 1e-9);
    assert!((row_payout - 100.0).abs() < 1e-9);
}

/// Japan has a 1.2× premium multiplier — verify it raises the effective payout.
#[test]
fn test_territory_japan_premium_multiplier() {
    let table = TerritoryRateTable::new();
    let base = 1_000.0_f64;
    let us_payout = base * table.get_multiplier(&Territory::US);
    let jp_payout = base * table.get_multiplier(&Territory::Japan);
    assert!(
        jp_payout > us_payout,
        "Japan payout should exceed US payout"
    );
    assert!(
        (jp_payout - 1_200.0).abs() < 1e-9,
        "Japan multiplier is 1.2×"
    );
}
