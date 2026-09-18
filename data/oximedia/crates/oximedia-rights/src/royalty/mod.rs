//! Royalty calculation and tracking module.

#![allow(dead_code)]

pub mod calculate;
pub mod payment;
pub mod report;
pub mod statement;
pub mod territory;
pub mod types;

pub use calculate::RoyaltyCalculator;
pub use payment::{PaymentStatus, RoyaltyPayment};
pub use report::RoyaltyReport;
pub use types::{RoyaltyCalculation, RoyaltyStatement as TypesStatement, RoyaltyType};

// ── RoyaltyRate ───────────────────────────────────────────────────────────────

/// Rate configuration for royalty computation
#[derive(Debug, Clone)]
pub struct RoyaltyRate {
    /// Percentage of revenue owed (e.g. 10.0 means 10 %)
    pub percentage: f32,
    /// Minimum payment regardless of calculated amount
    pub min_payment: f32,
    /// ISO 4217 currency code (e.g. "USD")
    pub currency: String,
}

impl RoyaltyRate {
    /// Create a new royalty rate
    pub fn new(percentage: f32, min_payment: f32, currency: &str) -> Self {
        Self {
            percentage,
            min_payment,
            currency: currency.to_string(),
        }
    }

    /// Calculate royalty for the given `revenue`, enforcing the minimum payment
    pub fn calculate(&self, revenue: f32) -> f32 {
        let calculated = revenue * (self.percentage / 100.0);
        calculated.max(self.min_payment)
    }

    /// Returns `true` when the rate percentage is in the range (0, 100]
    pub fn is_valid(&self) -> bool {
        self.percentage > 0.0 && self.percentage <= 100.0
    }
}

// ── RoyaltyPeriod ─────────────────────────────────────────────────────────────

/// Reporting period for royalty statements
#[derive(Debug, Clone, PartialEq)]
pub enum RoyaltyPeriod {
    /// One calendar month
    Monthly,
    /// Three calendar months
    Quarterly,
    /// Twelve calendar months
    Annual,
}

impl RoyaltyPeriod {
    /// Number of calendar months this period covers
    pub fn months(&self) -> u32 {
        match self {
            RoyaltyPeriod::Monthly => 1,
            RoyaltyPeriod::Quarterly => 3,
            RoyaltyPeriod::Annual => 12,
        }
    }
}

// ── RoyaltyStatement ──────────────────────────────────────────────────────────

/// A single computed royalty statement for a payee over a period
#[derive(Debug, Clone)]
pub struct RoyaltyStatement {
    /// Name of the payee (rights holder)
    pub payee: String,
    /// Reporting period
    pub period: RoyaltyPeriod,
    /// Number of plays / streams in the period
    pub plays: u64,
    /// Total revenue generated in the period
    pub revenue: f32,
    /// Rate applied to compute the royalty
    pub rate: RoyaltyRate,
    /// Pre-computed royalty amount owed
    pub amount_due: f32,
}

impl RoyaltyStatement {
    /// Build a `RoyaltyStatement` by computing `amount_due` from `plays`, `revenue`, and `rate`
    pub fn compute(
        payee: &str,
        period: RoyaltyPeriod,
        plays: u64,
        revenue: f32,
        rate: &RoyaltyRate,
    ) -> Self {
        let amount_due = rate.calculate(revenue);
        Self {
            payee: payee.to_string(),
            period,
            plays,
            revenue,
            rate: rate.clone(),
            amount_due,
        }
    }
}

// ── RoyaltyLedger ─────────────────────────────────────────────────────────────

/// Collection of royalty statements forming an accounting ledger
#[derive(Debug, Default)]
pub struct RoyaltyLedger {
    /// All statements recorded
    pub statements: Vec<RoyaltyStatement>,
}

impl RoyaltyLedger {
    /// Create an empty ledger
    pub fn new() -> Self {
        Self::default()
    }

    /// Add a statement to the ledger
    pub fn add(&mut self, stmt: RoyaltyStatement) {
        self.statements.push(stmt);
    }

    /// Sum of `amount_due` for all statements belonging to `payee`
    pub fn total_due(&self, payee: &str) -> f32 {
        self.statements
            .iter()
            .filter(|s| s.payee == payee)
            .map(|s| s.amount_due)
            .sum()
    }

    /// Total number of plays across all statements for `payee`
    pub fn total_plays(&self, payee: &str) -> u64 {
        self.statements
            .iter()
            .filter(|s| s.payee == payee)
            .map(|s| s.plays)
            .sum()
    }

    /// Sorted, deduplicated list of payee names in the ledger
    pub fn payees(&self) -> Vec<&str> {
        let mut names: Vec<&str> = self.statements.iter().map(|s| s.payee.as_str()).collect();
        names.sort_unstable();
        names.dedup();
        names
    }
}

// ── Unit tests ────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn usd_rate(pct: f32) -> RoyaltyRate {
        RoyaltyRate::new(pct, 1.0, "USD")
    }

    // ── RoyaltyRate ──────────────────────────────────────────────────────────

    #[test]
    fn test_rate_calculate_basic() {
        let rate = usd_rate(10.0);
        assert!((rate.calculate(200.0) - 20.0).abs() < f32::EPSILON);
    }

    #[test]
    fn test_rate_min_payment_enforced() {
        let rate = usd_rate(10.0);
        // 10% of 0 = 0, but min is 1.0
        assert!((rate.calculate(0.0) - 1.0).abs() < f32::EPSILON);
    }

    #[test]
    fn test_rate_is_valid_positive_pct() {
        assert!(usd_rate(15.0).is_valid());
    }

    #[test]
    fn test_rate_is_invalid_zero_pct() {
        assert!(!RoyaltyRate::new(0.0, 0.0, "USD").is_valid());
    }

    #[test]
    fn test_rate_is_invalid_over_100() {
        assert!(!RoyaltyRate::new(101.0, 0.0, "USD").is_valid());
    }

    // ── RoyaltyPeriod ────────────────────────────────────────────────────────

    #[test]
    fn test_period_months_monthly() {
        assert_eq!(RoyaltyPeriod::Monthly.months(), 1);
    }

    #[test]
    fn test_period_months_quarterly() {
        assert_eq!(RoyaltyPeriod::Quarterly.months(), 3);
    }

    #[test]
    fn test_period_months_annual() {
        assert_eq!(RoyaltyPeriod::Annual.months(), 12);
    }

    // ── RoyaltyStatement ─────────────────────────────────────────────────────

    #[test]
    fn test_statement_compute_amount_due() {
        let rate = usd_rate(10.0);
        let stmt = RoyaltyStatement::compute("Alice", RoyaltyPeriod::Monthly, 500, 100.0, &rate);
        assert!((stmt.amount_due - 10.0).abs() < f32::EPSILON);
        assert_eq!(stmt.payee, "Alice");
        assert_eq!(stmt.plays, 500);
    }

    #[test]
    fn test_statement_compute_enforces_min_payment() {
        let rate = usd_rate(5.0);
        let stmt = RoyaltyStatement::compute("Bob", RoyaltyPeriod::Annual, 0, 0.0, &rate);
        assert!((stmt.amount_due - 1.0).abs() < f32::EPSILON);
    }

    // ── RoyaltyLedger ────────────────────────────────────────────────────────

    #[test]
    fn test_ledger_total_due_for_payee() {
        let rate = usd_rate(10.0);
        let mut ledger = RoyaltyLedger::new();
        ledger.add(RoyaltyStatement::compute(
            "Alice",
            RoyaltyPeriod::Monthly,
            100,
            200.0,
            &rate,
        ));
        ledger.add(RoyaltyStatement::compute(
            "Alice",
            RoyaltyPeriod::Monthly,
            100,
            300.0,
            &rate,
        ));
        // 20 + 30 = 50
        assert!((ledger.total_due("Alice") - 50.0).abs() < f32::EPSILON);
    }

    #[test]
    fn test_ledger_total_due_different_payees_isolated() {
        let rate = usd_rate(10.0);
        let mut ledger = RoyaltyLedger::new();
        ledger.add(RoyaltyStatement::compute(
            "Alice",
            RoyaltyPeriod::Monthly,
            100,
            200.0,
            &rate,
        ));
        ledger.add(RoyaltyStatement::compute(
            "Bob",
            RoyaltyPeriod::Monthly,
            100,
            400.0,
            &rate,
        ));
        assert!((ledger.total_due("Alice") - 20.0).abs() < f32::EPSILON);
        assert!((ledger.total_due("Bob") - 40.0).abs() < f32::EPSILON);
    }

    #[test]
    fn test_ledger_total_plays_aggregated() {
        let rate = usd_rate(10.0);
        let mut ledger = RoyaltyLedger::new();
        ledger.add(RoyaltyStatement::compute(
            "Alice",
            RoyaltyPeriod::Monthly,
            100,
            0.0,
            &rate,
        ));
        ledger.add(RoyaltyStatement::compute(
            "Alice",
            RoyaltyPeriod::Quarterly,
            250,
            0.0,
            &rate,
        ));
        assert_eq!(ledger.total_plays("Alice"), 350);
    }

    #[test]
    fn test_ledger_payees_sorted_deduplicated() {
        let rate = usd_rate(10.0);
        let mut ledger = RoyaltyLedger::new();
        ledger.add(RoyaltyStatement::compute(
            "Zara",
            RoyaltyPeriod::Monthly,
            10,
            10.0,
            &rate,
        ));
        ledger.add(RoyaltyStatement::compute(
            "Alice",
            RoyaltyPeriod::Monthly,
            10,
            10.0,
            &rate,
        ));
        ledger.add(RoyaltyStatement::compute(
            "Alice",
            RoyaltyPeriod::Quarterly,
            10,
            10.0,
            &rate,
        ));
        let payees = ledger.payees();
        assert_eq!(payees, vec!["Alice", "Zara"]);
    }

    #[test]
    fn test_ledger_unknown_payee_returns_zero() {
        let ledger = RoyaltyLedger::new();
        assert_eq!(ledger.total_due("Ghost"), 0.0);
        assert_eq!(ledger.total_plays("Ghost"), 0);
    }
}
