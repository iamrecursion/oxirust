//! Royalty type definitions, rates, and per-period calculation aggregation

#![allow(dead_code)]

/// The category of royalty being tracked
#[derive(Debug, Clone, PartialEq)]
pub enum RoyaltyType {
    /// Mechanical license (reproduction of music)
    MechanicalLicense,
    /// Performance royalty (public performance of music)
    PerformanceRoyalty,
    /// Synchronization license (music in video)
    SyncLicense,
    /// Master recording license
    MasterLicense,
    /// Public performance (venue / broadcast)
    PublicPerformance,
}

impl RoyaltyType {
    /// Human-readable label for this royalty type
    pub fn label(&self) -> &str {
        match self {
            RoyaltyType::MechanicalLicense => "Mechanical License",
            RoyaltyType::PerformanceRoyalty => "Performance Royalty",
            RoyaltyType::SyncLicense => "Sync License",
            RoyaltyType::MasterLicense => "Master License",
            RoyaltyType::PublicPerformance => "Public Performance",
        }
    }
}

/// Rate configuration for a specific royalty type
#[derive(Debug, Clone)]
pub struct RoyaltyRate {
    /// Category this rate applies to
    pub royalty_type: RoyaltyType,
    /// Percentage of revenue owed (e.g. 10.0 means 10%)
    pub rate_pct: f64,
    /// Minimum payment regardless of calculated amount
    pub min_payment: f64,
    /// ISO 4217 currency code (e.g. "USD")
    pub currency: String,
}

impl RoyaltyRate {
    /// Create a new royalty rate
    pub fn new(
        royalty_type: RoyaltyType,
        rate_pct: f64,
        min_payment: f64,
        currency: impl Into<String>,
    ) -> Self {
        Self {
            royalty_type,
            rate_pct,
            min_payment,
            currency: currency.into(),
        }
    }

    /// Calculate the royalty amount for the given revenue (enforces minimum)
    pub fn calculate(&self, revenue: f64) -> f64 {
        let calculated = revenue * (self.rate_pct / 100.0);
        calculated.max(self.min_payment)
    }
}

/// A single royalty calculation for one piece of content in a period
#[derive(Debug, Clone)]
pub struct RoyaltyCalculation {
    /// Content identifier
    pub content_id: String,
    /// Number of plays / streams in the period
    pub plays: u64,
    /// Total revenue generated in the period
    pub revenue: f64,
    /// Rate information used to compute the royalty
    pub rate: RoyaltyRate,
}

impl RoyaltyCalculation {
    /// Create a new royalty calculation
    pub fn new(content_id: impl Into<String>, plays: u64, revenue: f64, rate: RoyaltyRate) -> Self {
        Self {
            content_id: content_id.into(),
            plays,
            revenue,
            rate,
        }
    }

    /// Total royalty amount owed for this calculation
    pub fn total_royalty(&self) -> f64 {
        self.rate.calculate(self.revenue)
    }

    /// Royalty amount per individual play
    pub fn per_play(&self) -> f64 {
        if self.plays == 0 {
            return 0.0;
        }
        self.total_royalty() / self.plays as f64
    }
}

/// A summary statement of all royalties due for a period
#[derive(Debug)]
pub struct RoyaltyStatement {
    /// Period start (Unix timestamp)
    pub period_start: u64,
    /// Period end (Unix timestamp)
    pub period_end: u64,
    calculations: Vec<RoyaltyCalculation>,
    cached_total: f64,
}

impl RoyaltyStatement {
    /// Create a new empty statement for the given period
    pub fn new(start: u64, end: u64) -> Self {
        Self {
            period_start: start,
            period_end: end,
            calculations: Vec::new(),
            cached_total: 0.0,
        }
    }

    /// Add a royalty calculation to this statement
    pub fn add(&mut self, calc: RoyaltyCalculation) {
        self.cached_total += calc.total_royalty();
        self.calculations.push(calc);
    }

    /// Get the total amount due (sum of all calculation royalties)
    pub fn total_due(&self) -> f64 {
        self.cached_total
    }

    /// Summarise the total amount due grouped by royalty type label
    pub fn summary_by_type(&self) -> Vec<(String, f64)> {
        use std::collections::HashMap;
        let mut map: HashMap<String, f64> = HashMap::new();
        for calc in &self.calculations {
            *map.entry(calc.rate.royalty_type.label().to_string())
                .or_insert(0.0) += calc.total_royalty();
        }
        let mut result: Vec<(String, f64)> = map.into_iter().collect();
        result.sort_by(|a, b| a.0.cmp(&b.0));
        result
    }

    /// Number of calculations in this statement
    pub fn calculation_count(&self) -> usize {
        self.calculations.len()
    }

    /// Access the underlying calculations slice
    pub fn calculations(&self) -> &[RoyaltyCalculation] {
        &self.calculations
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_rate(royalty_type: RoyaltyType) -> RoyaltyRate {
        RoyaltyRate::new(royalty_type, 10.0, 1.0, "USD")
    }

    #[test]
    fn test_royalty_rate_calculate_basic() {
        let rate = make_rate(RoyaltyType::MechanicalLicense);
        let amount = rate.calculate(200.0);
        assert!((amount - 20.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_royalty_rate_min_payment_enforced() {
        let rate = make_rate(RoyaltyType::SyncLicense);
        // 10% of 0.0 = 0.0, but min is 1.0
        let amount = rate.calculate(0.0);
        assert!((amount - 1.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_royalty_calculation_total() {
        let rate = make_rate(RoyaltyType::PerformanceRoyalty);
        let calc = RoyaltyCalculation::new("track-1", 500, 100.0, rate);
        assert!((calc.total_royalty() - 10.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_royalty_calculation_per_play() {
        let rate = make_rate(RoyaltyType::MasterLicense);
        let calc = RoyaltyCalculation::new("track-2", 100, 200.0, rate);
        // total = 20.0, per_play = 20.0 / 100 = 0.2
        assert!((calc.per_play() - 0.2).abs() < 1e-9);
    }

    #[test]
    fn test_royalty_calculation_per_play_zero_plays() {
        let rate = make_rate(RoyaltyType::PublicPerformance);
        let calc = RoyaltyCalculation::new("track-3", 0, 100.0, rate);
        assert_eq!(calc.per_play(), 0.0);
    }

    #[test]
    fn test_statement_new_empty() {
        let stmt = RoyaltyStatement::new(0, 1_000_000);
        assert_eq!(stmt.total_due(), 0.0);
        assert_eq!(stmt.calculation_count(), 0);
    }

    #[test]
    fn test_statement_add_and_total() {
        let mut stmt = RoyaltyStatement::new(0, 1_000_000);
        stmt.add(RoyaltyCalculation::new(
            "a",
            100,
            100.0,
            make_rate(RoyaltyType::MechanicalLicense),
        ));
        stmt.add(RoyaltyCalculation::new(
            "b",
            200,
            200.0,
            make_rate(RoyaltyType::SyncLicense),
        ));
        // 10% of 100 = 10, 10% of 200 = 20 → total = 30
        assert!((stmt.total_due() - 30.0).abs() < f64::EPSILON);
        assert_eq!(stmt.calculation_count(), 2);
    }

    #[test]
    fn test_statement_summary_by_type_groups_correctly() {
        let mut stmt = RoyaltyStatement::new(0, 1_000_000);
        stmt.add(RoyaltyCalculation::new(
            "a",
            100,
            100.0,
            make_rate(RoyaltyType::MechanicalLicense),
        ));
        stmt.add(RoyaltyCalculation::new(
            "b",
            100,
            100.0,
            make_rate(RoyaltyType::MechanicalLicense),
        ));
        let summary = stmt.summary_by_type();
        assert_eq!(summary.len(), 1);
        assert_eq!(summary[0].0, "Mechanical License");
        assert!((summary[0].1 - 20.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_statement_summary_multiple_types() {
        let mut stmt = RoyaltyStatement::new(0, 1_000_000);
        stmt.add(RoyaltyCalculation::new(
            "a",
            100,
            100.0,
            make_rate(RoyaltyType::MechanicalLicense),
        ));
        stmt.add(RoyaltyCalculation::new(
            "b",
            100,
            100.0,
            make_rate(RoyaltyType::SyncLicense),
        ));
        let summary = stmt.summary_by_type();
        assert_eq!(summary.len(), 2);
    }

    #[test]
    fn test_royalty_type_labels() {
        assert_eq!(RoyaltyType::MechanicalLicense.label(), "Mechanical License");
        assert_eq!(
            RoyaltyType::PerformanceRoyalty.label(),
            "Performance Royalty"
        );
        assert_eq!(RoyaltyType::SyncLicense.label(), "Sync License");
        assert_eq!(RoyaltyType::MasterLicense.label(), "Master License");
        assert_eq!(RoyaltyType::PublicPerformance.label(), "Public Performance");
    }

    #[test]
    fn test_statement_period_stored() {
        let stmt = RoyaltyStatement::new(500, 1_500);
        assert_eq!(stmt.period_start, 500);
        assert_eq!(stmt.period_end, 1_500);
    }

    #[test]
    fn test_rate_fields_stored_correctly() {
        let rate = RoyaltyRate::new(RoyaltyType::SyncLicense, 12.5, 5.0, "EUR");
        assert_eq!(rate.rate_pct, 12.5);
        assert_eq!(rate.min_payment, 5.0);
        assert_eq!(rate.currency, "EUR");
    }

    #[test]
    fn test_calculation_content_id_stored() {
        let rate = make_rate(RoyaltyType::MechanicalLicense);
        let calc = RoyaltyCalculation::new("my-content-123", 10, 50.0, rate);
        assert_eq!(calc.content_id, "my-content-123");
        assert_eq!(calc.plays, 10);
        assert_eq!(calc.revenue, 50.0);
    }
}
