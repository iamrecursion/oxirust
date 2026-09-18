//! Royalty calculation engine.
//!
//! Provides types for computing royalty payments from basis-point rates,
//! organising payments into statements, and querying totals by type or
//! rights-holder.

#![allow(dead_code)]
#![allow(missing_docs)]
#![allow(clippy::cast_precision_loss)]

// ── RoyaltyRate ──────────────────────────────────────────────────────────────

/// A royalty rate expressed in basis points (1 bp = 0.01 %).
///
/// Storing as integer basis points avoids floating-point rounding during
/// multiplication while still allowing a human-friendly percentage view.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RoyaltyRate {
    /// Basis-point value (e.g. 500 = 5 %).
    pub basis_points: u32,
}

impl RoyaltyRate {
    /// Construct a rate from raw basis points.
    #[must_use]
    pub const fn new(basis_points: u32) -> Self {
        Self { basis_points }
    }

    /// Return the rate as a percentage (e.g. 500 bp → 5.0 %).
    #[must_use]
    pub fn pct(&self) -> f32 {
        self.basis_points as f32 / 100.0
    }

    /// Construct a rate from a percentage value (e.g. 5.0 → 500 bp).
    ///
    /// The percentage is rounded to the nearest basis point.
    #[must_use]
    pub fn from_pct(pct: f32) -> Self {
        let bp = (pct * 100.0).round() as u32;
        Self { basis_points: bp }
    }
}

// ── RoyaltyType ──────────────────────────────────────────────────────────────

/// The category of a royalty payment.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum RoyaltyType {
    /// Mechanical royalty (reproduction of a composition).
    Mechanical,
    /// Performance royalty (public performance / broadcast).
    Performance,
    /// Synchronisation royalty (use of music in AV works).
    Synchronisation,
    /// Master-use royalty (use of a specific sound recording).
    MasterUse,
    /// Print royalty (sheet music / lyrics reproduction).
    Print,
}

impl RoyaltyType {
    /// Returns `true` for royalty types that relate to the publishing side
    /// of music rights (composition / lyrics), as opposed to master rights.
    #[must_use]
    pub fn is_publishing(&self) -> bool {
        matches!(
            self,
            Self::Mechanical | Self::Performance | Self::Synchronisation | Self::Print
        )
    }
}

// ── RoyaltyPayment ────────────────────────────────────────────────────────────

/// A single royalty payment owed to a rights holder.
#[derive(Debug, Clone)]
pub struct RoyaltyPayment {
    /// Name (or identifier) of the rights holder receiving the payment.
    pub rights_holder: String,
    /// Category of this royalty.
    pub royalty_type: RoyaltyType,
    /// Agreed royalty rate.
    pub rate: RoyaltyRate,
    /// Gross revenue on which the royalty is calculated, in US cents.
    pub gross_revenue_cents: u64,
}

impl RoyaltyPayment {
    /// Create a new royalty payment.
    pub fn new(
        rights_holder: impl Into<String>,
        royalty_type: RoyaltyType,
        rate: RoyaltyRate,
        gross_revenue_cents: u64,
    ) -> Self {
        Self {
            rights_holder: rights_holder.into(),
            royalty_type,
            rate,
            gross_revenue_cents,
        }
    }

    /// Calculated royalty amount in US cents.
    ///
    /// Computed as `gross_revenue_cents × basis_points / 10_000`.
    #[must_use]
    pub fn amount_cents(&self) -> u64 {
        self.gross_revenue_cents
            .saturating_mul(u64::from(self.rate.basis_points))
            / 10_000
    }

    /// Calculated royalty amount in US dollars (floating-point).
    #[must_use]
    pub fn amount_dollars(&self) -> f64 {
        self.amount_cents() as f64 / 100.0
    }
}

// ── RoyaltyStatement ─────────────────────────────────────────────────────────

/// A statement grouping royalty payments for a reporting period.
///
/// The period is expressed as Unix timestamps in seconds.
#[derive(Debug, Default)]
pub struct RoyaltyStatement {
    /// Individual royalty payments included in this statement.
    pub payments: Vec<RoyaltyPayment>,
    /// Start of the reporting period (Unix seconds).
    pub period_start: u64,
    /// End of the reporting period (Unix seconds).
    pub period_end: u64,
}

impl RoyaltyStatement {
    /// Create a new, empty statement for the given period.
    #[must_use]
    pub fn new(period_start: u64, period_end: u64) -> Self {
        Self {
            payments: Vec::new(),
            period_start,
            period_end,
        }
    }

    /// Append a payment to the statement.
    pub fn add(&mut self, payment: RoyaltyPayment) {
        self.payments.push(payment);
    }

    /// Sum of all payment amounts in US cents.
    #[must_use]
    pub fn total_cents(&self) -> u64 {
        self.payments.iter().map(RoyaltyPayment::amount_cents).sum()
    }

    /// Payments filtered by royalty type.
    #[must_use]
    pub fn payments_by_type(&self, rt: &RoyaltyType) -> Vec<&RoyaltyPayment> {
        self.payments
            .iter()
            .filter(|p| &p.royalty_type == rt)
            .collect()
    }

    /// Total royalties owed to a specific rights holder (in cents).
    #[must_use]
    pub fn holder_total(&self, holder: &str) -> u64 {
        self.payments
            .iter()
            .filter(|p| p.rights_holder == holder)
            .map(RoyaltyPayment::amount_cents)
            .sum()
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn rate(bp: u32) -> RoyaltyRate {
        RoyaltyRate::new(bp)
    }

    // ── RoyaltyRate ──

    #[test]
    fn test_rate_pct_500bp() {
        assert!((RoyaltyRate::new(500).pct() - 5.0_f32).abs() < f32::EPSILON);
    }

    #[test]
    fn test_rate_pct_1bp() {
        assert!((RoyaltyRate::new(1).pct() - 0.01_f32).abs() < f32::EPSILON);
    }

    #[test]
    fn test_rate_from_pct_roundtrip() {
        let r = RoyaltyRate::from_pct(7.5);
        assert_eq!(r.basis_points, 750);
    }

    #[test]
    fn test_rate_from_pct_zero() {
        let r = RoyaltyRate::from_pct(0.0);
        assert_eq!(r.basis_points, 0);
    }

    // ── RoyaltyType ──

    #[test]
    fn test_royalty_type_is_publishing_mechanical() {
        assert!(RoyaltyType::Mechanical.is_publishing());
    }

    #[test]
    fn test_royalty_type_is_publishing_master_use() {
        assert!(!RoyaltyType::MasterUse.is_publishing());
    }

    #[test]
    fn test_royalty_type_is_publishing_sync() {
        assert!(RoyaltyType::Synchronisation.is_publishing());
    }

    // ── RoyaltyPayment ──

    #[test]
    fn test_payment_amount_cents_basic() {
        // 10 % of $100.00 = $10.00 = 1000 cents
        let p = RoyaltyPayment::new("ACME", RoyaltyType::Mechanical, rate(1000), 10_000);
        assert_eq!(p.amount_cents(), 1000);
    }

    #[test]
    fn test_payment_amount_cents_zero_rate() {
        let p = RoyaltyPayment::new("ACME", RoyaltyType::Performance, rate(0), 50_000);
        assert_eq!(p.amount_cents(), 0);
    }

    #[test]
    fn test_payment_amount_dollars() {
        // 5 % of $200.00 = $10.00
        let p = RoyaltyPayment::new("Bob", RoyaltyType::Print, rate(500), 20_000);
        assert!((p.amount_dollars() - 10.0_f64).abs() < 1e-9);
    }

    // ── RoyaltyStatement ──

    #[test]
    fn test_statement_total_empty() {
        let s = RoyaltyStatement::new(0, 1_000_000);
        assert_eq!(s.total_cents(), 0);
    }

    #[test]
    fn test_statement_total_multiple_payments() {
        let mut s = RoyaltyStatement::new(0, 1_000_000);
        // 10 % of $100 = $10
        s.add(RoyaltyPayment::new(
            "Alice",
            RoyaltyType::Mechanical,
            rate(1000),
            10_000,
        ));
        // 5 % of $200 = $10
        s.add(RoyaltyPayment::new(
            "Bob",
            RoyaltyType::Performance,
            rate(500),
            20_000,
        ));
        assert_eq!(s.total_cents(), 2000);
    }

    #[test]
    fn test_statement_payments_by_type() {
        let mut s = RoyaltyStatement::new(0, 1_000_000);
        s.add(RoyaltyPayment::new(
            "Alice",
            RoyaltyType::Mechanical,
            rate(1000),
            10_000,
        ));
        s.add(RoyaltyPayment::new(
            "Bob",
            RoyaltyType::Performance,
            rate(500),
            20_000,
        ));
        let mech = s.payments_by_type(&RoyaltyType::Mechanical);
        assert_eq!(mech.len(), 1);
        assert_eq!(mech[0].rights_holder, "Alice");
    }

    #[test]
    fn test_statement_holder_total() {
        let mut s = RoyaltyStatement::new(0, 1_000_000);
        s.add(RoyaltyPayment::new(
            "Alice",
            RoyaltyType::Mechanical,
            rate(1000),
            10_000,
        ));
        s.add(RoyaltyPayment::new(
            "Alice",
            RoyaltyType::Print,
            rate(500),
            10_000,
        ));
        s.add(RoyaltyPayment::new(
            "Bob",
            RoyaltyType::Performance,
            rate(500),
            20_000,
        ));
        // Alice: 1000 + 500 = 1500 cents
        assert_eq!(s.holder_total("Alice"), 1500);
        assert_eq!(s.holder_total("Bob"), 1000);
    }

    #[test]
    fn test_statement_holder_total_unknown() {
        let s = RoyaltyStatement::new(0, 1_000_000);
        assert_eq!(s.holder_total("Unknown"), 0);
    }
}
