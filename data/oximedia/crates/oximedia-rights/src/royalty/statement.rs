//! Royalty statement generation and period tracking
//!
//! Provides period-based royalty statements with full deduction tracking,
//! payment scheduling, and advance recoupment accounting.

use serde::{Deserialize, Serialize};

/// Types of deductions applied to gross revenue before royalty calculation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum DeductionType {
    /// Distribution platform fee (e.g. Spotify takes ~30%)
    DistributionFee,
    /// Currency conversion fee
    ConversionFee,
    /// Withholding tax deducted at source
    WithholdingTax,
    /// Mechanical licensing fee
    MechanicalFee,
    /// Neighbouring rights collection fee
    CollectionSocietyFee,
    /// Packaging deduction (physical media)
    PackagingDeduction,
    /// Returns and refunds
    Returns,
    /// Custom deduction type
    Custom(String),
}

impl DeductionType {
    /// Human-readable label for this deduction type
    pub fn label(&self) -> String {
        match self {
            DeductionType::DistributionFee => "Distribution Fee".to_string(),
            DeductionType::ConversionFee => "Currency Conversion Fee".to_string(),
            DeductionType::WithholdingTax => "Withholding Tax".to_string(),
            DeductionType::MechanicalFee => "Mechanical Fee".to_string(),
            DeductionType::CollectionSocietyFee => "Collection Society Fee".to_string(),
            DeductionType::PackagingDeduction => "Packaging Deduction".to_string(),
            DeductionType::Returns => "Returns & Refunds".to_string(),
            DeductionType::Custom(name) => name.clone(),
        }
    }
}

/// A single deduction item applied to gross revenue.
/// All amounts are in USD cents to avoid floating-point accumulation errors.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Deduction {
    /// Type of this deduction
    pub deduction_type: DeductionType,
    /// Amount in USD cents (positive = reduction from gross)
    pub amount_cents: i64,
    /// Optional percentage this represents of gross revenue
    pub percentage: Option<f64>,
    /// Description for display
    pub description: String,
}

impl Deduction {
    /// Create a deduction with a fixed amount in cents
    pub fn fixed(deduction_type: DeductionType, amount_cents: i64, description: &str) -> Self {
        Self {
            deduction_type,
            amount_cents,
            percentage: None,
            description: description.to_string(),
        }
    }

    /// Create a deduction as a percentage of gross revenue
    ///
    /// # Arguments
    /// * `gross_cents` - Gross revenue in USD cents
    /// * `percentage` - Percentage to deduct (0.0–100.0)
    pub fn percentage(
        deduction_type: DeductionType,
        gross_cents: i64,
        percentage: f64,
        description: &str,
    ) -> Self {
        let amount_cents = (gross_cents as f64 * percentage / 100.0).round() as i64;
        Self {
            deduction_type,
            amount_cents,
            percentage: Some(percentage),
            description: description.to_string(),
        }
    }
}

/// Payment frequency for scheduled royalty distributions
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum PaymentSchedule {
    /// Monthly payments
    Monthly,
    /// Quarterly payments (every 3 months)
    Quarterly,
    /// Bi-annual payments (every 6 months)
    BiAnnual,
    /// Annual payments
    Annual,
    /// On-demand / as-earned
    OnDemand,
}

impl PaymentSchedule {
    /// Number of days in one payment period
    pub fn period_days(&self) -> u32 {
        match self {
            PaymentSchedule::Monthly => 30,
            PaymentSchedule::Quarterly => 91,
            PaymentSchedule::BiAnnual => 182,
            PaymentSchedule::Annual => 365,
            PaymentSchedule::OnDemand => 0,
        }
    }

    /// Human-readable label
    pub fn label(&self) -> &'static str {
        match self {
            PaymentSchedule::Monthly => "Monthly",
            PaymentSchedule::Quarterly => "Quarterly",
            PaymentSchedule::BiAnnual => "Bi-Annual",
            PaymentSchedule::Annual => "Annual",
            PaymentSchedule::OnDemand => "On-Demand",
        }
    }
}

/// A complete royalty statement for one accounting period.
///
/// All monetary amounts are stored in USD cents to avoid floating-point
/// precision issues. Use the helper methods to get dollar values.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RoyaltyStatement {
    /// Statement identifier
    pub statement_id: String,
    /// Asset or work identifier this statement covers
    pub asset_id: String,
    /// Period start as Unix timestamp (seconds)
    pub period_start: u64,
    /// Period end as Unix timestamp (seconds)
    pub period_end: u64,
    /// Gross revenue in USD cents before any deductions
    pub gross_revenue_cents: i64,
    /// All deductions applied before royalty calculation
    pub deductions: Vec<Deduction>,
    /// Net revenue in cents after all deductions
    pub net_revenue_cents: i64,
    /// Per-party royalty amounts: (party_name, cents)
    pub royalty_amounts: Vec<(String, i64)>,
    /// Whether the advance has been fully recouped as of this statement
    pub is_recouped: bool,
    /// Remaining unrecouped advance balance in cents (0 if fully recouped)
    pub advance_balance_cents: i64,
    /// Payment schedule for this statement
    pub payment_schedule: PaymentSchedule,
    /// Any notes or memo for this statement
    pub notes: String,
}

impl RoyaltyStatement {
    /// Create a new statement with the given parameters.
    ///
    /// Automatically computes net_revenue_cents from gross minus deductions.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        asset_id: impl Into<String>,
        period_start: u64,
        period_end: u64,
        gross_revenue_cents: i64,
        deductions: Vec<Deduction>,
        royalty_amounts: Vec<(String, i64)>,
        advance_balance_cents: i64,
        payment_schedule: PaymentSchedule,
    ) -> Self {
        let total_deductions: i64 = deductions.iter().map(|d| d.amount_cents).sum();
        let net_revenue_cents = gross_revenue_cents - total_deductions;
        let is_recouped = advance_balance_cents <= 0;

        Self {
            statement_id: uuid::Uuid::new_v4().to_string(),
            asset_id: asset_id.into(),
            period_start,
            period_end,
            gross_revenue_cents,
            deductions,
            net_revenue_cents,
            royalty_amounts,
            is_recouped,
            advance_balance_cents: advance_balance_cents.max(0),
            payment_schedule,
            notes: String::new(),
        }
    }

    /// Gross revenue in USD dollars
    pub fn gross_revenue_usd(&self) -> f64 {
        self.gross_revenue_cents as f64 / 100.0
    }

    /// Net revenue in USD dollars
    pub fn net_revenue_usd(&self) -> f64 {
        self.net_revenue_cents as f64 / 100.0
    }

    /// Total royalties payable to all parties in USD dollars
    pub fn total_royalties_usd(&self) -> f64 {
        let total_cents: i64 = self.royalty_amounts.iter().map(|(_, c)| c).sum();
        total_cents as f64 / 100.0
    }

    /// Royalty amount for a specific party, in USD dollars
    pub fn party_royalty_usd(&self, party: &str) -> f64 {
        self.royalty_amounts
            .iter()
            .find(|(p, _)| p == party)
            .map_or(0.0, |(_, cents)| *cents as f64 / 100.0)
    }

    /// Total deductions in USD dollars
    pub fn total_deductions_usd(&self) -> f64 {
        let total_cents: i64 = self.deductions.iter().map(|d| d.amount_cents).sum();
        total_cents as f64 / 100.0
    }

    /// Remaining advance balance in USD dollars
    pub fn advance_balance_usd(&self) -> f64 {
        self.advance_balance_cents as f64 / 100.0
    }

    /// Add a note to this statement
    pub fn with_note(mut self, note: impl Into<String>) -> Self {
        self.notes = note.into();
        self
    }

    /// Format a summary of this statement for display
    pub fn summary(&self) -> String {
        let period_start_secs = self.period_start as i64;
        let period_end_secs = self.period_end as i64;
        let start_dt = chrono::DateTime::from_timestamp(period_start_secs, 0)
            .unwrap_or_default()
            .format("%Y-%m-%d");
        let end_dt = chrono::DateTime::from_timestamp(period_end_secs, 0)
            .unwrap_or_default()
            .format("%Y-%m-%d");

        format!(
            "Statement {} | Period: {} to {} | Gross: ${:.2} | Net: ${:.2} | Royalties: ${:.2} | Recouped: {}",
            &self.statement_id[..8],
            start_dt,
            end_dt,
            self.gross_revenue_usd(),
            self.net_revenue_usd(),
            self.total_royalties_usd(),
            if self.is_recouped { "Yes" } else { "No" }
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_statement(gross_cents: i64, deductions: Vec<Deduction>) -> RoyaltyStatement {
        let royalties = vec![
            ("Artist".to_string(), 1000i64),
            ("Publisher".to_string(), 500i64),
        ];
        RoyaltyStatement::new(
            "asset-001",
            1_700_000_000u64,
            1_702_592_000u64,
            gross_cents,
            deductions,
            royalties,
            0,
            PaymentSchedule::Quarterly,
        )
    }

    #[test]
    fn test_statement_net_revenue() {
        let deductions = vec![Deduction::percentage(
            DeductionType::DistributionFee,
            10_000,
            30.0,
            "Platform fee",
        )];
        let stmt = make_statement(10_000, deductions);
        // 10000 gross - 3000 distribution fee = 7000 net
        assert_eq!(stmt.net_revenue_cents, 7_000);
        assert!((stmt.net_revenue_usd() - 70.0).abs() < 0.01);
    }

    #[test]
    fn test_statement_gross_usd() {
        let stmt = make_statement(1_000_000, vec![]);
        assert!((stmt.gross_revenue_usd() - 10_000.0).abs() < 0.01);
    }

    #[test]
    fn test_statement_total_royalties() {
        let stmt = make_statement(100_000, vec![]);
        // Artist: 1000 cents + Publisher: 500 cents = 1500 cents = $15.00
        assert!((stmt.total_royalties_usd() - 15.0).abs() < 0.01);
    }

    #[test]
    fn test_party_royalty() {
        let stmt = make_statement(100_000, vec![]);
        assert!((stmt.party_royalty_usd("Artist") - 10.0).abs() < 0.01);
        assert!((stmt.party_royalty_usd("Publisher") - 5.0).abs() < 0.01);
        assert!((stmt.party_royalty_usd("Unknown") - 0.0).abs() < 0.01);
    }

    #[test]
    fn test_statement_is_recouped() {
        let stmt = make_statement(100_000, vec![]);
        // advance_balance = 0 means recouped
        assert!(stmt.is_recouped);
        assert_eq!(stmt.advance_balance_cents, 0);
    }

    #[test]
    fn test_statement_not_recouped() {
        let royalties = vec![("Artist".to_string(), 5000i64)];
        let stmt = RoyaltyStatement::new(
            "asset-002",
            1_700_000_000u64,
            1_702_592_000u64,
            50_000,
            vec![],
            royalties,
            250_000, // $2500 still owed
            PaymentSchedule::Monthly,
        );
        assert!(!stmt.is_recouped);
        assert!((stmt.advance_balance_usd() - 2500.0).abs() < 0.01);
    }

    #[test]
    fn test_deduction_fixed() {
        let d = Deduction::fixed(DeductionType::WithholdingTax, 1500, "10% WHT");
        assert_eq!(d.amount_cents, 1500);
        assert!(d.percentage.is_none());
    }

    #[test]
    fn test_deduction_percentage() {
        let d = Deduction::percentage(DeductionType::ConversionFee, 20_000, 2.5, "FX conversion");
        // 2.5% of 20000 = 500
        assert_eq!(d.amount_cents, 500);
        assert_eq!(d.percentage, Some(2.5));
    }

    #[test]
    fn test_payment_schedule_labels() {
        assert_eq!(PaymentSchedule::Monthly.label(), "Monthly");
        assert_eq!(PaymentSchedule::Quarterly.label(), "Quarterly");
        assert_eq!(PaymentSchedule::BiAnnual.label(), "Bi-Annual");
        assert_eq!(PaymentSchedule::Annual.label(), "Annual");
    }

    #[test]
    fn test_payment_schedule_period_days() {
        assert_eq!(PaymentSchedule::Monthly.period_days(), 30);
        assert_eq!(PaymentSchedule::Quarterly.period_days(), 91);
        assert_eq!(PaymentSchedule::BiAnnual.period_days(), 182);
        assert_eq!(PaymentSchedule::Annual.period_days(), 365);
        assert_eq!(PaymentSchedule::OnDemand.period_days(), 0);
    }

    #[test]
    fn test_statement_summary() {
        let stmt = make_statement(100_000, vec![]);
        let summary = stmt.summary();
        assert!(summary.contains("Gross:"));
        assert!(summary.contains("Net:"));
        assert!(summary.contains("Royalties:"));
    }

    #[test]
    fn test_multiple_deductions() {
        let deductions = vec![
            Deduction::percentage(DeductionType::DistributionFee, 10_000, 30.0, "Platform 30%"),
            Deduction::fixed(DeductionType::WithholdingTax, 300, "WHT"),
            Deduction::percentage(DeductionType::ConversionFee, 10_000, 1.0, "FX fee"),
        ];
        let stmt = make_statement(10_000, deductions);
        // 3000 + 300 + 100 = 3400 total deductions
        // net = 10000 - 3400 = 6600
        assert_eq!(stmt.net_revenue_cents, 6_600);
        assert!((stmt.total_deductions_usd() - 34.0).abs() < 0.01);
    }
}
