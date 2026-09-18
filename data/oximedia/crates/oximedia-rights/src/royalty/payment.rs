//! Royalty payment tracking

#[cfg(not(target_arch = "wasm32"))]
use crate::database::RightsDatabase;
use crate::Result;
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use uuid::Uuid;

/// Payment status
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum PaymentStatus {
    /// Pending payment
    Pending,
    /// Paid
    Paid,
    /// Cancelled
    Cancelled,
}

impl PaymentStatus {
    /// Convert to string
    pub fn as_str(&self) -> &str {
        match self {
            PaymentStatus::Pending => "pending",
            PaymentStatus::Paid => "paid",
            PaymentStatus::Cancelled => "cancelled",
        }
    }

    /// Parse from string
    #[allow(clippy::should_implement_trait)]
    pub fn from_str(s: &str) -> Self {
        match s {
            "pending" => PaymentStatus::Pending,
            "paid" => PaymentStatus::Paid,
            "cancelled" => PaymentStatus::Cancelled,
            _ => PaymentStatus::Pending,
        }
    }
}

/// Royalty payment
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RoyaltyPayment {
    /// Unique identifier
    pub id: String,
    /// Grant ID
    pub grant_id: String,
    /// Owner ID
    pub owner_id: String,
    /// Amount
    pub amount: f64,
    /// Currency
    pub currency: String,
    /// Payment period start
    pub period_start: DateTime<Utc>,
    /// Payment period end
    pub period_end: DateTime<Utc>,
    /// Status
    pub status: PaymentStatus,
    /// Payment date
    pub payment_date: Option<DateTime<Utc>>,
}

impl RoyaltyPayment {
    /// Create a new royalty payment
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        grant_id: impl Into<String>,
        owner_id: impl Into<String>,
        amount: f64,
        currency: impl Into<String>,
        period_start: DateTime<Utc>,
        period_end: DateTime<Utc>,
    ) -> Self {
        Self {
            id: Uuid::new_v4().to_string(),
            grant_id: grant_id.into(),
            owner_id: owner_id.into(),
            amount,
            currency: currency.into(),
            period_start,
            period_end,
            status: PaymentStatus::Pending,
            payment_date: None,
        }
    }

    /// Mark as paid
    pub fn mark_paid(&mut self) {
        self.status = PaymentStatus::Paid;
        self.payment_date = Some(Utc::now());
    }

    #[cfg(not(target_arch = "wasm32"))]
    /// Save to database
    pub async fn save(&self, db: &RightsDatabase) -> Result<()> {
        db.pool()
            .execute(
                r"
            INSERT INTO royalty_payments
            (id, grant_id, owner_id, amount, currency, payment_period_start, payment_period_end,
             status, payment_date, calculation_data_json, created_at, updated_at)
            VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12)
            ON CONFLICT(id) DO UPDATE SET
                status = excluded.status,
                payment_date = excluded.payment_date,
                updated_at = excluded.updated_at
            ",
                &[
                    &self.id,
                    &self.grant_id,
                    &self.owner_id,
                    &self.amount,
                    &self.currency,
                    &self.period_start.to_rfc3339(),
                    &self.period_end.to_rfc3339(),
                    &self.status.as_str(),
                    &self.payment_date.map(|d| d.to_rfc3339()),
                    &"{}",
                    &Utc::now().to_rfc3339(),
                    &Utc::now().to_rfc3339(),
                ],
            )
            .await?;

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_payment_creation() {
        let now = Utc::now();
        let payment = RoyaltyPayment::new(
            "grant1",
            "owner1",
            100.0,
            "USD",
            now,
            now + chrono::Duration::days(30),
        );

        assert_eq!(payment.amount, 100.0);
        assert_eq!(payment.status, PaymentStatus::Pending);
    }
}
