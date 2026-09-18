//! Royalty reporting

use serde::{Deserialize, Serialize};

/// Royalty report
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RoyaltyReport {
    /// Owner ID
    pub owner_id: String,
    /// Total amount
    pub total_amount: f64,
    /// Currency
    pub currency: String,
    /// Number of payments
    pub payment_count: u32,
}

impl RoyaltyReport {
    /// Create a new royalty report
    pub fn new(owner_id: impl Into<String>) -> Self {
        Self {
            owner_id: owner_id.into(),
            total_amount: 0.0,
            currency: "USD".to_string(),
            payment_count: 0,
        }
    }
}
