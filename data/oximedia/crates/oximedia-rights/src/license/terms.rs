//! License terms and conditions

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Payment terms
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PaymentTerms {
    /// Amount
    pub amount: f64,
    /// Currency code (ISO 4217)
    pub currency: String,
    /// Payment schedule
    pub schedule: PaymentSchedule,
    /// Payment method
    pub method: Option<String>,
}

/// Payment schedule
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum PaymentSchedule {
    /// One-time payment
    OneTime,
    /// Annual payment
    Annual,
    /// Monthly payment
    Monthly,
    /// Per-use payment
    PerUse,
    /// Custom schedule
    Custom(String),
}

/// License terms and conditions
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LicenseTerms {
    /// Payment terms
    pub payment: Option<PaymentTerms>,
    /// Attribution requirements
    pub attribution: Option<String>,
    /// Warranty disclaimer
    pub warranty_disclaimer: bool,
    /// Liability limit
    pub liability_limit: Option<f64>,
    /// Indemnification clause
    pub indemnification: bool,
    /// Termination conditions
    pub termination_conditions: Vec<String>,
    /// Additional terms
    pub additional_terms: HashMap<String, String>,
}

impl LicenseTerms {
    /// Create new license terms
    pub fn new() -> Self {
        Self {
            payment: None,
            attribution: None,
            warranty_disclaimer: true,
            liability_limit: None,
            indemnification: false,
            termination_conditions: vec![],
            additional_terms: HashMap::new(),
        }
    }

    /// Set payment terms
    pub fn with_payment(mut self, payment: PaymentTerms) -> Self {
        self.payment = Some(payment);
        self
    }

    /// Set attribution requirement
    pub fn with_attribution(mut self, attribution: impl Into<String>) -> Self {
        self.attribution = Some(attribution.into());
        self
    }

    /// Enable indemnification
    pub fn with_indemnification(mut self) -> Self {
        self.indemnification = true;
        self
    }

    /// Set liability limit
    pub fn with_liability_limit(mut self, limit: f64) -> Self {
        self.liability_limit = Some(limit);
        self
    }

    /// Add termination condition
    pub fn add_termination_condition(mut self, condition: impl Into<String>) -> Self {
        self.termination_conditions.push(condition.into());
        self
    }

    /// Add additional term
    pub fn add_term(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        self.additional_terms.insert(key.into(), value.into());
        self
    }
}

impl Default for LicenseTerms {
    fn default() -> Self {
        Self::new()
    }
}

impl PaymentTerms {
    /// Create new payment terms
    pub fn new(amount: f64, currency: impl Into<String>, schedule: PaymentSchedule) -> Self {
        Self {
            amount,
            currency: currency.into(),
            schedule,
            method: None,
        }
    }

    /// Set payment method
    pub fn with_method(mut self, method: impl Into<String>) -> Self {
        self.method = Some(method.into());
        self
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_license_terms_builder() {
        let terms = LicenseTerms::new()
            .with_attribution("Copyright (c) 2024")
            .with_indemnification()
            .with_liability_limit(10000.0)
            .add_termination_condition("Breach of terms");

        assert_eq!(terms.attribution, Some("Copyright (c) 2024".to_string()));
        assert!(terms.indemnification);
        assert_eq!(terms.liability_limit, Some(10000.0));
        assert_eq!(terms.termination_conditions.len(), 1);
    }

    #[test]
    fn test_payment_terms() {
        let payment =
            PaymentTerms::new(1000.0, "USD", PaymentSchedule::OneTime).with_method("Credit Card");

        assert_eq!(payment.amount, 1000.0);
        assert_eq!(payment.currency, "USD");
        assert_eq!(payment.method, Some("Credit Card".to_string()));
    }

    #[test]
    fn test_additional_terms() {
        let terms = LicenseTerms::new().add_term("custom_clause", "Some custom clause");

        assert_eq!(
            terms.additional_terms.get("custom_clause"),
            Some(&"Some custom clause".to_string())
        );
    }
}
