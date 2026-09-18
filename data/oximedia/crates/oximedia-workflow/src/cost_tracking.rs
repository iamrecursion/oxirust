//! Workflow cost tracking: per-step costs, budget limits, and cost center allocation.

#![allow(dead_code)]
#![allow(clippy::cast_precision_loss)]

use std::collections::HashMap;

/// Currency used for cost tracking.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Currency {
    /// US Dollars.
    Usd,
    /// Euros.
    Eur,
    /// British Pounds.
    Gbp,
    /// Japanese Yen.
    Jpy,
}

impl Currency {
    /// Return the ISO 4217 currency code.
    #[must_use]
    pub fn code(&self) -> &'static str {
        match self {
            Self::Usd => "USD",
            Self::Eur => "EUR",
            Self::Gbp => "GBP",
            Self::Jpy => "JPY",
        }
    }
}

/// A monetary amount with associated currency.
#[derive(Debug, Clone)]
pub struct Money {
    /// Amount in the smallest unit (e.g., cents for USD).
    pub amount_cents: i64,
    /// Currency of the amount.
    pub currency: Currency,
}

impl Money {
    /// Create a new money value from a decimal amount (e.g., 12.50 USD).
    #[must_use]
    pub fn from_decimal(amount: f64, currency: Currency) -> Self {
        Self {
            amount_cents: (amount * 100.0).round() as i64,
            currency,
        }
    }

    /// Return the decimal representation.
    #[must_use]
    pub fn as_decimal(&self) -> f64 {
        self.amount_cents as f64 / 100.0
    }

    /// Add another money value (must be same currency).
    #[must_use]
    pub fn add(&self, other: &Self) -> Option<Self> {
        if self.currency != other.currency {
            return None;
        }
        Some(Self {
            amount_cents: self.amount_cents + other.amount_cents,
            currency: self.currency.clone(),
        })
    }

    /// Check whether the amount exceeds a budget limit.
    #[must_use]
    pub fn exceeds(&self, budget: &Self) -> bool {
        self.currency == budget.currency && self.amount_cents > budget.amount_cents
    }
}

/// A cost entry for a single workflow step.
#[derive(Debug, Clone)]
pub struct StepCost {
    /// Step identifier.
    pub step_id: String,
    /// Human-readable step name.
    pub step_name: String,
    /// Estimated cost before execution.
    pub estimated: Money,
    /// Actual cost after execution (None if not yet run).
    pub actual: Option<Money>,
    /// Cost center this step is billed to.
    pub cost_center: Option<String>,
}

impl StepCost {
    /// Create a new step cost with an estimate.
    #[must_use]
    pub fn new(step_id: &str, step_name: &str, estimated: Money) -> Self {
        Self {
            step_id: step_id.to_string(),
            step_name: step_name.to_string(),
            estimated,
            actual: None,
            cost_center: None,
        }
    }

    /// Record the actual cost after the step executes.
    pub fn record_actual(&mut self, actual: Money) {
        self.actual = Some(actual);
    }

    /// Assign to a cost center.
    pub fn assign_cost_center(&mut self, center: &str) {
        self.cost_center = Some(center.to_string());
    }

    /// Variance between actual and estimated (positive = over-budget).
    #[must_use]
    pub fn variance_cents(&self) -> Option<i64> {
        self.actual
            .as_ref()
            .map(|a| a.amount_cents - self.estimated.amount_cents)
    }
}

/// Budget limit configuration.
#[derive(Debug, Clone)]
pub struct BudgetLimit {
    /// Maximum allowed spend.
    pub limit: Money,
    /// Warning threshold (spend at which to alert).
    pub warning_at: Money,
    /// Whether the budget has been exceeded.
    pub exceeded: bool,
}

impl BudgetLimit {
    /// Create a new budget limit.
    #[must_use]
    pub fn new(limit: Money, warning_fraction: f64) -> Self {
        let warning_cents = (limit.amount_cents as f64 * warning_fraction).round() as i64;
        let warning_at = Money {
            amount_cents: warning_cents,
            currency: limit.currency.clone(),
        };
        Self {
            limit,
            warning_at,
            exceeded: false,
        }
    }

    /// Check if a given spend amount exceeds or warns the budget.
    pub fn evaluate(&mut self, spent: &Money) -> BudgetEvaluation {
        if spent.exceeds(&self.limit) {
            self.exceeded = true;
            BudgetEvaluation::Exceeded
        } else if spent.exceeds(&self.warning_at) {
            BudgetEvaluation::Warning
        } else {
            BudgetEvaluation::Ok
        }
    }
}

/// Result of a budget evaluation.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum BudgetEvaluation {
    /// Spend is within budget.
    Ok,
    /// Spend is above warning threshold but within budget.
    Warning,
    /// Budget has been exceeded.
    Exceeded,
}

/// Cost center definition.
#[derive(Debug, Clone)]
pub struct CostCenter {
    /// Cost center code.
    pub code: String,
    /// Human-readable description.
    pub description: String,
    /// Optional budget limit for this cost center.
    pub budget: Option<BudgetLimit>,
}

impl CostCenter {
    /// Create a new cost center without a budget.
    #[must_use]
    pub fn new(code: &str, description: &str) -> Self {
        Self {
            code: code.to_string(),
            description: description.to_string(),
            budget: None,
        }
    }

    /// Set the budget for this cost center.
    #[must_use]
    pub fn with_budget(mut self, budget: BudgetLimit) -> Self {
        self.budget = Some(budget);
        self
    }
}

/// Workflow cost ledger: aggregates all step costs for a workflow.
#[derive(Debug, Default)]
pub struct CostLedger {
    /// All step cost entries.
    steps: Vec<StepCost>,
    /// Registered cost centers.
    cost_centers: HashMap<String, CostCenter>,
    /// Default currency for totals.
    currency: Option<Currency>,
}

impl CostLedger {
    /// Create a new empty ledger.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Create a ledger with a default currency.
    #[must_use]
    pub fn with_currency(currency: Currency) -> Self {
        Self {
            currency: Some(currency),
            ..Default::default()
        }
    }

    /// Register a cost center.
    pub fn register_cost_center(&mut self, center: CostCenter) {
        self.cost_centers.insert(center.code.clone(), center);
    }

    /// Add a step cost entry.
    pub fn add_step(&mut self, step: StepCost) {
        self.steps.push(step);
    }

    /// Record actual cost for a step.
    pub fn record_actual(&mut self, step_id: &str, actual: Money) -> bool {
        for step in &mut self.steps {
            if step.step_id == step_id {
                step.record_actual(actual);
                return true;
            }
        }
        false
    }

    /// Total estimated cost (returns None if any currencies mismatch or no steps).
    #[must_use]
    pub fn total_estimated(&self) -> Option<Money> {
        self.sum_money(self.steps.iter().map(|s| &s.estimated))
    }

    /// Total actual cost (steps without actual cost are skipped).
    #[must_use]
    pub fn total_actual(&self) -> Option<Money> {
        let actuals: Vec<&Money> = self
            .steps
            .iter()
            .filter_map(|s| s.actual.as_ref())
            .collect();
        if actuals.is_empty() {
            return None;
        }
        self.sum_money(actuals.into_iter())
    }

    /// Costs allocated to a specific cost center.
    #[must_use]
    pub fn cost_center_total(&self, center_code: &str) -> Option<Money> {
        let relevant: Vec<&Money> = self
            .steps
            .iter()
            .filter(|s| s.cost_center.as_deref() == Some(center_code))
            .filter_map(|s| s.actual.as_ref())
            .collect();
        if relevant.is_empty() {
            return None;
        }
        self.sum_money(relevant.into_iter())
    }

    fn sum_money<'a>(&self, iter: impl Iterator<Item = &'a Money>) -> Option<Money> {
        let items: Vec<&'a Money> = iter.collect();
        if items.is_empty() {
            return None;
        }
        let currency = items[0].currency.clone();
        if items.iter().any(|m| m.currency != currency) {
            return None; // Currency mismatch
        }
        let total_cents: i64 = items.iter().map(|m| m.amount_cents).sum();
        Some(Money {
            amount_cents: total_cents,
            currency,
        })
    }

    /// Number of step entries.
    #[must_use]
    pub fn step_count(&self) -> usize {
        self.steps.len()
    }

    /// Number of registered cost centers.
    #[must_use]
    pub fn cost_center_count(&self) -> usize {
        self.cost_centers.len()
    }

    /// Evaluate total actual spend against a budget limit.
    #[must_use]
    pub fn evaluate_budget(&self, mut limit: BudgetLimit) -> Option<BudgetEvaluation> {
        let actual = self.total_actual()?;
        Some(limit.evaluate(&actual))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn usd(amount: f64) -> Money {
        Money::from_decimal(amount, Currency::Usd)
    }

    fn eur(amount: f64) -> Money {
        Money::from_decimal(amount, Currency::Eur)
    }

    #[test]
    fn test_money_from_decimal() {
        let m = usd(12.50);
        assert_eq!(m.amount_cents, 1250);
        assert!((m.as_decimal() - 12.50).abs() < 0.001);
    }

    #[test]
    fn test_money_add_same_currency() {
        let a = usd(10.00);
        let b = usd(5.25);
        let result = a.add(&b).expect("should succeed in test");
        assert_eq!(result.amount_cents, 1525);
    }

    #[test]
    fn test_money_add_different_currency() {
        let a = usd(10.00);
        let b = eur(5.00);
        assert!(a.add(&b).is_none());
    }

    #[test]
    fn test_money_exceeds() {
        let spent = usd(150.00);
        let budget = usd(100.00);
        assert!(spent.exceeds(&budget));
        let ok = usd(50.00);
        assert!(!ok.exceeds(&budget));
    }

    #[test]
    fn test_currency_code() {
        assert_eq!(Currency::Usd.code(), "USD");
        assert_eq!(Currency::Eur.code(), "EUR");
        assert_eq!(Currency::Gbp.code(), "GBP");
        assert_eq!(Currency::Jpy.code(), "JPY");
    }

    #[test]
    fn test_step_cost_new_and_actual() {
        let mut step = StepCost::new("s1", "Transcode", usd(20.00));
        assert!(step.actual.is_none());
        step.record_actual(usd(22.50));
        assert!(step.actual.is_some());
        assert_eq!(step.variance_cents(), Some(250));
    }

    #[test]
    fn test_step_cost_assign_center() {
        let mut step = StepCost::new("s1", "QC", usd(5.00));
        step.assign_cost_center("CC-001");
        assert_eq!(step.cost_center.as_deref(), Some("CC-001"));
    }

    #[test]
    fn test_budget_limit_ok() {
        let limit = BudgetLimit::new(usd(100.00), 0.8);
        let mut bl = limit;
        assert_eq!(bl.evaluate(&usd(50.00)), BudgetEvaluation::Ok);
    }

    #[test]
    fn test_budget_limit_warning() {
        let mut bl = BudgetLimit::new(usd(100.00), 0.8);
        assert_eq!(bl.evaluate(&usd(85.00)), BudgetEvaluation::Warning);
    }

    #[test]
    fn test_budget_limit_exceeded() {
        let mut bl = BudgetLimit::new(usd(100.00), 0.8);
        assert_eq!(bl.evaluate(&usd(110.00)), BudgetEvaluation::Exceeded);
        assert!(bl.exceeded);
    }

    #[test]
    fn test_ledger_total_estimated() {
        let mut ledger = CostLedger::new();
        ledger.add_step(StepCost::new("s1", "Step 1", usd(10.00)));
        ledger.add_step(StepCost::new("s2", "Step 2", usd(20.00)));
        let total = ledger.total_estimated().expect("should succeed in test");
        assert_eq!(total.amount_cents, 3000);
    }

    #[test]
    fn test_ledger_total_actual_none_if_no_actuals() {
        let mut ledger = CostLedger::new();
        ledger.add_step(StepCost::new("s1", "Step 1", usd(10.00)));
        assert!(ledger.total_actual().is_none());
    }

    #[test]
    fn test_ledger_record_actual() {
        let mut ledger = CostLedger::new();
        ledger.add_step(StepCost::new("s1", "Step 1", usd(10.00)));
        assert!(ledger.record_actual("s1", usd(12.00)));
        let total = ledger.total_actual().expect("should succeed in test");
        assert_eq!(total.amount_cents, 1200);
    }

    #[test]
    fn test_ledger_record_actual_missing() {
        let mut ledger = CostLedger::new();
        assert!(!ledger.record_actual("nonexistent", usd(5.00)));
    }

    #[test]
    fn test_ledger_cost_center_registration() {
        let mut ledger = CostLedger::new();
        ledger.register_cost_center(CostCenter::new("CC-001", "Production"));
        assert_eq!(ledger.cost_center_count(), 1);
    }

    #[test]
    fn test_cost_center_total() {
        let mut ledger = CostLedger::new();
        let mut s1 = StepCost::new("s1", "Encode", usd(15.00));
        s1.assign_cost_center("CC-001");
        s1.record_actual(usd(16.00));
        let mut s2 = StepCost::new("s2", "QC", usd(5.00));
        s2.assign_cost_center("CC-002");
        s2.record_actual(usd(5.50));
        ledger.add_step(s1);
        ledger.add_step(s2);
        let cc1_total = ledger
            .cost_center_total("CC-001")
            .expect("should succeed in test");
        assert_eq!(cc1_total.amount_cents, 1600);
    }

    #[test]
    fn test_ledger_step_count() {
        let mut ledger = CostLedger::new();
        ledger.add_step(StepCost::new("s1", "Step 1", usd(10.00)));
        ledger.add_step(StepCost::new("s2", "Step 2", usd(20.00)));
        ledger.add_step(StepCost::new("s3", "Step 3", usd(30.00)));
        assert_eq!(ledger.step_count(), 3);
    }

    #[test]
    fn test_evaluate_budget_exceeded() {
        let mut ledger = CostLedger::new();
        let mut step = StepCost::new("s1", "Expensive", usd(200.00));
        step.record_actual(usd(200.00));
        ledger.add_step(step);
        let result = ledger.evaluate_budget(BudgetLimit::new(usd(100.00), 0.8));
        assert_eq!(result, Some(BudgetEvaluation::Exceeded));
    }

    #[test]
    fn test_money_as_decimal_roundtrip() {
        let m = Money::from_decimal(99.99, Currency::Gbp);
        assert!((m.as_decimal() - 99.99).abs() < 0.001);
    }
}
