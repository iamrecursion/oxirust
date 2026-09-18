// Copyright 2024 OxiMedia Project
// Licensed under the Apache License, Version 2.0

//! Budget and cost management.

use crate::error::{Error, Result};
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Budget alert type
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum AlertType {
    /// Warning threshold reached
    Warning,
    /// Critical threshold reached
    Critical,
    /// Budget exceeded
    Exceeded,
}

/// Budget alert
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BudgetAlert {
    /// Project ID
    pub project_id: String,
    /// Alert type
    pub alert_type: AlertType,
    /// Message
    pub message: String,
    /// Timestamp
    pub timestamp: DateTime<Utc>,
}

/// Project budget
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProjectBudget {
    /// Project ID
    pub project_id: String,
    /// Total budget
    pub total: f64,
    /// Spent amount
    pub spent: f64,
    /// Reserved amount
    pub reserved: f64,
    /// Warning threshold (0.0 to 1.0)
    pub warning_threshold: f64,
    /// Critical threshold (0.0 to 1.0)
    pub critical_threshold: f64,
    /// Alerts
    pub alerts: Vec<BudgetAlert>,
}

impl ProjectBudget {
    /// Create a new project budget
    #[must_use]
    pub fn new(project_id: String, total: f64) -> Self {
        Self {
            project_id,
            total,
            spent: 0.0,
            reserved: 0.0,
            warning_threshold: 0.8,
            critical_threshold: 0.95,
            alerts: Vec::new(),
        }
    }

    /// Get available budget
    #[must_use]
    pub fn available(&self) -> f64 {
        (self.total - self.spent - self.reserved).max(0.0)
    }

    /// Get utilization (0.0 to 1.0)
    #[must_use]
    pub fn utilization(&self) -> f64 {
        (self.spent + self.reserved) / self.total
    }

    /// Reserve amount
    pub fn reserve(&mut self, amount: f64) -> Result<()> {
        if self.available() < amount {
            return Err(Error::BudgetExceeded {
                allocated: self.total,
                spent: self.spent + self.reserved + amount,
            });
        }

        self.reserved += amount;
        self.check_thresholds();

        Ok(())
    }

    /// Spend reserved amount
    pub fn spend_reserved(&mut self, amount: f64) -> Result<()> {
        if self.reserved < amount {
            return Err(Error::Other("Not enough reserved funds".to_string()));
        }

        self.reserved -= amount;
        self.spent += amount;
        self.check_thresholds();

        Ok(())
    }

    /// Release reserved amount
    pub fn release_reserved(&mut self, amount: f64) {
        self.reserved -= amount.min(self.reserved);
    }

    /// Check and create alerts
    fn check_thresholds(&mut self) {
        let util = self.utilization();

        if util >= 1.0 {
            self.add_alert(AlertType::Exceeded, "Budget exceeded".to_string());
        } else if util >= self.critical_threshold {
            self.add_alert(
                AlertType::Critical,
                "Critical threshold reached".to_string(),
            );
        } else if util >= self.warning_threshold {
            self.add_alert(AlertType::Warning, "Warning threshold reached".to_string());
        }
    }

    fn add_alert(&mut self, alert_type: AlertType, message: String) {
        // Don't add duplicate alerts
        if self.alerts.iter().any(|a| a.alert_type == alert_type) {
            return;
        }

        let alert = BudgetAlert {
            project_id: self.project_id.clone(),
            alert_type,
            message,
            timestamp: Utc::now(),
        };

        self.alerts.push(alert);
    }
}

/// Budget tracker
pub struct BudgetTracker {
    budgets: HashMap<String, ProjectBudget>,
}

impl BudgetTracker {
    /// Create a new budget tracker
    #[must_use]
    pub fn new() -> Self {
        Self {
            budgets: HashMap::new(),
        }
    }

    /// Create budget for project
    pub fn create_budget(&mut self, project_id: String, total: f64) {
        let budget = ProjectBudget::new(project_id.clone(), total);
        self.budgets.insert(project_id, budget);
    }

    /// Get budget for project
    #[must_use]
    pub fn get_budget(&self, project_id: &str) -> Option<&ProjectBudget> {
        self.budgets.get(project_id)
    }

    /// Get mutable budget for project
    pub fn get_budget_mut(&mut self, project_id: &str) -> Option<&mut ProjectBudget> {
        self.budgets.get_mut(project_id)
    }

    /// Reserve funds
    pub fn reserve(&mut self, project_id: &str, amount: f64) -> Result<()> {
        let budget = self.budgets.get_mut(project_id).ok_or_else(|| {
            Error::Configuration(format!("Budget not found for project: {project_id}"))
        })?;

        budget.reserve(amount)
    }

    /// Spend reserved funds
    pub fn spend(&mut self, project_id: &str, amount: f64) -> Result<()> {
        let budget = self.budgets.get_mut(project_id).ok_or_else(|| {
            Error::Configuration(format!("Budget not found for project: {project_id}"))
        })?;

        budget.spend_reserved(amount)
    }

    /// Get all alerts
    #[must_use]
    pub fn get_all_alerts(&self) -> Vec<BudgetAlert> {
        self.budgets
            .values()
            .flat_map(|b| b.alerts.clone())
            .collect()
    }
}

impl Default for BudgetTracker {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_project_budget_creation() {
        let budget = ProjectBudget::new("project1".to_string(), 1000.0);
        assert_eq!(budget.total, 1000.0);
        assert_eq!(budget.available(), 1000.0);
    }

    #[test]
    fn test_reserve_funds() -> Result<()> {
        let mut budget = ProjectBudget::new("project1".to_string(), 1000.0);

        budget.reserve(100.0)?;
        assert_eq!(budget.reserved, 100.0);
        assert_eq!(budget.available(), 900.0);

        Ok(())
    }

    #[test]
    fn test_spend_reserved() -> Result<()> {
        let mut budget = ProjectBudget::new("project1".to_string(), 1000.0);

        budget.reserve(100.0)?;
        budget.spend_reserved(50.0)?;

        assert_eq!(budget.spent, 50.0);
        assert_eq!(budget.reserved, 50.0);

        Ok(())
    }

    #[test]
    fn test_budget_exceeded() {
        let mut budget = ProjectBudget::new("project1".to_string(), 1000.0);

        let result = budget.reserve(1500.0);
        assert!(result.is_err());
    }

    #[test]
    fn test_threshold_alerts() -> Result<()> {
        let mut budget = ProjectBudget::new("project1".to_string(), 1000.0);

        // Reserve 85% - should trigger warning
        budget.reserve(850.0)?;
        assert!(!budget.alerts.is_empty());

        Ok(())
    }

    #[test]
    fn test_budget_tracker() -> Result<()> {
        let mut tracker = BudgetTracker::new();

        tracker.create_budget("project1".to_string(), 1000.0);
        tracker.reserve("project1", 100.0)?;

        let budget = tracker
            .get_budget("project1")
            .expect("should succeed in test");
        assert_eq!(budget.reserved, 100.0);

        Ok(())
    }
}
