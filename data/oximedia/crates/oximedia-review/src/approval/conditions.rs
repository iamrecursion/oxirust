//! Conditional approval handling.

use serde::{Deserialize, Serialize};

/// Approval condition.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ApprovalCondition {
    /// Condition ID.
    pub id: String,
    /// Condition description.
    pub description: String,
    /// Whether condition is met.
    pub met: bool,
    /// Priority level.
    pub priority: ConditionPriority,
}

/// Priority level for conditions.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum ConditionPriority {
    /// Low priority - optional.
    Low,
    /// Normal priority - should be addressed.
    Normal,
    /// High priority - must be addressed.
    High,
    /// Critical priority - blocking.
    Critical,
}

impl ApprovalCondition {
    /// Create a new condition.
    #[must_use]
    pub fn new(id: impl Into<String>, description: impl Into<String>) -> Self {
        Self {
            id: id.into(),
            description: description.into(),
            met: false,
            priority: ConditionPriority::Normal,
        }
    }

    /// Set the priority.
    #[must_use]
    pub fn with_priority(mut self, priority: ConditionPriority) -> Self {
        self.priority = priority;
        self
    }

    /// Mark condition as met.
    pub fn mark_met(&mut self) {
        self.met = true;
    }

    /// Mark condition as unmet.
    pub fn mark_unmet(&mut self) {
        self.met = false;
    }

    /// Check if condition is blocking.
    #[must_use]
    pub fn is_blocking(&self) -> bool {
        self.priority == ConditionPriority::Critical && !self.met
    }
}

/// Condition set for an approval.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConditionSet {
    /// All conditions.
    pub conditions: Vec<ApprovalCondition>,
}

impl ConditionSet {
    /// Create a new condition set.
    #[must_use]
    pub fn new() -> Self {
        Self {
            conditions: Vec::new(),
        }
    }

    /// Add a condition.
    pub fn add_condition(&mut self, condition: ApprovalCondition) {
        self.conditions.push(condition);
    }

    /// Get unmet conditions.
    #[must_use]
    pub fn unmet_conditions(&self) -> Vec<&ApprovalCondition> {
        self.conditions.iter().filter(|c| !c.met).collect()
    }

    /// Get blocking conditions.
    #[must_use]
    pub fn blocking_conditions(&self) -> Vec<&ApprovalCondition> {
        self.conditions.iter().filter(|c| c.is_blocking()).collect()
    }

    /// Check if all conditions are met.
    #[must_use]
    pub fn all_met(&self) -> bool {
        self.conditions.iter().all(|c| c.met)
    }

    /// Check if any blocking conditions exist.
    #[must_use]
    pub fn has_blocking_conditions(&self) -> bool {
        self.conditions.iter().any(ApprovalCondition::is_blocking)
    }

    /// Get completion percentage.
    #[must_use]
    pub fn completion_percentage(&self) -> f64 {
        if self.conditions.is_empty() {
            return 100.0;
        }

        let met_count = self.conditions.iter().filter(|c| c.met).count();
        (met_count as f64 / self.conditions.len() as f64) * 100.0
    }
}

impl Default for ConditionSet {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_condition_creation() {
        let condition = ApprovalCondition::new("c1", "Fix audio");
        assert_eq!(condition.id, "c1");
        assert!(!condition.met);
        assert_eq!(condition.priority, ConditionPriority::Normal);
    }

    #[test]
    fn test_condition_with_priority() {
        let condition =
            ApprovalCondition::new("c1", "Fix audio").with_priority(ConditionPriority::High);

        assert_eq!(condition.priority, ConditionPriority::High);
    }

    #[test]
    fn test_condition_mark_met() {
        let mut condition = ApprovalCondition::new("c1", "Fix audio");
        assert!(!condition.met);

        condition.mark_met();
        assert!(condition.met);

        condition.mark_unmet();
        assert!(!condition.met);
    }

    #[test]
    fn test_condition_is_blocking() {
        let mut condition =
            ApprovalCondition::new("c1", "Fix audio").with_priority(ConditionPriority::Critical);

        assert!(condition.is_blocking());

        condition.mark_met();
        assert!(!condition.is_blocking());
    }

    #[test]
    fn test_condition_set() {
        let mut set = ConditionSet::new();

        set.add_condition(ApprovalCondition::new("c1", "Fix audio"));
        set.add_condition(
            ApprovalCondition::new("c2", "Fix video").with_priority(ConditionPriority::High),
        );

        assert_eq!(set.conditions.len(), 2);
        assert_eq!(set.unmet_conditions().len(), 2);
        assert!(!set.all_met());

        set.conditions[0].mark_met();
        assert_eq!(set.unmet_conditions().len(), 1);
        assert!((set.completion_percentage() - 50.0).abs() < 0.001);

        set.conditions[1].mark_met();
        assert!(set.all_met());
        assert!((set.completion_percentage() - 100.0).abs() < 0.001);
    }

    #[test]
    fn test_priority_ordering() {
        assert!(ConditionPriority::Low < ConditionPriority::Normal);
        assert!(ConditionPriority::Normal < ConditionPriority::High);
        assert!(ConditionPriority::High < ConditionPriority::Critical);
    }
}
