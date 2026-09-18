//! Frame time budget analysis.

use super::breakdown::{FrameBreakdown, FrameStage};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::time::Duration;

/// Frame budget for a specific target FPS.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FrameBudget {
    /// Target frame time.
    pub target_frame_time: Duration,

    /// Budget per stage.
    pub stage_budgets: HashMap<FrameStage, Duration>,

    /// Target FPS.
    pub target_fps: f64,
}

impl FrameBudget {
    /// Create a new frame budget for a target FPS.
    pub fn new(target_fps: f64) -> Self {
        let target_frame_time = Duration::from_secs_f64(1.0 / target_fps);
        Self {
            target_frame_time,
            stage_budgets: HashMap::new(),
            target_fps,
        }
    }

    /// Set the budget for a stage.
    pub fn set_stage_budget(&mut self, stage: FrameStage, budget: Duration) {
        self.stage_budgets.insert(stage, budget);
    }

    /// Set the budget for a stage as a percentage of total frame time.
    pub fn set_stage_budget_percentage(&mut self, stage: FrameStage, percentage: f64) {
        let budget =
            Duration::from_secs_f64(self.target_frame_time.as_secs_f64() * (percentage / 100.0));
        self.stage_budgets.insert(stage, budget);
    }

    /// Get the budget for a stage.
    pub fn get_stage_budget(&self, stage: FrameStage) -> Option<Duration> {
        self.stage_budgets.get(&stage).copied()
    }

    /// Analyze a frame breakdown against this budget.
    pub fn analyze(&self, breakdown: &FrameBreakdown) -> BudgetAnalysis {
        let mut violations = Vec::new();
        let mut total_overage = Duration::ZERO;

        // Check total frame time
        let frame_overage = breakdown.total_time.saturating_sub(self.target_frame_time);
        let frame_within_budget = frame_overage == Duration::ZERO;

        // Check each stage
        for (stage, &budget) in &self.stage_budgets {
            let actual = breakdown.get_stage_time(*stage);
            if actual > budget {
                let overage = actual - budget;
                let percentage = (overage.as_secs_f64() / budget.as_secs_f64()) * 100.0;
                violations.push(BudgetViolation {
                    stage: *stage,
                    budget,
                    actual,
                    overage,
                    overage_percentage: percentage,
                });
                total_overage += overage;
            }
        }

        violations.sort_by(|a, b| b.overage.cmp(&a.overage));

        BudgetAnalysis {
            budget: self.clone(),
            breakdown: breakdown.clone(),
            frame_within_budget,
            frame_overage,
            violations,
            total_overage,
        }
    }
}

/// Budget violation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BudgetViolation {
    /// Stage that violated budget.
    pub stage: FrameStage,

    /// Budget for this stage.
    pub budget: Duration,

    /// Actual time spent.
    pub actual: Duration,

    /// Amount over budget.
    pub overage: Duration,

    /// Overage as percentage of budget.
    pub overage_percentage: f64,
}

impl BudgetViolation {
    /// Check if this is a critical violation (>50% over budget).
    pub fn is_critical(&self) -> bool {
        self.overage_percentage > 50.0
    }

    /// Check if this is a significant violation (>25% over budget).
    pub fn is_significant(&self) -> bool {
        self.overage_percentage > 25.0
    }

    /// Get a description of the violation.
    pub fn description(&self) -> String {
        let severity = if self.is_critical() {
            "CRITICAL"
        } else if self.is_significant() {
            "SIGNIFICANT"
        } else {
            "MINOR"
        };

        format!(
            "[{}] {:?}: {:?} over budget ({:.1}% over, budget: {:?}, actual: {:?})",
            severity, self.stage, self.overage, self.overage_percentage, self.budget, self.actual
        )
    }
}

/// Budget analysis result.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BudgetAnalysis {
    /// Budget used for analysis.
    pub budget: FrameBudget,

    /// Frame breakdown analyzed.
    pub breakdown: FrameBreakdown,

    /// Whether frame time is within budget.
    pub frame_within_budget: bool,

    /// Frame time overage.
    pub frame_overage: Duration,

    /// Budget violations.
    pub violations: Vec<BudgetViolation>,

    /// Total overage across all stages.
    pub total_overage: Duration,
}

impl BudgetAnalysis {
    /// Check if there are any violations.
    pub fn has_violations(&self) -> bool {
        !self.violations.is_empty() || !self.frame_within_budget
    }

    /// Get critical violations.
    pub fn critical_violations(&self) -> Vec<&BudgetViolation> {
        self.violations.iter().filter(|v| v.is_critical()).collect()
    }

    /// Generate a report.
    pub fn report(&self) -> String {
        let mut report = String::new();

        report.push_str(&format!("Target FPS: {:.1}\n", self.budget.target_fps));
        report.push_str(&format!(
            "Target Frame Time: {:?}\n",
            self.budget.target_frame_time
        ));
        report.push_str(&format!(
            "Actual Frame Time: {:?}\n",
            self.breakdown.total_time
        ));

        if self.frame_within_budget {
            report.push_str("✓ Frame within budget\n\n");
        } else {
            report.push_str(&format!(
                "✗ Frame over budget by {:?}\n\n",
                self.frame_overage
            ));
        }

        if self.violations.is_empty() {
            report.push_str("No stage budget violations.\n");
        } else {
            report.push_str(&format!("Budget Violations ({}):\n", self.violations.len()));
            for violation in &self.violations {
                report.push_str(&format!("  {}\n", violation.description()));
            }
        }

        report
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_frame_budget() {
        let budget = FrameBudget::new(60.0);
        assert_eq!(budget.target_fps, 60.0);
        assert_eq!(
            budget.target_frame_time,
            Duration::from_secs_f64(1.0 / 60.0)
        );
    }

    #[test]
    fn test_stage_budget() {
        let mut budget = FrameBudget::new(60.0);
        budget.set_stage_budget(FrameStage::Render, Duration::from_millis(10));

        assert_eq!(
            budget.get_stage_budget(FrameStage::Render),
            Some(Duration::from_millis(10))
        );
    }

    #[test]
    fn test_stage_budget_percentage() {
        let mut budget = FrameBudget::new(60.0);
        budget.set_stage_budget_percentage(FrameStage::Render, 50.0);

        let render_budget = budget
            .get_stage_budget(FrameStage::Render)
            .expect("should succeed in test");
        let expected = Duration::from_secs_f64((1.0 / 60.0) * 0.5);

        assert!((render_budget.as_secs_f64() - expected.as_secs_f64()).abs() < 0.0001);
    }

    #[test]
    fn test_budget_analysis_within() {
        let mut budget = FrameBudget::new(60.0);
        budget.set_stage_budget(FrameStage::Render, Duration::from_millis(10));

        let mut breakdown = FrameBreakdown::new();
        breakdown.add_stage(FrameStage::Render, Duration::from_millis(5));

        let analysis = budget.analyze(&breakdown);
        assert!(!analysis.has_violations());
    }

    #[test]
    fn test_budget_analysis_violations() {
        let mut budget = FrameBudget::new(60.0);
        budget.set_stage_budget(FrameStage::Render, Duration::from_millis(10));

        let mut breakdown = FrameBreakdown::new();
        breakdown.add_stage(FrameStage::Render, Duration::from_millis(20));

        let analysis = budget.analyze(&breakdown);
        assert!(analysis.has_violations());
        assert_eq!(analysis.violations.len(), 1);
        assert_eq!(analysis.violations[0].stage, FrameStage::Render);
    }

    #[test]
    fn test_violation_severity() {
        let violation = BudgetViolation {
            stage: FrameStage::Render,
            budget: Duration::from_millis(10),
            actual: Duration::from_millis(20),
            overage: Duration::from_millis(10),
            overage_percentage: 100.0,
        };

        assert!(violation.is_critical());
        assert!(violation.is_significant());
    }
}
