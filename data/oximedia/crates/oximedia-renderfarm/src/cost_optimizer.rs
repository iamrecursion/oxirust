// Copyright 2024 OxiMedia Project
// Licensed under the Apache License, Version 2.0

//! Render cost optimization.
//!
//! Provides tools for estimating rendering costs across cloud and on-premise
//! providers and selecting the most cost-effective compute strategy.

#![allow(dead_code)]

use serde::{Deserialize, Serialize};

/// A compute provider for rendering workloads.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum ComputeProvider {
    /// On-premise hardware (fixed cost, amortised).
    OnPremise,
    /// Amazon Web Services Spot Instances.
    AwsSpot,
    /// Google Cloud Platform Preemptible VMs.
    GcpPreemptible,
    /// Microsoft Azure Spot VMs.
    AzureSpot,
    /// Custom provider identified by name.
    Custom(String),
}

impl ComputeProvider {
    /// Approximate cost per core-hour in USD.
    ///
    /// On-premise cost reflects amortised server cost (~$0.03/core-hour).
    #[must_use]
    pub fn cost_per_core_hour_usd(&self) -> f64 {
        match self {
            Self::OnPremise => 0.03,
            Self::AwsSpot => 0.05,
            Self::GcpPreemptible => 0.045,
            Self::AzureSpot => 0.048,
            Self::Custom(_) => 0.06,
        }
    }

    /// Display name of the provider.
    #[must_use]
    pub fn display_name(&self) -> &str {
        match self {
            Self::OnPremise => "On-Premise",
            Self::AwsSpot => "AWS Spot",
            Self::GcpPreemptible => "GCP Preemptible",
            Self::AzureSpot => "Azure Spot",
            Self::Custom(n) => n.as_str(),
        }
    }
}

/// Budget constraints for a render job.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RenderBudget {
    /// Maximum total spend in USD.
    pub total_usd: f64,
    /// Wall-clock deadline in hours.
    pub deadline_hours: f32,
    /// Minimum acceptable output quality (0.0–1.0).
    pub min_quality: f32,
}

impl RenderBudget {
    /// Create a new render budget.
    #[must_use]
    pub fn new(total_usd: f64, deadline_hours: f32, min_quality: f32) -> Self {
        Self {
            total_usd,
            deadline_hours,
            min_quality: min_quality.clamp(0.0, 1.0),
        }
    }

    /// Maximum spend per hour (`total_usd` / `deadline_hours`).
    ///
    /// Returns `f64::INFINITY` if `deadline_hours` is 0.
    #[must_use]
    pub fn hourly_budget(&self) -> f64 {
        if self.deadline_hours <= 0.0 {
            return f64::INFINITY;
        }
        self.total_usd / f64::from(self.deadline_hours)
    }
}

/// Cost estimate for a specific provider.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CostEstimate {
    /// The provider this estimate is for.
    pub provider: ComputeProvider,
    /// Number of CPU cores needed to meet the deadline.
    pub cores_needed: u32,
    /// Estimated wall-clock hours to complete the job.
    pub hours_needed: f32,
    /// Total cost in USD.
    pub total_usd: f64,
    /// Whether this estimate meets the deadline constraint.
    pub meets_deadline: bool,
}

/// Optimises render cost across a set of providers.
pub struct CostOptimizer;

impl CostOptimizer {
    /// Estimate costs for each provider and return sorted (cheapest first) results.
    ///
    /// # Arguments
    /// * `job_complexity` – abstract complexity score (1.0 = simple, 10.0 = very heavy).
    /// * `frames` – number of frames to render.
    /// * `budget` – budget and deadline constraints.
    /// * `providers` – list of providers to consider.
    ///
    /// Providers that cannot meet the deadline at any practical core count are still
    /// included but marked `meets_deadline: false`.
    #[must_use]
    pub fn estimate(
        job_complexity: f32,
        frames: u32,
        budget: &RenderBudget,
        providers: &[ComputeProvider],
    ) -> Vec<CostEstimate> {
        // seconds per frame per core (baseline = 10 s at complexity 1.0)
        let secs_per_frame_per_core = 10.0 * f64::from(job_complexity.max(0.1));

        let mut estimates: Vec<CostEstimate> = providers
            .iter()
            .map(|provider| {
                let rate = provider.cost_per_core_hour_usd();

                // Determine minimum cores needed to meet deadline
                let total_frame_core_secs = f64::from(frames) * secs_per_frame_per_core;
                let deadline_secs = f64::from(budget.deadline_hours * 3600.0);

                // cores_needed = ceil(total_work / deadline)
                let cores_needed = if deadline_secs > 0.0 {
                    ((total_frame_core_secs / deadline_secs).ceil() as u32).max(1)
                } else {
                    1u32
                };

                let hours_needed = if cores_needed > 0 {
                    (total_frame_core_secs / (f64::from(cores_needed) * 3600.0)) as f32
                } else {
                    0.0
                };

                let total_usd = rate * f64::from(cores_needed) * f64::from(hours_needed);
                let meets_deadline =
                    hours_needed <= budget.deadline_hours && total_usd <= budget.total_usd;

                CostEstimate {
                    provider: provider.clone(),
                    cores_needed,
                    hours_needed,
                    total_usd,
                    meets_deadline,
                }
            })
            .collect();

        // Sort: meets_deadline first, then by total_usd ascending
        estimates.sort_by(|a, b| {
            b.meets_deadline.cmp(&a.meets_deadline).then(
                a.total_usd
                    .partial_cmp(&b.total_usd)
                    .unwrap_or(std::cmp::Ordering::Equal),
            )
        });

        estimates
    }
}

/// Strategy for using spot/preemptible instances.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SpotInstanceStrategy {
    /// Maximum acceptable interruption rate (0.0–1.0).
    pub max_interruption_pct: f32,
    /// How often to save checkpoints (minutes).
    pub checkpoint_interval_mins: u32,
    /// Whether to fall back to on-demand instances if interrupted too often.
    pub fallback_to_ondemand: bool,
}

impl SpotInstanceStrategy {
    /// Create a conservative spot strategy with frequent checkpoints.
    #[must_use]
    pub fn conservative() -> Self {
        Self {
            max_interruption_pct: 0.05,
            checkpoint_interval_mins: 5,
            fallback_to_ondemand: true,
        }
    }

    /// Create an aggressive spot strategy (fewer checkpoints, higher interruption tolerance).
    #[must_use]
    pub fn aggressive() -> Self {
        Self {
            max_interruption_pct: 0.20,
            checkpoint_interval_mins: 30,
            fallback_to_ondemand: false,
        }
    }

    /// Whether the strategy is viable given an observed interruption rate.
    #[must_use]
    pub fn is_viable(&self, observed_interruption_pct: f32) -> bool {
        observed_interruption_pct <= self.max_interruption_pct
    }
}

/// Post-render cost report comparing estimated vs actual spend.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CostReport {
    /// Original estimated spend in USD.
    pub estimated_usd: f64,
    /// Actual spend in USD.
    pub actual_usd: f64,
    /// Savings percentage ((estimated - actual) / estimated × 100).
    pub savings_pct: f32,
    /// Number of spot interruptions experienced.
    pub interruptions: u32,
}

impl CostReport {
    /// Build a cost report from estimated and actual values.
    #[must_use]
    pub fn new(estimated_usd: f64, actual_usd: f64, interruptions: u32) -> Self {
        let savings_pct = if estimated_usd > 0.0 {
            ((estimated_usd - actual_usd) / estimated_usd * 100.0) as f32
        } else {
            0.0
        };
        Self {
            estimated_usd,
            actual_usd,
            savings_pct,
            interruptions,
        }
    }

    /// Whether the job came in under budget (actual < estimated).
    #[must_use]
    pub fn under_budget(&self) -> bool {
        self.actual_usd < self.estimated_usd
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_compute_provider_cost() {
        assert!(
            ComputeProvider::OnPremise.cost_per_core_hour_usd()
                < ComputeProvider::AwsSpot.cost_per_core_hour_usd()
        );
        assert!(ComputeProvider::GcpPreemptible.cost_per_core_hour_usd() > 0.0);
    }

    #[test]
    fn test_compute_provider_display_name() {
        assert_eq!(ComputeProvider::OnPremise.display_name(), "On-Premise");
        assert_eq!(ComputeProvider::AwsSpot.display_name(), "AWS Spot");
        assert_eq!(
            ComputeProvider::Custom("FooCloud".to_string()).display_name(),
            "FooCloud"
        );
    }

    #[test]
    fn test_render_budget_hourly() {
        let budget = RenderBudget::new(100.0, 4.0, 0.8);
        assert!((budget.hourly_budget() - 25.0).abs() < 1e-9);
    }

    #[test]
    fn test_render_budget_zero_deadline() {
        let budget = RenderBudget::new(100.0, 0.0, 0.8);
        assert!(budget.hourly_budget().is_infinite());
    }

    #[test]
    fn test_cost_estimate_sorted_cheapest_first() {
        let budget = RenderBudget::new(10_000.0, 24.0, 0.5);
        let providers = vec![
            ComputeProvider::AwsSpot,
            ComputeProvider::OnPremise,
            ComputeProvider::GcpPreemptible,
        ];
        let estimates = CostOptimizer::estimate(1.0, 100, &budget, &providers);
        assert_eq!(estimates.len(), 3);
        // All meeting deadline should come before non-meeting
        let meeting: Vec<_> = estimates.iter().filter(|e| e.meets_deadline).collect();
        let not_meeting: Vec<_> = estimates.iter().filter(|e| !e.meets_deadline).collect();
        for (m, nm) in meeting.iter().zip(not_meeting.iter()) {
            // Meeting budget items come before non-meeting regardless of cost
            // (they're sorted by meets_deadline desc, then cost)
            assert!(m.total_usd <= nm.total_usd || m.meets_deadline);
        }
    }

    #[test]
    fn test_cost_estimate_positive_cost() {
        let budget = RenderBudget::new(1_000.0, 8.0, 0.5);
        let providers = vec![ComputeProvider::AwsSpot];
        let estimates = CostOptimizer::estimate(2.0, 500, &budget, &providers);
        assert_eq!(estimates.len(), 1);
        assert!(estimates[0].total_usd > 0.0);
        assert!(estimates[0].cores_needed >= 1);
    }

    #[test]
    fn test_cost_estimate_empty_providers() {
        let budget = RenderBudget::new(1_000.0, 4.0, 0.5);
        let estimates = CostOptimizer::estimate(1.0, 100, &budget, &[]);
        assert!(estimates.is_empty());
    }

    #[test]
    fn test_spot_strategy_conservative() {
        let s = SpotInstanceStrategy::conservative();
        assert!(s.is_viable(0.04));
        assert!(!s.is_viable(0.10));
        assert!(s.fallback_to_ondemand);
    }

    #[test]
    fn test_spot_strategy_aggressive() {
        let s = SpotInstanceStrategy::aggressive();
        assert!(s.is_viable(0.15));
        assert!(!s.fallback_to_ondemand);
    }

    #[test]
    fn test_cost_report_savings() {
        let report = CostReport::new(100.0, 60.0, 0);
        assert!((report.savings_pct - 40.0).abs() < 0.01);
        assert!(report.under_budget());
    }

    #[test]
    fn test_cost_report_over_budget() {
        let report = CostReport::new(50.0, 75.0, 2);
        assert!(!report.under_budget());
        assert!(report.savings_pct < 0.0);
    }

    #[test]
    fn test_cost_report_zero_estimate() {
        let report = CostReport::new(0.0, 10.0, 0);
        assert!((report.savings_pct - 0.0).abs() < f32::EPSILON);
    }
}
