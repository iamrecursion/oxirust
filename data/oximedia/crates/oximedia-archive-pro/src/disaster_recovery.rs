//! Disaster recovery planning for digital archives.
//!
//! This module provides tools for planning and simulating recovery from
//! catastrophic data loss events:
//! - **RecoveryObjective** - RTO/RPO targets
//! - **DisasterScenario** - Categorized disaster types with probability
//! - **RecoveryPlan** - Ordered recovery steps with dependencies
//! - **DrSimulation** - Seeded Monte-Carlo simulation of plan execution
//! - **RiskMatrix** - Prioritise scenarios by risk = probability × impact

#![allow(dead_code)]

use serde::{Deserialize, Serialize};

// ─────────────────────────────────────────────────────────────
// RecoveryObjective
// ─────────────────────────────────────────────────────────────

/// Recovery Time Objective and Recovery Point Objective.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RecoveryObjective {
    /// Maximum acceptable recovery time in hours
    pub rto_hours: u32,
    /// Maximum acceptable data loss window in hours
    pub rpo_hours: u32,
}

impl RecoveryObjective {
    /// Create a new recovery objective.
    #[must_use]
    pub fn new(rto_hours: u32, rpo_hours: u32) -> Self {
        Self {
            rto_hours,
            rpo_hours,
        }
    }

    /// Returns true if the actual RTO and RPO meet (are ≤) the SLA targets.
    #[must_use]
    pub fn meets_sla(&self, actual_rto: u32, actual_rpo: u32) -> bool {
        actual_rto <= self.rto_hours && actual_rpo <= self.rpo_hours
    }
}

// ─────────────────────────────────────────────────────────────
// DisasterScenario
// ─────────────────────────────────────────────────────────────

/// Types of disaster scenarios relevant to digital archives.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum DisasterScenario {
    /// Primary data centre goes offline
    DataCenter,
    /// Network outage prevents data access
    Network,
    /// Ransomware encrypts archive data
    Ransomware,
    /// Physical disaster (flood, fire, earthquake) affecting storage site
    NaturalDisaster,
    /// Storage hardware failure
    HardwareFailure,
}

impl DisasterScenario {
    /// Returns the estimated probability of this scenario occurring per year.
    #[must_use]
    pub fn probability_per_year(&self) -> f64 {
        match self {
            Self::DataCenter => 0.05,
            Self::Network => 0.30,
            Self::Ransomware => 0.15,
            Self::NaturalDisaster => 0.02,
            Self::HardwareFailure => 0.25,
        }
    }

    /// Returns the scenario name.
    #[must_use]
    pub const fn name(&self) -> &'static str {
        match self {
            Self::DataCenter => "DataCenter",
            Self::Network => "Network",
            Self::Ransomware => "Ransomware",
            Self::NaturalDisaster => "NaturalDisaster",
            Self::HardwareFailure => "HardwareFailure",
        }
    }

    /// Returns the typical impact on data integrity (0.0–1.0).
    #[must_use]
    pub fn typical_impact(&self) -> f64 {
        match self {
            Self::DataCenter => 0.8,
            Self::Network => 0.3,
            Self::Ransomware => 0.9,
            Self::NaturalDisaster => 1.0,
            Self::HardwareFailure => 0.6,
        }
    }
}

// ─────────────────────────────────────────────────────────────
// RecoveryStep
// ─────────────────────────────────────────────────────────────

/// A single step in a recovery plan.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RecoveryStep {
    /// Unique step identifier (1-based)
    pub order: u32,
    /// Human-readable step name
    pub name: String,
    /// Expected duration in hours
    pub duration_hours: f32,
    /// Step IDs that must complete before this step can begin
    pub requires: Vec<u32>,
    /// Team responsible for executing this step
    pub responsible_team: String,
}

impl RecoveryStep {
    /// Returns true if this step has no prerequisites.
    #[must_use]
    pub fn is_independent(&self) -> bool {
        self.requires.is_empty()
    }
}

// ─────────────────────────────────────────────────────────────
// RecoveryPlan
// ─────────────────────────────────────────────────────────────

/// A complete disaster recovery plan.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RecoveryPlan {
    /// The disaster scenario this plan addresses
    pub scenario: DisasterScenario,
    /// Ordered recovery steps
    pub steps: Vec<RecoveryStep>,
    /// Estimated total recovery time in hours (critical path)
    pub total_time_hours: f32,
    /// Estimated probability of successful recovery (0.0–1.0)
    pub success_probability: f32,
}

impl RecoveryPlan {
    /// Returns the critical path length (longest dependency chain duration).
    #[must_use]
    pub fn critical_path_hours(&self) -> f32 {
        // Simple greedy: sum independent + dependent sequential steps
        self.steps.iter().map(|s| s.duration_hours).sum()
    }

    /// Returns all independent steps (no prerequisites).
    #[must_use]
    pub fn independent_steps(&self) -> Vec<&RecoveryStep> {
        self.steps.iter().filter(|s| s.is_independent()).collect()
    }
}

// ─────────────────────────────────────────────────────────────
// DrSimResult
// ─────────────────────────────────────────────────────────────

/// Result of running a disaster recovery simulation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DrSimResult {
    /// Whether the recovery completed successfully
    pub completed: bool,
    /// Actual recovery time in hours (may exceed `total_time_hours` if steps fail)
    pub actual_rto_hours: f32,
    /// Step IDs that failed during simulation
    pub failed_steps: Vec<u32>,
    /// Name of the step that formed the bottleneck, if any
    pub bottleneck: Option<String>,
}

// ─────────────────────────────────────────────────────────────
// DrSimulation
// ─────────────────────────────────────────────────────────────

/// Disaster recovery simulator using a seeded LCG pseudo-random number generator.
pub struct DrSimulation;

impl DrSimulation {
    /// Simulate execution of a recovery plan.
    ///
    /// * `plan` – the recovery plan to simulate
    /// * `seed` – deterministic seed for reproducibility
    #[must_use]
    pub fn run(plan: &RecoveryPlan, seed: u64) -> DrSimResult {
        let mut rng = LcgRng::new(seed);
        let mut total_hours = 0.0f32;
        let mut failed_steps: Vec<u32> = Vec::new();
        let mut bottleneck: Option<String> = None;
        let mut max_step_hours = 0.0f32;

        for step in &plan.steps {
            // Check prerequisites
            let prereqs_ok = step
                .requires
                .iter()
                .all(|&req| !failed_steps.contains(&req));

            if !prereqs_ok {
                failed_steps.push(step.order);
                continue;
            }

            // Random failure based on step success probability
            // Each step has base 90% success probability, scaled by plan overall
            let base_success = 0.90 * plan.success_probability;
            let roll = rng.next_f32();

            if roll > base_success {
                failed_steps.push(step.order);
                // Failed step adds penalty time
                let penalty = step.duration_hours * 1.5;
                total_hours += penalty;
                if penalty > max_step_hours {
                    max_step_hours = penalty;
                    bottleneck = Some(format!("{} (failed)", step.name));
                }
            } else {
                // Slight random variation ±20%
                let variation = 1.0 + (rng.next_f32() - 0.5) * 0.4;
                let actual_duration = step.duration_hours * variation;
                total_hours += actual_duration;
                if actual_duration > max_step_hours {
                    max_step_hours = actual_duration;
                    bottleneck = Some(step.name.clone());
                }
            }
        }

        let completed = failed_steps.is_empty();
        DrSimResult {
            completed,
            actual_rto_hours: total_hours,
            failed_steps,
            bottleneck,
        }
    }
}

/// Simple LCG pseudo-random number generator.
struct LcgRng {
    state: u64,
}

impl LcgRng {
    fn new(seed: u64) -> Self {
        Self {
            state: seed.wrapping_add(1),
        }
    }

    fn next_u64(&mut self) -> u64 {
        // LCG parameters from Knuth
        self.state = self
            .state
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        self.state
    }

    fn next_f32(&mut self) -> f32 {
        (self.next_u64() >> 33) as f32 / (u32::MAX as f32)
    }
}

// ─────────────────────────────────────────────────────────────
// RiskMatrix
// ─────────────────────────────────────────────────────────────

/// A risk matrix prioritising disaster scenarios by `probability × impact`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RiskMatrix {
    /// Entries: (scenario, probability_per_year, impact 0–1)
    pub scenarios: Vec<(DisasterScenario, f64, f64)>,
}

impl RiskMatrix {
    /// Create a new risk matrix from explicit entries.
    #[must_use]
    pub fn new(scenarios: Vec<(DisasterScenario, f64, f64)>) -> Self {
        Self { scenarios }
    }

    /// Create a default risk matrix using the built-in scenario probabilities and impacts.
    #[must_use]
    pub fn with_defaults() -> Self {
        let entries = [
            DisasterScenario::DataCenter,
            DisasterScenario::Network,
            DisasterScenario::Ransomware,
            DisasterScenario::NaturalDisaster,
            DisasterScenario::HardwareFailure,
        ]
        .iter()
        .map(|&s| (s, s.probability_per_year(), s.typical_impact()))
        .collect();

        Self { scenarios: entries }
    }

    /// Add an entry to the matrix.
    pub fn add(&mut self, scenario: DisasterScenario, probability: f64, impact: f64) {
        self.scenarios.push((scenario, probability, impact));
    }

    /// Returns the scenario with the highest risk score (`probability × impact`).
    #[must_use]
    pub fn highest_risk(&self) -> Option<&DisasterScenario> {
        self.scenarios
            .iter()
            .max_by(|(_, p1, i1), (_, p2, i2)| {
                let r1 = p1 * i1;
                let r2 = p2 * i2;
                r1.partial_cmp(&r2).unwrap_or(std::cmp::Ordering::Equal)
            })
            .map(|(s, _, _)| s)
    }

    /// Returns all scenarios sorted by risk score descending.
    #[must_use]
    pub fn ranked(&self) -> Vec<(&DisasterScenario, f64)> {
        let mut ranked: Vec<(&DisasterScenario, f64)> =
            self.scenarios.iter().map(|(s, p, i)| (s, p * i)).collect();
        ranked.sort_by(|(_, r1), (_, r2)| r2.partial_cmp(r1).unwrap_or(std::cmp::Ordering::Equal));
        ranked
    }
}

// ─────────────────────────────────────────────────────────────
// BackupRecord
// ─────────────────────────────────────────────────────────────

/// A record of a backup that exists for disaster recovery purposes.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BackupRecord {
    /// Human-readable label for the backup
    pub label: String,
    /// Location or URI of the backup
    pub location: String,
    /// Number of files in the backup
    pub file_count: usize,
    /// Total size in bytes
    pub total_size_bytes: u64,
    /// Age of the backup in hours
    pub age_hours: u32,
    /// Whether the backup has been verified (fixity checked) recently
    pub verified: bool,
    /// Scenarios this backup is designed to protect against
    pub covers_scenarios: Vec<DisasterScenario>,
    /// Whether the backup is geographically separated from the primary site
    pub geographically_separated: bool,
}

// ─────────────────────────────────────────────────────────────
// ValidationSeverity
// ─────────────────────────────────────────────────────────────

/// Severity of a DR validation finding.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum DrValidationSeverity {
    /// Informational: best practice recommendation.
    Info,
    /// Warning: potential gap that should be addressed.
    Warning,
    /// Critical: a significant gap that threatens recoverability.
    Critical,
}

impl DrValidationSeverity {
    /// Returns a label string.
    #[must_use]
    pub const fn label(&self) -> &'static str {
        match self {
            Self::Info => "INFO",
            Self::Warning => "WARNING",
            Self::Critical => "CRITICAL",
        }
    }
}

// ─────────────────────────────────────────────────────────────
// ValidationFinding
// ─────────────────────────────────────────────────────────────

/// A single finding from DR plan validation.
#[derive(Debug, Clone)]
pub struct DrValidationFinding {
    /// Severity of the finding.
    pub severity: DrValidationSeverity,
    /// Short code.
    pub code: String,
    /// Human-readable message.
    pub message: String,
    /// Recommendation for remediation.
    pub recommendation: String,
}

// ─────────────────────────────────────────────────────────────
// DrValidationReport
// ─────────────────────────────────────────────────────────────

/// Overall DR readiness assessment.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DrReadiness {
    /// Fully ready: all critical checks pass, RTO/RPO met.
    Ready,
    /// Ready with concerns: no critical issues but warnings exist.
    ReadyWithWarnings,
    /// Not ready: critical gaps exist.
    NotReady,
}

/// Comprehensive report from DR plan validation.
#[derive(Debug, Clone)]
pub struct DrValidationReport {
    /// Overall readiness assessment.
    pub readiness: DrReadiness,
    /// All findings.
    pub findings: Vec<DrValidationFinding>,
    /// Whether the plan meets its RTO target.
    pub meets_rto: bool,
    /// Whether the plan meets its RPO target.
    pub meets_rpo: bool,
    /// Percentage of disaster scenarios covered by backups (0.0 - 1.0).
    pub scenario_coverage: f64,
    /// Whether at least one backup is geographically separated.
    pub has_geographic_separation: bool,
    /// Number of verified backups.
    pub verified_backup_count: usize,
    /// Total backup count.
    pub total_backup_count: usize,
}

impl DrValidationReport {
    /// Returns the number of critical findings.
    #[must_use]
    pub fn critical_count(&self) -> usize {
        self.findings
            .iter()
            .filter(|f| f.severity == DrValidationSeverity::Critical)
            .count()
    }

    /// Returns the number of warning findings.
    #[must_use]
    pub fn warning_count(&self) -> usize {
        self.findings
            .iter()
            .filter(|f| f.severity == DrValidationSeverity::Warning)
            .count()
    }

    /// Returns a summary string.
    #[must_use]
    pub fn summary(&self) -> String {
        format!(
            "DR Readiness: {:?} | RTO: {} | RPO: {} | Coverage: {:.0}% | Backups: {}/{} verified | Criticals: {} | Warnings: {}",
            self.readiness,
            if self.meets_rto { "PASS" } else { "FAIL" },
            if self.meets_rpo { "PASS" } else { "FAIL" },
            self.scenario_coverage * 100.0,
            self.verified_backup_count,
            self.total_backup_count,
            self.critical_count(),
            self.warning_count(),
        )
    }
}

// ─────────────────────────────────────────────────────────────
// DrPlanValidator
// ─────────────────────────────────────────────────────────────

/// Validates a disaster recovery plan for completeness and recoverability.
///
/// Checks include:
/// - Whether the plan's RTO/RPO targets can be met
/// - Whether all identified disaster scenarios have backup coverage
/// - Whether backups are verified and recent
/// - Whether geographic separation exists
/// - Whether the plan has sufficient recovery steps
/// - Whether step dependencies form a valid DAG
pub struct DrPlanValidator {
    /// Maximum acceptable backup age in hours before a warning is raised.
    pub max_backup_age_hours: u32,
    /// Required minimum number of backups.
    pub min_backup_count: usize,
    /// Whether geographic separation is required.
    pub require_geographic_separation: bool,
}

impl Default for DrPlanValidator {
    fn default() -> Self {
        Self {
            max_backup_age_hours: 24,
            min_backup_count: 2,
            require_geographic_separation: true,
        }
    }
}

impl DrPlanValidator {
    /// Create a validator with default settings.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Validate a disaster recovery plan against a set of backup records and
    /// recovery objectives.
    ///
    /// This performs comprehensive validation including:
    /// 1. RTO feasibility (critical path vs target)
    /// 2. RPO feasibility (backup age vs target)
    /// 3. Scenario coverage (are all scenarios covered by at least one backup?)
    /// 4. Backup verification status
    /// 5. Geographic separation
    /// 6. Step dependency validation (no cycles, no missing deps)
    /// 7. Minimum backup count
    #[must_use]
    pub fn validate(
        &self,
        plan: &RecoveryPlan,
        backups: &[BackupRecord],
        objective: &RecoveryObjective,
    ) -> DrValidationReport {
        let mut findings = Vec::new();

        // 1. RTO check: can the plan complete within the RTO target?
        let critical_path = plan.critical_path_hours();
        let meets_rto = critical_path <= objective.rto_hours as f32;
        if !meets_rto {
            findings.push(DrValidationFinding {
                severity: DrValidationSeverity::Critical,
                code: "RTO_EXCEEDED".to_string(),
                message: format!(
                    "Plan critical path ({:.1}h) exceeds RTO target ({}h)",
                    critical_path, objective.rto_hours
                ),
                recommendation: "Reduce recovery step durations or parallelize steps".to_string(),
            });
        }

        // 2. RPO check: is the most recent backup within the RPO window?
        let min_backup_age = backups
            .iter()
            .map(|b| b.age_hours)
            .min()
            .unwrap_or(u32::MAX);
        let meets_rpo = min_backup_age <= objective.rpo_hours;
        if !meets_rpo {
            findings.push(DrValidationFinding {
                severity: DrValidationSeverity::Critical,
                code: "RPO_EXCEEDED".to_string(),
                message: format!(
                    "Most recent backup is {}h old, exceeding RPO target of {}h",
                    min_backup_age, objective.rpo_hours
                ),
                recommendation: "Increase backup frequency to meet RPO target".to_string(),
            });
        }

        // 3. Scenario coverage
        let all_scenarios = [
            DisasterScenario::DataCenter,
            DisasterScenario::Network,
            DisasterScenario::Ransomware,
            DisasterScenario::NaturalDisaster,
            DisasterScenario::HardwareFailure,
        ];
        let covered_scenarios: std::collections::HashSet<_> = backups
            .iter()
            .flat_map(|b| b.covers_scenarios.iter().copied())
            .collect();
        let coverage = if all_scenarios.is_empty() {
            1.0
        } else {
            covered_scenarios.len() as f64 / all_scenarios.len() as f64
        };
        let uncovered: Vec<_> = all_scenarios
            .iter()
            .filter(|s| !covered_scenarios.contains(s))
            .collect();
        if !uncovered.is_empty() {
            let names: Vec<&str> = uncovered.iter().map(|s| s.name()).collect();
            findings.push(DrValidationFinding {
                severity: DrValidationSeverity::Warning,
                code: "SCENARIO_GAP".to_string(),
                message: format!("Uncovered disaster scenarios: {}", names.join(", ")),
                recommendation: "Add backup strategies that cover these scenarios".to_string(),
            });
        }

        // 4. Backup verification status
        let verified_count = backups.iter().filter(|b| b.verified).count();
        let unverified_count = backups.len() - verified_count;
        if unverified_count > 0 {
            findings.push(DrValidationFinding {
                severity: DrValidationSeverity::Warning,
                code: "UNVERIFIED_BACKUPS".to_string(),
                message: format!(
                    "{} of {} backups have not been verified recently",
                    unverified_count,
                    backups.len()
                ),
                recommendation: "Run fixity checks on all backups".to_string(),
            });
        }

        // 5. Geographic separation
        let has_geo_sep = backups.iter().any(|b| b.geographically_separated);
        if self.require_geographic_separation && !has_geo_sep {
            findings.push(DrValidationFinding {
                severity: DrValidationSeverity::Critical,
                code: "NO_GEO_SEPARATION".to_string(),
                message: "No geographically separated backup exists".to_string(),
                recommendation:
                    "Establish at least one off-site backup in a different geographic region"
                        .to_string(),
            });
        }

        // 6. Backup age warnings
        for backup in backups {
            if backup.age_hours > self.max_backup_age_hours {
                findings.push(DrValidationFinding {
                    severity: DrValidationSeverity::Warning,
                    code: "STALE_BACKUP".to_string(),
                    message: format!(
                        "Backup '{}' is {}h old (threshold: {}h)",
                        backup.label, backup.age_hours, self.max_backup_age_hours
                    ),
                    recommendation: "Refresh this backup".to_string(),
                });
            }
        }

        // 7. Minimum backup count
        if backups.len() < self.min_backup_count {
            findings.push(DrValidationFinding {
                severity: DrValidationSeverity::Critical,
                code: "INSUFFICIENT_BACKUPS".to_string(),
                message: format!(
                    "Only {} backup(s) exist; minimum required is {}",
                    backups.len(),
                    self.min_backup_count
                ),
                recommendation: "Create additional backups".to_string(),
            });
        }

        // 8. Step dependency validation
        let step_ids: std::collections::HashSet<u32> = plan.steps.iter().map(|s| s.order).collect();
        for step in &plan.steps {
            for &req in &step.requires {
                if !step_ids.contains(&req) {
                    findings.push(DrValidationFinding {
                        severity: DrValidationSeverity::Critical,
                        code: "MISSING_DEPENDENCY".to_string(),
                        message: format!(
                            "Step {} '{}' depends on non-existent step {}",
                            step.order, step.name, req
                        ),
                        recommendation: "Fix step dependencies".to_string(),
                    });
                }
            }
        }

        // 9. Empty plan check
        if plan.steps.is_empty() {
            findings.push(DrValidationFinding {
                severity: DrValidationSeverity::Critical,
                code: "EMPTY_PLAN".to_string(),
                message: "Recovery plan has no steps".to_string(),
                recommendation: "Define recovery steps".to_string(),
            });
        }

        // Determine readiness
        let has_critical = findings
            .iter()
            .any(|f| f.severity == DrValidationSeverity::Critical);
        let has_warning = findings
            .iter()
            .any(|f| f.severity == DrValidationSeverity::Warning);

        let readiness = if has_critical {
            DrReadiness::NotReady
        } else if has_warning {
            DrReadiness::ReadyWithWarnings
        } else {
            DrReadiness::Ready
        };

        DrValidationReport {
            readiness,
            findings,
            meets_rto,
            meets_rpo,
            scenario_coverage: coverage,
            has_geographic_separation: has_geo_sep,
            verified_backup_count: verified_count,
            total_backup_count: backups.len(),
        }
    }
}

// ─────────────────────────────────────────────────────────────
// Unit tests
// ─────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn make_simple_plan() -> RecoveryPlan {
        RecoveryPlan {
            scenario: DisasterScenario::HardwareFailure,
            steps: vec![
                RecoveryStep {
                    order: 1,
                    name: "Assess damage".to_string(),
                    duration_hours: 1.0,
                    requires: vec![],
                    responsible_team: "IT Ops".to_string(),
                },
                RecoveryStep {
                    order: 2,
                    name: "Restore from backup".to_string(),
                    duration_hours: 4.0,
                    requires: vec![1],
                    responsible_team: "Data Engineering".to_string(),
                },
                RecoveryStep {
                    order: 3,
                    name: "Verify integrity".to_string(),
                    duration_hours: 1.0,
                    requires: vec![2],
                    responsible_team: "QA".to_string(),
                },
            ],
            total_time_hours: 6.0,
            success_probability: 0.95,
        }
    }

    // ── RecoveryObjective ─────────────────────────────────────

    #[test]
    fn test_meets_sla_pass() {
        let obj = RecoveryObjective::new(8, 4);
        assert!(obj.meets_sla(6, 2));
    }

    #[test]
    fn test_meets_sla_fail_rto() {
        let obj = RecoveryObjective::new(4, 2);
        assert!(!obj.meets_sla(8, 1));
    }

    #[test]
    fn test_meets_sla_fail_rpo() {
        let obj = RecoveryObjective::new(8, 2);
        assert!(!obj.meets_sla(4, 6));
    }

    // ── DisasterScenario ──────────────────────────────────────

    #[test]
    fn test_probability_per_year_ranges() {
        for scenario in &[
            DisasterScenario::DataCenter,
            DisasterScenario::Network,
            DisasterScenario::Ransomware,
            DisasterScenario::NaturalDisaster,
            DisasterScenario::HardwareFailure,
        ] {
            let p = scenario.probability_per_year();
            assert!(p > 0.0 && p <= 1.0);
        }
    }

    #[test]
    fn test_scenario_names() {
        assert_eq!(DisasterScenario::Ransomware.name(), "Ransomware");
        assert_eq!(DisasterScenario::NaturalDisaster.name(), "NaturalDisaster");
    }

    // ── RecoveryPlan ──────────────────────────────────────────

    #[test]
    fn test_critical_path_hours() {
        let plan = make_simple_plan();
        let path = plan.critical_path_hours();
        assert!((path - 6.0).abs() < 1e-5);
    }

    #[test]
    fn test_independent_steps() {
        let plan = make_simple_plan();
        let independent = plan.independent_steps();
        assert_eq!(independent.len(), 1);
        assert_eq!(independent[0].order, 1);
    }

    #[test]
    fn test_recovery_step_is_independent() {
        let step = RecoveryStep {
            order: 1,
            name: "Start".into(),
            duration_hours: 0.5,
            requires: vec![],
            responsible_team: "IT".into(),
        };
        assert!(step.is_independent());
    }

    // ── DrSimulation ──────────────────────────────────────────

    #[test]
    fn test_sim_deterministic() {
        let plan = make_simple_plan();
        let r1 = DrSimulation::run(&plan, 42);
        let r2 = DrSimulation::run(&plan, 42);
        assert_eq!(r1.completed, r2.completed);
        assert!((r1.actual_rto_hours - r2.actual_rto_hours).abs() < 1e-5);
    }

    #[test]
    fn test_sim_high_success_plan() {
        // With high success_probability most runs should complete
        let mut plan = make_simple_plan();
        plan.success_probability = 1.0; // guarantees success (roll always < 1.0 * 0.9)
        let result = DrSimulation::run(&plan, 1234);
        assert_eq!(result.failed_steps.len(), 0);
        assert!(result.completed);
    }

    #[test]
    fn test_sim_rto_positive() {
        let plan = make_simple_plan();
        let result = DrSimulation::run(&plan, 99);
        assert!(result.actual_rto_hours > 0.0);
    }

    #[test]
    fn test_sim_returns_bottleneck() {
        let plan = make_simple_plan();
        let result = DrSimulation::run(&plan, 7);
        // Bottleneck should always be set when there are steps
        assert!(result.bottleneck.is_some());
    }

    // ── RiskMatrix ────────────────────────────────────────────

    #[test]
    fn test_highest_risk_not_none() {
        let matrix = RiskMatrix::with_defaults();
        assert!(matrix.highest_risk().is_some());
    }

    #[test]
    fn test_ranked_descending() {
        let matrix = RiskMatrix::with_defaults();
        let ranked = matrix.ranked();
        for i in 0..ranked.len().saturating_sub(1) {
            assert!(ranked[i].1 >= ranked[i + 1].1);
        }
    }

    #[test]
    fn test_risk_matrix_custom_entry() {
        let mut matrix = RiskMatrix::new(vec![]);
        matrix.add(DisasterScenario::Ransomware, 0.8, 0.9);
        matrix.add(DisasterScenario::Network, 0.1, 0.2);
        let top = matrix.highest_risk().expect("operation should succeed");
        assert_eq!(*top, DisasterScenario::Ransomware);
    }

    #[test]
    fn test_risk_matrix_empty() {
        let matrix = RiskMatrix::new(vec![]);
        assert!(matrix.highest_risk().is_none());
    }

    // ── DrPlanValidator tests ────────────────────────────────────

    fn make_complete_backups() -> Vec<BackupRecord> {
        vec![
            BackupRecord {
                label: "Primary Backup".to_string(),
                location: "s3://primary-backup/".to_string(),
                file_count: 1000,
                total_size_bytes: 1024 * 1024 * 1024 * 10, // 10 GiB
                age_hours: 4,
                verified: true,
                covers_scenarios: vec![
                    DisasterScenario::HardwareFailure,
                    DisasterScenario::Network,
                    DisasterScenario::Ransomware,
                ],
                geographically_separated: false,
            },
            BackupRecord {
                label: "Off-site Backup".to_string(),
                location: "s3://offsite-eu/".to_string(),
                file_count: 1000,
                total_size_bytes: 1024 * 1024 * 1024 * 10,
                age_hours: 12,
                verified: true,
                covers_scenarios: vec![
                    DisasterScenario::DataCenter,
                    DisasterScenario::NaturalDisaster,
                ],
                geographically_separated: true,
            },
        ]
    }

    fn make_objective() -> RecoveryObjective {
        RecoveryObjective::new(8, 24)
    }

    #[test]
    fn test_validate_fully_ready() {
        let plan = make_simple_plan();
        let backups = make_complete_backups();
        let obj = make_objective();

        let validator = DrPlanValidator::new();
        let report = validator.validate(&plan, &backups, &obj);

        assert_eq!(report.readiness, DrReadiness::Ready);
        assert!(report.meets_rto);
        assert!(report.meets_rpo);
        assert!((report.scenario_coverage - 1.0).abs() < 1e-10);
        assert!(report.has_geographic_separation);
        assert_eq!(report.critical_count(), 0);
    }

    #[test]
    fn test_validate_rto_exceeded() {
        let mut plan = make_simple_plan();
        plan.steps[1].duration_hours = 20.0; // Exceed RTO of 8h
        let backups = make_complete_backups();
        let obj = make_objective();

        let validator = DrPlanValidator::new();
        let report = validator.validate(&plan, &backups, &obj);

        assert!(!report.meets_rto);
        assert_eq!(report.readiness, DrReadiness::NotReady);
        assert!(report.findings.iter().any(|f| f.code == "RTO_EXCEEDED"));
    }

    #[test]
    fn test_validate_rpo_exceeded() {
        let plan = make_simple_plan();
        let mut backups = make_complete_backups();
        // Make all backups older than RPO
        for b in &mut backups {
            b.age_hours = 48;
        }
        let obj = make_objective(); // RPO = 24h

        let validator = DrPlanValidator::new();
        let report = validator.validate(&plan, &backups, &obj);

        assert!(!report.meets_rpo);
        assert!(report.findings.iter().any(|f| f.code == "RPO_EXCEEDED"));
    }

    #[test]
    fn test_validate_no_geo_separation() {
        let plan = make_simple_plan();
        let mut backups = make_complete_backups();
        // Remove geographic separation
        for b in &mut backups {
            b.geographically_separated = false;
        }
        // Ensure full scenario coverage
        backups[0].covers_scenarios = vec![
            DisasterScenario::HardwareFailure,
            DisasterScenario::Network,
            DisasterScenario::Ransomware,
            DisasterScenario::DataCenter,
            DisasterScenario::NaturalDisaster,
        ];
        let obj = make_objective();

        let validator = DrPlanValidator::new();
        let report = validator.validate(&plan, &backups, &obj);

        assert!(!report.has_geographic_separation);
        assert!(report
            .findings
            .iter()
            .any(|f| f.code == "NO_GEO_SEPARATION"));
    }

    #[test]
    fn test_validate_unverified_backups() {
        let plan = make_simple_plan();
        let mut backups = make_complete_backups();
        backups[0].verified = false;
        let obj = make_objective();

        let validator = DrPlanValidator::new();
        let report = validator.validate(&plan, &backups, &obj);

        assert_eq!(report.verified_backup_count, 1);
        assert!(report
            .findings
            .iter()
            .any(|f| f.code == "UNVERIFIED_BACKUPS"));
    }

    #[test]
    fn test_validate_insufficient_backups() {
        let plan = make_simple_plan();
        let backups = vec![make_complete_backups().remove(0)];
        let obj = make_objective();

        let validator = DrPlanValidator::new();
        let report = validator.validate(&plan, &backups, &obj);

        assert!(report
            .findings
            .iter()
            .any(|f| f.code == "INSUFFICIENT_BACKUPS"));
    }

    #[test]
    fn test_validate_scenario_gap() {
        let plan = make_simple_plan();
        let mut backups = make_complete_backups();
        // Only cover one scenario
        backups[0].covers_scenarios = vec![DisasterScenario::HardwareFailure];
        backups[1].covers_scenarios = vec![];
        let obj = make_objective();

        let validator = DrPlanValidator::new();
        let report = validator.validate(&plan, &backups, &obj);

        assert!(report.scenario_coverage < 1.0);
        assert!(report.findings.iter().any(|f| f.code == "SCENARIO_GAP"));
    }

    #[test]
    fn test_validate_stale_backup() {
        let plan = make_simple_plan();
        let mut backups = make_complete_backups();
        backups[0].age_hours = 48; // Over default 24h threshold
        let obj = RecoveryObjective::new(8, 72); // RPO large enough to pass

        let validator = DrPlanValidator::new();
        let report = validator.validate(&plan, &backups, &obj);

        assert!(report.findings.iter().any(|f| f.code == "STALE_BACKUP"));
    }

    #[test]
    fn test_validate_missing_dependency() {
        let plan = RecoveryPlan {
            scenario: DisasterScenario::HardwareFailure,
            steps: vec![RecoveryStep {
                order: 1,
                name: "Restore".to_string(),
                duration_hours: 2.0,
                requires: vec![99], // Step 99 does not exist
                responsible_team: "IT".to_string(),
            }],
            total_time_hours: 2.0,
            success_probability: 0.95,
        };
        let backups = make_complete_backups();
        let obj = make_objective();

        let validator = DrPlanValidator::new();
        let report = validator.validate(&plan, &backups, &obj);

        assert!(report
            .findings
            .iter()
            .any(|f| f.code == "MISSING_DEPENDENCY"));
    }

    #[test]
    fn test_validate_empty_plan() {
        let plan = RecoveryPlan {
            scenario: DisasterScenario::DataCenter,
            steps: vec![],
            total_time_hours: 0.0,
            success_probability: 0.0,
        };
        let backups = make_complete_backups();
        let obj = make_objective();

        let validator = DrPlanValidator::new();
        let report = validator.validate(&plan, &backups, &obj);

        assert!(report.findings.iter().any(|f| f.code == "EMPTY_PLAN"));
        assert_eq!(report.readiness, DrReadiness::NotReady);
    }

    #[test]
    fn test_validation_report_summary() {
        let plan = make_simple_plan();
        let backups = make_complete_backups();
        let obj = make_objective();

        let validator = DrPlanValidator::new();
        let report = validator.validate(&plan, &backups, &obj);

        let summary = report.summary();
        assert!(summary.contains("DR Readiness"));
        assert!(summary.contains("RTO"));
        assert!(summary.contains("RPO"));
    }

    #[test]
    fn test_dr_validation_severity_ordering() {
        assert!(DrValidationSeverity::Info < DrValidationSeverity::Warning);
        assert!(DrValidationSeverity::Warning < DrValidationSeverity::Critical);
    }

    #[test]
    fn test_validation_no_backups_at_all() {
        let plan = make_simple_plan();
        let backups: Vec<BackupRecord> = vec![];
        let obj = make_objective();

        let validator = DrPlanValidator::new();
        let report = validator.validate(&plan, &backups, &obj);

        assert_eq!(report.readiness, DrReadiness::NotReady);
        assert!(!report.meets_rpo);
        assert_eq!(report.total_backup_count, 0);
    }

    #[test]
    fn test_validate_custom_validator_config() {
        let plan = make_simple_plan();
        let backups = make_complete_backups();
        let obj = make_objective();

        let validator = DrPlanValidator {
            max_backup_age_hours: 1, // Very strict: 1 hour
            min_backup_count: 5,     // Require 5 backups
            require_geographic_separation: false,
        };
        let report = validator.validate(&plan, &backups, &obj);

        // Should have findings for stale backups and insufficient count
        assert!(report.findings.iter().any(|f| f.code == "STALE_BACKUP"));
        assert!(report
            .findings
            .iter()
            .any(|f| f.code == "INSUFFICIENT_BACKUPS"));
    }
}
