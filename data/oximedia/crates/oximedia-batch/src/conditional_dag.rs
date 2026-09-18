//! Conditional DAG with result-based branch routing.
//!
//! Extends a standard DAG with [`BranchCondition`] edges so that downstream
//! jobs are only scheduled when the upstream outcome satisfies the condition.
//! Edges whose conditions can never be satisfied given the current set of
//! outcomes are classified as *skipped*.

#![allow(dead_code)]

use std::collections::{HashMap, HashSet, VecDeque};

// ─── Branch condition ────────────────────────────────────────────────────────

/// Condition controlling whether a downstream job should be executed.
#[derive(Debug, Clone)]
pub enum BranchCondition {
    /// Execute if the predecessor succeeded.
    OnSuccess,
    /// Execute if the predecessor failed.
    OnFailure,
    /// Execute if the predecessor's string output equals the given value.
    OnOutput(String),
    /// Always execute regardless of the predecessor's outcome (default).
    Always,
    /// Disabled branch – never execute.
    Never,
    /// Execute if the predecessor's numeric output is ≥ the threshold.
    Threshold(f64),
}

// ─── ConditionalEdge ─────────────────────────────────────────────────────────

/// A directed edge with an attached branch condition.
#[derive(Debug, Clone)]
pub struct ConditionalEdge {
    /// Source job ID.
    pub from: String,
    /// Destination job ID.
    pub to: String,
    /// The condition that must hold for `to` to be scheduled.
    pub condition: BranchCondition,
}

// ─── JobOutcome ───────────────────────────────────────────────────────────────

/// The recorded result of a completed job.
#[derive(Debug, Clone)]
pub struct JobOutcome {
    /// Job ID.
    pub job_id: String,
    /// Whether the job succeeded.
    pub success: bool,
    /// Optional string output (used for [`BranchCondition::OnOutput`]).
    pub output: Option<String>,
    /// Optional numeric output (used for [`BranchCondition::Threshold`]).
    pub value: Option<f64>,
}

impl JobOutcome {
    /// Convenience constructor for a successful outcome with no output.
    #[must_use]
    pub fn success(job_id: impl Into<String>) -> Self {
        Self {
            job_id: job_id.into(),
            success: true,
            output: None,
            value: None,
        }
    }

    /// Convenience constructor for a failed outcome.
    #[must_use]
    pub fn failure(job_id: impl Into<String>) -> Self {
        Self {
            job_id: job_id.into(),
            success: false,
            output: None,
            value: None,
        }
    }
}

// ─── ConditionalDag ───────────────────────────────────────────────────────────

/// A DAG of jobs with conditional branch edges.
///
/// A job becomes *ready* when **every** incoming edge whose source has an
/// outcome satisfies its [`BranchCondition`].  A job is *skipped* when at
/// least one incoming edge's condition is permanently unsatisfiable given the
/// recorded outcomes.
#[derive(Debug, Default)]
pub struct ConditionalDag {
    /// All registered job IDs.
    pub jobs: Vec<String>,
    /// All conditional edges.
    pub edges: Vec<ConditionalEdge>,
    /// Recorded outcomes, keyed by job ID.
    pub results: HashMap<String, JobOutcome>,
}

impl ConditionalDag {
    /// Create an empty conditional DAG.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Register a job node (idempotent).
    pub fn add_job(&mut self, id: &str) -> &mut Self {
        if !self.jobs.iter().any(|j| j == id) {
            self.jobs.push(id.to_string());
        }
        self
    }

    /// Add a conditional edge from `from` to `to`.
    ///
    /// Both nodes are registered if they are not already present.
    pub fn add_edge(&mut self, from: &str, to: &str, condition: BranchCondition) -> &mut Self {
        self.add_job(from);
        self.add_job(to);
        self.edges.push(ConditionalEdge {
            from: from.to_string(),
            to: to.to_string(),
            condition,
        });
        self
    }

    /// Record the outcome of a completed job.
    pub fn record_outcome(&mut self, outcome: JobOutcome) {
        self.results.insert(outcome.job_id.clone(), outcome);
    }

    // ── Internal helpers ─────────────────────────────────────────────────────

    /// Check whether a single edge condition is satisfied given the outcome of
    /// the source job.
    fn condition_satisfied(condition: &BranchCondition, outcome: &JobOutcome) -> bool {
        match condition {
            BranchCondition::Always => true,
            BranchCondition::Never => false,
            BranchCondition::OnSuccess => outcome.success,
            BranchCondition::OnFailure => !outcome.success,
            BranchCondition::OnOutput(expected) => {
                outcome.output.as_deref() == Some(expected.as_str())
            }
            BranchCondition::Threshold(threshold) => {
                outcome.value.map_or(false, |v| v >= *threshold)
            }
        }
    }

    /// Check whether a condition can ever be satisfied given that the upstream
    /// job completed with `outcome`.  Returns `false` only when the condition
    /// is provably unsatisfiable (e.g., `OnSuccess` but the job failed).
    fn condition_can_be_satisfied(condition: &BranchCondition, outcome: &JobOutcome) -> bool {
        match condition {
            BranchCondition::Never => false,
            BranchCondition::Always => true,
            BranchCondition::OnSuccess => outcome.success,
            BranchCondition::OnFailure => !outcome.success,
            BranchCondition::OnOutput(expected) => {
                outcome.output.as_deref() == Some(expected.as_str())
            }
            BranchCondition::Threshold(threshold) => {
                outcome.value.map_or(false, |v| v >= *threshold)
            }
        }
    }

    // ── Public query API ─────────────────────────────────────────────────────

    /// Return the IDs of jobs that are ready to run.
    ///
    /// A job is *ready* when:
    /// 1. It has not yet been recorded as completed, **and**
    /// 2. It is not skipped, **and**
    /// 3. For every incoming edge whose source has been recorded, the condition
    ///    is satisfied.
    /// 4. Jobs with no predecessors are always ready (if not yet completed).
    #[must_use]
    pub fn ready_jobs(&self) -> Vec<String> {
        let skipped = self.skipped_jobs_set();
        self.jobs
            .iter()
            .filter(|job_id| {
                // Already completed or skipped → not ready.
                if self.results.contains_key(*job_id) || skipped.contains(*job_id) {
                    return false;
                }
                // Gather incoming edges.
                let incoming: Vec<&ConditionalEdge> =
                    self.edges.iter().filter(|e| &e.to == *job_id).collect();

                // No predecessors → always ready.
                if incoming.is_empty() {
                    return true;
                }

                // All predecessors must have outcomes AND all conditions satisfied.
                incoming.iter().all(|edge| {
                    match self.results.get(&edge.from) {
                        None => false, // predecessor not yet done
                        Some(outcome) => Self::condition_satisfied(&edge.condition, outcome),
                    }
                })
            })
            .cloned()
            .collect()
    }

    /// Internal helper that returns the set of skipped job IDs.
    fn skipped_jobs_set(&self) -> HashSet<String> {
        let mut skipped = HashSet::new();
        // A job is skipped when ANY incoming edge is permanently unsatisfiable.
        for job_id in &self.jobs {
            if self.results.contains_key(job_id) {
                continue; // already ran – not skipped
            }
            let incoming: Vec<&ConditionalEdge> =
                self.edges.iter().filter(|e| &e.to == job_id).collect();

            for edge in &incoming {
                if let Some(outcome) = self.results.get(&edge.from) {
                    if !Self::condition_can_be_satisfied(&edge.condition, outcome) {
                        skipped.insert(job_id.clone());
                        break;
                    }
                }
            }
        }
        skipped
    }

    /// Return the IDs of jobs that are skipped because at least one incoming
    /// edge condition is permanently unsatisfiable.
    #[must_use]
    pub fn skipped_jobs(&self) -> Vec<String> {
        self.skipped_jobs_set().into_iter().collect()
    }

    /// Return `true` when every job either has a recorded outcome or is skipped.
    #[must_use]
    pub fn is_complete(&self) -> bool {
        let skipped = self.skipped_jobs_set();
        self.jobs
            .iter()
            .all(|j| self.results.contains_key(j) || skipped.contains(j))
    }

    /// Detect whether the edge graph contains a cycle (DFS with colouring).
    ///
    /// Returns `true` if a cycle exists.
    #[must_use]
    pub fn has_cycle(&self) -> bool {
        // 0 = white (unvisited), 1 = grey (in stack), 2 = black (done)
        let mut color: HashMap<&str, u8> = HashMap::new();

        for job in &self.jobs {
            if color.get(job.as_str()).copied().unwrap_or(0) == 0
                && self.dfs_cycle(job.as_str(), &mut color)
            {
                return true;
            }
        }
        false
    }

    fn dfs_cycle<'a>(&'a self, node: &'a str, color: &mut HashMap<&'a str, u8>) -> bool {
        color.insert(node, 1);
        for edge in &self.edges {
            if edge.from == node {
                let c = color.get(edge.to.as_str()).copied().unwrap_or(0);
                if c == 1 {
                    return true;
                }
                if c == 0 && self.dfs_cycle(edge.to.as_str(), color) {
                    return true;
                }
            }
        }
        color.insert(node, 2);
        false
    }

    /// Compute a topological ordering using Kahn's algorithm.
    ///
    /// # Errors
    ///
    /// Returns `Err(String)` if the graph contains a cycle.
    pub fn topological_order(&self) -> Result<Vec<String>, String> {
        let mut in_degree: HashMap<&str, usize> =
            self.jobs.iter().map(|j| (j.as_str(), 0usize)).collect();

        for edge in &self.edges {
            *in_degree.entry(edge.to.as_str()).or_insert(0) += 1;
        }

        let mut queue: VecDeque<&str> = in_degree
            .iter()
            .filter(|(_, &d)| d == 0)
            .map(|(&n, _)| n)
            .collect();

        let mut order = Vec::with_capacity(self.jobs.len());

        while let Some(node) = queue.pop_front() {
            order.push(node.to_string());
            for edge in &self.edges {
                if edge.from == node {
                    let deg = in_degree.entry(edge.to.as_str()).or_insert(0);
                    *deg = deg.saturating_sub(1);
                    if *deg == 0 {
                        queue.push_back(edge.to.as_str());
                    }
                }
            }
        }

        if order.len() == self.jobs.len() {
            Ok(order)
        } else {
            Err("Cycle detected in conditional DAG".to_string())
        }
    }
}

// ─── tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── helpers ──────────────────────────────────────────────────────────────

    fn linear_dag() -> ConditionalDag {
        // a --Always--> b --Always--> c
        let mut dag = ConditionalDag::new();
        dag.add_edge("a", "b", BranchCondition::Always);
        dag.add_edge("b", "c", BranchCondition::Always);
        dag
    }

    // ── basic topology ────────────────────────────────────────────────────────

    #[test]
    fn test_root_job_is_always_ready() {
        let dag = linear_dag();
        let ready = dag.ready_jobs();
        assert!(ready.contains(&"a".to_string()));
    }

    #[test]
    fn test_downstream_not_ready_until_predecessor_completes() {
        let dag = linear_dag();
        assert!(!dag.ready_jobs().contains(&"b".to_string()));
    }

    #[test]
    fn test_linear_chain_progresses_correctly() {
        let mut dag = linear_dag();
        dag.record_outcome(JobOutcome::success("a"));
        assert!(dag.ready_jobs().contains(&"b".to_string()));
        dag.record_outcome(JobOutcome::success("b"));
        assert!(dag.ready_jobs().contains(&"c".to_string()));
        dag.record_outcome(JobOutcome::success("c"));
        assert!(dag.ready_jobs().is_empty());
    }

    // ── on_success / on_failure routing ──────────────────────────────────────

    #[test]
    fn test_on_success_routes_correctly() {
        let mut dag = ConditionalDag::new();
        dag.add_edge("a", "b", BranchCondition::OnSuccess);
        dag.add_edge("a", "c", BranchCondition::OnFailure);

        dag.record_outcome(JobOutcome::success("a"));
        let ready = dag.ready_jobs();
        assert!(ready.contains(&"b".to_string()), "b should be ready");
        assert!(!ready.contains(&"c".to_string()), "c should be skipped");
    }

    #[test]
    fn test_on_failure_routes_correctly() {
        let mut dag = ConditionalDag::new();
        dag.add_edge("a", "b", BranchCondition::OnSuccess);
        dag.add_edge("a", "c", BranchCondition::OnFailure);

        dag.record_outcome(JobOutcome::failure("a"));
        let ready = dag.ready_jobs();
        assert!(!ready.contains(&"b".to_string()), "b should be skipped");
        assert!(ready.contains(&"c".to_string()), "c should be ready");
    }

    #[test]
    fn test_skipped_jobs_reported_correctly() {
        let mut dag = ConditionalDag::new();
        dag.add_edge("a", "b", BranchCondition::OnFailure);

        // a succeeded → OnFailure branch can never fire.
        dag.record_outcome(JobOutcome::success("a"));
        let skipped = dag.skipped_jobs();
        assert!(skipped.contains(&"b".to_string()));
    }

    // ── on_output ─────────────────────────────────────────────────────────────

    #[test]
    fn test_on_output_matches() {
        let mut dag = ConditionalDag::new();
        dag.add_edge("a", "b", BranchCondition::OnOutput("ok".to_string()));

        let mut outcome = JobOutcome::success("a");
        outcome.output = Some("ok".to_string());
        dag.record_outcome(outcome);

        assert!(dag.ready_jobs().contains(&"b".to_string()));
    }

    #[test]
    fn test_on_output_no_match_skips() {
        let mut dag = ConditionalDag::new();
        dag.add_edge("a", "b", BranchCondition::OnOutput("ok".to_string()));

        let mut outcome = JobOutcome::success("a");
        outcome.output = Some("error".to_string());
        dag.record_outcome(outcome);

        assert!(dag.skipped_jobs().contains(&"b".to_string()));
    }

    // ── threshold ─────────────────────────────────────────────────────────────

    #[test]
    fn test_threshold_above_fires() {
        let mut dag = ConditionalDag::new();
        dag.add_edge("a", "b", BranchCondition::Threshold(0.8));

        let mut outcome = JobOutcome::success("a");
        outcome.value = Some(0.95);
        dag.record_outcome(outcome);

        assert!(dag.ready_jobs().contains(&"b".to_string()));
    }

    #[test]
    fn test_threshold_below_skips() {
        let mut dag = ConditionalDag::new();
        dag.add_edge("a", "b", BranchCondition::Threshold(0.8));

        let mut outcome = JobOutcome::success("a");
        outcome.value = Some(0.5);
        dag.record_outcome(outcome);

        assert!(dag.skipped_jobs().contains(&"b".to_string()));
    }

    #[test]
    fn test_threshold_at_exact_boundary_fires() {
        let mut dag = ConditionalDag::new();
        dag.add_edge("a", "b", BranchCondition::Threshold(0.5));

        let mut outcome = JobOutcome::success("a");
        outcome.value = Some(0.5);
        dag.record_outcome(outcome);

        assert!(dag.ready_jobs().contains(&"b".to_string()));
    }

    // ── never condition ────────────────────────────────────────────────────────

    #[test]
    fn test_never_condition_skips_immediately() {
        let mut dag = ConditionalDag::new();
        dag.add_edge("a", "b", BranchCondition::Never);

        dag.record_outcome(JobOutcome::success("a"));
        assert!(dag.skipped_jobs().contains(&"b".to_string()));
    }

    // ── is_complete ───────────────────────────────────────────────────────────

    #[test]
    fn test_is_complete_when_all_done_or_skipped() {
        let mut dag = ConditionalDag::new();
        dag.add_edge("a", "b", BranchCondition::OnSuccess);
        dag.add_edge("a", "c", BranchCondition::OnFailure);

        // a succeeds → c is skipped; run b.
        dag.record_outcome(JobOutcome::success("a"));
        dag.record_outcome(JobOutcome::success("b"));

        assert!(dag.is_complete());
    }

    // ── cycle detection ────────────────────────────────────────────────────────

    #[test]
    fn test_no_cycle_in_linear_dag() {
        assert!(!linear_dag().has_cycle());
    }

    #[test]
    fn test_cycle_detected() {
        let mut dag = ConditionalDag::new();
        dag.add_edge("a", "b", BranchCondition::Always);
        dag.add_edge("b", "c", BranchCondition::Always);
        dag.add_edge("c", "a", BranchCondition::Always); // creates cycle
        assert!(dag.has_cycle());
    }

    // ── topological_order ─────────────────────────────────────────────────────

    #[test]
    fn test_topological_order_linear() {
        let dag = linear_dag();
        let order = dag.topological_order().expect("should succeed");
        let ai = order.iter().position(|x| x == "a").expect("a not found");
        let bi = order.iter().position(|x| x == "b").expect("b not found");
        let ci = order.iter().position(|x| x == "c").expect("c not found");
        assert!(ai < bi && bi < ci);
    }

    #[test]
    fn test_topological_order_cycle_errors() {
        let mut dag = ConditionalDag::new();
        dag.add_edge("x", "y", BranchCondition::Always);
        dag.add_edge("y", "x", BranchCondition::Always);
        assert!(dag.topological_order().is_err());
    }
}
