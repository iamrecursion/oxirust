#![allow(dead_code)]
//! Metrics and analytics for review sessions.
//!
//! This module tracks review performance metrics such as response times,
//! comment density, reviewer participation, and resolution rates.
//! It helps teams understand review efficiency and identify bottlenecks.

use std::collections::HashMap;

/// Tracks a single reviewer's participation metrics.
#[derive(Debug, Clone)]
pub struct ReviewerMetrics {
    /// Reviewer identifier.
    pub reviewer_id: String,
    /// Number of comments left by this reviewer.
    pub comment_count: u64,
    /// Number of annotations made.
    pub annotation_count: u64,
    /// Number of issues raised.
    pub issues_raised: u64,
    /// Number of issues resolved.
    pub issues_resolved: u64,
    /// Average response time in milliseconds.
    pub avg_response_time_ms: u64,
    /// Total time spent reviewing in milliseconds.
    pub total_review_time_ms: u64,
}

impl ReviewerMetrics {
    /// Create metrics for a new reviewer.
    #[must_use]
    pub fn new(reviewer_id: &str) -> Self {
        Self {
            reviewer_id: reviewer_id.to_string(),
            comment_count: 0,
            annotation_count: 0,
            issues_raised: 0,
            issues_resolved: 0,
            avg_response_time_ms: 0,
            total_review_time_ms: 0,
        }
    }

    /// Get the issue resolution rate (0.0 to 1.0).
    #[must_use]
    #[allow(clippy::cast_precision_loss)]
    pub fn resolution_rate(&self) -> f64 {
        if self.issues_raised == 0 {
            return 1.0;
        }
        self.issues_resolved as f64 / self.issues_raised as f64
    }

    /// Get the total number of interactions (comments + annotations).
    #[must_use]
    pub fn total_interactions(&self) -> u64 {
        self.comment_count + self.annotation_count
    }
}

/// Time-based statistics for a review session.
#[derive(Debug, Clone)]
pub struct TimeMetrics {
    /// Session creation time in milliseconds since epoch.
    pub created_at_ms: u64,
    /// Time when the first reviewer responded, in milliseconds since epoch.
    pub first_response_ms: Option<u64>,
    /// Time to first response in milliseconds.
    pub time_to_first_response_ms: Option<u64>,
    /// Time when the review was completed, in milliseconds since epoch.
    pub completed_at_ms: Option<u64>,
    /// Total duration from creation to completion in milliseconds.
    pub total_duration_ms: Option<u64>,
    /// Number of rounds/iterations before completion.
    pub iteration_count: u32,
}

impl TimeMetrics {
    /// Create new time metrics.
    #[must_use]
    pub fn new(created_at_ms: u64) -> Self {
        Self {
            created_at_ms,
            first_response_ms: None,
            time_to_first_response_ms: None,
            completed_at_ms: None,
            total_duration_ms: None,
            iteration_count: 0,
        }
    }

    /// Record the first response.
    pub fn record_first_response(&mut self, timestamp_ms: u64) {
        if self.first_response_ms.is_none() {
            self.first_response_ms = Some(timestamp_ms);
            self.time_to_first_response_ms = Some(timestamp_ms.saturating_sub(self.created_at_ms));
        }
    }

    /// Record completion.
    pub fn record_completion(&mut self, timestamp_ms: u64) {
        self.completed_at_ms = Some(timestamp_ms);
        self.total_duration_ms = Some(timestamp_ms.saturating_sub(self.created_at_ms));
    }

    /// Increment the iteration count.
    pub fn increment_iteration(&mut self) {
        self.iteration_count += 1;
    }

    /// Check if the review is completed.
    #[must_use]
    pub fn is_completed(&self) -> bool {
        self.completed_at_ms.is_some()
    }

    /// Get the elapsed time since creation.
    #[must_use]
    pub fn elapsed_ms(&self, current_time_ms: u64) -> u64 {
        current_time_ms.saturating_sub(self.created_at_ms)
    }
}

/// Comment statistics for a review session.
#[derive(Debug, Clone, Default)]
pub struct CommentStats {
    /// Total number of comments.
    pub total: u64,
    /// Number of resolved comments.
    pub resolved: u64,
    /// Number of unresolved comments.
    pub unresolved: u64,
    /// Number of issue-type comments.
    pub issues: u64,
    /// Number of suggestion-type comments.
    pub suggestions: u64,
    /// Number of question-type comments.
    pub questions: u64,
    /// Number of general comments.
    pub general: u64,
}

impl CommentStats {
    /// Create empty comment statistics.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Get the resolution rate (0.0 to 1.0).
    #[must_use]
    #[allow(clippy::cast_precision_loss)]
    pub fn resolution_rate(&self) -> f64 {
        if self.total == 0 {
            return 1.0;
        }
        self.resolved as f64 / self.total as f64
    }

    /// Get the percentage of comments that are issues.
    #[must_use]
    #[allow(clippy::cast_precision_loss)]
    pub fn issue_rate(&self) -> f64 {
        if self.total == 0 {
            return 0.0;
        }
        self.issues as f64 / self.total as f64
    }

    /// Record a new comment by type.
    pub fn record_comment(&mut self, comment_type: CommentType, resolved: bool) {
        self.total += 1;
        if resolved {
            self.resolved += 1;
        } else {
            self.unresolved += 1;
        }
        match comment_type {
            CommentType::Issue => self.issues += 1,
            CommentType::Suggestion => self.suggestions += 1,
            CommentType::Question => self.questions += 1,
            CommentType::General => self.general += 1,
        }
    }
}

/// Type of comment for metrics tracking.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CommentType {
    /// Issue that must be fixed.
    Issue,
    /// Suggestion for improvement.
    Suggestion,
    /// Question requiring clarification.
    Question,
    /// General feedback.
    General,
}

/// Aggregated metrics for a complete review session.
#[derive(Debug, Clone)]
pub struct SessionMetrics {
    /// Session identifier.
    pub session_id: String,
    /// Per-reviewer metrics.
    pub reviewers: HashMap<String, ReviewerMetrics>,
    /// Time-based metrics.
    pub time: TimeMetrics,
    /// Comment statistics.
    pub comments: CommentStats,
    /// Total number of versions reviewed.
    pub version_count: u32,
    /// Number of approval cycles.
    pub approval_cycles: u32,
}

impl SessionMetrics {
    /// Create new session metrics.
    #[must_use]
    pub fn new(session_id: &str, created_at_ms: u64) -> Self {
        Self {
            session_id: session_id.to_string(),
            reviewers: HashMap::new(),
            time: TimeMetrics::new(created_at_ms),
            comments: CommentStats::new(),
            version_count: 0,
            approval_cycles: 0,
        }
    }

    /// Get or create metrics for a reviewer.
    pub fn get_or_create_reviewer(&mut self, reviewer_id: &str) -> &mut ReviewerMetrics {
        self.reviewers
            .entry(reviewer_id.to_string())
            .or_insert_with(|| ReviewerMetrics::new(reviewer_id))
    }

    /// Get the number of active reviewers.
    #[must_use]
    pub fn active_reviewer_count(&self) -> usize {
        self.reviewers
            .values()
            .filter(|r| r.total_interactions() > 0)
            .count()
    }

    /// Get the average comment count per reviewer.
    #[must_use]
    #[allow(clippy::cast_precision_loss)]
    pub fn avg_comments_per_reviewer(&self) -> f64 {
        if self.reviewers.is_empty() {
            return 0.0;
        }
        let total: u64 = self.reviewers.values().map(|r| r.comment_count).sum();
        total as f64 / self.reviewers.len() as f64
    }

    /// Get the most active reviewer by total interactions.
    #[must_use]
    pub fn most_active_reviewer(&self) -> Option<&str> {
        self.reviewers
            .values()
            .max_by_key(|r| r.total_interactions())
            .map(|r| r.reviewer_id.as_str())
    }

    /// Generate a text summary of the metrics.
    #[must_use]
    pub fn summary(&self) -> String {
        let mut s = String::new();
        s.push_str(&format!("Session: {}\n", self.session_id));
        s.push_str(&format!("Reviewers: {}\n", self.reviewers.len()));
        s.push_str(&format!("Comments: {} total\n", self.comments.total));
        s.push_str(&format!(
            "Resolution rate: {:.1}%\n",
            self.comments.resolution_rate() * 100.0
        ));
        s.push_str(&format!("Versions: {}\n", self.version_count));
        s.push_str(&format!("Approval cycles: {}\n", self.approval_cycles));
        if let Some(ttfr) = self.time.time_to_first_response_ms {
            s.push_str(&format!("Time to first response: {} ms\n", ttfr));
        }
        s
    }
}

// ── Cycle time analytics ─────────────────────────────────────────────────────

/// A single review cycle: one round of feedback → revision → re-review.
#[derive(Debug, Clone)]
pub struct ReviewCycle {
    /// Cycle number (1-indexed).
    pub cycle_number: u32,
    /// Timestamp when feedback was submitted (ms since epoch).
    pub feedback_submitted_ms: u64,
    /// Timestamp when the revision was completed (ms since epoch).
    pub revision_completed_ms: Option<u64>,
    /// Timestamp when re-review was approved (ms since epoch).
    pub approved_ms: Option<u64>,
    /// Number of comments open at the start of this cycle.
    pub open_comment_count: usize,
}

impl ReviewCycle {
    /// Create a new cycle.
    #[must_use]
    pub fn new(cycle_number: u32, feedback_submitted_ms: u64, open_comment_count: usize) -> Self {
        Self {
            cycle_number,
            feedback_submitted_ms,
            revision_completed_ms: None,
            approved_ms: None,
            open_comment_count,
        }
    }

    /// Duration from feedback submission to revision completion.
    #[must_use]
    pub fn revision_turnaround_ms(&self) -> Option<u64> {
        self.revision_completed_ms
            .map(|r| r.saturating_sub(self.feedback_submitted_ms))
    }

    /// Duration from revision completion to approval.
    #[must_use]
    pub fn review_turnaround_ms(&self) -> Option<u64> {
        match (self.revision_completed_ms, self.approved_ms) {
            (Some(rev), Some(apr)) => Some(apr.saturating_sub(rev)),
            _ => None,
        }
    }

    /// Total cycle duration from feedback submission to approval.
    #[must_use]
    pub fn total_cycle_ms(&self) -> Option<u64> {
        self.approved_ms
            .map(|a| a.saturating_sub(self.feedback_submitted_ms))
    }
}

/// Aggregated cycle time statistics across multiple review cycles.
#[derive(Debug, Clone, Default)]
pub struct CycleTimeStats {
    /// Individual cycles ordered by cycle number.
    pub cycles: Vec<ReviewCycle>,
}

impl CycleTimeStats {
    /// Create empty statistics.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Add a cycle.
    pub fn add_cycle(&mut self, cycle: ReviewCycle) {
        self.cycles.push(cycle);
    }

    /// Count completed cycles (those with an `approved_ms`).
    #[must_use]
    pub fn completed_count(&self) -> usize {
        self.cycles
            .iter()
            .filter(|c| c.approved_ms.is_some())
            .count()
    }

    /// Mean total cycle duration in ms across completed cycles.
    #[must_use]
    #[allow(clippy::cast_precision_loss)]
    pub fn mean_cycle_ms(&self) -> Option<f64> {
        let completed: Vec<u64> = self
            .cycles
            .iter()
            .filter_map(|c| c.total_cycle_ms())
            .collect();
        if completed.is_empty() {
            return None;
        }
        let sum: u64 = completed.iter().sum();
        Some(sum as f64 / completed.len() as f64)
    }

    /// Minimum total cycle duration among completed cycles.
    #[must_use]
    pub fn min_cycle_ms(&self) -> Option<u64> {
        self.cycles.iter().filter_map(|c| c.total_cycle_ms()).min()
    }

    /// Maximum total cycle duration among completed cycles.
    #[must_use]
    pub fn max_cycle_ms(&self) -> Option<u64> {
        self.cycles.iter().filter_map(|c| c.total_cycle_ms()).max()
    }

    /// Trend: last completed cycle duration minus first (positive = getting slower).
    #[must_use]
    pub fn trend_ms(&self) -> Option<i64> {
        let durations: Vec<u64> = self
            .cycles
            .iter()
            .filter_map(|c| c.total_cycle_ms())
            .collect();
        if durations.len() < 2 {
            return None;
        }
        let first = *durations.first()?;
        let last = *durations.last()?;
        Some(last as i64 - first as i64)
    }

    /// Mean revision turnaround across cycles that have it.
    #[must_use]
    #[allow(clippy::cast_precision_loss)]
    pub fn mean_revision_turnaround_ms(&self) -> Option<f64> {
        let values: Vec<u64> = self
            .cycles
            .iter()
            .filter_map(|c| c.revision_turnaround_ms())
            .collect();
        if values.is_empty() {
            return None;
        }
        Some(values.iter().sum::<u64>() as f64 / values.len() as f64)
    }
}

/// Computes a review health score (0.0 to 100.0).
#[derive(Debug, Clone)]
pub struct ReviewHealthScore {
    /// The computed health score.
    pub score: f64,
    /// Individual factor scores.
    pub factors: HashMap<String, f64>,
}

impl ReviewHealthScore {
    /// Compute the health score from session metrics.
    #[must_use]
    #[allow(clippy::cast_precision_loss)]
    pub fn compute(metrics: &SessionMetrics) -> Self {
        let mut factors = HashMap::new();

        // Factor 1: Resolution rate (0-25 points)
        let resolution_score = metrics.comments.resolution_rate() * 25.0;
        factors.insert("resolution_rate".to_string(), resolution_score);

        // Factor 2: Reviewer participation (0-25 points)
        let participation_score = if metrics.reviewers.is_empty() {
            0.0
        } else {
            let active_ratio =
                metrics.active_reviewer_count() as f64 / metrics.reviewers.len() as f64;
            active_ratio * 25.0
        };
        factors.insert("participation".to_string(), participation_score);

        // Factor 3: Low iteration count (0-25 points)
        let iteration_score = if metrics.approval_cycles == 0 {
            25.0
        } else {
            (25.0 / metrics.approval_cycles as f64).min(25.0)
        };
        factors.insert("iterations".to_string(), iteration_score);

        // Factor 4: Response time (0-25 points)
        let response_score = match metrics.time.time_to_first_response_ms {
            Some(ttfr) if ttfr < 3_600_000 => 25.0,  // < 1 hour
            Some(ttfr) if ttfr < 86_400_000 => 15.0, // < 1 day
            Some(_) => 5.0,
            None => 0.0,
        };
        factors.insert("response_time".to_string(), response_score);

        let score = factors.values().sum::<f64>().min(100.0);

        Self { score, factors }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_reviewer_metrics_new() {
        let m = ReviewerMetrics::new("alice");
        assert_eq!(m.reviewer_id, "alice");
        assert_eq!(m.comment_count, 0);
        assert_eq!(m.total_interactions(), 0);
    }

    #[test]
    fn test_reviewer_resolution_rate_no_issues() {
        let m = ReviewerMetrics::new("alice");
        assert!((m.resolution_rate() - 1.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_reviewer_resolution_rate_partial() {
        let mut m = ReviewerMetrics::new("alice");
        m.issues_raised = 10;
        m.issues_resolved = 7;
        assert!((m.resolution_rate() - 0.7).abs() < f64::EPSILON);
    }

    #[test]
    fn test_reviewer_total_interactions() {
        let mut m = ReviewerMetrics::new("alice");
        m.comment_count = 5;
        m.annotation_count = 3;
        assert_eq!(m.total_interactions(), 8);
    }

    #[test]
    fn test_time_metrics_first_response() {
        let mut t = TimeMetrics::new(1000);
        t.record_first_response(2500);
        assert_eq!(t.time_to_first_response_ms, Some(1500));
        // Second call should not overwrite
        t.record_first_response(5000);
        assert_eq!(t.time_to_first_response_ms, Some(1500));
    }

    #[test]
    fn test_time_metrics_completion() {
        let mut t = TimeMetrics::new(1000);
        assert!(!t.is_completed());
        t.record_completion(5000);
        assert!(t.is_completed());
        assert_eq!(t.total_duration_ms, Some(4000));
    }

    #[test]
    fn test_time_metrics_elapsed() {
        let t = TimeMetrics::new(1000);
        assert_eq!(t.elapsed_ms(3000), 2000);
    }

    #[test]
    fn test_comment_stats_empty() {
        let stats = CommentStats::new();
        assert_eq!(stats.total, 0);
        assert!((stats.resolution_rate() - 1.0).abs() < f64::EPSILON);
        assert!((stats.issue_rate() - 0.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_comment_stats_record() {
        let mut stats = CommentStats::new();
        stats.record_comment(CommentType::Issue, false);
        stats.record_comment(CommentType::Issue, true);
        stats.record_comment(CommentType::Suggestion, false);
        stats.record_comment(CommentType::General, true);
        assert_eq!(stats.total, 4);
        assert_eq!(stats.resolved, 2);
        assert_eq!(stats.unresolved, 2);
        assert_eq!(stats.issues, 2);
        assert_eq!(stats.suggestions, 1);
        assert_eq!(stats.general, 1);
        assert_eq!(stats.questions, 0);
    }

    #[test]
    fn test_session_metrics_creation() {
        let m = SessionMetrics::new("session-1", 1000);
        assert_eq!(m.session_id, "session-1");
        assert_eq!(m.reviewers.len(), 0);
        assert_eq!(m.version_count, 0);
    }

    #[test]
    fn test_session_metrics_reviewer() {
        let mut m = SessionMetrics::new("session-1", 1000);
        {
            let r = m.get_or_create_reviewer("alice");
            r.comment_count = 3;
        }
        {
            let r = m.get_or_create_reviewer("bob");
            r.comment_count = 5;
        }
        assert_eq!(m.reviewers.len(), 2);
        assert_eq!(m.active_reviewer_count(), 2);
    }

    #[test]
    fn test_session_avg_comments() {
        let mut m = SessionMetrics::new("s1", 0);
        m.get_or_create_reviewer("a").comment_count = 4;
        m.get_or_create_reviewer("b").comment_count = 6;
        assert!((m.avg_comments_per_reviewer() - 5.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_most_active_reviewer() {
        let mut m = SessionMetrics::new("s1", 0);
        m.get_or_create_reviewer("a").comment_count = 2;
        m.get_or_create_reviewer("b").comment_count = 10;
        assert_eq!(m.most_active_reviewer(), Some("b"));
    }

    #[test]
    fn test_session_summary() {
        let m = SessionMetrics::new("s1", 0);
        let summary = m.summary();
        assert!(summary.contains("Session: s1"));
        assert!(summary.contains("Reviewers: 0"));
    }

    #[test]
    fn test_review_health_score() {
        let mut m = SessionMetrics::new("s1", 0);
        m.time.record_first_response(1000); // Very fast response
        m.comments.record_comment(CommentType::Issue, true);
        m.get_or_create_reviewer("a").comment_count = 1;
        let health = ReviewHealthScore::compute(&m);
        assert!(health.score > 0.0);
        assert!(health.score <= 100.0);
        assert!(health.factors.contains_key("resolution_rate"));
        assert!(health.factors.contains_key("participation"));
        assert!(health.factors.contains_key("response_time"));
    }

    // ── Cycle time analytics tests ────────────────────────────────────────────

    fn make_complete_cycle(n: u32, start: u64, rev: u64, approve: u64) -> ReviewCycle {
        let mut c = ReviewCycle::new(n, start, 5);
        c.revision_completed_ms = Some(rev);
        c.approved_ms = Some(approve);
        c
    }

    #[test]
    fn test_cycle_revision_turnaround() {
        let c = make_complete_cycle(1, 1000, 3000, 5000);
        assert_eq!(c.revision_turnaround_ms(), Some(2000));
    }

    #[test]
    fn test_cycle_review_turnaround() {
        let c = make_complete_cycle(1, 1000, 3000, 5000);
        assert_eq!(c.review_turnaround_ms(), Some(2000));
    }

    #[test]
    fn test_cycle_total_cycle_ms() {
        let c = make_complete_cycle(1, 1000, 3000, 5000);
        assert_eq!(c.total_cycle_ms(), Some(4000));
    }

    #[test]
    fn test_cycle_no_approval_gives_none() {
        let c = ReviewCycle::new(1, 0, 3);
        assert!(c.total_cycle_ms().is_none());
        assert!(c.review_turnaround_ms().is_none());
    }

    #[test]
    fn test_cycle_stats_empty() {
        let stats = CycleTimeStats::new();
        assert_eq!(stats.completed_count(), 0);
        assert!(stats.mean_cycle_ms().is_none());
        assert!(stats.min_cycle_ms().is_none());
        assert!(stats.max_cycle_ms().is_none());
        assert!(stats.trend_ms().is_none());
    }

    #[test]
    fn test_cycle_stats_mean() {
        let mut stats = CycleTimeStats::new();
        stats.add_cycle(make_complete_cycle(1, 0, 1000, 2000)); // 2000
        stats.add_cycle(make_complete_cycle(2, 0, 1000, 4000)); // 4000
        let mean = stats.mean_cycle_ms().expect("has completed cycles");
        assert!((mean - 3000.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_cycle_stats_min_max() {
        let mut stats = CycleTimeStats::new();
        stats.add_cycle(make_complete_cycle(1, 0, 500, 1000)); // 1000
        stats.add_cycle(make_complete_cycle(2, 0, 500, 3000)); // 3000
        assert_eq!(stats.min_cycle_ms(), Some(1000));
        assert_eq!(stats.max_cycle_ms(), Some(3000));
    }

    #[test]
    fn test_cycle_stats_trend_improving() {
        let mut stats = CycleTimeStats::new();
        stats.add_cycle(make_complete_cycle(1, 0, 2000, 5000)); // 5000
        stats.add_cycle(make_complete_cycle(2, 0, 1000, 3000)); // 3000
        let trend = stats.trend_ms().expect("two cycles");
        assert!(trend < 0, "trend should be negative (improving)");
    }

    #[test]
    fn test_cycle_stats_trend_single_cycle_is_none() {
        let mut stats = CycleTimeStats::new();
        stats.add_cycle(make_complete_cycle(1, 0, 1000, 5000));
        assert!(stats.trend_ms().is_none());
    }

    #[test]
    fn test_cycle_stats_mean_revision_turnaround() {
        let mut stats = CycleTimeStats::new();
        stats.add_cycle(make_complete_cycle(1, 0, 2000, 5000)); // turnaround=2000
        stats.add_cycle(make_complete_cycle(2, 0, 4000, 6000)); // turnaround=4000
        let mean = stats.mean_revision_turnaround_ms().expect("has data");
        assert!((mean - 3000.0).abs() < f64::EPSILON);
    }
}
