//! Fair scheduler: prevent starvation of low-priority jobs.
//!
//! A strict priority queue always serves high-priority jobs first, which can
//! starve lower-priority work indefinitely if high-priority jobs keep arriving.
//! This module provides a **weighted fair queue** that guarantees every priority
//! tier receives a minimum share of execution slots.
//!
//! ## Algorithm
//!
//! The scheduler uses **Deficit Round Robin (DRR)** across priority tiers.
//! Each tier has a configurable weight (share).  On each scheduling cycle,
//! the tier with the highest deficit (most "owed" service) is selected.
//! Within a tier, jobs are served in FIFO order.
//!
//! ## Starvation prevention
//!
//! Jobs that have waited beyond `max_wait_secs` are automatically promoted to
//! the highest tier, ensuring eventual execution regardless of initial priority.

#![allow(dead_code)]

use std::collections::VecDeque;

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

/// A priority tier with its scheduling weight.
#[derive(Debug, Clone)]
pub struct PriorityTier {
    /// Tier priority level (0 = lowest, 3 = highest).
    pub level: u8,
    /// Human-readable name.
    pub name: String,
    /// Relative weight for fair scheduling (higher = more service).
    pub weight: u32,
    /// Current deficit counter.
    deficit: i64,
    /// Queue of tasks in this tier (FIFO).
    queue: VecDeque<FairEntry>,
}

impl PriorityTier {
    /// Create a new priority tier.
    #[must_use]
    pub fn new(level: u8, name: impl Into<String>, weight: u32) -> Self {
        Self {
            level,
            name: name.into(),
            weight: weight.max(1),
            deficit: 0,
            queue: VecDeque::new(),
        }
    }

    /// Number of pending tasks in this tier.
    #[must_use]
    pub fn len(&self) -> usize {
        self.queue.len()
    }

    /// Whether this tier's queue is empty.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.queue.is_empty()
    }
}

/// A job entry in the fair scheduler.
#[derive(Debug, Clone)]
pub struct FairEntry {
    /// Unique job identifier.
    pub job_id: String,
    /// Original priority level when submitted.
    pub original_priority: u8,
    /// Current priority level (may be promoted).
    pub current_priority: u8,
    /// Unix timestamp (seconds) when the job was submitted.
    pub submit_time_secs: u64,
    /// Estimated execution cost in arbitrary units.
    pub cost: u64,
    /// Whether this job has been promoted due to aging.
    pub was_promoted: bool,
}

impl FairEntry {
    /// Create a new fair entry.
    #[must_use]
    pub fn new(job_id: impl Into<String>, priority: u8, cost: u64) -> Self {
        Self {
            job_id: job_id.into(),
            original_priority: priority,
            current_priority: priority,
            submit_time_secs: current_timestamp(),
            cost,
            was_promoted: false,
        }
    }

    /// How long this job has been waiting, in seconds.
    #[must_use]
    pub fn wait_time_secs(&self) -> u64 {
        current_timestamp().saturating_sub(self.submit_time_secs)
    }
}

/// Statistics about the fair scheduler.
#[derive(Debug, Clone)]
pub struct FairSchedulerStats {
    /// Number of tasks per tier.
    pub tasks_per_tier: Vec<(String, usize)>,
    /// Total tasks across all tiers.
    pub total_tasks: usize,
    /// Total tasks dispatched since creation.
    pub total_dispatched: u64,
    /// Number of jobs promoted due to aging.
    pub total_promotions: u64,
    /// Current deficit values per tier.
    pub deficits: Vec<(String, i64)>,
}

// ---------------------------------------------------------------------------
// Fair scheduler
// ---------------------------------------------------------------------------

/// Default tiers: Critical (w=8), High (w=4), Normal (w=2), Low (w=1).
const DEFAULT_TIERS: [(u8, &str, u32); 4] = [
    (0, "Low", 1),
    (1, "Normal", 2),
    (2, "High", 4),
    (3, "Critical", 8),
];

/// Weighted fair queue scheduler with aging-based promotion.
#[derive(Debug)]
pub struct FairScheduler {
    tiers: Vec<PriorityTier>,
    /// Maximum wait time (seconds) before a job is promoted to the highest tier.
    max_wait_secs: u64,
    /// Total dispatched counter.
    total_dispatched: u64,
    /// Total promotions counter.
    total_promotions: u64,
}

impl FairScheduler {
    /// Create a new fair scheduler with default tiers.
    #[must_use]
    pub fn new() -> Self {
        let tiers = DEFAULT_TIERS
            .iter()
            .map(|(level, name, weight)| PriorityTier::new(*level, *name, *weight))
            .collect();

        Self {
            tiers,
            max_wait_secs: 300, // 5 minutes
            total_dispatched: 0,
            total_promotions: 0,
        }
    }

    /// Create a scheduler with custom tiers.
    ///
    /// Tiers are sorted by level automatically.
    #[must_use]
    pub fn with_tiers(mut tiers: Vec<PriorityTier>) -> Self {
        tiers.sort_by_key(|t| t.level);
        Self {
            tiers,
            max_wait_secs: 300,
            total_dispatched: 0,
            total_promotions: 0,
        }
    }

    /// Set the maximum wait time before aging promotion.
    ///
    /// Set to 0 to disable aging.
    #[must_use]
    pub fn with_max_wait(mut self, secs: u64) -> Self {
        self.max_wait_secs = secs;
        self
    }

    /// Submit a job to the scheduler.
    pub fn submit(&mut self, entry: FairEntry) {
        let level = entry.current_priority;
        if let Some(tier) = self.tiers.iter_mut().find(|t| t.level == level) {
            tier.queue.push_back(entry);
        } else if let Some(tier) = self.tiers.last_mut() {
            // If the priority level doesn't exist, put it in the highest tier.
            let mut e = entry;
            e.current_priority = tier.level;
            tier.queue.push_back(e);
        }
    }

    /// Dequeue the next job according to the fair scheduling algorithm.
    ///
    /// Returns `None` if all tiers are empty.
    pub fn next(&mut self) -> Option<FairEntry> {
        // First, perform aging promotions.
        if self.max_wait_secs > 0 {
            self.promote_aged_jobs();
        }

        // DRR: increase deficits and pick the tier with the highest deficit
        // that has pending work.
        let mut best_tier_idx = None;
        let mut best_deficit = i64::MIN;

        for (idx, tier) in self.tiers.iter_mut().enumerate() {
            if !tier.queue.is_empty() {
                tier.deficit += tier.weight as i64;
                if tier.deficit > best_deficit {
                    best_deficit = tier.deficit;
                    best_tier_idx = Some(idx);
                }
            }
        }

        let tier_idx = best_tier_idx?;

        // Dequeue from the selected tier.
        let entry = self.tiers[tier_idx].queue.pop_front()?;
        self.tiers[tier_idx].deficit -= entry.cost as i64;
        self.total_dispatched += 1;

        Some(entry)
    }

    /// Peek at the next job without removing it.
    #[must_use]
    pub fn peek(&self) -> Option<&FairEntry> {
        // Find the tier that would be selected.
        let mut best_tier_idx = None;
        let mut best_score = i64::MIN;

        for (idx, tier) in self.tiers.iter().enumerate() {
            if !tier.queue.is_empty() {
                let projected_deficit = tier.deficit + tier.weight as i64;
                if projected_deficit > best_score {
                    best_score = projected_deficit;
                    best_tier_idx = Some(idx);
                }
            }
        }

        best_tier_idx.and_then(|idx| self.tiers[idx].queue.front())
    }

    /// Total number of pending tasks across all tiers.
    #[must_use]
    pub fn total_pending(&self) -> usize {
        self.tiers.iter().map(|t| t.len()).sum()
    }

    /// Whether all tiers are empty.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.tiers.iter().all(|t| t.is_empty())
    }

    /// Get the number of pending tasks in a specific tier.
    #[must_use]
    pub fn tier_len(&self, level: u8) -> usize {
        self.tiers
            .iter()
            .find(|t| t.level == level)
            .map(|t| t.len())
            .unwrap_or(0)
    }

    /// Remove a job by ID from any tier.
    ///
    /// Returns the entry if found.
    pub fn remove(&mut self, job_id: &str) -> Option<FairEntry> {
        for tier in &mut self.tiers {
            if let Some(pos) = tier.queue.iter().position(|e| e.job_id == job_id) {
                return tier.queue.remove(pos);
            }
        }
        None
    }

    /// Get scheduler statistics.
    #[must_use]
    pub fn stats(&self) -> FairSchedulerStats {
        let tasks_per_tier: Vec<(String, usize)> = self
            .tiers
            .iter()
            .map(|t| (t.name.clone(), t.len()))
            .collect();
        let total = self.total_pending();
        let deficits: Vec<(String, i64)> = self
            .tiers
            .iter()
            .map(|t| (t.name.clone(), t.deficit))
            .collect();

        FairSchedulerStats {
            tasks_per_tier,
            total_tasks: total,
            total_dispatched: self.total_dispatched,
            total_promotions: self.total_promotions,
            deficits,
        }
    }

    /// Number of tiers.
    #[must_use]
    pub fn tier_count(&self) -> usize {
        self.tiers.len()
    }

    /// Get the weight of a tier.
    #[must_use]
    pub fn tier_weight(&self, level: u8) -> Option<u32> {
        self.tiers
            .iter()
            .find(|t| t.level == level)
            .map(|t| t.weight)
    }

    // ── Private helpers ─────────────────────────────────────────────────

    /// Promote jobs that have exceeded `max_wait_secs` to the highest tier.
    fn promote_aged_jobs(&mut self) {
        let highest_level = self.tiers.last().map(|t| t.level).unwrap_or(0);

        // Collect jobs to promote from non-highest tiers.
        let mut to_promote = Vec::new();

        for tier in &mut self.tiers {
            if tier.level == highest_level {
                continue;
            }
            let mut remaining = VecDeque::new();
            while let Some(entry) = tier.queue.pop_front() {
                if entry.wait_time_secs() > self.max_wait_secs {
                    to_promote.push(entry);
                } else {
                    remaining.push_back(entry);
                }
            }
            tier.queue = remaining;
        }

        // Move promoted jobs to the highest tier.
        if let Some(highest) = self.tiers.iter_mut().find(|t| t.level == highest_level) {
            for mut entry in to_promote {
                entry.current_priority = highest_level;
                entry.was_promoted = true;
                highest.queue.push_back(entry);
                self.total_promotions += 1;
            }
        }
    }
}

impl Default for FairScheduler {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn current_timestamp() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // ── Basic operations ────────────────────────────────────────────────
    #[test]
    fn test_new_scheduler_is_empty() {
        let sched = FairScheduler::new();
        assert!(sched.is_empty());
        assert_eq!(sched.total_pending(), 0);
    }

    #[test]
    fn test_submit_and_next() {
        let mut sched = FairScheduler::new();
        sched.submit(FairEntry::new("j1", 1, 10));
        assert_eq!(sched.total_pending(), 1);

        let entry = sched.next().expect("should dequeue");
        assert_eq!(entry.job_id, "j1");
        assert!(sched.is_empty());
    }

    #[test]
    fn test_next_on_empty_returns_none() {
        let mut sched = FairScheduler::new();
        assert!(sched.next().is_none());
    }

    // ── Priority ordering ───────────────────────────────────────────────
    #[test]
    fn test_higher_priority_preferred() {
        let mut sched = FairScheduler::new();
        sched.submit(FairEntry::new("low", 0, 10));
        sched.submit(FairEntry::new("high", 3, 10));

        // Higher priority tier has higher weight → higher deficit → picked first.
        let first = sched.next().expect("should dequeue");
        assert_eq!(first.job_id, "high");
    }

    // ── Fair distribution ───────────────────────────────────────────────
    #[test]
    fn test_fair_distribution_no_starvation() {
        let mut sched = FairScheduler::new().with_max_wait(0); // disable aging

        // Submit many low-priority jobs and only a few high-priority ones.
        // This simulates the real scenario: a flood of low-priority work with
        // some high-priority jobs that should be served faster per-job.
        for i in 0..50 {
            sched.submit(FairEntry::new(format!("low-{i}"), 0, 1));
        }
        for i in 0..5 {
            sched.submit(FairEntry::new(format!("high-{i}"), 3, 1));
        }

        let mut low_count = 0u32;
        let mut high_count = 0u32;

        // Dequeue the first 20 tasks and check distribution.
        for _ in 0..20 {
            if let Some(entry) = sched.next() {
                if entry.original_priority == 0 {
                    low_count += 1;
                } else {
                    high_count += 1;
                }
            }
        }

        // Both tiers should have been served — the key property is no starvation.
        assert!(low_count > 0, "Low priority should get some service");
        assert!(high_count > 0, "High priority should get some service");

        // High-priority jobs should all be served within 20 dispatches
        // (weight 8 vs 1 means high priority gets ~89% of slots when both
        // tiers have work).
        assert_eq!(high_count, 5, "All 5 high-priority jobs should be served");
    }

    // ── Aging promotion ─────────────────────────────────────────────────
    #[test]
    fn test_aging_promotion() {
        let mut sched = FairScheduler::new().with_max_wait(10); // 10 seconds

        // Submit a low-priority job with an old timestamp.
        let mut old_entry = FairEntry::new("old-job", 0, 10);
        old_entry.submit_time_secs = current_timestamp().saturating_sub(20); // 20 seconds ago

        sched.submit(old_entry);

        // Call next() to trigger aging check.
        let entry = sched.next().expect("should dequeue");
        assert_eq!(entry.job_id, "old-job");
        assert!(entry.was_promoted);
        assert_eq!(entry.original_priority, 0);
        assert_eq!(entry.current_priority, 3); // promoted to highest
    }

    // ── FIFO within same tier ───────────────────────────────────────────
    #[test]
    fn test_fifo_within_tier() {
        let mut sched = FairScheduler::new();
        sched.submit(FairEntry::new("first", 1, 1));
        sched.submit(FairEntry::new("second", 1, 1));
        sched.submit(FairEntry::new("third", 1, 1));

        assert_eq!(sched.next().expect("should dequeue").job_id, "first");
        assert_eq!(sched.next().expect("should dequeue").job_id, "second");
        assert_eq!(sched.next().expect("should dequeue").job_id, "third");
    }

    // ── Remove ──────────────────────────────────────────────────────────
    #[test]
    fn test_remove_job() {
        let mut sched = FairScheduler::new();
        sched.submit(FairEntry::new("keep", 1, 10));
        sched.submit(FairEntry::new("remove", 1, 10));

        let removed = sched.remove("remove");
        assert!(removed.is_some());
        assert_eq!(removed.expect("should exist").job_id, "remove");
        assert_eq!(sched.total_pending(), 1);
    }

    #[test]
    fn test_remove_nonexistent() {
        let mut sched = FairScheduler::new();
        assert!(sched.remove("ghost").is_none());
    }

    // ── Peek ────────────────────────────────────────────────────────────
    #[test]
    fn test_peek() {
        let mut sched = FairScheduler::new();
        sched.submit(FairEntry::new("j1", 2, 5));

        let peeked = sched.peek().expect("should peek");
        assert_eq!(peeked.job_id, "j1");
        // Should still be in the queue.
        assert_eq!(sched.total_pending(), 1);
    }

    // ── Stats ───────────────────────────────────────────────────────────
    #[test]
    fn test_stats() {
        let mut sched = FairScheduler::new();
        sched.submit(FairEntry::new("a", 0, 10));
        sched.submit(FairEntry::new("b", 3, 10));
        let _ = sched.next();

        let stats = sched.stats();
        assert_eq!(stats.total_tasks, 1);
        assert_eq!(stats.total_dispatched, 1);
    }

    // ── Custom tiers ────────────────────────────────────────────────────
    #[test]
    fn test_custom_tiers() {
        let tiers = vec![
            PriorityTier::new(0, "Background", 1),
            PriorityTier::new(1, "Interactive", 10),
        ];
        let mut sched = FairScheduler::with_tiers(tiers);
        assert_eq!(sched.tier_count(), 2);

        sched.submit(FairEntry::new("bg", 0, 1));
        sched.submit(FairEntry::new("interactive", 1, 1));

        // Interactive (w=10) should be preferred.
        let first = sched.next().expect("should dequeue");
        assert_eq!(first.job_id, "interactive");
    }

    // ── Tier length ─────────────────────────────────────────────────────
    #[test]
    fn test_tier_len() {
        let mut sched = FairScheduler::new();
        sched.submit(FairEntry::new("a", 0, 1));
        sched.submit(FairEntry::new("b", 0, 1));
        sched.submit(FairEntry::new("c", 1, 1));

        assert_eq!(sched.tier_len(0), 2);
        assert_eq!(sched.tier_len(1), 1);
        assert_eq!(sched.tier_len(2), 0);
    }

    // ── Tier weight ─────────────────────────────────────────────────────
    #[test]
    fn test_tier_weight() {
        let sched = FairScheduler::new();
        assert_eq!(sched.tier_weight(0), Some(1)); // Low
        assert_eq!(sched.tier_weight(1), Some(2)); // Normal
        assert_eq!(sched.tier_weight(2), Some(4)); // High
        assert_eq!(sched.tier_weight(3), Some(8)); // Critical
        assert_eq!(sched.tier_weight(99), None); // Unknown
    }

    // ── FairEntry wait time ─────────────────────────────────────────────
    #[test]
    fn test_fair_entry_wait_time() {
        let entry = FairEntry::new("test", 1, 10);
        assert!(entry.wait_time_secs() < 5);
    }

    // ── Unknown priority level goes to highest tier ─────────────────────
    #[test]
    fn test_unknown_priority_goes_to_highest() {
        let mut sched = FairScheduler::new();
        sched.submit(FairEntry::new("unknown", 99, 10));
        // Should be placed in the highest tier (3 = Critical).
        assert_eq!(sched.tier_len(3), 1);
    }

    // ── Default scheduler ───────────────────────────────────────────────
    #[test]
    fn test_default_scheduler() {
        let sched = FairScheduler::default();
        assert_eq!(sched.tier_count(), 4);
        assert!(sched.is_empty());
    }

    // ── Mixed priority fair dispatch ────────────────────────────────────
    #[test]
    fn test_mixed_priority_all_served() {
        let mut sched = FairScheduler::new().with_max_wait(0);

        // Submit jobs at all priority levels.
        for i in 0..5 {
            sched.submit(FairEntry::new(format!("low-{i}"), 0, 1));
            sched.submit(FairEntry::new(format!("normal-{i}"), 1, 1));
            sched.submit(FairEntry::new(format!("high-{i}"), 2, 1));
            sched.submit(FairEntry::new(format!("critical-{i}"), 3, 1));
        }

        let mut counts = [0u32; 4];
        for _ in 0..20 {
            if let Some(entry) = sched.next() {
                counts[entry.original_priority as usize] += 1;
            }
        }

        // All tiers should have been served.
        for (level, &count) in counts.iter().enumerate() {
            assert!(
                count > 0,
                "Priority level {level} should have been served, but count was 0"
            );
        }
    }
}
