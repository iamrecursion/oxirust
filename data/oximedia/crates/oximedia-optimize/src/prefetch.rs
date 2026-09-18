//! Content prefetching optimization for media delivery pipelines.
//!
//! Provides a priority queue of prefetch hints that schedulers can use to
//! proactively load assets before they are requested by a client.

#![allow(dead_code)]

/// Strategy that determined why an asset should be prefetched.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PrefetchStrategy {
    /// Assets are fetched in sequential playlist order.
    Sequential,
    /// A predictive model (e.g. ML-based) suggested the asset.
    Predictive,
    /// Personalization data for the current user suggests the asset.
    UserBased,
    /// The asset is historically popular at this time of day.
    TimeOfDay,
}

impl PrefetchStrategy {
    /// Returns `true` if the strategy relies on user-specific data.
    #[must_use]
    pub fn is_personalized(&self) -> bool {
        matches!(self, Self::UserBased | Self::Predictive)
    }
}

/// A single hint requesting that an asset be prefetched.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PrefetchHint {
    /// Unique asset identifier.
    pub asset_id: u64,
    /// Priority (0 = lowest, 255 = highest).
    pub priority: u8,
    /// Strategy that produced this hint.
    pub reason: PrefetchStrategy,
    /// Expected Unix epoch when the asset will be accessed.
    pub estimated_access_epoch: u64,
}

impl PrefetchHint {
    /// Returns `true` if the estimated access time is within the next 60 seconds.
    #[must_use]
    pub fn is_urgent(&self, now: u64) -> bool {
        self.estimated_access_epoch > now && self.estimated_access_epoch - now <= 60
    }
}

/// A bounded priority queue of prefetch hints.
#[derive(Debug, Clone)]
pub struct PrefetchQueue {
    /// Current set of prefetch hints.
    pub hints: Vec<PrefetchHint>,
    /// Maximum number of hints allowed in the queue at one time.
    pub max_concurrent: u8,
}

impl PrefetchQueue {
    /// Creates a new `PrefetchQueue` with the given concurrency limit.
    #[must_use]
    pub fn new(max_concurrent: u8) -> Self {
        Self {
            hints: Vec::new(),
            max_concurrent,
        }
    }

    /// Adds a hint if the queue is not full and the asset is not already queued.
    pub fn add(&mut self, hint: PrefetchHint) {
        if self.is_full() {
            return;
        }
        if self.hints.iter().any(|h| h.asset_id == hint.asset_id) {
            return;
        }
        self.hints.push(hint);
    }

    /// Returns the top `n` hints sorted by priority descending.
    #[must_use]
    pub fn top_priority_hints(&self, n: usize) -> Vec<&PrefetchHint> {
        let mut sorted: Vec<&PrefetchHint> = self.hints.iter().collect();
        sorted.sort_by(|a, b| b.priority.cmp(&a.priority));
        sorted.truncate(n);
        sorted
    }

    /// Removes the hint for the given asset ID (e.g. after the asset has been loaded).
    pub fn remove(&mut self, asset_id: u64) {
        self.hints.retain(|h| h.asset_id != asset_id);
    }

    /// Returns the number of hints currently in the queue.
    #[must_use]
    pub fn pending_count(&self) -> usize {
        self.hints.len()
    }

    /// Returns `true` if the queue has reached its concurrency limit.
    #[must_use]
    pub fn is_full(&self) -> bool {
        self.hints.len() >= self.max_concurrent as usize
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_hint(id: u64, priority: u8, reason: PrefetchStrategy, access: u64) -> PrefetchHint {
        PrefetchHint {
            asset_id: id,
            priority,
            reason,
            estimated_access_epoch: access,
        }
    }

    #[test]
    fn test_sequential_not_personalized() {
        assert!(!PrefetchStrategy::Sequential.is_personalized());
    }

    #[test]
    fn test_user_based_is_personalized() {
        assert!(PrefetchStrategy::UserBased.is_personalized());
    }

    #[test]
    fn test_predictive_is_personalized() {
        assert!(PrefetchStrategy::Predictive.is_personalized());
    }

    #[test]
    fn test_time_of_day_not_personalized() {
        assert!(!PrefetchStrategy::TimeOfDay.is_personalized());
    }

    #[test]
    fn test_is_urgent_within_60s() {
        let hint = make_hint(1, 100, PrefetchStrategy::Sequential, 1059);
        assert!(hint.is_urgent(1000));
    }

    #[test]
    fn test_is_not_urgent_after_60s() {
        let hint = make_hint(1, 100, PrefetchStrategy::Sequential, 1061);
        assert!(!hint.is_urgent(1000));
    }

    #[test]
    fn test_is_not_urgent_past_due() {
        // estimated_access_epoch <= now
        let hint = make_hint(1, 100, PrefetchStrategy::Sequential, 999);
        assert!(!hint.is_urgent(1000));
    }

    #[test]
    fn test_queue_add_and_pending_count() {
        let mut q = PrefetchQueue::new(10);
        q.add(make_hint(1, 50, PrefetchStrategy::Sequential, 2000));
        q.add(make_hint(2, 80, PrefetchStrategy::Predictive, 2010));
        assert_eq!(q.pending_count(), 2);
    }

    #[test]
    fn test_queue_no_duplicate_asset_ids() {
        let mut q = PrefetchQueue::new(10);
        q.add(make_hint(42, 50, PrefetchStrategy::Sequential, 2000));
        q.add(make_hint(42, 90, PrefetchStrategy::UserBased, 2001));
        assert_eq!(q.pending_count(), 1);
    }

    #[test]
    fn test_queue_is_full() {
        let mut q = PrefetchQueue::new(2);
        q.add(make_hint(1, 50, PrefetchStrategy::Sequential, 2000));
        q.add(make_hint(2, 60, PrefetchStrategy::Sequential, 2001));
        assert!(q.is_full());
    }

    #[test]
    fn test_queue_does_not_exceed_max() {
        let mut q = PrefetchQueue::new(2);
        q.add(make_hint(1, 50, PrefetchStrategy::Sequential, 2000));
        q.add(make_hint(2, 60, PrefetchStrategy::Sequential, 2001));
        q.add(make_hint(3, 70, PrefetchStrategy::Sequential, 2002)); // should be rejected
        assert_eq!(q.pending_count(), 2);
    }

    #[test]
    fn test_queue_remove() {
        let mut q = PrefetchQueue::new(10);
        q.add(make_hint(1, 50, PrefetchStrategy::Sequential, 2000));
        q.add(make_hint(2, 80, PrefetchStrategy::Sequential, 2001));
        q.remove(1);
        assert_eq!(q.pending_count(), 1);
        assert_eq!(q.hints[0].asset_id, 2);
    }

    #[test]
    fn test_top_priority_hints_ordering() {
        let mut q = PrefetchQueue::new(10);
        q.add(make_hint(1, 10, PrefetchStrategy::Sequential, 2000));
        q.add(make_hint(2, 90, PrefetchStrategy::UserBased, 2001));
        q.add(make_hint(3, 50, PrefetchStrategy::TimeOfDay, 2002));
        let top = q.top_priority_hints(2);
        assert_eq!(top[0].asset_id, 2); // priority 90
        assert_eq!(top[1].asset_id, 3); // priority 50
    }

    #[test]
    fn test_top_priority_hints_clamp_n() {
        let mut q = PrefetchQueue::new(10);
        q.add(make_hint(1, 10, PrefetchStrategy::Sequential, 2000));
        let top = q.top_priority_hints(100); // n > queue size
        assert_eq!(top.len(), 1);
    }
}
