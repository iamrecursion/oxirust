//! Lock contention detection.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::time::{Duration, Instant};

/// Contention event.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ContentionEvent {
    /// Lock identifier.
    pub lock_id: u64,

    /// Lock name.
    pub lock_name: String,

    /// Thread ID that experienced contention.
    pub thread_id: u64,

    /// Wait time.
    pub wait_time: Duration,

    /// Timestamp.
    #[serde(skip, default = "Instant::now")]
    pub timestamp: Instant,
}

/// Contention detector.
#[derive(Debug)]
pub struct ContentionDetector {
    events: Vec<ContentionEvent>,
    lock_stats: HashMap<u64, LockStats>,
    threshold: Duration,
}

/// Lock statistics.
#[derive(Debug, Clone)]
pub struct LockStats {
    /// Lock name.
    pub name: String,

    /// Total wait time.
    pub total_wait_time: Duration,

    /// Contention count.
    pub contention_count: u64,

    /// Maximum wait time.
    pub max_wait_time: Duration,
}

impl ContentionDetector {
    /// Create a new contention detector.
    pub fn new(threshold: Duration) -> Self {
        Self {
            events: Vec::new(),
            lock_stats: HashMap::new(),
            threshold,
        }
    }

    /// Record a contention event.
    pub fn record(&mut self, event: ContentionEvent) {
        if event.wait_time < self.threshold {
            return;
        }

        let stats = self.lock_stats.entry(event.lock_id).or_insert(LockStats {
            name: event.lock_name.clone(),
            total_wait_time: Duration::ZERO,
            contention_count: 0,
            max_wait_time: Duration::ZERO,
        });

        stats.total_wait_time += event.wait_time;
        stats.contention_count += 1;
        if event.wait_time > stats.max_wait_time {
            stats.max_wait_time = event.wait_time;
        }

        self.events.push(event);
    }

    /// Get events for a lock.
    pub fn events_for_lock(&self, lock_id: u64) -> Vec<&ContentionEvent> {
        self.events
            .iter()
            .filter(|e| e.lock_id == lock_id)
            .collect()
    }

    /// Get contended locks.
    pub fn contended_locks(&self) -> Vec<(u64, &LockStats)> {
        let mut locks: Vec<_> = self
            .lock_stats
            .iter()
            .map(|(id, stats)| (*id, stats))
            .collect();
        locks.sort_by(|a, b| b.1.contention_count.cmp(&a.1.contention_count));
        locks
    }

    /// Get total contention events.
    pub fn event_count(&self) -> usize {
        self.events.len()
    }

    /// Generate a report.
    pub fn report(&self) -> String {
        let mut report = String::new();

        report.push_str(&format!("Contention Events: {}\n", self.event_count()));
        report.push_str(&format!("Contended Locks: {}\n\n", self.lock_stats.len()));

        for (lock_id, stats) in self.contended_locks() {
            let avg_wait = if stats.contention_count > 0 {
                stats.total_wait_time / stats.contention_count as u32
            } else {
                Duration::ZERO
            };

            report.push_str(&format!(
                "Lock {} ({}): {} contentions, avg wait {:?}, max wait {:?}\n",
                lock_id, stats.name, stats.contention_count, avg_wait, stats.max_wait_time
            ));
        }

        report
    }
}

impl Default for ContentionDetector {
    fn default() -> Self {
        Self::new(Duration::from_millis(1))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_contention_detector() {
        let mut detector = ContentionDetector::new(Duration::from_millis(1));

        let event = ContentionEvent {
            lock_id: 1,
            lock_name: "mutex_1".to_string(),
            thread_id: 100,
            wait_time: Duration::from_millis(10),
            timestamp: Instant::now(),
        };

        detector.record(event);
        assert_eq!(detector.event_count(), 1);
    }

    #[test]
    fn test_threshold_filtering() {
        let mut detector = ContentionDetector::new(Duration::from_millis(10));

        detector.record(ContentionEvent {
            lock_id: 1,
            lock_name: "mutex_1".to_string(),
            thread_id: 100,
            wait_time: Duration::from_millis(5),
            timestamp: Instant::now(),
        });

        detector.record(ContentionEvent {
            lock_id: 1,
            lock_name: "mutex_1".to_string(),
            thread_id: 100,
            wait_time: Duration::from_millis(15),
            timestamp: Instant::now(),
        });

        assert_eq!(detector.event_count(), 1); // Only the one above threshold
    }

    #[test]
    fn test_contended_locks() {
        let mut detector = ContentionDetector::new(Duration::from_millis(1));

        for _ in 0..5 {
            detector.record(ContentionEvent {
                lock_id: 1,
                lock_name: "mutex_1".to_string(),
                thread_id: 100,
                wait_time: Duration::from_millis(10),
                timestamp: Instant::now(),
            });
        }

        for _ in 0..3 {
            detector.record(ContentionEvent {
                lock_id: 2,
                lock_name: "mutex_2".to_string(),
                thread_id: 101,
                wait_time: Duration::from_millis(10),
                timestamp: Instant::now(),
            });
        }

        let locks = detector.contended_locks();
        assert_eq!(locks.len(), 2);
        assert_eq!(locks[0].0, 1); // Most contended first
    }
}
