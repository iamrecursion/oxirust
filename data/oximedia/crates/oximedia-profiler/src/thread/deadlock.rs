//! Deadlock detection.

use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};

/// Deadlock information.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DeadlockInfo {
    /// Threads involved in the deadlock.
    pub threads: Vec<u64>,

    /// Locks involved in the deadlock.
    pub locks: Vec<u64>,

    /// Lock acquisition chain.
    pub chain: Vec<(u64, u64)>, // (thread_id, lock_id)
}

/// Lock acquisition.
#[allow(dead_code)]
#[derive(Debug, Clone)]
struct Acquisition {
    thread_id: u64,
    lock_id: u64,
}

/// Deadlock detector.
#[derive(Debug)]
pub struct DeadlockDetector {
    // Thread -> locks it holds
    held_locks: HashMap<u64, HashSet<u64>>,
    // Thread -> locks it's waiting for
    waiting_for: HashMap<u64, u64>,
    // Lock -> thread that holds it
    lock_owners: HashMap<u64, u64>,
}

impl DeadlockDetector {
    /// Create a new deadlock detector.
    pub fn new() -> Self {
        Self {
            held_locks: HashMap::new(),
            waiting_for: HashMap::new(),
            lock_owners: HashMap::new(),
        }
    }

    /// Record a lock acquisition.
    pub fn acquire_lock(&mut self, thread_id: u64, lock_id: u64) {
        self.held_locks
            .entry(thread_id)
            .or_default()
            .insert(lock_id);
        self.lock_owners.insert(lock_id, thread_id);
        self.waiting_for.remove(&thread_id);
    }

    /// Record a lock wait.
    pub fn wait_for_lock(&mut self, thread_id: u64, lock_id: u64) {
        self.waiting_for.insert(thread_id, lock_id);
    }

    /// Record a lock release.
    pub fn release_lock(&mut self, thread_id: u64, lock_id: u64) {
        if let Some(locks) = self.held_locks.get_mut(&thread_id) {
            locks.remove(&lock_id);
        }
        self.lock_owners.remove(&lock_id);
    }

    /// Detect deadlocks.
    pub fn detect(&self) -> Vec<DeadlockInfo> {
        let mut deadlocks = Vec::new();

        for (&waiting_thread, &waiting_lock) in &self.waiting_for {
            if let Some(cycle) = self.find_cycle(waiting_thread, waiting_lock) {
                let deadlock = self.build_deadlock_info(cycle);
                deadlocks.push(deadlock);
            }
        }

        deadlocks
    }

    /// Find a cycle in the wait graph.
    fn find_cycle(&self, start_thread: u64, start_lock: u64) -> Option<Vec<(u64, u64)>> {
        let mut chain = vec![(start_thread, start_lock)];
        let mut visited_threads = HashSet::new();
        visited_threads.insert(start_thread);

        let mut current_lock = start_lock;

        loop {
            let owner = self.lock_owners.get(&current_lock)?;

            if *owner == start_thread {
                // Found a cycle!
                return Some(chain);
            }

            if visited_threads.contains(owner) {
                // Found a cycle, but not involving the start thread
                return None;
            }

            visited_threads.insert(*owner);

            let next_lock = self.waiting_for.get(owner)?;
            chain.push((*owner, *next_lock));
            current_lock = *next_lock;
        }
    }

    /// Build deadlock info from a cycle.
    fn build_deadlock_info(&self, chain: Vec<(u64, u64)>) -> DeadlockInfo {
        let threads: Vec<_> = chain.iter().map(|(t, _)| *t).collect();
        let locks: Vec<_> = chain.iter().map(|(_, l)| *l).collect();

        DeadlockInfo {
            threads,
            locks,
            chain,
        }
    }

    /// Generate a report.
    pub fn report(&self) -> String {
        let deadlocks = self.detect();
        let mut report = String::new();

        if deadlocks.is_empty() {
            report.push_str("No deadlocks detected.\n");
        } else {
            report.push_str(&format!("Deadlocks Detected: {}\n\n", deadlocks.len()));

            for (i, deadlock) in deadlocks.iter().enumerate() {
                report.push_str(&format!("Deadlock #{}:\n", i + 1));
                report.push_str(&format!("  Threads: {:?}\n", deadlock.threads));
                report.push_str(&format!("  Locks: {:?}\n", deadlock.locks));
                report.push_str("  Chain:\n");

                for (thread, lock) in &deadlock.chain {
                    report.push_str(&format!(
                        "    Thread {} waiting for Lock {}\n",
                        thread, lock
                    ));
                }

                report.push('\n');
            }
        }

        report
    }
}

impl Default for DeadlockDetector {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_deadlock_detector() {
        let mut detector = DeadlockDetector::new();

        // Thread 1 holds Lock 1
        detector.acquire_lock(1, 100);

        // Thread 2 holds Lock 2
        detector.acquire_lock(2, 200);

        // Thread 1 waits for Lock 2
        detector.wait_for_lock(1, 200);

        // Thread 2 waits for Lock 1 (creates deadlock)
        detector.wait_for_lock(2, 100);

        let deadlocks = detector.detect();
        assert!(!deadlocks.is_empty());
    }

    #[test]
    fn test_no_deadlock() {
        let mut detector = DeadlockDetector::new();

        detector.acquire_lock(1, 100);
        detector.release_lock(1, 100);
        detector.acquire_lock(2, 100);

        let deadlocks = detector.detect();
        assert!(deadlocks.is_empty());
    }

    #[test]
    fn test_release_breaks_deadlock() {
        let mut detector = DeadlockDetector::new();

        detector.acquire_lock(1, 100);
        detector.acquire_lock(2, 200);
        detector.wait_for_lock(1, 200);
        detector.wait_for_lock(2, 100);

        assert!(!detector.detect().is_empty());

        // Release one lock
        detector.release_lock(1, 100);

        // Deadlock should be broken
        assert!(detector.detect().is_empty());
    }
}
