//! Raft-like consensus primitives.
//!
//! Provides building-block types for implementing a Raft consensus
//! protocol in the distributed encoding cluster. These primitives
//! focus on state management and log structures.
//!
//! # Raft Consensus Protocol Usage
//!
//! The Raft algorithm guarantees consensus in a distributed cluster by electing
//! a single *leader* that coordinates all writes. The key phases are:
//!
//! ## Leader Election
//!
//! - All nodes start as **Followers** with term 0.
//! - If a Follower does not receive a heartbeat within the **election timeout**
//!   (typically 150–300 ms), it increments its term, transitions to **Candidate**,
//!   and broadcasts `RequestVote` RPCs.
//! - A node grants a vote if it has not voted in the current term and the
//!   candidate's log is at least as up-to-date as its own.
//! - A Candidate wins if it receives a majority (⌊N/2⌋ + 1) of votes and
//!   transitions to **Leader**.
//!
//! ## Log Replication
//!
//! - The Leader appends each client command to its local [`RaftLog`], then
//!   broadcasts `AppendEntries` RPCs (replicated to followers in parallel).
//! - Once a majority of nodes have acknowledged the entry, the Leader marks
//!   it as *committed* (updates `commit_index`).
//! - Followers apply committed entries to their state machines in order.
//!
//! ## Heartbeat and Timeout Values
//!
//! | Constant            | Typical Value | Notes                                      |
//! |---------------------|---------------|--------------------------------------------|
//! | Heartbeat interval  | 50 ms         | Leader sends empty `AppendEntries` per hop |
//! | Election timeout    | 150–300 ms    | Randomised to avoid split votes            |
//! | RPC timeout         | 30 ms         | After which the RPC is retried             |

#![allow(dead_code)]

use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Instant;

/// A single entry in the Raft replicated log.
#[derive(Debug, Clone)]
pub struct LogEntry {
    /// Term in which this entry was created.
    pub term: u64,
    /// Index of this entry in the log (1-based).
    pub index: u64,
    /// The command/payload encoded as a string.
    pub command: String,
}

impl LogEntry {
    /// Create a new log entry.
    #[must_use]
    pub fn new(term: u64, index: u64, command: impl Into<String>) -> Self {
        Self {
            term,
            index,
            command: command.into(),
        }
    }

    /// Returns true if the entry is valid (term > 0 and index > 0).
    #[must_use]
    pub fn is_valid(&self) -> bool {
        self.term > 0 && self.index > 0
    }
}

/// Role of a node in the Raft protocol.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RaftRole {
    /// The current term's leader.
    Leader,
    /// A regular member following the leader.
    Follower,
    /// A node seeking election.
    Candidate,
}

impl RaftRole {
    /// Returns true if this node can accept write operations.
    #[must_use]
    pub fn can_accept_writes(&self) -> bool {
        matches!(self, Self::Leader)
    }

    /// Returns a human-readable name.
    #[must_use]
    pub fn name(&self) -> &str {
        match self {
            Self::Leader => "Leader",
            Self::Follower => "Follower",
            Self::Candidate => "Candidate",
        }
    }
}

impl std::fmt::Display for RaftRole {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.name())
    }
}

/// Persistent and volatile state for a Raft node.
#[derive(Debug)]
pub struct RaftState {
    /// Latest term this node has seen.
    pub current_term: u64,
    /// Candidate node ID this node voted for in the current term.
    pub voted_for: Option<String>,
    /// Index of the highest log entry known to be committed.
    pub commit_index: u64,
    /// Index of the highest log entry applied to the state machine.
    pub last_applied: u64,
    /// Current role.
    pub role: RaftRole,
}

impl RaftState {
    /// Create a new Raft state in Follower role with term 0.
    #[must_use]
    pub fn new() -> Self {
        Self {
            current_term: 0,
            voted_for: None,
            commit_index: 0,
            last_applied: 0,
            role: RaftRole::Follower,
        }
    }

    /// Advance the current term to `new_term` (only if larger).
    pub fn advance_term(&mut self, new_term: u64) {
        if new_term > self.current_term {
            self.current_term = new_term;
            self.voted_for = None;
        }
    }

    /// Transition to Candidate and start a new election.
    pub fn become_candidate(&mut self) {
        self.current_term += 1;
        self.role = RaftRole::Candidate;
        self.voted_for = None;
    }

    /// Transition to Leader.
    pub fn become_leader(&mut self) {
        self.role = RaftRole::Leader;
    }

    /// Step down to Follower with the given term.
    pub fn become_follower(&mut self, term: u64) {
        self.current_term = term;
        self.role = RaftRole::Follower;
        self.voted_for = None;
    }

    /// Record a vote cast for the given candidate in the current term.
    pub fn vote_for(&mut self, candidate_id: impl Into<String>) {
        self.voted_for = Some(candidate_id.into());
    }

    /// Advance the commit index if `index` is larger than the current value.
    pub fn update_commit_index(&mut self, index: u64) {
        if index > self.commit_index {
            self.commit_index = index;
        }
    }

    /// Advance `last_applied` if `index` is larger and not beyond `commit_index`.
    pub fn apply_up_to(&mut self, index: u64) {
        if index <= self.commit_index && index > self.last_applied {
            self.last_applied = index;
        }
    }
}

impl Default for RaftState {
    fn default() -> Self {
        Self::new()
    }
}

/// The Raft replicated log.
#[derive(Debug, Default)]
pub struct RaftLog {
    /// All log entries in order.
    pub entries: Vec<LogEntry>,
}

impl RaftLog {
    /// Create a new empty log.
    #[must_use]
    pub fn new() -> Self {
        Self {
            entries: Vec::new(),
        }
    }

    /// Append an entry to the log.
    pub fn append(&mut self, entry: LogEntry) {
        self.entries.push(entry);
    }

    /// Get the entry at the given 1-based index.
    #[must_use]
    pub fn get(&self, index: u64) -> Option<&LogEntry> {
        if index == 0 {
            return None;
        }
        self.entries.get((index - 1) as usize)
    }

    /// Returns the index of the last entry (0 if the log is empty).
    #[must_use]
    pub fn last_index(&self) -> u64 {
        self.entries.len() as u64
    }

    /// Returns the term of the last entry (0 if the log is empty).
    #[must_use]
    pub fn last_term(&self) -> u64 {
        self.entries.last().map_or(0, |e| e.term)
    }

    /// Returns all entries up to and including `commit_index`.
    #[must_use]
    pub fn committed_entries(&self, commit_index: u64) -> Vec<&LogEntry> {
        self.entries
            .iter()
            .filter(|e| e.index <= commit_index)
            .collect()
    }

    /// Truncate the log to `last_kept_index`, removing all entries after it.
    pub fn truncate_after(&mut self, last_kept_index: u64) {
        self.entries.retain(|e| e.index <= last_kept_index);
    }

    /// Returns true if the log is empty.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }
}

/// Latency metrics for a Raft node.
///
/// All values are stored as microseconds in [`AtomicU64`] counters so they can
/// be read from any thread without acquiring a lock.
///
/// Use [`RaftMetrics::record_propose_commit`] after a proposal is committed
/// and [`RaftMetrics::record_heartbeat_rtt`] after each heartbeat round-trip.
/// Then call [`RaftMetrics::report`] to obtain a snapshot.
#[derive(Debug, Default)]
pub struct RaftMetrics {
    /// Total number of commit latency samples recorded.
    propose_commit_samples: AtomicU64,
    /// Sum of propose-to-commit latencies in microseconds.
    propose_commit_sum_us: AtomicU64,
    /// Maximum propose-to-commit latency seen (microseconds).
    propose_commit_max_us: AtomicU64,
    /// Total number of heartbeat RTT samples recorded.
    heartbeat_samples: AtomicU64,
    /// Sum of heartbeat RTTs in microseconds.
    heartbeat_sum_us: AtomicU64,
    /// Maximum heartbeat RTT seen (microseconds).
    heartbeat_max_us: AtomicU64,
}

/// A point-in-time snapshot of [`RaftMetrics`].
#[derive(Debug, Clone)]
pub struct RaftMetricsSnapshot {
    /// Number of propose-to-commit latency samples.
    pub propose_commit_samples: u64,
    /// Average propose-to-commit latency in milliseconds.
    pub propose_commit_avg_ms: f64,
    /// Maximum propose-to-commit latency in milliseconds.
    pub propose_commit_max_ms: f64,
    /// Number of heartbeat RTT samples.
    pub heartbeat_samples: u64,
    /// Average heartbeat RTT in milliseconds.
    pub heartbeat_rtt_avg_ms: f64,
    /// Maximum heartbeat RTT in milliseconds.
    pub heartbeat_rtt_max_ms: f64,
}

impl RaftMetrics {
    /// Create a new, zeroed metrics instance.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Record the elapsed time from a proposal being submitted to being
    /// committed.  Pass the `Instant` captured when the proposal was first
    /// submitted; this method captures the current time to compute the
    /// elapsed duration.
    pub fn record_propose_commit(&self, propose_start: Instant) {
        let elapsed_us = propose_start.elapsed().as_micros() as u64;
        self.propose_commit_samples.fetch_add(1, Ordering::Relaxed);
        self.propose_commit_sum_us
            .fetch_add(elapsed_us, Ordering::Relaxed);
        // Update max (relaxed compare-and-swap loop)
        let mut current = self.propose_commit_max_us.load(Ordering::Relaxed);
        while elapsed_us > current {
            match self.propose_commit_max_us.compare_exchange_weak(
                current,
                elapsed_us,
                Ordering::Relaxed,
                Ordering::Relaxed,
            ) {
                Ok(_) => break,
                Err(c) => current = c,
            }
        }
    }

    /// Record the round-trip time for a single heartbeat.  Pass the `Instant`
    /// when the heartbeat was sent; this method captures the current time.
    pub fn record_heartbeat_rtt(&self, send_start: Instant) {
        let elapsed_us = send_start.elapsed().as_micros() as u64;
        self.heartbeat_samples.fetch_add(1, Ordering::Relaxed);
        self.heartbeat_sum_us
            .fetch_add(elapsed_us, Ordering::Relaxed);
        let mut current = self.heartbeat_max_us.load(Ordering::Relaxed);
        while elapsed_us > current {
            match self.heartbeat_max_us.compare_exchange_weak(
                current,
                elapsed_us,
                Ordering::Relaxed,
                Ordering::Relaxed,
            ) {
                Ok(_) => break,
                Err(c) => current = c,
            }
        }
    }

    /// Record a raw propose-to-commit latency value in microseconds (for
    /// testing / synthetic benchmarks where you control the exact value).
    pub fn record_propose_commit_us(&self, latency_us: u64) {
        self.propose_commit_samples.fetch_add(1, Ordering::Relaxed);
        self.propose_commit_sum_us
            .fetch_add(latency_us, Ordering::Relaxed);
        let mut current = self.propose_commit_max_us.load(Ordering::Relaxed);
        while latency_us > current {
            match self.propose_commit_max_us.compare_exchange_weak(
                current,
                latency_us,
                Ordering::Relaxed,
                Ordering::Relaxed,
            ) {
                Ok(_) => break,
                Err(c) => current = c,
            }
        }
    }

    /// Record a raw heartbeat RTT value in microseconds.
    pub fn record_heartbeat_rtt_us(&self, rtt_us: u64) {
        self.heartbeat_samples.fetch_add(1, Ordering::Relaxed);
        self.heartbeat_sum_us.fetch_add(rtt_us, Ordering::Relaxed);
        let mut current = self.heartbeat_max_us.load(Ordering::Relaxed);
        while rtt_us > current {
            match self.heartbeat_max_us.compare_exchange_weak(
                current,
                rtt_us,
                Ordering::Relaxed,
                Ordering::Relaxed,
            ) {
                Ok(_) => break,
                Err(c) => current = c,
            }
        }
    }

    /// Return a snapshot of the current metrics.
    #[must_use]
    pub fn report(&self) -> RaftMetricsSnapshot {
        let pc_samples = self.propose_commit_samples.load(Ordering::Relaxed);
        let pc_sum = self.propose_commit_sum_us.load(Ordering::Relaxed);
        let pc_max = self.propose_commit_max_us.load(Ordering::Relaxed);

        let hb_samples = self.heartbeat_samples.load(Ordering::Relaxed);
        let hb_sum = self.heartbeat_sum_us.load(Ordering::Relaxed);
        let hb_max = self.heartbeat_max_us.load(Ordering::Relaxed);

        let pc_avg_ms = if pc_samples > 0 {
            pc_sum as f64 / pc_samples as f64 / 1000.0
        } else {
            0.0
        };

        let hb_avg_ms = if hb_samples > 0 {
            hb_sum as f64 / hb_samples as f64 / 1000.0
        } else {
            0.0
        };

        RaftMetricsSnapshot {
            propose_commit_samples: pc_samples,
            propose_commit_avg_ms: pc_avg_ms,
            propose_commit_max_ms: pc_max as f64 / 1000.0,
            heartbeat_samples: hb_samples,
            heartbeat_rtt_avg_ms: hb_avg_ms,
            heartbeat_rtt_max_ms: hb_max as f64 / 1000.0,
        }
    }

    /// Reset all counters to zero.
    pub fn reset(&self) {
        self.propose_commit_samples.store(0, Ordering::Relaxed);
        self.propose_commit_sum_us.store(0, Ordering::Relaxed);
        self.propose_commit_max_us.store(0, Ordering::Relaxed);
        self.heartbeat_samples.store(0, Ordering::Relaxed);
        self.heartbeat_sum_us.store(0, Ordering::Relaxed);
        self.heartbeat_max_us.store(0, Ordering::Relaxed);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_log_entry_is_valid() {
        assert!(LogEntry::new(1, 1, "cmd").is_valid());
        assert!(!LogEntry::new(0, 1, "cmd").is_valid()); // term = 0
        assert!(!LogEntry::new(1, 0, "cmd").is_valid()); // index = 0
        assert!(!LogEntry::new(0, 0, "cmd").is_valid());
    }

    #[test]
    fn test_raft_role_can_accept_writes() {
        assert!(RaftRole::Leader.can_accept_writes());
        assert!(!RaftRole::Follower.can_accept_writes());
        assert!(!RaftRole::Candidate.can_accept_writes());
    }

    #[test]
    fn test_raft_role_display() {
        assert_eq!(RaftRole::Leader.to_string(), "Leader");
        assert_eq!(RaftRole::Follower.to_string(), "Follower");
        assert_eq!(RaftRole::Candidate.to_string(), "Candidate");
    }

    #[test]
    fn test_raft_state_initial() {
        let state = RaftState::new();
        assert_eq!(state.current_term, 0);
        assert!(state.voted_for.is_none());
        assert_eq!(state.commit_index, 0);
        assert_eq!(state.last_applied, 0);
        assert_eq!(state.role, RaftRole::Follower);
    }

    #[test]
    fn test_raft_state_advance_term() {
        let mut state = RaftState::new();
        state.vote_for("node1");
        state.advance_term(5);
        assert_eq!(state.current_term, 5);
        // advance_term should clear voted_for
        assert!(state.voted_for.is_none());

        // Should not regress
        state.advance_term(3);
        assert_eq!(state.current_term, 5);
    }

    #[test]
    fn test_raft_state_become_candidate() {
        let mut state = RaftState::new();
        state.become_candidate();
        assert_eq!(state.current_term, 1);
        assert_eq!(state.role, RaftRole::Candidate);
    }

    #[test]
    fn test_raft_state_become_leader() {
        let mut state = RaftState::new();
        state.become_candidate();
        state.become_leader();
        assert_eq!(state.role, RaftRole::Leader);
    }

    #[test]
    fn test_raft_state_become_follower() {
        let mut state = RaftState::new();
        state.become_leader();
        state.become_follower(7);
        assert_eq!(state.role, RaftRole::Follower);
        assert_eq!(state.current_term, 7);
        assert!(state.voted_for.is_none());
    }

    #[test]
    fn test_raft_state_update_commit_index() {
        let mut state = RaftState::new();
        state.update_commit_index(5);
        assert_eq!(state.commit_index, 5);
        // Should not go backwards
        state.update_commit_index(3);
        assert_eq!(state.commit_index, 5);
    }

    #[test]
    fn test_raft_state_apply_up_to() {
        let mut state = RaftState::new();
        state.update_commit_index(10);
        state.apply_up_to(7);
        assert_eq!(state.last_applied, 7);
        // Cannot exceed commit_index
        state.apply_up_to(15);
        assert_eq!(state.last_applied, 7);
    }

    #[test]
    fn test_raft_log_append_and_get() {
        let mut log = RaftLog::new();
        assert!(log.is_empty());
        assert_eq!(log.last_index(), 0);
        assert_eq!(log.last_term(), 0);

        log.append(LogEntry::new(1, 1, "set x=1"));
        log.append(LogEntry::new(1, 2, "set y=2"));
        log.append(LogEntry::new(2, 3, "set z=3"));

        assert_eq!(log.last_index(), 3);
        assert_eq!(log.last_term(), 2);
        assert!(!log.is_empty());
    }

    #[test]
    fn test_raft_log_get_valid_index() {
        let mut log = RaftLog::new();
        log.append(LogEntry::new(1, 1, "cmd1"));
        log.append(LogEntry::new(2, 2, "cmd2"));

        let e = log.get(1).expect("get should return a value");
        assert_eq!(e.command, "cmd1");
        assert_eq!(e.term, 1);
    }

    #[test]
    fn test_raft_log_get_invalid_index() {
        let log = RaftLog::new();
        assert!(log.get(0).is_none());
        assert!(log.get(1).is_none());
    }

    #[test]
    fn test_raft_log_committed_entries() {
        let mut log = RaftLog::new();
        log.append(LogEntry::new(1, 1, "a"));
        log.append(LogEntry::new(1, 2, "b"));
        log.append(LogEntry::new(2, 3, "c"));

        let committed = log.committed_entries(2);
        assert_eq!(committed.len(), 2);
        assert_eq!(committed[0].command, "a");
        assert_eq!(committed[1].command, "b");
    }

    #[test]
    fn test_raft_log_truncate_after() {
        let mut log = RaftLog::new();
        log.append(LogEntry::new(1, 1, "a"));
        log.append(LogEntry::new(1, 2, "b"));
        log.append(LogEntry::new(2, 3, "c"));

        log.truncate_after(2);
        assert_eq!(log.last_index(), 2);
        assert!(log.get(3).is_none());
    }

    // ---- RaftMetrics tests ----

    #[test]
    fn test_raft_metrics_captures_latency() {
        let metrics = RaftMetrics::new();

        // Record 3 synthetic propose-to-commit samples
        metrics.record_propose_commit_us(2_000); // 2 ms
        metrics.record_propose_commit_us(4_000); // 4 ms
        metrics.record_propose_commit_us(6_000); // 6 ms

        // Record 2 heartbeat RTT samples
        metrics.record_heartbeat_rtt_us(500); // 0.5 ms
        metrics.record_heartbeat_rtt_us(1_500); // 1.5 ms

        let snap = metrics.report();

        assert_eq!(snap.propose_commit_samples, 3);
        // avg = (2+4+6)/3 = 4 ms
        assert!((snap.propose_commit_avg_ms - 4.0).abs() < 0.01);
        // max = 6 ms
        assert!((snap.propose_commit_max_ms - 6.0).abs() < 0.01);

        assert_eq!(snap.heartbeat_samples, 2);
        // avg = (0.5+1.5)/2 = 1.0 ms
        assert!((snap.heartbeat_rtt_avg_ms - 1.0).abs() < 0.01);
        // max = 1.5 ms
        assert!((snap.heartbeat_rtt_max_ms - 1.5).abs() < 0.01);
    }

    #[test]
    fn test_raft_metrics_empty_report() {
        let metrics = RaftMetrics::new();
        let snap = metrics.report();
        assert_eq!(snap.propose_commit_samples, 0);
        assert_eq!(snap.heartbeat_samples, 0);
        assert_eq!(snap.propose_commit_avg_ms, 0.0);
        assert_eq!(snap.heartbeat_rtt_avg_ms, 0.0);
    }

    #[test]
    fn test_raft_metrics_reset() {
        let metrics = RaftMetrics::new();
        metrics.record_propose_commit_us(1_000);
        metrics.record_heartbeat_rtt_us(500);
        metrics.reset();

        let snap = metrics.report();
        assert_eq!(snap.propose_commit_samples, 0);
        assert_eq!(snap.heartbeat_samples, 0);
    }

    #[test]
    fn test_raft_metrics_instant_recording() {
        let metrics = RaftMetrics::new();
        let t = Instant::now();
        // Tiny sleep to ensure elapsed > 0
        std::thread::sleep(std::time::Duration::from_micros(100));
        metrics.record_propose_commit(t);
        let snap = metrics.report();
        assert_eq!(snap.propose_commit_samples, 1);
        assert!(snap.propose_commit_max_ms >= 0.0);
    }

    #[test]
    fn test_raft_metrics_max_tracks_correctly() {
        let metrics = RaftMetrics::new();
        metrics.record_propose_commit_us(100);
        metrics.record_propose_commit_us(9_000); // 9 ms
        metrics.record_propose_commit_us(500);

        let snap = metrics.report();
        assert!((snap.propose_commit_max_ms - 9.0).abs() < 0.01);
    }
}
