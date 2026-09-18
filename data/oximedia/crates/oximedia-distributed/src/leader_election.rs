#![allow(dead_code)]
//! Leader election primitives for `OxiMedia` distributed cluster.
//!
//! Provides a simplified Bully/term-based leader election model without external
//! dependencies: nodes nominate themselves, collect votes, and the node with the
//! most votes in a term wins.

use std::collections::HashMap;
use std::time::{Duration, Instant};

/// State of a node in the election process.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ElectionState {
    /// No election is currently running on this node.
    Idle,
    /// This node has started or joined an election and awaits votes.
    Candidate,
    /// This node follows the elected leader.
    Follower,
    /// This node won the election and is the current leader.
    Leader,
}

impl ElectionState {
    /// Return `true` if the node is currently a candidate.
    #[must_use]
    pub fn is_candidate(self) -> bool {
        self == ElectionState::Candidate
    }

    /// Return `true` if the node holds leadership.
    #[must_use]
    pub fn is_leader(self) -> bool {
        self == ElectionState::Leader
    }

    /// Human-readable label.
    #[must_use]
    pub fn label(self) -> &'static str {
        match self {
            ElectionState::Idle => "idle",
            ElectionState::Candidate => "candidate",
            ElectionState::Follower => "follower",
            ElectionState::Leader => "leader",
        }
    }
}

/// A vote cast by a single node in a given election term.
#[derive(Debug, Clone)]
pub struct NodeVote {
    /// ID of the node casting the vote.
    pub voter_id: String,
    /// ID of the node being voted for.
    pub candidate_id: String,
    /// Election term this vote belongs to.
    pub term: u64,
    /// When the vote was cast.
    pub cast_at: Instant,
    /// Optional reason / justification for the vote.
    pub reason: Option<String>,
}

impl NodeVote {
    /// Create a new vote.
    #[must_use]
    pub fn new(voter_id: impl Into<String>, candidate_id: impl Into<String>, term: u64) -> Self {
        Self {
            voter_id: voter_id.into(),
            candidate_id: candidate_id.into(),
            term,
            cast_at: Instant::now(),
            reason: None,
        }
    }

    /// Return `true` if the vote is for the given term and the voter and candidate
    /// are distinct, non-empty nodes.
    #[must_use]
    pub fn is_valid(&self, expected_term: u64) -> bool {
        self.term == expected_term
            && !self.voter_id.is_empty()
            && !self.candidate_id.is_empty()
            && self.voter_id != self.candidate_id
    }

    /// Age of the vote in milliseconds.
    #[must_use]
    pub fn age_ms(&self, now: Instant) -> u64 {
        now.saturating_duration_since(self.cast_at).as_millis() as u64
    }
}

/// Manager for a single-node's view of the cluster election.
#[derive(Debug)]
pub struct ElectionManager {
    /// This node's identifier.
    pub node_id: String,
    /// Current election term.
    pub term: u64,
    /// State of this node in the current term.
    pub state: ElectionState,
    /// Votes received in the current term, keyed by `voter_id`.
    votes: HashMap<String, NodeVote>,
    /// Total cluster size (used to determine quorum).
    cluster_size: usize,
    /// When the current election was started.
    election_started_at: Option<Instant>,
    /// Timeout after which the election is considered failed.
    election_timeout: Duration,
}

impl ElectionManager {
    /// Create a new election manager for `node_id` in a cluster of `cluster_size` nodes.
    #[must_use]
    pub fn new(
        node_id: impl Into<String>,
        cluster_size: usize,
        election_timeout: Duration,
    ) -> Self {
        Self {
            node_id: node_id.into(),
            term: 0,
            state: ElectionState::Idle,
            votes: HashMap::new(),
            cluster_size,
            election_started_at: None,
            election_timeout,
        }
    }

    /// Advance to the next term and transition this node to `Candidate`.
    pub fn start_election(&mut self) {
        self.term += 1;
        self.state = ElectionState::Candidate;
        self.votes.clear();
        self.election_started_at = Some(Instant::now());
    }

    /// Record a vote for the current term.
    ///
    /// Returns `true` if the vote was accepted (valid for current term and not a
    /// duplicate from the same voter).
    pub fn record_vote(&mut self, vote: NodeVote) -> bool {
        if !vote.is_valid(self.term) {
            return false;
        }
        // Idempotent: ignore duplicate votes from same voter.
        if self.votes.contains_key(&vote.voter_id) {
            return false;
        }
        self.votes.insert(vote.voter_id.clone(), vote);
        self.update_state();
        true
    }

    /// Return the candidate with the most votes, or `None` if no votes yet.
    #[must_use]
    pub fn winner(&self) -> Option<&str> {
        let mut tally: HashMap<&str, usize> = HashMap::new();
        for vote in self.votes.values() {
            *tally.entry(vote.candidate_id.as_str()).or_insert(0) += 1;
        }
        tally
            .into_iter()
            .max_by_key(|(_, count)| *count)
            .map(|(id, _)| id)
    }

    /// Quorum required to win: majority of the cluster.
    #[must_use]
    pub fn quorum(&self) -> usize {
        self.cluster_size / 2 + 1
    }

    /// Number of votes received in the current term.
    #[must_use]
    pub fn vote_count(&self) -> usize {
        self.votes.len()
    }

    /// Return `true` if the election has timed out.
    #[must_use]
    pub fn is_timed_out(&self, now: Instant) -> bool {
        match self.election_started_at {
            None => false,
            Some(started) => now.saturating_duration_since(started) >= self.election_timeout,
        }
    }

    /// Transition this node to follower of `leader_id`, resetting election state.
    pub fn become_follower(&mut self) {
        self.state = ElectionState::Follower;
        self.votes.clear();
        self.election_started_at = None;
    }

    /// Force-set this node as leader (called after winning election externally).
    pub fn become_leader(&mut self) {
        self.state = ElectionState::Leader;
    }

    fn update_state(&mut self) {
        // Check if this node reached quorum.
        let my_id = self.node_id.clone();
        let my_votes = self
            .votes
            .values()
            .filter(|v| v.candidate_id == my_id)
            .count();
        if my_votes >= self.quorum() {
            self.state = ElectionState::Leader;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Duration;

    fn make_manager(node_id: &str, cluster_size: usize) -> ElectionManager {
        ElectionManager::new(node_id, cluster_size, Duration::from_secs(5))
    }

    fn cast_vote(manager: &mut ElectionManager, voter: &str, candidate: &str) -> bool {
        let vote = NodeVote::new(voter, candidate, manager.term);
        manager.record_vote(vote)
    }

    #[test]
    fn test_election_state_labels() {
        assert_eq!(ElectionState::Idle.label(), "idle");
        assert_eq!(ElectionState::Candidate.label(), "candidate");
        assert_eq!(ElectionState::Follower.label(), "follower");
        assert_eq!(ElectionState::Leader.label(), "leader");
    }

    #[test]
    fn test_is_candidate() {
        assert!(ElectionState::Candidate.is_candidate());
        assert!(!ElectionState::Leader.is_candidate());
        assert!(!ElectionState::Idle.is_candidate());
    }

    #[test]
    fn test_is_leader() {
        assert!(ElectionState::Leader.is_leader());
        assert!(!ElectionState::Candidate.is_leader());
    }

    #[test]
    fn test_node_vote_is_valid() {
        let vote = NodeVote::new("voter1", "node2", 3);
        assert!(vote.is_valid(3));
        assert!(!vote.is_valid(2)); // wrong term
    }

    #[test]
    fn test_node_vote_invalid_self_vote() {
        let vote = NodeVote::new("node1", "node1", 1);
        assert!(!vote.is_valid(1)); // voter == candidate
    }

    #[test]
    fn test_node_vote_invalid_empty_ids() {
        let vote = NodeVote::new("", "node2", 1);
        assert!(!vote.is_valid(1));
    }

    #[test]
    fn test_start_election_increments_term() {
        let mut mgr = make_manager("n1", 5);
        assert_eq!(mgr.term, 0);
        mgr.start_election();
        assert_eq!(mgr.term, 1);
        assert!(mgr.state.is_candidate());
    }

    #[test]
    fn test_start_election_clears_votes() {
        let mut mgr = make_manager("n1", 3);
        mgr.start_election();
        cast_vote(&mut mgr, "n2", "n1");
        mgr.start_election(); // new election
        assert_eq!(mgr.vote_count(), 0);
    }

    #[test]
    fn test_record_vote_accepted() {
        let mut mgr = make_manager("n1", 3);
        mgr.start_election();
        assert!(cast_vote(&mut mgr, "n2", "n1"));
        assert_eq!(mgr.vote_count(), 1);
    }

    #[test]
    fn test_record_vote_duplicate_rejected() {
        let mut mgr = make_manager("n1", 5);
        mgr.start_election();
        assert!(cast_vote(&mut mgr, "n2", "n1"));
        assert!(!cast_vote(&mut mgr, "n2", "n1")); // duplicate
        assert_eq!(mgr.vote_count(), 1);
    }

    #[test]
    fn test_winner_after_majority() {
        let mut mgr = make_manager("n1", 3);
        mgr.start_election();
        cast_vote(&mut mgr, "n2", "n1");
        cast_vote(&mut mgr, "n3", "n1");
        assert_eq!(mgr.winner(), Some("n1"));
        assert!(mgr.state.is_leader());
    }

    #[test]
    fn test_quorum_calculation() {
        let mgr3 = make_manager("n1", 3);
        assert_eq!(mgr3.quorum(), 2);
        let mgr5 = make_manager("n1", 5);
        assert_eq!(mgr5.quorum(), 3);
    }

    #[test]
    fn test_become_follower() {
        let mut mgr = make_manager("n1", 3);
        mgr.start_election();
        mgr.become_follower();
        assert_eq!(mgr.state, ElectionState::Follower);
        assert_eq!(mgr.vote_count(), 0);
    }

    #[test]
    fn test_is_timed_out() {
        let mut mgr = ElectionManager::new("n1", 3, Duration::from_millis(1));
        mgr.start_election();
        std::thread::sleep(Duration::from_millis(5));
        assert!(mgr.is_timed_out(Instant::now()));
    }

    #[test]
    fn test_not_timed_out_before_deadline() {
        let mut mgr = ElectionManager::new("n1", 3, Duration::from_secs(60));
        mgr.start_election();
        assert!(!mgr.is_timed_out(Instant::now()));
    }
}
