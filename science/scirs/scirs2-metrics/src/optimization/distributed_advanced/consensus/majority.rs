//! Simple Majority Consensus Implementation
//!
//! Implementation of simple majority voting consensus for distributed optimization.

use crate::error::{MetricsError, Result};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Simple majority consensus implementation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MajorityConsensus {
    node_id: String,
    nodes: Vec<String>,
    votes: HashMap<String, bool>,
    proposal_id: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Vote {
    voter_id: String,
    proposal_id: u64,
    decision: bool,
}

/// Simple majority algorithm state
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MajorityState {
    pub current_proposal_id: u64,
    pub active_votes: HashMap<String, Vote>,
    pub completed_proposals: Vec<u64>,
    pub consensus_threshold: f64,
}

impl Default for MajorityState {
    fn default() -> Self {
        Self {
            current_proposal_id: 0,
            active_votes: HashMap::new(),
            completed_proposals: Vec::new(),
            consensus_threshold: 0.5, // Simple majority (>50%)
        }
    }
}

impl MajorityConsensus {
    pub fn new(node_id: String, nodes: Vec<String>) -> Self {
        Self {
            node_id,
            nodes,
            votes: HashMap::new(),
            proposal_id: 0,
        }
    }

    pub fn propose(&mut self, data: Vec<u8>) -> Result<u64> {
        self.proposal_id += 1;
        self.votes.clear();
        // Auto-vote for own proposal
        self.votes.insert(self.node_id.clone(), true);
        Ok(self.proposal_id)
    }

    pub fn vote(&mut self, vote: Vote) -> Result<()> {
        if vote.proposal_id != self.proposal_id {
            return Err(MetricsError::InvalidOperation("Invalid proposal ID".into()));
        }

        self.votes.insert(vote.voter_id, vote.decision);
        Ok(())
    }

    pub fn has_majority(&self) -> bool {
        let yes_votes = self.votes.values().filter(|&&v| v).count();
        let total_nodes = self.nodes.len();
        yes_votes > total_nodes / 2
    }

    pub fn is_decided(&self) -> bool {
        let total_votes = self.votes.len();
        let total_nodes = self.nodes.len();

        // Check if we have enough votes to make a decision
        let yes_votes = self.votes.values().filter(|&&v| v).count();
        let no_votes = total_votes - yes_votes;

        // Majority reached
        yes_votes > total_nodes / 2 || no_votes > total_nodes / 2
    }

    pub fn get_result(&self) -> Option<bool> {
        if self.is_decided() {
            Some(self.has_majority())
        } else {
            None
        }
    }
}
