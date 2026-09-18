//! Consensus Coordinator Implementation
//!
//! Coordinates different consensus algorithms for distributed optimization.

use super::{
    majority::MajorityConsensus, pbft::PbftConsensus, proof_of_stake::ProofOfStakeConsensus,
    raft::RaftConsensus,
};
use crate::error::{MetricsError, Result};

/// Consensus coordinator that manages different consensus algorithms
#[derive(Debug)]
pub struct ConsensusCoordinator {
    algorithm_type: ConsensusType,
    raft: Option<RaftConsensus>,
    pbft: Option<PbftConsensus>,
    pos: Option<ProofOfStakeConsensus>,
    majority: Option<MajorityConsensus>,
}

#[derive(Debug, Clone)]
pub enum ConsensusType {
    Raft,
    Pbft,
    ProofOfStake,
    Majority,
}

impl ConsensusCoordinator {
    /// Create new consensus coordinator from configuration
    pub fn new(
        config: crate::optimization::distributed::config::ConsensusConfig,
    ) -> crate::error::Result<Self> {
        use crate::optimization::distributed::config::ConsensusAlgorithm;
        match config.algorithm {
            ConsensusAlgorithm::Raft => Ok(Self::new_raft(
                config.node_id.unwrap_or_else(|| "default".to_string()),
                config.peers.unwrap_or_default(),
            )),
            ConsensusAlgorithm::Pbft => Ok(Self::new_pbft(
                config.node_id.unwrap_or_else(|| "default".to_string()),
                config.peers.unwrap_or_default(),
            )),
            ConsensusAlgorithm::ProofOfStake => Ok(Self::new_proof_of_stake(
                config.node_id.unwrap_or_else(|| "default".to_string()),
                100,
            )),
            ConsensusAlgorithm::SimpleMajority => Ok(Self::new_majority(
                config.node_id.unwrap_or_else(|| "default".to_string()),
                config.peers.unwrap_or_default(),
            )),
            ConsensusAlgorithm::DelegatedProofOfStake => Ok(Self::new_proof_of_stake(
                config.node_id.unwrap_or_else(|| "default".to_string()),
                100,
            )), // Treat similar to ProofOfStake for now
            ConsensusAlgorithm::None => Err(crate::error::MetricsError::ConsensusError(
                "No consensus algorithm specified".to_string(),
            )),
        }
    }

    pub fn new_raft(node_id: String, peers: Vec<String>) -> Self {
        Self {
            algorithm_type: ConsensusType::Raft,
            raft: Some(RaftConsensus::new(node_id, peers)),
            pbft: None,
            pos: None,
            majority: None,
        }
    }

    pub fn new_pbft(node_id: String, replicas: Vec<String>) -> Self {
        Self {
            algorithm_type: ConsensusType::Pbft,
            raft: None,
            pbft: Some(PbftConsensus::new(node_id, replicas)),
            pos: None,
            majority: None,
        }
    }

    pub fn new_proof_of_stake(node_id: String, stake: u64) -> Self {
        Self {
            algorithm_type: ConsensusType::ProofOfStake,
            raft: None,
            pbft: None,
            pos: Some(ProofOfStakeConsensus::new(node_id, stake)),
            majority: None,
        }
    }

    pub fn new_majority(node_id: String, nodes: Vec<String>) -> Self {
        Self {
            algorithm_type: ConsensusType::Majority,
            raft: None,
            pbft: None,
            pos: None,
            majority: Some(MajorityConsensus::new(node_id, nodes)),
        }
    }

    pub fn propose(&mut self, data: Vec<u8>) -> Result<String> {
        match self.algorithm_type {
            ConsensusType::Raft => {
                if let Some(ref mut raft) = self.raft {
                    raft.start_election()?;
                    Ok("raft_proposal".to_string())
                } else {
                    Err(MetricsError::InvalidOperation(
                        "Raft not initialized".into(),
                    ))
                }
            }
            ConsensusType::Pbft => {
                if let Some(ref mut pbft) = self.pbft {
                    let msg = pbft.pre_prepare(data)?;
                    Ok(format!("pbft_{}", msg.sequence))
                } else {
                    Err(MetricsError::InvalidOperation(
                        "PBFT not initialized".into(),
                    ))
                }
            }
            ConsensusType::ProofOfStake => {
                if let Some(ref pos) = self.pos {
                    let validator = pos.select_validator()?;
                    Ok(format!("pos_{}", validator))
                } else {
                    Err(MetricsError::InvalidOperation("PoS not initialized".into()))
                }
            }
            ConsensusType::Majority => {
                if let Some(ref mut majority) = self.majority {
                    let proposal_id = majority.propose(data)?;
                    Ok(format!("majority_{}", proposal_id))
                } else {
                    Err(MetricsError::InvalidOperation(
                        "Majority not initialized".into(),
                    ))
                }
            }
        }
    }
}
