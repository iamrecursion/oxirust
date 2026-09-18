//! # Proof Result, Resolution Step, Strategy, and Statistics
//!
//! This module defines the supporting types used by the resolution prover:
//! the outcome of a proof attempt, a single resolution step in a derivation,
//! the available proof strategies, and the statistics gathered during
//! proof search.

use serde::{Deserialize, Serialize};

use super::clause::Clause;
use super::literal::Literal;

/// Result of a resolution proof attempt.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub enum ProofResult {
    /// The clause set is unsatisfiable (empty clause derived)
    Unsatisfiable {
        /// Steps taken to derive empty clause
        steps: usize,
        /// Derivation path (sequence of resolution steps)
        derivation: Vec<ResolutionStep>,
    },
    /// The clause set is satisfiable (no contradiction found)
    Satisfiable,
    /// Proof attempt reached saturation without finding empty clause
    Saturated {
        /// Number of clauses generated
        clauses_generated: usize,
    },
    /// Proof search reached resource limit
    ResourceLimitReached {
        /// Number of steps attempted
        steps_attempted: usize,
    },
}

impl ProofResult {
    /// Check if the result proves unsatisfiability.
    pub fn is_unsatisfiable(&self) -> bool {
        matches!(self, ProofResult::Unsatisfiable { .. })
    }

    /// Check if the result proves satisfiability.
    pub fn is_satisfiable(&self) -> bool {
        matches!(self, ProofResult::Satisfiable)
    }

    /// Get the number of steps taken.
    pub fn steps(&self) -> usize {
        match self {
            ProofResult::Unsatisfiable { steps, .. } => *steps,
            ProofResult::ResourceLimitReached { steps_attempted } => *steps_attempted,
            ProofResult::Saturated { clauses_generated } => *clauses_generated,
            ProofResult::Satisfiable => 0,
        }
    }
}

/// A single resolution step in a proof.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ResolutionStep {
    /// First parent clause
    pub parent1: Clause,
    /// Second parent clause
    pub parent2: Clause,
    /// Resulting resolvent clause
    pub resolvent: Clause,
    /// Literal that was resolved on (from parent1)
    pub resolved_literal: Literal,
}

/// Resolution proof strategy.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub enum ResolutionStrategy {
    /// Generate all possible resolvents until empty clause or saturation
    Saturation {
        /// Maximum number of clauses to generate
        max_clauses: usize,
    },
    /// Focus resolution on specific set of clauses
    SetOfSupport {
        /// Maximum resolution steps
        max_steps: usize,
    },
    /// Only resolve with unit clauses (more efficient)
    UnitResolution {
        /// Maximum resolution steps
        max_steps: usize,
    },
    /// Linear resolution: chain from initial clause
    Linear {
        /// Maximum chain length
        max_depth: usize,
    },
}

/// Statistics for resolution proof search.
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct ProverStats {
    /// Total clauses generated
    pub clauses_generated: usize,
    /// Resolution steps performed
    pub resolution_steps: usize,
    /// Tautologies removed
    pub tautologies_removed: usize,
    /// Clauses subsumed
    pub clauses_subsumed: usize,
    /// Empty clause found
    pub empty_clause_found: bool,
}
