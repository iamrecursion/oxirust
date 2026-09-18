//! Regularized (aging) evolution for neural architecture search.
//!
//! Implements the algorithm from:
//!
//! > Real et al. (2019) "Regularized Evolution for Image Classifier Architecture Search"
//! > <https://arxiv.org/abs/1802.01548>
//!
//! The key idea is a **cyclic population** (VecDeque): the oldest individual is
//! evicted whenever the population exceeds `population_size`, even if it happens
//! to be the highest-scoring member.  This aging pressure prevents premature
//! convergence and keeps the search exploratory.
//!
//! The **ask/tell** API lets the caller supply evaluation scores without any
//! stored objective closure, matching the hyperparameter-optimization convention
//! used elsewhere in this crate.

use std::collections::VecDeque;

use crate::error::{TrainError, TrainResult};

use super::sampler::ArchSampler;
use super::space::{ArchSearchSpace, Architecture};

// ─── NasResult ──────────────────────────────────────────────────────────────

/// Summary of a completed (or in-progress) NAS run.
#[derive(Debug, Clone)]
pub struct NasResult {
    /// Best architecture found at the time of the call.
    pub best: Architecture,
    /// Score of the best architecture (higher = better).
    pub best_score: f64,
    /// All evaluated (architecture, score) pairs in evaluation order.
    pub history: Vec<(Architecture, f64)>,
}

// ─── RegularizedEvolution ───────────────────────────────────────────────────

/// Regularized (aging) evolution NAS searcher.
///
/// Uses a cyclic population with tournament selection and single-step mutation.
pub struct RegularizedEvolution {
    /// Cyclic population of (architecture, score) pairs ordered by age (oldest first).
    population: VecDeque<(Architecture, f64)>,
    /// Target population size.
    pub population_size: usize,
    /// Number of random members sampled per tournament.
    pub tournament_size: usize,
    /// Architecture sampler (owns the search space and RNG).
    sampler: ArchSampler,
    /// Full evaluation history in tell() order.
    history: Vec<(Architecture, f64)>,
    /// True once the population has been filled for the first time.
    filled: bool,
}

impl RegularizedEvolution {
    /// Create a new regularized evolution searcher.
    ///
    /// # Arguments
    ///
    /// * `space` - Architecture search space.
    /// * `population_size` - Target size of the cyclic population (≥ 2).
    /// * `tournament_size` - Number of randomly drawn population members per tournament (≤ population_size).
    /// * `seed` - RNG seed for reproducibility.
    ///
    /// # Errors
    ///
    /// Returns [`TrainError::InvalidParameter`] when:
    /// - `population_size` < 2
    /// - `tournament_size` == 0 or `tournament_size` > `population_size`
    pub fn new(
        space: ArchSearchSpace,
        population_size: usize,
        tournament_size: usize,
        seed: u64,
    ) -> TrainResult<Self> {
        if population_size < 2 {
            return Err(TrainError::InvalidParameter(format!(
                "population_size ({population_size}) must be ≥ 2"
            )));
        }
        if tournament_size == 0 {
            return Err(TrainError::InvalidParameter(
                "tournament_size must be ≥ 1".to_string(),
            ));
        }
        if tournament_size > population_size {
            return Err(TrainError::InvalidParameter(format!(
                "tournament_size ({tournament_size}) must be ≤ population_size ({population_size})"
            )));
        }

        Ok(Self {
            population: VecDeque::new(),
            population_size,
            tournament_size,
            sampler: ArchSampler::new(space, seed),
            history: Vec::new(),
            filled: false,
        })
    }

    /// Ask for the next architecture to evaluate.
    ///
    /// * While the population has fewer than `population_size` members (warm-up
    ///   phase), returns a freshly sampled random architecture.
    /// * Once the population is full, performs tournament selection among
    ///   `tournament_size` randomly chosen members and returns the winner
    ///   mutated by one step.
    pub fn ask(&mut self) -> TrainResult<Architecture> {
        if !self.filled {
            // Warm-up: fill the population with random architectures.
            self.sampler.random_architecture()
        } else {
            // Tournament selection + mutation.
            let winner = self.tournament_select()?;
            self.sampler.mutate(&winner)
        }
    }

    /// Tell the result of evaluating an architecture.
    ///
    /// Appends `(arch, score)` to the full history and pushes it into the
    /// population (at the back / newest position).  When the population exceeds
    /// `population_size`, the oldest entry (front) is evicted.
    pub fn tell(&mut self, arch: Architecture, score: f64) {
        self.history.push((arch.clone(), score));
        self.population.push_back((arch, score));
        if self.population.len() >= self.population_size {
            self.filled = true;
        }
        if self.population.len() > self.population_size {
            self.population.pop_front();
        }
    }

    /// Return a reference to the highest-scored (architecture, score) pair
    /// currently in the population, or `None` if the population is empty.
    pub fn best(&self) -> Option<&(Architecture, f64)> {
        self.population
            .iter()
            .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal))
    }

    /// Produce a [`NasResult`] summarising the current search state.
    ///
    /// Returns `None` if no evaluations have been recorded yet.
    pub fn result(&self) -> Option<NasResult> {
        // best over the *full* history (not just surviving population members)
        let (best_arch, best_score) = self
            .history
            .iter()
            .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal))?;

        Some(NasResult {
            best: best_arch.clone(),
            best_score: *best_score,
            history: self.history.clone(),
        })
    }

    // ── private helpers ──────────────────────────────────────────────────

    /// Draw `tournament_size` members uniformly at random (without replacement
    /// when possible) from the population and return a clone of the one with
    /// the highest score.
    fn tournament_select(&mut self) -> TrainResult<Architecture> {
        let pop_len = self.population.len();
        if pop_len == 0 {
            return Err(TrainError::InvalidParameter(
                "tournament_select called on empty population".to_string(),
            ));
        }

        // Collect `tournament_size` distinct indices (Fisher-Yates partial shuffle).
        let sample_size = self.tournament_size.min(pop_len);
        let mut indices: Vec<usize> = (0..pop_len).collect();

        // Partial Fisher-Yates: bring `sample_size` elements to the front.
        // For step i, pick j uniformly from [i, pop_len) and swap.
        for i in 0..sample_size {
            // gen_range_usize(i, pop_len) returns a value in [i, pop_len)
            let j = self.sampler.gen_range_usize(i, pop_len);
            indices.swap(i, j);
        }

        // Best among the sample.
        let best_idx = indices[..sample_size]
            .iter()
            .max_by(|&&a, &&b| {
                self.population[a]
                    .1
                    .partial_cmp(&self.population[b].1)
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
            .copied()
            .ok_or_else(|| {
                TrainError::InvalidParameter("tournament sample was empty".to_string())
            })?;

        Ok(self.population[best_idx].0.clone())
    }
}
