//! Random architecture search for NAS.
//!
//! A simple baseline: architectures are sampled uniformly at random from the
//! [`ArchSearchSpace`] with no model of performance.  Useful as a lower-bound
//! comparison against more sophisticated strategies such as
//! [`RegularizedEvolution`](super::evolution::RegularizedEvolution).

use crate::error::TrainResult;

use super::evolution::NasResult;
use super::sampler::ArchSampler;
use super::space::{ArchSearchSpace, Architecture};

// ─── RandomArchSearch ───────────────────────────────────────────────────────

/// Random-search NAS: samples architectures uniformly at random, keeps track
/// of the best seen so far.
pub struct RandomArchSearch {
    sampler: ArchSampler,
    best: Option<(Architecture, f64)>,
    history: Vec<(Architecture, f64)>,
}

impl RandomArchSearch {
    /// Create a new random architecture searcher.
    ///
    /// # Arguments
    ///
    /// * `space` - Architecture search space to sample from.
    /// * `seed` - RNG seed for reproducibility.
    pub fn new(space: ArchSearchSpace, seed: u64) -> Self {
        Self {
            sampler: ArchSampler::new(space, seed),
            best: None,
            history: Vec::new(),
        }
    }

    /// Ask for the next architecture to evaluate.
    ///
    /// Always returns a fresh uniformly random architecture independent of
    /// previous results.
    pub fn ask(&mut self) -> TrainResult<Architecture> {
        self.sampler.random_architecture()
    }

    /// Tell the result of evaluating an architecture.
    ///
    /// Updates the running best when `score` exceeds the current best.
    pub fn tell(&mut self, arch: Architecture, score: f64) {
        let is_better = self
            .best
            .as_ref()
            .is_none_or(|(_, best_score)| score > *best_score);

        if is_better {
            self.best = Some((arch.clone(), score));
        }
        self.history.push((arch, score));
    }

    /// Return a reference to the best (architecture, score) pair seen so far,
    /// or `None` if no evaluations have been recorded.
    pub fn best(&self) -> Option<&(Architecture, f64)> {
        self.best.as_ref()
    }

    /// Produce a [`NasResult`] summarising the current search state.
    ///
    /// Returns `None` if no evaluations have been recorded yet.
    pub fn result(&self) -> Option<NasResult> {
        let (best_arch, best_score) = self.best.as_ref()?;
        Some(NasResult {
            best: best_arch.clone(),
            best_score: *best_score,
            history: self.history.clone(),
        })
    }
}
