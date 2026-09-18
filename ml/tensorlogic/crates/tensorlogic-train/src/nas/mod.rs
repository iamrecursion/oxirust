//! Neural Architecture Search (NAS) for TensorLogic.
//!
//! This module provides infrastructure for automatically discovering high-performing
//! neural network architectures without manual design.  The search follows the
//! **ask/tell** convention used throughout this crate: callers request a candidate
//! architecture via `ask()`, evaluate it externally, then report the score via
//! `tell()`.  No objective closure is stored.
//!
//! ## Algorithms
//!
//! | Type | Struct | Notes |
//! |------|--------|-------|
//! | Random search | [`RandomArchSearch`] | Uniform baseline, no state |
//! | Regularized evolution | [`RegularizedEvolution`] | Real et al. 2019 [^1] |
//!
//! ## Regularized (Aging) Evolution
//!
//! Based on Real et al. (2019) *"Regularized Evolution for Image Classifier
//! Architecture Search"* (AAAI 2019, <https://arxiv.org/abs/1802.01548>).
//!
//! The algorithm maintains a **cyclic population** of fixed size
//! (`population_size`).  Once the population is full:
//!
//! 1. Draw `tournament_size` members uniformly at random.
//! 2. Select the highest-scoring member (the *winner*).
//! 3. Apply a single random mutation to the winner to produce a child.
//! 4. Evaluate the child externally (ask/tell cycle).
//! 5. Add the child to the population and evict the **oldest** member
//!    (front of the deque) — regardless of its score.
//!
//! Aging pressure prevents premature convergence and keeps exploration alive
//! throughout the search budget.
//!
//! ## Search Space
//!
//! [`ArchSearchSpace`] constrains:
//! - Depth range (`min_depth`..`max_depth`)
//! - Per-layer width options (discrete set of `usize`)
//! - Per-layer activation options (e.g. `"relu"`, `"gelu"`, `"tanh"`)
//! - Per-layer operation options (e.g. `"linear"`, `"conv"`, `"attention"`)
//!
//! [`ArchSampler`] draws uniformly from this space and implements the four
//! mutation operators (change op, change width, change activation, add/remove
//! layer).
//!
//! ## Quick Start
//!
//! ```rust,no_run
//! use tensorlogic_train::nas::{ArchSearchSpace, RegularizedEvolution};
//!
//! let space = ArchSearchSpace::new(
//!     2, 6,
//!     vec![64, 128, 256],
//!     vec!["relu".to_string(), "gelu".to_string()],
//!     vec!["linear".to_string(), "conv".to_string()],
//! ).unwrap();
//!
//! let mut evo = RegularizedEvolution::new(space, 20, 5, 42).unwrap();
//!
//! for _ in 0..100 {
//!     let arch = evo.ask().unwrap();
//!     // …evaluate arch externally…
//!     let score = 0.9_f64; // placeholder
//!     evo.tell(arch, score);
//! }
//!
//! if let Some((best, score)) = evo.best() {
//!     println!("Best depth={}, score={score:.4}", best.depth());
//! }
//! ```
//!
//! [^1]: Real, E., Aggarwal, A., Huang, Y., & Le, Q. V. (2019). Regularized
//!       evolution for image classifier architecture search. *Proceedings of the
//!       AAAI Conference on Artificial Intelligence*, 33(01), 4780–4789.

pub mod evolution;
pub mod random_search;
pub mod sampler;
pub mod space;

#[cfg(test)]
mod tests;

pub use evolution::{NasResult, RegularizedEvolution};
pub use random_search::RandomArchSearch;
pub use sampler::ArchSampler;
pub use space::{ArchSearchSpace, Architecture, LayerSpec};
