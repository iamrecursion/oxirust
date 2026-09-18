//! Loopy Belief Propagation (LBP) for general (cyclic) factor graphs.
//!
//! Standard sum-product BP is exact only on tree-structured factor graphs.
//! When cycles are present, the same algorithm run iteratively is called
//! **Loopy Belief Propagation** (LBP).  LBP is not guaranteed to converge,
//! and even when it does the fixed-point beliefs are generally only
//! approximations to the true marginals.  However, in practice LBP gives
//! excellent approximations and is widely used in computer-vision, coding
//! theory, and neurosymbolic AI.
//!
//! # Key design decisions
//!
//! * **Asynchronous vs synchronous schedule** — both supported via
//!   [`UpdateSchedule`].  Asynchronous (residual BP) tends to converge
//!   faster on loopy graphs by prioritizing the messages with the largest
//!   residual at each step.
//! * **Message damping** — applied per-message using an [`LbpDampingPolicy`]
//!   that can be uniform, adaptive (residual-based), or off.
//! * **Log-domain arithmetic** — all message computations are performed in
//!   log space via [`LogMessage`] to avoid numerical underflow in deeply
//!   loopy graphs.
//! * **Convergence monitoring** — [`LbpConvergenceMonitor`] tracks the
//!   per-message L∞ residual and the global maximum across iterations.
//! * **Cycle detection** — [`CycleDetector`] identifies whether the graph
//!   has cycles and reports the approximate girth, helping callers choose
//!   appropriate algorithms.
//! * **Free energy** — [`BetheFreeEnergy`] computes the Bethe approximation
//!   to the free energy (and therefore the partition function) from the
//!   converged beliefs, giving a quality measure for the LBP solution.
//!
//! # References
//!
//! * Yedidia, Freeman & Weiss (2003) — "Understanding Belief Propagation
//!   and its Generalizations"
//! * Koller & Friedman (2009) — *Probabilistic Graphical Models*
//! * Sutton & McCallum (2012) — *An Introduction to CRFs*

mod beliefs;
mod config;
mod cycle;
mod energy;
mod engine;
mod schedules;
mod types;

#[cfg(test)]
mod tests;

pub use config::{LoopyBpConfig, LoopyBpResult};
pub use cycle::{CycleAnalysis, CycleDetector};
pub use energy::{bethe_free_energy, BetheFreeEnergy};
pub use engine::LoopyBeliefPropagation;
pub use types::{
    LbpConvergenceMonitor, LbpDampingPolicy, LbpIterStats, LogMessage, UpdateSchedule,
};
