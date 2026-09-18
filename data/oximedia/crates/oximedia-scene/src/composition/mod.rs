//! Shot composition analysis.
//!
//! Analyzes framing, balance, and composition rules.

pub mod balance;
pub mod depth;
pub mod rules;

pub use balance::{BalanceAnalyzer, BalanceMetrics};
pub use depth::{DepthAnalyzer, DepthCues};
pub use rules::{CompositionAnalyzer, CompositionScore};
