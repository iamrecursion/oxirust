//! Low-rank tensor approximation via truncated SVD.
//!
//! This module implements:
//! - [`TruncatedSvd`]: Power-iteration truncated SVD (no external SVD library)
//! - [`LowRankApproximation`]: High-level approximation API
//! - [`LowRankInferencePass`]: Graph pass for identifying approximation candidates

pub mod approximation;
pub mod config;
pub mod error;
pub mod svd;

pub use approximation::{
    LowRankApproximation, LowRankCandidate, LowRankInferencePass, LowRankPassStats,
};
pub use config::LowRankConfig;
pub use error::LowRankError;
pub use svd::{SvdResult, TruncatedSvd};
