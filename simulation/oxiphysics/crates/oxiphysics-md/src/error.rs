// Copyright 2026 COOLJAPAN OU (Team KitaSan)
// SPDX-License-Identifier: Apache-2.0

//! Error types for oxiphysics-md

use thiserror::Error;

/// Main error type for the md module
#[derive(Debug, Error)]
pub enum Error {
    /// Generic error
    #[error("{0}")]
    General(String),
    /// SETTLE was given a degenerate (collinear / zero-area / non-finite) reference
    /// geometry, or a degenerate intermediate state, so the analytic solution does
    /// not exist. The string describes the specific failure.
    #[error("SETTLE degenerate reference geometry: {0}")]
    SettleDegenerate(String),
    /// Two quantities that must have the same length (for example the training
    /// descriptors and the training energies of a regression, or the positions
    /// and the neighbour lists of a configuration) do not.
    #[error("dimension mismatch in {context}: expected {expected}, got {got}")]
    DimensionMismatch {
        /// What was being compared, e.g. `"KRR training energies"`.
        context: &'static str,
        /// The length the other quantity dictates.
        expected: usize,
        /// The length that was supplied.
        got: usize,
    },
}

/// Result type alias
pub type Result<T> = std::result::Result<T, Error>;

/// Convenience alias used by the MD-facing API (e.g. SETTLE).
pub type MdError = Error;
