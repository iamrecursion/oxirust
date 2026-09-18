// Copyright 2026 COOLJAPAN OU (Team KitaSan)
// SPDX-License-Identifier: Apache-2.0

//! Error type of the modal-analysis routines.

use thiserror::Error;

/// Why a modal-analysis routine refused its input or failed to produce a result.
///
/// The eigen solvers ([`inverse_iteration`](super::inverse_iteration),
/// [`lanczos_iteration`](super::lanczos_iteration)), the mode-shape normalisation
/// ([`normalize_mode_shape`](super::normalize_mode_shape)) and the static
/// correction ([`residual_flexibility`](super::residual_flexibility)) report
/// degenerate input through this type instead of panicking or returning a fabricated
/// result (for example a mode with `ω = 0`).
#[derive(Debug, Clone, PartialEq, Error)]
#[non_exhaustive]
pub enum ModalError {
    /// The system has no degrees of freedom.
    #[error("the system has no degrees of freedom")]
    EmptySystem,
    /// A matrix or vector does not have the size the system requires.
    #[error("dimension mismatch: expected {expected}, got {got}")]
    DimensionMismatch {
        /// Size the operation requires.
        expected: usize,
        /// Size that was supplied.
        got: usize,
    },
    /// More modes were requested than the system has degrees of freedom.
    #[error("{requested} modes requested but the system has only {n} degrees of freedom")]
    TooManyModes {
        /// Number of modes that was asked for.
        requested: usize,
        /// Number of degrees of freedom of the system.
        n: usize,
    },
    /// Mass normalisation was requested without a mass matrix.
    #[error("a mass matrix is required for mass normalisation")]
    MassMatrixRequired,
    /// The iteration did not deliver mode `mode` (zero based): it did not reach
    /// the tolerance within the iteration limit, the deflated vector collapsed, or
    /// the Krylov space was exhausted before enough Ritz pairs existed (repeated
    /// eigenvalues).
    #[error("mode {mode} did not converge")]
    NotConverged {
        /// Zero-based index of the first mode that could not be computed.
        mode: usize,
    },
    /// The generalised mass `φᵀMφ` of a mode is not positive, so the mode cannot
    /// be mass-normalised (a zero mode, or a mass matrix that is not positive
    /// definite).
    #[error(
        "the generalised mass φᵀMφ = {value:e} is not positive, the mode cannot be mass-normalised"
    )]
    NonPositiveGeneralizedMass {
        /// The offending value of `φᵀMφ`.
        value: f64,
    },
    /// The static solve `K u = F` did not converge (a singular or indefinite
    /// stiffness matrix, for instance a structure with rigid-body modes).
    #[error("the static solve K u = F did not converge (relative residual {residual:e})")]
    StaticSolveFailed {
        /// Relative residual `‖F − K u‖ / ‖F‖` at which the solver stopped.
        residual: f64,
    },
}

impl From<ModalError> for crate::error::Error {
    fn from(error: ModalError) -> Self {
        match error {
            ModalError::NotConverged { mode } => crate::error::Error::General(format!(
                "modal analysis: mode {mode} did not converge"
            )),
            ModalError::StaticSolveFailed { residual } => crate::error::Error::NotConverged {
                solver: "static solve of the residual flexibility".to_string(),
                iterations: 0,
                residual,
            },
            other => crate::error::Error::InvalidInput(other.to_string()),
        }
    }
}
