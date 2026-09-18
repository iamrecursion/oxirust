//! Exponential family trait for Variational Message Passing.
//!
//! In VMP (Winn & Bishop, 2005) every variable has a variational distribution that
//! belongs to a known exponential family. This module defines the minimal contract
//! every such distribution must satisfy so that the engine can perform coordinate
//! ascent purely in natural-parameter space.
//!
//! The canonical form of an exponential family distribution is
//!
//! ```text
//!   p(x | η) = h(x) · exp(ηᵀ u(x) − A(η))
//! ```
//!
//! where
//! - `η` is the vector of **natural parameters**,
//! - `u(x)` is the vector of **sufficient statistics**,
//! - `A(η)` is the **log partition function** (a.k.a. cumulant function), and
//! - `h(x)` is the base measure (which cancels out of every VMP update and is
//!   therefore not part of the trait).
//!
//! A key property we rely on is
//!
//! ```text
//!   E_q[u(x)] = ∇_η A(η).
//! ```
//!
//! That identity is what lets the engine send "expected sufficient statistic"
//! messages between factors without ever touching raw probability tables.

use crate::error::{PgmError, Result};

/// Minimal contract every VMP-compatible distribution must implement.
///
/// Implementations represent a *variational distribution* over one variable, fully
/// parameterised by its natural parameters (η). The trait is deliberately kept
/// small — it only exposes what the coordinate-ascent update and ELBO computation
/// need:
///
/// 1. Read / write the natural parameter vector (`to_natural` / `update_natural`).
/// 2. Evaluate sufficient statistics at a point value (`sufficient_statistics`).
/// 3. Evaluate the log partition (`log_partition`) and expected sufficient
///    statistics (`expected_sufficient_statistics`) at the current η.
///
/// All vector shapes must match the fixed `natural_dim()` of the implementation.
pub trait ExponentialFamily: Clone {
    /// Name of the family (used for error messages, e.g. "Gaussian").
    fn family_name(&self) -> &'static str;

    /// Dimensionality of the natural-parameter vector.
    fn natural_dim(&self) -> usize;

    /// Return a copy of the current natural parameters.
    fn natural_params(&self) -> Vec<f64>;

    /// Read-only view of the natural parameter vector.
    ///
    /// Provided by default via `natural_params` but implementations may override
    /// to avoid the allocation if they already store η contiguously.
    fn to_natural(&self) -> Vec<f64> {
        self.natural_params()
    }

    /// Replace the natural parameters with `new_eta`.
    ///
    /// Returns an error if the dimensions do not match `natural_dim()`.
    fn set_natural(&mut self, new_eta: &[f64]) -> Result<()>;

    /// Additively update the natural parameters: η ← η + δ.
    ///
    /// Returns an error if the dimensions do not match `natural_dim()`.
    fn update_natural(&mut self, delta: &[f64]) -> Result<()> {
        if delta.len() != self.natural_dim() {
            return Err(PgmError::DimensionMismatch {
                expected: vec![self.natural_dim()],
                got: vec![delta.len()],
            });
        }
        let mut eta = self.natural_params();
        for (a, b) in eta.iter_mut().zip(delta.iter()) {
            *a += *b;
        }
        self.set_natural(&eta)
    }

    /// Sufficient statistics `u(x)` evaluated at the scalar or categorical value `value`.
    ///
    /// For discrete families `value.floor() as usize` is the category index; for
    /// continuous ones it is the raw real value. Returning a `Vec<f64>` keeps the
    /// interface uniform at the cost of one small heap allocation per call — this
    /// is only invoked in ELBO paths, not in the hot inner loop.
    fn sufficient_statistics(&self, value: f64) -> Vec<f64>;

    /// Log partition function `A(η)`.
    fn log_partition(&self, natural_params: &[f64]) -> Result<f64>;

    /// Expected sufficient statistics `E_q[u(x)] = ∇_η A(η)`.
    ///
    /// Computed from the *current* η stored inside `self`.
    fn expected_sufficient_statistics(&self) -> Vec<f64>;

    /// Differential entropy `H(q) = A(η) − ηᵀ E_q[u(x)] − E_q[log h(x)]`.
    ///
    /// The last term is zero for every family we ship (Gaussian with fixed
    /// precision, Categorical, Dirichlet); if a future family needs a non-trivial
    /// base measure it must override this default.
    fn entropy(&self) -> Result<f64> {
        let eta = self.natural_params();
        let a = self.log_partition(&eta)?;
        let ess = self.expected_sufficient_statistics();
        debug_assert_eq!(eta.len(), ess.len());
        let dot: f64 = eta.iter().zip(ess.iter()).map(|(e, s)| e * s).sum();
        Ok(a - dot)
    }
}
