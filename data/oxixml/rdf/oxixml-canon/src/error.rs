//! The error type produced by canonicalization.

use std::error::Error;
use std::fmt;

/// A specialized [`Result`](std::result::Result) for canonicalization operations.
pub type Result<T, E = CanonicalizationError> = std::result::Result<T, E>;

/// An error raised while canonicalizing an RDF dataset.
///
/// The [RDFC-1.0 security considerations](https://www.w3.org/TR/rdf-canon/#security-considerations)
/// warn that adversarial ("poison") datasets can drive the
/// [`hash-n-degree-quads`](https://www.w3.org/TR/rdf-canon/#hash-nd-quads) recursion into
/// super-polynomial (up to factorial) work. To keep canonicalization a total function on
/// untrusted input, this crate guards the recursion with a configurable
/// [`WorkBudget`](crate::WorkBudget) and reports [`WorkBudgetExceeded`](Self::WorkBudgetExceeded)
/// when the budget is exhausted rather than looping unboundedly.
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub enum CanonicalizationError {
    /// The configured [`WorkBudget`](crate::WorkBudget) was exhausted before a canonical form
    /// could be produced.
    ///
    /// This almost always indicates a deliberately adversarial input graph. Callers who trust the
    /// input may retry with a larger (or [unlimited](crate::WorkBudget::unlimited)) budget.
    WorkBudgetExceeded {
        /// The budget (in abstract work units) that was exceeded.
        budget: u64,
    },
}

impl fmt::Display for CanonicalizationError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::WorkBudgetExceeded { budget } => write!(
                f,
                "the RDFC-1.0 work budget of {budget} unit(s) was exceeded; \
                 the input is likely an adversarial (poison) dataset"
            ),
        }
    }
}

impl Error for CanonicalizationError {}
