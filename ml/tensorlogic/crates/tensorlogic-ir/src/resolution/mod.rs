//! # Resolution-Based Theorem Proving
//!
//! This module implements Robinson's resolution principle for automated theorem proving.
//! Resolution is a refutation-based proof procedure: to prove `Γ ⊢ φ`, we show that
//! `Γ ∪ {¬φ}` is unsatisfiable by deriving the empty clause (⊥).
//!
//! ## Overview
//!
//! **Resolution** is a complete inference rule for first-order logic:
//! - Given clauses `C₁ ∨ L` and `C₂ ∨ ¬L`, derive resolvent `C₁ ∨ C₂`
//! - The empty clause (∅) represents a contradiction
//! - If ∅ is derived, the original clause set is unsatisfiable
//!
//! ## Key Components
//!
//! ### Literals
//! A literal is an atom or its negation:
//! - Positive literal: `P(x, y)`
//! - Negative literal: `¬P(x, y)`
//!
//! ### Clauses
//! A clause is a disjunction of literals:
//! - `P(x) ∨ Q(x) ∨ ¬R(y)`
//! - Empty clause: `∅` (contradiction)
//! - Unit clause: single literal
//!
//! ### Resolution Rule
//! From clauses `C₁ ∨ L` and `C₂ ∨ ¬L`, derive `C₁ ∨ C₂`:
//! ```text
//!     C₁ ∨ L    C₂ ∨ ¬L
//!     ───────────────────
//!         C₁ ∨ C₂
//! ```
//!
//! ## Algorithms
//!
//! 1. **Saturation**: Generate all resolvents until empty clause or saturation
//! 2. **Set-of-Support**: Focus resolution on specific clause set
//! 3. **Linear Resolution**: Chain resolutions from initial clause
//! 4. **Unit Resolution**: Only resolve with unit clauses (more efficient)
//!
//! ## Example
//!
//! ```rust
//! use tensorlogic_ir::{TLExpr, Term, Clause, Literal, ResolutionProver};
//!
//! // Clauses: { P(a), ¬P(a) }
//! // This is unsatisfiable (derives empty clause via direct resolution)
//! let p_a = Literal::positive(TLExpr::pred("P", vec![Term::constant("a")]));
//! let not_p_a = Literal::negative(TLExpr::pred("P", vec![Term::constant("a")]));
//!
//! let mut prover = ResolutionProver::new();
//! prover.add_clause(Clause::from_literals(vec![p_a]));
//! prover.add_clause(Clause::from_literals(vec![not_p_a]));
//!
//! let result = prover.prove();
//! assert!(result.is_unsatisfiable());
//! ```
//!
//! ## Module Layout
//!
//! - [`literal`]: `Literal` type and matching helpers.
//! - [`clause`]: `Clause` type with substitution, renaming, and subsumption.
//! - [`proof`]: Proof result, resolution step, strategy and statistics types.
//! - [`prover`]: [`ResolutionProver`] driving the different proof strategies.
//! - [`cnf`]: Simplified conversion from [`crate::TLExpr`] to clausal normal form.

pub mod clause;
pub mod cnf;
pub mod literal;
pub mod proof;
pub mod prover;

pub use clause::Clause;
pub use cnf::to_cnf;
pub use literal::Literal;
pub use proof::{ProofResult, ProverStats, ResolutionStep, ResolutionStrategy};
pub use prover::ResolutionProver;

#[cfg(test)]
use crate::expr::TLExpr;
#[cfg(test)]
use crate::term::Term;
#[cfg(test)]
use crate::unification::Substitution;

#[cfg(test)]
mod tests;

#[cfg(test)]
mod tests_first_order;

#[cfg(test)]
mod tests_subsumption;
