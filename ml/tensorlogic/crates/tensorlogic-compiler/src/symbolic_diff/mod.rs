//! Symbolic differentiation of TLExpr.
//!
//! This module implements formal symbolic differentiation of TensorLogic expressions
//! with respect to named variables. It supports:
//!
//! - Standard arithmetic differentiation rules (sum, product, quotient, power, chain)
//! - Unary transcendental functions (sin, cos, exp, log, sqrt, abs)
//! - Logical differentiation (AND, OR, NOT, implication)
//! - Quantifiers treated as scoped operators (bound variable treated as independent)
//! - Let-bindings via substitution
//! - Jacobian computation for multiple variables
//! - Post-differentiation algebraic simplification
//!
//! # Design Notes
//!
//! In TLExpr, a scalar variable named `"x"` is represented as a zero-arity predicate:
//! `TLExpr::Pred { name: "x".to_string(), args: vec![] }`.
//! This convention is used throughout the codebase and is the basis for variable
//! detection in this module.
//!
//! Arithmetic negation (unary minus) does **not** have its own `TLExpr` variant.
//! It is represented as `TLExpr::Sub(Constant(0.0), inner)` or
//! `TLExpr::Mul(Constant(-1.0), inner)`. The simplification pass recognises the
//! `Sub(0, e)` form and folds it to a negated constant where possible.
//!
//! # Example
//!
//! ```rust
//! use tensorlogic_compiler::symbolic_diff::{differentiate, DiffConfig};
//! use tensorlogic_ir::TLExpr;
//!
//! // d(x * x)/dx  →  x * 1 + x * 1  →  (after simplification) x + x
//! let x = TLExpr::pred("x", vec![]);
//! let expr = TLExpr::mul(x.clone(), x.clone());
//! let config = DiffConfig::default();
//! let result = differentiate(&expr, "x", &config).expect("differentiate");
//! // result.derivative is the symbolic derivative (x + x)
//! ```

mod api;
mod diff_arith;
mod diff_core;
mod diff_fuzzy;
mod diff_logic;
mod diff_modal;
mod diff_sets;
mod helpers;
mod types;

#[cfg(test)]
mod tests;

pub use api::{differentiate, jacobian};
pub use helpers::simplify_derivative;
pub use types::{DiffConfig, DiffError, DiffResult};
