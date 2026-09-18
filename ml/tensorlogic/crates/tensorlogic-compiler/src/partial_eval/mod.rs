//! Partial evaluation / expression specialization for TLExpr.
//!
//! Given a set of concrete bindings for **some** variables while leaving others
//! free/symbolic, this module reduces an expression as much as possible. This is
//! strictly more powerful than plain constant propagation because it handles
//! *mixed* symbolic-concrete environments and enables expression specialization
//! across a range of inputs.
//!
//! # Variable Convention
//!
//! Following the established codebase convention, a scalar "variable" named `"x"`
//! is represented as a zero-arity predicate: `TLExpr::Pred { name: "x", args: [] }`.
//! Booleans are encoded as `TLExpr::Constant(1.0)` (true) and
//! `TLExpr::Constant(0.0)` (false).
//!
//! # Example
//!
//! ```rust
//! use tensorlogic_compiler::partial_eval::{PEEnv, PEConfig, partially_evaluate};
//! use tensorlogic_ir::TLExpr;
//!
//! // Expression: x + y  (both are zero-arity predicates acting as variables)
//! let x = TLExpr::pred("x", vec![]);
//! let y = TLExpr::pred("y", vec![]);
//! let expr = TLExpr::add(x, y);
//!
//! // Partially evaluate with x = 3.0; y stays symbolic
//! let env = PEEnv::new().with_f64("x", 3.0);
//! let config = PEConfig::default();
//! let result = partially_evaluate(&expr, &env, &config);
//!
//! // Result should be: Add(Constant(3.0), Pred("y", []))
//! // (not fully concrete because y is still free)
//! assert!(result.residual_vars.contains(&"y".to_string()));
//! ```

mod api;
mod helpers;
mod pe_arith;
mod pe_core;
mod pe_logic;
mod pe_math;
mod pe_passthrough;
mod pe_quantifiers;
mod types;

#[cfg(test)]
mod tests;

pub use api::{partially_evaluate, specialize, specialize_batch};
pub use types::{PEConfig, PEEnv, PEResult, PEStats, PEValue};
