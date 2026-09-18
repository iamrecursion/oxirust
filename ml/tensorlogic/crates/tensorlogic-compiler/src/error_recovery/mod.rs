//! Partial error recovery for multi-expression compilation.
//!
//! TensorLogic historically supported a single **strict** compilation mode:
//! [`crate::compile_to_einsum`] returns the first error it encounters and
//! aborts the whole compile. For real programs that carry several rules or
//! expressions this is overly brittle — a single bad rule prevents useful
//! feedback about the rest of the program.
//!
//! This module introduces a complementary **tolerant** mode. In tolerant
//! mode the compiler treats the input as a *program* (slice of
//! [`tensorlogic_ir::TLExpr`]), compiles each expression in isolation, and
//! collects per-expression diagnostics into a
//! [`DiagnosticCollector`]. Non-fatal problems in one expression never
//! prevent siblings from compiling.
//!
//! # Quick start
//!
//! ```
//! use tensorlogic_compiler::error_recovery::{
//!     compile_tolerant, RecoveryStrategy, Severity,
//! };
//! use tensorlogic_ir::{TLExpr, Term};
//!
//! let program = vec![
//!     TLExpr::pred("p", vec![Term::var("x")]),
//!     TLExpr::pred("q", vec![Term::var("y")]),
//! ];
//! let result = compile_tolerant(&program);
//! assert_eq!(result.graphs.len(), 2);
//! assert!(result.is_all_success());
//! assert!(result.diagnostics.is_empty());
//! # let _ = RecoveryStrategy::SkipOnError;
//! # let _ = Severity::Error;
//! ```
//!
//! # Design
//!
//! * **Scope**: recovery happens at the *compile-one-expression boundary*.
//!   Each expression's own compilation path keeps propagating `Result::Err`
//!   internally; the tolerant driver merely intercepts that `Err` (and any
//!   panic, via [`std::panic::catch_unwind`]) and converts it into a
//!   [`Diagnostic`]. No `catch_unwind` is sprinkled into inner passes.
//! * **Strict mode is untouched**: [`crate::compile_to_einsum`] and
//!   [`crate::compile_to_einsum_with_context`] retain their pre-existing
//!   behaviour. The tolerant driver is a *new* public entry point.
//! * **Configurable policy**: [`RecoveryStrategy`] selects one of
//!   `SkipOnError`, `SkipOnFatal`, `AbortOnAny` — see its docs for the
//!   full decision table.
//!
//! # Re-exports
//!
//! The commonly used types are re-exported at the module root so downstream
//! callers can `use tensorlogic_compiler::error_recovery::*`.

mod collector;
mod diagnostic;
mod strategy;
mod tolerant_compiler;

#[cfg(test)]
mod tests;

pub use collector::DiagnosticCollector;
pub use diagnostic::{Diagnostic, Severity, SourceSpan};
pub use strategy::{RecoveryAction, RecoveryStrategy};
pub use tolerant_compiler::{
    compile_tolerant, compile_tolerant_with_strategy, PartialCompilationResult, TolerantCompiler,
};
