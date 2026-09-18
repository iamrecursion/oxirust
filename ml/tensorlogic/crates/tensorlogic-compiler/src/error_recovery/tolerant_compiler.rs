//! Tolerant (partial-error-recovery) compilation driver.
//!
//! The [`TolerantCompiler`] compiles a *program* — i.e. a slice of
//! [`TLExpr`] — under a configurable [`RecoveryStrategy`]. Each expression is
//! compiled in **isolation**, so a single malformed expression never aborts
//! the compilation of its siblings (under the default strategy).
//!
//! Internally the driver:
//!
//! 1. Iterates the input slice in order.
//! 2. For each expression, calls [`compile_to_einsum_with_context`] inside
//!    [`std::panic::catch_unwind`] — any panic becomes a
//!    [`Severity::Fatal`] diagnostic rather than unwinding across the
//!    driver boundary.
//! 3. Any `Err(...)` from the compiler is converted into a
//!    [`Severity::Error`] diagnostic and the offending slot becomes `None`.
//! 4. The chosen [`RecoveryStrategy`] decides whether to continue, skip
//!    this expression only, or abort the whole program.
//!
//! The original strict entry points ([`compile_to_einsum`],
//! [`compile_to_einsum_with_context`]) are untouched.
//!
//! [`compile_to_einsum`]: crate::compile_to_einsum
//! [`compile_to_einsum_with_context`]: crate::compile_to_einsum_with_context
//! [`TLExpr`]: tensorlogic_ir::TLExpr

use std::panic::{self, AssertUnwindSafe};

use tensorlogic_ir::{EinsumGraph, TLExpr};

use crate::compile_to_einsum_with_context;
use crate::context::CompilerContext;

use super::collector::DiagnosticCollector;
use super::diagnostic::{Diagnostic, Severity};
use super::strategy::{RecoveryAction, RecoveryStrategy};

/// Result returned by [`TolerantCompiler::compile_program`].
///
/// The `graphs` vector has exactly the same length as the input slice.
/// `graphs[i] == None` iff expression `i` was skipped due to a blocking
/// diagnostic (Error or Fatal) or because the program was aborted while
/// processing expression `i` or earlier.
#[derive(Debug, Clone)]
pub struct PartialCompilationResult {
    /// Per-expression compilation output, aligned with the input slice.
    pub graphs: Vec<Option<EinsumGraph>>,
    /// Collected diagnostics, in insertion (i.e. expression) order.
    pub diagnostics: DiagnosticCollector,
    /// The recovery strategy that produced this result.
    pub strategy: RecoveryStrategy,
    /// `true` iff the driver stopped early (AbortOnAny / SkipOnFatal hit a
    /// blocker). Expressions after `aborted_at` have `None` graphs.
    pub aborted: bool,
    /// Index at which the driver aborted (only meaningful when `aborted` is
    /// `true`).
    pub aborted_at: Option<usize>,
}

impl PartialCompilationResult {
    /// Number of expressions that successfully produced a graph.
    pub fn success_count(&self) -> usize {
        self.graphs.iter().filter(|g| g.is_some()).count()
    }

    /// Number of expressions that failed to produce a graph.
    pub fn failure_count(&self) -> usize {
        self.graphs.iter().filter(|g| g.is_none()).count()
    }

    /// `true` if every expression produced a graph.
    pub fn is_all_success(&self) -> bool {
        self.graphs.iter().all(|g| g.is_some())
    }

    /// Iterate successful `(index, &graph)` pairs.
    pub fn successes(&self) -> impl Iterator<Item = (usize, &EinsumGraph)> {
        self.graphs
            .iter()
            .enumerate()
            .filter_map(|(i, g)| g.as_ref().map(|gg| (i, gg)))
    }

    /// Indices of expressions that produced no graph.
    pub fn failures(&self) -> Vec<usize> {
        self.graphs
            .iter()
            .enumerate()
            .filter_map(|(i, g)| if g.is_none() { Some(i) } else { None })
            .collect()
    }
}

/// Tolerant compilation façade.
///
/// The compiler is stateless besides its [`RecoveryStrategy`]; each call to
/// [`Self::compile_program`] starts from a fresh [`CompilerContext`] unless
/// the caller uses [`Self::compile_program_with_contexts`].
#[derive(Debug, Clone, Default)]
pub struct TolerantCompiler {
    strategy: RecoveryStrategy,
}

impl TolerantCompiler {
    /// Construct a tolerant compiler with [`RecoveryStrategy::SkipOnError`].
    pub fn new() -> Self {
        Self::default()
    }

    /// Construct a tolerant compiler with a specific recovery strategy.
    pub fn with_strategy(strategy: RecoveryStrategy) -> Self {
        Self { strategy }
    }

    /// Configured recovery strategy.
    pub fn strategy(&self) -> RecoveryStrategy {
        self.strategy
    }

    /// Update the recovery strategy in place.
    pub fn set_strategy(&mut self, strategy: RecoveryStrategy) {
        self.strategy = strategy;
    }

    /// Compile a program (slice of expressions) under the current recovery
    /// strategy using **one fresh [`CompilerContext`] per expression** so
    /// failures do not poison the context for siblings.
    pub fn compile_program(&self, program: &[TLExpr]) -> PartialCompilationResult {
        self.compile_program_with(program, |_idx| CompilerContext::new())
    }

    /// Compile a program, calling `make_ctx` once per expression to obtain a
    /// fresh context. This lets callers share domain declarations across
    /// expressions while still keeping each compilation isolated.
    pub fn compile_program_with<F>(
        &self,
        program: &[TLExpr],
        mut make_ctx: F,
    ) -> PartialCompilationResult
    where
        F: FnMut(usize) -> CompilerContext,
    {
        let collector = DiagnosticCollector::new();
        let mut graphs: Vec<Option<EinsumGraph>> = Vec::with_capacity(program.len());

        let mut aborted = false;
        let mut aborted_at: Option<usize> = None;

        for (idx, expr) in program.iter().enumerate() {
            if aborted {
                graphs.push(None);
                continue;
            }

            let mut ctx = make_ctx(idx);
            match self.compile_one(idx, expr, &mut ctx, &collector) {
                OneResult::Ok(graph) => graphs.push(Some(graph)),
                OneResult::Skipped => graphs.push(None),
                OneResult::Aborted => {
                    graphs.push(None);
                    aborted = true;
                    aborted_at = Some(idx);
                }
            }
        }

        PartialCompilationResult {
            graphs,
            diagnostics: collector,
            strategy: self.strategy,
            aborted,
            aborted_at,
        }
    }

    /// Compile a program re-using a caller-supplied vector of contexts. Every
    /// context is used exactly once, matched by index to the expression slot.
    ///
    /// The caller must provide `contexts.len() >= program.len()`; surplus
    /// contexts are ignored.
    pub fn compile_program_with_contexts(
        &self,
        program: &[TLExpr],
        contexts: &mut [CompilerContext],
    ) -> PartialCompilationResult {
        let collector = DiagnosticCollector::new();
        let mut graphs: Vec<Option<EinsumGraph>> = Vec::with_capacity(program.len());

        let mut aborted = false;
        let mut aborted_at: Option<usize> = None;

        for (idx, expr) in program.iter().enumerate() {
            if aborted {
                graphs.push(None);
                continue;
            }

            if idx >= contexts.len() {
                collector.push(
                    Diagnostic::fatal(format!(
                        "tolerant compiler: missing CompilerContext for expression #{}",
                        idx
                    ))
                    .with_expression_index(idx),
                );
                // Behave as if a fatal fired.
                let action = self.strategy.decide(Severity::Fatal);
                match action {
                    RecoveryAction::Continue => graphs.push(None),
                    RecoveryAction::SkipExpression => graphs.push(None),
                    RecoveryAction::AbortProgram => {
                        graphs.push(None);
                        aborted = true;
                        aborted_at = Some(idx);
                    }
                }
                continue;
            }

            match self.compile_one(idx, expr, &mut contexts[idx], &collector) {
                OneResult::Ok(graph) => graphs.push(Some(graph)),
                OneResult::Skipped => graphs.push(None),
                OneResult::Aborted => {
                    graphs.push(None);
                    aborted = true;
                    aborted_at = Some(idx);
                }
            }
        }

        PartialCompilationResult {
            graphs,
            diagnostics: collector,
            strategy: self.strategy,
            aborted,
            aborted_at,
        }
    }

    /// Compile a single expression in tolerant mode, pushing diagnostics
    /// into the supplied collector. Returns the per-expression outcome.
    fn compile_one(
        &self,
        idx: usize,
        expr: &TLExpr,
        ctx: &mut CompilerContext,
        collector: &DiagnosticCollector,
    ) -> OneResult {
        // catch_unwind only at the TOP LEVEL boundary, per the scope
        // discipline: panics become Fatal diagnostics instead of unwinding
        // across the driver.
        let unwind_result = panic::catch_unwind(AssertUnwindSafe(|| {
            compile_to_einsum_with_context(expr, ctx)
        }));

        match unwind_result {
            Ok(Ok(graph)) => OneResult::Ok(graph),
            Ok(Err(err)) => {
                let diag =
                    Diagnostic::error(format!("compilation error in expression #{}: {}", idx, err))
                        .with_expression_index(idx);
                collector.push(diag);
                self.react(idx, Severity::Error)
            }
            Err(payload) => {
                let msg = panic_payload_to_string(&payload);
                let diag = Diagnostic::fatal(format!(
                    "panic while compiling expression #{}: {}",
                    idx, msg
                ))
                .with_expression_index(idx);
                collector.push(diag);
                self.react(idx, Severity::Fatal)
            }
        }
    }

    /// Consult the strategy to decide whether to skip or abort after a
    /// diagnostic of the given severity has already been pushed.
    fn react(&self, _idx: usize, severity: Severity) -> OneResult {
        match self.strategy.decide(severity) {
            RecoveryAction::Continue => {
                // Non-blocking diagnostics never reach this arm (Error/Fatal
                // only). Guard defensively just in case.
                OneResult::Skipped
            }
            RecoveryAction::SkipExpression => OneResult::Skipped,
            RecoveryAction::AbortProgram => OneResult::Aborted,
        }
    }
}

/// Per-expression outcome of the tolerant driver.
#[derive(Debug)]
enum OneResult {
    Ok(EinsumGraph),
    Skipped,
    Aborted,
}

/// Public free function — tolerant counterpart of
/// [`crate::compile_to_einsum`].
///
/// Equivalent to
/// `TolerantCompiler::with_strategy(RecoveryStrategy::SkipOnError).compile_program(program)`.
pub fn compile_tolerant(program: &[TLExpr]) -> PartialCompilationResult {
    TolerantCompiler::new().compile_program(program)
}

/// Public free function — tolerant compilation with a caller-chosen strategy.
pub fn compile_tolerant_with_strategy(
    program: &[TLExpr],
    strategy: RecoveryStrategy,
) -> PartialCompilationResult {
    TolerantCompiler::with_strategy(strategy).compile_program(program)
}

/// Convert a `Box<dyn Any + Send>` panic payload into a human-readable string
/// without panicking again.
fn panic_payload_to_string(payload: &Box<dyn std::any::Any + Send>) -> String {
    if let Some(s) = payload.downcast_ref::<&'static str>() {
        (*s).to_string()
    } else if let Some(s) = payload.downcast_ref::<String>() {
        s.clone()
    } else {
        "<non-string panic payload>".to_string()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tensorlogic_ir::{TLExpr, Term};

    fn good_expr() -> TLExpr {
        TLExpr::pred("p", vec![Term::var("x")])
    }

    #[test]
    fn compile_tolerant_all_good() {
        let program = vec![good_expr(), good_expr(), good_expr()];
        let res = compile_tolerant(&program);
        assert_eq!(res.graphs.len(), 3);
        assert!(res.is_all_success());
        assert_eq!(res.success_count(), 3);
        assert!(!res.aborted);
        assert!(res.diagnostics.is_empty());
    }

    #[test]
    fn partial_result_success_iter() {
        let program = vec![good_expr(), good_expr()];
        let res = compile_tolerant(&program);
        let v: Vec<usize> = res.successes().map(|(i, _)| i).collect();
        assert_eq!(v, vec![0, 1]);
    }
}
