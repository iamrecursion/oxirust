//! Smoke tests for `tensorlogic::prelude`.
//!
//! These tests guarantee that every symbol re-exported from the prelude is
//! both reachable and usable through the flagship crate. Dropping or renaming
//! a prelude export will break compilation here, catching API regressions that
//! per-crate tests miss.

use tensorlogic::prelude::*;

/// Reference every prelude item by type so the compiler verifies each is
/// reachable via `tensorlogic::prelude::*`. No runtime behavior is asserted.
#[test]
fn prelude_types_import() {
    // Core construction + compilation + execution entry points.
    // Invoke each function in an expression so the imports are exercised.
    let _term: Term = Term::var("_");
    let _expr: TLExpr = TLExpr::pred("_", vec![]);
    let _compile_fn = compile_to_einsum;

    // Convenience re-exports.
    let _graph_slot: Option<EinsumGraph> = None;
    let _config_slot: Option<CompilationConfig> = None;
    let _executor_slot: Option<Scirs2Exec> = None;

    // Mandatory error types.
    let _ir_err_slot: Option<IrError> = None;
    let _infer_err_slot: Option<ExecutorError> = None;
    let _adapter_err_slot: Option<AdapterError> = None;
    let _compile_err_slot: Option<CompileError> = None;
    let _backend_err_slot: Option<TlBackendError> = None;

    // Optional error types — gated identically to their re-exports in lib.rs.
    #[cfg(feature = "oxirs")]
    let _oxirs_err_slot: Option<BridgeError> = None;
    #[cfg(feature = "quantrs")]
    let _quantrs_err_slot: Option<PgmError> = None;
    #[cfg(feature = "sklears")]
    let _sklears_err_slot: Option<KernelError> = None;
    #[cfg(feature = "train")]
    let _train_err_slot: Option<TrainError> = None;
    #[cfg(feature = "trustformers")]
    let _trustformers_err_slot: Option<TrustformerError> = None;

    // Reference the autodiff + executor traits so the imports are non-dead.
    fn _needs_executor<E: TlExecutor>() {}
    fn _needs_autodiff<A: TlAutodiff>() {}
    _needs_executor::<Scirs2Exec>();
    _needs_autodiff::<Scirs2Exec>();
}

/// Replicates the top-level doctest from `lib.rs` as a real `#[test]`.
/// Builds `knows(x,y) and knows(y,z)`, compiles to an einsum graph, binds
/// concrete input tensors, and runs a forward pass through the SciRS2
/// executor. The doctest in `lib.rs` is marked `no_run`, so this is the
/// first test that exercises the full build -> compile -> execute pipeline
/// through the flagship crate.
#[test]
fn end_to_end_forward() -> Result<(), Box<dyn std::error::Error>> {
    let x = Term::var("x");
    let y = Term::var("y");
    let z = Term::var("z");

    let knows_xy = TLExpr::pred("knows", vec![x, y.clone()]);
    let knows_yz = TLExpr::pred("knows", vec![y, z]);
    let premise = TLExpr::and(knows_xy, knows_yz);

    let graph = compile_to_einsum(&premise)?;
    assert!(
        !graph.tensors.is_empty(),
        "compiler should produce tensor slots"
    );
    assert!(
        !graph.nodes.is_empty(),
        "compiler should emit at least one node"
    );
    assert_eq!(
        graph.outputs.len(),
        1,
        "premise yields a single output tensor"
    );

    // Bind a uniform 2x2 tensor to every input slot. The body semantics are
    // unimportant here; the test asserts that the executor accepts the
    // flagship-crate graph and completes a forward pass without error.
    let mut executor = Scirs2Exec::new();
    let input_data = vec![0.25_f64, 0.5, 0.5, 0.75];
    let input_shape = vec![2_usize, 2];
    let input_count = graph.tensors.len().saturating_sub(graph.outputs.len());
    for name in graph.tensors.iter().take(input_count) {
        let tensor = Scirs2Exec::from_vec(input_data.clone(), input_shape.clone())?;
        executor.add_tensor(name.clone(), tensor);
    }

    let _result = executor.forward(&graph)?;
    Ok(())
}

/// Verifies constructor round-trip for `Term::var` + `TLExpr::pred` without
/// invoking the compiler. Guards against accidental signature drift.
#[test]
fn build_simple_predicate() {
    let a = Term::var("a");
    let b = Term::var("b");
    let expr = TLExpr::pred("parent", vec![a, b]);

    // The expression must be non-empty; format-debug as a sanity probe.
    let rendered = format!("{expr:?}");
    assert!(
        !rendered.is_empty(),
        "expression debug output should not be empty"
    );
}
